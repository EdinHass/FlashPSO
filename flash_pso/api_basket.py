"""FlashPSOBasket: GPU-accelerated American basket option pricer.

Supports two exercise parameterizations:
  SCALAR:    One boundary per timestep on weighted basket price.
  PER_ASSET: One boundary per asset per timestep (exercise when all cross).
"""
import math
import numpy as np
import torch
import triton
import triton.language as tl

from flash_pso.config import BasketOptionConfig, ComputeConfig, SwarmConfig
from flash_pso.enums import ExerciseStyle, RNGType
from flash_pso.asserts import validate_basket_inputs

_cc_major, _ = torch.cuda.get_device_capability()
if _cc_major >= 9:
    from flash_pso.sm90.pso_kernels import init_kernel, pso_update_kernel
    from flash_pso.sm90.pso_utils import reduce_pbest_local, reduce_pbest_global
    from flash_pso.sm90.payoff_kernels import mc_basket_payoff_kernel
    from flash_pso.sm90.mc_kernels import mc_basket_collapse_kernel, mc_basket_path_kernel
else:
    from flash_pso.sm80.pso_kernels import init_kernel, pso_update_kernel
    from flash_pso.sm80.pso_utils import reduce_pbest_local, reduce_pbest_global
    from flash_pso.sm80.payoff_kernels import mc_basket_payoff_kernel
    from flash_pso.sm80.mc_kernels import mc_basket_collapse_kernel, mc_basket_path_kernel


class FlashPSOBasket:
    """American basket option pricer using PSO + Monte Carlo on GPU.

    Args:
        option_config:  Basket parameters (S0s, K, r, vols, weights, corr, T).
        compute_config: Kernel tuning, RNG, and convergence settings.
        swarm_config:   PSO hyperparameters.
    """

    @property
    def pso_num_dimensions(self) -> int:
        if self.opt.exercise_style == ExerciseStyle.PER_ASSET:
            return self.opt.num_assets * self.opt.num_time_steps
        return self.opt.num_time_steps

    @property
    def mc_num_dimensions(self) -> int:
        return self.opt.num_time_steps

    @property
    def num_particles(self) -> int:
        return self.swarm.num_particles

    def __init__(self, option_config: BasketOptionConfig, compute_config: ComputeConfig,
                 swarm_config: SwarmConfig):
        self.opt = option_config
        self.comp = compute_config
        self.swarm = swarm_config
        validate_basket_inputs(self)

        N = self.opt.num_assets

        self.S0_vec = torch.tensor(self.opt.initial_stock_prices, device="cuda", dtype=torch.float32)
        self.log2_S0_vec = torch.log2(self.S0_vec)
        self.vol_vec = torch.tensor(self.opt.volatilities, device="cuda", dtype=torch.float32)
        self.weights_vec = torch.tensor(self.opt.weights, device="cuda", dtype=torch.float32)

        LOG2E = math.log2(math.e)
        self.drift_l2_vec = ((self.opt.risk_free_rate - 0.5 * self.vol_vec ** 2) * self.opt.time_step_size * LOG2E).to("cuda")
        self.vol_l2_vec = (self.vol_vec * math.sqrt(self.opt.time_step_size) * LOG2E).to("cuda")
        self.r_dt_l2 = -self.opt.risk_free_rate * self.opt.time_step_size * LOG2E
        self.terminal_discount = 2.0 ** (self.r_dt_l2 * self.mc_num_dimensions)

        corr = torch.tensor(self.opt.correlation_matrix, device="cuda", dtype=torch.float32)
        self.L_mat = torch.linalg.cholesky(corr).contiguous()

        total_path_blocks = self.opt.num_paths // self.comp.pso_paths_block_size
        self._num_compute_path_blocks = round(self.comp.compute_fraction * total_path_blocks)
        _num_bw_blocks = total_path_blocks - self._num_compute_path_blocks
        self._num_bw_paths = _num_bw_blocks * self.comp.pso_paths_block_size
        self._num_compute_paths = self._num_compute_path_blocks * self.comp.pso_paths_block_size

        pso_dim = self.pso_num_dimensions
        init_span = int(self.num_particles) * int(pso_dim)
        self.offset = 0
        self.mc_offset_philox = self._philox_reserve(int(self.opt.num_paths) * int(self.mc_num_dimensions) * N)
        self.init_pos_philox = self._philox_reserve(init_span)
        self.init_vel_philox = self._philox_reserve(init_span)
        self.pso_offset_philox = self._philox_reserve(init_span * 2)
        if self.comp.use_fixed_random:
            self.r1_offset_philox = self._philox_reserve(init_span)
            self.r2_offset_philox = self._philox_reserve(init_span)

        self.Z = torch.empty((0,), device="cuda")
        if self.comp.rng_type == RNGType.SOBOL:
            from flash_pso.rng.sobol import generate_sobol_normals
            self.Z = generate_sobol_normals(N, self.mc_num_dimensions, self.opt.num_paths, skip=0)

        if self._num_bw_paths > 0:
            if self.opt.exercise_style == ExerciseStyle.SCALAR:
                self.St = torch.empty((self.mc_num_dimensions, self._num_bw_paths), device="cuda", dtype=torch.float32).t()
                self._precompute_collapsed()
            else:
                self.St = torch.empty((self.mc_num_dimensions, N, self._num_bw_paths), device="cuda", dtype=torch.float32)
                self._precompute_per_asset()
        else:
            self.St = torch.empty((0,), device="cuda")

        P = self.num_particles
        self.positions = torch.empty((pso_dim, P), device="cuda", dtype=torch.float32).t()
        self.ln_positions = torch.empty((pso_dim, P), device="cuda", dtype=torch.float32).t()
        self.velocities = torch.empty((pso_dim, P), device="cuda", dtype=torch.float32).t()
        self.pbest_pos = torch.empty((pso_dim, P), device="cuda", dtype=torch.float32).t()
        if self.comp.use_fixed_random:
            self.r1 = torch.empty((pso_dim, P), device="cuda", dtype=torch.float32).t()
            self.r2 = torch.empty((pso_dim, P), device="cuda", dtype=torch.float32).t()
        else:
            self.r1 = torch.empty((0,), device="cuda")
            self.r2 = torch.empty((0,), device="cuda")

        self.pbest_payoff = torch.empty((P,), device="cuda", dtype=torch.float32)
        self.gbest_payoff = torch.full((1,), float("-inf"), device="cuda", dtype=torch.float32)
        self.gbest_pos = torch.empty((pso_dim,), device="cuda", dtype=torch.float32)

        self.pos_centers = self._build_pos_centers()
        self.search_windows = self._build_search_windows()

        self._initialize(generate_positions=True, generate_velocities=True)
        self.gbest_pos.copy_(self.positions[0])

        self._partial_payoffs = torch.empty(
            (self.opt.num_paths // self.comp.pso_paths_block_size, P), device="cuda", dtype=torch.float32)
        self._num_reduction_blocks = int(triton.cdiv(self.num_particles, self.comp.reduction_block_size))
        self._scratch_payoffs = torch.empty((self._num_reduction_blocks,), device="cuda", dtype=torch.float32)
        self._scratch_particle_idx = torch.empty((self._num_reduction_blocks,), device="cuda", dtype=torch.int32)
        buffer_size = self.comp.max_iterations // self.comp.sync_iters
        self.global_payoffs_cpu = np.empty((buffer_size,), dtype=np.float32)
        self.global_payoff_index = 0

    def optimize(self):
        """Run PSO optimization. Call get_option_price() or get_debiased_price() after."""
        self.global_payoff_index = 0
        self._PSO_update(iteration=0, eval_only=True)
        self._reduce_pbest()
        for iteration in range(0, self.comp.max_iterations, self.comp.sync_iters):
            for i in range(self.comp.sync_iters):
                self._PSO_update(iteration=iteration + i, eval_only=False)
                self._reduce_pbest()
            self.global_payoffs_cpu[self.global_payoff_index] = self.gbest_payoff.item()
            self.global_payoff_index += 1
            if self._is_converged():
                break

    def get_option_price(self) -> float:
        """In-sample price (upper bound, slight upward bias)."""
        return float(self.global_payoffs_cpu[self.global_payoff_index - 1])

    def get_debiased_price(self) -> float:
        """Out-of-sample price on fresh independent paths."""
        N = self.opt.num_assets
        fresh_mc_offset = self.offset
        self.offset += int(self.opt.num_paths) * int(self.mc_num_dimensions) * N
        saved = self.mc_offset_philox
        self.mc_offset_philox = fresh_mc_offset

        gbest_col = self.gbest_pos.unsqueeze(1).expand(-1, self.num_particles)
        self.positions.copy_(gbest_col.t())
        self.ln_positions.copy_(torch.log2(torch.clamp(gbest_col, min=1e-4)).t())

        if self.comp.rng_type == RNGType.SOBOL:
            from flash_pso.rng.sobol import generate_sobol_normals
            self.Z = generate_sobol_normals(N, self.mc_num_dimensions, self.opt.num_paths, skip=self.opt.num_paths)

        if self._num_bw_paths > 0:
            if self.opt.exercise_style == ExerciseStyle.SCALAR:
                self._precompute_collapsed()
            else:
                self._precompute_per_asset()

        self.pbest_payoff.fill_(float("-inf"))
        self.gbest_payoff.fill_(float("-inf"))

        self._PSO_update(iteration=0, eval_only=True)
        self._reduce_pbest()
        price = self.gbest_payoff.item()
        self.mc_offset_philox = saved
        return price

    def get_gbest_position(self) -> np.ndarray:
        return self.gbest_pos.cpu().numpy()

    def _philox_reserve(self, span: int) -> int:
        allocated = self.offset
        self.offset += max(1, span)
        return allocated

    def _build_pos_centers(self) -> torch.Tensor:
        """[pso_dim] init centers: strike for scalar, per-asset S0 for per-asset."""
        if self.opt.exercise_style == ExerciseStyle.SCALAR:
            return torch.full((self.pso_num_dimensions,), self.opt.strike_price,
                              device="cuda", dtype=torch.float32)
        per_step = torch.tensor(self.opt.initial_stock_prices, device="cuda", dtype=torch.float32)
        return per_step.repeat(self.mc_num_dimensions)

    def _build_search_windows(self) -> torch.Tensor:
        """[num_assets] search widths from perpetual American boundary."""
        r = self.opt.risk_free_rate
        if self.opt.exercise_style == ExerciseStyle.SCALAR:
            w = torch.tensor(self.opt.weights, device="cuda", dtype=torch.float32)
            vols = torch.tensor(self.opt.volatilities, device="cuda", dtype=torch.float32)
            corr = torch.tensor(self.opt.correlation_matrix, device="cuda", dtype=torch.float32)
            sig_eff_sq = (w @ (corr * torch.outer(vols, vols)) @ w).item()
            floor = (2 * r) / (2 * r + sig_eff_sq)
            width = min(1.0 - floor, 0.99)
            return torch.full((self.opt.num_assets,), width, device="cuda", dtype=torch.float32)
        else:
            vols_sq = self.vol_vec ** 2
            floors = (2 * r) / (2 * r + vols_sq)
            return torch.clamp(1.0 - floors, max=0.99)

    def _precompute_collapsed(self):
        grid = (triton.cdiv(self._num_bw_paths, self.comp.pso_paths_block_size),)
        mc_basket_collapse_kernel[grid](
            St_ptr=self.St, Z_ptr=self.Z, bandwidth_paths_start=self._num_compute_paths,
            log2_S0_ptr=self.log2_S0_vec, drift_ptr=self.drift_l2_vec,
            vol_ptr=self.vol_l2_vec, weights_ptr=self.weights_vec, L_ptr=self.L_mat,
            num_bw_paths=self._num_bw_paths, seed=self.comp.seed,
            mc_offset_philox=self.mc_offset_philox,
            BLOCK_SIZE=tl.constexpr(self.comp.pso_paths_block_size),
            NUM_TIME_STEPS=tl.constexpr(self.mc_num_dimensions),
            NUM_ASSETS=tl.constexpr(self.opt.num_assets),
            TOTAL_NUM_PATHS=tl.constexpr(self.opt.num_paths),
            USE_ANTITHETIC=tl.constexpr(self.comp.use_antithetic),
            USE_FP16=tl.constexpr(self.comp.use_fp16_cholesky),
            USE_PRECOMPUTED_Z=tl.constexpr(self.comp.rng_type == RNGType.SOBOL),
        )

    def _precompute_per_asset(self):
        grid = (triton.cdiv(self._num_bw_paths, self.comp.pso_paths_block_size),)
        mc_basket_path_kernel[grid](
            st_ptr=self.St, Z_ptr=self.Z,
            log2_S0_ptr=self.log2_S0_vec, drift_ptr=self.drift_l2_vec,
            vol_ptr=self.vol_l2_vec, L_ptr=self.L_mat,
            num_bw_paths=self._num_bw_paths, seed=self.comp.seed,
            bandwidth_paths_start=self._num_compute_paths,
            mc_offset_philox=self.mc_offset_philox,
            BLOCK_SIZE=tl.constexpr(self.comp.pso_paths_block_size),
            NUM_TIME_STEPS=tl.constexpr(self.mc_num_dimensions),
            NUM_ASSETS=tl.constexpr(self.opt.num_assets),
            TOTAL_NUM_PATHS=tl.constexpr(self.opt.num_paths),
            USE_ANTITHETIC=tl.constexpr(self.comp.use_antithetic),
            USE_FP16=tl.constexpr(self.comp.use_fp16_cholesky),
            USE_PRECOMPUTED_Z=tl.constexpr(self.comp.rng_type == RNGType.SOBOL),
        )

    def _PSO_update(self, iteration: int, eval_only: bool = False):
        pso_dim = self.pso_num_dimensions
        if not eval_only:
            grid_pso = lambda meta: (triton.cdiv(pso_dim * self.num_particles, self.comp.elementwise_block_size),)
            pso_update_kernel[grid_pso](
                positions_ptr=self.positions, ln_positions_ptr=self.ln_positions,
                velocities_ptr=self.velocities, pbest_pos_ptr=self.pbest_pos,
                gbest_pos_ptr=self.gbest_pos, r1_ptr=self.r1, r2_ptr=self.r2,
                iteration=iteration,
                INERTIA_WEIGHT=self.swarm.inertia_weight,
                COGNITIVE_WEIGHT=self.swarm.cognitive_weight,
                SOCIAL_WEIGHT=self.swarm.social_weight,
                NUM_DIMENSIONS=tl.constexpr(pso_dim),
                NUM_PARTICLES=tl.constexpr(self.num_particles),
                PSO_OFFSET_PHILOX=self.pso_offset_philox,
                SEED=self.comp.seed, USE_FIXED_RANDOM=self.comp.use_fixed_random,
                BLOCK_SIZE=self.comp.elementwise_block_size,
            )

        grid_mc = lambda meta: (
            triton.cdiv(self.num_particles, meta["BLOCK_SIZE_PARTICLES"]),
            self.opt.num_paths // self.comp.pso_paths_block_size,
        )
        mc_basket_payoff_kernel[grid_mc](
            ln_positions_ptr=self.ln_positions, st_ptr=self.St,
            partial_payoffs_ptr=self._partial_payoffs,
            log2_S0_ptr=self.log2_S0_vec, drift_ptr=self.drift_l2_vec,
            vol_ptr=self.vol_l2_vec, weights_ptr=self.weights_vec, L_ptr=self.L_mat,
            r_dt_l2=self.r_dt_l2, terminal_discount=self.terminal_discount,
            seed=self.comp.seed, mc_offset_philox=self.mc_offset_philox,
            NUM_DIMENSIONS=self.mc_num_dimensions, NUM_PARTICLES=self.num_particles,
            NUM_PATHS=self.opt.num_paths,
            NUM_COMPUTE_PATH_BLOCKS=self._num_compute_path_blocks,
            BLOCK_SIZE_PATHS=self.comp.pso_paths_block_size,
            NUM_ASSETS=tl.constexpr(self.opt.num_assets),
            OPTION_TYPE=int(self.opt.option_type), STRIKE_PRICE=self.opt.strike_price,
            USE_ANTITHETIC=self.comp.use_antithetic,
            EXERCISE_STYLE=int(self.opt.exercise_style),
            USE_FP16=self.comp.use_fp16_cholesky,
        )

    def _reduce_pbest(self):
        pso_dim = self.pso_num_dimensions
        actual_num_path_blocks = self.opt.num_paths // self.comp.pso_paths_block_size
        reduce_pbest_local[(self._num_reduction_blocks,)](
            pbest_payoffs_ptr=self.pbest_payoff, pbest_positions_ptr=self.pbest_pos,
            scratch_payoffs_ptr=self._scratch_payoffs, scratch_particle_idx_ptr=self._scratch_particle_idx,
            partial_payoffs_ptr=self._partial_payoffs, positions_ptr=self.positions,
            NUM_PARTICLES=tl.constexpr(self.num_particles), NUM_BLOCKS=tl.constexpr(self._num_reduction_blocks),
            NUM_DIMENSIONS=tl.constexpr(pso_dim),
            BLOCK_SIZE=tl.constexpr(self.comp.reduction_block_size),
            NUM_PATH_BLOCKS=tl.constexpr(actual_num_path_blocks),
            NUM_PATHS=tl.constexpr(self.opt.num_paths),
        )
        reduce_pbest_global[(1,)](
            scratch_payoffs_ptr=self._scratch_payoffs, scratch_particle_idx_ptr=self._scratch_particle_idx,
            pbest_positions_ptr=self.pbest_pos,
            gbest_payoff_ptr=self.gbest_payoff, gbest_position_ptr=self.gbest_pos,
            NUM_PARTICLES=tl.constexpr(self.num_particles),
            NUM_BLOCKS=tl.constexpr(self._num_reduction_blocks), NUM_DIMENSIONS=tl.constexpr(pso_dim),
            BLOCK_SIZE=triton.next_power_of_2(self._num_reduction_blocks),
        )

    def _initialize(self, generate_positions=True, generate_velocities=True):
        pso_dim = self.pso_num_dimensions
        n_total = self.num_particles * pso_dim
        grid = (triton.cdiv(n_total, self.comp.elementwise_block_size),)
        r1_off = self.r1_offset_philox if self.comp.use_fixed_random else 0
        r2_off = self.r2_offset_philox if self.comp.use_fixed_random else 0
        init_kernel[grid](
            positions_ptr=self.positions, ln_positions_ptr=self.ln_positions,
            velocities_ptr=self.velocities, pbest_costs_ptr=self.pbest_payoff,
            pbest_pos_ptr=self.pbest_pos, r1_ptr=self.r1, r2_ptr=self.r2,
            pos_centers_ptr=self.pos_centers, search_windows_ptr=self.search_windows,
            seed=self.comp.seed,
            NUM_DIMENSIONS=tl.constexpr(pso_dim),
            NUM_PARTICLES=tl.constexpr(self.num_particles),
            OPTION_TYPE=tl.constexpr(int(self.opt.option_type)),
            EXERCISE_STYLE=tl.constexpr(int(self.opt.exercise_style)),
            NUM_ASSETS=tl.constexpr(self.opt.num_assets),
            GENERATES_POSITIONS=tl.constexpr(generate_positions),
            GENERATES_VELOCITIES=tl.constexpr(generate_velocities),
            BLOCK_SIZE=tl.constexpr(self.comp.elementwise_block_size),
            POS_OFFSET_PHILOX=self.init_pos_philox,
            VEL_OFFSET_PHILOX=self.init_vel_philox,
            R1_OFFSET_PHILOX=r1_off, R2_OFFSET_PHILOX=r2_off,
            USE_FIXED_RANDOM=tl.constexpr(self.comp.use_fixed_random),
        )

    def _is_converged(self):
        if self.comp.convergence_threshold < 0 or self.global_payoff_index < 2:
            return False
        return abs(self.global_payoffs_cpu[self.global_payoff_index - 1]
                   - self.global_payoffs_cpu[self.global_payoff_index - 2]) < self.comp.convergence_threshold
