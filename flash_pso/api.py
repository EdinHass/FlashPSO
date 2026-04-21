"""FlashPSO: GPU-accelerated American option pricer via PSO + Monte Carlo.

Supports standard and Asian exercise styles on single-asset options.
"""
import math
import numpy as np
from typing import Optional

import torch
import triton
import triton.language as tl

from flash_pso.config import OptionConfig, ComputeConfig, SwarmConfig
from flash_pso.enums import OptionStyle, RNGType
from flash_pso.asserts import validate_inputs

_cc_major, _ = torch.cuda.get_device_capability()
print(f"Detected GPU with compute capability {_cc_major}.")
if _cc_major >= 9:
    print("Using SM90 kernels")
    from flash_pso.sm90.pso_kernels import init_kernel, pso_update_kernel
    from flash_pso.sm90.payoff_kernels import mc_payoff_kernel, mc_asian_payoff_kernel
    from flash_pso.sm90.mc_kernels import mc_path_kernel
    from flash_pso.sm90.pso_utils import reduce_pbest_local, reduce_pbest_global
else:
    print("Using SM80 kernels")
    from flash_pso.sm80.pso_kernels import init_kernel, pso_update_kernel
    from flash_pso.sm80.payoff_kernels import mc_payoff_kernel, mc_asian_payoff_kernel
    from flash_pso.sm80.mc_kernels import mc_path_kernel
    from flash_pso.sm80.pso_utils import reduce_pbest_local, reduce_pbest_global


def _triton_alloc(size: int, align: int, stream: int) -> int:
    return torch.cuda.caching_allocator_alloc(size, device=torch.cuda.current_device())

def _triton_free(ptr: int):
    torch.cuda.caching_allocator_delete(ptr)

triton.set_allocator(_triton_alloc)
_ = torch.tensor([0.0], device="cuda:0")


class FlashPSO:
    """American option pricer using Particle Swarm Optimization on GPU.

    The PSO searches for an optimal exercise boundary (one threshold per timestep).
    Each candidate boundary is evaluated by Monte Carlo simulation over all paths.

    Args:
        option_config:  Option parameters (S0, K, r, sigma, T, paths, steps).
        compute_config: Kernel tuning, RNG, and convergence settings.
        swarm_config:   PSO hyperparameters (particles, inertia, weights).
    """

    @property
    def num_particles(self) -> int:
        return self.swarm.num_particles

    @property
    def num_dimensions(self) -> int:
        return self.opt.num_time_steps

    def __init__(self, option_config: OptionConfig, compute_config: ComputeConfig,
                 swarm_config: SwarmConfig, precomputed_St: Optional[torch.Tensor] = None,
                 initial_velocities: Optional[torch.Tensor] = None,
                 initial_positions: Optional[torch.Tensor] = None):
        self.opt = option_config
        self.comp = compute_config
        self.swarm = swarm_config
        validate_inputs(self, initial_positions, initial_velocities, precomputed_St)

        self.log2_S0 = self.opt.log2_S0
        self.drift_l2 = self.opt.drift_l2
        self.vol_l2 = self.opt.vol_l2
        self.r_dt_l2 = self.opt.r_dt_l2
        self.terminal_discount = self.opt.terminal_discount

        mc_span = int(self.opt.num_paths) * int(self.num_dimensions)
        init_span = int(self.num_particles) * int(self.num_dimensions)
        self.offset = 0
        self.mc_offset_philox = self._philox_reserve(mc_span)
        self.init_pos_philox = self._philox_reserve(init_span)
        self.init_vel_philox = self._philox_reserve(init_span)
        self.pso_offset_philox = self._philox_reserve(init_span * 2)
        if self.comp.use_fixed_random:
            self.r1_offset_philox = self._philox_reserve(init_span)
            self.r2_offset_philox = self._philox_reserve(init_span)

        total_path_blocks = self.opt.num_paths // self.comp.pso_paths_block_size
        self._num_compute_path_blocks = round(self.comp.compute_fraction * total_path_blocks)
        _num_bw_path_blocks = total_path_blocks - self._num_compute_path_blocks
        self._num_bw_paths = _num_bw_path_blocks * self.comp.pso_paths_block_size
        self._num_compute_paths = self._num_compute_path_blocks * self.comp.pso_paths_block_size

        self.Z = torch.empty((0,), device="cuda")
        if self.comp.rng_type == RNGType.SOBOL:
            from flash_pso.rng.sobol import generate_sobol_normals_1d
            self.Z = generate_sobol_normals_1d(self.num_dimensions, self._num_bw_paths, skip=0)

        if self._num_bw_paths > 0:
            if precomputed_St is not None:
                self.St = precomputed_St.to("cuda")
            else:
                self.St = torch.empty((self.opt.num_time_steps, self._num_bw_paths), device="cuda", dtype=torch.float32).t()
                self._precompute_mc_paths()
        else:
            self.St = torch.empty((0,), device="cuda")

        D, P = self.num_dimensions, self.num_particles
        self.positions = torch.empty((D, P), device="cuda", dtype=torch.float32).t()
        self.ln_positions = torch.empty((D, P), device="cuda", dtype=torch.float32).t()
        self.velocities = torch.empty((D, P), device="cuda", dtype=torch.float32).t()
        self.pbest_pos = torch.empty((D, P), device="cuda", dtype=torch.float32).t()
        if self.comp.use_fixed_random:
            self.r1 = torch.empty((D, P), device="cuda", dtype=torch.float32).t()
            self.r2 = torch.empty((D, P), device="cuda", dtype=torch.float32).t()
        else:
            self.r1 = torch.empty((0,), device="cuda")
            self.r2 = torch.empty((0,), device="cuda")

        if initial_positions is not None: self.positions.copy_(initial_positions.to("cuda"))
        if initial_velocities is not None: self.velocities.copy_(initial_velocities.to("cuda"))

        self.pbest_payoff = torch.empty((P,), device="cuda", dtype=torch.float32)
        self.gbest_payoff = torch.full((1,), float("-inf"), device="cuda", dtype=torch.float32)
        self.gbest_pos = torch.empty((D,), device="cuda", dtype=torch.float32)

        self.pos_centers = torch.full((D,), self.opt.strike_price, device="cuda", dtype=torch.float32)
        self.search_windows = self._build_search_windows()

        self._initialize(initial_positions is None, initial_velocities is None)
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
        """Run PSO optimization loop. Call get_option_price() after."""
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
        """In-sample price (slight upward bias from same-sample optimization)."""
        return float(self.global_payoffs_cpu[self.global_payoff_index - 1])

    def get_debiased_price(self) -> float:
        fresh_mc_offset = self.offset
        self.offset += int(self.opt.num_paths) * int(self.num_dimensions)
        saved = self.mc_offset_philox
        self.mc_offset_philox = fresh_mc_offset

        if self.comp.rng_type == RNGType.SOBOL:
            from flash_pso.rng.sobol import generate_sobol_normals_1d
            self.Z = generate_sobol_normals_1d(self.num_dimensions, self._num_bw_paths, skip=self._num_bw_paths)

        if self._num_bw_paths > 0:
            self._precompute_mc_paths()

        gbest_expanded = self.gbest_pos.unsqueeze(0).expand(self.num_particles, -1)
        self.positions.copy_(gbest_expanded)
        self.ln_positions.copy_(torch.log2(torch.clamp(self.positions, min=1e-4)))
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

    def _build_search_windows(self) -> torch.Tensor:
        r_val = self.opt.risk_free_rate
        sig = self.opt.volatility
        floor_frac = (2 * r_val) / (2 * r_val + sig ** 2)
        width = min(1.0 - floor_frac, 0.99)
        return torch.tensor([width], device="cuda", dtype=torch.float32)

    def _initialize(self, gen_pos=True, gen_vel=True):
        n_total = self.num_particles * self.num_dimensions
        grid = (triton.cdiv(n_total, self.comp.elementwise_block_size),)
        r1_off = self.r1_offset_philox if self.comp.use_fixed_random else 0
        r2_off = self.r2_offset_philox if self.comp.use_fixed_random else 0
        init_kernel[grid](
            positions_ptr=self.positions, ln_positions_ptr=self.ln_positions,
            velocities_ptr=self.velocities, pbest_costs_ptr=self.pbest_payoff,
            pbest_pos_ptr=self.pbest_pos, r1_ptr=self.r1, r2_ptr=self.r2,
            pos_centers_ptr=self.pos_centers, search_windows_ptr=self.search_windows,
            seed=self.comp.seed,
            NUM_DIMENSIONS=tl.constexpr(self.num_dimensions),
            NUM_PARTICLES=tl.constexpr(self.num_particles),
            OPTION_TYPE=tl.constexpr(int(self.opt.option_type)),
            EXERCISE_STYLE=tl.constexpr(0), NUM_ASSETS=tl.constexpr(1),
            GENERATES_POSITIONS=tl.constexpr(gen_pos),
            GENERATES_VELOCITIES=tl.constexpr(gen_vel),
            BLOCK_SIZE=tl.constexpr(self.comp.elementwise_block_size),
            POS_OFFSET_PHILOX=self.init_pos_philox,
            VEL_OFFSET_PHILOX=self.init_vel_philox,
            R1_OFFSET_PHILOX=r1_off, R2_OFFSET_PHILOX=r2_off,
            USE_FIXED_RANDOM=tl.constexpr(self.comp.use_fixed_random),
        )

    def _precompute_mc_paths(self):
        grid = (triton.cdiv(self._num_bw_paths, self.comp.pso_paths_block_size),)
        mc_path_kernel[grid](
            St_ptr=self.St, Z_ptr=self.Z,
            log2_S0=self.log2_S0, drift_l2=self.drift_l2, vol_l2=self.vol_l2,
            num_paths=self._num_bw_paths, seed=self.comp.seed,
            mc_offset_philox=self.mc_offset_philox,
            bandwidth_paths_start=self._num_compute_paths,
            BLOCK_SIZE=tl.constexpr(self.comp.pso_paths_block_size),
            NUM_TIME_STEPS=tl.constexpr(self.opt.num_time_steps),
            USE_ANTITHETIC=tl.constexpr(self.comp.use_antithetic),
            USE_PRECOMPUTED_Z=tl.constexpr(self.comp.rng_type == RNGType.SOBOL),
        )

    def _PSO_update(self, iteration: int, eval_only: bool = False):
        if not eval_only:
            grid_pso = lambda meta: (triton.cdiv(self.num_dimensions * self.num_particles, self.comp.elementwise_block_size),)
            pso_update_kernel[grid_pso](
                positions_ptr=self.positions, ln_positions_ptr=self.ln_positions,
                velocities_ptr=self.velocities, pbest_pos_ptr=self.pbest_pos,
                gbest_pos_ptr=self.gbest_pos, r1_ptr=self.r1, r2_ptr=self.r2,
                iteration=iteration,
                INERTIA_WEIGHT=self.swarm.inertia_weight,
                COGNITIVE_WEIGHT=self.swarm.cognitive_weight,
                SOCIAL_WEIGHT=self.swarm.social_weight,
                NUM_DIMENSIONS=tl.constexpr(self.num_dimensions),
                NUM_PARTICLES=tl.constexpr(self.num_particles),
                PSO_OFFSET_PHILOX=self.pso_offset_philox,
                SEED=self.comp.seed, USE_FIXED_RANDOM=self.comp.use_fixed_random,
                BLOCK_SIZE=self.comp.elementwise_block_size,
            )

        grid_mc = lambda meta: (
            triton.cdiv(self.num_particles, meta["BLOCK_SIZE_PARTICLES"]),
            self.opt.num_paths // self.comp.pso_paths_block_size,
        )
        kernel = mc_asian_payoff_kernel if self.opt.option_style == OptionStyle.ASIAN else mc_payoff_kernel
        kernel[grid_mc](
            ln_positions_ptr=self.ln_positions, st_ptr=self.St,
            partial_payoffs_ptr=self._partial_payoffs,
            seed=self.comp.seed, mc_offset_philox=self.mc_offset_philox,
            log2_S0=self.log2_S0, drift_l2=self.drift_l2, vol_l2=self.vol_l2, r_dt_l2=self.r_dt_l2, terminal_discount=self.terminal_discount,
            NUM_DIMENSIONS=self.num_dimensions, NUM_PARTICLES=self.num_particles,
            NUM_PATHS=self.opt.num_paths,
            NUM_COMPUTE_PATH_BLOCKS=self._num_compute_path_blocks,
            BLOCK_SIZE_PATHS=self.comp.pso_paths_block_size,
            OPTION_TYPE=int(self.opt.option_type), STRIKE_PRICE=self.opt.strike_price,
            USE_ANTITHETIC=self.comp.use_antithetic,
        )

    def _reduce_pbest(self):
        actual_num_path_blocks = self.opt.num_paths // self.comp.pso_paths_block_size
        reduce_pbest_local[(self._num_reduction_blocks,)](
            pbest_payoffs_ptr=self.pbest_payoff, pbest_positions_ptr=self.pbest_pos,
            scratch_payoffs_ptr=self._scratch_payoffs, scratch_particle_idx_ptr=self._scratch_particle_idx,
            partial_payoffs_ptr=self._partial_payoffs, positions_ptr=self.positions,
            NUM_PARTICLES=tl.constexpr(self.num_particles), NUM_BLOCKS=tl.constexpr(self._num_reduction_blocks),
            NUM_DIMENSIONS=tl.constexpr(self.num_dimensions),
            BLOCK_SIZE=tl.constexpr(self.comp.reduction_block_size),
            NUM_PATH_BLOCKS=tl.constexpr(actual_num_path_blocks),
            NUM_PATHS=tl.constexpr(self.opt.num_paths),
        )
        reduce_pbest_global[(1,)](
            scratch_payoffs_ptr=self._scratch_payoffs, scratch_particle_idx_ptr=self._scratch_particle_idx,
            pbest_positions_ptr=self.pbest_pos,
            gbest_payoff_ptr=self.gbest_payoff, gbest_position_ptr=self.gbest_pos,
            NUM_PARTICLES=tl.constexpr(self.num_particles),
            NUM_BLOCKS=tl.constexpr(self._num_reduction_blocks), NUM_DIMENSIONS=tl.constexpr(self.num_dimensions),
            BLOCK_SIZE=triton.next_power_of_2(self._num_reduction_blocks),
        )

    def _is_converged(self):
        if self.comp.convergence_threshold < 0 or self.global_payoff_index < 2:
            return False
        return abs(self.global_payoffs_cpu[self.global_payoff_index - 1]
                   - self.global_payoffs_cpu[self.global_payoff_index - 2]) < self.comp.convergence_threshold
