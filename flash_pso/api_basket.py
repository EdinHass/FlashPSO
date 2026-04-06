import math
import numpy as np
import torch
import triton
import triton.language as tl

from flash_pso.config import BasketOptionConfig, ComputeConfig, SwarmConfig

_cc_major, _ = torch.cuda.get_device_capability()
if _cc_major >= 9:
    from flash_pso.sm90.pso_kernels import init_kernel, pso_update_kernel
    from flash_pso.sm90.pso_utils import reduce_pbest_local, reduce_pbest_global
    from flash_pso.sm90.basket_kernels import (
        mc_basket_payoff_kernel, mc_basket_collapse_kernel, mc_basket_path_kernel,
    )
else:
    from flash_pso.sm80.pso_kernels import init_kernel, pso_update_kernel
    from flash_pso.sm80.pso_utils import reduce_pbest_local, reduce_pbest_global
    from flash_pso.sm80.basket_kernels import (
        mc_basket_payoff_kernel, mc_basket_collapse_kernel, mc_basket_path_kernel,
    )


def _next_power_of_2(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


class FlashPSOBasket:
    """American basket option pricer using PSO + Monte Carlo.

    exercise_style=0 (scalar boundary):
      PSO dims = num_time_steps. Precompute -> collapsed 1D basket_lnS.

    exercise_style=1 (per-asset boundaries):
      PSO dims = num_assets x num_time_steps. Precompute -> per-asset lnS.
      Exercise when ALL individual assets cross their boundaries.
    """

    @property
    def pso_num_dimensions(self) -> int:
        if self.opt.exercise_style == 1:
            return self.opt.num_assets * self.opt.num_time_steps
        return self.opt.num_time_steps

    @property
    def mc_num_dimensions(self) -> int:
        return self.opt.num_time_steps

    @property
    def num_particles(self) -> int:
        return self.swarm.num_particles

    def __init__(
        self,
        option_config: BasketOptionConfig,
        compute_config: ComputeConfig,
        swarm_config: SwarmConfig,
    ):
        self.opt   = option_config
        self.comp  = compute_config
        self.swarm = swarm_config

        N = self.opt.num_assets
        self.padded_assets = _next_power_of_2(N)

        # ── Per-asset parameter vectors (padded to next power of 2) ──────────
        self.S0_vec = torch.ones(self.padded_assets, device="cuda", dtype=torch.float32)
        self.S0_vec[:N] = torch.tensor(self.opt.initial_stock_prices, device="cuda")

        self.vol_vec = torch.zeros(self.padded_assets, device="cuda", dtype=torch.float32)
        self.vol_vec[:N] = torch.tensor(self.opt.volatilities, device="cuda")

        self.weights_vec = torch.zeros(self.padded_assets, device="cuda", dtype=torch.float32)
        self.weights_vec[:N] = torch.tensor(self.opt.weights, device="cuda")

        LOG2E = 1.4426950408889634
        drift = (self.opt.risk_free_rate - 0.5 * self.vol_vec ** 2) * self.opt.time_step_size * LOG2E
        self.drift_vec = drift.to("cuda")
        self.vol_l2_vec = (self.vol_vec * math.sqrt(self.opt.time_step_size) * LOG2E).to("cuda")

        corr_tensor = torch.tensor(self.opt.correlation_matrix, device="cuda", dtype=torch.float32)
        L_matrix = torch.linalg.cholesky(corr_tensor)
        self.L_mat = torch.zeros(self.padded_assets, self.padded_assets, device="cuda", dtype=torch.float32)
        self.L_mat[:N, :N] = L_matrix

        # ── Path-block bookkeeping ───────────────────────────────────────────
        total_path_blocks             = self.opt.num_paths // self.comp.pso_paths_block_size
        self._num_compute_path_blocks = round(self.comp.compute_fraction * total_path_blocks)
        _num_bw_path_blocks           = total_path_blocks - self._num_compute_path_blocks
        self._num_bw_paths            = _num_bw_path_blocks * self.comp.pso_paths_block_size
        self._num_compute_paths       = self._num_compute_path_blocks * self.comp.pso_paths_block_size

        # ── Philox offset reservation (non-overlapping) ──────────────────────
        pso_dim   = self.pso_num_dimensions
        init_span = int(self.num_particles) * int(pso_dim)
        self.offset = 0
        self.mc_offset_philox  = self._philox_reserve(
            int(self.opt.num_paths) * int(self.mc_num_dimensions) * N
        )
        self.init_pos_philox   = self._philox_reserve(init_span)
        self.init_vel_philox   = self._philox_reserve(init_span)
        self.pso_offset_philox = self._philox_reserve(init_span * 2)

        if self.comp.use_fixed_random:
            self.r1_offset_philox = self._philox_reserve(init_span)
            self.r2_offset_philox = self._philox_reserve(init_span)

        # ── Precompute bandwidth paths ───────────────────────────────────────
        if self._num_bw_paths > 0:
            if self.opt.exercise_style == 0:
                self.St = torch.empty(
                    (self.mc_num_dimensions, self._num_bw_paths),
                    device="cuda", dtype=torch.float32,
                ).t()
                self._precompute_collapsed()
            else:
                self.St = torch.empty(
                    (self.mc_num_dimensions, self.padded_assets, self._num_bw_paths),
                    device="cuda", dtype=torch.float32,
                )
                self._precompute_per_asset()
        else:
            self.St = torch.empty((0,), device="cuda")

        # ── PSO state tensors ────────────────────────────────────────────────
        self.positions    = torch.empty((pso_dim, self.num_particles), device="cuda", dtype=torch.float32).t()
        self.ln_positions = torch.empty((pso_dim, self.num_particles), device="cuda", dtype=torch.float32).t()
        self.velocities   = torch.empty((pso_dim, self.num_particles), device="cuda", dtype=torch.float32).t()
        self.pbest_pos    = torch.empty((pso_dim, self.num_particles), device="cuda", dtype=torch.float32).t()

        if self.comp.use_fixed_random:
            self.r1 = torch.empty((pso_dim, self.num_particles), device="cuda", dtype=torch.float32).t()
            self.r2 = torch.empty((pso_dim, self.num_particles), device="cuda", dtype=torch.float32).t()
        else:
            self.r1 = torch.empty((0,), device="cuda")
            self.r2 = torch.empty((0,), device="cuda")

        self.pbest_payoff = torch.empty((self.num_particles,), device="cuda", dtype=torch.float32)
        self.gbest_payoff = torch.full((1,), float("-inf"), device="cuda", dtype=torch.float32)
        self.gbest_pos    = torch.empty((pso_dim,), device="cuda", dtype=torch.float32)

        # ── Position centers and search windows ──────────────────────────────
        self.pos_centers    = self._build_pos_centers()
        self.search_windows = self._build_search_windows()

        self._initialize(generate_positions=True, generate_velocities=True)
        self.gbest_pos.copy_(self.positions[0])

        self._partial_payoffs = torch.empty(
            (self.opt.num_paths // self.comp.pso_paths_block_size, self.num_particles),
            device="cuda", dtype=torch.float32,
        )

        buffer_size = self.comp.max_iterations // self.comp.sync_iters
        self.global_payoffs_cpu  = np.empty((buffer_size,), dtype=np.float32)
        self.global_payoff_index = 0

    # ── helpers ──────────────────────────────────────────────────────────────

    def _philox_reserve(self, span: int) -> int:
        allocated = self.offset
        self.offset += max(1, span)
        return allocated

    def _build_pos_centers(self) -> torch.Tensor:
        """[pso_num_dimensions] tensor of init centers.

        exercise_style=0: every dim centered on strike_price.
        exercise_style=1: dim (t * N + a) centered on S0_a.
        """
        N = self.opt.num_assets
        if self.opt.exercise_style == 0:
            return torch.full((self.pso_num_dimensions,), self.opt.strike_price,
                              device="cuda", dtype=torch.float32)
        else:
            per_step = torch.tensor(self.opt.initial_stock_prices,
                                    device="cuda", dtype=torch.float32)
            return per_step.repeat(self.mc_num_dimensions)

    def _build_search_windows(self) -> torch.Tensor:
        """[padded_assets] tensor of search window widths per asset.

        Uses perpetual American put floor: B* = K * 2r / (2r + sigma^2).
        Search window = 1 - B*/K = sigma^2 / (2r + sigma^2).

        exercise_style=0: uses basket effective volatility
          sigma_eff^2 = w' * Cov * w  (accounts for correlation).
        exercise_style=1: each asset gets its own width from its own vol.
        """
        r_val = self.opt.risk_free_rate

        if self.opt.exercise_style == 0:
            # Basket effective volatility: sigma_eff^2 = w' * Cov * w
            w = torch.tensor(self.opt.weights, device="cuda", dtype=torch.float32)
            vols = torch.tensor(self.opt.volatilities, device="cuda", dtype=torch.float32)
            corr = torch.tensor(self.opt.correlation_matrix, device="cuda", dtype=torch.float32)
            cov = corr * torch.outer(vols, vols)
            sig_eff_sq = (w @ cov @ w).item()
            floor_fraction = (2 * r_val) / (2 * r_val + sig_eff_sq)
            width = min(1.0 - floor_fraction, 0.99)
            return torch.full((self.padded_assets,), width,
                              device="cuda", dtype=torch.float32)
        else:
            vols_sq = self.vol_vec ** 2
            floor_fractions = (2 * r_val) / (2 * r_val + vols_sq)
            return torch.clamp(1.0 - floor_fractions, max=0.99)

    # ── precompute ───────────────────────────────────────────────────────────

    def _precompute_collapsed(self):
        grid = (triton.cdiv(self._num_bw_paths, self.comp.pso_paths_block_size),)
        mc_basket_collapse_kernel[grid](
            St_ptr=self.St, S0_ptr=self.S0_vec, drift_ptr=self.drift_vec,
            vol_ptr=self.vol_l2_vec, weights_ptr=self.weights_vec, L_ptr=self.L_mat,
            r=self.opt.risk_free_rate, dt=self.opt.time_step_size,
            num_bw_paths=self._num_bw_paths, seed=self.comp.seed,
            BLOCK_SIZE=tl.constexpr(self.comp.pso_paths_block_size),
            NUM_TIME_STEPS=tl.constexpr(self.mc_num_dimensions),
            BLOCK_SIZE_ASSETS=tl.constexpr(self.padded_assets),
            NUM_ASSETS=tl.constexpr(self.opt.num_assets),
            MC_OFFSET_PHILOX=tl.constexpr(self.mc_offset_philox),
            BANDWIDTH_PATHS_START=tl.constexpr(self._num_compute_paths),
            TOTAL_NUM_PATHS=tl.constexpr(self.opt.num_paths),
            USE_ANTITHETIC=tl.constexpr(self.comp.use_antithetic),
            USE_FP16=tl.constexpr(self.comp.use_fp16_cholesky),
        )

    def _precompute_per_asset(self):
        grid = (triton.cdiv(self._num_bw_paths, self.comp.pso_paths_block_size),)
        mc_basket_path_kernel[grid](
            st_ptr=self.St, S0_ptr=self.S0_vec, drift_ptr=self.drift_vec,
            vol_ptr=self.vol_l2_vec, L_ptr=self.L_mat,
            r=self.opt.risk_free_rate, dt=self.opt.time_step_size,
            num_bw_paths=self._num_bw_paths, seed=self.comp.seed,
            BLOCK_SIZE=tl.constexpr(self.comp.pso_paths_block_size),
            NUM_TIME_STEPS=tl.constexpr(self.mc_num_dimensions),
            BLOCK_SIZE_ASSETS=tl.constexpr(self.padded_assets),
            NUM_ASSETS=tl.constexpr(self.opt.num_assets),
            MC_OFFSET_PHILOX=tl.constexpr(self.mc_offset_philox),
            BANDWIDTH_PATHS_START=tl.constexpr(self._num_compute_paths),
            TOTAL_NUM_PATHS=tl.constexpr(self.opt.num_paths),
            USE_ANTITHETIC=tl.constexpr(self.comp.use_antithetic),
            USE_FP16=tl.constexpr(self.comp.use_fp16_cholesky),
        )

    # ── PSO loop ─────────────────────────────────────────────────────────────

    def _PSO_update(self, iteration: int, eval_only: bool = False):
        pso_dim = self.pso_num_dimensions

        if not eval_only:
            grid_pso = lambda meta: (
                triton.cdiv(pso_dim * self.num_particles, self.comp.elementwise_block_size),
            )
            pso_update_kernel[grid_pso](
                positions_ptr=self.positions,
                ln_positions_ptr=self.ln_positions,
                velocities_ptr=self.velocities,
                pbest_pos_ptr=self.pbest_pos,
                gbest_pos_ptr=self.gbest_pos,
                r1_ptr=self.r1,
                r2_ptr=self.r2,
                iteration=iteration,
                INERTIA_WEIGHT=self.swarm.inertia_weight,
                COGNITIVE_WEIGHT=self.swarm.cognitive_weight,
                SOCIAL_WEIGHT=self.swarm.social_weight,
                NUM_DIMENSIONS=pso_dim,
                NUM_PARTICLES=self.num_particles,
                PSO_OFFSET_PHILOX=self.pso_offset_philox,
                SEED=self.comp.seed,
                USE_FIXED_RANDOM=self.comp.use_fixed_random,
                BLOCK_SIZE=self.comp.elementwise_block_size,
            )

        grid_mc = lambda meta: (
            triton.cdiv(self.num_particles, meta["BLOCK_SIZE_PARTICLES"]),
            self.opt.num_paths // self.comp.pso_paths_block_size,
        )
        mc_basket_payoff_kernel[grid_mc](
            ln_positions_ptr=self.ln_positions,
            st_ptr=self.St,
            partial_payoffs_ptr=self._partial_payoffs,
            S0_ptr=self.S0_vec,
            drift_ptr=self.drift_vec,
            vol_ptr=self.vol_l2_vec,
            weights_ptr=self.weights_vec,
            L_ptr=self.L_mat,
            r=self.opt.risk_free_rate,
            dt=self.opt.time_step_size,
            SEED=self.comp.seed,
            NUM_DIMENSIONS=self.mc_num_dimensions,
            NUM_PARTICLES=self.num_particles,
            NUM_PATHS=self.opt.num_paths,
            NUM_COMPUTE_PATH_BLOCKS=self._num_compute_path_blocks,
            BLOCK_SIZE_PATHS=self.comp.pso_paths_block_size,
            BLOCK_SIZE_ASSETS=self.padded_assets,
            NUM_ASSETS=self.opt.num_assets,
            OPTION_TYPE=self.opt.option_type,
            STRIKE_PRICE=self.opt.strike_price,
            MC_OFFSET_PHILOX=self.mc_offset_philox,
            USE_ANTITHETIC=self.comp.use_antithetic,
            EXERCISE_STYLE=self.opt.exercise_style,
            USE_FP16=self.comp.use_fp16_cholesky,
        )

    def _reduce_pbest(self):
        pso_dim = self.pso_num_dimensions
        num_blocks = int(triton.cdiv(self.num_particles, self.comp.reduction_block_size))
        scratch_payoffs   = torch.empty((num_blocks,), device="cuda", dtype=torch.float32)
        scratch_positions = torch.empty((num_blocks, pso_dim), device="cuda", dtype=torch.float32)
        actual_num_path_blocks = self.opt.num_paths // self.comp.pso_paths_block_size

        reduce_pbest_local[(num_blocks,)](
            pbest_payoffs_ptr=self.pbest_payoff,
            pbest_positions_ptr=self.pbest_pos,
            scratch_payoffs_ptr=scratch_payoffs,
            scratch_positions_ptr=scratch_positions,
            partial_payoffs_ptr=self._partial_payoffs,
            positions_ptr=self.positions,
            NUM_PARTICLES=tl.constexpr(self.num_particles),
            NUM_BLOCKS=tl.constexpr(num_blocks),
            NUM_DIMENSIONS=tl.constexpr(pso_dim),
            BLOCK_SIZE=tl.constexpr(self.comp.reduction_block_size),
            NUM_PATH_BLOCKS=tl.constexpr(actual_num_path_blocks),
            NUM_PATHS=tl.constexpr(self.opt.num_paths),
        )
        reduce_pbest_global[(1,)](
            scratch_payoffs_ptr=scratch_payoffs,
            scratch_positions_ptr=scratch_positions,
            gbest_payoff_ptr=self.gbest_payoff,
            gbest_position_ptr=self.gbest_pos,
            NUM_BLOCKS=tl.constexpr(num_blocks),
            NUM_DIMENSIONS=tl.constexpr(pso_dim),
            BLOCK_SIZE=triton.next_power_of_2(num_blocks),
        )

    def _initialize(self, generate_positions=True, generate_velocities=True):
        pso_dim = self.pso_num_dimensions
        n_total = self.num_particles * pso_dim
        grid    = (triton.cdiv(n_total, self.comp.elementwise_block_size),)

        r1_off = self.r1_offset_philox if self.comp.use_fixed_random else 0
        r2_off = self.r2_offset_philox if self.comp.use_fixed_random else 0

        init_kernel[grid](
            positions_ptr=self.positions,
            ln_positions_ptr=self.ln_positions,
            velocities_ptr=self.velocities,
            pbest_costs_ptr=self.pbest_payoff,
            pbest_pos_ptr=self.pbest_pos,
            r1_ptr=self.r1,
            r2_ptr=self.r2,
            pos_centers_ptr=self.pos_centers,
            search_windows_ptr=self.search_windows,
            num_dimensions=pso_dim,
            num_particles=self.num_particles,
            seed=self.comp.seed,
            OPTION_TYPE=tl.constexpr(self.opt.option_type),
            EXERCISE_STYLE=tl.constexpr(self.opt.exercise_style),
            NUM_ASSETS=tl.constexpr(self.opt.num_assets),
            GENERATES_POSITIONS=tl.constexpr(generate_positions),
            GENERATES_VELOCITIES=tl.constexpr(generate_velocities),
            BLOCK_SIZE=tl.constexpr(self.comp.elementwise_block_size),
            POS_OFFSET_PHILOX=self.init_pos_philox,
            VEL_OFFSET_PHILOX=self.init_vel_philox,
            R1_OFFSET_PHILOX=r1_off,
            R2_OFFSET_PHILOX=r2_off,
            USE_FIXED_RANDOM=tl.constexpr(self.comp.use_fixed_random),
        )

    # ── debug ────────────────────────────────────────────────────────────────

    def _debug_print(self, iteration: int):
        if not self.comp.debug:
            return
        gbest_val = self.gbest_payoff.item()
        gbest_pos = self.gbest_pos.cpu().numpy()
        pos_preview = gbest_pos[:min(4, len(gbest_pos))]
        n_show = min(3, self.num_particles)
        print(f"[DEBUG iter={iteration:4d}] gbest_payoff={gbest_val:12.6f}  "
              f"gbest_pos[:4]={np.array2string(pos_preview, precision=2, suppress_small=True)}")
        for i in range(n_show):
            p_pos = self.positions[i].cpu().numpy()[:min(4, self.pso_num_dimensions)]
            print(f"  particle[{i}] pos[:4]={np.array2string(p_pos, precision=2, suppress_small=True)}")

    # ── public API ───────────────────────────────────────────────────────────

    def optimize(self):
        self.global_payoff_index = 0
        self._PSO_update(iteration=0, eval_only=True)
        self._reduce_pbest()
        self._debug_print(0)

        for iteration in range(0, self.comp.max_iterations, self.comp.sync_iters):
            for i in range(self.comp.sync_iters):
                self._PSO_update(iteration=iteration + i, eval_only=False)
                self._reduce_pbest()

            self.global_payoffs_cpu[self.global_payoff_index] = self.gbest_payoff.item()
            self.global_payoff_index += 1

            self._debug_print(iteration + self.comp.sync_iters)

            if self._is_converged():
                if self.comp.debug:
                    print(f"[DEBUG] Converged at iteration {iteration + self.comp.sync_iters}")
                break

    def get_option_price(self):
        """In-sample price (upper bound). May have upward bias from
        evaluating the optimized boundary on the same paths used to find it."""
        return self.global_payoffs_cpu[self.global_payoff_index - 1]

    def get_debiased_price(self):
        """Out-of-sample price. Evaluates the optimized boundary on fresh
        independent paths, eliminating optimization-over-estimation bias.

        Unbiased for V(tau*), the true value of the policy found by PSO.
        The only remaining gap to the true option value V* is PSO
        suboptimality, typically ~0.5% (see vanilla benchmark).

        Reference: Broadie & Glasserman (1997), 'Pricing American-style
        securities using simulation.'"""
        N = self.opt.num_assets

        # Allocate fresh Philox space (non-overlapping with optimization paths)
        fresh_mc_offset = self.offset
        self.offset += int(self.opt.num_paths) * int(self.mc_num_dimensions) * N

        saved_mc_offset = self.mc_offset_philox
        self.mc_offset_philox = fresh_mc_offset

        # Regenerate St with independent paths
        if self._num_bw_paths > 0:
            if self.opt.exercise_style == 0:
                self._precompute_collapsed()
            else:
                self._precompute_per_asset()

        # Reset payoffs so fresh evaluation gets recorded
        # (reduce only updates gbest if new payoff > stored payoff)
        self.pbest_payoff.fill_(float("-inf"))
        self.gbest_payoff.fill_(float("-inf"))

        # Evaluate existing boundary on fresh paths (no PSO update)
        self._PSO_update(iteration=0, eval_only=True)
        self._reduce_pbest()

        fresh_price = self.gbest_payoff.item()

        # Restore original MC offset
        self.mc_offset_philox = saved_mc_offset

        return fresh_price

    def get_gbest_position(self):
        return self.gbest_pos.cpu().numpy()

    def _is_converged(self):
        if self.comp.convergence_threshold < 0:
            return False
        if self.global_payoff_index < 2:
            return False
        return abs(
            self.global_payoffs_cpu[self.global_payoff_index - 1]
            - self.global_payoffs_cpu[self.global_payoff_index - 2]
        ) < self.comp.convergence_threshold
