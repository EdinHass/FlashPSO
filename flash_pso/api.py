import math
import numpy as np
from typing import Optional

import torch
import triton
import triton.language as tl

from flash_pso.config import OptionConfig, ComputeConfig, SwarmConfig
from flash_pso.asserts import validate_inputs

# ── HARDWARE DETECTION & DYNAMIC IMPORTS ──────────────────────────────────────
_cc_major, _ = torch.cuda.get_device_capability()

if _cc_major >= 9:
    print(f"Detected GPU with compute capability {_cc_major}. Using SM90 kernels.")
    from flash_pso.sm90.pso_kernels import init_kernel, pso_update_kernel, mc_payoff_kernel, mc_asian_payoff_kernel
    from flash_pso.sm90.mc_kernels import mc_path_kernel_col_stores
    from flash_pso.sm90.pso_utils import reduce_pbest_local, reduce_pbest_global
else:
    print(f"Detected GPU with compute capability {_cc_major}. Using SM80 kernels.")
    from flash_pso.sm80.pso_kernels import init_kernel, pso_update_kernel, mc_payoff_kernel, mc_asian_payoff_kernel
    from flash_pso.sm80.mc_kernels import mc_path_kernel_col_stores
    from flash_pso.sm80.pso_utils import reduce_pbest_local, reduce_pbest_global

def triton_alloc(size: int, align: int, stream: int) -> int:
    return torch.cuda.caching_allocator_alloc(size, device=torch.cuda.current_device())

def triton_free(ptr: int):
    torch.cuda.caching_allocator_delete(ptr)

triton.set_allocator(triton_alloc)
_ = torch.tensor([0.0], device="cuda:0") # Warmup


class FlashPSO:
    @property
    def num_particles(self) -> int:
        return self.swarm.num_particles

    @property
    def num_dimensions(self) -> int:
        return self.opt.num_time_steps

    def __init__(
        self,
        option_config: OptionConfig,
        compute_config: ComputeConfig,
        swarm_config: SwarmConfig,
        precomputed_St: Optional[torch.Tensor] = None,
        initial_velocities: Optional[torch.Tensor] = None,
        initial_positions: Optional[torch.Tensor] = None,
    ):
        self.opt   = option_config
        self.comp  = compute_config
        self.swarm = swarm_config

        validate_inputs(self, self.comp.manual_blocks, initial_positions, initial_velocities, precomputed_St)

        mc_span   = int(self.opt.num_paths) * int(self.num_dimensions)
        init_span = int(self.num_particles) * int(self.num_dimensions)

        self.offset = 0

        self.mc_offset_philox  = self._philox_reserve(mc_span)
        self.init_pos_philox   = self._philox_reserve(init_span)
        self.init_vel_philox   = self._philox_reserve(init_span)
        self.pso_offset_philox = self._philox_reserve(init_span * 2)

        if self.comp.use_fixed_random:
            self.r1_offset_philox = self._philox_reserve(init_span)
            self.r2_offset_philox = self._philox_reserve(init_span)

        total_path_blocks             = self.opt.num_paths // self.comp.pso_paths_block_size
        self._num_compute_path_blocks = round(self.comp.compute_fraction * total_path_blocks)
        _num_bw_path_blocks           = total_path_blocks - self._num_compute_path_blocks
        self._num_bw_paths            = _num_bw_path_blocks * self.comp.pso_paths_block_size
        self._num_compute_paths       = self._num_compute_path_blocks * self.comp.pso_paths_block_size

        if self._num_bw_paths > 0:
            if precomputed_St is not None:
                self.St = precomputed_St.to("cuda")
            else:
                self.St = torch.empty((self.opt.num_time_steps, self._num_bw_paths), device="cuda", dtype=torch.float32).t()
                self._precompute_mc_paths()
        else:
            self.St = torch.empty((0,), device="cuda")

        self.positions    = torch.empty((self.num_dimensions, self.num_particles), device="cuda", dtype=torch.float32).t()
        self.ln_positions = torch.empty((self.num_dimensions, self.num_particles), device="cuda", dtype=torch.float32).t()
        self.velocities   = torch.empty((self.num_dimensions, self.num_particles), device="cuda", dtype=torch.float32).t()
        self.pbest_pos    = torch.empty((self.num_dimensions, self.num_particles), device="cuda", dtype=torch.float32).t()

        if self.comp.use_fixed_random:
            self.r1 = torch.empty((self.num_dimensions, self.num_particles), device="cuda", dtype=torch.float32).t()
            self.r2 = torch.empty((self.num_dimensions, self.num_particles), device="cuda", dtype=torch.float32).t()
        else:
            self.r1 = torch.empty((0,), device="cuda")
            self.r2 = torch.empty((0,), device="cuda")

        if initial_positions  is not None: self.positions.copy_(initial_positions.to("cuda"))
        if initial_velocities is not None: self.velocities.copy_(initial_velocities.to("cuda"))

        self.pbest_payoff = torch.empty((self.num_particles,), device="cuda", dtype=torch.float32)
        self.gbest_payoff = torch.full((1,), float("-inf"), device="cuda", dtype=torch.float32)
        self.gbest_pos    = torch.empty((self.num_dimensions,), device="cuda", dtype=torch.float32)

        # ── Position centers and search windows ──────────────────────────────
        self.pos_centers = torch.full(
            (self.num_dimensions,), self.opt.strike_price,
            device="cuda", dtype=torch.float32,
        )
        self.search_windows = self._build_search_windows()

        self._initialize(initial_positions is None, initial_velocities is None)
        self.gbest_pos.copy_(self.positions[0])

        self._partial_payoffs = torch.empty(
            (self.opt.num_paths // self.comp.pso_paths_block_size, self.num_particles),
            device="cuda", dtype=torch.float32,
        )

        buffer_size = self.comp.max_iterations // self.comp.sync_iters
        self.global_payoffs_cpu  = np.empty((buffer_size,), dtype=np.float32)
        self.global_payoff_index = 0

    def _build_search_windows(self) -> torch.Tensor:
        """Single-element tensor with search window width.
        Uses perpetual put floor: sigma^2 / (2r + sigma^2)."""
        r_val = self.opt.risk_free_rate
        sig_val = self.opt.volatility
        floor_fraction = (2 * r_val) / (2 * r_val + sig_val ** 2)
        width = min(1.0 - floor_fraction, 0.99)
        return torch.tensor([width], device="cuda", dtype=torch.float32)

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
        return self.global_payoffs_cpu[self.global_payoff_index - 1]

    def get_gbest_position(self):
        return self.gbest_pos.cpu().numpy()

    def _philox_reserve(self, span: int) -> int:
        allocated = self.offset
        self.offset += max(1, span)
        return allocated

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
            p_pos = self.positions[i].cpu().numpy()[:min(4, self.num_dimensions)]
            print(f"  particle[{i}] pos[:4]={np.array2string(p_pos, precision=2, suppress_small=True)}")

    def _PSO_update(self, iteration: int, eval_only: bool = False):
        if not eval_only:
            grid_pso = lambda meta: (triton.cdiv(self.num_dimensions * self.num_particles, self.comp.elementwise_block_size),)

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
                NUM_DIMENSIONS=self.num_dimensions,
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

        if self.opt.option_style.lower() == "asian":
            active_mc_payoff_kernel = mc_asian_payoff_kernel
        else:
            active_mc_payoff_kernel = mc_payoff_kernel

        active_mc_payoff_kernel[grid_mc](
            ln_positions_ptr=self.ln_positions,
            st_ptr=self.St,
            partial_payoffs_ptr=self._partial_payoffs,
            S0=self.opt.initial_stock_price,
            r=self.opt.risk_free_rate,
            sigma=self.opt.volatility,
            dt=self.opt.time_step_size,
            SEED=self.comp.seed,
            NUM_DIMENSIONS=self.num_dimensions,
            NUM_PARTICLES=self.num_particles,
            NUM_PATHS=self.opt.num_paths,
            NUM_COMPUTE_PATH_BLOCKS=self._num_compute_path_blocks,
            BLOCK_SIZE_PATHS=self.comp.pso_paths_block_size,
            OPTION_TYPE=self.opt.option_type,
            STRIKE_PRICE=self.opt.strike_price,
            MC_OFFSET_PHILOX=self.mc_offset_philox,
            USE_ANTITHETIC=self.comp.use_antithetic,
        )

    def _reduce_pbest(self):
        num_blocks = int(triton.cdiv(self.num_particles, self.comp.reduction_block_size))

        scratch_payoffs   = torch.empty((num_blocks,), device="cuda", dtype=torch.float32)
        scratch_positions = torch.empty((num_blocks, self.num_dimensions), device="cuda", dtype=torch.float32)
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
            NUM_DIMENSIONS=tl.constexpr(self.num_dimensions),
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
            NUM_DIMENSIONS=tl.constexpr(self.num_dimensions),
            BLOCK_SIZE=triton.next_power_of_2(num_blocks),
        )

    def _initialize(self, generate_positions=True, generate_velocities=True):
        n_total = self.num_particles * self.num_dimensions
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
            num_dimensions=self.num_dimensions,
            num_particles=self.num_particles,
            seed=self.comp.seed,
            OPTION_TYPE=tl.constexpr(self.opt.option_type),
            EXERCISE_STYLE=tl.constexpr(0),  # vanilla is always scalar boundary
            NUM_ASSETS=tl.constexpr(1),       # vanilla is single asset
            GENERATES_POSITIONS=tl.constexpr(generate_positions),
            GENERATES_VELOCITIES=tl.constexpr(generate_velocities),
            BLOCK_SIZE=tl.constexpr(self.comp.elementwise_block_size),
            POS_OFFSET_PHILOX=self.init_pos_philox,
            VEL_OFFSET_PHILOX=self.init_vel_philox,
            R1_OFFSET_PHILOX=r1_off,
            R2_OFFSET_PHILOX=r2_off,
            USE_FIXED_RANDOM=tl.constexpr(self.comp.use_fixed_random),
        )

    def _precompute_mc_paths(self):
        grid = (triton.cdiv(self._num_bw_paths, self.comp.pso_paths_block_size),)

        mc_path_kernel_col_stores[grid](
            St_ptr=self.St,
            S0=self.opt.initial_stock_price,
            r=self.opt.risk_free_rate,
            sigma=self.opt.volatility,
            dt=self.opt.time_step_size,
            num_paths=self._num_bw_paths,
            seed=self.comp.seed,
            BLOCK_SIZE=tl.constexpr(self.comp.pso_paths_block_size),
            NUM_TIME_STEPS=tl.constexpr(self.opt.num_time_steps),
            MC_OFFSET_PHILOX=tl.constexpr(self.mc_offset_philox),
            BANDWIDTH_PATHS_START=tl.constexpr(self._num_compute_paths),
            USE_ANTITHETIC=tl.constexpr(self.comp.use_antithetic),
        )

    def _is_converged(self):
        if self.comp.convergence_threshold < 0:
            return False
        if self.global_payoff_index < 2:
            return False
        return abs(
            self.global_payoffs_cpu[self.global_payoff_index - 1]
            - self.global_payoffs_cpu[self.global_payoff_index - 2]
        ) < self.comp.convergence_threshold
