import math
import numpy as np
from typing import Optional

import torch
import triton
import triton.language as tl

from flash_pso.config import OptionConfig, ComputeConfig, SwarmConfig
from flash_pso.pso_kernels import init_kernel, pso_update_kernel, mc_payoff_kernel
from flash_pso.mc_kernels import mc_path_kernel_col_stores_tma
from flash_pso.pso_utils import reduce_pbest_local, reduce_pbest_global
from flash_pso.asserts import validate_inputs

# ── Custom Triton Allocator ───────────────────────────────────────────────────
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

        # ── Philox RNG Offset Reservations ───────────────────────────────────
        def _reserve(offset: int, span: int) -> int:
            return offset + max(1, span)

        mc_span   = int(self.opt.num_paths) * int(self.num_dimensions)
        init_span = int(self.num_particles) * int(self.num_dimensions)

        offset = 0
        self.mc_offset_philox  = offset; offset = _reserve(offset, mc_span)
        self.init_pos_philox   = offset; offset = _reserve(offset, init_span)
        self.init_vel_philox   = offset; offset = _reserve(offset, init_span)
        self.pso_offset_philox = offset; offset = _reserve(offset, init_span * 2)

        if self.comp.use_fixed_random:
            self.r1_offset_philox = offset; offset = _reserve(offset, init_span)
            self.r2_offset_philox = offset; offset = _reserve(offset, init_span)

        # ── Hybrid Compute/Bandwidth Balancing ───────────────────────────────
        total_path_blocks             = self.opt.num_paths // self.comp.pso_paths_block_size
        self._num_compute_path_blocks = round(self.comp.compute_fraction * total_path_blocks)
        self._num_bw_path_blocks      = total_path_blocks - self._num_compute_path_blocks
        self._num_bw_paths            = self._num_bw_path_blocks * self.comp.pso_paths_block_size
        self._num_compute_paths       = self._num_compute_path_blocks * self.comp.pso_paths_block_size

        # ── Memory Allocations ───────────────────────────────────────────────
        # Paths (Stored as Log-Space lnS if generated internally)
        if self._num_bw_paths > 0:
            if precomputed_St is not None:
                self.St = precomputed_St.to("cuda")
            else:
                self.St = torch.empty((self.opt.num_time_steps, self._num_bw_paths), device="cuda", dtype=torch.float32).t()
                self._precompute_mc_paths()
        else:
            self.St = torch.empty((0,), device="cuda")

        # Swarm State
        self.positions  = torch.empty((self.num_dimensions, self.num_particles), device="cuda", dtype=torch.float32).t()
        self.velocities = torch.empty((self.num_dimensions, self.num_particles), device="cuda", dtype=torch.float32).t()
        self.pbest_pos  = torch.empty((self.num_dimensions, self.num_particles), device="cuda", dtype=torch.float32).t()

        if self.comp.use_fixed_random:
            self.r1 = torch.empty((self.num_dimensions, self.num_particles), device="cuda", dtype=torch.float32).t()
            self.r2 = torch.empty((self.num_dimensions, self.num_particles), device="cuda", dtype=torch.float32).t()
        else:
            self.r1 = torch.empty((0,), device="cuda")
            self.r2 = torch.empty((0,), device="cuda")

        # Custom Initializations
        if initial_positions  is not None: self.positions.copy_(initial_positions.to("cuda"))
        if initial_velocities is not None: self.velocities.copy_(initial_velocities.to("cuda"))

        # Reductions & Payoffs
        self.pbest_payoff = torch.empty((self.num_particles,), device="cuda", dtype=torch.float32)
        self.gbest_payoff = torch.full((1,), float("-inf"), device="cuda", dtype=torch.float32)
        self.gbest_pos    = torch.empty((self.num_dimensions,), device="cuda", dtype=torch.float32)

        self._initialize(initial_positions is None, initial_velocities is None)
        self.gbest_pos.copy_(self.positions[0])

        self._partial_payoffs = torch.empty(
            (self.num_particles, self.opt.num_paths // self.comp.pso_paths_block_size),
            device="cuda", dtype=torch.float32,
        )

        # Precompute Absolute Discounts
        discounts = [math.exp(-self.opt.risk_free_rate * self.opt.time_step_size * (t + 1)) for t in range(self.num_dimensions)]
        self._discounts = torch.tensor(discounts, device="cuda", dtype=torch.float32).unsqueeze(0)

        # CPU Tracking
        buffer_size = self.comp.max_iterations // self.comp.sync_iters
        self.global_payoffs_cpu  = np.empty((buffer_size,), dtype=np.float32)
        self.global_payoff_index = 0

    # ── Core Optimization Loop ───────────────────────────────────────────────
    def optimize(self):
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
        else:
            print(f"[FlashPSO] reached max iterations ({self.comp.max_iterations})", flush=True)

    def get_option_price(self):
        return self.global_payoffs_cpu[self.global_payoff_index - 1]

    def get_gbest_position(self):
        return self.gbest_pos.cpu().numpy()

    # ── Kernel Launchers ─────────────────────────────────────────────────────
    def _PSO_update(self, iteration: int, eval_only: bool = False):
        if not eval_only:
            # 1. Light-weight 1D kernel for velocity/position math only
            BLOCK_SIZE_PSO = 256
            grid_pso = lambda meta: (triton.cdiv(self.num_dimensions * self.num_particles, BLOCK_SIZE_PSO),)

            pso_update_kernel[grid_pso](
                self.positions, self.velocities, self.pbest_pos, self.gbest_pos,
                self.r1, self.r2, iteration,
                self.swarm.inertia_weight, self.swarm.cognitive_weight, self.swarm.social_weight,
                self.num_dimensions, self.num_particles,
                self.pso_offset_philox, self.comp.seed,
                self.comp.use_fixed_random, BLOCK_SIZE_PSO
            )

        # 2. Dense, unrolled 2D/3D evaluation kernel
        grid_mc = lambda meta: (
            triton.cdiv(self.num_particles, meta["BLOCK_SIZE_PARTICLES"]),
            self.opt.num_paths // self.comp.pso_paths_block_size,
        )

        mc_payoff_kernel[grid_mc](
            positions_ptr=self.positions,
            st_ptr=self.St,
            partial_payoffs_ptr=self._partial_payoffs,
            discounts_ptr=self._discounts,
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
        grid    = (triton.cdiv(n_total, self.comp.init_block_size),)

        r1_off = self.r1_offset_philox if self.comp.use_fixed_random else 0
        r2_off = self.r2_offset_philox if self.comp.use_fixed_random else 0

        init_kernel[grid](
            positions_ptr=self.positions,
            velocities_ptr=self.velocities,
            pbest_costs_ptr=self.pbest_payoff,
            pbest_pos_ptr=self.pbest_pos,
            r1_ptr=self.r1,
            r2_ptr=self.r2,
            num_dimensions=self.num_dimensions,
            num_particles=self.num_particles,
            seed=self.comp.seed,
            GENERATES_POSITIONS=tl.constexpr(generate_positions),
            GENERATES_VELOCITIES=tl.constexpr(generate_velocities),
            BLOCK_SIZE=tl.constexpr(self.comp.init_block_size),
            POS_OFFSET_PHILOX=tl.constexpr(self.init_pos_philox),
            VEL_OFFSET_PHILOX=tl.constexpr(self.init_vel_philox),
            R1_OFFSET_PHILOX=tl.constexpr(r1_off),
            R2_OFFSET_PHILOX=tl.constexpr(r2_off),
            USE_FIXED_RANDOM=tl.constexpr(self.comp.use_fixed_random),
        )

    def _precompute_mc_paths(self):
        grid = (triton.cdiv(self._num_bw_paths, self.comp.mc_block_size),)

        mc_path_kernel_col_stores_tma[grid](
            St_ptr=self.St,
            S0=self.opt.initial_stock_price,
            r=self.opt.risk_free_rate,
            sigma=self.opt.volatility,
            dt=self.opt.time_step_size,
            num_paths=self._num_bw_paths,
            seed=self.comp.seed,
            BLOCK_SIZE=tl.constexpr(self.comp.mc_block_size),
            NUM_TIME_STEPS=tl.constexpr(self.opt.num_time_steps),
            MC_OFFSET_PHILOX=tl.constexpr(self.mc_offset_philox),
            PATH_START=tl.constexpr(self._num_compute_paths),
            USE_ANTITHETIC=tl.constexpr(self.comp.use_antithetic),
        )
    
    def _is_converged(self):
        if self.global_payoff_index < 2:
            return False
        return abs(
            self.global_payoffs_cpu[self.global_payoff_index - 1]
            - self.global_payoffs_cpu[self.global_payoff_index - 2]
        ) < self.comp.convergence_threshold
