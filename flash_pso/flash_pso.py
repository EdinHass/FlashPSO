import math
import numpy as np
from typing import Optional

import torch
import triton
import triton.language as tl

from flash_pso.config import OptionConfig, ComputeConfig, SwarmConfig
from flash_pso.pso_kernels import init_kernel, pso_kernel
from flash_pso.mc_kernels import mc_path_kernel_col_stores_tma
from flash_pso.pso_utils import reduce_pbest_local, reduce_pbest_global
from flash_pso.asserts import validate_inputs


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

        def _reserve(offset: int, span: int) -> int:
            return offset + max(1, span)

        mc_span   = int(self.opt.num_paths)    * int(self.num_dimensions)
        init_span = int(self.num_particles)    * int(self.num_dimensions)

        offset = 0
        self.mc_offset_philox  = offset;  offset = _reserve(offset, mc_span)
        self.init_pos_philox   = offset;  offset = _reserve(offset, init_span)
        self.init_vel_philox   = offset;  offset = _reserve(offset, init_span)
        self.pso_offset_philox = offset

        if not self.comp.compute_on_the_fly:
            if precomputed_St is not None:
                self.St = precomputed_St.to("cuda")
            else:
                # Column-major: allocate [NUM_TIME_STEPS, NUM_PATHS] then transpose.
                # Resulting strides [1, NUM_TIME_STEPS]: paths axis is contiguous.
                self.St = torch.empty(
                    (self.opt.num_time_steps, self.opt.num_paths), device="cuda", dtype=torch.float32
                ).t()
                self._precompute_mc_paths()
        else:
            self.St = torch.empty((0,), device="cuda")

        # Column-major: allocate [NUM_DIMENSIONS, NUM_PARTICLES] then transpose.
        # Resulting strides [1, NUM_DIMENSIONS]: particles axis is contiguous.
        self.positions = torch.empty(
            (self.num_dimensions, self.num_particles), device="cuda", dtype=torch.float32
        ).t()
        self.velocities = torch.empty(
            (self.num_dimensions, self.num_particles), device="cuda", dtype=torch.float32
        ).t()
        self.pbest_pos = torch.empty(
            (self.num_dimensions, self.num_particles), device="cuda", dtype=torch.float32
        ).t()

        if initial_positions is not None:
            self.positions.copy_(initial_positions.to("cuda"))
        if initial_velocities is not None:
            self.velocities.copy_(initial_velocities.to("cuda"))

        self.pbest_payoff = torch.empty((self.num_particles,), device="cuda", dtype=torch.float32)
        self.gbest_payoff = torch.full((1,), float("-inf"), device="cuda", dtype=torch.float32)
        self.gbest_pos    = torch.empty((self.num_dimensions,), device="cuda", dtype=torch.float32)

        self._initialize(initial_positions is None, initial_velocities is None)
        self.gbest_pos.copy_(self.positions[0])

        buffer_size = self.comp.max_iterations // self.comp.sync_iters
        self.global_payoffs_cpu  = np.empty((buffer_size,), dtype=np.float32)
        self.global_payoff_index = 0

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

    def _PSO_update(self, iteration: int, eval_only: bool = False):
        grid = lambda meta: (triton.cdiv(self.num_particles, meta["BLOCK_SIZE_PARTICLES"]),)

        pso_kernel[grid](
            positions_ptr=self.positions,
            velocities_ptr=self.velocities,
            pbest_payoff_ptr=self.pbest_payoff,
            pbest_pos_ptr=self.pbest_pos,
            gbest_payoff_ptr=self.gbest_payoff,
            gbest_pos_ptr=self.gbest_pos,
            st_ptr=self.St,
            iteration=iteration,
            S0=self.opt.initial_stock_price,
            r=self.opt.risk_free_rate,
            sigma=self.opt.volatility,
            dt=self.opt.time_step_size,
            SEED=tl.constexpr(self.comp.seed),
            INERTIA_WEIGHT=tl.constexpr(self.swarm.inertia_weight),
            COGNITIVE_WEIGHT=tl.constexpr(self.swarm.cognitive_weight),
            SOCIAL_WEIGHT=tl.constexpr(self.swarm.social_weight),
            NUM_DIMENSIONS=tl.constexpr(self.num_dimensions),
            NUM_PARTICLES=tl.constexpr(self.num_particles),
            NUM_PATHS=tl.constexpr(self.opt.num_paths),
            COMPUTE_ON_THE_FLY=tl.constexpr(self.comp.compute_on_the_fly),
            OPTION_TYPE=tl.constexpr(self.opt.option_type),
            STRIKE_PRICE=tl.constexpr(self.opt.strike_price),
            RISK_FREE_RATE=tl.constexpr(self.opt.risk_free_rate),
            TIME_STEP_SIZE=tl.constexpr(self.opt.time_step_size),
            PSO_OFFSET_PHILOX=tl.constexpr(self.pso_offset_philox),
            MC_OFFSET_PHILOX=tl.constexpr(self.mc_offset_philox),
            EVAL_ONLY=tl.constexpr(eval_only),
        )

    def _reduce_pbest(self):
        num_blocks = int(triton.cdiv(self.num_particles, self.comp.reduction_block_size))

        scratch_payoffs   = torch.empty((num_blocks,), device="cuda", dtype=torch.float32)
        scratch_positions = torch.empty((num_blocks, self.num_dimensions), device="cuda", dtype=torch.float32)

        reduce_pbest_local[(num_blocks,)](
            pbest_payoffs_ptr=self.pbest_payoff,
            pbest_positions_ptr=self.pbest_pos,
            scratch_payoffs_ptr=scratch_payoffs,
            scratch_positions_ptr=scratch_positions,
            NUM_PARTICLES=tl.constexpr(self.num_particles),
            NUM_BLOCKS=tl.constexpr(num_blocks),
            NUM_DIMENSIONS=tl.constexpr(self.num_dimensions),
            BLOCK_SIZE=tl.constexpr(self.comp.reduction_block_size),
        )

        stage2_block = triton.next_power_of_2(num_blocks)

        reduce_pbest_global[(1,)](
            scratch_payoffs_ptr=scratch_payoffs,
            scratch_positions_ptr=scratch_positions,
            gbest_payoff_ptr=self.gbest_payoff,
            gbest_position_ptr=self.gbest_pos,
            NUM_BLOCKS=tl.constexpr(num_blocks),
            NUM_DIMENSIONS=tl.constexpr(self.num_dimensions),
            BLOCK_SIZE=stage2_block,
        )

    def _is_converged(self):
        if self.global_payoff_index < 2:
            return False
        return abs(
            self.global_payoffs_cpu[self.global_payoff_index - 1]
            - self.global_payoffs_cpu[self.global_payoff_index - 2]
        ) < self.comp.convergence_threshold

    def _initialize(self, generate_positions=True, generate_velocities=True):
        n_total = self.num_particles * self.num_dimensions
        grid    = (triton.cdiv(n_total, self.comp.init_block_size),)

        init_kernel[grid](
            positions_ptr=self.positions,
            velocities_ptr=self.velocities,
            pbest_costs_ptr=self.pbest_payoff,
            pbest_pos_ptr=self.pbest_pos,
            num_dimensions=self.num_dimensions,
            num_particles=self.num_particles,
            seed=self.comp.seed,
            GENERATES_POSITIONS=tl.constexpr(generate_positions),
            GENERATES_VELOCITIES=tl.constexpr(generate_velocities),
            BLOCK_SIZE=tl.constexpr(self.comp.init_block_size),
            POS_OFFSET_PHILOX=tl.constexpr(self.init_pos_philox),
            VEL_OFFSET_PHILOX=tl.constexpr(self.init_vel_philox),
        )

    def _precompute_mc_paths(self):
        grid = (triton.cdiv(self.opt.num_paths, self.comp.mc_block_size),)

        mc_path_kernel_col_stores_tma[grid](
            St_ptr=self.St,
            S0=self.opt.initial_stock_price,
            r=self.opt.risk_free_rate,
            sigma=self.opt.volatility,
            dt=self.opt.time_step_size,
            num_paths=self.opt.num_paths,
            seed=self.comp.seed,
            BLOCK_SIZE=tl.constexpr(self.comp.mc_block_size),
            NUM_TIME_STEPS=tl.constexpr(self.opt.num_time_steps),
            MC_OFFSET_PHILOX=tl.constexpr(self.mc_offset_philox),
            )
