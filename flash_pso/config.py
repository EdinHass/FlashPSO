from dataclasses import dataclass
import itertools
import triton


@dataclass
class OptionConfig:
    # Market + Asset Params
    initial_stock_price: float
    strike_price: float
    risk_free_rate: float
    volatility: float
    time_to_maturity: float
    # Simulation Params
    num_paths: int
    num_time_steps: int
    option_type: int = 1        # 1=Put, 0=Call

    @property
    def time_step_size(self) -> float:
        return self.time_to_maturity / self.num_time_steps


@dataclass
class ComputeConfig:
    # compute_fraction controls what fraction of path blocks are generated
    # on-the-fly vs loaded from precomputed St. 1.0 = fully compute-on-the-fly
    # (no St allocation), 0.0 = fully bandwidth (all paths precomputed),
    # 0.5 = half and half. Allows tuning arithmetic intensity to the roofline.
    compute_fraction: float = 1.0

    manual_blocks: bool = False
    pso_paths_block_size: int = 256 
    pso_particles_block_size: int = 256
    pso_dim_block_size: int = 64
    mc_block_size: int = 1024
    init_block_size: int = 256
    reduction_block_size: int = 256
    seed: int = 42
    convergence_threshold: float = 1e-6
    max_iterations: int = 1000
    sync_iters: int = 10
    use_fixed_random: bool = False
    use_antithetic: bool = False

    @property
    def compute_on_the_fly(self) -> bool:
        return self.compute_fraction > 0.0


@dataclass
class SwarmConfig:
    num_particles: int
    inertia_weight: float = 0.5
    cognitive_weight: float = 0.5
    social_weight: float = 0.5


def get_autotune_configs():
    configs = []

    # BLOCK_SIZE_PATHS is NOT autotuned — fixed by pso_paths_block_size in ComputeConfig
    particle_blocks = [1]
    dim_blocks      = [1, 4]
    warps           = [4, 8]
    stages          = [1]
    loop_stages_l   = [1]
    loop_unroll_l   = [1, 4]
    warp_spec_l     = [False]
    flatten_l       = [False, True]

    for pt, d, w, s, ls, lu, ws, fl in itertools.product(
        particle_blocks, dim_blocks, warps, stages, loop_stages_l, loop_unroll_l,
        warp_spec_l, flatten_l,
    ):
        # warp_specialize requires >=4 warps
        if ws and w < 4:
            continue

        configs.append(
            triton.Config(
                {
                    "BLOCK_SIZE_PARTICLES": pt,
                    "BLOCK_SIZE_DIM":       d,
                    "LOOP_STAGES":          ls,
                    "LOOP_UNROLL":          lu,
                    "WARP_SPECIALIZE":      ws,
                    "LOOP_FLATTEN":         fl,
                },
                num_warps=w,
                num_stages=s,
                num_ctas=1,
            )
        )

    return configs
