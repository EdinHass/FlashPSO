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
    compute_on_the_fly: bool = True

    manual_blocks: bool = False
    pso_particles_block_size: int = 256
    pso_paths_block_size: int = 256
    pso_dim_block_size: int = 64
    mc_block_size: int = 1024
    init_block_size: int = 256
    reduction_block_size: int = 256

    seed: int = 42

    convergence_threshold: float = 1e-6
    max_iterations: int = 1000
    sync_iters: int = 10

    # if true uses fixed r1 and r2
    use_fixed_random: bool = False


@dataclass
class SwarmConfig:
    num_particles: int
    inertia_weight: float = 0.5
    cognitive_weight: float = 0.5
    social_weight: float = 0.5


def get_autotune_configs():
    configs = []
    path_blocks     = [64]
    particle_blocks = [1]
    dim_blocks      = [1]
    warps           = [4, 8]
    stages          = [1]
    num_ctas        = [1]

    for p, pt, d, w, s, cta in itertools.product(path_blocks, particle_blocks, dim_blocks, warps, stages, num_ctas):
        total_elements = p * pt * d
        if total_elements > 65536:
            continue
        configs.append(
            triton.Config(
                {"BLOCK_SIZE_PATHS": p, "BLOCK_SIZE_PARTICLES": pt, "BLOCK_SIZE_DIM": d},
                num_warps=w,
                num_stages=s,
                num_ctas=cta,
            )
        )

    return configs
