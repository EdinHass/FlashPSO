from dataclasses import dataclass
import itertools
import triton
from typing import List


@dataclass
class OptionConfig:
    initial_stock_price: float
    strike_price: float
    risk_free_rate: float
    volatility: float
    time_to_maturity: float
    num_paths: int
    num_time_steps: int
    option_type: int = 1
    option_style: str = "standard"
    barrier_level: float = float('inf')

    @property
    def time_step_size(self) -> float:
        return self.time_to_maturity / self.num_time_steps


@dataclass
class BasketOptionConfig:
    initial_stock_prices: List[float]
    strike_price: float
    risk_free_rate: float
    volatilities: List[float]
    weights: List[float]
    correlation_matrix: List[List[float]]
    time_to_maturity: float
    num_paths: int
    num_time_steps: int
    option_type: int = 1          # 0 = Call, 1 = Put
    option_style: str = "basket"
    exercise_style: int = 0       # 0 = scalar boundary, 1 = per-asset boundaries

    @property
    def num_assets(self) -> int:
        return len(self.initial_stock_prices)

    @property
    def time_step_size(self) -> float:
        return self.time_to_maturity / self.num_time_steps


@dataclass
class ComputeConfig:
    compute_fraction: float = 1.0
    manual_blocks: bool = False
    pso_paths_block_size: int = 256
    pso_particles_block_size: int = 256
    pso_dim_block_size: int = 64
    elementwise_block_size: int = 256
    reduction_block_size: int = 32
    max_iterations: int = 1000
    sync_iters: int = 10
    convergence_threshold: float = 1e-6
    seed: int = 42
    use_fixed_random: bool = False
    use_antithetic: bool = False
    use_fp16_cholesky: bool = False
    debug: bool = False

    @property
    def compute_on_the_fly(self) -> bool:
        return self.compute_fraction > 0.0


@dataclass
class SwarmConfig:
    num_particles: int
    inertia_weight: float = 0.7298
    cognitive_weight: float = 1.49618 
    social_weight: float = 1.49618 


# ═══════════════════════════════════════════════════════════════════════════════
# AUTOTUNE CONFIGURATIONS
# ═══════════════════════════════════════════════════════════════════════════════

def get_autotune_configs():
    """1-D (vanilla / Asian) payoff kernel autotune search space."""
    configs = []

    particle_blocks = [1]
    dim_blocks      = [1]
    warps           = [8]
    stages          = [1]
    loop_unroll_l   = [8]
    loop_stages_l   = [1]
    warp_spec_l     = [False]
    flatten_l       = [True]

    for pt, d, w, s, ls, lu, ws, fl in itertools.product(
        particle_blocks, dim_blocks, warps, stages,
        loop_stages_l, loop_unroll_l, warp_spec_l, flatten_l,
    ):
        if ws and w < 4:
            continue
        configs.append(triton.Config(
            {
                "BLOCK_SIZE_PARTICLES": pt,
                "BLOCK_SIZE_DIM": d,
                "LOOP_STAGES": ls,
                "LOOP_UNROLL": lu,
                "WARP_SPECIALIZE": ws,
                "LOOP_FLATTEN": fl,
            },
            num_warps=w,
            num_stages=s,
            num_ctas=1,
        ))
    return configs


def get_basket_autotune_configs():
    """Basket payoff kernel autotune search space.

    BLOCK_SIZE_ASSETS is NOT autotuned — it is set dynamically by the API
    to next_power_of_2(num_assets).
    """
    configs = []

    particle_blocks = [1]
    dim_blocks      = [1]
    warps           = [8]
    stages          = [1]
    loop_unroll_l   = [8]
    loop_stages_l   = [1]
    warp_spec_l     = [False]
    flatten_l       = [True]

    for pt, d, w, s, ls, lu, ws, fl in itertools.product(
        particle_blocks, dim_blocks, warps, stages,
        loop_stages_l, loop_unroll_l, warp_spec_l, flatten_l,
    ):
        if ws and w < 4:
            continue
        configs.append(triton.Config(
            {
                "BLOCK_SIZE_PARTICLES": pt,
                "BLOCK_SIZE_DIM": d,
                "LOOP_STAGES": ls,
                "LOOP_UNROLL": lu,
                "WARP_SPECIALIZE": ws,
                "LOOP_FLATTEN": fl,
            },
            num_warps=w,
            num_stages=s,
            num_ctas=1,
        ))
    return configs
