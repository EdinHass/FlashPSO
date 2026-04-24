from dataclasses import dataclass
import itertools
import triton
import math
from typing import List
 
from flash_pso.enums import OptionType, ExerciseStyle, OptionStyle, RNGType
 
 
@dataclass
class OptionConfig:
    initial_stock_price: float
    strike_price: float
    risk_free_rate: float
    volatility: float
    time_to_maturity: float
    num_paths: int
    num_time_steps: int
    option_type: OptionType = OptionType.PUT
    option_style: OptionStyle = OptionStyle.STANDARD
    barrier_level: float = float('inf')
 
    @property
    def time_step_size(self) -> float:
        return self.time_to_maturity / self.num_time_steps

    @property
    def log2_S0(self) -> float:
        return math.log2(self.initial_stock_price)

    @property
    def drift_l2(self) -> float:
        return (self.risk_free_rate - 0.5 * self.volatility * self.volatility) * self.time_step_size * math.log2(math.e)

    @property
    def vol_l2(self) -> float:
        return self.volatility * math.sqrt(self.time_step_size) * math.log2(math.e)

    @property
    def r_dt_l2(self) -> float:
        return -self.risk_free_rate * self.time_step_size * math.log2(math.e)

    @property
    def terminal_discount(self) -> float:
        return 2.0 ** (self.r_dt_l2 * self.num_time_steps)
 
 
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
    option_type: OptionType = OptionType.PUT
    option_style: OptionStyle = OptionStyle.BASKET
    exercise_style: ExerciseStyle = ExerciseStyle.SCALAR
 
    @property
    def num_assets(self) -> int:
        return len(self.initial_stock_prices)
 
    @property
    def time_step_size(self) -> float:
        return self.time_to_maturity / self.num_time_steps
 
 
@dataclass
class ComputeConfig:
    seed: int
    compute_fraction: float = 1.0
    pso_paths_block_size: int = 64
    elementwise_block_size: int = 256
    reduction_block_size: int = 32
    max_iterations: int = 1000
    sync_iters: int = 10
    convergence_threshold: float = 1e-6
    use_fixed_random: bool = False
    use_antithetic: bool = False
    use_fp16_cholesky: bool = False
    use_fp16_paths: bool = False
    rng_type: RNGType = RNGType.PHILOX
    randomize_paths: bool = False
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


def get_autotune_configs():
    """1-D (vanilla / Asian) payoff kernel autotune search space."""
    configs = []

    particle_blocks = [1, 4, 8, 16]
    dim_blocks      = [1, 4]
    warps           = [4, 8, 16]
    stages          = [1, 2, 3, 4]
    loop_unroll_l   = [1, 2, 4]
    loop_stages_l   = [1, 2]
    warp_spec_l     = [False]
    flatten_l       = [True]

    for pt, d, w, s, ls, lu, ws, fl in itertools.product(
        particle_blocks, dim_blocks, warps, stages,
        loop_stages_l, loop_unroll_l, warp_spec_l, flatten_l,
    ):
        if pt * d > 64:
            continue

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

    particle_blocks = [1, 4]            # Kept small due to exponential covariance math
    dim_blocks      = [1, 2]
    warps           = [4, 8, 16]
    stages          = [1, 2]            # Baskets chew through shared memory; cap at 2
    loop_unroll_l   = [1, 4, 8]
    loop_stages_l   = [1, 2]
    warp_spec_l     = [False]
    flatten_l       = [True]

    for pt, d, w, s, ls, lu, ws, fl in itertools.product(
        particle_blocks, dim_blocks, warps, stages,
        loop_stages_l, loop_unroll_l, warp_spec_l, flatten_l,
    ):
        if pt * d > 8:
            continue
            
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
