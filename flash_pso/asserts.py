"""Input validation for FlashPSO and FlashPSOBasket.

Raises ValueError with descriptive messages for invalid configurations.
"""
from flash_pso.enums import RNGType, ExerciseStyle

def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(msg)

def _require_positive(x, name: str) -> None:
    _require(x > 0, f"{name} must be positive")

def _require_non_negative(x, name: str) -> None:
    _require(x >= 0, f"{name} must be non-negative")

def _require_shape(x, shape, name: str) -> None:
    _require(
        x is None or tuple(x.shape) == tuple(shape),
        f"{name} shape mismatch: expected {shape}, got {None if x is None else tuple(x.shape)}"
    )

def _require_power_of_two(x: int, name: str) -> None:
    _require(x > 0 and (x & (x - 1)) == 0, f"{name} must be a power of two (got {x})")

def _require_valid_block_size(x: int, name: str) -> None:
    _require(
        x == 1 or (x >= 4 and (x & (x - 1)) == 0),
        f"{name} must be 1 or a power of two >= 4 (got {x})"
    )

def validate_inputs(pso, initial_positions=None,
                    initial_velocities=None, precomputed_St=None) -> None:
    """Validate vanilla FlashPSO configuration."""
    _validate_common(pso)
    
    _require((pso.opt.num_time_steps * pso.swarm.num_particles) % pso.comp.elementwise_block_size == 0,
             "Total grid size (num_time_steps * num_particles) must be divisible by elementwise_block_size for unmasked ops.")

    _require_shape(initial_positions, (pso.swarm.num_particles, pso.opt.num_time_steps), "initial_positions")
    _require_shape(initial_velocities, (pso.swarm.num_particles, pso.opt.num_time_steps), "initial_velocities")

    if precomputed_St is not None:
        total_path_blocks = pso.opt.num_paths // pso.comp.pso_paths_block_size
        num_compute_path_blocks = round(pso.comp.compute_fraction * total_path_blocks)
        num_bw_paths = (total_path_blocks - num_compute_path_blocks) * pso.comp.pso_paths_block_size
        _require_shape(precomputed_St, (num_bw_paths, pso.opt.num_time_steps), "precomputed_St")
        _require(precomputed_St.stride(0) == 1,
                 "precomputed_St must be column-major (stride(0)==1) for TMA")

    _require(pso.opt.option_type in (0, 1), "option_type must be CALL(0) or PUT(1)")
    _require_positive(pso.opt.initial_stock_price, "initial_stock_price")
    _require_positive(pso.opt.strike_price, "strike_price")
    _require_non_negative(pso.opt.risk_free_rate, "risk_free_rate")
    _require_non_negative(pso.opt.volatility, "volatility")
    _require_positive(pso.opt.time_to_maturity, "time_to_maturity")
    _require_power_of_two(pso.opt.num_paths, "num_paths")
    _require_power_of_two(pso.opt.num_time_steps, "num_time_steps")


def validate_basket_inputs(pso) -> None:
    """Validate FlashPSOBasket configuration."""
    _validate_common(pso)

    N = pso.opt.num_assets
    _require_power_of_two(N, "num_assets")
    
    _require((pso.opt.num_time_steps * N * pso.swarm.num_particles) % pso.comp.elementwise_block_size == 0,
             "Total grid size (num_time_steps * num_assets * num_particles) must be divisible by elementwise_block_size for unmasked ops.")

    _require(N >= 2, f"Basket requires at least 2 assets (got {N})")
    _require(len(pso.opt.volatilities) == N,
             f"volatilities length ({len(pso.opt.volatilities)}) must match num_assets ({N})")
    _require(len(pso.opt.weights) == N,
             f"weights length ({len(pso.opt.weights)}) must match num_assets ({N})")
    _require(len(pso.opt.correlation_matrix) == N,
             f"correlation_matrix must be {N}x{N}")
    for row in pso.opt.correlation_matrix:
        _require(len(row) == N, f"correlation_matrix rows must have length {N}")

    _require(pso.opt.option_type in (0, 1), "option_type must be CALL(0) or PUT(1)")
    _require(pso.opt.exercise_style in (0, 1), "exercise_style must be SCALAR(0) or PER_ASSET(1)")
    _require_positive(pso.opt.strike_price, "strike_price")
    _require_non_negative(pso.opt.risk_free_rate, "risk_free_rate")
    _require_positive(pso.opt.time_to_maturity, "time_to_maturity")
    _require_power_of_two(pso.opt.num_paths, "num_paths")
    _require_power_of_two(pso.opt.num_time_steps, "num_time_steps")

    for s0 in pso.opt.initial_stock_prices:
        _require_positive(s0, "initial_stock_price")
    for v in pso.opt.volatilities:
        _require_non_negative(v, "volatility")
    _require(abs(sum(pso.opt.weights) - 1.0) < 1e-6,
             f"weights must sum to 1.0 (got {sum(pso.opt.weights):.6f})")

    if pso.comp.rng_type == RNGType.SOBOL:
        _require(pso.comp.compute_fraction == 0.0,
                 "Sobol requires compute_fraction=0.0 (precompute only)")
        _require(not pso.comp.use_antithetic,
                 "Antithetic is redundant with Sobol quasi-random sequences")


def _validate_common(pso) -> None:
    """Shared validation for both vanilla and basket."""
    _require_positive(pso.comp.sync_iters, "sync_iters")
    _require_positive(pso.comp.max_iterations, "max_iterations")
    _require(pso.comp.max_iterations % pso.comp.sync_iters == 0,
             f"max_iterations must be divisible by sync_iters")
    _require(0.0 <= pso.comp.compute_fraction <= 1.0,
             f"compute_fraction must be in [0, 1]")
    _require(pso.opt.num_paths % pso.comp.pso_paths_block_size == 0,
             f"num_paths must be divisible by pso_paths_block_size")

    _require_positive(pso.swarm.num_particles, "num_particles")
    _require_power_of_two(pso.swarm.num_particles, "num_particles")
    _require_non_negative(pso.swarm.inertia_weight, "inertia_weight")
    _require_non_negative(pso.swarm.cognitive_weight, "cognitive_weight")
    _require_non_negative(pso.swarm.social_weight, "social_weight")

    _require_power_of_two(pso.comp.elementwise_block_size, "elementwise_block_size")
    _require_power_of_two(pso.comp.reduction_block_size, "reduction_block_size")
    _require(pso.comp.reduction_block_size <= pso.swarm.num_particles,
             f"reduction_block_size must be <= num_particles")
    _require(pso.comp.pso_paths_block_size >= 4
             and (pso.comp.pso_paths_block_size & (pso.comp.pso_paths_block_size - 1)) == 0,
             f"pso_paths_block_size must be a power of two >= 4")
    _require_non_negative(pso.comp.seed, "seed")

    if pso.comp.rng_type == RNGType.SOBOL:
        _require(pso.comp.compute_fraction == 0.0,
                 "Sobol requires compute_fraction=0.0 (precompute only)")
        _require(not pso.comp.use_antithetic,
                 "Antithetic is redundant with Sobol quasi-random sequences")
