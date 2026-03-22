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

def validate_inputs(pso, manual_blocks=False, initial_positions=None, initial_velocities=None, precomputed_St=None) -> None:
    # ── COMPUTE & ITERATION CONSTRAINTS ──────────────────────────────────────
    _require_positive(pso.comp.sync_iters, "sync_iters")
    _require_positive(pso.comp.max_iterations, "max_iterations")
    _require(
        pso.comp.max_iterations % pso.comp.sync_iters == 0,
        f"max_iterations ({pso.comp.max_iterations}) must be divisible by sync_iters ({pso.comp.sync_iters})"
    )

    # ── HARDWARE BLOCK DIVISIBILITY ──────────────────────────────────────────
    if manual_blocks:
        _require( 
            pso.swarm.num_particles % pso.comp.pso_particles_block_size == 0, 
            f"num_particles ({pso.swarm.num_particles}) must be divisible by pso_particles_block_size ({pso.comp.pso_particles_block_size})" 
        ) 
        _require( 
            pso.opt.num_paths % pso.comp.pso_paths_block_size == 0, 
            f"num_paths ({pso.opt.num_paths}) must be divisible by pso_paths_block_size ({pso.comp.pso_paths_block_size})" 
        )
        _require(
            pso.opt.num_time_steps % pso.comp.pso_dim_block_size == 0,
            f"num_time_steps ({pso.opt.num_time_steps}) must be divisible by pso_dim_block_size ({pso.comp.pso_dim_block_size})"
        )

    # ── OPTIONAL TENSOR SHAPES ───────────────────────────────────────────────
    _require_shape(initial_positions, (pso.swarm.num_particles, pso.opt.num_time_steps), "initial_positions")
    _require_shape(initial_velocities, (pso.swarm.num_particles, pso.opt.num_time_steps), "initial_velocities")
    _require_shape(precomputed_St, (pso.opt.num_paths, pso.opt.num_time_steps), "precomputed_St")

    # ── OPTION PARAMS ────────────────────────────────────────────────────────
    _require(pso.opt.option_type in (0, 1), "option_type must be 0 (Call) or 1 (Put)")
    _require_positive(pso.opt.initial_stock_price, "initial_stock_price")
    _require_positive(pso.opt.strike_price, "strike_price")
    _require_non_negative(pso.opt.risk_free_rate, "risk_free_rate")
    _require_non_negative(pso.opt.volatility, "volatility")
    _require_positive(pso.opt.time_to_maturity, "time_to_maturity")
    _require_positive(pso.opt.num_paths, "num_paths")
    _require_positive(pso.opt.num_time_steps, "num_time_steps")

    # ── SWARM PARAMS ─────────────────────────────────────────────────────────
    _require_positive(pso.swarm.num_particles, "num_particles")
    _require_non_negative(pso.swarm.inertia_weight, "inertia_weight")
    _require_non_negative(pso.swarm.cognitive_weight, "cognitive_weight")
    _require_non_negative(pso.swarm.social_weight, "social_weight")

    # ── COMPUTE PARAMS ───────────────────────────────────────────────────────
    _require_positive(pso.comp.mc_block_size, "mc_block_size")
    _require_positive(pso.comp.pso_particles_block_size, "pso_particles_block_size")
    _require_positive(pso.comp.pso_paths_block_size, "pso_paths_block_size")
    _require_positive(pso.comp.pso_dim_block_size, "pso_dim_block_size")
    _require_positive(pso.comp.init_block_size, "init_block_size")
    _require_positive(pso.comp.reduction_block_size, "reduction_block_size")
    _require_positive(pso.comp.convergence_threshold, "convergence_threshold")
    _require_non_negative(pso.comp.seed, "seed")
