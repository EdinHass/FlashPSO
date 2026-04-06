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
    # kernel has two branches — BLOCK_SIZE == 1 (scalar path, no TMA) or
    # BLOCK_SIZE >= 4 (TMA path, must be power of two). Values 2 and 3 are not
    # supported as they would silently fall into the wrong branch.
    _require(
        x == 1 or (x >= 4 and (x & (x - 1)) == 0),
        f"{name} must be 1 or a power of two >= 4 (got {x})"
    )

def validate_inputs(pso, manual_blocks=False, initial_positions=None, initial_velocities=None, precomputed_St=None) -> None:
    # compute and iteration constants
    _require_positive(pso.comp.sync_iters, "sync_iters")
    _require_positive(pso.comp.max_iterations, "max_iterations")
    _require(
        pso.comp.max_iterations % pso.comp.sync_iters == 0,
        f"max_iterations ({pso.comp.max_iterations}) must be divisible by sync_iters ({pso.comp.sync_iters})"
    )
    
    # Validate hybrid compute fraction
    _require(
        0.0 <= pso.comp.compute_fraction <= 1.0,
        f"compute_fraction must be between 0.0 and 1.0 (got {pso.comp.compute_fraction})"
    )

    if manual_blocks:
        _require(
            pso.swarm.num_particles % pso.comp.pso_particles_block_size == 0,
            f"num_particles ({pso.swarm.num_particles}) must be divisible by pso_particles_block_size ({pso.comp.pso_particles_block_size})"
        )
        _require(
            pso.opt.num_time_steps % pso.comp.pso_dim_block_size == 0,
            f"num_time_steps ({pso.opt.num_time_steps}) must be divisible by pso_dim_block_size ({pso.comp.pso_dim_block_size})"
        )
        _require_valid_block_size(pso.comp.pso_particles_block_size, "pso_particles_block_size")
        _require_valid_block_size(pso.comp.pso_dim_block_size, "pso_dim_block_size")

    # GLOBAL REQUIREMENT: pso_paths_block_size is NOT autotuned, so num_paths must always be divisible by it
    _require(
        pso.opt.num_paths % pso.comp.pso_paths_block_size == 0,
        f"num_paths ({pso.opt.num_paths}) must be divisible by pso_paths_block_size ({pso.comp.pso_paths_block_size})"
    )

    # optional tensor shapes (if provided, must match config)
    _require_shape(initial_positions, (pso.swarm.num_particles, pso.opt.num_time_steps), "initial_positions")
    _require_shape(initial_velocities, (pso.swarm.num_particles, pso.opt.num_time_steps), "initial_velocities")

    # precomputed_St shape logic (dependent on compute_fraction) and TMA stride requirements
    if precomputed_St is not None:
        total_path_blocks = pso.opt.num_paths // pso.comp.pso_paths_block_size
        num_compute_path_blocks = round(pso.comp.compute_fraction * total_path_blocks)
        num_bw_paths = (total_path_blocks - num_compute_path_blocks) * pso.comp.pso_paths_block_size
        
        _require_shape(precomputed_St, (num_bw_paths, pso.opt.num_time_steps), "precomputed_St")
        
        # TMA Hardware Constraint Validation
        _require(
            precomputed_St.stride(0) == 1,
            "precomputed_St MUST be Column-Major (stride(0) == 1) to satisfy TMA descriptor constraints. "
            "Ensure it is generated as log-space lnS and transposed e.g., torch.empty((time, paths)).t()"
        )

    # option params
    _require(pso.opt.option_type in (0, 1), "option_type must be 0 (Call) or 1 (Put)")
    _require_positive(pso.opt.initial_stock_price, "initial_stock_price")
    _require_positive(pso.opt.strike_price, "strike_price")
    _require_non_negative(pso.opt.risk_free_rate, "risk_free_rate")
    _require_non_negative(pso.opt.volatility, "volatility")
    _require_positive(pso.opt.time_to_maturity, "time_to_maturity")
    _require_power_of_two(pso.opt.num_paths, "num_paths")
    _require_power_of_two(pso.opt.num_time_steps, "num_time_steps")

    # swarm params
    _require_positive(pso.swarm.num_particles, "num_particles")
    _require_power_of_two(pso.swarm.num_particles, "num_particles")
    _require_non_negative(pso.swarm.inertia_weight, "inertia_weight")
    _require_non_negative(pso.swarm.cognitive_weight, "cognitive_weight")
    _require_non_negative(pso.swarm.social_weight, "social_weight")

    # compute config params
    _require_power_of_two(pso.comp.elementwise_block_size, "elementwise_block_size")
    _require_power_of_two(pso.comp.reduction_block_size, "reduction_block_size")

    total_elements = pso.swarm.num_particles * pso.opt.num_time_steps
    _require(
        total_elements % pso.comp.elementwise_block_size == 0,
        f"Total elements (particles * time_steps = {total_elements}) must be divisible by elementwise_block_size ({pso.comp.elementwise_block_size}). "
        "Increase your swarm size/dimensions, or decrease the elementwise_block_size."
    )

    # reduction_block_size must evenly divide num_particles to avoid phantom
    # lanes in the argmax — both are power-of-two so this means <=.
    _require(
        pso.comp.reduction_block_size <= pso.swarm.num_particles,
        f"reduction_block_size ({pso.comp.reduction_block_size}) must be <= num_particles ({pso.swarm.num_particles})"
    )

    # pso_paths_block_size always uses TMA so must be power-of-two >= 4
    _require(
        pso.comp.pso_paths_block_size >= 4 and (pso.comp.pso_paths_block_size & (pso.comp.pso_paths_block_size - 1)) == 0,
        f"pso_paths_block_size must be a power of two >= 4 (got {pso.comp.pso_paths_block_size})"
    )

    _require_non_negative(pso.comp.seed, "seed")
