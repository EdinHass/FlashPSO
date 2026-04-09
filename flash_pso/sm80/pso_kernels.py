"""PSO optimizer kernels: particle initialization and velocity update.
"""
import triton
import triton.language as tl

@triton.jit
def init_kernel(
    positions_ptr, ln_positions_ptr, velocities_ptr, pbest_costs_ptr,
    pbest_pos_ptr, r1_ptr, r2_ptr, pos_centers_ptr, search_windows_ptr,
    seed,
    OPTION_TYPE: tl.constexpr,
    EXERCISE_STYLE: tl.constexpr,
    NUM_ASSETS: tl.constexpr,
    GENERATES_POSITIONS: tl.constexpr,
    GENERATES_VELOCITIES: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    POS_OFFSET_PHILOX: tl.constexpr,
    VEL_OFFSET_PHILOX: tl.constexpr,
    R1_OFFSET_PHILOX: tl.constexpr,
    R2_OFFSET_PHILOX: tl.constexpr,
    USE_FIXED_RANDOM: tl.constexpr,
    NUM_DIMENSIONS: tl.constexpr,
    NUM_PARTICLES: tl.constexpr,
):
    """Initialize PSO particle positions, velocities, and pbest.

    Positions are initialized as flat boundaries within the perpetual American
    put/call floor/ceiling. Each particle gets one random depth (p_idx-based),
    applied uniformly across all timestep dimensions.

    Layout: positions/velocities stored column-major [dims, particles].
    """
    pid = tl.program_id(0)
    offs = (pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)).to(tl.int64)

    p_idx = offs % NUM_PARTICLES
    d_idx = offs // NUM_PARTICLES

    if GENERATES_POSITIONS:
        center = tl.load(pos_centers_ptr + d_idx)
        particle_rand = tl.rand(seed, POS_OFFSET_PHILOX + p_idx)

        if EXERCISE_STYLE == 0:
            a_idx = 0
        else:
            a_idx = d_idx % NUM_ASSETS
        search_window = tl.load(search_windows_ptr + a_idx)

        if OPTION_TYPE == 1:  
            pos = center * (1.0 - search_window * particle_rand)
        else:  
            pos = center * (1.0 + search_window * particle_rand)

        tl.store(positions_ptr + offs, pos)
        tl.store(ln_positions_ptr + offs, tl.log2(tl.maximum(pos, 1e-4)))
        tl.store(pbest_pos_ptr + offs, pos)
    else:
        pos = tl.load(positions_ptr + offs)
        tl.store(ln_positions_ptr + offs, tl.log2(tl.maximum(pos, 1e-4)))
        tl.store(pbest_pos_ptr + offs, pos)

    if GENERATES_VELOCITIES:
        center = tl.load(pos_centers_ptr + d_idx)
        vel = (tl.rand(seed, VEL_OFFSET_PHILOX + offs) - 0.5) * center * 0.1
        tl.store(velocities_ptr + offs, vel)

    if USE_FIXED_RANDOM:
        tl.store(r1_ptr + offs, tl.rand(seed, R1_OFFSET_PHILOX + offs))
        tl.store(r2_ptr + offs, tl.rand(seed, R2_OFFSET_PHILOX + offs))

    tl.store(pbest_costs_ptr + p_idx, float("-inf"), mask=(d_idx == 0))


@triton.jit
def pso_update_kernel(
    positions_ptr, ln_positions_ptr, velocities_ptr, pbest_pos_ptr,
    gbest_pos_ptr, r1_ptr, r2_ptr, iteration,
    INERTIA_WEIGHT: tl.constexpr,
    COGNITIVE_WEIGHT: tl.constexpr,
    SOCIAL_WEIGHT: tl.constexpr,
    NUM_DIMENSIONS: tl.constexpr,
    NUM_PARTICLES: tl.constexpr,
    PSO_OFFSET_PHILOX: tl.constexpr,
    SEED: tl.constexpr,
    USE_FIXED_RANDOM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Standard PSO velocity + position update with Constriction coefficients."""
    pid = tl.program_id(0)
    offs = (pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)).to(tl.int64)

    p_idx = offs % NUM_PARTICLES
    d_idx = offs // NUM_PARTICLES

    pos = tl.load(positions_ptr + offs)
    vel = tl.load(velocities_ptr + offs)
    pbest = tl.load(pbest_pos_ptr + offs)
    gbest = tl.load(gbest_pos_ptr + d_idx)

    if USE_FIXED_RANDOM:
        r1 = tl.load(r1_ptr + offs)
        r2 = tl.load(r2_ptr + offs)
    else:
        r1 = tl.rand(SEED, PSO_OFFSET_PHILOX + 2 * iteration * NUM_PARTICLES * NUM_DIMENSIONS + offs)
        r2 = tl.rand(SEED, PSO_OFFSET_PHILOX + (2 * iteration + 1) * NUM_PARTICLES * NUM_DIMENSIONS + offs)

    new_vel = INERTIA_WEIGHT * vel + COGNITIVE_WEIGHT * r1 * (pbest - pos) + SOCIAL_WEIGHT * r2 * (gbest - pos)
    new_pos = pos + new_vel

    tl.store(velocities_ptr + offs, new_vel)
    tl.store(positions_ptr + offs, new_pos)
    tl.store(ln_positions_ptr + offs, tl.log2(tl.maximum(new_pos, 1e-4)))
