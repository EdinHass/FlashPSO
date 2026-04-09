"""PSO optimizer kernels (SM90 — TMA for contiguous position/velocity access).

SM90 differences from SM80:
  - init_kernel: TMA stores for positions, ln_positions, velocities, pbest_pos.
  - pso_update_kernel: TMA loads/stores for positions, velocities, pbest_pos, ln_positions.
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
    """Initialize PSO particle positions, velocities, and pbest. SM90: TMA stores."""
    pid = tl.program_id(0)
    offs = (pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)).to(tl.int64)

    p_idx = offs % NUM_PARTICLES
    d_idx = offs // NUM_PARTICLES

    TOTAL = NUM_DIMENSIONS * NUM_PARTICLES

    if BLOCK_SIZE >= 4:
        pos_desc = tl.make_tensor_descriptor(base=positions_ptr, shape=[TOTAL], strides=[1], block_shape=[BLOCK_SIZE])
        ln_pos_desc = tl.make_tensor_descriptor(base=ln_positions_ptr, shape=[TOTAL], strides=[1], block_shape=[BLOCK_SIZE])
        pbest_desc = tl.make_tensor_descriptor(base=pbest_pos_ptr, shape=[TOTAL], strides=[1], block_shape=[BLOCK_SIZE])
        vel_desc = tl.make_tensor_descriptor(base=velocities_ptr, shape=[TOTAL], strides=[1], block_shape=[BLOCK_SIZE])

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

        ln_pos = tl.log2(tl.maximum(pos, 1e-4))

        if BLOCK_SIZE >= 4:
            tl.store_tensor_descriptor(pos_desc, [pid * BLOCK_SIZE], pos)
            tl.store_tensor_descriptor(ln_pos_desc, [pid * BLOCK_SIZE], ln_pos)
            tl.store_tensor_descriptor(pbest_desc, [pid * BLOCK_SIZE], pos)
        else:
            tl.store(positions_ptr + offs, pos)
            tl.store(ln_positions_ptr + offs, ln_pos)
            tl.store(pbest_pos_ptr + offs, pos)
    else:
        pos = tl.load(positions_ptr + offs)
        ln_pos = tl.log2(tl.maximum(pos, 1e-4))
        if BLOCK_SIZE >= 4:
            tl.store_tensor_descriptor(ln_pos_desc, [pid * BLOCK_SIZE], ln_pos)
            tl.store_tensor_descriptor(pbest_desc, [pid * BLOCK_SIZE], pos)
        else:
            tl.store(ln_positions_ptr + offs, ln_pos)
            tl.store(pbest_pos_ptr + offs, pos)

    if GENERATES_VELOCITIES:
        center = tl.load(pos_centers_ptr + d_idx)
        vel = (tl.rand(seed, VEL_OFFSET_PHILOX + offs) - 0.5) * center * 0.1
        if BLOCK_SIZE >= 4:
            tl.store_tensor_descriptor(vel_desc, [pid * BLOCK_SIZE], vel)
        else:
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
    """Standard PSO velocity + position update. SM90: TMA loads/stores for contiguous arrays."""
    pid = tl.program_id(0)
    offs = (pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)).to(tl.int64)

    p_idx = offs % NUM_PARTICLES
    d_idx = offs // NUM_PARTICLES

    TOTAL = NUM_DIMENSIONS * NUM_PARTICLES

    if BLOCK_SIZE >= 4:
        pos_desc = tl.make_tensor_descriptor(base=positions_ptr, shape=[TOTAL], strides=[1], block_shape=[BLOCK_SIZE])
        vel_desc = tl.make_tensor_descriptor(base=velocities_ptr, shape=[TOTAL], strides=[1], block_shape=[BLOCK_SIZE])
        pbest_desc = tl.make_tensor_descriptor(base=pbest_pos_ptr, shape=[TOTAL], strides=[1], block_shape=[BLOCK_SIZE])
        ln_pos_desc = tl.make_tensor_descriptor(base=ln_positions_ptr, shape=[TOTAL], strides=[1], block_shape=[BLOCK_SIZE])

        pos = tl.load_tensor_descriptor(pos_desc, [pid * BLOCK_SIZE])
        vel = tl.load_tensor_descriptor(vel_desc, [pid * BLOCK_SIZE])
        pbest = tl.load_tensor_descriptor(pbest_desc, [pid * BLOCK_SIZE])
    else:
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

    if BLOCK_SIZE >= 4:
        tl.store_tensor_descriptor(vel_desc, [pid * BLOCK_SIZE], new_vel)
        tl.store_tensor_descriptor(pos_desc, [pid * BLOCK_SIZE], new_pos)
        tl.store_tensor_descriptor(ln_pos_desc, [pid * BLOCK_SIZE], tl.log2(tl.maximum(new_pos, 1e-4)))
    else:
        tl.store(velocities_ptr + offs, new_vel)
        tl.store(positions_ptr + offs, new_pos)
        tl.store(ln_positions_ptr + offs, tl.log2(tl.maximum(new_pos, 1e-4)))
