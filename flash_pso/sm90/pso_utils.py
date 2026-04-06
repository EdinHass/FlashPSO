import triton
import triton.language as tl

@triton.jit
def reduce_pbest_local(
    pbest_payoffs_ptr,
    pbest_positions_ptr,
    scratch_payoffs_ptr,
    scratch_positions_ptr,
    partial_payoffs_ptr,
    positions_ptr,
    NUM_PARTICLES:    tl.constexpr,
    NUM_BLOCKS:       tl.constexpr,
    NUM_DIMENSIONS:   tl.constexpr,
    BLOCK_SIZE:       tl.constexpr,
    NUM_PATH_BLOCKS:  tl.constexpr,
    NUM_PATHS:        tl.constexpr,
):
    pid     = tl.program_id(0)
    offsets = (pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)).to(tl.int64)

    # TMA descriptors
    scratch_pos_desc = tl.make_tensor_descriptor(
        base=scratch_positions_ptr,
        shape=[NUM_BLOCKS, NUM_DIMENSIONS],
        strides=[NUM_DIMENSIONS, 1],
        block_shape=[1, NUM_DIMENSIONS],
    )

    # Sum partial payoffs across all path blocks
    # Layout: [NUM_PATH_BLOCKS, NUM_PARTICLES] — contiguous in particles
    payoff_total = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    if BLOCK_SIZE >= 4:
        partial_desc = tl.make_tensor_descriptor(
            base=partial_payoffs_ptr,
            shape=[NUM_PATH_BLOCKS, NUM_PARTICLES],
            strides=[NUM_PARTICLES, 1],
            block_shape=[1, BLOCK_SIZE],
        )
        for pb in range(NUM_PATH_BLOCKS):
            tile = tl.load_tensor_descriptor(partial_desc, [pb, pid * BLOCK_SIZE])
            payoff_total += tl.reshape(tile, [BLOCK_SIZE])
    else:
        for pb in range(NUM_PATH_BLOCKS):
            payoff_total += tl.load(partial_payoffs_ptr + pb * NUM_PARTICLES + offsets)

    payoff_avg    = payoff_total / NUM_PATHS
    pbest_payoff  = tl.load(pbest_payoffs_ptr + offsets)

    should_update = payoff_avg > pbest_payoff
    new_pbest     = tl.where(should_update, payoff_avg, pbest_payoff)
    tl.store(pbest_payoffs_ptr + offsets, new_pbest)

    # Conditionally copy positions to pbest for improved particles
    for d in range(NUM_DIMENSIONS):
        pos_vals = tl.load(positions_ptr + d * NUM_PARTICLES + offsets, mask=should_update, other=0.0)
        tl.store(pbest_positions_ptr + d * NUM_PARTICLES + offsets, pos_vals, mask=should_update)

    # Find block-local best and write to scratch
    local_best_payoff = tl.max(new_pbest, axis=0)
    best_lane         = tl.argmax(new_pbest, axis=0)
    best_particle_idx = pid * BLOCK_SIZE + best_lane

    dim_idx       = tl.arange(0, NUM_DIMENSIONS)
    best_position = tl.load(pbest_positions_ptr + dim_idx * NUM_PARTICLES + best_particle_idx)

    tl.store(scratch_payoffs_ptr + pid, local_best_payoff)
    tl.store_tensor_descriptor(scratch_pos_desc, [pid, 0], best_position[None, :])

@triton.jit
def reduce_pbest_global(
    scratch_payoffs_ptr,
    scratch_positions_ptr,
    gbest_payoff_ptr,
    gbest_position_ptr,
    NUM_BLOCKS:     tl.constexpr,
    NUM_DIMENSIONS: tl.constexpr,
    BLOCK_SIZE:     tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_SIZE).to(tl.int64)
    payoffs = tl.load(scratch_payoffs_ptr + offsets)

    global_best_payoff = tl.max(payoffs, axis=0)
    best_lane          = tl.argmax(payoffs, axis=0)

    # TMA load for scratch positions
    scratch_pos_desc = tl.make_tensor_descriptor(
        base=scratch_positions_ptr,
        shape=[NUM_BLOCKS, NUM_DIMENSIONS],
        strides=[NUM_DIMENSIONS, 1],
        block_shape=[1, NUM_DIMENSIONS],
    )
    best_position = tl.load_tensor_descriptor(scratch_pos_desc, [best_lane, 0])

    # TMA store for gbest position
    gbest_pos_desc = tl.make_tensor_descriptor(
        base=gbest_position_ptr,
        shape=[1, NUM_DIMENSIONS],
        strides=[NUM_DIMENSIONS, 1],
        block_shape=[1, NUM_DIMENSIONS],
    )

    old_gbest_payoff = tl.load(gbest_payoff_ptr)
    if global_best_payoff > old_gbest_payoff:
        tl.store(gbest_payoff_ptr, global_best_payoff)
        tl.store_tensor_descriptor(gbest_pos_desc, [0, 0], best_position)
