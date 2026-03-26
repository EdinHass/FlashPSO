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
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask    = offsets < NUM_PARTICLES

    payoff_total = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    # Standard range replaces static_range
    for pb in range(NUM_PATH_BLOCKS):
        payoff_total += tl.load(partial_payoffs_ptr + offsets * NUM_PATH_BLOCKS + pb, mask=mask, other=0.0)
    
    payoff_avg         = payoff_total / NUM_PATHS
    pbest_payoff_block = tl.load(pbest_payoffs_ptr + offsets, mask=mask, other=float("-inf"))
    
    should_update      = payoff_avg > pbest_payoff_block
    new_pbest          = tl.where(should_update, payoff_avg, pbest_payoff_block)
    tl.store(pbest_payoffs_ptr + offsets, new_pbest, mask=mask)

    update_mask = mask & should_update
    for d in range(NUM_DIMENSIONS):
        pos_vals = tl.load(positions_ptr   + d * NUM_PARTICLES + offsets, mask=update_mask, other=0.0)
        tl.store(pbest_positions_ptr + d * NUM_PARTICLES + offsets, pos_vals, mask=update_mask)

    local_best_payoff = tl.max(new_pbest, axis=0)
    best_lane         = tl.argmax(new_pbest, axis=0)
    best_particle_idx = pid * BLOCK_SIZE + best_lane

    dim_idx       = tl.arange(0, NUM_DIMENSIONS)
    pos_ptrs      = pbest_positions_ptr + dim_idx * NUM_PARTICLES + best_particle_idx
    best_position = tl.load(pos_ptrs)

    scratch_pos_desc = tl.make_tensor_descriptor(
        base=scratch_positions_ptr,
        shape=[NUM_BLOCKS, NUM_DIMENSIONS],      # pyright: ignore[reportArgumentType]
        strides=[NUM_DIMENSIONS, 1],             # pyright: ignore[reportArgumentType]
        block_shape=[1, NUM_DIMENSIONS],
    )
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
    offsets = tl.arange(0, BLOCK_SIZE)
    mask    = offsets < NUM_BLOCKS
    payoffs = tl.load(scratch_payoffs_ptr + offsets, mask=mask, other=float("-inf"))
    
    global_best_payoff = tl.max(payoffs, axis=0)
    best_lane          = tl.argmax(payoffs, axis=0)
    
    scratch_pos_desc = tl.make_tensor_descriptor(
        base=scratch_positions_ptr,
        shape=[NUM_BLOCKS, NUM_DIMENSIONS],      # pyright: ignore[reportArgumentType]
        strides=[NUM_DIMENSIONS, 1],             # pyright: ignore[reportArgumentType]
        block_shape=[1, NUM_DIMENSIONS],
    )
    best_position = tl.load_tensor_descriptor(scratch_pos_desc, [best_lane, 0])
    
    gbest_pos_desc = tl.make_tensor_descriptor(
        base=gbest_position_ptr,
        shape=[1, NUM_DIMENSIONS],               # pyright: ignore[reportArgumentType]
        strides=[NUM_DIMENSIONS, 1],             # pyright: ignore[reportArgumentType]
        block_shape=[1, NUM_DIMENSIONS],
    )
    
    old_gbest_payoff = tl.load(gbest_payoff_ptr)
    if global_best_payoff > old_gbest_payoff:
        tl.store(gbest_payoff_ptr, global_best_payoff)
        tl.store_tensor_descriptor(gbest_pos_desc, [0, 0], best_position)
