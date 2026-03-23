import triton
import triton.language as tl


# stage 1: each block reduces its own pbest and puts it into scratch
@triton.jit
def reduce_pbest_local(
    pbest_payoffs_ptr,
    pbest_positions_ptr,
    scratch_payoffs_ptr,
    scratch_positions_ptr,
    NUM_PARTICLES:  tl.constexpr,
    NUM_BLOCKS:     tl.constexpr,
    NUM_DIMENSIONS: tl.constexpr,
    BLOCK_SIZE:     tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask    = offsets < NUM_PARTICLES
    payoffs = tl.load(pbest_payoffs_ptr + offsets, mask=mask, other=float("-inf"))
    local_best_payoff = tl.max(payoffs, axis=0)
    best_lane         = tl.argmax(payoffs, axis=0)
    best_particle_idx = pid * BLOCK_SIZE + best_lane

    dim_idx       = tl.arange(0, NUM_DIMENSIONS)
    pos_ptrs      = pbest_positions_ptr + dim_idx * NUM_PARTICLES + best_particle_idx
    best_position = tl.load(pos_ptrs)                                        # [NUM_DIMENSIONS]

    scratch_pos_desc = tl.make_tensor_descriptor(
        base=scratch_positions_ptr,
        shape=[NUM_BLOCKS, NUM_DIMENSIONS],      # pyright: ignore[reportArgumentType]
        strides=[NUM_DIMENSIONS, 1],             # pyright: ignore[reportArgumentType]
        block_shape=[1, NUM_DIMENSIONS],
    )
    tl.store(scratch_payoffs_ptr + pid, local_best_payoff)
    tl.store_tensor_descriptor(scratch_pos_desc, [pid, 0], best_position[None, :])


# stage 2: single block reduces the block-level pbests from scratch to global gbest
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
