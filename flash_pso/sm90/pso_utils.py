"""Utility kernels for PSO reductions (SM90 — TMA loads/stores where applicable).

SM90 differences from SM80:
  - TMA load for partial_payoffs (when BLOCK_SIZE >= 4).
  - TMA store for gbest_position via descriptor.
"""
import triton
import triton.language as tl


@triton.jit
def reduce_pbest_local(
    pbest_payoffs_ptr,
    pbest_positions_ptr,
    scratch_payoffs_ptr,
    scratch_particle_idx_ptr,
    partial_payoffs_ptr,
    positions_ptr,
    NUM_PARTICLES: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    NUM_DIMENSIONS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUM_PATH_BLOCKS: tl.constexpr,
    NUM_PATHS: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = (pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)).to(tl.int64)

    payoff_total = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    if BLOCK_SIZE >= 4:
        partial_desc = tl.make_tensor_descriptor(
            base=partial_payoffs_ptr,
            shape=[NUM_PATH_BLOCKS, NUM_PARTICLES],
            strides=[NUM_PARTICLES, 1],
            block_shape=[1, BLOCK_SIZE],
        )
        for pb in tl.range(NUM_PATH_BLOCKS):
            tile = tl.load_tensor_descriptor(partial_desc, [pb, pid * BLOCK_SIZE])
            payoff_total += tl.reshape(tile, [BLOCK_SIZE])
    else:
        for pb in tl.range(NUM_PATH_BLOCKS):
            payoff_total += tl.load(partial_payoffs_ptr + pb * NUM_PARTICLES + offsets)

    payoff_avg = payoff_total / NUM_PATHS
    pbest_payoff = tl.load(pbest_payoffs_ptr + offsets)
    should_update = payoff_avg > pbest_payoff
    new_pbest = tl.where(should_update, payoff_avg, pbest_payoff)
    tl.store(pbest_payoffs_ptr + offsets, new_pbest)

    for d in tl.static_range(NUM_DIMENSIONS):
        pos_vals = tl.load(positions_ptr + d * NUM_PARTICLES + offsets)
        tl.store(pbest_positions_ptr + d * NUM_PARTICLES + offsets, pos_vals, mask=should_update)

    local_best_payoff = tl.max(new_pbest, axis=0)
    best_lane = tl.argmax(new_pbest, axis=0)
    best_particle_idx = pid * BLOCK_SIZE + best_lane

    tl.store(scratch_payoffs_ptr + pid, local_best_payoff)
    tl.store(scratch_particle_idx_ptr + pid, best_particle_idx)


@triton.jit
def reduce_pbest_global(
    scratch_payoffs_ptr,
    scratch_particle_idx_ptr,
    pbest_positions_ptr,
    gbest_payoff_ptr,
    gbest_position_ptr,
    NUM_PARTICLES: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    NUM_DIMENSIONS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_SIZE).to(tl.int64)
    mask = offsets < NUM_BLOCKS
    payoffs = tl.load(scratch_payoffs_ptr + offsets, mask=mask, other=float("-inf"))

    global_best_payoff = tl.max(payoffs, axis=0)
    best_lane = tl.argmax(payoffs, axis=0)

    old_gbest_payoff = tl.load(gbest_payoff_ptr)

    if global_best_payoff > old_gbest_payoff:
        best_idx = tl.load(scratch_particle_idx_ptr + best_lane)
        dim_idx = tl.arange(0, NUM_DIMENSIONS)
        best_position = tl.load(pbest_positions_ptr + dim_idx * NUM_PARTICLES + best_idx)

        tl.store(gbest_payoff_ptr, global_best_payoff)
        if NUM_DIMENSIONS >= 4:
            gbest_desc = tl.make_tensor_descriptor(
                base=gbest_position_ptr,
                shape=[1, NUM_DIMENSIONS],
                strides=[NUM_DIMENSIONS, 1],
                block_shape=[1, NUM_DIMENSIONS],
            )
            tl.store_tensor_descriptor(gbest_desc, [0, 0], best_position[None, :])
        else:
            tl.store(gbest_position_ptr + dim_idx, best_position)
