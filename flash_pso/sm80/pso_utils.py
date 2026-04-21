"""Utility kernels for PSO reductions (local and global bests)."""
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

    # Sum partial payoffs across all path blocks (dynamic loop avoids unroll bloat at 512+ blocks)
    payoff_total = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for pb in tl.range(NUM_PATH_BLOCKS):
        payoff_total += tl.load(partial_payoffs_ptr + pb * NUM_PARTICLES + offsets)

    payoff_avg = payoff_total / NUM_PATHS
    pbest_payoff = tl.load(pbest_payoffs_ptr + offsets)
    should_update = payoff_avg > pbest_payoff
    new_pbest = tl.where(should_update, payoff_avg, pbest_payoff)
    tl.store(pbest_payoffs_ptr + offsets, new_pbest)

    # Conditionally copy positions to pbest for improved particles
    for d in tl.range(NUM_DIMENSIONS, num_stages=2):
        pos_vals = tl.load(positions_ptr + d * NUM_PARTICLES + offsets)
        tl.store(pbest_positions_ptr + d * NUM_PARTICLES + offsets, pos_vals, mask=should_update)

    # Find block-local best and write particle index to scratch
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
        tl.store(gbest_position_ptr + dim_idx, best_position)
