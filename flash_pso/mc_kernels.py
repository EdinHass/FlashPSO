import triton
import triton.language as tl


# VERSION 1: Original per-step stores (low register pressure, coalesced writes)
@triton.jit
def mc_path_kernel_row_stores(
    St_ptr,
    S0, r, sigma, dt,
    num_paths, num_time_steps, seed,
    BLOCK_SIZE: tl.constexpr,
    MC_OFFSET_PHILOX: tl.constexpr,
):
    pid = tl.program_id(0)
    path_offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = path_offsets < num_paths
    drift = (r - 0.5 * sigma * sigma) * dt
    volatility_scaler = sigma * tl.sqrt(dt)
    current_lnS = tl.full([BLOCK_SIZE], tl.log(S0), dtype=tl.float32)
    for t in range(num_time_steps):
        random_offset = MC_OFFSET_PHILOX + path_offsets * num_time_steps + t
        Z = tl.randn(seed, random_offset)
        current_lnS = current_lnS + drift + volatility_scaler * Z
        # Column-major store: element [path, t] is at t * num_paths + path
        store_ptr = St_ptr + t * num_paths + path_offsets
        tl.store(store_ptr, tl.exp(current_lnS), mask=mask)


# VERSION 2: TMA full-block store (higher register pressure, one store per block)
@triton.jit
def mc_path_kernel_block_stores(
    St_ptr,
    S0, r, sigma, dt,
    num_paths, seed,
    BLOCK_SIZE: tl.constexpr,
    NUM_TIME_STEPS: tl.constexpr,
    MC_OFFSET_PHILOX: tl.constexpr,
):
    pid = tl.program_id(0)
    path_offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Column-major: strides [1, num_paths] — paths axis is contiguous per dim step.
    current_block = tl.make_tensor_descriptor(
        base=St_ptr,
        shape=[num_paths, NUM_TIME_STEPS],       # pyright: ignore[reportArgumentType]
        strides=[1, num_paths],                  # column-major # pyright: ignore[reportArgumentType]
        block_shape=[BLOCK_SIZE, NUM_TIME_STEPS],
    )
    drift = (r - 0.5 * sigma * sigma) * dt
    volatility_scaler = sigma * tl.sqrt(dt)
    current_lnS = tl.full([BLOCK_SIZE], tl.log(S0), dtype=tl.float32)
    result = tl.zeros([BLOCK_SIZE, NUM_TIME_STEPS], dtype=tl.float32)
    for t in tl.static_range(NUM_TIME_STEPS):
        random_offset = MC_OFFSET_PHILOX + path_offsets * NUM_TIME_STEPS + t
        Z = tl.randn(seed, random_offset)
        current_lnS = current_lnS + drift + volatility_scaler * Z
        col_mask = tl.arange(0, NUM_TIME_STEPS)[None, :] == t
        result = tl.where(col_mask, tl.exp(current_lnS)[:, None], result)
    tl.store_tensor_descriptor(current_block, [pid * BLOCK_SIZE, 0], result)


# VERSION 3: per-step column stores.
# Column-major layout: element [path, t] is at t * num_paths + path.
@triton.jit
def mc_path_kernel_col_stores_tma(
    St_ptr,
    S0, r, sigma, dt,
    num_paths, seed,
    BLOCK_SIZE: tl.constexpr,
    NUM_TIME_STEPS: tl.constexpr,
    MC_OFFSET_PHILOX: tl.constexpr,
):
    pid = tl.program_id(0)
    path_offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = path_offsets < num_paths
    drift = (r - 0.5 * sigma * sigma) * dt
    volatility_scaler = sigma * tl.sqrt(dt)
    current_lnS = tl.full([BLOCK_SIZE], tl.log(S0), dtype=tl.float32)
    for t in tl.static_range(NUM_TIME_STEPS):  # pyright: ignore
        random_offset = MC_OFFSET_PHILOX + path_offsets * NUM_TIME_STEPS + t
        Z = tl.randn(seed, random_offset)
        current_lnS = current_lnS + drift + volatility_scaler * Z
        # Column-major store: element [path, t] is at t * num_paths + path.
        # For fixed t, path_offsets are contiguous → coalesced writes.
        store_ptr = St_ptr + t * num_paths + path_offsets
        tl.store(store_ptr, tl.exp(current_lnS), mask=mask)
