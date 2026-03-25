import triton
import triton.language as tl


@triton.jit
def mc_path_kernel_row_stores(
    St_ptr,
    S0, r, sigma, dt,
    num_paths, num_time_steps, seed,
    BLOCK_SIZE:       tl.constexpr,
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
        store_ptr = St_ptr + t * num_paths + path_offsets
        tl.store(store_ptr, tl.exp(current_lnS), mask=mask)


@triton.jit
def mc_path_kernel_block_stores(
    St_ptr,
    S0, r, sigma, dt,
    num_paths, seed,
    BLOCK_SIZE:       tl.constexpr,
    NUM_TIME_STEPS:   tl.constexpr,
    MC_OFFSET_PHILOX: tl.constexpr,
):
    pid = tl.program_id(0)
    path_offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    current_block = tl.make_tensor_descriptor(
        base=St_ptr,
        shape=[num_paths, NUM_TIME_STEPS],       # pyright: ignore[reportArgumentType]
        strides=[1, num_paths],                  # pyright: ignore[reportArgumentType]
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


@triton.jit
def mc_path_kernel_col_stores_tma(
    St_ptr,
    S0, r, sigma, dt,
    num_paths, seed,
    BLOCK_SIZE:       tl.constexpr,
    NUM_TIME_STEPS:   tl.constexpr,
    MC_OFFSET_PHILOX: tl.constexpr,
    # FIX: PATH_START is the global path index of the first path this kernel
    # generates. For hybrid mode the bandwidth paths start at NUM_COMPUTE_PATHS,
    # so their Philox keys are MC_OFFSET_PHILOX + (PATH_START + local_idx) * NUM_TIME_STEPS + t
    # — matching exactly what pso_kernel would produce on-the-fly for those indices.
    # For fully precomputed mode PATH_START=0 and behavior is unchanged.
    PATH_START:       tl.constexpr,
):
    pid = tl.program_id(0)
    # local path indices within St buffer [0, num_paths)
    path_offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = path_offsets < num_paths
    drift = (r - 0.5 * sigma * sigma) * dt
    volatility_scaler = sigma * tl.sqrt(dt)
    current_lnS = tl.full([BLOCK_SIZE], tl.log(S0), dtype=tl.float32)
    for t in tl.static_range(NUM_TIME_STEPS):  # pyright: ignore
        # FIX: Philox key uses global path index (PATH_START + local) so
        # bandwidth paths are statistically independent from compute paths
        # and consistent with what pso_kernel generates on-the-fly.
        random_offset = MC_OFFSET_PHILOX + (PATH_START + path_offsets) * NUM_TIME_STEPS + t
        Z = tl.randn(seed, random_offset)
        current_lnS = current_lnS + drift + volatility_scaler * Z
        # store into compact buffer — local index, not global
        store_ptr = St_ptr + t * num_paths + path_offsets
        tl.store(store_ptr, tl.exp(current_lnS), mask=mask)
