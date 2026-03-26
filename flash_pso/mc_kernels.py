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
    pid          = tl.program_id(0)
    path_offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask         = path_offsets < num_paths
    drift        = (r - 0.5 * sigma * sigma) * dt
    vol          = sigma * tl.sqrt(dt)
    current_lnS  = tl.full([BLOCK_SIZE], tl.log(S0), dtype=tl.float32)
    for t in range(num_time_steps):
        Z           = tl.randn(seed, MC_OFFSET_PHILOX + path_offsets * num_time_steps + t)
        current_lnS = current_lnS + drift + vol * Z
        tl.store(St_ptr + t * num_paths + path_offsets, tl.exp(current_lnS), mask=mask)


@triton.jit
def mc_path_kernel_block_stores(
    St_ptr,
    S0, r, sigma, dt,
    num_paths, seed,
    BLOCK_SIZE:       tl.constexpr,
    NUM_TIME_STEPS:   tl.constexpr,
    MC_OFFSET_PHILOX: tl.constexpr,
):
    pid           = tl.program_id(0)
    path_offsets  = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    current_block = tl.make_tensor_descriptor(
        base=St_ptr,
        shape=[num_paths, NUM_TIME_STEPS],       # pyright: ignore[reportArgumentType]
        strides=[1, num_paths],                  # pyright: ignore[reportArgumentType]
        block_shape=[BLOCK_SIZE, NUM_TIME_STEPS],
    )
    drift       = (r - 0.5 * sigma * sigma) * dt
    vol         = sigma * tl.sqrt(dt)
    current_lnS = tl.full([BLOCK_SIZE], tl.log(S0), dtype=tl.float32)
    result      = tl.zeros([BLOCK_SIZE, NUM_TIME_STEPS], dtype=tl.float32)
    for t in range(NUM_TIME_STEPS):
        Z           = tl.randn(seed, MC_OFFSET_PHILOX + path_offsets * NUM_TIME_STEPS + t)
        current_lnS = current_lnS + drift + vol * Z
        col_mask    = tl.arange(0, NUM_TIME_STEPS)[None, :] == t
        result      = tl.where(col_mask, tl.exp(current_lnS)[:, None], result)
    tl.store_tensor_descriptor(current_block, [pid * BLOCK_SIZE, 0], result)


@triton.jit
def mc_path_kernel_col_stores_tma(
    St_ptr,
    S0, r, sigma, dt,
    num_paths, seed,
    BLOCK_SIZE:       tl.constexpr,
    NUM_TIME_STEPS:   tl.constexpr,
    MC_OFFSET_PHILOX: tl.constexpr,
    PATH_START:       tl.constexpr,
    USE_ANTITHETIC:   tl.constexpr,
):
    pid          = tl.program_id(0)
    path_offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # local indices in St
    mask         = path_offsets < num_paths
    drift        = (r - 0.5 * sigma * sigma) * dt
    vol          = sigma * tl.sqrt(dt)

    if USE_ANTITHETIC:
        # Only HALF paths need Philox — second half is derived by negation.
        # Each CTA writes BLOCK_SIZE paths but only pays for BLOCK_SIZE//2 randn calls.
        global_first_idx = PATH_START + pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE // 2)
        lnS_pos          = tl.full([BLOCK_SIZE // 2], tl.log(S0), dtype=tl.float32)
        lnS_neg          = tl.full([BLOCK_SIZE // 2], tl.log(S0), dtype=tl.float32)
        mask_first       = (pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE // 2)) < num_paths
        mask_second      = (pid * BLOCK_SIZE + BLOCK_SIZE // 2 + tl.arange(0, BLOCK_SIZE // 2)) < num_paths

        for t in range(NUM_TIME_STEPS):  # pyright: ignore
            Z       = tl.randn(seed, MC_OFFSET_PHILOX + global_first_idx * NUM_TIME_STEPS + t)
            lnS_pos = lnS_pos + drift + vol *  Z
            lnS_neg = lnS_neg + drift + vol * -Z
            tl.store(St_ptr + t * num_paths + pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE // 2),        tl.exp(lnS_pos), mask=mask_first)
            tl.store(St_ptr + t * num_paths + pid * BLOCK_SIZE + BLOCK_SIZE // 2 + tl.arange(0, BLOCK_SIZE // 2), tl.exp(lnS_neg), mask=mask_second)
    else:
        current_lnS = tl.full([BLOCK_SIZE], tl.log(S0), dtype=tl.float32)
        for t in range(NUM_TIME_STEPS):  # pyright: ignore
            rng_offs    = MC_OFFSET_PHILOX + (PATH_START + path_offsets) * NUM_TIME_STEPS + t
            Z           = tl.randn(seed, rng_offs)
            current_lnS = current_lnS + drift + vol * Z
            tl.store(St_ptr + t * num_paths + path_offsets, tl.exp(current_lnS), mask=mask)
