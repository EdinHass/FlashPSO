import triton
import triton.language as tl

@triton.jit
def mc_path_kernel_col_stores(
    St_ptr,
    S0, r, sigma, dt,
    num_paths, seed,
    BLOCK_SIZE:              tl.constexpr,
    NUM_TIME_STEPS:          tl.constexpr,
    MC_OFFSET_PHILOX:        tl.constexpr,
    BANDWIDTH_PATHS_START:   tl.constexpr,
    USE_ANTITHETIC:          tl.constexpr,
):
    pid = tl.program_id(0)
    st_desc = tl.make_tensor_descriptor(
        base=St_ptr,
        shape=[NUM_TIME_STEPS, num_paths],
        strides=[num_paths, 1],
        block_shape=[1, BLOCK_SIZE],
    )
    LOG2E    = 1.4426950408889634
    drift_l2 = (r - 0.5 * sigma * sigma) * dt * LOG2E
    vol_l2   = sigma * tl.sqrt(dt) * LOG2E
    lnS      = tl.full([BLOCK_SIZE], tl.log(S0) * LOG2E, dtype=tl.float32)

    if USE_ANTITHETIC:
        first_offs = (BANDWIDTH_PATHS_START + pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE // 2)).to(tl.int64)

        for t in range(NUM_TIME_STEPS):
            Z_half = tl.randn(seed, MC_OFFSET_PHILOX + first_offs * NUM_TIME_STEPS + t)
            Z = tl.ravel(tl.join(Z_half, -Z_half))

            lnS = lnS + drift_l2 + vol_l2 * Z
            tl.store_tensor_descriptor(st_desc, [t, pid * BLOCK_SIZE], tl.expand_dims(lnS, 0))
    else:
        path_offsets = (BANDWIDTH_PATHS_START + pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)).to(tl.int64)

        for t in range(NUM_TIME_STEPS):
            Z   = tl.randn(seed, MC_OFFSET_PHILOX + path_offsets * NUM_TIME_STEPS + t)
            lnS = lnS + drift_l2 + vol_l2 * Z

            tl.store_tensor_descriptor(st_desc, [t, pid * BLOCK_SIZE], tl.expand_dims(lnS, 0))
