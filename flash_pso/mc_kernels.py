import triton
import triton.language as tl

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
    pid = tl.program_id(0)

    # ── TMA DESCRIPTOR ───────────────────────────────────────────────────────
    # Hardware Constraint: The contiguous dimension MUST be the last dimension.
    # St_ptr memory is (paths, time) with strides (1, paths).
    # We define it here transposed so TMA sees `num_paths` as the last (contiguous) dimension.
    st_desc = tl.make_tensor_descriptor(
        base=St_ptr,
        shape=[NUM_TIME_STEPS, num_paths],           # pyright: ignore[reportArgumentType]
        strides=[num_paths, 1],                      # pyright: ignore[reportArgumentType]
        block_shape=[1, BLOCK_SIZE],                 # Last dim >= 16 bytes (BLOCK_SIZE * 4)
    )

    # ── BASE-2 LOG CONSTANTS (Matches downstream deferred payoff logic) ──────
    LOG2E    = 1.4426950408889634
    drift_l2 = (r - 0.5 * sigma * sigma) * dt * LOG2E
    vol_l2   = sigma * tl.sqrt(dt) * LOG2E
    lnS      = tl.full([BLOCK_SIZE], tl.log(S0) * LOG2E, dtype=tl.float32)

    # ══════════════════════════════════════════════════════════════════════════
    # ── PATH GENERATION LOOP ──────────────────────────────────────────────────
    # ══════════════════════════════════════════════════════════════════════════
    
    if USE_ANTITHETIC:
        # Only require half the RNG offset IDs
        first_offs = PATH_START + pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE // 2)
        
        for t in range(NUM_TIME_STEPS): # pyright: ignore
            # Generate half the block
            Z_half = tl.randn(seed, MC_OFFSET_PHILOX + first_offs * NUM_TIME_STEPS + t)
            
            # Interleave to perfectly match the downstream on-the-fly generators:
            # [Z0, -Z0, Z1, -Z1, Z2, -Z2...]
            Z = tl.ravel(tl.join(Z_half, -Z_half))
            
            lnS = lnS + drift_l2 + vol_l2 * Z
            
            # Write 1D slice directly to the 2D global memory matrix
            # tl.expand_dims(lnS, 0) makes the shape [1, BLOCK_SIZE] to match block_shape
            tl.store_tensor_descriptor(st_desc, [t, pid * BLOCK_SIZE], tl.expand_dims(lnS, 0))
            
    else:
        path_offsets = PATH_START + pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        
        for t in range(NUM_TIME_STEPS): # pyright: ignore
            Z   = tl.randn(seed, MC_OFFSET_PHILOX + path_offsets * NUM_TIME_STEPS + t)
            lnS = lnS + drift_l2 + vol_l2 * Z
            
            # Write 1D slice directly to the 2D global memory matrix
            tl.store_tensor_descriptor(st_desc, [t, pid * BLOCK_SIZE], tl.expand_dims(lnS, 0))
