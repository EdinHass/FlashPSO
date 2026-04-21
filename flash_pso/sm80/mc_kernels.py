"""Monte Carlo path generation kernels (SM80 — pointer stores, no TMA).

All path precomputation lives here:
  - mc_path_kernel:            Vanilla/Asian 1D GBM paths.
  - mc_basket_collapse_kernel: Collapsed basket_lnS for ExerciseStyle.SCALAR.
  - mc_basket_path_kernel:     Per-asset lnS for ExerciseStyle.PER_ASSET.

RNG offset formula (consistent with payoff kernels):
  Vanilla:  base + path * steps + step
  Basket:   base + asset * (steps * total_paths) + step * total_paths + path
"""
import triton
import triton.language as tl


# ── Vanilla / Asian 1D path precompute ───────────────────────────────────────

@triton.jit
def mc_path_kernel(
    St_ptr, Z_ptr,
    log2_S0, drift_l2, vol_l2,
    num_paths, seed, mc_offset_philox,
    bandwidth_paths_start,
    BLOCK_SIZE: tl.constexpr,
    NUM_TIME_STEPS: tl.constexpr,
    USE_ANTITHETIC: tl.constexpr,
    USE_PRECOMPUTED_Z: tl.constexpr,
):
    pid = tl.program_id(0)
    path_idx = (pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)).to(tl.int64)
    mask = path_idx < num_paths

    lnS = tl.full([BLOCK_SIZE], log2_S0, dtype=tl.float32)

    if USE_ANTITHETIC:
        first_offs = (bandwidth_paths_start + pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE // 2)).to(tl.int64)
    else:
        path_offsets = (bandwidth_paths_start + path_idx).to(tl.int64)

    for t in tl.range(NUM_TIME_STEPS):
        if USE_PRECOMPUTED_Z:
            Z = tl.load(Z_ptr + t * num_paths + path_idx, mask=mask)
        elif USE_ANTITHETIC:
            Z_half = tl.randn(seed, mc_offset_philox + first_offs * NUM_TIME_STEPS + t)
            Z = tl.ravel(tl.join(Z_half, -Z_half))
        else:
            Z = tl.randn(seed, mc_offset_philox + path_offsets * NUM_TIME_STEPS + t)

        lnS = lnS + drift_l2 + vol_l2 * Z
        tl.store(St_ptr + t * num_paths + path_idx, lnS, mask=mask)


# ── Basket collapsed precompute (ExerciseStyle.SCALAR) ───────────────────────

@triton.jit
def mc_basket_collapse_kernel(
    St_ptr, Z_ptr, log2_S0_ptr, drift_ptr, vol_ptr, weights_ptr, L_ptr,
    num_bw_paths, seed, mc_offset_philox, bandwidth_paths_start,
    BLOCK_SIZE: tl.constexpr, NUM_TIME_STEPS: tl.constexpr,
    NUM_ASSETS: tl.constexpr, TOTAL_NUM_PATHS: tl.constexpr,
    USE_ANTITHETIC: tl.constexpr, USE_FP16: tl.constexpr,
    USE_PRECOMPUTED_Z: tl.constexpr,
):
    pid = tl.program_id(0)
    path_offs = (pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)).to(tl.int64)
    assets_offs = tl.arange(0, NUM_ASSETS)

    log2_S0 = tl.load(log2_S0_ptr + assets_offs)
    drift = tl.load(drift_ptr + assets_offs)
    vol = tl.load(vol_ptr + assets_offs)
    weights = tl.load(weights_ptr + assets_offs)

    current_lnS = tl.broadcast_to(tl.expand_dims(log2_S0, 1), [NUM_ASSETS, BLOCK_SIZE])
    drift_exp = tl.expand_dims(drift, 1)
    vol_exp = tl.expand_dims(vol, 1)
    weights_exp = tl.expand_dims(weights, 1)

    actual_path_idx = bandwidth_paths_start + pid * BLOCK_SIZE
    if USE_ANTITHETIC:
        first_offs = (actual_path_idx + tl.arange(0, BLOCK_SIZE // 2)).to(tl.int64)
    else:
        actual_path_offs = (actual_path_idx + tl.arange(0, BLOCK_SIZE)).to(tl.int64)

    if NUM_ASSETS >= 16:
        L_r = assets_offs[:, None]
        L_c = tl.arange(0, NUM_ASSETS)[None, :]
        L_matrix = tl.load(L_ptr + L_r * NUM_ASSETS + L_c)

    for step_idx in range(NUM_TIME_STEPS):
        if NUM_ASSETS >= 16:
            if USE_PRECOMPUTED_Z:
                path_offs_local = tl.arange(0, BLOCK_SIZE)
                z_offs = (assets_offs[:, None] * NUM_TIME_STEPS + step_idx) * TOTAL_NUM_PATHS + actual_path_idx + path_offs_local[None, :]
                Z_raw = tl.load(Z_ptr + z_offs)
            elif USE_ANTITHETIC:
                rng_r = assets_offs[:, None]
                rng_c = first_offs[None, :]
                rng_idx = mc_offset_philox + rng_r * NUM_TIME_STEPS * TOTAL_NUM_PATHS + step_idx * TOTAL_NUM_PATHS + rng_c
                Z_half = tl.randn(seed, rng_idx)
                Z_raw = tl.reshape(tl.join(Z_half, -Z_half), [NUM_ASSETS, BLOCK_SIZE])
            else:
                rng_r = assets_offs[:, None]
                rng_c = actual_path_offs[None, :]
                rng_idx = mc_offset_philox + rng_r * NUM_TIME_STEPS * TOTAL_NUM_PATHS + step_idx * TOTAL_NUM_PATHS + rng_c
                Z_raw = tl.randn(seed, rng_idx)

            if USE_FP16:
                corr_Z = tl.dot(L_matrix.to(tl.float16), Z_raw.to(tl.float16), out_dtype=tl.float32)
            else:
                corr_Z = tl.dot(L_matrix, Z_raw)
        else:
            corr_Z = tl.zeros([NUM_ASSETS, BLOCK_SIZE], dtype=tl.float32)
            for j in tl.static_range(NUM_ASSETS):
                L_col = tl.load(L_ptr + assets_offs * NUM_ASSETS + j)
                if USE_PRECOMPUTED_Z:
                    path_offs_local = tl.arange(0, BLOCK_SIZE)
                    z_offs = (j * NUM_TIME_STEPS + step_idx) * TOTAL_NUM_PATHS + actual_path_idx + path_offs_local
                    Z_j = tl.load(Z_ptr + z_offs)
                elif USE_ANTITHETIC:
                    rng_idx = mc_offset_philox + j * NUM_TIME_STEPS * TOTAL_NUM_PATHS + step_idx * TOTAL_NUM_PATHS + first_offs
                    Z_half = tl.randn(seed, rng_idx)
                    Z_j = tl.ravel(tl.join(Z_half, -Z_half))
                else:
                    rng_idx = mc_offset_philox + j * NUM_TIME_STEPS * TOTAL_NUM_PATHS + step_idx * TOTAL_NUM_PATHS + actual_path_offs
                    Z_j = tl.randn(seed, rng_idx)
                corr_Z += L_col[:, None] * Z_j[None, :]

        current_lnS = current_lnS + drift_exp + vol_exp * corr_Z
        S_matrix = tl.exp2(current_lnS)
        basket_S = tl.sum(S_matrix * weights_exp, axis=0)
        tl.store(St_ptr + step_idx * num_bw_paths + path_offs, tl.log2(basket_S))


# ── Basket per-asset precompute (ExerciseStyle.PER_ASSET) ────────────────────

@triton.jit
def mc_basket_path_kernel(
    st_ptr, Z_ptr, log2_S0_ptr, drift_ptr, vol_ptr, L_ptr,
    num_bw_paths, seed, mc_offset_philox,
    bandwidth_paths_start,
    BLOCK_SIZE: tl.constexpr, NUM_TIME_STEPS: tl.constexpr,
    NUM_ASSETS: tl.constexpr, TOTAL_NUM_PATHS: tl.constexpr,
    USE_ANTITHETIC: tl.constexpr, USE_FP16: tl.constexpr,
    USE_PRECOMPUTED_Z: tl.constexpr,
):
    pid = tl.program_id(0)
    path_offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    assets_offs = tl.arange(0, NUM_ASSETS)

    log2_S0 = tl.load(log2_S0_ptr + assets_offs)
    drift = tl.load(drift_ptr + assets_offs)
    vol = tl.load(vol_ptr + assets_offs)

    current_lnS = tl.broadcast_to(tl.expand_dims(log2_S0, 1), [NUM_ASSETS, BLOCK_SIZE])
    drift_exp = tl.expand_dims(drift, 1)
    vol_exp = tl.expand_dims(vol, 1)

    actual_path_idx = bandwidth_paths_start + pid * BLOCK_SIZE
    if USE_ANTITHETIC:
        first_offs = (actual_path_idx + tl.arange(0, BLOCK_SIZE // 2)).to(tl.int64)
    else:
        actual_path_offs = (actual_path_idx + tl.arange(0, BLOCK_SIZE)).to(tl.int64)

    if NUM_ASSETS >= 16:
        L_r = assets_offs[:, None]
        L_c = tl.arange(0, NUM_ASSETS)[None, :]
        L_matrix = tl.load(L_ptr + L_r * NUM_ASSETS + L_c)

    for step_idx in range(NUM_TIME_STEPS):
        if NUM_ASSETS >= 16:
            if USE_PRECOMPUTED_Z:
                path_offs_local = tl.arange(0, BLOCK_SIZE)
                z_offs = (assets_offs[:, None] * NUM_TIME_STEPS + step_idx) * TOTAL_NUM_PATHS + actual_path_idx + path_offs_local[None, :]
                Z_raw = tl.load(Z_ptr + z_offs)
            elif USE_ANTITHETIC:
                rng_r = assets_offs[:, None]
                rng_c = first_offs[None, :]
                rng_idx = mc_offset_philox + rng_r * NUM_TIME_STEPS * TOTAL_NUM_PATHS + step_idx * TOTAL_NUM_PATHS + rng_c
                Z_half = tl.randn(seed, rng_idx)
                Z_raw = tl.reshape(tl.join(Z_half, -Z_half), [NUM_ASSETS, BLOCK_SIZE])
            else:
                rng_r = assets_offs[:, None]
                rng_c = actual_path_offs[None, :]
                rng_idx = mc_offset_philox + rng_r * NUM_TIME_STEPS * TOTAL_NUM_PATHS + step_idx * TOTAL_NUM_PATHS + rng_c
                Z_raw = tl.randn(seed, rng_idx)

            if USE_FP16:
                corr_Z = tl.dot(L_matrix.to(tl.float16), Z_raw.to(tl.float16), out_dtype=tl.float32)
            else:
                corr_Z = tl.dot(L_matrix, Z_raw)
        else:
            corr_Z = tl.zeros([NUM_ASSETS, BLOCK_SIZE], dtype=tl.float32)
            for j in tl.static_range(NUM_ASSETS):
                L_col = tl.load(L_ptr + assets_offs * NUM_ASSETS + j)
                if USE_PRECOMPUTED_Z:
                    path_offs_local = tl.arange(0, BLOCK_SIZE)
                    z_offs = (j * NUM_TIME_STEPS + step_idx) * TOTAL_NUM_PATHS + actual_path_idx + path_offs_local
                    Z_j = tl.load(Z_ptr + z_offs)
                elif USE_ANTITHETIC:
                    rng_idx = mc_offset_philox + j * NUM_TIME_STEPS * TOTAL_NUM_PATHS + step_idx * TOTAL_NUM_PATHS + first_offs
                    Z_half = tl.randn(seed, rng_idx)
                    Z_j = tl.ravel(tl.join(Z_half, -Z_half))
                else:
                    rng_idx = mc_offset_philox + j * NUM_TIME_STEPS * TOTAL_NUM_PATHS + step_idx * TOTAL_NUM_PATHS + actual_path_offs
                    Z_j = tl.randn(seed, rng_idx)
                corr_Z += L_col[:, None] * Z_j[None, :]

        current_lnS = current_lnS + drift_exp + vol_exp * corr_Z
        st_ptrs = (st_ptr + step_idx * (NUM_ASSETS * num_bw_paths)
                   + tl.expand_dims(assets_offs, 1) * num_bw_paths
                   + tl.expand_dims(path_offs, 0))
        tl.store(st_ptrs, current_lnS)
