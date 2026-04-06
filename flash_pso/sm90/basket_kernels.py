import triton
import triton.language as tl
from flash_pso.config import get_basket_autotune_configs


@triton.jit
def _cholesky_step(corr_Z, L_ptr, assets_offs, j, Z_j,
                   BLOCK_SIZE_ASSETS: tl.constexpr, BLOCK_SIZE_PATHS: tl.constexpr, USE_FP16: tl.constexpr):
    L_col_j = tl.load(L_ptr + assets_offs * BLOCK_SIZE_ASSETS + j)
    if USE_FP16:
        corr_Z += (tl.expand_dims(L_col_j.to(tl.float16), 1) * tl.expand_dims(Z_j.to(tl.float16), 0)).to(tl.float32)
    else:
        corr_Z += tl.expand_dims(L_col_j, 1) * tl.expand_dims(Z_j, 0)
    return corr_Z


# ══════════════════════════════════════════════════════════════════════════════
# COLLAPSE PRECOMPUTE — SM90: TMA store
# ══════════════════════════════════════════════════════════════════════════════
@triton.jit
def mc_basket_collapse_kernel(
    St_ptr, S0_ptr, drift_ptr, vol_ptr, weights_ptr, L_ptr,
    r, dt, num_bw_paths, seed,
    BLOCK_SIZE: tl.constexpr, NUM_TIME_STEPS: tl.constexpr,
    BLOCK_SIZE_ASSETS: tl.constexpr, NUM_ASSETS: tl.constexpr,
    MC_OFFSET_PHILOX: tl.constexpr, BANDWIDTH_PATHS_START: tl.constexpr,
    TOTAL_NUM_PATHS: tl.constexpr, USE_ANTITHETIC: tl.constexpr, USE_FP16: tl.constexpr,
):
    pid = tl.program_id(0)
    assets_offs = tl.arange(0, BLOCK_SIZE_ASSETS)
    st_desc = tl.make_tensor_descriptor(base=St_ptr, shape=[NUM_TIME_STEPS, num_bw_paths], strides=[num_bw_paths, 1], block_shape=[1, BLOCK_SIZE])
    S0 = tl.load(S0_ptr + assets_offs); drift = tl.load(drift_ptr + assets_offs)
    vol = tl.load(vol_ptr + assets_offs); weights = tl.load(weights_ptr + assets_offs)
    LOG2E = 1.4426950408889634
    S0_exp = tl.expand_dims(S0, 1)
    current_lnS = tl.broadcast_to(tl.log(S0_exp) * LOG2E, [BLOCK_SIZE_ASSETS, BLOCK_SIZE])
    drift_exp = tl.expand_dims(drift, 1); vol_exp = tl.expand_dims(vol, 1); weights_exp = tl.expand_dims(weights, 1)
    actual_path_idx = BANDWIDTH_PATHS_START + pid * BLOCK_SIZE
    if USE_ANTITHETIC:
        first_offs = (actual_path_idx + tl.arange(0, BLOCK_SIZE // 2)).to(tl.int64)
    else:
        actual_path_offs = (actual_path_idx + tl.arange(0, BLOCK_SIZE)).to(tl.int64)
    for step_idx in range(NUM_TIME_STEPS):
        corr_Z = tl.zeros([BLOCK_SIZE_ASSETS, BLOCK_SIZE], dtype=tl.float32)
        for j in tl.static_range(NUM_ASSETS):
            if USE_ANTITHETIC:
                rng_j = MC_OFFSET_PHILOX + j * NUM_TIME_STEPS * TOTAL_NUM_PATHS + first_offs * NUM_TIME_STEPS + step_idx
                Z_j_half = tl.randn(seed, rng_j)
                Z_j = tl.ravel(tl.join(Z_j_half, -Z_j_half))
            else:
                Z_j = tl.randn(seed, MC_OFFSET_PHILOX + j * NUM_TIME_STEPS * TOTAL_NUM_PATHS + actual_path_offs * NUM_TIME_STEPS + step_idx)
            corr_Z = _cholesky_step(corr_Z, L_ptr, assets_offs, j, Z_j, BLOCK_SIZE_ASSETS, BLOCK_SIZE, USE_FP16)
        current_lnS = current_lnS + drift_exp + vol_exp * corr_Z
        basket_S = tl.sum(tl.exp2(current_lnS) * weights_exp, axis=0)
        tl.store_tensor_descriptor(st_desc, [step_idx, pid * BLOCK_SIZE], tl.expand_dims(tl.log2(basket_S), 0))


# ══════════════════════════════════════════════════════════════════════════════
# PER-ASSET PRECOMPUTE — SM90: pointer stores (3D layout)
# ══════════════════════════════════════════════════════════════════════════════
@triton.jit
def mc_basket_path_kernel(
    st_ptr, S0_ptr, drift_ptr, vol_ptr, L_ptr,
    r, dt, num_bw_paths, seed,
    BLOCK_SIZE: tl.constexpr, NUM_TIME_STEPS: tl.constexpr,
    BLOCK_SIZE_ASSETS: tl.constexpr, NUM_ASSETS: tl.constexpr,
    MC_OFFSET_PHILOX: tl.constexpr, BANDWIDTH_PATHS_START: tl.constexpr,
    TOTAL_NUM_PATHS: tl.constexpr, USE_ANTITHETIC: tl.constexpr, USE_FP16: tl.constexpr,
):
    pid = tl.program_id(0)
    path_offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    assets_offs = tl.arange(0, BLOCK_SIZE_ASSETS)
    S0 = tl.load(S0_ptr + assets_offs); drift = tl.load(drift_ptr + assets_offs); vol = tl.load(vol_ptr + assets_offs)
    LOG2E = 1.4426950408889634
    S0_exp = tl.expand_dims(S0, 1)
    current_lnS = tl.broadcast_to(tl.log(S0_exp) * LOG2E, [BLOCK_SIZE_ASSETS, BLOCK_SIZE])
    drift_exp = tl.expand_dims(drift, 1); vol_exp = tl.expand_dims(vol, 1)
    actual_path_idx = BANDWIDTH_PATHS_START + pid * BLOCK_SIZE
    if USE_ANTITHETIC:
        first_offs = (actual_path_idx + tl.arange(0, BLOCK_SIZE // 2)).to(tl.int64)
    else:
        actual_path_offs = (actual_path_idx + tl.arange(0, BLOCK_SIZE)).to(tl.int64)
    for step_idx in range(NUM_TIME_STEPS):
        corr_Z = tl.zeros([BLOCK_SIZE_ASSETS, BLOCK_SIZE], dtype=tl.float32)
        for j in tl.static_range(NUM_ASSETS):
            if USE_ANTITHETIC:
                rng_j = MC_OFFSET_PHILOX + j * NUM_TIME_STEPS * TOTAL_NUM_PATHS + first_offs * NUM_TIME_STEPS + step_idx
                Z_j_half = tl.randn(seed, rng_j)
                Z_j = tl.ravel(tl.join(Z_j_half, -Z_j_half))
            else:
                Z_j = tl.randn(seed, MC_OFFSET_PHILOX + j * NUM_TIME_STEPS * TOTAL_NUM_PATHS + actual_path_offs * NUM_TIME_STEPS + step_idx)
            corr_Z = _cholesky_step(corr_Z, L_ptr, assets_offs, j, Z_j, BLOCK_SIZE_ASSETS, BLOCK_SIZE, USE_FP16)
        current_lnS = current_lnS + drift_exp + vol_exp * corr_Z
        st_ptrs = st_ptr + step_idx * (BLOCK_SIZE_ASSETS * num_bw_paths) + tl.expand_dims(assets_offs, 1) * num_bw_paths + tl.expand_dims(path_offs, 0)
        tl.store(st_ptrs, current_lnS)


# ══════════════════════════════════════════════════════════════════════════════
# UNIFIED BASKET PAYOFF — SM90 (TMA for scalar ln_positions)
# ══════════════════════════════════════════════════════════════════════════════
@triton.autotune(configs=get_basket_autotune_configs(), key=["NUM_PARTICLES", "NUM_PATHS", "NUM_DIMENSIONS", "NUM_COMPUTE_PATH_BLOCKS", "NUM_ASSETS", "EXERCISE_STYLE"])
@triton.jit
def mc_basket_payoff_kernel(
    ln_positions_ptr, st_ptr, partial_payoffs_ptr,
    S0_ptr, drift_ptr, vol_ptr, weights_ptr, L_ptr, r, dt,
    SEED: tl.constexpr, NUM_DIMENSIONS: tl.constexpr, NUM_PARTICLES: tl.constexpr, NUM_PATHS: tl.constexpr,
    NUM_COMPUTE_PATH_BLOCKS: tl.constexpr, BLOCK_SIZE_PARTICLES: tl.constexpr, BLOCK_SIZE_PATHS: tl.constexpr,
    BLOCK_SIZE_DIM: tl.constexpr, BLOCK_SIZE_ASSETS: tl.constexpr, NUM_ASSETS: tl.constexpr,
    OPTION_TYPE: tl.constexpr, STRIKE_PRICE: tl.constexpr, MC_OFFSET_PHILOX: tl.constexpr,
    USE_ANTITHETIC: tl.constexpr, EXERCISE_STYLE: tl.constexpr, USE_FP16: tl.constexpr,
    WARP_SPECIALIZE: tl.constexpr, LOOP_FLATTEN: tl.constexpr, LOOP_STAGES: tl.constexpr, LOOP_UNROLL: tl.constexpr,
):
    tl.assume(NUM_DIMENSIONS % BLOCK_SIZE_DIM == 0)
    GROUP_SIZE_M = 64
    pid_m, pid_n = tl.swizzle2d(tl.program_id(0), tl.program_id(1), tl.num_programs(0), tl.num_programs(1), GROUP_SIZE_M)
    pid = pid_m; path_block_idx = pid_n
    NUM_BW_PATHS = NUM_PATHS - NUM_COMPUTE_PATH_BLOCKS * BLOCK_SIZE_PATHS
    is_compute = path_block_idx < NUM_COMPUTE_PATH_BLOCKS
    bw_block_idx = path_block_idx - NUM_COMPUTE_PATH_BLOCKS
    bw_path_offs = (bw_block_idx * BLOCK_SIZE_PATHS + tl.arange(0, BLOCK_SIZE_PATHS)).to(tl.int64)
    p_idx = tl.max_contiguous(tl.multiple_of(pid * BLOCK_SIZE_PARTICLES + tl.arange(0, BLOCK_SIZE_PARTICLES), BLOCK_SIZE_PARTICLES), BLOCK_SIZE_PARTICLES)
    path_offs = path_block_idx * BLOCK_SIZE_PATHS + tl.arange(0, BLOCK_SIZE_PATHS)
    assets_offs = tl.arange(0, BLOCK_SIZE_ASSETS)

    # SM90 TMA for scalar ln_positions
    if EXERCISE_STYLE == 0 and BLOCK_SIZE_PARTICLES >= 4:
        ln_pos_desc = tl.make_tensor_descriptor(base=ln_positions_ptr, shape=[NUM_DIMENSIONS, NUM_PARTICLES], strides=[NUM_PARTICLES, 1], block_shape=[1, BLOCK_SIZE_PARTICLES])

    S0 = tl.load(S0_ptr + assets_offs); drift = tl.load(drift_ptr + assets_offs)
    vol = tl.load(vol_ptr + assets_offs); weights = tl.load(weights_ptr + assets_offs)
    LOG2E = 1.4426950408889634
    S0_exp = tl.expand_dims(S0, 1)
    current_lnS = tl.broadcast_to(tl.log(S0_exp) * LOG2E, [BLOCK_SIZE_ASSETS, BLOCK_SIZE_PATHS])
    drift_exp = tl.expand_dims(drift, 1); vol_exp = tl.expand_dims(vol, 1); weights_exp = tl.expand_dims(weights, 1)

    payoff_accum = tl.zeros([BLOCK_SIZE_PARTICLES], dtype=tl.float32)
    if BLOCK_SIZE_PARTICLES == 1:
        done_acc = tl.zeros([BLOCK_SIZE_PATHS], dtype=tl.int1)
        ex_S_acc = tl.zeros([BLOCK_SIZE_PATHS], dtype=tl.float32)
        ex_step_acc = tl.full([BLOCK_SIZE_PATHS], -1, dtype=tl.int32)
    else:
        done_acc = tl.zeros([BLOCK_SIZE_PATHS, BLOCK_SIZE_PARTICLES], dtype=tl.int1)
        ex_S_acc = tl.zeros([BLOCK_SIZE_PATHS, BLOCK_SIZE_PARTICLES], dtype=tl.float32)
        ex_step_acc = tl.full([BLOCK_SIZE_PATHS, BLOCK_SIZE_PARTICLES], -1, dtype=tl.int32)
    basket_S = tl.zeros([BLOCK_SIZE_PATHS], dtype=tl.float32)
    basket_lnS = tl.zeros([BLOCK_SIZE_PATHS], dtype=tl.float32)

    if USE_ANTITHETIC:
        first_offs = (path_block_idx * BLOCK_SIZE_PATHS + tl.arange(0, BLOCK_SIZE_PATHS // 2)).to(tl.int64)
    else:
        path_offs_i64 = path_offs.to(tl.int64)

    for dim_block_idx in tl.range(NUM_DIMENSIONS // BLOCK_SIZE_DIM, num_stages=LOOP_STAGES, loop_unroll_factor=LOOP_UNROLL, warp_specialize=WARP_SPECIALIZE, flatten=LOOP_FLATTEN):
        dim_offset = dim_block_idx * BLOCK_SIZE_DIM
        for d in tl.static_range(BLOCK_SIZE_DIM):
            step_idx = dim_offset + d
            if NUM_BW_PATHS > 0:
                if is_compute:
                    corr_Z = tl.zeros([BLOCK_SIZE_ASSETS, BLOCK_SIZE_PATHS], dtype=tl.float32)
                    for j in tl.static_range(NUM_ASSETS):
                        if USE_ANTITHETIC:
                            rng_j = MC_OFFSET_PHILOX + j * NUM_DIMENSIONS * NUM_PATHS + first_offs * NUM_DIMENSIONS + step_idx
                            Z_j_half = tl.randn(SEED, rng_j)
                            Z_j = tl.ravel(tl.join(Z_j_half, -Z_j_half))
                        else:
                            Z_j = tl.randn(SEED, MC_OFFSET_PHILOX + j * NUM_DIMENSIONS * NUM_PATHS + path_offs_i64 * NUM_DIMENSIONS + step_idx)
                        corr_Z = _cholesky_step(corr_Z, L_ptr, assets_offs, j, Z_j, BLOCK_SIZE_ASSETS, BLOCK_SIZE_PATHS, USE_FP16)
                    current_lnS = current_lnS + drift_exp + vol_exp * corr_Z
                    S_matrix = tl.exp2(current_lnS); basket_S = tl.sum(S_matrix * weights_exp, axis=0); basket_lnS = tl.log2(basket_S)
                else:
                    if EXERCISE_STYLE == 0:
                        basket_lnS = tl.load(st_ptr + step_idx * NUM_BW_PATHS + bw_path_offs)
                        basket_S = tl.exp2(basket_lnS)
                    else:
                        st_ptrs = st_ptr + step_idx * (BLOCK_SIZE_ASSETS * NUM_BW_PATHS) + tl.expand_dims(assets_offs, 1) * NUM_BW_PATHS + tl.expand_dims(bw_path_offs, 0)
                        current_lnS = tl.load(st_ptrs)
                        S_matrix = tl.exp2(current_lnS); basket_S = tl.sum(S_matrix * weights_exp, axis=0); basket_lnS = tl.log2(basket_S)
            else:
                corr_Z = tl.zeros([BLOCK_SIZE_ASSETS, BLOCK_SIZE_PATHS], dtype=tl.float32)
                for j in tl.static_range(NUM_ASSETS):
                    if USE_ANTITHETIC:
                        rng_j = MC_OFFSET_PHILOX + j * NUM_DIMENSIONS * NUM_PATHS + first_offs * NUM_DIMENSIONS + step_idx
                        Z_j_half = tl.randn(SEED, rng_j)
                        Z_j = tl.ravel(tl.join(Z_j_half, -Z_j_half))
                    else:
                        Z_j = tl.randn(SEED, MC_OFFSET_PHILOX + j * NUM_DIMENSIONS * NUM_PATHS + path_offs_i64 * NUM_DIMENSIONS + step_idx)
                    corr_Z = _cholesky_step(corr_Z, L_ptr, assets_offs, j, Z_j, BLOCK_SIZE_ASSETS, BLOCK_SIZE_PATHS, USE_FP16)
                current_lnS = current_lnS + drift_exp + vol_exp * corr_Z
                S_matrix = tl.exp2(current_lnS); basket_S = tl.sum(S_matrix * weights_exp, axis=0); basket_lnS = tl.log2(basket_S)

            # ── Exercise check ───────────────────────────────────────
            if EXERCISE_STYLE == 0:
                if BLOCK_SIZE_PARTICLES == 1:
                    ln_pos_slice = tl.load(ln_positions_ptr + step_idx * NUM_PARTICLES + p_idx)
                    any_ex = (basket_lnS < ln_pos_slice) if OPTION_TYPE == 1 else (basket_lnS > ln_pos_slice)
                    just_ex = any_ex & ~done_acc
                    ex_S_acc = tl.where(just_ex, basket_S, ex_S_acc); ex_step_acc = tl.where(just_ex, step_idx, ex_step_acc); done_acc = done_acc | any_ex
                else:
                    if BLOCK_SIZE_PARTICLES >= 4:
                        ln_pos_tile = tl.load_tensor_descriptor(ln_pos_desc, [step_idx, pid * BLOCK_SIZE_PARTICLES])
                        pos_2d = tl.broadcast_to(ln_pos_tile, [BLOCK_SIZE_PATHS, BLOCK_SIZE_PARTICLES])
                    else:
                        ln_pos_slice = tl.load(ln_positions_ptr + step_idx * NUM_PARTICLES + p_idx)
                        pos_2d = tl.broadcast_to(tl.expand_dims(ln_pos_slice, 0), [BLOCK_SIZE_PATHS, BLOCK_SIZE_PARTICLES])
                    lnS_2d = tl.broadcast_to(tl.expand_dims(basket_lnS, 1), [BLOCK_SIZE_PATHS, BLOCK_SIZE_PARTICLES])
                    S_2d = tl.broadcast_to(tl.expand_dims(basket_S, 1), [BLOCK_SIZE_PATHS, BLOCK_SIZE_PARTICLES])
                    any_ex = (pos_2d > lnS_2d) if OPTION_TYPE == 1 else (pos_2d < lnS_2d)
                    just_ex = any_ex & ~done_acc
                    ex_S_acc = tl.where(just_ex, S_2d, ex_S_acc); ex_step_acc = tl.where(just_ex, step_idx, ex_step_acc); done_acc = done_acc | any_ex
            else:
                if BLOCK_SIZE_PARTICLES == 1:
                    all_crossed = tl.full([BLOCK_SIZE_PATHS], 1, dtype=tl.int1)
                    for a in tl.static_range(NUM_ASSETS):
                        ln_bound_a = tl.load(ln_positions_ptr + (step_idx * NUM_ASSETS + a) * NUM_PARTICLES + p_idx)
                        a_sel = (assets_offs == a).to(tl.float32)
                        a_sel_2d = tl.broadcast_to(tl.expand_dims(a_sel, 1), [BLOCK_SIZE_ASSETS, BLOCK_SIZE_PATHS])
                        lnS_a = tl.sum(current_lnS * a_sel_2d, axis=0)
                        crossed = (lnS_a < ln_bound_a) if OPTION_TYPE == 1 else (lnS_a > ln_bound_a)
                        all_crossed = all_crossed & crossed
                    just_ex = all_crossed & ~done_acc
                    ex_S_acc = tl.where(just_ex, basket_S, ex_S_acc); ex_step_acc = tl.where(just_ex, step_idx, ex_step_acc); done_acc = done_acc | all_crossed
                else:
                    all_crossed = tl.full([BLOCK_SIZE_PATHS, BLOCK_SIZE_PARTICLES], 1, dtype=tl.int1)
                    for a in tl.static_range(NUM_ASSETS):
                        ln_bound_a = tl.load(ln_positions_ptr + (step_idx * NUM_ASSETS + a) * NUM_PARTICLES + p_idx)
                        a_sel = (assets_offs == a).to(tl.float32)
                        a_sel_2d = tl.broadcast_to(tl.expand_dims(a_sel, 1), [BLOCK_SIZE_ASSETS, BLOCK_SIZE_PATHS])
                        lnS_a = tl.sum(current_lnS * a_sel_2d, axis=0)
                        lnS_a_2d = tl.broadcast_to(tl.expand_dims(lnS_a, 1), [BLOCK_SIZE_PATHS, BLOCK_SIZE_PARTICLES])
                        bound_2d = tl.broadcast_to(tl.expand_dims(ln_bound_a, 0), [BLOCK_SIZE_PATHS, BLOCK_SIZE_PARTICLES])
                        crossed = (lnS_a_2d < bound_2d) if OPTION_TYPE == 1 else (lnS_a_2d > bound_2d)
                        all_crossed = all_crossed & crossed
                    S_2d = tl.broadcast_to(tl.expand_dims(basket_S, 1), [BLOCK_SIZE_PATHS, BLOCK_SIZE_PARTICLES])
                    just_ex = all_crossed & ~done_acc
                    ex_S_acc = tl.where(just_ex, S_2d, ex_S_acc); ex_step_acc = tl.where(just_ex, step_idx, ex_step_acc); done_acc = done_acc | all_crossed

    terminal_S_acc = basket_S
    r_dt_l2 = -r * dt * LOG2E; terminal_discount = tl.exp2(r_dt_l2 * NUM_DIMENSIONS)
    ex_disc_acc = tl.exp2((ex_step_acc.to(tl.float32) + 1.0) * r_dt_l2)
    if BLOCK_SIZE_PARTICLES == 1:
        final_S = tl.where(done_acc, ex_S_acc, terminal_S_acc); final_disc = tl.where(done_acc, ex_disc_acc, terminal_discount)
        payoff = tl.maximum(0.0, STRIKE_PRICE - final_S) if OPTION_TYPE == 1 else tl.maximum(0.0, final_S - STRIKE_PRICE)
        payoff_accum = payoff_accum + tl.sum(payoff * final_disc, axis=0, keep_dims=True)
    else:
        terminal_S_2d = tl.broadcast_to(tl.expand_dims(terminal_S_acc, 1), [BLOCK_SIZE_PATHS, BLOCK_SIZE_PARTICLES])
        final_S = tl.where(done_acc, ex_S_acc, terminal_S_2d); final_disc = tl.where(done_acc, ex_disc_acc, terminal_discount)
        payoff = tl.maximum(0.0, STRIKE_PRICE - final_S) if OPTION_TYPE == 1 else tl.maximum(0.0, final_S - STRIKE_PRICE)
        payoff_accum = payoff_accum + tl.sum(payoff * final_disc, axis=0)
    tl.store(partial_payoffs_ptr + path_block_idx * NUM_PARTICLES + p_idx, payoff_accum)
