"""Payoff evaluation kernels (SM80 — pointer loads, no TMA).

All payoff/exercise logic lives here:
  - mc_payoff_kernel:        Vanilla American put/call.
  - mc_asian_payoff_kernel:  American Asian (average price).
  - mc_basket_payoff_kernel: American basket (scalar or per-asset exercise).
"""
import triton
import triton.language as tl
from flash_pso.config import get_autotune_configs, get_basket_autotune_configs


@triton.jit
def process_dim_block_compute_vanilla(
    current_lnS, done_acc, ex_lnS_acc, ex_step_acc,
    ln_positions_ptr, dim_offset, p_idx, NUM_PARTICLES: tl.constexpr,
    seed, base_rng_offsets, drift_l2, vol_l2,
    BLOCK_SIZE_DIM: tl.constexpr, BLOCK_SIZE_PATHS: tl.constexpr, BLOCK_SIZE_PARTICLES: tl.constexpr,
    OPTION_TYPE: tl.constexpr, USE_ANTITHETIC: tl.constexpr
):
    for d in tl.static_range(BLOCK_SIZE_DIM):
        step_idx = dim_offset + d
        ln_pos_slice = tl.load(ln_positions_ptr + step_idx * NUM_PARTICLES + p_idx)
        
        if USE_ANTITHETIC:
            Z_half = tl.randn(seed, base_rng_offsets + step_idx)
            Z = tl.ravel(tl.join(Z_half, -Z_half))
        else:
            Z = tl.randn(seed, base_rng_offsets + step_idx)
            
        current_lnS = current_lnS + drift_l2 + vol_l2 * Z
        
        if BLOCK_SIZE_PARTICLES == 1:
            any_ex = (current_lnS < ln_pos_slice) if OPTION_TYPE == 1 else (current_lnS > ln_pos_slice)
            just_ex = any_ex & ~done_acc
            ex_lnS_acc = tl.where(just_ex, current_lnS, ex_lnS_acc)
            ex_step_acc = tl.where(just_ex, step_idx, ex_step_acc)
            done_acc = done_acc | any_ex
        else:
            any_ex = (ln_pos_slice[None, :] > current_lnS[:, None]) if OPTION_TYPE == 1 else (ln_pos_slice[None, :] < current_lnS[:, None])
            just_ex = any_ex & ~done_acc
            ex_lnS_acc = tl.where(just_ex, current_lnS[:, None], ex_lnS_acc)
            ex_step_acc = tl.where(just_ex, step_idx, ex_step_acc)
            done_acc = done_acc | any_ex

    return current_lnS, done_acc, ex_lnS_acc, ex_step_acc

@triton.jit
def process_dim_block_bandwidth_vanilla(
    current_lnS, done_acc, ex_lnS_acc, ex_step_acc,
    ln_positions_ptr, st_ptr, dim_offset, p_idx, NUM_PARTICLES: tl.constexpr,
    NUM_BW_PATHS: tl.constexpr, bw_path_offs,
    BLOCK_SIZE_DIM: tl.constexpr, BLOCK_SIZE_PATHS: tl.constexpr, BLOCK_SIZE_PARTICLES: tl.constexpr,
    OPTION_TYPE: tl.constexpr
):
    for d in tl.static_range(BLOCK_SIZE_DIM):
        step_idx = dim_offset + d
        ln_pos_slice = tl.load(ln_positions_ptr + step_idx * NUM_PARTICLES + p_idx)
        current_lnS = tl.load(st_ptr + step_idx * NUM_BW_PATHS + bw_path_offs)
        
        if BLOCK_SIZE_PARTICLES == 1:
            any_ex = (current_lnS < ln_pos_slice) if OPTION_TYPE == 1 else (current_lnS > ln_pos_slice)
            just_ex = any_ex & ~done_acc
            ex_lnS_acc = tl.where(just_ex, current_lnS, ex_lnS_acc)
            ex_step_acc = tl.where(just_ex, step_idx, ex_step_acc)
            done_acc = done_acc | any_ex
        else:
            any_ex = (ln_pos_slice[None, :] > current_lnS[:, None]) if OPTION_TYPE == 1 else (ln_pos_slice[None, :] < current_lnS[:, None])
            just_ex = any_ex & ~done_acc
            ex_lnS_acc = tl.where(just_ex, current_lnS[:, None], ex_lnS_acc)
            ex_step_acc = tl.where(just_ex, step_idx, ex_step_acc)
            done_acc = done_acc | any_ex

    return current_lnS, done_acc, ex_lnS_acc, ex_step_acc


@triton.autotune(configs=get_autotune_configs(),
                 key=["NUM_PARTICLES", "NUM_PATHS", "NUM_DIMENSIONS", "NUM_COMPUTE_PATH_BLOCKS"],
                 warmup=2, rep=3)
@triton.jit
def mc_payoff_kernel(
    ln_positions_ptr, st_ptr, partial_payoffs_ptr,
    seed, mc_offset_philox, log2_S0, drift_l2, vol_l2, r_dt_l2, terminal_discount,
    NUM_DIMENSIONS: tl.constexpr, NUM_PARTICLES: tl.constexpr, NUM_PATHS: tl.constexpr,
    NUM_COMPUTE_PATH_BLOCKS: tl.constexpr,
    BLOCK_SIZE_PARTICLES: tl.constexpr, BLOCK_SIZE_PATHS: tl.constexpr, BLOCK_SIZE_DIM: tl.constexpr,
    OPTION_TYPE: tl.constexpr, STRIKE_PRICE: tl.constexpr, USE_ANTITHETIC: tl.constexpr,
    WARP_SPECIALIZE: tl.constexpr, LOOP_FLATTEN: tl.constexpr,
    LOOP_STAGES: tl.constexpr, LOOP_UNROLL: tl.constexpr,
):
    tl.assume(NUM_DIMENSIONS % BLOCK_SIZE_DIM == 0)

    GROUP_SIZE_M = 64
    pid_m, pid_n = tl.swizzle2d(tl.program_id(0), tl.program_id(1), tl.num_programs(0), tl.num_programs(1), GROUP_SIZE_M)
    path_block_idx = pid_m; pid = pid_n

    NUM_BW_PATHS = NUM_PATHS - NUM_COMPUTE_PATH_BLOCKS * BLOCK_SIZE_PATHS
    is_compute = path_block_idx < NUM_COMPUTE_PATH_BLOCKS

    p_idx = tl.max_contiguous(tl.multiple_of(pid * BLOCK_SIZE_PARTICLES + tl.arange(0, BLOCK_SIZE_PARTICLES), BLOCK_SIZE_PARTICLES), BLOCK_SIZE_PARTICLES)
    path_offs = tl.max_contiguous(tl.multiple_of(path_block_idx * BLOCK_SIZE_PATHS + tl.arange(0, BLOCK_SIZE_PATHS), BLOCK_SIZE_PATHS), BLOCK_SIZE_PATHS).to(tl.int64)

    bw_block_idx = path_block_idx - NUM_COMPUTE_PATH_BLOCKS
    bw_path_offs = (bw_block_idx * BLOCK_SIZE_PATHS + tl.arange(0, BLOCK_SIZE_PATHS)).to(tl.int64)

    payoff_accum = tl.zeros([BLOCK_SIZE_PARTICLES], dtype=tl.float32)
    if BLOCK_SIZE_PARTICLES == 1:
        done_acc = tl.zeros([BLOCK_SIZE_PATHS], dtype=tl.int1)
        ex_lnS_acc = tl.zeros([BLOCK_SIZE_PATHS], dtype=tl.float32)
        ex_step_acc = tl.full([BLOCK_SIZE_PATHS], -1, dtype=tl.int32)
    else:
        done_acc = tl.zeros([BLOCK_SIZE_PATHS, BLOCK_SIZE_PARTICLES], dtype=tl.int1)
        ex_lnS_acc = tl.zeros([BLOCK_SIZE_PATHS, BLOCK_SIZE_PARTICLES], dtype=tl.float32)
        ex_step_acc = tl.full([BLOCK_SIZE_PATHS, BLOCK_SIZE_PARTICLES], -1, dtype=tl.int32)

    current_lnS = tl.full([BLOCK_SIZE_PATHS], log2_S0, dtype=tl.float32)

    if USE_ANTITHETIC:
        base_path_idx = path_block_idx * BLOCK_SIZE_PATHS
        first_offs = (base_path_idx + tl.arange(0, BLOCK_SIZE_PATHS // 2)).to(tl.int64)
        base_rng_offsets = mc_offset_philox + first_offs * NUM_DIMENSIONS
    else:
        base_rng_offsets = mc_offset_philox + path_offs * NUM_DIMENSIONS

    for dim_block_idx in tl.range(NUM_DIMENSIONS // BLOCK_SIZE_DIM, num_stages=LOOP_STAGES, loop_unroll_factor=LOOP_UNROLL, warp_specialize=WARP_SPECIALIZE, flatten=LOOP_FLATTEN):
        dim_offset = dim_block_idx * BLOCK_SIZE_DIM

        if NUM_BW_PATHS == NUM_PATHS:
            current_lnS, done_acc, ex_lnS_acc, ex_step_acc = process_dim_block_bandwidth_vanilla(
                current_lnS, done_acc, ex_lnS_acc, ex_step_acc,
                ln_positions_ptr, st_ptr, dim_offset, p_idx, NUM_PARTICLES,
                NUM_BW_PATHS, bw_path_offs,
                BLOCK_SIZE_DIM, BLOCK_SIZE_PATHS, BLOCK_SIZE_PARTICLES, OPTION_TYPE
            )
        elif NUM_BW_PATHS > 0:
            if is_compute:
                current_lnS, done_acc, ex_lnS_acc, ex_step_acc = process_dim_block_compute_vanilla(
                    current_lnS, done_acc, ex_lnS_acc, ex_step_acc,
                    ln_positions_ptr, dim_offset, p_idx, NUM_PARTICLES,
                    seed, base_rng_offsets, drift_l2, vol_l2,
                    BLOCK_SIZE_DIM, BLOCK_SIZE_PATHS, BLOCK_SIZE_PARTICLES, OPTION_TYPE, USE_ANTITHETIC
                )
            else:
                current_lnS, done_acc, ex_lnS_acc, ex_step_acc = process_dim_block_bandwidth_vanilla(
                    current_lnS, done_acc, ex_lnS_acc, ex_step_acc,
                    ln_positions_ptr, st_ptr, dim_offset, p_idx, NUM_PARTICLES,
                    NUM_BW_PATHS, bw_path_offs,
                    BLOCK_SIZE_DIM, BLOCK_SIZE_PATHS, BLOCK_SIZE_PARTICLES, OPTION_TYPE
                )
        else:
            current_lnS, done_acc, ex_lnS_acc, ex_step_acc = process_dim_block_compute_vanilla(
                current_lnS, done_acc, ex_lnS_acc, ex_step_acc,
                ln_positions_ptr, dim_offset, p_idx, NUM_PARTICLES,
                seed, base_rng_offsets, drift_l2, vol_l2,
                BLOCK_SIZE_DIM, BLOCK_SIZE_PATHS, BLOCK_SIZE_PARTICLES, OPTION_TYPE, USE_ANTITHETIC
            )

    ex_disc_acc = tl.exp2((ex_step_acc.to(tl.float32) + 1.0) * r_dt_l2)
    
    if BLOCK_SIZE_PARTICLES == 1:
        final_lnS = tl.where(done_acc, ex_lnS_acc, current_lnS)
        final_disc = tl.where(done_acc, ex_disc_acc, terminal_discount)
        final_S = tl.exp2(final_lnS)
        payoff = tl.maximum(0.0, STRIKE_PRICE - final_S) if OPTION_TYPE == 1 else tl.maximum(0.0, final_S - STRIKE_PRICE)
        payoff_accum = payoff_accum + tl.sum(payoff * final_disc, axis=0, keep_dims=True)
    else:
        final_lnS = tl.where(done_acc, ex_lnS_acc, current_lnS[:, None])
        final_disc = tl.where(done_acc, ex_disc_acc, terminal_discount)
        final_S = tl.exp2(final_lnS)
        payoff = tl.maximum(0.0, STRIKE_PRICE - final_S) if OPTION_TYPE == 1 else tl.maximum(0.0, final_S - STRIKE_PRICE)
        payoff_accum = payoff_accum + tl.sum(payoff * final_disc, axis=0)
        
    tl.store(partial_payoffs_ptr + path_block_idx * NUM_PARTICLES + p_idx, payoff_accum)


@triton.jit
def process_dim_block_compute_asian(
    current_lnS, running_sum, done_acc, ex_avg_acc, ex_step_acc,
    ln_positions_ptr, dim_offset, p_idx, NUM_PARTICLES: tl.constexpr,
    seed, base_rng_offsets, drift_l2, vol_l2,
    BLOCK_SIZE_DIM: tl.constexpr, BLOCK_SIZE_PATHS: tl.constexpr, BLOCK_SIZE_PARTICLES: tl.constexpr,
    OPTION_TYPE: tl.constexpr, USE_ANTITHETIC: tl.constexpr
):
    for d in tl.static_range(BLOCK_SIZE_DIM):
        step_idx = dim_offset + d
        ln_pos_slice = tl.load(ln_positions_ptr + step_idx * NUM_PARTICLES + p_idx)
        
        if USE_ANTITHETIC:
            Z_half = tl.randn(seed, base_rng_offsets + step_idx)
            Z = tl.ravel(tl.join(Z_half, -Z_half))
        else:
            Z = tl.randn(seed, base_rng_offsets + step_idx)
            
        current_lnS = current_lnS + drift_l2 + vol_l2 * Z
        step_S = tl.exp2(current_lnS)
        running_sum = running_sum + step_S
        step_avg = running_sum / (step_idx + 1.0)
        step_ln_avg = tl.log2(step_avg)
        
        if BLOCK_SIZE_PARTICLES == 1:
            any_ex = (step_ln_avg < ln_pos_slice) if OPTION_TYPE == 1 else (step_ln_avg > ln_pos_slice)
            just_ex = any_ex & ~done_acc
            ex_avg_acc = tl.where(just_ex, step_avg, ex_avg_acc)
            ex_step_acc = tl.where(just_ex, step_idx, ex_step_acc)
            done_acc = done_acc | any_ex
        else:
            any_ex = (ln_pos_slice[None, :] > step_ln_avg[:, None]) if OPTION_TYPE == 1 else (ln_pos_slice[None, :] < step_ln_avg[:, None])
            just_ex = any_ex & ~done_acc
            ex_avg_acc = tl.where(just_ex, step_avg[:, None], ex_avg_acc)
            ex_step_acc = tl.where(just_ex, step_idx, ex_step_acc)
            done_acc = done_acc | any_ex

    return current_lnS, running_sum, done_acc, ex_avg_acc, ex_step_acc

@triton.jit
def process_dim_block_bandwidth_asian(
    current_lnS, running_sum, done_acc, ex_avg_acc, ex_step_acc,
    ln_positions_ptr, st_ptr, dim_offset, p_idx, NUM_PARTICLES: tl.constexpr,
    NUM_BW_PATHS: tl.constexpr, bw_path_offs,
    BLOCK_SIZE_DIM: tl.constexpr, BLOCK_SIZE_PATHS: tl.constexpr, BLOCK_SIZE_PARTICLES: tl.constexpr,
    OPTION_TYPE: tl.constexpr
):
    for d in tl.static_range(BLOCK_SIZE_DIM):
        step_idx = dim_offset + d
        ln_pos_slice = tl.load(ln_positions_ptr + step_idx * NUM_PARTICLES + p_idx)
        current_lnS = tl.load(st_ptr + step_idx * NUM_BW_PATHS + bw_path_offs)
        
        step_S = tl.exp2(current_lnS)
        running_sum = running_sum + step_S
        step_avg = running_sum / (step_idx + 1.0)
        step_ln_avg = tl.log2(step_avg)
        
        if BLOCK_SIZE_PARTICLES == 1:
            any_ex = (step_ln_avg < ln_pos_slice) if OPTION_TYPE == 1 else (step_ln_avg > ln_pos_slice)
            just_ex = any_ex & ~done_acc
            ex_avg_acc = tl.where(just_ex, step_avg, ex_avg_acc)
            ex_step_acc = tl.where(just_ex, step_idx, ex_step_acc)
            done_acc = done_acc | any_ex
        else:
            any_ex = (ln_pos_slice[None, :] > step_ln_avg[:, None]) if OPTION_TYPE == 1 else (ln_pos_slice[None, :] < step_ln_avg[:, None])
            just_ex = any_ex & ~done_acc
            ex_avg_acc = tl.where(just_ex, step_avg[:, None], ex_avg_acc)
            ex_step_acc = tl.where(just_ex, step_idx, ex_step_acc)
            done_acc = done_acc | any_ex

    return current_lnS, running_sum, done_acc, ex_avg_acc, ex_step_acc


@triton.autotune(configs=get_autotune_configs(),
                 key=["NUM_PARTICLES", "NUM_PATHS", "NUM_DIMENSIONS", "NUM_COMPUTE_PATH_BLOCKS"],
                 warmup=2, rep=3)
@triton.jit
def mc_asian_payoff_kernel(
    ln_positions_ptr, st_ptr, partial_payoffs_ptr,
    seed, mc_offset_philox, log2_S0, drift_l2, vol_l2, r_dt_l2, terminal_discount,
    NUM_DIMENSIONS: tl.constexpr, NUM_PARTICLES: tl.constexpr, NUM_PATHS: tl.constexpr,
    NUM_COMPUTE_PATH_BLOCKS: tl.constexpr,
    BLOCK_SIZE_PARTICLES: tl.constexpr, BLOCK_SIZE_PATHS: tl.constexpr, BLOCK_SIZE_DIM: tl.constexpr,
    OPTION_TYPE: tl.constexpr, STRIKE_PRICE: tl.constexpr, USE_ANTITHETIC: tl.constexpr,
    WARP_SPECIALIZE: tl.constexpr, LOOP_FLATTEN: tl.constexpr,
    LOOP_STAGES: tl.constexpr, LOOP_UNROLL: tl.constexpr,
):
    tl.assume(NUM_DIMENSIONS % BLOCK_SIZE_DIM == 0)

    GROUP_SIZE_M = 64
    pid_m, pid_n = tl.swizzle2d(tl.program_id(0), tl.program_id(1), tl.num_programs(0), tl.num_programs(1), GROUP_SIZE_M)
    path_block_idx = pid_m; pid = pid_n

    NUM_BW_PATHS = NUM_PATHS - NUM_COMPUTE_PATH_BLOCKS * BLOCK_SIZE_PATHS
    is_compute = path_block_idx < NUM_COMPUTE_PATH_BLOCKS

    p_idx = tl.max_contiguous(tl.multiple_of(pid * BLOCK_SIZE_PARTICLES + tl.arange(0, BLOCK_SIZE_PARTICLES), BLOCK_SIZE_PARTICLES), BLOCK_SIZE_PARTICLES)
    path_offs = tl.max_contiguous(tl.multiple_of(path_block_idx * BLOCK_SIZE_PATHS + tl.arange(0, BLOCK_SIZE_PATHS), BLOCK_SIZE_PATHS), BLOCK_SIZE_PATHS).to(tl.int64)

    bw_block_idx = path_block_idx - NUM_COMPUTE_PATH_BLOCKS
    bw_path_offs = (bw_block_idx * BLOCK_SIZE_PATHS + tl.arange(0, BLOCK_SIZE_PATHS)).to(tl.int64)

    payoff_accum = tl.zeros([BLOCK_SIZE_PARTICLES], dtype=tl.float32)
    if BLOCK_SIZE_PARTICLES == 1:
        done_acc = tl.zeros([BLOCK_SIZE_PATHS], dtype=tl.int1)
        ex_avg_acc = tl.zeros([BLOCK_SIZE_PATHS], dtype=tl.float32)
        ex_step_acc = tl.full([BLOCK_SIZE_PATHS], -1, dtype=tl.int32)
    else:
        done_acc = tl.zeros([BLOCK_SIZE_PATHS, BLOCK_SIZE_PARTICLES], dtype=tl.int1)
        ex_avg_acc = tl.zeros([BLOCK_SIZE_PATHS, BLOCK_SIZE_PARTICLES], dtype=tl.float32)
        ex_step_acc = tl.full([BLOCK_SIZE_PATHS, BLOCK_SIZE_PARTICLES], -1, dtype=tl.int32)

    current_lnS = tl.full([BLOCK_SIZE_PATHS], log2_S0, dtype=tl.float32)
    running_sum = tl.zeros([BLOCK_SIZE_PATHS], dtype=tl.float32)

    if USE_ANTITHETIC:
        base_path_idx = path_block_idx * BLOCK_SIZE_PATHS
        first_offs = (base_path_idx + tl.arange(0, BLOCK_SIZE_PATHS // 2)).to(tl.int64)
        base_rng_offsets = mc_offset_philox + first_offs * NUM_DIMENSIONS
    else:
        base_rng_offsets = mc_offset_philox + path_offs * NUM_DIMENSIONS

    for dim_block_idx in tl.range(NUM_DIMENSIONS // BLOCK_SIZE_DIM, num_stages=LOOP_STAGES, loop_unroll_factor=LOOP_UNROLL, warp_specialize=WARP_SPECIALIZE, flatten=LOOP_FLATTEN):
        dim_offset = dim_block_idx * BLOCK_SIZE_DIM
        
        if NUM_BW_PATHS == NUM_PATHS:
            current_lnS, running_sum, done_acc, ex_avg_acc, ex_step_acc = process_dim_block_bandwidth_asian(
                current_lnS, running_sum, done_acc, ex_avg_acc, ex_step_acc,
                ln_positions_ptr, st_ptr, dim_offset, p_idx, NUM_PARTICLES,
                NUM_BW_PATHS, bw_path_offs,
                BLOCK_SIZE_DIM, BLOCK_SIZE_PATHS, BLOCK_SIZE_PARTICLES, OPTION_TYPE
            )
        elif NUM_BW_PATHS > 0:
            if is_compute:
                current_lnS, running_sum, done_acc, ex_avg_acc, ex_step_acc = process_dim_block_compute_asian(
                    current_lnS, running_sum, done_acc, ex_avg_acc, ex_step_acc,
                    ln_positions_ptr, dim_offset, p_idx, NUM_PARTICLES,
                    seed, base_rng_offsets, drift_l2, vol_l2,
                    BLOCK_SIZE_DIM, BLOCK_SIZE_PATHS, BLOCK_SIZE_PARTICLES, OPTION_TYPE, USE_ANTITHETIC
                )
            else:
                current_lnS, running_sum, done_acc, ex_avg_acc, ex_step_acc = process_dim_block_bandwidth_asian(
                    current_lnS, running_sum, done_acc, ex_avg_acc, ex_step_acc,
                    ln_positions_ptr, st_ptr, dim_offset, p_idx, NUM_PARTICLES,
                    NUM_BW_PATHS, bw_path_offs,
                    BLOCK_SIZE_DIM, BLOCK_SIZE_PATHS, BLOCK_SIZE_PARTICLES, OPTION_TYPE
                )
        else:
            current_lnS, running_sum, done_acc, ex_avg_acc, ex_step_acc = process_dim_block_compute_asian(
                current_lnS, running_sum, done_acc, ex_avg_acc, ex_step_acc,
                ln_positions_ptr, dim_offset, p_idx, NUM_PARTICLES,
                seed, base_rng_offsets, drift_l2, vol_l2,
                BLOCK_SIZE_DIM, BLOCK_SIZE_PATHS, BLOCK_SIZE_PARTICLES, OPTION_TYPE, USE_ANTITHETIC
            )

    terminal_avg_acc = running_sum / NUM_DIMENSIONS

    ex_disc_acc = tl.exp2((ex_step_acc.to(tl.float32) + 1.0) * r_dt_l2)
    
    if BLOCK_SIZE_PARTICLES == 1:
        final_avg = tl.where(done_acc, ex_avg_acc, terminal_avg_acc)
        final_disc = tl.where(done_acc, ex_disc_acc, terminal_discount)
        payoff = tl.maximum(0.0, STRIKE_PRICE - final_avg) if OPTION_TYPE == 1 else tl.maximum(0.0, final_avg - STRIKE_PRICE)
        payoff_accum = payoff_accum + tl.sum(payoff * final_disc, axis=0, keep_dims=True)
    else:
        final_avg = tl.where(done_acc, ex_avg_acc, terminal_avg_acc[:, None])
        final_disc = tl.where(done_acc, ex_disc_acc, terminal_discount)
        payoff = tl.maximum(0.0, STRIKE_PRICE - final_avg) if OPTION_TYPE == 1 else tl.maximum(0.0, final_avg - STRIKE_PRICE)
        payoff_accum = payoff_accum + tl.sum(payoff * final_disc, axis=0)
        
    tl.store(partial_payoffs_ptr + path_block_idx * NUM_PARTICLES + p_idx, payoff_accum)


@triton.jit
def process_dim_block_compute_basket(
    current_lnS, done_acc, ex_S_acc, ex_step_acc,
    ln_positions_ptr, dim_offset, p_idx, NUM_PARTICLES: tl.constexpr,
    seed, mc_offset_philox, path_offs_i64, first_offs_i64, drift_exp, vol_exp, weights_exp, L_ptr, a_offs_1d,
    NUM_DIMENSIONS: tl.constexpr, NUM_PATHS: tl.constexpr,
    BLOCK_SIZE_DIM: tl.constexpr, BLOCK_SIZE_PATHS: tl.constexpr, BLOCK_SIZE_PARTICLES: tl.constexpr, NUM_ASSETS: tl.constexpr,
    OPTION_TYPE: tl.constexpr, USE_ANTITHETIC: tl.constexpr, EXERCISE_STYLE: tl.constexpr, USE_FP16: tl.constexpr,
):
    if NUM_ASSETS >= 16:
        L_r = a_offs_1d[:, None]
        L_c = tl.arange(0, NUM_ASSETS)[None, :]
        L_matrix = tl.load(L_ptr + L_r * NUM_ASSETS + L_c)

    for d in tl.static_range(BLOCK_SIZE_DIM):
        step_idx = dim_offset + d

        if NUM_ASSETS >= 16:
            rand_r = a_offs_1d[:, None]
            if USE_ANTITHETIC:
                rand_c = first_offs_i64[None, :]
                rng_idx = mc_offset_philox + rand_r * NUM_DIMENSIONS * NUM_PATHS + step_idx * NUM_PATHS + rand_c
                Z_half = tl.randn(seed, rng_idx)
                Z_raw = tl.reshape(tl.join(Z_half, -Z_half), [NUM_ASSETS, BLOCK_SIZE_PATHS])
            else:
                rand_c = path_offs_i64[None, :]
                rng_idx = mc_offset_philox + rand_r * NUM_DIMENSIONS * NUM_PATHS + step_idx * NUM_PATHS + rand_c
                Z_raw = tl.randn(seed, rng_idx)

            if USE_FP16:
                corr_Z = tl.dot(L_matrix.to(tl.float16), Z_raw.to(tl.float16), out_dtype=tl.float32)
            else:
                corr_Z = tl.dot(L_matrix, Z_raw)
        else:
            corr_Z = tl.zeros([NUM_ASSETS, BLOCK_SIZE_PATHS], dtype=tl.float32)
            for j in tl.static_range(NUM_ASSETS):
                L_col = tl.load(L_ptr + a_offs_1d * NUM_ASSETS + j)
                if USE_ANTITHETIC:
                    rng_idx = mc_offset_philox + j * NUM_DIMENSIONS * NUM_PATHS + step_idx * NUM_PATHS + first_offs_i64
                    Z_half = tl.randn(seed, rng_idx)
                    Z_j = tl.ravel(tl.join(Z_half, -Z_half))
                else:
                    rng_idx = mc_offset_philox + j * NUM_DIMENSIONS * NUM_PATHS + step_idx * NUM_PATHS + path_offs_i64
                    Z_j = tl.randn(seed, rng_idx)
                corr_Z += L_col[:, None] * Z_j[None, :]

        current_lnS = current_lnS + drift_exp + vol_exp * corr_Z
        S_matrix = tl.exp2(current_lnS)
        basket_S = tl.sum(S_matrix * weights_exp, axis=0)
        basket_lnS = tl.log2(basket_S)

        if EXERCISE_STYLE == 0:
            ln_pos_slice = tl.load(ln_positions_ptr + step_idx * NUM_PARTICLES + p_idx)
            if BLOCK_SIZE_PARTICLES == 1:
                any_ex = (basket_lnS < ln_pos_slice) if OPTION_TYPE == 1 else (basket_lnS > ln_pos_slice)
            else:
                any_ex = (basket_lnS[:, None] < ln_pos_slice[None, :]) if OPTION_TYPE == 1 else (basket_lnS[:, None] > ln_pos_slice[None, :])
            just_ex = any_ex & ~done_acc
            if BLOCK_SIZE_PARTICLES == 1:
                ex_S_acc = tl.where(just_ex, basket_S, ex_S_acc)
            else:
                ex_S_acc = tl.where(just_ex, basket_S[:, None], ex_S_acc)
            ex_step_acc = tl.where(just_ex, step_idx, ex_step_acc)
            done_acc = done_acc | any_ex
        else:
            pad_val = float("inf") if OPTION_TYPE == 1 else float("-inf")
            bound_offs = (step_idx * NUM_ASSETS + a_offs_1d) * NUM_PARTICLES

            if BLOCK_SIZE_PARTICLES == 1:
                ln_bounds_a = tl.load(ln_positions_ptr + bound_offs + p_idx)
                crossed_2d = (current_lnS < ln_bounds_a[:, None]) if OPTION_TYPE == 1 else (current_lnS > ln_bounds_a[:, None])
                all_crossed = tl.min(crossed_2d.to(tl.int8), axis=0).to(tl.int1)
            else:
                p_offs = tl.arange(0, BLOCK_SIZE_PARTICLES)
                bound_offs_2d = bound_offs[:, None] + (p_idx + p_offs)[None, :]
                ln_bounds_a = tl.load(ln_positions_ptr + bound_offs_2d)
                crossed_3d = (current_lnS[:, :, None] < ln_bounds_a[:, None, :]) if OPTION_TYPE == 1 else (current_lnS[:, :, None] > ln_bounds_a[:, None, :])
                all_crossed = tl.min(crossed_3d.to(tl.int8), axis=0).to(tl.int1)

            just_ex = all_crossed & ~done_acc
            if BLOCK_SIZE_PARTICLES == 1:
                ex_S_acc = tl.where(just_ex, basket_S, ex_S_acc)
            else:
                ex_S_acc = tl.where(just_ex, basket_S[:, None], ex_S_acc)
            ex_step_acc = tl.where(just_ex, step_idx, ex_step_acc)
            done_acc = done_acc | all_crossed

    return current_lnS, done_acc, ex_S_acc, ex_step_acc


@triton.jit
def process_dim_block_bandwidth_basket(
    current_lnS, done_acc, ex_S_acc, ex_step_acc,
    ln_positions_ptr, st_ptr, dim_offset, p_idx, NUM_PARTICLES: tl.constexpr,
    NUM_BW_PATHS: tl.constexpr, bw_path_offs, weights_exp, a_offs_1d,
    BLOCK_SIZE_DIM: tl.constexpr, BLOCK_SIZE_PATHS: tl.constexpr, BLOCK_SIZE_PARTICLES: tl.constexpr, NUM_ASSETS: tl.constexpr,
    OPTION_TYPE: tl.constexpr, EXERCISE_STYLE: tl.constexpr,
):
    for d in tl.static_range(BLOCK_SIZE_DIM):
        step_idx = dim_offset + d

        if EXERCISE_STYLE == 0:
            ln_pos_slice = tl.load(ln_positions_ptr + step_idx * NUM_PARTICLES + p_idx)
            basket_lnS = tl.load(st_ptr + step_idx * NUM_BW_PATHS + bw_path_offs)
            basket_S = tl.exp2(basket_lnS)

            if BLOCK_SIZE_PARTICLES == 1:
                any_ex = (basket_lnS < ln_pos_slice) if OPTION_TYPE == 1 else (basket_lnS > ln_pos_slice)
                just_ex = any_ex & ~done_acc
                ex_S_acc = tl.where(just_ex, basket_S, ex_S_acc)
                ex_step_acc = tl.where(just_ex, step_idx, ex_step_acc)
                done_acc = done_acc | any_ex
            else:
                any_ex = (basket_lnS[:, None] < ln_pos_slice[None, :]) if OPTION_TYPE == 1 else (basket_lnS[:, None] > ln_pos_slice[None, :])
                just_ex = any_ex & ~done_acc
                ex_S_acc = tl.where(just_ex, basket_S[:, None], ex_S_acc)
                ex_step_acc = tl.where(just_ex, step_idx, ex_step_acc)
                done_acc = done_acc | any_ex
        else:
            st_offs = step_idx * (NUM_ASSETS * NUM_BW_PATHS) + a_offs_1d[:, None] * NUM_BW_PATHS + bw_path_offs[None, :]
            current_lnS = tl.load(st_ptr + st_offs)
            S_matrix = tl.exp2(current_lnS)
            basket_S = tl.sum(S_matrix * weights_exp, axis=0)

            pad_val = float("inf") if OPTION_TYPE == 1 else float("-inf")
            bound_offs = (step_idx * NUM_ASSETS + a_offs_1d) * NUM_PARTICLES

            if BLOCK_SIZE_PARTICLES == 1:
                ln_bounds_a = tl.load(ln_positions_ptr + bound_offs + p_idx)
                crossed_2d = (current_lnS < ln_bounds_a[:, None]) if OPTION_TYPE == 1 else (current_lnS > ln_bounds_a[:, None])
                all_crossed = tl.min(crossed_2d.to(tl.int8), axis=0).to(tl.int1)
            else:
                p_offs = tl.arange(0, BLOCK_SIZE_PARTICLES)
                bound_offs_2d = bound_offs[:, None] + (p_idx + p_offs)[None, :]
                ln_bounds_a = tl.load(ln_positions_ptr + bound_offs_2d)
                crossed_3d = (current_lnS[:, :, None] < ln_bounds_a[:, None, :]) if OPTION_TYPE == 1 else (current_lnS[:, :, None] > ln_bounds_a[:, None, :])
                all_crossed = tl.min(crossed_3d.to(tl.int8), axis=0).to(tl.int1)

            just_ex = all_crossed & ~done_acc
            if BLOCK_SIZE_PARTICLES == 1:
                ex_S_acc = tl.where(just_ex, basket_S, ex_S_acc)
            else:
                ex_S_acc = tl.where(just_ex, basket_S[:, None], ex_S_acc)
            ex_step_acc = tl.where(just_ex, step_idx, ex_step_acc)
            done_acc = done_acc | all_crossed

    return current_lnS, done_acc, ex_S_acc, ex_step_acc

@triton.autotune(configs=get_basket_autotune_configs(),
                 key=["NUM_PARTICLES", "NUM_PATHS", "NUM_DIMENSIONS", "NUM_COMPUTE_PATH_BLOCKS", "NUM_ASSETS", "EXERCISE_STYLE"])
@triton.jit
def mc_basket_payoff_kernel(
    ln_positions_ptr, st_ptr, partial_payoffs_ptr,
    log2_S0_ptr, drift_ptr, vol_ptr, weights_ptr, L_ptr,
    r_dt_l2, terminal_discount, seed, mc_offset_philox,
    NUM_DIMENSIONS: tl.constexpr, NUM_PARTICLES: tl.constexpr, NUM_PATHS: tl.constexpr,
    NUM_COMPUTE_PATH_BLOCKS: tl.constexpr,
    BLOCK_SIZE_PARTICLES: tl.constexpr, BLOCK_SIZE_PATHS: tl.constexpr, BLOCK_SIZE_DIM: tl.constexpr,
    NUM_ASSETS: tl.constexpr,
    OPTION_TYPE: tl.constexpr, STRIKE_PRICE: tl.constexpr,
    USE_ANTITHETIC: tl.constexpr, EXERCISE_STYLE: tl.constexpr, USE_FP16: tl.constexpr,
    WARP_SPECIALIZE: tl.constexpr, LOOP_FLATTEN: tl.constexpr,
    LOOP_STAGES: tl.constexpr, LOOP_UNROLL: tl.constexpr,
):
    tl.assume(NUM_DIMENSIONS % BLOCK_SIZE_DIM == 0)
    GROUP_SIZE_M = 64
    pid_m, pid_n = tl.swizzle2d(tl.program_id(0), tl.program_id(1), tl.num_programs(0), tl.num_programs(1), GROUP_SIZE_M)
    path_block_idx = pid_m; pid = pid_n
    
    NUM_BW_PATHS = NUM_PATHS - NUM_COMPUTE_PATH_BLOCKS * BLOCK_SIZE_PATHS
    is_compute = path_block_idx < NUM_COMPUTE_PATH_BLOCKS
    
    p_idx = tl.max_contiguous(tl.multiple_of(pid * BLOCK_SIZE_PARTICLES + tl.arange(0, BLOCK_SIZE_PARTICLES), BLOCK_SIZE_PARTICLES), BLOCK_SIZE_PARTICLES)
    path_offs = tl.max_contiguous(tl.multiple_of(path_block_idx * BLOCK_SIZE_PATHS + tl.arange(0, BLOCK_SIZE_PATHS), BLOCK_SIZE_PATHS), BLOCK_SIZE_PATHS).to(tl.int64)
    
    bw_block_idx = path_block_idx - NUM_COMPUTE_PATH_BLOCKS
    bw_path_offs = (bw_block_idx * BLOCK_SIZE_PATHS + tl.arange(0, BLOCK_SIZE_PATHS)).to(tl.int64)
    
    path_offs_i64 = path_offs.to(tl.int64)
    first_offs_i64 = (path_block_idx * BLOCK_SIZE_PATHS + tl.arange(0, BLOCK_SIZE_PATHS // 2)).to(tl.int64)

    a_offs_1d = tl.arange(0, NUM_ASSETS)

    log2_S0 = tl.load(log2_S0_ptr + a_offs_1d)
    drift = tl.load(drift_ptr + a_offs_1d)
    vol = tl.load(vol_ptr + a_offs_1d)
    weights = tl.load(weights_ptr + a_offs_1d)
    
    current_lnS = tl.broadcast_to(log2_S0[:, None], [NUM_ASSETS, BLOCK_SIZE_PATHS])
    drift_exp = drift[:, None]; vol_exp = vol[:, None]; weights_exp = weights[:, None]

    payoff_accum = tl.zeros([BLOCK_SIZE_PARTICLES], dtype=tl.float32)
    if BLOCK_SIZE_PARTICLES == 1:
        done_acc = tl.zeros([BLOCK_SIZE_PATHS], dtype=tl.int1)
        ex_S_acc = tl.zeros([BLOCK_SIZE_PATHS], dtype=tl.float32)
        ex_step_acc = tl.full([BLOCK_SIZE_PATHS], -1, dtype=tl.int32)
    else:
        done_acc = tl.zeros([BLOCK_SIZE_PATHS, BLOCK_SIZE_PARTICLES], dtype=tl.int1)
        ex_S_acc = tl.zeros([BLOCK_SIZE_PATHS, BLOCK_SIZE_PARTICLES], dtype=tl.float32)
        ex_step_acc = tl.full([BLOCK_SIZE_PATHS, BLOCK_SIZE_PARTICLES], -1, dtype=tl.int32)


    for dim_block_idx in tl.range(NUM_DIMENSIONS // BLOCK_SIZE_DIM, num_stages=LOOP_STAGES, loop_unroll_factor=LOOP_UNROLL, warp_specialize=WARP_SPECIALIZE, flatten=LOOP_FLATTEN):
        dim_offset = dim_block_idx * BLOCK_SIZE_DIM
        
        if NUM_BW_PATHS == NUM_PATHS:
            current_lnS, done_acc, ex_S_acc, ex_step_acc = process_dim_block_bandwidth_basket(
                current_lnS, done_acc, ex_S_acc, ex_step_acc,
                ln_positions_ptr, st_ptr, dim_offset, p_idx, NUM_PARTICLES,
                NUM_BW_PATHS, bw_path_offs, weights_exp, a_offs_1d,
                BLOCK_SIZE_DIM, BLOCK_SIZE_PATHS, BLOCK_SIZE_PARTICLES, NUM_ASSETS,
                OPTION_TYPE, EXERCISE_STYLE
            )
        elif NUM_BW_PATHS > 0:
            if is_compute:
                current_lnS, done_acc, ex_S_acc, ex_step_acc = process_dim_block_compute_basket(
                    current_lnS, done_acc, ex_S_acc, ex_step_acc,
                    ln_positions_ptr, dim_offset, p_idx, NUM_PARTICLES,
                    seed, mc_offset_philox, path_offs_i64, first_offs_i64, drift_exp, vol_exp, weights_exp, L_ptr, a_offs_1d,
                    NUM_DIMENSIONS, NUM_PATHS,
                    BLOCK_SIZE_DIM, BLOCK_SIZE_PATHS, BLOCK_SIZE_PARTICLES, NUM_ASSETS,
                    OPTION_TYPE, USE_ANTITHETIC, EXERCISE_STYLE, USE_FP16
                )
            else:
                current_lnS, done_acc, ex_S_acc, ex_step_acc = process_dim_block_bandwidth_basket(
                    current_lnS, done_acc, ex_S_acc, ex_step_acc,
                    ln_positions_ptr, st_ptr, dim_offset, p_idx, NUM_PARTICLES,
                    NUM_BW_PATHS, bw_path_offs, weights_exp, a_offs_1d,
                    BLOCK_SIZE_DIM, BLOCK_SIZE_PATHS, BLOCK_SIZE_PARTICLES, NUM_ASSETS,
                    OPTION_TYPE, EXERCISE_STYLE
                )
        else:
            current_lnS, done_acc, ex_S_acc, ex_step_acc = process_dim_block_compute_basket(
                current_lnS, done_acc, ex_S_acc, ex_step_acc,
                ln_positions_ptr, dim_offset, p_idx, NUM_PARTICLES,
                seed, mc_offset_philox, path_offs_i64, first_offs_i64, drift_exp, vol_exp, weights_exp, L_ptr, a_offs_1d,
                NUM_DIMENSIONS, NUM_PATHS,
                BLOCK_SIZE_DIM, BLOCK_SIZE_PATHS, BLOCK_SIZE_PARTICLES, NUM_ASSETS,
                OPTION_TYPE, USE_ANTITHETIC, EXERCISE_STYLE, USE_FP16
            )

    S_matrix_terminal = tl.exp2(current_lnS)
    masked_terminal_S = S_matrix_terminal
    basket_S_terminal = tl.sum(masked_terminal_S * weights_exp, axis=0)
    terminal_S_acc = basket_S_terminal
    
    ex_disc_acc = tl.exp2((ex_step_acc.to(tl.float32) + 1.0) * r_dt_l2)
    
    if BLOCK_SIZE_PARTICLES == 1:
        final_S = tl.where(done_acc, ex_S_acc, terminal_S_acc)
        final_disc = tl.where(done_acc, ex_disc_acc, terminal_discount)
        payoff = tl.maximum(0.0, STRIKE_PRICE - final_S) if OPTION_TYPE == 1 else tl.maximum(0.0, final_S - STRIKE_PRICE)
        payoff_accum = payoff_accum + tl.sum(payoff * final_disc, axis=0, keep_dims=True)
    else:
        final_S = tl.where(done_acc, ex_S_acc, terminal_S_acc[:, None])
        final_disc = tl.where(done_acc, ex_disc_acc, terminal_discount)
        payoff = tl.maximum(0.0, STRIKE_PRICE - final_S) if OPTION_TYPE == 1 else tl.maximum(0.0, final_S - STRIKE_PRICE)
        payoff_accum = payoff_accum + tl.sum(payoff * final_disc, axis=0)
        
    tl.store(partial_payoffs_ptr + path_block_idx * NUM_PARTICLES + p_idx, payoff_accum)
