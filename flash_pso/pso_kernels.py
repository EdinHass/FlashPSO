import triton
import triton.language as tl
from flash_pso.config import get_autotune_configs

@triton.jit
def _ex_combine(p_l, d_l, p_r, d_r):
    p_out = tl.where(d_l.to(tl.int1), p_l, p_r)
    d_out = d_l | d_r
    return p_out, d_out

@triton.jit
def _velocity_update(pos, vel, pbest, gbest, r1, r2,
                     W: tl.constexpr, C1: tl.constexpr, C2: tl.constexpr):
    return W * vel + C1 * r1 * (pbest - pos) + C2 * r2 * (gbest - pos)

# ── LOG-SPACE GENERATORS ──────────────────────────────────────────────────
# Notice we no longer return tl.exp2(lnS). We keep the paths in log-space!
@triton.jit
def _gen_path_1d(current_lnS, path_offs, path_block_idx, dim_offset, drift_l2, vol_l2,
                 SEED: tl.constexpr, MC_OFFSET_PHILOX: tl.constexpr,
                 BLOCK_SIZE_PATHS: tl.constexpr, NUM_DIMENSIONS: tl.constexpr,
                 USE_ANTITHETIC: tl.constexpr):
    if USE_ANTITHETIC:
        first_offs = (tl.multiple_of(path_block_idx * BLOCK_SIZE_PATHS, BLOCK_SIZE_PATHS) + tl.arange(0, BLOCK_SIZE_PATHS // 2))
        Z_half = tl.randn(SEED, MC_OFFSET_PHILOX + first_offs * NUM_DIMENSIONS + dim_offset)
        Z      = tl.ravel(tl.join(Z_half, -Z_half))
    else:
        Z = tl.randn(SEED, MC_OFFSET_PHILOX + path_offs * NUM_DIMENSIONS + dim_offset)
    lnS = current_lnS + drift_l2 + vol_l2 * Z
    return tl.expand_dims(lnS, 1), lnS

@triton.jit
def _gen_path_nd(current_lnS, path_offs, path_block_idx, dim_offset, drift_l2, vol_l2,
                 SEED: tl.constexpr, MC_OFFSET_PHILOX: tl.constexpr,
                 BLOCK_SIZE_PATHS: tl.constexpr, BLOCK_SIZE_DIM: tl.constexpr,
                 NUM_DIMENSIONS: tl.constexpr, USE_ANTITHETIC: tl.constexpr):
    d_offs = tl.arange(0, BLOCK_SIZE_DIM)
    if USE_ANTITHETIC:
        first_1d = (tl.multiple_of(path_block_idx * BLOCK_SIZE_PATHS, BLOCK_SIZE_PATHS) + tl.arange(0, BLOCK_SIZE_PATHS // 2))
        first_2d = tl.broadcast_to(tl.expand_dims(first_1d, 1), [BLOCK_SIZE_PATHS // 2, BLOCK_SIZE_DIM])
        d_2d     = tl.broadcast_to(tl.expand_dims(d_offs, 0),   [BLOCK_SIZE_PATHS // 2, BLOCK_SIZE_DIM])
        Z_half   = tl.randn(SEED, MC_OFFSET_PHILOX + first_2d * NUM_DIMENSIONS + dim_offset + d_2d)
        Z_2d = tl.reshape(tl.trans(tl.join(Z_half, -Z_half), (0, 2, 1)), [BLOCK_SIZE_PATHS, BLOCK_SIZE_DIM])
    else:
        p_2d = tl.broadcast_to(tl.expand_dims(path_offs, 1), [BLOCK_SIZE_PATHS, BLOCK_SIZE_DIM])
        d_2d = tl.broadcast_to(tl.expand_dims(d_offs, 0),    [BLOCK_SIZE_PATHS, BLOCK_SIZE_DIM])
        Z_2d = tl.randn(SEED, MC_OFFSET_PHILOX + p_2d * NUM_DIMENSIONS + dim_offset + d_2d)
    Z_cumsum  = tl.cumsum(Z_2d, axis=1)
    steps_2d  = tl.broadcast_to(tl.expand_dims(d_offs + 1, 0), [BLOCK_SIZE_PATHS, BLOCK_SIZE_DIM])
    lnS_2d    = (tl.broadcast_to(tl.expand_dims(current_lnS, 1), [BLOCK_SIZE_PATHS, BLOCK_SIZE_DIM]) + drift_l2 * steps_2d + vol_l2 * Z_cumsum)
    last_mask = tl.broadcast_to(tl.expand_dims(d_offs == BLOCK_SIZE_DIM - 1, 0), [BLOCK_SIZE_PATHS, BLOCK_SIZE_DIM])
    return lnS_2d, tl.sum(lnS_2d * last_mask.to(tl.float32), axis=1)
# ─────────────────────────────────────────────────────────────────────────

@triton.jit
def init_kernel(positions_ptr, velocities_ptr, pbest_costs_ptr, pbest_pos_ptr, r1_ptr, r2_ptr, num_dimensions, num_particles, seed, GENERATES_POSITIONS: tl.constexpr, GENERATES_VELOCITIES: tl.constexpr, BLOCK_SIZE: tl.constexpr, POS_OFFSET_PHILOX: tl.constexpr, VEL_OFFSET_PHILOX: tl.constexpr, R1_OFFSET_PHILOX: tl.constexpr, R2_OFFSET_PHILOX: tl.constexpr, USE_FIXED_RANDOM: tl.constexpr):
    pid = tl.program_id(0); offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    p_idx = offs // num_dimensions; d_idx = offs % num_dimensions; cm_offs = d_idx * num_particles + p_idx
    if GENERATES_POSITIONS:
        pos = tl.rand(seed, POS_OFFSET_PHILOX + offs) * 100.0
        tl.store(positions_ptr + cm_offs, pos); tl.store(pbest_pos_ptr + cm_offs, pos)
    else:
        pos = tl.load(positions_ptr + cm_offs); tl.store(pbest_pos_ptr + cm_offs, pos)
    if GENERATES_VELOCITIES:
        vel = tl.rand(seed, VEL_OFFSET_PHILOX + offs) * 5.0
        tl.store(velocities_ptr + cm_offs, vel)
    if USE_FIXED_RANDOM:
        tl.store(r1_ptr + cm_offs, tl.rand(seed, R1_OFFSET_PHILOX + offs))
        tl.store(r2_ptr + cm_offs, tl.rand(seed, R2_OFFSET_PHILOX + offs))
    tl.store(pbest_costs_ptr + p_idx, float("-inf"), mask=(d_idx == 0))

@triton.jit
def pso_update_kernel(positions_ptr, velocities_ptr, pbest_pos_ptr, gbest_pos_ptr, r1_ptr, r2_ptr, iteration, INERTIA_WEIGHT: tl.constexpr, COGNITIVE_WEIGHT: tl.constexpr, SOCIAL_WEIGHT: tl.constexpr, NUM_DIMENSIONS: tl.constexpr, NUM_PARTICLES: tl.constexpr, PSO_OFFSET_PHILOX: tl.constexpr, SEED: tl.constexpr, USE_FIXED_RANDOM: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0); offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE); mask = offs < (NUM_DIMENSIONS * NUM_PARTICLES)
    p_idx = offs % NUM_PARTICLES; d_idx = offs // NUM_PARTICLES
    pos = tl.load(positions_ptr + offs, mask=mask); vel = tl.load(velocities_ptr + offs, mask=mask); pbest = tl.load(pbest_pos_ptr + offs, mask=mask); gbest = tl.load(gbest_pos_ptr + d_idx, mask=mask)
    if USE_FIXED_RANDOM:
        r1 = tl.load(r1_ptr + offs, mask=mask); r2 = tl.load(r2_ptr + offs, mask=mask)
    else:
        roffs = p_idx * NUM_DIMENSIONS + d_idx
        r1 = tl.rand(SEED, PSO_OFFSET_PHILOX + 2 * iteration * NUM_PARTICLES * NUM_DIMENSIONS + roffs); r2 = tl.rand(SEED, PSO_OFFSET_PHILOX + (2 * iteration + 1) * NUM_PARTICLES * NUM_DIMENSIONS + roffs)
    new_vel = INERTIA_WEIGHT * vel + COGNITIVE_WEIGHT * r1 * (pbest - pos) + SOCIAL_WEIGHT * r2 * (gbest - pos)
    new_pos = pos + new_vel
    tl.store(velocities_ptr + offs, new_vel, mask=mask); tl.store(positions_ptr + offs, new_pos, mask=mask)

@triton.autotune(configs=get_autotune_configs(), key=["NUM_PARTICLES", "NUM_PATHS", "NUM_DIMENSIONS", "NUM_COMPUTE_PATH_BLOCKS"], warmup=2, rep=3)
@triton.jit
def mc_payoff_kernel(
    positions_ptr, st_ptr, partial_payoffs_ptr, discounts_ptr,
    S0, r, sigma, dt,
    SEED: tl.constexpr, NUM_DIMENSIONS: tl.constexpr, NUM_PARTICLES: tl.constexpr, NUM_PATHS: tl.constexpr,
    NUM_COMPUTE_PATH_BLOCKS: tl.constexpr, BLOCK_SIZE_PARTICLES: tl.constexpr, BLOCK_SIZE_PATHS: tl.constexpr, BLOCK_SIZE_DIM: tl.constexpr,
    OPTION_TYPE: tl.constexpr, STRIKE_PRICE: tl.constexpr, MC_OFFSET_PHILOX: tl.constexpr, USE_ANTITHETIC: tl.constexpr,
    WARP_SPECIALIZE: tl.constexpr, LOOP_FLATTEN: tl.constexpr, LOOP_STAGES: tl.constexpr, LOOP_UNROLL: tl.constexpr,
):
    tl.assume(NUM_DIMENSIONS % BLOCK_SIZE_DIM == 0)
    GROUP_SIZE_M = 8
    pid_m, pid_n = tl.swizzle2d(tl.program_id(0), tl.program_id(1), tl.num_programs(0), tl.num_programs(1), GROUP_SIZE_M)
    pid = pid_m; path_block_idx = pid_n

    NUM_BW_PATHS = NUM_PATHS - NUM_COMPUTE_PATH_BLOCKS * BLOCK_SIZE_PATHS
    p_idx = tl.max_contiguous(tl.multiple_of(pid * BLOCK_SIZE_PARTICLES + tl.arange(0, BLOCK_SIZE_PARTICLES), BLOCK_SIZE_PARTICLES), BLOCK_SIZE_PARTICLES)
    path_offs = tl.max_contiguous(tl.multiple_of(path_block_idx * BLOCK_SIZE_PATHS + tl.arange(0, BLOCK_SIZE_PATHS), BLOCK_SIZE_PATHS), BLOCK_SIZE_PATHS)
    is_compute = path_block_idx < NUM_COMPUTE_PATH_BLOCKS

    if BLOCK_SIZE_PARTICLES >= 4:
        pos_desc = tl.make_tensor_descriptor(base=positions_ptr, shape=[NUM_DIMENSIONS, NUM_PARTICLES], strides=[NUM_PARTICLES, 1], block_shape=[BLOCK_SIZE_DIM, BLOCK_SIZE_PARTICLES])

    NUM_BW_PATHS_SAFE = tl.constexpr(NUM_BW_PATHS if NUM_BW_PATHS > 0 else BLOCK_SIZE_PATHS)
    path_desc = tl.make_tensor_descriptor(base=st_ptr, shape=[NUM_DIMENSIONS, NUM_BW_PATHS_SAFE], strides=[NUM_BW_PATHS_SAFE, 1], block_shape=[BLOCK_SIZE_DIM, BLOCK_SIZE_PATHS])

    if BLOCK_SIZE_PARTICLES < 4:
        d_rows = tl.expand_dims(tl.arange(0, BLOCK_SIZE_DIM), 1)
        p_cols = tl.expand_dims(pid * BLOCK_SIZE_PARTICLES + tl.arange(0, BLOCK_SIZE_PARTICLES), 0)
        pos_ptrs = positions_ptr + d_rows * NUM_PARTICLES + p_cols

    # ── DEFERRED PAYOFF TRACKERS ─────────────────────────────────────────────
    payoff_accum = tl.zeros([BLOCK_SIZE_PARTICLES], dtype=tl.float32)

    if BLOCK_SIZE_PARTICLES == 1:
        done_acc    = tl.zeros([BLOCK_SIZE_PATHS], dtype=tl.int1)
        ex_lnS_acc  = tl.zeros([BLOCK_SIZE_PATHS], dtype=tl.float32)
        ex_disc_acc = tl.zeros([BLOCK_SIZE_PATHS], dtype=tl.float32)
    else:
        done_acc    = tl.zeros([BLOCK_SIZE_PATHS, BLOCK_SIZE_PARTICLES], dtype=tl.int1)
        ex_lnS_acc  = tl.zeros([BLOCK_SIZE_PATHS, BLOCK_SIZE_PARTICLES], dtype=tl.float32)
        ex_disc_acc = tl.zeros([BLOCK_SIZE_PATHS, BLOCK_SIZE_PARTICLES], dtype=tl.float32)

    terminal_lnS_acc = tl.zeros([BLOCK_SIZE_PATHS], dtype=tl.float32)

    LOG2E    = 1.4426950408889634
    drift_l2 = (r - 0.5 * sigma * sigma) * dt * LOG2E
    vol_l2   = sigma * tl.sqrt(dt) * LOG2E
    current_lnS = tl.full([BLOCK_SIZE_PATHS], tl.log(S0) * LOG2E, dtype=tl.float32)

    for dim_block_idx in tl.range(NUM_DIMENSIONS // BLOCK_SIZE_DIM, num_stages=LOOP_STAGES, loop_unroll_factor=LOOP_UNROLL, warp_specialize=WARP_SPECIALIZE, flatten=LOOP_FLATTEN):
        dim_offset = dim_block_idx * BLOCK_SIZE_DIM
        discount_tile = tl.load(discounts_ptr + dim_offset + tl.arange(0, BLOCK_SIZE_DIM))

        if BLOCK_SIZE_PARTICLES >= 4:
            pos_tile = tl.trans(tl.load_tensor_descriptor(pos_desc, [dim_offset, pid * BLOCK_SIZE_PARTICLES]))
        else:
            pos_tile = tl.trans(tl.load(pos_ptrs))
            pos_ptrs += BLOCK_SIZE_DIM * NUM_PARTICLES

        if NUM_BW_PATHS == 0:
            if BLOCK_SIZE_DIM == 1: path_tile, current_lnS = _gen_path_1d(current_lnS, path_offs, path_block_idx, dim_offset, drift_l2, vol_l2, SEED, MC_OFFSET_PHILOX, BLOCK_SIZE_PATHS, NUM_DIMENSIONS, USE_ANTITHETIC)
            else: path_tile, current_lnS = _gen_path_nd(current_lnS, path_offs, path_block_idx, dim_offset, drift_l2, vol_l2, SEED, MC_OFFSET_PHILOX, BLOCK_SIZE_PATHS, BLOCK_SIZE_DIM, NUM_DIMENSIONS, USE_ANTITHETIC)
        elif NUM_COMPUTE_PATH_BLOCKS == 0:
            path_tile = tl.trans(tl.load_tensor_descriptor(path_desc, [dim_offset, path_block_idx * BLOCK_SIZE_PATHS]))
        else:
            if is_compute:
                if BLOCK_SIZE_DIM == 1: path_tile, current_lnS = _gen_path_1d(current_lnS, path_offs, path_block_idx, dim_offset, drift_l2, vol_l2, SEED, MC_OFFSET_PHILOX, BLOCK_SIZE_PATHS, NUM_DIMENSIONS, USE_ANTITHETIC)
                else: path_tile, current_lnS = _gen_path_nd(current_lnS, path_offs, path_block_idx, dim_offset, drift_l2, vol_l2, SEED, MC_OFFSET_PHILOX, BLOCK_SIZE_PATHS, BLOCK_SIZE_DIM, NUM_DIMENSIONS, USE_ANTITHETIC)
            else:
                bw_block_idx = path_block_idx - NUM_COMPUTE_PATH_BLOCKS
                path_tile = tl.trans(tl.load_tensor_descriptor(path_desc, [dim_offset, bw_block_idx * BLOCK_SIZE_PATHS]))

        # ── LOG-SPACE BOUNDARY CHECKS ────────────────────────────────────────
        if BLOCK_SIZE_PARTICLES == 1 and BLOCK_SIZE_DIM == 1:
            if dim_block_idx == NUM_DIMENSIONS // BLOCK_SIZE_DIM - 1:
                terminal_lnS_acc = tl.reshape(path_tile, [BLOCK_SIZE_PATHS])
            
            lnS_vec       = tl.reshape(path_tile, [BLOCK_SIZE_PATHS])
            ln_pos_scalar = tl.log2(tl.reshape(pos_tile, [1]))
            discount      = tl.reshape(discount_tile, [1])

            any_ex      = lnS_vec < ln_pos_scalar if OPTION_TYPE == 1 else lnS_vec > ln_pos_scalar
            just_ex     = any_ex & ~done_acc
            
            ex_lnS_acc  = tl.where(just_ex, lnS_vec, ex_lnS_acc)
            ex_disc_acc = tl.where(just_ex, discount, ex_disc_acc)
            done_acc    = done_acc | any_ex

        elif BLOCK_SIZE_PARTICLES == 1 and BLOCK_SIZE_DIM >= 4:
            if dim_block_idx == NUM_DIMENSIONS // BLOCK_SIZE_DIM - 1:
                last_mask = tl.broadcast_to(tl.expand_dims(tl.arange(0, BLOCK_SIZE_DIM) == BLOCK_SIZE_DIM - 1, 0), [BLOCK_SIZE_PATHS, BLOCK_SIZE_DIM])
                terminal_lnS_acc = tl.sum(path_tile * last_mask.to(tl.float32), axis=1)
            
            ln_pos_2d = tl.broadcast_to(tl.expand_dims(tl.log2(tl.reshape(pos_tile, [BLOCK_SIZE_DIM])), 0), [BLOCK_SIZE_PATHS, BLOCK_SIZE_DIM])
            disc_2d   = tl.broadcast_to(tl.expand_dims(discount_tile, 0), [BLOCK_SIZE_PATHS, BLOCK_SIZE_DIM])
            
            ex_mask   = (ln_pos_2d > path_tile) if OPTION_TYPE == 1 else (ln_pos_2d < path_tile)
            ex_int_2d = ex_mask.to(tl.int32)
            
            # Combine finds the FIRST exercise in the dimension block
            block_lnS_1d, block_done_int = tl.reduce((path_tile, ex_int_2d), axis=1, combine_fn=_ex_combine)
            block_disc_1d, _             = tl.reduce((disc_2d, ex_int_2d), axis=1, combine_fn=_ex_combine)
            
            block_done = block_done_int.to(tl.int1)
            just_ex_1d = block_done & ~done_acc
            
            ex_lnS_acc  = tl.where(just_ex_1d, block_lnS_1d, ex_lnS_acc)
            ex_disc_acc = tl.where(just_ex_1d, block_disc_1d, ex_disc_acc)
            done_acc    = done_acc | block_done

        elif BLOCK_SIZE_PARTICLES >= 4 and BLOCK_SIZE_DIM == 1:
            if dim_block_idx == NUM_DIMENSIONS // BLOCK_SIZE_DIM - 1:
                terminal_lnS_acc = tl.reshape(path_tile, [BLOCK_SIZE_PATHS])
            
            ln_pos_2d = tl.broadcast_to(tl.expand_dims(tl.log2(tl.reshape(pos_tile,  [BLOCK_SIZE_PARTICLES])), 0),  [BLOCK_SIZE_PATHS, BLOCK_SIZE_PARTICLES])
            lnS_2d    = tl.broadcast_to(tl.expand_dims(tl.reshape(path_tile, [BLOCK_SIZE_PATHS]), 1), [BLOCK_SIZE_PATHS, BLOCK_SIZE_PARTICLES])
            discount  = tl.reshape(discount_tile, [1])
            
            any_ex_2d  = (ln_pos_2d > lnS_2d) if OPTION_TYPE == 1 else (ln_pos_2d < lnS_2d)
            just_ex_2d = any_ex_2d & ~done_acc
            
            ex_lnS_acc  = tl.where(just_ex_2d, lnS_2d, ex_lnS_acc)
            ex_disc_acc = tl.where(just_ex_2d, discount, ex_disc_acc)
            done_acc    = done_acc | any_ex_2d

        else:
            if dim_block_idx == NUM_DIMENSIONS // BLOCK_SIZE_DIM - 1:
                last_mask_2d = tl.broadcast_to(tl.expand_dims(tl.arange(0, BLOCK_SIZE_DIM) == BLOCK_SIZE_DIM - 1, 0), [BLOCK_SIZE_PATHS, BLOCK_SIZE_DIM])
                terminal_lnS_acc = tl.sum(path_tile * last_mask_2d.to(tl.float32), axis=1)
            
            ln_pos_3d = tl.broadcast_to(tl.expand_dims(tl.log2(pos_tile), 0), [BLOCK_SIZE_PATHS, BLOCK_SIZE_PARTICLES, BLOCK_SIZE_DIM])
            lnS_3d    = tl.broadcast_to(tl.expand_dims(path_tile, 1), [BLOCK_SIZE_PATHS, BLOCK_SIZE_PARTICLES, BLOCK_SIZE_DIM])
            disc_3d   = tl.broadcast_to(tl.expand_dims(tl.expand_dims(discount_tile, 0), 0), [BLOCK_SIZE_PATHS, BLOCK_SIZE_PARTICLES, BLOCK_SIZE_DIM])
            
            ex_3d     = (ln_pos_3d > lnS_3d) if OPTION_TYPE == 1 else (ln_pos_3d < lnS_3d)
            ex_int_3d = ex_3d.to(tl.int32)
            
            block_lnS_2d, block_done_int = tl.reduce((lnS_3d, ex_int_3d), axis=2, combine_fn=_ex_combine)
            block_disc_2d, _             = tl.reduce((disc_3d, ex_int_3d), axis=2, combine_fn=_ex_combine)
            
            block_done_2d = block_done_int.to(tl.int1)
            just_ex_2d    = block_done_2d & ~done_acc
            
            ex_lnS_acc  = tl.where(just_ex_2d, block_lnS_2d, ex_lnS_acc)
            ex_disc_acc = tl.where(just_ex_2d, block_disc_2d, ex_disc_acc)
            done_acc    = done_acc | block_done_2d

    # ── TERMINAL MATH (Executed exactly ONCE) ────────────────────────────────
    terminal_discount = tl.load(discounts_ptr + NUM_DIMENSIONS - 1)

    if BLOCK_SIZE_PARTICLES == 1:
        term_S       = tl.exp2(terminal_lnS_acc)
        early_S      = tl.exp2(ex_lnS_acc)
        term_payoff  = tl.maximum(0.0, STRIKE_PRICE - term_S) if OPTION_TYPE == 1 else tl.maximum(0.0, term_S - STRIKE_PRICE)
        early_payoff = tl.maximum(0.0, STRIKE_PRICE - early_S) if OPTION_TYPE == 1 else tl.maximum(0.0, early_S - STRIKE_PRICE)
        
        final_payoff = tl.where(done_acc, early_payoff * ex_disc_acc, term_payoff * terminal_discount)
        payoff_accum = payoff_accum + tl.sum(final_payoff, axis=0, keep_dims=True)
    else:
        term_S       = tl.exp2(terminal_lnS_acc)
        early_S      = tl.exp2(ex_lnS_acc)
        term_payoff  = tl.maximum(0.0, STRIKE_PRICE - term_S) if OPTION_TYPE == 1 else tl.maximum(0.0, term_S - STRIKE_PRICE)
        term_payoff_2d = tl.broadcast_to(tl.expand_dims(term_payoff, 1), [BLOCK_SIZE_PATHS, BLOCK_SIZE_PARTICLES])
        
        early_payoff = tl.maximum(0.0, STRIKE_PRICE - early_S) if OPTION_TYPE == 1 else tl.maximum(0.0, early_S - STRIKE_PRICE)
        
        final_payoff = tl.where(done_acc, early_payoff * ex_disc_acc, term_payoff_2d * terminal_discount)
        payoff_accum = payoff_accum + tl.sum(final_payoff, axis=0)

    NUM_PATH_BLOCKS = NUM_PATHS // BLOCK_SIZE_PATHS
    tl.store(partial_payoffs_ptr + p_idx * NUM_PATH_BLOCKS + path_block_idx, payoff_accum)
