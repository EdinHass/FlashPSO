import triton
import triton.language as tl
from flash_pso.config import get_autotune_configs


# Parallel first-exercise-date finder over a time-step tile.
# Used as combine_fn in tl.reduce — O(log D) rounds vs O(D) serial done_acc chain.
# State: (payoff_at_first_exercise, any_exercise_as_int32).
@triton.jit
def _ex_combine(p_l, d_l, p_r, d_r):
    p_out = tl.where(d_l.to(tl.int1), p_l, p_r)
    d_out = d_l | d_r
    return p_out, d_out


# PSO velocity formula. All inputs [BLOCK_SIZE_PARTICLES, BLOCK_SIZE_DIM].
@triton.jit
def _velocity_update(pos, vel, pbest, gbest, r1, r2,
                     W: tl.constexpr, C1: tl.constexpr, C2: tl.constexpr):
    return W * vel + C1 * r1 * (pbest - pos) + C2 * r2 * (gbest - pos)


# GBM path tile for BLOCK_SIZE_DIM==1.
# With USE_ANTITHETIC, generates BLOCK_SIZE_PATHS//2 Philox draws and derives
# the second half by negation — interleaved as (z0,-z0, z1,-z1, ...).
# Returns (path_tile [BLOCK_SIZE_PATHS, 1], new_current_lnS [BLOCK_SIZE_PATHS]).
@triton.jit
def _gen_path_1d(current_lnS, path_offs, path_block_idx, dim_offset, drift_l2, vol_l2,
                 SEED: tl.constexpr, MC_OFFSET_PHILOX: tl.constexpr,
                 BLOCK_SIZE_PATHS: tl.constexpr, NUM_DIMENSIONS: tl.constexpr,
                 USE_ANTITHETIC: tl.constexpr):
    if USE_ANTITHETIC:
        first_offs = (tl.multiple_of(path_block_idx * BLOCK_SIZE_PATHS, BLOCK_SIZE_PATHS)
                      + tl.arange(0, BLOCK_SIZE_PATHS // 2))
        Z_half = tl.randn(SEED, MC_OFFSET_PHILOX + first_offs * NUM_DIMENSIONS + dim_offset)
        Z      = tl.ravel(tl.join(Z_half, -Z_half))
    else:
        Z = tl.randn(SEED, MC_OFFSET_PHILOX + path_offs * NUM_DIMENSIONS + dim_offset)
    lnS = current_lnS + drift_l2 + vol_l2 * Z
    return tl.expand_dims(tl.exp2(lnS), 1), lnS


# GBM path tile for BLOCK_SIZE_DIM>=4.
# With USE_ANTITHETIC, generates [BLOCK_SIZE_PATHS//2, BLOCK_SIZE_DIM] Philox draws.
# tl.join -> [H,D,2], tl.trans -> [H,2,D], reshape -> [BLOCK_SIZE_PATHS,D]:
# rows interleaved (row 2i = +Z[i,:], row 2i+1 = -Z[i,:]).
# Returns (path_tile [BLOCK_SIZE_PATHS, BLOCK_SIZE_DIM], new_current_lnS [BLOCK_SIZE_PATHS]).
@triton.jit
def _gen_path_nd(current_lnS, path_offs, path_block_idx, dim_offset, drift_l2, vol_l2,
                 SEED: tl.constexpr, MC_OFFSET_PHILOX: tl.constexpr,
                 BLOCK_SIZE_PATHS: tl.constexpr, BLOCK_SIZE_DIM: tl.constexpr,
                 NUM_DIMENSIONS: tl.constexpr, USE_ANTITHETIC: tl.constexpr):
    d_offs = tl.arange(0, BLOCK_SIZE_DIM)
    if USE_ANTITHETIC:
        first_1d = (tl.multiple_of(path_block_idx * BLOCK_SIZE_PATHS, BLOCK_SIZE_PATHS)
                    + tl.arange(0, BLOCK_SIZE_PATHS // 2))
        first_2d = tl.broadcast_to(tl.expand_dims(first_1d, 1), [BLOCK_SIZE_PATHS // 2, BLOCK_SIZE_DIM])
        d_2d     = tl.broadcast_to(tl.expand_dims(d_offs, 0),   [BLOCK_SIZE_PATHS // 2, BLOCK_SIZE_DIM])
        Z_half   = tl.randn(SEED, MC_OFFSET_PHILOX + first_2d * NUM_DIMENSIONS + dim_offset + d_2d)
        # [H,D,2] -> trans [H,2,D] -> reshape [P,D]: row 2i=+Z[i], row 2i+1=-Z[i]
        Z_2d = tl.reshape(tl.trans(tl.join(Z_half, -Z_half), (0, 2, 1)), [BLOCK_SIZE_PATHS, BLOCK_SIZE_DIM])
    else:
        p_2d = tl.broadcast_to(tl.expand_dims(path_offs, 1), [BLOCK_SIZE_PATHS, BLOCK_SIZE_DIM])
        d_2d = tl.broadcast_to(tl.expand_dims(d_offs, 0),    [BLOCK_SIZE_PATHS, BLOCK_SIZE_DIM])
        Z_2d = tl.randn(SEED, MC_OFFSET_PHILOX + p_2d * NUM_DIMENSIONS + dim_offset + d_2d)
    Z_cumsum  = tl.cumsum(Z_2d, axis=1)
    steps_2d  = tl.broadcast_to(tl.expand_dims(d_offs + 1, 0), [BLOCK_SIZE_PATHS, BLOCK_SIZE_DIM])
    lnS_2d    = (tl.broadcast_to(tl.expand_dims(current_lnS, 1), [BLOCK_SIZE_PATHS, BLOCK_SIZE_DIM])
                 + drift_l2 * steps_2d + vol_l2 * Z_cumsum)
    last_mask = tl.broadcast_to(tl.expand_dims(d_offs == BLOCK_SIZE_DIM - 1, 0), [BLOCK_SIZE_PATHS, BLOCK_SIZE_DIM])
    return tl.exp2(lnS_2d), tl.sum(lnS_2d * last_mask.to(tl.float32), axis=1)


@triton.jit
def init_kernel(
    positions_ptr,
    velocities_ptr,
    pbest_costs_ptr,
    pbest_pos_ptr,
    r1_ptr,
    r2_ptr,
    num_dimensions,
    num_particles,
    seed,
    GENERATES_POSITIONS:  tl.constexpr,
    GENERATES_VELOCITIES: tl.constexpr,
    BLOCK_SIZE:           tl.constexpr,
    POS_OFFSET_PHILOX:    tl.constexpr,
    VEL_OFFSET_PHILOX:    tl.constexpr,
    R1_OFFSET_PHILOX:     tl.constexpr,
    R2_OFFSET_PHILOX:     tl.constexpr,
    USE_FIXED_RANDOM:     tl.constexpr,
):
    pid  = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    p_idx   = offs // num_dimensions
    d_idx   = offs % num_dimensions
    cm_offs = d_idx * num_particles + p_idx

    if GENERATES_POSITIONS:
        pos = tl.rand(seed, POS_OFFSET_PHILOX + offs) * 100.0
        tl.store(positions_ptr + cm_offs, pos)
        tl.store(pbest_pos_ptr + cm_offs, pos)
    else:
        pos = tl.load(positions_ptr + cm_offs)
        tl.store(pbest_pos_ptr + cm_offs, pos)

    if GENERATES_VELOCITIES:
        vel = tl.rand(seed, VEL_OFFSET_PHILOX + offs) * 5.0
        tl.store(velocities_ptr + cm_offs, vel)

    if USE_FIXED_RANDOM:
        tl.store(r1_ptr + cm_offs, tl.rand(seed, R1_OFFSET_PHILOX + offs))
        tl.store(r2_ptr + cm_offs, tl.rand(seed, R2_OFFSET_PHILOX + offs))

    tl.store(pbest_costs_ptr + p_idx, float("-inf"), mask=(d_idx == 0))


@triton.autotune(
    configs=get_autotune_configs(),
    key=["NUM_PARTICLES", "NUM_PATHS", "NUM_DIMENSIONS", "NUM_COMPUTE_PATH_BLOCKS"],
    warmup=2,
    rep=3,
)
@triton.jit
def pso_kernel(
    positions_ptr,          # [NUM_DIMENSIONS, NUM_PARTICLES] col-major
    velocities_ptr,         # [NUM_DIMENSIONS, NUM_PARTICLES] col-major
    pbest_payoff_ptr,       # [NUM_PARTICLES]
    pbest_pos_ptr,          # [NUM_DIMENSIONS, NUM_PARTICLES] col-major
    gbest_payoff_ptr,       # [1]
    gbest_pos_ptr,          # [NUM_DIMENSIONS]
    st_ptr,                 # [NUM_DIMENSIONS, NUM_BW_PATHS] col-major
    r1_ptr,                 # [NUM_DIMENSIONS, NUM_PARTICLES] col-major
    r2_ptr,                 # [NUM_DIMENSIONS, NUM_PARTICLES] col-major
    partial_payoffs_ptr,    # [NUM_PARTICLES, NUM_PATHS // BLOCK_SIZE_PATHS]
    discounts_ptr,          # [NUM_DIMENSIONS]
    iteration,
    S0, r, sigma, dt,
    SEED:                    tl.constexpr,
    INERTIA_WEIGHT:          tl.constexpr,
    COGNITIVE_WEIGHT:        tl.constexpr,
    SOCIAL_WEIGHT:           tl.constexpr,
    NUM_DIMENSIONS:          tl.constexpr,
    NUM_PARTICLES:           tl.constexpr,
    NUM_PATHS:               tl.constexpr,
    NUM_COMPUTE_PATH_BLOCKS: tl.constexpr,
    BLOCK_SIZE_PARTICLES:    tl.constexpr,
    BLOCK_SIZE_PATHS:        tl.constexpr,
    BLOCK_SIZE_DIM:          tl.constexpr,
    OPTION_TYPE:             tl.constexpr,
    STRIKE_PRICE:            tl.constexpr,
    PSO_OFFSET_PHILOX:       tl.constexpr,
    MC_OFFSET_PHILOX:        tl.constexpr,
    EVAL_ONLY:               tl.constexpr,
    USE_FIXED_RANDOM:        tl.constexpr,
    USE_ANTITHETIC:          tl.constexpr,
    WARP_SPECIALIZE:         tl.constexpr,
    LOOP_FLATTEN:            tl.constexpr,
    LOOP_STAGES:             tl.constexpr,
    LOOP_UNROLL:             tl.constexpr,
):
    tl.assume(NUM_DIMENSIONS % BLOCK_SIZE_DIM == 0)

    GROUP_SIZE_M   = 8
    pid_m, pid_n   = tl.swizzle2d(tl.program_id(0), tl.program_id(1),
                                   tl.num_programs(0), tl.num_programs(1), GROUP_SIZE_M)
    pid            = pid_m
    path_block_idx = pid_n

    NUM_BW_PATHS = NUM_PATHS - NUM_COMPUTE_PATH_BLOCKS * BLOCK_SIZE_PATHS

    p_idx = tl.max_contiguous(
        tl.multiple_of(pid * BLOCK_SIZE_PARTICLES + tl.arange(0, BLOCK_SIZE_PARTICLES), BLOCK_SIZE_PARTICLES),
        BLOCK_SIZE_PARTICLES,
    )
    path_offs = tl.max_contiguous(
        tl.multiple_of(path_block_idx * BLOCK_SIZE_PATHS + tl.arange(0, BLOCK_SIZE_PATHS), BLOCK_SIZE_PATHS),
        BLOCK_SIZE_PATHS,
    )

    is_compute = path_block_idx < NUM_COMPUTE_PATH_BLOCKS

    if BLOCK_SIZE_PARTICLES >= 4:
        pos_desc = tl.make_tensor_descriptor(
            base=positions_ptr,
            shape=[NUM_DIMENSIONS, NUM_PARTICLES],       # pyright: ignore[reportArgumentType]
            strides=[NUM_PARTICLES, 1],                  # pyright: ignore[reportArgumentType]
            block_shape=[BLOCK_SIZE_DIM, BLOCK_SIZE_PARTICLES],
        )
        vel_desc = tl.make_tensor_descriptor(
            base=velocities_ptr,
            shape=[NUM_DIMENSIONS, NUM_PARTICLES],       # pyright: ignore[reportArgumentType]
            strides=[NUM_PARTICLES, 1],                  # pyright: ignore[reportArgumentType]
            block_shape=[BLOCK_SIZE_DIM, BLOCK_SIZE_PARTICLES],
        )
        pbest_pos_desc = tl.make_tensor_descriptor(
            base=pbest_pos_ptr,
            shape=[NUM_DIMENSIONS, NUM_PARTICLES],       # pyright: ignore[reportArgumentType]
            strides=[NUM_PARTICLES, 1],                  # pyright: ignore[reportArgumentType]
            block_shape=[BLOCK_SIZE_DIM, BLOCK_SIZE_PARTICLES],
        )
        if USE_FIXED_RANDOM:
            r1_desc = tl.make_tensor_descriptor(
                base=r1_ptr,
                shape=[NUM_DIMENSIONS, NUM_PARTICLES],   # pyright: ignore[reportArgumentType]
                strides=[NUM_PARTICLES, 1],              # pyright: ignore[reportArgumentType]
                block_shape=[BLOCK_SIZE_DIM, BLOCK_SIZE_PARTICLES],
            )
            r2_desc = tl.make_tensor_descriptor(
                base=r2_ptr,
                shape=[NUM_DIMENSIONS, NUM_PARTICLES],   # pyright: ignore[reportArgumentType]
                strides=[NUM_PARTICLES, 1],              # pyright: ignore[reportArgumentType]
                block_shape=[BLOCK_SIZE_DIM, BLOCK_SIZE_PARTICLES],
            )

    NUM_BW_PATHS_SAFE = tl.constexpr(NUM_BW_PATHS if NUM_BW_PATHS > 0 else BLOCK_SIZE_PATHS)
    path_desc = tl.make_tensor_descriptor(
        base=st_ptr,
        shape=[NUM_DIMENSIONS, NUM_BW_PATHS_SAFE],       # pyright: ignore[reportArgumentType]
        strides=[NUM_BW_PATHS_SAFE, 1],                  # pyright: ignore[reportArgumentType]
        block_shape=[BLOCK_SIZE_DIM, BLOCK_SIZE_PATHS],
    )

    payoff_accum = tl.zeros([BLOCK_SIZE_PARTICLES], dtype=tl.float32)

    if BLOCK_SIZE_PARTICLES == 1:
        done_acc   = tl.zeros([BLOCK_SIZE_PATHS], dtype=tl.int1)
        payoff_acc = tl.zeros([BLOCK_SIZE_PATHS], dtype=tl.float32)
    else:
        done_acc   = tl.zeros([BLOCK_SIZE_PATHS, BLOCK_SIZE_PARTICLES], dtype=tl.int1)
        payoff_acc = tl.zeros([BLOCK_SIZE_PATHS, BLOCK_SIZE_PARTICLES], dtype=tl.float32)

    terminal_path_acc = tl.zeros([BLOCK_SIZE_PATHS], dtype=tl.float32)

    LOG2E    = 1.4426950408889634
    drift_l2 = (r - 0.5 * sigma * sigma) * dt * LOG2E
    vol_l2   = sigma * tl.sqrt(dt) * LOG2E

    current_lnS = tl.full([BLOCK_SIZE_PATHS], tl.log(S0) * LOG2E, dtype=tl.float32)

    for dim_block_idx in tl.range(
        NUM_DIMENSIONS // BLOCK_SIZE_DIM,
        num_stages=LOOP_STAGES,
        loop_unroll_factor=LOOP_UNROLL,
        warp_specialize=WARP_SPECIALIZE,
        flatten=LOOP_FLATTEN,
    ):
        dim_offset    = tl.multiple_of(dim_block_idx * BLOCK_SIZE_DIM, BLOCK_SIZE_DIM)
        discount_tile = tl.load(discounts_ptr + dim_offset + tl.arange(0, BLOCK_SIZE_DIM))

        # ── PSO update (path_block_idx==0 only) ──────────────────────────────
        if BLOCK_SIZE_PARTICLES >= 4:
            pos_tile = tl.trans(tl.load_tensor_descriptor(pos_desc, [dim_offset, pid * BLOCK_SIZE_PARTICLES]))

            if path_block_idx == 0 and not EVAL_ONLY:
                vel_tile       = tl.trans(tl.load_tensor_descriptor(vel_desc,       [dim_offset, pid * BLOCK_SIZE_PARTICLES]))
                pbest_pos_tile = tl.trans(tl.load_tensor_descriptor(pbest_pos_desc, [dim_offset, pid * BLOCK_SIZE_PARTICLES]))
                dim_idx        = dim_offset + tl.arange(0, BLOCK_SIZE_DIM)
                gbest_tile     = tl.broadcast_to(tl.expand_dims(tl.load(gbest_pos_ptr + dim_idx), 0), [BLOCK_SIZE_PARTICLES, BLOCK_SIZE_DIM])
                if USE_FIXED_RANDOM:
                    r1_tile = tl.trans(tl.load_tensor_descriptor(r1_desc, [dim_offset, pid * BLOCK_SIZE_PARTICLES]))
                    r2_tile = tl.trans(tl.load_tensor_descriptor(r2_desc, [dim_offset, pid * BLOCK_SIZE_PARTICLES]))
                else:
                    p2d     = tl.broadcast_to(tl.expand_dims(pid * BLOCK_SIZE_PARTICLES + tl.arange(0, BLOCK_SIZE_PARTICLES), 1), [BLOCK_SIZE_PARTICLES, BLOCK_SIZE_DIM])
                    d2d     = tl.broadcast_to(tl.expand_dims(dim_idx, 0), [BLOCK_SIZE_PARTICLES, BLOCK_SIZE_DIM])
                    roffs   = p2d * NUM_DIMENSIONS + d2d
                    r1_tile = tl.rand(SEED, PSO_OFFSET_PHILOX + 2 * iteration * NUM_PARTICLES * NUM_DIMENSIONS + roffs)
                    r2_tile = tl.rand(SEED, PSO_OFFSET_PHILOX + (2 * iteration + 1) * NUM_PARTICLES * NUM_DIMENSIONS + roffs)
                new_vel  = _velocity_update(pos_tile, vel_tile, pbest_pos_tile, gbest_tile, r1_tile, r2_tile,
                                            INERTIA_WEIGHT, COGNITIVE_WEIGHT, SOCIAL_WEIGHT)
                pos_tile = pos_tile + new_vel
                tl.store_tensor_descriptor(vel_desc, [dim_offset, pid * BLOCK_SIZE_PARTICLES], tl.trans(new_vel))
                tl.store_tensor_descriptor(pos_desc, [dim_offset, pid * BLOCK_SIZE_PARTICLES], tl.trans(pos_tile))

        else:
            dim_rows = tl.expand_dims(dim_offset + tl.arange(0, BLOCK_SIZE_DIM), 1)
            p_cols   = tl.expand_dims(pid * BLOCK_SIZE_PARTICLES + tl.arange(0, BLOCK_SIZE_PARTICLES), 0)
            pos_tile = tl.trans(tl.load(positions_ptr + dim_rows * NUM_PARTICLES + p_cols))

            if path_block_idx == 0 and not EVAL_ONLY:
                vel_tile       = tl.trans(tl.load(velocities_ptr + dim_rows * NUM_PARTICLES + p_cols))
                pbest_pos_tile = tl.trans(tl.load(pbest_pos_ptr  + dim_rows * NUM_PARTICLES + p_cols))
                dim_idx        = dim_offset + tl.arange(0, BLOCK_SIZE_DIM)
                gbest_tile     = tl.broadcast_to(tl.expand_dims(tl.load(gbest_pos_ptr + dim_idx), 0), [BLOCK_SIZE_PARTICLES, BLOCK_SIZE_DIM])
                if USE_FIXED_RANDOM:
                    r1_tile = tl.trans(tl.load(r1_ptr + dim_rows * NUM_PARTICLES + p_cols))
                    r2_tile = tl.trans(tl.load(r2_ptr + dim_rows * NUM_PARTICLES + p_cols))
                else:
                    p2d     = tl.broadcast_to(tl.expand_dims(pid * BLOCK_SIZE_PARTICLES + tl.arange(0, BLOCK_SIZE_PARTICLES), 1), [BLOCK_SIZE_PARTICLES, BLOCK_SIZE_DIM])
                    d2d     = tl.broadcast_to(tl.expand_dims(dim_idx, 0), [BLOCK_SIZE_PARTICLES, BLOCK_SIZE_DIM])
                    roffs   = p2d * NUM_DIMENSIONS + d2d
                    r1_tile = tl.rand(SEED, PSO_OFFSET_PHILOX + 2 * iteration * NUM_PARTICLES * NUM_DIMENSIONS + roffs)
                    r2_tile = tl.rand(SEED, PSO_OFFSET_PHILOX + (2 * iteration + 1) * NUM_PARTICLES * NUM_DIMENSIONS + roffs)
                new_vel  = _velocity_update(pos_tile, vel_tile, pbest_pos_tile, gbest_tile, r1_tile, r2_tile,
                                            INERTIA_WEIGHT, COGNITIVE_WEIGHT, SOCIAL_WEIGHT)
                pos_tile = pos_tile + new_vel
                tl.store(velocities_ptr + dim_rows * NUM_PARTICLES + p_cols, tl.trans(new_vel))
                tl.store(positions_ptr  + dim_rows * NUM_PARTICLES + p_cols, tl.trans(pos_tile))

        # ── path tile generation ──────────────────────────────────────────────
        if NUM_BW_PATHS == 0:
            if BLOCK_SIZE_DIM == 1:
                path_tile, current_lnS = _gen_path_1d(
                    current_lnS, path_offs, path_block_idx, dim_offset, drift_l2, vol_l2,
                    SEED, MC_OFFSET_PHILOX, BLOCK_SIZE_PATHS, NUM_DIMENSIONS, USE_ANTITHETIC)
            else:
                path_tile, current_lnS = _gen_path_nd(
                    current_lnS, path_offs, path_block_idx, dim_offset, drift_l2, vol_l2,
                    SEED, MC_OFFSET_PHILOX, BLOCK_SIZE_PATHS, BLOCK_SIZE_DIM, NUM_DIMENSIONS, USE_ANTITHETIC)

        elif NUM_COMPUTE_PATH_BLOCKS == 0:
            path_tile = tl.trans(tl.load_tensor_descriptor(path_desc, [dim_offset, path_block_idx * BLOCK_SIZE_PATHS]))

        else:
            if is_compute:
                if BLOCK_SIZE_DIM == 1:
                    path_tile, current_lnS = _gen_path_1d(
                        current_lnS, path_offs, path_block_idx, dim_offset, drift_l2, vol_l2,
                        SEED, MC_OFFSET_PHILOX, BLOCK_SIZE_PATHS, NUM_DIMENSIONS, USE_ANTITHETIC)
                else:
                    path_tile, current_lnS = _gen_path_nd(
                        current_lnS, path_offs, path_block_idx, dim_offset, drift_l2, vol_l2,
                        SEED, MC_OFFSET_PHILOX, BLOCK_SIZE_PATHS, BLOCK_SIZE_DIM, NUM_DIMENSIONS, USE_ANTITHETIC)
            else:
                bw_block_idx = path_block_idx - NUM_COMPUTE_PATH_BLOCKS
                path_tile    = tl.trans(tl.load_tensor_descriptor(path_desc, [dim_offset, bw_block_idx * BLOCK_SIZE_PATHS]))

        # ── payoff evaluation ─────────────────────────────────────────────────
        if BLOCK_SIZE_PARTICLES == 1 and BLOCK_SIZE_DIM == 1:
            if dim_block_idx == NUM_DIMENSIONS // BLOCK_SIZE_DIM - 1:
                terminal_path_acc = tl.reshape(path_tile, [BLOCK_SIZE_PATHS])
            path_vec    = tl.reshape(path_tile, [BLOCK_SIZE_PATHS])
            pos_scalar  = tl.reshape(pos_tile, [1])
            any_ex      = path_vec < pos_scalar if OPTION_TYPE == 1 else path_vec > pos_scalar
            discount    = tl.reshape(discount_tile, [1])
            payoff_vals = tl.maximum(0.0, STRIKE_PRICE - path_vec) * discount if OPTION_TYPE == 1 \
                          else tl.maximum(0.0, path_vec - STRIKE_PRICE) * discount
            just_ex    = any_ex & ~done_acc
            payoff_acc = tl.where(just_ex, payoff_vals, payoff_acc)
            done_acc   = done_acc | any_ex

        elif BLOCK_SIZE_PARTICLES == 1 and BLOCK_SIZE_DIM >= 4:
            if dim_block_idx == NUM_DIMENSIONS // BLOCK_SIZE_DIM - 1:
                last_mask         = tl.broadcast_to(tl.expand_dims(tl.arange(0, BLOCK_SIZE_DIM) == BLOCK_SIZE_DIM - 1, 0), [BLOCK_SIZE_PATHS, BLOCK_SIZE_DIM])
                terminal_path_acc = tl.sum(path_tile * last_mask.to(tl.float32), axis=1)
            pos_vec       = tl.reshape(pos_tile, [BLOCK_SIZE_DIM])
            pos_2d        = tl.broadcast_to(tl.expand_dims(pos_vec, 0), [BLOCK_SIZE_PATHS, BLOCK_SIZE_DIM])
            ex_mask       = (pos_2d > path_tile) if OPTION_TYPE == 1 else (pos_2d < path_tile)
            disc_2d       = tl.broadcast_to(tl.expand_dims(discount_tile, 0), [BLOCK_SIZE_PATHS, BLOCK_SIZE_DIM])
            payoffs_2d    = tl.maximum(0.0, STRIKE_PRICE - path_tile) * disc_2d if OPTION_TYPE == 1 \
                            else tl.maximum(0.0, path_tile - STRIKE_PRICE) * disc_2d
            ex_int_2d     = ex_mask.to(tl.int32)
            payoff_masked = payoffs_2d * ex_int_2d.to(tl.float32)
            block_payoff_1d, block_done_int = tl.reduce((payoff_masked, ex_int_2d), axis=1, combine_fn=_ex_combine)
            block_done    = block_done_int.to(tl.int1)
            just_ex_1d    = block_done & ~done_acc
            payoff_acc    = tl.where(just_ex_1d, block_payoff_1d, payoff_acc)
            done_acc      = done_acc | block_done

        elif BLOCK_SIZE_PARTICLES >= 4 and BLOCK_SIZE_DIM == 1:
            if dim_block_idx == NUM_DIMENSIONS // BLOCK_SIZE_DIM - 1:
                terminal_path_acc = tl.reshape(path_tile, [BLOCK_SIZE_PATHS])
            pos_vec        = tl.reshape(pos_tile,  [BLOCK_SIZE_PARTICLES])
            path_vec       = tl.reshape(path_tile, [BLOCK_SIZE_PATHS])
            pos_2d         = tl.broadcast_to(tl.expand_dims(pos_vec, 0),  [BLOCK_SIZE_PATHS, BLOCK_SIZE_PARTICLES])
            path_2d        = tl.broadcast_to(tl.expand_dims(path_vec, 1), [BLOCK_SIZE_PATHS, BLOCK_SIZE_PARTICLES])
            any_ex_2d      = (pos_2d > path_2d) if OPTION_TYPE == 1 else (pos_2d < path_2d)
            discount       = tl.reshape(discount_tile, [1])
            payoff_vals_2d = tl.maximum(0.0, STRIKE_PRICE - path_2d) * discount if OPTION_TYPE == 1 \
                             else tl.maximum(0.0, path_2d - STRIKE_PRICE) * discount
            just_ex_2d = any_ex_2d & ~done_acc
            payoff_acc = tl.where(just_ex_2d, payoff_vals_2d, payoff_acc)
            done_acc   = done_acc | any_ex_2d

        else:
            if dim_block_idx == NUM_DIMENSIONS // BLOCK_SIZE_DIM - 1:
                last_mask_2d      = tl.broadcast_to(tl.expand_dims(tl.arange(0, BLOCK_SIZE_DIM) == BLOCK_SIZE_DIM - 1, 0), [BLOCK_SIZE_PATHS, BLOCK_SIZE_DIM])
                terminal_path_acc = tl.sum(path_tile * last_mask_2d.to(tl.float32), axis=1)
            pos_3d        = tl.broadcast_to(tl.expand_dims(pos_tile,  0), [BLOCK_SIZE_PATHS, BLOCK_SIZE_PARTICLES, BLOCK_SIZE_DIM])
            path_3d       = tl.broadcast_to(tl.expand_dims(path_tile, 1), [BLOCK_SIZE_PATHS, BLOCK_SIZE_PARTICLES, BLOCK_SIZE_DIM])
            ex_3d         = (pos_3d > path_3d) if OPTION_TYPE == 1 else (pos_3d < path_3d)
            disc_3d       = tl.broadcast_to(tl.expand_dims(tl.expand_dims(discount_tile, 0), 0), [BLOCK_SIZE_PATHS, BLOCK_SIZE_PARTICLES, BLOCK_SIZE_DIM])
            payoffs_3d    = tl.maximum(0.0, STRIKE_PRICE - path_3d) * disc_3d if OPTION_TYPE == 1 \
                            else tl.maximum(0.0, path_3d - STRIKE_PRICE) * disc_3d
            ex_int_3d     = ex_3d.to(tl.int32)
            payoff_masked = payoffs_3d * ex_int_3d.to(tl.float32)
            block_payoff_2d, block_done_int = tl.reduce((payoff_masked, ex_int_3d), axis=2, combine_fn=_ex_combine)
            block_done_2d = block_done_int.to(tl.int1)
            just_ex_2d    = block_done_2d & ~done_acc
            payoff_acc    = tl.where(just_ex_2d, block_payoff_2d, payoff_acc)
            done_acc      = done_acc | block_done_2d

    # ── terminal payoff ───────────────────────────────────────────────────────
    terminal_discount = tl.load(discounts_ptr + NUM_DIMENSIONS - 1)

    if BLOCK_SIZE_PARTICLES == 1:
        term_payoff  = tl.maximum(0.0, STRIKE_PRICE - terminal_path_acc) if OPTION_TYPE == 1 \
                       else tl.maximum(0.0, terminal_path_acc - STRIKE_PRICE)
        payoff_acc   = tl.where(~done_acc, term_payoff * terminal_discount, payoff_acc)
        payoff_accum = payoff_accum + tl.sum(payoff_acc, axis=0, keep_dims=True)
    else:
        term_payoff_2d = tl.broadcast_to(
            tl.expand_dims(tl.maximum(0.0, STRIKE_PRICE - terminal_path_acc) if OPTION_TYPE == 1
                           else tl.maximum(0.0, terminal_path_acc - STRIKE_PRICE), 1),
            [BLOCK_SIZE_PATHS, BLOCK_SIZE_PARTICLES],
        )
        payoff_acc   = tl.where(~done_acc, term_payoff_2d * terminal_discount, payoff_acc)
        payoff_accum = payoff_accum + tl.sum(payoff_acc, axis=0)

    NUM_PATH_BLOCKS = NUM_PATHS // BLOCK_SIZE_PATHS
    tl.store(partial_payoffs_ptr + p_idx * NUM_PATH_BLOCKS + path_block_idx, payoff_accum)
