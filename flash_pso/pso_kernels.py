import triton
import triton.language as tl
from flash_pso.config import get_autotune_configs


# kernel to initialize particles' initial states
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
    pid = tl.program_id(0)
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
    key=["NUM_PARTICLES", "NUM_PATHS", "NUM_DIMENSIONS"],
)
@triton.jit
def pso_kernel(
    positions_ptr,      # physical [NUM_DIMENSIONS, NUM_PARTICLES], strides [NUM_PARTICLES, 1]
    velocities_ptr,     # physical [NUM_DIMENSIONS, NUM_PARTICLES], strides [NUM_PARTICLES, 1]
    pbest_payoff_ptr,   # [NUM_PARTICLES]
    pbest_pos_ptr,      # physical [NUM_DIMENSIONS, NUM_PARTICLES], strides [NUM_PARTICLES, 1]
    gbest_payoff_ptr,   # [1]
    gbest_pos_ptr,      # [NUM_DIMENSIONS]
    st_ptr,             # physical [NUM_DIMENSIONS, NUM_PATHS], strides [NUM_PATHS, 1]
                        # or empty if COMPUTE_ON_THE_FLY
    r1_ptr,             # physical [NUM_DIMENSIONS, NUM_PARTICLES] — used when USE_FIXED_RANDOM
    r2_ptr,             # physical [NUM_DIMENSIONS, NUM_PARTICLES] — used when USE_FIXED_RANDOM
    iteration,          # used when not USE_FIXED_RANDOM to key Philox counter
    S0, r, sigma, dt,
    SEED:                 tl.constexpr,
    INERTIA_WEIGHT:       tl.constexpr,
    COGNITIVE_WEIGHT:     tl.constexpr,
    SOCIAL_WEIGHT:        tl.constexpr,
    NUM_DIMENSIONS:       tl.constexpr,
    NUM_PARTICLES:        tl.constexpr,
    NUM_PATHS:            tl.constexpr,
    COMPUTE_ON_THE_FLY:   tl.constexpr,
    BLOCK_SIZE_PARTICLES: tl.constexpr,
    BLOCK_SIZE_PATHS:     tl.constexpr,
    BLOCK_SIZE_DIM:       tl.constexpr,
    OPTION_TYPE:          tl.constexpr,
    STRIKE_PRICE:         tl.constexpr,
    RISK_FREE_RATE:       tl.constexpr,
    TIME_STEP_SIZE:       tl.constexpr,
    PSO_OFFSET_PHILOX:    tl.constexpr,
    MC_OFFSET_PHILOX:     tl.constexpr,
    EVAL_ONLY:            tl.constexpr,
    USE_FIXED_RANDOM:     tl.constexpr,
):
    pid = tl.program_id(0)

    p_idx = pid * BLOCK_SIZE_PARTICLES + tl.arange(0, BLOCK_SIZE_PARTICLES)

    # tma descs only when BLOCK_SIZE_PARTICLES >= 4, otherwise the overhead of setting up the descriptor and transposes outweighs the benefits
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

    path_desc = tl.make_tensor_descriptor(
        base=st_ptr,
        shape=[NUM_DIMENSIONS, NUM_PATHS],               # pyright: ignore[reportArgumentType]
        strides=[NUM_PATHS, 1],                          # pyright: ignore[reportArgumentType]
        block_shape=[BLOCK_SIZE_DIM, BLOCK_SIZE_PATHS],
    )

    pbest_payoff_block = tl.load(pbest_payoff_ptr + p_idx)
    payoff_accum = tl.zeros([BLOCK_SIZE_PARTICLES], dtype=tl.float32)

    # iterate over path blocks
    for path_block_idx in range(NUM_PATHS // BLOCK_SIZE_PATHS):

        if BLOCK_SIZE_PARTICLES == 1:
            done_acc   = tl.zeros([BLOCK_SIZE_PATHS], dtype=tl.int1)
            payoff_acc = tl.zeros([BLOCK_SIZE_PATHS], dtype=tl.float32)
        else:
            done_acc   = tl.zeros([BLOCK_SIZE_PATHS, BLOCK_SIZE_PARTICLES], dtype=tl.int1)
            payoff_acc = tl.zeros([BLOCK_SIZE_PATHS, BLOCK_SIZE_PARTICLES], dtype=tl.float32)

        terminal_path_acc = tl.zeros([BLOCK_SIZE_PATHS], dtype=tl.float32)

        if COMPUTE_ON_THE_FLY:
            drift       = (r - 0.5 * sigma * sigma) * dt
            vol_scaler  = sigma * tl.sqrt(dt)
            current_lnS = tl.full([BLOCK_SIZE_PATHS], tl.log(S0), dtype=tl.float32)
            path_offs   = path_block_idx * BLOCK_SIZE_PATHS + tl.arange(0, BLOCK_SIZE_PATHS)

        # iterate over dimension blocks
        for dim_block_idx in range(NUM_DIMENSIONS // BLOCK_SIZE_DIM):
            dim_offset = dim_block_idx * BLOCK_SIZE_DIM

            # if large enough block size, load a tile of particles' positions into registers for reuse across paths. Otherwise, load on demand within the path loop.
            if BLOCK_SIZE_PARTICLES >= 4:
                pos_tile = tl.trans(tl.load_tensor_descriptor(pos_desc, [dim_offset, pid * BLOCK_SIZE_PARTICLES]))

                if path_block_idx == 0 and not EVAL_ONLY:
                    vel_tile       = tl.trans(tl.load_tensor_descriptor(vel_desc,       [dim_offset, pid * BLOCK_SIZE_PARTICLES]))
                    pbest_pos_tile = tl.trans(tl.load_tensor_descriptor(pbest_pos_desc, [dim_offset, pid * BLOCK_SIZE_PARTICLES]))
                    dim_idx        = dim_offset + tl.arange(0, BLOCK_SIZE_DIM)
                    gbest_tile     = tl.load(gbest_pos_ptr + dim_idx)

                    if USE_FIXED_RANDOM:
                        r1_tile = tl.trans(tl.load_tensor_descriptor(r1_desc, [dim_offset, pid * BLOCK_SIZE_PARTICLES]))
                        r2_tile = tl.trans(tl.load_tensor_descriptor(r2_desc, [dim_offset, pid * BLOCK_SIZE_PARTICLES]))
                    else:
                        p_offs   = (pid * BLOCK_SIZE_PARTICLES + tl.arange(0, BLOCK_SIZE_PARTICLES))[:, None]
                        rng_offs = p_offs * NUM_DIMENSIONS + dim_idx[None, :]
                        r1_tile  = tl.rand(SEED, PSO_OFFSET_PHILOX + 2 * iteration * NUM_PARTICLES * NUM_DIMENSIONS + rng_offs)
                        r2_tile  = tl.rand(SEED, PSO_OFFSET_PHILOX + (2 * iteration + 1) * NUM_PARTICLES * NUM_DIMENSIONS + rng_offs)

                    new_vel_tile = (INERTIA_WEIGHT   * vel_tile
                                    + COGNITIVE_WEIGHT * r1_tile * (pbest_pos_tile - pos_tile)
                                    + SOCIAL_WEIGHT    * r2_tile * (gbest_tile[None, :] - pos_tile))
                    pos_tile = pos_tile + new_vel_tile
                    tl.store_tensor_descriptor(vel_desc, [dim_offset, pid * BLOCK_SIZE_PARTICLES], tl.trans(new_vel_tile))
                    tl.store_tensor_descriptor(pos_desc, [dim_offset, pid * BLOCK_SIZE_PARTICLES], tl.trans(pos_tile))

            else:
                dim_rows = (dim_offset + tl.arange(0, BLOCK_SIZE_DIM))[:, None]
                p_cols   = (pid * BLOCK_SIZE_PARTICLES + tl.arange(0, BLOCK_SIZE_PARTICLES))[None, :]
                pos_tile = tl.trans(tl.load(positions_ptr + dim_rows * NUM_PARTICLES + p_cols))

                if path_block_idx == 0 and not EVAL_ONLY:
                    vel_tile       = tl.trans(tl.load(velocities_ptr + dim_rows * NUM_PARTICLES + p_cols))
                    pbest_pos_tile = tl.trans(tl.load(pbest_pos_ptr  + dim_rows * NUM_PARTICLES + p_cols))
                    dim_idx        = dim_offset + tl.arange(0, BLOCK_SIZE_DIM)
                    gbest_tile     = tl.load(gbest_pos_ptr + dim_idx)

                    if USE_FIXED_RANDOM:
                        r1_tile = tl.trans(tl.load(r1_ptr + dim_rows * NUM_PARTICLES + p_cols))
                        r2_tile = tl.trans(tl.load(r2_ptr + dim_rows * NUM_PARTICLES + p_cols))
                    else:
                        p_offs   = (pid * BLOCK_SIZE_PARTICLES + tl.arange(0, BLOCK_SIZE_PARTICLES))[:, None]
                        rng_offs = p_offs * NUM_DIMENSIONS + dim_idx[None, :]
                        r1_tile  = tl.rand(SEED, PSO_OFFSET_PHILOX + 2 * iteration * NUM_PARTICLES * NUM_DIMENSIONS + rng_offs)
                        r2_tile  = tl.rand(SEED, PSO_OFFSET_PHILOX + (2 * iteration + 1) * NUM_PARTICLES * NUM_DIMENSIONS + rng_offs)

                    new_vel_tile = (INERTIA_WEIGHT   * vel_tile
                                    + COGNITIVE_WEIGHT * r1_tile * (pbest_pos_tile - pos_tile)
                                    + SOCIAL_WEIGHT    * r2_tile * (gbest_tile[None, :] - pos_tile))
                    pos_tile = pos_tile + new_vel_tile
                    tl.store(velocities_ptr + dim_rows * NUM_PARTICLES + p_cols, tl.trans(new_vel_tile))
                    tl.store(positions_ptr  + dim_rows * NUM_PARTICLES + p_cols, tl.trans(pos_tile))

            # computing or loading the path tile
            if COMPUTE_ON_THE_FLY:
                d_offs       = tl.arange(0, BLOCK_SIZE_DIM)[None, :]
                rng_offs_mc  = path_offs[:, None] * NUM_DIMENSIONS + dim_offset + d_offs
                Z_2d         = tl.randn(SEED, MC_OFFSET_PHILOX + rng_offs_mc)
                Z_cumsum     = tl.cumsum(Z_2d, axis=1)
                steps_2d     = (tl.arange(0, BLOCK_SIZE_DIM) + 1)[None, :]
                lnS_2d       = current_lnS[:, None] + (drift * steps_2d) + (vol_scaler * Z_cumsum)
                path_tile    = tl.exp(lnS_2d)
                last_one_hot = (tl.arange(0, BLOCK_SIZE_DIM) == BLOCK_SIZE_DIM - 1)[None, :]
                current_lnS  = tl.sum(lnS_2d * last_one_hot.to(tl.float32), axis=1)
            else:
                path_tile = tl.trans(tl.load_tensor_descriptor(path_desc, [dim_offset, path_block_idx * BLOCK_SIZE_PATHS]))

            # simple case where each thread handles one particle and one dimension → no reduction needed within the block, just compare directly to the strike price and accumulate payoffs.
            if BLOCK_SIZE_PARTICLES == 1 and BLOCK_SIZE_DIM == 1:
                if dim_block_idx == NUM_DIMENSIONS // BLOCK_SIZE_DIM - 1:
                    terminal_path_acc = tl.reshape(path_tile, [BLOCK_SIZE_PATHS])
                path_vec    = tl.reshape(path_tile, [BLOCK_SIZE_PATHS])
                pos_scalar  = tl.reshape(pos_tile, [1])
                any_ex      = path_vec < pos_scalar if OPTION_TYPE == 1 else path_vec > pos_scalar
                discount    = tl.exp(-RISK_FREE_RATE * TIME_STEP_SIZE * (dim_offset + 1))
                payoff_vals = tl.maximum(0.0, STRIKE_PRICE - path_vec) * discount if OPTION_TYPE == 1 \
                              else tl.maximum(0.0, path_vec - STRIKE_PRICE) * discount
                just_ex     = any_ex & ~done_acc
                payoff_acc  = tl.where(just_ex, payoff_vals, payoff_acc)
                done_acc    = done_acc | any_ex

            # more complex case where we compare more dims at one time
            elif BLOCK_SIZE_PARTICLES == 1 and BLOCK_SIZE_DIM >= 4:
                if dim_block_idx == NUM_DIMENSIONS // BLOCK_SIZE_DIM - 1:
                    term_one_hot      = (tl.arange(0, BLOCK_SIZE_DIM) == BLOCK_SIZE_DIM - 1)[None, :]
                    terminal_path_acc = tl.sum(path_tile * term_one_hot.to(tl.float32), axis=1)
                pos_vec         = tl.reshape(pos_tile, [BLOCK_SIZE_DIM])
                ex_2d           = (pos_vec[None, :] > path_tile) if OPTION_TYPE == 1 else (pos_vec[None, :] < path_tile)
                ex_int_2d       = ex_2d.to(tl.int32)
                first_ex_idx_1d = tl.argmax(ex_int_2d, axis=1)
                any_ex_1d       = tl.max(ex_int_2d, axis=1) == 1
                one_hot_2d      = (tl.arange(0, BLOCK_SIZE_DIM)[None, :] == first_ex_idx_1d[:, None]).to(tl.float32)
                ex_path_1d      = tl.sum(path_tile * one_hot_2d, axis=1)
                discount_1d     = tl.exp(-RISK_FREE_RATE * TIME_STEP_SIZE * (dim_offset + first_ex_idx_1d + 1))
                payoff_vals_1d  = tl.maximum(0.0, STRIKE_PRICE - ex_path_1d) * discount_1d if OPTION_TYPE == 1 \
                                  else tl.maximum(0.0, ex_path_1d - STRIKE_PRICE) * discount_1d
                just_ex_1d      = any_ex_1d & ~done_acc
                payoff_acc      = tl.where(just_ex_1d, payoff_vals_1d, payoff_acc)
                done_acc        = done_acc | any_ex_1d

            # more complex case wehre we compare more particles at one time
            elif BLOCK_SIZE_PARTICLES >= 4 and BLOCK_SIZE_DIM == 1:
                if dim_block_idx == NUM_DIMENSIONS // BLOCK_SIZE_DIM - 1:
                    terminal_path_acc = tl.reshape(path_tile, [BLOCK_SIZE_PATHS])
                pos_vec        = tl.reshape(pos_tile,  [BLOCK_SIZE_PARTICLES])
                path_vec       = tl.reshape(path_tile, [BLOCK_SIZE_PATHS])
                any_ex_2d      = (pos_vec[None, :] > path_vec[:, None]) if OPTION_TYPE == 1 \
                                 else (pos_vec[None, :] < path_vec[:, None])
                discount       = tl.exp(-RISK_FREE_RATE * TIME_STEP_SIZE * (dim_offset + 1))
                payoff_vals_2d = tl.maximum(0.0, STRIKE_PRICE - path_vec[:, None]) * discount if OPTION_TYPE == 1 \
                                 else tl.maximum(0.0, path_vec[:, None] - STRIKE_PRICE) * discount
                just_ex_2d     = any_ex_2d & ~done_acc
                payoff_acc     = tl.where(just_ex_2d, payoff_vals_2d, payoff_acc)
                done_acc       = done_acc | any_ex_2d

            # most complex case where we compare multiple particles and multiple dimensions at the same time, requires reductions causing register pressure but amortized by the reuse of the loaded tiles
            else:
                if dim_block_idx == NUM_DIMENSIONS // BLOCK_SIZE_DIM - 1:
                    term_one_hot      = (tl.arange(0, BLOCK_SIZE_DIM) == BLOCK_SIZE_DIM - 1)[None, :]
                    terminal_path_acc = tl.sum(path_tile * term_one_hot.to(tl.float32), axis=1)
                pos_3d          = pos_tile[None, :, :]
                path_3d         = path_tile[:, None, :]
                ex_3d           = (pos_3d > path_3d) if OPTION_TYPE == 1 else (pos_3d < path_3d)
                ex_int_3d       = ex_3d.to(tl.int32)
                first_ex_idx_2d = tl.argmax(ex_int_3d, axis=2)
                any_ex_2d       = tl.max(ex_int_3d, axis=2) == 1
                one_hot_3d      = (tl.arange(0, BLOCK_SIZE_DIM)[None, None, :] == first_ex_idx_2d[:, :, None]).to(tl.float32)
                ex_path_2d      = tl.sum(path_3d * one_hot_3d, axis=2)
                discount_2d     = tl.exp(-RISK_FREE_RATE * TIME_STEP_SIZE * (dim_offset + first_ex_idx_2d + 1))
                payoff_vals_2d  = tl.maximum(0.0, STRIKE_PRICE - ex_path_2d) * discount_2d if OPTION_TYPE == 1 \
                                  else tl.maximum(0.0, ex_path_2d - STRIKE_PRICE) * discount_2d
                just_ex_2d      = any_ex_2d & ~done_acc
                payoff_acc      = tl.where(just_ex_2d, payoff_vals_2d, payoff_acc)
                done_acc        = done_acc | any_ex_2d

        # dim loop ends here

        terminal_discount = tl.exp(-RISK_FREE_RATE * TIME_STEP_SIZE * NUM_DIMENSIONS)

        if BLOCK_SIZE_PARTICLES == 1:
            term_payoff  = tl.maximum(0.0, STRIKE_PRICE - terminal_path_acc) if OPTION_TYPE == 1 \
                           else tl.maximum(0.0, terminal_path_acc - STRIKE_PRICE)
            payoff_acc   = tl.where(~done_acc, term_payoff * terminal_discount, payoff_acc)
            payoff_accum = payoff_accum + tl.sum(payoff_acc, axis=0, keep_dims=True)
        else:
            term_payoff_2d = tl.maximum(0.0, STRIKE_PRICE - terminal_path_acc)[:, None] if OPTION_TYPE == 1 \
                             else tl.maximum(0.0, terminal_path_acc - STRIKE_PRICE)[:, None]
            payoff_acc   = tl.where(~done_acc, term_payoff_2d * terminal_discount, payoff_acc)
            payoff_accum = payoff_accum + tl.sum(payoff_acc, axis=0)

    # path loop ends here

    payoff_avg = payoff_accum / NUM_PATHS

    should_update      = payoff_avg > pbest_payoff_block
    pbest_payoff_block = tl.where(should_update, payoff_avg, pbest_payoff_block)
    tl.store(pbest_payoff_ptr + p_idx, pbest_payoff_block)

    # writeback of pbest positions if we improved on the payoff
    for dim_block_idx in range(NUM_DIMENSIONS // BLOCK_SIZE_DIM):
        dim_offset = dim_block_idx * BLOCK_SIZE_DIM
        if BLOCK_SIZE_PARTICLES >= 4:
            pbest_pos_tile = tl.trans(tl.load_tensor_descriptor(pbest_pos_desc, [dim_offset, pid * BLOCK_SIZE_PARTICLES]))
            new_pos_tile   = tl.trans(tl.load_tensor_descriptor(pos_desc,       [dim_offset, pid * BLOCK_SIZE_PARTICLES]))
            pbest_pos_tile = tl.where(should_update[:, None], new_pos_tile, pbest_pos_tile)
            tl.store_tensor_descriptor(pbest_pos_desc, [dim_offset, pid * BLOCK_SIZE_PARTICLES], tl.trans(pbest_pos_tile))
        else:
            dim_rows       = (dim_offset + tl.arange(0, BLOCK_SIZE_DIM))[:, None]
            p_cols         = (pid * BLOCK_SIZE_PARTICLES + tl.arange(0, BLOCK_SIZE_PARTICLES))[None, :]
            pbest_pos_tile = tl.trans(tl.load(pbest_pos_ptr + dim_rows * NUM_PARTICLES + p_cols))
            new_pos_tile   = tl.trans(tl.load(positions_ptr + dim_rows * NUM_PARTICLES + p_cols))
            pbest_pos_tile = tl.where(should_update[:, None], new_pos_tile, pbest_pos_tile)
            tl.store(pbest_pos_ptr + dim_rows * NUM_PARTICLES + p_cols, tl.trans(pbest_pos_tile))
