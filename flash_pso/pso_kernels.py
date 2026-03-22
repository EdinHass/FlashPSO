import triton
import triton.language as tl
from flash_pso.config import get_autotune_configs


@triton.jit
def init_kernel(
    positions_ptr,
    velocities_ptr,
    pbest_costs_ptr,
    pbest_pos_ptr,
    num_dimensions,
    num_particles,
    seed,
    GENERATES_POSITIONS: tl.constexpr,
    GENERATES_VELOCITIES: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    POS_OFFSET_PHILOX: tl.constexpr,
    VEL_OFFSET_PHILOX: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # Decompose flat index into (particle, dim) then compute column-major address.
    # Physical layout is [NUM_DIMENSIONS, NUM_PARTICLES]: element [p, d] is at
    # d * num_particles + p.
    p_idx  = offs // num_dimensions
    d_idx  = offs % num_dimensions
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
    iteration,
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
    OPTION_TYPE:          tl.constexpr,  # 1=Put, 0=Call
    STRIKE_PRICE:         tl.constexpr,
    RISK_FREE_RATE:       tl.constexpr,
    TIME_STEP_SIZE:       tl.constexpr,
    PSO_OFFSET_PHILOX:    tl.constexpr,
    MC_OFFSET_PHILOX:     tl.constexpr,
    EVAL_ONLY:            tl.constexpr,
):
    pid = tl.program_id(0)

    # Particle indices for this block: [BLOCK_SIZE_PARTICLES]
    p_idx = pid * BLOCK_SIZE_PARTICLES + tl.arange(0, BLOCK_SIZE_PARTICLES)

    # ── TMA DESCRIPTORS ───────────────────────────────────────────────────────
    # Physical layout is [NUM_DIMENSIONS, NUM_PARTICLES] with row-major strides
    # [NUM_PARTICLES, 1]. TMA requires last dim to be contiguous (stride=1) — satisfied.
    # Tiles are [BLOCK_SIZE_DIM, BLOCK_SIZE_PARTICLES] and transposed after load to
    # restore the [BLOCK_SIZE_PARTICLES, BLOCK_SIZE_DIM] shape the rest of the kernel
    # expects. BLOCK_SIZE_PARTICLES is large → 16-byte minimum easily met.
    pos_desc = tl.make_tensor_descriptor(
        base=positions_ptr,
        shape=[NUM_DIMENSIONS, NUM_PARTICLES],           # pyright: ignore[reportArgumentType]
        strides=[NUM_PARTICLES, 1],                      # pyright: ignore[reportArgumentType]
        block_shape=[BLOCK_SIZE_DIM, BLOCK_SIZE_PARTICLES],
    )
    vel_desc = tl.make_tensor_descriptor(
        base=velocities_ptr,
        shape=[NUM_DIMENSIONS, NUM_PARTICLES],           # pyright: ignore[reportArgumentType]
        strides=[NUM_PARTICLES, 1],                      # pyright: ignore[reportArgumentType]
        block_shape=[BLOCK_SIZE_DIM, BLOCK_SIZE_PARTICLES],
    )
    pbest_pos_desc = tl.make_tensor_descriptor(
        base=pbest_pos_ptr,
        shape=[NUM_DIMENSIONS, NUM_PARTICLES],           # pyright: ignore[reportArgumentType]
        strides=[NUM_PARTICLES, 1],                      # pyright: ignore[reportArgumentType]
        block_shape=[BLOCK_SIZE_DIM, BLOCK_SIZE_PARTICLES],
    )
    path_desc = tl.make_tensor_descriptor(
        base=st_ptr,
        shape=[NUM_DIMENSIONS, NUM_PATHS],               # pyright: ignore[reportArgumentType]
        strides=[NUM_PATHS, 1],                          # pyright: ignore[reportArgumentType]
        block_shape=[BLOCK_SIZE_DIM, BLOCK_SIZE_PATHS],
    )

    pbest_payoff_block = tl.load(pbest_payoff_ptr + p_idx)  # [BLOCK_SIZE_PARTICLES]
    payoff_accum = tl.zeros([BLOCK_SIZE_PARTICLES], dtype=tl.float32)

    # ── PATH LOOP ─────────────────────────────────────────────────────────────
    for path_block_idx in range(NUM_PATHS // BLOCK_SIZE_PATHS):

        # [BLOCK_SIZE_PATHS, BLOCK_SIZE_PARTICLES] — tracks first exercise per (path, particle)
        done_2d   = tl.zeros([BLOCK_SIZE_PATHS, BLOCK_SIZE_PARTICLES], dtype=tl.int1)
        payoff_2d = tl.zeros([BLOCK_SIZE_PATHS, BLOCK_SIZE_PARTICLES], dtype=tl.float32)
        terminal_path_acc = tl.zeros([BLOCK_SIZE_PATHS], dtype=tl.float32)

        if COMPUTE_ON_THE_FLY:
            drift       = (r - 0.5 * sigma * sigma) * dt
            vol_scaler  = sigma * tl.sqrt(dt)
            current_lnS = tl.full([BLOCK_SIZE_PATHS], tl.log(S0), dtype=tl.float32)
            path_offs   = path_block_idx * BLOCK_SIZE_PATHS + tl.arange(0, BLOCK_SIZE_PATHS)

        # ── DIM LOOP ──────────────────────────────────────────────────────────
        for dim_block_idx in range(NUM_DIMENSIONS // BLOCK_SIZE_DIM):
            dim_offset = dim_block_idx * BLOCK_SIZE_DIM

            # ── 1. TMA load: positions tile, transposed to [BLOCK_SIZE_PARTICLES, BLOCK_SIZE_DIM]
            # Descriptor coords are [dim_offset, particle_offset] matching physical [DIM, PARTICLES].
            # tl.trans restores [BLOCK_SIZE_PARTICLES, BLOCK_SIZE_DIM] for the rest of the kernel.
            pos_tile = tl.trans(tl.load_tensor_descriptor(pos_desc, [dim_offset, pid * BLOCK_SIZE_PARTICLES]))

            # ── 2. PSO update — only on first path block ──────────────────────
            if path_block_idx == 0 and not EVAL_ONLY:
                vel_tile       = tl.trans(tl.load_tensor_descriptor(vel_desc,       [dim_offset, pid * BLOCK_SIZE_PARTICLES]))
                pbest_pos_tile = tl.trans(tl.load_tensor_descriptor(pbest_pos_desc, [dim_offset, pid * BLOCK_SIZE_PARTICLES]))

                dim_idx    = dim_offset + tl.arange(0, BLOCK_SIZE_DIM)
                gbest_tile = tl.load(gbest_pos_ptr + dim_idx)                   # [BLOCK_SIZE_DIM]

                p_offs   = (pid * BLOCK_SIZE_PARTICLES + tl.arange(0, BLOCK_SIZE_PARTICLES))[:, None]
                rng_offs = p_offs * NUM_DIMENSIONS + dim_idx[None, :]

                r1 = tl.rand(SEED, PSO_OFFSET_PHILOX + 2 * iteration * NUM_PARTICLES * NUM_DIMENSIONS + rng_offs)
                r2 = tl.rand(SEED, PSO_OFFSET_PHILOX + (2 * iteration + 1) * NUM_PARTICLES * NUM_DIMENSIONS + rng_offs)

                new_vel_tile = (INERTIA_WEIGHT   * vel_tile
                                + COGNITIVE_WEIGHT * r1 * (pbest_pos_tile - pos_tile)
                                + SOCIAL_WEIGHT    * r2 * (gbest_tile[None, :] - pos_tile))
                pos_tile = pos_tile + new_vel_tile

                # Transpose back to [BLOCK_SIZE_DIM, BLOCK_SIZE_PARTICLES] before storing
                tl.store_tensor_descriptor(vel_desc, [dim_offset, pid * BLOCK_SIZE_PARTICLES], tl.trans(new_vel_tile))
                tl.store_tensor_descriptor(pos_desc, [dim_offset, pid * BLOCK_SIZE_PARTICLES], tl.trans(pos_tile))

            # ── 3. Path tile: load or compute ─────────────────────────────────
            if COMPUTE_ON_THE_FLY:
                d_offs      = tl.arange(0, BLOCK_SIZE_DIM)[None, :]
                rng_offs_mc = path_offs[:, None] * NUM_DIMENSIONS + dim_offset + d_offs
                Z_2d        = tl.randn(SEED, MC_OFFSET_PHILOX + rng_offs_mc)

                Z_cumsum = tl.cumsum(Z_2d, axis=1)
                steps_2d = (tl.arange(0, BLOCK_SIZE_DIM) + 1)[None, :]
                lnS_2d   = current_lnS[:, None] + (drift * steps_2d) + (vol_scaler * Z_cumsum)
                path_tile = tl.exp(lnS_2d)                                      # [BLOCK_SIZE_PATHS, BLOCK_SIZE_DIM]

                last_one_hot = (tl.arange(0, BLOCK_SIZE_DIM) == BLOCK_SIZE_DIM - 1)[None, :]
                current_lnS  = tl.sum(lnS_2d * last_one_hot.to(tl.float32), axis=1)
            else:
                # Load [BLOCK_SIZE_DIM, BLOCK_SIZE_PATHS], transpose to [BLOCK_SIZE_PATHS, BLOCK_SIZE_DIM]
                path_tile = tl.trans(tl.load_tensor_descriptor(path_desc, [dim_offset, path_block_idx * BLOCK_SIZE_PATHS]))

            # ── 4. Terminal path extraction (last dim block only) ─────────────
            if dim_block_idx == NUM_DIMENSIONS // BLOCK_SIZE_DIM - 1:
                term_one_hot      = (tl.arange(0, BLOCK_SIZE_DIM) == BLOCK_SIZE_DIM - 1)[None, :]
                terminal_path_acc = tl.sum(path_tile * term_one_hot.to(tl.float32), axis=1)

            # ── 5. Exercise detection via 3D broadcast ────────────────────────
            # pos_3d:  [1,               BLOCK_SIZE_PARTICLES, BLOCK_SIZE_DIM]
            # path_3d: [BLOCK_SIZE_PATHS, 1,                   BLOCK_SIZE_DIM]
            # ex_3d:   [BLOCK_SIZE_PATHS, BLOCK_SIZE_PARTICLES, BLOCK_SIZE_DIM]
            pos_3d  = pos_tile[None, :, :]
            path_3d = path_tile[:, None, :]

            if OPTION_TYPE == 1:
                ex_3d = (pos_3d > path_3d)
            else:
                ex_3d = (pos_3d < path_3d)

            ex_int_3d = ex_3d.to(tl.int32)

            first_ex_idx_2d = tl.argmax(ex_int_3d, axis=2)  # [BLOCK_SIZE_PATHS, BLOCK_SIZE_PARTICLES]
            any_ex_2d       = tl.max(ex_int_3d, axis=2) == 1

            # ── 6. Extract path value at the exercise boundary ────────────────
            dim_idx_3d = tl.arange(0, BLOCK_SIZE_DIM)[None, None, :]
            one_hot_3d = (dim_idx_3d == first_ex_idx_2d[:, :, None]).to(tl.float32)
            ex_path_2d = tl.sum(path_3d * one_hot_3d, axis=2)               # [BLOCK_SIZE_PATHS, BLOCK_SIZE_PARTICLES]

            # ── 7. Compute payoff and update accumulator ──────────────────────
            actual_dim_2d = dim_offset + first_ex_idx_2d                    # [BLOCK_SIZE_PATHS, BLOCK_SIZE_PARTICLES]
            discount_2d   = tl.exp(-RISK_FREE_RATE * TIME_STEP_SIZE * (actual_dim_2d + 1))

            if OPTION_TYPE == 1:
                payoff_vals_2d = tl.maximum(0.0, STRIKE_PRICE - ex_path_2d) * discount_2d
            else:
                payoff_vals_2d = tl.maximum(0.0, ex_path_2d - STRIKE_PRICE) * discount_2d

            just_ex_2d = any_ex_2d & ~done_2d
            payoff_2d  = tl.where(just_ex_2d, payoff_vals_2d, payoff_2d)
            done_2d    = done_2d | any_ex_2d

        # ── END DIM LOOP ──────────────────────────────────────────────────────
        terminal_discount = tl.exp(-RISK_FREE_RATE * TIME_STEP_SIZE * NUM_DIMENSIONS)
        if OPTION_TYPE == 1:
            term_payoff_2d = tl.maximum(0.0, STRIKE_PRICE - terminal_path_acc)[:, None]
        else:
            term_payoff_2d = tl.maximum(0.0, terminal_path_acc - STRIKE_PRICE)[:, None]

        payoff_2d    = tl.where(~done_2d, term_payoff_2d * terminal_discount, payoff_2d)
        payoff_accum = payoff_accum + tl.sum(payoff_2d, axis=0)  # reduce paths → [BLOCK_SIZE_PARTICLES]

    # ── WRITE-BACK ────────────────────────────────────────────────────────────
    payoff_avg = payoff_accum / NUM_PATHS                                    # [BLOCK_SIZE_PARTICLES]

    should_update      = payoff_avg > pbest_payoff_block
    pbest_payoff_block = tl.where(should_update, payoff_avg, pbest_payoff_block)
    tl.store(pbest_payoff_ptr + p_idx, pbest_payoff_block)

    for dim_block_idx in range(NUM_DIMENSIONS // BLOCK_SIZE_DIM):
        dim_offset     = dim_block_idx * BLOCK_SIZE_DIM
        pbest_pos_tile = tl.trans(tl.load_tensor_descriptor(pbest_pos_desc, [dim_offset, pid * BLOCK_SIZE_PARTICLES]))
        new_pos_tile   = tl.trans(tl.load_tensor_descriptor(pos_desc,       [dim_offset, pid * BLOCK_SIZE_PARTICLES]))
        pbest_pos_tile = tl.where(should_update[:, None], new_pos_tile, pbest_pos_tile)
        tl.store_tensor_descriptor(pbest_pos_desc, [dim_offset, pid * BLOCK_SIZE_PARTICLES], tl.trans(pbest_pos_tile))
