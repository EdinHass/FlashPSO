import triton
import triton.language as tl
from flash_pso.config import get_autotune_configs

@triton.jit
def _velocity_update(pos, vel, pbest, gbest, r1, r2,
                     W: tl.constexpr, C1: tl.constexpr, C2: tl.constexpr):
    return W * vel + C1 * r1 * (pbest - pos) + C2 * r2 * (gbest - pos)

@triton.jit
def init_kernel(positions_ptr, ln_positions_ptr, velocities_ptr, pbest_costs_ptr, pbest_pos_ptr, r1_ptr, r2_ptr, num_dimensions, num_particles, seed, GENERATES_POSITIONS: tl.constexpr, GENERATES_VELOCITIES: tl.constexpr, BLOCK_SIZE: tl.constexpr, POS_OFFSET_PHILOX: tl.constexpr, VEL_OFFSET_PHILOX: tl.constexpr, R1_OFFSET_PHILOX: tl.constexpr, R2_OFFSET_PHILOX: tl.constexpr, USE_FIXED_RANDOM: tl.constexpr):
    pid = tl.program_id(0); offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    p_idx = offs // num_dimensions; d_idx = offs % num_dimensions; cm_offs = d_idx * num_particles + p_idx
    
    if GENERATES_POSITIONS:
        pos = tl.rand(seed, POS_OFFSET_PHILOX + offs) * 100.0
        tl.store(positions_ptr + cm_offs, pos)
        tl.store(ln_positions_ptr + cm_offs, tl.log2(tl.maximum(pos, 1e-4)))
        tl.store(pbest_pos_ptr + cm_offs, pos)
    else:
        pos = tl.load(positions_ptr + cm_offs)
        tl.store(ln_positions_ptr + cm_offs, tl.log2(tl.maximum(pos, 1e-4)))
        tl.store(pbest_pos_ptr + cm_offs, pos)
        
    if GENERATES_VELOCITIES:
        vel = tl.rand(seed, VEL_OFFSET_PHILOX + offs) * 5.0
        tl.store(velocities_ptr + cm_offs, vel)
    if USE_FIXED_RANDOM:
        tl.store(r1_ptr + cm_offs, tl.rand(seed, R1_OFFSET_PHILOX + offs))
        tl.store(r2_ptr + cm_offs, tl.rand(seed, R2_OFFSET_PHILOX + offs))
    tl.store(pbest_costs_ptr + p_idx, float("-inf"), mask=(d_idx == 0))

@triton.jit
def pso_update_kernel(positions_ptr, ln_positions_ptr, velocities_ptr, pbest_pos_ptr, gbest_pos_ptr, r1_ptr, r2_ptr, iteration, INERTIA_WEIGHT: tl.constexpr, COGNITIVE_WEIGHT: tl.constexpr, SOCIAL_WEIGHT: tl.constexpr, NUM_DIMENSIONS: tl.constexpr, NUM_PARTICLES: tl.constexpr, PSO_OFFSET_PHILOX: tl.constexpr, SEED: tl.constexpr, USE_FIXED_RANDOM: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0); offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE); mask = offs < (NUM_DIMENSIONS * NUM_PARTICLES)
    p_idx = offs % NUM_PARTICLES; d_idx = offs // NUM_PARTICLES
    pos = tl.load(positions_ptr + offs, mask=mask); vel = tl.load(velocities_ptr + offs, mask=mask); pbest = tl.load(pbest_pos_ptr + offs, mask=mask); gbest = tl.load(gbest_pos_ptr + d_idx, mask=mask)
    
    if USE_FIXED_RANDOM:
        r1 = tl.load(r1_ptr + offs, mask=mask); r2 = tl.load(r2_ptr + offs, mask=mask)
    else:
        roffs = p_idx * NUM_DIMENSIONS + d_idx
        r1 = tl.rand(SEED, PSO_OFFSET_PHILOX + 2 * iteration * NUM_PARTICLES * NUM_DIMENSIONS + roffs)
        r2 = tl.rand(SEED, PSO_OFFSET_PHILOX + (2 * iteration + 1) * NUM_PARTICLES * NUM_DIMENSIONS + roffs)
        
    new_vel = INERTIA_WEIGHT * vel + COGNITIVE_WEIGHT * r1 * (pbest - pos) + SOCIAL_WEIGHT * r2 * (gbest - pos)
    new_pos = pos + new_vel
    
    tl.store(velocities_ptr + offs, new_vel, mask=mask)
    tl.store(positions_ptr + offs, new_pos, mask=mask)
    tl.store(ln_positions_ptr + offs, tl.log2(tl.maximum(new_pos, 1e-4)), mask=mask) 

@triton.autotune(configs=get_autotune_configs(), key=["NUM_PARTICLES", "NUM_PATHS", "NUM_DIMENSIONS", "NUM_COMPUTE_PATH_BLOCKS"], warmup=2, rep=3)
@triton.jit
def mc_payoff_kernel(
    ln_positions_ptr, st_ptr, partial_payoffs_ptr, 
    S0, r, sigma, dt,
    SEED: tl.constexpr, NUM_DIMENSIONS: tl.constexpr, NUM_PARTICLES: tl.constexpr, NUM_PATHS: tl.constexpr,
    NUM_COMPUTE_PATH_BLOCKS: tl.constexpr, BLOCK_SIZE_PARTICLES: tl.constexpr, BLOCK_SIZE_PATHS: tl.constexpr, BLOCK_SIZE_DIM: tl.constexpr,
    OPTION_TYPE: tl.constexpr, STRIKE_PRICE: tl.constexpr, MC_OFFSET_PHILOX: tl.constexpr, USE_ANTITHETIC: tl.constexpr,
    WARP_SPECIALIZE: tl.constexpr, LOOP_FLATTEN: tl.constexpr, LOOP_STAGES: tl.constexpr, LOOP_UNROLL: tl.constexpr,
):
    tl.assume(NUM_DIMENSIONS % BLOCK_SIZE_DIM == 0)
    GROUP_SIZE_M = 64
    pid_m, pid_n = tl.swizzle2d(tl.program_id(0), tl.program_id(1), tl.num_programs(0), tl.num_programs(1), GROUP_SIZE_M)
    pid = pid_m; path_block_idx = pid_n

    NUM_BW_PATHS = NUM_PATHS - NUM_COMPUTE_PATH_BLOCKS * BLOCK_SIZE_PATHS
    is_compute = path_block_idx < NUM_COMPUTE_PATH_BLOCKS

    # Base 1D Pointers
    p_idx = tl.max_contiguous(tl.multiple_of(pid * BLOCK_SIZE_PARTICLES + tl.arange(0, BLOCK_SIZE_PARTICLES), BLOCK_SIZE_PARTICLES), BLOCK_SIZE_PARTICLES)
    path_offs = tl.max_contiguous(tl.multiple_of(path_block_idx * BLOCK_SIZE_PATHS + tl.arange(0, BLOCK_SIZE_PATHS), BLOCK_SIZE_PATHS), BLOCK_SIZE_PATHS)

    # Pre-calculate Bandwidth Offsets
    NUM_BW_PATHS_SAFE = tl.constexpr(NUM_BW_PATHS if NUM_BW_PATHS > 0 else BLOCK_SIZE_PATHS)
    bw_block_idx = path_block_idx - NUM_COMPUTE_PATH_BLOCKS
    bw_path_offs = bw_block_idx * BLOCK_SIZE_PATHS + tl.arange(0, BLOCK_SIZE_PATHS)
    # Mask ensures we don't invalid-load if the compiler traces the else-branch during a compute block
    bw_mask = (bw_path_offs >= 0) & (bw_path_offs < NUM_BW_PATHS_SAFE)

    # ── DEFERRED PAYOFF TRACKERS ─────────────────────────────────────────────
    payoff_accum = tl.zeros([BLOCK_SIZE_PARTICLES], dtype=tl.float32)

    if BLOCK_SIZE_PARTICLES == 1:
        done_acc    = tl.zeros([BLOCK_SIZE_PATHS], dtype=tl.int1)
        ex_lnS_acc  = tl.zeros([BLOCK_SIZE_PATHS], dtype=tl.float32)
        ex_step_acc = tl.full([BLOCK_SIZE_PATHS], -1, dtype=tl.int32)
    else:
        done_acc    = tl.zeros([BLOCK_SIZE_PATHS, BLOCK_SIZE_PARTICLES], dtype=tl.int1)
        ex_lnS_acc  = tl.zeros([BLOCK_SIZE_PATHS, BLOCK_SIZE_PARTICLES], dtype=tl.float32)
        ex_step_acc = tl.full([BLOCK_SIZE_PATHS, BLOCK_SIZE_PARTICLES], -1, dtype=tl.int32)

    LOG2E    = 1.4426950408889634
    drift_l2 = (r - 0.5 * sigma * sigma) * dt * LOG2E
    vol_l2   = sigma * tl.sqrt(dt) * LOG2E
    current_lnS = tl.full([BLOCK_SIZE_PATHS], tl.log(S0) * LOG2E, dtype=tl.float32)
    
    # Base offsets for Philox execution
    if USE_ANTITHETIC:
        base_path_idx = path_block_idx * BLOCK_SIZE_PATHS
        first_offs    = base_path_idx + tl.arange(0, BLOCK_SIZE_PATHS // 2)
        base_rng_offsets = MC_OFFSET_PHILOX + first_offs * NUM_DIMENSIONS
    else:
        base_rng_offsets = MC_OFFSET_PHILOX + path_offs * NUM_DIMENSIONS

    # ══════════════════════════════════════════════════════════════════════════
    # ── FUSED SEQUENTIAL EVALUATION LOOP (Direct 1D Loads) ──
    # ══════════════════════════════════════════════════════════════════════════
    for dim_block_idx in tl.range(NUM_DIMENSIONS // BLOCK_SIZE_DIM, num_stages=LOOP_STAGES, loop_unroll_factor=LOOP_UNROLL, warp_specialize=WARP_SPECIALIZE, flatten=LOOP_FLATTEN):
        dim_offset = dim_block_idx * BLOCK_SIZE_DIM
        
        # OUTER BRANCHING: Perfect SSA Scope Isolation
        if is_compute:
            for d in tl.static_range(BLOCK_SIZE_DIM):
                step_idx = dim_offset + d
                
                # DIRECT 1D LOAD (No slicing needed)
                ln_pos_slice = tl.load(ln_positions_ptr + step_idx * NUM_PARTICLES + p_idx)
                
                # 1. GENERATE
                if USE_ANTITHETIC:
                    Z_half = tl.randn(SEED, base_rng_offsets + step_idx)
                    Z = tl.ravel(tl.join(Z_half, -Z_half))
                else:
                    Z = tl.randn(SEED, base_rng_offsets + step_idx)
                
                current_lnS = current_lnS + drift_l2 + vol_l2 * Z
                step_lnS = current_lnS
                
                # 2. BOUNDARY CHECK
                if BLOCK_SIZE_PARTICLES == 1:
                    any_ex = (step_lnS < ln_pos_slice) if OPTION_TYPE == 1 else (step_lnS > ln_pos_slice)
                    just_ex = any_ex & ~done_acc
                    
                    ex_lnS_acc  = tl.where(just_ex, step_lnS, ex_lnS_acc)
                    ex_step_acc = tl.where(just_ex, step_idx, ex_step_acc)
                    done_acc    = done_acc | any_ex
                else:
                    lnS_2d = tl.broadcast_to(tl.expand_dims(step_lnS, 1), [BLOCK_SIZE_PATHS, BLOCK_SIZE_PARTICLES])
                    pos_2d = tl.broadcast_to(tl.expand_dims(ln_pos_slice, 0), [BLOCK_SIZE_PATHS, BLOCK_SIZE_PARTICLES])
                    
                    any_ex = (pos_2d > lnS_2d) if OPTION_TYPE == 1 else (pos_2d < lnS_2d)
                    just_ex = any_ex & ~done_acc
                    
                    ex_lnS_acc  = tl.where(just_ex, lnS_2d, ex_lnS_acc)
                    ex_step_acc = tl.where(just_ex, step_idx, ex_step_acc)
                    done_acc    = done_acc | any_ex

        else:
            for d in tl.static_range(BLOCK_SIZE_DIM):
                step_idx = dim_offset + d
                
                # DIRECT 1D LOADS (No slicing needed)
                ln_pos_slice = tl.load(ln_positions_ptr + step_idx * NUM_PARTICLES + p_idx)
                step_lnS = tl.load(st_ptr + step_idx * NUM_BW_PATHS_SAFE + bw_path_offs, mask=bw_mask)
                current_lnS = step_lnS  # Track for terminal math
                
                # 2. BOUNDARY CHECK
                if BLOCK_SIZE_PARTICLES == 1:
                    any_ex = (step_lnS < ln_pos_slice) if OPTION_TYPE == 1 else (step_lnS > ln_pos_slice)
                    just_ex = any_ex & ~done_acc
                    
                    ex_lnS_acc  = tl.where(just_ex, step_lnS, ex_lnS_acc)
                    ex_step_acc = tl.where(just_ex, step_idx, ex_step_acc)
                    done_acc    = done_acc | any_ex
                else:
                    lnS_2d = tl.broadcast_to(tl.expand_dims(step_lnS, 1), [BLOCK_SIZE_PATHS, BLOCK_SIZE_PARTICLES])
                    pos_2d = tl.broadcast_to(tl.expand_dims(ln_pos_slice, 0), [BLOCK_SIZE_PATHS, BLOCK_SIZE_PARTICLES])
                    
                    any_ex = (pos_2d > lnS_2d) if OPTION_TYPE == 1 else (pos_2d < lnS_2d)
                    just_ex = any_ex & ~done_acc
                    
                    ex_lnS_acc  = tl.where(just_ex, lnS_2d, ex_lnS_acc)
                    ex_step_acc = tl.where(just_ex, step_idx, ex_step_acc)
                    done_acc    = done_acc | any_ex

    # ── TERMINAL MATH ────────────────────────────────────────────────────────
    terminal_lnS_acc = current_lnS

    r_dt_l2 = -r * dt * LOG2E
    terminal_discount = tl.exp2(r_dt_l2 * NUM_DIMENSIONS)
    
    ex_disc_acc = tl.exp2((ex_step_acc.to(tl.float32) + 1.0) * r_dt_l2) 

    if BLOCK_SIZE_PARTICLES == 1:
        final_lnS  = tl.where(done_acc, ex_lnS_acc, terminal_lnS_acc)
        final_disc = tl.where(done_acc, ex_disc_acc, terminal_discount)
        
        final_S = tl.exp2(final_lnS)
        payoff  = tl.maximum(0.0, STRIKE_PRICE - final_S) if OPTION_TYPE == 1 else tl.maximum(0.0, final_S - STRIKE_PRICE)
        
        final_payoff = payoff * final_disc
        payoff_accum = payoff_accum + tl.sum(final_payoff, axis=0, keep_dims=True)
    else:
        terminal_lnS_2d = tl.broadcast_to(tl.expand_dims(terminal_lnS_acc, 1), [BLOCK_SIZE_PATHS, BLOCK_SIZE_PARTICLES])
        
        final_lnS  = tl.where(done_acc, ex_lnS_acc, terminal_lnS_2d)
        final_disc = tl.where(done_acc, ex_disc_acc, terminal_discount)
        
        final_S = tl.exp2(final_lnS)
        payoff  = tl.maximum(0.0, STRIKE_PRICE - final_S) if OPTION_TYPE == 1 else tl.maximum(0.0, final_S - STRIKE_PRICE)
        
        final_payoff = payoff * final_disc
        payoff_accum = payoff_accum + tl.sum(final_payoff, axis=0)

    NUM_PATH_BLOCKS = NUM_PATHS // BLOCK_SIZE_PATHS
    tl.store(partial_payoffs_ptr + p_idx * NUM_PATH_BLOCKS + path_block_idx, payoff_accum)
