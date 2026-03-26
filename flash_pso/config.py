from dataclasses import dataclass
import itertools
import triton


@dataclass
class OptionConfig:
    """Market and asset parameters for the option pricing simulation."""
    initial_stock_price: float
    strike_price: float
    risk_free_rate: float
    volatility: float
    time_to_maturity: float
    
    # Simulation Dimensions
    num_paths: int
    num_time_steps: int
    option_type: int = 1  # 1 = Put, 0 = Call

    @property
    def time_step_size(self) -> float:
        return self.time_to_maturity / self.num_time_steps


@dataclass
class ComputeConfig:
    """Hardware, execution, and kernel tuning parameters."""
    
    # ── Hybrid Compute/Bandwidth Balancing ────────────────────────
    # 0.0 = 100% Precomputed (Memory Bound)
    # 1.0 = 100% On-the-fly (Compute Bound)
    # 0.25 = Peak Pipeline Overlap (TMA + ALU parallel saturation)
    compute_fraction: float = 1.0
    
    # ── Memory Layout & Block Sizes ───────────────────────────────
    manual_blocks: bool = False
    pso_paths_block_size: int = 256 
    pso_particles_block_size: int = 256
    pso_dim_block_size: int = 64
    mc_block_size: int = 1024
    init_block_size: int = 256
    reduction_block_size: int = 256
    
    # ── Execution & Convergence ───────────────────────────────────
    max_iterations: int = 1000
    sync_iters: int = 10
    convergence_threshold: float = 1e-6
    
    # ── RNG State ─────────────────────────────────────────────────
    seed: int = 42
    use_fixed_random: bool = False
    use_antithetic: bool = False

    @property
    def compute_on_the_fly(self) -> bool:
        return self.compute_fraction > 0.0


@dataclass
class SwarmConfig:
    """Particle Swarm Optimization hyper-parameters."""
    num_particles: int
    inertia_weight: float = 0.5
    cognitive_weight: float = 0.5
    social_weight: float = 0.5


def get_autotune_configs():
    """Generates the Triton autotuning search space for the payoff kernel."""
    configs = []

    # Note: BLOCK_SIZE_PATHS is NOT autotuned (fixed by pso_paths_block_size in config)
    
    # Now that the kernels are split and payoffs are deferred to log-space, 
    # register pressure is low enough to safely explore a wider grid geometry.
    particle_blocks = [1, 4]
    dim_blocks      = [1, 4]
    warps           = [4, 8]
    stages          = [1, 2]
    
    # Compiler manipulation flags (Locked for optimal instruction pipelining)
    loop_unroll_l   = [8]
    loop_stages_l   = [1]
    warp_spec_l     = [False] # FIX: not working currently
    flatten_l       = [True]

    for pt, d, w, s, ls, lu, ws, fl in itertools.product(
        particle_blocks, dim_blocks, warps, stages, loop_stages_l, loop_unroll_l,
        warp_spec_l, flatten_l,
    ):
        # warp_specialize strictly requires >= 4 warps
        if ws and w < 4:
            continue

        configs.append(
            triton.Config(
                {
                    "BLOCK_SIZE_PARTICLES": pt,
                    "BLOCK_SIZE_DIM":       d,
                    "LOOP_STAGES":          ls,
                    "LOOP_UNROLL":          lu,
                    "WARP_SPECIALIZE":      ws,
                    "LOOP_FLATTEN":         fl,
                },
                num_warps=w,
                num_stages=s,
                num_ctas=1,
            )
        )

    return configs
