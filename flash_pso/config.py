from dataclasses import dataclass
import itertools
import triton

# This file defines the configuration classes for our FlashPSO implementation.

# The OptionConfig class holds all the parameters related to the option we want to price and the Monte Carlo simulation
@dataclass
class OptionConfig:
    # Market + Asset Params
    initial_stock_price: float  # The current, starting price of the stock
    strike_price: float         # The strike price of the option
    risk_free_rate: float       # The risk-free interest rate, expressed as a decimal (e.g., 0.05 for 5%). 
    volatility: float           # The volatility of the stock's returns, expressed as a decimal (e.g., 0.2 for 20% annual volatility).
    time_to_maturity: float     # How much time is left until the option expires, measured in years (e.g., 1.0 for 1 year, 0.5 for 6 months).

    # Simulation Params
    num_paths: int              # How many different timelines we simulate for the stock price. More paths = better accuracy but more computation.
    num_time_steps: int         # How many discrete time intervals we break the time to maturity into. More steps = better accuracy but more computation.
    option_type: int = 1        # 1 for Put (right to sell/betting price goes down), 0 for Call (right to buy/betting price goes up).

    @property
    def time_step_size(self) -> float:
        # The actual fraction of a year that passes during a single step in our loop.
        # If the option expires in 1 year and we have 10 steps, each step is 0.1 years.
        return self.time_to_maturity / self.num_time_steps

# The ComputeConfig class holds parameters related to how we want to perform the computations
@dataclass
class ComputeConfig:
    # If True, recompute paths in SMEM. If False, read from GMEM.
    compute_on_the_fly: bool = True 
    
    # Block sizes for Triton tuning
    manual_blocks: bool = False
    pso_particles_block_size: int = 256
    pso_paths_block_size: int = 256
    pso_dim_block_size: int = 64
    mc_block_size: int = 1024
    init_block_size: int = 256
    reduction_block_size: int = 256 

    # Random seed for reproducibility, used in path generation and particle initialization
    seed: int = 42

    # Stopping criterion for PSO convergence (if the global payoff doesn't improve by more than this amount, we stop)
    convergence_threshold: float = 1e-6

    # Maximum number of iterations to prevent infinite loops in case of non-convergence
    max_iterations: int = 1000

     # How often to sync global payoff back to CPU for convergence checking (every N iterations)
    sync_iters: int = 10

# The SwarmConfig class holds parameters related to the Particle Swarm Optimization algorithm itself
@dataclass
class SwarmConfig:
    num_particles: int            # The number of particles in the swarm (more particles can explore the solution space better but require more computation)
    inertia_weight: float = 0.5   # Controls how much a particle's current velocity influences its next move
    cognitive_weight: float = 0.5 # Controls how much a particle is influenced by its own best position found so far.
    social_weight: float = 0.5    # Controls how much a particle is influenced by the best position found by the ENTIRE swarm.

# This function generates a list of Triton autotuning configurations to explore different block sizes and kernel parameters.
def get_autotune_configs():
    configs = []
    path_blocks     = [64, 128]
    particle_blocks = [4]
    dim_blocks      = [2]
    warps           = [8]
    stages          = [1]
    num_ctas        = [1]

    # Generate every possible combination
    for p, pt, d, w, s, cta in itertools.product(path_blocks, particle_blocks, dim_blocks, warps, stages, num_ctas):

        # If the total elements in the 3D block exceed ~131k, it will likely OOM 
        # the registers or shared memory. Skip it.
        total_elements = p * pt * d
        if total_elements > 65536: 
            continue
            
        configs.append(
            triton.Config(
                {"BLOCK_SIZE_PATHS": p, "BLOCK_SIZE_PARTICLES": pt, "BLOCK_SIZE_DIM": d},
                num_warps=w,
                num_stages=s,
                num_ctas=cta
            )
        )
        
    return configs
