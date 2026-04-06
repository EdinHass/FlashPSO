import time
import numpy as np
from .utils import *
from flash_pso import FlashPSO
from flash_pso.config import OptionConfig, ComputeConfig, SwarmConfig
from .utils import _quantlib_american_price, get_reference_price, log

def exec_variance_benchmark():
    log("=" * 90)
    log("FLASH-PSO EMPIRICAL ACCURACY & VARIANCE SUITE")
    log("=" * 90)

    # ── HIGH VARIANCE TEST PARAMETERS ──
    TEST_S0 = 100.0
    TEST_K = 100.0
    TEST_R = 0.05
    TEST_VOL = 0.30
    TEST_T = 2.0

    # ── 1. Establish the Platinum Reference ──
    print("Computing 20,000-step Binomial Reference (this may take a moment)...")
    if HAS_QUANTLIB:
        ref_price = _quantlib_american_price('binomial', 20000, 
                                             s0=TEST_S0, k=TEST_K, r_rate=TEST_R, vol=TEST_VOL, t=TEST_T)
        ref_label = "QuantLib CRR (20k Steps)"
    else:
        ref_price = get_reference_price(s0=TEST_S0, k=TEST_K, r_rate=TEST_R, vol=TEST_VOL, t=TEST_T)
        ref_label = "Standard Reference"
        
    print(f"Reference Price [{ref_label}]: {ref_price:.6f}\n")

    # ── 2. Test Configuration ──
    NUM_RUNS = 50           # Number of random seeds to test
    PATHS = 131072          # High path count for low MC variance
    STEPS = 128             # Standard exercise points
    ITERS = 150             # Force deep convergence
    PARTICLES = 128

    print(f"Test Specs : {NUM_RUNS} runs | {PATHS} paths | {STEPS} steps | {PARTICLES} particles | {ITERS} iterations")
    print("-" * 90)
    print(f"{'Method':<20} | {'Mean Price':<12} | {'Bias (Err)':<12} | {'Std Dev':<10} | {'Std Error':<10} | {'RMSE':<10}")
    print("-" * 90)

    for use_anti in [False, True]:
        prices = np.zeros(NUM_RUNS, dtype=np.float32)
        
        start_time = time.perf_counter()
        
        for i in range(NUM_RUNS):
            opt = OptionConfig(
                initial_stock_price=TEST_S0, 
                strike_price=TEST_K, 
                risk_free_rate=TEST_R, 
                volatility=TEST_VOL,  
                time_to_maturity=TEST_T,    
                num_paths=PATHS, 
                num_time_steps=STEPS
            )
            comp = ComputeConfig(
                compute_fraction=0.0, 
                max_iterations=ITERS, 
                sync_iters=5, 
                seed=1000 + i,                  # <--- Increment seed every run
                use_antithetic=use_anti, 
                convergence_threshold=-1.0      # <--- Disable early stopping
            )
            
            swarm = SwarmConfig(num_particles=PARTICLES)

            # Reconstruct and Optimize
            flash = FlashPSO(option_config=opt, compute_config=comp, swarm_config=swarm)
            flash.optimize()
            
            prices[i] = flash.get_option_price()
            
            # Print a tiny progress indicator so you know it hasn't frozen
            print(".", end="", flush=True)

        elapsed = time.perf_counter() - start_time
        print("\r", end="") # Clear the progress dots
        
        # ── 3. Statistical Analysis ──
        mean_price = np.mean(prices)
        std_dev = np.std(prices, ddof=1)
        std_error = std_dev / np.sqrt(NUM_RUNS)
        bias = mean_price - ref_price
        rmse = np.sqrt(np.mean((prices - ref_price)**2))
        
        method_name = "Antithetic MC" if use_anti else "Standard MC"
        
        # ── 4. Print Results ──
        print(f"{method_name:<20} | {mean_price:12.6f} | {bias:12.6f} | {std_dev:10.6f} | {std_error:10.6f} | {rmse:10.6f}")

    print("-" * 90)
    print("Run complete.\n")

if __name__ == "__main__":
    _setup_env()
    exec_variance_benchmark()
