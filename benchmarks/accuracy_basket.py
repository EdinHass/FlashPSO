import time
import numpy as np
import math

from flash_pso import FlashPSOBasket
from flash_pso.config import BasketOptionConfig, ComputeConfig, SwarmConfig

try:
    import QuantLib as ql
    HAS_QUANTLIB = True
except ImportError:
    HAS_QUANTLIB = False

def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def get_ql_basket_reference(s0s, k, r_rate, vols, weights, rho, t, paths=200000, steps=128):
    """Computes a high-precision Longstaff-Schwartz MC reference using QuantLib."""
    if not HAS_QUANTLIB:
        return float('nan')
        
    today = ql.Date().todaysDate()
    ql.Settings.instance().evaluationDate = today
    ts = ql.YieldTermStructureHandle(ql.FlatForward(today, r_rate, ql.Actual365Fixed()))
    div = ql.YieldTermStructureHandle(ql.FlatForward(today, 0.0, ql.Actual365Fixed()))

    procs = []
    for s0, vol in zip(s0s, vols):
        vol_handle = ql.BlackVolTermStructureHandle(
            ql.BlackConstantVol(today, ql.NullCalendar(), vol, ql.Actual365Fixed())
        )
        procs.append(ql.BlackScholesMertonProcess(ql.QuoteHandle(ql.SimpleQuote(s0)), div, ts, vol_handle))

    n = len(s0s)
    matrix = [[1.0 if i == j else rho for j in range(n)] for i in range(n)]

    b_opt = ql.BasketOption(
        ql.AverageBasketPayoff(ql.PlainVanillaPayoff(ql.Option.Put, k), weights),
        ql.AmericanExercise(today, today + ql.Period(int(t * 365), ql.Days))
    )

    b_opt.setPricingEngine(ql.MCAmericanBasketEngine(
        ql.StochasticProcessArray(procs, matrix),
        'pseudorandom', timeSteps=steps, requiredSamples=paths, seed=42
    ))

    return b_opt.NPV()


def exec_basket_variance_benchmark():
    log("=" * 90)
    log("FLASH-PSO BASKET EMPIRICAL ACCURACY & VARIANCE SUITE")
    log("=" * 90)

    # ── BASKET TEST PARAMETERS ──
    # Testing a 2-Asset Basket (e.g., Tech & Utility)
    TEST_S0s = [100.0, 100.0]
    TEST_K = 100.0
    TEST_R = 0.05
    TEST_VOLS = [0.30, 0.30]
    TEST_WEIGHTS = [0.5, 0.5]
    TEST_RHO = 0.5         # Correlation between the assets
    TEST_T = 1.0

    # ── 1. Establish the Platinum Reference ──
    print("Computing 200,000-path QuantLib LSMC Reference (this may take a minute)...")
    if HAS_QUANTLIB:
        ref_price = get_ql_basket_reference(
            s0s=TEST_S0s, k=TEST_K, r_rate=TEST_R, vols=TEST_VOLS, 
            weights=TEST_WEIGHTS, rho=TEST_RHO, t=TEST_T, paths=100000, steps=128
        )
        ref_label = "QuantLib LSMC (200k Paths)"
    else:
        ref_price = float('nan')
        ref_label = "No Reference (Install QuantLib)"
        
    print(f"Reference Price [{ref_label}]: {ref_price:.6f}\n")

    # ── 2. Test Configuration ──
    NUM_RUNS = 50           # Number of random seeds to test
    PATHS = 131072          # High path count for low MC variance
    STEPS = 128             # Standard exercise points
    ITERS = 250             # Force deep convergence
    PARTICLES = 128

    print(f"Test Specs : {NUM_RUNS} runs | {PATHS} paths | {STEPS} steps | {PARTICLES} particles | {ITERS} iterations")
    print("-" * 90)
    print(f"{'Method':<20} | {'Mean Price':<12} | {'Bias (Err)':<12} | {'Std Dev':<10} | {'Std Error':<10} | {'RMSE':<10}")
    print("-" * 90)

    for use_anti in [False, True]:
        prices = np.zeros(NUM_RUNS, dtype=np.float32)
        
        start_time = time.perf_counter()
        
        for i in range(NUM_RUNS):
            opt = BasketOptionConfig(
                initial_stock_prices=TEST_S0s, 
                strike_price=TEST_K, 
                risk_free_rate=TEST_R, 
                volatilities=TEST_VOLS,
                weights=TEST_WEIGHTS,
                correlation_matrix=[[1.0, TEST_RHO], [TEST_RHO, 1.0]],
                time_to_maturity=TEST_T,    
                num_paths=PATHS, 
                num_time_steps=STEPS,
                option_type=1,          # 1 = Put
                option_style="basket",
                exercise_style=0        # 0 = Standard Collapsed Basket
            )
            
            comp = ComputeConfig(
                compute_fraction=0.0, 
                max_iterations=ITERS, 
                sync_iters=5, 
                seed=1000 + i,                  # <--- Increment seed every run
                use_antithetic=use_anti, 
                convergence_threshold=-1.0,     # <--- Disable early stopping for strict variance testing
                use_fp16_cholesky=False         # Keep FP32 for strict accuracy testing
            )
            
            # Swarm search_window=None triggers the new dynamic per-asset depth logic!
            swarm = SwarmConfig(num_particles=PARTICLES)

            # Reconstruct and Optimize
            flash = FlashPSOBasket(option_config=opt, compute_config=comp, swarm_config=swarm)
            flash.optimize()
            
            prices[i] = flash.get_debiased_price()
            
            # Print a tiny progress indicator so you know it hasn't frozen
            print(".", end="", flush=True)

        elapsed = time.perf_counter() - start_time
        print("\r", end="") # Clear the progress dots
        
        # ── 3. Statistical Analysis ──
        mean_price = np.mean(prices)
        std_dev = np.std(prices, ddof=1)
        std_error = std_dev / np.sqrt(NUM_RUNS)
        
        # Safely calculate bias and RMSE if we have a valid reference price
        if math.isnan(ref_price):
            bias = float('nan')
            rmse = float('nan')
        else:
            bias = mean_price - ref_price
            rmse = np.sqrt(np.mean((prices - ref_price)**2))
        
        method_name = "Antithetic MC" if use_anti else "Standard MC"
        
        # ── 4. Print Results ──
        print(f"{method_name:<20} | {mean_price:12.6f} | {bias:12.6f} | {std_dev:10.6f} | {std_error:10.6f} | {rmse:10.6f}")

    print("-" * 90)
    print("Run complete.\n")

if __name__ == "__main__":
    exec_basket_variance_benchmark()
