import time
import numpy as np
import math
import torch

from flash_pso import *
from flash_pso.config import OptionConfig, ComputeConfig, SwarmConfig

try:
    import QuantLib as ql
    HAS_QUANTLIB = True
except ImportError:
    HAS_QUANTLIB = False

def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def get_ql_american_reference(s0, k, r_rate, vol, t, steps=20000):
    """Computes a high-precision Binomial reference using QuantLib."""
    if not HAS_QUANTLIB:
        return float('nan')
        
    today = ql.Date().todaysDate()
    ql.Settings.instance().evaluationDate = today
    
    payoff = ql.PlainVanillaPayoff(ql.Option.Put, k)
    exercise = ql.AmericanExercise(today, today + ql.Period(int(t * 365), ql.Days))
    option = ql.VanillaOption(payoff, exercise)
    
    spot_handle = ql.QuoteHandle(ql.SimpleQuote(s0))
    flat_ts = ql.YieldTermStructureHandle(ql.FlatForward(today, r_rate, ql.Actual365Fixed()))
    flat_vol = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(today, ql.NullCalendar(), vol, ql.Actual365Fixed()))
    div_ts = ql.YieldTermStructureHandle(ql.FlatForward(today, 0.0, ql.Actual365Fixed()))
    
    bsm_process = ql.BlackScholesMertonProcess(spot_handle, div_ts, flat_ts, flat_vol)
    engine = ql.BinomialVanillaEngine(bsm_process, "crr", steps)
    option.setPricingEngine(engine)
    
    return option.NPV()

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
        ref_price = get_ql_american_reference(s0=TEST_S0, k=TEST_K, r_rate=TEST_R, vol=TEST_VOL, t=TEST_T)
        ref_label = "QuantLib CRR (20k Steps)"
    else:
        ref_price = float('nan')
        ref_label = "No Reference (Install QuantLib)"
        
    print(f"Reference Price [{ref_label}]: {ref_price:.6f}\n")

    # ── 2. Test Configuration ──
    NUM_RUNS = 50           
    PATHS = 131072          
    STEPS = 128             
    ITERS = 150             
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
                num_time_steps=STEPS,
                option_type=OptionType.PUT,
                option_style=OptionStyle.STANDARD
            )
            comp = ComputeConfig(
                compute_fraction=0.0, 
                max_iterations=ITERS, 
                sync_iters=5, 
                seed=1000 + i,                  
                use_antithetic=use_anti, 
                convergence_threshold=-1.0      
            )
            
            swarm = SwarmConfig(num_particles=PARTICLES)

            # Reconstruct and Optimize
            flash = FlashPSO(option_config=opt, compute_config=comp, swarm_config=swarm)
            flash.optimize()
            
            torch.cuda.synchronize()
            prices[i] = flash.get_option_price()
            
            print(".", end="", flush=True)

        elapsed = time.perf_counter() - start_time
        print("\r", end="") 
        
        # ── 3. Statistical Analysis ──
        mean_price = np.mean(prices)
        std_dev = np.std(prices, ddof=1)
        std_error = std_dev / np.sqrt(NUM_RUNS)
        
        if math.isnan(ref_price):
            bias = float('nan')
            rmse = float('nan')
        else:
            bias = mean_price - ref_price
            rmse = np.sqrt(np.mean((prices - ref_price)**2))
        
        method_name = "Antithetic MC" if use_anti else "Standard MC"
        
        print(f"{method_name:<20} | {mean_price:12.6f} | {bias:12.6f} | {std_dev:10.6f} | {std_error:10.6f} | {rmse:10.6f}")

    print("-" * 90)
    print("Run complete.\n")

if __name__ == "__main__":
    exec_variance_benchmark()
