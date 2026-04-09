import time
import numpy as np
import torch
from typing import Union

from flash_pso import FlashPSO, FlashPSOBasket
from flash_pso.config import OptionConfig, BasketOptionConfig, ComputeConfig, SwarmConfig
from flash_pso.enums import OptionStyle, OptionType
from .models import Method

# ─── EXTERNAL DEPENDENCIES ────────────────────────────────────────────────────
try:
    from references.mc import hybridMonteCarlo
    from references.pso import PSO_OpenCL_vec_fusion
    from references.longstaff import LSMC_OpenCL
    HAS_OPENCL = True
except ImportError:
    HAS_OPENCL = False

try:
    from references.benchmarks import binomialAmericanOption
    HAS_NATIVE_BINOMIAL = True
except ImportError:
    HAS_NATIVE_BINOMIAL = False

try:
    import QuantLib as ql
    HAS_QUANTLIB = True
except ImportError:
    HAS_QUANTLIB = False


# ══════════════════════════════════════════════════════════════════════════════
# ── QUANTLIB & NATIVE UTILS ───────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def _get_ql_process(s0: float, r_rate: float, vol: float):
    today = ql.Date().todaysDate()
    ql.Settings.instance().evaluationDate = today
    spot_handle = ql.QuoteHandle(ql.SimpleQuote(s0))
    flat_ts = ql.YieldTermStructureHandle(ql.FlatForward(today, r_rate, ql.Actual365Fixed()))
    div_ts = ql.YieldTermStructureHandle(ql.FlatForward(today, 0.0, ql.Actual365Fixed()))
    flat_vol = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(today, ql.NullCalendar(), vol, ql.Actual365Fixed()))
    bsm = ql.BlackScholesMertonProcess(spot_handle, div_ts, flat_ts, flat_vol)
    return today, bsm

def _quantlib_american_price(s0: float, k: float, r_rate: float, vol: float, t: float, opttype: str, 
                             engine_type: str = 'binomial', steps: int = 2000) -> float:
    if not HAS_QUANTLIB: return float('nan')
    today, bsm = _get_ql_process(s0, r_rate, vol)
    
    ql_opt_type = ql.Option.Put if opttype.upper() == 'P' else ql.Option.Call
    option = ql.VanillaOption(
        ql.PlainVanillaPayoff(ql_opt_type, k), 
        ql.AmericanExercise(today, today + ql.Period(int(t * 365), ql.Days))
    )
    
    engine = ql.BinomialVanillaEngine(bsm, "crr", steps) if engine_type == 'binomial' else ql.FdBlackScholesVanillaEngine(bsm, steps, steps)
    option.setPricingEngine(engine)
    return option.NPV()

def _quantlib_asian_price(s0: float, k: float, r_rate: float, vol: float, t: float, opttype: str, 
                          paths: int = 10000, steps: int = 128) -> float:
    if not HAS_QUANTLIB: return float('nan')
    today, bsm = _get_ql_process(s0, r_rate, vol)
    
    ql_opt_type = ql.Option.Put if opttype.upper() == 'P' else ql.Option.Call
    dates = sorted(list({today + ql.Period(max(1, int(i * t * 365 / steps)), ql.Days) for i in range(1, steps + 1)}))
    
    option = ql.DiscreteAveragingAsianOption(
        ql.Average.Arithmetic, 0.0, 0, dates, 
        ql.PlainVanillaPayoff(ql_opt_type, k), 
        ql.EuropeanExercise(today + ql.Period(int(t * 365), ql.Days))
    )
    option.setPricingEngine(ql.MCDiscreteArithmeticAPEngine(bsm, 'pseudorandom', requiredSamples=paths, seed=42))
    return option.NPV()

def _quantlib_basket_lsmc_price(s0s: list, k: float, r_rate: float, vols: list, weights: list, 
                                corr: list, t: float, opttype: str, paths: int = 100000, steps: int = 128) -> float:
    if not HAS_QUANTLIB: return float('nan')
    today = ql.Date().todaysDate()
    ql.Settings.instance().evaluationDate = today

    n_assets = len(s0s)
    processes = []
    for i in range(n_assets):
        spot_handle = ql.QuoteHandle(ql.SimpleQuote(s0s[i]))
        flat_ts = ql.YieldTermStructureHandle(ql.FlatForward(today, r_rate, ql.Actual365Fixed()))
        div_ts = ql.YieldTermStructureHandle(ql.FlatForward(today, 0.0, ql.Actual365Fixed()))
        flat_vol = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(today, ql.NullCalendar(), vols[i], ql.Actual365Fixed()))
        bsm = ql.BlackScholesMertonProcess(spot_handle, div_ts, flat_ts, flat_vol)
        processes.append(bsm)

    matrix = ql.Matrix(n_assets, n_assets)
    for i in range(n_assets):
        for j in range(n_assets):
            matrix[i][j] = corr[i][j]

    process_array = ql.StochasticProcessArray(processes, matrix)

    ql_opt_type = ql.Option.Put if opttype.upper() == 'P' else ql.Option.Call
    payoff = ql.AverageBasketPayoff(ql.PlainVanillaPayoff(ql_opt_type, k), ql.Array(weights))
    exercise = ql.AmericanExercise(today, today + ql.Period(int(t * 365), ql.Days))
    basket_option = ql.BasketOption(payoff, exercise)

    engine = ql.MCAmericanBasketEngine(process_array, 'pseudorandom', timeSteps=steps, requiredSamples=paths, seed=42)
    basket_option.setPricingEngine(engine)
    return basket_option.NPV()


# ══════════════════════════════════════════════════════════════════════════════
# ── HARDWARE-TIMED EXECUTION WRAPPERS ─────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def run_flash_pso(problem: Union[OptionConfig, BasketOptionConfig], compute: ComputeConfig, swarm: SwarmConfig, seed: int):
    comp = ComputeConfig(**{**compute.__dict__, "seed": seed})
    
    # 1. Resolve branches BEFORE hardware timing begins
    PricerClass = FlashPSOBasket if isinstance(problem, BasketOptionConfig) else FlashPSO
    
    # 2. Create GPU timing events
    start_init = torch.cuda.Event(enable_timing=True)
    end_init = torch.cuda.Event(enable_timing=True)
    start_exec = torch.cuda.Event(enable_timing=True)
    end_exec = torch.cuda.Event(enable_timing=True)
    
    # 3. Flush pipeline to ensure clean start
    torch.cuda.synchronize()
    
    # --- PHASE 1: Initialization ---
    start_init.record()
    pricer = PricerClass(problem, comp, swarm)
    end_init.record()
    
    torch.cuda.synchronize() # Barrier between phases
    
    # --- PHASE 2: Execution ---
    start_exec.record()
    pricer.optimize()
    end_exec.record()
    
    # 4. Wait for GPU to finish before calculating elapsed time
    torch.cuda.synchronize()
    
    init_ms = start_init.elapsed_time(end_init)
    exec_ms = start_exec.elapsed_time(end_exec)
    
    price = pricer.get_debiased_price()
    actual_iters = pricer.global_payoff_index * comp.sync_iters
    
    return price, init_ms, exec_ms, actual_iters


def run_opencl_pso(problem: Union[OptionConfig, BasketOptionConfig], compute: ComputeConfig, swarm: SwarmConfig, seed: int):
    if not HAS_OPENCL or isinstance(problem, BasketOptionConfig): 
        return float('nan'), 0, 0, 0
    
    np.random.seed(seed)
    hybridMonteCarlo.setSeed(seed)
    opt_str = 'P' if problem.option_type == OptionType.PUT else 'C'
    
    # Python overhead is unavoidable here, but we tighten the bounds as much as possible
    t0 = time.perf_counter()
    mc = hybridMonteCarlo(
        problem.initial_stock_price, problem.risk_free_rate, problem.volatility, 
        problem.time_to_maturity, problem.num_paths, problem.num_time_steps, 
        problem.strike_price, opt_str, swarm.num_particles
    )
    solver = PSO_OpenCL_vec_fusion(mc, swarm.num_particles, iterMax=compute.max_iterations)
    
    # Inject stopping criteria to match FlashPSO
    solver._criteria = compute.convergence_threshold
    t1 = time.perf_counter()
    
    res = solver.solvePsoAmerOption_cl()
    mc.cleanUp()
    t2 = time.perf_counter()
    
    return res[0], (t1 - t0) * 1000, (t2 - t1) * 1000, len(res[2])


def run_opencl_lsmc(problem: Union[OptionConfig, BasketOptionConfig], compute: ComputeConfig, swarm: SwarmConfig, seed: int):
    if not HAS_OPENCL or isinstance(problem, BasketOptionConfig): 
        return float('nan'), 0, 0, 0
        
    np.random.seed(seed)
    hybridMonteCarlo.setSeed(seed)
    opt_str = 'P' if problem.option_type == OptionType.PUT else 'C'
    
    t0 = time.perf_counter()
    mc = hybridMonteCarlo(
        problem.initial_stock_price, problem.risk_free_rate, problem.volatility, 
        problem.time_to_maturity, problem.num_paths, problem.num_time_steps, 
        problem.strike_price, opt_str, swarm.num_particles
    )
    solver = LSMC_OpenCL(mc, inverseType='GJ')
    t1 = time.perf_counter()
    
    price = solver.longstaff_schwartz_itm_path_fast_hybrid()[0]
    mc.cleanUp()
    t2 = time.perf_counter()
    
    return price, (t1 - t0) * 1000, (t2 - t1) * 1000, 1


def run_quantlib(problem: Union[OptionConfig, BasketOptionConfig], compute: ComputeConfig, swarm: SwarmConfig, seed: int):
    opt_str = 'P' if problem.option_type == OptionType.PUT else 'C'
    
    t0 = time.perf_counter()
    if isinstance(problem, BasketOptionConfig):
        price = _quantlib_basket_lsmc_price(
            s0s=problem.initial_stock_prices, k=problem.strike_price, r_rate=problem.risk_free_rate,
            vols=problem.volatilities, weights=problem.weights, corr=problem.correlation_matrix,
            t=problem.time_to_maturity, opttype=opt_str, paths=problem.num_paths, steps=problem.num_time_steps
        )
    else:
        if problem.option_style == OptionStyle.ASIAN:
            price = _quantlib_asian_price(
                s0=problem.initial_stock_price, k=problem.strike_price, r_rate=problem.risk_free_rate,
                vol=problem.volatility, t=problem.time_to_maturity, opttype=opt_str,
                paths=problem.num_paths, steps=problem.num_time_steps
            )
        else:
            price = _quantlib_american_price(
                s0=problem.initial_stock_price, k=problem.strike_price, r_rate=problem.risk_free_rate,
                vol=problem.volatility, t=problem.time_to_maturity, opttype=opt_str,
                engine_type='binomial', steps=2000
            )
    t1 = time.perf_counter()
    
    return price, 0, (t1 - t0) * 1000, 1


def run_native_binomial(problem: Union[OptionConfig, BasketOptionConfig], compute: ComputeConfig, swarm: SwarmConfig, seed: int):
    if not HAS_NATIVE_BINOMIAL or isinstance(problem, BasketOptionConfig): 
        return float('nan'), 0, 0, 0
        
    opt_str = 'P' if problem.option_type == OptionType.PUT else 'C'
    
    t0 = time.perf_counter()
    price = binomialAmericanOption(
        problem.initial_stock_price, problem.strike_price, problem.risk_free_rate, 
        problem.volatility, 2000, problem.time_to_maturity, opt_str
    )[0]
    t1 = time.perf_counter()
    
    return price, 0, (t1 - t0) * 1000, 1


WRAPPER_REGISTRY = {
    Method.FLASH_PSO: run_flash_pso,
    Method.OPENCL_PSO: run_opencl_pso,
    Method.OPENCL_LSMC: run_opencl_lsmc,
    Method.QUANTLIB: run_quantlib,
    Method.NATIVE_BINOMIAL: run_native_binomial,
}
