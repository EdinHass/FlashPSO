import time
import numpy as np
import torch
from typing import Union

from flash_pso import FlashPSO, FlashPSOBasket
from flash_pso.config import OptionConfig, BasketOptionConfig, ComputeConfig, SwarmConfig
from flash_pso.enums import OptionStyle, OptionType
from .models import Method

# external dependencies
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


# utils

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



# execution wrappers

def run_flash_pso(problem: Union[OptionConfig, BasketOptionConfig], compute: ComputeConfig, swarm: SwarmConfig, seed: int):
    comp = ComputeConfig(**{**compute.__dict__, "seed": seed})
    PricerClass = FlashPSOBasket if isinstance(problem, BasketOptionConfig) else FlashPSO
    
    # preallocate CUDA events for timing
    s_init, e_init = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s_exec, e_exec = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    
    torch.cuda.synchronize()
    
    # init
    s_init.record()
    pricer = PricerClass(problem, comp, swarm)
    e_init.record()
    
    torch.cuda.synchronize() 
    
    # execute
    s_exec.record()
    pricer.optimize()
    e_exec.record()
    
    torch.cuda.synchronize()
    
    # compute timings and results
    init_ms = s_init.elapsed_time(e_init)
    exec_ms = s_exec.elapsed_time(e_exec)
    
    # For Basket options, the "price" is the debiased price after optimization; for vanilla options, it's the direct option price.
    # We use this because basket options are more likely to overfit to the noise, so the debiased price is a more meaningful metric 
    # of the optimization quality.
    price = pricer.get_debiased_price()
    actual_iters = pricer.global_payoff_index * comp.sync_iters
    
    return price, init_ms, exec_ms, actual_iters


def run_opencl_pso(problem: Union[OptionConfig, BasketOptionConfig], compute: ComputeConfig, swarm: SwarmConfig, seed: int):
    if not HAS_OPENCL or isinstance(problem, BasketOptionConfig): 
        return float('nan'), 0, 0, 0
    
    np.random.seed(seed)
    hybridMonteCarlo.setSeed(seed)
    opt_str = 'P' if problem.option_type == OptionType.PUT else 'C'
    
    # Phase 1: Init
    t0 = time.perf_counter()
    mc = hybridMonteCarlo(
        problem.initial_stock_price, problem.risk_free_rate, problem.volatility, 
        problem.time_to_maturity, problem.num_paths, problem.num_time_steps, 
        problem.strike_price, opt_str, swarm.num_particles
    )
    solver = PSO_OpenCL_vec_fusion(mc, swarm.num_particles, iterMax=compute.max_iterations)
    solver._criteria = compute.convergence_threshold
    t1 = time.perf_counter()
    
    # Phase 2: Exec
    t2 = time.perf_counter()
    res = solver.solvePsoAmerOption_cl()
    mc.cleanUp()
    t3 = time.perf_counter()
    
    return res[0], (t1 - t0) * 1000, (t3 - t2) * 1000, len(res[2])


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
    
    t2 = time.perf_counter()
    price = solver.longstaff_schwartz_itm_path_fast_hybrid()[0]
    mc.cleanUp()
    t3 = time.perf_counter()
    
    return price, (t1 - t0) * 1000, (t3 - t2) * 1000, 1


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


# wrapper for CPU ablation study: move path gen and reduction back to CPU, keep velocity updates and payoff eval on GPU

class FlashPSOCPUReduction(FlashPSO):
    """FlashPSO with path generation and pbest/gbest reduction on CPU.

    Isolates the two operations that were moved from CPU to GPU in FlashPSO:
      - _precompute_mc_paths: GBM path generation (numpy, then H2D copy)
      - _reduce_pbest:        payoff aggregation and best-particle selection

    PSO velocity updates and payoff evaluation remain on GPU.
    """

    def _precompute_mc_paths(self):
        T = self.opt.num_time_steps
        N = self._num_bw_paths
        rng = np.random.default_rng(self.comp.seed)
        drift = np.float32(self.drift_l2)
        vol = np.float32(self.vol_l2)
        lnS = np.full(N, self.log2_S0, dtype=np.float32)
        # Fill [T, N] array matching the kernel's expected storage layout:
        # st_ptr + step * NUM_BW_PATHS + path_idx
        St_cpu = np.empty((T, N), dtype=np.float32)
        for t in range(T):
            Z = rng.standard_normal(N).astype(np.float32)
            lnS += drift + vol * Z
            St_cpu[t] = lnS
        # self.St is [N, T] logically but [T, N] in storage; .t() is the contiguous [T, N] view
        self.St.t().copy_(torch.from_numpy(St_cpu))

    def _reduce_pbest(self):
        partial_payoffs_cpu = self._partial_payoffs.cpu()
        positions_cpu = self.positions.cpu()

        total_path_blocks = self.opt.num_paths // self.comp.pso_paths_block_size
        payoffs_cpu = partial_payoffs_cpu[:total_path_blocks].sum(dim=0) / self.opt.num_paths

        pbest_payoff_cpu = self.pbest_payoff.cpu()
        pbest_pos_cpu = self.pbest_pos.cpu()

        improved = payoffs_cpu > pbest_payoff_cpu
        pbest_payoff_cpu[improved] = payoffs_cpu[improved]
        pbest_pos_cpu[improved] = positions_cpu[improved]

        best_idx = torch.argmax(pbest_payoff_cpu)
        if pbest_payoff_cpu[best_idx] > self.gbest_payoff.item():
            self.gbest_payoff.fill_(pbest_payoff_cpu[best_idx].item())
            self.gbest_pos.copy_(pbest_pos_cpu[best_idx].to("cuda"))

        self.pbest_payoff.copy_(pbest_payoff_cpu.to("cuda"))
        self.pbest_pos.copy_(pbest_pos_cpu.to("cuda"))


def run_flash_pso_cpu_ablation(problem: Union[OptionConfig, BasketOptionConfig], compute: ComputeConfig, swarm: SwarmConfig, seed: int):
    if isinstance(problem, BasketOptionConfig):
        return float('nan'), 0, 0, 0
    comp = ComputeConfig(**{**compute.__dict__, "seed": seed})

    # Use perf_counter + explicit syncs since CUDA events miss CPU-side numpy computation
    torch.cuda.synchronize()
    t_init_0 = time.perf_counter()
    pricer = FlashPSOCPUReduction(problem, comp, swarm)
    torch.cuda.synchronize()
    t_init_1 = time.perf_counter()

    torch.cuda.synchronize()
    t_exec_0 = time.perf_counter()
    pricer.optimize()
    torch.cuda.synchronize()
    t_exec_1 = time.perf_counter()

    return (
        pricer.get_option_price(),
        (t_init_1 - t_init_0) * 1000,
        (t_exec_1 - t_exec_0) * 1000,
        pricer.global_payoff_index * comp.sync_iters,
    )


WRAPPER_REGISTRY = {
    Method.FLASH_PSO: run_flash_pso,
    Method.FLASH_PSO_CPU: run_flash_pso_cpu_ablation,
    Method.OPENCL_PSO: run_opencl_pso,
    Method.OPENCL_LSMC: run_opencl_lsmc,
    Method.QUANTLIB: run_quantlib,
    Method.NATIVE_BINOMIAL: run_native_binomial,
}
