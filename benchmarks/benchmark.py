import os
import sys
import time
import numpy as np
import triton
import triton.testing

from references.utils import checkOpenCL
from references.mc import hybridMonteCarlo
from references.pso import PSO_OpenCL_vec_fusion
from references.longstaff import LSMC_Numpy, LSMC_OpenCL
from references.benchmarks import binomialAmericanOption

from flash_pso.flash_pso import FlashPSO
from flash_pso.config import OptionConfig, ComputeConfig, SwarmConfig


def log(msg):
    print(msg, flush=True)

def log_progress(msg):
    elapsed = time.time() - _START_TIME
    print(f"[{elapsed:7.1f}s] {msg}", flush=True)

_START_TIME = time.time()

MAX_ITER = 2 
WARMUP   = 3
REPS     = 10 

S0, r, sigma, T, K, opttype = 100.0, 0.03, 0.3, 1.0, 110.0, 'P'

BINOMIAL_STEPS = 2000
_GROUND_TRUTH  = None

def get_ground_truth():
    global _GROUND_TRUTH
    if _GROUND_TRUTH is None:
        price, _ = binomialAmericanOption(S0, K, r, sigma, BINOMIAL_STEPS, T, opttype=opttype)
        _GROUND_TRUTH = float(price)
        log_progress(f"  ground truth (binomial {BINOMIAL_STEPS} steps): {_GROUND_TRUTH:.6f}")
    return _GROUND_TRUTH

def _opencl_iters(result):    return len(result[-1])
def _opencl_solution(result): return result[0]
def _err(price):              return abs(price - get_ground_truth())


def _make_flash_pso(nFish, nPath, nPeriod, compute_fraction=1.0):
    opt = OptionConfig(
        initial_stock_price=S0, strike_price=K, risk_free_rate=r,
        volatility=sigma, time_to_maturity=T,
        num_paths=nPath, num_time_steps=nPeriod, option_type=1,
    )
    comp = ComputeConfig(
        compute_fraction=compute_fraction,
        manual_blocks=False,
        max_iterations=2,
        use_fixed_random=True,
        use_antithetic=True,
        sync_iters=1,
        seed=42,
    )
    swarm = SwarmConfig(num_particles=nFish)
    return FlashPSO(option_config=opt, compute_config=comp, swarm_config=swarm)


@triton.testing.perf_report([
    triton.testing.Benchmark(
        x_names=['nPath'],
        x_vals=[2**i for i in range(12, 20)],
        line_arg='provider',
        line_vals=['cl_vec_fusion', 'triton_compute', 'triton_bandwidth'],
        line_names=['OpenCL Vec Fusion', 'FlashPSO Compute', 'FlashPSO Bandwidth'],
        styles=[('black', ':'), ('brown', ':'), ('gray', '-'),
                ('blue', '-'), ('green', '--')],
        ylabel='Execution Time (ms)',
        plot_name='pso-vs-lsmc',
        args={'nFish': 64, 'nPeriod': 2**7, 'metric': 'time'},
    ),
])
def benchmark_pso(nPath, nFish, nPeriod, metric, provider):
    log_progress(f"  running  provider={provider:<16}  nFish={nFish:<6}  nPeriod={nPeriod}  nPath={nPath}")

    if provider == 'lsmc_numpy':
        mc   = hybridMonteCarlo(S0, r, sigma, T, nPath, nPeriod, K, opttype, nFish)
        lsmc = LSMC_Numpy(mc, inverseType='benchmark_pinv')
        for _ in range(WARMUP): result = lsmc.longstaff_schwartz_itm_path_fast()
        solution = float(result[0])
        times = []
        for _ in range(REPS):
            t0 = time.perf_counter()
            lsmc.longstaff_schwartz_itm_path_fast()
            times.append((time.perf_counter() - t0) * 1000)
        ms, min_ms, max_ms = tuple(np.quantile(times, [0.5, 0.2, 0.8]))
        mc.cleanUp()
        log_progress(f"  [lsmc_numpy] done  median={ms:.2f}ms  price={solution:.6f}  err={_err(solution):.6f}")

    elif provider == 'lsmc_opencl':
        mc   = hybridMonteCarlo(S0, r, sigma, T, nPath, nPeriod, K, opttype, nFish)
        lsmc = LSMC_OpenCL(mc, inverseType='GJ')
        for _ in range(WARMUP): result = lsmc.longstaff_schwartz_itm_path_fast_hybrid()
        solution = float(result[0])
        times = []
        for _ in range(REPS):
            t0 = time.perf_counter()
            lsmc.longstaff_schwartz_itm_path_fast_hybrid()
            times.append((time.perf_counter() - t0) * 1000)
        ms, min_ms, max_ms = tuple(np.quantile(times, [0.5, 0.2, 0.8]))
        mc.cleanUp()
        log_progress(f"  [lsmc_opencl] done  median={ms:.2f}ms  price={solution:.6f}  err={_err(solution):.6f}")

    elif provider == 'triton_compute':
        flash = _make_flash_pso(nFish, nPath, nPeriod, compute_fraction=0.1)
        flash.optimize()
        converged_at = flash.global_payoff_index * flash.comp.sync_iters
        solution     = flash.get_option_price()
        fn = lambda: flash.optimize()
        ms, min_ms, max_ms = triton.testing.do_bench(fn, warmup=WARMUP-1, rep=REPS, quantiles=[0.5, 0.2, 0.8])
        log_progress(f"  [triton_compute] done  median={ms:.2f}ms  iters={converged_at}"
                     f"  price={solution:.6f}  err={_err(solution):.6f}")

    elif provider == 'triton_bandwidth':
        flash = _make_flash_pso(nFish, nPath, nPeriod, compute_fraction=0.0)
        flash.optimize()
        converged_at = flash.global_payoff_index * flash.comp.sync_iters
        solution     = flash.get_option_price()
        fn = lambda: flash.optimize()
        ms, min_ms, max_ms = triton.testing.do_bench(fn, warmup=WARMUP-1, rep=REPS, quantiles=[0.5, 0.2, 0.8])
        log_progress(f"  [triton_bandwidth] done  median={ms:.2f}ms  iters={converged_at}"
                     f"  price={solution:.6f}  err={_err(solution):.6f}")

    else:  # cl_vec_fusion
        mc = hybridMonteCarlo(S0, r, sigma, T, nPath, nPeriod, K, opttype, nFish)
        fn = lambda: PSO_OpenCL_vec_fusion(mc, nFish, iterMax=MAX_ITER).solvePsoAmerOption_cl()
        result = None
        for _ in range(WARMUP): result = fn()
        converged_at = _opencl_iters(result)
        solution     = _opencl_solution(result)
        times = []
        for _ in range(REPS):
            t0 = time.perf_counter()
            fn()
            times.append((time.perf_counter() - t0) * 1000)
        ms, min_ms, max_ms = tuple(np.quantile(times, [0.5, 0.2, 0.8]))
        mc.cleanUp()
        log_progress(f"  [cl_vec_fusion] done  median={ms:.2f}ms  iters={converged_at}"
                     f"  price={solution:.6f}  err={_err(solution):.6f}")

    if metric == 'time':
        return ms, min_ms, max_ms
    else:
        total_bytes = 7 * nPeriod * 4 * nFish * MAX_ITER
        gbps = lambda t: total_bytes / (t * 1e-3) / 1e9
        return gbps(ms), gbps(max_ms), gbps(min_ms)


FRACTIONS       = [0.0, 0.25, 0.5, 0.75, 1.0]
FRACTION_LABELS = ['BW (0%)', 'Hybrid (25%)', 'Hybrid (50%)', 'Hybrid (75%)', 'Compute (100%)']
FRACTION_STYLES = [('green', '--'), ('cyan', '-'), ('orange', '-'), ('purple', '-'), ('blue', '-')]

@triton.testing.perf_report([
    triton.testing.Benchmark(
        x_names=['nPath'],
        x_vals=[2**i for i in range(12, 20)],
        line_arg='fraction_idx',
        line_vals=list(range(len(FRACTIONS))),
        line_names=FRACTION_LABELS,
        styles=FRACTION_STYLES,
        ylabel='Execution Time (ms)',
        plot_name='compute-fraction-sweep',
        args={'nFish': 32, 'nPeriod': 2**7, 'metric': 'time'},
    ),
])
def benchmark_fraction(nPath, nFish, nPeriod, metric, fraction_idx):
    frac  = FRACTIONS[fraction_idx]
    label = FRACTION_LABELS[fraction_idx]
    log_progress(f"  running  fraction={frac:.2f}  nFish={nFish:<6}  nPeriod={nPeriod}  nPath={nPath}")

    flash = _make_flash_pso(nFish, nPath, nPeriod, compute_fraction=frac)
    flash.optimize()
    converged_at = flash.global_payoff_index * flash.comp.sync_iters
    solution     = flash.get_option_price()

    fn = lambda: flash.optimize()
    ms, min_ms, max_ms = triton.testing.do_bench(fn, warmup=WARMUP-1, rep=REPS, quantiles=[0.5, 0.2, 0.8])

    log_progress(f"  [{label}] done  median={ms:.2f}ms  iters={converged_at}"
                 f"  price={solution:.6f}  err={_err(solution):.6f}")

    if metric == 'time':
        return ms, min_ms, max_ms
    else:
        total_bytes = 7 * nPeriod * 4 * nFish * MAX_ITER
        gbps = lambda t: total_bytes / (t * 1e-3) / 1e9
        return gbps(ms), gbps(max_ms), gbps(min_ms)


def _setup():
    global _START_TIME
    _START_TIME = time.time()
    log("=" * 60)
    log(f"Python : {sys.version}")
    log(f"Triton : {triton.__version__}")
    log(f"MAX_ITER={MAX_ITER}  WARMUP={WARMUP}  REPS={REPS}")
    log(f"S0={S0}  K={K}  r={r}  sigma={sigma}  T={T}  type={opttype}")
    log("=" * 60)
    log_progress("Initializing OpenCL ...")
    checkOpenCL()
    log_progress("OpenCL ready.")
    get_ground_truth()
    return os.path.join('benchmarks', 'plots')


def run_benchmarks():
    save_dir = _setup()
    os.makedirs(save_dir, exist_ok=True)
    log_progress("Starting benchmarks ...")
    log("")
    benchmark_pso.run(print_data=True, show_plots=False, save_path=save_dir)
    log("")
    log_progress("All benchmarks finished!")
    log(f"Total wall time: {time.time() - _START_TIME:.1f}s")


def run_fraction_benchmarks():
    save_dir = _setup()
    os.makedirs(save_dir, exist_ok=True)
    log_progress("Starting fraction sweep ...")
    log("")
    benchmark_fraction.run(print_data=True, show_plots=False, save_path=save_dir)
    log("")
    log_progress("All benchmarks finished!")
    log(f"Total wall time: {time.time() - _START_TIME:.1f}s")


if __name__ == "__main__":
    run_benchmarks()
