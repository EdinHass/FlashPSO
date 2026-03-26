import os
import time
import triton
import triton.testing

# Reference Imports
from references.utils import checkOpenCL
from references.mc import hybridMonteCarlo
from references.pso import PSO_OpenCL_vec_fusion
from references.longstaff import LSMC_OpenCL
from references.benchmarks import binomialAmericanOption

# Flash-PSO Imports
from flash_pso.flash_pso import FlashPSO
from flash_pso.config import OptionConfig, ComputeConfig, SwarmConfig

# ─── CONFIGURATION CONSTANTS ─────────────────────────────────────────────────
MAX_ITER = 2   # Steady-state iteration count
WARMUP   = 10
REPS     = 5

# Market parameters
S0, r, sigma, T, K, OPT_TYPE = 100.0, 0.03, 0.3, 1.0, 110.0, 'P'

def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

_REFERENCE_PRICE = None
def get_reference_price() -> float:
    global _REFERENCE_PRICE
    if _REFERENCE_PRICE is None:
        log("Calculating reference price (Binomial 20,000 steps)...")
        price, _ = binomialAmericanOption(S0, K, r, sigma, 20000, T, opttype=OPT_TYPE)
        _REFERENCE_PRICE = float(price)
        log(f"Reference Price Established: {_REFERENCE_PRICE:.6f}")
    return _REFERENCE_PRICE

def create_flash_pso(n_fish: int, n_path: int, n_period: int, compute_fraction: float = 1.0) -> FlashPSO:
    opt = OptionConfig(S0, K, r, sigma, T, n_path, n_period, option_type=1)
    comp = ComputeConfig(
        compute_fraction=compute_fraction,
        max_iterations=MAX_ITER,
        sync_iters=1,
        use_fixed_random=False,
        use_antithetic=True,
        convergence_threshold=1e-6,
        seed=1337
    )
    return FlashPSO(opt, comp, SwarmConfig(n_fish))

def run_bench(fn) -> tuple[float, float, float]:
    ms, min_ms, max_ms = triton.testing.do_bench(fn, warmup=WARMUP, rep=REPS, quantiles=[0.5, 0.2, 0.8])
    return ms / MAX_ITER, min_ms / MAX_ITER, max_ms / MAX_ITER

def print_telemetry(test_name: str, identifier: str, n_path: int, ms_iter: float, price: float, total_iters: int):
    ref = get_reference_price()
    err = abs(price - ref)
    log(f"[{test_name}] {identifier:<18} | Paths: {n_path:<7} | Iters: {total_iters:<4} | Iter: {ms_iter:7.4f}ms | Price: {price:8.4f} | Err: {err:7.4f}")

# ═════════════════════════════════════════════════════════════════════════════
# ─── ONE-OFF DIAGNOSTIC BENCHMARK ────────────────────────────────────────────
# ═════════════════════════════════════════════════════════════════════════════

def exec_single_test():
    """Runs a single configured simulation to quickly test accuracy and performance."""
    log("-" * 65)
    log("ONE-OFF DIAGNOSTIC RUN")
    log("-" * 65)
    
    # Configure your one-off test variables here
    n_fish = 128
    n_path = 65536
    n_period = 256
    frac = 0.25
    
    log(f"Configuration : Fish={n_fish}, Paths={n_path}, Steps={n_period}, Fraction={frac}")
    flash = create_flash_pso(n_fish, n_path, n_period, frac)
    
    t0 = time.perf_counter()
    flash.optimize()
    t1 = time.perf_counter()
    
    total_time_ms = (t1 - t0) * 1000
    ms_per_iter   = total_time_ms / MAX_ITER
    price         = flash.get_option_price()
    ref           = get_reference_price()
    abs_err       = abs(price - ref)
    rel_err       = (abs_err / ref) * 100
    iters_ran     = flash.global_payoff_index * flash.comp.sync_iters
    
    print()
    print(f"Total Time       : {total_time_ms:.2f} ms")
    print(f"Time / Iteration : {ms_per_iter:.4f} ms")
    print(f"Total Iterations : {iters_ran}")
    print(f"Option Price     : {price:.6f}")
    print(f"Reference Price  : {ref:.6f}")
    print(f"Absolute Error   : {abs_err:.6f}")
    print(f"Relative Error   : {rel_err:.4f}%")
    print()
    log("-" * 65)

# ═════════════════════════════════════════════════════════════════════════════
# ─── BENCHMARK 1: BASELINE COMPARISON ────────────────────────────────────────
# ═════════════════════════════════════════════════════════════════════════════

@triton.testing.perf_report([
    triton.testing.Benchmark(
        x_names=['nPath'], x_vals=[2**i for i in range(12, 21)],
        line_arg='provider', line_vals=['cl_pso', 'lsmc_cl', 'triton_hybrid_25', 'triton_compute_100'],
        line_names=['OpenCL Vec Fusion', 'LSMC OpenCL', 'FlashPSO (25%)', 'FlashPSO (100%)'],
        styles=[('black', ':'), ('gray', '--'), ('cyan', '-'), ('blue', '-')],
        ylabel='ms / Iteration', plot_name='01_baseline_comparison',
        args={'nFish': 64, 'nPeriod': 256}
    )
])
def bench_baseline(nPath, nFish, nPeriod, provider):
    if provider == 'cl_pso':
        mc = hybridMonteCarlo(S0, r, sigma, T, nPath, nPeriod, K, OPT_TYPE, nFish)
        pso = PSO_OpenCL_vec_fusion(mc, nFish, iterMax=MAX_ITER)
        median, min_ms, max_ms = run_bench(lambda: pso.solvePsoAmerOption_cl())
        price = pso.solvePsoAmerOption_cl()[0]
        print_telemetry("Baseline", "cl_pso", nPath, median, price, MAX_ITER)
        mc.cleanUp()
        return median, min_ms, max_ms
        
    elif provider == 'lsmc_cl':
        mc = hybridMonteCarlo(S0, r, sigma, T, nPath, nPeriod, K, OPT_TYPE, nFish)
        lsmc = LSMC_OpenCL(mc, 'GJ')
        median, min_ms, max_ms = run_bench(lambda: lsmc.longstaff_schwartz_itm_path_fast_hybrid())
        price = lsmc.longstaff_schwartz_itm_path_fast_hybrid()[0]
        # LSMC is a 1-shot solver, we log it as 1 iteration.
        print_telemetry("Baseline", "lsmc_cl", nPath, median, price, 1)
        mc.cleanUp()
        return median, min_ms, max_ms
        
    else:
        frac = 0.25 if '25' in provider else 1.0
        flash = create_flash_pso(nFish, nPath, nPeriod, compute_fraction=frac)
        median, min_ms, max_ms = run_bench(lambda: flash.optimize())
        iters = flash.global_payoff_index * flash.comp.sync_iters
        print_telemetry("Baseline", f"flash_{frac}", nPath, median, flash.get_option_price(), iters)
        return median, min_ms, max_ms

# ═════════════════════════════════════════════════════════════════════════════
# ─── BENCHMARK 2: ARITHMETIC INTENSITY SWEEP ─────────────────────────────────
# ═════════════════════════════════════════════════════════════════════════════

@triton.testing.perf_report([
    triton.testing.Benchmark(
        x_names=['nPath'], x_vals=[2**i for i in range(19, 20)],
        line_arg='frac', line_vals=[0.0, 0.25, 0.5, 0.75, 1.0],
        line_names=['Bandwidth (0%)', 'Hybrid (25%)', 'Hybrid (50%)', 'Hybrid (75%)', 'Compute (100%)'],
        styles=[('green', '--'), ('cyan', '-'), ('orange', '-'), ('purple', '-'), ('blue', '-')],
        ylabel='ms / Iteration', plot_name='02_compute_fraction_sweep',
        args={'nFish': 64, 'nPeriod': 64}
    )
])
def bench_fractions(nPath, nFish, nPeriod, frac):
    flash = create_flash_pso(nFish, nPath, nPeriod, frac)
    median, min_ms, max_ms = run_bench(lambda: flash.optimize())
    iters = flash.global_payoff_index * flash.comp.sync_iters
    print_telemetry("Sweep", f"frac_{frac}", nPath, median, flash.get_option_price(), iters)
    return median, min_ms, max_ms

# ═════════════════════════════════════════════════════════════════════════════
# ─── BENCHMARK 3: PARTICLE DENSITY SCALING ───────────────────────────────────
# ═════════════════════════════════════════════════════════════════════════════

@triton.testing.perf_report([
    triton.testing.Benchmark(
        x_names=['nFish'], x_vals=[32, 64, 128, 256, 512, 1024],
        line_arg='frac', line_vals=[0.0, 1.0],
        line_names=['Bandwidth-Bound (0%)', 'Compute-Bound (100%)'],
        ylabel='ms / Iteration', plot_name='03_particle_density_scaling',
        args={'nPath': 2**16, 'nPeriod': 128}
    )
])
def bench_density(nPath, nFish, nPeriod, frac):
    flash = create_flash_pso(nFish, nPath, nPeriod, frac)
    median, min_ms, max_ms = run_bench(lambda: flash.optimize())
    iters = flash.global_payoff_index * flash.comp.sync_iters
    ref = get_reference_price()
    price = flash.get_option_price()
    err = abs(price - ref)
    log(f"[Density] frac_{frac:<13} | Fish: {nFish:<8} | Iters: {iters:<4} | Iter: {median:7.4f}ms | Price: {price:8.4f} | Err: {err:7.4f}")
    return median, min_ms, max_ms

# ═════════════════════════════════════════════════════════════════════════════
# ─── BENCHMARK 4: ACCURACY AND CONVERGENCE TEST ──────────────────────────────
# ═════════════════════════════════════════════════════════════════════════════

def exec_accuracy_test():
    log("-" * 65)
    log("ACCURACY AND CONVERGENCE TEST")
    log("-" * 65)
    reference = get_reference_price()
    paths = [2**i for i in range(12, 21)]
    print(f"{'Paths':<12} | {'Iters':<6} | {'PSO Price':<12} | {'Absolute Error':<16} | {'Relative Error':<12}")
    print("-" * 65)
    for p in paths:
        flash = create_flash_pso(n_fish=128, n_path=p, n_period=512, compute_fraction=1.0)
        flash.comp.max_iterations = 200 
        flash.optimize()
        price = flash.get_option_price()
        abs_err = abs(price - reference)
        iters_ran = flash.global_payoff_index * flash.comp.sync_iters
        print(f"{p:<12} | {iters_ran:<6} | {price:12.6f} | {abs_err:16.6f} | {(abs_err / reference) * 100:11.4f}%")
    log("-" * 65)

# ═════════════════════════════════════════════════════════════════════════════
# ─── RUNNERS (Mapped to CLI commands) ────────────────────────────────────────
# ═════════════════════════════════════════════════════════════════════════════

def _setup_env():
    checkOpenCL()
    save_dir = os.path.join('benchmarks', 'results')
    os.makedirs(save_dir, exist_ok=True)
    return save_dir

def run_single():
    _setup_env()
    exec_single_test()

def run_baseline():
    save_dir = _setup_env()
    log("Running: Baseline Comparison...")
    bench_baseline.run(print_data=True, show_plots=False, save_path=save_dir)

def run_fraction_sweep():
    save_dir = _setup_env()
    log("Running: Arithmetic Intensity Sweep...")
    bench_fractions.run(print_data=True, show_plots=False, save_path=save_dir)

def run_particle_scaling():
    save_dir = _setup_env()
    log("Running: Particle Density Scaling...")
    bench_density.run(print_data=True, show_plots=False, save_path=save_dir)

def run_accuracy():
    _setup_env()
    exec_accuracy_test()

def run_all():
    save_dir = _setup_env()
    log("=" * 65)
    log("STARTING FULL FLASH-PSO SUITE")
    log("=" * 65)
    exec_single_test()
    bench_baseline.run(print_data=True, show_plots=False, save_path=save_dir)
    bench_fractions.run(print_data=True, show_plots=False, save_path=save_dir)
    bench_density.run(print_data=True, show_plots=False, save_path=save_dir)
    exec_accuracy_test()
    log("All benchmarks completed successfully.")

if __name__ == "__main__":
    run_all()
