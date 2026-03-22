import os
import sys
import time
import numpy as np
import triton
import triton.testing

# Baseline Imports
from references.utils import checkOpenCL
from references.mc import hybridMonteCarlo
from references.longstaff import LSMC_Numpy, LSMC_OpenCL
from references.pso import (
    PSO_Numpy,
    PSO_OpenCL_scalar,
    PSO_OpenCL_vec,
    PSO_OpenCL_vec_fusion,
)

# FlashPSO imports
from flash_pso.flash_pso import FlashPSO
from flash_pso.config import OptionConfig, ComputeConfig, SwarmConfig


def log(msg):
    print(msg, flush=True)

def log_progress(msg):
    elapsed = time.time() - _START_TIME
    print(f"[{elapsed:7.1f}s] {msg}", flush=True)

_START_TIME = time.time()


def do_bench_opencl(fn, warmup=5, rep=20, quantiles=[0.5, 0.2, 0.8]):
    """Reduced reps vs original — still statistically sound for slow OpenCL."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(rep):
        start = time.perf_counter()
        fn()
        end = time.perf_counter()
        times.append((end - start) * 1000)
    return tuple(np.quantile(times, quantiles))


def _make_flash_pso(nFish, nPath, nPeriod, S0, r, sigma, T, K):
    opt = OptionConfig(
        initial_stock_price=S0,
        strike_price=K,
        risk_free_rate=r,
        volatility=sigma,
        time_to_maturity=T,
        num_paths=nPath,
        num_time_steps=nPeriod,
        option_type=1,
    )
    comp = ComputeConfig(
        compute_on_the_fly=True,
        manual_blocks=False, # let autotuning find the best block sizes for each configuration
        max_iterations=200,
        sync_iters=10,
        seed=42,
    )
    swarm = SwarmConfig(num_particles=nFish)
    return FlashPSO(option_config=opt, compute_config=comp, swarm_config=swarm)


@triton.testing.perf_report([
    triton.testing.Benchmark(
        x_names=['nFish'],
        x_vals=[2**i for i in range(6, 13)],
        line_arg='provider',
        line_vals=['cl_scalar', 'cl_vec_4', 'cl_vec_fusion', 'triton_custom'],
        line_names=['OpenCL Scalar', 'OpenCL Vec (sz=4)', 'OpenCL Vec Fusion', 'Triton (Flash PSO)'],
        styles=[('blue', '-'), ('red', '-'), ('green', '--'), ('purple', '-.'), ('orange', '-')],
        ylabel='Execution Time (ms)',
        plot_name='pso-time',
        args={'nPath': 2**17, 'nPeriod': 2**8, 'metric': 'time'},
    ),
])
def benchmark_pso(nFish, nPath, nPeriod, metric, provider):
    S0, r, sigma, T, K, opttype = 100.0, 0.03, 0.3, 1.0, 110.0, 'P'
    WARMUP_REPS = 3
    REPETITIONS = 5

    log_progress(f"  running  provider={provider:<16}  nFish={nFish:<6}  nPeriod={nPeriod}  nPath={nPath}")

    if provider == 'triton_custom':
        flash = _make_flash_pso(nFish, nPath, nPeriod, S0, r, sigma, T, K)
        fn = lambda: flash.optimize()
        ms, min_ms, max_ms = triton.testing.do_bench(fn, warmup=WARMUP_REPS, rep=REPETITIONS,
                                                      quantiles=[0.5, 0.2, 0.8])
        log_progress(f"  [triton] done  median={ms:.2f}ms")

    else:
        mc = hybridMonteCarlo(S0, r, sigma, T, nPath, nPeriod, K, opttype, nFish)
        if provider == 'numpy':
            pso = PSO_Numpy(mc, nFish)
            fn = lambda: pso.solvePsoAmerOption_np()
        elif provider == 'cl_scalar':
            fn = lambda: PSO_OpenCL_scalar(mc, nFish, direction='backward').solvePsoAmerOption_cl()
        elif provider == 'cl_vec_4':
            fn = lambda: PSO_OpenCL_vec(mc, nFish, vec_size=4).solvePsoAmerOption_cl()
        elif provider == 'cl_vec_fusion':
            fn = lambda: PSO_OpenCL_vec_fusion(mc, nFish).solvePsoAmerOption_cl()
        else:
            raise ValueError(f"Unknown provider: {provider}")
        ms, min_ms, max_ms = do_bench_opencl(fn, quantiles=[0.5, 0.2, 0.8], rep=REPETITIONS, warmup=WARMUP_REPS)
        mc.cleanUp()
        log_progress(f"  [{provider}] done  median={ms:.2f}ms")

    if metric == 'time':
        return ms, min_ms, max_ms
    else:
        max_iter = 100
        bytes_per_float = 4
        memory_operations_per_particle = 7 * nPeriod * bytes_per_float
        total_bytes_accessed = memory_operations_per_particle * nFish * max_iter
        gbps = lambda t: total_bytes_accessed / (t * 1e-3) / 1e9
        return gbps(ms), gbps(max_ms), gbps(min_ms)


def run_benchmarks():
    global _START_TIME
    _START_TIME = time.time()

    log("=" * 60)
    log("Starting Unified Benchmarks")
    log(f"Python: {sys.version}")
    log(f"Triton: {triton.__version__}")
    log("=" * 60)

    log_progress("Initializing OpenCL ...")
    checkOpenCL()
    log_progress("OpenCL ready.")

    save_dir = os.path.join('benchmarks', 'plots')
    os.makedirs(save_dir, exist_ok=True)

    total_configs = 2 * 8 * 5
    log_progress(f"Starting PSO benchmarks ({total_configs} total configs). Saving to {save_dir} ...")
    log("")

    benchmark_pso.run(print_data=True, show_plots=False, save_path=save_dir)

    log("")
    log_progress("All benchmarks finished successfully!")
    log(f"Total wall time: {time.time() - _START_TIME:.1f}s")


if __name__ == "__main__":
    run_benchmarks()
