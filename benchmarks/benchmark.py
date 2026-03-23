import os
import sys
import time
import numpy as np
import triton
import triton.testing

from references.utils import checkOpenCL
from references.mc import hybridMonteCarlo
from references.pso import (
    PSO_OpenCL_scalar,
    PSO_OpenCL_vec,
    PSO_OpenCL_vec_fusion,
)

from flash_pso.flash_pso import FlashPSO
from flash_pso.config import OptionConfig, ComputeConfig, SwarmConfig


def log(msg):
    print(msg, flush=True)

def log_progress(msg):
    elapsed = time.time() - _START_TIME
    print(f"[{elapsed:7.1f}s] {msg}", flush=True)

_START_TIME = time.time()

# benchmark parameters
MAX_ITER = 100
WARMUP   = 3
REPS     = 5

def _opencl_iters(result):
    """len(result[-1]) == number of iterations run for all OpenCL return tuples."""
    return len(result[-1])

def _opencl_solution(result):
    """result[0] == C_hat (gbest_cost) for all OpenCL return tuples."""
    return result[0]


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
        manual_blocks=False,
        max_iterations=MAX_ITER,
        use_fixed_random=True,
        sync_iters=1,
        seed=42,
    )
    swarm = SwarmConfig(num_particles=nFish)
    return FlashPSO(option_config=opt, compute_config=comp, swarm_config=swarm)


@triton.testing.perf_report([
    triton.testing.Benchmark(
        x_names=['nFish'],
        x_vals=[2**i for i in range(6, 13)],
        line_arg='provider',
        line_vals=['cl_vec_fusion', 'triton_custom'],
        line_names=['OpenCL Vec Fusion', 'Triton (Flash PSO)'],
        styles=[('gray', '-'), ('blue', '-'), ('green', '--'), ('purple', '-.')],
        ylabel='Execution Time (ms)',
        plot_name='pso-time',
        args={'nPath': 2**17, 'nPeriod': 2**9, 'metric': 'time'},
    ),
])
def benchmark_pso(nFish, nPath, nPeriod, metric, provider):
    S0, r, sigma, T, K, opttype = 100.0, 0.03, 0.3, 1.0, 110.0, 'P'

    log_progress(f"  running  provider={provider:<16}  nFish={nFish:<6}  nPeriod={nPeriod}  nPath={nPath}")

    if provider == 'triton_custom':
        flash = _make_flash_pso(nFish, nPath, nPeriod, S0, r, sigma, T, K)

        flash.optimize()
        converged_at = flash.global_payoff_index * flash.comp.sync_iters
        solution     = flash.get_option_price()

        fn = lambda: flash.optimize()
        ms, min_ms, max_ms = triton.testing.do_bench(
            fn, warmup=WARMUP - 1, rep=REPS, quantiles=[0.5, 0.2, 0.8]
        )
        log_progress(f"  [triton] done  median={ms:.2f}ms  iters={converged_at}  price={solution:.6f}")

    else:
        mc = hybridMonteCarlo(S0, r, sigma, T, nPath, nPeriod, K, opttype, nFish)

        if provider == 'cl_scalar':
            fn = lambda: PSO_OpenCL_scalar(mc, nFish, direction='backward', iterMax=MAX_ITER).solvePsoAmerOption_cl()
        elif provider == 'cl_vec_4':
            fn = lambda: PSO_OpenCL_vec(mc, nFish, vec_size=4, iterMax=MAX_ITER).solvePsoAmerOption_cl()
        elif provider == 'cl_vec_fusion':
            fn = lambda: PSO_OpenCL_vec_fusion(mc, nFish, iterMax=MAX_ITER).solvePsoAmerOption_cl()
        else:
            raise ValueError(f"Unknown provider: {provider}")

        result = None
        for _ in range(WARMUP):
            result = fn()
        converged_at = _opencl_iters(result)
        solution     = _opencl_solution(result)

        times = []
        for _ in range(REPS):
            t0 = time.perf_counter()
            fn()
            times.append((time.perf_counter() - t0) * 1000)
        ms, min_ms, max_ms = tuple(np.quantile(times, [0.5, 0.2, 0.8]))

        mc.cleanUp()
        log_progress(f"  [{provider}] done  median={ms:.2f}ms  iters={converged_at}  price={solution:.6f}")

    if metric == 'time':
        return ms, min_ms, max_ms
    else:
        bytes_per_float = 4
        memory_operations_per_particle = 7 * nPeriod * bytes_per_float
        total_bytes = memory_operations_per_particle * nFish * MAX_ITER
        gbps = lambda t: total_bytes / (t * 1e-3) / 1e9
        return gbps(ms), gbps(max_ms), gbps(min_ms)


def run_benchmarks():
    global _START_TIME
    _START_TIME = time.time()

    log("=" * 60)
    log("Starting Unified Benchmarks")
    log(f"Python : {sys.version}")
    log(f"Triton : {triton.__version__}")
    log(f"MAX_ITER={MAX_ITER}  WARMUP={WARMUP}  REPS={REPS}")
    log("=" * 60)

    log_progress("Initializing OpenCL ...")
    checkOpenCL()
    log_progress("OpenCL ready.")

    save_dir = os.path.join('benchmarks', 'plots')
    os.makedirs(save_dir, exist_ok=True)

    log_progress("Starting PSO benchmarks ...")
    log("")

    benchmark_pso.run(print_data=True, show_plots=False, save_path=save_dir)

    log("")
    log_progress("All benchmarks finished!")
    log(f"Total wall time: {time.time() - _START_TIME:.1f}s")


if __name__ == "__main__":
    run_benchmarks()
