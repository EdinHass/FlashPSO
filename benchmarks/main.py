import traceback
import torch
from flash_pso.config import OptionConfig, BasketOptionConfig, ComputeConfig, SwarmConfig
from flash_pso.enums import OptionStyle, OptionType, RNGType, ExerciseStyle
from benchmarks.models import Method
from benchmarks.engine import Benchmark, BenchmarkSuite
from benchmarks.wrappers import WRAPPER_REGISTRY


def _get_base_configs():
    problem = OptionConfig(
        initial_stock_price=100.0, strike_price=100.0, risk_free_rate=0.05,
        volatility=0.30, time_to_maturity=2.0, num_paths=2**18,
        num_time_steps=128, option_style=OptionStyle.STANDARD, option_type=OptionType.PUT
    )
    compute = ComputeConfig(
        compute_fraction=0.0, max_iterations=30, sync_iters=5,
        convergence_threshold=-1, rng_type=RNGType.PHILOX, use_antithetic=False,
        seed=42
    )
    swarm = SwarmConfig(num_particles=128)
    return problem, compute, swarm

def run_core_bench():
    problem, base_compute, base_swarm = _get_base_configs()
    suite = BenchmarkSuite(title="Core Vanilla Performance & Baselines")

    sobol_comp = ComputeConfig(**{**base_compute.__dict__, "rng_type": RNGType.SOBOL})
    anti_comp = ComputeConfig(**{**base_compute.__dict__, "use_antithetic": True})

    suite.add(Benchmark("FlashPSO Sobol", Method.FLASH_PSO, problem, sobol_comp, base_swarm, runs=50))
    suite.add(Benchmark("FlashPSO Anti", Method.FLASH_PSO, problem, anti_comp, base_swarm, runs=50))
    suite.add(Benchmark("FlashPSO Std", Method.FLASH_PSO, problem, base_compute, base_swarm, runs=50))

    suite.add(Benchmark("OpenCL PSO (Std)", Method.OPENCL_PSO, problem, base_compute, base_swarm, runs=15))
    suite.add(Benchmark("OpenCL LSMC", Method.OPENCL_LSMC, problem, base_compute, base_swarm, runs=15))

    suite.run_all()
    suite.report()


def run_early_convergence_sweep():
    problem, base_compute, swarm = _get_base_configs()
    suite = BenchmarkSuite(title="Early Convergence (Low Iteration) Analysis")

    for it in [5, 10, 15, 25, 50, 100, 150]:
        c_std = ComputeConfig(**{**base_compute.__dict__, "max_iterations": it, "convergence_threshold": -1})
        c_anti = ComputeConfig(**{**c_std.__dict__, "use_antithetic": True})
        c_sobol = ComputeConfig(**{**c_std.__dict__, "rng_type": RNGType.SOBOL})

        suite.add(Benchmark(f"Flash Std (it={it})", Method.FLASH_PSO, problem, c_std, swarm, runs=20))
        suite.add(Benchmark(f"Flash Anti (it={it})", Method.FLASH_PSO, problem, c_anti, swarm, runs=20))
        suite.add(Benchmark(f"Flash Sobol (it={it})", Method.FLASH_PSO, problem, c_sobol, swarm, runs=20))
        suite.add(Benchmark(f"OpenCL PSO (it={it})", Method.OPENCL_PSO, problem, c_std, swarm, runs=10))

    suite.run_all()
    suite.report()

def run_path_block_sweep():
    problem, base_compute, swarm = _get_base_configs()
    suite = BenchmarkSuite(title="PSO Path Block Size Sweep")

    for bs in [4, 32, 128, 256, 512]:
        comp = ComputeConfig(**{**base_compute.__dict__, "pso_paths_block_size": bs})
        suite.add(Benchmark(f"FlashPSO (path_bs={bs})", Method.FLASH_PSO, problem, comp, swarm, runs=15))

    suite.run_all()
    suite.report()


def run_full_path_block_ablation():
    problem, base_compute, swarm = _get_base_configs()
    suite = BenchmarkSuite(title="Full Path Block Size Ablation")

    for bs in [problem.num_paths // 8, problem.num_paths // 4, problem.num_paths // 2, problem.num_paths]:
        comp = ComputeConfig(**{**base_compute.__dict__, "pso_paths_block_size": bs})
        suite.add(Benchmark(f"FlashPSO (path_bs={bs})", Method.FLASH_PSO, problem, comp, swarm, runs=15))

    suite.run_all()
    suite.report()

def run_cpu_ablation():
    problem, base_compute, swarm = _get_base_configs()
    suite = BenchmarkSuite(title="Reduction+Generation Location Ablation (GPU vs CPU)")

    suite.add(Benchmark("FlashPSO (CPU Redux)", Method.FLASH_PSO_CPU, problem, base_compute, swarm, runs=20))
    suite.add(Benchmark("FlashPSO (GPU Redux)", Method.FLASH_PSO, problem, base_compute, swarm, runs=20))

    suite.run_all()
    suite.report()

def run_particle_sweep():
    _, base_compute, _ = _get_base_configs()
    suite = BenchmarkSuite(title="Particle Count Scaling")
    # 2^18 paths: large enough for meaningful accuracy, fast enough for 5 particle counts × OpenCL runs
    problem = OptionConfig(
        initial_stock_price=100.0, strike_price=100.0, risk_free_rate=0.05,
        volatility=0.30, time_to_maturity=2.0, num_paths=2**18,
        num_time_steps=128, option_style=OptionStyle.STANDARD, option_type=OptionType.PUT
    )
    anti_comp = ComputeConfig(**{**base_compute.__dict__, "use_antithetic": True})

    for n in [32, 64, 128, 256, 512]:
        swarm = SwarmConfig(num_particles=n)
        suite.add(Benchmark(f"FlashPSO Std (P={n})",  Method.FLASH_PSO,   problem, base_compute, swarm, runs=30))
        suite.add(Benchmark(f"FlashPSO Anti (P={n})", Method.FLASH_PSO,   problem, anti_comp,    swarm, runs=30))
        suite.add(Benchmark(f"OpenCL PSO (P={n})",    Method.OPENCL_PSO,  problem, base_compute, swarm, runs=5))

    suite.run_all()
    suite.report()


def run_paths_sweep():
    _, base_compute, base_swarm = _get_base_configs()
    suite = BenchmarkSuite(title="Path Count Scaling")
    sobol_comp = ComputeConfig(**{**base_compute.__dict__, "rng_type": RNGType.SOBOL})

    for paths in [2**14, 2**16, 2**18, 2**20]:
        problem = OptionConfig(
            initial_stock_price=100.0, strike_price=100.0, risk_free_rate=0.05,
            volatility=0.30, time_to_maturity=2.0, num_paths=paths,
            num_time_steps=128, option_style=OptionStyle.STANDARD, option_type=OptionType.PUT
        )
        suite.add(Benchmark(f"FlashPSO Std (N={paths})",   Method.FLASH_PSO,    problem, base_compute, base_swarm, runs=30))
        suite.add(Benchmark(f"FlashPSO Sobol (N={paths})", Method.FLASH_PSO,    problem, sobol_comp,   base_swarm, runs=20))
        suite.add(Benchmark(f"OpenCL PSO (N={paths})",     Method.OPENCL_PSO,   problem, base_compute, base_swarm, runs=5))
        suite.add(Benchmark(f"OpenCL LSMC (N={paths})",    Method.OPENCL_LSMC,  problem, base_compute, base_swarm, runs=5))

    suite.run_all()
    suite.report()


def run_timesteps_sweep():
    _, base_compute, base_swarm = _get_base_configs()
    suite = BenchmarkSuite(title="Timestep Count Scaling")
    sobol_comp = ComputeConfig(**{**base_compute.__dict__, "rng_type": RNGType.SOBOL})

    for steps in [32, 64, 128, 256]:
        problem = OptionConfig(
            initial_stock_price=100.0, strike_price=100.0, risk_free_rate=0.05,
            volatility=0.30, time_to_maturity=2.0, num_paths=2**18,
            num_time_steps=steps, option_style=OptionStyle.STANDARD, option_type=OptionType.PUT
        )
        suite.add(Benchmark(f"FlashPSO Std (T={steps})",   Method.FLASH_PSO,    problem, base_compute, base_swarm, runs=30))
        suite.add(Benchmark(f"FlashPSO Sobol (T={steps})", Method.FLASH_PSO,    problem, sobol_comp,   base_swarm, runs=20))
        suite.add(Benchmark(f"OpenCL PSO (T={steps})",     Method.OPENCL_PSO,   problem, base_compute, base_swarm, runs=5))
        suite.add(Benchmark(f"OpenCL LSMC (T={steps})",    Method.OPENCL_LSMC,  problem, base_compute, base_swarm, runs=5))

    suite.run_all()
    suite.report()


def run_sync_iters_sweep():
    problem, base_compute, base_swarm = _get_base_configs()
    suite = BenchmarkSuite(title="Sync Iterations Convergence Sweep")
    # convergence_threshold enabled — observes how sync granularity affects how
    # early the algorithm stops and the resulting accuracy.
    # All values must divide max_iterations=150.
    sobol_comp = ComputeConfig(**{**base_compute.__dict__, "rng_type": RNGType.SOBOL})

    for sync in [1, 5, 10, 25, 50]:
        c_std   = ComputeConfig(**{**base_compute.__dict__, "sync_iters": sync, "convergence_threshold": 1e-6})
        c_sobol = ComputeConfig(**{**sobol_comp.__dict__,  "sync_iters": sync, "convergence_threshold": 1e-6})
        suite.add(Benchmark(f"FlashPSO Std (sync={sync})",   Method.FLASH_PSO, problem, c_std,   base_swarm, runs=20))
        suite.add(Benchmark(f"FlashPSO Sobol (sync={sync})", Method.FLASH_PSO, problem, c_sobol, base_swarm, runs=20))

    suite.run_all()
    suite.report()


def run_compute_sweep():
    base_problem, base_compute, swarm = _get_base_configs()
    suite = BenchmarkSuite(title="Compute Fraction × Path Volume")
    for paths in [2**24]:
        for path_block in [64]:
            problem = OptionConfig(**{**base_problem.__dict__, "num_paths": paths})
            for frac in [0.0, 0.25]:
                std_comp = ComputeConfig(**{**base_compute.__dict__, "compute_fraction": frac, "pso_paths_block_size": path_block})
                suite.add(Benchmark(f"Flash (f={frac}, N={paths}, path_block={path_block})", Method.FLASH_PSO,   problem, std_comp,     swarm, runs=30))
    suite.run_all()
    suite.report()

def run_basket_sweep():
    _, base_compute, base_swarm = _get_base_configs()
    suite = BenchmarkSuite(title="Multi-Asset Basket Options Sweep")
    for dim in [4, 16]:
        basket_cfg = BasketOptionConfig(
            initial_stock_prices=[100.0] * dim, strike_price=100.0, risk_free_rate=0.05,
            volatilities=[0.30] * dim, weights=[1.0/dim] * dim,
            correlation_matrix=[[1.0 if i==j else 0.5 for j in range(dim)] for i in range(dim)],
            time_to_maturity=2.0, num_paths=32768 if dim==4 else 8192,
            num_time_steps=64, option_style=OptionStyle.BASKET, option_type=OptionType.PUT
        )
        for style in [ExerciseStyle.SCALAR, ExerciseStyle.PER_ASSET]:
            p_cfg = BasketOptionConfig(**{**basket_cfg.__dict__, "exercise_style": style})
            suite.add(Benchmark(f"FlashPSO {style.name} ({dim}D)", Method.FLASH_PSO, p_cfg, base_compute, base_swarm, runs=15))
        suite.add(Benchmark(f"QuantLib LSMC ({dim}D)", Method.QUANTLIB, basket_cfg, base_compute, base_swarm, runs=1))
    suite.run_all()
    suite.report()


def run_all_benchmarks():
    print("\n" + "="*60)
    print("STARTING FULL FLASH-PSO BENCHMARK SUITE (INC. ABLATIONS)")
    print("="*60 + "\n")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    sweeps = [
        #("Core Vanilla Performance & Baselines",       run_core_bench),
        #("Particle Count Scaling",                     run_particle_sweep),
        #("Path Count Scaling",                         run_paths_sweep),
        #("Timestep Count Scaling",                     run_timesteps_sweep),
        ("Compute Fraction × Path Volume",              run_compute_sweep),
        #("Early Convergence Analysis",                 run_early_convergence_sweep),
        #("Sync Iterations Convergence Sweep",          run_sync_iters_sweep),
        #("Path Block Size Sweep",                      run_path_block_sweep),
        #("CPU vs GPU Reduction Ablation",              run_cpu_ablation),
        #("Full Path Block Size Ablation",              run_full_path_block_ablation),
        #("Multi-Asset Basket Options Sweep",           run_basket_sweep),
    ]

    failures = []
    for name, sweep_func in sweeps:
        try:
            sweep_func()
        except Exception as e:
            print(f"\n[ERROR] '{name}' failure:")
            traceback.print_exc()
            failures.append((name, str(e)))

    print("\n" + "="*60)
    if not failures:
        print(" RUN COMPLETE: All benchmarks executed successfully.")
    else:
        print(f" RUN COMPLETE: Finished with {len(failures)} sweep failures.")
    print("="*60 + "\n")

if __name__ == "__main__":
    run_all_benchmarks()
