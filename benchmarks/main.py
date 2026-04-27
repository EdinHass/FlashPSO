import traceback
import torch
import os
import gc
import shutil
from flash_pso.config import OptionConfig, BasketOptionConfig, ComputeConfig, SwarmConfig
from flash_pso.enums import OptionStyle, OptionType, RNGType, ExerciseStyle
from benchmarks.models import Method
from benchmarks.engine import Benchmark, BenchmarkSuite
from benchmarks.wrappers import WRAPPER_REGISTRY

def _get_base_configs():
    problem = OptionConfig(
        initial_stock_price=100.0, strike_price=100.0, risk_free_rate=0.05,
        volatility=0.20, time_to_maturity=2.0, num_paths=2**17,
        num_time_steps=128, option_style=OptionStyle.STANDARD, option_type=OptionType.PUT
    )
    compute = ComputeConfig(
        compute_fraction=0.0, max_iterations=150, sync_iters=1,
        convergence_threshold=-1.0, rng_type=RNGType.PHILOX, use_antithetic=False,
        seed=42, pso_paths_block_size=64
    )
    swarm = SwarmConfig(num_particles=128)
    return problem, compute, swarm


def run_core_bench():
    problem, base_compute, base_swarm = _get_base_configs()
    suite = BenchmarkSuite(title="Core Vanilla Performance & Baselines")

    sobol_comp = ComputeConfig(**{**base_compute.__dict__, "rng_type": RNGType.SOBOL})
    anti_comp = ComputeConfig(**{**base_compute.__dict__, "use_antithetic": True})

    suite.add(Benchmark("FlashPSO Std", Method.FLASH_PSO, problem, base_compute, base_swarm, runs=50))
    suite.add(Benchmark("FlashPSO Anti", Method.FLASH_PSO, problem, anti_comp, base_swarm, runs=50))
    suite.add(Benchmark("FlashPSO Sobol", Method.FLASH_PSO, problem, sobol_comp, base_swarm, runs=50))
    suite.add(Benchmark("Li & Chen PSO (Std)", Method.OPENCL_PSO, problem, base_compute, base_swarm, runs=20))
    suite.add(Benchmark("Li & Chen LSMC", Method.OPENCL_LSMC, problem, base_compute, base_swarm, runs=20))

    suite.run_all()
    suite.report()

def run_compute_frac_sweep():
    problem, base_compute, base_swarm = _get_base_configs()
    suite = BenchmarkSuite(title="Compute Fraction Sweep (Operational Intensity Analysis)")

    path_counts = [2**18] 
    block_sizes = [64, 128, 256]
    fractions = [0.25]

    for paths in path_counts:
        p_cfg = OptionConfig(**{**problem.__dict__, "num_paths": paths})
        
        for bs in block_sizes:
            for frac in fractions:
                c_cfg = ComputeConfig(**{
                    **base_compute.__dict__, 
                    "compute_fraction": frac,
                    "pso_paths_block_size": bs
                })
                
                suite.add(Benchmark(
                    f"FlashPSO Frac={frac} (N={paths}, BS={bs})", 
                    Method.FLASH_PSO, p_cfg, c_cfg, base_swarm, 
                    runs=20
                ))

    suite.run_all()
    suite.report()


def run_early_convergence_sweep():
    problem, base_compute, swarm = _get_base_configs()
    suite = BenchmarkSuite(title="Early Convergence (Low Iteration) Analysis")

    for it in [5, 10, 25, 50, 75, 100, 150, 200, 300]:
        c_std = ComputeConfig(**{**base_compute.__dict__, "max_iterations": it})
        c_anti = ComputeConfig(**{**c_std.__dict__, "use_antithetic": True})
        c_sobol = ComputeConfig(**{**c_std.__dict__, "rng_type": RNGType.SOBOL})

        suite.add(Benchmark(f"Flash Std (it={it})", Method.FLASH_PSO, problem, c_std, swarm, runs=20))
        suite.add(Benchmark(f"Flash Anti (it={it})", Method.FLASH_PSO, problem, c_anti, swarm, runs=20))
        suite.add(Benchmark(f"Flash Sobol (it={it})", Method.FLASH_PSO, problem, c_sobol, swarm, runs=20))
        
        suite.add(Benchmark(f"Li & Chen PSO (it={it})", Method.OPENCL_PSO, problem, c_std, swarm, runs=5))

    suite.add(Benchmark("Li & Chen LSMC (Baseline)", Method.OPENCL_LSMC, problem, base_compute, swarm, runs=5))
    suite.run_all()
    suite.report()


def run_particle_sweep():
    problem, base_compute, _ = _get_base_configs()
    suite = BenchmarkSuite(title="Particle Count × Path Block Size Sweep")
    anti_comp = ComputeConfig(**{**base_compute.__dict__, "use_antithetic": True})

    for n in [64, 128, 256, 512, 1024]:
        swarm = SwarmConfig(num_particles=n)
        for bs in [64, 128, 256]:
            c_std = ComputeConfig(**{**base_compute.__dict__, "pso_paths_block_size": bs})
            c_anti = ComputeConfig(**{**anti_comp.__dict__, "pso_paths_block_size": bs})
            
            suite.add(Benchmark(f"FlashPSO Std (P={n}, BS={bs})",  Method.FLASH_PSO, problem, c_std, swarm, runs=20))
            suite.add(Benchmark(f"FlashPSO Anti (P={n}, BS={bs})", Method.FLASH_PSO, problem, c_anti, swarm, runs=20))
            suite.add(Benchmark(f"FlashPSO Sobol (P={n}, BS={bs})", Method.FLASH_PSO, problem, ComputeConfig(**{**base_compute.__dict__, "rng_type": RNGType.SOBOL, "pso_paths_block_size": bs}), swarm, runs=20))
            
        suite.add(Benchmark(f"Li & Chen PSO (P={n})", Method.OPENCL_PSO, problem, base_compute, swarm, runs=5))

    suite.add(Benchmark("Li & Chen LSMC (Baseline)", Method.OPENCL_LSMC, problem, base_compute, SwarmConfig(num_particles=128), runs=5))
    suite.run_all()
    suite.report()


def run_paths_sweep():
    problem, base_compute, base_swarm = _get_base_configs()
    suite = BenchmarkSuite(title="Path Count × Path Block Size Sweep")
    anti_comp = ComputeConfig(**{**base_compute.__dict__, "use_antithetic": True})

    for paths in [2**14, 2**16, 2**18, 2**20]:
        problem_scaled = OptionConfig(**{**problem.__dict__, "num_paths": paths})
        
        for bs in [64, 128, 256, 512]:
            if bs > paths: continue
            
            c_std = ComputeConfig(**{**base_compute.__dict__, "pso_paths_block_size": bs})
            c_anti = ComputeConfig(**{**anti_comp.__dict__, "pso_paths_block_size": bs})
            
            runs_count = 25 if paths <= 2**16 else 15
            suite.add(Benchmark(f"FlashPSO Std (N={paths}, BS={bs})",  Method.FLASH_PSO, problem_scaled, c_std, base_swarm, runs=runs_count))
            suite.add(Benchmark(f"FlashPSO Anti (N={paths}, BS={bs})", Method.FLASH_PSO, problem_scaled, c_anti, base_swarm, runs=runs_count))
            suite.add(Benchmark(f"FlashPSO Sobol (N={paths}, BS={bs})", Method.FLASH_PSO, problem_scaled, ComputeConfig(**{**base_compute.__dict__, "rng_type": RNGType.SOBOL, "pso_paths_block_size": bs}), base_swarm, runs=runs_count))
            
        baseline_runs = 5 if paths <= 2**18 else 2
        suite.add(Benchmark(f"Li & Chen PSO (N={paths})", Method.OPENCL_PSO, problem_scaled, base_compute, base_swarm, runs=baseline_runs))
        suite.add(Benchmark(f"Li & Chen LSMC (N={paths})", Method.OPENCL_LSMC, problem_scaled, base_compute, base_swarm, runs=baseline_runs))
        suite.add(Benchmark(f"Native Binomial (N={paths})", Method.NATIVE_BINOMIAL, problem_scaled, base_compute, base_swarm, runs=1))

    suite.run_all()
    suite.report()


def run_fp16_paths_sweep():
    problem, base_compute, base_swarm = _get_base_configs()
    suite = BenchmarkSuite(title="FP16 vs FP32 Precomputed Paths Sweep (compute_fraction=0.0)")
    base_swarm = SwarmConfig(num_particles=128)

    for paths in [2**20, 2**22, 2**24]:
        problem_scaled = OptionConfig(**{**problem.__dict__, "num_paths": paths, "num_time_steps": 256})

        for bs in [64, 128]:
            if bs > paths: continue

            c_fp32 = ComputeConfig(**{**base_compute.__dict__, "pso_paths_block_size": bs, "use_fp16_paths": False})
            c_fp16 = ComputeConfig(**{**base_compute.__dict__, "pso_paths_block_size": bs, "use_fp16_paths": True})

            runs_count = 25 if paths <= 2**16 else 15
            suite.add(Benchmark(f"FlashPSO FP32 Paths (N={paths}, BS={bs})", Method.FLASH_PSO, problem_scaled, c_fp32, base_swarm, runs=runs_count))
            suite.add(Benchmark(f"FlashPSO FP16 Paths (N={paths}, BS={bs})", Method.FLASH_PSO, problem_scaled, c_fp16, base_swarm, runs=runs_count))

    suite.run_all()
    suite.report()


def run_timesteps_sweep():
    problem, base_compute, base_swarm = _get_base_configs()
    suite = BenchmarkSuite(title="Timestep Count × Path Block Size Sweep")
    anti_comp = ComputeConfig(**{**base_compute.__dict__, "use_antithetic": True})

    for steps in [32, 64, 128, 256]:
        problem_scaled = OptionConfig(**{**problem.__dict__, "num_time_steps": steps})
        
        for bs in [64, 128, 256, 512]:
            c_std = ComputeConfig(**{**base_compute.__dict__, "pso_paths_block_size": bs})
            c_anti = ComputeConfig(**{**anti_comp.__dict__, "pso_paths_block_size": bs})
            
            suite.add(Benchmark(f"FlashPSO Std (T={steps}, BS={bs})",  Method.FLASH_PSO, problem_scaled, c_std, base_swarm, runs=20))
            suite.add(Benchmark(f"FlashPSO Anti (T={steps}, BS={bs})", Method.FLASH_PSO, problem_scaled, c_anti, base_swarm, runs=20))
            suite.add(Benchmark(f"FlashPSO Sobol (T={steps}, BS={bs})", Method.FLASH_PSO, problem_scaled, ComputeConfig(**{**base_compute.__dict__, "rng_type": RNGType.SOBOL, "pso_paths_block_size": bs}), base_swarm, runs=20))
            
        suite.add(Benchmark(f"Li & Chen PSO (T={steps})", Method.OPENCL_PSO, problem_scaled, base_compute, base_swarm, runs=5))
        suite.add(Benchmark(f"Li & Chen LSMC (T={steps})", Method.OPENCL_LSMC, problem_scaled, base_compute, base_swarm, runs=5))

    suite.run_all()
    suite.report()


def run_sync_iters_sweep():
    problem, base_compute, base_swarm = _get_base_configs()
    suite = BenchmarkSuite(title="Sync Iterations Communication Delay Sweep")
    sobol_comp = ComputeConfig(**{**base_compute.__dict__, "rng_type": RNGType.SOBOL})

    for sync in [1, 5, 10, 20, 25, 50]:
        c_std = ComputeConfig(**{
            **base_compute.__dict__, 
            "sync_iters": sync, 
            "max_iterations": 1000, 
            "convergence_threshold": 1e-6
        })
        c_sobol = ComputeConfig(**{
            **sobol_comp.__dict__,  
            "sync_iters": sync, 
            "max_iterations": 1000, 
            "convergence_threshold": 1e-6
        })
        
        suite.add(Benchmark(f"FlashPSO Std (sync={sync})",   Method.FLASH_PSO, problem, c_std,   base_swarm, runs=20))
        suite.add(Benchmark(f"FlashPSO Anti (sync={sync})",  Method.FLASH_PSO, problem, ComputeConfig(**{**c_std.__dict__, "use_antithetic": True}), base_swarm, runs=20))
        suite.add(Benchmark(f"FlashPSO Sobol (sync={sync})", Method.FLASH_PSO, problem, c_sobol, base_swarm, runs=20))

    suite.add(Benchmark(f"Li & Chen PSO (Baseline)", Method.OPENCL_PSO, problem, base_compute, base_swarm, runs=5))
    suite.add(Benchmark("Li & Chen LSMC (Baseline)", Method.OPENCL_LSMC, problem, base_compute, base_swarm, runs=5))

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
            suite.add(Benchmark(f"FlashPSO {style.name} ({dim}D)", Method.FLASH_PSO, p_cfg, base_compute, base_swarm, runs=20))
        suite.add(Benchmark(f"QuantLib LSMC ({dim}D)", Method.QUANTLIB, basket_cfg, base_compute, base_swarm, runs=2))
        
    suite.run_all()
    suite.report()


def run_moneyness_sweep():
    problem, base_compute, swarm = _get_base_configs()
    suite = BenchmarkSuite(title="Moneyness Sweep (ITM, ATM, OTM)")
    anti_comp = ComputeConfig(**{**base_compute.__dict__, "use_antithetic": True})

    for strike, label in [(120.0, "ITM"), (100.0, "ATM"), (80.0, "OTM")]:
        p_cfg = OptionConfig(**{**problem.__dict__, "strike_price": strike})
        
        suite.add(Benchmark(f"FlashPSO Std ({label}, K={strike})", Method.FLASH_PSO, p_cfg, base_compute, swarm, runs=20))
        suite.add(Benchmark(f"FlashPSO Anti ({label}, K={strike})", Method.FLASH_PSO, p_cfg, anti_comp, swarm, runs=20))
        suite.add(Benchmark(f"FlashPSO Sobol ({label}, K={strike})", Method.FLASH_PSO, p_cfg, ComputeConfig(**{**base_compute.__dict__, "rng_type": RNGType.SOBOL}), swarm, runs=20))
        
        suite.add(Benchmark(f"Li & Chen PSO ({label})", Method.OPENCL_PSO, p_cfg, base_compute, swarm, runs=5))
        suite.add(Benchmark(f"Li & Chen LSMC ({label})", Method.OPENCL_LSMC, p_cfg, base_compute, swarm, runs=5))

    suite.run_all()
    suite.report()

def run_cpu_comparison():
    problem, base_compute, swarm = _get_base_configs()
    suite = BenchmarkSuite(title="CPU vs GPU FlashPSO Comparison")

    suite.add(Benchmark("FlashPSO GPU", Method.FLASH_PSO, problem, base_compute, swarm, runs=20))
    suite.add(Benchmark("FlashPSO CPU", Method.FLASH_PSO_CPU, problem, base_compute, swarm, runs=20))

    suite.run_all()
    suite.report()

def run_iso_work_sweep():
    problem, base_compute, _ = _get_base_configs()
    suite = BenchmarkSuite(title="Iso-Work Arithmetic Intensity Sweep (~16.7M Nodes)")

    configs = [
        (2**19, 32),
        (2**18, 64),
        (2**17, 128),
        (2**16, 256),
        (2**15, 512),
    ]

    for p_count in [32, 64, 128]:
        swarm = SwarmConfig(num_particles=p_count)
        for paths, steps in configs:
            p_cfg = OptionConfig(**{**problem.__dict__, "num_paths": paths, "num_time_steps": steps})
            
            bs = min(256, paths)
            c_std = ComputeConfig(**{**base_compute.__dict__, "pso_paths_block_size": bs})
            
            suite.add(Benchmark(f"FlashPSO Std (P={p_count}, N={paths}, T={steps})", Method.FLASH_PSO, p_cfg, c_std, swarm, runs=20))

    suite.run_all()
    suite.report()


def run_all_benchmarks():
    print("\n" + "="*60)
    print("STARTING FULL FLASH-PSO BENCHMARK SUITE")
    print("="*60 + "\n")

    sweeps = [
        ("Core Vanilla Performance & Baselines",       run_core_bench),
        ("Moneyness Sweep (ITM, ATM, OTM)",            run_moneyness_sweep),
        ("Iso-Work Arithmetic Intensity Sweep",        run_iso_work_sweep),
        ("Path Count Scaling",                         run_paths_sweep),
        ("FP16 vs FP32 Path Precision Sweep",          run_fp16_paths_sweep),
        ("Timestep Count Scaling",                     run_timesteps_sweep),
        ("Early Convergence Analysis",                 run_early_convergence_sweep),
        ("Sync Iterations Convergence Sweep",          run_sync_iters_sweep),
        ("CPU vs GPU FlashPSO Comparison",             run_cpu_comparison),
        ("Particle Count Scaling",                     run_particle_sweep),
        ("Compute Fraction Sweep",                     run_compute_frac_sweep),
    ]

    failures = []
    for i, (name, sweep_func) in enumerate(sweeps):
        try:
            sweep_func()
        except Exception as e:
            print(f"\n[ERROR] '{name}' failure:")
            traceback.print_exc()
            failures.append((name, str(e)))

        # VRAM flush happens every sweep to guarantee OOM safety
        if torch.cuda.is_available():
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            if (i + 1) % 2 == 0:
                triton_cache = os.path.expanduser("~/.triton/cache")
                if os.path.exists(triton_cache):
                    shutil.rmtree(triton_cache, ignore_errors=True)

    # Final cleanup at the very end
    triton_cache = os.path.expanduser("~/.triton/cache")
    if os.path.exists(triton_cache):
        shutil.rmtree(triton_cache, ignore_errors=True)

    print("\n" + "="*60)
    if not failures:
        print(" RUN COMPLETE: All benchmarks executed successfully.")
    else:
        print(f" RUN COMPLETE: Finished with {len(failures)} sweep failures.")
    print("="*60 + "\n")

if __name__ == "__main__":
    run_all_benchmarks()
