import os
import traceback
from flash_pso.config import OptionConfig, BasketOptionConfig, ComputeConfig, SwarmConfig
from flash_pso.enums import OptionStyle, OptionType, RNGType, ExerciseStyle
from benchmarks.models import Method
from benchmarks.engine import Benchmark, BenchmarkSuite

def _get_base_configs():
    problem = OptionConfig(
        initial_stock_price=100.0, strike_price=100.0, risk_free_rate=0.05, 
        volatility=0.30, time_to_maturity=2.0, num_paths=131072, 
        num_time_steps=128, option_style=OptionStyle.STANDARD, option_type=OptionType.PUT
    )
    compute = ComputeConfig(
        compute_fraction=0.0, max_iterations=150, sync_iters=5,
        convergence_threshold=-1, rng_type=RNGType.PHILOX, use_antithetic=False
    )
    swarm = SwarmConfig(num_particles=128)
    return problem, compute, swarm

def run_core_bench():
    problem, base_compute, base_swarm = _get_base_configs()
    suite = BenchmarkSuite(title="Core Vanilla Performance & Baselines")

    sobol_comp = ComputeConfig(**{**base_compute.__dict__, "rng_type": RNGType.SOBOL, "use_antithetic": False})
    suite.add(Benchmark("FlashPSO Sobol (Vanilla)", Method.FLASH_PSO, problem, sobol_comp, base_swarm, runs=50))
    
    anti_comp = ComputeConfig(**{**base_compute.__dict__, "rng_type": RNGType.PHILOX, "use_antithetic": True})
    suite.add(Benchmark("FlashPSO Anti (Vanilla)", Method.FLASH_PSO, problem, anti_comp, base_swarm, runs=50))

    std_comp = ComputeConfig(**{**base_compute.__dict__, "rng_type": RNGType.PHILOX, "use_antithetic": False})
    suite.add(Benchmark("FlashPSO Std (Vanilla)", Method.FLASH_PSO, problem, std_comp, base_swarm, runs=50))

    suite.add(Benchmark("OpenCL PSO (Vanilla)", Method.OPENCL_PSO, problem, base_compute, base_swarm, runs=50))
    suite.add(Benchmark("OpenCL LSMC (Vanilla)", Method.OPENCL_LSMC, problem, base_compute, base_swarm, runs=50))
    suite.add(Benchmark("QuantLib CRR (20k)", Method.QUANTLIB, problem, base_compute, base_swarm, runs=1))

    suite.run_all()
    suite.report()

def run_basket_sweep():
    _, base_compute, base_swarm = _get_base_configs()
    suite = BenchmarkSuite(title="Multi-Asset Basket Options Sweep")
    
    sobol_comp = ComputeConfig(**{**base_compute.__dict__, "rng_type": RNGType.SOBOL, "use_antithetic": False})
    anti_comp = ComputeConfig(**{**base_compute.__dict__, "rng_type": RNGType.PHILOX, "use_antithetic": True})
    std_comp = ComputeConfig(**{**base_compute.__dict__, "rng_type": RNGType.PHILOX, "use_antithetic": False})
    
    n_4 = 4
    basket_4d = BasketOptionConfig(
        initial_stock_prices=[100.0] * n_4, strike_price=100.0, risk_free_rate=0.05, 
        volatilities=[0.30] * n_4, weights=[1.0/n_4] * n_4, 
        correlation_matrix=[[1.0 if i==j else 0.5 for j in range(n_4)] for i in range(n_4)],
        time_to_maturity=2.0, num_paths=65536, num_time_steps=64, 
        option_style=OptionStyle.BASKET, option_type=OptionType.PUT
    )
    
    # 4D Scalar
    basket_4d_scalar = BasketOptionConfig(**{**basket_4d.__dict__, "exercise_style": ExerciseStyle.SCALAR})
    suite.add(Benchmark("FlashPSO Sobol (4D Scalar)", Method.FLASH_PSO, basket_4d_scalar, sobol_comp, base_swarm, runs=20))
    suite.add(Benchmark("FlashPSO Anti  (4D Scalar)", Method.FLASH_PSO, basket_4d_scalar, anti_comp, base_swarm, runs=20))
    suite.add(Benchmark("FlashPSO Std   (4D Scalar)", Method.FLASH_PSO, basket_4d_scalar, std_comp, base_swarm, runs=20))
    
    # 4D Per-Asset
    basket_4d_per_asset = BasketOptionConfig(**{**basket_4d.__dict__, "exercise_style": ExerciseStyle.PER_ASSET})
    suite.add(Benchmark("FlashPSO Sobol (4D PerAsset)", Method.FLASH_PSO, basket_4d_per_asset, sobol_comp, base_swarm, runs=20))
    suite.add(Benchmark("FlashPSO Anti  (4D PerAsset)", Method.FLASH_PSO, basket_4d_per_asset, anti_comp, base_swarm, runs=20))
    suite.add(Benchmark("FlashPSO Std   (4D PerAsset)", Method.FLASH_PSO, basket_4d_per_asset, std_comp, base_swarm, runs=20))
    
    suite.add(Benchmark("QuantLib LSMC  (4D Baseline)", Method.QUANTLIB, basket_4d, base_compute, base_swarm, runs=1))

    n_16 = 16
    basket_16d = BasketOptionConfig(
        initial_stock_prices=[100.0] * n_16, strike_price=100.0, risk_free_rate=0.05, 
        volatilities=[0.30] * n_16, weights=[1.0/n_16] * n_16, 
        correlation_matrix=[[1.0 if i==j else 0.5 for j in range(n_16)] for i in range(n_16)],
        time_to_maturity=2.0, num_paths=16384, num_time_steps=64, 
        option_style=OptionStyle.BASKET, option_type=OptionType.PUT
    )
    
    # 16D Scalar
    basket_16d_scalar = BasketOptionConfig(**{**basket_16d.__dict__, "exercise_style": ExerciseStyle.SCALAR})
    suite.add(Benchmark("FlashPSO Sobol (16D Scalar)", Method.FLASH_PSO, basket_16d_scalar, sobol_comp, base_swarm, runs=10))
    suite.add(Benchmark("FlashPSO Anti  (16D Scalar)", Method.FLASH_PSO, basket_16d_scalar, anti_comp, base_swarm, runs=10))
    suite.add(Benchmark("FlashPSO Std   (16D Scalar)", Method.FLASH_PSO, basket_16d_scalar, std_comp, base_swarm, runs=10))
    
    # 16D Per-Asset
    basket_16d_per_asset = BasketOptionConfig(**{**basket_16d.__dict__, "exercise_style": ExerciseStyle.PER_ASSET})
    suite.add(Benchmark("FlashPSO Sobol (16D PerAsset)", Method.FLASH_PSO, basket_16d_per_asset, sobol_comp, base_swarm, runs=10))
    suite.add(Benchmark("FlashPSO Anti  (16D PerAsset)", Method.FLASH_PSO, basket_16d_per_asset, anti_comp, base_swarm, runs=10))
    suite.add(Benchmark("FlashPSO Std   (16D PerAsset)", Method.FLASH_PSO, basket_16d_per_asset, std_comp, base_swarm, runs=10))
    
    suite.add(Benchmark("QuantLib LSMC  (16D Baseline)", Method.QUANTLIB, basket_16d, base_compute, base_swarm, runs=1))

    suite.run_all()
    suite.report()

def run_asian_sweep():
    _, base_compute, base_swarm = _get_base_configs()
    suite = BenchmarkSuite(title="Asian (Path-Dependent) Options Sweep")
    
    asian_problem = OptionConfig(
        initial_stock_price=100.0, strike_price=100.0, risk_free_rate=0.05, 
        volatility=0.30, time_to_maturity=2.0, num_paths=65536, 
        num_time_steps=128, option_style=OptionStyle.ASIAN, option_type=OptionType.PUT
    )
    
    sobol_comp = ComputeConfig(**{**base_compute.__dict__, "rng_type": RNGType.SOBOL, "use_antithetic": False})
    anti_comp = ComputeConfig(**{**base_compute.__dict__, "rng_type": RNGType.PHILOX, "use_antithetic": True})
    std_comp = ComputeConfig(**{**base_compute.__dict__, "rng_type": RNGType.PHILOX, "use_antithetic": False})
    
    suite.add(Benchmark("FlashPSO Sobol (Asian)", Method.FLASH_PSO, asian_problem, sobol_comp, base_swarm, runs=20))
    suite.add(Benchmark("FlashPSO Anti  (Asian)", Method.FLASH_PSO, asian_problem, anti_comp, base_swarm, runs=20))
    suite.add(Benchmark("FlashPSO Std   (Asian)", Method.FLASH_PSO, asian_problem, std_comp, base_swarm, runs=20))
    suite.add(Benchmark("QuantLib MC    (Asian)", Method.QUANTLIB, asian_problem, base_compute, base_swarm, runs=1))

    suite.run_all()
    suite.report()

def run_particle_sweep():
    problem, compute, _ = _get_base_configs()
    suite = BenchmarkSuite(title="Particle Density Scaling")
    for p in [32, 64, 128, 256, 512, 1024]:
        swarm = SwarmConfig(num_particles=p)
        suite.add(Benchmark(f"FlashPSO ({p} particles)", Method.FLASH_PSO, problem, compute, swarm, runs=20))
    suite.run_all()
    suite.report()

def run_path_sweep():
    base_problem, compute, swarm = _get_base_configs()
    suite = BenchmarkSuite(title="Monte Carlo Path Scaling")
    for paths in [16384, 65536, 131072, 262144, 524288]:
        problem = OptionConfig(**{**base_problem.__dict__, "num_paths": paths})
        suite.add(Benchmark(f"FlashPSO ({paths} paths)", Method.FLASH_PSO, problem, compute, swarm, runs=20))
    suite.run_all()
    suite.report()

def run_compute_sweep():
    problem, base_compute, swarm = _get_base_configs()
    suite = BenchmarkSuite(title="Compute Fraction & Antithetic Sweep")
    for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
        std_comp = ComputeConfig(**{**base_compute.__dict__, "compute_fraction": frac, "use_antithetic": False})
        suite.add(Benchmark(f"Philox Std (f={frac})", Method.FLASH_PSO, problem, std_comp, swarm, runs=20))
        anti_comp = ComputeConfig(**{**base_compute.__dict__, "compute_fraction": frac, "use_antithetic": True})
        suite.add(Benchmark(f"Philox Anti (f={frac})", Method.FLASH_PSO, problem, anti_comp, swarm, runs=20))
    suite.run_all()
    suite.report()

def run_all_benchmarks():
    print("\n" + "="*60)
    print("STARTING FULL FLASH-PSO BENCHMARK SUITE")
    print("="*60 + "\n")
    
    sweeps = [
        ("Multi-Asset Basket Sweep", run_basket_sweep),
        ("Particle Density Sweep", run_particle_sweep),
        ("Monte Carlo Path Sweep", run_path_sweep),
        ("Compute Fraction Sweep", run_compute_sweep)
    ]
    
    failures = []
    
    for name, sweep_func in sweeps:
        try:
            sweep_func()
        except Exception as e:
            print(f"\n[ERROR] '{name}' encountered a critical failure:")
            traceback.print_exc()
            print("Continuing to the next sweep...\n")
            failures.append((name, str(e)))
            
    print("\n" + "="*60)
    if not failures:
        print(" RUN COMPLETE: All benchmarks executed successfully.")
    else:
        print(f"RUN COMPLETE: Finished with {len(failures)} sweep failures.")
        print("-" * 60)
        for name, err in failures:
            print(f"[FAILED] {name}")
    print("="*60 + "\n")

if __name__ == "__main__":
    run_all_benchmarks()
