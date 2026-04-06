import triton.testing
from .utils import log, run_bench, S0, r, sigma, T, K, OPT_TYPE
from flash_pso import FlashPSO
from flash_pso.config import OptionConfig, ComputeConfig, SwarmConfig
from references.mc import hybridMonteCarlo
from references.pso import PSO_OpenCL_vec_fusion
from references.longstaff import LSMC_OpenCL

@triton.testing.perf_report([
    triton.testing.Benchmark(
        x_names=['nPath'], 
        x_vals=[2**i for i in range(12, 21)],
        line_arg='provider',
        line_vals=['cl_pso', 'lsmc_cl', 'triton_hybrid_25', 'triton_compute_100'],
        line_names=['OpenCL Vec Fusion', 'LSMC OpenCL', 'FlashPSO (25%)', 'FlashPSO (100%)'],
        styles=[('black', ':'), ('gray', '--'), ('cyan', '-'), ('blue', '-')],
        ylabel='ms / Iteration', 
        plot_name='01_baseline_comparison',
        args={'nFish': 64, 'nPeriod': 256, 'iterMax': 100}
    )
])
def bench_baseline(nPath, nFish, nPeriod, provider, iterMax):
    if provider == 'cl_pso':
        def run_cl_pso():
            mc = hybridMonteCarlo(S0, r, sigma, T, nPath, nPeriod, K, OPT_TYPE, nFish)
            PSO_OpenCL_vec_fusion(mc, nFish, iterMax=iterMax).solvePsoAmerOption_cl()
            mc.cleanUp()
        med, mn, mx = run_bench(run_cl_pso)
        return med / iterMax, mn / iterMax, mx / iterMax
        
    elif provider == 'lsmc_cl':
        def run_cl_lsmc():
            mc = hybridMonteCarlo(S0, r, sigma, T, nPath, nPeriod, K, OPT_TYPE, nFish)
            LSMC_OpenCL(mc, 'GJ').longstaff_schwartz_itm_path_fast_hybrid()
            mc.cleanUp()
        med, mn, mx = run_bench(run_cl_lsmc)
        return med, mn, mx
        
    else:
        frac = 0.25 if '25' in provider else 1.0
        opt = OptionConfig(initial_stock_price=S0, strike_price=K, risk_free_rate=r, volatility=sigma, time_to_maturity=T, num_paths=nPath, num_time_steps=nPeriod)
        comp = ComputeConfig(compute_fraction=frac, max_iterations=iterMax, sync_iters=5, seed=1337)
        swarm = SwarmConfig(num_particles=nFish)

        flash = FlashPSO(option_config=opt, compute_config=comp, swarm_config=swarm)
        flash.optimize()
        med, mn, mx = run_bench(lambda: flash.optimize())
        actual_iters = flash.global_payoff_index * flash.comp.sync_iters
        
        return med / max(1, actual_iters), mn / max(1, actual_iters), mx / max(1, actual_iters)

@triton.testing.perf_report([
    triton.testing.Benchmark(
        x_names=['nPath'], 
        x_vals=[2**i for i in range(16, 17)],
        line_arg='frac', 
        line_vals=[0.0, 0.25, 0.5, 0.75, 1.0],
        line_names=['Bandwidth (0%)', 'Hybrid (25%)', 'Hybrid (50%)', 'Hybrid (75%)', 'Compute (100%)'],
        styles=[('green', '--'), ('cyan', '-'), ('orange', '-'), ('purple', '-'), ('blue', '-')],
        ylabel='ms / Iteration', 
        plot_name='02_compute_fraction_sweep',
        args={'nFish': 64, 'nPeriod': 64, 'iterMax': 100}
    )
])
def bench_fractions(nPath, nFish, nPeriod, frac, iterMax):
    opt = OptionConfig(initial_stock_price=S0, strike_price=K, risk_free_rate=r, volatility=sigma, time_to_maturity=T, num_paths=nPath, num_time_steps=nPeriod)
    comp = ComputeConfig(compute_fraction=frac, max_iterations=iterMax, sync_iters=5, seed=1337)
    swarm = SwarmConfig(num_particles=nFish)

    flash = FlashPSO(option_config=opt, compute_config=comp, swarm_config=swarm)
    flash.optimize()
    med, mn, mx = run_bench(lambda: flash.optimize())
    actual_iters = flash.global_payoff_index * flash.comp.sync_iters
    return med / max(1, actual_iters), mn / max(1, actual_iters), mx / max(1, actual_iters)

@triton.testing.perf_report([
    triton.testing.Benchmark(
        x_names=['nFish'], 
        x_vals=[32, 64, 128, 256, 512, 1024],
        line_arg='frac', 
        line_vals=[0.0, 1.0],
        line_names=['Bandwidth-Bound (0%)', 'Compute-Bound (100%)'],
        ylabel='ms / Iteration', 
        plot_name='03_particle_density_scaling',
        args={'nPath': 2**16, 'nPeriod': 128, 'iterMax': 100}
    )
])
def bench_density(nPath, nFish, nPeriod, frac, iterMax):
    opt = OptionConfig(initial_stock_price=S0, strike_price=K, risk_free_rate=r, volatility=sigma, time_to_maturity=T, num_paths=nPath, num_time_steps=nPeriod)
    comp = ComputeConfig(compute_fraction=frac, max_iterations=iterMax, sync_iters=5, seed=1337)
    swarm = SwarmConfig(num_particles=nFish)

    flash = FlashPSO(option_config=opt, compute_config=comp, swarm_config=swarm)
    flash.optimize()
    med, mn, mx = run_bench(lambda: flash.optimize())
    actual_iters = flash.global_payoff_index * flash.comp.sync_iters
    return med / max(1, actual_iters), mn / max(1, actual_iters), mx / max(1, actual_iters)
