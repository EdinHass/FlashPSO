import time
from .utils import *
from flash_pso import FlashPSO
from flash_pso.config import OptionConfig, ComputeConfig, SwarmConfig

from references.mc import hybridMonteCarlo
from references.longstaff import LSMC_OpenCL, LSMC_Numpy

def exec_single_test():
    log("-" * 65); log("ONE-OFF DIAGNOSTIC RUN"); log("-" * 65)
    
    opt = OptionConfig(initial_stock_price=S0, strike_price=K, risk_free_rate=r, volatility=sigma, time_to_maturity=T, num_paths=65536, num_time_steps=256)
    comp = ComputeConfig(compute_fraction=0.25, max_iterations=100, sync_iters=5, seed=1337)
    swarm = SwarmConfig(num_particles=128)
    
    flash = FlashPSO(option_config=opt, compute_config=comp, swarm_config=swarm)
    flash.optimize()
    iters = flash.global_payoff_index * flash.comp.sync_iters
    
    total_ms, _, _ = run_bench(lambda: flash.optimize())
    price, ref = flash.get_option_price(), get_reference_price()
    
    print(f"\nTotal Time: {total_ms:.2f} ms | Iterations: {iters}")
    print(f"Price: {price:.6f} | Ref: {ref:.6f} | Rel Err: {(abs(price-ref)/ref)*100:.4f}%\n")
    log("-" * 65)

def exec_accuracy_test():
    log("-" * 80); log("ACCURACY AND CONVERGENCE TEST"); log("-" * 80)
    ref = get_reference_price()

    opt = OptionConfig(initial_stock_price=S0, strike_price=K, risk_free_rate=r, volatility=sigma, time_to_maturity=T, num_paths=1024, num_time_steps=512)
    comp = ComputeConfig(compute_fraction=1.0, max_iterations=200, sync_iters=5, seed=1337)
    swarm = SwarmConfig(num_particles=128)

    print(f"\n{'Paths':<12} | {'Iters':<6} | {'PSO Price':<12} | {'Relative Error':<12}")
    print("-" * 50)
    
    for p in [2**i for i in range(12, 21)]:
        opt.num_paths = p
        flash = FlashPSO(option_config=opt, compute_config=comp, swarm_config=swarm)
        flash.optimize()
        price = flash.get_option_price()
        iters = flash.global_payoff_index * flash.comp.sync_iters
        print(f"{p:<12} | {iters:<6} | {price:12.6f} | {(abs(price-ref)/ref)*100:11.4f}%")

def exec_library_latency_test():
    log("-" * 80); log("LIBRARY LATENCY BENCHMARK (triton do_bench)"); log("-" * 80)
    print(f"{'Library':<25} | {'Method':<25} | {'Median (ms)':<12} | {'Price':<12}")
    print("-" * 80)

    if HAS_QUANTLIB:
        for steps in [200, 2000]:
            med, _, _ = run_bench(lambda: _quantlib_american_price('binomial', steps))
            print(f"{'QuantLib':<25} | {f'CRR ({steps} steps)':<25} | {med:12.4f} | {_quantlib_american_price('binomial', steps):12.6f}")

    if HAS_OPTLIB:
        med, _, _ = run_bench(_optlib_american_price)
        print(f"{'optlib':<25} | {'BS2002 Approx':<25} | {med:12.4f} | {_optlib_american_price():12.6f}")

    if HAS_FASTVOL:
        for steps in [200, 2000]:
            med, _, _ = run_bench(lambda: _fastvol_american_price(steps))
            print(f"{'fastvol':<25} | {f'BOPM ({steps} steps)':<25} | {med:12.4f} | {_fastvol_american_price(steps):12.6f}")

    opt = OptionConfig(initial_stock_price=S0, strike_price=K, risk_free_rate=r, volatility=sigma, time_to_maturity=T, num_paths=131072, num_time_steps=128)
    comp = ComputeConfig(compute_fraction=1.0, max_iterations=100, sync_iters=5, seed=1337)
    swarm = SwarmConfig(num_particles=128)
    
    flash = FlashPSO(option_config=opt, compute_config=comp, swarm_config=swarm)
    flash.optimize()
    med, _, _ = run_bench(lambda: flash.optimize())
    print(f"{'FlashPSO':<25} | {f'PSO (131k paths)':<25} | {med:12.4f} | {flash.get_option_price():12.6f}")

def exec_lsmc_latency_test():
    log("-" * 80); log("LONGSTAFF-SCHWARTZ vs FLASH-PSO LATENCY BENCHMARK"); log("-" * 80)
    print(f"{'Implementation':<25} | {'Hardware / Specs':<25} | {'Median (ms)':<12} | {'Price':<12}")
    print("-" * 80)

    paths_test, steps_test = 65536, 128
    
    try:
        def run_np():
            mc = hybridMonteCarlo(S0, r, sigma, T, paths_test, steps_test, K, OPT_TYPE, 128)
            LSMC_Numpy(mc, inverseType='benchmark_lstsq', log='OFF').longstaff_schwartz_itm_path_fast()
            mc.cleanUp()
        med, _, _ = run_bench(run_np)
        mc_tmp = hybridMonteCarlo(S0, r, sigma, T, paths_test, steps_test, K, OPT_TYPE, 128)
        print(f"{'LSMC_Numpy':<25} | {f'CPU ({paths_test//1000}k)':<25} | {med:12.4f} | {LSMC_Numpy(mc_tmp, inverseType='benchmark_lstsq').longstaff_schwartz_itm_path_fast()[0]:12.6f}")
        mc_tmp.cleanUp()
    except Exception as e: log(f"[WARN] Numpy setup failed: {e}")

    try:
        def run_cl():
            mc = hybridMonteCarlo(S0, r, sigma, T, paths_test, steps_test, K, OPT_TYPE, 128)
            LSMC_OpenCL(mc, inverseType='GJ', log='OFF').longstaff_schwartz_itm_path_fast_hybrid()
            mc.cleanUp()
        med, _, _ = run_bench(run_cl)
        mc_tmp = hybridMonteCarlo(S0, r, sigma, T, paths_test, steps_test, K, OPT_TYPE, 128)
        print(f"{'LSMC_OpenCL':<25} | {f'GPU CL ({paths_test//1000}k)':<25} | {med:12.4f} | {LSMC_OpenCL(mc_tmp, inverseType='GJ').longstaff_schwartz_itm_path_fast_hybrid()[0]:12.6f}")
        mc_tmp.cleanUp()
    except Exception as e: log(f"[WARN] OpenCL setup failed: {e}")

    opt = OptionConfig(initial_stock_price=S0, strike_price=K, risk_free_rate=r, volatility=sigma, time_to_maturity=T, num_paths=paths_test, num_time_steps=steps_test)
    comp = ComputeConfig(compute_fraction=1.0, max_iterations=100, sync_iters=5, seed=1337)
    swarm = SwarmConfig(num_particles=128)
    
    flash = FlashPSO(option_config=opt, compute_config=comp, swarm_config=swarm)
    flash.optimize()
    med, _, _ = run_bench(lambda: flash.optimize())
    iters = flash.global_payoff_index * flash.comp.sync_iters
    print(f"{'FlashPSO':<25} | {f'GPU Triton ({iters} iter)':<25} | {med:12.4f} | {flash.get_option_price():12.6f}")
