import os
from references.utils import checkOpenCL
from .utils import log
from .core_perf import bench_baseline, bench_fractions, bench_density
from .comparisons import exec_single_test, exec_accuracy_test, exec_library_latency_test, exec_lsmc_latency_test
from .exotics import exec_asian_latency_test, exec_basket_benchmark
from .accuracy import exec_variance_benchmark
from .accuracy_basket import exec_basket_variance_benchmark

def _setup_env():
    checkOpenCL()
    save_dir = os.path.join('benchmarks', 'results')
    os.makedirs(save_dir, exist_ok=True)
    return save_dir

def run_single(): _setup_env(); exec_single_test()
def run_baseline(): bench_baseline.run(print_data=True, show_plots=False, save_path=_setup_env())
def run_fraction_sweep(): bench_fractions.run(print_data=True, show_plots=False, save_path=_setup_env())
def run_particle_scaling(): bench_density.run(print_data=True, show_plots=False, save_path=_setup_env())
def run_accuracy(): _setup_env(); exec_accuracy_test()
def run_variance(): _setup_env(); exec_variance_benchmark()
def run_basket_variance(): _setup_env(); exec_basket_variance_benchmark()

def run_library_latency(): _setup_env(); exec_library_latency_test()
def run_lsmc_latency(): _setup_env(); exec_lsmc_latency_test()
def run_asian_latency(): _setup_env(); exec_asian_latency_test()
def run_basket(): _setup_env(); exec_basket_benchmark()

def run_all():
    save_dir = _setup_env()
    log("=" * 65); log("STARTING FULL FLASH-PSO SUITE"); log("=" * 65)
    
    exec_single_test()
    bench_baseline.run(print_data=True, show_plots=False, save_path=save_dir)
    bench_fractions.run(print_data=True, show_plots=False, save_path=save_dir)
    bench_density.run(print_data=True, show_plots=False, save_path=save_dir)
    
    exec_accuracy_test()
    exec_variance_benchmark() 
    exec_library_latency_test()
    exec_lsmc_latency_test()
    exec_asian_latency_test()
    exec_basket_benchmark()    
    
    # ── ADDED TO FULL SUITE ──
    exec_basket_variance_benchmark()
    
    log("All benchmarks completed successfully.")

if __name__ == "__main__":
    run_all()
