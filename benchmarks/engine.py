import time
import os
import csv
import json
from datetime import datetime
import numpy as np
import torch
from typing import Union, List
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn

from flash_pso.config import OptionConfig, BasketOptionConfig, ComputeConfig, SwarmConfig
from flash_pso.enums import OptionStyle, OptionType, ExerciseStyle
from .models import Method, BenchmarkResult
from .wrappers import WRAPPER_REGISTRY, _quantlib_basket_lsmc_price, _quantlib_asian_price, _quantlib_american_price

# Safely import the kernels by matching the SM architecture logic from the main API
try:
    if torch.cuda.is_available():
        _cc_major, _ = torch.cuda.get_device_capability()
        if _cc_major >= 9:
            from flash_pso.sm90.payoff_kernels import mc_payoff_kernel, mc_asian_payoff_kernel, mc_basket_payoff_kernel
        else:
            from flash_pso.sm80.payoff_kernels import mc_payoff_kernel, mc_asian_payoff_kernel, mc_basket_payoff_kernel
    else:
        raise ImportError("CUDA not available")
except ImportError:
    mc_payoff_kernel = None
    mc_asian_payoff_kernel = None
    mc_basket_payoff_kernel = None

class ReferenceCache:
    _cache = {}

    @classmethod
    def get(cls, problem: Union[OptionConfig, BasketOptionConfig]) -> float:
        key = str(problem.__dict__)
        if key in cls._cache:
            return cls._cache[key]
            
        print(f"Computing High-Precision Reference Target...")
        
        if isinstance(problem, BasketOptionConfig):
            opt_str = 'P' if problem.option_type == OptionType.PUT else 'C'
            price = _quantlib_basket_lsmc_price(
                s0s=problem.initial_stock_prices, k=problem.strike_price, r_rate=problem.risk_free_rate,
                vols=problem.volatilities, weights=problem.weights, corr=problem.correlation_matrix,
                t=problem.time_to_maturity, opttype=opt_str, paths=500000, steps=problem.num_time_steps
            )
        else:
            opt_str = 'P' if problem.option_type == OptionType.PUT else 'C'
            if problem.option_style == OptionStyle.ASIAN:
                price = _quantlib_asian_price(
                    s0=problem.initial_stock_price, k=problem.strike_price, r_rate=problem.risk_free_rate,
                    vol=problem.volatility, t=problem.time_to_maturity, opttype=opt_str,
                    paths=200000, steps=problem.num_time_steps
                )
            else:
                price = _quantlib_american_price(
                    s0=problem.initial_stock_price, k=problem.strike_price, r_rate=problem.risk_free_rate,
                    vol=problem.volatility, t=problem.time_to_maturity, opttype=opt_str,
                    engine_type='binomial', steps=20000
                )
            
        cls._cache[key] = price
        return price


class Benchmark:
    def __init__(self, name: str, method: Method, 
                 problem: Union[OptionConfig, BasketOptionConfig], 
                 compute: ComputeConfig, swarm: SwarmConfig, runs: int = 50):
        self.name = name
        self.method = method
        self.problem = problem
        self.compute = compute
        self.swarm = swarm
        self.runs = runs if method not in [Method.QUANTLIB, Method.NATIVE_BINOMIAL] else 1

    def warmup(self):
        if self.method in (Method.FLASH_PSO, Method.FLASH_PSO_CPU):
            wrapper = WRAPPER_REGISTRY[self.method]
            for _ in range(10):
                wrapper(self.problem, self.compute, self.swarm, seed=999)
            if torch.cuda.is_available():
                torch.cuda.synchronize()

    def _export_triton_config(self):
        """Extracts the winning Triton configs and appends them to a JSON log."""
        compute_kernel = None
        
        if isinstance(self.problem, BasketOptionConfig):
            compute_kernel = mc_basket_payoff_kernel
        elif self.problem.option_style == OptionStyle.ASIAN:
            compute_kernel = mc_asian_payoff_kernel
        else:
            compute_kernel = mc_payoff_kernel

        c_cfg = getattr(compute_kernel, 'best_config', None)

        run_data = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "benchmark_name": self.name,
            "method": str(self.method),
            "compute_kernel": {
                "kwargs": c_cfg.kwargs if c_cfg else "No Autotune or Not Found",
                "num_warps": c_cfg.num_warps if c_cfg else None,
                "num_stages": c_cfg.num_stages if c_cfg else None,
            },
        }

        os.makedirs("./benchmarks/results", exist_ok=True)
        try:
            with open("./benchmarks/results/triton_configs.json", "a") as f:
                f.write(json.dumps(run_data) + "\n")
        except Exception as e:
            print(f"Failed to write config for {self.name}: {e}")

    def run(self, progress: Progress, task_id) -> BenchmarkResult:
        prices = np.zeros(self.runs, dtype=np.float32)
        init_times, exec_times, iters_ran, wall_times = [], [], [], []
        
        ref_price = ReferenceCache.get(self.problem)
        wrapper = WRAPPER_REGISTRY.get(self.method)
        is_gpu = (self.method in (Method.FLASH_PSO, Method.FLASH_PSO_CPU) and torch.cuda.is_available())

        self.warmup()  # Ensure warmup before timing

        if self.method in (Method.FLASH_PSO, Method.FLASH_PSO_CPU):
            self._export_triton_config()

        for i in range(self.runs):
            if is_gpu:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            wt0 = time.perf_counter()
            
            price, init_ms, exec_ms, actual_iters = wrapper(
                self.problem, self.compute, self.swarm, seed=1000 + i
            )
            
            if is_gpu: torch.cuda.synchronize()
            
            wt1 = time.perf_counter()
            
            prices[i] = price
            init_times.append(init_ms)
            exec_times.append(exec_ms)
            iters_ran.append(actual_iters)
            wall_times.append(wt1 - wt0)
            
            progress.advance(task_id)

        mean_price = np.mean(prices)
        
        return BenchmarkResult(
            name=self.name,
            method=self.method,
            runs=self.runs,
            mean_price=mean_price,
            bias=mean_price - ref_price,
            std_dev=np.std(prices, ddof=1) if self.runs > 1 else 0.0,
            std_error=(np.std(prices, ddof=1) / np.sqrt(self.runs)) if self.runs > 1 else 0.0,
            rmse=np.sqrt(np.mean((prices - ref_price)**2)),
            mean_iters=np.mean(iters_ran),
            mean_init_time_ms=np.mean(init_times),
            mean_exec_time_ms=np.mean(exec_times),
            mean_iter_time_ms=np.sum(exec_times) / max(1, np.sum(iters_ran)),
            mean_wall_time_s=np.mean(wall_times)
        )


class BenchmarkSuite:
    def __init__(self, title: str, output_dir: str = "./benchmarks/results"):
        self.title = title
        self.output_dir = output_dir
        self.benchmarks: List[Benchmark] = []
        self.results: List[BenchmarkResult] = []

    def add(self, benchmark: Benchmark):
        self.benchmarks.append(benchmark)

    def run_all(self):
        console = Console()
        console.print(f"\n[bold cyan]Starting Suite: {self.title}[/bold cyan]")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            overall_task = progress.add_task("[bold green]Total Progress", total=len(self.benchmarks))
            for bench in self.benchmarks:
                task = progress.add_task(f"[cyan]Running {bench.name}...", total=bench.runs)
                result = bench.run(progress, task)
                self.results.append(result)
                progress.update(task, visible=False)
                progress.advance(overall_task)

    def save_csv(self) -> str:
        os.makedirs(self.output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_title = self.title.replace(" ", "_").replace("/", "-").lower()
        filepath = os.path.join(self.output_dir, f"{safe_title}_{timestamp}.csv")
        
        with open(filepath, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                "Method", "Mean Price", "Bias", "Std Dev", "Std Err", 
                "RMSE", "Mean Iters", "Init Time (ms)", "Iter Time (ms)", "Wall Time (s)"
            ])
            for r in self.results:
                iters_str = f"{r.mean_iters:.1f}" if r.method not in [Method.QUANTLIB, Method.NATIVE_BINOMIAL, Method.OPENCL_LSMC] else "N/A"
                iter_time_str = f"{r.mean_iter_time_ms:.4f}"
                writer.writerow([
                    r.name, f"{r.mean_price:.6f}", f"{r.bias:+.6f}", f"{r.std_dev:.6f}",
                    f"{r.std_error:.6f}", f"{r.rmse:.6f}", iters_str,
                    f"{r.mean_init_time_ms:.2f}", iter_time_str, f"{r.mean_wall_time_s:.4f}"
                ])
        return filepath

    def report(self):
        console = Console()
        table = Table(title=f"Benchmark Results: {self.title}", show_header=True, header_style="bold magenta")
        
        table.add_column("Method", style="cyan", width=28)
        table.add_column("Mean Price", justify="right")
        table.add_column("Bias", justify="right")
        table.add_column("Std Dev", justify="right")
        table.add_column("RMSE", justify="right")
        table.add_column("Iters", justify="right", style="blue")
        table.add_column("Init (ms)", justify="right", style="yellow")
        table.add_column("Iter (ms)", justify="right", style="yellow")
        table.add_column("Wall (s)", justify="right", style="green")

        for r in self.results:
            iters_str = f"{r.mean_iters:.1f}" if r.method not in [Method.QUANTLIB, Method.NATIVE_BINOMIAL, Method.OPENCL_LSMC] else "-"
            table.add_row(
                r.name, f"{r.mean_price:.6f}", f"{r.bias:+.6f}", f"{r.std_dev:.6f}",
                f"{r.rmse:.6f}", iters_str, f"{r.mean_init_time_ms:.2f}",
                f"{r.mean_iter_time_ms:.4f}", f"{r.mean_wall_time_s:.4f}"
            )
            
        console.print(table)
        csv_path = self.save_csv()
        console.print(f"\n[bold green]✔[/bold green] Results saved to: [dim]{csv_path}[/dim]")
