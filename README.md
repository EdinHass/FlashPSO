# FlashPSO

> **Work in progress.** This is an active research project — APIs and results may change.

GPU-accelerated American option pricing using Particle Swarm Optimization, implemented in [Triton](https://github.com/openai/triton). Targets Hopper-class hardware (H100/A100).

## Overview

FlashPSO prices American options by running PSO entirely on the GPU. The core kernel fuses three operations into a single dispatch:

1. PSO velocity and position updates
2. On-the-fly GBM Monte Carlo path generation (no HBM reads for path data)
3. Early-exercise payoff evaluation

The on-the-fly path generation is the key design choice — by generating GBM paths in registers rather than reading precomputed paths from global memory, the kernel becomes compute-bound rather than bandwidth-bound. This pays off at large path/timestep counts typical of production workloads.

## Preliminary Results

Benchmarked on an NVIDIA A100-PCIE-40GB MIG 3g.20gb against an OpenCL vec4-fusion baseline. Both implementations use fixed random coefficients and run for the same number of iterations.

**nPeriod=512, nPath=131072 (production-scale workload)**

| nFish | OpenCL Vec Fusion | FlashPSO | Speedup |
|-------|------------------|----------|---------|
| 64    | 4542 ms          | 1016 ms  | **4.5×** |
| 128   | 10777 ms         | 1672 ms  | **6.4×** |
| 256   | 4537 ms          | 2301 ms  | **2.0×** |
| 512   | 4542 ms          | 2576 ms  | **1.8×** |

FlashPSO performs best at production-scale path counts (131k+ paths, 256+ timesteps) where eliminating HBM traffic for path data outweighs the cost of on-the-fly GBM generation. At small workloads the advantage shrinks because the kernel becomes compute-bound. By shifting the performance to be compute-bound rather than bandwidth-bound, FlashPSO allows users to scale up the path/timestep counts without being bottlenecked by memory bandwidth, critical for **fast** and **accurate** American option pricing.

## Requirements

- Python 3.10+
- PyTorch with CUDA
- Triton 3.x
- NVIDIA GPU (Ampere or newer recommended)

```bash
pip install torch triton
```

## Usage

```python
from flash_pso.flash_pso import FlashPSO
from flash_pso.config import OptionConfig, ComputeConfig, SwarmConfig

opt = OptionConfig(
    initial_stock_price=100.0,
    strike_price=110.0,
    risk_free_rate=0.03,
    volatility=0.3,
    time_to_maturity=1.0,
    num_paths=131072,
    num_time_steps=512,
    option_type=1,          # 1=Put, 0=Call
)
comp = ComputeConfig(
    compute_on_the_fly=True,
    use_fixed_random=True,  # False for statistically correct PSO
    max_iterations=100,
    sync_iters=10,
)
swarm = SwarmConfig(num_particles=128)

flash = FlashPSO(option_config=opt, compute_config=comp, swarm_config=swarm)
flash.optimize()
print(f"Option price: {flash.get_option_price():.4f}")
```

## Project Structure

```
flash_pso/
├── config.py          # OptionConfig, ComputeConfig, SwarmConfig
├── flash_pso.py       # Main FlashPSO class
├── pso_kernels.py     # Fused PSO + payoff Triton kernel
├── mc_kernels.py      # Monte Carlo path generation kernels
├── pso_utils.py       # pbest/gbest reduction kernels
└── asserts.py         # Input validation
benchmarks/
└── benchmark.py       # Comparison against OpenCL baseline
```

## Notes

- `use_fixed_random=True` reuses the same r1/r2 coefficients each iteration (matches OpenCL baseline convergence behavior). `False` resamples each iteration, which is statistically correct PSO but converges more slowly.
- `compute_on_the_fly=False` reads precomputed paths from global memory instead — useful if you have a precomputed path tensor or want to profile the bandwidth-bound regime.
- Block sizes must be `1` or a power of two `>= 4` due to the kernel's TMA vs scalar branching.
