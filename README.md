# FlashPSO

GPU-accelerated American option pricing via Particle Swarm Optimization, implemented in [Triton](https://github.com/openai/triton). Targets NVIDIA Ampere (A100) and Hopper (H100) hardware.

## Overview

FlashPSO prices American options by optimising an exercise boundary entirely on the GPU. The key design decision is **on-the-fly path generation**: rather than writing Monte Carlo paths to HBM and reading them back each iteration, the payoff kernel regenerates GBM paths in registers using a seeded counter-based RNG (Philox). This shifts the kernel from bandwidth-bound to compute-bound, which scales well at the large path and timestep counts required for accurate American option pricing.

Each PSO iteration fuses three operations into a single kernel dispatch:

1. On-the-fly GBM Monte Carlo path generation (Philox or antithetic Philox)
2. Early-exercise payoff evaluation against the current boundary candidates
3. Partial payoff accumulation across path blocks

PSO velocity/position updates and pbest/gbest reductions run in separate lightweight kernels, all on-GPU with no CPU synchronisation during the optimisation loop.

## Performance

Benchmarked against an OpenCL vec4-fused PSO baseline and Longstaff-Schwartz Monte Carlo (LSMC) on an NVIDIA A100.

**Setup:** American Put, S₀ = K = 100, r = 5%, σ = 30%, T = 2yr · 131,072 paths · 128 timesteps · 128 particles · 150 iterations

| Method | Mean Price | RMSE | Iter Time | Wall Time | vs OpenCL PSO |
|--------|-----------|------|-----------|-----------|---------------|
| FlashPSO (Sobol QMC) | 12.8429 | **0.0121** | 1.24 ms | 0.26 s | **184× faster** |
| FlashPSO (Antithetic) | 12.8475 | **0.0200** | 1.19 ms | 0.18 s | **192× faster** |
| FlashPSO (Standard) | 12.8375 | 0.0316 | 1.20 ms | 0.18 s | **190× faster** |
| OpenCL PSO (Std) | 12.8184 | 0.0394 | 228 ms | 36.5 s | — |
| OpenCL LSMC | 12.7900 | 0.0599 | 884 ms | 2.7 s | — |

FlashPSO (Standard) is **190× faster** than the OpenCL PSO baseline with **20% lower RMSE**. Enabling antithetic sampling or Sobol quasi-random sequences improves accuracy further with no meaningful throughput penalty — Sobol achieves **5× lower RMSE** than LSMC at a fraction of the runtime.

Sobol init time is higher (~70 ms) due to host-side sequence generation; subsequent runs reuse cached paths.

## Installation

```bash
pip install torch triton
pip install -e .
```

Python 3.12+, PyTorch 2.0+, Triton 3.6+, CUDA-capable GPU (Ampere or newer).

## Quick Start

```python
from flash_pso import FlashPSO, OptionConfig, ComputeConfig, SwarmConfig
from flash_pso.enums import OptionType, RNGType

opt = OptionConfig(
    initial_stock_price=100.0,
    strike_price=100.0,
    risk_free_rate=0.05,
    volatility=0.30,
    time_to_maturity=2.0,
    num_paths=131072,
    num_time_steps=128,
    option_type=OptionType.PUT,
)
comp = ComputeConfig(
    seed=42,
    max_iterations=150,
    sync_iters=5,
)
swarm = SwarmConfig(num_particles=128)

pricer = FlashPSO(opt, comp, swarm)
pricer.optimize()
print(f"In-sample price:  {pricer.get_option_price():.4f}")
print(f"Debiased price:   {pricer.get_debiased_price():.4f}")
```

`get_debiased_price()` re-evaluates the optimised boundary on a fresh independent set of paths, removing the in-sample upward bias.

### Variance reduction

```python
# Antithetic sampling — roughly 2× lower variance, same throughput
comp_anti = ComputeConfig(seed=42, max_iterations=150, sync_iters=5, use_antithetic=True)

# Sobol quasi-random sequences — lowest variance, slight init overhead
comp_sobol = ComputeConfig(seed=42, max_iterations=150, sync_iters=5, rng_type=RNGType.SOBOL)
```

## Configuration

### `OptionConfig`

| Parameter | Type | Description |
|-----------|------|-------------|
| `initial_stock_price` | float | S₀ |
| `strike_price` | float | K |
| `risk_free_rate` | float | Annualised risk-free rate |
| `volatility` | float | Annualised volatility σ |
| `time_to_maturity` | float | T in years |
| `num_paths` | int | Monte Carlo paths (power of two, ≥ 1024) |
| `num_time_steps` | int | Discretisation steps (power of two) |
| `option_type` | `OptionType` | `PUT` or `CALL` |

### `ComputeConfig`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `seed` | — | RNG seed (required) |
| `max_iterations` | 1000 | PSO iteration budget |
| `sync_iters` | 10 | Iterations between convergence checks |
| `convergence_threshold` | 1e-6 | Early-stop threshold (set < 0 to disable) |
| `compute_fraction` | 1.0 | Fraction of paths generated on-the-fly (0 = all precomputed, 1 = all on-the-fly) |
| `use_antithetic` | False | Antithetic path pairs for variance reduction |
| `rng_type` | `PHILOX` | `PHILOX` or `SOBOL` |
| `use_fixed_random` | False | Reuse r1/r2 PSO coefficients across iterations |

### `SwarmConfig`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_particles` | — | Number of PSO particles (power of two) |
| `inertia_weight` | 0.7298 | ω |
| `cognitive_weight` | 1.49618 | c₁ |
| `social_weight` | 1.49618 | c₂ |

## Experimental Features

The following option types are implemented and functional but have received less validation than the vanilla American pricer. APIs may change.

### American Asian Options

Average-price American options where the exercise payoff is based on the arithmetic mean of the asset price over the path.

```python
from flash_pso import FlashPSO, OptionConfig, ComputeConfig, SwarmConfig
from flash_pso.enums import OptionType, OptionStyle

opt = OptionConfig(
    initial_stock_price=100.0, strike_price=100.0,
    risk_free_rate=0.05, volatility=0.30, time_to_maturity=1.0,
    num_paths=131072, num_time_steps=128,
    option_type=OptionType.PUT,
    option_style=OptionStyle.ASIAN,
)
comp = ComputeConfig(seed=42, max_iterations=200, sync_iters=5)
swarm = SwarmConfig(num_particles=128)

pricer = FlashPSO(opt, comp, swarm)
pricer.optimize()
print(pricer.get_debiased_price())
```

### American Basket Options

Multi-asset basket options with correlated GBM paths (Cholesky decomposition). Supports two exercise parameterisations via `ExerciseStyle`:

- `SCALAR` — one boundary per timestep on the weighted basket price (PSO dimension = T)
- `PER_ASSET` — independent boundary per asset per timestep, exercise when all assets cross (PSO dimension = N × T)

```python
from flash_pso import FlashPSOBasket, BasketOptionConfig, ComputeConfig, SwarmConfig
from flash_pso.enums import OptionType, ExerciseStyle

opt = BasketOptionConfig(
    initial_stock_prices=[100.0, 100.0, 100.0],
    strike_price=100.0,
    risk_free_rate=0.05,
    volatilities=[0.20, 0.25, 0.30],
    weights=[1/3, 1/3, 1/3],
    correlation_matrix=[
        [1.0, 0.5, 0.3],
        [0.5, 1.0, 0.4],
        [0.3, 0.4, 1.0],
    ],
    time_to_maturity=1.0,
    num_paths=131072,
    num_time_steps=64,
    option_type=OptionType.PUT,
    exercise_style=ExerciseStyle.SCALAR,
)
comp = ComputeConfig(seed=42, max_iterations=200, sync_iters=5)
swarm = SwarmConfig(num_particles=128)

pricer = FlashPSOBasket(opt, comp, swarm)
pricer.optimize()
print(pricer.get_debiased_price())
```

## Project Structure

```
flash_pso/
├── api.py              # FlashPSO — vanilla and Asian American option pricer
├── api_basket.py       # FlashPSOBasket — multi-asset basket option pricer
├── config.py           # OptionConfig, BasketOptionConfig, ComputeConfig, SwarmConfig
├── enums.py            # OptionType, OptionStyle, ExerciseStyle, RNGType
├── asserts.py          # Input validation
├── rng/
│   └── sobol.py        # Sobol sequence generation with Brownian Bridge construction
├── sm80/               # Ampere-class kernels (A100, RTX 30/40)
│   ├── payoff_kernels.py
│   ├── pso_kernels.py
│   ├── mc_kernels.py
│   └── pso_utils.py
└── sm90/               # Hopper-class kernels (H100) — TMA-optimised
    ├── payoff_kernels.py
    ├── pso_kernels.py
    ├── mc_kernels.py
    └── pso_utils.py
benchmarks/
├── main.py             # Benchmark suites
├── engine.py           # Benchmark harness
├── wrappers.py         # Wrappers for FlashPSO, OpenCL PSO, LSMC, QuantLib
└── results/            # CSV benchmark output
references/
├── pso.py              # NumPy PSO baseline
├── mc.py               # CPU/OpenCL Monte Carlo
├── longstaff.py        # Longstaff-Schwartz LSMC
└── kernels/            # OpenCL reference kernels
```

## Running Benchmarks

```bash
# Core vanilla performance vs baselines
flash-pso-core

# Full benchmark suite
flash-pso-all
```

Benchmark dependencies (OpenCL, QuantLib) are optional:

```bash
pip install -e ".[bench]"
```

## Notes

- `compute_fraction=1.0` (default) generates all paths on-the-fly — compute-bound and optimal at large path/step counts. `compute_fraction=0.0` precomputes all paths to HBM, which is useful with Sobol sequences (which require precomputation) or for profiling the bandwidth-bound regime.
- `use_fixed_random=True` reuses the same r1/r2 PSO coefficients each iteration, matching the convergence behaviour of the OpenCL baseline. The default (`False`) resamples each iteration, which is statistically correct but may converge more slowly.
- Hardware dispatch is automatic: SM90 (TMA-optimised) kernels are used on Hopper-class GPUs; SM80 kernels are used on Ampere.
- All block sizes must be powers of two. Path count must be divisible by `pso_paths_block_size` (default 256).
