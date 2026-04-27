# FlashPSO

**GPU-Accelerated Particle Swarm Optimization for Monte Carlo American Option Pricing.** Implemented in [Triton](https://github.com/openai/triton); targets NVIDIA Ampere (A100) and Hopper (H100).

Princeton CS Independent Work · Spring 2026 · Edin Hasanovic, advised by Tri Dao.

## Overview

FlashPSO prices American options by searching for the optimal early-exercise boundary directly on the GPU. The exercise boundary lives in a $T$-dimensional space (one value per timestep); a swarm of $P$ particles iteratively updates candidate boundaries, scoring each by the average discounted payoff over $N$ Monte Carlo paths.

Prior GPU-PSO implementations parallelise only across the particle dimension. Because $P$ is typically small relative to modern GPU parallelism, this leaves most SMs idle while a handful of threads grind through deep sequential loops over the $N \times T$ path space. FlashPSO resolves this by **tiling across both particles and paths simultaneously**: the payoff kernel processes a block of particles against a block of paths, partial payoffs are reduced on-device, and a separate two-stage reduction extracts pbest/gbest. Initialisation, path generation, and reduction all run on the GPU — only the global best fitness scalar is copied back to the host (and only every `sync_iters` iterations, for convergence checking).

A heuristic flat-boundary initialisation (bounded by the perpetual-American closed form) and native variance reduction (antithetic variates, Sobol sequences with Brownian-bridge construction) further reduce the path count required to hit a target accuracy.

## Performance

Benchmarked against the Li & Chen OpenCL PSO baseline and Longstaff-Schwartz Monte Carlo (LSMC) on an NVIDIA A100 (80GB).

**Setup:** American Put, S₀ = K = 100, r = 5%, σ = 20%, T = 2yr · 131,072 paths · 128 timesteps · 128 particles · 150 iterations. Binomial-tree ground truth: **7.7232**.

| Method | Mean Price | Std Dev | RMSE | Wall Time | Speedup |
|--------|-----------|---------|------|-----------|---------|
| Li & Chen PSO        | 7.6755 | 0.0147 | 0.0494 | 36.21 s | 1.0× |
| LSMC Baseline        | 7.6776 | 0.0117 | 0.0467 | 2.58 s  | 14.0× |
| **FlashPSO Sobol**   | 7.7229 | **0.0065** | **0.0064** | 0.24 s | **150.8×** |
| **FlashPSO Anti**    | 7.7196 | 0.0120 | 0.0123 | 0.19 s | **190.5×** |
| **FlashPSO Std**     | 7.7136 | 0.0176 | 0.0196 | **0.19 s** | **190.5×** |

FlashPSO Standard achieves a **190× wall-time speedup** over the Li & Chen GPU-PSO baseline with **2.5× lower RMSE**. Sobol drives RMSE down a further 3× (to 0.0064) with no meaningful throughput penalty, beating both the LSMC baseline and the prior PSO implementation by orders of magnitude on accuracy.

**Time-to-accuracy.** Compared at matched RMSE, the gap widens further: the baseline needs ~48 s and 200 iterations to reach RMSE ≈ 0.049, while FlashPSO Standard reaches comparable accuracy in 25 iterations and ~0.035 s — roughly a **1360× speedup to target precision**.

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
    volatility=0.20,
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
| `use_fp16_paths` | False | Store precomputed paths in FP16 (basket / `compute_fraction < 1`). Halves HBM traffic but adds cast cost; only wins when the payoff kernel is HBM-bound. |
| `use_fp16_cholesky` | False | Store the basket Cholesky factor in FP16. |
| `pso_paths_block_size` | 64 | Path-block tile size (power of two, ≥ 4). |
| `randomize_paths` | False | Shuffle path ordering each iteration. |

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
- All block sizes must be powers of two. Path count must be divisible by `pso_paths_block_size` (default 64).

## Known Issues

- **Last-timestep boundary is unreliable.** The terminal exercise decision is handled as a special case (exercise iff in-the-money), so the optimiser receives no gradient signal on the final boundary value and it can drift to arbitrary values. This does **not** affect price correctness — the special-case rule is what gets evaluated — but the gbest vector at index `T-1` should not be interpreted as a learned boundary.
- **Dead-dimension spikes in the gbest vector.** Boundary values at timesteps where no path is near exercise receive no payoff signal and can show large spikes / non-monotonicity. This is a property of the PSO objective surface (zero gradient on insensitive dims), not a bug; price correctness is unaffected because those dims don't change any exercise decisions.

## TODO / Roadmap

- **Persistent kernels.** Replace per-iteration kernel launches with a single persistent grid that loops internally — eliminates launch overhead and lets pbest/gbest stay in shared memory across iterations.
- **Curved boundary initialisation.** Initial particles currently start from a flat boundary; seeding with a roughly-correct curved shape (e.g. a parametric American-put boundary approximation) should reduce iteration count to convergence.
- **Reduce overfitting.** In-sample bias on the optimised boundary remains material at small path counts. Worth exploring: out-of-sample validation during PSO, regularisation on boundary roughness, ensembled re-evaluation on independent path sets, or boundary smoothing/projection (monotone, in `[0, K]`) to prune dead dimensions.
- **CUTLASS port.** Triton hits a ceiling on register pressure and async-pipeline control for the fused payoff kernel; a CUTLASS implementation would expose explicit warp-specialisation, TMA, and `wgmma` for the basket Cholesky path.
- **FP16 path quantisation viability.** Initial benchmarks show FP16 paths and FP16-on-the-fly compute are slower than FP32 on A100 across all tested path counts (the kernel isn't HBM-bound thanks to L2 reuse across particles). Worth re-checking on H100 / B200 where higher compute throughput shifts the bottleneck back to bandwidth.
- **Optimise basket options.** Basket kernels are still well below peak: covariance math dominates, autotune space is more constrained (`pt * d ≤ 8`), and Cholesky-factor reuse across particles isn't fully exploited.
- **Other exotic options.** Lookback, barrier (knock-in/knock-out), and Bermudan variants would round out the exotic suite. The PSO formulation is general — main work is per-style payoff kernels and any required path-state augmentation (running min/max).
