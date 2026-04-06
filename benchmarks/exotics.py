from .utils import *
from flash_pso import FlashPSO
from flash_pso.config import OptionConfig, BasketOptionConfig, ComputeConfig, SwarmConfig

try:
    from flash_pso.api_basket import FlashPSOBasket
    HAS_BASKET_API = True
except ImportError:
    HAS_BASKET_API = False


def exec_asian_latency_test():
    log("-" * 80); log("ASIAN OPTION LATENCY BENCHMARK")
    print(" - QL MC  : Forced to European Discrete (American not supported).")
    print(" - QL FD  : Forced to European Continuous (American Discrete not supported).")
    print(" - FlashPSO: Pricing American Discrete natively.")
    print("-" * 80)
    print(f"{'Library':<25} | {'Method':<25} | {'Median (ms)':<12} | {'Price':<12}")
    print("-" * 80)

    if HAS_QUANTLIB:
        for paths in [10000, 50000]:
            med, _, _ = run_bench(lambda: _quantlib_asian_price(paths, 128))
            print(f"{'QuantLib':<25} | {f'CPU MC Eur ({paths//1000}k)':<25} | {med:12.4f} | {_quantlib_asian_price(paths, 128):12.6f}")
        for t, x, a in [(50, 50, 25), (100, 100, 50)]:
            med, _, _ = run_bench(lambda: _quantlib_asian_fd_price(t, x, a))
            print(f"{'QuantLib':<25} | {f'CPU FD Eur ({t}x{x}x{a})':<25} | {med:12.4f} | {_quantlib_asian_fd_price(t, x, a):12.6f}")

    opt = OptionConfig(
        initial_stock_price=S0, strike_price=K, risk_free_rate=r,
        volatility=sigma, time_to_maturity=T, num_paths=131072,
        num_time_steps=128, option_type=1, option_style="asian",
    )
    comp = ComputeConfig(compute_fraction=1.0, max_iterations=100, sync_iters=5, seed=1337)
    swarm = SwarmConfig(num_particles=128)

    def bench_asian():
        f = FlashPSO(option_config=opt, compute_config=comp, swarm_config=swarm)
        f.optimize()
        return f

    flash = bench_asian()
    med, _, _ = run_bench(bench_asian)
    print(f"{'FlashPSO':<25} | {f'GPU Amer PSO (131k)':<25} | {med:12.4f} | {flash.get_option_price():12.6f}")


def _check_particle_diversity(flash, label=""):
    """Print statistics about particle positions to verify init is working."""
    import torch
    pos = flash.positions.cpu()  # shape: (num_particles, pso_dim)
    n_particles = pos.shape[0]
    n_dims = pos.shape[1]

    # Check first 4 dims across all particles
    show_dims = min(4, n_dims)
    print(f"\n  [{label}] Particle diversity check ({n_particles} particles, {n_dims} dims):")
    print(f"  {'Dim':<6} | {'Min':>10} | {'Max':>10} | {'Mean':>10} | {'Std':>10}")
    for d in range(show_dims):
        col = pos[:, d]
        print(f"  {d:<6} | {col.min().item():10.2f} | {col.max().item():10.2f} | {col.mean().item():10.2f} | {col.std().item():10.2f}")

    # Check if all particles are identical
    if n_particles > 1:
        diffs = (pos[1:] - pos[0:1]).abs().max().item()
        print(f"  Max abs diff between particle 0 and others: {diffs:.6f}")
        if diffs < 1e-6:
            print("  *** WARNING: ALL PARTICLES IDENTICAL — init or PSO bug ***")


def exec_basket_benchmark():
    if not HAS_BASKET_API:
        log("[WARN] Basket API not detected.")
        return

    log("=" * 90)
    log("AMERICAN BASKET OPTION BENCHMARK")
    log("=" * 90)

    # ── QuantLib Baseline ────────────────────────────────────────────────────
    paths = 32768
    steps = 128
    rho = 0.7

    print(f"\n{'Library':<25} | {'Method':<25} | {'Median (ms)':<12} | {'Price':<12}")
    print("-" * 90)

    if HAS_QUANTLIB:
        today = ql.Date().todaysDate()
        ql.Settings.instance().evaluationDate = today
        ts  = ql.YieldTermStructureHandle(ql.FlatForward(today, r, ql.Actual365Fixed()))
        div = ql.YieldTermStructureHandle(ql.FlatForward(today, 0.0, ql.Actual365Fixed()))
        vol_handle = ql.BlackVolTermStructureHandle(
            ql.BlackConstantVol(today, ql.NullCalendar(), sigma, ql.Actual365Fixed()))
        procA = ql.BlackScholesMertonProcess(ql.QuoteHandle(ql.SimpleQuote(S0)), div, ts, vol_handle)
        procB = ql.BlackScholesMertonProcess(ql.QuoteHandle(ql.SimpleQuote(S0)), div, ts, vol_handle)
        b_opt = ql.BasketOption(
            ql.AverageBasketPayoff(ql.PlainVanillaPayoff(ql.Option.Put, K), [0.5, 0.5]),
            ql.AmericanExercise(today, today + ql.Period(int(T * 365), ql.Days)))
        b_opt.setPricingEngine(ql.MCAmericanBasketEngine(
            ql.StochasticProcessArray([procA, procB], [[1.0, rho], [rho, 1.0]]),
            'pseudorandom', timeSteps=steps, requiredSamples=paths, seed=42))
        med, _, _ = run_bench(lambda: (b_opt.recalculate(), b_opt.NPV())[1])
        print(f"{'QuantLib':<25} | {f'CPU LSMC ({paths//1000}k)':<25} | {med:12.4f} | {b_opt.NPV():12.6f}")

    # ══════════════════════════════════════════════════════════════════════════
    # DIAGNOSTIC: verify init produces diverse particles and payoff is correct
    # ══════════════════════════════════════════════════════════════════════════
    log("\n--- DIAGNOSTIC: Init & Payoff Verification ---")

    opt = BasketOptionConfig(
        initial_stock_prices=[S0, S0],
        strike_price=K,
        risk_free_rate=r,
        volatilities=[sigma, sigma],
        weights=[0.5, 0.5],
        correlation_matrix=[[1.0, rho], [rho, 1.0]],
        time_to_maturity=T,
        num_paths=paths,
        num_time_steps=steps,
        option_type=1,
        option_style="basket",
        exercise_style=1,
    )

    # Start with safest config: no fp16, no antithetic, no convergence exit
    comp_safe = ComputeConfig(
        compute_fraction=0.0,
        elementwise_block_size=256,
        reduction_block_size=32,
        max_iterations=100,
        sync_iters=1,
        seed=42,
        use_fixed_random=False,
        use_antithetic=False,
        use_fp16_cholesky=False,
        debug=False,
    )

    swarm = SwarmConfig(
        num_particles=64,
        inertia_weight=0.7298,
        cognitive_weight=1.49618,
        social_weight=1.49618,
    )

    # 1. Check particle diversity BEFORE optimize
    print("\n  Step 1: Check init diversity")
    flash_diag = FlashPSOBasket(option_config=opt, compute_config=comp_safe, swarm_config=swarm)
    _check_particle_diversity(flash_diag, "after init, before optimize")

    # 2. Run one eval (no PSO update) and check payoff
    flash_diag._PSO_update(iteration=0, eval_only=True)
    flash_diag._reduce_pbest()
    print(f"\n  Step 2: First eval (no PSO update)")
    print(f"  gbest_payoff = {flash_diag.gbest_payoff.item():.6f}")
    print(f"  gbest_pos[:4] = {flash_diag.gbest_pos.cpu().numpy()[:4]}")
    _check_particle_diversity(flash_diag, "after first eval")

    # 3. Run full optimize
    flash_diag2 = FlashPSOBasket(option_config=opt, compute_config=comp_safe, swarm_config=swarm)
    flash_diag2.optimize()
    print(f"\n  Step 3: Full optimize (safe config: no fp16, no antithetic, no early stop)")
    print(f"  Final price = {flash_diag2.get_option_price():.6f}")
    print(f"  gbest_pos[:4] = {flash_diag2.gbest_pos.cpu().numpy()[:4]}")
    _check_particle_diversity(flash_diag2, "after full optimize")

    # 4. Test with antithetic
    comp_anti = ComputeConfig(
        compute_fraction=0.0, elementwise_block_size=256, reduction_block_size=32,
        max_iterations=100, sync_iters=5, convergence_threshold=-1.0,
        seed=42, use_antithetic=True, use_fp16_cholesky=False, debug=False,
    )
    flash_anti = FlashPSOBasket(option_config=opt, compute_config=comp_anti, swarm_config=swarm)
    flash_anti.optimize()
    print(f"\n  Step 4: With antithetic (no fp16)")
    print(f"  Final price = {flash_anti.get_option_price():.6f}")

    # 5. Test with fp16
    comp_fp16 = ComputeConfig(
        compute_fraction=0.0, elementwise_block_size=256, reduction_block_size=32,
        max_iterations=100, sync_iters=5, convergence_threshold=-1.0,
        seed=42, use_antithetic=False, use_fp16_cholesky=True, debug=False,
    )
    flash_fp16 = FlashPSOBasket(option_config=opt, compute_config=comp_fp16, swarm_config=swarm)
    flash_fp16.optimize()
    print(f"\n  Step 5: With fp16 cholesky (no antithetic)")
    print(f"  Final price = {flash_fp16.get_option_price():.6f}")

    # 6. Test with both
    comp_both = ComputeConfig(
        compute_fraction=0.0, elementwise_block_size=256, reduction_block_size=32,
        max_iterations=100, sync_iters=5, convergence_threshold=-1.0,
        seed=42, use_antithetic=True, use_fp16_cholesky=True, debug=False,
    )
    flash_both = FlashPSOBasket(option_config=opt, compute_config=comp_both, swarm_config=swarm)
    flash_both.optimize()
    print(f"\n  Step 6: With both antithetic + fp16")
    print(f"  Final price = {flash_both.get_option_price():.6f}")

    # ══════════════════════════════════════════════════════════════════════════
    # PERFORMANCE: Compute Fraction Sweep (using safe config that works)
    # ══════════════════════════════════════════════════════════════════════════
    log("\n--- FlashPSO: Compute Fraction Sweep ---")
    print(f"{'Compute Frac':<14} | {'Median (ms)':<12} | {'Price':<12} | {'St VRAM (MB)':<14} | {'Notes':<30}")
    print("-" * 90)

    comp_bench = ComputeConfig(
        compute_fraction=0.0, elementwise_block_size=256, reduction_block_size=32,
        max_iterations=100, sync_iters=5, convergence_threshold=1e-6,
        seed=42, use_antithetic=False, use_fp16_cholesky=False, debug=False,
    )

    for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
        comp_bench.compute_fraction = frac

        def make_and_run(f=frac):
            c = ComputeConfig(
                compute_fraction=f, elementwise_block_size=256, reduction_block_size=32,
                max_iterations=100, sync_iters=5, convergence_threshold=1e-6,
                seed=42, use_antithetic=False, use_fp16_cholesky=False, debug=False,
            )
            flash = FlashPSOBasket(option_config=opt, compute_config=c, swarm_config=swarm)
            flash.optimize()
            return flash

        flash = make_and_run(frac)
        med, _, _ = run_bench(lambda f=frac: make_and_run(f))

        total_path_blocks = opt.num_paths // 256
        n_compute = round(frac * total_path_blocks)
        n_bw_paths = (total_path_blocks - n_compute) * 256
        padded = getattr(flash, 'padded_assets', 2)
        if opt.exercise_style == 0:
            st_mb = (steps * n_bw_paths * 4) / (1024 * 1024) if n_bw_paths > 0 else 0.0
        else:
            st_mb = (steps * padded * n_bw_paths * 4) / (1024 * 1024) if n_bw_paths > 0 else 0.0

        notes = "All from VRAM" if frac == 0.0 else "All on-the-fly" if frac == 1.0 else "Hybrid"
        print(f"{frac:<14.2f} | {med:12.4f} | {flash.get_option_price():12.6f} | {st_mb:14.2f} | {notes:<30}")

    # ══════════════════════════════════════════════════════════════════════════
    # PERFORMANCE: Path Count Scaling
    # ══════════════════════════════════════════════════════════════════════════
    log("\n--- FlashPSO: Path Count Scaling (compute_fraction=0.0) ---")
    print(f"{'Paths':<12} | {'Median (ms)':<12} | {'Price':<12}")
    print("-" * 50)

    for p in [1024, 4096, 16384, 32768, 65536, 131072]:
        def make_and_run_p(num_p=p):
            o = BasketOptionConfig(
                initial_stock_prices=[S0, S0], strike_price=K, risk_free_rate=r,
                volatilities=[sigma, sigma], weights=[0.5, 0.5],
                correlation_matrix=[[1.0, rho], [rho, 1.0]],
                time_to_maturity=T, num_paths=num_p, num_time_steps=steps,
                option_type=1, option_style="basket", exercise_style=0,
            )
            c = ComputeConfig(
                compute_fraction=0.0, elementwise_block_size=256, reduction_block_size=32,
                max_iterations=100, sync_iters=5, convergence_threshold=1e-6,
                seed=42, use_antithetic=False, use_fp16_cholesky=False, debug=False,
            )
            f = FlashPSOBasket(option_config=o, compute_config=c, swarm_config=swarm)
            f.optimize()
            return f

        flash = make_and_run_p(p)
        med, _, _ = run_bench(lambda num_p=p: make_and_run_p(num_p))
        print(f"{p:<12} | {med:12.4f} | {flash.get_option_price():12.6f}")
