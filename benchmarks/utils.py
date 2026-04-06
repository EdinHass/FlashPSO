import time
import triton.testing
from references.benchmarks import binomialAmericanOption

# pyright: reportUnboundVariable=false

try:
    import QuantLib as ql
    HAS_QUANTLIB = True
except ImportError:
    HAS_QUANTLIB = False

try:
    from optlib.optlib import american as _optlib_american_raw
    HAS_OPTLIB = True
except ImportError:
    try:
        from optlib import american as _optlib_american_raw
        HAS_OPTLIB = True
    except ImportError:
        HAS_OPTLIB = False

try:
    import fastvol as fv
    HAS_FASTVOL = True
except ImportError:
    HAS_FASTVOL = False

# ─── CONFIGURATION CONSTANTS (Defaults) ───
S0, r, sigma, T, K, OPT_TYPE = 100.0, 0.03, 0.3, 1.0, 110.0, 'P'

def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def run_bench(fn):
    # Standardized triton do_bench execution
    ms, min_ms, max_ms = triton.testing.do_bench(fn, warmup=10, rep=5, quantiles=[0.5, 0.2, 0.8])
    return ms, min_ms, max_ms

# Note: Removed the global cache so this can be called with different parameters!
def get_reference_price(s0=S0, k=K, r_rate=r, vol=sigma, t=T) -> float:
    log(f"Calculating reference price (Binomial 20k steps | Vol: {vol:.2f} | K: {k:.2f})...")
    price, _ = binomialAmericanOption(s0, k, r_rate, vol, 20000, t, opttype=OPT_TYPE)
    return float(price)

# ─── QUANTLIB WRAPPERS ───
def _get_ql_process(s0=S0, r_rate=r, vol=sigma):
    today = ql.Date().todaysDate()
    ql.Settings.instance().evaluationDate = today
    spot_handle = ql.QuoteHandle(ql.SimpleQuote(s0))
    flat_ts = ql.YieldTermStructureHandle(ql.FlatForward(today, r_rate, ql.Actual365Fixed()))
    div_ts = ql.YieldTermStructureHandle(ql.FlatForward(today, 0.0, ql.Actual365Fixed()))
    flat_vol = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(today, ql.NullCalendar(), vol, ql.Actual365Fixed()))
    bsm = ql.BlackScholesMertonProcess(spot_handle, div_ts, flat_ts, flat_vol)
    return today, bsm

def _quantlib_american_price(engine_type: str = 'binomial', steps: int = 2000, 
                             s0=S0, k=K, r_rate=r, vol=sigma, t=T) -> float:
    if not HAS_QUANTLIB: return float('nan')
    today, bsm = _get_ql_process(s0, r_rate, vol)
    option = ql.VanillaOption(ql.PlainVanillaPayoff(ql.Option.Put, k), ql.AmericanExercise(today, today + ql.Period(int(t * 365), ql.Days)))
    engine = ql.BinomialVanillaEngine(bsm, "crr", steps) if engine_type == 'binomial' else ql.FdBlackScholesVanillaEngine(bsm, steps, steps)
    option.setPricingEngine(engine)
    return option.NPV()

def _quantlib_asian_price(paths: int = 10000, steps: int = 128) -> float:
    if not HAS_QUANTLIB: return float('nan')
    today, bsm = _get_ql_process()
    dates = sorted(list({today + ql.Period(max(1, int(i * T * 365 / steps)), ql.Days) for i in range(1, steps + 1)}))
    option = ql.DiscreteAveragingAsianOption(ql.Average.Arithmetic, 0.0, 0, dates, ql.PlainVanillaPayoff(ql.Option.Put, K), ql.EuropeanExercise(today + ql.Period(int(T * 365), ql.Days)))
    option.setPricingEngine(ql.MCDiscreteArithmeticAPEngine(bsm, 'pseudorandom', requiredSamples=paths, seed=42))
    return option.NPV()

def _quantlib_asian_fd_price(t_grid=100, x_grid=100, a_grid=50) -> float:
    if not HAS_QUANTLIB: return float('nan')
    today, bsm = _get_ql_process()
    option = ql.ContinuousAveragingAsianOption(ql.Average.Arithmetic, ql.PlainVanillaPayoff(ql.Option.Put, K), ql.EuropeanExercise(today + ql.Period(int(T * 365), ql.Days)))
    option.setPricingEngine(ql.FdBlackScholesAsianEngine(bsm, t_grid, x_grid, a_grid))
    return option.NPV()

# ─── OTHER WRAPPERS ───
def _optlib_american_price() -> float:
    return _optlib_american_raw('p', S0, K, T, r, r, sigma)[0] if HAS_OPTLIB else float('nan')

def _fastvol_american_price(steps: int = 2000) -> float:
    if not HAS_FASTVOL: return float('nan')
    try: return float(fv.american.bopm.price_fp64(S0, K, T, r, 0.0, sigma, steps, False))
    except: return float('nan')
