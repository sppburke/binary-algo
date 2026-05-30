"""
Data loading + multi-timeframe causal feature engineering for 5-min binary direction.

Design:
  - Base decision grid = 1-minute bars (resampled from 10s).
  - Decision at end of 1m bar t; only info <= t is used in features (no lookahead).
  - Label = sign(close(t+H) - close(t)), H = 5 (minutes). Exact ties dropped.
  - Multi-timeframe context: indicators computed on 1m, 5m, 15m, 30m, 1h, 4h and
    aligned causally onto the 1m grid via the last *completed* higher-TF bar.
  - 'valid' mask drops decisions whose [t, t+H] window crosses a session/weekend gap.

All features are "question-based" / normalized per the Reddit guidance (trend up?,
RSI rising?, distance-to-EMA, range position, ...) rather than raw price levels.
"""
import os, glob, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

DATA_ROOT = "/media/sean/CORSAIR/tick_data/processed"
FEAT_DIR = "/media/sean/CORSAIR/binary-algo/features"
os.makedirs(FEAT_DIR, exist_ok=True)

HORIZON = 5  # minutes ahead for the binary label

# ----------------------------- loading -----------------------------------

def load_pair_year_10s(pair, year):
    files = sorted(glob.glob(os.path.join(DATA_ROOT, pair, f"{pair}_10s_{year}-*.parquet")))
    if not files:
        return None
    dfs = []
    for f in files:
        d = pd.read_parquet(f)
        # some files store datetime_utc as a column (RangeIndex), others as the index
        if "datetime_utc" in d.columns:
            d = d.set_index("datetime_utc")
        d.index = pd.to_datetime(d.index, utc=True, errors="coerce")
        dfs.append(d)
    df = pd.concat(dfs)
    df = df[df.index.notna()]
    df = df[~df.index.duplicated(keep="last")].sort_index()
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["close"])
    return df


def resample_1m(df10s):
    """Resample 10s -> 1m OHLCV. Mark bars that follow a >3min gap (weekend/holiday)."""
    o = df10s["open"].resample("1min").first()
    h = df10s["high"].resample("1min").max()
    l = df10s["low"].resample("1min").min()
    c = df10s["close"].resample("1min").last()
    v = df10s["volume"].resample("1min").sum()
    m = pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": v})
    m = m.dropna(subset=["close"])  # drop empty minutes (no ticks at all)
    # gap (in minutes) to previous available bar
    dt_min = m.index.to_series().diff().dt.total_seconds().div(60.0)
    m["gap_prev"] = dt_min.fillna(1.0).values
    return m


# --------------------------- indicators ----------------------------------

def _ema(s, span):
    return s.ewm(span=span, adjust=False).mean()

def _rsi(close, n=14):
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return (100 - 100/(1+rs)).fillna(50.0)

def _atr(h, l, c, n=14):
    pc = c.shift(1)
    tr = pd.concat([(h-l), (h-pc).abs(), (l-pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()

def _rolling_slope(s, n):
    # slope of linear fit over last n points, normalized by price.
    # slope*denom = sum(xd_i * c_i) since sum(xd)=0; compute via fixed-kernel convolution.
    x = np.arange(n); xd = x - x.mean(); denom = (xd**2).sum()
    v = s.values.astype(float)
    num = np.convolve(v, xd[::-1], mode="full")[n-1:len(v)]  # aligned to right edge
    out = np.full(len(v), np.nan)
    out[n-1:] = num
    return pd.Series(out/denom, index=s.index) / s

def _rolling_autocorr(ret, n):
    # lag-1 autocorrelation over rolling window, fully vectorized via rolling corr.
    return ret.rolling(n).corr(ret.shift(1))

def tf_features(bars, tf_label):
    """Compute a set of causal, normalized features on one timeframe's bar series.
    Returns a DataFrame indexed like `bars` (higher-TF timestamps)."""
    c, h, l = bars["close"], bars["high"], bars["low"]
    out = {}
    ret1 = c.pct_change()
    # momentum / returns over several lookbacks
    for k in (1, 3, 6, 12, 24):
        out[f"ret_{k}"] = c.pct_change(k)
    # EMA trend structure
    for span in (10, 20, 50, 100):
        ema = _ema(c, span)
        out[f"dist_ema{span}"] = c/ema - 1.0
        out[f"above_ema{span}"] = (c > ema).astype("float32")
    out["ema20_slope"] = _ema(c, 20).diff() / c
    out["ema_fast_slow"] = _ema(c, 10)/_ema(c, 50) - 1.0
    # RSI level + direction
    rsi = _rsi(c, 14)
    out["rsi"] = (rsi - 50) / 50
    out["rsi_rising"] = (rsi.diff() > 0).astype("float32")
    out["rsi_ob"] = (rsi > 70).astype("float32")
    out["rsi_os"] = (rsi < 30).astype("float32")
    # MACD
    macd = _ema(c, 12) - _ema(c, 26)
    sig = _ema(macd, 9)
    out["macd_hist"] = (macd - sig) / c
    out["macd_pos"] = (macd > sig).astype("float32")
    # Bollinger %b and width
    ma20 = c.rolling(20).mean(); sd20 = c.rolling(20).std()
    out["bb_pctb"] = (c - ma20) / (2*sd20 + 1e-12)
    out["bb_width"] = (4*sd20) / ma20
    # volatility
    out["rv_12"] = ret1.rolling(12).std()
    out["rv_24"] = ret1.rolling(24).std()
    out["atr_pct"] = _atr(h, l, c, 14) / c
    # range position over n
    for n in (12, 24, 48):
        hi = h.rolling(n).max(); lo = l.rolling(n).min()
        out[f"rangepos_{n}"] = (c - lo)/(hi - lo + 1e-12)
        out[f"dist_hi_{n}"] = c/hi - 1.0
        out[f"dist_lo_{n}"] = c/lo - 1.0
    # trend slope
    out["slope_20"] = _rolling_slope(c, 20)
    # autocorrelation of returns (momentum vs mean-revert regime)
    out["autocorr_10"] = _rolling_autocorr(ret1, 20)
    # recent up fraction & run
    sgn = np.sign(ret1).fillna(0)
    out["upfrac_10"] = (sgn > 0).rolling(10).mean()
    df = pd.DataFrame(out, index=bars.index)
    df.columns = [f"{tf_label}_{col}" for col in df.columns]
    return df


def resample_higher(m1, rule):
    o = m1["open"].resample(rule).first()
    h = m1["high"].resample(rule).max()
    l = m1["low"].resample(rule).min()
    c = m1["close"].resample(rule).last()
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": c}).dropna(subset=["close"])


TF_RULES = {"1m": None, "5m": "5min", "15m": "15min", "30m": "30min", "1h": "1h", "4h": "4h"}


def build_features(m1):
    """Build the full causal multi-timeframe feature matrix aligned to the 1m grid."""
    feats = []
    # 1m features on base grid
    feats.append(tf_features(m1[["open","high","low","close"]], "1m"))
    # higher timeframes: compute on resampled bars, then align with last COMPLETED bar.
    for tf, rule in TF_RULES.items():
        if rule is None:
            continue
        hb = resample_higher(m1, rule)
        f = tf_features(hb, tf)
        # shift by 1 so that at time t we only use the higher-TF bar that already CLOSED,
        # then reindex onto 1m grid with forward fill.
        f = f.shift(1).reindex(m1.index, method="ffill")
        feats.append(f)

    X = pd.concat(feats, axis=1)

    # ---- cross-timeframe agreement / time features (computed on 1m grid) ----
    above_cols = [c for c in X.columns if c.endswith("above_ema50")]
    if above_cols:
        X["mtf_trend_align"] = X[above_cols].mean(axis=1)
    rsi_cols = [c for c in X.columns if c.endswith("_rsi") and "rising" not in c]
    if rsi_cols:
        X["mtf_rsi_mean"] = X[rsi_cols].mean(axis=1)

    idx = m1.index
    hour = idx.hour + idx.minute/60.0
    X["hour_sin"] = np.sin(2*np.pi*hour/24)
    X["hour_cos"] = np.cos(2*np.pi*hour/24)
    X["dow"] = idx.dayofweek.astype("float32")
    # FX sessions (UTC approx): Asia 0-8, London 7-16, NY 12-21
    X["sess_london"] = (((idx.hour >= 7) & (idx.hour < 16)).astype("float32"))
    X["sess_ny"] = (((idx.hour >= 12) & (idx.hour < 21)).astype("float32"))
    X["sess_overlap"] = (((idx.hour >= 12) & (idx.hour < 16)).astype("float32"))
    # volume context
    v = m1["volume"]
    X["vol_z"] = (v - v.rolling(60).mean())/(v.rolling(60).std()+1e-9)
    X["zero_vol"] = (v == 0).astype("float32")
    X["gap_prev"] = m1["gap_prev"].values
    return X


def build_label(m1, horizon=HORIZON):
    c = m1["close"]
    fwd = c.shift(-horizon)
    ret = fwd/c - 1.0
    y = (ret > 0).astype("float32")
    y[ret == 0] = np.nan  # drop exact ties
    # validity: the [t, t+h] window must be contiguous (no weekend/holiday gap).
    gap = m1["gap_prev"]
    # gap_prev at bars t+1..t+h must all be ~1 minute
    future_gap_ok = pd.Series(True, index=m1.index)
    g = (gap <= 1.5).astype(int)
    # rolling forward: all of next `horizon` bars contiguous
    contig = g.shift(-1)
    for k in range(2, horizon+1):
        contig = contig * g.shift(-k)
    future_gap_ok = contig.fillna(0) > 0
    valid = future_gap_ok & fwd.notna() & y.notna()
    return y, ret, valid


def process_pair(pair, years=None):
    if years is None:
        years = [str(y) for y in range(2012, 2027)]
    for year in years:
        out_path = os.path.join(FEAT_DIR, f"{pair}_{year}.parquet")
        if os.path.exists(out_path):
            print(f"  {pair} {year}: exists, skip", flush=True)
            continue
        df10 = load_pair_year_10s(pair, year)
        if df10 is None:
            print(f"  {pair} {year}: no data", flush=True); continue
        m1 = resample_1m(df10)
        X = build_features(m1)
        y, ret, valid = build_label(m1)
        X["y"] = y.values
        X["fwd_ret"] = ret.values
        X["valid"] = valid.values
        X["close"] = m1["close"].values
        X = X.replace([np.inf, -np.inf], np.nan)
        # cast feature cols to float32 to save space
        for col in X.columns:
            if X[col].dtype == "float64":
                X[col] = X[col].astype("float32")
        X.to_parquet(out_path)
        nv = int(X["valid"].sum())
        print(f"  {pair} {year}: rows={len(X)} valid={nv} feat_cols={X.shape[1]-4} "
              f"y_mean={y[valid].mean():.4f}", flush=True)


if __name__ == "__main__":
    import sys
    pairs = sys.argv[1:] if len(sys.argv) > 1 else ["EURUSD"]
    for p in pairs:
        print(f"=== {p} ===", flush=True)
        process_pair(p)
