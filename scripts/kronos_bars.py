"""Build H-MINUTE OHLCV bars + H-min deriv-faithful direction labels for the Kronos higher-freq zero-shot.

Generalizes barcnn_bars.py to any horizon H (minutes). Same one-clock construction: aggregate the 1s tick-store
`mid` to H-min OHLC (right-closed/labeled -> bar timestamp == decision instant, causal), `nt` to volume, and label
each bar-close instant T with the EXACT deriv settlement (min1_production.wc_ret logic) at expiry HS=H*60s,
ties-LOSE, window must not straddle a >TOL gap. Output matches the 1m cache schema so kronos_dir.py reads it.

Run: ~/binary-algo-venv/bin/python kronos_bars.py <H_min> [splits=test,oos]
  -> ohlc_cache/EURUSD_<H>m_{split}.parquet  (cols: t,open,high,low,close,vol,y,mag,valid)
"""
import os, sys, time, numpy as np, pandas as pd

H = int(sys.argv[1]) if len(sys.argv) > 1 else 5            # LABEL horizon (minutes) — forward deriv outcome
SPLITS = (sys.argv[2].split(",") if len(sys.argv) > 2 else ["test", "oos"])
GRID = int(sys.argv[3]) if len(sys.argv) > 3 else H         # context BAR grid (minutes); GRID<H => "fine" cache
ROOT = "/home/sean/git/binary-algo"; TICK = f"{ROOT}/features_tick"; OUT = f"{ROOT}/ohlc_cache"
HS = H * 60; TOL = max(10, HS // 20); LAG = 1
NAME = f"EURUSD_{H}m" if GRID == H else f"EURUSD_g{GRID}_h{H}"


def labels_at(qT, ts, mid, hs=HS, tol=TOL, lag=LAG):
    qT = np.asarray(qT, "int64"); ts = np.asarray(ts, "int64"); mid = np.asarray(mid, float); n = len(ts)
    entry_t = qT + lag; ei = np.searchsorted(ts, entry_t, side="left")
    exit_t = entry_t + hs; xi = np.searchsorted(ts, exit_t, side="right") - 1
    eic = np.clip(ei, 0, n - 1); xic = np.clip(xi, 0, n - 1)
    valid = (ei < n) & (xi > ei) & ((ts[eic] - entry_t) <= tol) & ((exit_t - ts[xic]) <= tol)
    ret = mid[xic] / mid[eic] - 1.0; valid &= np.isfinite(ret)
    return ret, valid


def build_split(sp):
    t0 = time.time()
    b = pd.read_parquet(f"{TICK}/{sp}_1s.parquet", columns=["mid", "nt"])
    mid = b["mid"].astype(float); nt = b["nt"].astype(float); rs = f"{GRID}min"
    o = mid.resample(rs, label="right", closed="right").first()
    h = mid.resample(rs, label="right", closed="right").max()
    l = mid.resample(rs, label="right", closed="right").min()
    c = mid.resample(rs, label="right", closed="right").last()
    v = nt.resample(rs, label="right", closed="right").sum()
    bars = pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "vol": v}).dropna(subset=["close"])
    ts1s = b.index.values.astype("datetime64[s]").astype("int64"); mid1s = mid.values
    qT = bars.index.values.astype("datetime64[s]").astype("int64")
    ret, valid = labels_at(qT, ts1s, mid1s)
    bars["t"] = qT; bars["y"] = (ret > 0).astype(int); bars["mag"] = np.abs(ret); bars["valid"] = valid
    bars = bars.reset_index(drop=True); os.makedirs(OUT, exist_ok=True)
    bars.to_parquet(f"{OUT}/{NAME}_{sp}.parquet")
    yr = pd.to_datetime(bars["t"], unit="s", utc=True).dt.year.values
    moved = bars["valid"].values & (bars["mag"].values > 0)
    print(f"[{NAME} {sp}] grid={GRID}m label={H}m bars={len(bars):,} valid={int(bars['valid'].sum()):,} moved={int(moved.sum()):,} ({time.time()-t0:.0f}s)", flush=True)
    for Y in sorted(set(yr.tolist())):
        m = moved & (yr == Y)
        if m.sum() < 100: continue
        up = bars["y"].values[m].mean(); flag = "" if 0.47 <= up <= 0.53 else "  <-- OUT OF BAND!"
        print(f"    {Y}: moved={int(m.sum()):>7,} up_rate={up:.4f}{flag}", flush=True)


if __name__ == "__main__":
    for sp in SPLITS:
        build_split(sp)
    print(f"[done] -> ohlc_cache/{NAME}_{{{','.join(SPLITS)}}}.parquet", flush=True)
