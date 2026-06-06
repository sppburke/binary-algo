"""BAR-PATTERN CNN — step 1: build 1-MINUTE OHLCV bars + 60s deriv-faithful direction labels.

SCOPE: (EURUSD, 60s direction). Source = the SAME 1-second tick store the production 60s book uses
(features_tick/{split}_1s.parquet, cols mid,imb,micro,spread,nt,tsz). We aggregate the `mid` series to
1-minute OHLC (closed/label='right' so the bar TIMESTAMP == the decision instant == end of bar, fully
causal) and `nt` to a tick-count volume. The 60s up/down label at each bar-close instant T is computed
with the EXACT deriv settlement used everywhere in this repo (min1_production.wc_ret logic, applied at
the bar-close query times): entry = next 1s tick after T+1s, exit = last 1s tick <= T+61s, mid-to-mid,
ties LOSE, window must not straddle a >TOL gap. This keeps the bars and the labels on ONE clock.

Output: ohlc_cache/EURUSD_1m_{split}.parquet with columns
  t(int epoch s, == bar close/decision instant), open, high, low, close, vol, y(=ret>0), mag(=|ret|), valid
Run:  ~/binary-algo-venv/bin/python barcnn_bars.py
"""
import os, time, numpy as np, pandas as pd

ROOT = "/media/sean/CORSAIR/binary-algo"
TICK = f"{ROOT}/features_tick"
OUT  = f"{ROOT}/ohlc_cache"
HS, TOL, LAG = 60, 10, 1          # 60s expiry, 10s settlement tolerance, +1s deriv entry lag (T&C 2.2.3.1)
SPLITS = ("train", "val", "test", "oos")


def labels_at(qT, ts, mid, hs=HS, tol=TOL, lag=LAG):
    """Deriv Rise/Fall 60s outcome at decision instants qT (epoch s), evaluated on the 1s tick timeline.
    IDENTICAL settlement to min1_production.wc_ret, just queried at the bar-close instants instead of every tick."""
    qT = np.asarray(qT, dtype="int64"); ts = np.asarray(ts, dtype="int64"); mid = np.asarray(mid, dtype=float)
    n = len(ts)
    entry_t = qT + lag
    ei = np.searchsorted(ts, entry_t, side="left")           # first tick at/after decision+lag
    exit_t = entry_t + hs
    xi = np.searchsorted(ts, exit_t, side="right") - 1        # last tick at/before expiry
    eic = np.clip(ei, 0, n - 1); xic = np.clip(xi, 0, n - 1)
    valid = (ei < n) & (xi > ei)
    valid &= (ts[eic] - entry_t) <= tol                       # entry tick within tol after order
    valid &= (exit_t - ts[xic]) <= tol                        # exit tick within tol before expiry
    ret = mid[xic] / mid[eic] - 1.0
    valid &= np.isfinite(ret)                                 # ret==0 stays valid = a losing tie
    return ret, valid


def build_split(sp):
    t0 = time.time()
    b = pd.read_parquet(f"{TICK}/{sp}_1s.parquet")            # DatetimeIndex (tz-aware UTC), 1s bars (gaps dropped)
    mid = b["mid"].astype(float); nt = b["nt"].astype(float)
    # 1-MINUTE OHLCV, right-closed/right-labeled -> bar timestamp == end of bar == decision instant (causal)
    o = mid.resample("1min", label="right", closed="right").first()
    h = mid.resample("1min", label="right", closed="right").max()
    l = mid.resample("1min", label="right", closed="right").min()
    c = mid.resample("1min", label="right", closed="right").last()
    v = nt.resample("1min", label="right", closed="right").sum()
    bars = pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "vol": v}).dropna(subset=["close"])
    # labels at each bar-close instant from the underlying 1s tick timeline
    ts1s = b.index.values.astype("datetime64[s]").astype("int64")
    mid1s = mid.values
    qT = bars.index.values.astype("datetime64[s]").astype("int64")
    ret, valid = labels_at(qT, ts1s, mid1s)
    bars["t"] = qT
    bars["y"] = (ret > 0).astype(int)
    bars["mag"] = np.abs(ret)
    bars["valid"] = valid
    bars = bars.reset_index(drop=True)
    os.makedirs(OUT, exist_ok=True)
    bars.to_parquet(f"{OUT}/EURUSD_1m_{sp}.parquet")
    # sanity: moved up-rate per year must be in [0.47,0.53] (else fake-flat mirage / settlement bug)
    yr = pd.to_datetime(bars["t"], unit="s", utc=True).dt.year.values
    moved = bars["valid"].values & (bars["mag"].values > 0)
    print(f"[{sp}] bars={len(bars):,} valid={int(bars['valid'].sum()):,} moved={int(moved.sum()):,} "
          f"({time.time()-t0:.0f}s)", flush=True)
    for Y in sorted(set(yr.tolist())):
        m = moved & (yr == Y)
        if m.sum() < 100: continue
        up = bars["y"].values[m].mean()
        flag = "" if 0.47 <= up <= 0.53 else "  <-- OUT OF BAND!"
        print(f"    {Y}: moved={int(m.sum()):>7,}  up_rate={up:.4f}{flag}", flush=True)
    return bars


if __name__ == "__main__":
    for sp in SPLITS:
        build_split(sp)
    print("[done] -> ohlc_cache/EURUSD_1m_{train,val,test,oos}.parquet", flush=True)
