"""Sweep row C5a: ONLINE concept-drift control at 120s. Retarget min1_online (river ARF+ADWIN, prequential,
warm 2022-23) to a 2-minute-ahead label. Keystone efficiency control: if an adaptive forest that re-fits to
recent bars still sits at ~0.50 AUC every year, 2m direction is GENUINE efficiency (not stale-model drift)."""
import os, numpy as np, pandas as pd, harness as H
import min1_online as ON
FEAT, PAIR, STRIDE, FEATS = ON.FEAT, ON.PAIR, ON.STRIDE, ON.FEATS
def load120(years):
    base = list(H.feature_cols("EURUSD")); cols = [c for c in FEATS if c in base]
    parts = []
    for y in years:
        p = f"{FEAT}/{PAIR}_{y}.parquet"
        if not os.path.exists(p): continue
        d = pd.read_parquet(p, columns=base + H.META_COLS); d = d[~d.index.duplicated(keep="last")]
        idx = d.index; c = d["close"].values; n = len(c)
        secs = idx.values.astype("datetime64[s]").astype("int64")
        contig = np.zeros(n, bool); contig[:n-2] = (secs[2:] - secs[:-2]) == 120
        fwd = np.full(n, np.nan); fwd[:n-2] = c[2:]; ret = fwd / c - 1.0
        valid = contig & np.isfinite(ret) & (ret != 0)
        df = d.loc[valid, cols].copy(); df["_y"] = (ret[valid] > 0).astype(int); df["_year"] = idx.year.values[valid]
        parts.append(df.iloc[::STRIDE] if STRIDE > 1 else df)
    out = pd.concat(parts); out[cols] = out[cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out, cols
ON.load = load120
ON.main()
