"""Sweep row C5a: ONLINE concept-drift control at 300s (5m). Retarget min1_online (river ARF+ADWIN, prequential,
warm 2022-23) to a 5-minute-ahead label on contiguous 1-min EURUSD bars. KEYSTONE + adversarial confirmation of
the certified (5m,UP) edge: an adaptive forest that re-fits to recent bars is a different model class with NO
threshold selection. If it ALSO finds 5m direction signal (AUC>0.52, NY/selective acc>0.55 in 2025) it
corroborates m5xp UP; if it sits at ~0.50 every year, the gated UP edge stands alone (selection-robust check).
Label: sign(close[t+5min]-close[t]) over a wall-clock-contiguous 300s window; ties (ret==0) excluded."""
import os, numpy as np, pandas as pd, harness as H
import min1_online as ON
FEAT, PAIR, STRIDE, FEATS = ON.FEAT, ON.PAIR, ON.STRIDE, ON.FEATS


def load300(years):
    base = list(H.feature_cols("EURUSD")); cols = [c for c in FEATS if c in base]
    parts = []
    for y in years:
        p = f"{FEAT}/{PAIR}_{y}.parquet"
        if not os.path.exists(p): continue
        d = pd.read_parquet(p, columns=base + H.META_COLS); d = d[~d.index.duplicated(keep="last")]
        idx = d.index; c = d["close"].values; n = len(c)
        secs = idx.values.astype("datetime64[s]").astype("int64")
        contig = np.zeros(n, bool); contig[:n-5] = (secs[5:] - secs[:-5]) == 300
        fwd = np.full(n, np.nan); fwd[:n-5] = c[5:]; ret = fwd / c - 1.0
        valid = contig & np.isfinite(ret) & (ret != 0)
        df = d.loc[valid, cols].copy(); df["_y"] = (ret[valid] > 0).astype(int); df["_year"] = idx.year.values[valid]
        # keep sess_ny for the NY-conditional read (it's in FEATS already as sess_ny)
        parts.append(df.iloc[::STRIDE] if STRIDE > 1 else df)
    out = pd.concat(parts); out[cols] = out[cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out, cols


ON.load = load300
ON.main()
