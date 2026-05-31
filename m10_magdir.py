"""10-MIN last untested combination: does DIRECTION become predictable on bars where the MAGNITUDE model predicts a big
move? Magnitude (|fwd10|>=Q75) is forecastable (AUC ~0.73); direction isn't (~0.52). The sign-invariance theorem predicts
gating direction by predicted-magnitude is NULL (magnitude ⟂ sign), but it has never been RUN at 10m. Test: train a 10-min
magnitude model on 2012-21, take the native-10 direction ensemble, and per held-out year bucket NY bars by predicted
magnitude — is direction accuracy in the HIGH-magnitude bucket higher AND test25-stable? Honest: per-window + combined,
non-overlap 600s, CI95. Reuses models/m10_EURUSD_direction_* (no dir retrain); trains one magnitude LGB.
"""
import os, json, time, numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import harness as H
from m10_production import _load, _dirproba, nonoverlap_chrono, boot

MODELS = "/media/sean/CORSAIR/binary-algo/models"; PAIR = "EURUSD"; HOR = 10; STRIDE = 3
base = list(H.feature_cols("EURUSD"))
SPL = {"train": [str(y) for y in range(2012, 2022)], "val": ["2022", "2023"], "test24": ["2024"], "test25": ["2025"], "oos": ["2026"]}

def load2(years, stride=1):
    """Return df[base] + _y (direction) + _absfwd (|fwd10 ret|) + _ts + sess_ny, contiguous 10-min, ties dropped for dir."""
    parts = []
    for y in years:
        p = f"{H.FEAT_DIR}/{PAIR}_{y}.parquet"
        if not os.path.exists(p): continue
        df = pd.read_parquet(p, columns=base + H.META_COLS); df = df[~df.index.duplicated(keep="last")]
        idx = df.index; c = df["close"].values; n = len(c)
        secs = idx.values.astype("datetime64[s]").astype("int64")
        contig = np.zeros(n, bool)
        if n > HOR: contig[:n - HOR] = (secs[HOR:] - secs[:-HOR]) == HOR * 60
        fwd = np.full(n, np.nan); fwd[:n - HOR] = c[HOR:]; ret = fwd / c - 1.0
        valid = contig & np.isfinite(ret)
        d = df.loc[valid, base].copy()
        d["_y"] = (ret[valid] > 0).astype(float); d["_absfwd"] = np.abs(ret[valid])
        d["_ts"] = secs[valid]; d["_ny"] = (df["sess_ny"].values.astype(float) > 0.5)[valid]
        parts.append(d.iloc[::stride] if stride > 1 else d)
    return pd.concat(parts)

def main():
    t0 = time.time()
    TR = load2(SPL["train"], STRIDE)
    q75 = float(np.nanpercentile(TR["_absfwd"].values, 75))
    ytr_mag = (TR["_absfwd"].values >= q75).astype(int)
    print(f"[magdir] train={len(TR):,} |fwd10| Q75={q75:.2e}; training magnitude LGB {time.time()-t0:.0f}s", flush=True)
    M = lgb.LGBMClassifier(objective="binary", metric="auc", learning_rate=0.03, num_leaves=255, min_child_samples=200,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=10, n_estimators=700, n_jobs=20, verbosity=-1)
    M.fit(TR[base].astype("float32"), ytr_mag)
    p, L, G, C = _load()  # native-10 direction ensemble
    print(f"[magdir] magnitude trained; loaded native-10 direction {time.time()-t0:.0f}s", flush=True)

    # held-out: per year, bucket NY bars by predicted magnitude quartile, report direction acc per bucket
    print(f"\n{'window':>7} {'magAUC':>7}  dir-acc by predicted-magnitude quartile (NY):  Q1(low)   Q2       Q3       Q4(HIGH)", flush=True)
    hi_corr = {}
    for w in ("test24", "test25", "oos"):
        D = load2(SPL[w])
        Xb = D[base].astype("float32")
        pmag = M.predict_proba(Xb)[:, 1]; pdir = _dirproba(p, L, G, C, D)
        y = D["_y"].astype(int).values; absf = D["_absfwd"].values; ny = D["_ny"].values
        ts = D["_ts"].values.astype("int64")
        ymag = (absf >= q75).astype(int)
        magauc = roc_auc_score(ymag, pmag)
        # quartiles of predicted magnitude within NY
        nyv = ny & (y >= 0)  # all
        qs = np.quantile(pmag[ny], [0.25, 0.5, 0.75])
        accs = []
        for lo, hi in [(-np.inf, qs[0]), (qs[0], qs[1]), (qs[1], qs[2]), (qs[2], np.inf)]:
            m = ny & (pmag > lo) & (pmag <= hi) & (absf > 0)  # exclude exact ties for direction
            a = ((pdir[m] > 0.5).astype(int) == y[m]).mean() if m.sum() > 50 else float("nan")
            accs.append((m.sum(), a))
        print(f"{w:>7} {magauc:7.3f}  " + "  ".join(f"{a:.3f}(n{nn})" for nn, a in accs), flush=True)
        # HIGH-magnitude bucket, then ALSO selective by direction confidence, non-overlap, for the binding-window test
        hm = ny & (pmag > qs[2]) & (absf > 0)
        # selective: top-30% direction-confidence within the high-mag bucket
        conf = np.abs(pdir - 0.5)
        if hm.sum() > 100:
            cthr = np.quantile(conf[hm], 0.70)
            sm = hm & (conf >= cthr); sel = nonoverlap_chrono(ts, sm)
            corr = ((pdir[sel] > 0.5).astype(int) == y[sel]).astype(float) if len(sel) else np.array([])
            hi_corr[w] = corr
            a = corr.mean() if len(corr) else float("nan"); lo2, hi2 = boot(corr)
            print(f"        HIGH-mag × dir-conf-top30% selective: n={len(sel)} acc={a:.3f} CI95=[{lo2:.3f},{hi2:.3f}]", flush=True)
    if len(hi_corr) == 3:
        A = np.concatenate([hi_corr[w] for w in ("test24", "test25", "oos")]); lo, hi = boot(A)
        fl = min(hi_corr[w].mean() for w in hi_corr if len(hi_corr[w]))
        print(f"\n[HIGH-mag selective COMBINED] n={len(A)} acc={A.mean():.3f} CI95=[{lo:.3f},{hi:.3f}] FLOOR(worst-window)={fl:.3f}", flush=True)
    print(f"[magdir] DONE {time.time()-t0:.0f}s", flush=True)

if __name__ == "__main__":
    main()
