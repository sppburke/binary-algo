"""1-MIN ONLINE / CONCEPT-DRIFT ADAPTIVE MODEL (final creative lever) — is the 2025 wall stale-model drift, or real efficiency?

Every model tried so far is FROZEN (fit on the past, judged on the future) — and the 2025 macro inversion makes that map
anti-transfer (corr(VAL,OOS)=-0.54). This tests the opposite: a CONTINUOUSLY-ADAPTING model with an explicit drift detector
(river Adaptive Random Forest, ADWIN-based) trained PREQUENTIALLY (test-then-train) in strict time order on 1-min bars. If the
2025 regime is merely a delayed concept drift, an online learner that re-fits to recent bars should hold its 60s direction
accuracy through 2025; if 2025 is genuinely efficient (as walk-forward's mere +0.017 lift suggested), online learning won't help.

Label: sign(close[t+1min]-close[t]) on contiguous 1-min EURUSD bars (next-bar outcomes are DISJOINT 60s windows -> naturally
non-overlapping). Strict no-look-ahead: predict bar t, THEN learn (x_t,y_t). Warm-up on 2022-2023 (learn only). Per-window
confidence covcurve on 2024/2025/2026 (cross-window comparable, like m5_xpair). Ties (ret==0) excluded. Compact feature subset
for river throughput; stride to keep prequential tractable.
"""
import os, sys, time, json, numpy as np, pandas as pd
import harness as H
from river import forest, drift, preprocessing, linear_model, optim

FEAT = "/home/sean/git/binary-algo/features"; PAIR = "EURUSD"
STRIDE = int(os.environ.get("ON_STRIDE", "5"))     # prequential over every STRIDE-th bar (throughput)
MODELKIND = os.environ.get("ON_MODEL", "arf")      # arf | logreg
# compact, causal feature subset (REAL base cols) — multi-TF momentum/vol/range/trend/session
FEATS = ["1m_ret_1","1m_ret_3","1m_ret_6","5m_ret_1","5m_ret_3","15m_ret_1",
         "1m_rsi","5m_rsi","15m_rsi","1m_bb_width","5m_bb_width","15m_bb_width","1m_atr_pct","5m_atr_pct",
         "1m_dist_ema20","5m_dist_ema20","1m_rangepos_24","5m_rangepos_24","1m_autocorr_10","5m_autocorr_10",
         "1m_macd_hist","5m_macd_hist","1m_bb_pctb","5m_bb_pctb","mtf_trend_align","mtf_rsi_mean","vol_z",
         "sess_ny","sess_london","sess_overlap","hour_sin","hour_cos","gap_prev"]

def load(years):
    base = list(H.feature_cols("EURUSD"))
    cols = [c for c in FEATS if c in base]
    parts = []
    for y in years:
        p = f"{FEAT}/{PAIR}_{y}.parquet"
        if not os.path.exists(p): continue
        d = pd.read_parquet(p, columns=base + H.META_COLS); d = d[~d.index.duplicated(keep="last")]
        idx = d.index; c = d["close"].values; n = len(c)
        secs = idx.values.astype("datetime64[s]").astype("int64")
        contig = np.zeros(n, bool); contig[:n-1] = (secs[1:]-secs[:-1]) == 60
        fwd = np.full(n, np.nan); fwd[:n-1] = c[1:]; ret = fwd/c - 1.0
        valid = contig & np.isfinite(ret) & (ret != 0)
        df = d.loc[valid, cols].copy()
        df["_y"] = (ret[valid] > 0).astype(int); df["_year"] = idx.year.values[valid]
        parts.append(df.iloc[::STRIDE] if STRIDE > 1 else df)
    out = pd.concat(parts)
    out[cols] = out[cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)   # river can't ingest NaN/inf
    return out, cols

def make_model(cols):
    if MODELKIND == "logreg":   # online SGD logistic regression — fast, memory-trivial, adapts continuously
        return preprocessing.StandardScaler() | linear_model.LogisticRegression(optimizer=optim.SGD(0.01))
    if MODELKIND == "hat":      # single Hoeffding ADAPTIVE tree (ADWIN-driven), lighter than the forest
        from river import tree
        return tree.HoeffdingAdaptiveTreeClassifier(grace_period=300, max_depth=12, seed=0)
    # Adaptive Random Forest, drift-native; capped tree growth (max_depth/grace) to bound memory over ~300k samples
    return forest.ARFClassifier(n_models=5, max_depth=12, grace_period=300,
                                drift_detector=drift.ADWIN(), warning_detector=drift.ADWIN(), seed=0)

def main():
    t0 = time.time()
    print(f"[online] model={MODELKIND} stride={STRIDE}", flush=True)
    WARM, _ = load(["2022", "2023"]); EVAL, cols = load(["2024", "2025", "2026"])
    print(f"[online] feats={len(cols)} warm={len(WARM)} eval={len(EVAL)} loaded {time.time()-t0:.0f}s", flush=True)
    model = make_model(cols)
    # warm-up: learn only (no eval), strict order
    Xw = WARM[cols].astype("float32").values; yw = WARM["_y"].values
    for i in range(len(Xw)):
        xd = {c: float(v) for c, v in zip(cols, Xw[i])}
        model.learn_one(xd, int(yw[i]))
        if i and i % 50000 == 0: print(f"[online] warm {i}/{len(Xw)} {time.time()-t0:.0f}s", flush=True)
    print(f"[online] warm-up done on {len(Xw)} bars {time.time()-t0:.0f}s", flush=True)
    # prequential on eval: predict THEN learn
    Xe = EVAL[cols].astype("float32").values; ye = EVAL["_y"].values; yr = EVAL["_year"].values
    pcache = np.full(len(Xe), np.nan)
    for i in range(len(Xe)):
        xd = {c: float(v) for c, v in zip(cols, Xe[i])}
        pp = model.predict_proba_one(xd); p1 = pp.get(1, 0.5) if pp else 0.5
        pcache[i] = p1
        model.learn_one(xd, int(ye[i]))
        if i and i % 200000 == 0: print(f"[online] prequential {i}/{len(Xe)} {time.time()-t0:.0f}s", flush=True)
    print(f"[online] prequential done {time.time()-t0:.0f}s", flush=True)
    # per-window confidence covcurve (cross-window comparable); naturally non-overlapping (next-bar disjoint windows)
    conf = np.abs(pcache - 0.5); pred = (pcache > 0.5).astype(int)
    print(f"\n{'window':>7} {'AUC':>6} | conf-thr -> n, acc (selective)", flush=True)
    from sklearn.metrics import roc_auc_score
    summary = {}
    for w, yy in (("2024", 2024), ("2025", 2025), ("2026", 2026)):
        m = (yr == yy) & np.isfinite(pcache)
        auc = roc_auc_score(ye[m], pcache[m])
        line = f"{w:>7} {auc:6.3f} |"
        cells = {}
        for thr in (0.0, 0.02, 0.04, 0.06, 0.08, 0.10):
            mm = m & (conf >= thr)
            if mm.sum() < 30: cells[thr] = (int(mm.sum()), float("nan")); continue
            acc = (pred[mm] == ye[mm]).mean(); cells[thr] = (int(mm.sum()), float(acc))
            line += f"  {thr:.2f}:n{mm.sum()},{acc:.3f}"
        print(line, flush=True); summary[w] = dict(auc=float(auc), cov={str(k): v for k, v in cells.items()})
    # binding-window read: best selective test25 vs 0.56/0.65
    json.dump(summary, open("models/min1_online_summary.json", "w"), indent=2, default=float)
    print(f"\n[online] DONE {time.time()-t0:.0f}s  (2025 is the binding window; need selective >=0.56 to beat wall, >0.65 for goal)", flush=True)

if __name__ == "__main__":
    main()
