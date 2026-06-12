"""FAITHFUL CPCV of the ACTUAL m15_production book — settles whether the 0.647 is robust or era-favorable.

Unlike cpcv_certify.py (which used a SINGLE LGBM + q33 + all-era pool — not a fair test), this replicates the real m15 book:
  - the 3-MODEL ENSEMBLE (lgb + xgb + cat), m15 hyperparams (trees reduced to fit the 15-path budget — the one approximation),
  - the real GATE-SELECTION protocol: bb_width_thr = q{10,20,33} percentile of TRAIN-fit bbw, (q,cov) chosen on a VAL-tune split
    by max VAL accuracy (m15 V27 protocol), then gate = bbw<=thr & sess_ny & conf>=conf_thr,
  - the same LABEL (15m, ret>0, ties EXCLUDED) and LIVE-FAITHFUL chronological non-overlap (M15.nonoverlap_chrono, 900s),
applied under CombinatorialPurgedCV (N=6 groups, k=2 test => C(6,2)=15 OOS paths, PURGE+EMBARGO=1 label-horizon=15min).
Reports the 15-path GATED-SELECTIVE accuracy distribution (mean/p10/min/max) vs the chronological-split headline 0.647, plus each
path's test-era so we can see recent-vs-old. Memory-safe: one parquet at a time -> compact float32; subsample train fit.
"""
import os, json, time, numpy as np, pandas as pd
from itertools import combinations
import lightgbm as lgb, xgboost as xgb
from catboost import CatBoostClassifier
import harness as H
import m15_production as M15

FEAT = "/home/sean/git/binary-algo/features"
base = list(H.feature_cols("EURUSD"))
HOR = 15                      # 15 one-minute bars
N_GROUPS, K_TEST = 6, 2       # C(6,2)=15 purged-combinatorial paths
SUB_FIT = 150_000
YEARS = list(range(2012, 2027))
GAP_S = HOR * 60

def load_pooled():
    Xs, ys, tss, bbws, nys = [], [], [], [], []
    for y in YEARS:
        p = f"{FEAT}/EURUSD_{y}.parquet"
        if not os.path.exists(p): continue
        df = pd.read_parquet(p, columns=base + H.META_COLS); df = df[~df.index.duplicated(keep="last")].sort_index()
        c = df["close"].values; n = len(c)
        secs = df.index.values.astype("datetime64[s]").astype("int64")
        contig = np.zeros(n, bool); contig[:n - HOR] = (secs[HOR:] - secs[:-HOR]) == HOR * 60
        fwd = np.full(n, np.nan); fwd[:n - HOR] = c[HOR:]; ret = fwd / c - 1.0
        valid = contig & np.isfinite(ret) & (ret != 0)
        Xs.append(df.loc[valid, base].values.astype("float32")); ys.append((ret[valid] > 0).astype("int8"))
        tss.append(secs[valid]); bbws.append(df["15m_bb_width"].values.astype("float32")[valid])
        nys.append((df["sess_ny"].values.astype("float32") > 0.5)[valid]); del df
    X = np.concatenate(Xs); y = np.concatenate(ys); ts = np.concatenate(tss)
    bbw = np.concatenate(bbws); ny = np.concatenate(nys)
    o = np.argsort(ts, kind="mergesort")
    return X[o], y[o], ts[o], bbw[o], ny[o]

def train_ensemble(Xf, yf):
    L = lgb.LGBMClassifier(objective="binary", metric="auc", num_leaves=255, learning_rate=0.03, n_estimators=600,
                           subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=10, n_jobs=20, verbosity=-1).fit(Xf, yf)
    G = xgb.XGBClassifier(n_estimators=700, learning_rate=0.03, max_depth=8, subsample=0.8, colsample_bytree=0.5,
                          tree_method="hist", n_jobs=20, eval_metric="auc").fit(Xf, yf)
    C = CatBoostClassifier(iterations=700, learning_rate=0.03, depth=8, l2_leaf_reg=10, eval_metric="AUC",
                           thread_count=20, verbose=False).fit(np.nan_to_num(Xf, nan=-999), yf)
    return L, G, C

def proba(L, G, C, X):
    return (L.predict_proba(X)[:, 1] + G.predict_proba(X)[:, 1] + C.predict_proba(np.nan_to_num(X, nan=-999))[:, 1]) / 3.0

def boot(c, nb=4000, seed=7):
    c = np.asarray(c, float)
    if len(c) < 5: return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed); n = len(c)
    a = np.array([c[rng.integers(0, n, n)].mean() for _ in range(nb)]); return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))

def main():
    t0 = time.time()
    X, y, ts, bbw, ny = load_pooled()
    n = len(y); print(f"[cpcv15] pooled n={n:,} up-rate={y.mean():.4f} load {time.time()-t0:.0f}s", flush=True)
    edges = np.linspace(0, n, N_GROUPS + 1).astype(int)
    grp = [(edges[i], edges[i + 1]) for i in range(N_GROUPS)]
    embargo = HOR * 60; purge = HOR * 60 + embargo
    accs = []
    print(f"[cpcv15] {len(list(combinations(range(N_GROUPS),K_TEST)))} purged paths; 3-model ensemble + q-tuned gate", flush=True)
    for pi, testg in enumerate(combinations(range(N_GROUPS), K_TEST)):
        te = np.zeros(n, bool)
        for gi in testg: te[grp[gi][0]:grp[gi][1]] = True
        te_lo = min(ts[grp[gi][0]] for gi in testg); te_hi = max(ts[grp[gi][1] - 1] for gi in testg)
        # purge+embargo: drop train obs whose [t,t+HOR] window comes within `purge` of any test span
        tr = ~te
        for gi in testg:
            a, b = ts[grp[gi][0]], ts[grp[gi][1] - 1]
            tr &= ~((ts >= a - purge) & (ts <= b + embargo))
        tri = np.where(tr)[0]
        if len(tri) < 5000: continue
        # chronological fit / val-tune split (last 20% of train-time = gate-tuning VAL)
        cut = tri[int(len(tri) * 0.8)]; ct = ts[cut]
        fit = tri[ts[tri] < ct]; val = tri[ts[tri] >= ct]
        if len(fit) > SUB_FIT: fit = fit[np.linspace(0, len(fit) - 1, SUB_FIT).astype(int)]
        L, G, C = train_ensemble(X[fit], y[fit])
        pv = proba(L, G, C, X[val])
        # gate selection: bb_width_thr from FIT bbw percentile (m15), (q,cov) by max VAL acc
        best = None
        for q in (10, 20, 33):
            bthr = np.nanpercentile(bbw[fit], q); gmask = (bbw[val] <= bthr) & (ny[val] > 0)
            if gmask.sum() < 100: continue
            for cov in (0.05, 0.10):
                cthr = np.quantile(np.abs(pv[gmask] - 0.5), 1 - cov); sel = gmask & (np.abs(pv - 0.5) >= cthr)
                if sel.sum() < 100: continue
                a = ((pv[sel] > 0.5).astype(int) == y[val][sel]).mean()
                if best is None or a > best[0]: best = (a, bthr, cthr, q, cov)
        if best is None: continue
        _, bthr, cthr, q, cov = best
        # evaluate on test, live-faithful
        pt = proba(L, G, C, X[te]); tst = ts[te]
        gt = (bbw[te] <= bthr) & (ny[te] > 0) & (np.abs(pt - 0.5) >= cthr)
        sel = M15.nonoverlap_chrono(tst, gt, GAP_S)
        if len(sel) < 10: continue
        corr = ((pt[sel] > 0.5).astype(int) == y[te][sel]).astype(float); a = corr.mean()
        accs.append(a)
        yr_lo = int(str(np.datetime64(int(te_lo), "s"))[:4]); yr_hi = int(str(np.datetime64(int(te_hi), "s"))[:4])
        print(f"  path{pi:>2} test-grps={testg} era={yr_lo}-{yr_hi} q{q} cov{cov:.0%} n={len(sel)} acc={a:.4f} {time.time()-t0:.0f}s", flush=True)
    accs = np.array(accs)
    lo, hi = boot(accs) if len(accs) >= 5 else (float("nan"), float("nan"))
    print(f"\n[cpcv15] FAITHFUL CPCV ({len(accs)} paths): mean={accs.mean():.4f} p10={np.percentile(accs,10):.4f} "
          f"min={accs.min():.4f} max={accs.max():.4f} | CI95(mean)=[{lo:.4f},{hi:.4f}]", flush=True)
    print(f"[cpcv15] vs m15_production chronological-split headline 0.647 (2024 .689/2025 .582/2026 .663). breakeven 0.541.", flush=True)
    json.dump({"n_paths": int(len(accs)), "mean": float(accs.mean()), "p10": float(np.percentile(accs, 10)),
               "min": float(accs.min()), "max": float(accs.max()), "all": accs.tolist(),
               "note": "FAITHFUL: 3-model ensemble + q-tuned gate, 15 purged-combinatorial paths, vs headline 0.647"},
              open("min15_cpcv_result.json", "w"), indent=1)
    print(f"[cpcv15] DONE {time.time()-t0:.0f}s -> min15_cpcv_result.json", flush=True)

if __name__ == "__main__":
    main()
