"""PER-SIDE full-refit CPCV (step a+e) of the m30 base book -> certifies/floors (30m,UP) and (30m,DOWN).

Faithful retarget of m10_cpcv_side.py / m15_cpcv_side.py (the proven keystone-certification harness) to HOR=30.
Same purged-combinatorial design (6 groups, C(6,2)=15 paths, purge+embargo = 1 label-horizon, per-path REFIT of
the 3-model ensemble + VAL-tuned gate). Within each path's selected test trades, split by PREDICTED side -> per-side
path-accuracy distribution (mean/p10/min/max + frac-paths-clear-0.541).

ONLY changes vs m10: HOR 10->30, gate feature 5m_bb_width -> 1h_bb_width (the deployable EURUSD.m30.v1 gate,
m30_production.py comp_tf='1h'), nonoverlap via m30_production. This produces the honest base-book (30m,UP)/(30m,DOWN)
floors (both keys were UNTESTED) — the number the cross-pair keystone (m30_xpair_cpcv.py) must beat.

CERTIFY a side iff refit per-side p10 >= 0.541 AND >= ~80% of paths clear 0.541. Breakeven 0.541.
Usage: python m30_cpcv_side.py
"""
import os, json, time, numpy as np, pandas as pd
from itertools import combinations
import lightgbm as lgb, xgboost as xgb
from catboost import CatBoostClassifier
import harness as H
import m30_production as M30

FEAT = "/home/sean/git/binary-algo/features"
base = list(H.feature_cols("EURUSD"))
HOR = 30
N_GROUPS, K_TEST = 6, 2
SUB_FIT = 150_000
YEARS = list(range(2012, 2027))
GAP_S = HOR * 60
BE = 0.541
MIN_SIDE_N = 10
GATE_FEAT = "1h_bb_width"


def load_pooled():
    Xs, ys, tss, bbws, nys = [], [], [], [], []
    for y in YEARS:
        p = f"{FEAT}/EURUSD_{y}.parquet"
        if not os.path.exists(p):
            continue
        df = pd.read_parquet(p, columns=base + H.META_COLS); df = df[~df.index.duplicated(keep="last")].sort_index()
        c = df["close"].values; n = len(c)
        secs = df.index.values.astype("datetime64[s]").astype("int64")
        contig = np.zeros(n, bool); contig[:n - HOR] = (secs[HOR:] - secs[:-HOR]) == HOR * 60
        fwd = np.full(n, np.nan); fwd[:n - HOR] = c[HOR:]; ret = fwd / c - 1.0
        valid = contig & np.isfinite(ret) & (ret != 0)
        Xs.append(df.loc[valid, base].values.astype("float32")); ys.append((ret[valid] > 0).astype("int8"))
        tss.append(secs[valid]); bbws.append(df[GATE_FEAT].values.astype("float32")[valid])
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


def summ(a):
    a = np.array(a, float)
    if len(a) == 0:
        return {"n_paths": 0}
    return {"n_paths": int(len(a)), "mean": round(float(a.mean()), 4), "p10": round(float(np.percentile(a, 10)), 4),
            "min": round(float(a.min()), 4), "max": round(float(a.max()), 4),
            "frac_clear_BE": round(float((a >= BE).mean()), 3)}


def main():
    t0 = time.time()
    X, y, ts, bbw, ny = load_pooled()
    n = len(y); print(f"[cpcv30-side] pooled n={n:,} up-rate={y.mean():.4f} gate={GATE_FEAT} load {time.time()-t0:.0f}s", flush=True)
    edges = np.linspace(0, n, N_GROUPS + 1).astype(int)
    grp = [(edges[i], edges[i + 1]) for i in range(N_GROUPS)]
    embargo = HOR * 60; purge = HOR * 60 + embargo
    accs_c, accs_u, accs_d = [], [], []
    nside = {"UP": [], "DOWN": []}
    print(f"[cpcv30-side] {len(list(combinations(range(N_GROUPS),K_TEST)))} purged paths; per-side split", flush=True)
    for pi, testg in enumerate(combinations(range(N_GROUPS), K_TEST)):
        te = np.zeros(n, bool)
        for gi in testg:
            te[grp[gi][0]:grp[gi][1]] = True
        te_lo = min(ts[grp[gi][0]] for gi in testg); te_hi = max(ts[grp[gi][1] - 1] for gi in testg)
        tr = ~te
        for gi in testg:
            a, b = ts[grp[gi][0]], ts[grp[gi][1] - 1]
            tr &= ~((ts >= a - purge) & (ts <= b + embargo))
        tri = np.where(tr)[0]
        if len(tri) < 5000:
            continue
        cut = tri[int(len(tri) * 0.8)]; ct = ts[cut]
        fit = tri[ts[tri] < ct]; val = tri[ts[tri] >= ct]
        if len(fit) > SUB_FIT:
            fit = fit[np.linspace(0, len(fit) - 1, SUB_FIT).astype(int)]
        L, G, C = train_ensemble(X[fit], y[fit])
        pv = proba(L, G, C, X[val])
        best = None
        for q in (10, 20, 33):
            bthr = np.nanpercentile(bbw[fit], q); gmask = (bbw[val] <= bthr) & (ny[val] > 0)
            if gmask.sum() < 100:
                continue
            for cov in (0.05, 0.10):
                cthr = np.quantile(np.abs(pv[gmask] - 0.5), 1 - cov); sel = gmask & (np.abs(pv - 0.5) >= cthr)
                if sel.sum() < 100:
                    continue
                a = ((pv[sel] > 0.5).astype(int) == y[val][sel]).mean()
                if best is None or a > best[0]:
                    best = (a, bthr, cthr, q, cov)
        if best is None:
            continue
        _, bthr, cthr, q, cov = best
        pt = proba(L, G, C, X[te]); tst = ts[te]
        gt = (bbw[te] <= bthr) & (ny[te] > 0) & (np.abs(pt - 0.5) >= cthr)
        sel = M30.nonoverlap_chrono(tst, gt, GAP_S)
        if len(sel) < 10:
            continue
        pred = (pt[sel] > 0.5).astype(int); yt = y[te][sel]
        corr = (pred == yt).astype(float); ac = corr.mean(); accs_c.append(ac)
        su = pred == 1; sd = pred == 0
        au = ad = None
        if su.sum() >= MIN_SIDE_N:
            au = float((pred[su] == yt[su]).mean()); accs_u.append(au); nside["UP"].append(int(su.sum()))
        if sd.sum() >= MIN_SIDE_N:
            ad = float((pred[sd] == yt[sd]).mean()); accs_d.append(ad); nside["DOWN"].append(int(sd.sum()))
        yr_lo = int(str(np.datetime64(int(te_lo), "s"))[:4]); yr_hi = int(str(np.datetime64(int(te_hi), "s"))[:4])
        print(f"  path{pi:>2} era={yr_lo}-{yr_hi} q{q} cov{cov:.0%} n={len(sel)} comb={ac:.4f} "
              f"UP={('%.4f'%au) if au is not None else '  -  '}(n{int(su.sum())}) "
              f"DOWN={('%.4f'%ad) if ad is not None else '  -  '}(n{int(sd.sum())}) {time.time()-t0:.0f}s", flush=True)
    res = {"model": "m30 base 3-model ensemble @HOR=30 (refit-CPCV)", "gate_feat": GATE_FEAT,
           "combined": summ(accs_c), "UP": summ(accs_u), "DOWN": summ(accs_d),
           "median_path_n": {"UP": int(np.median(nside["UP"])) if nside["UP"] else 0,
                             "DOWN": int(np.median(nside["DOWN"])) if nside["DOWN"] else 0},
           "breakeven": BE,
           "CERT_RULE": "side CERTIFIED iff per-side p10>=0.541 AND frac_clear_BE>=0.80; this is the base-book FLOOR for the keystone to beat",
           "all_paths": {"combined": accs_c, "UP": accs_u, "DOWN": accs_d}}
    for s in ("combined", "UP", "DOWN"):
        r = res[s]
        cert = ""
        if s in ("UP", "DOWN") and r.get("n_paths", 0) >= 5:
            cert = " ==> CERTIFIED" if (r["p10"] >= BE and r["frac_clear_BE"] >= 0.80) else " ==> NOT certified"
        print(f"\n[{s}] {r}{cert}", flush=True)
    json.dump(res, open("m30_cpcv_side_result.json", "w"), indent=1)
    print(f"\n[cpcv30-side] DONE {time.time()-t0:.0f}s -> m30_cpcv_side_result.json", flush=True)


if __name__ == "__main__":
    main()
