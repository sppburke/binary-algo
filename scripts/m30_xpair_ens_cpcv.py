"""IMPROVE-lever I-ens: 3-MODEL ENSEMBLE (lgb+xgb+cat) ON THE POOLED CROSS-PAIR DATA — the untested best-combo.

The certified K2 keystone (m30_xpair_cpcv.py) used a SINGLE LightGBM on the pooled xpof matrix (lighter harness).
The base book (m30_cpcv_side.py) used a 3-model ensemble but on EURUSD-ONLY rows. The two effects were never
combined: ensembling (decorrelated model-class errors) AND pooling (more data + cross-sectional regularization).
This runs the program-standard 3-model ensemble (lgb 255/600 + xgb depth8/700 + cat depth8/700) ON the pooled
cross-pair xpof matrix, identical gate/CPCV/side-split as K2. MECHANISM: xgb/cat capture different feature
interactions than lgb; averaging across MODEL CLASSES (not just seeds — seed-ens was flat) shrinks the WORST
purged-refit paths, which is precisely the p10 + 2026-robustness limit.

INCUMBENT (single-LGB pool, m30_xpair_cpcv_result.json): UP p10 .5588 / DOWN p10 .5525, 15/15.
PRE-REGISTERED FALSIFIER: IMPROVES a side iff certified AND p10 > incumbent for that side; else SUBSUMED (model-class
diversity is not the binding limit). Settlement: book-native, moved-only (_y), ties LOSE, nonoverlap 1800s.

Usage: python m30_xpair_ens_cpcv.py
"""
import os
os.environ["MX_HOR"] = "30"
import json, time, numpy as np
from itertools import combinations
import lightgbm as lgb, xgboost as xgb
from catboost import CatBoostClassifier
import m5_xpair as MX

YEARS = list(range(2012, 2027))
N_GROUPS, K_TEST = 6, 2
SUB_FIT = 150_000
HOR = 30
GAP_S = HOR * 60
BE = 0.541
MIN_SIDE_N = 10
GATE_FEAT = "1h_bb_width"

try:
    _inc = json.load(open("m30_xpair_cpcv_result.json"))
    INCUMBENT = {"UP": _inc["UP"]["p10"], "DOWN": _inc["DOWN"]["p10"]}
except Exception:
    INCUMBENT = {"UP": 0.5588, "DOWN": 0.5525}


def build_pooled():
    ref = MX.augment(MX.build_xp([2020]), [2020], "xpof")
    cols = MX.feat_cols("xpof", ref, MX.xp_cols(ref))
    assert GATE_FEAT in ref.columns and "sess_ny" in ref.columns, "gate cols missing"
    del ref
    Xs, ys, tss, bbs, nys = [], [], [], [], []
    for y in YEARS:
        F = MX.build_xp([str(y)], stride=2)
        if F is None or len(F) == 0:
            continue
        F = MX.augment(F, [str(y)], "xpof")
        for c in cols:
            if c not in F.columns:
                F[c] = np.nan
        Xs.append(F[cols].values.astype("float32"))
        ys.append(F["_y"].astype("int8").values)
        tss.append(F["_ts"].values.astype("int64"))
        bbs.append(F[GATE_FEAT].values.astype("float32"))
        nys.append((F["sess_ny"].values.astype("float32") > 0.5))
        del F
    X = np.concatenate(Xs); y = np.concatenate(ys); ts = np.concatenate(tss)
    bbw = np.concatenate(bbs); ny = np.concatenate(nys)
    o = np.argsort(ts, kind="mergesort")
    return X[o], y[o], ts[o], bbw[o], ny[o], cols


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
    X, y, ts, bbw, ny, cols = build_pooled()
    n = len(y); print(f"[xpens30] pooled n={n:,} feats={len(cols)} up-rate={y.mean():.4f} gate={GATE_FEAT} "
                      f"incumbent={INCUMBENT} load {time.time()-t0:.0f}s", flush=True)
    edges = np.linspace(0, n, N_GROUPS + 1).astype(int)
    grp = [(edges[i], edges[i + 1]) for i in range(N_GROUPS)]
    embargo = HOR * 60; purge = HOR * 60 + embargo
    accs_c, accs_u, accs_d = [], [], []
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
            bthr = np.nanpercentile(bbw[fit], q); gmask = (bbw[val] <= bthr) & ny[val]
            if gmask.sum() < 100:
                continue
            for cov in (0.05, 0.10):
                cthr = np.quantile(np.abs(pv[gmask] - 0.5), 1 - cov); sel = gmask & (np.abs(pv - 0.5) >= cthr)
                if sel.sum() < 100:
                    continue
                acc = ((pv[sel] > 0.5).astype(int) == y[val][sel]).mean()
                if best is None or acc > best[0]:
                    best = (acc, bthr, cthr, q, cov)
        if best is None:
            continue
        _, bthr, cthr, q, cov = best
        pt = proba(L, G, C, X[te]); tst = ts[te]
        gt = (bbw[te] <= bthr) & ny[te] & (np.abs(pt - 0.5) >= cthr)
        sel = MX.nonoverlap_chrono(tst, gt, GAP_S)
        if len(sel) < 10:
            continue
        pred = (pt[sel] > 0.5).astype(int); yt = y[te][sel]
        corr = (pred == yt).astype(float); accs_c.append(corr.mean())
        su = pred == 1; sd = pred == 0
        au = ad = None
        if su.sum() >= MIN_SIDE_N:
            au = float((pred[su] == yt[su]).mean()); accs_u.append(au)
        if sd.sum() >= MIN_SIDE_N:
            ad = float((pred[sd] == yt[sd]).mean()); accs_d.append(ad)
        yl = int(str(np.datetime64(int(te_lo), "s"))[:4]); yh = int(str(np.datetime64(int(te_hi), "s"))[:4])
        print(f"  path{pi:>2} era={yl}-{yh} q{q} cov{cov:.0%} n={len(sel)} comb={corr.mean():.4f} "
              f"UP={('%.4f'%au) if au else '  -  '}(n{int(su.sum())}) DOWN={('%.4f'%ad) if ad else '  -  '}(n{int(sd.sum())}) {time.time()-t0:.0f}s", flush=True)
    res = {"model": "3-model ensemble (lgb+xgb+cat) on POOLED cross-pair xpof @MX_HOR=30", "gate_feat": GATE_FEAT,
           "combined": summ(accs_c), "UP": summ(accs_u), "DOWN": summ(accs_d),
           "incumbent_single_lgb": INCUMBENT, "breakeven": BE,
           "CERT_RULE": "IMPROVES a side iff certified AND p10 > single-LGB incumbent p10; else SUBSUMED",
           "all_paths": {"combined": accs_c, "UP": accs_u, "DOWN": accs_d}}
    for s in ("combined", "UP", "DOWN"):
        r = res[s]; tag = ""
        if s in ("UP", "DOWN") and r.get("n_paths", 0) >= 5:
            cert = r["p10"] >= BE and r["frac_clear_BE"] >= 0.80
            impr = cert and r["p10"] > INCUMBENT[s]
            tag = f" ==> {'CERTIFIED' if cert else 'NOT cert'}; {'IMPROVES incumbent (+%.4f)'%(r['p10']-INCUMBENT[s]) if impr else 'SUBSUMED (<= incumbent '+str(INCUMBENT[s])+')'}"
        print(f"\n[{s}] {r}{tag}", flush=True)
    json.dump(res, open("m30_xpair_ens_cpcv_result.json", "w"), indent=1)
    print(f"\n[xpens30] DONE {time.time()-t0:.0f}s -> m30_xpair_ens_cpcv_result.json", flush=True)


if __name__ == "__main__":
    main()
