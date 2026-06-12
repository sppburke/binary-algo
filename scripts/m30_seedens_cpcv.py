"""IMPROVE-lever I-seedens: SEED-ENSEMBLE of the certified 30m cross-pair primary (variance-reduction).

Identical to m30_xpair_cpcv.py EXCEPT the per-path primary is the AVERAGE of K LGBs with random_state/bagging_seed/
feature_fraction_seed = 0..K-1 (the subsample=0.8/colsample=0.5 randomness is seed-dependent -> averaging shrinks
run-to-run tail variance). MECHANISM: if the cross-pair p10 (15/15 but min .5475 UP / .5491 DOWN) is limited by
seed/tail variance rather than signal, averaging K seeds LIFTS the 10th-percentile path. Reports ENS (K-avg) vs S1
(single-seed control) per side.

INCUMBENT (m30_xpair_cpcv_result.json): UP p10 .5588 / DOWN p10 .5525, 15/15.
PRE-REGISTERED FALSIFIER: IMPROVES a side iff certified AND ENS p10 > incumbent p10 for that side; else SUBSUMED
(tail variance is not the binding limit — matches the 5m seedens null + the 10m/15m '11 levers killed/subsumed').
Settlement: book-native, moved-only (_y), ties LOSE, nonoverlap 1800s.

Usage: M30_K=4 python m30_seedens_cpcv.py
"""
import os
os.environ["MX_HOR"] = "30"
import json, time, numpy as np
from itertools import combinations
import lightgbm as lgb
import m5_xpair as MX

YEARS = list(range(2012, 2027))
N_GROUPS, K_TEST = 6, 2
SUB_FIT = 150_000
HOR = 30
GAP_S = HOR * 60
BE = 0.541
MIN_SIDE_N = 10
GATE_FEAT = "1h_bb_width"
K = int(os.environ.get("M30_K", "4"))

try:
    _inc = json.load(open("m30_xpair_cpcv_result.json"))
    INCUMBENT = {"UP": _inc["UP"]["p10"], "DOWN": _inc["DOWN"]["p10"]}
except Exception:
    INCUMBENT = {"UP": 0.5588, "DOWN": 0.5525}


def mk_lgb_seed(seed):
    return lgb.LGBMClassifier(objective="binary", metric="auc", learning_rate=0.03, num_leaves=127,
                             min_child_samples=400, subsample=0.8, subsample_freq=1, colsample_bytree=0.5,
                             reg_lambda=20, n_estimators=800, n_jobs=20, verbosity=-1,
                             random_state=seed, bagging_seed=seed, feature_fraction_seed=seed)


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


def summ(a):
    a = np.array(a, float)
    if len(a) == 0:
        return {"n_paths": 0}
    return {"n_paths": int(len(a)), "mean": round(float(a.mean()), 4), "p10": round(float(np.percentile(a, 10)), 4),
            "min": round(float(a.min()), 4), "max": round(float(a.max()), 4),
            "frac_clear_BE": round(float((a >= BE).mean()), 3)}


def side_split(pt, tst, bbw_te, ny_te, yte, bthr, cthr):
    gt = (bbw_te <= bthr) & ny_te & (np.abs(pt - 0.5) >= cthr)
    sel = MX.nonoverlap_chrono(tst, gt, GAP_S)
    if len(sel) < 10:
        return None
    pred = (pt[sel] > 0.5).astype(int); yt = yte[sel]
    out = {"comb": float((pred == yt).mean())}
    su = pred == 1; sd = pred == 0
    out["UP"] = float((pred[su] == yt[su]).mean()) if su.sum() >= MIN_SIDE_N else None
    out["DOWN"] = float((pred[sd] == yt[sd]).mean()) if sd.sum() >= MIN_SIDE_N else None
    return out


def main():
    t0 = time.time()
    X, y, ts, bbw, ny, cols = build_pooled()
    n = len(y); print(f"[seedens30] pooled n={n:,} feats={len(cols)} K={K} up-rate={y.mean():.4f} incumbent={INCUMBENT} load {time.time()-t0:.0f}s", flush=True)
    edges = np.linspace(0, n, N_GROUPS + 1).astype(int)
    grp = [(edges[i], edges[i + 1]) for i in range(N_GROUPS)]
    embargo = HOR * 60; purge = HOR * 60 + embargo
    R = {"ENS": {"comb": [], "UP": [], "DOWN": []}, "S1": {"comb": [], "UP": [], "DOWN": []}}
    for pi, testg in enumerate(combinations(range(N_GROUPS), K_TEST)):
        te = np.zeros(n, bool)
        for gi in testg:
            te[grp[gi][0]:grp[gi][1]] = True
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
        prs_v, prs_t = [], []
        for s in range(K):
            m = mk_lgb_seed(s).fit(X[fit], y[fit])
            prs_v.append(m.predict_proba(X[val])[:, 1]); prs_t.append(m.predict_proba(X[te])[:, 1]); del m
        prs_v = np.array(prs_v); prs_t = np.array(prs_t)
        tst = ts[te]; bbw_te = bbw[te]; ny_te = ny[te]; yte = y[te]
        for tag, pv, pt in (("ENS", prs_v.mean(0), prs_t.mean(0)), ("S1", prs_v[0], prs_t[0])):
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
                        best = (acc, bthr, cthr)
            if best is None:
                continue
            _, bthr, cthr = best
            ss = side_split(pt, tst, bbw_te, ny_te, yte, bthr, cthr)
            if ss is None:
                continue
            R[tag]["comb"].append(ss["comb"])
            if ss["UP"] is not None:
                R[tag]["UP"].append(ss["UP"])
            if ss["DOWN"] is not None:
                R[tag]["DOWN"].append(ss["DOWN"])
        if pi % 5 == 0:
            print(f"  path{pi} ENS comb={R['ENS']['comb'][-1] if R['ENS']['comb'] else None} S1 comb={R['S1']['comb'][-1] if R['S1']['comb'] else None} {time.time()-t0:.0f}s", flush=True)
    res = {"model": f"cross-pair xpof @MX_HOR=30 SEED-ENSEMBLE K={K} vs single-seed", "K": K, "incumbent_xpair": INCUMBENT,
           "ENS": {s: summ(R["ENS"][s]) for s in ("comb", "UP", "DOWN")},
           "S1": {s: summ(R["S1"][s]) for s in ("comb", "UP", "DOWN")},
           "breakeven": BE, "CERT_RULE": "IMPROVES a side iff ENS certified AND ENS p10 > incumbent p10; else SUBSUMED"}
    for s in ("UP", "DOWN"):
        e = res["ENS"][s]
        if e.get("n_paths", 0) >= 5:
            cert = e["p10"] >= BE and e["frac_clear_BE"] >= 0.80
            impr = e["p10"] > INCUMBENT[s]
            res[f"verdict_{s}"] = f"ENS p10 {e['p10']} vs incumbent {INCUMBENT[s]} -> {'IMPROVES' if (cert and impr) else 'SUBSUMED'}"
    for tag in ("ENS", "S1"):
        for s in ("comb", "UP", "DOWN"):
            print(f"  [{tag}] {s}: {res[tag][s]}", flush=True)
    for s in ("UP", "DOWN"):
        print(f"  {res.get('verdict_'+s,'')}", flush=True)
    json.dump(res, open("m30_seedens_cpcv_result.json", "w"), indent=1)
    print(f"[seedens30] DONE {time.time()-t0:.0f}s -> m30_seedens_cpcv_result.json", flush=True)


if __name__ == "__main__":
    main()
