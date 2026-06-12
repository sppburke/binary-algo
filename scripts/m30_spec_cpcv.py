"""Pipeline step (c): PURPOSE-BUILT UP/DOWN meta-labeler SPECIALIST @MX_HOR=30, NESTED-REFIT CPCV.

Faithful retarget of min2_spec_cpcv.py to 30m. Per purged-combinatorial fold: refit the cross-pair primary on
fit-rows; fit a meta-LGB on within-fold VAL predicting P(primary correct) from ORTHOGONAL axes (agreement /
dispersion / comp60 / order-flow + confidence); tune (meta-q, 1h-compression-q, NY) per side on within-fold VAL
ties-strict side-win; apply to held-out test block; ties-strict per-side win. This is the side-specialist test
the goal's step (c) requires. CERTIFY a side iff p10>=0.541 AND >=80% paths clear.

INCUMBENT (symmetric cross-pair, m30_xpair_cpcv_result.json): UP p10 .5588 / DOWN p10 .5525.
PRE-REGISTERED FALSIFIER: the specialist IMPROVES a side iff certified AND p10 > incumbent; else SUBSUMED
(prior: subset/meta-gate training kills ranking — 2m specialist collapsed 0/15 under nested refit; 30m FI shows
the edge is POOLED base-feature training, so a meta-gated subset should not beat the symmetric pooled book).
Settlement: book-native, moved-only, ties LOSE/strict-win, nonoverlap 1800s.

Usage: python m30_spec_cpcv.py
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
HOR = 30; GAP_S = HOR * 60; BE = 0.541; MIN_SIDE_N = 20
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
    Xs, fws, tss, bbs, nys = [], [], [], [], []
    for y in YEARS:
        F = MX.build_xp([str(y)], stride=2)
        if F is None or len(F) == 0:
            continue
        F = MX.augment(F, [str(y)], "xpof")
        for c in cols:
            if c not in F.columns:
                F[c] = np.nan
        Xs.append(F[cols].values.astype("float32"))
        fws.append(F["_fwd"].astype("float32").values)
        tss.append(F["_ts"].values.astype("int64"))
        bbs.append(F[GATE_FEAT].values.astype("float32"))
        nys.append((F["sess_ny"].values.astype("float32") > 0.5))
        del F
    X = np.concatenate(Xs); fwd = np.concatenate(fws); ts = np.concatenate(tss)
    bb = np.concatenate(bbs); ny = np.concatenate(nys)
    o = np.argsort(ts, kind="mergesort")
    return X[o], fwd[o], ts[o], bb[o], ny[o], cols


def summ(a):
    a = np.array(a, float)
    if len(a) == 0:
        return {"n_paths": 0}
    return {"n_paths": int(len(a)), "mean": round(float(a.mean()), 4), "p10": round(float(np.percentile(a, 10)), 4),
            "min": round(float(a.min()), 4), "max": round(float(a.max()), 4),
            "frac_clear_BE": round(float((a >= BE).mean()), 3)}


def main():
    t0 = time.time()
    X, fwd, ts, bb, ny, cols = build_pooled()
    moved = fwd != 0.0; yfull = (fwd > 0).astype("int8"); n = len(fwd)
    meta_idx = np.array([i for i, c in enumerate(cols)
                         if c.startswith("agree") or c.startswith("disp") or c == "comp60" or c.startswith("OF_")])
    print(f"[spec30] pooled n={n:,} meta_feats={len(meta_idx)} incumbent={INCUMBENT} load {time.time()-t0:.0f}s", flush=True)
    edges = np.linspace(0, n, N_GROUPS + 1).astype(int)
    grp = [(edges[i], edges[i + 1]) for i in range(N_GROUPS)]
    embargo = HOR * 60; purge = HOR * 60 + embargo
    accs_u, accs_d = [], []
    for pi, testg in enumerate(combinations(range(N_GROUPS), K_TEST)):
        te = np.zeros(n, bool)
        for gi in testg:
            te[grp[gi][0]:grp[gi][1]] = True
        te_lo = min(ts[grp[gi][0]] for gi in testg); te_hi = max(ts[grp[gi][1] - 1] for gi in testg)
        tr = ~te
        for gi in testg:
            a, b = ts[grp[gi][0]], ts[grp[gi][1] - 1]
            tr &= ~((ts >= a - purge) & (ts <= b + embargo))
        tri = np.where(tr & moved)[0]
        if len(tri) < 5000:
            continue
        cut = tri[int(len(tri) * 0.8)]; ct = ts[cut]
        fit = tri[ts[tri] < ct]; val = tri[ts[tri] >= ct]
        if len(fit) > SUB_FIT:
            fit = fit[np.linspace(0, len(fit) - 1, SUB_FIT).astype(int)]
        P = lgb.LGBMClassifier(objective="binary", metric="auc", learning_rate=0.03, num_leaves=127,
                               min_child_samples=400, subsample=0.8, subsample_freq=1, colsample_bytree=0.5,
                               reg_lambda=20, n_estimators=800, n_jobs=20, verbosity=-1).fit(X[fit], yfull[fit])
        pv = P.predict_proba(X[val])[:, 1]; pt = P.predict_proba(X[te])[:, 1]
        def metaX(rows, p):
            conf = np.abs(p - 0.5).astype("float32")
            return np.column_stack([conf, X[rows][:, meta_idx]])
        yv_corr = ((pv > 0.5).astype(int) == yfull[val]).astype(int)
        M = lgb.LGBMClassifier(objective="binary", metric="auc", learning_rate=0.02, num_leaves=15,
                               min_child_samples=1000, subsample=0.8, subsample_freq=1, colsample_bytree=0.6,
                               reg_lambda=20, n_estimators=400, n_jobs=20, verbosity=-1).fit(metaX(val, pv), yv_corr)
        smv = M.predict_proba(metaX(val, pv))[:, 1]; smt = M.predict_proba(metaX(te, pt))[:, 1]
        predv = (pv > 0.5).astype(int)
        for side, sval, store in (("UP", 1, accs_u), ("DOWN", 0, accs_d)):
            best = None
            for useny in (True, False):
                nyv = ny[val] if useny else np.ones(val.shape[0], bool)
                for cq in (33, 100):
                    bthr = np.nanpercentile(bb[fit], cq)
                    for mq in (0.80, 0.90):
                        mthr = np.quantile(smv, mq)
                        sel = (predv == sval) & nyv & (bb[val] <= bthr) & (smv >= mthr)
                        if sel.sum() < 80:
                            continue
                        fvv = fwd[val][sel]
                        win = ((sval == 1) & (fvv > 0)) | ((sval == 0) & (fvv < 0))
                        acc = win.mean()
                        if best is None or acc > best[0]:
                            best = (acc, useny, float(bthr), float(mthr))
            if best is None:
                continue
            _, useny, bthr, mthr = best
            predt = (pt > 0.5).astype(int); nyt = ny[te] if useny else np.ones(te.sum(), bool)
            g = (predt == sval) & nyt & (bb[te] <= bthr) & (smt >= mthr)
            selA = MX.nonoverlap_chrono(ts[te], g, GAP_S)
            if len(selA) < MIN_SIDE_N:
                continue
            fvt = fwd[te][selA]
            win = (((sval == 1) & (fvt > 0)) | ((sval == 0) & (fvt < 0))).astype(float)
            store.append(win.mean())
            yl = int(str(np.datetime64(int(te_lo), "s"))[:4]); yh = int(str(np.datetime64(int(te_hi), "s"))[:4])
            print(f"  path{pi:>2} era={yl}-{yh} {side} n={len(selA)} acc={win.mean():.4f} ny{int(useny)} {time.time()-t0:.0f}s", flush=True)
    res = {"model": "UP/DOWN meta-labeler specialist @MX_HOR=30, NESTED-REFIT CPCV (primary+meta refit per fold)",
           "breakeven": BE, "incumbent_xpair": INCUMBENT, "UP": summ(accs_u), "DOWN": summ(accs_d),
           "CERT_RULE": "side IMPROVES iff p10>=.541 & frac>=.80 AND p10>incumbent; else SUBSUMED"}
    for s in ("UP", "DOWN"):
        r = res[s]; tag = ""
        if r.get("n_paths", 0) >= 5:
            cert = r["p10"] >= BE and r["frac_clear_BE"] >= 0.80
            impr = cert and r["p10"] > INCUMBENT[s]
            tag = f" ==> {'CERTIFIED' if cert else 'NOT cert'}; {'IMPROVES' if impr else 'SUBSUMED (<= incumbent '+str(INCUMBENT[s])+')'}"
        print(f"[{s}] {r}{tag}", flush=True)
    json.dump(res, open("m30_spec_cpcv_result.json", "w"), indent=1)
    print(f"[spec30] DONE {time.time()-t0:.0f}s -> m30_spec_cpcv_result.json", flush=True)


if __name__ == "__main__":
    main()
