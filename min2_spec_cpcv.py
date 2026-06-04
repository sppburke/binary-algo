"""NESTED-REFIT CPCV of the UP meta-labeler SPECIALIST @MX_HOR=2 — adversarial verification of the ONLY frozen-positive
this session (min2_spec UP: held-out .575/.5525/.6009, "SURVIVES"). The probe's frozen bb-gate showed .61 and DISSOLVED
under per-fold refit (cross-pair refit-CPCV UP p10 .5096); the meta-gate is a RICHER gate (adds agreement/dispersion/OF),
NOT covered by that gate scan, so it needs its OWN refit test. Mirrors m5_aci_cpcv (nested refit: primary AND meta-gate
both refit per purged-combinatorial fold). If the frozen survival is an overfit-gate artifact it collapses here.

Per fold: refit primary lgb on fit-rows (moved); fit meta lgb on within-fold VAL (target = primary correct, side-rows);
select (meta-q, compression-q, cov, NY) by within-fold VAL UP-acc; apply to held-out test block; ties-strict UP win.
CERTIFY UP iff p10>=0.541 AND >=80% paths clear. Breakeven 0.541. Reuses min2_xpair_cpcv.build_pooled (one heavy build).
Usage: python min2_spec_cpcv.py
"""
import os
os.environ["MX_HOR"] = "2"
import json, time, numpy as np
from itertools import combinations
import lightgbm as lgb
import m5_xpair as MX
import min2_xpair_cpcv as XC

N_GROUPS, K_TEST = XC.N_GROUPS, XC.K_TEST
SUB_FIT = XC.SUB_FIT
HOR = 2; GAP_S = 120; BE = 0.541; MIN_SIDE_N = 25


def summ(a):
    a = np.array(a, float)
    if len(a) == 0:
        return {"n_paths": 0}
    return {"n_paths": int(len(a)), "mean": round(float(a.mean()), 4), "p10": round(float(np.percentile(a, 10)), 4),
            "min": round(float(a.min()), 4), "max": round(float(a.max()), 4),
            "frac_clear_BE": round(float((a >= BE).mean()), 3)}


def main():
    t0 = time.time()
    X, fwd, ts, bb, ny, cols = XC.build_pooled()
    moved = fwd != 0.0; yfull = (fwd > 0).astype("int8"); n = len(fwd)
    meta_idx = np.array([i for i, c in enumerate(cols)
                         if c.startswith("agree") or c.startswith("disp") or c == "comp60" or c.startswith("OF_")])
    print(f"[speccpcv] pooled n={n:,} meta_feats={len(meta_idx)} load {time.time()-t0:.0f}s", flush=True)
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
        # meta target on VAL moved bars: primary correct?
        yv_corr = ((pv > 0.5).astype(int) == yfull[val]).astype(int)
        M = lgb.LGBMClassifier(objective="binary", metric="auc", learning_rate=0.02, num_leaves=15,
                               min_child_samples=1000, subsample=0.8, subsample_freq=1, colsample_bytree=0.6,
                               reg_lambda=20, n_estimators=400, n_jobs=20, verbosity=-1).fit(metaX(val, pv), yv_corr)
        smv = M.predict_proba(metaX(val, pv))[:, 1]; smt = M.predict_proba(metaX(te, pt))[:, 1]
        for side, sval, store in (("UP", 1, accs_u), ("DOWN", 0, accs_d)):
            # tune (meta-q, comp-q, cov, NY) on within-fold VAL UP/DOWN acc, ties-strict
            best = None
            predv = (pv > 0.5).astype(int)
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
            if side == "UP":
                yl = int(str(np.datetime64(int(te_lo), "s"))[:4]); yh = int(str(np.datetime64(int(te_hi), "s"))[:4])
                print(f"  path{pi:>2} era={yl}-{yh} UP n={len(selA)} acc={win.mean():.4f} ny{int(useny)} {time.time()-t0:.0f}s", flush=True)
    res = {"model": "UP/DOWN meta-labeler specialist @MX_HOR=2, NESTED-REFIT CPCV (primary+meta refit per fold)",
           "breakeven": BE, "UP": summ(accs_u), "DOWN": summ(accs_d),
           "CERT_RULE": "side CERTIFIED iff p10>=.541 & frac_clear>=.80"}
    for s in ("UP", "DOWN"):
        r = res[s]; tag = ""
        if r.get("n_paths", 0) >= 5:
            cert = r["p10"] >= BE and r["frac_clear_BE"] >= 0.80
            tag = f" ==> {'CERTIFIED' if cert else 'NOT certified'}"
        print(f"[{s}] {r}{tag}", flush=True)
    json.dump(res, open("min2_spec_cpcv_result.json", "w"), indent=1)
    print(f"[speccpcv] DONE {time.time()-t0:.0f}s -> min2_spec_cpcv_result.json", flush=True)


if __name__ == "__main__":
    main()
