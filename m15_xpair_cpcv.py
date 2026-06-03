"""PER-SIDE full-refit CPCV of the CROSS-PAIR primary @15m (A6c) — certifies whether the cross-pair
USD-residual+OF signal (m5xp source, MX_HOR=15) beats the certified base-book side floors (UP .5475 / DOWN .5486).

Mirrors m15_cpcv_side.py design (6 groups, C(6,2)=15 purged paths, purge+embargo = 1 horizon, per-fold REFIT,
gate tuned on a VAL split by max-VAL-acc, test trades split by predicted side) — but refits the SINGLE cross-pair
lgb primary (not the 3-model base ensemble). Gate = comp(15m_bb_width q)xNY (the base feature 15m_bb_width is in
the xpof frame). Memory-safe: per-year build, stride-2 pooled, float32, subsample fit. Single lgb at n_est=800.

CERTIFY a side iff per-side p10 >= 0.541 AND >= ~80% paths clear 0.541, AND it beats the base book's floor
(UP .5475 / DOWN .5486) to be a real improvement. Breakeven 0.541.
Usage: python m15_xpair_cpcv.py
"""
import os
os.environ["MX_HOR"] = "15"
import json, time, numpy as np
from itertools import combinations
import lightgbm as lgb
import m5_xpair as MX

YEARS = list(range(2012, 2027))
N_GROUPS, K_TEST = 6, 2
SUB_FIT = 150_000
HOR = 15
GAP_S = HOR * 60
BE = 0.541
MIN_SIDE_N = 10
BOOK_FLOOR = {"UP": 0.5475, "DOWN": 0.5486}


def build_pooled():
    # canonical columns from one full reference year
    ref = MX.augment(MX.build_xp([2020]), [2020], "xpof")
    cols = MX.feat_cols("xpof", ref, MX.xp_cols(ref))
    assert "15m_bb_width" in ref.columns and "sess_ny" in ref.columns, "gate cols missing"
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
        bbs.append(F["15m_bb_width"].values.astype("float32"))
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


def main():
    t0 = time.time()
    X, y, ts, bbw, ny, cols = build_pooled()
    n = len(y); print(f"[xpcpcv] pooled n={n:,} feats={len(cols)} up-rate={y.mean():.4f} load {time.time()-t0:.0f}s", flush=True)
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
        P = lgb.LGBMClassifier(objective="binary", metric="auc", learning_rate=0.03, num_leaves=127,
                               min_child_samples=400, subsample=0.8, subsample_freq=1, colsample_bytree=0.5,
                               reg_lambda=20, n_estimators=800, n_jobs=20, verbosity=-1).fit(X[fit], y[fit])
        pv = P.predict_proba(X[val])[:, 1]
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
        pt = P.predict_proba(X[te])[:, 1]; tst = ts[te]
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
    res = {"model": "cross-pair xpof primary @MX_HOR=15", "combined": summ(accs_c), "UP": summ(accs_u), "DOWN": summ(accs_d),
           "book_floor": BOOK_FLOOR, "breakeven": BE,
           "CERT_RULE": "side CERTIFIED iff p10>=.541 & frac>=.80; IMPROVEMENT iff p10 > book floor"}
    for s in ("combined", "UP", "DOWN"):
        r = res[s]
        tag = ""
        if s in ("UP", "DOWN") and r.get("n_paths", 0) >= 5:
            cert = r["p10"] >= BE and r["frac_clear_BE"] >= 0.80
            impr = r["p10"] > BOOK_FLOOR[s]
            tag = f" ==> {'CERTIFIED' if cert else 'NOT certified'}{' + IMPROVES book' if (cert and impr) else (' (<= book floor)' if cert else '')}"
        print(f"\n[{s}] {r}{tag}", flush=True)
    json.dump(res, open("m15_xpair_cpcv_result.json", "w"), indent=1)
    print(f"\n[xpcpcv] DONE {time.time()-t0:.0f}s -> m15_xpair_cpcv_result.json", flush=True)


if __name__ == "__main__":
    main()
