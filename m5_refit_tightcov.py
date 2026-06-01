"""Tight-COV confirmation of the (5m,UP) full-refit CPCV. The cov0.30 refit gave NOT-CERTIFIED (p10 0.534, 54%
of paths). The book operates at a TIGHTER gate (~top 5-10% by meta), so this re-runs the per-fold refit and
evaluates UP-selective accuracy at MULTIPLE covers {0.30,0.15,0.10,0.05} in ONE build, to settle whether the
book's actual operating point survives refit or also deflates (15m precedent: tight-cov selacc MAX was only
0.559 across 28 paths -> tighter cover never rescued it). CERTIFY a cov iff path_p10>=0.541 AND >=80% paths clear.
Reuses m5_cpcv_refit.build_frugal/mk_lgb (no re-implementation)."""
import json, time, itertools, numpy as np
from sklearn.metrics import roc_auc_score
import m5_cpcv_refit as RF
import m5_xpair_production as XP

COVS = [0.30, 0.15, 0.10, 0.05]
N_GROUPS, K_TEST = 8, 2


def main():
    t0 = time.time()
    p = json.load(open(XP.art("strategy.json"))); cols = p["primary_feats"]
    X, y, ts, ny = RF.build_frugal(cols)
    order = np.argsort(ts, kind="stable"); X, y, ts, ny = X[order], y[order], ts[order], ny[order]
    n = len(y); print(f"[tightcov] pooled n={n:,} stride={RF.STRIDE} build {time.time()-t0:.0f}s", flush=True)
    edges = np.linspace(0, n, N_GROUPS + 1).astype(int); g = np.zeros(n, np.int8)
    for k in range(N_GROUPS): g[edges[k]:edges[k+1]] = k
    rng = np.random.default_rng(7); HORS = EMB = 300; all_idx = np.arange(n)
    acc = {c: [] for c in COVS}; nsel = {c: [] for c in COVS}; aucs = []
    for ci, combo in enumerate(itertools.combinations(range(N_GROUPS), K_TEST)):
        te = all_idx[np.isin(g, combo)]; tr = all_idx[~np.isin(g, combo)]
        keep = np.ones(len(tr), bool); tt = ts[tr]
        for grp in combo:
            gi = all_idx[g == grp]; lo, hi = ts[gi[0]], ts[gi[-1]]
            keep &= ~((tt >= lo - HORS) & (tt <= hi + EMB))
        tr = tr[keep]
        if len(tr) > RF.SUBSAMPLE: tr = np.sort(rng.choice(tr, RF.SUBSAMPLE, replace=False))
        m = RF.mk_lgb(); m.fit(X[tr], y[tr]); pr = m.predict_proba(X[te])[:, 1]; yte = y[te]
        aucs.append(roc_auc_score(yte, pr))
        gate = ny[te] & (pr > 0.5); conf = np.abs(pr - 0.5)
        for c in COVS:
            if gate.sum() >= 40:
                cthr = np.quantile(conf[gate], 1 - c); sel = gate & (conf >= cthr)
                if sel.sum() >= 20:
                    acc[c].append(float((yte[sel] == 1).mean())); nsel[c].append(int(sel.sum())); continue
            acc[c].append(np.nan); nsel[c].append(0)
        del m, pr
        if ci % 7 == 0: print(f"  [tightcov] path {ci+1}/28 AUC={aucs[-1]:.4f} up@.10={acc[0.10][-1]} ({time.time()-t0:.0f}s)", flush=True)
    out = {"test": "(5m,UP) full-refit CPCV at multiple covers", "breakeven": 0.541, "stride": RF.STRIDE,
           "auc_mean": round(float(np.mean(aucs)), 4), "per_cov": {}}
    for c in COVS:
        a = np.array(acc[c], float); ns = np.array(nsel[c]); valid = a[np.isfinite(a) & (ns >= 20)]
        if len(valid) == 0:
            out["per_cov"][str(c)] = {"n_valid": 0}; continue
        p10 = float(np.percentile(valid, 10)); frac = float((valid >= 0.541).mean())
        out["per_cov"][str(c)] = {"n_valid_paths": int(len(valid)), "mean": round(float(valid.mean()), 4),
                                  "p10": round(p10, 4), "min": round(float(valid.min()), 4), "max": round(float(valid.max()), 4),
                                  "frac_clear_0.541": round(frac, 3), "med_n": int(np.median(ns[ns >= 20])),
                                  "CERTIFIED": bool(p10 >= 0.541 and frac >= 0.80)}
    any_cert = any(v.get("CERTIFIED") for v in out["per_cov"].values())
    out["verdict"] = ("CERTIFIED at >=1 cover under refit" if any_cert else
                      "NOT CERTIFIED at ANY cover under per-fold refit -> (5m,UP) deflates like 15m; regime-dependent, "
                      "positive only on the forward split, not robust across arbitrary folds.")
    json.dump(out, open("m5_refit_tightcov_result.json", "w"), indent=1)
    for c, v in out["per_cov"].items():
        print(f"  cov {c}: {v}", flush=True)
    print(f"[tightcov] VERDICT: {out['verdict']}", flush=True)


if __name__ == "__main__":
    main()
