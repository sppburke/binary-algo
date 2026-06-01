"""GOLD-STANDARD full-refit CPCV for the certified (5m,UP) edge — the cpcv_certify.py analog that deflated the
15m claim 0.647->0.5455. Unlike m5_cpcv.py (path-robustness of the FROZEN book's predictions), this REFITS the
cross-pair direction PRIMARY on each purged-combinatorial train fold and evaluates the UP-side selective
accuracy on the held-out test fold — testing model-SELECTION overfitting, the risk the frozen-book CPCV can't see.

Design (Lopez de Prado): pooled cross-pair matrix 2012-2026 (build ONCE; stride to cap memory). N=8 chronological
groups, k=2 test -> C(8,2)=28 purged paths, purge+embargo = 1 label horizon (300s). Per path: refit lgb (m5xp
primary params, fewer trees for the 28x budget) on purged train (subsample<=100k), predict test, UP-side =
NY-gated & pred>0.5, selective = top-confidence cover; report acc per path. CERTIFY iff UP-selective path_p10
>= 0.541 AND >=80% of paths clear 0.541.  Settlement: book-native contiguous-300s, moved-only (_y), ties at build.

Usage: M5_STRIDE=3 python m5_cpcv_refit.py"""
import os, json, time, itertools, numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import m5_xpair as MX
import m5_xpair_production as XP

N_GROUPS, K_TEST = 8, 2
COV = 0.30                       # top-confidence cover among UP candidates (looser than book's ~0.05 for n)
SUBSAMPLE = 100_000
STRIDE = int(os.environ.get("M5_STRIDE", "6"))
ALL_YEARS = [str(y) for y in range(2012, 2027)]


def build_frugal(cols):
    """Build the pooled cross-pair matrix ONE YEAR AT A TIME into float32 arrays (build_xp processes years
    independently — no cross-year dependency — so this equals the all-years build at a fraction of peak RAM)."""
    import gc
    Xs, ys, tss, nys = [], [], [], []
    for yr in ALL_YEARS:
        D = MX.build_xp([yr], stride=STRIDE)
        if len(D) == 0:
            continue
        D = MX.augment(D, [yr], XP.MODE)
        Xs.append(D[cols].astype("float32").to_numpy())
        ys.append(D["_y"].astype(np.int8).values)
        tss.append(D["_ts"].values.astype("int64"))
        nys.append((D["sess_ny"].values > 0.5))
        del D; gc.collect()
    return (np.concatenate(Xs), np.concatenate(ys), np.concatenate(tss), np.concatenate(nys))


def mk_lgb(n_est=700):
    return lgb.LGBMClassifier(objective="binary", metric="auc", learning_rate=0.03, num_leaves=127,
        min_child_samples=300, subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=20,
        n_estimators=n_est, n_jobs=20, verbosity=-1)


def main():
    t0 = time.time()
    p = json.load(open(XP.art("strategy.json"))); cols = p["primary_feats"]
    X, y, ts, ny = build_frugal(cols)
    order = np.argsort(ts, kind="stable")
    X, y, ts, ny = X[order], y[order], ts[order], ny[order]
    n = len(y)
    print(f"[refit-cpcv] pooled n={n:,} stride={STRIDE} up-rate={y.mean():.4f} feats={len(cols)} build {time.time()-t0:.0f}s", flush=True)
    edges = np.linspace(0, n, N_GROUPS + 1).astype(int)
    g = np.zeros(n, np.int8)
    for k in range(N_GROUPS):
        g[edges[k]:edges[k+1]] = k
    rng = np.random.default_rng(7); HORS = 300; EMB = 300
    all_idx = np.arange(n)
    up_accs, up_ns, aucs, per = [], [], [], []
    for ci, combo in enumerate(itertools.combinations(range(N_GROUPS), K_TEST)):
        te_mask = np.isin(g, combo); te = all_idx[te_mask]; tr = all_idx[~te_mask]
        # purge+embargo around each contiguous test group
        keep = np.ones(len(tr), bool); tt = ts[tr]
        for grp in combo:
            gi = all_idx[g == grp]; lo, hi = ts[gi[0]], ts[gi[-1]]
            keep &= ~((tt >= lo - HORS) & (tt <= hi + EMB))
        tr = tr[keep]
        if len(tr) > SUBSAMPLE:
            tr = np.sort(rng.choice(tr, SUBSAMPLE, replace=False))
        m = mk_lgb(); m.fit(X[tr], y[tr])
        pr = m.predict_proba(X[te])[:, 1]; yte = y[te]
        auc = roc_auc_score(yte, pr); aucs.append(auc)
        # UP-side selective: NY & pred up, top-COV by confidence
        gate = ny[te] & (pr > 0.5); conf = np.abs(pr - 0.5)
        if gate.sum() >= 40:
            cthr = np.quantile(conf[gate], 1 - COV); sel = gate & (conf >= cthr)
            if sel.sum() >= 25:
                acc = float((yte[sel] == 1).mean()); up_accs.append(acc); up_ns.append(int(sel.sum()))
            else:
                up_accs.append(np.nan); up_ns.append(int(sel.sum()))
        else:
            up_accs.append(np.nan); up_ns.append(0)
        del m, pr
        if ci % 7 == 0:
            print(f"  [refit-cpcv] path {ci+1}/28 combo{combo} AUC={auc:.4f} UPsel={up_accs[-1]} n={up_ns[-1]} ({time.time()-t0:.0f}s)", flush=True)
    up = np.array(up_accs, float); ns = np.array(up_ns)
    valid = up[np.isfinite(up) & (ns >= 25)]
    aucs = np.array(aucs)
    p10 = float(np.percentile(valid, 10)) if len(valid) else float("nan")
    frac = float((valid >= 0.541).mean()) if len(valid) else float("nan")
    certified = bool(len(valid) and p10 >= 0.541 and frac >= 0.80)
    out = {"test": "(5m,UP) FULL-REFIT purged-combinatorial CPCV (cross-pair primary refit per path)",
           "breakeven": 0.541, "stride": STRIDE, "pooled_n": int(n), "cov": COV,
           "auc_mean": round(float(aucs.mean()), 4), "auc_p10": round(float(np.percentile(aucs, 10)), 4),
           "up_sel_n_valid_paths": int(len(valid)), "up_sel_mean": round(float(valid.mean()), 4) if len(valid) else None,
           "up_sel_p10": round(p10, 4), "up_sel_min": round(float(valid.min()), 4) if len(valid) else None,
           "up_sel_med_n": int(np.median(ns[ns >= 25])) if (ns >= 25).any() else 0,
           "frac_paths_clear_0.541": round(frac, 3),
           "all_up_sel": [None if not np.isfinite(s) else round(float(s), 4) for s in up],
           "verdict": {"CERTIFIED": certified,
                       "statement": (f"CERTIFIED under FULL REFIT: UP-selective path_p10 {p10:.4f}, {frac:.0%} of paths clear 0.541 "
                                     "-> the 5m UP edge survives per-path model refit (not a selection mirage; contrast 15m 0.647->0.5455)."
                                     if certified else
                                     f"NOT CERTIFIED under refit: UP-selective path_p10 {p10:.4f}, only {frac:.0%} clear 0.541.")}}
    json.dump(out, open("m5_cpcv_refit_result.json", "w"), indent=1)
    print(f"[refit-cpcv] AUC mean={aucs.mean():.4f} p10={np.percentile(aucs,10):.4f}", flush=True)
    print(f"[refit-cpcv] UP-sel: mean={valid.mean():.4f} p10={p10:.4f} min={valid.min():.4f} frac>=0.541={frac:.2f} (n_valid={len(valid)}, med_n={int(np.median(ns[ns>=25])) if (ns>=25).any() else 0})", flush=True)
    print(f"[refit-cpcv] VERDICT: {out['verdict']['statement']}", flush=True)


if __name__ == "__main__":
    main()
