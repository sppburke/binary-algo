"""C3: NESTED-REFIT CPCV of a SEED-ENSEMBLED cross-pair primary (UP side, at the certified primary-confidence
cover). The deployed primary is trained multithreaded with NO estimator seed (MODEL_REGISTRY note) -> run-to-run
tail variance; the cov0.05 cert showed p10 0.553 but the cov0.30 refit p10 was 0.534 vs mean 0.542 (a tail-variance
gap, DL-review #1). Hypothesis: averaging K seed-varied primaries per fold SHRINKS that tail -> LIFTS the cov0.05
path_p10 above the single-model 0.553. Per purged-combinatorial path: fit K primaries (random_state 0..K-1),
average predict_proba, evaluate UP-selective accuracy at covs {0.05,0.10,0.15}. CERTIFY-improvement iff seed-ens
UP cov0.05 p10 > single-model .553 AND >=80% paths clear. (Falsifier: KILL the seed-ens lever unless p10 rises
above .553.) Reuses RF.build_frugal + a seeded mk_lgb. Settlement: book-native, moved-only (_y), ties LOSE.

Usage: M5_STRIDE=6 M5_K=4 python m5_seedens_cpcv.py
"""
import os, json, time, itertools, numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import m5_xpair as MX
import m5_xpair_production as XP
import m5_cpcv_refit as RF

N_GROUPS, K_TEST = 8, 2
STRIDE = int(os.environ.get("M5_STRIDE", "6"))
K = int(os.environ.get("M5_K", "4"))
COVS = [0.05, 0.10, 0.15]
SUBSAMPLE = 100_000


def mk_lgb_seed(seed, n_est=700):
    return lgb.LGBMClassifier(objective="binary", metric="auc", learning_rate=0.03, num_leaves=127,
        min_child_samples=300, subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=20,
        n_estimators=n_est, n_jobs=20, verbosity=-1, random_state=seed,
        bagging_seed=seed, feature_fraction_seed=seed)


def summ(a_list, n_list):
    a = np.array(a_list, float); nn = np.array(n_list); v = a[np.isfinite(a) & (nn >= 20)]
    if len(v) == 0:
        return {"n_valid": 0}
    p10 = float(np.percentile(v, 10)); frac = float((v >= 0.541).mean())
    return {"n_valid": int(len(v)), "mean": round(float(v.mean()), 4), "p10": round(p10, 4),
            "min": round(float(v.min()), 4), "max": round(float(v.max()), 4),
            "frac_clear_0.541": round(frac, 3), "med_n": int(np.median(nn[nn >= 20])),
            "CERTIFIED": bool(p10 >= 0.541 and frac >= 0.80)}


def main():
    t0 = time.time()
    p = json.load(open(XP.art("strategy.json"))); cols = p["primary_feats"]
    X, y, ts, ny = RF.build_frugal(cols)
    order = np.argsort(ts, kind="stable"); X, y, ts, ny = X[order], y[order], ts[order], ny[order]
    n = len(y)
    print(f"[seedens] pooled n={n:,} stride={STRIDE} K={K} feats={len(cols)} build {time.time()-t0:.0f}s", flush=True)
    edges = np.linspace(0, n, N_GROUPS + 1).astype(int); g = np.zeros(n, np.int8)
    for k in range(N_GROUPS):
        g[edges[k]:edges[k+1]] = k
    rng = np.random.default_rng(7); HORS = EMB = 300; all_idx = np.arange(n)
    acc = {("ENS", c): [] for c in COVS}; nsel = {("ENS", c): [] for c in COVS}
    acc1 = {("S1", c): [] for c in COVS}; nsel1 = {("S1", c): [] for c in COVS}   # single-seed baseline (control)
    aucs_e, aucs_1 = [], []
    for ci, combo in enumerate(itertools.combinations(range(N_GROUPS), K_TEST)):
        te = all_idx[np.isin(g, combo)]; tr = all_idx[~np.isin(g, combo)]
        keep = np.ones(len(tr), bool); tt = ts[tr]
        for grp in combo:
            gi = all_idx[g == grp]; lo, hi = ts[gi[0]], ts[gi[-1]]
            keep &= ~((tt >= lo - HORS) & (tt <= hi + EMB))
        tr = tr[keep]
        if len(tr) > SUBSAMPLE:
            tr = np.sort(rng.choice(tr, SUBSAMPLE, replace=False))
        prs = []
        for s in range(K):
            m = mk_lgb_seed(s); m.fit(X[tr], y[tr]); prs.append(m.predict_proba(X[te])[:, 1]); del m
        prs = np.array(prs)
        yte = y[te]
        for tag, pr, au in (("ENS", prs.mean(0), aucs_e), ("S1", prs[0], aucs_1)):
            au.append(roc_auc_score(yte, pr)); conf = np.abs(pr - 0.5); gate = ny[te] & (pr > 0.5)
            A = acc if tag == "ENS" else acc1; Nn = nsel if tag == "ENS" else nsel1
            for c in COVS:
                if gate.sum() >= 40:
                    cthr = np.quantile(conf[gate], 1 - c); sel = gate & (conf >= cthr)
                    if sel.sum() >= 20:
                        A[(tag, c)].append(float((yte[sel] == 1).mean())); Nn[(tag, c)].append(int(sel.sum())); continue
                A[(tag, c)].append(np.nan); Nn[(tag, c)].append(0)
        if ci % 7 == 0:
            print(f"  [seedens] path {ci+1}/28 ENS@.05={acc[('ENS',0.05)][-1]} S1@.05={acc1[('S1',0.05)][-1]} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    out = {"test": f"(5m,UP) SEED-ENSEMBLE (K={K}) nested-refit CPCV vs single-seed control, primary-conf cover",
           "breakeven": 0.541, "stride": STRIDE, "K": K, "pooled_n": int(n),
           "auc_mean_ens": round(float(np.mean(aucs_e)), 4), "auc_mean_single": round(float(np.mean(aucs_1)), 4),
           "ENS": {f"cov{c}": summ(acc[("ENS", c)], nsel[("ENS", c)]) for c in COVS},
           "SINGLE": {f"cov{c}": summ(acc1[("S1", c)], nsel1[("S1", c)]) for c in COVS}}
    e05 = out["ENS"]["cov0.05"]; s05 = out["SINGLE"]["cov0.05"]
    improves = bool(e05.get("CERTIFIED") and e05.get("p10", 0) > 0.553)
    out["verdict"] = {"seed_ens_improves": improves,
        "statement": (f"Seed-ensemble (K={K}) UP cov0.05 p10 {e05.get('p10')} (frac {e05.get('frac_clear_0.541')}) "
                      f"vs single-seed {s05.get('p10')} and incumbent .553 -> "
                      + ("IMPROVES the certified floor." if improves else
                         "does NOT lift the floor above .553 (tail variance not the binding limit)."))}
    json.dump(out, open("m5_seedens_cpcv_result.json", "w"), indent=1)
    for k, v in out["ENS"].items():
        print(f"  ENS {k}: {v}", flush=True)
    for k, v in out["SINGLE"].items():
        print(f"  SINGLE {k}: {v}", flush=True)
    print(f"[seedens] VERDICT: {out['verdict']['statement']}", flush=True)


if __name__ == "__main__":
    main()
