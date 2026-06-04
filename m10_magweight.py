"""EDGE-IMPROVEMENT I3 @10m: |return|-weighted retrain of the cross-pair pipeline (m5_magweight retargeted to
MX_HOR=10). Hypothesis: large-|move| bars carry more predictable SIGN; up-weighting training toward them should
channel the strong 10m MAGNITUDE signal (magAUC .81/.74/.71) into a better DIRECTION edge — esp. on the WEAKER
DOWN side (certified cross-pair DOWN p10 .5683 is the lower of the two; DOWN moves are jump/informed-dominated so
the magnitude->direction bridge has the most headroom there). Same cross-pair features + meta-gate; only the
primary's sample_weight = (|fwd|/median)^POW changes.

INCUMBENT (cross-pair EURUSD.m10xp.v1): refit-CPCV UP p10 .5863 / DOWN p10 .5683 (15/15); forward cov10 2025
UP .605 / DOWN .574. PRE-REGISTERED FALSIFIER: KILL unless a side's binding-2025 forward win-rate >= incumbent
(UP .605 / DOWN .574) with healthy n AND up-rate tripwire in [.47,.53] AND it would plausibly lift the refit p10
(>.5863 UP / >.5683 DOWN). This is an IMPROVEMENT attempt — we want it to win, esp. DOWN."""
import os
os.environ["MX_HOR"] = "10"
import sys, json, time, numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import m5_xpair as MX
import m5_xpair_production as XP

MODE = XP.MODE; SPL = MX.SPL
POW = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5   # weight = (|fwd|/median)^POW, clipped


def magweight(fwd, pow_=POW):
    a = np.abs(fwd).astype(float); med = np.median(a[a > 0]) or 1e-9
    w = (a / med) ** pow_
    return np.clip(w, 0.1, 10.0)


def boot(c, nb=4000, seed=7):
    c = np.asarray(c, float)
    if len(c) < 5: return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed); n = len(c)
    return tuple(float(x) for x in np.percentile([c[rng.integers(0, n, n)].mean() for _ in range(nb)], [2.5, 97.5]))


def main():
    t0 = time.time()
    TR = MX.build_xp(SPL["train"], 4); VA = MX.build_xp(SPL["val"])
    xpc = MX.xp_cols(TR); TR = MX.augment(TR, SPL["train"], MODE); VA = MX.augment(VA, SPL["val"], MODE)
    cols = MX.feat_cols(MODE, TR, xpc)
    ytr = TR["_y"].astype(int).values; yva = VA["_y"].astype(int).values
    w_tr = magweight(TR["_fwd"].values)
    print(f"[magw10] POW={POW} train={len(TR):,} val={len(VA):,} feats={len(cols)} wt[min/med/max]={w_tr.min():.2f}/{np.median(w_tr):.2f}/{w_tr.max():.2f} {time.time()-t0:.0f}s", flush=True)
    P = lgb.LGBMClassifier(objective="binary", metric="auc", learning_rate=0.02, num_leaves=127, min_child_samples=400,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=20, n_estimators=3000, n_jobs=20, verbosity=-1)
    P.fit(TR[cols].astype("float32"), ytr, sample_weight=w_tr,
          eval_set=[(VA[cols].astype("float32"), yva)], eval_metric="auc",
          callbacks=[lgb.early_stopping(150), lgb.log_evaluation(0)])
    pva = P.predict_proba(VA[cols].astype("float32"))[:, 1]
    print(f"[magw10] primary VAL AUC={roc_auc_score(yva, pva):.4f} (xpair incumbent ~0.526) {time.time()-t0:.0f}s", flush=True)
    mcols = XP.meta_feats(VA); vny = VA["sess_ny"].values > 0.5
    Xm = XP._Xmeta(VA, pva, mcols); ycorr = ((pva > 0.5).astype(int) == yva).astype(int)
    M = lgb.LGBMClassifier(objective="binary", metric="auc", learning_rate=0.02, num_leaves=15, min_child_samples=1000,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.6, reg_lambda=20, n_estimators=400, n_jobs=20, verbosity=-1)
    M.fit(Xm[vny], ycorr[vny]); sval = M.predict_proba(Xm)[:, 1]
    tsv = VA["_ts"].values.astype("int64"); vyr = XP.yr(tsv)

    def half_acc(thr):
        accs = []
        for yy in sorted(set(vyr.tolist())):
            m = (VA["sess_ny"].values > 0.5) & (vyr == yy) & (sval >= thr); sel = MX.nonoverlap_chrono(tsv, m)
            if len(sel) < 40: continue
            accs.append(((pva[sel] > 0.5).astype(int) == yva[sel]).mean())
        return min(accs) if len(accs) == len(set(vyr.tolist())) else float("nan")
    best = None
    for q in (0.80, 0.85, 0.88, 0.90, 0.92, 0.94, 0.95):
        thr = float(np.quantile(sval[vny], q)); hm = half_acc(thr)
        nn = len(MX.nonoverlap_chrono(tsv, vny & (sval >= thr)))
        if nn < 200 or np.isnan(hm): continue
        if best is None or hm > best[1]: best = (thr, hm, q)
    THR, HM, Q = best
    print(f"[magw10] FROZEN meta_thr={THR:.4f} (VAL q{Q}) worst-half={HM:.3f}", flush=True)

    res = {}
    for w, label in (("test24", "2024"), ("test25", "2025"), ("oos", "2026")):
        D = MX.build_xp(SPL[w]); D = MX.augment(D, SPL[w], MODE)
        pr = P.predict_proba(D[cols].astype("float32"))[:, 1]; y = D["_y"].astype(int).values
        sm = M.predict_proba(XP._Xmeta(D, pr, mcols))[:, 1]; ts = D["_ts"].values.astype("int64")
        ny = D["sess_ny"].values > 0.5
        sel = MX.nonoverlap_chrono(ts, ny & (sm >= THR)); pred = (pr[sel] > 0.5).astype(int); yy = y[sel]
        for side, nm in ((1, "UP"), (0, "DOWN")):
            ssel = pred == side
            if ssel.sum() < 5: res[f"{label}_{nm}"] = {"n": int(ssel.sum()), "acc": None}; continue
            corr = (pred[ssel] == yy[ssel]).astype(float); lo, hi = boot(corr)
            res[f"{label}_{nm}"] = {"n": int(ssel.sum()), "acc": round(float(corr.mean()), 4), "ci": [round(lo, 4), round(hi, 4)],
                                    "up_rate": round(float(yy[ssel].mean()), 4)}
        res[f"{label}_uprate_all"] = round(float(yy.mean()), 4)
        del D
    inc = {"UP": {"2024": 0.642, "2025": 0.605, "2026": 0.680}, "DOWN": {"2024": 0.599, "2025": 0.574, "2026": 0.476}}
    up25 = res.get("2025_UP", {}).get("acc"); dn25 = res.get("2025_DOWN", {}).get("acc")
    beats_up = up25 is not None and up25 >= inc["UP"]["2025"]
    beats_dn = dn25 is not None and dn25 >= inc["DOWN"]["2025"]
    out = {"exp": "I3 |return|-weighted cross-pair retrain @MX_HOR=10", "POW": POW,
           "incumbent_xpair_forward_cov10": inc, "incumbent_xpair_refit_p10": {"UP": 0.5863, "DOWN": 0.5683},
           "primary_val_auc": round(float(roc_auc_score(yva, pva)), 4), "meta_thr": round(THR, 4),
           "per_year_side": res,
           "verdict": {"IMPROVES_UP_2025": bool(beats_up), "IMPROVES_DOWN_2025": bool(beats_dn),
                       "note": f"UP25={up25} vs {inc['UP']['2025']}; DOWN25={dn25} vs {inc['DOWN']['2025']} -> "
                               + ("a side improves binding-2025; refit-CPCV it" if (beats_up or beats_dn) else "no binding-year improvement either side")}}
    json.dump(out, open("m10_magweight_result.json", "w"), indent=1)
    for k, v in res.items():
        if isinstance(v, dict): print(f"  {k:16} n={v.get('n')} acc={v.get('acc')} CI={v.get('ci')} up_rate={v.get('up_rate')}", flush=True)
    print(f"[magw10] {out['verdict']['note']}", flush=True)
    print(f"[magw10] -> m10_magweight_result.json ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
