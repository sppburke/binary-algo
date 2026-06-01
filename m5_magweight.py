"""EDGE-IMPROVEMENT EXP-1: |return|-weighted retrain of the m5xp cross-pair pipeline. Hypothesis: our MAGNITUDE
signal is strong (magAUC 0.73-0.79) while raw direction is ~0.52 — so up-weighting TRAINING toward large-|move|
bars (where the sign is most predictable) should channel the magnitude strength into a BETTER directional UP
edge on the gated high-confidence bars. Same cross-pair features + meta-gate as m5xp; only the primary's
sample_weight changes. Then side-split UP/DOWN per year vs the incumbent and (if it improves) certify on CPCV.

INCUMBENT to beat (m5xp UP, m5_updown_result.json): .605/.577/.615 ; binding 2025 = 0.577.
PRE-REGISTERED FALSIFIER: KILL unless UP per-year ≥ incumbent in the BINDING 2025 year (≥0.577) with healthy n
AND up-rate tripwire ∈[.47,.53]. This is an IMPROVEMENT attempt, not a null-confirmation — we want it to win."""
import sys, json, time, numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import m5_xpair as MX
import m5_xpair_production as XP

MODE = XP.MODE; SPL = MX.SPL
POW = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0   # weight = (|fwd|/median)^POW, clipped


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
    print(f"[magw] POW={POW} train={len(TR):,} val={len(VA):,} feats={len(cols)} wt[min/med/max]={w_tr.min():.2f}/{np.median(w_tr):.2f}/{w_tr.max():.2f} {time.time()-t0:.0f}s", flush=True)
    P = lgb.LGBMClassifier(objective="binary", metric="auc", learning_rate=0.02, num_leaves=127, min_child_samples=400,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=20, n_estimators=3000, n_jobs=20, verbosity=-1)
    P.fit(TR[cols].astype("float32"), ytr, sample_weight=w_tr,
          eval_set=[(VA[cols].astype("float32"), yva)], eval_metric="auc",
          callbacks=[lgb.early_stopping(150), lgb.log_evaluation(0)])
    pva = P.predict_proba(VA[cols].astype("float32"))[:, 1]
    print(f"[magw] primary VAL AUC={roc_auc_score(yva, pva):.4f} (incumbent ~0.523) {time.time()-t0:.0f}s", flush=True)
    # meta-labeler (unchanged recipe)
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
    print(f"[magw] FROZEN meta_thr={THR:.4f} (VAL q{Q}) worst-half={HM:.3f}", flush=True)

    # side-split per year vs incumbent
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
    inc = {"2024": 0.605, "2025": 0.577, "2026": 0.615}
    up = {k.split("_")[0]: v for k, v in res.items() if k.endswith("_UP")}
    beats25 = up.get("2025", {}).get("acc") and up["2025"]["acc"] >= 0.577
    out = {"exp": "EXP-1 |return|-weighted m5xp retrain", "POW": POW, "incumbent_up": inc,
           "primary_val_auc": round(float(roc_auc_score(yva, pva)), 4), "meta_thr": round(THR, 4),
           "per_year_side": res,
           "verdict": {"IMPROVES_UP": bool(beats25),
                       "note": ("UP 2025 %s vs incumbent 0.577 -> %s" % (up.get("2025", {}).get("acc"),
                                "IMPROVES (certify on CPCV)" if beats25 else "no improvement on the binding year"))}}
    json.dump(out, open(f"m5_magweight_result.json", "w"), indent=1)
    for k, v in res.items():
        if isinstance(v, dict): print(f"  {k:14} n={v.get('n')} acc={v.get('acc')} CI={v.get('ci')} up_rate={v.get('up_rate')}", flush=True)
    print(f"[magw] {out['verdict']['note']}", flush=True)
    print(f"[magw] -> m5_magweight_result.json ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
