"""Sweep row A8c: a genuine (5m,DOWN) SPECIALIST. The symmetric m5xp book's down-side fails the binding 2025
year (0.538). This retrains a DOWN-SPECIFIC meta-labeler on down-call correctness, using down-relevant features
(USD-basket strength, overbought RSI/BB extremes, cross-pair agreement/dispersion/OF, plus the existing meta
score) — re-ranking the down-predicted bars to find a down-confident pocket the symmetric meta misses. Primary
ranker unchanged (subset-training the primary kills ranking, per the registry); only the meta is specialized.
Disciplined: threshold by worst-VAL-half DOWN accuracy, per-year CI95, moved-only.

PRE-REGISTERED FALSIFIER: KILL A8c unless the VAL-worst-half-selected DOWN-specialist gate gives 2025 (binding)
DOWN-acc with n>=100 AND CI95-lo>=0.541. The confidence-curve ceiling (2025 DOWN ~0.555 at any symmetric gate)
makes this an uphill test; a specialist only survives if it finds a 2025 down-subset the symmetric meta blurs."""
import json, numpy as np
import lightgbm as lgb
import m5_xpair as MX
import m5_xpair_production as XP

WINS = {"val": ["2022", "2023"], "test24": ["2024"], "test25": ["2025"], "oos": ["2026"]}
DOWN_EXTRA = ["5m_rsi", "15m_rsi", "5m_bb_pctb", "5m_rangepos_24", "5m_dist_ema20", "1m_rsi"]


def boot(c, nb=4000, seed=7):
    c = np.asarray(c, float)
    if len(c) < 5: return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed); n = len(c)
    a = np.array([c[rng.integers(0, n, n)].mean() for _ in range(nb)])
    return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))


def build():
    p, P, M = XP._load(); cols = p["primary_feats"]; mcols = p["meta_feats"]
    out = {}
    for w, yrs in WINS.items():
        D = MX.build_xp(yrs); D = MX.augment(D, yrs, XP.MODE)
        pr = P.predict(D[cols].astype("float32"))
        sm = M.predict(XP._Xmeta(D, pr, mcols))
        bask = [c for c in D.columns if c.startswith("usdbask")]
        extra = [c for c in DOWN_EXTRA if c in D.columns]
        conf = np.abs(pr - 0.5).astype("float32")
        Xd = np.column_stack([conf, sm.astype("float32")] +
                             [D[c].values.astype("float32") for c in mcols] +
                             [D[c].values.astype("float32") for c in bask] +
                             [D[c].values.astype("float32") for c in extra])
        out[w] = {"pr": pr, "y": D["_y"].astype(int).values, "ts": D["_ts"].values.astype("int64"),
                  "ny": D["sess_ny"].values > 0.5, "Xd": Xd}
        del D
    return out


def main():
    A = build()
    # train down-specialist meta on VAL down-predicted NY bars: target = down call correct (y==0)
    v = A["val"]; dn = v["ny"] & (v["pr"] < 0.5)
    ycorr = (v["y"][dn] == 0).astype(int)
    Md = lgb.LGBMClassifier(objective="binary", metric="auc", learning_rate=0.02, num_leaves=15,
        min_child_samples=800, subsample=0.8, subsample_freq=1, colsample_bytree=0.6, reg_lambda=20,
        n_estimators=400, n_jobs=20, verbosity=-1)
    Md.fit(v["Xd"][dn], ycorr)
    yr = lambda ts: (np.asarray(ts, dtype="datetime64[s]").astype("datetime64[Y]").astype(int) + 1970)
    vyr = yr(v["ts"][dn]); sval = Md.predict_proba(v["Xd"][dn])[:, 1]
    vts = v["ts"][dn]; vy = v["y"][dn]; vpr = v["pr"][dn]

    def down_eval(ts, y, mask):
        s = MX.nonoverlap_chrono(ts[mask], np.ones(int(mask.sum()), bool)) if mask.sum() else np.array([], int)
        if len(s) == 0: return 0, float("nan")
        return len(s), float((y[mask][s] == 0).mean())

    # select threshold by worst-VAL-half (min over 2022,2023) DOWN acc among VAL down bars
    best = None
    for q in (0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95):
        thr = float(np.quantile(sval, q)); accs = []; ns = []
        for yy in (2022, 2023):
            mm = (vyr == yy) & (sval >= thr)
            nn, acc = down_eval(vts, vy, mm)
            ns.append(nn); accs.append(acc)
        if min(ns) < 40 or any(np.isnan(accs)): continue
        wh = min(accs)
        if best is None or wh > best[1]: best = (thr, wh, q)
    if best is None:
        print("[A8c] no VAL threshold met min-n", flush=True)
        json.dump({"row": "A8c DOWN specialist", "KILLED": True, "reason": "no VAL threshold"}, open("m5_downspec_result.json", "w"), indent=1); return
    THR, WH, Q = best
    per = {}
    for w, label in (("test24", "2024"), ("test25", "2025"), ("oos", "2026")):
        d = A[w]; dn2 = d["ny"] & (d["pr"] < 0.5)
        sc = Md.predict_proba(d["Xd"][dn2])[:, 1]
        sel = MX.nonoverlap_chrono(d["ts"][dn2], sc >= THR)
        if len(sel) == 0: per[label] = {"n": 0, "acc": None, "ci": [None, None]}; continue
        corr = (d["y"][dn2][sel] == 0).astype(float); lo, hi = boot(corr)
        per[label] = {"n": int(len(sel)), "acc": round(float(corr.mean()), 4), "ci": [round(lo, 4), round(hi, 4)]}
    b25 = per.get("2025", {})
    survived = bool(b25.get("acc") and b25.get("n", 0) >= 100 and b25["ci"][0] is not None and b25["ci"][0] >= 0.541)
    out = {"row": "A8c DOWN specialist (down-specific meta-labeler)", "breakeven": 0.541,
           "val_thr": round(THR, 4), "val_q": Q, "val_worsthalf_down": round(WH, 4), "per_year_down": per,
           "falsifier": {"DOWN_SPECIALIST_WORKS": survived,
                         "verdict": (f"A8c SURVIVES: 2025 DOWN={b25.get('acc')} n{b25.get('n')} CI-lo clears 0.541"
                                     if survived else
                                     f"A8c KILLED: down-specialist 2025 DOWN={b25.get('acc')} n{b25.get('n')} CI-lo<0.541 "
                                     "-> a purpose-built DOWN meta cannot crack the 2025 wall either; (5m,DOWN) dead.")}}
    json.dump(out, open("m5_downspec_result.json", "w"), indent=1)
    print(f"[A8c] val_thr={THR:.4f}(q{Q}) worsthalf_down={WH:.4f}", flush=True)
    for k, vv in per.items(): print(f"   {k}: n={vv['n']} down_acc={vv['acc']} CI={vv['ci']}", flush=True)
    print(f"[A8c] {out['falsifier']['verdict']}", flush=True)
    print("[A8c] -> m5_downspec_result.json", flush=True)


if __name__ == "__main__":
    main()
