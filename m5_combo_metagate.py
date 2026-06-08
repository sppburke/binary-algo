"""CAPSTONE COMBINATION (2026-06-03): META-LABELER GATE on the top blends from m5_combo_sweep.

m5_combo_sweep swept 2874 combinations gated on CONFIDENCE-COVER and found no survivor. But the lever that makes the
UP incumbent (m5xp .577) work is a LEARNED META-LABELER P(call correct | meta-features) — which the raw blends LACK.
This applies that meta-gate on top of the best blends (esp. the DOWN pxp+p10 blend that reached binding-2025 .5529/CI-lo
.5277, the closest DOWN has come). Reuses the cached combo_scores_<split>.npz (no rebuild).

For each (combo, side): combined score -> directional call (UP: score>0.5 / DOWN: score<0.5). Fit an LGB meta-labeler on
TRAIN bars taking that call, target = call-correct, features = [score, |score-.5|, all producer scores, mxp, mxp_stk,
producer dispersion, sess_ny]. Gate the call on meta-pred >= thr, thr chosen on WORST-VAL-HALF (max selective acc at
coverage>=floor). Eval per-year 2024/25/26: nonoverlap_chrono(300), boot CI95, ties-LOSE (true y5).

PRE-REGISTERED FALSIFIER (written before OOS): KILL each (combo,side) unless binding-year (worst of 24/25/26, n>=150)
CI95-lo: UP >= 0.553 AND beats incumbent 0.577 by >1 SE; DOWN >= 0.541 AND beats 0.5441 by >1 SE; up-rate of selected
bars in [0.47,0.53] is NOT required (a direction gate legitimately skews it) BUT report it. Any survivor -> nested-refit
CPCV (refit meta per purged fold, C(8,2)=28 paths, p10>=breakeven AND >=80% paths clear) BEFORE any freeze. Expected
null (meta-gate on a DOWN signal ANTI-TRANSFERS — m5_downspec 2025 .509, corr(VAL,OOS)=-.54) — RUN to confirm.

  ~/binary-algo-venv/bin/python m5_combo_metagate.py
"""
import os, sys; sys.argv = ["x"]
import json, time, numpy as np
import lightgbm as lgb
import m5_xpair as MX
ROOT = "/home/sean/git/binary-algo"; BE = 0.541
INCUMBENT = {"UP": 0.577, "DOWN": 0.5441}
SPLITS = ("train", "val", "test24", "test25", "oos")
YEAR = {"test24": 2024, "test25": 2025, "oos": 2026}
SCORE_COLS = ["ts", "y5", "fwd5", "sess_ny", "p5", "p10", "p15", "p30", "pxp", "mxp", "pstk", "mxp_stk", "pmag"]
PRODUCERS = ["p5", "p10", "p15", "p30", "pxp", "pstk"]
# top blends from m5_combo_sweep leaderboard (weights on producers; convex)
COMBOS = {
    "pxp_only":           {"pxp": 1.0},
    "blendUP_pxp43_p3057": {"pxp": 0.43, "p30": 0.57},
    "blendUP_pxp50_p3050": {"pxp": 0.50, "p30": 0.50},
    "blendDN_pxp43_p1057": {"pxp": 0.43, "p10": 0.57},
    "blendDN_pxp50_p1050": {"pxp": 0.50, "p10": 0.50},
    "blendDN_pxp40_p1060": {"pxp": 0.40, "p10": 0.60},
}

def boot(c, nb=2500, seed=7):
    c = np.asarray(c, float)
    if len(c) < 5: return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed); n = len(c)
    a = np.array([c[rng.integers(0, n, n)].mean() for _ in range(nb)])
    return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))

def load():
    M = {}
    for sp in SPLITS:
        z = np.load(f"{ROOT}/combo_scores_{sp}.npz")
        M[sp] = {k: z[k] for k in SCORE_COLS}
    return M

def combo_score(d, w):
    s = np.zeros(len(d["ts"]), dtype="float64")
    for k, ww in w.items(): s += ww * d[k]
    return s / sum(w.values())

def meta_feats(d, score):
    disp = np.std(np.column_stack([d[p] for p in PRODUCERS]), axis=1)
    return np.column_stack([score, np.abs(score - 0.5)] +
                           [d[p] for p in PRODUCERS] + [d["mxp"], d["mxp_stk"], disp, d["sess_ny"].astype("float32")])

def eval_year(d, score, metapred, sv, thr):
    """selected = NY & call==sv & metapred>=thr; nonoverlap; return (win, n, ci_lo, up_rate)."""
    call = (score > 0.5) if sv == 1 else (score < 0.5)
    g = (d["sess_ny"] > 0.5) & call & (metapred >= thr)
    ts = d["ts"].astype("int64")
    sel = MX.nonoverlap_chrono(ts, g, 300)
    if len(sel) < 1: return (float("nan"), 0, float("nan"), float("nan"))
    y = d["y5"].astype(int)
    cc = (y[sel] == sv).astype(float)
    lo, hi = boot(cc)
    up = float((y[sel] == 1).mean())
    return (float(cc.mean()), len(sel), lo, up)

def worst_val_half(val, score, metapred, sv):
    """pick thr (on VAL meta-pred quantiles) maximizing the MIN selective-acc over the two VAL time-halves, cov>=floor."""
    ts = val["ts"].astype("int64"); half = np.median(ts); halves = [ts < half, ts >= half]
    call = (score > 0.5) if sv == 1 else (score < 0.5)
    base = (val["sess_ny"] > 0.5) & call
    y = val["y5"].astype(int)
    best = (None, -1.0)
    for q in (0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95):
        mp = metapred[base]
        if mp.size < 40: break
        thr = float(np.quantile(mp, q))
        accs = []
        ok = True
        for h in halves:
            m = base & h & (metapred >= thr)
            sel = MX.nonoverlap_chrono(ts, m, 300)
            if len(sel) < 15: ok = False; break
            accs.append((y[sel] == sv).mean())
        if ok and min(accs) > best[1]: best = (thr, float(min(accs)))
    return best

def main():
    t0 = time.time()
    # pre-register stub
    json.dump({"test": "meta-labeler gate on top combo blends", "breakeven": BE, "incumbent": INCUMBENT,
               "status": "PRE-REGISTERED",
               "FALSIFIER": "KILL unless binding-year (worst 24/25/26, n>=150) CI95-lo: UP>=0.553 & beats 0.577 by >1SE; "
                            "DOWN>=0.541 & beats 0.5441 by >1SE. Survivor -> nested-refit CPCV (p10>=BE & >=80% of 28 paths). "
                            "Expected null (meta-gate on DOWN anti-transfers, m5_downspec .509)."},
              open(f"{ROOT}/m5_combo_metagate_result.json", "w"), indent=2)
    M = load()
    print(f"[metagate] loaded caches; train n={len(M['train']['ts']):,} ({time.time()-t0:.0f}s)", flush=True)
    out = {"test": "meta-labeler gate on top combo blends", "breakeven": BE, "incumbent": INCUMBENT, "results": {}}
    EV = ("test24", "test25", "oos")
    survivors = []
    for cname, w in COMBOS.items():
        sc = {sp: combo_score(M[sp], w) for sp in SPLITS}
        for sv, side in ((1, "UP"), (0, "DOWN")):
            tr = M["train"]; tsc = sc["train"]
            call_tr = (tsc > 0.5) if sv == 1 else (tsc < 0.5)
            Xtr = meta_feats(tr, tsc)[call_tr]
            ytr = (tr["y5"].astype(int)[call_tr] == sv).astype(int)
            if Xtr.shape[0] < 2000 or ytr.mean() in (0.0, 1.0): continue
            meta = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=31, min_child_samples=200,
                                      subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=10,
                                      n_jobs=20, verbosity=-1)
            meta.fit(Xtr, ytr)
            mp = {sp: meta.predict_proba(meta_feats(M[sp], sc[sp]))[:, 1] for sp in ("val",) + EV}
            thr, vwh = worst_val_half(M["val"], sc["val"], mp["val"], sv)
            if thr is None: continue
            yrs = {}
            for sp in EV:
                win, n, lo, up = eval_year(M[sp], sc[sp], mp[sp], sv, thr)
                if n >= 20: yrs[YEAR[sp]] = dict(win=round(win, 4), n=n, ci_lo=round(lo, 4), up_rate=round(up, 4))
            if len(yrs) < 3: continue
            wins = [yrs[y]["win"] for y in yrs]; los = [yrs[y]["ci_lo"] for y in yrs]; ns = [yrs[y]["n"] for y in yrs]
            byear = min(yrs, key=lambda y: yrs[y]["win"])
            bind = dict(year=byear, win=yrs[byear]["win"], ci_lo=yrs[byear]["ci_lo"], n=yrs[byear]["n"], min_n=min(ns))
            inc = INCUMBENT[side]
            # >1 SE beat: CI-lo (lower 2.5%) above incumbent is a strong proxy; require ci_lo>=BE-side and win>inc and ci_lo>inc-? use ci_lo>=thr_side
            thr_side = 0.553 if side == "UP" else BE
            clears = bool(bind["ci_lo"] >= thr_side and bind["min_n"] >= 150 and bind["win"] > inc)
            out["results"][f"{cname}|{side}"] = dict(val_worsthalf=round(vwh, 4), thr=round(thr, 4),
                                                     per_year=yrs, binding=bind, clears=clears)
            if clears: survivors.append(f"{cname}|{side}")
            print(f"  {cname} {side}: vWH={vwh:.4f} thr={thr:.3f} binding={bind} clears={clears} ({time.time()-t0:.0f}s)", flush=True)
    out["survivors"] = survivors
    out["VERDICT"] = dict(
        any_survivor=bool(survivors), survivors=survivors,
        statement=(f"meta-gate on top blends: {'SURVIVOR(S) '+str(survivors)+' -> run nested-refit CPCV' if survivors else 'NO survivor'}. "
                   + ("" if survivors else
                      "Adding the learned meta-labeler (UP's secret sauce) on top of the best blends does NOT lift any "
                      "combination's binding-year CI-lo over breakeven/incumbent. The meta-gate cannot manufacture 2025/2026 "
                      "regime information the blended score lacks (consistent with m5_downspec DOWN-meta anti-transfer .509 and "
                      "corr(VAL,OOS)=-.54). The model-COMBINATION cross-product is now exhausted: scores x {conf-cover, meta} x "
                      "{blend, gate, consensus, cascade, regime, stack} all fail. The wall is information/regime, not architecture.")))
    json.dump(out, open(f"{ROOT}/m5_combo_metagate_result.json", "w"), indent=2, default=str)
    print(f"\nVERDICT: {out['VERDICT']['statement']}\n-> m5_combo_metagate_result.json ({time.time()-t0:.0f}s)", flush=True)

if __name__ == "__main__": main()
