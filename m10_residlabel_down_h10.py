"""RESIDUAL-LABEL DOWN-RESCUE @ H=10 (600s) — DISCOVER lever for the 10m DOWN side (N19 retargeted from
m5_residlabel_down_h5.py). Faithful retarget: H 5->10, GAP 600s, incumbent = the certified 10m cross-pair DOWN
floor (refit p10 .5683), output m10_residlabel_down_h10_result.json. Machinery IMPORTED verbatim from
min1_residtarget (build_resid/_causal_beta/_forward_basket) — single source of truth, no replication.

DISTINCT MECHANISM: RELABELS the target  y_resid = sign(eu_fwd10 - beta_t*basket_fwd10), removing the common-USD
factor from the LABEL ("did EURUSD fall by MORE than the dollar's broad move implies"). The certified cross-pair
book uses the USD-residual as a FEATURE; this makes it the LABEL. Tests whether a residual-relabel DOWN model can
BEAT the already-certified DOWN p10 .5683 (an IMPROVEMENT attempt — 10m DOWN already certifies, unlike 5m where it
was dead, so the bar is higher here).

ZERO-LOOK-AHEAD @ H=10: beta_t window ends at anchor a_hi=t-H-1=t-11; last bar in any beta label is t-1, strictly
before t. forward basket + label both 600s-contiguity-gated. window math imported unchanged.

PRE-REGISTERED FALSIFIER (written to stub BEFORE OOS): KILL unless the H=10 residual-sign DOWN-SPLIT binding-year
(worst of 2024/2025/2026, n>=150) CI95-lo clears 0.541 AND strictly beats the certified cross-pair DOWN p10 .5683,
with up-rate in [0.47,0.53] AND a stable same-side DOWN edge in BOTH 2024 AND 2026. Else: residual-relabel does not
improve the certified DOWN edge at 10m.
    ~/binary-algo-venv/bin/python m10_residlabel_down_h10.py
"""
import os, sys, json, time, gc
import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

H = 10
GAP = H * 60                       # 600s
os.environ["MX_HOR"] = str(H)

import m5_xpair as MX
MX.HOR = H; MX.GAP_S = GAP         # pin module globals so build_xp/build_resid use H=10
import min1_residtarget as R
R.H_GAP = GAP

from m5_xpair import (SPL, augment, feat_cols, xp_cols, nonoverlap_chrono)

ROOT = "/home/sean/git/binary-algo"
OUT = f"{ROOT}/m10_residlabel_down_h10_result.json"
BREAKEVEN = 0.541
INCUMBENT_XPAIR_DOWN_P10 = 0.5683  # EURUSD.m10xp.v1 refit-CPCV DOWN p10 (the bar to beat at 10m)
COV = 0.05
TRAIN_CAP = R.TRAIN_CAP
STRIDE_TRAIN = 8
MODE = "xp"
DOWN = 0


def boot(c, nb=2500, seed=7):
    c = np.asarray(c, float)
    if len(c) < 5:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed); n = len(c)
    a = np.array([c[rng.integers(0, n, n)].mean() for _ in range(nb)])
    return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))


def downsplit_year(D, score, cover_thr):
    y = D["_y"].astype(int).values
    ts = D["_ts"].values.astype("int64")
    ny = D["sess_ny"].values > 0.5
    conf = np.abs(score - 0.5)
    g = ny & (score < 0.5)
    uprate = float(y.mean())
    if g.sum() < 20:
        return {"win": None, "n": 0, "ci": [None, None]}, uprate
    cthr = cover_thr if cover_thr is not None else float(np.quantile(conf[g], 1 - COV))
    m = g & (conf >= cthr)
    sel = nonoverlap_chrono(ts, m, GAP)
    if len(sel) < 20:
        return {"win": None, "n": len(sel), "ci": [None, None]}, uprate
    cc = (y[sel] == DOWN).astype(float)
    lo, hi = boot(cc)
    return {"win": round(float(cc.mean()), 4), "n": len(sel),
            "ci": [round(lo, 4), round(hi, 4)]}, uprate


def pooled_year(D, score):
    yres = D["_yresid"].astype(int).values
    ts = D["_ts"].values.astype("int64")
    ny = D["sess_ny"].values > 0.5
    conf = np.abs(score - 0.5)
    if ny.sum() < 20:
        return {"win": None, "n": 0, "ci": [None, None]}
    cthr = float(np.quantile(conf[ny], 1 - COV))
    m = ny & (conf >= cthr)
    sel = nonoverlap_chrono(ts, m, GAP)
    if len(sel) < 20:
        return {"win": None, "n": len(sel), "ci": [None, None]}
    pred_up = (score[sel] > 0.5).astype(int)
    cc = (pred_up == yres[sel]).astype(float)
    lo, hi = boot(cc)
    return {"win": round(float(cc.mean()), 4), "n": len(sel), "ci": [round(lo, 4), round(hi, 4)]}


def worst_val_half_cover_thr(VA, score):
    ts = VA["_ts"].values.astype("int64")
    ny = VA["sess_ny"].values > 0.5
    y = VA["_y"].astype(int).values
    conf = np.abs(score - 0.5)
    g = ny & (score < 0.5)
    mid = np.median(ts)
    h1 = ts < mid; h2 = ~h1
    best = None
    for cov in (0.10, 0.05, 0.03, 0.02):
        pool = conf[g]
        if pool.size < 500:
            continue
        thr = float(np.quantile(pool, 1 - cov))
        accs = []
        ok = True
        for hmask in (h1, h2):
            m = g & hmask & (conf >= thr)
            sel = nonoverlap_chrono(ts, m, GAP)
            if len(sel) < 60:
                ok = False; break
            accs.append((y[sel] == DOWN).mean())
        if not ok:
            continue
        worst = min(accs)
        if best is None or worst > best[1]:
            best = (thr, worst, cov)
    return best


def main():
    t0 = time.time()
    print(f"######## m10_residlabel_down_h10  H={H} (600s) mode={MODE} ########", flush=True)
    print(f"[cfg] beta_win={R.BETA_WIN} beta_minp={R.BETA_MINP} train_cap={TRAIN_CAP} "
          f"stride_train={STRIDE_TRAIN} cov={COV} breakeven={BREAKEVEN} "
          f"incumbent_xpair_down_p10={INCUMBENT_XPAIR_DOWN_P10}", flush=True)

    stub = {
        "experiment": "m10_residlabel_down_h10",
        "design": "DOWN-split of residual-label sign(eu_fwd10 - causal_beta_t*basket_fwd10) @ H=10 (600s)",
        "distinct_from_prior_DOWN_kills": ("RELABELS the target (removes common-USD factor from the LABEL); "
            "the certified cross-pair book uses the residual as a FEATURE not a LABEL"),
        "machinery": "IMPORTED verbatim from min1_residtarget (build_resid/_causal_beta/_forward_basket) @ H=10",
        "beta_win": R.BETA_WIN, "beta_minp": R.BETA_MINP, "train_cap": TRAIN_CAP,
        "cov": COV, "breakeven": BREAKEVEN, "incumbent_xpair_down_p10": INCUMBENT_XPAIR_DOWN_P10,
        "discipline": ("NY gate, DOWN-vote gate (resid<0.5), nonoverlap_chrono(600), worst-VAL-half cover select, "
                       "per-year boot CI95, ties LOSE (raw _y), moved up-rate in [0.47,0.53] tripwire"),
        "PRE_REGISTERED_FALSIFIER": (
            "KILL (residual-relabel does not improve certified 10m DOWN) unless the H=10 residual-sign DOWN-SPLIT "
            "binding-year (worst of 2024/2025/2026, n>=150) CI95-lo clears 0.541 AND strictly beats the certified "
            "cross-pair DOWN p10 .5683, with up-rate in [0.47,0.53] AND a stable same-side DOWN edge in BOTH 2024 "
            "AND 2026 (single-2025-only or sign-flip => KILL)."),
        "status": "PRE-REGISTERED (OOS not yet run)",
        "down_split": {}, "pooled_resid_sign": {}, "VERDICT": None,
    }
    with open(OUT, "w") as f:
        json.dump(stub, f, indent=2, default=float)
    print(f"[pre-register] wrote falsifier stub -> {OUT}", flush=True)

    TR = R.build_resid(SPL["train"], H, stride=STRIDE_TRAIN)
    if TR is None or len(TR) == 0:
        stub["VERDICT"] = {"error": "no train"}; json.dump(stub, open(OUT, "w"), indent=2, default=float); return
    if len(TR) > TRAIN_CAP:
        TR = TR.sample(n=TRAIN_CAP, random_state=7).sort_index()
    VA = R.build_resid(SPL["val"], H, stride=2)

    xpc = [c for c in xp_cols(TR) if not c.startswith("_")]
    TR = augment(TR, SPL["train"], MODE); VA = augment(VA, SPL["val"], MODE)
    cols = feat_cols(MODE, TR, xpc)
    cols = [c for c in cols if not c.startswith("_")]
    print(f"[build] train={len(TR):,} val={len(VA):,} feats={len(cols)} "
          f"resid-uprate(train)={TR['_yresid'].mean():.4f} raw-uprate(train)={TR['_y'].mean():.4f} "
          f"{time.time()-t0:.0f}s", flush=True)

    yA = TR["_yresid"].astype(int).values
    XtrA = TR[cols].astype("float32")
    XvaA = VA[cols].astype("float32")
    yvaA = VA["_yresid"].astype(int).values
    LA = lgb.LGBMClassifier(objective="binary", metric="auc", learning_rate=0.02, num_leaves=63,
                            min_child_samples=400, subsample=0.8, subsample_freq=1, colsample_bytree=0.5,
                            reg_lambda=20, n_estimators=2500, n_jobs=10, verbosity=-1)
    LA.fit(XtrA, yA, eval_set=[(XvaA, yvaA)], eval_metric="auc",
           callbacks=[lgb.early_stopping(120), lgb.log_evaluation(0)])
    pvaA = LA.predict_proba(XvaA)[:, 1]
    aucvA = roc_auc_score(yvaA, pvaA)
    print(f"[val] resid-sign VAL AUC={aucvA:.4f} best_iter={LA.best_iteration_} {time.time()-t0:.0f}s", flush=True)

    fr = worst_val_half_cover_thr(VA, pvaA)
    cover_thr = fr[0] if fr is not None else None
    print(f"[val] FROZEN DOWN cover: thr={cover_thr} worsthalf={None if fr is None else round(fr[1],4)} "
          f"cov={None if fr is None else fr[2]}", flush=True)
    del TR, XtrA, VA, XvaA; gc.collect()

    down = {}; pooled = {}; uprates = {}
    YEARMAP = {"2024": ["2024"], "2025": ["2025"], "2026": ["2026"]}
    for yr, yrs in YEARMAP.items():
        D = R.build_resid(yrs, H, stride=1)
        if D is None or len(D) == 0:
            down[yr] = {"win": None, "n": 0, "ci": [None, None], "error": "empty"}
            pooled[yr] = {"win": None, "n": 0, "ci": [None, None]}; continue
        XD = D[cols].astype("float32")
        score = LA.predict_proba(XD)[:, 1]
        ds, uprate = downsplit_year(D, score, cover_thr)
        ps = pooled_year(D, score)
        down[yr] = ds; pooled[yr] = ps; uprates[yr] = round(uprate, 4)
        print(f"=== {yr} === raw-uprate={uprate:.4f} | DOWN-split n{ds['n']} win={ds['win']} ci={ds['ci']} "
              f"| pooled-resid n{ps['n']} win={ps['win']} ci={ps['ci']}", flush=True)
        del D, XD, score; gc.collect()

    elig = {yr: d for yr, d in down.items() if d.get("win") is not None and d.get("n", 0) >= 150}
    if elig:
        bind_yr = min(elig, key=lambda k: elig[k]["win"])
        bind = elig[bind_yr]
        binding = {"year": bind_yr, "win": bind["win"], "n": bind["n"], "ci_lo": bind["ci"][0]}
    else:
        binding = {"year": None, "win": None, "n": 0, "ci_lo": None}

    def upr_ok(yr):
        u = uprates.get(yr)
        return (u is not None) and (0.47 <= u <= 0.53)
    all_upr_ok = all(upr_ok(yr) for yr in ("2024", "2025", "2026") if yr in uprates) and len(uprates) > 0

    bind_lo = binding["ci_lo"]
    bind_n = binding["n"] or 0
    clears_be = bind_lo is not None and bind_lo > BREAKEVEN
    beats_inc = bind_lo is not None and bind_lo > INCUMBENT_XPAIR_DOWN_P10
    nbig = bind_n >= 150

    def edge24_26():
        d24 = down.get("2024", {}).get("win"); d26 = down.get("2026", {}).get("win")
        if d24 is None or d26 is None:
            return False
        return (d24 > 0.5) and (d26 > 0.5)
    stable_2yr = edge24_26()

    d25 = down.get("2025", {})
    d25_lo = d25.get("ci", [None, None])[0]

    survives = bool(clears_be and beats_inc and nbig and all_upr_ok and stable_2yr)
    killed = not survives

    verdict = {
        "binding": binding,
        "uprates": uprates, "uprate_tripwire_ok": bool(all_upr_ok),
        "binding_ci_lo_clears_breakeven_0.541": bool(clears_be),
        "binding_ci_lo_beats_xpair_down_0.5683": bool(beats_inc),
        "binding_n_ge_150": bool(nbig),
        "stable_DOWN_edge_2024_AND_2026": bool(stable_2yr),
        "2025_down_ci_lo": d25_lo,
        "SURVIVES": survives, "KILLED": killed,
        "statement": (
            "RESIDUAL-LABEL DOWN @ H=10 " + ("SURVIVES — improves certified DOWN." if survives else "KILLED — does not improve certified DOWN .5683.") +
            f" binding-year={binding['year']} win={binding['win']} ci_lo={bind_lo} (n={bind_n});"
            f" clears breakeven={clears_be}; beats xpair DOWN .5683={beats_inc};"
            f" up-rate tripwire ok={all_upr_ok}; stable DOWN edge 2024&2026={stable_2yr}.")
    }

    stub.update({
        "status": "COMPLETE",
        "val_auc_resid": float(aucvA),
        "frozen_down_cover": (None if fr is None else {"thr": fr[0], "worsthalf": round(fr[1], 4), "cov": fr[2]}),
        "down_split": down, "pooled_resid_sign": pooled, "uprates": uprates,
        "VERDICT": verdict,
    })
    with open(OUT, "w") as f:
        json.dump(stub, f, indent=2, default=float)
    print(f"\n[VERDICT] {verdict['statement']}", flush=True)
    print(f"[saved] {OUT}  ({time.time()-t0:.0f}s total)", flush=True)
    del LA; gc.collect()


if __name__ == "__main__":
    main()
