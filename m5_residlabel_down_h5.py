"""RESIDUAL-LABEL DOWN-RESCUE @ H=5 (300s) — the never-executed DOWN lever.

WHY THIS IS DISTINCT FROM EVERY PRIOR DOWN KILL
------------------------------------------------
Every prior 5m DOWN attempt (magweight POW=0.5/0.25, seed-ensemble, side-split, A8b/A8c
specialist, conf-curve, loss-batch focal/quantile/asym, calibration, nested-refit CPCV)
operated on the RAW EURUSD-sign target  y = sign(eu_fwd5)  and only ever reweighted,
gated, calibrated, pooled, or re-objectived it. They all hit the same 2025-USD-regime wall
(DOWN 2025 sign is coin-flip in the cross-pair features; incumbent magweight DOWN p10 .5441
grazes breakeven, forward-2025 CI-lo .533 < breakeven).

This script RELABELS the target instead:
    y_resid = sign( eu_fwd5  -  beta_t * basket_fwd5 )
where basket_fwd5 is the forward 5-bar USD-weakness basket (EURUSD-equivalent) and beta_t is
a CAUSAL rolling-OLS slope of realized eu_fwd on realized basket_fwd using ONLY forward-return
pairs that fully resolved strictly before bar t (zero look-ahead). This is a *different
mechanism* — it removes the common USD factor from the LABEL, asking "did EURUSD fall by MORE
than the dollar's broad move would imply" rather than "did EURUSD fall". The USD-basket residual
was previously only ever a FEATURE; min1_residtarget made it the LABEL but its H-loop ran only
H=15 and H=1 — H=5 (the 5m settlement horizon) was NEVER executed, and min1_residtarget never
emitted a DOWN-SPLIT readout (it reports residual-sign acc + an UP-biased agreement gate).

WHAT THIS ADDS
--------------
1. Runs min1_residtarget's EXACT machinery at H=5 (imported, not re-implemented — see below).
2. A DOWN-SPLIT readout the original omits: P(true 300s move is DOWN | residual model votes DOWN),
   per-year 2024/2025/2026, deriv-faithful (m5_lossbatch.per_year discipline).

MACHINERY: IMPORTED, NOT REPLICATED
-----------------------------------
min1_residtarget exposes every needed piece at module scope, so we IMPORT them verbatim rather
than copy/paste (single source of truth, no drift risk):
    R.build_resid(years, H, stride)  -> XP frame + _fwdbask,_beta,_resid,_yresid,_y,_fwd,_ts,sess_ny
    R._causal_beta(...)              -> the zero-look-ahead rolling-OLS slope beta_t
    R._forward_basket(...)           -> forward H-bar USD-weakness basket aligned to XP anchors
    R.BETA_WIN / R.BETA_MINP         -> 500 / 100 (unchanged)
The model + feature handling mirrors min1_residtarget.run(): same LGBM hyperparameters, same
xp_cols leakage strip (drop every "_"-prefixed col so _fwdbask/_beta/_resid/_yresid/_fwd can
NEVER enter the feature set), same augment/feat_cols, same TRAIN_CAP subsample.

ZERO-LOOK-AHEAD AT H=5 — exactly how it is guaranteed
-----------------------------------------------------
beta_t is computed by R._causal_beta(eu_fwd, bask_fwd, secs, H=5). A forward pair anchored at
bar j (eu_fwd[j], bask_fwd[j]) SETTLES at bar j+H, i.e. at wall-clock time secs[j]+H*60.
For bar t we may only use pairs whose settlement time <= secs[t]. With contiguous 1-min bars the
most recent admissible anchor is t-H. _causal_beta therefore takes:
    a_hi = t - H - 1            # last anchor whose 5-bar outcome resolved STRICTLY before secs[t]
    a_lo = a_hi - win + 1       # window over anchors [a_lo, a_hi]
and uses running prefix sums of (eu_fwd,bask_fwd) pairs shifted by H. With H=5: a_hi = t-6, so
the window ends at anchor t-6, whose label window covers bars (t-6 .. t-1) — the last bar used in
ANY label that feeds beta_t is bar t-1, strictly before t. NO future bar (>= t) ever touches
beta_t. The forward basket itself (R._forward_basket) is gated by the contiguity check
(secs[H:]-secs[:-H])==H*60 == 300s, so a 5-bar forward value exists only when the 5 minutes are
wall-clock contiguous. The XP label _y / _fwd (from m5_xpair.build_xp at HOR=5) is likewise the
forward 5-bar return with the same 300s contiguity gate and ties (fwd==0) dropped. We preserve
min1_residtarget's window math byte-for-byte by importing _causal_beta — we do not re-derive it.

DOWN-SPLIT EVAL DISCIPLINE (replicated from m5_lossbatch.per_year, side=DOWN)
----------------------------------------------------------------------------
  - NY gate            sess_ny > 0.5
  - DOWN vote gate     resid_score < 0.5            (model votes DOWN)
  - confidence         conf = |resid_score - 0.5|; cover = top COV=0.05 of conf among gated bars
                       (cover threshold selected on the WORST-VAL-HALF, then APPLIED to each year)
  - non-overlap        MX.nonoverlap_chrono(ts, mask, 300)
  - ties LOSE          correctness uses the RAW EURUSD-sign label _y (0/1); a bar counts as a DOWN
                       win iff _y == 0, and ties (fwd==0) were already dropped upstream by build_xp
  - per-year CI95      2500-draw bootstrap, 2024 / 2025 / 2026
  - moved up-rate      raw EURUSD up-rate on each eval window must be in [0.47,0.53] (no ffill mirage)
  - breakeven          0.541 (deriv mid-to-mid ties-lose)

PRE-REGISTERED FALSIFIER (written to the result stub BEFORE OOS)
---------------------------------------------------------------
KILL the residual-label DOWN path (DOWN-rescue family then confirmed exhausted on-disk) unless the
H=5 residual-sign DOWN-SPLIT binding-year (worst of 2024/2025/2026, n>=150) CI95-lo clears 0.541
AND strictly beats incumbent magweight DOWN p10 .5441, with up-rate in [0.47,0.53] AND a stable
same-side DOWN edge in BOTH 2024 AND 2026 (single-2025-only or sign-flip across years => KILL).
If 2025 DOWN-split CI95-lo <= 0.541, the only remaining DOWN lever is paywalled funded
risk-reversal data.

Memory-safe (one parquet/year at a time via build_resid; train subsample cap; small models; gc).
DOES NOT run automatically as part of any sweep — invoke explicitly:
    ~/binary-algo-venv/bin/python m5_residlabel_down_h5.py
"""
import os, sys, json, time, gc
import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

# ---- horizon MUST be set before importing/using the cross-pair machinery: build_xp reads
#      MX.HOR as a module global at call time, so pin H=5 (300s) up front. ----
H = 5
GAP = H * 60                       # 300s
os.environ["MX_HOR"] = str(H)

import m5_xpair as MX
MX.HOR = H; MX.GAP_S = GAP         # pin module globals so build_xp/build_resid use H=5
import min1_residtarget as R       # IMPORT the residual-label machinery (no replication)
R.H_GAP = GAP                      # _sel_acc in min1_residtarget reads the module global H_GAP

from m5_xpair import (SPL, augment, feat_cols, xp_cols, nonoverlap_chrono)

ROOT = "/home/sean/git/binary-algo"
OUT = f"{ROOT}/m5_residlabel_down_h5_result.json"
BREAKEVEN = 0.541
INCUMBENT_MAGW_DOWN_P10 = 0.5441   # EURUSD.m5xp_magw_down.v1 refit-CPCV cov0.05 p10 (the bar to beat)
COV = 0.05                         # m5_lossbatch cover
TRAIN_CAP = R.TRAIN_CAP            # 120k, inherited
STRIDE_TRAIN = 8                   # ~10y train at H=5; matches the spirit of run()'s stride for tractability
MODE = "xp"                        # cross-pair-only (matches min1_residtarget run mode; no base/OF needed for the label test)
DOWN = 0                           # DOWN side (raw label _y == 0)


def boot(c, nb=2500, seed=7):
    """2500-draw bootstrap CI95 (m5_lossbatch.boot)."""
    c = np.asarray(c, float)
    if len(c) < 5:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed); n = len(c)
    a = np.array([c[rng.integers(0, n, n)].mean() for _ in range(nb)])
    return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))


def downsplit_year(D, score, cover_thr):
    """DOWN-split selective accuracy on ONE eval window, m5_lossbatch.per_year discipline (side=DOWN).

    score      = residual-model P(up). DOWN vote == score<0.5.
    cover_thr  = confidence threshold (frozen on worst-VAL-half); if None, derive per-window @ COV.
    Returns dict{win,n,ci} + uprate (raw EURUSD up-rate on the FULL window, for the tripwire).
    """
    y = D["_y"].astype(int).values                          # RAW EURUSD sign (ties already dropped upstream)
    ts = D["_ts"].values.astype("int64")
    ny = D["sess_ny"].values > 0.5
    conf = np.abs(score - 0.5)
    g = ny & (score < 0.5)                                   # NY + model votes DOWN
    uprate = float(y.mean())
    if g.sum() < 20:
        return {"win": None, "n": 0, "ci": [None, None]}, uprate
    cthr = cover_thr if cover_thr is not None else float(np.quantile(conf[g], 1 - COV))
    m = g & (conf >= cthr)
    sel = nonoverlap_chrono(ts, m, GAP)
    if len(sel) < 20:
        return {"win": None, "n": len(sel), "ci": [None, None]}, uprate
    cc = (y[sel] == DOWN).astype(float)                     # DOWN win iff raw move was down; ties LOSE
    lo, hi = boot(cc)
    return {"win": round(float(cc.mean()), 4), "n": len(sel),
            "ci": [round(lo, 4), round(hi, 4)]}, uprate


def pooled_year(D, score):
    """Context-only POOLED residual-sign per-year accuracy (both sides, cov0.05 conf gate, NY, non-overlap).

    P(true residual move agrees with the residual model's vote) — analogue of min1_residtarget's
    resid_sign readout, kept for context next to the DOWN-split.
    """
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
    """Choose the DOWN-side cover threshold maximizing the WORSE of the two VAL time-halves'
    DOWN-split selective acc (per discipline: NEVER VAL-acc-max). Returns (thr, worsthalf, cov)."""
    ts = VA["_ts"].values.astype("int64")
    ny = VA["sess_ny"].values > 0.5
    y = VA["_y"].astype(int).values
    conf = np.abs(score - 0.5)
    g = ny & (score < 0.5)                                   # DOWN-vote pool
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
    print(f"######## m5_residlabel_down_h5  H={H} (300s) mode={MODE} ########", flush=True)
    print(f"[cfg] beta_win={R.BETA_WIN} beta_minp={R.BETA_MINP} train_cap={TRAIN_CAP} "
          f"stride_train={STRIDE_TRAIN} cov={COV} breakeven={BREAKEVEN} "
          f"incumbent_magw_down_p10={INCUMBENT_MAGW_DOWN_P10}", flush=True)

    # ---- PRE-REGISTER the falsifier stub BEFORE any OOS evaluation ----
    stub = {
        "experiment": "m5_residlabel_down_h5",
        "design": "DOWN-split of residual-label sign(eu_fwd5 - causal_beta_t*basket_fwd5) @ H=5 (300s)",
        "distinct_from_prior_DOWN_kills": ("RELABELS the target (removes common-USD factor from the LABEL); "
            "every prior DOWN kill reweighted/gated/calibrated/pooled/re-objectived the RAW EURUSD-sign target"),
        "machinery": "IMPORTED verbatim from min1_residtarget (build_resid/_causal_beta/_forward_basket); H=5 never run there",
        "zero_lookahead": ("beta_t window ends at anchor a_hi=t-H-1 (=t-6 at H=5); last bar in any beta label is t-1, "
                           "strictly before t. forward basket + label both 300s-contiguity-gated. window math imported unchanged."),
        "beta_win": R.BETA_WIN, "beta_minp": R.BETA_MINP, "train_cap": TRAIN_CAP,
        "cov": COV, "breakeven": BREAKEVEN, "incumbent_magw_down_p10": INCUMBENT_MAGW_DOWN_P10,
        "discipline": ("NY gate, DOWN-vote gate (resid<0.5), nonoverlap_chrono(300), worst-VAL-half cover select, "
                       "per-year boot CI95, ties LOSE (raw _y), moved up-rate in [0.47,0.53] tripwire"),
        "PRE_REGISTERED_FALSIFIER": (
            "KILL the residual-label DOWN path (DOWN-rescue family then confirmed exhausted on-disk) unless the H=5 "
            "residual-sign DOWN-SPLIT binding-year (worst of 2024/2025/2026, n>=150) CI95-lo clears 0.541 AND strictly "
            "beats incumbent magweight DOWN p10 .5441, with up-rate in [0.47,0.53] AND a stable same-side DOWN edge in "
            "BOTH 2024 AND 2026 (single-2025-only or sign-flip across years => KILL). If 2025 DOWN-split CI95-lo <= 0.541, "
            "the only remaining DOWN lever is paywalled funded risk-reversal data."),
        "status": "PRE-REGISTERED (OOS not yet run)",
        "down_split": {}, "pooled_resid_sign": {}, "VERDICT": None,
    }
    with open(OUT, "w") as f:
        json.dump(stub, f, indent=2, default=float)
    print(f"[pre-register] wrote falsifier stub -> {OUT}", flush=True)

    # ---- BUILD train/val residual-label frames via the IMPORTED machinery (H=5) ----
    TR = R.build_resid(SPL["train"], H, stride=STRIDE_TRAIN)
    if TR is None or len(TR) == 0:
        stub["VERDICT"] = {"error": "no train"}; json.dump(stub, open(OUT, "w"), indent=2, default=float); return
    if len(TR) > TRAIN_CAP:
        TR = TR.sample(n=TRAIN_CAP, random_state=7).sort_index()
    VA = R.build_resid(SPL["val"], H, stride=2)

    # leakage strip: drop EVERY underscore col (_fwd,_ts,_y,_fwdbask,_beta,_resid,_yresid,hour) from features —
    # all are deterministic functions of the FORWARD return; only legitimate cross-pair feats survive.
    xpc = [c for c in xp_cols(TR) if not c.startswith("_")]
    TR = augment(TR, SPL["train"], MODE); VA = augment(VA, SPL["val"], MODE)
    cols = feat_cols(MODE, TR, xpc)
    cols = [c for c in cols if not c.startswith("_")]        # belt-and-suspenders leakage guard
    print(f"[build] train={len(TR):,} val={len(VA):,} feats={len(cols)} "
          f"resid-uprate(train)={TR['_yresid'].mean():.4f} raw-uprate(train)={TR['_y'].mean():.4f} "
          f"{time.time()-t0:.0f}s", flush=True)

    # ---- TRAIN the residual-sign model (the NEW label) — same LGBM config as min1_residtarget.run ----
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

    # ---- freeze DOWN-side cover threshold on WORST-VAL-HALF (never VAL-acc-max) ----
    fr = worst_val_half_cover_thr(VA, pvaA)
    cover_thr = fr[0] if fr is not None else None
    print(f"[val] FROZEN DOWN cover: thr={cover_thr} worsthalf={None if fr is None else round(fr[1],4)} "
          f"cov={None if fr is None else fr[2]}", flush=True)
    del TR, XtrA, VA, XvaA; gc.collect()

    # ---- PER-YEAR held-out DOWN-split + pooled resid-sign ----
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

    # ---- BINDING year = worst DOWN-split win across years with n>=150 ----
    elig = {yr: d for yr, d in down.items() if d.get("win") is not None and d.get("n", 0) >= 150}
    if elig:
        bind_yr = min(elig, key=lambda k: elig[k]["win"])
        bind = elig[bind_yr]
        binding = {"year": bind_yr, "win": bind["win"], "n": bind["n"], "ci_lo": bind["ci"][0]}
    else:
        binding = {"year": None, "win": None, "n": 0, "ci_lo": None}

    # ---- VERDICT vs the pre-registered falsifier ----
    def upr_ok(yr):
        u = uprates.get(yr)
        return (u is not None) and (0.47 <= u <= 0.53)
    all_upr_ok = all(upr_ok(yr) for yr in ("2024", "2025", "2026") if yr in uprates) and len(uprates) > 0

    bind_lo = binding["ci_lo"]
    bind_n = binding["n"] or 0
    clears_be = bind_lo is not None and bind_lo > BREAKEVEN
    beats_inc = bind_lo is not None and bind_lo > INCUMBENT_MAGW_DOWN_P10  # strictly beat incumbent floor
    nbig = bind_n >= 150

    # stable same-side DOWN edge in BOTH 2024 AND 2026 (single-2025-only or sign-flip => KILL).
    # "same-side DOWN edge" = DOWN-split win > 0.5 (model's DOWN votes are better than coin-flip).
    def edge24_26():
        d24 = down.get("2024", {}).get("win"); d26 = down.get("2026", {}).get("win")
        if d24 is None or d26 is None:
            return False
        return (d24 > 0.5) and (d26 > 0.5)
    stable_2yr = edge24_26()

    d25 = down.get("2025", {})
    d25_lo = d25.get("ci", [None, None])[0]
    paywall_only = (d25_lo is None) or (d25_lo <= BREAKEVEN)

    survives = bool(clears_be and beats_inc and nbig and all_upr_ok and stable_2yr)
    killed = not survives

    verdict = {
        "binding": binding,
        "uprates": uprates, "uprate_tripwire_ok": bool(all_upr_ok),
        "binding_ci_lo_clears_breakeven_0.541": bool(clears_be),
        "binding_ci_lo_beats_incumbent_0.5441": bool(beats_inc),
        "binding_n_ge_150": bool(nbig),
        "stable_DOWN_edge_2024_AND_2026": bool(stable_2yr),
        "2025_down_ci_lo": d25_lo,
        "only_remaining_lever_is_paywalled_risk_reversal": bool(paywall_only),
        "SURVIVES": survives, "KILLED": killed,
        "statement": (
            "RESIDUAL-LABEL DOWN @ H=5 " + ("SURVIVES the pre-registered falsifier." if survives else "KILLED.") +
            f" binding-year={binding['year']} win={binding['win']} ci_lo={bind_lo} (n={bind_n});"
            f" clears breakeven={clears_be}; beats incumbent .5441={beats_inc};"
            f" up-rate tripwire ok={all_upr_ok}; stable DOWN edge 2024&2026={stable_2yr}."
            + ("" if survives else
               " DOWN-rescue family confirmed exhausted on-disk." +
               (" 2025 DOWN-split CI95-lo <= 0.541 => the only remaining DOWN lever is paywalled funded "
                "risk-reversal data." if paywall_only else "")))
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
