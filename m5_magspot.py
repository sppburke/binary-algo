"""MAGNITUDE x DIRECTION for the SPOT deployment (2026-06-03) — the combination I initially skipped.

Magnitude is sign-invariant for the binary HIT-RATE (m5_magdyn / sign-invariance theorem) — BUT the SPOT deployment's
binding constraint is different: spot P&L = signed move - spread, and the avg 5m EURUSD move is only ~1.76 pip, so the
spread eats most of it (UP deployable only at <=0.1 pip; DOWN never). Magnitude is the ONE strongly-forecastable quantity
here (AUC ~.74). HYPOTHESIS: gating the certified direction trades to HIGH-predicted-move bars raises the expected move
well above the spread -> lifts spot expectancy / widens the deployable spread. This was NOT run (pmag was disabled in the
combo sweep). RUN it.

Predicts move-size from the SAME m5xp features (leakage-safe: features<=t, target=|fwd5|), gates the certified UP (NY &
meta>=THR & pr>0.5) and DOWN (pr<0.5) trades on predicted-magnitude quantile, evaluates SPOT P&L (signed move - round-trip
spread) per (mag-gate q) x (spread). q selected on WORST-VAL-half spot expectancy.

PRE-REGISTERED FALSIFIER (before OOS): magnitude-gating HELPS iff at the VAL-selected q it makes a WIDER spread robustly
spot-deployable (binding-year, all 3 yrs CI95-lo>0, n>=150) than the ungated q=0 baseline — e.g. makes 0.2 or 0.4 pip
positive where ungated failed. Else: magnitude does not rescue spot (confirms the move-size gain is offset by lower
directional hit-rate on large/jump-driven moves). Expected: marginal — large 5m moves are partly jump/informed (less
sign-predictable), so the move-size gain may be cancelled. RUN to find out.

  ~/binary-algo-venv/bin/python m5_magspot.py
"""
import os, sys; sys.argv = ["x"]
import json, time, numpy as np
import lightgbm as lgb
import m5_xpair as MX, m5_xpair_production as XP
ROOT = "/home/sean/git/binary-algo"; PIP = 0.0001 / 1.16
SPREADS = [0.1, 0.2, 0.4]; QS = [0.0, 0.3, 0.5, 0.7]   # mag-gate: keep top (1-q) by predicted move-size
WINDOWS = (("val", "VAL"), ("test24", 2024), ("test25", 2025), ("oos", 2026))
def boot(c, nb=2500, seed=7):
    c = np.asarray(c, float)
    if len(c) < 5: return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed); n = len(c)
    a = np.array([c[rng.integers(0, n, n)].mean() for _ in range(nb)])
    return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))
def maxdd(eq):
    peak = np.maximum.accumulate(eq); return float(((eq - peak) / peak).min())

def main():
    t0 = time.time()
    json.dump({"test": "magnitude-gate x direction for SPOT deployment", "spreads_pip": SPREADS, "mag_gate_q": QS,
               "status": "PRE-REGISTERED",
               "FALSIFIER": "mag-gating HELPS iff VAL-selected q makes a WIDER spread robustly spot-deployable (all 3 yrs "
                            "CI95-lo>0, n>=150) than ungated q=0. Else magnitude does not rescue spot."},
              open(f"{ROOT}/m5_magspot_result.json", "w"), indent=2)
    p, P, M = XP._load(); cols = p["primary_feats"]; mcols = p["meta_feats"]; THR = p["meta_thr"]
    # train magnitude predictor on TRAIN: target = |fwd5| >= median(|fwd5|) on moved bars (leakage-safe; feats<=t)
    TR = MX.augment(MX.build_xp(XP.SPL["train"], 6), XP.SPL["train"], XP.MODE)
    fwtr = TR["_fwd"].astype("float64").values; amed = float(np.median(np.abs(fwtr[fwtr != 0])))
    ymag = (np.abs(fwtr) >= amed).astype(int)
    magm = lgb.LGBMClassifier(n_estimators=800, learning_rate=0.03, num_leaves=127, min_child_samples=300,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=20, n_jobs=20, verbosity=-1)
    magm.fit(TR[cols].astype("float32"), ymag)
    print(f"[magspot] magnitude GBM trained (|move|>= {amed/PIP:.2f}pip median); train AUC-proxy fit done ({time.time()-t0:.0f}s)", flush=True)
    del TR
    # collect per-window gated trades (both sides) with predicted-magnitude + actual signed move
    data = {}
    for w, lab in WINDOWS:
        D = MX.augment(MX.build_xp(XP.SPL[w]), XP.SPL[w], XP.MODE)
        pr = P.predict(D[cols].astype("float32")); sm = M.predict(XP._Xmeta(D, pr, mcols))
        mag = magm.predict_proba(D[cols].astype("float32"))[:, 1]
        ts = D["_ts"].values.astype("int64"); ny = D["sess_ny"].values > 0.5; fwd = D["_fwd"].astype("float64").values
        for side_up, side in ((True, "UP"), (False, "DOWN")):
            g = ny & (sm >= THR) & ((pr > 0.5) if side_up else (pr < 0.5))
            sel = MX.nonoverlap_chrono(ts, g, 300)
            if len(sel):
                data[(lab, side)] = dict(ts=ts[sel], mv=fwd[sel], mag=mag[sel])
        print(f"[magspot] {lab}: gated UP n={len(data.get((lab,'UP'),{}).get('ts',[]))} DOWN n={len(data.get((lab,'DOWN'),{}).get('ts',[]))} ({time.time()-t0:.0f}s)", flush=True)
        del D
    out = {"test": "magnitude-gate x direction for SPOT", "mag_median_pip": round(amed / PIP, 3), "sides": {}}
    for side_up, side in ((True, "UP"), (False, "DOWN")):
        def spot_pnl(mv, sp): return (mv - sp * PIP) if side_up else (-mv - sp * PIP)
        # VAL worst-half selection of (q, spread-agnostic) by spot expectancy at 0.2pip; report all q x spread forward
        val = data.get(("VAL", side))
        side_out = {"by_q": {}}
        for q in QS:
            grid = {}
            for sp in SPREADS:
                yrs = {}
                for lab in (2024, 2025, 2026):
                    d = data.get((lab, side))
                    if not d: continue
                    thr = np.quantile(d["mag"], q) if q > 0 else -1.0
                    keep = d["mag"] >= thr
                    pnl = spot_pnl(d["mv"][keep], sp)
                    if len(pnl) < 20: continue
                    lo, hi = boot(pnl); eq = np.cumprod(1 + pnl)
                    yrs[lab] = dict(n=int(len(pnl)), exp_pip=round(float(pnl.mean() / PIP), 4),
                                    exp_ci_lo_pip=round(lo / PIP, 4), spot_winrate=round(float((pnl > 0).mean()), 4),
                                    avg_move_pip=round(float(np.abs(d["mv"][keep]).mean() / PIP), 3),
                                    maxDD_1x=round(maxdd(eq), 4))
                if len(yrs) == 3:
                    los = [yrs[y]["exp_ci_lo_pip"] for y in yrs]; ns = [yrs[y]["n"] for y in yrs]
                    grid[f"{sp}pip"] = dict(per_year=yrs, binding_ci_lo=round(min(los), 4), min_n=min(ns),
                                            deployable=bool(min(los) > 0 and min(ns) >= 150))
            # VAL worst-half spot expectancy at this q (0.2pip) for selection
            vsel = None
            if val is not None:
                ts = val["ts"]; half = np.median(ts); accs = []
                for h in (ts < half, ts >= half):
                    thr = np.quantile(val["mag"], q) if q > 0 else -1.0
                    m = h & (val["mag"] >= thr); pnl = spot_pnl(val["mv"][m], 0.2)
                    if m.sum() >= 15: accs.append(float(pnl.mean() / PIP))
                vsel = round(min(accs), 4) if len(accs) == 2 else None
            side_out["by_q"][f"q{q}"] = dict(val_worsthalf_exp02_pip=vsel, forward=grid,
                                             avg_move_at_q_pip=round(float(np.abs(data[(2025, side)]["mv"][data[(2025, side)]["mag"] >= (np.quantile(data[(2025, side)]["mag"], q) if q > 0 else -1)]).mean() / PIP), 3) if (2025, side) in data else None)
        out["sides"][side] = side_out
        # best q by VAL worst-half, then its widest robustly-deployable spread
        cand = [(qd["val_worsthalf_exp02_pip"], qk, qd) for qk, qd in side_out["by_q"].items() if qd["val_worsthalf_exp02_pip"] is not None]
        bestq = max(cand)[1] if cand else None
        side_out["VAL_selected_q"] = bestq
        widest = None
        if bestq:
            for sp in SPREADS:
                if side_out["by_q"][bestq]["forward"].get(f"{sp}pip", {}).get("deployable"): widest = sp
        side_out["widest_deployable_spread_at_selected_q_pip"] = widest
        print(f"[magspot] {side}: VAL-selected {bestq} -> widest deployable spread {widest}pip", flush=True)
    # baseline (q0) widest deployable vs selected-q widest -> did magnitude help?
    def widest_q0(side):
        g = out["sides"][side]["by_q"]["q0.0"]["forward"]; w = None
        for sp in SPREADS:
            if g.get(f"{sp}pip", {}).get("deployable"): w = sp
        return w
    upw0 = widest_q0("UP"); upwq = out["sides"]["UP"]["widest_deployable_spread_at_selected_q_pip"]
    dnw0 = widest_q0("DOWN"); dnwq = out["sides"]["DOWN"]["widest_deployable_spread_at_selected_q_pip"]
    helped = bool((upwq or 0) > (upw0 or 0) or (dnwq or -1) is not None and (dnwq or 0) > (dnw0 or 0) or (dnw0 is None and dnwq is not None))
    out["VERDICT"] = dict(magnitude_helps_spot=helped,
        UP_widest_deployable_spread_pip=dict(ungated_q0=upw0, mag_gated=upwq),
        DOWN_widest_deployable_spread_pip=dict(ungated_q0=dnw0, mag_gated=dnwq),
        statement=(f"Magnitude-gate x direction for SPOT: UP widest deployable spread ungated {upw0}pip -> mag-gated {upwq}pip; "
                   f"DOWN ungated {dnw0} -> mag-gated {dnwq}. "
                   + ("Magnitude-gating WIDENS the deployable spread -> a real deployment improvement; verify w/ CPCV before freeze."
                      if helped else
                      "Magnitude-gating does NOT widen the deployable spread: the larger-move gain is offset by lower directional "
                      "hit-rate on big/jump-driven moves (the move-size and sign-predictability trade off). Spot deployability "
                      "unchanged; magnitude confirmed non-rescuing for direction even via the spot channel.")))
    json.dump(out, open(f"{ROOT}/m5_magspot_result.json", "w"), indent=2, default=str)
    print(f"\nVERDICT: {out['VERDICT']['statement']}\n-> m5_magspot_result.json ({time.time()-t0:.0f}s)", flush=True)

if __name__ == "__main__": main()
