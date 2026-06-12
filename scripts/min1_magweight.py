"""Backlog lever #1: |return|-weighted (POW=0.5 "magweight") retrain at 60s, DOWN-split (+ UP).

SCOPE: (EURUSD, 60s, DOWN) primary, (EURUSD, 60s, UP) secondary.  GENERIC method = SWEEP_MATRIX Tier-I |ret|-weighted
       loss / IDEAS_LOG magweight.  Per-key results -> EURUSD_RESULTS.md §60s + sweeps/EURUSD_1m{,_backlog}.md.
       Tier-2 evidence = min1_magweight_result.json.

RETARGET of m5_magweight_cpcv.py (POW=0.5), the ONLY lever that ever gave the 5m DOWN side a (razor-thin)
certification (cov0.05 p10 .5441 — though the final 5m verdict was "DOWN genuinely exhausted, not robust;
seed-ensemble did not rescue it").  SUBSTRATE CHOICE (learning from D3a, 2026-06-03): the 5m version rode the
CROSS-PAIR matrix, but cross-pair/USD conditioning is now KILLED at 60s (min1_downcond.py + EURUSD_RESULTS row 3),
so I retarget magweight onto the min1 TICK microstructure features — the substrate where the 60s UP-filter's signal
actually lives.  If |ret|-weighting cannot unlock DOWN on the BEST 60s substrate, that is the decisive kill.

MECHANISM (why it could carry sign, vs the sign-invariance null): magweight is NOT a magnitude FEATURE (which gates
size not sign — theorem arXiv:2512.15720); it RE-WEIGHTS the training loss toward large-move bars, on the
hypothesis that direction is more learnable on big moves (a move with a cause).  The weight uses the realized |ret|
only as a TRAIN sample weight (label-time info, never fed at inference) — no leakage.  PRIOR LOW: the _redteam_magdir60
control already showed 60s cond-acc ~.512-.517 even at the most extreme confidence (gap to .65 is informational,
dirAUC .510 vs magAUC .787) — large-move bars carry little extra DIRECTION at 60s.  Run it because it is the
closest-to-working on-disk DOWN lever; expect to confirm the null fast.

SETTLEMENT: deriv-faithful via min1_production.prep / wc_ret (mid-to-mid, next-tick entry +1s, ties LOSE), moved-bars
only, nonoverlap_chrono (gap=HS+TOL), per-year boot CI95.  Same tick splits the incumbent .522/.522/.516 was measured on.

PRE-REGISTERED FALSIFIER (written to the result JSON BEFORE the verdict is read):
  KILL the DOWN-rescue UNLESS the magweighted DOWN side, in the BINDING 2025 held-out year, at some confidence cover
  in {0.05,0.10,0.15} with n>=100, has bootstrap CI95-lower >= 0.541 AND point acc > the incumbent 60s DOWN .522.
  KILL the UP-improve UNLESS magweighted UP beats the incumbent up-filter floor .520 / OOS .613 on the worst held-out
  year with CI-lower clearing 0.541.  Forward-split first; escalate to nested-refit CPCV ONLY if DOWN forward survives.
  If DOWN fails -> 60s DOWN is exhausted on-disk under loss-reweighting too (the closest-to-working lever); record DEAD.
"""
import os, json, time, numpy as np
import lightgbm as lgb
import min1_production as MP

POW = float(os.environ.get("MIN1_POW", "0.5"))
COVS = (0.05, 0.10, 0.15)
BREAKEVEN = 0.541
INCUMBENT_DOWN_2025 = 0.522
NY = lambda idx: (np.asarray(idx.hour) >= 13) & (np.asarray(idx.hour) < 22)    # NY session (UTC), matches MX sess_ny


def magweight(absret, pow_=POW):
    pos = absret[absret > 0]
    med = float(np.median(pos)) if len(pos) else 1.0
    return np.clip((absret / (med + 1e-12)) ** pow_, 0.1, 10.0)


def stat(corr):
    if len(corr) < 5: return {"n": int(len(corr)), "acc": None, "ci": [None, None]}
    lo, hi = MP.boot(corr)
    return {"n": int(len(corr)), "acc": round(float(corr.mean()), 4), "ci": [round(lo, 4), round(hi, 4)]}


def side_cover(pr, y, mag, ts, ny, idx, side):
    """side='DOWN'(pred<0.5) or 'UP'(pred>0.5): per-cover independent-trade accuracy, split by calendar year."""
    pred = (pr > 0.5).astype(int); conf = np.abs(pr - 0.5)
    gate0 = ny & (mag > 0) & ((pr < 0.5) if side == "DOWN" else (pr > 0.5))
    years = np.asarray(idx.year)
    out = {}
    for c in COVS:
        g = gate0.copy()
        if g.sum() < 40:
            out[f"cov{c}"] = {"_": "thin"}; continue
        cthr = np.quantile(conf[g], 1 - c); sel_mask = g & (conf >= cthr)
        tr = MP.nonoverlap_chrono(ts, sel_mask)
        per = {}
        for yr in (2024, 2025, 2026):
            t = tr[years[tr] == yr]
            if len(t) == 0: continue                       # emit only years present in THIS split (avoid clobber on merge)
            correct = ((pred[t] == y[t]) & (mag[t] > 0)).astype(float)
            per[str(yr)] = stat(correct)
        out[f"cov{c}"] = per
    return out


def main():
    t0 = time.time()
    btr, bva = MP.load_split("train"), MP.load_split("val")
    Xtr, ytr, mtr, vtr, _, _ = MP.prep(btr)
    Xva, yva, mva, vva, _, _ = MP.prep(bva)
    iall = np.where(vtr & (mtr > 0))[0][::MP.TRSTRIDE_ALL]
    XA, yA, wA = Xtr.iloc[iall], ytr[iall], magweight(mtr[iall])
    iva = np.where(vva & (mva > 0))[0]
    print(f"[magw60] prep {time.time()-t0:.0f}s; train n={len(iall):,} POW={POW}", flush=True)
    cache = f"/tmp/min1_magw_pow{POW}.txt"
    if os.path.exists(cache):
        L = lgb.Booster(model_file=cache); print(f"[magw60] loaded cached model {cache}", flush=True)
        predict = lambda Xv: L.predict(Xv)
    else:
        clf = MP.mk_lgb()
        clf.fit(XA, yA, sample_weight=wA, eval_set=[(Xva.iloc[iva], yva[iva])], eval_metric="auc",
                callbacks=[lgb.early_stopping(120), lgb.log_evaluation(0)])
        clf.booster_.save_model(cache); L = clf
        print(f"[magw60] fit {time.time()-t0:.0f}s best_iter={L.best_iteration_} -> cached", flush=True)
        predict = lambda Xv: L.predict_proba(Xv)[:, 1]

    # evaluate on test(2024H2-2025) + oos(2026), split by calendar year
    res = {"DOWN": {}, "UP": {}}
    for sp in ("test", "oos"):
        b = MP.load_split(sp); X, y, mag, valid, ts, idx = MP.prep(b)
        pr = predict(X.values); ny = NY(idx) & valid
        for side in ("DOWN", "UP"):
            sc = side_cover(pr, y, mag, ts, ny, idx, side)
            for cov, v in sc.items():
                res[side].setdefault(cov, {}).update({k: vv for k, vv in v.items()})
        print(f"[magw60] {sp} done {time.time()-t0:.0f}s", flush=True)

    # -------- pre-registered falsifier on BINDING 2025 --------
    def binding(side):
        best = None
        for c in COVS:
            v = res[side].get(f"cov{c}", {}).get("2025")
            if v and v["acc"] is not None and v["n"] >= 100:
                if best is None or v["acc"] > best["acc"]: best = {"cov": c, **v}
        return best
    dwn = binding("DOWN"); up = binding("UP")
    down_surv = bool(dwn and dwn["ci"][0] is not None and dwn["ci"][0] >= BREAKEVEN and dwn["acc"] > INCUMBENT_DOWN_2025)
    verdict = (f"DOWN SURVIVES forward @60s: 2025 cov{dwn['cov']} acc={dwn['acc']} (n{dwn['n']}) CI-lo {dwn['ci'][0]} "
               f">=0.541 & > incumbent .522 -> ESCALATE to nested-refit CPCV"
               if down_surv else
               f"DOWN KILLED @60s: best binding-2025 magweight DOWN = "
               f"{dwn['acc'] if dwn else None} (cov{dwn['cov'] if dwn else '-'}, n{dwn['n'] if dwn else 0}, "
               f"CI-lo {dwn['ci'][0] if dwn else None}) does NOT clear 0.541 / beat incumbent .522 -> "
               f"|ret|-weighting does not unlock 60s DOWN on the tick substrate; DOWN exhausted on-disk.")
    out = {"key": "(EURUSD,60s,DOWN) primary / UP secondary", "method": f"min1 tick magweight POW={POW} retrain",
           "breakeven": BREAKEVEN, "POW": POW, "substrate": "min1 tick microstructure (62 feats)",
           "settlement": "deriv-faithful wc_ret (ties LOSE), moved-only, nonoverlap_chrono, per-year CI95",
           "incumbent_60s_DOWN": "0.522/0.522/0.516", "incumbent_60s_UP": "0.520/0.584/0.613 (filter)",
           "prior_5m_magweight": "DOWN cov0.05 p10 .5441 razor-thin, not robust (seed-ens failed); UP cert .553",
           "per_side_cover": res, "binding_2025": {"DOWN": dwn, "UP": up},
           "falsifier": {"DOWN_forward_survives": down_surv, "verdict": verdict}}
    json.dump(out, open("min1_magweight_result.json", "w"), indent=1)
    print("\n[magw60] DOWN binding-2025:", dwn, flush=True)
    print("[magw60] UP   binding-2025:", up, flush=True)
    print("[magw60] FALSIFIER:", verdict, flush=True)
    print("[magw60] -> min1_magweight_result.json", flush=True)


if __name__ == "__main__":
    main()
