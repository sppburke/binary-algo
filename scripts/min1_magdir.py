"""Backlog lever #2: magnitude->direction BRIDGE at 60s, DOWN-split (+ UP).

SCOPE: (EURUSD, 60s, DOWN) primary, (EURUSD, 60s, UP) secondary.  GENERIC method = SWEEP_MATRIX magnitude->direction
       bridge / IDEAS_LOG.  Per-key results -> EURUSD_RESULTS.md §60s + sweeps/EURUSD_1m{,_backlog}.md.  Tier-2 = min1_magdir_result.json.

HYPOTHESIS: |ret| is the ONE certified 60s signal (frozen min1 magnitude model, magAUC ~.787 vs dirAUC ~.510).  IF
the direction side has any edge it should live on LARGE moves (a move with a cause is more directional).  Bridge:
bet DOWN only on bars the FROZEN magnitude model flags as a predicted-large move AND the FROZEN direction ensemble
leans down, NY session only.  This is the cleanest test of "does the 60s direction edge HIDE on large moves?".

MECHANISM vs the null: sign-invariance (arXiv:2512.15720) predicts magnitude gates SIZE not SIGN -> conditioning on
predicted-large-move should add NO direction.  The _redteam_magdir60 control already found 60s cond-acc ~.512-.517
even at the most extreme confidence (gap to .65 is INFORMATIONAL, not coverage).  PRIOR LOW.  Run it because it
directly probes the one place sign could hide, and it SUBSUMES any further "avoid-losers"/confidence-only DOWN gate.

LEAKAGE GUARD: the magnitude model predicts P(|ret60|>=Q) from causal features (no future info); used only as a
TRADE-SELECTION gate, never as a direction feature.  The predicted-large THRESHOLD is FROZEN on VAL (magp quantile),
applied unchanged to 2024/2025/2026 — no per-split peeking, no VAL-acc-max.  Settlement deriv-faithful (MP.prep / wc_ret,
ties LOSE), moved-only, nonoverlap_chrono, per-year boot CI95.

PRE-REGISTERED FALSIFIER (written BEFORE the verdict): KILL the bridge UNLESS the predicted-large-move DOWN side, in
the BINDING 2025 year, at some VAL-frozen mag-quantile gate with n>=100, has bootstrap CI95-lower >= 0.541 AND point
acc > the incumbent 60s DOWN .522.  If it fails -> the 60s direction edge does NOT hide on large moves (sign-invariance
confirmed empirically at the operating point); DOWN exhausted via the magnitude bridge.
"""
import json, time, numpy as np, joblib
import min1_production as MP

BREAKEVEN = 0.541
INCUMBENT_DOWN_2025 = 0.522
MAGQS = (0.0, 0.5, 0.7, 0.8, 0.9, 0.95)            # keep bars with magp >= VAL-quantile(q) (predicted-large); 0.0 = all NY
NY = lambda idx: (np.asarray(idx.hour) >= 13) & (np.asarray(idx.hour) < 22)


def stat(corr):
    if len(corr) < 5: return {"n": int(len(corr)), "acc": None, "ci": [None, None]}
    lo, hi = MP.boot(corr)
    return {"n": int(len(corr)), "acc": round(float(corr.mean()), 4), "ci": [round(lo, 4), round(hi, 4)]}


def main():
    t0 = time.time()
    p, L, G, C, S = MP._load()
    M = joblib.load(MP.art("magnitude.joblib"))
    feats = p["feature_names"]

    # --- freeze predicted-large magp thresholds on VAL (no peeking at test/oos) ---
    bva = MP.load_split("val"); Xv, yv, mv, vv, _, _ = MP.prep(bva)
    magp_v = M.predict_proba(Xv[feats])[:, 1]
    nyv = NY(bva.index) & vv & (mv > 0)
    tau = {q: float(np.quantile(magp_v[nyv], q)) for q in MAGQS}
    print(f"[magdir] VAL magp thresholds {tau} {time.time()-t0:.0f}s", flush=True)

    res = {"DOWN": {}, "UP": {}}
    for sp in ("test", "oos"):
        b = MP.load_split(sp); X, y, mag, valid, ts, idx = MP.prep(b)
        dirp = MP._blend(p, L, G, C, S, X); magp = M.predict_proba(X[feats])[:, 1]
        pred = (dirp > 0.5).astype(int); ny = NY(idx) & valid & (mag > 0); years = np.asarray(idx.year)
        for side in ("DOWN", "UP"):
            sidemask = (dirp < 0.5) if side == "DOWN" else (dirp > 0.5)
            for q in MAGQS:
                g = ny & sidemask & (magp >= tau[q])
                if g.sum() < 40: continue
                tr = MP.nonoverlap_chrono(ts, g)
                for yr in (2024, 2025, 2026):
                    t = tr[years[tr] == yr]
                    if len(t) == 0: continue
                    correct = ((pred[t] == y[t]) & (mag[t] > 0)).astype(float)
                    res[side].setdefault(f"magq{q}", {})[str(yr)] = stat(correct)
        print(f"[magdir] {sp} done {time.time()-t0:.0f}s", flush=True)

    def binding(side):
        best = None
        for q in MAGQS:
            v = res[side].get(f"magq{q}", {}).get("2025")
            if v and v["acc"] is not None and v["n"] >= 100:
                if best is None or v["acc"] > best["acc"]: best = {"magq": q, **v}
        return best
    dwn, up = binding("DOWN"), binding("UP")
    down_surv = bool(dwn and dwn["ci"][0] is not None and dwn["ci"][0] >= BREAKEVEN and dwn["acc"] > INCUMBENT_DOWN_2025)
    verdict = (f"DOWN SURVIVES @60s bridge: 2025 magq{dwn['magq']} acc={dwn['acc']} (n{dwn['n']}) CI-lo {dwn['ci'][0]} "
               f">=0.541 & > incumbent .522 -> ESCALATE to refit-CPCV"
               if down_surv else
               f"DOWN KILLED @60s bridge: best binding-2025 large-move DOWN = "
               f"{dwn['acc'] if dwn else None} (magq{dwn['magq'] if dwn else '-'}, n{dwn['n'] if dwn else 0}, "
               f"CI-lo {dwn['ci'][0] if dwn else None}) does NOT clear 0.541 / beat .522 -> the 60s direction edge "
               f"does NOT hide on large moves; sign-invariance confirmed at the operating point; DOWN exhausted via bridge.")
    out = {"key": "(EURUSD,60s,DOWN) primary / UP secondary", "method": "magnitude->direction bridge (frozen min1 mag+dir)",
           "breakeven": BREAKEVEN, "val_frozen_magp_thresholds": tau,
           "settlement": "deriv-faithful wc_ret (ties LOSE), moved-only, nonoverlap_chrono, per-year CI95",
           "incumbent_60s_DOWN": "0.522/0.522/0.516",
           "per_side_magq": res, "binding_2025": {"DOWN": dwn, "UP": up},
           "falsifier": {"DOWN_survives": down_surv, "verdict": verdict}}
    json.dump(out, open("min1_magdir_result.json", "w"), indent=1)
    print("\n[magdir] DOWN binding-2025:", dwn, "\n[magdir] UP binding-2025:", up, flush=True)
    print("[magdir] FALSIFIER:", verdict, flush=True)
    print("[magdir] -> min1_magdir_result.json", flush=True)


if __name__ == "__main__":
    main()
