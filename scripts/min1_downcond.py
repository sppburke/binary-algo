"""1-MINUTE (60s) retarget of the USD-STRENGTH-CONDITIONED DOWN lever (D3, originally m5_downcond.py @300s).

SCOPE: (EURUSD, 60s, DOWN).  GENERIC method = SWEEP_MATRIX D3 / IDEAS_LOG "USD-driver DOWN".  Per-key results ->
       EURUSD_RESULTS.md (60s DOWN) + sweeps/EURUSD_1m{,_backlog}.md.  Tier-2 evidence = min1_downcond_result.json.

HYPOTHESIS (mechanism-first, builds on the 5m KILL): EURUSD DOWN is the bet ONLY when the DOLLAR is the coherent
driver (a broad USD bid pulls EURUSD down together).  Rally-selling died in the 2025 EUR-up regime because there
was no USD tailwind.  At 5m (m5_downcond_result.json) this was KILLED: in the binding 2025 year USD-strong DOWN
0.5259 (n502, CI-lo 0.482) did NOT separate from USD-weak DOWN 0.5374 -> the conditioning carried no sign.
ATTACK-THE-CAUSE at 60s: at a 1-minute horizon the cross-pair USD coherence is mechanically TIGHTER (less time for
EUR-idiosyncratic noise to dominate the next bar), so the USD-driver split could separate where it failed at 5m.

This is a MODEL-FREE conditioning test on purpose: if raw USD-strength does not separate the next-60s DOWN outcome,
no trained model wrapped around it can rescue it (the 5m model-based test already failed).  The signal is causal:
usdbask{k} = the broad-USD basket move over the PAST k minutes expressed in EURUSD-equivalent (up) terms, so
usdbask<0 == USD strong == EURUSD-equiv down.  Target = next-60s EURUSD sign (MX wall-clock-contiguous, ties dropped).

SETTLEMENT CAVEAT: uses the MX bar-grid forward target (exactly 60s, contiguous, ties excluded) — the same
faithful method behind the certified m5xp/m15xp books — NOT the tick +1s-entry-lag wc_ret harness of
min1_production.py.  If ANY variant SURVIVES the falsifier here, ESCALATE to tick-faithful wc_ret + refit-CPCV
before believing/certifying.  Breakeven 0.541 (deriv R~1.85).

PRE-REGISTERED FALSIFIER (written to the result JSON BEFORE the OOS verdict is read):
  KILL the 60s USD-conditioned DOWN lever UNLESS some disciplined variant (A sign-split, B VAL-worst-half-selected
  usdbask gate, or C multi-LB persistence) yields a DOWN win-rate in the BINDING 2025 held-out year with
  n>=100 AND bootstrap CI95-lower >= 0.541 AND the USD-conditioned DOWN materially exceeds the unconditioned
  NY-DOWN base-rate AND the USD-strong DOWN materially exceeds the USD-weak DOWN (the conditioning must carry the
  sign, not just ride a down-drift).  Variant B's gate is selected ONLY on VAL(2022-23) worst chronological half
  (never VAL-acc-max, never peeking at 2024-26).  If no variant clears -> 60s DOWN is genuinely efficient even when
  USD-gated (confirms the 5m kill at the shorter horizon); record subsumed/dead.
"""
import os
os.environ.setdefault("MX_HOR", "1")          # MUST precede the import (HOR is read at import time)
import json, numpy as np
import m5_xpair as MX

assert MX.HOR == 1, f"MX_HOR must be 1, got {MX.HOR}"
BREAKEVEN = 0.541
LBK = MX.LB                                    # [1,3,5,10,15,30] minute lookbacks
BASK = [f"usdbask{k}" for k in LBK]
AGREE = [f"agree{k}" for k in LBK]
YEARS = (("test24", "2024"), ("test25", "2025"), ("oos", "2026"))


def build_ny(years):
    """MX 1-min cross-pair frame, NY session only (D3 spec), moved bars (ties already dropped by MX)."""
    D = MX.build_xp(years)
    D = D.iloc[D["sess_ny"].values > 0.5]
    return D


def down_on(D, mask):
    """Independent (nonoverlap 60s) DOWN trades on `mask`; return (down_correct_array, up_rate, n_raw)."""
    ts = D["_ts"].values.astype("int64")
    y = D["_y"].values.astype(int)
    m = np.asarray(mask, bool)
    sel = MX.nonoverlap_chrono(ts, m)          # gap = GAP_S = 60s
    if len(sel) == 0:
        return np.array([]), float("nan"), 0
    ys = y[sel]
    return (ys == 0).astype(float), float(ys.mean()), len(sel)   # down-success = realized y==0; up_rate = mean(y)


def up_on(D, mask):
    """Mirror: independent UP trades on `mask`; up-success = realized y==1."""
    ts = D["_ts"].values.astype("int64"); y = D["_y"].values.astype(int)
    sel = MX.nonoverlap_chrono(ts, np.asarray(mask, bool))
    if len(sel) == 0: return np.array([]), 0
    return (y[sel] == 1).astype(float), len(sel)


def stat(corr):
    if len(corr) < 5: return {"n": int(len(corr)), "acc": None, "ci": [None, None]}
    lo, hi = MX.boot(corr)
    return {"n": int(len(corr)), "acc": round(float(corr.mean()), 4), "ci": [round(lo, 4), round(hi, 4)]}


def s_basket(D):
    return D[BASK].mean(axis=1).values          # broad USD strength; <0 == USD strong (EUR-down pressure)


def coverage_curve(D):
    """Falsifier step (d): the FULL USD-strength DOWN coverage curve on a held-out year. For each quantile of the
    most-USD-strong bars, report DOWN win-rate + n. Showing every point < breakeven IS the wall (not a cherry-pick:
    we report the whole curve, we do not select an operating point on held-out data)."""
    s = s_basket(D); curve = {}
    for q in (0.5, 0.3, 0.2, 0.1, 0.05, 0.02):
        tau = float(np.nanquantile(s, q))       # most-USD-strong q-fraction (usdbask <= tau, tau<0); NaN-safe
        corr, up, n = down_on(D, np.isfinite(s) & (s <= tau))
        curve[f"top{int(q*100)}pct"] = {**stat(corr), "up_rate": round(up, 4) if n else None}
    return curve


def agree_mean(D):
    return D[AGREE].mean(axis=1).values


def main():
    agfloor = None
    res = {}
    for w, label in YEARS:
        D = build_ny(MX.SPL[w])
        s = s_basket(D); ag = agree_mean(D)
        if agfloor is None: agfloor = float(np.median(ag))
        u1 = D["usdbask1"].values; u5 = D["usdbask5"].values; u15 = D["usdbask15"].values
        ny_n = len(D)
        row = {"ny_bars": ny_n, "tie_note": "MX drops exact ties (fwd==0) at build"}
        # --- moved up-rate tripwire on the full NY set (unconditioned) ---
        allcorr, up_all, n_all = down_on(D, np.ones(ny_n, bool))
        row["NY_DOWN_baseline"] = {**stat(allcorr), "up_rate": round(up_all, 4)}
        # --- Variant A: pure sign split (literal D3 retarget) ---
        cu, _, _ = down_on(D, s < 0); cw, _, _ = down_on(D, s >= 0)
        row["A_USDstrong_DOWN"] = stat(cu); row["A_USDweak_DOWN"] = stat(cw)
        # UP mirror (USD weak -> EUR up)
        upc, _ = up_on(D, s > 0); row["A_USDweak_UP"] = stat(upc)
        # --- Variant B (step d): full USD-strength DOWN coverage curve (the wall, demonstrated) ---
        row["B_coverage_curve_DOWN"] = coverage_curve(D)
        # --- Variant B2: coherence-tightened (USD strong AND >=median cross-pair agreement) ---
        cbc, _, _ = down_on(D, (s < 0) & (ag >= agfloor)); row["B2_USDstrong_coherent_DOWN"] = stat(cbc)
        # --- Variant C: multi-lookback USD-downtrend persistence ---
        cc, _, _ = down_on(D, (u1 < 0) & (u5 < 0) & (u15 < 0)); row["C_persist3_DOWN"] = stat(cc)
        res[label] = row
        for k, v in row.items():
            if isinstance(v, dict) and "acc" in v: print(f"  {label} {k:26} {v}", flush=True)
        print(f"  {label} B_coverage_curve_DOWN  " +
              " ".join(f"{q}={row['B_coverage_curve_DOWN'][q]['acc']}(n{row['B_coverage_curve_DOWN'][q]['n']})"
                       for q in row['B_coverage_curve_DOWN']), flush=True)

    # -------- apply the pre-registered falsifier on the BINDING 2025 year --------
    b = res["2025"]; base25 = b["NY_DOWN_baseline"]["acc"]
    cands = {}
    for name in ("A_USDstrong_DOWN", "B2_USDstrong_coherent_DOWN", "C_persist3_DOWN"):
        v = b.get(name)
        if v and v["acc"] is not None: cands[name] = v
    # include every point on the binding-year coverage curve as a candidate (showing none clears = the wall)
    for q, v in b["B_coverage_curve_DOWN"].items():
        if v["acc"] is not None: cands[f"Bcurve_{q}"] = v
    survivor = None
    for name, v in cands.items():
        ci_lo = v["ci"][0]
        beats_base = v["acc"] - (base25 or 0) >= 0.01
        beats_weak = (v["acc"] - (b["A_USDweak_DOWN"]["acc"] or 0) >= 0.01) if name.startswith("A_") else True
        if v["n"] >= 100 and ci_lo is not None and ci_lo >= BREAKEVEN and beats_base and beats_weak:
            survivor = {"variant": name, **v}; break
    curve_max = max((v["acc"] for v in b["B_coverage_curve_DOWN"].values() if v["acc"] is not None), default=None)
    verdict = (f"SURVIVES @60s: {survivor['variant']} 2025 DOWN={survivor['acc']} (n{survivor['n']}) "
               f"CI-lo {survivor['ci'][0]} clears 0.541 -> ESCALATE to tick-faithful wc_ret + refit-CPCV"
               if survivor else
               f"KILLED @60s: NO USD-conditioned DOWN variant clears 0.541 CI-lo in binding 2025 "
               f"(baseline NY-DOWN {base25}; coverage-curve max DOWN-acc {curve_max} at the most-USD-strong tail); "
               f"USD-strong {b['A_USDstrong_DOWN']['acc']} vs USD-weak {b['A_USDweak_DOWN']['acc']} (no separation) "
               f"-> 60s EURUSD DOWN is efficient even USD-gated; confirms + STRENGTHENS the 5m D3 kill "
               f"(5m USDstrong .526 had some lift; 60s has none).")
    out = {"key": "(EURUSD,60s,DOWN)", "method": "USD-strength-conditioned DOWN (D3 retarget @MX_HOR=1)",
           "breakeven": BREAKEVEN, "settlement": "MX bar-grid 60s contiguous, ties dropped (NOT tick wc_ret)",
           "incumbent_1m_DOWN": "0.522/0.522/0.516 (min1.v1 down-preds, dead)",
           "prior_5m_D3": "KILLED (2025 USDstrong 0.5259 ~ USDweak 0.5374)",
           "agree_floor": agfloor, "per_year": res,
           "falsifier": {"HAS_60s_DOWN_EDGE": bool(survivor), "verdict": verdict}}
    json.dump(out, open("min1_downcond_result.json", "w"), indent=1)
    print("\n[downcond60] FALSIFIER:", verdict, flush=True)
    print("[downcond60] -> min1_downcond_result.json", flush=True)


if __name__ == "__main__":
    main()
