"""Sweep rows N12 + N13 (5m): time-IRREVERSIBILITY / signed odd-moment as a STANDALONE 5m direction feature.
N13 = time-reversal asymmetry of the signed structure function gamma(tau)=<r_t^2 r_{t-tau}> - <r_t r_{t-tau}^2>
(an ODD functional -> flips under time reversal -> sign-carrying, distinct from the sign-blind generalized Hurst
spectrum). N12 = a sign-decomposed irreversibility proxy: trailing up-run steepness minus down-run steepness
('gains gradual vs losses sudden' as a directional state). Both single-pair EURUSD, causal (trailing window
only). Tested as standalone signed predictors of next-300s sign; polarity (continuation/reversal) SELECTED on
VAL worst-half. The scalar I_W irreversibility was killed 0.503 at 2m; these are the SIGNED variants.

PRE-REGISTERED FALSIFIER: KILL N12/N13 unless the VAL-worst-half-selected feature holds moved-sign-acc >= 0.52
with a STABLE same-side edge in BOTH 2024 AND 2026. Pre-kill N13 if |corr| with the killed N8 RS-skew > 0.7."""
import json, numpy as np, pandas as pd
import harness as H

FWD = 5
YEARS = [2022, 2023, 2024, 2025, 2026]
WINS = [30, 60, 120]      # trailing windows (minutes) for the statistic
TAUS = [1, 3, 5]          # lags for the structure function


def loadret():
    d = pd.concat([pd.read_parquet(f"{H.FEAT_DIR}/EURUSD_{y}.parquet", columns=["close"]) for y in YEARS])
    d = d[~d.index.duplicated(keep="last")].sort_index()
    lp = np.log(d["close"].values); idx = d.index
    return idx, lp


def boot(c, nb=4000, seed=7):
    c = np.asarray(c, float)
    if len(c) < 5: return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed); n = len(c)
    a = np.array([c[rng.integers(0, n, n)].mean() for _ in range(nb)])
    return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))


def main():
    idx, lp = loadret()
    secs = idx.values.astype("datetime64[s]").astype("int64"); n = len(lp)
    contig = np.zeros(n, bool); contig[:n-FWD] = (secs[FWD:] - secs[:-FWD]) == FWD * 60
    fwd = np.full(n, np.nan); fwd[:n-FWD] = lp[FWD:] - lp[:-FWD]
    yr = idx.year.values
    valid0 = contig & np.isfinite(fwd) & (fwd != 0); yup = (fwd > 0).astype(float)
    r = np.zeros(n); r[1:] = np.diff(lp)              # 1-min log returns
    rs = pd.Series(r)

    cands = {}
    # N13: rolling signed structure-function gamma(tau) = E[r_t^2 r_{t-tau}] - E[r_t r_{t-tau}^2]
    for W in WINS:
        for tau in TAUS:
            a = (r ** 2) * np.concatenate([[0]*tau, r[:-tau]])           # r_t^2 * r_{t-tau}
            b = r * np.concatenate([[0]*tau, (r ** 2)[:-tau]])           # r_t   * r_{t-tau}^2
            gamma = pd.Series(a).rolling(W, min_periods=W//2).mean().values - pd.Series(b).rolling(W, min_periods=W//2).mean().values
            cands[f"N13_sf_W{W}_t{tau}"] = np.sign(gamma)
    # N12: sign-decomposed irreversibility proxy = trailing mean(|up returns|) - mean(|down returns|)
    #      (recent up-steepness vs down-steepness; 'gains gradual / losses sudden' -> a directional state)
    up = np.where(r > 0, r, 0.0); dn = np.where(r < 0, -r, 0.0)
    for W in WINS:
        us = pd.Series(up).rolling(W, min_periods=W//2).mean().values
        ds = pd.Series(dn).rolling(W, min_periods=W//2).mean().values
        cands[f"N12_irrev_W{W}"] = np.sign(us - ds)

    def acc_year(pred, Y):
        m = valid0 & (yr == Y) & np.isfinite(pred)
        if m.sum() < 80: return None, 0
        return float(((pred[m] > 0).astype(float) == yup[m]).mean()), int(m.sum())

    rows = []
    for name, sig in cands.items():
        for pol, plab in ((1, "cont"), (-1, "rev")):
            pred = pol * sig
            a22, n22 = acc_year(pred, 2022); a23, n23 = acc_year(pred, 2023)
            if a22 is None or a23 is None or min(n22, n23) < 200: continue
            a24, n24 = acc_year(pred, 2024); a25, n25 = acc_year(pred, 2025); a26, n26 = acc_year(pred, 2026)
            rows.append({"feat": f"{name}:{plab}", "val_worsthalf": round(min(a22, a23), 4),
                         "2024": round(a24, 4) if a24 else None, "2025": round(a25, 4) if a25 else None,
                         "2026": round(a26, 4) if a26 else None, "n26": n26})
    rows_ok = [r0 for r0 in rows if r0["2024"] and r0["2026"]]
    sel = max(rows_ok, key=lambda r0: r0["val_worsthalf"]) if rows_ok else None
    def stable(r0):
        return (r0 and r0["2024"] and r0["2026"] and (r0["2024"]-0.5)*(r0["2026"]-0.5) > 0 and min(r0["2024"], r0["2026"]) >= 0.52)
    killed = not stable(sel)
    best26 = max(rows_ok, key=lambda r0: (r0["2026"] or 0)) if rows_ok else None
    out = {"row": "N12 (sign-decomposed irreversibility) + N13 (signed structure-function) @300s",
           "n_candidates": len(rows), "val_selected": sel, "best_by_2026_hindsight": best26,
           "falsifier": {"KILLED": bool(killed),
                         "verdict": ("KILLED: no signed-irreversibility / odd-moment feature holds a stable >=0.52 same-side "
                                     "edge in 2024 AND 2026 -> time-irreversibility carries no exploitable 5m SIGN (the scalar "
                                     "I_W was already 0.503; the signed variants are no better)."
                                     if killed else f"SURVIVED: {sel}")}}
    json.dump(out, open("m5_irrev_result.json", "w"), indent=1)
    print(f"[irrev] candidates={len(rows)} VAL-selected={sel}", flush=True)
    print(f"[irrev] best-by-2026(hindsight)={best26}", flush=True)
    print(f"[irrev] FALSIFIER: {out['falsifier']['verdict']}", flush=True)
    print("[irrev] -> m5_irrev_result.json", flush=True)


if __name__ == "__main__":
    main()
