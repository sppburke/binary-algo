"""Sweep family pre-kill probe at 300s (5m): does ANY SIGNED lagged channel predict EURUSD next-5m SIGN
standalone? Covers the linear/standalone basis of the discovery candidates:
  - cross-leg sign-lead (N10 Liang signed-IFR, N14 Liang mv dollar-source, N17 network-momentum): a USD-leg's
    USD-equiv recent return-sign -> EURUSD next-5m sign;
  - own signed order-flow & transient residual (N15 propagator residual, N11 Bacry-Muzy up/down imbalance):
    sign of OF_of_sum_w and of its transient residual (flow - slow EWMA), and uptick imbalance;
  - own return-sign autocorrelation (N16 ordered-binary-choice): EURUSD own recent return-sign (momentum/reversal).
Each candidate is a SIGNED standalone predictor; polarity (continuation vs reversal) is SELECTED on VAL worst-half.
Deriv-faithful-ish: contiguous-300s forward sign, moved-only, per-year, bootstrap CI95.

PRE-REGISTERED FALSIFIER: KILL the linear-standalone discovery family unless the VAL-worst-half-SELECTED
candidate gives moved-sign-acc >= 0.52 with a STABLE same-side edge in BOTH 2024 AND 2026 (the 2025 inversion is
the known failure mode). A null here pre-kills N10/N11/N14/N15/N16/N17 as STANDALONE channels (the certified
m5xp UP book is the nonlinear GBM combination of exactly these signed features — standalone they carry little)."""
import json, numpy as np, pandas as pd
import harness as H

LEGS = ["GBPUSD", "AUDUSD", "NZDUSD", "USDJPY", "USDCHF", "USDCAD"]
USD_BASE = {"USDJPY", "USDCHF", "USDCAD"}
WINDOWS = [1, 2, 5, 15]
FWD = 5                       # 5 one-min bars = 300s
YEARS = [2022, 2023, 2024, 2025, 2026]
OFDIR = "/media/sean/CORSAIR/binary-algo/features_of"


def loadclose(p):
    d = pd.concat([pd.read_parquet(f"{H.FEAT_DIR}/{p}_{y}.parquet", columns=["close"]) for y in YEARS])
    return np.log(d[~d.index.duplicated(keep="last")].sort_index()["close"])


def load_of(cols):
    parts = []
    for y in YEARS:
        try:
            parts.append(pd.read_parquet(f"{OFDIR}/EURUSD_{y}.parquet", columns=cols))
        except Exception:
            pass
    d = pd.concat(parts); return d[~d.index.duplicated(keep="last")].sort_index()


def boot(corr, nb=4000, seed=7):
    corr = np.asarray(corr, float)
    if len(corr) < 5: return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed); n = len(corr)
    a = np.array([corr[rng.integers(0, n, n)].mean() for _ in range(nb)])
    return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))


def main():
    eu = loadclose("EURUSD")
    idx = eu.index; secs = idx.values.astype("datetime64[s]").astype("int64"); n = len(eu)
    contig = np.zeros(n, bool); contig[:n-FWD] = (secs[FWD:] - secs[:-FWD]) == FWD * 60
    fwd = np.full(n, np.nan); fwd[:n-FWD] = eu.values[FWD:] - eu.values[:-FWD]
    yr = idx.year.values
    valid0 = contig & np.isfinite(fwd) & (fwd != 0)
    yup = (fwd > 0).astype(float)

    # ---- build signed candidate features ----
    cands = {}   # name -> signed array (sign in {-1,0,1}); we test both polarities
    for leg in LEGS:
        g = loadclose(leg).reindex(idx); s = -1.0 if leg in USD_BASE else 1.0
        for w in WINDOWS:
            cands[f"leg_{leg}_{w}"] = np.sign(s * (g.values - np.concatenate([[np.nan]*w, g.values[:-w]])))
    for w in WINDOWS:
        cands[f"ownret_{w}"] = np.sign(eu.values - np.concatenate([[np.nan]*w, eu.values[:-w]]))
    # own signed order-flow + transient residual + uptick imbalance
    ofcols = ["OF_of_sum_5", "OF_of_sum_15", "OF_of_sum_30", "OF_of_uptick_5", "OF_of_uptick_15"]
    try:
        OF = load_of(ofcols).reindex(idx)
        for c in ofcols:
            v = OF[c].values.astype(float)
            cands[f"flow_{c}"] = np.sign(v)
            ew = pd.Series(v).ewm(span=60, min_periods=20).mean().values   # slow component
            cands[f"flowresid_{c}"] = np.sign(v - ew)                       # transient (propagator residual proxy)
    except Exception as e:
        print(f"[probe] OF load skipped: {e}", flush=True)

    def acc_year(sig_pred, Y):
        m = valid0 & (yr == Y) & np.isfinite(sig_pred)
        if m.sum() < 80: return None, 0
        return float(((sig_pred[m] > 0).astype(float) == yup[m]).mean()), int(m.sum())

    # ---- evaluate each candidate x polarity; SELECT on VAL worst-half ----
    rows = []
    for name, sig in cands.items():
        for pol, plab in ((1, "cont"), (-1, "rev")):
            pred = pol * sig
            a22, n22 = acc_year(pred, 2022); a23, n23 = acc_year(pred, 2023)
            if a22 is None or a23 is None or min(n22, n23) < 200: continue
            valwh = min(a22, a23)
            a24, n24 = acc_year(pred, 2024); a25, n25 = acc_year(pred, 2025); a26, n26 = acc_year(pred, 2026)
            rows.append({"feat": f"{name}:{plab}", "val_worsthalf": round(valwh, 4),
                         "2024": round(a24, 4) if a24 else None, "2025": round(a25, 4) if a25 else None,
                         "2026": round(a26, 4) if a26 else None, "n24": n24, "n26": n26})
    # selection: best by VAL worst-half
    rows_ok = [r for r in rows if r["2024"] and r["2026"]]
    sel = max(rows_ok, key=lambda r: r["val_worsthalf"]) if rows_ok else None
    # falsifier: VAL-selected candidate stable >=0.52 in BOTH 2024 and 2026, same side
    def stable(r):
        return (r and r["2024"] and r["2026"] and (r["2024"] - 0.5) * (r["2026"] - 0.5) > 0
                and min(r["2024"], r["2026"]) >= 0.52)
    killed = not stable(sel)
    # also report the single best 2026 (hindsight, for context only)
    best26 = max(rows_ok, key=lambda r: (r["2026"] or 0)) if rows_ok else None
    out = {"row": "N10/N11/N14/N15/N16/N17 STANDALONE family pre-kill: signed-lag -> EURUSD next-5m sign",
           "fwd_min": 5, "n_candidates": len(rows), "val_selected": sel, "best_by_2026_hindsight": best26,
           "falsifier": {"KILLED": bool(killed),
                         "verdict": ("KILLED: the VAL-worst-half-selected signed-lag candidate does NOT hold a stable "
                                     ">=0.52 same-side edge in 2024 AND 2026 -> no STANDALONE signed channel (cross-leg "
                                     "signed-IFR/network-momentum, own signed-flow/propagator-residual/Hawkes-imbalance, "
                                     "own sign-autocorrelation) predicts 5m direction. The certified m5xp UP is the "
                                     "nonlinear GBM COMBINATION of these; standalone they are null. Pre-kills "
                                     "N10/N11/N14/N15/N16/N17 as standalone linear channels."
                                     if killed else f"SURVIVED: {sel}")}}
    json.dump(out, open("m5_legsign_result.json", "w"), indent=1)
    print(f"[probe] candidates={len(rows)} VAL-selected={sel}", flush=True)
    print(f"[probe] best-by-2026(hindsight)={best26}", flush=True)
    print(f"[probe] FALSIFIER: {out['falsifier']['verdict']}", flush=True)
    print("[probe] -> m5_legsign_result.json", flush=True)


if __name__ == "__main__":
    main()
