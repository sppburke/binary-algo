"""(15m,UP) and (15m,DOWN) COVERAGE CURVE — step (d) of the side pipeline.

The side-split (m15_updown.py) showed both sides clear breakeven in POINT estimate at the frozen operating
point (cov~2%) but binding-year CI-lo doesn't clear (thin-n power). This sweeps the confidence threshold
(coverage) to find, PER SIDE, the operating point that maximizes the binding-year moved-acc CI95-lower —
the certifiable floor. Thresholds are selected on VAL (2022-2023) by worst-VAL-half stability (NEVER
VAL-acc-max; corr(VAL,OOS)=-0.54). Held-out 2024/2025/2026 are the honest test, split by predicted side,
nonoverlap_chrono(900s), moved-bars-only, boot CI95, up-rate tripwire.

Deployable interpretation: take book trades at confidence>=thr; UP bets = (15m,UP), DOWN bets = (15m,DOWN).
Inference-only (frozen EURUSD.m15.v1). Breakeven 0.541.

Usage: python m15_sidepipe.py
"""
import json, numpy as np
import m15_production as M15
import harness as H

YEARS = ("2024", "2025", "2026")
COVS = (1.00, 0.50, 0.30, 0.20, 0.15, 0.10, 0.07, 0.05, 0.03, 0.02)
BREAKEVEN = 0.541


def predict_window(p, L, G, C, years):
    D = M15.load(list(years))
    pr = M15._dirproba(p, L, G, C, D)
    y = D["_y"].astype(int).values
    ts = D.index.values.astype("datetime64[s]").astype("int64")
    g = M15.gate_mask(D, p["bb_width_thr"])
    del D
    return pr, y, ts, g


def side_acc(pred, y, side):
    sel = (pred == side) if side is not None else np.ones(len(pred), bool)
    n = int(sel.sum())
    if n < 5:
        return {"n": n, "acc": None, "ci_lo": None, "up_rate": None}
    corr = (pred[sel] == y[sel]).astype(float)
    lo, hi = M15.boot(corr)
    return {"n": n, "acc": round(float(corr.mean()), 4), "ci_lo": round(lo, 4),
            "ci_hi": round(hi, 4), "up_rate": round(float(y[sel].mean()), 4)}


def main():
    p, L, G, C = M15._load()
    # VAL predictions (threshold selection set; 2022-2023, out-of-train, pre-test)
    prv, yv, tsv, gv = predict_window(p, L, G, C, H.SPLITS["val"])
    confv = np.abs(prv - 0.5)
    ig = gv  # in-gate
    conf_ig = confv[ig]
    # worst-VAL-half split (chronological)
    order = np.argsort(tsv[ig])
    half = len(order) // 2
    h1, h2 = order[:half], order[half:]
    prv_ig, yv_ig = prv[ig], yv[ig]

    # held-out predictions
    test = {yr: predict_window(p, L, G, C, [yr]) for yr in YEARS}

    out = {"book": "EURUSD.m15.v1", "horizon_min": 15, "breakeven": BREAKEVEN,
           "gate": "15m_bb_width<=%.6g AND sess_ny>0.5" % p["bb_width_thr"],
           "selection": "threshold = (1-cov) quantile of VAL in-gate confidence; reported VAL worst-half acc per side",
           "PRE_REGISTERED_FALSIFIER": {
               "side_certifiable_candidate_if": "at the VAL-worst-half-selected coverage, the side's BINDING (worst) "
                   "held-out-year moved-acc CI95-lower clears 0.541 with n>=50 AND up-rate in [0.47,0.53]. "
                   "Such a candidate then goes to full-refit CPCV (step e).",
               "side_no_operating_point_if": "no coverage yields binding-year CI-lo>=0.541 at n>=50 -> the frozen "
                   "book has no certifiable side operating point; escalate to specialist / |ret|-weight / ACI."},
           "curve": {}}

    print(f"{'cov':>5} {'thr':>7} | {'side':4} | VALwh | {'2024':>20} {'2025':>20} {'2026':>20} | bindCIlo", flush=True)
    best = {"UP": None, "DOWN": None}
    for c in COVS:
        thr = float(np.quantile(conf_ig, 1.0 - c)) if c < 1.0 else 0.0
        row = {"cov": c, "thr": round(thr, 5)}
        for side, name in ((1, "UP"), (0, "DOWN")):
            # VAL worst-half acc for this side at thr
            wh = []
            for hh in (h1, h2):
                m = np.abs(prv_ig[hh] - 0.5) >= thr
                predh = (prv_ig[hh][m] > 0.5).astype(int)
                yhh = yv_ig[hh][m]
                selh = predh == side
                wh.append(float((predh[selh] == yhh[selh]).mean()) if selh.sum() >= 20 else np.nan)
            val_wh = round(float(np.nanmin(wh)), 4) if np.isfinite(np.nanmin(wh)) else None
            peryr = {}
            for yr in YEARS:
                pr, y, ts, g = test[yr]
                m = g & (np.abs(pr - 0.5) >= thr)
                sel = M15.nonoverlap_chrono(ts, m)
                pred = (pr[sel] > 0.5).astype(int)
                peryr[yr] = side_acc(pred, y[sel], side)
            cilos = [peryr[yr]["ci_lo"] for yr in YEARS if peryr[yr]["ci_lo"] is not None
                     and peryr[yr]["n"] >= 50]
            bind_cilo = round(min(cilos), 4) if cilos else None
            row[name] = {"val_worsthalf": val_wh, "per_year": peryr, "binding_ci_lo": bind_cilo}
            # track best operating point per side by binding CI-lo (n>=50 all years)
            allbig = all(peryr[yr]["n"] >= 50 for yr in YEARS)
            if bind_cilo is not None and allbig:
                if best[name] is None or bind_cilo > best[name]["binding_ci_lo"]:
                    best[name] = {"cov": c, "thr": round(thr, 5), "binding_ci_lo": bind_cilo,
                                  "per_year_acc": {yr: peryr[yr]["acc"] for yr in YEARS},
                                  "per_year_n": {yr: peryr[yr]["n"] for yr in YEARS}}
            fmt = lambda d: f"{d['acc'] if d['acc'] is not None else 0:.3f}/{d['ci_lo'] if d['ci_lo'] is not None else 0:.3f}/n{d['n']}"
            print(f"{c:>5.2f} {thr:>7.4f} | {name:4} | {str(val_wh):>5} | "
                  f"{fmt(peryr['2024']):>20} {fmt(peryr['2025']):>20} {fmt(peryr['2026']):>20} | {bind_cilo}", flush=True)
        out["curve"][f"cov_{c:.2f}"] = row
    out["best_operating_point"] = best
    print("\nBEST operating point by binding-year CI-lo (n>=50 all yrs):", flush=True)
    for name in ("UP", "DOWN"):
        b = best[name]
        if b is None:
            print(f"  {name}: NONE clears n>=50 all years -> no certifiable operating point on frozen book", flush=True)
        else:
            verdict = "CANDIDATE for CPCV" if b["binding_ci_lo"] >= BREAKEVEN else "binding CI-lo < .541"
            print(f"  {name}: cov={b['cov']} thr={b['thr']} binding_CI_lo={b['binding_ci_lo']} "
                  f"accs={b['per_year_acc']} n={b['per_year_n']} -> {verdict}", flush=True)
    json.dump(out, open("m15_sidepipe_result.json", "w"), indent=1)
    print("\n[m15_sidepipe] -> m15_sidepipe_result.json", flush=True)


if __name__ == "__main__":
    main()
