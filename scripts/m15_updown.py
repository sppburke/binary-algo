"""SIDE-SPLIT of the FROZEN 15m book EURUSD.m15.v1 -> the measured (EURUSD,15m,UP) and (15m,DOWN) keys.

PREREQUISITE for the 15m goal: 15m has only ever been COMBINED-tested. This replays the frozen book's
backtest gating (inference-only, NO training), takes the independent nonoverlap_chrono(900s) trades, splits
them by PREDICTED direction (pr>0.5 = UP bet, pr<0.5 = DOWN bet), and scores per year (2024/2025/2026)
moved-bars-only with bootstrap CI95 + up-rate tripwire. Mirrors m5_updown.py (5m) / min1_updown (60s).

Two views per side:
  - operating : the deployed frozen operating point (gate AND |pr-0.5|>=conf_thr ; cov~0.02, THIN)
  - gate_only : the comp(bb_width)xNY regime only, no confidence selection (HIGHER-n, statistically powered
                measurement of WHERE the directional edge lives; not the deployed point)

Deriv-faithful: the book's own settlement (15m wall-clock-contiguous close-to-close, _y=(ret>0), tie bars
ret==0 dropped at build). Breakeven 0.541. COMBINED per-year at operating MUST reproduce m15_production backtest.

Usage: python m15_updown.py
"""
import json, numpy as np
import m15_production as M15

YEARS = ("2024", "2025", "2026")
BREAKEVEN = 0.541


def side_eval(per_year):
    out = {}
    for label, (pred, y) in per_year.items():
        for side, name in ((1, "UP"), (0, "DOWN"), (None, "COMBINED")):
            sel = (pred == side) if side is not None else np.ones(len(pred), bool)
            n = int(sel.sum())
            if n < 5:
                out[f"{label}_{name}"] = {"n": n, "acc": None, "ci": [None, None], "out_up_rate": None}
                continue
            corr = (pred[sel] == y[sel]).astype(float)
            lo, hi = M15.boot(corr)
            out[f"{label}_{name}"] = {"n": n, "acc": round(float(corr.mean()), 4),
                                      "ci": [round(lo, 4), round(hi, 4)], "out_up_rate": round(float(y[sel].mean()), 4)}
    return out


def collect(p, L, G, C, view):
    per_year, diag = {}, {}
    for yr in YEARS:
        D = M15.load([yr])
        if len(D) == 0:
            continue
        pr = M15._dirproba(p, L, G, C, D)
        y = D["_y"].astype(int).values
        ts = D.index.values.astype("datetime64[s]").astype("int64")
        g = M15.gate_mask(D, p["bb_width_thr"])
        m = (g & (np.abs(pr - 0.5) >= p["conf_thr"])) if view == "operating" else g
        sel = M15.nonoverlap_chrono(ts, m)
        pred = (pr[sel] > 0.5).astype(int)
        per_year[yr] = (pred, y[sel])
        diag[yr] = {"n": int(len(sel)), "pred_up_share": round(float(pred.mean()), 4) if len(sel) else None,
                    "moved_up_rate": round(float(y[sel].mean()), 4) if len(sel) else None}
        del D
    return per_year, diag


def run_view(p, L, G, C, view):
    per_year, diag = collect(p, L, G, C, view)
    res = side_eval(per_year)
    allpred = np.concatenate([per_year[y][0] for y in per_year])
    ally = np.concatenate([per_year[y][1] for y in per_year])
    print(f"\n===== VIEW: {view} =====", flush=True)
    print(f"  ALL trades={len(allpred)} pred-up-share={allpred.mean():.3f} moved-up-rate={ally.mean():.4f}", flush=True)
    for yr in YEARS:
        for name in ("COMBINED", "UP", "DOWN"):
            v = res.get(f"{yr}_{name}")
            if v is None or v["acc"] is None:
                if v is not None:
                    print(f"  {yr} {name:9} n={v['n']:>4} acc=  NA  (thin)", flush=True)
                continue
            flag = " <-CLEARS" if (v["ci"][0] is not None and v["ci"][0] >= BREAKEVEN) else ""
            print(f"  {yr} {name:9} n={v['n']:>4} acc={v['acc']:.4f} CI95=[{v['ci'][0]:.3f},{v['ci'][1]:.3f}] "
                  f"up_rate={v['out_up_rate']:.3f}{flag}", flush=True)
    return {"view": view, "diag": diag, "per_year_side": res,
            "all_pred_up_share": round(float(allpred.mean()), 4), "all_moved_up_rate": round(float(ally.mean()), 4)}


def main():
    p, L, G, C = M15._load()
    out = {
        "book": "EURUSD.m15.v1", "horizon_min": 15, "breakeven": BREAKEVEN,
        "gate": "15m_bb_width<=%.6g AND sess_ny>0.5" % p["bb_width_thr"], "conf_thr": p["conf_thr"],
        "settlement": "book-native: 15m wall-clock-contiguous close-to-close, _y=(ret>0), ties(ret==0) dropped at "
                      "build; nonoverlap_chrono 900s; moved-only; per-year bootstrap CI95",
        "PRE_REGISTERED_FALSIFIER": {
            "purpose": "Locate where the 15m directional edge lives by side. NOT a certification.",
            "side_LIVE_if": "at operating OR gate_only, the side's BINDING (worst) held-out year moved-acc CI95-lower "
                            "clears 0.541 with n>=50 AND up-rate in [0.47,0.53]. LIVE side -> full (a)-(e) pipeline.",
            "side_DEAD_prelim_if": "even gate_only binding-year point estimate < 0.541 (regime-level no-edge).",
            "integrity_check": "operating COMBINED per-year reproduces m15_production backtest (2024 .689/2025 .582/2026 .663).",
            "prior": "UP-tilt (dip-buy) recurs at 60s/5m but is horizon+regime specific; measure, don't copy. "
                     "DOWN prior dead/marginal elsewhere (rally-selling ~0.52)."
        },
    }
    for view in ("operating", "gate_only"):
        out[view] = run_view(p, L, G, C, view)
    json.dump(out, open("m15_updown_result.json", "w"), indent=1)
    print("\n[m15_updown] -> m15_updown_result.json", flush=True)


if __name__ == "__main__":
    main()
