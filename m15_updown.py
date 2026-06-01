"""SIDE-SPLIT of the FROZEN 15m book (EURUSD.m15.v1) -> the measured (EURUSD, 15m, UP) and (15m, DOWN) keys.
The 15m book has the program's STRONGEST combined direction edge (0.647 recent / 0.579 CPCV-faithful) but the
side-split was UNTESTED. Deriv forex floor is 1m (verified live 2026-06-01), so 15m is tradeable. Inference-only:
replicate m15_production.backtest's gated selective trades, split by PREDICTED direction, score per year
(2024/2025/2026) moved-bars-only with bootstrap CI95. Deriv-faithful (book-native: mid-to-mid contiguous-900s,
_y moved-only ties-excluded; nonoverlap_chrono 900s). Mirrors m5_updown. Breakeven 0.541.

Usage: python m15_updown.py"""
import json, numpy as np
import m15_production as M15

YEARS = ("2024", "2025", "2026")


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
            out[f"{label}_{name}"] = {"n": n, "acc": round(float(corr.mean()), 4), "ci": [round(lo, 4), round(hi, 4)],
                                      "out_up_rate": round(float(y[sel].mean()), 4)}
    return out


def main():
    p, L, G, C = M15._load()
    per_year = {}
    for yr in YEARS:
        D = M15.load([yr])
        if len(D) == 0:
            continue
        pr = M15._dirproba(p, L, G, C, D); y = D["_y"].astype(int).values
        ts = D.index.values.astype("datetime64[s]").astype("int64")
        g = M15.gate_mask(D, p["bb_width_thr"]); m = g & (np.abs(pr - 0.5) >= p["conf_thr"])
        sel = M15.nonoverlap_chrono(ts, m)
        per_year[yr] = ((pr[sel] > 0.5).astype(int), y[sel])
        del D
    allpred = np.concatenate([per_year[l][0] for l in per_year])
    ally = np.concatenate([per_year[l][1] for l in per_year])
    upr = float(ally.mean())
    print(f"[m15_updown] indep trades={len(allpred)} moved up-rate(all)={upr:.4f} pred-up share={allpred.mean():.3f}", flush=True)
    res = side_eval(per_year)
    for k, v in res.items():
        print(f"  {k:16} n={v['n']:>4} acc={v['acc']} CI={v['ci']} out_up_rate={v['out_up_rate']}", flush=True)
    out = {"book": "EURUSD.m15.v1", "horizon_s": 900, "breakeven": 0.541, "gate": p.get("gate", "15m_bb_width<=q & sess_ny"),
           "settlement": "book-native: mid-to-mid contiguous-900s, _y moved-only ties-excluded; nonoverlap_chrono 900s; per-year CI95",
           "moved_up_rate_all": upr, "pred_up_share": float(allpred.mean()), "per_year_side": res}
    json.dump(out, open("m15_updown_result.json", "w"), indent=1)
    print("[m15_updown] -> m15_updown_result.json", flush=True)


if __name__ == "__main__":
    main()
