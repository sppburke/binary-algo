"""Sweep row N4 (DISCOVERED): Market-Intraday-Momentum term-structure. Tests whether a signed intraday
return (since session-open, or over a recent interval) predicts the SIGN of EURUSD's next-2m move, optionally
conditioned on time-of-day. This is a MOMENTUM hypothesis — opposite to the min2 reversion book — so it's a
genuinely distinct directional channel. Minute screen (cheap). Per year.

PRE-REGISTERED FALSIFIER: KILL unless some signed-interval predictor gives next-2m moved-acc >= 0.52 with the
SAME sign edge in 2024 AND 2026 (stable, not regime-flipped)."""
import json, numpy as np, pandas as pd
import harness as H

FWD = 2
INTERVALS = {"since_open": None, "last30": 30, "last60": 60, "last120": 120}


def main():
    e = pd.concat([pd.read_parquet(f"{H.FEAT_DIR}/EURUSD_{y}.parquet", columns=["close"]) for y in (2024, 2025, 2026)])
    e = np.log(e[~e.index.duplicated(keep="last")].sort_index()["close"])
    idx = e.index
    day = idx.normalize()
    fwd = (e.shift(-FWD) - e).values
    yr = idx.year.values
    hour = idx.hour.values
    # return since day-open (first bar of each UTC day)
    open_px = e.groupby(day).transform("first")
    feats = {}
    feats["since_open"] = (e - open_px).values
    for nm, w in (("last30", 30), ("last60", 60), ("last120", 120)):
        feats[nm] = (e - e.shift(w)).values
    results = []
    for nm, f in feats.items():
        for direction, dname in ((+1, "momentum"), (-1, "reversion")):
            pred_up = (direction * f > 0).astype(float)
            row = {"interval": nm, "mode": dname}
            for Y in (2024, 2025, 2026):
                m = (yr == Y) & np.isfinite(fwd) & (fwd != 0) & np.isfinite(f)
                if m.sum() < 100:
                    row[str(Y)] = None; continue
                yup = (fwd[m] > 0).astype(float)
                row[str(Y)] = round(float((pred_up[m] == yup).mean()), 4)
            results.append(row)
    survivors = [r for r in results if r["2024"] and r["2026"]
                 and ((r["2024"] - 0.5) * (r["2026"] - 0.5) > 0)
                 and min(abs(r["2024"] - 0.5), abs(r["2026"] - 0.5)) >= 0.02]
    killed = len(survivors) == 0
    best = max(results, key=lambda r: (r["2026"] or 0))
    out = {"row": "N4 Market-Intraday-Momentum term-structure", "fwd_min": FWD, "all": results,
           "survivors_24and26": survivors,
           "falsifier": {"KILLED": bool(killed),
                         "verdict": ("KILLED: no signed intraday-interval predictor gives a stable same-sign next-2m "
                                     "edge in 2024 AND 2026 (intraday momentum/reversion is null at 2m granularity)"
                                     if killed else f"SURVIVED: {survivors}")}}
    json.dump(out, open("min2_mim_result.json", "w"), indent=1)
    print(f"[N4] best by 2026: {best}", flush=True)
    print(f"[N4] survivors: {len(survivors)}", flush=True)
    print(f"[N4] FALSIFIER: {out['falsifier']['verdict']}", flush=True)
    print("[N4] -> min2_mim_result.json", flush=True)


if __name__ == "__main__":
    main()
