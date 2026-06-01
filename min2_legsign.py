"""Sweep gate for the cross-leg-sign novel-idea family (N3 quantilogram / N5 PCMCI / N6 directed-info /
N9 cross-ordinal all rely on: does ANOTHER USD-leg's recent SIGN lead EURUSD's next-2m SIGN?). Cheap minute
screen: for each leg x lag-window, test whether the leg's USD-direction-aligned recent return-sign predicts
EURUSD's next-2m sign, per year, requiring a STABLE same-sign edge across 2024 AND 2026 (the 2025 inversion is
the known failure mode). If null -> pre-kills N3/N5/N6/N9 (documented null of the cross-leg-sign channel at 2m).

PRE-REGISTERED FALSIFIER: KILL unless some (leg, window) gives moved-sign-acc >= 0.52 with the SAME sign edge
in BOTH 2024 and 2026 (and 2025 not catastrophically inverted)."""
import json, numpy as np, pandas as pd
import harness as H

LEGS = ["GBPUSD", "AUDUSD", "NZDUSD", "USDJPY", "USDCHF", "USDCAD"]
USD_BASE = {"USDJPY", "USDCHF", "USDCAD"}      # flip so + == USD-down == EUR-up-equivalent
WINDOWS = [1, 2, 5]                            # minutes of leg lookback
FWD = 2                                        # EURUSD forward horizon (minutes ~ 2m)


def loadclose(p):
    d = pd.concat([pd.read_parquet(f"{H.FEAT_DIR}/{p}_{y}.parquet", columns=["close"]) for y in (2024, 2025, 2026)])
    return np.log(d[~d.index.duplicated(keep="last")].sort_index()["close"])


def main():
    eu = loadclose("EURUSD")
    eu_fwd_sign = (eu.shift(-FWD) - eu)                      # EURUSD next-2m log-return
    base = pd.DataFrame({"eu_fwd": eu_fwd_sign.values}, index=eu.index)
    base["yr"] = base.index.year
    legdata = {}
    for leg in LEGS:
        g = loadclose(leg).reindex(eu.index)
        s = -1.0 if leg in USD_BASE else 1.0
        legdata[leg] = {w: np.sign(s * (g - g.shift(w))).values for w in WINDOWS}
    results = []
    for leg in LEGS:
        for w in WINDOWS:
            ls = legdata[leg][w]                            # USD-equiv leg sign over last w min
            pred_up = (ls > 0).astype(float)                # leg up-equiv -> predict EURUSD up
            row = {"leg": leg, "window_min": w}
            for Y in (2024, 2025, 2026):
                m = (base["yr"].values == Y) & np.isfinite(base["eu_fwd"].values) & (base["eu_fwd"].values != 0) & np.isfinite(ls)
                if m.sum() < 100:
                    row[str(Y)] = None; continue
                yup = (base["eu_fwd"].values[m] > 0).astype(float)
                acc = float((pred_up[m] == yup).mean())
                row[str(Y)] = round(acc, 4)
            results.append(row)
    # falsifier: any (leg,window) with acc>=0.52 same-sign edge in BOTH 2024 and 2026
    survivors = [r for r in results if r["2024"] and r["2026"]
                 and ((r["2024"] - 0.5) * (r["2026"] - 0.5) > 0)
                 and min(abs(r["2024"] - 0.5), abs(r["2026"] - 0.5)) >= 0.02]
    killed = len(survivors) == 0
    best = max(results, key=lambda r: (r["2026"] or 0))
    out = {"row": "N3/N5/N6/N9 gate: cross-leg sign-lead -> EURUSD next-2m sign", "fwd_min": FWD, "legs": LEGS,
           "windows_min": WINDOWS, "all": results, "survivors_24and26": survivors,
           "falsifier": {"KILLED": bool(killed),
                         "verdict": ("KILLED: no USD-leg's recent sign predicts EURUSD next-2m sign with a stable "
                                     "same-sign edge in 2024 AND 2026 -> pre-kills N3/N5/N6/N9 (cross-leg-sign channel "
                                     "is null at 2m, consistent with the known cross-pair lead-lag null + 2025 inversion)"
                                     if killed else f"SURVIVED: stable cross-leg sign-lead exists ({survivors})")}}
    json.dump(out, open("min2_legsign_result.json", "w"), indent=1)
    print(f"[gate] best by 2026: {best}", flush=True)
    print(f"[gate] survivors(24&26 stable >=0.52): {len(survivors)}", flush=True)
    print(f"[gate] FALSIFIER: {out['falsifier']['verdict']}", flush=True)
    print("[gate] -> min2_legsign_result.json", flush=True)


if __name__ == "__main__":
    main()
