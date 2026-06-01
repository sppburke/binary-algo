"""Sweep row N7 (DISCOVERED): asymmetric up/down-tick intensity imbalance (Hawkes-proxy). Compute signed
mid-tick self-excitation: up-intensity vs down-intensity (EWMA of up/down 1s-tick indicators) and the run-length
asymmetry; test whether the signed intensity predicts EURUSD's next-2m SIGN. Deriv-faithful 120s, per-year.

PRE-REGISTERED FALSIFIER: KILL unless sign(up_int - down_int) [momentum or reversion] gives moved-acc >= 0.52
with the SAME sign edge in 2024 AND 2026."""
import json, numpy as np, pandas as pd
import min2_production as M2

SPANS = [30, 60, 120]  # intensity EWMA spans (seconds)


def year_of(ts):
    return pd.to_datetime(np.asarray(ts), unit="s", utc=True).year.values


def main():
    feat_results = []
    per = {}
    for sp, yrs in (("test", (2024, 2025)), ("oos", (2026,))):
        b = M2.load_split(sp); mid = b["mid"].values.astype(float)
        ts = b.index.values.astype("datetime64[s]").astype("int64")
        ret, valid = M2.wc_ret(ts, mid, 120, M2.TOL_S, M2.ENTRY_LAG_S)
        y = (ret > 0).astype(int); mag = np.abs(ret)
        d = np.sign(np.diff(mid, prepend=mid[0]))            # +1 up-tick, -1 down-tick, 0 flat
        up = (d > 0).astype(float); dn = (d < 0).astype(float)
        yr = year_of(ts)
        for S in SPANS:
            ui = pd.Series(up).ewm(span=S).mean().values
            di = pd.Series(dn).ewm(span=S).mean().values
            sig = ui - di                                   # signed intensity imbalance
            for mode, mname in ((+1, "momentum"), (-1, "reversion")):
                pred = (mode * sig > 0).astype(int)
                for Y in yrs:
                    m = valid & (yr == Y) & (mag > 0) & np.isfinite(sig)
                    if m.sum() >= 200:
                        per[f"S{S}_{mname}_{Y}"] = round(float((pred[m] == y[m]).mean()), 4)
        del b
    # falsifier: any (S,mode) with same-sign edge >=0.52 in 2024 AND 2026
    survivors = []
    for S in SPANS:
        for mname in ("momentum", "reversion"):
            a24 = per.get(f"S{S}_{mname}_2024"); a26 = per.get(f"S{S}_{mname}_2026")
            if a24 and a26 and (a24 - 0.5) * (a26 - 0.5) > 0 and min(abs(a24 - 0.5), abs(a26 - 0.5)) >= 0.02:
                survivors.append({"span": S, "mode": mname, "2024": a24, "2026": a26})
    killed = len(survivors) == 0
    best = max(per.items(), key=lambda kv: kv[1]) if per else (None, None)
    out = {"row": "N7 asymmetric tick-intensity imbalance (Hawkes-proxy)", "horizon_s": 120, "spans_s": SPANS,
           "per_config": per, "survivors_24and26": survivors,
           "falsifier": {"KILLED": bool(killed),
                         "verdict": ("KILLED: signed tick-intensity imbalance carries no stable same-sign 2m edge in "
                                     "2024 AND 2026 (tick self-excitation decays before 120s, as the raw-imbalance "
                                     "decay curve predicted)" if killed else f"SURVIVED: {survivors}")}}
    json.dump(out, open("min2_hawkes_result.json", "w"), indent=1)
    print(f"[N7] best config: {best}", flush=True)
    print(f"[N7] survivors: {len(survivors)}", flush=True)
    print(f"[N7] FALSIFIER: {out['falsifier']['verdict']}", flush=True)
    print("[N7] -> min2_hawkes_result.json", flush=True)


if __name__ == "__main__":
    main()
