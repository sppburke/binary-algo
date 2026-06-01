"""Sweep row N8 (DISCOVERED): signed-semivariance-skew sign-conditioning on magnitude-flagged bars. The
magnitude model flags WHEN a big 2m move is coming (certified, AUC .74); test whether the SIGN of recent
realized signed-semivariance skew (RS+ - RS-, a SIGNED quantity unlike sign-invariant RV) predicts the
DIRECTION of the resolution on those big-move bars. §8.2 nulled direction-on-magnitude using raw quartiles;
this uses the RS-skew conditioner specifically. Deriv-faithful 120s, per-year, moved-bars.

PRE-REGISTERED FALSIFIER: KILL unless, on big-move bars, next-2m up-rate differs from 0.50 by >=0.02 with the
SAME sign in 2024 AND 2026 (i.e. sign(RS+ - RS-) carries stable direction)."""
import json, numpy as np, pandas as pd, joblib
import min2_production as M2

W = 300  # realized-semivariance window (seconds)


def year_of(ts):
    return pd.to_datetime(np.asarray(ts), unit="s", utc=True).year.values


def main():
    p = json.load(open("models/min2_EURUSD_strategy.json"))
    feats = p["feature_names"]
    Mmod = joblib.load("models/min2_EURUSD_magnitude.joblib")
    # big-move threshold = top-decile magnitude proba on VAL
    bv = M2.load_split("val"); Xv, yv, magv, vv, tsv, idxv = M2.prep(bv)
    pmv = Mmod.predict_proba(Xv[feats])[:, 1]
    gate_thr = float(np.quantile(pmv[vv], 0.90)); del bv, Xv
    rows = {}
    for sp, yrs in (("test", (2024, 2025)), ("oos", (2026,))):
        b = M2.load_split(sp); X, y, mag, valid, ts, idx = M2.prep(b)
        mid = b["mid"].values.astype(float); r = np.zeros(len(mid)); r[1:] = np.diff(np.log(mid))
        rsp = pd.Series(np.where(r > 0, r * r, 0.0)).rolling(W, min_periods=30).sum().values
        rsm = pd.Series(np.where(r < 0, r * r, 0.0)).rolling(W, min_periods=30).sum().values
        skew = rsp - rsm                                    # signed semivariance skew
        pm = Mmod.predict_proba(X[feats])[:, 1]
        big = valid & (pm >= gate_thr) & np.isfinite(skew) & (mag > 0)
        yr = year_of(ts)
        for Y in yrs:
            for sname, smask in (("skew+", skew > 0), ("skew-", skew < 0)):
                m = big & (yr == Y) & smask
                if m.sum() >= 30:
                    rows[f"{Y}_{sname}"] = {"n": int(m.sum()), "up_rate": round(float(y[m].mean()), 4)}
                else:
                    rows[f"{Y}_{sname}"] = {"n": int(m.sum()), "up_rate": None}
        del b, X
    # falsifier: does sign(skew) split up-rate away from 0.50, same direction in 2024 & 2026?
    def edge(Y):
        sp_, sm_ = rows.get(f"{Y}_skew+", {}), rows.get(f"{Y}_skew-", {})
        if sp_.get("up_rate") is None or sm_.get("up_rate") is None:
            return None
        return sp_["up_rate"] - sm_["up_rate"]              # >0 means skew+ -> more up
    e24, e26 = edge(2024), edge(2026)
    survived = (e24 is not None and e26 is not None and e24 * e26 > 0 and min(abs(e24), abs(e26)) >= 0.04)
    out = {"row": "N8 signed-semivariance-skew sign-conditioning on magnitude bars", "horizon_s": 120,
           "W_sec": W, "mag_gate_decile": 0.90, "per_year_skew_bin": rows,
           "edge_skew+_minus_skew-": {"2024": e24, "2026": e26},
           "falsifier": {"KILLED": (not survived),
                         "verdict": ("KILLED: sign(RS+ - RS-) does not split next-2m up-rate with a stable same-sign "
                                     "edge in 2024 & 2026 on big-move bars (direction stays ~0.50 -> sign-invariance "
                                     "holds even under signed-semivariance conditioning)" if not survived else
                                     "SURVIVED: RS-skew sign-conditions direction on magnitude bars -> NEW LEVER")}}
    json.dump(out, open("min2_rsskew_result.json", "w"), indent=1)
    for k, v in rows.items():
        print(f"  {k}: n={v['n']} up_rate={v['up_rate']}", flush=True)
    print(f"[N8] edge(skew+ - skew-): 2024={e24} 2026={e26}", flush=True)
    print(f"[N8] FALSIFIER: {out['falsifier']['verdict']}", flush=True)
    print("[N8] -> min2_rsskew_result.json", flush=True)


if __name__ == "__main__":
    main()
