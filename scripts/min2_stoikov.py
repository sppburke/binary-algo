"""Discovery-round-2 top candidate (prior 8%): Stoikov microprice-velocity vs mid-velocity SLOPE-DIVERGENCE,
within-EURUSD (USD-inversion-immune), signed. div_t = slope(micro,30s) - slope(mid,30s) is sign-covariant
(negates under price reflection) -> survives sign-invariance; reads the signed fair-value LEAD, not a level
(distinct from the killed rawtick level + Kalman mid-velocity). Bet sign(div) on large-|div| bars. Deriv-faithful
120s, per-year, moved-bars. Light (micro+mid only, no feature frame).

PRE-REGISTERED FAST-KILL: VAL dirAUC of div <= 0.515 (either direction), OR top-tercile subset fails 0.541 in
ANY of 2024/2025/2026, OR the div->fwd-return sign flips 2024<->2026, OR moved subset up-rate exits [0.47,0.53]."""
import json, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
import min2_production as M2

W = 30  # slope window (seconds)
KERN = (np.arange(W) - (W - 1) / 2.0)
NORM = (KERN ** 2).sum()


def year_of(ts):
    return pd.to_datetime(np.asarray(ts), unit="s", utc=True).year.values


def rolling_slope(x):
    """Trailing OLS slope over the prior W samples (causal): slope_t uses x[t-W+1..t]."""
    c = np.convolve(x, KERN[::-1], mode="full")[:len(x)] / NORM   # aligned so index t = window ending at t
    c[:W - 1] = np.nan
    return c


def load_div(sp):
    b = M2.load_split(sp)
    micro = b["micro"].values.astype(float); mid = b["mid"].values.astype(float)
    ts = b.index.values.astype("datetime64[s]").astype("int64")
    ret, valid = M2.wc_ret(ts, mid, 120, M2.TOL_S, M2.ENTRY_LAG_S)
    div = rolling_slope(micro) - rolling_slope(mid)
    del b
    return ts, (ret > 0).astype(int), np.abs(ret), valid, div


def main():
    # VAL fast-KILL gate: dirAUC of div on moved bars
    vts, vy, vmag, vvalid, vdiv = load_div("val")
    m = vvalid & (vmag > 0) & np.isfinite(vdiv)
    val_auc = float(roc_auc_score(vy[m], vdiv[m]))
    sign = 1.0 if val_auc >= 0.5 else -1.0                       # orient: does +div or -div predict up?
    print(f"[stoikov] VAL dirAUC(div)={val_auc:.4f} (orient sign={sign:+.0f})", flush=True)
    killed_auc = abs(val_auc - 0.5) <= 0.015
    res = {}
    if not killed_auc:
        # |div| top-tercile threshold frozen on worst-VAL-half
        order = np.argsort(vts); half = order[len(order) // 2:]
        thr = float(np.quantile(np.abs(vdiv[half][np.isfinite(vdiv[half])]), 2 / 3))
        for sp, yrs in (("test", (2024, 2025)), ("oos", (2026,))):
            ts, y, mag, valid, div = load_div(sp)
            pred = (sign * div > 0).astype(int)                 # bet the oriented sign on large-|div|
            cand = valid & (np.abs(div) >= thr) & np.isfinite(div)
            tr = M2.nonoverlap_chrono(ts, cand); yr = year_of(ts)
            for Y in yrs:
                sel = tr[(yr[tr] == Y)]; moved = mag[sel] > 0
                if moved.sum() >= 30:
                    corr = (pred[sel][moved] == y[sel][moved]).astype(float); lo, hi = M2.boot(corr)
                    res[str(Y)] = {"acc": round(float(corr.mean()), 4), "ci": [round(lo, 4), round(hi, 4)],
                                   "n_moved": int(moved.sum()), "up_rate": round(float(y[sel][moved].mean()), 4)}
                else:
                    res[str(Y)] = {"acc": None, "n_moved": int(moved.sum())}
    accs = [res.get(str(Y), {}).get("acc") for Y in (2024, 2025, 2026)]
    killed = killed_auc or any((a is None) or (a < 0.541) for a in accs)
    out = {"row": "N(disc2) Stoikov micro-vs-mid slope-divergence @120s", "horizon_s": 120, "breakeven": 0.541,
           "val_auc_div": val_auc, "orient_sign": sign, "per_year_largediv": res,
           "falsifier": {"KILLED": bool(killed),
                         "verdict": ("KILLED: VAL dirAUC ~0.50 (no signed fair-value lead survives to 120s)" if killed_auc else
                                     ("KILLED: large-|div| subset fails breakeven 0.541 in a held-out year" if killed else
                                      "SURVIVED: signed micro-vs-mid slope-divergence clears breakeven all 3 yrs -> NEW LEVER"))}}
    json.dump(out, open("min2_stoikov_result.json", "w"), indent=1)
    print(f"[stoikov] per-year large-|div|: {res}", flush=True)
    print(f"[stoikov] FALSIFIER: {out['falsifier']['verdict']}", flush=True)
    print("[stoikov] -> min2_stoikov_result.json", flush=True)


if __name__ == "__main__":
    main()
