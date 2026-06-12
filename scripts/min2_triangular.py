"""Sweep row N2 (DISCOVERED, top prior ~15%): TRIANGULAR USD-CANCELING RESIDUAL for 2m EURUSD direction.
Hypothesis: the relative-value residual of EURUSD vs GBPUSD (rolling causal cointegration) has the USD common
factor ALGEBRAICALLY removed (GBPUSD shares the USD leg), so its mean-reversion sign is a directional signal
IMMUNE to the 2025 USD-factor inversion that killed RMT/xpair. When EURUSD is rich vs GBP (z>0) -> bet DOWN;
cheap (z<0) -> bet UP. Signal computed on minute bars (slow relative-value), aligned causally (merge_asof
backward) to the EURUSD 2m tick clock; settlement deriv-faithful (wc_ret 120s, ties LOSE), moved-bars, per-yr CI95.

PRE-REGISTERED FALSIFIER: KILL if 2025 selective reversion acc <= 0.534 (the raw-xpair floor) OR the edge sign
flips between 2024 and 2026 OR no held-out year clears breakeven 0.541. Threshold |z| selected on worst-VAL-half."""
import json, numpy as np, pandas as pd
import harness as H
import min2_production as M2

W = 500   # rolling-cointegration window (minutes, ~8h)
HS = 120


def year_of(ts):
    return pd.to_datetime(np.asarray(ts), unit="s", utc=True).year.values


def minute_z():
    """Causal rolling-cointegration residual z of logEURUSD vs logGBPUSD, per minute, indexed by int-sec."""
    def loadclose(p):
        d = pd.concat([pd.read_parquet(f"{H.FEAT_DIR}/{p}_{y}.parquet", columns=["close"]) for y in range(2022, 2027)])
        return d[~d.index.duplicated(keep="last")].sort_index()
    e = loadclose("EURUSD"); g = loadclose("GBPUSD")
    df = pd.DataFrame({"e": np.log(e["close"]), "g": np.log(g["close"])}).dropna().sort_index()
    # causal rolling OLS beta of e on g over W (uses [t-W+1, t], available at t)
    me = df["e"].rolling(W).mean(); mg = df["g"].rolling(W).mean()
    cov = (df["e"] * df["g"]).rolling(W).mean() - me * mg
    var = (df["g"] * df["g"]).rolling(W).mean() - mg * mg
    beta = cov / var.replace(0, np.nan)
    alpha = me - beta * mg
    resid = df["e"] - (alpha + beta * df["g"])
    z = resid / resid.rolling(W).std().replace(0, np.nan)
    out = pd.DataFrame({"sec": df.index.values.astype("datetime64[s]").astype("int64"), "z": z.values}).dropna()
    return out.sort_values("sec").reset_index(drop=True)


def tick_labels(sp):
    b = M2.load_split(sp); mid = b["mid"].values.astype(float)
    ts = b.index.values.astype("datetime64[s]").astype("int64")
    ret, valid = M2.wc_ret(ts, mid, HS, M2.TOL_S, M2.ENTRY_LAG_S)
    del b
    return ts, (ret > 0).astype(int), np.abs(ret), valid


def build(sp, z):
    ts, y, mag, valid = tick_labels(sp)
    d = pd.merge_asof(pd.DataFrame({"sec": ts}), z, on="sec", direction="backward")
    zz = d["z"].values
    ok = valid & np.isfinite(zz)
    pred = (zz < 0).astype(int)              # reversion: EURUSD cheap vs GBP -> bet UP
    return ts, y, mag, ok, pred, np.abs(zz)


def sideacc(ts, y, mag, pred, mask, label, side):
    yr = year_of(ts)
    sel = mask & (yr == int(label))
    if side is not None:
        sel = sel & (pred == side)
    moved = sel & (mag > 0)
    if moved.sum() < 5:
        return {"n": int(sel.sum()), "n_moved": int(moved.sum()), "acc": None, "ci": [None, None]}
    corr = (pred[moved] == y[moved]).astype(float); lo, hi = M2.boot(corr)
    return {"n": int(sel.sum()), "n_moved": int(moved.sum()), "acc": float(corr.mean()), "ci": [lo, hi]}


def main():
    z = minute_z(); print(f"[N2] minute z rows={len(z):,}", flush=True)
    # threshold |z| selected on worst-VAL-half by reversion moved-acc (require n>=50)
    vts, vy, vmag, vok, vpred, vabz = build("val", z)
    order = np.argsort(vts); half = set(order[len(order) // 2:].tolist())
    hmask = np.zeros(len(vts), bool); hmask[list(half)] = True
    best = None
    for thr in (0.0, 0.5, 1.0, 1.5, 2.0):
        sel = vok & hmask & (vabz >= thr); m = sel & (vmag > 0)
        sub = M2.nonoverlap_chrono(vts, sel)
        ms = sub[vmag[sub] > 0]
        if len(ms) < 50:
            continue
        acc = (vpred[ms] == vy[ms]).mean()
        if best is None or acc > best[1]:
            best = (thr, float(acc), len(ms))
    thr = best[0] if best else 0.0
    print(f"[N2] selected |z|>={thr} (worst-VAL-half rev acc={best[1] if best else None}, n={best[2] if best else 0})", flush=True)
    res = {}
    for sp, yrs in (("test", ("2024", "2025")), ("oos", ("2026",))):
        ts, y, mag, ok, pred, abz = build(sp, z)
        cand = ok & (abz >= thr)
        tr = M2.nonoverlap_chrono(ts, cand)
        trmask = np.zeros(len(ts), bool); trmask[tr] = True
        for label in yrs:
            res[label] = {"COMB": sideacc(ts, y, mag, pred, trmask, label, None),
                          "UP": sideacc(ts, y, mag, pred, trmask, label, 1),
                          "DOWN": sideacc(ts, y, mag, pred, trmask, label, 0)}
    a24 = res["2024"]["COMB"]["acc"]; a25 = res["2025"]["COMB"]["acc"]; a26 = res["2026"]["COMB"]["acc"]
    flip = (a24 is not None and a26 is not None and (a24 - 0.5) * (a26 - 0.5) < 0)
    killed = (a25 is None or a25 <= 0.534) or flip or not all(
        res[Y]["COMB"]["acc"] and res[Y]["COMB"]["acc"] >= 0.541 for Y in ("2024", "2025", "2026"))
    out = {"row": "N2 triangular USD-canceling residual (EURUSD vs GBPUSD)", "horizon_s": HS, "breakeven": 0.541,
           "window_min": W, "z_threshold": thr, "per_year": res,
           "falsifier": {"KILLED": bool(killed),
                         "verdict": ("KILLED: 2025<=0.534 or sign-flip 24<->26 or a year below breakeven (USD-canceling "
                                     "residual carries no robust 2m direction)" if killed else
                                     "SURVIVED: USD-immune residual reversion clears breakeven all 3 yrs incl 2025 -> NEW LEVER")}}
    json.dump(out, open("min2_triangular_result.json", "w"), indent=1)
    for Y in ("2024", "2025", "2026"):
        r = res[Y]; print(f"  {Y}: COMB {r['COMB']['acc']} (n{r['COMB']['n_moved']}) | UP {r['UP']['acc']} | DOWN {r['DOWN']['acc']}", flush=True)
    print(f"[N2] FALSIFIER: {out['falsifier']['verdict']}", flush=True)
    print("[N2] -> min2_triangular_result.json", flush=True)


if __name__ == "__main__":
    main()
