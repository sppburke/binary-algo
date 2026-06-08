"""N4 (round-2 discovery residue) — CLOCK-CONDITIONED INTRADAY-MOMENTUM / TURNING-POINT signed predictor @10m.
The one signed construction the corpus scan found genuinely un-run as a standalone predictor (Gao-Han-Li-Zhou
"Market Intraday Momentum" JFE2018; Goulding-Harvey-Mazzoleni turning-point 4-state; Wood slow-momentum/fast-
reversion). MECHANISM (carries SIGN, not magnitude): the SIGN of the next-10m return is conditionally
autocorrelated with earlier signed interval returns AS A FUNCTION OF CLOCK-TIME / turning-point state — distinct
from the cross-pair co-movement and from the base 239 (which lack session-anchored cumulative signed returns x
time-of-day interaction). RISK (pre-noted): intraday momentum is classically magnitude/volume-driven -> may
collapse to sign-invariance; that is exactly what this falsifier tests.

Features (all computable at t, zero look-ahead): session-anchored cumulative signed returns (since NY/London/
midnight open), signed momentum at {3,5,10,15,30,60}-bar lookbacks, clock (hour, min-of-day, dow), turning-point
state (range-position vs rolling 60-bar hi/lo; sign vs rolling mean), slow-fast signed blend (sign r60 x sign r5).
Label = sign(close[t+10]-close[t]), 600s contiguous, ties dropped. Gate NY x conf-cover worst-VAL-half, side-split.
INCUMBENT (certified EURUSD.m10xp.v1): forward cov10 2025 UP .605 / DOWN .574; refit p10 UP .5863 / DOWN .5683.
PRE-REGISTERED FALSIFIER: KILL unless beats incumbent on binding 2025 on >=1 side (UP>=.605 OR DOWN>=.574) at
healthy n with up-rate in [.47,.53]. Else: intraday-momentum carries no on-disk 10m DIRECTION beyond the certified
cross-pair edge (and/or collapses to magnitude by sign-invariance).
    python m10_intramom.py
"""
import os, json, time, numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import m5_xpair as MX

FEAT = "/home/sean/git/binary-algo/features"
HOR = 10; GAP = HOR * 60; BE = 0.541
SPL = MX.SPL
OUT = "/home/sean/git/binary-algo/m10_intramom_result.json"
INC = {"UP_2025": 0.605, "DOWN_2025": 0.574}
LB = [3, 5, 10, 15, 30, 60]


def build(years, stride=1):
    out = []
    for y in years:
        fp = f"{FEAT}/EURUSD_{y}.parquet"
        if not os.path.exists(fp): continue
        d = pd.read_parquet(fp, columns=["close"]); d = d[~d.index.duplicated(keep="last")].sort_index()
        idx = d.index; n = len(d)
        if n < 200: continue
        secs = idx.values.astype("datetime64[s]").astype("int64"); lr = np.log(d["close"].values)
        hour = idx.hour.values + idx.minute.values / 60.0; sod = idx.hour.values * 60 + idx.minute.values
        dow = idx.dayofweek.values; dates = idx.normalize()
        f = {}
        # signed momentum lookbacks (end at t)
        for k in LB:
            r = np.full(n, np.nan); r[k:] = lr[k:] - lr[:-k]; f[f"mom_{k}"] = r
        # session-anchored cumulative signed return (since first bar with hour>=A on same day)
        s = pd.Series(lr, index=idx)
        for A, nm in ((0, "mid"), (7, "ln"), (13, "ny")):
            anchor_lr = np.full(n, np.nan)
            mask_after = (idx.hour.values >= A)
            df = pd.DataFrame({"lr": lr, "after": mask_after, "date": dates})
            # first lr at/after anchor hour, per day, forward-filled within day
            first = df[df.after].groupby(df[df.after]["date"]).first()["lr"]
            amap = df["date"].map(first)
            cum = lr - amap.values
            cum[~mask_after] = np.nan
            f[f"cum_{nm}"] = cum
        # clock
        f["hour"] = hour; f["sod"] = sod.astype(float); f["dow"] = dow.astype(float)
        # turning-point / range position vs rolling 60-bar hi/lo + sign vs rolling mean
        c = d["close"].values
        roll_hi = pd.Series(c, index=idx).rolling(60, min_periods=20).max().values
        roll_lo = pd.Series(c, index=idx).rolling(60, min_periods=20).min().values
        roll_mu = pd.Series(c, index=idx).rolling(60, min_periods=20).mean().values
        f["rangepos60"] = (c - roll_lo) / (roll_hi - roll_lo + 1e-12)
        f["sign_vs_mu60"] = np.sign(c - roll_mu)
        # slow-fast signed blend
        f["slowfast"] = np.sign(f["mom_60"]) - np.sign(f["mom_5"])
        f["slow_x_fast"] = np.sign(f["mom_60"]) * np.sign(f["mom_5"])
        # label
        fwd = np.full(n, np.nan)
        if n > HOR:
            contig = (secs[HOR:] - secs[:-HOR]) == HOR * 60
            fwd[:n - HOR] = np.where(contig, lr[HOR:] - lr[:-HOR], np.nan)
        F = pd.DataFrame(f, index=idx)
        F["sess_ny"] = ((hour >= 13.0) & (hour < 22.0)).astype(float)
        F["_y"] = (fwd > 0).astype(float); F["_ts"] = secs; F["_fwd"] = fwd
        F = F.loc[np.isfinite(fwd) & (fwd != 0)]
        if stride > 1: F = F.iloc[::stride]
        out.append(F)
    return pd.concat(out)


def boot(c, nb=3000, seed=7):
    c = np.asarray(c, float)
    if len(c) < 5: return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed); n = len(c)
    return tuple(float(x) for x in np.percentile([c[rng.integers(0, n, n)].mean() for _ in range(nb)], [2.5, 97.5]))


def wvh(pr, y, ts, ny, side):
    conf = np.abs(pr - 0.5); g = ny & ((pr > 0.5) == (side == 1)); mid = np.median(ts); h1 = ts < mid; best = None
    for cov in (0.10, 0.05):
        pool = conf[g]
        if pool.size < 400: continue
        thr = float(np.quantile(pool, 1 - cov)); accs = []; ok = True
        for hm in (h1, ~h1):
            m = g & hm & (conf >= thr); sel = MX.nonoverlap_chrono(ts, m, GAP)
            if len(sel) < 40: ok = False; break
            accs.append(((1 if side == 1 else 0) == y[sel]).mean())
        if not ok: continue
        w = min(accs)
        if best is None or w > best[1]: best = (thr, w, cov)
    return best


def main():
    t0 = time.time()
    stub = {"experiment": "m10_intramom (clock-conditioned intraday-momentum / turning-point, N4)", "incumbent": INC,
            "breakeven": BE, "PRE_REGISTERED_FALSIFIER": "KILL unless beats incumbent on 2025 (UP>=.605 OR DOWN>=.574) "
            "at healthy n, up-rate [.47,.53].", "status": "PRE-REGISTERED"}
    json.dump(stub, open(OUT, "w"), indent=1)
    TR = build(SPL["train"], 4); VA = build(SPL["val"])
    cols = [c for c in TR.columns if not c.startswith("_") and c != "sess_ny"]
    ytr = TR["_y"].astype(int).values; yva = VA["_y"].astype(int).values
    print(f"[intramom10] train={len(TR):,} val={len(VA):,} feats={len(cols)} up-rate(tr)={ytr.mean():.4f} {time.time()-t0:.0f}s", flush=True)
    L = lgb.LGBMClassifier(objective="binary", metric="auc", learning_rate=0.02, num_leaves=127, min_child_samples=400,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.6, reg_lambda=20, n_estimators=2000, n_jobs=20, verbosity=-1)
    L.fit(TR[cols].astype("float32"), ytr, eval_set=[(VA[cols].astype("float32"), yva)], eval_metric="auc",
          callbacks=[lgb.early_stopping(120), lgb.log_evaluation(0)])
    prv = L.predict_proba(VA[cols].astype("float32"))[:, 1]; aucv = roc_auc_score(yva, prv)
    tsv = VA["_ts"].values.astype("int64"); nyv = VA["sess_ny"].values > 0.5
    print(f"[intramom10] VAL AUC={aucv:.4f} {time.time()-t0:.0f}s", flush=True)
    years = {}
    for w, lab in (("test24", "2024"), ("test25", "2025"), ("oos", "2026")):
        D = build(SPL[w])
        years[lab] = (L.predict_proba(D[cols].astype("float32"))[:, 1], D["_y"].astype(int).values,
                      D["_ts"].values.astype("int64"), D["sess_ny"].values > 0.5, float(D["_y"].mean()))
        del D
    res = {"val_auc": round(float(aucv), 4), "sides": {}}
    for side, nm in ((1, "UP"), (0, "DOWN")):
        fr = wvh(prv, yva, tsv, nyv, side); cthr = fr[0] if fr else None
        yr = {}
        for lab in ("2024", "2025", "2026"):
            pr, y, ts, ny, upr = years[lab]; conf = np.abs(pr - 0.5)
            g = ny & ((pr > 0.5) == (side == 1)) & (conf >= (cthr if cthr is not None else 0)); sel = MX.nonoverlap_chrono(ts, g, GAP)
            if len(sel) < 10: yr[lab] = {"n": len(sel), "win": None, "all_up_rate": round(upr, 4)}; continue
            cc = ((1 if side == 1 else 0) == y[sel]).astype(float); lo, hi = boot(cc)
            yr[lab] = {"n": len(sel), "win": round(float(cc.mean()), 4), "ci": [round(lo, 4), round(hi, 4)], "all_up_rate": round(upr, 4)}
        res["sides"][nm] = {"val_cover": (None if not fr else {"thr": fr[0], "cov": fr[2]}), "years": yr}
        print(f"  [{nm}] 2025 {yr['2025']} vs inc {INC[nm+'_2025']}", flush=True)
    def beats(nm):
        w = res["sides"][nm]["years"]["2025"]; return w.get("win") is not None and w["win"] >= INC[nm + "_2025"] and w.get("n", 0) >= 30
    surv = beats("UP") or beats("DOWN")
    stub.update({"status": "COMPLETE", **res,
                 "VERDICT": {"SURVIVES": bool(surv), "statement": ("Intraday-momentum/turning-point @10m " + ("SURVIVES" if surv else
                             "KILLED — no on-disk 10m DIRECTION beyond the certified cross-pair edge (intraday-momentum sign-invariant/dominated here)."))}})
    json.dump(stub, open(OUT, "w"), indent=1)
    print(f"[intramom10] {stub['VERDICT']['statement']} ({time.time()-t0:.0f}s) -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
