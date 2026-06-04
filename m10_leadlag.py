"""N17 (bounded) — STRICTLY-LAGGED cross-pair lead-lag @10m, the last genuinely-distinct on-disk direction
mechanism. The certified book uses CONCURRENT cross-pair returns (windows ending AT t). This tests the
ANTI-CONTEMPORANEOUS channel: do PAST cross-pair moves of the 6 non-EUR majors (windows ending strictly BEFORE
t) + EURUSD's own lagged moves predict the next-10m EURUSD sign? (5m verdict: trending NULL — lagged dominated by
concurrent; at 10m concurrent already certifies both sides, so prior is low. Bounded: lgb only, no RFF/TE tail.)

Zero look-ahead: every feature is a return over [t-2L, t-L], ending at bar t-L < t. Label = sign(close[t+10]-close[t]),
600s wall-clock contiguous, ties dropped. Gate NY x conf-cover, worst-VAL-half. Side-split per year.
INCUMBENT (certified EURUSD.m10xp.v1): forward cov10 2025 UP .605 / DOWN .574; refit p10 UP .5863 / DOWN .5683.
PRE-REGISTERED FALSIFIER: KILL the lagged lead-lag channel unless it beats the incumbent on binding 2025 on >=1
side (UP>=.605 OR DOWN>=.574) at healthy n with up-rate in [.47,.53]. Else: anti-contemporaneous channel adds
nothing over the concurrent cross-pair at 10m (confirms 5m null).
    python m10_leadlag.py
"""
import os, json, time, numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import m5_xpair as MX

FEAT = "/media/sean/CORSAIR/binary-algo/features"
PAIRS = ["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDJPY", "USDCHF", "USDCAD"]
USD_BASE = {"USDJPY", "USDCHF", "USDCAD"}
NONEU = [p for p in PAIRS if p != "EURUSD"]
LAGS = [1, 2, 5, 10, 15, 30]   # minute-bar lags L; feature = return over [t-2L, t-L] (ends strictly before t)
HOR = 10; GAP = HOR * 60; BE = 0.541; COV = 0.05
SPL = MX.SPL
OUT = "/media/sean/CORSAIR/binary-algo/m10_leadlag_result.json"
INC = {"UP_2025": 0.605, "DOWN_2025": 0.574}


def sgn(p): return -1.0 if p in USD_BASE else 1.0


def build(years, stride=1):
    out = []
    for y in years:
        cl = {}; ok = True
        for p in PAIRS:
            fp = f"{FEAT}/{p}_{y}.parquet"
            if not os.path.exists(fp): ok = False; break
            d = pd.read_parquet(fp, columns=["close"]); d = d[~d.index.duplicated(keep="last")]; cl[p] = d["close"]
        if not ok: continue
        df = pd.DataFrame(cl).dropna()
        if len(df) < 200: continue
        idx = df.index; secs = idx.values.astype("datetime64[s]").astype("int64"); n = len(df)
        lr = {p: np.log(df[p].values) for p in PAIRS}
        feats = {}
        for L in LAGS:
            # return over [t-2L, t-L]: ends at t-L, strictly before t (NO term at t)
            for p in PAIRS:
                r = np.full(n, np.nan)
                if n > 2 * L:
                    r[2 * L:] = lr[p][L:n - L] - lr[p][:n - 2 * L]
                feats[f"lag_{p}_{L}"] = sgn(p) * r if p != "EURUSD" else r   # eu-equiv for non-EUR; raw for EUR
            # lagged USD-weakness basket (eu-equiv) and EURUSD-minus-basket lagged residual
            bask = np.nanmean(np.vstack([feats[f"lag_{p}_{L}"] for p in NONEU]), axis=0)
            feats[f"lagbask_{L}"] = bask
            feats[f"lagresid_{L}"] = feats[f"lag_EURUSD_{L}"] - bask
        hours = idx.hour.values + idx.minute.values / 60.0
        feats["sess_ny"] = ((hours >= 13.0) & (hours < 22.0)).astype(float)
        # label: next-10m EURUSD sign, 600s contiguous
        fwd = np.full(n, np.nan)
        if n > HOR:
            contig = (secs[HOR:] - secs[:-HOR]) == HOR * 60
            fr = lr["EURUSD"][HOR:] - lr["EURUSD"][:-HOR]
            fwd[:n - HOR] = np.where(contig, fr, np.nan)
        F = pd.DataFrame(feats, index=idx); F["_y"] = (fwd > 0).astype(float); F["_ts"] = secs; F["_fwd"] = fwd
        F = F.loc[np.isfinite(fwd) & (fwd != 0)]
        if stride > 1: F = F.iloc[::stride]
        out.append(F)
    return pd.concat(out)


def boot(c, nb=3000, seed=7):
    c = np.asarray(c, float)
    if len(c) < 5: return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed); n = len(c)
    return tuple(float(x) for x in np.percentile([c[rng.integers(0, n, n)].mean() for _ in range(nb)], [2.5, 97.5]))


def wvh_cover(pr, y, ts, ny, side):
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
    stub = {"experiment": "m10_leadlag (strictly-lagged cross-pair, N17 bounded)", "incumbent": INC, "breakeven": BE,
            "PRE_REGISTERED_FALSIFIER": "KILL lagged lead-lag unless beats incumbent on 2025 (UP>=.605 OR DOWN>=.574) "
            "at healthy n, up-rate in [.47,.53].", "status": "PRE-REGISTERED"}
    json.dump(stub, open(OUT, "w"), indent=1)
    TR = build(SPL["train"], 4); VA = build(SPL["val"])
    cols = [c for c in TR.columns if c.startswith("lag")]
    ytr = TR["_y"].astype(int).values; yva = VA["_y"].astype(int).values
    print(f"[leadlag10] train={len(TR):,} val={len(VA):,} feats={len(cols)} up-rate(tr)={ytr.mean():.4f} {time.time()-t0:.0f}s", flush=True)
    L = lgb.LGBMClassifier(objective="binary", metric="auc", learning_rate=0.02, num_leaves=127, min_child_samples=400,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=20, n_estimators=2000, n_jobs=20, verbosity=-1)
    L.fit(TR[cols].astype("float32"), ytr, eval_set=[(VA[cols].astype("float32"), yva)], eval_metric="auc",
          callbacks=[lgb.early_stopping(120), lgb.log_evaluation(0)])
    prv = L.predict_proba(VA[cols].astype("float32"))[:, 1]
    aucv = roc_auc_score(yva, prv)
    tsv = VA["_ts"].values.astype("int64"); nyv = VA["sess_ny"].values > 0.5
    print(f"[leadlag10] VAL AUC={aucv:.4f} (concurrent xpair ~.526) {time.time()-t0:.0f}s", flush=True)
    years = {}
    for w, lab in (("test24", "2024"), ("test25", "2025"), ("oos", "2026")):
        D = build([w[-4:] if w != "oos" else "2026"] if False else SPL[w])
        years[lab] = (L.predict_proba(D[cols].astype("float32"))[:, 1], D["_y"].astype(int).values,
                      D["_ts"].values.astype("int64"), D["sess_ny"].values > 0.5, float(D["_y"].mean()))
        del D
    res = {"val_auc": round(float(aucv), 4), "sides": {}}
    for side, nm in ((1, "UP"), (0, "DOWN")):
        fr = wvh_cover(prv, yva, tsv, nyv, side); cthr = fr[0] if fr else None
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
                 "VERDICT": {"SURVIVES": bool(surv), "statement": ("Lagged lead-lag @10m " + ("SURVIVES" if surv else
                             "KILLED — anti-contemporaneous channel adds nothing over concurrent cross-pair (confirms 5m null)."))}})
    json.dump(stub, open(OUT, "w"), indent=1)
    print(f"[leadlag10] {stub['VERDICT']['statement']} ({time.time()-t0:.0f}s) -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
