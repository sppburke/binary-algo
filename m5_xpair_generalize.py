"""EDGE-GENERALIZATION TEST (#3 gate, 2026-06-03): does the certified EURUSD 5m dip-buy UP edge EXIST on the other 6
USD majors, or is it EURUSD-specific? Decides whether a cross-pair PORTFOLIO of the edge can diversify (low 5m return
correlation, mean 0.054, means a portfolio WOULD diversify IF the edge generalizes).

For each major X, build the SAME cross-pair construction as m5_xpair but TARGETED on X (X's own return + the USD-common
basket sign-aligned to X's own up-direction + lead-lag residuals + dispersion/agreement), label = X's own next-5m sign.
Train a primary GBM (m5xp-style), gate UP at the CERTIFIED operating point (NY & pr>0.5 & cov0.05 confidence-cover),
evaluate per-year 2024/25/26 with nonoverlap_chrono + boot CI95 + moved up-rate tripwire. EURUSD is the CONTROL (must
reproduce ~.55+). Same deriv-faithful discipline (ties-LOSE breakeven 0.541).

PRE-REGISTERED FALSIFIER (before OOS): the edge GENERALIZES iff >=2 NON-EUR majors show binding-year (worst of 2024/25/26,
n>=150) UP-gated win CI95-lo > 0.53 (comparable to EURUSD's certified .553 floor). EURUSD control must reproduce ~.55+
(harness sanity). If all 6 others are ~.50-.52 -> the dip-buy UP edge is EURUSD-SPECIFIC -> cross-pair portfolio (#3) DEAD.

  ~/binary-algo-venv/bin/python m5_xpair_generalize.py
"""
import os, sys; sys.argv = ["x"]
import json, time, numpy as np, pandas as pd
import lightgbm as lgb
ROOT = "/media/sean/CORSAIR/binary-algo"; FEAT = f"{ROOT}/features"; BE = 0.541; COV = 0.05; STRIDE = 6
PAIRS = ["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDJPY", "USDCHF", "USDCAD"]
USD_BASE = {"USDJPY", "USDCHF", "USDCAD"}
LB = [1, 3, 5, 10, 15, 30]; HOR = 5
SPL = {"train": [str(y) for y in range(2012, 2022)], "test24": ["2024"], "test25": ["2025"], "oos": ["2026"]}
def sgn(p): return -1.0 if p in USD_BASE else 1.0   # p's move in USD-weakness-up direction

def build_target(target, years, stride=1):
    """Cross-pair features TARGETED on `target` (mirror of m5_xpair.build_xp, EURUSD->target, basket sign-aligned to
    target's own up-direction). label _y = (target next-HOR log return > 0). Leakage-safe: all feats use returns ending<=t."""
    wX = sgn(target); others = [p for p in PAIRS if p != target]; out = []
    for y in years:
        cl = {}; ok = True
        for p in PAIRS:
            fp = f"{FEAT}/{p}_{y}.parquet"
            if not os.path.exists(fp): ok = False; break
            d = pd.read_parquet(fp, columns=["close"]); d = d[~d.index.duplicated(keep="last")]; cl[p] = d["close"]
        if not ok: continue
        df = pd.DataFrame(cl).dropna()
        if len(df) < 100: continue
        idx = df.index; secs = idx.values.astype("datetime64[s]").astype("int64"); n = len(df)
        lr = {p: np.log(df[p].values) for p in PAIRS}
        rets = {p: {k: np.concatenate([[np.nan] * k, lr[p][k:] - lr[p][:-k]]) for k in LB} for p in PAIRS}
        feats = {}
        for k in LB:
            x_r = rets[target][k]
            basket = np.nanmean(np.vstack([wX * sgn(p) * rets[p][k] for p in others]), axis=0)  # USD-common in target-own-up dir
            disp = np.nanstd(np.vstack([wX * sgn(p) * rets[p][k] for p in others]), axis=0)
            agree = np.nanmean(np.vstack([(np.sign(wX * sgn(p) * rets[p][k]) == np.sign(basket)).astype(float) for p in others]), axis=0)
            feats[f"x_r{k}"] = x_r; feats[f"usdbask{k}"] = basket
            feats[f"catchup{k}"] = basket - x_r; feats[f"xresid{k}"] = x_r - basket
            feats[f"disp{k}"] = disp; feats[f"agree{k}"] = agree
            for p in others:
                feats[f"ll_{p}{k}"] = wX * sgn(p) * rets[p][k] - x_r
        hours = idx.hour.values + idx.minute.values / 60.0
        feats["sess_ny"] = ((hours >= 13.0) & (hours < 22.0)).astype(float)
        feats["sess_ln"] = ((hours >= 7.0) & (hours < 16.0)).astype(float); feats["hour"] = hours
        feats["comp60"] = pd.Series(rets[target][1]).rolling(60, min_periods=20).std().values
        fwd = np.full(n, np.nan)
        if n > HOR:
            contig = (secs[HOR:] - secs[:-HOR]) == HOR * 60
            fwd[:n - HOR] = np.where(contig, lr[target][HOR:] - lr[target][:-HOR], np.nan)
        F = pd.DataFrame(feats, index=idx); F["_y"] = (fwd > 0).astype(float); F["_ts"] = secs; F["_fwd"] = fwd
        F = F.loc[np.isfinite(fwd) & (fwd != 0)]
        if stride > 1: F = F.iloc[::stride]
        out.append(F)
    return pd.concat(out)

def nonoverlap(ts, mask, gap=HOR * 60):
    take = []; block = -1
    for i in np.where(mask)[0]:
        if ts[i] >= block: take.append(i); block = ts[i] + gap
    return np.array(take, int)
def boot(c, nb=2500, seed=7):
    c = np.asarray(c, float)
    if len(c) < 5: return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed); n = len(c)
    a = np.array([c[rng.integers(0, n, n)].mean() for _ in range(nb)]); return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))

def main():
    t0 = time.time()
    json.dump({"test": "edge-generalization across 7 USD majors (#3 gate)", "breakeven": BE, "cov": COV, "status": "PRE-REGISTERED",
               "FALSIFIER": "GENERALIZES iff >=2 non-EUR majors binding-year UP CI95-lo>0.53 (n>=150); EURUSD control ~.55+. "
                            "Else edge is EURUSD-specific -> cross-pair portfolio (#3) DEAD."},
              open(f"{ROOT}/m5_xpair_generalize_result.json", "w"), indent=2)
    res = {}
    for X in PAIRS:
        TR = build_target(X, SPL["train"], STRIDE); cols = [c for c in TR.columns if not c.startswith("_")]
        m = lgb.LGBMClassifier(n_estimators=1500, learning_rate=0.03, num_leaves=127, min_child_samples=300, subsample=0.8,
                               subsample_freq=1, colsample_bytree=0.5, reg_lambda=20, n_jobs=20, verbosity=-1)
        m.fit(TR[cols].astype("float32"), TR["_y"].astype(int).values); del TR
        yrs = {}
        for w, yr in (("test24", 2024), ("test25", 2025), ("oos", 2026)):
            D = build_target(X, SPL[w]); pr = m.predict_proba(D[cols].astype("float32"))[:, 1]
            y = D["_y"].astype(int).values; ts = D["_ts"].values.astype("int64"); ny = D["sess_ny"].values > 0.5
            conf = np.abs(pr - 0.5); g = ny & (pr > 0.5)
            if g.sum() < 20: continue
            cthr = np.quantile(conf[g], 1 - COV); sel = nonoverlap(ts, g & (conf >= cthr))
            if len(sel) >= 20:
                cc = (y[sel] == 1).astype(float); lo, hi = boot(cc)
                yrs[yr] = dict(win=round(float(cc.mean()), 4), n=len(sel), ci_lo=round(lo, 4),
                               moved_up_rate=round(float((y[ny] == 1).mean()), 4))
            del D
        if len(yrs) == 3:
            byr = min(yrs, key=lambda z: yrs[z]["ci_lo"]); ns = [yrs[z]["n"] for z in yrs]
            res[X] = dict(per_year=yrs, binding=dict(year=byr, win=yrs[byr]["win"], ci_lo=yrs[byr]["ci_lo"], min_n=min(ns)))
        print(f"[gen] {X}: {res.get(X, {}).get('binding')} per-year={ {z: yrs[z]['win'] for z in yrs} } ({time.time()-t0:.0f}s)", flush=True)
    eur = res.get("EURUSD", {}).get("binding", {})
    others_pass = [X for X in PAIRS if X != "EURUSD" and res.get(X, {}).get("binding", {}).get("ci_lo", -9) > 0.53 and res.get(X, {}).get("binding", {}).get("min_n", 0) >= 150]
    generalizes = len(others_pass) >= 2
    out = json.load(open(f"{ROOT}/m5_xpair_generalize_result.json"))
    out.update(dict(per_pair=res, EURUSD_control_binding=eur, non_eur_pairs_with_edge=others_pass,
        VERDICT=dict(edge_generalizes=generalizes, n_other_majors_with_edge=len(others_pass),
            statement=(f"EURUSD control binding {eur.get('win')} (CI-lo {eur.get('ci_lo')}) — harness {'OK' if eur.get('ci_lo',0)>0.50 else 'CHECK'}. "
                f"Non-EUR majors with binding-year UP CI95-lo>0.53: {others_pass} ({len(others_pass)}/6). "
                + ("EDGE GENERALIZES -> cross-pair PORTFOLIO (#3) is VIABLE: build the per-pair edges + diversified book "
                   "(low 5m return corr 0.054 -> real variance reduction). Verify each survivor via CPCV + check spreads."
                   if generalizes else
                   "EDGE IS EURUSD-SPECIFIC -> cross-pair portfolio (#3) DEAD: the other majors do not carry a comparable "
                   "dip-buy UP edge, so a portfolio adds noise not edge. The certified edge is idiosyncratic to EURUSD.")))))
    json.dump(out, open(f"{ROOT}/m5_xpair_generalize_result.json", "w"), indent=2, default=str)
    print(f"\nVERDICT: {out['VERDICT']['statement']}\n-> m5_xpair_generalize_result.json ({time.time()-t0:.0f}s)", flush=True)

if __name__ == "__main__": main()
