"""EDGE lever I3 @15m: |return|-weighted retrain of the 15m direction ensemble + worst-VAL-half gate.

Two changes vs the frozen m15 book, isolable via the POW arg:
  - POW>0: primary trained with sample_weight=(|fwd_15m|/median)^POW (up-weight large-move bars, where sign is
    more predictable) — the mechanism that RESCUED the 5m DOWN side (m5_magweight POW=0.5 -> two-sided ~.56).
  - POW=0: unweighted control (weight==1) — isolates the second change: gate re-selected by WORST-VAL-HALF
    (the disciplined selection) instead of the frozen book's VAL-acc-max (corr(VAL,OOS)=-0.54 trap).

Same 239 base features, same comp(bb_width)xNY gate FORM, same 15m settlement (_y=ret>0 ties-excluded,
nonoverlap_chrono 900s). 3-model ensemble at reduced trees (600/700/700) for the side-rebalance test (the
frozen book uses 3000 — this is an approximation flagged in the JSON; if a POW wins we refit at full fidelity
+ CPCV before any freeze). Side-split per year UP/DOWN/COMBINED vs the side incumbents. Breakeven 0.541.

Usage: python m15_magweight.py [POW=0.5]
"""
import sys, json, time, os, numpy as np, pandas as pd
import lightgbm as lgb, xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
import harness as H
import m15_production as M15

POW = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
HOR = 15
base = list(H.feature_cols("EURUSD"))
SUB_FIT = 150_000
BE = 0.541


def load_fwd(years, stride=1):
    """Like M15.load but also keep _fwd (ret) for the magnitude weight + bb_width/sess_ny for the gate."""
    parts = []
    for y in years:
        p = f"{H.FEAT_DIR}/EURUSD_{y}.parquet"
        if not os.path.exists(p):
            continue
        df = pd.read_parquet(p, columns=base + H.META_COLS); df = df[~df.index.duplicated(keep="last")]
        c = df["close"].values; n = len(c)
        secs = df.index.values.astype("datetime64[s]").astype("int64")
        contig = np.zeros(n, bool)
        if n > HOR:
            contig[:n - HOR] = (secs[HOR:] - secs[:-HOR]) == HOR * 60
        fwd = np.full(n, np.nan); fwd[:n - HOR] = c[HOR:]; ret = fwd / c - 1.0
        valid = contig & np.isfinite(ret) & (ret != 0)
        d = df.loc[valid, base].copy()
        d["_y"] = (ret[valid] > 0).astype(int)
        d["_ret"] = ret[valid]
        d["_ts"] = secs[valid]
        parts.append(d.iloc[::stride] if stride > 1 else d)
    return pd.concat(parts)


def magweight(ret, pow_):
    a = np.abs(ret).astype(float); med = np.median(a[a > 0]) or 1e-9
    return np.clip((a / med) ** pow_, 0.1, 10.0)


def proba(L, G, C, X):
    return (L.predict_proba(X)[:, 1] + G.predict_proba(X)[:, 1]
            + C.predict_proba(np.nan_to_num(X, nan=-999))[:, 1]) / 3.0


def main():
    t0 = time.time()
    TR = load_fwd(H.SPLITS["train"], stride=3)
    VA = load_fwd(H.SPLITS["val"])
    if len(TR) > SUB_FIT:
        idx = np.linspace(0, len(TR) - 1, SUB_FIT).astype(int); TR = TR.iloc[idx]
    Xtr = TR[base].astype("float32").values; ytr = TR["_y"].values
    w = magweight(TR["_ret"].values, POW)
    print(f"[m15magw] POW={POW} train={len(TR):,} val={len(VA):,} wt[min/med/max]={w.min():.2f}/{np.median(w):.2f}/{w.max():.2f} {time.time()-t0:.0f}s", flush=True)
    L = lgb.LGBMClassifier(objective="binary", metric="auc", num_leaves=255, learning_rate=0.03, n_estimators=600,
                           subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=10, n_jobs=20, verbosity=-1).fit(Xtr, ytr, sample_weight=w)
    G = xgb.XGBClassifier(n_estimators=700, learning_rate=0.03, max_depth=8, subsample=0.8, colsample_bytree=0.5,
                          tree_method="hist", n_jobs=20, eval_metric="auc").fit(Xtr, ytr, sample_weight=w)
    C = CatBoostClassifier(iterations=700, learning_rate=0.03, depth=8, l2_leaf_reg=10, eval_metric="AUC",
                           thread_count=20, verbose=False).fit(np.nan_to_num(Xtr, nan=-999), ytr, sample_weight=w)
    print(f"[m15magw] trained 3-model {time.time()-t0:.0f}s", flush=True)
    Xva = VA[base].astype("float32").values; yva = VA["_y"].values
    pva = proba(L, G, C, Xva)
    print(f"[m15magw] VAL AUC={roc_auc_score(yva, pva):.4f} (frozen ~0.528)", flush=True)
    bbwv = VA["15m_bb_width"].values.astype(float); nyv = VA["sess_ny"].values.astype(float) > 0.5
    tsv = VA["_ts"].values
    # worst-VAL-half gate selection: split VAL chronologically; pick (q,cov) maximizing the WORSE half's combined acc
    order = np.argsort(tsv); h1 = set(order[:len(order)//2].tolist())
    is_h1 = np.array([i in h1 for i in range(len(tsv))])
    Qbbw = {q: float(np.nanpercentile(VA["15m_bb_width"].values.astype(float), q)) for q in (10, 20, 33)}
    best = None
    for q in (10, 20, 33):
        gmask = (bbwv <= Qbbw[q]) & nyv
        if gmask.sum() < 300:
            continue
        confv = np.abs(pva - 0.5)
        for cov in (0.02, 0.05, 0.10, 0.20):
            cthr = float(np.quantile(confv[gmask], 1 - cov)); sel = gmask & (confv >= cthr)
            halves = []
            for hmask in (is_h1, ~is_h1):
                s = sel & hmask
                if s.sum() < 40:
                    halves.append(np.nan); continue
                halves.append(((pva[s] > 0.5).astype(int) == yva[s]).mean())
            wh = np.nanmin(halves) if np.all(np.isfinite(halves)) else np.nan
            if np.isnan(wh) or sel.sum() < 150:
                continue
            if best is None or wh > best[0]:
                best = (wh, q, cov, cthr, Qbbw[q])
    wh, q, cov, cthr, bthr = best
    print(f"[m15magw] gate worst-VAL-half: q{q} cov{cov:.0%} conf_thr={cthr:.4f} bb<={bthr:.2e} worsthalf={wh:.4f}", flush=True)

    res = {}
    for yr in ("2024", "2025", "2026"):
        D = load_fwd([yr])
        pr = proba(L, G, C, D[base].astype("float32").values); y = D["_y"].values
        ts = D["_ts"].values; bbw = D["15m_bb_width"].values.astype(float); ny = D["sess_ny"].values.astype(float) > 0.5
        m = (bbw <= bthr) & ny & (np.abs(pr - 0.5) >= cthr)
        sel = M15.nonoverlap_chrono(ts, m); pred = (pr[sel] > 0.5).astype(int); yy = y[sel]
        for side, nm in ((1, "UP"), (0, "DOWN"), (None, "COMBINED")):
            ss = (pred == side) if side is not None else np.ones(len(pred), bool)
            if ss.sum() < 5:
                res[f"{yr}_{nm}"] = {"n": int(ss.sum()), "acc": None}; continue
            corr = (pred[ss] == yy[ss]).astype(float); lo, hi = M15.boot(corr)
            res[f"{yr}_{nm}"] = {"n": int(ss.sum()), "acc": round(float(corr.mean()), 4),
                                 "ci": [round(lo, 4), round(hi, 4)], "up_rate": round(float(yy[ss].mean()), 4)}
        del D
    out = {"exp": "I3 |ret|-weighted 15m retrain + worst-VAL-half gate", "POW": POW,
           "approx": "3-model at 600/700/700 trees (frozen book uses 3000) — side-rebalance test; refit full+CPCV if a POW wins",
           "gate": {"q": q, "cov": cov, "conf_thr": round(cthr, 4), "bb_thr": bthr, "val_worsthalf": round(wh, 4)},
           "val_auc": round(float(roc_auc_score(yva, pva)), 4),
           "side_incumbent_frozen": {"UP": "binding 2025 .607", "DOWN": "binding 2025 .559"},
           "breakeven": BE, "per_year_side": res}
    json.dump(out, open(f"m15_magweight_result_pow{POW}.json", "w"), indent=1)
    for k, v in res.items():
        if v.get("acc") is not None:
            print(f"  {k:14} n={v['n']:>4} acc={v['acc']:.4f} CI={v['ci']} up_rate={v['up_rate']}", flush=True)
    print(f"[m15magw] -> m15_magweight_result_pow{POW}.json ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
