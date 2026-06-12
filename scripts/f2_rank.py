"""F2 — Cross-sectional rank / dollar-neutral target across the 7 USD majors.

Thesis (GAPS critique #3, arXiv:2105.10019): EURUSD is ~97% USD-factor, so a single pair's
direction is dominated by the dollar. Removing the common factor cross-sectionally should expose
a more predictable *idiosyncratic* (foreign-leg) signal, and a dollar-neutral long-short of the
7 majors is the natural target.

Construction (causal, same 239-feature stack, same splits as harness.py):
  - Express each pair's 15m fwd return in FOREIGN-currency-vs-USD convention:
      fret_i = sign_i * ret_i,  sign=+1 for EUR/GBP/AUD/NZD USD, -1 for USDCAD/CHF/JPY.
  - Align all 7 on common 1-min timestamps (all valid); common_t = mean_i fret_i (the dollar factor).
  - idio_i = fret_i - common_t  (dollar-neutral, foreign conv).
  - Per pair, target_IDIO = sign of idio in the pair's OWN convention; target_RAW = sign of own ret.
  - Train one LightGBM per pair for RAW and for IDIO (apples-to-apples), report AUC + selective acc
    on TEST 2024-25 and 2026 OOS, threshold frozen on VAL.
  - LONG-SHORT: use idio models' scores to rank the 7 pairs each t; long top-2 / short bottom-2
    (dollar-neutral); report spread-return hit-rate + Sharpe on TEST/OOS.

GATE: idio AUC materially > 0.527 (raw per-pair baseline) AND/OR long-short clears break-even.
"""
import os, sys, time, numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import harness as H

PAIRS = ["EURUSD","GBPUSD","AUDUSD","NZDUSD","USDCAD","USDCHF","USDJPY"]
SIGN  = {p: (-1 if p.startswith("USD") else +1) for p in PAIRS}   # foreign-ccy-vs-USD
HOR = 15
STR = {"train": int(sys.argv[1]) if len(sys.argv) > 1 else 6, "val": 2, "test": 2, "oos": 1}
FEAT = H.FEAT_DIR

def _concat(p, cols, split):
    parts = []
    for y in H.SPLITS[split]:
        f = f"{FEAT}/{p}_{y}.parquet"
        if os.path.exists(f):
            d = pd.read_parquet(f, columns=cols)
            d = d[~d.index.duplicated(keep="last")]
            parts.append(d)
    return pd.concat(parts) if parts else pd.DataFrame(columns=cols)

def fwd_ret(close):
    idx = close.index; c = close.values; n = len(c)
    secs = idx.values.astype("datetime64[s]").astype("int64")
    contig = np.zeros(n, bool)
    if n > HOR: contig[:n-HOR] = (secs[HOR:] - secs[:-HOR]) == HOR*60
    fwd = np.full(n, np.nan); fwd[:n-HOR] = c[HOR:]; ret = fwd/c - 1.0
    return pd.Series(ret, index=idx), pd.Series(contig, index=idx)

def targets(split):
    """Return DataFrames (own-conv) of raw-sign and idio-sign targets, indexed by common ts."""
    rets, valids = {}, {}
    for p in PAIRS:
        cv = _concat(p, ["close","valid"], split)
        cv = cv[~cv.index.duplicated(keep="last")].sort_index()
        r, v = fwd_ret(cv["close"])
        valids[p] = (v.values & cv["valid"].values & np.isfinite(r.values) & (r.values != 0))
        rets[p] = pd.Series(SIGN[p]*r.values, index=cv.index)     # foreign conv
        valids[p] = pd.Series(valids[p], index=cv.index)
    R = pd.DataFrame(rets); V = pd.DataFrame(valids).fillna(False)
    R = R[~R.index.duplicated(keep="last")]; V = V.reindex(R.index).fillna(False)
    allv = V.all(axis=1)
    R = R[allv]                                                   # common ts, all 7 valid, foreign conv
    common = R.mean(axis=1)
    idio = R.sub(common, axis=0)                                  # dollar-neutral, foreign conv
    raw_own  = pd.DataFrame({p: (R[p].values*SIGN[p]   > 0).astype(int) for p in PAIRS}, index=R.index)
    idio_own = pd.DataFrame({p: (idio[p].values*SIGN[p] > 0).astype(int) for p in PAIRS}, index=R.index)
    return raw_own, idio_own, idio   # idio (foreign conv, continuous) for the long-short payoff

def feats_at(p, split, ts, cols):
    X = _concat(p, cols, split)
    X = X[~X.index.duplicated(keep="last")].reindex(ts)
    return X.astype("float32")

def fit_eval(label, Xtr, ytr, Xva, yva, Xte, yte, Xoo, yoo):
    m = lgb.LGBMClassifier(objective="binary", metric="auc", learning_rate=0.02, num_leaves=255,
        min_child_samples=300, subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=10,
        n_estimators=3000, n_jobs=20, verbosity=-1)
    m.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric="auc",
          callbacks=[lgb.early_stopping(150), lgb.log_evaluation(0)])
    pv, pt, po = (m.predict_proba(Xva)[:,1], m.predict_proba(Xte)[:,1], m.predict_proba(Xoo)[:,1])
    at, ao = roc_auc_score(yte, pt), roc_auc_score(yoo, po)
    # selective acc at 0.2% coverage, threshold frozen on VAL
    thr = np.quantile(np.abs(pv-0.5), 1-0.002)
    rt = H.apply_threshold(yte, pt, thr); ro = H.apply_threshold(yoo, po, thr)
    print(f"    {label:>5}: AUC test={at:.4f} oos={ao:.4f} | sel@0.2%cov "
          f"TEST {rt['accuracy']:.3f}(n{rt['n']}) OOS {ro['accuracy']:.3f}(n{ro['n']})", flush=True)
    return {"auc_test": at, "auc_oos": ao, "pt": pt, "po": po}

def main():
    cols = H.feature_cols("EURUSD")
    print(f"F2 rank — {len(cols)} features, strides {STR}", flush=True)
    T = {s: targets(s) for s in ("train","val","test","oos")}
    for s in T:
        idx = T[s][0].index
        print(f"  {s}: common-aligned 15m bars = {len(idx):,}", flush=True)
    # subsample train index by stride (val/test/oos lighter)
    keep = {s: T[s][0].index[::STR[s]] for s in T}
    res_raw, res_idio = {}, {}
    for p in PAIRS:
        t0 = time.time()
        Xtr = feats_at(p,"train",keep["train"],cols); Xva = feats_at(p,"val",keep["val"],cols)
        Xte = feats_at(p,"test",keep["test"],cols);   Xoo = feats_at(p,"oos",keep["oos"],cols)
        def yv(which, s): return T[s][which].loc[keep[s], p].values
        print(f"  [{p}] tr={Xtr.shape} te={Xte.shape} oo={Xoo.shape} ({time.time()-t0:.0f}s load)", flush=True)
        res_raw[p]  = fit_eval("RAW",  Xtr, yv(0,"train"), Xva, yv(0,"val"), Xte, yv(0,"test"), Xoo, yv(0,"oos"))
        res_idio[p] = fit_eval("IDIO", Xtr, yv(1,"train"), Xva, yv(1,"val"), Xte, yv(1,"test"), Xoo, yv(1,"oos"))
    # ---- summary ----
    print("\n==== AUC summary (RAW = per-pair sign | IDIO = dollar-neutral) ====")
    print(f"  {'pair':>7} {'rawTEST':>8} {'idioTEST':>9} {'rawOOS':>8} {'idioOOS':>9} {'dOOS':>7}")
    for p in PAIRS:
        dr = res_idio[p]["auc_oos"] - res_raw[p]["auc_oos"]
        print(f"  {p:>7} {res_raw[p]['auc_test']:>8.4f} {res_idio[p]['auc_test']:>9.4f} "
              f"{res_raw[p]['auc_oos']:>8.4f} {res_idio[p]['auc_oos']:>9.4f} {dr:>+7.4f}")
    mr_t = np.mean([res_raw[p]['auc_test'] for p in PAIRS]); mi_t = np.mean([res_idio[p]['auc_test'] for p in PAIRS])
    mr_o = np.mean([res_raw[p]['auc_oos'] for p in PAIRS]);  mi_o = np.mean([res_idio[p]['auc_oos'] for p in PAIRS])
    print(f"  {'MEAN':>7} {mr_t:>8.4f} {mi_t:>9.4f} {mr_o:>8.4f} {mi_o:>9.4f} {mi_o-mr_o:>+7.4f}")

    # ---- LONG-SHORT portfolio from idio scores (the rank target) ----
    for split in ("test","oos"):
        idx = keep[split]; idio_fc = T[split][2].loc[idx]      # foreign-conv realized idio
        # foreign-conv expected idio score per pair = SIGN * (p_idio_own - 0.5)
        S = pd.DataFrame({p: SIGN[p]*(res_idio[p]["pt" if split=="test" else "po"]-0.5) for p in PAIRS}, index=idx)
        rk = S.rank(axis=1)                                     # 1=lowest score .. 7=highest
        longs = rk >= 6; shorts = rk <= 2                       # top-2 / bottom-2
        spread = (idio_fc.where(longs).mean(axis=1) - idio_fc.where(shorts).mean(axis=1))
        sp = spread.dropna()
        hit = (sp > 0).mean(); shp = sp.mean()/(sp.std()+1e-12)*np.sqrt(252*96)  # ~96 15m-bars/day
        # also: directional accuracy of every (pair,t) idio pick
        allpick = pd.DataFrame({p: np.sign(S[p]) == np.sign(idio_fc[p]) for p in PAIRS})
        pick_acc = allpick.values[np.isfinite(idio_fc.values) & (idio_fc.values!=0)].mean()
        print(f"\n[LONG-SHORT {split.upper()}] top2-bottom2 dollar-neutral: "
              f"spread hit-rate={hit:.4f} (n={len(sp):,}) ann.Sharpe={shp:.2f} | per-pick idio acc={pick_acc:.4f}")
    print("\nGATE: idio meanAUC_OOS materially > 0.527 OR long-short OOS hit-rate > 0.571  =>  F2 PASS")

if __name__ == "__main__":
    main()
