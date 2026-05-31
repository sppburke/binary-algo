"""30-MINUTE EURUSD binary direction — EXPERIMENTAL LAB.

Honest, OOS-verified selective accuracy. Reuses the 239 causal multi-TF features (pipeline.py); recomputes the
label at HOR=30 1-min bars with strict wall-clock contiguity. Settlement = deriv Rise/Fall close-to-close
(next-tick entry negligible at 30m); ties lose; payout-deduction EV (breakeven ~0.541). NO spread.

Two-stage so I can iterate on GATES without retraining:
  python m30_lab.py train <tag> [featset]   # fit model(s), cache probs+regime per window -> models/m5_<tag>.npz
  python m30_lab.py eval  <tag>             # load cache, sweep gates x coverages, honest indep acc + CI95 per window

Eval discipline: conf threshold is picked on VAL for a target coverage *inside the gate*, then the SAME threshold is
applied to TEST24 / TEST25 / OOS26. Independent trades only (non-overlap chronological, gap=1800s). A config only
"counts" if acc holds across ALL held-out windows with CI95 excluding breakeven (corr(VAL,OOS) was -0.54 historically).
"""
import sys, os, json, time, numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import harness as H

MODELS = "/media/sean/CORSAIR/binary-algo/models"
HOR = 5; GAP_S = HOR*60
PAIR = "EURUSD"
base = list(H.feature_cols(PAIR))
WINDOWS = {"val":["2022","2023"], "test24":["2024"], "test25":["2025"], "oos":["2026"]}

# regime columns persisted alongside probs so gates can be swept without reloading the 239-feat parquets
REGIME = ["15m_bb_width","30m_bb_width","1h_bb_width","4h_bb_width",
          "sess_london","sess_ny","sess_overlap","hour_sin","hour_cos","dow","gap_prev","vol_z",
          "30m_rsi","1h_rsi","4h_rsi","15m_rsi",
          "30m_autocorr_10","1h_autocorr_10","15m_autocorr_10",
          "30m_above_ema50","1h_above_ema50","4h_above_ema50","mtf_trend_align","mtf_rsi_mean",
          "30m_atr_pct","1h_atr_pct","30m_rv_24","1h_rv_24",
          "30m_rangepos_24","1h_rangepos_24","30m_slope_20","1h_slope_20",
          "30m_dist_ema50","1h_dist_ema50","30m_bb_pctb","1h_bb_pctb","30m_macd_pos","1h_macd_pos"]
REGIME = [c for c in REGIME if c in base]

def load(years, stride=1, cols=None):
    cols = base if cols is None else cols
    parts=[]
    for y in years:
        p=f"{H.FEAT_DIR}/{PAIR}_{y}.parquet"
        if not os.path.exists(p): continue
        df=pd.read_parquet(p, columns=list(dict.fromkeys(cols+["close"]))); df=df[~df.index.duplicated(keep="last")]
        c=df["close"].values; n=len(c)
        secs=df.index.values.astype("datetime64[s]").astype("int64")
        contig=np.zeros(n,bool)
        if n>HOR: contig[:n-HOR]=(secs[HOR:]-secs[:-HOR])==HOR*60
        fwd=np.full(n,np.nan); fwd[:n-HOR]=c[HOR:]; ret=fwd/c-1.0
        yv=(ret>0).astype(float); valid=contig&np.isfinite(ret)&(ret!=0)
        d=df.loc[valid,[x for x in cols if x in df.columns]].copy()
        d["_y"]=yv[valid]; d["_ts"]=secs[valid]
        parts.append(d.iloc[::stride] if stride>1 else d)
    return pd.concat(parts)

def mk_lgb(**kw):
    p=dict(objective="binary",metric="auc",learning_rate=0.02,num_leaves=255,min_child_samples=300,
           subsample=0.8,subsample_freq=1,colsample_bytree=0.5,reg_lambda=10.0,n_estimators=3000,n_jobs=20,verbosity=-1)
    p.update(kw); return lgb.LGBMClassifier(**p)

# ---------------- honest selective evaluation ----------------
def nonoverlap_chrono(ts, mask, gap=GAP_S):
    take=[]; block_until=-1
    for i in np.where(mask)[0]:
        if ts[i] < block_until: continue
        take.append(i); block_until=int(ts[i])+gap
    return np.array(take,dtype=int)

def boot(corr, nb=5000, seed=7):
    corr=np.asarray(corr,dtype=float)
    if len(corr)<5: return (float("nan"),float("nan"))
    rng=np.random.default_rng(seed); n=len(corr)
    a=np.array([corr[rng.integers(0,n,n)].mean() for _ in range(nb)])
    return float(np.percentile(a,2.5)),float(np.percentile(a,97.5))

def indep_acc(p, y, ts, gatemask, conf_thr):
    """Independent (non-overlap chrono) accuracy among gate & confidence>=thr. Returns (n, acc, lo, hi, ev85)."""
    m = gatemask & (np.abs(p-0.5) >= conf_thr)
    if m.sum()==0: return (0, float("nan"), float("nan"), float("nan"), float("nan"))
    sel = nonoverlap_chrono(ts, m)
    if len(sel)==0: return (0, float("nan"), float("nan"), float("nan"), float("nan"))
    corr = ((p[sel]>0.5).astype(int)==y[sel]).astype(float)
    acc=corr.mean(); lo,hi=boot(corr); ev=acc*0.85-(1-acc)
    return (len(sel), acc, lo, hi, ev)

def thr_for_cov(p, y, gatemask, cov):
    """conf threshold so that within the gate, top-`cov` fraction (by confidence) is selected (set on VAL)."""
    conf=np.abs(p-0.5)[gatemask]
    if len(conf)<50: return None
    return float(np.quantile(conf, 1-cov))

# ---------------- gates ----------------
def build_gates(R, qthr):
    """R: dict of regime arrays for one window. qthr: dict of TRAIN percentile thresholds. Returns {name: bool mask}."""
    n=len(R["_y"]); ones=np.ones(n,bool)
    def col(c): return R[c].astype(float) if c in R else np.full(n,np.nan)
    g={}
    g["none"]=ones
    g["ny"]=col("sess_ny")>0.5
    g["london"]=col("sess_london")>0.5
    # compression on each TF (bottom-tertile band width, TRAIN threshold)
    for tf in ("15m","30m","1h"):
        c=f"{tf}_bb_width"
        if c in R: g[f"comp_{tf}"]=col(c)<=qthr.get(c,np.nan)
    if "comp_30m" in g: g["comp30_ny"]=g["comp_30m"]&(col("sess_ny")>0.5)
    if "comp_15m" in g: g["comp15_ny"]=g["comp_15m"]&(col("sess_ny")>0.5)
    if "comp_1h" in g:  g["comp1h_ny"]=g["comp_1h"]&(col("sess_ny")>0.5)
    # trend-confluence: all of 30m/1h/4h on same side of ema50
    if all(f"{tf}_above_ema50" in R for tf in ("30m","1h","4h")):
        up=(col("30m_above_ema50")>0.5)&(col("1h_above_ema50")>0.5)&(col("4h_above_ema50")>0.5)
        dn=(col("30m_above_ema50")<0.5)&(col("1h_above_ema50")<0.5)&(col("4h_above_ema50")<0.5)
        g["trend_align"]=up|dn
    # expansion (top-tertile) — momentum regime
    for tf in ("30m","1h"):
        c=f"{tf}_bb_width"
        if c in R: g[f"expand_{tf}"]=col(c)>=qthr.get(c+"_hi",np.nan)
    return g

def main_train(tag, featset="all"):
    t0=time.time(); os.makedirs(MODELS,exist_ok=True)
    cols = base if featset=="all" else [c for c in base if c]  # placeholder for future subsets
    STRIDE=3
    print(f"[train:{tag}] loading TRAIN 2012-2021 stride{STRIDE}...",flush=True)
    TR=load([str(y) for y in range(2012,2022)], STRIDE, cols=cols)
    VA=load(WINDOWS["val"],1,cols=cols)
    ytr=TR["_y"].astype(int).values; yva=VA["_y"].astype(int).values
    feats=[c for c in cols if c in TR.columns]
    print(f"[train:{tag}] train={len(TR):,} val={len(VA):,} feats={len(feats)} load={time.time()-t0:.0f}s",flush=True)
    L=mk_lgb()
    L.fit(TR[feats].astype("float32"),ytr,eval_set=[(VA[feats].astype("float32"),yva)],
          eval_metric="auc",callbacks=[lgb.early_stopping(200),lgb.log_evaluation(0)])
    print(f"[train:{tag}] fit done {time.time()-t0:.0f}s best_iter={L.best_iteration_}",flush=True)
    # TRAIN percentile thresholds for compression/expansion gates
    qthr={}
    for c in REGIME:
        if c.endswith("bb_width") and c in TR.columns:
            v=TR[c].values.astype(float)
            qthr[c]=float(np.nanpercentile(v,33)); qthr[c+"_hi"]=float(np.nanpercentile(v,67))
    # cache probs + regime per window
    cache={"feats":feats,"qthr":json.dumps(qthr),"best_iter":L.best_iteration_ or 0}
    aucs={}
    for w,yrs in WINDOWS.items():
        D=load(yrs,1,cols=cols)
        p=L.predict_proba(D[feats].astype("float32"))[:,1]
        y=D["_y"].astype(int).values; ts=D["_ts"].values.astype("int64")
        aucs[w]=float(roc_auc_score(y,p))
        cache[f"{w}_p"]=p.astype("float32"); cache[f"{w}_y"]=y.astype("int8"); cache[f"{w}_ts"]=ts
        for c in REGIME:
            if c in D.columns: cache[f"{w}_r_{c}"]=D[c].values.astype("float32")
        print(f"[train:{tag}] {w}: n={len(y):,} AUC={aucs[w]:.4f}",flush=True)
        del D
    np.savez_compressed(f"{MODELS}/m5_{tag}.npz", **cache)
    print(f"[train:{tag}] cached models/m5_{tag}.npz  AUC "+" ".join(f"{w}={aucs[w]:.4f}" for w in WINDOWS)+f"  DONE {time.time()-t0:.0f}s",flush=True)

def _load_window(z, w):
    R={"_y":z[f"{w}_y"], "_p":z[f"{w}_p"], "_ts":z[f"{w}_ts"]}
    for k in z.files:
        if k.startswith(f"{w}_r_"): R[k[len(f"{w}_r_"):]]=z[k]
    return R

def main_eval(tag, covs=(0.10,0.05,0.02,0.01)):
    z=np.load(f"{MODELS}/m5_{tag}.npz", allow_pickle=True)
    qthr=json.loads(str(z["qthr"]))
    W={w:_load_window(z,w) for w in WINDOWS}
    print(f"\n===== EVAL m5_{tag} =====  (indep non-overlap {GAP_S}s, chronological; CI95 boot; breakeven~0.541)")
    print("AUC: "+"  ".join(f"{w}={roc_auc_score(W[w]['_y'],W[w]['_p']):.4f}" for w in WINDOWS))
    Gs={w:build_gates(W[w],qthr) for w in WINDOWS}
    gate_names=list(Gs["val"].keys())
    for gname in gate_names:
        gval=Gs["val"][gname]
        if gval.sum()<200: continue
        print(f"\n--- gate={gname}  (VAL gate n={int(gval.sum()):,}) ---")
        for cov in covs:
            thr=thr_for_cov(W["val"]["_p"], W["val"]["_y"], gval, cov)
            if thr is None: continue
            row=[]
            for w in WINDOWS:
                g=Gs[w].get(gname, np.ones(len(W[w]["_y"]),bool))
                n,acc,lo,hi,ev=indep_acc(W[w]["_p"],W[w]["_y"],W[w]["_ts"],g,thr)
                row.append(f"{w}:n{n} acc={acc:.3f}[{lo:.3f},{hi:.3f}]" if n else f"{w}:n0")
            # held-out flag: all of test24/test25/oos acc>=0.75 with lo>0.5
            print(f"  cov{cov:.0%} thr={thr:.4f}  "+"  ".join(row))

if __name__=="__main__":
    mode=sys.argv[1] if len(sys.argv)>1 else "eval"
    tag=sys.argv[2] if len(sys.argv)>2 else "base"
    if mode=="train": main_train(tag, sys.argv[3] if len(sys.argv)>3 else "all")
    else: main_eval(tag)
