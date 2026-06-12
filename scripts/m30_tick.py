"""30m EURUSD — Iteration 5: TICK-MICROSTRUCTURE features for a 30-min binary (the untried data source).

Every prior 30m attempt used 1-min OHLCV indicators. Here the decision uses sub-second microstructure STATE:
order-flow imbalance (imb + persistence), microprice deviation, realized-vol regime (rv30..rv3600), spread,
tick intensity, stretch/range-position/bbw at up to 3600s — reusing min1_production.feats (62 causal feats).
Label = deriv-faithful Rise/Fall at HS=1800s (wc_ret: next-tick entry, last tick<=expiry, mid-to-mid, ties lose).

Hypothesis: even though order-flow PRICE-IMPACT decays by ~15m (FRB/EBS), the microstructure REGIME (vol state,
imbalance persistence) may condition the 30m direction enough to lift the selective tail above the 0.59 OHLCV frontier.
Honest eval: select threshold on VAL(2024H1), verify across TEST-2024 / TEST-2025 / OOS-2026, independent
(non-overlap chrono 1810s), CI95. A result counts only if it holds across all held-out windows.

  python m30_tick.py train   # train LGB, cache probs+regime -> models/m30tick.npz, print selective table
  python m30_tick.py eval
"""
import sys, os, json, time, numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from min1_production import feats, wc_ret, nonoverlap_chrono, boot, load_split, set_pair

MODELS="/home/sean/git/binary-algo/models"
HS=1800; TOL=10; GAP=HS+TOL; LAG=1      # deriv 30-min; next-tick entry negligible at 1800s
set_pair("EURUSD")

def prep(b, stride=1):
    """Return feats X (strided rows), y, valid, ts, hour, and regime cols — label at HS=1800 deriv-faithful."""
    X=feats(b); mid=b["mid"].values.astype(float)
    ts=b.index.values.astype("datetime64[s]").astype("int64")
    ret,valid=wc_ret(ts,mid,HS,TOL,LAG); y=(ret>0).astype(int)
    hour=b.index.hour.values; year=b.index.year.values
    if stride>1:
        idx=np.arange(0,len(X),stride)
        return X.iloc[idx], y[idx], valid[idx], ts[idx], hour[idx], year[idx]
    return X, y, valid, ts, hour, year

def mk_lgb(): return lgb.LGBMClassifier(objective="binary",metric="auc",learning_rate=0.02,num_leaves=350,
    min_child_samples=200,subsample=0.8,subsample_freq=1,colsample_bytree=0.6,reg_lambda=8,n_estimators=4000,n_jobs=20,verbosity=-1)

def gates(X, hour, qbbw):
    ny=(hour>=12)&(hour<21)
    bbw=X["bbw1800"].values.astype(float); comp=bbw<=qbbw
    return {"none":np.ones(len(X),bool),"ny":ny,"comp":comp,"comp_ny":comp&ny}

def indep(p,y,ts,gate,thr):
    m=gate&(np.abs(p-0.5)>=thr)
    if m.sum()==0: return (0,float("nan"),float("nan"),float("nan"))
    sel=nonoverlap_chrono(ts,m,GAP)
    if len(sel)==0: return (0,float("nan"),float("nan"),float("nan"))
    corr=((p[sel]>0.5).astype(int)==y[sel]).astype(float)
    a=corr.mean(); lo,hi=boot(corr); return (len(sel),a,lo,hi)

def train():
    t0=time.time(); os.makedirs(MODELS,exist_ok=True)
    print("[tick30] prep TRAIN (13.8M 1s rows -> feats, stride 20)...",flush=True)
    btr=load_split("train"); Xtr,ytr,vtr,tstr,htr,_=prep(btr, stride=20)
    bva=load_split("val");   Xva,yva,vva,tsva,hva,_=prep(bva, stride=1)
    del btr,bva
    itr=np.where(vtr)[0]; iva=np.where(vva)[0]
    qbbw=float(np.nanpercentile(Xtr["bbw1800"].values.astype(float)[vtr],33))
    print(f"[tick30] train_valid={len(itr):,} val_valid={len(iva):,} feats={Xtr.shape[1]} prep={time.time()-t0:.0f}s",flush=True)
    L=mk_lgb()
    L.fit(Xtr.iloc[itr],ytr[itr],eval_set=[(Xva.iloc[iva],yva[iva])],eval_metric="auc",
          callbacks=[lgb.early_stopping(150),lgb.log_evaluation(0)])
    print(f"[tick30] lgb iter={L.best_iteration_} auc_val={L.best_score_['valid_0']['auc']:.4f} {time.time()-t0:.0f}s",flush=True)
    feat_names=list(Xtr.columns); del Xtr
    cache={"feat_names":json.dumps(feat_names),"qbbw":qbbw}
    # windows: val(2024H1), test split into 2024 & 2025, oos(2026)
    def add(tag, b):
        X,y,v,ts,hour,year=prep(b, stride=1)
        Xf=X[feat_names]; p=L.predict_proba(Xf)[:,1]
        cache[f"{tag}_p"]=p.astype("float32")[v]; cache[f"{tag}_y"]=y.astype("int8")[v]
        cache[f"{tag}_ts"]=ts[v]; cache[f"{tag}_hour"]=hour.astype("int8")[v]
        cache[f"{tag}_bbw"]=X["bbw1800"].values.astype("float32")[v]
        print(f"[tick30] {tag}: n={int(v.sum()):,} AUC={roc_auc_score(y[v],p[v]):.4f}",flush=True)
        return p,y,v,ts,hour,year
    cache["val_p"]=L.predict_proba(Xva[feat_names].iloc[iva])[:,1].astype("float32")
    cache["val_y"]=yva[iva].astype("int8"); cache["val_ts"]=tsva[iva]; cache["val_hour"]=hva[iva].astype("int8")
    cache["val_bbw"]=Xva["bbw1800"].values.astype("float32")[iva]
    print(f"[tick30] val: n={len(iva):,} AUC={roc_auc_score(yva[iva],cache['val_p']):.4f}",flush=True)
    del Xva
    bte=load_split("test"); Xte,yte,vte,tste,hte,yrte=prep(bte, stride=1); del bte
    Xtef=Xte[feat_names]; pte=L.predict_proba(Xtef)[:,1]
    for yr,tag in ((2024,"test24"),(2025,"test25")):
        sel=(yrte==yr)&vte
        cache[f"{tag}_p"]=pte.astype("float32")[sel]; cache[f"{tag}_y"]=yte.astype("int8")[sel]
        cache[f"{tag}_ts"]=tste[sel]; cache[f"{tag}_hour"]=hte.astype("int8")[sel]
        cache[f"{tag}_bbw"]=Xte["bbw1800"].values.astype("float32")[sel]
        print(f"[tick30] {tag}: n={int(sel.sum()):,} AUC={roc_auc_score(yte[sel],pte[sel]):.4f}",flush=True)
    del Xte,Xtef
    boo=load_split("oos"); Xoo,yoo,voo,tsoo,hoo,_=prep(boo, stride=1); del boo
    poo=L.predict_proba(Xoo[feat_names])[:,1]
    cache["oos_p"]=poo.astype("float32")[voo]; cache["oos_y"]=yoo.astype("int8")[voo]
    cache["oos_ts"]=tsoo[voo]; cache["oos_hour"]=hoo.astype("int8")[voo]; cache["oos_bbw"]=Xoo["bbw1800"].values.astype("float32")[voo]
    print(f"[tick30] oos: n={int(voo.sum()):,} AUC={roc_auc_score(yoo[voo],poo[voo]):.4f}",flush=True)
    np.savez_compressed(f"{MODELS}/m30tick.npz", **cache)
    print(f"[tick30] cached DONE {time.time()-t0:.0f}s",flush=True)
    eval_cache()

def eval_cache():
    z=np.load(f"{MODELS}/m30tick.npz",allow_pickle=True); qbbw=float(z["qbbw"])
    WIN=["val","test24","test25","oos"]
    W={w:{"p":z[f"{w}_p"],"y":z[f"{w}_y"],"ts":z[f"{w}_ts"],"hour":z[f"{w}_hour"],"bbw":z[f"{w}_bbw"]} for w in WIN}
    print(f"\n===== EVAL m30tick (microstructure) =====  indep {GAP}s chrono; CI95; breakeven~0.541")
    print("AUC: "+"  ".join(f"{w}={roc_auc_score(W[w]['y'],W[w]['p']):.4f}" for w in WIN))
    class GX:  # gate per window
        pass
    def gate_of(w,name):
        hour=W[w]["hour"]; ny=(hour>=12)&(hour<21); comp=W[w]["bbw"]<=qbbw
        return {"none":np.ones(len(hour),bool),"ny":ny,"comp":comp,"comp_ny":comp&ny}[name]
    for gname in ("none","ny","comp","comp_ny"):
        gval=gate_of("val",gname)
        if gval.sum()<200: continue
        print(f"\n--- gate={gname} (VAL n={int(gval.sum()):,}) ---")
        for cov in (0.10,0.05,0.02,0.01):
            conf=np.abs(W["val"]["p"]-0.5)[gval]
            if len(conf)<50: continue
            thr=float(np.quantile(conf,1-cov))
            cells=[]
            for w in WIN:
                g=gate_of(w,gname); n,a,lo,hi=indep(W[w]["p"],W[w]["y"],W[w]["ts"],g,thr)
                cells.append(f"{w}:n{n} {a:.3f}[{lo:.2f},{hi:.2f}]" if n else f"{w}:n0")
            print(f"  cov{cov:.0%} thr={thr:.4f}  "+"  ".join(cells))

if __name__=="__main__":
    mode=sys.argv[1] if len(sys.argv)>1 else "eval"
    (train if mode=="train" else eval_cache)()
