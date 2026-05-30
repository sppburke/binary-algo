"""Multi-horizon test: the Reddit author's key advice was 'start with 15m' (5m AUC 0.51 vs
15m 0.57). Our causal features are horizon-independent, so we recompute the label at horizon H
from the cached 1m 'close' and reuse the full feature set. Tests how predictability + selective
accuracy scale with horizon: H in {5,10,15,30,60} minutes.

Validity: require the window [t, t+H] be contiguous (timestamp H bars ahead == t + H minutes),
so labels never span weekend/holiday gaps. 2026 stays fully OOS."""
import sys, time, numpy as np, pandas as pd, lightgbm as lgb, os
from sklearn.metrics import roc_auc_score
import harness as H, crosspair as CP

PAIR="EURUSD"; HOR=int(sys.argv[1]) if len(sys.argv)>1 else 15
STRIDE=4
base_cols=list(H.feature_cols(PAIR))
cross_cols=CP.cross_feature_names(PAIR)
feats=base_cols+cross_cols

def load_split(split, stride=1):
    parts=[]
    for y in H.SPLITS[split]:
        df=CP.load_year_merged(PAIR,y,base_cols=base_cols+H.META_COLS)
        if df is None: continue
        idx=df.index
        c=df["close"].values
        n=len(c)
        # contiguity: timestamp H ahead == t + H minutes
        # resolution-proof epoch seconds (some years are datetime64[us], others [ns])
        secs=idx.values.astype("datetime64[s]").astype("int64")
        contig=np.zeros(n,dtype=bool)
        if n>HOR:
            contig[:n-HOR]=(secs[HOR:]-secs[:-HOR])==HOR*60
        fwd=np.full(n,np.nan); fwd[:n-HOR]=c[HOR:]
        ret=fwd/c-1.0
        y=(ret>0).astype(float)
        valid=contig & np.isfinite(ret) & (ret!=0)
        d=df[feats].copy(); d["_y"]=y; d["_valid"]=valid
        d=d[d["_valid"]]
        if stride>1: d=d.iloc[::stride]
        parts.append(d)
    df=pd.concat(parts)
    X=df[feats].astype("float32"); y=df["_y"].astype(int).values
    return X,y

def main():
    t0=time.time()
    Xtr,ytr=load_split("train",STRIDE); Xva,yva=load_split("val"); Xte,yte=load_split("test"); Xoo,yoo=load_split("oos")
    print(f"H={HOR}m shapes tr={Xtr.shape} va={Xva.shape} te={Xte.shape} oo={Xoo.shape} ymean_tr={ytr.mean():.4f} load={time.time()-t0:.0f}s",flush=True)
    m=lgb.LGBMClassifier(objective="binary",metric="auc",learning_rate=0.03,num_leaves=255,
        min_child_samples=300,subsample=0.8,subsample_freq=1,colsample_bytree=0.5,reg_lambda=10.0,
        n_estimators=4000,n_jobs=20,verbosity=-1)
    m.fit(Xtr,ytr,eval_set=[(Xva,yva)],eval_metric="auc",callbacks=[lgb.early_stopping(150),lgb.log_evaluation(0)])
    pv=m.predict_proba(Xva)[:,1]; pt=m.predict_proba(Xte)[:,1]; po=m.predict_proba(Xoo)[:,1]
    print(f"H={HOR}m AUC val={roc_auc_score(yva,pv):.4f} test={roc_auc_score(yte,pt):.4f} oos={roc_auc_score(yoo,po):.4f}")
    H.report(f"TEST H{HOR}",yte,pt); H.report(f"OOS  H{HOR}",yoo,po)
    print(f"\n--- selective (VAL thr) H={HOR}m ---")
    for tgt in (0.75,0.70,0.65,0.60):
        bv=H.threshold_for_target(yva,pv,target=tgt,min_n=300)
        if bv is None: print(f"target {tgt:.0%}: not hit on VAL"); continue
        rt=H.apply_threshold(yte,pt,bv['conf_thr']); ro=H.apply_threshold(yoo,po,bv['conf_thr'])
        print(f"target {tgt:.0%}: VAL cov={bv['coverage']:.3%} -> TEST cov={rt['coverage']:.3%} acc={rt['accuracy']:.3f} n={rt['n']}"
              f" | OOS cov={ro['coverage']:.3%} acc={ro['accuracy']:.3f} n={ro['n']}")
    np.savez(f"models/probs_H{HOR}.npz",pva=pv,pte=pt,poo=po,yva=yva,yte=yte,yoo=yoo)
    print(f"H{HOR} DONE")

if __name__=="__main__":
    main()
