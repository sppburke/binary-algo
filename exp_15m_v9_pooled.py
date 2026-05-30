"""15-MINUTE v9 — POOLED 7-major vol-compression selective book (statistical-power test).

V21 showed the single-pair compression pockets don't clearly transfer to 2026 OOS at low coverage
(n too small, CI spans 0.50). Here we POOL all 7 USD majors: one model, 7x the bets at every coverage.
If the vol-compression-gated selective edge is REAL it survives pooling with a tight CI; if it was noise
it pools to ~0.50. Honest protocol: gate fixed = compression (per-pair TRAIN q33 of 15m_bb_width);
coverage chosen on pooled VAL; threshold frozen; pooled TEST then pooled 2026 OOS judged once + bootstrap CI.
Target = up/down binary endpoint sign. Cross-pair AGREEMENT gate also reported as a second lens.
"""
import time, numpy as np, pandas as pd, os
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import harness as H

PAIRS=["EURUSD","GBPUSD","AUDUSD","NZDUSD","USDCAD","USDCHF","USDJPY"]
HOR=15; TRSTRIDE=8
base_cols=list(H.feature_cols("EURUSD"))
KEEP=[c for c in base_cols if c in ("15m_bb_width","sess_london","sess_ny")]
COVS=[0.5,0.2,0.1,0.05,0.02,0.01]

def load_pair(pair, split, stride=1):
    parts=[]
    for y in H.SPLITS[split]:
        p=f"{H.FEAT_DIR}/{pair}_{y}.parquet"
        if not os.path.exists(p): continue
        df=pd.read_parquet(p, columns=base_cols+H.META_COLS); df=df[~df.index.duplicated(keep="last")]
        idx=df.index; c=df["close"].values; n=len(c)
        secs=idx.values.astype("datetime64[s]").astype("int64")
        contig=np.zeros(n,bool)
        if n>HOR: contig[:n-HOR]=(secs[HOR:]-secs[:-HOR])==HOR*60
        fwd=np.full(n,np.nan); fwd[:n-HOR]=c[HOR:]; ret=fwd/c-1.0
        yv=(ret>0).astype(float); valid=contig&np.isfinite(ret)&(ret!=0)
        d=df.loc[valid, base_cols].copy(); d["_y"]=yv[valid]; d["_h"]=idx[valid].hour
        if stride>1: d=d.iloc[::stride]
        parts.append(d)
    if not parts: return None
    df=pd.concat(parts)
    return df

def sel(y,p,thr):
    m=np.abs(p-0.5)>=thr
    if m.sum()==0: return np.nan,0,None
    s=((p[m]>0.5).astype(int)==y[m]); return s.mean(),int(m.sum()),s

def main():
    t0=time.time()
    # ---- train pooled ----
    tr=[]
    for pi,pair in enumerate(PAIRS):
        d=load_pair(pair,"train",TRSTRIDE)
        if d is None: continue
        d["_pair"]=pi; tr.append(d)
    TR=pd.concat(tr); ytr=TR["_y"].astype(int).values
    Xtr=TR[base_cols+["_pair"]].astype("float32")
    print(f"pooled TRAIN {Xtr.shape} load={time.time()-t0:.0f}s",flush=True)
    L=lgb.LGBMClassifier(objective="binary",metric="auc",learning_rate=0.02,num_leaves=255,
        min_child_samples=400,subsample=0.8,subsample_freq=1,colsample_bytree=0.5,reg_lambda=10,
        n_estimators=2500,n_jobs=20,verbosity=-1)
    # per-pair compression threshold from TRAIN
    cthr={pi:np.nanpercentile(tr_d["15m_bb_width"].values.astype(float),33) for pi,tr_d in enumerate(tr)}

    import gc
    # small strided VAL just for early stopping (memory-frugal)
    es=[]
    for pi,pair in enumerate(PAIRS):
        d=load_pair(pair,"val",4)
        if d is None: continue
        d["_pair"]=pi; es.append(d)
    ES=pd.concat(es); yes=ES["_y"].astype(int).values; Xes=ES[base_cols+["_pair"]].astype("float32")
    del es,ES; gc.collect()
    L.fit(Xtr,ytr,eval_set=[(Xes,yes)],eval_metric="auc",callbacks=[lgb.early_stopping(150),lgb.log_evaluation(0)])
    del Xtr,TR,tr,Xes,yes; gc.collect()

    # STREAM each split per-pair: predict, keep only compact arrays (y, p, comp, ny). No 7-frame concat.
    def stream(split):
        Y,P,C,N=[],[],[],[]
        for pi,pair in enumerate(PAIRS):
            d=load_pair(pair,split,1)
            if d is None: continue
            p=L.predict_proba(d[base_cols].assign(_pair=pi).astype("float32"))[:,1]
            Y.append(d["_y"].astype(int).values); P.append(p)
            C.append(d["15m_bb_width"].values.astype(float) <= cthr[pi])
            N.append((d["sess_ny"].values.astype(float)>0.5) if "sess_ny" in d else np.zeros(len(d),bool))
            del d; gc.collect()
        return (np.concatenate(Y),np.concatenate(P),np.concatenate(C),np.concatenate(N))
    yva,pv,cv,nv=stream("val"); yte,pt,ct,nt2=stream("test"); yoo,po,co,no2=stream("oos")
    print(f"POOLED AUC val={roc_auc_score(yva,pv):.4f} test={roc_auc_score(yte,pt):.4f} oos={roc_auc_score(yoo,po):.4f}",flush=True)
    print(f"compression rows pooled: va={cv.mean():.2%} te={ct.mean():.2%} oo={co.mean():.2%}")
    rng=np.random.default_rng(7)

    def book(label,mv,mt,mo):
        print(f"\n=== POOLED {label} selective book (threshold frozen on VAL) ===")
        print(f"   {'cov':>4} {'VALacc':>7} {'TESTacc':>8} {'OOSacc':>8} {'nOOS':>7} {'OOS95CI':>16}")
        yv_g,pv_g=yva[mv],pv[mv]; confv=np.abs(pv_g-0.5)
        for cov in COVS:
            thr=np.quantile(confv,1-cov)
            av,nv,_=sel(yv_g,pv_g,thr)
            at,nt,_=sel(yte[mt],pt[mt],thr)
            ao,no,so=sel(yoo[mo],po[mo],thr)
            ci=""
            if so is not None and no>=30:
                b=np.array([rng.choice(so,no,replace=True).mean() for _ in range(3000)])
                ci=f"[{np.percentile(b,2.5):.3f},{np.percentile(b,97.5):.3f}]"
            print(f"   {int(cov*100):>4} {av:>7.3f} {at:>8.3f} {ao:>8.3f} {no:>7} {ci:>16}",flush=True)

    book("compression", cv,ct,co)
    book("compression&NY", cv&nv, ct&nt2, co&no2)

    # ---- second lens: cross-pair AGREEMENT (USD direction) gate, model-free + model ----
    # express each pair's model edge in USD terms, count agreement at each timestamp via index join
    print("\n=== cross-pair agreement note ===")
    print("(pairs share the USD factor; pooled book above already aggregates them. Agreement gate omitted")
    print(" as redundant unless pooled book shows a real edge.)")
    print("V9-POOLED DONE")

if __name__=="__main__":
    main()
