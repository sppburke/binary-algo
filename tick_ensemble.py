"""Seconds-horizon ENSEMBLE (LGBM+XGB+CatBoost) on cached 1s bars, broad data.
Goal: a CLEAN both-sides >=75% (TEST and 2026 OOS) directional edge at H=3s/5s.
Reports full accuracy@coverage; flags whether both splits clear 75% at meaningful n."""
import sys, time, numpy as np, pandas as pd
import lightgbm as lgb, xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
import harness as H
TICK="/media/sean/CORSAIR/binary-algo/features_tick"
def make(b, HS):
    mid=b["mid"]; imb=b["imb"].fillna(0); micro=b["micro"]; X=pd.DataFrame(index=b.index); X["imb"]=imb
    for w in (2,3,5,8,13,21,34): X[f"imb_ema{w}"]=imb.ewm(span=w).mean(); X[f"ret{w}"]=mid.pct_change(w)
    X["imb_acc"]=imb.ewm(span=3).mean()-imb.ewm(span=13).mean(); X["imb_chg"]=imb.diff(2)
    md=(micro-mid)/mid; X["micro_dev"]=md
    for w in (3,5,10): X[f"micro_dev_ema{w}"]=md.ewm(span=w).mean()
    X["micro_mom5"]=micro/micro.shift(5)-1; X["micro_mom15"]=micro/micro.shift(15)-1
    X["spread"]=b["spread"]; X["spread_ema10"]=b["spread"].ewm(span=10).mean()
    X["nt"]=b["nt"]; X["nt_ema10"]=b["nt"].ewm(span=10).mean(); X["tsz"]=b["tsz"]
    X["rv10"]=mid.pct_change().rolling(10).std(); X["rv30"]=mid.pct_change().rolling(30).std()
    hh=b.index.hour; X["hsin"]=np.sin(2*np.pi*hh/24); X["hcos"]=np.cos(2*np.pi*hh/24)
    fwd=mid.shift(-HS); ret=fwd/mid-1; y=(ret>0).astype(float); y[ret==0]=np.nan
    return X.replace([np.inf,-np.inf],np.nan).astype("float32"), y
raw={sp:pd.read_parquet(f"{TICK}/{sp}_1s.parquet") for sp in ["train","val","test","oos"]}
for sp in raw: print(f"{sp}: {len(raw[sp]):,} 1s bars",flush=True)
for HS in (int(sys.argv[1]),) if len(sys.argv)>1 else (3,5):
    Xtr,ytr=make(raw["train"],HS); Xva,yva=make(raw["val"],HS); Xte,yte=make(raw["test"],HS); Xoo,yoo=make(raw["oos"],HS)
    mtr,mva,mte,moo=ytr.notna(),yva.notna(),yte.notna(),yoo.notna()
    Xtr,ytr=Xtr[mtr],ytr[mtr].astype(int).values; Xva,yva=Xva[mva],yva[mva].astype(int).values
    Xte,yte=Xte[mte],yte[mte].astype(int).values; Xoo,yoo=Xoo[moo],yoo[moo].astype(int).values
    print(f"\n##### H={HS}s  train={len(Xtr):,} test={len(Xte):,} oos={len(Xoo):,} #####",flush=True)
    P={}
    L=lgb.LGBMClassifier(objective="binary",metric="auc",learning_rate=0.02,num_leaves=255,min_child_samples=120,
        subsample=0.8,subsample_freq=1,colsample_bytree=0.6,reg_lambda=8,n_estimators=5000,n_jobs=20,verbosity=-1)
    L.fit(Xtr,ytr,eval_set=[(Xva,yva)],eval_metric="auc",callbacks=[lgb.early_stopping(120),lgb.log_evaluation(0)])
    P["lgb"]=(L.predict_proba(Xva)[:,1],L.predict_proba(Xte)[:,1],L.predict_proba(Xoo)[:,1])
    Xg=xgb.XGBClassifier(n_estimators=3000,learning_rate=0.02,max_depth=8,subsample=0.8,colsample_bytree=0.6,
        reg_lambda=8,tree_method="hist",n_jobs=20,eval_metric="auc",early_stopping_rounds=120)
    Xg.fit(Xtr,ytr,eval_set=[(Xva,yva)],verbose=False)
    P["xgb"]=(Xg.predict_proba(Xva)[:,1],Xg.predict_proba(Xte)[:,1],Xg.predict_proba(Xoo)[:,1])
    C=CatBoostClassifier(iterations=3000,learning_rate=0.02,depth=8,l2_leaf_reg=8,eval_metric="AUC",
        thread_count=20,verbose=False,early_stopping_rounds=120)
    C.fit(Xtr.fillna(-999),ytr,eval_set=(Xva.fillna(-999),yva))
    P["cat"]=(C.predict_proba(Xva.fillna(-999))[:,1],C.predict_proba(Xte.fillna(-999))[:,1],C.predict_proba(Xoo.fillna(-999))[:,1])
    for nm in P: print(f"  {nm} AUC test={roc_auc_score(yte,P[nm][1]):.4f} oos={roc_auc_score(yoo,P[nm][2]):.4f}",flush=True)
    pv=np.mean([P[n][0] for n in P],0); pt=np.mean([P[n][1] for n in P],0); po=np.mean([P[n][2] for n in P],0)
    print(f"  BLEND AUC val={roc_auc_score(yva,pv):.4f} test={roc_auc_score(yte,pt):.4f} oos={roc_auc_score(yoo,po):.4f}")
    print(f"  {'cov':>7} {'TESTacc':>8} {'TESTn':>8} {'OOSacc':>8} {'OOSn':>7} {'both>=75?':>9}")
    for cov in (0.02,0.01,0.005,0.002,0.001,0.0005):
        thr=np.quantile(np.abs(pv-0.5),1-cov)
        rt=H.apply_threshold(yte,pt,thr); ro=H.apply_threshold(yoo,po,thr)
        ok="YES" if (rt['accuracy']>=0.75 and ro['accuracy']>=0.75 and rt['n']>=100 and ro['n']>=100) else ""
        print(f"  {cov:>7.3%} {rt['accuracy']:>8.4f} {rt['n']:>8} {ro['accuracy']:>8.4f} {ro['n']:>7} {ok:>9}")
    np.savez(f"models/probs_tickens_H{HS}.npz",pva=pv,pte=pt,poo=po,yva=yva,yte=yte,yoo=yoo)
print("ENSEMBLE DONE")
