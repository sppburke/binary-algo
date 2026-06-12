"""15-MINUTE v2: base MTF + cross-pair + DAILY/WEEKLY context features + ensemble + selective.
Daily context (multi-day trend, range position, seasonality, intraday position) should help most
at the 15m horizon. 2026 OOS held out."""
import time, numpy as np, pandas as pd, os
import lightgbm as lgb, xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
import harness as H, crosspair as CP, dailyctx as DC

HOR=15; STRIDE=3
base_cols=list(H.feature_cols("EURUSD")); cross_cols=CP.cross_feature_names("EURUSD")
daily_cols=DC.daily_feature_names()
feats=base_cols+cross_cols+daily_cols

def load_split(split, stride=1):
    parts=[]
    for y in H.SPLITS[split]:
        df=CP.load_year_merged("EURUSD",y,base_cols=base_cols+H.META_COLS)
        if df is None: continue
        dly=DC.build_daily_features(df["close"]); df=pd.concat([df,dly],axis=1)
        idx=df.index; c=df["close"].values; n=len(c)
        secs=idx.values.astype("datetime64[s]").astype("int64")
        contig=np.zeros(n,dtype=bool)
        if n>HOR: contig[:n-HOR]=(secs[HOR:]-secs[:-HOR])==HOR*60
        fwd=np.full(n,np.nan); fwd[:n-HOR]=c[HOR:]; ret=fwd/c-1.0
        y=(ret>0).astype(float); valid=contig&np.isfinite(ret)&(ret!=0)
        d=df.loc[valid,[col for col in feats if col in df.columns]].copy()
        d["_y"]=y[valid]
        if stride>1: d=d.iloc[::stride]
        parts.append(d)
    df=pd.concat(parts)
    for f in feats:
        if f not in df.columns: df[f]=np.nan
    return df[feats].astype("float32"), df["_y"].astype(int).values

t0=time.time()
Xtr,ytr=load_split("train",STRIDE); Xva,yva=load_split("val"); Xte,yte=load_split("test"); Xoo,yoo=load_split("oos")
print(f"15m-v2 shapes tr={Xtr.shape} va={Xva.shape} te={Xte.shape} oo={Xoo.shape} nfeat={len(feats)} ymean={ytr.mean():.4f} load={time.time()-t0:.0f}s",flush=True)
P={}
L=lgb.LGBMClassifier(objective="binary",metric="auc",learning_rate=0.02,num_leaves=255,min_child_samples=300,
    subsample=0.8,subsample_freq=1,colsample_bytree=0.5,reg_lambda=10,n_estimators=5000,n_jobs=20,verbosity=-1)
L.fit(Xtr,ytr,eval_set=[(Xva,yva)],eval_metric="auc",callbacks=[lgb.early_stopping(150),lgb.log_evaluation(0)])
P["lgb"]=(L.predict_proba(Xva)[:,1],L.predict_proba(Xte)[:,1],L.predict_proba(Xoo)[:,1])
print(f"lgb AUC test={roc_auc_score(yte,P['lgb'][1]):.4f} oos={roc_auc_score(yoo,P['lgb'][2]):.4f}",flush=True)
G=xgb.XGBClassifier(n_estimators=3000,learning_rate=0.02,max_depth=8,subsample=0.8,colsample_bytree=0.5,
    reg_lambda=10,tree_method="hist",n_jobs=20,eval_metric="auc",early_stopping_rounds=150)
G.fit(Xtr,ytr,eval_set=[(Xva,yva)],verbose=False)
P["xgb"]=(G.predict_proba(Xva)[:,1],G.predict_proba(Xte)[:,1],G.predict_proba(Xoo)[:,1])
print(f"xgb AUC test={roc_auc_score(yte,P['xgb'][1]):.4f} oos={roc_auc_score(yoo,P['xgb'][2]):.4f}",flush=True)
C=CatBoostClassifier(iterations=3000,learning_rate=0.02,depth=8,l2_leaf_reg=10,eval_metric="AUC",
    thread_count=20,verbose=False,early_stopping_rounds=150)
C.fit(Xtr.fillna(-999),ytr,eval_set=(Xva.fillna(-999),yva))
P["cat"]=(C.predict_proba(Xva.fillna(-999))[:,1],C.predict_proba(Xte.fillna(-999))[:,1],C.predict_proba(Xoo.fillna(-999))[:,1])
print(f"cat AUC test={roc_auc_score(yte,P['cat'][1]):.4f} oos={roc_auc_score(yoo,P['cat'][2]):.4f}",flush=True)
pv=np.mean([P[n][0] for n in P],0); pt=np.mean([P[n][1] for n in P],0); po=np.mean([P[n][2] for n in P],0)
print(f"\nBLEND 15m-v2 AUC val={roc_auc_score(yva,pv):.4f} test={roc_auc_score(yte,pt):.4f} oos={roc_auc_score(yoo,po):.4f}")
print(f"  {'cov':>7} {'TESTacc':>8} {'TESTn':>8} {'OOSacc':>8} {'OOSn':>7} {'both>=75?':>9}")
for cov in (0.2,0.1,0.05,0.02,0.01,0.005,0.002):
    thr=np.quantile(np.abs(pv-0.5),1-cov)
    rt=H.apply_threshold(yte,pt,thr); ro=H.apply_threshold(yoo,po,thr)
    ok="YES" if (rt['accuracy']>=0.75 and ro['accuracy']>=0.75 and rt['n']>=100 and ro['n']>=100) else ""
    print(f"  {cov:>7.2%} {rt['accuracy']:>8.4f} {rt['n']:>8} {ro['accuracy']:>8.4f} {ro['n']:>7} {ok:>9}")
imp=sorted(zip(feats,L.feature_importances_),key=lambda x:-x[1])
print("Top 20:",[n for n,v in imp[:20]])
print("Daily feats in top 40:",[n for n,v in imp[:40] if n.startswith("DLY")])
np.savez("models/probs_15m_v2.npz",pva=pv,pte=pt,poo=po,yva=yva,yte=yte,yoo=yoo)
print("EXP15M_V2 DONE")
