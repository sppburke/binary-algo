"""1-min IMPROVEMENT (apply 2-min lessons): regime-specialist direction + reversion filter + 2m cross-horizon
confirmation. Caches all probs to models/probs_min1_v15.npz for the selection sweep.

Transfers from the 2-min win:
  (1) direction SPECIALIST trained only on compression-release bars (62-feat set, richer than the 53-feat 1m prod)
  (2) all-bars direction ensemble (for blending)
  (3) the 2-min model's p_2m at each bar = causal cross-horizon confirmation (uses only data <= t; bet settles t+60)
  (4) magnitude model (kept — it HELPS at 60s, unlike 120s)
"""
import os, time, numpy as np, pandas as pd
import lightgbm as lgb, xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
ROOT="/home/sean/git/binary-algo"; TICK=f"{ROOT}/features_tick"; MODELS=f"{ROOT}/models"
from min2_v1 import feats           # identical 62-feature builder (features are horizon-independent)
HS=60; TRSTRIDE_ALL=5; TRSTRIDE_SPEC=2

def prep60(b):
    X=feats(b); mid=b["mid"]
    ret=(mid.shift(-HS)/mid-1).values
    valid=np.isfinite(ret)&(ret!=0)
    ts=b.index.values.astype("datetime64[s]").astype("int64")
    return X,(ret>0).astype(int),np.abs(ret),valid,ts,b.index

def mk_lgb(n=4000): return lgb.LGBMClassifier(objective="binary",metric="auc",learning_rate=0.02,num_leaves=350,
    min_child_samples=200,subsample=0.8,subsample_freq=1,colsample_bytree=0.6,reg_lambda=8,n_estimators=n,n_jobs=20,verbosity=-1)
def mk_spec(n=6000): return lgb.LGBMClassifier(objective="binary",metric="auc",learning_rate=0.01,num_leaves=512,
    min_child_samples=300,subsample=0.7,subsample_freq=1,colsample_bytree=0.5,reg_lambda=20,reg_alpha=2,n_estimators=n,n_jobs=20,verbosity=-1)

# 2-min model loader (for cross-horizon confirmation)
def load_2m():
    import json
    p=json.load(open(f"{MODELS}/min2_EURUSD_strategy.json"))
    L=lgb.Booster(model_file=f"{MODELS}/min2_EURUSD_dir_v1_lgb.txt")
    G=xgb.XGBClassifier(); G.load_model(f"{MODELS}/min2_EURUSD_dir_v1_xgb.json")
    C=CatBoostClassifier(); C.load_model(f"{MODELS}/min2_EURUSD_dir_v1_cat.cbm")
    S=lgb.Booster(model_file=f"{MODELS}/min2_EURUSD_dir_spec_lgb.txt")
    return p,L,G,C,S
def p2m(p2,L2,G2,C2,S2,X):
    Xo=X[p2["feature_names"]]
    pa=(L2.predict(Xo.values)+G2.predict_proba(Xo)[:,1]+C2.predict_proba(Xo.fillna(-999))[:,1])/3.0
    ps=S2.predict(Xo.values); return (1-p2["w_spec"])*pa+p2["w_spec"]*ps

def main():
    t0=time.time()
    btr=pd.read_parquet(f"{TICK}/train_1s.parquet"); bva=pd.read_parquet(f"{TICK}/val_1s.parquet")
    bte=pd.read_parquet(f"{TICK}/test_1s.parquet"); boo=pd.read_parquet(f"{TICK}/oos_1s.parquet")
    Xtr,ytr,mtr,vtr,_,_=prep60(btr); Xva,yva,mva,vva,tsva,iva=prep60(bva)
    Xte,yte,mte,vte,tste,ite=prep60(bte); Xoo,yoo,moo,voo,tsoo,ioo=prep60(boo)
    feat_names=list(Xtr.columns)
    # compression-release regime thresholds from TRAIN (q67 / p70, same as 2m)
    bbwtr=Xtr["bbw1800"].values.astype(float); reltr=Xtr["rel_ratio"].values.astype(float)
    bq=float(np.nanpercentile(bbwtr[vtr],67)); rq=float(np.nanpercentile(reltr[vtr&(bbwtr<=bq)],70))
    print(f"[v15] regime bbw1800<={bq:.3e} rel>={rq:.3f} {time.time()-t0:.0f}s",flush=True)
    iall=np.where(vtr)[0][::TRSTRIDE_ALL]
    XA=Xtr.iloc[iall]; yA=ytr[iall]
    L=mk_lgb(); L.fit(XA,yA,eval_set=[(Xva.iloc[np.where(vva)[0]],yva[np.where(vva)[0]])],eval_metric="auc",
        callbacks=[lgb.early_stopping(120),lgb.log_evaluation(0)]); print(f"[v15] allbars-lgb {time.time()-t0:.0f}s",flush=True)
    G=xgb.XGBClassifier(n_estimators=2500,learning_rate=0.02,max_depth=9,subsample=0.8,colsample_bytree=0.6,
        reg_lambda=8,tree_method="hist",n_jobs=20,eval_metric="auc",early_stopping_rounds=120)
    G.fit(XA,yA,eval_set=[(Xva.iloc[np.where(vva)[0]],yva[np.where(vva)[0]])],verbose=False); print(f"[v15] xgb {time.time()-t0:.0f}s",flush=True)
    C=CatBoostClassifier(iterations=2500,learning_rate=0.02,depth=9,l2_leaf_reg=8,eval_metric="AUC",
        thread_count=20,verbose=False,early_stopping_rounds=120)
    C.fit(XA.fillna(-999),yA,eval_set=(Xva.iloc[np.where(vva)[0]].fillna(-999),yva[np.where(vva)[0]])); print(f"[v15] cat {time.time()-t0:.0f}s",flush=True)
    magthr=np.nanpercentile(mtr[iall],67); M=mk_lgb(2500)
    M.fit(XA,(mtr[iall]>=magthr).astype(int),eval_set=[(Xva.iloc[np.where(vva)[0]],(mva[np.where(vva)[0]]>=magthr).astype(int))],
        eval_metric="auc",callbacks=[lgb.early_stopping(100),lgb.log_evaluation(0)]); print(f"[v15] mag {time.time()-t0:.0f}s",flush=True)
    # compression-release specialist on 60s label
    def rmask(X,valid):
        b=X["bbw1800"].values.astype(float); r=X["rel_ratio"].values.astype(float)
        return valid&(b<=bq)&(r>=rq)
    isp=np.where(rmask(Xtr,vtr))[0][::TRSTRIDE_SPEC]; ivsp=np.where(rmask(Xva,vva))[0]
    S=mk_spec(); S.fit(Xtr.iloc[isp],ytr[isp],eval_set=[(Xva.iloc[ivsp],yva[ivsp])],eval_metric="auc",
        callbacks=[lgb.early_stopping(200),lgb.log_evaluation(0)]); print(f"[v15] specialist best_iter={S.best_iteration_} {time.time()-t0:.0f}s",flush=True)
    p2,L2,G2,C2,S2=load_2m()
    def allp(X): return (L.predict_proba(X)[:,1]+G.predict_proba(X)[:,1]+C.predict_proba(X.fillna(-999))[:,1])/3.0
    out={}
    for nm,(Xf,yf,mf,vf,tsf,idxf) in [("va",(Xva,yva,mva,vva,tsva,iva)),("te",(Xte,yte,mte,vte,tste,ite)),("oo",(Xoo,yoo,moo,voo,tsoo,ioo))]:
        iv=np.where(vf)[0]; Xs=Xf.iloc[iv]
        pa=allp(Xs).astype("float32"); ps=S.predict_proba(Xs)[:,1].astype("float32")
        out[f"{nm}_pall"]=pa; out[f"{nm}_pspec"]=ps; out[f"{nm}_p2m"]=p2m(p2,L2,G2,C2,S2,Xs).astype("float32")
        out[f"{nm}_mag"]=M.predict_proba(Xs)[:,1].astype("float32")
        out[f"{nm}_y"]=yf[iv].astype("int8"); out[f"{nm}_ts"]=tsf[iv].astype("int64")
        out[f"{nm}_bbw1800"]=Xf["bbw1800"].values[iv].astype("float32")
        out[f"{nm}_rel"]=Xf["rel_ratio"].values[iv].astype("float32")
        out[f"{nm}_ret60"]=Xf["ret60"].values[iv].astype("float32")
        out[f"{nm}_ret120"]=Xf["ret120"].values[iv].astype("float32")
        out[f"{nm}_ret300"]=Xf["ret300"].values[iv].astype("float32")
        out[f"{nm}_sess"]=((idxf[iv].hour>=13)&(idxf[iv].hour<21)).astype("int8")
        out[f"{nm}_month"]=idxf[iv].to_period("M").astype(str).values.astype("U7")
        reg=(Xf["bbw1800"].values[iv]<=bq)&(Xf["rel_ratio"].values[iv]>=rq)
        print(f"[v15 {nm}] AUC_all={roc_auc_score(yf[iv],pa):.4f} AUC_spec_inreg={roc_auc_score(yf[iv][reg],ps[reg]):.4f} "
              f"AUC_2m_inreg={roc_auc_score(yf[iv][reg],out[f'{nm}_p2m'][reg]):.4f} regn={int(reg.sum())}",flush=True)
    np.savez_compressed(f"{MODELS}/probs_min1_v15.npz",feat_names=np.array(feat_names),bq=bq,rq=rq,magthr=magthr,**out)
    print(f"[v15 done] saved probs_min1_v15.npz {time.time()-t0:.0f}s",flush=True)

if __name__=="__main__": main()
