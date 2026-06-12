"""2-MINUTE EURUSD binary research — phase A: train direction ensemble + magnitude at 120s, cache probs.

Mirrors min1_production but HS=120, GAP=120. Saves prob arrays + labels + regime cols + ts to
models/probs_min2_v1.npz so selection logic can be iterated without retraining.
"""
import sys, os, json, time, numpy as np, pandas as pd
import lightgbm as lgb, xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score

ROOT="/home/sean/git/binary-algo"; TICK=f"{ROOT}/features_tick"; MODELS=f"{ROOT}/models"
HS, GAP = 120, 120
TRSTRIDE = 5
VASTRIDE = 1

def feats(b):
    mid=b["mid"]; imb=b["imb"].fillna(0); micro=b["micro"]; r1=mid.pct_change()
    X=pd.DataFrame(index=b.index)
    X["imb"]=imb
    for w in (3,5,10,20,40,80): X[f"imb_ema{w}"]=imb.ewm(span=w).mean()
    X["imb_acc"]=imb.ewm(span=5).mean()-imb.ewm(span=40).mean(); X["imb_chg"]=imb.diff(3)
    sgn=np.sign(imb)
    X["imb_sgn_ac30"]=(sgn*sgn.shift(1)).rolling(30).mean()
    X["imb_sameside30"]=(sgn==sgn.shift(1)).rolling(30).mean()
    rl=sgn.groupby((sgn!=sgn.shift()).cumsum()).cumcount()+1; X["imb_runlen"]=(rl*sgn).clip(-50,50)
    md=(micro-mid)/mid; X["micro_dev"]=md
    for w in (5,15,30,60): X[f"micro_dev_ema{w}"]=md.ewm(span=w).mean()
    X["micro_mom15"]=micro/micro.shift(15)-1; X["micro_mom60"]=micro/micro.shift(60)-1
    X["spread"]=b["spread"]; X["spread_ema30"]=b["spread"].ewm(span=30).mean()
    X["nt"]=b["nt"]; X["nt_ema30"]=b["nt"].ewm(span=30).mean()
    X["tsz"]=b["tsz"]; X["tsz_ema30"]=b["tsz"].ewm(span=30).mean()
    # longer-horizon returns/vol (more relevant at 120s)
    for w in (5,15,30,60,120,300,600,1800,3600): X[f"ret{w}"]=mid.pct_change(w)
    for w in (30,60,300,900,1800,3600): X[f"rv{w}"]=r1.rolling(w).std()
    for w in (60,300,900,1800,3600): X[f"emadist{w}"]=mid/mid.ewm(span=w).mean()-1
    for w in (120,300,900,1800):
        sd=r1.rolling(w).std()*np.sqrt(w); X[f"stretch{w}"]=(mid/mid.ewm(span=w).mean()-1)/(sd+1e-9)
    for w in (300,900,1800,3600):
        hi=mid.rolling(w).max(); lo=mid.rolling(w).min(); X[f"rangepos{w}"]=(mid-lo)/(hi-lo+1e-12)
    for w in (300,900,1800,3600): X[f"bbw{w}"]=(r1.rolling(w).std()*np.sqrt(w))
    X["rel_ratio"]=X["bbw300"]/(X["bbw1800"]+1e-12)
    X["rel_ratio2"]=X["bbw900"]/(X["bbw3600"]+1e-12)
    hh=b.index.hour; X["hsin"]=np.sin(2*np.pi*hh/24); X["hcos"]=np.cos(2*np.pi*hh/24)
    X["dow"]=b.index.dayofweek.astype("float32")
    return X.replace([np.inf,-np.inf],np.nan).astype("float32")

def prep(b):
    X=feats(b); mid=b["mid"]
    ret=(mid.shift(-HS)/mid-1).values
    valid=np.isfinite(ret)&(ret!=0)
    ts=b.index.values.astype("datetime64[s]").astype("int64")
    return X,(ret>0).astype(int),np.abs(ret),valid,ts,b.index

def mk_lgb(n=4000): return lgb.LGBMClassifier(objective="binary",metric="auc",learning_rate=0.02,num_leaves=350,
    min_child_samples=200,subsample=0.8,subsample_freq=1,colsample_bytree=0.6,reg_lambda=8,n_estimators=n,n_jobs=20,verbosity=-1)

def main():
    t0=time.time(); os.makedirs(MODELS,exist_ok=True)
    print("[load] reading parquets",flush=True)
    btr=pd.read_parquet(f"{TICK}/train_1s.parquet"); bva=pd.read_parquet(f"{TICK}/val_1s.parquet")
    bte=pd.read_parquet(f"{TICK}/test_1s.parquet"); boo=pd.read_parquet(f"{TICK}/oos_1s.parquet")
    Xtr_,ytr_,mtr_,vtr,_,_=prep(btr); Xva_,yva_,mva_,vva,tsva,idxva=prep(bva)
    Xte_,yte_,mte_,vte,tste,idxte=prep(bte); Xoo_,yoo_,moo_,voo,tsoo,idxoo=prep(boo)
    feat_names=list(Xtr_.columns)
    itr=np.where(vtr)[0][::TRSTRIDE]
    Xtr=Xtr_.iloc[itr]; ytr=ytr_[itr]
    print(f"[prep] {time.time()-t0:.0f}s; dir-train n={len(ytr):,}; nfeat={len(feat_names)}",flush=True)
    L=mk_lgb(); L.fit(Xtr,ytr,eval_set=[(Xva_.iloc[np.where(vva)[0]],yva_[np.where(vva)[0]])],
        eval_metric="auc",callbacks=[lgb.early_stopping(120),lgb.log_evaluation(0)])
    print(f"[lgb] {time.time()-t0:.0f}s best_iter={L.best_iteration_}",flush=True)
    G=xgb.XGBClassifier(n_estimators=2500,learning_rate=0.02,max_depth=9,subsample=0.8,colsample_bytree=0.6,
        reg_lambda=8,tree_method="hist",n_jobs=20,eval_metric="auc",early_stopping_rounds=120)
    G.fit(Xtr,ytr,eval_set=[(Xva_.iloc[np.where(vva)[0]],yva_[np.where(vva)[0]])],verbose=False)
    print(f"[xgb] {time.time()-t0:.0f}s",flush=True)
    C=CatBoostClassifier(iterations=2500,learning_rate=0.02,depth=9,l2_leaf_reg=8,eval_metric="AUC",
        thread_count=20,verbose=False,early_stopping_rounds=120)
    C.fit(Xtr.fillna(-999),ytr,eval_set=(Xva_.iloc[np.where(vva)[0]].fillna(-999),yva_[np.where(vva)[0]]))
    print(f"[cat] {time.time()-t0:.0f}s",flush=True)
    magthr=np.nanpercentile(mtr_[itr],67); ymag=(mtr_[itr]>=magthr).astype(int)
    M=mk_lgb(2500); M.fit(Xtr,ymag,eval_set=[(Xva_.iloc[np.where(vva)[0]],(mva_[np.where(vva)[0]]>=magthr).astype(int))],
        eval_metric="auc",callbacks=[lgb.early_stopping(100),lgb.log_evaluation(0)])
    print(f"[mag] {time.time()-t0:.0f}s",flush=True)
    def prob(Xf): return (L.predict_proba(Xf)[:,1]+G.predict_proba(Xf)[:,1]+C.predict_proba(Xf.fillna(-999))[:,1])/3.0
    def magp(Xf): return M.predict_proba(Xf)[:,1]
    out={}
    for nm,(Xf,yf,mf,vf,tsf,idxf) in [("va",(Xva_,yva_,mva_,vva,tsva,idxva)),
                                       ("te",(Xte_,yte_,mte_,vte,tste,idxte)),
                                       ("oo",(Xoo_,yoo_,moo_,voo,tsoo,idxoo))]:
        iv=np.where(vf)[0]
        out[f"{nm}_p"]=prob(Xf.iloc[iv]).astype("float32")
        out[f"{nm}_mag"]=magp(Xf.iloc[iv]).astype("float32")
        out[f"{nm}_y"]=yf[iv].astype("int8")
        out[f"{nm}_ts"]=tsf[iv].astype("int64")
        out[f"{nm}_bbw1800"]=Xf["bbw1800"].values[iv].astype("float32")
        out[f"{nm}_bbw3600"]=Xf["bbw3600"].values[iv].astype("float32")
        out[f"{nm}_rel"]=Xf["rel_ratio"].values[iv].astype("float32")
        out[f"{nm}_rel2"]=Xf["rel_ratio2"].values[iv].astype("float32")
        out[f"{nm}_sess"]=((idxf[iv].hour>=13)&(idxf[iv].hour<21)).astype("int8")  # NY
        out[f"{nm}_month"]=idxf[iv].to_period("M").astype(str).values.astype("U7")
        print(f"[probs {nm}] n={len(iv):,} auc={roc_auc_score(yf[iv],out[f'{nm}_p']):.4f}",flush=True)
    np.savez_compressed(f"{MODELS}/probs_min2_v1.npz",feat_names=np.array(feat_names),magthr=magthr,**out)
    print(f"[done] saved probs_min2_v1.npz {time.time()-t0:.0f}s",flush=True)

if __name__=="__main__": main()
