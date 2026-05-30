"""2-min v3: REGIME-SPECIALIST direction model (trained only on compression-release bars) + trend alignment.

Hypothesis: the all-bars direction model is AUC 0.51 because most bars are noise. Training a direction model
ONLY on compression-release bars (where a move is developing) should lift in-regime direction AUC, and adding
higher-TF trend alignment (continuation vs reversion) should sharpen it further. Caches specialist probs +
trend sign + regime cols to models/probs_min2_v3.npz for the selection sweep.
"""
import os, time, numpy as np, pandas as pd
import lightgbm as lgb, xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
ROOT="/media/sean/CORSAIR/binary-algo"; TICK=f"{ROOT}/features_tick"; MODELS=f"{ROOT}/models"
HS=120; TRSTRIDE=3
from min2_v1 import feats, prep   # reuse 62-feature builder + 120s prep

# training regime = broad compression-release so specialist sees the betting distribution
def regime_mask(X):
    bbw=X["bbw1800"].values.astype(float); rel=X["rel_ratio"].values.astype(float)
    # thresholds fit on TRAIN below; here placeholder replaced at runtime
    return bbw, rel

def mk_lgb(n=4000): return lgb.LGBMClassifier(objective="binary",metric="auc",learning_rate=0.02,num_leaves=255,
    min_child_samples=120,subsample=0.8,subsample_freq=1,colsample_bytree=0.6,reg_lambda=8,n_estimators=n,n_jobs=20,verbosity=-1)

def main():
    t0=time.time()
    btr=pd.read_parquet(f"{TICK}/train_1s.parquet"); bva=pd.read_parquet(f"{TICK}/val_1s.parquet")
    bte=pd.read_parquet(f"{TICK}/test_1s.parquet"); boo=pd.read_parquet(f"{TICK}/oos_1s.parquet")
    Xtr,ytr,mtr,vtr,_,itr_idx=prep(btr); Xva,yva,mva,vva,tsva,iva_idx=prep(bva)
    Xte,yte,mte,vte,tste,ite_idx=prep(bte); Xoo,yoo,moo,voo,tsoo,ioo_idx=prep(boo)
    feat_names=list(Xtr.columns)
    # regime thresholds from TRAIN (broad: bbw<=q67, rel>=p70)
    bbwtr=Xtr["bbw1800"].values.astype(float); reltr=Xtr["rel_ratio"].values.astype(float)
    bq=np.nanpercentile(bbwtr[vtr],67); rq=np.nanpercentile(reltr[vtr&(bbwtr<=bq)],70)
    print(f"[regime] bbw1800<={bq:.3e} rel>={rq:.3f}  {time.time()-t0:.0f}s",flush=True)
    def rmask(X,valid):
        b=X["bbw1800"].values.astype(float); r=X["rel_ratio"].values.astype(float)
        return valid&(b<=bq)&(r>=rq)
    mtr_reg=rmask(Xtr,vtr); mva_reg=rmask(Xva,vva)
    itr=np.where(mtr_reg)[0][::TRSTRIDE]; iva=np.where(mva_reg)[0]
    XtrR=Xtr.iloc[itr]; ytrR=ytr[itr]; XvaR=Xva.iloc[iva]; yvaR=yva[iva]
    print(f"[specialist-train] n={len(ytrR):,} val-regime n={len(yvaR):,} base-rate={ytrR.mean():.3f}",flush=True)
    L=mk_lgb(); L.fit(XtrR,ytrR,eval_set=[(XvaR,yvaR)],eval_metric="auc",callbacks=[lgb.early_stopping(150),lgb.log_evaluation(0)])
    print(f"[lgb] best_iter={L.best_iteration_} {time.time()-t0:.0f}s",flush=True)
    G=xgb.XGBClassifier(n_estimators=2500,learning_rate=0.02,max_depth=8,subsample=0.8,colsample_bytree=0.6,
        reg_lambda=8,tree_method="hist",n_jobs=20,eval_metric="auc",early_stopping_rounds=150)
    G.fit(XtrR,ytrR,eval_set=[(XvaR,yvaR)],verbose=False); print(f"[xgb] {time.time()-t0:.0f}s",flush=True)
    C=CatBoostClassifier(iterations=2500,learning_rate=0.02,depth=8,l2_leaf_reg=8,eval_metric="AUC",
        thread_count=20,verbose=False,early_stopping_rounds=150)
    C.fit(XtrR.fillna(-999),ytrR,eval_set=(XvaR.fillna(-999),yvaR)); print(f"[cat] {time.time()-t0:.0f}s",flush=True)
    def prob(Xf): return (L.predict_proba(Xf)[:,1]+G.predict_proba(Xf)[:,1]+C.predict_proba(Xf.fillna(-999))[:,1])/3.0
    out={}
    for nm,(Xf,yf,vf,tsf,idxf) in [("va",(Xva,yva,vva,tsva,iva_idx)),("te",(Xte,yte,vte,tste,ite_idx)),("oo",(Xoo,yoo,voo,tsoo,ioo_idx))]:
        iv=np.where(vf)[0]
        p=prob(Xf.iloc[iv]).astype("float32")
        out[f"{nm}_p"]=p; out[f"{nm}_y"]=yf[iv].astype("int8"); out[f"{nm}_ts"]=tsf[iv].astype("int64")
        out[f"{nm}_bbw1800"]=Xf["bbw1800"].values[iv].astype("float32")
        out[f"{nm}_rel"]=Xf["rel_ratio"].values[iv].astype("float32")
        out[f"{nm}_bbw3600"]=Xf["bbw3600"].values[iv].astype("float32")
        out[f"{nm}_rel2"]=Xf["rel_ratio2"].values[iv].astype("float32")
        # trend signs (higher-TF) for alignment tests
        out[f"{nm}_emad900"]=Xf["emadist900"].values[iv].astype("float32")
        out[f"{nm}_emad3600"]=Xf["emadist3600"].values[iv].astype("float32")
        out[f"{nm}_ret300"]=Xf["ret300"].values[iv].astype("float32")
        out[f"{nm}_sess"]=((idxf[iv].hour>=13)&(idxf[iv].hour<21)).astype("int8")
        out[f"{nm}_month"]=idxf[iv].to_period("M").astype(str).values.astype("U7")
        # AUC within the betting regime
        reg=(Xf["bbw1800"].values[iv]<=bq)&(Xf["rel_ratio"].values[iv]>=rq)
        a_all=roc_auc_score(yf[iv],p); a_reg=roc_auc_score(yf[iv][reg],p[reg]) if reg.sum()>50 and len(set(yf[iv][reg].tolist()))>1 else float("nan")
        print(f"[probs {nm}] n={len(iv):,} AUC_all={a_all:.4f} AUC_inregime={a_reg:.4f} (regn={int(reg.sum())})",flush=True)
    np.savez_compressed(f"{MODELS}/probs_min2_v3.npz",feat_names=np.array(feat_names),bq=bq,rq=rq,**out)
    print(f"[done] saved probs_min2_v3.npz {time.time()-t0:.0f}s",flush=True)

if __name__=="__main__": main()
