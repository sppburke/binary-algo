"""2-min v4: tuned LGBM-ONLY direction specialist on compression-release bars (LGBM had highest in-regime AUC).

Saves LGBM-only specialist probs to models/probs_min2_v4.npz for combining with v1 (all-bars). Bigger leaves,
lower lr, more data (stride 2) to squeeze the small in-regime direction signal.
"""
import os, time, numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
ROOT="/media/sean/CORSAIR/binary-algo"; TICK=f"{ROOT}/features_tick"; MODELS=f"{ROOT}/models"
from min2_v1 import feats, prep
HS=120; TRSTRIDE=2
def main():
    t0=time.time()
    btr=pd.read_parquet(f"{TICK}/train_1s.parquet"); bva=pd.read_parquet(f"{TICK}/val_1s.parquet")
    bte=pd.read_parquet(f"{TICK}/test_1s.parquet"); boo=pd.read_parquet(f"{TICK}/oos_1s.parquet")
    Xtr,ytr,mtr,vtr,_,_=prep(btr); Xva,yva,mva,vva,tsva,iva=prep(bva)
    Xte,yte,mte,vte,tste,ite=prep(bte); Xoo,yoo,moo,voo,tsoo,ioo=prep(boo)
    bbwtr=Xtr["bbw1800"].values.astype(float); reltr=Xtr["rel_ratio"].values.astype(float)
    bq=np.nanpercentile(bbwtr[vtr],67); rq=np.nanpercentile(reltr[vtr&(bbwtr<=bq)],70)
    def rmask(X,valid):
        b=X["bbw1800"].values.astype(float); r=X["rel_ratio"].values.astype(float)
        return valid&(b<=bq)&(r>=rq)
    itr=np.where(rmask(Xtr,vtr))[0][::TRSTRIDE]; ivr=np.where(rmask(Xva,vva))[0]
    XtrR=Xtr.iloc[itr]; ytrR=ytr[itr]; XvaR=Xva.iloc[ivr]; yvaR=yva[ivr]
    print(f"[v4] regime bbw<={bq:.2e} rel>={rq:.3f} train n={len(ytrR):,} {time.time()-t0:.0f}s",flush=True)
    L=lgb.LGBMClassifier(objective="binary",metric="auc",learning_rate=0.01,num_leaves=512,min_child_samples=300,
        subsample=0.7,subsample_freq=1,colsample_bytree=0.5,reg_lambda=20,reg_alpha=2,n_estimators=6000,n_jobs=20,verbosity=-1)
    L.fit(XtrR,ytrR,eval_set=[(XvaR,yvaR)],eval_metric="auc",callbacks=[lgb.early_stopping(200),lgb.log_evaluation(0)])
    print(f"[v4] lgb best_iter={L.best_iteration_} {time.time()-t0:.0f}s",flush=True)
    out={}
    for nm,(Xf,yf,vf,tsf) in [("va",(Xva,yva,vva,tsva)),("te",(Xte,yte,vte,tste)),("oo",(Xoo,yoo,voo,tsoo))]:
        iv=np.where(vf)[0]; p=L.predict_proba(Xf.iloc[iv])[:,1].astype("float32")
        out[f"{nm}_p"]=p
        reg=(Xf["bbw1800"].values[iv]<=bq)&(Xf["rel_ratio"].values[iv]>=rq)
        print(f"[v4 {nm}] AUC_inregime={roc_auc_score(yf[iv][reg],p[reg]):.4f} regn={int(reg.sum())}",flush=True)
    np.savez_compressed(f"{MODELS}/probs_min2_v4.npz",bq=bq,rq=rq,**out)
    print(f"[v4 done] {time.time()-t0:.0f}s",flush=True)
if __name__=="__main__": main()
