"""15-min IMPROVEMENT (apply 2-min lessons): regime-specialist direction + reversion filter + compression-release.

Baseline (V27 / m15_production): compression(15m_bb_width<=q)xNY, all-bars ensemble → 0.642 combined
(2024 0.691 / 2025 0.590 / 2026 0.592). Transfers tested here:
  (1) direction SPECIALIST trained only on compression x NY bars
  (2) reversion trend filter: bet AGAINST the last 15-min move (15m_ret_1) / 5m_ret_3
  (3) compression-RELEASE (not just low vol): 5m_bb_width/30m_bb_width high (short-TF vol expanding)
Caches probs to models/probs_min15_v2.npz for the selection sweep.
"""
import os, time, json, numpy as np, pandas as pd
import lightgbm as lgb, xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
import harness as H
MODELS="/media/sean/CORSAIR/binary-algo/models"; HOR=15; STRIDE=3
base=list(H.feature_cols("EURUSD")); PAIR="EURUSD"

def load(years, stride=1):
    parts=[]
    for y in years:
        p=f"{H.FEAT_DIR}/{PAIR}_{y}.parquet"
        if not os.path.exists(p): continue
        df=pd.read_parquet(p, columns=base+H.META_COLS); df=df[~df.index.duplicated(keep="last")]
        idx=df.index; c=df["close"].values; n=len(c)
        secs=idx.values.astype("datetime64[s]").astype("int64")
        contig=np.zeros(n,bool)
        if n>HOR: contig[:n-HOR]=(secs[HOR:]-secs[:-HOR])==HOR*60
        fwd=np.full(n,np.nan); fwd[:n-HOR]=c[HOR:]; ret=fwd/c-1.0
        yv=(ret>0).astype(float); valid=contig&np.isfinite(ret)&(ret!=0)
        d=df.loc[valid,base].copy(); d["_y"]=yv[valid]; d["_idx"]=idx[valid]
        parts.append(d.iloc[::stride] if stride>1 else d)
    return pd.concat(parts)

def mk_lgb(n=3000): return lgb.LGBMClassifier(objective="binary",metric="auc",learning_rate=0.02,num_leaves=255,
    min_child_samples=200,subsample=0.8,subsample_freq=1,colsample_bytree=0.5,reg_lambda=10,n_estimators=n,n_jobs=20,verbosity=-1)
def mk_spec(n=5000): return lgb.LGBMClassifier(objective="binary",metric="auc",learning_rate=0.01,num_leaves=300,
    min_child_samples=150,subsample=0.7,subsample_freq=1,colsample_bytree=0.5,reg_lambda=15,reg_alpha=2,n_estimators=n,n_jobs=20,verbosity=-1)

def regime_mask(D, bbw_thr):
    return (D["15m_bb_width"].values.astype(float)<=bbw_thr)&(D["sess_ny"].values.astype(float)>0.5)

def main():
    t0=time.time()
    TR=load(H.SPLITS["train"],STRIDE); VA=load(H.SPLITS["val"]); TE=load(H.SPLITS["test"]); OO=load(H.SPLITS["oos"])
    ytr=TR["_y"].astype(int).values; yva=VA["_y"].astype(int).values
    print(f"[15m] load {time.time()-t0:.0f}s train={len(TR):,} val={len(VA):,} test={len(TE):,} oos={len(OO):,}",flush=True)
    # regime threshold from TRAIN: 15m_bb_width q67 (moderate compression, per 2m lesson q67>q33)
    bq=float(np.nanpercentile(TR["15m_bb_width"].values.astype(float),67))
    print(f"[15m] regime 15m_bb_width<={bq:.3e} & NY {time.time()-t0:.0f}s",flush=True)
    # all-bars ensemble
    L=mk_lgb(); L.fit(TR[base].astype("float32"),ytr,eval_set=[(VA[base].astype("float32"),yva)],eval_metric="auc",
        callbacks=[lgb.early_stopping(150),lgb.log_evaluation(0)]); print(f"[15m] allbars-lgb {time.time()-t0:.0f}s",flush=True)
    G=xgb.XGBClassifier(n_estimators=2000,learning_rate=0.02,max_depth=8,subsample=0.8,colsample_bytree=0.5,
        reg_lambda=10,tree_method="hist",n_jobs=20,eval_metric="auc",early_stopping_rounds=150)
    G.fit(TR[base].astype("float32"),ytr,eval_set=[(VA[base].astype("float32"),yva)],verbose=False); print(f"[15m] xgb {time.time()-t0:.0f}s",flush=True)
    C=CatBoostClassifier(iterations=2000,learning_rate=0.02,depth=8,l2_leaf_reg=10,eval_metric="AUC",
        thread_count=20,verbose=False,early_stopping_rounds=150)
    C.fit(TR[base].fillna(-999),ytr,eval_set=(VA[base].fillna(-999),yva)); print(f"[15m] cat {time.time()-t0:.0f}s",flush=True)
    # specialist: train only on compression x NY bars
    rtr=regime_mask(TR,bq); rva=regime_mask(VA,bq)
    print(f"[15m] specialist train n={int(rtr.sum()):,} base-rate={ytr[rtr].mean():.3f}",flush=True)
    S=mk_spec(); S.fit(TR[base][rtr].astype("float32"),ytr[rtr],eval_set=[(VA[base][rva].astype("float32"),yva[rva])],
        eval_metric="auc",callbacks=[lgb.early_stopping(200),lgb.log_evaluation(0)]); print(f"[15m] specialist best_iter={S.best_iteration_} {time.time()-t0:.0f}s",flush=True)
    def allp(X): return (L.predict_proba(X)[:,1]+G.predict_proba(X)[:,1]+C.predict_proba(X.fillna(-999))[:,1])/3.0
    out={}
    for nm,D in (("va",VA),("te",TE),("oo",OO)):
        X=D[base].astype("float32")
        pa=allp(X).astype("float32"); ps=S.predict_proba(X)[:,1].astype("float32")
        out[f"{nm}_pall"]=pa; out[f"{nm}_pspec"]=ps; out[f"{nm}_y"]=D["_y"].astype("int8").values
        out[f"{nm}_bbw15"]=D["15m_bb_width"].values.astype("float32")
        out[f"{nm}_bbw5"]=D["5m_bb_width"].values.astype("float32")
        out[f"{nm}_bbw30"]=D["30m_bb_width"].values.astype("float32")
        out[f"{nm}_ny"]=D["sess_ny"].values.astype("float32")
        out[f"{nm}_ret15_1"]=D["15m_ret_1"].values.astype("float32")
        out[f"{nm}_ret5_3"]=D["5m_ret_3"].values.astype("float32")
        out[f"{nm}_ret15_3"]=D["15m_ret_3"].values.astype("float32")
        out[f"{nm}_month"]=pd.PeriodIndex(D["_idx"].values,freq="M").astype(str).values.astype("U7")
        reg=regime_mask(D,bq)
        print(f"[15m {nm}] AUC_all={roc_auc_score(D['_y'].astype(int).values,pa):.4f} "
              f"AUC_spec_inreg={roc_auc_score(D['_y'].astype(int).values[reg],ps[reg]):.4f} regn={int(reg.sum())}",flush=True)
    np.savez_compressed(f"{MODELS}/probs_min15_v2.npz",bq=bq,**out)
    print(f"[15m done] saved probs_min15_v2.npz {time.time()-t0:.0f}s",flush=True)

if __name__=="__main__": main()
