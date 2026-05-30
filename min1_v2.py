"""1-MINUTE v2 — microprice-denoised label + magnitude-filtered training + ensemble + persistence feats.

Improvements over v1 (which hit OOS 0.725@0.2%, 0.731@0.1% non-overlapping):
  - TRAIN target = microprice direction sign(micro(t+60)-micro(t)) — removes bid-ask bounce noise
    (Stoikov microprice is a cleaner short-horizon fair value). EVAL target = real mid direction
    (the actual binary settlement), so honesty is preserved.
  - Magnitude-filtered training: train only on bars whose |micro move| >= VAL/TRAIN median, so the
    model learns to separate clear up/down cases (the ones a selective book actually bets).
  - Ensemble LGBM+XGB+CatBoost.
  - Added order-flow PERSISTENCE features (signed-imbalance run-length, sign-autocorr, same-side frac).
  - Honest non-overlapping (60s-gap) selective frontier + OOS split into 2026-H1/H2 + bootstrap CI.
"""
import sys, time, numpy as np, pandas as pd
import lightgbm as lgb, xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
TICK="/media/sean/CORSAIR/binary-algo/features_tick"
HS=60; TRSTRIDE=5

def feats(b):
    mid=b["mid"]; imb=b["imb"].fillna(0); micro=b["micro"]; r1=mid.pct_change()
    X=pd.DataFrame(index=b.index)
    X["imb"]=imb
    for w in (3,5,10,20,40,80): X[f"imb_ema{w}"]=imb.ewm(span=w).mean()
    X["imb_acc"]=imb.ewm(span=5).mean()-imb.ewm(span=40).mean(); X["imb_chg"]=imb.diff(3)
    # --- persistence of signed flow ---
    sgn=np.sign(imb)
    X["imb_sgn_ac30"]=(sgn*sgn.shift(1)).rolling(30).mean()         # vectorized sign-autocorr
    X["imb_sameside30"]=(sgn==sgn.shift(1)).rolling(30).mean()
    rl=sgn.groupby((sgn!=sgn.shift()).cumsum()).cumcount()+1
    X["imb_runlen"]=(rl*sgn).clip(-50,50)
    md=(micro-mid)/mid; X["micro_dev"]=md
    for w in (5,15,30,60): X[f"micro_dev_ema{w}"]=md.ewm(span=w).mean()
    X["micro_mom15"]=micro/micro.shift(15)-1; X["micro_mom60"]=micro/micro.shift(60)-1
    X["spread"]=b["spread"]; X["spread_ema30"]=b["spread"].ewm(span=30).mean()
    X["nt"]=b["nt"]; X["nt_ema30"]=b["nt"].ewm(span=30).mean()
    X["tsz"]=b["tsz"]; X["tsz_ema30"]=b["tsz"].ewm(span=30).mean()
    for w in (5,15,30,60,120,300,600,1800,3600): X[f"ret{w}"]=mid.pct_change(w)
    for w in (30,60,300,900,1800): X[f"rv{w}"]=r1.rolling(w).std()
    for w in (60,300,900,1800,3600): X[f"emadist{w}"]=mid/mid.ewm(span=w).mean()-1
    for w in (60,300,900):
        sd=r1.rolling(w).std()*np.sqrt(w); X[f"stretch{w}"]=(mid/mid.ewm(span=w).mean()-1)/(sd+1e-9)
    for w in (300,900,1800):
        hi=mid.rolling(w).max(); lo=mid.rolling(w).min(); X[f"rangepos{w}"]=(mid-lo)/(hi-lo+1e-12)
    for w in (300,900,1800): X[f"bbw{w}"]=(r1.rolling(w).std()*np.sqrt(w))
    hh=b.index.hour; X["hsin"]=np.sin(2*np.pi*hh/24); X["hcos"]=np.cos(2*np.pi*hh/24)
    X["dow"]=b.index.dayofweek.astype("float32")
    return X.replace([np.inf,-np.inf],np.nan).astype("float32")

def main():
    t0=time.time()
    raw={sp:pd.read_parquet(f"{TICK}/{sp}_1s.parquet") for sp in ["train","val","test","oos"]}
    X={sp:feats(raw[sp]) for sp in raw}
    MICRO={sp:raw[sp]["micro"] for sp in raw}; MID={sp:raw[sp]["mid"] for sp in raw}
    TS={sp:raw[sp].index.values.astype("datetime64[s]").astype("int64") for sp in raw}
    print(f"features {time.time()-t0:.0f}s | nfeat={X['train'].shape[1]}",flush=True)
    # micro-move (train target & magnitude) and mid-move (eval target)
    def micmove(sp): return MICRO[sp].shift(-HS)/MICRO[sp]-1
    def midmove(sp): return MID[sp].shift(-HS)/MID[sp]-1
    ytr_raw=micmove("train"); thr_mag=np.nanquantile(np.abs(ytr_raw.values),0.40)  # drop noisiest 40%
    def prep_train():
        mm=micmove("train"); m=mm.notna().values & (np.abs(mm.values)>=thr_mag) & (mm.values!=0)
        idx=np.where(m)[0][::TRSTRIDE]
        return X["train"].iloc[idx], (mm.values[idx]>0).astype(int)
    def prep_eval(sp):
        mm=midmove(sp); m=mm.notna().values & (mm.values!=0)
        idx=np.where(m)[0]
        return X[sp].iloc[idx], (mm.values[idx]>0).astype(int), idx
    Xtr,ytr=prep_train()
    Xva,yva,iva=prep_eval("val"); Xte,yte,ite=prep_eval("test"); Xoo,yoo,ioo=prep_eval("oos")
    print(f"prep {time.time()-t0:.0f}s | train={len(ytr):,}(magfilt) test={len(yte):,} oos={len(yoo):,} thr_mag={thr_mag:.2e}",flush=True)
    P={}
    L=lgb.LGBMClassifier(objective="binary",metric="auc",learning_rate=0.02,num_leaves=350,min_child_samples=200,
        subsample=0.8,subsample_freq=1,colsample_bytree=0.6,reg_lambda=8,n_estimators=4000,n_jobs=20,verbosity=-1)
    L.fit(Xtr,ytr,eval_set=[(Xva,yva)],eval_metric="auc",callbacks=[lgb.early_stopping(120),lgb.log_evaluation(0)])
    P["lgb"]=(L.predict_proba(Xva)[:,1],L.predict_proba(Xte)[:,1],L.predict_proba(Xoo)[:,1])
    G=xgb.XGBClassifier(n_estimators=2500,learning_rate=0.02,max_depth=9,subsample=0.8,colsample_bytree=0.6,
        reg_lambda=8,tree_method="hist",n_jobs=20,eval_metric="auc",early_stopping_rounds=120)
    G.fit(Xtr,ytr,eval_set=[(Xva,yva)],verbose=False)
    P["xgb"]=(G.predict_proba(Xva)[:,1],G.predict_proba(Xte)[:,1],G.predict_proba(Xoo)[:,1])
    C=CatBoostClassifier(iterations=2500,learning_rate=0.02,depth=9,l2_leaf_reg=8,eval_metric="AUC",
        thread_count=20,verbose=False,early_stopping_rounds=120)
    C.fit(Xtr.fillna(-999),ytr,eval_set=(Xva.fillna(-999),yva))
    P["cat"]=(C.predict_proba(Xva.fillna(-999))[:,1],C.predict_proba(Xte.fillna(-999))[:,1],C.predict_proba(Xoo.fillna(-999))[:,1])
    pv=np.mean([P[n][0] for n in P],0); pt=np.mean([P[n][1] for n in P],0); po=np.mean([P[n][2] for n in P],0)
    for n in P: print(f"  {n} AUC oos={roc_auc_score(yoo,P[n][2]):.4f}",flush=True)
    print(f"BLEND AUC val={roc_auc_score(yva,pv):.4f} test={roc_auc_score(yte,pt):.4f} oos={roc_auc_score(yoo,po):.4f}",flush=True)

    def nonov(ts,conf,corr,thr,gap=HS):
        sel=np.where(conf>=thr)[0]
        if len(sel)==0: return np.nan,0,None
        order=sel[np.argsort(-conf[sel])]; used=np.empty(0,"int64"); take=[]
        for i in order:
            if used.size and np.any(np.abs(used-ts[i])<gap): continue
            take.append(i); used=np.append(used,ts[i])
        take=np.array(take); return corr[take].mean(), len(take), corr[take]
    ct=(pt>0.5).astype(int)==yte; co=(po>0.5).astype(int)==yoo
    tst=TS["test"][ite]; tso=TS["oos"][ioo]; cft=np.abs(pt-0.5); cfo=np.abs(po-0.5); cfv=np.abs(pv-0.5)
    oo_mid=len(ioo)//2; tso_h1=tso[:oo_mid]; tso_h2=tso[oo_mid:]
    rng=np.random.default_rng(1)
    print(f"\n{'cov':>7} {'TESTacc':>8} {'TESTn':>7} {'OOSacc':>8} {'OOSn':>6} {'OOS95CI':>15} {'H1':>6} {'H2':>6} {'75?':>4}  (non-overlap)")
    for cov in (0.05,0.02,0.01,0.005,0.002,0.001):
        thr=np.quantile(cfv,1-cov)
        at,nt,_=nonov(tst,cft,ct,thr); ao,no,so=nonov(tso,cfo,co,thr)
        a1,n1,_=nonov(tso_h1,cfo[:oo_mid],co[:oo_mid],thr); a2,n2,_=nonov(tso_h2,cfo[oo_mid:],co[oo_mid:],thr)
        ci=""
        if so is not None and no>=20:
            b=np.array([rng.choice(so,no,replace=True).mean() for _ in range(4000)]); ci=f"[{np.percentile(b,2.5):.2f},{np.percentile(b,97.5):.2f}]"
        ok="**" if (not np.isnan(ao) and ao>=0.75 and no>=50 and not np.isnan(at) and at>=0.72) else ""
        print(f"{cov:>7.2%} {at:>8.3f} {nt:>7} {ao:>8.3f} {no:>6} {ci:>15} {a1:>6.3f} {a2:>6.3f} {ok:>4}",flush=True)
    np.savez("models/probs_min1_v2.npz",pva=pv,pte=pt,poo=po,yva=yva,yte=yte,yoo=yoo)
    print("MIN1-V2 DONE")

if __name__=="__main__":
    main()
