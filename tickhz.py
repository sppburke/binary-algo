"""HONEST horizon sweep on tick microstructure — re-verify whether the pre-audit seconds-scale edge (0.71-0.81)
survives bias-corrected methodology (deriv-faithful wc_ret label, chronological non-overlap independence, CI95).

The pre-audit achievability curve (5s 0.71 ... 60s 0.60) used the bar-count horizon bug + greedy de-overlap that the
2026-05 audit showed inflate results. Here: TRUE wall-clock expiry at HS seconds, chronological first-come de-overlap,
bootstrap CI over INDEPENDENT trades. Features computed ONCE (horizon-independent), reused across horizons.

  python tickhz.py 5 15 30 60
"""
import sys, time, numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from min1_production import feats, wc_ret, nonoverlap_chrono, boot, load_split, set_pair
set_pair("EURUSD"); TOL=2; LAG=1

def prep_feats(sp, stride=1):
    b=load_split(sp); X=feats(b)
    mid=b["mid"].values.astype(float); ts=b.index.values.astype("datetime64[s]").astype("int64")
    hour=b.index.hour.values
    if stride>1:
        idx=np.arange(0,len(X),stride); return X.iloc[idx].reset_index(drop=True), mid[idx], ts[idx], hour[idx]
    return X.reset_index(drop=True), mid, ts, hour

def label(ts, mid, HS):
    ret,valid=wc_ret(ts, mid, HS, max(TOL,HS//30+1), LAG); return (ret>0).astype(int), valid

def mk(): return lgb.LGBMClassifier(objective="binary",metric="auc",learning_rate=0.03,num_leaves=300,
    min_child_samples=200,subsample=0.8,subsample_freq=1,colsample_bytree=0.6,reg_lambda=8,n_estimators=2500,n_jobs=20,verbosity=-1)

def sel(p,y,ts,hour,HS,gate,cov_thr_src=None,cov=0.02):
    gap=HS+max(TOL,HS//30+1)
    ny=(hour>=12)&(hour<21); g={"none":np.ones(len(y),bool),"ny":ny}[gate]
    conf=np.abs(p-0.5)
    thr=cov_thr_src if cov_thr_src is not None else float(np.quantile(conf[g],1-cov))
    m=g&(conf>=thr)
    if m.sum()==0: return (0,float("nan"),float("nan"),float("nan"),thr)
    s=nonoverlap_chrono(ts,m,gap)
    if len(s)==0: return (0,float("nan"),float("nan"),float("nan"),thr)
    corr=((p[s]>0.5).astype(int)==y[s]).astype(float); a=corr.mean(); lo,hi=boot(corr); return (len(s),a,lo,hi,thr)

def main(hzs):
    t0=time.time()
    print("[tickhz] computing features once per split (horizon-independent)...",flush=True)
    Xtr,midtr,tstr,htr=prep_feats("train",1)   # FULL res — label needs 1s resolution; subsample only at fit
    Xva,midva,tsva,hva=prep_feats("val",1)
    Xte,midte,tste,hte=prep_feats("test",1)
    Xoo,midoo,tsoo,hoo=prep_feats("oos",1)
    print(f"[tickhz] feats done {time.time()-t0:.0f}s  tr={len(Xtr):,} va={len(Xva):,} te={len(Xte):,} oo={len(Xoo):,}",flush=True)
    for HS in hzs:
        ytr,vtr=label(tstr,midtr,HS); yva,vva=label(tsva,midva,HS)
        yte,vte=label(tste,midte,HS); yoo,voo=label(tsoo,midoo,HS)
        itr=np.where(vtr)[0][::20]   # subsample valid TRAIN rows for fit speed (label already full-res correct)
        L=mk(); L.fit(Xtr.iloc[itr],ytr[itr],eval_set=[(Xva.iloc[np.where(vva)[0]],yva[np.where(vva)[0]])],
            eval_metric="auc",callbacks=[lgb.early_stopping(120),lgb.log_evaluation(0)])
        pva=L.predict_proba(Xva)[:,1]; pte=L.predict_proba(Xte)[:,1]; poo=L.predict_proba(Xoo)[:,1]
        aoo=roc_auc_score(yoo[voo],poo[voo]); ate=roc_auc_score(yte[vte],pte[vte])
        print(f"\n=== HS={HS}s ===  AUC test={ate:.4f} oos={aoo:.4f}  (indep non-overlap, chronological, CI95; breakeven~0.541)",flush=True)
        for gate in ("none","ny"):
            for cov in (0.02,0.01,0.005,0.002,0.001):
                # threshold set on VAL within gate, applied to TEST and OOS (honest)
                _,_,_,_,thr=sel(pva[vva],yva[vva],tsva[vva],hva[vva],HS,gate,None,cov)
                nte,a_te,lo_te,hi_te,_=sel(pte[vte],yte[vte],tste[vte],hte[vte],HS,gate,thr)
                noo,a_oo,lo_oo,hi_oo,_=sel(poo[voo],yoo[voo],tsoo[voo],hoo[voo],HS,gate,thr)
                print(f"   {gate:4s} cov{cov:.1%} thr={thr:.4f}  TEST:n{nte} {a_te:.3f}[{lo_te:.2f},{hi_te:.2f}]   OOS:n{noo} {a_oo:.3f}[{lo_oo:.2f},{hi_oo:.2f}]",flush=True)
    print(f"\n[tickhz] DONE {time.time()-t0:.0f}s",flush=True)

if __name__=="__main__":
    hzs=[int(x) for x in sys.argv[1:]] or [5,15,30,60]
    main(hzs)
