"""V5 — extreme-event specialist + meta-labeling.
Train a model ONLY on samples where the recent 5m move is large (top-decile), the regime where
reversion edge is strongest, using base+cross features. Then select its most-confident bets.
Tests whether a model (vs the overfit hand-rule in eda3) finds a generalizing high-precision pocket.
Also runs a meta-labeling layer: secondary model predicts P(primary correct)."""
import sys, time, numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import roc_auc_score
import harness as H, dataset as D

STRIDE=2
EVENT_Q=0.90  # keep samples whose |5m_ret_1| >= train-q90 (strongest-reversion regime)
kw=dict(use_cross=True,use_of=True,use_peer_of=False)

def load(split,stride=1):
    X,y,aux=D.load_split(split,stride=stride,**kw); return X,y,aux

t0=time.time()
Xtr,ytr,atr=load("train",STRIDE); Xva,yva,ava=load("val"); Xte,yte,ate=load("test"); Xoo,yoo,aoo=load("oos")
thr=np.nanquantile(np.abs(Xtr["5m_ret_1"].values),EVENT_Q)
def ev(X): return np.abs(X["5m_ret_1"].values)>=thr
mtr,mva,mte,moo=ev(Xtr),ev(Xva),ev(Xte),ev(Xoo)
print(f"event threshold |5m_ret|>={thr:.6f}; event rows tr={mtr.sum()} va={mva.sum()} te={mte.sum()} oo={moo.sum()} load={time.time()-t0:.0f}s",flush=True)

m=lgb.LGBMClassifier(objective="binary",metric="auc",learning_rate=0.02,num_leaves=127,
    min_child_samples=400,subsample=0.8,subsample_freq=1,colsample_bytree=0.5,reg_lambda=15.0,
    n_estimators=4000,n_jobs=20,verbosity=-1)
m.fit(Xtr[mtr],ytr[mtr],eval_set=[(Xva[mva],yva[mva])],eval_metric="auc",
      callbacks=[lgb.early_stopping(150),lgb.log_evaluation(0)])
pv=m.predict_proba(Xva[mva])[:,1]; pt=m.predict_proba(Xte[mte])[:,1]; po=m.predict_proba(Xoo[moo])[:,1]
yv,yt,yo=yva[mva],yte[mte],yoo[moo]
print(f"EVENT-MODEL AUC val={roc_auc_score(yv,pv):.4f} test={roc_auc_score(yt,pt):.4f} oos={roc_auc_score(yo,po):.4f}")
print("\n== Selective accuracy on EVENT samples (VAL threshold -> TEST/OOS) ==")
H.report("VAL ",yv,pv); H.report("TEST",yt,pt); H.report("OOS ",yo,po)
for cov in (0.2,0.1,0.05,0.02,0.01):
    conf=np.abs(pv-0.5); q=np.quantile(conf,1-cov)
    rt=H.apply_threshold(yt,pt,q); ro=H.apply_threshold(yo,po,q)
    print(f"VALcov={cov:.2%}: TEST n={rt['n']} cov={rt['coverage']:.3%} acc={rt['accuracy']:.4f}"
          f" | OOS n={ro['n']} cov={ro['coverage']:.3%} acc={ro['accuracy']:.4f}")
for tgt in (0.75,0.70,0.65,0.60):
    bv=H.threshold_for_target(yv,pv,target=tgt,min_n=100)
    if bv is None: print(f"target {tgt:.0%}: not hit on VAL-events"); continue
    rt=H.apply_threshold(yt,pt,bv['conf_thr']); ro=H.apply_threshold(yo,po,bv['conf_thr'])
    print(f"target {tgt:.0%}: VAL cov={bv['coverage']:.2%} -> TEST acc={rt['accuracy']:.3f} n={rt['n']}"
          f" | OOS acc={ro['accuracy']:.3f} n={ro['n']}")
