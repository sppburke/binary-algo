"""15-MINUTE binary direction — full ensemble + selective.
Reuses base multi-timeframe + cross-pair features (horizon-independent) with a 15-min gap-aware
label. LGBM+XGB+CatBoost ensemble, blended, with accuracy@coverage and an explicit both-sides
>=75% gate on TEST (2024-25) and 2026 OOS (held out). Saves probs for verify.py."""
import time, numpy as np, pandas as pd, os
import lightgbm as lgb, xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
import harness as H
import exp_horizon as EH   # provides load_split with 15-min label (HOR=15 default) + base/cross feats

assert EH.HOR==15, f"expected 15-min horizon, got {EH.HOR}"
STRIDE=3
t0=time.time()
Xtr,ytr=EH.load_split("train",STRIDE); Xva,yva=EH.load_split("val")
Xte,yte=EH.load_split("test"); Xoo,yoo=EH.load_split("oos")
print(f"15m shapes tr={Xtr.shape} va={Xva.shape} te={Xte.shape} oo={Xoo.shape} ymean_tr={ytr.mean():.4f} load={time.time()-t0:.0f}s",flush=True)

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
print(f"\nBLEND 15m AUC val={roc_auc_score(yva,pv):.4f} test={roc_auc_score(yte,pt):.4f} oos={roc_auc_score(yoo,po):.4f}")
print("\naccuracy@coverage (VAL-confidence threshold applied to held-out TEST & 2026 OOS):")
print(f"  {'cov':>7} {'TESTacc':>8} {'TESTn':>8} {'OOSacc':>8} {'OOSn':>7} {'both>=75?':>9}")
for cov in (0.5,0.2,0.1,0.05,0.02,0.01,0.005,0.002,0.001):
    thr=np.quantile(np.abs(pv-0.5),1-cov)
    rt=H.apply_threshold(yte,pt,thr); ro=H.apply_threshold(yoo,po,thr)
    ok="YES" if (rt['accuracy']>=0.75 and ro['accuracy']>=0.75 and rt['n']>=100 and ro['n']>=100) else ""
    print(f"  {cov:>7.2%} {rt['accuracy']:>8.4f} {rt['n']:>8} {ro['accuracy']:>8.4f} {ro['n']:>7} {ok:>9}")
print("\nVAL-target thresholds:")
for tgt in (0.75,0.70,0.65,0.60):
    bv=H.threshold_for_target(yva,pv,target=tgt,min_n=300)
    if bv is None: print(f"  target {tgt:.0%}: not hit on VAL"); continue
    rt=H.apply_threshold(yte,pt,bv['conf_thr']); ro=H.apply_threshold(yoo,po,bv['conf_thr'])
    print(f"  target {tgt:.0%}: VAL cov={bv['coverage']:.2%} -> TEST {rt['accuracy']:.3f} (n={rt['n']}) | OOS {ro['accuracy']:.3f} (n={ro['n']})")
os.makedirs("models",exist_ok=True)
np.savez("models/probs_15m.npz",pva=pv,pte=pt,poo=po,yva=yva,yte=yte,yoo=yoo)
print("EXP15M DONE")
