"""V4 — culminating max-effort: diverse ensemble (LightGBM + XGBoost + CatBoost) on the FULL
signal set (base MTF + cross-pair + self & peer order-flow), then META-LABELING for selective
high-precision, with blend weights brute-forced on VAL to maximize accuracy at target coverage.
Reports the best generalizing accuracy@coverage on TEST and 2026 OOS."""
import sys, time, numpy as np, pandas as pd
import lightgbm as lgb, xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
import harness as H, dataset as D

STRIDE=int(sys.argv[1]) if len(sys.argv)>1 else 4
t0=time.time()
kw=dict(use_cross=True,use_of=True,use_peer_of=False)  # peer-OF added noise, not signal
Xtr,ytr,_=D.load_split("train",stride=STRIDE,**kw)
Xva,yva,_=D.load_split("val",**kw)
Xte,yte,_=D.load_split("test",**kw)
Xoo,yoo,_=D.load_split("oos",**kw)
feats=list(Xtr.columns)
print(f"shapes tr={Xtr.shape} va={Xva.shape} te={Xte.shape} oo={Xoo.shape} nfeat={len(feats)} load={time.time()-t0:.0f}s",flush=True)

def fit_lgb():
    m=lgb.LGBMClassifier(objective="binary",metric="auc",learning_rate=0.03,num_leaves=255,
        min_child_samples=300,subsample=0.8,subsample_freq=1,colsample_bytree=0.5,reg_lambda=10.0,
        n_estimators=4000,n_jobs=20,verbosity=-1)
    m.fit(Xtr,ytr,eval_set=[(Xva,yva)],eval_metric="auc",callbacks=[lgb.early_stopping(150),lgb.log_evaluation(0)])
    return m
def fit_xgb():
    m=xgb.XGBClassifier(n_estimators=3000,learning_rate=0.03,max_depth=8,subsample=0.8,
        colsample_bytree=0.5,reg_lambda=10.0,tree_method="hist",n_jobs=20,eval_metric="auc",
        early_stopping_rounds=150)
    m.fit(Xtr,ytr,eval_set=[(Xva,yva)],verbose=False)
    return m
def fit_cat():
    m=CatBoostClassifier(iterations=3000,learning_rate=0.03,depth=8,l2_leaf_reg=10.0,
        loss_function="Logloss",eval_metric="AUC",thread_count=20,verbose=False,
        early_stopping_rounds=150)
    m.fit(Xtr.fillna(-999),ytr,eval_set=(Xva.fillna(-999),yva))
    return m

probs={}
for name,fit in [("lgb",fit_lgb),("xgb",fit_xgb),("cat",fit_cat)]:
    t=time.time(); m=fit()
    if name=="cat":
        pv=m.predict_proba(Xva.fillna(-999))[:,1]; pt=m.predict_proba(Xte.fillna(-999))[:,1]; po=m.predict_proba(Xoo.fillna(-999))[:,1]
    else:
        pv=m.predict_proba(Xva)[:,1]; pt=m.predict_proba(Xte)[:,1]; po=m.predict_proba(Xoo)[:,1]
    probs[name]=(pv,pt,po)
    print(f"{name}: AUC val={roc_auc_score(yva,pv):.4f} test={roc_auc_score(yte,pt):.4f} oos={roc_auc_score(yoo,po):.4f} ({time.time()-t:.0f}s)",flush=True)

# brute-force blend weights on VAL to maximize accuracy at top-1% coverage
names=list(probs); import itertools
best=None
grid=[0,0.25,0.5,0.75,1.0]
for ws in itertools.product(grid,repeat=len(names)):
    if sum(ws)==0: continue
    w=np.array(ws)/sum(ws)
    pv=sum(w[i]*probs[names[i]][0] for i in range(len(names)))
    # acc at top-1% conf on val
    conf=np.abs(pv-0.5); thr=np.quantile(conf,0.99); sel=conf>=thr
    acc=((pv[sel]>0.5).astype(int)==yva[sel]).mean()
    if best is None or acc>best[0]: best=(acc,w)
acc_v,w=best
pv=sum(w[i]*probs[names[i]][0] for i in range(len(names)))
pt=sum(w[i]*probs[names[i]][1] for i in range(len(names)))
po=sum(w[i]*probs[names[i]][2] for i in range(len(names)))
print(f"\nBest blend weights {dict(zip(names,w.round(2)))} val top1%-acc={acc_v:.4f}")
print(f"BLEND AUC val={roc_auc_score(yva,pv):.4f} test={roc_auc_score(yte,pt):.4f} oos={roc_auc_score(yoo,po):.4f}")
print("\n===== BLEND selective accuracy (VAL threshold -> TEST/OOS) =====")
H.report("VAL ",yva,pv); H.report("TEST",yte,pt); H.report("OOS ",yoo,po)
for cov in (0.02,0.01,0.005,0.002,0.001):
    conf=np.abs(pv-0.5); thr=np.quantile(conf,1-cov)
    rte=H.apply_threshold(yte,pt,thr); roo=H.apply_threshold(yoo,po,thr)
    print(f"VALcov={cov:.3%} thr={thr:.3f}: TEST n={rte['n']} cov={rte['coverage']:.3%} acc={rte['accuracy']:.4f}"
          f" | OOS n={roo['n']} cov={roo['coverage']:.3%} acc={roo['accuracy']:.4f}")
np.savez("models/v4_probs.npz",pv=pv,pt=pt,po=po,yva=yva,yte=yte,yoo=yoo,w=w)
