"""Train the chosen config (base + cross-pair + self order-flow) for EVERY pair and save
OOS probabilities, so the verification harness can report the prediction rate per pair.
This is the deliverable: a reproducible per-pair selective strategy + honest verification."""
import sys, time, numpy as np, pandas as pd, lightgbm as lgb, os
from sklearn.metrics import roc_auc_score
import harness as H, dataset as D

PAIRS=sys.argv[1:] or ["EURUSD","GBPUSD","AUDUSD","NZDUSD","USDCAD","USDCHF","USDJPY"]
STRIDE=4
os.makedirs("models",exist_ok=True)
summary=[]
for pair in PAIRS:
    t0=time.time()
    kw=dict(target=pair,use_cross=True,use_of=True,use_peer_of=False)
    Xtr,ytr,_=D.load_split("train",stride=STRIDE,**kw)
    Xva,yva,_=D.load_split("val",**kw)
    Xte,yte,_=D.load_split("test",**kw)
    Xoo,yoo,_=D.load_split("oos",**kw)
    m=lgb.LGBMClassifier(objective="binary",metric="auc",learning_rate=0.03,num_leaves=255,
        min_child_samples=300,subsample=0.8,subsample_freq=1,colsample_bytree=0.5,reg_lambda=10.0,
        n_estimators=4000,n_jobs=20,verbosity=-1)
    m.fit(Xtr,ytr,eval_set=[(Xva,yva)],eval_metric="auc",callbacks=[lgb.early_stopping(150),lgb.log_evaluation(0)])
    pv=m.predict_proba(Xva)[:,1]; pt=m.predict_proba(Xte)[:,1]; po=m.predict_proba(Xoo)[:,1]
    np.savez(f"models/probs_{pair}.npz",pva=pv,pte=pt,poo=po,yva=yva,yte=yte,yoo=yoo)
    a=(roc_auc_score(yva,pv),roc_auc_score(yte,pt),roc_auc_score(yoo,po))
    # best generalizing selective acc: VAL top-1% threshold -> test/oos
    conf=np.abs(pv-0.5); thr=np.quantile(conf,0.99)
    rt=H.apply_threshold(yte,pt,thr); ro=H.apply_threshold(yoo,po,thr)
    summary.append((pair,*a,rt['accuracy'],ro['accuracy'],ro['n']))
    print(f"{pair}: AUC val/test/oos={a[0]:.4f}/{a[1]:.4f}/{a[2]:.4f} "
          f"top1%sel TEST acc={rt['accuracy']:.4f} OOS acc={ro['accuracy']:.4f} ({time.time()-t0:.0f}s)",flush=True)

print("\n===== PER-PAIR SUMMARY =====")
print(f"{'pair':>7} {'AUCval':>7} {'AUCtest':>7} {'AUCoos':>7} {'selTEST':>8} {'selOOS':>8} {'oosN':>6}")
for r in summary:
    print(f"{r[0]:>7} {r[1]:>7.4f} {r[2]:>7.4f} {r[3]:>7.4f} {r[4]:>8.4f} {r[5]:>8.4f} {r[6]:>6}")
