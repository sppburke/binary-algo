"""V2 — add cross-pair lead-lag features (6 USD peers) to the EURUSD reversion model.
Hypothesis: peer moves carry orthogonal short-horizon directional info that lifts AUC
above the single-pair 0.52 wall and enriches the high-precision fade pocket."""
import sys, time
import numpy as np, pandas as pd, lightgbm as lgb
import harness as H, crosspair as CP

TARGET="EURUSD"; STRIDE=int(sys.argv[1]) if len(sys.argv)>1 else 3
base_feats=H.feature_cols(TARGET)
cross_feats=CP.cross_feature_names(TARGET)
all_feats=base_feats+cross_feats
base_cols=base_feats+H.META_COLS

def load_split_merged(split, stride=1):
    parts=[]
    for y in H.SPLITS[split]:
        m=CP.load_year_merged(TARGET,y,base_cols=base_cols)
        if m is None: continue
        m=m[m.valid==True]
        if stride>1: m=m.iloc[::stride]
        parts.append(m)
    df=pd.concat(parts)
    X=df[all_feats].astype("float32")
    y=df["y"].astype(int).values
    return X,y,df[["y","fwd_ret","close"]]

t0=time.time()
Xtr,ytr,_=load_split_merged("train",STRIDE)
Xva,yva,_=load_split_merged("val")
Xte,yte,_=load_split_merged("test")
Xoo,yoo,_=load_split_merged("oos")
print(f"shapes tr={Xtr.shape} va={Xva.shape} te={Xte.shape} oo={Xoo.shape} load={time.time()-t0:.0f}s",flush=True)

params=dict(objective="binary",metric="auc",learning_rate=0.03,num_leaves=255,
    min_child_samples=200,subsample=0.8,subsample_freq=1,colsample_bytree=0.5,
    reg_lambda=5.0,n_estimators=3000,n_jobs=20,verbosity=-1)
model=lgb.LGBMClassifier(**params)
t0=time.time()
model.fit(Xtr,ytr,eval_set=[(Xva,yva)],eval_metric="auc",
    callbacks=[lgb.early_stopping(120),lgb.log_evaluation(300)])
print(f"train={time.time()-t0:.0f}s best_iter={model.best_iteration_}",flush=True)

pva=model.predict_proba(Xva)[:,1]; pte=model.predict_proba(Xte)[:,1]; poo=model.predict_proba(Xoo)[:,1]
print("\n================= RESULTS V2 cross-pair =================")
H.report("VAL ",yva,pva); H.report("TEST",yte,pte); H.report("OOS ",yoo,poo)
print("\n--- selective threshold chosen on VAL ---")
for tgt in (0.75,0.70,0.65,0.60):
    bv=H.threshold_for_target(yva,pva,target=tgt)
    if bv is None: print(f"target {tgt:.0%}: NOT reachable on VAL"); continue
    rte=H.apply_threshold(yte,pte,bv["conf_thr"]); roo=H.apply_threshold(yoo,poo,bv["conf_thr"])
    print(f"target {tgt:.0%}: VAL cov={bv['coverage']:.3%} acc={bv['accuracy']:.3f} thr={bv['conf_thr']:.3f}"
          f" -> TEST cov={rte['coverage']:.3%} acc={rte['accuracy']:.3f} | OOS cov={roo['coverage']:.3%} acc={roo['accuracy']:.3f}")
imp=sorted(zip(all_feats,model.feature_importances_),key=lambda x:-x[1])
print("\nTop 30 feats:",[f"{n}:{v}" for n,v in imp[:30]])
print("\nCross-feat in top-50:",[n for n,v in imp[:50] if n.startswith("X")])
