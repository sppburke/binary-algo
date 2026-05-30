"""V3 — combine ALL orthogonal signals: base MTF + cross-pair lead-lag + order-flow proxy.
Focus on the EXTREME-confidence tail (where a generalizing 75% would live, if anywhere).
Flags: --no-cross --no-of --peer-of --ensemble"""
import sys, time
import numpy as np, pandas as pd, lightgbm as lgb
import harness as H, dataset as D

args=set(sys.argv[1:])
STRIDE=3
use_cross="--no-cross" not in args
use_of="--no-of" not in args
use_peer_of="--peer-of" in args
tag=f"cross={use_cross} of={use_of} peer_of={use_peer_of}"

t0=time.time()
Xtr,ytr,_=D.load_split("train",stride=STRIDE,use_cross=use_cross,use_of=use_of,use_peer_of=use_peer_of)
Xva,yva,auxva=D.load_split("val",use_cross=use_cross,use_of=use_of,use_peer_of=use_peer_of)
Xte,yte,auxte=D.load_split("test",use_cross=use_cross,use_of=use_of,use_peer_of=use_peer_of)
Xoo,yoo,auxoo=D.load_split("oos",use_cross=use_cross,use_of=use_of,use_peer_of=use_peer_of)
feats=list(Xtr.columns)
print(f"[{tag}] shapes tr={Xtr.shape} va={Xva.shape} te={Xte.shape} oo={Xoo.shape} nfeat={len(feats)} load={time.time()-t0:.0f}s",flush=True)

params=dict(objective="binary",metric="auc",learning_rate=0.03,num_leaves=255,
    min_child_samples=300,subsample=0.8,subsample_freq=1,colsample_bytree=0.5,
    reg_lambda=10.0,n_estimators=4000,n_jobs=20,verbosity=-1)
m=lgb.LGBMClassifier(**params)
t0=time.time()
m.fit(Xtr,ytr,eval_set=[(Xva,yva)],eval_metric="auc",callbacks=[lgb.early_stopping(150),lgb.log_evaluation(400)])
print(f"train={time.time()-t0:.0f}s best_iter={m.best_iteration_}",flush=True)

pva=m.predict_proba(Xva)[:,1]; pte=m.predict_proba(Xte)[:,1]; poo=m.predict_proba(Xoo)[:,1]
print(f"\n========== RESULTS V3 [{tag}] ==========")
H.report("VAL ",yva,pva); H.report("TEST",yte,pte); H.report("OOS ",yoo,poo)

print("\n--- EXTREME tail accuracy (top-confidence bets) on TEST & OOS ---")
for cov in (0.05,0.02,0.01,0.005,0.002,0.001):
    # threshold from VAL at this coverage, apply to test/oos
    conf=np.abs(pva-0.5); thr=np.quantile(conf,1-cov)
    rte=H.apply_threshold(yte,pte,thr); roo=H.apply_threshold(yoo,poo,thr)
    print(f"VALcov={cov:.3%} thr={thr:.3f}: TEST n={rte['n']:>6} cov={rte['coverage']:.3%} acc={rte['accuracy']:.4f}"
          f" | OOS n={roo['n']:>5} cov={roo['coverage']:.3%} acc={roo['accuracy']:.4f}")

print("\n--- VAL-chosen target thresholds ---")
for tgt in (0.75,0.70,0.65,0.60):
    bv=H.threshold_for_target(yva,pva,target=tgt)
    if bv is None: print(f"target {tgt:.0%}: NOT reachable on VAL"); continue
    rte=H.apply_threshold(yte,pte,bv["conf_thr"]); roo=H.apply_threshold(yoo,poo,bv["conf_thr"])
    print(f"target {tgt:.0%}: VAL cov={bv['coverage']:.3%} -> TEST n={rte['n']} cov={rte['coverage']:.3%} acc={rte['accuracy']:.3f}"
          f" | OOS n={roo['n']} cov={roo['coverage']:.3%} acc={roo['accuracy']:.3f}")
imp=sorted(zip(feats,m.feature_importances_),key=lambda x:-x[1])
print("\nTop 30:",[f"{n}:{v}" for n,v in imp[:30]])
print("OF in top40:",[n for n,v in imp[:40] if "OF_" in n])
print("cross in top40:",[n for n,v in imp[:40] if n.startswith("X")])

# save model + test/oos probs for downstream meta-labeling
import joblib, os
os.makedirs("models",exist_ok=True)
joblib.dump({"model":m,"feats":feats,"tag":tag}, f"models/v3_{'base' if not (use_cross or use_of) else 'full'}.joblib")
np.savez("models/v3_probs.npz", pva=pva,pte=pte,poo=poo,yva=yva,yte=yte,yoo=yoo)
