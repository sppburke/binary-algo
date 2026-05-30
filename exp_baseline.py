"""
Variant V1 — Baseline LightGBM on full 239-feature multi-timeframe set.
Establishes the wall and the accuracy@coverage curve (selective prediction).
"""
import sys, time, json
import numpy as np
import lightgbm as lgb
import harness as H

PAIR = sys.argv[1] if len(sys.argv) > 1 else "EURUSD"
TRAIN_STRIDE = int(sys.argv[2]) if len(sys.argv) > 2 else 3

t0 = time.time()
print(f"Loading {PAIR} (train stride={TRAIN_STRIDE})...", flush=True)
Xtr, ytr, _ = H.load_split(PAIR, "train", stride=TRAIN_STRIDE)
Xva, yva, _ = H.load_split(PAIR, "val")
Xte, yte, _ = H.load_split(PAIR, "test")
Xoo, yoo, _ = H.load_split(PAIR, "oos")
print(f"shapes train={Xtr.shape} val={Xva.shape} test={Xte.shape} oos={Xoo.shape} "
      f"load={time.time()-t0:.0f}s", flush=True)

params = dict(
    objective="binary", metric="auc", learning_rate=0.03,
    num_leaves=255, min_child_samples=200, subsample=0.8, subsample_freq=1,
    colsample_bytree=0.6, reg_lambda=5.0, max_depth=-1,
    n_estimators=2000, n_jobs=20, verbosity=-1,
)
model = lgb.LGBMClassifier(**params)
t0 = time.time()
model.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric="auc",
          callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)])
print(f"train={time.time()-t0:.0f}s best_iter={model.best_iteration_}", flush=True)

pva = model.predict_proba(Xva)[:, 1]
pte = model.predict_proba(Xte)[:, 1]
poo = model.predict_proba(Xoo)[:, 1]

print("\n================= RESULTS V1 baseline =================")
H.report("VAL ", yva, pva)
H.report("TEST", yte, pte)
H.report("OOS ", yoo, poo)

print("\n--- selective threshold chosen on VAL for target 75% ---")
for tgt in (0.75, 0.70, 0.65, 0.60):
    bv = H.threshold_for_target(yva, pva, target=tgt)
    if bv is None:
        print(f"target {tgt:.0%}: NOT reachable on VAL"); continue
    rte = H.apply_threshold(yte, pte, bv["conf_thr"])
    roo = H.apply_threshold(yoo, poo, bv["conf_thr"])
    print(f"target {tgt:.0%}: VAL cov={bv['coverage']:.3%} acc={bv['accuracy']:.3f} "
          f"thr={bv['conf_thr']:.3f} -> TEST cov={rte['coverage']:.3%} acc={rte['accuracy']:.3f} "
          f"| OOS cov={roo['coverage']:.3%} acc={roo['accuracy']:.3f}")

# feature importance
imp = sorted(zip(H.feature_cols(PAIR), model.feature_importances_), key=lambda x: -x[1])
print("\nTop 25 features:", [f"{n}:{v}" for n, v in imp[:25]])
