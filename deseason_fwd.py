"""FORWARD-HOLDOUT falsifier for the time-of-day magnitude lever (deployment-faithful, NOT pooled CPCV).

The adversarial review (workflow wwtci9slp, statistical lens) showed the +0.014 pooled-CPCV tod lift does NOT survive a
frozen-past forward holdout (train<=2022 -> test 2024-26 gave dAUC -0.0054) because CPCV's flanked test folds let the
model memorize era-LOCAL intraday-vol seasonality, which is NON-STATIONARY (the hour-of-day vol shape flattened in 2025).
This quantifies the decay per-year: train on 2012-2023, test EACH of 2024/2025/2026 separately, base vs +tod.

Reuses deseason_mag.load() (identical feature construction to the certified model). Reports per-year base/+tod AUC +
decile lift + dAUC. The honest, deployment-faithful magnitude of the tod lever.
"""
import sys, time, numpy as np, pandas as pd
import deseason_mag as DM
from sklearn.metrics import roc_auc_score
T0 = time.time()
def hb(m): print(f"[{time.time()-T0:.0f}s] {m}", flush=True)

HOR = int(sys.argv[1]) if len(sys.argv) > 1 else 30
TRAIN_MAX = 2023
hb(f"loading hor={HOR}m ...")
pe, rv30, rv120, aret, ts, mod, tod = DM.load(HOR)
base = np.column_stack([-pe, rv30, rv120]).astype(np.float32)
yr = pd.to_datetime(ts, unit="s", utc=True).year.values
tr = yr <= TRAIN_MAX
thr = float(np.nanquantile(aret[tr], 0.75))                 # TRAIN-only large-move threshold
ytr = (aret[tr] >= thr).astype(int)
hb(f"train(<= {TRAIN_MAX}) n={tr.sum():,} thr={thr:.6g} up-rate={ytr.mean():.3f}")

def lift(pp, a):
    q = np.quantile(pp, [0.9, 0.1]); return float(a[pp >= q[0]].mean() / max(a[pp <= q[1]].mean(), 1e-12))

arms = {"base": base, "+tod": np.column_stack([base, tod]).astype(np.float32)}
fitted = {}
for nm, X in arms.items():
    m = DM.mk_lgb(); m.fit(X[tr], ytr); fitted[nm] = m; hb(f"fit {nm} ({X.shape[1]} feats)")

print(f"\n{'year':>6} | {'n':>9} | {'base AUC':>8} {'lift':>5} | {'+tod AUC':>8} {'lift':>5} | {'dAUC':>8} {'dLift':>6}")
rows = {}
for Y in (2024, 2025, 2026):
    te = yr == Y
    if te.sum() < 1000: continue
    yte = (aret[te] >= thr).astype(int); a = aret[te]
    pb = fitted["base"].predict_proba(base[te])[:, 1]
    pt = fitted["+tod"].predict_proba(arms["+tod"][te])[:, 1]
    ab, at = roc_auc_score(yte, pb), roc_auc_score(yte, pt)
    lb, lt = lift(pb, a), lift(pt, a)
    rows[str(Y)] = dict(n=int(te.sum()), base_auc=round(ab, 4), tod_auc=round(at, 4),
                        dAUC=round(at - ab, 4), base_lift=round(lb, 2), tod_lift=round(lt, 2), dLift=round(lt - lb, 2))
    print(f"{Y:>6} | {te.sum():>9,} | {ab:>8.4f} {lb:>5.2f} | {at:>8.4f} {lt:>5.2f} | {at-ab:>+8.4f} {lt-lb:>+6.2f}")
# pooled 2024-2026
te = yr >= 2024; yte = (aret[te] >= thr).astype(int); a = aret[te]
pb = fitted["base"].predict_proba(base[te])[:, 1]; pt = fitted["+tod"].predict_proba(arms["+tod"][te])[:, 1]
ab, at = roc_auc_score(yte, pb), roc_auc_score(yte, pt)
rows["2024-2026"] = dict(n=int(te.sum()), base_auc=round(ab, 4), tod_auc=round(at, 4), dAUC=round(at - ab, 4),
                         base_lift=round(lift(pb, a), 2), tod_lift=round(lift(pt, a), 2))
print(f"{'24-26':>6} | {te.sum():>9,} | {ab:>8.4f} {lift(pb,a):>5.2f} | {at:>8.4f} {lift(pt,a):>5.2f} | {at-ab:>+8.4f}")
import json
json.dump({"horizon_min": HOR, "train_max": TRAIN_MAX, "design": "frozen-past forward holdout (deployment-faithful)",
           "by_year": rows}, open(f"/home/sean/git/binary-algo/deseason_fwd_{HOR}m_result.json", "w"), indent=1)
hb("DONE")
