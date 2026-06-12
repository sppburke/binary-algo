"""10-MIN WALK-FORWARD retrain: does reducing the train->test gap break the test25 wall?
All frozen models train 2012-2021 and test 2024/25/26 (3-4yr gap). This retrains annually with an EXPANDING window so
each test year is predicted by a model trained through the PRIOR year (1yr gap). Leakage-free. Directly targets the
2025 regime wall: if test25 jumps from ~0.55-0.59 to 0.62+, gap-reduction is a real lever. Cross-pair+base+OF primary,
NY × confidence selective (cov2% frozen on the val year), non-overlap 600s, CI95. (5m precedent: walk-forward gave
test25 only +0.011 — marginal; re-test at 10m before concluding.)
"""
import os
os.environ["MX_HOR"] = "10"
import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import harness as H, m5_xpair as MX

FOLDS = [("2024", [str(y) for y in range(2012, 2023)], "2023"),
         ("2025", [str(y) for y in range(2012, 2024)], "2024"),
         ("2026", [str(y) for y in range(2012, 2025)], "2025")]
GAP = 600

def build(years, stride=1):
    F = MX.build_xp(years, stride); F = MX.augment(F, years, "xpof"); return F

def fold(test_y, tr_years, va_year, stride=4):
    xtmp = MX.build_xp([va_year]); xpc = MX.xp_cols(xtmp)
    TR = build(tr_years, stride); VA = build([va_year]); TE = build([test_y])
    cols = MX.feat_cols("xpof", TR, xpc)
    cols = [c for c in cols if c in TR.columns and c in VA.columns and c in TE.columns]
    ytr = TR["_y"].astype(int).values; yva = VA["_y"].astype(int).values; yte = TE["_y"].astype(int).values
    L = lgb.LGBMClassifier(objective="binary", metric="auc", learning_rate=0.02, num_leaves=127, min_child_samples=400,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=20, n_estimators=2500, n_jobs=20, verbosity=-1)
    L.fit(TR[cols].astype("float32"), ytr, eval_set=[(VA[cols].astype("float32"), yva)], eval_metric="auc",
          callbacks=[lgb.early_stopping(150), lgb.log_evaluation(0)])
    pva = L.predict_proba(VA[cols].astype("float32"))[:, 1]; pte = L.predict_proba(TE[cols].astype("float32"))[:, 1]
    gv = VA["sess_ny"].values > 0.5; confv = np.abs(pva - 0.5)
    thr = float(np.quantile(confv[gv], 1 - 0.02))
    gt = TE["sess_ny"].values > 0.5; tst = TE["_ts"].values.astype("int64")
    m = gt & (np.abs(pte - 0.5) >= thr); sel = MX.nonoverlap_chrono(tst, m, GAP)
    corr = ((pte[sel] > 0.5).astype(int) == yte[sel]).astype(float) if len(sel) else np.array([])
    acc = corr.mean() if len(sel) else float("nan"); lo, hi = MX.boot(corr)
    auc = roc_auc_score(yte, pte)
    return test_y, len(TR), auc, len(sel), acc, lo, hi, corr

def main():
    print("[m10 walkforward] expanding-window annual retrain (1yr gap) vs frozen-2021 baseline (3-4yr gap); HOR=10", flush=True)
    allc = []
    for ty, tr, va in FOLDS:
        ty_, ntr, auc, nsel, acc, lo, hi, corr = fold(ty, tr, va)
        print(f"=== test {ty} (train {tr[0]}-{tr[-1]} n={ntr:,}, val {va}) === AUC={auc:.4f} n={nsel} acc={acc:.3f} CI95=[{lo:.3f},{hi:.3f}]", flush=True)
        allc.append(corr)
    A = np.concatenate(allc); lo, hi = MX.boot(A)
    print(f"=== WALK-FORWARD COMBINED === n={len(A)} acc={A.mean():.3f} CI95=[{lo:.3f},{hi:.3f}] (breakeven~0.541)", flush=True)

if __name__ == "__main__":
    main()
