"""NZDUSD 15m — OFI VAL screen using pre-computed features_of/ (18 OF features).
Converts OFI avenue from Tier-3 (SUBSUMED via 5-major cross-pair pattern) to
Tier-1 NZDUSD-specific evidence.

18 OF features: OF_of_norm/sum at 1/3/5/10/15/30m, OF_of_uptick 5/15,
                OF_kyle 5/15, OF_of_accel, OF_of_persist

Falsifier: VAL AUC > base (.5219) + .010 = .5319 → ESCALATE to CPCV.
Expected: KILLED (~+.001-.003).
"""
import os, json
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score

FEAT_DIR = "/home/sean/git/binary-algo/features"
OF_DIR   = "/home/sean/git/binary-algo/features_of"
PAIR     = "NZDUSD"
TR_STRIDE    = 6
NUM_LEAVES   = 255
BASE_VAL_AUC = 0.5219
THRESHOLD    = BASE_VAL_AUC + 0.010   # 0.5319

META_COLS = {"y", "fwd_ret", "valid", "close"}

TRAIN_YEARS = [str(y) for y in range(2012, 2022)]
VAL_YEARS   = ["2022", "2023"]


def load_split_with_of(years, stride=1):
    """Process year-by-year to avoid OOM: valid+stride applied per year before concat."""
    parts = []
    feat_cols_determined = False
    feat_cols = None

    for yr in years:
        bp = f"{FEAT_DIR}/{PAIR}_{yr}.parquet"
        op = f"{OF_DIR}/{PAIR}_{yr}.parquet"
        if not os.path.exists(bp):
            continue

        base = pd.read_parquet(bp)
        if not feat_cols_determined:
            feat_cols = [c for c in base.columns if c not in META_COLS]
            feat_cols_determined = True

        # valid filter before merge (reduces rows before OF join)
        base = base[base["valid"] == True]

        if os.path.exists(op):
            of = pd.read_parquet(op)
            if base.index.dtype != of.index.dtype:
                of.index = of.index.astype(base.index.dtype)
            base = base.join(of, how="left")

        if stride > 1:
            base = base.iloc[::stride]

        parts.append(base)
        del of  # free immediately

    df = pd.concat(parts)
    of_new_cols = [c for c in df.columns if c.startswith("OF_")]
    all_feat_cols = feat_cols + of_new_cols

    X = df[all_feat_cols].astype("float32")
    keep = X.isna().mean(axis=1) < 0.5
    X = X[keep]
    y = df.loc[keep.index[keep], "y"].astype(int).values
    return X, y


print("Loading train...")
X_tr, y_tr = load_split_with_of(TRAIN_YEARS, stride=TR_STRIDE)
print(f"  train: {X_tr.shape}, label balance: {y_tr.mean():.3f}")

print("Loading val...")
X_vl, y_vl = load_split_with_of(VAL_YEARS, stride=1)
print(f"  val:   {X_vl.shape}, label balance: {y_vl.mean():.3f}")

# check OF coverage in val
of_cols = [c for c in X_vl.columns if c.startswith("OF_")]
print(f"  OF cols: {len(of_cols)} | val NaN rate: {X_vl[of_cols].isna().mean().mean():.3f}")

print(f"\nTotal features: {X_tr.shape[1]} (base 239 + {len(of_cols)} OF)")

model = LGBMClassifier(
    n_estimators=600, num_leaves=NUM_LEAVES, learning_rate=0.02,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.5,
    min_child_samples=400, reg_lambda=20,
    n_jobs=-1, random_state=42, verbose=-1,
)
print("\nFitting...")
model.fit(X_tr, y_tr)

p_vl = model.predict_proba(X_vl)[:, 1]
val_auc = roc_auc_score(y_vl, p_vl)
lift    = val_auc - BASE_VAL_AUC
status  = "ESCALATE" if val_auc >= THRESHOLD else "SUBSUMED"

# top-10 OF features by importance
imp = pd.Series(model.feature_importances_, index=X_tr.columns)
of_imp = imp[of_cols].sort_values(ascending=False)

print(f"\n=== OFI VAL SCREEN RESULT ===")
print(f"Base VAL AUC : {BASE_VAL_AUC:.4f}")
print(f"OFI VAL AUC  : {val_auc:.4f}")
print(f"Lift         : {lift:+.4f}")
print(f"Threshold    : {THRESHOLD:.4f}")
print(f"Status       : {status}")
print(f"\nOF feature importances (top 10 / {len(of_cols)}):")
print(of_imp.head(10).to_string())

result = {
    "ofi_val_auc"   : float(val_auc),
    "base_val_auc"  : BASE_VAL_AUC,
    "lift"          : float(lift),
    "status"        : status,
    "of_features"   : len(of_cols),
    "total_features": int(X_tr.shape[1]),
    "of_cols"       : of_cols,
}
with open("nzdusd_15m_ofi_screen_result.json", "w") as fh:
    json.dump(result, fh, indent=2)
print("\nSaved → nzdusd_15m_ofi_screen_result.json")
