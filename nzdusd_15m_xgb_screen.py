"""NZDUSD 15m — XGBoost (A1 row) + LGB+XGB blend screen.

SWEEP_MATRIX A1: 3-model GBM ensemble (lgb+xgb+cat). LightGBM K=3 already certified.
XGBoost = level-wise tree growth (vs LightGBM leaf-wise) — genuinely different inductive
bias that the LGB architectural ensemble (127/255/511 leaves) did NOT test.

Prior (Tier-3 from arch-ens K=9 .5254): XGB likely gives ~.519-.525, LGB+XGB ensemble
probably dilutes below K=3 LGB (.5256 stride-6). But "run, don't argue" (SWEEP_MATRIX A1).

Falsifier: LGB+XGB ensemble VAL AUC > .5319 → ESCALATE to CPCV.
           LGB+XGB ensemble VAL AUC > .5362 → SUPERSEDES incumbent.
"""
import os, sys, json
import numpy as np
import pandas as pd
sys.path.insert(0, "/home/sean/git/binary-algo")
import harness as H
from sessions import session_mask
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

PAIR        = "NZDUSD"
HOR = 15; STEP = 60; GAP = HOR * STEP
RESULT_FILE = "nzdusd_15m_xgb_result.json"
TRAIN_YEARS  = [str(y) for y in range(2012, 2022)]
VAL_YEARS    = ["2022", "2023"]
TR_STRIDE    = 6
BASE_VAL_AUC = 0.5219
OWN_K3_AUC   = 0.5362
ESCALATE_THR = 0.5319

FEATS = H.feature_cols(PAIR)


def load_bars(years, stride=1, session="ny"):
    dfs = []
    for yr in years:
        p = f"{H.FEAT_DIR}/{PAIR}_{yr}.parquet"
        if not os.path.exists(p):
            continue
        d = pd.read_parquet(p, columns=FEATS + ["close"])
        d = d[~d.index.duplicated(keep="last")]
        c  = d["close"].values.astype(float)
        ts = d.index.values.astype("datetime64[s]").astype("int64")
        n  = len(d)
        contig = np.zeros(n, bool)
        contig[:n - HOR] = (ts[HOR:] - ts[:-HOR]) == GAP
        fr = np.full(n, np.nan)
        fr[:n - HOR] = c[HOR:] / c[:-HOR] - 1.0
        keep_feat = d[FEATS].isna().mean(axis=1).values < 0.5
        valid     = contig & np.isfinite(fr) & keep_feat
        smask     = session_mask(ts, session)
        valid     = valid & smask
        idx = np.where(valid)[0]
        if stride > 1:
            idx = idx[::stride]
        sub = d.iloc[idx].copy()
        sub["_fwd"] = fr[idx]
        dfs.append(sub)
    return pd.concat(dfs).sort_index() if dfs else pd.DataFrame()


if __name__ == "__main__":
    print("=" * 70)
    print("NZDUSD 15m — XGBoost A1 screen + LGB K=3 blend")
    print(f"Escalate threshold: VAL AUC > {ESCALATE_THR:.4f}")
    print("=" * 70)

    train_yrs = TRAIN_YEARS[::TR_STRIDE]

    print(f"\n[1] Loading bars...")
    tr_df = load_bars(train_yrs, stride=1)
    vl_df = load_bars(VAL_YEARS,  stride=1)
    print(f"  train: {len(tr_df):,} | val: {len(vl_df):,}")

    X_tr = tr_df[FEATS].astype("float32")
    y_tr = (tr_df["_fwd"] > 0).astype(int)
    X_vl = vl_df[FEATS].astype("float32")
    y_vl = (vl_df["_fwd"] > 0).astype(int)

    print(f"\n[2] LightGBM K=3 seeds (incumbent architecture)...")
    lgb_probs = []
    for s in [42, 0, 7]:
        m = LGBMClassifier(
            n_estimators=600, num_leaves=255, learning_rate=0.02,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.5,
            min_child_samples=400, reg_lambda=20,
            n_jobs=-1, verbosity=-1, random_state=s,
        )
        m.fit(X_tr, y_tr)
        p = m.predict_proba(X_vl)[:, 1]
        auc_s = roc_auc_score(y_vl, p)
        print(f"  LGB seed={s}: {auc_s:.4f}")
        lgb_probs.append(p)
    p_lgb = np.mean(lgb_probs, axis=0)
    auc_lgb = roc_auc_score(y_vl, p_lgb)
    print(f"  LGB K=3 ensemble: {auc_lgb:.4f}")

    print(f"\n[3] XGBoost (level-wise trees, different inductive bias)...")
    # Match LGB params as closely as possible: colsample=0.5, subsample=0.8
    xgb = XGBClassifier(
        n_estimators=600, max_leaves=255, learning_rate=0.02,
        subsample=0.8, colsample_bytree=0.5,
        min_child_weight=400, reg_lambda=20,
        tree_method="hist", device="cpu",
        n_jobs=-1, verbosity=0, random_state=42,
        eval_metric="logloss",
    )
    xgb.fit(X_tr, y_tr)
    p_xgb = xgb.predict_proba(X_vl)[:, 1]
    auc_xgb = roc_auc_score(y_vl, p_xgb)
    print(f"  XGB single-seed: {auc_xgb:.4f}")

    # Correlation between LGB and XGB predictions
    pred_corr = np.corrcoef(p_lgb, p_xgb)[0, 1]
    print(f"  LGB vs XGB prediction correlation: {pred_corr:.4f}")

    print(f"\n[4] Blend experiments...")
    results = {}
    for alpha in [0.9, 0.8, 0.7, 0.5]:
        p_blend = alpha * p_lgb + (1 - alpha) * p_xgb
        auc_blend = roc_auc_score(y_vl, p_blend)
        results[alpha] = float(auc_blend)
        print(f"  α={alpha:.1f} LGB + {1-alpha:.1f} XGB: {auc_blend:.4f}")

    best_alpha = max(results, key=results.get)
    best_auc   = results[best_alpha]
    lift       = best_auc - BASE_VAL_AUC
    lift_k3    = best_auc - OWN_K3_AUC
    escalate   = best_auc > ESCALATE_THR

    print(f"\n=== XGB + LGB BLEND VAL RESULT ===")
    print(f"Base single-seed (stride-6) : {BASE_VAL_AUC:.4f}")
    print(f"Own-pair K=3 (full train)   : {OWN_K3_AUC:.4f}")
    print(f"LGB K=3 ensemble            : {auc_lgb:.4f}")
    print(f"XGB single-seed             : {auc_xgb:.4f}")
    print(f"LGB/XGB corr                : {pred_corr:.4f}")
    print(f"Best blend (α={best_alpha:.1f})        : {best_auc:.4f}")
    print(f"Lift vs stride-6 base       : {lift:+.4f}")
    print(f"Lift vs own K=3             : {lift_k3:+.4f}")
    print(f"Escalate ({ESCALATE_THR})   : {'YES → ESCALATE' if escalate else 'NO → SUBSUMED'}")

    result = {
        "model": "NZDUSD.15m XGBoost A1 screen + LGB K=3 blend",
        "base_val_auc_stride6": BASE_VAL_AUC,
        "own_k3_val_auc_full": OWN_K3_AUC,
        "lgb_k3_val_auc": float(auc_lgb),
        "xgb_single_val_auc": float(auc_xgb),
        "lgb_xgb_corr": float(pred_corr),
        "blend_results": results,
        "best_blend_alpha": float(best_alpha),
        "best_blend_auc": float(best_auc),
        "lift_vs_base": float(lift),
        "lift_vs_own_k3": float(lift_k3),
        "escalate_threshold": ESCALATE_THR,
        "escalate": bool(escalate),
        "decision": "ESCALATE to CPCV" if escalate else "SUBSUMED",
    }
    with open(RESULT_FILE, "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"\nSaved → {RESULT_FILE}")
