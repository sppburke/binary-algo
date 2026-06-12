"""NZDUSD 15m — D1 features full K=3 seed-ens VAL screen.

Stride-6 proxy confirmed AUC .5471 >> .5319 threshold.
This script runs full 2012-2021 training with K=3 seed-ens to get
the proper VAL AUC for CPCV escalation decision.

Lookahead verification:
  - d1_ret_1 = prior CALENDAR DAY's close-to-close return
  - d1_close sampled at 23:59 UTC (last 1m bar of each calendar day)
  - ffill to 1m: NY session (14:00-21:00 UTC) always sees YESTERDAY's value
  - 23:59 stamp propagates FORWARD from that timestamp until next day's 23:59
  - For 14:00-21:00 UTC bars: last propagated stamp = prior day's 23:59 → SAFE
"""
import os, sys, json
import numpy as np
import pandas as pd
sys.path.insert(0, "/home/sean/git/binary-algo")
import harness as H
from sessions import session_mask
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score

PAIR        = "NZDUSD"
HOR = 15; STEP = 60; GAP = HOR * STEP
RESULT_FILE = "nzdusd_15m_d1feats_k3_result.json"
TRAIN_YEARS  = [str(y) for y in range(2012, 2022)]
VAL_YEARS    = ["2022", "2023"]
BASE_VAL_K3  = 0.5362
ESCALATE_THR = 0.5319

FEATS = H.feature_cols(PAIR)
D1_FEAT_NAMES = ["d1_ret_1", "d1_ret_5", "d1_above_ema20", "d1_above_ema100"]
ALL_FCOLS = FEATS + D1_FEAT_NAMES


def compute_d1_features(df):
    close = df["close"].copy()
    d1_close = close.resample("1D").last().dropna()
    ema20  = d1_close.ewm(span=20, adjust=False).mean()
    ema100 = d1_close.ewm(span=100, adjust=False).mean()
    d1_ret_1 = d1_close.pct_change(1)
    d1_ret_5 = d1_close.pct_change(5)
    d1_sig = pd.DataFrame({
        "d1_above_ema20":  (d1_close >= ema20).astype(float),
        "d1_above_ema100": (d1_close >= ema100).astype(float),
        "d1_ret_1":  d1_ret_1,
        "d1_ret_5":  d1_ret_5,
    })
    return d1_sig.reindex(df.index, method="ffill")


def load_bars(years, session="ny"):
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
        d1f = compute_d1_features(d[["close"]])
        for col in D1_FEAT_NAMES:
            d[col] = d1f[col].values if col in d1f.columns else np.nan
        idx = np.where(valid)[0]
        sub = d.iloc[idx].copy()
        sub["_fwd"] = fr[idx]
        dfs.append(sub)
    return pd.concat(dfs).sort_index() if dfs else pd.DataFrame()


if __name__ == "__main__":
    print("=" * 70)
    print("NZDUSD 15m — D1 features FULL K=3 seed-ens VAL screen")
    print("=" * 70)
    print("Loading train (2012-2021 full)...")
    tr_df = load_bars(TRAIN_YEARS)
    vl_df = load_bars(VAL_YEARS)
    print(f"  train: {len(tr_df):,} | val: {len(vl_df):,}")

    for df in [tr_df, vl_df]:
        df[D1_FEAT_NAMES] = df[D1_FEAT_NAMES].ffill().fillna(0.0)

    X_tr = tr_df[ALL_FCOLS].astype("float32")
    y_tr = (tr_df["_fwd"] > 0).astype(int)
    X_vl = vl_df[ALL_FCOLS].astype("float32")
    y_vl = (vl_df["_fwd"] > 0).astype(int)

    print("\nBase K=3 (239 feats, full train)...")
    probs_base = []
    for s in [42, 0, 7]:
        m = LGBMClassifier(
            n_estimators=600, num_leaves=255, learning_rate=0.02,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.5,
            min_child_samples=400, reg_lambda=20,
            n_jobs=-1, verbosity=-1, random_state=s,
        )
        m.fit(X_tr[FEATS], y_tr)
        p = m.predict_proba(X_vl[FEATS])[:, 1]
        probs_base.append(p)
        print(f"  seed={s}: {roc_auc_score(y_vl, p):.4f}")
    p_base_k3 = np.mean(probs_base, axis=0)
    auc_base_k3 = roc_auc_score(y_vl, p_base_k3)
    print(f"  K=3 ensemble: {auc_base_k3:.4f}")

    print("\n+D1 K=3 (243 feats, full train)...")
    probs_d1 = []
    for s in [42, 0, 7]:
        m = LGBMClassifier(
            n_estimators=600, num_leaves=255, learning_rate=0.02,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.5,
            min_child_samples=400, reg_lambda=20,
            n_jobs=-1, verbosity=-1, random_state=s,
        )
        m.fit(X_tr, y_tr)
        p = m.predict_proba(X_vl)[:, 1]
        probs_d1.append(p)
        print(f"  seed={s}: {roc_auc_score(y_vl, p):.4f}")
    p_d1_k3 = np.mean(probs_d1, axis=0)
    auc_d1_k3 = roc_auc_score(y_vl, p_d1_k3)
    print(f"  K=3 ensemble: {auc_d1_k3:.4f}")

    lift = auc_d1_k3 - auc_base_k3
    lift_vs_ref = auc_d1_k3 - BASE_VAL_K3
    escalate = auc_d1_k3 > ESCALATE_THR

    print(f"\n=== D1+K3 VAL RESULT ===")
    print(f"Base K=3 (this run)  : {auc_base_k3:.4f}")
    print(f"Base K=3 (ref)       : {BASE_VAL_K3:.4f}")
    print(f"+D1 K=3              : {auc_d1_k3:.4f}")
    print(f"Lift vs this run     : {lift:+.4f}")
    print(f"Lift vs ref K=3      : {lift_vs_ref:+.4f}")
    print(f"Escalate (>{ESCALATE_THR}): {'YES → PROCEED TO CPCV' if escalate else 'NO → SUBSUMED'}")

    result = {
        "model": "NZDUSD 15m +D1 features full K=3",
        "d1_features": D1_FEAT_NAMES,
        "base_k3_val_auc_check": float(auc_base_k3),
        "base_k3_val_auc_ref": BASE_VAL_K3,
        "d1_k3_val_auc": float(auc_d1_k3),
        "lift_vs_check": float(lift),
        "lift_vs_ref": float(lift_vs_ref),
        "escalate": bool(escalate),
        "decision": "ESCALATE to CPCV" if escalate else "SUBSUMED",
    }
    with open(RESULT_FILE, "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"\nSaved → {RESULT_FILE}")
