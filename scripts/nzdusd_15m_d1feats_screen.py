"""NZDUSD 15m — Daily (D1) trend features screen.

Feature gap: 239-feat set covers 1m/5m/15m/30m/1h/4h timeframes.
Max lookback via 4h_ema100 = 100 bars × 4h = 400h ≈ 17 days.
D1 trend context (20-day, 100-day EMA; weekly momentum) = genuinely absent.

Mechanism: Medium-to-long-term NZD trend alignment with 15m direction.
- Multi-month trend (D1_above_ema100 = 100-day EMA): USD-strength regime indicator
- Monthly trend (D1_above_ema20 = 20-day EMA): RBNZ cycle proxy
- D1 momentum (d1_ret_1, d1_ret_5): prior-day and 5-day momentum

Features added (4 new):
  d1_above_ema20: above 20-day moving EMA (approx 1 month)
  d1_above_ema100: above 100-day moving EMA (approx 5 months)
  d1_ret_1: prior daily bar return
  d1_ret_5: 5-day cumulative return (weekly trend)
Total: 239 + 4 = 243 features

Expected result: SUBSUMED. 4h features already capture up to 17-day trend; adding
50-100 day context at 15m scale unlikely to survive colsample dilution.
Falsifier: VAL AUC > .5319 → ESCALATE.
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
RESULT_FILE = "nzdusd_15m_d1feats_result.json"
TRAIN_YEARS  = [str(y) for y in range(2012, 2022)]
VAL_YEARS    = ["2022", "2023"]
TR_STRIDE    = 6
BASE_VAL_AUC = 0.5219
OWN_K3_AUC   = 0.5362
ESCALATE_THR = 0.5319

FEATS = H.feature_cols(PAIR)
D1_FEAT_NAMES = ["d1_above_ema20", "d1_above_ema100", "d1_ret_1", "d1_ret_5"]


def compute_d1_features(df):
    """Compute daily trend features from 1m close prices."""
    close = df["close"].copy()
    # Resample to daily close
    d1_close = close.resample("1D").last().dropna()
    # 20-day and 100-day EMAs on daily closes
    ema20  = d1_close.ewm(span=20, adjust=False).mean()
    ema100 = d1_close.ewm(span=100, adjust=False).mean()
    # Daily returns
    d1_ret_1 = d1_close.pct_change(1)
    d1_ret_5 = d1_close.pct_change(5)
    # Combine into daily signals
    d1_sig = pd.DataFrame({
        "d1_above_ema20":  (d1_close >= ema20).astype(float),
        "d1_above_ema100": (d1_close >= ema100).astype(float),
        "d1_ret_1":  d1_ret_1,
        "d1_ret_5":  d1_ret_5,
    })
    # Forward-fill daily signals to 1m resolution (each 1m bar gets that day's values)
    d1_ff = d1_sig.reindex(df.index, method="ffill")
    return d1_ff


def load_bars_with_d1(years, stride=1, session="ny"):
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

        # Compute D1 features on full year
        d1f = compute_d1_features(d[["close"]])
        for col in D1_FEAT_NAMES:
            d[col] = d1f[col].values if col in d1f.columns else np.nan

        idx = np.where(valid)[0]
        if stride > 1:
            idx = idx[::stride]
        sub = d.iloc[idx].copy()
        sub["_fwd"] = fr[idx]
        dfs.append(sub)
    return pd.concat(dfs).sort_index() if dfs else pd.DataFrame()


if __name__ == "__main__":
    print("=" * 70)
    print("NZDUSD 15m — D1 daily trend features screen")
    print(f"Features: d1_above_ema20, d1_above_ema100, d1_ret_1, d1_ret_5")
    print(f"Escalate threshold: VAL AUC > {ESCALATE_THR:.4f}")
    print("=" * 70)

    all_fcols = FEATS + D1_FEAT_NAMES
    train_yrs = TRAIN_YEARS[::TR_STRIDE]
    tr_df = load_bars_with_d1(train_yrs, stride=1)
    vl_df = load_bars_with_d1(VAL_YEARS, stride=1)
    print(f"  train: {len(tr_df):,} | val: {len(vl_df):,}")
    print(f"  Feature total: {len(FEATS)} base + 4 D1 = {len(all_fcols)}")

    tr_df[D1_FEAT_NAMES] = tr_df[D1_FEAT_NAMES].fillna(method="ffill").fillna(0.0)
    vl_df[D1_FEAT_NAMES] = vl_df[D1_FEAT_NAMES].fillna(method="ffill").fillna(0.0)

    for f in D1_FEAT_NAMES:
        c = np.corrcoef(tr_df[f].fillna(0), tr_df["_fwd"] > 0)[0, 1]
        print(f"  {f}: target corr = {c:.4f}")

    X_tr = tr_df[all_fcols].astype("float32")
    y_tr = (tr_df["_fwd"] > 0).astype(int)
    X_vl = vl_df[all_fcols].astype("float32")
    y_vl = (vl_df["_fwd"] > 0).astype(int)

    print(f"\nBase (239 feats, seed=42)...")
    m_base = LGBMClassifier(
        n_estimators=600, num_leaves=255, learning_rate=0.02,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.5,
        min_child_samples=400, reg_lambda=20,
        n_jobs=-1, verbosity=-1, random_state=42,
    )
    m_base.fit(X_tr[FEATS], y_tr)
    p_base = m_base.predict_proba(X_vl[FEATS])[:, 1]
    auc_base = roc_auc_score(y_vl, p_base)
    print(f"  Base check: {auc_base:.4f}")

    print(f"\n+D1 features (243 feats, seed=42)...")
    m_d1 = LGBMClassifier(
        n_estimators=600, num_leaves=255, learning_rate=0.02,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.5,
        min_child_samples=400, reg_lambda=20,
        n_jobs=-1, verbosity=-1, random_state=42,
    )
    m_d1.fit(X_tr, y_tr)
    p_d1 = m_d1.predict_proba(X_vl)[:, 1]
    auc_d1 = roc_auc_score(y_vl, p_d1)

    imp = pd.Series(m_d1.feature_importances_, index=all_fcols)
    d1_imp = imp[D1_FEAT_NAMES].sort_values(ascending=False)
    d1_share = imp[D1_FEAT_NAMES].sum() / imp.sum()

    lift      = auc_d1 - BASE_VAL_AUC
    lift_chk  = auc_d1 - auc_base
    escalate  = auc_d1 > ESCALATE_THR

    print(f"\n=== D1 DAILY FEAT VAL RESULT ===")
    print(f"Base (stride-6 ref)            : {BASE_VAL_AUC:.4f}")
    print(f"Base check (this run)          : {auc_base:.4f}")
    print(f"Own-pair K=3 (full train)      : {OWN_K3_AUC:.4f}")
    print(f"+D1 features (243 feats)       : {auc_d1:.4f}")
    print(f"Lift vs ref                    : {lift:+.4f}")
    print(f"Lift vs base this run          : {lift_chk:+.4f}")
    print(f"D1 importance share            : {d1_share:.2%}")
    print(f"Escalate (>{ESCALATE_THR})     : {'YES' if escalate else 'NO → SUBSUMED'}")
    for f, v in d1_imp.items():
        print(f"  {f}: {v}")

    result = {
        "model": "NZDUSD 15m D1 daily trend features",
        "features": D1_FEAT_NAMES,
        "base_val_auc_ref": BASE_VAL_AUC,
        "base_val_auc_check": float(auc_base),
        "own_k3_val_auc": OWN_K3_AUC,
        "d1_feats_val_auc": float(auc_d1),
        "lift_vs_ref": float(lift),
        "lift_vs_check": float(lift_chk),
        "d1_importance_share": float(d1_share),
        "escalate": bool(escalate),
        "decision": "ESCALATE" if escalate else "SUBSUMED",
    }
    with open(RESULT_FILE, "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"\nSaved → {RESULT_FILE}")
