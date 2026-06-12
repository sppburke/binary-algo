"""NZDUSD 15m — Realized signed-semivariance (D7) feature screen.

NOVEL_METHODS_RESEARCH §3 D7: Patton-Sheppard (2015) RS+/RS- confirmed GENUINE
directional content on 7-pair cross-sectional model (passes sign-flip surrogate null,
+.015 lift over surrogate at 15m). But D7 was tested on cross-sectional model, NOT
NZDUSD own-pair 239-feat model.

Feature gap confirmed: H.feature_cols('NZDUSD') = 239 features, ZERO RS+/RS- features.
Existing: 1m_rv_12, 5m_rv_12 etc (unsigned RV). Missing: RS_diff = RS+ - RS- (signed).

Mechanism: RS+ = upward 1m moves in window; RS- = downward 1m moves.
RS_diff > 0 → upside variance dominated → directional skew in recent price action.
GBM can learn sign × window → 15m continuation / reversal at multiple horizons.

D7 cross-sectional result: "+semivar does NOT add to base book (Δ15m -0154/-0029/-0107)"
HOWEVER: that was 7-pair pooled model with ~.56 base. For NZDUSD own-pair .5219 base,
the orthogonal RS_diff channel may behave differently.

Features added (9 new):
  Windows: 15, 30, 60 (1m bars lookahead via close price rolling)
  Per window: rs_pos (upside RV), rs_neg (downside RV), rs_diff (signed component)
Total: 239 + 9 = 248 features

Falsifier: VAL AUC > .5319 → ESCALATE to CPCV.
           VAL AUC > .5219 by any amount → document direction.
           VAL AUC < .5219 → confirms D7 cross-sectional result (dilutes own-pair).
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
RESULT_FILE = "nzdusd_15m_semivar_result.json"
TRAIN_YEARS  = [str(y) for y in range(2012, 2022)]
VAL_YEARS    = ["2022", "2023"]
TR_STRIDE    = 6
BASE_VAL_AUC = 0.5219
OWN_K3_AUC   = 0.5362
ESCALATE_THR = 0.5319
RS_WINDOWS   = [15, 30, 60]

FEATS = H.feature_cols(PAIR)


def compute_semivar_features(df, windows=RS_WINDOWS):
    """Compute RS+, RS-, RS_diff at multiple rolling windows from close price."""
    close = df["close"].values.astype(float)
    n = len(close)
    ret1m = np.zeros(n)
    ret1m[1:] = close[1:] / close[:-1] - 1.0
    r2 = ret1m ** 2
    r2_pos = r2 * (ret1m > 0)
    r2_neg = r2 * (ret1m < 0)

    new_cols = {}
    for w in windows:
        rs_pos = pd.Series(r2_pos).rolling(w, min_periods=max(1, w // 2)).mean().values
        rs_neg = pd.Series(r2_neg).rolling(w, min_periods=max(1, w // 2)).mean().values
        rs_diff = rs_pos - rs_neg
        new_cols[f"rs_pos_{w}"] = rs_pos
        new_cols[f"rs_neg_{w}"] = rs_neg
        new_cols[f"rs_diff_{w}"] = rs_diff
    return new_cols


def load_bars_with_semivar(years, stride=1, session="ny"):
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

        # Compute RS features on full year BEFORE striding
        sv = compute_semivar_features(d)
        for k, v in sv.items():
            d[k] = v

        idx = np.where(valid)[0]
        if stride > 1:
            idx = idx[::stride]
        sub = d.iloc[idx].copy()
        sub["_fwd"] = fr[idx]
        dfs.append(sub)
    return pd.concat(dfs).sort_index() if dfs else pd.DataFrame()


if __name__ == "__main__":
    print("=" * 70)
    print("NZDUSD 15m — Realized signed-semivariance (D7) own-pair screen")
    print(f"Escalate threshold: VAL AUC > {ESCALATE_THR:.4f}")
    print("=" * 70)

    rs_feat_names = [f"rs_{t}_{w}" for w in RS_WINDOWS for t in ["pos", "neg", "diff"]]
    all_fcols = FEATS + rs_feat_names
    print(f"\n[1] Loading bars + computing RS+/RS-/RS_diff at windows {RS_WINDOWS}...")
    train_yrs = TRAIN_YEARS[::TR_STRIDE]
    tr_df = load_bars_with_semivar(train_yrs, stride=1)
    vl_df = load_bars_with_semivar(VAL_YEARS, stride=1)
    print(f"  train: {len(tr_df):,} | val: {len(vl_df):,}")
    print(f"  Feature total: {len(FEATS)} base + {len(rs_feat_names)} RS = {len(all_fcols)}")

    # Fill RS NaNs with 0
    tr_df[rs_feat_names] = tr_df[rs_feat_names].fillna(0.0)
    vl_df[rs_feat_names] = vl_df[rs_feat_names].fillna(0.0)

    # Quick RS+/RS- sanity: confirm they are non-zero and distinct from RV
    for w in RS_WINDOWS:
        rv_key = f"{w}m_rv_12" if f"{w}m_rv_12" in FEATS else None
        rs_corr = np.corrcoef(tr_df[f"rs_diff_{w}"].fillna(0), tr_df["_fwd"] > 0)[0, 1] if len(tr_df) > 0 else 0
        print(f"  rs_diff_{w}: corr with target = {rs_corr:.4f}")

    X_tr = tr_df[all_fcols].astype("float32")
    y_tr = (tr_df["_fwd"] > 0).astype(int)
    X_vl = vl_df[all_fcols].astype("float32")
    y_vl = (vl_df["_fwd"] > 0).astype(int)

    print(f"\n[2] Base (239 feats, seed=42) for comparison...")
    m_base = LGBMClassifier(
        n_estimators=600, num_leaves=255, learning_rate=0.02,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.5,
        min_child_samples=400, reg_lambda=20,
        n_jobs=-1, verbosity=-1, random_state=42,
    )
    m_base.fit(X_tr[FEATS], y_tr)
    p_base = m_base.predict_proba(X_vl[FEATS])[:, 1]
    auc_base_check = roc_auc_score(y_vl, p_base)
    print(f"  Base check: {auc_base_check:.4f} (ref .5219)")

    print(f"\n[3] +RS features (248 feats, seed=42)...")
    m_rs = LGBMClassifier(
        n_estimators=600, num_leaves=255, learning_rate=0.02,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.5,
        min_child_samples=400, reg_lambda=20,
        n_jobs=-1, verbosity=-1, random_state=42,
    )
    m_rs.fit(X_tr, y_tr)
    p_rs = m_rs.predict_proba(X_vl)[:, 1]
    auc_rs = roc_auc_score(y_vl, p_rs)

    # Feature importance for RS features
    imp = pd.Series(m_rs.feature_importances_, index=all_fcols)
    rs_imp_share = imp[rs_feat_names].sum() / imp.sum()
    rs_top = imp[rs_feat_names].sort_values(ascending=False).head(5)

    lift      = auc_rs - BASE_VAL_AUC
    lift_chk  = auc_rs - auc_base_check
    escalate  = auc_rs > ESCALATE_THR

    print(f"\n=== SEMIVARIANCE D7 VAL RESULT ===")
    print(f"Base single-seed (stride-6 ref): {BASE_VAL_AUC:.4f}")
    print(f"Base check (this run)          : {auc_base_check:.4f}")
    print(f"Own-pair K=3 (full train)      : {OWN_K3_AUC:.4f}")
    print(f"+RS features (248 feats)       : {auc_rs:.4f}")
    print(f"Lift vs stride-6 ref           : {lift:+.4f}")
    print(f"Lift vs base this run          : {lift_chk:+.4f}")
    print(f"RS feature importance share    : {rs_imp_share:.2%}")
    print(f"Escalate (>{ESCALATE_THR})     : {'YES' if escalate else 'NO → SUBSUMED'}")
    print(f"\nTop RS features by importance:")
    for f, v in rs_top.items():
        print(f"  {f}: {v}")

    result = {
        "model": "NZDUSD.15m realized signed-semivariance (D7) own-pair screen",
        "novel_methods_ref": "NOVEL_METHODS_RESEARCH §3 D7 — Patton-Sheppard 2015 RS+/RS-",
        "rs_windows": RS_WINDOWS,
        "base_val_auc_ref": BASE_VAL_AUC,
        "base_val_auc_check": float(auc_base_check),
        "own_k3_val_auc": OWN_K3_AUC,
        "semivar_val_auc": float(auc_rs),
        "lift_vs_ref": float(lift),
        "lift_vs_check": float(lift_chk),
        "rs_importance_share": float(rs_imp_share),
        "escalate_threshold": ESCALATE_THR,
        "escalate": bool(escalate),
        "decision": "ESCALATE to CPCV" if escalate else "SUBSUMED",
        "features_total": len(all_fcols),
        "rs_features": len(rs_feat_names),
    }
    with open(RESULT_FILE, "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"\nSaved → {RESULT_FILE}")
