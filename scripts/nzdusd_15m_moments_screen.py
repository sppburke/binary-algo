"""NZDUSD 15m — rolling skewness/kurtosis + synthetic AUDNZD screen.

UNTESTED features identified after full sweep:
1. Rolling SKEWNESS of 1m returns (30/60/120/240 bar windows) — absent from 239 feats
2. Rolling KURTOSIS of 1m returns (same windows) — absent from 239 feats  
3. Synthetic AUDNZD close (= AUDUSD / NZDUSD) TA features — orthogonal to xpair pooling

Mechanism:
- Skew: negative rolling skew → occasional large UP moves dominate → potential continuation or reversion
- Kurtosis: fat-tailed recent returns → regime shift signal
- AUDNZD: AUD-NZD differential captures NZD-idiosyncratic moves; orthogonal to AUDUSD alone
"""
import os, sys, json
import numpy as np
import pandas as pd
sys.path.insert(0, "/home/sean/git/binary-algo")
import harness as H
from sessions import session_mask
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score

PAIR = "NZDUSD"; HOR = 15; STEP = 60; GAP = HOR * STEP
RESULT_FILE = "nzdusd_15m_moments_screen_result.json"
TRAIN_YEARS = [str(y) for y in range(2012, 2022)]
VAL_YEARS   = ["2022", "2023"]
TR_STRIDE   = 6
BASE_VAL_AUC = 0.5219
OWN_K3_AUC   = 0.5362
ESCALATE_THR = 0.5319
FEATS = H.feature_cols(PAIR)
WINDOWS = [30, 60, 120, 240]
MOMENT_NAMES = ([f"skew_{w}" for w in WINDOWS] +
                [f"kurt_{w}" for w in WINDOWS])

def compute_moment_features(close_1m, ts_1m):
    """Rolling skewness and kurtosis of 1m returns."""
    r = np.zeros(len(close_1m))
    r[1:] = close_1m[1:] / close_1m[:-1] - 1.0
    out = {}
    rs = pd.Series(r)
    for w in WINDOWS:
        out[f"skew_{w}"] = rs.rolling(w, min_periods=max(1, w//2)).skew().values
        out[f"kurt_{w}"] = rs.rolling(w, min_periods=max(1, w//2)).kurt().values
    return pd.DataFrame(out)

def compute_audnzd_features(nzdusd_close_1m, audusd_close_1m):
    """Synthetic AUDNZD = AUDUSD / NZDUSD — cross features."""
    audnzd = audusd_close_1m / nzdusd_close_1m
    r = np.zeros(len(audnzd)); r[1:] = audnzd[1:] / audnzd[:-1] - 1.0
    s = pd.Series(audnzd); rs = pd.Series(r)
    ema20 = s.ewm(span=20*15, adjust=False).mean()  # 20-bar 15m ≈ 300 1m bars
    rsi_w = 14 * 15
    up = rs.clip(lower=0); dn = (-rs).clip(lower=0)
    avg_up = up.rolling(rsi_w, min_periods=1).mean()
    avg_dn = dn.rolling(rsi_w, min_periods=1).mean()
    rsi = 100 - 100 / (1 + avg_up / (avg_dn + 1e-12))
    return pd.DataFrame({
        "audnzd_ret1":   r,
        "audnzd_ret15":  rs.rolling(15).sum().values,
        "audnzd_above_ema": (s >= ema20).astype(float).values,
        "audnzd_rsi":    rsi.values,
    })

AUDNZD_NAMES = ["audnzd_ret1", "audnzd_ret15", "audnzd_above_ema", "audnzd_rsi"]

def load_bars(years, stride=1, session="ny"):
    dfs = []
    for yr in years:
        pnzd = f"{H.FEAT_DIR}/NZDUSD_{yr}.parquet"
        paud = f"{H.FEAT_DIR}/AUDUSD_{yr}.parquet"
        if not os.path.exists(pnzd):
            continue
        d = pd.read_parquet(pnzd, columns=FEATS + ["close"])
        d = d[~d.index.duplicated(keep="last")]
        c = d["close"].values.astype(float)
        ts = d.index.values.astype("datetime64[s]").astype("int64"); n = len(d)
        contig = np.zeros(n, bool)
        contig[:n-HOR] = (ts[HOR:] - ts[:-HOR]) == GAP
        fr = np.full(n, np.nan)
        fr[:n-HOR] = c[HOR:] / c[:-HOR] - 1.0
        valid = contig & np.isfinite(fr) & (d[FEATS].isna().mean(axis=1).values < 0.5)
        smask = session_mask(ts, session)
        valid = valid & smask

        # Moment features (from 1m close)
        mf = compute_moment_features(c, ts)
        for col in MOMENT_NAMES:
            d[col] = mf[col].values if col in mf.columns else np.nan

        # AUDNZD synthetic features
        if os.path.exists(paud):
            daud = pd.read_parquet(paud, columns=["close"])
            daud = daud[~daud.index.duplicated(keep="last")]
            # Align to NZDUSD index
            aud_close = daud["close"].reindex(d.index, method="ffill").values.astype(float)
            af = compute_audnzd_features(c, aud_close)
            for col in AUDNZD_NAMES:
                d[col] = af[col].values if col in af.columns else np.nan
        else:
            for col in AUDNZD_NAMES:
                d[col] = np.nan

        idx = np.where(valid)[0]
        if stride > 1:
            idx = idx[::stride]
        sub = d.iloc[idx].copy()
        sub["_fwd"] = fr[idx]
        dfs.append(sub)
    return pd.concat(dfs).sort_index() if dfs else pd.DataFrame()

if __name__ == "__main__":
    print("=" * 70)
    print("NZDUSD 15m — rolling moments + synthetic AUDNZD screen")
    print("=" * 70)
    NEW_FEATS = MOMENT_NAMES + AUDNZD_NAMES
    ALL_FEATS = FEATS + NEW_FEATS
    print(f"Features: {len(FEATS)} base + {len(MOMENT_NAMES)} moments + {len(AUDNZD_NAMES)} AUDNZD = {len(ALL_FEATS)} total")

    tr_df = load_bars(TRAIN_YEARS[::TR_STRIDE], stride=1)
    vl_df = load_bars(VAL_YEARS, stride=1)
    print(f"train: {len(tr_df):,} | val: {len(vl_df):,}")

    for df in [tr_df, vl_df]:
        df[NEW_FEATS] = df[NEW_FEATS].ffill().fillna(0.0)

    # Raw correlations
    for f in NEW_FEATS:
        c = np.corrcoef(tr_df[f].fillna(0), tr_df["_fwd"] > 0)[0, 1]
        if abs(c) > 0.003:
            print(f"  {f}: target corr = {c:.6f}")

    X_tr = tr_df[ALL_FEATS].astype("float32")
    y_tr = (tr_df["_fwd"] > 0).astype(int)
    X_vl = vl_df[ALL_FEATS].astype("float32")
    y_vl = (vl_df["_fwd"] > 0).astype(int)

    # Base check
    m_base = LGBMClassifier(n_estimators=600, num_leaves=255, learning_rate=0.02,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.5,
        min_child_samples=400, reg_lambda=20, n_jobs=-1, verbosity=-1, random_state=42)
    m_base.fit(X_tr[FEATS], y_tr)
    auc_base = roc_auc_score(y_vl, m_base.predict_proba(X_vl[FEATS])[:, 1])

    # +All new features
    m_new = LGBMClassifier(n_estimators=600, num_leaves=255, learning_rate=0.02,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.5,
        min_child_samples=400, reg_lambda=20, n_jobs=-1, verbosity=-1, random_state=42)
    m_new.fit(X_tr, y_tr)
    auc_new = roc_auc_score(y_vl, m_new.predict_proba(X_vl)[:, 1])

    imp = pd.Series(m_new.feature_importances_, index=ALL_FEATS)
    new_share = imp[NEW_FEATS].sum() / imp.sum()
    top_new = imp[NEW_FEATS].sort_values(ascending=False).head(6)

    lift = auc_new - BASE_VAL_AUC
    lift_chk = auc_new - auc_base
    escalate = auc_new > ESCALATE_THR

    print(f"\n=== RESULT ===")
    print(f"Base ref        : {BASE_VAL_AUC:.4f}")
    print(f"Base check      : {auc_base:.4f}")
    print(f"Own K=3 ref     : {OWN_K3_AUC:.4f}")
    print(f"+Moments+AUDNZD : {auc_new:.4f}  (lift vs ref: {lift:+.4f}, vs check: {lift_chk:+.4f})")
    print(f"New feat share  : {new_share:.2%}")
    print(f"Escalate        : {'YES' if escalate else 'NO → SUBSUMED'}")
    print("Top new features:")
    for f, v in top_new.items():
        print(f"  {f}: {v}")

    result = {
        "model": "NZDUSD 15m moments+AUDNZD screen",
        "new_features": NEW_FEATS,
        "base_val_auc_ref": BASE_VAL_AUC,
        "base_val_auc_check": float(auc_base),
        "own_k3_val_auc": OWN_K3_AUC,
        "moments_audnzd_val_auc": float(auc_new),
        "lift_vs_ref": float(lift),
        "lift_vs_check": float(lift_chk),
        "new_feat_share": float(new_share),
        "escalate": bool(escalate),
        "decision": "ESCALATE" if escalate else "SUBSUMED",
        "top_new_features": {str(f): int(v) for f, v in top_new.items()},
    }
    with open(RESULT_FILE, "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"\nSaved → {RESULT_FILE}")
