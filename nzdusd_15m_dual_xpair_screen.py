"""NZDUSD 15m — Dual-xpair VAL screen: AUDUSD + USDCAD features combined.

AUDUSD xpair: Pacific/Asia/risk-on factor (corr~.88, A6 CPCV AUC .5345)
USDCAD xpair: Oil/North America commodity factor (corr~.55, VAL .5299)
Mechanism: NZD sensitive to BOTH Asia commodity AND oil/North America drivers.
AUDUSD captures the former; USDCAD captures the latter — partially orthogonal.

Falsifier: VAL AUC > .5319 → ESCALATE to CPCV.
Expected: dilution similar to MEGA-COMBO (more features → noise); but
these xpairs have genuine signal unlike F6/F8/F9 external noise.

Feature total: 239 NZDUSD + 239 AUDUSD (xpa_*) + 239 USDCAD (xpu_*) = 717
"""
import os, sys, json
import numpy as np
import pandas as pd
sys.path.insert(0, "/home/sean/git/binary-algo")
import harness as H
from sessions import session_mask
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score

PAIR_BASE = "NZDUSD"
HOR = 15; STEP = 60; GAP = HOR * STEP
RESULT_FILE = "nzdusd_15m_dual_xpair_result.json"
TRAIN_YEARS  = [str(y) for y in range(2012, 2022)]
VAL_YEARS    = ["2022", "2023"]
TR_STRIDE    = 6
NUM_LEAVES   = 255
BASE_VAL_AUC = 0.5219
OWN_K3_AUC   = 0.5362
ESCALATE_THR = 0.5319
SEEDS        = [42]  # single seed — dual xpair K=3 ensemble shown to hurt

FEATS = H.feature_cols(PAIR_BASE)


def load_bars(pair, years, stride=1, session="ny"):
    dfs = []
    for yr in years:
        p = f"{H.FEAT_DIR}/{pair}_{yr}.parquet"
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


def load_xpair_feats(pair, prefix, years, session="ny"):
    dfs = []
    for yr in years:
        p = f"{H.FEAT_DIR}/{pair}_{yr}.parquet"
        if not os.path.exists(p):
            continue
        d = pd.read_parquet(p, columns=FEATS)
        d = d[~d.index.duplicated(keep="last")]
        ts = d.index.values.astype("datetime64[s]").astype("int64")
        smask = session_mask(ts, session)
        dfs.append(d[smask])
    if not dfs:
        return pd.DataFrame()
    xp = pd.concat(dfs).sort_index()
    xp.columns = [f"{prefix}{c}" for c in xp.columns]
    return xp


if __name__ == "__main__":
    print("=" * 70)
    print("NZDUSD 15m — Dual-xpair VAL screen (AUDUSD + USDCAD features)")
    print(f"Falsifier: VAL AUC > {ESCALATE_THR:.4f} → ESCALATE")
    print("=" * 70)

    train_yrs = TRAIN_YEARS[::TR_STRIDE]
    all_years = TRAIN_YEARS + VAL_YEARS

    print(f"\n[1] Loading NZDUSD base bars...")
    tr_base = load_bars(PAIR_BASE, train_yrs, stride=1)
    vl_base = load_bars(PAIR_BASE, VAL_YEARS,  stride=1)
    print(f"  train: {len(tr_base):,} | val: {len(vl_base):,}")

    print(f"\n[2] Loading AUDUSD xpair (xpa_) and USDCAD xpair (xpu_)...")
    xpa = load_xpair_feats("AUDUSD", "xpa_", all_years)
    xpu = load_xpair_feats("USDCAD", "xpu_", all_years)
    print(f"  AUDUSD: {len(xpa):,} bars | USDCAD: {len(xpu):,} bars")

    print(f"\n[3] Merging...")
    tr_merged = tr_base.join(xpa, how="left").join(xpu, how="left")
    vl_merged = vl_base.join(xpa, how="left").join(xpu, how="left")

    xpa_cols = [c for c in tr_merged.columns if c.startswith("xpa_")]
    xpu_cols = [c for c in tr_merged.columns if c.startswith("xpu_")]
    all_fcols = [c for c in FEATS + xpa_cols + xpu_cols if c in tr_merged.columns]

    # Fill xpair NaNs
    tr_merged[xpa_cols + xpu_cols] = tr_merged[xpa_cols + xpu_cols].fillna(0.0)
    vl_merged[xpa_cols + xpu_cols] = vl_merged[xpa_cols + xpu_cols].fillna(0.0)

    print(f"  Feature total: {len(FEATS)} NZDUSD + {len(xpa_cols)} AUDUSD + {len(xpu_cols)} USDCAD = {len(all_fcols)}")
    print(f"  AUD match: train={(tr_merged[xpa_cols]!=0).any(axis=1).mean():.3f} val={(vl_merged[xpa_cols]!=0).any(axis=1).mean():.3f}")
    print(f"  CAD match: train={(tr_merged[xpu_cols]!=0).any(axis=1).mean():.3f} val={(vl_merged[xpu_cols]!=0).any(axis=1).mean():.3f}")

    X_tr = tr_merged[all_fcols].astype("float32")
    y_tr = (tr_merged["_fwd"] > 0).astype(int)
    X_vl = vl_merged[all_fcols].astype("float32")
    y_vl = (vl_merged["_fwd"] > 0).astype(int)

    print(f"\n[4] Fitting single-seed GBM (seed=42)...")
    m = LGBMClassifier(
        n_estimators=600, num_leaves=NUM_LEAVES, learning_rate=0.02,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.5,
        min_child_samples=400, reg_lambda=20,
        n_jobs=-1, verbosity=-1, random_state=42,
    )
    m.fit(X_tr, y_tr)
    p_vl  = m.predict_proba(X_vl)[:, 1]
    val_auc = roc_auc_score(y_vl, p_vl)
    lift    = val_auc - BASE_VAL_AUC
    escalate = val_auc > ESCALATE_THR

    # Top importances breakdown
    imp = pd.Series(m.feature_importances_, index=all_fcols)
    nz_imp = imp[[c for c in all_fcols if not c.startswith("xp")]].sum()
    au_imp = imp[xpa_cols].sum()
    ca_imp = imp[xpu_cols].sum()
    total_imp = imp.sum()

    print(f"\n=== DUAL XPAIR VAL RESULT ===")
    print(f"Base single-seed   : {BASE_VAL_AUC:.4f}")
    print(f"Own-pair K=3       : {OWN_K3_AUC:.4f}")
    print(f"Dual xpair AUC     : {val_auc:.4f}")
    print(f"Lift vs base       : {lift:+.4f}")
    print(f"CPCV threshold     : {ESCALATE_THR:.4f} → {'ESCALATE' if escalate else 'SUBSUMED'}")
    print(f"\nFeature group importance share:")
    print(f"  NZDUSD base:    {nz_imp/total_imp:.2%}")
    print(f"  AUDUSD xpair:   {au_imp/total_imp:.2%}")
    print(f"  USDCAD xpair:   {ca_imp/total_imp:.2%}")
    print(f"\nTop-10 features:")
    for feat, val in imp.sort_values(ascending=False).head(10).items():
        print(f"  {feat}: {val}")

    result = {
        "model": "NZDUSD.15m dual-xpair (AUDUSD+USDCAD) single-seed VAL screen",
        "base_val_auc": BASE_VAL_AUC,
        "own_k3_val_auc": OWN_K3_AUC,
        "dual_xpair_val_auc": float(val_auc),
        "lift_vs_base": float(lift),
        "escalate_threshold": ESCALATE_THR,
        "escalate": bool(escalate),
        "decision": "ESCALATE to CPCV" if escalate else "SUBSUMED",
        "features_base": len(FEATS),
        "features_audusd": len(xpa_cols),
        "features_usdcad": len(xpu_cols),
        "features_total": len(all_fcols),
        "imp_share_nzdusd": float(nz_imp/total_imp),
        "imp_share_audusd": float(au_imp/total_imp),
        "imp_share_usdcad": float(ca_imp/total_imp),
    }
    with open(RESULT_FILE, "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"\nSaved → {RESULT_FILE}")
