"""NZDUSD 15m — USDCAD cross-pair features VAL screen.

Backlog item: USDCAD xpair (corr~.55 vs AUDUSD .88 which ERA-STRUCTURALLY failed UP).
If VAL AUC > base (.5219) + .010 = .5319 → ESCALATE to full refit-CPCV.
Otherwise SUBSUME — commodity xpair below AUDUSD signal ceiling.

Features: 239 NZDUSD base + 239 USDCAD cross-pair (prefixed xp_) = 478 total.
Merge: left join NZDUSD timestamps × USDCAD timestamps; missing USDCAD bars → fill 0.
Train: 2012 + 2018 (stride=6); Val: 2022–2023 (full); NY session only.
"""
import os, sys, json
import numpy as np
import pandas as pd
sys.path.insert(0, "/home/sean/git/binary-algo")
import harness as H
from sessions import session_mask
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score

PAIR_BASE   = "NZDUSD"
PAIR_XPAIR  = "USDCAD"
HOR = 15; STEP = 60; GAP = HOR * STEP
RESULT_FILE = "nzdusd_15m_usdcad_xpair_screen_result.json"
TRAIN_YEARS = [str(y) for y in range(2012, 2022)]
VAL_YEARS   = ["2022", "2023"]
TR_STRIDE   = 6          # 2012, 2018
NUM_LEAVES  = 255
BASE_VAL_AUC = 0.5219
ESCALATE_THR = BASE_VAL_AUC + 0.010   # 0.5319

FEATS = H.feature_cols(PAIR_BASE)   # same list for both pairs


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
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs).sort_index()


def load_xpair_feats(pair, years, session="ny"):
    """Load cross-pair features for ALL years (not strided) — we left-join later."""
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
    xp.columns = [f"xp_{c}" for c in xp.columns]
    return xp


def merge_xpair(base_df, xp_df):
    """Left join base NZDUSD bars with USDCAD cross-pair features on exact timestamp."""
    merged = base_df.join(xp_df, how="left")
    xp_cols = [c for c in merged.columns if c.startswith("xp_")]
    merged[xp_cols] = merged[xp_cols].fillna(0.0)
    return merged


if __name__ == "__main__":
    print("=" * 70)
    print("NZDUSD 15m — USDCAD cross-pair features VAL screen")
    print(f"Base: {PAIR_BASE} (239 feats) + XPair: {PAIR_XPAIR} (239 feats) = 478 total")
    print(f"Falsifier: VAL AUC > {ESCALATE_THR:.4f} → ESCALATE; else SUBSUME")
    print("=" * 70)

    train_yrs = TRAIN_YEARS[::TR_STRIDE]
    print(f"\n[1] Loading NZDUSD base bars ({train_yrs} train, {VAL_YEARS} val)...")
    tr_base = load_bars(PAIR_BASE, train_yrs, stride=1, session="ny")
    vl_base = load_bars(PAIR_BASE, VAL_YEARS,  stride=1, session="ny")
    print(f"  NZDUSD train: {len(tr_base):,} bars | val: {len(vl_base):,} bars")

    print(f"\n[2] Loading USDCAD cross-pair features (all years)...")
    all_years = TRAIN_YEARS + VAL_YEARS
    xp_all    = load_xpair_feats(PAIR_XPAIR, all_years, session="ny")
    print(f"  USDCAD xpair features loaded: {len(xp_all):,} bars, {len(xp_all.columns)} cols")

    print(f"\n[3] Merging xpair features into base bars...")
    tr_merged = merge_xpair(tr_base, xp_all)
    vl_merged = merge_xpair(vl_base, xp_all)

    xp_cols   = [c for c in tr_merged.columns if c.startswith("xp_")]
    all_fcols = FEATS + xp_cols
    all_fcols = [c for c in all_fcols if c in tr_merged.columns]

    xp_hit_tr = (tr_merged[xp_cols] != 0).any(axis=1).mean()
    xp_hit_vl = (vl_merged[xp_cols] != 0).any(axis=1).mean()
    print(f"  Xpair match rate — train: {xp_hit_tr:.3f}, val: {xp_hit_vl:.3f}")
    print(f"  Feature total: {len(FEATS)} base + {len(xp_cols)} xpair = {len(all_fcols)}")

    X_tr = tr_merged[all_fcols].fillna(0).astype("float32")
    y_tr = (tr_merged["_fwd"] > 0).astype(int)
    X_vl = vl_merged[all_fcols].fillna(0).astype("float32")
    y_vl = (vl_merged["_fwd"] > 0).astype(int)

    print(f"\n[4] Training GBM (leaves={NUM_LEAVES}, {len(all_fcols)} feats)...")
    model = LGBMClassifier(
        n_estimators=600, num_leaves=NUM_LEAVES, learning_rate=0.02,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.5,
        min_child_samples=400, reg_lambda=20,
        n_jobs=-1, verbosity=-1, random_state=42,
    )
    model.fit(X_tr, y_tr)

    prob_vl  = model.predict_proba(X_vl)[:, 1]
    val_auc  = roc_auc_score(y_vl, prob_vl)
    lift     = val_auc - BASE_VAL_AUC
    escalate = val_auc > ESCALATE_THR

    print(f"\n[5] Results:")
    print(f"  Base VAL AUC : {BASE_VAL_AUC:.4f}")
    print(f"  XPair VAL AUC: {val_auc:.4f}")
    print(f"  Lift         : {lift:+.4f}")
    print(f"  Threshold    : {ESCALATE_THR:.4f}")
    print(f"  Decision     : {'ESCALATE to CPCV' if escalate else 'SUBSUME — lift < .010'}")

    imp     = dict(zip(all_fcols, model.feature_importances_))
    xp_imp  = sorted([(k, v) for k, v in imp.items() if k.startswith("xp_")], key=lambda x: -x[1])
    print(f"\n  Top-15 USDCAD cross-pair importances:")
    for k, v in xp_imp[:15]:
        print(f"    {k}: {v}")

    result = {
        "model": "NZDUSD.15m USDCAD-xpair VAL screen",
        "key": "NZDUSD.15m",
        "base_pair": PAIR_BASE,
        "xpair": PAIR_XPAIR,
        "corr_note": "NZDUSD/USDCAD corr~.55 (weaker than AUDUSD .88 which ERA-STRUCTURALLY failed UP)",
        "base_val_auc": BASE_VAL_AUC,
        "xpair_val_auc": float(val_auc),
        "lift": float(lift),
        "escalate_threshold": ESCALATE_THR,
        "escalate": bool(escalate),
        "decision": "ESCALATE to CPCV" if escalate else "SUBSUME — lift < .010",
        "xpair_match_rate_train": float(xp_hit_tr),
        "xpair_match_rate_val": float(xp_hit_vl),
        "features_base": len(FEATS),
        "features_xpair": len(xp_cols),
        "features_total": len(all_fcols),
        "train_years": train_yrs,
        "val_years": VAL_YEARS,
        "top_xp_importances": [(k, int(v)) for k, v in xp_imp[:15]],
        "falsifier": "escalate if xpair_val_auc > base_val_auc + .010",
    }
    with open(RESULT_FILE, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Saved: {RESULT_FILE}")
    print("=" * 70)
