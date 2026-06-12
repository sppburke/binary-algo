"""NZDUSD 15m — USDCAD xpair + seed-ens K=3 VAL screen.

Single-seed USDCAD xpair gave VAL AUC .5299 (below .5319 threshold).
Own-pair seed-ens K=3 lifted VAL by +.014 (.5219→.5362).
If similar lift applies: .5299 + .014 = .5439 → above .5319 CPCV threshold.

AUDUSD xpair was ERA-STRUCTURAL for UP (g[4,5] path failed under K=3).
USDCAD corr~.55 (weaker than AUDUSD .88) may be more regime-stable.

Falsifier: VAL AUC > .5362 (current K=3 own-pair) → genuine improvement;
           VAL AUC > .5319 → escalate to full CPCV.
"""
import os, sys, json
import numpy as np
import pandas as pd
sys.path.insert(0, "/home/sean/git/binary-algo")
import harness as H
from sessions import session_mask
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score

PAIR_BASE  = "NZDUSD"
PAIR_XPAIR = "USDCAD"
HOR = 15; STEP = 60; GAP = HOR * STEP
RESULT_FILE = "nzdusd_15m_usdcad_xpair_seedens_result.json"

TRAIN_YEARS  = [str(y) for y in range(2012, 2022)]
VAL_YEARS    = ["2022", "2023"]
TR_STRIDE    = 6
NUM_LEAVES   = 255
BASE_VAL_AUC = 0.5219
OWN_K3_AUC   = 0.5362   # K=3 own-pair ceiling to beat
ESCALATE_THR = 0.5319   # base + .010
SEEDS        = [42, 0, 7]

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


def load_xpair_feats(pair, years, session="ny"):
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
    merged = base_df.join(xp_df, how="left")
    xp_cols = [c for c in merged.columns if c.startswith("xp_")]
    merged[xp_cols] = merged[xp_cols].fillna(0.0)
    return merged


if __name__ == "__main__":
    print("=" * 70)
    print("NZDUSD 15m — USDCAD xpair + seed-ens K=3 VAL screen")
    print(f"Falsifier: VAL AUC > {OWN_K3_AUC:.4f} (own K=3) → genuine lift")
    print(f"           VAL AUC > {ESCALATE_THR:.4f} → ESCALATE to CPCV")
    print("=" * 70)

    train_yrs = TRAIN_YEARS[::TR_STRIDE]
    print(f"\n[1] Loading bars...")
    tr_base = load_bars(PAIR_BASE, train_yrs, stride=1, session="ny")
    vl_base = load_bars(PAIR_BASE, VAL_YEARS,  stride=1, session="ny")
    print(f"  NZDUSD train: {len(tr_base):,} | val: {len(vl_base):,}")

    all_years = TRAIN_YEARS + VAL_YEARS
    xp_all    = load_xpair_feats(PAIR_XPAIR, all_years, session="ny")
    print(f"  USDCAD xpair: {len(xp_all):,} bars, {len(xp_all.columns)} cols")

    tr_merged = merge_xpair(tr_base, xp_all)
    vl_merged = merge_xpair(vl_base, xp_all)
    xp_cols   = [c for c in tr_merged.columns if c.startswith("xp_")]
    all_fcols = [c for c in FEATS + xp_cols if c in tr_merged.columns]
    print(f"  Feature total: {len(FEATS)} base + {len(xp_cols)} xpair = {len(all_fcols)}")
    print(f"  Xpair match: train={( tr_merged[xp_cols] != 0).any(axis=1).mean():.3f} val={(vl_merged[xp_cols] != 0).any(axis=1).mean():.3f}")

    X_tr = tr_merged[all_fcols].fillna(0).astype("float32")
    y_tr = (tr_merged["_fwd"] > 0).astype(int)
    X_vl = vl_merged[all_fcols].fillna(0).astype("float32")
    y_vl = (vl_merged["_fwd"] > 0).astype(int)

    print(f"\n[2] Fitting K={len(SEEDS)} seeds...")
    probs = []
    for s in SEEDS:
        m = LGBMClassifier(
            n_estimators=600, num_leaves=NUM_LEAVES, learning_rate=0.02,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.5,
            min_child_samples=400, reg_lambda=20,
            n_jobs=-1, verbosity=-1, random_state=s,
        )
        m.fit(X_tr, y_tr)
        p = m.predict_proba(X_vl)[:, 1]
        auc_s = roc_auc_score(y_vl, p)
        print(f"  seed={s}: VAL AUC={auc_s:.4f}")
        probs.append(p)

    p_ens = np.mean(probs, axis=0)
    val_auc  = roc_auc_score(y_vl, p_ens)
    lift     = val_auc - BASE_VAL_AUC
    lift_vs_own_k3 = val_auc - OWN_K3_AUC
    escalate = val_auc > ESCALATE_THR

    print(f"\n=== USDCAD XPAIR SEED-ENS K=3 VAL RESULT ===")
    print(f"Base VAL AUC (own single-seed) : {BASE_VAL_AUC:.4f}")
    print(f"Own-pair K=3 VAL AUC           : {OWN_K3_AUC:.4f}")
    print(f"USDCAD xpair K=3 VAL AUC       : {val_auc:.4f}")
    print(f"Lift vs base                   : {lift:+.4f}")
    print(f"Lift vs own K=3                : {lift_vs_own_k3:+.4f}")
    print(f"CPCV threshold (.5319)         : {'ESCALATE' if escalate else 'BELOW — SUBSUMED'}")

    result = {
        "model": "NZDUSD.15m USDCAD-xpair seed-ens K=3 VAL screen",
        "seeds": SEEDS,
        "base_val_auc": BASE_VAL_AUC,
        "own_k3_val_auc": OWN_K3_AUC,
        "xpair_seedens_val_auc": float(val_auc),
        "lift_vs_base": float(lift),
        "lift_vs_own_k3": float(lift_vs_own_k3),
        "escalate_threshold": ESCALATE_THR,
        "escalate": bool(escalate),
        "decision": "ESCALATE to CPCV" if escalate else "SUBSUMED — below .5319",
        "features_total": len(all_fcols),
    }
    with open(RESULT_FILE, "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"\nSaved → {RESULT_FILE}")
