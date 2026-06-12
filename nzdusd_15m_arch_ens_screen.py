"""NZDUSD 15m — Architectural diversity ensemble screen.

Hypothesis: seed diversity (K=3, same num_leaves=255) is one form of ensemble diversity.
ARCHITECTURAL diversity (different num_leaves = different bias-variance tradeoffs) may
add orthogonal signal not captured by seed diversity alone.

num_leaves=127: shallower trees → higher bias, lower variance → finds simple robust patterns
num_leaves=255: standard incumbent
num_leaves=511: deeper trees → lower bias, higher variance → finds complex regime patterns

K=3 seeds × 3 architectures = 9 models. Ensemble = mean probability.

Mechanism: quiet/trending regimes → shallow wins; volatile/complex regimes → deep wins.
The mean of all 9 captures regime-robust patterns that any single architecture misses.

Prior evidence: GBPUSD K=8 gave UNEXPECTED LIFT in 340-feat space (not saturated at K=3).
NZDUSD 239-feat space: K=3 seed diversity alone → VAL .5362. Architectural diversity adds
a second dimension of diversification beyond seed randomness.

Falsifier: VAL AUC > .5362 (own K=3 incumbent) → genuine lift.
           VAL AUC > .5319 → ESCALATE to CPCV.
           Both: report full K=9 ensemble VAL AUC + per-architecture VAL AUC.
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
RESULT_FILE = "nzdusd_15m_arch_ens_result.json"
TRAIN_YEARS  = [str(y) for y in range(2012, 2022)]
VAL_YEARS    = ["2022", "2023"]
TR_STRIDE    = 6
BASE_VAL_AUC = 0.5219
OWN_K3_AUC   = 0.5362
ESCALATE_THR = 0.5319

SEEDS       = [42, 0, 7]
LEAF_VALUES = [127, 255, 511]   # shallower / standard / deeper

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
    print("NZDUSD 15m — Architectural diversity ensemble (127/255/511 leaves × K=3 seeds)")
    print(f"Falsifier (incumbent): VAL AUC > {OWN_K3_AUC:.4f}")
    print(f"Escalate threshold  : VAL AUC > {ESCALATE_THR:.4f}")
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

    print(f"\n[2] Fitting {len(LEAF_VALUES)} × {len(SEEDS)} = {len(LEAF_VALUES)*len(SEEDS)} models...")
    all_probs = []
    arch_results = {}

    for nl in LEAF_VALUES:
        arch_probs = []
        for s in SEEDS:
            m = LGBMClassifier(
                n_estimators=600, num_leaves=nl, learning_rate=0.02,
                subsample=0.8, subsample_freq=1, colsample_bytree=0.5,
                min_child_samples=400, reg_lambda=20,
                n_jobs=-1, verbosity=-1, random_state=s,
            )
            m.fit(X_tr, y_tr)
            p = m.predict_proba(X_vl)[:, 1]
            auc_s = roc_auc_score(y_vl, p)
            print(f"  num_leaves={nl} seed={s}: VAL AUC={auc_s:.4f}")
            arch_probs.append(p)
            all_probs.append(p)

        p_arch = np.mean(arch_probs, axis=0)
        auc_arch = roc_auc_score(y_vl, p_arch)
        arch_results[nl] = float(auc_arch)
        print(f"  → num_leaves={nl} K=3 ensemble AUC: {auc_arch:.4f}")

    print(f"\n[3] Full K=9 ensemble...")
    p_full = np.mean(all_probs, axis=0)
    full_auc  = roc_auc_score(y_vl, p_full)
    lift      = full_auc - BASE_VAL_AUC
    lift_k3   = full_auc - OWN_K3_AUC
    escalate  = full_auc > ESCALATE_THR

    print(f"\n=== ARCHITECTURAL ENSEMBLE VAL RESULT ===")
    print(f"Base single-seed         : {BASE_VAL_AUC:.4f}")
    print(f"Own-pair K=3 (incumbent) : {OWN_K3_AUC:.4f}")
    for nl, auc in arch_results.items():
        tag = " ← standard" if nl == 255 else ""
        print(f"Arch K=3 (leaves={nl:3d})  : {auc:.4f}{tag}")
    print(f"Full arch-ens K=9        : {full_auc:.4f}")
    print(f"Lift vs base             : {lift:+.4f}")
    print(f"Lift vs own K=3          : {lift_k3:+.4f}")
    print(f"Escalate threshold       : {ESCALATE_THR:.4f} → {'ESCALATE to CPCV' if escalate else 'SUBSUMED'}")
    if lift_k3 > 0:
        print(f"*** SUPERSEDES incumbent by {lift_k3:+.4f} ***")

    result = {
        "model": "NZDUSD.15m architectural ensemble (127/255/511 leaves × K=3 seeds = 9 models)",
        "seeds": SEEDS,
        "leaf_values": LEAF_VALUES,
        "total_models": len(SEEDS) * len(LEAF_VALUES),
        "base_val_auc": BASE_VAL_AUC,
        "own_k3_val_auc": OWN_K3_AUC,
        "per_arch_val_auc": arch_results,
        "full_arch_ens_val_auc": float(full_auc),
        "lift_vs_base": float(lift),
        "lift_vs_own_k3": float(lift_k3),
        "escalate_threshold": ESCALATE_THR,
        "escalate": bool(escalate),
        "decision": "ESCALATE to CPCV" if escalate else "SUBSUMED",
        "features": len(FEATS),
    }
    with open(RESULT_FILE, "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"\nSaved → {RESULT_FILE}")
