"""NZDUSD 15m — K=8 seed-ens VAL screen (own-pair NY).

Does K=8 seed-ens give meaningful AUC lift over K=3 on the NY VAL set?
USDCHF got unexpected K=8 lift on 340-feat XPAIR space. NZDUSD has 239-feat OWN-PAIR space.
AUC ceiling ~.535; K=3 VAL AUC=.5321. Expected: diminishing returns, <.001 AUC lift.

Falsifier: ESCALATE to full 15-path CPCV if K=8 VAL AUC > K=3 VAL AUC + 0.0015.
Otherwise SUBSUME K=8 CPCV (seed diversity saturated at K=3 for 239-feat space).
"""
import os, json, time, numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sessions import session_mask
import harness as H

TARGET = "NZDUSD"; HOR = 15; STEP = 60; GAP = HOR * STEP
FEAT_DIR = H.FEAT_DIR; FEATS = H.feature_cols(TARGET)
YEARS = list(range(2012, 2027))
TRAIN_YEARS = [str(y) for y in range(2012, 2022)]
VAL_YEARS = ["2022", "2023"]
RESULT = "nzdusd_15m_seedens8_screen_result.json"
NUM_LEAVES = 255
SEEDS_TO_TEST = [1, 2, 3, 5, 8]  # test cumulative K values


def build_years(years, stride=1):
    Xs, fwds, tss = [], [], []
    for y in years:
        p = f"{FEAT_DIR}/{TARGET}_{y}.parquet"
        if not os.path.exists(p):
            continue
        d = pd.read_parquet(p, columns=FEATS + ["close"])
        d = d[~d.index.duplicated(keep="last")]
        c = d["close"].values.astype(float)
        ts = d.index.values.astype("datetime64[s]").astype("int64")
        n = len(d)
        contig = np.zeros(n, bool)
        contig[:n - HOR] = (ts[HOR:] - ts[:-HOR]) == GAP
        fr = np.full(n, np.nan)
        fr[:n - HOR] = c[HOR:] / c[:-HOR] - 1.0
        X = d[FEATS].astype("float32")
        keepf = X.isna().mean(axis=1).values < 0.5
        valid = contig & np.isfinite(fr) & keepf
        moved = valid & (fr != 0.0)
        idx = np.where(valid)[0]
        if stride > 1:
            idx = idx[::stride]
        Xs.append(X.values[idx])
        fwds.append(fr[idx])
        tss.append(ts[idx])
    return np.concatenate(Xs), np.concatenate(fwds), np.concatenate(tss)


def main():
    t0 = time.time()
    print("[k8screen] Building train/val data ...", flush=True)
    Xtr, ftr, tstr = build_years(TRAIN_YEARS, stride=6)
    Xva, fva, tsva = build_years(VAL_YEARS, stride=1)

    ytr = (ftr > 0).astype(int)
    yva = (fva > 0).astype(int)
    nyva = session_mask(tsva, "ny")
    mva = fva != 0.0
    val_ny_moved = nyva & mva

    nytr = session_mask(tstr, "ny")
    mtr = ftr != 0.0
    train_ny_moved = nytr & mtr

    print(f"  Train NY moved: {train_ny_moved.sum():,} | Val NY moved: {val_ny_moved.sum():,}", flush=True)

    # Train K_MAX=8 models with different seeds
    K_MAX = 8
    models = []
    for k in range(K_MAX):
        print(f"[k8screen] Training seed {k} ...", flush=True)
        m = lgb.LGBMClassifier(
            objective="binary", metric="auc", learning_rate=0.02,
            num_leaves=NUM_LEAVES, min_child_samples=400,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.5,
            reg_lambda=20, n_estimators=2000, n_jobs=16, verbosity=-1,
            random_state=k, bagging_seed=k, feature_fraction_seed=k
        )
        m.fit(
            Xtr[train_ny_moved], ytr[train_ny_moved],
            eval_set=[(Xva[val_ny_moved], yva[val_ny_moved])],
            eval_metric="auc",
            callbacks=[lgb.early_stopping(150), lgb.log_evaluation(0)]
        )
        models.append(m)
        elapsed = time.time() - t0
        print(f"  seed {k}: iter={m.best_iteration_} ({elapsed:.0f}s)", flush=True)

    # Compute cumulative AUC at K=1,2,3,5,8
    probs_va = np.array([m.predict_proba(Xva)[:, 1] for m in models])  # (K_MAX, N_val)
    res = {"key": "NZDUSD.15m.ny.seedens8_screen", "incumbent_k3_auc": 0.5321,
           "falsifier": "ESCALATE if K=8 AUC > K=3 AUC + 0.0015; else SUBSUME",
           "k_aucs": {}}

    for k in [1, 2, 3, 5, 8]:
        ens_prob = probs_va[:k].mean(axis=0)
        auc = float(roc_auc_score(yva[val_ny_moved], ens_prob[val_ny_moved]))
        res["k_aucs"][f"K={k}"] = round(auc, 5)
        print(f"  K={k}: VAL NY AUC = {auc:.5f}", flush=True)

    k3_auc = res["k_aucs"]["K=3"]
    k8_auc = res["k_aucs"]["K=8"]
    lift = k8_auc - k3_auc
    escalate = lift > 0.0015
    res["verdict"] = {
        "k3_auc": k3_auc, "k8_auc": k8_auc, "lift_k3_to_k8": round(lift, 5),
        "ESCALATE_to_CPCV": escalate,
        "note": ("ESCALATE: K=8 gives meaningful lift — run full CPCV" if escalate
                 else f"SUBSUME: K=8 lift={lift:+.5f} < 0.0015 threshold — seed diversity saturated at K=3 for 239-feat own-pair space")
    }
    json.dump(res, open(RESULT, "w"), indent=2)
    print(f"\n[k8screen] K=3 AUC={k3_auc:.5f} | K=8 AUC={k8_auc:.5f} | lift={lift:+.5f} | ESCALATE={escalate} -> {RESULT}  {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
