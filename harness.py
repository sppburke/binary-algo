"""
Split + evaluation harness for 5-min binary direction.

Splits (time-ordered, no shuffling):
    TRAIN 2012-2021 | VAL 2022-2023 | TEST 2024-2025 | OOS 2026 (held out)

Core metric: accuracy@coverage (selective prediction). Confidence = |p - 0.5|.
We choose a confidence threshold on VAL to hit a target accuracy, then report the
coverage + accuracy that threshold yields on TEST and OOS.
"""
import os, glob, json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

FEAT_DIR = "/media/sean/CORSAIR/binary-algo/features"
META_COLS = ["y", "fwd_ret", "valid", "close"]

SPLITS = {
    "train": [str(y) for y in range(2012, 2022)],   # 2012-2021
    "val":   ["2022", "2023"],
    "test":  ["2024", "2025"],
    "oos":   ["2026"],
}


def feature_cols(pair="EURUSD"):
    f = sorted(glob.glob(os.path.join(FEAT_DIR, f"{pair}_*.parquet")))[0]
    cols = list(pd.read_parquet(f).columns)
    return [c for c in cols if c not in META_COLS]


def load_split(pair, split, stride=1, cols=None, dropna_thresh=0.5):
    """Load a named split. stride>1 subsamples rows (decorrelates overlapping labels,
    saves memory) — applied to the *valid* rows only. Returns X, y, aux(df)."""
    years = SPLITS[split]
    feats = feature_cols(pair) if cols is None else cols
    usecols = feats + META_COLS
    parts = []
    for y in years:
        p = os.path.join(FEAT_DIR, f"{pair}_{y}.parquet")
        if os.path.exists(p):
            parts.append(pd.read_parquet(p, columns=usecols))
    df = pd.concat(parts)
    df = df[df["valid"] == True]
    if stride > 1:
        df = df.iloc[::stride]
    X = df[feats].astype("float32")
    # rows with too many NaNs dropped; remaining NaNs -> LightGBM handles natively
    keep = X.isna().mean(axis=1) < dropna_thresh
    X = X[keep]
    aux = df.loc[keep.index[keep], ["y", "fwd_ret", "close"]].copy()
    y = aux["y"].astype(int).values
    return X, y, aux


# --------------------------- selective metrics ---------------------------

def accuracy_at_coverages(y_true, p, coverages=(1.0, 0.5, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01)):
    """For each target coverage, take the most-confident fraction and report accuracy."""
    conf = np.abs(p - 0.5)
    order = np.argsort(-conf)
    yt = np.asarray(y_true)[order]
    pred = (p[order] > 0.5).astype(int)
    correct = (pred == yt).astype(float)
    n = len(yt)
    rows = []
    for cov in coverages:
        k = max(1, int(round(cov * n)))
        acc = correct[:k].mean()
        thr = conf[order][k-1]
        rows.append({"coverage": cov, "n": k, "accuracy": acc, "conf_thr": float(thr)})
    return pd.DataFrame(rows)


def threshold_for_target(y_true, p, target=0.75, min_n=200):
    """Find the smallest confidence threshold (=> largest coverage) on this set whose
    selected subset reaches >= target accuracy with at least min_n bets."""
    conf = np.abs(p - 0.5)
    order = np.argsort(-conf)
    yt = np.asarray(y_true)[order]
    pred = (p[order] > 0.5).astype(int)
    correct = (pred == yt).astype(float)
    cum_acc = np.cumsum(correct) / np.arange(1, len(yt)+1)
    n = len(yt)
    best = None
    for k in range(min_n, n+1):
        if cum_acc[k-1] >= target:
            best = {"k": k, "coverage": k/n, "accuracy": cum_acc[k-1],
                    "conf_thr": float(conf[order][k-1])}
    return best  # largest k meeting target (None if never)


def apply_threshold(y_true, p, conf_thr):
    conf = np.abs(p - 0.5)
    sel = conf >= conf_thr
    if sel.sum() == 0:
        return {"coverage": 0.0, "accuracy": float("nan"), "n": 0}
    pred = (p[sel] > 0.5).astype(int)
    acc = (pred == np.asarray(y_true)[sel]).mean()
    return {"coverage": float(sel.mean()), "accuracy": float(acc), "n": int(sel.sum())}


def report(name, y_true, p):
    auc = roc_auc_score(y_true, p)
    acc = ((p > 0.5).astype(int) == y_true).mean()
    print(f"[{name}] n={len(y_true)} AUC={auc:.4f} acc@full={acc:.4f}")
    tbl = accuracy_at_coverages(y_true, p)
    print(tbl.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    return {"auc": float(auc), "acc_full": float(acc), "cov_table": tbl}
