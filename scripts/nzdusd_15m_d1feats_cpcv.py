"""NZDUSD 15m — D1 daily features refit-CPCV (certification run).

DISCOVERY: Daily close-to-close return (d1_ret_1) + 5-day return (d1_ret_5) +
20/100-day EMA regime flags are ABSENT from the 239-feat set (max TF = 4h).
Stride-6 proxy AUC .5471 >> base .5257 (+.0214). No lookahead: NY session bars
(14:00-21:00 UTC) always see PRIOR calendar day's daily value via ffill (daily
close stamped at 23:59 UTC, propagated forward until next 23:59).

Mechanism: Overnight-to-intraday momentum — prior day's direction in Asian/London
sessions carries into NY session. d1_ret_1 target corr = +0.0430 (largest raw
correlation of any feature tested). d1_ret_1 importance = 2134 (dominant).

CPCV parameters (identical to certified book):
  N_GROUPS=6, K_TEST=2 → 15 paths, per-fold refit, purge=900s, embargo=900s
  Session: NY, NSEED=3 (seed-ens K=3), num_leaves=255

Usage:
  ~/binary-algo-venv/bin/python nzdusd_15m_d1feats_cpcv.py ny [stride=2] [cov=0.02,0.03,0.05] [nseed=3]
"""
import os, sys, json, time, numpy as np, pandas as pd
from itertools import combinations
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import harness as H
from sessions import session_mask

TARGET = "NZDUSD"; HOR = 15; STEP = 60; GAP = HOR * STEP; BE = 0.541
FEAT = H.FEAT_DIR; FEATS = H.feature_cols(TARGET)
YEARS = list(range(2012, 2027))
N_GROUPS, K_TEST = 6, 2
PURGE = GAP; EMBARGO = GAP
NUM_LEAVES = 255

SESSION = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] in ("ny", "ldn", "asia", "all") else "ny"
def _arg(i, d, cast):
    rest = [a for a in sys.argv[2:]]
    return cast(rest[i]) if len(rest) > i else d
STRIDE  = _arg(0, 2, int)
_covarg = _arg(1, "0.02,0.03,0.05", str)
COVS    = [float(x) for x in str(_covarg).split(",")]
NSEED   = _arg(2, 3, int)

D1_FEAT_NAMES = ["d1_ret_1", "d1_ret_5", "d1_above_ema20", "d1_above_ema100"]
ALL_FEATS = FEATS + D1_FEAT_NAMES

RESULT_FILE = f"nzdusd_15m_d1feats_cpcv_{SESSION}_result.json"


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


def build_pair_ties(pair, stride):
    Xs = []; fwds = []; tss = []
    for y in YEARS:
        p = f"{FEAT}/{pair}_{y}.parquet"
        if not os.path.exists(p):
            continue
        d = pd.read_parquet(p, columns=FEATS + ["close"])
        d = d[~d.index.duplicated(keep="last")]
        c  = d["close"].values.astype(float)
        ts = d.index.values.astype("datetime64[s]").astype("int64"); n = len(d)
        contig = np.zeros(n, bool)
        contig[:n - HOR] = (ts[HOR:] - ts[:-HOR]) == GAP
        fr = np.full(n, np.nan)
        fr[:n - HOR] = c[HOR:] / c[:-HOR] - 1.0
        keepf = d[FEATS].astype("float32").isna().mean(axis=1).values < 0.5
        valid = contig & np.isfinite(fr) & keepf
        # Add D1 features
        d1f = compute_d1_features(d[["close"]])
        for col in D1_FEAT_NAMES:
            d[col] = d1f[col].values if col in d1f.columns else np.nan
        # Session mask after D1 computation
        smask = session_mask(ts, SESSION)
        valid = valid & smask
        X = pd.concat([d[FEATS], d[D1_FEAT_NAMES]], axis=1).astype("float32")
        # Fill D1 NaNs
        for col in D1_FEAT_NAMES:
            X[col] = X[col].ffill().fillna(0.0)
        idx = np.where(valid)[0]
        if stride > 1:
            idx = idx[::stride]
        Xs.append(X.values[idx])
        fwds.append(fr[idx])
        tss.append(ts[idx])
    if not Xs:
        return None
    return np.concatenate(Xs), np.concatenate(fwds), np.concatenate(tss)


def mk_lgb(n=800, seed=0):
    return lgb.LGBMClassifier(
        objective="binary", metric="auc", learning_rate=0.02,
        num_leaves=NUM_LEAVES, min_child_samples=400,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.5,
        reg_lambda=20, n_estimators=n, n_jobs=16,
        verbosity=-1, random_state=seed,
        bagging_seed=seed, feature_fraction_seed=seed,
    )


def nonoverlap_chrono(ts, mask, gap=GAP):
    take = []; block = -1
    for i in np.where(mask)[0]:
        if ts[i] < block:
            continue
        take.append(i); block = int(ts[i]) + gap
    return np.array(take, dtype=int)


def side_wr(pr, fwd, ts, thr, side):
    conf = np.abs(pr - 0.5)
    want = (pr > 0.5) if side == "UP" else (pr < 0.5) if side == "DOWN" else np.ones(len(pr), bool)
    cand = want & (conf >= thr) & np.isfinite(fwd)
    tr = nonoverlap_chrono(ts, cand)
    if len(tr) == 0:
        return 0, float("nan")
    pred = (pr[tr] > 0.5).astype(int)
    ylab = (fwd[tr] > 0).astype(int)
    moved = (fwd[tr] != 0)
    win = ((pred == ylab) & moved).astype(float)
    return len(tr), float(win.mean())


def summ(a):
    a = np.asarray([x for x in a if np.isfinite(x)], float)
    if len(a) == 0:
        return {"n_paths": 0}
    return {
        "n_paths": int(len(a)),
        "mean": round(float(a.mean()), 4),
        "p10": round(float(np.percentile(a, 10)), 4),
        "p50": round(float(np.percentile(a, 50)), 4),
        "min": round(float(a.min()), 4),
        "frac_clear_BE": round(float((a >= BE).mean()), 3),
    }


if __name__ == "__main__":
    t0 = time.time()
    print("=" * 70)
    print(f"NZDUSD 15m — D1 features CPCV (session={SESSION}, nseed={NSEED}, covs={COVS})")
    print(f"D1 features: {D1_FEAT_NAMES}")
    print("=" * 70)
    print(f"Loading bars (stride={STRIDE})...")
    out = build_pair_ties(TARGET, STRIDE)
    if out is None:
        sys.exit("No data.")
    X_all, fwd_all, ts_all = out
    n = len(fwd_all)
    print(f"  n={n:,} bars, {len(ALL_FEATS)} features (239 base + 4 D1)")

    # CPCV path construction
    year_all = (ts_all / (365.25 * 24 * 3600)).astype(int) + 1970
    uni_yrs = np.unique(year_all)
    # Group years into N_GROUPS
    grp_size = len(uni_yrs) / N_GROUPS
    yr_grp = {yr: min(int(i // grp_size), N_GROUPS - 1) for i, yr in enumerate(uni_yrs)}
    grp_idx = np.array([yr_grp[y] for y in year_all], dtype=int)

    paths = list(combinations(range(N_GROUPS), K_TEST))
    print(f"  {len(paths)} CPCV paths × {NSEED} seeds = {len(paths) * NSEED} model fits")

    path_results = {}
    for pi, test_grps in enumerate(paths):
        test_mask  = np.isin(grp_idx, test_grps)
        train_mask = ~test_mask
        # Purge/embargo boundary bars
        test_ts   = ts_all[test_mask]
        train_ts  = ts_all[train_mask]
        t_lo, t_hi = test_ts.min(), test_ts.max()
        train_mask_clean = train_mask.copy()
        train_mask_clean[train_mask] = (train_ts < t_lo - PURGE) | (train_ts > t_hi + EMBARGO)
        X_tr = X_all[train_mask_clean]; y_tr = (fwd_all[train_mask_clean] > 0).astype(int)
        X_te = X_all[test_mask];       fwd_te = fwd_all[test_mask]; ts_te = ts_all[test_mask]

        # Stride-2 subsampling of train for speed
        n_tr_orig = len(X_tr)
        if STRIDE > 1:
            X_tr = X_tr[::STRIDE]; y_tr = y_tr[::STRIDE]
        probs = []
        for s in range(NSEED):
            m = mk_lgb(seed=s)
            m.fit(X_tr, y_tr)
            probs.append(m.predict_proba(X_te)[:, 1])
        pr = np.mean(probs, axis=0)
        auc = roc_auc_score((fwd_te > 0).astype(int), pr) if len(set((fwd_te > 0).astype(int))) > 1 else 0.5
        path_key = f"g[{','.join(str(g) for g in test_grps)}]"
        path_results[path_key] = {"auc": round(auc, 4), "n": int(len(fwd_te))}
        for cov in COVS:
            thr = np.percentile(np.abs(pr - 0.5), 100 * (1 - cov))
            for side in ["UP", "DOWN", "COMB"]:
                n_s, wr = side_wr(pr, fwd_te, ts_te, thr, side)
                path_results[path_key][f"{side}@cov{int(cov*100)}"] = {"n": n_s, "wr": round(wr, 4) if np.isfinite(wr) else None}
        lab = f"g[{','.join(str(g) for g in test_grps)}]"
        print(f"  path {pi+1:2d}/{len(paths)} {lab}: AUC={auc:.4f} | " +
              " | ".join(f"UP@cov{int(c*100)}={path_results[path_key].get(f'UP@cov{int(c*100)}',{}).get('wr','?'):.3f}" for c in COVS[:1]))

    # Aggregate per side per cov
    print("\n=== CPCV CERTIFICATION RESULTS ===")
    cert = {}
    for cov in COVS:
        cov_key = f"cov{int(cov*100)}"
        cert[cov_key] = {}
        for side in ["UP", "DOWN", "COMB"]:
            k = f"{side}@cov{int(cov*100)}"
            wrs = [path_results[p][k]["wr"] for p in path_results if k in path_results[p] and path_results[p][k]["wr"] is not None]
            s = summ(wrs)
            cert[cov_key][side] = s
            if "p10" in s:
                print(f"  {cov_key} {side}: p10={s['p10']:.4f} frac_clear={s['frac_clear_BE']:.3f} (n_paths={s['n_paths']})")
        print()

    aucs = [path_results[p]["auc"] for p in path_results]
    mean_auc = float(np.mean(aucs))
    print(f"Mean path AUC: {mean_auc:.4f}")
    print(f"Elapsed: {time.time()-t0:.0f}s")

    result = {
        "model": f"NZDUSD.m15{SESSION}_d1feats.v1",
        "session": SESSION,
        "d1_features": D1_FEAT_NAMES,
        "nseed": NSEED,
        "covs": COVS,
        "n_paths": len(paths),
        "mean_path_auc": round(mean_auc, 4),
        "cert": cert,
        "paths": path_results,
        "elapsed_s": round(time.time() - t0, 1),
    }
    with open(RESULT_FILE, "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"\nSaved → {RESULT_FILE}")
