"""NZDUSD 15m — D1 daily features VAL screen (LOOKAHEAD-FIXED).

BUG IN PREVIOUS VERSION: resample("1D").last() labels each bin with its
START timestamp (00:00 UTC) but holds the END-OF-DAY value (23:59 UTC).
ffill propagated the end-of-day value to ALL bars in that day, including
those at 14:00-21:00 UTC (NY session). Result: NY session bars saw TODAY's
full-day return (which includes the target 15m return) → LOOKAHEAD BIAS.
Confirmed: Jan 4 14:00 bar saw d1_ret_1=0.001583 = Jan 4's actual daily ret.
Previous AUC .5724 is INVALID.

FIX: shift d1_close by 1 BEFORE computing all features.
  d1_close_raw = close.resample("1D").last()
  d1_close = d1_close_raw.shift(1)  # "Jan 4 00:00" now holds Jan 3 close
  d1_ret_1 = d1_close.pct_change(1)  # = (Jan3 - Jan2)/Jan2 at "Jan 4 00:00"
  ffill → Jan 4 NY bars see Jan 3's return ✓

Lookahead-free mechanism test: does LAGGED 1-day return (yesterday's close-to-close)
predict today's 15m direction? FX momentum at 1-5 day horizon: documented in academic
literature (e.g., Menkhoff et al. 2012 currency momentum; Gao et al. 2018 intraday).
Expected: MUCH smaller than 0.0430 (maybe 0.005-0.015). May or may not clear .5319.
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
RESULT_FILE = "nzdusd_15m_d1feats_clean_result.json"
TRAIN_YEARS  = [str(y) for y in range(2012, 2022)]
VAL_YEARS    = ["2022", "2023"]
TR_STRIDE    = 6
BASE_VAL_AUC = 0.5219
OWN_K3_AUC   = 0.5362
ESCALATE_THR = 0.5319

FEATS = H.feature_cols(PAIR)
D1_FEAT_NAMES = ["d1_ret_1", "d1_ret_5", "d1_above_ema20", "d1_above_ema100"]


def compute_d1_features_clean(df):
    """Compute daily trend features with SHIFT(1) to eliminate lookahead.

    resample("1D").last() labels the bin with its START (00:00 UTC) but holds the
    END-OF-DAY value. shift(1) moves all values forward by one bin so that
    "Jan 4 00:00" holds Jan 3's end-of-day close — fully known at Jan 4 start.
    """
    close = df["close"].copy()
    d1_close_raw = close.resample("1D").last().dropna()
    d1_close = d1_close_raw.shift(1)  # SHIFT: "Jan4 00:00" now = Jan3 23:59 close
    ema20  = d1_close.ewm(span=20, adjust=False).mean()
    ema100 = d1_close.ewm(span=100, adjust=False).mean()
    d1_ret_1 = d1_close.pct_change(1)   # (Jan3 - Jan2)/Jan2 at Jan4 00:00
    d1_ret_5 = d1_close.pct_change(5)   # 5-day return ending Jan3 at Jan4 00:00
    d1_sig = pd.DataFrame({
        "d1_above_ema20":  (d1_close >= ema20).astype(float),
        "d1_above_ema100": (d1_close >= ema100).astype(float),
        "d1_ret_1":  d1_ret_1,
        "d1_ret_5":  d1_ret_5,
    })
    return d1_sig.reindex(df.index, method="ffill")


def verify_no_lookahead(df, year="2022"):
    """Spot-check: Jan 4 NY session bars should see Jan 3's return."""
    close = df["close"].copy()
    d1_close_raw = close.resample("1D").last().dropna()
    d1_close = d1_close_raw.shift(1)
    d1_ret_1 = d1_close.pct_change(1)
    d1_sig = pd.DataFrame({"d1_ret_1": d1_ret_1}).reindex(df.index, method="ffill")

    # Find first NY session bar on Jan 4
    jan4_mask = (df.index.date == pd.Timestamp(f"{year}-01-04").date())
    jan4 = df[jan4_mask]
    jan4_ny = jan4.between_time("14:00", "21:00")
    if len(jan4_ny) == 0:
        return
    first_bar = jan4_ny.index[0]
    seen_ret = d1_sig.loc[first_bar, "d1_ret_1"]

    # Expected: Jan 3 return (Jan2 close→Jan3 close)
    jan3_date = pd.Timestamp(f"{year}-01-03").date()
    jan2_date = pd.Timestamp(f"{year}-01-02").date()
    jan3_ts = [t for t in d1_close_raw.index if t.date() == jan3_date]
    jan2_ts = [t for t in d1_close_raw.index if t.date() == jan2_date]
    if jan3_ts and jan2_ts:
        jan3_close = float(d1_close_raw[jan3_ts[0]])
        jan2_close = float(d1_close_raw[jan2_ts[0]])
        expected = (jan3_close - jan2_close) / jan2_close
        # What would LOOKAHEAD look like (Jan 4's own return)?
        jan4_ts = [t for t in d1_close_raw.index if t.date() == pd.Timestamp(f"{year}-01-04").date()]
        lookahead = (d1_close_raw[jan4_ts[0]] - jan3_close) / jan3_close if jan4_ts else None
        print(f"LOOKAHEAD VERIFICATION (Jan 4 14:00 UTC bar):")
        print(f"  Seen d1_ret_1    : {seen_ret:.6f}")
        print(f"  Expected (Jan 3 ret): {expected:.6f}")
        print(f"  Lookahead (Jan 4 ret): {lookahead:.6f}")
        print(f"  Match expected   : {abs(seen_ret - expected) < 1e-7}")
        print(f"  *** LOOKAHEAD ELIMINATED: {not (lookahead and abs(seen_ret - lookahead) < 1e-7)} ***")


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
        d1f = compute_d1_features_clean(d[["close"]])
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
    print("NZDUSD 15m — D1 features screen (LOOKAHEAD-FIXED, shift=1)")
    print("=" * 70)

    # First: verify fix
    d_check = pd.read_parquet(f"{H.FEAT_DIR}/NZDUSD_2022.parquet", columns=["close"])
    d_check = d_check[~d_check.index.duplicated(keep="last")]
    verify_no_lookahead(d_check, "2022")
    print()

    all_fcols = FEATS + D1_FEAT_NAMES
    train_yrs = TRAIN_YEARS[::TR_STRIDE]  # stride-6: 2012, 2018
    tr_df = load_bars(train_yrs, stride=1)
    vl_df = load_bars(VAL_YEARS, stride=1)
    print(f"  train: {len(tr_df):,} | val: {len(vl_df):,}")

    for df in [tr_df, vl_df]:
        df[D1_FEAT_NAMES] = df[D1_FEAT_NAMES].ffill().fillna(0.0)

    # Raw correlations with target
    for f in D1_FEAT_NAMES:
        c = np.corrcoef(tr_df[f].fillna(0), tr_df["_fwd"] > 0)[0, 1]
        print(f"  {f}: target corr = {c:.6f}")

    X_tr = tr_df[all_fcols].astype("float32")
    y_tr = (tr_df["_fwd"] > 0).astype(int)
    X_vl = vl_df[all_fcols].astype("float32")
    y_vl = (vl_df["_fwd"] > 0).astype(int)

    print(f"\nBase (239 feats, seed=42, stride-6)...")
    m_base = LGBMClassifier(
        n_estimators=600, num_leaves=255, learning_rate=0.02,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.5,
        min_child_samples=400, reg_lambda=20,
        n_jobs=-1, verbosity=-1, random_state=42,
    )
    m_base.fit(X_tr[FEATS], y_tr)
    auc_base = roc_auc_score(y_vl, m_base.predict_proba(X_vl[FEATS])[:, 1])
    print(f"  Base: {auc_base:.4f}")

    print(f"\n+D1 clean features (243 feats, seed=42, stride-6)...")
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
    d1_top = imp[D1_FEAT_NAMES].sort_values(ascending=False)
    d1_share = imp[D1_FEAT_NAMES].sum() / imp.sum()

    lift     = auc_d1 - BASE_VAL_AUC
    lift_chk = auc_d1 - auc_base
    escalate = auc_d1 > ESCALATE_THR

    print(f"\n=== D1 CLEAN VAL RESULT ===")
    print(f"Base ref (stride-6)    : {BASE_VAL_AUC:.4f}")
    print(f"Base check (this run)  : {auc_base:.4f}")
    print(f"Own K=3 (ref, full tr) : {OWN_K3_AUC:.4f}")
    print(f"+D1 clean (243 feats)  : {auc_d1:.4f}")
    print(f"Lift vs ref            : {lift:+.4f}")
    print(f"Lift vs check          : {lift_chk:+.4f}")
    print(f"D1 importance share    : {d1_share:.2%}")
    print(f"Escalate (>{ESCALATE_THR}): {'YES' if escalate else 'NO → SUBSUMED'}")
    for f, v in d1_top.items():
        print(f"  {f}: {v}")

    result = {
        "model": "NZDUSD 15m D1 clean (shift=1, NO lookahead)",
        "features": D1_FEAT_NAMES,
        "lookahead_fix": "shift(1) on d1_close_raw before computing EMA/ret",
        "previous_auc_was_invalid": 0.5471,
        "base_val_auc_ref": BASE_VAL_AUC,
        "base_val_auc_check": float(auc_base),
        "own_k3_val_auc": OWN_K3_AUC,
        "d1_clean_val_auc": float(auc_d1),
        "lift_vs_ref": float(lift),
        "lift_vs_check": float(lift_chk),
        "d1_importance_share": float(d1_share),
        "escalate": bool(escalate),
        "decision": "ESCALATE" if escalate else "SUBSUMED",
    }
    with open(RESULT_FILE, "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"\nSaved → {RESULT_FILE}")
