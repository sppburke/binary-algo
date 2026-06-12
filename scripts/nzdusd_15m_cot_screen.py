"""NZDUSD 15m — F8 CFTC COT net speculative positioning VAL screen.

Downloads CFTC Commitments of Traders (legacy) for NZD futures.
Feature: weekly net non-commercial position (large speculator sentiment).
Forward-filled to daily frequency, then merged to 15m bars.

Prior: F3 in backlog assessed as "~null" (Tier-3 cross-pair inference only).
This screen provides Tier-1 NZDUSD evidence for the COT avenue.

Expected result: KILLED — weekly data too coarse for 15m horizon.
Mechanism: weekly crowded-trade positioning → 15m direction (very noisy).

Falsifier: VAL AUC > base (.5219) + .010 = .5319 → ESCALATE; else SUBSUME.
"""
import os, sys, io, json, zipfile
import numpy as np
import pandas as pd

sys.path.insert(0, "/home/sean/git/binary-algo")
import harness as H
from sessions import session_mask
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score

PAIR = "NZDUSD"
HOR = 15; STEP = 60; GAP = HOR * STEP
RESULT_FILE = "nzdusd_15m_cot_screen_result.json"
TRAIN_YEARS = [str(y) for y in range(2012, 2022)]
VAL_YEARS   = ["2022", "2023"]
TR_STRIDE   = 6
NUM_LEAVES  = 255
BASE_VAL_AUC = 0.5219
ESCALATE_THR = BASE_VAL_AUC + 0.010   # 0.5319

COT_YEARS = list(range(2011, 2026))
FEATS = H.feature_cols(PAIR)


def fetch_cot_year(year):
    """Download and parse CFTC legacy COT for one year. Returns raw DataFrame or None."""
    import requests
    url = f"https://www.cftc.gov/files/dea/history/deacot{year}.zip"
    alt = f"https://www.cftc.gov/files/dea/history/deacot{year}.txt"
    for u in [url, alt]:
        try:
            r = requests.get(u, timeout=30)
            if r.status_code != 200:
                continue
            if u.endswith(".zip"):
                z = zipfile.ZipFile(io.BytesIO(r.content))
                names = z.namelist()
                for n in names:
                    if n.lower().endswith(".csv") or n.lower().endswith(".txt"):
                        with z.open(n) as f:
                            df = pd.read_csv(f, low_memory=False)
                        return df
            else:
                df = pd.read_csv(io.StringIO(r.text), low_memory=False)
                return df
        except Exception as e:
            print(f"  [{year}] {type(e).__name__}: {e}")
    return None


def extract_nzd_cot(df):
    """Extract NZD futures net non-commercial (speculative) position."""
    mask = df['Market and Exchange Names'].astype(str).str.upper().str.contains('NEW ZEALAND', na=False)
    nzd = df[mask].copy()
    if len(nzd) == 0:
        return None
    # Date field varies by year — try both column names
    date_col = None
    for c in ['As of Date in Form YYYY-MM-DD', 'Report_Date_as_YYYY-MM-DD',
              'As of Date in Form YYMMDD', 'DATE']:
        if c in nzd.columns:
            date_col = c
            break
    if date_col is None:
        # Fallback: first column with 'date' in name (case-insensitive)
        date_cols = [c for c in nzd.columns if 'date' in c.lower()]
        if date_cols:
            date_col = date_cols[0]
    if date_col is None:
        return None
    nzd['date'] = pd.to_datetime(nzd[date_col], errors='coerce')
    nzd = nzd.dropna(subset=['date'])
    # Net non-commercial position
    long_col  = next((c for c in nzd.columns if 'NonComm' in c and 'Long'  in c and 'All' in c), None)
    short_col = next((c for c in nzd.columns if 'NonComm' in c and 'Short' in c and 'All' in c), None)
    if long_col is None or short_col is None:
        # Try alternate column naming
        long_col  = next((c for c in nzd.columns if 'Noncommercial' in c and 'Long'  in c), None)
        short_col = next((c for c in nzd.columns if 'Noncommercial' in c and 'Short' in c), None)
    if long_col is None or short_col is None:
        return None
    nzd['net_spec'] = pd.to_numeric(nzd[long_col], errors='coerce') - \
                      pd.to_numeric(nzd[short_col], errors='coerce')
    return nzd[['date', 'net_spec']].dropna().set_index('date').sort_index()


def build_cot_features(cot_daily: pd.DataFrame) -> pd.DataFrame:
    """Build lagged COT features from daily (forward-filled) series."""
    s = cot_daily['net_spec']
    feats = {
        'cot_net_prev':   s.shift(1),
        'cot_chg4w':      (s - s.shift(28)).shift(1),
        'cot_chg13w':     (s - s.shift(91)).shift(1),
        'cot_mom4w':      s.pct_change(28).shift(1),
        'cot_z52w':       ((s - s.rolling(365).mean()) / s.rolling(365).std().clip(lower=1e-4)).shift(1),
    }
    return pd.DataFrame(feats, index=cot_daily.index)


def load_bars(years, stride=1, session="ny"):
    dfs = []
    for yr in years:
        p = f"{H.FEAT_DIR}/{PAIR}_{yr}.parquet"
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
        keep_feat = d[FEATS].isna().mean(axis=1).values < 0.5
        valid = contig & np.isfinite(fr) & keep_feat & session_mask(ts, session)
        idx = np.where(valid)[0][::stride]
        sub = d.iloc[idx].copy()
        sub["_fwd"] = fr[idx]
        dfs.append(sub)
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs).sort_index()


def merge_daily_to_bars(bars: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    bar_dates = bars.index.normalize().tz_localize(None).astype("datetime64[us]")
    dai_idx   = daily.index.astype("datetime64[us]")
    d2 = daily.copy()
    d2.index = dai_idx
    d2["_d"] = d2.index
    b2 = bars.copy()
    b2["_d"] = bar_dates.values
    merged = pd.merge_asof(
        b2.reset_index().sort_values("_d"),
        d2.sort_values("_d"),
        on="_d",
        direction="backward"
    ).set_index("datetime_utc").drop(columns=["_d"])
    return merged


if __name__ == "__main__":
    print("=" * 70)
    print("NZDUSD 15m — F8 CFTC COT NZD net speculative positioning VAL screen")
    print(f"Falsifier: VAL AUC > {ESCALATE_THR:.4f} → ESCALATE; else SUBSUME (Tier-1 COT closure)")
    print("=" * 70)

    print("\n[1] Downloading CFTC legacy COT data (NZD futures)...")
    cot_frames = []
    for yr in COT_YEARS:
        df = fetch_cot_year(yr)
        if df is not None:
            nzd = extract_nzd_cot(df)
            if nzd is not None and len(nzd) > 0:
                cot_frames.append(nzd)
                print(f"  {yr}: {len(nzd)} NZD COT rows, "
                      f"net_spec range [{nzd['net_spec'].min():.0f}, {nzd['net_spec'].max():.0f}]")
            else:
                print(f"  {yr}: no NZD rows extracted")
        else:
            print(f"  {yr}: download failed — skipping")

    if not cot_frames:
        print("\n  FATAL: no COT data downloaded. External source confirmed inaccessible.")
        result = {
            "model": "NZDUSD.15m CFTC COT screen (F8)",
            "status": "EXTERNAL-BLOCKED",
            "reason": "CFTC COT download failed for all years — source inaccessible from this environment",
            "decision": "EXTERNAL-BLOCKED — Tier-1 inaccessibility confirmed",
        }
        with open(RESULT_FILE, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  Saved: {RESULT_FILE}")
        sys.exit(0)

    # Combine all years
    cot_all = pd.concat(cot_frames).sort_index()
    cot_all = cot_all[~cot_all.index.duplicated(keep='last')]
    print(f"\n  Total COT rows: {len(cot_all)}, "
          f"{cot_all.index.min().date()} to {cot_all.index.max().date()}")

    # Forward-fill to daily
    date_range = pd.date_range(cot_all.index.min(), cot_all.index.max(), freq='D')
    cot_daily = cot_all.reindex(date_range).fillna(method='ffill')
    print(f"  Forward-filled daily COT: {len(cot_daily)} days")

    print("\n[2] Building COT features...")
    cot_feats = build_cot_features(cot_daily)
    cot_cols  = cot_feats.columns.tolist()
    print(f"  COT features: {cot_cols}")

    print("\n[3] Loading 15m bars...")
    train_yrs = TRAIN_YEARS[::TR_STRIDE]
    train_df  = load_bars(train_yrs, stride=1, session="ny")
    val_df    = load_bars(VAL_YEARS,  stride=1, session="ny")
    print(f"  Train {train_yrs}: {len(train_df):,} bars | Val {VAL_YEARS}: {len(val_df):,} bars")

    print("\n[4] Merging COT features into bars...")
    train_m = merge_daily_to_bars(train_df, cot_feats)
    val_m   = merge_daily_to_bars(val_df,   cot_feats)

    all_feat_cols = FEATS + cot_cols
    all_feat_cols = [c for c in all_feat_cols if c in train_m.columns]
    cot_coverage_tr = train_m[cot_cols].notna().all(axis=1).mean()
    cot_coverage_vl = val_m[cot_cols].notna().all(axis=1).mean()
    print(f"  COT coverage — train: {cot_coverage_tr:.3f}, val: {cot_coverage_vl:.3f}")
    print(f"  Features: {len(FEATS)} base + {len(cot_cols)} COT = {len(all_feat_cols)}")

    X_tr = train_m[all_feat_cols].fillna(0).astype("float32")
    y_tr = (train_m["_fwd"] > 0).astype(int)
    X_vl = val_m[all_feat_cols].fillna(0).astype("float32")
    y_vl = (val_m["_fwd"] > 0).astype(int)

    print(f"\n[5] Training GBM (leaves={NUM_LEAVES}, {len(all_feat_cols)} feats)...")
    model = LGBMClassifier(
        n_estimators=600, num_leaves=NUM_LEAVES, learning_rate=0.02,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.5,
        min_child_samples=400, reg_lambda=20,
        n_jobs=-1, verbosity=-1, random_state=42,
    )
    model.fit(X_tr, y_tr)

    prob_vl = model.predict_proba(X_vl)[:, 1]
    val_auc = roc_auc_score(y_vl, prob_vl)
    lift    = val_auc - BASE_VAL_AUC
    escalate = val_auc > ESCALATE_THR

    print(f"\n[6] Results:")
    print(f"  Base VAL AUC : {BASE_VAL_AUC:.4f}")
    print(f"  COT  VAL AUC : {val_auc:.4f}")
    print(f"  Lift         : {lift:+.4f}")
    print(f"  Threshold    : {ESCALATE_THR:.4f}")
    print(f"  Decision     : {'ESCALATE to CPCV' if escalate else 'SUBSUME — COT lift < .010'}")

    imp     = dict(zip(all_feat_cols, model.feature_importances_))
    cot_imp = sorted([(k, v) for k, v in imp.items() if k in cot_cols], key=lambda x: -x[1])
    print(f"\n  COT feature importances:")
    for k, v in cot_imp:
        print(f"    {k}: {v}")

    result = {
        "model": "NZDUSD.15m CFTC COT NZD net-speculative VAL screen (F8)",
        "key": "NZDUSD.15m",
        "base_val_auc": BASE_VAL_AUC,
        "cot_val_auc": float(val_auc),
        "lift": float(lift),
        "escalate_threshold": ESCALATE_THR,
        "escalate": bool(escalate),
        "decision": "ESCALATE to CPCV" if escalate else "SUBSUME — COT lift < .010 (Tier-1 closure)",
        "cot_rows_loaded": len(cot_all),
        "cot_years_attempted": COT_YEARS,
        "cot_daily_rows": len(cot_daily),
        "cot_coverage_train": float(cot_coverage_tr),
        "cot_coverage_val": float(cot_coverage_vl),
        "cot_importances": [(k, int(v)) for k, v in cot_imp],
        "train_years": train_yrs,
        "val_years": VAL_YEARS,
        "falsifier": "escalate if cot_val_auc > base_val_auc + .010",
        "status": "ESCALATE" if escalate else "SUBSUMED",
    }
    with open(RESULT_FILE, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Saved: {RESULT_FILE}")
    print("=" * 70)
