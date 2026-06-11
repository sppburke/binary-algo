"""NZDUSD 15m — F9 RBNZ OCR + US Fed funds rate differential VAL screen.

Converts "NZ-US rate differential: EXTERNAL-BLOCKED" to Tier-1 NZDUSD evidence.

Data strategy (in order):
  1. RBNZ: try B3 CSV from rbnz.govt.nz stats API
  2. FRED: try IRSTCI01NZM156N (NZ immediate rates, monthly) via public CSV endpoint
  3. Hard-coded: fallback OCR history from RBNZ published decisions (exact dates/rates)
For US rate: Yahoo Finance ^IRX (13-week T-bill, closely tracks Fed funds, daily).

Features: nz_ocr (forward-filled), us_rate (daily), nz_us_spread (OCR - us_rate),
          nz_us_spread_chg30d, nz_us_spread_z52w.

Expected result: KILLED — OCR step-function changes 8×/year, too slow for 15m horizon.
Falsifier: VAL AUC > base (.5219) + .010 = .5319 → ESCALATE; else SUBSUME.
"""
import os, sys, io, json
import numpy as np
import pandas as pd

sys.path.insert(0, "/home/sean/git/binary-algo")
import harness as H
from sessions import session_mask
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score

PAIR = "NZDUSD"
HOR = 15; STEP = 60; GAP = HOR * STEP
RESULT_FILE = "nzdusd_15m_rbnz_ocr_screen_result.json"
TRAIN_YEARS = [str(y) for y in range(2012, 2022)]
VAL_YEARS   = ["2022", "2023"]
TR_STRIDE   = 6
NUM_LEAVES  = 255
BASE_VAL_AUC = 0.5219
ESCALATE_THR = BASE_VAL_AUC + 0.010

FEATS = H.feature_cols(PAIR)

# RBNZ OCR history (hard-coded fallback from RBNZ published decisions)
# Source: rbnz.govt.nz/monetary-policy/official-cash-rate-decisions
# Each entry: (date, OCR rate %)
RBNZ_OCR_HARDCODED = [
    # Rate as at start of our modelling window
    ("2010-12-08", 3.00),
    ("2011-03-09", 2.50),   # Christchurch earthquake emergency cut
    # Flat 2011-2014 (2.50%)
    ("2014-03-13", 2.75),
    ("2014-04-24", 3.00),
    ("2014-06-12", 3.25),
    ("2014-07-24", 3.50),
    # Cut cycle
    ("2015-06-11", 3.25),
    ("2015-07-23", 3.00),
    ("2015-09-10", 2.75),
    ("2016-03-10", 2.25),
    ("2016-08-11", 2.00),
    ("2016-11-10", 1.75),
    # Flat 2016-2019 (1.75%)
    ("2019-05-08", 1.50),
    ("2019-08-07", 1.00),
    # COVID emergency cuts
    ("2020-03-16", 0.25),
    # Flat 2020-2021 (0.25%)
    # Hike cycle
    ("2021-10-06", 0.50),
    ("2021-11-24", 0.75),
    ("2022-02-23", 1.00),
    ("2022-04-13", 1.50),
    ("2022-05-25", 2.00),
    ("2022-07-13", 2.50),
    ("2022-08-17", 3.00),
    ("2022-10-05", 3.50),
    ("2022-11-23", 4.25),
    ("2023-02-22", 4.75),
    ("2023-04-05", 5.25),
    ("2023-05-24", 5.50),
    # Flat 2023-mid-2024 (5.50%)
    ("2024-08-14", 5.25),
    ("2024-10-09", 4.75),
    ("2024-11-27", 4.25),
]


def fetch_rbnz_ocr():
    """Try to fetch RBNZ B3 OCR data from rbnz.govt.nz."""
    import requests
    # RBNZ statistical data download — series B3 (OCR)
    urls = [
        "https://www.rbnz.govt.nz/-/media/project/sites/rbnz/files/statistics/series/b/b3/b3.csv",
        "https://www.rbnz.govt.nz/-/media/project/sites/rbnz/files/statistics/historical/historical-b3.csv",
    ]
    for url in urls:
        try:
            r = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
            if r.status_code == 200 and len(r.content) > 100:
                df = pd.read_csv(io.StringIO(r.text), skiprows=5, header=None)
                # Try to extract date + OCR columns
                print(f"  RBNZ B3: fetched {len(df)} rows from {url}")
                return df
        except Exception as e:
            print(f"  RBNZ B3 fetch failed ({url}): {e}")
    return None


def fetch_fred_nz_rate():
    """Try FRED IRSTCI01NZM156N (NZ immediate rates, monthly) via public CSV."""
    import requests
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=IRSTCI01NZM156N"
    try:
        r = requests.get(url, timeout=15)
        if r.status_code == 200:
            df = pd.read_csv(io.StringIO(r.text))
            df.columns = ["date", "rate"]
            df["date"] = pd.to_datetime(df["date"])
            df = df[df["rate"] != "."].copy()
            df["rate"] = pd.to_numeric(df["rate"], errors="coerce")
            df = df.dropna(subset=["rate"]).set_index("date").sort_index()
            print(f"  FRED IRSTCI01NZM156N: {len(df)} monthly rows, "
                  f"{df.index.min().date()} to {df.index.max().date()}")
            return df["rate"]
        else:
            print(f"  FRED HTTP {r.status_code}")
    except Exception as e:
        print(f"  FRED fetch failed: {e}")
    return None


def build_ocr_daily(ocr_series: pd.Series, date_range: pd.DatetimeIndex) -> pd.Series:
    """Forward-fill OCR to daily frequency."""
    ocr_daily = ocr_series.reindex(date_range).ffill()
    return ocr_daily


def build_rbnz_features(nz_daily: pd.Series, us_daily: pd.Series) -> pd.DataFrame:
    """Build NZ-US spread features from daily series."""
    spread = nz_daily - us_daily
    feats = {
        "nz_ocr_prev":       nz_daily.shift(1),
        "us_rate_prev":      us_daily.shift(1),
        "nz_us_spread_prev": spread.shift(1),
        "nz_us_chg30d":      (spread - spread.shift(30)).shift(1),
        "nz_us_z52w":        ((spread - spread.rolling(365).mean())
                               / spread.rolling(365).std().clip(lower=1e-4)).shift(1),
    }
    return pd.DataFrame(feats)


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


def merge_daily_to_bars(bars: pd.DataFrame, daily_feats: pd.DataFrame) -> pd.DataFrame:
    bar_dates = bars.index.normalize().tz_localize(None).astype("datetime64[us]")
    d2 = daily_feats.copy()
    d2.index = d2.index.astype("datetime64[us]")
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
    print("NZDUSD 15m — F9 RBNZ OCR / NZ-US rate differential VAL screen")
    print(f"Falsifier: VAL AUC > {ESCALATE_THR:.4f} → ESCALATE; else SUBSUME (Tier-1 closure)")
    print("=" * 70)

    import yfinance as yf

    # --- NZ side: RBNZ OCR ---
    print("\n[1] Fetching NZ OCR rate...")
    nz_ocr_series = None
    source_nz = None

    rbnz_raw = fetch_rbnz_ocr()
    if rbnz_raw is not None:
        # Try to parse RBNZ B3 CSV format
        try:
            # B3 format: first col = date, some column = OCR rate
            # Column names vary; look for "Official Cash Rate" or similar
            rbnz_raw.columns = range(len(rbnz_raw.columns))
            date_col = rbnz_raw[0]
            # Find the OCR column (usually col 1 or 2)
            for ci in range(1, min(5, len(rbnz_raw.columns))):
                col = pd.to_numeric(rbnz_raw[ci], errors='coerce')
                if col.notna().sum() > 100:
                    dates = pd.to_datetime(date_col, errors='coerce')
                    valid = dates.notna() & col.notna()
                    nz_ocr_series = col[valid].values
                    idx = dates[valid]
                    nz_ocr_series = pd.Series(nz_ocr_series, index=idx).sort_index()
                    source_nz = "RBNZ B3 (live)"
                    print(f"  NZ OCR from RBNZ: {len(nz_ocr_series)} rows, "
                          f"{nz_ocr_series.index.min().date()} to "
                          f"{nz_ocr_series.index.max().date()}")
                    break
        except Exception as e:
            print(f"  RBNZ B3 parse failed: {e}")
            nz_ocr_series = None

    if nz_ocr_series is None:
        fred_nz = fetch_fred_nz_rate()
        if fred_nz is not None and len(fred_nz) > 50:
            nz_ocr_series = fred_nz
            source_nz = "FRED IRSTCI01NZM156N (monthly)"
        else:
            print("  Falling back to hard-coded RBNZ OCR history")
            dates  = pd.to_datetime([x[0] for x in RBNZ_OCR_HARDCODED])
            rates  = [x[1] for x in RBNZ_OCR_HARDCODED]
            nz_ocr_series = pd.Series(rates, index=dates).sort_index()
            source_nz = "Hard-coded RBNZ decisions (public record)"
            print(f"  NZ OCR: {len(nz_ocr_series)} decision dates, "
                  f"{nz_ocr_series.index.min().date()} to "
                  f"{nz_ocr_series.index.max().date()}")

    # --- US side: 13-week T-bill (^IRX) as Fed funds proxy ---
    print("\n[2] Fetching US short-term rate (^IRX, 13-week T-bill)...")
    try:
        irx = yf.download("^IRX", start="2011-01-01", end="2025-12-31",
                          progress=False, auto_adjust=True)
        us_rate = irx["Close"].squeeze()
        us_rate.index = pd.to_datetime(us_rate.index).normalize().tz_localize(None)
        us_rate = us_rate.sort_index()
        print(f"  ^IRX: {len(us_rate)} days, {us_rate.index.min().date()} to {us_rate.index.max().date()}")
        source_us = "Yahoo Finance ^IRX (daily)"
    except Exception as e:
        print(f"  ^IRX failed: {e} — using hard-coded Fed funds")
        us_rate = None

    if us_rate is None or len(us_rate) < 100:
        # Hard-coded US Fed funds upper bound (approximate)
        fed_dates  = pd.to_datetime(["2012-01-01","2015-12-17","2016-12-15","2017-03-16",
                                     "2017-06-15","2017-12-14","2018-03-22","2018-06-14",
                                     "2018-09-27","2018-12-20","2019-08-01","2019-09-19",
                                     "2019-10-31","2020-03-03","2020-03-16","2022-03-17",
                                     "2022-05-05","2022-06-16","2022-07-28","2022-09-22",
                                     "2022-11-03","2022-12-15","2023-02-02","2023-03-23",
                                     "2023-05-04","2023-07-27","2024-09-19","2024-11-07","2024-12-19"])
        fed_rates  = [0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00,
                      2.25, 2.50, 2.25, 2.00, 1.75, 1.25, 0.25, 0.50,
                      1.00, 1.75, 2.50, 3.25, 4.00, 4.50, 4.75, 5.00,
                      5.25, 5.50, 5.00, 4.75, 4.50]
        us_rate = pd.Series(fed_rates, index=fed_dates).sort_index()
        source_us = "Hard-coded Fed funds (known FOMC dates)"
        print(f"  US rate: {len(us_rate)} FOMC dates")

    # --- Build daily combined series ---
    print("\n[3] Building daily spread features...")
    date_range = pd.date_range("2011-01-01", "2025-12-31", freq="D")
    nz_daily   = build_ocr_daily(nz_ocr_series, date_range)
    # For US: resample to daily if needed
    if isinstance(us_rate.index, pd.DatetimeIndex):
        us_daily = us_rate.reindex(date_range).ffill().bfill()
    else:
        us_daily = pd.Series(dtype=float, index=date_range)
    daily_feats = build_rbnz_features(nz_daily, us_daily)
    rbnz_cols   = daily_feats.columns.tolist()
    print(f"  Spread features: {rbnz_cols}")
    print(f"  NZ-US spread range: [{(nz_daily - us_daily).min():.2f}%, {(nz_daily - us_daily).max():.2f}%]")

    print("\n[4] Loading 15m bars...")
    train_yrs = TRAIN_YEARS[::TR_STRIDE]
    train_df  = load_bars(train_yrs, stride=1, session="ny")
    val_df    = load_bars(VAL_YEARS,  stride=1, session="ny")
    print(f"  Train {train_yrs}: {len(train_df):,} bars | Val {VAL_YEARS}: {len(val_df):,} bars")

    print("\n[5] Merging RBNZ features into bars...")
    train_m = merge_daily_to_bars(train_df, daily_feats)
    val_m   = merge_daily_to_bars(val_df,   daily_feats)

    all_feat_cols = FEATS + rbnz_cols
    all_feat_cols = [c for c in all_feat_cols if c in train_m.columns]
    cov_tr = train_m[rbnz_cols].notna().all(axis=1).mean()
    cov_vl = val_m[rbnz_cols].notna().all(axis=1).mean()
    print(f"  RBNZ feature coverage — train: {cov_tr:.3f}, val: {cov_vl:.3f}")
    print(f"  Features: {len(FEATS)} base + {len(rbnz_cols)} RBNZ = {len(all_feat_cols)}")

    X_tr = train_m[all_feat_cols].fillna(0).astype("float32")
    y_tr = (train_m["_fwd"] > 0).astype(int)
    X_vl = val_m[all_feat_cols].fillna(0).astype("float32")
    y_vl = (val_m["_fwd"] > 0).astype(int)

    print(f"\n[6] Training GBM (leaves={NUM_LEAVES}, {len(all_feat_cols)} feats)...")
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

    print(f"\n[7] Results:")
    print(f"  Base VAL AUC  : {BASE_VAL_AUC:.4f}")
    print(f"  RBNZ VAL AUC  : {val_auc:.4f}")
    print(f"  Lift          : {lift:+.4f}")
    print(f"  Threshold     : {ESCALATE_THR:.4f}")
    print(f"  Decision      : {'ESCALATE to CPCV' if escalate else 'SUBSUME — RBNZ lift < .010'}")

    imp      = dict(zip(all_feat_cols, model.feature_importances_))
    rbnz_imp = sorted([(k, v) for k, v in imp.items() if k in rbnz_cols], key=lambda x: -x[1])
    print(f"\n  RBNZ feature importances:")
    for k, v in rbnz_imp:
        print(f"    {k}: {v}")

    result = {
        "model": "NZDUSD.15m RBNZ OCR + NZ-US spread VAL screen (F9)",
        "key": "NZDUSD.15m",
        "source_nz": source_nz,
        "source_us": source_us,
        "base_val_auc": BASE_VAL_AUC,
        "rbnz_val_auc": float(val_auc),
        "lift": float(lift),
        "escalate_threshold": ESCALATE_THR,
        "escalate": bool(escalate),
        "decision": "ESCALATE to CPCV" if escalate else "SUBSUME — RBNZ OCR lift < .010 (Tier-1 closure)",
        "nz_ocr_rows": int(len(nz_ocr_series)),
        "us_rate_rows": int(len(us_rate)),
        "coverage_train": float(cov_tr),
        "coverage_val": float(cov_vl),
        "rbnz_importances": [(k, int(v)) for k, v in rbnz_imp],
        "train_years": train_yrs,
        "val_years": VAL_YEARS,
        "falsifier": "escalate if rbnz_val_auc > base_val_auc + .010",
        "status": "ESCALATE" if escalate else "SUBSUMED",
    }
    with open(RESULT_FILE, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Saved: {RESULT_FILE}")
    print("=" * 70)
