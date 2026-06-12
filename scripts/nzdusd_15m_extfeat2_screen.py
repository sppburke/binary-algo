"""NZDUSD 15m — Expanded external features VAL screen (F6).

Extends F5 (VIX/DXY/US10Y/Gold) with 5 Antipodean/commodity/regional tickers:
  NZX50   (^NZ50)    — NZD-specific equity market; captures RBNZ/NZ-economy tone
  ASX200  (^AXJO)    — Antipodean equity; NZD's closest economic partner
  AUDNZD  (AUDNZD=X) — AUD/NZD cross; captures AUD-NZD differential not in pair bars
  Copper  (HG=F)     — "Dr. Copper" industrial demand / China proxy (vs Gold = safe-haven)
  HangSeng(^HSI)     — China equity; NZ's largest export market (dairy/meat)

Total: 9 tickers × 5 lags = 45 external features + 239 base = 284 total.

Falsifier: ESCALATE to full CPCV if VAL AUC > base VAL AUC (.5219) + .010 = .5319.
Otherwise SUBSUME — no accessible daily external data moves 15m AUC by >.010.
"""
import os, sys, json
import numpy as np
import pandas as pd
sys.path.insert(0, "/home/sean/git/binary-algo")
import harness as H
from sessions import session_mask
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score

PAIR = "NZDUSD"
HOR = 15; STEP = 60; GAP = HOR * STEP
RESULT_FILE = "nzdusd_15m_extfeat2_screen_result.json"
TRAIN_YEARS = [str(y) for y in range(2012, 2022)]
VAL_YEARS   = ["2022", "2023"]
TR_STRIDE   = 6       # 2012, 2018
NUM_LEAVES  = 255
BASE_VAL_AUC = 0.5219
ESCALATE_THR = BASE_VAL_AUC + 0.010  # 0.5319

# F5 tickers + F6 Antipodean/commodity/regional additions
TICKERS = {
    "^VIX":     "vix",
    "DX-Y.NYB": "dxy",
    "^TNX":     "us10y",
    "GC=F":     "gold",
    "^NZ50":    "nzx50",
    "^AXJO":    "asx200",
    "AUDNZD=X": "audnzd",
    "HG=F":     "copper",
    "^HSI":     "hsi",
}

FEATS = H.feature_cols(PAIR)


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
        valid = contig & np.isfinite(fr) & keep_feat
        smask = session_mask(ts, session)
        valid = valid & smask
        idx = np.where(valid)[0]
        if stride > 1:
            idx = idx[::stride]
        sub = d.iloc[idx].copy()
        sub["_fwd"] = fr[idx]
        sub["_ts"]  = ts[idx]
        dfs.append(sub)
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs).sort_index()


def fetch_external_daily():
    import yfinance as yf
    frames = {}
    for ticker, name in TICKERS.items():
        try:
            data = yf.download(ticker, start="2011-01-01", end="2025-12-31",
                               progress=False, auto_adjust=True)
            if len(data) == 0:
                print(f"  WARNING: {ticker} ({name}) returned empty — skipping")
                continue
            close = data["Close"].squeeze()
            close.name = name
            close.index = pd.to_datetime(close.index).normalize().tz_localize(None)
            frames[name] = close
            print(f"  {name} ({ticker}): {len(data)} days, "
                  f"{data.index[0].date()} to {data.index[-1].date()}")
        except Exception as e:
            print(f"  ERROR fetching {ticker} ({name}): {e}")
    if not frames:
        return pd.DataFrame()
    daily = pd.concat(frames.values(), axis=1)
    daily.index.name = "date"
    return daily.sort_index()


def build_ext_feats(daily: pd.DataFrame) -> pd.DataFrame:
    feats = {}
    for col in daily.columns:
        s = daily[col]
        feats[f"{col}_prev"]   = s.shift(1)
        feats[f"{col}_ret1d"]  = s.pct_change(1).shift(1)
        feats[f"{col}_mom5d"]  = s.pct_change(5).shift(1)
        feats[f"{col}_mom20d"] = s.pct_change(20).shift(1)
        roll = s.rolling(63)
        feats[f"{col}_z63"]    = ((s - roll.mean()) / roll.std().clip(lower=1e-8)).shift(1)
    return pd.DataFrame(feats, index=daily.index)


def merge_ext_to_bars(bars: pd.DataFrame, ext: pd.DataFrame) -> pd.DataFrame:
    bar_dates = bars.index.normalize().tz_localize(None)
    bar_dates_us = bar_dates.astype("datetime64[us]")
    ext_index_us = ext.index.astype("datetime64[us]")
    ext_dated = ext.copy()
    ext_dated.index = ext_index_us
    ext_dated["_d"] = ext_dated.index
    bars = bars.copy()
    bars["_d"] = bar_dates_us.values
    merged = pd.merge_asof(
        bars.reset_index().sort_values("_d"),
        ext_dated.sort_values("_d"),
        on="_d",
        direction="backward"
    ).set_index("datetime_utc").drop(columns=["_d"])
    return merged


if __name__ == "__main__":
    print("=" * 70)
    print("NZDUSD 15m — F6 Expanded external features VAL screen")
    print(f"Tickers: {', '.join(TICKERS.values())}")
    print(f"Falsifier: VAL AUC > {ESCALATE_THR:.4f} → ESCALATE to CPCV")
    print("=" * 70)

    print("\n[1] Loading bars (train stride=6, val full)...")
    train_yrs = TRAIN_YEARS[::TR_STRIDE]
    train_df  = load_bars(train_yrs, stride=1, session="ny")
    val_df    = load_bars(VAL_YEARS,  stride=1, session="ny")
    print(f"  Train {train_yrs}: {len(train_df):,} bars | Val {VAL_YEARS}: {len(val_df):,} bars")

    print("\n[2] Fetching external daily data from Yahoo Finance...")
    daily_df = fetch_external_daily()
    if daily_df.empty:
        print("  FATAL: no external data — aborting")
        sys.exit(1)

    print("\n[3] Building lagged external features...")
    ext_df   = build_ext_feats(daily_df)
    ext_cols = ext_df.columns.tolist()
    print(f"  Tickers loaded: {list(daily_df.columns)}")
    print(f"  External feature columns: {len(ext_cols)}")

    print("\n[4] Merging into bars...")
    train_m = merge_ext_to_bars(train_df, ext_df)
    val_m   = merge_ext_to_bars(val_df,   ext_df)

    all_feat_cols = FEATS + ext_cols
    all_feat_cols = [c for c in all_feat_cols if c in train_m.columns]
    print(f"  Base: {len(FEATS)} | Ext: {len(ext_cols)} | Total in model: {len(all_feat_cols)}")

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
    print(f"  Ext  VAL AUC : {val_auc:.4f}")
    print(f"  Lift         : {lift:+.4f}")
    print(f"  Threshold    : {ESCALATE_THR:.4f}")
    print(f"  Decision     : {'ESCALATE to CPCV' if escalate else 'SUBSUME — lift < .010'}")

    imp      = dict(zip(all_feat_cols, model.feature_importances_))
    ext_imp  = sorted([(k, v) for k, v in imp.items() if k in ext_cols], key=lambda x: -x[1])
    print(f"\n  Top-15 external feature importances:")
    for k, v in ext_imp[:15]:
        print(f"    {k}: {v}")

    result = {
        "model": "NZDUSD.15m expanded-external VAL screen (F6)",
        "key": "NZDUSD.15m",
        "base_val_auc": BASE_VAL_AUC,
        "ext_val_auc": float(val_auc),
        "lift": float(lift),
        "escalate_threshold": ESCALATE_THR,
        "escalate": bool(escalate),
        "decision": "ESCALATE to CPCV" if escalate else "SUBSUME — lift < .010",
        "tickers_attempted": list(TICKERS.keys()),
        "tickers_loaded": list(daily_df.columns),
        "ext_features_tested": ext_cols,
        "top_ext_importances": [(k, int(v)) for k, v in ext_imp[:15]],
        "train_years": train_yrs,
        "val_years": VAL_YEARS,
        "falsifier": "escalate if ext_val_auc > base_val_auc + .010",
    }
    with open(RESULT_FILE, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Saved: {RESULT_FILE}")
    print("=" * 70)
