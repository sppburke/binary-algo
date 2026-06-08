"""USDJPY 15-MINUTE binary DIRECTION lever — E4 INFORMATION BARS (volume-clock sampling).

LEVER (E4): instead of sampling the market on the wall clock, sample by INFORMATION ARRIVAL —
accumulate volume until a fixed quota, close a bar, repeat (Lopez de Prado AFML ch2 "volume bars").
The thesis: event-time bars homogenize the return distribution and may expose direction structure that
the fixed-15m time bar hides. We would build volume-clock bars, fit a LGBM direction model on event-bar
features, and test whether the event-bar signal's held-out moved-AUC (mapped back to the deriv-faithful
fixed-15m time-forward label) clears the certified NY own-pair incumbent (~.539 AUC / ~.58-.60 WR).

  Reference construction: vbars.py (EURUSD volume/dollar bars from the 10s OHLCV cache).

HARD PREREQUISITE (method spec, checked FIRST at runtime, not assumed):
  Information bars are ONLY constructible if a raw VOLUME / TICK_VOLUME series exists on disk. A volume
  clock requires a cumulative, non-negative arrival quantity to slice into equal-quota bars. We probe the
  USDJPY feature parquet columns. If there is no raw volume column, the lever is DATA-LIMITED and we write
  the data-limited verdict and STOP — we do NOT fabricate a volume series (a z-scored / normalized column
  cannot be accumulated; negative values break the cumsum quota, and a derived z-score is not arrival flow).

  Verified at authoring time (2026-06-08): USDJPY_*.parquet has 243 cols; the only volume-like names are
  `vol_z` (a volatility/volume Z-SCORE, mean~0 std~1.09, takes negative values) and `zero_vol` (a sparse
  0/1 flag). Neither is a raw cumulative volume. The repo's raw 10s OHLCV cache with a true `volume` column
  exists for EURUSD only (/home/sean/git/processed/EURUSD); base15m docstring: "USDJPY has NO 1s tick cache;
  only EURUSD does." So on current disk this lever is DATA-LIMITED for USDJPY.

EVAL DISCIPLINE (would apply IF volume existed): DECISION rows restricted to NY session
(sessions.session_mask(ts,'ny')); features stay causal/continuous. EVAL label is ALWAYS the deriv-faithful
fixed-15m sign = sign(close[t+15]-close[t]) with ties (move==0) LOSING — exactly what usdjpy_15m_base.build()
returns via (y, moved). Per held-out year: moved-AUC of the lever signal on NY moved bars + selective
win-rate at cov3% via base.side_eval (nonoverlap gap=900 built in). Sanity: moved up-rate ~0.47-0.53.

FALSIFIER (pre-registered, written BEFORE any held-out read):
  KILL if held-out moved-AUC <= 0.539 (no lift over the certified NY own-pair incumbent) in every year,
  AND NY cov3% win-rate CI95-lower clears 0.541 in < 2 years. Also KILL if DATA-LIMITED (no volume on disk;
  information bars not constructible) — direction (AUC<=.51) cannot be measured without the bars.

Usage: /home/sean/binary-algo-venv/bin/python usdjpy_15m_infobars.py
"""
import os, sys, json, time, glob
import numpy as np, pandas as pd

KEY        = "infobars"
RESULT     = "/home/sean/git/binary-algo/usdjpy_15m_infobars_result.json"
FEAT_DIR   = "/home/sean/git/binary-algo/features"
PAIR       = "USDJPY"
RAW_EURUSD = "/home/sean/git/processed/EURUSD"   # the ONLY raw-volume cache the repo ships (cf. vbars.py)
BE         = 0.541
INCUMBENT_AUC = 0.539

# A column counts as a raw volume series only if it is plausibly a cumulative, NON-NEGATIVE arrival quantity.
# Pure-name z-scores / flags are explicitly excluded; we also reject any candidate that takes negative values.
VOL_NAME_RE  = ("volume", "tick_volume", "tickvol", "tick_vol")
VOL_EXCLUDE  = ("vol_z", "zero_vol", "volz", "vol_zscore")  # derived/normalized -> NOT a volume clock source


def find_volume_column():
    """Return (colname, evidence) for a raw volume series in the USDJPY parquet, else (None, evidence)."""
    fs = sorted(glob.glob(os.path.join(FEAT_DIR, f"{PAIR}_*.parquet")))
    if not fs:
        return None, {"checked": "glob", "note": f"no {PAIR} feature parquets found in {FEAT_DIR}"}
    f = fs[0]
    cols = list(pd.read_parquet(f, columns=None).head(0).columns)
    cand = [c for c in cols
            if any(k in c.lower() for k in VOL_NAME_RE) and c.lower() not in VOL_EXCLUDE]
    vollike = [c for c in cols if "vol" in c.lower() or "tick" in c.lower()]
    ev = {"checked": f, "n_cols": len(cols), "vol_like_cols": vollike, "raw_volume_candidates": cand}
    if not cand:
        return None, ev
    # A name matched -> confirm it is a usable arrival quantity (non-negative, varying). Reject z-scores.
    col = cand[0]
    s = pd.read_parquet(f, columns=[col])[col].astype(float)
    ev["candidate_stats"] = {"col": col, "min": float(s.min()), "max": float(s.max()),
                             "mean": float(s.mean()), "frac_negative": float((s < 0).mean())}
    if float((s < 0).mean()) > 0.0 or float(s.max()) <= 0.0:
        ev["rejection"] = f"{col} takes negative / non-positive values -> not a cumulative volume; rejected"
        return None, ev
    return col, ev


def main():
    t0 = time.time()
    # ---- PRE-REGISTER falsifier (write BEFORE any held-out read) ----
    res = {
        "key": "USDJPY.15m.ny",
        "lever": "E4 information bars (volume-clock sampling; AFML ch2; ref vbars.py)",
        "settlement": "deriv-faithful fixed-15m label sign(close[t+15]-close[t]), ties LOSE, "
                      "breakeven 0.541, gap=900s nonoverlap; DECISION rows NY-session (America/New_York 08-17)",
        "incumbent": {"ny_ownpair_gbm_auc": INCUMBENT_AUC, "ny_ownpair_gbm_wr": "~0.58-0.60"},
        "falsifier": {
            "registered_utc": "pre-held-out",
            "KILL_if": ("DATA-LIMITED (no raw volume column on disk -> information bars not constructible) "
                        "OR held-out moved-AUC <= 0.539 in every year (no lift vs certified NY incumbent) "
                        "AND NY cov3% win-rate CI95-lower clears 0.541 in < 2 held-out years (AUC<=.51 = pure magnitude)"),
            "rationale": ("Volume bars need a cumulative non-negative arrival quantity. SIGN-INVARIANCE "
                          "(arXiv:2512.15720): state/complexity/vol gates target move SIZE not SIGN, so this is run "
                          "faithfully and the number decides. But on current disk USDJPY ships NO raw volume — only "
                          "EURUSD has the 10s OHLCV cache vbars.py consumes — so the bars cannot be built honestly."),
        },
        "val_or_signal_auc": None,
        "years": {},
    }
    json.dump(res, open(RESULT, "w"), indent=2)

    # ---- HARD PREREQUISITE: does a raw volume series exist on disk? (checked FIRST) ----
    vol_col, ev = find_volume_column()
    res["data_check"] = ev
    print(f"[infobars] volume probe: candidate={vol_col!r} vol_like={ev.get('vol_like_cols')} "
          f"({time.time()-t0:.1f}s)", flush=True)

    if vol_col is None:
        note = ("DATA-LIMITED: no volume on disk; information bars not constructible. USDJPY feature parquet "
                f"has no raw volume/tick_volume column (vol-like cols are derived: {ev.get('vol_like_cols')}); "
                "the repo's raw OHLCV+volume cache exists for EURUSD only (vbars.py). A volume clock needs a "
                "cumulative non-negative arrival quantity, which is absent — not fabricating one.")
        res["verdict"] = {"KILLED": False, "beats_base_auc": False, "note": note}
        json.dump(res, open(RESULT, "w"), indent=2)
        print(f"[infobars] {note}", flush=True)
        print(f"[infobars] VERDICT: KILLED=False (DATA-LIMITED) -> {RESULT}  total={time.time()-t0:.1f}s", flush=True)
        return

    # ============================================================================================
    # The block below runs ONLY if a usable raw volume series is found on disk (not the current case
    # for USDJPY). It builds volume-clock bars, fits a LGBM direction model on event-bar features, and
    # evaluates the signal against the deriv-faithful fixed-15m time-forward label, NY-session decisions.
    # ============================================================================================
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score
    sys.path.insert(0, "/home/sean/git/binary-algo")
    import usdjpy_15m_base as base
    from sessions import session_mask

    HOR_S = base.GAP                       # 900s deriv-faithful horizon
    SPL   = base.SPL
    V_THR = None                           # set from train-year median (≈ 15-min-equivalent volume quota)

    def load_year_ohlcv(year):
        """Load minute close + raw volume for a year, contiguous-aware. Returns DataFrame[ts,close,vol]."""
        p = os.path.join(FEAT_DIR, f"{PAIR}_{year}.parquet")
        if not os.path.exists(p):
            return None
        d = pd.read_parquet(p, columns=["close", vol_col])
        d = d[~d.index.duplicated(keep="last")].sort_index()
        ts = d.index.values.astype("datetime64[s]").astype("int64")
        return pd.DataFrame({"ts": ts, "close": d["close"].astype(float).values,
                             "vol": d[vol_col].astype(float).values})

    def make_volbars(d, thr):
        """Volume-clock bars: cut a bar each time cumulative volume crosses a multiple of thr (vbars._bars)."""
        ts = d["ts"].values; c = d["close"].values; v = np.clip(d["vol"].values, 0, None)
        bid = np.floor(np.cumsum(v) / thr).astype("int64")
        chg = np.r_[True, bid[1:] != bid[:-1]]; starts = np.where(chg)[0]; ends = np.r_[starts[1:], len(bid)]
        rows = []
        for s, e in zip(starts, ends):
            sl = slice(s, e)
            rows.append((ts[e-1], c[s], c[sl].max(), c[sl].min(), c[e-1], v[sl].sum(), e-s, ts[e-1]-ts[s]+60))
        return pd.DataFrame(rows, columns=["t", "open", "high", "low", "close", "vol", "nbar", "dur_s"]), ts, c

    def bar_feats(b):
        c = b["close"]; X = pd.DataFrame(index=b.index); r = c.pct_change()
        for k in (1, 2, 3, 5, 8, 13, 21): X[f"ret{k}"] = c.pct_change(k)
        for k in (5, 10, 20, 50): X[f"dist_ema{k}"] = c / c.ewm(span=k).mean() - 1
        dd = c.diff()
        for n in (7, 14, 21):
            X[f"rsi{n}"] = 100 - 100 / (1 + dd.clip(lower=0).ewm(alpha=1/n, adjust=False).mean() /
                                        ((-dd).clip(lower=0).ewm(alpha=1/n, adjust=False).mean() + 1e-12))
        for n in (10, 20, 50): X[f"rv{n}"] = r.rolling(n).std()
        for n in (14, 30):
            hh = b["high"].rolling(n).max(); ll = b["low"].rolling(n).min()
            X[f"rangepos{n}"] = (c - ll) / (hh - ll + 1e-12)
        X["dur_s"] = b["dur_s"]
        X["dur_z"] = (b["dur_s"] - b["dur_s"].rolling(50).mean()) / (b["dur_s"].rolling(50).std() + 1e-9)
        X["bb_pctb"] = (c - c.rolling(20).mean()) / (2 * c.rolling(20).std() + 1e-12)
        return X.replace([np.inf, -np.inf], np.nan)

    def label_bars(b, ts_min, c_min):
        """deriv-faithful fixed-15m label off the MINUTE series at each bar close (no event-bar leakage)."""
        tgt = b["t"].values + HOR_S
        j = np.searchsorted(ts_min, tgt, side="left"); j = np.clip(j, 0, len(ts_min) - 1)
        ok = np.abs(ts_min[j] - tgt) <= 60          # forward minute within 60s of +900s (contiguity)
        fwd = c_min[j]; ret = fwd / b["close"].values - 1.0
        y = (ret > 0).astype(int); moved = ok & np.isfinite(ret) & (ret != 0)
        ny = session_mask(b["t"].values, "ny")      # restrict DECISIONS to NY session
        return y, (moved & ny), b["t"].values.astype("int64")

    def build_window(years, thr):
        Xs, ys, ms, tss = [], [], [], []
        for yr in years:
            d = load_year_ohlcv(yr)
            if d is None: continue
            b, ts_min, c_min = make_volbars(d, thr)
            b.index = pd.RangeIndex(len(b))
            X = bar_feats(b); y, m, t = label_bars(b, ts_min, c_min)
            Xs.append(X); ys.append(y); ms.append(m); tss.append(t)
        return (pd.concat(Xs, ignore_index=True), np.concatenate(ys),
                np.concatenate(ms), np.concatenate(tss))

    # quota = train-year median 15-min volume (≈ time-bar-equivalent info per event bar)
    dtr0 = load_year_ohlcv(SPL["train"][-1])
    V_THR = float(np.nanmedian(dtr0["vol"])) * 15.0
    res["v_thr"] = V_THR
    print(f"[infobars] volume quota V_THR={V_THR:.3g} (15x median minute vol of {SPL['train'][-1]})", flush=True)

    Xtr, ytr, mtr, _    = build_window(SPL["train"], V_THR)
    Xva, yva, mva, tsva = build_window(SPL["val"],   V_THR)
    feat = list(Xtr.columns)
    itr, iva = mtr, mva
    print(f"[infobars] train_bars={len(Xtr):,} moved={int(itr.sum()):,} val_moved={int(iva.sum()):,} "
          f"feats={len(feat)} ({time.time()-t0:.0f}s)", flush=True)

    L = lgb.LGBMClassifier(objective="binary", metric="auc", learning_rate=0.02, num_leaves=127,
                           min_child_samples=200, subsample=0.8, subsample_freq=1, colsample_bytree=0.6,
                           reg_lambda=12, n_estimators=2000, n_jobs=20, verbosity=-1)
    L.fit(Xtr[itr], ytr[itr], eval_set=[(Xva[iva], yva[iva])], eval_metric="auc",
          callbacks=[lgb.early_stopping(150), lgb.log_evaluation(0)])
    pva = L.predict_proba(Xva)[:, 1]
    val_auc = float(roc_auc_score(yva[iva], pva[iva]))
    res["val_or_signal_auc"] = val_auc
    print(f"[infobars] best_iter={L.best_iteration_} VAL moved-AUC(NY)={val_auc:.4f} {time.time()-t0:.0f}s", flush=True)

    # freeze cov3% threshold on VAL confidence
    confv = np.abs(pva[iva] - 0.5)
    THR = float(np.quantile(confv, 1 - 0.03))

    res["years"] = {}
    for w in ("test24", "test25", "oos"):
        Xw, yw, mw, tsw = build_window(SPL[w], V_THR)
        pr = L.predict_proba(Xw)[:, 1]
        if mw.sum() < 5:
            res["years"][w] = {"auc": float("nan"), "cov3_wr": float("nan"), "n_moved": int(mw.sum())}
            continue
        auc = float(roc_auc_score(yw[mw], pr[mw]))
        up_rate = float(yw[mw].mean())
        gate = base.side_eval(pr, yw, mw, tsw, THR)
        g = gate["COMBINED"] if gate else {"n": 0, "wr": float("nan"), "ci": [float("nan")] * 2}
        res["years"][w] = {"auc": auc, "moved_up_rate": up_rate,
                           "cov3_wr": float(g["wr"]), "cov3_n": int(g["n"]), "cov3_ci": g["ci"],
                           "n_moved": int(mw.sum()), "tripwire_ok": bool(0.47 <= up_rate <= 0.53)}
        print(f"=== {w} === moved-AUC(NY)={auc:.4f} up-rate={up_rate:.4f} | cov3% n{g['n']} "
              f"wr={g['wr']:.4f} CI[{g['ci'][0]:.3f},{g['ci'][1]:.3f}]", flush=True)

    # ---- apply falsifier ----
    aucs    = [res["years"][w]["auc"] for w in ("test24", "test25", "oos") if np.isfinite(res["years"][w].get("auc", np.nan))]
    beats   = any(a > INCUMBENT_AUC for a in aucs)
    clears  = [w for w in ("test24", "test25", "oos")
               if np.isfinite(res["years"][w].get("cov3_ci", [np.nan])[0]) and res["years"][w]["cov3_ci"][0] >= BE]
    survive = beats or (len(clears) >= 2)
    res["verdict"] = {
        "KILLED": bool(not survive),
        "beats_base_auc": bool(beats),
        "cov3_CIlo_clears_BE_years": clears,
        "note": (f"E4 volume-clock info bars; held-out moved-AUC(NY) max="
                 f"{max(aucs) if aucs else float('nan'):.4f} vs incumbent {INCUMBENT_AUC}; "
                 f"cov3% CI-lower clears {BE} in {len(clears)} yr(s). "
                 "Sign-invariance: info bars expected to gate SIZE not SIGN — number decides."),
    }
    json.dump(res, open(RESULT, "w"), indent=2)
    print(f"[infobars] VERDICT: {'KILLED' if res['verdict']['KILLED'] else 'SURVIVED'} "
          f"-> {RESULT}  total={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
