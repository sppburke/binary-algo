"""EXPERIMENT #1 — CKS EVENT ORDER-FLOW IMBALANCE (Cont-Kukanov-Stoikov) for 60s EURUSD direction.

WHY untried (verified this session): every on-disk "OFI" is either tick-rule sign(close-open)*vol
(event_signs.py / m5_tick OFI) or the STATIC size imbalance `imb=(bid_vol-ask_vol)/(bid_vol+ask_vol)`
(tick1s_cache.py:22, the min1 reversion book's order-flow input). Neither conditions the volume on the
direction of the best-bid/best-ask PRICE move. CKS event-OFI does exactly that:

  e_n = q^bid_n * 1[P^bid_n >= P^bid_{n-1}] - q^bid_{n-1} * 1[P^bid_n <= P^bid_{n-1}]
        - q^ask_n * 1[P^ask_n <= P^ask_{n-1}] + q^ask_{n-1} * 1[P^ask_n >= P^ask_{n-1}]

We aggregate e_n to 1s (signed sum) and form signed OFI over 5/15/30/60s windows, then test:
  (a) STANDALONE 60s direction LGBM on the CKS-OFI block.
  (b) SWAP it in to REPLACE the tick-rule OFI (`imb`) in the min1 reversion book (retrain ensemble,
      re-run the exact deriv-faithful reversion backtest).

Deriv-faithful 60s throughout: settlement = min1_production.wc_ret (next-tick entry lag, last-tick<=expiry,
mid-to-mid, ties LOSE), non-overlap chronological (no look-ahead), MOVED-BARS-ONLY (|ret|>0), per-year
2024/2025/2026, bootstrap CI95. Select on VAL by worst-VAL-half. Subsample <=120k.

PRE-REGISTERED FALSIFIER: KILL if standalone VAL dirAUC <= 0.515 AND the swap does not lift the binding
2025 window with CI clearing the prior (min1 book 2025 baseline).

Memory safety: a 15m CPCV is running concurrently. Build the CKS cache day-by-day (accumulate only the small
1s bars), process one parquet at a time, del big frames, models small, subsample <=120k.

Splits (mirror tick1s_cache.py):
  train 2021/2022/2023 | val 2024-04,05 | test 2024-09..11 + 2025-02..04 + 2025-09..11 | oos 2026-02..04
Usage:  python min1_cksofi.py build      # one-time CKS 1s cache (features_tick_cks/)
        python min1_cksofi.py run        # standalone (a) + swap (b); writes min1_cksofi_result.json
        python min1_cksofi.py            # build (if missing) then run
"""
import sys, os, glob, json, time, calendar, gc
import numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

ROOT = "/home/sean/git/binary-algo"
RAW  = "/media/sean/CORSAIR/tick_data/raw"
PAIR = "EURUSD"
TICK = f"{ROOT}/features_tick"                 # existing 1s micro cache (mid, imb, ...)
CKS  = f"{ROOT}/features_tick_cks"             # CKS 1s cache we build here
RESULT = f"{ROOT}/min1_cksofi_result.json"
os.makedirs(CKS, exist_ok=True)

# reuse the deriv-faithful settlement, bootstrap, de-overlap, and the min1 feature builder
from min1_production import (wc_ret, boot, nonoverlap_chrono, feats as min1_feats,
                             HS, TOL_S, ENTRY_LAG_S, GAP, mk_lgb)

SUBSAMPLE = 120_000                            # hard cap on any fit's row count
CKS_WINS  = (5, 15, 30, 60)                    # signed OFI windows (s)
MIN_AVAIL_MB = 2200                            # abort if system free mem drops below this (protect concurrent CPCV)

def _avail_mb():
    try:
        with open("/proc/meminfo") as f:
            for ln in f:
                if ln.startswith("MemAvailable:"):
                    return int(ln.split()[1]) // 1024
    except Exception:
        return 99999
    return 99999

def _guard(where):
    a = _avail_mb()
    if a < MIN_AVAIL_MB:
        raise MemoryError(f"GUARD abort at {where}: avail={a}MB < {MIN_AVAIL_MB}MB (protecting concurrent CPCV)")
    return a

def mo(y, ms):
    out = []
    for m in ms:
        out += [f"{y}-{m:02d}-{d:02d}" for d in range(1, calendar.monthrange(y, m)[1] + 1)]
    return out
SPLIT_DATES = {
    "train": mo(2021, [3, 6, 9, 12]) + mo(2022, [2, 4, 6, 8, 10, 12]) + mo(2023, [2, 4, 6, 8, 10, 12]),
    "val":   mo(2024, [4, 5]),
    "test":  mo(2024, [9, 10, 11]) + mo(2025, [2, 3, 4, 9, 10, 11]),
    "oos":   mo(2026, [2, 3, 4]),
}

# ----------------------------- CKS event-OFI 1s cache -----------------------------
def _day_cks(files):
    """Resample ONE day's hourly tick files to 1s CKS-event-OFI sums (bounded memory)."""
    parts = [pd.read_parquet(f, columns=["bid", "ask", "bid-vol", "ask-vol", "timestamp_utc"]) for f in files]
    t = pd.concat(parts, ignore_index=True)
    t = t.sort_values("timestamp_utc")
    bid = t["bid"].values.astype(float); ask = t["ask"].values.astype(float)
    qb  = t["bid-vol"].values.astype(float); qa = t["ask-vol"].values.astype(float)
    ts  = t["timestamp_utc"].values.astype(float)
    n = len(t)
    if n < 2:
        return None
    # CKS event-OFI per tick n (uses n and n-1). e[0] undefined -> 0.
    e = np.zeros(n, dtype=float)
    db = bid[1:] - bid[:-1]           # P^bid_n - P^bid_{n-1}
    da = ask[1:] - ask[:-1]           # P^ask_n - P^ask_{n-1}
    qb_n, qb_p = qb[1:], qb[:-1]
    qa_n, qa_p = qa[1:], qa[:-1]
    e[1:] = (qb_n * (db >= 0).astype(float) - qb_p * (db <= 0).astype(float)
             - qa_n * (da <= 0).astype(float) + qa_p * (da >= 0).astype(float))
    sec = np.floor(ts).astype("int64")
    df = pd.DataFrame({"sec": sec, "e": e})
    g = df.groupby("sec")
    out = pd.DataFrame({
        "cks_e": g["e"].sum(),          # signed CKS event-OFI summed over the second
        "cks_nev": g["e"].count(),      # event count (for normalization / context)
    })
    out.index = pd.to_datetime(out.index, unit="s", utc=True)
    del t, df, parts, bid, ask, qb, qa, ts, e
    return out.astype("float32")

def build_split(sp):
    p = f"{CKS}/{sp}_cks1s.parquet"
    if os.path.exists(p):
        print(f"[build] {sp}: exists", flush=True); return
    t0 = time.time(); days = []
    for d in SPLIT_DATES[sp]:
        files = sorted(glob.glob(f"{RAW}/{PAIR}/{PAIR}_{d}_*.parquet"))
        if not files:
            continue
        dd = _day_cks(files)
        if dd is not None:
            days.append(dd)
        gc.collect()
    if not days:
        print(f"[build] {sp}: NO DATA", flush=True); return
    b = pd.concat(days); del days; gc.collect()
    b.to_parquet(p)
    print(f"[build] {sp}: {len(b):,} 1s-bars saved ({time.time()-t0:.0f}s)", flush=True)
    del b; gc.collect()

def build_all():
    for sp in SPLIT_DATES:
        build_split(sp)
    print("[build] CKS CACHE DONE", flush=True)

# ----------------------------- assemble per-split frame -----------------------------
def load_merged(sp):
    """Merge CKS 1s sums onto the existing 1s micro cache (shares mid + index). Returns the
    micro-bar frame b (with mid, imb, ...) plus the CKS columns aligned on the 1s index."""
    bm = pd.read_parquet(f"{TICK}/{sp}_1s.parquet")          # mid, imb, micro, spread, nt, tsz
    bc = pd.read_parquet(f"{CKS}/{sp}_cks1s.parquet")        # cks_e, cks_nev
    # left-join CKS onto micro bars (micro bars are the trading clock used by min1)
    b = bm.join(bc, how="left")
    b["cks_e"] = b["cks_e"].fillna(0.0).astype("float32")
    b["cks_nev"] = b["cks_nev"].fillna(0.0).astype("float32")
    del bm, bc; gc.collect()
    return b

def _month_slices(b):
    """Yield (period_str, integer positions) for each calendar month, in chronological order.
    A month is contiguous in the monotonic index; rolling windows (max 3600s=1h) reset at month
    boundaries — the same per-chunk convention the 1s cache uses, negligible for a tree fit and
    boundary-faithful within each month's interior (the de-overlap eval is per-month anyway)."""
    mp = b.index.tz_localize(None).to_period("M")
    codes, uniq = pd.factorize(mp, sort=True)
    for k, per in enumerate(uniq):
        pos = np.where(codes == k)[0]
        yield str(per), pos

def feats_chunked_fit(b, fb, valid, mag, ts, want_rows, seed=7):
    """Build features MONTH-BY-MONTH (bounded memory), keep only moved&valid bars, subsample to
    ~want_rows total spread across months. Returns (Xfit, yfit) as small arrays. Never holds the
    full-split feature frame — peak is one month (~0.7M rows x 62 cols ~ 170MB)."""
    parts_X = []; parts_y = []
    months = list(_month_slices(b))
    per_month = max(1, want_rows // max(1, len(months)))
    for per, pos in months:
        sl = b.iloc[pos]
        Xm = fb(sl)
        vm = valid[pos] & (mag[pos] > 0)
        idx = np.where(vm)[0]
        if len(idx) == 0:
            del sl, Xm; gc.collect(); continue
        if len(idx) > per_month:
            stride = int(np.ceil(len(idx) / per_month)); idx = idx[::stride]
        parts_X.append(Xm.iloc[idx].copy())
        parts_y.append(pos[idx])
        del sl, Xm; gc.collect()
    Xfit = pd.concat(parts_X); del parts_X; gc.collect()
    yrows = np.concatenate(parts_y)
    return Xfit, yrows

def feats_chunked_predict(b, fb, L, cols_needed=()):
    """Predict p MONTH-BY-MONTH; also collect requested gate columns. Returns (p, {col:array}).
    Never holds the full feature frame."""
    n = len(b); p = np.empty(n, dtype="float32")
    extra = {c: np.empty(n, dtype="float32") for c in cols_needed}
    for per, pos in _month_slices(b):
        sl = b.iloc[pos]; Xm = fb(sl)
        p[pos] = L.predict_proba(Xm)[:, 1].astype("float32")
        for c in cols_needed:
            extra[c][pos] = Xm[c].values.astype("float32")
        del sl, Xm; gc.collect()
    return p, extra

def cks_block(b):
    """Standalone CKS-OFI feature block: signed windowed OFI + simple causal transforms. All causal."""
    e = b["cks_e"]; nev = b["cks_nev"]
    X = pd.DataFrame(index=b.index)
    X["cks_e"] = e
    for w in CKS_WINS:
        s = e.rolling(w, min_periods=1).sum()
        X[f"cks_ofi{w}"] = s
        # scale-normalized OFI (divide by event-volume proxy to get a unit-free imbalance)
        denom = (e.abs().rolling(w, min_periods=1).sum() + nev.rolling(w, min_periods=1).sum() * 1e3 + 1e-9)
        X[f"cks_ofin{w}"] = s / denom
    for w in (5, 15, 60):
        X[f"cks_ema{w}"] = e.ewm(span=w).mean()
    X["cks_acc"] = e.ewm(span=5).mean() - e.ewm(span=30).mean()
    X["cks_nev30"] = nev.ewm(span=30).mean()
    return X.replace([np.inf, -np.inf], np.nan).astype("float32")

def prep_labels(b, lag_s=ENTRY_LAG_S):
    """Deriv-faithful 60s labels on the micro-bar clock (identical to min1_production.prep)."""
    mid = b["mid"].values.astype(float)
    ts = b.index.values.astype("datetime64[s]").astype("int64")
    ret, valid = wc_ret(ts, mid, HS, TOL_S, lag_s)
    y = (ret > 0).astype(int)
    return y, np.abs(ret), valid, ts

def year_of(ts):
    return pd.to_datetime(ts, unit="s", utc=True).year

def subsample_idx(idx, cap=SUBSAMPLE, seed=7):
    if len(idx) <= cap:
        return idx
    stride = int(np.ceil(len(idx) / cap))
    return idx[::stride]

# ----------------------------- deriv-faithful per-year eval -----------------------------
def eval_book(ts, y, mag, valid, pred_up, conf, conf_thr, gate_extra=None):
    """Non-overlap chronological, moved-bars-only, per-year acc + CI95.
    gate_extra: optional boolean mask (e.g. the reversion/regime gate). Returns dict per year + combined."""
    base = valid & (conf >= conf_thr)
    if gate_extra is not None:
        base = base & gate_extra
    tr = nonoverlap_chrono(ts, base)
    out = {}
    yrs = year_of(ts[tr]) if len(tr) else np.array([])
    # moved-bars-only: a tie (mag==0) is a LOSING trade, but it is still a real selected trade on deriv.
    # We report MOVED-bars accuracy (the honest signal quality) AND keep ties as losses in the win-rate.
    for label, mask in (("2024", yrs == 2024), ("2025", yrs == 2025), ("2026", yrs == 2026), ("all", np.ones(len(tr), bool))):
        sel = tr[mask] if len(tr) else tr
        if len(sel) == 0:
            out[label] = {"n": 0, "winrate": None, "ci": [None, None], "moved_acc": None, "n_moved": 0}
            continue
        win = ((pred_up[sel] == y[sel]) & (mag[sel] > 0)).astype(float)   # tie => loss
        lo, hi = boot(win)
        moved = mag[sel] > 0
        mv_acc = (pred_up[sel][moved] == y[sel][moved]).mean() if moved.sum() else None
        out[label] = {"n": int(len(sel)), "winrate": float(win.mean()), "ci": [float(lo), float(hi)],
                      "moved_acc": (float(mv_acc) if mv_acc is not None else None), "n_moved": int(moved.sum())}
    return out

# ----------------------------- (a) STANDALONE CKS-OFI direction -----------------------------
def run_standalone():
    print("\n===== (a) STANDALONE CKS-OFI 60s DIRECTION =====", flush=True)
    _guard("a:load_train")
    # TRAIN: chunked feature build -> 120k moved-bar fit slice, then free the merged train frame
    btr = load_merged("train"); ytr, mtr, vtr, tstr = prep_labels(btr)
    XA, fit_rows = feats_chunked_fit(btr, cks_block, vtr, mtr, tstr, SUBSAMPLE)
    yA = ytr[fit_rows]; feat_names = list(XA.columns)
    del btr, ytr, mtr, vtr, tstr; gc.collect()
    # VAL: labels on the small merged frame; chunked feature build for the early-stop slice
    bva = load_merged("val"); yva, mva, vva, tsva = prep_labels(bva)
    Xva = cks_block(bva)                                          # val is small (1.3M x 14 cols) -> fine
    iva = np.where(vva & (mva > 0))[0]
    print(f"[a] train fit rows={len(XA):,} (cap {SUBSAMPLE:,}) val rows={len(iva):,}", flush=True)
    iva_sorted = iva[np.argsort(tsva[iva])]
    half = iva_sorted[len(iva_sorted) // 2:]                     # worst-VAL-half early-stop (NEVER acc-max)
    L = mk_lgb(2000)
    L.fit(XA, yA, eval_set=[(Xva.iloc[half], yva[half])], eval_metric="auc",
          callbacks=[lgb.early_stopping(120), lgb.log_evaluation(0)])
    del XA, yA; gc.collect()
    pva_half = L.predict_proba(Xva.iloc[half])[:, 1]
    val_auc_worsthalf = float(roc_auc_score(yva[half], pva_half))
    pva_full = L.predict_proba(Xva.iloc[iva])[:, 1]
    val_auc_full = float(roc_auc_score(yva[iva], pva_full))
    # confidence threshold: 5% coverage on the worst-VAL-half (live-faithful, not acc-max)
    conf_va = np.abs(pva_half - 0.5)
    conf_thr = float(np.quantile(conf_va, 0.95))
    print(f"[a] VAL dirAUC worst-half={val_auc_worsthalf:.4f} full={val_auc_full:.4f} conf_thr(5%cov)={conf_thr:.4f}", flush=True)
    del bva, Xva, yva, mva, vva, tsva, pva_half, pva_full, conf_va; gc.collect()
    # evaluate per-year on test + oos (deriv-faithful), one split at a time, chunked predict
    years = {}
    for sp in ("test", "oos"):
        b = load_merged(sp); y, mag, valid, ts = prep_labels(b)
        p, _ = feats_chunked_predict(b, cks_block, L)
        pred = (p > 0.5).astype(int); conf = np.abs(p - 0.5)
        res = eval_book(ts, y, mag, valid, pred, conf, conf_thr)
        # also a no-threshold (full-coverage) moved-bars AUC per year for signal-quality read
        for yr, mask in (("2024", year_of(ts) == 2024), ("2025", year_of(ts) == 2025), ("2026", year_of(ts) == 2026)):
            m = mask & valid & (mag > 0)
            if m.sum() > 50:
                res[yr] = res.get(yr, {}) | {"full_moved_auc": float(roc_auc_score(y[m], p[m])), "full_moved_n": int(m.sum())}
        years[sp] = res
        del b, p, pred, conf, y, mag, valid, ts; gc.collect()
    return {"val_auc_worsthalf": val_auc_worsthalf, "val_auc_full": val_auc_full,
            "conf_thr": conf_thr, "feat_names": feat_names, "per_window": years}

# ----------------------------- (b) SWAP CKS-OFI into the min1 reversion book -----------------------------
def feats_swapped(b):
    """min1 feature builder with the tick-rule OFI `imb` REPLACED by the CKS event-OFI.
    We overwrite b['imb'] with a scale-matched CKS-OFI signal, then call the unmodified min1 feats()
    so EVERY imb-derived feature (imb_ema*, imb_acc, imb_chg, imb_sgn_ac30, imb_runlen, ...) is now
    computed from CKS event-OFI instead of static size imbalance. All other features unchanged."""
    bb = b.copy()
    # CKS 1s signed OFI, squashed to [-1,1] like imb (so downstream EMAs/sign-runs are scale-comparable).
    # tanh of the per-second OFI normalized by a rolling abs-scale -> unit-free, sign-preserving.
    e = bb["cks_e"].astype(float)
    scale = e.abs().ewm(span=300).mean() + 1e-9
    bb["imb"] = np.tanh(e / (3.0 * scale)).astype("float32")
    return min1_feats(bb)

def _fit_dir_ensemble(Xtr, ytr, Xva_half, yva_half, n=2000):
    """Single LGBM all-bars direction model (memory-light substitute for the lgb+xgb+cat trio;
    we compare like-for-like: ONE LGBM on imb-features vs ONE LGBM on CKS-features)."""
    L = mk_lgb(n)
    L.fit(Xtr, ytr, eval_set=[(Xva_half, yva_half)], eval_metric="auc",
          callbacks=[lgb.early_stopping(120), lgb.log_evaluation(0)])
    return L

def run_swap():
    """MEMORY-LEAN (CHUNKED): never build a full-split 62-col feature frame. All feature builds are
    MONTH-BY-MONTH (peak ~one month ~0.7M rows). For each variant:
      1) TRAIN: chunked -> 120k moved-bar fit slice + full-train regime quantiles (bbw1800/rel_ratio
         collected per-month) -> free.
      2) VAL: build small val feats once (1.3M rows, fine) -> fit LGBM on the 120k slice -> freeze
         the min1 reversion gate (qb,rqt) + conf_thr -> free.
      3) TEST/OOS: chunked predict (collect p + bbw1800/rel_ratio/ret300 gate cols per-month) -> eval."""
    print("\n===== (b) SWAP CKS-OFI INTO min1 REVERSION BOOK =====", flush=True)
    GATE_COLS = ("bbw1800", "rel_ratio", "ret300")
    variants = {}
    for tag, fb in (("baseline_imb", min1_feats), ("swap_cks", feats_swapped)):
        # --- TRAIN: chunked 120k fit slice + full-train regime quantiles ---
        _guard(f"b:{tag}:load_train")
        btr = load_merged("train"); ytr, mtr, vtr, tstr = prep_labels(btr)
        XA, fit_rows = feats_chunked_fit(btr, fb, vtr, mtr, tstr, SUBSAMPLE)
        yA = ytr[fit_rows]; feat_names = list(XA.columns)
        # full-train bbw1800/rel_ratio (valid bars) for the regime quantiles, collected chunked
        bbw_acc = []; rel_acc = []; vmask_acc = []
        for per, pos in _month_slices(btr):
            Xm = fb(btr.iloc[pos])
            bbw_acc.append(Xm["bbw1800"].values.astype("float32"))
            rel_acc.append(Xm["rel_ratio"].values.astype("float32"))
            vmask_acc.append(vtr[pos])
            del Xm; gc.collect()
        bbw_tr = np.concatenate(bbw_acc).astype(float); rel_tr = np.concatenate(rel_acc).astype(float)
        vtr_ord = np.concatenate(vmask_acc)
        qb = float(np.nanpercentile(bbw_tr[vtr_ord], 67))
        rq = float(np.nanpercentile(rel_tr[vtr_ord & (bbw_tr <= qb)], 70))
        del btr, ytr, mtr, vtr, tstr, bbw_acc, rel_acc, vmask_acc, bbw_tr, rel_tr, vtr_ord; gc.collect()
        print(f"[b:{tag}] train fit rows={len(XA):,} regime bbw<={qb:.2e} rel>={rq:.3f}", flush=True)
        # --- VAL: small frame, fit + freeze gate/thr ---
        bva = load_merged("val"); yva, mva, vva, tsva = prep_labels(bva)
        iva = np.where(vva & (mva > 0))[0]; iva_sorted = iva[np.argsort(tsva[iva])]
        half = iva_sorted[len(iva_sorted) // 2:]                  # worst-VAL-half early-stop
        Xva = fb(bva)                                             # val 1.3M rows -> ~0.3GB, fine
        L = _fit_dir_ensemble(XA, yA, Xva.iloc[half], yva[half])
        del XA, yA; gc.collect()
        pv = L.predict_proba(Xva)[:, 1]
        val_auc = float(roc_auc_score(yva[iva], pv[iva]))
        bbw = Xva["bbw1800"].values; rel = Xva["rel_ratio"].values; r300 = Xva["ret300"].values
        base = vva & (bbw <= qb) & (rel >= rq)
        rqt = float(np.nanpercentile(rel[base], 80))
        gate = base & (rel >= rqt) & (np.sign(pv - 0.5) == -np.sign(r300))
        conf_thr = float(np.quantile(np.abs(pv[gate] - 0.5), 0.95))
        del bva, Xva, pv, yva, mva, vva, tsva, bbw, rel, r300, base, gate; gc.collect()
        print(f"[b:{tag}] VAL dirAUC={val_auc:.4f} regime bbw<={qb:.2e} rel>={rqt:.3f} conf_thr={conf_thr:.5f}", flush=True)
        # --- TEST/OOS: chunked predict + gate cols, deriv-faithful per-year eval ---
        per = {}
        for sp in ("test", "oos"):
            b = load_merged(sp); y, mag, valid, ts = prep_labels(b)
            p, gcols = feats_chunked_predict(b, fb, L, cols_needed=GATE_COLS)
            pred = (p > 0.5).astype(int); conf = np.abs(p - 0.5)
            gate_b = (gcols["bbw1800"] <= qb) & (gcols["rel_ratio"] >= rqt) & (np.sign(p - 0.5) == -np.sign(gcols["ret300"]))
            per[sp] = eval_book(ts, y, mag, valid, pred, conf, conf_thr, gate_extra=gate_b)
            del b, p, gcols, pred, conf, y, mag, valid, ts, gate_b; gc.collect()
        variants[tag] = {"val_auc": val_auc, "conf_thr": conf_thr, "qb": qb, "rqt": rqt, "per_window": per}
        del L; gc.collect()
    return variants

# ----------------------------- falsifier + main -----------------------------
def _w(d, sp, yr, key):
    try:
        return d["per_window"][sp][yr].get(key)
    except Exception:
        return None

def decide(standalone, swap):
    """PRE-REGISTERED FALSIFIER: kill iff standalone VAL dirAUC<=0.515 AND swap does not lift the
    binding 2025 window with CI clearing the prior (the imb baseline 2025 win-rate)."""
    val_auc = standalone["val_auc_worsthalf"]
    a_fail = val_auc <= 0.515
    # binding 2025 window appears in test split (2025-02..04 + 2025-09..11)
    base = swap["baseline_imb"]["per_window"]["test"]["2025"]
    swp = swap["swap_cks"]["per_window"]["test"]["2025"]
    base_wr = base.get("winrate"); swp_wr = swp.get("winrate"); swp_ci = swp.get("ci", [None, None])
    # "lift with CI clearing the prior": swap 2025 win-rate CI lower-bound > baseline 2025 win-rate
    swap_lifts = (swp_wr is not None and base_wr is not None and swp_ci[0] is not None
                  and swp_wr > base_wr and swp_ci[0] > base_wr)
    killed = a_fail and (not swap_lifts)
    return {"standalone_val_auc_worsthalf": val_auc, "standalone_fail(<=0.515)": bool(a_fail),
            "baseline_2025_winrate": base_wr, "swap_2025_winrate": swp_wr, "swap_2025_ci": swp_ci,
            "swap_lifts_2025_ci_clears_prior": bool(swap_lifts),
            "FALSIFIER_KILLED": bool(killed),
            "verdict": ("KILLED: standalone VAL AUC<=0.515 AND swap did not lift 2025 with CI clearing prior"
                        if killed else
                        ("standalone passed VAL AUC gate" if not a_fail else
                         "standalone failed VAL AUC but swap lifted 2025 -> not killed"))}

def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode in ("build", "all"):
        build_all()
        if mode == "build":
            return
    t0 = time.time()
    standalone = run_standalone()
    swap = run_swap()
    falsifier = decide(standalone, swap)
    out = {"experiment": "CKS event-OFI (Cont-Kukanov-Stoikov) for 60s EURUSD direction",
           "settlement": "deriv-faithful wc_ret: next-tick entry lag, last-tick<=expiry, mid-to-mid, ties LOSE",
           "discipline": "non-overlap chronological, moved-bars-only, per-year, bootstrap CI95, select on worst-VAL-half, subsample<=120k",
           "horizon_s": HS, "gap_s": GAP, "subsample_cap": SUBSAMPLE,
           "splits": {"train": "2021/2022/2023", "val": "2024-04,05",
                      "test": "2024-09..11 + 2025-02..04 + 2025-09..11", "oos": "2026-02..04"},
           "a_standalone": standalone, "b_swap": swap, "falsifier": falsifier,
           "elapsed_s": round(time.time() - t0, 1)}
    json.dump(out, open(RESULT, "w"), indent=2)
    print("\n========== SUMMARY ==========", flush=True)
    print(f"(a) standalone VAL dirAUC worst-half = {standalone['val_auc_worsthalf']:.4f} (full {standalone['val_auc_full']:.4f})")
    for sp in ("test", "oos"):
        for yr in ("2024", "2025", "2026"):
            r = standalone["per_window"][sp].get(yr)
            if r and r.get("n"):
                print(f"    [a {sp} {yr}] n={r['n']} winrate={r['winrate']} CI={r['ci']} moved_auc={r.get('full_moved_auc')}")
    print(f"(b) VAL dirAUC: imb={swap['baseline_imb']['val_auc']:.4f}  cks={swap['swap_cks']['val_auc']:.4f}")
    for tag in ("baseline_imb", "swap_cks"):
        for sp in ("test", "oos"):
            for yr in ("2024", "2025", "2026"):
                r = swap[tag]["per_window"][sp].get(yr)
                if r and r.get("n"):
                    print(f"    [b {tag} {sp} {yr}] n={r['n']} winrate={r['winrate']} CI={r['ci']}")
    print(f"FALSIFIER: {falsifier['verdict']}")
    print(f"result -> {RESULT}  ({out['elapsed_s']}s)")

if __name__ == "__main__":
    main()
