"""EXPERIMENT — CROSS-IMPACT ORDER-FLOW IMBALANCE MATRIX (Cont-Cucuringu-Zhang) for 60s EURUSD direction.

PRE-REGISTERED FALSIFIER (an ~8-12% long-shot; a clean null is the expected, valuable outcome).

HONEST PRIOR (do not relitigate): single-pair CKS event-OFI on EURUSD is NULL at 60s
(min1_cksofi_result.json: VAL dirAUC worst-half 0.4993, moved-bars AUC 0.499-0.501 across 2024/25/26).
The cross-pair RETURN residual reached only 0.516 AUC. This tests one remaining lever: USD-wide informed
order flow that may appear FIRST in OTHER USD legs and lead EURUSD's 60s sign.

THE EXPERIMENT (cross-impact):
  Build signed CKS event-OFI for all 7 majors (EURUSD,GBPUSD,AUDUSD,NZDUSD,USDJPY,USDCHF,USDCAD), express each
  in EURUSD-equivalent USD-direction (flip sign for USD-BASE pairs USDJPY/USDCHF/USDCAD), aggregate to 1s, then
  form a [7-pair x {0,1,2,5s lag}] signed-OFI feature block (+ a few causal transforms). Train ONE LGBM on the
  cross-OFI block to predict EURUSD next-60s sign. Report:
    (a) OFF-DIAGONAL (cross-pair, non-EURUSD) importance contribution + sign-stability across 2024/2025/2026.
    (b) standalone VAL dirAUC + per-year moved-bars accuracy with CI95.

CKS event-OFI per raw tick n (uses n and n-1; e[0]=0):
  e_n = q^bid_n*1[P^bid_n>=P^bid_{n-1}] - q^bid_{n-1}*1[P^bid_n<=P^bid_{n-1}]
        - q^ask_n*1[P^ask_n<=P^ask_{n-1}] + q^ask_{n-1}*1[P^ask_n>=P^ask_{n-1}]
e_n>0 == buy-side pressure on THIS pair's quote == base currency strengthening vs USD for XXXUSD pairs, but
USD strengthening for USDXXX pairs. EURUSD up == EUR up / USD down. So to express every pair as
"+ == EUR-equivalent up (USD down)": KEEP sign for XXXUSD legs (EURUSD,GBPUSD,AUDUSD,NZDUSD), FLIP sign for
USDXXX legs (USDJPY,USDCHF,USDCAD).

THE MIRAGE WE AVOID (non-negotiable): a 7-pair timestamp-INTERSECTION + ffill manufactures ~50% fake-flat
EURUSD bars -> AUC 0.7 that collapses to 0.49 on moved bars. We NEVER intersect timestamps and NEVER ffill a
cross-pair OFI as if it were a price. We keep EURUSD's OWN micro-bar clock (features_tick/, which already drops
empty seconds; verified moved up-rate 0.5006 on val). Each cross-pair's 1s signed-OFI SUM is left-joined onto
that clock and missing seconds are filled with 0.0 (no event in that second == no signed flow, which is the
true value of an OFI sum, NOT a forward-filled stale price). Labels + moved-bars are EURUSD-clean throughout.

DISCIPLINE (deriv-faithful, reused from min1_production / min1_cksofi):
  settlement = wc_ret (next-tick entry lag, last-tick<=expiry, mid-to-mid, ties LOSE); non-overlap chronological;
  MOVED-BARS-ONLY (|ret|>0); per-year 2024/2025/2026; bootstrap CI95; select operating point on VAL by
  worst-VAL-half (NEVER acc-max); subsample fit <=120k; one pair/month at a time, del frames, peak <2GB.

PRE-REGISTERED KILL CONDITION:
  KILL if standalone VAL dirAUC <= 0.515
     OR no window's moved-bars accuracy CI95-lower clears 0.515
     OR the off-diagonal cross-pair gain-importance sign-flips between 2024 and 2026
        (sign of mean signed-OFI*coef contribution, i.e. directional agreement reverses).

Usage:  python min1_xofi.py build   # build the 6 non-EURUSD CKS 1s caches (features_tick_xofi/)
        python min1_xofi.py run      # assemble block + train + eval; writes min1_xofi_result.json
        python min1_xofi.py          # build (if missing) then run
"""
import sys, os, glob, json, time, calendar, gc
import numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

ROOT = "/home/sean/git/binary-algo"
RAW  = "/home/sean/git/raw"
TICK = f"{ROOT}/features_tick"                 # EURUSD 1s micro cache (mid, imb, ...) == the trading clock
CKS  = f"{ROOT}/features_tick_cks"             # cached EURUSD CKS (cks_e, cks_nev) from min1_cksofi.py
XOFI = f"{ROOT}/features_tick_xofi"            # the 6 cross-pair CKS 1s caches we build here
RESULT = f"{ROOT}/min1_xofi_result.json"
os.makedirs(XOFI, exist_ok=True)

# reuse the deriv-faithful settlement, bootstrap, de-overlap, LGBM factory, and constants
from min1_production import wc_ret, boot, nonoverlap_chrono, mk_lgb, HS, TOL_S, ENTRY_LAG_S, GAP
# reuse the EURUSD CKS day-builder + split-date map from the single-pair experiment
from min1_cksofi import _day_cks, SPLIT_DATES, _avail_mb, MIN_AVAIL_MB

# 7 majors. XXXUSD (USD = quote) keep sign; USDXXX (USD = base) flip sign to EUR-equivalent USD-direction.
PAIRS   = ["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDJPY", "USDCHF", "USDCAD"]
USD_BASE = {"USDJPY", "USDCHF", "USDCAD"}     # flip sign for these
SIGN     = {p: (-1.0 if p in USD_BASE else 1.0) for p in PAIRS}
LAGS     = (0, 1, 2, 5)                        # the {0,1,2,5s} lag grid of the cross-impact block
SUBSAMPLE = 120_000

def _guard(where):
    a = _avail_mb()
    if a < MIN_AVAIL_MB:
        raise MemoryError(f"GUARD abort at {where}: avail={a}MB < {MIN_AVAIL_MB}MB")
    return a

# ----------------------------- build the 6 cross-pair CKS 1s caches -----------------------------
def build_pair_split(pair, sp):
    """Build pair's signed CKS event-OFI 1s SUMS for split sp (day-by-day, bounded memory).
    Reuses min1_cksofi._day_cks (identical CKS formula); stores raw cks_e (sign NOT yet applied)."""
    p = f"{XOFI}/{pair}_{sp}_cks1s.parquet"
    if os.path.exists(p):
        print(f"[build] {pair}/{sp}: exists", flush=True); return
    t0 = time.time(); days = []
    for d in SPLIT_DATES[sp]:
        files = sorted(glob.glob(f"{RAW}/{pair}/{pair}_{d}_*.parquet"))
        if not files:
            continue
        dd = _day_cks(files)           # -> DataFrame[cks_e, cks_nev] indexed by UTC second
        if dd is not None:
            days.append(dd)
        gc.collect()
    if not days:
        print(f"[build] {pair}/{sp}: NO DATA", flush=True); return
    b = pd.concat(days); del days; gc.collect()
    _guard(f"build:{pair}:{sp}")
    b.to_parquet(p)
    print(f"[build] {pair}/{sp}: {len(b):,} 1s-bars ({time.time()-t0:.0f}s)", flush=True)
    del b; gc.collect()

def build_all():
    # EURUSD CKS is already cached in features_tick_cks/ (verified). Build the other 6.
    for pair in PAIRS:
        if pair == "EURUSD":
            continue
        for sp in SPLIT_DATES:
            build_pair_split(pair, sp)
    print("[build] CROSS-PAIR CKS CACHE DONE", flush=True)

# ----------------------------- assemble the cross-impact block on the EURUSD clock -----------------------------
def _pair_cks1s(pair, sp):
    """Load a pair's cached signed CKS 1s sums. EURUSD comes from features_tick_cks/, others from XOFI/."""
    fp = f"{CKS}/{sp}_cks1s.parquet" if pair == "EURUSD" else f"{XOFI}/{pair}_{sp}_cks1s.parquet"
    d = pd.read_parquet(fp)[["cks_e"]].astype("float32")
    return d

def load_block(sp):
    """Build the [7-pair x {0,1,2,5s lag}] signed-OFI block on EURUSD's OWN micro-bar clock.

    NO timestamp intersection, NO price ffill. EURUSD micro bars (features_tick/) are the trading clock; each
    cross-pair's 1s signed-OFI SUM is left-joined onto that clock and missing seconds -> 0.0 (no event == no
    signed flow). Lags are in SECONDS on the EURUSD 1s clock. Returns (b_eur, X) where b_eur carries the EURUSD
    mid (for labels) and X is the feature block. Memory: one pair's 1s sum at a time, then dropped."""
    b = pd.read_parquet(f"{TICK}/{sp}_1s.parquet")[["mid"]]    # EURUSD trading clock + mid for labels
    idx = b.index
    X = pd.DataFrame(index=idx)
    diag_cols = {}                                            # base (lag-0) signed OFI per pair, for sign-stability
    for pair in PAIRS:
        e = _pair_cks1s(pair, sp)["cks_e"].astype(float) * SIGN[pair]   # EUR-equivalent USD-direction
        # align this pair's 1s OFI SUM onto the EURUSD clock; missing second -> 0.0 (true OFI value, not a price)
        e_al = e.reindex(idx).fillna(0.0)
        del e; gc.collect()
        for L in LAGS:
            col = f"{pair}_ofi_l{L}"
            X[col] = e_al.shift(L).fillna(0.0).astype("float32")   # causal: only past seconds
        # a couple of cheap causal multi-second sums per pair (still cross-impact, all causal)
        X[f"{pair}_ofi_s5"]  = e_al.rolling(5,  min_periods=1).sum().astype("float32")
        X[f"{pair}_ofi_s15"] = e_al.rolling(15, min_periods=1).sum().astype("float32")
        diag_cols[pair] = f"{pair}_ofi_l0"
        del e_al; gc.collect()
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype("float32")
    return b, X, diag_cols

def _build_block_on_idx(idx, pair_series):
    """Build the cross-impact block for a given EURUSD sub-index (a month slice). pair_series maps
    pair -> its FULL signed-and-aligned 1s OFI series (reindexed to the parent split clock, NaN->0).
    Lags/rolls are computed WITHIN the slice (boundary reset at month edges; <=15s window, negligible)."""
    X = pd.DataFrame(index=idx)
    for pair in PAIRS:
        e_al = pair_series[pair].loc[idx]
        for L in LAGS:
            X[f"{pair}_ofi_l{L}"] = e_al.shift(L).fillna(0.0).astype("float32")
        X[f"{pair}_ofi_s5"]  = e_al.rolling(5,  min_periods=1).sum().astype("float32")
        X[f"{pair}_ofi_s15"] = e_al.rolling(15, min_periods=1).sum().astype("float32")
    return X.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype("float32")

def load_fit_slice(sp, want_rows):
    """MEMORY-LEAN train builder: never holds the full-split 42-col block (train ~13.8M rows -> ~2.3GB
    would OOM the <2GB guard). Builds the block MONTH-BY-MONTH on the EURUSD clock, keeps only moved&valid
    bars, subsamples to ~want_rows total. Cross-pair OFI missing-second->0 (NOT ffill). Returns (Xfit, yfit)."""
    b = pd.read_parquet(f"{TICK}/{sp}_1s.parquet")[["mid"]]
    y, mag, valid, ts = prep_labels(b)
    idx = b.index
    # build each pair's signed+aligned 1s OFI series ONCE on the full clock (1 col float32 each ~55MB) then slice
    pair_series = {}
    for pair in PAIRS:
        e = _pair_cks1s(pair, sp)["cks_e"].astype("float32") * np.float32(SIGN[pair])
        pair_series[pair] = e.reindex(idx).fillna(0.0).astype("float32")
        del e; gc.collect()
    feat_cols = [f"{p}_ofi_l{L}" for p in PAIRS for L in LAGS] + [f"{p}_ofi_s{w}" for p in PAIRS for w in (5, 15)]
    # month slices (contiguous on the monotonic index)
    mp = idx.tz_localize(None).to_period("M")
    codes, uniq = pd.factorize(mp, sort=True)
    per_month = max(1, want_rows // max(1, len(uniq)))
    parts_X = []; parts_y = []
    for k in range(len(uniq)):
        pos = np.where(codes == k)[0]
        sub_idx = idx[pos]
        Xm = _build_block_on_idx(sub_idx, pair_series)[feat_cols]
        vm = valid[pos] & (mag[pos] > 0)
        sel = np.where(vm)[0]
        if len(sel) == 0:
            del Xm; gc.collect(); continue
        if len(sel) > per_month:
            stride = int(np.ceil(len(sel) / per_month)); sel = sel[::stride]
        parts_X.append(Xm.iloc[sel].copy()); parts_y.append(y[pos[sel]])
        del Xm; gc.collect()
    del pair_series; gc.collect()
    Xfit = pd.concat(parts_X); yfit = np.concatenate(parts_y)
    del parts_X, parts_y, b, y, mag, valid, ts; gc.collect()
    return Xfit, yfit, feat_cols

def prep_labels(b):
    """Deriv-faithful 60s labels on the EURUSD micro-bar clock (ties LOSE)."""
    mid = b["mid"].values.astype(float)
    ts = b.index.values.astype("datetime64[s]").astype("int64")
    ret, valid = wc_ret(ts, mid, HS, TOL_S, ENTRY_LAG_S)
    y = (ret > 0).astype(int)
    return y, np.abs(ret), valid, ts

def year_of(ts):
    return pd.to_datetime(ts, unit="s", utc=True).year

# ----------------------------- deriv-faithful per-year eval (moved-bars-only) -----------------------------
def eval_book(ts, y, mag, valid, pred_up, conf, conf_thr):
    """Non-overlap chronological, moved-bars-only, per-year acc + CI95 (mirrors min1_cksofi.eval_book)."""
    base = valid & (conf >= conf_thr)
    tr = nonoverlap_chrono(ts, base)
    out = {}
    yrs = year_of(ts[tr]) if len(tr) else np.array([])
    for label, mask in (("2024", yrs == 2024), ("2025", yrs == 2025), ("2026", yrs == 2026),
                        ("all", np.ones(len(tr), bool))):
        sel = tr[mask] if len(tr) else tr
        if len(sel) == 0:
            out[label] = {"n": 0, "winrate": None, "ci": [None, None],
                          "moved_acc": None, "moved_acc_ci": [None, None], "n_moved": 0}
            continue
        win = ((pred_up[sel] == y[sel]) & (mag[sel] > 0)).astype(float)   # tie => loss
        lo, hi = boot(win)
        moved = mag[sel] > 0
        if moved.sum():
            mv_correct = (pred_up[sel][moved] == y[sel][moved]).astype(float)
            mv_acc = float(mv_correct.mean()); mlo, mhi = boot(mv_correct)
        else:
            mv_acc, mlo, mhi = None, None, None
        out[label] = {"n": int(len(sel)), "winrate": float(win.mean()), "ci": [float(lo), float(hi)],
                      "moved_acc": mv_acc, "moved_acc_ci": [mlo, mhi], "n_moved": int(moved.sum())}
    return out

# ----------------------------- (a)+(b) standalone cross-impact OFI direction -----------------------------
def run():
    print("\n===== CROSS-IMPACT OFI MATRIX -> 60s EURUSD DIRECTION =====", flush=True)
    _guard("run:load_train")
    # TRAIN: chunked month-by-month fit slice (never holds the full ~2.3GB block) -> <=120k moved&valid rows
    XA, yA, feat_names = load_fit_slice("train", SUBSAMPLE)
    # sanity: EURUSD train moved up-rate (must be ~0.49-0.50, NOT a 7-pair-intersection mirage)
    print(f"[fit] train rows={len(XA):,} (cap {SUBSAMPLE:,}) train-moved up-rate={yA.mean():.4f}", flush=True)
    _guard("run:after_train_slice")

    # VAL: small frame; worst-VAL-half early stop + operating point (NEVER acc-max)
    bva, Xva, _ = load_block("val"); Xva = Xva[feat_names]; yva, mva, vva, tsva = prep_labels(bva)
    iva = np.where(vva & (mva > 0))[0]
    val_up = float(yva[iva].mean())
    iva_sorted = iva[np.argsort(tsva[iva])]
    half = iva_sorted[len(iva_sorted) // 2:]                  # worst-VAL-half (chronological 2nd half)
    print(f"[fit] val moved rows={len(iva):,} val-moved up-rate={val_up:.4f}", flush=True)
    L = mk_lgb(2000)
    L.fit(XA, yA, eval_set=[(Xva.iloc[half], yva[half])], eval_metric="auc",
          callbacks=[lgb.early_stopping(120), lgb.log_evaluation(0)])
    del XA, yA; gc.collect()
    pva_half = L.predict_proba(Xva.iloc[half])[:, 1]
    val_auc_worsthalf = float(roc_auc_score(yva[half], pva_half))
    pva_full = L.predict_proba(Xva.iloc[iva])[:, 1]
    val_auc_full = float(roc_auc_score(yva[iva], pva_full))
    conf_thr = float(np.quantile(np.abs(pva_half - 0.5), 0.95))   # 5% coverage on worst-VAL-half
    print(f"[fit] VAL dirAUC worst-half={val_auc_worsthalf:.4f} full={val_auc_full:.4f} "
          f"conf_thr(5%cov)={conf_thr:.4f}", flush=True)
    del bva, Xva, yva, mva, vva, tsva, pva_half, pva_full; gc.collect()

    # importance: gain per feature; split off-diagonal (cross-pair, non-EURUSD) vs diagonal (EURUSD)
    gain = L.booster_.feature_importance(importance_type="gain").astype(float)
    imp = {f: float(g) for f, g in zip(feat_names, gain)}
    tot = sum(imp.values()) + 1e-12
    per_pair_imp = {p: sum(v for f, v in imp.items() if f.startswith(p + "_")) for p in PAIRS}
    offdiag_imp = sum(v for p, v in per_pair_imp.items() if p != "EURUSD")
    print(f"[imp] off-diagonal (cross-pair) gain share = {offdiag_imp / tot:.3f} "
          f"(EURUSD diag share = {per_pair_imp['EURUSD'] / tot:.3f})", flush=True)

    # per-year eval on test + oos + per-year off-diagonal SIGN-STABILITY of the cross-impact contribution.
    # contribution sign per pair per year = sign( mean over moved bars of [ signed_OFI(lag0) * (p-0.5) ] ),
    # i.e. does this pair's EUR-equiv buy pressure agree with the model's up-vote? Stable iff sign is the same
    # 2024 vs 2026 (the falsifier's off-diagonal sign-flip test).
    years = {}
    pair_sign_by_year = {p: {} for p in PAIRS}
    for sp in ("test", "oos"):
        b, X, diag = load_block(sp); X = X[feat_names]; y, mag, valid, ts = prep_labels(b)
        p = L.predict_proba(X)[:, 1].astype("float32")
        pred = (p > 0.5).astype(int); conf = np.abs(p - 0.5)
        res = eval_book(ts, y, mag, valid, pred, conf, conf_thr)
        yr_arr = year_of(ts)
        for yr in (2024, 2025, 2026):
            m = (yr_arr == yr) & valid & (mag > 0)
            if m.sum() > 50:
                res[str(yr)] = res.get(str(yr), {}) | {
                    "full_moved_auc": float(roc_auc_score(y[m], p[m])), "full_moved_n": int(m.sum())}
                # off-diagonal sign per pair this year (vectorized over moved bars in this split-year)
                vote = (p[m] - 0.5)
                for pr in PAIRS:
                    ofi = X[diag[pr]].values[m]
                    contrib = float(np.mean(ofi * vote))
                    pair_sign_by_year[pr].setdefault(str(yr), contrib)
        years[sp] = res
        del b, X, p, pred, conf, y, mag, valid, ts; gc.collect()

    # consolidate per-pair sign across years (prefer test-year value; oos overwrites only 2026 if test lacks it)
    sign_table = {}
    flips_offdiag = []
    for pr in PAIRS:
        d = pair_sign_by_year[pr]
        sign_table[pr] = {yr: d.get(yr) for yr in ("2024", "2025", "2026")}
        s24, s26 = d.get("2024"), d.get("2026")
        if pr != "EURUSD" and s24 is not None and s26 is not None and np.sign(s24) != np.sign(s26) \
           and abs(s24) > 0 and abs(s26) > 0:
            flips_offdiag.append(pr)

    standalone = {
        "val_auc_worsthalf": val_auc_worsthalf, "val_auc_full": val_auc_full,
        "val_moved_uprate": val_up, "conf_thr": conf_thr, "n_features": len(feat_names),
        "per_pair_gain_share": {p: per_pair_imp[p] / tot for p in PAIRS},
        "offdiag_gain_share": offdiag_imp / tot,
        "offdiag_signed_contrib_by_year": {p: sign_table[p] for p in PAIRS if p != "EURUSD"},
        "diag_signed_contrib_by_year": sign_table["EURUSD"],
        "offdiag_sign_flips_2024v2026": flips_offdiag,
        "per_window": years,
    }
    return standalone

# ----------------------------- falsifier -----------------------------
def decide(s):
    """KILL if VAL dirAUC<=0.515 OR no window moved-acc CI-lower clears 0.515 OR off-diag sign-flips 24->26."""
    val_auc = s["val_auc_worsthalf"]
    a_fail = val_auc <= 0.515
    # best moved-acc CI95-lower across all reported windows
    best_lo = None; best_at = None
    for sp in ("test", "oos"):
        for yr in ("2024", "2025", "2026", "all"):
            r = s["per_window"][sp].get(yr)
            if r and r.get("moved_acc_ci") and r["moved_acc_ci"][0] is not None and r.get("n_moved", 0) >= 50:
                lo = r["moved_acc_ci"][0]
                if best_lo is None or lo > best_lo:
                    best_lo, best_at = lo, f"{sp}/{yr}"
    no_window_clears = (best_lo is None) or (best_lo <= 0.515)
    flips = bool(s["offdiag_sign_flips_2024v2026"])
    killed = a_fail or no_window_clears or flips
    reasons = []
    if a_fail: reasons.append(f"VAL dirAUC {val_auc:.4f}<=0.515")
    if no_window_clears: reasons.append(f"no window moved-acc CI-lower>0.515 (best {best_lo} @ {best_at})")
    if flips: reasons.append(f"off-diag sign-flip 2024->2026: {s['offdiag_sign_flips_2024v2026']}")
    return {
        "val_auc_worsthalf": val_auc, "val_auc_fail(<=0.515)": bool(a_fail),
        "best_moved_acc_ci_lower": best_lo, "best_window": best_at,
        "no_window_clears_0.515": bool(no_window_clears),
        "offdiag_sign_flips": s["offdiag_sign_flips_2024v2026"], "offdiag_flipped": flips,
        "FALSIFIER_KILLED": bool(killed),
        "verdict": ("KILLED: " + "; ".join(reasons)) if killed else
                   "SURVIVED: VAL AUC>0.515 AND a window's moved-acc CI-lower>0.515 AND no off-diag sign-flip",
        "beats_single_pair_0.499": bool(val_auc > 0.499 + 0.005),
        "clears_0.65": bool(val_auc >= 0.65),
    }

def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode in ("build", "all"):
        build_all()
        if mode == "build":
            return
    t0 = time.time()
    standalone = run()
    falsifier = decide(standalone)
    out = {
        "experiment": "Cross-impact order-flow imbalance matrix (Cont-Cucuringu-Zhang) for 60s EURUSD direction",
        "settlement": "deriv-faithful wc_ret: next-tick entry lag, last-tick<=expiry, mid-to-mid, ties LOSE",
        "discipline": ("EURUSD-clean clock (NO 7-pair intersection, NO price ffill; cross-pair OFI missing-sec->0); "
                       "non-overlap chronological; moved-bars-only; per-year; bootstrap CI95; "
                       "select on worst-VAL-half; subsample<=120k"),
        "sign_convention": "EUR-equivalent USD-direction: keep XXXUSD sign, FLIP USDJPY/USDCHF/USDCAD",
        "pairs": PAIRS, "lags_s": list(LAGS), "horizon_s": HS, "gap_s": GAP, "subsample_cap": SUBSAMPLE,
        "splits": {"train": "2021/2022/2023", "val": "2024-04,05",
                   "test": "2024-09..11 + 2025-02..04 + 2025-09..11", "oos": "2026-02..04"},
        "standalone": standalone, "falsifier": falsifier, "elapsed_s": round(time.time() - t0, 1),
    }
    json.dump(out, open(RESULT, "w"), indent=2)
    print("\n========== SUMMARY ==========", flush=True)
    print(f"VAL dirAUC worst-half = {standalone['val_auc_worsthalf']:.4f} (full {standalone['val_auc_full']:.4f}); "
          f"val-moved up-rate {standalone['val_moved_uprate']:.4f}")
    print(f"off-diagonal (cross-pair) gain share = {standalone['offdiag_gain_share']:.3f}")
    print(f"off-diag sign-flips 2024->2026 = {standalone['offdiag_sign_flips_2024v2026'] or 'NONE'}")
    for sp in ("test", "oos"):
        for yr in ("2024", "2025", "2026"):
            r = standalone["per_window"][sp].get(yr)
            if r and r.get("n_moved"):
                print(f"  [{sp} {yr}] n={r['n']} winrate={r['winrate']} "
                      f"moved_acc={r.get('moved_acc')} moved_ci={r.get('moved_acc_ci')} "
                      f"full_moved_auc={r.get('full_moved_auc')}")
    print(f"FALSIFIER: {falsifier['verdict']}")
    print(f"result -> {RESULT}  ({out['elapsed_s']}s)")

if __name__ == "__main__":
    main()
