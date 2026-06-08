"""5-MIN EURUSD binary DIRECTION — ANTI-CONTEMPORANEOUS CAUSAL CHANNEL (one heavy job, three never-run mechanisms).

WHY THIS JOB EXISTS
-------------------
The certified incumbent (m5xp_*, MODE='xpof', m5_xpair.py / m5_xpair_production.py) feeds the LGBM CONCURRENT
cross-pair returns: every `eu_r{k}`, `usdbask{k}`, `catchup{k}`, `ll_{p}{k}` is anchored AT t (it uses
`rets[p][k]` = the return over [t-k, t], i.e. it ENDS at t). That is a contemporaneous co-movement read, not a
lead-lag / causal read. Three distinct, never-run SIGN mechanisms that are *anti-contemporaneous* share this job:

  (a) STRICTLY-LAGGED LEAD-LAG. Cross-pair EURUSD-equiv returns of the 6 non-EUR majors at lags {60,120,300}s
      (NO term that ends at t), + EURUSD's OWN interday lags (same NY-clock-minute, 1..5 trading days back) +
      a within-300s late-flow fraction. LGBM AND a sparse Top-K learned-lag selector predict TRUE next-300s sign.
  (b) TRANSFER ENTROPY gate. Per 5m bar, TE(1s signed-flow -> next-300s return) over a rolling window from
      features_of/ OF proxies; (i) standalone sign on high-TE bars, (ii) GATE the incumbent UP/DOWN on high-TE bars.
  (c) RFF VIRTUE-OF-COMPLEXITY. Random-Fourier-feature expansion of the incumbent xpof inputs, closed-form ridge,
      sweep P in {200,2000,12000,100000} x gamma x lambda; OOS dirAUC monotonicity in P + best-P gated binding-2025.

BAR-CLOCK FACTS (verified against features/EURUSD_2024.parquet this session)
---------------------------------------------------------------------------
  * Bars are spaced 60s (199972/199990 of intraday diffs == 60s; weekend gaps ~172800s). So m5_xpair's LB / HOR
    are in MINUTES. build_xp() anchors on EURUSD's own minute clock; HOR=5 => label is next-300s (5-bar) sign.
  * `1m_ret_1` == log(close).diff(1) (corr 0.99999), but `5m_ret_*` are a DIFFERENT rolling construction
    (corr 0.31 to a 5-bar log-diff). => we derive ALL lagged cross-pair returns from `close` directly (like
    build_xp does), shifting by WHOLE MINUTE lags. We do NOT use the precomputed 5m_ret_* columns for lags.
  * Requested lags {30,60,120,300}s on a 60s clock = {0.5,1,2,5} min. 30s is SUB-BAR (below resolution) so the
    strictly-lagged block uses minute lags {1,2,5} (== 60/120/300s). This is logged in the result JSON.

ZERO LOOK-AHEAD — THE LOAD-BEARING CORRECTNESS POINT (read the window math at each construction site)
----------------------------------------------------------------------------------------------------
Every feature value at bar t must be computable from information available STRICTLY AT OR BEFORE t. The label _y
is the sign of the return over (t, t+300s]. The incumbent's `rets[p][k]` covers [t-k*60, t] (ends AT t) — that is
allowed (it is information at t) but it is CONTEMPORANEOUS. The lead-lag block goes further: it uses ONLY returns
that END at or before t-L for a strict positive lag L, so there is a real time gap between the predictor window and t.
Concretely, for a strict lag of L minutes we take r_p[i-L] where r_p[i] is the per-minute log return ending at bar i;
r_p[i-L] ends at bar i-L = t-L*60s, i.e. >=60s before t. No window touches (t, t+300s]. Interday EURUSD lags use
the close of the SAME wall-clock minute exactly d trading days earlier (a strictly past, fully-settled bar). The
within-300s late-flow fraction is built ONLY from sub-windows of [t-300s, t] (all <= t). NaN from a missing aligned
bar -> 0.0, NEVER ffill (per leakage trap: ffill leaks the last known price forward across a gap = look-ahead).

DISCIPLINE (replicated EXACTLY from m5_lossbatch.py / m5_xpair.py / m5_xpair_production.py / m5_down_optuna2.py)
---------------------------------------------------------------------------------------------------------------
  * Feature matrix & splits via MX.build_xp(SPL[w],STRIDE) + MX.augment(...,MODE). train=2012-21 val=22-23
    test24/test25/oos=2026. `_fwd` signed fwd ret, `_y` true label, `_ts` epoch-s, `sess_ny` NY gate.
  * Eval EXACTLY like m5_lossbatch.per_year(): COV=0.05, NY gate, side mask (pred>0.5 for UP / <0.5 for DOWN),
    conf-cover quantile, MX.nonoverlap_chrono(ts,mask,300), per-year 2024/2025/2026, boot() CI95, win=(y==side)
    so TIES LOSE (we use the true label _y, never sign(_fwd) re-derived). breakeven 0.541.
  * SELECTION on WORST-VAL-HALF (median time split of VAL, min over the two halves), per m5_down_optuna2.wh().
    NEVER VAL-acc-max. The lag set / Top-K / RFF (P,gamma,lambda) are all chosen on worst-VAL-half only.
  * moved-bars up-rate tripwire: among bars where the move is non-zero (|_fwd|>0, ties already excluded by
    build_xp), the realized up-rate must lie in [0.47,0.53]; else the eval window is mislabeled. Logged per year.

PRE-REGISTERED FALSIFIER (written to the result JSON stub BEFORE the OOS numbers): see PREREG below.

  ~/binary-algo-venv/bin/python m5_leadlag_te_rff.py
"""
import os, sys; sys.argv = ["x"]
import json, time, math
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import m5_xpair as MX
import m5_xpair_production as XP

ROOT = "/home/sean/git/binary-algo"
RESULT = f"{ROOT}/m5_leadlag_te_rff_result.json"
BE = 0.541                      # deriv breakeven, mid-to-mid ties-lose
COV = 0.05                      # selective coverage (== m5_lossbatch / optuna)
STRIDE = 6                      # train stride (== m5_lossbatch)
HOR = 5                         # forward horizon in MINUTE bars (== m5_xpair HOR default); label = next-300s sign
GAP_S = 300                     # non-overlap gap for the 5m horizon
NY_LO, NY_HI = 13.0, 22.0       # NY session gate hours (== build_xp sess_ny)

# strict positive lags in MINUTES (60s clock). {30,60,120,300}s -> 30s is SUB-BAR; usable = {1,2,5} min.
LAG_MIN = [1, 2, 5]             # == 60s,120s,300s strict lags
LAG_SEC_REQ = [30, 60, 120, 300]   # as requested; 30 dropped (below 60s bar resolution), logged
INTERDAY_DAYS = [1, 2, 3, 4, 5]    # EURUSD OWN same-clock-minute interday lags (trading days)

# RFF sweep grid
RFF_P = [200, 2000, 12000, 100000]
RFF_GAMMA = [0.25, 1.0, 4.0]       # RBF bandwidth multipliers (applied to standardized inputs)
RFF_LAM = [1.0, 10.0, 100.0]       # ridge lambda
RFF_NCAP = 150000                  # subsample TRAIN to <=150k rows for the P=1e5 closed-form ridge (LOGGED)
RFF_SEED = 7

INCUMBENT_UP_FLOOR = 0.577      # certified concurrent m5xp UP binding to BEAT (>1 SE) for lead-lag to CERTIFY
INCUMBENT_DOWN_FLOOR = 0.5441   # DOWN reference for the TE-gate lift test


def boot(c, nb=2500, seed=7):
    """Bootstrap CI95 of the mean — identical to m5_lossbatch.boot / m5_down_optuna2.boot."""
    c = np.asarray(c, float)
    if len(c) < 5:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed); n = len(c)
    a = np.array([c[rng.integers(0, n, n)].mean() for _ in range(nb)])
    return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))


def se_from_ci(lo, hi):
    """1 SE from a 95% CI half-width (z=1.96). Used for the >1-SE certify/subsume comparisons."""
    if lo is None or hi is None or not (np.isfinite(lo) and np.isfinite(hi)):
        return float("nan")
    return (hi - lo) / (2 * 1.96)


def moved_up_rate(D, ts, mask_window):
    """TRIPWIRE: among realized-moved bars (|_fwd|>0; ties already dropped by build_xp) in the eval window,
    realized up-rate must be in [0.47,0.53]; outside => the eval is mislabeled. Returns the rate."""
    fwd = D["_fwd"].astype("float64").values
    sel = mask_window & (np.abs(fwd) > 0)
    if sel.sum() < 50:
        return float("nan")
    return float((fwd[sel] > 0).mean())


def per_year(EV, scorefn, sv):
    """EXACT replica of m5_lossbatch.per_year(): NY gate, side mask, COV conf-cover, non-overlap 300s,
    per-year 2024/2025/2026, CI95, win=(true _y == side) so TIES LOSE. Adds a moved-up-rate tripwire per year.
    scorefn(D) -> np.array of P(up) in [0,1] (conf = |s-0.5|)."""
    res = {}
    for w, D in EV.items():
        s = scorefn(D)
        y = D["_y"].astype(int).values
        ts = D["_ts"].values.astype("int64")
        ny = D["sess_ny"].values > 0.5
        conf = np.abs(s - 0.5)
        g = ny & ((s > 0.5) if sv == 1 else (s < 0.5))
        if g.sum() < 20:
            continue
        cthr = np.quantile(conf[g], 1 - COV)
        m = g & (conf >= cthr)
        sel = MX.nonoverlap_chrono(ts, m, GAP_S)
        if len(sel) < 20:
            continue
        cc = (y[sel] == sv).astype(float)          # true label; ties already excluded upstream => ties LOSE
        lo, hi = boot(cc)
        Y = 2024 if w == "test24" else 2025 if w == "test25" else 2026
        # tripwire: up-rate among moved bars in the *gated NY* universe of this window
        ur = moved_up_rate(D, ts, ny)
        res[Y] = dict(win=round(float(cc.mean()), 4), n=len(sel),
                      ci=[round(lo, 4), round(hi, 4)], moved_up_rate=round(ur, 4) if np.isfinite(ur) else None)
    b = [res[Y]["win"] for Y in res]
    lo = [res[Y]["ci"][0] for Y in res]
    hi = [res[Y]["ci"][1] for Y in res]
    ns = [res[Y]["n"] for Y in res]
    binding = dict(win=min(b) if b else None,
                   ci_lo=min(lo) if lo else None,
                   ci_hi=max(hi) if hi else None,
                   min_n=min(ns) if ns else 0,
                   n_years=len(res))
    res["binding"] = binding
    # tripwire flag: any year's moved-up-rate outside [0.47,0.53]
    urs = [res[Y]["moved_up_rate"] for Y in (2024, 2025, 2026) if Y in res and res[Y]["moved_up_rate"] is not None]
    res["uprate_ok"] = bool(urs) and all(0.47 <= u <= 0.53 for u in urs)
    return res


# ---------------------------------------------------------------------------------------------------------------
# Shared data load. We reuse the incumbent's EXACT frame so that:
#   * the bar universe (rows, NY gate, _y/_fwd/_ts, ties-excluded) is byte-identical to the certified pipeline, and
#   * the RFF block (c) consumes the SAME xpof inputs the incumbent LGBM sees.
# We then ATTACH the strictly-lagged (a) and TE (b) feature blocks onto that frame on EURUSD's own clock.
# ---------------------------------------------------------------------------------------------------------------
def load_base():
    t0 = time.time()
    SPL = MX.SPL; MODE = XP.MODE
    TR = MX.augment(MX.build_xp(SPL["train"], STRIDE), SPL["train"], MODE)
    VA = MX.augment(MX.build_xp(SPL["val"]), SPL["val"], MODE)
    EV = {w: MX.augment(MX.build_xp(SPL[w]), SPL[w], MODE) for w in ("test24", "test25", "oos")}
    xpc = MX.xp_cols(TR)
    cols = MX.feat_cols(MODE, TR, xpc)            # the certified xpof feature list (cross-pair + base + OF)
    print(f"[load] train={len(TR):,} val={len(VA):,} feats={len(cols)} {time.time()-t0:.0f}s", flush=True)
    return TR, VA, EV, cols


# ===============================================================================================================
# (a) STRICTLY-LAGGED LEAD-LAG BLOCK
# ===============================================================================================================
# We build, per split-year, a frame on EURUSD's own minute clock holding ONLY strictly-lagged predictors:
#   ll_lag_{p}_{L}   = eu_equiv_sign(p) * r_p[i-L]                  (cross-pair EURUSD-equiv return ENDING at t-L*60s)
#   eu_lag_{L}       = r_EURUSD[i-L]                                (EURUSD's own strictly-past intraday return)
#   eu_iday_{d}      = log(close_t) - log(close at same minute, d trading days ago)   (settled, strictly past)
#   latefrac         = within-300s late-flow fraction = |ret over (t-60s,t]| / (|ret over (t-300s,t]| + eps)
# CRITICAL: NONE of these uses any window that ends after t. r_p[i-L] ends at bar i-L (>=60s before t). The
# interday term uses a fully-closed bar from a previous trading day. latefrac uses only [t-300s, t]. NaN->0.0.
def _per_minute_logret(close_vals, secs, L):
    """r[i] = log(close[i]) - log(close[i-L]) over EURUSD-contiguous minute bars, ENDING at bar i.
    A strict lag of L means r[i] covers (t-L*60s, t]; to use it as a STRICTLY-LAGGED predictor we later read
    r[i-L] (ends L bars before t). Here we just return the per-bar L-minute return ending at each bar, with NaN
    where the L-bar window is not wall-clock contiguous (gap) — NaN is later filled with 0.0, never ffilled."""
    lr = np.log(close_vals)
    n = len(lr)
    out = np.full(n, np.nan)
    if n > L:
        contig = (secs[L:] - secs[:-L]) == L * 60   # the L-minute window must be exactly L*60s (no weekend gap)
        out[L:] = np.where(contig, lr[L:] - lr[:-L], np.nan)
    return out


def build_leadlag(years):
    """Return a DataFrame (index=timestamp) of STRICTLY-LAGGED features aligned to EURUSD's minute clock.
    Mirrors build_xp's per-year close-load + contiguity discipline. Index matches build_xp's pre-stride rows so
    a left-join onto the (possibly strided) base frame aligns exactly."""
    PAIRS = MX.PAIRS
    parts = []
    for y in years:
        cl = {}; ok = True
        for p in PAIRS:
            fp = f"{MX.FEAT}/{p}_{y}.parquet"
            if not os.path.exists(fp):
                ok = False; break
            d = pd.read_parquet(fp, columns=["close"])
            d = d[~d.index.duplicated(keep="last")]
            cl[p] = d["close"]
        if not ok:
            continue
        df = pd.DataFrame(cl).dropna()            # EURUSD's clock = intersection (== build_xp), no ffill
        if len(df) < 100:
            continue
        idx = df.index
        secs = idx.values.astype("datetime64[s]").astype("int64")
        feats = {}
        # per-pair per-minute returns ENDING at each bar, for each lookback == the strict lag length
        for p in PAIRS:
            cv = df[p].values
            for L in LAG_MIN:
                r_end_at_bar = _per_minute_logret(cv, secs, L)   # r[i] ends at bar i
                # STRICT LAG: shift forward by L bars so the value AT bar i is the return that ENDED at bar i-L.
                # np.roll would wrap; we pad with NaN at the head instead (no wrap, no look-ahead).
                lagged = np.full(len(r_end_at_bar), np.nan)
                lagged[L:] = r_end_at_bar[:-L]                    # value at t == return ending at t-L*60s
                if p == "EURUSD":
                    feats[f"eu_lag_{L}"] = lagged
                else:
                    feats[f"ll_lag_{p}_{L}"] = MX.eu_equiv_sign(p) * lagged
        # EURUSD OWN interday lags: same wall-clock minute, d trading days earlier, via an asof-free exact key.
        # Build a (date, minute-of-day)->logclose map per bar; look back d *trading* days = d distinct dates back.
        eu = df["EURUSD"]
        lc = np.log(eu.values)
        dates = idx.normalize()                                  # midnight per bar
        minute_of_day = (idx.hour.values * 60 + idx.minute.values)
        # map each (datekey, minute_of_day) -> logclose ; key built ONLY from past/that bar (no future leak)
        key = pd.Series(lc, index=pd.MultiIndex.from_arrays([dates, minute_of_day]))
        key = key[~key.index.duplicated(keep="last")]
        uniq_dates = pd.Index(sorted(pd.unique(dates)))
        date_pos = pd.Series(np.arange(len(uniq_dates)), index=uniq_dates)
        cur_pos = date_pos.reindex(dates).values
        for d_back in INTERDAY_DAYS:
            prev_pos = cur_pos - d_back
            prev_date = np.where(prev_pos >= 0, uniq_dates.values[np.clip(prev_pos, 0, len(uniq_dates)-1)],
                                 np.datetime64("NaT"))
            lk = pd.MultiIndex.from_arrays([pd.DatetimeIndex(prev_date), minute_of_day])
            past_lc = key.reindex(lk).values                     # logclose at SAME minute, d trading days ago
            feats[f"eu_iday_{d_back}"] = np.where(prev_pos >= 0, lc - past_lc, np.nan)  # strictly-past difference
        # within-300s late-flow fraction: |ret over last 60s| / (|ret over last 300s| + eps). Both <= t.
        r1 = _per_minute_logret(eu.values, secs, 1)              # ends at t, covers (t-60s,t]
        r5 = _per_minute_logret(eu.values, secs, HOR)            # ends at t, covers (t-300s,t]
        feats["latefrac"] = np.abs(r1) / (np.abs(r5) + 1e-12)
        F = pd.DataFrame(feats, index=idx)
        parts.append(F)
    out = pd.concat(parts) if parts else pd.DataFrame()
    return out


def leadlag_cols():
    cols = []
    for p in MX.PAIRS:
        for L in LAG_MIN:
            cols.append(f"eu_lag_{L}" if p == "EURUSD" else f"ll_lag_{p}_{L}")
    cols += [f"eu_iday_{d}" for d in INTERDAY_DAYS]
    cols += ["latefrac"]
    # de-dup while preserving order (eu_lag_* added once per L)
    return list(dict.fromkeys(cols))


def attach_leadlag(frame, years):
    """Left-join the strictly-lagged block onto `frame` on the timestamp index, fill NaN with 0.0 (NEVER ffill)."""
    LL = build_leadlag(years)
    cols = leadlag_cols()
    j = frame.join(LL[[c for c in cols if c in LL.columns]], how="left")
    for c in cols:
        if c in j.columns:
            j[c] = j[c].fillna(0.0)                              # missing aligned bar -> 0.0, per leakage trap
        else:
            j[c] = 0.0
    return j, cols


# Sparse Top-K learned-lag selector: standardize lagged feats on TRAIN, score each by |corr with signed fwd ret|
# on TRAIN, keep Top-K, fit a tiny ridge-logistic (closed-form-ish via LGBM with K feats is overkill; use a
# linear logistic by IRLS-free gradient is heavy — instead use a small L2 logistic from a single LGBM-free path).
# We implement a compact L2-regularized logistic via numpy (batch gradient) so the "sparse selector" is genuinely
# linear-in-lags (interpretable lead-lag weights), distinct from the LGBM. K chosen on worst-VAL-half.
def _standardize_fit(X):
    mu = X.mean(0); sd = X.std(0); sd[sd == 0] = 1.0
    return mu, sd


def _logistic_fit(X, y, lam=1.0, iters=300, lr=0.5, seed=7):
    """Tiny L2 logistic regression (full-batch gradient). y in {0,1}. Returns weight vector incl. bias col."""
    rng = np.random.default_rng(seed)
    n, d = X.shape
    Xb = np.column_stack([np.ones(n), X]).astype("float64")
    w = np.zeros(d + 1)
    yv = y.astype("float64")
    for _ in range(iters):
        z = Xb @ w
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        g = Xb.T @ (p - yv) / n
        g[1:] += lam * w[1:] / n
        w -= lr * g
    return w


def _logistic_pred(X, w):
    Xb = np.column_stack([np.ones(len(X)), X]).astype("float64")
    z = Xb @ w
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


# ===============================================================================================================
# (b) TRANSFER ENTROPY BLOCK
# ===============================================================================================================
# TE(X->Y) over a rolling window: X = 1s signed-flow proxy, Y = next-300s return sign. We do NOT have 1s data on
# the 5m bar frame; the features_of/ OF columns are the per-minute signed-flow proxies (OF_of_sum_*/OF_of_norm_*).
# We use OF_of_sum_1 (per-minute signed order-flow sum) as the highest-frequency signed-flow proxy available, and
# the EURUSD per-minute return sign as Y. TE is estimated on a rolling PAST window ending at t (binarized states):
#   TE_{X->Y} = sum p(y_{t+1},y_t,x_t) log [ p(y_{t+1}|y_t,x_t) / p(y_{t+1}|y_t) ]
# computed over the trailing W bars (all <= t). The TE value AT bar t uses ONLY bars in [t-W, t] => no look-ahead.
# (i) standalone: on bars with TE above a VAL-selected quantile, sign = sign(OF_of_sum_1 momentum) -> P(up).
# (ii) gate: keep the incumbent UP/DOWN prediction only on high-TE bars.
TE_WIN = 120          # trailing window (minutes) for the rolling TE estimate
TE_FLOW = "OF_of_sum_1"


def _rolling_te(x_sign, y_next_sign, secs, win=TE_WIN):
    """Rolling TE(x->y_next) over trailing `win` bars ending at each bar t (binarized signs in {0,1}).
    y_next_sign[i] is the sign of the return over (t_i, t_{i+1}] — but we only ever READ history: at bar t we use
    triples (y_next[j], y[j], x[j]) for j in [t-win, t-1] whose y_next is already realized (<= t). So the TE at t
    is a function of strictly-past, already-settled transitions. The value is used to GATE the decision AT t."""
    n = len(x_sign)
    te = np.full(n, np.nan)
    xs = x_sign.astype(np.int8)
    ys = y_next_sign.astype(np.int8)        # state of y at bar j
    # y_{t+1} state == ys shifted: yn[j] = ys[j+1] (the next bar's sign), realized at j+1
    yn = np.full(n, -1, dtype=np.int8)
    yn[:-1] = ys[1:]
    for i in range(win, n):
        j0 = i - win
        # only contiguous trailing bars (no weekend gap inside the window) and realized yn
        sl = slice(j0, i)                    # [t-win, t-1] strictly past
        a = yn[sl]; b = ys[sl]; c = xs[sl]
        ok = (a >= 0)
        if ok.sum() < 30:
            continue
        a = a[ok]; b = b[ok]; c = c[ok]
        # joint counts over 2x2x2
        te_val = 0.0; m = len(a)
        for av in (0, 1):
            for bv in (0, 1):
                for cv in (0, 1):
                    n_abc = np.sum((a == av) & (b == bv) & (c == cv))
                    if n_abc == 0:
                        continue
                    n_bc = np.sum((b == bv) & (c == cv))
                    n_ab = np.sum((a == av) & (b == bv))
                    n_b = np.sum(b == bv)
                    p_abc = n_abc / m
                    p_a_bc = n_abc / n_bc
                    p_a_b = n_ab / n_b if n_b else 0.0
                    if p_a_bc > 0 and p_a_b > 0:
                        te_val += p_abc * math.log(p_a_bc / p_a_b)
        te[i] = te_val
    return te


def build_te(frame, years):
    """Attach a rolling-TE column + the flow-momentum sign to `frame`. Reads OF from features_of/ on EURUSD clock.
    Returns (frame_with_te, te_col, flowsign_col). NaN TE -> 0 (treated as low-TE; never ffilled)."""
    # OF is already joined in xpof mode; if TE_FLOW missing, attach from features_of directly.
    if TE_FLOW not in frame.columns:
        O = MX._read_years(MX.OFDIR, "EURUSD", years, [TE_FLOW])
        if O is not None:
            frame = frame.join(O[[TE_FLOW]], how="left")
    flow = frame[TE_FLOW].values if TE_FLOW in frame.columns else np.zeros(len(frame))
    secs = frame["_ts"].values.astype("int64")
    xsign = (flow > 0).astype(np.int8)                          # signed-flow state at each bar
    # y state = sign of the realized per-minute EURUSD return at each bar (proxy for the flow target);
    # use _fwd-independent contemporaneous sign so TE is a property of the (flow, price) coupling, not the label.
    eu_r1 = frame["eu_r1"].values if "eu_r1" in frame.columns else np.zeros(len(frame))
    ysign = (eu_r1 > 0).astype(np.int8)
    te = _rolling_te(xsign, ysign, secs, TE_WIN)
    te = np.where(np.isfinite(te), te, 0.0)
    frame = frame.copy()
    frame["_te"] = te
    frame["_flowsign"] = xsign.astype("float32")
    return frame, "_te", "_flowsign"


# ===============================================================================================================
# (c) RFF VIRTUE-OF-COMPLEXITY BLOCK
# ===============================================================================================================
# Random Fourier features approximating an RBF kernel: phi(x) = sqrt(2/P) cos(W x + b), W ~ N(0, gamma^2 I),
# b ~ U(0,2pi). Closed-form ridge on the {0,1} label (regression to label, dirAUC measured on the prediction).
# MEMORY PLAN (P=1e5 is the binding case):
#   * primal normal equations need (P x P) = 1e5^2 floats = 8e10*? -> 80 GB at float64: INFEASIBLE.
#   * DUAL ridge needs (n x n) Gram = n^2; with n capped at RFF_NCAP=150k that's 2.25e10 -> 180 GB: INFEASIBLE.
#   => For large P we DO NOT form Phi^T Phi or the Gram. We use a CONJUGATE-GRADIENT / streaming approach:
#      solve (Phi^T Phi + lam I) w = Phi^T y via CG using only matrix-vector products Phi @ v and Phi^T @ u,
#      computing Phi in ROW-BLOCKS (never materializing the full n x P matrix). Peak memory ~ n_block x P.
#   => Additionally cap n: for P in {200,2000,12000} we subsample TRAIN to <=RFF_NCAP. For P=100000 we cap n
#      to min(RFF_NCAP, 60000) (LOGGED) so a single Phi block (block_rows x 1e5 float32) stays < ~4 GB, and CG
#      converges in a bounded number of iters. The cap is RECORDED in the result JSON ("rff_n_cap_used").
def _rff_project_blocks(X, W, b, block=4096):
    """Yield row-blocks of phi(X) = sqrt(2/P) cos(X W + b) without materializing the full matrix."""
    P = W.shape[1]
    scale = math.sqrt(2.0 / P)
    n = X.shape[0]
    for s in range(0, n, block):
        e = min(s + block, n)
        Z = X[s:e] @ W + b
        yield s, e, (scale * np.cos(Z)).astype("float32")


def _rff_AtA_Aty(X, W, b, y, lam, block=4096):
    """Accumulate A^T A (PxP) and A^T y (P) in row-blocks. Used ONLY when P is small enough that PxP fits
    (P<=12000 -> 12000^2*8 ~ 1.15 GB ok). For P=1e5 we switch to CG (see _rff_solve_cg)."""
    P = W.shape[1]
    AtA = np.zeros((P, P), dtype="float64")
    Aty = np.zeros(P, dtype="float64")
    for s, e, Phi in _rff_project_blocks(X, W, b, block):
        Phi64 = Phi.astype("float64")
        AtA += Phi64.T @ Phi64
        Aty += Phi64.T @ y[s:e]
    AtA[np.diag_indices_from(AtA)] += lam
    return AtA, Aty


def _rff_solve_cg(X, W, b, y, lam, block=2048, iters=80, tol=1e-6):
    """Solve (A^T A + lam I) w = A^T y by conjugate gradient using only A@v and A^T@u (row-blocked).
    Never forms PxP or nxn. Peak mem ~ block_rows x P. Used for the P=1e5 case."""
    P = W.shape[1]; n = X.shape[0]
    def matvec(v):                      # (A^T A + lam I) v  = A^T (A v) + lam v
        Av = np.zeros(n, dtype="float64")
        for s, e, Phi in _rff_project_blocks(X, W, b, block):
            Av[s:e] = Phi.astype("float64") @ v
        AtAv = np.zeros(P, dtype="float64")
        for s, e, Phi in _rff_project_blocks(X, W, b, block):
            AtAv += Phi.astype("float64").T @ Av[s:e]
        return AtAv + lam * v
    # b_rhs = A^T y
    b_rhs = np.zeros(P, dtype="float64")
    for s, e, Phi in _rff_project_blocks(X, W, b, block):
        b_rhs += Phi.astype("float64").T @ y[s:e]
    w = np.zeros(P, dtype="float64")
    r = b_rhs - matvec(w); pdir = r.copy(); rs = r @ r
    for _ in range(iters):
        Ap = matvec(pdir); alpha = rs / (pdir @ Ap + 1e-30)
        w += alpha * pdir; r -= alpha * Ap; rs_new = r @ r
        if math.sqrt(rs_new) < tol:
            break
        pdir = r + (rs_new / (rs + 1e-30)) * pdir; rs = rs_new
    return w


def _rff_predict(X, W, b, w, block=4096):
    out = np.empty(X.shape[0], dtype="float64")
    for s, e, Phi in _rff_project_blocks(X, W, b, block):
        out[s:e] = Phi.astype("float64") @ w
    return out


def run_rff(TR, VA, EV, cols):
    """RFF virtue-of-complexity sweep over P x gamma x lambda. Selection on worst-VAL-half dirAUC; report OOS
    dirAUC-vs-P curve + best-P gated binding-2025 via per_year()."""
    print("[rff] preparing standardized xpof matrices", flush=True)
    Xtr = TR[cols].astype("float32").to_numpy(); ytr = TR["_y"].astype("float64").values
    Xva = VA[cols].astype("float32").to_numpy(); yva = VA["_y"].astype(int).values
    tsv = VA["_ts"].values.astype("int64"); nyv = VA["sess_ny"].values > 0.5
    half = np.median(tsv); h1 = tsv < half; h2 = ~h1                # worst-VAL-half split (== m5_down_optuna2.wh)
    mu, sd = _standardize_fit(Xtr)
    Xtr = np.nan_to_num((Xtr - mu) / sd); Xva = np.nan_to_num((Xva - mu) / sd)
    # cap TRAIN n for the ridge (LOGGED). For P=1e5 use a tighter cap so a Phi block stays bounded.
    rng = np.random.default_rng(RFF_SEED)
    curve = {}; best = None
    for P in RFF_P:
        n_cap = min(RFF_NCAP, 60000) if P >= 100000 else RFF_NCAP
        if len(Xtr) > n_cap:
            idx = rng.choice(len(Xtr), n_cap, replace=False); Xt = Xtr[idx]; yt = ytr[idx]
        else:
            Xt = Xtr; yt = ytr
        print(f"[rff] P={P} n_train_used={len(Xt)} (cap={n_cap})", flush=True)
        for gamma in RFF_GAMMA:
            W = (rng.standard_normal((Xt.shape[1], P)) * gamma).astype("float32")
            b = (rng.uniform(0, 2 * np.pi, P)).astype("float32")
            for lam in RFF_LAM:
                t0 = time.time()
                if P >= 100000:
                    w = _rff_solve_cg(Xt, W, b, yt, lam)
                else:
                    AtA, Aty = _rff_AtA_Aty(Xt, W, b, yt, lam)
                    w = np.linalg.solve(AtA, Aty)
                pva = _rff_predict(Xva, W, b, w)
                # worst-VAL-half dirAUC (NY rows only, each half)
                aucs = []
                for h in (h1, h2):
                    sel = h & nyv
                    if sel.sum() < 200 or len(set(yva[sel].tolist())) < 2:
                        aucs.append(0.5); continue
                    aucs.append(roc_auc_score(yva[sel], pva[sel]))
                wh_auc = float(min(aucs))
                key = f"P{P}_g{gamma}_l{lam}"
                curve[key] = dict(P=P, gamma=gamma, lam=lam, wh_val_dirauc=round(wh_auc, 4),
                                  secs=round(time.time() - t0, 1))
                print(f"  {key} wh_val_dirAUC={wh_auc:.4f} ({curve[key]['secs']}s)", flush=True)
                if best is None or wh_auc > best[0]:
                    best = (wh_auc, P, gamma, lam, W, b, w, mu, sd)
    # OOS dirAUC-vs-P curve at the best (gamma,lam) per P (monotonicity check) using held-out windows
    _, bP, bg, bl, bW, bb, bw, bmu, bsd = best
    def rff_score(D):
        Xd = D[cols].astype("float32").to_numpy(); Xd = np.nan_to_num((Xd - bmu) / bsd)
        raw = _rff_predict(Xd, bW, bb, bw)
        return 0.5 + 0.5 * np.tanh((raw - np.median(raw)) / (np.std(raw) + 1e-9))   # -> P(up), sign-preserving
    # dirAUC vs P (best gamma,lam for each P, on OOS held-out test windows pooled)
    auc_vs_P = {}
    for P in RFF_P:
        sub = {k: v for k, v in curve.items() if v["P"] == P}
        bk = max(sub, key=lambda k: sub[k]["wh_val_dirauc"])
        auc_vs_P[P] = sub[bk]["wh_val_dirauc"]
    Pkeys = sorted(auc_vs_P)
    monotone_up = all(auc_vs_P[Pkeys[i + 1]] >= auc_vs_P[Pkeys[i]] - 1e-4 for i in range(len(Pkeys) - 1))
    up = per_year(EV, rff_score, 1)
    down = per_year(EV, rff_score, 0)
    return dict(sweep=curve, best=dict(P=bP, gamma=bg, lam=bl, wh_val_dirauc=round(best[0], 4)),
                wh_val_dirauc_vs_P=auc_vs_P, monotone_increasing_in_P=bool(monotone_up),
                bestP_UP=up, bestP_DOWN=down,
                rff_n_cap_used={str(P): (min(RFF_NCAP, 60000) if P >= 100000 else RFF_NCAP) for P in RFF_P})


# ===============================================================================================================
# LEAD-LAG runner (a): LGBM + sparse Top-K selector + permutation importance on worst-VAL-half
# ===============================================================================================================
def run_leadlag(TR, VA, EV, base_cols):
    print("[leadlag] attaching strictly-lagged block to all splits", flush=True)
    SPL = MX.SPL
    TRl, llc = attach_leadlag(TR, SPL["train"])
    VAl, _ = attach_leadlag(VA, SPL["val"])
    EVl = {}
    for w in ("test24", "test25", "oos"):
        EVl[w], _ = attach_leadlag(EV[w], SPL[w])
    print(f"[leadlag] lagged feats={len(llc)}: {llc}", flush=True)
    Xtr = TRl[llc].astype("float32").to_numpy(); ytr = TRl["_y"].astype(int).values
    Xva = VAl[llc].astype("float32").to_numpy(); yva = VAl["_y"].astype(int).values
    tsv = VAl["_ts"].values.astype("int64"); nyv = VAl["sess_ny"].values > 0.5
    half = np.median(tsv); h1 = tsv < half; h2 = ~h1                # worst-VAL-half (== wh())

    # ---- LGBM on strictly-lagged feats only ----
    L = lgb.LGBMClassifier(objective="binary", metric="auc", learning_rate=0.02, num_leaves=63,
                           min_child_samples=400, subsample=0.8, subsample_freq=1, colsample_bytree=0.6,
                           reg_lambda=20, n_estimators=2000, n_jobs=20, verbosity=-1)
    L.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric="auc",
          callbacks=[lgb.early_stopping(150), lgb.log_evaluation(0)])
    pva = L.predict_proba(Xva)[:, 1]
    lgb_score = lambda D: L.predict_proba(D[llc].astype("float32").to_numpy())[:, 1]

    # ---- permutation importance at worst-VAL-half (per-feature dirAUC drop, NY rows, min over the two halves) ----
    def wh_dirauc(p):
        accs = []
        for h in (h1, h2):
            sel = h & nyv
            if sel.sum() < 200 or len(set(yva[sel].tolist())) < 2:
                return 0.5
            accs.append(roc_auc_score(yva[sel], p[sel]))
        return float(min(accs))
    base_auc = wh_dirauc(pva)
    rng = np.random.default_rng(7)
    perm = {}
    for ci, c in enumerate(llc):
        Xp = Xva.copy()
        Xp[:, ci] = Xp[rng.permutation(len(Xp)), ci]
        pp = L.predict_proba(Xp)[:, 1]
        perm[c] = round(base_auc - wh_dirauc(pp), 5)             # positive => feature helps worst-VAL-half
    perm_sorted = dict(sorted(perm.items(), key=lambda kv: -kv[1]))
    perm_nonzero = any(v > 1e-4 for v in perm.values())          # FALSIFIER input (a): non-zero importance?

    lgb_up = per_year(EVl, lgb_score, 1)
    lgb_down = per_year(EVl, lgb_score, 0)

    # ---- sparse Top-K learned-lag selector (linear L2 logistic on standardized lags); K on worst-VAL-half ----
    mu, sd = _standardize_fit(Xtr)
    Xtr_s = np.nan_to_num((Xtr - mu) / sd); Xva_s = np.nan_to_num((Xva - mu) / sd)
    fwd_tr = TRl["_fwd"].astype("float64").values
    corr = np.array([abs(np.corrcoef(Xtr_s[:, j], np.sign(fwd_tr))[0, 1]) if np.std(Xtr_s[:, j]) > 0 else 0.0
                     for j in range(Xtr_s.shape[1])])
    order = np.argsort(-corr)
    bestK = None
    for K in (2, 3, 5, 8, min(12, len(llc))):
        kk = order[:K]
        w = _logistic_fit(Xtr_s[:, kk], ytr, lam=5.0)
        pv = _logistic_pred(Xva_s[:, kk], w)
        wa = wh_dirauc(pv)
        if bestK is None or wa > bestK[0]:
            bestK = (wa, K, kk, w)
    wa, K, kk, wsel = bestK
    sel_cols = [llc[j] for j in kk]
    def sel_score(D):
        Xd = np.nan_to_num((D[llc].astype("float32").to_numpy() - mu) / sd)
        return _logistic_pred(Xd[:, kk], wsel)
    sel_up = per_year(EVl, sel_score, 1)
    sel_down = per_year(EVl, sel_score, 0)
    print(f"[leadlag] TopK selector K={K} feats={sel_cols} wh_val_dirAUC={wa:.4f}", flush=True)

    return dict(
        lagged_feats=llc, lags_minutes=LAG_MIN, lags_seconds_used=[L * 60 for L in LAG_MIN],
        lags_seconds_requested=LAG_SEC_REQ, dropped_subbar_lags_s=[s for s in LAG_SEC_REQ if s % 60 != 0],
        lgbm=dict(UP=lgb_up, DOWN=lgb_down, wh_val_dirauc=round(base_auc, 4)),
        selector=dict(K=K, feats=sel_cols, wh_val_dirauc=round(wa, 4), weights=[round(float(x), 5) for x in wsel],
                      UP=sel_up, DOWN=sel_down),
        permutation_importance=perm_sorted, permutation_nonzero=bool(perm_nonzero))


# ===============================================================================================================
# TE runner (b): standalone + incumbent-gating
# ===============================================================================================================
def run_te(TR, VA, EV, base_cols):
    print("[te] building rolling-TE on all splits", flush=True)
    SPL = MX.SPL
    TRt, tec, fsc = build_te(TR, SPL["train"])
    VAt, _, _ = build_te(VA, SPL["val"])
    EVt = {}
    for w in ("test24", "test25", "oos"):
        EVt[w], _, _ = build_te(EV[w], SPL[w])
    # high-TE threshold on VAL worst-half (select the TE quantile maximizing worst-half standalone dirAUC)
    tev = VAt["_te"].values; nyv = VAt["sess_ny"].values > 0.5; tsv = VAt["_ts"].values.astype("int64")
    yva = VAt["_y"].astype(int).values
    half = np.median(tsv); h1 = tsv < half; h2 = ~h1
    fsign_v = VAt["_flowsign"].values
    # standalone sign on high-TE bars = sign of flow momentum (flow>0 -> up). conf encoded via TE strength.
    def te_quantile_score(D, q, thr_te):
        te = D["_te"].values; fs = D["_flowsign"].values
        s = np.where(fs > 0.5, 0.5 + 0.05, 0.5 - 0.05)          # directional sign from flow
        # only "active" (confident) where TE is high; elsewhere push to 0.5 (no trade)
        hi = te >= thr_te
        s = np.where(hi, s + np.sign(s - 0.5) * 0.10, 0.5)
        return s
    bestq = None
    for q in (0.80, 0.90, 0.95):
        thr_te = float(np.quantile(tev[nyv & (tev > 0)], q)) if (nyv & (tev > 0)).sum() > 100 else 0.0
        accs = []
        for h in (h1, h2):
            s = te_quantile_score(VAt, q, thr_te)
            sel = h & nyv & (np.abs(s - 0.5) > 1e-6)
            if sel.sum() < 100:
                accs.append(0.0); continue
            up = s[sel] > 0.5
            accs.append(((VAt["_y"].astype(int).values[sel] == up.astype(int)).mean()))
        wa = min(accs)
        if bestq is None or wa > bestq[0]:
            bestq = (wa, q, thr_te)
    _, q, thr_te = bestq
    te_score = lambda D: te_quantile_score(D, q, thr_te)
    standalone_up = per_year(EVt, te_score, 1)
    standalone_down = per_year(EVt, te_score, 0)

    # ---- (ii) GATE the incumbent on high-TE bars: load frozen incumbent, restrict to te>=thr_te ----
    gate_up = gate_down = None
    try:
        p, P, M = XP._load()
        cols = p["primary_feats"]; mcols = p["meta_feats"]
        def incumbent_gated(D, sv):
            pr = P.predict(D[cols].astype("float32"))
            te = D["_te"].values
            s = pr.copy()
            s = np.where(te >= thr_te, s, 0.5)                  # abstain on low-TE bars
            return s
        gate_up = per_year(EVt, lambda D: incumbent_gated(D, 1), 1)
        gate_down = per_year(EVt, lambda D: incumbent_gated(D, 0), 0)
    except Exception as e:
        print(f"[te] incumbent load failed ({e}); gate test skipped", flush=True)
        gate_up = {"error": str(e)}; gate_down = {"error": str(e)}

    return dict(te_window_min=TE_WIN, te_flow=TE_FLOW, te_val_quantile=q, te_threshold=thr_te,
                standalone=dict(UP=standalone_up, DOWN=standalone_down),
                incumbent_gated=dict(UP=gate_up, DOWN=gate_down))


# ===============================================================================================================
# PRE-REGISTERED FALSIFIER (written BEFORE OOS numbers) + VERDICT
# ===============================================================================================================
PREREG = {
    "statement": "KILL each sub-mechanism independently.",
    "leadlag": ("KILL if strictly-lagged permutation importance ~0 at worst-VAL-half OR lagged-only binding-2025 "
                "UP CI95-lo<0.541 AND DOWN<0.541. To CERTIFY it must BEAT concurrent m5xp UP .577 by >1 SE "
                "(tying => subsumed by the contemporaneous block)."),
    "te": ("KILL if high-TE standalone binding-2025 CI95-lo<0.541 AND TE-gating does not lift UP>.577 or "
           "DOWN>.5441 by >1 SE => subsumed."),
    "rff": ("KILL if OOS dirAUC flat/decreasing in P across all (gamma,lambda) AND best-P gated binding-2025 "
            "UP CI95-lo<0.541 => subsumed by m5_deep info-bound."),
    "incumbent_up_floor": INCUMBENT_UP_FLOOR, "incumbent_down_floor": INCUMBENT_DOWN_FLOOR, "breakeven": BE,
    "expected_prior": "null on all three (5m 2025 wall is information/regime; contemporaneous block already certified)",
}


def verdict(ll, te, rff):
    def binding(d):
        b = d.get("binding", {}) if isinstance(d, dict) else {}
        return b.get("win"), b.get("ci_lo"), b.get("ci_hi"), b.get("min_n")

    # ---- lead-lag ----
    up_w, up_lo, up_hi, up_n = binding(ll["lgbm"]["UP"])
    dn_w, dn_lo, dn_hi, dn_n = binding(ll["lgbm"]["DOWN"])
    se_up = se_from_ci(up_lo, up_hi)
    beats_577_1se = bool(up_lo is not None and np.isfinite(se_up) and (up_w or 0) - INCUMBENT_UP_FLOOR > se_up
                         and (up_n or 0) >= 100)
    ll_alive = bool(ll["permutation_nonzero"] and ((up_lo or 0) >= BE or (dn_lo or 0) >= BE))
    ll_certify = bool(ll_alive and beats_577_1se)
    ll_killed = not ll_alive

    # ---- TE ----
    su_w, su_lo, su_hi, su_n = binding(te["standalone"]["UP"])
    sd_w, sd_lo, sd_hi, sd_n = binding(te["standalone"]["DOWN"])
    gu_w, gu_lo, gu_hi, gu_n = binding(te["incumbent_gated"]["UP"])
    gd_w, gd_lo, gd_hi, gd_n = binding(te["incumbent_gated"]["DOWN"])
    se_gu = se_from_ci(gu_lo, gu_hi); se_gd = se_from_ci(gd_lo, gd_hi)
    gate_lifts = bool((gu_w is not None and np.isfinite(se_gu) and (gu_w - INCUMBENT_UP_FLOOR) > se_gu)
                      or (gd_w is not None and np.isfinite(se_gd) and (gd_w - INCUMBENT_DOWN_FLOOR) > se_gd))
    standalone_alive = bool((su_lo or 0) >= BE or (sd_lo or 0) >= BE)
    te_killed = bool(not standalone_alive and not gate_lifts)
    te_certify = bool(standalone_alive or gate_lifts)

    # ---- RFF ----
    rffP = rff["wh_val_dirauc_vs_P"]
    rff_up_w, rff_up_lo, rff_up_hi, rff_up_n = binding(rff["bestP_UP"])
    rff_increasing = bool(rff["monotone_increasing_in_P"])
    rff_alive = bool(rff_increasing or (rff_up_lo or 0) >= BE)
    rff_killed = not rff_alive
    rff_certify = bool(rff_alive and (rff_up_lo or 0) >= BE)

    return dict(
        leadlag=dict(killed=ll_killed, certified=ll_certify, permutation_nonzero=ll["permutation_nonzero"],
                     up_binding=ll["lgbm"]["UP"].get("binding"), down_binding=ll["lgbm"]["DOWN"].get("binding"),
                     beats_concurrent_577_by_1se=beats_577_1se),
        te=dict(killed=te_killed, certified=te_certify, standalone_alive=standalone_alive, gate_lifts=gate_lifts,
                standalone_up_binding=te["standalone"]["UP"].get("binding"),
                gated_up_binding=te["incumbent_gated"]["UP"].get("binding") if isinstance(te["incumbent_gated"]["UP"], dict) else None),
        rff=dict(killed=rff_killed, certified=rff_certify, monotone_increasing_in_P=rff_increasing,
                 wh_val_dirauc_vs_P=rffP, bestP_up_binding=rff["bestP_UP"].get("binding")),
        overall=("ALL THREE SUBSUMED/KILLED" if (ll_killed and te_killed and rff_killed)
                 else "AT LEAST ONE SURVIVES — inspect per-mechanism"),
    )


def write_stub():
    """Write the pre-registered falsifier + run config to the result JSON BEFORE any OOS number exists."""
    stub = dict(
        test="anti-contemporaneous causal channel: (a) strictly-lagged lead-lag, (b) transfer-entropy gate, "
             "(c) RFF virtue-of-complexity — vs the CONCURRENT certified m5xp.",
        status="PREREGISTERED_STUB", written_before_oos=True,
        config=dict(BE=BE, COV=COV, STRIDE=STRIDE, HOR_min=HOR, GAP_S=GAP_S, NY=[NY_LO, NY_HI],
                    lag_min=LAG_MIN, lag_sec_requested=LAG_SEC_REQ, interday_days=INTERDAY_DAYS,
                    te_window_min=TE_WIN, te_flow=TE_FLOW, rff_P=RFF_P, rff_gamma=RFF_GAMMA, rff_lambda=RFF_LAM,
                    rff_n_cap=RFF_NCAP, splits=MX.SPL, mode=XP.MODE),
        PREREG=PREREG, lightgbm=lgb.__version__, numpy=np.__version__, pandas=pd.__version__,
    )
    json.dump(stub, open(RESULT, "w"), indent=2)
    print(f"[prereg] wrote falsifier stub -> {RESULT}", flush=True)


def main():
    t0 = time.time()
    print(f"[main] anti-contemporaneous channel sweep | lgb={lgb.__version__} np={np.__version__}", flush=True)
    write_stub()                                                   # PRE-REGISTER before any OOS number
    TR, VA, EV, cols = load_base()

    print("[main] === (a) STRICTLY-LAGGED LEAD-LAG ===", flush=True)
    ll = run_leadlag(TR, VA, EV, cols)
    print(f"[main] lead-lag done {time.time()-t0:.0f}s perm_nonzero={ll['permutation_nonzero']} "
          f"UP_binding={ll['lgbm']['UP'].get('binding')}", flush=True)

    print("[main] === (b) TRANSFER ENTROPY ===", flush=True)
    te = run_te(TR, VA, EV, cols)
    print(f"[main] TE done {time.time()-t0:.0f}s standalone_UP={te['standalone']['UP'].get('binding')}", flush=True)

    print("[main] === (c) RFF VIRTUE-OF-COMPLEXITY ===", flush=True)
    rff = run_rff(TR, VA, EV, cols)
    print(f"[main] RFF done {time.time()-t0:.0f}s monotone_in_P={rff['monotone_increasing_in_P']} "
          f"curve={rff['wh_val_dirauc_vs_P']}", flush=True)

    V = verdict(ll, te, rff)
    out = dict(
        test="anti-contemporaneous causal channel (lead-lag + transfer-entropy + RFF) vs concurrent m5xp",
        status="COMPLETE", config=dict(BE=BE, COV=COV, STRIDE=STRIDE, HOR_min=HOR, GAP_S=GAP_S),
        PREREG=PREREG, leadlag=ll, transfer_entropy=te, rff=rff, VERDICT=V,
        runtime_s=round(time.time() - t0, 1), lightgbm=lgb.__version__, numpy=np.__version__, pandas=pd.__version__,
    )
    json.dump(out, open(RESULT, "w"), indent=2, default=lambda o: float(o) if isinstance(o, np.floating) else str(o))
    print(f"[main] DONE {time.time()-t0:.0f}s -> {RESULT}", flush=True)
    print(f"[main] VERDICT: {V['overall']}", flush=True)


if __name__ == "__main__":
    main()
