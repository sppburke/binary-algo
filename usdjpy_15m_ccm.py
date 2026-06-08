"""USDJPY 15-MIN binary DIRECTION — C4 CONVERGENT CROSS-MAPPING lever (lagged, causal).

SCOPE: USDJPY · 15m · NY session. Tests the C4 hypothesis: does any OTHER USD-major's 15m return
CONVERGENTLY CROSS-MAP USDJPY's next-15m return *sign*? CCM (Sugihara 2012; lagged variant Ye 2015)
detects directional dynamical coupling X->Y by asking whether Y's time-delay shadow manifold can
recover X with skill that converges as library length grows. Here we run a TRACTABLE, *predictive*
CCM: delay-embed each candidate driver, kNN cross-map to forecast USDJPY's forward 15m return, and use
the SIGN of that cross-mapped forecast as the direction signal for the best-converging driver
(chosen on TRAIN ONLY). Subsampled heavily for tractability.

SIGN-INVARIANCE PRIOR (arXiv:2512.15720): coupling/complexity gates are expected to gate move SIZE,
not SIGN. The honest expectation is a null (AUC ~ 0.50). We RUN the direction test faithfully anyway
and let the falsifier decide: KILL if held-out moved-AUC <= 0.51 (coupling gates magnitude, not sign).

LEAKAGE / MIRAGE FIREWALL (from min1_ccm.py CCM_DESIGN.md):
  * Embedding is built ONLY from CAUSAL PAST returns of the driver (the delay vector at time t uses
    r[t], r[t-tau], ... r[t-(E-1)tau]); the cross-map predicts USDJPY's FORWARD return r_fwd (lagged,
    tp>=0 steps ahead). The library (training manifold) is TRAIN-ONLY; held-out points are predicted
    against the frozen TRAIN library. No held-out target ever enters its own neighbor set.
  * Theiler window excludes temporally-trivial neighbors. Driver selection (which pair, E, tp) is on
    TRAIN only, frozen before any held-out read.
  * The EVAL LABEL is ALWAYS the deriv-faithful fixed-15m sign from build(): sign(close[t+15]-close[t]),
    ties (move==0) LOSING. DECISION rows restricted to NY (features stay causal/continuous). Per
    held-out year: moved-AUC of the CCM signal vs up/down label on NY moved bars + selective win-rate
    at cov3% via side_eval (nonoverlap gap=900 built in).

INCUMBENT to beat: certified NY own-pair GBM signal AUC ~.539 / win-rate ~.58-.60. SURVIVES only if
held-out moved-AUC exceeds ~.539 OR NY cov3% win-rate CI-lower clears 0.541 in >=2 years.

Usage:  ~/binary-algo-venv/bin/python usdjpy_15m_ccm.py
"""
import os, sys, json, time, gc
import numpy as np, pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

import harness as H
from sessions import session_mask
from usdjpy_15m_base import build, side_eval, covcurve, nonoverlap_chrono, boot, BE

ROOT   = "/home/sean/git/binary-algo"
FEAT   = H.FEAT_DIR
KEY    = "ccm"
RESULT = f"{ROOT}/usdjpy_15m_{KEY}_result.json"

TARGET = "USDJPY"
HOR, STEP, GAP = 15, 60, 15 * 60          # 15-min horizon -> 900s, nonoverlap gap 900s
# candidate drivers: the 6 OTHER USD-majors (USDJPY itself is the target). EUR-direction sign is
# IRRELEVANT to the cross-map (the target is always USDJPY's forward return); CCM picks the best
# converging driver regardless of the driver's own sign convention.
DRIVERS = ["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDCHF", "USDCAD"]

# CCM params (CCM_DESIGN.md S2): E=3-4, tau=1 step (= 1 native 60s bar), E+1 neighbors, exp weights,
# Theiler window. tp = forward cross-map lag in BARS (>=0 means driver's past predicts USDJPY future).
E_GRID  = [3, 4]
TAU     = 1
TP_GRID = [0, 1, 3]                       # forward lags (bars) -> 0/60/180s lead; tp>=0 keeps it causal
L_GRID  = [200, 400, 800, 1600]           # library lengths for the convergence check (TRAIN only)
TRAIN_LIB_CAP = 8000                      # max TRAIN manifold rows kept (random subsample of VALID
                                          # native-clock indices) — the kNN-tractability lever
TRAIN_SCORE_N = 4000                      # TRAIN cross-map targets scored when selecting driver/E/tp
# NB: tractability is achieved by random subsampling of VALID native-clock indices inside the CCM core
# (TRAIN_LIB_CAP / TRAIN_SCORE_N), NOT by row-striding the merged frame — the delay embedding must run
# on the contiguous 60s clock or it silently embeds a multi-bar-spaced trajectory.
RNG = np.random.default_rng(7)

SPL = {"train": [str(y) for y in range(2012, 2022)], "val": ["2022", "2023"],
       "test24": ["2024"], "test25": ["2025"], "oos": ["2026"]}


# ----------------------------- close-only data (low RAM) -----------------------------
def load_returns(pair, years):
    """Per-pair forward+past returns on the native 60s clock, contiguity-masked.
    Returns dict with: ts(int64), r_past (log-return r[t]-r[t-1] aligned to t, causal),
    r_fwd (close[t+HOR]/close[t]-1 forward 15m return), contig(bool, ts[t+HOR]-ts[t]==GAP),
    step_ok(bool, ts[t]-ts[t-1]==STEP so the delay vector is gap-free).
    NO price ever ffill'd; rows live on the genuine bar clock."""
    Ts, RP, RF, CT, SK = [], [], [], [], []
    for y in years:
        p = f"{FEAT}/{pair}_{y}.parquet"
        if not os.path.exists(p):
            continue
        d = pd.read_parquet(p, columns=["close"])
        d = d[~d.index.duplicated(keep="last")]
        c = d["close"].values.astype(float)
        ts = d.index.values.astype("datetime64[s]").astype("int64")
        n = len(d)
        lc = np.log(c)
        r_past = np.full(n, np.nan); r_past[1:] = np.diff(lc)              # causal 1-bar log-return at t
        step_ok = np.zeros(n, bool); step_ok[1:] = (ts[1:] - ts[:-1]) == STEP
        r_fwd = np.full(n, np.nan); r_fwd[:n - HOR] = c[HOR:] / c[:-HOR] - 1.0
        contig = np.zeros(n, bool); contig[:n - HOR] = (ts[HOR:] - ts[:-HOR]) == GAP
        Ts.append(ts); RP.append(r_past); RF.append(r_fwd); CT.append(contig); SK.append(step_ok)
    if not Ts:
        return None
    return dict(ts=np.concatenate(Ts), r_past=np.concatenate(RP), r_fwd=np.concatenate(RF),
                contig=np.concatenate(CT), step_ok=np.concatenate(SK))


def align_target_drivers(years):
    """Build per-driver delay-embeddable returns ALIGNED to USDJPY's clock by timestamp (inner-join on
    ts). Returns (ts, r_fwd_usdjpy, {driver: r_past_aligned}, step_ok_target). Missing driver bars are
    dropped from the intersection (NEVER ffill'd)."""
    tgt = load_returns(TARGET, years)
    base = pd.DataFrame({"ts": tgt["ts"], "r_fwd": tgt["r_fwd"], "contig": tgt["contig"],
                         "step_ok": tgt["step_ok"]}).drop_duplicates("ts")
    drv_cols = {}
    for p in DRIVERS:
        dd = load_returns(p, years)
        if dd is None:
            continue
        s = pd.DataFrame({"ts": dd["ts"], f"rp_{p}": dd["r_past"],
                          f"sk_{p}": dd["step_ok"]}).drop_duplicates("ts")
        base = base.merge(s, on="ts", how="inner")
        drv_cols[p] = f"rp_{p}"
    base = base.sort_values("ts").reset_index(drop=True)
    return base, drv_cols


# ----------------------------- CCM core (delay embed + lagged kNN cross-map) -----------------------------
def embed(r, step_ok, E, tau):
    """Time-delay embedding of a causal return series r on a single contiguous-step grid.
    Row t = [r_t, r_{t-tau}, ..., r_{t-(E-1)tau}]. A row is VALID only if all E lags AND the (E-1)*tau
    consecutive 60s steps leading into t are gap-free (step_ok), so the manifold never spans a data gap.
    Returns (M[n,E], valid_bool[n])."""
    n = len(r); span = (E - 1) * tau
    M = np.full((n, E), np.nan)
    for k in range(E):
        M[span:, k] = r[np.arange(span, n) - k * tau]
    valid = np.isfinite(M).all(axis=1)
    # require the span+1 consecutive steps ending at t to be contiguous (no intervening gap)
    contig_run = np.zeros(n, bool)
    if n > span:
        ok = np.ones(n, bool)
        for j in range(span + 1):
            shifted = np.zeros(n, bool)
            if j == 0:
                shifted[:] = True
            else:
                shifted[j:] = step_ok[j:]  # step_ok[i] = (ts[i]-ts[i-1]==STEP); need all of last span steps
            ok &= shifted
        contig_run = ok
    valid &= contig_run
    return M, valid


def crossmap_predict(M_lib, fwd_lib, M_qry, k):
    """Predict the forward target for each query point by exp-weighted kNN of its delay vector in the
    TRAIN library manifold. Returns yhat[n_qry]. Library = (M_lib, fwd_lib). This is the C4 estimator:
    library = driver-manifold rows mapped to USDJPY's forward return; query = held-out driver-manifold
    rows; the forecast is the cross-mapped USDJPY forward return whose SIGN is the direction signal."""
    if len(M_lib) < k + 1:
        return np.full(len(M_qry), np.nan)
    tree = cKDTree(M_lib)
    dist, nn = tree.query(M_qry, k=k)
    if k == 1:
        dist = dist[:, None]; nn = nn[:, None]
    d1 = dist[:, [0]]; d1 = np.where(d1 <= 0, 1e-12, d1)
    w = np.exp(-dist / d1)
    ws = w.sum(axis=1, keepdims=True); w = np.where(ws > 0, w / ws, 0.0)
    yhat = (w * fwd_lib[nn]).sum(axis=1)
    return yhat


def crossmap_skill_train(M, fwd, valid, L, k, theiler, rng):
    """In-sample cross-map skill rho(library L) for the TRAIN convergence diagnostic, using leave-the-
    Theiler-window-out neighbors. Pearson(yhat, fwd) over a scored subset. Used ONLY to pick the
    best-converging driver/E/tp on TRAIN."""
    idx = np.where(valid)[0]
    if len(idx) < L + theiler + 5:
        return np.nan
    Mv = M[idx]; fv = fwd[idx]
    # library subsample + scored targets, both from the valid set
    lib_sel = np.sort(rng.choice(len(idx), size=min(L, len(idx)), replace=False))
    n_score = min(TRAIN_SCORE_N, len(idx))
    qry_sel = np.sort(rng.choice(len(idx), size=n_score, replace=False))
    tree = cKDTree(Mv[lib_sel])
    kq = min(k + 3 * theiler + 8, len(lib_sel))
    dist, nn = tree.query(Mv[qry_sel], k=kq)
    cand = idx[lib_sel[nn]]                       # original indices of neighbors
    qabs = idx[qry_sel][:, None]
    ok = np.abs(cand - qabs) > theiler            # Theiler exclusion
    pick = ok & (np.cumsum(ok, axis=1) <= k)
    good = pick.sum(axis=1) >= k
    dd = np.where(pick, dist, np.inf)
    d1 = dd.min(axis=1, keepdims=True); d1 = np.where(d1 <= 0, 1e-12, d1)
    w = np.where(pick, np.exp(-dd / d1), 0.0)
    wss = w.sum(axis=1, keepdims=True); w = np.where(wss > 0, w / wss, 0.0)
    fwd_lib = fv[lib_sel]
    yhat = np.where(good, (w * fwd_lib[nn]).sum(axis=1), np.nan)
    yt = fv[qry_sel]
    m = np.isfinite(yhat) & good
    if m.sum() < 20 or np.std(yhat[m]) < 1e-18 or np.std(yt[m]) < 1e-18:
        return np.nan
    return float(np.corrcoef(yhat[m], yt[m])[0, 1])


# ----------------------------- driver / E / tp selection on TRAIN -----------------------------
def select_driver(train, drv_cols):
    """Pick (driver, E, tp) maximising TRAIN convergence: cross-map skill must RISE with library L
    (slope>0) and the skill at Lmax is the selection score. Returns (best_dict, table)."""
    table = {}
    best = None
    theiler_base = max(E_GRID) * TAU + max(TP_GRID) + 2
    for pair, col in drv_cols.items():
        rp = train[col].values.astype(float)
        sk = train[f"sk_{pair}"].values.astype(bool)
        for E in E_GRID:
            M, mvalid = embed(rp, sk, E, TAU)
            for tp in TP_GRID:
                # forward target aligned to the embedding base: predict r_fwd at t (already 15m fwd),
                # optionally shifted tp bars ahead (driver past -> USDJPY future at t+tp).
                fwd = np.full(len(M), np.nan)
                base_valid = mvalid & train["contig"].values.astype(bool)
                if tp == 0:
                    fwd = train["r_fwd"].values.astype(float)
                    vmask = base_valid & np.isfinite(fwd)
                else:
                    rf = train["r_fwd"].values.astype(float)
                    fwd[:len(M) - tp] = rf[tp:]
                    shifted_ok = np.zeros(len(M), bool); shifted_ok[:len(M) - tp] = base_valid[tp:]
                    vmask = base_valid & shifted_ok & np.isfinite(fwd)
                theiler = E * TAU + tp + 1
                rhos = []
                for L in L_GRID:
                    rho = crossmap_skill_train(M, fwd, vmask, L, k=E + 1, theiler=theiler, rng=RNG)
                    rhos.append(rho)
                fin = [(L, r) for L, r in zip(L_GRID, rhos) if np.isfinite(r)]
                if len(fin) < 2:
                    continue
                Ls = np.array([f[0] for f in fin], float)
                rr = np.array([f[1] for f in fin], float)
                slope = float(np.polyfit(np.log(Ls), rr, 1)[0])
                rho_max = float(rr[-1])
                conv = bool(slope > 0 and rho_max > 0)
                key = f"{pair}.E{E}.tp{tp}"
                table[key] = {"pair": pair, "E": E, "tp": tp,
                              "rho_by_L": {int(L): (None if not np.isfinite(r) else round(float(r), 5))
                                           for L, r in zip(L_GRID, rhos)},
                              "conv_slope_logL": round(slope, 5), "rho_Lmax": round(rho_max, 5),
                              "converges": conv}
                # selection score: prefer converging drivers, rank by rho at Lmax
                score = rho_max + (0.0 if conv else -1.0)
                if best is None or score > best["score"]:
                    best = {"score": score, "pair": pair, "E": E, "tp": tp,
                            "slope": slope, "rho_Lmax": rho_max, "converges": conv}
    return best, table


# ----------------------------- held-out signal: cross-map sign from frozen TRAIN library -------------
def ccm_signal(train, hold, drv_cols, pair, E, tp):
    """Build the C4 direction signal on a held-out split: embed the driver's causal past on the held-out
    clock, cross-map against the FROZEN TRAIN library (driver manifold -> USDJPY forward return), and
    return the cross-mapped forecast yhat per held-out USDJPY bar. SIGN(yhat) is the direction signal.
    Returns (ts, yhat, valid) aligned to held-out rows."""
    col = drv_cols[pair]
    # ---- TRAIN library (frozen) ----
    rp_tr = train[col].values.astype(float); sk_tr = train[f"sk_{pair}"].values.astype(bool)
    Mtr, vtr = embed(rp_tr, sk_tr, E, TAU)
    base_valid_tr = vtr & train["contig"].values.astype(bool)
    rf_tr = train["r_fwd"].values.astype(float)
    if tp == 0:
        fwd_tr = rf_tr; vmask_tr = base_valid_tr & np.isfinite(fwd_tr)
    else:
        fwd_tr = np.full(len(Mtr), np.nan); fwd_tr[:len(Mtr) - tp] = rf_tr[tp:]
        shifted = np.zeros(len(Mtr), bool); shifted[:len(Mtr) - tp] = base_valid_tr[tp:]
        vmask_tr = base_valid_tr & shifted & np.isfinite(fwd_tr)
    lib_idx = np.where(vmask_tr)[0]
    if len(lib_idx) > TRAIN_LIB_CAP:
        lib_idx = np.sort(RNG.choice(lib_idx, TRAIN_LIB_CAP, replace=False))
    M_lib = Mtr[lib_idx]; fwd_lib = fwd_tr[lib_idx]
    # ---- held-out queries ----
    rp_h = hold[col].values.astype(float); sk_h = hold[f"sk_{pair}"].values.astype(bool)
    Mh, vh = embed(rp_h, sk_h, E, TAU)
    qmask = vh & np.isfinite(Mh).all(axis=1)
    ts_h = hold["ts"].values.astype("int64")
    yhat = np.full(len(Mh), np.nan)
    if qmask.sum() > 0 and len(M_lib) > E + 2:
        yh = crossmap_predict(M_lib, fwd_lib, Mh[qmask], k=E + 1)
        yhat[np.where(qmask)[0]] = yh
    return ts_h, yhat, qmask & np.isfinite(yhat)


def yhat_to_proba(yhat):
    """Map the cross-mapped forward-return forecast to a pseudo-probability in (0,1) for side_eval:
    sign(yhat) gives direction; |yhat| (rank-scaled) gives confidence. p = 0.5 + 0.5*sign*rankmag."""
    p = np.full(len(yhat), 0.5)
    fin = np.isfinite(yhat)
    if fin.sum() == 0:
        return p
    v = yhat[fin]
    mag = np.abs(v)
    # rank-scale magnitude to [0,1] so the cov% quantiles in side_eval are well-spread
    order = mag.argsort()
    rank = np.empty(len(mag)); rank[order] = np.linspace(0, 1, len(mag))
    p[np.where(fin)[0]] = 0.5 + 0.5 * np.sign(v) * rank
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return p


# ----------------------------- per-year NY eval (deriv-faithful label from build) -----------------------------
def eval_year(w, train, drv_cols, pair, E, tp):
    """Held-out year: build() gives the AUTHORITATIVE deriv-faithful label/moved/ts; CCM gives the signal.
    Restrict DECISION rows to NY. Report moved-AUC vs up/down label on NY moved bars + cov3% side_eval."""
    Xw, yw, mw, tsw = build(SPL[w], 1)             # full-resolution held-out (label is authoritative)
    del Xw; gc.collect()
    hold, _ = align_target_drivers(SPL[w])   # FIX: function returns (df, drv_cols); take the df (was capturing the tuple -> hold[col] TypeError)
    ts_h, yhat, vsig = ccm_signal(train, hold, drv_cols, pair, E, tp)
    # align CCM signal (on intersection clock) back onto build()'s ts via timestamp map
    sig_map = {int(t): float(v) for t, v, ok in zip(ts_h, yhat, vsig) if ok}
    pr = np.full(len(tsw), 0.5)
    has = np.zeros(len(tsw), bool)
    for i, t in enumerate(tsw):
        v = sig_map.get(int(t))
        if v is not None:
            pr[i] = v; has[i] = True
    pr = yhat_to_proba(np.where(has, pr, np.nan))   # map raw forecast -> pseudo-proba (NaN where no sig)
    pr = np.where(has, pr, 0.5)
    ny = session_mask(tsw, "ny")
    dec = ny & mw & has                              # NY moved bars with a live CCM signal
    up_rate = float(yw[ny & mw].mean()) if (ny & mw).sum() else float("nan")
    auc = float("nan")
    if dec.sum() >= 30 and len(np.unique(yw[dec])) == 2:
        auc = float(roc_auc_score(yw[dec], pr[dec]))
    # selective win-rate at cov3% via side_eval on the NY decision universe (ties LOSE, gap=900 built in)
    ny_idx = np.where(ny & has)[0]
    cov3 = None; cc = None
    if len(ny_idx) > 0:
        prn, yn, mn, tsn = pr[ny_idx], yw[ny_idx], mw[ny_idx], tsw[ny_idx]
        conf = np.abs(prn - 0.5)
        thr3 = float(np.quantile(conf, 0.97)) if len(conf) else 0.0
        cov3 = side_eval(prn, yn, mn, tsn, thr3)
        cc = covcurve(prn, yn, mn, tsn)
    return {"moved_auc": auc, "moved_up_rate": up_rate, "n_decision": int(dec.sum()),
            "n_ny_signal": int((ny & has).sum()),
            "cov3_side": cov3, "covcurve": cc, "tripwire_ok": bool(0.47 <= up_rate <= 0.53)}


def main():
    t0 = time.time()
    # ---- PRE-REGISTER falsifier BEFORE any held-out read ----
    res = {
        "key": "USDJPY.15m.ny",
        "lever": "C4 convergent cross-mapping (lagged, causal): delay-embed each USD-major driver "
                 "(E in {3,4}, tau=1), kNN cross-map to forecast USDJPY forward 15m return, take SIGN "
                 "of the cross-mapped forecast of the best-TRAIN-converging driver as the direction signal",
        "settlement": "deriv-faithful fixed-15m sign(close[t+15]-close[t]), ties LOSE, BE=0.541, "
                      "gap=900s nonoverlap_chrono; DECISION rows restricted to NY (features causal)",
        "drivers_candidate": DRIVERS, "E_grid": E_GRID, "tau": TAU, "tp_grid": TP_GRID,
        "L_grid": L_GRID, "train_lib_cap": TRAIN_LIB_CAP, "train_score_n": TRAIN_SCORE_N,
        "splits": SPL,
        "falsifier": {
            "registered_utc": "pre-held-out",
            "KILL_if": "best held-out NY moved-AUC <= 0.51 (coupling gates magnitude, not sign)  "
                       "OR no held-out year NY cov3% win-rate CI95-lower clears 0.541",
            "rationale": "Sign-invariance theorem (arXiv:2512.15720): state/coupling/complexity gates "
                         "(HMM/RMT/CCM/Kalman/entropy/Hurst) are expected to gate move SIZE not SIGN. "
                         "Honest prior: null (~0.50). Incumbent NY own-pair GBM AUC ~.539 / WR .58-.60; "
                         "SURVIVE only if held-out moved-AUC > ~.539 OR NY cov3% WR CI-lo >= 0.541 in >=2 yrs.",
            "incumbent": {"ny_ownpair_gbm_auc": 0.539, "ny_winrate": [0.58, 0.60]},
        },
    }
    json.dump(res, open(RESULT, "w"), indent=2)

    # ---- TRAIN: load returns, select (driver, E, tp) on TRAIN ONLY ----
    # IMPORTANT: the delay embedding MUST live on the contiguous native 60s clock — NEVER stride the
    # merged frame (that would silently embed a 6-bar-spaced trajectory while step_ok falsely reads
    # contiguous). Tractability comes from random TRAIN_LIB_CAP/TRAIN_SCORE_N subsampling of VALID
    # native-clock indices inside the CCM core, not from row-striding the frame.
    print("[ccm15m] loading TRAIN returns ...", flush=True)
    train, drv_cols = align_target_drivers(SPL["train"])
    gc.collect()
    print(f"[ccm15m] TRAIN rows={len(train):,} drivers={list(drv_cols)} build={time.time()-t0:.0f}s", flush=True)

    best, table = select_driver(train, drv_cols)
    if best is None:
        res["selection"] = {"error": "no_convergent_driver_scored"}
        res["verdict"] = {"KILLED": True, "beats_base_auc": False,
                          "note": "CCM core scored no finite cross-map skill on TRAIN (embedding/data issue)"}
        json.dump(res, open(RESULT, "w"), indent=2)
        print("[ccm15m] ABORT: no driver scored", flush=True); return
    pair, E, tp = best["pair"], best["E"], best["tp"]
    res["selection"] = {"driver": pair, "E": E, "tp_bars": tp, "tp_s": tp * STEP,
                        "train_conv_slope_logL": round(best["slope"], 5),
                        "train_rho_Lmax": round(best["rho_Lmax"], 5),
                        "train_converges": bool(best["converges"]), "table": table}
    res["signal_auc"] = res["val_or_signal_auc"] = round(best["rho_Lmax"], 5)  # TRAIN cross-map skill proxy
    print(f"[ccm15m] SELECTED driver={pair} E={E} tp={tp} (rho_Lmax={best['rho_Lmax']:.4f} "
          f"slope={best['slope']:.4f} conv={best['converges']}) {time.time()-t0:.0f}s", flush=True)

    # ---- held-out per-year NY eval (label authoritative from build) ----
    res["years"] = {}
    for w in ("test24", "test25", "oos"):
        yr = eval_year(w, train, drv_cols, pair, E, tp)
        cov3 = yr["cov3_side"]["COMBINED"] if (yr["cov3_side"] and "COMBINED" in yr["cov3_side"]) else \
               {"n": 0, "wr": float("nan"), "ci": [float("nan")] * 2}
        res["years"][w] = {
            "auc": yr["moved_auc"], "moved_up_rate": yr["moved_up_rate"],
            "n_decision": yr["n_decision"], "n_ny_signal": yr["n_ny_signal"],
            "cov3_wr": cov3["wr"], "cov3_n": cov3["n"], "cov3_ci": cov3["ci"],
            "cov3_side": yr["cov3_side"], "covcurve": yr["covcurve"],
            "tripwire_ok": yr["tripwire_ok"],
        }
        print(f"=== {w} === moved-AUC={yr['moved_auc']:.4f} up-rate={yr['moved_up_rate']:.4f} "
              f"n_dec={yr['n_decision']} | cov3% COMB n{cov3['n']} wr={cov3['wr']:.4f} "
              f"CI[{cov3['ci'][0]:.3f},{cov3['ci'][1]:.3f}] tripwire={yr['tripwire_ok']}", flush=True)

    # ---- apply pre-registered falsifier ----
    aucs = [res["years"][w]["auc"] for w in ("test24", "test25", "oos") if np.isfinite(res["years"][w]["auc"])]
    best_auc = max(aucs) if aucs else float("nan")
    auc_kill = (not aucs) or best_auc <= 0.51
    cov3_clears = [w for w in ("test24", "test25", "oos")
                   if np.isfinite(res["years"][w]["cov3_ci"][0]) and res["years"][w]["cov3_ci"][0] >= BE]
    beats_base = bool(aucs and best_auc > 0.539)
    killed = bool(auc_kill and len(cov3_clears) < 2)
    res["verdict"] = {
        "KILLED": killed,
        "beats_base_auc": beats_base,
        "best_holdout_moved_auc": (None if not aucs else round(best_auc, 5)),
        "cov3_years_CIlo_clears_BE": cov3_clears,
        "note": ("KILLED: " if killed else "SURVIVED: ") +
                (f"best held-out NY moved-AUC {best_auc:.4f} <= 0.51 (coupling gates magnitude not sign) "
                 f"and cov3% CI-lo clears 0.541 in {len(cov3_clears)} yr(s)" if killed else
                 f"best held-out NY moved-AUC {best_auc:.4f}; cov3% CI-lo clears 0.541 in years {cov3_clears}; "
                 f"beats_base_auc(>.539)={beats_base}"),
    }
    json.dump(res, open(RESULT, "w"), indent=2)
    print(f"\n[ccm15m] VERDICT: {'KILLED' if killed else 'SURVIVED'} "
          f"(best_auc={best_auc if aucs else float('nan'):.4f}; cov3-clears={cov3_clears}; "
          f"beats_base={beats_base}) -> {RESULT}  total={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
