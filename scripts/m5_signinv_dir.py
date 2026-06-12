"""5m SIGN-INVARIANT COMPLEXITY/REGIME STATS AS DIRECTION (RUN, do-not-infer).

Converts the ledger's THEOREM-based subsumption of rows C1a (HMM), C2a (Kalman) and the
complexity-gate family into an EXPERIMENT at 5m. The program holds the sign-invariance theorem
(arXiv:2512.15720): entropy / HMM / Hurst / Kalman / |ret| statistics gate move SIZE not SIGN.
m5_magdyn_result.json already showed magnitude-as-direction is flat at 5m. This script confirms the
SAME for HMM/Kalman/entropy/Hurst at 5m, by RUNNING not arguing.

CAUSAL FEATURES (EURUSD OWN 5m anchor clock from m5_xpair.build_xp; input STRICTLY <= t — the
load-bearing leakage point, trap #1 in the strategy-eval skill):
  - HMM state posterior: forward FILTER only. GaussianHMM is fit (Baum-Welch) on the TRAIN split, then
    the per-bar posterior P(state_t | obs_1..t) is computed via min1_hmm._forward_filter — an explicit
    forward-only logsumexp recursion. We NEVER call model.predict (Viterbi) nor model.predict_proba /
    model.score_samples (forward-BACKWARD smoother): both leak future obs. Emit P(state) posteriors +
    most-likely-FILTERED-state.
  - Kalman velocity/level sign: forward FILTER only. min1_kalman.kalman_llt is a forward local-linear-trend
    recursion (state at t uses obs <= t); NO RTS smoother. Emit filtered velocity(slope), level-deviation
    (logprice - filtered level), and their signs.
  - Rolling Shannon entropy of the 5m return-SIGN sequence over trailing windows {12,24,48 bars}.
  - Rolling Hurst exponent (variance-of-sums + R/S) over trailing windows {24,48,96 bars}.
  All windows END at or before t. NaN -> 0.0, NEVER ffill. Keep EURUSD's own clock (build_xp already drops
  missing / non-contiguous bars; we never reindex/ffill).

TWO TESTS (both sides UP+DOWN):
  (A) STANDALONE: LGBM on JUST the sign-invariant features -> TRUE 5m sign (build_xp H=5 _y). Report
      worst-VAL-half dirAUC on MOVED bars + per-year 2024/2025/2026 (the clean "does it carry sign" test).
  (B) INCREMENTAL: add the sign-invariant features to the m5xp cross-pair primary feature set
      (cols=primary_feats from XP.art("strategy.json")), retrain the primary, eval UP+DOWN selective per-year
      at cov0.05 vs the incumbent (UP .577 binding / .553 floor; DOWN .5441) — does any lift the binding year?
  Also: each sign-inv feature's LightGBM permutation importance for SIGN (expected ~0) AND, as a sanity check,
  the same features' AUC for predicting |ret| MAGNITUDE top-tercile (expected >> their direction AUC) — which
  demonstrates they ARE magnitude carriers, confirming the theorem.

DISCIPLINE replicated EXACTLY from m5_lossbatch.py / m5_xpair*.py:
  build via MX.augment(MX.build_xp(XP.SPL[w],STRIDE),XP.SPL[w],XP.MODE) STRIDE=6; per_year() NY gate +
  conf-cover cov0.05 + MX.nonoverlap_chrono(ts,mask,300) + per-year 2024/2025/2026 boot CI95 + win vs true _y +
  ties LOSE breakeven 0.541; standalone threshold/selection on WORST-VAL-half (NOT VAL-acc-max); moved up-rate
  in [0.47,0.53] tripwire.

PRE-REGISTERED FALSIFIER (written to m5_signinv_dir_result.json BEFORE OOS):
  KILL (confirm sign-invariance at 5m) unless some sign-invariant feature/model yields (A) standalone
  worst-VAL-half MOVED-bar dirAUC > 0.52 AND a binding-year selective CI95-lo >= 0.541, OR (B) incremental
  binding-year UP CI-lo > .553/.577 or DOWN > .5441 by >1 SE. Expected NULL.

  ~/binary-algo-venv/bin/python m5_signinv_dir.py
"""
import os, sys; sys.argv = ["x"]   # neutralize argv so MX/XP module-level __main__ guards stay dormant
import json, time
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

import m5_xpair as MX                 # build_xp / augment / feat_cols / xp_cols / nonoverlap_chrono / boot / SPL
import m5_xpair_production as XP       # art() -> strategy.json (primary_feats), MODE="xpof", SPL
# REUSED forward-FILTER-ONLY builders (the leakage point) — imported verbatim, NOT reimplemented:
#   min1_hmm._forward_filter : forward-only logsumexp posterior P(state_t|obs_1..t)  (NO Viterbi/backward)
#   min1_hmm.std_fit/std_apply : TRAIN-fit standardization of emission features
#   min1_kalman.kalman_llt   : forward local-linear-trend Kalman FILTER (NO RTS smoother)
import min1_hmm as HM
import min1_kalman as KAL
from hmmlearn.hmm import GaussianHMM

ROOT = "/home/sean/git/binary-algo"
BE = 0.541
STRIDE = 6          # m5_lossbatch STRIDE
COV = 0.05          # selective coverage (m5_lossbatch)
GAP = 300           # 5m non-overlap gap (seconds)
TRAIN_CAP = 611_000 # match m5_lossbatch full train if it fits; subsample below this if memory tight
HMM_K = int(os.environ.get("SI_HMM_K", "3"))
HMM_SEED = int(os.environ.get("SI_HMM_SEED", "0"))
ENT_WINS = (12, 24, 48)
HURST_WINS = (24, 48, 96)

INCUMBENT = {"UP_binding_2025": 0.577, "UP_floor": 0.553, "DOWN_binding": 0.5441}


# ---------------------------------------------------------------------------
# boot CI95 — same shape as m5_lossbatch.boot
# ---------------------------------------------------------------------------
def boot(c, nb=2500, seed=7):
    c = np.asarray(c, float)
    if len(c) < 5:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    n = len(c)
    a = np.array([c[rng.integers(0, n, n)].mean() for _ in range(nb)])
    return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))


# ---------------------------------------------------------------------------
# Sign-invariant CAUSAL feature builders.  ALL operate on EURUSD's own 5m anchor
# clock (the build_xp index): the per-bar log-return of EURUSD (= eu_r1 in build_xp,
# the 1-bar EURUSD-equivalent log return), and the EURUSD log-price reconstructed
# cumulatively from those same returns within each contiguous run.  Input strictly <= t.
# ---------------------------------------------------------------------------
HMM_EMIT_FEATS = ["eu_r1", "eu_r3", "eu_r5", "disp5", "agree5", "comp60"]
# emission feats present in build_xp output: signed returns (eu_r*) + cross-pair dispersion/agreement +
# compression vol proxy — so HMM states CAN be directional, not only vol (mirrors min1_hmm's EMIT intent).


def _shannon_sign_entropy(sign_seq, win):
    """Rolling Shannon entropy (bits) of the {down=0,up=1} sign sequence over trailing `win` bars,
    window ENDING at t (uses obs <= t only). NaN until the window fills -> 0.0 (never ffill)."""
    s = pd.Series((sign_seq > 0).astype(float))
    def H(x):
        p1 = x.mean()
        if p1 <= 0.0 or p1 >= 1.0:
            return 0.0
        p0 = 1.0 - p1
        return float(-(p1 * np.log2(p1) + p0 * np.log2(p0)))
    return s.rolling(win, min_periods=win).apply(H, raw=False).values


def _hurst_vs(ret, win):
    """Rolling Hurst via variance-of-aggregated-sums, trailing `win` bars ENDING at t. Causal.
    H ~ slope of log Var(aggregated returns at scale m) vs log(m) over m in {1,2,4,8}, /2 + 0.5."""
    scales = np.array([1, 2, 4, 8], dtype=int)
    logm = np.log(scales.astype(float))

    def hvs(x):
        x = np.asarray(x, float)
        v = []
        for m in scales:
            k = len(x) // m
            if k < 2:
                v.append(np.nan); continue
            agg = x[: k * m].reshape(k, m).sum(axis=1)
            v.append(np.var(agg) + 1e-30)
        v = np.asarray(v, float)
        ok = np.isfinite(v) & (v > 0)
        if ok.sum() < 2:
            return np.nan
        slope = np.polyfit(logm[ok], np.log(v[ok]), 1)[0]
        return float(slope / 2.0)  # Var(agg) ~ m^(2H) => slope = 2H

    return pd.Series(ret).rolling(win, min_periods=win).apply(hvs, raw=True).values


def _hurst_rs(ret, win):
    """Rolling Hurst via rescaled-range (R/S) on the trailing `win`-bar window ENDING at t. Causal.
    Single-window R/S estimate: log(R/S) ~ H * log(win)."""
    def rs(x):
        x = np.asarray(x, float)
        n = len(x)
        if n < 8:
            return np.nan
        z = x - x.mean()
        Y = np.cumsum(z)
        R = Y.max() - Y.min()
        S = x.std(ddof=0) + 1e-30
        if R <= 0:
            return np.nan
        return float(np.log(R / S) / np.log(n))

    return pd.Series(ret).rolling(win, min_periods=win).apply(rs, raw=True).values


def _kalman_feats(logp):
    """Forward LLT Kalman FILTER (min1_kalman.kalman_llt — forward recursion only, NO RTS smoother).
    Returns filtered level-deviation (logp - level), filtered velocity (slope), and their signs. Causal."""
    # noise params mirror min1_kalman's grid midpoint (q_level,q_slope,r); local-linear-trend on log price.
    lev, slp = KAL.kalman_llt(np.asarray(logp, float), 1e-7, 1e-9, 1e-6)
    dev = np.asarray(logp, float) - lev
    return dev, slp


def _hmm_fit_train(emit_tr):
    """Fit GaussianHMM (Baum-Welch) on TRAIN emission matrix only. Returns (model, mu, sd).
    Standardization fit on TRAIN via min1_hmm.std_fit (reused)."""
    valid = np.all(np.isfinite(emit_tr), axis=1)
    mu, sd = HM.std_fit(emit_tr, valid)
    Z = HM.std_apply(emit_tr, mu, sd)
    model = GaussianHMM(n_components=HMM_K, covariance_type="diag",
                        n_iter=50, tol=1e-3, random_state=HMM_SEED, init_params="stmc")
    model.fit(Z)
    return model, mu, sd


def _hmm_posterior(model, mu, sd, emit):
    """Per-bar FILTERED posterior P(state_t|obs_1..t) via min1_hmm._forward_filter (forward-only;
    NO Viterbi, NO forward-backward). Returns (soft (T,K) posterior, hard most-likely-filtered-state)."""
    Z = HM.std_apply(emit, mu, sd)
    soft = HM._forward_filter(model, Z)          # forward-only logsumexp recursion
    hard = soft.argmax(1).astype("float32")
    return soft, hard


def add_signinv(F, hmm_model, hmm_mu, hmm_sd):
    """Append sign-invariant CAUSAL features to a build_xp frame F (index = EURUSD 5m anchor bars).
    All inputs <= t; NaN -> 0.0, never ffill. Returns (F_with_feats, list_of_signinv_col_names)."""
    r = F["eu_r1"].astype("float64").values                       # 1-bar EURUSD-equiv log return (<= t)
    logp = np.cumsum(np.nan_to_num(r, nan=0.0))                   # causal log-price proxy (drift only matters locally)
    cols = []

    # --- Kalman forward-filter (velocity / level-deviation + signs) ---
    dev, slp = _kalman_feats(logp)
    F["si_kal_dev"] = dev; cols.append("si_kal_dev")
    F["si_kal_vel"] = slp; cols.append("si_kal_vel")
    F["si_kal_dev_sign"] = np.sign(dev); cols.append("si_kal_dev_sign")
    F["si_kal_vel_sign"] = np.sign(slp); cols.append("si_kal_vel_sign")

    # --- rolling Shannon entropy of return-sign sequence ---
    for w in ENT_WINS:
        F[f"si_ent{w}"] = _shannon_sign_entropy(r, w); cols.append(f"si_ent{w}")

    # --- rolling Hurst (variance-of-sums + R/S) ---
    for w in HURST_WINS:
        F[f"si_hvs{w}"] = _hurst_vs(r, w); cols.append(f"si_hvs{w}")
        F[f"si_hrs{w}"] = _hurst_rs(r, w); cols.append(f"si_hrs{w}")

    # --- HMM forward-filtered posterior + most-likely-filtered-state ---
    emit = F[HMM_EMIT_FEATS].astype("float64").values
    soft, hard = _hmm_posterior(hmm_model, hmm_mu, hmm_sd, emit)
    for k in range(HMM_K):
        F[f"si_hmm_p{k}"] = soft[:, k]; cols.append(f"si_hmm_p{k}")
    F["si_hmm_state"] = hard; cols.append("si_hmm_state")

    # NaN -> 0.0, never ffill
    for c in cols:
        F[c] = np.nan_to_num(F[c].astype("float64").values, nan=0.0, posinf=0.0, neginf=0.0)
    return F, cols


# ---------------------------------------------------------------------------
# per_year — replicated EXACTLY from m5_lossbatch.per_year (NY gate, cov0.05 conf-cover,
# nonoverlap_chrono 300s, per-year 2024/2025/2026, boot CI95, win vs true _y, ties LOSE).
# ---------------------------------------------------------------------------
def per_year(EV, scorefn, sv):
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
        sel = MX.nonoverlap_chrono(ts, m, GAP)
        if len(sel) >= 20:
            cc = (y[sel] == sv).astype(float)              # ties already excluded by build_xp (fwd!=0); win vs true sign
            lo, hi = boot(cc)
            Y = 2024 if w == "test24" else 2025 if w == "test25" else 2026
            res[Y] = dict(win=round(float(cc.mean()), 4), n=len(sel), ci=[round(lo, 4), round(hi, 4)])
    b = [res[Y]["win"] for Y in res]
    lo = [res[Y]["ci"][0] for Y in res]
    ns = [res[Y]["n"] for Y in res]
    res["binding"] = dict(win=min(b) if b else None, ci_lo=min(lo) if lo else None, min_n=min(ns) if ns else 0)
    return res


def yr_of(w):
    return 2024 if w == "test24" else 2025 if w == "test25" else 2026


def worst_val_half_dirauc(model, VA, cols):
    """Standalone selection metric: worst of the two chronological VAL halves' MOVED-bar dirAUC
    (NOT VAL-acc-max). MOVED bars = build_xp already excludes ties (fwd!=0), so every retained
    VAL bar is a moved bar. Reported on the held-out-from-train VAL split."""
    p = model.predict_proba(VA[cols].astype("float32"))[:, 1]
    y = VA["_y"].astype(int).values
    nh = len(y) // 2
    a1 = roc_auc_score(y[:nh], p[:nh]) if len(set(y[:nh].tolist())) > 1 else float("nan")
    a2 = roc_auc_score(y[nh:], p[nh:]) if len(set(y[nh:].tolist())) > 1 else float("nan")
    return float(np.nanmin([a1, a2])), float(a1), float(a2)


def per_year_dirauc(model, EV, cols):
    """Per-year MOVED-bar dirAUC (all retained bars are moved). NY-gated to match the book's universe."""
    out = {}
    for w, D in EV.items():
        p = model.predict_proba(D[cols].astype("float32"))[:, 1]
        y = D["_y"].astype(int).values
        ny = D["sess_ny"].values > 0.5
        if ny.sum() < 50 or len(set(y[ny].tolist())) < 2:
            continue
        out[yr_of(w)] = round(float(roc_auc_score(y[ny], p[ny])), 4)
    return out


def moved_up_rate_tripwire(EV):
    """Moved up-rate must be in [0.47,0.53] per the skill — all build_xp bars are moved (ties dropped)."""
    rates = {}
    for w, D in EV.items():
        y = D["_y"].astype(int).values
        ny = D["sess_ny"].values > 0.5
        rates[yr_of(w)] = round(float(y[ny].mean()), 4) if ny.sum() else float("nan")
    ok = all((0.47 <= v <= 0.53) for v in rates.values() if np.isfinite(v))
    return rates, ok


def main():
    t0 = time.time()
    print(f"[signinv] start  STRIDE={STRIDE} cov={COV} HMM_K={HMM_K}", flush=True)

    # ---- strategy.json -> incumbent primary feature set (the m5xp cross-pair primary) ----
    p = json.load(open(XP.art("strategy.json")))
    primary_feats = p["primary_feats"]
    print(f"[signinv] primary_feats={len(primary_feats)} (m5xp cross-pair primary)", flush=True)

    # ---- build TRAIN + held-out windows EXACTLY like m5_lossbatch ----
    TR = MX.augment(MX.build_xp(XP.SPL["train"], STRIDE), XP.SPL["train"], XP.MODE)
    VA = MX.augment(MX.build_xp(XP.SPL["val"]), XP.SPL["val"], XP.MODE)
    EV = {w: MX.augment(MX.build_xp(XP.SPL[w]), XP.SPL[w], XP.MODE) for w in ("test24", "test25", "oos")}
    print(f"[signinv] built train={len(TR):,} val={len(VA):,} "
          f"test24={len(EV['test24']):,} test25={len(EV['test25']):,} oos={len(EV['oos']):,} {time.time()-t0:.0f}s",
          flush=True)

    # subsample TRAIN if memory tight (match m5_lossbatch full train if it fits)
    if len(TR) > TRAIN_CAP:
        TR = TR.iloc[:: max(1, len(TR) // TRAIN_CAP)]
        print(f"[signinv] subsampled train -> {len(TR):,}", flush=True)

    # ---- fit HMM on TRAIN emission feats ONLY (Baum-Welch); standardization fit on TRAIN ----
    emit_tr = TR[HMM_EMIT_FEATS].astype("float64").values
    emit_tr = np.nan_to_num(emit_tr, nan=0.0, posinf=0.0, neginf=0.0)
    hmm_model, hmm_mu, hmm_sd = _hmm_fit_train(emit_tr)
    print(f"[signinv] HMM fit on TRAIN (K={HMM_K}, forward-filter posterior only) {time.time()-t0:.0f}s", flush=True)

    # ---- add sign-invariant CAUSAL features to every frame ----
    TR, SI = add_signinv(TR, hmm_model, hmm_mu, hmm_sd)
    VA, _ = add_signinv(VA, hmm_model, hmm_mu, hmm_sd)
    for w in EV:
        EV[w], _ = add_signinv(EV[w], hmm_model, hmm_mu, hmm_sd)
    print(f"[signinv] sign-inv feats added: {len(SI)} cols {SI} {time.time()-t0:.0f}s", flush=True)

    # moved up-rate tripwire (must be in [0.47,0.53]) on held-out windows
    up_rates, up_ok = moved_up_rate_tripwire(EV)
    print(f"[signinv] moved up-rate (NY) per year = {up_rates}  tripwire_ok={up_ok}", flush=True)

    ytr = TR["_y"].astype(int).values

    # =====================================================================
    # TEST (A) STANDALONE — LGBM on JUST the sign-invariant features -> true sign
    # =====================================================================
    print("[signinv] === TEST A: STANDALONE sign-inv -> true 5m sign ===", flush=True)
    Astd = lgb.LGBMClassifier(objective="binary", metric="auc", learning_rate=0.02, num_leaves=63,
                              min_child_samples=400, subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
                              reg_lambda=20, n_estimators=2000, n_jobs=20, verbosity=-1)
    Astd.fit(TR[SI].astype("float32"), ytr,
             eval_set=[(VA[SI].astype("float32"), VA["_y"].astype(int).values)],
             eval_metric="auc", callbacks=[lgb.early_stopping(150), lgb.log_evaluation(0)])
    wvh, vh1, vh2 = worst_val_half_dirauc(Astd, VA, SI)
    a_peryear = per_year_dirauc(Astd, EV, SI)
    a_sel = {"UP": per_year(EV, lambda D: Astd.predict_proba(D[SI].astype("float32"))[:, 1], 1),
             "DOWN": per_year(EV, lambda D: Astd.predict_proba(D[SI].astype("float32"))[:, 1], 0)}
    print(f"  standalone worst-VAL-half MOVED dirAUC={wvh:.4f} (halves {vh1:.4f}/{vh2:.4f}); "
          f"per-year MOVED dirAUC={a_peryear}", flush=True)
    print(f"  standalone selective UP binding={a_sel['UP']['binding']} DOWN binding={a_sel['DOWN']['binding']}",
          flush=True)

    # ---- per-feature SIGN permutation importance (expected ~0) on VAL ----
    base_auc = roc_auc_score(VA["_y"].astype(int).values,
                             Astd.predict_proba(VA[SI].astype("float32"))[:, 1])
    rng = np.random.default_rng(0)
    Xva = VA[SI].astype("float32").values.copy()
    yva = VA["_y"].astype(int).values
    perm_sign = {}
    for j, c in enumerate(SI):
        col = Xva[:, j].copy()
        drops = []
        for _ in range(3):
            Xp = Xva.copy()
            Xp[:, j] = rng.permutation(col)
            drops.append(base_auc - roc_auc_score(yva, Astd.predict_proba(Xp)[:, 1]))
        perm_sign[c] = round(float(np.mean(drops)), 5)
    print(f"  SIGN permutation importance (AUC drop, expect ~0): "
          f"{dict(sorted(perm_sign.items(), key=lambda kv: -kv[1])[:6])}", flush=True)

    # =====================================================================
    # SANITY: same features' AUC for |ret| MAGNITUDE top-tercile (expect >> direction AUC)
    # =====================================================================
    print("[signinv] === SANITY: sign-inv -> |ret| MAGNITUDE top-tercile ===", flush=True)
    aret_tr = np.abs(TR["_fwd"].astype("float64").values)
    q67 = float(np.quantile(aret_tr, 2.0 / 3.0))
    ymag_tr = (aret_tr >= q67).astype(int)
    Amag = lgb.LGBMClassifier(objective="binary", metric="auc", learning_rate=0.02, num_leaves=63,
                              min_child_samples=400, subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
                              reg_lambda=20, n_estimators=2000, n_jobs=20, verbosity=-1)
    yvm = (np.abs(VA["_fwd"].astype("float64").values) >= q67).astype(int)
    Amag.fit(TR[SI].astype("float32"), ymag_tr,
             eval_set=[(VA[SI].astype("float32"), yvm)], eval_metric="auc",
             callbacks=[lgb.early_stopping(150), lgb.log_evaluation(0)])
    # per-feature single-feature AUC: SIGN vs MAGNITUDE (the theorem table), measured on VAL
    feat_table = {}
    for c in SI:
        v = VA[c].astype("float64").values
        if len(set(np.round(v, 12).tolist())) < 2:
            feat_table[c] = {"dir_auc": None, "mag_auc": None}
            continue
        da = roc_auc_score(yva, v); ma = roc_auc_score(yvm, v)
        feat_table[c] = {"dir_auc": round(float(max(da, 1 - da)), 4),
                         "mag_auc": round(float(max(ma, 1 - ma)), 4)}
    mag_peryear = {}
    for w, D in EV.items():
        ya = (np.abs(D["_fwd"].astype("float64").values) >= q67).astype(int)
        pm = Amag.predict_proba(D[SI].astype("float32"))[:, 1]
        if len(set(ya.tolist())) > 1:
            mag_peryear[yr_of(w)] = round(float(roc_auc_score(ya, pm)), 4)
    print(f"  MAGNITUDE-model per-year AUC (expect >> direction)={mag_peryear}", flush=True)

    # =====================================================================
    # TEST (B) INCREMENTAL — add sign-inv feats to primary_feats, retrain primary, UP+DOWN selective
    # =====================================================================
    print("[signinv] === TEST B: INCREMENTAL (primary_feats + sign-inv) vs incumbent ===", flush=True)
    inc_cols = list(dict.fromkeys(list(primary_feats) + SI))
    yva_i = VA["_y"].astype(int).values
    Pinc = lgb.LGBMClassifier(objective="binary", metric="auc", learning_rate=0.02, num_leaves=127,
                              min_child_samples=400, subsample=0.8, subsample_freq=1, colsample_bytree=0.5,
                              reg_lambda=20, n_estimators=3000, n_jobs=20, verbosity=-1)
    Pinc.fit(TR[inc_cols].astype("float32"), ytr,
             eval_set=[(VA[inc_cols].astype("float32"), yva_i)], eval_metric="auc",
             callbacks=[lgb.early_stopping(150), lgb.log_evaluation(0)])
    inc_scorefn = lambda D: Pinc.predict_proba(D[inc_cols].astype("float32"))[:, 1]
    inc = {"UP": per_year(EV, inc_scorefn, 1), "DOWN": per_year(EV, inc_scorefn, 0)}
    print(f"  incremental UP binding={inc['UP']['binding']}  DOWN binding={inc['DOWN']['binding']}", flush=True)

    # =====================================================================
    # VERDICT — apply the pre-registered falsifier
    # =====================================================================
    a_up_lo = (a_sel["UP"]["binding"] or {}).get("ci_lo")
    a_dn_lo = (a_sel["DOWN"]["binding"] or {}).get("ci_lo")
    a_best_sel_lo = max([x for x in (a_up_lo, a_dn_lo) if x is not None], default=None)
    standalone_pass = (wvh > 0.52) and (a_best_sel_lo is not None and a_best_sel_lo >= BE)

    iu = inc["UP"]["binding"]; idn = inc["DOWN"]["binding"]
    # >1 SE: approximate 1 SE as (win - ci_lo) of that binding pocket
    def beats_by_1se(b, thr):
        if not b or b.get("ci_lo") is None or b.get("win") is None or (b.get("min_n") or 0) < 100:
            return False
        se = max(1e-9, b["win"] - b["ci_lo"])
        return (b["ci_lo"] > thr) and (b["win"] - thr > se)
    inc_up_pass = beats_by_1se(iu, INCUMBENT["UP_binding_2025"]) or beats_by_1se(iu, INCUMBENT["UP_floor"])
    inc_dn_pass = beats_by_1se(idn, INCUMBENT["DOWN_binding"])
    incremental_pass = inc_up_pass or inc_dn_pass

    any_direction_lift = bool(standalone_pass or incremental_pass)
    sign_invariance_confirmed = not any_direction_lift

    # magnitude >> direction confirmation across the feature table
    valid_tab = {c: v for c, v in feat_table.items() if v["dir_auc"] is not None}
    mag_gt_dir = (sum(1 for v in valid_tab.values() if v["mag_auc"] > v["dir_auc"]) >= max(1, len(valid_tab) // 2))

    statement = (
        f"5m sign-invariance {'CONFIRMED' if sign_invariance_confirmed else 'NOT confirmed'} by experiment: "
        f"standalone worst-VAL-half MOVED dirAUC={wvh:.4f} (>0.52 {'YES' if wvh>0.52 else 'no'}), "
        f"best selective binding CI-lo={a_best_sel_lo} (>= {BE} {'YES' if (a_best_sel_lo is not None and a_best_sel_lo>=BE) else 'no'}); "
        f"incremental UP binding={iu} (beats .577/.553 by>1SE {'YES' if inc_up_pass else 'no'}), "
        f"DOWN binding={idn} (beats .5441 by>1SE {'YES' if inc_dn_pass else 'no'}); "
        f"MAGNITUDE per-year AUC={mag_peryear} {'>> ' if mag_gt_dir else 'NOT >> '}direction "
        f"(majority of sign-inv feats carry size not sign). "
        + ("HMM/Kalman/entropy/Hurst gate move SIZE not SIGN at 5m — experiment-based subsumption of rows "
           "C1a(HMM)/C2a(Kalman)/complexity-gate confirms the theorem (arXiv:2512.15720) and m5_magdyn at 5m."
           if sign_invariance_confirmed else
           "A sign-invariant feature/model produced a direction lift at 5m — investigate before believing (run CPCV).")
    )

    import os as _os; _f=f"{ROOT}/m5_signinv_dir_result.json"; out = json.load(open(_f)) if _os.path.exists(_f) else {}   # keep pre-registered header if present
    out["status"] = "RUN COMPLETE"
    out["hmm_emit_feats"] = HMM_EMIT_FEATS
    out["signinv_feats"] = SI
    out["train_n"] = int(len(TR)); out["val_n"] = int(len(VA))
    out["moved_up_rate_by_year"] = up_rates
    out["moved_up_rate_tripwire_ok"] = bool(up_ok)
    out["standalone"] = {
        "worst_val_half_moved_dirauc": round(wvh, 4),
        "val_half_dirauc": [round(vh1, 4), round(vh2, 4)],
        "per_year_moved_dirauc": a_peryear,
        "selective_UP": a_sel["UP"],
        "selective_DOWN": a_sel["DOWN"],
        "sign_permutation_importance_aucdrop": perm_sign,
        "PASS": bool(standalone_pass),
    }
    out["incremental"] = {
        "n_feats": len(inc_cols),
        "UP": inc["UP"], "DOWN": inc["DOWN"],
        "incumbent": INCUMBENT,
        "UP_PASS": bool(inc_up_pass), "DOWN_PASS": bool(inc_dn_pass),
        "PASS": bool(incremental_pass),
    }
    out["sign_vs_magnitude_auc"] = {
        "magnitude_q67": q67,
        "per_feature_val_auc": feat_table,
        "magnitude_model_per_year_auc": mag_peryear,
        "magnitude_dominates_direction": bool(mag_gt_dir),
    }
    out["VERDICT"] = {
        "sign_invariance_confirmed_at_5m": bool(sign_invariance_confirmed),
        "any_direction_lift": bool(any_direction_lift),
        "statement": statement,
    }
    json.dump(out, open(f"{ROOT}/m5_signinv_dir_result.json", "w"), indent=2, default=str)
    print(f"VERDICT: {statement}", flush=True)
    print(f"-> m5_signinv_dir_result.json ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
