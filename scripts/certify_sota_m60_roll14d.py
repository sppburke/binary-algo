#!/usr/bin/env python3
"""Strictly DEV-only (< 2026-08-01) 5-gate causal rolling-refit evaluator for [USDJPY·60m·UP].

Strictly DEV research evidence (< 2026-08-01T00:00:00Z, zero holdout touches, zero executable book writes):
- Loads ONLY pinned < 2026-08-01 DEV inputs via component-by-component O_NOFOLLOW verification
  from scripts/certify_sota_stack05.py (sealed >= 2026-08-01 holdout untouched).
- Strictly decision-time candidate selection & 60m spacing:
    * Test blocks score ALL decision-time-eligible NY rows (<50% NaN, 08:00 <= NY < 15:50,
      UTC < 21, cutoff_ok) WITHOUT peeking at future settlement tick availability (`valid_true`).
    * Training pools strictly require settled labels with `exit_true < t_refit` AND
      `exit_true < min_test_entry` plus block-purged validation (`gap = 3600s`).
    * Chronological 60m spacing (`nonoverlap_chrono`) runs BEFORE checking settlement availability;
      unsettled missing-tick slots are counted explicitly alongside settled results and conservative
      worst-case WR/EV (treating any unsettled slot as a loss).
- Evaluates the causal 14-day trailing-120-day rolling-refit `core66_nys_ens5` (cov=0.04, 60m UP)
  policy across all 4 calendar boundary shifts (+0d, +3d, +7d, +10d), plus 15 sub-block pair-paths
  of the causal rolling OOS stream (explicitly labeled rolling-OOS stability, NOT 15 independent
  refit-CPCV folds under `scripts/certificate.py`), G2 drift-matched NY hour x rv_60m placebo,
  G3 +4s/+29s entry delay & freshness subgroups, G4 16-shift circular week-shift sign-null replay,
  and G5 flat R=1.82 payout EV + retrospective DEV 4-book vs 3-book portfolio sensitivity.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import numpy as np
import pandas as pd
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import LogisticRegression, SGDClassifier
from zoneinfo import ZoneInfo

from certify_sota_stack05 import (
    CUTOFF_UTC,
    DEFAULT_DEV_DATASET_DIR,
    DEFAULT_TICKS_ROOT,
    NULL_SHIFTS_WEEKS,
    NULL_SPAN_S,
    OperationalHoldError,
    _clipped_logits,
    _eval_platt,
    intervals_overlap_any,
    load_pinned_dev_manifest,
    load_pinned_dev_parquet,
    load_pinned_dev_ticks,
    nonoverlap_chrono,
    select_core66_columns,
    settle_with_intervals,
    sha256_bytes,
)

SCHEMA = "sota-m60-roll14d-5gate-dossier/v1"
BOOK_ID = "USDJPY.sota_roll14d_m60_up.v1"
PAIR = "USDJPY"
HORIZON_MIN = 60
HORIZON_S = 3600
TOL_S = 60
COV = 0.04
QUOTED_PAYOUT_RATIO = 1.82
MIN_PAYOUT_RATIO = 1.78
EV_MARGIN = 0.005
PROGRAM_HURDLE_WR = 0.541  # Program 0.541 ties-lose WR hurdle; flat R=1.82 breakeven is 1/1.82 = 0.54945
TRAIN_DAYS = 120
REFIT_CADENCE_DAYS = 14
BOUNDARY_SHIFTS_DAYS: tuple[int, ...] = (0, 3, 7, 10)
DELAYS_EXTRA_S: tuple[int, ...] = (0, 2, 4, 9, 29)
NY_TZ = ZoneInfo("America/New_York")
MIN_PATH_TRADES = 25


def build_decision_eligible_indices_60m(
    df_dev: pd.DataFrame,
    fcols: list[str],
    *,
    cutoff_utc: int = CUTOFF_UTC,
) -> np.ndarray:
    """Strictly decision-time Deriv 60m tradeable NY session mask (08:00 <= NY < 15:50, UTC < 21, <50% NaN).

    Intentionally does NOT peek at future settlement tick availability (`valid_true` / `ret_fwd_true`).
    """
    ts = df_dev["decision_ts"].to_numpy(dtype=np.int64)
    if len(ts) >= 2 and not bool(np.all(ts[:-1] < ts[1:])):
        raise OperationalHoldError("df_dev decision_ts is not strictly ascending")
    X = df_dev[fcols].to_numpy(dtype=np.float64)
    row_ok = np.mean(~np.isfinite(X), axis=1) < 0.50
    dt_utc = pd.to_datetime(ts, unit="s", utc=True)
    dt_ny = dt_utc.tz_convert(NY_TZ)
    wd = dt_ny.dayofweek.to_numpy()
    ny_mod = dt_ny.hour.to_numpy() * 60 + dt_ny.minute.to_numpy()
    # Rollover cutoff for 60m: 16:50 - 60m = 15:50 NY (950 min)
    in_ny = (wd < 5) & (ny_mod >= 480) & (ny_mod < 950)
    utc_ok = dt_utc.hour.to_numpy() < 21
    cutoff_ok = (ts + 1 + 29 + HORIZON_S + TOL_S) < int(cutoff_utc)
    elig = row_ok & in_ny & utc_ok & cutoff_ok
    return np.flatnonzero(elig).astype(np.int64)


def fit_rolling_window_ens5(
    X_all: np.ndarray,
    ts_all: np.ndarray,
    entry_ts: np.ndarray,
    exit_ts: np.ndarray,
    ret_arr: np.ndarray,
    pool_idx: np.ndarray,
    test_idx: np.ndarray,
    *,
    cov: float = COV,
    gap: int = HORIZON_S,
    val_frac: float = 0.15,
    min_val: int = 500,
    donor_entry_ts: np.ndarray | None = None,
    donor_exit_ts: np.ndarray | None = None,
    return_estimators: bool = False,
) -> dict[str, Any]:
    """Fit 5-seed RBF Nystroem + SGD on a causal trailing pool with block-purged val & Platt calibration."""
    pool = np.asarray(pool_idx, dtype=np.int64)
    is_null = donor_entry_ts is not None and donor_exit_ts is not None
    if len(pool) < min_val + 1000:
        return {"ok": False, "reason": "pool_too_small"}

    n_val = max(int(min_val), int(round(len(pool) * float(val_frac))))
    val_idx = pool[-n_val:]
    val_first_ts = int(ts_all[val_idx[0]])
    if is_null:
        assert donor_entry_ts is not None and donor_exit_ts is not None
        # In circular-shift null mode, enforce receiver-time block purge AND strict donor-interval overlap purge
        # (never index real entry_ts/exit_ts on unsettled real rows whose shifted donor label is valid!).
        fit_all_idx = pool[ts_all[pool] < val_first_ts - int(gap)]
        conflict = intervals_overlap_any(
            donor_entry_ts[fit_all_idx],
            donor_exit_ts[fit_all_idx],
            donor_entry_ts[val_idx],
            donor_exit_ts[val_idx],
            pad=int(gap),
        )
        fit_all_idx = fit_all_idx[~conflict]
    else:
        val_first_entry = int(entry_ts[val_idx[0]])
        fit_all_idx = pool[
            (ts_all[pool] < val_first_ts - int(gap))
            & (exit_ts[pool] > 0)
            & (exit_ts[pool] < val_first_ts)
            & (exit_ts[pool] < val_first_entry)
        ]

    if len(fit_all_idx) < 800:
        return {"ok": False, "reason": "fit_pool_too_small"}

    mv_in_fit = ret_arr[fit_all_idx] != 0.0
    mv_fit_idx = fit_all_idx[mv_in_fit]
    if len(mv_fit_idx) < 400:
        return {"ok": False, "reason": "mv_fit_too_small"}
    yf = (ret_arr[mv_fit_idx] > 0.0).astype(np.int64)
    if min(int(np.sum(yf)), int(len(yf) - np.sum(yf))) < 20:
        return {"ok": False, "reason": "fit_class_imbalance"}

    y_val = (ret_arr[val_idx] > 0.0).astype(np.int64)
    if min(int(np.sum(y_val)), int(len(y_val) - np.sum(y_val))) < 10:
        return {"ok": False, "reason": "val_class_imbalance"}

    Xf_all = X_all[fit_all_idx]
    med = np.nanmedian(np.where(np.isfinite(Xf_all), Xf_all, np.nan), axis=0)
    med = np.where(np.isfinite(med), med, 0.0).astype(np.float64)
    Zf_all = np.where(np.isfinite(Xf_all), Xf_all, med).astype(np.float64)
    mu = Zf_all.mean(axis=0).astype(np.float64)
    sd = (Zf_all.std(axis=0) + 1e-12).astype(np.float64)
    Zf_all = np.clip((Zf_all - mu) / sd, -5.0, 5.0)

    Xv = X_all[val_idx]
    Zv = np.clip((np.where(np.isfinite(Xv), Xv, med).astype(np.float64) - mu) / sd, -5.0, 5.0)
    has_test = len(test_idx) > 0
    if has_test:
        Xt = X_all[test_idx]
        Zt = np.clip((np.where(np.isfinite(Xt), Xt, med).astype(np.float64) - mu) / sd, -5.0, 5.0)
    else:
        Zt = np.empty((0, X_all.shape[1]), dtype=np.float64)

    ny_list: list[Any] = []
    sgd_list: list[Any] = []
    p_val = np.zeros(len(val_idx), dtype=np.float64)
    p_test = np.zeros(len(test_idx), dtype=np.float64)
    gamma = 1.0 / max(1, X_all.shape[1])

    for seed in range(5):
        rng = np.random.default_rng(1000 + seed * 37)
        if len(Zf_all) > 45_000:
            sub = rng.choice(len(Zf_all), 45_000, replace=False)
        else:
            sub = np.arange(len(Zf_all))
        ny = Nystroem(kernel="rbf", gamma=gamma, n_components=500, random_state=seed)
        ny.fit(Zf_all[sub])
        Fk_all = ny.transform(Zf_all)
        sgd = SGDClassifier(loss="log_loss", penalty="l2", alpha=1e-4, max_iter=30, tol=1e-4, random_state=seed)
        sgd.fit(Fk_all[mv_in_fit], yf)
        p_val += sgd.predict_proba(ny.transform(Zv))[:, 1]
        if has_test:
            p_test += sgd.predict_proba(ny.transform(Zt))[:, 1]
        if return_estimators:
            ny_list.append(ny)
            sgd_list.append(sgd)

    p_val /= 5.0
    if has_test:
        p_test /= 5.0

    if not bool(np.all(np.isfinite(p_val))) or (has_test and not bool(np.all(np.isfinite(p_test)))):
        return {"ok": False, "reason": "nonfinite_scores"}

    conf_thr = float(np.quantile(np.abs(p_val - 0.5), 1.0 - float(cov), method="linear"))
    cal_clf = LogisticRegression(C=1.0, max_iter=1000).fit(_clipped_logits(p_val).reshape(-1, 1), y_val)
    p_coef = float(cal_clf.coef_[0, 0])
    p_int = float(cal_clf.intercept_[0])
    platt_ok = bool(math.isfinite(p_coef) and p_coef > 0.0 and math.isfinite(p_int))

    out: dict[str, Any] = {
        "ok": True,
        "conf_thr": conf_thr,
        "platt_ok": platt_ok,
        "platt_coef": p_coef if math.isfinite(p_coef) else None,
        "platt_intercept": p_int if math.isfinite(p_int) else None,
        "n_fit": int(len(fit_all_idx)),
        "n_val": int(len(val_idx)),
        "p_val": p_val,
        "p_test": p_test,
    }
    if return_estimators:
        out.update({"med": med, "mu": mu, "sd": sd, "ny_list": ny_list, "sgd_list": sgd_list, "val_idx": val_idx})
    return out


def _bootstrap_ev_ci95(wins_01: np.ndarray, payout: float = QUOTED_PAYOUT_RATIO, n_boot: int = 5000, seed: int = 7) -> list[float] | None:
    if len(wins_01) == 0:
        return None
    rng = np.random.default_rng(seed)
    ev_i = wins_01.astype(np.float64) * float(payout) - 1.0
    n = len(ev_i)
    means = np.array([ev_i[rng.integers(0, n, n)].mean() for _ in range(n_boot)], dtype=np.float64)
    return [round(float(np.percentile(means, 2.5, method="linear")), 6), round(float(np.percentile(means, 97.5, method="linear")), 6)]


def _compute_hr_rv_excess(
    chosen_global_idx: np.ndarray,
    oos_universe_idx: np.ndarray,
    ts_all: np.ndarray,
    rv60_all: np.ndarray,
    ret_arr: np.ndarray,
) -> dict[str, Any]:
    """Compute exact NY hour x rv_60m tercile drift-matched excess win rate."""
    if len(chosen_global_idx) == 0 or len(oos_universe_idx) < 30:
        return {"excess_hr_rv": None, "baseline_wr": None, "realized_wr": None, "matched_n": 0}
    u_ts = ts_all[oos_universe_idx]
    u_rv = rv60_all[oos_universe_idx]
    u_win = (ret_arr[oos_universe_idx] > 0.0).astype(np.float64)
    u_hr = pd.to_datetime(u_ts, unit="s", utc=True).tz_convert(NY_TZ).hour.to_numpy()
    fin_rv = u_rv[np.isfinite(u_rv)]
    if len(fin_rv) < 30:
        return {"excess_hr_rv": None, "baseline_wr": None, "realized_wr": None, "matched_n": 0}
    q1, q2 = float(np.percentile(fin_rv, 33.333333)), float(np.percentile(fin_rv, 66.666667))
    if not (math.isfinite(q1) and math.isfinite(q2) and q2 > q1):
        return {"excess_hr_rv": None, "baseline_wr": None, "realized_wr": None, "matched_n": 0}
    u_bin = np.where(u_rv <= q1, 0, np.where(u_rv <= q2, 1, 2))
    strata: dict[tuple[int, int], float] = {}
    for hr in sorted(set(int(h) for h in u_hr.tolist())):
        for b in (0, 1, 2):
            m = (u_hr == hr) & (u_bin == b) & np.isfinite(u_rv)
            if int(np.sum(m)) > 0:
                strata[(hr, b)] = float(np.mean(u_win[m]))
    overall_base = float(np.mean(u_win))

    c_ts = ts_all[chosen_global_idx]
    c_rv = rv60_all[chosen_global_idx]
    c_win = (ret_arr[chosen_global_idx] > 0.0).astype(np.float64)
    c_hr = pd.to_datetime(c_ts, unit="s", utc=True).tz_convert(NY_TZ).hour.to_numpy()
    c_bin = np.where(c_rv <= q1, 0, np.where(c_rv <= q2, 1, 2))
    base_vec = np.array(
        [strata.get((int(h), int(b)), overall_base) if math.isfinite(float(r)) else overall_base for h, b, r in zip(c_hr, c_bin, c_rv)],
        dtype=np.float64,
    )
    diff = c_win - base_vec
    return {
        "excess_hr_rv": round(float(np.mean(diff)), 6),
        "baseline_wr": round(float(np.mean(base_vec)), 6),
        "realized_wr": round(float(np.mean(c_win)), 6),
        "matched_n": int(len(chosen_global_idx)),
    }


def _evaluate_rolling_stream_once(
    *,
    X_all: np.ndarray,
    ts_all: np.ndarray,
    rv60_all: np.ndarray,
    entry_true: np.ndarray,
    exit_true: np.ndarray,
    valid_arr: np.ndarray,
    ret_arr: np.ndarray,
    decision_eligible_idx: np.ndarray,
    shift_days: int = 0,
    donor_entry: np.ndarray | None = None,
    donor_exit: np.ndarray | None = None,
    eval_delays: dict[int, tuple[np.ndarray, np.ndarray]] | None = None,
    tick_ts: np.ndarray | None = None,
    tick_px: np.ndarray | None = None,
    selection_mode: str = "platt_ev",
    also_raw_top_cov: bool = False,
) -> dict[str, Any]:
    """Run one complete strictly-causal 14-day trailing-120-day rolling-refit OOS stream.

    Candidate selection and chronological 60m spacing operate strictly on decision-time-eligible
    rows (no peek at future `valid_arr`/`ret_arr`); training pools require settled labels strictly
    before `t_refit` (`exit_true < t_cur` AND `exit_true < min_test_entry`).
    """
    is_null = donor_entry is not None and donor_exit is not None
    elig_ts = ts_all[decision_eligible_idx]
    elig_settled = valid_arr[decision_eligible_idx] & np.isfinite(ret_arr[decision_eligible_idx])
    elig_exit = None if is_null else exit_true[decision_eligible_idx]
    t_first = int(elig_ts[0])
    t_last = int(elig_ts[-1])
    t_oos_start = t_first + TRAIN_DAYS * 86_400 + int(shift_days) * 86_400
    step_s = REFIT_CADENCE_DAYS * 86_400
    train_s = TRAIN_DAYS * 86_400

    # Collect causal test predictions & thresholds per block (decision-time universe only!)
    blocks: list[dict[str, Any]] = []
    cand_indices: list[int] = []
    raw_cand_indices: list[int] = []
    oos_universe_list: list[np.ndarray] = []

    t_cur = t_oos_start
    while t_cur <= t_last:
        t_end = min(t_cur + step_s, CUTOFF_UTC)
        test_mask = (elig_ts >= t_cur) & (elig_ts < t_end)
        test_idx = decision_eligible_idx[test_mask]
        if len(test_idx) == 0:
            t_cur += step_s
            continue
        oos_universe_list.append(test_idx)

        t_train_lo = t_cur - train_s
        if is_null:
            assert donor_entry is not None and donor_exit is not None
            # Circular week-shift null: enforce receiver-time causal window + strict donor-interval purge vs test
            pool_mask = elig_settled & (elig_ts >= t_train_lo) & (elig_ts < t_cur - HORIZON_S)
            pool_cand = decision_eligible_idx[pool_mask]
            test_settled_idx = test_idx[valid_arr[test_idx] & np.isfinite(ret_arr[test_idx])]
            if len(test_settled_idx) > 0 and len(pool_cand) > 0:
                ov_d = intervals_overlap_any(
                    donor_entry[pool_cand],
                    donor_exit[pool_cand],
                    donor_entry[test_settled_idx],
                    donor_exit[test_settled_idx],
                    pad=HORIZON_S,
                )
                pool_idx = pool_cand[~ov_d]
            else:
                pool_idx = pool_cand
        else:
            # Real DEV stream: require training labels settled strictly before t_refit AND earliest positive test entry
            valid_test_entries = entry_true[test_idx][(entry_true[test_idx] > 0) & valid_arr[test_idx]]
            min_test_entry = int(min(t_cur + 1, int(np.min(valid_test_entries)))) if len(valid_test_entries) else (t_cur + 1)
            pool_mask = (
                elig_settled
                & (elig_ts >= t_train_lo)
                & (elig_ts < t_cur - HORIZON_S)
                & (elig_exit > 0)
                & (elig_exit < t_cur)
                & (elig_exit < min_test_entry)
            )
            pool_idx = decision_eligible_idx[pool_mask]

        fit = fit_rolling_window_ens5(
            X_all,
            ts_all,
            entry_true,
            exit_true,
            ret_arr,
            pool_idx,
            test_idx,
            cov=COV,
            gap=HORIZON_S,
            donor_entry_ts=donor_entry,
            donor_exit_ts=donor_exit,
        )
        if not fit["ok"]:
            blocks.append({"t_refit": int(t_cur), "t_end": int(t_end), "ok": False, "reason": fit["reason"], "n_test": int(len(test_idx))})
            t_cur += step_s
            continue

        p_test = fit["p_test"]
        thr = float(fit["conf_thr"])
        raw_mask = np.isfinite(p_test) & (p_test > 0.5) & (np.abs(p_test - 0.5) >= thr)
        if also_raw_top_cov:
            raw_cand_indices.extend(test_idx[raw_mask].tolist())
        if selection_mode == "platt_ev":
            if not fit["platt_ok"]:
                sel_mask = np.zeros(len(test_idx), dtype=bool)
            else:
                p_cal = _eval_platt(p_test, float(fit["platt_coef"]), float(fit["platt_intercept"]))
                sel_mask = raw_mask & np.isfinite(p_cal) & ((p_cal * QUOTED_PAYOUT_RATIO - 1.0) >= EV_MARGIN)
        elif selection_mode == "raw_top_cov":
            sel_mask = raw_mask
        else:
            raise ValueError(f"unsupported selection_mode {selection_mode!r}")

        cand_indices.extend(test_idx[sel_mask].tolist())
        blocks.append(
            {
                "t_refit": int(t_cur),
                "t_end": int(t_end),
                "ok": True,
                "n_fit": fit["n_fit"],
                "n_val": fit["n_val"],
                "n_test": int(len(test_idx)),
                "conf_thr": round(thr, 6),
                "platt_ok": bool(fit["platt_ok"]),
                "platt_coef": round(float(fit["platt_coef"]), 6) if fit["platt_coef"] is not None else None,
                "n_cand": int(np.sum(sel_mask)),
            }
        )
        t_cur += step_s

    oos_universe_all = np.concatenate(oos_universe_list) if oos_universe_list else np.empty(0, dtype=np.int64)
    oos_universe = (
        oos_universe_all[valid_arr[oos_universe_all] & np.isfinite(ret_arr[oos_universe_all])]
        if len(oos_universe_all)
        else np.empty(0, dtype=np.int64)
    )

    def _summarize_cands(c_list: list[int], mode_name: str) -> dict[str, Any]:
        cand_arr = np.asarray(sorted(set(c_list)), dtype=np.int64)
        # Apply 60m chronological spacing strictly on decision-time candidates BEFORE checking settlement ticks!
        if len(cand_arr) > 0:
            keep_local = nonoverlap_chrono(ts_all[cand_arr], np.ones(len(cand_arr), dtype=bool), gap=HORIZON_S)
            chosen_all = cand_arr[keep_local]
        else:
            chosen_all = np.empty(0, dtype=np.int64)

        n_selected_total = int(len(chosen_all))
        settled_mask = (valid_arr[chosen_all] & np.isfinite(ret_arr[chosen_all])) if n_selected_total > 0 else np.empty(0, dtype=bool)
        chosen = chosen_all[settled_mask] if n_selected_total > 0 else np.empty(0, dtype=np.int64)
        n = int(len(chosen))
        n_unsettled = n_selected_total - n

        wins_arr = (ret_arr[chosen] > 0.0).astype(np.int64) if n > 0 else np.empty(0, dtype=np.int64)
        wins = int(np.sum(wins_arr))
        wr = float(wins / n) if n > 0 else None
        ev = float((wins / n) * QUOTED_PAYOUT_RATIO - 1.0) if n > 0 else None
        cons_wr = float(wins / n_selected_total) if n_selected_total > 0 else None
        cons_ev = float((wins / n_selected_total) * QUOTED_PAYOUT_RATIO - 1.0) if n_selected_total > 0 else None
        ev_ci95 = _bootstrap_ev_ci95(wins_arr, payout=QUOTED_PAYOUT_RATIO) if n > 0 else None
        placebo = _compute_hr_rv_excess(chosen, oos_universe, ts_all, rv60_all, ret_arr)

        # 6 chronological sub-blocks -> 15 pair-paths (CPCV-style tail stability over strictly causal rolling OOS!)
        sub_paths: list[dict[str, Any]] = []
        if len(oos_universe) >= 6:
            t_lo_u = int(ts_all[oos_universe[0]])
            t_hi_u = int(ts_all[oos_universe[-1]]) + 1
            edges = np.linspace(t_lo_u, t_hi_u, 7, dtype=np.int64)
            c_ts = ts_all[chosen]
            u_ts = ts_all[oos_universe]
            for ga in range(6):
                for gb in range(ga + 1, 6):
                    ma = (c_ts >= edges[ga]) & (c_ts < edges[ga + 1])
                    mb = (c_ts >= edges[gb]) & (c_ts < edges[gb + 1])
                    p_ch = chosen[ma | mb]
                    ua = (u_ts >= edges[ga]) & (u_ts < edges[ga + 1])
                    ub = (u_ts >= edges[gb]) & (u_ts < edges[gb + 1])
                    p_uni = oos_universe[ua | ub]
                    pn = int(len(p_ch))
                    pw = int(np.sum(ret_arr[p_ch] > 0.0)) if pn > 0 else 0
                    pwr = float(pw / pn) if pn > 0 else None
                    pev = float((pw / pn) * QUOTED_PAYOUT_RATIO - 1.0) if pn > 0 else None
                    p_plc = _compute_hr_rv_excess(p_ch, p_uni, ts_all, rv60_all, ret_arr)
                    p_yrs = float(((edges[ga + 1] - edges[ga]) + (edges[gb + 1] - edges[gb])) / 31_557_600.0)
                    sub_paths.append(
                        {
                            "test_groups": [ga, gb],
                            "n": pn,
                            "wins": pw,
                            "wr": round(pwr, 6) if pwr is not None else None,
                            "ev": round(pev, 6) if pev is not None else None,
                            "excess_hr_rv": p_plc["excess_hr_rv"],
                            "supported": bool(pn >= MIN_PATH_TRADES),
                            "path_years": round(p_yrs, 6),
                        }
                    )

        sup_paths = [p for p in sub_paths if p["supported"]]
        p10_wr = float(np.percentile([p["wr"] for p in sup_paths], 10, method="linear")) if sup_paths else None
        p10_ev = float(np.percentile([p["ev"] for p in sup_paths], 10, method="linear")) if sup_paths else None
        paths_clear = sum(1 for p in sup_paths if p["wr"] is not None and p["wr"] >= PROGRAM_HURDLE_WR)
        paths_ev_pos = sum(1 for p in sup_paths if p["ev"] is not None and p["ev"] > 0.0)
        paths_plc_pos = sum(1 for p in sup_paths if p["excess_hr_rv"] is not None and p["excess_hr_rv"] > 0.0)
        mean_plc_paths = (
            float(np.mean([p["excess_hr_rv"] for p in sup_paths if p["excess_hr_rv"] is not None]))
            if any(p["excess_hr_rv"] is not None for p in sup_paths)
            else None
        )

        delay_summary: dict[str, Any] = {}
        if eval_delays is not None and n > 0:
            for d_s, (d_ret, d_val) in eval_delays.items():
                ok_d = d_val[chosen] & np.isfinite(d_ret[chosen])
                dn = int(np.sum(ok_d))
                dw = int(np.sum(d_ret[chosen[ok_d]] > 0.0)) if dn > 0 else 0
                dwr = float(dw / dn) if dn > 0 else None
                dev = float((dw / dn) * QUOTED_PAYOUT_RATIO - 1.0) if dn > 0 else None
                delay_summary[str(d_s)] = {
                    "n": dn,
                    "wins": dw,
                    "wr": round(dwr, 6) if dwr is not None else None,
                    "ev": round(dev, 6) if dev is not None else None,
                }

        freshness_summary: dict[str, Any] = {}
        if tick_ts is not None and tick_px is not None and n > 0:
            chg_mask = np.r_[True, np.diff(tick_px) != 0.0]
            chg_ts = tick_ts[chg_mask]
            q_idx = np.searchsorted(tick_ts, ts_all[chosen], side="left") - 1
            c_idx = np.searchsorted(chg_ts, ts_all[chosen], side="left") - 1
            q_age = np.where(q_idx >= 0, ts_all[chosen] - tick_ts[np.clip(q_idx, 0, len(tick_ts) - 1)], 999999)
            c_age = np.where(c_idx >= 0, ts_all[chosen] - chg_ts[np.clip(c_idx, 0, len(chg_ts) - 1)], 999999)
            for label, mask_f in (("fresh_quote_le2s", q_age <= 2), ("fresh_midchg_le2s", c_age <= 2)):
                sub_ch = chosen[mask_f]
                fn = int(len(sub_ch))
                fw = int(np.sum(ret_arr[sub_ch] > 0.0)) if fn > 0 else 0
                fwr = float(fw / fn) if fn > 0 else None
                fev = float((fw / fn) * QUOTED_PAYOUT_RATIO - 1.0) if fn > 0 else None
                fplc = _compute_hr_rv_excess(sub_ch, oos_universe, ts_all, rv60_all, ret_arr)
                freshness_summary[label] = {
                    "n": fn,
                    "wins": fw,
                    "wr": round(fwr, 6) if fwr is not None else None,
                    "ev": round(fev, 6) if fev is not None else None,
                    "excess_hr_rv": fplc["excess_hr_rv"],
                }

        oos_years = float((ts_all[oos_universe[-1]] - ts_all[oos_universe[0]]) / 31_557_600.0) if len(oos_universe) >= 2 else 0.0
        est_trades_yr = float(n / oos_years) if oos_years > 0.0 else 0.0

        return {
            "shift_days": int(shift_days),
            "selection_mode": mode_name,
            "oos_start_utc": dt.datetime.fromtimestamp(int(ts_all[oos_universe[0]]), tz=dt.timezone.utc).isoformat() if len(oos_universe) else None,
            "oos_end_utc": dt.datetime.fromtimestamp(int(ts_all[oos_universe[-1]]), tz=dt.timezone.utc).isoformat() if len(oos_universe) else None,
            "oos_years": round(oos_years, 6),
            "n_blocks": len(blocks),
            "n_blocks_ok": sum(1 for b in blocks if b["ok"]),
            "n_selected_total": n_selected_total,
            "n": n,
            "n_unsettled_missing_tick": n_unsettled,
            "wins": wins,
            "wr": round(wr, 6) if wr is not None else None,
            "ev": round(ev, 6) if ev is not None else None,
            "conservative_wr_unsettled_as_loss": round(cons_wr, 6) if cons_wr is not None else None,
            "conservative_ev_unsettled_as_loss": round(cons_ev, 6) if cons_ev is not None else None,
            "ev_ci95": ev_ci95,
            "est_trades_per_year": round(est_trades_yr, 2),
            "placebo": placebo,
            "rolling_15paths": {
                "paths_counted": len(sup_paths),
                "paths_clear_0541": paths_clear,
                "paths_ev_positive": paths_ev_pos,
                "paths_placebo_positive": paths_plc_pos,
                "p10_wr": round(p10_wr, 6) if p10_wr is not None else None,
                "p10_ev": round(p10_ev, 6) if p10_ev is not None else None,
                "mean_excess_hr_rv": round(mean_plc_paths, 6) if mean_plc_paths is not None else None,
                "paths": sub_paths,
            },
            "delays": delay_summary,
            "freshness": freshness_summary,
            "trade_ts": [int(x) for x in ts_all[chosen].tolist()],
            "trade_wins": [int(x) for x in wins_arr.tolist()],
            "blocks": blocks,
        }

    res_main = _summarize_cands(cand_indices, selection_mode)
    if also_raw_top_cov:
        res_main["raw_top_cov_comparator"] = _summarize_cands(raw_cand_indices, "raw_top_cov")
    return res_main


# Shared read-only globals populated BEFORE ProcessPoolExecutor(fork) to avoid 1.7 GB pickle overhead!
_SHARED: dict[str, Any] = {}


def _shift_worker(sd: int) -> tuple[int, dict[str, Any]]:
    t_s = time.monotonic()
    res = _evaluate_rolling_stream_once(
        X_all=_SHARED["X_all"],
        ts_all=_SHARED["ts_all"],
        rv60_all=_SHARED["rv60_all"],
        entry_true=_SHARED["entry_true"],
        exit_true=_SHARED["exit_true"],
        valid_arr=_SHARED["valid_true"],
        ret_arr=_SHARED["ret_fwd_true"],
        decision_eligible_idx=_SHARED["decision_eligible_idx"],
        shift_days=int(sd),
        eval_delays=_SHARED["eval_delays"],
        tick_ts=_SHARED["tick_ts"],
        tick_px=_SHARED["tick_px"],
        selection_mode=_SHARED["selection_mode"],
        also_raw_top_cov=(int(sd) == 0),
    )
    print(f"[m60-cert] shift +{sd}d done in {time.monotonic() - t_s:.1f}s: n_sel={res['n_selected_total']} n={res['n']} unset={res['n_unsettled_missing_tick']} wr={res['wr']} ev={res['ev']} ci95={res['ev_ci95']}", flush=True)
    return int(sd), res


def _null_worker(w: int) -> dict[str, Any]:
    t_s = time.monotonic()
    X_all = _SHARED["X_all"]
    ts_all = _SHARED["ts_all"]
    rv60_all = _SHARED["rv60_all"]
    entry_true = _SHARED["entry_true"]
    exit_true = _SHARED["exit_true"]
    ret_fwd_true = _SHARED["ret_fwd_true"]
    valid_true = _SHARED["valid_true"]
    decision_eligible_idx = _SHARED["decision_eligible_idx"]
    ts_to_row = _SHARED["ts_to_row"]

    t0 = int(ts_all[0])
    elig_ts = ts_all[decision_eligible_idx]
    tgt_ts = t0 + ((elig_ts - t0 + int(w) * 604800) % NULL_SPAN_S)
    n_rows = len(ts_all)
    ret_donor = np.full(n_rows, np.nan, dtype=np.float64)
    valid_donor = np.zeros(n_rows, dtype=bool)
    entry_donor = np.full(n_rows, -1, dtype=np.int64)
    exit_donor = np.full(n_rows, -1, dtype=np.int64)
    for pos_i, tgt in enumerate(tgt_ts):
        j = ts_to_row.get(int(tgt), -1)
        if j >= 0 and valid_true[j] and math.isfinite(float(ret_fwd_true[j])):
            r_idx = int(decision_eligible_idx[pos_i])
            ret_donor[r_idx] = ret_fwd_true[j]
            valid_donor[r_idx] = True
            entry_donor[r_idx] = entry_true[j]
            exit_donor[r_idx] = exit_true[j]

    res = _evaluate_rolling_stream_once(
        X_all=X_all,
        ts_all=ts_all,
        rv60_all=rv60_all,
        entry_true=entry_true,
        exit_true=exit_true,
        valid_arr=valid_donor,
        ret_arr=ret_donor,
        decision_eligible_idx=decision_eligible_idx,
        shift_days=0,
        donor_entry=entry_donor,
        donor_exit=exit_donor,
        selection_mode="raw_top_cov",
    )
    ev_ci = res["ev_ci95"]
    plc_ex = (res["placebo"] or {}).get("excess_hr_rv")
    print(f"[m60-cert] null W={w} done in {time.monotonic() - t_s:.1f}s: n={res['n']} wr={res['wr']} excess={plc_ex}", flush=True)
    return {
        "shift_weeks": int(w),
        "n": res["n"],
        "wins": res["wins"],
        "wr": res["wr"],
        "ev": res["ev"],
        "ev_ci95": ev_ci,
        "excess_hr_rv": plc_ex,
        "readable": bool(res["n"] >= 50),
        "joint_pass": bool(
            res["n"] >= 50
            and ev_ci is not None
            and ev_ci[0] > 0.0
            and plc_ex is not None
            and plc_ex >= 0.02
        ),
    }


def _evaluate_4book_incremental_portfolio(
    repo_root: Path,
    m60_trade_ts: list[int],
    m60_trade_wins: list[int],
    m60_p10_wr: float,
    m60_oos_start_ep: int,
) -> dict[str, Any]:
    """Retrospective DEV portfolio sensitivity comparing live 3-book vs 4-book (+ m60_roll14d) on g2..g5.

    NOTE: Retrospective sensitivity ONLY (Kelly fractions use full-DEV p10s, not strictly causal
    pre-period sizing). Reports BOTH the full 6-path g2..g5 window (disclosing that g2 begins
    2025-12-07T08:00Z before m60's 120d warmup ends) AND the strict common window filtering ALL 4 books
    to `ts >= m60_oos_start_ep` with path_years scaled to the exact common duration.
    """
    from certify_sota_stack05 import SHARED_GROUP_BOUNDS_UTC
    from replay_sota_portfolio import INITIAL_CASH, PATH_YEARS, reduce_scenario_paths, simulate_path_events

    if not m60_trade_ts:
        return {"available": False, "reason": "no_m60_trades"}

    post_warmup_pairs: list[tuple[int, int]] = [(a, b) for a in range(2, 6) for b in range(a + 1, 6)]
    live_specs = [
        ("ens5_m30", "results/json/deriv_5gate_USDJPY_sota_ens5_m30_up_v1_result.json", 1800, 1.82),
        ("stack05_m30", "results/json/deriv_5gate_USDJPY_sota_stack05_m30_up_v2_result.json", 1800, 1.82),
        ("ens5_m15", "results/json/deriv_5gate_USDJPY_sota_ens5_m15_up_v1_result.json", 900, 1.776447),
    ]
    live_path_map: dict[str, dict[str, dict[str, Any]]] = {}
    book_kelly: dict[int, float] = {}
    for prio, (sid, rel_p, hor_s, pay_r) in enumerate(live_specs):
        dos = json.loads((repo_root / rel_p).read_text(encoding="utf-8"))
        g1 = dos["G1_cpcv"]
        p10_b = float(g1["p10_wr"])
        book_kelly[prio] = min(0.08, 0.25 * max(0.0, (p10_b * pay_r - 1.0) / (pay_r - 1.0)))
        live_path_map[sid] = {f"{p['test_groups'][0]}_{p['test_groups'][1]}": p for p in g1["paths"]}

    prio_m60 = 3
    book_kelly[prio_m60] = min(0.08, 0.25 * max(0.0, (float(m60_p10_wr) * QUOTED_PAYOUT_RATIO - 1.0) / (QUOTED_PAYOUT_RATIO - 1.0)))

    m60_arr_ts = np.asarray(m60_trade_ts, dtype=np.int64)
    m60_arr_w = np.asarray(m60_trade_wins, dtype=np.int64)
    path_ids = [f"{ga}_{gb}" for ga, gb in post_warmup_pairs]

    def _run_window(strict_common: bool) -> tuple[dict[str, Any], dict[str, float | None]]:
        corr_lists: dict[str, list[float]] = {sid: [] for sid, _, _, _ in live_specs}
        sims: dict[str, Any] = {}
        for label, include_m60, cap in (
            ("3b_live_cap3", False, 3),
            ("4b_with_roll14d_m60_cap3", True, 3),
            ("4b_with_roll14d_m60_cap4", True, 4),
        ):
            flat_res: list[dict[str, Any]] = []
            qk_res: list[dict[str, Any]] = []
            tot_tr_f = tot_sk_f = tot_tr_k = tot_sk_k = 0
            tot_yrs = 0.0
            for ga, gb in post_warmup_pairs:
                pid = f"{ga}_{gb}"
                lo_a, hi_a = SHARED_GROUP_BOUNDS_UTC[ga], SHARED_GROUP_BOUNDS_UTC[ga + 1]
                lo_b, hi_b = SHARED_GROUP_BOUNDS_UTC[gb], SHARED_GROUP_BOUNDS_UTC[gb + 1]
                eff_lo_a = max(lo_a, m60_oos_start_ep) if strict_common else lo_a
                eff_lo_b = max(lo_b, m60_oos_start_ep) if strict_common else lo_b
                eff_s = max(0, hi_a - eff_lo_a) + max(0, hi_b - eff_lo_b)
                p_yrs = float(eff_s / 31_557_600.0) if strict_common else float(PATH_YEARS)
                tot_yrs += p_yrs
                evs: list[tuple[int, int, int, int, float]] = []
                dp_map: dict[str, dict[str, float]] = {}
                for prio, (sid, _, hor_s, pay_r) in enumerate(live_specs):
                    p_obj = live_path_map[sid][pid]
                    dp_s: dict[str, float] = {}
                    for ti, wi in zip(p_obj["trade_ts"], p_obj["trade_wins"]):
                        t_int = int(ti)
                        if strict_common and t_int < m60_oos_start_ep:
                            continue
                        evs.append((t_int, prio, hor_s, int(wi), pay_r))
                        dstr = dt.datetime.fromtimestamp(t_int, tz=dt.timezone.utc).astimezone(NY_TZ).date().isoformat()
                        dp_s[dstr] = dp_s.get(dstr, 0.0) + (pay_r - 1.0 if int(wi) == 1 else -1.0)
                    dp_map[sid] = dp_s

                in_m60 = ((m60_arr_ts >= eff_lo_a) & (m60_arr_ts < hi_a)) | ((m60_arr_ts >= eff_lo_b) & (m60_arr_ts < hi_b))
                dp_m60: dict[str, float] = {}
                for ti, wi in zip(m60_arr_ts[in_m60].tolist(), m60_arr_w[in_m60].tolist()):
                    if include_m60:
                        evs.append((int(ti), prio_m60, HORIZON_S, int(wi), QUOTED_PAYOUT_RATIO))
                    dstr = dt.datetime.fromtimestamp(int(ti), tz=dt.timezone.utc).astimezone(NY_TZ).date().isoformat()
                    dp_m60[dstr] = dp_m60.get(dstr, 0.0) + (QUOTED_PAYOUT_RATIO - 1.0 if int(wi) == 1 else -1.0)

                if label == "4b_with_roll14d_m60_cap4":
                    days_p = sorted(set(dp_m60.keys()).union(*[set(d.keys()) for d in dp_map.values()]))
                    vm = np.array([dp_m60.get(d, 0.0) for d in days_p], dtype=np.float64)
                    for sid, _, _, _ in live_specs:
                        vo = np.array([dp_map[sid].get(d, 0.0) for d in days_p], dtype=np.float64)
                        if np.std(vm) > 0 and np.std(vo) > 0:
                            corr_lists[sid].append(float(np.corrcoef(vm, vo)[0, 1]))

                sf = simulate_path_events(evs, max_open_cap=cap, sizing_family="flat", initial_cash=INITIAL_CASH, path_years=p_yrs)
                sk = simulate_path_events(evs, max_open_cap=cap, sizing_family="kelly", book_kelly_fractions=book_kelly, initial_cash=INITIAL_CASH, path_years=p_yrs)
                flat_res.append(sf)
                qk_res.append(sk)
                tot_tr_f += sf["trades"]
                tot_sk_f += sf["skipped_cap"]
                tot_tr_k += sk["trades"]
                tot_sk_k += sk["skipped_cap"]

            rf = reduce_scenario_paths(path_ids, flat_res)
            rk = reduce_scenario_paths(path_ids, qk_res)
            sims[label] = {
                "flat_1usd": {
                    "mean_trades_per_year": round(tot_tr_f / max(tot_yrs, 1e-9), 1),
                    "mean_skipped_per_year": round(tot_sk_f / max(tot_yrs, 1e-9), 1),
                    "p10_g": round(float(rf["p10_g"]), 4) if rf["p10_g"] is not None else None,
                    "mean_g": round(float(rf["mean_g"]), 4) if rf["mean_g"] is not None else None,
                    "p90_dd": round(float(rf["p90_dd"]), 4) if rf["p90_dd"] is not None else None,
                    "p10_g_over_p90_dd": round(float(rf["p10_g_over_p90_dd"]), 3) if rf["p10_g_over_p90_dd"] is not None else None,
                },
                "kelly_0.25": {
                    "mean_trades_per_year": round(tot_tr_k / max(tot_yrs, 1e-9), 1),
                    "mean_skipped_per_year": round(tot_sk_k / max(tot_yrs, 1e-9), 1),
                    "p10_g": round(float(rk["p10_g"]), 4) if rk["p10_g"] is not None else None,
                    "mean_g": round(float(rk["mean_g"]), 4) if rk["mean_g"] is not None else None,
                    "p90_dd": round(float(rk["p90_dd"]), 4) if rk["p90_dd"] is not None else None,
                    "worst_dd": round(float(rk["worst_path_max_dd"]), 4) if rk["worst_path_max_dd"] is not None else None,
                    "p10_g_over_p90_dd": round(float(rk["p10_g_over_p90_dd"]), 3) if rk["p10_g_over_p90_dd"] is not None else None,
                },
            }
        corrs = {sid: (round(float(np.mean(vals)), 4) if vals else None) for sid, vals in corr_lists.items()}
        return sims, corrs

    sims_full, corrs_full = _run_window(strict_common=False)
    sims_common, corrs_common = _run_window(strict_common=True)
    return {
        "available": True,
        "scope": "retrospective_dev_sensitivity_only_full_dev_p10_kelly_not_causal_certificate",
        "window_disclosure": (
            f"Shared group g2 starts at {dt.datetime.fromtimestamp(SHARED_GROUP_BOUNDS_UTC[2], tz=dt.timezone.utc).isoformat()} "
            f"while m60_roll14d OOS starts at {dt.datetime.fromtimestamp(m60_oos_start_ep, tz=dt.timezone.utc).isoformat()}; "
            "strict_common_window filters all 4 books to ts >= m60_oos_start_ep and scales path_years accordingly."
        ),
        "path_ids": path_ids,
        "quarter_kelly_fractions_full_dev_p10": {
            "ens5_m30": round(book_kelly[0], 5),
            "stack05_m30": round(book_kelly[1], 5),
            "ens5_m15": round(book_kelly[2], 5),
            "roll14d_m60": round(book_kelly[3], 5),
        },
        "strict_common_window_t_ge_m60_start": {
            "mean_daily_pnl_corr_vs_roll14d_m60": corrs_common,
            "simulations": sims_common,
        },
        "full_g2_to_g5_6paths": {
            "mean_daily_pnl_corr_vs_roll14d_m60": corrs_full,
            "simulations": sims_full,
        },
    }


def run_dev_certification(
    *,
    dev_dir: Path = DEFAULT_DEV_DATASET_DIR,
    ticks_root: Path = DEFAULT_TICKS_ROOT,
    repo_root: Path = REPO_ROOT,
    workers: int = 8,
    selection_mode: str = "platt_ev",
) -> dict[str, Any]:
    t0 = time.monotonic()
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    dev_manifest = load_pinned_dev_manifest(dev_dir)
    df_dev = load_pinned_dev_parquet(PAIR, dev_dir=dev_dir)
    fcols = select_core66_columns(df_dev)
    tick_ts, tick_px, tick_files = load_pinned_dev_ticks(
        PAIR,
        dev_dir=dev_dir,
        ticks_root=ticks_root,
        manifest=dev_manifest,
        first_day="2025-08-11",
        last_day="2026-07-31",
    )

    ts_all = df_dev["decision_ts"].to_numpy(dtype=np.int64)
    rv60_all = df_dev["rv_60m"].to_numpy(dtype=np.float64)
    X_all = df_dev[fcols].to_numpy(dtype=np.float64)

    ret_fwd_true, valid_true, entry_true, exit_true = settle_with_intervals(
        ts_all, tick_ts, tick_px, horizon_s=HORIZON_S, delay_s=0, tol_s=TOL_S
    )
    eval_delays: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for d_s in DELAYS_EXTRA_S:
        dr, dv, _, _ = settle_with_intervals(ts_all, tick_ts, tick_px, horizon_s=HORIZON_S, delay_s=d_s, tol_s=TOL_S)
        eval_delays[d_s] = (dr, dv)

    decision_eligible_idx = build_decision_eligible_indices_60m(df_dev, fcols, cutoff_utc=CUTOFF_UTC)
    settled_elig_idx = decision_eligible_idx[valid_true[decision_eligible_idx] & np.isfinite(ret_fwd_true[decision_eligible_idx])]
    elig_ret = ret_fwd_true[settled_elig_idx]
    tie_rate = float(np.mean(elig_ret == 0.0))
    moved_ret = elig_ret[elig_ret != 0.0]
    moved_up_rate = float(np.mean(moved_ret > 0.0))

    # Populate shared read-only globals BEFORE forking workers (zero 1.7GB pickle overhead!)
    _SHARED.clear()
    _SHARED.update({
        "X_all": X_all,
        "ts_all": ts_all,
        "rv60_all": rv60_all,
        "entry_true": entry_true,
        "exit_true": exit_true,
        "ret_fwd_true": ret_fwd_true,
        "valid_true": valid_true,
        "decision_eligible_idx": decision_eligible_idx,
        "eval_delays": eval_delays,
        "tick_ts": tick_ts,
        "tick_px": tick_px,
        "selection_mode": selection_mode,
        "ts_to_row": {int(t): idx for idx, t in enumerate(ts_all)},
    })
    print(f"[m60-cert] DEV data loaded ({len(decision_eligible_idx)} decision-elig, {len(settled_elig_idx)} settled, moved_up={moved_up_rate:.4f}, tie={tie_rate:.5f}); running 4 shifts + 16 nulls on {workers} workers...", flush=True)

    max_w = max(1, min(int(workers), 8))
    shift_results: dict[str, dict[str, Any]] = {}
    null_rows: list[dict[str, Any]] = []
    if max_w > 1:
        import multiprocessing as mp
        ctx = mp.get_context("fork")
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_w, mp_context=ctx) as ex:
            fut_shifts = [ex.submit(_shift_worker, sd) for sd in BOUNDARY_SHIFTS_DAYS]
            fut_nulls = [ex.submit(_null_worker, w) for w in NULL_SHIFTS_WEEKS]
            for f in fut_shifts:
                sd, s_res = f.result()
                shift_results[f"plus_{sd}d"] = s_res
            for f in fut_nulls:
                null_rows.append(f.result())
    else:
        for sd in BOUNDARY_SHIFTS_DAYS:
            _, s_res = _shift_worker(sd)
            shift_results[f"plus_{sd}d"] = s_res
        for w in NULL_SHIFTS_WEEKS:
            null_rows.append(_null_worker(w))
    null_rows.sort(key=lambda r: r["shift_weeks"])

    primary = shift_results["plus_0d"]
    raw_0d = primary.pop("raw_top_cov_comparator")

    readable_nulls = [r for r in null_rows if r["readable"]]
    null_wrs = [float(r["wr"]) for r in readable_nulls if r["wr"] is not None]
    null_exc = [float(r["excess_hr_rv"]) for r in readable_nulls if r["excess_hr_rv"] is not None]
    null_joint_passes = sum(1 for r in readable_nulls if r["joint_pass"])
    real_cmp_wr = float(raw_0d["wr"]) if raw_0d["wr"] is not None else 0.0
    real_cmp_exc = float((raw_0d["placebo"] or {}).get("excess_hr_rv") or -1.0)
    n_null_ge_wr = sum(1 for v in null_wrs if v >= real_cmp_wr)
    n_null_ge_exc = sum(1 for v in null_exc if v >= real_cmp_exc)
    rank_wr_str = f"{len(null_wrs) + 1 - n_null_ge_wr}/{len(null_wrs) + 1}"
    rank_exc_str = f"{len(null_exc) + 1 - n_null_ge_exc}/{len(null_exc) + 1}"

    # 3. Evaluate all 5 DEV research gates
    all_shifts_g1 = all(
        (s["n"] >= 100)
        and (s["wr"] is not None and s["wr"] >= PROGRAM_HURDLE_WR)
        and (s["ev"] is not None and s["ev"] > 0.0)
        and (s["ev_ci95"] is not None and s["ev_ci95"][0] > 0.0)
        for s in shift_results.values()
    )
    rp15 = primary["rolling_15paths"]
    p15_ok = bool(
        rp15["paths_counted"] >= 12
        and rp15["paths_clear_0541"] >= 12
        and (rp15["paths_clear_0541"] / max(1, rp15["paths_counted"])) >= 0.80
        and rp15["p10_wr"] is not None
        and rp15["p10_wr"] >= PROGRAM_HURDLE_WR
    )
    g1_passed = bool(all_shifts_g1 and p15_ok)

    all_shifts_g2 = all(
        ((s["placebo"] or {}).get("excess_hr_rv") or -1.0) >= 0.02
        for s in shift_results.values()
    )
    g2_passed = bool(
        all_shifts_g2
        and rp15["paths_placebo_positive"] >= 12
        and (rp15["mean_excess_hr_rv"] is not None and rp15["mean_excess_hr_rv"] >= 0.02)
    )

    all_shifts_d4 = all(
        ((s["delays"].get("4") or {}).get("wr") or 0.0) >= PROGRAM_HURDLE_WR
        and ((s["delays"].get("4") or {}).get("ev") or -1.0) > 0.0
        for s in shift_results.values()
    )
    fq = (primary["freshness"].get("fresh_quote_le2s") or {})
    fm = (primary["freshness"].get("fresh_midchg_le2s") or {})
    fresh_ok = bool(
        (fq.get("wr") or 0.0) >= PROGRAM_HURDLE_WR
        and (fm.get("wr") or 0.0) >= PROGRAM_HURDLE_WR
        and (fq.get("excess_hr_rv") or -1.0) > 0.0
        and (fm.get("excess_hr_rv") or -1.0) > 0.0
    )
    g3_passed = bool(all_shifts_d4 and fresh_ok)

    g4_passed = bool(
        len(readable_nulls) >= 8
        and n_null_ge_wr == 0
        and n_null_ge_exc == 0
        and null_joint_passes == 0
    )

    g5_passed = bool(
        all((s["ev"] is not None and s["ev"] > 0.0 and s["ev_ci95"] is not None and s["ev_ci95"][0] > 0.0) for s in shift_results.values())
        and rp15["paths_ev_positive"] >= 12
        and (rp15["p10_ev"] is not None and rp15["p10_ev"] > 0.0)
    )

    # Mirage check (tie_rate <= 0.02, tick-settled, zero forward fill) + Owner Decision 1 drift amendment via G2+G4
    mirage_ok = bool(0.0 < tie_rate <= 0.02)
    all_5_passed = bool(g1_passed and g2_passed and g3_passed and g4_passed and g5_passed and mirage_ok)

    # 4. Retrospective DEV 4-book vs live 3-book portfolio sensitivity
    p10_for_port = float(rp15["p10_wr"]) if rp15["p10_wr"] is not None else float(primary["wr"] or 0.541)
    m60_oos_start_ep = int(ts_all[decision_eligible_idx[0]]) + TRAIN_DAYS * 86_400
    port_cmp = _evaluate_4book_incremental_portfolio(
        repo_root=repo_root,
        m60_trade_ts=primary["trade_ts"],
        m60_trade_wins=primary["trade_wins"],
        m60_p10_wr=p10_for_port,
        m60_oos_start_ep=m60_oos_start_ep,
    )

    # Compact per-shift summaries (keep full trade_ts/trade_wins on plus_0d for reproducibility)
    shifts_clean: dict[str, Any] = {}
    for k, v in shift_results.items():
        sv = dict(v)
        if k != "plus_0d":
            sv.pop("trade_ts", None)
            sv.pop("trade_wins", None)
            sv.pop("blocks", None)
        shifts_clean[k] = sv

    return {
        "schema": SCHEMA,
        "evaluation_kind": "dev_rolling_oos_research_candidate_not_strict_cpcv",
        "executable_book_authorized": False,
        "book_id": BOOK_ID,
        "pair": PAIR,
        "horizon_min": HORIZON_MIN,
        "side": "UP",
        "contract_family": "risefall",
        "model_variant": "core66_nys_ens5_roll14d",
        "feature_count": len(fcols),
        "cov": COV,
        "train_window_days": TRAIN_DAYS,
        "refit_cadence_days": REFIT_CADENCE_DAYS,
        "selection_mode": selection_mode,
        "quoted_payout_ratio": QUOTED_PAYOUT_RATIO,
        "min_payout_ratio": MIN_PAYOUT_RATIO,
        "cutoff_utc": CUTOFF_UTC,
        "tick_files_count": len(tick_files),
        "decision_eligible_rows": int(len(decision_eligible_idx)),
        "settled_eligible_rows": int(len(settled_elig_idx)),
        "tie_rate": round(tie_rate, 6),
        "moved_up_rate": round(moved_up_rate, 6),
        "mirage_check_passed": mirage_ok,
        "evaluation_falsifier": (
            "Exploratory DEV research falsifier (NOT a pre-run timestamped lab registration and NOT strict refit-CPCV "
            "under scripts/certificate.py; executable freeze requires explicit Owner Protocol Amendment + atomic executor "
            "vintage revocation + approved archive downloader): "
            "KILL unless decision-time causal 14d trailing-120d rolling refit (cov=0.04, core66_nys_ens5) passes: "
            "G1 (all 4 boundary shifts +0d/+3d/+7d/+10d WR>=0.541 & EV_CI95_lo>0, AND >=12/15 overlapping C(6,2) rolling sub-block pair-paths clear 0.541 with p10_wr>=0.541), "
            "G2 (all 4 shifts NY hr x rv_60m excess >= +0.02 & >=12/15 paths positive), "
            "G3 (all 4 shifts +4s delay WR>=0.541 & EV>0, plus fresh <=2s subgroups >0.541 & excess>0), "
            "G4 (top rank on WR & drift-matched excess vs circular week-shift donor-interval-purged sign-null replay with 0 joint null passes), and "
            "G5 (>=12/15 rolling pair-paths EV>0 & p10_ev>0 at R=1.82)."
        ),
        "G1_rolling_oos": {
            "passed": g1_passed,
            "all_4_shifts_passed": all_shifts_g1,
            "rolling_15paths_passed": p15_ok,
            "primary_n_selected_total": primary["n_selected_total"],
            "primary_n": primary["n"],
            "primary_n_unsettled_missing_tick": primary["n_unsettled_missing_tick"],
            "primary_wr": primary["wr"],
            "primary_ev": primary["ev"],
            "primary_conservative_wr_unsettled_as_loss": primary["conservative_wr_unsettled_as_loss"],
            "primary_conservative_ev_unsettled_as_loss": primary["conservative_ev_unsettled_as_loss"],
            "primary_ev_ci95": primary["ev_ci95"],
            "p10_wr": rp15["p10_wr"],
            "paths_clear_0541": rp15["paths_clear_0541"],
            "paths_counted": rp15["paths_counted"],
            "est_trades_per_year": primary["est_trades_per_year"],
            "min_shift_wr": min(s["wr"] for s in shift_results.values() if s["wr"] is not None),
            "min_shift_ev": min(s["ev"] for s in shift_results.values() if s["ev"] is not None),
        },
        "G2_placebo": {
            "passed": g2_passed,
            "verdict": "SKILL BEYOND DRIFT TIMING" if g2_passed else "FAIL",
            "primary_excess_hr_rv": (primary["placebo"] or {}).get("excess_hr_rv"),
            "min_shift_excess_hr_rv": min((s["placebo"] or {}).get("excess_hr_rv") for s in shift_results.values()),
            "paths_positive": rp15["paths_placebo_positive"],
            "mean_excess_hour_vol": rp15["mean_excess_hr_rv"],
        },
        "G3_entry_delay": {
            "passed": g3_passed,
            "verdict_delay": "OPERATIONALLY ROBUST" if all_shifts_d4 else "FAIL",
            "verdict_staleness": "NOT A STALE-FEED ARTIFACT" if fresh_ok else "FAIL",
            "primary_delays": primary["delays"],
            "min_shift_d4_wr": min((s["delays"].get("4") or {}).get("wr") for s in shift_results.values()),
            "primary_freshness": primary["freshness"],
        },
        "G4_sign_null": {
            "passed": g4_passed,
            "verdict": "SIGNIFICANT vs circular week-shift donor-interval-purged sign-null replay" if g4_passed else "FAIL",
            "null_protocol": "circular_week_shift_donor_interval_purged_with_receiver_time_causal_blocks",
            "selection_mode": "raw_top_cov_apples_to_apples",
            "real_raw_top4_n": raw_0d["n"],
            "real_raw_top4_wr": raw_0d["wr"],
            "real_raw_top4_ev": raw_0d["ev"],
            "real_raw_top4_ev_ci95": raw_0d["ev_ci95"],
            "real_raw_top4_excess_hr_rv": (raw_0d["placebo"] or {}).get("excess_hr_rv"),
            "readable_nulls": len(readable_nulls),
            "rank_wr": rank_wr_str,
            "rank_excess_hr_rv": rank_exc_str,
            "null_max_wr": round(max(null_wrs), 6) if null_wrs else None,
            "null_max_excess_hr_rv": round(max(null_exc), 6) if null_exc else None,
            "null_joint_passes": null_joint_passes,
            "nulls": null_rows,
        },
        "G5_payout": {
            "passed": g5_passed,
            "verdict": "PAYOUT-CLEARED" if g5_passed else "FAIL",
            "quoted_payout_ratio": QUOTED_PAYOUT_RATIO,
            "min_payout_ratio": MIN_PAYOUT_RATIO,
            "paths_ev_positive": rp15["paths_ev_positive"],
            "paths_counted": rp15["paths_counted"],
            "p10_ev": rp15["p10_ev"],
            "mean_ev": primary["ev"],
            "portfolio_4book_comparison": port_cmp,
        },
        "all_5_gates_passed": all_5_passed,
        "boundary_shifts": shifts_clean,
        "elapsed_s": round(time.monotonic() - t0, 2),
    }


def _validate_safe_result_out(raw_out: str, repo_root: Path = REPO_ROOT) -> tuple[Path, Path, str]:
    """Validate --out is a *_result.json file directly inside non-symlinked repo_root/results/json."""
    raw_p = Path(raw_out)
    if ".." in raw_p.parts:
        raise SystemExit(f"refusing unsafe --out path: {raw_out!r}")
    out_p = raw_p if raw_p.is_absolute() else (repo_root / raw_p)
    try:
        rel = out_p.relative_to(repo_root)
    except ValueError:
        raise SystemExit(f"--out must be inside {repo_root / 'results' / 'json'} (refusing {raw_out!r})")
    if len(rel.parts) != 3 or rel.parts[0] != "results" or rel.parts[1] != "json":
        raise SystemExit(
            f"--out must be a *_result.json file directly inside {repo_root / 'results' / 'json'} (refusing {raw_out!r})"
        )
    if not raw_p.name.endswith("_result.json") or raw_p.name != rel.parts[2]:
        raise SystemExit(f"--out filename must end with '_result.json' (refusing {raw_out!r})")
    cur = repo_root
    for part in rel.parts:
        cur = cur / part
        if cur.is_symlink():
            raise SystemExit(f"refusing symlink component in --out path: {cur}")
    results_dir = repo_root / "results"
    json_dir = results_dir / "json"
    repo_resolved = repo_root.resolve()
    if results_dir.exists() and results_dir.resolve() != (repo_resolved / "results"):
        raise SystemExit(f"refusing non-canonical results directory: {results_dir}")
    if json_dir.exists() and json_dir.resolve() != (repo_resolved / "results" / "json"):
        raise SystemExit(f"refusing non-canonical results/json directory: {json_dir}")
    return results_dir, json_dir, rel.parts[2]


def _write_result_json_nofollow(raw_out: str, payload: str, repo_root: Path = REPO_ROOT) -> Path:
    """Validate --out and write strictly through O_DIRECTORY | O_NOFOLLOW descriptors."""
    results_dir, json_dir, filename = _validate_safe_result_out(raw_out, repo_root=repo_root)
    if not results_dir.exists():
        os.mkdir(results_dir)
    res_fd = os.open(str(results_dir), os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        try:
            json_fd = os.open("json", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=res_fd)
        except FileNotFoundError:
            os.mkdir("json", dir_fd=res_fd)
            json_fd = os.open("json", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=res_fd)
        try:
            fd = os.open(filename, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o644, dir_fd=json_fd)
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(payload)
                fh.flush()
                os.fsync(fh.fileno())
            os.fsync(json_fd)
        finally:
            os.close(json_fd)
    finally:
        os.close(res_fd)
    return json_dir / filename


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Strictly DEV (<2026-08-01) 5-gate causal rolling-refit evaluator for [USDJPY·60m·UP]")
    ap.add_argument("--certify-dev", action="store_true", help="Run full <2026-08-01 DEV 5-gate causal rolling evaluation")
    ap.add_argument("--selection-mode", choices=("platt_ev", "raw_top_cov"), default="platt_ev")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="results/json/deriv_5gate_USDJPY_sota_roll14d_m60_up_v1_result.json")
    args = ap.parse_args(argv)

    if not args.certify_dev:
        ap.print_help()
        return 0

    _validate_safe_result_out(args.out, repo_root=REPO_ROOT)
    res = run_dev_certification(workers=args.workers, selection_mode=args.selection_mode)
    out_p = _write_result_json_nofollow(
        args.out,
        json.dumps(res, indent=2, sort_keys=True, allow_nan=False) + "\n",
        repo_root=REPO_ROOT,
    )
    summary = {
        "out": str(out_p),
        "evaluation_kind": res["evaluation_kind"],
        "executable_book_authorized": res["executable_book_authorized"],
        "all_5_gates_passed": res["all_5_gates_passed"],
        "G1": {k: res["G1_rolling_oos"][k] for k in ("passed", "primary_n_selected_total", "primary_n", "primary_n_unsettled_missing_tick", "primary_wr", "primary_ev", "primary_ev_ci95", "p10_wr", "paths_clear_0541", "paths_counted")},
        "G2": {k: res["G2_placebo"][k] for k in ("passed", "primary_excess_hr_rv", "min_shift_excess_hr_rv", "paths_positive", "mean_excess_hour_vol")},
        "G3": {k: res["G3_entry_delay"][k] for k in ("passed", "min_shift_d4_wr", "verdict_delay", "verdict_staleness")},
        "G4": {k: res["G4_sign_null"][k] for k in ("passed", "readable_nulls", "rank_wr", "rank_excess_hr_rv", "null_max_wr", "null_max_excess_hr_rv", "null_joint_passes")},
        "G5": {k: res["G5_payout"][k] for k in ("passed", "paths_ev_positive", "p10_ev", "mean_ev")},
        "elapsed_s": res["elapsed_s"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
