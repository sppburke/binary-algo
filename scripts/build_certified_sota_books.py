"""Build and cryptographically freeze all G1-G5 (+ G6 where applicable) certified SOTA books.

Strictly enforces:
1. Pre-holdout cutoff (`decision_ts < 2026-08-01T00:00:00Z`) on USDJPY so Look 2 (`to 2027-02-26`)
   remains 100% untouched per User Decision A2.
2. Fail-closed verification of all 5 mandatory on-disk certification JSON dossiers (`G1_cpcv`,
   `G2_placebo`, `G3_entry_delay`, `G4_sign_null`, `G5_payout`) before writing any artifact.
3. Chronological block-purged validation split (`gap = H`) for out-of-sample confidence threshold
   (`cov = 0.03`) and out-of-sample Platt ties-lose probability calibration (`logit(p) -> P(win)`).
4. Pure-NumPy reload parity check (`max_abs_diff < 1e-11` vs. fitted `sklearn` pipeline) across
   1,000 real decision rows via `sota_book_freeze.py`.
"""

from __future__ import annotations

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# BLOCKER #8 fix: verify sibling /home/sean/git/binary-algo/features exists BEFORE any import that can reach pipeline.py!
_PIPELINE_FEATURES_DIR = Path("/home/sean/git/binary-algo/features")
if not _PIPELINE_FEATURES_DIR.is_dir():
    raise RuntimeError(
        f"refusing import: required existing directory missing ({_PIPELINE_FEATURES_DIR}); "
        f"pipeline.py must not create it"
    )

import numpy as np
import pandas as pd
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import LogisticRegression, SGDClassifier
from zoneinfo import ZoneInfo

from sota_book_freeze import (
    extract_nystroem_arrays,
    extract_nystroem_ens_arrays,
    freeze_sota_book,
)
from sota_book_runtime import DEMO_CANARY_LIFECYCLE, REPO_ROOT, SotaBookError

CUTOFF_UTC = int(datetime(2026, 8, 1, 0, 0, 0, tzinfo=timezone.utc).timestamp())
NY_TZ = ZoneInfo("America/New_York")
DEFAULT_DEV_DATASET = Path("/mnt/data/deriv-sota-lab/cache/dataset_dev_v0.1")


def _session_and_tradeable_mask(ts: np.ndarray, horizon_minutes: int, drop_ny_16h: bool) -> np.ndarray:
    dt_utc = pd.to_datetime(ts, unit="s", utc=True)
    dt_ny = dt_utc.tz_convert(NY_TZ)
    wd = dt_ny.dayofweek.to_numpy()
    ny_mod = dt_ny.hour.to_numpy() * 60 + dt_ny.minute.to_numpy()
    in_ny = (wd < 5) & (ny_mod >= 480) & (ny_mod < 1020)
    # Venue sub-5h blackout 21:00-23:59 UTC + NY 16:50-H rollover blackout
    sub5h_ok = dt_utc.hour.to_numpy() < 21
    rollover_cut = 16 * 60 + 50 - int(horizon_minutes)
    rollover_ok = ~((ny_mod >= rollover_cut) & (ny_mod < 18 * 60))
    elig = in_ny & sub5h_ok & rollover_ok
    if drop_ny_16h:
        elig &= ~((ny_mod >= 960) & (ny_mod < 995))
    return elig


def _fit_deploy_nystroem_and_platt(
    df: pd.DataFrame,
    *,
    horizon_minutes: int,
    weight_absret: bool,
    drop_ny_16h: bool,
    coverage: float = 0.03,
    seed: int = 0,
) -> tuple[list[str], dict[str, np.ndarray], float, dict[str, dict[str, float]], pd.DataFrame, np.ndarray]:
    ts = df["decision_ts"].to_numpy(dtype=np.int64)
    if ts.max() >= CUTOFF_UTC:
        raise SotaBookError(f"dataset contains post-2026-07-31 rows ({ts.max()} >= {CUTOFF_UTC})")

    fcols = [
        c
        for c in df.columns
        if c not in ("bar_start_ts", "decision_ts") and not c.startswith(("ret_fwd_", "valid_"))
    ]
    fwd = df[f"ret_fwd_{horizon_minutes}m"].to_numpy(dtype=np.float64)
    valid = df[f"valid_{horizon_minutes}m"].to_numpy(dtype=bool)
    X_all = df[fcols].to_numpy(dtype=np.float64)

    # Drop rows with >= 50% missing features (exact cpcv_deriv09 rule)
    row_ok = np.mean(~np.isfinite(X_all), axis=1) < 0.50
    elig = row_ok & valid & _session_and_tradeable_mask(ts, horizon_minutes, drop_ny_16h)
    moved = elig & np.isfinite(fwd) & (fwd != 0.0)

    idx_moved = np.flatnonzero(moved)
    gap = int(horizon_minutes) * 60
    # Purged chronological last 15% validation split for OOS threshold & Platt side calibration
    n_val = max(500, int(round(len(idx_moved) * 0.15)))
    val_idx = idx_moved[-n_val:]
    val_start_ts = int(ts[val_idx[0]])
    fit_idx = idx_moved[ts[idx_moved] < (val_start_ts - gap)]

    Xf = X_all[fit_idx]
    ff = fwd[fit_idx]
    yf = (ff > 0).astype(int)
    med = np.nanmedian(np.where(np.isfinite(Xf), Xf, np.nan), axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    Zf = np.where(np.isfinite(Xf), Xf, med).astype(np.float64)
    mu = Zf.mean(axis=0)
    sd = Zf.std(axis=0) + 1e-12
    Zf = np.clip((Zf - mu) / sd, -5.0, 5.0)

    ny = Nystroem(kernel="rbf", gamma=1.0 / Zf.shape[1], n_components=500, random_state=seed)
    Fk = ny.fit_transform(Zf)
    sw = (np.abs(ff) / (np.abs(ff).mean() + 1e-18)) if weight_absret else None
    sgd = SGDClassifier(loss="log_loss", alpha=1e-4, max_iter=30, tol=1e-4, random_state=seed)
    sgd.fit(Fk, yf, sample_weight=sw)

    # Score chronological validation block (strictly purged from fit_idx)
    # Include ties in validation target so Platt calibration honors ties-lose settlement!
    elig_val_idx = np.flatnonzero(elig & (ts >= val_start_ts) & np.isfinite(fwd))
    Xv = X_all[elig_val_idx]
    fv = fwd[elig_val_idx]
    Zv = np.clip((np.where(np.isfinite(Xv), Xv, med).astype(np.float64) - mu) / sd, -5.0, 5.0)
    pval = sgd.predict_proba(ny.transform(Zv))[:, 1]
    conf_thr = float(np.quantile(np.abs(pval - 0.5), 1.0 - coverage))

    logits_v = np.clip(
        np.log(np.clip(pval, 1e-6, 1.0 - 1e-6) / np.clip(1.0 - pval, 1e-6, 1.0 - 1e-6)),
        -15.0,
        15.0,
    ).reshape(-1, 1)
    cal_up = LogisticRegression(C=1.0, max_iter=1000).fit(logits_v, (fv > 0).astype(int))
    platt = {
        "UP": {
            "coef": float(cal_up.coef_[0, 0]),
            "intercept": float(cal_up.intercept_[0]),
            "n_val_oos": int(len(elig_val_idx)),
        }
    }

    # Prepare 1,000 finite rows for pure-NumPy vs sklearn parity check
    fin_rows = np.flatnonzero(np.all(np.isfinite(X_all), axis=1))[:1000]
    parity_df = pd.DataFrame(X_all[fin_rows], columns=fcols)
    Zp = np.clip((X_all[fin_rows] - mu) / sd, -5.0, 5.0)
    parity_probs = sgd.predict_proba(ny.transform(Zp))[:, 1]

    arrays = extract_nystroem_arrays(med, mu, sd, ny, sgd)
    return fcols, arrays, conf_thr, platt, parity_df, parity_probs


def build_usdjpy_15m_c2nysabs(repo_root: Path = REPO_ROOT, dev_dir: Path = DEFAULT_DEV_DATASET) -> Path:
    rjson = repo_root / "results" / "json"
    g1 = json.loads((rjson / "deriv_c2nysabs_USDJPY_15m_result.json").read_text())
    g2 = json.loads((rjson / "deriv_a6placebo_USDJPY_15m_result.json").read_text())
    g3 = json.loads((rjson / "deriv_a7delay_USDJPY_15m_result.json").read_text())
    g4 = json.loads((rjson / "deriv_a8signnull_c2nysabs_USDJPY_15m_result.json").read_text())
    g5_raw = json.loads((rjson / "deriv_a24pay_USDJPY_15m_result.json").read_text())
    g5_up = g5_raw["sides"]["UP"]

    u1 = g1["verdict_goal"]["UP"]
    u1_mean = float(np.mean([p["bycov"]["0.03"]["UP"][1] for p in g1["paths"] if p["bycov"]["0.03"]["UP"][0] >= 25]))
    if not (u1["paths_clear"] >= 12 and u1["p10"] >= 0.541):
        raise SotaBookError("USDJPY 15m c2nysabs failed G1")
    if g2["sides"]["UP"]["verdict"] != "SKILL BEYOND DRIFT TIMING":
        raise SotaBookError("USDJPY 15m c2nysabs failed G2")
    if g3["sides"]["UP"]["verdict_delay"] != "OPERATIONALLY ROBUST":
        raise SotaBookError("USDJPY 15m c2nysabs failed G3")
    if g4.get("verdict") != "SIGNIFICANT vs sign-null replay":
        raise SotaBookError(f"USDJPY 15m c2nysabs failed G4: {g4.get('verdict')}")
    if g5_up["verdict"] != "PAYOUT-CLEARED":
        raise SotaBookError("USDJPY 15m c2nysabs failed G5")

    df = pd.read_parquet(dev_dir / "USDJPY.parquet")
    fcols, arrays, conf_thr, platt, p_df, p_probs = _fit_deploy_nystroem_and_platt(
        df,
        horizon_minutes=15,
        weight_absret=True,
        drop_ny_16h=False,
        coverage=0.03,
        seed=0,
    )

    certs: dict[str, dict[str, Any]] = {
        "G1_cpcv": {
            "passed": True,
            "evidence_json": "results/json/deriv_c2nysabs_USDJPY_15m_result.json",
            "paths_clear": int(u1["paths_clear"]),
            "paths_counted": int(u1["paths_counted"]),
            "p10_wr": round(float(u1["p10"]), 6),
            "mean_wr": round(u1_mean, 6),
            "validity_note": "Owner Decision 1 drift amendment satisfied via G2_placebo",
        },
        "G2_placebo": {
            "passed": True,
            "evidence_json": "results/json/deriv_a6placebo_USDJPY_15m_result.json",
            "verdict": g2["sides"]["UP"]["verdict"],
            "mean_excess_hour_vol": round(float(g2["sides"]["UP"]["summary"]["excess_hour_vol"]["mean"]), 6),
            "paths_positive": int(g2["sides"]["UP"]["summary"]["excess_hour_vol"]["n_positive"]),
        },
        "G3_entry_delay": {
            "passed": True,
            "evidence_json": "results/json/deriv_a7delay_USDJPY_15m_result.json",
            "verdict_delay": g3["sides"]["UP"]["verdict_delay"],
            "verdict_staleness": g3["sides"]["UP"]["verdict_staleness"],
            "d4_p10_wr": round(float(g3["sides"]["UP"]["summary_by_delay"]["4"]["p10"]), 6),
            "d4_paths_clear": int(g3["sides"]["UP"]["summary_by_delay"]["4"]["paths_clear"]),
        },
        "G4_sign_null": {
            "passed": True,
            "evidence_json": "results/json/deriv_a8signnull_c2nysabs_USDJPY_15m_result.json",
            "verdict": g4["verdict"],
            "real_p10_rank_of": g4["real_p10_rank_of"],
            "null_max_p10": round(float(g4["null_p10_quantiles"]["100"]), 6),
            "nulls_meeting_goal_path_test": int(g4["nulls_meeting_goal_path_test"]),
        },
        "G5_payout": {
            "passed": True,
            "evidence_json": "results/json/deriv_a24pay_USDJPY_15m_result.json",
            "verdict": g5_up["verdict"],
            "pricing": g5_up["pricing"],
            "paths_ev_positive": int(g5_up["summary"]["paths_ev_positive"]),
            "paths_counted": int(g5_up["summary"]["paths_counted"]),
            "p10_ev": round(float(g5_up["summary"]["p10_ev"]), 6),
            "mean_ev": round(float(g5_up["summary"]["mean_ev"]), 6),
            "actual_quotes_ev": round(float(g5_up["actual_quotes"]["ev_actual"]), 6),
            "mean_payout_ratio": round(float(g5_up["summary"]["mean_payout"]), 6),
        },
        "G6_holdout_or_prospective": {
            "passed": False,
            "mode": "prospective_demo_venue_settled",
            "reason": "Sealed USDJPY Look 2 holdout preserved untouched per User Decision A2; promote after >=50 Demo VENUE_SETTLED trades with EV CI95_lo > 0 via scripts/deriv_live_pnl_ledger.py.",
        },
    }

    return freeze_sota_book(
        book_id="USDJPY.sota_c2nysabs_m15_up.v1",
        pair="USDJPY",
        horizon_minutes=15,
        allowed_sides=["UP"],
        lifecycle_status=DEMO_CANARY_LIFECYCLE,
        model_kind="nystroem",
        feature_cols=fcols,
        coverage=0.03,
        confidence_threshold=conf_thr,
        refit_cpcv_p10_floors={"UP": float(u1["p10"])},
        platt_calibration=platt,
        gate_certificates=certs,
        summary=(
            "[USDJPY·15m·UP] SOTA RBF-Nystroem (500 components) + |return|-weighted logistic classifier "
            "on 66 true-Deriv-tick features. Passed G1 (15/15, p10=0.5760, mean=0.6000, 3/3 fwd=0.5785), "
            "G2 (+0.063 hour x vol placebo), G3 (15/15 at +4s delay, p10=0.575), G4 (16x week-shift sign-null), "
            "and G5 (PAYOUT-CLEARED, 14/15 EV>0, p10_ev=+0.0143, mean_ev=+0.0621, actual_quotes_ev=+0.2629)."
        ),
        metrics={
            "cpcv_paths_clear": f"{u1['paths_clear']}/{u1['paths_counted']}",
            "cpcv_p10_wr": round(float(u1["p10"]), 4),
            "cpcv_mean_wr": round(u1_mean, 4),
            "cpcv_p10_ev": round(float(g5_up["summary"]["p10_ev"]), 4),
            "cpcv_mean_ev": round(float(g5_up["summary"]["mean_ev"]), 4),
            "actual_quotes_ev": round(float(g5_up["actual_quotes"]["ev_actual"]), 4),
        },
        source_script="scripts/build_certified_sota_books.py",
        weights_arrays=arrays,
        contract_family="risefall",
        drop_ny_16h=False,
        payout_edge_margin=0.005,
        min_payout_ratio=1.76,
        parity_check_df=p_df,
        parity_expected_probs=p_probs,
        repo_root=repo_root,
    )


def build_usdjpy_30m_e49nys30(repo_root: Path = REPO_ROOT, dev_dir: Path = DEFAULT_DEV_DATASET) -> Path:
    rjson = repo_root / "results" / "json"
    g1 = json.loads((rjson / "deriv_e49nys30_USDJPY_30m_result.json").read_text())
    g2 = json.loads((rjson / "deriv_a8placebo_USDJPY_30m_result.json").read_text())
    g3 = json.loads((rjson / "deriv_a9delay_USDJPY_30m_result.json").read_text())
    g45 = json.loads((rjson / "deriv_a10signnull_e49nys30_USDJPY_30m_result.json").read_text())
    g5_up = g45["G5_payout_30m"]

    u1 = g1["verdict_goal"]["UP"]
    u1_mean = float(np.mean([p["bycov"]["0.03"]["UP"][1] for p in g1["paths"] if p["bycov"]["0.03"]["UP"][0] >= 25]))
    if not (u1["paths_clear"] >= 12 and u1["p10"] >= 0.541):
        raise SotaBookError("USDJPY 30m e49nys30 failed G1")
    if g2["sides"]["UP"]["verdict"] != "SKILL BEYOND DRIFT TIMING":
        raise SotaBookError("USDJPY 30m e49nys30 failed G2")
    if g3["sides"]["UP"]["verdict_delay"] != "OPERATIONALLY ROBUST":
        raise SotaBookError("USDJPY 30m e49nys30 failed G3")
    if g45.get("verdict") != "SIGNIFICANT vs sign-null replay":
        raise SotaBookError(f"USDJPY 30m e49nys30 failed G4: {g45.get('verdict')}")
    if g5_up["verdict"] != "PAYOUT-CLEARED":
        raise SotaBookError("USDJPY 30m e49nys30 failed G5")

    df = pd.read_parquet(dev_dir / "USDJPY.parquet")
    fcols, arrays, conf_thr, platt, p_df, p_probs = _fit_deploy_nystroem_and_platt(
        df,
        horizon_minutes=30,
        weight_absret=False,
        drop_ny_16h=False,
        coverage=0.03,
        seed=0,
    )

    certs: dict[str, dict[str, Any]] = {
        "G1_cpcv": {
            "passed": True,
            "evidence_json": "results/json/deriv_e49nys30_USDJPY_30m_result.json",
            "paths_clear": int(u1["paths_clear"]),
            "paths_counted": int(u1["paths_counted"]),
            "p10_wr": round(float(u1["p10"]), 6),
            "mean_wr": round(u1_mean, 6),
            "validity_note": "Owner Decision 1 drift amendment satisfied via G2_placebo",
        },
        "G2_placebo": {
            "passed": True,
            "evidence_json": "results/json/deriv_a8placebo_USDJPY_30m_result.json",
            "verdict": g2["sides"]["UP"]["verdict"],
            "mean_excess_hour_vol": round(float(g2["sides"]["UP"]["summary"]["excess_hour_vol"]["mean"]), 6),
            "paths_positive": int(g2["sides"]["UP"]["summary"]["excess_hour_vol"]["n_positive"]),
        },
        "G3_entry_delay": {
            "passed": True,
            "evidence_json": "results/json/deriv_a9delay_USDJPY_30m_result.json",
            "verdict_delay": g3["sides"]["UP"]["verdict_delay"],
            "verdict_staleness": g3["sides"]["UP"]["verdict_staleness"],
            "d4_p10_wr": round(float(g3["sides"]["UP"]["summary_by_delay"]["4"]["p10"]), 6),
            "d4_paths_clear": int(g3["sides"]["UP"]["summary_by_delay"]["4"]["paths_clear"]),
        },
        "G4_sign_null": {
            "passed": True,
            "evidence_json": "results/json/deriv_a10signnull_e49nys30_USDJPY_30m_result.json",
            "verdict": g45["verdict"],
            "real_p10_rank_of": g45["real_p10_rank_of"],
            "null_max_p10": round(float(g45["null_p10_quantiles"]["100"]), 6),
            "nulls_meeting_goal_path_test": int(g45["nulls_meeting_goal_path_test"]),
        },
        "G5_payout": {
            "passed": True,
            "evidence_json": "results/json/deriv_a10signnull_e49nys30_USDJPY_30m_result.json",
            "survey_evidence_json": "results/json/deriv_winter_quote_survey_v10_v12_result.json",
            "verdict": g5_up["verdict"],
            "paths_ev_positive": int(g5_up["paths_ev_positive"]),
            "paths_counted": int(g5_up["paths_counted"]),
            "p10_ev": round(float(g5_up["p10_ev"]), 6),
            "mean_ev": round(float(g5_up["mean_ev"]), 6),
            "fwd_mean_ev": round(float(g5_up["fwd_mean_ev"]), 6),
            "mean_payout_ratio": 1.82,
        },
        "G6_holdout_or_prospective": {
            "passed": False,
            "mode": "prospective_demo_venue_settled",
            "reason": "Sealed USDJPY Look 2 holdout preserved untouched per User Decision A2; promote after >=50 Demo VENUE_SETTLED trades with EV CI95_lo > 0 via scripts/deriv_live_pnl_ledger.py.",
        },
    }

    return freeze_sota_book(
        book_id="USDJPY.sota_e49nys30_m30_up.v1",
        pair="USDJPY",
        horizon_minutes=30,
        allowed_sides=["UP"],
        lifecycle_status=DEMO_CANARY_LIFECYCLE,
        model_kind="nystroem",
        feature_cols=fcols,
        coverage=0.03,
        confidence_threshold=conf_thr,
        refit_cpcv_p10_floors={"UP": float(u1["p10"])},
        platt_calibration=platt,
        gate_certificates=certs,
        summary=(
            "[USDJPY·30m·UP] SOTA RBF-Nystroem (500 components) logistic classifier on 66 true-Deriv-tick "
            "features at 30-minute horizon (flat 1.82 Deriv NY quote, 0.0000 directional skew per V-11 survey). "
            "Passed G1 (15/15, p10=0.5984, mean=0.6342), G2 (+0.0753 hour x vol placebo on 15/15 paths), "
            "G3 (15/15 at +4s delay, p10=0.5939), G4 (16x week-shift sign-null), and G5 (PAYOUT-CLEARED, "
            "15/15 EV>0, p10_ev=+0.0891, mean_ev=+0.1543, fwd_mean_ev=+0.0619)."
        ),
        metrics={
            "cpcv_paths_clear": f"{u1['paths_clear']}/{u1['paths_counted']}",
            "cpcv_p10_wr": round(float(u1["p10"]), 4),
            "cpcv_mean_wr": round(u1_mean, 4),
            "cpcv_p10_ev": round(float(g5_up["p10_ev"]), 4),
            "cpcv_mean_ev": round(float(g5_up["mean_ev"]), 4),
            "fwd_mean_ev": round(float(g5_up["fwd_mean_ev"]), 4),
        },
        source_script="scripts/build_certified_sota_books.py",
        weights_arrays=arrays,
        contract_family="risefall",
        drop_ny_16h=False,
        payout_edge_margin=0.005,
        min_payout_ratio=1.78,
        parity_check_df=p_df,
        parity_expected_probs=p_probs,
        repo_root=repo_root,
    )


def _strict_int_field(obj: dict[str, Any], key: str, book_id: str) -> int:
    val = obj.get(key)
    if isinstance(val, bool) or not isinstance(val, int):
        raise SotaBookError(f"{book_id}: field {key!r} must be an exact int, got {val!r}")
    return val


def _strict_finite_float(obj: dict[str, Any], key: str, book_id: str) -> float:
    val = obj.get(key)
    if isinstance(val, bool) or not isinstance(val, (int, float)) or not np.isfinite(float(val)):
        raise SotaBookError(f"{book_id}: field {key!r} must be a finite float, got {val!r}")
    return float(val)


def freeze_from_v2_dossier(
    *,
    book_id: str,
    dossier_json_rel: str,
    df_dev: pd.DataFrame,
    ret_fwd_true: np.ndarray,
    valid_true: np.ndarray,
    min_payout_ratio: float,
    allow_overwrite: bool = False,
    repo_root: Path = REPO_ROOT,
) -> Path:
    """Cryptographically freeze a passed `sota-5gate-certification-dossier/v2` Rise/Fall book (`nystroem` or `nystroem-ens`)."""
    dos_path = repo_root / dossier_json_rel
    dos = json.loads(dos_path.read_text(encoding="utf-8"))
    if dos.get("schema") != "sota-5gate-certification-dossier/v2":
        raise SotaBookError(f"{book_id}: invalid dossier schema {dos.get('schema')!r}")
    dos_id = dos.get("book_id")
    if not isinstance(dos_id, str) or dos_id != book_id:
        raise SotaBookError(f"{book_id}: dossier book_id mismatch ({dos_id!r} != {book_id!r})")
    if dos.get("contract_family") != "risefall":
        raise SotaBookError(f"{book_id}: expected risefall dossier, got {dos.get('contract_family')!r}")
    if dos.get("shared_calendar_aligned") is not True:
        raise SotaBookError(f"{book_id}: dossier must be aligned on full DEV shared_group_bounds_utc")
    if _strict_int_field(dos, "cutoff_utc", book_id) != CUTOFF_UTC:
        raise SotaBookError(f"{book_id}: dossier cutoff_utc {dos.get('cutoff_utc')!r} != {CUTOFF_UTC}")
    if dos.get("all_5_gates_passed") is not True or dos.get("drift_check_passed") is not True:
        raise SotaBookError(f"{book_id}: dossier {dossier_json_rel} did not pass all 5 gates and drift check")

    for req_gate in ("G1_cpcv", "G2_placebo", "G3_entry_delay", "G4_sign_null", "G5_payout"):
        g_obj = dos.get(req_gate)
        if not isinstance(g_obj, dict) or g_obj.get("passed") is not True:
            raise SotaBookError(f"{book_id}: individual dossier gate {req_gate} is missing or False")

    g1_chk, g2_chk, g3_chk, g4_chk, g5_chk = (
        dos["G1_cpcv"],
        dos["G2_placebo"],
        dos["G3_entry_delay"],
        dos["G4_sign_null"],
        dos["G5_payout"],
    )
    p_cnt = _strict_int_field(g1_chk, "paths_counted", book_id)
    p_clr = _strict_int_field(g1_chk, "paths_clear_0541", book_id)
    p10_wr_v = _strict_finite_float(g1_chk, "p10_wr", book_id)
    _strict_finite_float(g1_chk, "mean_wr", book_id)
    if p_cnt < 12 or p_clr < 12 or (p_clr / max(1, p_cnt)) < 0.80 or p10_wr_v < 0.541:
        raise SotaBookError(f"{book_id}: G1_cpcv quantitative criteria not met")
    if _strict_int_field(g2_chk, "paths_positive", book_id) < 12 or _strict_finite_float(g2_chk, "mean_excess_hour_vol", book_id) < 0.02:
        raise SotaBookError(f"{book_id}: G2_placebo quantitative criteria not met")
    if _strict_int_field(g3_chk, "d4_paths_clear", book_id) < 12 or _strict_finite_float(g3_chk, "d4_p10_wr", book_id) < 0.541:
        raise SotaBookError(f"{book_id}: G3_entry_delay quantitative criteria not met")
    _strict_finite_float(g4_chk, "null_max_p10", book_id)
    if _strict_int_field(g4_chk, "null_passes", book_id) > 1 or not str(g4_chk.get("verdict") or "").startswith("SIGNIFICANT"):
        raise SotaBookError(f"{book_id}: G4_sign_null quantitative criteria not met")
    if _strict_int_field(g5_chk, "paths_ev_positive", book_id) < 12 or _strict_finite_float(g5_chk, "p10_ev", book_id) <= 0.0:
        raise SotaBookError(f"{book_id}: G5_payout quantitative criteria not met")

    pair = str(dos["pair"]).upper()
    hm = _strict_int_field(dos, "horizon_min", book_id)
    side = str(dos["side"]).upper()
    cov = _strict_finite_float(dos, "cov", book_id)
    variant = str(dos["model_variant"])
    allowed_variants = {"core66_nys_1seed", "core66_nys_absret_1seed", "core66_nys_ens5", "core66_nys_ens5_absret"}
    if variant not in allowed_variants:
        raise SotaBookError(f"{book_id}: unsupported pure-NumPy freeze variant {variant!r}")

    moved_up = _strict_finite_float(dos, "moved_up_rate", book_id)
    tie_r = _strict_finite_float(dos, "tie_rate", book_id)
    if not (0.0 < moved_up < 1.0 and 0.0 < tie_r < 1.0):
        raise SotaBookError(f"{book_id}: invalid moved_up_rate={moved_up} or tie_rate={tie_r}")
    in_strict_band = bool(0.47 <= moved_up <= 0.53 and 0.0 < tie_r <= 0.03)
    oos_obj = dos.get("frozen_2025_to_2026_oos")
    if not isinstance(oos_obj, dict):
        raise SotaBookError(f"{book_id}: missing frozen_2025_to_2026_oos")
    oos26_ev = _strict_finite_float(oos_obj, "ev", book_id)
    if oos26_ev <= 0.0:
        raise SotaBookError(f"{book_id}: frozen 2025->2026 OOS EV must be positive ({oos26_ev})")
    # Enforce strict [0.47, 0.53] moved band OR the program's USDJPY Owner Decision 1 drift amendment
    # (docs/DERIV_SOTA_PROGRAM.md §"Owner decisions (resolved)" #1: bounded [0.45, 0.55] drift + mirage tie_rate <= 0.02 + G2 + G4).
    owner_dec1_ok = bool(pair == "USDJPY" and hm in {15, 30} and 0.45 <= moved_up <= 0.55 and 0.0 < tie_r <= 0.02 and oos26_ev > 0.0)
    if not (in_strict_band or owner_dec1_ok):
        raise SotaBookError(
            f"{book_id}: moved_up_rate={moved_up} outside [0.47, 0.53] without qualifying Owner Decision 1 drift amendment"
        )
    validity_note = (
        "Strict [0.47, 0.53] moved up-rate band satisfied"
        if in_strict_band
        else "Owner Decision 1 drift amendment satisfied via G2_placebo + G4_sign_null + frozen 2025->2026 OOS"
    )
    weight_absret = "absret" in variant
    n_seeds = 5 if "ens5" in variant else 1
    model_kind = "nystroem-ens" if n_seeds > 1 else "nystroem"

    macro_cols = {"hrs_to_high_macro", "high_macro_next_24h", "hrs_since_macro", "last_macro_abs_z"}
    extra_vol = ("rv_up_", "rv_dn_", "rv_asym_", "jump_share_", "parkinson_", "park_comp_", "rv_4320m", "rv_7200m")
    use_core66 = variant.startswith("core66_")
    fcols = [
        c for c in df_dev.columns
        if c not in ("bar_start_ts", "decision_ts")
        and c not in macro_cols
        and not c.startswith(("ret_fwd_", "max_ret_", "min_ret_", "valid_"))
        and not (use_core66 and c.startswith(extra_vol))
    ]
    if _strict_int_field(dos, "feature_count", book_id) != len(fcols):
        raise SotaBookError(f"{book_id}: feature_count mismatch ({len(fcols)} != dossier {dos.get('feature_count')})")

    ts = df_dev["decision_ts"].to_numpy(dtype=np.int64)
    if ts.max() >= CUTOFF_UTC or np.any(valid_true & ((ts + 1 + hm * 60) >= CUTOFF_UTC)):
        raise SotaBookError(f"{book_id}: post-holdout timestamp or expiry detected (>= {CUTOFF_UTC})")

    X_all = df_dev[fcols].to_numpy(dtype=np.float64)
    row_ok = np.mean(~np.isfinite(X_all), axis=1) < 0.50
    elig = row_ok & valid_true & np.isfinite(ret_fwd_true) & _session_and_tradeable_mask(ts, hm, False)
    moved = elig & (ret_fwd_true != 0.0)

    idx_elig = np.flatnonzero(elig)
    gap = hm * 60
    n_val = max(500, int(round(len(idx_elig) * 0.15)))
    val_idx = idx_elig[-n_val:]
    val_start_ts = int(ts[val_idx[0]])
    fit_all_idx = idx_elig[ts[idx_elig] < (val_start_ts - gap)]
    mv_in_fit = ret_fwd_true[fit_all_idx] != 0.0

    Xf_all = X_all[fit_all_idx]
    ff_mv = ret_fwd_true[fit_all_idx[mv_in_fit]]
    yf = (ff_mv > 0.0).astype(int)
    med = np.nanmedian(np.where(np.isfinite(Xf_all), Xf_all, np.nan), axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    Zf_all = np.where(np.isfinite(Xf_all), Xf_all, med).astype(np.float64)
    mu = Zf_all.mean(axis=0)
    sd = Zf_all.std(axis=0) + 1e-12
    Zf_all = np.clip((Zf_all - mu) / sd, -5.0, 5.0)
    sw = (np.abs(ff_mv) / (np.abs(ff_mv).mean() + 1e-18)) if weight_absret else None

    elig_val_idx = np.flatnonzero(elig & (ts >= val_start_ts))
    Xv = X_all[elig_val_idx]
    fv = ret_fwd_true[elig_val_idx]
    Zv = np.clip((np.where(np.isfinite(Xv), Xv, med).astype(np.float64) - mu) / sd, -5.0, 5.0)

    fin_rows = np.flatnonzero(np.all(np.isfinite(X_all), axis=1))[:1000]
    parity_df = pd.DataFrame(X_all[fin_rows], columns=fcols)
    Zp = np.clip((X_all[fin_rows] - mu) / sd, -5.0, 5.0)

    ny_list, sgd_list = [], []
    pval_acc = np.zeros(len(Zv), dtype=np.float64)
    ppar_acc = np.zeros(len(Zp), dtype=np.float64)
    for sd_i in range(n_seeds):
        rng = np.random.default_rng(1000 + sd_i * 37)
        sub = rng.choice(len(Zf_all), size=min(len(Zf_all), 45_000), replace=False) if len(Zf_all) > 45_000 else np.arange(len(Zf_all))
        ny_s = Nystroem(kernel="rbf", gamma=1.0 / max(1, Zf_all.shape[1]), n_components=500, random_state=sd_i)
        ny_s.fit(Zf_all[sub])
        Fk_all = ny_s.transform(Zf_all)
        sgd_s = SGDClassifier(loss="log_loss", penalty="l2", alpha=1e-4, max_iter=30, tol=1e-4, random_state=sd_i)
        sgd_s.fit(Fk_all[mv_in_fit], yf, sample_weight=sw)
        ny_list.append(ny_s)
        sgd_list.append(sgd_s)
        pval_acc += sgd_s.predict_proba(ny_s.transform(Zv))[:, 1]
        ppar_acc += sgd_s.predict_proba(ny_s.transform(Zp))[:, 1]

    pval = pval_acc / float(n_seeds)
    parity_probs = ppar_acc / float(n_seeds)
    conf_thr = float(np.quantile(np.abs(pval - 0.5), 1.0 - cov))

    logits_v = np.clip(
        np.log(np.clip(pval, 1e-6, 1.0 - 1e-6) / np.clip(1.0 - pval, 1e-6, 1.0 - 1e-6)),
        -15.0,
        15.0,
    ).reshape(-1, 1)
    y_cal_side = ((fv > 0.0) if side == "UP" else (fv < 0.0)).astype(int)
    cal_clf = LogisticRegression(C=1.0, max_iter=1000).fit(logits_v, y_cal_side)
    platt = {
        side: {
            "coef": float(cal_clf.coef_[0, 0]),
            "intercept": float(cal_clf.intercept_[0]),
            "n_val_oos": int(len(elig_val_idx)),
        }
    }
    arrays = (
        extract_nystroem_ens_arrays(med, mu, sd, ny_list, sgd_list)
        if n_seeds > 1
        else extract_nystroem_arrays(med, mu, sd, ny_list[0], sgd_list[0])
    )

    g1 = dos["G1_cpcv"]
    g2 = dos["G2_placebo"]
    g3 = dos["G3_entry_delay"]
    g4 = dos["G4_sign_null"]
    g5 = dos["G5_payout"]
    certs: dict[str, dict[str, Any]] = {
        "G1_cpcv": {
            "passed": True,
            "evidence_json": dossier_json_rel,
            "paths_clear": int(g1["paths_clear_0541"]),
            "paths_counted": int(g1["paths_counted"]),
            "p10_wr": float(g1["p10_wr"]),
            "mean_wr": float(g1["mean_wr"]),
            "forward_mean_ev": g1.get("forward_mean_ev"),
            "validity_note": validity_note,
        },
        "G2_placebo": {
            "passed": True,
            "evidence_json": dossier_json_rel,
            "verdict": g2["verdict"],
            "mean_excess_hour_vol": float(g2["mean_excess_hour_vol"]),
            "paths_positive": int(g2["paths_positive"]),
        },
        "G3_entry_delay": {
            "passed": True,
            "evidence_json": dossier_json_rel,
            "verdict_delay": g3["verdict_delay"],
            "verdict_staleness": g3["verdict_staleness"],
            "d4_p10_wr": float(g3["d4_p10_wr"]),
            "d4_paths_clear": int(g3["d4_paths_clear"]),
        },
        "G4_sign_null": {
            "passed": True,
            "evidence_json": dossier_json_rel,
            "verdict": g4["verdict"],
            "real_p10_rank": g4["real_p10_rank"],
            "null_max_p10": float(g4["null_max_p10"]),
            "null_passes": int(g4["null_passes"]),
        },
        "G5_payout": {
            "passed": True,
            "evidence_json": dossier_json_rel,
            "verdict": g5["verdict"],
            "paths_ev_positive": int(g5["paths_ev_positive"]),
            "paths_counted": int(g1["paths_counted"]),
            "p10_ev": float(g5["p10_ev"]),
            "mean_ev": float(g5["mean_ev"]),
            "quoted_payout_ratio": float(dos["quoted_payout_ratio"]),
            "annual_floor_log_growth_p10": float(g5["annual_floor_log_growth_path_p10"]),
        },
        "G6_holdout_or_prospective": {
            "passed": False,
            "mode": "prospective_demo_venue_settled",
            "reason": "Sealed >= 2026-08-01 holdout preserved untouched; promote after >=50 Demo VENUE_SETTLED trades with EV CI95_lo > 0 via scripts/deriv_live_pnl_ledger.py.",
        },
    }

    return freeze_sota_book(
        book_id=book_id,
        pair=pair,
        horizon_minutes=hm,
        allowed_sides=[side],
        lifecycle_status=DEMO_CANARY_LIFECYCLE,
        model_kind=model_kind,
        feature_cols=fcols,
        coverage=cov,
        confidence_threshold=conf_thr,
        refit_cpcv_p10_floors={side: float(g1["p10_wr"])},
        platt_calibration=platt,
        gate_certificates=certs,
        summary=(
            f"[{pair}·{hm}m·{side}] SOTA {model_kind} ({n_seeds}-seed 500-comp RBF Nystroem) on {len(fcols)} native "
            f"1s-tick features (cov={cov:.3f}, quote={dos['quoted_payout_ratio']}). Passed G1 ({g1['paths_clear_0541']}/{g1['paths_counted']}, "
            f"p10={g1['p10_wr']:.4f}, mean={g1['mean_wr']:.4f}), G2 (+{g2['mean_excess_hour_vol']:.4f} hour x vol placebo), "
            f"G3 (+4s p10={g3['d4_p10_wr']:.4f}), G4 (rank {g4['real_p10_rank']}), and G5 (p10_ev={g5['p10_ev']:+.4f}, "
            f"mean_ev={g5['mean_ev']:+.4f}, g*_p10={g5['annual_floor_log_growth_path_p10']:.4f}/yr)."
        ),
        metrics={
            "cpcv_paths_clear": f"{g1['paths_clear_0541']}/{g1['paths_counted']}",
            "cpcv_p10_wr": round(float(g1["p10_wr"]), 4),
            "cpcv_mean_wr": round(float(g1["mean_wr"]), 4),
            "cpcv_p10_ev": round(float(g5["p10_ev"]), 4),
            "cpcv_mean_ev": round(float(g5["mean_ev"]), 4),
            "annual_floor_log_growth_p10": round(float(g5["annual_floor_log_growth_path_p10"]), 4),
            "est_trades_per_year": int(g5["est_trades_per_year"]),
        },
        source_script="scripts/build_certified_sota_books.py",
        weights_arrays=arrays,
        contract_family="risefall",
        drop_ny_16h=False,
        payout_edge_margin=0.005,
        min_payout_ratio=min_payout_ratio,
        parity_check_df=parity_df,
        parity_expected_probs=parity_probs,
        allow_overwrite=allow_overwrite,
        repo_root=repo_root,
    )


def build_usdjpy_15m_ens5(
    repo_root: Path = REPO_ROOT,
    dev_dir: Path = Path("/mnt/data/deriv-sota-lab/cache/dataset_all23_mh_paths_v1"),
    *,
    allow_overwrite: bool = False,
) -> Path:
    import deriv_dataset as dd
    from datetime import date

    old_allow = __import__("os").environ.get("DERIV_ALLOW_INCOMPLETE")
    __import__("os").environ["DERIV_ALLOW_INCOMPLETE"] = "1"
    try:
        files = dd.day_files("USDJPY", date(2025, 8, 11), date(2026, 7, 31))
        ts_tk, px_tk = dd.load_ticks(files)
    finally:
        if old_allow is None:
            __import__("os").environ.pop("DERIV_ALLOW_INCOMPLETE", None)
        else:
            __import__("os").environ["DERIV_ALLOW_INCOMPLETE"] = old_allow
    keep = ts_tk < CUTOFF_UTC
    ts_tk, px_tk = ts_tk[keep], px_tk[keep]
    df_src = dev_dir / "USDJPY.parquet"
    if not df_src.exists():
        df_src = DEFAULT_DEV_DATASET / "USDJPY.parquet"
    df_dev = pd.read_parquet(df_src)
    df_dev = df_dev.loc[df_dev["decision_ts"] < CUTOFF_UTC].reset_index(drop=True)
    ts_all = df_dev["decision_ts"].to_numpy(dtype=np.int64)
    rf_true, val_true = dd.settle(ts_all, ts_tk, px_tk, 900, 30)
    val_true &= (ts_all + dd.LAG_S + 900) < CUTOFF_UTC
    return freeze_from_v2_dossier(
        book_id="USDJPY.sota_ens5_m15_up.v1",
        dossier_json_rel="results/json/deriv_5gate_USDJPY_sota_ens5_m15_up_v1_result.json",
        df_dev=df_dev,
        ret_fwd_true=rf_true,
        valid_true=val_true,
        min_payout_ratio=1.76,
        allow_overwrite=allow_overwrite,
        repo_root=repo_root,
    )


def build_usdjpy_30m_ens5(
    repo_root: Path = REPO_ROOT,
    dev_dir: Path = Path("/mnt/data/deriv-sota-lab/cache/dataset_all23_mh_paths_v1"),
    *,
    allow_overwrite: bool = False,
) -> Path:
    import deriv_dataset as dd
    from datetime import date

    old_allow = __import__("os").environ.get("DERIV_ALLOW_INCOMPLETE")
    __import__("os").environ["DERIV_ALLOW_INCOMPLETE"] = "1"
    try:
        files = dd.day_files("USDJPY", date(2025, 8, 11), date(2026, 7, 31))
        ts_tk, px_tk = dd.load_ticks(files)
    finally:
        if old_allow is None:
            __import__("os").environ.pop("DERIV_ALLOW_INCOMPLETE", None)
        else:
            __import__("os").environ["DERIV_ALLOW_INCOMPLETE"] = old_allow
    keep = ts_tk < CUTOFF_UTC
    ts_tk, px_tk = ts_tk[keep], px_tk[keep]
    df_src = dev_dir / "USDJPY.parquet"
    if not df_src.exists():
        df_src = DEFAULT_DEV_DATASET / "USDJPY.parquet"
    df_dev = pd.read_parquet(df_src)
    df_dev = df_dev.loc[df_dev["decision_ts"] < CUTOFF_UTC].reset_index(drop=True)
    ts_all = df_dev["decision_ts"].to_numpy(dtype=np.int64)
    rf_true, val_true = dd.settle(ts_all, ts_tk, px_tk, 1800, 60)
    val_true &= (ts_all + dd.LAG_S + 1800) < CUTOFF_UTC
    return freeze_from_v2_dossier(
        book_id="USDJPY.sota_ens5_m30_up.v1",
        dossier_json_rel="results/json/deriv_5gate_USDJPY_sota_ens5_m30_up_v1_result.json",
        df_dev=df_dev,
        ret_fwd_true=rf_true,
        valid_true=val_true,
        min_payout_ratio=1.78,
        allow_overwrite=allow_overwrite,
        repo_root=repo_root,
    )


def _validate_fixture_root(fixture_root: Path) -> Path:
    import pwd

    fr = Path(fixture_root)
    if not fr.is_absolute() or fr.is_symlink() or not fr.is_dir():
        raise SotaBookError(f"invalid --fixture-root {fixture_root}")
    res = fr.resolve(strict=True)
    pw_home = Path(pwd.getpwuid(__import__("os").getuid()).pw_dir).resolve()
    forbidden = [pw_home, Path("/home/sean/git/binary-algo").resolve(), Path("/home/sean/git/binary-algo-vps").resolve(), Path("/mnt/data/deriv").resolve(), Path("/mnt/data/deriv-sota-lab").resolve()]
    for fb in forbidden:
        try:
            res.relative_to(fb)
            raise SotaBookError(f"refusing --fixture-root inside protected root {fb}: {res}")
        except ValueError:
            pass
    return res


def _build_stack05_7input_parity_fixture(
    X_all: np.ndarray,
    ts_all: np.ndarray,
    val_idx: np.ndarray,
    fcols: list[str],
    med: np.ndarray,
    mu: np.ndarray,
    sd: np.ndarray,
    ny_list: list[Any],
    sgd_list: list[Any],
    lgb_clf: Any,
    lgb_best_iter: int,
    conf_thr: float,
    platt_up: dict[str, float],
) -> dict[str, Any]:
    """Build non-empty 7-input score_row() parity fixture covering finite, <50%-NaN, UP boundary, and DOWN rows."""
    from sota_book_runtime import _expit

    X_val = X_all[val_idx]
    ts_val = ts_all[val_idx]
    nan_counts = np.sum(np.isnan(X_val), axis=1)
    finite_pos = np.flatnonzero(nan_counts == 0)
    nan_ok_pos = np.flatnonzero((nan_counts > 0) & (nan_counts < int(0.50 * len(fcols))))

    # Predict probabilities across validation slice to guarantee UP, boundary, and DOWN coverage
    Z_val = np.clip((np.where(np.isfinite(X_val), X_val, med).astype(np.float64) - mu) / sd, -5.0, 5.0)
    p_nys = np.mean([sgd.predict_proba(ny.transform(Z_val))[:, 1] for ny, sgd in zip(ny_list, sgd_list)], axis=0)
    p_lgb = np.asarray(lgb_clf.booster_.predict(np.asarray(X_val, dtype=np.float64), num_iteration=int(lgb_best_iter)), dtype=np.float64)
    p_raw = 0.5 * p_nys + 0.5 * p_lgb

    up_clear = np.flatnonzero((p_raw > 0.5) & (np.abs(p_raw - 0.5) >= conf_thr))
    up_below = np.flatnonzero((p_raw > 0.5) & (np.abs(p_raw - 0.5) < conf_thr))
    down_rows = np.flatnonzero(p_raw < 0.5)

    chosen: list[int] = []
    for bucket, limit in ((finite_pos, 40), (nan_ok_pos, 15), (up_clear, 20), (up_below, 15), (down_rows, 15)):
        for pos in bucket[:limit]:
            ip = int(pos)
            if ip not in chosen:
                chosen.append(ip)
    if not chosen:
        chosen = list(range(min(50, len(val_idx))))

    X_sel = X_val[chosen].copy()
    ep_sel = ts_val[chosen].astype(np.int64).copy()
    # Guarantee at least one admissible <50%-NaN row even if val had none
    if not np.any(np.isnan(X_sel)):
        nan_row = X_sel[0].copy()
        nan_row[:3] = np.nan
        X_sel = np.vstack([X_sel, nan_row[None, :]])
        ep_sel = np.append(ep_sel, ep_sel[0])

    Z_sel = np.clip((np.where(np.isfinite(X_sel), X_sel, med).astype(np.float64) - mu) / sd, -5.0, 5.0)
    p_nys_s = np.mean([sgd.predict_proba(ny.transform(Z_sel))[:, 1] for ny, sgd in zip(ny_list, sgd_list)], axis=0)
    p_lgb_s = np.asarray(lgb_clf.booster_.predict(np.asarray(X_sel, dtype=np.float64), num_iteration=int(lgb_best_iter)), dtype=np.float64)
    probs = (0.5 * p_nys_s + 0.5 * p_lgb_s).astype(np.float64)

    coef = float(platt_up["coef"])
    intercept = float(platt_up["intercept"])
    up_cal = np.full(len(probs), np.nan, dtype=np.float64)
    directions: list[str] = []
    gates: list[bool] = []
    cal_present: list[bool] = []
    for i, p in enumerate(probs):
        d = "UP" if p > 0.5 else ("DOWN" if p < 0.5 else "FLAT")
        directions.append(d)
        conf = abs(float(p) - 0.5)
        gates.append(bool(d == "UP" and conf >= conf_thr))
        if d == "UP":
            pc = float(np.clip(p, 1e-6, 1.0 - 1e-6))
            lg = float(np.clip(math.log(pc / (1.0 - pc)), -15.0, 15.0))
            up_cal[i] = float(_expit(np.array([coef * lg + intercept]))[0])
            cal_present.append(True)
        else:
            cal_present.append(False)

    return {
        "df": pd.DataFrame(X_sel, columns=fcols),
        "epochs": ep_sel,
        "probs": probs,
        "up_cal_probs": up_cal,
        "directions": directions,
        "gates": gates,
        "row_cal_present": cal_present,
    }


def _authenticate_stack05_pass_inputs(
    *,
    expected_receipt_sha256: str,
    repo_root: Path,
    bundle_dir: Path | None = None,
    arm_name: str = "stack05_m30_up_v1",
) -> tuple[dict[str, Any], dict[str, Any], bytes, bytes]:
    from certify_sota_stack05 import (
        BOOK_ID,
        DOSSIER_SCHEMA,
        G4_MODE_PLATT_EV,
        G4_MODE_RAW_TOP5,
        RECEIPT_SCHEMA,
        artifact_names_for_arm,
        sha256_bytes,
        strict_json_loads,
        validate_arm_g4_mode,
        validate_fit_ledger_obj,
    )

    names = artifact_names_for_arm(arm_name)
    rcpt_path = (bundle_dir / names["receipt"]) if bundle_dir is not None else (repo_root / "logs" / names["receipt"])
    dos_path = (bundle_dir / names["dossier"]) if bundle_dir is not None else (repo_root / "results" / "json" / names["dossier"])
    ldg_path = (bundle_dir / names["ledger"]) if bundle_dir is not None else (repo_root / "logs" / names["ledger"])
    for p in (rcpt_path, dos_path, ldg_path):
        if not p.is_file():
            raise SotaBookError(f"{BOOK_ID}: missing required certifier artifact {p}")

    rcpt_raw = rcpt_path.read_bytes()
    if sha256_bytes(rcpt_raw) != expected_receipt_sha256:
        raise SotaBookError(f"{BOOK_ID}: receipt SHA-256 mismatch at {rcpt_path}")
    rcpt = strict_json_loads(rcpt_raw)
    if rcpt.get("schema") != RECEIPT_SCHEMA or rcpt.get("verdict") != "PASS":
        raise SotaBookError(f"{BOOK_ID}: certifier receipt is not PASS ({rcpt.get('verdict')!r})")
    if str(rcpt.get("arm_name") or "stack05_m30_up_v1") != arm_name:
        raise SotaBookError(f"{BOOK_ID}: certifier receipt arm_name mismatch ({rcpt.get('arm_name')!r} != {arm_name!r})")

    dos_raw = dos_path.read_bytes()
    if sha256_bytes(dos_raw) != rcpt.get("dossier_sha256"):
        raise SotaBookError(f"{BOOK_ID}: dossier SHA-256 mismatch vs receipt")
    dos = strict_json_loads(dos_raw)
    if dos.get("schema") != DOSSIER_SCHEMA or dos.get("book_id") != BOOK_ID or dos.get("verdict") != "PASS":
        raise SotaBookError(f"{BOOK_ID}: invalid or non-PASS v3 dossier")
    if str(dos.get("arm_name") or "stack05_m30_up_v1") != arm_name or str((dos.get("preregistration_payload") or {}).get("arm_name") or "stack05_m30_up_v1") != arm_name:
        raise SotaBookError(f"{BOOK_ID}: dossier/prereg arm_name mismatch vs {arm_name!r}")
    prereg_g4_mode = (
        G4_MODE_RAW_TOP5
        if (dos.get("preregistration_payload") or {}).get("g4_selection_mode") == "raw_top5_apples_to_apples"
        else G4_MODE_PLATT_EV
    )
    try:
        validate_arm_g4_mode(arm_name, prereg_g4_mode)
    except Exception as exc:
        raise SotaBookError(f"{BOOK_ID}: {exc}") from exc
    if dos.get("all_5_gates_passed") is not True or dos.get("drift_check_passed") is not True:
        raise SotaBookError(f"{BOOK_ID}: dossier missing all_5_gates_passed or drift_check_passed")
    for g in ("G1_cpcv", "G2_placebo", "G3_entry_delay", "G4_sign_null", "G5_payout"):
        if not (isinstance(dos.get(g), dict) and dos[g].get("passed") is True):
            raise SotaBookError(f"{BOOK_ID}: dossier gate {g} did not pass")
    if (dos["G4_sign_null"] or {}).get("verdict") != "DESCRIPTIVE_RANK_DIAGNOSTIC_PASS":
        raise SotaBookError(f"{BOOK_ID}: invalid G4 verdict {dos['G4_sign_null'].get('verdict')!r}")
    if (dos["G5_payout"] or {}).get("classification") != "fixed_ratio_dev_sensitivity_only_not_decision5":
        raise SotaBookError(f"{BOOK_ID}: G5 missing fixed-ratio sensitivity classification")

    ldg_raw = ldg_path.read_bytes()
    if sha256_bytes(ldg_raw) != rcpt.get("certifier_fit_ledger_prefix_sha256"):
        raise SotaBookError(f"{BOOK_ID}: prefix fit ledger SHA-256 mismatch vs receipt")
    ldg = strict_json_loads(ldg_raw)
    cert_n, bld_n = validate_fit_ledger_obj(ldg, expected_prereg_sha=str(rcpt["preregistration_sha256"]), allow_builder_attempt=False)
    if cert_n != int(rcpt["certifier_attempt_count"]) or bld_n != 0:
        raise SotaBookError(f"{BOOK_ID}: prefix fit ledger counts invalid (cert={cert_n}, builder={bld_n})")
    return rcpt, dos, rcpt_raw, ldg_raw


def _build_stack05_gate_certs_and_metrics(dos: dict[str, Any], dossier_json_rel: str) -> tuple[dict[str, dict[str, Any]], dict[str, Any], str]:
    g1 = dos["G1_cpcv"]
    g2 = dos["G2_placebo"]
    g3 = dos["G3_entry_delay"]
    g4 = dos["G4_sign_null"]
    g5 = dos["G5_payout"]
    validity_note = (
        "Strict [0.47, 0.53] moved up-rate band satisfied"
        if dos.get("in_strict_047_053_moved_band")
        else "Owner Decision 1 drift amendment satisfied via G2_placebo + G4_sign_null + frozen 2025->2026 OOS"
    )
    certs: dict[str, dict[str, Any]] = {
        "G1_cpcv": {
            "passed": True,
            "evidence_json": dossier_json_rel,
            "paths_clear": int(g1["paths_clear_0541"]),
            "paths_clear_0565": int(g1["paths_clear_0565"]),
            "paths_counted": int(g1["paths_counted"]),
            "p10_wr": float(g1["p10_wr"]),
            "mean_wr": float(g1["mean_wr"]),
            "forward_mean_ev": float(g1["forward_mean_ev"]),
            "validity_note": validity_note,
        },
        "G2_placebo": {
            "passed": True,
            "evidence_json": dossier_json_rel,
            "verdict": str(g2["verdict"]),
            "mean_excess_hour_vol": float(g2["mean_excess_hour_vol"]),
            "paths_positive": int(g2["paths_positive"]),
        },
        "G3_entry_delay": {
            "passed": True,
            "evidence_json": dossier_json_rel,
            "verdict_delay": str(g3["verdict_delay"]),
            "verdict_staleness": str(g3["verdict_staleness"]),
            "d4_p10_wr": float(g3["d4_p10_wr"]),
            "d4_paths_clear": int(g3["d4_paths_clear"]),
        },
        "G4_sign_null": {
            "passed": True,
            "evidence_json": dossier_json_rel,
            "verdict": str(g4["verdict"]),
            "real_p10_rank": g4["real_p10_rank"],
            "null_max_p10": float(g4["null_max_p10"]),
            "null_passes": int(g4["null_passes"]),
        },
        "G5_payout": {
            "passed": True,
            "evidence_json": dossier_json_rel,
            "classification": "fixed_ratio_dev_sensitivity_only_not_decision5",
            "decision5_per_trade_30m_payout_reading": "unavailable",
            "verdict": str(g5["verdict"]),
            "paths_ev_positive": int(g5["paths_ev_positive"]),
            "paths_counted": int(g1["paths_counted"]),
            "p10_ev": float(g5["p10_ev"]),
            "mean_ev": float(g5["mean_ev"]),
            "quoted_payout_ratio": float(dos["quoted_payout_ratio"]),
            "annual_floor_log_growth_p10": float(g5["p10_ann_log_growth"]),
        },
        "G6_holdout_or_prospective": {
            "passed": False,
            "mode": "prospective_demo_venue_settled",
            "reason": "Sealed >= 2026-08-01 holdout preserved untouched; promote after >=50 Demo VENUE_SETTLED trades with EV CI95_lo > 0 via scripts/deriv_live_pnl_ledger.py.",
        },
    }
    metrics = {
        "cpcv_paths_clear": f"{g1['paths_clear_0541']}/{g1['paths_counted']}",
        "cpcv_paths_clear_0565": f"{g1['paths_clear_0565']}/{g1['paths_counted']}",
        "cpcv_p10_wr": round(float(g1["p10_wr"]), 4),
        "cpcv_mean_wr": round(float(g1["mean_wr"]), 4),
        "cpcv_p10_ev": round(float(g5["p10_ev"]), 4),
        "cpcv_mean_ev": round(float(g5["mean_ev"]), 4),
        "annual_floor_log_growth_p10": round(float(g5["p10_ann_log_growth"]), 4),
        "est_trades_per_year": int(g5["est_trades_per_year"]),
        "decision5_per_trade_30m_payout_reading": "unavailable",
    }
    summary = (
        f"[USDJPY·30m·UP] SOTA nystroem-lgb-stack (50/50 5-seed 500-comp RBF Nystroem + single-booster LightGBM) "
        f"on 66 native 1s-tick features (cov=0.05, fixed-ratio R=1.82 DEV sensitivity; Decision 5 30m per-trade reading unavailable). "
        f"Passed G1 ({g1['paths_clear_0541']}/{g1['paths_counted']}, >=0.565: {g1['paths_clear_0565']}/{g1['paths_counted']}, "
        f"p10={g1['p10_wr']:.4f}, mean={g1['mean_wr']:.4f}), G2 (+{g2['mean_excess_hour_vol']:.4f}), "
        f"G3 (+4s p10={g3['d4_p10_wr']:.4f}), G4 ({g4['verdict']}, rank {g4['real_p10_rank']}), "
        f"and G5 sensitivity (p10_ev={g5['p10_ev']:+.4f}, g*_p10={g5['p10_ann_log_growth']:.4f}/yr)."
    )
    return certs, metrics, summary


def verify_staged_stack05_in_dir(
    *,
    stage_dir: Path,
    expected_receipt_sha256: str,
    repo_root: Path = REPO_ROOT,
    arm_name: str = "stack05_m30_up_v1",
) -> dict[str, Any]:
    """Authenticate complete staged $T/staged_book/, manifest, parity, and 1-builder $T ledger (zero fits; §3.2 (6b) + B3)."""
    import tempfile
    from certify_sota_stack05 import (
        BOOK_ID,
        artifact_names_for_arm,
        canonical_json_bytes,
        sha256_bytes,
        strict_json_loads,
        validate_fit_ledger_obj,
    )
    from sota_book_freeze import _verify_7input_stack_parity, fsync_dir
    from sota_book_runtime import SOTA_INDEX_SCHEMA, compute_sota_content_id_from_buffers, load_sota_book

    names = artifact_names_for_arm(arm_name)
    rcpt, dos, _, prefix_ldg_raw = _authenticate_stack05_pass_inputs(
        expected_receipt_sha256=expected_receipt_sha256,
        repo_root=repo_root,
        arm_name=arm_name,
    )
    prereg_sha = str(rcpt["preregistration_sha256"])
    stage_manifest_path = stage_dir / "staged_install_manifest.json"
    staged_book_dir = stage_dir / "staged_book"
    t_ledger_path = stage_dir / names["ledger"]
    if not stage_manifest_path.is_file() or not staged_book_dir.is_dir() or not t_ledger_path.is_file():
        raise SotaBookError(f"{BOOK_ID}: incomplete staged evidence in {stage_dir} (holding before witness/install)")

    sm = strict_json_loads(stage_manifest_path.read_bytes())
    if sm.get("schema") != "sota-stack05-staged-install/v1":
        raise SotaBookError(f"{BOOK_ID}: invalid staged_install_manifest schema {sm.get('schema')!r}")
    if sm.get("certifier_receipt_sha256") != expected_receipt_sha256 or sm.get("preregistration_sha256") != prereg_sha:
        raise SotaBookError(f"{BOOK_ID}: staged_install_manifest receipt/prereg mismatch")
    if sm.get("book_content_id") != rcpt.get("final_fit_expected_content_id"):
        raise SotaBookError(f"{BOOK_ID}: staged content_id mismatch vs receipt")

    t_ldg_raw = t_ledger_path.read_bytes()
    t_ldg = strict_json_loads(t_ldg_raw)
    cert_n, bld_n = validate_fit_ledger_obj(t_ldg, expected_prereg_sha=prereg_sha, allow_builder_attempt=True)
    if cert_n != int(rcpt["certifier_attempt_count"]) or bld_n != 1:
        raise SotaBookError(f"{BOOK_ID}: staged $T ledger counts invalid (cert={cert_n}, builder={bld_n})")
    prefix_ldg = strict_json_loads(prefix_ldg_raw)
    if t_ldg["attempts"][:-1] != prefix_ldg["attempts"]:
        raise SotaBookError(f"{BOOK_ID}: staged $T ledger prefix diverges from immutable 0-builder prefix")
    builder_attempt = t_ldg["attempts"][-1]
    if builder_attempt.get("stage") != "builder" or builder_attempt.get("role") != "builder_final_fit":
        raise SotaBookError(f"{BOOK_ID}: unexpected builder attempt entry {builder_attempt!r}")
    post_builder_sha = sha256_bytes(canonical_json_bytes(t_ldg))
    if post_builder_sha != sm.get("post_builder_fit_ledger_sha256"):
        raise SotaBookError(f"{BOOK_ID}: staged ledger sha256 mismatch vs staged_install_manifest")

    # Authenticate staged book artifacts against receipt.final_fit_expected_artifacts
    exp_arts = sorted(rcpt["final_fit_expected_artifacts"], key=lambda x: x["filename"])
    bufs: list[tuple[str, bytes]] = []
    for ea in exp_arts:
        fn = str(ea["filename"])
        p = staged_book_dir / fn
        if not p.is_file():
            raise SotaBookError(f"{BOOK_ID}: missing staged book file {p}")
        raw = p.read_bytes()
        if len(raw) != int(ea["bytes"]) or sha256_bytes(raw) != str(ea["sha256"]):
            raise SotaBookError(f"{BOOK_ID}: staged file {fn!r} digest/size mismatch vs receipt")
        bufs.append((fn, raw))
    cid = compute_sota_content_id_from_buffers(bufs)
    if cid != str(rcpt["final_fit_expected_content_id"]):
        raise SotaBookError(f"{BOOK_ID}: staged content_id {cid!r} != expected {rcpt['final_fit_expected_content_id']!r}")

    # Re-verify staged 7-input reload parity inside stage_dir with zero fits
    par_npz_path = stage_dir / "staged_parity_fixture.npz"
    if not par_npz_path.is_file():
        raise SotaBookError(f"{BOOK_ID}: missing staged_parity_fixture.npz in {stage_dir}")
    with np.load(par_npz_path, allow_pickle=False) as pz:
        X_par = pz["X"].copy()
        ep_par = pz["epochs"].copy()
        p_par = pz["probs"].copy()
        cal_par = pz["up_cal_probs"].copy()
        dir_codes = pz["dir_codes"].copy()
        gates_par = pz["gates"].copy()
        cal_pres_par = pz["row_cal_present"].copy()
    code_to_dir = {0: "FLAT", 1: "UP", -1: "DOWN"}
    dirs_par = [code_to_dir[int(c)] for c in dir_codes.tolist()]
    strat_obj = strict_json_loads((staged_book_dir / f"{BOOK_ID}_strategy.json").read_bytes())
    fcols = list(strat_obj["feature_cols"])
    man_path = staged_book_dir / f"{BOOK_ID}.manifest.json"
    if not man_path.is_file():
        raise SotaBookError(f"{BOOK_ID}: missing staged manifest {man_path}")
    man_obj = strict_json_loads(man_path.read_bytes())
    with tempfile.TemporaryDirectory(prefix=".verify_stage_", dir=stage_dir) as td:
        troot = Path(td)
        tb = troot / "books" / BOOK_ID
        tb.mkdir(parents=True)
        for p in staged_book_dir.iterdir():
            (tb / p.name).write_bytes(p.read_bytes())
        (troot / "books" / "SOTA_INDEX.json").write_text(
            json.dumps(
                {
                    "schema": SOTA_INDEX_SCHEMA,
                    "books": [
                        {
                            "id": BOOK_ID,
                            "content_id": cid,
                            "pair": "USDJPY",
                            "horizon_minutes": 30,
                            "allowed_sides": ["UP"],
                            "model_kind": "nystroem-lgb-stack",
                            "lifecycle_status": DEMO_CANARY_LIFECYCLE,
                            "manifest": f"books/{BOOK_ID}/{BOOK_ID}.manifest.json",
                            "metrics": man_obj["metrics"],
                        }
                    ],
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        loaded = load_sota_book(BOOK_ID, repo_root=troot, allow_demo_canary=True)
        _verify_7input_stack_parity(
            BOOK_ID,
            loaded,
            parity_check_df=pd.DataFrame(X_par, columns=fcols),
            parity_check_epochs=ep_par,
            parity_expected_probs=p_par,
            parity_expected_up_cal_probs=cal_par,
            parity_expected_directions=dirs_par,
            parity_expected_gates=gates_par.tolist(),
            parity_expected_row_cal_present=cal_pres_par.tolist(),
        )
    # B3: re-fsync $T on retry before witnessing/installing
    fsync_dir(staged_book_dir)
    fsync_dir(stage_dir)

    witness_line = (
        f"Builder fit debit: arm={arm_name} preregistration_sha256={prereg_sha} "
        f"receipt_sha256={expected_receipt_sha256} ledger_sha256={post_builder_sha} "
        f"attempt_id={int(builder_attempt['attempt_id'])}"
    )
    return {
        "stage_manifest": sm,
        "builder_attempt": builder_attempt,
        "post_builder_fit_ledger_sha256": post_builder_sha,
        "witness_line": witness_line,
        "parity": {
            "df": pd.DataFrame(X_par, columns=fcols),
            "epochs": ep_par,
            "probs": p_par,
            "up_cal_probs": cal_par,
            "directions": dirs_par,
            "gates": gates_par.tolist(),
            "row_cal_present": cal_pres_par.tolist(),
        },
        "manifest": man_obj,
        "strategy": strat_obj,
        "artifacts": bufs,
    }


def stage_stack05_in_lab_dir(
    *,
    out_dir: Path,
    expected_receipt_sha256: str,
    progress_md: Path,
    repo_root: Path = REPO_ROOT,
    dev_dir: Path = DEFAULT_DEV_DATASET,
    ticks_root: Path = Path("/mnt/data/deriv/ticks"),
    allow_fixture_digest_override: bool = False,
    fixture_root: Path | None = None,
    arm_name: str = "stack05_m30_up_v1",
) -> dict[str, Any]:
    """Step (6a): Lab-capped staging fit writing STRICTLY inside out_dir ($T); zero checkout/$A writes."""
    import tempfile
    from certify_sota_stack05 import (
        BOOK_ID,
        HORIZON_S,
        PAIR,
        TOL_S,
        artifact_names_for_arm,
        build_all_role_specs,
        build_eligible_indices,
        build_stack05_strategy_dict,
        canonical_json_bytes,
        debit_fit_ledger,
        fit_role_calibration,
        fit_role_trainer,
        load_pinned_dev_parquet,
        load_pinned_dev_ticks,
        load_verified_scores_npz,
        select_core66_columns,
        settle_with_intervals,
        sha256_bytes,
        strict_json_loads,
        validate_fit_ledger_obj,
        verify_source_closure_and_loaded_modules,
    )
    from sota_book_freeze import (
        _verify_7input_stack_parity,
        atomic_write_file_bytes,
        canonical_deterministic_manifest_sha256,
        extract_nystroem_lgb_stack_arrays,
        fsync_dir,
        fsync_file,
        serialize_sota_lgb_txt_bytes,
        serialize_sota_strategy_bytes,
        serialize_sota_weights_npz_bytes,
    )
    from sota_book_runtime import (
        SOTA_INDEX_SCHEMA,
        SOTA_MANIFEST_SCHEMA,
        check_pipeline_features_dir_exists,
        compute_sota_content_id_from_buffers,
        load_sota_book,
    )

    if allow_fixture_digest_override:
        if fixture_root is None:
            raise SotaBookError("--allow-fixture-digest-override requires explicit validated --fixture-root")
        _validate_fixture_root(fixture_root)
    else:
        check_pipeline_features_dir_exists()
        verify_source_closure_and_loaded_modules(repo_root)

    names = artifact_names_for_arm(arm_name)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rcpt, dos, _, prefix_ldg_raw = _authenticate_stack05_pass_inputs(
        expected_receipt_sha256=expected_receipt_sha256,
        repo_root=repo_root,
        arm_name=arm_name,
    )
    prereg_sha = str(rcpt["preregistration_sha256"])

    t_ledger_path = out_dir / names["ledger"]
    t_lock_path = out_dir / names["lock"]
    stage_manifest_path = out_dir / "staged_install_manifest.json"
    staged_book_dir = out_dir / "staged_book"

    # D2 + B3 + BLOCKER #6 fix: initialize $T ledger ONCE under exclusive debit lock with noreplace semantics!
    import fcntl
    with open(t_lock_path, "a+", encoding="utf-8") as lfh:
        try:
            fcntl.flock(lfh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SotaBookError(f"{BOOK_ID}: staging lock busy: {t_lock_path}") from exc
        if t_ledger_path.exists():
            cur_ldg = strict_json_loads(t_ledger_path.read_bytes())
            _, bld_n = validate_fit_ledger_obj(cur_ldg, expected_prereg_sha=prereg_sha, allow_builder_attempt=True)
            if bld_n > 0:
                if stage_manifest_path.is_file():
                    info = verify_staged_stack05_in_dir(
                        stage_dir=out_dir,
                        expected_receipt_sha256=expected_receipt_sha256,
                        repo_root=repo_root,
                        arm_name=arm_name,
                    )
                    return {
                        "staged": True,
                        "resumed_zero_fit": True,
                        "book_content_id": info["stage_manifest"]["book_content_id"],
                        "post_builder_fit_ledger_sha256": info["post_builder_fit_ledger_sha256"],
                        "witness_line": info["witness_line"],
                    }
                raise SotaBookError(
                    f"{BOOK_ID}: builder debit already spent in {t_ledger_path} without complete authenticated stage (HOLDING with debit intact)"
                )
        else:
            atomic_write_file_bytes(t_ledger_path, prefix_ldg_raw, noreplace=True, sync_parent=True)

    # Authenticate certifier final_fit_val before debiting
    scores_dict, _, scores_sha = load_verified_scores_npz(repo_root / "logs" / names["scores"])
    if scores_sha != str(rcpt["scores_npz_sha256"]):
        raise SotaBookError(f"{BOOK_ID}: checkout scores.npz sha256 mismatch vs receipt")
    cert_ff_val = scores_dict["final_fit_val"]

    # Load pinned DEV inputs & rebuild exact final_fit split
    df_dev = load_pinned_dev_parquet(PAIR, dev_dir=dev_dir, pinned_parquets=(None if not allow_fixture_digest_override else {PAIR: ((dev_dir / f"{PAIR}.parquet").stat().st_size, sha256_bytes((dev_dir / f"{PAIR}.parquet").read_bytes()))}))
    fcols = select_core66_columns(df_dev)
    tick_ts, tick_px, _ = load_pinned_dev_ticks(
        PAIR,
        dev_dir=dev_dir,
        ticks_root=ticks_root,
        expected_manifest_sha256=(None if allow_fixture_digest_override else "4f434cfd799e237647776b7f3e3e14b04be305594bd55bf37344321be9185c4a"),
    )
    ts_all = df_dev["decision_ts"].to_numpy(dtype=np.int64)
    X_all = df_dev[fcols].to_numpy(dtype=np.float64)
    ret_fwd_true, valid_true, entry_true, exit_true = settle_with_intervals(
        ts_all, tick_ts, tick_px, horizon_s=HORIZON_S, delay_s=0, tol_s=TOL_S
    )
    eligible_idx = build_eligible_indices(df_dev, fcols, valid_true, ret_fwd_true)
    role_specs = build_all_role_specs(ts_all, ret_fwd_true, valid_true, entry_true, exit_true, eligible_idx)
    ff_spec = next(r for r in role_specs if r["role"] == "final_fit")

    # Debit the single builder_final_fit attempt inside $T BEFORE training!
    builder_attempt = debit_fit_ledger(
        t_ledger_path,
        t_lock_path,
        prereg_sha=prereg_sha,
        stage="builder",
        role="builder_final_fit",
    )
    t_ldg_after = strict_json_loads(t_ledger_path.read_bytes())
    post_builder_sha = sha256_bytes(canonical_json_bytes(t_ldg_after))

    fit_res = fit_role_trainer(X_all, fcols, ff_spec, return_estimators=True)
    val_diff = float(np.max(np.abs(fit_res["val_p_raw"] - cert_ff_val)))
    if not math.isfinite(val_diff) or val_diff >= 1e-12:
        raise SotaBookError(f"{BOOK_ID}: builder final_fit_val parity diff {val_diff:.3e} >= 1e-12")

    cal_m = fit_role_calibration(fit_res["val_p_raw"], ff_spec["ret_arr"][ff_spec["val_idx"]], 0.05)
    if not cal_m["cal_admissible"]:
        raise SotaBookError(f"{BOOK_ID}: builder final_fit UP calibration inadmissible")
    platt_cal = {
        "UP": {
            "coef": float(cal_m["platt_coef"]),
            "intercept": float(cal_m["platt_intercept"]),
            "n_val_oos": int(cal_m["n_val_oos"]),
        }
    }
    strat = build_stack05_strategy_dict(
        feature_cols=fcols,
        confidence_threshold=float(cal_m["conf_thr"]),
        p10_wr=float(dos["G1_cpcv"]["p10_wr"]),
        platt_calibration=platt_cal,
    )
    w_arrays = extract_nystroem_lgb_stack_arrays(
        fit_res["med"],
        fit_res["mu"],
        fit_res["sd"],
        fit_res["ny_list"],
        fit_res["sgd_list"],
        blend_weight=0.5,
    )
    strat_bytes = serialize_sota_strategy_bytes(strat)
    weights_bytes = serialize_sota_weights_npz_bytes(w_arrays)
    lgb_bytes = serialize_sota_lgb_txt_bytes(fit_res["lgb_clf"], fcols, num_iteration=int(fit_res["lgb_best_iteration"]))

    items = [
        (f"{BOOK_ID}_strategy.json", strat_bytes),
        (f"{BOOK_ID}_weights.npz", weights_bytes),
        (f"{BOOK_ID}_lgb_s0.txt", lgb_bytes),
    ]
    cid = compute_sota_content_id_from_buffers(items)
    if cid != str(rcpt["final_fit_expected_content_id"]):
        raise SotaBookError(f"{BOOK_ID}: builder content_id {cid!r} != certifier expected {rcpt['final_fit_expected_content_id']!r}")
    got_arts = sorted(
        [{"filename": n, "bytes": len(b), "sha256": sha256_bytes(b)} for n, b in items],
        key=lambda x: x["filename"],
    )
    exp_arts = sorted(rcpt["final_fit_expected_artifacts"], key=lambda x: x["filename"])
    if got_arts != exp_arts:
        raise SotaBookError(f"{BOOK_ID}: builder artifact digests mismatch vs certifier receipt")

    dossier_rel = f"results/json/{names['dossier']}"
    certs, metrics, summary = _build_stack05_gate_certs_and_metrics(dos, dossier_rel)
    manifest = {
        "schema": SOTA_MANIFEST_SCHEMA,
        "id": BOOK_ID,
        "content_id": cid,
        "pair": "USDJPY",
        "horizon_minutes": 30,
        "allowed_sides": ["UP"],
        "lifecycle_status": DEMO_CANARY_LIFECYCLE,
        "model_kind": "nystroem-lgb-stack",
        "settlement_substrate": "deriv_1s_ticks",
        "summary": summary,
        "metrics": metrics,
        "gate_certificates": certs,
        "artifacts": [{"file": a["filename"], "bytes": a["bytes"], "sha256": a["sha256"]} for a in got_arts],
        "evidence_binding": {
            "dossier_path": dossier_rel,
            "dossier_sha256": str(rcpt["dossier_sha256"]),
            "certifier_receipt_sha256": expected_receipt_sha256,
            "preregistration_sha256": prereg_sha,
            "certifier_fit_ledger_prefix_sha256": str(rcpt["certifier_fit_ledger_prefix_sha256"]),
            "post_builder_fit_ledger_sha256": post_builder_sha,
        },
        "source": {
            "script": "scripts/build_certified_sota_books.py",
            "source_sha256": sha256_bytes((repo_root / "scripts/build_certified_sota_books.py").read_bytes()),
            "git_sha": None,
        },
        "env": {"python": sys.executable, "python_version": sys.version.split()[0]},
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    man_bytes = (json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")

    # Build & verify 7-input reload parity strictly inside $T!
    par = _build_stack05_7input_parity_fixture(
        X_all,
        ts_all,
        ff_spec["val_idx"],
        fcols,
        fit_res["med"],
        fit_res["mu"],
        fit_res["sd"],
        fit_res["ny_list"],
        fit_res["sgd_list"],
        fit_res["lgb_clf"],
        int(fit_res["lgb_best_iteration"]),
        float(cal_m["conf_thr"]),
        platt_cal["UP"],
    )
    staged_book_dir.mkdir(parents=True, exist_ok=True)
    for fname, raw in items + [(f"{BOOK_ID}.manifest.json", man_bytes)]:
        dst = staged_book_dir / fname
        atomic_write_file_bytes(dst, raw, sync_parent=False)
        fsync_file(dst)
    # D2: fsync every staged book file AND $T/staged_book/ BEFORE writing staged_install_manifest.json!
    fsync_dir(staged_book_dir)

    with tempfile.TemporaryDirectory(prefix=".parity_", dir=out_dir) as ptd:
        proot = Path(ptd)
        pb = proot / "books" / BOOK_ID
        pb.mkdir(parents=True)
        for p in staged_book_dir.iterdir():
            (pb / p.name).write_bytes(p.read_bytes())
        (proot / "books" / "SOTA_INDEX.json").write_text(
            json.dumps(
                {
                    "schema": SOTA_INDEX_SCHEMA,
                    "books": [
                        {
                            "id": BOOK_ID,
                            "content_id": cid,
                            "pair": "USDJPY",
                            "horizon_minutes": 30,
                            "allowed_sides": ["UP"],
                            "model_kind": "nystroem-lgb-stack",
                            "lifecycle_status": DEMO_CANARY_LIFECYCLE,
                            "manifest": f"books/{BOOK_ID}/{BOOK_ID}.manifest.json",
                            "metrics": metrics,
                        }
                    ],
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        loaded = load_sota_book(BOOK_ID, repo_root=proot, allow_demo_canary=True)
        _verify_7input_stack_parity(
            BOOK_ID,
            loaded,
            parity_check_df=par["df"],
            parity_check_epochs=par["epochs"],
            parity_expected_probs=par["probs"],
            parity_expected_up_cal_probs=par["up_cal_probs"],
            parity_expected_directions=par["directions"],
            parity_expected_gates=par["gates"],
            parity_expected_row_cal_present=par["row_cal_present"],
        )

    # Save parity fixture through an open binary file handle (prevents np.savez appending .npz to .tmp!) + fsync $T (D2)
    dir_to_code = {"FLAT": 0, "UP": 1, "DOWN": -1}
    par_tmp = out_dir / ".staged_parity_fixture.npz.tmp"
    with open(par_tmp, "wb") as pfh:
        np.savez(
            pfh,
            X=par["df"].to_numpy(dtype=np.float64),
            epochs=np.asarray(par["epochs"], dtype=np.int64),
            probs=np.asarray(par["probs"], dtype=np.float64),
            up_cal_probs=np.asarray(par["up_cal_probs"], dtype=np.float64),
            dir_codes=np.asarray([dir_to_code[d] for d in par["directions"]], dtype=np.int8),
            gates=np.asarray(par["gates"], dtype=bool),
            row_cal_present=np.asarray(par["row_cal_present"], dtype=bool),
        )
        pfh.flush()
        os.fsync(pfh.fileno())
    par_tmp.replace(out_dir / "staged_parity_fixture.npz")
    fsync_dir(out_dir)

    sm = {
        "schema": "sota-stack05-staged-install/v1",
        "book_id": BOOK_ID,
        "book_content_id": cid,
        "preregistration_sha256": prereg_sha,
        "certifier_receipt_sha256": expected_receipt_sha256,
        "certifier_fit_ledger_prefix_sha256": str(rcpt["certifier_fit_ledger_prefix_sha256"]),
        "post_builder_fit_ledger_sha256": post_builder_sha,
        "builder_attempt": builder_attempt,
        "artifacts": got_arts,
        "deterministic_manifest_sha256": canonical_deterministic_manifest_sha256(manifest),
        "val_parity_max_abs_diff": val_diff,
        "parity_rows": len(par["df"]),
    }
    atomic_write_file_bytes(stage_manifest_path, (json.dumps(sm, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8"), sync_parent=True)
    fsync_dir(out_dir)

    witness_line = (
        f"Builder fit debit: arm={arm_name} preregistration_sha256={prereg_sha} "
        f"receipt_sha256={expected_receipt_sha256} ledger_sha256={post_builder_sha} "
        f"attempt_id={int(builder_attempt['attempt_id'])}"
    )
    return {
        "staged": True,
        "resumed_zero_fit": False,
        "book_content_id": cid,
        "post_builder_fit_ledger_sha256": post_builder_sha,
        "witness_line": witness_line,
        "stage_dir": str(out_dir),
    }


def install_staged_stack05(
    *,
    stage_dir: Path,
    bundle_dir: Path,
    expected_receipt_sha256: str,
    progress_md: Path,
    repo_root: Path = REPO_ROOT,
    pre_index_hook: Any | None = None,
    arm_name: str = "stack05_m30_up_v1",
) -> Path:
    """Step (6b): Strictly ZERO-FIT installer publishing staged book, 0444 builder_suffix.json & SOTA_INDEX.json."""
    from certify_sota_stack05 import (
        BOOK_ID,
        CANONICAL_BUILDER_SUFFIX_NAME,
        artifact_names_for_arm,
        parse_progress_builder_witnesses,
    )
    from sota_book_freeze import complete_interrupted_stack_index_install

    names = artifact_names_for_arm(arm_name)
    expected_install_path = repo_root / "logs" / "sota_stack05_expected_install.json"
    book_dir = repo_root / "books" / BOOK_ID

    # Authenticate untouched immutable 0-builder prefix in BOTH checkout AND bundle_dir ($A) FIRST (incl. on retry)!
    rcpt_chk, dos_chk, _, ldg_chk = _authenticate_stack05_pass_inputs(
        expected_receipt_sha256=expected_receipt_sha256,
        repo_root=repo_root,
        bundle_dir=None,
        arm_name=arm_name,
    )
    rcpt_arc, _, _, ldg_arc = _authenticate_stack05_pass_inputs(
        expected_receipt_sha256=expected_receipt_sha256,
        repo_root=repo_root,
        bundle_dir=bundle_dir,
        arm_name=arm_name,
    )
    if ldg_chk != ldg_arc or rcpt_chk != rcpt_arc:
        raise SotaBookError(f"{BOOK_ID}: checkout and archive 0-builder prefixes disagree")

    # Zero-fit interrupted dir-to-index retry path (B2 + BLOCKER #7: fully authenticates archive builder_suffix, reconstructed ledger & witness FIRST!)
    if book_dir.exists() and expected_install_path.is_file():
        from certify_sota_stack05 import (
            canonical_json_bytes,
            sha256_bytes,
            strict_json_loads,
            validate_fit_ledger_obj,
        )
        suffix_p = Path(bundle_dir) / CANONICAL_BUILDER_SUFFIX_NAME
        if not suffix_p.is_file():
            raise SotaBookError(f"{BOOK_ID}: interrupted retry refused — missing {suffix_p}")
        suffix_obj = strict_json_loads(suffix_p.read_bytes())
        if not isinstance(suffix_obj, dict):
            raise SotaBookError(f"{BOOK_ID}: interrupted retry refused — invalid builder_suffix.json object")
        prereg_sha_r = str(rcpt_chk["preregistration_sha256"])
        if suffix_obj.get("preregistration_sha256") != prereg_sha_r:
            raise SotaBookError(f"{BOOK_ID}: interrupted retry refused — builder_suffix preregistration_sha256 mismatch")
        if suffix_obj.get("certifier_receipt_sha256") != expected_receipt_sha256:
            raise SotaBookError(f"{BOOK_ID}: interrupted retry refused — builder_suffix certifier_receipt_sha256 mismatch")
        if suffix_obj.get("certifier_fit_ledger_prefix_sha256") != rcpt_chk.get("certifier_fit_ledger_prefix_sha256"):
            raise SotaBookError(f"{BOOK_ID}: interrupted retry refused — builder_suffix prefix ledger sha mismatch")
        if suffix_obj.get("book_content_id") != rcpt_chk.get("final_fit_expected_content_id"):
            raise SotaBookError(f"{BOOK_ID}: interrupted retry refused — builder_suffix content_id mismatch")
        b_att = suffix_obj.get("builder_attempt")
        if not isinstance(b_att, dict) or b_att.get("stage") != "builder" or b_att.get("role") != "builder_final_fit":
            raise SotaBookError(f"{BOOK_ID}: interrupted retry refused — builder_suffix missing valid builder_final_fit attempt")
        prefix_ldg_obj = strict_json_loads(ldg_chk)
        recon_ldg = {
            "schema": prefix_ldg_obj["schema"],
            "preregistration_sha256": prefix_ldg_obj["preregistration_sha256"],
            "max_fit_units": prefix_ldg_obj["max_fit_units"],
            "attempts": list(prefix_ldg_obj["attempts"]) + [b_att],
        }
        recon_cert, recon_bld = validate_fit_ledger_obj(recon_ldg, expected_prereg_sha=prereg_sha_r, allow_builder_attempt=True)
        if recon_cert != int(rcpt_chk["certifier_attempt_count"]) or recon_bld != 1:
            raise SotaBookError(f"{BOOK_ID}: interrupted retry refused — reconstructed ledger counts invalid ({recon_cert}, {recon_bld})")
        recon_post_sha = sha256_bytes(canonical_json_bytes(recon_ldg))
        if suffix_obj.get("post_builder_fit_ledger_sha256") != recon_post_sha:
            raise SotaBookError(f"{BOOK_ID}: interrupted retry refused — builder_suffix post_builder_fit_ledger_sha256 mismatch")
        witnesses_r = [
            w
            for w in parse_progress_builder_witnesses(progress_md)
            if w["preregistration_sha256"] == prereg_sha_r and w["receipt_sha256"] == expected_receipt_sha256 and w["arm"] == arm_name
        ]
        if (
            len(witnesses_r) != 1
            or witnesses_r[0]["ledger_sha256"] != recon_post_sha
            or int(witnesses_r[0]["attempt_id"]) != int(b_att["attempt_id"])
        ):
            raise SotaBookError(f"{BOOK_ID}: interrupted retry refused — archive builder_suffix/PROGRESS witness mismatch")
        return complete_interrupted_stack_index_install(
            book_id=BOOK_ID,
            repo_root=repo_root,
            expected_install_path=expected_install_path,
        )

    staged = verify_staged_stack05_in_dir(
        stage_dir=stage_dir,
        expected_receipt_sha256=expected_receipt_sha256,
        repo_root=repo_root,
        arm_name=arm_name,
    )
    prereg_sha = str(rcpt_chk["preregistration_sha256"])
    witnesses = [
        w
        for w in parse_progress_builder_witnesses(progress_md)
        if w["preregistration_sha256"] == prereg_sha and w["receipt_sha256"] == expected_receipt_sha256 and w["arm"] == arm_name
    ]
    if len(witnesses) != 1 or witnesses[0]["ledger_sha256"] != staged["post_builder_fit_ledger_sha256"]:
        raise SotaBookError(
            f"{BOOK_ID}: PROGRESS.md missing exact Builder fit debit witness before zero-fit install "
            f"(expected ledger_sha256={staged['post_builder_fit_ledger_sha256']})"
        )

    # Load weights_arrays & lgb_model_bytes directly from authenticated staged_book/ (ZERO FITS!)
    sbook_dir = stage_dir / "staged_book"
    with np.load(sbook_dir / f"{BOOK_ID}_weights.npz", allow_pickle=False) as z:
        weights_arrays = {k: z[k].copy() for k in z.files}
    lgb_bytes = (f"{BOOK_ID}_lgb_s0.txt", (sbook_dir / f"{BOOK_ID}_lgb_s0.txt").read_bytes())

    strat = staged["strategy"]
    par = staged["parity"]
    dossier_rel = f"results/json/{names['dossier']}"
    certs, metrics, summary = _build_stack05_gate_certs_and_metrics(dos_chk, dossier_rel)
    ev_binding = {
        "dossier_path": dossier_rel,
        "dossier_sha256": str(rcpt_chk["dossier_sha256"]),
        "certifier_receipt_sha256": expected_receipt_sha256,
        "preregistration_sha256": prereg_sha,
        "certifier_fit_ledger_prefix_sha256": str(rcpt_chk["certifier_fit_ledger_prefix_sha256"]),
        "post_builder_fit_ledger_sha256": staged["post_builder_fit_ledger_sha256"],
    }
    suffix_meta = {
        "preregistration_sha256": prereg_sha,
        "certifier_receipt_sha256": expected_receipt_sha256,
        "certifier_fit_ledger_prefix_sha256": str(rcpt_chk["certifier_fit_ledger_prefix_sha256"]),
        "builder_attempt": staged["builder_attempt"],
        "post_builder_fit_ledger_sha256": staged["post_builder_fit_ledger_sha256"],
        "progress_witness_line": witnesses[0]["line"],
    }

    return freeze_sota_book(
        book_id=BOOK_ID,
        pair="USDJPY",
        horizon_minutes=30,
        allowed_sides=["UP"],
        lifecycle_status=DEMO_CANARY_LIFECYCLE,
        model_kind="nystroem-lgb-stack",
        feature_cols=list(strat["feature_cols"]),
        coverage=float(strat["coverage"]),
        confidence_threshold=float(strat["confidence_threshold"]),
        refit_cpcv_p10_floors=dict(strat["refit_cpcv_p10_floors"]),
        platt_calibration=dict(strat["platt_calibration"]),
        gate_certificates=certs,
        summary=summary,
        metrics=metrics,
        source_script="scripts/build_certified_sota_books.py",
        weights_arrays=weights_arrays,
        lgb_model_bytes=lgb_bytes,
        contract_family="risefall",
        drop_ny_16h=False,
        payout_edge_margin=0.005,
        min_payout_ratio=1.78,
        parity_check_df=par["df"],
        parity_check_epochs=par["epochs"],
        parity_expected_probs=par["probs"],
        parity_expected_up_cal_probs=par["up_cal_probs"],
        parity_expected_directions=par["directions"],
        parity_expected_gates=par["gates"],
        parity_expected_row_cal_present=par["row_cal_present"],
        evidence_binding=ev_binding,
        expected_content_id=str(rcpt_chk["final_fit_expected_content_id"]),
        expected_artifact_digests=list(rcpt_chk["final_fit_expected_artifacts"]),
        expected_install_path=expected_install_path,
        builder_suffix_path=bundle_dir / CANONICAL_BUILDER_SUFFIX_NAME,
        builder_suffix_metadata=suffix_meta,
        archive_sync_stop_at=bundle_dir.parent.parent,
        pre_index_hook=pre_index_hook,
        allow_overwrite=False,
        repo_root=repo_root,
    )


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--books", default="", help="Optional specific book builder target (e.g. 'stack05')")
    ap.add_argument("--arm-name", default="stack05_m30_up_v1", help="Registered stack05 arm name (default: stack05_m30_up_v1)")
    ap.add_argument("--stage-only", action="store_true", help="Stack05 step (6a): capped fit staging strictly inside --out-dir")
    ap.add_argument("--install-staged-stack05", action="store_true", help="Stack05 step (6b): zero-fit installer from --stage-dir")
    ap.add_argument("--verify-staged-stack05", action="store_true", help="Stack05 step (6b) pre-witness zero-fit stage check")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--stage-dir", default=None)
    ap.add_argument("--bundle-dir", default=None)
    ap.add_argument("--expected-receipt-sha256", default="")
    ap.add_argument("--progress-md", default="/mnt/data/deriv-sota-lab/PROGRESS.md")
    ap.add_argument("--allow-fixture-digest-override", action="store_true")
    ap.add_argument("--fixture-root", default=None)
    ap.add_argument("--only-ens5", action="store_true", help="Build/refresh only the two 5-seed ensemble books (default)")
    ap.add_argument("--include-legacy", action="store_true", help="Also rebuild the two earlier 1-seed baseline books")
    ap.add_argument("--allow-overwrite", action="store_true", help="Allow overwriting existing book directory")
    args = ap.parse_args()

    if args.books == "stack05" or args.stage_only or args.install_staged_stack05 or args.verify_staged_stack05:
        from certify_sota_stack05 import default_archive_dir_for_arm, default_out_dir_for_arm
        arm_name = str(args.arm_name)
        eff_out_dir = Path(args.out_dir) if args.out_dir else default_out_dir_for_arm(arm_name)
        eff_stage_dir = Path(args.stage_dir) if args.stage_dir else default_out_dir_for_arm(arm_name)
        eff_bundle_dir = Path(args.bundle_dir) if args.bundle_dir else default_archive_dir_for_arm(arm_name)
        if not args.expected_receipt_sha256:
            raise SotaBookError("stack05 builder requires --expected-receipt-sha256")
        if args.stage_only and args.install_staged_stack05:
            raise SotaBookError("--stage-only and --install-staged-stack05 are mutually exclusive")
        if args.stage_only:
            res = stage_stack05_in_lab_dir(
                out_dir=eff_out_dir,
                expected_receipt_sha256=args.expected_receipt_sha256.strip(),
                progress_md=Path(args.progress_md),
                allow_fixture_digest_override=bool(args.allow_fixture_digest_override),
                fixture_root=Path(args.fixture_root) if args.fixture_root else None,
                arm_name=arm_name,
            )
            print(json.dumps(res, indent=2, sort_keys=True))
            return 0
        if args.verify_staged_stack05:
            info = verify_staged_stack05_in_dir(
                stage_dir=eff_stage_dir,
                expected_receipt_sha256=args.expected_receipt_sha256.strip(),
                arm_name=arm_name,
            )
            print(
                json.dumps(
                    {
                        "verified": True,
                        "book_content_id": info["stage_manifest"]["book_content_id"],
                        "post_builder_fit_ledger_sha256": info["post_builder_fit_ledger_sha256"],
                        "witness_line": info["witness_line"],
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        if args.install_staged_stack05:
            man = install_staged_stack05(
                stage_dir=eff_stage_dir,
                bundle_dir=eff_bundle_dir,
                expected_receipt_sha256=args.expected_receipt_sha256.strip(),
                progress_md=Path(args.progress_md),
                arm_name=arm_name,
            )
            print(f"FROZEN: {man}")
            return 0
        raise SotaBookError("specify --stage-only, --verify-staged-stack05, or --install-staged-stack05 for --books stack05")

    if args.include_legacy and not args.only_ens5:
        m15 = build_usdjpy_15m_c2nysabs()
        print(f"FROZEN: {m15}")
        m30 = build_usdjpy_30m_e49nys30()
        print(f"FROZEN: {m30}")
    e15 = build_usdjpy_15m_ens5(allow_overwrite=args.allow_overwrite)
    print(f"FROZEN: {e15}")
    e30 = build_usdjpy_30m_ens5(allow_overwrite=args.allow_overwrite)
    print(f"FROZEN: {e30}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
