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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--only-ens5", action="store_true", help="Build/refresh only the two 5-seed ensemble books (default)")
    ap.add_argument("--include-legacy", action="store_true", help="Also rebuild the two earlier 1-seed baseline books")
    ap.add_argument("--allow-overwrite", action="store_true", help="Allow overwriting existing book directory")
    args = ap.parse_args()
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
