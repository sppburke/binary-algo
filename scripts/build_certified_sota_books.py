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

from sota_book_freeze import extract_nystroem_arrays, freeze_sota_book
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


def main() -> int:
    m15 = build_usdjpy_15m_c2nysabs()
    print(f"FROZEN: {m15}")
    m30 = build_usdjpy_30m_e49nys30()
    print(f"FROZEN: {m30}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
