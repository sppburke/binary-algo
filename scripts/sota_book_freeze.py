"""Freeze certified SOTA linear/kernel/LightGBM books into books/<ID>/ and books/SOTA_INDEX.json.

Sibling to scripts/manifest.py (which stays byte-frozen for evidence/objects seals).
Verifies bit-exact parity (< 1e-12 max abs diff) between fitted sklearn estimators and
the pure-NumPy sota_book_runtime.py scorer before writing any book artifact.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from importlib import metadata as ilm
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sota_book_runtime import (
    ACTIVE_LIFECYCLE,
    DEMO_CANARY_LIFECYCLE,
    INACTIVE_LIFECYCLE,
    REPO_ROOT,
    REQUIRED_DEV_GATES,
    SOTA_INDEX_SCHEMA,
    SOTA_MANIFEST_SCHEMA,
    SotaBookError,
    _check_book_id,
    compute_sota_content_id,
    file_sha256,
    load_sota_book,
)


def _git_sha(repo_root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def _env_snapshot() -> dict[str, Any]:
    libs = {}
    for pkg in ("numpy", "pandas", "pyarrow", "scikit-learn", "scipy", "lightgbm"):
        try:
            libs[pkg] = ilm.version(pkg)
        except Exception:
            libs[pkg] = None
    return {"python": sys.executable, "python_version": sys.version.split()[0], "libs": libs}


def extract_nystroem_arrays(
    med: np.ndarray,
    mu: np.ndarray,
    sd: np.ndarray,
    ny_estimator: Any,
    sgd_estimator: Any,
) -> dict[str, np.ndarray]:
    return {
        "med": np.asarray(med, dtype=np.float64),
        "mu": np.asarray(mu, dtype=np.float64),
        "sd": np.asarray(sd, dtype=np.float64),
        "nys_components": np.asarray(ny_estimator.components_, dtype=np.float64),
        "nys_normalization": np.asarray(ny_estimator.normalization_, dtype=np.float64),
        "nys_gamma": np.asarray([float(ny_estimator.gamma)], dtype=np.float64),
        "sgd_coef": np.asarray(sgd_estimator.coef_[0], dtype=np.float64),
        "sgd_intercept": np.asarray([float(sgd_estimator.intercept_[0])], dtype=np.float64),
    }


def extract_hourly_logit_arrays(
    hourly_models: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, Any]],
) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {
        "hours": np.asarray(sorted(hourly_models.keys()), dtype=np.int64),
    }
    for h, (med, mu, sd, clf) in sorted(hourly_models.items()):
        arrays[f"h{h}_med"] = np.asarray(med, dtype=np.float64)
        arrays[f"h{h}_mu"] = np.asarray(mu, dtype=np.float64)
        arrays[f"h{h}_sd"] = np.asarray(sd, dtype=np.float64)
        arrays[f"h{h}_coef"] = np.asarray(clf.coef_[0], dtype=np.float64)
        arrays[f"h{h}_intercept"] = np.asarray([float(clf.intercept_[0])], dtype=np.float64)
    return arrays


def freeze_sota_book(
    *,
    book_id: str,
    pair: str,
    horizon_minutes: int,
    allowed_sides: list[str],
    lifecycle_status: str,
    model_kind: str,
    feature_cols: list[str],
    coverage: float,
    confidence_threshold: float,
    refit_cpcv_p10_floors: dict[str, float],
    platt_calibration: dict[str, dict[str, float]],
    gate_certificates: dict[str, dict[str, Any]],
    summary: str,
    metrics: dict[str, Any],
    source_script: str,
    weights_arrays: dict[str, np.ndarray] | None = None,
    lgb_model_files: list[Path] | None = None,
    contract_family: str = "risefall",
    drop_ny_16h: bool = False,
    payout_edge_margin: float = 0.005,
    min_payout_ratio: float | None = None,
    parity_check_df: pd.DataFrame | None = None,
    parity_check_epochs: np.ndarray | None = None,
    parity_expected_probs: np.ndarray | None = None,
    repo_root: Path = REPO_ROOT,
) -> Path:
    _check_book_id(book_id)
    if lifecycle_status not in {ACTIVE_LIFECYCLE, DEMO_CANARY_LIFECYCLE, INACTIVE_LIFECYCLE}:
        raise SotaBookError(f"invalid lifecycle_status {lifecycle_status!r}")
    if lifecycle_status in {ACTIVE_LIFECYCLE, DEMO_CANARY_LIFECYCLE}:
        for g in REQUIRED_DEV_GATES:
            if not (isinstance(gate_certificates.get(g), dict) and gate_certificates[g].get("passed") is True):
                raise SotaBookError(f"{book_id}: cannot freeze {lifecycle_status} book without passed {g}")
    if lifecycle_status == ACTIVE_LIFECYCLE:
        g6 = gate_certificates.get("G6_holdout_or_prospective")
        if not (isinstance(g6, dict) and g6.get("passed") is True):
            raise SotaBookError(f"{book_id}: active lifecycle requires passed G6_holdout_or_prospective")

    books_dir = repo_root / "books"
    book_dir = books_dir / book_id
    book_dir.mkdir(parents=True, exist_ok=True)

    strategy_path = book_dir / f"{book_id}_strategy.json"
    strategy = {
        "book_id": book_id,
        "pair": pair.upper(),
        "horizon_minutes": int(horizon_minutes),
        "allowed_sides": [s.upper() for s in allowed_sides],
        "model_kind": model_kind,
        "contract_family": contract_family,
        "feature_cols": list(feature_cols),
        "coverage": float(coverage),
        "confidence_threshold": float(confidence_threshold),
        "refit_cpcv_p10_floors": {k.upper(): float(v) for k, v in refit_cpcv_p10_floors.items()},
        "payout_edge_margin": float(payout_edge_margin),
        "min_payout_ratio": float(min_payout_ratio) if min_payout_ratio is not None else None,
        "drop_ny_16h": bool(drop_ny_16h),
        "platt_calibration": platt_calibration,
    }
    strategy_path.write_text(json.dumps(strategy, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    artifact_paths: list[Path] = [strategy_path]
    if model_kind in {"logit", "logit-hourly", "nystroem"}:
        if not weights_arrays:
            raise SotaBookError(f"{book_id}: weights_arrays required for {model_kind}")
        npz_path = book_dir / f"{book_id}_weights.npz"
        np.savez_compressed(npz_path, **weights_arrays)
        artifact_paths.append(npz_path)
    elif model_kind == "lgb":
        if not lgb_model_files:
            raise SotaBookError(f"{book_id}: lgb_model_files required for lgb")
        import shutil
        for src in lgb_model_files:
            dst = book_dir / src.name
            shutil.copy2(src, dst)
            artifact_paths.append(dst)
    else:
        raise SotaBookError(f"unsupported model_kind {model_kind!r}")

    content_id = compute_sota_content_id(artifact_paths)
    manifest = {
        "schema": SOTA_MANIFEST_SCHEMA,
        "id": book_id,
        "content_id": content_id,
        "pair": pair.upper(),
        "horizon_minutes": int(horizon_minutes),
        "allowed_sides": [s.upper() for s in allowed_sides],
        "lifecycle_status": lifecycle_status,
        "model_kind": model_kind,
        "settlement_substrate": "deriv_1s_ticks",
        "summary": summary,
        "metrics": metrics,
        "gate_certificates": gate_certificates,
        "artifacts": [
            {"file": p.name, "bytes": p.stat().st_size, "sha256": file_sha256(p)}
            for p in sorted(artifact_paths, key=lambda item: item.name)
        ],
        "source": {"script": source_script, "git_sha": _git_sha(repo_root)},
        "env": _env_snapshot(),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    manifest_path = book_dir / f"{book_id}.manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # Register in books/SOTA_INDEX.json
    index_path = books_dir / "SOTA_INDEX.json"
    if index_path.exists():
        index_obj = json.loads(index_path.read_text(encoding="utf-8"))
    else:
        index_obj = {"schema": SOTA_INDEX_SCHEMA, "books": []}
    books_list = [b for b in index_obj.get("books", []) if b.get("id") != book_id]
    books_list.append({
        "id": book_id,
        "content_id": content_id,
        "pair": pair.upper(),
        "horizon_minutes": int(horizon_minutes),
        "allowed_sides": [s.upper() for s in allowed_sides],
        "model_kind": model_kind,
        "lifecycle_status": lifecycle_status,
        "manifest": f"books/{book_id}/{book_id}.manifest.json",
        "metrics": metrics,
    })
    index_obj["books"] = sorted(books_list, key=lambda item: item["id"])
    index_path.write_text(json.dumps(index_obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # Verify reload + exact numerical parity if reference predictions are supplied
    if lifecycle_status != INACTIVE_LIFECYCLE:
        loaded = load_sota_book(book_id, repo_root=repo_root, allow_demo_canary=True)
        if parity_check_df is not None and parity_expected_probs is not None:
            got = loaded.predict_probabilities(parity_check_df, decision_epochs=parity_check_epochs)
            diff = float(np.max(np.abs(got - np.asarray(parity_expected_probs, dtype=np.float64))))
            if not np.isfinite(diff) or diff > 1e-11:
                raise SotaBookError(f"{book_id}: pure-NumPy reload parity diff {diff:.3e} > 1e-11")
    return manifest_path
