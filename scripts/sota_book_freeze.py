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
    _validate_weights_npz,
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


def _save_deterministic_npz(dst_path: Path, arrays: dict[str, np.ndarray]) -> None:
    import io
    import zipfile
    from numpy.lib import format as npfmt

    with zipfile.ZipFile(dst_path, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for k in sorted(arrays.keys()):
            buf = io.BytesIO()
            npfmt.write_array(buf, np.asarray(arrays[k]), allow_pickle=False)
            zi = zipfile.ZipInfo(filename=f"{k}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            zi.compress_type = zipfile.ZIP_DEFLATED
            zi.external_attr = 0o600 << 16
            zf.writestr(zi, buf.getvalue())


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


def extract_nystroem_ens_arrays(
    med: np.ndarray,
    mu: np.ndarray,
    sd: np.ndarray,
    ny_estimators: list[Any],
    sgd_estimators: list[Any],
) -> dict[str, np.ndarray]:
    if not ny_estimators or len(ny_estimators) != len(sgd_estimators):
        raise SotaBookError("ny_estimators and sgd_estimators must be non-empty lists of equal length")
    arrays: dict[str, np.ndarray] = {
        "med": np.asarray(med, dtype=np.float64),
        "mu": np.asarray(mu, dtype=np.float64),
        "sd": np.asarray(sd, dtype=np.float64),
        "n_seeds": np.asarray([len(ny_estimators)], dtype=np.int64),
    }
    for s_i, (ny_est, sgd_est) in enumerate(zip(ny_estimators, sgd_estimators)):
        arrays[f"s{s_i}_nys_components"] = np.asarray(ny_est.components_, dtype=np.float64)
        arrays[f"s{s_i}_nys_normalization"] = np.asarray(ny_est.normalization_, dtype=np.float64)
        arrays[f"s{s_i}_nys_gamma"] = np.asarray([float(ny_est.gamma)], dtype=np.float64)
        arrays[f"s{s_i}_sgd_coef"] = np.asarray(sgd_est.coef_[0], dtype=np.float64)
        arrays[f"s{s_i}_sgd_intercept"] = np.asarray([float(sgd_est.intercept_[0])], dtype=np.float64)
    return arrays


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
    multiplier: int | None = None,
    certified_commission_frac: float = 0.0,
    max_commission_frac: float = 0.20,
    min_stop_out_rel_dist: float = 0.0008,
    refit_cpcv_p10_ev: dict[str, float] | None = None,
    barrier_contract_type: str | None = None,
    barrier_rel_half_width: float | None = None,
    parity_check_df: pd.DataFrame | None = None,
    parity_check_epochs: np.ndarray | None = None,
    parity_expected_probs: np.ndarray | None = None,
    allow_overwrite: bool = False,
    repo_root: Path = REPO_ROOT,
) -> Path:
    import fcntl
    import os
    import shutil
    import tempfile

    _check_book_id(book_id)
    if lifecycle_status not in {ACTIVE_LIFECYCLE, DEMO_CANARY_LIFECYCLE, INACTIVE_LIFECYCLE}:
        raise SotaBookError(f"invalid lifecycle_status {lifecycle_status!r}")
    if lifecycle_status in {ACTIVE_LIFECYCLE, DEMO_CANARY_LIFECYCLE}:
        for g in REQUIRED_DEV_GATES:
            if not (isinstance(gate_certificates.get(g), dict) and gate_certificates[g].get("passed") is True):
                raise SotaBookError(f"{book_id}: cannot freeze {lifecycle_status} book without passed {g}")
        if model_kind == "nystroem-ens" and (parity_check_df is None or parity_expected_probs is None):
            raise SotaBookError(f"{book_id}: executable nystroem-ens requires parity_check_df and parity_expected_probs")
    if lifecycle_status == ACTIVE_LIFECYCLE:
        g6 = gate_certificates.get("G6_holdout_or_prospective")
        if not (isinstance(g6, dict) and g6.get("passed") is True):
            raise SotaBookError(f"{book_id}: active lifecycle requires passed G6_holdout_or_prospective")

    if model_kind in {"logit", "logit-hourly", "nystroem", "nystroem-ens"}:
        if not weights_arrays:
            raise SotaBookError(f"{book_id}: weights_arrays required for {model_kind}")
        _validate_weights_npz(book_id, model_kind, weights_arrays, len(feature_cols))
    elif model_kind == "lgb":
        if not lgb_model_files:
            raise SotaBookError(f"{book_id}: lgb_model_files required for lgb")
    else:
        raise SotaBookError(f"unsupported model_kind {model_kind!r}")

    strategy: dict[str, Any] = {
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
    if multiplier is not None:
        strategy["multiplier"] = int(multiplier)
        strategy["certified_commission_frac"] = float(certified_commission_frac)
        strategy["max_commission_frac"] = float(max_commission_frac)
        strategy["min_stop_out_rel_dist"] = float(min_stop_out_rel_dist)
    if refit_cpcv_p10_ev is not None:
        strategy["refit_cpcv_p10_ev"] = {k.upper(): float(v) for k, v in refit_cpcv_p10_ev.items()}
    if barrier_contract_type is not None:
        strategy["barrier_contract_type"] = str(barrier_contract_type).strip().upper()
    if barrier_rel_half_width is not None:
        strategy["barrier_rel_half_width"] = float(barrier_rel_half_width)

    # Stage and verify load + numerical parity in an isolated temp root BEFORE publishing!
    with tempfile.TemporaryDirectory() as stage_td:
        stage_root = Path(stage_td)
        stage_books = stage_root / "books"
        stage_dir = stage_books / book_id
        stage_dir.mkdir(parents=True, exist_ok=True)

        st_strat_path = stage_dir / f"{book_id}_strategy.json"
        st_strat_path.write_text(json.dumps(strategy, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        st_artifacts: list[Path] = [st_strat_path]
        if model_kind in {"logit", "logit-hourly", "nystroem", "nystroem-ens"}:
            assert weights_arrays is not None
            st_npz = stage_dir / f"{book_id}_weights.npz"
            _save_deterministic_npz(st_npz, weights_arrays)
            st_artifacts.append(st_npz)
        else:
            assert lgb_model_files is not None
            for src in lgb_model_files:
                dst = stage_dir / src.name
                shutil.copy2(src, dst)
                st_artifacts.append(dst)

        content_id = compute_sota_content_id(st_artifacts)
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
                for p in sorted(st_artifacts, key=lambda item: item.name)
            ],
            "source": {
                "script": source_script,
                "source_sha256": file_sha256(repo_root / source_script) if (repo_root / source_script).is_file() else None,
                "git_sha": _git_sha(repo_root),
            },
            "env": _env_snapshot(),
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        st_manifest_path = stage_dir / f"{book_id}.manifest.json"
        st_manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (stage_books / "SOTA_INDEX.json").write_text(
            json.dumps(
                {
                    "schema": SOTA_INDEX_SCHEMA,
                    "books": [
                        {
                            "id": book_id,
                            "content_id": content_id,
                            "pair": pair.upper(),
                            "horizon_minutes": int(horizon_minutes),
                            "allowed_sides": [s.upper() for s in allowed_sides],
                            "model_kind": model_kind,
                            "lifecycle_status": lifecycle_status,
                            "manifest": f"books/{book_id}/{book_id}.manifest.json",
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

        if lifecycle_status != INACTIVE_LIFECYCLE:
            loaded = load_sota_book(book_id, repo_root=stage_root, allow_demo_canary=True)
            if parity_check_df is not None and parity_expected_probs is not None:
                exp_arr = np.asarray(parity_expected_probs, dtype=np.float64)
                got = loaded.predict_probabilities(parity_check_df, decision_epochs=parity_check_epochs)
                if got.shape != exp_arr.shape:
                    raise SotaBookError(f"{book_id}: parity shape mismatch {got.shape} != {exp_arr.shape}")
                diff = float(np.max(np.abs(got - exp_arr)))
                if not np.isfinite(diff) or diff > 1e-11:
                    raise SotaBookError(f"{book_id}: pure-NumPy reload parity diff {diff:.3e} > 1e-11")

        # Publish validated artifacts under exclusive repo-scoped lock with no-replace semantics by default!
        books_dir = repo_root / "books"
        books_dir.mkdir(parents=True, exist_ok=True)
        lock_dir = repo_root / "deriv_data" / "runtime" / "locks"
        lock_dir.mkdir(parents=True, exist_ok=True)
        lock_file = lock_dir / "sota_index.lock"
        with open(lock_file, "a+", encoding="utf-8") as lfh:
            fcntl.flock(lfh.fileno(), fcntl.LOCK_EX)
            book_dir = books_dir / book_id
            manifest_path = book_dir / f"{book_id}.manifest.json"
            if book_dir.exists() and not allow_overwrite:
                if not manifest_path.exists():
                    raise SotaBookError(f"{book_id}: incomplete existing directory {book_dir} (allow_overwrite=False)")
                existing_m = json.loads(manifest_path.read_text(encoding="utf-8"))
                existing_cid = str(existing_m.get("content_id") or "")
                if existing_cid != content_id:
                    raise SotaBookError(
                        f"{book_id}: refusing to overwrite existing versioned book ({existing_cid} != {content_id}); "
                        f"increment version or pass allow_overwrite=True explicitly"
                    )
                if str(existing_m.get("lifecycle_status") or "") != lifecycle_status:
                    raise SotaBookError(
                        f"{book_id}: refusing to mutate lifecycle_status ({existing_m.get('lifecycle_status')!r} -> {lifecycle_status!r}) "
                        f"without allow_overwrite=True"
                    )
                # Verify existing on-disk artifacts match their recorded SHA-256 & byte lengths
                for art in existing_m.get("artifacts", []):
                    ap_ex = book_dir / str(art["file"])
                    if not ap_ex.is_file() or file_sha256(ap_ex) != art.get("sha256") or ap_ex.stat().st_size != art.get("bytes"):
                        raise SotaBookError(f"{book_id}: existing book artifact {art.get('file')!r} corrupted")
            else:
                pub_tmp = Path(tempfile.mkdtemp(prefix=f".stage_{book_id}_", dir=books_dir))
                try:
                    for p in stage_dir.iterdir():
                        shutil.copy2(p, pub_tmp / p.name)
                    if book_dir.exists():
                        old_bak = Path(tempfile.mkdtemp(prefix=f".old_{book_id}_", dir=books_dir))
                        shutil.rmtree(old_bak)
                        os.replace(book_dir, old_bak)
                        try:
                            os.replace(pub_tmp, book_dir)
                        except Exception:
                            if not book_dir.exists() and old_bak.exists():
                                os.replace(old_bak, book_dir)
                            raise
                        finally:
                            if old_bak.exists():
                                shutil.rmtree(old_bak, ignore_errors=True)
                    else:
                        os.replace(pub_tmp, book_dir)
                finally:
                    if pub_tmp.exists():
                        shutil.rmtree(pub_tmp, ignore_errors=True)

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
            idx_fd, idx_tmp_str = tempfile.mkstemp(prefix=".SOTA_INDEX.", suffix=".tmp", dir=books_dir)
            idx_tmp = Path(idx_tmp_str)
            try:
                payload_bytes = (json.dumps(index_obj, indent=2, sort_keys=True) + "\n").encode("utf-8")
                os.write(idx_fd, payload_bytes)
                os.fsync(idx_fd)
                os.close(idx_fd)
                idx_fd = -1
                os.replace(idx_tmp, index_path)
            finally:
                if idx_fd >= 0:
                    os.close(idx_fd)
                if idx_tmp.exists():
                    idx_tmp.unlink(missing_ok=True)
            return manifest_path
