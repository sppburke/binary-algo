"""Freeze certified SOTA linear/kernel/LightGBM books into books/<ID>/ and books/SOTA_INDEX.json.

Sibling to scripts/manifest.py (which stays byte-frozen for evidence/objects seals).
Verifies bit-exact parity (< 1e-12 max abs diff) between fitted sklearn estimators and
the pure-NumPy sota_book_runtime.py scorer before writing any book artifact.
"""

from __future__ import annotations

import copy
import ctypes
import errno
import hashlib
import io
import json
import math
import os
import stat
import subprocess
import sys
import time
import zipfile
from importlib import metadata as ilm
from pathlib import Path
from typing import Any

import lightgbm as lgb
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
    compute_sota_content_id_from_buffers,
    file_sha256,
    load_sota_book,
)

AT_FDCWD = -100
RENAME_NOREPLACE = 1
_LIBC = ctypes.CDLL("libc.so.6", use_errno=True)
_LIBC.renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
_LIBC.renameat2.restype = ctypes.c_int


def fsync_file(path: Path) -> None:
    fd = os.open(str(path), os.O_RDONLY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def fsync_dir(path: Path) -> None:
    fd = os.open(str(path), os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def fsync_ancestry(path: Path, stop_at: Path | None = None) -> None:
    """Fsync directory path and parent directories bottom-up through stop_at (inclusive)."""
    cur = Path(path)
    stop_res = stop_at.resolve() if stop_at is not None else None
    seen: set[Path] = set()
    while True:
        res = cur.resolve()
        if res in seen:
            break
        seen.add(res)
        fsync_dir(cur)
        if stop_res is not None and res == stop_res:
            break
        if cur.parent == cur:
            break
        cur = cur.parent


def rename_noreplace(src: Path, dst: Path) -> None:
    """Atomic Linux renameat2(..., RENAME_NOREPLACE=1) failing with FileExistsError if dst exists."""
    rc = _LIBC.renameat2(
        AT_FDCWD,
        os.fsencode(str(src)),
        AT_FDCWD,
        os.fsencode(str(dst)),
        ctypes.c_uint(RENAME_NOREPLACE),
    )
    if rc != 0:
        err = ctypes.get_errno()
        raise OSError(err, os.strerror(err), str(src), None, str(dst))


def atomic_write_file_bytes(
    dst: Path,
    payload: bytes,
    *,
    mode: int | None = None,
    noreplace: bool = False,
    sync_parent: bool = True,
) -> None:
    """Write payload to same-FS temp file, set mode BEFORE fsync, publish atomically, fsync parent."""
    import tempfile

    parent = dst.parent
    parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_str = tempfile.mkstemp(prefix=f".{dst.name}.", suffix=".tmp", dir=parent)
    tmp = Path(tmp_str)
    try:
        view = memoryview(payload)
        written = 0
        while written < len(view):
            written += os.write(fd, view[written:])
        if mode is not None:
            os.fchmod(fd, mode)
        os.fsync(fd)
        os.close(fd)
        fd = -1
        if noreplace:
            rename_noreplace(tmp, dst)
        else:
            os.replace(tmp, dst)
        if sync_parent:
            fsync_dir(parent)
    finally:
        if fd >= 0:
            os.close(fd)
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def _git_sha(repo_root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def serialize_sota_strategy_bytes(strategy: dict[str, Any]) -> bytes:
    return (json.dumps(strategy, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def serialize_sota_weights_npz_bytes(arrays: dict[str, np.ndarray]) -> bytes:
    from numpy.lib import format as npfmt

    out = io.BytesIO()
    with zipfile.ZipFile(out, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for k in sorted(arrays.keys()):
            buf = io.BytesIO()
            npfmt.write_array(buf, np.asarray(arrays[k]), allow_pickle=False)
            zi = zipfile.ZipInfo(filename=f"{k}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            zi.compress_type = zipfile.ZIP_DEFLATED
            zi.external_attr = 0o600 << 16
            zf.writestr(zi, buf.getvalue())
    return out.getvalue()


def serialize_sota_lgb_txt_bytes(
    booster_or_clf: Any,
    feature_cols: list[str],
    *,
    num_iteration: int,
) -> bytes:
    booster = booster_or_clf.booster_ if hasattr(booster_or_clf, "booster_") else booster_or_clf
    if not isinstance(booster, lgb.Booster):
        raise SotaBookError(f"expected LightGBM Booster or fitted LGBMClassifier, got {type(booster_or_clf).__name__}")
    n_iter = int(num_iteration)
    if n_iter < 1:
        raise SotaBookError(f"num_iteration must be >= 1, got {num_iteration!r}")
    if int(booster.num_feature()) != len(feature_cols):
        raise SotaBookError(
            f"LightGBM num_feature={booster.num_feature()} != len(feature_cols)={len(feature_cols)}"
        )
    names = list(booster.feature_name())
    if names != list(feature_cols):
        raise SotaBookError(
            f"LightGBM booster.feature_name() mismatch vs ordered feature_cols "
            f"({names[:5]!r} != {list(feature_cols)[:5]!r})"
        )
    raw_txt = booster.model_to_string(num_iteration=n_iter)
    if not isinstance(raw_txt, str) or not raw_txt.strip():
        raise SotaBookError("LightGBM model_to_string() produced empty output")
    reloaded = lgb.Booster(model_str=raw_txt)
    if int(reloaded.num_feature()) != len(feature_cols) or list(reloaded.feature_name()) != list(feature_cols):
        raise SotaBookError("LightGBM reloaded booster failed ordered feature_name() check")
    return raw_txt.encode("utf-8")


def canonical_deterministic_manifest_sha256(manifest: dict[str, Any]) -> str:
    """Hash canonical manifest JSON excluding ONLY declared install-time provenance fields."""
    m = copy.deepcopy(manifest)
    m.pop("created_utc", None)
    src = m.get("source")
    if isinstance(src, dict):
        src.pop("git_sha", None)
    env = m.get("env")
    if isinstance(env, dict):
        env.pop("python", None)
    payload = json.dumps(m, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _save_deterministic_npz(dst_path: Path, arrays: dict[str, np.ndarray]) -> None:
    dst_path.write_bytes(serialize_sota_weights_npz_bytes(arrays))


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


def extract_nystroem_lgb_stack_arrays(
    med: np.ndarray,
    mu: np.ndarray,
    sd: np.ndarray,
    ny_estimators: list[Any],
    sgd_estimators: list[Any],
    *,
    blend_weight: float = 0.5,
) -> dict[str, np.ndarray]:
    if len(ny_estimators) != 5 or len(sgd_estimators) != 5:
        raise SotaBookError("nystroem-lgb-stack requires exact 5 Nystroem/SGD seed pairs")
    bw = float(blend_weight)
    if not math.isfinite(bw) or bw != 0.5:
        raise SotaBookError(f"nystroem-lgb-stack requires exact blend_weight==0.5, got {blend_weight!r}")
    arrays = extract_nystroem_ens_arrays(med, mu, sd, ny_estimators, sgd_estimators)
    arrays["blend_weight"] = np.asarray([bw], dtype=np.float64)
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


def _verify_7input_stack_parity(
    book_id: str,
    loaded: Any,
    *,
    parity_check_df: pd.DataFrame,
    parity_check_epochs: np.ndarray,
    parity_expected_probs: np.ndarray,
    parity_expected_up_cal_probs: np.ndarray,
    parity_expected_directions: list[str] | np.ndarray,
    parity_expected_gates: list[bool] | np.ndarray,
    parity_expected_row_cal_present: list[bool] | np.ndarray,
) -> None:
    n = len(parity_check_df)
    if n < 1:
        raise SotaBookError(f"{book_id}: 7-input stack parity inputs must be non-empty")
    epochs = np.asarray(parity_check_epochs, dtype=np.int64)
    exp_probs = np.asarray(parity_expected_probs, dtype=np.float64)
    exp_cal = np.asarray(parity_expected_up_cal_probs, dtype=np.float64)
    exp_dirs = [str(d) for d in parity_expected_directions]
    exp_gates = [bool(g) for g in parity_expected_gates]
    exp_cal_pres = [bool(c) for c in parity_expected_row_cal_present]
    if not (
        epochs.shape == (n,)
        and exp_probs.shape == (n,)
        and exp_cal.shape == (n,)
        and len(exp_dirs) == n
        and len(exp_gates) == n
        and len(exp_cal_pres) == n
    ):
        raise SotaBookError(f"{book_id}: 7-input stack parity lengths do not match ({n})")

    # Verify batch predict_probabilities parity
    got_batch = loaded.predict_probabilities(parity_check_df, decision_epochs=epochs)
    if got_batch.shape != (n,) or not np.isfinite(got_batch).all():
        raise SotaBookError(f"{book_id}: non-finite batch probabilities during parity check")
    batch_diff = float(np.max(np.abs(got_batch - exp_probs)))
    if not math.isfinite(batch_diff) or batch_diff >= 1e-11:
        raise SotaBookError(f"{book_id}: stack batch reload parity diff {batch_diff:.3e} >= 1e-11")

    # Verify row-by-row score_row() parity across all 7 dimensions
    for i in range(n):
        row = parity_check_df.iloc[i]
        sc = loaded.score_row(row, int(epochs[i]))
        if not math.isfinite(sc.proba_up_raw) or abs(sc.proba_up_raw - float(exp_probs[i])) >= 1e-11:
            raise SotaBookError(
                f"{book_id}: score_row[{i}] raw prob diff {abs(sc.proba_up_raw - float(exp_probs[i])):.3e} >= 1e-11"
            )
        if sc.direction != exp_dirs[i]:
            raise SotaBookError(f"{book_id}: score_row[{i}] direction {sc.direction!r} != {exp_dirs[i]!r}")
        if bool(sc.gate_passed) != exp_gates[i]:
            raise SotaBookError(f"{book_id}: score_row[{i}] gate_passed {sc.gate_passed!r} != {exp_gates[i]!r}")
        cal_pres = sc.calibrated_win_prob is not None
        if cal_pres != exp_cal_pres[i]:
            raise SotaBookError(f"{book_id}: score_row[{i}] cal_present {cal_pres!r} != {exp_cal_pres[i]!r}")
        if cal_pres:
            assert sc.calibrated_win_prob is not None
            if not math.isfinite(exp_cal[i]) or abs(sc.calibrated_win_prob - float(exp_cal[i])) >= 1e-11:
                raise SotaBookError(
                    f"{book_id}: score_row[{i}] calibrated prob diff "
                    f"{abs(sc.calibrated_win_prob - float(exp_cal[i])):.3e} >= 1e-11"
                )


def _publish_sota_index_atomic(books_dir: Path, entry: dict[str, Any]) -> None:
    import tempfile

    book_id = str(entry["id"])
    index_path = books_dir / "SOTA_INDEX.json"
    if index_path.exists():
        index_obj = json.loads(index_path.read_text(encoding="utf-8"))
    else:
        index_obj = {"schema": SOTA_INDEX_SCHEMA, "books": []}
    books_list = [b for b in index_obj.get("books", []) if b.get("id") != book_id]
    books_list.append(entry)
    index_obj["books"] = sorted(books_list, key=lambda item: item["id"])
    idx_fd, idx_tmp_str = tempfile.mkstemp(prefix=".SOTA_INDEX.", suffix=".tmp", dir=books_dir)
    idx_tmp = Path(idx_tmp_str)
    try:
        payload_bytes = (json.dumps(index_obj, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
        view = memoryview(payload_bytes)
        written = 0
        while written < len(view):
            written += os.write(idx_fd, view[written:])
        os.fsync(idx_fd)
        os.close(idx_fd)
        idx_fd = -1
        os.replace(idx_tmp, index_path)
        fsync_dir(books_dir)
    finally:
        if idx_fd >= 0:
            os.close(idx_fd)
        if idx_tmp.exists():
            idx_tmp.unlink(missing_ok=True)


def _verify_installed_book_dir_strict(
    book_dir: Path,
    *,
    book_id: str,
    expected_artifacts: list[dict[str, Any]],
    expected_content_id: str,
    expected_det_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    """Strictly authenticate an installed book directory via O_DIRECTORY|O_NOFOLLOW + S_ISREG + st_nlink==1."""
    dir_fd = os.open(str(book_dir), os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        st_dir = os.fstat(dir_fd)
        if not stat.S_ISDIR(st_dir.st_mode):
            raise SotaBookError(f"{book_id}: {book_dir} is not a directory")
        manifest_name = f"{book_id}.manifest.json"
        expected_names = {str(a["file"]) for a in expected_artifacts} | {manifest_name}
        actual_names = set(os.listdir(dir_fd))
        if actual_names != expected_names:
            raise SotaBookError(
                f"{book_id}: installed directory file set mismatch "
                f"(actual={sorted(actual_names)} != expected={sorted(expected_names)})"
            )
        art_buffers: list[tuple[str, bytes]] = []
        exp_by_name = {str(a["file"]): a for a in expected_artifacts}
        manifest_obj: dict[str, Any] | None = None
        for name in sorted(expected_names):
            f_fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=dir_fd)
            try:
                fst = os.fstat(f_fd)
                if not stat.S_ISREG(fst.st_mode) or fst.st_nlink != 1:
                    raise SotaBookError(
                        f"{book_id}: installed artifact {name!r} failed S_ISREG/st_nlink==1 check (nlink={fst.st_nlink})"
                    )
                chunks: list[bytes] = []
                while True:
                    chunk = os.read(f_fd, 1 << 20)
                    if not chunk:
                        break
                    chunks.append(chunk)
                raw = b"".join(chunks)
            finally:
                os.close(f_fd)
            if name == manifest_name:
                manifest_obj = json.loads(raw.decode("utf-8"))
            else:
                ea = exp_by_name[name]
                if len(raw) != int(ea["bytes"]) or hashlib.sha256(raw).hexdigest() != str(ea["sha256"]):
                    raise SotaBookError(f"{book_id}: installed artifact {name!r} failed single-read bytes/sha256 check")
                art_buffers.append((name, raw))
        cid = compute_sota_content_id_from_buffers(art_buffers)
        if cid != expected_content_id:
            raise SotaBookError(f"{book_id}: installed content_id {cid!r} != expected {expected_content_id!r}")
        assert manifest_obj is not None
        if manifest_obj.get("content_id") != expected_content_id:
            raise SotaBookError(f"{book_id}: manifest content_id mismatch")
        if expected_det_manifest_sha256 is not None:
            det_sha = canonical_deterministic_manifest_sha256(manifest_obj)
            if det_sha != expected_det_manifest_sha256:
                raise SotaBookError(
                    f"{book_id}: deterministic_manifest_sha256 {det_sha!r} != expected {expected_det_manifest_sha256!r}"
                )
        return manifest_obj
    finally:
        os.close(dir_fd)


def complete_interrupted_stack_index_install(
    *,
    book_id: str,
    repo_root: Path = REPO_ROOT,
    expected_install_path: Path,
) -> Path:
    """Zero-fit interrupted post-rename index recovery with both books/ fsync barriers (spec §3.1 + B2)."""
    import fcntl

    _check_book_id(book_id)
    if not expected_install_path.is_file():
        raise SotaBookError(f"{book_id}: missing expected install descriptor {expected_install_path} (holding release)")
    desc = json.loads(expected_install_path.read_text(encoding="utf-8"))
    if desc.get("book_id") != book_id:
        raise SotaBookError(f"{book_id}: expected install descriptor book_id mismatch ({desc.get('book_id')!r})")

    books_dir = repo_root / "books"
    lock_dir = repo_root / "deriv_data" / "runtime" / "locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_file = lock_dir / "sota_index.lock"
    with open(lock_file, "a+", encoding="utf-8") as lfh:
        fcntl.flock(lfh.fileno(), fcntl.LOCK_EX)
        book_dir = books_dir / book_id
        manifest_obj = _verify_installed_book_dir_strict(
            book_dir,
            book_id=book_id,
            expected_artifacts=list(desc["artifacts"]),
            expected_content_id=str(desc["book_content_id"]),
            expected_det_manifest_sha256=str(desc["deterministic_manifest_sha256"]),
        )
        # B2: Re-fsync books/ AFTER authentication and BEFORE any books/SOTA_INDEX.json replacement!
        fsync_dir(books_dir)
        entry = {
            "id": book_id,
            "content_id": str(manifest_obj["content_id"]),
            "pair": str(manifest_obj["pair"]).upper(),
            "horizon_minutes": int(manifest_obj["horizon_minutes"]),
            "allowed_sides": [str(s).upper() for s in manifest_obj["allowed_sides"]],
            "model_kind": str(manifest_obj["model_kind"]),
            "lifecycle_status": str(manifest_obj["lifecycle_status"]),
            "manifest": f"books/{book_id}/{book_id}.manifest.json",
            "metrics": manifest_obj["metrics"],
        }
        # _publish_sota_index_atomic calls fsync_dir(books_dir) AGAIN after index replacement!
        _publish_sota_index_atomic(books_dir, entry)
        expected_install_path.unlink(missing_ok=True)
        fsync_dir(expected_install_path.parent)
        return book_dir / f"{book_id}.manifest.json"


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
    lgb_model_bytes: tuple[str, bytes] | None = None,
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
    parity_expected_up_cal_probs: np.ndarray | None = None,
    parity_expected_directions: list[str] | np.ndarray | None = None,
    parity_expected_gates: list[bool] | np.ndarray | None = None,
    parity_expected_row_cal_present: list[bool] | np.ndarray | None = None,
    evidence_binding: dict[str, Any] | None = None,
    expected_content_id: str | None = None,
    expected_artifact_digests: list[dict[str, Any]] | None = None,
    expected_install_path: Path | None = None,
    builder_suffix_path: Path | None = None,
    builder_suffix_metadata: dict[str, Any] | None = None,
    archive_sync_stop_at: Path | None = None,
    pre_index_hook: Any | None = None,
    allow_overwrite: bool = False,
    repo_root: Path = REPO_ROOT,
) -> Path:
    import fcntl
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
        if model_kind == "nystroem-lgb-stack":
            if any(
                x is None
                for x in (
                    parity_check_df,
                    parity_check_epochs,
                    parity_expected_probs,
                    parity_expected_up_cal_probs,
                    parity_expected_directions,
                    parity_expected_gates,
                    parity_expected_row_cal_present,
                )
            ):
                raise SotaBookError(f"{book_id}: executable nystroem-lgb-stack requires all 7 parity inputs")
    if lifecycle_status == ACTIVE_LIFECYCLE:
        g6 = gate_certificates.get("G6_holdout_or_prospective")
        if not (isinstance(g6, dict) and g6.get("passed") is True):
            raise SotaBookError(f"{book_id}: active lifecycle requires passed G6_holdout_or_prospective")

    if model_kind in {"logit", "logit-hourly", "nystroem", "nystroem-ens"}:
        if not weights_arrays:
            raise SotaBookError(f"{book_id}: weights_arrays required for {model_kind}")
        _validate_weights_npz(book_id, model_kind, weights_arrays, len(feature_cols))
    elif model_kind == "nystroem-lgb-stack":
        if not weights_arrays:
            raise SotaBookError(f"{book_id}: weights_arrays required for nystroem-lgb-stack")
        _validate_weights_npz(book_id, model_kind, weights_arrays, len(feature_cols))
        if lgb_model_bytes is None and (not lgb_model_files or len(lgb_model_files) != 1):
            raise SotaBookError(f"{book_id}: nystroem-lgb-stack requires exact 1 LightGBM booster file or bytes")
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

    # Stage on the books/ filesystem so reload parity and publication live on the same filesystem!
    books_dir = repo_root / "books"
    books_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".verify_{book_id}_", dir=books_dir) as stage_td:
        stage_root = Path(stage_td)
        stage_books = stage_root / "books"
        stage_dir = stage_books / book_id
        stage_dir.mkdir(parents=True, exist_ok=True)

        st_strat_path = stage_dir / f"{book_id}_strategy.json"
        st_strat_path.write_bytes(serialize_sota_strategy_bytes(strategy))
        st_artifacts: list[Path] = [st_strat_path]
        if model_kind in {"logit", "logit-hourly", "nystroem", "nystroem-ens"}:
            assert weights_arrays is not None
            st_npz = stage_dir / f"{book_id}_weights.npz"
            _save_deterministic_npz(st_npz, weights_arrays)
            st_artifacts.append(st_npz)
        elif model_kind == "nystroem-lgb-stack":
            assert weights_arrays is not None
            st_npz = stage_dir / f"{book_id}_weights.npz"
            _save_deterministic_npz(st_npz, weights_arrays)
            st_artifacts.append(st_npz)
            expected_lgb_name = f"{book_id}_lgb_s0.txt"
            if lgb_model_bytes is not None:
                lgb_name, lgb_raw = lgb_model_bytes
                if lgb_name != expected_lgb_name:
                    raise SotaBookError(f"{book_id}: expected LightGBM filename {expected_lgb_name!r}, got {lgb_name!r}")
                dst_lgb = stage_dir / expected_lgb_name
                dst_lgb.write_bytes(lgb_raw)
            else:
                assert lgb_model_files is not None
                src = lgb_model_files[0]
                if src.name != expected_lgb_name:
                    raise SotaBookError(f"{book_id}: expected LightGBM filename {expected_lgb_name!r}, got {src.name!r}")
                dst_lgb = stage_dir / expected_lgb_name
                shutil.copy2(src, dst_lgb)
            st_artifacts.append(dst_lgb)
        else:
            assert lgb_model_files is not None
            for src in lgb_model_files:
                dst = stage_dir / src.name
                shutil.copy2(src, dst)
                st_artifacts.append(dst)

        content_id = compute_sota_content_id(st_artifacts)
        artifact_records = [
            {"file": p.name, "bytes": p.stat().st_size, "sha256": file_sha256(p)}
            for p in sorted(st_artifacts, key=lambda item: item.name)
        ]
        if expected_content_id is not None and content_id != expected_content_id:
            raise SotaBookError(
                f"{book_id}: computed content_id {content_id!r} != expected {expected_content_id!r}"
            )
        if expected_artifact_digests is not None:
            exp_sorted = sorted(
                [{"file": str(a.get("file", a.get("filename"))), "bytes": int(a["bytes"]), "sha256": str(a["sha256"])} for a in expected_artifact_digests],
                key=lambda x: x["file"],
            )
            if artifact_records != exp_sorted:
                raise SotaBookError(f"{book_id}: staged artifact digests do not match certifier expected digests")

        manifest: dict[str, Any] = {
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
            "artifacts": artifact_records,
            "source": {
                "script": source_script,
                "source_sha256": file_sha256(repo_root / source_script) if (repo_root / source_script).is_file() else None,
                "git_sha": _git_sha(repo_root),
            },
            "env": _env_snapshot(),
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        if evidence_binding is not None:
            manifest["evidence_binding"] = evidence_binding
        st_manifest_path = stage_dir / f"{book_id}.manifest.json"
        st_manifest_path.write_bytes((json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8"))
        index_entry = {
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
        (stage_books / "SOTA_INDEX.json").write_text(
            json.dumps({"schema": SOTA_INDEX_SCHEMA, "books": [index_entry]}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        if lifecycle_status != INACTIVE_LIFECYCLE:
            loaded = load_sota_book(book_id, repo_root=stage_root, allow_demo_canary=True)
            if model_kind == "nystroem-lgb-stack":
                assert parity_check_df is not None and parity_check_epochs is not None
                assert parity_expected_probs is not None and parity_expected_up_cal_probs is not None
                assert parity_expected_directions is not None and parity_expected_gates is not None
                assert parity_expected_row_cal_present is not None
                _verify_7input_stack_parity(
                    book_id,
                    loaded,
                    parity_check_df=parity_check_df,
                    parity_check_epochs=parity_check_epochs,
                    parity_expected_probs=parity_expected_probs,
                    parity_expected_up_cal_probs=parity_expected_up_cal_probs,
                    parity_expected_directions=parity_expected_directions,
                    parity_expected_gates=parity_expected_gates,
                    parity_expected_row_cal_present=parity_expected_row_cal_present,
                )
            elif parity_check_df is not None and parity_expected_probs is not None:
                exp_arr = np.asarray(parity_expected_probs, dtype=np.float64)
                got = loaded.predict_probabilities(parity_check_df, decision_epochs=parity_check_epochs)
                if got.shape != exp_arr.shape:
                    raise SotaBookError(f"{book_id}: parity shape mismatch {got.shape} != {exp_arr.shape}")
                diff = float(np.max(np.abs(got - exp_arr)))
                if not np.isfinite(diff) or diff > 1e-11:
                    raise SotaBookError(f"{book_id}: pure-NumPy reload parity diff {diff:.3e} > 1e-11")

        det_manifest_sha256 = canonical_deterministic_manifest_sha256(manifest)

        # Publish validated artifacts under exclusive repo-scoped lock with RENAME_NOREPLACE + full fsync ordering!
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
                require_det = det_manifest_sha256 if (model_kind == "nystroem-lgb-stack" or expected_install_path is not None) else None
                _verify_installed_book_dir_strict(
                    book_dir,
                    book_id=book_id,
                    expected_artifacts=artifact_records,
                    expected_content_id=content_id,
                    expected_det_manifest_sha256=require_det,
                )
                existing_m = json.loads(manifest_path.read_text(encoding="utf-8"))
                if str(existing_m.get("lifecycle_status") or "") != lifecycle_status:
                    raise SotaBookError(
                        f"{book_id}: refusing to mutate lifecycle_status ({existing_m.get('lifecycle_status')!r} -> {lifecycle_status!r}) "
                        f"without allow_overwrite=True"
                    )
                # Ensure books/ durability barrier before index replacement even on idempotent retry (B2)
                fsync_dir(books_dir)
            else:
                pub_tmp = Path(tempfile.mkdtemp(prefix=f".stage_{book_id}_", dir=books_dir))
                try:
                    for p in sorted(stage_dir.iterdir(), key=lambda x: x.name):
                        dst_f = pub_tmp / p.name
                        shutil.copy2(p, dst_f)
                        # D2: fsync every staged book file before dir fsync / descriptor publication
                        fsync_file(dst_f)
                    # D2: fsync staged book directory before publishing descriptor/suffix or renaming
                    fsync_dir(pub_tmp)

                    # Persist pre-rename expected install descriptor + mandatory 0444 builder_suffix.json BEFORE renameat2!
                    if expected_install_path is not None:
                        exp_payload = (
                            json.dumps(
                                {
                                    "schema": "sota-stack05-expected-install/v1",
                                    "book_id": book_id,
                                    "book_content_id": content_id,
                                    "deterministic_manifest_sha256": det_manifest_sha256,
                                    "artifacts": artifact_records,
                                },
                                indent=2,
                                sort_keys=True,
                                allow_nan=False,
                            )
                            + "\n"
                        ).encode("utf-8")
                        atomic_write_file_bytes(expected_install_path, exp_payload, sync_parent=True)

                    if builder_suffix_path is not None:
                        if builder_suffix_metadata is None:
                            raise SotaBookError(f"{book_id}: builder_suffix_metadata required when builder_suffix_path is set")
                        suffix_obj = dict(builder_suffix_metadata)
                        suffix_obj.update(
                            {
                                "schema": "sota-stack05-builder-suffix/v1",
                                "book_id": book_id,
                                "book_content_id": content_id,
                                "artifacts": [
                                    {"filename": a["file"], "bytes": a["bytes"], "sha256": a["sha256"]}
                                    for a in artifact_records
                                ],
                                "deterministic_manifest_sha256": det_manifest_sha256,
                            }
                        )
                        suffix_bytes = (json.dumps(suffix_obj, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
                        if builder_suffix_path.exists():
                            if builder_suffix_path.read_bytes() != suffix_bytes:
                                raise SotaBookError(f"{book_id}: conflicting existing builder_suffix.json at {builder_suffix_path}")
                        else:
                            atomic_write_file_bytes(
                                builder_suffix_path,
                                suffix_bytes,
                                mode=0o444,
                                noreplace=True,
                                sync_parent=True,
                            )
                        if archive_sync_stop_at is not None:
                            fsync_ancestry(builder_suffix_path.parent, stop_at=archive_sync_stop_at)

                    if book_dir.exists():
                        # allow_overwrite=True path (non-production)
                        old_bak = Path(tempfile.mkdtemp(prefix=f".old_{book_id}_", dir=books_dir))
                        shutil.rmtree(old_bak)
                        os.replace(book_dir, old_bak)
                        try:
                            os.replace(pub_tmp, book_dir)
                            fsync_dir(books_dir)
                        except Exception:
                            if not book_dir.exists() and old_bak.exists():
                                os.replace(old_bak, book_dir)
                            raise
                        finally:
                            if old_bak.exists():
                                shutil.rmtree(old_bak, ignore_errors=True)
                    else:
                        rename_noreplace(pub_tmp, book_dir)
                        if pre_index_hook is not None:
                            pre_index_hook()
                        fsync_dir(books_dir)
                finally:
                    if pub_tmp.exists():
                        shutil.rmtree(pub_tmp, ignore_errors=True)

            # Publish SOTA_INDEX.json + fsync books/ AGAIN, and ONLY THEN unlink expected_install_path + fsync logs/
            _publish_sota_index_atomic(books_dir, index_entry)
            if expected_install_path is not None and expected_install_path.exists():
                expected_install_path.unlink(missing_ok=True)
                fsync_dir(expected_install_path.parent)
            return manifest_path
