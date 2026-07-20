#!/usr/bin/env python3
"""Operate one fixed, credential-free USDCHF 15-minute prospective shadow.

The collector reads the existing public candle and live-tick stores, scores one
inactive candidate beside the exact active comparison book, and records only
outcome-blind decisions.  It has no account, proposal, routing, or purchase
surface.  Protected prices become readable by the reducer only after a
strict-new analysis-access receipt.
"""

from __future__ import annotations

import argparse
from bisect import bisect_right
import fcntl
import hashlib
import json
import math
import os
import re
import signal
import sqlite3
import stat
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import date, datetime, time as wall_time, timedelta, timezone
from pathlib import Path, PurePosixPath
from statistics import NormalDist
from typing import Any, Iterator, Mapping, Sequence
from zoneinfo import ZoneInfo

import lightgbm as lgb
import numpy as np
import pandas as pd

import book_runtime
import evidence_store as evidence
import m15_book_refresh_stats as refresh_stats
import usdchf_m15_current_refit_v1 as current_refit
from live_features import PAIRS as LIVE_FEATURE_PAIRS
from live_features import LiveFeatureBuilder, LiveFeatureError


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = "scripts/usdchf_m15_prospective_v1.py"
SPEC_PATH = "scripts/usdchf_m15_prospective_v1_spec.json"
SERVICE_PATH = "ops/deriv-usdchf-prospective.service"
SPEC_SHA256 = "a30c1bf01968c0d1f73b18b7fa5accc5ca0de9a6346c91f263752d1b197ac114"
PREREG_KIND = "research.usdchf_m15_prospective.prereg/v1"
PREFLIGHT_SCHEMA = "usdchf-m15-prospective-preflight/v1"
TRIAL_METADATA_SCHEMA = "usdchf-m15-prospective-trial-metadata/v1"
DECISION_SCHEMA = "usdchf-m15-prospective-decision/v1"
TRIAL_INVALID_SCHEMA = "usdchf-m15-prospective-trial-invalid/v1"
RESULT_SCHEMA = "usdchf-m15-prospective-result/v1"
VERIFICATION_SCHEMA = "usdchf-m15-prospective-verification/v1"
PAIR = "USDCHF"
HEX64 = re.compile(r"[0-9a-f]{64}")
HEX40 = re.compile(r"[0-9a-f]{40}")
NY = ZoneInfo("America/New_York")
UTC = timezone.utc
SQLITE_BUSY_TIMEOUT_MS = 5_000
PREFLIGHT_REGISTRATION_MAX_AGE_SECONDS = 86_400
FORBIDDEN_ENVIRONMENT = (
    "DERIV_APP_ID",
    "DERIV_PAT",
    "DERIV_ACCOUNT_ID",
    "DERIV_DEMO_ABSOLUTE_BREAKEVEN_CEILING",
    "DERIV_DEMO_ALLOCATION_ARBITRATION_MS",
    "DERIV_DEMO_ALLOCATION_BATCH_SIZE",
    "DERIV_DEMO_CONTRACT_EXPIRY_GRACE_SECONDS",
    "DERIV_DEMO_DISABLED_PAIR_SIDES",
    "DERIV_DEMO_EXECUTOR",
    "DERIV_DEMO_LOG_DIR",
    "DERIV_DEMO_MAX_BREAKEVEN",
    "DERIV_DEMO_MAX_USD_FACTOR_OPEN",
    "DERIV_DEMO_MODE_ARGS",
    "DERIV_DEMO_PAIRS",
    "DERIV_DEMO_PAYOUT_EDGE_MARGIN",
    "DERIV_DEMO_QUOTE_SNAPSHOTS",
    "DERIV_DEMO_RECONCILE_BATCH_SIZE",
    "DERIV_DEMO_RECONCILE_INTERVAL_SECONDS",
    "DERIV_DEMO_SAME_PAIR_MIN_GAP_SECONDS",
    "DERIV_DEMO_STAKE",
    "DERIV_DEMO_STORE_DIR",
    "DERIV_SUPERVISOR_MODE_ARGS",
    "DERIV_ALLOW_EXPERIMENTAL_SHIFTED_DEMO_BUY",
    "DERIV_RUNTIME_LOCK_ROOT",
    "DERIV_OPTIONS_WS_URL",
    "DERIV_LEGACY_WS_URL",
)
BOUND_CODE_PATHS = (
    SCRIPT_PATH,
    SPEC_PATH,
    SERVICE_PATH,
    "ops/deriv-market-stream.service",
    "scripts/deriv_market_stream.py",
    "scripts/deriv_async_client.py",
    "scripts/deriv_backfill.py",
    "scripts/usdchf_m15_current_refit_v1.py",
    "scripts/usdchf_m15_current_refit_v1_spec.json",
    "scripts/book_runtime.py",
    "scripts/live_features.py",
    "scripts/harness.py",
    "scripts/pipeline.py",
    "scripts/deriv_client.py",
    "scripts/m15_book_refresh_stats.py",
    "scripts/deriv_runtime_core.py",
    "scripts/deriv_floor_resolver.py",
    "scripts/min1_production.py",
    "scripts/sessions.py",
    "scripts/evidence_store.py",
    "ENVIRONMENT_libs.txt",
    "requirements-deriv-demo.txt",
)
TRIAL_TABLES = ("trial_metadata", "decisions", "trial_seal", "trial_invalid")
PACKET_TABLES = (*TRIAL_TABLES, "settlement_rows")
TERMINAL_STATUSES = (
    "SHADOW_SURVIVOR_INACTIVE",
    "SHADOW_REJECTED",
    "INCONCLUSIVE",
    "INVALID",
)
RUNTIME_DISTRIBUTIONS = (
    "numpy",
    "pandas",
    "pyarrow",
    "lightgbm",
    "scikit-learn",
    "scipy",
    "websocket-client",
    "websockets",
    "xgboost",
    "catboost",
    "joblib",
)


class ProspectiveError(evidence.EvidenceError):
    """The fixed prospective protocol failed closed."""


class TransientFeatureGap(ProspectiveError):
    """The current store cannot supply a fresh complete row without repair."""


class SealedSourceProtocolError(ProspectiveError):
    """A deterministic source-seal identity check failed after the logical cutoff."""


class RetryableSealInterruption(RuntimeError):
    """A committed logical seal needs the service restart policy to resume it."""


class StoreOnlySentinel:
    """A feature-builder client seam that makes accidental network use fatal."""

    def __getattr__(self, name: str) -> Any:
        raise ProspectiveError(f"store-only feature builder attempted client method {name!r}")


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(value: Any) -> bytes:
    return evidence.canonical_json_bytes(value)


def _exact_fields(value: Any, fields: set[str] | frozenset[str], name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ProspectiveError(f"{name} must be a JSON object")
    actual = set(value)
    if actual != set(fields):
        raise ProspectiveError(
            f"{name} fields differ: missing={sorted(set(fields) - actual)} "
            f"unknown={sorted(actual - set(fields))}"
        )
    return value


def _text(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value.strip() != value
        or any(ord(char) < 0x20 for char in value)
    ):
        raise ProspectiveError(f"{name} must be a non-empty canonical string")
    return value


def _hex64(value: Any, name: str) -> str:
    if not isinstance(value, str) or HEX64.fullmatch(value) is None:
        raise ProspectiveError(f"{name} must be lowercase 64-hex")
    return value


def _canonical_relative(value: Any, name: str) -> str:
    text = _text(value, name)
    if "\\" in text or "\x00" in text:
        raise ProspectiveError(f"{name} must be a canonical POSIX path")
    path = PurePosixPath(text)
    if path.is_absolute() or path.as_posix() != text or any(
        part in {"", ".", ".."} for part in path.parts
    ):
        raise ProspectiveError(f"{name} must be a canonical repo-relative POSIX path")
    return text


def _regular_file(path: Path, name: str) -> os.stat_result:
    if path.is_symlink():
        raise ProspectiveError(f"{name} must not be a symlink: {path}")
    try:
        info = path.stat(follow_symlinks=False)
    except OSError as exc:
        raise ProspectiveError(f"cannot inspect {name}: {path}: {exc}") from exc
    if not stat.S_ISREG(info.st_mode):
        raise ProspectiveError(f"{name} is not a regular file: {path}")
    return info


def _real_directory(path: Path, name: str, *, create: bool = False) -> Path:
    if path.is_symlink():
        raise ProspectiveError(f"{name} must not be a symlink: {path}")
    if create:
        path.mkdir(parents=True, exist_ok=True, mode=0o700)
    if not path.is_dir() or path.is_symlink():
        raise ProspectiveError(f"{name} is not a real directory: {path}")
    return path.resolve(strict=True)


def _ensure_directory_chain(root: Path, relative: PurePosixPath, name: str) -> Path:
    current = root
    for part in relative.parts:
        current = current / part
        if current.exists() and current.is_symlink():
            raise ProspectiveError(f"{name} contains a symlink: {current}")
        current.mkdir(mode=0o700, exist_ok=True)
        if current.is_symlink() or not current.is_dir():
            raise ProspectiveError(f"{name} component is not a real directory: {current}")
    return current


def _validated_root(repo_root: str | os.PathLike[str]) -> Path:
    argument = Path(repo_root)
    if argument.is_symlink():
        raise ProspectiveError("repository root must not be a symlink")
    root = argument.resolve(strict=True)
    if not root.is_dir() or not (root / ".git").exists():
        raise ProspectiveError(f"not a repository root: {root}")
    for relative in ("scripts", "results/json", "evidence/objects"):
        _real_directory(root / relative, relative)
    return root


def _load_spec(root: Path) -> dict[str, Any]:
    path = root / SPEC_PATH
    _regular_file(path, "prospective spec")
    raw = path.read_bytes()
    return _decode_spec(raw, "prospective spec")


def _decode_spec(raw: bytes, name: str) -> dict[str, Any]:
    if _sha256_bytes(raw) != SPEC_SHA256:
        raise ProspectiveError(f"{name} bytes differ from the reviewed contract")
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProspectiveError(f"{name} is not valid UTF-8 JSON: {exc}") from exc
    if not isinstance(value, dict) or value.get("schema") != "usdchf-m15-prospective-spec/v1":
        raise ProspectiveError(f"{name} schema differs")
    if value.get("pair") != PAIR or value.get("horizon_seconds") != 900:
        raise ProspectiveError(f"{name} pair/horizon differs")
    if value.get("activation") is not False or value.get("terminal_statuses") != list(TERMINAL_STATUSES):
        raise ProspectiveError(f"{name} lifecycle contract differs")
    return value


def _load_h0_spec(root: Path, h0: str) -> dict[str, Any]:
    if not isinstance(h0, str) or HEX40.fullmatch(h0) is None:
        raise ProspectiveError("recovery H0 is malformed")
    process = subprocess.run(
        ["git", "cat-file", "blob", f"{h0}:{SPEC_PATH}"],
        cwd=root,
        capture_output=True,
        check=False,
    )
    if process.returncode != 0:
        detail = process.stderr.decode("utf-8", errors="replace").strip()
        raise ProspectiveError(f"cannot load recovery spec from H0: {detail}")
    return _decode_spec(process.stdout, "H0 recovery spec")


def _git(root: Path, *arguments: str, check: bool = True) -> str:
    process = subprocess.run(
        ["git", *arguments], cwd=root, text=True, capture_output=True, check=False
    )
    if check and process.returncode != 0:
        raise ProspectiveError(
            f"git {' '.join(arguments)} failed: {(process.stderr or process.stdout).strip()}"
        )
    return process.stdout.strip()


def _artifact_refs(root: Path, paths: Sequence[str]) -> tuple[evidence.ArtifactRef, ...]:
    return tuple(evidence.ArtifactRef.capture(path, repo_root=root) for path in paths)


def _candidate_package(root: Path, spec: Mapping[str, Any]) -> dict[str, Any]:
    candidate = spec["candidate"]
    try:
        verification = current_refit.verify(candidate["seal_id"], repo_root=root)
    except current_refit.RefitError as exc:
        raise ProspectiveError(f"candidate lineage verification failed: {exc}") from exc
    if (
        verification.get("terminal_id") != candidate["terminal_id"]
        or verification.get("h1_git_sha") != candidate["h1_git_sha"]
        or verification.get("status") != "INACTIVE_UNTESTED_PROSPECTIVE"
    ):
        raise ProspectiveError("candidate verified lineage identity differs")
    source_spec = current_refit._load_spec(root)
    package = current_refit._validate_package(
        root, source_spec, candidate["seal_id"], candidate["h1_git_sha"]
    )
    if package["manifest_sha256"] != candidate["manifest_sha256"]:
        raise ProspectiveError("candidate package-manifest identity differs")
    result_path = root / candidate["result_path"]
    if _sha256_file(result_path) != candidate["result_sha256"]:
        raise ProspectiveError("candidate result identity differs")
    terminal = evidence.verify_object(candidate["terminal_id"], repo_root=root)
    if terminal.kind != source_spec["evidence_kinds"]["terminal"]:
        raise ProspectiveError("candidate terminal evidence kind differs")
    if terminal.payload.get("status") != "INACTIVE_UNTESTED_PROSPECTIVE":
        raise ProspectiveError("candidate is not the inactive untested package")
    return package


def _load_candidate(root: Path, spec: Mapping[str, Any]) -> book_runtime.LoadedBook:
    package = _candidate_package(root, spec)
    strategy = package["strategy"]
    candidate = spec["candidate"]
    directory = Path(package["directory"])
    feature_cols = strategy.get("feature_cols")
    models = strategy.get("models")
    if (
        not isinstance(feature_cols, list)
        or len(feature_cols) != candidate["feature_count"]
        or len(set(feature_cols)) != len(feature_cols)
        or not isinstance(models, list)
        or [row.get("seed") for row in models] != candidate["seed_order"]
        or strategy.get("artifact_id") != candidate["artifact_id"]
        or strategy.get("pair") != PAIR
        or strategy.get("side") != "combined"
        or strategy.get("ensemble") != "arithmetic_mean_in_seed_order"
        or strategy.get("feature_dtype") != candidate["feature_dtype"]
        or strategy.get("confidence_threshold") != candidate["confidence_threshold"]
        or strategy.get("coverage") != 0.01
        or strategy.get("seed_order") != candidate["seed_order"]
        or strategy.get("decision_shift_seconds") != 60
        or strategy.get("horizon_seconds") != 900
        or strategy.get("activation") is not False
    ):
        raise ProspectiveError("candidate feature/model order differs")
    model_paths = [directory / str(row["file"]) for row in models]
    boosters = [lgb.Booster(model_file=str(path)) for path in model_paths]
    if {booster.num_feature() for booster in boosters} != {len(feature_cols)}:
        raise ProspectiveError("candidate model feature counts differ")
    strategy_path = directory / "m15xpny_USDCHF_r202605_strategy.json"
    loaded = book_runtime.LoadedBook(
        pair=PAIR,
        book_id=candidate["artifact_id"],
        book_dir=directory,
        strategy_path=strategy_path,
        model_paths=model_paths,
        strategy=dict(strategy),
        feature_cols=[str(value) for value in feature_cols],
        boosters=boosters,
        conf_thr=float(candidate["confidence_threshold"]),
        content_id=book_runtime._content_id([strategy_path, *model_paths]),
        coverage=0.01,
        coverage_source="sealed_candidate_strategy",
    )
    if loaded.strategy.get("activation") is not False:
        raise ProspectiveError("candidate loading attempted to grant activation")
    return loaded


def _load_incumbent(root: Path, spec: Mapping[str, Any]) -> book_runtime.LoadedBook:
    incumbent = spec["incumbent"]
    for relative, expected in [
        (incumbent["manifest_path"], incumbent["manifest_sha256"]),
        (incumbent["strategy_path"], incumbent["strategy_sha256"]),
        *[(row["path"], row["sha256"]) for row in incumbent["model_order"]],
    ]:
        if _sha256_file(root / relative) != expected:
            raise ProspectiveError(f"incumbent artifact identity differs: {relative}")
    loaded = book_runtime.load_book(PAIR, incumbent["book_id"])
    if (
        loaded.content_id != incumbent["loaded_content_id"]
        or loaded.conf_thr != incumbent["confidence_threshold"]
    ):
        raise ProspectiveError("incumbent runtime identity differs")
    return loaded


def authenticate_policies(
    repo_root: str | os.PathLike[str] = REPO_ROOT,
) -> tuple[book_runtime.LoadedBook, book_runtime.LoadedBook, dict[str, Any]]:
    root = _validated_root(repo_root)
    spec = _load_spec(root)
    candidate = _load_candidate(root, spec)
    incumbent = _load_incumbent(root, spec)
    if candidate.feature_cols != incumbent.feature_cols:
        raise ProspectiveError("candidate and incumbent feature order differs")
    if len(candidate.feature_cols) != 337:
        raise ProspectiveError("common policy feature count differs")
    return candidate, incumbent, spec


def _feature_hash(row: pd.Series, columns: Sequence[str], feature_epoch: int) -> str:
    values = row.loc[list(columns)].to_numpy(dtype="float32", copy=True)
    if values.shape != (337,) or not np.isfinite(values).all():
        raise ProspectiveError("feature row is not an exact finite float32 vector")
    header = _canonical(
        {
            "schema": "usdchf-m15-prospective-feature-vector/v1",
            "timestamp_epoch": int(feature_epoch),
            "dtype": "float32",
            "columns": list(columns),
        }
    )
    digest = hashlib.sha256()
    for component in (header, values.tobytes(order="C")):
        digest.update(len(component).to_bytes(8, "big"))
        digest.update(component)
    return digest.hexdigest()


def _score_payload(book: book_runtime.LoadedBook, row: pd.Series) -> dict[str, Any]:
    values = row.loc[book.feature_cols].to_numpy(dtype="float32", copy=True)
    if values.shape != (337,) or not np.isfinite(values).all():
        raise ProspectiveError("scoring row is not an exact finite float32 vector")
    ordered = pd.Series(values, index=book.feature_cols, dtype="float32")
    scalar = book.score(ordered)
    frame = pd.DataFrame([values], columns=book.feature_cols, dtype="float32")
    batch = book.predict_probabilities(frame)
    components = book.gate_components(frame, batch)
    if (
        batch.shape != (1,)
        or float(batch[0]) != scalar.proba
        or bool(components.gate_passed[0]) != scalar.gate_passed
    ):
        raise ProspectiveError(f"{book.book_id}: scalar/batch score or gate parity failed")
    return {
        "book_id": book.book_id,
        "content_id": book.content_id,
        "probability": scalar.proba,
        "confidence": scalar.confidence,
        "threshold": scalar.threshold,
        "direction": scalar.direction,
        "confidence_gate_passed": bool(components.confidence_passed[0]),
        "structural_gate_passed": bool(components.structural_passed[0]),
        "gate_passed": scalar.gate_passed,
        "gate_reasons": list(scalar.gate_reasons),
    }


def _require_public_environment(environ: Mapping[str, str] | None = None) -> None:
    env = os.environ if environ is None else environ
    present = sorted(name for name in FORBIDDEN_ENVIRONMENT if name in env)
    if present:
        raise ProspectiveError(
            "forbidden credential or execution variables are present by name: "
            + ",".join(present)
        )
    if "MX_HOR" in env and env["MX_HOR"] != "15":
        raise ProspectiveError("ambient MX_HOR must be absent or exactly 15")


def _utc(value: Any, name: str) -> datetime:
    text = _text(value, name)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ProspectiveError(f"{name} is not ISO-8601") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ProspectiveError(f"{name} must be offset-aware")
    return parsed.astimezone(UTC)


def _canonical_utc(ts: datetime) -> str:
    return ts.astimezone(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _parse_session_date(value: str) -> date:
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise ProspectiveError("session date must be YYYY-MM-DD") from exc
    if parsed.weekday() >= 5:
        raise ProspectiveError("preflight session date must be a weekday")
    return parsed


def _latest_prior_weekday(value: date) -> date:
    candidate = value - timedelta(days=1)
    while candidate.weekday() >= 5:
        candidate -= timedelta(days=1)
    return candidate


def _tick_store_paths(root: Path, spec: Mapping[str, Any]) -> tuple[Path, Path]:
    shard_dir = root / spec["paths"]["tick_shards"]
    progress = root / spec["paths"]["tick_progress"]
    _real_directory(shard_dir, "USDCHF live tick shard directory")
    _regular_file(progress, "USDCHF tick progress cursor")
    return shard_dir, progress


def _read_tick_rows(
    root: Path,
    spec: Mapping[str, Any],
    *,
    start_epoch: int | None = None,
    end_epoch: int | None = None,
    required_intervals: Sequence[tuple[int, int]] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Read one append-safe persisted-page snapshot without repair or reordering."""

    shard_dir, progress_path = _tick_store_paths(root, spec)
    progress_before = progress_path.read_bytes()
    try:
        progress = json.loads(progress_before)
    except json.JSONDecodeError as exc:
        raise ProspectiveError("USDCHF tick progress cursor is invalid JSON") from exc
    if (
        not isinstance(progress, dict)
        or not isinstance(progress.get("last_persisted_epoch"), int)
        or not isinstance(progress.get("pages"), int)
        or isinstance(progress.get("pages"), bool)
        or int(progress["pages"]) < 1
    ):
        raise ProspectiveError("USDCHF tick progress cursor is malformed")
    if start_epoch is not None and end_epoch is not None and int(start_epoch) > int(end_epoch):
        raise ProspectiveError("tick read start is after its end")
    intervals = _normalized_tick_intervals(
        required_intervals,
        start_epoch=start_epoch,
        end_epoch=end_epoch,
    )
    interval_starts = tuple(lower for lower, _upper in intervals)
    committed_pages = int(progress["pages"])
    seen = bytearray(committed_pages)
    selected: list[tuple[int, Path, int, int]] = []
    shard_name = re.compile(r"([0-9]+)_([0-9]+)_([0-9]+)\.parquet")
    try:
        entries = os.scandir(shard_dir)
    except OSError as exc:
        raise ProspectiveError(f"cannot enumerate USDCHF tick shards: {exc}") from exc
    with entries:
        for entry in entries:
            if not entry.name.endswith(".parquet"):
                continue
            match = shard_name.fullmatch(entry.name)
            if match is None:
                raise ProspectiveError(f"tick shard name is malformed: {entry.name}")
            page, oldest, newest = (int(value) for value in match.groups())
            if oldest < 0 or newest < oldest:
                raise ProspectiveError(f"tick shard range is malformed: {entry.name}")
            if page >= committed_pages:
                continue
            if seen[page]:
                raise ProspectiveError(f"tick shard page index is duplicated: {page}")
            seen[page] = 1
            if _tick_range_is_needed(
                oldest, newest, intervals, interval_starts=interval_starts
            ):
                selected.append((page, Path(entry.path), oldest, newest))
    if seen.count(1) != committed_pages:
        raise ProspectiveError("committed tick shard inventory has a missing page")
    selected.sort(key=lambda row: row[0])

    chunks: list[pd.DataFrame] = []
    pending: list[pd.DataFrame] = []
    identity_digest = hashlib.sha256()
    for page, path, encoded_oldest, encoded_newest in selected:
        before = path.stat(follow_symlinks=False)
        if path.is_symlink() or not stat.S_ISREG(before.st_mode):
            raise ProspectiveError(f"tick shard is not a regular file: {path}")
        frame = pd.read_parquet(path, columns=["epoch", "quote", "bid", "ask"])
        after = path.stat(follow_symlinks=False)
        if (
            before.st_ino != after.st_ino
            or before.st_size != after.st_size
            or before.st_mtime_ns != after.st_mtime_ns
        ):
            raise ProspectiveError(f"tick shard changed while being read: {path.name}")
        if list(frame.columns) != ["epoch", "quote", "bid", "ask"]:
            raise ProspectiveError(f"tick shard schema differs: {path.name}")
        if not pd.api.types.is_integer_dtype(frame["epoch"].dtype) or pd.api.types.is_bool_dtype(
            frame["epoch"].dtype
        ):
            raise ProspectiveError(f"tick shard clock is not exact integer: {path.name}")
        try:
            epochs = frame["epoch"].to_numpy(dtype="int64", copy=False)
        except (TypeError, ValueError) as exc:
            raise ProspectiveError(f"tick shard clock is not integer: {path.name}") from exc
        if not len(epochs):
            raise ProspectiveError(f"tick shard is empty: {path.name}")
        if int(epochs.min()) != encoded_oldest or int(epochs.max()) != encoded_newest:
            raise ProspectiveError(f"tick shard filename range differs: {path.name}")
        identity = {
            "page": page,
            "file": path.name,
            "bytes": int(after.st_size),
            "mtime_ns": int(after.st_mtime_ns),
            "minimum_epoch": encoded_oldest,
            "maximum_epoch": encoded_newest,
        }
        identity_raw = _canonical(identity)
        identity_digest.update(len(identity_raw).to_bytes(8, "big"))
        identity_digest.update(identity_raw)
        mask = _tick_interval_mask(epochs, intervals)
        frame = frame.loc[mask]
        if len(frame):
            pending.append(frame.reset_index(drop=True))
            if len(pending) == 512:
                chunks.append(pd.concat(pending, ignore_index=True))
                pending.clear()
    if pending:
        chunks.append(pd.concat(pending, ignore_index=True))
    rows = (
        pd.concat(chunks, ignore_index=True)
        if chunks
        else pd.DataFrame(columns=["epoch", "quote", "bid", "ask"])
    )
    if len(rows):
        clock = rows["epoch"].to_numpy(dtype="int64", copy=False)
        admitted_clock = clock[_live_tick_mask(rows)]
        if len(admitted_clock) > 1 and np.any(np.diff(admitted_clock) <= 0):
            raise ProspectiveError(
                "admitted live tick clock is not strict (duplicate or out of order)"
            )
        if len(admitted_clock) and int(admitted_clock[-1]) > int(
            progress["last_persisted_epoch"]
        ):
            raise ProspectiveError("tick shard exceeds persisted progress cursor")
    descriptor = {
        "source": spec["paths"]["tick_shards"],
        "progress_path": spec["paths"]["tick_progress"],
        "progress_sha256": _sha256_bytes(progress_before),
        "last_persisted_epoch": int(progress["last_persisted_epoch"]),
        "shards_seen": len(selected),
        "shard_metadata_sha256": identity_digest.hexdigest(),
    }
    return rows, descriptor


def _normalized_tick_intervals(
    intervals: Sequence[tuple[int, int]] | None,
    *,
    start_epoch: int | None,
    end_epoch: int | None,
) -> list[tuple[int, int]]:
    if intervals is None:
        lower = 0 if start_epoch is None else int(start_epoch)
        upper = (2**63 - 1) if end_epoch is None else int(end_epoch)
        return [(lower, upper)]
    normalized: list[tuple[int, int]] = []
    lower_limit = 0 if start_epoch is None else int(start_epoch)
    upper_limit = (2**63 - 1) if end_epoch is None else int(end_epoch)
    for value in intervals:
        if (
            not isinstance(value, tuple)
            or len(value) != 2
            or any(not isinstance(item, int) or isinstance(item, bool) for item in value)
        ):
            raise ProspectiveError("tick interval must contain two exact integers")
        lower = max(lower_limit, int(value[0]))
        upper = min(upper_limit, int(value[1]))
        if lower <= upper:
            normalized.append((lower, upper))
    normalized.sort()
    merged: list[tuple[int, int]] = []
    for lower, upper in normalized:
        if merged and lower <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], upper))
        else:
            merged.append((lower, upper))
    return merged


def _tick_range_is_needed(
    oldest: int,
    newest: int,
    intervals: Sequence[tuple[int, int]],
    *,
    interval_starts: Sequence[int],
) -> bool:
    if not intervals:
        return False
    if len(interval_starts) != len(intervals):
        raise ProspectiveError("tick interval index differs from its ranges")
    index = bisect_right(interval_starts, int(newest)) - 1
    return index >= 0 and intervals[index][1] >= int(oldest)


def _tick_interval_mask(
    epochs: np.ndarray, intervals: Sequence[tuple[int, int]]
) -> np.ndarray:
    if not intervals:
        return np.zeros(len(epochs), dtype=bool)
    starts = np.asarray([row[0] for row in intervals], dtype="int64")
    ends = np.asarray([row[1] for row in intervals], dtype="int64")
    positions = np.searchsorted(starts, epochs, side="right") - 1
    valid = positions >= 0
    out = np.zeros(len(epochs), dtype=bool)
    out[valid] = epochs[valid] <= ends[positions[valid]]
    return out


def _live_tick_mask(rows: pd.DataFrame) -> np.ndarray:
    if list(rows.columns) != ["epoch", "quote", "bid", "ask"]:
        raise ProspectiveError("tick rows have an unexpected schema")
    try:
        values = rows[["quote", "bid", "ask"]].to_numpy(
            dtype="float64", copy=False
        )
    except (TypeError, ValueError) as exc:
        raise ProspectiveError("tick quote/bid/ask values are not numeric") from exc
    return np.isfinite(values).all(axis=1) & (values > 0).all(axis=1)


def _score_latest_store_row(
    root: Path,
    spec: Mapping[str, Any],
    candidate: book_runtime.LoadedBook,
    incumbent: book_runtime.LoadedBook,
    *,
    stale_seconds: int = 180,
) -> dict[str, Any]:
    candle_store = root / spec["paths"]["candle_store"]
    _real_directory(candle_store, "public candle store")
    for attempt in range(2):
        try:
            identities = _validate_candle_store_snapshot(candle_store)
            builder = LiveFeatureBuilder(
                StoreOnlySentinel(),
                store_dir=candle_store,
                stale_seconds=stale_seconds,
            )
            try:
                row = builder.feature_row(PAIR, candidate.feature_cols)
            except LiveFeatureError as exc:
                if _is_transient_feature_gap(exc):
                    raise TransientFeatureGap(str(exc)) from exc
                raise ProspectiveError(f"malformed live feature source: {exc}") from exc
            _assert_candle_store_unchanged(candle_store, identities)
            break
        except TransientFeatureGap as exc:
            if attempt == 0 and "changed while" in str(exc):
                continue
            raise
    else:  # pragma: no cover - the loop either breaks or raises
        raise TransientFeatureGap("rolling store did not yield a stable snapshot")
    timestamp = pd.Timestamp(row.timestamp)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    timestamp = timestamp.tz_convert("UTC")
    feature_epoch = int(timestamp.timestamp())
    decision_epoch = feature_epoch + int(spec["runtime"]["decision_shift_seconds"])
    if feature_epoch % 60 or decision_epoch % 60:
        raise ProspectiveError("feature/decision clock is not a completed-minute grid")
    if not bool(refresh_stats.canonical_runtime_ny_mask([decision_epoch])[0]):
        raise ProspectiveError("preflight decision is outside the pinned NY decision session")
    return {
        "feature_timestamp_utc": _canonical_utc(
            datetime.fromtimestamp(feature_epoch, UTC)
        ),
        "decision_close_utc": _canonical_utc(
            datetime.fromtimestamp(decision_epoch, UTC)
        ),
        "decision_close_epoch": decision_epoch,
        "feature_sha256": _feature_hash(row.row, candidate.feature_cols, feature_epoch),
        "feature_dtype": "float32",
        "feature_count": len(candidate.feature_cols),
        "row_source": row.source,
        "warnings": list(row.warnings),
        "candidate": _score_payload(candidate, row.row),
        "incumbent": _score_payload(incumbent, row.row),
    }


def _candle_store_identity(path: Path) -> tuple[int, int, int, int]:
    info = _regular_file(path, "rolling candle store")
    return (int(info.st_dev), int(info.st_ino), int(info.st_size), int(info.st_mtime_ns))


def _validate_candle_store_snapshot(
    candle_store: Path,
) -> dict[str, tuple[int, int, int, int]]:
    """Reject source repairs that LiveFeatureBuilder otherwise performs internally."""

    identities: dict[str, tuple[int, int, int, int]] = {}
    for pair in LIVE_FEATURE_PAIRS:
        path = candle_store / f"{pair}.parquet"
        if not path.exists():
            raise TransientFeatureGap(f"{pair}: rolling store file missing: {path}")
        before = _candle_store_identity(path)
        try:
            frame = pd.read_parquet(path)
        except Exception as exc:
            raise ProspectiveError(f"{pair}: rolling store cannot be read: {exc}") from exc
        after = _candle_store_identity(path)
        if before != after:
            raise TransientFeatureGap(f"{pair}: rolling store changed while validated")
        required = {"timestamp", "open", "high", "low", "close"}
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ProspectiveError(f"{pair}: rolling store missing columns {missing}")
        try:
            timestamps = pd.DatetimeIndex(pd.to_datetime(frame["timestamp"], utc=True))
        except (TypeError, ValueError) as exc:
            raise ProspectiveError(f"{pair}: rolling store timestamp is malformed") from exc
        if (
            timestamps.hasnans
            or not timestamps.is_monotonic_increasing
            or timestamps.has_duplicates
            or np.any(timestamps.asi8 % 60_000_000_000 != 0)
        ):
            raise ProspectiveError(
                f"{pair}: rolling store clock is not a strict completed-minute grid"
            )
        try:
            prices = frame[["open", "high", "low", "close"]].to_numpy(
                dtype="float64", copy=False
            )
        except (TypeError, ValueError) as exc:
            raise ProspectiveError(f"{pair}: rolling store prices are not numeric") from exc
        if not np.isfinite(prices).all() or np.any(prices <= 0):
            raise ProspectiveError(f"{pair}: rolling store prices are not finite and positive")
        identities[pair] = after
    return identities


def _assert_candle_store_unchanged(
    candle_store: Path, identities: Mapping[str, tuple[int, int, int, int]]
) -> None:
    for pair, expected in identities.items():
        if _candle_store_identity(candle_store / f"{pair}.parquet") != expected:
            raise TransientFeatureGap(f"{pair}: rolling store changed while scoring")


def _is_transient_feature_gap(error: LiveFeatureError) -> bool:
    message = str(error)
    transient_fragments = (
        "rolling store file missing:",
        "rolling store has too few completed rows:",
        "no complete live feature row for requested schema",
        "latest feature row is stale:",
    )
    if any(fragment in message for fragment in transient_fragments):
        return True
    return "cannot build requested feature row; missing=[] nan=[" in message


def build_preflight(
    session_date: str,
    *,
    repo_root: str | os.PathLike[str] = REPO_ROOT,
    now: datetime | None = None,
) -> dict[str, Any]:
    _require_public_environment()
    root = _validated_root(repo_root)
    candidate, incumbent, spec = authenticate_policies(root)
    selected_date = _parse_session_date(session_date)
    observation_start_utc = (now or datetime.now(UTC)).astimezone(UTC)
    if selected_date != _latest_prior_weekday(
        observation_start_utc.astimezone(NY).date()
    ):
        raise ProspectiveError(
            "preflight requires the newest prior New York weekday"
        )
    start_ny = datetime.combine(selected_date, wall_time(8, 0), tzinfo=NY)
    required_end_ny = datetime.combine(selected_date, wall_time(16, 49, 1), tzinfo=NY)
    if required_end_ny.astimezone(UTC) >= observation_start_utc:
        raise ProspectiveError("preflight provenance session is not fully spent")
    start_epoch = int(start_ny.astimezone(UTC).timestamp())
    end_epoch = int(
        datetime.combine(selected_date, wall_time(16, 50), tzinfo=NY)
        .astimezone(UTC)
        .timestamp()
    ) - 1
    ticks, descriptor = _read_tick_rows(
        root, spec, start_epoch=start_epoch, end_epoch=end_epoch
    )
    if ticks.empty:
        raise ProspectiveError("preflight session has no persisted USDCHF ticks")
    clock = ticks["epoch"].to_numpy(dtype="int64", copy=False)
    live = _live_tick_mask(ticks)
    finite_epochs = clock[live]
    if (
        not len(finite_epochs)
        or int(finite_epochs[0]) > start_epoch + 10
        or int(finite_epochs[-1]) < int(required_end_ny.astimezone(UTC).timestamp())
    ):
        raise ProspectiveError("preflight session lacks full live bid/ask span")
    expected_seconds = end_epoch - start_epoch + 1
    covered_seconds = len(np.unique(finite_epochs))
    coverage = covered_seconds / expected_seconds
    if coverage < 0.995:
        raise ProspectiveError(
            f"preflight live finite-bid/ask coverage {coverage:.6f} is below 0.995"
        )
    scoring = _score_latest_store_row(root, spec, candidate, incumbent)
    generated_utc = (now or datetime.now(UTC)).astimezone(UTC)
    payload = {
        "schema": PREFLIGHT_SCHEMA,
        "pair": PAIR,
        "host": os.uname().nodename,
        "spent_session_date_ny": selected_date.isoformat(),
        "generated_utc": _canonical_utc(generated_utc),
        "provider": {
            "endpoint_class": "public_Deriv_market_stream_store",
            "candle_store": spec["paths"]["candle_store"],
            "tick_store": spec["paths"]["tick_shards"],
            "tick_descriptor": descriptor,
        },
        "tick_provenance": {
            "session_start_epoch": start_epoch,
            "session_end_epoch": end_epoch,
            "rows": len(ticks),
            "admitted_live_rows": int(live.sum()),
            "recovered_or_malformed_rows_rejected": int((~live).sum()),
            "finite_second_coverage": coverage,
            "admission_rule": "finite_positive_quote_bid_ask",
        },
        "scoring_parity": scoring,
        "outcome_fields_present": False,
        "activation": False,
    }
    return _validate_preflight_value(payload, spec)


def _atomic_new(path: Path, raw: bytes, *, mode: int = 0o600) -> None:
    parent = _real_directory(path.parent, f"parent of {path}", create=True)
    stage = parent / f".{path.name}.building"
    directory = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        try:
            fcntl.flock(directory, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ProspectiveError(f"another strict-new writer owns {parent}") from exc
        if path.exists() or path.is_symlink():
            if path.is_symlink() or not stage.exists() or stage.is_symlink():
                raise ProspectiveError(f"strict-new destination already exists: {path}")
            destination_info = _regular_file(path, "strict-new recovery destination")
            stage_info = _regular_file(stage, "strict-new recovery stage")
            if (
                destination_info.st_dev != stage_info.st_dev
                or destination_info.st_ino != stage_info.st_ino
                or path.read_bytes() != raw
            ):
                raise ProspectiveError("strict-new recovery link or bytes differ")
            stage.unlink()
            os.fsync(directory)
            recovered = _regular_file(path, "strict-new recovered destination")
            if recovered.st_nlink != 1:
                raise ProspectiveError("strict-new recovered destination has foreign hard links")
            return
        if stage.exists() or stage.is_symlink():
            stale = _regular_file(stage, "strict-new stale stage")
            if stale.st_nlink != 1:
                raise ProspectiveError("strict-new stale stage has foreign hard links")
            stage.unlink()
            os.fsync(directory)
        descriptor = os.open(
            stage,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0),
            mode,
        )
        try:
            view = memoryview(raw)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise ProspectiveError(f"short write: {stage}")
                view = view[written:]
            os.fchmod(descriptor, mode)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        try:
            os.link(stage, path, follow_symlinks=False)
        except FileExistsError as exc:
            raise ProspectiveError(f"strict-new destination already exists: {path}") from exc
        os.fsync(directory)
        stage.unlink()
        os.fsync(directory)
    finally:
        try:
            fcntl.flock(directory, fcntl.LOCK_UN)
        finally:
            os.close(directory)


def write_preflight(
    session_date: str,
    output: str,
    *,
    repo_root: str | os.PathLike[str] = REPO_ROOT,
) -> dict[str, Any]:
    root = _validated_root(repo_root)
    relative = _canonical_relative(output, "preflight output")
    spec = _load_spec(root)
    expected = spec["paths"]["preflight_result"]
    if relative != expected:
        raise ProspectiveError(f"preflight output must be the fixed path {expected}")
    destination = root / relative
    stage = destination.parent / f".{destination.name}.building"
    if (destination.exists() or destination.is_symlink()) and (
        stage.exists() or stage.is_symlink()
    ):
        _regular_file(destination, "preflight recovery destination")
        try:
            recovered = evidence.decode_canonical_json(destination.read_bytes())
        except evidence.EvidenceError as exc:
            raise ProspectiveError(
                f"preflight recovery destination is not canonical JSON: {exc}"
            ) from exc
        recovered = _validate_preflight_value(recovered, spec)
        if recovered["spent_session_date_ny"] != _parse_session_date(
            session_date
        ).isoformat():
            raise ProspectiveError("preflight recovery session date differs")
        _atomic_new(destination, _canonical(recovered))
        return recovered
    payload = _validate_preflight_value(
        build_preflight(session_date, repo_root=root), spec
    )
    _ensure_directory_chain(
        root, PurePosixPath(relative).parent, "preflight output path"
    )
    _atomic_new(destination, _canonical(payload))
    return payload


def _validate_preflight_value(
    value: Any, spec: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    fields = {
        "schema",
        "pair",
        "host",
        "spent_session_date_ny",
        "generated_utc",
        "provider",
        "tick_provenance",
        "scoring_parity",
        "outcome_fields_present",
        "activation",
    }
    _exact_fields(value, fields, "preflight receipt")
    if (
        value["schema"] != PREFLIGHT_SCHEMA
        or value["pair"] != PAIR
        or value["host"] != _text(value["host"], "preflight host")
        or value["outcome_fields_present"] is not False
        or value["activation"] is not False
    ):
        raise ProspectiveError("preflight lifecycle boundary differs")
    selected_date = _parse_session_date(value["spent_session_date_ny"])
    generated = _utc(value["generated_utc"], "preflight generated_utc")
    required_end = datetime.combine(selected_date, wall_time(16, 49, 1), tzinfo=NY)
    if generated <= required_end.astimezone(UTC):
        raise ProspectiveError("preflight provenance session was not spent when generated")
    if selected_date != _latest_prior_weekday(generated.astimezone(NY).date()):
        raise ProspectiveError(
            "preflight requires the newest prior New York weekday"
        )
    paths = (spec or {}).get("paths", {})
    candle_store = paths.get("candle_store", "deriv_data/candles_1m_daemon")
    tick_shards = paths.get("tick_shards", "deriv_data/ticks_1s/_pages/USDCHF")
    tick_progress = paths.get("tick_progress", "deriv_data/ticks_1s/USDCHF_progress.json")
    provider = _exact_fields(
        value["provider"],
        {"endpoint_class", "candle_store", "tick_store", "tick_descriptor"},
        "preflight provider",
    )
    if (
        provider["endpoint_class"] != "public_Deriv_market_stream_store"
        or provider["candle_store"] != candle_store
        or provider["tick_store"] != tick_shards
    ):
        raise ProspectiveError("preflight provider identity differs")
    descriptor = _exact_fields(
        provider["tick_descriptor"],
        {
            "source",
            "progress_path",
            "progress_sha256",
            "last_persisted_epoch",
            "shards_seen",
            "shard_metadata_sha256",
        },
        "preflight tick descriptor",
    )
    if (
        descriptor["source"] != tick_shards
        or descriptor["progress_path"] != tick_progress
        or not isinstance(descriptor["last_persisted_epoch"], int)
        or not isinstance(descriptor["shards_seen"], int)
        or descriptor["shards_seen"] < 1
    ):
        raise ProspectiveError("preflight tick descriptor differs")
    _hex64(descriptor["progress_sha256"], "preflight progress sha256")
    _hex64(descriptor["shard_metadata_sha256"], "preflight shard metadata sha256")
    provenance = value.get("tick_provenance")
    provenance = _exact_fields(
        provenance,
        {
            "session_start_epoch",
            "session_end_epoch",
            "rows",
            "admitted_live_rows",
            "recovered_or_malformed_rows_rejected",
            "finite_second_coverage",
            "admission_rule",
        },
        "preflight tick provenance",
    )
    session_start = int(
        datetime.combine(selected_date, wall_time(8, 0), tzinfo=NY)
        .astimezone(UTC)
        .timestamp()
    )
    session_end = int(
        datetime.combine(selected_date, wall_time(16, 50), tzinfo=NY)
        .astimezone(UTC)
        .timestamp()
    ) - 1
    if (
        provenance["admission_rule"] != "finite_positive_quote_bid_ask"
        or provenance["session_start_epoch"] != session_start
        or provenance["session_end_epoch"] != session_end
        or any(
            not isinstance(provenance[name], int) or provenance[name] < 0
            for name in ("rows", "admitted_live_rows", "recovered_or_malformed_rows_rejected")
        )
        or provenance["recovered_or_malformed_rows_rejected"] < 1
        or provenance["rows"]
        != provenance["admitted_live_rows"]
        + provenance["recovered_or_malformed_rows_rejected"]
        or not isinstance(provenance["finite_second_coverage"], (int, float))
        or not math.isfinite(float(provenance["finite_second_coverage"]))
        or float(provenance["finite_second_coverage"]) < 0.995
        or not math.isclose(
            float(provenance["finite_second_coverage"]),
            provenance["admitted_live_rows"] / (session_end - session_start + 1),
            rel_tol=0.0,
            abs_tol=1e-15,
        )
    ):
        raise ProspectiveError("preflight tick-provenance gate differs")
    scoring = _exact_fields(
        value["scoring_parity"],
        {
            "feature_timestamp_utc",
            "decision_close_utc",
            "decision_close_epoch",
            "feature_sha256",
            "feature_dtype",
            "feature_count",
            "row_source",
            "warnings",
            "candidate",
            "incumbent",
        },
        "preflight scoring parity",
    )
    feature_time = _utc(scoring["feature_timestamp_utc"], "preflight feature timestamp")
    decision_time = _utc(scoring["decision_close_utc"], "preflight decision timestamp")
    decision_epoch = scoring["decision_close_epoch"]
    feature_age_seconds = (generated - feature_time).total_seconds()
    if (
        not isinstance(decision_epoch, int)
        or int(feature_time.timestamp()) + 60 != decision_epoch
        or int(decision_time.timestamp()) != decision_epoch
        or decision_time > generated
        or not 60 <= feature_age_seconds <= 180
        or decision_epoch % 60
        or not bool(refresh_stats.canonical_runtime_ny_mask([decision_epoch])[0])
        or scoring["feature_count"] != 337
        or scoring["feature_dtype"] != "float32"
        or not isinstance(scoring["warnings"], list)
        or any(not isinstance(row, str) for row in scoring["warnings"])
    ):
        raise ProspectiveError("preflight score clock or feature identity differs")
    _hex64(scoring["feature_sha256"], "preflight feature sha256")
    _text(scoring["row_source"], "preflight row source")
    candidate_id = (spec or {}).get("candidate", {}).get(
        "artifact_id", "USDCHF.m15ny_xpair_seedens.r202605.v1"
    )
    incumbent_id = (spec or {}).get("incumbent", {}).get(
        "book_id", "USDCHF.m15ny_xpair_seedens.v1"
    )
    thresholds = {
        "candidate": float(
            (spec or {}).get("candidate", {}).get(
                "confidence_threshold", 0.04310172144482041
            )
        ),
        "incumbent": float(
            (spec or {}).get("incumbent", {}).get(
                "confidence_threshold", 0.16933929389708485
            )
        ),
    }
    for policy, book_id in (("candidate", candidate_id), ("incumbent", incumbent_id)):
        score = _exact_fields(
            scoring[policy],
            {
                "book_id",
                "content_id",
                "probability",
                "confidence",
                "threshold",
                "direction",
                "confidence_gate_passed",
                "structural_gate_passed",
                "gate_passed",
                "gate_reasons",
            },
            f"preflight {policy} score",
        )
        probability = score["probability"]
        confidence = score["confidence"]
        threshold = score["threshold"]
        if (
            score["book_id"] != book_id
            or not isinstance(score["content_id"], str)
            or not score["content_id"]
            or (
                policy == "incumbent"
                and spec is not None
                and score["content_id"] != spec["incumbent"]["loaded_content_id"]
            )
            or not isinstance(probability, (int, float))
            or not 0 <= float(probability) <= 1
            or not isinstance(confidence, (int, float))
            or float(confidence) != abs(float(probability) - 0.5)
            or threshold != thresholds[policy]
            or score["direction"] != ("UP" if float(probability) >= 0.5 else "DOWN")
            or type(score["confidence_gate_passed"]) is not bool
            or score["confidence_gate_passed"] is not (
                float(confidence) >= float(threshold)
            )
            or type(score["structural_gate_passed"]) is not bool
            or type(score["gate_passed"]) is not bool
            or score["gate_passed"] is not (
                bool(score["confidence_gate_passed"])
                and bool(score["structural_gate_passed"])
            )
            or not isinstance(score["gate_reasons"], list)
            or any(not isinstance(row, str) for row in score["gate_reasons"])
        ):
            raise ProspectiveError(f"preflight {policy} score identity differs")
    return value


def _load_preflight(root: Path, relative: str) -> dict[str, Any]:
    path_text = _canonical_relative(relative, "preflight path")
    path = root / path_text
    _regular_file(path, "preflight receipt")
    raw = path.read_bytes()
    try:
        value = evidence.decode_canonical_json(raw)
    except evidence.EvidenceError as exc:
        raise ProspectiveError(f"preflight receipt is not canonical JSON: {exc}") from exc
    return _validate_preflight_value(value, _load_spec(root))


def _h0_state(root: Path) -> tuple[str, set[str]]:
    if _git(root, "branch", "--show-current") != "main":
        raise ProspectiveError("preregistration requires branch main")
    head = _git(root, "rev-parse", "HEAD")
    if head != _git(root, "rev-parse", "origin/main") or HEX40.fullmatch(head) is None:
        raise ProspectiveError("preregistration requires HEAD == origin/main")
    changed: set[str] = set()
    for line in _git(root, "status", "--porcelain=v1", "--untracked-files=all").splitlines():
        if not line:
            continue
        path = line[3:]
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        changed.add(path)
    if any(re.fullmatch(r"evidence/objects/[0-9a-f]{64}\.json", path) is None for path in changed):
        raise ProspectiveError("preregistration found foreign worktree state")
    return head, changed


def _verified_runtime_environment(root: Path, h0: str) -> dict[str, Any]:
    import importlib.metadata
    import site

    # The repository-wide authority wins for duplicate names; the Deriv
    # runtime authority supplies provider/statistics-only pins such as SciPy.
    authority_paths = ("ENVIRONMENT_libs.txt", "requirements-deriv-demo.txt")
    authorities: dict[str, bytes] = {}
    try:
        authorities = {
            path: current_refit._git_blob_bytes(root, h0, path)
            for path in authority_paths
        }
    except current_refit.RefitError as exc:
        raise ProspectiveError(f"cannot load H0 runtime authority: {exc}") from exc
    pins: dict[str, str] = {}
    for path, authority in authorities.items():
        try:
            lines = authority.decode("utf-8", errors="strict").splitlines()
        except UnicodeDecodeError as exc:
            raise ProspectiveError(f"H0 runtime authority is not UTF-8: {path}") from exc
        for line in lines:
            value = line.split("#", 1)[0].strip()
            if not value or "==" not in value:
                continue
            name, version = value.split("==", 1)
            normalized = name.strip().lower()
            exact = version.strip()
            if not normalized or not exact:
                raise ProspectiveError("H0 runtime authority contains malformed pins")
            if normalized in pins:
                continue
            pins[normalized] = exact
    try:
        expected = {name: pins[name] for name in RUNTIME_DISTRIBUTIONS}
        installed = {
            name: importlib.metadata.version(name) for name in RUNTIME_DISTRIBUTIONS
        }
    except KeyError as exc:
        raise ProspectiveError(f"H0 runtime authority lacks exact pin {exc.args[0]}") from exc
    except importlib.metadata.PackageNotFoundError as exc:
        raise ProspectiveError(f"required runtime distribution is absent: {exc}") from exc
    expected_prefix = Path.home() / "binary-algo-venv"
    expected_executable = expected_prefix / "bin/python"
    value = {
        "authorities": {
            path: _sha256_bytes(authority) for path, authority in authorities.items()
        },
        "python": sys.version.split()[0],
        "executable": str(Path(sys.executable).absolute()),
        "prefix": str(Path(sys.prefix).absolute()),
        "base_prefix": str(Path(sys.base_prefix).absolute()),
        "user_site_enabled": site.ENABLE_USER_SITE,
        "distributions": installed,
    }
    if (
        Path(sys.executable).absolute() != expected_executable
        or Path(sys.prefix).absolute() != expected_prefix
        or site.ENABLE_USER_SITE is not False
        or installed != expected
    ):
        raise ProspectiveError(
            f"runtime differs from the isolated H0 authority: {value}"
        )
    return value


def _candidate_paths(root: Path, spec: Mapping[str, Any]) -> list[str]:
    package = PurePosixPath(spec["candidate"]["package_path"])
    source_spec_path = spec["candidate"]["source_spec_path"]
    source_spec = json.loads((root / source_spec_path).read_text())
    return [
        spec["candidate"]["result_path"],
        *[(package / name).as_posix() for name in source_spec["package_files"]],
        f"evidence/objects/{spec['candidate']['seal_id']}.json",
        f"evidence/objects/{spec['candidate']['terminal_id']}.json",
    ]


def _bound_paths(root: Path, spec: Mapping[str, Any]) -> tuple[str, ...]:
    incumbent = spec["incumbent"]
    paths = [
        *BOUND_CODE_PATHS,
        *_candidate_paths(root, spec),
        incumbent["index_path"],
        incumbent["manifest_path"],
        incumbent["strategy_path"],
        *[row["path"] for row in incumbent["model_order"]],
    ]
    if len(paths) != len(set(paths)):
        raise ProspectiveError("preregistration bound paths contain duplicates")
    return tuple(paths)


def _validate_t0(t0_utc: str, t0_ny: str, *, now: datetime | None = None) -> dict[str, Any]:
    utc_value = _utc(t0_utc, "t0_utc")
    try:
        local = datetime.fromisoformat(_text(t0_ny, "t0_ny"))
    except ValueError as exc:
        raise ProspectiveError("t0_ny is not ISO-8601") from exc
    if local.tzinfo is None or local.utcoffset() is None:
        raise ProspectiveError("t0_ny must include the applicable UTC offset")
    local_ny = local.astimezone(NY)
    if (
        local.replace(tzinfo=None) != local_ny.replace(tzinfo=None)
        or local.utcoffset() != local_ny.utcoffset()
        or local_ny.weekday() != 0
        or (
        local_ny.hour,
        local_ny.minute,
        local_ny.second,
        local_ny.microsecond,
        ) != (8, 0, 0, 0)
    ):
        raise ProspectiveError("T0 must be exactly Monday 08:00:00 America/New_York")
    if local_ny.astimezone(UTC) != utc_value or utc_value.microsecond:
        raise ProspectiveError("T0 UTC and New York representations differ")
    if utc_value <= (now or datetime.now(UTC)).astimezone(UTC):
        raise ProspectiveError("T0 must be strictly in the future")
    return {
        "utc": _canonical_utc(utc_value),
        "ny": local_ny.isoformat(timespec="seconds"),
        "epoch": int(utc_value.timestamp()),
    }


def _validate_registered_t0(value: Any) -> dict[str, Any]:
    t0 = _exact_fields(value, {"utc", "ny", "epoch"}, "registered T0")
    utc_value = _utc(t0["utc"], "registered T0 UTC")
    try:
        local = datetime.fromisoformat(_text(t0["ny"], "registered T0 New York"))
    except ValueError as exc:
        raise ProspectiveError("registered T0 New York value is not ISO-8601") from exc
    if local.tzinfo is None or local.utcoffset() is None:
        raise ProspectiveError("registered T0 New York value must include its offset")
    local_ny = local.astimezone(NY)
    if (
        local.replace(tzinfo=None) != local_ny.replace(tzinfo=None)
        or local.utcoffset() != local_ny.utcoffset()
        or local_ny.weekday() != 0
        or (local_ny.hour, local_ny.minute, local_ny.second, local_ny.microsecond)
        != (8, 0, 0, 0)
        or local_ny.astimezone(UTC) != utc_value
        or utc_value.microsecond
        or not isinstance(t0["epoch"], int)
        or t0["epoch"] != int(utc_value.timestamp())
        or t0["utc"] != _canonical_utc(utc_value)
        or t0["ny"] != local_ny.isoformat(timespec="seconds")
    ):
        raise ProspectiveError("registered T0 identity differs")
    return t0


def _prereg_payload(
    spec: Mapping[str, Any],
    *,
    h0: str,
    t0: Mapping[str, Any],
    preflight_path: str,
    preflight_payload: Mapping[str, Any],
    runtime_environment: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": spec["evidence_kinds"]["prereg"],
        "issue": spec["issue"],
        "h0_git_sha": h0,
        "pair": PAIR,
        "horizon_seconds": spec["horizon_seconds"],
        "candidate_id": spec["candidate"]["artifact_id"],
        "incumbent_id": spec["incumbent"]["book_id"],
        "t0": dict(t0),
        "preflight": {
            "path": preflight_path,
            "sha256": _sha256_bytes(_canonical(dict(preflight_payload))),
            "payload": dict(preflight_payload),
        },
        "runtime_environment": dict(runtime_environment),
        "source_paths": {
            "candles": spec["paths"]["candle_store"],
            "tick_shards": spec["paths"]["tick_shards"],
            "tick_progress": spec["paths"]["tick_progress"],
        },
        "trial_path_template": spec["paths"]["trial_template"],
        "decision_contract": {
            "session": spec["runtime"],
            "fixed_look": spec["fixed_look"],
            "statistics": spec["statistics"],
            "common_complete_feature_grid": True,
            "missed_decision_backfill": False,
            "ties_lose": True,
        },
        "evidence_class": "OUTCOME_ONLY_TICK_PROXY",
        "money": None,
        "activation": False,
    }


def _collection_payload(
    spec: Mapping[str, Any], prereg: evidence.EvidenceEnvelope
) -> dict[str, Any]:
    return {
        "schema": spec["evidence_kinds"]["collection_access_receipt"],
        "prereg_id": prereg.object_id,
        "authorized_actor": "usdchf_m15_prospective_v1.collector_and_mechanical_supplier",
        "authorized_sources": [
            spec["paths"]["candle_store"],
            spec["paths"]["tick_shards"],
            spec["paths"]["tick_progress"],
        ],
        "allowed_observations": "features_scores_gates_clocks_finiteness_and_fixed_entry_exit_tick_rows",
        "forbidden_outputs": [
            "return",
            "correctness",
            "accuracy",
            "endpoint",
            "confidence_bound",
            "money",
        ],
        "activation": False,
    }


def preregister(
    *,
    t0_utc: str,
    t0_ny: str,
    preflight_path: str,
    user_approved_t0: bool,
    repo_root: str | os.PathLike[str] = REPO_ROOT,
    now: datetime | None = None,
) -> tuple[evidence.EvidenceEnvelope, evidence.EvidenceEnvelope]:
    _require_public_environment()
    if user_approved_t0 is not True:
        raise ProspectiveError("preregistration requires explicit user-approved T0")
    root = _validated_root(repo_root)
    spec = _load_spec(root)
    expected_preflight = spec["paths"]["preflight_result"]
    relative = _canonical_relative(preflight_path, "preflight path")
    if relative != expected_preflight:
        raise ProspectiveError(f"preflight path must be {expected_preflight}")
    preflight_payload = _load_preflight(root, relative)
    registration_now = (now or datetime.now(UTC)).astimezone(UTC)
    generated = _utc(preflight_payload["generated_utc"], "preflight generated_utc")
    preflight_age = (registration_now - generated).total_seconds()
    if not 0 <= preflight_age <= PREFLIGHT_REGISTRATION_MAX_AGE_SECONDS:
        raise ProspectiveError(
            "preflight must be current within 24 hours at preregistration"
        )
    h0, preliminary = _h0_state(root)
    runtime_environment = _verified_runtime_environment(root, h0)
    if not preliminary:
        candidate, incumbent, authenticated_spec = authenticate_policies(root)
        scoring = preflight_payload["scoring_parity"]
        if (
            authenticated_spec != spec
            or scoring["candidate"]["content_id"] != candidate.content_id
            or scoring["incumbent"]["content_id"] != incumbent.content_id
        ):
            raise ProspectiveError("preflight policy bytes differ at preregistration")
    t0 = _validate_t0(t0_utc, t0_ny, now=registration_now)
    if _utc(preflight_payload["generated_utc"], "preflight generated_utc") >= _utc(
        t0["utc"], "registered T0 UTC"
    ):
        raise ProspectiveError("preflight must be generated before the registered T0")
    artifacts = _artifact_refs(root, _bound_paths(root, spec))
    prereg = evidence.EvidenceEnvelope.create(
        kind=spec["evidence_kinds"]["prereg"],
        payload=_prereg_payload(
            spec,
            h0=h0,
            t0=t0,
            preflight_path=relative,
            preflight_payload=preflight_payload,
            runtime_environment=runtime_environment,
        ),
        artifacts=artifacts,
    )
    collection = evidence.EvidenceEnvelope.create(
        kind=spec["evidence_kinds"]["collection_access_receipt"],
        payload=_collection_payload(spec, prereg),
        dependencies=[prereg.object_id],
    )
    allowed = {
        f"evidence/objects/{prereg.object_id}.json",
        f"evidence/objects/{collection.object_id}.json",
    }
    if not preliminary.issubset(allowed):
        raise ProspectiveError("preregistration recovery found foreign evidence state")
    evidence.publish(prereg, repo_root=root)
    evidence.publish(collection, repo_root=root)
    return prereg, collection


def _domain_objects(root: Path, spec: Mapping[str, Any]) -> dict[str, list[evidence.EvidenceEnvelope]]:
    kinds = set(spec["evidence_kinds"].values())
    groups = {kind: [] for kind in kinds}
    for envelope in evidence.inspect_store_metadata(repo_root=root):
        if envelope.kind in groups:
            groups[envelope.kind].append(envelope)
    return groups


def _validate_prereg(
    root: Path, prereg_id: str
) -> tuple[evidence.EvidenceEnvelope, evidence.EvidenceEnvelope, dict[str, Any]]:
    _hex64(prereg_id, "prereg_id")
    spec = _load_spec(root)
    try:
        prereg = evidence.verify_object(prereg_id, repo_root=root)
    except evidence.EvidenceError as exc:
        raise ProspectiveError(f"preregistration does not verify: {exc}") from exc
    if prereg.kind != spec["evidence_kinds"]["prereg"] or prereg.dependencies:
        raise ProspectiveError("preregistration kind or ancestry differs")
    fields = {
        "schema",
        "issue",
        "h0_git_sha",
        "pair",
        "horizon_seconds",
        "candidate_id",
        "incumbent_id",
        "t0",
        "preflight",
        "runtime_environment",
        "source_paths",
        "trial_path_template",
        "decision_contract",
        "evidence_class",
        "money",
        "activation",
    }
    payload = _exact_fields(prereg.payload, fields, "preregistration payload")
    h0 = payload["h0_git_sha"]
    if not isinstance(h0, str) or HEX40.fullmatch(h0) is None:
        raise ProspectiveError("preregistration H0 is malformed")
    t0 = _validate_registered_t0(payload["t0"])
    preflight = payload["preflight"]
    if (
        not isinstance(preflight, dict)
        or set(preflight) != {"path", "sha256", "payload"}
        or preflight["path"] != spec["paths"]["preflight_result"]
        or not isinstance(preflight["payload"], dict)
        or _sha256_bytes(_canonical(preflight["payload"])) != preflight["sha256"]
    ):
        raise ProspectiveError("preregistration preflight identity differs")
    preflight_payload = _validate_preflight_value(preflight["payload"], spec)
    if _utc(preflight_payload["generated_utc"], "preflight generated_utc") >= _utc(
        t0["utc"], "registered T0 UTC"
    ):
        raise ProspectiveError("preflight must be generated before the registered T0")
    expected_payload = _prereg_payload(
        spec,
        h0=h0,
        t0=t0,
        preflight_path=preflight["path"],
        preflight_payload=preflight_payload,
        runtime_environment=_verified_runtime_environment(root, h0),
    )
    if payload != expected_payload:
        raise ProspectiveError("preregistration fixed contract differs")
    expected_artifacts = _artifact_refs(root, _bound_paths(root, spec))
    if prereg.artifacts != expected_artifacts:
        raise ProspectiveError("preregistration artifact bindings differ")
    groups = _domain_objects(root, spec)
    receipts = [
        item
        for item in groups[spec["evidence_kinds"]["collection_access_receipt"]]
        if item.dependencies == (prereg.object_id,)
    ]
    expected_collection = evidence.EvidenceEnvelope.create(
        kind=spec["evidence_kinds"]["collection_access_receipt"],
        payload=_collection_payload(spec, prereg),
        dependencies=[prereg.object_id],
    )
    if len(receipts) != 1 or receipts[0].as_dict() != expected_collection.as_dict():
        raise ProspectiveError("collection access receipt is missing or conflicting")
    collection = evidence.verify_object(receipts[0].object_id, repo_root=root)
    return prereg, collection, spec


def _validate_prereg_metadata(
    root: Path,
    prereg_id: str,
    spec: Mapping[str, Any],
) -> tuple[evidence.EvidenceEnvelope, evidence.EvidenceEnvelope]:
    """Validate fixed envelope metadata without reading bound artifact bytes."""

    _hex64(prereg_id, "metadata-only prereg_id")
    groups = _domain_objects(root, spec)
    matches = [
        item
        for item in groups[spec["evidence_kinds"]["prereg"]]
        if item.object_id == prereg_id
    ]
    if len(matches) != 1:
        raise ProspectiveError("metadata-only preregistration is missing or ambiguous")
    prereg = matches[0]
    fields = {
        "schema",
        "issue",
        "h0_git_sha",
        "pair",
        "horizon_seconds",
        "candidate_id",
        "incumbent_id",
        "t0",
        "preflight",
        "runtime_environment",
        "source_paths",
        "trial_path_template",
        "decision_contract",
        "evidence_class",
        "money",
        "activation",
    }
    payload = _exact_fields(prereg.payload, fields, "metadata-only preregistration")
    h0 = payload["h0_git_sha"]
    if (
        prereg.kind != spec["evidence_kinds"]["prereg"]
        or prereg.dependencies
        or not isinstance(h0, str)
        or HEX40.fullmatch(h0) is None
    ):
        raise ProspectiveError("metadata-only preregistration identity differs")
    t0 = _validate_registered_t0(payload["t0"])
    preflight = _exact_fields(
        payload["preflight"], {"path", "sha256", "payload"}, "metadata-only preflight"
    )
    preflight_payload = _validate_preflight_value(preflight["payload"], spec)
    if (
        preflight["path"] != spec["paths"]["preflight_result"]
        or _sha256_bytes(_canonical(preflight_payload)) != preflight["sha256"]
        or _utc(preflight_payload["generated_utc"], "preflight generated_utc")
        >= _utc(t0["utc"], "registered T0 UTC")
    ):
        raise ProspectiveError("metadata-only preflight identity differs")
    runtime = _exact_fields(
        payload["runtime_environment"],
        {
            "authorities",
            "python",
            "executable",
            "prefix",
            "base_prefix",
            "user_site_enabled",
            "distributions",
        },
        "metadata-only runtime environment",
    )
    if (
        not isinstance(runtime["authorities"], dict)
        or set(runtime["authorities"])
        != {"ENVIRONMENT_libs.txt", "requirements-deriv-demo.txt"}
        or any(
            not isinstance(path, str) or not isinstance(digest, str) or HEX64.fullmatch(digest) is None
            for path, digest in runtime["authorities"].items()
        )
        or not isinstance(runtime["distributions"], dict)
        or set(runtime["distributions"]) != set(RUNTIME_DISTRIBUTIONS)
        or runtime["user_site_enabled"] is not False
    ):
        raise ProspectiveError("metadata-only runtime environment differs")
    expected_payload = _prereg_payload(
        spec,
        h0=h0,
        t0=t0,
        preflight_path=preflight["path"],
        preflight_payload=preflight_payload,
        runtime_environment=runtime,
    )
    if payload != expected_payload:
        raise ProspectiveError("metadata-only preregistration contract differs")
    receipts = [
        item
        for item in groups[spec["evidence_kinds"]["collection_access_receipt"]]
        if item.dependencies == (prereg.object_id,)
    ]
    expected_collection = evidence.EvidenceEnvelope.create(
        kind=spec["evidence_kinds"]["collection_access_receipt"],
        payload=_collection_payload(spec, prereg),
        dependencies=[prereg.object_id],
    )
    if len(receipts) != 1 or receipts[0].as_dict() != expected_collection.as_dict():
        raise ProspectiveError("metadata-only collection receipt differs")
    return prereg, receipts[0]


def _metadata_trial_candidates(
    root: Path,
    spec: Mapping[str, Any] | None,
    prereg_id: str | None,
) -> list[tuple[evidence.EvidenceEnvelope, evidence.EvidenceEnvelope, dict[str, Any] | None]]:
    if spec is None:
        metadata = evidence.inspect_store_metadata(repo_root=root)
        registrations = [item for item in metadata if item.kind == PREREG_KIND]
        if prereg_id is not None:
            registrations = [item for item in registrations if item.object_id == prereg_id]
        candidates: list[tuple[str, Mapping[str, Any]]] = []
        for item in registrations:
            if not isinstance(item.payload, dict):
                continue
            try:
                recovery_spec = _load_h0_spec(root, item.payload.get("h0_git_sha"))
            except ProspectiveError:
                continue
            candidates.append((item.object_id, recovery_spec))
    else:
        groups = _domain_objects(root, spec)
        ids = (
            [prereg_id]
            if prereg_id is not None
            else [item.object_id for item in groups[spec["evidence_kinds"]["prereg"]]]
        )
        candidates = [(object_id, spec) for object_id in ids]
    out: list[
        tuple[evidence.EvidenceEnvelope, evidence.EvidenceEnvelope, dict[str, Any] | None]
    ] = []
    for object_id, candidate_spec in candidates:
        try:
            groups = _domain_objects(root, candidate_spec)
            prereg, collection = _validate_prereg_metadata(
                root, object_id, candidate_spec
            )
            if _matching_objects(
                groups,
                candidate_spec["evidence_kinds"]["terminal"],
                prereg.object_id,
            ):
                continue
            paths = _trial_paths(root, candidate_spec, prereg.object_id, create=False)
            if not paths["database"].exists():
                continue
            conn = _open_trial_readonly(paths["database"])
            try:
                metadata = conn.execute(
                    "SELECT prereg_id, created_epoch FROM trial_metadata WHERE singleton=1"
                ).fetchone()
                if (
                    metadata is None
                    or metadata["prereg_id"] != prereg.object_id
                    or int(metadata["created_epoch"])
                    >= int(prereg.payload["t0"]["epoch"])
                ):
                    continue
                invalid = _load_protocol_invalid(conn, prereg.object_id)
            finally:
                conn.close()
            out.append((prereg, collection, invalid))
        except (ProspectiveError, evidence.EvidenceError, sqlite3.DatabaseError, OSError):
            continue
    return out


def _record_startup_protocol_invalid(
    root: Path,
    spec: Mapping[str, Any] | None,
    prereg_id: str | None,
    error: Exception,
) -> bool:
    candidates = _metadata_trial_candidates(root, spec, prereg_id)
    if len(candidates) != 1:
        return False
    prereg, _collection, existing = candidates[0]
    detected_epoch = int(time.time())
    if detected_epoch < int(prereg.payload["t0"]["epoch"]):
        return False
    if existing is not None:
        return True
    recovery_spec = spec or _load_h0_spec(root, prereg.payload["h0_git_sha"])
    paths = _trial_paths(root, recovery_spec, prereg.object_id, create=False)
    with _trial_lock(paths["lock"]):
        conn = _open_trial(paths["database"], create=False)
        try:
            if _load_protocol_invalid(conn, prereg.object_id) is not None:
                return True
            if _load_trial_seal(conn, prereg.object_id) is not None:
                raise RetryableSealInterruption(
                    "startup authority failed after the logical seal; restore the "
                    "exact H1-bound artifacts and restart seal completion"
                )
            payload = _protocol_invalid_payload(
                prereg.object_id,
                detected_epoch=detected_epoch,
                stage="STARTUP_AUTHORITY",
                reason=f"{type(error).__name__}: {error}",
            )
            _insert_protocol_invalid_once(conn, payload)
        finally:
            conn.close()
    return True


def _require_h1(
    root: Path,
    prereg: evidence.EvidenceEnvelope,
    collection: evidence.EvidenceEnvelope,
) -> str:
    if _git(root, "branch", "--show-current") != "main":
        raise ProspectiveError("collection requires branch main")
    head = _git(root, "rev-parse", "HEAD")
    if head != _git(root, "rev-parse", "origin/main"):
        raise ProspectiveError("collection requires HEAD == origin/main")
    changed = {
        line[3:].split(" -> ", 1)[-1]
        for line in _git(
            root, "status", "--porcelain=v1", "--untracked-files=all"
        ).splitlines()
        if line
    }
    if changed:
        spec = _load_spec(root)
        groups = _domain_objects(root, spec)
        sealed, _source_ref = _validated_sealed_source(
            root, prereg, collection, spec, groups=groups
        )
        allowed = {f"evidence/objects/{sealed.object_id}.json"}
        if changed != allowed:
            raise ProspectiveError(
                "collection repository state differs from the single recoverable sealed source"
            )
    parents = _git(root, "rev-list", "--parents", "-n", "1", head).split()
    h0 = prereg.payload["h0_git_sha"]
    if parents != [head, h0]:
        raise ProspectiveError("H1 must have the preregistered H0 as its sole parent")
    changed = _git(root, "diff", "--name-only", h0, head).splitlines()
    expected = sorted(
        [
            f"evidence/objects/{prereg.object_id}.json",
            f"evidence/objects/{collection.object_id}.json",
        ]
    )
    if sorted(changed) != expected or len(changed) != 2:
        raise ProspectiveError("H1 must contain exactly the preregistration and collection receipt")
    return head


def _trial_relative(spec: Mapping[str, Any], prereg_id: str) -> str:
    relative = str(spec["paths"]["trial_template"]).format(prereg_id=prereg_id)
    return _canonical_relative(relative, "trial database path")


def _trial_paths(
    root: Path,
    spec: Mapping[str, Any],
    prereg_id: str,
    *,
    create: bool = True,
) -> dict[str, Path]:
    database = root / _trial_relative(spec, prereg_id)
    trial_dir = database.parent
    expected_parent = (
        root
        / "deriv_data/prospective"
        / spec["candidate"]["artifact_id"]
        / prereg_id
    )
    if trial_dir != expected_parent:
        raise ProspectiveError("trial database path differs from the fixed namespace")
    current = root
    for part in trial_dir.relative_to(root).parts:
        current = current / part
        if current.exists() and current.is_symlink():
            raise ProspectiveError(f"trial path contains a symlink: {current}")
        if create:
            current.mkdir(mode=0o700, exist_ok=True)
        if current.exists() and (current.is_symlink() or not current.is_dir()):
            raise ProspectiveError(f"trial path component is not a real directory: {current}")
    if create and trial_dir.resolve(strict=True) != expected_parent.resolve(strict=True):
        raise ProspectiveError("trial path escapes the fixed namespace")
    return {
        "dir": trial_dir,
        "database": database,
        "lock": trial_dir / "trial.lock",
        "packet": trial_dir / "sealed_source.sqlite",
        "packet_stage": trial_dir / ".sealed_source.building.sqlite",
    }


@contextmanager
def _trial_lock(path: Path) -> Iterator[None]:
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise ProspectiveError("trial lock is not a single-link regular file")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ProspectiveError("another prospective writer holds the trial lock") from exc
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


TRIAL_COLUMNS = {
    "trial_metadata": (
        "singleton",
        "prereg_id",
        "schema",
        "metadata_json",
        "metadata_sha256",
        "created_epoch",
    ),
    "decisions": (
        "prereg_id",
        "decision_close_epoch",
        "schema",
        "common_scoreable",
        "candidate_direction",
        "incumbent_direction",
        "candidate_gate_passed",
        "incumbent_gate_passed",
        "exclusion_reason",
        "recorded_epoch",
        "row_json",
        "row_sha256",
    ),
    "trial_seal": (
        "prereg_id",
        "cutoff_epoch",
        "trigger",
        "sealed_epoch",
        "seal_json",
        "seal_sha256",
    ),
    "trial_invalid": (
        "prereg_id",
        "detected_epoch",
        "stage",
        "reason",
        "invalid_json",
        "invalid_sha256",
    ),
}


def _create_trial_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS trial_metadata (
            singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
            prereg_id TEXT NOT NULL UNIQUE,
            schema TEXT NOT NULL,
            metadata_json TEXT NOT NULL,
            metadata_sha256 TEXT NOT NULL CHECK (length(metadata_sha256) = 64),
            created_epoch INTEGER NOT NULL
        ) STRICT;
        CREATE TABLE IF NOT EXISTS decisions (
            prereg_id TEXT NOT NULL REFERENCES trial_metadata(prereg_id) ON DELETE RESTRICT,
            decision_close_epoch INTEGER NOT NULL,
            schema TEXT NOT NULL,
            common_scoreable INTEGER NOT NULL CHECK (common_scoreable IN (0,1)),
            candidate_direction TEXT CHECK (candidate_direction IN ('UP','DOWN') OR candidate_direction IS NULL),
            incumbent_direction TEXT CHECK (incumbent_direction IN ('UP','DOWN') OR incumbent_direction IS NULL),
            candidate_gate_passed INTEGER CHECK (candidate_gate_passed IN (0,1) OR candidate_gate_passed IS NULL),
            incumbent_gate_passed INTEGER CHECK (incumbent_gate_passed IN (0,1) OR incumbent_gate_passed IS NULL),
            exclusion_reason TEXT NOT NULL,
            recorded_epoch INTEGER NOT NULL,
            row_json TEXT NOT NULL,
            row_sha256 TEXT NOT NULL CHECK (length(row_sha256) = 64),
            CHECK ((common_scoreable = 1 AND exclusion_reason = '') OR (common_scoreable = 0 AND length(exclusion_reason) > 0)),
            PRIMARY KEY (prereg_id, decision_close_epoch)
        ) WITHOUT ROWID, STRICT;
        CREATE TABLE IF NOT EXISTS trial_seal (
            prereg_id TEXT PRIMARY KEY REFERENCES trial_metadata(prereg_id) ON DELETE RESTRICT,
            cutoff_epoch INTEGER NOT NULL,
            trigger TEXT NOT NULL CHECK (trigger IN ('COUNT_TRIGGER','WEEK_26_CAP')),
            sealed_epoch INTEGER NOT NULL,
            seal_json TEXT NOT NULL,
            seal_sha256 TEXT NOT NULL CHECK (length(seal_sha256) = 64)
        ) WITHOUT ROWID, STRICT;
        CREATE TABLE IF NOT EXISTS trial_invalid (
            prereg_id TEXT PRIMARY KEY REFERENCES trial_metadata(prereg_id) ON DELETE RESTRICT,
            detected_epoch INTEGER NOT NULL,
            stage TEXT NOT NULL CHECK (length(stage) > 0),
            reason TEXT NOT NULL CHECK (length(reason) > 0),
            invalid_json TEXT NOT NULL,
            invalid_sha256 TEXT NOT NULL CHECK (length(invalid_sha256) = 64)
        ) WITHOUT ROWID, STRICT;
        CREATE TRIGGER IF NOT EXISTS trial_metadata_no_update BEFORE UPDATE ON trial_metadata BEGIN SELECT RAISE(ABORT, 'immutable trial_metadata'); END;
        CREATE TRIGGER IF NOT EXISTS trial_metadata_no_delete BEFORE DELETE ON trial_metadata BEGIN SELECT RAISE(ABORT, 'immutable trial_metadata'); END;
        CREATE TRIGGER IF NOT EXISTS decisions_no_update BEFORE UPDATE ON decisions BEGIN SELECT RAISE(ABORT, 'immutable decisions'); END;
        CREATE TRIGGER IF NOT EXISTS decisions_no_delete BEFORE DELETE ON decisions BEGIN SELECT RAISE(ABORT, 'immutable decisions'); END;
        CREATE TRIGGER IF NOT EXISTS trial_seal_no_update BEFORE UPDATE ON trial_seal BEGIN SELECT RAISE(ABORT, 'immutable trial_seal'); END;
        CREATE TRIGGER IF NOT EXISTS trial_seal_no_delete BEFORE DELETE ON trial_seal BEGIN SELECT RAISE(ABORT, 'immutable trial_seal'); END;
        CREATE TRIGGER IF NOT EXISTS trial_invalid_no_update BEFORE UPDATE ON trial_invalid BEGIN SELECT RAISE(ABORT, 'immutable trial_invalid'); END;
        CREATE TRIGGER IF NOT EXISTS trial_invalid_no_delete BEFORE DELETE ON trial_invalid BEGIN SELECT RAISE(ABORT, 'immutable trial_invalid'); END;
        """
    )


def _assert_trial_schema(conn: sqlite3.Connection, *, packet: bool = False) -> None:
    def definitions(database: sqlite3.Connection) -> tuple[tuple[Any, ...], ...]:
        return tuple(
            tuple(row)
            for row in database.execute(
                """
                SELECT type, name, tbl_name, sql
                FROM sqlite_schema
                WHERE type IN ('table','trigger','index','view')
                  AND name NOT LIKE 'sqlite_%'
                ORDER BY type, name
                """
            )
        )

    expected_conn = sqlite3.connect(":memory:", isolation_level=None)
    try:
        expected_conn.execute("PRAGMA foreign_keys=ON")
        _create_trial_schema(expected_conn)
        if packet:
            _create_settlement_table(expected_conn)
        expected_definitions = definitions(expected_conn)
    finally:
        expected_conn.close()
    if definitions(conn) != expected_definitions:
        raise ProspectiveError("SQLite schema definitions differ from the fixed contract")
    tables = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_schema WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
    }
    expected_tables = set(PACKET_TABLES if packet else TRIAL_TABLES)
    if tables != expected_tables:
        raise ProspectiveError(
            f"SQLite tables differ: expected={sorted(expected_tables)} actual={sorted(tables)}"
        )
    expected_columns = dict(TRIAL_COLUMNS)
    if packet:
        expected_columns["settlement_rows"] = (
            "prereg_id",
            "decision_close_epoch",
            "status",
            "reason",
            "entry_epoch",
            "entry_quote",
            "entry_bid",
            "entry_ask",
            "exit_epoch",
            "exit_quote",
            "exit_bid",
            "exit_ask",
        )
    for table, expected in expected_columns.items():
        columns = tuple(row[1] for row in conn.execute(f"PRAGMA table_info({table})"))
        if columns != expected:
            raise ProspectiveError(f"SQLite {table} columns differ")
        table_sql = conn.execute(
            "SELECT sql FROM sqlite_schema WHERE type='table' AND name=?", (table,)
        ).fetchone()
        if table_sql is None or "STRICT" not in str(table_sql[0]).upper():
            raise ProspectiveError(f"SQLite {table} is not a STRICT table")
    expected_triggers = {
        "trial_metadata_no_update",
        "trial_metadata_no_delete",
        "decisions_no_update",
        "decisions_no_delete",
        "trial_seal_no_update",
        "trial_seal_no_delete",
        "trial_invalid_no_update",
        "trial_invalid_no_delete",
    }
    if packet:
        expected_triggers.update(
            {"settlement_rows_no_update", "settlement_rows_no_delete"}
        )
    triggers = {
        row[0]
        for row in conn.execute("SELECT name FROM sqlite_schema WHERE type='trigger'")
    }
    if triggers != expected_triggers:
        raise ProspectiveError("SQLite immutable-trigger inventory differs")
    forbidden = {
        "return",
        "correct",
        "accuracy",
        "yield",
        "profit",
        "payout",
        "stake",
        "money",
    }
    for table in TRIAL_TABLES:
        if any(any(token in name.lower() for token in forbidden) for name in TRIAL_COLUMNS[table]):
            raise ProspectiveError("live trial schema contains an efficacy or money column")
    if conn.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
        raise ProspectiveError("SQLite integrity_check failed")


def _open_trial(path: Path, *, create: bool = True) -> sqlite3.Connection:
    if path.is_symlink():
        raise ProspectiveError("trial database must not be a symlink")
    if not create and not path.exists():
        raise ProspectiveError("trial database is missing")
    before = None
    if path.exists():
        before = _regular_file(path, "trial database")
        if before.st_nlink != 1:
            raise ProspectiveError("trial database must have exactly one hard link")
    conn = sqlite3.connect(
        str(path), timeout=SQLITE_BUSY_TIMEOUT_MS / 1000.0, isolation_level=None
    )
    conn.row_factory = sqlite3.Row
    try:
        conn.execute(f"PRAGMA busy_timeout={SQLITE_BUSY_TIMEOUT_MS}")
        conn.execute("PRAGMA foreign_keys=ON")
        mode = conn.execute("PRAGMA journal_mode=WAL").fetchone()[0]
        conn.execute("PRAGMA synchronous=FULL")
        conn.execute("PRAGMA application_id=1431119105")
        conn.execute("PRAGMA user_version=1")
        if str(mode).lower() != "wal":
            raise ProspectiveError("trial SQLite journal mode is not WAL")
        _create_trial_schema(conn)
        pragmas = {
            "journal_mode": str(conn.execute("PRAGMA journal_mode").fetchone()[0]).lower(),
            "synchronous": int(conn.execute("PRAGMA synchronous").fetchone()[0]),
            "foreign_keys": int(conn.execute("PRAGMA foreign_keys").fetchone()[0]),
            "application_id": int(conn.execute("PRAGMA application_id").fetchone()[0]),
            "user_version": int(conn.execute("PRAGMA user_version").fetchone()[0]),
        }
        if pragmas != {
            "journal_mode": "wal",
            "synchronous": 2,
            "foreign_keys": 1,
            "application_id": 1431119105,
            "user_version": 1,
        }:
            raise ProspectiveError(f"trial SQLite PRAGMAs differ: {pragmas}")
        after = _regular_file(path, "opened trial database")
        if after.st_nlink != 1 or (
            before is not None
            and (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino)
        ):
            raise ProspectiveError("trial database identity changed while being opened")
        _assert_trial_schema(conn)
        return conn
    except Exception:
        conn.close()
        raise


def _metadata_payload(
    prereg: evidence.EvidenceEnvelope,
    collection: evidence.EvidenceEnvelope,
    spec: Mapping[str, Any],
    h1: str,
) -> dict[str, Any]:
    return {
        "schema": TRIAL_METADATA_SCHEMA,
        "prereg_id": prereg.object_id,
        "collection_access_receipt_id": collection.object_id,
        "h0_git_sha": prereg.payload["h0_git_sha"],
        "h1_git_sha": h1,
        "t0": prereg.payload["t0"],
        "pair": PAIR,
        "horizon_seconds": spec["horizon_seconds"],
        "candidate_id": spec["candidate"]["artifact_id"],
        "incumbent_id": spec["incumbent"]["book_id"],
        "trial_path": _trial_relative(spec, prereg.object_id),
        "source_paths": prereg.payload["source_paths"],
        "preflight": prereg.payload["preflight"],
        "observer": SCRIPT_PATH,
        "service": SERVICE_PATH,
        "host": os.uname().nodename,
        "runtime_environment": prereg.payload["runtime_environment"],
        "outcome_fields_present": False,
        "activation": False,
    }


def _begin(conn: sqlite3.Connection) -> None:
    conn.execute("BEGIN IMMEDIATE")


def _rollback(conn: sqlite3.Connection) -> None:
    if conn.in_transaction:
        conn.execute("ROLLBACK")


def _initialize_trial(
    conn: sqlite3.Connection,
    metadata: Mapping[str, Any],
    *,
    created_epoch: int | None = None,
) -> bool:
    raw = _canonical(dict(metadata)).decode("utf-8")
    digest = _sha256_bytes(raw.encode("utf-8"))
    _begin(conn)
    try:
        inserted = conn.execute(
            """
            INSERT INTO trial_metadata (
                singleton, prereg_id, schema, metadata_json, metadata_sha256, created_epoch
            ) VALUES (1, ?, ?, ?, ?, ?)
            ON CONFLICT(singleton) DO NOTHING
            RETURNING prereg_id
            """,
            (
                metadata["prereg_id"],
                TRIAL_METADATA_SCHEMA,
                raw,
                digest,
                int(created_epoch if created_epoch is not None else time.time()),
            ),
        ).fetchone()
        if inserted is None:
            existing = conn.execute("SELECT * FROM trial_metadata WHERE singleton=1").fetchone()
            if (
                existing is None
                or existing["prereg_id"] != metadata["prereg_id"]
                or existing["schema"] != TRIAL_METADATA_SCHEMA
                or existing["metadata_json"] != raw
                or existing["metadata_sha256"] != digest
            ):
                raise ProspectiveError("trial metadata conflicts with the preregistration")
        conn.execute("COMMIT")
        return inserted is not None
    except Exception:
        _rollback(conn)
        raise


def _provider_identity(prereg: evidence.EvidenceEnvelope) -> dict[str, Any]:
    return {
        "class": "public_Deriv_persisted_store",
        "preflight_path": prereg.payload["preflight"]["path"],
        "preflight_sha256": prereg.payload["preflight"]["sha256"],
        "candle_store": prereg.payload["source_paths"]["candles"],
        "tick_shards": prereg.payload["source_paths"]["tick_shards"],
    }


def _decision_payload(
    *,
    prereg_id: str,
    decision_epoch: int,
    recorded_epoch: int,
    provider: Mapping[str, Any],
    scoring: Mapping[str, Any] | None,
    exclusion_reason: str,
) -> dict[str, Any]:
    common = scoring is not None
    if common == bool(exclusion_reason):
        raise ProspectiveError("decision exclusion reason/common-scoreable state differs")
    feature_epoch = decision_epoch - 60
    return {
        "schema": DECISION_SCHEMA,
        "prereg_id": prereg_id,
        "decision_close_epoch": int(decision_epoch),
        "decision_close_utc": _canonical_utc(datetime.fromtimestamp(decision_epoch, UTC)),
        "feature_timestamp_epoch": feature_epoch,
        "feature_timestamp_utc": _canonical_utc(datetime.fromtimestamp(feature_epoch, UTC)),
        "recorded_epoch": int(recorded_epoch),
        "provider": dict(provider),
        "source_observation": (
            {
                "row_source": scoring["row_source"],
                "warnings": list(scoring["warnings"]),
            }
            if common
            else None
        ),
        "feature_sha256": scoring["feature_sha256"] if common else None,
        "feature_dtype": "float32" if common else None,
        "feature_count": 337 if common else None,
        "candidate": dict(scoring["candidate"]) if common else None,
        "incumbent": dict(scoring["incumbent"]) if common else None,
        "common_scoreable": common,
        "exclusion_reason": exclusion_reason,
        "outcome_fields_present": False,
        "activation": False,
    }


def _decision_db_values(payload: Mapping[str, Any]) -> tuple[Any, ...]:
    raw = _canonical(dict(payload)).decode("utf-8")
    candidate = payload["candidate"]
    incumbent = payload["incumbent"]
    return (
        payload["prereg_id"],
        int(payload["decision_close_epoch"]),
        DECISION_SCHEMA,
        int(bool(payload["common_scoreable"])),
        candidate["direction"] if candidate is not None else None,
        incumbent["direction"] if incumbent is not None else None,
        int(bool(candidate["gate_passed"])) if candidate is not None else None,
        int(bool(incumbent["gate_passed"])) if incumbent is not None else None,
        payload["exclusion_reason"],
        int(payload["recorded_epoch"]),
        raw,
        _sha256_bytes(raw.encode("utf-8")),
    )


def _validate_decision_value(value: Any) -> dict[str, Any]:
    fields = {
        "schema",
        "prereg_id",
        "decision_close_epoch",
        "decision_close_utc",
        "feature_timestamp_epoch",
        "feature_timestamp_utc",
        "recorded_epoch",
        "provider",
        "source_observation",
        "feature_sha256",
        "feature_dtype",
        "feature_count",
        "candidate",
        "incumbent",
        "common_scoreable",
        "exclusion_reason",
        "outcome_fields_present",
        "activation",
    }
    row = _exact_fields(value, fields, "decision payload")
    decision_epoch = row["decision_close_epoch"]
    feature_epoch = row["feature_timestamp_epoch"]
    if (
        row["schema"] != DECISION_SCHEMA
        or not isinstance(decision_epoch, int)
        or not isinstance(feature_epoch, int)
        or feature_epoch + 60 != decision_epoch
        or decision_epoch % 60
        or row["decision_close_utc"]
        != _canonical_utc(datetime.fromtimestamp(decision_epoch, UTC))
        or row["feature_timestamp_utc"]
        != _canonical_utc(datetime.fromtimestamp(feature_epoch, UTC))
        or not bool(refresh_stats.canonical_runtime_ny_mask([decision_epoch])[0])
        or not isinstance(row["recorded_epoch"], int)
        or row["recorded_epoch"] < decision_epoch + 20
        or not isinstance(row["provider"], dict)
        or row["outcome_fields_present"] is not False
        or row["activation"] is not False
        or type(row["common_scoreable"]) is not bool
        or not isinstance(row["exclusion_reason"], str)
    ):
        raise ProspectiveError("decision clock, provider, or lifecycle differs")
    if not row["common_scoreable"]:
        if (
            not row["exclusion_reason"]
            or any(
                row[name] is not None
                for name in (
                    "source_observation",
                    "feature_sha256",
                    "feature_dtype",
                    "feature_count",
                    "candidate",
                    "incumbent",
                )
            )
        ):
            raise ProspectiveError("unscoreable decision payload differs")
        return row
    if row["exclusion_reason"] or row["recorded_epoch"] >= decision_epoch + 80:
        raise ProspectiveError("scoreable decision was not captured at its live deadline")
    source = _exact_fields(
        row["source_observation"], {"row_source", "warnings"}, "decision source observation"
    )
    _text(source["row_source"], "decision row source")
    if not isinstance(source["warnings"], list) or any(
        not isinstance(item, str) for item in source["warnings"]
    ):
        raise ProspectiveError("decision source warnings differ")
    _hex64(row["feature_sha256"], "decision feature sha256")
    if row["feature_dtype"] != "float32" or row["feature_count"] != 337:
        raise ProspectiveError("decision feature identity differs")
    score_fields = {
        "book_id",
        "content_id",
        "probability",
        "confidence",
        "threshold",
        "direction",
        "confidence_gate_passed",
        "structural_gate_passed",
        "gate_passed",
        "gate_reasons",
    }
    for policy in ("candidate", "incumbent"):
        score = _exact_fields(row[policy], score_fields, f"decision {policy} score")
        probability = score["probability"]
        confidence = score["confidence"]
        threshold = score["threshold"]
        if (
            not isinstance(score["book_id"], str)
            or not score["book_id"]
            or not isinstance(score["content_id"], str)
            or not score["content_id"]
            or not isinstance(probability, (int, float))
            or not math.isfinite(float(probability))
            or not 0 <= float(probability) <= 1
            or not isinstance(confidence, (int, float))
            or float(confidence) != abs(float(probability) - 0.5)
            or not isinstance(threshold, (int, float))
            or not math.isfinite(float(threshold))
            or float(threshold) < 0
            or score["direction"] != ("UP" if float(probability) >= 0.5 else "DOWN")
            or type(score["confidence_gate_passed"]) is not bool
            or score["confidence_gate_passed"] is not (
                float(confidence) >= float(threshold)
            )
            or type(score["structural_gate_passed"]) is not bool
            or type(score["gate_passed"]) is not bool
            or score["gate_passed"] is not (
                score["confidence_gate_passed"] and score["structural_gate_passed"]
            )
            or not isinstance(score["gate_reasons"], list)
            or any(not isinstance(item, str) for item in score["gate_reasons"])
        ):
            raise ProspectiveError(f"decision {policy} score differs")
    return row


def _protocol_invalid_payload(
    prereg_id: str,
    *,
    detected_epoch: int,
    stage: str,
    reason: str,
) -> dict[str, Any]:
    normalized_stage = re.sub(r"[^A-Za-z0-9_.:-]+", "_", str(stage)).strip("_")[:80]
    normalized_reason = re.sub(r"[^A-Za-z0-9_.: /-]+", "_", str(reason)).strip()[:500]
    if not normalized_stage or not normalized_reason:
        raise ProspectiveError("protocol-invalid stage and reason must be non-empty")
    return {
        "schema": TRIAL_INVALID_SCHEMA,
        "prereg_id": _hex64(prereg_id, "protocol-invalid prereg_id"),
        "detected_epoch": int(detected_epoch),
        "stage": normalized_stage,
        "reason": normalized_reason,
        "outcome_fields_present": False,
        "activation": False,
    }


def _validate_protocol_invalid_value(value: Any) -> dict[str, Any]:
    payload = _exact_fields(
        value,
        {
            "schema",
            "prereg_id",
            "detected_epoch",
            "stage",
            "reason",
            "outcome_fields_present",
            "activation",
        },
        "trial protocol-invalid payload",
    )
    if (
        payload["schema"] != TRIAL_INVALID_SCHEMA
        or not isinstance(payload["detected_epoch"], int)
        or payload["detected_epoch"] < 0
        or not isinstance(payload["stage"], str)
        or not payload["stage"]
        or not isinstance(payload["reason"], str)
        or not payload["reason"]
        or payload["outcome_fields_present"] is not False
        or payload["activation"] is not False
    ):
        raise ProspectiveError("trial protocol-invalid payload differs")
    _hex64(payload["prereg_id"], "trial protocol-invalid prereg_id")
    return payload


def _load_protocol_invalid(
    conn: sqlite3.Connection, prereg_id: str
) -> dict[str, Any] | None:
    row = conn.execute(
        "SELECT * FROM trial_invalid WHERE prereg_id=?", (prereg_id,)
    ).fetchone()
    if row is None:
        return None
    raw = row["invalid_json"].encode("utf-8")
    if _sha256_bytes(raw) != row["invalid_sha256"]:
        raise ProspectiveError("trial protocol-invalid row hash differs")
    try:
        payload = evidence.decode_canonical_json(raw)
    except evidence.EvidenceError as exc:
        raise ProspectiveError(
            f"trial protocol-invalid row is not canonical JSON: {exc}"
        ) from exc
    payload = _validate_protocol_invalid_value(payload)
    expected = (
        payload["prereg_id"],
        payload["detected_epoch"],
        payload["stage"],
        payload["reason"],
        row["invalid_json"],
        row["invalid_sha256"],
    )
    if (
        payload["prereg_id"] != prereg_id
        or tuple(row[name] for name in TRIAL_COLUMNS["trial_invalid"]) != expected
    ):
        raise ProspectiveError("trial protocol-invalid columns differ from their payload")
    return payload


def _insert_protocol_invalid_once(
    conn: sqlite3.Connection, payload: Mapping[str, Any]
) -> bool:
    normalized = _validate_protocol_invalid_value(dict(payload))
    raw = _canonical(normalized).decode("utf-8")
    values = (
        normalized["prereg_id"],
        normalized["detected_epoch"],
        normalized["stage"],
        normalized["reason"],
        raw,
        _sha256_bytes(raw.encode("utf-8")),
    )
    _begin(conn)
    try:
        inserted = conn.execute(
            """
            INSERT INTO trial_invalid (
                prereg_id, detected_epoch, stage, reason, invalid_json, invalid_sha256
            ) VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(prereg_id) DO NOTHING
            RETURNING detected_epoch
            """,
            values,
        ).fetchone()
        if inserted is None:
            existing = _load_protocol_invalid(conn, normalized["prereg_id"])
            if existing is None or existing != normalized:
                raise ProspectiveError(
                    "protocol-invalid retry conflicts with immutable state"
                )
        conn.execute("COMMIT")
        return inserted is not None
    except Exception:
        _rollback(conn)
        raise


def _insert_decision_once(conn: sqlite3.Connection, payload: Mapping[str, Any]) -> bool:
    values = _decision_db_values(payload)
    _begin(conn)
    try:
        if conn.execute("SELECT 1 FROM trial_invalid").fetchone() is not None:
            raise ProspectiveError("cannot insert a decision after protocol INVALID")
        if conn.execute("SELECT 1 FROM trial_seal").fetchone() is not None:
            raise ProspectiveError("cannot insert a decision after the trial seal")
        inserted = conn.execute(
            """
            INSERT INTO decisions (
                prereg_id, decision_close_epoch, schema, common_scoreable,
                candidate_direction, incumbent_direction, candidate_gate_passed,
                incumbent_gate_passed, exclusion_reason, recorded_epoch,
                row_json, row_sha256
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(prereg_id, decision_close_epoch) DO NOTHING
            RETURNING decision_close_epoch
            """,
            values,
        ).fetchone()
        if inserted is None:
            existing = conn.execute(
                "SELECT * FROM decisions WHERE prereg_id=? AND decision_close_epoch=?",
                (payload["prereg_id"], int(payload["decision_close_epoch"])),
            ).fetchone()
            if existing is None or tuple(existing[name] for name in TRIAL_COLUMNS["decisions"]) != values:
                raise ProspectiveError("decision retry conflicts with an immutable row")
        conn.execute("COMMIT")
        return inserted is not None
    except Exception:
        _rollback(conn)
        raise


def _expected_decision_epochs(start_epoch: int, end_epoch: int) -> np.ndarray:
    if end_epoch < start_epoch:
        return np.empty(0, dtype="int64")
    start = ((int(start_epoch) + 59) // 60) * 60
    end = (int(end_epoch) // 60) * 60
    if end < start:
        return np.empty(0, dtype="int64")
    values = np.arange(start, end + 1, 60, dtype="int64")
    return values[refresh_stats.canonical_runtime_ny_mask(values)]


def _last_decision_epoch(conn: sqlite3.Connection, prereg_id: str) -> int | None:
    row = conn.execute(
        "SELECT MAX(decision_close_epoch) AS value FROM decisions WHERE prereg_id=?",
        (prereg_id,),
    ).fetchone()
    return int(row["value"]) if row is not None and row["value"] is not None else None


def _assert_write_authority(
    root: Path,
    prereg_id: str,
) -> tuple[evidence.EvidenceEnvelope, evidence.EvidenceEnvelope, dict[str, Any], str]:
    _require_public_environment()
    prereg, collection, spec = _validate_prereg(root, prereg_id)
    h1 = _require_h1(root, prereg, collection)
    return prereg, collection, spec, h1


def _capture_due(
    root: Path,
    prereg: evidence.EvidenceEnvelope,
    spec: Mapping[str, Any],
    conn: sqlite3.Connection,
    candidate: book_runtime.LoadedBook,
    incumbent: book_runtime.LoadedBook,
    *,
    now_epoch: int,
    capture_through_epoch: int | None = None,
) -> int:
    if conn.execute("SELECT 1 FROM trial_seal").fetchone() is not None:
        return 0
    t0_epoch = int(prereg.payload["t0"]["epoch"])
    observed_epoch = int(now_epoch)
    through_epoch = (
        observed_epoch
        if capture_through_epoch is None
        else int(capture_through_epoch)
    )
    if through_epoch > observed_epoch:
        raise ProspectiveError("capture boundary is after the observation clock")
    target = (
        (through_epoch - int(spec["runtime"]["capture_delay_seconds"])) // 60
    ) * 60
    if target < t0_epoch:
        return 0
    last = _last_decision_epoch(conn, prereg.object_id)
    start = t0_epoch if last is None else last + 60
    due = _expected_decision_epochs(start, target)
    inserted = 0
    provider = _provider_identity(prereg)
    for decision_epoch in due:
        decision = int(decision_epoch)
        _assert_write_authority(root, prereg.object_id)
        if decision < target:
            payload = _decision_payload(
                prereg_id=prereg.object_id,
                decision_epoch=decision,
                recorded_epoch=observed_epoch,
                provider=provider,
                scoring=None,
                exclusion_reason="UNSCORED_MISSED_BEFORE_OBSERVATION",
            )
        else:
            try:
                scoring = _score_latest_store_row(root, spec, candidate, incumbent)
                if int(scoring["decision_close_epoch"]) != decision:
                    raise TransientFeatureGap(
                        "latest feature row does not map to the expected decision"
                    )
                registered_scoring = prereg.payload["preflight"]["payload"][
                    "scoring_parity"
                ]
                if (
                    scoring["row_source"] != registered_scoring["row_source"]
                    or scoring["warnings"] != registered_scoring["warnings"]
                ):
                    raise ProspectiveError(
                        "live feature provider semantics differ from preflight"
                    )
                payload = _decision_payload(
                    prereg_id=prereg.object_id,
                    decision_epoch=decision,
                    recorded_epoch=observed_epoch,
                    provider=provider,
                    scoring=scoring,
                    exclusion_reason="",
                )
            except TransientFeatureGap as exc:
                reason = re.sub(r"[^A-Za-z0-9_.:-]+", "_", f"{type(exc).__name__}:{exc}")[:240]
                payload = _decision_payload(
                    prereg_id=prereg.object_id,
                    decision_epoch=decision,
                    recorded_epoch=observed_epoch,
                    provider=provider,
                    scoring=None,
                    exclusion_reason=f"UNSCORED_{reason}",
                )
        inserted += int(_insert_decision_once(conn, payload))
    return inserted


def _load_decisions(conn: sqlite3.Connection, prereg_id: str) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT * FROM decisions WHERE prereg_id=? ORDER BY decision_close_epoch",
        (prereg_id,),
    ).fetchall()
    out: list[dict[str, Any]] = []
    previous: int | None = None
    for row in rows:
        raw = row["row_json"].encode("utf-8")
        if _sha256_bytes(raw) != row["row_sha256"]:
            raise ProspectiveError("decision row hash differs")
        try:
            value = evidence.decode_canonical_json(raw)
        except evidence.EvidenceError as exc:
            raise ProspectiveError(f"decision row is not canonical JSON: {exc}") from exc
        value = _validate_decision_value(value)
        if tuple(row[name] for name in TRIAL_COLUMNS["decisions"]) != _decision_db_values(value):
            raise ProspectiveError("decision columns differ from their canonical payload")
        epoch = value.get("decision_close_epoch")
        if not isinstance(epoch, int) or (previous is not None and epoch <= previous):
            raise ProspectiveError("decision row clock is not strict")
        previous = epoch
        out.append(value)
    return out


def _select_settlement(
    decision_epoch: int,
    ticks: pd.DataFrame,
    live_mask: np.ndarray,
    *,
    horizon_seconds: int = 900,
    tolerance_seconds: int = 10,
) -> dict[str, Any]:
    """Mechanically select fixed entry/exit observations without computing efficacy."""

    if len(ticks) != len(live_mask):
        raise ProspectiveError("tick rows and provenance mask are not aligned")
    admitted = ticks.loc[live_mask, ["epoch", "quote", "bid", "ask"]].reset_index(drop=True)
    if admitted.empty:
        return {"status": "UNSETTLED", "reason": "NO_ADMITTED_LIVE_TICKS"}
    clock = admitted["epoch"].to_numpy(dtype="int64", copy=False)
    if np.any(np.diff(clock) <= 0):
        raise ProspectiveError("admitted live tick clock is not strict")
    return _select_settlement_from_admitted(
        decision_epoch,
        admitted,
        clock,
        horizon_seconds=horizon_seconds,
        tolerance_seconds=tolerance_seconds,
    )


def _select_settlement_from_admitted(
    decision_epoch: int,
    admitted: pd.DataFrame,
    clock: np.ndarray,
    *,
    horizon_seconds: int,
    tolerance_seconds: int,
) -> dict[str, Any]:
    entry_target = int(decision_epoch) + 1
    exit_target = entry_target + int(horizon_seconds)
    future = admitted.loc[clock > int(decision_epoch), ["epoch"]]
    future_clock = future["epoch"].to_numpy(dtype="int64", copy=False)
    observations = np.concatenate(
        (np.asarray([int(decision_epoch)], dtype="int64"), future_clock)
    )
    synthetic_closes = np.ones(len(observations), dtype="float64")
    try:
        _synthetic_returns, canonical_valid, canonical_exit = (
            refresh_stats.canonical_settlement_with_exit_on_decision_grid(
                [int(decision_epoch)],
                observations,
                synthetic_closes,
                horizon_s=int(horizon_seconds),
                tol_s=int(tolerance_seconds),
                lag_s=1,
            )
        )
    except refresh_stats.StatisticsInputError as exc:
        raise ProspectiveError(f"canonical settlement selection failed: {exc}") from exc

    def unsettled(reason: str) -> dict[str, Any]:
        if bool(canonical_valid[0]):
            raise ProspectiveError(
                f"local settlement mapping contradicted canonical validity: {reason}"
            )
        return {"status": "UNSETTLED", "reason": reason}

    entry_position = int(np.searchsorted(clock, entry_target, side="left"))
    exit_position = int(np.searchsorted(clock, exit_target, side="right") - 1)
    if entry_position >= len(clock):
        return unsettled("ENTRY_MISSING")
    if int(clock[entry_position]) - entry_target > int(tolerance_seconds):
        return unsettled("ENTRY_LATE")
    if exit_position < 0 or exit_target - int(clock[exit_position]) > int(tolerance_seconds):
        return unsettled("EXIT_EARLY")
    if exit_position <= entry_position:
        return unsettled("ENTRY_EXIT_NOT_DISTINCT")
    entry = admitted.iloc[entry_position]
    exit_row = admitted.iloc[exit_position]
    if (
        not bool(canonical_valid[0])
        or int(canonical_exit[0]) != int(exit_row["epoch"])
    ):
        raise ProspectiveError("local settlement mapping differs from canonical owner")
    return {
        "status": "SETTLED",
        "reason": "",
        "entry_epoch": int(entry["epoch"]),
        "entry_quote": float(entry["quote"]),
        "entry_bid": float(entry["bid"]),
        "entry_ask": float(entry["ask"]),
        "exit_epoch": int(exit_row["epoch"]),
        "exit_quote": float(exit_row["quote"]),
        "exit_bid": float(exit_row["bid"]),
        "exit_ask": float(exit_row["ask"]),
    }


def _settlement_rows(
    decisions: Sequence[Mapping[str, Any]], ticks: pd.DataFrame, spec: Mapping[str, Any]
) -> list[dict[str, Any]]:
    live = _live_tick_mask(ticks)
    admitted = ticks.loc[live, ["epoch", "quote", "bid", "ask"]].reset_index(drop=True)
    clock = admitted["epoch"].to_numpy(dtype="int64", copy=False)
    if len(clock) > 1 and np.any(np.diff(clock) <= 0):
        raise ProspectiveError("admitted live tick clock is not strict")
    out: list[dict[str, Any]] = []
    for decision in decisions:
        base = {
            "prereg_id": decision["prereg_id"],
            "decision_close_epoch": int(decision["decision_close_epoch"]),
        }
        if not decision["common_scoreable"]:
            out.append(
                {
                    **base,
                    "status": "NOT_COMMON_SCOREABLE",
                    "reason": decision["exclusion_reason"],
                }
            )
            continue
        if admitted.empty:
            selected = {"status": "UNSETTLED", "reason": "NO_ADMITTED_LIVE_TICKS"}
        else:
            selected = _select_settlement_from_admitted(
                int(decision["decision_close_epoch"]),
                admitted,
                clock,
                horizon_seconds=int(spec["horizon_seconds"]),
                tolerance_seconds=int(spec["runtime"]["settlement_tolerance_seconds"]),
            )
        out.append({**base, **selected})
    return out


def _selection_manifest(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    outcome_blind = [
        {
            "decision_close_epoch": int(row["decision_close_epoch"]),
            "status": row["status"],
            "reason": row["reason"],
            "entry_epoch": row.get("entry_epoch"),
            "exit_epoch": row.get("exit_epoch"),
        }
        for row in rows
    ]
    return {
        "rows": len(rows),
        "settled": sum(row["status"] == "SETTLED" for row in rows),
        "unsettled": sum(row["status"] == "UNSETTLED" for row in rows),
        "not_common_scoreable": sum(
            row["status"] == "NOT_COMMON_SCOREABLE" for row in rows
        ),
        "timestamp_manifest_sha256": _sha256_bytes(_canonical(outcome_blind)),
    }


def _settlement_rows_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    columns = (
        "prereg_id",
        "decision_close_epoch",
        "status",
        "reason",
        "entry_epoch",
        "entry_quote",
        "entry_bid",
        "entry_ask",
        "exit_epoch",
        "exit_quote",
        "exit_bid",
        "exit_ask",
    )
    normalized = [dict(zip(columns, _settlement_values(row), strict=True)) for row in rows]
    normalized.sort(key=lambda row: (row["prereg_id"], row["decision_close_epoch"]))
    return _sha256_bytes(_canonical(normalized))


def _scheduled_counts(
    decisions: Sequence[Mapping[str, Any]], settlement: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    if len(decisions) != len(settlement):
        raise ProspectiveError("decision and settlement rows are not aligned")
    eligible_positions = [
        index for index, row in enumerate(settlement) if row["status"] == "SETTLED"
    ]
    timestamps = np.asarray(
        [int(decisions[index]["decision_close_epoch"]) for index in eligible_positions],
        dtype="int64",
    )
    result: dict[str, Any] = {"common_settled": len(timestamps)}
    for policy in ("candidate", "incumbent"):
        probabilities = np.asarray(
            [float(decisions[index][policy]["probability"]) for index in eligible_positions],
            dtype="float64",
        )
        gates = np.asarray(
            [bool(decisions[index][policy]["gate_passed"]) for index in eligible_positions],
            dtype=bool,
        )
        selected_indices = refresh_stats.canonical_nonoverlap_chrono(
            timestamps, gates, gap_s=900
        )
        selected_probabilities = probabilities[selected_indices]
        up = int(np.sum(selected_probabilities >= 0.5))
        down = int(np.sum(selected_probabilities < 0.5))
        result[policy] = {"combined": int(len(selected_indices)), "UP": up, "DOWN": down}
    return result


def _settlement_tick_intervals(
    decisions: Sequence[Mapping[str, Any]], spec: Mapping[str, Any]
) -> list[tuple[int, int]]:
    horizon = int(spec["horizon_seconds"])
    tolerance = int(spec["runtime"]["settlement_tolerance_seconds"])
    intervals: list[tuple[int, int]] = []
    for row in decisions:
        if not bool(row["common_scoreable"]):
            continue
        entry_target = int(row["decision_close_epoch"]) + 1
        exit_target = entry_target + horizon
        intervals.extend(
            [
                (entry_target, entry_target + tolerance),
                (exit_target - tolerance, exit_target),
            ]
        )
    return _normalized_tick_intervals(intervals, start_epoch=None, end_epoch=None)


def _week_boundaries(prereg: evidence.EvidenceEnvelope, spec: Mapping[str, Any]) -> list[tuple[int, int]]:
    t0 = datetime.fromtimestamp(int(prereg.payload["t0"]["epoch"]), UTC).astimezone(NY)
    if t0.weekday() != 0 or (t0.hour, t0.minute, t0.second) != (8, 0, 0):
        raise ProspectiveError("registered T0 is not a Monday 08:00 New York boundary")
    cap = int(spec["fixed_look"]["cap_weeks"])
    out: list[tuple[int, int]] = []
    for week in range(1, cap + 1):
        boundary_date = t0.date() + timedelta(days=4 + 7 * (week - 1))
        boundary = datetime.combine(boundary_date, wall_time(17, 0), tzinfo=NY)
        out.append((week, int(boundary.astimezone(UTC).timestamp())))
    return out


def _cutoff_decision(
    prereg: evidence.EvidenceEnvelope,
    spec: Mapping[str, Any],
    decisions: Sequence[Mapping[str, Any]],
    ticks: pd.DataFrame,
    *,
    now_epoch: int,
) -> dict[str, Any] | None:
    earliest = int(spec["fixed_look"]["earliest_weeks"])
    minimum = int(spec["fixed_look"]["minimum_candidate_per_side"])
    cap = int(spec["fixed_look"]["cap_weeks"])
    for week, boundary in _week_boundaries(prereg, spec):
        if boundary > int(now_epoch):
            break
        if week < earliest:
            continue
        prefix = [row for row in decisions if int(row["decision_close_epoch"]) < boundary]
        settlement = _settlement_rows(prefix, ticks, spec)
        counts = _scheduled_counts(prefix, settlement)
        count_ready = (
            counts["candidate"]["UP"] >= minimum
            and counts["candidate"]["DOWN"] >= minimum
        )
        if count_ready or week == cap:
            return {
                "cutoff_epoch": boundary,
                "week": week,
                "trigger": "COUNT_TRIGGER" if count_ready else "WEEK_26_CAP",
                "counts": counts,
                "selection": _selection_manifest(settlement),
            }
    return None


def _seal_payload(
    prereg_id: str,
    cutoff: Mapping[str, Any],
    *,
    settlement_sha256: str,
    sealed_epoch: int,
) -> dict[str, Any]:
    return {
        "schema": "usdchf-m15-prospective-trial-seal/v1",
        "prereg_id": prereg_id,
        "cutoff_epoch": int(cutoff["cutoff_epoch"]),
        "week": int(cutoff["week"]),
        "trigger": cutoff["trigger"],
        "sealed_epoch": int(sealed_epoch),
        "counts": cutoff["counts"],
        "selection": cutoff["selection"],
        "settlement_sha256": _hex64(
            settlement_sha256, "sealed settlement sha256"
        ),
        "outcome_fields_present": False,
        "activation": False,
    }


def _insert_trial_seal_once(
    conn: sqlite3.Connection, payload: Mapping[str, Any]
) -> bool:
    raw = _canonical(dict(payload)).decode("utf-8")
    values = (
        payload["prereg_id"],
        int(payload["cutoff_epoch"]),
        payload["trigger"],
        int(payload["sealed_epoch"]),
        raw,
        _sha256_bytes(raw.encode("utf-8")),
    )
    _begin(conn)
    try:
        if conn.execute("SELECT 1 FROM trial_invalid").fetchone() is not None:
            raise ProspectiveError("cannot seal a protocol-INVALID trial")
        inserted = conn.execute(
            """
            INSERT INTO trial_seal (
                prereg_id, cutoff_epoch, trigger, sealed_epoch, seal_json, seal_sha256
            ) VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(prereg_id) DO NOTHING
            RETURNING cutoff_epoch
            """,
            values,
        ).fetchone()
        if inserted is None:
            existing = conn.execute(
                "SELECT * FROM trial_seal WHERE prereg_id=?", (payload["prereg_id"],)
            ).fetchone()
            if existing is None or tuple(existing[name] for name in TRIAL_COLUMNS["trial_seal"]) != values:
                raise ProspectiveError("trial seal retry conflicts with immutable state")
        conn.execute("COMMIT")
        return inserted is not None
    except Exception:
        _rollback(conn)
        raise


def _load_trial_seal(conn: sqlite3.Connection, prereg_id: str) -> dict[str, Any] | None:
    row = conn.execute(
        "SELECT * FROM trial_seal WHERE prereg_id=?", (prereg_id,)
    ).fetchone()
    if row is None:
        return None
    raw = row["seal_json"].encode("utf-8")
    if _sha256_bytes(raw) != row["seal_sha256"]:
        raise ProspectiveError("trial seal hash differs")
    try:
        value = evidence.decode_canonical_json(raw)
    except evidence.EvidenceError as exc:
        raise ProspectiveError(f"trial seal is not canonical JSON: {exc}") from exc
    if not isinstance(value, dict) or value.get("prereg_id") != prereg_id:
        raise ProspectiveError("trial seal identity differs")
    _exact_fields(
        value,
        {
            "schema",
            "prereg_id",
            "cutoff_epoch",
            "week",
            "trigger",
            "sealed_epoch",
            "counts",
            "selection",
            "settlement_sha256",
            "outcome_fields_present",
            "activation",
        },
        "trial seal payload",
    )
    _hex64(value["settlement_sha256"], "trial seal settlement sha256")
    expected = (
        value["prereg_id"],
        int(value["cutoff_epoch"]),
        value["trigger"],
        int(value["sealed_epoch"]),
        row["seal_json"],
        row["seal_sha256"],
    )
    if tuple(row[name] for name in TRIAL_COLUMNS["trial_seal"]) != expected:
        raise ProspectiveError("trial seal columns differ from their canonical payload")
    return value


def _create_settlement_table(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE settlement_rows (
            prereg_id TEXT NOT NULL,
            decision_close_epoch INTEGER NOT NULL,
            status TEXT NOT NULL CHECK (status IN ('SETTLED','UNSETTLED','NOT_COMMON_SCOREABLE')),
            reason TEXT NOT NULL,
            entry_epoch INTEGER,
            entry_quote REAL,
            entry_bid REAL,
            entry_ask REAL,
            exit_epoch INTEGER,
            exit_quote REAL,
            exit_bid REAL,
            exit_ask REAL,
            PRIMARY KEY (prereg_id, decision_close_epoch),
            FOREIGN KEY (prereg_id, decision_close_epoch)
                REFERENCES decisions(prereg_id, decision_close_epoch) ON DELETE RESTRICT,
            CHECK (
                (status = 'SETTLED' AND reason = '' AND
                 entry_epoch IS NOT NULL AND entry_quote IS NOT NULL AND
                 entry_bid IS NOT NULL AND entry_ask IS NOT NULL AND
                 exit_epoch IS NOT NULL AND exit_quote IS NOT NULL AND
                 exit_bid IS NOT NULL AND exit_ask IS NOT NULL)
                OR
                (status != 'SETTLED' AND length(reason) > 0 AND
                 entry_epoch IS NULL AND entry_quote IS NULL AND
                 entry_bid IS NULL AND entry_ask IS NULL AND
                 exit_epoch IS NULL AND exit_quote IS NULL AND
                 exit_bid IS NULL AND exit_ask IS NULL)
            )
        ) WITHOUT ROWID, STRICT
        ;
        CREATE TRIGGER settlement_rows_no_update BEFORE UPDATE ON settlement_rows
        BEGIN SELECT RAISE(ABORT, 'immutable settlement_rows'); END;
        CREATE TRIGGER settlement_rows_no_delete BEFORE DELETE ON settlement_rows
        BEGIN SELECT RAISE(ABORT, 'immutable settlement_rows'); END;
        """
    )


def _settlement_values(row: Mapping[str, Any]) -> tuple[Any, ...]:
    settled = row["status"] == "SETTLED"
    return (
        row["prereg_id"],
        int(row["decision_close_epoch"]),
        row["status"],
        row["reason"],
        int(row["entry_epoch"]) if settled else None,
        float(row["entry_quote"]) if settled else None,
        float(row["entry_bid"]) if settled else None,
        float(row["entry_ask"]) if settled else None,
        int(row["exit_epoch"]) if settled else None,
        float(row["exit_quote"]) if settled else None,
        float(row["exit_bid"]) if settled else None,
        float(row["exit_ask"]) if settled else None,
    )


def _assert_settlement_rows_exact(
    conn: sqlite3.Connection, settlement: Sequence[Mapping[str, Any]]
) -> None:
    columns = (
        "prereg_id",
        "decision_close_epoch",
        "status",
        "reason",
        "entry_epoch",
        "entry_quote",
        "entry_bid",
        "entry_ask",
        "exit_epoch",
        "exit_quote",
        "exit_bid",
        "exit_ask",
    )
    actual = [
        tuple(row[name] for name in columns)
        for row in conn.execute(
            "SELECT * FROM settlement_rows ORDER BY prereg_id, decision_close_epoch"
        )
    ]
    expected = sorted(
        (_settlement_values(row) for row in settlement),
        key=lambda row: (row[0], row[1]),
    )
    if actual != expected:
        raise ProspectiveError("packet settlement rows differ from mechanical selection")


def _assert_packet_content(
    conn: sqlite3.Connection, prereg_id: str, expected_rows: int | None = None
) -> None:
    _assert_trial_schema(conn, packet=True)
    if (
        int(conn.execute("PRAGMA application_id").fetchone()[0]) != 1431119105
        or int(conn.execute("PRAGMA user_version").fetchone()[0]) != 1
    ):
        raise ProspectiveError("packet SQLite identity differs")
    if conn.execute("PRAGMA foreign_key_check").fetchone() is not None:
        raise ProspectiveError("packet foreign-key check failed")
    metadata = conn.execute("SELECT * FROM trial_metadata").fetchall()
    if len(metadata) != 1 or metadata[0]["prereg_id"] != prereg_id:
        raise ProspectiveError("packet metadata identity differs")
    metadata_raw = metadata[0]["metadata_json"].encode("utf-8")
    if (
        _sha256_bytes(metadata_raw) != metadata[0]["metadata_sha256"]
        or metadata[0]["schema"] != TRIAL_METADATA_SCHEMA
        or metadata[0]["singleton"] != 1
        or not isinstance(metadata[0]["created_epoch"], int)
        or metadata[0]["created_epoch"] < 0
    ):
        raise ProspectiveError("packet metadata payload differs")
    try:
        metadata_value = evidence.decode_canonical_json(metadata_raw)
    except evidence.EvidenceError as exc:
        raise ProspectiveError(f"packet metadata is not canonical JSON: {exc}") from exc
    if (
        not isinstance(metadata_value, dict)
        or metadata_value.get("schema") != TRIAL_METADATA_SCHEMA
        or metadata_value.get("prereg_id") != prereg_id
        or metadata_value.get("outcome_fields_present") is not False
        or metadata_value.get("activation") is not False
    ):
        raise ProspectiveError("packet metadata lifecycle differs")
    seals = conn.execute("SELECT prereg_id FROM trial_seal").fetchall()
    if len(seals) != 1 or seals[0]["prereg_id"] != prereg_id:
        raise ProspectiveError("packet trial seal differs")
    if conn.execute("SELECT 1 FROM trial_invalid").fetchone() is not None:
        raise ProspectiveError("valid sealed packet contains a protocol-INVALID marker")
    decision_count = int(conn.execute("SELECT COUNT(*) FROM decisions").fetchone()[0])
    settlement_count = int(conn.execute("SELECT COUNT(*) FROM settlement_rows").fetchone()[0])
    if settlement_count != decision_count or (
        expected_rows is not None and decision_count != int(expected_rows)
    ):
        raise ProspectiveError("packet decision/settlement row counts differ")
    _load_decisions(conn, prereg_id)
    _load_trial_seal(conn, prereg_id)
    for row in conn.execute(
        """
        SELECT entry_epoch, entry_quote, entry_bid, entry_ask,
               exit_epoch, exit_quote, exit_bid, exit_ask
        FROM settlement_rows WHERE status='SETTLED'
        """
    ):
        values = [row[name] for name in row.keys()]
        if (
            any(value is None for value in values)
            or not all(math.isfinite(float(value)) and float(value) > 0 for value in values[1:4] + values[5:8])
            or int(values[4]) <= int(values[0])
        ):
            raise ProspectiveError("packet contains a malformed settled row")


def _open_packet_readonly(path: Path) -> sqlite3.Connection:
    _regular_file(path, "sealed source packet")
    uri = f"file:{path}?mode=ro&immutable=1"
    conn = sqlite3.connect(uri, uri=True, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only=ON")
    if conn.execute("PRAGMA query_only").fetchone()[0] != 1:
        conn.close()
        raise ProspectiveError("packet SQLite query_only did not engage")
    return conn


def _validate_packet_file(
    root: Path,
    path: Path,
    prereg_id: str,
    *,
    expected_rows: int | None = None,
    expected_settlement: Sequence[Mapping[str, Any]] | None = None,
) -> evidence.ArtifactRef:
    if any(Path(str(path) + suffix).exists() for suffix in ("-wal", "-shm", "-journal")):
        raise ProspectiveError("sealed source packet has a SQLite sidecar")
    packet_info = _regular_file(path, "sealed source packet")
    if packet_info.st_nlink != 1:
        raise ProspectiveError("sealed source packet must have exactly one hard link")
    if packet_info.st_mode & 0o222:
        raise ProspectiveError("sealed source packet must be read-only")
    conn = _open_packet_readonly(path)
    try:
        if int(conn.execute("PRAGMA application_id").fetchone()[0]) != 1431119105:
            raise ProspectiveError("packet application_id differs")
        if int(conn.execute("PRAGMA user_version").fetchone()[0]) != 1:
            raise ProspectiveError("packet user_version differs")
        _assert_packet_content(conn, prereg_id, expected_rows)
        if expected_settlement is not None:
            _assert_settlement_rows_exact(conn, expected_settlement)
    finally:
        conn.close()
    relative = path.relative_to(root).as_posix()
    return evidence.ArtifactRef(
        path=relative,
        bytes=int(path.stat().st_size),
        sha256=_sha256_file(path),
    )


def _build_packet(
    *,
    root: Path,
    source_conn: sqlite3.Connection,
    paths: Mapping[str, Path],
    prereg_id: str,
    settlement: Sequence[Mapping[str, Any]],
) -> evidence.ArtifactRef:
    destination = paths["packet"]
    stage = paths["packet_stage"]
    if destination.exists() or destination.is_symlink():
        if destination.is_symlink():
            raise ProspectiveError("sealed packet destination is a symlink")
        destination_info = _regular_file(destination, "sealed packet destination")
        stale_stage = False
        if stage.exists() or stage.is_symlink():
            if stage.is_symlink():
                raise ProspectiveError("packet recovery stage is a symlink")
            stage_info = _regular_file(stage, "packet recovery stage")
            if (
                stage_info.st_dev == destination_info.st_dev
                and stage_info.st_ino == destination_info.st_ino
            ):
                stage.unlink()
                directory = os.open(
                    destination.parent,
                    os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
                )
                try:
                    os.fsync(directory)
                finally:
                    os.close(directory)
            elif stage_info.st_nlink == 1 and stage_info.st_mode & 0o222:
                stale_stage = True
            else:
                raise ProspectiveError("packet recovery stage conflicts with destination")
        packet_ref = _validate_packet_file(
            root,
            destination,
            prereg_id,
            expected_rows=len(settlement),
            expected_settlement=settlement,
        )
        if stale_stage:
            stage.unlink()
            directory = os.open(
                destination.parent,
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
            )
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        return packet_ref
    if stage.exists() or stage.is_symlink():
        if stage.is_symlink() or not stage.is_file():
            raise ProspectiveError("packet recovery stage is not a regular file")
        stage_info = stage.stat(follow_symlinks=False)
        if stage_info.st_nlink != 1:
            raise ProspectiveError("packet recovery stage has foreign hard links")
        complete = _packet_stage_is_complete(stage, prereg_id, settlement)
        if complete:
            stage.chmod(0o400)
            descriptor = os.open(stage, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            try:
                os.link(stage, destination, follow_symlinks=False)
            except FileExistsError as exc:
                raise ProspectiveError(
                    "strict-new packet destination already exists"
                ) from exc
            directory = os.open(
                destination.parent,
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
            )
            try:
                os.fsync(directory)
                stage.unlink()
                os.fsync(directory)
            finally:
                os.close(directory)
            return _validate_packet_file(
                root,
                destination,
                prereg_id,
                expected_rows=len(settlement),
                expected_settlement=settlement,
            )
        if not stage_info.st_mode & 0o222:
            raise ProspectiveError("finalized packet recovery stage is invalid")
        stage.unlink()
    for suffix in ("-wal", "-shm", "-journal"):
        sidecar = Path(str(stage) + suffix)
        if sidecar.exists() or sidecar.is_symlink():
            if sidecar.is_symlink() or not sidecar.is_file():
                raise ProspectiveError("packet recovery sidecar is malformed")
            sidecar.unlink()
    destination_conn = sqlite3.connect(str(stage), isolation_level=None)
    destination_conn.row_factory = sqlite3.Row
    try:
        source_conn.backup(destination_conn)
        destination_conn.execute("PRAGMA foreign_keys=ON")
        destination_conn.execute("PRAGMA synchronous=FULL")
        _create_settlement_table(destination_conn)
        destination_conn.execute("BEGIN IMMEDIATE")
        try:
            destination_conn.executemany(
                """
                INSERT INTO settlement_rows (
                    prereg_id, decision_close_epoch, status, reason,
                    entry_epoch, entry_quote, entry_bid, entry_ask,
                    exit_epoch, exit_quote, exit_bid, exit_ask
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [_settlement_values(row) for row in settlement],
            )
            destination_conn.execute("COMMIT")
        except Exception:
            _rollback(destination_conn)
            raise
        _assert_packet_content(destination_conn, prereg_id, len(settlement))
        mode = destination_conn.execute("PRAGMA journal_mode=DELETE").fetchone()[0]
        if str(mode).lower() != "delete":
            raise ProspectiveError("packet did not leave WAL mode")
    finally:
        destination_conn.close()
    if any(Path(str(stage) + suffix).exists() for suffix in ("-wal", "-shm", "-journal")):
        raise ProspectiveError("packet stage retained a SQLite sidecar")
    stage.chmod(0o400)
    descriptor = os.open(stage, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    try:
        os.link(stage, destination, follow_symlinks=False)
    except FileExistsError as exc:
        raise ProspectiveError("strict-new packet destination already exists") from exc
    directory = os.open(destination.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory)
        stage.unlink()
        os.fsync(directory)
    finally:
        os.close(directory)
    return _validate_packet_file(
        root,
        destination,
        prereg_id,
        expected_rows=len(settlement),
        expected_settlement=settlement,
    )


def _packet_stage_is_complete(
    stage: Path,
    prereg_id: str,
    settlement: Sequence[Mapping[str, Any]],
) -> bool:
    if any(Path(str(stage) + suffix).exists() for suffix in ("-wal", "-shm", "-journal")):
        return False
    try:
        conn = _open_packet_readonly(stage)
        try:
            _assert_packet_content(conn, prereg_id, len(settlement))
            _assert_settlement_rows_exact(conn, settlement)
        finally:
            conn.close()
    except (ProspectiveError, sqlite3.DatabaseError, OSError):
        return False
    return True


def _sealed_source_payload(
    spec: Mapping[str, Any],
    prereg: evidence.EvidenceEnvelope,
    collection: evidence.EvidenceEnvelope,
    packet_ref: evidence.ArtifactRef,
    seal: Mapping[str, Any],
    tick_descriptor: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": spec["evidence_kinds"]["sealed_source"],
        "prereg_id": prereg.object_id,
        "collection_access_receipt_id": collection.object_id,
        "source_artifact": packet_ref.as_dict(),
        "cutoff_epoch": seal["cutoff_epoch"],
        "trigger": seal["trigger"],
        "week": seal["week"],
        "outcome_blind_counts": seal["counts"],
        "selection": seal["selection"],
        "tick_source_descriptor": dict(tick_descriptor),
        "contains_prices": True,
        "efficacy_computed": False,
        "money": None,
        "activation": False,
    }


def _publish_sealed_source(
    *,
    root: Path,
    spec: Mapping[str, Any],
    prereg: evidence.EvidenceEnvelope,
    collection: evidence.EvidenceEnvelope,
    packet_ref: evidence.ArtifactRef,
    seal: Mapping[str, Any],
    tick_descriptor: Mapping[str, Any],
) -> evidence.EvidenceEnvelope:
    packet_ref.isolated_identity(repo_root=root)
    expected = evidence.EvidenceEnvelope.create(
        kind=spec["evidence_kinds"]["sealed_source"],
        payload=_sealed_source_payload(
            spec, prereg, collection, packet_ref, seal, tick_descriptor
        ),
        dependencies=[collection.object_id],
    )
    groups = _domain_objects(root, spec)
    existing = [
        item
        for item in groups[spec["evidence_kinds"]["sealed_source"]]
        if isinstance(item.payload, dict) and item.payload.get("prereg_id") == prereg.object_id
    ]
    if existing and (len(existing) != 1 or existing[0].as_dict() != expected.as_dict()):
        raise ProspectiveError("sealed source evidence is conflicting")
    evidence.publish(expected, repo_root=root)
    return expected


def _complete_seal_once(
    *,
    root: Path,
    prereg: evidence.EvidenceEnvelope,
    collection: evidence.EvidenceEnvelope,
    spec: Mapping[str, Any],
    conn: sqlite3.Connection,
    paths: Mapping[str, Path],
    seal: Mapping[str, Any],
) -> evidence.EvidenceEnvelope:
    decisions = [
        row
        for row in _load_decisions(conn, prereg.object_id)
        if int(row["decision_close_epoch"]) < int(seal["cutoff_epoch"])
    ]
    ticks, descriptor = _read_tick_rows(
        root,
        spec,
        start_epoch=int(prereg.payload["t0"]["epoch"]),
        end_epoch=int(seal["cutoff_epoch"]) - 1,
        required_intervals=_settlement_tick_intervals(decisions, spec),
    )
    settlement = _settlement_rows(decisions, ticks, spec)
    if (
        _selection_manifest(settlement) != seal["selection"]
        or _settlement_rows_sha256(settlement) != seal["settlement_sha256"]
    ):
        raise ProspectiveError("sealed tick-selection manifest changed after cutoff")
    packet_ref = _build_packet(
        root=root,
        source_conn=conn,
        paths=paths,
        prereg_id=prereg.object_id,
        settlement=settlement,
    )
    return _publish_sealed_source(
        root=root,
        spec=spec,
        prereg=prereg,
        collection=collection,
        packet_ref=packet_ref,
        seal=seal,
        tick_descriptor=descriptor,
    )


def _complete_seal(
    *,
    root: Path,
    prereg: evidence.EvidenceEnvelope,
    collection: evidence.EvidenceEnvelope,
    spec: Mapping[str, Any],
    conn: sqlite3.Connection,
    paths: Mapping[str, Path],
    seal: Mapping[str, Any],
) -> evidence.EvidenceEnvelope:
    try:
        return _complete_seal_once(
            root=root,
            prereg=prereg,
            collection=collection,
            spec=spec,
            conn=conn,
            paths=paths,
            seal=seal,
        )
    except SealedSourceProtocolError:
        raise
    except ProspectiveError as exc:
        raise SealedSourceProtocolError(str(exc)) from exc


def _seal_if_due(
    *,
    root: Path,
    prereg: evidence.EvidenceEnvelope,
    collection: evidence.EvidenceEnvelope,
    spec: Mapping[str, Any],
    conn: sqlite3.Connection,
    paths: Mapping[str, Path],
    now_epoch: int,
    evaluate_through_epoch: int | None = None,
) -> evidence.EvidenceEnvelope | None:
    observed_epoch = int(now_epoch)
    evaluation_epoch = (
        observed_epoch
        if evaluate_through_epoch is None
        else int(evaluate_through_epoch)
    )
    if evaluation_epoch > observed_epoch:
        raise ProspectiveError("look boundary is after the observation clock")
    existing = _load_trial_seal(conn, prereg.object_id)
    if existing is not None:
        return _complete_seal(
            root=root,
            prereg=prereg,
            collection=collection,
            spec=spec,
            conn=conn,
            paths=paths,
            seal=existing,
        )
    decisions = _load_decisions(conn, prereg.object_id)
    ticks, _ = _read_tick_rows(
        root,
        spec,
        start_epoch=int(prereg.payload["t0"]["epoch"]),
        end_epoch=evaluation_epoch,
        required_intervals=_settlement_tick_intervals(decisions, spec),
    )
    cutoff = _cutoff_decision(
        prereg, spec, decisions, ticks, now_epoch=evaluation_epoch
    )
    if cutoff is None:
        return None
    prefix = [
        row for row in decisions if int(row["decision_close_epoch"]) < int(cutoff["cutoff_epoch"])
    ]
    settlement = _settlement_rows(prefix, ticks, spec)
    cutoff = dict(cutoff)
    cutoff["selection"] = _selection_manifest(settlement)
    payload = _seal_payload(
        prereg.object_id,
        cutoff,
        settlement_sha256=_settlement_rows_sha256(settlement),
        sealed_epoch=observed_epoch,
    )
    _assert_write_authority(root, prereg.object_id)
    _insert_trial_seal_once(conn, payload)
    return _complete_seal(
        root=root,
        prereg=prereg,
        collection=collection,
        spec=spec,
        conn=conn,
        paths=paths,
        seal=payload,
    )


def _settled_correctness(direction: str, entry_quote: float, exit_quote: float) -> bool:
    if direction not in {"UP", "DOWN"}:
        raise ProspectiveError("settled direction must be UP or DOWN")
    if not all(
        isinstance(value, (int, float, np.integer, np.floating))
        and math.isfinite(float(value))
        and float(value) > 0
        for value in (entry_quote, exit_quote)
    ):
        raise ProspectiveError("settled quotes must be finite and positive")
    return bool(
        float(exit_quote) > float(entry_quote)
        if direction == "UP"
        else float(exit_quote) < float(entry_quote)
    )


def _canonical_return_from_settlement(
    decision_epoch: int,
    row: Mapping[str, Any],
    spec: Mapping[str, Any],
) -> float:
    entry_epoch = int(row["entry_epoch"])
    exit_epoch = int(row["exit_epoch"])
    entry_quote = float(row["entry_quote"])
    exit_quote = float(row["exit_quote"])
    observation_clock = np.asarray(
        [int(decision_epoch), entry_epoch, exit_epoch], dtype="int64"
    )
    closes = np.asarray([entry_quote, entry_quote, exit_quote], dtype="float64")
    try:
        returns, valid, exits = refresh_stats.canonical_settlement_with_exit_on_decision_grid(
            [int(decision_epoch)],
            observation_clock,
            closes,
            horizon_s=int(spec["horizon_seconds"]),
            tol_s=int(spec["runtime"]["settlement_tolerance_seconds"]),
            lag_s=int(spec["runtime"]["entry_lag_seconds"]),
        )
    except refresh_stats.StatisticsInputError as exc:
        raise ProspectiveError(f"canonical settlement validation failed: {exc}") from exc
    if not bool(valid[0]) or int(exits[0]) != exit_epoch:
        raise ProspectiveError("packet settlement differs from the canonical fixed-target rule")
    return float(returns[0])


def _endpoint_value(value: Any, name: str) -> float | None:
    if value is None:
        return None
    if not isinstance(value, (int, float, np.integer, np.floating)) or not math.isfinite(
        float(value)
    ):
        raise ProspectiveError(f"{name} must be finite or null")
    return float(value)


def _classify_result(
    spec: Mapping[str, Any],
    *,
    endpoint_results: Mapping[str, Mapping[str, Any]],
    counts: Mapping[str, Any],
    complete_months: Sequence[Mapping[str, Any]],
    protocol_errors: Sequence[str] = (),
) -> dict[str, Any]:
    names = list(spec["statistics"]["endpoint_order"])
    if list(endpoint_results) != names:
        raise ProspectiveError("endpoint result order differs from the registered family")
    if any(not isinstance(error, str) or not error for error in protocol_errors):
        raise ProspectiveError("protocol errors must be non-empty strings")
    minimum = int(spec["fixed_look"]["minimum_candidate_per_side"])
    candidate = counts.get("candidate")
    incumbent = counts.get("incumbent")
    common_settled = counts.get("common_settled")
    if (
        not isinstance(candidate, Mapping)
        or not isinstance(incumbent, Mapping)
        or not isinstance(common_settled, (int, np.integer))
        or int(common_settled) < 0
    ):
        raise ProspectiveError("scheduled counts are missing candidate or incumbent")
    required_counts: dict[str, int] = {}
    for policy, values in (("candidate", candidate), ("incumbent", incumbent)):
        for side in ("combined", "UP", "DOWN"):
            value = values.get(side)
            if not isinstance(value, (int, np.integer)) or int(value) < 0:
                raise ProspectiveError(f"{policy} {side} count is invalid")
            if policy == "candidate":
                required_counts[side] = int(value)
        if (
            int(values["combined"]) != int(values["UP"]) + int(values["DOWN"])
            or int(values["combined"]) > int(common_settled)
        ):
            raise ProspectiveError(f"{policy} scheduled counts are inconsistent")

    normalized: dict[str, dict[str, float | None]] = {}
    for name in names:
        row = endpoint_results[name]
        if not isinstance(row, Mapping):
            raise ProspectiveError(f"endpoint {name} is malformed")
        normalized[name] = {
            "point_estimate": _endpoint_value(
                row.get("point_estimate"), f"{name}.point_estimate"
            ),
            "simultaneous_lower_bound": _endpoint_value(
                row.get("simultaneous_lower_bound"),
                f"{name}.simultaneous_lower_bound",
            ),
        }

    side_statuses: dict[str, str] = {}
    for side, name in zip(("UP", "DOWN"), names[1:], strict=True):
        point = normalized[name]["point_estimate"]
        lower = normalized[name]["simultaneous_lower_bound"]
        count = required_counts[side]
        if protocol_errors:
            status = "INVALID"
        elif count >= minimum and lower is not None and lower > 0:
            status = "SUPPORTED_SHADOW"
        elif count >= minimum and point is not None and point <= 0:
            status = "REJECTED"
        else:
            status = "INCONCLUSIVE"
        side_statuses[side] = status

    lower_screen = all(
        normalized[name]["simultaneous_lower_bound"] is not None
        and normalized[name]["simultaneous_lower_bound"] > 0
        for name in names
    )
    count_screen = all(required_counts[side] >= minimum for side in ("UP", "DOWN"))
    incumbent_combined = int(incumbent["combined"])
    candidate_combined = required_counts["combined"]
    activity_ratio = (
        candidate_combined / incumbent_combined if incumbent_combined > 0 else None
    )
    activity_screen = (
        candidate_combined
        >= float(spec["statistics"]["activity_ratio_minimum"]) * incumbent_combined
    )
    month_screen = True
    for index, row in enumerate(complete_months):
        if not isinstance(row, Mapping):
            raise ProspectiveError(f"complete_months[{index}] is malformed")
        up = row.get("candidate_UP")
        down = row.get("candidate_DOWN")
        combined = row.get("candidate_combined")
        month_yield = _endpoint_value(
            row.get("candidate_whole_book_yield"),
            f"complete_months[{index}].candidate_whole_book_yield",
        )
        if (
            not isinstance(up, (int, np.integer))
            or not isinstance(down, (int, np.integer))
            or not isinstance(combined, (int, np.integer))
            or int(combined) != int(up) + int(down)
            or int(up) < 1
            or int(down) < 1
            or month_yield is None
            or month_yield < 0
        ):
            month_screen = False

    screens = {
        "protocol_integrity": not protocol_errors,
        "minimum_candidate_count_each_side": count_screen,
        "all_three_simultaneous_lower_bounds_above_zero": lower_screen,
        "candidate_activity_at_least_80pct_incumbent": activity_screen,
        "complete_month_coverage_and_nonnegative_yield": month_screen,
        "candidate_activity_ratio": activity_ratio,
    }
    if protocol_errors:
        terminal = "INVALID"
    elif all(
        (
            count_screen,
            lower_screen,
            activity_screen,
            month_screen,
        )
    ):
        terminal = "SHADOW_SURVIVOR_INACTIVE"
    elif count_screen and any(
        normalized[name]["point_estimate"] is not None
        and normalized[name]["point_estimate"] <= 0
        for name in names
    ):
        terminal = "SHADOW_REJECTED"
    else:
        terminal = "INCONCLUSIVE"
    return {
        "side_statuses": side_statuses,
        "terminal_status": terminal,
        "promotion_screens": screens,
    }


def _complete_month_rows(
    timestamps: np.ndarray,
    candidate_selected: np.ndarray,
    candidate_direction_up: np.ndarray,
    candidate_utility: np.ndarray,
    *,
    t0_epoch: int,
    cutoff_epoch: int,
) -> list[dict[str, Any]]:
    t0_local = datetime.fromtimestamp(int(t0_epoch), UTC).astimezone(NY)
    cutoff_local = datetime.fromtimestamp(int(cutoff_epoch), UTC).astimezone(NY)
    first = (t0_local.replace(day=1, hour=0, minute=0, second=0, microsecond=0) + timedelta(days=32)).replace(day=1)
    rows: list[dict[str, Any]] = []
    local_index = pd.to_datetime(timestamps, unit="s", utc=True).tz_convert(NY)
    while first < cutoff_local:
        following = (first + timedelta(days=32)).replace(day=1)
        if following > cutoff_local:
            break
        mask = np.asarray(
            (local_index.year == first.year) & (local_index.month == first.month),
            dtype=bool,
        )
        selected = candidate_selected & mask
        up = int(np.sum(selected & candidate_direction_up))
        down = int(np.sum(selected & ~candidate_direction_up))
        combined = int(np.sum(selected))
        rows.append(
            {
                "month": f"{first.year:04d}-{first.month:02d}",
                "candidate_UP": up,
                "candidate_DOWN": down,
                "candidate_combined": combined,
                "candidate_whole_book_yield": (
                    float(np.sum(candidate_utility[mask])) / combined
                    if combined
                    else None
                ),
            }
        )
        first = following
    return rows


def _noncomputable_family(
    endpoints: Sequence[refresh_stats.DateRatioEndpoint],
    master_calendar: np.ndarray,
    spec: Mapping[str, Any],
) -> dict[str, Any]:
    alpha = float(spec["statistics"]["alpha"])
    family_size = len(endpoints)
    reason = "insufficient_represented_dates_for_registered_block_lengths"
    return {
        "schema": "m15-book-refresh-family-inference/v1",
        "family": "prospective_shadow_3",
        "tail": "lower_only",
        "family_size": family_size,
        "alpha": alpha,
        "bootstrap_replicates": int(spec["statistics"]["bootstrap_replicates"]),
        "bootstrap_seed": int(spec["statistics"]["bootstrap_seed"]),
        "block_lengths": list(spec["statistics"]["block_lengths"]),
        "bonferroni_floor": float(NormalDist().inv_cdf(1.0 - alpha / family_size)),
        "master_calendar": [int(value) for value in master_calendar],
        "endpoints": [
            {
                "name": endpoint.name,
                "estimate": endpoint.estimate,
                "observed_denominator": endpoint.observed_denominator,
                "computable": False,
                "noncomputable_reasons": [reason],
                "lower": None,
                "upper": None,
                "by_length": [
                    {
                        "block_length": int(length),
                        "standard_error": None,
                        "critical_value": None,
                        "lower": None,
                        "upper": None,
                    }
                    for length in spec["statistics"]["block_lengths"]
                ],
            }
            for endpoint in endpoints
        ],
    }


def _reduce_packet(
    conn: sqlite3.Connection,
    *,
    spec: Mapping[str, Any],
    prereg: evidence.EvidenceEnvelope,
    sealed_source: evidence.EvidenceEnvelope,
) -> dict[str, Any]:
    source = sealed_source.payload
    expected_rows = int(source["selection"]["rows"])
    _assert_packet_content(conn, prereg.object_id, expected_rows)
    if "source_paths" in prereg.payload and "preflight" in prereg.payload:
        metadata_raw = conn.execute(
            "SELECT metadata_json FROM trial_metadata WHERE singleton=1"
        ).fetchone()[0].encode("utf-8")
        metadata = evidence.decode_canonical_json(metadata_raw)
        metadata_fields = {
            "schema",
            "prereg_id",
            "collection_access_receipt_id",
            "h0_git_sha",
            "h1_git_sha",
            "t0",
            "pair",
            "horizon_seconds",
            "candidate_id",
            "incumbent_id",
            "trial_path",
            "source_paths",
            "preflight",
            "observer",
            "service",
            "host",
            "runtime_environment",
            "outcome_fields_present",
            "activation",
        }
        metadata = _exact_fields(metadata, metadata_fields, "trial metadata payload")
        if (
            metadata["prereg_id"] != prereg.object_id
            or metadata["collection_access_receipt_id"]
            != source["collection_access_receipt_id"]
            or metadata["h0_git_sha"] != prereg.payload["h0_git_sha"]
            or not isinstance(metadata["h1_git_sha"], str)
            or HEX40.fullmatch(metadata["h1_git_sha"]) is None
            or metadata["t0"] != prereg.payload["t0"]
            or metadata["pair"] != PAIR
            or metadata["horizon_seconds"] != spec["horizon_seconds"]
            or metadata["candidate_id"] != spec["candidate"]["artifact_id"]
            or metadata["incumbent_id"] != spec["incumbent"]["book_id"]
            or metadata["trial_path"] != _trial_relative(spec, prereg.object_id)
            or metadata["source_paths"] != prereg.payload["source_paths"]
            or metadata["preflight"] != prereg.payload["preflight"]
            or metadata["observer"] != SCRIPT_PATH
            or metadata["service"] != SERVICE_PATH
            or metadata["runtime_environment"]
            != prereg.payload["runtime_environment"]
            or metadata["outcome_fields_present"] is not False
            or metadata["activation"] is not False
        ):
            raise ProspectiveError("trial metadata differs from preregistered authority")
    decisions = _load_decisions(conn, prereg.object_id)
    seal = _load_trial_seal(conn, prereg.object_id)
    if seal is None:
        raise ProspectiveError("sealed packet lacks its trial seal")
    if (
        seal["cutoff_epoch"] != source["cutoff_epoch"]
        or seal["week"] != source["week"]
        or seal["trigger"] != source["trigger"]
        or seal["counts"] != source["outcome_blind_counts"]
        or seal["selection"] != source["selection"]
    ):
        raise ProspectiveError("sealed source and packet seal differ")
    cutoff_epoch = int(source["cutoff_epoch"])
    expected_clock = _expected_decision_epochs(
        int(prereg.payload["t0"]["epoch"]), cutoff_epoch - 1
    )
    actual_clock = np.asarray(
        [int(row["decision_close_epoch"]) for row in decisions], dtype="int64"
    )
    if not np.array_equal(actual_clock, expected_clock):
        raise ProspectiveError("packet does not contain the exact preregistered decision grid")
    expected_provider = (
        _provider_identity(prereg)
        if "preflight" in prereg.payload and "source_paths" in prereg.payload
        else None
    )
    preflight_scores = (
        prereg.payload["preflight"]["payload"]["scoring_parity"]
        if expected_provider is not None
        else None
    )
    for decision in decisions:
        if expected_provider is not None and decision["provider"] != expected_provider:
            raise ProspectiveError("decision provider identity differs from preregistration")
        if not decision["common_scoreable"]:
            continue
        for policy, book_id in (
            ("candidate", spec["candidate"]["artifact_id"]),
            ("incumbent", spec["incumbent"]["book_id"]),
        ):
            score = decision[policy]
            if (
                score["book_id"] != book_id
                or score["threshold"] != spec[policy]["confidence_threshold"]
                or (
                    preflight_scores is not None
                    and score["content_id"] != preflight_scores[policy]["content_id"]
                )
            ):
                raise ProspectiveError(f"decision {policy} fixed identity differs")

    settlement_rows = [
        dict(row)
        for row in conn.execute(
            "SELECT * FROM settlement_rows WHERE prereg_id=? ORDER BY decision_close_epoch",
            (prereg.object_id,),
        )
    ]
    if _settlement_rows_sha256(settlement_rows) != seal["settlement_sha256"]:
        raise ProspectiveError("packet settlement bytes differ from the immutable seal")
    settlement_clock = np.asarray(
        [int(row["decision_close_epoch"]) for row in settlement_rows], dtype="int64"
    )
    if not np.array_equal(actual_clock, settlement_clock):
        raise ProspectiveError("packet settlement rows are not aligned with decisions")
    manifest_rows: list[dict[str, Any]] = []
    common_positions: list[int] = []
    returns: list[float] = []
    for position, (decision, settled) in enumerate(zip(decisions, settlement_rows, strict=True)):
        status = settled["status"]
        manifest = {
            "prereg_id": prereg.object_id,
            "decision_close_epoch": int(settled["decision_close_epoch"]),
            "status": status,
            "reason": settled["reason"],
        }
        if status == "SETTLED":
            if not decision["common_scoreable"]:
                raise ProspectiveError("an unscoreable decision was marked settled")
            for name in (
                "entry_epoch",
                "entry_quote",
                "entry_bid",
                "entry_ask",
                "exit_epoch",
                "exit_quote",
                "exit_bid",
                "exit_ask",
            ):
                manifest[name] = settled[name]
            returns.append(
                _canonical_return_from_settlement(
                    int(decision["decision_close_epoch"]), settled, spec
                )
            )
            common_positions.append(position)
        elif status == "NOT_COMMON_SCOREABLE":
            if decision["common_scoreable"] or settled["reason"] != decision["exclusion_reason"]:
                raise ProspectiveError("unscoreable settlement reason differs from its decision")
        elif status == "UNSETTLED":
            if not decision["common_scoreable"]:
                raise ProspectiveError("unsettled row does not belong to the common scoreable grid")
        else:
            raise ProspectiveError("packet settlement status differs")
        manifest_rows.append(manifest)
    if _selection_manifest(manifest_rows) != source["selection"]:
        raise ProspectiveError("packet settlement manifest differs from the source seal")

    common = np.asarray(common_positions, dtype="int64")
    common_timestamps = actual_clock[common]
    common_returns = np.asarray(returns, dtype="float64")
    arms: dict[str, refresh_stats.ScheduledArm] = {}
    for policy in ("candidate", "incumbent"):
        rows = [decisions[index][policy] for index in common_positions]
        probabilities = np.asarray([float(row["probability"]) for row in rows])
        structural = np.asarray(
            [bool(row["structural_gate_passed"]) for row in rows], dtype=bool
        )
        threshold = float(spec[policy]["confidence_threshold"])
        persisted_gate = np.asarray([bool(row["gate_passed"]) for row in rows], dtype=bool)
        expected_gate = structural & (np.abs(probabilities - 0.5) >= threshold)
        if not np.array_equal(persisted_gate, expected_gate):
            raise ProspectiveError(f"{policy} persisted gate differs from the fixed rule")
        for row, probability in zip(rows, probabilities, strict=True):
            if row["direction"] != ("UP" if probability >= 0.5 else "DOWN"):
                raise ProspectiveError(f"{policy} direction differs from its score")
        try:
            arms[policy] = refresh_stats.scheduled_arm(
                common_timestamps,
                probabilities,
                common_returns,
                threshold=threshold,
                structural_gate=structural,
                gap_s=int(spec["horizon_seconds"]),
            )
        except refresh_stats.StatisticsInputError as exc:
            raise ProspectiveError(f"{policy} scheduling failed: {exc}") from exc

    counts = {"common_settled": int(len(common))}
    for policy, arm in arms.items():
        counts[policy] = {
            "combined": int(np.sum(arm.selected_mask)),
            "UP": int(np.sum(arm.selected_mask & arm.direction_up)),
            "DOWN": int(np.sum(arm.selected_mask & ~arm.direction_up)),
        }
    if counts != source["outcome_blind_counts"]:
        raise ProspectiveError("analysis schedule differs from outcome-blind sealed counts")

    candidate_selected = arms["candidate"].selected_mask
    candidate_direction = arms["candidate"].direction_up
    candidate_utility = arms["candidate"].utility
    incumbent_utility = arms["incumbent"].utility
    endpoint_names = list(spec["statistics"]["endpoint_order"])
    breakeven = float(spec["statistics"]["breakeven"])
    endpoints = [
        refresh_stats.endpoint_from_rows(
            endpoint_names[0],
            common_timestamps,
            candidate_utility - incumbent_utility,
            np.ones(len(common_timestamps), dtype="float64"),
        )
    ]
    for side, direction_up, name in zip(
        ("UP", "DOWN"), (True, False), endpoint_names[1:], strict=True
    ):
        side_mask = candidate_selected & (candidate_direction == direction_up)
        correctness = arms["candidate"].correct.astype("float64")
        endpoints.append(
            refresh_stats.endpoint_from_rows(
                name,
                common_timestamps,
                np.where(side_mask, correctness - breakeven, 0.0),
                side_mask.astype("float64"),
            )
        )
    master_calendar = np.unique(refresh_stats.ny_date_keys(common_timestamps))
    if len(master_calendar) < max(spec["statistics"]["block_lengths"]):
        inference_value = _noncomputable_family(endpoints, master_calendar, spec)
    else:
        try:
            inference_value = refresh_stats.infer_shadow_family(
                endpoints,
                master_calendar=master_calendar,
                k_shadow=1,
                bootstrap_seed=int(spec["statistics"]["bootstrap_seed"]),
                block_lengths=spec["statistics"]["block_lengths"],
                bootstrap_replicates=int(spec["statistics"]["bootstrap_replicates"]),
                alpha=float(spec["statistics"]["alpha"]),
            ).as_dict()
        except refresh_stats.StatisticsInputError as exc:
            raise ProspectiveError(f"registered family inference failed: {exc}") from exc
    endpoint_results: dict[str, dict[str, Any]] = {}
    for endpoint in inference_value["endpoints"]:
        endpoint_results[endpoint["name"]] = {
            "point_estimate": endpoint["estimate"],
            "simultaneous_lower_bound": endpoint["lower"],
            "observed_denominator": endpoint["observed_denominator"],
            "computable": endpoint["computable"],
            "noncomputable_reasons": list(endpoint["noncomputable_reasons"]),
        }
    complete_months = _complete_month_rows(
        common_timestamps,
        candidate_selected,
        candidate_direction,
        candidate_utility,
        t0_epoch=int(prereg.payload["t0"]["epoch"]),
        cutoff_epoch=cutoff_epoch,
    )
    classification = _classify_result(
        spec,
        endpoint_results=endpoint_results,
        counts=counts,
        complete_months=complete_months,
    )
    return {
        "schema": RESULT_SCHEMA,
        "issue": spec["issue"],
        "prereg_id": prereg.object_id,
        "sealed_source_id": sealed_source.object_id,
        "analysis_access_receipt_id": None,
        "pair": PAIR,
        "horizon_seconds": int(spec["horizon_seconds"]),
        "candidate_id": spec["candidate"]["artifact_id"],
        "incumbent_id": spec["incumbent"]["book_id"],
        "look": {
            "t0": prereg.payload["t0"],
            "cutoff_epoch": cutoff_epoch,
            "week": int(source["week"]),
            "trigger": source["trigger"],
            "one_look": True,
        },
        "evidence_class": "OUTCOME_ONLY_TICK_PROXY",
        "selection": source["selection"],
        "schedule_counts": counts,
        "endpoint_results": endpoint_results,
        "inference": inference_value,
        "complete_months": complete_months,
        **classification,
        "status": classification["terminal_status"],
        "protocol_errors": [],
        "claim_limit": spec["claim_limit"],
        "money": None,
        "activation": False,
    }


def _matching_objects(
    groups: Mapping[str, Sequence[evidence.EvidenceEnvelope]],
    kind: str,
    prereg_id: str,
) -> list[evidence.EvidenceEnvelope]:
    return [
        item
        for item in groups[kind]
        if isinstance(item.payload, dict) and item.payload.get("prereg_id") == prereg_id
    ]


def _expected_packet_relative(spec: Mapping[str, Any], prereg_id: str) -> str:
    database = PurePosixPath(_trial_relative(spec, prereg_id))
    return (database.parent / "sealed_source.sqlite").as_posix()


def _validated_sealed_source(
    root: Path,
    prereg: evidence.EvidenceEnvelope,
    collection: evidence.EvidenceEnvelope,
    spec: Mapping[str, Any],
    *,
    groups: Mapping[str, Sequence[evidence.EvidenceEnvelope]] | None = None,
) -> tuple[evidence.EvidenceEnvelope, evidence.ArtifactRef]:
    objects = groups or _domain_objects(root, spec)
    matches = _matching_objects(
        objects, spec["evidence_kinds"]["sealed_source"], prereg.object_id
    )
    if len(matches) != 1:
        raise ProspectiveError(f"expected exactly one sealed source, found {len(matches)}")
    sealed = matches[0]
    fields = {
        "schema",
        "prereg_id",
        "collection_access_receipt_id",
        "source_artifact",
        "cutoff_epoch",
        "trigger",
        "week",
        "outcome_blind_counts",
        "selection",
        "tick_source_descriptor",
        "contains_prices",
        "efficacy_computed",
        "money",
        "activation",
    }
    payload = _exact_fields(sealed.payload, fields, "sealed source payload")
    if (
        sealed.kind != spec["evidence_kinds"]["sealed_source"]
        or sealed.artifacts
        or sealed.dependencies != (collection.object_id,)
        or payload["schema"] != sealed.kind
        or payload["prereg_id"] != prereg.object_id
        or payload["collection_access_receipt_id"] != collection.object_id
        or payload["contains_prices"] is not True
        or payload["efficacy_computed"] is not False
        or payload["money"] is not None
        or payload["activation"] is not False
    ):
        raise ProspectiveError("sealed source lifecycle or ancestry differs")
    try:
        source_ref = evidence.ArtifactRef.from_dict(payload["source_artifact"])
    except evidence.EvidenceError as exc:
        raise ProspectiveError(f"sealed source artifact reference is malformed: {exc}") from exc
    if source_ref.path != _expected_packet_relative(spec, prereg.object_id):
        raise ProspectiveError("sealed source packet path differs")
    week = payload["week"]
    cutoff_epoch = payload["cutoff_epoch"]
    if (
        not isinstance(week, int)
        or not int(spec["fixed_look"]["earliest_weeks"])
        <= week
        <= int(spec["fixed_look"]["cap_weeks"])
        or not isinstance(cutoff_epoch, int)
        or _week_boundaries(prereg, spec)[week - 1][1] != cutoff_epoch
    ):
        raise ProspectiveError("sealed source cutoff boundary differs")
    counts = payload["outcome_blind_counts"]
    selection = payload["selection"]
    if (
        not isinstance(counts, dict)
        or not isinstance(selection, dict)
        or set(selection)
        != {"rows", "settled", "unsettled", "not_common_scoreable", "timestamp_manifest_sha256"}
        or any(
            not isinstance(selection[name], int) or selection[name] < 0
            for name in ("rows", "settled", "unsettled", "not_common_scoreable")
        )
        or selection["rows"]
        != selection["settled"] + selection["unsettled"] + selection["not_common_scoreable"]
    ):
        raise ProspectiveError("sealed source selection summary differs")
    _hex64(selection["timestamp_manifest_sha256"], "selection manifest")
    for policy in ("candidate", "incumbent"):
        policy_counts = counts.get(policy)
        if (
            not isinstance(policy_counts, dict)
            or set(policy_counts) != {"combined", "UP", "DOWN"}
            or any(not isinstance(value, int) or value < 0 for value in policy_counts.values())
            or policy_counts["combined"] != policy_counts["UP"] + policy_counts["DOWN"]
        ):
            raise ProspectiveError(f"sealed source {policy} counts differ")
    if counts.get("common_settled") != selection["settled"]:
        raise ProspectiveError("sealed source common-settled count differs")
    descriptor = _exact_fields(
        payload["tick_source_descriptor"],
        {
            "source",
            "progress_path",
            "progress_sha256",
            "last_persisted_epoch",
            "shards_seen",
            "shard_metadata_sha256",
        },
        "sealed source tick descriptor",
    )
    if (
        descriptor["source"] != spec["paths"]["tick_shards"]
        or descriptor["progress_path"] != spec["paths"]["tick_progress"]
        or not isinstance(descriptor["last_persisted_epoch"], int)
        or not isinstance(descriptor["shards_seen"], int)
        or descriptor["shards_seen"] < 0
    ):
        raise ProspectiveError("sealed source tick descriptor differs")
    _hex64(descriptor["progress_sha256"], "sealed progress sha256")
    _hex64(descriptor["shard_metadata_sha256"], "sealed shard metadata sha256")
    minimum = int(spec["fixed_look"]["minimum_candidate_per_side"])
    ready = counts["candidate"]["UP"] >= minimum and counts["candidate"]["DOWN"] >= minimum
    if (
        payload["trigger"] not in {"COUNT_TRIGGER", "WEEK_26_CAP"}
        or (payload["trigger"] == "COUNT_TRIGGER" and not ready)
        or (
            payload["trigger"] == "WEEK_26_CAP"
            and (week != int(spec["fixed_look"]["cap_weeks"]) or ready)
        )
    ):
        raise ProspectiveError("sealed source stopping trigger differs")
    return sealed, source_ref


def _analysis_receipt_payload(
    spec: Mapping[str, Any],
    prereg: evidence.EvidenceEnvelope,
    sealed: evidence.EvidenceEnvelope,
    source_ref: evidence.ArtifactRef,
) -> dict[str, Any]:
    return {
        "schema": spec["evidence_kinds"]["analysis_access_receipt"],
        "prereg_id": prereg.object_id,
        "sealed_source_id": sealed.object_id,
        "source_artifact": source_ref.as_dict(),
        "authorized_actor": "usdchf_m15_prospective_v1.fixed_result_reducer",
        "allowed_observations": "one_verified_sealed_packet_read_and_registered_fixed_analysis",
        "strict_new": True,
        "money": None,
        "activation": False,
    }


def _expected_analysis_receipt(
    spec: Mapping[str, Any],
    prereg: evidence.EvidenceEnvelope,
    sealed: evidence.EvidenceEnvelope,
    source_ref: evidence.ArtifactRef,
) -> evidence.EvidenceEnvelope:
    return evidence.EvidenceEnvelope.create(
        kind=spec["evidence_kinds"]["analysis_access_receipt"],
        payload=_analysis_receipt_payload(spec, prereg, sealed, source_ref),
        dependencies=[sealed.object_id],
    )


def _validated_analysis_receipt(
    groups: Mapping[str, Sequence[evidence.EvidenceEnvelope]],
    spec: Mapping[str, Any],
    prereg: evidence.EvidenceEnvelope,
    sealed: evidence.EvidenceEnvelope,
    source_ref: evidence.ArtifactRef,
) -> evidence.EvidenceEnvelope:
    matches = _matching_objects(
        groups, spec["evidence_kinds"]["analysis_access_receipt"], prereg.object_id
    )
    expected = _expected_analysis_receipt(spec, prereg, sealed, source_ref)
    if len(matches) != 1 or matches[0].as_dict() != expected.as_dict():
        raise ProspectiveError("analysis access receipt is missing or conflicting")
    return matches[0]


RESULT_FIELDS = {
    "schema",
    "issue",
    "prereg_id",
    "sealed_source_id",
    "analysis_access_receipt_id",
    "pair",
    "horizon_seconds",
    "candidate_id",
    "incumbent_id",
    "look",
    "evidence_class",
    "selection",
    "schedule_counts",
    "endpoint_results",
    "inference",
    "complete_months",
    "side_statuses",
    "terminal_status",
    "promotion_screens",
    "status",
    "protocol_errors",
    "claim_limit",
    "money",
    "activation",
}


def _validate_result_value(
    value: Any,
    spec: Mapping[str, Any],
    prereg_id: str,
    sealed_source_id: str | None,
    analysis_receipt_id: str | None,
    *,
    prereg: evidence.EvidenceEnvelope,
    sealed: evidence.EvidenceEnvelope | None,
) -> dict[str, Any]:
    result = _exact_fields(value, RESULT_FIELDS, "prospective result")
    if (
        result["schema"] != RESULT_SCHEMA
        or result["issue"] != spec["issue"]
        or result["prereg_id"] != prereg_id
        or result["sealed_source_id"] != sealed_source_id
        or result["analysis_access_receipt_id"] != analysis_receipt_id
        or result["pair"] != PAIR
        or result["horizon_seconds"] != int(spec["horizon_seconds"])
        or result["candidate_id"] != spec["candidate"]["artifact_id"]
        or result["incumbent_id"] != spec["incumbent"]["book_id"]
        or result["evidence_class"] != "OUTCOME_ONLY_TICK_PROXY"
        or result["status"] not in TERMINAL_STATUSES
        or result["terminal_status"] != result["status"]
        or result["money"] is not None
        or result["activation"] is not False
        or result["claim_limit"] != spec["claim_limit"]
    ):
        raise ProspectiveError("prospective result fixed identity differs")
    no_source_invalid = sealed_source_id is None and analysis_receipt_id is None
    if (sealed_source_id is None) != (analysis_receipt_id is None):
        raise ProspectiveError("prospective result source/access lifecycle differs")
    look = _exact_fields(
        result["look"],
        {"t0", "cutoff_epoch", "week", "trigger", "one_look"},
        "prospective result look",
    )
    _validate_registered_t0(look["t0"])
    if look["t0"] != prereg.payload["t0"] or prereg.object_id != prereg_id:
        raise ProspectiveError("prospective result T0 differs from preregistration")
    if no_source_invalid:
        if (
            sealed is not None
            or result["status"] != "INVALID"
            or result["selection"] is not None
            or look["cutoff_epoch"] is not None
            or look["week"] is not None
            or look["trigger"] != "PROTOCOL_INVALID"
            or look["one_look"] is not True
        ):
            raise ProspectiveError("no-source INVALID result lifecycle differs")
    else:
        _hex64(sealed_source_id, "result sealed_source_id")
        _hex64(analysis_receipt_id, "result analysis_access_receipt_id")
        if (
            sealed is None
            or sealed.object_id != sealed_source_id
            or look["cutoff_epoch"] != sealed.payload["cutoff_epoch"]
            or look["week"] != sealed.payload["week"]
            or look["trigger"] != sealed.payload["trigger"]
            or look["one_look"] is not True
            or result["selection"] != sealed.payload["selection"]
            or result["schedule_counts"]
            != sealed.payload["outcome_blind_counts"]
        ):
            raise ProspectiveError(
                "prospective result context differs from the sealed source"
            )
    endpoint_order = list(spec["statistics"]["endpoint_order"])
    if not isinstance(result["endpoint_results"], dict) or set(
        result["endpoint_results"]
    ) != set(endpoint_order):
        raise ProspectiveError("prospective result endpoint inventory differs")
    inference = result["inference"]
    if inference is not None:
        inference_rows = inference.get("endpoints") if isinstance(inference, dict) else None
        if (
            not isinstance(inference_rows, list)
            or [row.get("name") for row in inference_rows if isinstance(row, dict)]
            != endpoint_order
            or len(inference_rows) != len(endpoint_order)
        ):
            raise ProspectiveError("prospective result inference order differs")
    if not isinstance(result["side_statuses"], dict) or set(result["side_statuses"]) != {"UP", "DOWN"} or any(
        status not in {"SUPPORTED_SHADOW", "REJECTED", "INCONCLUSIVE", "INVALID"}
        for status in result["side_statuses"].values()
    ):
        raise ProspectiveError("prospective result side statuses differ")
    errors = result["protocol_errors"]
    if not isinstance(errors, list) or any(not isinstance(row, str) or not row for row in errors):
        raise ProspectiveError("prospective result protocol errors differ")
    if (result["status"] == "INVALID") != bool(errors):
        raise ProspectiveError("prospective INVALID status and protocol errors differ")
    if (result["status"] == "INVALID") != (inference is None):
        raise ProspectiveError("prospective result inference lifecycle differs")
    if inference is not None:
        if (
            inference.get("schema") != "m15-book-refresh-family-inference/v1"
            or inference.get("family") != "prospective_shadow_3"
            or inference.get("tail") != "lower_only"
            or inference.get("family_size") != 3
            or inference.get("alpha") != spec["statistics"]["alpha"]
            or inference.get("bootstrap_replicates")
            != spec["statistics"]["bootstrap_replicates"]
            or inference.get("bootstrap_seed") != spec["statistics"]["bootstrap_seed"]
            or inference.get("block_lengths") != spec["statistics"]["block_lengths"]
        ):
            raise ProspectiveError("prospective result inference contract differs")
        for row, name in zip(inference["endpoints"], endpoint_order, strict=True):
            endpoint = result["endpoint_results"][name]
            if (
                row.get("estimate") != endpoint.get("point_estimate")
                or row.get("lower") != endpoint.get("simultaneous_lower_bound")
                or row.get("observed_denominator") != endpoint.get("observed_denominator")
                or row.get("computable") != endpoint.get("computable")
                or row.get("noncomputable_reasons")
                != endpoint.get("noncomputable_reasons")
            ):
                raise ProspectiveError(f"prospective endpoint {name} differs from inference")
    if not isinstance(result["complete_months"], list):
        raise ProspectiveError("prospective complete-month screen differs")
    classification = _classify_result(
        spec,
        endpoint_results={
            name: result["endpoint_results"][name] for name in endpoint_order
        },
        counts=result["schedule_counts"],
        complete_months=result["complete_months"],
        protocol_errors=errors,
    )
    if any(result[name] != classification[name] for name in classification):
        raise ProspectiveError("prospective result classification differs")
    return result


def _invalid_result(
    spec: Mapping[str, Any],
    prereg: evidence.EvidenceEnvelope,
    sealed: evidence.EvidenceEnvelope,
    error: str,
) -> dict[str, Any]:
    reason = re.sub(r"[^A-Za-z0-9_.: -]+", "_", error).strip()[:500]
    if not reason:
        reason = "unspecified fixed-analysis failure"
    names = list(spec["statistics"]["endpoint_order"])
    endpoint_results = {
        name: {
            "point_estimate": None,
            "simultaneous_lower_bound": None,
            "observed_denominator": 0.0,
            "computable": False,
            "noncomputable_reasons": ["protocol_invalid"],
        }
        for name in names
    }
    counts = sealed.payload["outcome_blind_counts"]
    classification = _classify_result(
        spec,
        endpoint_results=endpoint_results,
        counts=counts,
        complete_months=[],
        protocol_errors=[reason],
    )
    return {
        "schema": RESULT_SCHEMA,
        "issue": spec["issue"],
        "prereg_id": prereg.object_id,
        "sealed_source_id": sealed.object_id,
        "analysis_access_receipt_id": None,
        "pair": PAIR,
        "horizon_seconds": int(spec["horizon_seconds"]),
        "candidate_id": spec["candidate"]["artifact_id"],
        "incumbent_id": spec["incumbent"]["book_id"],
        "look": {
            "t0": prereg.payload["t0"],
            "cutoff_epoch": sealed.payload["cutoff_epoch"],
            "week": sealed.payload["week"],
            "trigger": sealed.payload["trigger"],
            "one_look": True,
        },
        "evidence_class": "OUTCOME_ONLY_TICK_PROXY",
        "selection": sealed.payload["selection"],
        "schedule_counts": counts,
        "endpoint_results": endpoint_results,
        "inference": None,
        "complete_months": [],
        **classification,
        "status": "INVALID",
        "protocol_errors": [reason],
        "claim_limit": spec["claim_limit"],
        "money": None,
        "activation": False,
    }


def _invalid_without_source_result(
    spec: Mapping[str, Any],
    prereg: evidence.EvidenceEnvelope,
    invalid: Mapping[str, Any],
) -> dict[str, Any]:
    reason = f"{invalid['stage']}: {invalid['reason']}"
    names = list(spec["statistics"]["endpoint_order"])
    endpoint_results = {
        name: {
            "point_estimate": None,
            "simultaneous_lower_bound": None,
            "observed_denominator": 0.0,
            "computable": False,
            "noncomputable_reasons": ["protocol_invalid_before_source_seal"],
        }
        for name in names
    }
    counts = {
        "common_settled": 0,
        "candidate": {"combined": 0, "UP": 0, "DOWN": 0},
        "incumbent": {"combined": 0, "UP": 0, "DOWN": 0},
    }
    classification = _classify_result(
        spec,
        endpoint_results=endpoint_results,
        counts=counts,
        complete_months=[],
        protocol_errors=[reason],
    )
    return {
        "schema": RESULT_SCHEMA,
        "issue": spec["issue"],
        "prereg_id": prereg.object_id,
        "sealed_source_id": None,
        "analysis_access_receipt_id": None,
        "pair": PAIR,
        "horizon_seconds": int(spec["horizon_seconds"]),
        "candidate_id": spec["candidate"]["artifact_id"],
        "incumbent_id": spec["incumbent"]["book_id"],
        "look": {
            "t0": prereg.payload["t0"],
            "cutoff_epoch": None,
            "week": None,
            "trigger": "PROTOCOL_INVALID",
            "one_look": True,
        },
        "evidence_class": "OUTCOME_ONLY_TICK_PROXY",
        "selection": None,
        "schedule_counts": counts,
        "endpoint_results": endpoint_results,
        "inference": None,
        "complete_months": [],
        **classification,
        "status": "INVALID",
        "protocol_errors": [reason],
        "claim_limit": spec["claim_limit"],
        "money": None,
        "activation": False,
    }


def _terminal_payload(
    spec: Mapping[str, Any],
    result: Mapping[str, Any],
    result_ref: evidence.ArtifactRef,
) -> dict[str, Any]:
    return {
        "schema": spec["evidence_kinds"]["terminal"],
        "prereg_id": result["prereg_id"],
        "sealed_source_id": result["sealed_source_id"],
        "analysis_access_receipt_id": result["analysis_access_receipt_id"],
        "result_artifact": result_ref.as_dict(),
        "normalized_result": dict(result),
        "status": result["status"],
        "evidence_class": "OUTCOME_ONLY_TICK_PROXY",
        "claim_limit": spec["claim_limit"],
        "money": None,
        "activation": False,
    }


def _expected_terminal(
    spec: Mapping[str, Any],
    result: Mapping[str, Any],
    result_ref: evidence.ArtifactRef,
    *,
    dependency_id: str | None = None,
) -> evidence.EvidenceEnvelope:
    dependency = dependency_id or result["analysis_access_receipt_id"]
    _hex64(dependency, "terminal dependency")
    return evidence.EvidenceEnvelope.create(
        kind=spec["evidence_kinds"]["terminal"],
        payload=_terminal_payload(spec, result, result_ref),
        artifacts=[result_ref],
        dependencies=[dependency],
    )


def _publish_result_terminal(
    root: Path,
    spec: Mapping[str, Any],
    result: Mapping[str, Any],
    *,
    require_new_result: bool,
    dependency_id: str | None = None,
) -> evidence.EvidenceEnvelope:
    result_path = root / spec["paths"]["result"]
    raw = _canonical(dict(result))
    if require_new_result:
        _atomic_new(result_path, raw, mode=0o444)
    else:
        stage = result_path.parent / f".{result_path.name}.building"
        if stage.exists() or stage.is_symlink():
            _atomic_new(result_path, raw, mode=0o444)
        _regular_file(result_path, "prospective result")
        if result_path.read_bytes() != raw:
            raise ProspectiveError("existing prospective result differs from its fixed recovery")
    result_info = _regular_file(result_path, "prospective result")
    if result_info.st_nlink != 1 or result_info.st_mode & 0o222:
        raise ProspectiveError(
            "prospective result must be read-only with exactly one hard link"
        )
    result_ref = evidence.ArtifactRef.capture(spec["paths"]["result"], repo_root=root)
    terminal = _expected_terminal(
        spec, result, result_ref, dependency_id=dependency_id
    )
    try:
        evidence.publish(terminal, repo_root=root, require_new=True)
    except evidence.EvidenceError as exc:
        raise ProspectiveError(f"cannot publish strict-new terminal: {exc}") from exc
    return terminal


def _deserialize_packet(raw: bytes) -> sqlite3.Connection:
    if not isinstance(raw, bytes) or not raw:
        raise ProspectiveError("sealed source packet bytes are empty")
    conn = sqlite3.connect(":memory:", isolation_level=None)
    conn.row_factory = sqlite3.Row
    try:
        if not hasattr(conn, "deserialize"):
            raise ProspectiveError("Python SQLite deserialize support is required")
        conn.deserialize(raw)
        conn.execute("PRAGMA query_only=ON")
        if conn.execute("PRAGMA query_only").fetchone()[0] != 1:
            raise ProspectiveError("in-memory packet query_only did not engage")
        return conn
    except Exception:
        conn.close()
        raise


def _discover_unique(
    root: Path,
    spec: Mapping[str, Any],
    *,
    state: str,
) -> str:
    groups = _domain_objects(root, spec)
    candidates: list[str] = []
    for item in groups[spec["evidence_kinds"]["prereg"]]:
        try:
            prereg, collection, _ = _validate_prereg(root, item.object_id)
            terminals = _matching_objects(
                groups, spec["evidence_kinds"]["terminal"], prereg.object_id
            )
            if state == "launchable":
                if terminals:
                    continue
                _require_h1(root, prereg, collection)
            elif state == "incomplete":
                if terminals:
                    continue
            elif state == "completed":
                if len(terminals) != 1:
                    continue
            elif state != "any":
                raise ProspectiveError(f"unknown discovery state {state!r}")
            candidates.append(prereg.object_id)
        except ProspectiveError:
            continue
    if len(candidates) != 1:
        raise ProspectiveError(
            f"expected exactly one {state} preregistration, found {len(candidates)}"
        )
    return candidates[0]


def _command_prereg_id(
    value: str | None,
    *,
    root: Path,
    spec: Mapping[str, Any],
    state: str,
) -> str:
    return _hex64(value, "prereg_id") if value is not None else _discover_unique(
        root, spec, state=state
    )


def _pending_look_boundaries(
    prereg: evidence.EvidenceEnvelope,
    spec: Mapping[str, Any],
    now_epoch: int,
    checked_boundary: int | None,
) -> list[int]:
    earliest = int(spec["fixed_look"]["earliest_weeks"])
    return [
        boundary
        for week, boundary in _week_boundaries(prereg, spec)
        if week >= earliest
        and boundary <= int(now_epoch)
        and (checked_boundary is None or boundary > checked_boundary)
    ]


def _remove_unstarted_trial_database(paths: Mapping[str, Path]) -> None:
    if paths["packet"].exists() or paths["packet_stage"].exists():
        raise ProspectiveError("unstarted trial unexpectedly contains a source packet")
    database = paths["database"]
    candidates = [database, *(Path(str(database) + suffix) for suffix in ("-wal", "-shm", "-journal"))]
    for path in candidates:
        if not path.exists() and not path.is_symlink():
            continue
        info = _regular_file(path, "unstarted trial SQLite file")
        if info.st_nlink != 1:
            raise ProspectiveError("unstarted trial SQLite file has foreign hard links")
        path.unlink()
    directory = os.open(
        paths["dir"], os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    )
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def collect(
    prereg_id: str | None = None,
    *,
    repo_root: str | os.PathLike[str] = REPO_ROOT,
    poll_seconds: float = 5.0,
) -> dict[str, Any]:
    if not isinstance(poll_seconds, (int, float)) or not 0 < float(poll_seconds) <= 5:
        raise ProspectiveError("collector poll_seconds must lie in (0,5]")
    root = _validated_root(repo_root)
    initial_spec: dict[str, Any] | None = None
    try:
        initial_spec = _load_spec(root)
        _require_public_environment()
        resolved = _command_prereg_id(
            prereg_id, root=root, spec=initial_spec, state="launchable"
        )
        prereg, collection, spec, h1 = _assert_write_authority(root, resolved)
        if prereg.payload["preflight"]["payload"]["host"] != os.uname().nodename:
            raise ProspectiveError("collector host differs from the VPS preflight host")
        candidate, incumbent, authenticated_spec = authenticate_policies(root)
        registered_scores = prereg.payload["preflight"]["payload"]["scoring_parity"]
        if (
            authenticated_spec != spec
            or registered_scores["candidate"]["content_id"] != candidate.content_id
            or registered_scores["incumbent"]["content_id"] != incumbent.content_id
        ):
            raise ProspectiveError(
                "authenticated policy spec differs from collection authority"
            )
    except Exception as exc:
        try:
            recorded = _record_startup_protocol_invalid(
                root, initial_spec, prereg_id, exc
            )
        except RetryableSealInterruption:
            raise
        except Exception as record_exc:
            raise ProspectiveError(
                f"startup authority failed ({type(exc).__name__}: {exc}); "
                f"durable INVALID recording also failed: {record_exc}"
            ) from exc
        if recorded:
            raise ProspectiveError(
                f"startup authority failed and is durably INVALID: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        raise
    paths = _trial_paths(root, spec, resolved)
    stop_requested = False

    def request_stop(_signum: int, _frame: Any) -> None:
        nonlocal stop_requested
        stop_requested = True

    previous_handlers = {
        signal.SIGINT: signal.signal(signal.SIGINT, request_stop),
        signal.SIGTERM: signal.signal(signal.SIGTERM, request_stop),
    }
    try:
        with _trial_lock(paths["lock"]):
            launch_epoch = int(time.time())
            if (
                not paths["database"].exists()
                and launch_epoch >= int(prereg.payload["t0"]["epoch"])
            ):
                raise ProspectiveError(
                    "unstarted preregistration missed T0; a reviewed successor is required"
                )
            conn = _open_trial(paths["database"])
            trial_initialized = False
            try:
                existing_metadata = conn.execute(
                    "SELECT created_epoch FROM trial_metadata WHERE singleton=1"
                ).fetchone()
                trial_initialized = existing_metadata is not None
                if (
                    existing_metadata is None
                    and launch_epoch >= int(prereg.payload["t0"]["epoch"])
                ):
                    raise ProspectiveError(
                        "unstarted preregistration missed T0; a reviewed successor is required"
                    )
                metadata_inserted = _initialize_trial(
                    conn,
                    _metadata_payload(prereg, collection, spec, h1),
                )
                if metadata_inserted and time.time() >= int(prereg.payload["t0"]["epoch"]):
                    conn.close()
                    conn = None
                    _remove_unstarted_trial_database(paths)
                    raise ProspectiveError(
                        "trial metadata commit did not complete before T0; "
                        "a reviewed successor is required"
                    )
                trial_initialized = True
                created_epoch = int(
                    conn.execute(
                        "SELECT created_epoch FROM trial_metadata WHERE singleton=1"
                    ).fetchone()[0]
                )
                if created_epoch >= int(prereg.payload["t0"]["epoch"]):
                    raise ProspectiveError("trial metadata was not sealed before T0")
                existing_invalid = _load_protocol_invalid(conn, prereg.object_id)
                if existing_invalid is not None:
                    raise ProspectiveError(
                        "trial is durably INVALID: "
                        f"{existing_invalid['stage']}: {existing_invalid['reason']}"
                    )
                checked_boundary: int | None = None
                while not stop_requested:
                    now_epoch = int(time.time())
                    existing_seal = _load_trial_seal(conn, prereg.object_id)
                    if existing_seal is not None:
                        sealed = _seal_if_due(
                            root,
                            prereg=prereg,
                            collection=collection,
                            spec=spec,
                            conn=conn,
                            paths=paths,
                            now_epoch=now_epoch,
                        )
                        assert sealed is not None
                        return {
                            "prereg_id": prereg.object_id,
                            "sealed_source_id": sealed.object_id,
                            "state": "SEALED_SOURCE_READY",
                            "outcome_blind": True,
                            "money": None,
                            "activation": False,
                        }
                    for due_boundary in _pending_look_boundaries(
                        prereg, spec, now_epoch, checked_boundary
                    ):
                        _capture_due(
                            root,
                            prereg,
                            spec,
                            conn,
                            candidate,
                            incumbent,
                            now_epoch=now_epoch,
                            capture_through_epoch=due_boundary,
                        )
                        sealed = _seal_if_due(
                            root=root,
                            prereg=prereg,
                            collection=collection,
                            spec=spec,
                            conn=conn,
                            paths=paths,
                            now_epoch=now_epoch,
                            evaluate_through_epoch=due_boundary,
                        )
                        checked_boundary = due_boundary
                        if sealed is not None:
                            return {
                                "prereg_id": prereg.object_id,
                                "sealed_source_id": sealed.object_id,
                                "state": "SEALED_SOURCE_READY",
                                "outcome_blind": True,
                                "money": None,
                                "activation": False,
                            }
                    if now_epoch >= int(prereg.payload["t0"]["epoch"]):
                        _capture_due(
                            root,
                            prereg,
                            spec,
                            conn,
                            candidate,
                            incumbent,
                            now_epoch=now_epoch,
                        )
                    time.sleep(float(poll_seconds))
                return {
                    "prereg_id": prereg.object_id,
                    "state": "STOPPED_BEFORE_SEAL",
                    "outcome_blind": True,
                    "money": None,
                    "activation": False,
                }
            except Exception as exc:
                detected_epoch = int(time.time())
                seal_committed = (
                    _load_trial_seal(conn, prereg.object_id)
                    if trial_initialized and conn is not None
                    else None
                )
                if seal_committed is not None and not isinstance(
                    exc, SealedSourceProtocolError
                ):
                    raise RetryableSealInterruption(
                        "sealed source completion was interrupted; restart to resume "
                        "the immutable settlement-bound transaction"
                    ) from exc
                if (
                    trial_initialized
                    and detected_epoch >= int(prereg.payload["t0"]["epoch"])
                    and _load_protocol_invalid(conn, prereg.object_id) is None
                ):
                    invalid = _protocol_invalid_payload(
                        prereg.object_id,
                        detected_epoch=detected_epoch,
                        stage="COLLECTION_PROTOCOL",
                        reason=f"{type(exc).__name__}: {exc}",
                    )
                    try:
                        _insert_protocol_invalid_once(conn, invalid)
                    except Exception as record_exc:
                        raise ProspectiveError(
                            f"{exc}; durable INVALID recording also failed: {record_exc}"
                        ) from exc
                if detected_epoch >= int(prereg.payload["t0"]["epoch"]):
                    if isinstance(exc, ProspectiveError):
                        raise
                    raise ProspectiveError(
                        f"post-T0 collection failure: {type(exc).__name__}: {exc}"
                    ) from exc
                raise
            finally:
                if conn is not None:
                    conn.close()
    finally:
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)


def _open_trial_readonly(path: Path) -> sqlite3.Connection:
    before = _regular_file(path, "trial database")
    if before.st_nlink != 1:
        raise ProspectiveError("trial database must have exactly one hard link")
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, isolation_level=None)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA query_only=ON")
        conn.execute("PRAGMA foreign_keys=ON")
        if (
            conn.execute("PRAGMA query_only").fetchone()[0] != 1
            or int(conn.execute("PRAGMA application_id").fetchone()[0]) != 1431119105
            or int(conn.execute("PRAGMA user_version").fetchone()[0]) != 1
        ):
            raise ProspectiveError("trial read-only SQLite identity differs")
        after = _regular_file(path, "opened trial database")
        if after.st_nlink != 1 or (before.st_dev, before.st_ino) != (
            after.st_dev,
            after.st_ino,
        ):
            raise ProspectiveError("trial database identity changed while being opened")
        _assert_trial_schema(conn)
        return conn
    except Exception:
        conn.close()
        raise


def status(
    prereg_id: str | None = None,
    *,
    repo_root: str | os.PathLike[str] = REPO_ROOT,
    now_epoch: int | None = None,
) -> dict[str, Any]:
    _require_public_environment()
    root = _validated_root(repo_root)
    now_value = int(time.time()) if now_epoch is None else int(now_epoch)
    spec: dict[str, Any] | None = None
    try:
        spec = _load_spec(root)
        resolved = _command_prereg_id(prereg_id, root=root, spec=spec, state="any")
        prereg, _collection, spec = _validate_prereg(root, resolved)
    except ProspectiveError as authority_error:
        candidates = _metadata_trial_candidates(root, spec, prereg_id)
        invalid_candidates = [row for row in candidates if row[2] is not None]
        if len(invalid_candidates) != 1:
            raise
        prereg, _collection, invalid = invalid_candidates[0]
        assert invalid is not None
        return {
            "schema": "usdchf-m15-prospective-status/v1",
            "prereg_id": prereg.object_id,
            "t0": prereg.payload["t0"],
            "now_epoch": now_value,
            "outcome_blind": True,
            "money": None,
            "activation": False,
            "state": "INVALID",
            "detected_epoch": invalid["detected_epoch"],
            "stage": invalid["stage"],
            "reason": invalid["reason"],
            "full_authority_verified": False,
            "authority_error": f"{type(authority_error).__name__}: {authority_error}",
        }
    assert spec is not None
    resolved = prereg.object_id
    paths = _trial_paths(root, spec, resolved, create=False)
    base = {
        "schema": "usdchf-m15-prospective-status/v1",
        "prereg_id": resolved,
        "t0": prereg.payload["t0"],
        "now_epoch": now_value,
        "outcome_blind": True,
        "money": None,
        "activation": False,
    }
    groups = _domain_objects(root, spec)
    if _matching_objects(groups, spec["evidence_kinds"]["terminal"], resolved):
        return {**base, "state": "TERMINAL_RECORDED"}
    if not paths["database"].exists():
        return {
            **base,
            "state": (
                "PRE_T0_NOT_STARTED"
                if now_value < int(prereg.payload["t0"]["epoch"])
                else "NOT_STARTED"
            ),
        }
    conn = _open_trial_readonly(paths["database"])
    try:
        metadata = conn.execute(
            "SELECT prereg_id FROM trial_metadata WHERE singleton=1"
        ).fetchone()
        if metadata is None:
            return {
                **base,
                "state": (
                    "PRE_T0_NOT_STARTED"
                    if now_value < int(prereg.payload["t0"]["epoch"])
                    else "MISSED_T0_UNSTARTED"
                ),
            }
        if metadata["prereg_id"] != resolved:
            raise ProspectiveError("trial metadata belongs to a foreign preregistration")
        invalid = _load_protocol_invalid(conn, resolved)
        if invalid is not None:
            return {
                **base,
                "state": "INVALID",
                "detected_epoch": invalid["detected_epoch"],
                "stage": invalid["stage"],
                "reason": invalid["reason"],
            }
        decisions = _load_decisions(conn, resolved)
        seal = _load_trial_seal(conn, resolved)
        common = sum(bool(row["common_scoreable"]) for row in decisions)
        gaps = len(decisions) - common
        if seal is not None:
            counts = seal["counts"]
            availability = {
                key: seal["selection"][key]
                for key in ("settled", "unsettled", "not_common_scoreable")
            }
            store_freshness = None
            state = "SEALED"
        else:
            ticks, descriptor = _read_tick_rows(
                root,
                spec,
                start_epoch=int(prereg.payload["t0"]["epoch"]),
                end_epoch=now_value,
                required_intervals=_settlement_tick_intervals(decisions, spec),
            )
            settlement = _settlement_rows(decisions, ticks, spec)
            counts = _scheduled_counts(decisions, settlement)
            availability = {
                "settled": sum(row["status"] == "SETTLED" for row in settlement),
                "unsettled": sum(row["status"] == "UNSETTLED" for row in settlement),
                "not_common_scoreable": sum(
                    row["status"] == "NOT_COMMON_SCOREABLE" for row in settlement
                ),
            }
            store_freshness = {
                "last_persisted_epoch": descriptor["last_persisted_epoch"],
                "lag_seconds": max(0, now_value - descriptor["last_persisted_epoch"]),
            }
            state = (
                "PRE_T0"
                if now_value < int(prereg.payload["t0"]["epoch"])
                else "COLLECTING"
            )
        weeks_elapsed = sum(
            boundary <= now_value for _week, boundary in _week_boundaries(prereg, spec)
        )
        return {
            **base,
            "state": state,
            "decision_rows": len(decisions),
            "common_scoreable_rows": common,
            "explicit_gap_rows": gaps,
            "complete_weeks_elapsed": weeks_elapsed,
            "schedule_counts": counts,
            "settlement_availability": availability,
            "store_freshness": store_freshness,
        }
    finally:
        conn.close()


def _persisted_protocol_invalid(
    root: Path,
    spec: Mapping[str, Any],
    prereg: evidence.EvidenceEnvelope,
) -> dict[str, Any] | None:
    paths = _trial_paths(root, spec, prereg.object_id, create=False)
    if not paths["database"].exists():
        return None
    conn = _open_trial_readonly(paths["database"])
    try:
        invalid = _load_protocol_invalid(conn, prereg.object_id)
    finally:
        conn.close()
    if invalid is not None and invalid["detected_epoch"] < int(
        prereg.payload["t0"]["epoch"]
    ):
        raise ProspectiveError("protocol INVALID was recorded before T0")
    return invalid


def _analyze_once(
    prereg_id: str | None = None,
    *,
    repo_root: str | os.PathLike[str] = REPO_ROOT,
) -> dict[str, Any]:
    _require_public_environment()
    root = _validated_root(repo_root)
    spec = _load_spec(root)
    try:
        resolved = _command_prereg_id(
            prereg_id, root=root, spec=spec, state="incomplete"
        )
        prereg, collection, spec = _validate_prereg(root, resolved)
    except ProspectiveError as authority_error:
        invalid_candidates = [
            row
            for row in _metadata_trial_candidates(root, spec, prereg_id)
            if row[2] is not None
        ]
        if len(invalid_candidates) == 1:
            raise ProspectiveError(
                "durable protocol INVALID is recorded; restore the exact H1-bound "
                "artifacts solely to publish and verify its terminal: "
                f"{authority_error}"
            ) from authority_error
        raise
    groups = _domain_objects(root, spec)
    terminals = _matching_objects(groups, spec["evidence_kinds"]["terminal"], resolved)
    if terminals:
        return verify_prospective(resolved, repo_root=root)
    invalid = _persisted_protocol_invalid(root, spec, prereg)
    if invalid is not None:
        if _matching_objects(
            groups, spec["evidence_kinds"]["sealed_source"], resolved
        ) or _matching_objects(
            groups, spec["evidence_kinds"]["analysis_access_receipt"], resolved
        ):
            raise ProspectiveError(
                "pre-seal protocol INVALID conflicts with source/access evidence"
            )
        expected = _invalid_without_source_result(spec, prereg, invalid)
        result_path = root / spec["paths"]["result"]
        if result_path.is_symlink():
            raise ProspectiveError("protocol-INVALID result path is a symlink")
        if result_path.exists():
            _regular_file(result_path, "protocol-INVALID result")
            try:
                recovered = evidence.decode_canonical_json(result_path.read_bytes())
            except evidence.EvidenceError as exc:
                raise ProspectiveError(
                    f"protocol-INVALID result is not canonical JSON: {exc}"
                ) from exc
            result = _validate_result_value(
                recovered,
                spec,
                prereg.object_id,
                None,
                None,
                prereg=prereg,
                sealed=None,
            )
            if result != expected:
                raise ProspectiveError(
                    "existing protocol-INVALID result differs from durable state"
                )
            require_new_result = False
        else:
            result = _validate_result_value(
                expected,
                spec,
                prereg.object_id,
                None,
                None,
                prereg=prereg,
                sealed=None,
            )
            require_new_result = True
        _publish_result_terminal(
            root,
            spec,
            result,
            require_new_result=require_new_result,
            dependency_id=collection.object_id,
        )
        return result
    sealed, source_ref = _validated_sealed_source(
        root, prereg, collection, spec, groups=groups
    )
    access = _matching_objects(
        groups, spec["evidence_kinds"]["analysis_access_receipt"], resolved
    )
    result_path = root / spec["paths"]["result"]
    if access:
        receipt = _validated_analysis_receipt(
            groups, spec, prereg, sealed, source_ref
        )
        if result_path.exists() and not result_path.is_symlink():
            _regular_file(result_path, "prospective result")
            try:
                recovered = evidence.decode_canonical_json(result_path.read_bytes())
            except evidence.EvidenceError as exc:
                raise ProspectiveError(
                    f"interrupted prospective result is not canonical JSON: {exc}"
                ) from exc
            result = _validate_result_value(
                recovered,
                spec,
                prereg.object_id,
                sealed.object_id,
                receipt.object_id,
                prereg=prereg,
                sealed=sealed,
            )
            _publish_result_terminal(
                root,
                spec,
                result,
                require_new_result=False,
                dependency_id=receipt.object_id,
            )
            return result
        if result_path.is_symlink():
            raise ProspectiveError("prospective result recovery path is a symlink")
        reduced = _invalid_result(
            spec,
            prereg,
            sealed,
            "analysis interrupted after the access receipt; source packet was not reread",
        )
        reduced["analysis_access_receipt_id"] = receipt.object_id
        result = _validate_result_value(
            reduced,
            spec,
            prereg.object_id,
            sealed.object_id,
            receipt.object_id,
            prereg=prereg,
            sealed=sealed,
        )
        _publish_result_terminal(
            root,
            spec,
            result,
            require_new_result=True,
            dependency_id=receipt.object_id,
        )
        return result
    if result_path.exists() or result_path.is_symlink():
        raise ProspectiveError("prospective result exists without a terminal")
    for suffix in ("-wal", "-shm", "-journal"):
        if (root / f"{source_ref.path}{suffix}").exists():
            raise ProspectiveError("sealed source packet has a SQLite sidecar")
    try:
        source_ref.isolated_identity(repo_root=root)
    except evidence.EvidenceError as exc:
        raise ProspectiveError(f"sealed source isolation failed: {exc}") from exc
    receipt = _expected_analysis_receipt(spec, prereg, sealed, source_ref)
    try:
        evidence.publish(receipt, repo_root=root, require_new=True)
    except evidence.EvidenceError as exc:
        raise ProspectiveError(f"cannot publish strict-new analysis receipt: {exc}") from exc

    reduced: dict[str, Any]
    try:
        raw = source_ref.read_verified(repo_root=root)
        conn = _deserialize_packet(raw)
        try:
            reduced = _reduce_packet(
                conn, spec=spec, prereg=prereg, sealed_source=sealed
            )
        finally:
            conn.close()
    except Exception as exc:
        reduced = _invalid_result(spec, prereg, sealed, f"{type(exc).__name__}: {exc}")
    reduced = dict(reduced)
    reduced["analysis_access_receipt_id"] = receipt.object_id
    result = _validate_result_value(
        reduced,
        spec,
        prereg.object_id,
        sealed.object_id,
        receipt.object_id,
        prereg=prereg,
        sealed=sealed,
    )
    _publish_result_terminal(
        root,
        spec,
        result,
        require_new_result=True,
        dependency_id=receipt.object_id,
    )
    return result


def analyze(
    prereg_id: str | None = None,
    *,
    repo_root: str | os.PathLike[str] = REPO_ROOT,
) -> dict[str, Any]:
    _require_public_environment()
    root = _validated_root(repo_root)
    spec = _load_spec(root)
    try:
        resolved = _command_prereg_id(
            prereg_id, root=root, spec=spec, state="incomplete"
        )
    except ProspectiveError as incomplete_error:
        if prereg_id is None:
            try:
                resolved = _command_prereg_id(
                    None, root=root, spec=spec, state="completed"
                )
            except ProspectiveError:
                resolved = ""
            if resolved:
                paths = _trial_paths(root, spec, resolved, create=False)
                with _trial_lock(paths["lock"]):
                    return _analyze_once(resolved, repo_root=root)
        candidates = [
            row
            for row in _metadata_trial_candidates(root, spec, prereg_id)
            if row[2] is not None
        ]
        if len(candidates) != 1:
            raise incomplete_error
        resolved = candidates[0][0].object_id
    paths = _trial_paths(root, spec, resolved, create=True)
    with _trial_lock(paths["lock"]):
        return _analyze_once(resolved, repo_root=root)


def verify_prospective(
    prereg_id: str | None = None,
    *,
    repo_root: str | os.PathLike[str] = REPO_ROOT,
) -> dict[str, Any]:
    root = _validated_root(repo_root)
    spec = _load_spec(root)
    resolved = _command_prereg_id(
        prereg_id, root=root, spec=spec, state="completed"
    )
    prereg, collection, spec = _validate_prereg(root, resolved)
    groups = _domain_objects(root, spec)
    terminals = _matching_objects(
        groups, spec["evidence_kinds"]["terminal"], prereg.object_id
    )
    if len(terminals) != 1:
        raise ProspectiveError(f"expected exactly one terminal, found {len(terminals)}")
    terminal = terminals[0]
    terminal_fields = {
        "schema",
        "prereg_id",
        "sealed_source_id",
        "analysis_access_receipt_id",
        "result_artifact",
        "normalized_result",
        "status",
        "evidence_class",
        "claim_limit",
        "money",
        "activation",
    }
    payload = _exact_fields(terminal.payload, terminal_fields, "terminal payload")
    try:
        result_ref = evidence.ArtifactRef.from_dict(payload["result_artifact"])
    except evidence.EvidenceError as exc:
        raise ProspectiveError(f"terminal result artifact is malformed: {exc}") from exc
    if (
        terminal.kind != spec["evidence_kinds"]["terminal"]
        or terminal.artifacts != (result_ref,)
        or result_ref.path != spec["paths"]["result"]
    ):
        raise ProspectiveError("terminal ancestry or result binding differs")
    try:
        raw = result_ref.read_verified(repo_root=root)
        result = evidence.decode_canonical_json(raw)
    except evidence.EvidenceError as exc:
        raise ProspectiveError(f"terminal result does not verify: {exc}") from exc
    no_source_invalid = (
        isinstance(result, dict)
        and result.get("sealed_source_id") is None
        and result.get("analysis_access_receipt_id") is None
    )
    if no_source_invalid:
        invalid = _persisted_protocol_invalid(root, spec, prereg)
        if invalid is None:
            raise ProspectiveError("no-source INVALID terminal lacks durable trial state")
        if _matching_objects(
            groups, spec["evidence_kinds"]["sealed_source"], prereg.object_id
        ) or _matching_objects(
            groups,
            spec["evidence_kinds"]["analysis_access_receipt"],
            prereg.object_id,
        ):
            raise ProspectiveError("no-source INVALID conflicts with source/access evidence")
        result = _validate_result_value(
            result,
            spec,
            prereg.object_id,
            None,
            None,
            prereg=prereg,
            sealed=None,
        )
        if result != _invalid_without_source_result(spec, prereg, invalid):
            raise ProspectiveError("no-source INVALID differs from durable trial state")
        expected = _expected_terminal(
            spec, result, result_ref, dependency_id=collection.object_id
        )
        sealed_source_id = None
        analysis_receipt_id = None
    else:
        sealed, source_ref = _validated_sealed_source(
            root, prereg, collection, spec, groups=groups
        )
        receipt = _validated_analysis_receipt(
            groups, spec, prereg, sealed, source_ref
        )
        result = _validate_result_value(
            result,
            spec,
            prereg.object_id,
            sealed.object_id,
            receipt.object_id,
            prereg=prereg,
            sealed=sealed,
        )
        expected = _expected_terminal(spec, result, result_ref)
        sealed_source_id = sealed.object_id
        analysis_receipt_id = receipt.object_id
    if terminal.as_dict() != expected.as_dict() or payload["normalized_result"] != result:
        raise ProspectiveError("terminal cannot be reconstructed from its result")
    return {
        "schema": VERIFICATION_SCHEMA,
        "prereg_id": prereg.object_id,
        "sealed_source_id": sealed_source_id,
        "analysis_access_receipt_id": analysis_receipt_id,
        "terminal_id": terminal.object_id,
        "result_artifact": result_ref.as_dict(),
        "status": result["status"],
        "source_packet_read": False,
        "money": None,
        "activation": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    preflight = commands.add_parser("preflight", help="run spent-session parity checks")
    preflight.add_argument("--session-date", required=True)
    preflight.add_argument(
        "--output",
        default="deriv_data/prospective/preflight/usdchf_m15_prospective_v1_preflight.json",
    )
    prereg = commands.add_parser("preregister", help="seal the future T0 and collection authority")
    prereg.add_argument("--t0-utc", required=True)
    prereg.add_argument("--t0-ny", required=True)
    prereg.add_argument("--preflight", required=True)
    prereg.add_argument("--user-approved-t0", action="store_true")
    for name, help_text in (
        ("collect", "run the buy-incapable public-store observer"),
        ("status", "show outcome-blind trial state"),
        ("analyze", "consume the one protected fixed look"),
        ("verify", "verify a completed terminal without the source packet"),
    ):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--prereg-id")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "preflight":
            output = write_preflight(args.session_date, args.output)
        elif args.command == "preregister":
            prereg, collection = preregister(
                t0_utc=args.t0_utc,
                t0_ny=args.t0_ny,
                preflight_path=args.preflight,
                user_approved_t0=args.user_approved_t0,
            )
            output = {
                "prereg_id": prereg.object_id,
                "collection_access_receipt_id": collection.object_id,
                "activation": False,
            }
        elif args.command == "collect":
            output = collect(args.prereg_id)
        elif args.command == "status":
            output = status(args.prereg_id)
        elif args.command == "analyze":
            output = analyze(args.prereg_id)
        else:
            output = verify_prospective(args.prereg_id)
        print(json.dumps(output, sort_keys=True, separators=(",", ":")))
        return 0
    except RetryableSealInterruption as exc:
        print(f"RETRYABLE: {exc}", file=sys.stderr)
        return 1
    except (ProspectiveError, evidence.EvidenceError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
