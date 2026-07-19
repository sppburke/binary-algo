#!/usr/bin/env python3
"""Seal, build, and verify one inactive USDCHF 15-minute refit package.

This operator creates no efficacy measurement and has no book, runtime,
deployment, demo-buy, or real-money authority.  Its three public commands are
``seal``, ``fit --seal-id``, and ``verify --seal-id``.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import fcntl
import hashlib
import json
import os
import re
import secrets
import stat
import subprocess
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterator, Mapping, Sequence

import evidence_store as evidence


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = "scripts/usdchf_m15_current_refit_v1.py"
SPEC_PATH = "scripts/usdchf_m15_current_refit_v1_spec.json"
SPEC_SHA256 = "595a03a32f9dfad9d5ccdafcdaee0b2155556d944c30deada488b6aee4b1611d"
SOURCE_SCHEMA = "usdchf-m15-current-refit-source-manifest/v1"
SEAL_PAYLOAD_SCHEMA = "usdchf-m15-current-refit-execution-seal/v1"
TERMINAL_PAYLOAD_SCHEMA = "usdchf-m15-current-refit-terminal/v1"
RESULT_SCHEMA = "usdchf-m15-current-refit-result/v1"
CHECKPOINT_SCHEMA = "usdchf-m15-current-refit-seed-checkpoint/v1"
STRATEGY_SCHEMA = "usdchf-m15-current-refit-strategy/v1"
RESOLVED_SCHEMA = "usdchf-m15-current-refit-resolved-spec/v1"
PACKAGE_MANIFEST_SCHEMA = "usdchf-m15-current-refit-package-manifest/v1"
HEX64 = re.compile(r"[0-9a-f]{64}")
REQUIRED_SOURCE_FIELDS = ("y", "fwd_ret", "valid", "close", "datetime_utc")
RUNTIME_DISTRIBUTIONS = (
    "numpy",
    "pandas",
    "pyarrow",
    "lightgbm",
    "scikit-learn",
)
SUCCESS = "INACTIVE_UNTESTED_PROSPECTIVE"
NO_CANDIDATE = "NO_CANDIDATE"


class RefitError(RuntimeError):
    """The sealed refit contract or its local state failed closed."""


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(8 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(value: Any) -> bytes:
    return evidence.canonical_json_bytes(value)


def _canonical_relative(value: str, *, name: str = "path") -> str:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise RefitError(f"{name} must be a canonical relative POSIX path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or path.as_posix() != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise RefitError(f"{name} must be a canonical relative POSIX path")
    return value


def _regular_file(path: Path, *, name: str) -> os.stat_result:
    if path.is_symlink():
        raise RefitError(f"{name} must not be a symlink: {path}")
    try:
        info = path.stat(follow_symlinks=False)
    except OSError as exc:
        raise RefitError(f"cannot inspect {name}: {path}: {exc}") from exc
    if not stat.S_ISREG(info.st_mode):
        raise RefitError(f"{name} is not a regular file: {path}")
    return info


def _read_canonical(path: Path, *, name: str) -> tuple[bytes, Any]:
    _regular_file(path, name=name)
    raw = path.read_bytes()
    try:
        value = evidence.decode_canonical_json(raw)
    except evidence.EvidenceError as exc:
        raise RefitError(f"{name} is not canonical JSON: {path}: {exc}") from exc
    return raw, value


def _fsync_dir(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _ensure_real_directory(
    root: Path, relative: str, *, mode: int = 0o700
) -> Path:
    """Create a repository-relative directory without following symlink components."""

    canonical = _canonical_relative(relative, name="directory path")
    current = root
    for part in PurePosixPath(canonical).parts:
        current = current / part
        try:
            info = current.stat(follow_symlinks=False)
        except FileNotFoundError:
            try:
                current.mkdir(mode=mode)
                _fsync_dir(current.parent)
            except FileExistsError:
                pass
            info = current.stat(follow_symlinks=False)
        if current.is_symlink() or not stat.S_ISDIR(info.st_mode):
            raise RefitError(f"directory component is missing, special, or symlinked: {current}")
    return current


def _existing_real_directory(root: Path, relative: str) -> Path:
    """Resolve an existing repository-relative directory without following symlinks."""

    canonical = _canonical_relative(relative, name="directory path")
    current = root
    for part in PurePosixPath(canonical).parts:
        current = current / part
        try:
            info = current.stat(follow_symlinks=False)
        except FileNotFoundError as exc:
            raise RefitError(f"required directory component is missing: {current}") from exc
        if current.is_symlink() or not stat.S_ISDIR(info.st_mode):
            raise RefitError(f"directory component is missing, special, or symlinked: {current}")
    return current


def _atomic_new(path: Path, raw: bytes, *, read_only: bool = True) -> bool:
    """Install exact bytes without replacement; return False for an exact retry."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise RefitError(f"destination must not be a symlink: {path}")
    if path.exists():
        if not path.is_file() or path.read_bytes() != raw:
            raise RefitError(f"existing destination differs: {path}")
        return False
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{secrets.token_hex(8)}")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(temporary, flags, 0o600)
    linked = False
    try:
        view = memoryview(raw)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise RefitError(f"short write while staging {path}")
            view = view[written:]
        if read_only:
            os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        try:
            os.link(temporary, path, follow_symlinks=False)
            linked = True
        except FileExistsError:
            if path.is_symlink() or not path.is_file() or path.read_bytes() != raw:
                raise RefitError(f"no-replace publication lost to conflicting bytes: {path}")
        _fsync_dir(path.parent)
        return linked
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _validated_root(repo_root: str | os.PathLike[str]) -> Path:
    root = Path(repo_root).resolve()
    if not root.is_dir() or not (root / ".git").exists():
        raise RefitError(f"not a repository root: {root}")
    for relative in ("scripts", "results/json", "evidence/objects"):
        _existing_real_directory(root, relative)
    return root


def _load_spec(root: Path) -> dict[str, Any]:
    path = root / SPEC_PATH
    raw, value = _read_canonical(path, name="execution spec")
    if _sha256_bytes(raw) != SPEC_SHA256:
        raise RefitError("execution spec bytes differ from the reviewed contract")
    if not isinstance(value, dict) or value.get("schema") != "usdchf-m15-current-refit-spec/v1":
        raise RefitError("execution spec schema differs")
    if value.get("pair") != "USDCHF" or value.get("side") != "combined":
        raise RefitError("execution spec identity differs")
    if value.get("activation") is not False:
        raise RefitError("execution spec must forbid activation")
    if value.get("allowed_terminal_statuses") != [SUCCESS, NO_CANDIDATE]:
        raise RefitError("execution spec terminal order differs")
    return value


def _git(root: Path, *arguments: str, check: bool = True) -> str:
    process = subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if check and process.returncode != 0:
        raise RefitError(
            f"git {' '.join(arguments)} failed: {process.stderr.strip()}"
        )
    return process.stdout.strip()


def _git_head(root: Path) -> str:
    head = _git(root, "rev-parse", "HEAD")
    if HEX64.fullmatch(head) is None and re.fullmatch(r"[0-9a-f]{40}", head) is None:
        raise RefitError("Git HEAD is malformed")
    return head


def _git_status(root: Path) -> dict[str, str]:
    process = subprocess.run(
        ["git", "status", "--porcelain=v1", "-z", "--untracked-files=all"],
        cwd=root,
        check=True,
        stdout=subprocess.PIPE,
    )
    records = process.stdout.split(b"\0")
    status: dict[str, str] = {}
    index = 0
    while index < len(records):
        record = records[index]
        index += 1
        if not record:
            continue
        if len(record) < 4 or record[2:3] != b" ":
            raise RefitError("cannot parse git porcelain state")
        code = record[:2].decode("ascii", errors="strict")
        if "R" in code or "C" in code:
            raise RefitError("renamed/copied worktree state is not admissible")
        path = record[3:].decode("utf-8", errors="strict")
        _canonical_relative(path, name="git status path")
        if path in status:
            raise RefitError(f"duplicate git status path: {path}")
        status[path] = code
    return status


def _require_main_at_origin(root: Path) -> str:
    if _git(root, "branch", "--show-current") != "main":
        raise RefitError("operator requires branch main")
    head = _git_head(root)
    origin = _git(root, "rev-parse", "origin/main")
    if head != origin:
        raise RefitError(f"HEAD differs from origin/main: HEAD={head} origin={origin}")
    return head


def _is_ancestor(root: Path, ancestor: str, descendant: str) -> bool:
    process = subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, descendant], cwd=root
    )
    return process.returncode == 0


def _tracked_at(root: Path, revision: str, relative: str) -> bool:
    process = subprocess.run(
        ["git", "cat-file", "-e", f"{revision}:{relative}"],
        cwd=root,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return process.returncode == 0


def _git_blob_identity(root: Path, revision: str, relative: str) -> dict[str, Any]:
    _canonical_relative(relative)
    line = _git(root, "ls-tree", revision, "--", relative)
    if not line or "\t" not in line:
        raise RefitError(f"path is not tracked at {revision}: {relative}")
    prefix, returned = line.split("\t", 1)
    parts = prefix.split()
    if returned != relative or len(parts) != 3 or parts[1] != "blob":
        raise RefitError(f"unexpected Git tree identity for {relative}")
    mode, _, blob = parts
    process = subprocess.run(
        ["git", "show", f"{revision}:{relative}"],
        cwd=root,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if process.returncode != 0:
        raise RefitError(f"cannot read Git blob at {revision}: {relative}")
    raw = process.stdout
    return {
        "path": relative,
        "mode": mode,
        "blob": blob,
        "bytes": len(raw),
        "sha256": _sha256_bytes(raw),
    }


def _git_blob_bytes(root: Path, revision: str, relative: str) -> bytes:
    _canonical_relative(relative)
    process = subprocess.run(
        ["git", "show", f"{revision}:{relative}"],
        cwd=root,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if process.returncode != 0:
        raise RefitError(f"cannot read Git blob at {revision}: {relative}")
    return process.stdout


def _runtime_pins(raw: bytes) -> dict[str, str]:
    try:
        lines = raw.decode("utf-8", errors="strict").splitlines()
    except UnicodeDecodeError as exc:
        raise RefitError("runtime authority is not UTF-8") from exc
    pins: dict[str, str] = {}
    for line in lines:
        value = line.strip()
        if not value or value.startswith("#") or "==" not in value:
            continue
        name, version = value.split("==", 1)
        normalized = name.strip().lower()
        exact = version.strip()
        if not normalized or not exact or normalized in pins:
            raise RefitError("runtime authority contains malformed or duplicate pins")
        pins[normalized] = exact
    try:
        return {name: pins[name] for name in RUNTIME_DISTRIBUTIONS}
    except KeyError as exc:
        raise RefitError(f"runtime authority lacks exact pin {exc.args[0]}") from exc


def _validate_runtime_contract(
    root: Path, h0: str, value: Any
) -> dict[str, Any]:
    expected_fields = {
        "python",
        "executable",
        "prefix",
        "base_prefix",
        "user_site_enabled",
        "distributions",
    }
    expected_prefix = Path.home() / "binary-algo-venv"
    expected_executable = expected_prefix / "bin/python"
    expected_distributions = _runtime_pins(
        _git_blob_bytes(root, h0, "ENVIRONMENT_libs.txt")
    )
    if (
        not isinstance(value, dict)
        or set(value) != expected_fields
        or not isinstance(value.get("python"), str)
        or re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", value["python"]) is None
        or value.get("executable") != str(expected_executable)
        or value.get("prefix") != str(expected_prefix)
        or not isinstance(value.get("base_prefix"), str)
        or value.get("user_site_enabled") is not False
        or value.get("distributions") != expected_distributions
    ):
        raise RefitError("runtime version contract differs from the H0 authority")
    return dict(value)


def _verified_runtime_version_contract(root: Path, h0: str) -> dict[str, Any]:
    """Authenticate the active interpreter and installed fit dependencies."""

    import importlib.metadata
    import site

    expected_prefix = Path.home() / "binary-algo-venv"
    expected_executable = expected_prefix / "bin/python"
    try:
        distributions = {
            name: importlib.metadata.version(name) for name in RUNTIME_DISTRIBUTIONS
        }
    except importlib.metadata.PackageNotFoundError as exc:
        raise RefitError(f"required runtime distribution is absent: {exc}") from exc
    value = {
        "python": sys.version.split()[0],
        "executable": str(Path(sys.executable).absolute()),
        "prefix": str(Path(sys.prefix).absolute()),
        "base_prefix": str(Path(sys.base_prefix).absolute()),
        "user_site_enabled": site.ENABLE_USER_SITE,
        "distributions": distributions,
    }
    if (
        Path(sys.executable).absolute() != expected_executable
        or Path(sys.prefix).absolute() != expected_prefix
        or site.ENABLE_USER_SITE is not False
    ):
        raise RefitError(f"fit runtime is outside the isolated repository venv: {value}")
    return _validate_runtime_contract(root, h0, value)


def _verify_shared_live(root: Path, identities: Sequence[Mapping[str, Any]]) -> None:
    for identity in identities:
        path = root / str(identity["path"])
        info = _regular_file(path, name="shared implementation")
        if (
            info.st_size != identity["bytes"]
            or _sha256_file(path) != identity["sha256"]
            or _git(root, "hash-object", "--", str(identity["path"]))
            != identity["blob"]
        ):
            raise RefitError(
                f"live shared implementation differs from sealed H0: {identity['path']}"
            )


@contextmanager
def _research_lock(root: Path, spec: Mapping[str, Any]) -> Iterator[None]:
    path = root / str(spec["lock_path"])
    expected = root / "logs/research_campaign/run_campaign.lock"
    if path != expected:
        raise RefitError("execution spec lock path differs")
    directory = _ensure_real_directory(root, "logs/research_campaign")
    if path.is_symlink():
        raise RefitError("research lock must not be a symlink")
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or stat.S_IMODE(opened.st_mode) != 0o600
        ):
            raise RefitError("research lock must be private and singly linked")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if exc.errno in {errno.EACCES, errno.EAGAIN}:
                raise RefitError("another repository research operation is active") from exc
            raise
        named = path.stat(follow_symlinks=False)
        if (opened.st_dev, opened.st_ino) != (named.st_dev, named.st_ino):
            raise RefitError("research lock path changed during acquisition")
        _fsync_dir(directory)
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _source_paths(spec: Mapping[str, Any]) -> tuple[str, ...]:
    paths = tuple(
        f"features/{pair}_{year}.parquet"
        for pair in spec["feature_source_pair_order"]
        for year in spec["fit_year_order"]
    )
    if len(paths) != 105 or len(paths) != len(set(paths)):
        raise RefitError("source inventory is not the exact ordered 105-file universe")
    return paths


def _source_row(root: Path, relative: str) -> dict[str, Any]:
    import pandas as pd
    import pyarrow.parquet as parquet

    path = root / relative
    before = _regular_file(path, name="feature source")
    table = parquet.ParquetFile(path)
    fields = tuple(table.schema_arrow.names)
    if any(field not in fields for field in REQUIRED_SOURCE_FIELDS):
        raise RefitError(f"feature source lacks required fields: {relative}")
    frame = pd.read_parquet(path, columns=[])
    index = frame.index
    if (
        not isinstance(index, pd.DatetimeIndex)
        or index.tz is None
        or str(index.tz) != "UTC"
        or len(index) != table.metadata.num_rows
        or len(index) == 0
        or not index.is_monotonic_increasing
        or not index.is_unique
    ):
        raise RefitError(f"feature source timestamp index is invalid: {relative}")
    minimum = index.min().isoformat().replace("+00:00", "Z")
    maximum = index.max().isoformat().replace("+00:00", "Z")
    digest = _sha256_file(path)
    after = path.stat(follow_symlinks=False)
    stable = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    ) == (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if not stable:
        raise RefitError(f"feature source changed during inventory: {relative}")
    return {
        "path": relative,
        "bytes": int(after.st_size),
        "sha256": digest,
        "rows": int(len(index)),
        "min_timestamp": minimum,
        "max_timestamp": maximum,
        "required_fields": list(REQUIRED_SOURCE_FIELDS),
    }


def _build_source_manifest(root: Path, spec: Mapping[str, Any], h0: str) -> dict[str, Any]:
    sources = [_source_row(root, relative) for relative in _source_paths(spec)]
    return {
        "schema": SOURCE_SCHEMA,
        "artifact_id": spec["artifact_id"],
        "h0_git_sha": h0,
        "source_count": len(sources),
        "total_bytes": sum(row["bytes"] for row in sources),
        "fit_year_order": list(spec["fit_year_order"]),
        "calibration_year_order": list(spec["calibration_year_order"]),
        "source_order": sources,
        "raw_sources_tracked": False,
        "raw_sources_are_artifact_refs": False,
    }


def _shared_identities(root: Path, spec: Mapping[str, Any], h0: str) -> list[dict[str, Any]]:
    return [
        _git_blob_identity(root, h0, str(relative))
        for relative in spec["shared_identity_paths"]
    ]


def _expected_seal(
    root: Path,
    spec: Mapping[str, Any],
    source_manifest: Mapping[str, Any],
    h0: str,
    runtime_contract: Mapping[str, Any],
) -> evidence.EvidenceEnvelope:
    source_path = str(spec["source_manifest_path"])
    source_ref = evidence.ArtifactRef.capture(source_path, repo_root=root)
    payload = {
        "schema": SEAL_PAYLOAD_SCHEMA,
        "artifact_id": spec["artifact_id"],
        "authority": spec["authority"],
        "h0_git_sha": h0,
        "shared_git_identities": _shared_identities(root, spec, h0),
        "runtime_version_contract": dict(runtime_contract),
        "source_manifest": source_ref.as_dict(),
        "source_count": source_manifest["source_count"],
        "activation": False,
        "lifecycle": spec["lifecycle"],
        "claim_limit": spec["claim_limit"],
        "raw_sources_are_artifact_refs": False,
    }
    artifacts = (
        evidence.ArtifactRef.capture(SCRIPT_PATH, repo_root=root),
        evidence.ArtifactRef.capture(SPEC_PATH, repo_root=root),
        source_ref,
    )
    return evidence.EvidenceEnvelope.create(
        kind=spec["evidence_kinds"]["seal"],
        payload=payload,
        artifacts=artifacts,
        dependencies=(),
    )


def _manifest_timestamp(value: Any, *, name: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise RefitError(f"{name} must be a canonical UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise RefitError(f"{name} must be a canonical UTC timestamp") from exc
    if (
        parsed.tzinfo != timezone.utc
        or parsed.isoformat().replace("+00:00", "Z") != value
    ):
        raise RefitError(f"{name} must be a canonical UTC timestamp")
    return parsed


def _validate_source_manifest(value: Any, spec: Mapping[str, Any], h0: str) -> None:
    expected_fields = {
        "schema", "artifact_id", "h0_git_sha", "source_count", "total_bytes",
        "fit_year_order", "calibration_year_order", "source_order",
        "raw_sources_tracked", "raw_sources_are_artifact_refs",
    }
    if not isinstance(value, dict) or set(value) != expected_fields:
        raise RefitError("source manifest fields differ")
    if (
        value["schema"] != SOURCE_SCHEMA
        or value["artifact_id"] != spec["artifact_id"]
        or value["h0_git_sha"] != h0
        or value["fit_year_order"] != spec["fit_year_order"]
        or value["calibration_year_order"] != spec["calibration_year_order"]
        or value["raw_sources_tracked"] is not False
        or value["raw_sources_are_artifact_refs"] is not False
    ):
        raise RefitError("source manifest identity differs")
    rows = value["source_order"]
    if not isinstance(rows, list) or len(rows) != 105:
        raise RefitError("source manifest row count differs")
    row_fields = {
        "path",
        "bytes",
        "sha256",
        "rows",
        "min_timestamp",
        "max_timestamp",
        "required_fields",
    }
    for position, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != row_fields:
            raise RefitError(f"source manifest row fields differ at position {position}")
        if (
            isinstance(row["bytes"], bool)
            or not isinstance(row["bytes"], int)
            or row["bytes"] <= 0
            or isinstance(row["rows"], bool)
            or not isinstance(row["rows"], int)
            or row["rows"] <= 0
            or not isinstance(row["sha256"], str)
            or HEX64.fullmatch(row["sha256"]) is None
            or row["required_fields"] != list(REQUIRED_SOURCE_FIELDS)
        ):
            raise RefitError(f"source manifest row identity differs at position {position}")
        minimum = _manifest_timestamp(
            row["min_timestamp"], name=f"source row {position} minimum"
        )
        maximum = _manifest_timestamp(
            row["max_timestamp"], name=f"source row {position} maximum"
        )
        if minimum > maximum:
            raise RefitError(f"source manifest timestamp order differs at position {position}")
    if [row["path"] for row in rows] != list(_source_paths(spec)):
        raise RefitError("source manifest path order differs")
    if (
        isinstance(value["source_count"], bool)
        or not isinstance(value["source_count"], int)
        or isinstance(value["total_bytes"], bool)
        or not isinstance(value["total_bytes"], int)
        or value["source_count"] != len(rows)
        or value["total_bytes"] != sum(row["bytes"] for row in rows)
    ):
        raise RefitError("source manifest aggregates differ")


def _validate_seal(
    root: Path, spec: Mapping[str, Any], seal_id: str
) -> tuple[evidence.EvidenceEnvelope, dict[str, Any]]:
    if HEX64.fullmatch(seal_id) is None:
        raise RefitError("seal ID must be lowercase 64-hex")
    try:
        envelope = evidence.verify_object(seal_id, repo_root=root)
    except evidence.EvidenceError as exc:
        raise RefitError(f"execution seal verification failed: {exc}") from exc
    if envelope.kind != spec["evidence_kinds"]["seal"] or envelope.dependencies:
        raise RefitError("execution seal kind/dependencies differ")
    artifacts = [artifact.path for artifact in envelope.artifacts]
    if artifacts != [SCRIPT_PATH, SPEC_PATH, str(spec["source_manifest_path"])]:
        raise RefitError("execution seal artifact order differs")
    raw, source = _read_canonical(
        root / str(spec["source_manifest_path"]), name="source manifest"
    )
    del raw
    h0 = envelope.payload.get("h0_git_sha") if isinstance(envelope.payload, dict) else None
    if not isinstance(h0, str) or re.fullmatch(r"[0-9a-f]{40}", h0) is None:
        raise RefitError("execution seal H0 is missing")
    _validate_source_manifest(source, spec, h0)
    runtime_contract = _validate_runtime_contract(
        root, h0, envelope.payload.get("runtime_version_contract")
    )
    expected = _expected_seal(root, spec, source, h0, runtime_contract)
    if envelope.as_dict() != expected.as_dict():
        raise RefitError("execution seal domain payload differs")
    return envelope, source


def seal(
    *,
    repo_root: str | os.PathLike[str] = REPO_ROOT,
    fault: Callable[[str], None] | None = None,
) -> evidence.EvidenceEnvelope:
    """Publish or recover the exact pre-fit execution seal at clean H0."""

    root = _validated_root(repo_root)
    spec = _load_spec(root)
    with _research_lock(root, spec):
        h0 = _require_main_at_origin(root)
        reviewed = str(spec["authority"]["reviewed_repo_head"])
        if not _is_ancestor(root, reviewed, h0):
            raise RefitError("H0 does not descend from the reviewed repository baseline")
        if _tracked_at(root, h0, str(spec["source_manifest_path"])):
            raise RefitError("seal may run only before the source manifest enters H1")
        preliminary = _git_status(root)
        source_path = str(spec["source_manifest_path"])
        evidence_paths = {
            path
            for path in preliminary
            if re.fullmatch(r"evidence/objects/[0-9a-f]{64}\.json", path)
        }
        permitted_preliminary = ({source_path} if source_path in preliminary else set()) | evidence_paths
        if (
            set(preliminary) != permitted_preliminary
            or len(evidence_paths) > 1
            or (evidence_paths and source_path not in preliminary)
            or any(code != "??" for code in preliminary.values())
        ):
            raise RefitError(f"H0 has foreign worktree state: {preliminary}")

        source_value = _build_source_manifest(root, spec, h0)
        source_raw = _canonical(source_value)
        source_destination = root / source_path
        _atomic_new(source_destination, source_raw)
        if fault is not None:
            fault("after_source_manifest")
        runtime_contract = _verified_runtime_version_contract(root, h0)
        expected = _expected_seal(
            root, spec, source_value, h0, runtime_contract
        )
        expected_evidence_path = f"evidence/objects/{expected.object_id}.json"
        if evidence_paths and evidence_paths != {expected_evidence_path}:
            raise RefitError(f"H0 has a foreign evidence object: {sorted(evidence_paths)}")
        _verify_shared_live(root, expected.payload["shared_git_identities"])
        destination = root / expected_evidence_path
        if destination.exists() or destination.is_symlink():
            actual = evidence.verify_object(expected.object_id, repo_root=root)
            if actual.as_dict() != expected.as_dict():
                raise RefitError("existing execution seal differs")
        else:
            try:
                evidence.publish(expected, repo_root=root, require_new=True)
            except evidence.EvidenceError as exc:
                raise RefitError(f"cannot publish strict-new execution seal: {exc}") from exc
        if fault is not None:
            fault("after_execution_seal")
        final_status = _git_status(root)
        expected_status = {source_path, expected_evidence_path}
        if set(final_status) != expected_status or any(
            code != "??" for code in final_status.values()
        ):
            raise RefitError(f"pre-H1 state differs from exact seal outputs: {final_status}")
        return expected


def _authenticate_h1(
    root: Path, spec: Mapping[str, Any], seal_envelope: evidence.EvidenceEnvelope
) -> tuple[str, str]:
    h1 = _require_main_at_origin(root)
    parents = _git(root, "rev-list", "--parents", "-n", "1", h1).split()
    if len(parents) != 2 or parents[0] != h1:
        raise RefitError("H1 must have exactly one parent")
    h0 = str(seal_envelope.payload["h0_git_sha"])
    if parents[1] != h0:
        raise RefitError("H1 sole parent differs from sealed H0")
    changed = _git(root, "diff", "--name-only", h0, h1).splitlines()
    expected = sorted(
        [
            str(spec["source_manifest_path"]),
            f"evidence/objects/{seal_envelope.object_id}.json",
        ]
    )
    if sorted(changed) != expected or len(changed) != len(expected):
        raise RefitError(f"H0..H1 paths differ from exact seal outputs: {changed}")
    for relative in expected:
        if not _tracked_at(root, h1, relative):
            raise RefitError(f"H1 does not track required seal output: {relative}")
    return h1, h0


def _source_status(
    root: Path,
    spec: Mapping[str, Any],
    source_manifest: Mapping[str, Any],
) -> str:
    paths = [root / row["path"] for row in source_manifest["source_order"]]
    present = [path.exists() or path.is_symlink() for path in paths]
    if not any(present):
        return "unavailable"
    if not all(present):
        raise RefitError("raw feature sources are partially present")
    rebuilt = _build_source_manifest(root, spec, str(source_manifest["h0_git_sha"]))
    if rebuilt != source_manifest:
        raise RefitError("raw feature source identity differs from the seal")
    return "match"


def _i64_sha(values: Any) -> str:
    import numpy as np

    return _sha256_bytes(np.asarray(values, dtype="<i8").tobytes(order="C"))


def _u8_sha(values: Any) -> str:
    import numpy as np

    return _sha256_bytes(np.asarray(values, dtype="uint8").tobytes(order="C"))


def _checkpoint_manifest_path(model_path: Path) -> Path:
    return model_path.with_name(f"{model_path.name}.checkpoint.json")


def _checkpoint_context(
    *,
    seal_id: str,
    h1: str,
    phase: str,
    seed: int,
    position: int,
    feature_cols: Sequence[str],
    parameters: Mapping[str, Any],
    partition: Mapping[str, Any],
    model_filename: str,
) -> dict[str, Any]:
    return {
        "schema": CHECKPOINT_SCHEMA,
        "seal_id": seal_id,
        "h1_git_sha": h1,
        "phase": phase,
        "seed": int(seed),
        "seed_order_position": int(position),
        "feature_cols_sha256": _sha256_bytes(_canonical(list(feature_cols))),
        "model_parameters": dict(parameters),
        "fit_row_id_sha256": partition["fit_row_id_sha256"],
        "early_stopping_row_id_sha256": partition["early_stopping_row_id_sha256"],
        "fit_label_u8_sha256": partition["fit_label_u8_sha256"],
        "early_stopping_label_u8_sha256": partition[
            "early_stopping_label_u8_sha256"
        ],
        "model_filename": model_filename,
    }


def _validate_checkpoint(
    model_path: Path,
    sidecar_path: Path,
    *,
    expected_context: Mapping[str, Any],
    feature_count: int,
) -> Any:
    from m15_book_refresh_adapters import SeedCheckpoint
    import lightgbm as lgb

    model_info = _regular_file(model_path, name="seed checkpoint model")
    sidecar_info = _regular_file(sidecar_path, name="seed checkpoint sidecar")
    if model_info.st_mode & 0o222 or sidecar_info.st_mode & 0o222:
        raise RefitError("seed checkpoint pair must be read-only")
    _, sidecar = _read_canonical(sidecar_path, name="seed checkpoint sidecar")
    dynamic = {"best_iteration", "model_bytes", "model_sha256"}
    if not isinstance(sidecar, dict) or set(sidecar) != set(expected_context) | dynamic:
        raise RefitError(f"seed checkpoint sidecar fields differ: {sidecar_path}")
    if any(sidecar.get(key) != value for key, value in expected_context.items()):
        raise RefitError(f"seed checkpoint context differs: {sidecar_path}")
    digest = _sha256_file(model_path)
    best = sidecar.get("best_iteration")
    if (
        isinstance(best, bool)
        or not isinstance(best, int)
        or best <= 0
        or sidecar.get("model_bytes") != model_info.st_size
        or sidecar.get("model_sha256") != digest
    ):
        raise RefitError(f"seed checkpoint identity differs: {sidecar_path}")
    try:
        booster = lgb.Booster(model_file=str(model_path))
        if booster.num_feature() != feature_count or booster.current_iteration() != best:
            raise RefitError(f"seed checkpoint model structure differs: {model_path}")
    except RefitError:
        raise
    except Exception as exc:
        raise RefitError(f"cannot load seed checkpoint model: {model_path}: {exc}") from exc
    return SeedCheckpoint(
        pair="USDCHF",
        seed=int(sidecar["seed"]),
        best_iteration=best,
        model_path=model_path,
        sha256=digest,
    )


def _authenticate_checkpoint_orphan(
    path: Path,
    *,
    is_model: bool,
    expected_context: Mapping[str, Any],
    feature_count: int,
) -> None:
    """Authenticate the sole member of an interrupted owned checkpoint pair."""

    info = _regular_file(path, name="owned seed-checkpoint orphan")
    if info.st_nlink != 1:
        raise RefitError(f"owned checkpoint orphan has multiple links: {path}")
    if is_model:
        import lightgbm as lgb

        try:
            booster = lgb.Booster(model_file=str(path))
            valid = booster.num_feature() == feature_count and booster.current_iteration() > 0
        except Exception as exc:
            raise RefitError(f"model-only checkpoint orphan is not authentic: {path}") from exc
        if not valid:
            raise RefitError(f"model-only checkpoint orphan structure differs: {path}")
        return
    if info.st_mode & 0o222:
        raise RefitError(f"sidecar-only checkpoint orphan must be read-only: {path}")
    _, sidecar = _read_canonical(path, name="seed checkpoint sidecar orphan")
    dynamic = {"best_iteration", "model_bytes", "model_sha256"}
    if (
        not isinstance(sidecar, dict)
        or set(sidecar) != set(expected_context) | dynamic
        or any(sidecar.get(key) != value for key, value in expected_context.items())
        or isinstance(sidecar.get("best_iteration"), bool)
        or not isinstance(sidecar.get("best_iteration"), int)
        or sidecar["best_iteration"] <= 0
        or isinstance(sidecar.get("model_bytes"), bool)
        or not isinstance(sidecar.get("model_bytes"), int)
        or sidecar["model_bytes"] <= 0
        or not isinstance(sidecar.get("model_sha256"), str)
        or HEX64.fullmatch(sidecar["model_sha256"]) is None
    ):
        raise RefitError(f"sidecar-only checkpoint orphan context differs: {path}")


def _validate_checkpoint_workspace(
    root: Path, spec: Mapping[str, Any], seal_id: str
) -> None:
    workspace = root / str(spec["workspace_path"]) / seal_id
    if workspace.is_symlink() or not workspace.is_dir():
        raise RefitError("seal checkpoint workspace is missing or symlinked")
    seeds = [int(value) for value in spec["model"]["seed_order"]]
    actual_phases = {entry.name for entry in workspace.iterdir()}
    if not actual_phases.issubset({"canonical", "audit"}):
        raise RefitError("seal checkpoint workspace contains foreign state")
    phase_states: dict[str, list[str]] = {}
    for phase in actual_phases:
        phase_directory = workspace / phase
        if phase_directory.is_symlink() or not phase_directory.is_dir():
            raise RefitError(f"{phase} checkpoint phase is not a real directory")
        if {entry.name for entry in phase_directory.iterdir()} != {"models"}:
            raise RefitError(f"{phase} checkpoint phase contains foreign state")
        model_directory = phase_directory / "models"
        if model_directory.is_symlink() or not model_directory.is_dir():
            raise RefitError(f"{phase} checkpoint model directory differs")
        allowed = {
            name
            for seed in seeds
            for name in (
                f"USDCHF_{phase}_s{seed}_lgb.txt",
                f"USDCHF_{phase}_s{seed}_lgb.txt.checkpoint.json",
            )
        }
        entries = list(model_directory.iterdir())
        entry_names = {entry.name for entry in entries}
        if not entry_names.issubset(allowed):
            raise RefitError(f"{phase} checkpoint model directory contains foreign state")
        for entry in entries:
            _regular_file(entry, name=f"{phase} checkpoint workspace member")
        states: list[str] = []
        for seed in seeds:
            model_name = f"USDCHF_{phase}_s{seed}_lgb.txt"
            sidecar_name = f"{model_name}.checkpoint.json"
            model_present = model_name in entry_names
            sidecar_present = sidecar_name in entry_names
            states.append(
                "pair"
                if model_present and sidecar_present
                else "orphan"
                if model_present or sidecar_present
                else "absent"
            )
        first_incomplete = next(
            (position for position, state in enumerate(states) if state != "pair"),
            len(states),
        )
        if any(state != "absent" for state in states[first_incomplete + 1 :]):
            raise RefitError(f"{phase} checkpoint state is not a sealed-order prefix")
        phase_states[phase] = states
    canonical_complete = phase_states.get("canonical") == ["pair"] * len(seeds)
    if "audit" in actual_phases and not canonical_complete:
        raise RefitError("audit checkpoint state exists before canonical completion")


def _fit_phase(
    *,
    root: Path,
    spec: Mapping[str, Any],
    seal_id: str,
    h1: str,
    phase: str,
    fit_rows: Any,
    fit_idx: Any,
    validation_rows: Any,
    early_idx: Any,
    calibration_idx: Any,
    partition: Mapping[str, Any],
    fault: Callable[[str], None] | None,
    allow_fit: bool = True,
) -> dict[str, Any]:
    import numpy as np
    from m15_book_refresh_adapters import (
        calibrate_policy,
        fit_seed_checkpoint,
        model_parameters,
        predict_checkpoint_mean,
    )

    if phase not in {"canonical", "audit"}:
        raise RefitError("unknown fit phase")
    seeds = [int(value) for value in spec["model"]["seed_order"]]
    workspace_relative = f"{spec['workspace_path']}/{seal_id}/{phase}"
    model_directory_relative = f"{workspace_relative}/models"
    if allow_fit:
        model_dir = _ensure_real_directory(root, model_directory_relative)
    else:
        try:
            model_dir = _existing_real_directory(root, model_directory_relative)
        except RefitError as exc:
            raise RefitError(
                f"post-fit recovery lacks the complete {phase} checkpoint directory"
            ) from exc
    _validate_checkpoint_workspace(root, spec, seal_id)
    expected_names: set[str] = set()
    checkpoints: list[Any] = []
    parameters_by_seed: list[dict[str, Any]] = []
    for position, seed in enumerate(seeds):
        model_path = model_dir / f"USDCHF_{phase}_s{seed}_lgb.txt"
        sidecar_path = _checkpoint_manifest_path(model_path)
        expected_names.update({model_path.name, sidecar_path.name})
        parameters = model_parameters(
            "USDCHF", seed, n_jobs=int(spec["model"]["num_threads"])
        )
        parameters_by_seed.append(parameters)
        context = _checkpoint_context(
            seal_id=seal_id,
            h1=h1,
            phase=phase,
            seed=seed,
            position=position,
            feature_cols=fit_rows.feature_cols,
            parameters=parameters,
            partition=partition,
            model_filename=model_path.name,
        )
        model_exists = model_path.exists() or model_path.is_symlink()
        sidecar_exists = sidecar_path.exists() or sidecar_path.is_symlink()
        if not allow_fit and not (model_exists and sidecar_exists):
            raise RefitError(
                f"post-fit recovery lacks the complete {phase} seed {seed} checkpoint pair"
            )
        if model_exists != sidecar_exists:
            orphan = model_path if model_exists else sidecar_path
            _authenticate_checkpoint_orphan(
                orphan,
                is_model=model_exists,
                expected_context=context,
                feature_count=len(fit_rows.feature_cols),
            )
            orphan.chmod(0o600)
            orphan.unlink()
            _fsync_dir(model_dir)
            model_exists = sidecar_exists = False
        if model_exists and sidecar_exists:
            checkpoint = _validate_checkpoint(
                model_path,
                sidecar_path,
                expected_context=context,
                feature_count=len(fit_rows.feature_cols),
            )
        else:
            if not allow_fit:
                raise RefitError(
                    f"post-fit recovery may not refit {phase} seed {seed}"
                )
            checkpoint = fit_seed_checkpoint(
                "USDCHF",
                seed,
                fit_rows,
                fit_idx,
                validation_rows,
                early_idx,
                model_path,
                n_jobs=int(spec["model"]["num_threads"]),
            )
            if (
                checkpoint.pair != "USDCHF"
                or checkpoint.seed != seed
                or Path(checkpoint.model_path) != model_path
                or checkpoint.sha256 != _sha256_file(model_path)
            ):
                raise RefitError(f"new seed checkpoint identity differs: {model_path}")
            model_path.chmod(0o444)
            sidecar = {
                **context,
                "best_iteration": int(checkpoint.best_iteration),
                "model_bytes": model_path.stat().st_size,
                "model_sha256": checkpoint.sha256,
            }
            _atomic_new(sidecar_path, _canonical(sidecar))
            checkpoint = _validate_checkpoint(
                model_path,
                sidecar_path,
                expected_context=context,
                feature_count=len(fit_rows.feature_cols),
            )
        checkpoints.append(checkpoint)
        if fault is not None:
            fault(f"after_{phase}_seed_{seed}")
    actual_names = {path.name for path in model_dir.iterdir()}
    if actual_names != expected_names:
        raise RefitError(f"{phase} checkpoint directory contains foreign state")

    calibration_positions = np.asarray(calibration_idx, dtype="int64")
    selected_probabilities = predict_checkpoint_mean(
        checkpoints, validation_rows.X[calibration_positions]
    )
    if selected_probabilities.shape != (len(calibration_positions),) or not np.isfinite(
        selected_probabilities
    ).all():
        raise RefitError(f"{phase} calibration probabilities are invalid")
    probabilities = np.full(len(validation_rows.fit_row_id), np.nan, dtype="float64")
    probabilities[calibration_positions] = selected_probabilities
    admitted = np.zeros(len(validation_rows.fit_row_id), dtype=bool)
    admitted[calibration_positions] = True
    calibration = calibrate_policy(
        validation_rows,
        probabilities,
        admitted,
        target_coverage=float(spec["model"]["target_coverage"]),
        structural_scope_mask=None,
    )
    output_names = list(spec["package_files"][:3])
    model_rows = [
        {
            "file": output_names[position],
            "seed": int(checkpoint.seed),
            "bytes": int(checkpoint.model_path.stat().st_size),
            "sha256": checkpoint.sha256,
            "best_iteration": int(checkpoint.best_iteration),
        }
        for position, checkpoint in enumerate(checkpoints)
    ]
    strategy = {
        "schema": STRATEGY_SCHEMA,
        "artifact_id": spec["artifact_id"],
        "pair": "USDCHF",
        "side": "combined",
        "target": "direction",
        "lifecycle": spec["lifecycle"],
        "activation": False,
        "seal_id": seal_id,
        "h1_git_sha": h1,
        "information_cutoff": spec["split"]["information_cutoff"],
        "feature_cols": list(fit_rows.feature_cols),
        "feature_count": len(fit_rows.feature_cols),
        "feature_dtype": "float32",
        "seed_order": seeds,
        "models": model_rows,
        "ensemble": spec["model"]["ensemble"],
        "coverage": float(calibration.target_coverage),
        "confidence_threshold": float(calibration.confidence_threshold),
        "gate": {
            "confidence": "abs(p-0.5)>=confidence_threshold",
            "probability_rule": spec["model"]["probability_rule"],
            "session": spec["model"]["session"],
            "structural_column": calibration.structural_column,
            "structural_threshold": calibration.structural_threshold,
        },
        "horizon_seconds": int(spec["model"]["horizon_seconds"]),
        "decision_shift_seconds": int(spec["model"]["decision_shift_seconds"]),
        "claim_limit": spec["claim_limit"],
    }
    resolved = {
        "schema": RESOLVED_SCHEMA,
        "artifact_id": spec["artifact_id"],
        "seal_id": seal_id,
        "h1_git_sha": h1,
        "lifecycle": spec["lifecycle"],
        "activation": False,
        "source_manifest_path": spec["source_manifest_path"],
        "source_manifest_sha256": _sha256_file(root / str(spec["source_manifest_path"])),
        "fit_year_order": list(spec["fit_year_order"]),
        "calibration_year_order": list(spec["calibration_year_order"]),
        "split": dict(spec["split"]),
        "model_contract": dict(spec["model"]),
        "model_parameters_by_seed": parameters_by_seed,
        "partition": dict(partition),
        "feature_cols_sha256": _sha256_bytes(_canonical(list(fit_rows.feature_cols))),
        "models": model_rows,
        "strategy_sha256": _sha256_bytes(_canonical(strategy)),
        "calibration": {
            "target_coverage": float(calibration.target_coverage),
            "confidence_threshold": float(calibration.confidence_threshold),
            "calibration_rows": int(calibration.calibration_rows),
            "confidence_rows": int(calibration.confidence_rows),
            "structural_column": calibration.structural_column,
            "structural_quantile": calibration.structural_quantile,
            "structural_threshold": calibration.structural_threshold,
        },
        "no_efficacy_measurement": True,
    }
    return {
        "checkpoints": checkpoints,
        "strategy": strategy,
        "strategy_bytes": _canonical(strategy),
        "resolved": resolved,
        "resolved_bytes": _canonical(resolved),
    }


def _validate_usdchf_horizon_contract(*, require_loaded: bool) -> None:
    ambient = os.environ.get("MX_HOR")
    if ambient not in {None, "15"}:
        raise RefitError("MX_HOR must be absent or the canonical value 15")
    public = sys.modules.get("usdchf_15m_xpair")
    frozen = sys.modules.get("usdchf_15m_xpair_frozen")
    if public is not None and (
        getattr(public, "HOR", None) != 15
        or getattr(public, "GAP_S", None) != 900
    ):
        raise RefitError("loaded USDCHF xpair builder has a non-15m label horizon")
    if frozen is not None and (
        getattr(frozen, "HOR", None) != 15
        or getattr(frozen, "GAP", None) != 900
        or getattr(frozen, "XP", None) is not public
    ):
        raise RefitError("loaded frozen USDCHF builder has a non-15m dependency contract")
    if require_loaded and (public is None or frozen is None):
        raise RefitError("USDCHF xpair builder modules did not resolve")


def _build_rows_and_partitions(root: Path, spec: Mapping[str, Any]) -> dict[str, Any]:
    _validate_usdchf_horizon_contract(require_loaded=False)
    import numpy as np
    from m15_book_refresh_adapters import (
        build_calibration_rows,
        build_fit_rows,
        calibration_indices,
        contract_for,
        early_stopping_indices,
        fitting_indices,
    )

    contract = contract_for("USDCHF")
    model = spec["model"]
    if (
        contract.n_features != model["feature_count"]
        or contract.fit_stride != model["fit_stride"]
        or list(contract.seeds) != model["seed_order"]
        or contract.target_coverage != model["target_coverage"]
        or contract.early_stopping_ties != model["early_stopping_ties"]
    ):
        raise RefitError("live USDCHF adapter contract differs from the seal")
    feature_dir = root / "features"
    fit_rows = build_fit_rows(
        "USDCHF", spec["fit_year_order"], feature_dir=feature_dir
    )
    _validate_usdchf_horizon_contract(require_loaded=True)
    validation_rows = build_calibration_rows(
        "USDCHF", spec["calibration_year_order"], feature_dir=feature_dir
    )
    _validate_usdchf_horizon_contract(require_loaded=True)
    if (
        fit_rows.feature_cols != validation_rows.feature_cols
        or len(fit_rows.feature_cols) != model["feature_count"]
        or fit_rows.builder_stride != model["fit_stride"]
        or validation_rows.builder_stride != model["calibration_stride"]
    ):
        raise RefitError("fit/calibration feature or stride contract differs")
    split = spec["split"]
    fit_idx = fitting_indices(
        fit_rows,
        entry_at_or_after=split["fit_entry_at_or_after"],
        label_exit_before=split["fit_label_exit_before"],
        cap=int(model["fit_cap"]),
    )
    early_idx = early_stopping_indices(
        validation_rows,
        entry_at_or_after=split["calibration_entry_at_or_after"],
        label_exit_before=split["calibration_label_exit_before"],
    )
    calibration_idx = calibration_indices(
        validation_rows,
        entry_at_or_after=split["calibration_entry_at_or_after"],
        label_exit_before=split["calibration_label_exit_before"],
    )
    if not len(fit_idx) or not len(early_idx) or not len(calibration_idx):
        raise RefitError("fit/early-stopping/calibration partition is empty")
    if not np.array_equal(early_idx, calibration_idx):
        raise RefitError("USDCHF early-stopping/calibration membership differs")
    fit_boundary = int(np.datetime64("2026-04-01T00:00:00", "ns").astype("int64"))
    calibration_boundary = int(
        np.datetime64("2026-05-09T00:00:00", "ns").astype("int64")
    )
    if (
        np.any(fit_rows.label_exit_ns[fit_idx] >= fit_boundary)
        or np.any(validation_rows.fit_row_id[early_idx] < fit_boundary)
        or np.any(validation_rows.label_exit_ns[early_idx] >= calibration_boundary)
        or np.intersect1d(
            fit_rows.fit_row_id[fit_idx], validation_rows.fit_row_id[early_idx]
        ).size
    ):
        raise RefitError("partition boundary/disjointness contract differs")
    partition = {
        "fit_rows": int(len(fit_idx)),
        "early_stopping_rows": int(len(early_idx)),
        "calibration_rows": int(len(calibration_idx)),
        "fit_row_id_sha256": _i64_sha(fit_rows.fit_row_id[fit_idx]),
        "early_stopping_row_id_sha256": _i64_sha(
            validation_rows.fit_row_id[early_idx]
        ),
        "calibration_row_id_sha256": _i64_sha(
            validation_rows.fit_row_id[calibration_idx]
        ),
        "fit_label_u8_sha256": _u8_sha(fit_rows.y[fit_idx]),
        "early_stopping_label_u8_sha256": _u8_sha(validation_rows.y[early_idx]),
        "calibration_label_u8_sha256": _u8_sha(
            validation_rows.y[calibration_idx]
        ),
    }
    return {
        "fit_rows": fit_rows,
        "fit_idx": fit_idx,
        "validation_rows": validation_rows,
        "early_idx": early_idx,
        "calibration_idx": calibration_idx,
        "partition": partition,
    }


def _rename_noreplace(source: Path, destination: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise RefitError("renameat2(RENAME_NOREPLACE) is unavailable")
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    rc = renameat2(-100, os.fsencode(source), -100, os.fsencode(destination), 1)
    if rc != 0:
        error = ctypes.get_errno()
        if error == errno.EEXIST:
            raise FileExistsError(destination)
        raise RefitError(
            f"atomic RENAME_NOREPLACE failed: {os.strerror(error)}: "
            f"{source} -> {destination}"
        )


def _remove_owned_tree(path: Path, *, owner: Path) -> None:
    try:
        path.resolve().relative_to(owner.resolve())
    except (OSError, ValueError) as exc:
        raise RefitError(f"refusing to remove path outside owned workspace: {path}") from exc
    if path.is_symlink() or not path.is_dir():
        raise RefitError(f"owned cleanup target is not a real directory: {path}")
    entries = list(path.rglob("*"))
    for entry in entries:
        if entry.is_symlink():
            raise RefitError(f"owned cleanup tree contains a symlink: {entry}")
        if not entry.is_file() and not entry.is_dir():
            raise RefitError(f"owned cleanup tree contains a special entry: {entry}")
    path.chmod(0o700)
    for entry in entries:
        if entry.is_dir():
            entry.chmod(0o700)
    for entry in sorted(entries, key=lambda item: len(item.parts), reverse=True):
        if entry.is_file():
            entry.chmod(0o600)
            entry.unlink()
        elif entry.is_dir():
            entry.rmdir()
    path.rmdir()


def _package_rows(directory: Path, filenames: Sequence[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name in filenames:
        path = directory / name
        info = _regular_file(path, name="package artifact")
        rows.append({"file": name, "bytes": int(info.st_size), "sha256": _sha256_file(path)})
    return rows


def _package_manifest(
    *,
    spec: Mapping[str, Any],
    seal_id: str,
    h1: str,
    artifact_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "schema": PACKAGE_MANIFEST_SCHEMA,
        "artifact_id": spec["artifact_id"],
        "seal_id": seal_id,
        "h1_git_sha": h1,
        "lifecycle": spec["lifecycle"],
        "activation": False,
        "artifact_order": [dict(row) for row in artifact_rows],
        "completion": True,
        "claim_limit": spec["claim_limit"],
    }


def _package_stage_path(root: Path, spec: Mapping[str, Any], seal_id: str) -> Path:
    destination = root / str(spec["package_path"])
    return destination.parent / f".{destination.name}.stage-{seal_id}"


def _validate_package(
    root: Path,
    spec: Mapping[str, Any],
    seal_id: str,
    h1: str,
) -> dict[str, Any]:
    directory = root / str(spec["package_path"])
    if directory.is_symlink() or not directory.is_dir():
        raise RefitError(f"package directory is missing or symlinked: {directory}")
    names = list(spec["package_files"])
    actual = {path.name for path in directory.iterdir()}
    if actual != set(names) or len(actual) != len(names):
        raise RefitError("package file inventory differs")
    manifest_name = names[-1]
    _, manifest = _read_canonical(directory / manifest_name, name="package manifest")
    rows = _package_rows(directory, names[:-1])
    expected_manifest = _package_manifest(
        spec=spec, seal_id=seal_id, h1=h1, artifact_rows=rows
    )
    if manifest != expected_manifest:
        raise RefitError("package manifest differs from package bytes")
    _, strategy = _read_canonical(directory / names[3], name="refit strategy")
    _, resolved = _read_canonical(directory / names[4], name="resolved refit spec")
    if (
        not isinstance(strategy, dict)
        or strategy.get("schema") != STRATEGY_SCHEMA
        or strategy.get("artifact_id") != spec["artifact_id"]
        or strategy.get("seal_id") != seal_id
        or strategy.get("h1_git_sha") != h1
        or strategy.get("activation") is not False
    ):
        raise RefitError("package strategy identity differs")
    if (
        not isinstance(resolved, dict)
        or resolved.get("schema") != RESOLVED_SCHEMA
        or resolved.get("artifact_id") != spec["artifact_id"]
        or resolved.get("seal_id") != seal_id
        or resolved.get("h1_git_sha") != h1
        or resolved.get("activation") is not False
        or resolved.get("strategy_sha256") != _sha256_bytes(_canonical(strategy))
    ):
        raise RefitError("package resolved spec identity differs")
    expected_model_rows = resolved.get("models")
    strategy_model_rows = strategy.get("models")
    if (
        not isinstance(expected_model_rows, list)
        or len(expected_model_rows) != 3
        or not isinstance(strategy_model_rows, list)
        or len(strategy_model_rows) != 3
    ):
        raise RefitError("resolved model inventory is malformed")
    for position, row in enumerate(expected_model_rows):
        actual_row = rows[position]
        if (
            not isinstance(row, dict)
            or row.get("file") != actual_row["file"]
            or row.get("bytes") != actual_row["bytes"]
            or row.get("sha256") != actual_row["sha256"]
            or strategy_model_rows[position] != row
        ):
            raise RefitError("resolved/strategy model identity differs from package")
    return {
        "directory": directory,
        "manifest": manifest,
        "manifest_sha256": _sha256_file(directory / manifest_name),
        "strategy": strategy,
        "resolved": resolved,
        "artifact_rows": rows
        + [
            {
                "file": manifest_name,
                "bytes": int((directory / manifest_name).stat().st_size),
                "sha256": _sha256_file(directory / manifest_name),
            }
        ],
    }


def _install_package(
    *,
    root: Path,
    spec: Mapping[str, Any],
    seal_id: str,
    h1: str,
    canonical_phase: Mapping[str, Any],
) -> dict[str, Any]:
    destination = root / str(spec["package_path"])
    _ensure_real_directory(root, PurePosixPath(str(spec["package_path"])).parent.as_posix())
    stage = _package_stage_path(root, spec, seal_id)
    if stage.exists() or stage.is_symlink():
        raise RefitError("pre-existing package stage is not an authenticated recovery state")
    stage.mkdir(parents=True, mode=0o700)
    names = list(spec["package_files"])
    for position, checkpoint in enumerate(canonical_phase["checkpoints"]):
        source = Path(checkpoint.model_path)
        _atomic_new(stage / names[position], source.read_bytes())
    _atomic_new(stage / names[3], canonical_phase["strategy_bytes"])
    _atomic_new(stage / names[4], canonical_phase["resolved_bytes"])
    artifact_rows = _package_rows(stage, names[:-1])
    manifest = _package_manifest(
        spec=spec, seal_id=seal_id, h1=h1, artifact_rows=artifact_rows
    )
    _atomic_new(stage / names[-1], _canonical(manifest))
    if {path.name for path in stage.iterdir()} != set(names):
        raise RefitError("staged package contains undeclared files")
    stage.chmod(0o555)
    _fsync_dir(stage)
    try:
        _rename_noreplace(stage, destination)
        _fsync_dir(destination.parent)
    except FileExistsError:
        _remove_owned_tree(stage, owner=destination.parent)
    if stat.S_IMODE(destination.stat(follow_symlinks=False).st_mode) != 0o555:
        raise RefitError("installed package directory must be immutable mode 0555")
    package = _validate_package(root, spec, seal_id, h1)
    if package["manifest"] != manifest:
        raise RefitError("installed package differs from canonical stage")
    return package


def _success_result(
    *,
    spec: Mapping[str, Any],
    seal_id: str,
    h1: str,
    package: Mapping[str, Any],
) -> dict[str, Any]:
    resolved = package["resolved"]
    return {
        "schema": RESULT_SCHEMA,
        "status": SUCCESS,
        "artifact_id": spec["artifact_id"],
        "seal_id": seal_id,
        "h1_git_sha": h1,
        "lifecycle": spec["lifecycle"],
        "activation": False,
        "package_path": spec["package_path"],
        "package_manifest_sha256": package["manifest_sha256"],
        "source_manifest_path": spec["source_manifest_path"],
        "source_manifest_sha256": resolved["source_manifest_sha256"],
        "partition": resolved["partition"],
        "models": resolved["models"],
        "calibration_policy": resolved["calibration"],
        "canonical_audit_bytes_equal": True,
        "spent_window_efficacy_weight": 0,
        "no_efficacy_measurement": True,
        "claim_limit": spec["claim_limit"],
    }


def _failure_result(
    *,
    spec: Mapping[str, Any],
    seal_id: str,
    h1: str,
    comparisons: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": RESULT_SCHEMA,
        "status": NO_CANDIDATE,
        "artifact_id": spec["artifact_id"],
        "seal_id": seal_id,
        "h1_git_sha": h1,
        "lifecycle": spec["lifecycle"],
        "activation": False,
        "package_path": None,
        "reason": "canonical_audit_semantic_bytes_differ",
        "comparisons": dict(comparisons),
        "spent_window_efficacy_weight": 0,
        "no_efficacy_measurement": True,
        "claim_limit": spec["claim_limit"],
    }


def _expected_terminal(
    *,
    root: Path,
    spec: Mapping[str, Any],
    seal_id: str,
    h1: str,
    result: Mapping[str, Any],
) -> evidence.EvidenceEnvelope:
    status = result.get("status")
    result_ref = evidence.ArtifactRef.capture(str(spec["result_path"]), repo_root=root)
    artifacts: list[evidence.ArtifactRef] = [result_ref]
    package_path: str | None = None
    if status == SUCCESS:
        package_path = str(spec["package_path"])
        package = _validate_package(root, spec, seal_id, h1)
        artifacts.extend(
            evidence.ArtifactRef.capture(
                f"{package_path}/{name}", repo_root=root
            )
            for name in spec["package_files"]
        )
        if result != _success_result(
            spec=spec, seal_id=seal_id, h1=h1, package=package
        ):
            raise RefitError("success result differs from package")
    elif status == NO_CANDIDATE:
        package_destination = root / str(spec["package_path"])
        if package_destination.exists() or package_destination.is_symlink():
            raise RefitError("NO_CANDIDATE result conflicts with a package destination")
        comparisons = result.get("comparisons")
        if (
            not isinstance(comparisons, dict)
            or set(comparisons)
            != {
                "model_bytes_equal",
                "strategy_bytes_equal",
                "resolved_spec_bytes_equal",
            }
            or not isinstance(comparisons["model_bytes_equal"], list)
            or len(comparisons["model_bytes_equal"]) != 3
            or any(type(value) is not bool for value in comparisons["model_bytes_equal"])
            or type(comparisons["strategy_bytes_equal"]) is not bool
            or type(comparisons["resolved_spec_bytes_equal"]) is not bool
            or (
                all(comparisons["model_bytes_equal"])
                and comparisons["strategy_bytes_equal"]
                and comparisons["resolved_spec_bytes_equal"]
            )
            or result
            != _failure_result(
                spec=spec,
                seal_id=seal_id,
                h1=h1,
                comparisons=comparisons,
            )
        ):
            raise RefitError("NO_CANDIDATE result is not an exact completed mismatch")
    else:
        raise RefitError("terminal result status differs")
    payload = {
        "schema": TERMINAL_PAYLOAD_SCHEMA,
        "artifact_id": spec["artifact_id"],
        "seal_id": seal_id,
        "h1_git_sha": h1,
        "status": status,
        "lifecycle": spec["lifecycle"],
        "activation": False,
        "package_path": package_path,
        "no_efficacy_measurement": True,
        "claim_limit": spec["claim_limit"],
    }
    return evidence.EvidenceEnvelope.create(
        kind=spec["evidence_kinds"]["terminal"],
        payload=payload,
        artifacts=artifacts,
        dependencies=(seal_id,),
    )


def _load_result(root: Path, spec: Mapping[str, Any]) -> dict[str, Any]:
    _, value = _read_canonical(root / str(spec["result_path"]), name="refit result")
    if (
        not isinstance(value, dict)
        or value.get("schema") != RESULT_SCHEMA
        or value.get("artifact_id") != spec["artifact_id"]
        or value.get("activation") is not False
        or value.get("no_efficacy_measurement") is not True
        or value.get("status") not in spec["allowed_terminal_statuses"]
    ):
        raise RefitError("refit result identity differs")
    return value


def _publish_terminal(
    *,
    root: Path,
    spec: Mapping[str, Any],
    seal_id: str,
    h1: str,
    result: Mapping[str, Any],
) -> evidence.EvidenceEnvelope:
    expected = _expected_terminal(
        root=root, spec=spec, seal_id=seal_id, h1=h1, result=result
    )
    try:
        evidence.publish(expected, repo_root=root)
        actual = evidence.verify_object(expected.object_id, repo_root=root)
    except evidence.EvidenceError as exc:
        raise RefitError(f"terminal evidence publication failed: {exc}") from exc
    if actual.as_dict() != expected.as_dict():
        raise RefitError("terminal evidence differs after publication")
    return actual


def _admissible_output_status(
    root: Path,
    spec: Mapping[str, Any],
    seal_id: str,
    h1: str,
) -> None:
    status = _git_status(root)
    allowed = {str(spec["result_path"])}
    allowed.update(
        f"{spec['package_path']}/{name}" for name in spec["package_files"]
    )
    result_path = root / str(spec["result_path"])
    if result_path.exists() and not result_path.is_symlink():
        result = _load_result(root, spec)
        if result.get("seal_id") != seal_id or result.get("h1_git_sha") != h1:
            raise RefitError("existing result belongs to another seal/H1")
        expected = _expected_terminal(
            root=root, spec=spec, seal_id=seal_id, h1=h1, result=result
        )
        allowed.add(f"evidence/objects/{expected.object_id}.json")
    foreign = set(status) - allowed
    non_untracked = {path: code for path, code in status.items() if code != "??"}
    if foreign or non_untracked:
        raise RefitError(
            "fit recovery found foreign or staged worktree state: "
            f"foreign={sorted(foreign)} status={non_untracked}"
        )


def fit(
    seal_id: str,
    *,
    repo_root: str | os.PathLike[str] = REPO_ROOT,
    fault: Callable[[str], None] | None = None,
) -> evidence.EvidenceEnvelope:
    """Run or resume the one sealed six-seed refit cycle."""

    root = _validated_root(repo_root)
    spec = _load_spec(root)
    with _research_lock(root, spec):
        seal_envelope, source_manifest = _validate_seal(root, spec, seal_id)
        h1, _ = _authenticate_h1(root, spec, seal_envelope)
        _verify_shared_live(root, seal_envelope.payload["shared_git_identities"])
        runtime_contract = _verified_runtime_version_contract(
            root, str(seal_envelope.payload["h0_git_sha"])
        )
        if runtime_contract != seal_envelope.payload["runtime_version_contract"]:
            raise RefitError("live fit runtime differs from the sealed runtime")
        _admissible_output_status(root, spec, seal_id, h1)
        if _source_status(root, spec, source_manifest) != "match":
            raise RefitError("fit requires every sealed raw feature source")

        result_path = root / str(spec["result_path"])
        package_path = root / str(spec["package_path"])
        stage_path = _package_stage_path(root, spec, seal_id)
        post_fit_state = any(
            path.exists() or path.is_symlink()
            for path in (stage_path, package_path, result_path)
        )
        existing_result = (
            _load_result(root, spec)
            if result_path.exists() or result_path.is_symlink()
            else None
        )
        data = _build_rows_and_partitions(root, spec)
        if _source_status(root, spec, source_manifest) != "match":
            raise RefitError("raw feature source changed during matrix construction")
        _verify_shared_live(root, seal_envelope.payload["shared_git_identities"])
        canonical_phase = _fit_phase(
            root=root,
            spec=spec,
            seal_id=seal_id,
            h1=h1,
            phase="canonical",
            fit_rows=data["fit_rows"],
            fit_idx=data["fit_idx"],
            validation_rows=data["validation_rows"],
            early_idx=data["early_idx"],
            calibration_idx=data["calibration_idx"],
            partition=data["partition"],
            fault=fault,
            allow_fit=not post_fit_state,
        )
        audit_phase = _fit_phase(
            root=root,
            spec=spec,
            seal_id=seal_id,
            h1=h1,
            phase="audit",
            fit_rows=data["fit_rows"],
            fit_idx=data["fit_idx"],
            validation_rows=data["validation_rows"],
            early_idx=data["early_idx"],
            calibration_idx=data["calibration_idx"],
            partition=data["partition"],
            fault=fault,
            allow_fit=not post_fit_state,
        )
        _verify_shared_live(root, seal_envelope.payload["shared_git_identities"])
        comparisons = {
            "model_bytes_equal": [
                Path(left.model_path).read_bytes() == Path(right.model_path).read_bytes()
                for left, right in zip(
                    canonical_phase["checkpoints"], audit_phase["checkpoints"]
                )
            ],
            "strategy_bytes_equal": canonical_phase["strategy_bytes"]
            == audit_phase["strategy_bytes"],
            "resolved_spec_bytes_equal": canonical_phase["resolved_bytes"]
            == audit_phase["resolved_bytes"],
        }
        all_equal = (
            all(comparisons["model_bytes_equal"])
            and comparisons["strategy_bytes_equal"]
            and comparisons["resolved_spec_bytes_equal"]
        )
        if fault is not None:
            fault("after_audit")
        if not all_equal:
            if package_path.exists() or package_path.is_symlink():
                raise RefitError("audit mismatch conflicts with an existing package")
            result = _failure_result(
                spec=spec,
                seal_id=seal_id,
                h1=h1,
                comparisons=comparisons,
            )
            if existing_result is not None and existing_result != result:
                raise RefitError("existing result differs from authenticated audit mismatch")
        else:
            if existing_result is not None and existing_result.get("status") != SUCCESS:
                raise RefitError("existing result conflicts with authenticated audit equality")
            package = _install_package(
                root=root,
                spec=spec,
                seal_id=seal_id,
                h1=h1,
                canonical_phase=canonical_phase,
            )
            if fault is not None:
                fault("after_package_install")
            result = _success_result(
                spec=spec, seal_id=seal_id, h1=h1, package=package
            )
        _atomic_new(result_path, _canonical(result))
        if fault is not None:
            fault("after_result")
        terminal = _publish_terminal(
            root=root, spec=spec, seal_id=seal_id, h1=h1, result=result
        )
        if fault is not None:
            fault("after_terminal")
        _admissible_output_status(root, spec, seal_id, h1)
        return terminal


def _validate_h1_revision(
    root: Path,
    spec: Mapping[str, Any],
    seal_envelope: evidence.EvidenceEnvelope,
    h1: str,
) -> None:
    parents = _git(root, "rev-list", "--parents", "-n", "1", h1).split()
    h0 = str(seal_envelope.payload["h0_git_sha"])
    if len(parents) != 2 or parents != [h1, h0]:
        raise RefitError("recorded H1 does not have sealed H0 as its sole parent")
    changed = _git(root, "diff", "--name-only", h0, h1).splitlines()
    expected = sorted(
        [
            str(spec["source_manifest_path"]),
            f"evidence/objects/{seal_envelope.object_id}.json",
        ]
    )
    if sorted(changed) != expected or len(changed) != 2:
        raise RefitError("recorded H1 changed paths differ from the seal transaction")
    current = _git_head(root)
    if not _is_ancestor(root, h1, current):
        raise RefitError("current HEAD does not descend from recorded H1")


def verify(
    seal_id: str,
    *,
    repo_root: str | os.PathLike[str] = REPO_ROOT,
) -> dict[str, Any]:
    """Verify tracked lineage/package bytes without importing the fit adapter."""

    root = _validated_root(repo_root)
    spec = _load_spec(root)
    if _git(root, "branch", "--show-current") != "main":
        raise RefitError("verification requires branch main")
    seal_envelope, source_manifest = _validate_seal(root, spec, seal_id)
    result = _load_result(root, spec)
    if result.get("seal_id") != seal_id:
        raise RefitError("result seal ID differs")
    h1 = result.get("h1_git_sha")
    if not isinstance(h1, str) or re.fullmatch(r"[0-9a-f]{40}", h1) is None:
        raise RefitError("result H1 identity is malformed")
    _validate_h1_revision(root, spec, seal_envelope, h1)
    expected = _expected_terminal(
        root=root, spec=spec, seal_id=seal_id, h1=h1, result=result
    )
    try:
        terminal = evidence.verify_object(expected.object_id, repo_root=root)
    except evidence.EvidenceError as exc:
        raise RefitError(f"terminal verification failed: {exc}") from exc
    if terminal.as_dict() != expected.as_dict():
        raise RefitError("terminal evidence differs from domain reconstruction")
    _admissible_output_status(root, spec, seal_id, h1)
    source_status = _source_status(root, spec, source_manifest)
    return {
        "schema": "usdchf-m15-current-refit-verification/v1",
        "status": result["status"],
        "artifact_id": spec["artifact_id"],
        "seal_id": seal_id,
        "terminal_id": terminal.object_id,
        "h1_git_sha": h1,
        "source_status": source_status,
        "activation": False,
        "no_efficacy_measurement": True,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("seal", help="seal exact H0 code and source identities")
    fit_parser = commands.add_parser("fit", help="run or resume the serial refit")
    fit_parser.add_argument("--seal-id", required=True)
    verify_parser = commands.add_parser("verify", help="verify without fitting")
    verify_parser.add_argument("--seal-id", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "seal":
            envelope = seal(repo_root=args.repo_root)
            output = {
                "status": "SEALED_PRE_FIT",
                "seal_id": envelope.object_id,
                "activation": False,
            }
        elif args.command == "fit":
            envelope = fit(args.seal_id, repo_root=args.repo_root)
            output = {
                "status": envelope.payload["status"],
                "seal_id": args.seal_id,
                "terminal_id": envelope.object_id,
                "activation": False,
            }
        else:
            output = verify(args.seal_id, repo_root=args.repo_root)
        print(json.dumps(output, sort_keys=True, separators=(",", ":")))
        return 0
    except Exception as exc:
        print(f"BLOCKED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
