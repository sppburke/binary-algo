"""manifest.py — freeze a model "book" into a versioned, provenance-bearing manifest.

WHY THIS EXISTS (read MODEL_REGISTRY.md): the frozen GBM books in this program are trained multithreaded
(`n_jobs/thread_count=20`) WITHOUT a seed on the estimators, so they are NOT bit-for-bit retrainable — the
persisted artifact is the ONLY verbatim copy. So the reproducibility strategy is FREEZE-THE-ARTIFACT +
RECORD-A-MANIFEST (not config-driven re-derivation). A manifest is a RECORD (written at save time), not a
DRIVER (read by training). It binds a stable semantic id -> {git sha, hyperparams, feature/data fingerprint,
env versions, artifact sha256s, metrics} so a model can be referenced directly and re-loaded verbatim later.

Use at the end of a production train() to stamp the book:
    import manifest
    m = manifest.build(book_id="EURUSD.m15.v1", timeframe="15m", side="combined", role="direction",
                       script="m15_production.py", summary="...", metrics={...},
                       artifacts=[art("direction_lgb.txt"), art("direction_xgb.json"), art("direction_cat.cbm")],
                       hyperparams={...}, strategy_json="models/m15_EURUSD_strategy.json",
                       feature_fingerprint=manifest.dir_fingerprint(H.FEAT_DIR, ("EURUSD_*.parquet",)))
    manifest.freeze(m, artifacts_src=[...], books_dir="books")   # copies artifacts + writes books/<id>.manifest.json
"""
import ctypes
import errno
import fcntl
import glob
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator

# ROOT is retained for legacy build()/freeze() call compatibility.  Candidate
# publication uses the actual repository root below.
ROOT = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = Path(__file__).resolve().parents[1]

CANDIDATE_MANIFEST_SCHEMA = "book-manifest/candidate-v1"
CANDIDATE_RECEIPT_SCHEMA = "candidate-publish-receipt/v1"
CANDIDATE_LOCK_SCHEMA = "candidate-publish-lock/v1"
INACTIVE_CANDIDATE_LIFECYCLE = "inactive_shadow_candidate"
_PUBLICATION_LOCK_STATE = threading.local()


class CandidatePublicationError(RuntimeError):
    pass


class _CommitGuardFailure(RuntimeError):
    """Distinguish a caller guard rejection from an infrastructure failure."""

    def __init__(self, error: BaseException):
        super().__init__(str(error))
        self.error = error


def _run_commit_guard(commit_guard: Callable[[], None] | None) -> None:
    if commit_guard is None:
        return
    try:
        commit_guard()
    except BaseException as exc:
        raise _CommitGuardFailure(exc) from exc


def git_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return None


def git_dirty():
    try:
        return bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip())
    except Exception:
        return None


def file_sha256(path, buf=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(buf), b""):
            h.update(chunk)
    return h.hexdigest()


def dir_fingerprint(d, patterns=("*.parquet",)):
    """Fast content-change fingerprint of a data dir: hashes the sorted (name:size:mtime) listing.
    Does NOT read file contents (cheap), but detects any change to the feature set that produced a book."""
    files = []
    for p in patterns:
        files += glob.glob(os.path.join(d, p))
    files = sorted(set(files))
    listing, total = [], 0
    for f in files:
        st = os.stat(f)
        listing.append(f"{os.path.basename(f)}:{st.st_size}:{int(st.st_mtime)}")
        total += st.st_size
    return {"dir": d, "patterns": list(patterns), "n_files": len(files), "total_bytes": total,
            "sha256_of_listing": hashlib.sha256("\n".join(listing).encode()).hexdigest()}


def env_snapshot(py=None):
    """Capture the ACTUAL training-interpreter versions of the libraries that determine tree construction.
    (Hand-maintained notes drift — e.g. ENVIRONMENT_libs.txt said pandas 3.0.3 while the venv ran 2.3.3.)
    Uses importlib.metadata (works on pip-less uv venvs); reports the interpreter that imported this module."""
    from importlib import metadata as ilm
    keep = ["lightgbm", "xgboost", "catboost", "scikit-learn", "numpy", "pandas", "pyarrow", "scipy"]
    pinned = {}
    for name in keep:
        try:
            pinned[name] = ilm.version(name)
        except Exception:
            pinned[name] = None
    return {"python": py or sys.executable, "python_version": sys.version.split()[0], "key_libs": pinned}


def build(book_id, *, timeframe, side, role, script, summary, metrics, artifacts, hyperparams,
          strategy_json=None, feature_fingerprint=None, depends_on=None, notes=None, created_utc=None):
    """Assemble a manifest dict. `artifacts` = list of on-disk paths to the trained-model files (hashed here)."""
    arts = [{"file": os.path.basename(a), "bytes": os.path.getsize(a), "sha256": file_sha256(a)}
            for a in artifacts]
    # content_id = byte-unique fingerprint of the model (hash of the sorted artifact hashes). The semantic
    # `id` is the handle you cite; `content_id` is the proof two references are the same model bytes.
    content_id = hashlib.sha256("".join(sorted(a["sha256"] for a in arts)).encode()).hexdigest()[:16]
    return {
        "schema": "book-manifest/v1",
        "id": book_id,
        "content_id": content_id,
        "currency": "EURUSD",
        "timeframe": timeframe,        # "1-5s" | "60s" | "5m" | "10m" | "15m" | "30m" | "120s"
        "side": side,                  # "up" | "down" | "combined"
        "role": role,                  # "direction" | "magnitude" | "gate" | "stack-meta"
        "summary": summary,
        "metrics": metrics,            # e.g. {"oos_2026": 0.663, "combined": 0.647, "cpcv_mean": 0.579, "breakeven": 0.541}
        "source": {"script": script, "git_sha": git_sha(), "git_dirty": git_dirty()},
        "hyperparams": hyperparams,    # best-effort; source.script @ git_sha is AUTHORITATIVE for full config
        "artifacts": arts,             # {file, bytes, sha256} — the verbatim model bytes
        "strategy_json": strategy_json,
        "feature_fingerprint": feature_fingerprint,
        "env": env_snapshot(),
        "determinism": {
            "estimator_seed": "unset",
            "threads": "n_jobs/thread_count=20",
            "bitwise_retrainable": False,
            "note": "GBMs trained multithreaded without a seed -> NOT bit-for-bit retrainable; the persisted "
                    "artifact is the only verbatim copy. Re-running the script yields a DIFFERENT model.",
        },
        "depends_on": depends_on,      # e.g. a stack book that consumes a parent book's artifacts
        "notes": notes,
        "created_utc": created_utc,    # pass in to keep manifests stable across regenerations
    }


def freeze(m, artifacts_src, books_dir=os.path.join(ROOT, "books")):
    """Copy the trained-model artifacts into books/<id>/ (a version-controlled frozen snapshot) and write
    books/<id>.manifest.json. The copy doubles as an on-drive backup independent of the working models/ dir."""
    bid = m["id"]
    dest = os.path.join(books_dir, bid)
    os.makedirs(dest, exist_ok=True)
    for src in artifacts_src:
        shutil.copy2(src, os.path.join(dest, os.path.basename(src)))
    mpath = os.path.join(books_dir, f"{bid}.manifest.json")
    with open(mpath, "w") as f:
        json.dump(m, f, indent=2)
    return mpath


# ---------------------------------------------------------------------------
# Candidate-only publication.  This is intentionally separate from the v1
# build()/freeze() path above: old manifests and content_id semantics are part
# of the historical record and must not be rewritten.


def canonical_json_bytes(value: Any) -> bytes:
    """Canonical campaign JSON encoding (no trailing newline)."""
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise CandidatePublicationError(f"value is not canonical-JSON encodable: {exc}") from exc


def _is_hex_id(value: Any, lengths: tuple[int, ...] = (64,)) -> bool:
    return (
        isinstance(value, str)
        and len(value) in lengths
        and re.fullmatch(r"[0-9a-f]+", value) is not None
    )


def _validate_book_id(book_id: Any) -> str:
    if (
        not isinstance(book_id, str)
        or not book_id
        or Path(book_id).name != book_id
        or book_id in {".", ".."}
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", book_id) is None
    ):
        raise CandidatePublicationError(f"invalid candidate book id {book_id!r}")
    return book_id


def _validated_repo(repo_root: str | os.PathLike[str]) -> tuple[Path, Path]:
    root_arg = Path(repo_root)
    if root_arg.is_symlink():
        raise CandidatePublicationError(f"repository root must not be a symlink: {root_arg}")
    try:
        root = root_arg.resolve(strict=True)
    except OSError as exc:
        raise CandidatePublicationError(f"repository root does not exist: {root_arg}") from exc
    books = root / "books"
    if books.is_symlink() or not books.is_dir():
        raise CandidatePublicationError(f"candidate destination parent must be {root}/books")
    return root, books


def _require_real_directory_chain(root: Path, directory: Path, *, name: str) -> Path:
    """Require every directory component below an already-resolved root to be real."""

    try:
        relative = directory.relative_to(root)
    except ValueError as exc:
        raise CandidatePublicationError(f"{name} must be under repository root") from exc
    if any(part in {"", ".", ".."} for part in relative.parts):
        raise CandidatePublicationError(f"{name} contains an unsafe path component")
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink() or not current.is_dir():
            raise CandidatePublicationError(
                f"{name} must contain only existing non-symlink directories: {current}"
            )
    return current


@contextmanager
def _candidate_publication_lock(root: Path) -> Iterator[None]:
    """Serialize sanctioned transitions, re-entering only on the owning thread."""

    depth = getattr(_PUBLICATION_LOCK_STATE, "depth", 0)
    if depth:
        if getattr(_PUBLICATION_LOCK_STATE, "root", None) != root:
            raise CandidatePublicationError(
                "cannot re-enter a candidate publication lock for a different repository"
            )
        _PUBLICATION_LOCK_STATE.depth = depth + 1
        try:
            yield
        finally:
            _PUBLICATION_LOCK_STATE.depth -= 1
        return

    parent = _require_real_directory_chain(
        root,
        root / "logs" / "m15_book_refresh",
        name="candidate publication lock parent",
    )
    path = parent / ".candidate-publication.lock"
    if path.is_symlink():
        raise CandidatePublicationError("candidate publication lock must not be a symlink")
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags, 0o600)
    except OSError as exc:
        raise CandidatePublicationError(
            f"cannot open candidate publication lock {path}: {exc}"
        ) from exc
    try:
        descriptor_stat = os.fstat(fd)
        if (
            not stat.S_ISREG(descriptor_stat.st_mode)
            or descriptor_stat.st_nlink != 1
            or stat.S_IMODE(descriptor_stat.st_mode) != 0o600
        ):
            raise CandidatePublicationError(
                "candidate publication lock must be a private, singly linked regular file"
            )
        fcntl.flock(fd, fcntl.LOCK_EX)
        try:
            path_stat = path.stat(follow_symlinks=False)
        except OSError as exc:
            raise CandidatePublicationError(
                "candidate publication lock path disappeared"
            ) from exc
        if (
            path_stat.st_dev != descriptor_stat.st_dev
            or path_stat.st_ino != descriptor_stat.st_ino
        ):
            raise CandidatePublicationError(
                "candidate publication lock path changed during acquisition"
            )
        _fsync_dir(parent)
        _PUBLICATION_LOCK_STATE.root = root
        _PUBLICATION_LOCK_STATE.depth = 1
        try:
            yield
        finally:
            del _PUBLICATION_LOCK_STATE.depth
            del _PUBLICATION_LOCK_STATE.root
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def _regular_source(path: str | os.PathLike[str]) -> Path:
    src = Path(path)
    if src.is_symlink():
        raise CandidatePublicationError(f"candidate source must not be a symlink: {src}")
    try:
        resolved = src.resolve(strict=True)
    except OSError as exc:
        raise CandidatePublicationError(f"candidate source does not exist: {src}") from exc
    if not resolved.is_file():
        raise CandidatePublicationError(f"candidate source is not a regular file: {src}")
    return resolved


def _candidate_artifacts(artifacts: list[str | os.PathLike[str]]) -> list[dict[str, Any]]:
    if not isinstance(artifacts, list) or not artifacts:
        raise CandidatePublicationError("candidate artifacts must be a non-empty list")
    out: list[dict[str, Any]] = []
    names: set[str] = set()
    for raw in artifacts:
        path = _regular_source(raw)
        if path.name in names:
            raise CandidatePublicationError(f"duplicate candidate artifact filename {path.name}")
        names.add(path.name)
        out.append({"file": path.name, "bytes": path.stat().st_size, "sha256": file_sha256(path)})
    return out


def build_candidate_manifest(
    book_id: str,
    *,
    repo_root: str | os.PathLike[str] = REPO_ROOT,
    currency: str,
    timeframe: str,
    side: str,
    role: str,
    lifecycle_status: str,
    bundle_id: str,
    prereg_id: str,
    run_id: str,
    implementation_git_sha: str,
    source_script: str,
    determinism: dict[str, Any],
    artifacts: list[str | os.PathLike[str]],
    strategy_json: str,
    summary: str,
    metrics: dict[str, Any],
    hyperparams: dict[str, Any] | None = None,
    feature_fingerprint: dict[str, Any] | None = None,
    depends_on: Any = None,
    notes: Any = None,
    created_utc: str | None = None,
) -> dict[str, Any]:
    """Build a strict inactive-candidate manifest from explicit sealed inputs."""
    root, _ = _validated_repo(repo_root)
    bid = _validate_book_id(book_id)
    if lifecycle_status != INACTIVE_CANDIDATE_LIFECYCLE:
        raise CandidatePublicationError(
            f"candidate lifecycle must be {INACTIVE_CANDIDATE_LIFECYCLE!r}"
        )
    for name, value in (("bundle_id", bundle_id), ("prereg_id", prereg_id), ("run_id", run_id)):
        if not _is_hex_id(value):
            raise CandidatePublicationError(f"{name} must be 64 lowercase hex characters")
    if not _is_hex_id(implementation_git_sha, (40, 64)):
        raise CandidatePublicationError("implementation_git_sha must be 40 or 64 lowercase hex characters")
    if not isinstance(currency, str) or not currency:
        raise CandidatePublicationError("currency must be explicit")
    if (
        not isinstance(source_script, str)
        or not source_script
        or Path(source_script).is_absolute()
        or ".." in Path(source_script).parts
        or Path(source_script).as_posix() != source_script
    ):
        raise CandidatePublicationError("source_script must be a repo-relative POSIX path")
    if not isinstance(determinism, dict) or not determinism:
        raise CandidatePublicationError("actual determinism settings must be a non-empty object")
    if not isinstance(metrics, dict):
        raise CandidatePublicationError("candidate metrics must be an object")

    artifact_rows = _candidate_artifacts(artifacts)
    artifact_names = {row["file"] for row in artifact_rows}
    if Path(strategy_json).name != strategy_json or strategy_json not in artifact_names:
        raise CandidatePublicationError("strategy_json must name one declared candidate artifact")
    manifest = {
        "schema": CANDIDATE_MANIFEST_SCHEMA,
        "id": bid,
        "currency": currency,
        "timeframe": timeframe,
        "side": side,
        "role": role,
        "lifecycle_status": lifecycle_status,
        "bundle_id": bundle_id,
        "prereg_id": prereg_id,
        "run_id": run_id,
        "summary": summary,
        "metrics": metrics,
        "source": {
            "script": source_script,
            "implementation_git_sha": implementation_git_sha,
            "git_dirty": False,
            "repository": str(root.relative_to(root)),
        },
        "determinism": determinism,
        "hyperparams": hyperparams or {},
        "artifacts": artifact_rows,
        "strategy_json": strategy_json,
        "resolved_bundle_spec": f"{bid}.resolved_bundle_spec.json",
        "feature_fingerprint": feature_fingerprint,
        "depends_on": depends_on,
        "notes": notes,
        "created_utc": created_utc,
    }
    canonical_json_bytes(manifest)
    return manifest


def _write_new_bytes(path: Path, data: bytes) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        view = memoryview(data)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise CandidatePublicationError(f"short write for {path}")
            view = view[written:]
        os.fsync(fd)
    finally:
        os.close(fd)


def _atomic_write_new_read_only(path: Path, data: bytes) -> os.stat_result:
    """Atomically expose complete read-only bytes at a destination that must not exist."""

    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        view = memoryview(data)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise CandidatePublicationError(f"short write for {temporary}")
            view = view[written:]
        os.fchmod(fd, 0o444)
        os.fsync(fd)
        temporary_stat = os.fstat(fd)
        os.close(fd)
        fd = -1
        _rename_noreplace(temporary, path)
        _fsync_dir(path.parent)
        published_stat, published = _read_regular_identity(path)
        if (
            not _same_inode(published_stat, temporary_stat)
            or published != data
            or published_stat.st_mode & 0o222
        ):
            raise CandidatePublicationError(
                f"new read-only file differs immediately after commit: {path}"
            )
        return published_stat
    finally:
        if fd >= 0:
            os.close(fd)
        if temporary.exists():
            temporary.unlink()


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _fsync_tree(path: Path) -> None:
    dirs = [path]
    for child in sorted(path.rglob("*")):
        if child.is_symlink():
            raise CandidatePublicationError(f"candidate payload contains symlink {child}")
        if child.is_dir():
            dirs.append(child)
        elif child.is_file():
            fd = os.open(child, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        else:
            raise CandidatePublicationError(f"candidate payload contains non-regular entry {child}")
    for directory in reversed(dirs):
        _fsync_dir(directory)


def _make_tree_read_only(path: Path) -> None:
    for child in sorted(path.rglob("*"), reverse=True):
        if child.is_file():
            child.chmod(0o444)
        elif child.is_dir():
            child.chmod(0o555)
    path.chmod(0o555)
    _fsync_tree(path)


def _remove_owned_stage(path: Path) -> None:
    if not path.exists():
        return
    for child in path.rglob("*"):
        try:
            child.chmod(0o700 if child.is_dir() else 0o600)
        except OSError:
            pass
    try:
        path.chmod(0o700)
    except OSError:
        pass
    shutil.rmtree(path)


def _payload_hashes(candidate_dir: Path, receipt_name: str) -> dict[str, str]:
    payload: dict[str, str] = {}
    for path in sorted(candidate_dir.rglob("*")):
        if path.is_symlink():
            raise CandidatePublicationError(f"candidate payload contains symlink {path}")
        if path.is_dir():
            raise CandidatePublicationError(f"candidate payload must be flat; found directory {path}")
        if not path.is_file():
            raise CandidatePublicationError(f"candidate payload contains non-regular entry {path}")
        rel = path.relative_to(candidate_dir).as_posix()
        if rel == receipt_name:
            continue
        payload[rel] = file_sha256(path)
    return payload


def _load_canonical_object(path: Path) -> tuple[bytes, dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise CandidatePublicationError(f"required candidate file is not regular: {path}")
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise CandidatePublicationError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise CandidatePublicationError(f"{path} must contain a JSON object")
    if raw != canonical_json_bytes(value):
        raise CandidatePublicationError(f"{path} is not canonically encoded")
    return raw, value


def _validate_candidate_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("schema") != CANDIDATE_MANIFEST_SCHEMA:
        raise CandidatePublicationError("candidate manifest schema mismatch")
    _validate_book_id(manifest.get("id"))
    if manifest.get("lifecycle_status") != INACTIVE_CANDIDATE_LIFECYCLE:
        raise CandidatePublicationError("candidate manifest lifecycle mismatch")
    for name in ("bundle_id", "prereg_id", "run_id"):
        if not _is_hex_id(manifest.get(name)):
            raise CandidatePublicationError(f"candidate manifest {name} is invalid")
    source = manifest.get("source")
    if not isinstance(source, dict) or not _is_hex_id(source.get("implementation_git_sha"), (40, 64)):
        raise CandidatePublicationError("candidate manifest source implementation SHA is invalid")
    if source.get("git_dirty") is not False:
        raise CandidatePublicationError("candidate manifest source must be sealed clean")
    script = source.get("script")
    if (
        not isinstance(script, str)
        or not script
        or Path(script).is_absolute()
        or ".." in Path(script).parts
        or Path(script).as_posix() != script
    ):
        raise CandidatePublicationError("candidate source script must be a repo-relative POSIX path")
    for name in ("currency", "timeframe", "side", "role", "summary"):
        if not isinstance(manifest.get(name), str) or not manifest[name]:
            raise CandidatePublicationError(f"candidate manifest {name} must be a non-empty string")
    if manifest["timeframe"] != "15m" or manifest["side"] != "combined" \
            or manifest["role"] != "direction":
        raise CandidatePublicationError("candidate manifest policy kind is outside the refresh contract")
    if not isinstance(manifest.get("determinism"), dict) or not manifest["determinism"]:
        raise CandidatePublicationError("candidate manifest determinism is missing")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise CandidatePublicationError("candidate manifest artifacts are missing")
    seen: set[str] = set()
    for row in artifacts:
        if not isinstance(row, dict):
            raise CandidatePublicationError("candidate manifest artifact row must be an object")
        name = row.get("file")
        if not isinstance(name, str) or Path(name).name != name or name in seen:
            raise CandidatePublicationError(f"invalid candidate artifact filename {name!r}")
        seen.add(name)
        if not isinstance(row.get("bytes"), int) or row["bytes"] < 0 or not _is_hex_id(row.get("sha256")):
            raise CandidatePublicationError(f"invalid candidate artifact metadata for {name}")
    if manifest.get("strategy_json") not in seen:
        raise CandidatePublicationError("candidate strategy_json is not a declared artifact")


def _receipt_from_candidate_dir(candidate_dir: Path) -> tuple[bytes, dict[str, Any], dict[str, Any]]:
    if candidate_dir.is_symlink() or not candidate_dir.is_dir():
        raise CandidatePublicationError(f"candidate directory is not a real directory: {candidate_dir}")
    if candidate_dir.stat().st_mode & 0o222:
        raise CandidatePublicationError(f"candidate directory is not sealed read-only: {candidate_dir}")
    for entry in candidate_dir.rglob("*"):
        if entry.stat(follow_symlinks=False).st_mode & 0o222:
            raise CandidatePublicationError(f"candidate payload is not sealed read-only: {entry}")
    receipts = list(candidate_dir.glob("*.publish_receipt.json"))
    if len(receipts) != 1:
        raise CandidatePublicationError(
            f"candidate directory must contain exactly one publish receipt, found {len(receipts)}"
        )
    receipt_path = receipts[0]
    receipt_raw, receipt = _load_canonical_object(receipt_path)
    if receipt.get("schema") != CANDIDATE_RECEIPT_SCHEMA:
        raise CandidatePublicationError("candidate receipt schema mismatch")
    if set(receipt) != {
        "schema", "destination", "book_id", "prereg_id", "run_id", "bundle_id", "payload"
    }:
        raise CandidatePublicationError("candidate receipt fields differ from the sealed schema")
    book_id = _validate_book_id(receipt.get("book_id"))
    expected_name = f"{book_id}.publish_receipt.json"
    if receipt_path.name != expected_name:
        raise CandidatePublicationError("candidate receipt filename does not match book id")
    for name in ("bundle_id", "prereg_id", "run_id"):
        if not _is_hex_id(receipt.get(name)):
            raise CandidatePublicationError(f"candidate receipt {name} is invalid")
    expected_destination = f"books/{book_id}"
    if receipt.get("destination") != expected_destination:
        raise CandidatePublicationError("candidate receipt destination mismatch")
    payload = receipt.get("payload")
    if not isinstance(payload, dict) or expected_name in payload:
        raise CandidatePublicationError("candidate receipt payload is invalid or self-referential")
    actual_payload = _payload_hashes(candidate_dir, expected_name)
    if payload != actual_payload:
        raise CandidatePublicationError("candidate receipt payload hashes do not match directory bytes")

    manifest_path = candidate_dir / f"{book_id}.manifest.json"
    _, manifest = _load_canonical_object(manifest_path)
    _validate_candidate_manifest(manifest)
    spec_name = f"{book_id}.resolved_bundle_spec.json"
    _, spec = _load_canonical_object(candidate_dir / spec_name)
    if spec.get("schema") != "m15-book-refresh-resolved-bundle-spec/v1":
        raise CandidatePublicationError("candidate resolved bundle spec schema mismatch")
    if manifest.get("resolved_bundle_spec") != spec_name:
        raise CandidatePublicationError("candidate manifest resolved-spec filename mismatch")
    spec_book_id = spec.get("book_id", spec.get("id"))
    if spec_book_id != book_id:
        raise CandidatePublicationError("resolved bundle spec book id mismatch")
    for name in ("bundle_id", "prereg_id"):
        if manifest.get(name) != receipt.get(name):
            raise CandidatePublicationError(f"manifest/receipt {name} mismatch")
        if spec.get(name) != receipt.get(name):
            raise CandidatePublicationError(f"resolved-spec/receipt {name} mismatch")
    if manifest.get("run_id") != receipt.get("run_id"):
        raise CandidatePublicationError("manifest/receipt run_id mismatch")
    if "run_id" in spec:
        raise CandidatePublicationError("resolved bundle spec contains circular run_id material")
    if spec.get("lifecycle_status") != manifest.get("lifecycle_status"):
        raise CandidatePublicationError("resolved-spec/manifest lifecycle mismatch")
    declared_artifacts = {row["file"]: row for row in manifest["artifacts"]}
    for name, row in declared_artifacts.items():
        path = candidate_dir / name
        if path.is_symlink() or not path.is_file():
            raise CandidatePublicationError(f"declared candidate artifact missing: {name}")
        if path.stat().st_size != row["bytes"] or file_sha256(path) != row["sha256"]:
            raise CandidatePublicationError(f"declared candidate artifact changed: {name}")
    expected_files = set(declared_artifacts) | {manifest_path.name, spec_name}
    if set(actual_payload) != expected_files:
        raise CandidatePublicationError("candidate payload has missing or undeclared non-receipt files")
    return receipt_raw, receipt, manifest


def candidate_publish_lock(
    candidate_dir: str | os.PathLike[str],
) -> dict[str, Any]:
    """Return the lock material that must be sealed outside the candidate dir."""
    directory = Path(candidate_dir)
    receipt_raw, receipt, _ = _receipt_from_candidate_dir(directory)
    return {
        "schema": CANDIDATE_LOCK_SCHEMA,
        "destination": receipt["destination"],
        "book_id": receipt["book_id"],
        "prereg_id": receipt["prereg_id"],
        "run_id": receipt["run_id"],
        "bundle_id": receipt["bundle_id"],
        "receipt_sha256": hashlib.sha256(receipt_raw).hexdigest(),
        "payload": receipt["payload"],
    }


def stage_candidate(
    manifest: dict[str, Any],
    resolved_bundle_spec: dict[str, Any],
    artifacts_src: list[str | os.PathLike[str]],
    *,
    repo_root: str | os.PathLike[str] = REPO_ROOT,
) -> dict[str, Any]:
    """Build and fsync one complete same-filesystem candidate publication stage."""
    _, books = _validated_repo(repo_root)
    _validate_candidate_manifest(manifest)
    book_id = manifest["id"]
    destination = books / book_id
    if destination.exists() or destination.is_symlink():
        raise CandidatePublicationError(f"candidate destination already exists: {destination}")
    if not isinstance(resolved_bundle_spec, dict):
        raise CandidatePublicationError("resolved bundle spec must be an object")
    if resolved_bundle_spec.get("schema") != "m15-book-refresh-resolved-bundle-spec/v1":
        raise CandidatePublicationError("candidate resolved bundle spec schema mismatch")
    spec_book_id = resolved_bundle_spec.get("book_id", resolved_bundle_spec.get("id"))
    if spec_book_id != book_id or resolved_bundle_spec.get("bundle_id") != manifest["bundle_id"]:
        raise CandidatePublicationError("resolved bundle spec identity does not match candidate manifest")
    if resolved_bundle_spec.get("prereg_id") != manifest["prereg_id"]:
        raise CandidatePublicationError("resolved bundle spec prereg_id mismatch")
    if resolved_bundle_spec.get("lifecycle_status") != manifest["lifecycle_status"]:
        raise CandidatePublicationError("resolved bundle spec lifecycle mismatch")
    # run_id is derived from the already-sealed bundle IDs and therefore cannot
    # be bundle-identity material without a hash cycle.  It is bound by the
    # manifest, receipt, and externally sealed joint-result publication lock.
    if "run_id" in resolved_bundle_spec:
        raise CandidatePublicationError("resolved bundle spec must not contain circular run_id material")

    sources = [_regular_source(path) for path in artifacts_src]
    by_name = {path.name: path for path in sources}
    if len(by_name) != len(sources):
        raise CandidatePublicationError("candidate artifact sources contain duplicate basenames")
    declared = {row["file"]: row for row in manifest["artifacts"]}
    if set(by_name) != set(declared):
        raise CandidatePublicationError("candidate artifact sources do not exactly match manifest")

    stage = Path(tempfile.mkdtemp(prefix=f".{book_id}.stage-", dir=books))
    try:
        for name in sorted(by_name):
            source = by_name[name]
            row = declared[name]
            if source.stat().st_size != row["bytes"] or file_sha256(source) != row["sha256"]:
                raise CandidatePublicationError(f"candidate source changed after manifest build: {source}")
            target = stage / name
            with source.open("rb") as src:
                _write_new_bytes(target, src.read())
            if target.stat().st_size != row["bytes"] or file_sha256(target) != row["sha256"]:
                raise CandidatePublicationError(f"candidate staged copy failed verification: {name}")

        spec_name = f"{book_id}.resolved_bundle_spec.json"
        manifest_name = f"{book_id}.manifest.json"
        receipt_name = f"{book_id}.publish_receipt.json"
        _write_new_bytes(stage / spec_name, canonical_json_bytes(resolved_bundle_spec))
        _write_new_bytes(stage / manifest_name, canonical_json_bytes(manifest))
        payload = _payload_hashes(stage, receipt_name)
        receipt = {
            "schema": CANDIDATE_RECEIPT_SCHEMA,
            "destination": f"books/{book_id}",
            "book_id": book_id,
            "prereg_id": manifest["prereg_id"],
            "run_id": manifest["run_id"],
            "bundle_id": manifest["bundle_id"],
            "payload": payload,
        }
        _write_new_bytes(stage / receipt_name, canonical_json_bytes(receipt))
        _fsync_tree(stage)
        _make_tree_read_only(stage)
        lock = candidate_publish_lock(stage)
        return {
            "stage_dir": str(stage),
            "destination": f"books/{book_id}",
            "receipt_path": str(stage / receipt_name),
            "publish_lock": lock,
        }
    except Exception:
        _remove_owned_stage(stage)
        raise


def _coerce_publish_lock(
    source: dict[str, Any] | str | os.PathLike[str],
    book_id: str,
    *,
    repo_root: Path,
) -> dict[str, Any]:
    if isinstance(source, dict):
        raise CandidatePublicationError(
            "publication requires a sealed external joint-result file, not an in-memory lock"
        )
    path = Path(source)
    if not path.is_absolute():
        path = repo_root / path
    expected_parent = repo_root / "results" / "json"
    if path.parent != expected_parent:
        raise CandidatePublicationError(
            "external publication lock is not the exact tracked direct file in repo results/json"
        )
    _require_real_directory_chain(
        repo_root, expected_parent, name="external joint-result parent"
    )
    if path.is_symlink() or not path.is_file():
        raise CandidatePublicationError(f"external publication lock is not a regular file: {path}")
    path_stat, raw_bytes = _read_regular_identity(path)
    if path_stat.st_mode & 0o222:
        raise CandidatePublicationError(f"external publication lock must be read-only: {path}")
    try:
        joint = json.loads(raw_bytes)
    except (OSError, json.JSONDecodeError) as exc:
        raise CandidatePublicationError(f"cannot read external publication lock: {exc}") from exc
    if not isinstance(joint, dict) or raw_bytes != canonical_json_bytes(joint):
        raise CandidatePublicationError("external joint result must be one canonical JSON object")
    if joint.get("schema") != "m15-book-refresh-joint-replay-result/v1":
        raise CandidatePublicationError("external publication lock is not a joint replay result")
    if not _is_hex_id(joint.get("run_id")) or not _is_hex_id(joint.get("prereg_id")):
        raise CandidatePublicationError("external joint result identities are malformed")
    expected_path = (
        repo_root / "results" / "json"
        / f"m15_book_refresh_{joint['run_id']}_joint_replay_result.json"
    )
    if path != expected_path:
        raise CandidatePublicationError(
            "external publication lock is not the exact tracked campaign joint-result path"
        )
    survivors = joint.get("S")
    if not isinstance(survivors, list) or book_id not in survivors or len(set(survivors)) != len(survivors):
        raise CandidatePublicationError(f"{book_id} is not uniquely authorized by sealed survivor set S")
    locks = joint.get("candidate_publish_locks")
    if not isinstance(locks, dict) or len(locks) != len(survivors) or set(locks) != set(survivors):
        raise CandidatePublicationError("candidate lock keys must exactly equal ordered survivor set S")
    statuses = joint.get("statuses")
    candidate_statuses = joint.get("candidate_statuses")
    candidate_status = (
        candidate_statuses.get(book_id) if isinstance(candidate_statuses, dict) else None
    )
    if (
        not isinstance(statuses, dict)
        or not isinstance(candidate_statuses, dict)
        or set(candidate_statuses) != set(survivors)
        or not isinstance(candidate_status, dict)
        or candidate_status.get("status") != "PROMOTE_TO_SHADOW"
    ):
        raise CandidatePublicationError("joint result does not carry the survivor promotion status")
    pair = candidate_status.get("pair")
    pair_status = statuses.get(pair) if isinstance(pair, str) else None
    if not isinstance(pair_status, dict) or pair_status.get("status") != "PROMOTE_TO_SHADOW":
        raise CandidatePublicationError("candidate promotion status differs from pair status")
    raw = locks.get(book_id) if isinstance(locks, dict) else None
    if not isinstance(raw, dict):
        raise CandidatePublicationError(f"external publication lock for {book_id} is missing or ambiguous")
    if raw.get("schema") != CANDIDATE_LOCK_SCHEMA:
        raise CandidatePublicationError("external publication lock schema mismatch")
    if set(raw) != {
        "schema", "destination", "book_id", "prereg_id", "run_id", "bundle_id",
        "receipt_sha256", "payload",
    }:
        raise CandidatePublicationError("external candidate lock fields differ from sealed schema")
    if raw.get("run_id") != joint["run_id"] or raw.get("prereg_id") != joint["prereg_id"]:
        raise CandidatePublicationError("candidate lock identities differ from joint result")
    return raw


def _authenticate_external_lock(
    candidate_dir: Path,
    lock_source: dict[str, Any] | str | os.PathLike[str],
    expected_book_id: str | None = None,
    repo_root: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    receipt_raw, receipt, manifest = _receipt_from_candidate_dir(candidate_dir)
    book_id = receipt["book_id"]
    if expected_book_id is not None and book_id != expected_book_id:
        raise CandidatePublicationError("candidate directory book id does not match requested recovery")
    if repo_root is None:
        raise CandidatePublicationError("external lock authentication requires a validated repo root")
    lock = _coerce_publish_lock(lock_source, book_id, repo_root=repo_root)
    for name in ("destination", "book_id", "prereg_id", "run_id", "bundle_id", "payload"):
        if lock.get(name) != receipt.get(name):
            raise CandidatePublicationError(f"external publication lock {name} mismatch")
    expected_receipt_sha = hashlib.sha256(receipt_raw).hexdigest()
    if lock.get("receipt_sha256") != expected_receipt_sha:
        raise CandidatePublicationError("external publication lock receipt byte hash mismatch")
    return lock, manifest


def _renameat2(source: Path, destination: Path, *, flags: int, operation: str) -> None:
    """Call Linux renameat2 with an explicit operation and fail closed."""

    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise CandidatePublicationError(
            f"renameat2({operation}) is unavailable; refusing unsafe filesystem transition"
        )
    renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    renameat2.restype = ctypes.c_int
    rc = renameat2(-100, os.fsencode(source), -100, os.fsencode(destination), flags)
    if rc != 0:
        err = ctypes.get_errno()
        if flags == 1 and err == errno.EEXIST:
            raise CandidatePublicationError(f"candidate destination already exists: {destination}")
        raise CandidatePublicationError(
            f"atomic {operation} failed: {os.strerror(err)} ({source} -> {destination})"
        )


def _rename_noreplace(source: Path, destination: Path) -> None:
    """Linux atomic directory publish with a kernel-enforced no-replace flag."""

    _renameat2(
        source,
        destination,
        flags=1,  # RENAME_NOREPLACE
        operation="RENAME_NOREPLACE",
    )


def _rename_exchange(source: Path, destination: Path) -> None:
    """Atomically exchange two existing paths on the same filesystem."""

    _renameat2(
        source,
        destination,
        flags=2,  # RENAME_EXCHANGE
        operation="RENAME_EXCHANGE",
    )


def candidate_index_entry(manifest: dict[str, Any]) -> dict[str, Any]:
    """Deterministic INDEX projection from an authenticated candidate manifest."""
    _validate_candidate_manifest(manifest)
    book_id = manifest["id"]
    return {
        "id": book_id,
        "currency": manifest["currency"],
        "timeframe": manifest["timeframe"],
        "side": manifest["side"],
        "role": manifest["role"],
        "script": manifest["source"]["script"],
        "metrics": manifest["metrics"],
        "manifest": f"books/{book_id}/{book_id}.manifest.json",
        "summary": manifest["summary"],
        "depends_on": manifest.get("depends_on"),
        "lifecycle_status": manifest["lifecycle_status"],
        "bundle_id": manifest["bundle_id"],
        "prereg_id": manifest["prereg_id"],
        "run_id": manifest["run_id"],
    }


def _read_index(index_path: Path) -> tuple[bytes, dict[str, Any], list[dict[str, Any]]]:
    if index_path.is_symlink() or not index_path.is_file():
        raise CandidatePublicationError(f"INDEX is not a regular file: {index_path}")
    try:
        raw_bytes = index_path.read_bytes()
        raw = json.loads(raw_bytes)
    except (OSError, json.JSONDecodeError) as exc:
        raise CandidatePublicationError(f"cannot read INDEX: {exc}") from exc
    if not isinstance(raw, dict) or raw.get("schema") != "book-index/v1" \
            or not isinstance(raw.get("books"), list):
        raise CandidatePublicationError("INDEX must contain a top-level books list")
    books = raw["books"]
    ids: set[str] = set()
    for row in books:
        if not isinstance(row, dict) or not isinstance(row.get("id"), str):
            raise CandidatePublicationError("INDEX contains a malformed book entry")
        if row["id"] in ids:
            raise CandidatePublicationError(f"INDEX contains duplicate id {row['id']}")
        ids.add(row["id"])
    return raw_bytes, raw, books


def _read_regular_identity(path: Path) -> tuple[os.stat_result, bytes]:
    """Read one non-symlink regular-file inode and prove its name stayed bound."""

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise CandidatePublicationError(f"cannot open regular file {path}: {exc}") from exc
    try:
        descriptor_stat = os.fstat(fd)
        if not stat.S_ISREG(descriptor_stat.st_mode):
            raise CandidatePublicationError(f"path is not a regular file: {path}")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(fd, 1 << 20)
            if not chunk:
                break
            chunks.append(chunk)
        try:
            path_stat = path.stat(follow_symlinks=False)
        except OSError as exc:
            raise CandidatePublicationError(
                f"regular-file name changed while reading: {path}"
            ) from exc
        if (
            path_stat.st_dev != descriptor_stat.st_dev
            or path_stat.st_ino != descriptor_stat.st_ino
        ):
            raise CandidatePublicationError(
                f"regular-file name changed while reading: {path}"
            )
        return descriptor_stat, b"".join(chunks)
    finally:
        os.close(fd)


def _same_inode(left: os.stat_result, right: os.stat_result) -> bool:
    return left.st_dev == right.st_dev and left.st_ino == right.st_ino


def _atomic_replace_bytes(
    path: Path,
    data: bytes,
    *,
    expected_current: bytes,
    expected_identity: os.stat_result | None = None,
) -> os.stat_result:
    """Replace exact expected bytes without silently overwriting a competing write.

    The advisory publication lock serializes cooperating writers only.  The
    kernel exchange makes the displaced target available for authentication;
    if it is not the exact preimage, the exchange is rolled back before this
    function reports a conflict.
    """

    if path.is_symlink():
        raise CandidatePublicationError(f"refusing to replace symlink {path}")
    try:
        original = path.stat(follow_symlinks=False)
    except OSError as exc:
        raise CandidatePublicationError(f"cannot stat replacement target {path}: {exc}") from exc
    if not stat.S_ISREG(original.st_mode):
        raise CandidatePublicationError(f"replacement target is not a regular file: {path}")
    if expected_identity is not None and not _same_inode(original, expected_identity):
        raise CandidatePublicationError(
            f"replacement target identity differs from authenticated CAS value: {path}"
        )
    original_mode = stat.S_IMODE(original.st_mode)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent)
    tmp = Path(tmp_name)
    preserve_temporary = False
    try:
        view = memoryview(data)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise CandidatePublicationError(f"short write for {tmp}")
            view = view[written:]
        os.fchmod(fd, original_mode)
        os.fsync(fd)
        candidate_stat = os.fstat(fd)
        os.close(fd)
        fd = -1
        current_stat, current = _read_regular_identity(path)
        if (
            not _same_inode(current_stat, original)
            or (
                expected_identity is not None
                and not _same_inode(current_stat, expected_identity)
            )
            or current != expected_current
        ):
            raise CandidatePublicationError(
                f"replacement target bytes or identity changed before commit: {path}"
            )

        _rename_exchange(tmp, path)
        preserve_temporary = True
        _fsync_dir(path.parent)

        published_stat, published = _read_regular_identity(path)
        displaced_stat, displaced = _read_regular_identity(tmp)
        if not _same_inode(published_stat, candidate_stat) or published != data:
            raise CandidatePublicationError(
                f"replacement target changed immediately after atomic exchange: {path}; "
                f"displaced preimage retained at {tmp}"
            )

        if _same_inode(displaced_stat, original) and displaced == expected_current:
            # Re-authenticate our exposed inode immediately before discarding the
            # exact displaced preimage. A competing post-exchange replacement is
            # never rolled back over or reported as our successful write.
            final_stat, final = _read_regular_identity(path)
            if not _same_inode(final_stat, candidate_stat) or final != data:
                raise CandidatePublicationError(
                    f"replacement target changed before commit completed: {path}; "
                    f"displaced preimage retained at {tmp}"
                )
            tmp.unlink()
            preserve_temporary = False
            _fsync_dir(path.parent)
            return candidate_stat

        # A non-cooperating writer changed the target after our initial read.
        # Exchange the actual displaced bytes back, but only while the public
        # name is still bound to our known candidate inode.
        rollback_guard_stat, rollback_guard = _read_regular_identity(path)
        if not _same_inode(rollback_guard_stat, candidate_stat) or rollback_guard != data:
            raise CandidatePublicationError(
                f"replacement conflict could not be rolled back safely: {path}; "
                f"displaced bytes retained at {tmp}"
            )
        _rename_exchange(tmp, path)
        _fsync_dir(path.parent)
        restored_stat, restored = _read_regular_identity(path)
        rollback_tmp_stat, rollback_tmp = _read_regular_identity(tmp)
        if (
            not _same_inode(restored_stat, displaced_stat)
            or restored != displaced
            or not _same_inode(rollback_tmp_stat, candidate_stat)
            or rollback_tmp != data
        ):
            raise CandidatePublicationError(
                f"replacement conflict rollback could not be authenticated: {path}; "
                f"temporary evidence retained at {tmp}"
            )
        tmp.unlink()
        preserve_temporary = False
        _fsync_dir(path.parent)
        raise CandidatePublicationError(
            f"replacement target changed before atomic commit; competing bytes restored: {path}"
        )
    finally:
        if fd >= 0:
            os.close(fd)
        if not preserve_temporary and os.path.lexists(tmp):
            tmp.unlink()


def _atomic_remove_created_bytes(
    path: Path,
    *,
    expected_current: bytes,
    expected_identity: os.stat_result,
) -> None:
    """Remove an invocation-owned new file without unlinking competing bytes."""

    current_stat, current = _read_regular_identity(path)
    if (
        not _same_inode(current_stat, expected_identity)
        or current != expected_current
    ):
        raise CandidatePublicationError(
            f"created-file rollback CAS differs; conflict evidence retained at {path}"
        )

    fd, evidence_name = tempfile.mkstemp(
        prefix=f".{path.name}.rollback-", dir=path.parent
    )
    os.close(fd)
    evidence = Path(evidence_name)
    evidence.unlink()
    try:
        _rename_noreplace(path, evidence)
        _fsync_dir(path.parent)
    except Exception:
        # The public name was not mutated when RENAME_NOREPLACE failed.
        raise

    moved_stat, moved = _read_regular_identity(evidence)
    if (
        _same_inode(moved_stat, expected_identity)
        and moved == expected_current
    ):
        evidence.unlink()
        _fsync_dir(path.parent)
        return

    if not os.path.lexists(path):
        try:
            _rename_noreplace(evidence, path)
            _fsync_dir(path.parent)
        except Exception as exc:
            raise CandidatePublicationError(
                f"created-file rollback conflict could not be restored; "
                f"evidence retained at {evidence}"
            ) from exc
        restored_stat, restored = _read_regular_identity(path)
        if not _same_inode(restored_stat, moved_stat) or restored != moved:
            raise CandidatePublicationError(
                f"created-file rollback conflict restoration could not be "
                f"authenticated; evidence retained at {path}"
            )
        raise CandidatePublicationError(
            f"created-file rollback moved competing bytes; conflict evidence "
            f"restored at {path}"
        )
    raise CandidatePublicationError(
        f"created-file rollback found a competing public name; conflict "
        f"evidence retained at {evidence} and {path}"
    )


def _registry_entry_present_exactly_once(
    current: bytes,
    entry: bytes,
    book_id: bytes,
) -> bool:
    """Return exact whole-line presence; reject every ambiguous registration.

    Candidate registration entries are deliberately one physical Markdown
    line.  A substring, duplicate exact line, or second line mentioning the
    same book is a conflict rather than an idempotent recovery match.
    """
    if not entry or b"\n" in entry or b"\r" in entry:
        raise CandidatePublicationError("registry entry must be exactly one non-empty line")
    lines = current.splitlines()
    exact_count = sum(line == entry for line in lines)
    book_lines = [line for line in lines if book_id in line]
    if exact_count > 1 or len(book_lines) > 1:
        raise CandidatePublicationError("registry contains duplicate or ambiguous candidate entries")
    if exact_count == 1:
        if book_lines != [entry]:
            raise CandidatePublicationError("registry contains conflicting candidate text")
        return True
    if book_lines:
        raise CandidatePublicationError("registry contains conflicting candidate text")
    return False


def _registration_state(
    manifest: dict[str, Any],
    *,
    root: Path,
    registry_path: str | os.PathLike[str] | None,
    registry_entry: str | bytes | None,
    phase_seal_path: str | os.PathLike[str] | None,
    phase_seal: dict[str, Any] | None,
) -> dict[str, Any]:
    if registry_path is None or registry_entry is None:
        raise CandidatePublicationError(
            "candidate publication requires MODEL_REGISTRY path and exact entry"
        )
    if phase_seal_path is None or phase_seal is None:
        raise CandidatePublicationError(
            "candidate publication requires the exact campaign phase seal"
        )
    expected = candidate_index_entry(manifest)
    index_path = root / "books" / "INDEX.json"
    index_current, index_raw, books = _read_index(index_path)
    matches = [row for row in books if row["id"] == expected["id"]]
    if matches and canonical_json_bytes(matches[0]) != canonical_json_bytes(expected):
        raise CandidatePublicationError(f"conflicting INDEX registration for {expected['id']}")
    state: dict[str, Any] = {
        "index_path": index_path,
        "index_current": index_current,
        "index_raw": index_raw,
        "index_missing": not matches,
        "expected_index_entry": expected,
        "registry_missing": False,
        "phase_missing": False,
    }

    path = Path(registry_path)
    if not path.is_absolute():
        path = root / path
    expected_registry_path = root / "MODEL_REGISTRY.md"
    if path != expected_registry_path:
        raise CandidatePublicationError("registry path must be exact repo-root MODEL_REGISTRY.md")
    _require_real_directory_chain(root, path.parent, name="MODEL_REGISTRY parent")
    if path.is_symlink() or not path.is_file():
        raise CandidatePublicationError("registry path must be an existing regular file")
    entry_bytes = registry_entry.encode("utf-8") if isinstance(registry_entry, str) else bytes(registry_entry)
    required_registry_material = (
        expected["id"], expected["bundle_id"], expected["prereg_id"], expected["run_id"],
        "lifecycle_status=inactive_shadow_candidate", "inactive", "no activation",
    )
    if any(value.encode() not in entry_bytes for value in required_registry_material):
        raise CandidatePublicationError(
            "registry entry must bind all identities and the inactive/no-activation lifecycle"
        )
    current = path.read_bytes()
    exact = _registry_entry_present_exactly_once(
        current, entry_bytes, expected["id"].encode()
    )
    state.update(
        registry_path=path,
        registry_current=current,
        registry_entry=entry_bytes,
        registry_missing=not exact,
    )

    if not isinstance(phase_seal, dict):
        raise CandidatePublicationError("phase seal must be an object")
    expected_phase_fields = {
        "schema",
        "status",
        "lifecycle_status",
        "book_id",
        "bundle_id",
        "prereg_id",
        "run_id",
    }
    if set(phase_seal) != expected_phase_fields:
        raise CandidatePublicationError(
            "candidate phase seal fields differ from the exact no-activation schema"
        )
    if phase_seal.get("schema") != "m15-book-refresh-candidate-publication/v1" \
            or phase_seal.get("status") != "sealed" \
            or phase_seal.get("lifecycle_status") != INACTIVE_CANDIDATE_LIFECYCLE:
        raise CandidatePublicationError("candidate phase seal schema/status/lifecycle mismatch")
    for name in ("book_id", "bundle_id", "prereg_id", "run_id"):
        if phase_seal.get(name) != manifest.get(name if name != "book_id" else "id"):
            raise CandidatePublicationError(f"phase seal {name} mismatch")
    path = Path(phase_seal_path)
    if not path.is_absolute():
        path = root / path
    if path.is_symlink():
        raise CandidatePublicationError("phase seal path must not be a symlink")
    expected_phase_path = (
        root / "logs" / "m15_book_refresh" / manifest["prereg_id"]
        / "candidate_publication" / f"{manifest['id']}.json"
    )
    if path != expected_phase_path:
        raise CandidatePublicationError("candidate phase seal path differs from campaign contract")
    _require_real_directory_chain(
        root, expected_phase_path.parent, name="candidate phase-seal parent"
    )
    expected_bytes = canonical_json_bytes(phase_seal)
    if path.exists():
        if path.read_bytes() != expected_bytes:
            raise CandidatePublicationError(f"conflicting phase seal at {path}")
        if not path.is_file() or path.stat().st_mode & 0o222:
            raise CandidatePublicationError(f"existing phase seal is not read-only: {path}")
    state.update(phase_path=path, phase_bytes=expected_bytes, phase_missing=not path.exists())
    return state


def _rollback_registration_mutations(
    mutations: list[dict[str, Any]],
) -> list[str]:
    errors: list[str] = []
    for mutation in reversed(mutations):
        try:
            if mutation["kind"] == "replace":
                _atomic_replace_bytes(
                    mutation["path"],
                    mutation["before"],
                    expected_current=mutation["after"],
                    expected_identity=mutation["after_identity"],
                )
            elif mutation["kind"] == "create":
                _atomic_remove_created_bytes(
                    mutation["path"],
                    expected_current=mutation["after"],
                    expected_identity=mutation["after_identity"],
                )
            else:
                raise CandidatePublicationError(
                    f"unknown publication rollback mutation {mutation['kind']!r}"
                )
        except Exception as exc:
            errors.append(f"{mutation['path']}: {exc}")
    return errors


def _require_publish_preflight_restored(preflight: dict[str, Any]) -> None:
    current_index, _, _ = _read_index(preflight["index_path"])
    if current_index != preflight["index_current"]:
        raise CandidatePublicationError(
            "INDEX does not match the authenticated pre-publication preimage"
        )
    _, current_registry = _read_regular_identity(preflight["registry_path"])
    if current_registry != preflight["registry_current"]:
        raise CandidatePublicationError(
            "MODEL_REGISTRY does not match the authenticated pre-publication preimage"
        )
    if os.path.lexists(preflight["phase_path"]):
        raise CandidatePublicationError(
            "phase-seal path is not absent after publication rollback"
        )


def _restore_published_stage(
    *,
    stage: Path,
    destination: Path,
    expected_identity: os.stat_result,
    external_lock: dict[str, Any] | str | os.PathLike[str],
    lock: dict[str, Any],
    manifest: dict[str, Any],
    root: Path,
    books: Path,
) -> None:
    """Restore only the exact invocation-owned destination to its old stage name."""

    if os.path.lexists(stage):
        raise CandidatePublicationError(
            f"candidate stage name is occupied; published evidence retained at {destination}"
        )
    try:
        current_identity = destination.stat(follow_symlinks=False)
    except OSError as exc:
        raise CandidatePublicationError(
            f"published destination disappeared before rollback: {destination}"
        ) from exc
    if (
        not stat.S_ISDIR(current_identity.st_mode)
        or not _same_inode(current_identity, expected_identity)
    ):
        raise CandidatePublicationError(
            f"published destination rollback CAS differs; conflict evidence retained "
            f"at {destination}"
        )
    current_lock, current_manifest = _authenticate_external_lock(
        destination,
        external_lock,
        expected_book_id=manifest["id"],
        repo_root=root,
    )
    if current_lock != lock or current_manifest != manifest:
        raise CandidatePublicationError(
            f"published destination rollback authentication differs; evidence retained "
            f"at {destination}"
        )

    _rename_noreplace(destination, stage)
    _fsync_dir(books)
    try:
        restored_identity = stage.stat(follow_symlinks=False)
    except OSError as exc:
        raise CandidatePublicationError(
            f"published destination rollback lost the moved evidence at {stage}"
        ) from exc
    if not _same_inode(restored_identity, expected_identity):
        if os.path.lexists(destination):
            raise CandidatePublicationError(
                "published destination rollback moved a competing directory; "
                f"conflict evidence retained at {stage} and {destination}"
            )
        try:
            _rename_noreplace(stage, destination)
            _fsync_dir(books)
            returned_identity = destination.stat(follow_symlinks=False)
            if not _same_inode(returned_identity, restored_identity):
                raise CandidatePublicationError(
                    "competing destination restoration identity differs"
                )
        except Exception as exc:
            raise CandidatePublicationError(
                "published destination rollback could not restore a competing "
                f"directory; conflict evidence retained at {stage} or {destination}"
            ) from exc
        raise CandidatePublicationError(
            "published destination rollback encountered a competing directory; "
            f"conflict evidence restored at {destination}"
        )
    try:
        restored_lock, restored_manifest = _authenticate_external_lock(
            stage,
            external_lock,
            expected_book_id=manifest["id"],
            repo_root=root,
        )
    except Exception as exc:
        raise CandidatePublicationError(
            f"published destination rollback could not authenticate restored stage; "
            f"conflict evidence retained at {stage}"
        ) from exc
    if (
        restored_lock != lock
        or restored_manifest != manifest
        or os.path.lexists(destination)
    ):
        raise CandidatePublicationError(
            f"published destination rollback differs after rename; conflict evidence "
            f"retained at {stage}"
        )


def _raise_guard_failure_after_rollback(
    failure: _CommitGuardFailure,
    rollback_errors: list[str],
) -> None:
    if rollback_errors:
        details = "; ".join(rollback_errors)
        raise CandidatePublicationError(
            "commit guard rejected publication and rollback could not be "
            f"authenticated; conflict evidence retained: {details}"
        ) from failure.error
    error = failure.error
    raise error.with_traceback(error.__traceback__)


def _commit_registration(
    state: dict[str, Any],
    *,
    commit_guard: Callable[[], None] | None = None,
    mutations: list[dict[str, Any]] | None = None,
) -> None:
    recorded_mutations = mutations if mutations is not None else []
    # Re-read before each write so a concurrent or partially recovered conflict
    # cannot be overwritten on the strength of an earlier check.
    expected_index = state["expected_index_entry"]
    current_index, _, current_books = _read_index(state["index_path"])
    if current_index != state["index_current"]:
        raise CandidatePublicationError("INDEX changed after registration preflight")
    current_index_matches = [row for row in current_books if row["id"] == expected_index["id"]]
    if current_index_matches and canonical_json_bytes(current_index_matches[0]) != canonical_json_bytes(expected_index):
        raise CandidatePublicationError("INDEX changed to a conflicting registration")
    if not state["index_missing"] and not current_index_matches:
        raise CandidatePublicationError("existing exact INDEX registration disappeared")

    if "registry_path" in state:
        current = state["registry_path"].read_bytes()
        if current != state["registry_current"]:
            raise CandidatePublicationError("registry changed after registration preflight")
        entry = state["registry_entry"]
        registry_exact = _registry_entry_present_exactly_once(
            current, entry, expected_index["id"].encode()
        )
        if not state["registry_missing"] and not registry_exact:
            raise CandidatePublicationError("existing exact registry registration disappeared")
    if "phase_path" in state:
        path = state["phase_path"]
        if path.exists():
            if path.is_symlink() or not path.is_file() \
                    or path.read_bytes() != state["phase_bytes"] \
                    or path.stat().st_mode & 0o222:
                raise CandidatePublicationError("phase seal changed to conflicting bytes")
        elif not state["phase_missing"]:
            raise CandidatePublicationError("existing exact phase seal disappeared")

    _run_commit_guard(commit_guard)

    if state["index_missing"]:
        current_index, raw, books = _read_index(state["index_path"])
        if current_index != state["index_current"]:
            raise CandidatePublicationError("INDEX changed before candidate registration")
        matches = [row for row in books if row["id"] == state["expected_index_entry"]["id"]]
        if matches:
            if canonical_json_bytes(matches[0]) != canonical_json_bytes(state["expected_index_entry"]):
                raise CandidatePublicationError("INDEX changed to a conflicting registration")
        else:
            raw["books"].append(state["expected_index_entry"])
            data = (json.dumps(raw, indent=1, ensure_ascii=True, allow_nan=False) + "\n").encode("utf-8")
            written_identity = _atomic_replace_bytes(
                state["index_path"], data, expected_current=current_index
            )
            recorded_mutations.append(
                {
                    "kind": "replace",
                    "path": state["index_path"],
                    "before": current_index,
                    "after": data,
                    "after_identity": written_identity,
                }
            )
            _run_commit_guard(commit_guard)

    if state.get("registry_missing"):
        path = state["registry_path"]
        current = path.read_bytes()
        if current != state["registry_current"]:
            raise CandidatePublicationError("registry changed before candidate registration")
        entry = state["registry_entry"]
        exact = _registry_entry_present_exactly_once(
            current, entry, state["expected_index_entry"]["id"].encode()
        )
        if not exact:
            separator = b"" if not current or current.endswith(b"\n") else b"\n"
            data = current + separator + entry + b"\n"
            written_identity = _atomic_replace_bytes(
                path,
                data,
                expected_current=current,
            )
            recorded_mutations.append(
                {
                    "kind": "replace",
                    "path": path,
                    "before": current,
                    "after": data,
                    "after_identity": written_identity,
                }
            )
            _run_commit_guard(commit_guard)

    if state.get("phase_missing"):
        path = state["phase_path"]
        if path.exists():
            if path.is_symlink() or not path.is_file() \
                    or path.read_bytes() != state["phase_bytes"] \
                    or path.stat().st_mode & 0o222:
                raise CandidatePublicationError("phase seal changed to conflicting bytes")
        else:
            written_identity = _atomic_write_new_read_only(
                path, state["phase_bytes"]
            )
            recorded_mutations.append(
                {
                    "kind": "create",
                    "path": path,
                    "after": state["phase_bytes"],
                    "after_identity": written_identity,
                }
            )
            _run_commit_guard(commit_guard)


def publish_candidate(
    stage_dir: str | os.PathLike[str],
    external_lock: dict[str, Any] | str | os.PathLike[str],
    *,
    repo_root: str | os.PathLike[str] = REPO_ROOT,
    registry_path: str | os.PathLike[str],
    registry_entry: str | bytes,
    phase_seal_path: str | os.PathLike[str],
    phase_seal: dict[str, Any],
    commit_guard: Callable[[], None] | None = None,
    after_rename: Callable[[Path], None] | None = None,
) -> Path:
    """Authenticate, atomically publish, and register a staged candidate.

    ``after_rename`` is a deterministic fault-injection hook for the required
    post-rename/pre-registration recovery test. ``commit_guard`` is checked
    inside the repository lock before publication/registration and after each
    externally visible mutation; a guard rejection rolls back owned changes.
    """
    if commit_guard is not None and after_rename is not None:
        raise CandidatePublicationError(
            "commit_guard cannot be combined with the re-entrant after_rename hook"
        )
    root, books = _validated_repo(repo_root)
    stage = Path(stage_dir)
    if not stage.is_absolute():
        stage = root / stage
    if stage.parent != books or stage.is_symlink() or not stage.is_dir():
        raise CandidatePublicationError(
            "candidate stage must be a real directory directly under exact repo books/"
        )
    lock, manifest = _authenticate_external_lock(stage, external_lock, repo_root=root)
    destination = root / lock["destination"]
    if destination.parent != books or destination.name != manifest["id"]:
        raise CandidatePublicationError("external lock destination is outside the exact repo books parent")
    if destination.exists() or destination.is_symlink():
        raise CandidatePublicationError(f"candidate destination already exists: {destination}")

    with _candidate_publication_lock(root):
        preflight = _registration_state(
            manifest,
            root=root,
            registry_path=registry_path,
            registry_entry=registry_entry,
            phase_seal_path=phase_seal_path,
            phase_seal=phase_seal,
        )
        if not preflight["index_missing"]:
            raise CandidatePublicationError(
                f"candidate INDEX registration pre-exists destination: {manifest['id']}"
            )
        if not preflight["registry_missing"]:
            raise CandidatePublicationError(
                f"candidate text registry registration pre-exists destination: {manifest['id']}"
            )
        if not preflight["phase_missing"]:
            raise CandidatePublicationError(
                f"candidate phase seal pre-exists destination: {manifest['id']}"
            )
        _fsync_tree(stage)
        if destination.exists() or destination.is_symlink():
            raise CandidatePublicationError(
                f"candidate destination appeared before rename: {destination}"
            )
        rechecked_lock, rechecked_manifest = _authenticate_external_lock(
            stage, external_lock, repo_root=root
        )
        if rechecked_lock != lock or rechecked_manifest != manifest:
            raise CandidatePublicationError(
                "candidate publication identity changed during pre-rename checks"
            )
        stage_identity = stage.stat(follow_symlinks=False)
        published_identity: os.stat_result | None = None
        registration_mutations: list[dict[str, Any]] = []
        try:
            _run_commit_guard(commit_guard)
            _rename_noreplace(stage, destination)
            _fsync_dir(books)
            published_identity = destination.stat(follow_symlinks=False)
            if not _same_inode(published_identity, stage_identity):
                raise CandidatePublicationError(
                    "published candidate directory identity differs after rename"
                )
            published_lock, published_manifest = _authenticate_external_lock(
                destination,
                external_lock,
                expected_book_id=manifest["id"],
                repo_root=root,
            )
            if published_lock != lock or published_manifest != manifest:
                raise CandidatePublicationError(
                    "published candidate identity differs from the authenticated stage"
                )
            _run_commit_guard(commit_guard)
            if after_rename is not None:
                after_rename(destination)
            final_lock, final_manifest = _authenticate_external_lock(
                destination,
                external_lock,
                expected_book_id=manifest["id"],
                repo_root=root,
            )
            if final_lock != lock or final_manifest != manifest:
                raise CandidatePublicationError(
                    "published candidate identity changed before registration"
                )
            # A same-thread fault hook may exercise recovery while the outer
            # transaction owns the re-entrant repository lock. Re-preflight so an
            # exact nested recovery is idempotent and any other nested registration
            # becomes part of the authenticated current state rather than stale
            # state that the outer call could overwrite.
            registration = _registration_state(
                final_manifest,
                root=root,
                registry_path=registry_path,
                registry_entry=registry_entry,
                phase_seal_path=phase_seal_path,
                phase_seal=phase_seal,
            )
            _commit_registration(
                registration,
                commit_guard=commit_guard,
                mutations=registration_mutations,
            )
        except _CommitGuardFailure as failure:
            rollback_errors = _rollback_registration_mutations(
                registration_mutations
            )
            if published_identity is not None and not rollback_errors:
                try:
                    _require_publish_preflight_restored(preflight)
                    _restore_published_stage(
                        stage=stage,
                        destination=destination,
                        expected_identity=published_identity,
                        external_lock=external_lock,
                        lock=lock,
                        manifest=manifest,
                        root=root,
                        books=books,
                    )
                except Exception as exc:
                    rollback_errors.append(str(exc))
            _raise_guard_failure_after_rollback(failure, rollback_errors)
    return destination


def recover_candidate_publication(
    book_id: str,
    external_lock: dict[str, Any] | str | os.PathLike[str],
    *,
    repo_root: str | os.PathLike[str] = REPO_ROOT,
    registry_path: str | os.PathLike[str],
    registry_entry: str | bytes,
    phase_seal_path: str | os.PathLike[str],
    phase_seal: dict[str, Any],
    commit_guard: Callable[[], None] | None = None,
) -> Path:
    """Finish only an externally authenticated post-rename publication.

    A commit-guard rejection rolls back registration writes made by this call;
    the preexisting authenticated destination is never moved or removed.
    """
    root, books = _validated_repo(repo_root)
    bid = _validate_book_id(book_id)
    destination = books / bid
    if destination.is_symlink() or not destination.is_dir():
        raise CandidatePublicationError(f"published candidate destination is missing: {destination}")
    with _candidate_publication_lock(root):
        lock, manifest = _authenticate_external_lock(
            destination, external_lock, expected_book_id=bid, repo_root=root
        )
        if lock["destination"] != f"books/{bid}":
            raise CandidatePublicationError("external recovery destination mismatch")
        registration = _registration_state(
            manifest,
            root=root,
            registry_path=registry_path,
            registry_entry=registry_entry,
            phase_seal_path=phase_seal_path,
            phase_seal=phase_seal,
        )
        registration_mutations: list[dict[str, Any]] = []
        try:
            _commit_registration(
                registration,
                commit_guard=commit_guard,
                mutations=registration_mutations,
            )
        except _CommitGuardFailure as failure:
            rollback_errors = _rollback_registration_mutations(
                registration_mutations
            )
            _raise_guard_failure_after_rollback(failure, rollback_errors)
    return destination
