"""Bounded Demo-only rolling-refit SOTA policy loader, hash-chained transition journal, and atomic vintage handle.

Sibling to scripts/sota_book_runtime.py and scripts/sota_book_freeze.py (both kept byte-identical
for PHASE1_SOURCE_CLOSURE_FILES seals). Governs Owner Decision 15 (2026-11-26):
- Reads tracked rolling policies from books/SOTA_ROLLING_INDEX.json (never books/SOTA_INDEX.json)
  with strict Demo-only enforcement (`demo_rolling_refit_canary`, live CR* unconditionally refused).
- Authenticates hash-chained append-only transitions.jsonl (committed & fsynced BEFORE state.json)
  plus durable SQLite sota_risk_state watermarks (`rolling_watermark:{book_id}:{policy_content_id}`)
  preventing vintage replay, equal-seq substitution, and wall-clock rollback across restarts.
- Serializes refit commits (`LOCK_EX`) against rolling transport handoffs (`LOCK_SH | LOCK_NB`) via
  a dedicated per-book `.transition.flock` (zero contention on static-book sends).
- Loads vintage artifacts strictly once into memory via component-by-component O_NOFOLLOW, hashing
  and parsing the exact same bytes (zero TOCTOU), and immediately drops in-memory weights on
  abstention, revocation, expiry, rollback, or corruption (never falling back to an older vintage).
"""

from __future__ import annotations

import errno
import fcntl
import hashlib
import io
import json
import math
import os
import pwd
import re
import sqlite3
import stat
import time
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from deriv_runtime_core_v2 import SUPPORTED_CONTRACT_FAMILIES, SUPPORTED_HORIZONS_MINUTES
from sota_book_freeze import atomic_write_file_bytes, fsync_dir
from sota_book_runtime import (
    REPO_ROOT,
    SOTA_MANIFEST_SCHEMA,
    LoadedSotaBook,
    SotaBookError,
    _check_book_id,
    _validate_weights_npz,
    compute_sota_content_id_from_buffers,
    load_enabled_sota_books,
    load_sota_book,
    load_sota_index,
)

SOTA_ROLLING_INDEX_SCHEMA = "sota-rolling-book-index/v1"
SOTA_ROLLING_POLICY_SCHEMA = "sota-rolling-policy/v1"
SOTA_ROLLING_VINTAGE_MANIFEST_SCHEMA = "sota-rolling-vintage-manifest/v1"
SOTA_ROLLING_TRANSITION_SCHEMA = "sota-rolling-transition/v1"
ROLLING_BOOK_MODE = "rolling_refit"
DEMO_ROLLING_LIFECYCLE = "demo_rolling_refit_canary"
REQUIRED_ROLLING_DEV_GATES: tuple[str, ...] = (
    "G1_rolling_oos",
    "G2_placebo",
    "G3_entry_delay",
    "G4_sign_null",
    "G5_payout",
)
VALID_TRANSITION_STATUSES = frozenset({"active", "abstained", "revoked"})
DEFAULT_ROLLING_VINTAGE_DIR_REL = Path("deriv_data/runtime/rolling_vintages")
MAX_CLOCK_ROLLBACK_TOLERANCE_S = 60

_HEX16_RE = re.compile(r"^[0-9a-f]{16}$")
_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
_VINTAGE_ID_RE = re.compile(r"^[0-9]{8}T[0-9]{6}Z_[0-9a-f]{16}$")


class RollingVintageError(SotaBookError):
    pass


class RollingTransitionLockBusyError(RollingVintageError):
    pass


class _SymlinkOrPathError(RollingVintageError):
    pass


class _MissingFileError(FileNotFoundError):
    pass


def _strict_int(val: Any) -> int | None:
    return val if type(val) is int else None


def _strict_float(val: Any) -> float | None:
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        return None
    f = float(val)
    return f if math.isfinite(f) else None


def _utc_iso_whole(epoch: int) -> str:
    return datetime.fromtimestamp(int(epoch), tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _canonical_json_bytes(obj: dict[str, Any]) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _strict_json_loads(raw: bytes) -> Any:
    def _no_dups(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for k, v in pairs:
            if k in out:
                raise RollingVintageError(f"duplicate JSON key rejected: {k!r}")
            out[k] = v
        return out

    def _no_const(tok: str) -> Any:
        raise RollingVintageError(f"non-finite JSON constant rejected: {tok!r}")

    try:
        return json.loads(raw.decode("utf-8"), object_pairs_hook=_no_dups, parse_constant=_no_const)
    except RollingVintageError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError, RecursionError) as exc:
        raise RollingVintageError(f"malformed rolling JSON: {exc}") from exc


def _open_dir_nofollow(dir_path: Path, *, trusted_root: Path | None = None) -> tuple[int, Path]:
    """Open dir_path strictly with component-by-component O_DIRECTORY | O_NOFOLLOW."""
    p_raw = Path(dir_path)
    if ".." in p_raw.parts:
        raise _SymlinkOrPathError(f"dotdot_path_refused: {p_raw}")
    p_abs = p_raw if p_raw.is_absolute() else (Path.cwd() / p_raw)
    if ".." in p_abs.parts:
        raise _SymlinkOrPathError(f"dotdot_path_refused: {p_abs}")

    anchor = Path("/")
    rel_parts = list(p_abs.parts[1:])
    if trusted_root is not None:
        tr_raw = Path(trusted_root)
        tr_abs = tr_raw if tr_raw.is_absolute() else (Path.cwd() / tr_raw)
        try:
            tr_res = tr_abs.resolve(strict=True)
        except OSError:
            tr_res = None
        if tr_res is not None:
            for cand in (tr_abs, tr_res):
                try:
                    rel = p_abs.relative_to(cand)
                    anchor = tr_res
                    rel_parts = list(rel.parts)
                    break
                except ValueError:
                    continue

    try:
        cur_fd = os.open(str(anchor), os.O_RDONLY | os.O_DIRECTORY)
    except OSError as exc:
        if exc.errno == errno.ENOENT:
            raise _MissingFileError(str(anchor)) from exc
        raise _SymlinkOrPathError(f"cannot open anchor {anchor}: {exc}") from exc

    try:
        for comp in rel_parts:
            if comp in ("", "."):
                continue
            if comp == "..":
                raise _SymlinkOrPathError("dotdot_component_refused")
            try:
                nxt_fd = os.open(comp, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=cur_fd)
            except OSError as exc:
                if exc.errno == errno.ENOENT:
                    raise _MissingFileError(comp) from exc
                if exc.errno in (errno.ELOOP, errno.ENOTDIR):
                    raise _SymlinkOrPathError(f"symlink_or_nondir_refused:{comp}") from exc
                raise _SymlinkOrPathError(f"open_dir_failed:{comp}:{exc}") from exc
            os.close(cur_fd)
            cur_fd = nxt_fd
        return cur_fd, anchor.joinpath(*rel_parts)
    except Exception:
        os.close(cur_fd)
        raise


def _ensure_dir_nofollow(dir_path: Path, *, mode: int = 0o700, trusted_root: Path | None = None) -> Path:
    """Ensure dir_path exists strictly without traversing or creating symlinks."""
    p_raw = Path(dir_path)
    if ".." in p_raw.parts:
        raise _SymlinkOrPathError(f"dotdot_path_refused: {p_raw}")
    p_abs = p_raw if p_raw.is_absolute() else (Path.cwd() / p_raw)
    if ".." in p_abs.parts:
        raise _SymlinkOrPathError(f"dotdot_path_refused: {p_abs}")

    anchor = Path("/")
    rel_parts = list(p_abs.parts[1:])
    if trusted_root is not None:
        tr_raw = Path(trusted_root)
        tr_abs = tr_raw if tr_raw.is_absolute() else (Path.cwd() / tr_raw)
        try:
            tr_res = tr_abs.resolve(strict=True)
        except OSError:
            tr_res = None
        if tr_res is not None:
            for cand in (tr_abs, tr_res):
                try:
                    rel = p_abs.relative_to(cand)
                    anchor = tr_res
                    rel_parts = list(rel.parts)
                    break
                except ValueError:
                    continue

    try:
        cur_fd = os.open(str(anchor), os.O_RDONLY | os.O_DIRECTORY)
    except OSError as exc:
        raise _SymlinkOrPathError(f"cannot open anchor {anchor}: {exc}") from exc
    try:
        for comp in rel_parts:
            if comp in ("", "."):
                continue
            if comp == "..":
                raise _SymlinkOrPathError("dotdot_component_refused")
            try:
                nxt_fd = os.open(comp, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=cur_fd)
            except OSError as exc:
                if exc.errno == errno.ENOENT:
                    try:
                        os.mkdir(comp, mode=mode, dir_fd=cur_fd)
                        os.fsync(cur_fd)
                        nxt_fd = os.open(comp, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=cur_fd)
                    except OSError as mk_exc:
                        raise _SymlinkOrPathError(f"mkdir_failed:{comp}:{mk_exc}") from mk_exc
                elif exc.errno in (errno.ELOOP, errno.ENOTDIR):
                    raise _SymlinkOrPathError(f"symlink_or_nondir_refused:{comp}") from exc
                else:
                    raise _SymlinkOrPathError(f"ensure_dir_failed:{comp}:{exc}") from exc
            os.close(cur_fd)
            cur_fd = nxt_fd
        return anchor.joinpath(*rel_parts)
    finally:
        os.close(cur_fd)


def _read_regular_file_bytes_nofollow(parent_fd: int, filename: str) -> tuple[bytes, os.stat_result]:
    if not filename or Path(filename).name != filename or filename in (".", ".."):
        raise _SymlinkOrPathError(f"invalid_filename:{filename!r}")
    try:
        fd = os.open(filename, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent_fd)
    except OSError as exc:
        if exc.errno == errno.ENOENT:
            raise _MissingFileError(filename) from exc
        if exc.errno == errno.ELOOP:
            raise _SymlinkOrPathError(f"symlink_file_refused:{filename}") from exc
        raise _SymlinkOrPathError(f"open_file_failed:{filename}:{exc}") from exc
    try:
        try:
            st = os.fstat(fd)
            if not stat.S_ISREG(st.st_mode):
                raise _SymlinkOrPathError(f"not_regular_file:{filename}")
            chunks: list[bytes] = []
            while True:
                chunk = os.read(fd, 1 << 20)
                if not chunk:
                    break
                chunks.append(chunk)
            return b"".join(chunks), st
        except RollingVintageError:
            raise
        except OSError as exc:
            raise _SymlinkOrPathError(f"read_file_failed:{filename}:{exc}") from exc
    finally:
        os.close(fd)


def _atomic_write_bytes_dirfd(dir_fd: int, filename: str, data: bytes, *, mode: int = 0o600) -> None:
    """Atomically write filename inside dir_fd (O_CREAT|O_EXCL|O_NOFOLLOW temp -> fsync -> os.replace -> fsync dir)."""
    if not filename or Path(filename).name != filename or filename in (".", ".."):
        raise _SymlinkOrPathError(f"invalid_filename:{filename!r}")
    tmp_name = f".{filename}.tmp.{os.getpid()}.{time.time_ns()}"
    fd = -1
    try:
        fd = os.open(tmp_name, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, mode, dir_fd=dir_fd)
        os.fchmod(fd, mode)
        view = memoryview(data)
        written = 0
        while written < len(view):
            written += os.write(fd, view[written:])
        os.fsync(fd)
        os.close(fd)
        fd = -1
        os.replace(tmp_name, filename, src_dir_fd=dir_fd, dst_dir_fd=dir_fd)
        os.fsync(dir_fd)
    except OSError as exc:
        if fd >= 0:
            try:
                os.close(fd)
            except OSError:
                pass
        try:
            os.unlink(tmp_name, dir_fd=dir_fd)
        except OSError:
            pass
        raise RollingVintageError(f"atomic_write_failed:{filename}:{exc}") from exc


def canonical_rolling_vintage_root(*, repo_root: Path = REPO_ROOT) -> Path:
    return Path(repo_root) / DEFAULT_ROLLING_VINTAGE_DIR_REL


def canonical_vps_connection_paths(*, pw_home: Path | None = None) -> tuple[Path, Path]:
    """Return canonical host-wide VPS public connection lock & admission state paths (no CLI override)."""
    home = Path(pw_home) if pw_home is not None else Path(pwd.getpwuid(os.getuid()).pw_dir).resolve(strict=True)
    state_dir = home / ".local" / "state" / "binary-algo"
    return state_dir / "deriv_public_connection.lock", state_dir / "deriv_public_connection.admission.json"


@contextmanager
def acquire_rolling_transition_lock(
    book_vintage_dir: Path,
    *,
    exclusive: bool = True,
    nonblocking: bool = True,
    trusted_root: Path | None = None,
) -> Iterator[int]:
    """Acquire per-book `.transition.flock` (0600, O_NOFOLLOW) verifying inode stability and yielding bound dir_fd."""
    try:
        resolved_dir = _ensure_dir_nofollow(book_vintage_dir, mode=0o700, trusted_root=trusted_root)
        dir_fd, _ = _open_dir_nofollow(resolved_dir, trusted_root=trusted_root)
    except RollingVintageError:
        raise
    except OSError as exc:
        raise RollingVintageError(f"rolling_transition_lock_dir_error:{exc}") from exc
    fd = -1
    try:
        try:
            fd = os.open(".transition.flock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600, dir_fd=dir_fd)
            st_open = os.fstat(fd)
            if not stat.S_ISREG(st_open.st_mode):
                raise RollingVintageError(f"rolling_transition_lock_not_regular:{resolved_dir}")
            os.fchmod(fd, 0o600)
            os.set_inheritable(fd, False)
            flags = (fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH) | (fcntl.LOCK_NB if nonblocking else 0)
            try:
                fcntl.flock(fd, flags)
            except BlockingIOError as exc:
                raise RollingTransitionLockBusyError(f"rolling_transition_lock_busy:{resolved_dir}") from exc
            # Verify the locked inode is STILL the live directory entry (defeats unlink+replace inode split!)
            st_live = os.stat(".transition.flock", dir_fd=dir_fd, follow_symlinks=False)
            if (int(st_open.st_dev), int(st_open.st_ino)) != (int(st_live.st_dev), int(st_live.st_ino)):
                raise RollingVintageError(f"rolling_transition_lock_inode_replaced:{resolved_dir}")
        except RollingVintageError:
            raise
        except OSError as exc:
            raise RollingVintageError(f"rolling_transition_lock_error:{exc}") from exc
        yield dir_fd
    finally:
        if fd >= 0:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)
        os.close(dir_fd)


# ---------------------------------------------------------------------------
# Tracked Policy Index & Manifest Verification (books/SOTA_ROLLING_INDEX.json)
# ---------------------------------------------------------------------------


def load_rolling_index(repo_root: Path = REPO_ROOT) -> dict[str, dict[str, Any]]:
    idx_path = Path(repo_root) / "books" / "SOTA_ROLLING_INDEX.json"
    if not idx_path.exists():
        return {}
    books_fd, _ = _open_dir_nofollow(Path(repo_root) / "books", trusted_root=repo_root)
    try:
        raw_bytes, _ = _read_regular_file_bytes_nofollow(books_fd, "SOTA_ROLLING_INDEX.json")
    finally:
        os.close(books_fd)
    raw = _strict_json_loads(raw_bytes)
    if not isinstance(raw, dict) or raw.get("schema") != SOTA_ROLLING_INDEX_SCHEMA or not isinstance(raw.get("books"), list):
        raise RollingVintageError(f"malformed SOTA_ROLLING_INDEX.json at {idx_path}")
    out: dict[str, dict[str, Any]] = {}
    for ent in raw["books"]:
        if not isinstance(ent, dict):
            raise RollingVintageError("non-object entry in SOTA_ROLLING_INDEX.json")
        bid = _check_book_id(str(ent.get("id") or ""))
        if bid in out:
            raise RollingVintageError(f"duplicate rolling book_id {bid}")
        if ent.get("book_mode") != ROLLING_BOOK_MODE:
            raise RollingVintageError(f"{bid}: invalid book_mode {ent.get('book_mode')!r} in SOTA_ROLLING_INDEX.json")
        out[bid] = ent
    return out


def _validate_rolling_gates(book_id: str, lifecycle: str, manifest: dict[str, Any], allow_demo_canary: bool) -> None:
    if not allow_demo_canary:
        raise RollingVintageError(
            f"{book_id}: rolling-refit books are strictly Demo-only ({DEMO_ROLLING_LIFECYCLE}); live CR* refused"
        )
    if lifecycle != DEMO_ROLLING_LIFECYCLE:
        raise RollingVintageError(f"{book_id}: invalid rolling lifecycle_status {lifecycle!r}")
    if manifest.get("book_mode") != ROLLING_BOOK_MODE:
        raise RollingVintageError(f"{book_id}: manifest book_mode must be {ROLLING_BOOK_MODE!r}")
    amend = manifest.get("protocol_amendment")
    if type(amend) is not str or not amend.strip():
        raise RollingVintageError(f"{book_id}: rolling-refit manifest requires non-empty protocol_amendment")
    certs = manifest.get("gate_certificates")
    if not isinstance(certs, dict):
        raise RollingVintageError(f"{book_id}: missing gate_certificates dict")
    if "G1_cpcv" in certs:
        raise RollingVintageError(f"{book_id}: rolling-refit manifest must use G1_rolling_oos, never spoof G1_cpcv")
    for g in REQUIRED_ROLLING_DEV_GATES:
        g_obj = certs.get(g)
        if not isinstance(g_obj, dict) or g_obj.get("passed") is not True:
            raise RollingVintageError(f"{book_id}: mandatory rolling DEV gate {g} is missing or not passed")
    g6 = certs.get("G6_holdout_or_prospective")
    if not isinstance(g6, dict) or g6.get("live_real_money_eligible") is not False:
        raise RollingVintageError(f"{book_id}: rolling G6 must explicitly set live_real_money_eligible=false")


@dataclass(frozen=True)
class RollingSotaPolicy:
    book_id: str
    pair: str
    horizon_minutes: int
    allowed_sides: tuple[str, ...]
    lifecycle_status: str
    model_kind: str
    contract_family: str
    feature_cols: tuple[str, ...]
    coverage: float
    train_days: int
    warmup_days: int
    refit_cadence_days: int
    max_vintage_age_days: int
    cadence_origin_utc_epoch: int
    block_val_gap_s: int
    settle_tol_s: int
    require_positive_platt_slope: bool
    drop_ny_16h: bool
    min_payout_ratio: float
    payout_edge_margin: float
    rolling_oos_p10_floors: dict[str, float]
    content_id: str  # immutable 16-hex policy content_id
    protocol_amendment: str
    manifest: dict[str, Any]
    policy_dict: dict[str, Any]


def load_rolling_policy(
    book_id: str,
    *,
    repo_root: Path = REPO_ROOT,
    allow_demo_canary: bool = True,
) -> RollingSotaPolicy:
    _check_book_id(book_id)
    idx = load_rolling_index(repo_root=repo_root)
    if book_id not in idx:
        raise RollingVintageError(f"{book_id}: not registered in books/SOTA_ROLLING_INDEX.json")
    entry = idx[book_id]
    book_dir = Path(repo_root) / "books" / book_id
    b_fd, _ = _open_dir_nofollow(book_dir, trusted_root=repo_root)
    try:
        man_bytes, _ = _read_regular_file_bytes_nofollow(b_fd, f"{book_id}.manifest.json")
        manifest = _strict_json_loads(man_bytes)
        if not isinstance(manifest, dict) or manifest.get("schema") != SOTA_MANIFEST_SCHEMA:
            raise RollingVintageError(f"{book_id}: invalid manifest schema")
        if manifest.get("id") != book_id:
            raise RollingVintageError(f"{book_id}: manifest id mismatch")
        lifecycle = str(entry.get("lifecycle_status") or "")
        if lifecycle != manifest.get("lifecycle_status"):
            raise RollingVintageError(f"{book_id}: lifecycle mismatch between SOTA_ROLLING_INDEX.json and manifest")
        _validate_rolling_gates(book_id, lifecycle, manifest, allow_demo_canary=allow_demo_canary)
        if manifest.get("settlement_substrate") != "deriv_1s_ticks":
            raise RollingVintageError(f"{book_id}: settlement_substrate must be 'deriv_1s_ticks'")

        artifacts = manifest.get("artifacts")
        if not isinstance(artifacts, list) or len(artifacts) != 1:
            raise RollingVintageError(f"{book_id}: rolling-refit manifest must contain exact 1 tracked policy artifact")
        art = artifacts[0]
        pol_fname = str(art.get("file") or "")
        if pol_fname != f"{book_id}_policy.json":
            raise RollingVintageError(f"{book_id}: unexpected rolling policy filename {pol_fname!r}")
        pol_bytes, _ = _read_regular_file_bytes_nofollow(b_fd, pol_fname)
    finally:
        os.close(b_fd)

    if _sha256_bytes(pol_bytes) != art.get("sha256") or len(pol_bytes) != art.get("bytes"):
        raise RollingVintageError(f"{book_id}: policy artifact failed SHA-256 or byte-length verification")
    cid = compute_sota_content_id_from_buffers([(pol_fname, pol_bytes)])
    if cid != manifest.get("content_id") or cid != entry.get("content_id"):
        raise RollingVintageError(f"{book_id}: policy content_id mismatch ({cid})")

    pol = _strict_json_loads(pol_bytes)
    if not isinstance(pol, dict) or pol.get("schema") != SOTA_ROLLING_POLICY_SCHEMA:
        raise RollingVintageError(f"{book_id}: invalid rolling policy schema")
    if pol.get("book_id") != book_id or pol.get("book_mode") != ROLLING_BOOK_MODE:
        raise RollingVintageError(f"{book_id}: policy book_id/book_mode mismatch")
    if "refit_cpcv_p10_floors" in pol:
        raise RollingVintageError(f"{book_id}: rolling policy must use rolling_oos_p10_floors, not refit_cpcv_p10_floors")

    pair = str(manifest["pair"]).upper()
    if str(pol.get("pair") or "").upper() != pair:
        raise RollingVintageError(f"{book_id}: policy pair mismatch")
    hm = _strict_int(manifest.get("horizon_minutes"))
    if hm is None or hm not in SUPPORTED_HORIZONS_MINUTES or _strict_int(pol.get("horizon_minutes")) != hm:
        raise RollingVintageError(f"{book_id}: unsupported or mismatched horizon_minutes")
    family = str(pol.get("contract_family") or "")
    if family not in SUPPORTED_CONTRACT_FAMILIES or family != "risefall":
        raise RollingVintageError(f"{book_id}: rolling policy currently supports risefall only, got {family!r}")
    model_kind = str(manifest.get("model_kind") or "")
    if model_kind != "nystroem-ens" or pol.get("model_kind") != model_kind:
        raise RollingVintageError(f"{book_id}: rolling policy supports nystroem-ens only, got {model_kind!r}")
    allowed_sides = tuple(str(s).upper() for s in (pol.get("allowed_sides") or []))
    if allowed_sides != ("UP",) or tuple(str(s).upper() for s in (manifest.get("allowed_sides") or [])) != allowed_sides:
        raise RollingVintageError(f"{book_id}: rolling policy requires allowed_sides=['UP']")
    fcols = tuple(str(c) for c in (pol.get("feature_cols") or []))
    if len(fcols) != 66 or len(set(fcols)) != 66:
        raise RollingVintageError(f"{book_id}: rolling policy requires exact 66 unique core66 feature_cols")

    floors_raw = pol.get("rolling_oos_p10_floors")
    if not isinstance(floors_raw, dict) or "UP" not in floors_raw:
        raise RollingVintageError(f"{book_id}: missing rolling_oos_p10_floors['UP']")
    up_fl = _strict_float(floors_raw["UP"])
    if up_fl is None or not (0.541 <= up_fl < 1.0):
        raise RollingVintageError(f"{book_id}: invalid rolling_oos_p10_floors['UP']={floors_raw['UP']!r}")

    cov = _strict_float(pol.get("coverage"))
    min_pay = _strict_float(pol.get("min_payout_ratio"))
    ev_margin = _strict_float(pol.get("payout_edge_margin"))
    train_days = _strict_int(pol.get("train_days"))
    warmup_days = _strict_int(pol.get("warmup_days"))
    cadence_days = _strict_int(pol.get("refit_cadence_days"))
    max_age_days = _strict_int(pol.get("max_vintage_age_days"))
    origin_ep = _strict_int(pol.get("cadence_origin_utc_epoch"))
    val_gap = _strict_int(pol.get("block_val_gap_s"))
    tol_s = _strict_int(pol.get("settle_tol_s"))
    if (
        cov is None or not (0.0 < cov <= 0.25)
        or min_pay is None or min_pay < 1.70
        or ev_margin is None or ev_margin < 0.005
        or train_days is None or train_days < 60
        or warmup_days is None or warmup_days < 5
        or cadence_days is None or cadence_days < 1
        or max_age_days is None or max_age_days != cadence_days
        or origin_ep is None or origin_ep <= 0
        or val_gap is None or val_gap != hm * 60
        or tol_s is None or tol_s <= 0
        or pol.get("require_positive_platt_slope") is not True
    ):
        raise RollingVintageError(f"{book_id}: invalid rolling policy numeric bounds or invariants")

    return RollingSotaPolicy(
        book_id=book_id,
        pair=pair,
        horizon_minutes=hm,
        allowed_sides=allowed_sides,
        lifecycle_status=lifecycle,
        model_kind=model_kind,
        contract_family=family,
        feature_cols=fcols,
        coverage=cov,
        train_days=train_days,
        warmup_days=warmup_days,
        refit_cadence_days=cadence_days,
        max_vintage_age_days=max_age_days,
        cadence_origin_utc_epoch=origin_ep,
        block_val_gap_s=val_gap,
        settle_tol_s=tol_s,
        require_positive_platt_slope=True,
        drop_ny_16h=bool(pol.get("drop_ny_16h", False)),
        min_payout_ratio=min_pay,
        payout_edge_margin=ev_margin,
        rolling_oos_p10_floors={"UP": up_fl},
        content_id=cid,
        protocol_amendment=str(manifest["protocol_amendment"]),
        manifest=manifest,
        policy_dict=pol,
    )


# ---------------------------------------------------------------------------
# Authoritative Hash-Chained Transition Journal & Atomic State Mirror
# ---------------------------------------------------------------------------


def validate_transition_record(
    rec: Any,
    *,
    expected_book_id: str,
    expected_policy_cid: str,
    expected_seq: int,
    expected_prev_sha: str,
    min_published_epoch: int,
    max_vintage_age_days: int,
) -> dict[str, Any]:
    if not isinstance(rec, dict):
        raise RollingVintageError("transition record must be a JSON object")
    if rec.get("schema") != SOTA_ROLLING_TRANSITION_SCHEMA:
        raise RollingVintageError(f"invalid transition schema {rec.get('schema')!r}")
    if rec.get("book_id") != expected_book_id or rec.get("policy_content_id") != expected_policy_cid:
        raise RollingVintageError("transition book_id or policy_content_id mismatch")
    seq = _strict_int(rec.get("seq"))
    if seq != expected_seq:
        raise RollingVintageError(f"transition seq mismatch: got {seq!r}, expected {expected_seq}")
    if rec.get("prev_transition_sha256") != expected_prev_sha:
        raise RollingVintageError("transition prev_transition_sha256 chain break")
    t_sha = rec.get("transition_sha256")
    if type(t_sha) is not str or not _HEX64_RE.fullmatch(t_sha):
        raise RollingVintageError("invalid transition_sha256 format")
    body = {k: v for k, v in rec.items() if k != "transition_sha256"}
    if _sha256_bytes(_canonical_json_bytes(body)) != t_sha:
        raise RollingVintageError(f"transition_sha256 digest mismatch at seq={seq}")

    pub_ep = _strict_int(rec.get("published_epoch"))
    sched_ep = _strict_int(rec.get("scheduled_refit_epoch"))
    next_ep = _strict_int(rec.get("next_refit_epoch"))
    if pub_ep is None or pub_ep <= 0 or pub_ep < min_published_epoch:
        raise RollingVintageError(f"non-monotonic or invalid published_epoch at seq={seq}")
    if rec.get("published_utc") != _utc_iso_whole(pub_ep):
        raise RollingVintageError(f"published_utc mismatch at seq={seq}")
    if sched_ep is None or sched_ep <= 0 or next_ep is None or next_ep <= sched_ep:
        raise RollingVintageError(f"invalid scheduled/next refit epochs at seq={seq}")
    max_span_s = int(max_vintage_age_days) * 86_400
    if (next_ep - sched_ep) > max_span_s:
        raise RollingVintageError(f"next_refit_epoch exceeds max_vintage_age_days ceiling at seq={seq}")

    status = rec.get("status")
    if type(status) is not str or status not in VALID_TRANSITION_STATUSES:
        raise RollingVintageError(f"invalid transition status {status!r} at seq={seq}")

    if status == "active":
        vid = rec.get("vintage_id")
        vcid = rec.get("vintage_content_id")
        vmsha = rec.get("vintage_manifest_sha256")
        vf = _strict_int(rec.get("valid_from_epoch"))
        vu = _strict_int(rec.get("valid_until_epoch"))
        p_coef = _strict_float(rec.get("platt_coef"))
        p_int = _strict_float(rec.get("platt_intercept"))
        if (
            type(vid) is not str or not _VINTAGE_ID_RE.fullmatch(vid)
            or type(vcid) is not str or not _HEX16_RE.fullmatch(vcid)
            or not vid.endswith(f"_{vcid}")
            or type(vmsha) is not str or not _HEX64_RE.fullmatch(vmsha)
        ):
            raise RollingVintageError(f"invalid active vintage identity at seq={seq}")
        if vf is None or vu is None or vf < pub_ep or vf < sched_ep or vu <= vf or (vu - sched_ep) > max_span_s:
            raise RollingVintageError(f"invalid active validity window at seq={seq}")
        if rec.get("valid_from_utc") != _utc_iso_whole(vf) or rec.get("valid_until_utc") != _utc_iso_whole(vu):
            raise RollingVintageError(f"valid_from_utc/valid_until_utc mismatch at seq={seq}")
        if p_coef is None or p_coef <= 0.0 or p_int is None:
            raise RollingVintageError(f"active transition requires positive platt_coef at seq={seq}")
    else:
        if status == "abstained" and pub_ep < sched_ep:
            raise RollingVintageError(f"abstained transition published_epoch precedes scheduled_refit_epoch at seq={seq}")
        if any(rec.get(k) is not None for k in ("vintage_id", "vintage_content_id", "vintage_manifest_sha256", "valid_from_epoch", "valid_until_epoch")):
            raise RollingVintageError(f"{status} transition must clear vintage fields at seq={seq}")
        if type(rec.get("reason")) is not str or not str(rec["reason"]).strip():
            raise RollingVintageError(f"{status} transition missing non-empty reason at seq={seq}")
    return rec


def load_verified_transitions(
    book_vintage_dir: Path,
    *,
    expected_book_id: str,
    expected_policy_cid: str,
    max_vintage_age_days: int,
    trusted_root: Path | None = None,
    dir_fd: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """Read and hash-chain-verify transitions.jsonl, writer_watermark.json & state.json via single-read O_NOFOLLOW."""
    owns_fd = dir_fd is None
    if owns_fd:
        try:
            active_fd, _ = _open_dir_nofollow(book_vintage_dir, trusted_root=trusted_root)
        except _MissingFileError:
            return [], None
    else:
        assert dir_fd is not None
        active_fd = dir_fd
    try:
        try:
            j_bytes, _ = _read_regular_file_bytes_nofollow(active_fd, "transitions.jsonl")
            has_journal = True
        except _MissingFileError:
            j_bytes = b""
            has_journal = False
        try:
            s_bytes, _ = _read_regular_file_bytes_nofollow(active_fd, "state.json")
            has_state = True
        except _MissingFileError:
            s_bytes = b""
            has_state = False
        try:
            w_bytes, _ = _read_regular_file_bytes_nofollow(active_fd, "writer_watermark.json")
            has_writer_wm = True
        except _MissingFileError:
            w_bytes = b""
            has_writer_wm = False
    finally:
        if owns_fd:
            os.close(active_fd)

    if not has_journal and not has_state and not has_writer_wm:
        return [], None
    if not has_journal and (has_state or has_writer_wm):
        raise RollingVintageError("state.json or writer_watermark.json exists without authoritative transitions.jsonl")

    lines = [ln for ln in j_bytes.splitlines() if ln.strip()]
    if not lines:
        if has_state or has_writer_wm:
            raise RollingVintageError("empty transitions.jsonl with present state.json/writer_watermark.json")
        return [], None

    transitions: list[dict[str, Any]] = []
    prev_sha = "GENESIS"
    min_pub = 1
    for idx, raw_ln in enumerate(lines, start=1):
        rec = validate_transition_record(
            _strict_json_loads(raw_ln),
            expected_book_id=expected_book_id,
            expected_policy_cid=expected_policy_cid,
            expected_seq=idx,
            expected_prev_sha=prev_sha,
            min_published_epoch=min_pub,
            max_vintage_age_days=max_vintage_age_days,
        )
        transitions.append(rec)
        prev_sha = str(rec["transition_sha256"])
        min_pub = int(rec["published_epoch"])

    if has_writer_wm:
        wm_w = _strict_json_loads(w_bytes)
        w_seq = _strict_int(wm_w.get("max_seq")) if isinstance(wm_w, dict) else None
        w_sha = wm_w.get("last_transition_sha256") if isinstance(wm_w, dict) else None
        if (
            not isinstance(wm_w, dict)
            or wm_w.get("book_id") != expected_book_id
            or wm_w.get("policy_content_id") != expected_policy_cid
            or w_seq is None
            or w_seq < 1
            or type(w_sha) is not str
            or not _HEX64_RE.fullmatch(w_sha)
        ):
            raise RollingVintageError("corrupt writer_watermark.json")
        if len(transitions) < w_seq:
            raise RollingVintageError("rolling_writer_watermark_seq_rollback")
        if transitions[w_seq - 1]["transition_sha256"] != w_sha:
            raise RollingVintageError("rolling_writer_watermark_chain_fork")

    state_obj: dict[str, Any] | None = None
    if has_state:
        st_parsed = _strict_json_loads(s_bytes)
        if not isinstance(st_parsed, dict):
            raise RollingVintageError("state.json is not a JSON object")
        state_obj = st_parsed
    return transitions, state_obj


def compute_effective_vintage_intervals(
    transitions: list[dict[str, Any]],
) -> dict[tuple[int, str], dict[str, Any]]:
    """Compute exact live authorization intervals [eff_start_epoch, eff_end_epoch) for each active transition.

    When a later transition supersedes or revokes an active vintage at T_{i+1}.published_epoch < valid_until_epoch,
    eff_end_epoch is strictly capped at T_{i+1}.published_epoch.
    """
    out: dict[tuple[int, str], dict[str, Any]] = {}
    n = len(transitions)
    for i, rec in enumerate(transitions):
        if rec.get("status") != "active":
            continue
        eff_start = int(rec["valid_from_epoch"])
        eff_end = int(rec["valid_until_epoch"])
        if i + 1 < n:
            next_pub = int(transitions[i + 1]["published_epoch"])
            eff_end = min(eff_end, next_pub)
        out[(int(rec["seq"]), str(rec["transition_sha256"]))] = {
            "seq": int(rec["seq"]),
            "transition_sha256": str(rec["transition_sha256"]),
            "vintage_id": str(rec["vintage_id"]),
            "vintage_content_id": str(rec["vintage_content_id"]),
            "vintage_manifest_sha256": str(rec["vintage_manifest_sha256"]),
            "scheduled_refit_epoch": int(rec["scheduled_refit_epoch"]),
            "valid_from_epoch": int(rec["valid_from_epoch"]),
            "valid_until_epoch": int(rec["valid_until_epoch"]),
            "eff_start_epoch": eff_start,
            "eff_end_epoch": eff_end,
        }
    return out


def _writer_watermark_bytes(rec: dict[str, Any]) -> bytes:
    wm = {
        "schema": "sota-rolling-writer-watermark/v1",
        "book_id": str(rec["book_id"]),
        "policy_content_id": str(rec["policy_content_id"]),
        "max_seq": int(rec["seq"]),
        "last_transition_sha256": str(rec["transition_sha256"]),
        "published_epoch": int(rec["published_epoch"]),
        "updated_utc": str(rec["published_utc"]),
    }
    return _canonical_json_bytes(wm) + b"\n"


def repair_state_mirror_from_journal(
    book_vintage_dir: Path,
    *,
    expected_book_id: str,
    expected_policy_cid: str,
    max_vintage_age_days: int,
    trusted_root: Path | None = None,
    dir_fd: int | None = None,
) -> list[dict[str, Any]]:
    """Under LOCK_EX on .transition.flock, verify transitions.jsonl and repair ONLY a missing or strict-prefix state.json.

    Refuses fail-closed whenever state.json is ahead of or diverges from the verified journal (preventing
    journal-truncation rollback from overwriting an intact revocation mirror with an active predecessor!).
    """
    owns_fd = dir_fd is None
    active_fd: int | None = dir_fd
    if owns_fd:
        try:
            active_fd, _ = _open_dir_nofollow(book_vintage_dir, trusted_root=trusted_root)
        except _MissingFileError:
            return []
    assert active_fd is not None
    try:
        transitions, state_obj = load_verified_transitions(
            book_vintage_dir,
            expected_book_id=expected_book_id,
            expected_policy_cid=expected_policy_cid,
            max_vintage_age_days=max_vintage_age_days,
            trusted_root=trusted_root,
            dir_fd=active_fd,
        )
        if not transitions:
            return transitions

        tail = transitions[-1]
        if state_obj is not None and state_obj != tail:
            s_seq = _strict_int(state_obj.get("seq"))
            if s_seq is None or s_seq < 1 or s_seq > len(transitions):
                raise RollingVintageError(
                    f"rolling_state_mirror_ahead_or_invalid: state_seq={s_seq!r}, journal_len={len(transitions)}"
                )
            if state_obj != transitions[s_seq - 1]:
                raise RollingVintageError(f"rolling_state_mirror_diverges_from_journal at seq={s_seq}")

        # Ensure durable writer_watermark.json is synced before/alongside state.json mirror
        need_wm = True
        try:
            wb, _ = _read_regular_file_bytes_nofollow(active_fd, "writer_watermark.json")
            w_obj = _strict_json_loads(wb)
            if (
                isinstance(w_obj, dict)
                and w_obj.get("max_seq") == int(tail["seq"])
                and w_obj.get("last_transition_sha256") == str(tail["transition_sha256"])
            ):
                need_wm = False
        except _MissingFileError:
            need_wm = True
        if need_wm:
            _atomic_write_bytes_dirfd(active_fd, "writer_watermark.json", _writer_watermark_bytes(tail), mode=0o600)

        if state_obj != tail:
            _atomic_write_bytes_dirfd(active_fd, "state.json", _canonical_json_bytes(tail) + b"\n", mode=0o600)
        return transitions
    finally:
        if owns_fd:
            os.close(active_fd)


def commit_rolling_transition(
    book_vintage_dir: Path,
    *,
    policy: RollingSotaPolicy,
    status: str,
    published_epoch: int | None = None,
    scheduled_refit_epoch: int,
    next_refit_epoch: int,
    vintage_id: str | None = None,
    vintage_content_id: str | None = None,
    vintage_manifest_sha256: str | None = None,
    valid_from_epoch: int | None = None,
    valid_until_epoch: int | None = None,
    train_window_utc: list[str] | None = None,
    platt_coef: float | None = None,
    platt_intercept: float | None = None,
    reason: str | None = None,
    trusted_root: Path | None = None,
    expected_prev_seq: int | None = None,
    expected_prev_sha: str | None = None,
    forbid_abstained_block: bool = False,
    forbid_revoked_without_resume: bool = False,
    already_locked: bool = False,
    dir_fd: int | None = None,
) -> dict[str, Any]:
    """Atomically replace transitions.jsonl, writer_watermark.json & state.json via bound dir_fd under LOCK_EX."""
    if status not in VALID_TRANSITION_STATUSES:
        raise RollingVintageError(f"unsupported transition status {status!r}")
    resolved_dir = _ensure_dir_nofollow(book_vintage_dir, mode=0o700, trusted_root=trusted_root)
    lock_cm = (
        nullcontext(dir_fd)
        if already_locked
        else acquire_rolling_transition_lock(resolved_dir, exclusive=True, nonblocking=False, trusted_root=trusted_root)
    )
    with lock_cm as bound_fd:
        owns_fd = bound_fd is None
        active_fd = _open_dir_nofollow(resolved_dir, trusted_root=trusted_root)[0] if owns_fd else int(bound_fd)
        try:
            transitions = repair_state_mirror_from_journal(
                resolved_dir,
                expected_book_id=policy.book_id,
                expected_policy_cid=policy.content_id,
                max_vintage_age_days=policy.max_vintage_age_days,
                trusted_root=trusted_root,
                dir_fd=active_fd,
            )
            prev_seq = len(transitions)
            prev_sha = str(transitions[-1]["transition_sha256"]) if transitions else "GENESIS"
            if expected_prev_seq is not None and prev_seq != expected_prev_seq:
                raise RollingVintageError(f"rolling_journal_head_changed: seq {prev_seq} != expected {expected_prev_seq}")
            if expected_prev_sha is not None and prev_sha != expected_prev_sha:
                raise RollingVintageError("rolling_journal_head_changed: prev_sha mismatch")

            min_pub = int(transitions[-1]["published_epoch"]) if transitions else 1
            pub_ep = int(time.time()) if published_epoch is None else int(published_epoch)
            if pub_ep < min_pub:
                raise RollingVintageError(f"published_epoch {pub_ep} precedes prior transition {min_pub}")

            if forbid_abstained_block:
                if any(
                    t.get("status") == "abstained"
                    and (int(t["scheduled_refit_epoch"]) == int(scheduled_refit_epoch) or pub_ep < int(t["next_refit_epoch"]))
                    for t in transitions
                ):
                    raise RollingVintageError("rolling_block_abstained_locked_out")
            if forbid_revoked_without_resume and transitions and transitions[-1].get("status") == "revoked":
                raise RollingVintageError("rolling_manual_revocation_active_requires_resume")

            body: dict[str, Any] = {
                "schema": SOTA_ROLLING_TRANSITION_SCHEMA,
                "book_id": policy.book_id,
                "policy_content_id": policy.content_id,
                "seq": prev_seq + 1,
                "prev_transition_sha256": prev_sha,
                "status": status,
                "published_epoch": pub_ep,
                "published_utc": _utc_iso_whole(pub_ep),
                "scheduled_refit_epoch": int(scheduled_refit_epoch),
                "scheduled_refit_utc": _utc_iso_whole(int(scheduled_refit_epoch)),
                "next_refit_epoch": int(next_refit_epoch),
                "next_refit_utc": _utc_iso_whole(int(next_refit_epoch)),
                "vintage_id": vintage_id if status == "active" else None,
                "vintage_content_id": vintage_content_id if status == "active" else None,
                "vintage_manifest_sha256": vintage_manifest_sha256 if status == "active" else None,
                "valid_from_epoch": int(valid_from_epoch) if (status == "active" and valid_from_epoch is not None) else None,
                "valid_from_utc": _utc_iso_whole(int(valid_from_epoch)) if (status == "active" and valid_from_epoch is not None) else None,
                "valid_until_epoch": int(valid_until_epoch) if (status == "active" and valid_until_epoch is not None) else None,
                "valid_until_utc": _utc_iso_whole(int(valid_until_epoch)) if (status == "active" and valid_until_epoch is not None) else None,
                "train_window_utc": list(train_window_utc) if train_window_utc is not None else None,
                "platt_coef": float(platt_coef) if platt_coef is not None else None,
                "platt_intercept": float(platt_intercept) if platt_intercept is not None else None,
                "reason": str(reason) if status != "active" else None,
            }
            t_sha = _sha256_bytes(_canonical_json_bytes(body))
            rec = dict(body)
            rec["transition_sha256"] = t_sha
            validate_transition_record(
                rec,
                expected_book_id=policy.book_id,
                expected_policy_cid=policy.content_id,
                expected_seq=prev_seq + 1,
                expected_prev_sha=prev_sha,
                min_published_epoch=min_pub,
                max_vintage_age_days=policy.max_vintage_age_days,
            )
            line_bytes = _canonical_json_bytes(rec) + b"\n"
            full_journal_bytes = b"".join(_canonical_json_bytes(t) + b"\n" for t in transitions) + line_bytes

            # Step A: Atomically replace whole authoritative transitions.jsonl via bound dir_fd (zero torn appends!)
            _atomic_write_bytes_dirfd(active_fd, "transitions.jsonl", full_journal_bytes, mode=0o600)
            # Step B: Atomically persist writer-side anti-rollback watermark via bound dir_fd
            _atomic_write_bytes_dirfd(active_fd, "writer_watermark.json", _writer_watermark_bytes(rec), mode=0o600)
            # Step C: Atomically publish state.json mirror matching exact tail record via bound dir_fd
            _atomic_write_bytes_dirfd(active_fd, "state.json", line_bytes, mode=0o600)
            return rec
        finally:
            if owns_fd:
                os.close(active_fd)


# ---------------------------------------------------------------------------
# Durable SQLite Anti-Rollback Watermark (sota_risk_state)
# ---------------------------------------------------------------------------


def rolling_watermark_key(book_id: str, policy_content_id: str) -> str:
    return f"rolling_watermark:{book_id}:{policy_content_id}"


def read_rolling_watermark(conn: sqlite3.Connection, book_id: str, policy_content_id: str) -> dict[str, Any] | None:
    row = conn.execute(
        "SELECT value_json FROM sota_risk_state WHERE key=?",
        (rolling_watermark_key(book_id, policy_content_id),),
    ).fetchone()
    if row is None:
        return None
    obj = _strict_json_loads(str(row[0]).encode("utf-8"))
    if not isinstance(obj, dict):
        raise RollingVintageError("malformed rolling_watermark in sota_risk_state")
    return obj


def verify_and_update_rolling_watermark(
    conn: sqlite3.Connection | None,
    *,
    book_id: str,
    policy_content_id: str,
    transitions: list[dict[str, Any]],
    now_epoch: int,
    persist: bool = False,
) -> dict[str, Any] | None:
    """Verify transition chain extends persisted SQLite watermark with no rollback/fork; optionally persist advance."""
    wm = read_rolling_watermark(conn, book_id, policy_content_id) if conn is not None else None
    if wm is not None:
        wm_seq = _strict_int(wm.get("max_seq"))
        wm_sha = wm.get("last_transition_sha256")
        wm_ep = _strict_int(wm.get("max_seen_epoch"))
        if wm_seq is None or wm_seq < 0 or wm_ep is None or wm_ep <= 0:
            raise RollingVintageError("corrupt_rolling_watermark_row")
        if wm_seq > 0:
            if type(wm_sha) is not str or not _HEX64_RE.fullmatch(wm_sha):
                raise RollingVintageError("corrupt_rolling_watermark_sha")
            if len(transitions) < wm_seq:
                raise RollingVintageError("rolling_watermark_seq_rollback")
            if transitions[wm_seq - 1]["transition_sha256"] != wm_sha:
                raise RollingVintageError("rolling_watermark_chain_fork_or_substitution")
        if int(now_epoch) < (wm_ep - MAX_CLOCK_ROLLBACK_TOLERANCE_S):
            raise RollingVintageError("rolling_clock_rollback_detected")

    if not transitions:
        return wm

    tail = transitions[-1]
    new_seq = int(tail["seq"])
    new_sha = str(tail["transition_sha256"])
    prev_ep = int(wm["max_seen_epoch"]) if (wm is not None and _strict_int(wm.get("max_seen_epoch")) is not None) else 0
    new_ep = max(prev_ep, int(now_epoch), int(tail["published_epoch"]))
    updated = {
        "schema": "sota-rolling-watermark/v1",
        "book_id": book_id,
        "policy_content_id": policy_content_id,
        "max_seq": new_seq,
        "last_transition_sha256": new_sha,
        "max_seen_epoch": new_ep,
        "updated_utc": _utc_iso_whole(new_ep),
    }
    if persist and conn is not None:
        needs_write = (
            wm is None
            or wm.get("max_seq") != new_seq
            or wm.get("last_transition_sha256") != new_sha
            or int(wm.get("max_seen_epoch") or 0) < new_ep
        )
        if needs_write:
            conn.execute("BEGIN IMMEDIATE")
            try:
                conn.execute(
                    """
                    INSERT INTO sota_risk_state (key, value_json, updated_utc)
                    VALUES (?, ?, ?)
                    ON CONFLICT(key) DO UPDATE SET value_json=excluded.value_json, updated_utc=excluded.updated_utc
                    """,
                    (
                        rolling_watermark_key(book_id, policy_content_id),
                        json.dumps(updated, sort_keys=True),
                        updated["updated_utc"],
                    ),
                )
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise
    return updated


# ---------------------------------------------------------------------------
# Single-Read O_NOFOLLOW Vintage Loader & RollingSotaBookHandle
# ---------------------------------------------------------------------------


def load_verified_vintage_book(
    book_vintage_dir: Path,
    *,
    policy: RollingSotaPolicy,
    transition_rec: dict[str, Any],
    trusted_root: Path | None = None,
) -> LoadedSotaBook:
    """Load and authenticate one immutable runtime vintage strictly via single-read O_NOFOLLOW bytes."""
    vid = str(transition_rec["vintage_id"])
    want_cid = str(transition_rec["vintage_content_id"])
    want_msha = str(transition_rec["vintage_manifest_sha256"])
    if not _VINTAGE_ID_RE.fullmatch(vid):
        raise RollingVintageError(f"invalid vintage_id {vid!r}")

    v_dir = Path(book_vintage_dir) / "vintages" / vid
    v_fd, _ = _open_dir_nofollow(v_dir, trusted_root=trusted_root)
    try:
        man_bytes, _ = _read_regular_file_bytes_nofollow(v_fd, "vintage.manifest.json")
        if _sha256_bytes(man_bytes) != want_msha:
            raise RollingVintageError(f"{vid}: vintage.manifest.json SHA-256 mismatch")
        v_man = _strict_json_loads(man_bytes)
        if not isinstance(v_man, dict) or v_man.get("schema") != SOTA_ROLLING_VINTAGE_MANIFEST_SCHEMA:
            raise RollingVintageError(f"{vid}: invalid vintage manifest schema")
        if (
            v_man.get("book_id") != policy.book_id
            or v_man.get("policy_content_id") != policy.content_id
            or v_man.get("vintage_id") != vid
            or v_man.get("vintage_content_id") != want_cid
        ):
            raise RollingVintageError(f"{vid}: vintage manifest identity mismatch vs policy/transition")

        arts = v_man.get("artifacts")
        if not isinstance(arts, list) or len(arts) != 2:
            raise RollingVintageError(f"{vid}: vintage manifest must list exact 2 artifacts")
        art_map: dict[str, bytes] = {}
        for art in arts:
            fname = str(art.get("file") or "")
            raw_b, _ = _read_regular_file_bytes_nofollow(v_fd, fname)
            if _sha256_bytes(raw_b) != art.get("sha256") or len(raw_b) != art.get("bytes"):
                raise RollingVintageError(f"{vid}: artifact {fname!r} failed SHA-256 or byte-length check")
            art_map[fname] = raw_b
    finally:
        os.close(v_fd)

    strat_name = f"{policy.book_id}_strategy.json"
    weights_name = f"{policy.book_id}_weights.npz"
    if set(art_map.keys()) != {strat_name, weights_name}:
        raise RollingVintageError(f"{vid}: unexpected vintage artifact names {sorted(art_map.keys())}")
    got_cid = compute_sota_content_id_from_buffers([(k, v) for k, v in art_map.items()])
    if got_cid != want_cid:
        raise RollingVintageError(f"{vid}: computed vintage content_id {got_cid} != {want_cid}")

    strat = _strict_json_loads(art_map[strat_name])
    if not isinstance(strat, dict):
        raise RollingVintageError(f"{vid}: malformed strategy JSON")
    if (
        strat.get("book_id") != policy.book_id
        or strat.get("policy_content_id") != policy.content_id
        or strat.get("scheduled_refit_utc") != transition_rec.get("scheduled_refit_utc")
        or strat.get("valid_from_utc") != transition_rec.get("valid_from_utc")
        or strat.get("valid_until_utc") != transition_rec.get("valid_until_utc")
        or str(strat.get("pair") or "").upper() != policy.pair
        or _strict_int(strat.get("horizon_minutes")) != policy.horizon_minutes
        or tuple(str(s).upper() for s in (strat.get("allowed_sides") or [])) != policy.allowed_sides
        or str(strat.get("contract_family") or "") != policy.contract_family
        or tuple(str(c) for c in (strat.get("feature_cols") or [])) != policy.feature_cols
        or bool(strat.get("drop_ny_16h", False)) != policy.drop_ny_16h
        or "refit_cpcv_p10_floors" in strat
    ):
        raise RollingVintageError(f"{vid}: vintage strategy spec diverges from immutable policy")

    cov_v = _strict_float(strat.get("coverage"))
    thr_v = _strict_float(strat.get("confidence_threshold"))
    mp_v = _strict_float(strat.get("min_payout_ratio"))
    em_v = _strict_float(strat.get("payout_edge_margin"))
    fl_v = _strict_float((strat.get("rolling_oos_p10_floors") or {}).get("UP"))
    if (
        cov_v is None or not math.isclose(cov_v, policy.coverage, abs_tol=1e-12)
        or thr_v is None or not (0.0 <= thr_v < 0.5)
        or mp_v is None or not math.isclose(mp_v, policy.min_payout_ratio, abs_tol=1e-12)
        or em_v is None or not math.isclose(em_v, policy.payout_edge_margin, abs_tol=1e-12)
        or fl_v is None or not math.isclose(fl_v, policy.rolling_oos_p10_floors["UP"], abs_tol=1e-12)
    ):
        raise RollingVintageError(f"{vid}: vintage numeric strategy parameters diverge from policy")

    platt_cal = strat.get("platt_calibration")
    up_cal = platt_cal.get("UP") if isinstance(platt_cal, dict) else None
    p_coef = _strict_float(up_cal.get("coef")) if isinstance(up_cal, dict) else None
    p_int = _strict_float(up_cal.get("intercept")) if isinstance(up_cal, dict) else None
    if p_coef is None or p_coef <= 0.0 or p_int is None:
        raise RollingVintageError(f"{vid}: vintage platt_calibration['UP'] missing or non-positive")
    if not math.isclose(p_coef, float(transition_rec["platt_coef"]), rel_tol=1e-9, abs_tol=1e-9):
        raise RollingVintageError(f"{vid}: vintage platt_coef disagrees with transition record")

    # Parse weights strictly from the SAME authenticated in-memory bytes (zero re-open TOCTOU!)
    with np.load(io.BytesIO(art_map[weights_name]), allow_pickle=False) as z:
        weights_npz = {k: z[k].copy() for k in z.files}
    _validate_weights_npz(policy.book_id, policy.model_kind, weights_npz, len(policy.feature_cols))

    return LoadedSotaBook(
        book_id=policy.book_id,
        pair=policy.pair,
        horizon_minutes=policy.horizon_minutes,
        allowed_sides=policy.allowed_sides,
        lifecycle_status=policy.lifecycle_status,
        model_kind=policy.model_kind,
        contract_family=policy.contract_family,
        feature_cols=list(policy.feature_cols),
        coverage=policy.coverage,
        conf_thr=thr_v,
        up_floor=policy.rolling_oos_p10_floors["UP"],
        down_floor=None,
        payout_edge_margin=policy.payout_edge_margin,
        min_payout_ratio=policy.min_payout_ratio,
        drop_ny_16h=policy.drop_ny_16h,
        platt_calibration={"UP": {"coef": p_coef, "intercept": p_int}},
        content_id=want_cid,
        manifest=v_man,
        weights_npz=weights_npz,
        boosters=None,
    )


class RollingSotaBookHandle:
    """Dynamic fail-closed runtime handle for a rolling-refit SOTA policy."""

    def __init__(
        self,
        policy: RollingSotaPolicy,
        *,
        runtime_vintage_root: Path,
        trusted_root: Path | None = None,
    ):
        self.policy = policy
        self.runtime_vintage_root = Path(runtime_vintage_root)
        self.trusted_root = trusted_root
        self.book_vintage_dir = self.runtime_vintage_root / policy.book_id
        self.book_mode = ROLLING_BOOK_MODE
        self._active_book: LoadedSotaBook | None = None
        self._active_transition: dict[str, Any] | None = None
        self._max_seen_seq: int = 0
        self._max_seen_sha: str | None = None
        self._max_seen_epoch: int = 0
        self._locked_dir_fd: int | None = None
        self._last_verified_valid_from_epoch: int | None = None
        self._last_verified_valid_until_epoch: int | None = None

    @contextmanager
    def hold_transition_read_lock(self, *, nonblocking: bool = True) -> Iterator[int]:
        """Hold LOCK_SH on .transition.flock and bind all journal reads inside the context to its dir_fd."""
        with acquire_rolling_transition_lock(
            self.book_vintage_dir,
            exclusive=False,
            nonblocking=nonblocking,
            trusted_root=self.trusted_root,
        ) as d_fd:
            prev = self._locked_dir_fd
            self._locked_dir_fd = d_fd
            try:
                yield d_fd
            finally:
                self._locked_dir_fd = prev

    # Expose policy-level attributes matching LoadedSotaBook interface for preflight / anchor / executor
    @property
    def book_id(self) -> str:
        return self.policy.book_id

    @property
    def pair(self) -> str:
        return self.policy.pair

    @property
    def horizon_minutes(self) -> int:
        return self.policy.horizon_minutes

    @property
    def allowed_sides(self) -> tuple[str, ...]:
        return self.policy.allowed_sides

    @property
    def lifecycle_status(self) -> str:
        return self.policy.lifecycle_status

    @property
    def model_kind(self) -> str:
        return self.policy.model_kind

    @property
    def contract_family(self) -> str:
        return self.policy.contract_family

    @property
    def feature_cols(self) -> list[str]:
        return list(self.policy.feature_cols)

    @property
    def coverage(self) -> float:
        return self.policy.coverage

    @property
    def up_floor(self) -> float | None:
        return self.policy.rolling_oos_p10_floors.get("UP")

    @property
    def down_floor(self) -> float | None:
        return None

    @property
    def payout_edge_margin(self) -> float:
        return self.policy.payout_edge_margin

    @property
    def min_payout_ratio(self) -> float | None:
        return self.policy.min_payout_ratio

    @property
    def drop_ny_16h(self) -> bool:
        return self.policy.drop_ny_16h

    @property
    def content_id(self) -> str:
        return self.policy.content_id

    @property
    def manifest(self) -> dict[str, Any]:
        return self.policy.manifest

    @property
    def multiplier(self) -> int | None:
        return None

    @property
    def certified_commission_frac(self) -> float:
        return 0.0

    @property
    def max_commission_frac(self) -> float:
        return 0.20

    @property
    def min_stop_out_rel_dist(self) -> float:
        return 0.0008

    @property
    def barrier_contract_type(self) -> str | None:
        return None

    @property
    def barrier_rel_half_width(self) -> float | None:
        return None

    def side_floor(self, side: str) -> float:
        s = side.upper()
        if s not in self.allowed_sides or s not in self.policy.rolling_oos_p10_floors:
            raise RollingVintageError(f"{self.book_id}: side {s!r} not in allowed_sides {self.allowed_sides}")
        return float(self.policy.rolling_oos_p10_floors[s])

    def effective_floor(self, side: str, extra_margin: float = 0.0) -> float:
        return self.side_floor(side) - float(self.policy.payout_edge_margin) - float(extra_margin)

    def _unload(self) -> None:
        self._active_book = None
        self._active_transition = None

    def _check_mem_watermark(self, transitions: list[dict[str, Any]], now_epoch: int) -> None:
        if self._max_seen_epoch > 0 and now_epoch < (self._max_seen_epoch - MAX_CLOCK_ROLLBACK_TOLERANCE_S):
            raise RollingVintageError("rolling_clock_rollback_detected")
        if self._max_seen_seq > 0:
            if len(transitions) < self._max_seen_seq:
                raise RollingVintageError("rolling_mem_watermark_seq_rollback")
            if transitions[self._max_seen_seq - 1]["transition_sha256"] != self._max_seen_sha:
                raise RollingVintageError("rolling_mem_watermark_chain_fork")

    def _advance_mem_watermark(self, tail: dict[str, Any], now_epoch: int) -> None:
        self._max_seen_seq = max(self._max_seen_seq, int(tail["seq"]))
        if self._max_seen_seq == int(tail["seq"]):
            self._max_seen_sha = str(tail["transition_sha256"])
        self._max_seen_epoch = max(self._max_seen_epoch, int(now_epoch), int(tail["published_epoch"]))

    def resolve_active_vintage(
        self,
        now_epoch: int,
        *,
        conn: sqlite3.Connection | None = None,
        persist_watermark: bool = False,
    ) -> tuple[LoadedSotaBook | None, dict[str, Any] | None, str | None]:
        """Resolve and authenticate the current active vintage at now_epoch; fail closed on ANY issue."""
        ep = int(now_epoch)
        try:
            transitions, state_obj = load_verified_transitions(
                self.book_vintage_dir,
                expected_book_id=self.policy.book_id,
                expected_policy_cid=self.policy.content_id,
                max_vintage_age_days=self.policy.max_vintage_age_days,
                trusted_root=self.trusted_root,
            )
            self._check_mem_watermark(transitions, ep)
            verify_and_update_rolling_watermark(
                conn,
                book_id=self.policy.book_id,
                policy_content_id=self.policy.content_id,
                transitions=transitions,
                now_epoch=ep,
                persist=False,
            )
        except Exception as exc:
            self._unload()
            return None, None, f"rolling_journal_or_watermark_error:{exc}"

        if not transitions:
            self._unload()
            return None, None, "rolling_vintage_not_initialized"

        tail = transitions[-1]
        if state_obj != tail:
            self._unload()
            return None, tail, "rolling_state_mirror_mismatch"

        self._advance_mem_watermark(tail, ep)
        if persist_watermark and conn is not None:
            try:
                verify_and_update_rolling_watermark(
                    conn,
                    book_id=self.policy.book_id,
                    policy_content_id=self.policy.content_id,
                    transitions=transitions,
                    now_epoch=ep,
                    persist=True,
                )
            except Exception as exc:
                self._unload()
                return None, None, f"rolling_watermark_persist_failed:{exc}"

        st_val = str(tail["status"])
        if st_val == "abstained":
            self._unload()
            if ep < int(tail["scheduled_refit_epoch"]) or ep < int(tail["published_epoch"]):
                return None, tail, "rolling_vintage_abstained_not_yet_effective"
            if ep >= int(tail["next_refit_epoch"]):
                return None, tail, "rolling_vintage_abstained_expired"
            return None, tail, f"rolling_vintage_abstained:{tail.get('reason')}"
        if st_val != "active":
            self._unload()
            return None, tail, f"rolling_vintage_{st_val}:{tail.get('reason')}"

        vf = int(tail["valid_from_epoch"])
        vu = int(tail["valid_until_epoch"])
        if ep < vf:
            self._unload()
            return None, tail, "rolling_vintage_not_yet_active"
        if ep >= vu:
            self._unload()
            return None, tail, "rolling_vintage_expired"

        if (
            self._active_book is not None
            and self._active_transition is not None
            and self._active_transition.get("transition_sha256") == tail["transition_sha256"]
            and self._active_transition.get("vintage_id") == tail["vintage_id"]
        ):
            return self._active_book, tail, None

        try:
            loaded = load_verified_vintage_book(
                self.book_vintage_dir,
                policy=self.policy,
                transition_rec=tail,
                trusted_root=self.trusted_root,
            )
        except Exception as exc:
            self._unload()
            return None, tail, f"rolling_vintage_corrupt:{exc}"

        self._active_book = loaded
        self._active_transition = tail
        return loaded, tail, None

    def verify_vintage_active_for_signal(
        self,
        signal_payload: dict[str, Any],
        now_epoch: float,
        *,
        conn: sqlite3.Connection | None = None,
        persist_watermark: bool = False,
    ) -> float:
        """Verify signal's vintage is STILL active, unrevoked, and unexpired; return remaining validity seconds."""
        ep_int = int(math.floor(float(now_epoch)))
        want_pol = str(signal_payload.get("policy_content_id") or "")
        want_vid = str(signal_payload.get("vintage_id") or "")
        want_cid = str(signal_payload.get("book_content_id") or "")
        want_seq = _strict_int(signal_payload.get("transition_seq"))
        want_sha = str(signal_payload.get("transition_sha256") or "")
        if (
            want_pol != self.policy.content_id
            or not want_vid
            or not want_cid
            or want_seq is None
            or not _HEX64_RE.fullmatch(want_sha)
        ):
            raise RollingVintageError("rolling_signal_missing_provenance")

        transitions, state_obj = load_verified_transitions(
            self.book_vintage_dir,
            expected_book_id=self.policy.book_id,
            expected_policy_cid=self.policy.content_id,
            max_vintage_age_days=self.policy.max_vintage_age_days,
            trusted_root=self.trusted_root,
            dir_fd=self._locked_dir_fd,
        )
        self._check_mem_watermark(transitions, ep_int)
        verify_and_update_rolling_watermark(
            conn,
            book_id=self.policy.book_id,
            policy_content_id=self.policy.content_id,
            transitions=transitions,
            now_epoch=ep_int,
            persist=persist_watermark,
        )
        if not transitions:
            self._unload()
            raise RollingVintageError("rolling_vintage_not_initialized")
        tail = transitions[-1]
        if state_obj != tail:
            self._unload()
            raise RollingVintageError("rolling_state_mirror_mismatch")
        self._advance_mem_watermark(tail, ep_int)
        if tail.get("status") != "active":
            self._unload()
            raise RollingVintageError(f"rolling_vintage_{tail.get('status')}:{tail.get('reason')}")
        if (
            int(tail["seq"]) != want_seq
            or str(tail["transition_sha256"]) != want_sha
            or str(tail["vintage_id"]) != want_vid
            or str(tail["vintage_content_id"]) != want_cid
        ):
            self._unload()
            raise RollingVintageError("rolling_vintage_superseded_or_mismatch")
        vf_i = int(tail["valid_from_epoch"])
        vu_i = int(tail["valid_until_epoch"])
        sig_vf = _strict_int(signal_payload.get("vintage_valid_from_epoch"))
        sig_vu = _strict_int(signal_payload.get("vintage_valid_until_epoch"))
        if sig_vf != vf_i or sig_vu != vu_i:
            self._unload()
            raise RollingVintageError("rolling_signal_window_mismatch")
        self._last_verified_valid_from_epoch = vf_i
        self._last_verified_valid_until_epoch = vu_i
        vf = float(vf_i)
        vu = float(vu_i)
        if float(now_epoch) < vf:
            self._unload()
            raise RollingVintageError("rolling_vintage_not_yet_active")
        rem_s = vu - float(now_epoch)
        if rem_s <= 0.0:
            self._unload()
            raise RollingVintageError("rolling_vintage_expired")
        return rem_s


# ---------------------------------------------------------------------------
# Unified Loader Delegating Static Books to sota_book_runtime (Byte-Identical)
# ---------------------------------------------------------------------------


def load_sota_book_any(
    book_id: str,
    *,
    repo_root: Path = REPO_ROOT,
    allow_demo_canary: bool = True,
    runtime_vintage_root: Path | None = None,
) -> LoadedSotaBook | RollingSotaBookHandle:
    """Load either a static frozen book (books/SOTA_INDEX.json) or a Demo rolling policy (books/SOTA_ROLLING_INDEX.json).

    Static books resolve directly from SOTA_INDEX.json without reading SOTA_ROLLING_INDEX.json so a malformed
    rolling index can never block static book startup.
    """
    _check_book_id(book_id)
    static_idx = load_sota_index(repo_root=repo_root)
    if book_id in static_idx:
        return load_sota_book(book_id, repo_root=repo_root, allow_demo_canary=allow_demo_canary)
    rolling_idx = load_rolling_index(repo_root=repo_root)
    if book_id in rolling_idx:
        policy = load_rolling_policy(book_id, repo_root=repo_root, allow_demo_canary=allow_demo_canary)
        v_root = (
            Path(runtime_vintage_root)
            if runtime_vintage_root is not None
            else canonical_rolling_vintage_root(repo_root=repo_root)
        )
        return RollingSotaBookHandle(policy, runtime_vintage_root=v_root, trusted_root=repo_root)
    raise SotaBookError(f"{book_id}: not registered in SOTA_INDEX.json or SOTA_ROLLING_INDEX.json")


def load_enabled_sota_books_any(
    *,
    repo_root: Path = REPO_ROOT,
    account_mode: str = "demo",
    book_ids: list[str] | None = None,
    runtime_vintage_root: Path | None = None,
) -> list[LoadedSotaBook | RollingSotaBookHandle]:
    """Load explicitly requested static + rolling books (book_ids=None delegates strictly to static books)."""
    if book_ids is None:
        return list(load_enabled_sota_books(repo_root=repo_root, account_mode=account_mode, book_ids=None))
    if len(set(book_ids)) != len(book_ids):
        raise SotaBookError(f"duplicate book_id in book_ids: {book_ids!r}")
    allow_canary = str(account_mode).lower() == "demo"
    return [
        load_sota_book_any(
            bid,
            repo_root=repo_root,
            allow_demo_canary=allow_canary,
            runtime_vintage_root=runtime_vintage_root,
        )
        for bid in book_ids
    ]
