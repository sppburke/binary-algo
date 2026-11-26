#!/usr/bin/env python3
"""Preregistered DEV 5-gate certifier & zero-fit bundle verifier for USDJPY.sota_stack05_m30_up.v1.

Phase 1 (Child 1, spec §3.2):
- Confines sys.path pre-import to this checkout (refusing sibling /home/sean/git/binary-algo imports
  and never importing cpcv_deriv.py), checks /home/sean/git/binary-algo/features pre-import, and
  authenticates the exact 15-file PHASE1_SOURCE_CLOSURE_FILES digest closure.
- Loads ONLY pinned < 2026-08-01 DEV inputs via component-by-component O_NOFOLLOW verification.
- Runs exact 262 canonical roles (15 real CPCV, 4 forward, frozen_past, incumbent_frozen_past,
  240 null shifts W=3..48, test-free final_fit) with exact 4-key pre-debit fit ledger, single-threaded
  5-seed RBF Nystroem + LightGBM (passing feature_name=feature_cols), common linear percentiles,
  de-overlapped G3 freshness subgroup baseline (Owner Decision #1), G1_cpcv.paths replay export,
  raw strict_certificate interop (certificate._reread_result == (2, [], [], [])), and independent
  in-memory final_fit_expected_content_id on PASS.
- Provides strictly read-only, zero-fit `--verify-bundle` CLI authenticating every receipt digest,
  523 NPZ arrays, reconstructed selections, G1_cpcv.paths.trade_wins, reducers, fit ledger prefix,
  and (under --allow-builder-suffix) mandatory builder_suffix.json + PROGRESS.md debit witness.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Pre-import sys.path confinement (spec §3.2)
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
SIBLING_REPO_ROOT = Path("/home/sean/git/binary-algo").resolve()


def _is_within(candidate: Path, root: Path) -> bool:
    try:
        candidate.resolve().relative_to(root)
        return True
    except (OSError, ValueError):
        return False


def confine_sys_path(repo_root: Path = REPO_ROOT) -> None:
    """Place this checkout's scripts/ first and strip any sibling /home/sean/git/binary-algo path."""
    scripts_dir = (repo_root / "scripts").resolve()
    own_root = repo_root.resolve()
    cleaned: list[str] = []
    for entry in sys.path:
        if not entry:
            continue
        p = Path(entry)
        try:
            rp = p.resolve()
        except OSError:
            cleaned.append(entry)
            continue
        if rp != own_root and not _is_within(rp, own_root) and _is_within(rp, SIBLING_REPO_ROOT):
            continue
        if rp == scripts_dir:
            continue
        cleaned.append(entry)
    sys.path[:] = [str(scripts_dir)] + cleaned


confine_sys_path(REPO_ROOT)

# BLOCKER #8 fix: verify sibling /home/sean/git/binary-algo/features exists BEFORE any import that can reach pipeline.py!
_PIPELINE_FEATURES_DIR = Path("/home/sean/git/binary-algo/features")
if not _PIPELINE_FEATURES_DIR.is_dir():
    raise RuntimeError(
        f"refusing import: required existing directory missing ({_PIPELINE_FEATURES_DIR}); "
        f"pipeline.py must not create it"
    )

import argparse
import copy
import datetime as dt
import fcntl
import hashlib
import io
import itertools
import json
import math
import re
import stat
import time
import zipfile
from decimal import Decimal, localcontext
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from numpy.lib import format as npfmt
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import LogisticRegression, SGDClassifier
from zoneinfo import ZoneInfo

import certificate as cert_mod
from deriv_runtime_core_v2 import evaluate_sota_payout_gate
from sota_book_freeze import (
    atomic_write_file_bytes,
    extract_nystroem_lgb_stack_arrays,
    fsync_ancestry,
    fsync_dir,
    fsync_file,
    rename_noreplace,
    serialize_sota_lgb_txt_bytes,
    serialize_sota_strategy_bytes,
    serialize_sota_weights_npz_bytes,
)
from sota_book_runtime import (
    SotaBookError,
    check_pipeline_features_dir_exists,
    compute_sota_content_id_from_buffers,
)

# ---------------------------------------------------------------------------
# Pinned Constants & Source Closure (spec §3.2)
# ---------------------------------------------------------------------------

BOOK_ID = "USDJPY.sota_stack05_m30_up.v1"
ARM_NAME = "stack05_m30_up_v1"
PAIR = "USDJPY"
HORIZON_MIN = 30
HORIZON_S = 1800
TOL_S = 60
COV = 0.05
INCUMBENT_COV = 0.03
QUOTED_PAYOUT_RATIO = 1.82
MIN_PAYOUT_RATIO = 1.78
EV_MARGIN = 0.005
BREAKEVEN_WR = 0.541
DIAG_HIGH_WR = 0.565
MIN_PATH_TRADES = 25

CUTOFF_UTC = 1785542400  # 2026-08-01T00:00:00Z
FROZEN_PAST_SPLIT_UTC = 1767225600  # 2026-01-01T00:00:00Z
SHARED_GROUP_BOUNDS_UTC = [
    1754870400,
    1759982400,
    1765094400,
    1770206400,
    1775318400,
    1780430400,
    1785542400,
]
PATH_YEARS = 10_224_000 / 31_557_600
NULL_SPAN_S = 30_660_960
NULL_SHIFTS_WEEKS = [3 * k for k in range(1, 17)]
DELAYS_EXTRA_S = [0, 2, 4, 9, 29]
NY_TZ = ZoneInfo("America/New_York")

DEFAULT_DEV_DATASET_DIR = Path("/mnt/data/deriv-sota-lab/cache/dataset_dev_v0.1")
DEFAULT_TICKS_ROOT = Path("/mnt/data/deriv/ticks")
DEFAULT_OUT_DIR = Path("/mnt/data/deriv-sota-lab/tests/stack05_m30_up_v1")
DEFAULT_ARCHIVE_DIR = Path("/mnt/data/deriv-sota-lab/archive/stack05_m30_up_v1")
DEFAULT_PROGRESS_MD = Path("/mnt/data/deriv-sota-lab/PROGRESS.md")

DEV_MANIFEST_SHA256 = "4f434cfd799e237647776b7f3e3e14b04be305594bd55bf37344321be9185c4a"
HISTORICAL_COMPARATOR_REL = "results/json/deriv_5gate_USDJPY_m30_up_stack_cov05_result.json"
HISTORICAL_COMPARATOR_SHA256 = "6edf5a9881b7c4c6089ed15205045c6a7e0b037c3f6c6d187fdc8aad078be328"

PINNED_DEV_PARQUETS: dict[str, tuple[int, str]] = {
    "USDJPY": (104211487, "9d9072fbafd8b58937db4d3a1a24a200f092a3e6c55367f65f2a814ac0ae141f"),
    "AUDUSD": (101156579, "bfbf2677c10ad959bd2ef5fcde80e7d2ecb77c549c232d005fff86f5512dae93"),
    "EURUSD": (101121240, "04e953dd737313ef29ba3421888cf523759b9df0ab846ccaa4ae0c3f73d51cc0"),
    "GBPUSD": (102613207, "79ecfa921312a262cc8f062308bd3e653504ea00d6c99ac031186fbfb869cf0b"),
    "NZDUSD": (99989704, "417c3eb0be8d4f5452882893c24db2d7d7851d522d35b9091b8bd2cf8240c3a6"),
    "USDCAD": (101289834, "71b97c19c02f62683becb0fab74f8711610f2035f88dd5b4f28664a57faff614"),
    "USDCHF": (100253644, "5d1b8774958284296f52ea2ff66fe7d3fd51c8163d871439aec7a40c52942630"),
}

PHASE1_SOURCE_CLOSURE_FILES: tuple[str, ...] = (
    "book_runtime.py",
    "build_certified_sota_books.py",
    "certificate.py",
    "certify_sota_stack05.py",
    "deriv_backfill.py",
    "deriv_client.py",
    "deriv_dataset.py",
    "deriv_floor_resolver.py",
    "deriv_runtime_core.py",
    "deriv_runtime_core_v2.py",
    "harness.py",
    "live_features.py",
    "pipeline.py",
    "sota_book_freeze.py",
    "sota_book_runtime.py",
)

CANONICAL_DOSSIER_NAME = "deriv_5gate_USDJPY_sota_stack05_m30_up_v1_result.json"
CANONICAL_LOCK_NAME = "sota_stack05_cert.lock"
CANONICAL_CKPT_NAME = "sota_stack05_cert_ckpt.json"
CANONICAL_SCORES_NAME = "sota_stack05_cert_scores.npz"
CANONICAL_LEDGER_NAME = "sota_stack05_fit_ledger.json"
CANONICAL_RECEIPT_NAME = "sota_stack05_cert_receipt.json"
CANONICAL_BUILDER_SUFFIX_NAME = "builder_suffix.json"

G4_MODE_PLATT_EV = "platt_ev"
G4_MODE_RAW_TOP5 = "raw_top5"
REGISTERED_ARM_G4_MODES: dict[str, str] = {
    "stack05_m30_up_v1": G4_MODE_PLATT_EV,
    "stack05_m30_up_v2": G4_MODE_RAW_TOP5,
}


def validate_arm_g4_mode(arm_name: str, g4_mode: str) -> None:
    if g4_mode not in (G4_MODE_PLATT_EV, G4_MODE_RAW_TOP5):
        raise OperationalHoldError(f"unsupported g4_mode: {g4_mode!r}")
    expected = REGISTERED_ARM_G4_MODES.get(arm_name)
    if expected is not None and g4_mode != expected:
        raise OperationalHoldError(
            f"registered arm {arm_name!r} requires g4_mode={expected!r}, got {g4_mode!r}"
        )


def default_out_dir_for_arm(arm_name: str = ARM_NAME) -> Path:
    artifact_names_for_arm(arm_name)
    return DEFAULT_OUT_DIR.parent / arm_name


def default_archive_dir_for_arm(arm_name: str = ARM_NAME) -> Path:
    artifact_names_for_arm(arm_name)
    return DEFAULT_ARCHIVE_DIR.parent / arm_name


def artifact_names_for_arm(arm_name: str = ARM_NAME) -> dict[str, str]:
    """Return canonical artifact filenames for arm_name (exact legacy names preserved for v1)."""
    if arm_name == ARM_NAME:
        return {
            "dossier": CANONICAL_DOSSIER_NAME,
            "lock": CANONICAL_LOCK_NAME,
            "ckpt": CANONICAL_CKPT_NAME,
            "scores": CANONICAL_SCORES_NAME,
            "ledger": CANONICAL_LEDGER_NAME,
            "receipt": CANONICAL_RECEIPT_NAME,
            "builder_suffix": CANONICAL_BUILDER_SUFFIX_NAME,
        }
    if arm_name == "stack05" or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", arm_name):
        raise OperationalHoldError(f"invalid or reserved arm_name: {arm_name!r}")
    return {
        "dossier": f"deriv_5gate_USDJPY_sota_{arm_name}_result.json",
        "lock": f"sota_{arm_name}_cert.lock",
        "ckpt": f"sota_{arm_name}_cert_ckpt.json",
        "scores": f"sota_{arm_name}_cert_scores.npz",
        "ledger": f"sota_{arm_name}_fit_ledger.json",
        "receipt": f"sota_{arm_name}_cert_receipt.json",
        "builder_suffix": CANONICAL_BUILDER_SUFFIX_NAME,
    }

FIT_LEDGER_SCHEMA = "sota-stack05-fit-ledger/v1"
RECEIPT_SCHEMA = "sota-5gate-certification-receipt/v1"
DOSSIER_SCHEMA = "sota-5gate-certification-dossier/v3"
MAX_FIT_UNITS = 263
MAX_CERTIFIER_ATTEMPTS = 262
MAX_BUILDER_ATTEMPTS = 1

MACRO_COLS = frozenset({"hrs_to_high_macro", "high_macro_next_24h", "hrs_since_macro", "last_macro_abs_z"})
EXTRA_VOL_PREFIXES = ("rv_up_", "rv_dn_", "rv_asym_", "jump_share_", "parkinson_", "park_comp_", "rv_4320m", "rv_7200m")


class OperationalHoldError(RuntimeError):
    """Fail-closed operational / custody / verification hold (NEVER misclassified as scientific KILL-STACK)."""


# ---------------------------------------------------------------------------
# Strict Canonical JSON, Hashing & O_NOFOLLOW DEV Input Loaders
# ---------------------------------------------------------------------------


def canonical_json_bytes(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def pretty_json_bytes(obj: Any) -> bytes:
    return (json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def strict_json_loads(raw: bytes | str) -> Any:
    text = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else str(raw)

    def _no_dups(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for k, v in pairs:
            if k in out:
                raise OperationalHoldError(f"duplicate JSON key {k!r}")
            out[k] = v
        return out

    def _reject_const(tok: str) -> Any:
        raise OperationalHoldError(f"non-finite JSON constant rejected: {tok!r}")

    return json.loads(text, object_pairs_hook=_no_dups, parse_constant=_reject_const)


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _open_dir_nofollow(root: Path, rel_dirs: list[str]) -> int:
    # Walk from filesystem root or absolute anchor component-by-component with O_NOFOLLOW
    abs_root = Path(root)
    if not abs_root.is_absolute():
        raise OperationalHoldError(f"O_NOFOLLOW loader requires absolute root path, got {root}")
    parts = list(abs_root.parts)
    assert parts[0] == "/"
    fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
    try:
        for comp in parts[1:] + list(rel_dirs):
            if comp in ("", ".", "..") or "/" in comp:
                raise OperationalHoldError(f"illegal path component {comp!r}")
            try:
                nxt = os.open(comp, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            except OSError as exc:
                raise OperationalHoldError(f"O_NOFOLLOW directory walk refused at {comp!r}: {exc}") from exc
            os.close(fd)
            fd = nxt
        return fd
    except Exception:
        os.close(fd)
        raise


def read_verified_file_nofollow(
    base_dir: Path,
    rel_parts: list[str],
    *,
    expected_bytes: int | None = None,
    expected_sha256: str | None = None,
) -> bytes:
    """Component-by-component O_NOFOLLOW walk + S_ISREG + st_nlink==1 + single-read digest check."""
    if not rel_parts:
        raise OperationalHoldError("empty relative path for O_NOFOLLOW read")
    dir_parts = rel_parts[:-1]
    fname = rel_parts[-1]
    if fname in ("", ".", "..") or "/" in fname:
        raise OperationalHoldError(f"illegal filename {fname!r}")
    dir_fd = _open_dir_nofollow(base_dir, dir_parts)
    try:
        try:
            f_fd = os.open(fname, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=dir_fd)
        except OSError as exc:
            raise OperationalHoldError(f"O_NOFOLLOW file open refused for {base_dir.joinpath(*rel_parts)}: {exc}") from exc
        try:
            st = os.fstat(f_fd)
            if not stat.S_ISREG(st.st_mode):
                raise OperationalHoldError(f"non-regular file refused: {base_dir.joinpath(*rel_parts)}")
            if st.st_nlink != 1:
                raise OperationalHoldError(
                    f"multi-link file refused (st_nlink={st.st_nlink}): {base_dir.joinpath(*rel_parts)}"
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
    finally:
        os.close(dir_fd)

    if expected_bytes is not None and len(raw) != int(expected_bytes):
        raise OperationalHoldError(
            f"byte size mismatch for {base_dir.joinpath(*rel_parts)}: {len(raw)} != {expected_bytes}"
        )
    if expected_sha256 is not None:
        got = sha256_bytes(raw)
        if got != expected_sha256:
            raise OperationalHoldError(
                f"SHA-256 mismatch for {base_dir.joinpath(*rel_parts)}: {got} != {expected_sha256}"
            )
    return raw


def load_pinned_dev_manifest(
    dev_dir: Path = DEFAULT_DEV_DATASET_DIR,
    *,
    expected_sha256: str | None = DEV_MANIFEST_SHA256,
) -> dict[str, Any]:
    raw = read_verified_file_nofollow(dev_dir, ["manifest.json"], expected_sha256=expected_sha256)
    obj = strict_json_loads(raw)
    if not isinstance(obj, dict) or not isinstance(obj.get("inputs"), dict):
        raise OperationalHoldError(f"malformed DEV manifest at {dev_dir / 'manifest.json'}")
    return obj


def load_pinned_dev_parquet(
    pair: str,
    dev_dir: Path = DEFAULT_DEV_DATASET_DIR,
    *,
    pinned_parquets: dict[str, tuple[int, str]] | None = None,
) -> pd.DataFrame:
    p = str(pair).upper()
    pin_map = PINNED_DEV_PARQUETS if pinned_parquets is None else pinned_parquets
    if p not in pin_map:
        raise OperationalHoldError(f"pair {p!r} is not in PINNED_DEV_PARQUETS")
    exp_bytes, exp_sha = pin_map[p]
    raw = read_verified_file_nofollow(dev_dir, [f"{p}.parquet"], expected_bytes=exp_bytes, expected_sha256=exp_sha)
    df = pd.read_parquet(io.BytesIO(raw))
    if "decision_ts" not in df.columns or len(df) == 0:
        raise OperationalHoldError(f"DEV parquet {p} missing decision_ts or empty")
    if int(df["decision_ts"].max()) >= CUTOFF_UTC:
        raise OperationalHoldError(f"DEV parquet {p} contains post-cutoff timestamp >= {CUTOFF_UTC}")
    return df


_TICK_DATE_RE = re.compile(r"^([A-Z0-9]+)_(\d{4}-\d{2}-\d{2})_ticks\.parquet$")


def load_pinned_dev_ticks(
    pair: str,
    *,
    dev_dir: Path = DEFAULT_DEV_DATASET_DIR,
    ticks_root: Path = DEFAULT_TICKS_ROOT,
    manifest: dict[str, Any] | None = None,
    first_day: str = "2025-08-11",
    last_day: str = "2026-07-31",
    cutoff_utc: int = CUTOFF_UTC,
    expected_manifest_sha256: str | None = DEV_MANIFEST_SHA256,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load strictly pinned DEV daily ticks (< 2026-08-01) via O_NOFOLLOW; empty input holds."""
    p = str(pair).upper()
    if last_day >= "2026-08-01":
        raise OperationalHoldError(f"refusing post-DEV tick date last_day={last_day!r}")
    man = manifest if manifest is not None else load_pinned_dev_manifest(dev_dir, expected_sha256=expected_manifest_sha256)
    entries = (man.get("inputs") or {}).get(p)
    if not isinstance(entries, list) or not entries:
        raise OperationalHoldError(f"empty or missing manifest inputs for {p}")

    parts_ts: list[np.ndarray] = []
    parts_px: list[np.ndarray] = []
    loaded_names: list[str] = []
    for ent in entries:
        fname = str(ent["file"])
        m = _TICK_DATE_RE.match(fname)
        if not m or m.group(1) != p:
            raise OperationalHoldError(f"unexpected tick filename {fname!r} for pair {p}")
        day_s = m.group(2)
        if day_s < "2025-08-11" or day_s > "2026-07-31":
            raise OperationalHoldError(f"manifest tick file {fname!r} outside registered DEV window 2025-08-11..2026-07-31")
        if day_s < first_day or day_s > last_day:
            continue
        raw = read_verified_file_nofollow(
            ticks_root,
            [p, fname],
            expected_bytes=int(ent["bytes"]),
            expected_sha256=str(ent["sha256"]),
        )
        tbl = pq.read_table(io.BytesIO(raw), columns=["datetime_utc", "price"])
        ts_arr = tbl.column(0).cast("int64").to_numpy() // 1_000_000_000
        px_arr = tbl.column(1).to_numpy().astype(np.float64, copy=False)
        parts_ts.append(ts_arr)
        parts_px.append(px_arr)
        loaded_names.append(fname)

    if not parts_ts:
        raise OperationalHoldError(f"empty_registered_dev_ticks: no tick files matched {p} in {first_day}..{last_day}")
    ts = np.concatenate(parts_ts).astype(np.int64, copy=False)
    px = np.concatenate(parts_px).astype(np.float64, copy=False)
    if len(ts) == 0:
        raise OperationalHoldError(f"empty_registered_dev_ticks: zero ticks loaded for {p}")
    keep = ts < int(cutoff_utc)
    ts = ts[keep]
    px = px[keep]
    if len(ts) == 0:
        raise OperationalHoldError(f"empty_registered_dev_ticks: zero pre-cutoff ticks for {p}")
    order = np.argsort(ts, kind="stable")
    return ts[order], px[order], loaded_names


def verify_source_closure_and_loaded_modules(repo_root: Path = REPO_ROOT) -> tuple[str, dict[str, str]]:
    """Compute local_source_closure_sha256 over the exact 15 files and authenticate sys.modules."""
    own_root = repo_root.resolve()
    scripts_dir = (own_root / "scripts").resolve()
    file_hashes: dict[str, str] = {}
    for name in sorted(PHASE1_SOURCE_CLOSURE_FILES):
        raw = read_verified_file_nofollow(scripts_dir, [name])
        file_hashes[name] = sha256_bytes(raw)
    closure_sha = sha256_bytes(canonical_json_bytes(file_hashes))

    allowed_set = set(PHASE1_SOURCE_CLOSURE_FILES)
    for mod_name, mod in list(sys.modules.items()):
        if mod_name == "cpcv_deriv":
            raise OperationalHoldError("forbidden import detected: cpcv_deriv")
        mfile = getattr(mod, "__file__", None)
        if not mfile:
            continue
        try:
            mp = Path(mfile).resolve()
        except OSError:
            continue
        if own_root != SIBLING_REPO_ROOT and _is_within(mp, SIBLING_REPO_ROOT):
            raise OperationalHoldError(f"sibling repo module imported: {mod_name} -> {mp}")
        if _is_within(mp, scripts_dir) and mp.name not in allowed_set:
            raise OperationalHoldError(f"module outside PHASE1_SOURCE_CLOSURE_FILES imported: {mod_name} -> {mp.name}")
    return closure_sha, file_hashes


# ---------------------------------------------------------------------------
# Pure Settlement, Eligibility, Splitting, Purging & Strategy Builder
# ---------------------------------------------------------------------------


def nonoverlap_chrono(ts: np.ndarray, mask: np.ndarray, gap: int = HORIZON_S) -> np.ndarray:
    """First-come chronological non-overlap selection with strict gap >= H (no cpcv_deriv import)."""
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return np.empty(0, dtype=np.int64)
    chosen: list[int] = []
    last_t = -10**18
    g = int(gap)
    for i in idx:
        t = int(ts[i])
        if t - last_t >= g:
            chosen.append(int(i))
            last_t = t
    return np.asarray(chosen, dtype=np.int64)


def settle_with_intervals(
    decision_ts: np.ndarray,
    tick_ts: np.ndarray,
    tick_px: np.ndarray,
    *,
    horizon_s: int = HORIZON_S,
    delay_s: int = 0,
    tol_s: int = TOL_S,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Deriv-faithful realized-interval settlement bit-identical to deriv_dataset.settle() at d=0.

    Returns (ret, valid, entry_real_ts, exit_real_ts). Safely returns NaN/False/-1/-1 on empty ticks.
    """
    dec = np.asarray(decision_ts, dtype=np.int64)
    m = len(dec)
    ts = np.asarray(tick_ts, dtype=np.int64)
    px = np.asarray(tick_px, dtype=np.float64)
    n = len(ts)
    if n == 0 or m == 0:
        return (
            np.full(m, np.nan, dtype=np.float64),
            np.zeros(m, dtype=bool),
            np.full(m, -1, dtype=np.int64),
            np.full(m, -1, dtype=np.int64),
        )
    entry_t = dec + 1 + int(delay_s)
    expiry_t = entry_t + int(horizon_s)
    ei = np.searchsorted(ts, entry_t, side="left")
    xi = np.searchsorted(ts, expiry_t, side="right") - 1
    eic = np.clip(ei, 0, n - 1)
    xic = np.clip(xi, 0, n - 1)
    valid = (ei < n) & (xi > ei) & ((ts[eic] - entry_t) <= int(tol_s)) & ((expiry_t - ts[xic]) <= int(tol_s))
    with np.errstate(divide="ignore", invalid="ignore"):
        ret = np.where(valid, px[xic] / px[eic] - 1.0, np.nan)
    valid = valid & np.isfinite(ret)
    entry_real = np.where(valid, ts[eic], -1).astype(np.int64)
    exit_real = np.where(valid, ts[xic], -1).astype(np.int64)
    return ret.astype(np.float64), valid.astype(bool), entry_real, exit_real


def select_core66_columns(df_dev: pd.DataFrame) -> list[str]:
    fcols = [
        c
        for c in df_dev.columns
        if c not in ("bar_start_ts", "decision_ts")
        and c not in MACRO_COLS
        and not c.startswith(("ret_fwd_", "max_ret_", "min_ret_", "valid_"))
        and not c.startswith(EXTRA_VOL_PREFIXES)
    ]
    if len(fcols) != 66:
        raise OperationalHoldError(f"expected exact 66 core66 columns, got {len(fcols)}")
    return fcols


def build_eligible_indices(
    df_dev: pd.DataFrame,
    fcols: list[str],
    valid_true: np.ndarray,
    ret_fwd_true: np.ndarray,
    *,
    cutoff_utc: int = CUTOFF_UTC,
) -> np.ndarray:
    ts = df_dev["decision_ts"].to_numpy(dtype=np.int64)
    if len(ts) >= 2 and not bool(np.all(ts[:-1] < ts[1:])):
        raise OperationalHoldError("df_dev decision_ts is not strictly ascending")
    X = df_dev[fcols].to_numpy(dtype=np.float64)
    row_ok = np.mean(~np.isfinite(X), axis=1) < 0.50
    dt_utc = pd.to_datetime(ts, unit="s", utc=True)
    dt_ny = dt_utc.tz_convert(NY_TZ)
    wd = dt_ny.dayofweek.to_numpy()
    ny_mod = dt_ny.hour.to_numpy() * 60 + dt_ny.minute.to_numpy()
    in_ny = (wd < 5) & (ny_mod >= 480) & (ny_mod < 980)  # 08:00 <= NY < 16:20
    utc_ok = dt_utc.hour.to_numpy() < 21
    cutoff_ok = (ts + 1 + 29 + HORIZON_S + TOL_S) < int(cutoff_utc)
    elig = row_ok & valid_true & np.isfinite(ret_fwd_true) & in_ny & utc_ok & cutoff_ok
    return np.flatnonzero(elig).astype(np.int64)


def _merge_closed_intervals(starts: np.ndarray, ends: np.ndarray, pad: int = 0) -> tuple[np.ndarray, np.ndarray]:
    if len(starts) == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    lo = np.asarray(starts, dtype=np.int64) - int(pad)
    hi = np.asarray(ends, dtype=np.int64) + int(pad)
    order = np.argsort(lo, kind="mergesort")
    lo = lo[order]
    hi = hi[order]
    m_lo: list[int] = [int(lo[0])]
    m_hi: list[int] = [int(hi[0])]
    for a, b in zip(lo[1:], hi[1:]):
        ia, ib = int(a), int(b)
        if ia <= m_hi[-1]:
            if ib > m_hi[-1]:
                m_hi[-1] = ib
        else:
            m_lo.append(ia)
            m_hi.append(ib)
    return np.asarray(m_lo, dtype=np.int64), np.asarray(m_hi, dtype=np.int64)


def intervals_overlap_any(
    a_entry: np.ndarray,
    a_exit: np.ndarray,
    b_entry: np.ndarray,
    b_exit: np.ndarray,
    *,
    pad: int = 0,
) -> np.ndarray:
    """Boolean mask over A: True iff [a_entry, a_exit] overlaps any [b_entry - pad, b_exit + pad]."""
    if len(a_entry) == 0 or len(b_entry) == 0:
        return np.zeros(len(a_entry), dtype=bool)
    m_lo, m_hi = _merge_closed_intervals(b_entry, b_exit, pad=pad)
    k = np.searchsorted(m_hi, a_entry, side="left")
    in_bounds = k < len(m_lo)
    out = np.zeros(len(a_entry), dtype=bool)
    if np.any(in_bounds):
        kb = k[in_bounds]
        out[in_bounds] = m_lo[kb] <= a_exit[in_bounds]
    return out


def split_train_val(
    pool_idx: np.ndarray,
    ts: np.ndarray,
    entry_ts: np.ndarray,
    exit_ts: np.ndarray,
    ret_fwd: np.ndarray,
    *,
    gap: int = HORIZON_S,
    val_frac: float = 0.15,
    min_val: int = 500,
    donor_entry_ts: np.ndarray | None = None,
    donor_exit_ts: np.ndarray | None = None,
    test_idx: np.ndarray | None = None,
) -> dict[str, Any]:
    """Exact inner split & ordered pre-score support guards (spec §3.2 + B2 donor-purge precedence)."""
    pool = np.asarray(pool_idx, dtype=np.int64)
    g = int(gap)
    is_null = donor_entry_ts is not None and donor_exit_ts is not None

    def _fail(reason: str) -> dict[str, Any]:
        return {
            "support_ok": False,
            "failure_reason": reason,
            "val_h1_idx": np.empty(0, dtype=np.int64),
            "val_h2_idx": np.empty(0, dtype=np.int64),
            "val_idx": np.empty(0, dtype=np.int64),
            "fit_all_idx": np.empty(0, dtype=np.int64),
            "mv_fit_idx": np.empty(0, dtype=np.int64),
            "es_sub_idx": np.empty(0, dtype=np.int64),
            "tr_sub_idx": np.empty(0, dtype=np.int64),
        }

    if len(pool) < min_val:
        return _fail("pool_below_min_val")

    n_val_raw = max(int(min_val), int(round(len(pool) * float(val_frac))))
    val_raw_idx = pool[-n_val_raw:]
    mid = len(val_raw_idx) // 2
    if mid < 1 or (len(val_raw_idx) - mid) < 1:
        return _fail("val_half_empty")

    val_h2_idx = val_raw_idx[mid:]
    h1_cand = val_raw_idx[:mid]
    h2_first_ts = int(ts[val_h2_idx[0]])
    h2_first_entry = int(entry_ts[val_h2_idx[0]])
    val_h1_idx = h1_cand[(ts[h1_cand] < h2_first_ts - g) & (exit_ts[h1_cand] < h2_first_entry)]
    if is_null:
        assert donor_entry_ts is not None and donor_exit_ts is not None
        conflict = intervals_overlap_any(
            donor_entry_ts[val_h1_idx],
            donor_exit_ts[val_h1_idx],
            donor_entry_ts[val_h2_idx],
            donor_exit_ts[val_h2_idx],
            pad=g,
        )
        val_h1_idx = val_h1_idx[~conflict]

    val_idx = np.concatenate([val_h1_idx, val_h2_idx])
    y_val = ret_fwd[val_idx] > 0.0
    pos_v = int(np.sum(y_val))
    neg_v = int(len(val_idx) - pos_v)
    if len(val_h1_idx) < 1 or len(val_h2_idx) < 1 or len(val_idx) < int(min_val) or min(pos_v, neg_v) < 10:
        return _fail("val_support_failed")

    val_first_ts = int(ts[val_idx[0]])
    val_first_entry = int(entry_ts[val_idx[0]])
    fit_all_idx = pool[(ts[pool] < val_first_ts - g) & (exit_ts[pool] < val_first_entry)]
    if is_null:
        assert donor_entry_ts is not None and donor_exit_ts is not None
        conflict = intervals_overlap_any(
            donor_entry_ts[fit_all_idx],
            donor_exit_ts[fit_all_idx],
            donor_entry_ts[val_idx],
            donor_exit_ts[val_idx],
            pad=g,
        )
        fit_all_idx = fit_all_idx[~conflict]
    if len(fit_all_idx) < 1000:
        return _fail("fit_all_below_1000")

    mv_fit_idx = fit_all_idx[ret_fwd[fit_all_idx] != 0.0]
    if len(mv_fit_idx) < 300:
        return _fail("mv_fit_below_300")

    n_es = max(300, int(round(len(mv_fit_idx) * 0.12)))
    es_sub_idx = mv_fit_idx[-n_es:]
    es_first_ts = int(ts[es_sub_idx[0]])
    es_first_entry = int(entry_ts[es_sub_idx[0]])
    tr_sub_idx = mv_fit_idx[(ts[mv_fit_idx] < es_first_ts - g) & (exit_ts[mv_fit_idx] < es_first_entry)]
    if is_null:
        assert donor_entry_ts is not None and donor_exit_ts is not None
        conflict = intervals_overlap_any(
            donor_entry_ts[tr_sub_idx],
            donor_exit_ts[tr_sub_idx],
            donor_entry_ts[es_sub_idx],
            donor_exit_ts[es_sub_idx],
            pad=g,
        )
        tr_sub_idx = tr_sub_idx[~conflict]

    y_tr = ret_fwd[tr_sub_idx] > 0.0
    y_mv = ret_fwd[mv_fit_idx] > 0.0
    pos_tr, neg_tr = int(np.sum(y_tr)), int(len(tr_sub_idx) - np.sum(y_tr))
    pos_mv, neg_mv = int(np.sum(y_mv)), int(len(mv_fit_idx) - np.sum(y_mv))
    if len(tr_sub_idx) < 500 or len(es_sub_idx) < 300 or min(pos_tr, neg_tr) < 20 or min(pos_mv, neg_mv) < 20:
        return _fail("tr_or_es_support_failed")

    # Assert zero realized label-source interval overlap across every boundary
    lbl_e = donor_entry_ts if is_null else entry_ts
    lbl_x = donor_exit_ts if is_null else exit_ts
    assert lbl_e is not None and lbl_x is not None
    pad_check = g if is_null else 0
    for name_a, a, name_b, b in (
        ("fit_all", fit_all_idx, "val", val_idx),
        ("val_h1", val_h1_idx, "val_h2", val_h2_idx),
        ("tr_sub", tr_sub_idx, "es_sub", es_sub_idx),
    ):
        if np.any(intervals_overlap_any(lbl_e[a], lbl_x[a], lbl_e[b], lbl_x[b], pad=pad_check)):
            raise OperationalHoldError(f"label-source interval overlap detected between {name_a} and {name_b}")
    if test_idx is not None and len(test_idx) > 0:
        for name_a, a in (("pool", pool), ("fit_all", fit_all_idx), ("val", val_idx)):
            if np.any(intervals_overlap_any(lbl_e[a], lbl_x[a], lbl_e[test_idx], lbl_x[test_idx], pad=pad_check)):
                raise OperationalHoldError(f"label-source interval overlap detected between {name_a} and test")

    return {
        "support_ok": True,
        "failure_reason": None,
        "val_h1_idx": val_h1_idx,
        "val_h2_idx": val_h2_idx,
        "val_idx": val_idx,
        "fit_all_idx": fit_all_idx,
        "mv_fit_idx": mv_fit_idx,
        "es_sub_idx": es_sub_idx,
        "tr_sub_idx": tr_sub_idx,
    }


def _hash_role_indices(
    fit_all_idx: np.ndarray,
    val_h1_idx: np.ndarray,
    val_h2_idx: np.ndarray,
    val_idx: np.ndarray,
    tr_sub_idx: np.ndarray,
    es_sub_idx: np.ndarray,
    test_idx: np.ndarray,
) -> str:
    h = hashlib.sha256()
    for name, arr in (
        ("fit_all", fit_all_idx),
        ("val_h1", val_h1_idx),
        ("val_h2", val_h2_idx),
        ("val", val_idx),
        ("tr_sub", tr_sub_idx),
        ("es_sub", es_sub_idx),
        ("test", test_idx),
    ):
        a64 = np.asarray(arr, dtype=np.int64)
        h.update(name.encode("utf-8"))
        h.update(b":")
        h.update(str(len(a64)).encode("utf-8"))
        h.update(b":")
        h.update(a64.tobytes())
        h.update(b"\0")
    return h.hexdigest()


def canonical_role_names() -> list[str]:
    names: list[str] = []
    pairs = list(itertools.combinations(range(6), 2))
    for ga, gb in pairs:
        names.append(f"real_{ga}_{gb}")
    for g in range(2, 6):
        names.append(f"fwd_{g}")
    names.append("frozen_past")
    names.append("incumbent_frozen_past")
    for w in NULL_SHIFTS_WEEKS:
        for ga, gb in pairs:
            names.append(f"null_{w}_{ga}_{gb}")
    names.append("final_fit")
    assert len(names) == 262
    return names


def canonical_scores_npz_keys() -> list[str]:
    keys: list[str] = []
    for r in canonical_role_names():
        if r == "final_fit":
            keys.append("final_fit_val")
        else:
            keys.append(f"{r}_val")
            keys.append(f"{r}_test")
    assert len(keys) == 523
    return keys


def build_all_role_specs(
    ts_all: np.ndarray,
    ret_fwd_true: np.ndarray,
    valid_true: np.ndarray,
    entry_true: np.ndarray,
    exit_true: np.ndarray,
    eligible_idx: np.ndarray,
) -> list[dict[str, Any]]:
    """Build exact 262 role split specifications in canonical order (zero fitting)."""
    bounds = SHARED_GROUP_BOUNDS_UTC
    t0 = int(ts_all[0])
    # Exact timestamp lookup map for null donor joins (exact equality ONLY; never as-of fallback!)
    ts_to_row = {int(t): idx for idx, t in enumerate(ts_all)}
    pairs = list(itertools.combinations(range(6), 2))
    roles: list[dict[str, Any]] = []

    def _pack_role(
        role: str,
        kind: str,
        pool_idx: np.ndarray,
        test_idx: np.ndarray,
        ret_arr: np.ndarray,
        donor_entry: np.ndarray | None = None,
        donor_exit: np.ndarray | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        is_assessment = role != "final_fit"
        if is_assessment and len(test_idx) < 1:
            sp = {
                "support_ok": False,
                "failure_reason": "empty_test_idx",
                "val_h1_idx": np.empty(0, dtype=np.int64),
                "val_h2_idx": np.empty(0, dtype=np.int64),
                "val_idx": np.empty(0, dtype=np.int64),
                "fit_all_idx": np.empty(0, dtype=np.int64),
                "mv_fit_idx": np.empty(0, dtype=np.int64),
                "es_sub_idx": np.empty(0, dtype=np.int64),
                "tr_sub_idx": np.empty(0, dtype=np.int64),
            }
        else:
            sp = split_train_val(
                pool_idx,
                ts_all,
                entry_true,
                exit_true,
                ret_arr,
                gap=HORIZON_S,
                val_frac=0.15,
                min_val=500,
                donor_entry_ts=donor_entry,
                donor_exit_ts=donor_exit,
                test_idx=test_idx if is_assessment else None,
            )
        r_hash = _hash_role_indices(
            sp["fit_all_idx"],
            sp["val_h1_idx"],
            sp["val_h2_idx"],
            sp["val_idx"],
            sp["tr_sub_idx"],
            sp["es_sub_idx"],
            test_idx,
        )
        out = {
            "role": role,
            "kind": kind,
            "cov": INCUMBENT_COV if role == "incumbent_frozen_past" else COV,
            "is_incumbent": role == "incumbent_frozen_past",
            "test_idx": test_idx,
            "ret_arr": ret_arr,
            "role_indices_sha256": r_hash,
            **sp,
        }
        if extra:
            out.update(extra)
        return out

    # 1. 15 real_<g_a>_<g_b>
    elig_ts = ts_all[eligible_idx]
    for ga, gb in pairs:
        lo_a, hi_a = bounds[ga], bounds[ga + 1]
        lo_b, hi_b = bounds[gb], bounds[gb + 1]
        in_a = (elig_ts >= lo_a) & (elig_ts < hi_a)
        in_b = (elig_ts >= lo_b) & (elig_ts < hi_b)
        test_mask = in_a | in_b
        test_idx = eligible_idx[test_mask]
        emb_a = (elig_ts >= lo_a - HORIZON_S) & (elig_ts < hi_a + HORIZON_S)
        emb_b = (elig_ts >= lo_b - HORIZON_S) & (elig_ts < hi_b + HORIZON_S)
        pool_cand = eligible_idx[~(test_mask | emb_a | emb_b)]
        if len(test_idx) > 0 and len(pool_cand) > 0:
            ov = intervals_overlap_any(entry_true[pool_cand], exit_true[pool_cand], entry_true[test_idx], exit_true[test_idx], pad=0)
            pool_idx = pool_cand[~ov]
        else:
            pool_idx = pool_cand
        roles.append(
            _pack_role(
                f"real_{ga}_{gb}",
                "real",
                pool_idx,
                test_idx,
                ret_fwd_true,
                extra={"path_id": f"{ga}_{gb}", "test_groups": [ga, gb]},
            )
        )

    # 2. 4 fwd_2..fwd_5
    for g in range(2, 6):
        lo, hi = bounds[g], bounds[g + 1]
        test_idx = eligible_idx[(elig_ts >= lo) & (elig_ts < hi)]
        if len(test_idx) > 0:
            min_entry = int(np.min(entry_true[test_idx]))
            pool_idx = eligible_idx[(elig_ts < lo - HORIZON_S) & (exit_true[eligible_idx] < min_entry)]
        else:
            pool_idx = np.empty(0, dtype=np.int64)
        roles.append(_pack_role(f"fwd_{g}", "fwd", pool_idx, test_idx, ret_fwd_true, extra={"group": g}))

    # 3 & 4. frozen_past & incumbent_frozen_past (exact identical split!)
    fp_test = eligible_idx[(elig_ts >= FROZEN_PAST_SPLIT_UTC) & (elig_ts < CUTOFF_UTC)]
    if len(fp_test) > 0:
        fp_min_entry = int(np.min(entry_true[fp_test]))
        fp_pool = eligible_idx[(elig_ts < FROZEN_PAST_SPLIT_UTC - HORIZON_S) & (exit_true[eligible_idx] < fp_min_entry)]
    else:
        fp_pool = np.empty(0, dtype=np.int64)
    roles.append(_pack_role("frozen_past", "frozen_past", fp_pool, fp_test, ret_fwd_true))
    roles.append(_pack_role("incumbent_frozen_past", "incumbent_frozen_past", fp_pool, fp_test, ret_fwd_true))

    # 5. 240 null_<W>_<g_a>_<g_b> (exact timestamp equality join; missing/invalid exact donors discarded!)
    n_rows = len(ts_all)
    for w in NULL_SHIFTS_WEEKS:
        tgt_ts_elig = t0 + ((elig_ts - t0 + w * 604800) % NULL_SPAN_S)
        keep_mask = np.zeros(len(eligible_idx), dtype=bool)
        donor_idx_list = np.full(len(eligible_idx), -1, dtype=np.int64)
        for pos_i, tgt in enumerate(tgt_ts_elig):
            j = ts_to_row.get(int(tgt), -1)
            if j >= 0 and valid_true[j] and math.isfinite(float(ret_fwd_true[j])):
                keep_mask[pos_i] = True
                donor_idx_list[pos_i] = j
        kept_recv = eligible_idx[keep_mask]
        kept_donor = donor_idx_list[keep_mask]
        # Construct donor arrays indexed by receiver row index
        ret_donor = np.full(n_rows, np.nan, dtype=np.float64)
        entry_donor = np.full(n_rows, -1, dtype=np.int64)
        exit_donor = np.full(n_rows, -1, dtype=np.int64)
        ret_donor[kept_recv] = ret_fwd_true[kept_donor]
        entry_donor[kept_recv] = entry_true[kept_donor]
        exit_donor[kept_recv] = exit_true[kept_donor]
        kept_ts = ts_all[kept_recv]
        n_discarded = int(len(eligible_idx) - len(kept_recv))

        for ga, gb in pairs:
            lo_a, hi_a = bounds[ga], bounds[ga + 1]
            lo_b, hi_b = bounds[gb], bounds[gb + 1]
            in_a = (kept_ts >= lo_a) & (kept_ts < hi_a)
            in_b = (kept_ts >= lo_b) & (kept_ts < hi_b)
            tmask = in_a | in_b
            test_idx = kept_recv[tmask]
            emb_a = (kept_ts >= lo_a - HORIZON_S) & (kept_ts < hi_a + HORIZON_S)
            emb_b = (kept_ts >= lo_b - HORIZON_S) & (kept_ts < hi_b + HORIZON_S)
            pool_cand = kept_recv[~(tmask | emb_a | emb_b)]
            if len(test_idx) > 0 and len(pool_cand) > 0:
                ov_r = intervals_overlap_any(
                    entry_true[pool_cand], exit_true[pool_cand], entry_true[test_idx], exit_true[test_idx], pad=0
                )
                ov_d = intervals_overlap_any(
                    entry_donor[pool_cand], exit_donor[pool_cand], entry_donor[test_idx], exit_donor[test_idx], pad=HORIZON_S
                )
                pool_idx = pool_cand[~(ov_r | ov_d)]
            else:
                pool_idx = pool_cand
            roles.append(
                _pack_role(
                    f"null_{w}_{ga}_{gb}",
                    "null",
                    pool_idx,
                    test_idx,
                    ret_donor,
                    donor_entry=entry_donor,
                    donor_exit=exit_donor,
                    extra={"week_shift": w, "path_id": f"{ga}_{gb}", "test_groups": [ga, gb], "missing_donors_discarded": n_discarded},
                )
            )

    # 6. final_fit (test-free)
    roles.append(_pack_role("final_fit", "final_fit", eligible_idx, np.empty(0, dtype=np.int64), ret_fwd_true))
    assert [r["role"] for r in roles] == canonical_role_names()
    return roles


def build_stack05_strategy_dict(
    *,
    feature_cols: list[str],
    confidence_threshold: float,
    p10_wr: float,
    platt_calibration: dict[str, dict[str, float]],
) -> dict[str, Any]:
    """Single canonical strategy dict owner shared by certifier (expected content_id) and builder."""
    return {
        "book_id": BOOK_ID,
        "pair": PAIR,
        "horizon_minutes": HORIZON_MIN,
        "allowed_sides": ["UP"],
        "model_kind": "nystroem-lgb-stack",
        "contract_family": "risefall",
        "feature_cols": list(feature_cols),
        "coverage": float(COV),
        "confidence_threshold": float(confidence_threshold),
        "refit_cpcv_p10_floors": {"UP": float(p10_wr)},
        "payout_edge_margin": float(EV_MARGIN),
        "min_payout_ratio": float(MIN_PAYOUT_RATIO),
        "drop_ny_16h": False,
        "platt_calibration": platt_calibration,
    }


# ---------------------------------------------------------------------------
# Deterministic Trainer (5-seed RBF Nystroem + Single LightGBM with feature_name)
# ---------------------------------------------------------------------------


def fit_role_trainer(
    X_all: np.ndarray,
    feature_cols: list[str],
    role_spec: dict[str, Any],
    *,
    return_estimators: bool = False,
) -> dict[str, Any]:
    """Fit exact spec §3.2 role trainer (candidate stack or incumbent 5-seed Nystroem)."""
    fit_all_idx = role_spec["fit_all_idx"]
    mv_fit_idx = role_spec["mv_fit_idx"]
    tr_sub_idx = role_spec["tr_sub_idx"]
    es_sub_idx = role_spec["es_sub_idx"]
    val_idx = role_spec["val_idx"]
    test_idx = role_spec["test_idx"]
    ret_arr = role_spec["ret_arr"]
    is_incumbent = bool(role_spec["is_incumbent"])

    Xf_all = X_all[fit_all_idx]
    med = np.nanmedian(np.where(np.isfinite(Xf_all), Xf_all, np.nan), axis=0)
    med = np.where(np.isfinite(med), med, 0.0).astype(np.float64)
    Zf_all = np.where(np.isfinite(Xf_all), Xf_all, med).astype(np.float64)
    mu = Zf_all.mean(axis=0).astype(np.float64)
    sd = (Zf_all.std(axis=0) + 1e-12).astype(np.float64)
    Zf_all = np.clip((Zf_all - mu) / sd, -5.0, 5.0)

    Xv = X_all[val_idx]
    Zv = np.clip((np.where(np.isfinite(Xv), Xv, med).astype(np.float64) - mu) / sd, -5.0, 5.0)
    has_test = len(test_idx) > 0
    if has_test:
        Xt = X_all[test_idx]
        Zt = np.clip((np.where(np.isfinite(Xt), Xt, med).astype(np.float64) - mu) / sd, -5.0, 5.0)
    else:
        Xt = np.empty((0, X_all.shape[1]), dtype=np.float64)
        Zt = np.empty((0, X_all.shape[1]), dtype=np.float64)

    mv_in_fit = ret_arr[fit_all_idx] != 0.0
    yf = (ret_arr[mv_fit_idx] > 0.0).astype(np.int64)
    assert int(np.sum(mv_in_fit)) == len(mv_fit_idx)

    ny_list: list[Any] = []
    sgd_list: list[Any] = []
    p_nys_val = np.zeros(len(val_idx), dtype=np.float64)
    p_nys_test = np.zeros(len(test_idx), dtype=np.float64)

    for seed in range(5):
        rng = np.random.default_rng(1000 + seed * 37)
        if len(Zf_all) > 45_000:
            sub = rng.choice(len(Zf_all), 45_000, replace=False)
        else:
            sub = np.arange(len(Zf_all))
        ny = Nystroem(kernel="rbf", gamma=1.0 / 66.0, n_components=500, random_state=seed)
        ny.fit(Zf_all[sub])
        Fk_all = ny.transform(Zf_all)
        sgd = SGDClassifier(loss="log_loss", penalty="l2", alpha=1e-4, max_iter=30, tol=1e-4, random_state=seed)
        sgd.fit(Fk_all[mv_in_fit], yf)
        p_nys_val += sgd.predict_proba(ny.transform(Zv))[:, 1]
        if has_test:
            p_nys_test += sgd.predict_proba(ny.transform(Zt))[:, 1]
        if return_estimators:
            ny_list.append(ny)
            sgd_list.append(sgd)

    p_nys_val /= 5.0
    if has_test:
        p_nys_test /= 5.0

    lgb_clf = None
    lgb_best_iter = None
    if is_incumbent:
        val_p_raw = p_nys_val
        test_p_raw = p_nys_test
    else:
        y_tr = (ret_arr[tr_sub_idx] > 0.0).astype(np.int64)
        y_es = (ret_arr[es_sub_idx] > 0.0).astype(np.int64)
        X_tr_raw = np.asarray(X_all[tr_sub_idx], dtype=np.float64)
        X_es_raw = np.asarray(X_all[es_sub_idx], dtype=np.float64)
        lgb_clf = lgb.LGBMClassifier(
            n_estimators=220,
            learning_rate=0.03,
            num_leaves=15,
            min_child_samples=200,
            subsample=0.8,
            subsample_freq=0,
            colsample_bytree=0.7,
            reg_lambda=5.0,
            random_state=42,
            deterministic=True,
            force_col_wise=True,
            n_jobs=1,
            verbose=-1,
        )
        lgb_clf.fit(
            X_tr_raw,
            y_tr,
            feature_name=list(feature_cols),
            eval_set=[(X_es_raw, y_es)],
            eval_metric="binary_logloss",
            callbacks=[lgb.early_stopping(25, verbose=False)],
        )
        bi = getattr(lgb_clf, "best_iteration_", None)
        lgb_best_iter = int(bi) if (bi is not None and int(bi) > 0) else 220
        booster = lgb_clf.booster_
        if list(booster.feature_name()) != list(feature_cols):
            raise OperationalHoldError("LightGBM booster.feature_name() mismatch after fit")
        p_lgb_val = np.asarray(booster.predict(np.asarray(Xv, dtype=np.float64), num_iteration=lgb_best_iter), dtype=np.float64)
        val_p_raw = 0.5 * p_nys_val + 0.5 * p_lgb_val
        if has_test:
            p_lgb_test = np.asarray(booster.predict(np.asarray(Xt, dtype=np.float64), num_iteration=lgb_best_iter), dtype=np.float64)
            test_p_raw = 0.5 * p_nys_test + 0.5 * p_lgb_test
        else:
            test_p_raw = np.empty(0, dtype=np.float64)

    out: dict[str, Any] = {
        "val_p_raw": val_p_raw.astype(np.float64),
        "test_p_raw": test_p_raw.astype(np.float64),
        "lgb_best_iteration": lgb_best_iter,
    }
    if return_estimators:
        out.update(
            {
                "med": med,
                "mu": mu,
                "sd": sd,
                "ny_list": ny_list,
                "sgd_list": sgd_list,
                "lgb_clf": lgb_clf,
            }
        )
    return out


# ---------------------------------------------------------------------------
# Calibration & Zero-Fit Evaluation / Reducers (spec §3.2)
# ---------------------------------------------------------------------------


def _clipped_logits(p: np.ndarray) -> np.ndarray:
    pc = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    return np.clip(np.log(pc / (1.0 - pc)), -15.0, 15.0)


def _eval_platt(p_raw: np.ndarray, coef: float, intercept: float) -> np.ndarray:
    z = float(coef) * _clipped_logits(p_raw) + float(intercept)
    out = np.empty_like(z, dtype=np.float64)
    pos = z >= 0.0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return np.where(np.isfinite(z), out, np.nan)


def fit_role_calibration(val_p_raw: np.ndarray, ret_val: np.ndarray, cov: float) -> dict[str, Any]:
    """Fit UP Platt calibration & conf_thr strictly on val_idx (ties-lose target ret_val > 0)."""
    if len(val_p_raw) == 0 or not bool(np.all(np.isfinite(val_p_raw))):
        return {
            "cal_admissible": False,
            "failure_reason": "nonfinite_val_scores",
            "n_val_oos": int(len(val_p_raw)),
            "conf_thr": None,
            "platt_coef": None,
            "platt_intercept": None,
        }
    conf_thr = float(np.quantile(np.abs(val_p_raw - 0.5), 1.0 - float(cov), method="linear"))
    y_cal = (ret_val > 0.0).astype(np.int64)
    if len(val_p_raw) < 500 or int(np.sum(y_cal)) == 0 or int(np.sum(y_cal)) == len(y_cal):
        return {
            "cal_admissible": False,
            "failure_reason": "up_calibration_inadmissible",
            "n_val_oos": int(len(val_p_raw)),
            "conf_thr": conf_thr if math.isfinite(conf_thr) else None,
            "platt_coef": None,
            "platt_intercept": None,
        }
    clf = LogisticRegression(C=1.0, max_iter=1000).fit(_clipped_logits(val_p_raw).reshape(-1, 1), y_cal)
    coef = float(clf.coef_[0, 0])
    intercept = float(clf.intercept_[0])
    if not (math.isfinite(conf_thr) and math.isfinite(coef) and coef > 0.0 and math.isfinite(intercept)):
        return {
            "cal_admissible": False,
            "failure_reason": "up_calibration_inadmissible",
            "n_val_oos": int(len(val_p_raw)),
            "conf_thr": conf_thr if math.isfinite(conf_thr) else None,
            "platt_coef": coef if math.isfinite(coef) else None,
            "platt_intercept": intercept if math.isfinite(intercept) else None,
        }
    return {
        "cal_admissible": True,
        "failure_reason": None,
        "n_val_oos": int(len(val_p_raw)),
        "conf_thr": conf_thr,
        "platt_coef": coef,
        "platt_intercept": intercept,
    }


def _select_role_trades(
    p_raw: np.ndarray,
    ts_sub: np.ndarray,
    ret_sub: np.ndarray,
    *,
    conf_thr: float,
    platt_coef: float,
    platt_intercept: float,
) -> dict[str, Any]:
    p_cal = _eval_platt(p_raw, platt_coef, platt_intercept)
    mask = (
        np.isfinite(p_raw)
        & (p_raw > 0.5)
        & (np.abs(p_raw - 0.5) >= float(conf_thr))
        & np.isfinite(p_cal)
        & ((p_cal * QUOTED_PAYOUT_RATIO - 1.0) >= EV_MARGIN)
    )
    sel_local = nonoverlap_chrono(ts_sub, mask, gap=HORIZON_S)
    trade_ts = ts_sub[sel_local].astype(np.int64)
    trade_wins = (ret_sub[sel_local] > 0.0).astype(np.int64)
    n = int(len(sel_local))
    wins = int(np.sum(trade_wins)) if n > 0 else 0
    wr = float(wins / n) if n > 0 else None
    ev = float((wins / n) * QUOTED_PAYOUT_RATIO - 1.0) if n > 0 else None
    return {
        "sel_local": sel_local,
        "trade_ts": trade_ts,
        "trade_wins": trade_wins,
        "trade_ts_sha256": sha256_bytes(trade_ts.tobytes()),
        "n": n,
        "wins": wins,
        "wr": wr,
        "ev": ev,
    }


def _select_role_trades_raw_top_cov(
    p_raw: np.ndarray,
    ts_sub: np.ndarray,
    ret_sub: np.ndarray,
    *,
    conf_thr: float,
) -> dict[str, Any]:
    """Raw uncalibrated top-coverage UP selection (no Platt/EV filter; apples-to-apples G4 comparator)."""
    mask = (
        np.isfinite(p_raw)
        & (p_raw > 0.5)
        & (np.abs(p_raw - 0.5) >= float(conf_thr))
    )
    sel_local = nonoverlap_chrono(ts_sub, mask, gap=HORIZON_S)
    trade_ts = ts_sub[sel_local].astype(np.int64)
    trade_wins = (ret_sub[sel_local] > 0.0).astype(np.int64)
    n = int(len(sel_local))
    wins = int(np.sum(trade_wins)) if n > 0 else 0
    wr = float(wins / n) if n > 0 else None
    ev = float((wins / n) * QUOTED_PAYOUT_RATIO - 1.0) if n > 0 else None
    return {
        "sel_local": sel_local,
        "trade_ts": trade_ts,
        "trade_wins": trade_wins,
        "trade_ts_sha256": sha256_bytes(trade_ts.tobytes()),
        "n": n,
        "wins": wins,
        "wr": wr,
        "ev": ev,
    }


def _linear_p10(values: list[float]) -> float | None:
    finite = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not finite:
        return None
    return float(np.percentile(finite, 10, method="linear"))


def _bootstrap_ci95_wr(wins_arr: np.ndarray, n_boot: int = 5000, seed: int = 7) -> list[float] | None:
    if len(wins_arr) == 0:
        return None
    rng = np.random.default_rng(seed)
    n = len(wins_arr)
    w = wins_arr.astype(np.float64)
    means = np.array([w[rng.integers(0, n, n)].mean() for _ in range(n_boot)], dtype=np.float64)
    return [float(np.percentile(means, 2.5, method="linear")), float(np.percentile(means, 97.5, method="linear"))]


def _decimal_ann_net_usd_1000(wins: int, n: int, f_q: float, r_pay: float, path_years: float) -> float:
    """Deterministic 60-digit Decimal overflow-safe annualized net USD per $1000 diagnostic."""
    with localcontext() as ctx:
        ctx.prec = 60
        df = Decimal(str(f_q))
        dr = Decimal(str(r_pay))
        dpy = Decimal(str(path_years))
        g_win = (Decimal(1) + df * (dr - Decimal(1))).ln()
        g_los = (Decimal(1) - df).ln()
        ann_g = (Decimal(wins) * g_win + Decimal(n - wins) * g_los) / dpy
        # Cap exponent only if mathematically astronomically overflowing float64 (> 700)
        if ann_g > Decimal("700"):
            ann_g = Decimal("700")
        net = Decimal(1000) * (ann_g.exp() - Decimal(1))
        return float(net)


def _compute_historical_6pair_max_tie_rate(
    dev_dir: Path = DEFAULT_DEV_DATASET_DIR,
    *,
    pinned_parquets: dict[str, tuple[int, str]] | None = None,
) -> dict[str, Any]:
    """Compute exact 6-pair historical tie-rate comparator on NY weekday 08:00-16:20 & valid_30m."""
    from fractions import Fraction

    peers = ("EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDCAD", "USDCHF")
    rates: dict[str, dict[str, Any]] = {}
    max_frac = Fraction(0, 1)
    max_pair = ""
    for p in peers:
        df_p = load_pinned_dev_parquet(p, dev_dir=dev_dir, pinned_parquets=pinned_parquets)
        ts = df_p["decision_ts"].to_numpy(dtype=np.int64)
        dt_ny = pd.to_datetime(ts, unit="s", utc=True).tz_convert(NY_TZ)
        wd = dt_ny.dayofweek.to_numpy()
        ny_mod = dt_ny.hour.to_numpy() * 60 + dt_ny.minute.to_numpy()
        ret = df_p["ret_fwd_30m"].to_numpy(dtype=np.float64)
        val = df_p["valid_30m"].to_numpy(dtype=bool)
        mask = (wd < 5) & (ny_mod >= 480) & (ny_mod < 980) & val & np.isfinite(ret)
        tot = int(np.sum(mask))
        ties = int(np.sum(mask & (ret == 0.0)))
        if tot <= 0:
            raise OperationalHoldError(f"empty 30m comparator population for {p}")
        fr = Fraction(ties, tot)
        rates[p] = {"ties": ties, "total": tot, "rate": float(fr)}
        if fr > max_frac:
            max_frac = fr
            max_pair = p
    return {"peers": rates, "max_pair": max_pair, "max_ties": max_frac.numerator, "max_total": max_frac.denominator, "max_rate": float(max_frac)}


def evaluate_all_gates_from_scores(
    *,
    df_dev: pd.DataFrame,
    fcols: list[str],
    tick_ts: np.ndarray,
    tick_px: np.ndarray,
    ret_fwd_true: np.ndarray,
    valid_true: np.ndarray,
    entry_true: np.ndarray,
    exit_true: np.ndarray,
    eligible_idx: np.ndarray,
    role_specs: list[dict[str, Any]],
    scores_dict: dict[str, np.ndarray],
    cal_by_role: dict[str, dict[str, Any]],
    peer_tie_info: dict[str, Any],
    preregistration_payload: dict[str, Any],
    preregistration_sha256: str,
    local_source_closure_sha256: str,
    final_fit_expected_identity: dict[str, Any] | None = None,
    strict_verify_cal: bool = False,
    arm_name: str = ARM_NAME,
    g4_mode: str | None = None,
) -> dict[str, Any]:
    """Pure zero-fit evaluation of all 262 roles, G1-G5, final_fit, and raw strict_certificate."""
    from fractions import Fraction

    eff_g4_mode = g4_mode
    if eff_g4_mode is None:
        eff_g4_mode = (
            G4_MODE_RAW_TOP5
            if preregistration_payload.get("g4_selection_mode") == "raw_top5_apples_to_apples"
            else G4_MODE_PLATT_EV
        )
    if eff_g4_mode not in (G4_MODE_PLATT_EV, G4_MODE_RAW_TOP5):
        raise OperationalHoldError(f"unsupported g4_mode: {eff_g4_mode!r}")
    arm_names = artifact_names_for_arm(arm_name)

    ts_all = df_dev["decision_ts"].to_numpy(dtype=np.int64)
    rv60_all = df_dev["rv_60m"].to_numpy(dtype=np.float64)

    # 1. Reconstruct every role's metrics strictly from scores_dict + cal_by_role (zero fitting!)
    role_records: dict[str, dict[str, Any]] = {}
    role_trades: dict[str, dict[str, Any]] = {}
    raw_top5_trades: dict[str, dict[str, Any]] = {}
    for spec in role_specs:
        role = spec["role"]
        is_final = role == "final_fit"
        is_null = spec["kind"] == "null"
        is_real = spec["kind"] == "real"
        val_key = "final_fit_val" if is_final else f"{role}_val"
        test_key = None if is_final else f"{role}_test"
        val_scores = np.asarray(scores_dict[val_key], dtype=np.float64)
        test_scores = np.empty(0, dtype=np.float64) if is_final else np.asarray(scores_dict[test_key], dtype=np.float64)

        if not spec["support_ok"]:
            if len(val_scores) != 0 or len(test_scores) != 0:
                raise OperationalHoldError(f"{role}: expected empty score arrays on pre-fit support failure")
            role_records[role] = {
                "support_ok": False,
                "cal_admissible": False,
                "failure_reason": spec["failure_reason"],
                "n_val_oos": 0,
                "conf_thr": None,
                "platt_coef": None,
                "platt_intercept": None,
                "role_indices_sha256": spec["role_indices_sha256"],
                "trade_ts_sha256": sha256_bytes(b""),
                "n": 0,
                "wins": 0,
                "wr": None,
                "ev": None,
                "supported": False,
            }
            role_trades[role] = {"trade_ts": np.empty(0, dtype=np.int64), "trade_wins": np.empty(0, dtype=np.int64), "sel_local": np.empty(0, dtype=np.int64)}
            continue

        if len(val_scores) != len(spec["val_idx"]):
            raise OperationalHoldError(f"{role}: val score length {len(val_scores)} != {len(spec['val_idx'])}")

        c_meta = cal_by_role[role]
        # Verify conf_thr deterministically from val_scores (no fitting!)
        if bool(np.all(np.isfinite(val_scores))):
            exp_thr = float(np.quantile(np.abs(val_scores - 0.5), 1.0 - float(spec["cov"]), method="linear"))
        else:
            exp_thr = None

        cal_ok = bool(c_meta["cal_admissible"])
        if strict_verify_cal:
            if exp_thr is None:
                if cal_ok:
                    raise OperationalHoldError(f"{role}: nonfinite val scores cannot have cal_admissible=True")
            else:
                if c_meta["conf_thr"] is None or abs(float(c_meta["conf_thr"]) - exp_thr) > 1e-12:
                    raise OperationalHoldError(f"{role}: conf_thr mismatch ({c_meta['conf_thr']} != {exp_thr})")
            if cal_ok:
                pc = c_meta.get("platt_coef")
                pi = c_meta.get("platt_intercept")
                nv = c_meta.get("n_val_oos")
                if (
                    pc is None
                    or pi is None
                    or not math.isfinite(float(pc))
                    or float(pc) <= 0.0
                    or not math.isfinite(float(pi))
                    or int(nv) != len(spec["val_idx"])
                    or int(nv) < 500
                ):
                    raise OperationalHoldError(f"{role}: inadmissible persisted Platt parameters")

        # Under raw_top5 G4 mode, null roles always evaluate raw top-5% confidence regardless of Platt admissibility!
        if eff_g4_mode == G4_MODE_RAW_TOP5 and is_null:
            if len(test_scores) != len(spec["test_idx"]):
                raise OperationalHoldError(f"{role}: raw_top5 null test score length {len(test_scores)} != {len(spec['test_idx'])}")
            if exp_thr is not None:
                tr_raw = _select_role_trades_raw_top_cov(
                    test_scores,
                    ts_all[spec["test_idx"]],
                    spec["ret_arr"][spec["test_idx"]],
                    conf_thr=exp_thr,
                )
            else:
                tr_raw = {"sel_local": np.empty(0, dtype=np.int64), "trade_ts": np.empty(0, dtype=np.int64), "trade_wins": np.empty(0, dtype=np.int64), "trade_ts_sha256": sha256_bytes(b""), "n": 0, "wins": 0, "wr": None, "ev": None}
            supported_null = bool(tr_raw["n"] >= MIN_PATH_TRADES)
            role_records[role] = {
                "support_ok": True,
                "cal_admissible": cal_ok,
                "failure_reason": c_meta.get("failure_reason"),
                "n_val_oos": int(len(spec["val_idx"])),
                "conf_thr": exp_thr,
                "platt_coef": c_meta.get("platt_coef"),
                "platt_intercept": c_meta.get("platt_intercept"),
                "role_indices_sha256": spec["role_indices_sha256"],
                "trade_ts_sha256": tr_raw["trade_ts_sha256"],
                "n": tr_raw["n"],
                "wins": tr_raw["wins"],
                "wr": tr_raw["wr"],
                "ev": tr_raw["ev"],
                "supported": supported_null,
            }
            role_trades[role] = tr_raw
            raw_top5_trades[role] = tr_raw
            continue

        if not cal_ok:
            if eff_g4_mode == G4_MODE_RAW_TOP5 and is_real:
                if len(test_scores) != len(spec["test_idx"]):
                    raise OperationalHoldError(f"{role}: raw_top5 real test score length {len(test_scores)} != {len(spec['test_idx'])}")
                if exp_thr is not None:
                    raw_top5_trades[role] = _select_role_trades_raw_top_cov(
                        test_scores,
                        ts_all[spec["test_idx"]],
                        spec["ret_arr"][spec["test_idx"]],
                        conf_thr=exp_thr,
                    )
                else:
                    raw_top5_trades[role] = {
                        "sel_local": np.empty(0, dtype=np.int64),
                        "trade_ts": np.empty(0, dtype=np.int64),
                        "trade_wins": np.empty(0, dtype=np.int64),
                        "trade_ts_sha256": sha256_bytes(b""),
                        "n": 0,
                        "wins": 0,
                        "wr": None,
                        "ev": None,
                    }
            elif not is_final and len(test_scores) != 0:
                raise OperationalHoldError(f"{role}: expected empty test scores on calibration refusal")
            role_records[role] = {
                "support_ok": True,
                "cal_admissible": False,
                "failure_reason": c_meta.get("failure_reason") or "up_calibration_inadmissible",
                "n_val_oos": int(len(spec["val_idx"])),
                "conf_thr": exp_thr,
                "platt_coef": c_meta.get("platt_coef"),
                "platt_intercept": c_meta.get("platt_intercept"),
                "role_indices_sha256": spec["role_indices_sha256"],
                "trade_ts_sha256": sha256_bytes(b""),
                "n": 0,
                "wins": 0,
                "wr": None,
                "ev": None,
                "supported": False,
            }
            role_trades[role] = {"trade_ts": np.empty(0, dtype=np.int64), "trade_wins": np.empty(0, dtype=np.int64), "sel_local": np.empty(0, dtype=np.int64)}
            continue

        assert exp_thr is not None
        p_coef = float(c_meta["platt_coef"])
        p_int = float(c_meta["platt_intercept"])

        if is_final:
            n_h1 = len(spec["val_h1_idx"])
            tr_h1 = _select_role_trades(
                val_scores[:n_h1],
                ts_all[spec["val_h1_idx"]],
                spec["ret_arr"][spec["val_h1_idx"]],
                conf_thr=exp_thr,
                platt_coef=p_coef,
                platt_intercept=p_int,
            )
            tr_h2 = _select_role_trades(
                val_scores[n_h1:],
                ts_all[spec["val_h2_idx"]],
                spec["ret_arr"][spec["val_h2_idx"]],
                conf_thr=exp_thr,
                platt_coef=p_coef,
                platt_intercept=p_int,
            )
            ff_ok = bool(tr_h1["n"] >= 1 and tr_h2["n"] >= 1 and min(float(tr_h1["wr"]), float(tr_h2["wr"])) >= BREAKEVEN_WR)
            role_records[role] = {
                "support_ok": True,
                "cal_admissible": True,
                "failure_reason": None,
                "n_val_oos": int(len(spec["val_idx"])),
                "conf_thr": exp_thr,
                "platt_coef": p_coef,
                "platt_intercept": p_int,
                "role_indices_sha256": spec["role_indices_sha256"],
                "trade_ts_sha256": sha256_bytes(np.concatenate([tr_h1["trade_ts"], tr_h2["trade_ts"]]).tobytes()),
                "n": int(tr_h1["n"] + tr_h2["n"]),
                "wins": int(tr_h1["wins"] + tr_h2["wins"]),
                "wr": float((tr_h1["wins"] + tr_h2["wins"]) / (tr_h1["n"] + tr_h2["n"])) if (tr_h1["n"] + tr_h2["n"]) > 0 else None,
                "ev": float(((tr_h1["wins"] + tr_h2["wins"]) / (tr_h1["n"] + tr_h2["n"])) * QUOTED_PAYOUT_RATIO - 1.0)
                if (tr_h1["n"] + tr_h2["n"]) > 0
                else None,
                "supported": ff_ok,
                "val_first_half": {"n": tr_h1["n"], "wins": tr_h1["wins"], "wr": tr_h1["wr"], "ev": tr_h1["ev"]},
                "val_second_half": {"n": tr_h2["n"], "wins": tr_h2["wins"], "wr": tr_h2["wr"], "ev": tr_h2["ev"]},
                "passed": ff_ok,
            }
            role_trades[role] = {"val_first_half": tr_h1, "val_second_half": tr_h2}
        else:
            if len(test_scores) != len(spec["test_idx"]):
                raise OperationalHoldError(f"{role}: test score length {len(test_scores)} != {len(spec['test_idx'])}")
            tr = _select_role_trades(
                test_scores,
                ts_all[spec["test_idx"]],
                spec["ret_arr"][spec["test_idx"]],
                conf_thr=exp_thr,
                platt_coef=p_coef,
                platt_intercept=p_int,
            )
            if eff_g4_mode == G4_MODE_RAW_TOP5 and is_real:
                raw_top5_trades[role] = _select_role_trades_raw_top_cov(
                    test_scores,
                    ts_all[spec["test_idx"]],
                    spec["ret_arr"][spec["test_idx"]],
                    conf_thr=exp_thr,
                )
            min_sup = 50 if role in ("frozen_past", "incumbent_frozen_past") else MIN_PATH_TRADES
            supported = bool(tr["n"] >= min_sup)
            role_records[role] = {
                "support_ok": True,
                "cal_admissible": True,
                "failure_reason": None,
                "n_val_oos": int(len(spec["val_idx"])),
                "conf_thr": exp_thr,
                "platt_coef": p_coef,
                "platt_intercept": p_int,
                "role_indices_sha256": spec["role_indices_sha256"],
                "trade_ts_sha256": tr["trade_ts_sha256"],
                "n": tr["n"],
                "wins": tr["wins"],
                "wr": tr["wr"],
                "ev": tr["ev"],
                "supported": supported,
            }
            role_trades[role] = tr

    # 2. Aggregate raw validity & per-path moved_up_rate (strictly separated from G1 drift amendment!)
    elig_ret = ret_fwd_true[eligible_idx]
    elig_total = int(len(elig_ret))
    elig_ties = int(np.sum(elig_ret == 0.0))
    elig_moved = elig_ret[elig_ret != 0.0]
    if elig_total == 0 or len(elig_moved) == 0:
        moved_up_rate = None
        tie_rate = None
        agg_band_ok = False
    else:
        moved_up_rate = float(np.mean(elig_moved > 0.0))
        tie_rate = float(elig_ties / elig_total)
        agg_band_ok = bool(0.47 <= moved_up_rate <= 0.53 and 0.0 < tie_rate <= 0.03)

    real_specs = [s for s in role_specs if s["kind"] == "real"]
    g1_paths_export: list[dict[str, Any]] = []
    cert_paths_top: list[dict[str, Any]] = []
    path_moved_rates: list[float | None] = []
    counted_real_wrs: list[float] = []
    counted_real_roles: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]] = []

    for sp in real_specs:
        role = sp["role"]
        rec = role_records[role]
        tr = role_trades[role]
        p_ret = ret_fwd_true[sp["test_idx"]]
        p_mv = p_ret[p_ret != 0.0]
        p_mur = float(np.mean(p_mv > 0.0)) if len(p_mv) > 0 else None
        path_moved_rates.append(p_mur)
        g1_paths_export.append(
            {
                "path_id": sp["path_id"],
                "test_groups": list(sp["test_groups"]),
                "supported": bool(rec["supported"]),
                "n": int(rec["n"]),
                "wins": int(rec["wins"]),
                "wr": rec["wr"],
                "ev": rec["ev"],
                "path_moved_up_rate": p_mur,
                "path_years": PATH_YEARS,
                "trade_ts": [int(x) for x in tr["trade_ts"].tolist()],
                "trade_wins": [int(x) for x in tr["trade_wins"].tolist()],
            }
        )
        cert_paths_top.append(
            {
                "path_id": sp["path_id"],
                "bycov": {
                    "0.05": {
                        "UP": [int(rec["n"]), rec["wr"]],
                        "DOWN": [0, None],
                    }
                },
            }
        )
        if rec["supported"] and rec["wr"] is not None:
            counted_real_wrs.append(float(rec["wr"]))
            counted_real_roles.append((sp, rec, tr))

    paths_in_raw_tripwire = bool(
        len(path_moved_rates) == 15 and all(r is not None and 0.45 <= r <= 0.55 for r in path_moved_rates)
    )
    raw_strict_band_ok = bool(agg_band_ok and paths_in_raw_tripwire)

    paths_counted = len(counted_real_wrs)
    paths_clear_0541 = int(sum(w >= BREAKEVEN_WR for w in counted_real_wrs))
    paths_clear_0565 = int(sum(w >= DIAG_HIGH_WR for w in counted_real_wrs))
    g1_p10_wr = _linear_p10(counted_real_wrs)
    g1_mean_wr = float(np.mean(counted_real_wrs)) if counted_real_wrs else None

    # Forward folds (fold clears iff wr >= 0.541!)
    fwd_specs = [s for s in role_specs if s["kind"] == "fwd"]
    fwd_rows: list[dict[str, Any]] = []
    fwd_counted_evs: list[float] = []
    fwd_counted_wrs: list[float] = []
    fwd_clear_0541 = 0
    stitched_ts: list[np.ndarray] = []
    stitched_wins: list[np.ndarray] = []
    for sp in fwd_specs:
        rec = role_records[sp["role"]]
        tr = role_trades[sp["role"]]
        cleared = bool(rec["supported"] and rec["wr"] is not None and float(rec["wr"]) >= BREAKEVEN_WR)
        fwd_rows.append(
            {
                "role": sp["role"],
                "group": sp["group"],
                "supported": bool(rec["supported"]),
                "n": int(rec["n"]),
                "wins": int(rec["wins"]),
                "wr": rec["wr"],
                "ev": rec["ev"],
                "cleared_0541": cleared,
            }
        )
        if rec["supported"] and rec["wr"] is not None and rec["ev"] is not None:
            fwd_counted_wrs.append(float(rec["wr"]))
            fwd_counted_evs.append(float(rec["ev"]))
            if cleared:
                fwd_clear_0541 += 1
        if len(tr["trade_ts"]) > 0:
            stitched_ts.append(tr["trade_ts"])
            stitched_wins.append(tr["trade_wins"])

    fwd_counted = len(fwd_counted_evs)
    fwd_mean_wr = float(np.mean(fwd_counted_wrs)) if fwd_counted_wrs else None
    fwd_mean_ev = float(np.mean(fwd_counted_evs)) if fwd_counted_evs else None
    fwd_ok = bool(fwd_counted >= 3 and (fwd_clear_0541 / max(1, fwd_counted)) >= 0.50 and fwd_mean_ev is not None and fwd_mean_ev > 0.0)

    if stitched_ts:
        all_st_ts = np.concatenate(stitched_ts)
        all_st_w = np.concatenate(stitched_wins)
        m25 = all_st_ts < FROZEN_PAST_SPLIT_UTC
        m26 = all_st_ts >= FROZEN_PAST_SPLIT_UTC
        stitched_diag = {
            "2025": {
                "n": int(np.sum(m25)),
                "wins": int(np.sum(all_st_w[m25])),
                "wr": float(np.mean(all_st_w[m25])) if np.any(m25) else None,
                "ci95": _bootstrap_ci95_wr(all_st_w[m25]),
            },
            "2026_pre_aug": {
                "n": int(np.sum(m26)),
                "wins": int(np.sum(all_st_w[m26])),
                "wr": float(np.mean(all_st_w[m26])) if np.any(m26) else None,
                "ci95": _bootstrap_ci95_wr(all_st_w[m26]),
            },
        }
    else:
        stitched_diag = {
            "2025": {"n": 0, "wins": 0, "wr": None, "ci95": None},
            "2026_pre_aug": {"n": 0, "wins": 0, "wr": None, "ci95": None},
        }

    # frozen_past & incumbent_frozen_past
    fp_rec = role_records["frozen_past"]
    fp_passed = bool(fp_rec["supported"] and fp_rec["n"] >= 50 and fp_rec["ev"] is not None and float(fp_rec["ev"]) > 0.0)
    inc_rec = role_records["incumbent_frozen_past"]
    inc_passed = bool(inc_rec["supported"] and inc_rec["n"] >= 50 and inc_rec["ev"] is not None and float(inc_rec["ev"]) > 0.0)

    # 3. G2_placebo (NY-hour x rv_60m terciles weighted by selected trades)
    g2_excess_list: list[float] = []
    g2_cuts_ok = True
    g2_paths_detail: list[dict[str, Any]] = []
    dt_ny_all = pd.to_datetime(ts_all, unit="s", utc=True).tz_convert(NY_TZ)
    ny_hour_all = dt_ny_all.hour.to_numpy(dtype=np.int64)

    for sp, rec, tr in counted_real_roles:
        t_idx = sp["test_idx"]
        v_rv = rv60_all[t_idx]
        fin_rv = v_rv[np.isfinite(v_rv)]
        if len(fin_rv) < 3:
            g2_cuts_ok = False
            g2_paths_detail.append({"path_id": sp["path_id"], "valid_cuts": False, "excess": None})
            continue
        q1, q2 = [float(x) for x in np.nanquantile(fin_rv, [1.0 / 3.0, 2.0 / 3.0])]
        if not (math.isfinite(q1) and math.isfinite(q2) and q1 < q2):
            g2_cuts_ok = False
            g2_paths_detail.append({"path_id": sp["path_id"], "valid_cuts": False, "excess": None})
            continue

        def _vol_bin(rv_vals: np.ndarray) -> np.ndarray:
            out_b = np.full(len(rv_vals), -1, dtype=np.int64)
            ok = np.isfinite(rv_vals)
            out_b[ok & (rv_vals <= q1)] = 0
            out_b[ok & (rv_vals > q1) & (rv_vals <= q2)] = 1
            out_b[ok & (rv_vals > q2)] = 2
            return out_b

        hrs_all = ny_hour_all[t_idx]
        vb_all = _vol_bin(v_rv)
        wins_all = (ret_fwd_true[t_idx] > 0.0).astype(np.float64)
        cell_wr: dict[tuple[int, int], float] = {}
        for h_val in set(hrs_all.tolist()):
            for b_val in (-1, 0, 1, 2):
                m_c = (hrs_all == h_val) & (vb_all == b_val)
                if np.any(m_c):
                    cell_wr[(int(h_val), int(b_val))] = float(np.mean(wins_all[m_c]))

        sel_local = tr["sel_local"]
        sel_hrs = hrs_all[sel_local]
        sel_vb = vb_all[sel_local]
        matched_base = np.array([cell_wr[(int(h), int(b))] for h, b in zip(sel_hrs, sel_vb)], dtype=np.float64)
        ex_p = float(rec["wr"]) - float(np.mean(matched_base))
        g2_excess_list.append(ex_p)
        g2_paths_detail.append({"path_id": sp["path_id"], "valid_cuts": True, "q1": q1, "q2": q2, "excess": ex_p})

    g2_paths_positive = int(sum(x > 0.0 for x in g2_excess_list))
    g2_mean_excess = float(np.mean(g2_excess_list)) if g2_excess_list else None
    g2_passed = bool(
        paths_counted >= 12
        and g2_cuts_ok
        and len(g2_excess_list) == paths_counted
        and g2_paths_positive >= 12
        and g2_mean_excess is not None
        and g2_mean_excess >= 0.02
    )
    g2_obj = {
        "passed": g2_passed,
        "verdict": "SKILL BEYOND DRIFT TIMING" if g2_passed else "PLACEBO_FAIL",
        "paths_counted": paths_counted,
        "paths_positive": g2_paths_positive,
        "all_cuts_valid": g2_cuts_ok,
        "mean_excess_hour_vol": g2_mean_excess,
        "paths": g2_paths_detail,
    }

    # 4. G3_entry_delay (Owner Decision #1 — de-overlapped <=2s freshness subgroup baseline!)
    chg_t = tick_ts[np.r_[True, np.diff(tick_px) != 0]] if len(tick_ts) > 0 else np.empty(0, dtype=np.int64)

    def _prior_age(dec_ts_arr: np.ndarray, ref_ts: np.ndarray) -> np.ndarray:
        if len(ref_ts) == 0 or len(dec_ts_arr) == 0:
            return np.full(len(dec_ts_arr), np.nan, dtype=np.float64)
        k = np.searchsorted(ref_ts, dec_ts_arr, side="left") - 1
        out_a = np.full(len(dec_ts_arr), np.nan, dtype=np.float64)
        ok = k >= 0
        if np.any(ok):
            out_a[ok] = (dec_ts_arr[ok] - ref_ts[k[ok]]).astype(np.float64)
        return out_a

    by_delay_wrs: dict[int, list[float]] = {d: [] for d in DELAYS_EXTRA_S}
    sub_ex: dict[str, list[float]] = {"fresh_quote_le2s": [], "fresh_midchg_le2s": [], "older_midchg_gt2s": []}

    for sp, rec, tr in counted_real_roles:
        sel_ts = tr["trade_ts"]
        n_tr = len(sel_ts)
        for d in DELAYS_EXTRA_S:
            r_d, v_d, _, _ = settle_with_intervals(sel_ts, tick_ts, tick_px, horizon_s=HORIZON_S, delay_s=d, tol_s=TOL_S)
            # Invalid delayed outcomes charged 0 in d=0 denominator n
            wins_d = int(np.sum(v_d & (r_d > 0.0)))
            by_delay_wrs[d].append(float(wins_d / n_tr))

        # Freshness subgroups using strictly prior ticks (d=0 outcomes)
        t_idx = sp["test_idx"]
        elig_p_ts = ts_all[t_idx]
        elig_p_win = ret_fwd_true[t_idx] > 0.0
        elig_q_age = _prior_age(elig_p_ts, tick_ts)
        elig_m_age = _prior_age(elig_p_ts, chg_t)

        sel_p_win = tr["trade_wins"].astype(bool)
        sel_q_age = _prior_age(sel_ts, tick_ts)
        sel_m_age = _prior_age(sel_ts, chg_t)

        sub_masks = {
            "fresh_quote_le2s": (np.isfinite(sel_q_age) & (sel_q_age <= 2.0), np.isfinite(elig_q_age) & (elig_q_age <= 2.0)),
            "fresh_midchg_le2s": (np.isfinite(sel_m_age) & (sel_m_age <= 2.0), np.isfinite(elig_m_age) & (elig_m_age <= 2.0)),
            "older_midchg_gt2s": (np.isfinite(sel_m_age) & (sel_m_age > 2.0), np.isfinite(elig_m_age) & (elig_m_age > 2.0)),
        }
        for sname, (sm, em) in sub_masks.items():
            n_sel_sub = int(np.sum(sm))
            base_idx = nonoverlap_chrono(elig_p_ts, em, gap=HORIZON_S)
            n_elig_sub = int(len(base_idx))
            if n_sel_sub >= 1 and n_elig_sub >= 1:
                wr_sel = float(np.mean(sel_p_win[sm]))
                wr_base = float(np.mean(elig_p_win[base_idx]))
                sub_ex[sname].append(wr_sel - wr_base)

    delay_summary: dict[str, Any] = {}
    for d in DELAYS_EXTRA_S:
        vals = by_delay_wrs[d]
        delay_summary[str(d)] = {
            "paths_counted": len(vals),
            "paths_clear": int(sum(w >= BREAKEVEN_WR for w in vals)),
            "p10_wr": _linear_p10(vals),
            "mean_wr": float(np.mean(vals)) if vals else None,
        }
    d4_s = delay_summary["4"]
    fresh_summary: dict[str, Any] = {}
    for sname, ex_vals in sub_ex.items():
        fresh_summary[sname] = {
            "included_paths": len(ex_vals),
            "excess": float(np.mean(ex_vals)) if ex_vals else None,
        }
    g3_delay_ok = bool(
        paths_counted >= 12
        and d4_s["paths_clear"] >= 12
        and d4_s["p10_wr"] is not None
        and float(d4_s["p10_wr"]) >= BREAKEVEN_WR
    )
    g3_fresh_ok = bool(
        fresh_summary["fresh_quote_le2s"]["included_paths"] >= 1
        and fresh_summary["fresh_quote_le2s"]["excess"] is not None
        and float(fresh_summary["fresh_quote_le2s"]["excess"]) > 0.0
        and fresh_summary["fresh_midchg_le2s"]["included_paths"] >= 1
        and fresh_summary["fresh_midchg_le2s"]["excess"] is not None
        and float(fresh_summary["fresh_midchg_le2s"]["excess"]) > 0.0
    )
    g3_passed = bool(g3_delay_ok and g3_fresh_ok)
    g3_obj = {
        "passed": g3_passed,
        "verdict_delay": "OPERATIONALLY ROBUST" if g3_delay_ok else "FRAGILE",
        "verdict_staleness": "NOT A STALE-FEED ARTIFACT" if g3_fresh_ok else "FRESHNESS_SUBGROUP_FAIL",
        "d4_paths_clear": int(d4_s["paths_clear"]),
        "d4_p10_wr": d4_s["p10_wr"],
        "summary_by_delay": delay_summary,
        "freshness_subgroups": fresh_summary,
    }

    # 5. G4_sign_null (CPCV-only comparator; DESCRIPTIVE_RANK_DIAGNOSTIC_* labels ONLY)
    null_shift_summaries: list[dict[str, Any]] = []
    readable_null_p10s: list[float] = []
    null_passes = 0
    n_null_ge_real = 0
    n_null_ge_real_raw = 0
    n_null_ge_real_platt = 0

    if eff_g4_mode == G4_MODE_RAW_TOP5:
        real_raw_wrs = [
            float(raw_top5_trades[s["role"]]["wr"])
            for s in real_specs
            if s["role"] in raw_top5_trades
            and int(raw_top5_trades[s["role"]]["n"]) >= MIN_PATH_TRADES
            and raw_top5_trades[s["role"]]["wr"] is not None
        ]
        real_raw_paths_counted = len(real_raw_wrs)
        real_raw_p10_wr = _linear_p10(real_raw_wrs)
    else:
        real_raw_paths_counted = None
        real_raw_p10_wr = None

    for w in NULL_SHIFTS_WEEKS:
        w_specs = [s for s in role_specs if s["kind"] == "null" and s["week_shift"] == w]
        w_wrs = [
            float(role_records[s["role"]]["wr"])
            for s in w_specs
            if role_records[s["role"]]["supported"] and role_records[s["role"]]["wr"] is not None
        ]
        pc_w = len(w_wrs)
        clr_w = int(sum(x >= BREAKEVEN_WR for x in w_wrs))
        p10_w = _linear_p10(w_wrs)
        readable_w = pc_w >= 12
        passed_like_real_w = bool(
            readable_w and clr_w >= 12 and (clr_w / max(1, pc_w)) >= 0.80 and p10_w is not None and p10_w >= BREAKEVEN_WR
        )
        if readable_w and p10_w is not None:
            readable_null_p10s.append(float(p10_w))
            if g1_p10_wr is not None and float(p10_w) >= float(g1_p10_wr):
                n_null_ge_real_platt += 1
            if eff_g4_mode == G4_MODE_RAW_TOP5:
                if real_raw_p10_wr is not None and float(p10_w) >= float(real_raw_p10_wr):
                    n_null_ge_real_raw += 1
            else:
                if g1_p10_wr is not None and float(p10_w) >= float(g1_p10_wr):
                    n_null_ge_real += 1
        if passed_like_real_w:
            null_passes += 1
        null_shift_summaries.append(
            {
                "week_shift": w,
                "paths_counted": pc_w,
                "paths_clear_0541": clr_w,
                "null_p10_wr": p10_w,
                "readable": readable_w,
                "null_passed_like_real": passed_like_real_w,
            }
        )

    readable_null_shifts = len(readable_null_p10s)
    null_max_p10 = float(max(readable_null_p10s)) if readable_null_p10s else None
    if eff_g4_mode == G4_MODE_RAW_TOP5:
        n_null_ge_real = max(n_null_ge_real_raw, n_null_ge_real_platt)
        g4_passed = bool(
            g1_p10_wr is not None
            and real_raw_paths_counted is not None
            and real_raw_paths_counted >= 12
            and real_raw_p10_wr is not None
            and readable_null_shifts >= 12
            and null_passes <= 1
            and n_null_ge_real_raw <= 1
            and n_null_ge_real_platt <= 1
        )
        g4_obj: dict[str, Any] = {
            "passed": g4_passed,
            "verdict": "DESCRIPTIVE_RANK_DIAGNOSTIC_PASS" if g4_passed else "DESCRIPTIVE_RANK_DIAGNOSTIC_FAIL",
            "g4_selection_mode": "raw_top5_apples_to_apples",
            "real_raw_top5_paths_counted": real_raw_paths_counted,
            "real_raw_top5_p10_wr": real_raw_p10_wr,
            "readable_null_shifts": readable_null_shifts,
            "null_passes": null_passes,
            "n_null_ge_real_raw": n_null_ge_real_raw,
            "n_null_ge_real_platt": n_null_ge_real_platt,
            "n_null_ge_real": n_null_ge_real,
            "null_max_p10": null_max_p10,
            "real_p10_rank": f"{1 + n_null_ge_real}/{readable_null_shifts + 1}" if (g1_p10_wr is not None and real_raw_p10_wr is not None) else None,
            "shifts": null_shift_summaries,
        }
    else:
        g4_passed = bool(
            g1_p10_wr is not None and readable_null_shifts >= 12 and null_passes <= 1 and n_null_ge_real <= 1
        )
        g4_obj = {
            "passed": g4_passed,
            "verdict": "DESCRIPTIVE_RANK_DIAGNOSTIC_PASS" if g4_passed else "DESCRIPTIVE_RANK_DIAGNOSTIC_FAIL",
            "readable_null_shifts": readable_null_shifts,
            "null_passes": null_passes,
            "n_null_ge_real": n_null_ge_real,
            "null_max_p10": null_max_p10,
            "real_p10_rank": f"{1 + n_null_ge_real}/{readable_null_shifts + 1}" if g1_p10_wr is not None else None,
            "shifts": null_shift_summaries,
        }

    # 6. Drift amendment & G1_cpcv aggregate
    max_6pair_frac = Fraction(int(peer_tie_info["max_ties"]), int(peer_tie_info["max_total"]))
    usdjpy_tie_frac = Fraction(elig_ties, elig_total) if elig_total > 0 else None
    paths_in_amended_band = bool(
        len(path_moved_rates) == 15 and all(r is not None and 0.43 <= r <= 0.57 for r in path_moved_rates)
    )
    owner_dec1_amended_ok = bool(
        PAIR == "USDJPY"
        and HORIZON_MIN == 30
        and moved_up_rate is not None
        and 0.45 <= moved_up_rate <= 0.55
        and tie_rate is not None
        and 0.0 < tie_rate <= 0.02
        and usdjpy_tie_frac is not None
        and usdjpy_tie_frac <= max_6pair_frac
        and paths_in_amended_band
        and g2_passed
        and g4_passed
        and fp_passed
    )
    drift_check_passed = bool(raw_strict_band_ok or owner_dec1_amended_ok)
    cpcv_path_test_passed = bool(
        paths_counted >= 12
        and paths_clear_0541 >= 12
        and (paths_clear_0541 / max(1, paths_counted)) >= 0.80
        and g1_p10_wr is not None
        and g1_p10_wr >= BREAKEVEN_WR
    )
    g1_passed = bool(cpcv_path_test_passed and fwd_ok and fp_passed and drift_check_passed)
    g1_obj = {
        "passed": g1_passed,
        "cpcv_path_test_passed": cpcv_path_test_passed,
        "paths_total": 15,
        "paths_counted": paths_counted,
        "paths_clear_0541": paths_clear_0541,
        "paths_clear_0565": paths_clear_0565,
        "p10_wr": g1_p10_wr,
        "mean_wr": g1_mean_wr,
        "forward_counted": fwd_counted,
        "forward_clear_0541": fwd_clear_0541,
        "forward_mean_wr": fwd_mean_wr,
        "forward_mean_ev": fwd_mean_ev,
        "forward_passed": fwd_ok,
        "forward_folds": fwd_rows,
        "stitched_forward_diagnostic": stitched_diag,
        "frozen_past_passed": fp_passed,
        "drift_amendment_used": bool(not raw_strict_band_ok and owner_dec1_amended_ok),
        "paths": g1_paths_export,
    }

    # 7. G5_payout (Fixed-Ratio R = 1.82 DEV Sensitivity, Explicit Unavailable Branch)
    if paths_counted == 0 or g1_p10_wr is None:
        g5_passed = False
        g5_obj = {
            "passed": False,
            "classification": "fixed_ratio_dev_sensitivity_only_not_decision5",
            "verdict": "FIXED_RATIO_SENSITIVITY_FAIL",
            "quoted_payout_ratio": QUOTED_PAYOUT_RATIO,
            "paths_counted": 0,
            "paths_ev_positive": 0,
            "p10_ev": None,
            "mean_ev": None,
            "f_quarter_kelly": None,
            "p10_ann_log_growth": None,
            "p10_net_usd_1000": None,
            "est_trades_per_year": 0,
            "payout_gate_passed": False,
        }
    else:
        evs_p: list[float] = []
        g_paths: list[float] = []
        net_paths: list[float] = []
        ns_p: list[int] = []
        f_q = 0.25 * max(0.0, (float(g1_p10_wr) * QUOTED_PAYOUT_RATIO - 1.0) / (QUOTED_PAYOUT_RATIO - 1.0))
        for _, rec, _ in counted_real_roles:
            n_p = int(rec["n"])
            w_p = int(rec["wins"])
            evs_p.append((w_p / n_p) * QUOTED_PAYOUT_RATIO - 1.0)
            ns_p.append(n_p)
            g_p = (w_p * math.log1p(f_q * (QUOTED_PAYOUT_RATIO - 1.0)) + (n_p - w_p) * math.log1p(-f_q)) / PATH_YEARS
            g_paths.append(g_p)
            net_paths.append(_decimal_ann_net_usd_1000(w_p, n_p, f_q, QUOTED_PAYOUT_RATIO, PATH_YEARS))
        paths_ev_pos = int(sum(x > 0.0 for x in evs_p))
        p10_ev = _linear_p10(evs_p)
        mean_ev = float(np.mean(evs_p))
        p10_g = _linear_p10(g_paths)
        p10_net = _linear_p10(net_paths)
        est_tpy = int(round(float(np.mean(ns_p)) / PATH_YEARS))
        pg_dec = evaluate_sota_payout_gate(
            {"ask_price": "1.0", "payout": "1.82"},
            effective_floor=float(g1_p10_wr) - EV_MARGIN,
            absolute_breakeven_ceiling=0.60,
            calibrated_win_prob=None,
            ev_margin=EV_MARGIN,
            min_payout_ratio=MIN_PAYOUT_RATIO,
            require_calibrated_ev=False,
        )
        g5_passed = bool(
            paths_counted >= 12 and paths_ev_pos >= 12 and p10_ev is not None and p10_ev > 0.0 and pg_dec.passed
        )
        g5_obj = {
            "passed": g5_passed,
            "classification": "fixed_ratio_dev_sensitivity_only_not_decision5",
            "verdict": "FIXED_RATIO_SENSITIVITY_PASS" if g5_passed else "FIXED_RATIO_SENSITIVITY_FAIL",
            "quoted_payout_ratio": QUOTED_PAYOUT_RATIO,
            "paths_counted": paths_counted,
            "paths_ev_positive": paths_ev_pos,
            "p10_ev": p10_ev,
            "mean_ev": mean_ev,
            "f_quarter_kelly": f_q,
            "p10_ann_log_growth": p10_g,
            "p10_net_usd_1000": p10_net,
            "est_trades_per_year": est_tpy,
            "payout_gate_passed": bool(pg_dec.passed),
            "payout_gate_reason": pg_dec.reason,
        }

    # 8. Final verdict & failed_predicates
    ff_rec = role_records["final_fit"]
    ff_passed = bool(ff_rec.get("passed", False))
    failed_predicates: list[str] = []
    if not g1_passed:
        failed_predicates.append("G1_cpcv")
    if not g2_passed:
        failed_predicates.append("G2_placebo")
    if not g3_passed:
        failed_predicates.append("G3_entry_delay")
    if not g4_passed:
        failed_predicates.append("G4_sign_null")
    if not g5_passed:
        failed_predicates.append("G5_payout")
    if not ff_passed:
        failed_predicates.append("final_fit")

    terminal_verdict = "PASS" if not failed_predicates else "KILL-STACK"

    # 9. Raw scripts/certificate.py strict_certificate interop (strictly from raw CPCV figures + raw validity!)
    raw_up_cert = bool(raw_strict_band_ok and paths_counted >= 12 and paths_clear_0541 >= 12 and g1_p10_wr is not None and g1_p10_wr >= BREAKEVEN_WR)
    validity_obj = {
        "VALID": bool(raw_strict_band_ok),
        "in_session_up_rate_in_band": bool(agg_band_ok),
        "paths_up_rate_in_tripwire": bool(paths_in_raw_tripwire),
        "moved_up_rate": moved_up_rate,
        "tie_rate": tie_rate,
        "eligible_rows": elig_total,
        "eligible_ties": elig_ties,
    }
    verdict_goal = {
        "UP": {
            "CERTIFIED": raw_up_cert,
            "paths_counted": paths_counted,
            "paths_clear": paths_clear_0541,
            "p10": g1_p10_wr,
        },
        "DOWN": {
            "CERTIFIED": False,
            "paths_counted": 0,
            "paths_clear": 0,
            "p10": None,
        },
    }

    dossier: dict[str, Any] = {
        "schema": DOSSIER_SCHEMA,
        "book_id": BOOK_ID,
        "arm_name": arm_name,
        "pair": PAIR,
        "horizon_min": HORIZON_MIN,
        "side": "UP",
        "contract_family": "risefall",
        "model_variant": "core66_nystroem5_lgb_stack_v1",
        "cov": COV,
        "operating_cov": COV,
        "quoted_payout_ratio": QUOTED_PAYOUT_RATIO,
        "cutoff_utc": CUTOFF_UTC,
        "shared_calendar_aligned": True,
        "shared_group_bounds_utc": list(SHARED_GROUP_BOUNDS_UTC),
        "path_years": PATH_YEARS,
        "feature_count": len(fcols),
        "feature_cols": list(fcols),
        "tick_substrate": "deriv_1s_ticks_no_forward_fill",
        "local_source_closure_sha256": local_source_closure_sha256,
        "preregistration_sha256": preregistration_sha256,
        "preregistration_payload": preregistration_payload,
        "historical_comparator": {
            "path": HISTORICAL_COMPARATOR_REL,
            "sha256": HISTORICAL_COMPARATOR_SHA256,
        },
        "moved_up_rate": moved_up_rate,
        "tie_rate": tie_rate,
        "in_strict_047_053_moved_band": bool(raw_strict_band_ok),
        "owner_decision1_amendment": {
            "qualified": bool(owner_dec1_amended_ok),
            "usdjpy_elig_ties": elig_ties,
            "usdjpy_elig_total": elig_total,
            "peer_6pair_tie_comparator": peer_tie_info,
        },
        "drift_check_passed": drift_check_passed,
        "validity": validity_obj,
        "paths": cert_paths_top,
        "verdict_goal": verdict_goal,
        "paths_clear_0565": paths_clear_0565,
        "paths_counted": paths_counted,
        "decision5_per_trade_30m_payout_reading": "unavailable",
        "G1_cpcv": g1_obj,
        "G2_placebo": g2_obj,
        "G3_entry_delay": g3_obj,
        "G4_sign_null": g4_obj,
        "G5_payout": g5_obj,
        "frozen_2025_to_2026_oos": {
            "passed": fp_passed,
            "n": fp_rec["n"],
            "wins": fp_rec["wins"],
            "wr": fp_rec["wr"],
            "ev": fp_rec["ev"],
            "2024_status": "unavailable_in_deriv_archive",
        },
        "incumbent_frozen_past_comparison": {
            "passed": inc_passed,
            "n": inc_rec["n"],
            "wins": inc_rec["wins"],
            "wr": inc_rec["wr"],
            "ev": inc_rec["ev"],
        },
        "final_fit": ff_rec,
        "roles": role_records,
        "all_5_gates_passed": bool(g1_passed and g2_passed and g3_passed and g4_passed and g5_passed),
        "verdict": terminal_verdict,
        "failed_predicates": failed_predicates,
    }
    if final_fit_expected_identity is not None:
        dossier["final_fit_expected_content_id"] = final_fit_expected_identity["content_id"]
        dossier["final_fit_expected_artifacts"] = final_fit_expected_identity["artifacts"]

    # Evaluate raw strict_certificate, canonicalizing NaN -> None, and verify _reread_result == (2, [], [], [])!
    sc_up = cert_mod.strict_certificate(dossier, "UP")
    sc_up_clean = {k: (None if isinstance(v, float) and not math.isfinite(v) else v) for k, v in sc_up.items()}
    dossier["strict_certificate_up"] = sc_up_clean
    exact, differ, unreadable, legacy_only = cert_mod._reread_result(arm_names["dossier"], dossier)
    if (exact, differ, unreadable, legacy_only) != (2, [], [], []):
        raise OperationalHoldError(
            f"certificate._reread_result mismatch: exact={exact} differ={differ} unreadable={unreadable} legacy={legacy_only}"
        )
    return dossier


# ---------------------------------------------------------------------------
# Fit Ledger (Exact 4 Keys), Uncompressed 523-Key NPZ & Custody Verification
# ---------------------------------------------------------------------------


def validate_fit_ledger_obj(
    ledger: dict[str, Any],
    *,
    expected_prereg_sha: str,
    allow_builder_attempt: bool = False,
) -> tuple[int, int]:
    if not isinstance(ledger, dict):
        raise OperationalHoldError("fit ledger is not a JSON object")
    if set(ledger.keys()) != {"schema", "preregistration_sha256", "max_fit_units", "attempts"}:
        raise OperationalHoldError(f"fit ledger must have exact 4 keys, got {sorted(ledger.keys())}")
    if ledger["schema"] != FIT_LEDGER_SCHEMA:
        raise OperationalHoldError(f"invalid fit ledger schema {ledger['schema']!r}")
    if ledger["preregistration_sha256"] != expected_prereg_sha:
        raise OperationalHoldError("fit ledger preregistration_sha256 mismatch")
    if ledger["max_fit_units"] != MAX_FIT_UNITS:
        raise OperationalHoldError(f"fit ledger max_fit_units {ledger['max_fit_units']!r} != {MAX_FIT_UNITS}")
    attempts = ledger["attempts"]
    if not isinstance(attempts, list) or len(attempts) > MAX_FIT_UNITS:
        raise OperationalHoldError(f"invalid fit ledger attempts list (len={len(attempts) if isinstance(attempts, list) else 'bad'})")

    cert_n = 0
    bld_n = 0
    for idx, att in enumerate(attempts, 1):
        if not isinstance(att, dict) or set(att.keys()) != {"attempt_id", "stage", "role", "preregistration_sha256", "started_utc"}:
            raise OperationalHoldError(f"malformed attempt entry #{idx}")
        if att["attempt_id"] != idx:
            raise OperationalHoldError(f"non-contiguous attempt_id {att['attempt_id']} != {idx}")
        if att["preregistration_sha256"] != expected_prereg_sha:
            raise OperationalHoldError(f"attempt #{idx} preregistration_sha256 mismatch")
        st = att["stage"]
        if st == "certifier":
            cert_n += 1
        elif st == "builder":
            if att["role"] != "builder_final_fit":
                raise OperationalHoldError(f"attempt #{idx} builder role must be 'builder_final_fit', got {att['role']!r}")
            bld_n += 1
        else:
            raise OperationalHoldError(f"unknown attempt stage {st!r}")
    if cert_n > MAX_CERTIFIER_ATTEMPTS:
        raise OperationalHoldError(f"certifier attempts {cert_n} > {MAX_CERTIFIER_ATTEMPTS}")
    if bld_n > (MAX_BUILDER_ATTEMPTS if allow_builder_attempt else 0):
        raise OperationalHoldError(f"builder attempts {bld_n} exceed allowed limit")
    return cert_n, bld_n


def init_or_load_fit_ledger(ledger_path: Path, prereg_sha: str, *, allow_builder_attempt: bool = False) -> dict[str, Any]:
    if ledger_path.exists():
        obj = strict_json_loads(ledger_path.read_bytes())
        validate_fit_ledger_obj(obj, expected_prereg_sha=prereg_sha, allow_builder_attempt=allow_builder_attempt)
        return obj
    obj = {
        "schema": FIT_LEDGER_SCHEMA,
        "preregistration_sha256": prereg_sha,
        "max_fit_units": MAX_FIT_UNITS,
        "attempts": [],
    }
    atomic_write_file_bytes(ledger_path, canonical_json_bytes(obj), sync_parent=True)
    return obj


def debit_fit_ledger(
    ledger_path: Path,
    lock_path: Path,
    *,
    prereg_sha: str,
    stage: str,
    role: str,
) -> dict[str, Any]:
    """Atomically debit 1 attempt under LOCK_EX|LOCK_NB BEFORE any trainer invocation."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "a+", encoding="utf-8") as lfh:
        try:
            fcntl.flock(lfh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise OperationalHoldError(f"fit lock busy: {lock_path}") from exc
        ledger = init_or_load_fit_ledger(ledger_path, prereg_sha, allow_builder_attempt=(stage == "builder"))
        cert_n, bld_n = validate_fit_ledger_obj(ledger, expected_prereg_sha=prereg_sha, allow_builder_attempt=(stage == "builder"))
        if stage == "certifier" and cert_n >= MAX_CERTIFIER_ATTEMPTS:
            raise OperationalHoldError(f"certifier attempt budget exhausted ({cert_n}/{MAX_CERTIFIER_ATTEMPTS})")
        if stage == "builder" and bld_n >= MAX_BUILDER_ATTEMPTS:
            raise OperationalHoldError(f"builder attempt budget exhausted ({bld_n}/{MAX_BUILDER_ATTEMPTS})")
        if len(ledger["attempts"]) >= MAX_FIT_UNITS:
            raise OperationalHoldError(f"total fit budget exhausted ({len(ledger['attempts'])}/{MAX_FIT_UNITS})")
        attempt = {
            "attempt_id": len(ledger["attempts"]) + 1,
            "stage": stage,
            "role": role,
            "preregistration_sha256": prereg_sha,
            "started_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        ledger["attempts"].append(attempt)
        atomic_write_file_bytes(ledger_path, canonical_json_bytes(ledger), sync_parent=True)
        return attempt


def write_deterministic_uncompressed_npz(dst_path: Path, arrays_ordered: list[tuple[str, np.ndarray]]) -> bytes:
    """Write deterministic uncompressed (ZIP_STORED) 1D float64 .npz preserving exact key order."""
    out = io.BytesIO()
    with zipfile.ZipFile(out, mode="w", compression=zipfile.ZIP_STORED) as zf:
        for key, arr in arrays_ordered:
            a = np.asarray(arr, dtype=np.float64)
            if a.ndim != 1:
                raise OperationalHoldError(f"scores.npz array {key!r} must be 1D, got shape {a.shape}")
            buf = io.BytesIO()
            npfmt.write_array(buf, a, allow_pickle=False)
            zi = zipfile.ZipInfo(filename=f"{key}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            zi.compress_type = zipfile.ZIP_STORED
            zi.external_attr = 0o600 << 16
            zf.writestr(zi, buf.getvalue())
    raw = out.getvalue()
    atomic_write_file_bytes(dst_path, raw, sync_parent=True)
    return raw


def load_verified_scores_npz(npz_path: Path) -> tuple[dict[str, np.ndarray], int, str]:
    raw = npz_path.read_bytes()
    expected_keys = canonical_scores_npz_keys()
    with np.load(io.BytesIO(raw), allow_pickle=False) as z:
        if list(z.files) != expected_keys:
            raise OperationalHoldError(
                f"scores.npz key order/set mismatch (len={len(z.files)} vs {len(expected_keys)})"
            )
        out: dict[str, np.ndarray] = {}
        for k in expected_keys:
            a = z[k]
            if a.ndim != 1 or a.dtype != np.float64:
                raise OperationalHoldError(f"scores.npz[{k!r}] must be 1D float64, got {a.shape} {a.dtype}")
            out[k] = a.copy()
    return out, len(raw), sha256_bytes(raw)


def parse_progress_builder_witnesses(progress_md: Path) -> list[dict[str, str]]:
    """Parse `Builder fit debit` witness lines from PROGRESS.md."""
    if not progress_md.is_file():
        return []
    text = progress_md.read_text(encoding="utf-8")
    out: list[dict[str, str]] = []
    pat = re.compile(
        r"Builder fit debit:\s*arm=(\S+)\s+preregistration_sha256=([0-9a-f]{64})\s+"
        r"receipt_sha256=([0-9a-f]{64})\s+ledger_sha256=([0-9a-f]{64})\s+attempt_id=(\d+)"
    )
    for line in text.splitlines():
        m = pat.search(line)
        if m:
            out.append(
                {
                    "line": line.strip(),
                    "arm": m.group(1),
                    "preregistration_sha256": m.group(2),
                    "receipt_sha256": m.group(3),
                    "ledger_sha256": m.group(4),
                    "attempt_id": m.group(5),
                }
            )
    return out


def verify_bundle(
    *,
    expected_receipt_sha256: str,
    bundle_dir: Path | None = None,
    repo_root: Path = REPO_ROOT,
    expected_verdict: str | None = None,
    allow_builder_suffix: bool = False,
    progress_md: Path | None = None,
    dev_dir: Path = DEFAULT_DEV_DATASET_DIR,
    ticks_root: Path = DEFAULT_TICKS_ROOT,
    pinned_parquets: dict[str, tuple[int, str]] | None = None,
    expected_manifest_sha256: str | None = DEV_MANIFEST_SHA256,
    skip_closure_check: bool = False,
    arm_name: str = ARM_NAME,
    _injected_data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Strictly read-only, ZERO-FIT authentication of certifier bundle (+ optional builder suffix)."""
    if not skip_closure_check:
        verify_source_closure_and_loaded_modules(repo_root)

    names = artifact_names_for_arm(arm_name)
    if bundle_dir is not None:
        bdir = Path(bundle_dir)
        receipt_path = bdir / names["receipt"]
        dossier_path = bdir / names["dossier"]
        scores_path = bdir / names["scores"]
        ledger_path = bdir / names["ledger"]
        suffix_path = bdir / names["builder_suffix"]
    else:
        receipt_path = repo_root / "logs" / names["receipt"]
        dossier_path = repo_root / "results" / "json" / names["dossier"]
        scores_path = repo_root / "logs" / names["scores"]
        ledger_path = repo_root / "logs" / names["ledger"]
        suffix_path = None

    for req_p in (receipt_path, dossier_path, scores_path, ledger_path):
        if not req_p.is_file():
            raise OperationalHoldError(f"missing bundle artifact: {req_p}")

    receipt_raw = receipt_path.read_bytes()
    actual_rcpt_sha = sha256_bytes(receipt_raw)
    if actual_rcpt_sha != expected_receipt_sha256:
        raise OperationalHoldError(f"receipt SHA-256 mismatch: {actual_rcpt_sha} != {expected_receipt_sha256}")
    receipt = strict_json_loads(receipt_raw)
    if receipt.get("schema") != RECEIPT_SCHEMA:
        raise OperationalHoldError(f"invalid receipt schema: {receipt.get('schema')!r}")
    rcpt_arm = str(receipt.get("arm_name") or ARM_NAME)
    if rcpt_arm != arm_name:
        raise OperationalHoldError(f"receipt arm_name mismatch: {rcpt_arm!r} != {arm_name!r}")

    dossier_raw = dossier_path.read_bytes()
    actual_dos_sha = sha256_bytes(dossier_raw)
    if actual_dos_sha != receipt.get("dossier_sha256"):
        raise OperationalHoldError(f"dossier SHA-256 mismatch: {actual_dos_sha} != {receipt.get('dossier_sha256')}")
    dossier = strict_json_loads(dossier_raw)
    dos_arm = str(dossier.get("arm_name") or ARM_NAME)
    prereg_arm = str((dossier.get("preregistration_payload") or {}).get("arm_name") or ARM_NAME)
    if dos_arm != arm_name or prereg_arm != arm_name:
        raise OperationalHoldError(f"dossier/prereg arm_name mismatch: dossier={dos_arm!r}, prereg={prereg_arm!r}, expected={arm_name!r}")
    prereg_g4_mode = (
        G4_MODE_RAW_TOP5
        if (dossier.get("preregistration_payload") or {}).get("g4_selection_mode") == "raw_top5_apples_to_apples"
        else G4_MODE_PLATT_EV
    )
    validate_arm_g4_mode(arm_name, prereg_g4_mode)

    scores_dict, scores_bytes, scores_sha = load_verified_scores_npz(scores_path)
    if scores_bytes != int(receipt["scores_npz_bytes"]) or scores_sha != str(receipt["scores_npz_sha256"]):
        raise OperationalHoldError("scores.npz bytes or sha256 mismatch vs receipt")

    ledger_raw = ledger_path.read_bytes()
    ledger_sha = sha256_bytes(ledger_raw)
    if ledger_sha != str(receipt["certifier_fit_ledger_prefix_sha256"]):
        raise OperationalHoldError("fit ledger prefix SHA-256 mismatch vs receipt")
    ledger = strict_json_loads(ledger_raw)
    prereg_sha = str(receipt["preregistration_sha256"])
    if dossier.get("preregistration_sha256") != prereg_sha:
        raise OperationalHoldError("dossier preregistration_sha256 mismatch vs receipt")
    cert_n, bld_n = validate_fit_ledger_obj(ledger, expected_prereg_sha=prereg_sha, allow_builder_attempt=False)
    if cert_n != int(receipt["certifier_attempt_count"]) or bld_n != 0:
        raise OperationalHoldError(f"prefix ledger counts mismatch (cert={cert_n}, builder={bld_n})")

    # Verify builder suffix & PROGRESS.md witness bijection
    witnesses = [
        w
        for w in (parse_progress_builder_witnesses(progress_md) if progress_md is not None else [])
        if w["preregistration_sha256"] == prereg_sha and w["receipt_sha256"] == expected_receipt_sha256 and w["arm"] == arm_name
    ]
    if not allow_builder_suffix:
        if witnesses:
            raise OperationalHoldError(f"unexpected Builder fit debit witness(es) present without --allow-builder-suffix: {len(witnesses)}")
        if suffix_path is not None and suffix_path.exists():
            raise OperationalHoldError(f"unexpected builder_suffix.json present without --allow-builder-suffix: {suffix_path}")
    else:
        if suffix_path is None or not suffix_path.is_file():
            raise OperationalHoldError(f"--allow-builder-suffix requires mandatory builder_suffix.json at {suffix_path}")
        if progress_md is None:
            raise OperationalHoldError("--allow-builder-suffix requires --progress-md to verify Builder fit debit witness")
        suffix = strict_json_loads(suffix_path.read_bytes())
        if suffix.get("schema") != "sota-stack05-builder-suffix/v1":
            raise OperationalHoldError(f"invalid builder_suffix schema: {suffix.get('schema')!r}")
        if suffix.get("certifier_receipt_sha256") != expected_receipt_sha256:
            raise OperationalHoldError("builder_suffix certifier_receipt_sha256 mismatch")
        if suffix.get("preregistration_sha256") != prereg_sha:
            raise OperationalHoldError("builder_suffix preregistration_sha256 mismatch")
        if suffix.get("certifier_fit_ledger_prefix_sha256") != ledger_sha:
            raise OperationalHoldError("builder_suffix prefix ledger sha256 mismatch")
        if suffix.get("book_content_id") != receipt.get("final_fit_expected_content_id"):
            raise OperationalHoldError(
                f"builder_suffix book_content_id {suffix.get('book_content_id')!r} != receipt {receipt.get('final_fit_expected_content_id')!r}"
            )
        b_att = suffix.get("builder_attempt")
        if not isinstance(b_att, dict) or b_att.get("stage") != "builder" or b_att.get("role") != "builder_final_fit":
            raise OperationalHoldError(f"invalid builder_suffix builder_attempt: {b_att!r}")
        reconstructed_ledger = copy.deepcopy(ledger)
        reconstructed_ledger["attempts"].append(b_att)
        _, recon_bld = validate_fit_ledger_obj(reconstructed_ledger, expected_prereg_sha=prereg_sha, allow_builder_attempt=True)
        if recon_bld != 1:
            raise OperationalHoldError(f"expected exact 1 builder attempt after suffix reconstruction, got {recon_bld}")
        recon_sha = sha256_bytes(canonical_json_bytes(reconstructed_ledger))
        if recon_sha != suffix.get("post_builder_fit_ledger_sha256"):
            raise OperationalHoldError("reconstructed post-builder fit ledger SHA-256 mismatch vs builder_suffix.json")
        if len(witnesses) != 1:
            raise OperationalHoldError(f"expected exact 1 Builder fit debit witness in {progress_md}, found {len(witnesses)}")
        if witnesses[0]["ledger_sha256"] != recon_sha:
            raise OperationalHoldError("PROGRESS.md Builder fit debit witness ledger_sha256 mismatch")
        if suffix.get("progress_witness_line") != witnesses[0]["line"]:
            raise OperationalHoldError("builder_suffix progress_witness_line mismatch vs PROGRESS.md")

    # Zero-fit reconstruction from pinned DEV inputs (or injected fixture inputs for unit tests)
    if _injected_data is not None:
        df_dev = _injected_data["df_dev"]
        fcols = _injected_data["fcols"]
        tick_ts = _injected_data["tick_ts"]
        tick_px = _injected_data["tick_px"]
        ret_fwd_true = _injected_data["ret_fwd_true"]
        valid_true = _injected_data["valid_true"]
        entry_true = _injected_data["entry_true"]
        exit_true = _injected_data["exit_true"]
        eligible_idx = _injected_data["eligible_idx"]
        role_specs = _injected_data["role_specs"]
        peer_tie_info = _injected_data["peer_tie_info"]
    else:
        df_dev = load_pinned_dev_parquet(PAIR, dev_dir=dev_dir, pinned_parquets=pinned_parquets)
        fcols = select_core66_columns(df_dev)
        tick_ts, tick_px, _ = load_pinned_dev_ticks(
            PAIR,
            dev_dir=dev_dir,
            ticks_root=ticks_root,
            expected_manifest_sha256=expected_manifest_sha256,
        )
        ts_all = df_dev["decision_ts"].to_numpy(dtype=np.int64)
        ret_fwd_true, valid_true, entry_true, exit_true = settle_with_intervals(
            ts_all, tick_ts, tick_px, horizon_s=HORIZON_S, delay_s=0, tol_s=TOL_S
        )
        eligible_idx = build_eligible_indices(df_dev, fcols, valid_true, ret_fwd_true)
        role_specs = build_all_role_specs(ts_all, ret_fwd_true, valid_true, entry_true, exit_true, eligible_idx)
        peer_tie_info = _compute_historical_6pair_max_tie_rate(dev_dir=dev_dir, pinned_parquets=pinned_parquets)

    stored_roles = dossier.get("roles")
    if not isinstance(stored_roles, dict):
        raise OperationalHoldError("dossier missing roles dict")
    cal_by_role = {
        r: {
            "cal_admissible": bool(stored_roles[r]["cal_admissible"]),
            "failure_reason": stored_roles[r].get("failure_reason"),
            "n_val_oos": stored_roles[r].get("n_val_oos"),
            "conf_thr": stored_roles[r].get("conf_thr"),
            "platt_coef": stored_roles[r].get("platt_coef"),
            "platt_intercept": stored_roles[r].get("platt_intercept"),
        }
        for r in [s["role"] for s in role_specs]
    }
    recon = evaluate_all_gates_from_scores(
        df_dev=df_dev,
        fcols=fcols,
        tick_ts=tick_ts,
        tick_px=tick_px,
        ret_fwd_true=ret_fwd_true,
        valid_true=valid_true,
        entry_true=entry_true,
        exit_true=exit_true,
        eligible_idx=eligible_idx,
        role_specs=role_specs,
        scores_dict=scores_dict,
        cal_by_role=cal_by_role,
        peer_tie_info=peer_tie_info,
        preregistration_payload=dossier["preregistration_payload"],
        preregistration_sha256=prereg_sha,
        local_source_closure_sha256=str(dossier["local_source_closure_sha256"]),
        final_fit_expected_identity=(
            {
                "content_id": dossier["final_fit_expected_content_id"],
                "artifacts": dossier["final_fit_expected_artifacts"],
            }
            if "final_fit_expected_content_id" in dossier
            else None
        ),
        strict_verify_cal=True,
        arm_name=arm_name,
    )

    # Compare every role's reconstructed metadata & trade_ts_sha256
    for rname, exp_rec in recon["roles"].items():
        got_rec = stored_roles.get(rname)
        if not isinstance(got_rec, dict):
            raise OperationalHoldError(f"missing stored role {rname!r}")
        for k in ("support_ok", "cal_admissible", "failure_reason", "role_indices_sha256", "trade_ts_sha256", "n", "wins", "supported"):
            if got_rec.get(k) != exp_rec.get(k):
                raise OperationalHoldError(f"role {rname!r} field {k!r} mismatch: {got_rec.get(k)!r} != {exp_rec.get(k)!r}")

    # B3 + D1: Verify EVERY element of G1_cpcv.paths.{path_id, test_groups, supported, n, wins, wr, trade_ts, trade_wins}!
    got_g1_paths = (dossier.get("G1_cpcv") or {}).get("paths")
    exp_g1_paths = recon["G1_cpcv"]["paths"]
    if not isinstance(got_g1_paths, list) or len(got_g1_paths) != len(exp_g1_paths):
        raise OperationalHoldError("G1_cpcv.paths length mismatch")
    for idx_p, (gp, ep) in enumerate(zip(got_g1_paths, exp_g1_paths)):
        for k in ("path_id", "test_groups", "supported", "n", "wins", "trade_ts"):
            if gp.get(k) != ep.get(k):
                raise OperationalHoldError(f"G1_cpcv.paths[{idx_p}].{k} mismatch")
        if gp.get("wr") is None or ep.get("wr") is None:
            if gp.get("wr") != ep.get("wr"):
                raise OperationalHoldError(f"G1_cpcv.paths[{idx_p}].wr null mismatch")
        elif abs(float(gp["wr"]) - float(ep["wr"])) > 1e-12:
            raise OperationalHoldError(f"G1_cpcv.paths[{idx_p}].wr mismatch")
        if gp.get("trade_wins") != ep.get("trade_wins"):
            raise OperationalHoldError(
                f"G1_cpcv.paths[{idx_p}].trade_wins mismatch (aligned binary outcomes tampered or inconsistent)"
            )

    # Authenticate source closure agreement between dossier & receipt and compare reconstructed gate records!
    if dossier.get("local_source_closure_sha256") != receipt.get("local_source_closure_sha256"):
        raise OperationalHoldError("dossier local_source_closure_sha256 mismatch vs receipt")
    for gate_k in ("G1_cpcv", "G2_placebo", "G3_entry_delay", "G4_sign_null", "G5_payout", "final_fit", "validity", "verdict_goal", "strict_certificate_up"):
        if canonical_json_bytes(dossier.get(gate_k)) != canonical_json_bytes(recon.get(gate_k)):
            raise OperationalHoldError(f"reconstructed gate record {gate_k!r} mismatch vs stored dossier")
    if bool(dossier.get("all_5_gates_passed")) != bool(recon.get("all_5_gates_passed")):
        raise OperationalHoldError("all_5_gates_passed mismatch vs reconstructed gates")

    # Compare verdicts, failed_predicates, and raw certificate reread
    if dossier.get("verdict") != recon["verdict"] or receipt.get("verdict") != recon["verdict"]:
        raise OperationalHoldError(
            f"verdict mismatch: stored dossier={dossier.get('verdict')!r}, receipt={receipt.get('verdict')!r}, reconstructed={recon['verdict']!r}"
        )
    if dossier.get("failed_predicates") != recon["failed_predicates"] or receipt.get("failed_predicates") != recon["failed_predicates"]:
        raise OperationalHoldError("failed_predicates mismatch vs reconstructed gates")
    if expected_verdict is not None and recon["verdict"] != expected_verdict:
        raise OperationalHoldError(f"verified verdict {recon['verdict']!r} != expected {expected_verdict!r}")
    if recon["verdict"] == "PASS" and not receipt.get("final_fit_expected_content_id"):
        raise OperationalHoldError("PASS receipt missing final_fit_expected_content_id")

    exact, differ, unreadable, legacy_only = cert_mod._reread_result(names["dossier"], dossier)
    if (exact, differ, unreadable, legacy_only) != (2, [], [], []):
        raise OperationalHoldError(
            f"stored dossier failed certificate._reread_result: {(exact, differ, unreadable, legacy_only)}"
        )

    return {
        "verified": True,
        "verdict": recon["verdict"],
        "failed_predicates": recon["failed_predicates"],
        "receipt_sha256": actual_rcpt_sha,
        "dossier_sha256": actual_dos_sha,
        "certifier_attempt_count": cert_n,
        "builder_suffix_verified": allow_builder_suffix,
        "reread_exact": exact,
    }


# ---------------------------------------------------------------------------
# Full Certification Execution Engine (with Checkpoint & Wall-Time Guard)
# ---------------------------------------------------------------------------


def run_certification(
    *,
    out_dir: Path | None = None,
    max_wall_s: float = 14400.0,
    dev_dir: Path = DEFAULT_DEV_DATASET_DIR,
    ticks_root: Path = DEFAULT_TICKS_ROOT,
    repo_root: Path = REPO_ROOT,
    arm_name: str = ARM_NAME,
    g4_mode: str | None = None,
) -> dict[str, Any]:
    eff_g4_mode = g4_mode if g4_mode is not None else REGISTERED_ARM_G4_MODES.get(arm_name, G4_MODE_PLATT_EV)
    validate_arm_g4_mode(arm_name, eff_g4_mode)
    g4_mode = eff_g4_mode
    t_start = time.monotonic()
    check_pipeline_features_dir_exists()
    closure_sha, closure_files = verify_source_closure_and_loaded_modules(repo_root)

    names = artifact_names_for_arm(arm_name)
    out_dir = Path(out_dir) if out_dir is not None else default_out_dir_for_arm(arm_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    lock_path = out_dir / names["lock"]
    ckpt_path = out_dir / names["ckpt"]
    ledger_path = out_dir / names["ledger"]
    scores_path = out_dir / names["scores"]
    dossier_path = out_dir / names["dossier"]
    receipt_path = out_dir / names["receipt"]
    role_cache_dir = out_dir / (".role_scores_cache" if arm_name == ARM_NAME else f".role_scores_cache_{arm_name}")
    role_cache_dir.mkdir(parents=True, exist_ok=True)

    # Authenticate pinned DEV inputs BEFORE any debit or fit!
    dev_manifest = load_pinned_dev_manifest(dev_dir)
    df_dev = load_pinned_dev_parquet(PAIR, dev_dir=dev_dir)
    fcols = select_core66_columns(df_dev)
    tick_ts, tick_px, tick_files = load_pinned_dev_ticks(PAIR, dev_dir=dev_dir, ticks_root=ticks_root, manifest=dev_manifest)
    peer_tie_info = _compute_historical_6pair_max_tie_rate(dev_dir=dev_dir)

    # Verify historical comparator SHA-256
    comp_raw = read_verified_file_nofollow(repo_root, HISTORICAL_COMPARATOR_REL.split("/"), expected_sha256=HISTORICAL_COMPARATOR_SHA256)
    assert len(comp_raw) > 0

    preregistration_payload: dict[str, Any] = {
        "schema": "sota-stack05-preregistration/v1",
        "arm_name": arm_name,
        "book_id": BOOK_ID,
        "pair": PAIR,
        "horizon_min": HORIZON_MIN,
        "side": "UP",
        "cov": COV,
        "quoted_payout_ratio": QUOTED_PAYOUT_RATIO,
        "cutoff_utc": CUTOFF_UTC,
        "shared_group_bounds_utc": list(SHARED_GROUP_BOUNDS_UTC),
        "path_years": PATH_YEARS,
        "dev_manifest_sha256": DEV_MANIFEST_SHA256,
        "pinned_dev_parquets": {k: {"bytes": v[0], "sha256": v[1]} for k, v in sorted(PINNED_DEV_PARQUETS.items())},
        "historical_comparator_sha256": HISTORICAL_COMPARATOR_SHA256,
        "local_source_closure_sha256": closure_sha,
        "feature_cols": list(fcols),
        "g3_freshness_rule": "owner_decision_1_deoverlapped_le2s_subgroup_baseline",
        "percentile_method": "linear",
    }
    if g4_mode == G4_MODE_RAW_TOP5:
        preregistration_payload["g4_selection_mode"] = "raw_top5_apples_to_apples"
    prereg_sha = sha256_bytes(canonical_json_bytes(preregistration_payload))
    init_or_load_fit_ledger(ledger_path, prereg_sha, allow_builder_attempt=False)

    ts_all = df_dev["decision_ts"].to_numpy(dtype=np.int64)
    X_all = df_dev[fcols].to_numpy(dtype=np.float64)
    ret_fwd_true, valid_true, entry_true, exit_true = settle_with_intervals(
        ts_all, tick_ts, tick_px, horizon_s=HORIZON_S, delay_s=0, tol_s=TOL_S
    )
    eligible_idx = build_eligible_indices(df_dev, fcols, valid_true, ret_fwd_true)
    role_specs = build_all_role_specs(ts_all, ret_fwd_true, valid_true, entry_true, exit_true, eligible_idx)

    # Load existing authenticated checkpoint if resuming
    if ckpt_path.exists():
        ckpt = strict_json_loads(ckpt_path.read_bytes())
        if ckpt.get("preregistration_sha256") != prereg_sha:
            raise OperationalHoldError("checkpoint preregistration_sha256 mismatch")
        completed_roles: dict[str, dict[str, Any]] = dict(ckpt.get("completed_roles") or {})
    else:
        ckpt = {
            "schema": "sota-stack05-cert-ckpt/v1",
            "preregistration_sha256": prereg_sha,
            "preregistration_payload": preregistration_payload,
            "completed_roles": {},
        }
        completed_roles = {}

    scores_dict: dict[str, np.ndarray] = {}
    cal_by_role: dict[str, dict[str, Any]] = {}
    final_fit_bundle_info: dict[str, Any] | None = None

    for idx_r, sp in enumerate(role_specs, 1):
        role = sp["role"]
        is_final = role == "final_fit"
        val_key = "final_fit_val" if is_final else f"{role}_val"
        test_key = None if is_final else f"{role}_test"

        if (time.monotonic() - t_start) > float(max_wall_s):
            raise OperationalHoldError(f"wall_timeout_exceeded ({max_wall_s}s) before role {role}")

        # Check reusable authenticated checkpoint (0 fit units consumed!)
        cached_meta = completed_roles.get(role)
        val_npy = role_cache_dir / f"{val_key}.npy"
        test_npy = (role_cache_dir / f"{test_key}.npy") if test_key else None
        can_reuse = (
            cached_meta is not None
            and cached_meta.get("role_indices_sha256") == sp["role_indices_sha256"]
            and val_npy.is_file()
            and sha256_bytes(val_npy.read_bytes()) == cached_meta.get("val_npy_sha256")
            and (test_npy is None or (test_npy.is_file() and sha256_bytes(test_npy.read_bytes()) == cached_meta.get("test_npy_sha256")))
            and (not is_final or cached_meta.get("final_fit_artifact_buffers_meta") is not None)
        )
        if can_reuse:
            assert cached_meta is not None
            scores_dict[val_key] = np.load(val_npy, allow_pickle=False).astype(np.float64)
            if test_key is not None and test_npy is not None:
                scores_dict[test_key] = np.load(test_npy, allow_pickle=False).astype(np.float64)
            cal_by_role[role] = cached_meta["cal"]
            if is_final:
                final_fit_bundle_info = cached_meta.get("final_fit_artifact_buffers_meta")
            continue

        if not sp["support_ok"]:
            v_arr = np.empty(0, dtype=np.float64)
            t_arr = np.empty(0, dtype=np.float64)
            cal_m = {
                "cal_admissible": False,
                "failure_reason": sp["failure_reason"],
                "n_val_oos": 0,
                "conf_thr": None,
                "platt_coef": None,
                "platt_intercept": None,
            }
            ff_meta = None
        else:
            # Debit BEFORE trainer invocation!
            debit_fit_ledger(ledger_path, lock_path, prereg_sha=prereg_sha, stage="certifier", role=role)
            fit_res = fit_role_trainer(X_all, fcols, sp, return_estimators=is_final)
            v_arr = fit_res["val_p_raw"]
            cal_m = fit_role_calibration(v_arr, sp["ret_arr"][sp["val_idx"]], float(sp["cov"]))
            if not cal_m["cal_admissible"] and not (g4_mode == G4_MODE_RAW_TOP5 and sp["kind"] in ("real", "null")):
                t_arr = np.empty(0, dtype=np.float64)
            else:
                t_arr = fit_res["test_p_raw"]

            ff_meta = None
            if is_final and cal_m["cal_admissible"]:
                # Precompute weights & booster serialized bytes in memory (independent of p10_wr!)
                w_arrays = extract_nystroem_lgb_stack_arrays(
                    fit_res["med"],
                    fit_res["mu"],
                    fit_res["sd"],
                    fit_res["ny_list"],
                    fit_res["sgd_list"],
                    blend_weight=0.5,
                )
                w_bytes = serialize_sota_weights_npz_bytes(w_arrays)
                lgb_bytes = serialize_sota_lgb_txt_bytes(
                    fit_res["lgb_clf"], fcols, num_iteration=int(fit_res["lgb_best_iteration"])
                )
                # Store serialized final_fit binary buffers in $T cache so resume needs 0 fits
                w_cache = role_cache_dir / f"{BOOK_ID}_weights.npz"
                l_cache = role_cache_dir / f"{BOOK_ID}_lgb_s0.txt"
                atomic_write_file_bytes(w_cache, w_bytes, sync_parent=True)
                atomic_write_file_bytes(l_cache, lgb_bytes, sync_parent=True)
                ff_meta = {
                    "weights_name": f"{BOOK_ID}_weights.npz",
                    "weights_bytes": len(w_bytes),
                    "weights_sha256": sha256_bytes(w_bytes),
                    "lgb_name": f"{BOOK_ID}_lgb_s0.txt",
                    "lgb_bytes": len(lgb_bytes),
                    "lgb_sha256": sha256_bytes(lgb_bytes),
                    "lgb_best_iteration": int(fit_res["lgb_best_iteration"]),
                }

        # Persist role cache + checkpoint durably
        v_buf = io.BytesIO()
        npfmt.write_array(v_buf, v_arr.astype(np.float64), allow_pickle=False)
        v_raw = v_buf.getvalue()
        atomic_write_file_bytes(val_npy, v_raw, sync_parent=False)
        t_sha = None
        if test_key is not None and test_npy is not None:
            t_buf = io.BytesIO()
            npfmt.write_array(t_buf, t_arr.astype(np.float64), allow_pickle=False)
            t_raw = t_buf.getvalue()
            atomic_write_file_bytes(test_npy, t_raw, sync_parent=False)
            t_sha = sha256_bytes(t_raw)
            scores_dict[test_key] = t_arr
        fsync_dir(role_cache_dir)
        scores_dict[val_key] = v_arr
        cal_by_role[role] = cal_m
        if is_final:
            final_fit_bundle_info = ff_meta

        completed_roles[role] = {
            "role_indices_sha256": sp["role_indices_sha256"],
            "val_npy_sha256": sha256_bytes(v_raw),
            "test_npy_sha256": t_sha,
            "cal": cal_m,
            "final_fit_artifact_buffers_meta": ff_meta,
        }
        ckpt["completed_roles"] = completed_roles
        atomic_write_file_bytes(ckpt_path, canonical_json_bytes(ckpt), sync_parent=True)

        if idx_r in (1, 15, 21) or idx_r % 30 == 0 or idx_r == len(role_specs):
            print(
                f"[certify_sota_stack05] {idx_r}/{len(role_specs)} role={role} elapsed={time.monotonic()-t_start:.1f}s",
                flush=True,
            )

    # First evaluate all gates to obtain exact verdict & G1 p10_wr
    dossier_prelim = evaluate_all_gates_from_scores(
        df_dev=df_dev,
        fcols=fcols,
        tick_ts=tick_ts,
        tick_px=tick_px,
        ret_fwd_true=ret_fwd_true,
        valid_true=valid_true,
        entry_true=entry_true,
        exit_true=exit_true,
        eligible_idx=eligible_idx,
        role_specs=role_specs,
        scores_dict=scores_dict,
        cal_by_role=cal_by_role,
        peer_tie_info=peer_tie_info,
        preregistration_payload=preregistration_payload,
        preregistration_sha256=prereg_sha,
        local_source_closure_sha256=closure_sha,
        final_fit_expected_identity=None,
        strict_verify_cal=True,
        arm_name=arm_name,
        g4_mode=g4_mode,
    )

    # On PASS, independently compute final_fit_expected_content_id & artifact digests in memory!
    ff_expected_identity: dict[str, Any] | None = None
    if dossier_prelim["verdict"] == "PASS":
        if final_fit_bundle_info is None:
            raise OperationalHoldError("PASS verdict reached without final_fit serialized buffers")
        ff_cal = cal_by_role["final_fit"]
        platt_map = {
            "UP": {
                "coef": float(ff_cal["platt_coef"]),
                "intercept": float(ff_cal["platt_intercept"]),
                "n_val_oos": int(ff_cal["n_val_oos"]),
            }
        }
        strat_dict = build_stack05_strategy_dict(
            feature_cols=fcols,
            confidence_threshold=float(ff_cal["conf_thr"]),
            p10_wr=float(dossier_prelim["G1_cpcv"]["p10_wr"]),
            platt_calibration=platt_map,
        )
        strat_bytes = serialize_sota_strategy_bytes(strat_dict)
        w_bytes = (role_cache_dir / f"{BOOK_ID}_weights.npz").read_bytes()
        l_bytes = (role_cache_dir / f"{BOOK_ID}_lgb_s0.txt").read_bytes()
        if sha256_bytes(w_bytes) != final_fit_bundle_info["weights_sha256"] or sha256_bytes(l_bytes) != final_fit_bundle_info["lgb_sha256"]:
            raise OperationalHoldError("cached final_fit binary buffer digest mismatch")
        items = [
            (f"{BOOK_ID}_strategy.json", strat_bytes),
            (f"{BOOK_ID}_weights.npz", w_bytes),
            (f"{BOOK_ID}_lgb_s0.txt", l_bytes),
        ]
        exp_cid = compute_sota_content_id_from_buffers(items)
        exp_arts = [
            {"filename": name, "bytes": len(raw), "sha256": sha256_bytes(raw)}
            for name, raw in sorted(items, key=lambda kv: kv[0])
        ]
        ff_expected_identity = {"content_id": exp_cid, "artifacts": exp_arts}
        dossier = evaluate_all_gates_from_scores(
            df_dev=df_dev,
            fcols=fcols,
            tick_ts=tick_ts,
            tick_px=tick_px,
            ret_fwd_true=ret_fwd_true,
            valid_true=valid_true,
            entry_true=entry_true,
            exit_true=exit_true,
            eligible_idx=eligible_idx,
            role_specs=role_specs,
            scores_dict=scores_dict,
            cal_by_role=cal_by_role,
            peer_tie_info=peer_tie_info,
            preregistration_payload=preregistration_payload,
            preregistration_sha256=prereg_sha,
            local_source_closure_sha256=closure_sha,
            final_fit_expected_identity=ff_expected_identity,
            strict_verify_cal=True,
            arm_name=arm_name,
            g4_mode=g4_mode,
        )
    else:
        dossier = dossier_prelim

    # Write canonical 523-key uncompressed scores.npz, dossier, and receipt LAST
    ordered_keys = canonical_scores_npz_keys()
    scores_raw = write_deterministic_uncompressed_npz(scores_path, [(k, scores_dict[k]) for k in ordered_keys])
    scores_sha = sha256_bytes(scores_raw)

    ledger_raw = ledger_path.read_bytes()
    ledger_obj = strict_json_loads(ledger_raw)
    cert_n, bld_n = validate_fit_ledger_obj(ledger_obj, expected_prereg_sha=prereg_sha, allow_builder_attempt=False)
    assert bld_n == 0

    dossier_bytes = pretty_json_bytes(dossier)
    atomic_write_file_bytes(dossier_path, dossier_bytes, sync_parent=True)
    dossier_sha = sha256_bytes(dossier_bytes)

    receipt: dict[str, Any] = {
        "schema": RECEIPT_SCHEMA,
        "book_id": BOOK_ID,
        "arm_name": arm_name,
        "preregistration_sha256": prereg_sha,
        "local_source_closure_sha256": closure_sha,
        "dossier_sha256": dossier_sha,
        "scores_npz_sha256": scores_sha,
        "scores_npz_bytes": len(scores_raw),
        "certifier_attempt_count": cert_n,
        "certifier_fit_ledger_prefix_sha256": sha256_bytes(ledger_raw),
        "verdict": dossier["verdict"],
        "failed_predicates": dossier["failed_predicates"],
    }
    if ff_expected_identity is not None:
        receipt["final_fit_expected_content_id"] = ff_expected_identity["content_id"]
        receipt["final_fit_expected_artifacts"] = ff_expected_identity["artifacts"]

    receipt_bytes = pretty_json_bytes(receipt)
    atomic_write_file_bytes(receipt_path, receipt_bytes, sync_parent=True)
    receipt_sha = sha256_bytes(receipt_bytes)
    print(
        f"[certify_sota_stack05] DONE verdict={dossier['verdict']} failed={dossier['failed_predicates']} "
        f"certifier_attempts={cert_n} receipt_sha256={receipt_sha}",
        flush=True,
    )
    return {
        "verdict": dossier["verdict"],
        "failed_predicates": dossier["failed_predicates"],
        "receipt_sha256": receipt_sha,
        "dossier_sha256": dossier_sha,
        "certifier_attempt_count": cert_n,
        "out_dir": str(out_dir),
    }


def publish_certifier_prefix_bundle(
    *,
    src_dir: Path,
    checkout_root: Path = REPO_ROOT,
    archive_dir: Path | None = None,
    lab_root: Path | None = None,
    arm_name: str = ARM_NAME,
) -> dict[str, str]:
    """Durably stage & atomically publish the 4 immutable 0-builder prefix files to checkout & $A (spec §3.2 + B1)."""
    names = artifact_names_for_arm(arm_name)
    archive_dir = Path(archive_dir) if archive_dir is not None else default_archive_dir_for_arm(arm_name)
    if archive_dir.name.startswith("stack05_") and archive_dir.name != arm_name:
        raise OperationalHoldError(f"refusing cross-arm archive target {archive_dir} for arm {arm_name!r}")
    if archive_dir.is_dir():
        allowed_names = {names["scores"], names["ledger"], names["dossier"], names["receipt"], names["builder_suffix"]}
        for existing in archive_dir.iterdir():
            if existing.name not in allowed_names:
                raise OperationalHoldError(f"refusing cross-arm or non-empty archive {archive_dir}: unexpected file {existing.name}")
    scores_n, ledger_n, dossier_n, receipt_n = names["scores"], names["ledger"], names["dossier"], names["receipt"]
    src_files = {
        scores_n: src_dir / scores_n,
        ledger_n: src_dir / ledger_n,
        dossier_n: src_dir / dossier_n,
        receipt_n: src_dir / receipt_n,
    }
    raw_map: dict[str, bytes] = {}
    sha_map: dict[str, str] = {}
    for name, p in src_files.items():
        if not p.is_file():
            raise OperationalHoldError(f"missing source certifier artifact: {p}")
        raw = p.read_bytes()
        raw_map[name] = raw
        sha_map[name] = sha256_bytes(raw)
    rcpt_obj = strict_json_loads(raw_map[receipt_n])
    if str(rcpt_obj.get("arm_name") or ARM_NAME) != arm_name:
        raise OperationalHoldError(f"source receipt arm_name mismatch for {arm_name!r}")

    order = (scores_n, ledger_n, dossier_n, receipt_n)
    checkout_targets = {
        scores_n: checkout_root / "logs" / scores_n,
        ledger_n: checkout_root / "logs" / ledger_n,
        dossier_n: checkout_root / "results" / "json" / dossier_n,
        receipt_n: checkout_root / "logs" / receipt_n,
    }

    def _publish_one(dst: Path, raw: bytes, mode: int | None) -> None:
        if dst.exists():
            if dst.read_bytes() != raw:
                raise OperationalHoldError(f"conflicting existing publication at {dst}")
            return
        atomic_write_file_bytes(dst, raw, mode=mode, noreplace=True, sync_parent=True)

    # 1. Publish to checkout (scores -> ledger -> dossier -> receipt LAST)
    for name in order:
        _publish_one(checkout_targets[name], raw_map[name], mode=None)
    fsync_dir(checkout_root / "logs")
    fsync_dir(checkout_root / "results" / "json")

    # 2. Publish immutable 0444 prefix to archive $A (mode 0444 set BEFORE fsync!)
    archive_dir.mkdir(parents=True, exist_ok=True)
    for name in order:
        _publish_one(archive_dir / name, raw_map[name], mode=0o444)

    # B1: Authenticate all 4 published archive files and sync $A, $LAB/archive, and $LAB bottom-up on BOTH initial & retry!
    for name in order:
        p = archive_dir / name
        if sha256_bytes(p.read_bytes()) != sha_map[name]:
            raise OperationalHoldError(f"archive verification failed for {p}")
    stop_root = lab_root if lab_root is not None else archive_dir.parent.parent
    fsync_ancestry(archive_dir, stop_at=stop_root)
    return sha_map


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm-name", default=ARM_NAME)
    ap.add_argument("--g4-mode", choices=(G4_MODE_PLATT_EV, G4_MODE_RAW_TOP5), default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--max-wall-s", type=float, default=14400.0)
    ap.add_argument("--dev-dir", default=str(DEFAULT_DEV_DATASET_DIR))
    ap.add_argument("--ticks-root", default=str(DEFAULT_TICKS_ROOT))
    ap.add_argument("--verify-bundle", action="store_true")
    ap.add_argument("--expected-receipt-sha256", default="")
    ap.add_argument("--expected-verdict", choices=("PASS", "KILL-STACK"), default=None)
    ap.add_argument("--allow-builder-suffix", action="store_true")
    ap.add_argument("--progress-md", default=None)
    ap.add_argument(
        "--bundle-dir",
        default=None,
        help="Archive directory to verify (default: verify checkout logs/ + results/json/ copy, or per-arm archive when --allow-builder-suffix is set)",
    )
    ap.add_argument("--publish-prefix", action="store_true", help="Publish 4 immutable 0-builder prefix files from --out-dir to checkout & --archive-dir")
    ap.add_argument("--archive-dir", default=None)
    args = ap.parse_args(argv)
    arm_name = str(args.arm_name)
    out_dir = Path(args.out_dir) if args.out_dir else default_out_dir_for_arm(arm_name)
    archive_dir = Path(args.archive_dir) if args.archive_dir else default_archive_dir_for_arm(arm_name)
    eff_g4_mode = args.g4_mode if args.g4_mode is not None else REGISTERED_ARM_G4_MODES.get(arm_name, G4_MODE_PLATT_EV)
    validate_arm_g4_mode(arm_name, eff_g4_mode)

    if args.verify_bundle:
        if not args.expected_receipt_sha256:
            print("ERROR: --verify-bundle requires --expected-receipt-sha256", file=sys.stderr)
            return 2
        verify_bdir = Path(args.bundle_dir) if args.bundle_dir else (default_archive_dir_for_arm(arm_name) if args.allow_builder_suffix else None)
        res = verify_bundle(
            expected_receipt_sha256=args.expected_receipt_sha256.strip(),
            bundle_dir=verify_bdir,
            repo_root=REPO_ROOT,
            expected_verdict=args.expected_verdict,
            allow_builder_suffix=bool(args.allow_builder_suffix),
            progress_md=Path(args.progress_md) if args.progress_md else None,
            dev_dir=Path(args.dev_dir),
            ticks_root=Path(args.ticks_root),
            arm_name=arm_name,
        )
        print(json.dumps(res, indent=2, sort_keys=True))
        return 0

    if args.publish_prefix:
        shas = publish_certifier_prefix_bundle(
            src_dir=out_dir,
            checkout_root=REPO_ROOT,
            archive_dir=archive_dir,
            arm_name=arm_name,
        )
        print(json.dumps({"published": True, "sha256": shas}, indent=2, sort_keys=True))
        return 0

    res = run_certification(
        out_dir=out_dir,
        max_wall_s=float(args.max_wall_s),
        dev_dir=Path(args.dev_dir),
        ticks_root=Path(args.ticks_root),
        repo_root=REPO_ROOT,
        arm_name=arm_name,
        g4_mode=eff_g4_mode,
    )
    print(json.dumps(res, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
