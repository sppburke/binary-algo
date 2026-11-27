#!/usr/bin/env python3
"""Strictly offline (zero network, zero credentials) causal 14-day trailing-120-day rolling refitter.

Reads ONLY finalized Deriv Archive Downloader 1.7 daily tick parquets + sidecars under the pinned
operational archive root (<repo_root>/deriv_data/rolling_ticks_archive; refuses research roots
/mnt/data/deriv and /mnt/data/deriv-sota-lab and any symlinks), verifies completeness, checksums,
and freshness under shared `.download.lock`, builds core66 features with streaming low-RAM memory
(< 1 GB peak RSS on 3.3 GiB VPS), fits the 5-seed RBF Nystroem + SGD ensemble with strict
exit_true < t_refit purge and 3600s block-purged validation, verifies < 1e-11 reload parity, and
commits atomically to hash-chained transitions.jsonl + state.json under per-book `.transition.flock`.

Exit codes:
  0  completed (published active vintage, published abstention, or skipped because --if-due not due)
  2  bad arguments / unsafe path / policy error
  4  incomplete, stale, damaged, or interrupted tick archive (no active vintage published)
  5  model fit failed (ok=False; distinct from completed Platt abstention)
  75 archive `.download.lock` is held by an active downloader run (try again later)
"""

from __future__ import annotations

import argparse
import errno
import fcntl
import gc
import hashlib
import json
import math
import os
import resource
import sys
import tempfile
import time
from contextlib import contextmanager
from datetime import date as date_cls, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import deriv_dataset as dd
from certify_sota_m60_roll14d import build_decision_eligible_indices_60m, fit_rolling_window_ens5
from certify_sota_stack05 import settle_with_intervals
from deriv_sota_tick_stream import MAJORS
from sota_book_freeze import (
    atomic_write_file_bytes,
    extract_nystroem_ens_arrays,
    fsync_dir,
    fsync_file,
    rename_noreplace,
    serialize_sota_strategy_bytes,
    serialize_sota_weights_npz_bytes,
)
from sota_book_runtime import compute_sota_content_id_from_buffers
from sota_rolling_runtime import (
    DEFAULT_ROLLING_VINTAGE_DIR_REL,
    SOTA_ROLLING_VINTAGE_MANIFEST_SCHEMA,
    RollingSotaPolicy,
    RollingVintageError,
    _ensure_dir_nofollow,
    _open_dir_nofollow,
    _read_regular_file_bytes_nofollow,
    _sha256_bytes,
    _strict_json_loads,
    _utc_iso_whole,
    acquire_rolling_transition_lock,
    commit_rolling_transition,
    load_rolling_policy,
    load_verified_vintage_book,
    repair_state_mirror_from_journal,
)

DEFAULT_BOOK_ID = "USDJPY.sota_roll14d_m60_up.v1"
DEFAULT_ARCHIVE_DIR_REL = Path("deriv_data/rolling_ticks_archive")
KNOWN_FX_GLOBAL_HOLIDAYS_MMDD = frozenset({(12, 25), (1, 1)})
MAX_ZERO_TICK_NONHOLIDAY_WEEKDAYS = 3
MAX_CONFIRMED_GAP_WEEKDAYS = 4
FRESHNESS_TAIL_CALENDAR_DAYS = 4


class ArchiveIncompleteError(RuntimeError):
    pass


class ArchiveLockBusyError(RuntimeError):
    pass


class ModelFitFailedError(RuntimeError):
    pass


def _resolve_confined_operational_dir(
    raw_dir: str | Path,
    *,
    repo_root: Path,
    allowed_parent_rel: Path,
    label: str,
) -> Path:
    """Confine operational directories strictly beneath <repo_root>/<allowed_parent_rel>; refuse research roots & symlinks."""
    p = Path(raw_dir)
    if ".." in p.parts:
        raise ValueError(f"refusing '..' in {label}: {raw_dir!r}")
    p_abs = p if p.is_absolute() else (repo_root / p)
    allowed_base = (repo_root / allowed_parent_rel).resolve()
    forbidden_roots = (Path("/mnt/data/deriv"), Path("/mnt/data/deriv-sota-lab"))
    for fb in forbidden_roots:
        try:
            p_abs.resolve().relative_to(fb.resolve())
            raise ValueError(f"{label} refuses research root {fb}: {raw_dir!r}")
        except (OSError, ValueError) as exc:
            if "refuses research root" in str(exc):
                raise

    # Verify lexical containment inside repo_root / allowed_parent_rel (supporting symlinked repo_root on vultr!)
    # while refusing ANY symlink component inside deriv_data/!
    repo_res = repo_root.resolve()
    rel = None
    for base_cand in (repo_root, repo_res):
        try:
            rel = p_abs.relative_to(base_cand)
            p_abs = repo_root / rel
            break
        except ValueError:
            continue
    if rel is None:
        # Handle alias path where parent of repo_root was symlinked (e.g. ~/git/binary-algo -> /mnt/.../binary-algo)
        try:
            rel = p_abs.resolve().relative_to(repo_res)
            p_abs = repo_root / rel
        except (OSError, ValueError) as exc:
            raise ValueError(f"{label} must lie inside {allowed_base}: {raw_dir!r}") from exc
    if not rel.parts or rel.parts[0] != str(allowed_parent_rel):
        raise ValueError(f"{label} must lie inside {repo_root / allowed_parent_rel}: {raw_dir!r}")
    cur = repo_res
    for part in rel.parts:
        cur = cur / part
        if cur.is_symlink():
            raise ValueError(f"refusing symlink component in {label}: {cur}")
    if p_abs.exists() and not p_abs.resolve().is_relative_to(allowed_base):
        raise ValueError(f"{label} escaped {allowed_base}: {p_abs}")
    return p_abs


@contextmanager
def acquire_archive_shared_lock(archive_dir: Path, *, trusted_root: Path | None = None) -> Iterator[int]:
    """Acquire non-blocking LOCK_SH on <archive_dir>/.download.lock (mutually exclusive with Downloader 1.7)."""
    dir_fd, _ = _open_dir_nofollow(archive_dir, trusted_root=trusted_root)
    fd = -1
    try:
        try:
            fd = os.open(".download.lock", os.O_RDONLY | os.O_NOFOLLOW, dir_fd=dir_fd)
        except OSError as exc:
            if exc.errno == errno.ENOENT:
                fd = os.open(".download.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600, dir_fd=dir_fd)
            elif exc.errno == errno.ELOOP:
                raise ArchiveIncompleteError("symlinked .download.lock refused") from exc
            else:
                raise
        try:
            fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ArchiveLockBusyError(f"archive download lock busy: {archive_dir / '.download.lock'}") from exc
        yield fd
    finally:
        if fd >= 0:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)
        os.close(dir_fd)


def compute_scheduled_refit_epoch(now_epoch: int, *, origin_epoch: int, cadence_days: int) -> tuple[int, int]:
    """Return deterministic (scheduled_refit_epoch, next_refit_epoch) on exact cadence boundaries <= now_epoch."""
    step_s = int(cadence_days) * 86_400
    if int(now_epoch) < int(origin_epoch):
        raise ValueError(f"now_epoch {now_epoch} precedes cadence_origin_utc_epoch {origin_epoch}")
    k = (int(now_epoch) - int(origin_epoch)) // step_s
    sched = int(origin_epoch) + k * step_s
    return sched, sched + step_s


def _audit_and_plan_pair_files(
    ticks_root: Path,
    pair: str,
    *,
    first_day: date_cls,
    last_day: date_cls,
    trusted_root: Path | None = None,
) -> tuple[list[tuple[date_cls, Path, str]], dict[str, int]]:
    """Audit Downloader 1.7 sidecars & parquets for one pair over [first_day, last_day] (inclusive).

    Returns verified `(day, parquet_path, expected_sha256)` list and coverage summary counts.
    """
    pair_dir = ticks_root / pair
    try:
        p_fd, _ = _open_dir_nofollow(pair_dir, trusted_root=trusted_root)
    except Exception as exc:
        raise ArchiveIncompleteError(f"{pair}: missing or invalid ticks directory {pair_dir}: {exc}") from exc

    verified_files: list[tuple[date_cls, Path, str]] = []
    counts = {
        "complete_days": 0,
        "confirmed_gap_days": 0,
        "global_holiday_closed_days": 0,
        "nonholiday_zero_tick_weekdays": 0,
        "weekend_days": 0,
    }
    latest_tick_day: date_cls | None = None

    try:
        cur = first_day
        while cur <= last_day:
            ds = cur.isoformat()
            meta_name = f"{pair}_{ds}_ticks.meta.json"
            pq_name = f"{pair}_{ds}_ticks.parquet"
            is_weekend = cur.weekday() >= 5
            try:
                meta_bytes, _ = _read_regular_file_bytes_nofollow(p_fd, meta_name)
                meta = _strict_json_loads(meta_bytes)
                if not isinstance(meta, dict):
                    raise ArchiveIncompleteError(f"{pair} {ds}: sidecar is not a JSON object")
            except FileNotFoundError:
                if is_weekend:
                    counts["weekend_days"] += 1
                    cur += timedelta(days=1)
                    continue
                raise ArchiveIncompleteError(f"{pair} {ds}: missing downloader sidecar {meta_name}")

            # Refuse unfinished atomic replacement (Downloader 1.7 crash marker!)
            if meta.get("parquet_sha256_next") is not None:
                raise ArchiveIncompleteError(f"{pair} {ds}: unfinished downloader update (parquet_sha256_next present)")

            # Verify Downloader 1.7 sidecar identity & completeness invariants (defeats cross-pair/wrong-day swaps!)
            want_sym = f"frx{pair}"
            got_sym = meta.get("symbol")
            got_date = meta.get("date")
            if got_sym != want_sym or got_date != ds:
                raise ArchiveIncompleteError(
                    f"{pair} {ds}: sidecar identity mismatch (symbol={got_sym!r} date={got_date!r} != {want_sym!r} {ds!r})"
                )
            if meta.get("clipped_by_retention") is True or meta.get("clipped_by_now") is True:
                raise ArchiveIncompleteError(f"{pair} {ds}: clipped sidecar refused")

            is_closed = bool(meta.get("market_closed") is True)
            is_complete = bool(meta.get("complete") is True)
            is_reconf_gap = bool(meta.get("gaps_reconfirmed") is True and isinstance(meta.get("gaps_first_seen"), str) and bool(meta["gaps_first_seen"].strip()))

            if is_weekend:
                counts["weekend_days"] += 1
                if not is_closed and is_complete and meta.get("parquet_sha256"):
                    pq_bytes, _ = _read_regular_file_bytes_nofollow(p_fd, pq_name)
                    if _sha256_bytes(pq_bytes) != meta["parquet_sha256"]:
                        raise ArchiveIncompleteError(f"{pair} {ds}: parquet_sha256 mismatch")
                    verified_files.append((cur, pair_dir / pq_name, str(meta["parquet_sha256"])))
                cur += timedelta(days=1)
                continue

            if is_closed:
                if not is_complete:
                    raise ArchiveIncompleteError(f"{pair} {ds}: market_closed=true without complete=true")
                if (cur.month, cur.day) in KNOWN_FX_GLOBAL_HOLIDAYS_MMDD:
                    counts["global_holiday_closed_days"] += 1
                else:
                    # Charge non-holiday twice-empty weekday closures against strict gap budget!
                    counts["nonholiday_zero_tick_weekdays"] += 1
                cur += timedelta(days=1)
                continue

            if not (is_complete or is_reconf_gap):
                raise ArchiveIncompleteError(f"{pair} {ds}: unconfirmed incomplete weekday (complete=false, gaps_reconfirmed=false)")

            exp_sha = meta.get("parquet_sha256")
            if type(exp_sha) is not str or len(exp_sha) != 64:
                raise ArchiveIncompleteError(f"{pair} {ds}: missing valid parquet_sha256 in sidecar")
            try:
                pq_bytes, _ = _read_regular_file_bytes_nofollow(p_fd, pq_name)
            except FileNotFoundError as exc:
                raise ArchiveIncompleteError(f"{pair} {ds}: missing tick parquet {pq_name}") from exc
            if _sha256_bytes(pq_bytes) != exp_sha:
                raise ArchiveIncompleteError(f"{pair} {ds}: parquet_sha256 mismatch")
            if is_complete:
                counts["complete_days"] += 1
            else:
                counts["confirmed_gap_days"] += 1
            verified_files.append((cur, pair_dir / pq_name, exp_sha))
            latest_tick_day = cur
            cur += timedelta(days=1)
    finally:
        os.close(p_fd)

    if counts["nonholiday_zero_tick_weekdays"] > MAX_ZERO_TICK_NONHOLIDAY_WEEKDAYS:
        raise ArchiveIncompleteError(
            f"{pair}: {counts['nonholiday_zero_tick_weekdays']} non-holiday zero-tick weekdays > budget {MAX_ZERO_TICK_NONHOLIDAY_WEEKDAYS}"
        )
    if counts["confirmed_gap_days"] > MAX_CONFIRMED_GAP_WEEKDAYS:
        raise ArchiveIncompleteError(
            f"{pair}: {counts['confirmed_gap_days']} confirmed-gap weekdays > budget {MAX_CONFIRMED_GAP_WEEKDAYS}"
        )
    if latest_tick_day is None or (last_day - latest_tick_day).days > FRESHNESS_TAIL_CALENDAR_DAYS:
        raise ArchiveIncompleteError(
            f"{pair}: stale archive tail (latest_tick_day={latest_tick_day}, required within {FRESHNESS_TAIL_CALENDAR_DAYS}d of {last_day})"
        )
    return verified_files, counts


def _read_day_ticks_verified(
    pq_path: Path,
    *,
    day: date_cls,
    cutoff_utc: int,
    expected_sha256: str,
    trusted_root: Path | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Read one daily parquet strictly verifying O_NOFOLLOW, SHA-256, whole-second ns UTC, strict monotonicity, finite >0 prices, < cutoff_utc."""
    import io
    import pyarrow as pa

    p_fd, _ = _open_dir_nofollow(pq_path.parent, trusted_root=trusted_root)
    try:
        raw, _ = _read_regular_file_bytes_nofollow(p_fd, pq_path.name)
    finally:
        os.close(p_fd)
    if _sha256_bytes(raw) != expected_sha256:
        raise ArchiveIncompleteError(f"TOCTOU checksum mismatch reading {pq_path.name}")
    tbl = pq.read_table(io.BytesIO(raw), columns=["datetime_utc", "price"])
    if len(tbl) == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
    if tbl.column(0).type != pa.timestamp("ns", tz="UTC"):
        raise ArchiveIncompleteError(f"{pq_path.name}: expected timestamp[ns, tz=UTC], got {tbl.column(0).type}")
    ts_ns = tbl.column(0).cast("int64").to_numpy()
    if np.any(ts_ns % 1_000_000_000 != 0):
        raise ArchiveIncompleteError(f"{pq_path.name}: non-whole-second timestamps refused")
    ts_s = ts_ns // 1_000_000_000
    px = tbl.column(1).to_numpy().astype(np.float64, copy=False)
    if not np.all(np.isfinite(px)) or not np.all(px > 0.0):
        raise ArchiveIncompleteError(f"{pq_path.name}: non-finite or non-positive prices refused")
    if len(ts_s) > 1 and not bool(np.all(np.diff(ts_s) > 0)):
        raise ArchiveIncompleteError(f"{pq_path.name}: non-increasing timestamps refused")
    day_lo = int(datetime(day.year, day.month, day.day, 0, 0, 0, tzinfo=timezone.utc).timestamp())
    day_hi = day_lo + 86_400
    if int(ts_s[0]) < day_lo or int(ts_s[-1]) >= day_hi:
        raise ArchiveIncompleteError(f"{pq_path.name}: timestamps outside UTC day {day.isoformat()}")
    if int(ts_s[-1]) >= int(cutoff_utc):
        raise ArchiveIncompleteError(f"{pq_path.name}: contains tick >= t_refit cutoff ({ts_s[-1]} >= {cutoff_utc})")
    return ts_s, px


def _build_streaming_feature_frame_and_target_ticks(
    ticks_root: Path,
    policy: RollingSotaPolicy,
    *,
    first_day: date_cls,
    last_day: date_cls,
    cutoff_utc: int,
    trusted_root: Path | None = None,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, dict[str, Any]]:
    """Build exact core66 feature frame pair-by-pair / day-by-day for low VPS peak RSS (< 1 GB).

    For the 6 non-target MAJORS, computes only minute-bar `close` day-by-day (~1 MB per pair total)
    and discards raw ticks immediately. Keeps raw 1s ticks ONLY for policy.pair (`USDJPY`) to run
    exact `dd.minute_bars` and `settle_with_intervals(..., horizon_s=3600, tol_s=60)`.
    """
    target = policy.pair.upper()
    audit_by_pair: dict[str, Any] = {}
    closes_others: dict[str, pd.Series] = {}
    tgt_ts_parts: list[np.ndarray] = []
    tgt_px_parts: list[np.ndarray] = []

    for p in MAJORS:
        day_list, counts = _audit_and_plan_pair_files(
            ticks_root, p, first_day=first_day, last_day=last_day, trusted_root=trusted_root
        )
        audit_by_pair[p] = counts
        if p == target:
            for d_cur, pq_p, sha in day_list:
                ts_d, px_d = _read_day_ticks_verified(pq_p, day=d_cur, cutoff_utc=cutoff_utc, expected_sha256=sha, trusted_root=trusted_root)
                if len(ts_d):
                    tgt_ts_parts.append(ts_d)
                    tgt_px_parts.append(px_d)
        else:
            close_idx_parts: list[np.ndarray] = []
            close_val_parts: list[np.ndarray] = []
            for d_cur, pq_p, sha in day_list:
                ts_d, px_d = _read_day_ticks_verified(pq_p, day=d_cur, cutoff_utc=cutoff_utc, expected_sha256=sha, trusted_root=trusted_root)
                if len(ts_d):
                    m = (ts_d // 60) * 60
                    # Last tick of each minute equals dd.minute_bars(ts, px)['close'] exactly!
                    is_last = np.empty(len(m), dtype=bool)
                    is_last[:-1] = m[:-1] != m[1:]
                    is_last[-1] = True
                    close_idx_parts.append(m[is_last] + 60)
                    close_val_parts.append(px_d[is_last])
            if not close_idx_parts:
                raise ArchiveIncompleteError(f"{p}: zero ticks across [{first_day}, {last_day}]")
            closes_others[p] = pd.Series(
                np.concatenate(close_val_parts),
                index=np.concatenate(close_idx_parts).astype(np.int64),
            )
            del close_idx_parts, close_val_parts
            gc.collect()

    if not tgt_ts_parts:
        raise ArchiveIncompleteError(f"{target}: zero ticks across [{first_day}, {last_day}]")
    ts_tgt = np.concatenate(tgt_ts_parts)
    px_tgt = np.concatenate(tgt_px_parts)
    del tgt_ts_parts, tgt_px_parts
    gc.collect()

    b_tgt = dd.minute_bars(ts_tgt, px_tgt)
    X = dd.own_features(b_tgt)
    dec = b_tgt.index.values.astype(np.int64) + 60
    target_close_t = pd.Series(dec, index=b_tgt.index)
    X = X.join(dd.cross_features(target_close_t, closes_others).set_index(b_tgt.index))
    del b_tgt, closes_others, target_close_t
    X["decision_ts"] = dec
    X = X.astype({c: "float32" for c in X.columns if X[c].dtype == np.float64})
    X = X.reset_index(names="bar_start_ts")
    missing = [c for c in policy.feature_cols if c not in X.columns]
    if missing:
        raise ArchiveIncompleteError(f"missing core66 columns after feature build: {missing[:10]}")
    gc.collect()
    return X, ts_tgt, px_tgt, audit_by_pair


def _stage_immutable_vintage_dir(
    book_vintage_dir: Path,
    *,
    policy: RollingSotaPolicy,
    fit: dict[str, Any],
    X_val_df: pd.DataFrame,
    val_epochs: np.ndarray,
    expected_val_probs: np.ndarray,
    published_epoch: int,
    scheduled_refit_epoch: int,
    valid_from_epoch: int,
    valid_until_epoch: int,
    train_window_utc: list[str],
    trusted_root: Path | None = None,
) -> tuple[str, str, str]:
    """Serialize weights/strategy/manifest, verify <1e-11 reload parity, stage immutable vintages/<vintage_id>/."""
    arrays = extract_nystroem_ens_arrays(fit["med"], fit["mu"], fit["sd"], fit["ny_list"], fit["sgd_list"])
    weights_bytes = serialize_sota_weights_npz_bytes(arrays)

    # First compute vintage_content_id over deterministic weights + preliminary strategy, then embed vintage_id
    base_strat: dict[str, Any] = {
        "schema": "sota-rolling-vintage-strategy/v1",
        "book_id": policy.book_id,
        "policy_content_id": policy.content_id,
        "pair": policy.pair,
        "horizon_minutes": policy.horizon_minutes,
        "allowed_sides": list(policy.allowed_sides),
        "contract_family": policy.contract_family,
        "model_kind": policy.model_kind,
        "feature_cols": list(policy.feature_cols),
        "coverage": policy.coverage,
        "confidence_threshold": float(fit["conf_thr"]),
        "drop_ny_16h": policy.drop_ny_16h,
        "min_payout_ratio": policy.min_payout_ratio,
        "payout_edge_margin": policy.payout_edge_margin,
        "rolling_oos_p10_floors": dict(policy.rolling_oos_p10_floors),
        "platt_calibration": {
            "UP": {"coef": float(fit["platt_coef"]), "intercept": float(fit["platt_intercept"])}
        },
        "scheduled_refit_utc": _utc_iso_whole(scheduled_refit_epoch),
        "valid_from_utc": _utc_iso_whole(valid_from_epoch),
        "valid_until_utc": _utc_iso_whole(valid_until_epoch),
        "train_window_utc": list(train_window_utc),
        "n_fit": int(fit["n_fit"]),
        "n_val": int(fit["n_val"]),
    }
    strat_fname = f"{policy.book_id}_strategy.json"
    weights_fname = f"{policy.book_id}_weights.npz"
    strat_bytes = serialize_sota_strategy_bytes(base_strat)
    vintage_cid = compute_sota_content_id_from_buffers([(strat_fname, strat_bytes), (weights_fname, weights_bytes)])
    compact_ts = datetime.fromtimestamp(int(scheduled_refit_epoch), tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    vintage_id = f"{compact_ts}_{vintage_cid}"

    v_manifest = {
        "schema": SOTA_ROLLING_VINTAGE_MANIFEST_SCHEMA,
        "book_id": policy.book_id,
        "policy_content_id": policy.content_id,
        "vintage_id": vintage_id,
        "vintage_content_id": vintage_cid,
        "pair": policy.pair,
        "horizon_minutes": policy.horizon_minutes,
        "allowed_sides": list(policy.allowed_sides),
        "model_kind": policy.model_kind,
        "published_utc": _utc_iso_whole(published_epoch),
        "scheduled_refit_utc": _utc_iso_whole(scheduled_refit_epoch),
        "valid_from_utc": _utc_iso_whole(valid_from_epoch),
        "valid_until_utc": _utc_iso_whole(valid_until_epoch),
        "train_window_utc": list(train_window_utc),
        "artifacts": [
            {"file": strat_fname, "bytes": len(strat_bytes), "sha256": _sha256_bytes(strat_bytes)},
            {"file": weights_fname, "bytes": len(weights_bytes), "sha256": _sha256_bytes(weights_bytes)},
        ],
    }
    v_man_bytes = serialize_sota_strategy_bytes(v_manifest)
    v_man_sha = _sha256_bytes(v_man_bytes)

    vintages_dir = _ensure_dir_nofollow(book_vintage_dir / "vintages", mode=0o700, trusted_root=trusted_root)
    final_vdir = vintages_dir / vintage_id

    if final_vdir.exists():
        # Verify existing immutable vintage directory matches exact digests (idempotent re-stage safety)
        v_fd, _ = _open_dir_nofollow(final_vdir, trusted_root=trusted_root)
        try:
            ex_m, _ = _read_regular_file_bytes_nofollow(v_fd, "vintage.manifest.json")
            ex_s, _ = _read_regular_file_bytes_nofollow(v_fd, strat_fname)
            ex_w, _ = _read_regular_file_bytes_nofollow(v_fd, weights_fname)
        finally:
            os.close(v_fd)
        if _sha256_bytes(ex_m) != v_man_sha or _sha256_bytes(ex_s) != _sha256_bytes(strat_bytes) or _sha256_bytes(ex_w) != _sha256_bytes(weights_bytes):
            raise RollingVintageError(f"conflicting pre-existing vintage directory {final_vdir}")
    else:
        tmp_vdir = Path(tempfile.mkdtemp(prefix=f".stage_{vintage_id}_", dir=vintages_dir))
        os.chmod(tmp_vdir, 0o700)
        try:
            atomic_write_file_bytes(tmp_vdir / strat_fname, strat_bytes, mode=0o444, sync_parent=False)
            atomic_write_file_bytes(tmp_vdir / weights_fname, weights_bytes, mode=0o444, sync_parent=False)
            atomic_write_file_bytes(tmp_vdir / "vintage.manifest.json", v_man_bytes, mode=0o444, sync_parent=True)
            fsync_dir(tmp_vdir)
            rename_noreplace(tmp_vdir, final_vdir)
            fsync_dir(vintages_dir)
        except Exception:
            if tmp_vdir.exists():
                for ch in tmp_vdir.iterdir():
                    ch.unlink(missing_ok=True)
                tmp_vdir.rmdir()
            raise

    # Verify reload & bit-exact (< 1e-11) prediction parity through canonical sota_rolling_runtime loader BEFORE publishing!
    probe_tr = {
        "vintage_id": vintage_id,
        "vintage_content_id": vintage_cid,
        "vintage_manifest_sha256": v_man_sha,
        "scheduled_refit_utc": _utc_iso_whole(scheduled_refit_epoch),
        "valid_from_utc": _utc_iso_whole(valid_from_epoch),
        "valid_until_utc": _utc_iso_whole(valid_until_epoch),
        "platt_coef": float(fit["platt_coef"]),
        "platt_intercept": float(fit["platt_intercept"]),
    }
    reloaded = load_verified_vintage_book(book_vintage_dir, policy=policy, transition_rec=probe_tr, trusted_root=trusted_root)
    got_probs = reloaded.predict_probabilities(X_val_df, decision_epochs=val_epochs)
    max_diff = float(np.max(np.abs(got_probs - expected_val_probs))) if len(got_probs) else 0.0
    if not math.isfinite(max_diff) or max_diff >= 1e-11:
        raise ModelFitFailedError(f"reloaded rolling vintage parity diff {max_diff:.3e} >= 1e-11")
    return vintage_id, vintage_cid, v_man_sha


def run_rolling_refit(
    *,
    book_id: str = DEFAULT_BOOK_ID,
    repo_root: Path = REPO_ROOT,
    archive_dir: Path | None = None,
    vintage_root: Path | None = None,
    now_epoch: int | None = None,
    if_due: bool = False,
    force: bool = False,
    resume_after_revoke: bool = False,
) -> dict[str, Any]:
    """Run one strictly offline causal 14-day trailing-120-day rolling refit & atomic journal publication."""
    t0 = time.monotonic()
    policy = load_rolling_policy(book_id, repo_root=repo_root, allow_demo_canary=True)
    arch_p = _resolve_confined_operational_dir(
        archive_dir if archive_dir is not None else (repo_root / DEFAULT_ARCHIVE_DIR_REL),
        repo_root=repo_root,
        allowed_parent_rel=Path("deriv_data"),
        label="--archive-dir",
    )
    vroot_p = _resolve_confined_operational_dir(
        vintage_root if vintage_root is not None else (repo_root / DEFAULT_ROLLING_VINTAGE_DIR_REL),
        repo_root=repo_root,
        allowed_parent_rel=Path("deriv_data"),
        label="--vintage-root",
    )
    book_vdir = _ensure_dir_nofollow(vroot_p / policy.book_id, mode=0o700, trusted_root=repo_root)

    now_ep = int(time.time()) if now_epoch is None else int(now_epoch)
    sched_ep, next_ep = compute_scheduled_refit_epoch(
        now_ep,
        origin_epoch=policy.cadence_origin_utc_epoch,
        cadence_days=policy.refit_cadence_days,
    )
    valid_from_ep = max(sched_ep, now_ep)
    valid_until_ep = next_ep
    if valid_from_ep >= valid_until_ep:
        raise ValueError(f"late refit already past scheduled expiry ({valid_from_ep} >= {valid_until_ep})")

    # 1. Under LOCK_EX on .transition.flock, verify/repair prefix mirror & check abstention/revocation/cadence!
    with acquire_rolling_transition_lock(book_vdir, exclusive=True, nonblocking=False, trusted_root=repo_root) as lock_dir_fd:
        transitions = repair_state_mirror_from_journal(
            book_vdir,
            expected_book_id=policy.book_id,
            expected_policy_cid=policy.content_id,
            max_vintage_age_days=policy.max_vintage_age_days,
            trusted_root=repo_root,
            dir_fd=lock_dir_fd,
        )
        start_seq = len(transitions)
        start_sha = str(transitions[-1]["transition_sha256"]) if transitions else "GENESIS"
        if transitions:
            # Strict block-wide abstention lockout: ANY abstention in the current block locks out refits
            # through next_refit_epoch EVEN after a later --revoke-reason, --force, or --resume-after-revoke!
            abstained_in_block = [
                t for t in transitions
                if t.get("status") == "abstained"
                and (int(t["scheduled_refit_epoch"]) == sched_ep or now_ep < int(t["next_refit_epoch"]))
            ]
            if abstained_in_block:
                ab_t = abstained_in_block[-1]
                return {
                    "status": "skipped_abstained_block_locked",
                    "book_id": policy.book_id,
                    "policy_content_id": policy.content_id,
                    "scheduled_refit_utc": ab_t["scheduled_refit_utc"],
                    "next_refit_utc": ab_t["next_refit_utc"],
                    "seq": int(ab_t["seq"]),
                    "reason": ab_t.get("reason"),
                    "elapsed_s": round(time.monotonic() - t0, 2),
                }
            tail = transitions[-1]
            # Durable manual revocation: stays halted across scheduled --if-due and --force until --resume-after-revoke!
            if tail.get("status") == "revoked" and not resume_after_revoke:
                return {
                    "status": "skipped_manual_revocation_active",
                    "book_id": policy.book_id,
                    "policy_content_id": policy.content_id,
                    "scheduled_refit_utc": tail["scheduled_refit_utc"],
                    "next_refit_utc": tail["next_refit_utc"],
                    "seq": int(tail["seq"]),
                    "reason": tail.get("reason"),
                    "elapsed_s": round(time.monotonic() - t0, 2),
                }
            tail_sched = int(tail["scheduled_refit_epoch"])
            tail_next = int(tail["next_refit_epoch"])
            # --if-due skips when current scheduled block is already published (active and unexpired)
            if if_due and not force and tail_sched >= sched_ep and now_ep < tail_next:
                if tail.get("status") == "active" and now_ep < int(tail["valid_until_epoch"]):
                    return {
                        "status": "skipped_not_due",
                        "book_id": policy.book_id,
                        "policy_content_id": policy.content_id,
                        "vintage_id": tail["vintage_id"],
                        "valid_until_utc": tail["valid_until_utc"],
                        "next_refit_utc": tail["next_refit_utc"],
                        "seq": int(tail["seq"]),
                        "elapsed_s": round(time.monotonic() - t0, 2),
                    }

    # 2. Read and audit operational archive strictly under shared `.download.lock` (refuses concurrent Downloader 1.7!)
    ticks_root = arch_p / "ticks"
    refit_day = datetime.fromtimestamp(sched_ep, tz=timezone.utc).date()
    last_day = refit_day - timedelta(days=1)
    first_day = refit_day - timedelta(days=policy.train_days + policy.warmup_days)
    train_lo_ep = sched_ep - policy.train_days * 86_400

    with acquire_archive_shared_lock(arch_p, trusted_root=repo_root):
        frame, ts_tgt, px_tgt, audit_by_pair = _build_streaming_feature_frame_and_target_ticks(
            ticks_root,
            policy,
            first_day=first_day,
            last_day=last_day,
            cutoff_utc=sched_ep,
            trusted_root=repo_root,
        )

    # 3. Exact causal settlement (explicit horizon_s=3600, tol_s=60!) & decision-time eligibility (explicit cutoff_utc=sched_ep!)
    fcols = list(policy.feature_cols)
    dec_ts = frame["decision_ts"].to_numpy(dtype=np.int64)
    horizon_s = policy.horizon_minutes * 60
    assert horizon_s == 3600, f"expected 3600s horizon, got {horizon_s}"
    ret_arr, valid_arr, entry_true, exit_true = settle_with_intervals(
        dec_ts,
        ts_tgt,
        px_tgt,
        horizon_s=horizon_s,
        tol_s=policy.settle_tol_s,
    )
    del ts_tgt, px_tgt
    gc.collect()

    elig_idx = build_decision_eligible_indices_60m(frame, fcols, cutoff_utc=sched_ep)
    X_all = frame[fcols].to_numpy(dtype=np.float32)
    del frame
    gc.collect()

    elig_ts = dec_ts[elig_idx]
    elig_settled = valid_arr[elig_idx] & np.isfinite(ret_arr[elig_idx])
    elig_exit = exit_true[elig_idx]
    pool_mask = (
        elig_settled
        & (elig_ts >= train_lo_ep)
        & (elig_ts < sched_ep - horizon_s)
        & (elig_exit > 0)
        & (elig_exit < sched_ep)
    )
    pool_idx = elig_idx[pool_mask]

    fit = fit_rolling_window_ens5(
        X_all,
        dec_ts,
        entry_true,
        exit_true,
        ret_arr,
        pool_idx,
        np.empty(0, dtype=np.int64),
        cov=policy.coverage,
        gap=policy.block_val_gap_s,
        return_estimators=True,
    )
    train_window_utc = [_utc_iso_whole(train_lo_ep), _utc_iso_whole(sched_ep)]
    rss_mb = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1)

    # Distinguish fit failure (ok=False) from completed Platt abstention (ok=True, platt_ok=False)!
    if not fit.get("ok"):
        raise ModelFitFailedError(f"rolling_fit_failed:{fit.get('reason')}")

    # Hold LOCK_EX across publication-time sampling, head recheck, staging, and atomic journal commit!
    # Prevents concurrent revocation/abstention races and ensures pub_ep is sampled AT commit time (never stale/backdated).
    with acquire_rolling_transition_lock(book_vdir, exclusive=True, nonblocking=False, trusted_root=repo_root) as commit_dir_fd:
        pub_ep = int(time.time()) if now_epoch is None else int(now_epoch)
        if pub_ep >= valid_until_ep:
            raise ValueError(f"fit completed past scheduled expiry ({pub_ep} >= {valid_until_ep}); discarding stale fit")

        if not fit.get("platt_ok"):
            rec = commit_rolling_transition(
                book_vdir,
                policy=policy,
                status="abstained",
                published_epoch=pub_ep,
                scheduled_refit_epoch=sched_ep,
                next_refit_epoch=next_ep,
                train_window_utc=train_window_utc,
                platt_coef=fit.get("platt_coef"),
                platt_intercept=fit.get("platt_intercept"),
                reason="platt_nonpositive",
                trusted_root=repo_root,
                expected_prev_seq=start_seq,
                expected_prev_sha=start_sha,
                already_locked=True,
                dir_fd=commit_dir_fd,
            )
            return {
                "status": "abstained",
                "book_id": policy.book_id,
                "policy_content_id": policy.content_id,
                "seq": int(rec["seq"]),
                "transition_sha256": rec["transition_sha256"],
                "reason": rec["reason"],
                "platt_coef": rec["platt_coef"],
                "platt_intercept": rec["platt_intercept"],
                "scheduled_refit_utc": rec["scheduled_refit_utc"],
                "next_refit_utc": rec["next_refit_utc"],
                "n_fit": int(fit["n_fit"]),
                "n_val": int(fit["n_val"]),
                "archive_audit": audit_by_pair,
                "peak_rss_mb": rss_mb,
                "elapsed_s": round(time.monotonic() - t0, 2),
            }

        # Re-check journal head, abstention lockout & manual revocation under LOCK_EX before staging files
        cur_transitions = repair_state_mirror_from_journal(
            book_vdir,
            expected_book_id=policy.book_id,
            expected_policy_cid=policy.content_id,
            max_vintage_age_days=policy.max_vintage_age_days,
            trusted_root=repo_root,
            dir_fd=commit_dir_fd,
        )
        cur_seq = len(cur_transitions)
        cur_sha = str(cur_transitions[-1]["transition_sha256"]) if cur_transitions else "GENESIS"
        if cur_seq != start_seq or cur_sha != start_sha:
            raise RollingVintageError("rolling_journal_head_changed_during_fit")
        if any(
            t.get("status") == "abstained"
            and (int(t["scheduled_refit_epoch"]) == sched_ep or pub_ep < int(t["next_refit_epoch"]))
            for t in cur_transitions
        ):
            raise RollingVintageError("rolling_block_abstained_locked_out")
        if cur_transitions and cur_transitions[-1].get("status") == "revoked" and not resume_after_revoke:
            raise RollingVintageError("rolling_manual_revocation_active_requires_resume")

        val_idx = fit["val_idx"]
        sample_n = min(len(val_idx), 128)
        chk_idx = val_idx[-sample_n:]
        X_val_df = pd.DataFrame(X_all[chk_idx].astype(np.float64), columns=fcols)
        val_epochs = dec_ts[chk_idx]
        exp_val_probs = np.asarray(fit["p_val"][-sample_n:], dtype=np.float64)

        eff_valid_from = max(sched_ep, pub_ep)
        if eff_valid_from >= valid_until_ep:
            raise ValueError(f"publication epoch {pub_ep} reached or exceeded scheduled expiry {valid_until_ep}")

        vintage_id, vintage_cid, v_man_sha = _stage_immutable_vintage_dir(
            book_vdir,
            policy=policy,
            fit=fit,
            X_val_df=X_val_df,
            val_epochs=val_epochs,
            expected_val_probs=exp_val_probs,
            published_epoch=pub_ep,
            scheduled_refit_epoch=sched_ep,
            valid_from_epoch=eff_valid_from,
            valid_until_epoch=valid_until_ep,
            train_window_utc=train_window_utc,
            trusted_root=repo_root,
        )
        # Re-check clock AFTER serialization, fsync, and reload parity so an expired fit NEVER publishes as active!
        commit_now_ep = int(time.time()) if now_epoch is None else int(now_epoch)
        if commit_now_ep >= valid_until_ep:
            raise ValueError(f"fit crossed scheduled expiry during staging ({commit_now_ep} >= {valid_until_ep}); discarding stale fit")
        rec = commit_rolling_transition(
            book_vdir,
            policy=policy,
            status="active",
            published_epoch=pub_ep,
            scheduled_refit_epoch=sched_ep,
            next_refit_epoch=next_ep,
            vintage_id=vintage_id,
            vintage_content_id=vintage_cid,
            vintage_manifest_sha256=v_man_sha,
            valid_from_epoch=eff_valid_from,
            valid_until_epoch=valid_until_ep,
            train_window_utc=train_window_utc,
            platt_coef=float(fit["platt_coef"]),
            platt_intercept=float(fit["platt_intercept"]),
            trusted_root=repo_root,
            expected_prev_seq=start_seq,
            expected_prev_sha=start_sha,
            forbid_abstained_block=True,
            forbid_revoked_without_resume=(not resume_after_revoke),
            already_locked=True,
            dir_fd=commit_dir_fd,
        )
    return {
        "status": "published_active",
        "book_id": policy.book_id,
        "policy_content_id": policy.content_id,
        "seq": int(rec["seq"]),
        "transition_sha256": rec["transition_sha256"],
        "vintage_id": vintage_id,
        "vintage_content_id": vintage_cid,
        "valid_from_utc": rec["valid_from_utc"],
        "valid_until_utc": rec["valid_until_utc"],
        "next_refit_utc": rec["next_refit_utc"],
        "conf_thr": float(fit["conf_thr"]),
        "platt_coef": float(fit["platt_coef"]),
        "platt_intercept": float(fit["platt_intercept"]),
        "n_fit": int(fit["n_fit"]),
        "n_val": int(fit["n_val"]),
        "archive_audit": audit_by_pair,
        "peak_rss_mb": rss_mb,
        "elapsed_s": round(time.monotonic() - t0, 2),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Strictly offline causal 14-day trailing-120-day rolling refitter")
    ap.add_argument("--book-id", default=DEFAULT_BOOK_ID)
    ap.add_argument("--archive-dir", default=str(REPO_ROOT / DEFAULT_ARCHIVE_DIR_REL))
    ap.add_argument("--vintage-root", default=str(REPO_ROOT / DEFAULT_ROLLING_VINTAGE_DIR_REL))
    ap.add_argument("--if-due", action="store_true", help="Skip cleanly when the current 14-day scheduled block is already published")
    ap.add_argument("--force", action="store_true", help="Refit current active block (never overrides an abstained block lockout or manual revocation)")
    ap.add_argument("--resume-after-revoke", action="store_true", help="Explicitly clear a manual --revoke-reason halt (still never overrides an abstained block)")
    ap.add_argument("--revoke-reason", default="", help="Immediately publish durable status='revoked' transition for this book")
    args = ap.parse_args(argv)

    try:
        if args.revoke_reason:
            policy = load_rolling_policy(args.book_id, repo_root=REPO_ROOT, allow_demo_canary=True)
            vroot_p = _resolve_confined_operational_dir(
                args.vintage_root, repo_root=REPO_ROOT, allowed_parent_rel=Path("deriv_data"), label="--vintage-root"
            )
            book_vdir = _ensure_dir_nofollow(vroot_p / policy.book_id, mode=0o700, trusted_root=REPO_ROOT)
            with acquire_rolling_transition_lock(book_vdir, exclusive=True, nonblocking=False, trusted_root=REPO_ROOT) as rev_dir_fd:
                now_ep = int(time.time())
                sched_ep, next_ep = compute_scheduled_refit_epoch(
                    now_ep, origin_epoch=policy.cadence_origin_utc_epoch, cadence_days=policy.refit_cadence_days
                )
                rec = commit_rolling_transition(
                    book_vdir,
                    policy=policy,
                    status="revoked",
                    published_epoch=now_ep,
                    scheduled_refit_epoch=sched_ep,
                    next_refit_epoch=next_ep,
                    reason=str(args.revoke_reason).strip(),
                    trusted_root=REPO_ROOT,
                    already_locked=True,
                    dir_fd=rev_dir_fd,
                )
            print(json.dumps({"status": "revoked", "seq": rec["seq"], "transition_sha256": rec["transition_sha256"], "reason": rec["reason"]}, indent=2, sort_keys=True))
            return 0

        out = run_rolling_refit(
            book_id=args.book_id,
            repo_root=REPO_ROOT,
            archive_dir=Path(args.archive_dir),
            vintage_root=Path(args.vintage_root),
            if_due=bool(args.if_due),
            force=bool(args.force),
            resume_after_revoke=bool(args.resume_after_revoke),
        )
        print(json.dumps(out, indent=2, sort_keys=True))
        return 0
    except ArchiveLockBusyError as exc:
        print(json.dumps({"status": "archive_lock_busy", "error": str(exc)}, indent=2, sort_keys=True), file=sys.stderr)
        return 75
    except ArchiveIncompleteError as exc:
        print(json.dumps({"status": "archive_incomplete_or_invalid", "error": str(exc)}, indent=2, sort_keys=True), file=sys.stderr)
        return 4
    except ModelFitFailedError as exc:
        print(json.dumps({"status": "model_fit_failed", "error": str(exc)}, indent=2, sort_keys=True), file=sys.stderr)
        return 5
    except (ValueError, RollingVintageError) as exc:
        print(json.dumps({"status": "invalid_config_or_policy", "error": str(exc)}, indent=2, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
