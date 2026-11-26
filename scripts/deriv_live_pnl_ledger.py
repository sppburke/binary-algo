"""Per-book exact-cent VENUE_SETTLED P&L ledger and strictly prospective G6 cohort promotion gate.

Sibling to scripts/deriv_economics_ledger_v2.py (which stays byte-frozen for ROS-3A seals).
Supports:
- Strictly prospective Phase-1 G6 cohort evaluation (spec §3.5): canonical queue enforcement
  (`--db` diagnostic alias sets `promotion_eligible: False` on non-canonical paths), stored anchor
  matching (`registered_utc < cohort_start_utc <= eval_now_utc`) and frozen book manifest agreement,
  strict `venue_start_epoch >= cohort_start_epoch` membership, full contract-spec & Deriv
  `proposal_open_contract` venue-field agreement, closed-interval & signal-gap non-overlap checks,
  and exact `Fraction(s[124] + 39 * s[125], 40) / (n * 100) > 0` bootstrap over 5,000 default_rng(7) draws.
- Backwards-compatible legacy summary mode when invoked without prospective cohort parameters.
"""

from __future__ import annotations

import argparse
import errno
import hashlib
import json
import math
import os
import re
import sqlite3
import stat
import tempfile
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import quote as url_quote

import numpy as np
from scipy.stats import beta as scipy_beta, t as scipy_t

from deriv_runtime_core_v2 import SOTA_DERIV_SYMBOLS, contract_type_for, is_valid_demo_account_id

SETTLED_STATUSES = frozenset({"won", "lost", "sold", "expired", "cancelled"})
G6_ELIGIBLE_SETTLED_STATUSES = frozenset({"won", "lost", "sold"})
PROMOTION_HARD_MIN_TRADES = 50
MIN_PARITY_FILLS_FLOOR = 10
MIN_HORIZON_FILLS_FLOOR = 15
MAX_CLOCK_SKEW_S = 60
POLICY_SPRT_ALPHA_BROKEN = 0.05
POLICY_SPRT_BETA_BROKEN = 0.10
POLICY_SPRT_ALPHA_EV = 0.05
POLICY_SPRT_BETA_EV = 0.20
POLICY_SPRT_P0_EV = 1.0 / 1.78
POLICY_SPRT_H1_WR = 0.625
_VALID_BOOK_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_REQUIRED_CONTRACT_COLS = frozenset({
    "contract_id", "signal_id", "book_id", "pair", "horizon_minutes", "side",
    "buy_price_cents", "payout_cents", "sell_price_cents", "profit_cents",
    "status", "opened_utc", "closed_utc",
})
_REQUIRED_SIGNAL_COLS = frozenset({"signal_id", "book_id", "status"})
_REQUIRED_RISK_COLS = frozenset({"key", "value_json"})
_SAFE_MVT_REASON_CODES = frozenset({
    "invalid_eval_now_utc",
    "queue_not_v3",
    "non_canonical_db_diagnostic_only",
    "missing_book_id",
    "invalid_expected_stake_usd",
    "invalid_or_missing_cohort_start_utc",
    "unregistered_book_id",
    "expected_content_id_conflicts_with_index",
    "missing_expected_content_id",
    "queue_account_binding_missing_or_mismatch",
    "missing_stored_cohort_anchor",
    "invalid_anchor_clocks",
    "cohort_start_utc_mismatch_vs_anchor",
    "anchor_clock_ordering_invalid",
    "anchor_required_stake_cents_mismatch",
    "anchor_identity_mismatch",
    "anchor_spec_mismatch_vs_manifest",
    "unresolved_buy_signals",
    "min_trades_arg_below_hard_floor",
    "migrated_missing_closed_utc",
    "migrated_row_not_pre_anchor",
    "missing_or_invalid_venue_start_epoch",
    "missing_or_invalid_venue_end_epoch",
    "invalid_venue_end_epoch",
    "linked_signal_book_id_mismatch",
    "signal_close_utc_mismatch_or_invalid",
    "non_settled_or_ineligible_status",
    "missing_or_invalid_closed_utc",
    "closed_utc_precedes_venue_end",
    "migrated_legacy_in_cohort",
    "duplicate_contract_or_signal",
    "cost_unknown",
    "stake_not_authoritative",
    "missing_or_nonpositive_payout_cents",
    "economics_mismatch",
    "settlement_status_profit_mismatch",
    "queue_schema_error",
    "missing_required_queue_tables",
    "account_mismatch",
    "book_content_id_mismatch",
    "venue_duration_mismatch",
    "venue_terminal_status_mismatch",
    "linked_signal_spec_mismatch",
    "manifest_spec_mismatch",
    "venue_symbol_or_contract_type_mismatch",
    "invalid_signal_close_utc",
    "closed_interval_overlap",
    "signal_gap_below_horizon",
    "insufficient_prospective_trades",
    "ev_ci95_lo_not_positive",
    "cohort_not_yet_started",
    "invalid_prospective_anchor_order",
})
DEFAULT_PHASE1_PORTFOLIO_BOOKS = (
    "USDJPY.sota_ens5_m30_up.v1",
    "USDJPY.sota_stack05_m30_up.v1",
    "USDJPY.sota_ens5_m15_up.v1",
)
DEFAULT_MIN_PAYOUT_BY_HORIZON = {15: 1.76, 30: 1.78, 60: 1.78}
_ALLOWED_LOG_EVENTS = frozenset({
    "book_scored",
    "buy_confirmed",
    "buy_unknown",
    "buy_unknown_ambiguous_candidates",
    "buy_unknown_poc_incomplete",
    "buy_unknown_poc_verify_failed",
    "buy_unknown_recovered",
    "buy_unknown_recovery_conflict",
    "buy_unknown_unmatched",
    "buy_refused_pre_send",
    "contract_overdue_open",
    "contract_settled",
    "external_watermark_lag_skip",
    "feature_or_book_skip",
    "invalid_family_payload",
    "kill_switch_tripped",
    "multiplier_gate_decision",
    "multiplier_timed_sell_sent",
    "payout_gate_decision",
    "poc_contract_id_mismatch",
    "portfolio_reconcile_error",
    "proposal_failed",
    "quote_ask_blocked_risk",
    "reconcile_error",
    "reconcile_failed",
    "reconcile_settled",
    "signal_blocked_risk",
    "signal_rejected_post_quote",
    "signal_rejected_pre_proposal",
    "signal_skipped_pre_proposal",
    "venue_time_refresh_failed",
    "warmup_complete",
    "warmup_start",
})
REPO_ROOT = Path(__file__).resolve().parents[1]


def _redact_acct(account_id: str | None) -> str | None:
    raw = str(account_id or "").strip()
    if not raw:
        return None
    return f"sha256:{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:12]}"


def _parse_iso_utc(val: str | None) -> datetime | None:
    if not val or not isinstance(val, str):
        return None
    try:
        dt_v = datetime.fromisoformat(val.replace("Z", "+00:00"))
        if dt_v.tzinfo is None:
            return None
        return dt_v.astimezone(timezone.utc)
    except Exception:
        return None


def _parse_whole_second_epoch(val: str | None) -> int | None:
    dt_v = _parse_iso_utc(val)
    if dt_v is None or dt_v.microsecond != 0:
        return None
    return int(dt_v.timestamp())


def _load_registered_books(repo_root: Path | None) -> dict[str, dict[str, Any]]:
    if repo_root is None:
        return {}
    idx_path = repo_root / "books" / "SOTA_INDEX.json"
    if not idx_path.is_file():
        return {}
    try:
        obj = json.loads(idx_path.read_text(encoding="utf-8"))
        out: dict[str, dict[str, Any]] = {}
        for b in obj.get("books") or []:
            if isinstance(b, dict) and b.get("id") and b.get("content_id"):
                bid = str(b["id"])
                ent = dict(b)
                man_path = repo_root / "books" / bid / f"{bid}.manifest.json"
                strat_path = repo_root / "books" / bid / f"{bid}_strategy.json"
                if man_path.is_file():
                    ent["_manifest"] = json.loads(man_path.read_text(encoding="utf-8"))
                if strat_path.is_file():
                    ent["_strategy"] = json.loads(strat_path.read_text(encoding="utf-8"))
                out[bid] = ent
        return out
    except Exception:
        return {}


def _bootstrap_ci95_float(values: np.ndarray, n_boot: int = 5000, seed: int = 7) -> tuple[float, float]:
    if len(values) < 5:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    n = len(values)
    means = np.array([values[rng.integers(0, n, n)].mean() for _ in range(n_boot)], dtype=np.float64)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def exact_fraction_bootstrap_ci95(profits_cents: list[int], stake_cents: int = 100, *, n_boot: int = 5000, seed: int = 7) -> tuple[Fraction, Fraction]:
    """Exact Fraction linear 2.5th & 97.5th percentile bootstrap (pure Python ints — zero int64 overflow!)."""
    n = len(profits_cents)
    if n < 1 or stake_cents <= 0:
        raise ValueError("exact_fraction_bootstrap_ci95 requires n >= 1 and stake_cents > 0")
    py_profits = [int(x) for x in profits_cents]
    rng = np.random.default_rng(seed)
    # Sum profits strictly in unbounded Python ints per bootstrap draw (immune to NumPy int64 overflow!)
    sums: list[int] = [sum(py_profits[i] for i in rng.integers(0, n, n).tolist()) for _ in range(n_boot)]
    sums.sort()
    # Linear percentile index for 2.5% of 5000: 0.025 * 4999 = 124.975 = 124 + 39/40 -> (s[124] + 39*s[125]) / 40
    # Linear percentile index for 97.5% of 5000: 0.975 * 4999 = 4874.025 = 4874 + 1/40 -> (39*s[4874] + s[4875]) / 40
    denom = n * int(stake_cents)
    if n_boot == 5000:
        lo = Fraction(sums[124] + 39 * sums[125], 40 * denom)
        hi = Fraction(39 * sums[4874] + sums[4875], 40 * denom)
    else:
        pos_lo = Fraction(25, 1000) * (n_boot - 1)
        i_lo = int(pos_lo)
        f_lo = pos_lo - i_lo
        lo = (Fraction(sums[i_lo]) * (1 - f_lo) + Fraction(sums[min(i_lo + 1, n_boot - 1)]) * f_lo) / denom
        pos_hi = Fraction(975, 1000) * (n_boot - 1)
        i_hi = int(pos_hi)
        f_hi = pos_hi - i_hi
        hi = (Fraction(sums[i_hi]) * (1 - f_hi) + Fraction(sums[min(i_hi + 1, n_boot - 1)]) * f_hi) / denom
    return lo, hi


class _SymlinkRefusedError(OSError):
    pass


class _MissingPathError(FileNotFoundError):
    pass


def _strict_int_or_none(val: Any) -> int | None:
    return val if type(val) is int else None


def _sanitize_mvt_reason_code(reason: str) -> str:
    """Map an internal G6/queue reason to a fixed allowlisted code without echoing raw IDs or DB values."""
    s = str(reason or "").strip()
    if s in _SAFE_MVT_REASON_CODES:
        return s
    # Prefix form: e.g. 'closed_interval_overlap:cid1:cid2' or 'insufficient_prospective_trades:0<50'
    head = s.split(":", 1)[0]
    if head in _SAFE_MVT_REASON_CODES:
        return head
    # Suffix form: e.g. '<contract_id>:missing_or_invalid_venue_start_epoch' or '<cid>:non_settled_or_ineligible_status:open'
    parts = s.split(":")
    if len(parts) >= 2 and parts[1] in _SAFE_MVT_REASON_CODES:
        return parts[1]
    return "unclassified_integrity_anomaly"


_UNSET_TRUSTED_ROOT = object()


def _open_dir_nofollow(
    dir_path: Path,
    *,
    trusted_root: Path | None = None,
    resolved_trusted_root: Any = _UNSET_TRUSTED_ROOT,
) -> tuple[int, Path]:
    """Open dir_path strictly with component-by-component O_NOFOLLOW from trusted_root (resolved once) or '/'."""
    p_raw = Path(dir_path)
    if ".." in p_raw.parts:
        raise _SymlinkRefusedError("dotdot_path_refused")
    p_abs = p_raw if p_raw.is_absolute() else (Path.cwd() / p_raw)
    if ".." in p_abs.parts:
        raise _SymlinkRefusedError("dotdot_path_refused")

    anchor = Path("/")
    rel_parts: list[str] = list(p_abs.parts[1:])
    if trusted_root is not None:
        tr_raw = Path(trusted_root)
        tr_abs = tr_raw if tr_raw.is_absolute() else (Path.cwd() / tr_raw)
        if resolved_trusted_root is not _UNSET_TRUSTED_ROOT:
            tr_res: Path | None = resolved_trusted_root
        else:
            try:
                tr_res = tr_abs.resolve(strict=True)
            except OSError:
                tr_res = None
        if tr_res is not None:
            matched = False
            for cand in (tr_abs, tr_res):
                try:
                    rel = p_abs.relative_to(cand)
                    anchor = tr_res
                    rel_parts = list(rel.parts)
                    matched = True
                    break
                except ValueError:
                    continue
            if not matched:
                anchor = Path("/")
                rel_parts = list(p_abs.parts[1:])

    try:
        cur_fd = os.open(str(anchor), os.O_RDONLY | os.O_DIRECTORY)
    except OSError as exc:
        if exc.errno == errno.ENOENT:
            raise _MissingPathError(str(anchor)) from exc
        raise
    try:
        for comp in rel_parts:
            if comp in ("", "."):
                continue
            if comp == "..":
                raise _SymlinkRefusedError("dotdot_component_refused")
            try:
                nxt_fd = os.open(comp, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=cur_fd)
            except OSError as exc:
                if exc.errno == errno.ENOENT:
                    raise _MissingPathError(comp) from exc
                if exc.errno in (errno.ELOOP, errno.ENOTDIR):
                    raise _SymlinkRefusedError(comp) from exc
                raise
            os.close(cur_fd)
            cur_fd = nxt_fd
        return cur_fd, anchor.joinpath(*rel_parts)
    except Exception:
        os.close(cur_fd)
        raise


def _copy_fd_to_path(src_fd: int, dst_path: Path) -> None:
    os.lseek(src_fd, 0, os.SEEK_SET)
    with dst_path.open("wb") as out_fh:
        while True:
            chunk = os.read(src_fd, 1 << 20)
            if not chunk:
                break
            out_fh.write(chunk)


def _stat_sig(st: os.stat_result) -> tuple[int, int, int]:
    return (int(st.st_ino), int(st.st_mtime_ns), int(st.st_size))


def _read_db_snapshot(db_path: Path, *, trusted_root: Path | None = None) -> dict[str, Any]:
    """Read ONE consistent read-only SQLite snapshot without creating ANY -wal/-shm sidecars in db_path.parent."""
    p_raw = Path(db_path)
    if ".." in p_raw.parts or not p_raw.name or p_raw.name in (".", ".."):
        raise _SymlinkRefusedError("invalid_db_path")
    p_abs = p_raw if p_raw.is_absolute() else (Path.cwd() / p_raw)
    fname = p_abs.name
    tr_resolved: Path | None = None
    if trusted_root is not None:
        tr_raw = Path(trusted_root)
        tr_abs = tr_raw if tr_raw.is_absolute() else (Path.cwd() / tr_raw)
        try:
            tr_resolved = tr_abs.resolve(strict=True)
        except OSError:
            tr_resolved = None
    parent_fd, resolved_parent = _open_dir_nofollow(
        p_abs.parent, trusted_root=trusted_root, resolved_trusted_root=tr_resolved
    )

    def _inspect_open_leaf() -> tuple[int, os.stat_result]:
        try:
            fd = os.open(fname, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent_fd)
        except OSError as exc:
            if exc.errno == errno.ENOENT:
                raise _MissingPathError(fname) from exc
            if exc.errno == errno.ELOOP:
                raise _SymlinkRefusedError(fname) from exc
            raise
        try:
            st = os.fstat(fd)
            if not stat.S_ISREG(st.st_mode):
                raise ValueError("queue_db_not_regular_file")
            return fd, st
        except Exception:
            os.close(fd)
            raise

    def _open_optional_sidecar(sfx: str) -> tuple[int | None, os.stat_result | None]:
        sname = f"{fname}{sfx}"
        try:
            sfd = os.open(sname, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent_fd)
        except OSError as exc:
            if exc.errno == errno.ENOENT:
                return None, None
            if exc.errno == errno.ELOOP:
                raise _SymlinkRefusedError(sname) from exc
            raise
        try:
            sst = os.fstat(sfd)
            if not stat.S_ISREG(sst.st_mode):
                raise ValueError("invalid_sidecar_file")
            return sfd, sst
        except Exception:
            os.close(sfd)
            raise

    def _query_conn(conn: sqlite3.Connection) -> dict[str, Any]:
        conn.row_factory = sqlite3.Row
        conn.execute("BEGIN")
        user_ver = int(conn.execute("PRAGMA user_version").fetchone()[0])
        tables = {str(r[0]) for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        missing_tables = sorted({"sota_contracts", "sota_signals"} - tables)
        schema_errors: list[str] = []
        all_contracts: list[dict[str, Any]] = []
        unresolved_sigs: list[dict[str, Any]] = []
        if not missing_tables:
            con_cols = {str(r["name"]) for r in conn.execute("PRAGMA table_info(sota_contracts)").fetchall()}
            sig_cols = {str(r["name"]) for r in conn.execute("PRAGMA table_info(sota_signals)").fetchall()}
            if not _REQUIRED_CONTRACT_COLS.issubset(con_cols):
                schema_errors.append("missing_required_contract_columns")
            if not _REQUIRED_SIGNAL_COLS.issubset(sig_cols):
                schema_errors.append("missing_required_signal_columns")
            if not schema_errors:
                def _col(name: str, alias: str | None = None) -> str:
                    al = alias or name
                    return f"c.{name} AS {al}" if name in con_cols else f"NULL AS {al}"

                sig_join_fields = (
                    "s.book_id AS sig_book_id, s.pair AS sig_pair, s.horizon_minutes AS sig_horizon_minutes, s.side AS sig_side, s.signal_close_utc AS sig_signal_close_utc"
                    if {"book_id", "pair", "horizon_minutes", "side", "signal_close_utc"}.issubset(sig_cols)
                    else "NULL AS sig_book_id, NULL AS sig_pair, NULL AS sig_horizon_minutes, NULL AS sig_side, NULL AS sig_signal_close_utc"
                )
                rows = conn.execute(
                    f"""
                    SELECT c.contract_id, c.signal_id, c.book_id, c.pair, c.horizon_minutes, c.side,
                           c.buy_price_cents, c.payout_cents, c.sell_price_cents, c.profit_cents,
                           c.status, c.opened_utc, c.closed_utc,
                           {_col('account_mode')}, {_col('account_id')}, {_col('book_content_id')},
                           {_col('contract_family')}, {_col('is_migrated_legacy')}, {_col('cost_unknown')},
                           {_col('stake_authoritative')}, {_col('venue_start_epoch')}, {_col('venue_end_epoch')},
                           {_col('venue_terminal_status')}, {_col('venue_underlying_symbol')},
                           {_col('venue_contract_type')}, {_col('signal_close_utc')},
                           {sig_join_fields}
                    FROM sota_contracts c
                    LEFT JOIN sota_signals s ON s.signal_id = c.signal_id
                    ORDER BY c.opened_utc ASC, c.contract_id ASC
                    """
                ).fetchall()
                all_contracts = [dict(r) for r in rows]
                sig_rows = conn.execute(
                    """
                    SELECT signal_id, book_id, status FROM sota_signals
                    WHERE status IN ('buy_intent', 'buy_unknown')
                    """
                ).fetchall()
                unresolved_sigs = [dict(r) for r in sig_rows]
        risk_state: dict[str, Any] = {}
        risk_state_keys: set[str] = set()
        if "sota_risk_state" in tables:
            risk_cols = {str(r["name"]) for r in conn.execute("PRAGMA table_info(sota_risk_state)").fetchall()}
            if not _REQUIRED_RISK_COLS.issubset(risk_cols):
                schema_errors.append("missing_required_risk_state_columns")
            else:
                for rr in conn.execute("SELECT key, value_json FROM sota_risk_state").fetchall():
                    k_str = str(rr["key"])
                    risk_state_keys.add(k_str)
                    try:
                        risk_state[k_str] = json.loads(str(rr["value_json"]))
                    except Exception:
                        risk_state[k_str] = {"_malformed_json": True}
        conn.execute("COMMIT")
        return {
            "user_version": user_ver,
            "tables": tables,
            "missing_tables": missing_tables,
            "schema_errors": schema_errors,
            "all_contracts": all_contracts,
            "unresolved_sigs": unresolved_sigs,
            "risk_state": risk_state,
            "risk_state_keys": risk_state_keys,
        }

    try:
        # ALWAYS copy directly from the held O_NOFOLLOW file descriptors into an ephemeral private directory!
        # Guarantees ZERO TOCTOU pathname re-open AND ZERO -wal/-shm sidecar creation in db_path.parent!
        for _attempt in range(3):
            db_fd, st_db_pre = _inspect_open_leaf()
            wal_fd = None
            jrn_fd = None
            try:
                wal_fd, st_wal_pre = _open_optional_sidecar("-wal")
                jrn_fd, st_jrn_pre = _open_optional_sidecar("-journal")
                has_sidecar = (wal_fd is not None) or (jrn_fd is not None)

                with tempfile.TemporaryDirectory(prefix="sota_ro_snap_") as snap_td:
                    tmp_dir = Path(snap_td)
                    tmp_db = tmp_dir / "snapshot.sqlite"
                    _copy_fd_to_path(db_fd, tmp_db)
                    if wal_fd is not None:
                        _copy_fd_to_path(wal_fd, tmp_dir / "snapshot.sqlite-wal")
                    if jrn_fd is not None:
                        _copy_fd_to_path(jrn_fd, tmp_dir / "snapshot.sqlite-journal")

                    # Verify DB + sidecars did not rotate/checkpoint mid-copy
                    st_db_post = os.stat(fname, dir_fd=parent_fd, follow_symlinks=False)
                    if _stat_sig(st_db_post) != _stat_sig(st_db_pre):
                        continue
                    if st_wal_pre is not None:
                        try:
                            st_wal_post = os.stat(f"{fname}-wal", dir_fd=parent_fd, follow_symlinks=False)
                        except OSError:
                            continue
                        if _stat_sig(st_wal_post) != _stat_sig(st_wal_pre):
                            continue
                    if st_jrn_pre is not None:
                        try:
                            st_jrn_post = os.stat(f"{fname}-journal", dir_fd=parent_fd, follow_symlinks=False)
                        except OSError:
                            continue
                        if _stat_sig(st_jrn_post) != _stat_sig(st_jrn_pre):
                            continue

                    encoded_tmp = url_quote(str(tmp_db), safe="/")
                    uri_flags = "mode=ro" if has_sidecar else "mode=ro&immutable=1"
                    conn = sqlite3.connect(f"file:{encoded_tmp}?{uri_flags}", uri=True, isolation_level=None)
                    try:
                        out_snap = _query_conn(conn)
                        out_snap["resolved_db_path"] = resolved_parent / fname
                        out_snap["db_dev_ino"] = (int(st_db_pre.st_dev), int(st_db_pre.st_ino))
                        out_snap["trusted_root_resolved"] = tr_resolved
                        return out_snap
                    finally:
                        conn.close()
            finally:
                if db_fd >= 0:
                    os.close(db_fd)
                if wal_fd is not None:
                    os.close(wal_fd)
                if jrn_fd is not None:
                    os.close(jrn_fd)
        raise sqlite3.OperationalError("snapshot_unstable_during_concurrent_wal_checkpoint")
    finally:
        os.close(parent_fd)


def _verify_snapshot_is_canonical(
    snap: dict[str, Any],
    canonical_db_path: Path,
    *,
    trusted_root: Path | None,
) -> bool:
    """Verify the already-opened snapshot matches canonical_db_path via O_NOFOLLOW stat (st_dev, st_ino) + resolved path."""
    try:
        c_pfd, c_rpar = _open_dir_nofollow(
            canonical_db_path.parent,
            trusted_root=trusted_root,
            resolved_trusted_root=snap.get("trusted_root_resolved"),
        )
        try:
            c_st = os.stat(canonical_db_path.name, dir_fd=c_pfd, follow_symlinks=False)
            canon_resolved = c_rpar / canonical_db_path.name
            return bool(
                stat.S_ISREG(c_st.st_mode)
                and snap.get("resolved_db_path") == canon_resolved
                and snap.get("db_dev_ino") == (int(c_st.st_dev), int(c_st.st_ino))
            )
        finally:
            os.close(c_pfd)
    except (OSError, _SymlinkRefusedError, _MissingPathError):
        return False


def summarize_settled_queue(
    db_path: Path,
    *,
    min_settled_for_promotion: int = 50,
    book_id: str | None = None,
    expected_content_id: str | None = None,
    repo_root: Path | None = REPO_ROOT,
    require_registered_book: bool = False,
    cohort_start_utc: str | None = None,
    expected_account_id: str | None = None,
    expected_stake_usd: str | float | None = None,
    eval_now_utc: str | None = None,
    canonical_db_path: Path | None = None,
    enforce_canonical_db: bool = False,
) -> dict[str, Any]:
    """Public per-book settled queue evaluator; always acquires its own verified read-only snapshot from db_path."""
    return _summarize_settled_queue_impl(
        db_path,
        min_settled_for_promotion=min_settled_for_promotion,
        book_id=book_id,
        expected_content_id=expected_content_id,
        repo_root=repo_root,
        require_registered_book=require_registered_book,
        cohort_start_utc=cohort_start_utc,
        expected_account_id=expected_account_id,
        expected_stake_usd=expected_stake_usd,
        eval_now_utc=eval_now_utc,
        canonical_db_path=canonical_db_path,
        enforce_canonical_db=enforce_canonical_db,
        _snapshot=None,
    )


def _summarize_settled_queue_impl(
    db_path: Path,
    *,
    min_settled_for_promotion: int = 50,
    book_id: str | None = None,
    expected_content_id: str | None = None,
    repo_root: Path | None = REPO_ROOT,
    require_registered_book: bool = False,
    cohort_start_utc: str | None = None,
    expected_account_id: str | None = None,
    expected_stake_usd: str | float | None = None,
    eval_now_utc: str | None = None,
    canonical_db_path: Path | None = None,
    enforce_canonical_db: bool = False,
    _snapshot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate per-book settled queue; enforces full prospective G6 contract when cohort_start_utc or v3 anchor mode is requested."""
    prospective_mode = bool(
        cohort_start_utc is not None
        or expected_account_id is not None
        or expected_stake_usd is not None
        or eval_now_utc is not None
        or enforce_canonical_db
    )
    schema = "deriv-sota-live-pnl-ledger/v3" if prospective_mode else "deriv-sota-live-pnl-ledger/v2"
    db_p = Path(db_path)
    if _snapshot is None:
        if db_p.is_symlink():
            return {
                "schema": schema,
                "db_path": str(db_p),
                "book_id": book_id,
                "settled_contracts": 0,
                "open_contracts": 0,
                "unresolved_signals": 0,
                "promotion_eligible_books": [],
                "promotion_eligible": False if book_id else None,
                "reasons": ["queue_db_symlink_refused"],
                "by_book": {},
            }
        if not db_p.exists():
            return {
                "schema": schema,
                "db_path": str(db_p),
                "book_id": book_id,
                "settled_contracts": 0,
                "open_contracts": 0,
                "unresolved_signals": 0,
                "promotion_eligible_books": [],
                "promotion_eligible": False if book_id else None,
                "reasons": ["queue_db_missing"],
                "by_book": {},
            }
        if not db_p.is_file():
            return {
                "schema": schema,
                "db_path": str(db_p),
                "book_id": book_id,
                "settled_contracts": 0,
                "open_contracts": 0,
                "unresolved_signals": 0,
                "promotion_eligible_books": [],
                "promotion_eligible": False if book_id else None,
                "reasons": ["queue_db_not_regular_file"],
                "by_book": {},
            }

    registered_books = _load_registered_books(repo_root)
    registered_cids = {k: str(v["content_id"]) for k, v in registered_books.items()}

    if _snapshot is not None:
        snap = _snapshot
    else:
        try:
            snap = _read_db_snapshot(db_p, trusted_root=repo_root)
        except _SymlinkRefusedError:
            return {
                "schema": schema,
                "db_path": str(db_p),
                "book_id": book_id,
                "settled_contracts": 0,
                "open_contracts": 0,
                "unresolved_signals": 0,
                "promotion_eligible_books": [],
                "promotion_eligible": False if book_id else None,
                "reasons": ["queue_db_symlink_refused"],
                "by_book": {},
            }
        except _MissingPathError:
            return {
                "schema": schema,
                "db_path": str(db_p),
                "book_id": book_id,
                "settled_contracts": 0,
                "open_contracts": 0,
                "unresolved_signals": 0,
                "promotion_eligible_books": [],
                "promotion_eligible": False if book_id else None,
                "reasons": ["queue_db_missing"],
                "by_book": {},
            }
        except (sqlite3.Error, OSError, ValueError):
            return {
                "schema": schema,
                "db_path": str(db_p),
                "book_id": book_id,
                "settled_contracts": 0,
                "open_contracts": 0,
                "unresolved_signals": 0,
                "promotion_eligible_books": [],
                "promotion_eligible": False if book_id else None,
                "reasons": ["queue_db_schema_or_read_error"],
                "by_book": {},
            }
    user_ver = int(snap["user_version"])
    all_contracts = list(snap["all_contracts"])
    unresolved_sigs = list(snap["unresolved_sigs"])
    risk_state = dict(snap["risk_state"])
    missing_tables = list(snap.get("missing_tables") or [])
    schema_errors = list(snap.get("schema_errors") or [])

    # ------------------------------------------------------------------
    # Strictly Prospective G6 Mode (spec §3.5)
    # ------------------------------------------------------------------
    if prospective_mode:
        eval_dt = _parse_iso_utc(eval_now_utc) if eval_now_utc is not None else datetime.now(timezone.utc)
        blockers_top: list[str] = []
        if eval_dt is None:
            blockers_top.append("invalid_eval_now_utc")
        if missing_tables:
            blockers_top.append("missing_required_queue_tables")
        if schema_errors:
            blockers_top.append("queue_schema_error")
        if user_ver != 3:
            blockers_top.append(f"queue_not_v3:user_version={user_ver}")
        if enforce_canonical_db:
            canon = canonical_db_path if canonical_db_path is not None else (
                (repo_root or REPO_ROOT) / "deriv_data" / "runtime" / "sota_trade_queue_demo.sqlite"
            )
            if not _verify_snapshot_is_canonical(snap, canon, trusted_root=repo_root):
                blockers_top.append("non_canonical_db_diagnostic_only")

        if not book_id:
            blockers_top.append("missing_book_id")
            return {
                "schema": schema,
                "db_path": str(db_p),
                "book_id": None,
                "promotion_eligible_books": [],
                "promotion_eligible": False,
                "reasons": blockers_top,
                "by_book": {},
            }

        # Validate expected_stake_usd == 1.00 (100 exact cents)
        expected_stake_cents = 100
        if expected_stake_usd is not None:
            try:
                sf = Fraction(str(expected_stake_usd).strip())
                if sf * 100 != 100:
                    blockers_top.append(f"invalid_expected_stake_usd:{expected_stake_usd}")
            except Exception:
                blockers_top.append(f"invalid_expected_stake_usd:{expected_stake_usd}")

        cohort_ep = _parse_whole_second_epoch(cohort_start_utc)
        if cohort_ep is None:
            blockers_top.append("invalid_or_missing_cohort_start_utc")
        eval_ep = int(eval_dt.timestamp()) if eval_dt is not None else None

        reg_ent = registered_books.get(book_id)
        reg_cid = registered_cids.get(book_id)
        if require_registered_book and not reg_ent:
            blockers_top.append("unregistered_book_id")
        want_cid = reg_cid or expected_content_id
        if reg_cid and expected_content_id and expected_content_id != reg_cid:
            blockers_top.append(f"expected_content_id_conflicts_with_index:{expected_content_id}!={reg_cid}")
        if not want_cid:
            blockers_top.append("missing_expected_content_id")

        binding = risk_state.get("queue_account_binding")
        acct_id = expected_account_id or (binding.get("account_id") if isinstance(binding, dict) else None)
        if (
            not isinstance(binding, dict)
            or not acct_id
            or not is_valid_demo_account_id(str(acct_id))
            or binding.get("account_id") != acct_id
            or str(binding.get("account_mode") or "").lower() != "demo"
        ):
            blockers_top.append("queue_account_binding_missing_or_mismatch")

        # Validate stored prospective cohort anchor
        anchor_key = f"g6_cohort_anchor:{acct_id}:{book_id}:{want_cid}" if (acct_id and want_cid) else ""
        anchor = risk_state.get(anchor_key) if anchor_key else None
        if not isinstance(anchor, dict):
            blockers_top.append("missing_stored_cohort_anchor")
        else:
            reg_ep = _parse_whole_second_epoch(anchor.get("registered_utc"))
            anc_start_ep = _parse_whole_second_epoch(anchor.get("cohort_start_utc"))
            if reg_ep is None or anc_start_ep is None:
                blockers_top.append("invalid_anchor_clocks")
            elif cohort_ep is not None and anc_start_ep != cohort_ep:
                blockers_top.append("cohort_start_utc_mismatch_vs_anchor")
            elif eval_ep is not None and not (reg_ep < anc_start_ep <= eval_ep):
                blockers_top.append("anchor_clock_ordering_invalid")
            if anchor.get("required_stake_cents") != expected_stake_cents:
                blockers_top.append("anchor_required_stake_cents_mismatch")
            if anchor.get("account_id") != acct_id or anchor.get("book_id") != book_id or anchor.get("book_content_id") != want_cid:
                blockers_top.append("anchor_identity_mismatch")

        # Expected manifest spec for contract-spec verification
        exp_pair = None
        exp_horizon = None
        exp_sides: set[str] | None = None
        exp_family = None
        if reg_ent:
            man = reg_ent.get("_manifest") or reg_ent
            strat = reg_ent.get("_strategy") or {}
            exp_pair = str(man.get("pair") or "").upper() or None
            exp_horizon = int(man["horizon_minutes"]) if man.get("horizon_minutes") is not None else None
            exp_sides = {str(s).upper() for s in (man.get("allowed_sides") or [])} or None
            exp_family = str(strat.get("contract_family", "risefall"))
            if isinstance(anchor, dict):
                if (
                    str(anchor.get("pair") or "").upper() != exp_pair
                    or anchor.get("horizon_minutes") != exp_horizon
                    or (exp_sides is not None and str(anchor.get("side") or "").upper() not in exp_sides)
                    or str(anchor.get("contract_family") or "") != exp_family
                ):
                    blockers_top.append("anchor_spec_mismatch_vs_manifest")

        b_contracts = [c for c in all_contracts if str(c["book_id"]) == book_id]
        b_uncertain = [s for s in unresolved_sigs if str(s["book_id"]) == book_id]
        if b_uncertain:
            blockers_top.append(f"unresolved_buy_signals:{len(b_uncertain)}")

        effective_min = max(PROMOTION_HARD_MIN_TRADES, int(min_settled_for_promotion))
        if int(min_settled_for_promotion) < PROMOTION_HARD_MIN_TRADES:
            blockers_top.append(f"min_trades_arg_below_hard_floor:{min_settled_for_promotion}<{PROMOTION_HARD_MIN_TRADES}")

        cohort_rows: list[dict[str, Any]] = []
        excluded_pre_anchor = 0
        excluded_post_eval = 0
        seen_cids: set[str] = set()
        seen_sids: set[str] = set()
        anomalies: list[str] = []
        anomaly_codes: list[str] = []

        def _add_anom(cid_s: str, code_s: str, suffix: str | None = None) -> None:
            anomaly_codes.append(code_s)
            anomalies.append(f"{cid_s}:{code_s}:{suffix}" if suffix is not None else f"{cid_s}:{code_s}")

        for c in b_contracts:
            cid = str(c["contract_id"])
            sid = str(c["signal_id"])
            st = str(c["status"] or "")
            raw_mig = c["is_migrated_legacy"]
            is_mig = 0 if raw_mig is None else _strict_int_or_none(raw_mig)
            if is_mig is None:
                _add_anom(cid, "migrated_legacy_in_cohort")
                continue
            v_start = c["venue_start_epoch"]
            v_end = c["venue_end_epoch"]
            closed_dt = _parse_iso_utc(c["closed_utc"])

            # Validate clocks & joined signal identity/clocks BEFORE pre-anchor exclusion (BLOCKER #5)!
            if is_mig == 1 and v_start is None:
                if closed_dt is None:
                    _add_anom(cid, "migrated_missing_closed_utc")
                    continue
                if cohort_ep is not None and int(closed_dt.timestamp()) < cohort_ep:
                    excluded_pre_anchor += 1
                    continue
                _add_anom(cid, "migrated_row_not_pre_anchor")
                continue

            if _strict_int_or_none(v_start) is None or v_start <= 0:
                _add_anom(cid, "missing_or_invalid_venue_start_epoch")
                continue
            if st in G6_ELIGIBLE_SETTLED_STATUSES and (_strict_int_or_none(v_end) is None or v_end < v_start):
                _add_anom(cid, "missing_or_invalid_venue_end_epoch")
                continue
            if v_end is not None and (_strict_int_or_none(v_end) is None or v_end < v_start):
                _add_anom(cid, "invalid_venue_end_epoch")
                continue
            if c["sig_book_id"] is None or str(c["sig_book_id"]) != str(c["book_id"]):
                _add_anom(cid, "linked_signal_book_id_mismatch")
                continue
            c_sc_raw = c["signal_close_utc"]
            s_sc_raw = c["sig_signal_close_utc"]
            if not c_sc_raw or not s_sc_raw or str(c_sc_raw) != str(s_sc_raw) or _parse_iso_utc(c_sc_raw) is None:
                _add_anom(cid, "signal_close_utc_mismatch_or_invalid")
                continue

            # Pre-anchor exclusion (including boundary-crossing contracts started < cohort_start_epoch!)
            if cohort_ep is not None and v_start < cohort_ep:
                excluded_pre_anchor += 1
                continue

            # Non-excluded contract in cohort window [cohort_start_epoch, ...)
            if st not in G6_ELIGIBLE_SETTLED_STATUSES:
                _add_anom(cid, "non_settled_or_ineligible_status", st)
                continue
            if closed_dt is None:
                _add_anom(cid, "missing_or_invalid_closed_utc")
                continue
            closed_ep = int(closed_dt.timestamp())
            # Allow bounded <=60s NTP skew between VPS wall clock and Deriv venue_end_epoch,
            # while catching forged/impossible closed_utc << venue_end_epoch!
            if closed_ep < (v_end - MAX_CLOCK_SKEW_S):
                _add_anom(cid, "closed_utc_precedes_venue_end")
                continue
            if eval_dt is not None and (closed_dt > eval_dt or (eval_ep is not None and v_end > eval_ep)):
                excluded_post_eval += 1
                continue

            if is_mig != 0:
                _add_anom(cid, "migrated_legacy_in_cohort")
                continue
            if cid in seen_cids or sid in seen_sids:
                _add_anom(cid, "duplicate_contract_or_signal")
                continue
            seen_cids.add(cid)
            seen_sids.add(sid)

            # Full provenance, economics & contract-spec verification (strict int types — reject SQLite text-in-INT!)
            if _strict_int_or_none(c["cost_unknown"]) != 0:
                _add_anom(cid, "cost_unknown")
                continue
            if _strict_int_or_none(c["stake_authoritative"]) != 1:
                _add_anom(cid, "stake_not_authoritative")
                continue
            po_c = _strict_int_or_none(c["payout_cents"])
            if po_c is None or po_c <= 0:
                _add_anom(cid, "missing_or_nonpositive_payout_cents")
                continue
            bp = _strict_int_or_none(c["buy_price_cents"])
            sp_c = _strict_int_or_none(c["sell_price_cents"])
            pf = _strict_int_or_none(c["profit_cents"])
            if bp != expected_stake_cents or sp_c is None or pf is None or (sp_c - bp != pf):
                _add_anom(cid, "economics_mismatch")
                continue
            st_low = st.lower()
            if st_low == "won":
                if sp_c != po_c or sp_c <= bp or pf <= 0:
                    _add_anom(cid, "settlement_status_profit_mismatch")
                    continue
            elif st_low == "lost":
                if sp_c != 0 or pf != -bp:
                    _add_anom(cid, "settlement_status_profit_mismatch")
                    continue
            elif st_low == "sold":
                if sp_c != bp or pf != 0:
                    _add_anom(cid, "settlement_status_profit_mismatch")
                    continue
            if c["account_id"] != acct_id or str(c["account_mode"] or "").lower() != "demo":
                _add_anom(cid, "account_mismatch")
                continue
            if want_cid and c["book_content_id"] != want_cid:
                _add_anom(cid, "book_content_id_mismatch")
                continue
            if _strict_int_or_none(v_end) is None or v_end < v_start:
                _add_anom(cid, "invalid_venue_end_epoch")
                continue
            hm = _strict_int_or_none(c["horizon_minutes"])
            if hm is None or hm <= 0 or (v_end - v_start) != hm * 60:
                _add_anom(cid, "venue_duration_mismatch", f"{v_end - v_start}!={hm * 60 if hm is not None else 'bad'}")
                continue
            if str(c["venue_terminal_status"] or "").lower() != st.lower():
                _add_anom(cid, "venue_terminal_status_mismatch")
                continue

            c_pair = str(c["pair"] or "").upper()
            c_side = str(c["side"] or "").upper()
            c_fam = str(c["contract_family"] or "")
            # Linked signal agreement
            if (
                str(c["sig_pair"] or "").upper() != c_pair
                or _strict_int_or_none(c["sig_horizon_minutes"]) != hm
                or str(c["sig_side"] or "").upper() != c_side
            ):
                _add_anom(cid, "linked_signal_spec_mismatch")
                continue
            # Manifest agreement
            if (
                (exp_pair is not None and c_pair != exp_pair)
                or (exp_horizon is not None and hm != exp_horizon)
                or (exp_sides is not None and c_side not in exp_sides)
                or (exp_family is not None and c_fam != exp_family)
            ):
                _add_anom(cid, "manifest_spec_mismatch")
                continue
            # Venue symbol & contract_type agreement
            want_sym = SOTA_DERIV_SYMBOLS.get(c_pair)
            try:
                want_ctype = contract_type_for(c_side, family=c_fam)
            except Exception:
                want_ctype = None
            if c["venue_underlying_symbol"] != want_sym or str(c["venue_contract_type"] or "").upper() != want_ctype:
                _add_anom(cid, "venue_symbol_or_contract_type_mismatch")
                continue
            sig_close_dt = _parse_iso_utc(c["signal_close_utc"] or c["sig_signal_close_utc"])
            if sig_close_dt is None:
                _add_anom(cid, "invalid_signal_close_utc")
                continue

            cohort_rows.append(
                {
                    "contract_id": cid,
                    "v_start": v_start,
                    "v_end": v_end,
                    "sig_close_ep": int(sig_close_dt.timestamp()),
                    "horizon_s": hm * 60,
                    "buy_price_cents": bp,
                    "payout_cents": po_c,
                    "profit_cents": pf,
                }
            )

        # Closed-interval overlap & signal-gap < horizon_minutes * 60 check
        cohort_rows.sort(key=lambda r: (r["v_start"], r["v_end"], r["contract_id"]))
        for a, b in zip(cohort_rows, cohort_rows[1:]):
            if b["v_start"] <= a["v_end"]:
                anomaly_codes.append("closed_interval_overlap")
                anomalies.append(f"closed_interval_overlap:{a['contract_id']}:{b['contract_id']}")
        by_sig = sorted(cohort_rows, key=lambda r: (r["sig_close_ep"], r["contract_id"]))
        for a, b in zip(by_sig, by_sig[1:]):
            if (b["sig_close_ep"] - a["sig_close_ep"]) < a["horizon_s"]:
                anomaly_codes.append("signal_gap_below_horizon")
                anomalies.append(f"signal_gap_below_horizon:{a['contract_id']}:{b['contract_id']}")

        blockers = list(blockers_top)
        blockers.extend(anomalies)
        n_coh = len(cohort_rows)
        stat_codes: list[str] = []
        if n_coh < effective_min:
            blockers.append(f"insufficient_prospective_trades:{n_coh}<{effective_min}")
            stat_codes.append("insufficient_prospective_trades")

        ci_lo_str = None
        ci_hi_str = None
        ci_lo_f = None
        ci_hi_f = None
        roi_f = None
        wins = sum(1 for r in cohort_rows if r["profit_cents"] > 0)
        stake_sum = sum(r["buy_price_cents"] for r in cohort_rows)
        profit_sum = sum(r["profit_cents"] for r in cohort_rows)
        if n_coh > 0:
            roi_f = float(Fraction(profit_sum, stake_sum))
            ci_lo_frac, ci_hi_frac = exact_fraction_bootstrap_ci95(
                [r["profit_cents"] for r in cohort_rows], stake_cents=expected_stake_cents, n_boot=5000, seed=7
            )
            ci_lo_str = f"{ci_lo_frac.numerator}/{ci_lo_frac.denominator}"
            ci_hi_str = f"{ci_hi_frac.numerator}/{ci_hi_frac.denominator}"
            ci_lo_f = float(ci_lo_frac)
            ci_hi_f = float(ci_hi_frac)
            if ci_lo_frac <= 0:
                blockers.append("ev_ci95_lo_not_positive")
                stat_codes.append("ev_ci95_lo_not_positive")

        ok = len(blockers) == 0
        struct_codes = [_sanitize_mvt_reason_code(r) for r in blockers_top]
        b_summary = {
            "n": n_coh,
            "wins": wins,
            "win_rate": round(wins / n_coh, 4) if n_coh > 0 else None,
            "excluded_pre_anchor": excluded_pre_anchor,
            "excluded_post_eval": excluded_post_eval,
            "stake_usd": round(stake_sum / 100.0, 2),
            "net_profit_usd": round(profit_sum / 100.0, 2),
            "realized_ev_per_stake": round(roi_f, 6) if roi_f is not None else None,
            "ev_ci95": [round(ci_lo_f, 6) if ci_lo_f is not None else None, round(ci_hi_f, 6) if ci_hi_f is not None else None],
            "ev_ci95_exact_fraction": [ci_lo_str, ci_hi_str],
            "promotion_eligible": ok,
            "reasons": blockers,
            "_structural_and_anomaly_codes": struct_codes + list(anomaly_codes),
            "_stat_codes": stat_codes,
        }
        return {
            "schema": schema,
            "db_path": str(db_p),
            "book_id": book_id,
            "cohort_start_utc": cohort_start_utc,
            "eval_now_utc": eval_dt.isoformat() if eval_dt is not None else eval_now_utc,
            "promotion_scope": "single_book_prospective_g6",
            "settled_contracts": n_coh,
            "open_contracts": sum(1 for c in b_contracts if c["status"] == "open"),
            "unresolved_signals": len(b_uncertain),
            "by_book": {book_id: b_summary},
            "promotion_eligible_books": [book_id] if ok else [],
            "promotion_eligible": ok,
            "reasons": blockers,
        }

    # ------------------------------------------------------------------
    # Backwards-compatible v2 summary path
    # ------------------------------------------------------------------
    legacy_missing_tables = [t for t in ("sota_contracts", "sota_signals") if t in missing_tables]
    legacy_schema_errors = bool(schema_errors)
    settled_all = [x for x in all_contracts if x["status"] in SETTLED_STATUSES]
    open_all = [x for x in all_contracts if x["status"] == "open"]
    observed_books = sorted(
        {str(x["book_id"]) for x in all_contracts}
        | {str(x["book_id"]) for x in unresolved_sigs}
        | ({book_id} if book_id else set())
    )

    def _agg_book(bid: str) -> dict[str, Any]:
        b_settled = [x for x in settled_all if str(x["book_id"]) == bid]
        b_open = [x for x in open_all if str(x["book_id"]) == bid]
        b_unknown_c = [x for x in all_contracts if str(x["book_id"]) == bid and x["status"] == "unknown"]
        b_uncertain_s = [x for x in unresolved_sigs if str(x["book_id"]) == bid]

        blockers: list[str] = []
        if legacy_missing_tables:
            blockers.append("missing_required_queue_tables")
        if legacy_schema_errors:
            blockers.append("queue_schema_error")
        if b_open:
            blockers.append(f"open_contracts:{len(b_open)}")
        if b_unknown_c:
            blockers.append(f"unknown_contracts:{len(b_unknown_c)}")
        if b_uncertain_s:
            blockers.append(f"unresolved_buy_signals:{len(b_uncertain_s)}")
        effective_min = max(PROMOTION_HARD_MIN_TRADES, int(min_settled_for_promotion))
        if int(min_settled_for_promotion) < PROMOTION_HARD_MIN_TRADES:
            blockers.append(f"min_trades_arg_below_hard_floor:{min_settled_for_promotion}<{PROMOTION_HARD_MIN_TRADES}")
        if len(b_settled) < effective_min:
            blockers.append(f"insufficient_settled_trades:{len(b_settled)}<{effective_min}")
        non_demo = [x for x in b_settled if str(x.get("account_mode") or "").strip().lower() != "demo"]
        if non_demo:
            blockers.append(f"non_demo_provenance_rows:{len(non_demo)}")
        missing_cid = [x for x in b_settled if not str(x.get("book_content_id") or "").strip()]
        if missing_cid:
            blockers.append(f"missing_book_content_id_rows:{len(missing_cid)}")
        cids_seen = sorted({str(x["book_content_id"]).strip() for x in b_settled if str(x.get("book_content_id") or "").strip()})
        if len(cids_seen) > 1:
            blockers.append(f"mixed_book_content_ids:{cids_seen}")
        reg_cid = registered_cids.get(bid)
        if require_registered_book and not reg_cid:
            blockers.append("unregistered_book_id")
        want_cid = reg_cid if reg_cid else (expected_content_id if (book_id == bid and expected_content_id) else None)
        if reg_cid and expected_content_id and book_id == bid and expected_content_id != reg_cid:
            blockers.append(f"expected_content_id_conflicts_with_index:{expected_content_id}!={reg_cid}")
        if want_cid and cids_seen and cids_seen != [want_cid]:
            blockers.append(f"book_content_id_mismatch:{cids_seen}!={want_cid}")

        if not b_settled:
            return {
                "n": 0,
                "open_contracts": len(b_open),
                "unresolved_signals": len(b_uncertain_s),
                "promotion_eligible": False,
                "reasons": blockers,
            }

        bad_bp = [x for x in b_settled if x["buy_price_cents"] is None or int(x["buy_price_cents"]) <= 0]
        if bad_bp:
            blockers.append(f"invalid_nonpositive_buy_price:{len(bad_bp)}")
        valid_settled = [x for x in b_settled if x["buy_price_cents"] is not None and int(x["buy_price_cents"]) > 0]
        if not valid_settled:
            return {
                "n": len(b_settled),
                "open_contracts": len(b_open),
                "unresolved_signals": len(b_uncertain_s),
                "promotion_eligible": False,
                "reasons": blockers,
            }

        stake_cents = sum(int(x["buy_price_cents"]) for x in valid_settled)
        profit_cents = sum(int(x["profit_cents"] or 0) for x in valid_settled)
        wins = sum(1 for x in valid_settled if int(x["profit_cents"] or 0) > 0)
        unit_returns = np.array(
            [float(Fraction(int(x["profit_cents"] or 0), int(x["buy_price_cents"]))) for x in valid_settled],
            dtype=np.float64,
        )
        ci_lo, ci_hi = _bootstrap_ci95_float(unit_returns)
        if not (np.isfinite(ci_lo) and ci_lo > 0.0):
            blockers.append("ev_ci95_lo_not_positive")

        roi_exact = Fraction(profit_cents, stake_cents) if stake_cents > 0 else Fraction(0, 1)
        fixed_pay_rows = [x for x in valid_settled if x["payout_cents"] is not None and int(x["payout_cents"]) > 0]
        mean_pay = (
            round(float(np.mean([int(x["payout_cents"]) / int(x["buy_price_cents"]) for x in fixed_pay_rows])), 4)
            if fixed_pay_rows
            else None
        )
        ok = len(blockers) == 0
        return {
            "n": len(b_settled),
            "wins": wins,
            "win_rate": round(wins / len(valid_settled), 4),
            "open_contracts": len(b_open),
            "unknown_contracts": len(b_unknown_c),
            "unresolved_signals": len(b_uncertain_s),
            "stake_usd": round(stake_cents / 100.0, 2),
            "net_profit_usd": round(profit_cents / 100.0, 2),
            "mean_payout_ratio": mean_pay,
            "realized_ev_per_stake": round(float(roi_exact), 4),
            "ev_ci95": [round(ci_lo, 4) if np.isfinite(ci_lo) else None, round(ci_hi, 4) if np.isfinite(ci_hi) else None],
            "promotion_eligible": bool(ok),
            "reasons": blockers,
        }

    by_book = {bid: _agg_book(bid) for bid in observed_books}
    eligible_books = [bid for bid, stats in by_book.items() if stats.get("promotion_eligible") is True]
    total_stake_cents = sum(int(x["buy_price_cents"] or 0) for x in settled_all)
    total_profit_cents = sum(int(x["profit_cents"] or 0) for x in settled_all)

    target_eligible: bool | None = None
    reasons: list[str] = []
    if book_id is not None:
        b_res = by_book.get(book_id, {"promotion_eligible": False, "reasons": ["book_not_observed"]})
        target_eligible = bool(b_res.get("promotion_eligible", False))
        reasons = list(b_res.get("reasons", []))
    else:
        if legacy_missing_tables:
            reasons.append("missing_required_queue_tables")
        if legacy_schema_errors:
            reasons.append("queue_schema_error")

    return {
        "schema": schema,
        "db_path": str(db_p),
        "book_id": book_id,
        "promotion_scope": "single_book" if book_id else "per_book_only_never_pooled",
        "settled_contracts": len(settled_all),
        "open_contracts": len(open_all),
        "unresolved_signals": len(unresolved_sigs),
        "portfolio_accounting_totals": {
            "stake_usd": round(total_stake_cents / 100.0, 2),
            "net_profit_usd": round(total_profit_cents / 100.0, 2),
            "note": "Accounting sum only; win rates and bootstrap CIs are never pooled across books/timeframes.",
        },
        "by_book": by_book,
        "promotion_eligible_books": eligible_books,
        "promotion_eligible": target_eligible,
        "reasons": reasons,
    }


def evaluate_wald_sprt(
    wins: int,
    n: int,
    *,
    p0: float,
    p1: float,
    alpha: float = 0.05,
    beta: float = 0.10,
    outcomes: Sequence[bool] | None = None,
) -> dict[str, Any]:
    """Exact Bernoulli Wald Sequential Probability Ratio Test for H0: p <= p0 vs H1: p >= p1.

    When chronological `outcomes` are supplied, evaluates true sequential path-wise stopping
    (once LLR hits <= ln_b, the test permanently stops at REJECT_TO_H0; later wins cannot undo it).
    """
    if not (0.0 < p0 < p1 < 1.0) or (p1 - p0) < 1e-6:
        raise ValueError(f"invalid SPRT hypotheses p0={p0}, p1={p1}")
    if not (0.0 < alpha < 0.5 and 0.0 < beta < 0.5):
        raise ValueError(f"invalid SPRT error bounds alpha={alpha}, beta={beta}")
    if outcomes is not None:
        seq = [bool(x) for x in outcomes]
        tot = len(seq)
        k = sum(1 for x in seq if x)
    else:
        seq = None
        k = int(wins)
        tot = int(n)
    if tot < 0 or k < 0 or k > tot:
        raise ValueError(f"invalid wins={k} for n={tot}")

    ln_a = math.log((1.0 - beta) / alpha)
    ln_b = math.log(beta / (1.0 - alpha))
    step_win = math.log(p1 / p0)
    step_loss = math.log((1.0 - p1) / (1.0 - p0))
    kl_h1 = p1 * math.log(p1 / p0) + (1.0 - p1) * math.log((1.0 - p1) / (1.0 - p0))
    kl_h0 = p0 * math.log(p0 / p1) + (1.0 - p0) * math.log((1.0 - p0) / (1.0 - p1))
    en_h1 = ((1.0 - beta) * ln_a + beta * ln_b) / kl_h1
    en_h0 = -(alpha * ln_a + (1.0 - alpha) * ln_b) / kl_h0

    llr = k * step_win + (tot - k) * step_loss
    first_decision = "CONTINUE"
    stopped_at_n: int | None = None
    post_accept_degradation = False
    degraded_at_n: int | None = None

    if seq is not None and tot > 0:
        running = 0.0
        for idx_s, won_s in enumerate(seq, start=1):
            running += step_win if won_s else step_loss
            if stopped_at_n is None:
                if running <= ln_b:
                    first_decision = "REJECT_TO_H0"
                    stopped_at_n = idx_s
                elif running >= ln_a:
                    first_decision = "ACCEPT_H1"
                    stopped_at_n = idx_s
            elif first_decision == "ACCEPT_H1" and not post_accept_degradation and running <= ln_b:
                post_accept_degradation = True
                degraded_at_n = idx_s
        status = "REJECT_TO_H0" if post_accept_degradation else first_decision
        llr = running
    else:
        if tot > 0 and llr <= ln_b:
            first_decision = "REJECT_TO_H0"
            stopped_at_n = tot
        elif tot > 0 and llr >= ln_a:
            first_decision = "ACCEPT_H1"
            stopped_at_n = tot
        status = first_decision

    if status == "ACCEPT_H1":
        min_wins_more: int | None = 0
    elif status == "REJECT_TO_H0":
        min_wins_more = None
    else:
        min_wins_more = max(1, int(math.ceil((ln_a - llr) / step_win)))

    return {
        "p0": round(float(p0), 6),
        "p1": round(float(p1), 6),
        "alpha": float(alpha),
        "beta": float(beta),
        "n": tot,
        "wins": k,
        "llr": round(float(llr), 6),
        "upper_bound_ln_a": round(float(ln_a), 6),
        "lower_bound_ln_b": round(float(ln_b), 6),
        "status": status,
        "first_decision": first_decision,
        "stopped_at_n": stopped_at_n,
        "post_accept_degradation": post_accept_degradation,
        "degraded_at_n": degraded_at_n,
        "min_consecutive_wins_to_accept": min_wins_more,
        "expected_n_under_h1": round(float(en_h1), 1),
        "expected_n_under_h0": round(float(en_h0), 1),
    }


def evaluate_payout_horizon_parity(
    ratios: list[float],
    *,
    min_payout_ratio: float,
    min_fills: int = 15,
) -> dict[str, Any]:
    """Exact one-sided 99% Student-t lower bound on live realized payout ratio R = payout / stake."""
    vals = [float(x) for x in ratios if math.isfinite(float(x)) and float(x) > 0.0]
    n = len(vals)
    if n == 0:
        return {
            "n": 0,
            "min_fills_required": int(min_fills),
            "min_payout_ratio_floor": float(min_payout_ratio),
            "mean_ratio": None,
            "std_ratio": None,
            "min_ratio": None,
            "max_ratio": None,
            "ratio_ci99_lo": None,
            "empirical_breakeven": None,
            "passed": False,
        }
    all_identical = all(x == vals[0] for x in vals)
    if all_identical:
        mean_r = vals[0]
        min_r = vals[0]
        max_r = vals[0]
        std_r = 0.0 if n >= 2 else None
        ci99_lo = vals[0] if n >= 2 else None
    else:
        arr = np.asarray(vals, dtype=np.float64)
        mean_r = float(np.mean(arr))
        min_r = float(np.min(arr))
        max_r = float(np.max(arr))
        if n >= 2:
            std_r = float(np.std(arr, ddof=1))
            se = std_r / math.sqrt(n)
            t_crit = float(scipy_t.ppf(0.99, df=n - 1))
            ci99_lo = mean_r - t_crit * se
        else:
            std_r = None
            ci99_lo = None
    passed = bool(n >= int(min_fills) and ci99_lo is not None and ci99_lo >= float(min_payout_ratio))
    return {
        "n": n,
        "min_fills_required": int(min_fills),
        "min_payout_ratio_floor": float(min_payout_ratio),
        "mean_ratio": round(mean_r, 6),
        "std_ratio": round(std_r, 6) if std_r is not None else None,
        "min_ratio": round(min_r, 6),
        "max_ratio": round(max_r, 6),
        "ratio_ci99_lo": round(float(ci99_lo), 6) if ci99_lo is not None else None,
        "empirical_breakeven": round(1.0 / mean_r, 6) if mean_r > 1.0 else None,
        "passed": passed,
    }


def _scan_sanitized_runtime_logs_with_identity(
    log_dir: Path | None,
    *,
    trusted_root: Path | None = None,
) -> tuple[dict[str, Any] | None, Path | None, tuple[int, int] | None]:
    if log_dir is None:
        return None, None, None
    try:
        root_fd, resolved_dir = _open_dir_nofollow(Path(log_dir), trusted_root=trusted_root)
    except _SymlinkRefusedError:
        return (
            {"log_dir_present": False, "symlink_refused": True, "symlink_files_skipped": 0, "kill_present": True, "event_counts": {}},
            None,
            None,
        )
    except _MissingPathError:
        return (
            {"log_dir_present": False, "symlink_refused": False, "symlink_files_skipped": 0, "kill_present": False, "event_counts": {}},
            None,
            None,
        )
    except OSError:
        return (
            {"log_dir_present": False, "symlink_refused": True, "symlink_files_skipped": 0, "kill_present": True, "event_counts": {}},
            None,
            None,
        )

    kill_present = False
    symlink_refused = False
    symlink_skipped = 0
    counts: dict[str, int] = {}
    root_dev_ino: tuple[int, int] | None = None

    def _check_kill_in_fd(dfd: int) -> None:
        nonlocal kill_present, symlink_refused
        try:
            kst = os.lstat("KILL", dir_fd=dfd)
            kill_present = True
            if stat.S_ISLNK(kst.st_mode):
                symlink_refused = True
        except FileNotFoundError:
            pass
        except OSError:
            kill_present = True

    def _walk_fd(dfd: int, depth: int = 0) -> None:
        nonlocal symlink_skipped
        if depth > 8:
            return
        try:
            entries = sorted(os.listdir(dfd))
        except OSError:
            return
        for name in entries:
            try:
                est = os.lstat(name, dir_fd=dfd)
            except OSError:
                continue
            if stat.S_ISLNK(est.st_mode):
                symlink_skipped += 1
                continue
            if stat.S_ISDIR(est.st_mode):
                try:
                    sub_fd = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=dfd)
                except OSError:
                    symlink_skipped += 1
                    continue
                try:
                    _walk_fd(sub_fd, depth + 1)
                finally:
                    os.close(sub_fd)
            elif stat.S_ISREG(est.st_mode) and name.endswith(".jsonl"):
                try:
                    f_fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=dfd)
                except OSError:
                    symlink_skipped += 1
                    continue
                try:
                    fst = os.fstat(f_fd)
                    if not stat.S_ISREG(fst.st_mode):
                        symlink_skipped += 1
                        continue
                    with os.fdopen(f_fd, "r", encoding="utf-8", errors="replace") as fh:
                        f_fd = -1
                        for line in fh:
                            try:
                                obj = json.loads(line)
                            except Exception:
                                continue
                            if not isinstance(obj, dict):
                                continue
                            ev_raw = obj.get("event")
                            if type(ev_raw) is not str:
                                continue
                            ev = ev_raw.strip()
                            if not ev:
                                continue
                            key_ev = ev if ev in _ALLOWED_LOG_EVENTS else "other_events"
                            counts[key_ev] = counts.get(key_ev, 0) + 1
                finally:
                    if f_fd >= 0:
                        os.close(f_fd)

    try:
        rst = os.fstat(root_fd)
        root_dev_ino = (int(rst.st_dev), int(rst.st_ino))
        _check_kill_in_fd(root_fd)
        # Check nested sota_executor_demo/KILL strictly via O_NOFOLLOW (never follow symlinked subdir!)
        try:
            demo_st = os.lstat("sota_executor_demo", dir_fd=root_fd)
            if stat.S_ISLNK(demo_st.st_mode):
                symlink_refused = True
                kill_present = True
            elif stat.S_ISDIR(demo_st.st_mode):
                try:
                    sub_demo_fd = os.open("sota_executor_demo", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=root_fd)
                except OSError:
                    symlink_refused = True
                    kill_present = True
                else:
                    try:
                        _check_kill_in_fd(sub_demo_fd)
                    finally:
                        os.close(sub_demo_fd)
        except FileNotFoundError:
            pass
        except OSError:
            symlink_refused = True
            kill_present = True

        _walk_fd(root_fd, 0)
    finally:
        os.close(root_fd)

    return (
        {
            "log_dir_present": True,
            "symlink_refused": symlink_refused,
            "symlink_files_skipped": symlink_skipped,
            "kill_present": kill_present,
            "event_counts": {k: counts[k] for k in sorted(counts)},
        },
        resolved_dir,
        root_dev_ino,
    )


def _scan_sanitized_runtime_logs(log_dir: Path | None, *, trusted_root: Path | None = None) -> dict[str, Any] | None:
    summary, _, _ = _scan_sanitized_runtime_logs_with_identity(log_dir, trusted_root=trusted_root)
    return summary


def evaluate_portfolio_mvt(
    db_path: Path,
    *,
    book_ids: tuple[str, ...] = DEFAULT_PHASE1_PORTFOLIO_BOOKS,
    repo_root: Path | None = REPO_ROOT,
    canonical_db_path: Path | None = None,
    canonical_log_dir: Path | None = None,
    extra_kill_check_dirs: tuple[Path, ...] = (),
    enforce_canonical_db: bool = True,
    eval_now_utc: str | None = None,
    log_dir: Path | None = None,
    min_parity_fills: int = MIN_PARITY_FILLS_FLOOR,
    min_horizon_fills: int = MIN_HORIZON_FILLS_FLOOR,
    alpha_broken: float = POLICY_SPRT_ALPHA_BROKEN,
    beta_broken: float = POLICY_SPRT_BETA_BROKEN,
    alpha_ev: float = POLICY_SPRT_ALPHA_EV,
    beta_ev: float = POLICY_SPRT_BETA_EV,
    p0_ev_fixed: float = POLICY_SPRT_P0_EV,
    h1_wr: float = POLICY_SPRT_H1_WR,
) -> dict[str, Any]:
    """Read-only Portfolio Minimal-Viable-Trades Validator (Pillars 1-3 + per-book G6).

    Reads ONE atomic zero-sidecar read-only SQLite snapshot, enforces strict O_NOFOLLOW beneath
    repo_root, applies first-come temporal de-overlap on BOTH cross-book Bernoulli inference AND
    per-horizon payout CI99, emits strictly sanitized fixed reason codes, and NEVER authorizes live trading.
    """
    db_p = Path(db_path)
    db_path_fp = f"sha256:{hashlib.sha256(str(db_p).encode('utf-8')).hexdigest()[:12]}"
    eval_dt = _parse_iso_utc(eval_now_utc) if eval_now_utc is not None else datetime.now(timezone.utc)
    eval_iso = eval_dt.isoformat() if eval_dt is not None else None
    eval_invalid = bool(eval_now_utc is not None and eval_dt is None)

    registered = _load_registered_books(repo_root)
    safe_book_ids: list[str] = []
    invalid_book_id_seen = False
    unregistered_book_id_seen = False
    for b_raw in book_ids:
        if not isinstance(b_raw, str) or not _VALID_BOOK_ID_RE.fullmatch(b_raw):
            invalid_book_id_seen = True
            safe_book_ids.append("<invalid_book_id>")
        elif b_raw not in registered:
            # Never echo arbitrary caller strings even if regex-shaped!
            unregistered_book_id_seen = True
            safe_book_ids.append("<unregistered_book_id>")
        else:
            safe_book_ids.append(b_raw)

    canon_log_dir = Path(canonical_log_dir) if canonical_log_dir is not None else (
        (repo_root or REPO_ROOT) / "logs" / "sota_runtime"
    )
    log_summary, log_resolved_dir, log_dev_ino = _scan_sanitized_runtime_logs_with_identity(
        log_dir, trusted_root=repo_root
    )
    kill_present = bool(log_summary.get("kill_present")) if isinstance(log_summary, dict) else False
    log_symlink_refused = bool(log_summary.get("symlink_refused")) if isinstance(log_summary, dict) else False
    log_dir_canonical = False
    if log_resolved_dir is not None and log_dev_ino is not None:
        try:
            cl_fd, cl_res = _open_dir_nofollow(canon_log_dir, trusted_root=repo_root)
            try:
                cl_st = os.fstat(cl_fd)
                log_dir_canonical = (
                    log_resolved_dir == cl_res
                    and log_dev_ino == (int(cl_st.st_dev), int(cl_st.st_ino))
                )
            finally:
                os.close(cl_fd)
        except (OSError, _SymlinkRefusedError, _MissingPathError):
            log_dir_canonical = False
    # Always inspect the configured canonical runtime log dir AND any extra configured runtime dirs for KILL/symlink tampering!
    extra_dirs_to_scan: list[Path] = []
    if not log_dir_canonical:
        extra_dirs_to_scan.append(canon_log_dir)
    for ex_d in extra_kill_check_dirs:
        extra_dirs_to_scan.append(Path(ex_d))
    for scan_d in extra_dirs_to_scan:
        aux_scan, _, _ = _scan_sanitized_runtime_logs_with_identity(scan_d, trusted_root=repo_root)
        if isinstance(aux_scan, dict):
            if aux_scan.get("kill_present"):
                kill_present = True
            if aux_scan.get("symlink_refused"):
                log_symlink_refused = True
    log_dir_verified = bool(
        isinstance(log_summary, dict)
        and log_summary.get("log_dir_present") is True
        and log_dir_canonical
        and not log_symlink_refused
        and not kill_present
    )

    def _early_return(verdict_val: str, reasons_list: list[str], *, canon_flag: bool = False) -> dict[str, Any]:
        return {
            "schema": "deriv-sota-portfolio-mvt-validator/v1",
            "live_trading_authorized": False,
            "advisory_note": "Read-only statistical readiness report; Live CR* execution always requires explicit owner authorization.",
            "db_canonical": canon_flag,
            "log_dir_canonical": log_dir_canonical,
            "db_path_sha256": db_path_fp,
            "account_id_sha256": None,
            "eval_now_utc": eval_iso,
            "book_ids": safe_book_ids,
            "verdict": verdict_val,
            "reasons": reasons_list,
            "runtime_logs": log_summary,
        }

    # Acquire ONE zero-sidecar read-only snapshot via strict O_NOFOLLOW before inspecting anything else!
    try:
        snap = _read_db_snapshot(db_p, trusted_root=repo_root)
    except _SymlinkRefusedError:
        r_sym = ["symlinked_queue_db_refused"]
        if kill_present:
            r_sym.insert(0, "kill_marker_present")
        if log_symlink_refused:
            r_sym.insert(0, "log_dir_symlink_refused")
        if eval_invalid:
            r_sym.append("invalid_eval_now_utc")
        return _early_return("HALT_INVESTIGATE", r_sym, canon_flag=False)
    except _MissingPathError:
        r_miss: list[str] = []
        if log_symlink_refused:
            r_miss.append("log_dir_symlink_refused")
        if kill_present:
            r_miss.append("kill_marker_present")
        if eval_invalid:
            r_miss.append("invalid_eval_now_utc")
        if invalid_book_id_seen:
            r_miss.append("invalid_book_id")
        if unregistered_book_id_seen:
            r_miss.append("unregistered_book_id")
        r_miss.append("queue_db_missing")
        # KILL / log symlink / invalid eval always take strict precedence over missing DB!
        v_miss = "HALT_INVESTIGATE" if (kill_present or log_symlink_refused or eval_invalid or invalid_book_id_seen or unregistered_book_id_seen) else "HOLD_COLLECTING"
        return _early_return(v_miss, r_miss, canon_flag=False)
    except (sqlite3.Error, OSError, ValueError):
        r_err = ["queue_db_schema_or_read_error"]
        if kill_present:
            r_err.insert(0, "kill_marker_present")
        if log_symlink_refused:
            r_err.insert(0, "log_dir_symlink_refused")
        if eval_invalid:
            r_err.append("invalid_eval_now_utc")
        return _early_return("HALT_INVESTIGATE", r_err, canon_flag=False)

    canon = canonical_db_path if canonical_db_path is not None else (
        (repo_root or REPO_ROOT) / "deriv_data" / "runtime" / "sota_trade_queue_demo.sqlite"
    )
    canon_ok = _verify_snapshot_is_canonical(snap, canon, trusted_root=repo_root)

    user_ver = int(snap["user_version"])
    risk_state = snap["risk_state"]
    risk_state_keys = snap["risk_state_keys"]
    all_contracts = snap["all_contracts"]
    unresolved_total = len(snap["unresolved_sigs"])
    missing_tables = snap["missing_tables"]
    schema_errors = list(snap.get("schema_errors") or [])

    binding = risk_state.get("queue_account_binding") if isinstance(risk_state.get("queue_account_binding"), dict) else {}
    acct_id = str(binding.get("account_id") or "").strip()
    acct_mode = str(binding.get("account_mode") or "").strip().lower()
    acct_demo_ok = bool(acct_id and acct_mode == "demo" and is_valid_demo_account_id(acct_id))
    # Fail closed on ANY phase1_safety_latch row, including malformed JSON (matching executor is_safety_latch_set)!
    latch_set = "phase1_safety_latch" in risk_state_keys

    hard_blockers: list[str] = []
    if eval_invalid:
        hard_blockers.append("invalid_eval_now_utc")
    if invalid_book_id_seen:
        hard_blockers.append("invalid_book_id")
    if unregistered_book_id_seen:
        hard_blockers.append("unregistered_book_id")
    if enforce_canonical_db and not canon_ok:
        hard_blockers.append("non_canonical_db_diagnostic_only")
    if missing_tables:
        hard_blockers.append("missing_required_queue_tables")
    if schema_errors:
        hard_blockers.append("queue_schema_error")
    if user_ver != 3:
        hard_blockers.append("queue_not_v3")
    if not acct_demo_ok:
        hard_blockers.append("queue_account_binding_invalid_or_non_demo")
    if latch_set:
        hard_blockers.append("phase1_safety_latch_active")
    if log_symlink_refused:
        hard_blockers.append("log_dir_symlink_refused")
    if kill_present:
        hard_blockers.append("kill_marker_present")
    if unresolved_total > 0:
        hard_blockers.append("unresolved_buy_signals")

    diagnostic_only_reasons: list[str] = []
    if not enforce_canonical_db or not canon_ok:
        diagnostic_only_reasons.append("diagnostic_non_canonical_or_unchecked_db")
    if log_dir is not None and isinstance(log_summary, dict) and log_summary.get("log_dir_present") is True and not log_dir_canonical:
        diagnostic_only_reasons.append("diagnostic_non_canonical_log_dir")
    if tuple(safe_book_ids) != DEFAULT_PHASE1_PORTFOLIO_BOOKS:
        diagnostic_only_reasons.append("diagnostic_non_canonical_book_set")
    if (
        int(min_parity_fills) != MIN_PARITY_FILLS_FLOOR
        or int(min_horizon_fills) != MIN_HORIZON_FILLS_FLOOR
        or not math.isclose(float(alpha_broken), POLICY_SPRT_ALPHA_BROKEN, rel_tol=0.0, abs_tol=1e-12)
        or not math.isclose(float(beta_broken), POLICY_SPRT_BETA_BROKEN, rel_tol=0.0, abs_tol=1e-12)
        or not math.isclose(float(alpha_ev), POLICY_SPRT_ALPHA_EV, rel_tol=0.0, abs_tol=1e-12)
        or not math.isclose(float(beta_ev), POLICY_SPRT_BETA_EV, rel_tol=0.0, abs_tol=1e-12)
        or not math.isclose(float(p0_ev_fixed), POLICY_SPRT_P0_EV, rel_tol=0.0, abs_tol=1e-12)
        or not math.isclose(float(h1_wr), POLICY_SPRT_H1_WR, rel_tol=0.0, abs_tol=1e-12)
    ):
        diagnostic_only_reasons.append("diagnostic_policy_override")

    per_book: dict[str, Any] = {}
    horizon_rows: dict[int, list[dict[str, Any]]] = {}
    floors_by_horizon: dict[int, float] = {}
    pooled_rows: list[dict[str, Any]] = []
    seen_cids: set[str] = set()
    anomaly_count = 0
    missing_anchors: list[str] = []
    cohort_not_started_books: list[str] = []
    eval_ep = int(eval_dt.timestamp()) if eval_dt is not None else None

    for bid in safe_book_ids:
        if bid in ("<invalid_book_id>", "<unregistered_book_id>"):
            continue
        reg_ent = registered[bid]
        cid_want = str(reg_ent["content_id"])
        hm = int(reg_ent["horizon_minutes"])
        strat = reg_ent.get("_strategy") or {}
        floor_r = float(strat.get("min_payout_ratio") or DEFAULT_MIN_PAYOUT_BY_HORIZON.get(hm, 1.78))
        horizon_rows.setdefault(hm, [])
        floors_by_horizon[hm] = max(floors_by_horizon.get(hm, 0.0), floor_r)

        anc_key = f"g6_cohort_anchor:{acct_id}:{bid}:{cid_want}" if acct_demo_ok else ""
        anc = risk_state.get(anc_key) if anc_key else None
        if not isinstance(anc, dict) or not anc.get("cohort_start_utc"):
            missing_anchors.append(bid)
            per_book[bid] = {"anchor_registered": False, "g6_promotion_eligible": False, "n": 0, "reasons": ["missing_stored_cohort_anchor"]}
            continue

        coh_ep = _parse_whole_second_epoch(anc.get("cohort_start_utc"))
        reg_ep = _parse_whole_second_epoch(anc.get("registered_utc"))
        # NEVER echo raw unvalidated cohort_start_utc from DB!
        coh_iso = (
            datetime.fromtimestamp(coh_ep, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            if coh_ep is not None
            else None
        )
        if reg_ep is None or coh_ep is None or not (reg_ep < coh_ep):
            anomaly_count += 1
            per_book[bid] = {
                "anchor_registered": True,
                "cohort_start_utc": coh_iso,
                "g6_promotion_eligible": False,
                "n": 0,
                "integrity_anomalies": ["invalid_prospective_anchor_order"],
                "g6_reasons": ["anchor_clock_ordering_invalid"],
            }
            continue
        if eval_ep is not None and eval_ep < coh_ep:
            cohort_not_started_books.append(bid)
            per_book[bid] = {
                "anchor_registered": True,
                "cohort_start_utc": coh_iso,
                "horizon_minutes": hm,
                "n": 0,
                "wins": 0,
                "g6_promotion_eligible": False,
                "integrity_anomalies": [],
                "g6_reasons": ["cohort_not_yet_started"],
            }
            continue

        assert coh_iso is not None
        g6_rep = _summarize_settled_queue_impl(
            db_p,
            min_settled_for_promotion=PROMOTION_HARD_MIN_TRADES,
            book_id=bid,
            expected_content_id=cid_want,
            repo_root=repo_root,
            require_registered_book=True,
            cohort_start_utc=coh_iso,
            expected_account_id=acct_id,
            expected_stake_usd="1.00",
            eval_now_utc=eval_iso,
            canonical_db_path=canonical_db_path,
            enforce_canonical_db=enforce_canonical_db,
            _snapshot=snap,
        )
        b_stat = (g6_rep.get("by_book") or {}).get(bid) or {}
        # Read structured anomaly & structural codes directly — immune to contract_id prefix spoofing!
        safe_anomalies = list(b_stat.get("_structural_and_anomaly_codes") or [])
        safe_g6_reasons = safe_anomalies + list(b_stat.get("_stat_codes") or [])
        if safe_anomalies:
            anomaly_count += len(safe_anomalies)

        per_book[bid] = {
            "anchor_registered": True,
            "cohort_start_utc": coh_iso,
            "horizon_minutes": hm,
            "n": int(b_stat.get("n") or 0),
            "wins": int(b_stat.get("wins") or 0),
            "win_rate": b_stat.get("win_rate"),
            "realized_ev_per_stake": b_stat.get("realized_ev_per_stake"),
            "ev_ci95": b_stat.get("ev_ci95"),
            "g6_promotion_eligible": bool(b_stat.get("promotion_eligible")),
            "integrity_anomalies": safe_anomalies,
            "g6_reasons": safe_g6_reasons,
        }

        if acct_demo_ok and not safe_anomalies and not missing_tables and not schema_errors and not eval_invalid:
            for c in all_contracts:
                if str(c.get("book_id")) != bid:
                    continue
                if str(c.get("status") or "") not in G6_ELIGIBLE_SETTLED_STATUSES:
                    continue
                raw_mig = c.get("is_migrated_legacy")
                if raw_mig is not None and _strict_int_or_none(raw_mig) != 0:
                    continue
                v_st = _strict_int_or_none(c.get("venue_start_epoch"))
                v_en = _strict_int_or_none(c.get("venue_end_epoch"))
                if v_st is None or v_en is None or v_st < coh_ep or v_en < v_st:
                    continue
                closed_dt = _parse_iso_utc(c.get("closed_utc"))
                if closed_dt is None:
                    continue
                closed_ep = int(closed_dt.timestamp())
                if closed_ep < (v_en - MAX_CLOCK_SKEW_S):
                    continue
                if eval_dt is not None and (closed_dt > eval_dt or (eval_ep is not None and v_en > eval_ep)):
                    continue
                cid = str(c["contract_id"])
                if cid in seen_cids:
                    continue
                bp = _strict_int_or_none(c.get("buy_price_cents"))
                po = _strict_int_or_none(c.get("payout_cents"))
                pf = _strict_int_or_none(c.get("profit_cents"))
                if bp is None or po is None or pf is None or bp <= 0 or po <= 0:
                    continue
                st_row = str(c.get("status") or "").lower()
                seen_cids.add(cid)
                ratio = float(Fraction(po, bp))
                row_rec = {
                    "contract_id": cid,
                    "book_id": bid,
                    "horizon_minutes": hm,
                    "status": st_row,
                    "v_start": v_st,
                    "v_end": v_en,
                    "buy_price_cents": bp,
                    "payout_cents": po,
                    "profit_cents": pf,
                    "ratio": ratio,
                }
                horizon_rows[hm].append(row_rec)
                pooled_rows.append(row_rec)

    if missing_anchors:
        hard_blockers.append("missing_cohort_anchors")
    if anomaly_count > 0:
        hard_blockers.append("contract_integrity_anomalies")

    # Keep early-'sold' contracts in accounting totals, but exclude them from fixed-payout binary Bernoulli inference!
    binary_rows = [r for r in pooled_rows if r["status"] in {"won", "lost"}]
    n_early_sold_excluded = len(pooled_rows) - len(binary_rows)
    # First-come cross-book temporal de-overlap for strictly independent Bernoulli inference!
    binary_rows.sort(key=lambda r: (r["v_start"], r["v_end"], r["contract_id"]))
    independent_rows: list[dict[str, Any]] = []
    last_end_ep: int | None = None
    for row in binary_rows:
        if last_end_ep is None or row["v_start"] > last_end_ep:
            independent_rows.append(row)
            last_end_ep = row["v_end"]
    n_total_acct = len(pooled_rows)
    n_indep = len(independent_rows)
    n_overlap_excl = len(binary_rows) - n_indep

    # Order independent outcomes chronologically by settlement (v_end, v_start, contract_id) for path-wise SPRT
    indep_by_settle = sorted(independent_rows, key=lambda r: (r["v_end"], r["v_start"], r["contract_id"]))
    indep_outcomes = [r["status"] == "won" for r in indep_by_settle]
    wins_indep = sum(1 for x in indep_outcomes if x)

    stake_sum_all = sum(r["buy_price_cents"] for r in pooled_rows)
    profit_sum_all = sum(r["profit_cents"] for r in pooled_rows)
    realized_ev_all = float(Fraction(profit_sum_all, stake_sum_all)) if stake_sum_all > 0 else None

    stake_sum_indep = sum(r["buy_price_cents"] for r in indep_by_settle)
    profit_sum_indep = sum(r["profit_cents"] for r in indep_by_settle)
    realized_ev_indep = float(Fraction(profit_sum_indep, stake_sum_indep)) if stake_sum_indep > 0 else None
    mean_pay_indep = float(np.mean([r["ratio"] for r in indep_by_settle])) if n_indep > 0 else None
    realized_be = (1.0 / mean_pay_indep) if (mean_pay_indep is not None and mean_pay_indep > 1.0) else None

    # Pillar 1: Execution & Parity (requires verified canonical runtime log dir + zero hard blockers + >= min_parity_fills)
    pillar1_passed = bool(
        len(hard_blockers) == 0
        and len(cohort_not_started_books) == 0
        and log_dir_verified
        and n_indep >= int(min_parity_fills)
    )
    pillar1 = {
        "passed": pillar1_passed,
        "min_parity_fills_required": int(min_parity_fills),
        "n_independent_settled": n_indep,
        "n_total_accounting_settled": n_total_acct,
        "n_cross_book_overlaps_excluded_from_inference": n_overlap_excl,
        "n_early_sold_excluded_from_bernoulli": n_early_sold_excluded,
        "unresolved_signals": unresolved_total,
        "integrity_anomalies": anomaly_count,
        "safety_latch_active": latch_set,
        "kill_present": kill_present,
        "log_dir_canonical": log_dir_canonical,
        "log_dir_verified": log_dir_verified,
        "hard_blockers": list(hard_blockers),
        "cohort_not_started_books": cohort_not_started_books,
        "runtime_logs": log_summary,
    }

    # Pillar 2: Live Venue Payout Parity per horizon using strictly non-overlapping horizon observations!
    by_horizon: dict[str, Any] = {}
    for hm in sorted(horizon_rows):
        h_sorted = sorted(horizon_rows[hm], key=lambda r: (r["v_start"], r["v_end"], r["contract_id"]))
        h_indep: list[float] = []
        last_h_end: int | None = None
        for r in h_sorted:
            if last_h_end is None or r["v_start"] > last_h_end:
                h_indep.append(float(r["ratio"]))
                last_h_end = r["v_end"]
        h_eval = evaluate_payout_horizon_parity(
            h_indep, min_payout_ratio=floors_by_horizon[hm], min_fills=min_horizon_fills
        )
        h_eval["n_total_accounting"] = len(h_sorted)
        h_eval["n_overlaps_excluded"] = len(h_sorted) - len(h_indep)
        by_horizon[f"{hm}m"] = h_eval
    pillar2_passed = bool(by_horizon) and all(v["passed"] is True for v in by_horizon.values())
    pillar2 = {"passed": pillar2_passed, "by_horizon": by_horizon}

    # Pillar 3: Sequential WR Validation on strictly non-overlapping independent stream with FIXED hypotheses
    sprt_broken = evaluate_wald_sprt(
        wins_indep, n_indep, p0=0.50, p1=float(h1_wr), alpha=alpha_broken, beta=beta_broken, outcomes=indep_outcomes
    )
    sprt_ev = evaluate_wald_sprt(
        wins_indep, n_indep, p0=float(p0_ev_fixed), p1=float(h1_wr), alpha=alpha_ev, beta=beta_ev, outcomes=indep_outcomes
    )
    cp_lo95 = (
        0.0 if wins_indep == 0 else float(scipy_beta.ppf(0.05, wins_indep, n_indep - wins_indep + 1))
    ) if n_indep > 0 else None
    if n_indep > 0:
        if realized_be is None:
            post_uninf = 0.0
            post_cpcv = 0.0
        else:
            post_uninf = float(1.0 - scipy_beta.cdf(realized_be, 1 + wins_indep, 1 + n_indep - wins_indep))
            post_cpcv = float(1.0 - scipy_beta.cdf(realized_be, 60 + wins_indep, 40 + n_indep - wins_indep))
    else:
        post_uninf = None
        post_cpcv = None
    all_books_g6 = (
        tuple(safe_book_ids) == DEFAULT_PHASE1_PORTFOLIO_BOOKS
        and len(per_book) == len(DEFAULT_PHASE1_PORTFOLIO_BOOKS)
        and all(b.get("g6_promotion_eligible") is True for b in per_book.values())
    )

    pillar3 = {
        "n_independent": n_indep,
        "wins_independent": wins_indep,
        "win_rate_independent": round(wins_indep / n_indep, 6) if n_indep > 0 else None,
        "n_total_accounting": n_total_acct,
        "n_early_sold_excluded_from_bernoulli": n_early_sold_excluded,
        "stake_usd_total": round(stake_sum_all / 100.0, 2),
        "net_profit_usd_total": round(profit_sum_all / 100.0, 2),
        "realized_ev_per_stake_total": round(realized_ev_all, 6) if realized_ev_all is not None else None,
        "realized_ev_per_stake_independent": round(realized_ev_indep, 6) if realized_ev_indep is not None else None,
        "realized_mean_payout_ratio": round(mean_pay_indep, 6) if mean_pay_indep is not None else None,
        "realized_breakeven": round(realized_be, 6) if realized_be is not None else None,
        "fixed_ev_null_p0": round(float(p0_ev_fixed), 6),
        "clopper_pearson_one_sided_95_lo": round(cp_lo95, 6) if cp_lo95 is not None else None,
        "posterior_p_gt_realized_be_uninformative_beta11": round(post_uninf, 6) if post_uninf is not None else None,
        "posterior_p_gt_realized_be_cpcv_prior_beta60_40": round(post_cpcv, 6) if post_cpcv is not None else None,
        "sprt_broken_execution": sprt_broken,
        "sprt_positive_ev": sprt_ev,
        "all_books_strict_g6_eligible": all_books_g6,
    }

    reasons: list[str] = list(hard_blockers) + list(diagnostic_only_reasons)
    if sprt_broken["status"] == "REJECT_TO_H0":
        reasons.append("sprt_broken_execution_rejected_to_h0")
    if sprt_ev["status"] == "REJECT_TO_H0":
        reasons.append("sprt_positive_ev_rejected_to_h0")

    ev_pos = (realized_ev_indep is not None and realized_ev_indep > 0.0 and realized_ev_all is not None and realized_ev_all > 0.0)
    if hard_blockers or sprt_broken["status"] == "REJECT_TO_H0" or sprt_ev["status"] == "REJECT_TO_H0":
        verdict = "HALT_INVESTIGATE"
    elif diagnostic_only_reasons:
        verdict = "DIAGNOSTIC_ONLY_NOT_PORTFOLIO_READY"
    elif pillar1_passed and pillar2_passed and sprt_ev["status"] == "ACCEPT_H1" and all_books_g6 and ev_pos:
        verdict = "READY_FOR_OWNER_FULL_LIVE_AND_G6_REVIEW"
    elif pillar1_passed and pillar2_passed and sprt_ev["status"] == "ACCEPT_H1" and ev_pos:
        verdict = "READY_FOR_OWNER_MVT_EV_REVIEW_G6_PENDING"
    elif pillar1_passed and pillar2_passed and sprt_broken["status"] == "ACCEPT_H1" and ev_pos:
        verdict = "READY_FOR_OWNER_MICRO_LIVE_REVIEW_G6_PENDING"
    else:
        verdict = "HOLD_COLLECTING"
        if cohort_not_started_books:
            reasons.append("cohort_not_yet_started")
        if not pillar1_passed:
            if not log_dir_verified:
                reasons.append("runtime_log_dir_missing_or_unverified")
            if n_indep < int(min_parity_fills):
                reasons.append("pillar1_insufficient_independent_fills")
        if not pillar2_passed:
            reasons.append("pillar2_horizon_payout_pending")
        if sprt_broken["status"] != "ACCEPT_H1":
            reasons.append("pillar3_sprt_broken_continue")

    return {
        "schema": "deriv-sota-portfolio-mvt-validator/v1",
        "live_trading_authorized": False,
        "advisory_note": "Read-only statistical readiness report; Live CR* execution always requires explicit owner authorization.",
        "db_canonical": canon_ok,
        "log_dir_canonical": log_dir_canonical,
        "db_path_sha256": db_path_fp,
        "account_id_sha256": _redact_acct(acct_id) if acct_id else None,
        "eval_now_utc": eval_iso,
        "book_ids": safe_book_ids,
        "verdict": verdict,
        "reasons": reasons,
        "pillar1_execution_and_parity": pillar1,
        "pillar2_live_payout_parity": pillar2,
        "pillar3_portfolio_sequential_wr": pillar3,
        "by_book": per_book,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=None, help="Diagnostic alias; non-canonical path always yields promotion_eligible=False in prospective mode")
    ap.add_argument("--portfolio-mvt-report", action="store_true", help="Emit read-only 3-pillar Portfolio Minimal-Viable-Trades SPRT readiness report")
    ap.add_argument("--book-ids", default=",".join(DEFAULT_PHASE1_PORTFOLIO_BOOKS), help="Comma-separated book IDs for --portfolio-mvt-report")
    ap.add_argument("--log-dir", default=os.environ.get("DERIV_SOTA_LOG_DIR", str(REPO_ROOT / "logs" / "sota_runtime")), help="Runtime log dir for KILL/event parity scan in --portfolio-mvt-report")
    ap.add_argument("--min-parity-fills", type=int, default=10)
    ap.add_argument("--min-horizon-fills", type=int, default=15)
    ap.add_argument("--book-id", default=None, help="Evaluate promotion gate for one specific SOTA book_id")
    ap.add_argument("--cohort-start-utc", default=None, help="Prospective G6 cohort start timestamp (RFC3339 whole-second UTC)")
    ap.add_argument("--expected-account-id", default=None)
    ap.add_argument("--expected-stake-usd", default=None)
    ap.add_argument("--eval-now-utc", default=None)
    ap.add_argument("--min-trades", type=int, default=50)
    args = ap.parse_args(argv)
    canon_db = REPO_ROOT / "deriv_data" / "runtime" / "sota_trade_queue_demo.sqlite"
    default_canon_log = REPO_ROOT / "logs" / "sota_runtime"
    env_log_raw = os.environ.get("DERIV_SOTA_LOG_DIR", "").strip()
    if env_log_raw:
        env_log_p = Path(env_log_raw)
        configured_canon_log = env_log_p if env_log_p.is_absolute() else (REPO_ROOT / env_log_p)
    else:
        configured_canon_log = default_canon_log
    db_path = Path(args.db) if args.db else canon_db
    if args.portfolio_mvt_report:
        bids = tuple(x.strip() for x in str(args.book_ids or "").split(",") if x.strip())
        log_dir_p = Path(args.log_dir) if args.log_dir else configured_canon_log
        if not log_dir_p.is_absolute():
            log_dir_p = REPO_ROOT / log_dir_p
        extra_kill_dirs = (default_canon_log,) if configured_canon_log != default_canon_log else ()
        report = evaluate_portfolio_mvt(
            db_path,
            book_ids=bids or DEFAULT_PHASE1_PORTFOLIO_BOOKS,
            repo_root=REPO_ROOT,
            canonical_db_path=canon_db,
            canonical_log_dir=configured_canon_log,
            extra_kill_check_dirs=extra_kill_dirs,
            enforce_canonical_db=True,
            eval_now_utc=args.eval_now_utc,
            log_dir=log_dir_p,
            min_parity_fills=args.min_parity_fills,
            min_horizon_fills=args.min_horizon_fills,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    # BLOCKER #5: CLI ALWAYS evaluates strictly in canonical prospective G6 mode (never promoting retrospective/non-canonical DBs!)
    report = summarize_settled_queue(
        db_path,
        min_settled_for_promotion=args.min_trades,
        book_id=args.book_id,
        repo_root=REPO_ROOT,
        require_registered_book=True,
        cohort_start_utc=args.cohort_start_utc,
        expected_account_id=args.expected_account_id,
        expected_stake_usd=args.expected_stake_usd,
        eval_now_utc=args.eval_now_utc,
        canonical_db_path=canon_db,
        enforce_canonical_db=True,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
