"""Durable SQLite queue for the Deriv hot trade executor (issue #6).

SQLite is the queue of record. Wake-ups are only a post-commit latency hint:
startup scans and polling must still find every claimable row.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import sqlite3
import statistics
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable


DEFAULT_RUNTIME_DIR = Path("deriv_data/runtime")
DEFAULT_DB_PATH = DEFAULT_RUNTIME_DIR / "trade_queue.sqlite"
DEFAULT_WAKEUP_PATH = DEFAULT_RUNTIME_DIR / "trade_queue.wakeup.sock"
GATE_RESULT_PATH = Path("deriv_trade_executor_gate_result.json")

SQLITE_BUSY_TIMEOUT_MS = 5000
QUEUE_POLL_INTERVAL_MS = 100
EXECUTOR_WORKER_CONCURRENCY = 12
SYNC_FALLBACK_TIMEOUT_S = 10.0
SYNC_FALLBACK_MAX_CONCURRENCY = 3
ASYNC_PROPOSAL_TIMEOUT_MS = 2000
BUY_REQUEST_TIMEOUT_MS = 5000
AUTH_HEALTH_INTERVAL_S = 60
HEARTBEAT_INTERVAL_MS = 100
EVENT_LOOP_HEARTBEAT_MAX_GAP_MS = 250
DB_TRANSACTION_P99_MAX_MS = 50
DB_TRANSACTION_MAX_MS = 250
COMMIT_TO_WAKEUP_RECEIVED_P95_MAX_MS = 25
PRODUCER_QUEUE_MODE_POLL_SECONDS = 0.25
PRODUCER_QUEUE_MODE_FANOUT_LIMIT = 6
WALL_SIGNAL_MAX_AGE_S = 60.0
SHIFTED_SIGNAL_MAX_AGE_S = 2.0
HOT_PROPOSAL_MAX_BUY_AGE_MS = 1000
SAME_PAIR_MIN_GAP_SECONDS = 900
MAX_USD_FACTOR_OPEN = 2
ALLOCATION_ARBITRATION_MS = 250
RECONCILE_BATCH_SIZE = 12
RECONCILE_INTERVAL_SECONDS = 5.0
CONTRACT_EXPIRY_GRACE_SECONDS = 600.0

SIGNAL_STATUSES = (
    "queued",
    "claimed",
    "buy_intent",
    "bought",
    "contract_closed",
    "terminal_skip",
    "signal_expired",
    "buy_blocked_auth",
    "buy_unknown",
)

TERMINAL_SIGNAL_STATUSES = {
    "bought",
    "contract_closed",
    "terminal_skip",
    "signal_expired",
    "buy_blocked_auth",
    "buy_unknown",
}

LEGAL_TRANSITIONS = {
    "queued": {"claimed", "terminal_skip", "signal_expired", "buy_blocked_auth"},
    "claimed": {"terminal_skip", "signal_expired", "buy_blocked_auth", "buy_intent", "queued"},
    "buy_intent": {"bought", "buy_unknown", "buy_blocked_auth", "terminal_skip"},
    "bought": {"contract_closed", "buy_unknown"},
    "contract_closed": set(),
    "terminal_skip": set(),
    "signal_expired": set(),
    "buy_blocked_auth": set(),
    "buy_unknown": {"contract_closed"},
}

CONTRACT_STATUSES = ("open", "sold", "won", "lost", "expired", "cancelled", "unknown")


class QueueError(RuntimeError):
    pass


@dataclass(frozen=True)
class EnqueueResult:
    inserted: bool
    signal_id: str
    conflict_signal_id: str | None = None
    conflict_reason: str | None = None
    commit_to_wakeup_sent_ms: float | None = None


@dataclass(frozen=True)
class ClaimedSignal:
    signal_id: str
    payload: dict[str, Any]
    enqueue_to_claim_ms: float | None
    lease_deadline_utc: str


@dataclass(frozen=True)
class ExposureReservation:
    ok: bool
    reason: str | None = None
    active_same_pair: int = 0
    active_usd_factor: int = 0
    blocking_signal_ids: tuple[str, ...] = ()


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_iso(ts: datetime | None = None) -> str:
    return (ts or utc_now()).astimezone(timezone.utc).isoformat()


def parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    xs = sorted(values)
    return round(xs[min(len(xs) - 1, int(q * (len(xs) - 1)))], 3)


def fixed_defaults() -> dict[str, Any]:
    return {
        "sqlite_busy_timeout_ms": SQLITE_BUSY_TIMEOUT_MS,
        "queue_poll_interval_ms": QUEUE_POLL_INTERVAL_MS,
        "executor_worker_concurrency": EXECUTOR_WORKER_CONCURRENCY,
        "sync_fallback_timeout_s": SYNC_FALLBACK_TIMEOUT_S,
        "sync_fallback_max_concurrency": SYNC_FALLBACK_MAX_CONCURRENCY,
        "async_proposal_timeout_ms": ASYNC_PROPOSAL_TIMEOUT_MS,
        "buy_request_timeout_ms": BUY_REQUEST_TIMEOUT_MS,
        "auth_health_interval_s": AUTH_HEALTH_INTERVAL_S,
        "heartbeat_interval_ms": HEARTBEAT_INTERVAL_MS,
        "event_loop_heartbeat_max_gap_ms": EVENT_LOOP_HEARTBEAT_MAX_GAP_MS,
        "db_transaction_p99_max_ms": DB_TRANSACTION_P99_MAX_MS,
        "db_transaction_max_ms": DB_TRANSACTION_MAX_MS,
        "commit_to_wakeup_received_p95_max_ms": COMMIT_TO_WAKEUP_RECEIVED_P95_MAX_MS,
        "producer_queue_mode_poll_seconds": PRODUCER_QUEUE_MODE_POLL_SECONDS,
        "producer_queue_mode_fanout_limit": PRODUCER_QUEUE_MODE_FANOUT_LIMIT,
        "wall_signal_max_age_s": WALL_SIGNAL_MAX_AGE_S,
        "shifted_signal_max_age_s": SHIFTED_SIGNAL_MAX_AGE_S,
        "hot_proposal_max_buy_age_ms": HOT_PROPOSAL_MAX_BUY_AGE_MS,
        "same_pair_min_gap_seconds": SAME_PAIR_MIN_GAP_SECONDS,
        "max_usd_factor_open": MAX_USD_FACTOR_OPEN,
        "allocation_arbitration_ms": ALLOCATION_ARBITRATION_MS,
        "reconcile_batch_size": RECONCILE_BATCH_SIZE,
        "reconcile_interval_seconds": RECONCILE_INTERVAL_SECONDS,
        "contract_expiry_grace_seconds": CONTRACT_EXPIRY_GRACE_SECONDS,
    }


def signal_id_for(signal: dict[str, Any]) -> str:
    parts = [
        str(signal["pair"]).upper(),
        str(signal["side"]).upper(),
        str(signal.get("lane", "wall")),
        str(int(signal.get("offset_id", 0))),
        str(signal["signal_close_utc"]),
        str(signal["book_id"]),
    ]
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


def payload_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:16]


def _runtime_path(path: Path | str) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def connect_queue(db_path: Path | str = DEFAULT_DB_PATH, *, synchronous: str = "FULL") -> sqlite3.Connection:
    """Open a queue connection and assert the durability PRAGMAs."""
    db = _runtime_path(db_path)
    conn = sqlite3.connect(str(db), timeout=SQLITE_BUSY_TIMEOUT_MS / 1000.0, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute(f"PRAGMA busy_timeout={SQLITE_BUSY_TIMEOUT_MS}")
    conn.execute("PRAGMA foreign_keys=ON")
    mode = conn.execute("PRAGMA journal_mode=WAL").fetchone()[0]
    conn.execute(f"PRAGMA synchronous={synchronous}")
    if str(synchronous).upper() != "FULL":
        conn.execute("PRAGMA application_id=606")
    migrate(conn)
    pragmas = assert_pragmas(conn, require_full=(str(synchronous).upper() == "FULL"))
    if str(mode).lower() != "wal" or pragmas["journal_mode"].lower() != "wal":
        raise QueueError(f"SQLite journal_mode did not become WAL: {mode!r}/{pragmas['journal_mode']!r}")
    return conn


def assert_pragmas(conn: sqlite3.Connection, *, require_full: bool = True) -> dict[str, Any]:
    row = {
        "journal_mode": str(conn.execute("PRAGMA journal_mode").fetchone()[0]),
        "synchronous": int(conn.execute("PRAGMA synchronous").fetchone()[0]),
        "foreign_keys": int(conn.execute("PRAGMA foreign_keys").fetchone()[0]),
        "busy_timeout_ms": int(conn.execute("PRAGMA busy_timeout").fetchone()[0]),
    }
    # SQLite reports FULL as 2.
    ok = row["journal_mode"].lower() == "wal" and row["foreign_keys"] == 1 and row["busy_timeout_ms"] == SQLITE_BUSY_TIMEOUT_MS
    if require_full:
        ok = ok and row["synchronous"] == 2
    if not ok:
        raise QueueError(f"queue PRAGMA assertion failed: {row}")
    return row


def migrate(conn: sqlite3.Connection) -> None:
    status_check = ",".join(f"'{s}'" for s in SIGNAL_STATUSES)
    contract_check = ",".join(f"'{s}'" for s in CONTRACT_STATUSES)
    conn.executescript(
        f"""
        CREATE TABLE IF NOT EXISTS signals (
            signal_id TEXT PRIMARY KEY,
            pair TEXT NOT NULL,
            side TEXT NOT NULL CHECK (side IN ('UP','DOWN')),
            lane TEXT NOT NULL DEFAULT 'wall',
            offset_id INTEGER NOT NULL DEFAULT 0,
            signal_close_utc TEXT NOT NULL,
            book_id TEXT NOT NULL,
            effective_floor REAL NOT NULL,
            payload_json TEXT NOT NULL,
            payload_hash TEXT NOT NULL,
            status TEXT NOT NULL CHECK (status IN ({status_check})),
            producer_enqueue_utc TEXT NOT NULL,
            created_utc TEXT NOT NULL,
            updated_utc TEXT NOT NULL,
            lease_owner TEXT,
            lease_deadline_utc TEXT,
            terminal_reason TEXT,
            terminal_code TEXT,
            terminal_source TEXT,
            terminal_utc TEXT,
            buy_intent_utc TEXT,
            risk_reserved INTEGER NOT NULL DEFAULT 0 CHECK (risk_reserved IN (0,1)),
            reservation_owner TEXT,
            reservation_utc TEXT,
            error TEXT,
            raw_hash TEXT,
            UNIQUE(pair, lane, offset_id, signal_close_utc)
        );

        CREATE TABLE IF NOT EXISTS contracts (
            contract_id TEXT PRIMARY KEY,
            signal_id TEXT NOT NULL UNIQUE REFERENCES signals(signal_id) ON DELETE RESTRICT,
            contract_status TEXT NOT NULL CHECK (contract_status IN ({contract_check})),
            proposal_id TEXT,
            buy_price REAL,
            stake REAL,
            expected_expiry_utc TEXT,
            sell_price REAL,
            profit REAL,
            payout REAL,
            entry_tick REAL,
            exit_tick REAL,
            terminal_utc TEXT,
            last_platform_update_utc TEXT,
            created_utc TEXT NOT NULL,
            updated_utc TEXT NOT NULL,
            raw_hash TEXT
        );

        CREATE TABLE IF NOT EXISTS executor_runs (
            run_id TEXT PRIMARY KEY,
            started_utc TEXT NOT NULL,
            finished_utc TEXT,
            config_json TEXT NOT NULL,
            gate_json TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_signals_claimable
            ON signals(status, lease_deadline_utc, created_utc, signal_id)
            WHERE status = 'queued' OR status = 'claimed';

        CREATE INDEX IF NOT EXISTS idx_signals_status_created
            ON signals(status, created_utc);

        CREATE INDEX IF NOT EXISTS idx_signals_terminal_reason
            ON signals(status, terminal_reason, terminal_utc);

        CREATE INDEX IF NOT EXISTS idx_signals_pair_status_time
            ON signals(pair, status, signal_close_utc);

        CREATE INDEX IF NOT EXISTS idx_contracts_status_expiry
            ON contracts(contract_status, expected_expiry_utc, updated_utc);
        """
    )
    _migrate_contract_columns(conn)


def _migrate_contract_columns(conn: sqlite3.Connection) -> None:
    existing = {row["name"] for row in conn.execute("PRAGMA table_info(contracts)")}
    columns = {
        "sell_price": "REAL",
        "profit": "REAL",
        "payout": "REAL",
        "entry_tick": "REAL",
        "exit_tick": "REAL",
        "terminal_utc": "TEXT",
        "last_platform_update_utc": "TEXT",
    }
    for name, typ in columns.items():
        if name not in existing:
            conn.execute(f"ALTER TABLE contracts ADD COLUMN {name} {typ}")


def _begin_immediate(conn: sqlite3.Connection) -> float:
    t0 = time.perf_counter()
    conn.execute("BEGIN IMMEDIATE")
    return t0


def _finish_tx(conn: sqlite3.Connection, t0: float, stats: list[float] | None = None) -> float:
    conn.execute("COMMIT")
    ms = (time.perf_counter() - t0) * 1000.0
    if stats is not None:
        stats.append(ms)
    return ms


def _rollback(conn: sqlite3.Connection) -> None:
    try:
        conn.execute("ROLLBACK")
    except sqlite3.OperationalError:
        pass


def enqueue_signal(
    conn: sqlite3.Connection,
    signal: dict[str, Any],
    *,
    now: datetime | None = None,
    tx_stats: list[float] | None = None,
) -> EnqueueResult:
    """Idempotently enqueue one selected-side signal.

    The natural economic key excludes `side` and `book_id`, so a metadata
    change or selected-side flip for one pair/bar cannot double-buy it.
    """
    ts = now or utc_now()
    payload = dict(signal)
    payload["pair"] = str(payload["pair"]).upper()
    payload["side"] = str(payload["side"]).upper()
    payload.setdefault("lane", "wall")
    payload.setdefault("offset_id", 0)
    if payload["side"] not in {"UP", "DOWN"}:
        raise QueueError(f"invalid side {payload['side']!r}")
    sid = signal_id_for(payload)
    enq_utc = utc_iso(ts)
    t0 = _begin_immediate(conn)
    try:
        row = conn.execute(
            """
            INSERT INTO signals (
                signal_id, pair, side, lane, offset_id, signal_close_utc, book_id,
                effective_floor, payload_json, payload_hash, status,
                producer_enqueue_utc, created_utc, updated_utc
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'queued', ?, ?, ?)
            ON CONFLICT(pair, lane, offset_id, signal_close_utc) DO NOTHING
            RETURNING signal_id
            """,
            (
                sid,
                payload["pair"],
                payload["side"],
                str(payload.get("lane", "wall")),
                int(payload.get("offset_id", 0)),
                str(payload["signal_close_utc"]),
                str(payload["book_id"]),
                float(payload["effective_floor"]),
                json.dumps(payload, sort_keys=True, default=str),
                payload_hash(payload),
                enq_utc,
                enq_utc,
                enq_utc,
            ),
        ).fetchone()
        _finish_tx(conn, t0, tx_stats)
    except Exception:
        _rollback(conn)
        raise
    if row is not None:
        return EnqueueResult(inserted=True, signal_id=sid)
    existing = get_by_natural_key(
        conn,
        pair=payload["pair"],
        lane=str(payload.get("lane", "wall")),
        offset_id=int(payload.get("offset_id", 0)),
        signal_close_utc=str(payload["signal_close_utc"]),
    )
    if existing is None:
        raise QueueError("enqueue conflict occurred but existing natural row was not found")
    reason = "duplicate_signal_id" if existing["signal_id"] == sid else "selected_side_natural_key_conflict"
    return EnqueueResult(False, sid, conflict_signal_id=existing["signal_id"], conflict_reason=reason)


def get_by_natural_key(
    conn: sqlite3.Connection,
    *,
    pair: str,
    lane: str,
    offset_id: int,
    signal_close_utc: str,
) -> sqlite3.Row | None:
    return conn.execute(
        """
        SELECT * FROM signals
        WHERE pair = ? AND lane = ? AND offset_id = ? AND signal_close_utc = ?
        """,
        (pair, lane, int(offset_id), signal_close_utc),
    ).fetchone()


def get_signal(conn: sqlite3.Connection, signal_id: str) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM signals WHERE signal_id = ?", (signal_id,)).fetchone()


def claim_next(
    conn: sqlite3.Connection,
    *,
    owner: str,
    lease_seconds: float = 30.0,
    now: datetime | None = None,
    tx_stats: list[float] | None = None,
) -> ClaimedSignal | None:
    ts = now or utc_now()
    now_s = utc_iso(ts)
    deadline = utc_iso(ts + timedelta(seconds=lease_seconds))
    t0 = _begin_immediate(conn)
    try:
        row = conn.execute(
            """
            SELECT * FROM signals
            WHERE status = 'queued'
               OR (status = 'claimed' AND (lease_deadline_utc IS NULL OR lease_deadline_utc <= ?))
            ORDER BY created_utc, signal_id
            LIMIT 1
            """,
            (now_s,),
        ).fetchone()
        if row is None:
            _finish_tx(conn, t0, tx_stats)
            return None
        conn.execute(
            """
            UPDATE signals
            SET status = 'claimed', lease_owner = ?, lease_deadline_utc = ?, updated_utc = ?
            WHERE signal_id = ? AND (
                status = 'queued'
                OR (status = 'claimed' AND (lease_deadline_utc IS NULL OR lease_deadline_utc <= ?))
            )
            """,
            (owner, deadline, now_s, row["signal_id"], now_s),
        )
        updated = get_signal(conn, row["signal_id"])
        _finish_tx(conn, t0, tx_stats)
    except Exception:
        _rollback(conn)
        raise
    if updated is None:
        return None
    try:
        enq = parse_utc(updated["producer_enqueue_utc"])
        enqueue_to_claim_ms = (ts - enq).total_seconds() * 1000.0
    except Exception:
        enqueue_to_claim_ms = None
    return ClaimedSignal(
        signal_id=updated["signal_id"],
        payload=json.loads(updated["payload_json"]),
        enqueue_to_claim_ms=round(enqueue_to_claim_ms, 3) if enqueue_to_claim_ms is not None else None,
        lease_deadline_utc=updated["lease_deadline_utc"],
    )


def _claimed_from_row(row: sqlite3.Row, claimed_at: datetime) -> ClaimedSignal:
    try:
        enq = parse_utc(row["producer_enqueue_utc"])
        enqueue_to_claim_ms = (claimed_at - enq).total_seconds() * 1000.0
    except Exception:
        enqueue_to_claim_ms = None
    return ClaimedSignal(
        signal_id=row["signal_id"],
        payload=json.loads(row["payload_json"]),
        enqueue_to_claim_ms=round(enqueue_to_claim_ms, 3) if enqueue_to_claim_ms is not None else None,
        lease_deadline_utc=row["lease_deadline_utc"],
    )


def claim_batch(
    conn: sqlite3.Connection,
    *,
    owner: str,
    limit: int,
    lease_seconds: float = 30.0,
    now: datetime | None = None,
    tx_stats: list[float] | None = None,
) -> list[ClaimedSignal]:
    if limit <= 0:
        return []
    ts = now or utc_now()
    now_s = utc_iso(ts)
    deadline = utc_iso(ts + timedelta(seconds=lease_seconds))
    t0 = _begin_immediate(conn)
    try:
        rows = conn.execute(
            """
            SELECT * FROM signals
            WHERE status = 'queued'
               OR (status = 'claimed' AND (lease_deadline_utc IS NULL OR lease_deadline_utc <= ?))
            ORDER BY created_utc, signal_id
            LIMIT ?
            """,
            (now_s, int(limit)),
        ).fetchall()
        if not rows:
            _finish_tx(conn, t0, tx_stats)
            return []
        ids = [row["signal_id"] for row in rows]
        placeholders = ",".join("?" for _ in ids)
        conn.execute(
            f"""
            UPDATE signals
            SET status = 'claimed', lease_owner = ?, lease_deadline_utc = ?, updated_utc = ?
            WHERE signal_id IN ({placeholders}) AND (
                status = 'queued'
                OR (status = 'claimed' AND (lease_deadline_utc IS NULL OR lease_deadline_utc <= ?))
            )
            """,
            (owner, deadline, now_s, *ids, now_s),
        )
        updated = conn.execute(
            f"SELECT * FROM signals WHERE signal_id IN ({placeholders}) ORDER BY created_utc, signal_id",
            ids,
        ).fetchall()
        _finish_tx(conn, t0, tx_stats)
    except Exception:
        _rollback(conn)
        raise
    return [_claimed_from_row(row, ts) for row in updated if row["status"] == "claimed"]


def transition_signal(
    conn: sqlite3.Connection,
    signal_id: str,
    from_statuses: Iterable[str],
    to_status: str,
    *,
    reason: str | None = None,
    code: str | None = None,
    source: str | None = None,
    error: str | None = None,
    raw_hash_value: str | None = None,
    now: datetime | None = None,
    tx_stats: list[float] | None = None,
) -> bool:
    """The only supported queue-status mutator."""
    allowed_from = set(from_statuses)
    if to_status not in SIGNAL_STATUSES:
        raise QueueError(f"unknown target status {to_status!r}")
    ts = now or utc_now()
    now_s = utc_iso(ts)
    t0 = _begin_immediate(conn)
    try:
        row = conn.execute("SELECT status FROM signals WHERE signal_id = ?", (signal_id,)).fetchone()
        if row is None:
            _finish_tx(conn, t0, tx_stats)
            return False
        current = row["status"]
        if current not in allowed_from:
            _finish_tx(conn, t0, tx_stats)
            return False
        if to_status not in LEGAL_TRANSITIONS[current]:
            raise QueueError(f"invalid transition {current}->{to_status}")
        terminal_utc = now_s if to_status in TERMINAL_SIGNAL_STATUSES else None
        buy_intent_utc = now_s if to_status == "buy_intent" else None
        conn.execute(
            """
            UPDATE signals
            SET status = ?,
                updated_utc = ?,
                terminal_reason = COALESCE(?, terminal_reason),
                terminal_code = COALESCE(?, terminal_code),
                terminal_source = COALESCE(?, terminal_source),
                terminal_utc = COALESCE(?, terminal_utc),
                buy_intent_utc = COALESCE(?, buy_intent_utc),
                error = COALESCE(?, error),
                raw_hash = COALESCE(?, raw_hash)
            WHERE signal_id = ? AND status = ?
            """,
            (to_status, now_s, reason, code, source, terminal_utc, buy_intent_utc, error, raw_hash_value, signal_id, current),
        )
        _finish_tx(conn, t0, tx_stats)
        return True
    except Exception:
        _rollback(conn)
        raise


def reserve_risk(
    conn: sqlite3.Connection,
    signal_id: str,
    *,
    owner: str,
    max_open: int,
    now: datetime | None = None,
    tx_stats: list[float] | None = None,
) -> bool:
    """Reserve one exposure slot without holding a DB lock across network I/O."""
    if max_open <= 0:
        return True
    ts = now or utc_now()
    now_s = utc_iso(ts)
    t0 = _begin_immediate(conn)
    try:
        open_count = conn.execute(
            """
            SELECT COUNT(*) FROM signals
            WHERE risk_reserved = 1
              AND status NOT IN ('terminal_skip','signal_expired','buy_blocked_auth','contract_closed')
            """
        ).fetchone()[0]
        if open_count >= max_open:
            _finish_tx(conn, t0, tx_stats)
            return False
        cur = conn.execute(
            """
            UPDATE signals
            SET risk_reserved = 1, reservation_owner = ?, reservation_utc = ?, updated_utc = ?
            WHERE signal_id = ? AND status = 'claimed' AND risk_reserved = 0
            """,
            (owner, now_s, now_s, signal_id),
        )
        _finish_tx(conn, t0, tx_stats)
        return cur.rowcount == 1
    except Exception:
        _rollback(conn)
        raise


def usd_factor_direction(pair: str, side: str) -> str:
    pair_u = str(pair).upper()
    side_u = str(side).upper()
    if side_u not in {"UP", "DOWN"}:
        raise QueueError(f"invalid side {side!r}")
    if pair_u.startswith("USD"):
        return "USD_STRENGTH" if side_u == "UP" else "USD_WEAKNESS"
    return "USD_STRENGTH" if side_u == "DOWN" else "USD_WEAKNESS"


def active_exposure_rows(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute(
        """
        SELECT
            s.signal_id, s.pair, s.side, s.status, s.signal_close_utc, s.risk_reserved,
            c.contract_id, c.contract_status, c.expected_expiry_utc, c.terminal_utc
        FROM signals s
        LEFT JOIN contracts c ON c.signal_id = s.signal_id
        WHERE (
            s.status IN ('buy_intent','bought','buy_unknown')
            OR (s.status = 'claimed' AND s.risk_reserved = 1)
        )
          AND s.status != 'contract_closed'
          AND (c.contract_status IS NULL OR c.contract_status IN ('open','unknown'))
        ORDER BY s.updated_utc, s.signal_id
        """
    ).fetchall()


def reserve_allocator_exposure(
    conn: sqlite3.Connection,
    signal_id: str,
    *,
    owner: str,
    same_pair_min_gap_seconds: int = SAME_PAIR_MIN_GAP_SECONDS,
    max_usd_factor_open: int = MAX_USD_FACTOR_OPEN,
    now: datetime | None = None,
    tx_stats: list[float] | None = None,
) -> ExposureReservation:
    """Atomically reserve allocator exposure for one claimed signal."""
    ts = now or utc_now()
    now_s = utc_iso(ts)
    t0 = _begin_immediate(conn)
    try:
        row = conn.execute("SELECT * FROM signals WHERE signal_id = ?", (signal_id,)).fetchone()
        if row is None:
            _finish_tx(conn, t0, tx_stats)
            return ExposureReservation(False, "allocation_missing_signal")
        if row["status"] != "claimed" or int(row["risk_reserved"]) != 0:
            _finish_tx(conn, t0, tx_stats)
            return ExposureReservation(False, "allocation_not_claimed")
        pair = str(row["pair"]).upper()
        side = str(row["side"]).upper()
        usd_dir = usd_factor_direction(pair, side)
        active = [r for r in active_exposure_rows(conn) if r["signal_id"] != signal_id]
        same_pair_rows = [r for r in active if str(r["pair"]).upper() == pair]
        if same_pair_min_gap_seconds > 0 and same_pair_rows:
            _finish_tx(conn, t0, tx_stats)
            return ExposureReservation(
                False,
                "allocation_same_pair_overlap",
                active_same_pair=len(same_pair_rows),
                active_usd_factor=sum(1 for r in active if usd_factor_direction(r["pair"], r["side"]) == usd_dir),
                blocking_signal_ids=tuple(str(r["signal_id"]) for r in same_pair_rows[:5]),
            )
        usd_rows = [r for r in active if usd_factor_direction(r["pair"], r["side"]) == usd_dir]
        if max_usd_factor_open > 0 and len(usd_rows) >= max_usd_factor_open:
            _finish_tx(conn, t0, tx_stats)
            return ExposureReservation(
                False,
                "allocation_usd_factor_cap",
                active_same_pair=len(same_pair_rows),
                active_usd_factor=len(usd_rows),
                blocking_signal_ids=tuple(str(r["signal_id"]) for r in usd_rows[:5]),
            )
        cur = conn.execute(
            """
            UPDATE signals
            SET risk_reserved = 1, reservation_owner = ?, reservation_utc = ?, updated_utc = ?
            WHERE signal_id = ? AND status = 'claimed' AND risk_reserved = 0
            """,
            (owner, now_s, now_s, signal_id),
        )
        _finish_tx(conn, t0, tx_stats)
    except Exception:
        _rollback(conn)
        raise
    if cur.rowcount != 1:
        return ExposureReservation(False, "allocation_reservation_lost")
    return ExposureReservation(
        True,
        active_same_pair=len(same_pair_rows),
        active_usd_factor=len(usd_rows),
    )


def release_risk(
    conn: sqlite3.Connection,
    signal_id: str,
    *,
    now: datetime | None = None,
    tx_stats: list[float] | None = None,
) -> bool:
    ts = now or utc_now()
    t0 = _begin_immediate(conn)
    try:
        cur = conn.execute(
            """
            UPDATE signals
            SET risk_reserved = 0, reservation_owner = NULL, reservation_utc = NULL, updated_utc = ?
            WHERE signal_id = ?
            """,
            (utc_iso(ts), signal_id),
        )
        _finish_tx(conn, t0, tx_stats)
        return cur.rowcount == 1
    except Exception:
        _rollback(conn)
        raise


def record_contract(
    conn: sqlite3.Connection,
    *,
    signal_id: str,
    contract_id: str,
    contract_status: str = "open",
    proposal_id: str | None = None,
    buy_price: float | None = None,
    stake: float | None = None,
    expected_expiry_utc: str | None = None,
    raw_hash_value: str | None = None,
    now: datetime | None = None,
    tx_stats: list[float] | None = None,
) -> None:
    if contract_status not in CONTRACT_STATUSES:
        raise QueueError(f"invalid contract status {contract_status!r}")
    ts = utc_iso(now or utc_now())
    t0 = _begin_immediate(conn)
    try:
        conn.execute(
            """
            INSERT INTO contracts (
                contract_id, signal_id, contract_status, proposal_id, buy_price, stake,
                expected_expiry_utc, last_platform_update_utc, created_utc, updated_utc, raw_hash
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (str(contract_id), signal_id, contract_status, proposal_id, buy_price, stake, expected_expiry_utc, ts, ts, ts, raw_hash_value),
        )
        _finish_tx(conn, t0, tx_stats)
    except Exception:
        _rollback(conn)
        raise


def _number_or_none(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _terminal_contract_status(poc: dict[str, Any], current: str) -> tuple[str, bool]:
    raw = str(poc.get("status") or current or "open").lower()
    if raw not in CONTRACT_STATUSES:
        raw = "unknown"
    terminal = bool(poc.get("is_sold")) or raw in {"sold", "won", "lost", "expired", "cancelled"}
    if terminal and raw in {"open", "unknown"}:
        raw = "sold"
    return raw, terminal


def update_contract_from_proposal(
    conn: sqlite3.Connection,
    *,
    contract_id: str,
    proposal_open_contract: dict[str, Any],
    raw_hash_value: str | None = None,
    now: datetime | None = None,
    tx_stats: list[float] | None = None,
) -> dict[str, Any]:
    """Persist proposal_open_contract state and close the linked signal on terminal evidence."""
    poc = proposal_open_contract.get("proposal_open_contract", proposal_open_contract)
    remote_id = poc.get("contract_id")
    if remote_id is not None and str(remote_id) != str(contract_id):
        raise QueueError(f"proposal_open_contract id mismatch: {remote_id!r} != {contract_id!r}")
    ts = utc_iso(now or utc_now())
    t0 = _begin_immediate(conn)
    try:
        row = conn.execute("SELECT * FROM contracts WHERE contract_id = ?", (str(contract_id),)).fetchone()
        if row is None:
            _finish_tx(conn, t0, tx_stats)
            return {"updated": False, "reason": "contract_missing", "terminal": False}
        status, terminal = _terminal_contract_status(poc, row["contract_status"])
        terminal_utc = ts if terminal else row["terminal_utc"]
        conn.execute(
            """
            UPDATE contracts
            SET contract_status = ?,
                sell_price = COALESCE(?, sell_price),
                profit = COALESCE(?, profit),
                payout = COALESCE(?, payout),
                entry_tick = COALESCE(?, entry_tick),
                exit_tick = COALESCE(?, exit_tick),
                terminal_utc = COALESCE(?, terminal_utc),
                last_platform_update_utc = ?,
                updated_utc = ?,
                raw_hash = COALESCE(?, raw_hash)
            WHERE contract_id = ?
            """,
            (
                status,
                _number_or_none(poc.get("sell_price")),
                _number_or_none(poc.get("profit")),
                _number_or_none(poc.get("payout")),
                _number_or_none(poc.get("entry_tick")),
                _number_or_none(poc.get("exit_tick")),
                terminal_utc,
                ts,
                ts,
                raw_hash_value,
                str(contract_id),
            ),
        )
        closed_signal = False
        if terminal:
            cur = conn.execute(
                """
                UPDATE signals
                SET status = 'contract_closed',
                    updated_utc = ?,
                    terminal_reason = COALESCE(terminal_reason, 'contract_terminal'),
                    terminal_source = COALESCE(terminal_source, 'proposal_open_contract'),
                    terminal_utc = COALESCE(terminal_utc, ?),
                    raw_hash = COALESCE(?, raw_hash)
                WHERE signal_id = ? AND status IN ('bought','buy_unknown')
                """,
                (ts, ts, raw_hash_value, row["signal_id"]),
            )
            closed_signal = cur.rowcount == 1
        _finish_tx(conn, t0, tx_stats)
        return {
            "updated": True,
            "contract_id": str(contract_id),
            "signal_id": row["signal_id"],
            "contract_status": status,
            "terminal": terminal,
            "closed_signal": closed_signal,
        }
    except Exception:
        _rollback(conn)
        raise


def open_contracts_for_reconcile(
    conn: sqlite3.Connection,
    *,
    limit: int,
    now: datetime | None = None,
    expiry_grace_seconds: float = CONTRACT_EXPIRY_GRACE_SECONDS,
) -> list[sqlite3.Row]:
    if limit <= 0:
        return []
    eligible_before = utc_iso((now or utc_now()) - timedelta(seconds=max(0.0, float(expiry_grace_seconds))))
    return conn.execute(
        """
        SELECT
            c.*, s.pair, s.side, s.status AS signal_status, s.signal_close_utc
        FROM contracts c
        JOIN signals s ON s.signal_id = c.signal_id
        WHERE c.contract_status IN ('open','unknown')
          AND s.status IN ('bought','buy_unknown')
          AND (c.expected_expiry_utc IS NULL OR c.expected_expiry_utc <= ?)
        ORDER BY COALESCE(c.last_platform_update_utc, c.created_utc), c.contract_id
        LIMIT ?
        """,
        (eligible_before, int(limit)),
    ).fetchall()


def queue_counts(conn: sqlite3.Connection) -> dict[str, int]:
    return {
        row["status"]: int(row["n"])
        for row in conn.execute("SELECT status, COUNT(*) AS n FROM signals GROUP BY status")
    }


def duplicate_counts(conn: sqlite3.Connection) -> dict[str, int]:
    natural = conn.execute(
        """
        SELECT COUNT(*) FROM (
            SELECT pair, lane, offset_id, signal_close_utc, COUNT(*) AS n
            FROM signals GROUP BY pair, lane, offset_id, signal_close_utc HAVING n > 1
        )
        """
    ).fetchone()[0]
    contracts = conn.execute(
        """
        SELECT COUNT(*) FROM (
            SELECT signal_id, COUNT(*) AS n FROM contracts GROUP BY signal_id HAVING n > 1
        )
        """
    ).fetchone()[0]
    return {"selected_side_natural_tuple": int(natural), "contracts_per_signal": int(contracts)}


def send_wakeup(socket_path: Path | str = DEFAULT_WAKEUP_PATH, *, payload: dict[str, Any] | None = None) -> bool:
    """Best-effort Unix datagram wake-up. Failure does not affect correctness."""
    path = Path(socket_path)
    if not path.exists():
        return False
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    try:
        sock.settimeout(0.05)
        body = json.dumps(payload or {"event": "queue_wakeup", "timestamp_utc": utc_iso()}, sort_keys=True).encode("utf-8")
        sock.sendto(body, str(path))
        return True
    except OSError:
        return False
    finally:
        sock.close()


def write_gate_result(path: Path, doc: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(doc, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    tmp.replace(path)


def _sample_signal(i: int, *, side: str = "UP", pair: str = "USDJPY", book_id: str = "USDJPY.m15ny_seedens.v1") -> dict[str, Any]:
    return {
        "pair": pair,
        "side": side,
        "lane": "wall",
        "offset_id": 0,
        "signal_close_utc": f"2026-07-01T18:{i // 60:02d}:{i % 60:02d}+00:00",
        "book_id": book_id,
        "effective_floor": 0.58,
        "bar_age_s": 1.0,
        "is_latest_bar": True,
    }


def run_smoke() -> int:
    failures: list[str] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        print(f"{'PASS' if ok else 'FAIL'}: {name}" + (f" - {detail}" if detail else ""))
        if not ok:
            failures.append(name)

    root = Path(tempfile.mkdtemp(prefix="deriv_trade_queue_"))
    conn = connect_queue(root / "queue.sqlite")
    tx_ms: list[float] = []
    pragmas = assert_pragmas(conn)
    check("SQLite PRAGMAs asserted", pragmas["journal_mode"].lower() == "wal" and pragmas["synchronous"] == 2)

    sig = _sample_signal(0)
    r1 = enqueue_signal(conn, sig, tx_stats=tx_ms)
    r2 = enqueue_signal(conn, sig, tx_stats=tx_ms)
    check("duplicate enqueue returns existing row", r1.inserted and not r2.inserted and r2.conflict_signal_id == r1.signal_id)
    flipped = dict(sig, side="DOWN", book_id="different-book")
    r3 = enqueue_signal(conn, flipped, tx_stats=tx_ms)
    check("selected-side natural tuple excludes side/book_id", not r3.inserted and r3.conflict_reason == "selected_side_natural_key_conflict")

    claim = claim_next(conn, owner="worker-1", tx_stats=tx_ms)
    check("claim returns queued row", claim is not None and claim.signal_id == r1.signal_id)
    assert claim is not None
    bad = transition_signal(conn, claim.signal_id, ["queued"], "buy_intent", tx_stats=tx_ms)
    check("CAS rejects wrong from_status", bad is False)
    ok = transition_signal(conn, claim.signal_id, ["claimed"], "terminal_skip", reason="unit_smoke", source="smoke", tx_stats=tx_ms)
    check("claimed -> terminal_skip transition works", ok and get_signal(conn, claim.signal_id)["status"] == "terminal_skip")
    try:
        transition_signal(conn, claim.signal_id, ["terminal_skip"], "claimed", tx_stats=tx_ms)
    except QueueError:
        invalid_ok = True
    else:
        invalid_ok = False
    check("invalid transition raises", invalid_ok)

    for i in range(1, 13):
        enqueue_signal(conn, _sample_signal(i), tx_stats=tx_ms)
    claims = [claim_next(conn, owner=f"w{i}", tx_stats=tx_ms) for i in range(12)]
    check("12-row burst can be claimed before release", all(c is not None for c in claims) and len({c.signal_id for c in claims if c}) == 12)

    batch_conn = connect_queue(root / "batch.sqlite")
    tx_batch: list[float] = []
    for i in range(20, 26):
        enqueue_signal(batch_conn, _sample_signal(i), tx_stats=tx_batch)
    batch = claim_batch(batch_conn, owner="batch", limit=4, tx_stats=tx_batch)
    check("claim_batch claims bounded FIFO subset", len(batch) == 4 and queue_counts(batch_conn).get("claimed") == 4 and queue_counts(batch_conn).get("queued") == 2)

    risk_conn = connect_queue(root / "risk.sqlite")
    tx2: list[float] = []
    s1 = enqueue_signal(risk_conn, _sample_signal(1), tx_stats=tx2).signal_id
    s2 = enqueue_signal(risk_conn, _sample_signal(2), tx_stats=tx2).signal_id
    claim_next(risk_conn, owner="a", tx_stats=tx2)
    claim_next(risk_conn, owner="b", tx_stats=tx2)
    check("max_open=1 first reservation succeeds", reserve_risk(risk_conn, s1, owner="a", max_open=1, tx_stats=tx2))
    check("max_open=1 second reservation blocks", not reserve_risk(risk_conn, s2, owner="b", max_open=1, tx_stats=tx2))

    alloc_conn = connect_queue(root / "allocator.sqlite")
    tx_alloc: list[float] = []
    alloc_signals = [
        _sample_signal(30, pair="USDJPY", side="UP"),
        _sample_signal(31, pair="USDCAD", side="UP", book_id="USDCAD.m15ny_seedens.v1"),
        _sample_signal(32, pair="GBPUSD", side="DOWN", book_id="GBPUSD.m15ny_xpair_seedens8.v1"),
    ]
    alloc_ids = [enqueue_signal(alloc_conn, sig, tx_stats=tx_alloc).signal_id for sig in alloc_signals]
    claim_batch(alloc_conn, owner="alloc", limit=3, tx_stats=tx_alloc)
    r_a = reserve_allocator_exposure(alloc_conn, alloc_ids[0], owner="alloc", max_usd_factor_open=2, tx_stats=tx_alloc)
    r_b = reserve_allocator_exposure(alloc_conn, alloc_ids[1], owner="alloc", max_usd_factor_open=2, tx_stats=tx_alloc)
    r_c = reserve_allocator_exposure(alloc_conn, alloc_ids[2], owner="alloc", max_usd_factor_open=2, tx_stats=tx_alloc)
    check("allocator USD-factor cap blocks third USD-strength exposure", r_a.ok and r_b.ok and not r_c.ok and r_c.reason == "allocation_usd_factor_cap")

    pair_conn = connect_queue(root / "allocator_pair.sqlite")
    tx_pair: list[float] = []
    p1 = enqueue_signal(pair_conn, _sample_signal(40, pair="AUDUSD", side="DOWN", book_id="AUDUSD.m15ny_seedens.v1"), tx_stats=tx_pair).signal_id
    p2 = enqueue_signal(pair_conn, _sample_signal(41, pair="AUDUSD", side="DOWN", book_id="AUDUSD.m15ny_seedens.v1"), tx_stats=tx_pair).signal_id
    claim_batch(pair_conn, owner="pair", limit=2, tx_stats=tx_pair)
    p1r = reserve_allocator_exposure(pair_conn, p1, owner="pair", max_usd_factor_open=0, tx_stats=tx_pair)
    p2r = reserve_allocator_exposure(pair_conn, p2, owner="pair", max_usd_factor_open=0, tx_stats=tx_pair)
    check("allocator same-pair exposure blocks second same-pair reservation", p1r.ok and not p2r.ok and p2r.reason == "allocation_same_pair_overlap")

    settle_conn = connect_queue(root / "settlement.sqlite")
    tx_settle: list[float] = []
    sid = enqueue_signal(settle_conn, _sample_signal(50), tx_stats=tx_settle).signal_id
    claim_next(settle_conn, owner="settle", tx_stats=tx_settle)
    transition_signal(settle_conn, sid, ["claimed"], "buy_intent", source="smoke", tx_stats=tx_settle)
    record_contract(settle_conn, signal_id=sid, contract_id="contract-smoke", contract_status="open", tx_stats=tx_settle)
    transition_signal(settle_conn, sid, ["buy_intent"], "bought", reason="smoke_buy", source="smoke", tx_stats=tx_settle)
    upd = update_contract_from_proposal(
        settle_conn,
        contract_id="contract-smoke",
        proposal_open_contract={
            "proposal_open_contract": {
                "contract_id": "contract-smoke",
                "status": "sold",
                "is_sold": 1,
                "profit": "0.85",
                "sell_price": "1.85",
                "payout": "1.85",
            }
        },
        tx_stats=tx_settle,
    )
    closed = get_signal(settle_conn, sid)["status"] == "contract_closed"
    check("settlement update closes bought signal and stores terminal fields", upd["terminal"] and upd["closed_signal"] and closed)

    future_conn = connect_queue(root / "future_reconcile.sqlite")
    tx_future: list[float] = []
    future_id = enqueue_signal(future_conn, _sample_signal(60), tx_stats=tx_future).signal_id
    claim_next(future_conn, owner="future", tx_stats=tx_future)
    transition_signal(future_conn, future_id, ["claimed"], "buy_intent", source="smoke", tx_stats=tx_future)
    record_contract(
        future_conn,
        signal_id=future_id,
        contract_id="future-contract-smoke",
        contract_status="open",
        expected_expiry_utc=utc_iso(datetime.now(timezone.utc) + timedelta(minutes=10)),
        tx_stats=tx_future,
    )
    transition_signal(future_conn, future_id, ["buy_intent"], "bought", reason="smoke_buy", source="smoke", tx_stats=tx_future)
    future_rows = open_contracts_for_reconcile(future_conn, limit=12, expiry_grace_seconds=600)
    check("reconciliation does not poll before expiry-plus-grace", len(future_rows) == 0)

    aged = connect_queue(root / "aged.sqlite")
    tx3: list[float] = []
    now = utc_now()
    aged.execute("BEGIN IMMEDIATE")
    for i in range(100_000):
        sig_i = _sample_signal(i + 1000, pair="AUDUSD")
        sid_i = signal_id_for(sig_i)
        ts = utc_iso(now - timedelta(days=2))
        aged.execute(
            """
            INSERT INTO signals (
                signal_id, pair, side, lane, offset_id, signal_close_utc, book_id, effective_floor,
                payload_json, payload_hash, status, producer_enqueue_utc, created_utc, updated_utc,
                terminal_reason, terminal_source, terminal_utc
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'terminal_skip', ?, ?, ?, 'aged', 'smoke', ?)
            """,
            (
                sid_i,
                sig_i["pair"],
                sig_i["side"],
                sig_i["lane"],
                sig_i["offset_id"],
                sig_i["signal_close_utc"],
                sig_i["book_id"],
                sig_i["effective_floor"],
                json.dumps(sig_i, sort_keys=True),
                payload_hash(sig_i),
                ts,
                ts,
                ts,
                ts,
            ),
        )
    aged.execute("COMMIT")
    for i in range(12):
        enqueue_signal(aged, _sample_signal(i, pair="GBPUSD"), now=now, tx_stats=tx3)
    t_claim = time.perf_counter()
    aged_claims = [claim_next(aged, owner=f"aged-{i}", now=now + timedelta(milliseconds=i), tx_stats=tx3) for i in range(12)]
    aged_ms = (time.perf_counter() - t_claim) * 1000.0
    check("aged queue claims 12 fresh rows with 100k retained terminals", all(c is not None for c in aged_claims), f"{aged_ms:.1f}ms")

    dups = duplicate_counts(conn)
    check("duplicate counters are zero by DB constraint", dups["selected_side_natural_tuple"] == 0 and dups["contracts_per_signal"] == 0)

    all_tx = tx_ms + tx_batch + tx2 + tx_alloc + tx_pair + tx_settle + tx_future + tx3
    p99 = pct(all_tx, 0.99)
    max_ms = max(all_tx)
    check("DB transaction max within gate", max_ms <= DB_TRANSACTION_MAX_MS, f"max={max_ms:.3f}ms p99={p99}")

    gate = {
        "gate": "issue#6 deriv_trade_queue deterministic smoke",
        "generated_utc": utc_iso(),
        "fixed_defaults": fixed_defaults(),
        "sqlite_pragmas": pragmas,
        "fake": {
            "claimed_before_release": 12,
            "batch_claimed": len(batch),
            "aged_terminal_rows": 100_000,
            "aged_claim_12_wall_ms": round(aged_ms, 3),
            "db_transaction_ms_p99": p99,
            "db_transaction_ms_max": round(max_ms, 3),
            "duplicate_counts": dups,
            "allocator_usd_cap_reason": r_c.reason,
            "allocator_same_pair_reason": p2r.reason,
            "settlement_update_terminal": upd["terminal"],
            "reconcile_future_before_grace_count": len(future_rows),
            "queue_counts": queue_counts(conn),
        },
        "live": {
            "ny_session_demo_buy": None,
            "live_forced_concurrency_passed": None,
            "live_forced_buy_overlap_passed": None,
            "live_sync_fallback_terminalization_passed": None,
            "live_sync_fallback_buy_submission_passed": None,
        },
        "closure_ready": False,
        "blocking_gates": [
            "live_demo_session_gate_unmeasured",
            "live_forced_concurrency_gate_unmeasured",
            "live_forced_fallback_gate_unmeasured",
        ],
    }
    write_gate_result(GATE_RESULT_PATH, gate)
    ok = not failures
    print(f"\n{'SMOKE ALL PASS' if ok else f'SMOKE {len(failures)} FAILURES: {failures}'}")
    print(f"gate_result={GATE_RESULT_PATH}")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    p.add_argument("--assert-pragmas", action="store_true")
    args = p.parse_args(argv)
    if args.smoke:
        return run_smoke()
    if args.assert_pragmas:
        conn = connect_queue(args.db)
        print(json.dumps(assert_pragmas(conn), indent=2, sort_keys=True))
        return 0
    p.error("pass --smoke or --assert-pragmas")
    return 2


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except QueueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
