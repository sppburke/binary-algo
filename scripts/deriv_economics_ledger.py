#!/usr/bin/env python3
"""Seal, export, and reduce one fixed Deriv demo Rise/Fall settlement.

The exporter is deliberately a credential-free, read-only record supplier.  It
does not import any Deriv runtime module.  The reducer is the only component
that interprets terminal economics, after the supplied packet has been sealed
and an access receipt has been durably published.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sqlite3
import stat
import sys
import tempfile
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Sequence
from zoneinfo import ZoneInfo

import evidence_store as evidence


REPO_ROOT = Path(__file__).resolve().parents[1]
ACQUISITION_KIND = "research.deriv_economics.acquisition_spec/v1"
SEALED_KIND = "research.deriv_economics.sealed_source/v1"
ACCESS_KIND = "research.deriv_economics.source_access_receipt/v1"
NORMALIZED_KIND = "research.deriv_economics.normalized_receipt/v1"
PACKET_SCHEMA = "deriv-demo-rise-fall-chain/v1"
ACCESS_STATE = "spent_may_have_been_read"

SOURCE_DATE = "2026-07-06"
SOURCE_TZ = "America/New_York"
SOURCE_PAIR = "USDJPY"
SOURCE_SIDE = "UP"
RUNTIME_REVISION = "c57512ad2cd77beeb9c57276b73505a75c874c62"
DB_LOGICAL_PATH = "deriv_data/runtime/trade_queue.sqlite"
LOG_LOGICAL_PATH = "logs/paper_trades/trade_executor/2026-07-06.jsonl.1"
REMOTE_ROOT = Path("/home/sean/binary-algo")
EXPORTER_PATH = "scripts/deriv_economics_ledger.py"
EVIDENCE_PATH = "scripts/evidence_store.py"
MANIFEST_PATH = "scripts/manifest.py"
BOUND_ARTIFACTS = (EXPORTER_PATH, EVIDENCE_PATH, MANIFEST_PATH)

SOURCE_KEYS = {
    "executor_startup": frozenset(
        {
            "absolute_breakeven_ceiling", "cutoff", "event", "fixed_defaults",
            "lock_path", "lock_root", "queue_db", "timestamp_utc",
        }
    ),
    "exec_client_connected": frozenset(
        {"authenticated", "endpoint", "event", "timestamp_utc"}
    ),
    "proposal_selected": frozenset(
        {
            "ask", "bar_age_s", "book_id", "effective_floor", "event",
            "is_latest_bar", "lane", "live_breakeven", "offset_id", "pair",
            "payout", "proposal_id", "proposal_source", "side",
            "signal_close_utc", "signal_id", "timestamp_utc", "worker_id",
        }
    ),
    "buy_submitted": frozenset(
        {
            "bar_age_s", "book_id", "buy_send_utc", "buy_transport",
            "claim_to_buy_submitted_ms", "effective_floor", "event",
            "is_latest_bar", "lane", "offset_id", "pair", "proposal_id",
            "side", "signal_close_utc", "signal_id", "timestamp_utc",
        }
    ),
    "buy_confirmed": frozenset(
        {
            "bar_age_s", "book_id", "contract_id", "effective_floor", "event",
            "is_latest_bar", "lane", "offset_id", "pair", "side",
            "signal_close_utc", "signal_id", "timestamp_utc",
        }
    ),
    "contract_update": frozenset(
        {
            "contract_id", "contract_status", "event", "pair", "raw_hash",
            "side", "signal_id", "terminal", "timestamp_utc",
        }
    ),
    "contract_closed": frozenset(
        {
            "contract_id", "event", "pair", "profit", "raw_hash",
            "sell_price", "side", "signal_id", "terminal_status",
            "timestamp_utc",
        }
    ),
}

PACKET_FIELDS = frozenset(
    {"schema", "source", "account", "linkage", "contract", "proposal", "buy", "terminal", "retention"}
)
EVIDENCE_CLASSES = (
    "OUTCOME_ONLY_TICK_PROXY",
    "CONSTANT_PAYOUT_STRESS",
    "QUOTE_CONDITIONED_NO_FILL",
    "VENUE_SETTLED",
)
SECRET_KEY = re.compile(
    r"(?:^|_)(?:token|password|passwd|secret|credential|authorization|account_id|api_key|app_id|endpoint|url)(?:$|_)",
    re.IGNORECASE,
)
URL_VALUE = re.compile(r"(?:https?|wss?)://|(?:^|[?&])(?:token|app_id|authorize)=", re.IGNORECASE)
HEX16 = re.compile(r"[0-9a-f]{16}")


class DerivEconomicsError(evidence.EvidenceError):
    """The fixed Deriv economics contract failed closed."""


def _exact_fields(value: Any, fields: set[str] | frozenset[str], name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise DerivEconomicsError(f"{name} must be a JSON object")
    actual = set(value)
    if actual != set(fields):
        raise DerivEconomicsError(
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
        raise DerivEconomicsError(f"{name} must be a non-empty canonical string")
    return value


def _utc(value: Any, name: str) -> datetime:
    text = _text(value, name)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise DerivEconomicsError(f"{name} is not ISO-8601") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise DerivEconomicsError(f"{name} must be offset-aware")
    if parsed.utcoffset().total_seconds() != 0:
        raise DerivEconomicsError(f"{name} must be UTC")
    return parsed.astimezone(timezone.utc)


def _cent(value: Any, name: str, *, positive: bool = False, nonnegative: bool = False) -> str:
    if type(value) not in {str, int, float} or type(value) is bool:
        raise DerivEconomicsError(f"{name} must be an exact decimal source value")
    if isinstance(value, str) and ("e" in value.lower() or re.fullmatch(r"-?(?:0|[1-9][0-9]*)\.[0-9]{2}", value) is None):
        raise DerivEconomicsError(f"{name} must be canonical two-decimal text")
    try:
        decimal = Decimal(str(value))
    except InvalidOperation as exc:
        raise DerivEconomicsError(f"{name} is not decimal") from exc
    if not decimal.is_finite() or decimal != decimal.quantize(Decimal("0.01")):
        raise DerivEconomicsError(f"{name} is not exactly cent-aligned")
    if positive and decimal <= 0:
        raise DerivEconomicsError(f"{name} must be positive")
    if nonnegative and decimal < 0:
        raise DerivEconomicsError(f"{name} must be non-negative")
    return format(decimal, ".2f")


def _decimal(value: Any, name: str) -> Decimal:
    if not isinstance(value, str):
        raise DerivEconomicsError(f"{name} must be canonical two-decimal JSON text")
    canonical = _cent(value, name)
    return Decimal(canonical)


def _reject_secret_keys(value: Any, *, location: str = "source") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if SECRET_KEY.search(str(key)) and key != "endpoint":
                raise DerivEconomicsError(f"secret-bearing key at {location}.{key}")
            _reject_secret_keys(item, location=f"{location}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_secret_keys(item, location=f"{location}[{index}]")


def _reject_packet_secrets(value: Any, *, location: str = "packet") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if SECRET_KEY.search(str(key)):
                raise DerivEconomicsError(f"forbidden packet key at {location}.{key}")
            _reject_packet_secrets(item, location=f"{location}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_packet_secrets(item, location=f"{location}[{index}]")
    elif isinstance(value, str) and URL_VALUE.search(value):
        raise DerivEconomicsError(f"forbidden URL/query value at {location}")


def acquisition_payload() -> dict[str, Any]:
    """Return the immutable, outcome-independent acquisition decision."""

    return {
        "schema": ACQUISITION_KIND,
        "acquisition_id": "ROS-3A-USDJPY-UP-2026-07-06-v1",
        "source": {
            "date": SOURCE_DATE,
            "timezone": SOURCE_TZ,
            "date_field": "signals.signal_close_utc",
            "pair": SOURCE_PAIR,
            "side": SOURCE_SIDE,
            "database": DB_LOGICAL_PATH,
            "log": LOG_LOGICAL_PATH,
        },
        "selector": {
            "eligibility": "complete_linked_demo_hold_to_expiry_terminal_chain",
            "order": ["contracts.created_utc", "contract_id"],
            "forbidden_order_inputs": ["outcome", "profit", "payout", "model_value"],
            "cardinality": 1,
        },
        "runtime_contract": {
            "recorded_query_time_revision": RUNTIME_REVISION,
            "contract_type": "CALL",
            "basis": "stake",
            "currency": "USD",
            "duration": 15,
            "duration_unit": "m",
            "environment": "DEMO",
            "funding": "VIRTUAL",
            "real_money": False,
        },
        "reference_semantics": {
            "hold_to_expiry": True,
            "call": "exit > entry",
            "put": "exit < entry",
            "equality": "loses",
        },
        "source_row_keys": {key: sorted(value) for key, value in SOURCE_KEYS.items()},
        "packet": {"schema": PACKET_SCHEMA, "top_level_fields": sorted(PACKET_FIELDS)},
        "equations": {
            "quoted_win_profit": "proposal_payout - proposal_ask",
            "quoted_loss_profit": "-proposal_ask",
            "terminal_implied_cost": "terminal_sell_price - terminal_recorded_profit",
            "realized_profit": "terminal_sell_price - terminal_implied_cost",
            "breakeven_fraction": "proposal_ask / proposal_payout",
        },
        "evidence_classes": list(EVIDENCE_CLASSES),
        "claim_limits": {
            "actual_request_response_bytes": "not_retained",
            "runtime_revision_at_trade": "not_retained",
            "entry_exit_ticks": "not_retained",
            "strategy_edge": False,
            "activation": False,
        },
        "provenance": {
            "demo": "observed same-executor-epoch authenticated=true plus recorded query-time code meaning",
            "product": "recorded_query_time_runtime_contract",
            "terminal": "observed linked database and executor log",
        },
        "falsifiers": [
            "authority or bound artifact mismatch before source open",
            "missing duplicate conflicting or non-demo lifecycle stage",
            "context identifier or UTC clock mismatch",
            "unknown or secret-bearing source or packet field",
            "sub-cent rounded non-finite or inconsistent economics",
            "more than one packet or normalized descendant",
        ],
    }


def build_acquisition_spec(repo_root: Path) -> evidence.EvidenceEnvelope:
    artifacts = [evidence.ArtifactRef.capture(path, repo_root=repo_root) for path in BOUND_ARTIFACTS]
    return evidence.EvidenceEnvelope.create(
        kind=ACQUISITION_KIND,
        payload=acquisition_payload(),
        artifacts=artifacts,
    )


def seal_acquisition(repo_root: Path = REPO_ROOT) -> evidence.EvidenceEnvelope:
    envelope = build_acquisition_spec(repo_root)
    evidence.publish(envelope, repo_root=repo_root)
    return evidence.verify_object(envelope.object_id, repo_root=repo_root)


def _validate_acquisition_spec(
    spec_id: str, *, repo_root: Path
) -> evidence.EvidenceEnvelope:
    try:
        spec = evidence.verify_object(spec_id, repo_root=repo_root)
    except evidence.EvidenceError as exc:
        raise DerivEconomicsError(f"acquisition authority does not verify: {exc}") from exc
    if spec.kind != ACQUISITION_KIND or spec.payload != acquisition_payload():
        raise DerivEconomicsError("acquisition authority kind or payload differs")
    expected = tuple(
        evidence.ArtifactRef.capture(path, repo_root=repo_root) for path in BOUND_ARTIFACTS
    )
    if spec.artifacts != expected or spec.dependencies:
        raise DerivEconomicsError("acquisition authority artifact bindings differ")
    return spec


def _decode_log(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DerivEconomicsError(f"invalid JSONL row {number}") from exc
            if not isinstance(row, dict):
                raise DerivEconomicsError(f"JSONL row {number} is not an object")
            _reject_secret_keys(row, location=f"log[{number}]")
            event = row.get("event")
            if event in SOURCE_KEYS:
                _exact_fields(row, SOURCE_KEYS[event], f"{event} row")
                if "url" in row:
                    raise DerivEconomicsError("url source key is forbidden")
                events.append(row)
    return events


def _one(rows: list[dict[str, Any]], event: str, signal_id: str) -> dict[str, Any]:
    matches = [row for row in rows if row["event"] == event and row.get("signal_id") == signal_id]
    if len(matches) != 1:
        raise DerivEconomicsError(f"{event} cardinality differs for admitted chain")
    return matches[0]


def _eligible_db_rows(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    required = {
        "signal_id", "pair", "side", "signal_close_utc", "contract_id",
        "contract_status", "proposal_id", "buy_price", "stake", "sell_price",
        "profit", "payout", "terminal_utc", "created_utc", "raw_hash",
    }
    query = """
        SELECT s.signal_id, s.pair, s.side, s.signal_close_utc,
               c.contract_id, c.contract_status, c.proposal_id, c.buy_price,
               c.stake, c.sell_price, c.profit, c.payout, c.terminal_utc,
               c.created_utc, c.raw_hash
        FROM signals AS s JOIN contracts AS c ON c.signal_id = s.signal_id
    """
    rows: list[dict[str, Any]] = []
    for source in conn.execute(query):
        row = dict(source)
        if set(row) != required:
            raise DerivEconomicsError("database projection fields differ")
        if row["pair"] != SOURCE_PAIR or row["side"] != SOURCE_SIDE:
            continue
        close = _utc(row["signal_close_utc"], "signals.signal_close_utc")
        if close.astimezone(ZoneInfo(SOURCE_TZ)).date().isoformat() != SOURCE_DATE:
            continue
        if row["contract_status"] not in {"won", "lost"}:
            continue
        if any(row[key] is None for key in required):
            raise DerivEconomicsError("eligible database row has a missing projection")
        rows.append(row)
    return rows


def _linked_candidate(db: dict[str, Any], events: list[dict[str, Any]]) -> dict[str, Any]:
    signal_id = _text(db["signal_id"], "signal_id")
    proposal = _one(events, "proposal_selected", signal_id)
    submitted = _one(events, "buy_submitted", signal_id)
    confirmed = _one(events, "buy_confirmed", signal_id)
    updates = [
        row for row in events
        if row["event"] == "contract_update"
        and row.get("signal_id") == signal_id
        and row.get("terminal") is True
    ]
    closes = [
        row for row in events
        if row["event"] == "contract_closed" and row.get("signal_id") == signal_id
    ]
    if len(updates) != 1 or len(closes) != 1:
        raise DerivEconomicsError("terminal lifecycle cardinality differs")
    update, closed = updates[0], closes[0]

    for row in (proposal, submitted, confirmed, update, closed):
        if row.get("pair") != SOURCE_PAIR or row.get("side") != SOURCE_SIDE:
            raise DerivEconomicsError("lifecycle pair or side differs")
    if not (
        proposal["proposal_id"] == submitted["proposal_id"] == db["proposal_id"]
        and str(confirmed["contract_id"]) == str(update["contract_id"])
        == str(closed["contract_id"]) == str(db["contract_id"])
    ):
        raise DerivEconomicsError("lifecycle identifiers differ")
    if update["contract_status"] != db["contract_status"] or closed["terminal_status"] != db["contract_status"]:
        raise DerivEconomicsError("terminal statuses differ")
    if update["raw_hash"] != closed["raw_hash"] or update["raw_hash"] != db["raw_hash"]:
        raise DerivEconomicsError("terminal raw-hash markers differ")
    if HEX16.fullmatch(str(db["raw_hash"])) is None:
        raise DerivEconomicsError("terminal raw hash is not the retained 16-hex marker")

    clocks = [
        _utc(proposal["timestamp_utc"], "proposal timestamp"),
        _utc(submitted["timestamp_utc"], "submission-recorded timestamp"),
        _utc(confirmed["timestamp_utc"], "confirmation-recorded timestamp"),
        _utc(db["terminal_utc"], "database terminal timestamp"),
        _utc(update["timestamp_utc"], "terminal-update timestamp"),
        _utc(closed["timestamp_utc"], "contract-closed timestamp"),
    ]
    if clocks != sorted(clocks):
        raise DerivEconomicsError("lifecycle clocks are not ordered")

    proposal_clock = clocks[0]
    startups = [row for row in events if row["event"] == "executor_startup" and _utc(row["timestamp_utc"], "startup timestamp") <= proposal_clock]
    if not startups:
        raise DerivEconomicsError("proposal has no preceding executor startup")
    startup = max(startups, key=lambda row: _utc(row["timestamp_utc"], "startup timestamp"))
    startup_clock = _utc(startup["timestamp_utc"], "startup timestamp")
    authenticated = [
        row for row in events
        if row["event"] == "exec_client_connected"
        and startup_clock <= _utc(row["timestamp_utc"], "connection timestamp") <= proposal_clock
        and row["authenticated"] is True
    ]
    if not authenticated:
        raise DerivEconomicsError("proposal lacks same-epoch authenticated demo observation")

    ask = _cent(proposal["ask"], "proposal ask", positive=True)
    proposal_payout = _cent(proposal["payout"], "proposal payout", positive=True)
    buy_price = _cent(db["buy_price"], "recorded buy-price projection", positive=True)
    stake = _cent(db["stake"], "stake", positive=True)
    sell = _cent(db["sell_price"], "terminal sell price", nonnegative=True)
    profit = _cent(db["profit"], "terminal recorded profit")
    terminal_payout = _cent(db["payout"], "terminal payout", positive=True)
    if (
        _cent(closed["sell_price"], "logged terminal sell price", nonnegative=True)
        != sell
        or _cent(closed["profit"], "logged terminal profit") != profit
    ):
        raise DerivEconomicsError("database and logged terminal economics differ")

    packet = {
        "schema": PACKET_SCHEMA,
        "source": {
            "acquisition_spec_id": None,
            "exporter_sha256": None,
            "database": DB_LOGICAL_PATH,
            "log": LOG_LOGICAL_PATH,
            "source_date": SOURCE_DATE,
            "selector": "earliest_complete_by_contract_created_utc_then_contract_id",
            "selected_source_clocks": {
                "proposal_utc": proposal["timestamp_utc"],
                "submission_recorded_utc": submitted["timestamp_utc"],
                "confirmation_recorded_utc": confirmed["timestamp_utc"],
                "database_terminal_utc": db["terminal_utc"],
                "terminal_update_utc": update["timestamp_utc"],
                "contract_closed_utc": closed["timestamp_utc"],
            },
            "recorded_query_time_runtime_revision": RUNTIME_REVISION,
            "runtime_revision_at_trade": "not_retained",
        },
        "account": {
            "environment": "DEMO",
            "funding": "VIRTUAL",
            "real_money": False,
            "provenance": {
                "same_executor_epoch_authenticated": True,
                "meaning": "recorded_query_time_code_marks_authenticated_true_only_for_demo_connections",
            },
        },
        "linkage": {
            "signal_id": signal_id,
            "proposal_id": str(db["proposal_id"]),
            "contract_id": str(db["contract_id"]),
        },
        "contract": {
            "pair": SOURCE_PAIR,
            "side": SOURCE_SIDE,
            "recorded_query_time_runtime_contract": {
                "contract_type": "CALL", "basis": "stake", "currency": "USD",
                "duration": 15, "duration_unit": "m",
            },
            "observed_behavior": {"terminal_status": db["contract_status"], "early_sale": False},
            "reference_semantics": {"hold_to_expiry": True, "equality": "loses"},
        },
        "proposal": {
            "observed_utc": proposal["timestamp_utc"],
            "proposal_id": str(proposal["proposal_id"]),
            "proposal_source": _text(proposal["proposal_source"], "proposal source"),
            "ask": ask,
            "payout": proposal_payout,
        },
        "buy": {
            "submission_recorded_utc": submitted["timestamp_utc"],
            "confirmation_recorded_utc": confirmed["timestamp_utc"],
            "proposal_id": str(db["proposal_id"]),
            "contract_id": str(db["contract_id"]),
            "recorded_buy_price_projection": buy_price,
            "stake": stake,
        },
        "terminal": {
            "observed_utc": closed["timestamp_utc"],
            "contract_id": str(db["contract_id"]),
            "status": db["contract_status"],
            "sell_price": sell,
            "recorded_profit": profit,
            "payout": terminal_payout,
            "terminal_utc": db["terminal_utc"],
            "raw_hash16": str(db["raw_hash"]),
        },
        "retention": {
            "request_payload_retained": False,
            "raw_response_retained": False,
            "buy_response_values_retained": False,
            "buy_response_hash_retained": False,
            "runtime_revision_at_trade_retained": False,
            "entry_tick_retained": False,
            "exit_tick_retained": False,
        },
    }
    _exact_fields(packet, PACKET_FIELDS, "packet")
    _reject_packet_secrets(packet)
    return {"packet": packet, "order": (_utc(db["created_utc"], "contract created timestamp"), str(db["contract_id"]))}


def _atomic_packet(output_dir: Path, packet: dict[str, Any]) -> tuple[Path, bytes, str]:
    raw = evidence.canonical_json_bytes(packet)
    digest = hashlib.sha256(raw).hexdigest()
    output_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    if output_dir.is_symlink() or not output_dir.is_dir():
        raise DerivEconomicsError("packet output directory is not a real directory")
    existing = sorted(output_dir.glob("*.json"))
    destination = output_dir / f"{digest}.json"
    if existing:
        if existing != [destination] or destination.read_bytes() != raw:
            raise DerivEconomicsError("a second or conflicting packet already exists")
        if stat.S_IMODE(destination.stat().st_mode) != 0o600:
            raise DerivEconomicsError("existing packet mode differs")
        return destination, raw, digest
    fd, temporary = tempfile.mkstemp(prefix=f".{digest}.", dir=output_dir)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wb", closefd=True) as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        fd = -1
        os.link(temporary, destination)
        os.unlink(temporary)
        temporary = ""
        directory_fd = os.open(output_dir, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if fd >= 0:
            os.close(fd)
        if temporary:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
    return destination, raw, digest


def export_packet(
    *,
    spec_id: str,
    authority_root: Path,
    db_path: Path,
    log_path: Path,
    output_dir: Path,
    enforce_remote_paths: bool = False,
) -> dict[str, Any]:
    """Supply one packet after authority validation; test seams pass exact temp paths."""

    spec = _validate_acquisition_spec(spec_id, repo_root=authority_root)
    expected_exporter = next(item for item in spec.artifacts if item.path == EXPORTER_PATH)
    expected_db = REMOTE_ROOT / DB_LOGICAL_PATH
    expected_log = REMOTE_ROOT / LOG_LOGICAL_PATH
    if enforce_remote_paths and (
        db_path != expected_db
        or log_path != expected_log
        or output_dir != REMOTE_ROOT / "deriv_data/venue_evidence"
    ):
        raise DerivEconomicsError("remote source paths differ from the sealed contract")

    uri = f"file:{db_path}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, isolation_level=None)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA query_only=ON")
        if conn.execute("PRAGMA query_only").fetchone()[0] != 1:
            raise DerivEconomicsError("SQLite query_only did not engage")
        conn.execute("BEGIN")
        db_rows = _eligible_db_rows(conn)
        events = _decode_log(log_path)
        candidates: list[dict[str, Any]] = []
        errors: list[str] = []
        for row in db_rows:
            try:
                candidates.append(_linked_candidate(row, events))
            except DerivEconomicsError as exc:
                errors.append(str(exc))
        conn.execute("COMMIT")
    except Exception:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()
    if not candidates:
        raise DerivEconomicsError(f"no complete eligible retained chain: rejected={len(errors)}")
    selected = min(candidates, key=lambda item: item["order"])["packet"]
    selected["source"]["acquisition_spec_id"] = spec.object_id
    selected["source"]["exporter_sha256"] = expected_exporter.sha256
    _reject_packet_secrets(selected)
    destination, raw, digest = _atomic_packet(output_dir, selected)
    clocks = selected["source"]["selected_source_clocks"]
    descriptor_root = REMOTE_ROOT if enforce_remote_paths else authority_root
    return {
        "relative_path": destination.relative_to(descriptor_root).as_posix(),
        "bytes": len(raw),
        "sha256": digest,
        "fixed_contract": {
            "source_date": SOURCE_DATE, "pair": SOURCE_PAIR, "side": SOURCE_SIDE,
            "contract_type": "CALL", "duration": 15, "duration_unit": "m",
            "environment": "DEMO", "real_money": False,
        },
        "source_clocks": clocks,
    }


def settle_direction(contract_type: str, entry: Decimal, exit_: Decimal) -> bool:
    if contract_type == "CALL":
        return exit_ > entry
    if contract_type == "PUT":
        return exit_ < entry
    raise DerivEconomicsError(f"unsupported contract type {contract_type!r}")


def evidence_class_result(
    evidence_class: str, *, realized_profit: str | None
) -> dict[str, Any]:
    if evidence_class not in EVIDENCE_CLASSES:
        raise DerivEconomicsError(f"unsupported evidence class {evidence_class!r}")
    if evidence_class != "VENUE_SETTLED" and realized_profit is not None:
        raise DerivEconomicsError(
            f"{evidence_class} cannot carry realized venue economics"
        )
    if evidence_class == "VENUE_SETTLED" and realized_profit is None:
        raise DerivEconomicsError("VENUE_SETTLED requires reconciled realized profit")
    return {"evidence_class": evidence_class, "realized_profit": realized_profit}


def _normalize_packet(
    packet: Any,
    *,
    source_ref: evidence.ArtifactRef,
    spec: evidence.EvidenceEnvelope,
    descriptor: dict[str, Any],
) -> dict[str, Any]:
    packet = _exact_fields(packet, PACKET_FIELDS, "packet")
    if packet["schema"] != PACKET_SCHEMA:
        raise DerivEconomicsError("packet schema differs")
    _reject_packet_secrets(packet)
    source = _exact_fields(
        packet["source"],
        {"acquisition_spec_id", "exporter_sha256", "database", "log", "source_date", "selector", "selected_source_clocks", "recorded_query_time_runtime_revision", "runtime_revision_at_trade"},
        "packet source",
    )
    exporter = next(item for item in spec.artifacts if item.path == EXPORTER_PATH)
    if (
        source["acquisition_spec_id"] != spec.object_id
        or source["exporter_sha256"] != exporter.sha256
        or source["database"] != DB_LOGICAL_PATH
        or source["log"] != LOG_LOGICAL_PATH
        or source["source_date"] != SOURCE_DATE
        or source["selector"]
        != "earliest_complete_by_contract_created_utc_then_contract_id"
        or source["recorded_query_time_runtime_revision"] != RUNTIME_REVISION
        or source["runtime_revision_at_trade"] != "not_retained"
    ):
        raise DerivEconomicsError("packet source provenance differs")
    clocks = _exact_fields(
        source["selected_source_clocks"],
        {
            "proposal_utc", "submission_recorded_utc", "confirmation_recorded_utc",
            "database_terminal_utc", "terminal_update_utc", "contract_closed_utc",
        },
        "selected source clocks",
    )
    ordered_clocks = [
        _utc(clocks[key], key)
        for key in (
            "proposal_utc", "submission_recorded_utc", "confirmation_recorded_utc",
            "database_terminal_utc", "terminal_update_utc", "contract_closed_utc",
        )
    ]
    if ordered_clocks != sorted(ordered_clocks):
        raise DerivEconomicsError("packet source clocks are not ordered")
    expected_fixed = {
        "source_date": SOURCE_DATE, "pair": SOURCE_PAIR, "side": SOURCE_SIDE,
        "contract_type": "CALL", "duration": 15, "duration_unit": "m",
        "environment": "DEMO", "real_money": False,
    }
    if descriptor["fixed_contract"] != expected_fixed or descriptor["source_clocks"] != clocks:
        raise DerivEconomicsError("non-outcome descriptor differs from packet")
    account = _exact_fields(packet["account"], {"environment", "funding", "real_money", "provenance"}, "account")
    if account != {"environment": "DEMO", "funding": "VIRTUAL", "real_money": False, "provenance": {"same_executor_epoch_authenticated": True, "meaning": "recorded_query_time_code_marks_authenticated_true_only_for_demo_connections"}}:
        raise DerivEconomicsError("packet is not the admitted demo provenance")
    linkage = _exact_fields(packet["linkage"], {"signal_id", "proposal_id", "contract_id"}, "linkage")
    for key in linkage:
        _text(linkage[key], f"linkage.{key}")
    contract = _exact_fields(packet["contract"], {"pair", "side", "recorded_query_time_runtime_contract", "observed_behavior", "reference_semantics"}, "contract")
    runtime = contract["recorded_query_time_runtime_contract"]
    if contract["pair"] != SOURCE_PAIR or contract["side"] != SOURCE_SIDE or runtime != {"contract_type": "CALL", "basis": "stake", "currency": "USD", "duration": 15, "duration_unit": "m"}:
        raise DerivEconomicsError("packet product contract differs")
    behavior = contract["observed_behavior"]
    if not isinstance(behavior, dict) or behavior.get("terminal_status") not in {"won", "lost"} or behavior.get("early_sale") is not False or set(behavior) != {"terminal_status", "early_sale"}:
        raise DerivEconomicsError("packet terminal behavior differs")
    if contract["reference_semantics"] != {"hold_to_expiry": True, "equality": "loses"}:
        raise DerivEconomicsError("packet reference semantics differ")

    proposal = _exact_fields(packet["proposal"], {"observed_utc", "proposal_id", "proposal_source", "ask", "payout"}, "proposal")
    buy = _exact_fields(packet["buy"], {"submission_recorded_utc", "confirmation_recorded_utc", "proposal_id", "contract_id", "recorded_buy_price_projection", "stake"}, "buy")
    terminal = _exact_fields(packet["terminal"], {"observed_utc", "contract_id", "status", "sell_price", "recorded_profit", "payout", "terminal_utc", "raw_hash16"}, "terminal")
    if proposal["proposal_id"] != linkage["proposal_id"] or buy["proposal_id"] != linkage["proposal_id"] or buy["contract_id"] != linkage["contract_id"] or terminal["contract_id"] != linkage["contract_id"]:
        raise DerivEconomicsError("packet linkage differs")
    if terminal["status"] != behavior["terminal_status"] or HEX16.fullmatch(_text(terminal["raw_hash16"], "raw_hash16")) is None:
        raise DerivEconomicsError("packet terminal status or marker differs")
    ask = _decimal(proposal["ask"], "proposal ask")
    proposal_payout = _decimal(proposal["payout"], "proposal payout")
    buy_price = _decimal(buy["recorded_buy_price_projection"], "recorded buy-price projection")
    stake = _decimal(buy["stake"], "stake")
    sell = _decimal(terminal["sell_price"], "terminal sell price")
    recorded_profit = _decimal(terminal["recorded_profit"], "terminal recorded profit")
    terminal_payout = _decimal(terminal["payout"], "terminal payout")
    if ask <= 0 or proposal_payout <= 0 or buy_price <= 0 or stake <= 0 or sell < 0 or terminal_payout <= 0:
        raise DerivEconomicsError("packet money signs differ")
    implied_cost = sell - recorded_profit
    realized = sell - implied_cost
    if not (implied_cost == buy_price == stake == ask and realized == recorded_profit):
        raise DerivEconomicsError("terminal economics do not reconcile")
    if terminal["status"] == "won":
        if recorded_profit <= 0 or sell != terminal_payout:
            raise DerivEconomicsError("winning terminal economics differ")
    elif sell != Decimal("0.00") or recorded_profit != -implied_cost:
        raise DerivEconomicsError("losing terminal economics differ")
    retention = packet["retention"]
    expected_retention = {
        "request_payload_retained": False, "raw_response_retained": False,
        "buy_response_values_retained": False, "buy_response_hash_retained": False,
        "runtime_revision_at_trade_retained": False, "entry_tick_retained": False,
        "exit_tick_retained": False,
    }
    if retention != expected_retention:
        raise DerivEconomicsError("packet retention boundary differs")
    classification = evidence_class_result(
        "VENUE_SETTLED", realized_profit=format(realized, ".2f")
    )
    return {
        "schema": NORMALIZED_KIND,
        "evidence_class": classification["evidence_class"],
        "source": {"artifact": source_ref.as_dict(), "acquisition_spec_id": spec.object_id},
        "observation": packet,
        "derivation": {
            "currency": "USD",
            "quoted_win_profit": format(proposal_payout - ask, ".2f"),
            "quoted_loss_profit": format(-ask, ".2f"),
            "terminal_implied_cost": format(implied_cost, ".2f"),
            "realized_profit": classification["realized_profit"],
            "breakeven_fraction": {"numerator": format(ask, ".2f"), "denominator": format(proposal_payout, ".2f")},
            "proposal_and_terminal_payout_fields_preserved_separately": True,
            "entry_exit_direction": "unverified_not_retained",
        },
        "claim_limits": {
            "request_response_bytes": "not_retained",
            "runtime_revision_at_trade": "not_retained",
            "entry_exit_ticks": "not_retained",
            "real_account_parity": "not_claimed",
            "strategy_edge": "not_claimed",
        },
        "activation": False,
    }


def _domain_objects(repo_root: Path, spec_id: str) -> dict[str, list[evidence.EvidenceEnvelope]]:
    all_domain = {SEALED_KIND: [], ACCESS_KIND: [], NORMALIZED_KIND: []}
    for envelope in evidence.inspect_store_metadata(repo_root=repo_root):
        if envelope.kind not in all_domain:
            continue
        payload = envelope.payload
        if envelope.artifacts or len(envelope.dependencies) != 1:
            raise DerivEconomicsError(f"{envelope.kind} has invalid envelope bindings")
        if envelope.kind == SEALED_KIND:
            _exact_fields(
                payload,
                {"schema", "acquisition_spec_id", "source_artifact", "descriptor"},
                "sealed-source payload",
            )
            domain_spec_id = payload["acquisition_spec_id"]
        elif envelope.kind == ACCESS_KIND:
            _exact_fields(
                payload,
                {
                    "schema", "acquisition_spec_id", "sealed_source_id",
                    "source_artifact", "access_state",
                },
                "source-access payload",
            )
            domain_spec_id = payload["acquisition_spec_id"]
        else:
            _exact_fields(
                payload,
                {
                    "schema", "evidence_class", "source", "observation",
                    "derivation", "claim_limits", "activation",
                },
                "normalized-receipt payload",
            )
            source = _exact_fields(
                payload["source"], {"artifact", "acquisition_spec_id"},
                "normalized source",
            )
            domain_spec_id = source["acquisition_spec_id"]
        if payload["schema"] != envelope.kind:
            raise DerivEconomicsError(f"{envelope.kind} payload schema differs")
        _text(domain_spec_id, f"{envelope.kind} acquisition_spec_id")
        all_domain[envelope.kind].append(envelope)

    sealed = [
        item
        for item in all_domain[SEALED_KIND]
        if item.payload["acquisition_spec_id"] == spec_id
        or item.dependencies == (spec_id,)
    ]
    sealed_ids = {item.object_id for item in sealed}
    access = [
        item
        for item in all_domain[ACCESS_KIND]
        if item.payload["acquisition_spec_id"] == spec_id
        or item.payload["sealed_source_id"] in sealed_ids
        or any(dependency in sealed_ids for dependency in item.dependencies)
    ]
    access_ids = {item.object_id for item in access}
    normalized = [
        item
        for item in all_domain[NORMALIZED_KIND]
        if item.payload["source"]["acquisition_spec_id"] == spec_id
        or any(dependency in access_ids for dependency in item.dependencies)
    ]
    return {SEALED_KIND: sealed, ACCESS_KIND: access, NORMALIZED_KIND: normalized}


def reduce_packet(
    *, repo_root: Path, spec_id: str, source_ref: evidence.ArtifactRef, descriptor: dict[str, Any]
) -> evidence.EvidenceEnvelope:
    spec = _validate_acquisition_spec(spec_id, repo_root=repo_root)
    _exact_fields(
        descriptor,
        {"relative_path", "bytes", "sha256", "fixed_contract", "source_clocks"},
        "descriptor",
    )
    if descriptor.get("relative_path") != source_ref.path or descriptor.get("bytes") != source_ref.bytes or descriptor.get("sha256") != source_ref.sha256:
        raise DerivEconomicsError("source descriptor differs from declared ArtifactRef")
    source_ref.isolated_identity(repo_root=repo_root)
    sealed = evidence.EvidenceEnvelope.create(
        kind=SEALED_KIND,
        payload={
            "schema": SEALED_KIND,
            "acquisition_spec_id": spec.object_id,
            "source_artifact": source_ref.as_dict(),
            "descriptor": descriptor,
        },
        dependencies=[spec.object_id],
    )
    receipt = evidence.EvidenceEnvelope.create(
        kind=ACCESS_KIND,
        payload={
            "schema": ACCESS_KIND,
            "acquisition_spec_id": spec.object_id,
            "sealed_source_id": sealed.object_id,
            "source_artifact": source_ref.as_dict(),
            "access_state": ACCESS_STATE,
        },
        dependencies=[sealed.object_id],
    )
    groups = _domain_objects(repo_root, spec.object_id)
    if any(len(items) > 1 for items in groups.values()):
        raise DerivEconomicsError("hidden or conflicting economics descendants")
    if groups[SEALED_KIND] and groups[SEALED_KIND][0].as_dict() != sealed.as_dict():
        raise DerivEconomicsError("conflicting sealed source exists")
    if groups[ACCESS_KIND] and groups[ACCESS_KIND][0].as_dict() != receipt.as_dict():
        raise DerivEconomicsError("conflicting access receipt exists")
    if groups[NORMALIZED_KIND]:
        normalized = groups[NORMALIZED_KIND][0]
        if not groups[ACCESS_KIND] or normalized.dependencies != (receipt.object_id,):
            raise DerivEconomicsError("normalized receipt ancestry differs")
        if normalized.payload.get("source", {}).get("artifact") != source_ref.as_dict():
            raise DerivEconomicsError("normalized receipt source differs")
        expected_payload = _normalize_packet(
            normalized.payload["observation"],
            source_ref=source_ref,
            spec=spec,
            descriptor=descriptor,
        )
        if normalized.payload != expected_payload:
            raise DerivEconomicsError("normalized receipt is not reproducible")
        evidence.verify_object(normalized.object_id, repo_root=repo_root)
        return normalized
    if groups[ACCESS_KIND]:
        raise DerivEconomicsError("BLOCKED: source access receipt exists without normalized receipt")
    evidence.publish(sealed, repo_root=repo_root)
    evidence.publish(receipt, repo_root=repo_root, require_new=True)
    raw = source_ref.read_verified(repo_root=repo_root)
    try:
        packet = evidence.decode_canonical_json(raw)
    except evidence.EvidenceError as exc:
        raise DerivEconomicsError(f"source packet bytes are invalid: {exc}") from exc
    normalized_payload = _normalize_packet(
        packet, source_ref=source_ref, spec=spec, descriptor=descriptor
    )
    normalized = evidence.EvidenceEnvelope.create(
        kind=NORMALIZED_KIND,
        payload=normalized_payload,
        dependencies=[receipt.object_id],
    )
    evidence.publish(normalized, repo_root=repo_root)
    return normalized


def _descriptor(value: str) -> dict[str, Any]:
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError as exc:
        raise DerivEconomicsError("descriptor is not JSON") from exc
    return _exact_fields(decoded, {"relative_path", "bytes", "sha256", "fixed_contract", "source_clocks"}, "descriptor")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("seal-acquisition")
    export = commands.add_parser("export-one")
    export.add_argument("--acquisition-spec-id", required=True)
    reduce = commands.add_parser("reduce")
    reduce.add_argument("--acquisition-spec-id", required=True)
    reduce.add_argument("--source-path", required=True)
    reduce.add_argument("--source-bytes", required=True, type=int)
    reduce.add_argument("--source-sha256", required=True)
    reduce.add_argument("--descriptor-json", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = args.repo_root.resolve()
    try:
        if args.command == "seal-acquisition":
            spec = seal_acquisition(root)
            print(json.dumps({"acquisition_spec_id": spec.object_id}, sort_keys=True))
        elif args.command == "export-one":
            descriptor = export_packet(
                spec_id=args.acquisition_spec_id,
                authority_root=root,
                db_path=REMOTE_ROOT / DB_LOGICAL_PATH,
                log_path=REMOTE_ROOT / LOG_LOGICAL_PATH,
                output_dir=REMOTE_ROOT / "deriv_data/venue_evidence",
                enforce_remote_paths=True,
            )
            print(json.dumps(descriptor, sort_keys=True, separators=(",", ":")))
        else:
            source_ref = evidence.ArtifactRef(
                path=args.source_path, bytes=args.source_bytes, sha256=args.source_sha256
            )
            normalized = reduce_packet(
                repo_root=root,
                spec_id=args.acquisition_spec_id,
                source_ref=source_ref,
                descriptor=_descriptor(args.descriptor_json),
            )
            print(json.dumps({"normalized_receipt_id": normalized.object_id}, sort_keys=True))
    except (DerivEconomicsError, evidence.EvidenceError, OSError, sqlite3.Error) as exc:
        print(f"Deriv economics ledger failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
