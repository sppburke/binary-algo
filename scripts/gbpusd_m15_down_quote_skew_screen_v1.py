#!/usr/bin/env python3
"""Record issue #21's fixed-source feasibility falsifier without reading outcomes."""

from __future__ import annotations

import argparse
import ast
import gzip
import hashlib
import json
import os
import re
import stat
import struct
import tempfile
from datetime import date, datetime, timedelta, timezone
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any, Mapping, Sequence
from zoneinfo import ZoneInfo

import evidence_store as evidence


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_ID = "gbpusd.m15.down.deriv-quote-skew.v1"
ARM_ID = "down_quote_follow_screen.v1"
SCRIPT_PATH = "scripts/gbpusd_m15_down_quote_skew_screen_v1.py"
SNAPSHOT_PATH = "results/json/gbpusd_m15_down_quote_skew_source_snapshot_v1.json"
RESULT_PATH = "results/json/gbpusd_m15_down_quote_skew_screen_v1_result.json"
SPEC_PATH = "results/json/gbpusd_m15_down_quote_skew_screen_v1_campaign_spec.json"
ARTIFACT_PATHS = (
    SNAPSHOT_PATH,
    "ENVIRONMENT_libs.txt",
    "scripts/deriv_quote_workers.py",
    "scripts/deriv_market_stream.py",
    "scripts/m15_book_refresh_stats.py",
    "scripts/min1_production.py",
    "scripts/deriv_runtime_core.py",
    "docs/DERIV_ECONOMICS_LEDGER.md",
)
QUOTE_ROOT_ENV = "GBPUSD_QUOTE_SKEW_QUOTE_ROOT"
TICK_ROOT_ENV = "GBPUSD_QUOTE_SKEW_TICK_ROOT"
INPUT_SCHEMA = "research.gbpusd_m15_down_quote_skew.input/v1"
SNAPSHOT_SCHEMA = "gbpusd-m15-down-quote-skew-source-snapshot/v1"
RESULT_SCHEMA = "gbpusd-m15-down-quote-skew-screen-result/v1"
RECEIPT_SCHEMA = "research-campaign-adapter-receipt/v1"
DECISION_SCHEMA = "research-campaign-family-decision/v1"
EXPECTED_ENVIRONMENT = {
    "numpy": "2.4.6",
    "pandas": "3.0.3",
    "pyarrow": "24.0.0",
    "scipy": "1.17.1",
}
MIN_DATES = 10
FIRST_NY_DATE = date(2026, 7, 2)
LAST_NY_DATE = date(2026, 7, 20)
NY = ZoneInfo("America/New_York")
UTC = timezone.utc
TICK_FIRST_EPOCH = int(datetime(2026, 7, 2, 12, 0, tzinfo=UTC).timestamp())
TICK_LAST_EPOCH = int(datetime(2026, 7, 20, 20, 50, 2, tzinfo=UTC).timestamp())
PAIRS = ("USDJPY", "USDCAD", "AUDUSD", "NZDUSD", "USDCHF", "GBPUSD")
CONTRACTS = ("CALL", "PUT")
LANES = tuple((pair, contract) for pair in PAIRS for contract in CONTRACTS)
QUOTE_FILE_RE = re.compile(r"^.+\.jsonl(?:\.\d+)?(?:\.gz)?$")
TICK_FILE_RE = re.compile(r"^(\d{6})_(\d+)_(\d+)\.parquet$")
TICK_TREE_DOMAIN = b"gbpusd-m15-down-tick-files/v1\0"
QUOTE_FIELDS = {
    "event",
    "timestamp_utc",
    "pair",
    "side",
    "contract_type",
    "ask",
    "payout",
    "live_breakeven",
    "invalid_reason",
    "proposal_id_present",
    "quote_age_ms",
    "source",
}
CLAIM_LIMIT = (
    "Fixed-window source-feasibility result only: no tick-price interpretation, outcome "
    "computation, quote-skew efficacy, fill, realized P&L, strategy candidate, certification, "
    "activation, deployment, or real-money claim."
)


class ScreenError(ValueError):
    """The campaign or source-seal contract is malformed."""


class NotComputable(RuntimeError):
    """A post-attempt source or environment gate failed."""


class SemanticJsonError(ValueError):
    """A syntactically complete JSON value violates strict source semantics."""


def _exact(value: Any, fields: set[str], name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        actual = set(value) if isinstance(value, Mapping) else set()
        raise ScreenError(
            f"{name} fields differ: missing={sorted(fields - actual)} "
            f"unknown={sorted(actual - fields)}"
        )
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _canonical_read(path: Path) -> Any:
    return evidence.decode_canonical_json(path.read_bytes())


def _canonical_write(path: Path, value: Any) -> None:
    path.write_bytes(evidence.canonical_json_bytes(value))


def _parse_utc(value: Any, name: str) -> tuple[datetime, int]:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ScreenError(f"{name} must be a UTC timestamp string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ScreenError(f"{name} is not ISO-8601") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise ScreenError(f"{name} must carry UTC offset zero")
    parsed = parsed.astimezone(UTC)
    delta = parsed - datetime(1970, 1, 1, tzinfo=UTC)
    micros = (delta.days * 86_400 + delta.seconds) * 1_000_000 + delta.microseconds
    return parsed, micros


def _strict_json(line: str) -> Any:
    def object_hook(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in pairs:
            if key in out:
                raise SemanticJsonError(f"duplicate JSON key {key!r}")
            out[key] = value
        return out

    def reject_constant(value: str) -> None:
        raise SemanticJsonError(f"non-finite JSON constant {value!r}")

    return json.loads(
        line,
        object_pairs_hook=object_hook,
        parse_constant=reject_constant,
    )


def _list_sha256(values: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    digest.update(b"[")
    for index, value in enumerate(values):
        if index:
            digest.update(b",")
        digest.update(evidence.canonical_json_bytes(dict(value)))
    digest.update(b"]")
    return digest.hexdigest()


def _tree_sha256(domain: bytes, entries: Sequence[tuple[str, int, bytes]]) -> str:
    digest = hashlib.sha256(domain)
    for name, size, raw_digest in entries:
        encoded = name.encode("utf-8")
        digest.update(struct.pack(">I", len(encoded)))
        digest.update(encoded)
        digest.update(struct.pack(">Q", size))
        digest.update(raw_digest)
    return digest.hexdigest()


def _regular_stat(path: Path, name: str) -> os.stat_result:
    try:
        info = path.stat(follow_symlinks=False)
    except OSError as exc:
        raise NotComputable(
            f"{name}_stat_failed:{path.name}:{exc.__class__.__name__}"
        ) from exc
    if path.is_symlink() or not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise NotComputable(f"{name}_type_differs:{path.name}")
    return info


def _stable_read(path: Path, name: str) -> bytes:
    before = _regular_stat(path, name)
    try:
        with path.open("rb") as handle:
            opened = os.fstat(handle.fileno())
            if (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino):
                raise NotComputable(f"{name}_binding_changed:{path.name}")
            raw = handle.read()
            after = os.fstat(handle.fileno())
    except NotComputable:
        raise
    except OSError as exc:
        raise NotComputable(
            f"{name}_read_failed:{path.name}:{exc.__class__.__name__}"
        ) from exc

    def identity(item: os.stat_result) -> tuple[int, int, int, int, int, int]:
        return (
            item.st_dev,
            item.st_ino,
            item.st_size,
            item.st_mtime_ns,
            item.st_ctime_ns,
            item.st_nlink,
        )

    if identity(opened) != identity(after) or len(raw) != opened.st_size:
        raise NotComputable(f"{name}_changed_during_read:{path.name}")
    return raw


def _prefix_read(path: Path) -> bytes:
    """Read a coherent plain-log prefix; later out-of-window appends are irrelevant."""

    before = _regular_stat(path, "quote_file")
    try:
        with path.open("rb") as handle:
            opened = os.fstat(handle.fileno())
            if (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino):
                raise NotComputable(f"quote_file_binding_changed:{path.name}")
            return handle.read()
    except NotComputable:
        raise
    except OSError as exc:
        raise NotComputable(
            f"quote_file_read_failed:{path.name}:{exc.__class__.__name__}"
        ) from exc


def _real_directory(path: Path, name: str) -> Path:
    if path.is_symlink():
        raise ScreenError(f"{name} must not be a symlink")
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise ScreenError(f"{name} does not exist") from exc
    if not resolved.is_dir():
        raise ScreenError(f"{name} must be a directory")
    return resolved


def _source_root(variable: str, name: str) -> Path:
    value = os.environ.get(variable)
    if not value:
        raise ScreenError(f"{variable} is missing")
    return _real_directory(Path(value), name)


def _quote_rows(path: Path) -> list[tuple[int, str, Mapping[str, Any]]]:
    raw = (
        _stable_read(path, "quote_file")
        if path.name.endswith(".gz")
        else _prefix_read(path)
    )
    if path.name.endswith(".gz"):
        try:
            raw = gzip.decompress(raw)
        except (OSError, EOFError) as exc:
            raise NotComputable(f"quote_gzip_invalid:{path.name}") from exc
    try:
        lines = raw.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise NotComputable(f"quote_utf8_invalid:{path.name}") from exc
    selected: list[tuple[int, str, Mapping[str, Any]]] = []
    for index, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            value = _strict_json(line)
        except json.JSONDecodeError as exc:
            if index == len(lines):
                continue
            raise NotComputable(f"quote_json_invalid:{path.name}:{index}") from exc
        except SemanticJsonError as exc:
            raise NotComputable(
                f"quote_json_semantic_invalid:{path.name}:{index}"
            ) from exc
        if not isinstance(value, Mapping) or value.get("event") != "quote_snapshot":
            continue
        try:
            timestamp, micros = _parse_utc(
                value.get("timestamp_utc"), f"{path.name}:{index}.timestamp_utc"
            )
        except ScreenError as exc:
            raise NotComputable(f"quote_timestamp_invalid:{path.name}:{index}") from exc
        ny_date = timestamp.astimezone(NY).date()
        if not FIRST_NY_DATE <= ny_date <= LAST_NY_DATE:
            continue
        if set(value) != QUOTE_FIELDS:
            raise NotComputable(f"quote_schema_invalid:{path.name}:{index}")
        try:
            evidence.canonical_json_bytes(dict(value))
        except evidence.EvidenceError as exc:
            raise NotComputable(
                f"quote_json_semantic_invalid:{path.name}:{index}"
            ) from exc
        selected.append((micros, ny_date.isoformat(), dict(value)))
    return selected


def _complete_block_count(rows: Sequence[tuple[int, str, Mapping[str, Any]]]) -> int:
    count = 0
    index = 0
    while index < len(rows):
        row = rows[index][2]
        if (row.get("pair"), row.get("contract_type")) != LANES[0]:
            index += 1
            continue
        candidate = rows[index : index + len(LANES)]
        observed = tuple(
            (item[2].get("pair"), item[2].get("contract_type")) for item in candidate
        )
        if len(candidate) == len(LANES) and observed == LANES:
            count += 1
            index += len(LANES)
        else:
            index += 1
    return count


def _quote_snapshot(root: Path) -> dict[str, Any]:
    rows: list[tuple[int, str, Mapping[str, Any]]] = []
    relevant_files: set[str] = set()
    try:
        entries = sorted(os.scandir(root), key=lambda item: item.name)
    except OSError as exc:
        raise NotComputable(f"quote_root_scan_failed:{exc.__class__.__name__}") from exc
    for entry in entries:
        if QUOTE_FILE_RE.fullmatch(entry.name) is None:
            continue
        selected = _quote_rows(Path(entry.path))
        if selected:
            relevant_files.add(entry.name)
            rows.extend(selected)
    rows.sort(key=lambda item: item[0])
    timestamps = [item[0] for item in rows]
    if not rows:
        raise NotComputable("quote_source_window_empty")
    if len(timestamps) != len(set(timestamps)):
        raise NotComputable("duplicate_quote_timestamp")
    values = [item[2] for item in rows]
    return {
        "logical_source": "public_deriv_quote_audit_jsonl",
        "selection": "embedded_timestamp_ny_dates_2026-07-02_through_2026-07-20",
        "relevant_file_count": len(relevant_files),
        "selected_row_count": len(rows),
        "selected_ny_dates": sorted({item[1] for item in rows}),
        "selected_rows_sha256": _list_sha256(values),
        "complete_12_lane_block_count": _complete_block_count(rows),
    }


def _tick_files(root: Path) -> list[Path]:
    files: list[Path] = []
    try:
        entries = list(os.scandir(root))
    except OSError as exc:
        raise NotComputable(f"tick_root_scan_failed:{exc.__class__.__name__}") from exc
    for entry in entries:
        match = TICK_FILE_RE.fullmatch(entry.name)
        if match is None:
            if entry.name.endswith(".parquet"):
                raise NotComputable(f"unknown_tick_shard_name:{entry.name}")
            continue
        oldest, newest = int(match.group(2)), int(match.group(3))
        if oldest > newest:
            raise NotComputable(f"tick_shard_bounds_reversed:{entry.name}")
        if newest < TICK_FIRST_EPOCH or oldest > TICK_LAST_EPOCH:
            continue
        _regular_stat(Path(entry.path), "tick_file")
        files.append(Path(entry.path))
    files.sort(key=lambda path: path.name)
    if not files:
        raise NotComputable("tick_source_window_empty")
    return files


def _possible_tick_dates(files: Sequence[Path]) -> list[str]:
    dates: set[str] = set()
    for path in files:
        match = TICK_FILE_RE.fullmatch(path.name)
        assert match is not None
        first = max(int(match.group(2)), TICK_FIRST_EPOCH)
        last = min(int(match.group(3)), TICK_LAST_EPOCH)
        current = datetime.fromtimestamp(first, UTC).astimezone(NY).date()
        final = datetime.fromtimestamp(last, UTC).astimezone(NY).date()
        while current <= final:
            dates.add(current.isoformat())
            current += timedelta(days=1)
    return sorted(dates)


def _tick_snapshot(root: Path) -> dict[str, Any]:
    files = _tick_files(root)
    entries: list[tuple[str, int, bytes]] = []
    total_bytes = 0
    for path in files:
        raw = _stable_read(path, "tick_file")
        entries.append((path.name, len(raw), hashlib.sha256(raw).digest()))
        total_bytes += len(raw)
    return {
        "logical_source": "public_deriv_ticks_1s_pages/GBPUSD",
        "selection_epoch_inclusive": [TICK_FIRST_EPOCH, TICK_LAST_EPOCH],
        "file_count": len(entries),
        "total_bytes": total_bytes,
        "tree_sha256": _tree_sha256(TICK_TREE_DOMAIN, entries),
        "tree_domain_hex": TICK_TREE_DOMAIN.hex(),
        "possible_ny_dates_from_filename_bounds": _possible_tick_dates(files),
    }


def _measurement() -> dict[str, Any]:
    return {
        "key": {
            "pair": "GBPUSD",
            "timeframe": "15m",
            "target": "direction",
            "side": "DOWN",
        },
        "fixed_quote_window_ny_dates_inclusive": [
            FIRST_NY_DATE.isoformat(),
            LAST_NY_DATE.isoformat(),
        ],
        "minimum_represented_ny_dates": MIN_DATES,
        "stage": "outcome_blind_source_feasibility",
        "stop_rule": "stop_not_computable_when_tick_source_cannot_represent_10_dates",
        "tick_prices_interpreted": False,
        "outcomes_computed": False,
        "source_class": "QUOTE_CONDITIONED_NO_FILL",
        "realized_profit": None,
        "activation": False,
    }


def _build_snapshot(quote_root: Path, tick_root: Path) -> dict[str, Any]:
    snapshot = {
        "schema": SNAPSHOT_SCHEMA,
        "measurement": _measurement(),
        "quote_source": _quote_snapshot(quote_root),
        "tick_source": _tick_snapshot(tick_root),
    }
    evidence.canonical_json_bytes(snapshot)
    return snapshot


def _seal_source(quote_root: Path, tick_root: Path) -> dict[str, Any]:
    first = _build_snapshot(quote_root, tick_root)
    second = _build_snapshot(quote_root, tick_root)
    if first != second:
        raise ScreenError("source changed during outcome-blind sealing")
    possible_dates = first["tick_source"]["possible_ny_dates_from_filename_bounds"]
    if len(possible_dates) >= MIN_DATES:
        raise ScreenError(
            "source feasibility no longer falsifies issue #21; review the full outcome plan before access"
        )
    return first


def _actual_environment() -> dict[str, str]:
    return {name: package_version(name) for name in EXPECTED_ENVIRONMENT}


def _guard_horizon_environment() -> None:
    value = os.environ.get("MX_HOR")
    if value is not None and value != "15":
        raise ScreenError("MX_HOR must be absent or exactly 15")


def _validate_campaign(campaign: Mapping[str, Any], arm: Mapping[str, Any]) -> None:
    if (
        campaign.get("campaign_id") != CAMPAIGN_ID
        or campaign.get("data_use") != "retrospective"
        or campaign.get("key")
        != {"pair": "GBPUSD", "timeframe": "15m", "target": "direction", "side": "DOWN"}
        or arm.get("arm_id") != ARM_ID
        or arm.get("native_result_path") != RESULT_PATH
        or campaign.get("adapter", {}).get("path") != SCRIPT_PATH
        or campaign.get("adapter", {}).get("native_result_contract") != RESULT_SCHEMA
        or list(arm.get("input_evidence_ids", []))
        != list(arm.get("data_use_basis_evidence_ids", []))
        or len(arm.get("input_evidence_ids", [])) != 1
    ):
        raise ScreenError("sealed campaign or arm identity differs")


def _validate_snapshot(value: Any) -> Mapping[str, Any]:
    snapshot = _exact(
        value, {"schema", "measurement", "quote_source", "tick_source"}, "snapshot"
    )
    quote = _exact(
        snapshot["quote_source"],
        {
            "logical_source",
            "selection",
            "relevant_file_count",
            "selected_row_count",
            "selected_ny_dates",
            "selected_rows_sha256",
            "complete_12_lane_block_count",
        },
        "snapshot.quote_source",
    )
    tick = _exact(
        snapshot["tick_source"],
        {
            "logical_source",
            "selection_epoch_inclusive",
            "file_count",
            "total_bytes",
            "tree_sha256",
            "tree_domain_hex",
            "possible_ny_dates_from_filename_bounds",
        },
        "snapshot.tick_source",
    )
    if (
        snapshot["schema"] != SNAPSHOT_SCHEMA
        or snapshot["measurement"] != _measurement()
        or quote["logical_source"] != "public_deriv_quote_audit_jsonl"
        or quote["selection"]
        != "embedded_timestamp_ny_dates_2026-07-02_through_2026-07-20"
        or tick["logical_source"] != "public_deriv_ticks_1s_pages/GBPUSD"
        or tick["selection_epoch_inclusive"] != [TICK_FIRST_EPOCH, TICK_LAST_EPOCH]
        or tick["tree_domain_hex"] != TICK_TREE_DOMAIN.hex()
    ):
        raise ScreenError("source snapshot identity differs")
    for section, fields in (
        (
            quote,
            (
                "relevant_file_count",
                "selected_row_count",
                "complete_12_lane_block_count",
            ),
        ),
        (tick, ("file_count", "total_bytes")),
    ):
        for field in fields:
            if type(section[field]) is not int or section[field] < 0:
                raise ScreenError(f"snapshot {field} must be a nonnegative integer")
    if (
        quote["relevant_file_count"] == 0
        or quote["selected_row_count"] == 0
        or quote["complete_12_lane_block_count"]
        > quote["selected_row_count"] // len(LANES)
        or tick["file_count"] == 0
    ):
        raise ScreenError("source snapshot is empty")
    for value in (quote["selected_rows_sha256"], tick["tree_sha256"]):
        if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise ScreenError("source snapshot digest differs")
    for dates in (
        quote["selected_ny_dates"],
        tick["possible_ny_dates_from_filename_bounds"],
    ):
        if not isinstance(dates, list) or dates != sorted(set(dates)):
            raise ScreenError("source snapshot date inventory differs")
        try:
            parsed_dates = [date.fromisoformat(value) for value in dates]
        except (TypeError, ValueError) as exc:
            raise ScreenError("source snapshot date is not ISO-8601") from exc
        if any(not FIRST_NY_DATE <= value <= LAST_NY_DATE for value in parsed_dates):
            raise ScreenError("source snapshot date is outside the fixed window")
    evidence.canonical_json_bytes(_thaw(snapshot))
    return snapshot


def _input_payload(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": INPUT_SCHEMA,
        "key": {
            "pair": "GBPUSD",
            "timeframe": "15m",
            "target": "direction",
            "side": "DOWN",
        },
        "data_use": "retrospective",
        "ordered_artifact_paths": list(ARTIFACT_PATHS),
        "source_snapshot": _thaw(snapshot),
        "expected_environment": EXPECTED_ENVIRONMENT,
        "protected_access": False,
        "measurement": _measurement(),
        "claim_limit": CLAIM_LIMIT,
    }


def _validate_input(value: Mapping[str, Any], expected_id: str) -> Mapping[str, Any]:
    raw = _exact(
        _thaw(value),
        {"schema", "object_id", "kind", "payload", "artifacts", "dependencies"},
        "input envelope",
    )
    payload = _exact(
        raw["payload"],
        {
            "schema",
            "key",
            "data_use",
            "ordered_artifact_paths",
            "source_snapshot",
            "expected_environment",
            "protected_access",
            "measurement",
            "claim_limit",
        },
        "input payload",
    )
    snapshot = _validate_snapshot(payload["source_snapshot"])
    paths = [item.get("path") for item in raw["artifacts"] if isinstance(item, Mapping)]
    snapshot_bytes = evidence.canonical_json_bytes(_thaw(snapshot))
    first = raw["artifacts"][0] if raw["artifacts"] else {}
    if (
        raw["object_id"] != expected_id
        or raw["kind"] != INPUT_SCHEMA
        or raw["dependencies"] != []
        or payload["schema"] != INPUT_SCHEMA
        or payload["key"]
        != {"pair": "GBPUSD", "timeframe": "15m", "target": "direction", "side": "DOWN"}
        or payload["data_use"] != "retrospective"
        or payload["ordered_artifact_paths"] != list(ARTIFACT_PATHS)
        or paths != list(ARTIFACT_PATHS)
        or payload["expected_environment"] != EXPECTED_ENVIRONMENT
        or payload["protected_access"] is not False
        or payload["measurement"] != _measurement()
        or payload["claim_limit"] != CLAIM_LIMIT
        or first.get("sha256") != hashlib.sha256(snapshot_bytes).hexdigest()
        or first.get("bytes") != len(snapshot_bytes)
    ):
        raise ScreenError("input envelope differs")
    return raw


def _verified_input(input_id: str) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    envelope = evidence.verify_object(input_id, repo_root=ROOT)
    raw = _validate_input(envelope.as_dict(), input_id)
    references = [
        evidence.ArtifactRef.from_dict(dict(item)) for item in raw["artifacts"]
    ]
    snapshot = evidence.decode_canonical_json(
        references[0].read_verified(repo_root=ROOT)
    )
    if snapshot != raw["payload"]["source_snapshot"]:
        raise ScreenError("input payload and snapshot artifact differ")
    for reference in references[1:]:
        reference.verify(repo_root=ROOT)
    return raw, snapshot


def _receipt(
    campaign: Mapping[str, Any],
    arm: Mapping[str, Any],
    status: str,
    reasons: Sequence[str],
) -> dict[str, Any]:
    return {
        "schema": RECEIPT_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "arm_id": ARM_ID,
        "native_result_contract": RESULT_SCHEMA,
        "status": status,
        "reasons": list(reasons),
        "native_result_path": RESULT_PATH,
        "input_evidence_ids": list(arm["input_evidence_ids"]),
        "data_use": campaign["data_use"],
        "data_use_basis_evidence_ids": list(arm["data_use_basis_evidence_ids"]),
        "protected_access": False,
    }


def preflight_arm(
    *,
    campaign: Mapping[str, Any],
    arm: Mapping[str, Any],
    input_evidence: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    _validate_campaign(campaign, arm)
    reasons: list[str] = []
    try:
        if len(input_evidence) != 1:
            raise ScreenError("exactly one input envelope is required")
        _validate_input(input_evidence[0], arm["input_evidence_ids"][0])
        _source_root(QUOTE_ROOT_ENV, "quote root")
        _source_root(TICK_ROOT_ENV, "tick root")
    except ScreenError as exc:
        reasons.append(str(exc))
    return {
        "campaign_id": CAMPAIGN_ID,
        "arm_id": ARM_ID,
        "data_use": "retrospective",
        "data_use_basis_evidence_ids": list(arm["data_use_basis_evidence_ids"]),
        "protected_access": False,
        "admissible": not reasons,
        "reasons": reasons,
    }


def _source_access(*, attempted: bool, verified: bool) -> dict[str, bool]:
    return {
        "source_identities_verified": verified,
        "quote_rows_may_have_been_read": attempted,
        "tick_bytes_may_have_been_hashed": attempted,
        "tick_prices_interpreted": False,
        "outcomes_computed": False,
        "network_access": False,
        "credential_access": False,
        "account_access": False,
        "trade_access": False,
    }


SOURCE_FAILURE_PREFIXES = (
    "source_root_invalid:",
    "quote_root_scan_failed:",
    "quote_file_",
    "quote_gzip_invalid:",
    "quote_utf8_invalid:",
    "quote_json_invalid:",
    "quote_json_semantic_invalid:",
    "quote_timestamp_invalid:",
    "quote_schema_invalid:",
    "quote_source_window_empty",
    "duplicate_quote_timestamp",
    "tick_root_scan_failed:",
    "unknown_tick_shard_name:",
    "tick_shard_bounds_reversed:",
    "tick_file_",
    "tick_source_window_empty",
    "sealed_source_identity_differs",
)


def _valid_unverified_reason(
    reason: str, environment: Mapping[str, str], *, source_attempted: bool
) -> bool:
    if not source_attempted:
        if not environment:
            return (
                re.fullmatch(
                    r"environment_import_failed:[A-Za-z_][A-Za-z0-9_]*", reason
                )
                is not None
            )
        return (
            dict(environment) != EXPECTED_ENVIRONMENT
            and reason == f"environment_versions_differ:{dict(environment)}"
        )
    return dict(environment) == EXPECTED_ENVIRONMENT and reason.startswith(
        SOURCE_FAILURE_PREFIXES
    )


def _result(
    arm: Mapping[str, Any],
    *,
    environment: Mapping[str, str],
    snapshot: Mapping[str, Any],
    reason: str,
    source_attempted: bool,
    source_verified: bool,
) -> dict[str, Any]:
    possible_dates = snapshot["tick_source"]["possible_ny_dates_from_filename_bounds"]
    return {
        "schema": RESULT_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "arm_id": ARM_ID,
        "input_evidence_ids": list(arm["input_evidence_ids"]),
        "status": "not_computable",
        "reasons": [reason],
        "protected_access": False,
        "claim_limit": CLAIM_LIMIT,
        "activation": False,
        "measurement": _measurement(),
        "environment": dict(environment),
        "source_snapshot": _thaw(snapshot),
        "source_access": _source_access(
            attempted=source_attempted, verified=source_verified
        ),
        "counts": {
            "quote_selected_rows": snapshot["quote_source"]["selected_row_count"],
            "quote_complete_blocks": snapshot["quote_source"][
                "complete_12_lane_block_count"
            ],
            "tick_files": snapshot["tick_source"]["file_count"],
            "possible_tick_ny_dates": list(possible_dates),
            "possible_tick_ny_date_count": len(possible_dates),
            "minimum_required_ny_dates": MIN_DATES,
        },
        "settlements": None,
        "inference": None,
        "outcome": "not_computable",
        "realized_profit": None,
    }


def run_arm(
    *, campaign: Mapping[str, Any], arm: Mapping[str, Any], temporary_output_path: Path
) -> dict[str, Any]:
    _validate_campaign(campaign, arm)
    if temporary_output_path.exists() or temporary_output_path.is_symlink():
        raise ScreenError("temporary output must not pre-exist")
    _, snapshot = _verified_input(arm["input_evidence_ids"][0])
    environment: dict[str, str] = {}
    source_attempted = False
    source_verified = False
    try:
        try:
            environment = _actual_environment()
        except Exception as exc:
            raise NotComputable(
                f"environment_import_failed:{exc.__class__.__name__}"
            ) from exc
        if environment != EXPECTED_ENVIRONMENT:
            raise NotComputable(f"environment_versions_differ:{environment}")
        source_attempted = True
        try:
            quote_root = _source_root(QUOTE_ROOT_ENV, "quote root")
            tick_root = _source_root(TICK_ROOT_ENV, "tick root")
        except ScreenError as exc:
            raise NotComputable(f"source_root_invalid:{exc}") from exc
        actual = _build_snapshot(
            quote_root,
            tick_root,
        )
        if actual != snapshot:
            raise NotComputable("sealed_source_identity_differs")
        source_verified = True
        possible_dates = snapshot["tick_source"][
            "possible_ny_dates_from_filename_bounds"
        ]
        reason = f"tick_source_cannot_represent_minimum_dates:{len(possible_dates)}<{MIN_DATES}"
    except NotComputable as exc:
        reason = str(exc)
    result = _result(
        arm,
        environment=environment,
        snapshot=snapshot,
        reason=reason,
        source_attempted=source_attempted,
        source_verified=source_verified,
    )
    _canonical_write(temporary_output_path, result)
    return validate_arm_result(
        campaign=campaign, arm=arm, native_result_path=temporary_output_path
    )


def _validate_result(
    value: Any, campaign: Mapping[str, Any], arm: Mapping[str, Any]
) -> dict[str, Any]:
    _validate_campaign(campaign, arm)
    raw = dict(
        _exact(
            value,
            {
                "schema",
                "campaign_id",
                "arm_id",
                "input_evidence_ids",
                "status",
                "reasons",
                "protected_access",
                "claim_limit",
                "activation",
                "measurement",
                "environment",
                "source_snapshot",
                "source_access",
                "counts",
                "settlements",
                "inference",
                "outcome",
                "realized_profit",
            },
            "native result",
        )
    )
    _, snapshot = _verified_input(arm["input_evidence_ids"][0])
    access = _exact(
        raw["source_access"],
        {
            "source_identities_verified",
            "quote_rows_may_have_been_read",
            "tick_bytes_may_have_been_hashed",
            "tick_prices_interpreted",
            "outcomes_computed",
            "network_access",
            "credential_access",
            "account_access",
            "trade_access",
        },
        "source access",
    )
    if (
        raw["schema"] != RESULT_SCHEMA
        or raw["campaign_id"] != CAMPAIGN_ID
        or raw["arm_id"] != ARM_ID
        or raw["input_evidence_ids"] != list(arm["input_evidence_ids"])
        or raw["status"] != "not_computable"
        or not isinstance(raw["reasons"], list)
        or len(raw["reasons"]) != 1
        or not isinstance(raw["reasons"][0], str)
        or not raw["reasons"][0]
        or raw["protected_access"] is not False
        or raw["claim_limit"] != CLAIM_LIMIT
        or raw["activation"] is not False
        or raw["measurement"] != _measurement()
        or raw["source_snapshot"] != snapshot
        or raw["settlements"] is not None
        or raw["inference"] is not None
        or raw["outcome"] != "not_computable"
        or raw["realized_profit"] is not None
        or any(type(item) is not bool for item in access.values())
        or any(
            access[field]
            for field in (
                "tick_prices_interpreted",
                "outcomes_computed",
                "network_access",
                "credential_access",
                "account_access",
                "trade_access",
            )
        )
    ):
        raise ScreenError("native result identity or claim boundary differs")
    source_verified = access["source_identities_verified"]
    source_attempted = access["quote_rows_may_have_been_read"]
    if access["tick_bytes_may_have_been_hashed"] is not source_attempted:
        raise ScreenError("native result source-attempt stages differ")
    if source_verified and not source_attempted:
        raise ScreenError("verified source was not marked attempted")
    expected_counts = _result(
        arm,
        environment=raw["environment"],
        snapshot=snapshot,
        reason=raw["reasons"][0],
        source_attempted=source_attempted,
        source_verified=source_verified,
    )["counts"]
    if raw["counts"] != expected_counts:
        raise ScreenError("native result counts differ")
    if source_verified:
        possible_dates = snapshot["tick_source"][
            "possible_ny_dates_from_filename_bounds"
        ]
        expected_reason = f"tick_source_cannot_represent_minimum_dates:{len(possible_dates)}<{MIN_DATES}"
        if (
            raw["environment"] != EXPECTED_ENVIRONMENT
            or len(possible_dates) >= MIN_DATES
            or raw["reasons"] != [expected_reason]
        ):
            raise ScreenError("verified source-feasibility decision differs")
    else:
        if not isinstance(raw["environment"], Mapping) or any(
            not isinstance(key, str) or not isinstance(version, str)
            for key, version in raw["environment"].items()
        ):
            raise ScreenError("not-computable environment record differs")
        if not _valid_unverified_reason(
            raw["reasons"][0], raw["environment"], source_attempted=source_attempted
        ):
            raise ScreenError("unverified source failure reason differs")
    evidence.canonical_json_bytes(raw)
    return raw


def validate_arm_result(
    *, campaign: Mapping[str, Any], arm: Mapping[str, Any], native_result_path: Path
) -> dict[str, Any]:
    if native_result_path.is_symlink() or not native_result_path.is_file():
        raise ScreenError("native result must be a regular non-symlink file")
    result = _validate_result(_canonical_read(native_result_path), campaign, arm)
    return _receipt(campaign, arm, result["status"], result["reasons"])


def evaluate_family(
    *,
    campaign: Mapping[str, Any],
    receipts: Sequence[Mapping[str, Any]],
    native_results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if len(receipts) != 1 or len(native_results) != 1:
        raise ScreenError("one complete screen result is required")
    arm = campaign["ordered_arms"][0]
    reference = evidence.ArtifactRef.from_dict(dict(native_results[0]))
    if reference.path != RESULT_PATH:
        raise ScreenError("native result path differs")
    result = _validate_result(
        evidence.decode_canonical_json(reference.read_verified(repo_root=ROOT)),
        campaign,
        arm,
    )
    return {
        "schema": DECISION_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "verdicts": [
            {"arm_id": ARM_ID, "status": "not_computable", "reasons": result["reasons"]}
        ],
        "ordered_candidates": [],
        "decision": "capability_deferred",
        "reasons": result["reasons"],
    }


def _spec(input_id: str) -> dict[str, Any]:
    return {
        "schema": "research-campaign-spec-input/v1",
        "campaign_id": CAMPAIGN_ID,
        "key": {
            "pair": "GBPUSD",
            "timeframe": "15m",
            "target": "direction",
            "side": "DOWN",
        },
        "question": "Can the fixed July 2026 Deriv quote archive support the preregistered GBPUSD 15m DOWN payout-aware screen?",
        "hypothesis": "The fixed public quote/tick source can represent at least 10 New York dates before any outcome read.",
        "falsifier": "The immutable GBPUSD tick shards can represent fewer than 10 fixed-window New York dates.",
        "data_use": "retrospective",
        "ordered_arms": [
            {
                "arm_id": ARM_ID,
                "role": "screen",
                "native_result_path": RESULT_PATH,
                "input_evidence_ids": [input_id],
                "data_use_basis_evidence_ids": [input_id],
            }
        ],
        "adapter": {
            "path": SCRIPT_PATH,
            "run_function": "run_arm",
            "preflight_function": "preflight_arm",
            "validate_function": "validate_arm_result",
            "evaluate_function": "evaluate_family",
            "native_result_contract": RESULT_SCHEMA,
        },
        "stopping_rule": "Authenticate the fixed outcome-free source identities once and stop before tick prices when fewer than 10 dates are representable.",
        "terminal_rule": {
            "allowed_decisions": [
                "no_candidate",
                "inactive_candidate",
                "capability_deferred",
            ],
            "activation": False,
        },
    }


def _resolve_tick_argument(argument: Path) -> Path:
    root = _real_directory(argument, "--tick-root")
    nested = root / "_pages" / "GBPUSD"
    if nested.is_dir() and not nested.is_symlink():
        return _real_directory(nested, "GBPUSD tick page root")
    if root.name == "GBPUSD" and root.parent.name == "_pages":
        return root
    raise ScreenError("--tick-root must be ticks_1s or its _pages/GBPUSD directory")


def _prepare_and_run(quote_argument: Path, tick_argument: Path) -> None:
    import research_campaign_v1 as runner

    quote_root = _real_directory(quote_argument, "--quote-root")
    tick_root = _resolve_tick_argument(tick_argument)
    snapshot_path = ROOT / SNAPSHOT_PATH
    if snapshot_path.exists():
        snapshot = _validate_snapshot(_canonical_read(snapshot_path))
    else:
        if (ROOT / SPEC_PATH).exists():
            raise ScreenError("campaign spec exists without its source snapshot")
        snapshot = _seal_source(quote_root, tick_root)
        _canonical_write(snapshot_path, snapshot)
    artifacts = [
        evidence.ArtifactRef.capture(path, repo_root=ROOT) for path in ARTIFACT_PATHS
    ]
    input_envelope = evidence.EvidenceEnvelope.create(
        kind=INPUT_SCHEMA,
        payload=_input_payload(snapshot),
        artifacts=artifacts,
        dependencies=[],
    )
    evidence.publish(input_envelope, repo_root=ROOT)
    spec = _spec(input_envelope.object_id)
    spec_path = ROOT / SPEC_PATH
    if spec_path.exists():
        if _canonical_read(spec_path) != spec:
            raise ScreenError("campaign spec path contains divergent bytes")
    else:
        _canonical_write(spec_path, spec)
    previous_quote = os.environ.get(QUOTE_ROOT_ENV)
    previous_tick = os.environ.get(TICK_ROOT_ENV)
    os.environ[QUOTE_ROOT_ENV] = str(quote_root)
    os.environ[TICK_ROOT_ENV] = str(tick_root)
    try:
        terminal = runner.run_campaign(spec_path, repo_root=ROOT)
    finally:
        if previous_quote is None:
            os.environ.pop(QUOTE_ROOT_ENV, None)
        else:
            os.environ[QUOTE_ROOT_ENV] = previous_quote
        if previous_tick is None:
            os.environ.pop(TICK_ROOT_ENV, None)
        else:
            os.environ[TICK_ROOT_ENV] = previous_tick
    print(f"[quote-skew] terminal={terminal.object_id}")


def _verify_completed() -> None:
    import research_campaign_v1 as runner

    terminal = runner.verify_campaign(CAMPAIGN_ID, repo_root=ROOT)
    spec = _canonical_read(ROOT / SPEC_PATH)
    validate_arm_result(
        campaign=spec,
        arm=spec["ordered_arms"][0],
        native_result_path=ROOT / RESULT_PATH,
    )
    print(f"[quote-skew] verified terminal={terminal.object_id}")


def _fixture_block(start: datetime) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, (pair, contract) in enumerate(LANES):
        timestamp = start + timedelta(milliseconds=index)
        rows.append(
            {
                "event": "quote_snapshot",
                "timestamp_utc": timestamp.isoformat(),
                "pair": pair,
                "side": "UP" if contract == "CALL" else "DOWN",
                "contract_type": contract,
                "ask": 0.55,
                "payout": 1.0,
                "live_breakeven": 0.55,
                "invalid_reason": None,
                "proposal_id_present": True,
                "quote_age_ms": 1.0,
                "source": "quote_audit",
            }
        )
    return rows


def _self_test() -> None:
    rows = _fixture_block(datetime(2026, 7, 2, 13, 0, tzinfo=UTC))
    assert (
        _list_sha256(rows)
        == hashlib.sha256(evidence.canonical_json_bytes(rows)).hexdigest()
    )
    assert (
        _complete_block_count(
            [
                (_parse_utc(row["timestamp_utc"], "fixture")[1], "2026-07-02", row)
                for row in rows
            ]
        )
        == 1
    )
    assert _strict_json('{"x":1}') == {"x": 1}
    for invalid in ('{"x":1,"x":2}', '{"x":NaN}'):
        try:
            _strict_json(invalid)
        except SemanticJsonError:
            pass
        else:
            raise AssertionError("semantic JSON defect was admitted")
    with tempfile.TemporaryDirectory(prefix="quote-skew-self-test-") as raw:
        root = Path(raw)
        quote_root = root / "quote"
        tick_root = root / "ticks_1s" / "_pages" / "GBPUSD"
        quote_root.mkdir()
        tick_root.mkdir(parents=True)
        first = "\n".join(json.dumps(row, sort_keys=True) for row in rows[:6]) + "\n"
        second = "\n".join(json.dumps(row, sort_keys=True) for row in rows[6:]) + "\n"
        (quote_root / "old.jsonl.1.gz").write_bytes(gzip.compress(first.encode()))
        active = quote_root / "active.jsonl"
        active.write_bytes(second.encode())
        shard = tick_root / f"000000_{TICK_FIRST_EPOCH}_{TICK_FIRST_EPOCH + 10}.parquet"
        shard.write_bytes(b"sealed-tick-fixture")
        snapshot = _build_snapshot(quote_root, tick_root)
        assert snapshot["quote_source"]["selected_row_count"] == 12
        assert snapshot["quote_source"]["complete_12_lane_block_count"] == 1
        before = snapshot["tick_source"]["tree_sha256"]
        later = _fixture_block(datetime(2026, 7, 21, 13, 0, tzinfo=UTC))
        with active.open("ab") as handle:
            handle.write(("\n".join(json.dumps(row) for row in later) + "\n").encode())
        assert _build_snapshot(quote_root, tick_root) == snapshot
        shard.write_bytes(b"tampered-tick-fixture")
        assert _tick_snapshot(tick_root)["tree_sha256"] != before
        truncated = quote_root / "truncated.jsonl"
        truncated.write_bytes(b'{"event":"quote_snapshot"')
        assert _quote_rows(truncated) == []
        semantic = quote_root / "semantic.jsonl"
        semantic.write_bytes(b'{"event":"quote_snapshot","event":"quote_snapshot"}')
        try:
            _quote_rows(semantic)
        except NotComputable as exc:
            assert "semantic" in str(exc)
        else:
            raise AssertionError(
                "final-line semantic JSON defect was excluded as truncation"
            )
    original = os.environ.get("MX_HOR")
    try:
        os.environ["MX_HOR"] = "5"
        try:
            _guard_horizon_environment()
        except ScreenError:
            pass
        else:
            raise AssertionError("ambient non-15m horizon was admitted")
        os.environ["MX_HOR"] = "15"
        _guard_horizon_environment()
    finally:
        if original is None:
            os.environ.pop("MX_HOR", None)
        else:
            os.environ["MX_HOR"] = original
    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    forbidden = {
        "socket",
        "requests",
        "websocket",
        "websockets",
        "deriv_client",
        "deriv_async_client",
    }
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert not imported.intersection(forbidden)
    assert not _valid_unverified_reason("arbitrary", {}, source_attempted=False)
    assert not _valid_unverified_reason(
        "tick_source_cannot_represent_minimum_dates:3<10",
        EXPECTED_ENVIRONMENT,
        source_attempted=True,
    )
    assert _valid_unverified_reason(
        "sealed_source_identity_differs", EXPECTED_ENVIRONMENT, source_attempted=True
    )
    print("[quote-skew] self-test PASS")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("self-test", "run", "verify"))
    parser.add_argument("--quote-root", type=Path)
    parser.add_argument("--tick-root", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    _guard_horizon_environment()
    args = _parser().parse_args(argv)
    if args.command == "self-test":
        if args.quote_root is not None or args.tick_root is not None:
            raise SystemExit("source roots are only valid with run")
        _self_test()
    elif args.command == "run":
        if args.quote_root is None or args.tick_root is None:
            raise SystemExit("run requires --quote-root and --tick-root")
        _prepare_and_run(args.quote_root, args.tick_root)
    else:
        if args.quote_root is not None or args.tick_root is not None:
            raise SystemExit("verify does not read external source roots")
        _verify_completed()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
