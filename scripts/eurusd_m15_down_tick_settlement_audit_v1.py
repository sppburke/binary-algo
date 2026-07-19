#!/usr/bin/env python3
"""Sealed local-tick settlement audit for issue #15 EURUSD 15m DOWN trades."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
import struct
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import evidence_store as evidence


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_ID = "eurusd.m15.down.tick-settlement-audit.v1"
ARM_ID = "historical_tick_settlement_audit.v1"
SCRIPT_PATH = "scripts/eurusd_m15_down_tick_settlement_audit_v1.py"
SNAPSHOT_PATH = "results/json/eurusd_m15_down_tick_source_snapshot_v1.json"
RESULT_PATH = "results/json/eurusd_m15_down_tick_settlement_audit_v1_result.json"
SPEC_PATH = "results/json/eurusd_m15_down_tick_settlement_audit_v1_campaign_spec.json"
SELECTION_PATH = "results/json/eurusd_m15_down_selection_audit_v1_result.json"
SELECTION_SPEC_PATH = "results/json/eurusd_m15_down_selection_audit_v1_campaign_spec.json"
ENV_PATH = "ENVIRONMENT_libs.txt"
RAW_ROOT_ENV = "EURUSD_M15_TICK_RAW_ROOT"
ISSUE15_TERMINAL = "4c3b0d20013b9b6301f965ac45aae9161fa6db4fc8e5bf7f1c123f2fd0b76b8d"
SELECTION_SHA256 = "819f22899a2b83983df2b76be04339c096cb74500addcbc1a6208692fbe4fdef"
SOURCE_TREE_SHA256 = "e7e803ecab4030778d99f3cd2c5296f444d862713ea9c8d5c07ead704c7e5f96"
SOURCE_FILE_COUNT = 77_255
SOURCE_TOTAL_BYTES = 4_122_905_970
SOURCE_DATE_COUNT = 2_837
UNIQUE_DECISIONS = 21_086
ROW_REFERENCES = 39_791
FIRST_SOURCE_FILE = "EURUSD_2012-01-02_01.parquet"
LAST_SOURCE_FILE = "EURUSD_2025-12-31_23.parquet"
SOURCE_DOMAIN = b"eurusd-m15-down-tick-source-tree/v1\0"
CUTOFF_2026 = 1_767_225_600.0
ENTRY_OFFSET_SECONDS = 61
EXPIRY_SECONDS = 900
TOLERANCE_SECONDS = 30.0
BREAKEVEN = 0.541
MIN_PATH_VALID_FRACTION = 0.95
MIN_PATH_VALID = 60
MIN_FROZEN_VALID = 50
EXPECTED_ENVIRONMENT = {"numpy": "2.4.6", "pandas": "3.0.3", "pyarrow": "24.0.0"}

INPUT_SCHEMA = "research.eurusd_m15_down_tick_settlement_audit.input/v1"
SNAPSHOT_SCHEMA = "eurusd-m15-down-tick-source-snapshot/v1"
RESULT_SCHEMA = "eurusd-m15-down-tick-settlement-audit-result/v1"
RECEIPT_SCHEMA = "research-campaign-adapter-receipt/v1"
DECISION_SCHEMA = "research-campaign-family-decision/v1"
CLAIM_LIMIT = (
    "Historical pre-2026 settlement-compatibility diagnostic on an archive of unknown original "
    "provenance; not Deriv-feed fidelity, payout evidence, fresh OOS, certification, activation, "
    "deployment authority, or a replacement for the registered 0.5742 proxy claim."
)
FILE_RE = re.compile(r"^EURUSD_(\d{4}-\d{2}-\d{2})_(\d{2})\.parquet$")


class AuditError(ValueError):
    """The sealed audit contract is malformed."""


class NotComputable(RuntimeError):
    """A post-attempt computability gate failed."""


def _exact(value: Any, fields: set[str], name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        actual = set(value) if isinstance(value, Mapping) else set()
        raise AuditError(
            f"{name} fields differ: missing={sorted(fields - actual)} "
            f"unknown={sorted(actual - fields)}"
        )
    return value


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise AuditError(f"{name} must be finite numeric")
    return float(value)


def _canonical_read(path: Path) -> Any:
    return evidence.decode_canonical_json(path.read_bytes())


def _canonical_write(path: Path, value: Any) -> None:
    path.write_bytes(evidence.canonical_json_bytes(value))


def _snapshot() -> dict[str, Any]:
    return {
        "schema": SNAPSHOT_SCHEMA,
        "logical_source": "local_raw_tick_archive/EURUSD",
        "provenance": "unknown",
        "selection_result": {"path": SELECTION_PATH, "sha256": SELECTION_SHA256},
        "selection_rule": (
            "union issue15 legacy_full_val and worst_val_half path-test timestamps plus "
            "frozen 2024/2025; candidate dates {d,d+1} intersect filename dates <=2025-12-31"
        ),
        "semantic_years": [2012, 2025],
        "required_utc_dates": SOURCE_DATE_COUNT,
        "expected_columns": ["timestamp_utc", "bid", "ask", "bid-vol", "ask-vol"],
        "first_file": FIRST_SOURCE_FILE,
        "last_file": LAST_SOURCE_FILE,
        "file_count": SOURCE_FILE_COUNT,
        "total_bytes": SOURCE_TOTAL_BYTES,
        "tree_sha256": SOURCE_TREE_SHA256,
        "tree_algorithm": {
            "domain_hex": SOURCE_DOMAIN.hex(),
            "order": "basename_utf8_ascending",
            "components": ["u32be_name_bytes", "name_utf8", "u64be_file_bytes", "sha256_raw"],
        },
    }


def _measurement() -> dict[str, Any]:
    return {
        "pair": "EURUSD",
        "timeframe": "15m",
        "side": "DOWN",
        "selectors": ["legacy_full_val", "worst_val_half"],
        "semantic_years": [2012, 2025],
        "protected_year_excluded": 2026,
        "entry_target": "decision_timestamp+60+1",
        "expiry_target": "entry_target+900",
        "entry_rule": "first_tick_at_or_after_target",
        "exit_rule": "last_tick_at_or_before_target",
        "tolerance_seconds": TOLERANCE_SECONDS,
        "price": "bid_ask_midpoint",
        "ties": "lose",
        "breakeven": BREAKEVEN,
        "minimum_path_valid_fraction": MIN_PATH_VALID_FRACTION,
        "minimum_path_valid_trades": MIN_PATH_VALID,
        "minimum_frozen_valid_trades": MIN_FROZEN_VALID,
        "p10": "numpy.percentile(method=linear)",
        "bootstrap": {"seed": 7, "resamples": 5000, "percentiles": [2.5, 97.5]},
    }


def _environment() -> dict[str, str]:
    import numpy
    import pandas
    import pyarrow

    actual = {"numpy": numpy.__version__, "pandas": pandas.__version__, "pyarrow": pyarrow.__version__}
    if actual != EXPECTED_ENVIRONMENT:
        raise NotComputable(f"environment_versions_differ:{actual}")
    return actual


def _raw_root() -> Path:
    value = os.environ.get(RAW_ROOT_ENV)
    if not value:
        raise AuditError(f"{RAW_ROOT_ENV} is missing")
    argument = Path(value)
    if argument.is_symlink():
        raise AuditError("raw root must not be a symlink")
    try:
        root = argument.resolve(strict=True)
    except OSError as exc:
        raise AuditError("raw root does not exist") from exc
    if not root.is_dir():
        raise AuditError("raw root must be a directory")
    return root


def _validate_campaign(campaign: Mapping[str, Any], arm: Mapping[str, Any]) -> None:
    if (
        campaign.get("campaign_id") != CAMPAIGN_ID
        or campaign.get("data_use") != "retrospective"
        or campaign.get("key")
        != {"pair": "EURUSD", "timeframe": "15m", "target": "direction", "side": "DOWN"}
        or arm.get("arm_id") != ARM_ID
        or arm.get("native_result_path") != RESULT_PATH
        or campaign.get("adapter", {}).get("path") != SCRIPT_PATH
        or campaign.get("adapter", {}).get("native_result_contract") != RESULT_SCHEMA
        or list(arm.get("input_evidence_ids", [])) != list(arm.get("data_use_basis_evidence_ids", []))
        or len(arm.get("input_evidence_ids", [])) != 1
    ):
        raise AuditError("sealed campaign or arm identity differs")


def _validate_input_envelope(value: Mapping[str, Any], expected_id: str) -> Mapping[str, Any]:
    raw = _exact(
        value,
        {"schema", "object_id", "kind", "payload", "artifacts", "dependencies"},
        "input envelope",
    )
    payload = _exact(
        raw["payload"],
        {
            "schema", "key", "data_use", "ordered_artifact_paths", "issue15_terminal",
            "source_snapshot", "expected_environment", "protected_access", "claim_limit",
        },
        "input payload",
    )
    expected_paths = [SNAPSHOT_PATH, SELECTION_PATH, ENV_PATH]
    artifact_paths = [item.get("path") for item in raw["artifacts"] if isinstance(item, Mapping)]
    if (
        raw["object_id"] != expected_id
        or raw["kind"] != INPUT_SCHEMA
        or list(raw["dependencies"]) != [ISSUE15_TERMINAL]
        or payload["schema"] != INPUT_SCHEMA
        or payload["key"]
        != {"pair": "EURUSD", "timeframe": "15m", "target": "direction", "side": "DOWN"}
        or payload["data_use"] != "retrospective"
        or payload["ordered_artifact_paths"] != expected_paths
        or artifact_paths != expected_paths
        or payload["issue15_terminal"] != ISSUE15_TERMINAL
        or payload["source_snapshot"] != _snapshot()
        or payload["expected_environment"] != EXPECTED_ENVIRONMENT
        or payload["protected_access"] is not False
        or payload["claim_limit"] != CLAIM_LIMIT
    ):
        raise AuditError("input envelope differs")
    return raw


def _receipt(
    campaign: Mapping[str, Any], arm: Mapping[str, Any], status: str, reasons: Sequence[str]
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
    *, campaign: Mapping[str, Any], arm: Mapping[str, Any], input_evidence: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    _validate_campaign(campaign, arm)
    reasons: list[str] = []
    try:
        if len(input_evidence) != 1:
            raise AuditError("exactly one input envelope is required")
        _validate_input_envelope(input_evidence[0], arm["input_evidence_ids"][0])
        _raw_root()
    except AuditError as exc:
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


def _verified_inputs(input_id: str) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    envelope = evidence.verify_object(input_id, repo_root=ROOT)
    raw = _validate_input_envelope(envelope.as_dict(), input_id)
    refs = [evidence.ArtifactRef.from_dict(dict(item)) for item in raw["artifacts"]]
    snapshot = evidence.decode_canonical_json(refs[0].read_verified(repo_root=ROOT))
    if snapshot != _snapshot():
        raise AuditError("source snapshot differs")
    selection_bytes = refs[1].read_verified(repo_root=ROOT)
    if hashlib.sha256(selection_bytes).hexdigest() != SELECTION_SHA256:
        raise AuditError("issue #15 result digest differs")
    selection = evidence.decode_canonical_json(selection_bytes)
    import eurusd_m15_down_selection_audit_v1 as prior

    prior_spec = _canonical_read(ROOT / SELECTION_SPEC_PATH)
    prior._validate_result(selection, prior_spec, prior_spec["ordered_arms"][0])
    return raw, selection


def _selection_rows(selection: Mapping[str, Any]) -> tuple[list[dict[str, Any]], set[int]]:
    references: list[dict[str, Any]] = []
    for path in selection["paths"]:
        for selector in ("legacy_full_val", "worst_val_half"):
            for trade in path["selectors"][selector]["test"]["trades"]:
                references.append(trade)
    for year in ("2024", "2025"):
        references.extend(selection["frozen_forward"]["years"][year]["trades"])
    timestamps = {int(row["timestamp"]) for row in references}
    if len(references) != ROW_REFERENCES or len(timestamps) != UNIQUE_DECISIONS:
        raise NotComputable("issue15_trade_inventory_differs")
    if any(ts >= int(CUTOFF_2026) for ts in timestamps):
        raise NotComputable("issue15_contains_2026_timestamp")
    return references, timestamps


def _required_labels(timestamps: set[int]) -> tuple[list[str], list[str]]:
    dates = sorted({datetime.fromtimestamp(ts, timezone.utc).date() for ts in timestamps})
    if len(dates) != SOURCE_DATE_COUNT:
        raise NotComputable("required_utc_date_count_differs")
    labels = sorted(
        {
            candidate.isoformat()
            for date in dates
            for candidate in (date, date + timedelta(days=1))
            if candidate <= datetime(2025, 12, 31, tzinfo=timezone.utc).date()
        }
    )
    return [date.isoformat() for date in dates], labels


def _source_files(root: Path, labels: Sequence[str]) -> tuple[list[Path], dict[str, list[Path]]]:
    allowed = set(labels)
    selected: list[Path] = []
    by_label: dict[str, list[Path]] = {label: [] for label in labels}
    for entry in os.scandir(root):
        name = entry.name
        if not name.startswith("EURUSD_"):
            continue
        match = FILE_RE.fullmatch(name)
        candidate_label = name[7:17] if len(name) >= 17 else ""
        if candidate_label in allowed and match is None:
            raise NotComputable(f"unknown_admitted_source_name:{name}")
        if match is None or match.group(1) not in allowed:
            continue
        info = entry.stat(follow_symlinks=False)
        if entry.is_symlink() or not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise NotComputable(f"source_file_type_differs:{name}")
        path = Path(entry.path)
        selected.append(path)
        by_label[match.group(1)].append(path)
    selected.sort(key=lambda path: path.name)
    for paths in by_label.values():
        paths.sort(key=lambda path: path.name)
    if (
        len(selected) != SOURCE_FILE_COUNT
        or not selected
        or selected[0].name != FIRST_SOURCE_FILE
        or selected[-1].name != LAST_SOURCE_FILE
    ):
        raise NotComputable("source_file_inventory_differs")
    if len({path.name for path in selected}) != len(selected):
        raise NotComputable("duplicate_source_basename")
    return selected, by_label


def _stable_read(path: Path) -> bytes:
    before = path.stat(follow_symlinks=False)
    if path.is_symlink() or not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
        raise NotComputable(f"source_file_type_differs:{path.name}")
    with path.open("rb") as handle:
        opened = os.fstat(handle.fileno())
        if (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino):
            raise NotComputable(f"source_binding_changed:{path.name}")
        raw = handle.read()
        after_fd = os.fstat(handle.fileno())
    after_name = path.stat(follow_symlinks=False)
    identity = lambda item: (
        item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns, item.st_ctime_ns, item.st_nlink
    )
    if identity(opened) != identity(after_fd) or identity(after_fd) != identity(after_name):
        raise NotComputable(f"source_file_changed:{path.name}")
    if len(raw) != opened.st_size:
        raise NotComputable(f"source_file_short_read:{path.name}")
    return raw


def _tree_digest(entries: Sequence[tuple[str, int, bytes]]) -> str:
    digest = hashlib.sha256(SOURCE_DOMAIN)
    for name, size, file_digest in entries:
        encoded = name.encode("utf-8")
        digest.update(struct.pack(">I", len(encoded)))
        digest.update(encoded)
        digest.update(struct.pack(">Q", size))
        digest.update(file_digest)
    return digest.hexdigest()


def _authenticate_source(files: Sequence[Path]) -> dict[str, tuple[int, str]]:
    identities: dict[str, tuple[int, str]] = {}
    entries: list[tuple[str, int, bytes]] = []
    total = 0
    for index, path in enumerate(files, 1):
        raw = _stable_read(path)
        raw_digest = hashlib.sha256(raw).digest()
        identities[path.name] = (len(raw), raw_digest.hex())
        entries.append((path.name, len(raw), raw_digest))
        total += len(raw)
        if index % 10_000 == 0:
            print(f"[tick-audit] authenticated source files={index}/{len(files)}", flush=True)
    if total != SOURCE_TOTAL_BYTES or _tree_digest(entries) != SOURCE_TREE_SHA256:
        raise NotComputable("source_tree_identity_differs")
    return identities


def _parse_label(
    label: str,
    by_label: Mapping[str, Sequence[Path]],
    identities: Mapping[str, tuple[int, str]],
    parsed_files: set[str],
) -> Any:
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq

    frames = []
    for path in by_label.get(label, []):
        if path.name in parsed_files:
            raise NotComputable(f"source_file_parsed_twice:{path.name}")
        raw = _stable_read(path)
        expected = identities.get(path.name)
        if expected != (len(raw), hashlib.sha256(raw).hexdigest()):
            raise NotComputable(f"source_file_changed_before_parse:{path.name}")
        try:
            table = pq.read_table(
                pa.BufferReader(raw),
                columns=["timestamp_utc", "bid", "ask"],
                filters=[("timestamp_utc", "<", CUTOFF_2026)],
            )
        except Exception as exc:
            raise NotComputable(f"source_parquet_read_failed:{path.name}:{type(exc).__name__}") from exc
        frame = table.to_pandas()
        frames.append(frame)
        parsed_files.add(path.name)
    if not frames:
        return pd.DataFrame(columns=["timestamp_utc", "bid", "ask"])
    return pd.concat(frames, ignore_index=True)


def _clean_ticks(frames: Sequence[Any]) -> tuple[Any, Any]:
    import numpy as np
    import pandas as pd

    available = [frame for frame in frames if len(frame)]
    if not available:
        return np.array([], dtype=float), np.array([], dtype=float)
    ticks = pd.concat(available, ignore_index=True).sort_values("timestamp_utc", kind="mergesort")
    values = ticks[["timestamp_utc", "bid", "ask"]].to_numpy(dtype=float)
    if (
        not np.all(np.isfinite(values))
        or np.any(values[:, 0] >= CUTOFF_2026)
        or np.any(values[:, 1] <= 0)
        or np.any(values[:, 2] <= 0)
        or np.any(values[:, 2] < values[:, 1])
    ):
        raise NotComputable("tick_quote_invariant_failed")
    duplicate = ticks.duplicated("timestamp_utc", keep=False)
    if bool(duplicate.any()):
        conflict = ticks.loc[duplicate].groupby("timestamp_utc", sort=False)[["bid", "ask"]].nunique()
        if bool((conflict > 1).any().any()):
            raise NotComputable("conflicting_duplicate_tick_timestamp")
        ticks = ticks.drop_duplicates("timestamp_utc", keep="first")
    ts = ticks["timestamp_utc"].to_numpy(dtype=float)
    if len(ts) > 1 and not bool(np.all(np.diff(ts) > 0)):
        raise NotComputable("tick_clock_not_strict")
    mid = ((ticks["bid"] + ticks["ask"]) / 2.0).to_numpy(dtype=float)
    return ts, mid


def _settle_batch(decisions: Sequence[int], tick_ts: Any, tick_mid: Any) -> list[dict[str, Any]]:
    import numpy as np

    output: list[dict[str, Any]] = []
    n = len(tick_ts)
    for decision in decisions:
        entry_target = float(decision + ENTRY_OFFSET_SECONDS)
        expiry_target = entry_target + EXPIRY_SECONDS
        row: dict[str, Any] = {
            "timestamp": int(decision),
            "valid": False,
            "reason": None,
            "entry_target": entry_target,
            "expiry_target": expiry_target,
            "entry_timestamp": None,
            "exit_timestamp": None,
            "entry_mid": None,
            "exit_mid": None,
            "entry_delay_seconds": None,
            "exit_staleness_seconds": None,
            "return_sign": None,
            "tick_direction": None,
            "down_win": None,
        }
        if n == 0:
            row["reason"] = "no_ticks"
            output.append(row)
            continue
        entry_index = int(np.searchsorted(tick_ts, entry_target, side="left"))
        if entry_index >= n:
            row["reason"] = "entry_missing"
            output.append(row)
            continue
        entry_delay = float(tick_ts[entry_index] - entry_target)
        if entry_delay > TOLERANCE_SECONDS:
            row["reason"] = "entry_late"
            output.append(row)
            continue
        exit_index = int(np.searchsorted(tick_ts, expiry_target, side="right")) - 1
        if exit_index <= entry_index:
            row["reason"] = "distinct_exit_missing"
            output.append(row)
            continue
        exit_staleness = float(expiry_target - tick_ts[exit_index])
        if exit_staleness > TOLERANCE_SECONDS:
            row["reason"] = "exit_stale"
            output.append(row)
            continue
        entry_mid = float(tick_mid[entry_index])
        exit_mid = float(tick_mid[exit_index])
        sign = 1 if exit_mid > entry_mid else (-1 if exit_mid < entry_mid else 0)
        row.update(
            {
                "valid": True,
                "entry_timestamp": float(tick_ts[entry_index]),
                "exit_timestamp": float(tick_ts[exit_index]),
                "entry_mid": entry_mid,
                "exit_mid": exit_mid,
                "entry_delay_seconds": entry_delay,
                "exit_staleness_seconds": exit_staleness,
                "return_sign": sign,
                "tick_direction": "UP" if sign > 0 else ("DOWN" if sign < 0 else "TIE"),
                "down_win": bool(sign < 0),
            }
        )
        output.append(row)
    return output


def _settle_all(
    timestamps: set[int], date_labels: Sequence[str], by_label: Mapping[str, Sequence[Path]],
    identities: Mapping[str, tuple[int, str]],
) -> list[dict[str, Any]]:
    decisions_by_date: dict[str, list[int]] = {}
    for ts in sorted(timestamps):
        label = datetime.fromtimestamp(ts, timezone.utc).date().isoformat()
        decisions_by_date.setdefault(label, []).append(ts)
    parsed_files: set[str] = set()
    cache: dict[str, Any] = {}
    settlements: list[dict[str, Any]] = []
    ordered_dates = sorted(decisions_by_date)
    for position, label in enumerate(ordered_dates):
        date = datetime.strptime(label, "%Y-%m-%d").date()
        needed = [label]
        following = date + timedelta(days=1)
        if following.year <= 2025:
            needed.append(following.isoformat())
        for source_label in needed:
            if source_label not in cache:
                cache[source_label] = _parse_label(source_label, by_label, identities, parsed_files)
        tick_ts, tick_mid = _clean_ticks([cache[source_label] for source_label in needed])
        settlements.extend(_settle_batch(decisions_by_date[label], tick_ts, tick_mid))
        if (position + 1) % 250 == 0:
            print(
                f"[tick-audit] settled UTC dates={position + 1}/{len(ordered_dates)} "
                f"decisions={len(settlements)}/{UNIQUE_DECISIONS}",
                flush=True,
            )
        if position + 1 < len(ordered_dates):
            next_date = datetime.strptime(ordered_dates[position + 1], "%Y-%m-%d").date()
            retain = {next_date.isoformat()}
            if next_date.year <= 2025 and (next_date + timedelta(days=1)).year <= 2025:
                retain.add((next_date + timedelta(days=1)).isoformat())
            cache = {key: value for key, value in cache.items() if key in retain}
        else:
            cache.clear()
    if parsed_files != set(identities):
        raise NotComputable("authenticated_and_parsed_source_sets_differ")
    if len(settlements) != UNIQUE_DECISIONS:
        raise NotComputable("settlement_inventory_differs")
    return settlements


def _summary(values: Sequence[float]) -> dict[str, Any]:
    import numpy as np

    array = np.asarray(values, dtype=float)
    if len(array) != 15 or not bool(np.all(np.isfinite(array))):
        raise NotComputable("complete_15_path_summary_unavailable")
    return {
        "n_paths": 15,
        "mean": float(array.mean()),
        "p10": float(np.percentile(array, 10, method="linear")),
        "min": float(array.min()),
        "max": float(array.max()),
        "frac_clear_breakeven": float((array >= BREAKEVEN).mean()),
    }


def _bootstrap(wins: Sequence[bool]) -> list[float]:
    import numpy as np

    values = np.asarray(wins, dtype=float)
    if len(values) < MIN_FROZEN_VALID:
        raise NotComputable("frozen_valid_trades_below_50")
    rng = np.random.default_rng(7)
    samples = np.array(
        [values[rng.integers(0, len(values), len(values))].mean() for _ in range(5000)]
    )
    return [float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))]


def _metric(trades: Sequence[Mapping[str, Any]], settled: Mapping[int, Mapping[str, Any]]) -> dict[str, Any]:
    rows = [settled[int(trade["timestamp"])] for trade in trades]
    valid_pairs = [(trade, row) for trade, row in zip(trades, rows) if row["valid"]]
    valid_n = len(valid_pairs)
    input_n = len(trades)
    invalid = Counter(row["reason"] for row in rows if not row["valid"])
    tick_wins = [bool(row["down_win"]) for _, row in valid_pairs]
    proxy_wins = [bool(trade["down_win"]) for trade, _ in valid_pairs]
    ties = sum(row["tick_direction"] == "TIE" for _, row in valid_pairs)
    tick_rate = sum(tick_wins) / valid_n if valid_n else None
    proxy_rate = sum(proxy_wins) / valid_n if valid_n else None
    return {
        "input_n": input_n,
        "valid_n": valid_n,
        "valid_fraction": valid_n / input_n if input_n else 0.0,
        "tick_down_win_rate": tick_rate,
        "same_subset_proxy_win_rate": proxy_rate,
        "tick_minus_proxy": tick_rate - proxy_rate if tick_rate is not None else None,
        "ties": ties,
        "invalid_reasons": dict(sorted(invalid.items())),
    }


def _views(
    selection: Mapping[str, Any], settlements: Sequence[Mapping[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], list[str]]:
    settled = {int(row["timestamp"]): row for row in settlements}
    if len(settled) != len(settlements):
        raise NotComputable("duplicate_settlement_timestamp")
    path_results = []
    rates: dict[str, dict[str, list[float]]] = {
        selector: {"tick": [], "proxy": []}
        for selector in ("legacy_full_val", "worst_val_half")
    }
    reasons: list[str] = []
    for index, path in enumerate(selection["paths"]):
        selector_results = {}
        for selector in ("legacy_full_val", "worst_val_half"):
            metric = _metric(path["selectors"][selector]["test"]["trades"], settled)
            selector_results[selector] = metric
            if metric["valid_fraction"] < MIN_PATH_VALID_FRACTION or metric["valid_n"] < MIN_PATH_VALID:
                reasons.append(f"{selector}_path_{index}_coverage_gate_failed")
            if metric["tick_down_win_rate"] is not None:
                rates[selector]["tick"].append(metric["tick_down_win_rate"])
                rates[selector]["proxy"].append(metric["same_subset_proxy_win_rate"])
        path_results.append(
            {
                "path_index": index,
                "test_groups": list(path["test_groups"]),
                "selectors": selector_results,
            }
        )
    summaries: dict[str, Any] = {"tick": {}, "same_subset_proxy": {}, "tick_minus_proxy": {}}
    for selector in ("legacy_full_val", "worst_val_half"):
        if len(rates[selector]["tick"]) == 15:
            tick_summary = _summary(rates[selector]["tick"])
            proxy_summary = _summary(rates[selector]["proxy"])
            summaries["tick"][selector] = tick_summary
            summaries["same_subset_proxy"][selector] = proxy_summary
            summaries["tick_minus_proxy"][selector] = {
                key: tick_summary[key] - proxy_summary[key]
                for key in ("mean", "p10", "min", "max", "frac_clear_breakeven")
            }
    frozen: dict[str, Any] = {}
    for year in ("2024", "2025"):
        trades = selection["frozen_forward"]["years"][year]["trades"]
        metric = _metric(trades, settled)
        if metric["valid_fraction"] < MIN_PATH_VALID_FRACTION or metric["valid_n"] < MIN_FROZEN_VALID:
            reasons.append(f"frozen_{year}_coverage_gate_failed")
            ci = []
        else:
            wins = [bool(settled[int(trade["timestamp"])]["down_win"]) for trade in trades if settled[int(trade["timestamp"])]["valid"]]
            ci = _bootstrap(wins)
        frozen[year] = {**metric, "bootstrap_ci95": ci}
    return path_results, summaries, frozen, sorted(set(reasons))


def _outcome(summaries: Mapping[str, Any], computable: bool) -> str:
    if not computable:
        return "not_computable"
    primary = summaries["tick"]["worst_val_half"]
    if primary["p10"] < BREAKEVEN or primary["frac_clear_breakeven"] < 0.8:
        return "historical_tick_settlement_falsified"
    return "historical_tick_settlement_not_falsified"


def _empty_result(arm: Mapping[str, Any], reason: str, environment: Mapping[str, str]) -> dict[str, Any]:
    return {
        "schema": RESULT_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "arm_id": ARM_ID,
        "input_evidence_ids": list(arm["input_evidence_ids"]),
        "status": "not_computable",
        "reasons": [reason],
        "protected_access": False,
        "claim_limit": CLAIM_LIMIT,
        "measurement": _measurement(),
        "environment": dict(environment),
        "source": {"snapshot": _snapshot(), "verified": False},
        "settlements": [],
        "path_results": [],
        "summaries": {},
        "frozen_forward": {},
        "outcome": "not_computable",
    }


def run_arm(
    *, campaign: Mapping[str, Any], arm: Mapping[str, Any], temporary_output_path: Path
) -> dict[str, Any]:
    _validate_campaign(campaign, arm)
    if temporary_output_path.exists() or temporary_output_path.is_symlink():
        raise AuditError("temporary output must not pre-exist")
    environment: dict[str, str] = {}
    try:
        _, selection = _verified_inputs(arm["input_evidence_ids"][0])
        environment = _environment()
        _, timestamps = _selection_rows(selection)
        date_labels, source_labels = _required_labels(timestamps)
        root = _raw_root()
        files, by_label = _source_files(root, source_labels)
        identities = _authenticate_source(files)
        settlements = _settle_all(timestamps, date_labels, by_label, identities)
        path_results, summaries, frozen, reasons = _views(selection, settlements)
        status = "completed" if not reasons else "not_computable"
        result = {
            "schema": RESULT_SCHEMA,
            "campaign_id": CAMPAIGN_ID,
            "arm_id": ARM_ID,
            "input_evidence_ids": list(arm["input_evidence_ids"]),
            "status": status,
            "reasons": reasons,
            "protected_access": False,
            "claim_limit": CLAIM_LIMIT,
            "measurement": _measurement(),
            "environment": environment,
            "source": {"snapshot": _snapshot(), "verified": True},
            "settlements": settlements,
            "path_results": path_results,
            "summaries": summaries,
            "frozen_forward": frozen,
            "outcome": _outcome(summaries, computable=not reasons),
        }
    except NotComputable as exc:
        result = _empty_result(arm, str(exc), environment)
    _canonical_write(temporary_output_path, result)
    return validate_arm_result(campaign=campaign, arm=arm, native_result_path=temporary_output_path)


def _validate_settlement(row: Mapping[str, Any]) -> None:
    fields = {
        "timestamp", "valid", "reason", "entry_target", "expiry_target", "entry_timestamp",
        "exit_timestamp", "entry_mid", "exit_mid", "entry_delay_seconds",
        "exit_staleness_seconds", "return_sign", "tick_direction", "down_win",
    }
    raw = _exact(row, fields, "settlement")
    if not isinstance(raw["timestamp"], int) or raw["timestamp"] >= int(CUTOFF_2026):
        raise AuditError("settlement timestamp differs")
    if raw["entry_target"] != raw["timestamp"] + ENTRY_OFFSET_SECONDS or raw["expiry_target"] != raw["entry_target"] + EXPIRY_SECONDS:
        raise AuditError("settlement targets differ")
    if raw["valid"] is True:
        if raw["reason"] is not None or raw["tick_direction"] not in {"UP", "DOWN", "TIE"}:
            raise AuditError("valid settlement status differs")
        for field in (
            "entry_timestamp", "exit_timestamp", "entry_mid", "exit_mid",
            "entry_delay_seconds", "exit_staleness_seconds",
        ):
            _finite(raw[field], field)
        if raw["entry_timestamp"] >= CUTOFF_2026 or raw["exit_timestamp"] >= CUTOFF_2026:
            raise AuditError("settlement used 2026 tick")
        sign = 1 if raw["exit_mid"] > raw["entry_mid"] else (-1 if raw["exit_mid"] < raw["entry_mid"] else 0)
        expected_direction = "UP" if sign > 0 else ("DOWN" if sign < 0 else "TIE")
        if raw["return_sign"] != sign or raw["tick_direction"] != expected_direction or raw["down_win"] is not (sign < 0):
            raise AuditError("settlement outcome differs")
    else:
        if not isinstance(raw["reason"], str) or raw["reason"] == "":
            raise AuditError("invalid settlement reason missing")


def _validate_result(
    value: Any, campaign: Mapping[str, Any], arm: Mapping[str, Any]
) -> dict[str, Any]:
    _validate_campaign(campaign, arm)
    fields = {
        "schema", "campaign_id", "arm_id", "input_evidence_ids", "status", "reasons",
        "protected_access", "claim_limit", "measurement", "environment", "source",
        "settlements", "path_results", "summaries", "frozen_forward", "outcome",
    }
    raw = _exact(value, fields, "native result")
    if (
        raw["schema"] != RESULT_SCHEMA
        or raw["campaign_id"] != CAMPAIGN_ID
        or raw["arm_id"] != ARM_ID
        or raw["input_evidence_ids"] != list(arm["input_evidence_ids"])
        or raw["protected_access"] is not False
        or raw["claim_limit"] != CLAIM_LIMIT
        or raw["measurement"] != _measurement()
        or raw["environment"] not in ({}, EXPECTED_ENVIRONMENT)
        or not isinstance(raw["reasons"], list)
    ):
        raise AuditError("native result identity differs")
    if raw["status"] == "not_computable" and raw["settlements"] == []:
        if not raw["reasons"] or raw["outcome"] != "not_computable" or raw["source"] != {"snapshot": _snapshot(), "verified": False}:
            raise AuditError("empty not-computable result differs")
        return dict(raw)
    if raw["source"] != {"snapshot": _snapshot(), "verified": True}:
        raise AuditError("verified source identity differs")
    if not isinstance(raw["settlements"], list) or len(raw["settlements"]) != UNIQUE_DECISIONS:
        raise AuditError("settlement count differs")
    for row in raw["settlements"]:
        _validate_settlement(row)
    _, selection = _verified_inputs(arm["input_evidence_ids"][0])
    expected_paths, expected_summaries, expected_frozen, expected_reasons = _views(selection, raw["settlements"])
    expected_status = "completed" if not expected_reasons else "not_computable"
    expected_outcome = _outcome(expected_summaries, computable=not expected_reasons)
    if (
        raw["path_results"] != expected_paths
        or raw["summaries"] != expected_summaries
        or raw["frozen_forward"] != expected_frozen
        or raw["reasons"] != expected_reasons
        or raw["status"] != expected_status
        or raw["outcome"] != expected_outcome
    ):
        raise AuditError("native summaries or outcome differ")
    evidence.canonical_json_bytes(raw)
    return dict(raw)


def validate_arm_result(
    *, campaign: Mapping[str, Any], arm: Mapping[str, Any], native_result_path: Path
) -> dict[str, Any]:
    if native_result_path.is_symlink() or not native_result_path.is_file():
        raise AuditError("native result must be a regular non-symlink file")
    result = _validate_result(_canonical_read(native_result_path), campaign, arm)
    return _receipt(campaign, arm, result["status"], result["reasons"])


def evaluate_family(
    *, campaign: Mapping[str, Any], receipts: Sequence[Mapping[str, Any]],
    native_results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if len(receipts) != 1 or len(native_results) != 1:
        raise AuditError("one complete arm result is required")
    arm = campaign["ordered_arms"][0]
    reference = evidence.ArtifactRef.from_dict(dict(native_results[0]))
    if reference.path != RESULT_PATH:
        raise AuditError("native result path differs")
    result = _validate_result(
        evidence.decode_canonical_json(reference.read_verified(repo_root=ROOT)), campaign, arm
    )
    if result["status"] == "not_computable":
        status, decision, candidates = "not_computable", "capability_deferred", []
        reasons = list(result["reasons"])
    elif result["outcome"] == "historical_tick_settlement_falsified":
        status, decision, candidates = "killed", "no_candidate", []
        reasons = ["historical_tick_settlement_falsified"]
    else:
        status, decision, candidates = "candidate", "inactive_candidate", [ARM_ID]
        reasons = []
    return {
        "schema": DECISION_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "verdicts": [{"arm_id": ARM_ID, "status": status, "reasons": reasons}],
        "ordered_candidates": candidates,
        "decision": decision,
        "reasons": reasons,
    }


def _input_payload() -> dict[str, Any]:
    return {
        "schema": INPUT_SCHEMA,
        "key": {"pair": "EURUSD", "timeframe": "15m", "target": "direction", "side": "DOWN"},
        "data_use": "retrospective",
        "ordered_artifact_paths": [SNAPSHOT_PATH, SELECTION_PATH, ENV_PATH],
        "issue15_terminal": ISSUE15_TERMINAL,
        "source_snapshot": _snapshot(),
        "expected_environment": EXPECTED_ENVIRONMENT,
        "protected_access": False,
        "claim_limit": CLAIM_LIMIT,
    }


def _spec(input_id: str) -> dict[str, Any]:
    return {
        "schema": "research-campaign-spec-input/v1",
        "campaign_id": CAMPAIGN_ID,
        "key": {"pair": "EURUSD", "timeframe": "15m", "target": "direction", "side": "DOWN"},
        "question": "Do issue #15 worst-half EURUSD 15m DOWN trades retain the historical threshold under the sealed local-tick settlement rule?",
        "hypothesis": "The exact historical DOWN trades retain p10 >=0.541 with at least 12/15 paths clearing 0.541.",
        "falsifier": "Tick-settled worst-half p10 below 0.541 or fewer than 12/15 paths clearing 0.541.",
        "data_use": "retrospective",
        "ordered_arms": [
            {
                "arm_id": ARM_ID,
                "role": "candidate",
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
        "stopping_rule": "Run the exact one-arm family once; stop on completion or a preregistered gate.",
        "terminal_rule": {
            "allowed_decisions": ["no_candidate", "inactive_candidate", "capability_deferred"],
            "activation": False,
        },
    }


def _prepare_and_run(raw_root: Path) -> None:
    import research_campaign_v1 as runner

    snapshot_path = ROOT / SNAPSHOT_PATH
    expected_snapshot = _snapshot()
    if snapshot_path.exists():
        if _canonical_read(snapshot_path) != expected_snapshot:
            raise AuditError("source snapshot path contains divergent bytes")
    else:
        _canonical_write(snapshot_path, expected_snapshot)
    artifacts = [
        evidence.ArtifactRef.capture(path, repo_root=ROOT)
        for path in (SNAPSHOT_PATH, SELECTION_PATH, ENV_PATH)
    ]
    input_envelope = evidence.EvidenceEnvelope.create(
        kind=INPUT_SCHEMA,
        payload=_input_payload(),
        artifacts=artifacts,
        dependencies=[ISSUE15_TERMINAL],
    )
    evidence.publish(input_envelope, repo_root=ROOT)
    spec = _spec(input_envelope.object_id)
    spec_path = ROOT / SPEC_PATH
    if spec_path.exists():
        if _canonical_read(spec_path) != spec:
            raise AuditError("campaign spec path contains divergent bytes")
    else:
        _canonical_write(spec_path, spec)
    root = raw_root
    if root.is_symlink() or not root.resolve(strict=True).is_dir():
        raise AuditError("--raw-root must resolve to a non-symlink directory")
    previous = os.environ.get(RAW_ROOT_ENV)
    os.environ[RAW_ROOT_ENV] = str(root.resolve(strict=True))
    try:
        terminal = runner.run_campaign(spec_path, repo_root=ROOT)
    finally:
        if previous is None:
            os.environ.pop(RAW_ROOT_ENV, None)
        else:
            os.environ[RAW_ROOT_ENV] = previous
    print(f"[tick-audit] terminal={terminal.object_id}")


def _verify_completed() -> None:
    import research_campaign_v1 as runner

    terminal = runner.verify_campaign(CAMPAIGN_ID, repo_root=ROOT)
    spec = _canonical_read(ROOT / SPEC_PATH)
    validate_arm_result(campaign=spec, arm=spec["ordered_arms"][0], native_result_path=ROOT / RESULT_PATH)
    print(f"[tick-audit] verified terminal={terminal.object_id}")


def _self_test() -> None:
    entries = [("a", 1, hashlib.sha256(b"x").digest()), ("b", 2, hashlib.sha256(b"y").digest())]
    assert _tree_digest(entries) != _tree_digest(list(reversed(entries)))
    ticks = __import__("numpy").array([61.0, 961.0, 962.0])
    mids = __import__("numpy").array([1.0, 0.9, 0.8])
    settled = _settle_batch([0], ticks, mids)[0]
    assert settled["valid"] and settled["down_win"] is True and settled["exit_timestamp"] == 961.0
    tie = _settle_batch([0], ticks, __import__("numpy").array([1.0, 1.0, 1.0]))[0]
    assert tie["valid"] and tie["tick_direction"] == "TIE" and tie["down_win"] is False
    late = _settle_batch([0], __import__("numpy").array([92.0, 961.0]), mids[:2])[0]
    assert not late["valid"] and late["reason"] == "entry_late"
    assert _outcome({"tick": {"worst_val_half": {"p10": 0.55, "frac_clear_breakeven": 0.8}}}, True) == "historical_tick_settlement_not_falsified"
    assert _outcome({"tick": {"worst_val_half": {"p10": 0.5409, "frac_clear_breakeven": 1.0}}}, True) == "historical_tick_settlement_falsified"
    sample = _empty_result(_spec("0" * 64)["ordered_arms"][0], "synthetic", {})
    sample["unknown"] = True
    try:
        _exact(sample, {
            "schema", "campaign_id", "arm_id", "input_evidence_ids", "status", "reasons",
            "protected_access", "claim_limit", "measurement", "environment", "source",
            "settlements", "path_results", "summaries", "frozen_forward", "outcome",
        }, "synthetic")
    except AuditError:
        pass
    else:
        raise AssertionError("strict schema accepted an unknown field")
    assert _spec("0" * 64)["terminal_rule"]["activation"] is False
    assert _snapshot()["tree_sha256"] == SOURCE_TREE_SHA256
    print("[tick-audit] self-test PASS")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("self-test", "run", "verify"))
    parser.add_argument("--raw-root", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "self-test":
        if args.raw_root is not None:
            raise SystemExit("--raw-root is only valid with run")
        _self_test()
    elif args.command == "run":
        if args.raw_root is None:
            raise SystemExit("run requires --raw-root")
        _prepare_and_run(args.raw_root)
    else:
        if args.raw_root is not None:
            raise SystemExit("verify does not read the raw root")
        _verify_completed()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
