#!/usr/bin/env python3
"""Generate outcome-blind 2026-Q1 phase-zero behavior authorities.

The default ``diagnostic`` mode preserves the stopped USDJPY/USDCAD
investigation and its non-acceptance artifact.  ``--mode acceptance`` is the
separate final six-target gate: it rebuilds the exact seven-source live feature
universe (including source-only USDCAD), rebuilds EURUSD order flow from strict
processed Q1 OHLCV, and verifies the six incumbent books plus five enabled live
providers.  Neither mode projects a label, forward return, validity flag,
settlement, or replay outcome from a parquet source.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import book_runtime
import live_features
import m15_book_refresh as refresh
import orderflow
import pipeline
from m15_book_refresh_adapters import (
    ScoreRows,
    _validate_score_feature_names,
    build_score_rows,
    slice_score_rows,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_ROOT = Path("/home/sean/git/processed")
OUTPUT_PATH = (
    REPO_ROOT
    / "results/json/m15_book_refresh_2026q1_phase_zero_behavior_authority.json"
)
# The diagnostic above remains the immutable USDJPY/USDCAD investigation.  The
# campaign acceptance authority is deliberately a separate schema and path so
# a diagnostic (including a stopped diagnostic) can never satisfy the runner's
# phase-zero gate by filename substitution.  The version suffix preserves the
# immutable prior authorities from outcome-blind launches that stopped before
# fitting or semantic replay (first on Parquet projection, then provider
# parity on non-binding NaN sign bits).
ACCEPTANCE_OUTPUT_PATH = (
    REPO_ROOT
    / "results/json/m15_book_refresh_2026q1_phase_zero_acceptance_v3.json"
)
PAIRS = ("USDJPY", "USDCAD")
ACCEPTANCE_TARGET_PAIRS = (
    "EURUSD",
    "USDJPY",
    "GBPUSD",
    "USDCHF",
    "AUDUSD",
    "NZDUSD",
)
FEATURE_SOURCE_PAIRS = (
    "EURUSD",
    "GBPUSD",
    "AUDUSD",
    "NZDUSD",
    "USDJPY",
    "USDCHF",
    "USDCAD",
)
START_DECISION = pd.Timestamp("2026-01-01T00:00:00Z")
END_DECISION = pd.Timestamp("2026-04-01T00:00:00Z")
DECISION_SHIFT_NS = np.int64(60 * 1_000_000_000)
RAW_BAR_NS = np.int64(10 * 1_000_000_000)
RAW_VALUE_COLUMNS = ("open", "high", "low", "close", "volume")
RAW_SCHEMA_COLUMNS = (*RAW_VALUE_COLUMNS, "datetime_utc")
BASELINE_COMMIT = "b74a549f45bddfd6310d80c1f6c130de0076a95f"
BASELINE_LIVE_FEATURES_BLOB_SHA1 = "bb55b4f17bd6e4f514435624bec2117a92ae7dad"
BASELINE_LIVE_FEATURES_SHA256 = (
    "4903217812213cbeb6902a02552f7bddcc741fa665e8e700aee10c2b1716feb4"
)
FLOAT32_LIVE_FEATURES_SHA256 = (
    "c485766d2dda85339938d59180056287ee5826ad275ff4d988588bdc2d201198"
)
PINNED_ACCEPTANCE_RUNTIME_VERSIONS = {
    "numpy": "2.4.6",
    "pandas": "3.0.3",
    "pyarrow": "24.0.0",
    "lightgbm": "4.6.0",
    "scikit-learn": "1.8.0",
}


class AuthorityError(RuntimeError):
    """The outcome-blind authority cannot be generated exactly."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repo_path(path: Path) -> str:
    try:
        return path.resolve(strict=True).relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path.resolve(strict=True))


def _datetime_index_ns(index: pd.DatetimeIndex, *, name: str) -> np.ndarray:
    """Return semantic UTC nanoseconds independent of pandas index resolution."""

    if not isinstance(index, pd.DatetimeIndex):
        raise AuthorityError(f"{name}: clock must be a pandas DatetimeIndex")
    if index.tz is None or index.hasnans:
        raise AuthorityError(f"{name}: clock must be non-null and timezone-aware")
    try:
        normalized = index.tz_convert("UTC").as_unit("ns")
    except (OverflowError, TypeError, ValueError) as exc:
        raise AuthorityError(f"{name}: clock cannot be represented exactly in ns") from exc
    if str(normalized.dtype) != "datetime64[ns, UTC]":
        raise AuthorityError(f"{name}: normalized clock unit is not UTC nanoseconds")
    values = np.ascontiguousarray(normalized.asi8, dtype="int64")
    reconstructed = pd.DatetimeIndex(
        pd.to_datetime(values, unit="ns", utc=True)
    ).as_unit("ns")
    if not reconstructed.equals(normalized):
        raise AuthorityError(f"{name}: nanosecond clock round-trip differs")
    return values


def _file_identity(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    if path.is_symlink() or not resolved.is_file():
        raise AuthorityError(f"input is not a regular file: {path}")
    return {
        "path": _repo_path(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": _sha256_file(resolved),
    }


def _implementation_identities() -> dict[str, str]:
    paths = tuple(Path(value) for value in refresh.source_files())
    if not paths:
        raise AuthorityError("refresh source-file authority is empty")
    repo_root = REPO_ROOT.resolve(strict=True)
    output: dict[str, str] = {}
    for path in paths:
        try:
            relative = path.resolve(strict=True).relative_to(repo_root).as_posix()
        except (OSError, ValueError) as exc:
            raise AuthorityError(
                f"refresh source-file authority escapes repository: {path}"
            ) from exc
        if path.is_symlink() or not path.is_file() or relative in output:
            raise AuthorityError(
                f"refresh source-file authority is missing/duplicated: {path}"
            )
        output[relative] = _sha256_file(path)
    return output


def _verified_acceptance_runtime() -> tuple[dict[str, str], dict[str, Any]]:
    try:
        contract = refresh.verified_runtime_version_contract()
    except refresh.RefreshError as exc:
        raise AuthorityError(str(exc)) from exc
    actual = dict(contract["actual_exact"])
    if actual != PINNED_ACCEPTANCE_RUNTIME_VERSIONS:
        differences = {
            name: {
                "expected": PINNED_ACCEPTANCE_RUNTIME_VERSIONS[name],
                "actual": actual[name],
            }
            for name in PINNED_ACCEPTANCE_RUNTIME_VERSIONS
            if actual[name] != PINNED_ACCEPTANCE_RUNTIME_VERSIONS[name]
        }
        raise AuthorityError(
            "acceptance runtime differs from the exact measurement contract: "
            f"{differences}"
        )
    return actual, contract


def _verify_live_feature_counterfactual_sources(
    implementation: dict[str, str],
) -> dict[str, Any]:
    current = implementation["scripts/live_features.py"]
    if current != FLOAT32_LIVE_FEATURES_SHA256:
        raise AuthorityError(
            "live_features.py no longer has the diagnosed float32 provider semantics"
        )
    baseline = subprocess.run(
        ["git", "show", f"{BASELINE_COMMIT}:scripts/live_features.py"],
        cwd=REPO_ROOT,
        check=True,
        stdout=subprocess.PIPE,
    ).stdout
    baseline_sha = _sha256_bytes(baseline)
    blob_sha = subprocess.run(
        ["git", "rev-parse", f"{BASELINE_COMMIT}:scripts/live_features.py"],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()
    if (
        baseline_sha != BASELINE_LIVE_FEATURES_SHA256
        or blob_sha != BASELINE_LIVE_FEATURES_BLOB_SHA1
    ):
        raise AuthorityError("unmodified-HEAD live provider counterfactual differs")
    current_bytes = (REPO_ROOT / "scripts/live_features.py").read_bytes()
    baseline_expression = b"closes[p].astype(float).values"
    current_expression = b'closes[p].to_numpy(dtype="float32", copy=False)'
    if baseline.count(baseline_expression) != 1 or current_bytes.count(current_expression) != 1:
        raise AuthorityError("live provider dtype source expressions differ")
    return {
        "production_provider_semantics_changed": True,
        "verification": (
            "exact baseline/current source hashes plus one exact close-conversion "
            "expression in each source"
        ),
        "baseline_unmodified_head": {
            "commit": BASELINE_COMMIT,
            "path": "scripts/live_features.py",
            "git_blob_sha1": blob_sha,
            "sha256": baseline_sha,
            "log_return_close_dtype": "float64",
        },
        "working_tree_provider": {
            "path": "scripts/live_features.py",
            "sha256": current,
            "log_return_close_dtype": "float32",
        },
        "source_semantic_delta": {
            "baseline_close_conversion": "Python float / NumPy float64",
            "working_tree_close_conversion": "NumPy float32",
            "behavioral_claims": "none_in_this_source_only_record",
        },
    }


def _processed_paths(pair: str) -> list[Path]:
    root = PROCESSED_ROOT / pair
    pattern = re.compile(rf"^{pair}_10s_(2026-\d{{2}}-\d{{2}})\.parquet$")
    paths: list[Path] = []
    for path in sorted(root.glob(f"{pair}_10s_2026-*.parquet")):
        match = pattern.fullmatch(path.name)
        if match is None:
            raise AuthorityError(f"malformed processed filename: {path}")
        day = pd.Timestamp(match.group(1), tz="UTC")
        if START_DECISION.normalize() <= day < END_DECISION.normalize():
            paths.append(path)
    if not paths:
        raise AuthorityError(f"{pair}: no exact 2026-Q1 processed inputs")
    return paths


def _book_input_identities(book: book_runtime.LoadedBook) -> dict[str, Any]:
    index = book_runtime.load_index()
    manifest_path, _ = book_runtime.load_registered_manifest(
        book.book_id, index[book.book_id]
    )
    return {
        "book_id": book.book_id,
        "content_id": book.content_id,
        "ordered_files": [
            _file_identity(REPO_ROOT / "books/INDEX.json"),
            _file_identity(manifest_path),
            _file_identity(book.strategy_path),
            *[_file_identity(path) for path in book.model_paths],
        ],
        "model_order": [_repo_path(path) for path in book.model_paths],
    }


def _input_identities(
    pair: str, book: book_runtime.LoadedBook, processed_paths: Sequence[Path]
) -> dict[str, Any]:
    cached = REPO_ROOT / "features" / f"{pair}_2026.parquet"
    return {
        "cached_feature_file": _file_identity(cached),
        "cached_parquet_projection_in_order": list(book.feature_cols),
        "processed_parquet_projection_in_order": list(RAW_SCHEMA_COLUMNS),
        "processed_parquet_required_schema_in_order": list(RAW_SCHEMA_COLUMNS),
        "forbidden_semantic_columns_read": [],
        "processed_raw_files_in_loader_order": [
            _file_identity(path) for path in processed_paths
        ],
        "cross_pair_inputs_actually_used": [],
        "cross_pair_inputs_used": False,
        "incumbent_book": _book_input_identities(book),
    }


def _raw_schema_fields(path: Path) -> list[dict[str, Any]]:
    schema = pq.ParquetFile(path).schema_arrow
    names = tuple(schema.names)
    if names != RAW_SCHEMA_COLUMNS:
        raise AuthorityError(
            f"{path}: processed schema must be exactly {RAW_SCHEMA_COLUMNS}, got {names}"
        )
    fields = [
        {"name": field.name, "type": str(field.type), "nullable": field.nullable}
        for field in schema
    ]
    expected_types = ["double", "double", "double", "double", "double", "timestamp[ns, tz=UTC]"]
    if [field["type"] for field in fields] != expected_types:
        raise AuthorityError(f"{path}: processed physical types differ: {fields}")
    return fields


def _clock_seal(values: np.ndarray, *, name: str) -> dict[str, Any]:
    clock = np.asarray(values, dtype="int64")
    if clock.ndim != 1 or len(clock) == 0:
        raise AuthorityError(f"{name}: clock must be a nonempty vector")
    if np.any(clock % RAW_BAR_NS != 0):
        raise AuthorityError(f"{name}: timestamps are not on the exact 10-second lattice")
    gaps = np.diff(clock)
    if np.any(gaps <= 0):
        raise AuthorityError(f"{name}: timestamps are unsorted or duplicated")
    if np.any(gaps % RAW_BAR_NS != 0):
        raise AuthorityError(f"{name}: timestamp gaps are not multiples of 10 seconds")
    return {
        "rows": len(clock),
        "first_utc": _timestamps(clock[:1])[0],
        "last_utc": _timestamps(clock[-1:])[0],
        "clock_i64le_sha256": _array_sha(clock, "<i8"),
        "strictly_increasing": True,
        "unique": True,
        "ten_second_lattice_exact": True,
        "all_gaps_positive_multiples_of_10s": True,
        "gap_gt_10s_count": int((gaps > RAW_BAR_NS).sum()),
        "minimum_gap_seconds": int(gaps.min() // 1_000_000_000) if len(gaps) else None,
        "maximum_gap_seconds": int(gaps.max() // 1_000_000_000) if len(gaps) else None,
    }


def _load_processed_q1_strict(
    pair: str, paths: Sequence[Path]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Project only causal OHLCV and reject source-clock defects without repair."""

    parts: list[pd.DataFrame] = []
    files: list[dict[str, Any]] = []
    previous_last: int | None = None
    for path in paths:
        fields = _raw_schema_fields(path)
        # Schema is rejected before this projection. No other physical column can
        # enter memory, including a label/outcome accidentally added to a source.
        table = pq.read_table(path, columns=list(RAW_SCHEMA_COLUMNS))
        if tuple(table.column_names) != RAW_SCHEMA_COLUMNS:
            raise AuthorityError(f"{path}: projected processed column order differs")
        frame = table.to_pandas(ignore_metadata=True)
        if tuple(frame.columns) != RAW_SCHEMA_COLUMNS:
            raise AuthorityError(f"{path}: decoded processed column order differs")
        timestamp = pd.DatetimeIndex(frame.pop("datetime_utc"))
        if timestamp.tz is None or str(timestamp.tz) != "UTC" or timestamp.hasnans:
            raise AuthorityError(f"{path}: processed timestamp must be non-null UTC")
        if tuple(frame.columns) != RAW_VALUE_COLUMNS:
            raise AuthorityError(f"{path}: projected OHLCV order differs")
        values = frame.loc[:, list(RAW_VALUE_COLUMNS)].to_numpy(
            dtype="float64", copy=False
        )
        if not np.isfinite(values).all():
            raise AuthorityError(f"{path}: processed OHLCV contains non-finite values")
        clock = _datetime_index_ns(
            timestamp, name=f"{pair}:{path.name} processed timestamp"
        )
        clock_report = _clock_seal(clock, name=f"{pair}:{path.name}")
        if previous_last is not None and int(clock[0]) <= previous_last:
            raise AuthorityError(
                f"{pair}: processed files overlap or are out of order at {path.name}"
            )
        previous_last = int(clock[-1])
        frame.index = timestamp
        parts.append(frame)
        files.append(
            {
                "path": _repo_path(path),
                "schema_in_order": fields,
                **clock_report,
            }
        )
    if not parts:
        raise AuthorityError(f"{pair}: strict processed loader returned no rows")
    raw = pd.concat(parts, axis=0, copy=False)
    global_clock = _datetime_index_ns(
        raw.index, name=f"{pair}:global processed clock"
    )
    global_report = _clock_seal(global_clock, name=f"{pair}:global processed clock")
    return raw, {
        "loader": "m15_phase_zero_behavior_authority._load_processed_q1_strict",
        "projection_in_order": list(RAW_SCHEMA_COLUMNS),
        "schema_policy": "exact_schema_only_no_extra_columns",
        "sort_or_dedup_performed": False,
        "semantic_or_outcome_columns_read": False,
        "ordered_files": files,
        "global_clock": global_report,
    }


def _raw_feature_frame(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    m1 = pipeline.resample_1m(raw)
    frame = pipeline.build_features(m1).replace([np.inf, -np.inf], np.nan)
    for column in frame.columns:
        if frame[column].dtype == np.dtype("float64"):
            frame[column] = frame[column].astype("float32")
    return m1, frame


def _resampled_close_availability_report(
    raw: pd.DataFrame, m1: pd.DataFrame
) -> dict[str, Any]:
    """Prove the last 10-second close used by each minute is available by decision."""

    raw_ns = _datetime_index_ns(raw.index, name="resampled-close raw clock")
    minute_ns = (raw_ns // DECISION_SHIFT_NS) * DECISION_SHIFT_NS
    last_positions = np.r_[
        np.flatnonzero(minute_ns[:-1] != minute_ns[1:]), len(minute_ns) - 1
    ]
    source_ns = minute_ns[last_positions]
    last_raw_label_ns = raw_ns[last_positions]
    m1_source_ns = _datetime_index_ns(
        m1.index, name="resampled-close minute clock"
    )
    if not np.array_equal(source_ns, m1_source_ns):
        raise AuthorityError("resampled minute clock differs from strict raw grouping")
    raw_close = np.ascontiguousarray(
        raw["close"].to_numpy(dtype="float64", copy=False)[last_positions]
    )
    m1_close = np.ascontiguousarray(m1["close"].to_numpy(dtype="float64", copy=False))
    if not np.array_equal(raw_close.view("uint64"), m1_close.view("uint64")):
        raise AuthorityError("resampled close is not the last strict raw close")
    availability_ns = last_raw_label_ns + RAW_BAR_NS
    decision_ns = source_ns + DECISION_SHIFT_NS
    if np.any(last_raw_label_ns < source_ns) or np.any(availability_ns > decision_ns):
        raise AuthorityError("a resampled close is not causally available by decision")
    slack_ns = decision_ns - availability_ns
    return {
        "rows": len(source_ns),
        "raw_bar_label_semantics": "left_edge_of_[label,label+10s)_bar",
        "close_availability_mapping": "raw_bar_label_plus_10_seconds",
        "decision_mapping": "minute_source_label_plus_60_seconds",
        "every_close_available_no_later_than_decision": True,
        "minute_source_clock_i64le_sha256": _array_sha(source_ns, "<i8"),
        "last_raw_label_i64le_sha256": _array_sha(last_raw_label_ns, "<i8"),
        "close_availability_i64le_sha256": _array_sha(availability_ns, "<i8"),
        "decision_clock_i64le_sha256": _array_sha(decision_ns, "<i8"),
        "resampled_close_f64le_sha256": _array_sha(m1_close, "<f8"),
        "minimum_availability_slack_seconds": int(slack_ns.min() // 1_000_000_000),
        "maximum_availability_slack_seconds": int(slack_ns.max() // 1_000_000_000),
    }


def _window_matrix(
    frame: pd.DataFrame, columns: Sequence[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.tz is None:
        raise AuthorityError("feature frame index must be timezone-aware")
    source_ns = _datetime_index_ns(frame.index, name="feature-frame source clock")
    decision_ns = source_ns + DECISION_SHIFT_NS
    mask = (
        (decision_ns >= np.int64(START_DECISION.value))
        & (decision_ns < np.int64(END_DECISION.value))
    )
    positions = np.flatnonzero(mask)
    matrix = np.ascontiguousarray(
        frame.iloc[positions].loc[:, list(columns)].to_numpy(
            dtype="float32", copy=True
        )
    )
    return matrix, source_ns[positions], decision_ns[positions]


def _read_cached_q1_covariates(
    path: Path, columns: Sequence[str]
) -> pd.DataFrame:
    """Decode only the causal pre-April feature window and declared columns."""

    projection = tuple(columns)
    if not projection or len(projection) != len(set(projection)):
        raise AuthorityError("cached covariate projection is empty or duplicated")
    _validate_score_feature_names(projection, pair=path.stem.split("_")[0])
    lower_source = START_DECISION - pd.Timedelta(seconds=60)
    upper_source = END_DECISION - pd.Timedelta(seconds=60)
    frame = pd.read_parquet(
        path,
        columns=list(projection),
        filters=[
            ("datetime_utc", ">=", lower_source.to_pydatetime()),
            ("datetime_utc", "<", upper_source.to_pydatetime()),
        ],
    )
    if tuple(frame.columns) != projection:
        raise AuthorityError(f"{path}: cached covariate projection order differs")
    if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.tz is None:
        raise AuthorityError(f"{path}: cached feature clock must be timezone-aware")
    source_ns = _datetime_index_ns(
        frame.index, name=f"{path}:cached feature clock"
    )
    if len(source_ns) == 0:
        raise AuthorityError(f"{path}: cached Q1 covariate projection is empty")
    if np.any(source_ns < lower_source.value) or np.any(source_ns >= upper_source.value):
        raise AuthorityError(f"{path}: cached projection escaped the causal Q1 window")
    return frame


def _array_sha(values: np.ndarray, dtype: str) -> str:
    return _sha256_bytes(
        np.asarray(values, dtype=np.dtype(dtype)).tobytes(order="C")
    )


def _timestamps(values: np.ndarray) -> list[str]:
    return [
        pd.Timestamp(int(value), unit="ns", tz="UTC")
        .isoformat()
        .replace("+00:00", "Z")
        for value in values
    ]


def _scalar_probe_exact(
    book: book_runtime.LoadedBook, matrix: np.ndarray, probabilities: np.ndarray
) -> tuple[int, bool]:
    count = min(257, len(probabilities))
    ranks = np.unique(
        np.floor(np.linspace(0, len(probabilities) - 1, count)).astype("int64")
    )
    exact = True
    for rank in ranks:
        scalar = book.score(
            pd.Series(matrix[rank], index=book.feature_cols, dtype="float64")
        )
        exact &= bool(
            np.float64(scalar.proba).view("uint64")
            == probabilities[rank].view("uint64")
        )
    return len(ranks), exact


def _full_selection(
    book: book_runtime.LoadedBook,
    matrix: np.ndarray,
    decision_ns: np.ndarray,
    eligible: np.ndarray,
) -> tuple[
    np.ndarray,
    book_runtime.BookGateComponents,
    np.ndarray,
]:
    frame = pd.DataFrame(matrix[eligible], columns=book.feature_cols)
    probabilities = book.predict_probabilities(frame)
    components = book.gate_components(frame, probabilities)
    return probabilities, components, decision_ns[eligible][components.gate_passed]


def _cached_only_scores(
    book: book_runtime.LoadedBook,
    matrix: np.ndarray,
    decision_ns: np.ndarray,
    positions: np.ndarray,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for position in positions:
        feature_row = pd.Series(
            matrix[position], index=book.feature_cols, dtype="float64"
        )
        score = book.score(feature_row)
        structural_reasons = [
            reason
            for reason in score.gate_reasons
            if not reason.startswith("confidence ")
        ]
        rows.append(
            {
                "decision_utc": _timestamps(
                    np.asarray([decision_ns[position]], dtype="int64")
                )[0],
                "proba": score.proba,
                "proba_f64_bits_hex": f"{np.float64(score.proba).view('uint64'):016x}",
                "direction": score.direction,
                "confidence": score.confidence,
                "confidence_threshold": score.threshold,
                "confidence_gate_passed": score.confidence >= score.threshold,
                "structural_gate_passed": not structural_reasons,
                "structural_gate_reasons": structural_reasons,
                "combined_gate_passed": score.gate_passed,
                "gate_reasons": score.gate_reasons,
                "cached_1m_autocorr_10": float(feature_row["1m_autocorr_10"]),
                "raw_rebuilt_1m_autocorr_10": "NaN",
            }
        )
    selected = [row["decision_utc"] for row in rows if row["combined_gate_passed"]]
    return {
        "population": "cached_finite_and_raw_rebuilt_nonfinite_rows_only",
        "rows": len(rows),
        "scorer": "book_runtime.LoadedBook.score",
        "direction_counts": dict(sorted(Counter(row["direction"] for row in rows).items())),
        "confidence_gate_pass_count": sum(
            row["confidence_gate_passed"] for row in rows
        ),
        "structural_gate_pass_count": sum(
            row["structural_gate_passed"] for row in rows
        ),
        "combined_gate_pass_count": len(selected),
        "selected_timestamps_utc": selected,
        "row_scores": rows,
    }


def _live_provider_report(
    pair: str,
    book: book_runtime.LoadedBook,
    m1: pd.DataFrame,
    matrix: np.ndarray,
    source_ns: np.ndarray,
    decision_ns: np.ndarray,
    directory: Path,
) -> dict[str, Any]:
    live_store = directory / "live_store"
    live_store.mkdir()
    live_m1 = m1.loc[m1.index + pd.Timedelta(seconds=60) < END_DECISION]
    live_m1.to_parquet(live_store / f"{pair}.parquet")
    builder = live_features.LiveFeatureBuilder(
        client=None,
        history_minutes=len(live_m1) + 1,
        tick_volume_count=0,
        store_dir=live_store,
        stale_seconds=10**9,
    )
    score_rows = ScoreRows(
        pair=pair,
        feature_cols=tuple(book.feature_cols),
        X=matrix,
        entry_ns=decision_ns,
        source_feature_ns=source_ns,
    )
    report = refresh._live_feature_builder_parity_report(
        pair, book.feature_cols, score_rows, builder
    )
    report["derived_rolling_store"] = {
        "rows": len(live_m1),
        "source_clock_i64le_sha256": _array_sha(
            _datetime_index_ns(
                live_m1.index, name=f"{pair}:diagnostic rolling-store clock"
            ),
            "<i8",
        ),
        "sha256": _sha256_file(live_store / f"{pair}.parquet"),
    }
    return report


def _behavior_verdict(checks: dict[str, bool]) -> str:
    required = {
        "finite_eligibility_exact",
        "common_probability_bitwise_exact",
        "common_direction_exact",
        "loadedbook_scalar_probes_exact",
        "common_gate_components_exact",
        "common_ordered_pre_schedule_exact",
        "full_population_ordered_pre_schedule_exact",
        "live_feature_builder_exact",
    }
    if set(checks) != required or any(type(value) is not bool for value in checks.values()):
        raise AuthorityError("behavior verdict checks are malformed")
    if not checks["finite_eligibility_exact"]:
        return "STOP_PHASE_ZERO_SCOREABILITY_MISMATCH"
    if all(checks.values()):
        return "PASS_BEHAVIOR_AUTHORITY"
    return "STOP_PHASE_ZERO_BEHAVIOR_MISMATCH"


def _analyze_pair(pair: str, book: book_runtime.LoadedBook) -> dict[str, Any]:
    paths = _processed_paths(pair)
    inputs_before = _input_identities(pair, book, paths)
    cached_path = REPO_ROOT / "features" / f"{pair}_2026.parquet"
    # Projection is binding: no y/fwd_ret/valid/close/settlement-like field is read.
    cached_projection = tuple(book.feature_cols)
    if len(set(cached_projection)) != len(cached_projection):
        raise AuthorityError(f"{pair}: cached covariate projection contains duplicates")
    _validate_score_feature_names(cached_projection, pair=pair)
    cached_frame = _read_cached_q1_covariates(cached_path, cached_projection)
    with tempfile.TemporaryDirectory(prefix=f"m15-authority-{pair.lower()}-") as name:
        temporary = Path(name)
        raw, raw_loader_report = _load_processed_q1_strict(pair, paths)
        m1, raw_frame = _raw_feature_frame(raw)
        close_availability_report = _resampled_close_availability_report(raw, m1)
        cached_X, cached_source_ns, cached_decision_ns = _window_matrix(
            cached_frame, book.feature_cols
        )
        raw_X, raw_source_ns, raw_decision_ns = _window_matrix(
            raw_frame, book.feature_cols
        )
        if not (
            np.array_equal(cached_source_ns, raw_source_ns)
            and np.array_equal(cached_decision_ns, raw_decision_ns)
        ):
            raise AuthorityError(f"{pair}: cached/raw feature clocks differ")
        if not np.array_equal(
            cached_decision_ns, cached_source_ns + DECISION_SHIFT_NS
        ):
            raise AuthorityError(f"{pair}: decision clock is not source +60s")

        cached_eligible = np.isfinite(cached_X).all(axis=1)
        raw_eligible = np.isfinite(raw_X).all(axis=1)
        common = cached_eligible & raw_eligible
        mismatch = cached_eligible != raw_eligible
        cached_only = cached_eligible & ~raw_eligible

        cached_common_frame = pd.DataFrame(
            cached_X[common], columns=book.feature_cols
        )
        raw_common_frame = pd.DataFrame(
            raw_X[common], columns=book.feature_cols
        )
        cached_common_probability = book.predict_probabilities(
            cached_common_frame
        )
        raw_common_probability = book.predict_probabilities(raw_common_frame)
        common_probability_exact = np.array_equal(
            cached_common_probability.view("uint64"),
            raw_common_probability.view("uint64"),
        )
        cached_common_components = book.gate_components(
            cached_common_frame, cached_common_probability
        )
        raw_common_components = book.gate_components(
            raw_common_frame, raw_common_probability
        )
        cached_common_gate = cached_common_components.gate_passed
        raw_common_gate = raw_common_components.gate_passed
        common_decisions = cached_decision_ns[common]
        cached_common_selected = common_decisions[cached_common_gate]
        raw_common_selected = common_decisions[raw_common_gate]

        cached_full_probability, cached_full_components, cached_full_selected = (
            _full_selection(
                book, cached_X, cached_decision_ns, cached_eligible
            )
        )
        raw_full_probability, raw_full_components, raw_full_selected = (
            _full_selection(book, raw_X, raw_decision_ns, raw_eligible)
        )
        cached_full_gate = cached_full_components.gate_passed
        raw_full_gate = raw_full_components.gate_passed
        cached_full_set = set(int(value) for value in cached_full_selected)
        raw_full_set = set(int(value) for value in raw_full_selected)

        cached_probe_rows, cached_probe_exact = _scalar_probe_exact(
            book, cached_X[common], cached_common_probability
        )
        raw_probe_rows, raw_probe_exact = _scalar_probe_exact(
            book, raw_X[common], raw_common_probability
        )
        diagnostics = refresh._decoded_diagnostics(
            book.feature_cols, cached_X, raw_X
        )
        live_report = _live_provider_report(
            pair,
            book,
            m1,
            raw_X,
            raw_source_ns,
            raw_decision_ns,
            temporary,
        )

        behavior_checks = {
            "finite_eligibility_exact": bool(not mismatch.any()),
            "common_probability_bitwise_exact": bool(common_probability_exact),
            "common_direction_exact": bool(
                np.array_equal(
                    cached_common_probability >= 0.5,
                    raw_common_probability >= 0.5,
                )
            ),
            "loadedbook_scalar_probes_exact": bool(
                cached_probe_rows > 0
                and cached_probe_rows == raw_probe_rows
                and cached_probe_exact
                and raw_probe_exact
            ),
            "common_gate_components_exact": bool(
                np.array_equal(
                    cached_common_components.confidence_passed,
                    raw_common_components.confidence_passed,
                )
                and np.array_equal(
                    cached_common_components.structural_passed,
                    raw_common_components.structural_passed,
                )
            ),
            "common_ordered_pre_schedule_exact": bool(
                np.array_equal(cached_common_selected, raw_common_selected)
            ),
            "full_population_ordered_pre_schedule_exact": bool(
                np.array_equal(cached_full_selected, raw_full_selected)
            ),
            "live_feature_builder_exact": bool(
                live_report.get("behavior_all_equal") is True
            ),
        }

        mismatch_positions = np.flatnonzero(mismatch)
        mismatch_columns = sorted(
            {
                book.feature_cols[column]
                for row in mismatch_positions
                for column in np.flatnonzero(
                    np.isfinite(cached_X[row]) != np.isfinite(raw_X[row])
                )
            }
        )
        mismatch_decisions = cached_decision_ns[mismatch]
        scoreability_source = {
            "columns": mismatch_columns,
            "rows": len(mismatch_positions),
            "cached_finite_raw_nonfinite": int(cached_only.sum()),
            "cached_nonfinite_raw_finite": int(
                ((~cached_eligible) & raw_eligible).sum()
            ),
            "first_decision_utc": (
                _timestamps(mismatch_decisions[:1])[0]
                if len(mismatch_decisions)
                else None
            ),
            "last_decision_utc": (
                _timestamps(mismatch_decisions[-1:])[0]
                if len(mismatch_decisions)
                else None
            ),
            "consecutive_one_minute_decisions": bool(
                len(mismatch_decisions) < 2
                or np.all(np.diff(mismatch_decisions) == DECISION_SHIFT_NS)
            ),
            "decision_i64le_sha256": _array_sha(mismatch_decisions, "<i8"),
        }
        cached_only_report = _cached_only_scores(
            book,
            cached_X,
            cached_decision_ns,
            np.flatnonzero(cached_only),
        )
        raw_observation_clock = _datetime_index_ns(
            raw.index, name=f"{pair}:diagnostic raw observation clock"
        )
        m1_clock = _datetime_index_ns(
            m1.index, name=f"{pair}:diagnostic minute clock"
        )
        m1_values = np.ascontiguousarray(
            m1.loc[:, ["open", "high", "low", "close", "volume", "gap_prev"]]
            .to_numpy(dtype="float64", copy=True)
        )

    inputs_after = _input_identities(pair, book, paths)
    if inputs_before != inputs_after:
        raise AuthorityError(f"{pair}: an input changed while authority was generated")

    full_symmetric = sorted(cached_full_set ^ raw_full_set)
    report = {
        "verdict": _behavior_verdict(behavior_checks),
        "verdict_checks": behavior_checks,
        "inputs": inputs_before,
        "strict_processed_source": raw_loader_report,
        "causal_resampled_close_availability": close_availability_report,
        "processed_raw_derived_identity": {
            "observation_rows": len(raw_observation_clock),
            "observation_clock_i64le_sha256": _array_sha(
                raw_observation_clock, "<i8"
            ),
            "one_minute_rows": len(m1_clock),
            "one_minute_clock_i64le_sha256": _array_sha(m1_clock, "<i8"),
            "one_minute_ohlcv_gap_f64le_sha256": _array_sha(m1_values, "<f8"),
        },
        "rows": len(cached_decision_ns),
        "source_feature_clock_i64le_sha256": _array_sha(
            cached_source_ns, "<i8"
        ),
        "decision_clock_i64le_sha256": _array_sha(cached_decision_ns, "<i8"),
        "ordered_float32_features": {
            "cached_sha256": _array_sha(cached_X, "<f4"),
            "raw_rebuilt_sha256": _array_sha(raw_X, "<f4"),
            "bitwise_equal": bool(
                np.array_equal(cached_X.view("uint32"), raw_X.view("uint32"))
            ),
            "behavior_neutral_decoded_drift": bool(
                all(behavior_checks.values())
            ),
            "decoded_diagnostics": diagnostics,
        },
        "finite_eligibility": {
            "cached_count": int(cached_eligible.sum()),
            "raw_rebuilt_count": int(raw_eligible.sum()),
            "common_count": int(common.sum()),
            "mismatch_count": int(mismatch.sum()),
            "cached_u8_sha256": _array_sha(cached_eligible, "u1"),
            "raw_rebuilt_u8_sha256": _array_sha(raw_eligible, "u1"),
            "cached_finite_cells_u8_sha256": _array_sha(
                np.isfinite(cached_X), "u1"
            ),
            "raw_rebuilt_finite_cells_u8_sha256": _array_sha(
                np.isfinite(raw_X), "u1"
            ),
        },
        "scoreability_mismatch_source": scoreability_source,
        "common_loadedbook_probability": {
            "population": "common_finite_rows_only",
            "scorer": "book_runtime.LoadedBook.predict_probabilities",
            "rows": int(common.sum()),
            "bit_mismatch_count": int(
                (
                    cached_common_probability.view("uint64")
                    != raw_common_probability.view("uint64")
                ).sum()
            ),
            "direction_mismatch_count": int(
                (
                    (cached_common_probability >= 0.5)
                    != (raw_common_probability >= 0.5)
                ).sum()
            ),
            "max_abs_difference": float(
                np.max(
                    np.abs(
                        cached_common_probability - raw_common_probability
                    )
                )
            ),
            "cached_f64le_sha256": _array_sha(
                cached_common_probability, "<f8"
            ),
            "raw_rebuilt_f64le_sha256": _array_sha(
                raw_common_probability, "<f8"
            ),
            "cached_scalar_probe_rows": cached_probe_rows,
            "cached_scalar_exact": cached_probe_exact,
            "raw_rebuilt_scalar_probe_rows": raw_probe_rows,
            "raw_rebuilt_scalar_exact": raw_probe_exact,
        },
        "gate_components": {
            "population": "common_finite_rows_only",
            "confidence": {
                "cached_pass_count": int(
                    cached_common_components.confidence_passed.sum()
                ),
                "raw_rebuilt_pass_count": int(
                    raw_common_components.confidence_passed.sum()
                ),
                "mismatch_count": int(
                    (
                        cached_common_components.confidence_passed
                        != raw_common_components.confidence_passed
                    ).sum()
                ),
                "cached_u8_sha256": _array_sha(
                    cached_common_components.confidence_passed, "u1"
                ),
                "raw_rebuilt_u8_sha256": _array_sha(
                    raw_common_components.confidence_passed, "u1"
                ),
            },
            "structural": {
                "cached_pass_count": int(
                    cached_common_components.structural_passed.sum()
                ),
                "raw_rebuilt_pass_count": int(
                    raw_common_components.structural_passed.sum()
                ),
                "mismatch_count": int(
                    (
                        cached_common_components.structural_passed
                        != raw_common_components.structural_passed
                    ).sum()
                ),
                "cached_u8_sha256": _array_sha(
                    cached_common_components.structural_passed, "u1"
                ),
                "raw_rebuilt_u8_sha256": _array_sha(
                    raw_common_components.structural_passed, "u1"
                ),
            },
            "combined": {
                "cached_pass_count": int(cached_common_gate.sum()),
                "raw_rebuilt_pass_count": int(raw_common_gate.sum()),
                "mismatch_count": int(
                    (cached_common_gate != raw_common_gate).sum()
                ),
                "cached_u8_sha256": _array_sha(cached_common_gate, "u1"),
                "raw_rebuilt_u8_sha256": _array_sha(
                    raw_common_gate, "u1"
                ),
            },
        },
        "ordered_pre_schedule": {
            "population": "common_finite_rows_only",
            "cached_count": len(cached_common_selected),
            "raw_rebuilt_count": len(raw_common_selected),
            "exact_equal": bool(
                np.array_equal(cached_common_selected, raw_common_selected)
            ),
            "symmetric_difference_count": len(
                set(int(value) for value in cached_common_selected)
                ^ set(int(value) for value in raw_common_selected)
            ),
            "cached_i64le_sha256": _array_sha(
                cached_common_selected, "<i8"
            ),
            "raw_rebuilt_i64le_sha256": _array_sha(
                raw_common_selected, "<i8"
            ),
        },
        "full_population_ordered_pre_schedule": {
            "population": "each_side_full_finite_population",
            "cached_eligible_rows": int(cached_eligible.sum()),
            "raw_rebuilt_eligible_rows": int(raw_eligible.sum()),
            "cached_gate_pass_count": int(cached_full_gate.sum()),
            "raw_rebuilt_gate_pass_count": int(raw_full_gate.sum()),
            "cached_confidence_pass_count": int(
                cached_full_components.confidence_passed.sum()
            ),
            "raw_rebuilt_confidence_pass_count": int(
                raw_full_components.confidence_passed.sum()
            ),
            "cached_structural_pass_count": int(
                cached_full_components.structural_passed.sum()
            ),
            "raw_rebuilt_structural_pass_count": int(
                raw_full_components.structural_passed.sum()
            ),
            "cached_count": len(cached_full_selected),
            "raw_rebuilt_count": len(raw_full_selected),
            "exact_equal": bool(
                np.array_equal(cached_full_selected, raw_full_selected)
            ),
            "actual_pre_schedule_behavior_changed": bool(
                not np.array_equal(cached_full_selected, raw_full_selected)
            ),
            "symmetric_difference_count": len(full_symmetric),
            "cached_only_selected_timestamps_utc": _timestamps(
                np.asarray(sorted(cached_full_set - raw_full_set), dtype="int64")
            ),
            "raw_rebuilt_only_selected_timestamps_utc": _timestamps(
                np.asarray(sorted(raw_full_set - cached_full_set), dtype="int64")
            ),
            "cached_i64le_sha256": _array_sha(cached_full_selected, "<i8"),
            "raw_rebuilt_i64le_sha256": _array_sha(raw_full_selected, "<i8"),
            "cached_probability_f64le_sha256": _array_sha(
                cached_full_probability, "<f8"
            ),
            "raw_rebuilt_probability_f64le_sha256": _array_sha(
                raw_full_probability, "<f8"
            ),
        },
        "cached_only_finite_loadedbook_scores": cached_only_report,
        "live_feature_builder_vs_raw_rebuilt": live_report,
    }
    return report


def _q1_source_mask(index: pd.DatetimeIndex) -> np.ndarray:
    if index.tz is None or index.hasnans:
        raise AuthorityError("Q1 source clock must be non-null and timezone-aware")
    source_ns = _datetime_index_ns(index, name="Q1 source window clock")
    decision_ns = source_ns + DECISION_SHIFT_NS
    return np.asarray(
        (decision_ns >= np.int64(START_DECISION.value))
        & (decision_ns < np.int64(END_DECISION.value)),
        dtype=bool,
    )


def _read_cached_q1_close(path: Path) -> pd.DataFrame:
    """Read only the close covariate needed by the public xpair recipe."""

    lower_source = START_DECISION - pd.Timedelta(seconds=60)
    upper_source = END_DECISION - pd.Timedelta(seconds=60)
    frame = pd.read_parquet(
        path,
        columns=["close"],
        filters=[
            ("datetime_utc", ">=", lower_source.to_pydatetime()),
            ("datetime_utc", "<", upper_source.to_pydatetime()),
        ],
    )
    if tuple(frame.columns) != ("close",):
        raise AuthorityError(f"{path}: cached close projection differs")
    if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.tz is None:
        raise AuthorityError(f"{path}: cached close clock must be timezone-aware")
    clock = _datetime_index_ns(frame.index, name=f"{path}:cached close clock")
    if len(clock) == 0 or np.any(np.diff(clock) <= 0):
        raise AuthorityError(f"{path}: cached close clock is empty or defective")
    values = frame["close"].to_numpy()
    try:
        finite = np.isfinite(values)
    except TypeError as exc:
        raise AuthorityError(f"{path}: cached close is not numeric") from exc
    if not np.all(finite):
        raise AuthorityError(f"{path}: cached close contains non-finite values")
    return frame


def _read_eurusd_q1_orderflow(
    path: Path, columns: Sequence[str]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Authenticate and project the outcome-free EURUSD order-flow source."""

    projection = tuple(columns)
    if not projection or any(not column.startswith("OF_") for column in projection):
        raise AuthorityError("EURUSD order-flow projection must be nonempty OF_ columns")
    if len(projection) != len(set(projection)):
        raise AuthorityError("EURUSD order-flow projection contains duplicates")
    schema = pq.ParquetFile(path).schema_arrow
    expected_schema = (*projection, "datetime_utc")
    if tuple(schema.names) != expected_schema:
        raise AuthorityError(
            f"EURUSD order-flow schema must be exactly {expected_schema}, "
            f"got {tuple(schema.names)}"
        )
    frame = _read_cached_q1_covariates(path, projection)
    return frame, {
        "source_kind": "cached_orderflow_comparator_projection",
        "schema_in_order": [
            {
                "name": field.name,
                "type": str(field.type),
                "nullable": field.nullable,
            }
            for field in schema
        ],
        "projection_in_order": list(projection),
        "semantic_or_outcome_columns_read": False,
        "rows": len(frame),
        "source_clock_i64le_sha256": _array_sha(
            _datetime_index_ns(frame.index, name=f"{path}:cached orderflow clock"),
            "<i8",
        ),
        "ordered_float32_values_sha256": _array_sha(
            frame.to_numpy(dtype="float32", copy=True), "<f4"
        ),
    }


def _acceptance_projection_by_source(
    books: dict[str, book_runtime.LoadedBook],
) -> dict[str, tuple[str, ...]]:
    projections: dict[str, tuple[str, ...]] = {}
    for source_pair in FEATURE_SOURCE_PAIRS:
        if source_pair not in books:
            projections[source_pair] = ("close",)
            continue
        base = tuple(
            column
            for column in books[source_pair].feature_cols
            if live_features.is_base_feature(column)
        )
        if not base:
            raise AuthorityError(f"{source_pair}: acceptance base projection is empty")
        if len(base) != len(set(base)):
            raise AuthorityError(
                f"{source_pair}: acceptance base projection contains duplicates"
            )
        projections[source_pair] = (*base, "close")
    return projections


def _acceptance_input_identities(
    books: dict[str, book_runtime.LoadedBook],
    processed_paths: dict[str, Sequence[Path]],
    projections: dict[str, tuple[str, ...]],
    orderflow_columns: Sequence[str],
) -> dict[str, Any]:
    cached_sources = [
        {
            "pair": pair,
            "file": _file_identity(
                REPO_ROOT / "features" / f"{pair}_2026.parquet"
            ),
            "projection_in_order": list(projections[pair]),
        }
        for pair in FEATURE_SOURCE_PAIRS
    ]
    processed_groups = [
        {
            "pair": pair,
            "projection_in_order": list(RAW_SCHEMA_COLUMNS),
            "ordered_files": [
                _file_identity(path) for path in processed_paths[pair]
            ],
        }
        for pair in FEATURE_SOURCE_PAIRS
    ]
    eurusd_processed = next(
        row for row in processed_groups if row["pair"] == "EURUSD"
    )
    return {
        "cached_feature_sources_in_feature_source_order": cached_sources,
        "processed_raw_source_groups_in_feature_source_order": processed_groups,
        "eurusd_cached_orderflow_comparator": {
            "file": _file_identity(
                REPO_ROOT / "features_of" / "EURUSD_2026.parquet"
            ),
            "projection_in_order": list(orderflow_columns),
        },
        "eurusd_rebuilt_orderflow_authority": {
            "source_pair": "EURUSD",
            "processed_raw_source_group_sha256": _identity_sha256(
                eurusd_processed
            ),
            "recipe": "orderflow.build_of_features(orderflow.of_1m(raw_10s))",
            "recipe_source": _file_identity(REPO_ROOT / "scripts/orderflow.py"),
            "projection_in_order": list(orderflow_columns),
        },
        "incumbent_books_in_target_pair_order": [
            _book_input_identities(books[pair])
            for pair in ACCEPTANCE_TARGET_PAIRS
        ],
        "forbidden_semantic_columns_read": [],
    }


def _identity_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return _sha256_bytes(encoded)


def _acceptance_pair_inputs(
    pair: str,
    book: book_runtime.LoadedBook,
    common_inputs: dict[str, Any],
    projections: dict[str, tuple[str, ...]],
) -> dict[str, Any]:
    cross_pair = pair in {"EURUSD", "GBPUSD", "USDCHF"}
    cached_rows = common_inputs["cached_feature_sources_in_feature_source_order"]
    processed_rows = common_inputs[
        "processed_raw_source_groups_in_feature_source_order"
    ]
    cached_by_pair = {row["pair"]: row for row in cached_rows}
    processed_by_pair = {row["pair"]: row for row in processed_rows}
    cross_inputs = (
        [cached_by_pair[source]["file"] for source in FEATURE_SOURCE_PAIRS]
        if cross_pair
        else []
    )
    if pair == "EURUSD":
        cross_inputs.append(
            common_inputs["eurusd_cached_orderflow_comparator"]["file"]
        )
    raw_groups = (
        [processed_by_pair[source] for source in FEATURE_SOURCE_PAIRS]
        if cross_pair
        else [processed_by_pair[pair]]
    )
    return {
        # Preserve the diagnostic authority's binding fields so the acceptance
        # validator can authenticate current incumbents and exact raw Q1 files.
        "cached_feature_file": cached_by_pair[pair]["file"],
        "cached_parquet_projection_in_order": list(book.feature_cols),
        "processed_parquet_projection_in_order": list(RAW_SCHEMA_COLUMNS),
        "processed_parquet_required_schema_in_order": list(RAW_SCHEMA_COLUMNS),
        "forbidden_semantic_columns_read": [],
        "processed_raw_files_in_loader_order": processed_by_pair[pair][
            "ordered_files"
        ],
        "cross_pair_inputs_actually_used": cross_inputs,
        "cross_pair_inputs_used": cross_pair,
        "incumbent_book": _book_input_identities(book),
        "cached_source_projections_in_feature_source_order": [
            {
                "pair": source,
                "projection_in_order": list(projections[source]),
            }
            for source in (FEATURE_SOURCE_PAIRS if cross_pair else (pair,))
        ],
        "processed_raw_source_groups_actually_used": raw_groups,
        "common_input_inventory_sha256": _identity_sha256(common_inputs),
    }


def _build_acceptance_feature_views(
    root: Path,
    books: dict[str, book_runtime.LoadedBook],
    processed_paths: dict[str, Sequence[Path]],
    projections: dict[str, tuple[str, ...]],
    orderflow_columns: Sequence[str],
) -> tuple[Path, Path, Path, Path, Path, dict[str, Any], dict[str, Any]]:
    """Rebuild each raw source once and materialize outcome-free Q1 views."""

    cached_features = root / "cached_features"
    raw_features = root / "raw_rebuilt_features"
    cached_orderflow = root / "cached_orderflow"
    raw_orderflow = root / "raw_orderflow"
    provider_store = root / "live_provider_store"
    for directory in (
        cached_features,
        raw_features,
        cached_orderflow,
        raw_orderflow,
        provider_store,
    ):
        directory.mkdir()

    source_reports: dict[str, Any] = {}
    rebuilt_orderflow_frame: pd.DataFrame | None = None
    rebuilt_orderflow_report: dict[str, Any] | None = None
    for source_pair in FEATURE_SOURCE_PAIRS:
        print(f"[phase-zero-acceptance] rebuild {source_pair}", flush=True)
        paths = processed_paths[source_pair]
        raw, loader_report = _load_processed_q1_strict(source_pair, paths)
        m1, raw_frame = _raw_feature_frame(raw)
        availability = _resampled_close_availability_report(raw, m1)
        if source_pair == "EURUSD":
            rebuilt_orderflow_frame = orderflow.build_of_features(
                orderflow.of_1m(raw)
            )
            rebuilt_orderflow_frame = rebuilt_orderflow_frame.replace(
                [np.inf, -np.inf], np.nan
            ).astype("float32")
            if tuple(rebuilt_orderflow_frame.columns) != tuple(orderflow_columns):
                raise AuthorityError(
                    "EURUSD raw order-flow rebuild schema/order differs from incumbent"
                )
            rebuilt_orderflow_frame = rebuilt_orderflow_frame.loc[
                _q1_source_mask(rebuilt_orderflow_frame.index),
                list(orderflow_columns),
            ]
            rebuilt_orderflow_frame.to_parquet(
                raw_orderflow / "EURUSD_2026.parquet"
            )
            rebuilt_orderflow_report = {
                "source_kind": "rebuilt_from_strict_processed_eurusd_10s",
                "recipe": (
                    "orderflow.build_of_features(orderflow.of_1m(raw_10s))"
                ),
                "recipe_source": _file_identity(
                    REPO_ROOT / "scripts/orderflow.py"
                ),
                "projection_in_order": list(orderflow_columns),
                "semantic_or_outcome_columns_read": False,
                "rows": len(rebuilt_orderflow_frame),
                "source_clock_i64le_sha256": _array_sha(
                    _datetime_index_ns(
                        rebuilt_orderflow_frame.index,
                        name="EURUSD rebuilt orderflow clock",
                    ),
                    "<i8",
                ),
                "ordered_float32_values_sha256": _array_sha(
                    rebuilt_orderflow_frame.to_numpy(
                        dtype="float32", copy=True
                    ),
                    "<f4",
                ),
                "same_strict_raw_load_as_eurusd_base_rebuild": True,
            }
        if not raw_frame.index.equals(m1.index):
            raise AuthorityError(
                f"{source_pair}: raw feature and resampled-minute clocks differ"
            )
        window = _q1_source_mask(raw_frame.index)
        base_columns = tuple(
            column for column in projections[source_pair] if column != "close"
        )
        missing = [column for column in base_columns if column not in raw_frame.columns]
        if missing:
            raise AuthorityError(
                f"{source_pair}: raw rebuild is missing base columns {missing[:12]}"
            )
        raw_view = raw_frame.loc[window, list(base_columns)].copy()
        raw_view["close"] = m1.loc[raw_view.index, "close"].to_numpy(
            dtype="float32", copy=True
        )
        raw_view = raw_view.loc[:, list(projections[source_pair])]
        if tuple(raw_view.columns) != projections[source_pair]:
            raise AuthorityError(f"{source_pair}: raw view projection order differs")
        raw_view.to_parquet(raw_features / f"{source_pair}_2026.parquet")

        cached_path = REPO_ROOT / "features" / f"{source_pair}_2026.parquet"
        cached_close = _read_cached_q1_close(cached_path)
        if base_columns:
            cached_base = _read_cached_q1_covariates(cached_path, base_columns)
            if not cached_base.index.equals(cached_close.index):
                raise AuthorityError(
                    f"{source_pair}: cached base and close clocks differ"
                )
            cached_view = cached_base.copy()
            cached_view["close"] = cached_close["close"].to_numpy(
                dtype="float32", copy=True
            )
        else:
            cached_view = cached_close.astype("float32", copy=True)
        cached_view = cached_view.loc[:, list(projections[source_pair])]
        cached_view.to_parquet(cached_features / f"{source_pair}_2026.parquet")

        provider_m1 = m1.loc[window].copy()
        if len(provider_m1) < 500:
            raise AuthorityError(
                f"{source_pair}: insufficient Q1 resampled rows for live provider"
            )
        provider_m1.to_parquet(provider_store / f"{source_pair}.parquet")
        raw_clock = _datetime_index_ns(
            raw.index, name=f"{source_pair}:acceptance raw observation clock"
        )
        m1_clock = _datetime_index_ns(
            m1.index, name=f"{source_pair}:acceptance minute clock"
        )
        m1_values = np.ascontiguousarray(
            m1.loc[:, ["open", "high", "low", "close", "volume", "gap_prev"]]
            .to_numpy(dtype="float64", copy=True)
        )
        source_reports[source_pair] = {
            "strict_processed_source": loader_report,
            "causal_resampled_close_availability": availability,
            "processed_raw_derived_identity": {
                "observation_rows": len(raw_clock),
                "observation_clock_i64le_sha256": _array_sha(raw_clock, "<i8"),
                "one_minute_rows": len(m1_clock),
                "one_minute_clock_i64le_sha256": _array_sha(m1_clock, "<i8"),
                "one_minute_ohlcv_gap_f64le_sha256": _array_sha(
                    m1_values, "<f8"
                ),
            },
            "raw_rebuilt_q1_feature_view": {
                "projection_in_order": list(projections[source_pair]),
                "rows": len(raw_view),
                "source_clock_i64le_sha256": _array_sha(
                    _datetime_index_ns(
                        raw_view.index,
                        name=f"{source_pair}:raw rebuilt feature-view clock",
                    ),
                    "<i8",
                ),
                "ordered_float32_values_sha256": _array_sha(
                    raw_view.to_numpy(dtype="float32", copy=True), "<f4"
                ),
            },
            "cached_q1_feature_view": {
                "projection_in_order": list(projections[source_pair]),
                "rows": len(cached_view),
                "source_clock_i64le_sha256": _array_sha(
                    _datetime_index_ns(
                        cached_view.index,
                        name=f"{source_pair}:cached feature-view clock",
                    ),
                    "<i8",
                ),
                "ordered_float32_values_sha256": _array_sha(
                    cached_view.to_numpy(dtype="float32", copy=True), "<f4"
                ),
            },
            "live_provider_rolling_store": {
                "rows": len(provider_m1),
                "source_clock_i64le_sha256": _array_sha(
                    _datetime_index_ns(
                        provider_m1.index,
                        name=f"{source_pair}:live provider rolling-store clock",
                    ),
                    "<i8",
                ),
                "ordered_ohlcv_gap_f64le_sha256": _array_sha(
                    provider_m1.loc[
                        :, ["open", "high", "low", "close", "volume", "gap_prev"]
                    ].to_numpy(dtype="float64", copy=True),
                    "<f8",
                ),
            },
            "raw_source_loaded_and_rebuilt_once": True,
            "semantic_or_outcome_columns_read": False,
        }
        del raw, m1, raw_frame, raw_view, cached_view, provider_m1

    if rebuilt_orderflow_frame is None or rebuilt_orderflow_report is None:
        raise AuthorityError("EURUSD raw order-flow rebuild was not produced")
    orderflow_path = REPO_ROOT / "features_of" / "EURUSD_2026.parquet"
    cached_of, cached_orderflow_report = _read_eurusd_q1_orderflow(
        orderflow_path, orderflow_columns
    )
    cached_of.to_parquet(cached_orderflow / "EURUSD_2026.parquet")
    orderflow_report = {
        "cached_comparator": cached_orderflow_report,
        "raw_rebuilt_from_processed": rebuilt_orderflow_report,
        "source_clock_exact": bool(
            cached_of.index.equals(rebuilt_orderflow_frame.index)
        ),
        "finite_mask_exact": bool(
            cached_of.shape == rebuilt_orderflow_frame.shape
            and np.array_equal(
                np.isfinite(cached_of.to_numpy(dtype="float32", copy=False)),
                np.isfinite(
                    rebuilt_orderflow_frame.to_numpy(
                        dtype="float32", copy=False
                    )
                ),
            )
        ),
        "ordered_float32_values_exact": bool(
            cached_of.shape == rebuilt_orderflow_frame.shape
            and cached_of.to_numpy(dtype="float32", copy=True).tobytes(order="C")
            == rebuilt_orderflow_frame.to_numpy(
                dtype="float32", copy=True
            ).tobytes(order="C")
        ),
    }
    return (
        cached_features,
        raw_features,
        cached_orderflow,
        raw_orderflow,
        provider_store,
        source_reports,
        orderflow_report,
    )


def _acceptance_provider_report(
    pair: str,
    book: book_runtime.LoadedBook,
    raw_score: ScoreRows,
    builder: live_features.LiveFeatureBuilder,
) -> dict[str, Any]:
    if pair == "EURUSD":
        orderflow_columns = tuple(
            column for column in book.feature_cols if column.startswith("OF_")
        )
        if not orderflow_columns:
            raise AuthorityError("EURUSD incumbent contains no OF_ schema to verify")
        public_rejected = False
        rejection = None
        try:
            builder.feature_row(pair, list(book.feature_cols))
        except live_features.LiveFeatureError as exc:
            public_rejected = True
            rejection = str(exc)
        joined = builder._joined_row_frame(pair, list(book.feature_cols))
        missing = [column for column in orderflow_columns if column not in joined.columns]
        exact_block = bool(
            public_rejected
            and tuple(missing) == orderflow_columns
            and not any(column.startswith("OF_") for column in joined.columns)
        )
        return {
            "status": "BLOCKED_SCHEMA",
            "reason": "verified_live_OF_provider_absent",
            "live_activation_eligible": False,
            "acceptance_semantics": (
                "verified_offline_behavior_pass_live_activation_blocked"
            ),
            "provider_state_exact": exact_block,
            "requested_orderflow_columns_in_order": list(orderflow_columns),
            "public_provider_missing_orderflow_columns_in_order": missing,
            "public_feature_row_rejected_missing_schema": public_rejected,
            "public_feature_row_rejection": rejection,
            "public_live_feature_builder_parity": None,
        }
    parity = refresh._live_feature_builder_parity_report(
        pair, book.feature_cols, raw_score, builder
    )
    diagnostics = parity.get("decoded_diagnostics")
    finite_values_exact = bool(
        isinstance(diagnostics, dict)
        and diagnostics.get("mismatched_cells") == 0
        and diagnostics.get("finite_nonfinite_disagreements") == 0
    )
    behavior_equal = bool(
        finite_values_exact
        and parity.get("feature_order_exact") is True
        and parity.get("source_clock_subset_exact") is True
        and parity.get("causal_decision_mapping_exact") is True
        and parity.get("finite_mask_exact") is True
        and parity.get("public_latest_row_exact") is True
    )
    parity["ordered_finite_float32_values_exact"] = finite_values_exact
    parity["nonfinite_nan_payload_bits_binding"] = False
    parity["behavior_all_equal"] = behavior_equal
    return {
        "status": "PASSED" if behavior_equal else "FAILED",
        "reason": None if behavior_equal else "provider_parity_failed",
        "live_activation_eligible": behavior_equal,
        "acceptance_semantics": "public_live_feature_builder_parity_required",
        "provider_state_exact": behavior_equal,
        "requested_orderflow_columns_in_order": [],
        "public_provider_missing_orderflow_columns_in_order": [],
        "public_feature_row_rejected_missing_schema": False,
        "public_feature_row_rejection": None,
        "public_live_feature_builder_parity": parity,
    }


def _loadedbook_scalar_parity_report(
    book: book_runtime.LoadedBook,
    cached_matrix: np.ndarray,
    raw_matrix: np.ndarray,
    cached_probability: np.ndarray,
    raw_probability: np.ndarray,
    cached_components: book_runtime.BookGateComponents,
    raw_components: book_runtime.BookGateComponents,
) -> dict[str, Any]:
    count = min(257, len(cached_probability))
    ranks = np.unique(
        np.floor(np.linspace(0, len(cached_probability) - 1, count)).astype(
            "int64"
        )
    )
    cached_raw_probability_bits_exact = True
    cached_raw_direction_exact = True
    cached_raw_gate_exact = True
    scalar_batch_direction_exact = True
    scalar_batch_gate_exact = True
    scalar_batch_probability_bits_exact = True
    for rank in ranks:
        cached = book.score(
            pd.Series(
                cached_matrix[rank], index=book.feature_cols, dtype="float64"
            )
        )
        raw = book.score(
            pd.Series(raw_matrix[rank], index=book.feature_cols, dtype="float64")
        )
        cached_bits = np.float64(cached.proba).view("uint64")
        raw_bits = np.float64(raw.proba).view("uint64")
        cached_raw_probability_bits_exact &= bool(cached_bits == raw_bits)
        cached_raw_direction_exact &= cached.direction == raw.direction
        cached_raw_gate_exact &= bool(
            cached.gate_passed == raw.gate_passed
            and cached.gate_reasons == raw.gate_reasons
        )
        scalar_batch_direction_exact &= bool(
            cached.direction
            == ("UP" if cached_probability[rank] >= 0.5 else "DOWN")
            and raw.direction
            == ("UP" if raw_probability[rank] >= 0.5 else "DOWN")
        )
        scalar_batch_gate_exact &= bool(
            cached.gate_passed == cached_components.gate_passed[rank]
            and raw.gate_passed == raw_components.gate_passed[rank]
        )
        scalar_batch_probability_bits_exact &= bool(
            cached_bits == cached_probability[rank].view("uint64")
            and raw_bits == raw_probability[rank].view("uint64")
        )
    behavior_exact = bool(
        len(ranks) > 0
        and cached_raw_probability_bits_exact
        and cached_raw_direction_exact
        and cached_raw_gate_exact
        and scalar_batch_direction_exact
        and scalar_batch_gate_exact
    )
    return {
        "probe_rows": len(ranks),
        "cached_raw_probability_bits_exact": cached_raw_probability_bits_exact,
        "cached_raw_direction_exact": cached_raw_direction_exact,
        "cached_raw_gate_exact": cached_raw_gate_exact,
        "scalar_batch_direction_exact": scalar_batch_direction_exact,
        "scalar_batch_gate_exact": scalar_batch_gate_exact,
        "scalar_batch_probability_bits_exact_diagnostic": (
            scalar_batch_probability_bits_exact
        ),
        "scalar_batch_probability_bits_binding": False,
        "behavior_exact": behavior_exact,
    }


def _acceptance_verdict(checks: dict[str, bool]) -> str:
    required = {
        "outcome_free_score_adapter_exact",
        "finite_eligibility_exact",
        "common_probability_bitwise_exact",
        "common_direction_exact",
        "loadedbook_scalar_probes_exact",
        "common_gate_components_exact",
        "common_ordered_pre_schedule_exact",
        "full_population_ordered_pre_schedule_exact",
        "provider_state_exact",
    }
    if set(checks) != required or any(
        type(value) is not bool for value in checks.values()
    ):
        raise AuthorityError("acceptance verdict checks are malformed")
    if not checks["finite_eligibility_exact"]:
        return "STOP_PHASE_ZERO_SCOREABILITY_MISMATCH"
    if all(checks.values()):
        return "PASS_BEHAVIOR_AUTHORITY"
    return "STOP_PHASE_ZERO_BEHAVIOR_MISMATCH"


def _analyze_acceptance_target(
    pair: str,
    book: book_runtime.LoadedBook,
    cached_score: ScoreRows,
    raw_score: ScoreRows,
    builder: live_features.LiveFeatureBuilder,
    inputs: dict[str, Any],
) -> dict[str, Any]:
    if not (
        cached_score.feature_cols == raw_score.feature_cols
        and cached_score.feature_cols == tuple(book.feature_cols)
        and cached_score.X.dtype == np.dtype("float32")
        and raw_score.X.dtype == np.dtype("float32")
        and np.array_equal(
            cached_score.source_feature_ns, raw_score.source_feature_ns
        )
        and np.array_equal(cached_score.entry_ns, raw_score.entry_ns)
    ):
        raise AuthorityError(f"{pair}: cached/raw outcome-free score contract differs")
    if not np.array_equal(
        cached_score.entry_ns,
        cached_score.source_feature_ns + DECISION_SHIFT_NS,
    ):
        raise AuthorityError(f"{pair}: acceptance decision clock is not source +60s")

    cached_X = cached_score.X
    raw_X = raw_score.X
    decision_ns = cached_score.entry_ns
    cached_eligible = np.isfinite(cached_X).all(axis=1)
    raw_eligible = np.isfinite(raw_X).all(axis=1)
    common = cached_eligible & raw_eligible
    mismatch = cached_eligible != raw_eligible
    if not common.any():
        raise AuthorityError(f"{pair}: no common finite phase-zero rows")

    cached_common = pd.DataFrame(
        cached_X[common], columns=book.feature_cols
    )
    raw_common = pd.DataFrame(raw_X[common], columns=book.feature_cols)
    cached_probability = book.predict_probabilities(cached_common)
    raw_probability = book.predict_probabilities(raw_common)
    cached_components = book.gate_components(cached_common, cached_probability)
    raw_components = book.gate_components(raw_common, raw_probability)
    common_decisions = decision_ns[common]
    cached_selected = common_decisions[cached_components.gate_passed]
    raw_selected = common_decisions[raw_components.gate_passed]

    cached_full_probability, cached_full_components, cached_full_selected = (
        _full_selection(book, cached_X, decision_ns, cached_eligible)
    )
    raw_full_probability, raw_full_components, raw_full_selected = _full_selection(
        book, raw_X, decision_ns, raw_eligible
    )
    scalar_parity = _loadedbook_scalar_parity_report(
        book,
        cached_X[common],
        raw_X[common],
        cached_probability,
        raw_probability,
        cached_components,
        raw_components,
    )
    provider = _acceptance_provider_report(pair, book, raw_score, builder)
    checks = {
        "outcome_free_score_adapter_exact": True,
        "finite_eligibility_exact": bool(not mismatch.any()),
        "common_probability_bitwise_exact": bool(
            np.array_equal(
                cached_probability.view("uint64"), raw_probability.view("uint64")
            )
        ),
        "common_direction_exact": bool(
            np.array_equal(
                cached_probability >= 0.5, raw_probability >= 0.5
            )
        ),
        "loadedbook_scalar_probes_exact": bool(
            scalar_parity["behavior_exact"] is True
        ),
        "common_gate_components_exact": bool(
            np.array_equal(
                cached_components.confidence_passed,
                raw_components.confidence_passed,
            )
            and np.array_equal(
                cached_components.structural_passed,
                raw_components.structural_passed,
            )
        ),
        "common_ordered_pre_schedule_exact": bool(
            np.array_equal(cached_selected, raw_selected)
        ),
        "full_population_ordered_pre_schedule_exact": bool(
            np.array_equal(cached_full_selected, raw_full_selected)
        ),
        "provider_state_exact": bool(provider["provider_state_exact"] is True),
    }
    mismatch_positions = np.flatnonzero(mismatch)
    mismatch_decisions = decision_ns[mismatch]
    mismatch_columns = sorted(
        {
            book.feature_cols[column]
            for row in mismatch_positions
            for column in np.flatnonzero(
                np.isfinite(cached_X[row]) != np.isfinite(raw_X[row])
            )
        }
    )
    cached_full_set = set(int(value) for value in cached_full_selected)
    raw_full_set = set(int(value) for value in raw_full_selected)
    return {
        "verdict": _acceptance_verdict(checks),
        "verdict_checks": checks,
        "inputs": inputs,
        "rows": len(decision_ns),
        "feature_count": len(book.feature_cols),
        "feature_dtype": "float32",
        "source_feature_clock_i64le_sha256": _array_sha(
            cached_score.source_feature_ns, "<i8"
        ),
        "decision_clock_i64le_sha256": _array_sha(decision_ns, "<i8"),
        "causal_decision_mapping_exact": True,
        "ordered_float32_features": {
            "cached_sha256": _array_sha(cached_X, "<f4"),
            "raw_rebuilt_sha256": _array_sha(raw_X, "<f4"),
            "bitwise_equal": bool(
                np.array_equal(cached_X.view("uint32"), raw_X.view("uint32"))
            ),
            "behavior_neutral_decoded_drift": bool(all(checks.values())),
            "decoded_diagnostics": refresh._decoded_diagnostics(
                book.feature_cols, cached_X, raw_X
            ),
        },
        "finite_eligibility": {
            "cached_count": int(cached_eligible.sum()),
            "raw_rebuilt_count": int(raw_eligible.sum()),
            "common_count": int(common.sum()),
            "mismatch_count": int(mismatch.sum()),
            "cached_u8_sha256": _array_sha(cached_eligible, "u1"),
            "raw_rebuilt_u8_sha256": _array_sha(raw_eligible, "u1"),
        },
        "scoreability_mismatch_source": {
            "columns": mismatch_columns,
            "rows": len(mismatch_positions),
            "cached_finite_raw_nonfinite": int(
                (cached_eligible & ~raw_eligible).sum()
            ),
            "cached_nonfinite_raw_finite": int(
                ((~cached_eligible) & raw_eligible).sum()
            ),
            "first_decision_utc": (
                _timestamps(mismatch_decisions[:1])[0]
                if len(mismatch_decisions)
                else None
            ),
            "last_decision_utc": (
                _timestamps(mismatch_decisions[-1:])[0]
                if len(mismatch_decisions)
                else None
            ),
            "decision_i64le_sha256": _array_sha(mismatch_decisions, "<i8"),
        },
        "common_loadedbook_probability": {
            "population": "common_finite_rows_only",
            "scorer": "book_runtime.LoadedBook.predict_probabilities",
            "rows": int(common.sum()),
            "bit_mismatch_count": int(
                (
                    cached_probability.view("uint64")
                    != raw_probability.view("uint64")
                ).sum()
            ),
            "direction_mismatch_count": int(
                (
                    (cached_probability >= 0.5)
                    != (raw_probability >= 0.5)
                ).sum()
            ),
            "max_abs_difference": float(
                np.max(np.abs(cached_probability - raw_probability))
            ),
            "cached_f64le_sha256": _array_sha(cached_probability, "<f8"),
            "raw_rebuilt_f64le_sha256": _array_sha(raw_probability, "<f8"),
            "loadedbook_scalar_parity": scalar_parity,
        },
        "gate_components": {
            "confidence": {
                "cached_pass_count": int(
                    cached_components.confidence_passed.sum()
                ),
                "raw_rebuilt_pass_count": int(
                    raw_components.confidence_passed.sum()
                ),
                "mismatch_count": int(
                    (
                        cached_components.confidence_passed
                        != raw_components.confidence_passed
                    ).sum()
                ),
            },
            "structural": {
                "cached_pass_count": int(
                    cached_components.structural_passed.sum()
                ),
                "raw_rebuilt_pass_count": int(
                    raw_components.structural_passed.sum()
                ),
                "mismatch_count": int(
                    (
                        cached_components.structural_passed
                        != raw_components.structural_passed
                    ).sum()
                ),
            },
            "combined": {
                "cached_pass_count": int(cached_components.gate_passed.sum()),
                "raw_rebuilt_pass_count": int(raw_components.gate_passed.sum()),
                "mismatch_count": int(
                    (
                        cached_components.gate_passed
                        != raw_components.gate_passed
                    ).sum()
                ),
            },
        },
        "ordered_pre_schedule": {
            "population": "common_finite_rows_only",
            "cached_count": len(cached_selected),
            "raw_rebuilt_count": len(raw_selected),
            "exact_equal": bool(np.array_equal(cached_selected, raw_selected)),
            "cached_i64le_sha256": _array_sha(cached_selected, "<i8"),
            "raw_rebuilt_i64le_sha256": _array_sha(raw_selected, "<i8"),
        },
        "full_population_ordered_pre_schedule": {
            "population": "each_side_full_finite_population",
            "cached_eligible_rows": int(cached_eligible.sum()),
            "raw_rebuilt_eligible_rows": int(raw_eligible.sum()),
            "cached_count": len(cached_full_selected),
            "raw_rebuilt_count": len(raw_full_selected),
            "exact_equal": bool(
                np.array_equal(cached_full_selected, raw_full_selected)
            ),
            "actual_pre_schedule_behavior_changed": bool(
                not np.array_equal(cached_full_selected, raw_full_selected)
            ),
            "symmetric_difference_count": len(cached_full_set ^ raw_full_set),
            "cached_i64le_sha256": _array_sha(cached_full_selected, "<i8"),
            "raw_rebuilt_i64le_sha256": _array_sha(
                raw_full_selected, "<i8"
            ),
            "cached_probability_f64le_sha256": _array_sha(
                cached_full_probability, "<f8"
            ),
            "raw_rebuilt_probability_f64le_sha256": _array_sha(
                raw_full_probability, "<f8"
            ),
        },
        "provider_qualification": provider,
    }


def _canonical_json_bytes(value: dict[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def _require_new_output(path: Path) -> None:
    if path.exists() or path.is_symlink():
        raise AuthorityError(f"authority output already exists; refusing overwrite: {path}")


def _atomic_create(path: Path, content: bytes) -> None:
    """Publish a read-only authority without ever replacing an existing path."""

    _require_new_output(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.tmp-", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
            os.fchmod(handle.fileno(), 0o444)
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise AuthorityError(
                f"authority output appeared concurrently; refusing overwrite: {path}"
            ) from exc
        temporary.unlink()
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


def generate(output: Path) -> dict[str, Any]:
    _require_new_output(output)
    implementation_before = _implementation_identities()
    counterfactual = _verify_live_feature_counterfactual_sources(
        implementation_before
    )
    books = book_runtime.load_target_books(list(PAIRS))
    report: dict[str, Any] = {
        "schema": "m15-book-refresh-phase-zero-behavior-authority/v2",
        "classification": "outcome_blind_pre_april_diagnostic",
        "semantic_labels_or_outcomes_accessed": False,
        "acceptance_artifact": False,
        "artifact_serialization": {
            "encoding": "UTF-8",
            "json_key_order": "lexicographic_recursive",
            "json_separators": [",", ":"],
            "allow_nan": False,
            "terminal_newline_bytes_hex": "0a",
        },
        "generation": {
            "working_directory": str(REPO_ROOT),
            "exact_executable_command": (
                "/home/sean/binary-algo-venv/bin/python "
                "scripts/m15_phase_zero_behavior_authority.py"
            ),
            "generator": "scripts/m15_phase_zero_behavior_authority.py",
            "action": "generate_and_atomically_create_canonical_json",
        },
        "recreate": {
            "exact_command": (
                "~/binary-algo-venv/bin/python "
                "scripts/m15_phase_zero_behavior_authority.py"
            ),
            "campaign_command_template": (
                "~/binary-algo-venv/bin/python scripts/m15_book_refresh.py "
                "--adapter-parity --prereg-id <sealed_prereg_id>"
            ),
            "campaign_function": "m15_book_refresh.run_adapter_parity",
        },
        "runtime_versions": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "lightgbm": importlib.metadata.version("lightgbm"),
            "pyarrow": importlib.metadata.version("pyarrow"),
        },
        "window": {
            "entry_at_or_after": START_DECISION.isoformat().replace("+00:00", "Z"),
            "entry_before": END_DECISION.isoformat().replace("+00:00", "Z"),
            "clock": "left_labeled_source_feature_timestamp_plus_60_seconds",
        },
        "implementation_sha256": implementation_before,
        "unmodified_head_live_provider_counterfactual": counterfactual,
        "pairs": {},
    }
    for pair in PAIRS:
        print(f"[phase-zero-authority] {pair}", flush=True)
        report["pairs"][pair] = _analyze_pair(pair, books[pair])
    implementation_after = _implementation_identities()
    if implementation_before != implementation_after:
        raise AuthorityError(
            "runner/provider/source implementation changed while generating authority"
        )
    _atomic_create(output, _canonical_json_bytes(report))
    return report


def generate_acceptance(output: Path) -> dict[str, Any]:
    """Generate the six-target, seven-source pre-April acceptance authority."""

    _require_new_output(output)
    pinned_runtime, runtime_contract = _verified_acceptance_runtime()
    if tuple(live_features.PAIRS) != FEATURE_SOURCE_PAIRS:
        raise AuthorityError(
            "public live feature-source order differs from acceptance authority"
        )
    implementation_before = _implementation_identities()
    counterfactual = _verify_live_feature_counterfactual_sources(
        implementation_before
    )
    books = book_runtime.load_target_books(list(ACCEPTANCE_TARGET_PAIRS))
    if tuple(books) != ACCEPTANCE_TARGET_PAIRS:
        raise AuthorityError("loaded acceptance book order differs")
    processed_paths = {
        pair: tuple(_processed_paths(pair)) for pair in FEATURE_SOURCE_PAIRS
    }
    projections = _acceptance_projection_by_source(books)
    orderflow_columns = tuple(
        column
        for column in books["EURUSD"].feature_cols
        if column.startswith("OF_")
    )
    source_inputs_before = _acceptance_input_identities(
        books, processed_paths, projections, orderflow_columns
    )

    report: dict[str, Any] = {
        "schema": "m15-book-refresh-phase-zero-acceptance/v1",
        "classification": "outcome_blind_pre_april_acceptance",
        "semantic_labels_or_outcomes_accessed": False,
        "acceptance_artifact": False,
        "all_six_targets_pass": False,
        "target_pair_order": list(ACCEPTANCE_TARGET_PAIRS),
        "feature_source_pair_order": list(FEATURE_SOURCE_PAIRS),
        "artifact_serialization": {
            "encoding": "UTF-8",
            "json_key_order": "lexicographic_recursive",
            "json_separators": [",", ":"],
            "allow_nan": False,
            "terminal_newline_bytes_hex": "0a",
        },
        "generation": {
            "working_directory": str(REPO_ROOT),
            "exact_executable_command": (
                f"{Path(sys.executable).absolute()} "
                "scripts/m15_phase_zero_behavior_authority.py --mode acceptance"
            ),
            "python_executable": str(Path(sys.executable).absolute()),
            "generator": "scripts/m15_phase_zero_behavior_authority.py",
            "action": "generate_and_atomically_create_canonical_json",
        },
        "recreate": {
            "exact_command": (
                "PYTHONNOUSERSITE=1 ~/binary-algo-venv/bin/python "
                "scripts/m15_phase_zero_behavior_authority.py --mode acceptance"
            ),
            "campaign_command_template": (
                "PYTHONNOUSERSITE=1 ~/binary-algo-venv/bin/python "
                "scripts/m15_book_refresh.py "
                "--adapter-parity --prereg-id <sealed_prereg_id>"
            ),
            "campaign_function": "m15_book_refresh.run_adapter_parity",
        },
        "runtime_versions": {
            "python": sys.version.split()[0],
            **pinned_runtime,
        },
        "runtime_version_contract": runtime_contract,
        "window": {
            "entry_at_or_after": START_DECISION.isoformat().replace(
                "+00:00", "Z"
            ),
            "entry_before": END_DECISION.isoformat().replace("+00:00", "Z"),
            "clock": "left_labeled_source_feature_timestamp_plus_60_seconds",
        },
        "implementation_sha256": implementation_before,
        "unmodified_head_live_provider_counterfactual": counterfactual,
        "source_inputs": source_inputs_before,
        "source_inputs_sha256": _identity_sha256(source_inputs_before),
        "source_input_identities_pre_post_exact": False,
        "raw_source_rebuilds": {},
        "eurusd_orderflow_behavior": {},
        "pairs": {},
    }
    with tempfile.TemporaryDirectory(prefix="m15-phase-zero-acceptance-") as name:
        temporary = Path(name)
        (
            cached_features,
            raw_features,
            cached_orderflow,
            raw_orderflow,
            provider_store,
            source_reports,
            orderflow_report,
        ) = _build_acceptance_feature_views(
            temporary,
            books,
            processed_paths,
            projections,
            orderflow_columns,
        )
        report["raw_source_rebuilds"] = source_reports
        report["eurusd_orderflow_behavior"] = orderflow_report
        maximum_provider_rows = max(
            row["live_provider_rolling_store"]["rows"]
            for row in source_reports.values()
        )
        builder = live_features.LiveFeatureBuilder(
            client=None,
            history_minutes=maximum_provider_rows + 1,
            tick_volume_count=0,
            store_dir=provider_store,
            stale_seconds=10**9,
        )
        for pair in ACCEPTANCE_TARGET_PAIRS:
            print(f"[phase-zero-acceptance] analyze {pair}", flush=True)
            book = books[pair]
            cached_score = slice_score_rows(
                build_score_rows(
                    pair,
                    ["2026"],
                    book.feature_cols,
                    feature_dir=cached_features,
                    orderflow_dir=cached_orderflow,
                ),
                entry_at_or_after=START_DECISION,
                entry_before=END_DECISION,
            )
            raw_score = slice_score_rows(
                build_score_rows(
                    pair,
                    ["2026"],
                    book.feature_cols,
                    feature_dir=raw_features,
                    orderflow_dir=raw_orderflow,
                ),
                entry_at_or_after=START_DECISION,
                entry_before=END_DECISION,
            )
            pair_inputs = _acceptance_pair_inputs(
                pair, book, source_inputs_before, projections
            )
            report["pairs"][pair] = _analyze_acceptance_target(
                pair, book, cached_score, raw_score, builder, pair_inputs
            )

    source_inputs_after = _acceptance_input_identities(
        books, processed_paths, projections, orderflow_columns
    )
    if source_inputs_before != source_inputs_after:
        raise AuthorityError(
            "an acceptance source input changed while authority was generated"
        )
    report["source_input_identities_pre_post_exact"] = True
    implementation_after = _implementation_identities()
    if implementation_before != implementation_after:
        raise AuthorityError(
            "runner/provider/source implementation changed while generating acceptance"
        )
    all_pass = bool(
        tuple(report["pairs"]) == ACCEPTANCE_TARGET_PAIRS
        and all(
            report["pairs"][pair]["verdict"] == "PASS_BEHAVIOR_AUTHORITY"
            for pair in ACCEPTANCE_TARGET_PAIRS
        )
    )
    report["all_six_targets_pass"] = all_pass
    report["acceptance_artifact"] = bool(
        all_pass
        and tuple(report["target_pair_order"]) == ACCEPTANCE_TARGET_PAIRS
        and tuple(report["feature_source_pair_order"]) == FEATURE_SOURCE_PAIRS
        and report["source_input_identities_pre_post_exact"] is True
    )
    _atomic_create(output, _canonical_json_bytes(report))
    return report


def _self_check() -> None:
    """Exercise strict projection/clock rejection and fail-closed verdict logic."""

    def fixture(clock: Sequence[pd.Timestamp]) -> pd.DataFrame:
        n = len(clock)
        base = np.arange(n, dtype="float64") + 1.0
        frame = pd.DataFrame(
            {
                "open": base,
                "high": base + 0.2,
                "low": base - 0.2,
                "close": base + 0.1,
                "volume": base * 10.0,
            },
            index=pd.DatetimeIndex(clock, name="datetime_utc").as_unit("ns"),
        )
        return frame

    def expect_rejected(paths: Sequence[Path], phrase: str) -> None:
        try:
            _load_processed_q1_strict("USDJPY", paths)
        except AuthorityError as exc:
            if phrase not in str(exc):
                raise AuthorityError(
                    f"self-check expected rejection containing {phrase!r}, got {exc!r}"
                ) from exc
        else:
            raise AuthorityError(f"self-check unexpectedly accepted {phrase}")

    with tempfile.TemporaryDirectory(prefix="m15-authority-self-check-") as name:
        root = Path(name)
        safe_a = root / "safe_a.parquet"
        safe_b = root / "safe_b.parquet"
        fixture(pd.date_range("2026-01-01T00:00:00Z", periods=3, freq="10s")).to_parquet(
            safe_a
        )
        fixture(pd.date_range("2026-01-01T00:01:00Z", periods=3, freq="10s")).to_parquet(
            safe_b
        )
        raw, audit = _load_processed_q1_strict("USDJPY", [safe_a, safe_b])
        if len(raw) != 6 or audit["semantic_or_outcome_columns_read"] is not False:
            raise AuthorityError("strict loader self-check result differs")
        availability = _resampled_close_availability_report(
            raw, pipeline.resample_1m(raw)
        )
        if availability["every_close_available_no_later_than_decision"] is not True:
            raise AuthorityError("causal close-availability self-check differs")

        extra = fixture(
            pd.date_range("2026-01-01T00:00:00Z", periods=3, freq="10s")
        )
        extra["y"] = np.zeros(len(extra), dtype="uint8")
        extra_path = root / "extra_schema.parquet"
        extra.to_parquet(extra_path)
        expect_rejected([extra_path], "processed schema must be exactly")

        bad_clocks = {
            "unsorted": [
                pd.Timestamp("2026-01-01T00:00:00Z"),
                pd.Timestamp("2026-01-01T00:00:20Z"),
                pd.Timestamp("2026-01-01T00:00:10Z"),
            ],
            "duplicate": [
                pd.Timestamp("2026-01-01T00:00:00Z"),
                pd.Timestamp("2026-01-01T00:00:10Z"),
                pd.Timestamp("2026-01-01T00:00:10Z"),
            ],
            "off_lattice": [
                pd.Timestamp("2026-01-01T00:00:00Z"),
                pd.Timestamp("2026-01-01T00:00:15Z"),
            ],
        }
        for label, clock in bad_clocks.items():
            path = root / f"{label}.parquet"
            fixture(clock).to_parquet(path)
            expected = (
                "timestamps are not on the exact 10-second lattice"
                if label == "off_lattice"
                else "timestamps are unsorted or duplicated"
            )
            expect_rejected([path], expected)
        overlap = root / "overlap.parquet"
        fixture(
            pd.date_range("2026-01-01T00:00:20Z", periods=2, freq="10s")
        ).to_parquet(overlap)
        expect_rejected([safe_a, overlap], "files overlap or are out of order")

        cached_path = root / "USDJPY_2026.parquet"
        cached_index = pd.DatetimeIndex(
            [
                "2025-12-31T23:58:00Z",
                "2025-12-31T23:59:00Z",
                "2026-03-31T23:58:00Z",
                "2026-03-31T23:59:00Z",
                "2026-04-01T00:00:00Z",
            ],
            name="datetime_utc",
        ).as_unit("ns")
        pd.DataFrame(
            {
                "1m_ret_1": np.arange(len(cached_index), dtype="float32"),
                "y": np.asarray(["DO_NOT_DECODE"] * len(cached_index), dtype=object),
            },
            index=cached_index,
        ).to_parquet(cached_path)
        cached = _read_cached_q1_covariates(cached_path, ("1m_ret_1",))
        if (
            tuple(cached.columns) != ("1m_ret_1",)
            or not np.array_equal(
                _datetime_index_ns(cached.index, name="self-check cached clock"),
                _datetime_index_ns(
                    cached_index[[1, 2]], name="self-check expected cached clock"
                ),
            )
        ):
            raise AuthorityError("cached Q1 predicate/projection self-check differs")

    checks = {
        "finite_eligibility_exact": True,
        "common_probability_bitwise_exact": True,
        "common_direction_exact": True,
        "loadedbook_scalar_probes_exact": True,
        "common_gate_components_exact": True,
        "common_ordered_pre_schedule_exact": True,
        "full_population_ordered_pre_schedule_exact": True,
        "live_feature_builder_exact": True,
    }
    if _behavior_verdict(checks) != "PASS_BEHAVIOR_AUTHORITY":
        raise AuthorityError("passing behavior verdict self-check differs")
    for name in checks:
        mutated = dict(checks)
        mutated[name] = False
        expected = (
            "STOP_PHASE_ZERO_SCOREABILITY_MISMATCH"
            if name == "finite_eligibility_exact"
            else "STOP_PHASE_ZERO_BEHAVIOR_MISMATCH"
        )
        if _behavior_verdict(mutated) != expected:
            raise AuthorityError(f"fail-closed behavior self-check differs for {name}")

    acceptance_checks = {
        "outcome_free_score_adapter_exact": True,
        "finite_eligibility_exact": True,
        "common_probability_bitwise_exact": True,
        "common_direction_exact": True,
        "loadedbook_scalar_probes_exact": True,
        "common_gate_components_exact": True,
        "common_ordered_pre_schedule_exact": True,
        "full_population_ordered_pre_schedule_exact": True,
        "provider_state_exact": True,
    }
    if _acceptance_verdict(acceptance_checks) != "PASS_BEHAVIOR_AUTHORITY":
        raise AuthorityError("passing acceptance verdict self-check differs")
    for name in acceptance_checks:
        mutated = dict(acceptance_checks)
        mutated[name] = False
        expected = (
            "STOP_PHASE_ZERO_SCOREABILITY_MISMATCH"
            if name == "finite_eligibility_exact"
            else "STOP_PHASE_ZERO_BEHAVIOR_MISMATCH"
        )
        if _acceptance_verdict(mutated) != expected:
            raise AuthorityError(
                f"fail-closed acceptance verdict self-check differs for {name}"
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("diagnostic", "acceptance"), default="diagnostic"
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-check", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.self_check:
        _self_check()
        print("[phase-zero-authority] self-check passed", flush=True)
        return
    output = args.output
    if output is None:
        output = (
            ACCEPTANCE_OUTPUT_PATH
            if args.mode == "acceptance"
            else OUTPUT_PATH
        )
    if not output.is_absolute():
        output = REPO_ROOT / output
    report = (
        generate_acceptance(output)
        if args.mode == "acceptance"
        else generate(output)
    )
    print(
        f"[phase-zero-authority] wrote {output} "
        f"sha256={_sha256_file(output)} pairs={len(report['pairs'])}",
        flush=True,
    )


if __name__ == "__main__":
    main()
