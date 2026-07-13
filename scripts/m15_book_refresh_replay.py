#!/usr/bin/env python3
"""Measured common-replay computation for the issue-#9 book refresh.

This module owns the bounded part of the replay that sits between the sealed
arm artifacts and publication:

* strict, non-repairing loading of the processed 10-second observation clock;
* deterministic scoring of the ordered A/B/C and negative-control ensembles;
* construction of each pair's full common eligible decision grid;
* imported settlement, runtime-session, and combined chronological scheduling;
* the fixed 24-endpoint and nine-endpoint represented-date families; and
* issue-#9 diagnostics and terminal-status inputs.

It deliberately performs no work at import time and never writes a file.  The
campaign runner remains responsible for access-order seals, identities,
read-only evidence, and candidate publication.  Feature construction is also
outside this module: callers pass :class:`m15_book_refresh_adapters.ScoreRows`
from the sealed outcome-free snapshot.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from m15_book_refresh_adapters import (
    AdapterError,
    FEATURE_SOURCE_PAIR_ORDER,
    PolicyCalibration,
    ScoreRows,
    build_score_rows,
    policy_selection_mask,
    slice_score_rows,
)
from m15_book_refresh_stats import (
    CONTROL_FAMILY_SIZE,
    MAIN_FAMILY_SIZE,
    PAIR_ORDER,
    DateRatioEndpoint,
    ScheduledArm,
    assign_terminal_status,
    canonical_runtime_ny_mask,
    canonical_settlement_with_exit_on_decision_grid,
    endpoint_from_rows,
    epoch_nanoseconds_to_seconds,
    infer_control_family,
    infer_main_family,
    master_calendar_from_endpoints,
    ny_date_keys,
    scheduled_arm,
    validate_epoch_nanoseconds,
    validate_epoch_seconds,
)


NS_PER_SECOND = 1_000_000_000
PROCESSED_BAR_CLOSE_SHIFT_SECONDS = 10
PRIMARY_ARMS = ("A", "B", "C")
CONTROL_ARMS = ("B_perm", "C_perm")
REPORT_MONTHS = (("April", 4), ("May", 5))
RAW_OBSERVATION_SCHEMA = (
    "open", "high", "low", "close", "volume", "datetime_utc"
)
RAW_OBSERVATION_VALUE_TYPES = (
    "double", "double", "double", "double", "double"
)
RAW_OBSERVATION_CLOCK_TYPES = (
    "timestamp[ns, tz=UTC]",
    "timestamp[us, tz=UTC]",
)


class ReplayInputError(ValueError):
    """A replay input violates a sealed clock, score, or campaign contract."""


@dataclass(frozen=True)
class ObservationSeries:
    """Processed closes on their causal right-edge observation timestamps."""

    timestamps_s: np.ndarray
    closes: np.ndarray
    source_files: tuple[dict[str, Any], ...]

    def __post_init__(self) -> None:
        ts = validate_epoch_seconds(
            self.timestamps_s, name="observation_series.timestamps_s"
        )
        close = np.asarray(self.closes)
        if close.ndim != 1 or len(close) != len(ts):
            raise ReplayInputError("observation close vector is not clock-aligned")
        if not np.issubdtype(close.dtype, np.number) or np.issubdtype(
            close.dtype, np.bool_
        ):
            raise ReplayInputError("observation closes must have a numeric dtype")
        close = np.asarray(close, dtype="float64")
        if not np.isfinite(close).all() or np.any(close <= 0):
            raise ReplayInputError(
                "observation closes must be finite and strictly positive"
            )
        if not isinstance(self.source_files, tuple) or not self.source_files:
            raise ReplayInputError("observation source-file inventory is empty")
        ts.setflags(write=False)
        close.setflags(write=False)
        object.__setattr__(self, "timestamps_s", ts)
        object.__setattr__(self, "closes", close)


@dataclass(frozen=True)
class ArmPolicy:
    """Sealed model-owned state needed to score and gate one replay arm.

    ``model_paths`` is already in the declared ensemble seed order.  It is
    never sorted here.  ``bundle_id`` is carried into evidence but is not
    recomputed; identity verification belongs to the arm-seal phase.
    """

    arm: str
    bundle_id: str
    feature_cols: tuple[str, ...]
    model_paths: tuple[Path, ...]
    model_sha256: tuple[str, ...]
    confidence_threshold: float
    target_coverage: float
    structural_column: str | None = None
    structural_quantile: float | None = None
    structural_threshold: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.arm, str) or not self.arm:
            raise ReplayInputError("arm name must be non-empty")
        if not _is_hex(self.bundle_id, 64):
            raise ReplayInputError(f"{self.arm}: bundle_id must be 64 lowercase hex")
        cols = tuple(str(value) for value in self.feature_cols)
        if not cols or len(cols) != len(set(cols)):
            raise ReplayInputError(
                f"{self.arm}: feature columns are empty or duplicated"
            )
        paths = tuple(Path(value) for value in self.model_paths)
        if not paths:
            raise ReplayInputError(f"{self.arm}: model path vector is empty")
        if len({str(value) for value in paths}) != len(paths):
            raise ReplayInputError(f"{self.arm}: model path vector contains duplicates")
        model_sha256 = tuple(self.model_sha256)
        if (
            len(model_sha256) != len(paths)
            or any(not _is_hex(value, 64) for value in model_sha256)
        ):
            raise ReplayInputError(
                f"{self.arm}: model SHA-256 vector is missing/misaligned"
            )
        threshold = _finite_float(
            self.confidence_threshold, f"{self.arm}.confidence_threshold"
        )
        coverage = _finite_float(self.target_coverage, f"{self.arm}.target_coverage")
        if threshold < 0 or not 0 < coverage <= 1:
            raise ReplayInputError(
                f"{self.arm}: invalid confidence threshold or coverage"
            )
        if (self.structural_column is None) != (self.structural_threshold is None):
            raise ReplayInputError(f"{self.arm}: incomplete structural-gate state")
        if self.structural_column is not None:
            if self.structural_column not in cols:
                raise ReplayInputError(
                    f"{self.arm}: structural column is absent from the feature schema"
                )
            _finite_float(self.structural_threshold, f"{self.arm}.structural_threshold")
            if self.structural_quantile is not None:
                quantile = _finite_float(
                    self.structural_quantile, f"{self.arm}.structural_quantile"
                )
                if not 0 <= quantile <= 1:
                    raise ReplayInputError(
                        f"{self.arm}: structural quantile is outside [0,1]"
                    )
        object.__setattr__(self, "feature_cols", cols)
        object.__setattr__(self, "model_paths", paths)
        object.__setattr__(self, "model_sha256", model_sha256)
        object.__setattr__(self, "confidence_threshold", threshold)
        object.__setattr__(self, "target_coverage", coverage)


@dataclass(frozen=True)
class ComparatorARepeatability:
    """Exact outcome-blind comparator-A behavior sealed by arm construction."""

    book_id: str
    bundle_id: str
    rows: int
    scoreable_rows: int
    selected_rows: int
    entry_ns_sha256: str
    probability_f64le_sha256: str
    scoreable_mask_u8_sha256: str
    confidence_mask_u8_sha256: str
    structural_mask_u8_sha256: str
    pre_schedule_mask_u8_sha256: str
    ordered_selected_entry_ns_sha256: str
    threshold: float
    structural_column: str | None
    structural_threshold: float | None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ComparatorARepeatability":
        required = tuple(cls.__dataclass_fields__)
        if not isinstance(value, Mapping) or set(value) != set(required):
            raise ReplayInputError(
                "comparator-A repeatability fields differ from the sealed contract"
            )
        book_id = value.get("book_id")
        bundle_id = value.get("bundle_id")
        if not isinstance(book_id, str) or not book_id or not _is_hex(bundle_id, 64):
            raise ReplayInputError("comparator-A repeatability identity is malformed")
        counts: dict[str, int] = {}
        for name in ("rows", "scoreable_rows", "selected_rows"):
            item = value.get(name)
            if isinstance(item, bool) or not isinstance(item, int) or item < 0:
                raise ReplayInputError(
                    f"comparator-A repeatability {name} is malformed"
                )
            counts[name] = item
        if (
            counts["rows"] <= 0
            or counts["scoreable_rows"] > counts["rows"]
            or counts["selected_rows"] > counts["scoreable_rows"]
        ):
            raise ReplayInputError("comparator-A repeatability counts are incoherent")
        hash_names = (
            "entry_ns_sha256",
            "probability_f64le_sha256",
            "scoreable_mask_u8_sha256",
            "confidence_mask_u8_sha256",
            "structural_mask_u8_sha256",
            "pre_schedule_mask_u8_sha256",
            "ordered_selected_entry_ns_sha256",
        )
        if any(not _is_hex(value.get(name), 64) for name in hash_names):
            raise ReplayInputError("comparator-A repeatability hash is malformed")
        threshold = _finite_float(value.get("threshold"), "comparator_A.threshold")
        if threshold < 0:
            raise ReplayInputError("comparator-A threshold is negative")
        structural_column = value.get("structural_column")
        structural_threshold_value = value.get("structural_threshold")
        if (structural_column is None) != (structural_threshold_value is None):
            raise ReplayInputError("comparator-A structural gate is incomplete")
        if structural_column is not None and (
            not isinstance(structural_column, str) or not structural_column
        ):
            raise ReplayInputError("comparator-A structural column is malformed")
        structural_threshold = (
            None
            if structural_threshold_value is None
            else _finite_float(
                structural_threshold_value, "comparator_A.structural_threshold"
            )
        )
        return cls(
            book_id=book_id,
            bundle_id=str(bundle_id),
            rows=counts["rows"],
            scoreable_rows=counts["scoreable_rows"],
            selected_rows=counts["selected_rows"],
            entry_ns_sha256=str(value["entry_ns_sha256"]),
            probability_f64le_sha256=str(value["probability_f64le_sha256"]),
            scoreable_mask_u8_sha256=str(value["scoreable_mask_u8_sha256"]),
            confidence_mask_u8_sha256=str(value["confidence_mask_u8_sha256"]),
            structural_mask_u8_sha256=str(value["structural_mask_u8_sha256"]),
            pre_schedule_mask_u8_sha256=str(value["pre_schedule_mask_u8_sha256"]),
            ordered_selected_entry_ns_sha256=str(
                value["ordered_selected_entry_ns_sha256"]
            ),
            threshold=threshold,
            structural_column=structural_column,
            structural_threshold=structural_threshold,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            name: getattr(self, name) for name in self.__dataclass_fields__
        }


@dataclass(frozen=True)
class PairReplayInput:
    """All sealed, outcome-free inputs for one pair's measured replay."""

    pair: str
    score_rows: ScoreRows
    observation_paths: tuple[Path, ...]
    primary_policies: Mapping[str, ArmPolicy]
    comparator_a_book: Any
    comparator_a_repeatability: ComparatorARepeatability
    observation_inventory: tuple[dict[str, Any], ...] = ()
    control_policies: Mapping[str, ArmPolicy] = field(default_factory=dict)
    invalid_reasons: tuple[str, ...] = ()
    whole_book_pass: bool = True

    def __post_init__(self) -> None:
        pair = str(self.pair).upper()
        if pair not in PAIR_ORDER or self.score_rows.pair != pair:
            raise ReplayInputError(f"unknown or mismatched replay pair {self.pair!r}")
        paths = tuple(Path(value) for value in self.observation_paths)
        if not paths or len({str(value) for value in paths}) != len(paths):
            raise ReplayInputError(
                f"{pair}: observation path list is empty or duplicated"
            )
        inventory = tuple(dict(value) for value in self.observation_inventory)
        if inventory and len(inventory) != len(paths):
            raise ReplayInputError(
                f"{pair}: observation inventory is not path-aligned"
            )
        primary = dict(self.primary_policies)
        if tuple(primary) != PRIMARY_ARMS:
            raise ReplayInputError(
                f"{pair}: primary policies must be ordered exactly {PRIMARY_ARMS}"
            )
        if any(primary[name].arm != name for name in PRIMARY_ARMS):
            raise ReplayInputError(
                f"{pair}: primary policy arm labels are inconsistent"
            )
        repeatability = self.comparator_a_repeatability
        if not isinstance(repeatability, ComparatorARepeatability):
            raise ReplayInputError(
                f"{pair}: comparator-A repeatability value is missing/malformed"
            )
        book = self.comparator_a_book
        if str(getattr(book, "pair", "")).upper() != pair:
            raise ReplayInputError(f"{pair}: comparator-A loaded-book pair differs")
        if repeatability.bundle_id != primary["A"].bundle_id:
            raise ReplayInputError(
                f"{pair}: comparator-A repeatability bundle differs from policy"
            )
        if repeatability.threshold != primary["A"].confidence_threshold:
            raise ReplayInputError(
                f"{pair}: comparator-A repeatability threshold differs from policy"
            )
        if (
            repeatability.structural_column != primary["A"].structural_column
            or repeatability.structural_threshold
            != primary["A"].structural_threshold
        ):
            raise ReplayInputError(
                f"{pair}: comparator-A repeatability structural gate differs from policy"
            )
        controls = dict(self.control_policies)
        if controls and tuple(controls) != CONTROL_ARMS:
            raise ReplayInputError(
                f"{pair}: control policies must be ordered exactly {CONTROL_ARMS}"
            )
        if any(controls[name].arm != name for name in controls):
            raise ReplayInputError(
                f"{pair}: control policy arm labels are inconsistent"
            )
        score_schema = tuple(self.score_rows.feature_cols)
        for name, policy in {**primary, **controls}.items():
            if policy.feature_cols != score_schema:
                raise ReplayInputError(
                    f"{pair} {name}: policy feature order differs from the common score matrix"
                )
        if not isinstance(self.whole_book_pass, (bool, np.bool_)):
            raise ReplayInputError(f"{pair}: whole_book_pass must be boolean")
        object.__setattr__(self, "pair", pair)
        object.__setattr__(self, "observation_paths", paths)
        object.__setattr__(self, "observation_inventory", inventory)
        object.__setattr__(self, "primary_policies", primary)
        object.__setattr__(self, "comparator_a_repeatability", repeatability)
        object.__setattr__(self, "control_policies", controls)
        object.__setattr__(
            self,
            "invalid_reasons",
            tuple(str(value) for value in self.invalid_reasons if str(value)),
        )
        object.__setattr__(self, "whole_book_pass", bool(self.whole_book_pass))


@dataclass(frozen=True)
class InvalidPairReplayInput:
    """Sealed pair-local defect that reserves the pair's fixed endpoint slots."""

    pair: str
    adapter_family: str
    primary_bundle_ids: Mapping[str, str]
    control_bundle_ids: Mapping[str, str] = field(default_factory=dict)
    invalid_reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        pair = str(self.pair).upper()
        if pair not in PAIR_ORDER:
            raise ReplayInputError(f"unknown invalid-placeholder pair {self.pair!r}")
        primary = dict(self.primary_bundle_ids)
        controls = dict(self.control_bundle_ids)
        if tuple(primary) != PRIMARY_ARMS or any(
            not _is_hex(primary[arm], 64) for arm in PRIMARY_ARMS
        ):
            raise ReplayInputError(f"{pair}: invalid placeholder primary IDs differ")
        if controls and (
            tuple(controls) != CONTROL_ARMS
            or any(not _is_hex(controls[arm], 64) for arm in CONTROL_ARMS)
        ):
            raise ReplayInputError(f"{pair}: invalid placeholder control IDs differ")
        reasons = tuple(str(value) for value in self.invalid_reasons if str(value))
        if not reasons:
            raise ReplayInputError(f"{pair}: invalid placeholder has no defect reason")
        object.__setattr__(self, "pair", pair)
        object.__setattr__(self, "primary_bundle_ids", primary)
        object.__setattr__(self, "control_bundle_ids", controls)
        object.__setattr__(self, "invalid_reasons", reasons)


@dataclass(frozen=True)
class ScoredArm:
    policy: ArmPolicy
    probabilities: np.ndarray
    pre_schedule_mask: np.ndarray
    schedule: ScheduledArm


@dataclass(frozen=True)
class ComparatorAScoreCache:
    """Arm-seal-validated A score reused without a second model read."""

    entry_ns_sha256: str
    probabilities: np.ndarray
    pre_schedule_mask: np.ndarray

    def __post_init__(self) -> None:
        probabilities = np.asarray(self.probabilities, dtype="float64")
        selected = np.asarray(self.pre_schedule_mask)
        if (
            not _is_hex(self.entry_ns_sha256, 64)
            or probabilities.ndim != 1
            or selected.ndim != 1
            or selected.dtype != np.bool_
            or selected.shape != probabilities.shape
        ):
            raise ReplayInputError("comparator-A score cache is malformed")
        probabilities.setflags(write=False)
        selected.setflags(write=False)
        object.__setattr__(self, "probabilities", probabilities)
        object.__setattr__(self, "pre_schedule_mask", selected)


@dataclass(frozen=True)
class PairComputation:
    pair: str
    timestamps_s: np.ndarray
    returns: np.ndarray
    dates: np.ndarray
    primary: Mapping[str, ScoredArm]
    controls: Mapping[str, ScoredArm]
    main_endpoints: Mapping[str, DateRatioEndpoint]
    control_endpoints: Mapping[str, DateRatioEndpoint]
    report: dict[str, Any]


@dataclass(frozen=True)
class ReplayCampaignResult:
    """Separate JSON-ready pair and joint result bodies for runner sealing."""

    pair_results: Mapping[str, dict[str, Any]]
    joint_result: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        value = {
            "schema": "m15-book-refresh-replay-campaign/v1",
            "pair_results": dict(self.pair_results),
            "joint_result": dict(self.joint_result),
        }
        _assert_json_serializable(value)
        return value


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ReplayInputError(f"cannot canonically encode binding value: {exc}") from exc


def _length_prefixed_digest(domain: str, components: Sequence[bytes]) -> str:
    if not isinstance(domain, str) or not domain:
        raise ReplayInputError("binding identity domain is malformed")
    digest = hashlib.sha256()
    for item in [domain.encode("utf-8"), *components]:
        if not isinstance(item, bytes):
            raise ReplayInputError("binding identity component is not bytes")
        digest.update(len(item).to_bytes(8, "big", signed=False))
        digest.update(item)
    return digest.hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_run_binding(
    spec: Mapping[str, Any],
    workspace: Path,
    binding: Mapping[str, Any],
    *,
    requested_prereg_id: str | None,
    requested_run_id: str | None,
) -> tuple[str, str, str, dict[tuple[str, str], str], dict[tuple[str, str], str]]:
    """Authenticate every identity-bearing field before any pair is scored."""

    required = {
        "schema",
        "run_id",
        "prereg_id",
        "implementation_git_sha",
        "evaluator_id",
        "controls_id",
        "control_arms",
        "primary_arms",
        "canonical_encoding",
        "predecessor_manifest_hashes",
        "semantic_replay_outcomes_accessed",
    }
    if set(binding) != required:
        raise ReplayInputError("run binding fields differ")
    bound_prereg = binding.get("prereg_id")
    bound_run = binding.get("run_id")
    implementation = binding.get("implementation_git_sha")
    if (
        binding.get("schema") != "m15-book-refresh-run-binding/v1"
        or binding.get("semantic_replay_outcomes_accessed") is not False
        or not _is_hex(bound_prereg, 64)
        or not _is_hex(bound_run, 64)
        or not _is_hex(implementation, 40)
        or not _is_hex(binding.get("evaluator_id"), 64)
        or not _is_hex(binding.get("controls_id"), 64)
        or binding.get("canonical_encoding") != spec["identity"]
    ):
        raise ReplayInputError("run binding identity/encoding contract differs")
    if requested_prereg_id is not None and requested_prereg_id != bound_prereg:
        raise ReplayInputError("requested prereg_id differs from run binding")
    if requested_run_id is not None and requested_run_id != bound_run:
        raise ReplayInputError("requested run_id differs from run binding")

    prereg = _read_json_object(workspace / "source_preregistration.json")
    if (
        prereg.get("prereg_id") != bound_prereg
        or prereg.get("implementation_git_sha") != implementation
        or prereg.get("evaluator_id") != binding["evaluator_id"]
        or prereg.get("canonical_encoding") != spec["identity"]
    ):
        raise ReplayInputError("source preregistration differs from run binding")
    expected_evaluator = _length_prefixed_digest(
        str(spec["identity"]["domains"]["evaluator_id"]),
        [_canonical_bytes(spec["evaluator"])],
    )
    if binding["evaluator_id"] != expected_evaluator:
        raise ReplayInputError("run binding evaluator_id does not match evaluator bytes")

    expected_primary = [
        [pair, arm]
        for pair in spec["pair_order"]
        for arm in spec["arm_order"]
    ]
    primary_rows = binding.get("primary_arms")
    if (
        not isinstance(primary_rows, list)
        or len(primary_rows) != len(expected_primary)
        or any(
            not isinstance(row, list)
            or len(row) != 3
            or row[:2] != expected
            or not _is_hex(row[2], 64)
            for row, expected in zip(primary_rows, expected_primary)
        )
    ):
        raise ReplayInputError("run binding primary-arm order/identity differs")
    expected_controls = [list(value) for value in spec["control_order"]]
    control_rows = binding.get("control_arms")
    if (
        not isinstance(control_rows, list)
        or len(control_rows) != len(expected_controls)
        or any(
            not isinstance(row, list)
            or len(row) != 3
            or row[:2] != expected
            or not _is_hex(row[2], 64)
            for row, expected in zip(control_rows, expected_controls)
        )
    ):
        raise ReplayInputError("run binding control-arm order/identity differs")
    expected_controls_id = _length_prefixed_digest(
        str(spec["identity"]["domains"]["controls_id"]),
        [_canonical_bytes(control_rows)],
    )
    if binding["controls_id"] != expected_controls_id:
        raise ReplayInputError("run binding controls_id is not derived from control arms")
    expected_run_id = _length_prefixed_digest(
        str(spec["identity"]["domains"]["run_id"]),
        [
            str(bound_prereg).encode("ascii"),
            str(binding["controls_id"]).encode("ascii"),
            _canonical_bytes(primary_rows),
        ],
    )
    if bound_run != expected_run_id:
        raise ReplayInputError("run binding run_id is not derived from sealed arm identities")

    predecessor_names = (
        "00_source_preregistration.json",
        "01_derived_data.json",
        "02_adapter_parity.json",
        "03_fit_and_repeat.json",
    )
    expected_predecessors: dict[str, str] = {}
    for name in predecessor_names:
        phase_path = workspace / "phases" / name
        if phase_path.is_symlink() or not phase_path.is_file():
            raise ReplayInputError(f"run-binding predecessor is missing: {name}")
        expected_predecessors[name[3:-5]] = _sha256_file(phase_path)
    if binding.get("predecessor_manifest_hashes") != expected_predecessors:
        raise ReplayInputError("run binding predecessor-manifest hashes differ")
    primary_ids = {(row[0], row[1]): row[2] for row in primary_rows}
    control_ids = {(row[0], row[1]): row[2] for row in control_rows}
    return str(bound_prereg), str(bound_run), str(implementation), primary_ids, control_ids


def build_measured_pair_inputs(
    spec: Mapping[str, Any],
    workspace: str | Path,
    books: Mapping[str, Any],
    *,
    prereg_id: str | None = None,
    run_id: str | None = None,
    arm_seal_output_hashes: Mapping[str, str],
    processed_root: str | Path = "/home/sean/git/processed",
    fit_result_loader: Callable[[Path, str, str], Mapping[str, Any]] | None = None,
) -> dict[str, PairReplayInput | InvalidPairReplayInput]:
    """Resolve sealed workspace artifacts into the six measured target inputs.

    This is the runner-facing integration boundary.  It reads only when
    explicitly called, uses the outcome-free replay feature view, preserves
    model/seed order from the sealed fit results, and derives no settlement
    outcome or summary itself.  The runner must already have authenticated the
    phase manifests; this helper cross-checks their arm identities but does not
    duplicate phase-store verification.
    """

    # Imported lazily to keep this pure computation module free of runner work
    # at import time while still authenticating the exact routed covariates the
    # runner sealed before arm construction.
    from m15_book_refresh import (
        PairRouteError,
        RefreshError,
        _feature_view_consumer_paths,
        _load_feature_view_routing_manifest,
        verified_feature_view_access,
    )

    _validate_replay_spec(spec)
    root = Path(workspace)
    if root.is_symlink() or not root.is_dir():
        raise ReplayInputError(f"workspace is not a regular directory: {root}")
    binding_paths = tuple(sorted((root / "arm_seal").glob("run_binding_*.json")))
    if len(binding_paths) != 1:
        raise ReplayInputError("workspace must contain exactly one sealed run binding")
    binding = _read_arm_seal_json(
        root, binding_paths[0], arm_seal_output_hashes
    )
    (
        bound_prereg,
        bound_run,
        bound_implementation,
        primary_ids,
        control_ids,
    ) = _validate_run_binding(
        spec,
        root,
        binding,
        requested_prereg_id=prereg_id,
        requested_run_id=run_id,
    )
    repeat_path = root / "arm_seal" / f"repeatability_{bound_run}.json"
    repeatability = _read_arm_seal_json(
        root, repeat_path, arm_seal_output_hashes
    )
    if (
        set(repeatability)
        != {
            "schema",
            "prereg_id",
            "semantic_replay_outcomes_accessed",
            "pairs",
            "controls",
            "run_id",
            "controls_id",
        }
        or repeatability.get("schema") != "m15-book-refresh-repeatability/v1"
        or repeatability.get("prereg_id") != bound_prereg
        or repeatability.get("run_id") != bound_run
        or repeatability.get("controls_id") != binding.get("controls_id")
        or repeatability.get("semantic_replay_outcomes_accessed") is not False
        or not isinstance(repeatability.get("pairs"), Mapping)
        or set(repeatability["pairs"]) != set(PAIR_ORDER)
        or not isinstance(repeatability.get("controls"), Mapping)
    ):
        raise ReplayInputError("repeatability result differs from run binding")
    data_lock = _read_json_object(root / "feature_snapshot" / "data_lock.json")
    if data_lock.get("prereg_id") != bound_prereg:
        raise ReplayInputError("derived-data lock differs from run binding")
    # A corrupt shared routing authority is a campaign defect, not six pair
    # defects.  Authenticate it once outside every pair-local exception fence.
    _load_feature_view_routing_manifest(root)

    source_preregistration = _read_json_object(
        root / "source_preregistration.json"
    )
    sealed_a_files = source_preregistration.get("comparator_A_files")
    if (
        source_preregistration.get("prereg_id") != bound_prereg
        or not isinstance(sealed_a_files, list)
        or any(not isinstance(row, Mapping) for row in sealed_a_files)
    ):
        raise ReplayInputError("source preregistration comparator-A inventory differs")
    sealed_a_by_path = {row.get("path"): row for row in sealed_a_files}
    if len(sealed_a_by_path) != len(sealed_a_files) or None in sealed_a_by_path:
        raise ReplayInputError("source comparator-A inventory paths are malformed")

    book_map = dict(books)
    if tuple(book_map) != tuple(PAIR_ORDER):
        raise ReplayInputError(
            "loaded-book map does not preserve the exact six-target replay order"
        )
    loader = fit_result_loader or _default_fit_result_loader
    feature_dir = root / "feature_views" / "features"
    orderflow_dir = root / "feature_views" / "features_of"
    output: dict[str, PairReplayInput | InvalidPairReplayInput] = {}
    representative_family = {
        pair: family
        for family, pair in spec["control_family"]["representatives"].items()
    }

    def pair_fit_result(pair: str, arm: str) -> dict[str, Any]:
        try:
            return dict(loader(root, pair, arm))
        except RefreshError as exc:
            raise ReplayInputError(
                f"{pair} {arm}: sealed fit authentication failed: {exc}"
            ) from exc

    def comparator_a_model_sha256(book: Any) -> tuple[str, ...]:
        values: list[str] = []
        for model_path_value in getattr(book, "model_paths", ()):
            model_path = Path(model_path_value)
            expected_path = f"books/{book.book_id}/{model_path.name}"
            sealed = sealed_a_by_path.get(expected_path)
            if (
                not isinstance(sealed, Mapping)
                or set(sealed) != {"path", "bytes", "sha256"}
                or not _is_hex(sealed.get("sha256"), 64)
                or model_path.is_symlink()
                or not model_path.is_file()
                or model_path.stat().st_size != sealed.get("bytes")
                or _sha256_file(model_path) != sealed.get("sha256")
            ):
                raise ReplayInputError(
                    f"{book.pair}: comparator-A model differs from preregistration: "
                    f"{expected_path}"
                )
            values.append(str(sealed["sha256"]))
        if not values:
            raise ReplayInputError(f"{book.pair}: comparator-A model inventory is empty")
        return tuple(values)

    def resolve_pair(pair: str) -> PairReplayInput:
        book = book_map[pair]
        if str(getattr(book, "pair", "")) != pair:
            raise ReplayInputError(f"{pair}: loaded-book pair mismatch")
        feature_cols = tuple(str(value) for value in getattr(book, "feature_cols", ()))
        consumer_paths = _feature_view_consumer_paths(
            root,
            pair,
            ["2026"],
            feature_dir=feature_dir,
            orderflow_dir=orderflow_dir,
        )
        with verified_feature_view_access(root, consumer_paths):
            rows = build_score_rows(
                pair,
                ["2026"],
                feature_cols,
                feature_dir=feature_dir,
                orderflow_dir=orderflow_dir,
            )
        replay_split = spec["splits"]["replay"]
        rows = slice_score_rows(
            rows,
            entry_at_or_after=replay_split["entry_at_or_after"],
            entry_before=replay_split["entry_before"],
        )
        a_repeat = repeatability.get("pairs", {}).get(pair, {}).get("A")
        if not isinstance(a_repeat, Mapping):
            raise ReplayInputError(
                f"{pair}: repeatability result is missing comparator A"
            )
        if a_repeat.get("bundle_id") != primary_ids[(pair, "A")]:
            raise ReplayInputError(
                f"{pair}: comparator-A bundle differs from run binding"
            )
        a_repeatability = ComparatorARepeatability.from_mapping(a_repeat)
        a_structural_column, a_structural_threshold = (
            _loaded_book_structural_gate(book)
        )
        coverage = getattr(book, "coverage", None)
        if coverage is None:
            coverage = spec["pairs"][pair]["calibration_target_coverage"]
        primary: dict[str, ArmPolicy] = {
            "A": ArmPolicy(
                arm="A",
                bundle_id=primary_ids[(pair, "A")],
                feature_cols=feature_cols,
                model_paths=tuple(
                    Path(value) for value in getattr(book, "model_paths", ())
                ),
                model_sha256=comparator_a_model_sha256(book),
                confidence_threshold=float(getattr(book, "conf_thr")),
                target_coverage=float(coverage),
                structural_column=(
                    a_structural_column
                ),
                structural_quantile=None,
                structural_threshold=(
                    float(a_structural_threshold)
                    if a_structural_threshold is not None
                    else None
                ),
            )
        }
        for arm in ("B", "C"):
            result = pair_fit_result(pair, arm)
            primary[arm] = _policy_from_fit_result(
                root,
                pair,
                arm,
                result,
                spec=spec,
                expected_bundle_id=primary_ids[(pair, arm)],
                expected_prereg_id=str(bound_prereg),
                expected_implementation_git_sha=str(bound_implementation),
            )
        controls: dict[str, ArmPolicy] = {}
        if pair in representative_family:
            family = representative_family[pair]
            for arm in CONTROL_ARMS:
                result = pair_fit_result(pair, arm)
                controls[arm] = _policy_from_fit_result(
                    root,
                    pair,
                    arm,
                    result,
                    spec=spec,
                    expected_bundle_id=control_ids[(family, arm)],
                    expected_prereg_id=str(bound_prereg),
                    expected_implementation_git_sha=str(bound_implementation),
                )
        observation_paths = discover_observation_files(
            pair,
            processed_root=processed_root,
            nominal_year="2026",
            observation_at_or_after=replay_split["entry_at_or_after"],
            observation_before=replay_split["settlement_exit_before"],
        )
        observation_inventory = _verify_observation_inventory(
            pair,
            observation_paths,
            data_lock.get("processed_sources", {}).get(pair),
        )
        return PairReplayInput(
            pair=pair,
            score_rows=rows,
            observation_paths=observation_paths,
            primary_policies=primary,
            comparator_a_book=book,
            comparator_a_repeatability=a_repeatability,
            observation_inventory=observation_inventory,
            control_policies=controls,
        )

    for pair in PAIR_ORDER:
        family = spec["pairs"][pair]["adapter_family"]
        pair_primary_ids = {
            arm: primary_ids[(pair, arm)] for arm in PRIMARY_ARMS
        }
        pair_control_ids = (
            {arm: control_ids[(family, arm)] for arm in CONTROL_ARMS}
            if pair in representative_family
            else {}
        )
        try:
            output[pair] = resolve_pair(pair)
        except (ReplayInputError, AdapterError, PairRouteError) as exc:
            output[pair] = InvalidPairReplayInput(
                pair=pair,
                adapter_family=family,
                primary_bundle_ids=pair_primary_ids,
                control_bundle_ids=pair_control_ids,
                invalid_reasons=(
                    f"pair_input_defect:{type(exc).__name__}:{str(exc)}",
                ),
            )
    return output


def run_workspace_replay(
    spec: Mapping[str, Any],
    workspace: str | Path,
    books: Mapping[str, Any],
    *,
    prereg_id: str,
    run_id: str,
    arm_seal_output_hashes: Mapping[str, str],
    processed_root: str | Path = "/home/sean/git/processed",
    fit_result_loader: Callable[[Path, str, str], Mapping[str, Any]] | None = None,
    bootstrap_chunk_size: int = 512,
) -> ReplayCampaignResult:
    """High-level measured call used by the runner immediately after arm seal.

    The returned joint body is deliberately a draft until the runner prepares
    the status-authorized candidate stages and supplies their external receipt
    locks to :func:`finalize_joint_result`.
    """

    if not _is_hex(prereg_id, 64) or not _is_hex(run_id, 64):
        raise ReplayInputError(
            "prereg_id and run_id must be full lowercase SHA-256 IDs"
        )
    inputs = build_measured_pair_inputs(
        spec,
        workspace,
        books,
        prereg_id=prereg_id,
        run_id=run_id,
        arm_seal_output_hashes=arm_seal_output_hashes,
        processed_root=processed_root,
        fit_result_loader=fit_result_loader,
    )
    measured = run_common_replay(
        spec, inputs, bootstrap_chunk_size=bootstrap_chunk_size
    )
    pair_results = json.loads(json.dumps(measured.pair_results, allow_nan=False))
    for pair in PAIR_ORDER:
        pair_results[pair].update(
            prereg_id=prereg_id,
            run_id=run_id,
            candidate_book_id=spec["pairs"][pair]["candidate_book_id"],
        )
    joint = json.loads(json.dumps(measured.joint_result, allow_nan=False))
    candidate_statuses: dict[str, dict[str, Any]] = {}
    candidate_book_ids_by_pair: dict[str, str] = {}
    survivor_ids: list[str] = []
    for pair in PAIR_ORDER:
        candidate_id = str(spec["pairs"][pair]["candidate_book_id"])
        candidate_book_ids_by_pair[pair] = candidate_id
        decision = dict(joint["statuses"][pair])
        if decision["status"] == "PROMOTE_TO_SHADOW":
            survivor_ids.append(candidate_id)
            candidate_statuses[candidate_id] = {
                "pair": pair,
                "bundle_id": pair_results[pair]["arms"]["C"]["bundle_id"],
                "status": decision["status"],
                "reasons": list(decision["reasons"]),
            }
    joint.update(
        schema="m15-book-refresh-joint-replay-draft/v1",
        prereg_id=prereg_id,
        run_id=run_id,
        candidate_book_ids_by_pair=candidate_book_ids_by_pair,
        candidate_statuses=candidate_statuses,
        S=survivor_ids,
        candidate_publish_locks=None,
        publication_lock_complete=False,
    )
    _assert_json_serializable(pair_results)
    _assert_json_serializable(joint)
    return ReplayCampaignResult(pair_results=pair_results, joint_result=joint)


def finalize_joint_result(
    result: ReplayCampaignResult,
    *,
    candidate_publish_locks: Mapping[str, Mapping[str, Any]],
) -> ReplayCampaignResult:
    """Purely bind exact candidate receipts to the status-bearing trust anchor.

    Candidate staging and receipt construction stay in ``manifest``/the runner;
    this helper only verifies that their locks are exactly the already-declared
    survivor set and emits the schema publication recovery is allowed to trust.
    """

    joint = json.loads(json.dumps(result.joint_result, allow_nan=False))
    if joint.get("schema") != "m15-book-refresh-joint-replay-draft/v1":
        raise ReplayInputError(
            "only an unfinalized workspace replay may be publication-locked"
        )
    prereg_id = joint.get("prereg_id")
    run_id = joint.get("run_id")
    if not _is_hex(prereg_id, 64) or not _is_hex(run_id, 64):
        raise ReplayInputError("draft joint result has malformed prereg_id/run_id")
    statuses = joint.get("candidate_statuses")
    if not isinstance(statuses, Mapping):
        raise ReplayInputError("draft joint result is missing candidate statuses")
    declared = list(joint.get("S", []))
    derived = [
        book_id
        for book_id, row in statuses.items()
        if isinstance(row, Mapping) and row.get("status") == "PROMOTE_TO_SHADOW"
    ]
    if declared != derived or len(declared) != len(set(declared)):
        raise ReplayInputError(
            "S does not equal the ordered PROMOTE_TO_SHADOW status set"
        )
    locks = dict(candidate_publish_locks)
    if tuple(locks) != tuple(declared):
        raise ReplayInputError(
            "candidate_publish_locks keys/order must equal S exactly"
        )
    for book_id in declared:
        lock = locks[book_id]
        if not isinstance(lock, Mapping):
            raise ReplayInputError(
                f"{book_id}: candidate publication lock is malformed"
            )
        if (
            set(lock)
            != {
                "schema",
                "destination",
                "book_id",
                "prereg_id",
                "run_id",
                "bundle_id",
                "receipt_sha256",
                "payload",
            }
            or lock.get("schema") != "candidate-publish-lock/v1"
        ):
            raise ReplayInputError(
                f"{book_id}: candidate publication lock fields differ"
            )
        if lock.get("destination") != f"books/{book_id}":
            raise ReplayInputError(
                f"{book_id}: candidate publication destination differs"
            )
        if lock.get("book_id") != book_id:
            raise ReplayInputError(
                f"{book_id}: candidate publication lock book ID differs"
            )
        if lock.get("prereg_id") != prereg_id or lock.get("run_id") != run_id:
            raise ReplayInputError(
                f"{book_id}: candidate publication lock run IDs differ"
            )
        if lock.get("bundle_id") != statuses[book_id].get("bundle_id"):
            raise ReplayInputError(
                f"{book_id}: candidate publication lock bundle ID differs"
            )
        if not _is_hex(lock.get("receipt_sha256"), 64):
            raise ReplayInputError(f"{book_id}: candidate receipt hash is malformed")
        payload = lock.get("payload")
        if (
            not isinstance(payload, Mapping)
            or not payload
            or any(
                not isinstance(name, str) or not _is_hex(digest, 64)
                for name, digest in payload.items()
            )
        ):
            raise ReplayInputError(
                f"{book_id}: candidate payload hash map is malformed"
            )
    joint.update(
        schema="m15-book-refresh-joint-replay-result/v1",
        candidate_publish_locks=locks,
        publication_lock_complete=True,
    )
    finalized = ReplayCampaignResult(
        pair_results=result.pair_results,
        joint_result=joint,
    )
    finalized.as_dict()
    return finalized


def discover_observation_files(
    pair: str,
    *,
    processed_root: str | Path = "/home/sean/git/processed",
    nominal_year: str = "2026",
    observation_at_or_after: str | pd.Timestamp | None = None,
    observation_before: str | pd.Timestamp | None = None,
) -> tuple[Path, ...]:
    """Return only schema-valid files intersecting the sealed close-clock window.

    Discovery projects the timestamp column alone.  It never reads a close or
    any other value while deciding whether a file can support settlement.
    """

    canonical = str(pair).upper()
    if canonical not in FEATURE_SOURCE_PAIR_ORDER:
        raise ReplayInputError(f"unknown pair {pair!r}")
    root = Path(processed_root) / canonical
    candidates = tuple(sorted(root.glob(f"{canonical}_10s_{nominal_year}-*.parquet")))
    if not candidates:
        raise ReplayInputError(
            f"{canonical}: no processed observation files for {nominal_year}"
        )
    if (observation_at_or_after is None) != (observation_before is None):
        raise ReplayInputError("observation discovery bounds must be supplied together")
    lower_ns = upper_ns = None
    if observation_at_or_after is not None:
        lower_ns = int(_utc_bound(observation_at_or_after, name="observation_at_or_after").value)
        upper_ns = int(_utc_bound(observation_before, name="observation_before").value)
        if lower_ns >= upper_ns:
            raise ReplayInputError("observation discovery window is empty/reversed")
    selected: list[Path] = []
    close_shift_ns = PROCESSED_BAR_CLOSE_SHIFT_SECONDS * NS_PER_SECOND
    for path in candidates:
        _validate_observation_schema(path)
        clock_only = _read_observation_projection(
            path,
            columns=("datetime_utc",),
            name="observation clock",
        )
        source_ns = _strict_utc_ns(clock_only["datetime_utc"], name=str(path))
        if lower_ns is None:
            selected.append(path)
            continue
        close_ns = source_ns + np.int64(close_shift_ns)
        if np.any((close_ns >= lower_ns) & (close_ns < upper_ns)):
            selected.append(path)
    if not selected:
        raise ReplayInputError(
            f"{canonical}: no processed observations intersect the sealed window"
        )
    return tuple(selected)


def _verify_observation_inventory(
    pair: str,
    paths: Sequence[Path],
    sealed_rows: Any,
) -> tuple[dict[str, Any], ...]:
    """Bind replay settlement bytes to the preregistered derived-data lock."""

    if not isinstance(sealed_rows, list) or not sealed_rows:
        raise ReplayInputError(
            f"{pair}: processed-source inventory is absent from data lock"
        )
    sealed_by_path: dict[str, Mapping[str, Any]] = {}
    sealed_order: list[str] = []
    for sealed in sealed_rows:
        if (
            not isinstance(sealed, Mapping)
            or set(sealed) != {"path", "bytes", "sha256"}
            or not isinstance(sealed.get("path"), str)
            or not isinstance(sealed.get("bytes"), int)
            or isinstance(sealed.get("bytes"), bool)
            or sealed["bytes"] < 0
            or not _is_hex(sealed.get("sha256"), 64)
            or sealed["path"] in sealed_by_path
        ):
            raise ReplayInputError(f"{pair}: malformed processed-source lock row")
        sealed_by_path[str(sealed["path"])] = sealed
        sealed_order.append(str(sealed["path"]))
    normalized: list[dict[str, Any]] = []
    selected_names: list[str] = []
    for path in paths:
        expected_path = f"processed/{pair}/{path.name}"
        sealed = sealed_by_path.get(expected_path)
        if sealed is None:
            raise ReplayInputError(
                f"{pair}: processed-source path was not present in data lock"
            )
        if (
            path.is_symlink()
            or not path.is_file()
            or sealed.get("bytes") != path.stat().st_size
        ):
            raise ReplayInputError(
                f"{pair}: processed-source path/size differs from data lock"
            )
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        if digest.hexdigest() != sealed.get("sha256"):
            raise ReplayInputError(
                f"{pair}: processed-source bytes differ from data lock"
            )
        selected_names.append(expected_path)
        normalized.append(dict(sealed))
    selected_positions = [sealed_order.index(name) for name in selected_names]
    if selected_positions != sorted(selected_positions):
        raise ReplayInputError(
            f"{pair}: processed-source subset order differs from data lock"
        )
    return tuple(normalized)


def _validate_observation_schema(path: Path) -> None:
    if path.is_symlink() or not path.is_file():
        raise ReplayInputError(f"observation source is not a regular file: {path}")
    try:
        schema = pq.ParquetFile(path).schema_arrow
    except (OSError, TypeError, ValueError, pa.ArrowException) as exc:
        raise ReplayInputError(f"cannot inspect observation schema {path}: {exc}") from exc
    if tuple(schema.names) != RAW_OBSERVATION_SCHEMA:
        raise ReplayInputError(
            f"{path}: processed schema must be exactly {RAW_OBSERVATION_SCHEMA}"
        )
    physical_types = tuple(str(field.type) for field in schema)
    if (
        physical_types[:-1] != RAW_OBSERVATION_VALUE_TYPES
        or physical_types[-1] not in RAW_OBSERVATION_CLOCK_TYPES
    ):
        raise ReplayInputError(f"{path}: processed physical types differ")


def _read_observation_projection(
    path: Path,
    *,
    columns: Sequence[str],
    name: str,
    filters: Sequence[tuple[str, str, Any]] | None = None,
) -> pd.DataFrame:
    """Read exact physical columns without reconstructing a pandas index.

    The processed files legitimately record ``datetime_utc`` as their pandas
    index in Arrow metadata while retaining it as the final physical Parquet
    field.  ``pandas.read_parquet`` consumes that metadata and silently removes
    the requested clock from ``DataFrame.columns``.  Reading the Arrow table
    and converting with ``ignore_metadata=True`` preserves the sealed physical
    projection and gives one unambiguous RangeIndex.
    """

    requested = list(columns)
    if not requested or len(set(requested)) != len(requested):
        raise ReplayInputError(f"{name} projection is empty or duplicated")
    try:
        table = pq.read_table(
            path,
            columns=requested,
            filters=list(filters) if filters is not None else None,
            use_threads=False,
            use_pandas_metadata=False,
        )
        frame = table.to_pandas(ignore_metadata=True, use_threads=False)
    except (OSError, TypeError, ValueError, pa.ArrowException) as exc:
        raise ReplayInputError(f"cannot read {name} {path}: {exc}") from exc
    if table.column_names != requested or list(frame.columns) != requested:
        raise ReplayInputError(f"{path}: {name} projection order differs")
    if not isinstance(frame.index, pd.RangeIndex) or not frame.index.equals(
        pd.RangeIndex(len(frame))
    ):
        raise ReplayInputError(f"{path}: {name} projection index is ambiguous")
    return frame


def _utc_bound(value: Any, *, name: str) -> pd.Timestamp:
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ReplayInputError(f"{name}: invalid UTC boundary") from exc
    if timestamp.tz is None or str(timestamp.tz) != "UTC" or pd.isna(timestamp):
        raise ReplayInputError(f"{name}: boundary must be timezone-aware UTC")
    return timestamp


def load_observation_series(
    paths: Sequence[str | Path],
    *,
    expected_inventory: Sequence[Mapping[str, Any]] | None = None,
    pair: str | None = None,
    observation_at_or_after: str | pd.Timestamp | None = None,
    observation_before: str | pd.Timestamp | None = None,
) -> ObservationSeries:
    """Load an exact raw observation clock, rejecting rather than repairing it.

    Source bars are left-labeled ``[label,label+10s)`` intervals.  Their close
    values are therefore placed on ``label+10s`` before settlement.  Unlike
    ``pipeline.load_pair_year_10s``, this function never sorts,
    deduplicates, drops undefined rows, or coerces malformed closes.  The caller
    supplies the sealed file order and any defect in or between files fails.
    """

    ordered = tuple(Path(value) for value in paths)
    if not ordered or len({str(value) for value in ordered}) != len(ordered):
        raise ReplayInputError("observation path list is empty or duplicated")
    if (observation_at_or_after is None) != (observation_before is None):
        raise ReplayInputError("observation load bounds must be supplied together")
    lower = upper = None
    if observation_at_or_after is not None:
        lower = _utc_bound(observation_at_or_after, name="observation_at_or_after")
        upper = _utc_bound(observation_before, name="observation_before")
        if lower >= upper:
            raise ReplayInputError("observation load window is empty/reversed")
    authenticated_before: tuple[dict[str, Any], ...] | None = None
    if expected_inventory is not None:
        if pair is None:
            raise ReplayInputError("expected observation inventory requires a pair")
        authenticated_before = _verify_observation_inventory(
            str(pair).upper(), ordered, list(expected_inventory)
        )
    clocks: list[np.ndarray] = []
    closes: list[np.ndarray] = []
    inventory: list[dict[str, Any]] = []
    for position, path in enumerate(ordered):
        _validate_observation_schema(path)
        filters = None
        if lower is not None and upper is not None:
            source_lower = lower - pd.Timedelta(
                seconds=PROCESSED_BAR_CLOSE_SHIFT_SECONDS
            )
            source_upper = upper - pd.Timedelta(
                seconds=PROCESSED_BAR_CLOSE_SHIFT_SECONDS
            )
            filters = [
                ("datetime_utc", ">=", source_lower.to_pydatetime()),
                ("datetime_utc", "<", source_upper.to_pydatetime()),
            ]
        frame = _read_observation_projection(
            path,
            columns=("datetime_utc", "close"),
            filters=filters,
            name="observation source",
        )
        if frame.empty:
            raise ReplayInputError(f"observation source is empty: {path}")
        raw_clock = frame["datetime_utc"]
        source_clock = _strict_utc_ns(raw_clock, name=str(path))
        if np.any(np.remainder(source_clock, 10 * NS_PER_SECOND) != 0):
            raise ReplayInputError(f"{path}: processed 10-second source labels are off-grid")
        close_shift_ns = np.int64(PROCESSED_BAR_CLOSE_SHIFT_SECONDS * NS_PER_SECOND)
        if np.any(source_clock > np.iinfo(np.int64).max - close_shift_ns):
            raise ReplayInputError(f"{path}: processed close-observation clock overflows")
        clock = source_clock + close_shift_ns
        if lower is not None and upper is not None:
            lower_ns = int(lower.value)
            upper_ns = int(upper.value)
            if np.any((clock < lower_ns) | (clock >= upper_ns)):
                raise ReplayInputError(
                    f"{path}: projected close clock escapes the sealed support window"
                )
        if "close" not in frame.columns:
            raise ReplayInputError(f"{path}: missing close column")
        raw_close = frame["close"].to_numpy(copy=False)
        if (
            raw_close.ndim != 1
            or not np.issubdtype(raw_close.dtype, np.number)
            or np.issubdtype(raw_close.dtype, np.bool_)
        ):
            raise ReplayInputError(
                f"{path}: close column is not one-dimensional numeric data"
            )
        close = np.asarray(raw_close, dtype="float64")
        if (
            len(close) != len(clock)
            or not np.isfinite(close).all()
            or np.any(close <= 0)
        ):
            raise ReplayInputError(
                f"{path}: close values are invalid or clock-misaligned"
            )
        clocks.append(clock)
        closes.append(close)
        inventory.append(
            {
                "source_identity": (
                    dict(authenticated_before[position])
                    if authenticated_before is not None
                    else {
                        "path": str(path),
                        "bytes": path.stat().st_size,
                        "sha256": _sha256_file(path),
                    }
                ),
                "position": position,
                "file": path.name,
                "rows": len(clock),
                "source_label_convention": "left_edge_of_[label,label+10s)_bar",
                "close_observation_shift_s": PROCESSED_BAR_CLOSE_SHIFT_SECONDS,
                "first_source_label_ns": int(source_clock[0]),
                "last_source_label_ns": int(source_clock[-1]),
                "first_close_observation_ns": int(clock[0]),
                "last_close_observation_ns": int(clock[-1]),
            }
        )
    combined_ns = np.concatenate(clocks)
    validate_epoch_nanoseconds(combined_ns, name="combined_observation_clock_ns")
    if np.any(np.remainder(combined_ns, NS_PER_SECOND) != 0):
        raise ReplayInputError("observation clock contains sub-second timestamps")
    combined_s = np.floor_divide(combined_ns, NS_PER_SECOND).astype("int64")
    validate_epoch_seconds(combined_s, name="combined_observation_clock_s")
    combined_close = np.concatenate(closes).astype("float64", copy=False)
    if authenticated_before is not None:
        authenticated_after = _verify_observation_inventory(
            str(pair).upper(), ordered, list(expected_inventory or ())
        )
        if authenticated_after != authenticated_before:
            raise ReplayInputError("processed-source identity changed during observation load")
    return ObservationSeries(combined_s, combined_close, tuple(inventory))


def _policy_model_identity(
    policy: ArmPolicy,
    position: int,
) -> tuple[int, int, int, int, str]:
    """Authenticate one declared model before/after any load or prediction."""

    path = policy.model_paths[position]
    expected = policy.model_sha256[position]
    if (
        path.is_symlink()
        or not path.is_file()
    ):
        raise ReplayInputError(
            f"{policy.arm}: model is missing or symlinked: {path}"
        )
    before = path.stat()
    digest = _sha256_file(path)
    after = path.stat()
    before_identity = (
        int(before.st_dev),
        int(before.st_ino),
        int(before.st_size),
        int(before.st_mtime_ns),
    )
    after_identity = (
        int(after.st_dev),
        int(after.st_ino),
        int(after.st_size),
        int(after.st_mtime_ns),
    )
    if before_identity != after_identity or digest != expected:
        raise ReplayInputError(
            f"{policy.arm}: model bytes changed or differ from sealed SHA-256: {path}"
        )
    return (*after_identity, digest)


def score_policy(rows: ScoreRows, policy: ArmPolicy) -> tuple[np.ndarray, np.ndarray]:
    """Score an ensemble in declared order and apply the canonical adapter gate."""

    if tuple(rows.feature_cols) != policy.feature_cols:
        raise ReplayInputError(
            f"{rows.pair} {policy.arm}: feature order differs from sealed policy"
        )
    matrix = np.ascontiguousarray(rows.X, dtype="float32")
    scoreable = np.isfinite(matrix).all(axis=1)
    probabilities = np.full(len(rows), np.nan, dtype="float64")
    model_predictions: list[np.ndarray] = []
    import lightgbm as lgb  # Delayed: importing this module never loads a model.

    for position, path in enumerate(policy.model_paths):
        before_identity = _policy_model_identity(policy, position)
        try:
            booster = lgb.Booster(model_file=str(path))
            if booster.num_feature() != len(policy.feature_cols):
                raise ReplayInputError(
                    f"{rows.pair} {policy.arm}: model feature count differs from sealed schema"
                )
            prediction = np.asarray(
                booster.predict(matrix[scoreable], num_threads=1), dtype="float64"
            )
        except ReplayInputError:
            raise
        except lgb.basic.LightGBMError as exc:
            raise ReplayInputError(
                f"{rows.pair} {policy.arm}: model load/prediction failed: {exc}"
            ) from exc
        if prediction.shape != (int(scoreable.sum()),):
            raise ReplayInputError(
                f"{rows.pair} {policy.arm}: model prediction is row-misaligned"
            )
        model_predictions.append(prediction)
        del booster
        after_identity = _policy_model_identity(policy, position)
        if after_identity != before_identity:
            raise ReplayInputError(
                f"{rows.pair} {policy.arm}: model identity changed during scoring: {path}"
            )
    probabilities[scoreable] = np.asarray(model_predictions, dtype="float64").mean(
        axis=0
    )
    finite = probabilities[scoreable]
    if not np.isfinite(finite).all() or np.any((finite < 0) | (finite > 1)):
        raise ReplayInputError(
            f"{rows.pair} {policy.arm}: non-finite/out-of-range probability"
        )

    # policy_selection_mask is the feature-family authority for the numeric
    # confidence and optional q33 structural gate.  The tiny proxy contains
    # precisely the fields that pure adapter helper consumes; no outcomes are
    # fabricated or exposed.
    row_proxy = SimpleNamespace(
        pair=rows.pair,
        fit_row_id=rows.entry_ns,
        feature_cols=rows.feature_cols,
        X=rows.X,
    )
    calibration = PolicyCalibration(
        pair=rows.pair,
        target_coverage=policy.target_coverage,
        confidence_threshold=policy.confidence_threshold,
        calibration_rows=0,
        confidence_rows=0,
        structural_rows=None,
        structural_column=policy.structural_column,
        structural_quantile=policy.structural_quantile,
        structural_threshold=policy.structural_threshold,
    )
    pre_schedule = np.asarray(
        policy_selection_mask(row_proxy, probabilities, calibration), dtype=bool
    )
    if pre_schedule.shape != (len(rows),):
        raise ReplayInputError(
            f"{rows.pair} {policy.arm}: adapter gate is row-misaligned"
        )
    probabilities.setflags(write=False)
    pre_schedule.setflags(write=False)
    return probabilities, pre_schedule


def _loaded_book_structural_gate(book: Any) -> tuple[str | None, float | None]:
    """Resolve comparator-A gate state from the authenticated loaded book."""

    strategy = getattr(book, "strategy", None)
    if not isinstance(strategy, Mapping):
        raise ReplayInputError("comparator-A loaded-book strategy is malformed")
    threshold: Any = None
    for key in ("bb_width_thr", "bbw_thr"):
        if isinstance(strategy.get(key), (int, float)):
            threshold = strategy[key]
            break
    gate = strategy.get("gate")
    if threshold is None and isinstance(gate, Mapping):
        for key in ("bb_width_thr", "bbw_thr"):
            if isinstance(gate.get(key), (int, float)):
                threshold = gate[key]
                break
    if threshold is None:
        return None, None
    return "15m_bb_width", _finite_float(
        threshold, "comparator_A.loaded_book_structural_threshold"
    )


def _score_comparator_a_canonical(
    rows: ScoreRows,
    policy: ArmPolicy,
    book: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Score A once through LoadedBook and preserve separate canonical gates."""

    pair = rows.pair
    if policy.arm != "A" or str(getattr(book, "pair", "")).upper() != pair:
        raise ReplayInputError(f"{pair}: comparator-A loaded-book identity differs")
    if tuple(getattr(book, "feature_cols", ())) != policy.feature_cols:
        raise ReplayInputError(f"{pair}: comparator-A loaded-book feature order differs")
    if tuple(Path(value) for value in getattr(book, "model_paths", ())) != tuple(
        policy.model_paths
    ):
        raise ReplayInputError(f"{pair}: comparator-A loaded-book model order differs")
    if _finite_float(getattr(book, "conf_thr", None), f"{pair}.A.book_threshold") != (
        policy.confidence_threshold
    ):
        raise ReplayInputError(f"{pair}: comparator-A loaded-book threshold differs")
    structural_column, structural_threshold = _loaded_book_structural_gate(book)
    if (
        structural_column != policy.structural_column
        or structural_threshold != policy.structural_threshold
    ):
        raise ReplayInputError(f"{pair}: comparator-A loaded-book gate differs")

    matrix = np.ascontiguousarray(rows.X, dtype="float32")
    scoreable = np.isfinite(matrix).all(axis=1)
    probabilities = np.full(len(rows), np.nan, dtype="float64")
    confidence = np.zeros(len(rows), dtype=bool)
    structural = np.zeros(len(rows), dtype=bool)
    frame = pd.DataFrame(
        matrix[scoreable].astype("float64", copy=False),
        columns=policy.feature_cols,
    )
    model_identities_before = tuple(
        _policy_model_identity(policy, position)
        for position in range(len(policy.model_paths))
    )
    from book_runtime import BookRuntimeError
    import lightgbm as lgb

    try:
        predicted = np.asarray(
            book.predict_probabilities(frame), dtype="float64"
        )
        components = book.gate_components(frame, predicted)
    except (BookRuntimeError, lgb.basic.LightGBMError) as exc:
        raise ReplayInputError(
            f"{pair}: canonical comparator-A scoring/gating failed: {exc}"
        ) from exc
    model_identities_after = tuple(
        _policy_model_identity(policy, position)
        for position in range(len(policy.model_paths))
    )
    if model_identities_after != model_identities_before:
        raise ReplayInputError(
            f"{pair}: comparator-A model identity changed during canonical scoring"
        )
    if (
        predicted.shape != (int(scoreable.sum()),)
        or not np.isfinite(predicted).all()
        or np.any((predicted < 0) | (predicted > 1))
    ):
        raise ReplayInputError(
            f"{pair}: canonical comparator-A probability vector is invalid"
        )
    component_confidence = np.asarray(components.confidence_passed)
    component_structural = np.asarray(components.structural_passed)
    if (
        component_confidence.dtype != np.bool_
        or component_structural.dtype != np.bool_
        or component_confidence.shape != predicted.shape
        or component_structural.shape != predicted.shape
    ):
        raise ReplayInputError(f"{pair}: canonical comparator-A gates are misaligned")
    probabilities[scoreable] = predicted
    confidence[scoreable] = component_confidence
    structural[scoreable] = component_structural
    pre_schedule = confidence & structural
    for value in (probabilities, confidence, structural, pre_schedule):
        value.setflags(write=False)
    return probabilities, confidence, structural, pre_schedule


def _comparator_a_replay_record(
    rows: ScoreRows,
    policy: ArmPolicy,
    book: Any,
    probabilities: np.ndarray,
    confidence: np.ndarray,
    structural: np.ndarray,
    pre_schedule: np.ndarray,
) -> dict[str, Any]:
    """Reconstruct the arm-seal A record from pre-outcome replay inputs."""

    matrix = np.ascontiguousarray(rows.X, dtype="float32")
    p = np.asarray(probabilities, dtype="float64")
    scoreable = np.isfinite(matrix).all(axis=1) & np.isfinite(p)
    confidence_array = np.asarray(confidence, dtype=bool)
    structural_array = np.asarray(structural, dtype=bool)
    selected = np.asarray(pre_schedule, dtype=bool)
    if any(
        value.shape != (len(rows),)
        for value in (p, confidence_array, structural_array, selected)
    ):
        raise ReplayInputError(f"{rows.pair}: comparator-A evidence is row-misaligned")
    if (
        np.any(confidence_array & ~scoreable)
        or np.any(structural_array & ~scoreable)
        or not np.array_equal(selected, confidence_array & structural_array)
    ):
        raise ReplayInputError(
            f"{rows.pair}: comparator-A evidence masks are incoherent"
        )
    entry_ns = np.asarray(rows.entry_ns, dtype="int64")
    return {
        "book_id": str(getattr(book, "book_id", "")),
        "bundle_id": policy.bundle_id,
        "rows": len(rows),
        "scoreable_rows": int(scoreable.sum()),
        "selected_rows": int(selected.sum()),
        "entry_ns_sha256": _array_sha(entry_ns, "<i8"),
        "probability_f64le_sha256": _array_sha(p, "<f8"),
        "scoreable_mask_u8_sha256": _array_sha(scoreable, "uint8"),
        "confidence_mask_u8_sha256": _array_sha(confidence_array, "uint8"),
        "structural_mask_u8_sha256": _array_sha(structural_array, "uint8"),
        "pre_schedule_mask_u8_sha256": _array_sha(selected, "uint8"),
        "ordered_selected_entry_ns_sha256": _array_sha(
            entry_ns[selected], "<i8"
        ),
        "threshold": policy.confidence_threshold,
        "structural_column": policy.structural_column,
        "structural_threshold": policy.structural_threshold,
    }


def _pair_runtime_score_rows(
    spec: Mapping[str, Any],
    pair_input: PairReplayInput,
) -> tuple[ScoreRows, np.ndarray]:
    """Construct the exact outcome-blind replay/runtime score window once."""

    replay_split = spec["splits"]["replay"]
    lower_ns = int(pd.Timestamp(replay_split["entry_at_or_after"]).value)
    upper_ns = int(pd.Timestamp(replay_split["entry_before"]).value)
    rows = pair_input.score_rows
    all_seconds = epoch_nanoseconds_to_seconds(
        rows.entry_ns, name=f"{pair_input.pair}.entry_ns"
    )
    if np.any(np.remainder(all_seconds, 60) != 0):
        raise ReplayInputError(
            f"{pair_input.pair}: decision clock is not an exact completed-minute grid"
        )
    split_positions = np.flatnonzero(
        (rows.entry_ns >= lower_ns) & (rows.entry_ns < upper_ns)
    )
    if not len(split_positions):
        raise ReplayInputError(f"{pair_input.pair}: replay split is empty")
    split_rows = _take_score_rows(rows, split_positions)
    split_seconds = all_seconds[split_positions]
    runtime_positions = np.flatnonzero(canonical_runtime_ny_mask(split_seconds))
    if not len(runtime_positions):
        raise ReplayInputError(
            f"{pair_input.pair}: runtime score window is empty"
        )
    return (
        _take_score_rows(split_rows, runtime_positions),
        split_seconds[runtime_positions],
    )


def _validate_comparator_a_repeatability(
    spec: Mapping[str, Any],
    pair_input: PairReplayInput,
) -> ComparatorAScoreCache:
    """Repeat A exactly before any pair is allowed to open observation closes."""

    runtime_rows, _runtime_seconds = _pair_runtime_score_rows(spec, pair_input)
    probabilities, confidence, structural, pre_schedule = (
        _score_comparator_a_canonical(
            runtime_rows,
            pair_input.primary_policies["A"],
            pair_input.comparator_a_book,
        )
    )
    actual = _comparator_a_replay_record(
        runtime_rows,
        pair_input.primary_policies["A"],
        pair_input.comparator_a_book,
        probabilities,
        confidence,
        structural,
        pre_schedule,
    )
    if actual != pair_input.comparator_a_repeatability.as_dict():
        raise ReplayInputError(
            f"{pair_input.pair}: comparator-A replay differs from arm-seal evidence"
        )
    return ComparatorAScoreCache(
        entry_ns_sha256=_array_sha(runtime_rows.entry_ns, "<i8"),
        probabilities=probabilities,
        pre_schedule_mask=pre_schedule,
    )


def _precompute_pair_policy_scores(
    spec: Mapping[str, Any],
    pair_input: PairReplayInput,
) -> dict[str, ComparatorAScoreCache]:
    """Score/cache every arm outcome-blind; evaluation must never reopen models."""

    runtime_rows, _runtime_seconds = _pair_runtime_score_rows(spec, pair_input)
    expected_arms = {
        **pair_input.primary_policies,
        **pair_input.control_policies,
    }
    output = {
        "A": _validate_comparator_a_repeatability(spec, pair_input)
    }
    entry_digest = _array_sha(runtime_rows.entry_ns, "<i8")
    for arm, policy in expected_arms.items():
        if arm == "A":
            continue
        probabilities, pre_schedule = score_policy(runtime_rows, policy)
        output[arm] = ComparatorAScoreCache(
            entry_ns_sha256=entry_digest,
            probabilities=probabilities,
            pre_schedule_mask=pre_schedule,
        )
    if tuple(output) != tuple(expected_arms):
        raise ReplayInputError(
            f"{pair_input.pair}: precomputed policy-score arm order differs"
        )
    return output


def preflight_policy_model_schemas(
    pair_inputs: Mapping[str, PairReplayInput | InvalidPairReplayInput],
) -> None:
    """Validate every sealed model/schema before any observation close is read."""

    import lightgbm as lgb

    for pair in PAIR_ORDER:
        pair_input = pair_inputs[pair]
        if isinstance(pair_input, InvalidPairReplayInput):
            continue
        _preflight_pair_policy_model_schemas(pair_input, lgb=lgb)


def _preflight_pair_policy_model_schemas(
    pair_input: PairReplayInput,
    *,
    lgb: Any | None = None,
) -> None:
    """Convert model-file/schema failures into an explicit pair input defect."""

    if lgb is None:
        import lightgbm as lgb

    pair = pair_input.pair
    try:
        for arm, policy in {
            **pair_input.primary_policies,
            **pair_input.control_policies,
        }.items():
            if policy.feature_cols != tuple(pair_input.score_rows.feature_cols):
                raise ReplayInputError(
                    f"{pair} {arm}: preflight feature order differs from score rows"
                )
            if arm == "A":
                boosters = tuple(
                    getattr(pair_input.comparator_a_book, "boosters", ())
                )
                if len(boosters) != len(policy.model_paths):
                    raise ReplayInputError(
                        f"{pair} A: loaded-book booster inventory differs"
                    )
                for position, booster in enumerate(boosters):
                    before_identity = _policy_model_identity(policy, position)
                    if booster.num_feature() != len(policy.feature_cols):
                        raise ReplayInputError(
                            f"{pair} A: loaded-book feature count differs"
                        )
                    if _policy_model_identity(policy, position) != before_identity:
                        raise ReplayInputError(
                            f"{pair} A: model identity changed during preflight"
                        )
                continue
            for position, path in enumerate(policy.model_paths):
                before_identity = _policy_model_identity(policy, position)
                booster = lgb.Booster(model_file=str(path))
                if booster.num_feature() != len(policy.feature_cols):
                    raise ReplayInputError(
                        f"{pair} {arm}: preflight model feature count differs"
                    )
                del booster
                if _policy_model_identity(policy, position) != before_identity:
                    raise ReplayInputError(
                        f"{pair} {arm}: model identity changed during preflight"
                    )
    except ReplayInputError:
        raise
    except lgb.basic.LightGBMError as exc:
        raise ReplayInputError(
            f"{pair}: cannot load/inspect sealed policy model: {exc}"
        ) from exc


def evaluate_pair(
    spec: Mapping[str, Any],
    pair_input: PairReplayInput,
    *,
    observation: ObservationSeries | None = None,
    policy_scores: Mapping[str, ComparatorAScoreCache] | None = None,
) -> PairComputation:
    """Score and settle one pair on its full common eligible runtime grid."""

    _validate_replay_spec(spec)
    pair = pair_input.pair
    pair_spec = spec["pairs"][pair]
    if pair_spec["adapter_family"] not in spec["control_family"]["representatives"]:
        raise ReplayInputError(f"{pair}: adapter family is not registered")
    replay = spec["splits"]["replay"]
    settlement_end_ns = int(pd.Timestamp(replay["settlement_exit_before"]).value)
    if settlement_end_ns % NS_PER_SECOND != 0:
        raise ReplayInputError(
            f"{pair}: settlement-exit boundary is not an exact epoch second"
        )
    settlement_end_s = settlement_end_ns // NS_PER_SECOND
    settlement = spec["evaluator"]["settlement"]
    horizon_s = int(settlement["horizon_s"])
    lag_s = int(settlement["lag_s"])
    tol_s = int(settlement["tol_s"])
    runtime_rows, runtime_seconds = _pair_runtime_score_rows(spec, pair_input)
    policies = {**pair_input.primary_policies, **pair_input.control_policies}
    if policy_scores is None:
        policy_scores = _precompute_pair_policy_scores(spec, pair_input)
    cached = dict(policy_scores)
    if tuple(cached) != tuple(policies):
        raise ReplayInputError(f"{pair}: precomputed policy-score arm order differs")
    entry_digest = _array_sha(runtime_rows.entry_ns, "<i8")
    scored: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for arm in policies:
        value = cached[arm]
        if (
            not isinstance(value, ComparatorAScoreCache)
            or value.entry_ns_sha256 != entry_digest
            or len(value.probabilities) != len(runtime_rows)
        ):
            raise ReplayInputError(
                f"{pair} {arm}: precomputed policy score differs from runtime rows"
            )
        scored[arm] = (value.probabilities, value.pre_schedule_mask)
    finite_features = np.isfinite(runtime_rows.X).all(axis=1)
    common_scoreable = finite_features.copy()
    for arm in PRIMARY_ARMS:
        probability = scored[arm][0]
        if np.any(finite_features & ~np.isfinite(probability)):
            raise ReplayInputError(
                f"{pair} {arm}: finite covariates produced a non-finite score"
            )
        common_scoreable &= np.isfinite(probability)
    for arm in pair_input.control_policies:
        if np.any(common_scoreable & ~np.isfinite(scored[arm][0])):
            raise ReplayInputError(
                f"{pair} {arm}: control is not scoreable on the common grid"
            )

    scoreable_positions = np.flatnonzero(common_scoreable)
    if not len(scoreable_positions):
        raise ReplayInputError(f"{pair}: full common scoreable grid is empty")
    decision_s = runtime_seconds[scoreable_positions]
    obs = observation or load_observation_series(
        pair_input.observation_paths,
        expected_inventory=pair_input.observation_inventory,
        pair=pair,
        observation_at_or_after=replay["entry_at_or_after"],
        observation_before=replay["settlement_exit_before"],
    )
    returns, canonical_settlement_valid, actual_exit_s = (
        canonical_settlement_with_exit_on_decision_grid(
            decision_s,
            obs.timestamps_s,
            obs.closes,
            horizon_s=horizon_s,
            tol_s=tol_s,
            lag_s=lag_s,
        )
    )
    canonical_settlement_valid = np.asarray(canonical_settlement_valid, dtype=bool)
    actual_exit_s = np.asarray(actual_exit_s, dtype="int64")
    returns = np.asarray(returns, dtype="float64")
    # The split boundary applies to the observation canonically used for the
    # exit, not to the nominal decision+lag+horizon target.  Equality is
    # excluded by the sealed ``settlement_exit_before`` contract.
    exit_before_boundary = actual_exit_s < settlement_end_s
    settlement_valid = canonical_settlement_valid & exit_before_boundary
    eligible_mask = settlement_valid & np.isfinite(returns)
    if not eligible_mask.any():
        raise ReplayInputError(
            f"{pair}: no common scoreable row has a valid settlement"
        )
    eligible_source_positions = scoreable_positions[eligible_mask]
    eligible_s = decision_s[eligible_mask]
    eligible_returns = returns[eligible_mask]
    eligible_exit_s = actual_exit_s[eligible_mask]
    validate_epoch_seconds(eligible_s, name=f"{pair}.eligible_timestamps_s")
    validate_epoch_seconds(eligible_exit_s, name=f"{pair}.eligible_exit_timestamps_s")
    if np.any(eligible_exit_s >= settlement_end_s):
        raise ReplayInputError(f"{pair}: eligible canonical exit reaches split boundary")

    primary: dict[str, ScoredArm] = {}
    controls: dict[str, ScoredArm] = {}
    for arm, policy in policies.items():
        probability_all, pre_all = scored[arm]
        probability = np.asarray(
            probability_all[eligible_source_positions], dtype="float64"
        )
        pre = np.asarray(pre_all[eligible_source_positions], dtype=bool)
        # The adapter already applied the exact model-owned threshold and
        # structural gate.  Passing that frozen preselection as the structural
        # mask with threshold zero delegates only combined scheduling and
        # ties-lose utility to the canonical statistics wrapper.
        schedule = scheduled_arm(
            eligible_s,
            probability,
            eligible_returns,
            threshold=0.0,
            structural_gate=pre,
            gap_s=int(spec["evaluator"]["scheduler"]["gap_s"]),
        )
        value = ScoredArm(policy, probability, pre, schedule)
        (primary if arm in PRIMARY_ARMS else controls)[arm] = value

    date_keys = ny_date_keys(eligible_s)
    main_endpoints = _main_pair_endpoints(pair, eligible_s, primary)
    control_endpoints = (
        _control_pair_endpoints(pair_spec["adapter_family"], eligible_s, controls)
        if controls
        else {}
    )
    report = _pair_report(
        pair,
        pair_spec["adapter_family"],
        eligible_s,
        eligible_returns,
        primary,
        controls,
        obs,
        runtime_rows=len(runtime_rows),
        common_scoreable_rows=len(scoreable_positions),
        settlement_valid_rows=int(settlement_valid.sum()),
        canonical_settlement_valid_rows=int(canonical_settlement_valid.sum()),
        settlement_boundary_rejected_rows=int(
            (canonical_settlement_valid & ~exit_before_boundary).sum()
        ),
        eligible_exit_s=eligible_exit_s,
        settlement_end_s=settlement_end_s,
        input_invalid_reasons=pair_input.invalid_reasons,
    )
    return PairComputation(
        pair=pair,
        timestamps_s=eligible_s,
        returns=eligible_returns,
        dates=date_keys,
        primary=primary,
        controls=controls,
        main_endpoints=main_endpoints,
        control_endpoints=control_endpoints,
        report=report,
    )


def _invalid_pair_computation(
    spec: Mapping[str, Any],
    pair_input: InvalidPairReplayInput,
) -> PairComputation:
    """Emit named, noncomputable slots without fabricating pair observations."""

    empty_i64 = np.array([], dtype="int64")
    empty_f64 = np.array([], dtype="float64")
    pair = pair_input.pair
    main_names = (
        "C_minus_B_yield",
        "C_minus_A_yield",
        "C_UP_accuracy_minus_0_5",
        "C_DOWN_accuracy_minus_0_5",
    )
    main = {
        name: DateRatioEndpoint(
            f"{pair}:{name}", empty_i64, empty_f64, empty_f64
        )
        for name in main_names
    }
    control: dict[str, DateRatioEndpoint] = {}
    if pair_input.control_bundle_ids:
        family = pair_input.adapter_family
        for name in (
            "C_perm_minus_B_perm_yield",
            "C_perm_UP_accuracy_minus_0_5",
            "C_perm_DOWN_accuracy_minus_0_5",
        ):
            control[name] = DateRatioEndpoint(
                f"{family}:{name}", empty_i64, empty_f64, empty_f64
            )
    report = {
        "schema": "m15-book-refresh-pair-invalid-placeholder/v1",
        "pair": pair,
        "adapter_family": pair_input.adapter_family,
        "invalid_placeholder": True,
        "input_invalid_reasons": list(pair_input.invalid_reasons),
        "observation_clock": None,
        "grid": {
            "runtime_rows": 0,
            "common_scoreable_rows": 0,
            "canonical_settlement_valid_rows": 0,
            "settlement_boundary_rejected_rows": 0,
            "settlement_valid_rows": 0,
            "eligible_rows": 0,
            "moved_rows": 0,
            "moved_up_rate": None,
            "moved_up_rate_bounds": [0.47, 0.53],
            "moved_up_rate_tripwire_passed": False,
        },
        "arms": {
            arm: {
                "bundle_id": pair_input.primary_bundle_ids[arm],
                "status": "noncomputable_pair_invalid",
            }
            for arm in PRIMARY_ARMS
        },
        "controls": {
            arm: {
                "bundle_id": pair_input.control_bundle_ids[arm],
                "status": "noncomputable_pair_invalid",
            }
            for arm in pair_input.control_bundle_ids
        },
    }
    return PairComputation(
        pair=pair,
        timestamps_s=empty_i64,
        returns=empty_f64,
        dates=empty_i64,
        primary={},
        controls={},
        main_endpoints=main,
        control_endpoints=control,
        report=report,
    )


def _invalid_from_pair_input(
    pair_input: PairReplayInput,
    reason: str,
    *,
    adapter_family: str,
) -> InvalidPairReplayInput:
    return InvalidPairReplayInput(
        pair=pair_input.pair,
        adapter_family=adapter_family,
        primary_bundle_ids={
            arm: pair_input.primary_policies[arm].bundle_id for arm in PRIMARY_ARMS
        },
        control_bundle_ids={
            arm: pair_input.control_policies[arm].bundle_id
            for arm in pair_input.control_policies
        },
        invalid_reasons=(reason,),
    )


def run_common_replay(
    spec: Mapping[str, Any],
    pair_inputs: Mapping[str, PairReplayInput | InvalidPairReplayInput],
    *,
    bootstrap_chunk_size: int = 512,
) -> ReplayCampaignResult:
    """Run the sealed six-target replay and fixed main/control inference once."""

    _validate_replay_spec(spec)
    ordered_inputs = dict(pair_inputs)
    if tuple(ordered_inputs) != tuple(PAIR_ORDER):
        raise ReplayInputError(f"pair inputs must preserve exact order {PAIR_ORDER}")
    representatives = dict(spec["control_family"]["representatives"])
    expected_control_pairs = set(representatives.values())
    for pair in PAIR_ORDER:
        pair_input = ordered_inputs[pair]
        has_controls = bool(
            pair_input.control_bundle_ids
            if isinstance(pair_input, InvalidPairReplayInput)
            else pair_input.control_policies
        )
        if has_controls != (pair in expected_control_pairs):
            raise ReplayInputError(
                f"{pair}: control-policy presence differs from representative table"
            )

    computations: dict[str, PairComputation] = {}
    model_ready: dict[str, PairReplayInput] = {}
    # First pass: authenticate every model/schema for all six target pairs.
    for pair in PAIR_ORDER:
        pair_input = ordered_inputs[pair]
        if isinstance(pair_input, InvalidPairReplayInput):
            computations[pair] = _invalid_pair_computation(spec, pair_input)
            continue
        try:
            _preflight_pair_policy_model_schemas(pair_input)
            model_ready[pair] = pair_input
        except (ReplayInputError, AdapterError) as exc:
            invalid = _invalid_from_pair_input(
                pair_input,
                f"pair_model_preflight_defect:{type(exc).__name__}:{str(exc)}",
                adapter_family=spec["pairs"][pair]["adapter_family"],
            )
            ordered_inputs[pair] = invalid
            computations[pair] = _invalid_pair_computation(spec, invalid)

    # Second pass remains outcome-blind: independently repeat comparator A and
    # score/cache every B/C/control arm for every model-valid pair.  The full
    # six-target pass must finish before any pair may open an observation close;
    # evaluation therefore never reopens a model.
    ready: dict[
        str, tuple[PairReplayInput, dict[str, ComparatorAScoreCache]]
    ] = {}
    for pair in PAIR_ORDER:
        pair_input = model_ready.get(pair)
        if pair_input is None:
            continue
        try:
            ready[pair] = (
                pair_input,
                _precompute_pair_policy_scores(spec, pair_input),
            )
        except (ReplayInputError, AdapterError) as exc:
            invalid = _invalid_from_pair_input(
                pair_input,
                f"pair_policy_prescore_defect:{type(exc).__name__}:{str(exc)}",
                adapter_family=spec["pairs"][pair]["adapter_family"],
            )
            ordered_inputs[pair] = invalid
            computations[pair] = _invalid_pair_computation(spec, invalid)

    # Third pass: and only now, load/settle each still-valid pair.
    for pair in PAIR_ORDER:
        prepared = ready.get(pair)
        if prepared is None:
            continue
        pair_input, policy_scores = prepared
        try:
            computations[pair] = evaluate_pair(
                spec,
                pair_input,
                policy_scores=policy_scores,
            )
        except (ReplayInputError, AdapterError) as exc:
            invalid = _invalid_from_pair_input(
                pair_input,
                f"pair_replay_defect:{type(exc).__name__}:{str(exc)}",
                adapter_family=spec["pairs"][pair]["adapter_family"],
            )
            ordered_inputs[pair] = invalid
            computations[pair] = _invalid_pair_computation(spec, invalid)

    main_endpoints = _ordered_main_endpoints(computations)
    if len(main_endpoints) != MAIN_FAMILY_SIZE:
        raise ReplayInputError("main endpoint constructor did not preserve fixed m0=24")
    if any(len(endpoint.date_keys) for endpoint in main_endpoints):
        master_calendar = master_calendar_from_endpoints(main_endpoints)
        master_calendar_source = "represented_endpoint_dates"
    else:
        # Six target-local defects must still reserve the fixed inference
        # family.  Derive its non-outcome calendar from the sealed runtime
        # window so zero-denominator placeholders remain explicitly
        # noncomputable instead of turning into a shared statistics crash.
        replay_split = spec["splits"]["replay"]
        lower_s = int(pd.Timestamp(replay_split["entry_at_or_after"]).timestamp())
        upper_s = int(pd.Timestamp(replay_split["entry_before"]).timestamp())
        minute_clock = np.arange(lower_s, upper_s, 60, dtype="int64")
        runtime_clock = minute_clock[canonical_runtime_ny_mask(minute_clock)]
        master_calendar = np.unique(ny_date_keys(runtime_clock)).astype("int64")
        if not len(master_calendar):
            raise ReplayInputError("sealed replay runtime calendar is empty")
        master_calendar_source = "sealed_runtime_window_no_valid_pairs"
    constants = spec["sealed_constants"]
    main_inference = infer_main_family(
        main_endpoints,
        master_calendar=master_calendar,
        bootstrap_seed=int(constants["bootstrap_seed"]),
        block_lengths=tuple(int(value) for value in constants["block_lengths"]),
        bootstrap_replicates=int(constants["production_bootstrap_replicates"]),
        bootstrap_chunk_size=bootstrap_chunk_size,
    )
    control_endpoints = _ordered_control_endpoints(spec, computations)
    if len(control_endpoints) != CONTROL_FAMILY_SIZE:
        raise ReplayInputError(
            "control endpoint constructor did not preserve fixed m0=9"
        )
    control_inference = infer_control_family(
        control_endpoints,
        master_calendar=master_calendar,
        bootstrap_seed=int(constants["control_bootstrap_seed"]),
        block_lengths=tuple(int(value) for value in constants["block_lengths"]),
        bootstrap_replicates=int(constants["production_bootstrap_replicates"]),
        bootstrap_chunk_size=bootstrap_chunk_size,
    )

    main_by_name = {row.name: row for row in main_inference.endpoints}
    control_by_name = {row.name: row for row in control_inference.endpoints}
    control_screens = _control_screens(spec, computations, control_by_name)
    global_control_failures = [
        family
        for family, row in control_screens.items()
        if row["null_promotion_screen_passed"]
    ]
    global_invalid = [
        f"negative_control_promotion:{value}" for value in global_control_failures
    ]

    pair_results: dict[str, dict[str, Any]] = {}
    statuses: dict[str, dict[str, Any]] = {}
    for pair in PAIR_ORDER:
        computation = computations[pair]
        family = spec["pairs"][pair]["adapter_family"]
        names = _pair_main_endpoint_names(pair)
        inference_rows = {name: main_by_name[name] for name in names.values()}
        statistics_computable = all(row.computable for row in inference_rows.values())
        control_estimable = bool(control_screens[family]["estimable"])
        invalid_reasons = list(computation.report["input_invalid_reasons"])
        invalid_reasons.extend(global_invalid)
        is_placeholder = computation.report.get("invalid_placeholder") is True
        if (
            not is_placeholder
            and not computation.report["grid"]["moved_up_rate_tripwire_passed"]
        ):
            invalid_reasons.append("full_common_grid_moved_up_rate_tripwire")

        if is_placeholder:
            combined_counts = {arm: 0 for arm in PRIMARY_ARMS}
            side_counts = {
                arm: {"UP": 0, "DOWN": 0} for arm in PRIMARY_ARMS
            }
            c_month_counts = {
                month: {"UP": 0, "DOWN": 0} for month, _ in REPORT_MONTHS
            }
            c_month_yield = {month: None for month, _ in REPORT_MONTHS}
            whole_book_pass = False
        else:
            combined_counts = {
                arm: int(computation.primary[arm].schedule.selected_mask.sum())
                for arm in PRIMARY_ARMS
            }
            side_counts = {
                arm: _side_counts(computation.primary[arm].schedule)
                for arm in PRIMARY_ARMS
            }
            c_month_counts = {}
            c_month_yield = {}
            for month, _ in REPORT_MONTHS:
                monthly = computation.report["arms"]["C"]["periods"][month]
                c_month_counts[month] = {
                    side: int(monthly[side]["trade_count"])
                    for side in ("UP", "DOWN")
                }
                value = monthly["whole"]["correctness_yield"]
                c_month_yield[month] = float(value) if value is not None else None
            normal_input = ordered_inputs[pair]
            if not isinstance(normal_input, PairReplayInput):
                raise ReplayInputError(f"{pair}: valid computation lost pair input")
            whole_book_pass = normal_input.whole_book_pass

        lower = {
            "C_minus_B_yield": inference_rows[names["C_minus_B_yield"]].lower,
            "C_minus_A_yield": inference_rows[names["C_minus_A_yield"]].lower,
            "C_UP_accuracy_minus_0_5": inference_rows[
                names["C_UP_accuracy_minus_0_5"]
            ].lower,
            "C_DOWN_accuracy_minus_0_5": inference_rows[
                names["C_DOWN_accuracy_minus_0_5"]
            ].lower,
        }
        upper = {
            "C_minus_A_yield": inference_rows[names["C_minus_A_yield"]].upper,
            "C_UP_accuracy_minus_0_5": inference_rows[
                names["C_UP_accuracy_minus_0_5"]
            ].upper,
            "C_DOWN_accuracy_minus_0_5": inference_rows[
                names["C_DOWN_accuracy_minus_0_5"]
            ].upper,
        }
        # assign_terminal_status returns before consuming interval maps when
        # ordinary sparsity/control non-estimability makes them undefined.
        decision = assign_terminal_status(
            invalid_reasons=invalid_reasons,
            statistics_computable=statistics_computable,
            control_estimable=control_estimable,
            side_trade_counts=side_counts,
            combined_trade_counts=combined_counts,
            promotion_lower_bounds=lower,
            retention_upper_bounds=upper,
            monthly_c_side_counts=c_month_counts,
            monthly_c_whole_book_yield=c_month_yield,
            whole_book_pass=whole_book_pass,
        )
        statuses[pair] = decision.as_dict()
        pair_result = dict(computation.report)
        pair_result.update(
            {
                "schema": "m15-book-refresh-pair-replay/v1",
                "terminal_status": decision.as_dict(),
                "status_inputs": {
                    "invalid_reasons": invalid_reasons,
                    "statistics_computable": statistics_computable,
                    "control_estimable": control_estimable,
                    "side_trade_counts": side_counts,
                    "combined_trade_counts": combined_counts,
                    "promotion_lower_bounds": lower,
                    "retention_upper_bounds": upper,
                    "monthly_c_side_counts": c_month_counts,
                    "monthly_c_whole_book_yield": c_month_yield,
                    "whole_book_pass": whole_book_pass,
                },
                "main_endpoints": [row.as_dict() for row in inference_rows.values()],
                "control_family": family,
                "control_estimable": control_estimable,
                "control_screen": control_screens[family],
            }
        )
        _assert_json_serializable(pair_result)
        pair_results[pair] = pair_result

    survivors = [
        pair for pair in PAIR_ORDER if statuses[pair]["status"] == "PROMOTE_TO_SHADOW"
    ]
    pair_summaries: dict[str, dict[str, Any]] = {}
    for pair in PAIR_ORDER:
        computation = computations[pair]
        if computation.report.get("invalid_placeholder") is True:
            trade_counts = {
                arm: {"combined": 0, "UP": 0, "DOWN": 0}
                for arm in PRIMARY_ARMS
            }
        else:
            trade_counts = {
                arm: {
                    "combined": int(
                        computation.primary[arm].schedule.selected_mask.sum()
                    ),
                    **_side_counts(computation.primary[arm].schedule),
                }
                for arm in PRIMARY_ARMS
            }
        pair_summaries[pair] = {
            "eligible_rows": computation.report["grid"]["eligible_rows"],
            "moved_up_rate": computation.report["grid"]["moved_up_rate"],
            "moved_up_rate_tripwire_passed": computation.report["grid"][
                "moved_up_rate_tripwire_passed"
            ],
            "trade_counts": trade_counts,
            "main_endpoints": [
                main_by_name[name].as_dict()
                for name in _pair_main_endpoint_names(pair).values()
            ],
            "terminal_status": statuses[pair],
        }
    joint = {
        "schema": "m15-book-refresh-joint-replay/v1",
        "pair_order": list(PAIR_ORDER),
        "master_calendar_source": master_calendar_source,
        "main_inference": main_inference.as_dict(),
        "control_inference": control_inference.as_dict(),
        "control_screens": control_screens,
        "global_invalid_reasons": global_invalid,
        "statuses": statuses,
        "survivor_pair_order": survivors,
        "pair_summaries": pair_summaries,
        "moved_up_rate_tripwires": {
            pair: {
                "rate": computations[pair].report["grid"]["moved_up_rate"],
                "bounds": computations[pair].report["grid"]["moved_up_rate_bounds"],
                "passed": computations[pair].report["grid"][
                    "moved_up_rate_tripwire_passed"
                ],
            }
            for pair in PAIR_ORDER
        },
    }
    _assert_json_serializable(joint)
    return ReplayCampaignResult(pair_results=pair_results, joint_result=joint)


def _strict_utc_ns(values: Any, *, name: str) -> np.ndarray:
    if isinstance(values, pd.Series):
        raw = values.array
    else:
        raw = values
    try:
        index = pd.DatetimeIndex(raw)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ReplayInputError(
            f"{name}: observation clock is not datetime-like"
        ) from exc
    if index.tz is None or str(index.tz) != "UTC":
        raise ReplayInputError(f"{name}: observation clock must be timezone-aware UTC")
    if index.hasnans:
        raise ReplayInputError(f"{name}: observation clock contains NaT")
    try:
        # Parquet sources legitimately mix datetime64[us, UTC] and
        # datetime64[ns, UTC].  Normalize the declared unit exactly; reading
        # ``asi8`` directly would misinterpret microseconds as nanoseconds.
        ns = np.asarray(index.as_unit("ns").asi8, dtype="int64")
    except (OverflowError, ValueError) as exc:
        raise ReplayInputError(
            f"{name}: observation clock cannot convert exactly to ns"
        ) from exc
    validate_epoch_nanoseconds(ns, name=f"{name}.timestamps_ns")
    return ns


def _take_score_rows(rows: ScoreRows, positions: np.ndarray) -> ScoreRows:
    take = np.asarray(positions)
    if take.ndim != 1 or not np.issubdtype(take.dtype, np.integer):
        raise ReplayInputError(
            f"{rows.pair}: score-row positions must be a 1-D integer vector"
        )
    take = take.astype("int64", copy=False)
    if np.any(take < 0) or np.any(take >= len(rows)):
        raise ReplayInputError(f"{rows.pair}: score-row position is out of bounds")
    if len(take) > 1 and np.any(take[1:] <= take[:-1]):
        raise ReplayInputError(
            f"{rows.pair}: score-row positions must be strictly increasing"
        )
    return ScoreRows(
        pair=rows.pair,
        feature_cols=rows.feature_cols,
        X=np.ascontiguousarray(rows.X[take], dtype="float32"),
        entry_ns=np.asarray(rows.entry_ns[take], dtype="int64"),
        source_feature_ns=np.asarray(rows.source_feature_ns[take], dtype="int64"),
    )


def _read_json_object(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ReplayInputError(f"JSON source is missing/nonregular: {path}")
    try:
        value = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as exc:
        raise ReplayInputError(f"cannot read JSON object {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ReplayInputError(f"JSON object required: {path}")
    return value


def _read_arm_seal_json(
    workspace: Path,
    path: Path,
    output_hashes: Mapping[str, str],
) -> dict[str, Any]:
    """Read once and authenticate the exact arm-seal bytes being consumed."""

    if not isinstance(output_hashes, Mapping):
        raise ReplayInputError("arm-seal output hash authority is malformed")
    try:
        relative = path.resolve(strict=True).relative_to(
            workspace.resolve(strict=True)
        )
    except (OSError, ValueError) as exc:
        raise ReplayInputError("arm-seal JSON path escapes the workspace") from exc
    key = f"workspace:{relative.as_posix()}"
    expected = output_hashes.get(key)
    if not _is_hex(expected, 64):
        raise ReplayInputError(f"arm-seal phase does not bind consumed JSON: {key}")
    if (
        path.is_symlink()
        or not path.is_file()
        or path.stat().st_mode & 0o222
    ):
        raise ReplayInputError(f"arm-seal JSON is not sealed read-only evidence: {path}")
    try:
        encoded = path.read_bytes()
    except OSError as exc:
        raise ReplayInputError(f"cannot read arm-seal JSON: {path}: {exc}") from exc
    if hashlib.sha256(encoded).hexdigest() != expected:
        raise ReplayInputError(f"arm-seal JSON differs from phase hash: {path}")
    try:
        value = json.loads(encoded)
    except json.JSONDecodeError as exc:
        raise ReplayInputError(f"cannot decode arm-seal JSON: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ReplayInputError(f"arm-seal JSON object required: {path}")
    return value


def _default_fit_result_loader(
    workspace: Path, pair: str, arm: str
) -> Mapping[str, Any]:
    return _read_json_object(
        workspace / "fits" / pair / arm / "canonical" / "fit_result.json"
    )


def _workspace_artifact(workspace: Path, value: Any, *, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ReplayInputError(f"{name}: workspace artifact path is malformed")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ReplayInputError(
            f"{name}: workspace artifact path must be safe and relative"
        )
    candidate = workspace / relative
    try:
        candidate.resolve(strict=True).relative_to(workspace.resolve(strict=True))
    except (OSError, ValueError) as exc:
        raise ReplayInputError(
            f"{name}: workspace artifact escapes or is missing"
        ) from exc
    if candidate.is_symlink() or not candidate.is_file():
        raise ReplayInputError(f"{name}: workspace artifact is not a regular file")
    return candidate


def _policy_from_fit_result(
    workspace: Path,
    pair: str,
    arm: str,
    result: Mapping[str, Any],
    *,
    spec: Mapping[str, Any],
    expected_bundle_id: str,
    expected_prereg_id: str,
    expected_implementation_git_sha: str,
) -> ArmPolicy:
    if (
        result.get("pair") != pair
        or result.get("arm") != arm
        or result.get("attempt") != "canonical"
        or result.get("bundle_id") != expected_bundle_id
    ):
        raise ReplayInputError(
            f"{pair} {arm}: canonical fit result differs from run binding"
        )
    models = result.get("models")
    if not isinstance(models, list):
        raise ReplayInputError(
            f"{pair} {arm}: fit model inventory is malformed"
        )
    resolved_path = _workspace_artifact(
        workspace,
        result.get("resolved_bundle_spec"),
        name=f"{pair}.{arm}.resolved_bundle_spec",
    )
    strategy_path = _workspace_artifact(
        workspace,
        result.get("strategy"),
        name=f"{pair}.{arm}.strategy",
    )
    model_paths = tuple(
        _workspace_artifact(workspace, value, name=f"{pair}.{arm}.model[{index}]")
        for index, value in enumerate(models)
    )
    resolved_with_id = _read_json_object(resolved_path)
    strategy = _read_json_object(strategy_path)
    resolved = dict(resolved_with_id)
    resolved_bundle_id = resolved.pop("bundle_id", None)
    is_control = arm.endswith("_perm")
    expected_lifecycle = (
        "inactive_shadow_candidate"
        if arm == "C" and not is_control
        else "workspace_only_nonpublishable"
    )
    expected_book_id = (
        spec["pairs"][pair]["candidate_book_id"]
        if arm == "C" and not is_control
        else f"workspace:{pair}:{arm}"
    )
    strategy_shared_fields = ("pair", "arm", "seed_order")
    if (
        resolved.get("schema") != "m15-book-refresh-resolved-bundle-spec/v1"
        or resolved.get("pair") != pair
        or resolved.get("arm") != arm
        or resolved.get("adapter_family") != spec["pairs"][pair]["adapter_family"]
        or resolved.get("builder") != spec["pairs"][pair]["builder"]
        or resolved.get("seed_order") != spec["pairs"][pair]["seed_order"]
        or resolved.get("prereg_id") != expected_prereg_id
        or resolved.get("implementation_git_sha")
        != expected_implementation_git_sha
        or resolved.get("book_id") != expected_book_id
        or resolved.get("lifecycle_status") != expected_lifecycle
        or resolved_bundle_id != expected_bundle_id
        or strategy.get("schema") != "m15-book-refresh-strategy/v1"
        or any(
            strategy.get(field) != resolved.get(field)
            for field in strategy_shared_fields
        )
    ):
        raise ReplayInputError(
            f"{pair} {arm}: resolved/strategy identity differs from run binding"
        )
    try:
        def canonical(value: Any) -> bytes:
            return json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")

        domain = str(spec["identity"]["domains"]["bundle_id"])
        components = [
            canonical(resolved),
            strategy_path.name.encode("utf-8"),
            strategy_path.read_bytes(),
        ]
        seen_names: set[str] = set()
        for model_path in model_paths:
            if model_path.name in seen_names:
                raise ReplayInputError(f"{pair} {arm}: duplicate model basename")
            seen_names.add(model_path.name)
            components.extend([model_path.name.encode("utf-8"), model_path.read_bytes()])
        digest = hashlib.sha256()
        for item in [domain.encode("utf-8"), *components]:
            digest.update(len(item).to_bytes(8, "big", signed=False))
            digest.update(item)
        computed_bundle_id = digest.hexdigest()
    except (KeyError, OSError, TypeError, ValueError) as exc:
        raise ReplayInputError(
            f"{pair} {arm}: cannot authenticate resolved bundle bytes: {exc}"
        ) from exc
    if not model_paths or computed_bundle_id != expected_bundle_id:
        raise ReplayInputError(
            f"{pair} {arm}: resolved/strategy/model bundle bytes differ"
        )
    feature_cols = resolved.get("feature_cols")
    calibration = resolved.get("calibration")
    gate = strategy.get("gate")
    if (
        not isinstance(feature_cols, list)
        or not isinstance(calibration, Mapping)
        or not isinstance(gate, Mapping)
        or resolved.get("feature_dtype") != "float32"
        or strategy.get("feature_dtype") != "float32"
        or strategy.get("n_features") != len(feature_cols)
    ):
        raise ReplayInputError(f"{pair} {arm}: bundle policy state is malformed")
    expected_strategy_policy = {
        "feature_cols": feature_cols,
        "coverage": calibration.get("target_coverage"),
        "conf_thr": calibration.get("confidence_threshold"),
        "structural_column": calibration.get("structural_column"),
        "structural_quantile": calibration.get("structural_quantile"),
        "structural_threshold": calibration.get("structural_threshold"),
    }
    actual_strategy_policy = {
        "feature_cols": strategy.get("feature_cols"),
        "coverage": strategy.get("coverage"),
        "conf_thr": strategy.get("conf_thr"),
        "structural_column": gate.get("structural_column"),
        "structural_quantile": gate.get("structural_quantile"),
        "structural_threshold": gate.get("structural_threshold"),
    }
    if actual_strategy_policy != expected_strategy_policy:
        raise ReplayInputError(
            f"{pair} {arm}: strategy policy differs from resolved calibration"
        )
    model_sha256 = result.get("model_sha256")
    if (
        not isinstance(model_sha256, list)
        or len(model_sha256) != len(model_paths)
        or any(not _is_hex(value, 64) for value in model_sha256)
        or [_sha256_file(path) for path in model_paths] != model_sha256
    ):
        raise ReplayInputError(f"{pair} {arm}: model SHA-256 inventory differs")
    return ArmPolicy(
        arm=arm,
        bundle_id=expected_bundle_id,
        feature_cols=tuple(str(value) for value in feature_cols),
        model_paths=model_paths,
        model_sha256=tuple(model_sha256),
        confidence_threshold=calibration.get("confidence_threshold"),
        target_coverage=calibration.get("target_coverage"),
        structural_column=calibration.get("structural_column"),
        structural_quantile=calibration.get("structural_quantile"),
        structural_threshold=calibration.get("structural_threshold"),
    )


def _validate_replay_spec(spec: Mapping[str, Any]) -> None:
    try:
        pairs = tuple(spec["pair_order"])
        constants = spec["sealed_constants"]
        evaluator = spec["evaluator"]
        main = spec["primary_family"]
        control = spec["control_family"]
    except (KeyError, TypeError) as exc:
        raise ReplayInputError(f"malformed replay spec: {exc}") from exc
    if pairs != tuple(PAIR_ORDER) or tuple(spec["arm_order"]) != PRIMARY_ARMS:
        raise ReplayInputError("replay pair/arm order differs from the sealed contract")
    if tuple(spec.get("feature_source_pair_order", ())) != FEATURE_SOURCE_PAIR_ORDER:
        raise ReplayInputError("replay feature-source order differs from the sealed contract")
    if int(main["m0"]) != MAIN_FAMILY_SIZE or int(control["m0"]) != CONTROL_FAMILY_SIZE:
        raise ReplayInputError("replay fixed-family size changed")
    if (
        list(constants["block_lengths"]) != [5, 10, 20]
        or int(constants["production_bootstrap_replicates"]) != 10_000
        or int(constants["bootstrap_seed"]) != 9001
        or int(constants["control_bootstrap_seed"]) != 9002
        or int(constants["ddof"]) != 1
    ):
        raise ReplayInputError("production bootstrap contract changed")
    settlement = evaluator["settlement"]
    scheduler = evaluator["scheduler"]
    mapping = evaluator.get("timestamp_mapping", {})
    if (
        mapping.get("processed_10s_close_observation_timestamp")
        != "source_label_plus_10_seconds"
        or mapping.get("decision_timestamp") != "feature_source_label_plus_60_seconds"
        or settlement["canonical_helper"] != "min1_production.wc_ret"
        or int(settlement["horizon_s"]) != 900
        or int(settlement["tol_s"]) != 10
        or int(settlement["lag_s"]) != 1
        or settlement["ties"] != "lose"
        or scheduler["canonical_helper"] != "min1_production.nonoverlap_chrono"
        or int(scheduler["gap_s"]) != 900
        or scheduler["scope"] != "combined_before_side"
    ):
        raise ReplayInputError("canonical settlement/scheduler contract changed")


def _main_pair_endpoints(
    pair: str,
    timestamps_s: np.ndarray,
    primary: Mapping[str, ScoredArm],
) -> dict[str, DateRatioEndpoint]:
    c = primary["C"].schedule
    endpoints = {
        "C_minus_B_yield": endpoint_from_rows(
            f"{pair}:C_minus_B_yield",
            timestamps_s,
            c.utility - primary["B"].schedule.utility,
            np.ones(len(timestamps_s), dtype="float64"),
        ),
        "C_minus_A_yield": endpoint_from_rows(
            f"{pair}:C_minus_A_yield",
            timestamps_s,
            c.utility - primary["A"].schedule.utility,
            np.ones(len(timestamps_s), dtype="float64"),
        ),
    }
    for side, direction in (("UP", True), ("DOWN", False)):
        selected = c.selected_mask & (c.direction_up == direction)
        endpoints[f"C_{side}_accuracy_minus_0_5"] = endpoint_from_rows(
            f"{pair}:C_{side}_accuracy_minus_0_5",
            timestamps_s,
            (c.correct.astype("float64") - 0.5) * selected,
            selected.astype("float64"),
        )
    return endpoints


def _control_pair_endpoints(
    family: str,
    timestamps_s: np.ndarray,
    controls: Mapping[str, ScoredArm],
) -> dict[str, DateRatioEndpoint]:
    if tuple(controls) != CONTROL_ARMS:
        raise ReplayInputError(f"{family}: incomplete control arm pair")
    c = controls["C_perm"].schedule
    endpoints = {
        "C_perm_minus_B_perm_yield": endpoint_from_rows(
            f"{family}:C_perm_minus_B_perm_yield",
            timestamps_s,
            c.utility - controls["B_perm"].schedule.utility,
            np.ones(len(timestamps_s), dtype="float64"),
        )
    }
    for side, direction in (("UP", True), ("DOWN", False)):
        selected = c.selected_mask & (c.direction_up == direction)
        endpoints[f"C_perm_{side}_accuracy_minus_0_5"] = endpoint_from_rows(
            f"{family}:C_perm_{side}_accuracy_minus_0_5",
            timestamps_s,
            (c.correct.astype("float64") - 0.5) * selected,
            selected.astype("float64"),
        )
    return endpoints


def _ordered_main_endpoints(
    computations: Mapping[str, PairComputation],
) -> list[DateRatioEndpoint]:
    order = (
        "C_minus_B_yield",
        "C_minus_A_yield",
        "C_UP_accuracy_minus_0_5",
        "C_DOWN_accuracy_minus_0_5",
    )
    return [
        computations[pair].main_endpoints[name] for name in order for pair in PAIR_ORDER
    ]


def _ordered_control_endpoints(
    spec: Mapping[str, Any],
    computations: Mapping[str, PairComputation],
) -> list[DateRatioEndpoint]:
    order = (
        "C_perm_minus_B_perm_yield",
        "C_perm_UP_accuracy_minus_0_5",
        "C_perm_DOWN_accuracy_minus_0_5",
    )
    output: list[DateRatioEndpoint] = []
    for family, pair in spec["control_family"]["representatives"].items():
        endpoints = computations[pair].control_endpoints
        output.extend(endpoints[name] for name in order)
    return output


def _pair_main_endpoint_names(pair: str) -> dict[str, str]:
    return {
        key: f"{pair}:{key}"
        for key in (
            "C_minus_B_yield",
            "C_minus_A_yield",
            "C_UP_accuracy_minus_0_5",
            "C_DOWN_accuracy_minus_0_5",
        )
    }


def _control_screens(
    spec: Mapping[str, Any],
    computations: Mapping[str, PairComputation],
    inference: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for family, pair in spec["control_family"]["representatives"].items():
        computation = computations[pair]
        if computation.report.get("invalid_placeholder") is True:
            names = [
                f"{family}:C_perm_minus_B_perm_yield",
                f"{family}:C_perm_UP_accuracy_minus_0_5",
                f"{family}:C_perm_DOWN_accuracy_minus_0_5",
            ]
            output[family] = {
                "representative_pair": pair,
                "estimable": False,
                "endpoint_names": names,
                "B_perm_trade_counts": {"combined": 0, "UP": 0, "DOWN": 0},
                "C_perm_trade_counts": {"combined": 0, "UP": 0, "DOWN": 0},
                "count_activity_screen_passed": False,
                "all_three_lower_bounds_above_zero": False,
                "null_promotion_screen_passed": False,
                "invalid_placeholder": True,
            }
            continue
        b = computation.controls["B_perm"].schedule
        c = computation.controls["C_perm"].schedule
        names = [
            f"{family}:C_perm_minus_B_perm_yield",
            f"{family}:C_perm_UP_accuracy_minus_0_5",
            f"{family}:C_perm_DOWN_accuracy_minus_0_5",
        ]
        rows = [inference[name] for name in names]
        estimable = all(row.computable for row in rows)
        counts_b = _side_counts(b)
        counts_c = _side_counts(c)
        total_b = int(b.selected_mask.sum())
        total_c = int(c.selected_mask.sum())
        count_activity = (
            counts_c["UP"] >= 50 and counts_c["DOWN"] >= 50 and total_c >= 0.8 * total_b
        )
        interval_pass = estimable and all(
            row.lower is not None and row.lower > 0 for row in rows
        )
        output[family] = {
            "representative_pair": pair,
            "estimable": estimable,
            "endpoint_names": names,
            "B_perm_trade_counts": {"combined": total_b, **counts_b},
            "C_perm_trade_counts": {"combined": total_c, **counts_c},
            "count_activity_screen_passed": bool(count_activity),
            "all_three_lower_bounds_above_zero": bool(interval_pass),
            "null_promotion_screen_passed": bool(count_activity and interval_pass),
        }
    return output


def _pair_report(
    pair: str,
    family: str,
    timestamps_s: np.ndarray,
    returns: np.ndarray,
    primary: Mapping[str, ScoredArm],
    controls: Mapping[str, ScoredArm],
    observation: ObservationSeries,
    *,
    runtime_rows: int,
    common_scoreable_rows: int,
    settlement_valid_rows: int,
    canonical_settlement_valid_rows: int,
    settlement_boundary_rejected_rows: int,
    eligible_exit_s: np.ndarray,
    settlement_end_s: int,
    input_invalid_reasons: Sequence[str],
) -> dict[str, Any]:
    moved = returns != 0
    moved_count = int(moved.sum())
    up_rate = _safe_ratio(int((returns > 0).sum()), moved_count)
    month_numbers = _local_month_numbers(timestamps_s)
    monthly_up_rate: dict[str, float | None] = {}
    for name, number in REPORT_MONTHS:
        in_month = month_numbers == number
        monthly_moved = moved & in_month
        monthly_up_rate[name] = _safe_ratio(
            int(((returns > 0) & in_month).sum()), int(monthly_moved.sum())
        )
    lower, upper = 0.47, 0.53
    tripwire = up_rate is not None and lower <= up_rate <= upper
    arms: dict[str, Any] = {}
    for arm, value in {**primary, **controls}.items():
        periods = {
            "combined": _arm_period_report(
                value.schedule, np.ones(len(timestamps_s), dtype=bool)
            ),
        }
        for month, number in REPORT_MONTHS:
            periods[month] = _arm_period_report(value.schedule, month_numbers == number)
        arms[arm] = {
            "bundle_id": value.policy.bundle_id,
            "threshold": value.policy.confidence_threshold,
            "target_coverage": value.policy.target_coverage,
            "structural_column": value.policy.structural_column,
            "structural_threshold": value.policy.structural_threshold,
            "pre_schedule_count": int(value.pre_schedule_mask.sum()),
            "pre_schedule_coverage": _safe_ratio(
                int(value.pre_schedule_mask.sum()), len(timestamps_s)
            ),
            "scheduled_count": int(value.schedule.selected_mask.sum()),
            "scheduled_coverage": _safe_ratio(
                int(value.schedule.selected_mask.sum()), len(timestamps_s)
            ),
            "probability_f64le_sha256": _array_sha(value.probabilities, "<f8"),
            "ordered_selected_timestamp_s_sha256": _array_sha(
                timestamps_s[value.schedule.selected_mask], "<i8"
            ),
            "moved_probability_metrics": _probability_metrics(
                value.probabilities, returns, moved
            ),
            "periods": periods,
        }
    diagnostics = _selection_diagnostics(
        timestamps_s, returns, primary["A"], primary["C"]
    )
    diagnostics["numeric_threshold_drift"] = {
        "B_minus_A": primary["B"].policy.confidence_threshold
        - primary["A"].policy.confidence_threshold,
        "C_minus_A": primary["C"].policy.confidence_threshold
        - primary["A"].policy.confidence_threshold,
        "C_minus_B": primary["C"].policy.confidence_threshold
        - primary["B"].policy.confidence_threshold,
    }
    return {
        "pair": pair,
        "adapter_family": family,
        "input_invalid_reasons": list(input_invalid_reasons),
        "observation_clock": {
            "source_files": list(observation.source_files),
            "rows": len(observation.timestamps_s),
            "first_timestamp_s": int(observation.timestamps_s[0]),
            "last_timestamp_s": int(observation.timestamps_s[-1]),
            "timestamp_s_sha256": _array_sha(observation.timestamps_s, "<i8"),
            "close_f64le_sha256": _array_sha(observation.closes, "<f8"),
            "strictly_increasing_unique": True,
        },
        "grid": {
            "runtime_rows": int(runtime_rows),
            "common_scoreable_rows": int(common_scoreable_rows),
            "canonical_settlement_valid_rows": int(
                canonical_settlement_valid_rows
            ),
            "settlement_boundary_rejected_rows": int(
                settlement_boundary_rejected_rows
            ),
            "settlement_valid_rows": int(settlement_valid_rows),
            "settlement_exit_before_s": int(settlement_end_s),
            "actual_exit_timestamp_s_sha256": _array_sha(eligible_exit_s, "<i8"),
            "first_actual_exit_timestamp_s": int(eligible_exit_s[0]),
            "last_actual_exit_timestamp_s": int(eligible_exit_s[-1]),
            "all_actual_exits_before_boundary": bool(
                np.all(eligible_exit_s < settlement_end_s)
            ),
            "eligible_rows": len(timestamps_s),
            "first_timestamp_s": int(timestamps_s[0]),
            "last_timestamp_s": int(timestamps_s[-1]),
            "timestamp_s_sha256": _array_sha(timestamps_s, "<i8"),
            "return_f64le_sha256": _array_sha(returns, "<f8"),
            "moved_rows": moved_count,
            "tie_rows": int((returns == 0).sum()),
            "tie_rate": _safe_ratio(int((returns == 0).sum()), len(returns)),
            "moved_up_rate": up_rate,
            "moved_up_rate_by_month": monthly_up_rate,
            "moved_up_rate_bounds": [lower, upper],
            "moved_up_rate_tripwire_passed": bool(tripwire),
        },
        "arms": arms,
        "A_C_diagnostics": diagnostics,
    }


def _arm_period_report(
    schedule: ScheduledArm, period_mask: np.ndarray
) -> dict[str, Any]:
    period = np.asarray(period_mask, dtype=bool)
    denominator = int(period.sum())
    output: dict[str, Any] = {}
    for name, side_mask in (
        ("whole", np.ones(len(period), dtype=bool)),
        ("UP", schedule.direction_up),
        ("DOWN", ~schedule.direction_up),
    ):
        selected = period & schedule.selected_mask & side_mask
        count = int(selected.sum())
        wins = int((schedule.correct & selected).sum())
        output[name] = {
            "eligible_count": denominator,
            "trade_count": count,
            "correct_count": wins,
            "accuracy": _safe_ratio(wins, count),
            "correctness_yield": (
                float(schedule.utility[selected].sum() / denominator)
                if denominator
                else None
            ),
        }
    return output


def _selection_diagnostics(
    timestamps_s: np.ndarray,
    returns: np.ndarray,
    a: ScoredArm,
    c: ScoredArm,
) -> dict[str, Any]:
    a_sel = a.schedule.selected_mask
    c_sel = c.schedule.selected_mask
    intersection = a_sel & c_sel
    union = a_sel | c_sel
    disagreement = a.schedule.direction_up != c.schedule.direction_up
    both_disagree = intersection & disagreement
    return {
        "scheduled": {
            "A_selected": int(a_sel.sum()),
            "C_selected": int(c_sel.sum()),
            "intersection": int(intersection.sum()),
            "union": int(union.sum()),
            "jaccard": _safe_ratio(int(intersection.sum()), int(union.sum())),
            "intersection_timestamp_s_sha256": _array_sha(
                timestamps_s[intersection], "<i8"
            ),
            "union_timestamp_s_sha256": _array_sha(timestamps_s[union], "<i8"),
        },
        "pre_schedule": {
            "A_selected": int(a.pre_schedule_mask.sum()),
            "C_selected": int(c.pre_schedule_mask.sum()),
            "intersection": int((a.pre_schedule_mask & c.pre_schedule_mask).sum()),
            "union": int((a.pre_schedule_mask | c.pre_schedule_mask).sum()),
        },
        "probability_correlation": _correlation(a.probabilities, c.probabilities),
        "confidence_correlation": _correlation(
            np.abs(a.probabilities - 0.5), np.abs(c.probabilities - 0.5)
        ),
        "direction_disagreement": {
            "eligible_rows": int(disagreement.sum()),
            "selected_union_rows": int((union & disagreement).sum()),
            "selected_intersection_rows": int(both_disagree.sum()),
            "A_correct_on_selected_intersection_disagreements": int(
                (a.schedule.correct & both_disagree).sum()
            ),
            "C_correct_on_selected_intersection_disagreements": int(
                (c.schedule.correct & both_disagree).sum()
            ),
            "ties_on_selected_intersection_disagreements": int(
                ((returns == 0) & both_disagree).sum()
            ),
        },
    }


def _probability_metrics(
    probabilities: np.ndarray,
    returns: np.ndarray,
    moved: np.ndarray,
) -> dict[str, Any]:
    count = int(moved.sum())
    if not count:
        return {"rows": 0, "auc": None, "brier": None, "log_loss": None}
    labels = (returns[moved] > 0).astype("uint8")
    p = np.asarray(probabilities[moved], dtype="float64")
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ReplayInputError("probability diagnostics received invalid scores")
    auc = float(roc_auc_score(labels, p)) if len(np.unique(labels)) == 2 else None
    return {
        "rows": count,
        "auc": auc,
        "brier": float(brier_score_loss(labels, p)),
        "log_loss": float(log_loss(labels, p, labels=[0, 1])),
    }


def _side_counts(schedule: ScheduledArm) -> dict[str, int]:
    return {
        "UP": int((schedule.selected_mask & schedule.direction_up).sum()),
        "DOWN": int((schedule.selected_mask & ~schedule.direction_up).sum()),
    }


def _local_month_numbers(timestamps_s: np.ndarray) -> np.ndarray:
    values = pd.to_datetime(timestamps_s, unit="s", utc=True).tz_convert(
        "America/New_York"
    )
    return values.month.to_numpy(dtype="int16")


def _correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    a = np.asarray(left, dtype="float64")
    b = np.asarray(right, dtype="float64")
    if len(a) < 2 or not np.isfinite(a).all() or not np.isfinite(b).all():
        return None
    if np.std(a, ddof=1) == 0 or np.std(b, ddof=1) == 0:
        return None
    value = float(np.corrcoef(a, b)[0, 1])
    return value if np.isfinite(value) else None


def _safe_ratio(numerator: int | float, denominator: int | float) -> float | None:
    return float(numerator / denominator) if denominator else None


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ReplayInputError(f"{name} must be a finite number")
    try:
        output = float(value)
    except (TypeError, ValueError) as exc:
        raise ReplayInputError(f"{name} must be a finite number") from exc
    if not np.isfinite(output):
        raise ReplayInputError(f"{name} must be a finite number")
    return output


def _array_sha(values: Any, dtype: str) -> str:
    array = np.ascontiguousarray(values, dtype=np.dtype(dtype))
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _is_hex(value: Any, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


def _assert_json_serializable(value: Any) -> None:
    try:
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ReplayInputError(
            f"replay result is not canonical-JSON serializable: {exc}"
        ) from exc


__all__ = [
    "ArmPolicy",
    "ComparatorARepeatability",
    "ComparatorAScoreCache",
    "InvalidPairReplayInput",
    "ObservationSeries",
    "PairComputation",
    "PairReplayInput",
    "ReplayCampaignResult",
    "ReplayInputError",
    "build_measured_pair_inputs",
    "discover_observation_files",
    "evaluate_pair",
    "finalize_joint_result",
    "load_observation_series",
    "run_common_replay",
    "run_workspace_replay",
    "score_policy",
]
