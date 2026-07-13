#!/usr/bin/env python3
"""Sealed three-arm refresh campaign for six incumbent 15-minute books.

The default action is intentionally inert.  ``--smoke`` exercises only
synthetic/pre-April fixtures in a temporary repository.  Measured actions are
forward-only and refuse semantic replay access until the source, derived-data,
and arm seals exist in that order.

Run from the repository root::

    ~/binary-algo-venv/bin/python scripts/m15_book_refresh.py --smoke

Issue #9 is the binding design.  This module owns orchestration and identities;
feature recipes stay in :mod:`m15_book_refresh_adapters`, settlement/scheduling
and inference stay in :mod:`m15_book_refresh_stats`, and candidate publication
stays in :mod:`manifest`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd

import manifest


REPO_ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = Path(__file__).with_name("m15_book_refresh_specs.json")
ENVIRONMENT_PATH = REPO_ROOT / "ENVIRONMENT_libs.txt"
RESULTS_JSON = REPO_ROOT / "results" / "json"
WORK_ROOT = REPO_ROOT / "logs" / "m15_book_refresh"
PROCESSED_ROOT = Path("/home/sean/git/processed")
PHASE_ZERO_AUTHORITY_PATH = (
    RESULTS_JSON / "m15_book_refresh_2026q1_phase_zero_acceptance_v2.json"
)
PHASE_ZERO_FLOAT32_LIVE_FEATURES_SHA256 = (
    "c485766d2dda85339938d59180056287ee5826ad275ff4d988588bdc2d201198"
)
TARGET_PAIR_ORDER = (
    "EURUSD", "USDJPY", "GBPUSD", "USDCHF", "AUDUSD", "NZDUSD",
)
FEATURE_SOURCE_PAIR_ORDER = (
    "EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDJPY", "USDCHF", "USDCAD",
)
CONTROL_ORDER = (
    ("eur_xpair_of", "B_perm"),
    ("eur_xpair_of", "C_perm"),
    ("gbpchf_xpair", "B_perm"),
    ("gbpchf_xpair", "C_perm"),
    ("own_pair", "B_perm"),
    ("own_pair", "C_perm"),
)
CONTROL_REPRESENTATIVES = {
    "eur_xpair_of": "EURUSD",
    "gbpchf_xpair": "GBPUSD",
    "own_pair": "USDJPY",
}
LEGACY_ACTIVE_ID_ORDER = (
    "EURUSD.m15xp.v1",
    "USDJPY.m15ny_seedens.v1",
    "GBPUSD.m15ny_xpair_seedens8.v1",
    "USDCHF.m15ny_xpair_seedens.v1",
    "USDCAD.m15ny_seedens.v1",
    "AUDUSD.m15ny_seedens.v1",
    "NZDUSD.m15ny_seedens.v1",
)
FORBIDDEN_SEMANTIC_COLUMNS = frozenset(
    {
        "y",
        "fwd_ret",
        "valid",
        "_fwd",
        "label",
        "target",
        "settlement",
        "moved",
        "outcome",
        "wc_ret",
        "win",
    }
)
PHASE_ORDER = (
    "source_preregistration",
    "derived_data",
    "adapter_parity",
    "fit_and_repeat",
    "arm_seal",
    "replay",
    "status_and_publication_lock",
    "candidate_publication",
    "shadow_spec",
)
SEMANTIC_REPLAY_PHASE = "arm_seal"


class RefreshError(RuntimeError):
    """A sealed campaign precondition or identity invariant failed."""


class PairRouteError(RefreshError):
    """A scoped consumed feature route is defective for one pair."""


@dataclass(frozen=True)
class FileIdentity:
    path: str
    bytes: int
    sha256: str


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def repo_relative(path: str | Path, *, root: Path = REPO_ROOT) -> str:
    candidate = Path(path)
    try:
        resolved = candidate.resolve(strict=True)
        relative = resolved.relative_to(root.resolve(strict=True))
    except (OSError, ValueError) as exc:
        raise RefreshError(f"path is not an existing file under repository root: {path}") from exc
    return relative.as_posix()


def file_identity(path: str | Path, *, root: Path = REPO_ROOT) -> FileIdentity:
    target = Path(path)
    if target.is_symlink() or not target.is_file():
        raise RefreshError(f"identity source must be a regular non-symlink file: {target}")
    return FileIdentity(
        path=repo_relative(target, root=root),
        bytes=target.stat().st_size,
        sha256=sha256_file(target),
    )


def length_prefixed_digest(domain: str, components: Sequence[bytes]) -> str:
    """Hash a domain and components, prefixing every item with unsigned u64 BE."""

    if not isinstance(domain, str) or not domain:
        raise RefreshError("identity domain must be a non-empty string")
    items = [domain.encode("utf-8"), *components]
    h = hashlib.sha256()
    for item in items:
        if not isinstance(item, bytes):
            raise RefreshError("identity components must be bytes")
        h.update(struct.pack(">Q", len(item)))
        h.update(item)
    return h.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return manifest.canonical_json_bytes(value)


def _validate_executable_statistics_contract(spec: Mapping[str, Any]) -> None:
    """Bind the executable null/statistics authorities to the sealed JSON."""

    import m15_book_refresh_stats as stats

    constants = spec["sealed_constants"]
    null_contract = spec["null_calibration"]
    expected_coverages = {
        pair: float(spec["pairs"][pair]["calibration_target_coverage"])
        for pair in TARGET_PAIR_ORDER
    }
    expected_fixtures = [
        {
            "name": row["name"],
            "rho_cross": float(row["rho_cross"]),
            "phi_label": float(row["phi_label"]),
            "phi_score": float(row["phi_score"]),
            "missing_pair_dates": float(row["missing_pair_dates"]),
        }
        for row in null_contract["fixtures"]
    ]
    actual_fixtures = [asdict(row) for row in stats.NULL_FIXTURES]
    checks = {
        "target_pair_order": tuple(stats.PAIR_ORDER) == TARGET_PAIR_ORDER,
        "feature_source_pair_order": (
            tuple(stats.FEATURE_SOURCE_PAIR_ORDER) == FEATURE_SOURCE_PAIR_ORDER
        ),
        "main_family_size": stats.MAIN_FAMILY_SIZE == spec["primary_family"]["m0"],
        "control_family_size": (
            stats.CONTROL_FAMILY_SIZE == spec["control_family"]["m0"]
        ),
        "production_block_lengths": (
            tuple(stats.PRODUCTION_BLOCK_LENGTHS)
            == tuple(constants["block_lengths"])
        ),
        "production_bootstrap_replicates": (
            stats.PRODUCTION_BOOTSTRAP_REPLICATES
            == constants["production_bootstrap_replicates"]
        ),
        "main_bootstrap_seed": stats.MAIN_BOOTSTRAP_SEED == constants["bootstrap_seed"],
        "control_bootstrap_seed": (
            stats.CONTROL_BOOTSTRAP_SEED == constants["control_bootstrap_seed"]
        ),
        "shadow_bootstrap_seed": (
            stats.SHADOW_BOOTSTRAP_SEED == constants["shadow_bootstrap_seed"]
        ),
        "pair_coverages": stats.PAIR_COVERAGE == expected_coverages,
        "null_seed": stats.NULL_TEST_SEED == constants["stats_null_test_seed"],
        "null_campaigns": (
            stats.NULL_CAMPAIGNS_PER_CELL == null_contract["campaigns_per_cell"]
        ),
        "null_bootstrap_replicates": (
            stats.NULL_BOOTSTRAP_REPLICATES
            == null_contract["bootstrap_replicates"]
        ),
        "null_block_lengths": (
            tuple(stats.NULL_BLOCK_LENGTHS)
            == tuple(null_contract["block_lengths"])
        ),
        "null_false_positive_limit": (
            stats.NULL_FALSE_POSITIVE_LIMIT
            == null_contract["false_positive_limit_inclusive"]
        ),
        "null_synthetic_dates": (
            stats.NULL_SYNTHETIC_NY_DATES == null_contract["synthetic_ny_dates"]
        ),
        "null_rows_per_pair_date": (
            stats.NULL_ROWS_PER_PAIR_DATE == null_contract["rows_per_pair_date"]
        ),
        "null_row_minutes": (
            stats.NULL_NY_START_HOUR == 8
            and stats.NULL_ROW_MINUTES == null_contract["row_minutes"]
            == "08:00_through_15:59_America/New_York"
        ),
        "null_rho_arm": stats.NULL_RHO_ARM == null_contract["rho_arm"],
        "null_fixtures": actual_fixtures == expected_fixtures,
        "null_fixture_order": (
            [row.name for row in stats.NULL_FIXTURES]
            == null_contract["fixture_index_order"]
        ),
        "null_helper_codes": stats.NULL_HELPER_CODES == null_contract["helper_codes"],
        "null_stream_codes": stats.NULL_STREAM_CODES == null_contract["stream_codes"],
        "deferred_shadow_phase_code": (
            stats.DEFERRED_SHADOW_PHASE_CODE
            == null_contract["binding_deferred_shadow"]["seed_binding"][
                "shadow_phase_code"
            ]
        ),
        "deferred_shadow_seed_domain": (
            stats.DEFERRED_SHADOW_SEED_DOMAIN
            == null_contract["binding_deferred_shadow"]["seed_binding"]["domain"]
        ),
        "mechanics_source_coverages": (
            stats._NULL_MECHANICS_COVERAGE
            == {**expected_coverages, "USDCAD": 0.02}
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RefreshError(
            "executable statistics/null contract differs from sealed JSON: "
            + ", ".join(failed)
        )


def load_spec(path: str | Path = SPEC_PATH) -> dict[str, Any]:
    """Load and reject any malformed or silently broadened campaign spec."""

    spec_path = Path(path)
    try:
        raw = spec_path.read_bytes()
        spec = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise RefreshError(f"cannot load refresh spec {spec_path}: {exc}") from exc
    if not isinstance(spec, dict) or spec.get("schema") != "m15-book-refresh-spec/v1":
        raise RefreshError("refresh spec schema mismatch")
    required_top = {
        "schema", "issue", "pair_order", "feature_source_pair_order",
        "feature_source_rules",
        "arm_order", "control_order",
        "sealed_constants", "splits", "evaluator", "cap_selector", "pairs",
        "identity", "phases", "phase_manifest", "primary_family",
        "control_family", "status_precedence", "promotion", "retention",
        "null_calibration", "publication", "lifecycle", "shadow",
    }
    if set(spec) != required_top:
        raise RefreshError(
            f"refresh spec top-level fields differ: missing={sorted(required_top-set(spec))} "
            f"unknown={sorted(set(spec)-required_top)}"
        )
    expected_pairs = list(TARGET_PAIR_ORDER)
    expected_sources = list(FEATURE_SOURCE_PAIR_ORDER)
    if spec["pair_order"] != expected_pairs or list(spec["pairs"]) != expected_pairs:
        raise RefreshError("target pair order is not the sealed six-book order")
    if spec["feature_source_pair_order"] != expected_sources:
        raise RefreshError("feature-source pair order is not the sealed seven-pair order")
    if set(expected_pairs) - set(expected_sources) or "USDCAD" in expected_pairs:
        raise RefreshError("target/source pair partition is malformed")
    if spec["feature_source_rules"] != {
        "authority": "live_features.PAIRS",
        "source_only_pairs": ["USDCAD"],
        "source_only_pairs_are_campaign_targets": False,
        "source_only_pairs_remain_required_for_cross_pair_feature_builds": True,
    }:
        raise RefreshError("feature-source-only routing rules changed")
    if spec["arm_order"] != ["A", "B", "C"]:
        raise RefreshError("arm order must be A/B/C")
    if spec["control_order"] != [list(row) for row in CONTROL_ORDER]:
        raise RefreshError("control order must be the exact six sealed permutation arms")
    if spec["control_family"].get("representatives") != CONTROL_REPRESENTATIVES:
        raise RefreshError("control representatives differ from the sealed family map")
    if spec["phases"] != list(PHASE_ORDER):
        raise RefreshError("phase order mismatch")
    constants = spec["sealed_constants"]
    if constants["horizon_seconds"] != 900 or constants["train_row_cap"] != 150000:
        raise RefreshError("horizon or timestamp cap changed")
    if constants["block_lengths"] != [5, 10, 20] or constants["ddof"] != 1:
        raise RefreshError("production bootstrap contract changed")
    if constants["probability_tie_rule"] != "p>=0.5_is_UP":
        raise RefreshError("probability direction-tie rule changed")
    if spec["evaluator"].get("timestamp_mapping") != {
        "processed_10s_source_label": "left_edge_of_[label,label+10s)_bar",
        "processed_10s_close_observation_timestamp": "source_label_plus_10_seconds",
        "feature_source_label": "left_edge_of_[label,label+60s)_completed_minute_bar",
        "decision_timestamp": "feature_source_label_plus_60_seconds",
        "feature_information_latest_source": (
            "processed_10s_bar_labeled_feature_source_label_plus_50_seconds_"
            "closing_at_decision_timestamp"
        ),
        "settlement_clock": "processed_10s_close_observation_timestamps",
        "causal_requirement": (
            "decision_and_settlement_observation_timestamps_never_precede_the_close_"
            "values_they_represent"
        ),
        "pinned_runtime_evidence": (
            "deriv_hot_daemon.score_from_snapshot_candidate_bar_close_equals_"
            "FeatureRow.timestamp_plus_1_minute"
        ),
    }:
        raise RefreshError("causal timestamp mapping changed")
    if spec["primary_family"].get("m0") != 24 or spec["control_family"].get("m0") != 9:
        raise RefreshError("fixed multiplicity family sizes changed")
    permutation = spec["control_family"].get("label_permutation", {})
    if permutation != {
        "generator": "numpy.random.Generator",
        "bit_generator": "numpy.random.PCG64",
        "seed_sequence": ["family_permutation_seed", "arm_code"],
        "arm_codes": {"B_perm": 0, "C_perm": 1},
        "partition_order": ["capped_fit_rows", "early_stopping_rows"],
        "state": "one_generator_consumed_sequentially_across_partition_order",
        "operation": "rng.permutation(len(partition))",
        "assignment": "labels[partition]=labels[partition][permutation]",
        "calibration_gate_label_values": "none",
    }:
        raise RefreshError("control label-permutation derivation changed")
    if spec["null_calibration"].get("campaigns_per_cell") != 400:
        raise RefreshError("null campaigns per cell changed")
    if spec["null_calibration"].get("bootstrap_replicates") != 1000:
        raise RefreshError("null bootstrap replicate count changed")
    if spec["null_calibration"].get("block_lengths") != [1, 2, 3]:
        raise RefreshError("null bootstrap lengths changed")
    null_contract = spec["null_calibration"]
    if (
        null_contract.get("false_positive_limit") != 32
        or null_contract.get("false_positive_limit_inclusive") != 32
        or null_contract.get("failure_count_minimum") != 33
    ):
        raise RefreshError("null false-positive acceptance limit changed")
    if null_contract.get("seed_sequence") != [
        9005,
        "fixture_index",
        "helper_code",
        "k_shadow_or_0",
        "campaign_index",
        "L_or_0",
        "stream_code",
    ]:
        raise RefreshError("null seed sequence changed")
    if null_contract.get("row_minutes") != "08:00_through_15:59_America/New_York":
        raise RefreshError("null row-minute range changed")
    if null_contract.get("fixture_index_order") != ["F0", "F1", "F2", "F3"]:
        raise RefreshError("null fixture index order changed")
    if null_contract.get("helper_codes") != {
        "main": 1,
        "control": 2,
        "shadow": 3,
    }:
        raise RefreshError("null helper codes changed")
    if null_contract.get("stream_codes") != {
        "labels": 1,
        "policy_shared": 2,
        "policy_left": 3,
        "policy_right": 4,
        "policy_reference": 5,
        "missing_dates": 6,
        "bootstrap": 7,
    }:
        raise RefreshError("null stream codes changed")
    if null_contract.get("error_budget") != {
        "reference_false_positive_probability": 0.05,
        "single_cell_failure_event": "X~Binomial(400,0.05); X>=33",
        "single_cell_failure_probability_exact": 0.0037687282460565734,
        "current_binding_cell_count": 8,
        "current_binding_union_bound": 0.030149825968452587,
        "deferred_binding_cell_count_max": 4,
        "combined_binding_cell_count_max": 12,
        "combined_binding_union_bound": 0.04522473895267888,
    }:
        raise RefreshError("null familywise error budget changed")
    if null_contract.get("binding_current") != {
        "helpers": {
            "main": {"m0": 24, "k_shadow": [0]},
            "control": {"m0": 9, "k_shadow": [0]},
        },
        "fixture_order": ["F0", "F1", "F2", "F3"],
        "cell_count": 8,
        "binding": True,
    }:
        raise RefreshError(
            "null helper cells must be the eight binding main/control cells only"
        )
    if null_contract.get("nonbinding_shadow_mechanics") != {
        "helper": "shadow",
        "pair_source": "feature_source_pair_order_prefix",
        "m0": "3*k_shadow",
        "k_shadow": [1, 2, 3, 4, 5, 6, 7],
        "accuracy_null": 0.5,
        "cell_count": 28,
        "binding": False,
        "excluded_from_acceptance": True,
        "purpose": "determinism_and_mechanics_only_not_shadow_boundary_calibration",
    }:
        raise RefreshError("prefix shadow mechanics must remain nonbinding at null 0.5")
    if null_contract.get("binding_deferred_shadow") != {
        "helper": "shadow",
        "pair_source": "exact_ordered_S_shadow_only",
        "m0": "3*k_shadow",
        "accuracy_null": 0.541,
        "fixture_order": ["F0", "F1", "F2", "F3"],
        "cell_count_max": 4,
        "binding": True,
        "run_after_ordered_S_shadow_is_sealed": True,
        "prefix_substitution_forbidden": True,
        "seed_binding": {
            "domain": "m15-book-refresh/deferred-shadow-null-seed-identity/v1",
            "shadow_phase_code": 1,
            "identity_payload": (
                "ordered_[pair,candidate_C_bundle_id]_from_ordered_S_shadow_seal"
            ),
            "identity_digest": (
                "sha256_length_prefixed_domain_and_canonical_identity_payload"
            ),
            "identity_digest_u32be": (
                "eight_consecutive_big_endian_uint32_words"
            ),
            "seed_sequence_layout": [
                "stats_null_test_seed",
                "shadow_phase_code",
                "identity_digest_u32be[0..7]",
                "fixture_index",
                "helper_code",
                "k_shadow",
                "campaign_index",
                "L_or_0",
                "stream_code",
            ],
        },
    }:
        raise RefreshError("deferred shadow calibration contract changed")
    _validate_executable_statistics_contract(spec)
    if spec["lifecycle"].get("legacy_active_ids") != list(LEGACY_ACTIVE_ID_ORDER):
        raise RefreshError("literal legacy-active IDs diverge from pair contracts")
    shadow = spec["shadow"]
    if set(shadow) != {
        "schemas", "statuses", "completed_retrospective_identity_fields",
        "empty_survivors_status", "survivors_but_zero_provider_eligible_status",
        "positive_eligible_status", "provider_qualification", "zero_eligible_cases",
        "zero_eligible_omits", "positive_eligible_handoff", "endpoint_count",
        "endpoint_order_per_candidate", "endpoint_formulas", "prospective_capture",
        "fixed_look", "statistics", "promotion_toward_activation",
        "activation_prerequisite", "family_reset_and_relook",
    }:
        raise RefreshError("prospective shadow contract fields differ")
    if shadow["schemas"] != {
        "zero_eligible": "m15-book-refresh-shadow-empty/v1",
        "positive_eligible": "m15-book-refresh-prospective-shadow-spec/v1",
    } or shadow["statuses"] != {
        "empty_survivors": "no_candidates",
        "survivors_but_zero_provider_eligible": "no_shadow_eligible_candidates",
        "positive_eligible_handoff": "handoff_only_T0_unset",
    }:
        raise RefreshError("prospective shadow schemas/statuses differ")
    if (
        shadow["provider_qualification"].get("static_live_provider_flags_are_authority")
        is not False
        or shadow["positive_eligible_handoff"].get("T0") is not None
        or shadow["positive_eligible_handoff"].get("start_shadow_in_this_issue") is not False
        or shadow["prospective_capture"].get("transport") != "recorded_Deriv_ticks"
        or shadow["prospective_capture"].get("backfill_allowed") is not False
        or shadow["fixed_look"].get("analysis_looks") != 1
        or shadow["fixed_look"].get("earliest_complete_weeks") != 8
        or shadow["fixed_look"].get("cap_complete_weeks") != 26
        or shadow["statistics"].get("replicates") != 10_000
        or shadow["statistics"].get("seed") != 9003
        or shadow["statistics"].get("block_lengths") != [5, 10, 20]
        or shadow["statistics"].get("ddof") != 1
        or shadow["statistics"].get("quantile_method") != "linear"
        or shadow["activation_prerequisite"].get("groups") != 6
        or shadow["activation_prerequisite"].get("paths") != 15
        or shadow["activation_prerequisite"].get("train_row_cap") != 150_000
        or shadow["activation_prerequisite"].get("inner_seed") != 13
        or shadow["activation_prerequisite"].get("inner_validation_date_probability") != 0.15
        or shadow["activation_prerequisite"].get("min_side_trades_per_path") != 25
        or shadow["activation_prerequisite"].get("side_p10_min") != 0.541
        or shadow["activation_prerequisite"].get("min_clear_paths") != 12
        or shadow["activation_prerequisite"].get("total_paths") != 15
    ):
        raise RefreshError("prospective shadow critical constants differ")
    expected_early_stopping_ties = {
        "EURUSD": "drop",
        "USDJPY": "drop",
        "GBPUSD": "include_as_down",
        "USDCHF": "include_as_down",
        "AUDUSD": "drop",
        "NZDUSD": "drop",
    }
    for pair in expected_pairs:
        row = spec["pairs"][pair]
        if row.get("feature_dtype") != "float32":
            raise RefreshError(f"{pair}: feature dtype changed")
        if row.get("seed_order") != list(range(len(row.get("seed_order", [])))):
            raise RefreshError(f"{pair}: seed order must be contiguous from zero")
        if not 0.0 < float(row.get("calibration_target_coverage", 0.0)) < 1.0:
            raise RefreshError(f"{pair}: invalid calibration coverage")
        if row.get("model", {}).get("class") != "lightgbm.LGBMClassifier":
            raise RefreshError(f"{pair}: unknown model class")
        if row.get("model", {}).get("deterministic") is not True:
            raise RefreshError(f"{pair}: deterministic LightGBM must remain enabled")
        if row.get("fit_ties") != "drop":
            raise RefreshError(f"{pair}: fitting tie rule changed")
        if row.get("early_stopping_ties") != expected_early_stopping_ties[pair]:
            raise RefreshError(f"{pair}: historical early-stopping tie rule changed")
    eur_gate = spec["pairs"]["EURUSD"].get("structural_gate", {})
    if eur_gate.get("calibration_scope") != (
        "full_exact_window_label_valid_rows_before_fixed_utc_session"
    ):
        raise RefreshError("EURUSD q33 calibration scope changed")
    from m15_book_refresh_adapters import (
        FEATURE_SOURCE_PAIR_ORDER as ADAPTER_FEATURE_SOURCE_PAIR_ORDER,
        contract_for,
        model_parameters,
    )
    import book_runtime
    import live_features

    if (
        tuple(spec["feature_source_pair_order"])
        != ADAPTER_FEATURE_SOURCE_PAIR_ORDER
        or list(spec["feature_source_pair_order"]) != list(live_features.PAIRS)
    ):
        raise RefreshError("spec feature-source order diverges from provider/adapter routing")
    if frozenset(LEGACY_ACTIVE_ID_ORDER) != book_runtime.LEGACY_ACTIVE_BOOK_IDS:
        raise RefreshError("campaign legacy whitelist diverges from production runtime")
    if [spec["pairs"][pair]["book_id_A"] for pair in expected_pairs] != [
        book_runtime.TARGET_BOOKS[pair] for pair in expected_pairs
    ]:
        raise RefreshError("campaign comparator-A IDs diverge from production targets")

    for pair in expected_pairs:
        row = spec["pairs"][pair]
        contract = contract_for(pair)
        structural = row["structural_gate"]
        if not (
            contract.family == row["adapter_family"]
            and contract.n_features == row["feature_count"]
            and contract.fit_stride == row["training_stride"]
            and contract.stride_phase == row["stride_phase"]
            and list(contract.seeds) == row["seed_order"]
            and contract.target_coverage == row["calibration_target_coverage"]
            and contract.num_leaves == row["model"]["num_leaves"]
            and contract.max_rounds == row["model"]["n_estimators"]
            and contract.early_stopping_rounds
            == row["model"]["early_stopping_rounds"]
            and contract.fit_session == row["fit_session"]
            and contract.calibration_session == row["calibration_session"]
            and contract.fit_ties == row["fit_ties"]
            and contract.early_stopping_ties == row["early_stopping_ties"]
            and contract.feature_dtype == row["feature_dtype"]
            and contract.replay_session == "dst_correct_ny"
        ):
            raise RefreshError(f"{pair}: executable adapter contract differs from spec")
        if structural["type"] == "none":
            if contract.structural_column is not None or contract.structural_quantile is not None:
                raise RefreshError(f"{pair}: executable structural gate differs from spec")
        elif not (
            structural["type"] == "column_quantile_le"
            and contract.structural_column == structural["column"]
            and contract.structural_quantile == structural["quantile"]
        ):
            raise RefreshError(f"{pair}: executable structural gate differs from spec")
        declared_model = row["model"]
        for seed in row["seed_order"]:
            actual = model_parameters(pair, seed, n_jobs=declared_model["num_threads"])
            expected = {
                key: declared_model[key]
                for key in (
                    "objective", "metric", "learning_rate", "num_leaves",
                    "min_child_samples", "subsample", "subsample_freq",
                    "colsample_bytree", "reg_lambda", "n_estimators",
                )
            }
            expected.update(
                n_jobs=declared_model["num_threads"],
                verbosity=-1,
                random_state=seed,
                bagging_seed=seed,
                feature_fraction_seed=seed,
                data_random_seed=seed,
                deterministic=declared_model["deterministic"],
                force_col_wise=declared_model["force_col_wise"],
            )
            if actual != expected:
                raise RefreshError(f"{pair}: seed {seed} model parameters differ from spec")
    # Prove the checked object round-trips through the exact canonical encoder.
    canonical_bytes(spec)
    return spec


def evaluator_payload(spec: Mapping[str, Any]) -> dict[str, Any]:
    payload = json.loads(json.dumps(spec["evaluator"]))
    if payload.get("schema") != "m15-book-refresh-evaluator/v1":
        raise RefreshError("evaluator schema mismatch")
    return payload


def evaluator_id(spec: Mapping[str, Any]) -> str:
    domain = spec["identity"]["domains"]["evaluator_id"]
    return length_prefixed_digest(domain, [canonical_bytes(evaluator_payload(spec))])


def derive_prereg_id(
    spec: Mapping[str, Any],
    prereg_payload: Mapping[str, Any],
    implementation_git_sha: str,
    *,
    spec_bytes: bytes | None = None,
) -> str:
    if not _is_hex(implementation_git_sha, 40):
        raise RefreshError("implementation git SHA must be 40 lowercase hex characters")
    payload = dict(prereg_payload)
    if "prereg_id" in payload:
        raise RefreshError("prereg payload must exclude derived prereg_id")
    ebytes = canonical_bytes(evaluator_payload(spec))
    eid = evaluator_id(spec)
    return length_prefixed_digest(
        spec["identity"]["domains"]["prereg_id"],
        [
            canonical_bytes(payload),
            spec_bytes if spec_bytes is not None else canonical_bytes(spec),
            ebytes,
            eid.encode("ascii"),
            implementation_git_sha.encode("ascii"),
        ],
    )


def derive_bundle_id(
    spec: Mapping[str, Any],
    resolved_bundle_spec: Mapping[str, Any],
    strategy_name: str,
    strategy_bytes: bytes,
    models: Sequence[tuple[str, bytes]],
) -> str:
    resolved = dict(resolved_bundle_spec)
    if "bundle_id" in resolved:
        raise RefreshError("resolved bundle spec must exclude derived bundle_id")
    if Path(strategy_name).name != strategy_name:
        raise RefreshError("strategy name must be a basename")
    seen: set[str] = set()
    components = [canonical_bytes(resolved), strategy_name.encode("utf-8"), strategy_bytes]
    for name, model_bytes in models:
        if Path(name).name != name or name in seen:
            raise RefreshError("model names must be unique basenames in declared seed order")
        seen.add(name)
        components.extend([name.encode("utf-8"), model_bytes])
    if not models:
        raise RefreshError("bundle identity requires at least one model")
    return length_prefixed_digest(spec["identity"]["domains"]["bundle_id"], components)


def derive_controls_id(spec: Mapping[str, Any], entries: Sequence[Sequence[str]]) -> str:
    expected = [list(row) for row in CONTROL_ORDER]
    if spec.get("control_order") != expected:
        raise RefreshError("control identity spec differs from the literal six-arm order")
    if spec.get("control_family", {}).get("representatives") != CONTROL_REPRESENTATIVES:
        raise RefreshError("control identity representative map differs")
    material: list[list[str]] = []
    if len(entries) != len(expected):
        raise RefreshError("controls identity requires all six ordered arms")
    for expected_key, entry in zip(expected, entries):
        row = list(entry)
        if len(row) != 3 or row[:2] != list(expected_key) or not _is_hex(row[2], 64):
            raise RefreshError("control identity order or bundle id mismatch")
        material.append(row)
    return length_prefixed_digest(
        spec["identity"]["domains"]["controls_id"], [canonical_bytes(material)]
    )


def derive_run_id(
    spec: Mapping[str, Any],
    prereg_id: str,
    controls_id: str,
    arm_entries: Sequence[Sequence[str]],
) -> str:
    if not _is_hex(prereg_id, 64) or not _is_hex(controls_id, 64):
        raise RefreshError("run identity requires valid prereg_id and controls_id")
    expected = [
        [pair, arm]
        for pair in spec["pair_order"]
        for arm in spec["arm_order"]
    ]
    if len(arm_entries) != 18:
        raise RefreshError("run identity requires the fixed 18 primary arms")
    material: list[list[str]] = []
    for key, entry in zip(expected, arm_entries):
        row = list(entry)
        if len(row) != 3 or row[:2] != key or not _is_hex(row[2], 64):
            raise RefreshError("run arm order or bundle id mismatch")
        material.append(row)
    return length_prefixed_digest(
        spec["identity"]["domains"]["run_id"],
        [prereg_id.encode("ascii"), controls_id.encode("ascii"), canonical_bytes(material)],
    )


def _is_hex(value: Any, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(ch in "0123456789abcdef" for ch in value)
    )


def atomic_write_new(path: str | Path, data: bytes, *, read_only: bool = False) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise RefreshError(f"sealed destination already exists: {destination}")
    fd, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.tmp-", dir=destination.parent)
    temporary = Path(temporary_name)
    try:
        view = memoryview(data)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise RefreshError(f"short write for {temporary}")
            view = view[written:]
        if read_only:
            # Set immutable publication mode on the inode before it acquires
            # the destination name.  There is no visible writable evidence
            # window for another process to modify.
            os.fchmod(fd, 0o444)
        os.fsync(fd)
        os.close(fd)
        fd = -1
        try:
            os.link(temporary, destination)
        except FileExistsError as exc:
            raise RefreshError(f"sealed destination appeared concurrently: {destination}") from exc
        temporary.unlink()
        _fsync_dir(destination.parent)
        return destination
    finally:
        if fd >= 0:
            os.close(fd)
        if temporary.exists():
            temporary.unlink()


def atomic_json_new(path: str | Path, value: Any, *, read_only: bool = True) -> Path:
    return atomic_write_new(path, canonical_bytes(value), read_only=read_only)


def _seal_bytes_exact(path: Path, data: bytes) -> Path:
    """Publish immutable bytes, or authenticate an identical interrupted write."""

    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file() or path.stat().st_mode & 0o222:
            raise RefreshError(f"existing sealed artifact is not read-only/regular: {path}")
        if path.read_bytes() != data:
            raise RefreshError(f"existing sealed artifact differs: {path}")
        return path
    return atomic_write_new(path, data, read_only=True)


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _remove_owned_tree(path: Path) -> None:
    """Remove only a campaign-owned partial tree, including read-only residue."""

    if not path.exists() and not path.is_symlink():
        return
    if path.is_symlink():
        path.unlink()
        return
    for child in sorted(path.rglob("*"), reverse=True):
        if child.is_symlink():
            continue
        try:
            child.chmod(0o700 if child.is_dir() else 0o600)
        except OSError:
            pass
    try:
        path.chmod(0o700)
    except OSError:
        pass
    shutil.rmtree(path)


def _record_new_exact_file(
    records: list[tuple[Path, bytes]],
    path: Path,
    *,
    existed_before: bool,
    expected: bytes,
) -> None:
    """Record only an exact file newly created by the current commit attempt."""

    if existed_before:
        return
    if path.is_symlink() or not path.is_file() or path.read_bytes() != expected:
        raise RefreshError(f"new commit artifact cannot be authenticated: {path}")
    records.append((path, expected))


def _rollback_new_exact_files(
    records: Sequence[tuple[Path, bytes]],
    *,
    new_directories: Sequence[Path] = (),
) -> None:
    """Remove only exact files/directories created by a failed commit guard."""

    for path, expected in reversed(list(records)):
        if path.is_symlink() or not path.is_file() or path.read_bytes() != expected:
            raise RefreshError(
                f"failed commit artifact changed; refusing unsafe rollback: {path}"
            )
        path.unlink()
        _fsync_dir(path.parent)
    for path in reversed(list(new_directories)):
        if path.is_symlink() or not path.is_dir():
            raise RefreshError(
                f"failed commit directory changed; refusing unsafe rollback: {path}"
            )
        try:
            path.rmdir()
        except OSError as exc:
            raise RefreshError(
                f"failed commit directory is not empty; refusing rollback: {path}"
            ) from exc
        _fsync_dir(path.parent)


class PhaseStore:
    """Forward-only, hash-verifying phase manifests for one prereg workspace."""

    def __init__(self, workspace: str | Path):
        self.workspace = Path(workspace)
        self.phase_dir = self.workspace / "phases"
        self.phase_dir.mkdir(parents=True, exist_ok=True)

    def path(self, phase: str) -> Path:
        try:
            index = PHASE_ORDER.index(phase)
        except ValueError as exc:
            raise RefreshError(f"unknown campaign phase {phase}") from exc
        return self.phase_dir / f"{index:02d}_{phase}.json"

    def load(self, phase: str) -> dict[str, Any]:
        path = self.path(phase)
        if path.is_symlink() or not path.is_file():
            raise RefreshError(f"required sealed phase is absent: {phase}")
        if path.stat().st_mode & 0o222:
            raise RefreshError(f"sealed phase is writable: {path}")
        try:
            value = json.loads(path.read_bytes())
        except (OSError, json.JSONDecodeError) as exc:
            raise RefreshError(f"cannot read phase manifest {path}: {exc}") from exc
        if not isinstance(value, dict) or set(value) != {
            "schema", "phase", "status", "canonical_attempt_id",
            "predecessor_manifest_hashes", "input_hashes", "output_hashes", "metadata",
        }:
            raise RefreshError(f"phase manifest fields differ: {path}")
        if value.get("schema") != "m15-book-refresh-phase/v1" or value.get("status") != "sealed":
            raise RefreshError(f"phase manifest is not sealed: {path}")
        if value.get("phase") != phase:
            raise RefreshError(f"phase name mismatch: {path}")
        index = PHASE_ORDER.index(phase)
        expected_predecessors: dict[str, str] = {}
        if index:
            prior = PHASE_ORDER[index - 1]
            prior_path = self.path(prior)
            self.load(prior)
            expected_predecessors[prior] = sha256_file(prior_path)
        if value.get("predecessor_manifest_hashes") != expected_predecessors:
            raise RefreshError(f"phase predecessor manifest hash differs: {path}")
        self._verify_hash_map(value.get("input_hashes", {}))
        self._verify_hash_map(value.get("output_hashes", {}))
        return value

    def has(self, phase: str) -> bool:
        path = self.path(phase)
        if not path.exists() and not path.is_symlink():
            return False
        self.load(phase)
        return True

    def seal(
        self,
        phase: str,
        *,
        canonical_attempt_id: str,
        input_paths: Sequence[str | Path] = (),
        output_paths: Sequence[str | Path] = (),
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        index = PHASE_ORDER.index(phase)
        predecessors: dict[str, str] = {}
        if index:
            prior = PHASE_ORDER[index - 1]
            prior_path = self.path(prior)
            self.load(prior)
            predecessors[prior] = sha256_file(prior_path)
        value = {
            "schema": "m15-book-refresh-phase/v1",
            "phase": phase,
            "status": "sealed",
            "canonical_attempt_id": canonical_attempt_id,
            "predecessor_manifest_hashes": predecessors,
            "input_hashes": self._identity_map(input_paths),
            "output_hashes": self._identity_map(output_paths),
            "metadata": dict(metadata or {}),
        }
        path = self.path(phase)
        encoded = canonical_bytes(value)
        if path.exists():
            if path.read_bytes() != encoded:
                raise RefreshError(f"sealed phase differs on resume: {phase}")
            self.load(phase)
            return value
        atomic_write_new(path, encoded, read_only=True)
        return value

    def require_replay_access(self) -> None:
        self.load(SEMANTIC_REPLAY_PHASE)

    def _identity_map(self, paths: Sequence[str | Path]) -> dict[str, str]:
        out: dict[str, str] = {}
        for raw in paths:
            path = Path(raw)
            if path.is_symlink() or not path.is_file():
                raise RefreshError(f"phase hash input must be a regular file: {path}")
            resolved = path.resolve()
            if resolved.is_relative_to(self.workspace.resolve()):
                key = "workspace:" + resolved.relative_to(self.workspace.resolve()).as_posix()
            else:
                key = "repo:" + repo_relative(path)
            if key in out:
                raise RefreshError(f"duplicate phase hash key: {key}")
            out[key] = sha256_file(path)
        return dict(sorted(out.items()))

    def _verify_hash_map(self, rows: Mapping[str, str]) -> None:
        if not isinstance(rows, dict):
            raise RefreshError("phase hash map is malformed")
        for name, expected in rows.items():
            if not isinstance(name, str) or not _is_hex(expected, 64):
                raise RefreshError("phase hash entry is malformed")
            if name.startswith("workspace:"):
                relative = name.removeprefix("workspace:")
                path = self.workspace / relative
                root = self.workspace.resolve()
            elif name.startswith("repo:"):
                relative = name.removeprefix("repo:")
                path = REPO_ROOT / relative
                root = REPO_ROOT.resolve()
            else:
                raise RefreshError(f"phase hash key has no authority prefix: {name}")
            try:
                path.resolve(strict=True).relative_to(root)
            except (OSError, ValueError) as exc:
                raise RefreshError(f"phase hash path escapes or is missing: {name}") from exc
            if path.is_symlink() or not path.is_file() or sha256_file(path) != expected:
                raise RefreshError(f"phase hash verification failed: {name}")


class SemanticAccessFence:
    """Injectable outcome reader that hard-fails until the arm seal verifies."""

    def __init__(self, phase_store: PhaseStore, reader: Callable[..., Any]):
        self.phase_store = phase_store
        self.reader = reader

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.phase_store.require_replay_access()
        return self.reader(*args, **kwargs)


def assert_strict_clock(ts: Sequence[int] | np.ndarray, *, name: str) -> np.ndarray:
    values = np.asarray(ts)
    if values.ndim != 1 or values.dtype.kind not in "iu":
        raise RefreshError(f"{name}: timestamps must be a one-dimensional integer vector")
    if len(values) and np.any(values[1:] <= values[:-1]):
        raise RefreshError(f"{name}: timestamps must be strictly increasing and unique")
    return values.astype("int64", copy=False)


def checked_seconds_from_ns(values: Sequence[int] | np.ndarray, *, name: str) -> np.ndarray:
    ns = assert_strict_clock(values, name=name)
    if np.any(ns % 1_000_000_000 != 0):
        raise RefreshError(f"{name}: nanosecond IDs are not whole epoch seconds")
    return ns // 1_000_000_000


def git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True,
        text=True, stdout=subprocess.PIPE,
    ).stdout.strip()


def git_porcelain() -> str:
    return subprocess.run(
        ["git", "status", "--porcelain"], cwd=REPO_ROOT, check=True,
        text=True, stdout=subprocess.PIPE,
    ).stdout


def issue_body() -> str:
    token = subprocess.run(
        ["gh", "auth", "token", "--user", "sppburke"], check=True,
        text=True, stdout=subprocess.PIPE,
    ).stdout.strip()
    env = dict(os.environ, GH_TOKEN=token)
    return subprocess.run(
        ["gh", "issue", "view", "9", "-R", "sppburke/binary-algo", "--json", "body", "--jq", ".body"],
        check=True, text=True, stdout=subprocess.PIPE, env=env,
    ).stdout


def source_files() -> list[Path]:
    names = [
        "ENVIRONMENT_libs.txt",
        "scripts/m15_book_refresh.py",
        "scripts/m15_book_refresh_specs.json",
        "scripts/m15_book_refresh_adapters.py",
        "scripts/m15_book_refresh_replay.py",
        "scripts/m15_book_refresh_stats.py",
        "scripts/m15_phase_zero_behavior_authority.py",
        "scripts/test_m15_book_refresh.py",
        "scripts/live_features.py",
        "scripts/manifest.py",
        "scripts/book_runtime.py",
        "scripts/deriv_floor_resolver.py",
        "scripts/deriv_hot_daemon.py",
        "scripts/deriv_runtime_core.py",
        "scripts/harness.py",
        "scripts/min1_production.py",
        "scripts/sessions.py",
        "scripts/pipeline.py",
        "scripts/orderflow.py",
        "scripts/m5_xpair.py",
        "scripts/gbpusd_15m_xpair.py",
        "scripts/gbpusd_15m_xpair_frozen.py",
        "scripts/usdchf_15m_xpair.py",
        "scripts/usdchf_15m_xpair_frozen.py",
        "scripts/usdchf_15m_freeze_xpny.py",
        "scripts/usdjpy_15m_base.py",
        "scripts/usdcad_15m_base.py",
        "scripts/audusd_15m_base.py",
        "scripts/nzdusd_15m_base.py",
    ]
    return [REPO_ROOT / name for name in names]


RUNTIME_DISTRIBUTIONS = (
    "numpy", "pandas", "pyarrow", "lightgbm", "scikit-learn",
)


def verified_runtime_version_contract() -> dict[str, Any]:
    """Fail closed unless the active interpreter matches tracked exact pins."""

    import importlib.metadata
    import site

    expected_prefix = Path.home() / "binary-algo-venv"
    expected_executable = expected_prefix / "bin" / "python"
    python_environment = {
        "executable": str(Path(sys.executable).absolute()),
        "prefix": str(Path(sys.prefix).absolute()),
        "base_prefix": str(Path(sys.base_prefix).absolute()),
        "expected_executable": str(expected_executable),
        "expected_prefix": str(expected_prefix),
        "PYTHONNOUSERSITE": os.environ.get("PYTHONNOUSERSITE"),
        "no_user_site_flag": int(sys.flags.no_user_site),
        "user_site_enabled": site.ENABLE_USER_SITE,
    }
    if (
        Path(sys.executable).absolute() != expected_executable
        or Path(sys.prefix).absolute() != expected_prefix
        or os.environ.get("PYTHONNOUSERSITE") != "1"
        or sys.flags.no_user_site != 1
        or site.ENABLE_USER_SITE is not False
    ):
        raise RefreshError(
            "measurement runtime must use the campaign venv with "
            f"PYTHONNOUSERSITE=1: {python_environment}"
        )

    if ENVIRONMENT_PATH.is_symlink() or not ENVIRONMENT_PATH.is_file():
        raise RefreshError("tracked runtime authority is missing/nonregular")
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", "ENVIRONMENT_libs.txt"],
        cwd=REPO_ROOT,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if tracked.returncode != 0 or tracked.stdout.strip() != "ENVIRONMENT_libs.txt":
        raise RefreshError("runtime authority is not the tracked ENVIRONMENT_libs.txt")
    pins: dict[str, str] = {}
    try:
        lines = ENVIRONMENT_PATH.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise RefreshError("cannot read tracked runtime authority") from exc
    for line in lines:
        value = line.strip()
        if not value or value.startswith("#") or "==" not in value:
            continue
        name, version = value.split("==", 1)
        normalized = name.strip().lower()
        exact = version.strip()
        if not normalized or not exact or normalized in pins:
            raise RefreshError("runtime authority contains malformed/duplicate pins")
        pins[normalized] = exact
    try:
        expected = {name: pins[name] for name in RUNTIME_DISTRIBUTIONS}
    except KeyError as exc:
        raise RefreshError(f"runtime authority lacks exact pin {exc.args[0]}") from exc
    try:
        actual = {
            name: importlib.metadata.version(name) for name in RUNTIME_DISTRIBUTIONS
        }
    except importlib.metadata.PackageNotFoundError as exc:
        raise RefreshError(f"required runtime distribution is absent: {exc}") from exc
    contract = {
        "authority": "ENVIRONMENT_libs.txt",
        "authority_file": asdict(file_identity(ENVIRONMENT_PATH)),
        "python_environment": python_environment,
        "expected_exact": expected,
        "actual_exact": actual,
        "all_exact": actual == expected,
    }
    if contract["all_exact"] is not True:
        differences = {
            name: {"expected": expected[name], "actual": actual[name]}
            for name in RUNTIME_DISTRIBUTIONS
            if expected[name] != actual[name]
        }
        raise RefreshError(f"runtime distributions differ from exact pins: {differences}")
    return contract


PHASE_ZERO_VERDICT_CHECKS = frozenset(
    {
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
)


def _phase_zero_current_incumbents(
    spec: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Resolve the exact current A schema and INDEX→model file inventory."""

    import book_runtime

    books = book_runtime.load_target_books(list(spec["pair_order"]))
    index = book_runtime.load_index()
    if list(books) != list(spec["pair_order"]):
        raise RefreshError("phase-zero current comparator-A order/set differs")
    repo_root = REPO_ROOT.resolve(strict=True)

    def relative(path: Path) -> str:
        try:
            return path.resolve(strict=True).relative_to(repo_root).as_posix()
        except (OSError, ValueError) as exc:
            raise RefreshError(
                f"phase-zero incumbent file escapes repository: {path}"
            ) from exc

    output: dict[str, dict[str, Any]] = {}
    index_path = REPO_ROOT / "books" / "INDEX.json"
    for pair in spec["pair_order"]:
        book = books[pair]
        expected_book_id = spec["pairs"][pair]["book_id_A"]
        index_entry = index.get(expected_book_id)
        if (
            getattr(book, "pair", None) != pair
            or getattr(book, "book_id", None) != expected_book_id
            or not isinstance(index_entry, Mapping)
            or not isinstance(getattr(book, "content_id", None), str)
            or not book.content_id
        ):
            raise RefreshError(f"{pair}: phase-zero current incumbent identity differs")
        manifest_path, _legacy = book_runtime.load_registered_manifest(
            expected_book_id, dict(index_entry)
        )
        model_paths = tuple(Path(value) for value in book.model_paths)
        ordered_files = (
            index_path,
            Path(manifest_path),
            Path(book.strategy_path),
            *model_paths,
        )
        if (
            not model_paths
            or len({relative(path) for path in ordered_files})
            != len(ordered_files)
        ):
            raise RefreshError(f"{pair}: phase-zero current incumbent files differ")
        feature_cols = list(book.feature_cols)
        if (
            len(feature_cols) != spec["pairs"][pair]["feature_count"]
            or len(feature_cols) != len(set(feature_cols))
            or any(not isinstance(column, str) or not column for column in feature_cols)
        ):
            raise RefreshError(f"{pair}: phase-zero current A feature schema differs")
        output[pair] = {
            "book_id": expected_book_id,
            "content_id": book.content_id,
            "feature_cols": feature_cols,
            "model_order": [relative(path) for path in model_paths],
            "ordered_files": ordered_files,
        }
    return output


def _authority_file_identity(row: Any, *, name: str) -> Path:
    """Authenticate one phase-zero input identity without following symlinks."""

    if (
        not isinstance(row, dict)
        or set(row) != {"path", "bytes", "sha256"}
        or not isinstance(row.get("path"), str)
        or not row["path"]
        or not isinstance(row.get("bytes"), int)
        or isinstance(row.get("bytes"), bool)
        or row["bytes"] < 0
        or not _is_hex(row.get("sha256"), 64)
    ):
        raise RefreshError(f"{name}: phase-zero file identity is malformed")
    declared = Path(row["path"])
    if declared.is_absolute():
        target = declared
    else:
        if ".." in declared.parts:
            raise RefreshError(f"{name}: phase-zero path contains '..'")
        target = REPO_ROOT / declared
        try:
            target.resolve(strict=True).relative_to(REPO_ROOT.resolve(strict=True))
        except (OSError, ValueError) as exc:
            raise RefreshError(f"{name}: phase-zero path escapes the repository") from exc
    if target.is_symlink() or not target.is_file():
        raise RefreshError(f"{name}: phase-zero input is missing/nonregular: {target}")
    if target.stat().st_size != row["bytes"] or sha256_file(target) != row["sha256"]:
        raise RefreshError(f"{name}: phase-zero input bytes differ: {target}")
    return target


PHASE_ZERO_CLOCK_FIELDS = {
    "rows",
    "first_utc",
    "last_utc",
    "clock_i64le_sha256",
    "strictly_increasing",
    "unique",
    "ten_second_lattice_exact",
    "all_gaps_positive_multiples_of_10s",
    "gap_gt_10s_count",
    "minimum_gap_seconds",
    "maximum_gap_seconds",
}


def _phase_zero_integer(
    value: Any, *, name: str, minimum: int = 0
) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise RefreshError(f"{name}: phase-zero integer evidence is malformed")
    return value


def _phase_zero_hash(value: Any, *, name: str) -> str:
    if not _is_hex(value, 64):
        raise RefreshError(f"{name}: phase-zero digest evidence is malformed")
    return str(value)


def _validate_phase_zero_clock_seal(value: Any, *, name: str) -> tuple[int, str]:
    if not isinstance(value, dict) or set(value) != PHASE_ZERO_CLOCK_FIELDS:
        raise RefreshError(f"{name}: phase-zero clock-seal fields differ")
    rows = _phase_zero_integer(value.get("rows"), name=f"{name}.rows", minimum=1)
    digest = _phase_zero_hash(
        value.get("clock_i64le_sha256"), name=f"{name}.clock"
    )
    if (
        not isinstance(value.get("first_utc"), str)
        or not value["first_utc"]
        or not isinstance(value.get("last_utc"), str)
        or not value["last_utc"]
        or value.get("strictly_increasing") is not True
        or value.get("unique") is not True
        or value.get("ten_second_lattice_exact") is not True
        or value.get("all_gaps_positive_multiples_of_10s") is not True
    ):
        raise RefreshError(f"{name}: phase-zero strict clock claims differ")
    gap_count = _phase_zero_integer(
        value.get("gap_gt_10s_count"), name=f"{name}.gap_count"
    )
    if gap_count > max(0, rows - 1):
        raise RefreshError(f"{name}: phase-zero gap count exceeds clock length")
    minimum_gap = value.get("minimum_gap_seconds")
    maximum_gap = value.get("maximum_gap_seconds")
    if rows == 1:
        if minimum_gap is not None or maximum_gap is not None:
            raise RefreshError(f"{name}: singleton phase-zero clock has gap evidence")
    elif (
        isinstance(minimum_gap, bool)
        or not isinstance(minimum_gap, int)
        or isinstance(maximum_gap, bool)
        or not isinstance(maximum_gap, int)
        or minimum_gap < 10
        or maximum_gap < minimum_gap
        or minimum_gap % 10
        or maximum_gap % 10
    ):
        raise RefreshError(f"{name}: phase-zero clock gap bounds differ")
    return rows, digest


def _validate_phase_zero_decoded_diagnostics(
    value: Any,
    *,
    feature_cols: Sequence[str],
    name: str,
    require_zero: bool,
) -> tuple[int, int]:
    if not isinstance(value, dict) or set(value) != {
        "mismatched_cells",
        "finite_nonfinite_disagreements",
        "columns",
    }:
        raise RefreshError(f"{name}: decoded diagnostics fields differ")
    total = _phase_zero_integer(
        value.get("mismatched_cells"), name=f"{name}.mismatched_cells"
    )
    finite = _phase_zero_integer(
        value.get("finite_nonfinite_disagreements"),
        name=f"{name}.finite_nonfinite_disagreements",
    )
    columns = value.get("columns")
    if not isinstance(columns, dict) or any(key not in feature_cols for key in columns):
        raise RefreshError(f"{name}: decoded diagnostic columns differ")
    summed_total = 0
    summed_finite = 0
    for column, row in columns.items():
        if not isinstance(row, dict) or set(row) != {
            "mismatched_cells",
            "finite_nonfinite_disagreements",
            "max_abs_difference",
            "max_ulp_difference",
        }:
            raise RefreshError(f"{name}.{column}: decoded diagnostic fields differ")
        column_total = _phase_zero_integer(
            row.get("mismatched_cells"),
            name=f"{name}.{column}.mismatched_cells",
            minimum=1,
        )
        column_finite = _phase_zero_integer(
            row.get("finite_nonfinite_disagreements"),
            name=f"{name}.{column}.finite_nonfinite_disagreements",
        )
        if column_finite > column_total:
            raise RefreshError(f"{name}.{column}: finite mismatch count differs")
        maximum_absolute = row.get("max_abs_difference")
        maximum_ulp = row.get("max_ulp_difference")
        if maximum_absolute is not None and (
            isinstance(maximum_absolute, bool)
            or not isinstance(maximum_absolute, (int, float))
            or not np.isfinite(float(maximum_absolute))
            or float(maximum_absolute) < 0.0
        ):
            raise RefreshError(f"{name}.{column}: maximum absolute drift differs")
        if maximum_ulp is not None:
            _phase_zero_integer(
                maximum_ulp, name=f"{name}.{column}.max_ulp_difference"
            )
        summed_total += column_total
        summed_finite += column_finite
    if total != summed_total or finite != summed_finite:
        raise RefreshError(f"{name}: decoded diagnostic aggregates differ")
    if finite != 0 or (require_zero and (total != 0 or columns != {})):
        raise RefreshError(f"{name}: binding decoded behavior differs")
    return total, finite


def _validate_phase_zero_source_rebuild(
    pair: str,
    value: Any,
    *,
    expected_projection: Sequence[str],
    expected_processed_identities: Sequence[Mapping[str, Any]],
    expected_raw_schema: Sequence[str],
) -> dict[str, Any]:
    required = {
        "strict_processed_source",
        "causal_resampled_close_availability",
        "processed_raw_derived_identity",
        "raw_rebuilt_q1_feature_view",
        "cached_q1_feature_view",
        "live_provider_rolling_store",
        "raw_source_loaded_and_rebuilt_once",
        "semantic_or_outcome_columns_read",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise RefreshError(f"{pair}: raw rebuild evidence fields differ")
    if (
        value.get("raw_source_loaded_and_rebuilt_once") is not True
        or value.get("semantic_or_outcome_columns_read") is not False
    ):
        raise RefreshError(f"{pair}: raw rebuild access claims differ")

    strict = value.get("strict_processed_source")
    strict_fields = {
        "loader",
        "projection_in_order",
        "schema_policy",
        "sort_or_dedup_performed",
        "semantic_or_outcome_columns_read",
        "ordered_files",
        "global_clock",
    }
    if (
        not isinstance(strict, dict)
        or set(strict) != strict_fields
        or strict.get("loader")
        != "m15_phase_zero_behavior_authority._load_processed_q1_strict"
        or strict.get("projection_in_order") != list(expected_raw_schema)
        or strict.get("schema_policy") != "exact_schema_only_no_extra_columns"
        or strict.get("sort_or_dedup_performed") is not False
        or strict.get("semantic_or_outcome_columns_read") is not False
        or not isinstance(strict.get("ordered_files"), list)
        or len(strict["ordered_files"]) != len(expected_processed_identities)
    ):
        raise RefreshError(f"{pair}: strict processed-source evidence differs")
    expected_types = [
        "double",
        "double",
        "double",
        "double",
        "double",
        "timestamp[ns, tz=UTC]",
    ]
    file_rows = 0
    first_utc: str | None = None
    last_utc: str | None = None
    for index, (reported, identity) in enumerate(
        zip(strict["ordered_files"], expected_processed_identities)
    ):
        if not isinstance(reported, dict) or set(reported) != {
            "path",
            "schema_in_order",
            *PHASE_ZERO_CLOCK_FIELDS,
        }:
            raise RefreshError(f"{pair}: strict processed file fields differ at {index}")
        if reported.get("path") != identity.get("path"):
            raise RefreshError(f"{pair}: strict processed file order/path differs")
        schema = reported.get("schema_in_order")
        if (
            not isinstance(schema, list)
            or len(schema) != len(expected_raw_schema)
            or [field.get("name") for field in schema if isinstance(field, dict)]
            != list(expected_raw_schema)
            or [field.get("type") for field in schema if isinstance(field, dict)]
            != expected_types
            or any(
                not isinstance(field, dict)
                or set(field) != {"name", "type", "nullable"}
                or type(field.get("nullable")) is not bool
                for field in schema
            )
        ):
            raise RefreshError(f"{pair}: strict processed schema evidence differs")
        clock = {key: reported[key] for key in PHASE_ZERO_CLOCK_FIELDS}
        rows, _ = _validate_phase_zero_clock_seal(
            clock, name=f"{pair}.strict_file[{index}]"
        )
        file_rows += rows
        if index == 0:
            first_utc = reported["first_utc"]
        last_utc = reported["last_utc"]
    global_rows, global_hash = _validate_phase_zero_clock_seal(
        strict.get("global_clock"), name=f"{pair}.strict_global_clock"
    )
    if (
        global_rows != file_rows
        or strict["global_clock"].get("first_utc") != first_utc
        or strict["global_clock"].get("last_utc") != last_utc
    ):
        raise RefreshError(f"{pair}: strict processed global clock is inconsistent")

    causal = value.get("causal_resampled_close_availability")
    causal_fields = {
        "rows",
        "raw_bar_label_semantics",
        "close_availability_mapping",
        "decision_mapping",
        "every_close_available_no_later_than_decision",
        "minute_source_clock_i64le_sha256",
        "last_raw_label_i64le_sha256",
        "close_availability_i64le_sha256",
        "decision_clock_i64le_sha256",
        "resampled_close_f64le_sha256",
        "minimum_availability_slack_seconds",
        "maximum_availability_slack_seconds",
    }
    if not isinstance(causal, dict) or set(causal) != causal_fields:
        raise RefreshError(f"{pair}: causal resampling evidence fields differ")
    minute_rows = _phase_zero_integer(
        causal.get("rows"), name=f"{pair}.causal.rows", minimum=1
    )
    minute_hash = _phase_zero_hash(
        causal.get("minute_source_clock_i64le_sha256"),
        name=f"{pair}.causal.minute_clock",
    )
    if (
        causal.get("raw_bar_label_semantics") != "left_edge_of_[label,label+10s)_bar"
        or causal.get("close_availability_mapping") != "raw_bar_label_plus_10_seconds"
        or causal.get("decision_mapping") != "minute_source_label_plus_60_seconds"
        or causal.get("every_close_available_no_later_than_decision") is not True
    ):
        raise RefreshError(f"{pair}: causal resampling claims differ")
    for field in (
        "last_raw_label_i64le_sha256",
        "close_availability_i64le_sha256",
        "decision_clock_i64le_sha256",
        "resampled_close_f64le_sha256",
    ):
        _phase_zero_hash(causal.get(field), name=f"{pair}.causal.{field}")
    minimum_slack = _phase_zero_integer(
        causal.get("minimum_availability_slack_seconds"),
        name=f"{pair}.causal.minimum_slack",
    )
    maximum_slack = _phase_zero_integer(
        causal.get("maximum_availability_slack_seconds"),
        name=f"{pair}.causal.maximum_slack",
    )
    if maximum_slack < minimum_slack or maximum_slack > 60:
        raise RefreshError(f"{pair}: causal availability slack differs")

    derived = value.get("processed_raw_derived_identity")
    if not isinstance(derived, dict) or set(derived) != {
        "observation_rows",
        "observation_clock_i64le_sha256",
        "one_minute_rows",
        "one_minute_clock_i64le_sha256",
        "one_minute_ohlcv_gap_f64le_sha256",
    }:
        raise RefreshError(f"{pair}: processed/raw derived identity fields differ")
    if (
        derived.get("observation_rows") != global_rows
        or derived.get("observation_clock_i64le_sha256") != global_hash
        or derived.get("one_minute_rows") != minute_rows
        or derived.get("one_minute_clock_i64le_sha256") != minute_hash
    ):
        raise RefreshError(f"{pair}: processed/raw derived clock identity differs")
    _phase_zero_hash(
        derived.get("one_minute_ohlcv_gap_f64le_sha256"),
        name=f"{pair}.derived.one_minute_values",
    )

    view_fields = {
        "projection_in_order",
        "rows",
        "source_clock_i64le_sha256",
        "ordered_float32_values_sha256",
    }
    views: dict[str, dict[str, Any]] = {}
    for label in ("raw_rebuilt_q1_feature_view", "cached_q1_feature_view"):
        view = value.get(label)
        if (
            not isinstance(view, dict)
            or set(view) != view_fields
            or view.get("projection_in_order") != list(expected_projection)
        ):
            raise RefreshError(f"{pair}: {label} evidence differs")
        bounded_rows = _phase_zero_integer(
            view.get("rows"), name=f"{pair}.{label}.rows", minimum=1
        )
        if bounded_rows > minute_rows:
            raise RefreshError(f"{pair}: {label} exceeds the full minute source")
        _phase_zero_hash(
            view.get("source_clock_i64le_sha256"),
            name=f"{pair}.{label}.clock",
        )
        _phase_zero_hash(
            view.get("ordered_float32_values_sha256"),
            name=f"{pair}.{label}.values",
        )
        views[label] = view
    raw_view = views["raw_rebuilt_q1_feature_view"]
    cached_view = views["cached_q1_feature_view"]
    bounded_rows = raw_view["rows"]
    bounded_clock_hash = raw_view["source_clock_i64le_sha256"]
    if (
        cached_view["rows"] != bounded_rows
        or cached_view["source_clock_i64le_sha256"] != bounded_clock_hash
    ):
        raise RefreshError(f"{pair}: bounded raw/cached view clocks differ")
    rolling = value.get("live_provider_rolling_store")
    if (
        not isinstance(rolling, dict)
        or set(rolling)
        != {
            "rows",
            "source_clock_i64le_sha256",
            "ordered_ohlcv_gap_f64le_sha256",
        }
        or rolling.get("rows") != bounded_rows
        or rolling.get("source_clock_i64le_sha256") != bounded_clock_hash
    ):
        raise RefreshError(f"{pair}: live-provider rolling-store evidence differs")
    _phase_zero_hash(
        rolling.get("ordered_ohlcv_gap_f64le_sha256"),
        name=f"{pair}.live_provider_rolling_store.values",
    )
    return {
        "observation_rows": global_rows,
        "observation_clock_sha256": global_hash,
        "full_minute_rows": minute_rows,
        "full_minute_clock_sha256": minute_hash,
        "rows": bounded_rows,
        "source_clock_sha256": bounded_clock_hash,
    }


def _validate_phase_zero_pair_behavior(
    pair: str,
    value: Any,
    *,
    feature_cols: Sequence[str],
) -> None:
    required = {
        "verdict",
        "verdict_checks",
        "inputs",
        "rows",
        "feature_count",
        "feature_dtype",
        "source_feature_clock_i64le_sha256",
        "decision_clock_i64le_sha256",
        "causal_decision_mapping_exact",
        "ordered_float32_features",
        "finite_eligibility",
        "scoreability_mismatch_source",
        "common_loadedbook_probability",
        "gate_components",
        "ordered_pre_schedule",
        "full_population_ordered_pre_schedule",
        "provider_qualification",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise RefreshError(f"{pair}: phase-zero pair behavior fields differ")
    rows = _phase_zero_integer(
        value.get("rows"), name=f"{pair}.behavior.rows", minimum=1
    )
    if (
        value.get("feature_count") != len(feature_cols)
        or value.get("feature_dtype") != "float32"
        or value.get("causal_decision_mapping_exact") is not True
    ):
        raise RefreshError(f"{pair}: phase-zero score-adapter evidence differs")
    _phase_zero_hash(
        value.get("source_feature_clock_i64le_sha256"),
        name=f"{pair}.behavior.source_clock",
    )
    _phase_zero_hash(
        value.get("decision_clock_i64le_sha256"),
        name=f"{pair}.behavior.decision_clock",
    )

    features = value.get("ordered_float32_features")
    if not isinstance(features, dict) or set(features) != {
        "cached_sha256",
        "raw_rebuilt_sha256",
        "bitwise_equal",
        "behavior_neutral_decoded_drift",
        "decoded_diagnostics",
    }:
        raise RefreshError(f"{pair}: ordered feature evidence fields differ")
    cached_features_hash = _phase_zero_hash(
        features.get("cached_sha256"), name=f"{pair}.features.cached"
    )
    raw_features_hash = _phase_zero_hash(
        features.get("raw_rebuilt_sha256"), name=f"{pair}.features.raw"
    )
    if (
        type(features.get("bitwise_equal")) is not bool
        or features.get("behavior_neutral_decoded_drift") is not True
        or features["bitwise_equal"]
        is not (cached_features_hash == raw_features_hash)
    ):
        raise RefreshError(f"{pair}: ordered feature equality claims differ")
    decoded_mismatches, _ = _validate_phase_zero_decoded_diagnostics(
        features.get("decoded_diagnostics"),
        feature_cols=feature_cols,
        name=f"{pair}.features.decoded",
        require_zero=False,
    )
    if features["bitwise_equal"] is True and decoded_mismatches != 0:
        raise RefreshError(f"{pair}: bitwise feature equality contradicts decoded drift")

    eligibility = value.get("finite_eligibility")
    if not isinstance(eligibility, dict) or set(eligibility) != {
        "cached_count",
        "raw_rebuilt_count",
        "common_count",
        "mismatch_count",
        "cached_u8_sha256",
        "raw_rebuilt_u8_sha256",
    }:
        raise RefreshError(f"{pair}: finite-eligibility evidence fields differ")
    cached_finite = _phase_zero_integer(
        eligibility.get("cached_count"), name=f"{pair}.finite.cached", minimum=1
    )
    raw_finite = _phase_zero_integer(
        eligibility.get("raw_rebuilt_count"), name=f"{pair}.finite.raw", minimum=1
    )
    common_finite = _phase_zero_integer(
        eligibility.get("common_count"), name=f"{pair}.finite.common", minimum=1
    )
    if (
        cached_finite != raw_finite
        or cached_finite != common_finite
        or cached_finite > rows
        or eligibility.get("mismatch_count") != 0
        or _phase_zero_hash(
            eligibility.get("cached_u8_sha256"), name=f"{pair}.finite.cached_hash"
        )
        != _phase_zero_hash(
            eligibility.get("raw_rebuilt_u8_sha256"), name=f"{pair}.finite.raw_hash"
        )
    ):
        raise RefreshError(f"{pair}: finite-eligibility equality evidence differs")

    mismatch = value.get("scoreability_mismatch_source")
    if not isinstance(mismatch, dict) or set(mismatch) != {
        "columns",
        "rows",
        "cached_finite_raw_nonfinite",
        "cached_nonfinite_raw_finite",
        "first_decision_utc",
        "last_decision_utc",
        "decision_i64le_sha256",
    }:
        raise RefreshError(f"{pair}: scoreability mismatch evidence fields differ")
    if (
        mismatch.get("columns") != []
        or mismatch.get("rows") != 0
        or mismatch.get("cached_finite_raw_nonfinite") != 0
        or mismatch.get("cached_nonfinite_raw_finite") != 0
        or mismatch.get("first_decision_utc") is not None
        or mismatch.get("last_decision_utc") is not None
        or mismatch.get("decision_i64le_sha256") != sha256_bytes(b"")
    ):
        raise RefreshError(f"{pair}: scoreability mismatch evidence is nonempty")

    probability = value.get("common_loadedbook_probability")
    if not isinstance(probability, dict) or set(probability) != {
        "population",
        "scorer",
        "rows",
        "bit_mismatch_count",
        "direction_mismatch_count",
        "max_abs_difference",
        "cached_f64le_sha256",
        "raw_rebuilt_f64le_sha256",
        "loadedbook_scalar_parity",
    }:
        raise RefreshError(f"{pair}: LoadedBook probability evidence fields differ")
    if (
        probability.get("population") != "common_finite_rows_only"
        or probability.get("scorer")
        != "book_runtime.LoadedBook.predict_probabilities"
        or probability.get("rows") != common_finite
        or probability.get("bit_mismatch_count") != 0
        or probability.get("direction_mismatch_count") != 0
        or probability.get("max_abs_difference") != 0.0
        or _phase_zero_hash(
            probability.get("cached_f64le_sha256"),
            name=f"{pair}.probability.cached",
        )
        != _phase_zero_hash(
            probability.get("raw_rebuilt_f64le_sha256"),
            name=f"{pair}.probability.raw",
        )
    ):
        raise RefreshError(f"{pair}: LoadedBook probability equality evidence differs")
    scalar = probability.get("loadedbook_scalar_parity")
    if not isinstance(scalar, dict) or set(scalar) != {
        "probe_rows",
        "cached_raw_probability_bits_exact",
        "cached_raw_direction_exact",
        "cached_raw_gate_exact",
        "scalar_batch_direction_exact",
        "scalar_batch_gate_exact",
        "scalar_batch_probability_bits_exact_diagnostic",
        "scalar_batch_probability_bits_binding",
        "behavior_exact",
    }:
        raise RefreshError(f"{pair}: LoadedBook scalar evidence fields differ")
    probe_rows = _phase_zero_integer(
        scalar.get("probe_rows"), name=f"{pair}.scalar.probe_rows", minimum=1
    )
    if (
        probe_rows != min(257, common_finite)
        or any(
            scalar.get(field) is not True
            for field in (
                "cached_raw_probability_bits_exact",
                "cached_raw_direction_exact",
                "cached_raw_gate_exact",
                "scalar_batch_direction_exact",
                "scalar_batch_gate_exact",
                "behavior_exact",
            )
        )
        or type(scalar.get("scalar_batch_probability_bits_exact_diagnostic"))
        is not bool
        or scalar.get("scalar_batch_probability_bits_binding") is not False
    ):
        raise RefreshError(f"{pair}: LoadedBook scalar behavior evidence differs")

    gates = value.get("gate_components")
    if not isinstance(gates, dict) or set(gates) != {
        "confidence",
        "structural",
        "combined",
    }:
        raise RefreshError(f"{pair}: gate-component evidence fields differ")
    gate_counts: dict[str, int] = {}
    for component in ("confidence", "structural", "combined"):
        row = gates[component]
        if not isinstance(row, dict) or set(row) != {
            "cached_pass_count",
            "raw_rebuilt_pass_count",
            "mismatch_count",
        }:
            raise RefreshError(f"{pair}: {component} gate evidence fields differ")
        cached_count = _phase_zero_integer(
            row.get("cached_pass_count"), name=f"{pair}.gate.{component}.cached"
        )
        raw_count = _phase_zero_integer(
            row.get("raw_rebuilt_pass_count"), name=f"{pair}.gate.{component}.raw"
        )
        if cached_count != raw_count or cached_count > common_finite or row.get(
            "mismatch_count"
        ) != 0:
            raise RefreshError(f"{pair}: {component} gate equality evidence differs")
        gate_counts[component] = cached_count
    if gate_counts["combined"] > min(
        gate_counts["confidence"], gate_counts["structural"]
    ):
        raise RefreshError(f"{pair}: combined gate count exceeds its components")

    selected = value.get("ordered_pre_schedule")
    if not isinstance(selected, dict) or set(selected) != {
        "population",
        "cached_count",
        "raw_rebuilt_count",
        "exact_equal",
        "cached_i64le_sha256",
        "raw_rebuilt_i64le_sha256",
    }:
        raise RefreshError(f"{pair}: ordered pre-schedule evidence fields differ")
    if (
        selected.get("population") != "common_finite_rows_only"
        or selected.get("cached_count") != gate_counts["combined"]
        or selected.get("raw_rebuilt_count") != gate_counts["combined"]
        or selected.get("exact_equal") is not True
        or _phase_zero_hash(
            selected.get("cached_i64le_sha256"), name=f"{pair}.selected.cached"
        )
        != _phase_zero_hash(
            selected.get("raw_rebuilt_i64le_sha256"), name=f"{pair}.selected.raw"
        )
    ):
        raise RefreshError(f"{pair}: ordered pre-schedule equality evidence differs")

    full = value.get("full_population_ordered_pre_schedule")
    if not isinstance(full, dict) or set(full) != {
        "population",
        "cached_eligible_rows",
        "raw_rebuilt_eligible_rows",
        "cached_count",
        "raw_rebuilt_count",
        "exact_equal",
        "actual_pre_schedule_behavior_changed",
        "symmetric_difference_count",
        "cached_i64le_sha256",
        "raw_rebuilt_i64le_sha256",
        "cached_probability_f64le_sha256",
        "raw_rebuilt_probability_f64le_sha256",
    }:
        raise RefreshError(f"{pair}: full-population pre-schedule fields differ")
    if (
        full.get("population") != "each_side_full_finite_population"
        or full.get("cached_eligible_rows") != cached_finite
        or full.get("raw_rebuilt_eligible_rows") != raw_finite
        or full.get("cached_count") != selected.get("cached_count")
        or full.get("raw_rebuilt_count") != selected.get("raw_rebuilt_count")
        or full.get("exact_equal") is not True
        or full.get("actual_pre_schedule_behavior_changed") is not False
        or full.get("symmetric_difference_count") != 0
        or _phase_zero_hash(
            full.get("cached_i64le_sha256"), name=f"{pair}.full.cached"
        )
        != _phase_zero_hash(
            full.get("raw_rebuilt_i64le_sha256"), name=f"{pair}.full.raw"
        )
        or _phase_zero_hash(
            full.get("cached_probability_f64le_sha256"),
            name=f"{pair}.full_probability.cached",
        )
        != _phase_zero_hash(
            full.get("raw_rebuilt_probability_f64le_sha256"),
            name=f"{pair}.full_probability.raw",
        )
        or full.get("cached_probability_f64le_sha256")
        != probability.get("cached_f64le_sha256")
        or full.get("raw_rebuilt_probability_f64le_sha256")
        != probability.get("raw_rebuilt_f64le_sha256")
    ):
        raise RefreshError(f"{pair}: full-population behavior evidence differs")


def _validate_phase_zero_authority(path: str | Path = PHASE_ZERO_AUTHORITY_PATH) -> dict[str, Any]:
    """Require a final-code, six-target, seven-source outcome-blind acceptance."""

    authority_path = Path(path)
    if (
        authority_path.is_symlink()
        or not authority_path.is_file()
        or authority_path.stat().st_mode & 0o222
    ):
        raise RefreshError("phase-zero authority must be a read-only regular file")
    artifact = _read_json(authority_path)
    required_top = {
        "schema", "classification", "semantic_labels_or_outcomes_accessed",
        "acceptance_artifact", "all_six_targets_pass", "target_pair_order",
        "feature_source_pair_order", "artifact_serialization", "generation",
        "recreate", "runtime_versions", "runtime_version_contract", "window",
        "implementation_sha256",
        "unmodified_head_live_provider_counterfactual", "source_inputs",
        "source_inputs_sha256", "source_input_identities_pre_post_exact",
        "raw_source_rebuilds", "eurusd_orderflow_behavior", "pairs",
    }
    if set(artifact) != required_top:
        raise RefreshError("phase-zero authority fields differ")
    if (
        artifact.get("schema") != "m15-book-refresh-phase-zero-acceptance/v1"
        or artifact.get("acceptance_artifact") is not True
        or artifact.get("all_six_targets_pass") is not True
        or artifact.get("classification") != "outcome_blind_pre_april_acceptance"
        or artifact.get("semantic_labels_or_outcomes_accessed") is not False
        or artifact.get("target_pair_order") != list(TARGET_PAIR_ORDER)
        or artifact.get("feature_source_pair_order") != list(FEATURE_SOURCE_PAIR_ORDER)
        or artifact.get("source_input_identities_pre_post_exact") is not True
        or artifact.get("artifact_serialization")
        != {
            "encoding": "UTF-8",
            "json_key_order": "lexicographic_recursive",
            "json_separators": [",", ":"],
            "allow_nan": False,
            "terminal_newline_bytes_hex": "0a",
        }
        or artifact.get("window")
        != {
            "clock": "left_labeled_source_feature_timestamp_plus_60_seconds",
            "entry_at_or_after": "2026-01-01T00:00:00Z",
            "entry_before": "2026-04-01T00:00:00Z",
        }
    ):
        raise RefreshError("phase-zero authority is not the required six-target acceptance contract")

    runtime_contract = verified_runtime_version_contract()
    runtime_versions = artifact.get("runtime_versions")
    if (
        artifact.get("runtime_version_contract") != runtime_contract
        or not isinstance(runtime_versions, dict)
        or set(runtime_versions) != {"python", *RUNTIME_DISTRIBUTIONS}
        or runtime_versions.get("python") != sys.version.split()[0]
        or {name: runtime_versions.get(name) for name in RUNTIME_DISTRIBUTIONS}
        != runtime_contract["expected_exact"]
    ):
        raise RefreshError("phase-zero runtime version authority differs")
    python_environment = runtime_contract.get("python_environment")
    if not isinstance(python_environment, dict):
        raise RefreshError("phase-zero Python environment authority is absent")
    executable = python_environment.get("executable")
    if artifact.get("generation") != {
        "working_directory": str(REPO_ROOT),
        "exact_executable_command": (
            f"{executable} scripts/m15_phase_zero_behavior_authority.py "
            "--mode acceptance"
        ),
        "python_executable": executable,
        "generator": "scripts/m15_phase_zero_behavior_authority.py",
        "action": "generate_and_atomically_create_canonical_json",
    }:
        raise RefreshError("phase-zero generation provenance differs")
    if artifact.get("recreate") != {
        "exact_command": (
            "PYTHONNOUSERSITE=1 ~/binary-algo-venv/bin/python "
            "scripts/m15_phase_zero_behavior_authority.py --mode acceptance"
        ),
        "campaign_command_template": (
            "PYTHONNOUSERSITE=1 ~/binary-algo-venv/bin/python "
            "scripts/m15_book_refresh.py --adapter-parity "
            "--prereg-id <sealed_prereg_id>"
        ),
        "campaign_function": "m15_book_refresh.run_adapter_parity",
    }:
        raise RefreshError("phase-zero recreate provenance differs")

    root = REPO_ROOT.resolve(strict=True)
    expected_implementation: dict[str, str] = {}
    for source in source_files():
        if source.is_symlink() or not source.is_file():
            raise RefreshError(f"phase-zero implementation source is missing: {source}")
        try:
            relative = source.resolve(strict=True).relative_to(root).as_posix()
        except (OSError, ValueError) as exc:
            raise RefreshError(f"phase-zero implementation source escapes repo: {source}") from exc
        expected_implementation[relative] = sha256_file(source)
    if artifact.get("implementation_sha256") != expected_implementation:
        raise RefreshError("phase-zero authority is not bound to every final implementation byte")
    if (
        expected_implementation.get("scripts/live_features.py")
        != PHASE_ZERO_FLOAT32_LIVE_FEATURES_SHA256
    ):
        raise RefreshError("phase-zero corrected live provider source hash differs")
    counterfactual = artifact.get("unmodified_head_live_provider_counterfactual")
    if not isinstance(counterfactual, dict) or set(counterfactual) != {
        "production_provider_semantics_changed",
        "verification",
        "baseline_unmodified_head",
        "working_tree_provider",
        "source_semantic_delta",
    }:
        raise RefreshError("phase-zero provider counterfactual fields differ")
    if (
        counterfactual.get("production_provider_semantics_changed") is not True
        or counterfactual.get("verification")
        != (
            "exact baseline/current source hashes plus one exact close-conversion "
            "expression in each source"
        )
        or counterfactual.get("baseline_unmodified_head")
        != {
            "commit": "b74a549f45bddfd6310d80c1f6c130de0076a95f",
            "path": "scripts/live_features.py",
            "git_blob_sha1": "bb55b4f17bd6e4f514435624bec2117a92ae7dad",
            "sha256": "4903217812213cbeb6902a02552f7bddcc741fa665e8e700aee10c2b1716feb4",
            "log_return_close_dtype": "float64",
        }
        or counterfactual.get("working_tree_provider")
        != {
            "path": "scripts/live_features.py",
            "sha256": expected_implementation["scripts/live_features.py"],
            "log_return_close_dtype": "float32",
        }
        or counterfactual.get("source_semantic_delta")
        != {
            "baseline_close_conversion": "Python float / NumPy float64",
            "working_tree_close_conversion": "NumPy float32",
            "behavioral_claims": "none_in_this_source_only_record",
        }
    ):
        raise RefreshError("phase-zero provider counterfactual differs")

    spec = load_spec()
    current_incumbents = _phase_zero_current_incumbents(spec)
    import live_features

    expected_raw_schema = ["open", "high", "low", "close", "volume", "datetime_utc"]

    def expected_processed_paths(pair: str) -> list[Path]:
        q1_start = pd.Timestamp("2026-01-01T00:00:00Z")
        q1_end = pd.Timestamp("2026-04-01T00:00:00Z")
        output: list[Path] = []
        for candidate in sorted(
            (PROCESSED_ROOT / pair).glob(f"{pair}_10s_2026-*.parquet")
        ):
            try:
                file_date = pd.Timestamp(candidate.stem[-10:], tz="UTC")
            except Exception as exc:
                raise RefreshError(
                    f"{pair}: malformed phase-zero processed filename {candidate.name}"
                ) from exc
            if q1_start <= file_date < q1_end:
                output.append(candidate)
        if not output:
            raise RefreshError(f"{pair}: phase-zero Q1 processed inventory is empty")
        return output

    expected_projections: dict[str, list[str]] = {}
    for source_pair in spec["feature_source_pair_order"]:
        if source_pair not in current_incumbents:
            expected_projections[source_pair] = ["close"]
        else:
            base = [
                column
                for column in current_incumbents[source_pair]["feature_cols"]
                if live_features.is_base_feature(column)
            ]
            expected_projections[source_pair] = [*base, "close"]

    source_inputs = artifact.get("source_inputs")
    if not isinstance(source_inputs, dict) or set(source_inputs) != {
        "cached_feature_sources_in_feature_source_order",
        "processed_raw_source_groups_in_feature_source_order",
        "eurusd_cached_orderflow_comparator",
        "eurusd_rebuilt_orderflow_authority",
        "incumbent_books_in_target_pair_order",
        "forbidden_semantic_columns_read",
    }:
        raise RefreshError("phase-zero seven-source input authority differs")
    if (
        source_inputs["forbidden_semantic_columns_read"] != []
        or artifact.get("source_inputs_sha256")
        != sha256_bytes(canonical_bytes(source_inputs))
    ):
        raise RefreshError("phase-zero source input digest/semantic contract differs")

    cached_sources = source_inputs["cached_feature_sources_in_feature_source_order"]
    processed_groups = source_inputs[
        "processed_raw_source_groups_in_feature_source_order"
    ]
    source_order = list(spec["feature_source_pair_order"])
    if (
        not isinstance(cached_sources, list)
        or not isinstance(processed_groups, list)
        or [row.get("pair") for row in cached_sources if isinstance(row, dict)]
        != source_order
        or [row.get("pair") for row in processed_groups if isinstance(row, dict)]
        != source_order
    ):
        raise RefreshError("phase-zero source inventories do not preserve source order")
    cached_by_pair: dict[str, dict[str, Any]] = {}
    processed_by_pair: dict[str, dict[str, Any]] = {}
    for source_pair, cached, processed in zip(
        source_order, cached_sources, processed_groups
    ):
        if (
            not isinstance(cached, dict)
            or set(cached) != {"pair", "file", "projection_in_order"}
            or cached.get("pair") != source_pair
            or cached.get("projection_in_order") != expected_projections[source_pair]
        ):
            raise RefreshError(f"{source_pair}: cached source authority differs")
        cached_path = _authority_file_identity(
            cached["file"], name=f"source_inputs.cached.{source_pair}"
        )
        if cached_path.resolve(strict=True) != (
            REPO_ROOT / "features" / f"{source_pair}_2026.parquet"
        ).resolve(strict=True):
            raise RefreshError(f"{source_pair}: cached source path differs")
        expected_processed = expected_processed_paths(source_pair)
        if (
            not isinstance(processed, dict)
            or set(processed) != {"pair", "projection_in_order", "ordered_files"}
            or processed.get("pair") != source_pair
            or processed.get("projection_in_order") != expected_raw_schema
            or not isinstance(processed.get("ordered_files"), list)
            or len(processed["ordered_files"]) != len(expected_processed)
        ):
            raise RefreshError(f"{source_pair}: processed source authority differs")
        for index, (identity, expected_path) in enumerate(
            zip(processed["ordered_files"], expected_processed)
        ):
            actual_path = _authority_file_identity(
                identity, name=f"source_inputs.processed.{source_pair}[{index}]"
            )
            if actual_path.resolve(strict=True) != expected_path.resolve(strict=True):
                raise RefreshError(f"{source_pair}: processed source order/path differs")
        cached_by_pair[source_pair] = cached
        processed_by_pair[source_pair] = processed

    expected_of_columns = [
        column
        for column in current_incumbents["EURUSD"]["feature_cols"]
        if column.startswith("OF_")
    ]
    cached_of = source_inputs["eurusd_cached_orderflow_comparator"]
    rebuilt_of = source_inputs["eurusd_rebuilt_orderflow_authority"]
    if (
        not expected_of_columns
        or not isinstance(cached_of, dict)
        or set(cached_of) != {"file", "projection_in_order"}
        or cached_of.get("projection_in_order") != expected_of_columns
        or not isinstance(rebuilt_of, dict)
        or set(rebuilt_of)
        != {
            "source_pair", "processed_raw_source_group_sha256", "recipe",
            "recipe_source", "projection_in_order",
        }
        or rebuilt_of.get("source_pair") != "EURUSD"
        or rebuilt_of.get("processed_raw_source_group_sha256")
        != sha256_bytes(canonical_bytes(processed_by_pair["EURUSD"]))
        or rebuilt_of.get("recipe")
        != "orderflow.build_of_features(orderflow.of_1m(raw_10s))"
        or rebuilt_of.get("projection_in_order") != expected_of_columns
    ):
        raise RefreshError("EURUSD raw/cached order-flow authority differs")
    cached_of_path = _authority_file_identity(
        cached_of["file"], name="source_inputs.eurusd_cached_orderflow"
    )
    recipe_path = _authority_file_identity(
        rebuilt_of["recipe_source"], name="source_inputs.eurusd_orderflow_recipe"
    )
    if (
        cached_of_path.resolve(strict=True)
        != (REPO_ROOT / "features_of" / "EURUSD_2026.parquet").resolve(strict=True)
        or recipe_path.resolve(strict=True)
        != (REPO_ROOT / "scripts" / "orderflow.py").resolve(strict=True)
    ):
        raise RefreshError("EURUSD order-flow authority path differs")

    incumbent_rows = source_inputs["incumbent_books_in_target_pair_order"]
    if not isinstance(incumbent_rows, list) or len(incumbent_rows) != len(TARGET_PAIR_ORDER):
        raise RefreshError("phase-zero incumbent source inventory differs")
    for pair, incumbent in zip(TARGET_PAIR_ORDER, incumbent_rows):
        current = current_incumbents[pair]
        if (
            not isinstance(incumbent, dict)
            or set(incumbent) != {"book_id", "content_id", "ordered_files", "model_order"}
            or incumbent.get("book_id") != current["book_id"]
            or incumbent.get("content_id") != current["content_id"]
            or incumbent.get("model_order") != current["model_order"]
            or not isinstance(incumbent.get("ordered_files"), list)
            or len(incumbent["ordered_files"]) != len(current["ordered_files"])
        ):
            raise RefreshError(f"{pair}: incumbent source identity differs")
        for index, (identity, expected_path) in enumerate(
            zip(incumbent["ordered_files"], current["ordered_files"])
        ):
            actual = _authority_file_identity(
                identity, name=f"source_inputs.incumbent.{pair}[{index}]"
            )
            if actual.resolve(strict=True) != Path(expected_path).resolve(strict=True):
                raise RefreshError(f"{pair}: incumbent source file order differs")

    raw_rebuilds = artifact.get("raw_source_rebuilds")
    if (
        not isinstance(raw_rebuilds, dict)
        or len(raw_rebuilds) != len(source_order)
        or set(raw_rebuilds) != set(source_order)
    ):
        raise RefreshError("phase-zero raw rebuild order differs")
    rebuild_evidence: dict[str, dict[str, Any]] = {}
    for source_pair in source_order:
        rebuild_evidence[source_pair] = _validate_phase_zero_source_rebuild(
            source_pair,
            raw_rebuilds[source_pair],
            expected_projection=expected_projections[source_pair],
            expected_processed_identities=processed_by_pair[source_pair][
                "ordered_files"
            ],
            expected_raw_schema=expected_raw_schema,
        )

    orderflow_behavior = artifact.get("eurusd_orderflow_behavior")
    if (
        not isinstance(orderflow_behavior, dict)
        or set(orderflow_behavior)
        != {
            "cached_comparator", "raw_rebuilt_from_processed", "source_clock_exact",
            "finite_mask_exact", "ordered_float32_values_exact",
        }
        or orderflow_behavior.get("source_clock_exact") is not True
        or orderflow_behavior.get("finite_mask_exact") is not True
        or orderflow_behavior.get("ordered_float32_values_exact") is not True
    ):
        raise RefreshError("EURUSD order-flow behavior acceptance differs")
    cached_orderflow_report = orderflow_behavior["cached_comparator"]
    raw_orderflow_report = orderflow_behavior["raw_rebuilt_from_processed"]
    cached_orderflow_fields = {
        "source_kind",
        "schema_in_order",
        "projection_in_order",
        "semantic_or_outcome_columns_read",
        "rows",
        "source_clock_i64le_sha256",
        "ordered_float32_values_sha256",
    }
    raw_orderflow_fields = {
        "source_kind",
        "recipe",
        "recipe_source",
        "projection_in_order",
        "semantic_or_outcome_columns_read",
        "rows",
        "source_clock_i64le_sha256",
        "ordered_float32_values_sha256",
        "same_strict_raw_load_as_eurusd_base_rebuild",
    }
    if (
        not isinstance(cached_orderflow_report, dict)
        or set(cached_orderflow_report) != cached_orderflow_fields
        or not isinstance(raw_orderflow_report, dict)
        or set(raw_orderflow_report) != raw_orderflow_fields
        or cached_orderflow_report.get("source_kind")
        != "cached_orderflow_comparator_projection"
        or raw_orderflow_report.get("source_kind")
        != "rebuilt_from_strict_processed_eurusd_10s"
        or raw_orderflow_report.get("recipe")
        != "orderflow.build_of_features(orderflow.of_1m(raw_10s))"
        or raw_orderflow_report.get("recipe_source") != rebuilt_of["recipe_source"]
        or cached_orderflow_report.get("projection_in_order") != expected_of_columns
        or raw_orderflow_report.get("projection_in_order") != expected_of_columns
        or cached_orderflow_report.get("semantic_or_outcome_columns_read") is not False
        or raw_orderflow_report.get("semantic_or_outcome_columns_read") is not False
        or raw_orderflow_report.get("same_strict_raw_load_as_eurusd_base_rebuild")
        is not True
    ):
        raise RefreshError("EURUSD detailed order-flow authority differs")
    cached_of_rows = _phase_zero_integer(
        cached_orderflow_report.get("rows"),
        name="EURUSD.orderflow.cached.rows",
        minimum=1,
    )
    raw_of_rows = _phase_zero_integer(
        raw_orderflow_report.get("rows"),
        name="EURUSD.orderflow.raw.rows",
        minimum=1,
    )
    cached_of_clock = _phase_zero_hash(
        cached_orderflow_report.get("source_clock_i64le_sha256"),
        name="EURUSD.orderflow.cached.clock",
    )
    raw_of_clock = _phase_zero_hash(
        raw_orderflow_report.get("source_clock_i64le_sha256"),
        name="EURUSD.orderflow.raw.clock",
    )
    cached_of_values = _phase_zero_hash(
        cached_orderflow_report.get("ordered_float32_values_sha256"),
        name="EURUSD.orderflow.cached.values",
    )
    raw_of_values = _phase_zero_hash(
        raw_orderflow_report.get("ordered_float32_values_sha256"),
        name="EURUSD.orderflow.raw.values",
    )
    cached_of_schema = cached_orderflow_report.get("schema_in_order")
    if (
        cached_of_rows != raw_of_rows
        or cached_of_clock != raw_of_clock
        or cached_of_values != raw_of_values
        or not isinstance(cached_of_schema, list)
        or [field.get("name") for field in cached_of_schema if isinstance(field, dict)]
        != [*expected_of_columns, "datetime_utc"]
        or any(
            not isinstance(field, dict)
            or set(field) != {"name", "type", "nullable"}
            or not isinstance(field.get("type"), str)
            or not field["type"]
            or type(field.get("nullable")) is not bool
            for field in cached_of_schema
        )
    ):
        raise RefreshError("EURUSD order-flow equality evidence is inconsistent")

    pairs = artifact.get("pairs")
    if (
        not isinstance(pairs, dict)
        or len(pairs) != len(spec["pair_order"])
        or set(pairs) != set(spec["pair_order"])
    ):
        raise RefreshError("phase-zero authority must contain exactly all six target pairs")
    required_inputs = {
        "cached_feature_file",
        "cached_parquet_projection_in_order",
        "processed_parquet_projection_in_order",
        "processed_parquet_required_schema_in_order",
        "forbidden_semantic_columns_read",
        "processed_raw_files_in_loader_order",
        "cross_pair_inputs_actually_used",
        "cross_pair_inputs_used",
        "incumbent_book",
        "cached_source_projections_in_feature_source_order",
        "processed_raw_source_groups_actually_used",
        "common_input_inventory_sha256",
    }
    for pair in spec["pair_order"]:
        row = pairs[pair]
        if not isinstance(row, dict):
            raise RefreshError(f"{pair}: phase-zero pair record is malformed")
        _validate_phase_zero_pair_behavior(
            pair,
            row,
            feature_cols=current_incumbents[pair]["feature_cols"],
        )
        checks = row.get("verdict_checks")
        if (
            row.get("verdict") != "PASS_BEHAVIOR_AUTHORITY"
            or not isinstance(checks, dict)
            or set(checks) != PHASE_ZERO_VERDICT_CHECKS
            or any(value is not True for value in checks.values())
        ):
            raise RefreshError(f"{pair}: phase-zero behavior authority did not pass")
        inputs = row.get("inputs")
        if not isinstance(inputs, dict) or set(inputs) != required_inputs:
            raise RefreshError(f"{pair}: phase-zero input contract differs")
        cached = inputs["cached_feature_file"]
        if (
            not isinstance(cached, dict)
            or cached.get("path") != f"features/{pair}_2026.parquet"
        ):
            raise RefreshError(f"{pair}: phase-zero cached feature path differs")
        _authority_file_identity(cached, name=f"{pair}.cached_feature_file")
        projection = inputs["cached_parquet_projection_in_order"]
        current_incumbent = current_incumbents[pair]
        if (
            not isinstance(projection, list)
            or projection != current_incumbent["feature_cols"]
            or any(not isinstance(value, str) or value in FORBIDDEN_SEMANTIC_COLUMNS
                   for value in projection)
        ):
            raise RefreshError(
                f"{pair}: phase-zero cached projection is not exact current A covariates"
            )
        if (
            inputs["processed_parquet_projection_in_order"] != expected_raw_schema
            or inputs["processed_parquet_required_schema_in_order"] != expected_raw_schema
            or inputs["forbidden_semantic_columns_read"] != []
        ):
            raise RefreshError(f"{pair}: phase-zero processed projection/schema differs")
        if inputs["processed_raw_files_in_loader_order"] != processed_by_pair[pair]["ordered_files"]:
            raise RefreshError(f"{pair}: phase-zero own processed inventory differs")
        cross_used = inputs["cross_pair_inputs_used"]
        cross_inputs = inputs["cross_pair_inputs_actually_used"]
        family = spec["pairs"][pair]["adapter_family"]
        if family == "own_pair":
            if cross_used is not False or cross_inputs != []:
                raise RefreshError(
                    f"{pair}: own-pair phase-zero record claims cross-pair inputs"
                )
        else:
            expected_cross_identities = [
                cached_by_pair[source_pair]["file"]
                for source_pair in spec["feature_source_pair_order"]
            ]
            if family == "eur_xpair_of":
                expected_cross_identities.append(cached_of["file"])
            if (
                cross_used is not True
                or not isinstance(cross_inputs, list)
                or cross_inputs != expected_cross_identities
            ):
                raise RefreshError(
                    f"{pair}: xpair phase-zero dependencies are absent/incoherent"
                )
        used_sources = (
            list(spec["feature_source_pair_order"])
            if family != "own_pair"
            else [pair]
        )
        if (
            inputs["cached_source_projections_in_feature_source_order"]
            != [
                {"pair": source, "projection_in_order": expected_projections[source]}
                for source in used_sources
            ]
            or inputs["processed_raw_source_groups_actually_used"]
            != [processed_by_pair[source] for source in used_sources]
            or inputs["common_input_inventory_sha256"]
            != artifact["source_inputs_sha256"]
        ):
            raise RefreshError(f"{pair}: source routing authority differs")
        incumbent = inputs["incumbent_book"]
        if (
            not isinstance(incumbent, dict)
            or set(incumbent) != {"book_id", "content_id", "model_order", "ordered_files"}
            or incumbent.get("book_id") != current_incumbent["book_id"]
            or incumbent.get("content_id") != current_incumbent["content_id"]
            or incumbent.get("model_order") != current_incumbent["model_order"]
            or not isinstance(incumbent.get("ordered_files"), list)
            or len(incumbent["ordered_files"])
            != len(current_incumbent["ordered_files"])
        ):
            raise RefreshError(f"{pair}: phase-zero incumbent identity differs")
        for index, (identity, expected_path) in enumerate(
            zip(incumbent["ordered_files"], current_incumbent["ordered_files"])
        ):
            actual_path = _authority_file_identity(
                identity, name=f"{pair}.incumbent[{index}]"
            )
            if actual_path.resolve(strict=True) != expected_path.resolve(strict=True):
                raise RefreshError(
                    f"{pair}: phase-zero incumbent file order/path differs"
                )
        provider = row.get("provider_qualification")
        provider_fields = {
            "status", "reason", "live_activation_eligible", "acceptance_semantics",
            "provider_state_exact", "requested_orderflow_columns_in_order",
            "public_provider_missing_orderflow_columns_in_order",
            "public_feature_row_rejected_missing_schema",
            "public_feature_row_rejection", "public_live_feature_builder_parity",
        }
        if not isinstance(provider, dict) or set(provider) != provider_fields:
            raise RefreshError(f"{pair}: provider qualification fields differ")
        if pair == "EURUSD":
            if (
                provider.get("status") != "BLOCKED_SCHEMA"
                or provider.get("reason") != "verified_live_OF_provider_absent"
                or provider.get("live_activation_eligible") is not False
                or provider.get("acceptance_semantics")
                != "verified_offline_behavior_pass_live_activation_blocked"
                or provider.get("provider_state_exact") is not True
                or provider.get("requested_orderflow_columns_in_order")
                != expected_of_columns
                or provider.get("public_provider_missing_orderflow_columns_in_order")
                != expected_of_columns
                or provider.get("public_feature_row_rejected_missing_schema") is not True
                or not isinstance(provider.get("public_feature_row_rejection"), str)
                or not provider["public_feature_row_rejection"]
                or provider.get("public_live_feature_builder_parity") is not None
            ):
                raise RefreshError("EURUSD provider block evidence differs")
        else:
            live = provider.get("public_live_feature_builder_parity")
            if (
                provider.get("status") != "PASSED"
                or provider.get("reason") is not None
                or provider.get("live_activation_eligible") is not True
                or provider.get("acceptance_semantics")
                != "public_live_feature_builder_parity_required"
                or provider.get("provider_state_exact") is not True
                or provider.get("requested_orderflow_columns_in_order") != []
                or provider.get("public_provider_missing_orderflow_columns_in_order") != []
                or provider.get("public_feature_row_rejected_missing_schema") is not False
                or provider.get("public_feature_row_rejection") is not None
                or not isinstance(live, dict)
                or live.get("pair") != pair
                or live.get("behavior_all_equal") is not True
                or live.get("ordered_finite_float32_values_exact") is not True
            ):
                raise RefreshError(f"{pair}: public provider parity differs")
            live_fields = {
                "schema",
                "pair",
                "public_class",
                "public_method_exercised",
                "provider_rows",
                "compared_rows",
                "feature_count",
                "feature_order_exact",
                "source_clock_subset_exact",
                "causal_decision_mapping_exact",
                "finite_mask_exact",
                "ordered_float32_feature_bytes_exact",
                "decoded_diagnostics",
                "public_latest_row_exact",
                "public_latest_source_feature_ns",
                "all_equal",
                "ordered_finite_float32_values_exact",
                "nonfinite_nan_payload_bits_binding",
                "behavior_all_equal",
            }
            if not isinstance(live, dict) or set(live) != live_fields:
                raise RefreshError(f"{pair}: public provider detailed fields differ")
            provider_rows = _phase_zero_integer(
                live.get("provider_rows"), name=f"{pair}.provider.rows", minimum=1
            )
            compared_rows = _phase_zero_integer(
                live.get("compared_rows"), name=f"{pair}.provider.compared", minimum=1
            )
            latest_source = live.get("public_latest_source_feature_ns")
            lower_source_ns = int(
                (pd.Timestamp("2026-01-01T00:00:00Z") - pd.Timedelta(seconds=60)).value
            )
            upper_source_ns = int(
                (pd.Timestamp("2026-04-01T00:00:00Z") - pd.Timedelta(seconds=60)).value
            )
            if (
                live.get("schema") != "m15-live-feature-builder-parity/v1"
                or live.get("public_class") != "live_features.LiveFeatureBuilder"
                or live.get("public_method_exercised") != "feature_row"
                or provider_rows < compared_rows
                or compared_rows != row["rows"]
                or live.get("feature_count") != len(current_incumbent["feature_cols"])
                or any(
                    live.get(field) is not True
                    for field in (
                        "feature_order_exact",
                        "source_clock_subset_exact",
                        "causal_decision_mapping_exact",
                        "finite_mask_exact",
                        "public_latest_row_exact",
                        "ordered_finite_float32_values_exact",
                        "behavior_all_equal",
                    )
                )
                or type(live.get("ordered_float32_feature_bytes_exact")) is not bool
                or live.get("all_equal")
                is not live.get("ordered_float32_feature_bytes_exact")
                or live.get("nonfinite_nan_payload_bits_binding") is not False
                or isinstance(latest_source, bool)
                or not isinstance(latest_source, int)
                or not lower_source_ns <= latest_source < upper_source_ns
            ):
                raise RefreshError(f"{pair}: public provider behavior evidence differs")
            _validate_phase_zero_decoded_diagnostics(
                live.get("decoded_diagnostics"),
                feature_cols=current_incumbent["feature_cols"],
                name=f"{pair}.provider.decoded",
                require_zero=True,
            )
    return artifact


def _preregistration_phase_inputs(null_path: Path, authority_path: Path) -> list[Path]:
    """Return the source-phase inputs exactly once in deterministic order."""

    paths = [*source_files(), null_path, authority_path]
    resolved = [path.resolve(strict=True) for path in paths]
    if len(resolved) != len(set(resolved)):
        raise RefreshError("preregistration phase inputs contain a duplicate path")
    return paths


def comparator_a_files(
    *,
    books: Mapping[str, Any] | None = None,
    index: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[Path]:
    """Return the exact registered v1 index/manifest/strategy/model inventory."""

    import book_runtime

    spec = load_spec()
    loaded_books = (
        dict(books)
        if books is not None
        else book_runtime.load_target_books(list(spec["pair_order"]))
    )
    loaded_index = dict(index) if index is not None else book_runtime.load_index()
    if list(loaded_books) != spec["pair_order"]:
        raise RefreshError("comparator A loaded-book order/set differs")
    paths: list[Path] = [REPO_ROOT / "books" / "INDEX.json"]
    for pair in spec["pair_order"]:
        book = loaded_books[pair]
        if (
            getattr(book, "pair", None) != pair
            or getattr(book, "book_id", None) != spec["pairs"][pair]["book_id_A"]
            or book.book_id not in loaded_index
        ):
            raise RefreshError(f"{pair}: comparator A loaded-book identity differs")
        manifest_path, _ = book_runtime.load_registered_manifest(
            book.book_id, dict(loaded_index[book.book_id])
        )
        paths.extend([manifest_path, book.strategy_path, *book.model_paths])
    if len({path.resolve() for path in paths}) != len(paths):
        raise RefreshError("comparator A inventory contains duplicate files")
    return paths


def _comparator_a_resolved_bundle(
    spec: Mapping[str, Any],
    prereg: Mapping[str, Any],
    pair: str,
    book: Any,
    index_entry: Mapping[str, Any],
    manifest_path: Path,
) -> dict[str, Any]:
    structural_gate = _comparator_a_structural_gate(book)
    return {
        "schema": "m15-book-refresh-resolved-bundle-spec/v1",
        "book_id": book.book_id,
        "pair": pair,
        "arm": "A",
        "adapter_family": spec["pairs"][pair]["adapter_family"],
        "builder": spec["pairs"][pair]["builder"],
        "implementation_git_sha": prereg["implementation_git_sha"],
        "prereg_id": prereg["prereg_id"],
        "legacy_content_id": book.content_id,
        "index_entry_sha256": sha256_bytes(canonical_bytes(dict(index_entry))),
        "legacy_manifest_sha256": sha256_file(manifest_path),
        "feature_cols": list(book.feature_cols),
        "feature_dtype": "float32",
        "coverage": book.coverage,
        "confidence_threshold": book.conf_thr,
        "structural_gate": structural_gate,
        "seed_order": list(range(len(book.model_paths))),
        "probability_tie_rule": "p>=0.5_is_UP",
        "historical_training_cap": "not_reconstructed",
    }


def _comparator_a_structural_gate(book: Any) -> dict[str, Any]:
    """Mirror LoadedBook's legacy structural-threshold resolution exactly."""

    strategy = getattr(book, "strategy", None)
    if not isinstance(strategy, Mapping):
        raise RefreshError("comparator A strategy is malformed")
    threshold: float | None = None
    for key in ("bb_width_thr", "bbw_thr"):
        value = strategy.get(key)
        if isinstance(value, (int, float)):
            threshold = float(value)
            break
    gate = strategy.get("gate")
    if threshold is None and isinstance(gate, Mapping):
        for key in ("bb_width_thr", "bbw_thr"):
            value = gate.get(key)
            if isinstance(value, (int, float)):
                threshold = float(value)
                break
    return {
        "structural_column": "15m_bb_width" if threshold is not None else None,
        "structural_threshold": threshold,
    }


def _comparator_a_bundle_id(
    spec: Mapping[str, Any],
    prereg: Mapping[str, Any],
    pair: str,
    book: Any,
    index_entry: Mapping[str, Any],
    manifest_path: Path,
) -> str:
    return derive_bundle_id(
        spec,
        _comparator_a_resolved_bundle(
            spec, prereg, pair, book, index_entry, manifest_path
        ),
        book.strategy_path.name,
        book.strategy_path.read_bytes(),
        [(path.name, path.read_bytes()) for path in book.model_paths],
    )


def _verify_loaded_comparator_binding(
    prereg: Mapping[str, Any],
    binding: Mapping[str, Any],
    books: Mapping[str, Any],
    index: Mapping[str, Mapping[str, Any]],
    *,
    allow_index_additions: bool = False,
) -> None:
    """Authenticate the exact already-loaded A objects against prereg and binding."""

    import book_runtime

    spec = load_spec()
    _verify_preregistered_comparator_files(
        prereg,
        comparator_a_files(books=books, index=index),
        allow_index_additions=allow_index_additions,
    )
    expected_rows = binding.get("primary_arms")
    if not isinstance(expected_rows, list):
        raise RefreshError("run binding primary-arm inventory is malformed")
    a_rows = [row for row in expected_rows if isinstance(row, list) and len(row) == 3
              and row[1] == "A"]
    if [row[:2] for row in a_rows] != [[pair, "A"] for pair in spec["pair_order"]]:
        raise RefreshError("run binding comparator A order/set differs")
    for pair, _arm, expected_bundle in a_rows:
        book = books[pair]
        index_entry = index.get(book.book_id)
        if not isinstance(index_entry, Mapping):
            raise RefreshError(f"{pair}: comparator A index entry is absent")
        manifest_path, _ = book_runtime.load_registered_manifest(
            book.book_id, dict(index_entry)
        )
        actual_bundle = _comparator_a_bundle_id(
            spec, prereg, pair, book, index_entry, manifest_path
        )
        if actual_bundle != expected_bundle:
            raise RefreshError(f"{pair}: loaded comparator A bundle differs from run binding")


def _verify_preregistered_comparator_files(
    prereg: Mapping[str, Any],
    paths: Sequence[Path],
    *,
    allow_index_additions: bool,
) -> None:
    """Authenticate A files while permitting only the INDEX container to grow.

    Candidate publication legitimately changes ``books/INDEX.json``.  Its six
    comparator-A entries remain authenticated separately by their run-bound
    bundle IDs, while every preregistered v1 manifest, strategy, and model must
    retain its exact path, order, size, and bytes.
    """

    sealed = prereg.get("comparator_A_files")
    if not isinstance(sealed, list) or any(not isinstance(row, dict) for row in sealed):
        raise RefreshError("preregistered comparator_A_files inventory is malformed")
    current = [
        asdict(file_identity(path, root=REPO_ROOT)) for path in paths
    ]
    if not allow_index_additions:
        if sealed != current:
            raise RefreshError(
                "preregistered comparator_A_files exact path/count/order/bytes changed"
            )
        return
    expected_index_path = repo_relative(
        REPO_ROOT / "books" / "INDEX.json", root=REPO_ROOT
    )
    if (
        len(sealed) != len(current)
        or not sealed
        or set(sealed[0]) != {"path", "bytes", "sha256"}
        or sealed[0].get("path") != expected_index_path
        or current[0].get("path") != expected_index_path
        or not isinstance(sealed[0].get("bytes"), int)
        or isinstance(sealed[0].get("bytes"), bool)
        or sealed[0]["bytes"] < 0
        or not _is_hex(sealed[0].get("sha256"), 64)
    ):
        raise RefreshError("preregistered comparator A INDEX identity is malformed")
    if sealed[1:] != current[1:]:
        raise RefreshError(
            "preregistered comparator A manifest/strategy/model bytes changed"
        )


def _authenticate_current_comparator_a(
    prereg: Mapping[str, Any],
    binding: Mapping[str, Any],
    *,
    allow_index_additions: bool,
) -> None:
    """Load and twice authenticate current A files and exact six INDEX entries."""

    import book_runtime

    spec = load_spec()
    index = book_runtime.load_index()
    books = book_runtime.load_target_books(list(spec["pair_order"]))
    _verify_loaded_comparator_binding(
        prereg,
        binding,
        books,
        index,
        allow_index_additions=allow_index_additions,
    )
    final_index = book_runtime.load_index()
    if final_index != index:
        raise RefreshError("comparator A INDEX changed during authentication")
    _verify_loaded_comparator_binding(
        prereg,
        binding,
        books,
        final_index,
        allow_index_additions=allow_index_additions,
    )


def historical_feature_files() -> list[Path]:
    """Return every root parquet that any sealed fit/calibration/parity may consume."""

    spec = load_spec()
    paths = [
        REPO_ROOT / "features" / f"{pair}_{year}.parquet"
        for year in range(2012, 2027)
        for pair in spec["feature_source_pair_order"]
    ]
    paths.extend(
        REPO_ROOT / "features_of" / f"EURUSD_{year}.parquet"
        for year in range(2012, 2027)
    )
    missing = [str(path) for path in paths if path.is_symlink() or not path.is_file()]
    if missing:
        raise RefreshError(f"historical feature inventory is missing files: {missing[:8]}")
    return paths


def _fit_historical_files(pair: str, years: Sequence[str]) -> list[Path]:
    spec = load_spec()
    family = spec["pairs"][pair]["adapter_family"]
    source_pairs = (
        spec["feature_source_pair_order"]
        if family in {"eur_xpair_of", "gbpchf_xpair"}
        else [pair]
    )
    paths = [
        REPO_ROOT / "features" / f"{source_pair}_{year}.parquet"
        for year in years
        for source_pair in source_pairs
    ]
    if family == "eur_xpair_of":
        paths.extend(REPO_ROOT / "features_of" / f"EURUSD_{year}.parquet" for year in years)
    return paths


def _verify_preregistered_inventory(
    prereg: Mapping[str, Any],
    field: str,
    paths: Sequence[Path],
    *,
    exact: bool = False,
) -> None:
    sealed = prereg.get(field)
    if not isinstance(sealed, list) or any(not isinstance(row, dict) for row in sealed):
        raise RefreshError(f"preregistered {field} inventory is malformed")
    sealed_by_path = {row.get("path"): row for row in sealed}
    if len(sealed_by_path) != len(sealed) or None in sealed_by_path:
        raise RefreshError(f"preregistered {field} inventory has duplicate/malformed paths")
    current = [asdict(file_identity(path)) for path in paths]
    if exact:
        if sealed != current:
            raise RefreshError(
                f"preregistered {field} exact path/count/order/bytes changed"
            )
        return
    try:
        expected = [sealed_by_path[row["path"]] for row in current]
    except KeyError as exc:
        raise RefreshError(f"{field} path was not preregistered: {exc.args[0]}") from exc
    if expected != current:
        raise RefreshError(f"preregistered {field} bytes/path/order changed")


PREREG_FALSIFIER = """The campaign is invalid on any sealed integrity, identity, access-order, parity, clock, settlement, repeatability, or negative-control defect. Otherwise a whole C book is PROMOTE_TO_SHADOW only when both paired yield lower bounds and both selected-side accuracy-minus-0.5 lower bounds are strictly positive, C has at least 50 settled scheduled trades per side, activity is at least 80% of A and B, and both months pass the sealed whole-book guardrails. Adequately sampled C is RETAIN_V1 when the C-A yield upper bound is negative or either C-side accuracy-minus-0.5 upper bound is non-positive. All other valid outcomes, including ordinary sparsity or genuine zero standard error, are INCONCLUSIVE. No retrospective result activates a book."""


def _null_calibration_path(implementation_git_sha: str) -> Path:
    if not _is_hex(implementation_git_sha, 40):
        raise RefreshError("null calibration requires a full implementation git SHA")
    return WORK_ROOT / "null_calibration" / f"null_calibration_{implementation_git_sha}.json"


def _expected_null_cell_keys(spec: Mapping[str, Any]) -> list[tuple[str, str, int]]:
    contract = spec["null_calibration"]
    fixtures = contract["binding_current"]["fixture_order"]
    if fixtures != [row["name"] for row in contract["fixtures"]]:
        raise RefreshError("binding null fixture order differs from fixture definitions")
    return [
        (fixture, helper, 0)
        for fixture in fixtures
        for helper in ("main", "control")
    ]


def _validate_null_calibration_artifact(
    path: Path,
    *,
    implementation_git_sha: str,
    require_accepted: bool,
) -> dict[str, Any]:
    from scipy.stats import beta

    spec = load_spec()
    if path.is_symlink() or not path.is_file() or path.stat().st_mode & 0o222:
        raise RefreshError("null calibration artifact must be a read-only regular file")
    artifact = _read_json(path)
    required_artifact_fields = {
        "schema", "implementation_git_sha", "spec_sha256", "source_files",
        "contract", "cells", "cells_sha256", "accepted",
        "semantic_replay_outcomes_accessed",
    }
    if set(artifact) != required_artifact_fields:
        raise RefreshError("null calibration artifact fields differ")
    if (
        artifact.get("schema") != "m15-book-refresh-null-calibration-result/v1"
        or artifact.get("implementation_git_sha") != implementation_git_sha
        or artifact.get("spec_sha256") != sha256_file(SPEC_PATH)
        or artifact.get("contract") != spec["null_calibration"]
        or artifact.get("semantic_replay_outcomes_accessed") is not False
    ):
        raise RefreshError("null calibration artifact identity/contract differs")
    current_sources = [asdict(file_identity(source)) for source in source_files()]
    if artifact.get("source_files") != current_sources:
        raise RefreshError("null calibration source-file identities differ")
    cells = artifact.get("cells")
    expected_cell_count = len(_expected_null_cell_keys(spec))
    if not isinstance(cells, list) or len(cells) != expected_cell_count:
        raise RefreshError(
            f"null calibration artifact must contain all {expected_cell_count} binding cells"
        )
    expected_cells_sha = length_prefixed_digest(
        "m15-book-refresh/null-calibration-cells/v1", [canonical_bytes(cells)]
    )
    if artifact.get("cells_sha256") != expected_cells_sha:
        raise RefreshError("null calibration cell payload digest differs")
    actual_keys = [
        (cell.get("fixture"), cell.get("helper"), cell.get("k_shadow"))
        for cell in cells if isinstance(cell, dict)
    ]
    if actual_keys != _expected_null_cell_keys(spec):
        raise RefreshError("null calibration cell order differs")
    contract = spec["null_calibration"]
    cell_acceptance: list[bool] = []
    for cell in cells:
        required_cell_fields = {
            "schema", "fixture", "helper", "k_shadow", "campaign_count",
            "bootstrap_replicates", "block_lengths", "false_positive_count",
            "false_positive_limit", "clopper_pearson_upper_95", "accepted",
            "deterministic_repeat", "campaign_digest", "ordered_shadow_pairs",
            "accuracy_null", "binding", "calibration_role", "cell_passed",
        }
        count = cell.get("false_positive_count")
        count_valid = (
            isinstance(count, int)
            and not isinstance(count, bool)
            and 0 <= count <= contract["campaigns_per_cell"]
        )
        expected_cp = None
        if count_valid:
            expected_cp = (
                1.0
                if count == contract["campaigns_per_cell"]
                else float(
                    beta.ppf(
                        0.95,
                        count + 1,
                        contract["campaigns_per_cell"] - count,
                    )
                )
            )
        expected_acceptance = bool(
            count_valid
            and cell.get("deterministic_repeat") is True
            and count <= contract["false_positive_limit_inclusive"]
        )
        exact = (
            set(cell) == required_cell_fields
            and cell.get("schema") == "m15-book-refresh-null-calibration-cell/v1"
            and cell.get("ordered_shadow_pairs") == []
            and cell.get("accuracy_null") == 0.5
            and cell.get("binding") is True
            and cell.get("calibration_role") == "binding_current"
            and cell.get("campaign_count") == contract["campaigns_per_cell"]
            and cell.get("bootstrap_replicates") == contract["bootstrap_replicates"]
            and cell.get("block_lengths") == contract["block_lengths"]
            and cell.get("false_positive_limit")
            == contract["false_positive_limit_inclusive"]
            and count_valid
            and isinstance(cell.get("clopper_pearson_upper_95"), (int, float))
            and np.isfinite(float(cell["clopper_pearson_upper_95"]))
            and abs(float(cell["clopper_pearson_upper_95"]) - float(expected_cp)) <= 1e-15
            and type(cell.get("deterministic_repeat")) is bool
            and cell.get("cell_passed") is expected_acceptance
            and cell.get("accepted") is expected_acceptance
            and _is_hex(cell.get("campaign_digest"), 64)
        )
        if not exact:
            raise RefreshError(
                "null calibration cell fields/derived values differ: "
                f"{cell.get('fixture')}/{cell.get('helper')}/{cell.get('k_shadow')}"
            )
        cell_acceptance.append(expected_acceptance)
    accepted = bool(all(cell_acceptance))
    if artifact.get("accepted") is not accepted:
        raise RefreshError("null calibration aggregate acceptance is inconsistent")
    if require_accepted and not accepted:
        failed = [
            f"{row.get('fixture')}/{row.get('helper')}/{row.get('k_shadow')}="
            f"{row.get('false_positive_count')}"
            for row in cells if not row.get("accepted")
        ]
        raise RefreshError(
            "exact null calibration did not accept; preregistration/replay blocked: "
            + ", ".join(failed)
        )
    return artifact


def run_null_calibration_gate() -> Path:
    """Run and seal the exact synthetic prerequisite from a clean commit."""

    from m15_book_refresh_stats import run_null_calibration_suite

    verified_runtime_version_contract()
    if git_porcelain():
        raise RefreshError("null calibration requires a clean implementation commit")
    sha = git_head()
    path = _null_calibration_path(sha)
    if path.exists() or path.is_symlink():
        _validate_null_calibration_artifact(
            path, implementation_git_sha=sha, require_accepted=True
        )
        return path
    spec = load_spec()
    contract = spec["null_calibration"]
    cells = run_null_calibration_suite(
        campaign_count=int(contract["campaigns_per_cell"]),
        bootstrap_replicates=int(contract["bootstrap_replicates"]),
        block_lengths=tuple(int(value) for value in contract["block_lengths"]),
        verify_repeat=True,
    )
    cell_values = [cell.as_dict() for cell in cells]
    payload = {
        "schema": "m15-book-refresh-null-calibration-result/v1",
        "implementation_git_sha": sha,
        "spec_sha256": sha256_file(SPEC_PATH),
        "source_files": [asdict(file_identity(source)) for source in source_files()],
        "contract": contract,
        "cells": cell_values,
        "cells_sha256": length_prefixed_digest(
            "m15-book-refresh/null-calibration-cells/v1",
            [canonical_bytes(cell_values)],
        ),
        "accepted": bool(all(cell.accepted for cell in cells)),
        "semantic_replay_outcomes_accessed": False,
    }
    atomic_json_new(path, payload)
    _validate_null_calibration_artifact(
        path, implementation_git_sha=sha, require_accepted=True
    )
    print(f"[null-calibration] PASS cells={len(cells)} artifact={path}", flush=True)
    return path


def preregister() -> str:
    """Create the measured source/prereg seal from a clean implementation commit."""

    runtime_contract = verified_runtime_version_contract()
    spec = load_spec()
    dirty_lines = [line for line in git_porcelain().splitlines() if line]
    provisional_partial = bool(
        len(dirty_lines) == 1
        and dirty_lines[0].startswith(
            "?? results/json/m15_book_refresh_"
        )
        and dirty_lines[0].endswith("_preregister_result.json")
    )
    if dirty_lines and not provisional_partial:
        raise RefreshError("source seal requires `git status --porcelain` to be empty")
    sha = git_head()
    null_path = _null_calibration_path(sha)
    null_acceptance = _validate_null_calibration_artifact(
        null_path, implementation_git_sha=sha, require_accepted=True
    )
    phase_zero = _validate_phase_zero_authority(PHASE_ZERO_AUTHORITY_PATH)
    body = issue_body()
    identities = [asdict(file_identity(path)) for path in source_files()]
    comparator_identities = [asdict(file_identity(path)) for path in comparator_a_files()]
    historical_identities = [asdict(file_identity(path)) for path in historical_feature_files()]
    processed_identities = {
        pair: _source_file_identities(pair)
        for pair in spec["feature_source_pair_order"]
    }
    payload = {
        "schema": "m15-book-refresh-preregistration/v1",
        "issue": spec["issue"],
        "issue_body_sha256": sha256_bytes(body.encode("utf-8")),
        "implementation_git_sha": sha,
        "source_files": identities,
        "comparator_A_files": comparator_identities,
        "historical_feature_files": historical_identities,
        "processed_source_files": processed_identities,
        "evaluator_id": evaluator_id(spec),
        "splits": spec["splits"],
        "cap_selector": spec["cap_selector"],
        "seeds": {
            key: spec["sealed_constants"][key]
            for key in (
                "bootstrap_seed", "control_bootstrap_seed", "shadow_bootstrap_seed",
                "stats_null_test_seed", "permutation_seeds",
            )
        },
        "block_lengths": spec["sealed_constants"]["block_lengths"],
        "canonical_encoding": spec["identity"],
        "runtime_version_contract": runtime_contract,
        "null_calibration": spec["null_calibration"],
        "null_calibration_acceptance": {
            "artifact": repo_relative(null_path),
            "sha256": sha256_file(null_path),
            "accepted": null_acceptance["accepted"],
            "cells": len(null_acceptance["cells"]),
        },
        "phase_zero_behavior_acceptance": {
            "artifact": repo_relative(PHASE_ZERO_AUTHORITY_PATH),
            "sha256": sha256_file(PHASE_ZERO_AUTHORITY_PATH),
            "accepted": phase_zero["acceptance_artifact"],
            "target_pairs": len(phase_zero["pairs"]),
            "feature_sources": len(phase_zero["feature_source_pair_order"]),
        },
        "falsifier": PREREG_FALSIFIER,
    }
    pid = derive_prereg_id(spec, payload, sha, spec_bytes=SPEC_PATH.read_bytes())
    result = dict(payload, prereg_id=pid)
    tracked = RESULTS_JSON / f"m15_book_refresh_{pid}_preregister_result.json"
    expected_partial_line = f"?? {tracked.relative_to(REPO_ROOT).as_posix()}"
    if dirty_lines and dirty_lines != [expected_partial_line]:
        raise RefreshError("source seal found a foreign partial preregistration result")
    workspace = WORK_ROOT / pid
    workspace.mkdir(parents=True, exist_ok=True)
    source_copy = workspace / "source_preregistration.json"
    _seal_json_exact(source_copy, result)
    _seal_json_exact(tracked, result)
    PhaseStore(workspace).seal(
        "source_preregistration",
        canonical_attempt_id=f"{pid}:source:0",
        input_paths=_preregistration_phase_inputs(
            null_path, PHASE_ZERO_AUTHORITY_PATH
        ),
        output_paths=[source_copy, tracked],
        metadata={"prereg_id": pid, "evaluator_id": result["evaluator_id"]},
    )
    print(f"[preregister] prereg_id={pid} implementation={sha}", flush=True)
    return pid


def workspace_for(prereg_id: str) -> tuple[Path, PhaseStore, dict[str, Any]]:
    runtime_contract = verified_runtime_version_contract()
    if not _is_hex(prereg_id, 64):
        raise RefreshError("prereg_id must be 64 lowercase hex characters")
    workspace = WORK_ROOT / prereg_id
    store = PhaseStore(workspace)
    source_phase = store.load("source_preregistration")
    source_path = workspace / "source_preregistration.json"
    try:
        prereg = json.loads(source_path.read_bytes())
    except (OSError, json.JSONDecodeError) as exc:
        raise RefreshError(f"cannot read source preregistration: {exc}") from exc
    if prereg.get("prereg_id") != prereg_id:
        raise RefreshError("workspace preregistration identity mismatch")
    if source_phase.get("metadata", {}).get("prereg_id") != prereg_id:
        raise RefreshError("source phase preregistration identity mismatch")
    if prereg.get("runtime_version_contract") != runtime_contract:
        raise RefreshError("workspace runtime version contract differs from preregistration")
    acceptance = prereg.get("null_calibration_acceptance", {})
    implementation_git_sha = str(prereg.get("implementation_git_sha"))
    expected_null_path = _null_calibration_path(implementation_git_sha)
    expected_null_relative = repo_relative(expected_null_path)
    null_path = REPO_ROOT / str(acceptance.get("artifact", ""))
    if (
        acceptance.get("artifact") != expected_null_relative
        or null_path != expected_null_path
        or acceptance.get("accepted") is not True
        or acceptance.get("cells")
        != len(_expected_null_cell_keys(load_spec()))
        or not _is_hex(acceptance.get("sha256"), 64)
        or null_path.is_symlink()
        or not null_path.is_file()
        or sha256_file(null_path) != acceptance["sha256"]
    ):
        raise RefreshError("workspace null calibration acceptance differs")
    _validate_null_calibration_artifact(
        null_path,
        implementation_git_sha=implementation_git_sha,
        require_accepted=True,
    )
    phase_zero_acceptance = prereg.get("phase_zero_behavior_acceptance", {})
    expected_authority_relative = repo_relative(PHASE_ZERO_AUTHORITY_PATH)
    if (
        phase_zero_acceptance.get("artifact") != expected_authority_relative
        or phase_zero_acceptance.get("accepted") is not True
        or phase_zero_acceptance.get("target_pairs") != 6
        or phase_zero_acceptance.get("feature_sources") != 7
        or not _is_hex(phase_zero_acceptance.get("sha256"), 64)
        or sha256_file(PHASE_ZERO_AUTHORITY_PATH)
        != phase_zero_acceptance.get("sha256")
    ):
        raise RefreshError("workspace phase-zero behavior acceptance differs")
    _validate_phase_zero_authority(PHASE_ZERO_AUTHORITY_PATH)
    return workspace, store, prereg


def _parquet_write_new(frame: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise RefreshError(f"snapshot destination already exists: {destination}")
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.tmp-", suffix=".parquet", dir=destination.parent
    )
    os.close(fd)
    temporary = Path(temporary_name)
    temporary.unlink()
    try:
        frame.to_parquet(temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError as exc:
            raise RefreshError(f"snapshot destination appeared concurrently: {destination}") from exc
        temporary.unlink()
        _fsync_dir(destination.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def _source_file_identities(pair: str, nominal_year: str = "2026") -> list[dict[str, Any]]:
    root = PROCESSED_ROOT / pair
    paths = sorted(root.glob(f"{pair}_10s_{nominal_year}-*.parquet"))
    if not paths:
        raise RefreshError(f"no processed source files for {pair} {nominal_year}")
    rows: list[dict[str, Any]] = []
    for path in paths:
        if path.is_symlink() or not path.is_file():
            raise RefreshError(
                f"processed source must be a regular non-symlink file: {path}"
            )
        before = path.stat()
        digest = sha256_file(path)
        after = path.stat()
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise RefreshError(
                f"processed source changed during identity read: {path}"
            )
        rows.append({
            "path": f"processed/{pair}/{path.name}",
            "bytes": before.st_size,
            "sha256": digest,
        })
    return rows


def _verify_all_processed_source_identities(
    prereg: Mapping[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    current = {
        pair: _source_file_identities(pair)
        for pair in load_spec()["feature_source_pair_order"]
    }
    if prereg.get("processed_source_files") != current:
        raise RefreshError("preregistered processed-source bytes/path/order changed")
    return current


def _strict_processed_clock_audit(pair: str) -> dict[str, Any]:
    """Seal the raw ordered observation clock without deriving settlement."""

    from m15_book_refresh_replay import (
        discover_observation_files,
        load_observation_series,
    )

    paths = discover_observation_files(pair, nominal_year="2026")
    series = load_observation_series(paths)
    timestamps = np.asarray(series.timestamps_s, dtype="int64")
    gaps = np.diff(timestamps)
    return {
        "schema": "m15-book-refresh-processed-clock-audit/v1",
        "pair": pair,
        "source_label_convention": "left_edge_of_[label,label+10s)_bar",
        "close_observation_shift_s": 10,
        "clock_role": "causal_processed_close_observation_timestamp",
        "source_files": list(series.source_files),
        "rows": len(timestamps),
        "first_timestamp_s": int(timestamps[0]),
        "last_timestamp_s": int(timestamps[-1]),
        "timestamp_i64le_sha256": sha256_bytes(
            np.asarray(timestamps, dtype="<i8").tobytes(order="C")
        ),
        "gap_i64le_sha256": sha256_bytes(
            np.asarray(gaps, dtype="<i8").tobytes(order="C")
        ),
        "minimum_gap_s": int(gaps.min()) if len(gaps) else None,
        "maximum_gap_s": int(gaps.max()) if len(gaps) else None,
        "ten_second_gaps": int((gaps == 10).sum()),
        "non_ten_second_gaps": int((gaps != 10).sum()),
        "strictly_increasing": True,
        "whole_second_clock": True,
        "finite_positive_close_row_alignment": True,
        "semantic_settlement_derived": False,
    }


def _blind_base_snapshot(pair: str, destination: Path) -> dict[str, Any]:
    """Build causal covariates and close only; never call a label builder."""

    import pipeline

    raw = pipeline.load_pair_year_10s(pair, "2026")
    if raw is None or raw.empty:
        raise RefreshError(f"{pair}: empty processed 2026 source")
    m1 = pipeline.resample_1m(raw)
    frame = pipeline.build_features(m1)
    frame["close"] = m1["close"].to_numpy()
    frame = frame.replace([np.inf, -np.inf], np.nan)
    for column in frame.columns:
        if frame[column].dtype == np.dtype("float64"):
            frame[column] = frame[column].astype("float32")
    if not isinstance(frame.index, pd.DatetimeIndex) or str(frame.index.tz) != "UTC":
        raise RefreshError(f"{pair}: blind snapshot index is not UTC")
    ns = frame.index.as_unit("ns").asi8
    assert_strict_clock(ns, name=f"{pair} blind decision clock")
    decision_ns = ns + np.int64(60 * 1_000_000_000)
    expected = [c for c in pd.read_parquet(REPO_ROOT / "features" / f"{pair}_2025.parquet").columns
                if c not in {"y", "fwd_ret", "valid"}]
    if list(frame.columns) != expected:
        raise RefreshError(f"{pair}: blind snapshot schema/order differs from incumbent base schema")
    _parquet_write_new(frame, destination)
    return {
        "rows": len(frame),
        "feature_source_label_shift_to_decision_s": 60,
        "first_feature_source_label_ns": int(ns[0]),
        "last_feature_source_label_ns": int(ns[-1]),
        "first_entry_ns": int(decision_ns[0]),
        "last_entry_ns": int(decision_ns[-1]),
        "decision_clock_i64le_sha256": sha256_bytes(
            np.asarray(decision_ns, dtype="<i8").tobytes(order="C")
        ),
        "columns": list(frame.columns),
        "dtypes": [str(dtype) for dtype in frame.dtypes],
        "sha256": sha256_file(destination),
    }


def _blind_orderflow_snapshot(destination: Path) -> dict[str, Any]:
    import orderflow
    import pipeline

    raw = pipeline.load_pair_year_10s("EURUSD", "2026")
    if raw is None or raw.empty:
        raise RefreshError("EURUSD: empty source for order-flow snapshot")
    frame = orderflow.build_of_features(orderflow.of_1m(raw))
    frame = frame.replace([np.inf, -np.inf], np.nan).astype("float32")
    if not isinstance(frame.index, pd.DatetimeIndex) or str(frame.index.tz) != "UTC":
        raise RefreshError("EURUSD OF: blind snapshot index is not UTC")
    source_ns = frame.index.as_unit("ns").asi8
    assert_strict_clock(source_ns, name="EURUSD OF blind decision clock")
    decision_ns = source_ns + np.int64(60 * 1_000_000_000)
    expected = list(pd.read_parquet(REPO_ROOT / "features_of" / "EURUSD_2025.parquet").columns)
    if list(frame.columns) != expected:
        raise RefreshError("EURUSD OF: blind snapshot schema/order differs from incumbent")
    _parquet_write_new(frame, destination)
    return {
        "rows": len(frame),
        "feature_source_label_shift_to_decision_s": 60,
        "first_feature_source_label_ns": int(source_ns[0]),
        "last_feature_source_label_ns": int(source_ns[-1]),
        "first_entry_ns": int(decision_ns[0]),
        "last_entry_ns": int(decision_ns[-1]),
        "decision_clock_i64le_sha256": sha256_bytes(
            np.asarray(decision_ns, dtype="<i8").tobytes(order="C")
        ),
        "columns": list(frame.columns),
        "dtypes": [str(dtype) for dtype in frame.dtypes],
        "sha256": sha256_file(destination),
    }


def _make_feature_views(workspace: Path, snapshot_root: Path) -> dict[str, Path]:
    """Create path-only views: historical cache plus the isolated 2026 snapshot."""

    views = workspace / "feature_views"
    feature_view = views / "features"
    orderflow_view = views / "features_of"
    feature_view.mkdir(parents=True, exist_ok=False)
    orderflow_view.mkdir(parents=True, exist_ok=False)
    pair_order = load_spec()["feature_source_pair_order"]
    for year in range(2012, 2026):
        for pair in pair_order:
            source = REPO_ROOT / "features" / f"{pair}_{year}.parquet"
            if source.is_symlink() or not source.is_file():
                raise RefreshError(f"missing canonical historical feature route: {source}")
            os.symlink(source.resolve(), feature_view / source.name)
    for pair in pair_order:
        source = snapshot_root / "features" / f"{pair}_2026.parquet"
        if source.is_symlink() or not source.is_file():
            raise RefreshError(f"missing canonical snapshot feature route: {source}")
        os.symlink(source.resolve(), feature_view / source.name)
    for year in range(2012, 2026):
        source = REPO_ROOT / "features_of" / f"EURUSD_{year}.parquet"
        if source.is_symlink() or not source.is_file():
            raise RefreshError(f"missing canonical historical order-flow route: {source}")
        os.symlink(source.resolve(), orderflow_view / source.name)
    source = snapshot_root / "features_of" / "EURUSD_2026.parquet"
    if source.is_symlink() or not source.is_file():
        raise RefreshError(f"missing canonical snapshot order-flow route: {source}")
    os.symlink(source.resolve(), orderflow_view / source.name)

    # A second view makes it impossible for pre-arm label builders to see an
    # April/May close.  Historical files are links; 2026 is a physical,
    # pre-replay truncation of the blind covariate snapshot.
    fit_feature_view = views / "fit_features"
    fit_orderflow_view = views / "fit_features_of"
    fit_feature_view.mkdir()
    fit_orderflow_view.mkdir()
    for source in sorted(feature_view.glob("*.parquet")):
        if source.name.endswith("_2026.parquet"):
            frame = pd.read_parquet(source)
            frame = frame.loc[frame.index < pd.Timestamp("2026-04-01", tz="UTC")]
            _parquet_write_new(frame, fit_feature_view / source.name)
        else:
            os.symlink(source.resolve(), fit_feature_view / source.name)
    for source in sorted(orderflow_view.glob("*.parquet")):
        if source.name == "EURUSD_2026.parquet":
            frame = pd.read_parquet(source)
            frame = frame.loc[frame.index < pd.Timestamp("2026-04-01", tz="UTC")]
            _parquet_write_new(frame, fit_orderflow_view / source.name)
        else:
            os.symlink(source.resolve(), fit_orderflow_view / source.name)
    _fsync_dir(feature_view)
    _fsync_dir(orderflow_view)
    _fsync_dir(fit_feature_view)
    _fsync_dir(fit_orderflow_view)
    _fsync_dir(views)
    return {
        "feature_view": feature_view,
        "orderflow_view": orderflow_view,
        "fit_feature_view": fit_feature_view,
        "fit_orderflow_view": fit_orderflow_view,
    }


FEATURE_VIEW_NAMES = (
    "features",
    "features_of",
    "fit_features",
    "fit_features_of",
)
FEATURE_VIEW_ROUTING_SCHEMA = "m15-book-refresh-feature-view-routing/v1"


def _normalized_route_target(workspace: Path, path: Path) -> str:
    """Return the exact resolved route target under a sealed authority root."""

    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise RefreshError(f"feature-view route target is missing: {path}") from exc
    workspace_root = workspace.resolve(strict=True)
    repo_root = REPO_ROOT.resolve(strict=True)
    if resolved.is_relative_to(workspace_root):
        return "workspace:" + resolved.relative_to(workspace_root).as_posix()
    if resolved.is_relative_to(repo_root):
        return "repo:" + resolved.relative_to(repo_root).as_posix()
    raise RefreshError(f"feature-view route target escapes sealed authorities: {path}")


def _feature_view_route_rows(
    workspace: Path, *, include_sha256: bool
) -> list[dict[str, Any]]:
    views = workspace / "feature_views"
    if views.is_symlink() or not views.is_dir():
        raise RefreshError("feature_views must be a regular workspace directory")
    rows: list[dict[str, Any]] = []
    hashes_by_target: dict[str, str] = {}
    for name in FEATURE_VIEW_NAMES:
        directory = views / name
        if directory.is_symlink() or not directory.is_dir():
            raise RefreshError(f"feature-view directory is missing or symlinked: {directory}")
        for path in sorted(directory.iterdir(), key=lambda item: item.name):
            if path.is_dir() and not path.is_symlink():
                raise RefreshError(f"unexpected nested feature-view directory: {path}")
            link_type = "symlink" if path.is_symlink() else "regular"
            if link_type == "regular" and not path.is_file():
                raise RefreshError(f"feature-view entry is not a regular file: {path}")
            if path.suffix != ".parquet":
                raise RefreshError(f"unexpected non-parquet feature-view route: {path}")
            try:
                target_stat = path.stat()
            except OSError as exc:
                raise RefreshError(f"feature-view route is dangling: {path}") from exc
            resolved_target = _normalized_route_target(workspace, path)
            if include_sha256:
                digest = hashes_by_target.get(resolved_target)
                if digest is None:
                    digest = sha256_file(path)
                    hashes_by_target[resolved_target] = digest
            else:
                digest = ""
            rows.append(
                {
                    "consumer_path": path.relative_to(workspace).as_posix(),
                    "resolved_target": resolved_target,
                    "bytes": int(target_stat.st_size),
                    "sha256": digest,
                    "link_type": link_type,
                    "target_stat": {
                        "device": int(target_stat.st_dev),
                        "inode": int(target_stat.st_ino),
                        "mtime_ns": int(target_stat.st_mtime_ns),
                    },
                }
            )
    return rows


def _feature_view_routing_payload(workspace: Path) -> dict[str, Any]:
    return {
        "schema": FEATURE_VIEW_ROUTING_SCHEMA,
        "feature_views_root": "feature_views",
        "routes": _feature_view_route_rows(workspace, include_sha256=True),
    }


def _load_feature_view_routing_manifest(workspace: Path) -> dict[str, Any]:
    """Authenticate the shared routing authority without reading routed data."""

    manifest_path = workspace / "feature_views" / "routing_manifest.json"
    if (
        manifest_path.is_symlink()
        or not manifest_path.is_file()
        or manifest_path.stat().st_mode & 0o222
    ):
        raise RefreshError("feature-view routing manifest is not sealed read-only evidence")
    manifest = _read_json(manifest_path)
    if set(manifest) != {"schema", "feature_views_root", "routes"}:
        raise RefreshError("feature-view routing manifest fields differ")
    if (
        manifest.get("schema") != FEATURE_VIEW_ROUTING_SCHEMA
        or manifest.get("feature_views_root") != "feature_views"
        or not isinstance(manifest.get("routes"), list)
    ):
        raise RefreshError("feature-view routing manifest contract differs")
    required_route_fields = {
        "consumer_path", "resolved_target", "bytes", "sha256", "link_type",
        "target_stat",
    }
    if any(not isinstance(row, dict) or set(row) != required_route_fields
           for row in manifest["routes"]):
        raise RefreshError("feature-view routing manifest contains a malformed route")
    if any(
        not _is_hex(row.get("sha256"), 64)
        or row.get("link_type") not in {"symlink", "regular"}
        or not isinstance(row.get("target_stat"), dict)
        or set(row["target_stat"]) != {"device", "inode", "mtime_ns"}
        for row in manifest["routes"]
    ):
        raise RefreshError("feature-view routing manifest identity is malformed")
    consumer_paths = [row["consumer_path"] for row in manifest["routes"]]
    if len(consumer_paths) != len(set(consumer_paths)):
        raise RefreshError("feature-view routing manifest contains duplicate routes")
    return manifest


def _verify_feature_view_routes(
    workspace: Path, *, verify_target_bytes: bool = False
) -> dict[str, Any]:
    """Fail closed if any declared feature route or target byte changed."""

    manifest = _load_feature_view_routing_manifest(workspace)
    actual = _feature_view_route_rows(
        workspace, include_sha256=verify_target_bytes
    )
    if not verify_target_bytes:
        if len(actual) != len(manifest["routes"]):
            raise RefreshError("feature-view route inventory count changed")
        actual = [dict(row, sha256=sealed["sha256"])
                  for row, sealed in zip(actual, manifest["routes"])]
    if manifest["routes"] != actual:
        raise RefreshError(
            "feature-view routing changed (missing/unexpected/repointed route or target bytes)"
        )
    return manifest


def _feature_view_consumer_paths(
    workspace: Path,
    pair: str,
    years: Sequence[str | int],
    *,
    feature_dir: Path,
    orderflow_dir: Path,
) -> tuple[Path, ...]:
    """Resolve only the sealed routes one family builder will consume."""

    spec = load_spec()
    canonical_pair = str(pair).upper()
    if canonical_pair not in spec["pairs"]:
        raise PairRouteError(f"unknown feature-view pair {pair!r}")
    normalized_years = tuple(str(value) for value in years)
    if not normalized_years or len(normalized_years) != len(set(normalized_years)):
        raise PairRouteError("feature-view consumer years are empty or duplicated")
    family = spec["pairs"][canonical_pair]["adapter_family"]
    source_pairs = (
        spec["feature_source_pair_order"]
        if family in {"eur_xpair_of", "gbpchf_xpair"}
        else [canonical_pair]
    )
    paths = [
        Path(feature_dir) / f"{source_pair}_{year}.parquet"
        for year in normalized_years
        for source_pair in source_pairs
    ]
    if family == "eur_xpair_of":
        paths.extend(
            Path(orderflow_dir) / f"EURUSD_{year}.parquet"
            for year in normalized_years
        )
    views_root = (workspace / "feature_views").resolve(strict=True)
    for path in paths:
        try:
            path.parent.resolve(strict=True).relative_to(views_root)
        except (OSError, ValueError) as exc:
            raise PairRouteError(
                f"builder consumer path escapes feature views: {path}"
            ) from exc
    return tuple(paths)


def _verify_scoped_feature_view_bytes(
    workspace: Path, consumer_paths: Sequence[str | Path]
) -> dict[str, str]:
    """Hash only exact declared routes consumed by one builder invocation."""

    manifest = _load_feature_view_routing_manifest(workspace)
    sealed = {row["consumer_path"]: row for row in manifest["routes"]}
    paths = tuple(Path(value) for value in consumer_paths)
    if not paths or len({str(path) for path in paths}) != len(paths):
        raise PairRouteError("scoped feature-view paths are empty or duplicated")
    output: dict[str, str] = {}
    for path in paths:
        try:
            relative = path.relative_to(workspace).as_posix()
        except ValueError as exc:
            raise PairRouteError(
                f"scoped feature-view path escapes workspace: {path}"
            ) from exc
        expected = sealed.get(relative)
        if expected is None:
            raise PairRouteError(f"scoped feature-view path was not sealed: {relative}")
        if path.is_dir() or (not path.is_symlink() and not path.is_file()):
            raise PairRouteError(
                f"scoped feature-view path is missing/nonregular: {path}"
            )
        try:
            target_stat = path.stat()
        except OSError as exc:
            raise PairRouteError(f"scoped feature-view route is dangling: {path}") from exc
        try:
            resolved_target = _normalized_route_target(workspace, path)
        except RefreshError as exc:
            raise PairRouteError(f"scoped feature-view target is invalid: {path}") from exc
        current = {
            "consumer_path": relative,
            "resolved_target": resolved_target,
            "bytes": int(target_stat.st_size),
            "sha256": sha256_file(path),
            "link_type": "symlink" if path.is_symlink() else "regular",
            "target_stat": {
                "device": int(target_stat.st_dev),
                "inode": int(target_stat.st_ino),
                "mtime_ns": int(target_stat.st_mtime_ns),
            },
        }
        if current != expected:
            raise PairRouteError(
                f"scoped feature-view route/target bytes changed: {relative}"
            )
        output[relative] = current["sha256"]
    return output


@contextmanager
def verified_feature_view_access(
    workspace: Path,
    consumer_paths: Sequence[str | Path],
) -> Iterator[None]:
    """Cryptographically authenticate exact consumed routes before and after."""

    before = _verify_scoped_feature_view_bytes(workspace, consumer_paths)
    try:
        yield
    finally:
        after = _verify_scoped_feature_view_bytes(workspace, consumer_paths)
        if after != before:
            raise PairRouteError(
                "scoped feature-view bytes changed during builder access"
            )


def build_blind_snapshot(prereg_id: str) -> Path:
    """Build and seal only covariates; no fitting/replay label function is called."""

    workspace, store, prereg = workspace_for(prereg_id)
    _verify_preregistered_inventory(
        prereg, "historical_feature_files", _fit_historical_files("EURUSD", ["2025"])
    )
    _verify_all_processed_source_identities(prereg)
    if store.has("derived_data"):
        store.load("derived_data")
        _verify_feature_view_routes(workspace, verify_target_bytes=True)
        return workspace / "feature_snapshot" / "data_lock.json"
    snapshot = workspace / "feature_snapshot"
    if snapshot.exists():
        _remove_owned_tree(snapshot)
    views_root = workspace / "feature_views"
    if views_root.exists():
        _remove_owned_tree(views_root)
    (snapshot / "features").mkdir(parents=True)
    (snapshot / "features_of").mkdir()
    report: dict[str, Any] = {
        "schema": "m15-book-refresh-data-lock/v1",
        "prereg_id": prereg_id,
        "implementation_git_sha": prereg["implementation_git_sha"],
        "semantic_replay_outcomes_accessed": False,
        "base": {},
        "orderflow": {},
        "processed_sources": {},
        "processed_clock_audit": {},
    }
    for pair in load_spec()["feature_source_pair_order"]:
        print(f"[snapshot] blind covariates {pair} 2026", flush=True)
        report["processed_sources"][pair] = _source_file_identities(pair)
        report["processed_clock_audit"][pair] = _strict_processed_clock_audit(pair)
        report["base"][pair] = _blind_base_snapshot(
            pair, snapshot / "features" / f"{pair}_2026.parquet"
        )
    report["orderflow"]["EURUSD"] = _blind_orderflow_snapshot(
        snapshot / "features_of" / "EURUSD_2026.parquet"
    )
    views = _make_feature_views(workspace, snapshot)
    routing_manifest = workspace / "feature_views" / "routing_manifest.json"
    atomic_json_new(routing_manifest, _feature_view_routing_payload(workspace))
    _verify_feature_view_routes(workspace)
    report["feature_views"] = {key: str(path.relative_to(workspace)) for key, path in views.items()}
    report["feature_view_routing_manifest"] = {
        "path": routing_manifest.relative_to(workspace).as_posix(),
        "sha256": sha256_file(routing_manifest),
        "route_count": len(_read_json(routing_manifest)["routes"]),
    }
    report["fit_view_2026_sha256"] = {
        path.name: sha256_file(path)
        for path in sorted(views["fit_feature_view"].glob("*_2026.parquet"))
    }
    report["fit_view_2026_sha256"]["features_of/EURUSD_2026.parquet"] = sha256_file(
        views["fit_orderflow_view"] / "EURUSD_2026.parquet"
    )
    lock_path = snapshot / "data_lock.json"
    atomic_json_new(lock_path, report)
    for path in snapshot.rglob("*"):
        if path.is_file() and not path.is_symlink():
            path.chmod(0o444)
    for path in sorted((p for p in snapshot.rglob("*") if p.is_dir()), reverse=True):
        path.chmod(0o555)
    snapshot.chmod(0o555)
    for path in [
        *sorted(views["fit_feature_view"].glob("*_2026.parquet")),
        views["fit_orderflow_view"] / "EURUSD_2026.parquet",
    ]:
        path.chmod(0o444)
    tracked = RESULTS_JSON / f"m15_book_refresh_{prereg_id}_data_lock_result.json"
    _seal_json_exact(tracked, report)
    outputs = [lock_path, *sorted((snapshot / "features").glob("*.parquet")),
               snapshot / "features_of" / "EURUSD_2026.parquet",
               *sorted(views["fit_feature_view"].glob("*_2026.parquet")),
               views["fit_orderflow_view"] / "EURUSD_2026.parquet",
               routing_manifest, tracked]
    store.seal(
        "derived_data",
        canonical_attempt_id=f"{prereg_id}:derived:0",
        input_paths=[workspace / "source_preregistration.json"],
        output_paths=outputs,
        metadata={
            "semantic_replay_outcomes_accessed": False,
            "tracked_result": repo_relative(tracked),
        },
    )
    print(f"[snapshot] sealed {lock_path}", flush=True)
    return lock_path


def _read_cached_parity_projection(
    source: Path,
    columns: Sequence[str],
) -> pd.DataFrame:
    """Predicate-push and project the exact causal Q1 covariate window."""

    import pyarrow as pa
    import pyarrow.parquet as pq

    if source.is_symlink() or not source.is_file():
        raise RefreshError(f"cached-parity source is missing/nonregular: {source}")
    try:
        schema = pq.ParquetFile(source).schema_arrow
    except Exception as exc:
        raise RefreshError(f"cached-parity source schema cannot be read: {source}") from exc
    missing = [column for column in columns if column not in schema.names]
    if missing:
        raise RefreshError(
            f"cached-parity source lacks sealed projection columns: {source}: {missing}"
        )
    if "datetime_utc" not in schema.names:
        raise RefreshError(f"cached-parity source lacks datetime_utc clock: {source}")
    clock_type = schema.field("datetime_utc").type
    if (
        not pa.types.is_timestamp(clock_type)
        or clock_type.unit != "ns"
        or clock_type.tz != "UTC"
    ):
        raise RefreshError(
            f"cached-parity datetime_utc physical type differs: {source}: {clock_type}"
        )
    try:
        pandas_metadata = json.loads((schema.metadata or {})[b"pandas"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RefreshError(
            f"cached-parity source lacks valid pandas index metadata: {source}"
        ) from exc
    if pandas_metadata.get("index_columns") != ["datetime_utc"]:
        raise RefreshError(
            f"cached-parity source index clock differs: {source}"
        )

    # Cached feature timestamps are left-labelled source minutes.  The public
    # score clock advances them by 60 seconds, so the exact Q1 decision window
    # maps to this half-open source interval.  Both predicates are handed to
    # Arrow before any projected column is decoded.
    lower_source = pd.Timestamp("2026-01-01T00:00:00Z") - pd.Timedelta(seconds=60)
    upper_source = pd.Timestamp("2026-04-01T00:00:00Z") - pd.Timedelta(seconds=60)
    frame = pd.read_parquet(
        source,
        columns=list(columns),
        engine="pyarrow",
        filters=[
            ("datetime_utc", ">=", lower_source.to_pydatetime()),
            ("datetime_utc", "<", upper_source.to_pydatetime()),
        ],
    )
    if list(frame.columns) != list(columns):
        raise RefreshError(f"cached-parity projection order differs: {source}")
    if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.tz is None:
        raise RefreshError(f"cached-parity source clock is not timezone-aware: {source}")
    source_ns = np.asarray(
        frame.index.tz_convert("UTC").as_unit("ns").asi8,
        dtype="int64",
    )
    if (
        len(source_ns) == 0
        or np.any(source_ns < np.int64(lower_source.value))
        or np.any(source_ns >= np.int64(upper_source.value))
        or frame.index.has_duplicates
        or not frame.index.is_monotonic_increasing
    ):
        raise RefreshError(
            f"cached-parity projection escaped or violated the exact causal Q1 clock: {source}"
        )
    return frame


def _make_cached_parity_view(workspace: Path) -> tuple[Path, Path]:
    data_lock = _read_json(workspace / "feature_snapshot" / "data_lock.json")
    if (
        data_lock.get("schema") != "m15-book-refresh-data-lock/v1"
        or data_lock.get("semantic_replay_outcomes_accessed") is not False
    ):
        raise RefreshError("cached parity requires the sealed outcome-blind data lock")
    root = workspace / "cached_pre_april_parity"
    features = root / "features"
    orderflow = root / "features_of"
    if root.exists():
        _remove_owned_tree(root)
    features.mkdir(parents=True)
    orderflow.mkdir()
    for pair in load_spec()["feature_source_pair_order"]:
        source = REPO_ROOT / "features" / f"{pair}_2026.parquet"
        columns = data_lock.get("base", {}).get(pair, {}).get("columns")
        if (
            not isinstance(columns, list)
            or not columns
            or len(columns) != len(set(columns))
            or any(not isinstance(column, str) or column in FORBIDDEN_SEMANTIC_COLUMNS
                   for column in columns)
        ):
            raise RefreshError(f"{pair}: sealed cached-parity projection is malformed")
        frame = _read_cached_parity_projection(source, columns)
        _parquet_write_new(frame, features / source.name)
    source = REPO_ROOT / "features_of" / "EURUSD_2026.parquet"
    columns = data_lock.get("orderflow", {}).get("EURUSD", {}).get("columns")
    if (
        not isinstance(columns, list)
        or not columns
        or len(columns) != len(set(columns))
        or any(not isinstance(column, str) or column in FORBIDDEN_SEMANTIC_COLUMNS
               for column in columns)
    ):
        raise RefreshError("EURUSD: sealed cached-parity OF projection is malformed")
    frame = _read_cached_parity_projection(source, columns)
    _parquet_write_new(frame, orderflow / source.name)
    return features, orderflow


def _predict_boosters(boosters: Sequence[Any], matrix: np.ndarray) -> np.ndarray:
    X = np.ascontiguousarray(matrix, dtype="float32")
    predictions = [
        np.asarray(booster.predict(X, num_threads=1), dtype="float64")
        for booster in boosters
    ]
    if not predictions:
        raise RefreshError("at least one booster is required")
    return np.asarray(predictions, dtype="float64").mean(axis=0)


def _float32_ulp_distance(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    a = np.asarray(left, dtype="float32")
    b = np.asarray(right, dtype="float32")
    ua = a.view("uint32").astype("uint64")
    ub = b.view("uint32").astype("uint64")
    sign = np.uint64(1 << 31)
    full = np.uint64(1 << 32)
    ka = np.where((ua & sign) != 0, full - ua, ua + sign)
    kb = np.where((ub & sign) != 0, full - ub, ub + sign)
    return np.where(ka >= kb, ka - kb, kb - ka)


def _decoded_diagnostics(
    columns: Sequence[str], left: np.ndarray, right: np.ndarray
) -> dict[str, Any]:
    if left.shape != right.shape:
        raise RefreshError("decoded diagnostics require aligned matrices")
    rows: dict[str, Any] = {}
    total = 0
    eligibility_differences = 0
    for index, name in enumerate(columns):
        a = left[:, index].astype("float32", copy=False)
        b = right[:, index].astype("float32", copy=False)
        finite_a = np.isfinite(a)
        finite_b = np.isfinite(b)
        finite_disagreement = finite_a != finite_b
        common = finite_a & finite_b
        different = np.zeros(len(a), dtype=bool)
        different[common] = a[common].view("uint32") != b[common].view("uint32")
        different |= finite_disagreement
        count = int(different.sum())
        if not count:
            continue
        total += count
        eligibility_differences += int(finite_disagreement.sum())
        if common.any():
            abs_diff = np.abs(a[common].astype("float64") - b[common].astype("float64"))
            ulp = _float32_ulp_distance(a[common], b[common])
            max_abs = float(abs_diff.max())
            max_ulp = int(ulp.max())
        else:
            max_abs = None
            max_ulp = None
        rows[name] = {
            "mismatched_cells": count,
            "finite_nonfinite_disagreements": int(finite_disagreement.sum()),
            "max_abs_difference": max_abs,
            "max_ulp_difference": max_ulp,
        }
    return {
        "mismatched_cells": total,
        "finite_nonfinite_disagreements": eligibility_differences,
        "columns": rows,
    }


def _book_gate_vector(
    book: Any, matrix: np.ndarray, probabilities: np.ndarray
) -> np.ndarray:
    """Compatibility adapter over the canonical loaded-book component API."""

    frame = pd.DataFrame(
        np.asarray(matrix, dtype="float64"), columns=book.feature_cols
    )
    return book.gate_components(frame, probabilities).gate_passed


def _build_pre_april_live_provider(
    parity_root: Path,
) -> tuple[Any, Path, dict[str, Any]]:
    """Build the public LiveFeatureBuilder from pre-April rolling-store inputs."""

    import live_features
    import pipeline

    store_dir = parity_root / "live_provider_store"
    store_dir.mkdir()
    cutoff = pd.Timestamp("2026-04-01T00:00:00Z")
    maximum_rows = 0
    inputs: dict[str, Any] = {}
    for pair in load_spec()["feature_source_pair_order"]:
        raw = pipeline.load_pair_year_10s(pair, "2026")
        if raw is None or raw.empty:
            raise RefreshError(f"{pair}: no 2026 source for LiveFeatureBuilder parity")
        m1 = pipeline.resample_1m(raw)
        # A left-labeled source minute becomes a decision one minute later.
        # Keep historical warm-up, but never let the public provider cross the
        # exact pre-April decision boundary.
        m1 = m1.loc[
            m1.index + pd.Timedelta(seconds=60) < cutoff
        ]
        if len(m1) < 500:
            raise RefreshError(f"{pair}: insufficient pre-April live-provider rows")
        maximum_rows = max(maximum_rows, len(m1))
        path = store_dir / f"{pair}.parquet"
        _parquet_write_new(m1, path)
        inputs[pair] = {
            "rows": len(m1),
            "first_source_feature_ns": int(m1.index.as_unit("ns").asi8[0]),
            "last_source_feature_ns": int(m1.index.as_unit("ns").asi8[-1]),
            "sha256": sha256_file(path),
        }
    builder = live_features.LiveFeatureBuilder(
        client=None,
        history_minutes=maximum_rows + 1,
        tick_volume_count=0,
        store_dir=store_dir,
        stale_seconds=10**9,
    )
    return builder, store_dir, inputs


def _live_feature_builder_parity_report(
    pair: str,
    feature_cols: Sequence[str],
    score_rows: Any,
    builder: Any,
) -> dict[str, Any]:
    """Compare the actual public LiveFeatureBuilder recipe with ScoreRows."""

    expected = list(feature_cols)
    public_row = builder.feature_row(pair, expected)
    provider_frame = builder._joined_row_frame(pair, expected)
    missing = [column for column in expected if column not in provider_frame.columns]
    if missing:
        raise RefreshError(f"{pair}: LiveFeatureBuilder parity missing {missing[:12]}")
    if not isinstance(provider_frame.index, pd.DatetimeIndex) or str(
        provider_frame.index.tz
    ) != "UTC":
        raise RefreshError(f"{pair}: LiveFeatureBuilder parity clock is not UTC")
    provider_source_ns = assert_strict_clock(
        np.asarray(provider_frame.index.as_unit("ns").asi8, dtype="int64"),
        name=f"{pair} LiveFeatureBuilder provider source clock",
    )
    positions = np.searchsorted(provider_source_ns, score_rows.source_feature_ns)
    clock_subset_exact = bool(
        np.all(positions < len(provider_source_ns))
        and np.array_equal(
            provider_source_ns[positions], score_rows.source_feature_ns
        )
    )
    if clock_subset_exact:
        provider_matrix = np.ascontiguousarray(
            provider_frame.iloc[positions].loc[:, expected].to_numpy(
                dtype="float32", copy=True
            )
        )
    else:
        provider_matrix = np.empty((0, len(expected)), dtype="float32")
    feature_shape_equal = provider_matrix.shape == score_rows.X.shape
    finite_mask_equal = bool(
        feature_shape_equal
        and np.array_equal(np.isfinite(provider_matrix), np.isfinite(score_rows.X))
    )
    feature_bytes_equal = bool(
        feature_shape_equal
        and provider_matrix.tobytes(order="C") == score_rows.X.tobytes(order="C")
    )
    decoded_diagnostics = (
        _decoded_diagnostics(expected, score_rows.X, provider_matrix)
        if feature_shape_equal
        else None
    )

    clean = provider_frame.dropna(subset=expected)
    latest_expected = clean.iloc[-1].loc[expected].to_numpy(
        dtype="float32", copy=True
    )
    latest_public = public_row.row.loc[expected].to_numpy(
        dtype="float32", copy=True
    )
    public_latest_exact = bool(
        pd.Timestamp(public_row.timestamp) == pd.Timestamp(clean.index[-1])
        and np.array_equal(latest_public.view("uint32"), latest_expected.view("uint32"))
    )
    shift_ns = np.int64(60 * 1_000_000_000)
    decision_mapping_exact = bool(
        np.array_equal(
            score_rows.entry_ns,
            score_rows.source_feature_ns + shift_ns,
        )
    )
    report = {
        "schema": "m15-live-feature-builder-parity/v1",
        "pair": pair,
        "public_class": "live_features.LiveFeatureBuilder",
        "public_method_exercised": "feature_row",
        "provider_rows": len(provider_frame),
        "compared_rows": len(score_rows.entry_ns),
        "feature_count": len(expected),
        "feature_order_exact": tuple(public_row.row.index) == tuple(expected),
        "source_clock_subset_exact": clock_subset_exact,
        "causal_decision_mapping_exact": decision_mapping_exact,
        "finite_mask_exact": finite_mask_equal,
        "ordered_float32_feature_bytes_exact": feature_bytes_equal,
        "decoded_diagnostics": decoded_diagnostics,
        "public_latest_row_exact": public_latest_exact,
        "public_latest_source_feature_ns": int(pd.Timestamp(public_row.timestamp).value),
    }
    report["all_equal"] = all(
        report[name] is True
        for name in (
            "feature_order_exact",
            "source_clock_subset_exact",
            "causal_decision_mapping_exact",
            "finite_mask_exact",
            "ordered_float32_feature_bytes_exact",
            "public_latest_row_exact",
        )
    )
    return report


def run_adapter_parity(prereg_id: str) -> Path:
    """Seal pre-April decoded diagnostics and exact policy-behavior parity."""

    from m15_book_refresh_adapters import (
        build_rows,
        build_score_rows,
        direct_incumbent_builder_parity_report,
        slice_score_rows,
        window_mask,
    )
    import book_runtime

    workspace, store, prereg = workspace_for(prereg_id)
    _verify_preregistered_inventory(
        prereg, "comparator_A_files", comparator_a_files(), exact=True
    )
    _verify_preregistered_inventory(
        prereg, "historical_feature_files", _fit_historical_files("EURUSD", ["2026"])
    )
    store.load("derived_data")
    _verify_feature_view_routes(workspace)
    if store.has("adapter_parity"):
        store.load("adapter_parity")
        _verify_feature_view_routes(workspace)
        return workspace / "parity" / "parity.json"
    parity_root = workspace / "parity"
    if parity_root.exists():
        _remove_owned_tree(parity_root)
    parity_root.mkdir()
    cached_features, cached_of = _make_cached_parity_view(workspace)
    rebuilt_features = workspace / "feature_views" / "fit_features"
    rebuilt_of = workspace / "feature_views" / "fit_features_of"
    spec = load_spec()
    books = book_runtime.load_target_books(list(spec["pair_order"]))
    processed_before = _verify_all_processed_source_identities(prereg)
    live_provider, live_provider_store, live_provider_inputs = (
        _build_pre_april_live_provider(parity_root)
    )
    processed_after = _verify_all_processed_source_identities(prereg)
    if processed_after != processed_before:
        raise RefreshError("processed-source identities changed while building live provider")
    live_provider_inputs["processed_source_inventory_sha256"] = sha256_bytes(
        canonical_bytes(processed_after)
    )
    live_provider_inputs["processed_source_identity_pre_post_exact"] = True
    report: dict[str, Any] = {
        "schema": "m15-book-refresh-parity/v1",
        "prereg_id": prereg_id,
        "implementation_git_sha": prereg["implementation_git_sha"],
        "semantic_replay_outcomes_accessed": False,
        "window": {
            "entry_at_or_after": "2026-01-01T00:00:00Z",
            "entry_before": "2026-04-01T00:00:00Z",
        },
        "live_feature_builder_inputs": live_provider_inputs,
        "pairs": {},
    }
    for pair in spec["pair_order"]:
        print(f"[parity] {pair} pre-April", flush=True)
        book = books[pair]
        cached_score = slice_score_rows(
            build_score_rows(
                pair, ["2026"], book.feature_cols,
                feature_dir=cached_features, orderflow_dir=cached_of,
            ),
            entry_at_or_after=report["window"]["entry_at_or_after"],
            entry_before=report["window"]["entry_before"],
        )
        rebuilt_consumers = _feature_view_consumer_paths(
            workspace,
            pair,
            ["2026"],
            feature_dir=rebuilt_features,
            orderflow_dir=rebuilt_of,
        )
        with verified_feature_view_access(workspace, rebuilt_consumers):
            rebuilt_score = slice_score_rows(
                build_score_rows(
                    pair, ["2026"], book.feature_cols,
                    feature_dir=rebuilt_features, orderflow_dir=rebuilt_of,
                ),
                entry_at_or_after=report["window"]["entry_at_or_after"],
                entry_before=report["window"]["entry_before"],
            )
        if cached_score.feature_cols != rebuilt_score.feature_cols \
                or cached_score.feature_cols != tuple(book.feature_cols):
            raise RefreshError(f"{pair}: feature order differs in parity probe")
        if cached_score.X.dtype != rebuilt_score.X.dtype or cached_score.X.dtype != np.float32:
            raise RefreshError(f"{pair}: feature dtype differs in parity probe")
        if not np.array_equal(
            cached_score.source_feature_ns, rebuilt_score.source_feature_ns
        ):
            raise RefreshError(f"{pair}: source feature clock differs in parity probe")
        if not np.array_equal(cached_score.entry_ns, rebuilt_score.entry_ns):
            raise RefreshError(f"{pair}: full score clock differs in parity probe")
        shift_ns = np.int64(60 * 1_000_000_000)
        if not (
            np.array_equal(
                cached_score.entry_ns, cached_score.source_feature_ns + shift_ns
            )
            and np.array_equal(
                rebuilt_score.entry_ns, rebuilt_score.source_feature_ns + shift_ns
            )
        ):
            raise RefreshError(f"{pair}: causal source-to-decision clock mapping differs")
        diagnostics = _decoded_diagnostics(
            cached_score.feature_cols, cached_score.X, rebuilt_score.X
        )
        eligible_cached = np.isfinite(cached_score.X).all(axis=1)
        eligible_rebuilt = np.isfinite(rebuilt_score.X).all(axis=1)
        if not np.array_equal(eligible_cached, eligible_rebuilt):
            raise RefreshError(f"{pair}: finite/non-finite score eligibility differs")
        eligible = eligible_cached
        if not eligible.any():
            raise RefreshError(f"{pair}: no mutually scoreable rows in parity overlap")
        cached_frame = pd.DataFrame(
            cached_score.X[eligible].astype("float64"), columns=book.feature_cols
        )
        p_cached = book.predict_probabilities(cached_frame)
        components_cached = book.gate_components(cached_frame, p_cached)
        del cached_frame
        rebuilt_frame = pd.DataFrame(
            rebuilt_score.X[eligible].astype("float64"), columns=book.feature_cols
        )
        p_rebuilt = book.predict_probabilities(rebuilt_frame)
        components_rebuilt = book.gate_components(rebuilt_frame, p_rebuilt)
        del rebuilt_frame
        probability_bitwise = np.array_equal(
            p_cached.view("uint64"), p_rebuilt.view("uint64")
        )
        if not probability_bitwise:
            raise RefreshError(f"{pair}: A probabilities are not bitwise equal")
        # Bind the batch adapter to the public scalar LoadedBook scorer on a
        # deterministic spread of rows.  The all-row cached/rebuilt comparison
        # above remains the behavior gate; these scalar probes prove the same
        # model order, float conversion, direction tie, and gate authority.
        probe_count = min(257, len(p_cached))
        probe_ranks = np.unique(
            np.floor(np.linspace(0, len(p_cached) - 1, probe_count)).astype("int64")
        )
        eligible_matrix = cached_score.X[eligible]
        for rank in probe_ranks:
            scalar = book.score(
                pd.Series(eligible_matrix[rank], index=book.feature_cols, dtype="float64")
            )
            if np.float64(scalar.proba).view("uint64") != p_cached[rank].view("uint64"):
                raise RefreshError(f"{pair}: batch probability differs from LoadedBook.score")
        confidence_gate_exact = np.array_equal(
            components_cached.confidence_passed,
            components_rebuilt.confidence_passed,
        )
        structural_gate_exact = np.array_equal(
            components_cached.structural_passed,
            components_rebuilt.structural_passed,
        )
        if not confidence_gate_exact:
            raise RefreshError(f"{pair}: A confidence-threshold decisions differ")
        if not structural_gate_exact:
            raise RefreshError(f"{pair}: A structural-gate decisions differ")
        gate_cached = components_cached.gate_passed
        gate_rebuilt = components_rebuilt.gate_passed
        selected_cached = cached_score.entry_ns[eligible][gate_cached]
        selected_rebuilt = rebuilt_score.entry_ns[eligible][gate_rebuilt]
        ordered_pre_schedule_exact = np.array_equal(selected_cached, selected_rebuilt)
        if not ordered_pre_schedule_exact:
            raise RefreshError(f"{pair}: ordered A pre-schedule selections differ")

        # The legacy label-bearing adapter is allowed only on this physically
        # truncated pre-April view.  Its covariates must be an exact subset of
        # the outcome-free score adapter, proving the live/offline recipe link.
        with verified_feature_view_access(workspace, rebuilt_consumers):
            legacy = build_rows(
                pair, ["2026"], purpose="parity",
                feature_dir=rebuilt_features, orderflow_dir=rebuilt_of,
            )
        with verified_feature_view_access(workspace, rebuilt_consumers):
            direct_builder_parity = direct_incumbent_builder_parity_report(
                legacy,
                ["2026"],
                purpose="parity",
                feature_dir=rebuilt_features,
                orderflow_dir=rebuilt_of,
                entry_at_or_after=report["window"]["entry_at_or_after"],
                label_exit_before=report["window"]["entry_before"],
                cap=int(load_spec()["sealed_constants"]["train_row_cap"]),
            )
        if not direct_builder_parity["all_equal"]:
            raise RefreshError(
                f"{pair}: normalized adapter differs from direct incumbent builder: "
                f"{direct_builder_parity}"
            )
        legacy_window = window_mask(
            legacy,
            entry_at_or_after=report["window"]["entry_at_or_after"],
            entry_before=report["window"]["entry_before"],
        )
        legacy_positions = np.flatnonzero(legacy_window)
        legacy_row_ids = legacy.fit_row_id[legacy_positions]
        positions = np.searchsorted(rebuilt_score.entry_ns, legacy_row_ids)
        if (np.any(positions >= len(rebuilt_score.entry_ns))
                or not np.array_equal(rebuilt_score.entry_ns[positions], legacy_row_ids)):
            raise RefreshError(f"{pair}: legacy parity rows are not on score clock")
        live_equal = np.array_equal(
            rebuilt_score.X[positions].view("uint32"),
            legacy.X[legacy_positions].view("uint32"),
        )
        if not live_equal:
            raise RefreshError(f"{pair}: outcome-free/live and legacy adapter values differ")
        live_feature_builder_parity = None
        if pair != "EURUSD":
            live_feature_builder_parity = _live_feature_builder_parity_report(
                pair,
                book.feature_cols,
                rebuilt_score,
                live_provider,
            )
            if not live_feature_builder_parity["all_equal"]:
                raise RefreshError(
                    f"{pair}: public LiveFeatureBuilder parity failed: "
                    f"{live_feature_builder_parity}"
                )
        provider_sources = [
            asdict(file_identity(REPO_ROOT / "scripts" / name))
            for name in (
                "live_features.py",
                "pipeline.py",
                "m15_book_refresh_adapters.py",
                "book_runtime.py",
                "deriv_hot_daemon.py",
            )
        ]
        provider_status = "BLOCKED_SCHEMA" if pair == "EURUSD" else "PASSED"
        provider_reason = (
            "verified_live_OF_provider_absent"
            if pair == "EURUSD"
            else None
        )
        provider_parity = {
            "schema": "m15-book-refresh-provider-parity/v1",
            "status": provider_status,
            "reason": provider_reason,
            "feature_schema_parity": True,
            "feature_value_parity": bool(
                live_equal
                and direct_builder_parity["all_equal"]
                and (
                    pair == "EURUSD"
                    or live_feature_builder_parity is not None
                    and live_feature_builder_parity["all_equal"]
                )
            ),
            "runtime_scoring_and_gate_parity": bool(
                probability_bitwise
                and confidence_gate_exact
                and structural_gate_exact
                and ordered_pre_schedule_exact
                and len(probe_ranks) > 0
                and direct_builder_parity["source_to_decision_shift_exact"]
                and (
                    pair == "EURUSD"
                    or live_feature_builder_parity is not None
                    and live_feature_builder_parity["all_equal"]
                )
            ),
            "provider_schema_identity": sha256_bytes(
                canonical_bytes(list(book.feature_cols))
            ),
            "provider_implementation_identity": length_prefixed_digest(
                "m15-book-refresh/provider-implementation/v1",
                [canonical_bytes(provider_sources)],
            ),
            "provider_source_files": provider_sources,
            "runtime_implementation_git_sha": prereg["implementation_git_sha"],
            "runtime_decision_clock_authority": {
                "function": "deriv_hot_daemon.score_from_snapshot",
                "field": "candidate_bar_close_utc",
                "mapping": "LiveFeatureBuilder.FeatureRow.timestamp+1minute",
                "source_feature_to_decision_shift_s": 60,
                "adapter_mapping_exact": bool(
                    direct_builder_parity["source_to_decision_shift_exact"]
                ),
                "authority_source_sha256": next(
                    row["sha256"]
                    for row in provider_sources
                    if row["path"].endswith("deriv_hot_daemon.py")
                ),
            },
            "direct_incumbent_builder_parity": direct_builder_parity,
            "live_feature_builder_parity": live_feature_builder_parity,
            "provider_kind": (
                "offline_xpair_plus_unavailable_live_orderflow"
                if pair == "EURUSD"
                else (
                    "public_live_feature_builder_cross_pair"
                    if load_spec()["pairs"][pair]["adapter_family"] == "gbpchf_xpair"
                    else "public_live_feature_builder_base"
                )
            ),
        }
        report["pairs"][pair] = {
            "book_id_A": book.book_id,
            "rows": len(cached_score.entry_ns),
            "scoreable_rows": int(eligible.sum()),
            "feature_count": len(book.feature_cols),
            "feature_dtype": "float32",
            "clock_exact": True,
            "source_feature_clock_i64le_sha256": sha256_bytes(
                np.asarray(cached_score.source_feature_ns, dtype="<i8").tobytes()
            ),
            "decision_clock_i64le_sha256": sha256_bytes(
                np.asarray(cached_score.entry_ns, dtype="<i8").tobytes()
            ),
            "source_feature_to_decision_shift_s": 60,
            "causal_decision_mapping_exact": True,
            "schema_order_exact": True,
            "finite_eligibility_exact": True,
            "probability_bitwise_exact": bool(probability_bitwise),
            "direction_exact": bool(np.array_equal(p_cached >= 0.5, p_rebuilt >= 0.5)),
            "confidence_gate_exact": bool(confidence_gate_exact),
            "structural_gate_exact": bool(structural_gate_exact),
            "ordered_pre_schedule_exact": bool(ordered_pre_schedule_exact),
            "pre_schedule_count": int(gate_cached.sum()),
            "outcome_free_live_adapter_exact": True,
            "loaded_book_scalar_probe_rows": len(probe_ranks),
            "loaded_book_scalar_probe_exact": True,
            "decoded_diagnostics": diagnostics,
            "provider_parity": provider_parity,
        }
    _verify_feature_view_routes(workspace)
    _remove_owned_tree(live_provider_store)
    path = parity_root / "parity.json"
    atomic_json_new(path, report)
    tracked = RESULTS_JSON / f"m15_book_refresh_{prereg_id}_parity_result.json"
    _seal_json_exact(tracked, report)
    store.seal(
        "adapter_parity",
        canonical_attempt_id=f"{prereg_id}:parity:0",
        input_paths=[workspace / "feature_snapshot" / "data_lock.json"],
        output_paths=[path, tracked],
        metadata={
            "semantic_replay_outcomes_accessed": False,
            "tracked_result": repo_relative(tracked),
        },
    )
    print(f"[parity] sealed {path}", flush=True)
    return path


def _fit_result_path(workspace: Path, pair: str, arm: str, attempt: str) -> Path:
    return workspace / "fits" / pair / arm / attempt / "fit_result.json"


def _read_json(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    try:
        value = json.loads(target.read_bytes())
    except (OSError, json.JSONDecodeError) as exc:
        raise RefreshError(f"cannot read JSON {target}: {exc}") from exc
    if not isinstance(value, dict):
        raise RefreshError(f"JSON object required: {target}")
    return value


def _validate_frozen_fit_contract(
    spec: Mapping[str, Any],
    *,
    pair: str,
    arm: str,
    resolved: Mapping[str, Any],
    strategy: Mapping[str, Any],
    result: Mapping[str, Any],
) -> None:
    """Authenticate every immutable B/C/control behavior against authority."""

    from m15_book_refresh_adapters import contract_for, model_parameters

    pair_spec = spec["pairs"][pair]
    contract = contract_for(pair)
    base_arm = arm[0]
    resolved_fields = {
        "schema", "book_id", "pair", "arm", "adapter_family", "builder",
        "implementation_git_sha", "prereg_id", "feature_cols", "feature_dtype",
        "fit_ties", "fit_session", "calibration_session", "training_stride",
        "stride_phase", "cap_selector", "fit_row_count", "fit_row_id_sha256",
        "early_stopping_row_count", "early_stopping_row_id_sha256",
        "calibration_row_count", "calibration_row_id_sha256",
        "fit_label_u8_sha256", "early_stopping_label_u8_sha256",
        "calibration_label_u8_sha256", "split", "model_parameters_by_seed",
        "seed_order", "best_iterations", "early_stopping_rule",
        "nominal_source_years", "nominal_file_spill_audit", "calibration",
        "probability_tie_rule", "control_permutation_seed",
        "control_permutation", "lifecycle_status", "bundle_id",
    }
    strategy_fields = {
        "schema", "book_id", "pair", "arm", "adapter_family", "builder",
        "implementation_git_sha", "prereg_id", "lifecycle_status",
        "feature_cols", "n_features", "feature_dtype", "seed_order", "ensemble",
        "coverage", "conf_thr", "gate", "horizon_seconds",
        "outer_evaluator_owned",
    }
    result_fields = {
        "schema", "prereg_id", "pair", "arm", "attempt", "bundle_id",
        "fit_row_count", "fit_row_id_sha256", "early_stopping_row_count",
        "early_stopping_row_id_sha256", "calibration_row_count",
        "calibration_row_id_sha256", "fit_label_u8_sha256",
        "early_stopping_label_u8_sha256", "calibration_label_u8_sha256",
        "split", "nominal_file_spill_audit", "best_iterations", "calibration",
        "feature_cols", "strategy", "resolved_bundle_spec", "models",
        "model_sha256", "seed_checkpoint_manifests",
        "seed_checkpoint_manifest_sha256", "control_permutation_seed",
        "control_permutation",
    }
    if set(resolved) != resolved_fields:
        raise RefreshError(f"{pair} {arm}: resolved frozen-contract fields differ")
    if set(strategy) != strategy_fields:
        raise RefreshError(f"{pair} {arm}: strategy frozen-contract fields differ")
    if set(result) != result_fields:
        raise RefreshError(f"{pair} {arm}: fit-result contract fields differ")

    exact_resolved = {
        "feature_dtype": pair_spec["feature_dtype"],
        "fit_ties": pair_spec["fit_ties"],
        "fit_session": pair_spec["fit_session"],
        "calibration_session": pair_spec["calibration_session"],
        "training_stride": pair_spec["training_stride"],
        "stride_phase": pair_spec["stride_phase"],
        "cap_selector": spec["cap_selector"],
        "split": spec["splits"][base_arm],
        "seed_order": pair_spec["seed_order"],
        "probability_tie_rule": spec["sealed_constants"]["probability_tie_rule"],
    }
    differences = [
        name for name, expected in exact_resolved.items()
        if resolved.get(name) != expected
    ]
    if differences:
        raise RefreshError(
            f"{pair} {arm}: immutable resolved behavior differs: {differences}"
        )

    expected_parameters = [
        model_parameters(
            pair,
            int(seed),
            n_jobs=int(pair_spec["model"]["num_threads"]),
        )
        for seed in pair_spec["seed_order"]
    ]
    if resolved.get("model_parameters_by_seed") != expected_parameters:
        raise RefreshError(f"{pair} {arm}: model parameters differ from adapter/spec authority")

    expected_years = {
        "fit": [
            str(year)
            for year in range(2012, 2022 if base_arm == "B" else 2026)
        ],
        "validation_and_calibration": (
            ["2021", "2022", "2023"]
            if base_arm == "B"
            else ["2025", "2026"]
        ),
    }
    if resolved.get("nominal_source_years") != expected_years:
        raise RefreshError(f"{pair} {arm}: nominal source-year contract differs")

    fit_count = resolved.get("fit_row_count")
    early_count = resolved.get("early_stopping_row_count")
    calibration_count = resolved.get("calibration_row_count")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in (fit_count, early_count, calibration_count)
    ) or int(fit_count) > int(spec["sealed_constants"]["train_row_cap"]):
        raise RefreshError(f"{pair} {arm}: frozen partition counts are invalid")

    early_rule = resolved.get("early_stopping_rule")
    expected_early_rule = {
        "rounds": contract.early_stopping_rounds,
        "metric": "auc",
        "rows": early_count,
        "ties": pair_spec["early_stopping_ties"],
    }
    if early_rule != expected_early_rule:
        raise RefreshError(f"{pair} {arm}: early-stopping contract differs")

    best_iterations = resolved.get("best_iterations")
    if (
        not isinstance(best_iterations, list)
        or len(best_iterations) != len(pair_spec["seed_order"])
        or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 0 < value <= int(pair_spec["model"]["n_estimators"])
            for value in best_iterations
        )
    ):
        raise RefreshError(f"{pair} {arm}: best-iteration vector is invalid")

    calibration = resolved.get("calibration")
    calibration_fields = {
        "pair", "target_coverage", "confidence_threshold", "calibration_rows",
        "confidence_rows", "structural_rows", "structural_column",
        "structural_quantile", "structural_threshold",
    }
    if not isinstance(calibration, dict) or set(calibration) != calibration_fields:
        raise RefreshError(f"{pair} {arm}: calibration state fields differ")
    structural_spec = pair_spec["structural_gate"]
    if structural_spec["type"] == "none":
        expected_column = None
        expected_quantile = None
    else:
        expected_column = structural_spec["column"]
        expected_quantile = structural_spec["quantile"]
    threshold = calibration.get("confidence_threshold")
    confidence_rows = calibration.get("confidence_rows")
    structural_threshold = calibration.get("structural_threshold")
    structural_rows = calibration.get("structural_rows")
    if (
        calibration.get("pair") != pair
        or calibration.get("target_coverage")
        != pair_spec["calibration_target_coverage"]
        or calibration.get("calibration_rows") != calibration_count
        or isinstance(confidence_rows, bool)
        or not isinstance(confidence_rows, int)
        or not 0 < confidence_rows <= int(calibration_count)
        or isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not np.isfinite(float(threshold))
        or not 0.0 <= float(threshold) <= 0.5
        or calibration.get("structural_column") != expected_column
        or calibration.get("structural_quantile") != expected_quantile
    ):
        raise RefreshError(f"{pair} {arm}: calibration frozen behavior differs")
    if expected_column is None:
        if structural_rows is not None or structural_threshold is not None:
            raise RefreshError(f"{pair} {arm}: undeclared structural calibration state")
    elif (
        isinstance(structural_rows, bool)
        or not isinstance(structural_rows, int)
        or structural_rows <= 0
        or isinstance(structural_threshold, bool)
        or not isinstance(structural_threshold, (int, float))
        or not np.isfinite(float(structural_threshold))
    ):
        raise RefreshError(f"{pair} {arm}: structural calibration state is invalid")

    expected_gate = {
        "confidence": "abs(p-0.5)>=conf_thr",
        "probability_tie": spec["sealed_constants"]["probability_tie_rule"],
        "structural_column": expected_column,
        "structural_quantile": expected_quantile,
        "structural_threshold": structural_threshold,
    }
    exact_strategy = {
        "feature_cols": resolved.get("feature_cols"),
        "n_features": len(resolved.get("feature_cols", [])),
        "feature_dtype": pair_spec["feature_dtype"],
        "seed_order": pair_spec["seed_order"],
        "ensemble": "arithmetic_mean_probability_in_seed_order",
        "coverage": pair_spec["calibration_target_coverage"],
        "conf_thr": threshold,
        "gate": expected_gate,
        "horizon_seconds": spec["sealed_constants"]["horizon_seconds"],
        "outer_evaluator_owned": [
            "ny_session", "last_start_16:35", "wc_ret", "nonoverlap_chrono"
        ],
    }
    strategy_differences = [
        name for name, expected in exact_strategy.items()
        if strategy.get(name) != expected
    ]
    if strategy_differences:
        raise RefreshError(
            f"{pair} {arm}: immutable strategy behavior differs: {strategy_differences}"
        )


def _validate_sealed_fit_result(
    workspace: Path,
    path: Path,
    *,
    prereg_id: str,
    pair: str,
    arm: str,
    attempt: str,
) -> dict[str, Any]:
    """Authenticate a resumable fit result from its declared artifact bytes."""

    spec = load_spec()
    result = _read_json(path)
    if result.get("schema") != "m15-book-refresh-fit-result/v1" or any(
        result.get(name) != expected
        for name, expected in (
            ("prereg_id", prereg_id), ("pair", pair), ("arm", arm), ("attempt", attempt)
        )
    ):
        raise RefreshError(f"conflicting sealed fit attempt: {path}")
    if path.is_symlink() or not path.is_file() or path.stat().st_mode & 0o222:
        raise RefreshError(f"fit result is not a read-only regular file: {path}")
    strategy = workspace / str(result.get("strategy", ""))
    resolved_path = workspace / str(result.get("resolved_bundle_spec", ""))
    models = [workspace / str(value) for value in result.get("models", [])]
    row_id_paths = {
        "fit": path.parent / "fit_row_id.i64le",
        "early_stopping": path.parent / "early_stopping_row_id.i64le",
        "calibration": path.parent / "calibration_row_id.i64le",
    }
    checkpoint_manifests = [
        workspace / str(value)
        for value in result.get("seed_checkpoint_manifests", [])
    ]
    artifacts = [
        strategy,
        resolved_path,
        *row_id_paths.values(),
        *models,
        *checkpoint_manifests,
    ]
    declared_relatives = [
        str(result.get("strategy", "")),
        str(result.get("resolved_bundle_spec", "")),
        *[str(value) for value in result.get("models", [])],
        *[str(value) for value in result.get("seed_checkpoint_manifests", [])],
    ]
    if [repo for repo in declared_relatives] != [
        _workspace_relative(workspace, item)
        for item in [strategy, resolved_path, *models, *checkpoint_manifests]
    ]:
        raise RefreshError(f"fit result artifact path normalization differs: {path}")
    if any(p.is_symlink() or not p.is_file() or p.stat().st_mode & 0o222 for p in artifacts):
        raise RefreshError(f"fit result has missing, symlinked, or writable artifacts: {path}")
    if [sha256_file(model) for model in models] != result.get("model_sha256"):
        raise RefreshError(f"fit result model hashes differ: {path}")
    for partition, row_ids in row_id_paths.items():
        count_field = f"{partition}_row_count"
        hash_field = f"{partition}_row_id_sha256"
        if (
            sha256_file(row_ids) != result.get(hash_field)
            or row_ids.stat().st_size != int(result.get(count_field, -1)) * 8
        ):
            raise RefreshError(
                f"fit result {partition} row-id seal differs: {path}"
            )
    feature_cols = result.get("feature_cols")
    if (
        not isinstance(feature_cols, list)
        or len(feature_cols) != int(spec["pairs"][pair]["feature_count"])
        or len(set(feature_cols)) != len(feature_cols)
        or any(not isinstance(column, str) or not column for column in feature_cols)
    ):
        raise RefreshError(f"fit result feature schema is malformed: {path}")
    strategy_value = _read_json(strategy)
    resolved_with_id = _read_json(resolved_path)
    resolved = dict(resolved_with_id)
    prereg = _read_json(workspace / "source_preregistration.json")
    implementation_git_sha = prereg.get("implementation_git_sha")
    base_arm = arm[0]
    is_control = arm.endswith("_perm")
    expected_lifecycle = (
        "inactive_shadow_candidate"
        if base_arm == "C" and not is_control
        else "workspace_only_nonpublishable"
    )
    expected_book_id = (
        spec["pairs"][pair]["candidate_book_id"]
        if base_arm == "C" and not is_control
        else f"workspace:{pair}:{arm}"
    )
    common_identity = {
        "schema": "m15-book-refresh-resolved-bundle-spec/v1",
        "pair": pair,
        "arm": arm,
        "prereg_id": prereg_id,
        "implementation_git_sha": implementation_git_sha,
        "adapter_family": spec["pairs"][pair]["adapter_family"],
        "builder": spec["pairs"][pair]["builder"],
        "seed_order": list(spec["pairs"][pair]["seed_order"]),
        "book_id": expected_book_id,
        "lifecycle_status": expected_lifecycle,
    }
    strategy_identity = dict(
        common_identity, schema="m15-book-refresh-strategy/v1"
    )
    if (
        prereg.get("prereg_id") != prereg_id
        or not _is_hex(implementation_git_sha, 40)
        or any(resolved.get(name) != value for name, value in common_identity.items())
        or any(
            strategy_value.get(name) != value
            for name, value in strategy_identity.items()
        )
    ):
        raise RefreshError(f"fit result resolved/strategy identity differs: {path}")
    if not (
        strategy_value.get("feature_cols") == feature_cols
        and strategy_value.get("n_features") == len(feature_cols)
        and strategy_value.get("feature_dtype") == "float32"
        and resolved.get("feature_cols") == feature_cols
        and resolved.get("feature_dtype") == "float32"
        and len(models) == len(spec["pairs"][pair]["seed_order"])
    ):
        raise RefreshError(f"fit result strategy/resolved/model feature contract differs: {path}")
    if resolved.get("early_stopping_rule", {}).get("ties") != spec["pairs"][pair][
        "early_stopping_ties"
    ]:
        raise RefreshError(f"fit result early-stopping tie contract differs: {path}")
    expected_split = spec["splits"][base_arm]
    duplicate_fields = (
        "fit_row_count",
        "fit_row_id_sha256",
        "early_stopping_row_count",
        "early_stopping_row_id_sha256",
        "calibration_row_count",
        "calibration_row_id_sha256",
        "fit_label_u8_sha256",
        "early_stopping_label_u8_sha256",
        "calibration_label_u8_sha256",
        "best_iterations",
        "calibration",
        "nominal_file_spill_audit",
        "split",
        "control_permutation_seed",
        "control_permutation",
    )
    duplicate_differences = [
        field for field in duplicate_fields
        if result.get(field) != resolved.get(field)
    ]
    if duplicate_differences:
        raise RefreshError(
            f"fit result duplicates differ from bundle-authenticated resolved spec: "
            f"{path}: {duplicate_differences}"
        )
    _validate_frozen_fit_contract(
        spec,
        pair=pair,
        arm=arm,
        resolved=resolved,
        strategy=strategy_value,
        result=result,
    )
    expected_model_names = [
        f"{pair}_{arm}_s{int(seed)}_lgb.txt"
        for seed in spec["pairs"][pair]["seed_order"]
    ]
    if [model.name for model in models] != expected_model_names:
        raise RefreshError(f"fit result model filename/seed order differs: {path}")
    if [manifest.name for manifest in checkpoint_manifests] != [
        f"{name}.checkpoint.json" for name in expected_model_names
    ]:
        raise RefreshError(f"fit result checkpoint filename/seed order differs: {path}")
    if strategy.name != f"{pair}_{arm}_strategy.json" or resolved_path.name != (
        "resolved_bundle_spec.json"
    ):
        raise RefreshError(f"fit result policy artifact filenames differ: {path}")
    permutation = result.get("control_permutation")
    control_seed = result.get("control_permutation_seed")
    if is_control:
        family = spec["pairs"][pair]["adapter_family"]
        expected_control_seed = int(
            spec["sealed_constants"]["permutation_seeds"][family]
        )
        expected_arm_code = int(
            spec["control_family"]["label_permutation"]["arm_codes"][arm]
        )
        if (
            control_seed != expected_control_seed
            or not isinstance(permutation, dict)
            or any(
                permutation.get(name) != value
                for name, value in spec["control_family"]["label_permutation"].items()
            )
            or permutation.get("family") != family
            or permutation.get("family_permutation_seed") != expected_control_seed
            or permutation.get("arm") != arm
            or permutation.get("arm_code") != expected_arm_code
            or permutation.get("resolved_seed_material")
            != [expected_control_seed, expected_arm_code]
            or permutation.get("partition_rows")
            != {
                "capped_fit_rows": result.get("fit_row_count"),
                "early_stopping_rows": result.get("early_stopping_row_count"),
            }
        ):
            raise RefreshError(f"fit result control-permutation contract differs: {path}")
    elif control_seed is not None or permutation is not None:
        raise RefreshError(f"primary fit result carries control permutation state: {path}")
    if result.get("split") != expected_split:
        raise RefreshError(f"fit result split differs from sealed {base_arm} split: {path}")
    calibration = resolved.get("calibration")
    gate = strategy_value.get("gate")
    if not isinstance(calibration, dict) or not isinstance(gate, dict):
        raise RefreshError(f"fit result bundle policy state is malformed: {path}")
    expected_strategy_policy = {
        "coverage": calibration.get("target_coverage"),
        "conf_thr": calibration.get("confidence_threshold"),
        "structural_column": calibration.get("structural_column"),
        "structural_quantile": calibration.get("structural_quantile"),
        "structural_threshold": calibration.get("structural_threshold"),
    }
    actual_strategy_policy = {
        "coverage": strategy_value.get("coverage"),
        "conf_thr": strategy_value.get("conf_thr"),
        "structural_column": gate.get("structural_column"),
        "structural_quantile": gate.get("structural_quantile"),
        "structural_threshold": gate.get("structural_threshold"),
    }
    if actual_strategy_policy != expected_strategy_policy:
        raise RefreshError(
            f"strategy policy differs from bundle-authenticated calibration: {path}"
        )
    resolved_best = resolved.get("best_iterations")
    if (
        not isinstance(resolved_best, list)
        or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0
               for value in resolved_best)
        or result.get("best_iterations") != resolved_best
    ):
        raise RefreshError(f"fit result best iterations are malformed: {path}")
    parameters_by_seed = resolved.get("model_parameters_by_seed")
    if (
        not isinstance(parameters_by_seed, list)
        or len(parameters_by_seed) != len(models)
        or any(not isinstance(value, dict) for value in parameters_by_seed)
    ):
        raise RefreshError(f"fit result model-parameter inventory differs: {path}")
    if (
        len(checkpoint_manifests) != len(models)
        or len(result.get("seed_checkpoint_manifest_sha256", [])) != len(models)
        or [sha256_file(value) for value in checkpoint_manifests]
        != result.get("seed_checkpoint_manifest_sha256")
    ):
        raise RefreshError(f"fit seed-checkpoint manifest inventory differs: {path}")
    for position, (seed, model, checkpoint_manifest) in enumerate(
        zip(spec["pairs"][pair]["seed_order"], models, checkpoint_manifests)
    ):
        checkpoint = _validate_seed_checkpoint_manifest(
            checkpoint_manifest,
            model,
            prereg_id=prereg_id,
            implementation_git_sha=str(resolved.get("implementation_git_sha")),
            pair=pair,
            arm=arm,
            attempt=attempt,
            seed=int(seed),
            seed_order_position=position,
            feature_cols=feature_cols,
            model_parameters=parameters_by_seed[position],
            fit_row_id_sha256=str(result.get("fit_row_id_sha256")),
            early_stopping_row_id_sha256=str(
                result.get("early_stopping_row_id_sha256")
            ),
            fit_label_sha256=str(result.get("fit_label_u8_sha256")),
            early_stopping_label_sha256=str(
                result.get("early_stopping_label_u8_sha256")
            ),
        )
        if checkpoint.best_iteration != int(result["best_iterations"][position]):
            raise RefreshError(f"fit checkpoint best iteration differs: {path}")
    bundle_id = resolved.pop("bundle_id", None)
    computed = derive_bundle_id(
        spec,
        resolved,
        strategy.name,
        strategy.read_bytes(),
        [(model.name, model.read_bytes()) for model in models],
    )
    if bundle_id != computed or result.get("bundle_id") != computed:
        raise RefreshError(f"fit result bundle identity differs: {path}")
    return result


def _workspace_relative(workspace: Path, path: str | Path) -> str:
    try:
        return Path(path).resolve(strict=True).relative_to(workspace.resolve(strict=True)).as_posix()
    except (OSError, ValueError) as exc:
        raise RefreshError(f"path is outside prereg workspace: {path}") from exc


def _i64le_sha256(values: Sequence[int] | np.ndarray) -> str:
    return sha256_bytes(np.asarray(values, dtype="<i8").tobytes(order="C"))


def _u8_sha256(values: Sequence[int] | np.ndarray) -> str:
    return sha256_bytes(np.asarray(values, dtype="uint8").tobytes(order="C"))


def _seed_checkpoint_manifest_path(model_path: Path) -> Path:
    return model_path.with_name(f"{model_path.name}.checkpoint.json")


def _validate_seed_checkpoint_manifest(
    manifest_path: Path,
    model_path: Path,
    *,
    prereg_id: str,
    implementation_git_sha: str,
    pair: str,
    arm: str,
    attempt: str,
    seed: int,
    seed_order_position: int,
    feature_cols: Sequence[str],
    model_parameters: Mapping[str, Any],
    fit_row_id_sha256: str,
    early_stopping_row_id_sha256: str,
    fit_label_sha256: str,
    early_stopping_label_sha256: str,
) -> Any:
    """Authenticate one resumable seed checkpoint and its exact fit context."""

    from m15_book_refresh_adapters import SeedCheckpoint

    if (
        manifest_path.is_symlink()
        or not manifest_path.is_file()
        or manifest_path.stat().st_mode & 0o222
        or model_path.is_symlink()
        or not model_path.is_file()
        or model_path.stat().st_mode & 0o222
    ):
        raise RefreshError(
            f"seed checkpoint is missing, symlinked, or writable: {manifest_path}"
        )
    value = _read_json(manifest_path)
    expected = {
        "schema": "m15-book-refresh-seed-checkpoint/v1",
        "prereg_id": prereg_id,
        "implementation_git_sha": implementation_git_sha,
        "pair": pair,
        "arm": arm,
        "attempt": attempt,
        "seed": int(seed),
        "seed_order_position": int(seed_order_position),
        "feature_cols_sha256": sha256_bytes(canonical_bytes(list(feature_cols))),
        "model_parameters": dict(model_parameters),
        "fit_row_id_sha256": fit_row_id_sha256,
        "early_stopping_row_id_sha256": early_stopping_row_id_sha256,
        "fit_label_u8_sha256": fit_label_sha256,
        "early_stopping_label_u8_sha256": early_stopping_label_sha256,
        "model_filename": model_path.name,
    }
    dynamic_fields = {"best_iteration", "model_sha256"}
    if set(value) != set(expected) | dynamic_fields or any(
        value.get(name) != expected_value for name, expected_value in expected.items()
    ):
        raise RefreshError(f"seed checkpoint context differs: {manifest_path}")
    best_iteration = value.get("best_iteration")
    if (
        isinstance(best_iteration, bool)
        or not isinstance(best_iteration, int)
        or best_iteration <= 0
        or value.get("model_sha256") != sha256_file(model_path)
    ):
        raise RefreshError(f"seed checkpoint identity differs: {manifest_path}")
    try:
        booster = __import__("lightgbm").Booster(model_file=str(model_path))
        model_features = int(booster.num_feature())
        model_iterations = int(booster.current_iteration())
    except Exception as exc:
        raise RefreshError(f"seed checkpoint model cannot be loaded: {model_path}: {exc}") from exc
    if model_features != len(feature_cols) or model_iterations != best_iteration:
        raise RefreshError(f"seed checkpoint model structure differs: {model_path}")
    return SeedCheckpoint(
        pair=pair,
        seed=int(seed),
        best_iteration=best_iteration,
        model_path=model_path,
        sha256=str(value["model_sha256"]),
    )


def _fit_or_resume_seed_checkpoints(
    *,
    workspace: Path,
    prereg_id: str,
    implementation_git_sha: str,
    pair: str,
    arm: str,
    attempt: str,
    fit_rows: Any,
    fit_indices: np.ndarray,
    validation_rows: Any,
    validation_indices: np.ndarray,
    model_dir: Path,
    seeds: Sequence[int],
    parameters_by_seed: Sequence[Mapping[str, Any]],
    n_jobs: int,
    fit_row_id_sha256: str,
    early_stopping_row_id_sha256: str,
    fit_label_sha256: str,
    early_stopping_label_sha256: str,
    fit_seed: Callable[..., Any] | None = None,
) -> tuple[list[Any], list[Path]]:
    """Resume exact sealed seed/model pairs; fit only missing seed checkpoints."""

    if len(seeds) != len(parameters_by_seed):
        raise RefreshError(f"{pair} {arm}: seed/model-parameter inventory differs")
    if fit_seed is None:
        from m15_book_refresh_adapters import fit_seed_checkpoint as fit_seed

    model_dir.mkdir(parents=True, exist_ok=True)
    checkpoints: list[Any] = []
    manifests: list[Path] = []
    for position, (seed, model_parameters_value) in enumerate(
        zip(seeds, parameters_by_seed)
    ):
        model_path = model_dir / f"{pair}_{arm}_s{int(seed)}_lgb.txt"
        manifest_path = _seed_checkpoint_manifest_path(model_path)
        model_exists = model_path.exists() or model_path.is_symlink()
        manifest_exists = manifest_path.exists() or manifest_path.is_symlink()
        if model_exists != manifest_exists:
            # A single member cannot authenticate its seed/fit context.  Remove
            # only this known unsealed orphan and refit; never clear other seeds.
            orphan = model_path if model_exists else manifest_path
            if orphan.is_symlink() or not orphan.is_file():
                raise RefreshError(f"invalid partial seed checkpoint: {orphan}")
            orphan.chmod(0o600)
            orphan.unlink()
            model_exists = manifest_exists = False
        validator_kwargs = {
            "prereg_id": prereg_id,
            "implementation_git_sha": implementation_git_sha,
            "pair": pair,
            "arm": arm,
            "attempt": attempt,
            "seed": int(seed),
            "seed_order_position": position,
            "feature_cols": fit_rows.feature_cols,
            "model_parameters": model_parameters_value,
            "fit_row_id_sha256": fit_row_id_sha256,
            "early_stopping_row_id_sha256": early_stopping_row_id_sha256,
            "fit_label_sha256": fit_label_sha256,
            "early_stopping_label_sha256": early_stopping_label_sha256,
        }
        if model_exists and manifest_exists:
            checkpoint = _validate_seed_checkpoint_manifest(
                manifest_path, model_path, **validator_kwargs
            )
        else:
            checkpoint = fit_seed(
                pair,
                int(seed),
                fit_rows,
                fit_indices,
                validation_rows,
                validation_indices,
                model_path,
                n_jobs=n_jobs,
            )
            if (
                checkpoint.pair != pair
                or checkpoint.seed != int(seed)
                or Path(checkpoint.model_path) != model_path
                or checkpoint.sha256 != sha256_file(model_path)
                or isinstance(checkpoint.best_iteration, bool)
                or int(checkpoint.best_iteration) <= 0
            ):
                raise RefreshError(f"new seed checkpoint identity differs: {model_path}")
            model_path.chmod(0o444)
            manifest_value = {
                "schema": "m15-book-refresh-seed-checkpoint/v1",
                "prereg_id": prereg_id,
                "implementation_git_sha": implementation_git_sha,
                "pair": pair,
                "arm": arm,
                "attempt": attempt,
                "seed": int(seed),
                "seed_order_position": position,
                "feature_cols_sha256": sha256_bytes(
                    canonical_bytes(list(fit_rows.feature_cols))
                ),
                "model_parameters": dict(model_parameters_value),
                "fit_row_id_sha256": fit_row_id_sha256,
                "early_stopping_row_id_sha256": early_stopping_row_id_sha256,
                "fit_label_u8_sha256": fit_label_sha256,
                "early_stopping_label_u8_sha256": early_stopping_label_sha256,
                "model_filename": model_path.name,
                "model_sha256": checkpoint.sha256,
                "best_iteration": int(checkpoint.best_iteration),
            }
            atomic_json_new(manifest_path, manifest_value)
            checkpoint = _validate_seed_checkpoint_manifest(
                manifest_path, model_path, **validator_kwargs
            )
        checkpoints.append(checkpoint)
        manifests.append(manifest_path)
    return checkpoints, manifests


def _permuted_rows(
    rows: Any, positions: np.ndarray, rng: np.random.Generator
) -> tuple[Any, dict[str, Any]]:
    positions = np.asarray(positions, dtype="int64")
    input_labels = np.asarray(rows.y[positions], dtype="uint8")
    permutation = np.asarray(rng.permutation(len(positions)), dtype="int64")
    labels = rows.y.copy()
    labels[positions] = input_labels[permutation]
    output_labels = np.asarray(labels[positions], dtype="uint8")
    return (
        replace(
            rows,
            y=labels,
            sealed_control_label_permutation=True,
        ),
        {
            "row_count": len(positions),
            "ordered_row_id_sha256": _i64le_sha256(rows.fit_row_id[positions]),
            "input_label_u8_sha256": _u8_sha256(input_labels),
            "permutation_index_i64le_sha256": _i64le_sha256(permutation),
            "permuted_label_u8_sha256": _u8_sha256(output_labels),
        },
    )


def _assert_control_partition_identity(
    primary: Mapping[str, Any],
    control: Mapping[str, Any],
    *,
    pair: str,
    control_arm: str,
) -> dict[str, Any]:
    """Prove a negative control changed labels, not partitions or split state."""

    identity_fields = (
        "feature_cols",
        "split",
        "fit_row_count",
        "fit_row_id_sha256",
        "early_stopping_row_count",
        "early_stopping_row_id_sha256",
        "calibration_row_count",
        "calibration_row_id_sha256",
    )
    differences = [
        name for name in identity_fields if primary.get(name) != control.get(name)
    ]
    if differences:
        raise RefreshError(
            f"{pair} {control_arm}: control non-label partition state differs: {differences}"
        )
    permutation = control.get("control_permutation")
    partitions = permutation.get("partitions") if isinstance(permutation, dict) else None
    expected_partition_rows = {
        "capped_fit_rows": control.get("fit_row_count"),
        "early_stopping_rows": control.get("early_stopping_row_count"),
    }
    if not isinstance(permutation, dict) or permutation.get(
        "partition_rows"
    ) != expected_partition_rows:
        raise RefreshError(f"{pair} {control_arm}: control partition counts differ")
    expected = {
        "capped_fit_rows": {
            "row_count": control.get("fit_row_count"),
            "ordered_row_id_sha256": control.get("fit_row_id_sha256"),
            "input_label_u8_sha256": primary.get("fit_label_u8_sha256"),
            "permuted_label_u8_sha256": control.get("fit_label_u8_sha256"),
        },
        "early_stopping_rows": {
            "row_count": control.get("early_stopping_row_count"),
            "ordered_row_id_sha256": control.get("early_stopping_row_id_sha256"),
            "input_label_u8_sha256": primary.get("early_stopping_label_u8_sha256"),
            "permuted_label_u8_sha256": control.get(
                "early_stopping_label_u8_sha256"
            ),
        },
    }
    if not isinstance(partitions, dict) or set(partitions) != set(expected):
        raise RefreshError(f"{pair} {control_arm}: control permutation ledger differs")
    permutation_evidence: dict[str, Any] = {}
    for name, expected_values in expected.items():
        actual = partitions.get(name)
        if (
            not isinstance(actual, dict)
            or set(actual) != set(expected_values) | {"permutation_index_i64le_sha256"}
            or any(actual.get(field) != value for field, value in expected_values.items())
            or not _is_hex(actual.get("permutation_index_i64le_sha256"), 64)
        ):
            raise RefreshError(
                f"{pair} {control_arm}: {name} permutation identity differs"
            )
        permutation_evidence[name] = dict(actual)
    return {
        "base_arm": control_arm[0],
        "partition_identity_exact": True,
        "partition_fields": list(identity_fields),
        "calibration_gate_label_values": "none",
        "permutations": permutation_evidence,
    }


def _nominal_spill_audit(
    pair: str,
    nominal_years: Sequence[str],
    feature_dir: Path,
    rows: Any,
    *,
    entry_at_or_after: str,
    label_exit_before: str,
) -> dict[str, Any]:
    """Seal how filename-year spill is admitted only by exact UTC bounds."""

    from m15_book_refresh_adapters import window_mask

    exact_window = window_mask(
        rows,
        entry_at_or_after=entry_at_or_after,
        label_exit_before=label_exit_before,
    )
    source_spill_parts: list[np.ndarray] = []
    files: list[dict[str, Any]] = []
    for nominal_year in nominal_years:
        path = feature_dir / f"{pair}_{nominal_year}.parquet"
        try:
            projected = pd.read_parquet(path, columns=[])
        except Exception as exc:
            raise RefreshError(f"{pair}: cannot audit nominal source clock {path}: {exc}") from exc
        index = projected.index
        if not isinstance(index, pd.DatetimeIndex) or index.tz is None:
            raise RefreshError(f"{pair}: nominal source clock is not timezone-aware: {path}")
        utc = index.tz_convert("UTC")
        clock_ns = utc.to_numpy(dtype="datetime64[ns]").astype("int64", copy=False)
        if len(clock_ns) > 1 and np.any(np.diff(clock_ns) <= 0):
            raise RefreshError(f"{pair}: nominal source clock is not strictly ordered: {path}")
        decision_ns = clock_ns + np.int64(60 * 1_000_000_000)
        decision_year = pd.to_datetime(decision_ns, unit="ns", utc=True).year.to_numpy(
            dtype="int64"
        )
        spill_ns = np.asarray(
            decision_ns[decision_year != int(nominal_year)], dtype="int64"
        )
        source_spill_parts.append(spill_ns)
        files.append(
            {
                "nominal_year": nominal_year,
                "source_rows": len(clock_ns),
                "source_clock_sha256": sha256_bytes(
                    np.asarray(clock_ns, dtype="<i8").tobytes(order="C")
                ),
                "source_label_shift_to_decision_s": 60,
                "filename_year_spill_rows": len(spill_ns),
                "filename_year_spill_clock_sha256": sha256_bytes(
                    np.asarray(spill_ns, dtype="<i8").tobytes(order="C")
                ),
            }
        )
    all_spill = (
        np.concatenate(source_spill_parts)
        if source_spill_parts
        else np.empty(0, dtype="int64")
    )
    unique_spill = np.unique(all_spill)
    adapter_spill = np.isin(rows.fit_row_id, unique_spill, assume_unique=False)
    admitted_spill = adapter_spill & exact_window
    return {
        "schema": "m15-book-refresh-nominal-spill-audit/v1",
        "pair": pair,
        "nominal_years": list(nominal_years),
        "split_authority": "exact_utc_entry_and_label_exit_only",
        "entry_at_or_after": entry_at_or_after,
        "label_exit_before": label_exit_before,
        "files": files,
        "source_spill_occurrences": len(all_spill),
        "source_spill_unique_timestamps": len(unique_spill),
        "adapter_spill_rows": int(adapter_spill.sum()),
        "admitted_exact_window_spill_rows": int(admitted_spill.sum()),
        "excluded_exact_window_spill_rows": int((adapter_spill & ~exact_window).sum()),
        "admitted_early_stopping_spill_rows": int(
            (admitted_spill & rows.early_stopping_mask).sum()
        ),
        "admitted_calibration_spill_rows": int(
            (admitted_spill & rows.calibration_mask).sum()
        ),
        "admitted_structural_scope_spill_rows": int(admitted_spill.sum()),
    }


def fit_one(prereg_id: str, pair: str, arm: str, attempt: str) -> Path:
    """Fit one primary/control arm in one fresh deterministic process."""

    from m15_book_refresh_adapters import (
        build_calibration_rows,
        build_fit_rows,
        calibrate_policy,
        calibration_indices,
        contract_for,
        early_stopping_indices,
        fitting_indices,
        model_parameters,
        predict_checkpoint_mean,
        window_mask,
    )

    spec = load_spec()
    pair = pair.upper()
    if pair not in spec["pairs"]:
        raise RefreshError(f"unsupported fit pair {pair}")
    if arm not in {"B", "C", "B_perm", "C_perm"}:
        raise RefreshError(f"unsupported fit arm {arm}")
    if attempt not in {"canonical", "audit"}:
        raise RefreshError("fit attempt must be canonical or audit")
    is_control = arm.endswith("_perm")
    if is_control and attempt != "canonical":
        raise RefreshError("negative controls have exactly one canonical attempt")
    if is_control:
        representative = spec["control_family"]["representatives"].get(spec["pairs"][pair]["adapter_family"])
        if representative != pair:
            raise RefreshError(f"{pair} is not the sealed control representative for its family")
    workspace, store, prereg = workspace_for(prereg_id)
    store.load("adapter_parity")
    _verify_feature_view_routes(workspace)
    destination = _fit_result_path(workspace, pair, arm, attempt)
    if destination.exists():
        _validate_sealed_fit_result(
            workspace, destination, prereg_id=prereg_id, pair=pair, arm=arm, attempt=attempt
        )
        return destination
    attempt_dir = destination.parent
    if attempt_dir.is_symlink() or (attempt_dir.exists() and not attempt_dir.is_dir()):
        raise RefreshError(f"fit attempt directory is not a regular directory: {attempt_dir}")
    attempt_dir.mkdir(parents=True, exist_ok=True)
    views = workspace / "feature_views"
    feature_dir = views / "fit_features"
    orderflow_dir = views / "fit_features_of"
    base_arm = arm[0]
    split = spec["splits"][base_arm]
    lifecycle_status = (
        "inactive_shadow_candidate"
        if base_arm == "C" and not is_control
        else "workspace_only_nonpublishable"
    )
    book_id = (
        spec["pairs"][pair]["candidate_book_id"]
        if base_arm == "C" and not is_control
        else f"workspace:{pair}:{arm}"
    )
    fit_years = [str(year) for year in range(2012, 2022 if base_arm == "B" else 2026)]
    # Nominal file names are never split authority.  Include the adjacent
    # boundary file so rows whose UTC timestamp spills across its filename
    # year can be admitted (or excluded) only by the exact split masks below.
    validation_years = (
        ["2021", "2022", "2023"] if base_arm == "B" else ["2025", "2026"]
    )
    _verify_preregistered_inventory(
        prereg,
        "historical_feature_files",
        _fit_historical_files(
            pair,
            [
                year
                for year in dict.fromkeys([*fit_years, *validation_years])
                if int(year) <= 2025
            ],
        ),
    )
    fit_consumers = _feature_view_consumer_paths(
        workspace,
        pair,
        fit_years,
        feature_dir=feature_dir,
        orderflow_dir=orderflow_dir,
    )
    with verified_feature_view_access(workspace, fit_consumers):
        fit_rows = build_fit_rows(
            pair, fit_years, feature_dir=feature_dir, orderflow_dir=orderflow_dir
        )
    validation_consumers = _feature_view_consumer_paths(
        workspace,
        pair,
        validation_years,
        feature_dir=feature_dir,
        orderflow_dir=orderflow_dir,
    )
    with verified_feature_view_access(workspace, validation_consumers):
        validation_rows = build_calibration_rows(
            pair, validation_years, feature_dir=feature_dir, orderflow_dir=orderflow_dir
        )
    if fit_rows.feature_cols != validation_rows.feature_cols:
        raise RefreshError(f"{pair} {arm}: fit and validation feature order differs")
    fit_idx = fitting_indices(
        fit_rows,
        entry_at_or_after=spec["splits"]["entry_floor"],
        label_exit_before=split["fit_label_exit_before"],
        cap=spec["sealed_constants"]["train_row_cap"],
    )
    validation_idx = early_stopping_indices(
        validation_rows,
        entry_at_or_after=split["calibration_entry_at_or_after"],
        label_exit_before=split["calibration_label_exit_before"],
    )
    calibration_idx = calibration_indices(
        validation_rows,
        entry_at_or_after=split["calibration_entry_at_or_after"],
        label_exit_before=split["calibration_label_exit_before"],
    )
    structural_scope = window_mask(
        validation_rows,
        entry_at_or_after=split["calibration_entry_at_or_after"],
        label_exit_before=split["calibration_label_exit_before"],
    )
    spill_audit = _nominal_spill_audit(
        pair,
        validation_years,
        feature_dir,
        validation_rows,
        entry_at_or_after=split["calibration_entry_at_or_after"],
        label_exit_before=split["calibration_label_exit_before"],
    )
    if not len(fit_idx) or not len(validation_idx) or not len(calibration_idx):
        raise RefreshError(f"{pair} {arm}: empty fit/validation/calibration partition")
    control_seed: int | None = None
    control_permutation: dict[str, Any] | None = None
    if is_control:
        family = spec["pairs"][pair]["adapter_family"]
        control_seed = int(spec["sealed_constants"]["permutation_seeds"][family])
        permutation_contract = spec["control_family"]["label_permutation"]
        arm_code = int(permutation_contract["arm_codes"][arm])
        rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([control_seed, arm_code])))
        fit_rows, fit_permutation = _permuted_rows(fit_rows, fit_idx, rng)
        validation_rows, early_permutation = _permuted_rows(
            validation_rows, validation_idx, rng
        )
        control_permutation = {
            **permutation_contract,
            "family": family,
            "family_permutation_seed": control_seed,
            "arm": arm,
            "arm_code": arm_code,
            "resolved_seed_material": [control_seed, arm_code],
            "partition_rows": {
                "capped_fit_rows": len(fit_idx),
                "early_stopping_rows": len(validation_idx),
            },
            "partitions": {
                "capped_fit_rows": fit_permutation,
                "early_stopping_rows": early_permutation,
            },
        }
    partition_values = {
        "fit": np.asarray(fit_rows.fit_row_id[fit_idx], dtype="<i8"),
        "early_stopping": np.asarray(
            validation_rows.fit_row_id[validation_idx], dtype="<i8"
        ),
        "calibration": np.asarray(
            validation_rows.fit_row_id[calibration_idx], dtype="<i8"
        ),
    }
    partition_paths = {
        name: attempt_dir / f"{name}_row_id.i64le" for name in partition_values
    }
    for name, values in partition_values.items():
        _seal_bytes_exact(partition_paths[name], values.tobytes(order="C"))
    partition_hashes = {
        name: sha256_file(partition_paths[name]) for name in partition_values
    }
    label_hashes = {
        "fit": _u8_sha256(fit_rows.y[fit_idx]),
        "early_stopping": _u8_sha256(validation_rows.y[validation_idx]),
        "calibration": _u8_sha256(validation_rows.y[calibration_idx]),
    }
    contract = contract_for(pair)
    seeds = list(contract.seeds)
    parameters_by_seed = [
        model_parameters(
            pair,
            seed,
            n_jobs=int(spec["pairs"][pair]["model"]["num_threads"]),
        )
        for seed in seeds
    ]
    model_dir = attempt_dir / "models"
    checkpoints, checkpoint_manifests = _fit_or_resume_seed_checkpoints(
        workspace=workspace,
        prereg_id=prereg_id,
        implementation_git_sha=str(prereg["implementation_git_sha"]),
        pair=pair,
        arm=arm,
        attempt=attempt,
        fit_rows=fit_rows,
        fit_indices=fit_idx,
        validation_rows=validation_rows,
        validation_indices=validation_idx,
        model_dir=model_dir,
        seeds=seeds,
        parameters_by_seed=parameters_by_seed,
        n_jobs=int(spec["pairs"][pair]["model"]["num_threads"]),
        fit_row_id_sha256=partition_hashes["fit"],
        early_stopping_row_id_sha256=partition_hashes["early_stopping"],
        fit_label_sha256=label_hashes["fit"],
        early_stopping_label_sha256=label_hashes["early_stopping"],
    )
    calibration_probability = predict_checkpoint_mean(checkpoints, validation_rows.X)
    admitted = np.zeros(len(validation_rows.fit_row_id), dtype=bool)
    admitted[calibration_idx] = True
    calibration = calibrate_policy(
        validation_rows,
        calibration_probability,
        admitted,
        target_coverage=float(spec["pairs"][pair]["calibration_target_coverage"]),
        structural_scope_mask=structural_scope,
    )
    strategy_name = f"{pair}_{arm}_strategy.json"
    strategy_path = attempt_dir / strategy_name
    strategy = {
        "schema": "m15-book-refresh-strategy/v1",
        "book_id": book_id,
        "pair": pair,
        "arm": arm,
        "adapter_family": spec["pairs"][pair]["adapter_family"],
        "builder": spec["pairs"][pair]["builder"],
        "implementation_git_sha": prereg["implementation_git_sha"],
        "prereg_id": prereg_id,
        "lifecycle_status": lifecycle_status,
        "feature_cols": list(fit_rows.feature_cols),
        "n_features": len(fit_rows.feature_cols),
        "feature_dtype": "float32",
        "seed_order": seeds,
        "ensemble": "arithmetic_mean_probability_in_seed_order",
        "coverage": calibration.target_coverage,
        "conf_thr": calibration.confidence_threshold,
        "gate": {
            "confidence": "abs(p-0.5)>=conf_thr",
            "probability_tie": "p>=0.5_is_UP",
            "structural_column": calibration.structural_column,
            "structural_quantile": calibration.structural_quantile,
            "structural_threshold": calibration.structural_threshold,
        },
        "horizon_seconds": 900,
        "outer_evaluator_owned": ["ny_session", "last_start_16:35", "wc_ret", "nonoverlap_chrono"],
    }
    _seal_json_exact(strategy_path, strategy)
    best_iterations = [int(row.best_iteration) for row in checkpoints]
    resolved_without_id = {
        "schema": "m15-book-refresh-resolved-bundle-spec/v1",
        "book_id": book_id,
        "pair": pair,
        "arm": arm,
        "adapter_family": spec["pairs"][pair]["adapter_family"],
        "builder": spec["pairs"][pair]["builder"],
        "implementation_git_sha": prereg["implementation_git_sha"],
        "prereg_id": prereg_id,
        "feature_cols": list(fit_rows.feature_cols),
        "feature_dtype": "float32",
        "fit_ties": spec["pairs"][pair]["fit_ties"],
        "fit_session": spec["pairs"][pair]["fit_session"],
        "calibration_session": spec["pairs"][pair]["calibration_session"],
        "training_stride": spec["pairs"][pair]["training_stride"],
        "stride_phase": spec["pairs"][pair]["stride_phase"],
        "cap_selector": spec["cap_selector"],
        "fit_row_count": len(fit_idx),
        "fit_row_id_sha256": partition_hashes["fit"],
        "early_stopping_row_count": len(validation_idx),
        "early_stopping_row_id_sha256": partition_hashes["early_stopping"],
        "calibration_row_count": len(calibration_idx),
        "calibration_row_id_sha256": partition_hashes["calibration"],
        "fit_label_u8_sha256": label_hashes["fit"],
        "early_stopping_label_u8_sha256": label_hashes["early_stopping"],
        "calibration_label_u8_sha256": label_hashes["calibration"],
        "split": split,
        "model_parameters_by_seed": parameters_by_seed,
        "seed_order": seeds,
        "best_iterations": best_iterations,
        "early_stopping_rule": {
            "rounds": contract.early_stopping_rounds,
            "metric": "auc",
            "rows": len(validation_idx),
            "ties": spec["pairs"][pair]["early_stopping_ties"],
        },
        "nominal_source_years": {
            "fit": fit_years,
            "validation_and_calibration": validation_years,
        },
        "nominal_file_spill_audit": spill_audit,
        "calibration": asdict(calibration),
        "probability_tie_rule": "p>=0.5_is_UP",
        "control_permutation_seed": control_seed,
        "control_permutation": control_permutation,
        "lifecycle_status": lifecycle_status,
    }
    model_material = [(row.model_path.name, row.model_path.read_bytes()) for row in checkpoints]
    bundle_id = derive_bundle_id(
        spec,
        resolved_without_id,
        strategy_name,
        strategy_path.read_bytes(),
        model_material,
    )
    resolved = dict(resolved_without_id, bundle_id=bundle_id)
    resolved_path = attempt_dir / "resolved_bundle_spec.json"
    _seal_json_exact(resolved_path, resolved)
    result = {
        "schema": "m15-book-refresh-fit-result/v1",
        "prereg_id": prereg_id,
        "pair": pair,
        "arm": arm,
        "attempt": attempt,
        "bundle_id": bundle_id,
        "fit_row_count": len(fit_idx),
        "fit_row_id_sha256": partition_hashes["fit"],
        "early_stopping_row_count": len(validation_idx),
        "early_stopping_row_id_sha256": partition_hashes["early_stopping"],
        "calibration_row_count": len(calibration_idx),
        "calibration_row_id_sha256": partition_hashes["calibration"],
        "fit_label_u8_sha256": label_hashes["fit"],
        "early_stopping_label_u8_sha256": label_hashes["early_stopping"],
        "calibration_label_u8_sha256": label_hashes["calibration"],
        "split": split,
        "nominal_file_spill_audit": spill_audit,
        "best_iterations": best_iterations,
        "calibration": asdict(calibration),
        "feature_cols": list(fit_rows.feature_cols),
        "strategy": _workspace_relative(workspace, strategy_path),
        "resolved_bundle_spec": _workspace_relative(workspace, resolved_path),
        "models": [_workspace_relative(workspace, row.model_path) for row in checkpoints],
        "model_sha256": [row.sha256 for row in checkpoints],
        "seed_checkpoint_manifests": [
            _workspace_relative(workspace, path) for path in checkpoint_manifests
        ],
        "seed_checkpoint_manifest_sha256": [
            sha256_file(path) for path in checkpoint_manifests
        ],
        "control_permutation_seed": control_seed,
        "control_permutation": control_permutation,
    }
    _seal_json_exact(destination, result)
    for path in attempt_dir.rglob("*"):
        if path.is_file() and not path.is_symlink():
            path.chmod(0o444)
    _verify_feature_view_routes(workspace)
    print(
        f"[fit-one] {pair} {arm} {attempt} bundle={bundle_id} "
        f"fit={len(fit_idx)} cal={len(calibration_idx)}",
        flush=True,
    )
    return destination


def _run_fit_subprocess(prereg_id: str, pair: str, arm: str, attempt: str) -> None:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--fit-one",
        "--prereg-id", prereg_id,
        "--pair", pair,
        "--arm", arm,
        "--attempt", attempt,
    ]
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def _fit_artifact_inventory(
    workspace: Path, result_path: Path, result: Mapping[str, Any]
) -> list[Path]:
    """Return every artifact whose bytes are needed to authenticate a fit."""

    paths = [
        result_path,
        workspace / str(result.get("strategy", "")),
        workspace / str(result.get("resolved_bundle_spec", "")),
        result_path.parent / "fit_row_id.i64le",
        result_path.parent / "early_stopping_row_id.i64le",
        result_path.parent / "calibration_row_id.i64le",
        *[workspace / str(value) for value in result.get("models", [])],
        *[
            workspace / str(value)
            for value in result.get("seed_checkpoint_manifests", [])
        ],
    ]
    normalized = [_workspace_relative(workspace, path) for path in paths]
    if len(normalized) != len(set(normalized)):
        raise RefreshError(f"fit artifact inventory contains duplicates: {result_path}")
    return paths


def _validate_expected_fit_results(
    spec: Mapping[str, Any], workspace: Path, prereg_id: str
) -> None:
    for pair in spec["pair_order"]:
        for arm in ("B", "C"):
            for attempt in ("canonical", "audit"):
                _validate_sealed_fit_result(
                    workspace,
                    _fit_result_path(workspace, pair, arm, attempt),
                    prereg_id=prereg_id,
                    pair=pair,
                    arm=arm,
                    attempt=attempt,
                )
    for representative in spec["control_family"]["representatives"].values():
        for arm in ("B_perm", "C_perm"):
            _validate_sealed_fit_result(
                workspace,
                _fit_result_path(workspace, representative, arm, "canonical"),
                prereg_id=prereg_id,
                pair=representative,
                arm=arm,
                attempt="canonical",
            )


def run_all_fits(prereg_id: str) -> Path:
    spec = load_spec()
    workspace, store, _ = workspace_for(prereg_id)
    store.load("adapter_parity")
    _verify_feature_view_routes(workspace)
    summary_path = workspace / "fits" / "fit_repeat_summary.json"
    if store.has("fit_and_repeat"):
        store.load("fit_and_repeat")
        _validate_expected_fit_results(spec, workspace, prereg_id)
        return summary_path
    for pair in spec["pair_order"]:
        for arm in ("B", "C"):
            for attempt in ("canonical", "audit"):
                if not _fit_result_path(workspace, pair, arm, attempt).exists():
                    _run_fit_subprocess(prereg_id, pair, arm, attempt)
    for family, representative in spec["control_family"]["representatives"].items():
        if spec["pairs"][representative]["adapter_family"] != family:
            raise RefreshError("control representative family mismatch")
        for arm in ("B_perm", "C_perm"):
            if not _fit_result_path(workspace, representative, arm, "canonical").exists():
                _run_fit_subprocess(prereg_id, representative, arm, "canonical")
    report: dict[str, Any] = {
        "schema": "m15-book-refresh-fit-repeatability/v1",
        "prereg_id": prereg_id,
        "primary": {},
        "controls": {},
    }
    outputs: list[Path] = []
    for pair in spec["pair_order"]:
        report["primary"][pair] = {}
        for arm in ("B", "C"):
            canonical_path = _fit_result_path(workspace, pair, arm, "canonical")
            audit_path = _fit_result_path(workspace, pair, arm, "audit")
            canonical = _validate_sealed_fit_result(
                workspace, canonical_path, prereg_id=prereg_id, pair=pair, arm=arm,
                attempt="canonical",
            )
            audit = _validate_sealed_fit_result(
                workspace, audit_path, prereg_id=prereg_id, pair=pair, arm=arm,
                attempt="audit",
            )
            fields = (
                "bundle_id",
                "fit_row_count", "fit_row_id_sha256", "fit_label_u8_sha256",
                "early_stopping_row_count", "early_stopping_row_id_sha256",
                "early_stopping_label_u8_sha256", "calibration_row_count",
                "calibration_row_id_sha256", "calibration_label_u8_sha256",
                "split", "best_iterations", "calibration", "model_sha256",
                "feature_cols",
            )
            differences = [name for name in fields if canonical.get(name) != audit.get(name)]
            if differences:
                raise RefreshError(f"{pair} {arm}: clean repeat differs: {differences}")
            report["primary"][pair][arm] = {
                "bundle_id": canonical["bundle_id"],
                "repeat_bundle_id": audit["bundle_id"],
                "exact": True,
            }
            outputs.extend(
                _fit_artifact_inventory(workspace, canonical_path, canonical)
            )
            outputs.extend(_fit_artifact_inventory(workspace, audit_path, audit))
    for family, representative in spec["control_family"]["representatives"].items():
        report["controls"][family] = {}
        for arm in ("B_perm", "C_perm"):
            path = _fit_result_path(workspace, representative, arm, "canonical")
            result = _validate_sealed_fit_result(
                workspace, path, prereg_id=prereg_id, pair=representative, arm=arm,
                attempt="canonical",
            )
            base_arm = arm[0]
            primary = _validate_sealed_fit_result(
                workspace,
                _fit_result_path(
                    workspace, representative, base_arm, "canonical"
                ),
                prereg_id=prereg_id,
                pair=representative,
                arm=base_arm,
                attempt="canonical",
            )
            identity = _assert_control_partition_identity(
                primary,
                result,
                pair=representative,
                control_arm=arm,
            )
            report["controls"][family][arm] = {
                "pair": representative,
                "bundle_id": result["bundle_id"],
                **identity,
            }
            outputs.extend(_fit_artifact_inventory(workspace, path, result))
    atomic_json_new(summary_path, report)
    outputs.append(summary_path)
    deduplicated_outputs: list[Path] = []
    seen_outputs: set[str] = set()
    for output in outputs:
        key = _workspace_relative(workspace, output)
        if key not in seen_outputs:
            seen_outputs.add(key)
            deduplicated_outputs.append(output)
    store.seal(
        "fit_and_repeat",
        canonical_attempt_id=f"{prereg_id}:fit:0",
        input_paths=[workspace / "parity" / "parity.json"],
        output_paths=deduplicated_outputs,
        metadata={"primary_repeats_exact": True, "third_attempts": 0},
    )
    _verify_feature_view_routes(workspace)
    print(f"[fit] all canonical/audit arms sealed {summary_path}", flush=True)
    return summary_path


def _fit_artifact_paths(workspace: Path, result: Mapping[str, Any]) -> tuple[Path, list[Path]]:
    strategy = workspace / str(result["strategy"])
    models = [workspace / str(path) for path in result["models"]]
    if strategy.is_symlink() or not strategy.is_file() or any(p.is_symlink() or not p.is_file() for p in models):
        raise RefreshError("fit result references a missing/nonregular artifact")
    if [sha256_file(path) for path in models] != list(result["model_sha256"]):
        raise RefreshError("fit model artifact hash differs from sealed result")
    return strategy, models


def _structural_mask(
    feature_cols: Sequence[str], matrix: np.ndarray, calibration: Mapping[str, Any]
) -> np.ndarray:
    selected = np.ones(len(matrix), dtype=bool)
    column = calibration.get("structural_column")
    threshold = calibration.get("structural_threshold")
    if column is None and threshold is None:
        return selected
    if column is None or threshold is None or column not in feature_cols:
        raise RefreshError("malformed structural calibration")
    values = matrix[:, list(feature_cols).index(column)].astype("float64")
    return np.isfinite(values) & (values <= float(threshold))


def _blind_probability_record(
    entry_ns: np.ndarray,
    feature_cols: Sequence[str],
    matrix: np.ndarray,
    probabilities: np.ndarray,
    *,
    threshold: float,
    calibration: Mapping[str, Any],
    confidence_mask: np.ndarray | None = None,
    structural_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    if len(entry_ns) != len(matrix) or len(probabilities) != len(matrix):
        raise RefreshError("blind probability record is not row-aligned")
    scoreable = np.isfinite(matrix).all(axis=1) & np.isfinite(probabilities)
    if confidence_mask is None:
        confidence = scoreable & (
            np.abs(probabilities - 0.5) >= float(threshold)
        )
    else:
        confidence = np.asarray(confidence_mask, dtype=bool)
    if structural_mask is None:
        structural = scoreable & _structural_mask(
            feature_cols, matrix, calibration
        )
    else:
        structural = np.asarray(structural_mask, dtype=bool)
    if confidence.shape != scoreable.shape or structural.shape != scoreable.shape:
        raise RefreshError("blind probability gate masks are not row-aligned")
    if np.any(confidence & ~scoreable) or np.any(structural & ~scoreable):
        raise RefreshError("blind probability gate masks escaped scoreable rows")
    selected = confidence & structural
    selected_ns = np.asarray(entry_ns[selected], dtype="<i8")
    return {
        "rows": len(entry_ns),
        "scoreable_rows": int(scoreable.sum()),
        "selected_rows": int(selected.sum()),
        "entry_ns_sha256": sha256_bytes(np.asarray(entry_ns, dtype="<i8").tobytes()),
        "probability_f64le_sha256": sha256_bytes(
            np.asarray(probabilities, dtype="<f8").tobytes()
        ),
        "scoreable_mask_u8_sha256": sha256_bytes(
            np.asarray(scoreable, dtype="uint8").tobytes()
        ),
        "confidence_mask_u8_sha256": sha256_bytes(
            np.asarray(confidence, dtype="uint8").tobytes()
        ),
        "structural_mask_u8_sha256": sha256_bytes(
            np.asarray(structural, dtype="uint8").tobytes()
        ),
        "pre_schedule_mask_u8_sha256": sha256_bytes(
            np.asarray(selected, dtype="uint8").tobytes()
        ),
        "ordered_selected_entry_ns_sha256": sha256_bytes(selected_ns.tobytes()),
        "threshold": float(threshold),
        "structural_column": calibration.get("structural_column"),
        "structural_threshold": calibration.get("structural_threshold"),
    }


def _blind_score_window(rows: Any) -> tuple[np.ndarray, np.ndarray]:
    from m15_book_refresh_stats import canonical_runtime_ny_mask, epoch_nanoseconds_to_seconds

    spec = load_spec()
    split = spec["splits"]["replay"]
    lo = int(pd.Timestamp(split["entry_at_or_after"]).value)
    hi = int(pd.Timestamp(split["entry_before"]).value)
    # This outcome-blind seal is deliberately the entry/runtime superset.  The
    # replay later applies canonical settlement validity and the strict bound
    # on the actual exit observation.  A nominal-expiry proxy here could omit a
    # row whose canonical last observation is nevertheless before the bound.
    window = (rows.entry_ns >= lo) & (rows.entry_ns < hi)
    positions = np.flatnonzero(window)
    seconds = epoch_nanoseconds_to_seconds(rows.entry_ns[positions], name=f"{rows.pair} blind entry_ns")
    runtime = canonical_runtime_ny_mask(seconds)
    positions = positions[runtime]
    return positions, seconds[runtime]


def _assert_fit_repeat_identity(
    canonical: Mapping[str, Any],
    audit: Mapping[str, Any],
    *,
    pair: str,
    arm: str,
) -> None:
    if canonical.get("bundle_id") != audit.get("bundle_id"):
        raise RefreshError(f"{pair} {arm}: canonical/audit bundle IDs differ")
    if canonical.get("best_iterations") != audit.get("best_iterations"):
        raise RefreshError(f"{pair} {arm}: canonical/audit best iterations differ")


def seal_arms(prereg_id: str) -> str:
    """Verify full-resolution blind repeatability, then derive controls/run IDs."""

    from m15_book_refresh_adapters import build_score_rows
    import book_runtime

    spec = load_spec()
    workspace, store, prereg = workspace_for(prereg_id)
    _verify_preregistered_inventory(
        prereg, "comparator_A_files", comparator_a_files(), exact=True
    )
    store.load("fit_and_repeat")
    existing = sorted((workspace / "arm_seal").glob("run_binding_*.json")) \
        if (workspace / "arm_seal").exists() else []
    if store.has("arm_seal"):
        store.load("arm_seal")
        _verify_feature_view_routes(workspace)
        if len(existing) != 1:
            raise RefreshError("sealed arm phase does not have one run binding")
        return _read_json(existing[0])["run_id"]
    arm_dir = workspace / "arm_seal"
    if arm_dir.exists():
        _remove_owned_tree(arm_dir)
    arm_dir.mkdir()
    feature_dir = workspace / "feature_views" / "features"
    orderflow_dir = workspace / "feature_views" / "features_of"
    books = book_runtime.load_target_books(list(spec["pair_order"]))
    index = book_runtime.load_index()
    arm_entries: list[list[str]] = []
    repeatability: dict[str, Any] = {
        "schema": "m15-book-refresh-repeatability/v1",
        "prereg_id": prereg_id,
        "semantic_replay_outcomes_accessed": False,
        "pairs": {},
        "controls": {},
    }
    blind_inputs: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, Any]] = {}
    for pair in spec["pair_order"]:
        print(f"[arm-seal] blind score {pair}", flush=True)
        book = books[pair]
        score_consumers = _feature_view_consumer_paths(
            workspace,
            pair,
            ["2026"],
            feature_dir=feature_dir,
            orderflow_dir=orderflow_dir,
        )
        with verified_feature_view_access(workspace, score_consumers):
            score_rows = build_score_rows(
                pair, ["2026"], book.feature_cols,
                feature_dir=feature_dir, orderflow_dir=orderflow_dir,
            )
        positions, _ = _blind_score_window(score_rows)
        entry_ns = score_rows.entry_ns[positions]
        matrix = score_rows.X[positions]
        scoreable = np.isfinite(matrix).all(axis=1)
        blind_inputs[pair] = (entry_ns, matrix, scoreable, book)
        p_a = np.full(len(matrix), np.nan, dtype="float64")
        p_a[scoreable] = book.predict_probabilities(
            pd.DataFrame(
                matrix[scoreable].astype("float64"), columns=book.feature_cols
            )
        )
        a_confidence = np.zeros(len(matrix), dtype=bool)
        a_structural_mask = np.zeros(len(matrix), dtype=bool)
        a_components = book.gate_components(
            pd.DataFrame(
                matrix[scoreable].astype("float64"), columns=book.feature_cols
            ),
            p_a[scoreable],
        )
        a_confidence[scoreable] = a_components.confidence_passed
        a_structural_mask[scoreable] = a_components.structural_passed
        a_structural = _comparator_a_structural_gate(book)
        a_record = _blind_probability_record(
            entry_ns, book.feature_cols, matrix, p_a,
            threshold=book.conf_thr, calibration=a_structural,
            confidence_mask=a_confidence,
            structural_mask=a_structural_mask,
        )
        index_entry = index[book.book_id]
        manifest_path, _legacy_manifest = book_runtime.load_registered_manifest(
            book.book_id, index_entry
        )
        a_bundle = _comparator_a_bundle_id(
            spec, prereg, pair, book, index_entry, manifest_path
        )
        arm_entries.append([pair, "A", a_bundle])
        repeatability["pairs"][pair] = {
            "A": dict(a_record, bundle_id=a_bundle, book_id=book.book_id),
        }
        for arm in ("B", "C"):
            canonical = _validate_sealed_fit_result(
                workspace, _fit_result_path(workspace, pair, arm, "canonical"),
                prereg_id=prereg_id, pair=pair, arm=arm, attempt="canonical",
            )
            audit = _validate_sealed_fit_result(
                workspace, _fit_result_path(workspace, pair, arm, "audit"),
                prereg_id=prereg_id, pair=pair, arm=arm, attempt="audit",
            )
            _assert_fit_repeat_identity(
                canonical, audit, pair=pair, arm=arm
            )
            canonical_resolved = _read_json(
                workspace / str(canonical["resolved_bundle_spec"])
            )
            audit_resolved = _read_json(
                workspace / str(audit["resolved_bundle_spec"])
            )
            if not (
                tuple(canonical_resolved["feature_cols"]) == tuple(book.feature_cols)
                and tuple(audit_resolved["feature_cols"]) == tuple(book.feature_cols)
            ):
                raise RefreshError(
                    f"{pair} {arm}: fit/audit feature order differs from the canonical A score matrix"
                )
            canonical_strategy, canonical_models = _fit_artifact_paths(workspace, canonical)
            audit_strategy, audit_models = _fit_artifact_paths(workspace, audit)
            if canonical_strategy.read_bytes() != audit_strategy.read_bytes():
                raise RefreshError(f"{pair} {arm}: repeat strategy bytes differ")
            p_canonical = np.full(len(matrix), np.nan, dtype="float64")
            p_audit = np.full(len(matrix), np.nan, dtype="float64")
            p_canonical[scoreable] = _predict_boosters(
                [__import__("lightgbm").Booster(model_file=str(path)) for path in canonical_models],
                matrix[scoreable],
            )
            p_audit[scoreable] = _predict_boosters(
                [__import__("lightgbm").Booster(model_file=str(path)) for path in audit_models],
                matrix[scoreable],
            )
            if not np.array_equal(p_canonical[scoreable].view("uint64"), p_audit[scoreable].view("uint64")):
                raise RefreshError(f"{pair} {arm}: full-resolution repeat probabilities differ")
            calibration = canonical_resolved["calibration"]
            record = _blind_probability_record(
                entry_ns, book.feature_cols, matrix, p_canonical,
                threshold=float(calibration["confidence_threshold"]),
                calibration=calibration,
            )
            audit_record = _blind_probability_record(
                entry_ns, book.feature_cols, matrix, p_audit,
                threshold=float(
                    audit_resolved["calibration"]["confidence_threshold"]
                ),
                calibration=audit_resolved["calibration"],
            )
            if record != audit_record:
                raise RefreshError(f"{pair} {arm}: blind policy repeatability differs")
            arm_entries.append([pair, arm, canonical["bundle_id"]])
            repeatability["pairs"][pair][arm] = dict(
                record,
                bundle_id=canonical["bundle_id"],
                repeat_bundle_id=audit["bundle_id"],
                best_iterations=canonical["best_iterations"],
                repeat_best_iterations=audit["best_iterations"],
                exact=True,
            )
    controls_entries: list[list[str]] = []
    for family, arm in spec["control_order"]:
        representative = spec["control_family"]["representatives"][family]
        result = _validate_sealed_fit_result(
            workspace, _fit_result_path(workspace, representative, arm, "canonical"),
            prereg_id=prereg_id, pair=representative, arm=arm, attempt="canonical",
        )
        primary = _validate_sealed_fit_result(
            workspace,
            _fit_result_path(workspace, representative, arm[0], "canonical"),
            prereg_id=prereg_id,
            pair=representative,
            arm=arm[0],
            attempt="canonical",
        )
        partition_identity = _assert_control_partition_identity(
            primary,
            result,
            pair=representative,
            control_arm=arm,
        )
        resolved = _read_json(workspace / str(result["resolved_bundle_spec"]))
        entry_ns, matrix, scoreable, book = blind_inputs[representative]
        if tuple(resolved["feature_cols"]) != tuple(book.feature_cols):
            raise RefreshError(
                f"{family} {arm}: control feature order differs from representative A"
            )
        _, control_models = _fit_artifact_paths(workspace, result)
        probabilities = np.full(len(matrix), np.nan, dtype="float64")
        probabilities[scoreable] = _predict_boosters(
            [__import__("lightgbm").Booster(model_file=str(path)) for path in control_models],
            matrix[scoreable],
        )
        calibration = resolved["calibration"]
        record = _blind_probability_record(
            entry_ns,
            book.feature_cols,
            matrix,
            probabilities,
            threshold=float(calibration["confidence_threshold"]),
            calibration=calibration,
        )
        repeatability["controls"].setdefault(family, {})[arm] = dict(
            record,
            pair=representative,
            bundle_id=result["bundle_id"],
            feature_order_exact=True,
            best_iterations=result["best_iterations"],
            partition_identity=partition_identity,
        )
        controls_entries.append([family, arm, result["bundle_id"]])
    cid = derive_controls_id(spec, controls_entries)
    rid = derive_run_id(spec, prereg_id, cid, arm_entries)
    predecessor_hashes = {
        phase: sha256_file(store.path(phase))
        for phase in PHASE_ORDER[: PHASE_ORDER.index("arm_seal")]
    }
    binding = {
        "schema": "m15-book-refresh-run-binding/v1",
        "run_id": rid,
        "prereg_id": prereg_id,
        "implementation_git_sha": prereg["implementation_git_sha"],
        "evaluator_id": evaluator_id(spec),
        "controls_id": cid,
        "control_arms": controls_entries,
        "primary_arms": arm_entries,
        "canonical_encoding": spec["identity"],
        "predecessor_manifest_hashes": predecessor_hashes,
        "semantic_replay_outcomes_accessed": False,
    }
    binding_path = arm_dir / f"run_binding_{rid}.json"
    repeat_path = arm_dir / f"repeatability_{rid}.json"
    atomic_json_new(binding_path, binding)
    repeatability.update(run_id=rid, controls_id=cid)
    atomic_json_new(repeat_path, repeatability)
    tracked_binding = RESULTS_JSON / f"m15_book_refresh_{rid}_run_binding_result.json"
    tracked_repeat = RESULTS_JSON / f"m15_book_refresh_{rid}_repeatability_result.json"
    _seal_json_exact(tracked_binding, binding)
    _seal_json_exact(tracked_repeat, repeatability)
    _verify_feature_view_routes(workspace)
    store.seal(
        "arm_seal",
        canonical_attempt_id=f"{prereg_id}:arm-seal:0",
        input_paths=[workspace / "fits" / "fit_repeat_summary.json"],
        output_paths=[binding_path, repeat_path, tracked_binding, tracked_repeat],
        metadata={
            "run_id": rid,
            "controls_id": cid,
            "semantic_replay_outcomes_accessed": False,
            "tracked_results": [repo_relative(tracked_binding), repo_relative(tracked_repeat)],
        },
    )
    print(f"[arm-seal] run_id={rid} controls_id={cid}", flush=True)
    return rid


def _run_binding(workspace: Path, store: PhaseStore) -> dict[str, Any]:
    phase = store.load("arm_seal")
    run_id = phase.get("metadata", {}).get("run_id")
    if not _is_hex(run_id, 64):
        raise RefreshError("arm-seal phase has no valid run_id")
    path = workspace / "arm_seal" / f"run_binding_{run_id}.json"
    binding = _read_json(path)
    if binding.get("run_id") != run_id:
        raise RefreshError("run-binding identity differs from arm-seal phase")
    return binding


def _seal_json_exact(path: Path, value: Mapping[str, Any]) -> Path:
    """Create canonical read-only evidence, or verify an identical partial write."""

    encoded = canonical_bytes(dict(value))
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file() or path.stat().st_mode & 0o222:
            raise RefreshError(f"existing evidence is not a read-only regular file: {path}")
        if path.read_bytes() != encoded:
            raise RefreshError(f"existing evidence differs from reconstructed bytes: {path}")
        return path
    return atomic_write_new(path, encoded, read_only=True)


def _seal_replay_start_marker(
    workspace: Path,
    prereg: Mapping[str, Any],
    binding: Mapping[str, Any],
) -> Path:
    """Durably spend the one semantic replay look before any outcome read.

    A marker without a sealed replay phase is deliberately non-resumable.  The
    runner cannot distinguish a crash before the first close read from a crash
    after it, so either state must quarantine the workspace rather than reread.
    """

    run_id = binding.get("run_id")
    if not _is_hex(run_id, 64):
        raise RefreshError("replay-start marker requires a valid run_id")
    replay_dir = workspace / "replay"
    if workspace.is_symlink() or not workspace.is_dir():
        raise RefreshError("replay workspace is missing or symlinked")
    try:
        workspace_root = workspace.resolve(strict=True)
    except OSError as exc:
        raise RefreshError("cannot resolve replay workspace") from exc
    if replay_dir.is_symlink() or (replay_dir.exists() and not replay_dir.is_dir()):
        raise RefreshError("replay output path is not a regular directory")
    replay_dir.mkdir(parents=True, exist_ok=True)
    try:
        replay_root = replay_dir.resolve(strict=True)
    except OSError as exc:
        raise RefreshError("cannot resolve replay output directory") from exc
    if replay_dir.is_symlink() or replay_root.parent != workspace_root:
        raise RefreshError("replay output directory escapes the exact workspace")
    marker = replay_dir / f"replay_started_{run_id}.json"
    children = sorted(replay_dir.iterdir(), key=lambda path: path.name)
    if marker.exists() or marker.is_symlink():
        raise RefreshError(
            "unsealed replay-start marker exists; semantic replay is quarantined "
            "and outcomes must not be reread"
        )
    if children:
        raise RefreshError(
            "unknown partial replay output exists; semantic replay is quarantined"
        )
    binding_path = workspace / "arm_seal" / f"run_binding_{run_id}.json"
    arm_manifest = workspace / "phases" / "04_arm_seal.json"
    payload = {
        "schema": "m15-book-refresh-replay-start/v1",
        "status": "semantic_look_irrevocably_started",
        "prereg_id": prereg.get("prereg_id"),
        "run_id": run_id,
        "implementation_git_sha": prereg.get("implementation_git_sha"),
        "evaluator_id": binding.get("evaluator_id"),
        "run_binding_sha256": sha256_file(binding_path),
        "arm_seal_manifest_sha256": sha256_file(arm_manifest),
        "semantic_replay_outcomes_may_have_been_accessed": True,
        "resume_policy": "fail_closed_no_reread_after_any_unsealed_start",
    }
    atomic_json_new(marker, payload)
    return marker


def run_replay(prereg_id: str) -> Path:
    """Run and seal the one measured common replay after the arm seal."""

    import book_runtime
    from m15_book_refresh_replay import run_workspace_replay

    workspace, store, prereg = workspace_for(prereg_id)
    binding = _run_binding(workspace, store)
    arm_phase = store.load("arm_seal")
    arm_seal_output_hashes = arm_phase.get("output_hashes")
    if not isinstance(arm_seal_output_hashes, dict):
        raise RefreshError("arm-seal phase output hashes are malformed")
    run_id = binding["run_id"]
    replay_dir = workspace / "replay"
    core_path = replay_dir / f"replay_core_{run_id}.json"
    if store.has("replay"):
        _authenticate_current_comparator_a(
            prereg, binding, allow_index_additions=True
        )
        store.load("replay")
        _authenticate_current_comparator_a(
            prereg, binding, allow_index_additions=True
        )
        return core_path

    def authenticated_fit_loader(root: Path, pair: str, arm: str) -> Mapping[str, Any]:
        return _validate_sealed_fit_result(
            root,
            _fit_result_path(root, pair, arm, "canonical"),
            prereg_id=prereg_id,
            pair=pair,
            arm=arm,
            attempt="canonical",
        )

    # Load A once, then authenticate the exact paths and bytes represented by
    # those loaded objects against both preregistration and the arm binding.
    spec = load_spec()
    books = book_runtime.load_target_books(list(spec["pair_order"]))
    index = book_runtime.load_index()
    _verify_loaded_comparator_binding(prereg, binding, books, index)
    _verify_all_processed_source_identities(prereg)
    marker_path = _seal_replay_start_marker(workspace, prereg, binding)
    # Authenticate the exact evidence again after the durable one-look marker;
    # the replay reader also hashes the bytes it parses against this map.
    store.load("arm_seal")
    result = run_workspace_replay(
        spec,
        workspace,
        books,
        prereg_id=prereg_id,
        run_id=run_id,
        arm_seal_output_hashes=arm_seal_output_hashes,
        fit_result_loader=authenticated_fit_loader,
    )
    # A persistent or in-place comparator mutation during scoring invalidates
    # the one look.  The durable marker remains, so a failure cannot be retried.
    _verify_loaded_comparator_binding(prereg, binding, books, index)
    _verify_all_processed_source_identities(prereg)
    store.load("arm_seal")
    payload = result.as_dict()
    atomic_json_new(core_path, payload)
    store.seal(
        "replay",
        canonical_attempt_id=f"{prereg_id}:replay:0",
        input_paths=[
            workspace / "arm_seal" / f"run_binding_{run_id}.json",
            workspace / "arm_seal" / f"repeatability_{run_id}.json",
        ],
        output_paths=[marker_path, core_path],
        metadata={
            "run_id": run_id,
            "semantic_replay_outcomes_accessed": True,
            "single_analysis_look": True,
        },
    )
    print(f"[replay] sealed one analysis look run_id={run_id}", flush=True)
    return core_path


def _load_replay_campaign(path: Path) -> Any:
    from m15_book_refresh_replay import ReplayCampaignResult

    payload = _read_json(path)
    if payload.get("schema") != "m15-book-refresh-replay-campaign/v1":
        raise RefreshError("replay campaign schema mismatch")
    pair_results = payload.get("pair_results")
    joint = payload.get("joint_result")
    if not isinstance(pair_results, dict) or not isinstance(joint, dict):
        raise RefreshError("replay campaign result is malformed")
    return ReplayCampaignResult(pair_results=pair_results, joint_result=joint)


def _partial_stage_for(book_id: str) -> list[Path]:
    return sorted((REPO_ROOT / "books").glob(f".{book_id}.stage-*"))


def _clear_owned_partial_stage(
    book_id: str, *, prereg_id: str, run_id: str, bundle_id: str
) -> None:
    stages = _partial_stage_for(book_id)
    if len(stages) > 1:
        raise RefreshError(f"{book_id}: multiple candidate stages exist")
    if not stages:
        return
    stage = stages[0]
    lock = manifest.candidate_publish_lock(stage)
    expected = {
        "book_id": book_id,
        "prereg_id": prereg_id,
        "run_id": run_id,
        "bundle_id": bundle_id,
    }
    if any(lock.get(name) != value for name, value in expected.items()):
        raise RefreshError(f"{book_id}: unknown or conflicting partial stage; refusing deletion")
    _remove_owned_tree(stage)


def _candidate_manifest_and_stage(
    spec: Mapping[str, Any],
    workspace: Path,
    prereg: Mapping[str, Any],
    run_id: str,
    pair: str,
    pair_result: Mapping[str, Any],
) -> dict[str, Any]:
    fit_result = _validate_sealed_fit_result(
        workspace,
        _fit_result_path(workspace, pair, "C", "canonical"),
        prereg_id=str(prereg["prereg_id"]),
        pair=pair,
        arm="C",
        attempt="canonical",
    )
    strategy, models = _fit_artifact_paths(workspace, fit_result)
    resolved_path = workspace / str(fit_result["resolved_bundle_spec"])
    resolved = _read_json(resolved_path)
    book_id = str(spec["pairs"][pair]["candidate_book_id"])
    bundle_id = str(fit_result["bundle_id"])
    if (
        resolved.get("book_id") != book_id
        or resolved.get("bundle_id") != bundle_id
        or resolved.get("prereg_id") != prereg["prereg_id"]
        or resolved.get("lifecycle_status") != "inactive_shadow_candidate"
        or "run_id" in resolved
    ):
        raise RefreshError(f"{pair}: candidate resolved bundle identity/lifecycle differs")
    _clear_owned_partial_stage(
        book_id,
        prereg_id=str(prereg["prereg_id"]),
        run_id=run_id,
        bundle_id=bundle_id,
    )
    data_lock = _read_json(workspace / "feature_snapshot" / "data_lock.json")
    manifest_value = manifest.build_candidate_manifest(
        book_id,
        repo_root=REPO_ROOT,
        currency=pair,
        timeframe="15m",
        side="combined",
        role="direction",
        lifecycle_status="inactive_shadow_candidate",
        bundle_id=bundle_id,
        prereg_id=str(prereg["prereg_id"]),
        run_id=run_id,
        implementation_git_sha=str(prereg["implementation_git_sha"]),
        source_script="scripts/m15_book_refresh.py",
        determinism={
            "seed_order": list(spec["pairs"][pair]["seed_order"]),
            "num_threads": int(spec["pairs"][pair]["model"]["num_threads"]),
            "deterministic": True,
            "force_col_wise": True,
            "canonical_attempt": "first_completed",
        },
        artifacts=[strategy, *models],
        strategy_json=strategy.name,
        summary=(
            f"{pair} 15m date-refresh retrospective survivor; immutable inactive shadow "
            "candidate only, with no activation or certification"
        ),
        metrics={
            "retrospective_status": pair_result["terminal_status"],
            "replay_C": pair_result["arms"]["C"]["periods"]["combined"],
            "breakeven": float(spec["sealed_constants"]["breakeven"]),
            "retrospective_not_certification": True,
        },
        hyperparams={
            "model_parameters_by_seed": resolved["model_parameters_by_seed"],
            "best_iterations": resolved["best_iterations"],
            "calibration": resolved["calibration"],
            "cap_selector": resolved["cap_selector"],
        },
        feature_fingerprint={
            "data_lock_sha256": sha256_file(workspace / "feature_snapshot" / "data_lock.json"),
            "base_snapshot_sha256": data_lock["base"][pair]["sha256"],
            "orderflow_snapshot_sha256": (
                data_lock["orderflow"]["EURUSD"]["sha256"] if pair == "EURUSD" else None
            ),
        },
        depends_on=spec["pairs"][pair]["book_id_A"],
        notes={
            "issue": 9,
            "activation": False,
            "prospective_shadow_and_refit_CPCV_required": True,
        },
        created_utc=None,
    )
    return manifest.stage_candidate(
        manifest_value,
        resolved,
        [strategy, *models],
        repo_root=REPO_ROOT,
    )


def lock_results_and_candidates(prereg_id: str) -> Path:
    """Seal statuses, candidate receipts, and the external publication trust anchor."""

    from m15_book_refresh_replay import finalize_joint_result

    spec = load_spec()
    workspace, store, prereg = workspace_for(prereg_id)
    binding = _run_binding(workspace, store)
    _authenticate_current_comparator_a(
        prereg, binding, allow_index_additions=True
    )
    run_id = binding["run_id"]
    store.load("replay")
    tracked_joint = RESULTS_JSON / f"m15_book_refresh_{run_id}_joint_replay_result.json"
    if store.has("status_and_publication_lock"):
        store.load("status_and_publication_lock")
        _authenticate_current_comparator_a(
            prereg, binding, allow_index_additions=True
        )
        return tracked_joint
    _authenticate_current_comparator_a(
        prereg, binding, allow_index_additions=False
    )
    core = _load_replay_campaign(workspace / "replay" / f"replay_core_{run_id}.json")
    survivor_ids = list(core.joint_result.get("S", []))
    candidate_statuses = core.joint_result.get("candidate_statuses", {})
    if not isinstance(candidate_statuses, dict):
        raise RefreshError("draft replay candidate statuses are malformed")
    locks: dict[str, dict[str, Any]] = {}
    pair_by_candidate = {
        str(spec["pairs"][pair]["candidate_book_id"]): pair for pair in spec["pair_order"]
    }
    for book_id in survivor_ids:
        pair = pair_by_candidate.get(book_id)
        if pair is None or candidate_statuses.get(book_id, {}).get("status") != "PROMOTE_TO_SHADOW":
            raise RefreshError(f"{book_id}: survivor status is not publication-authorized")
        staged = _candidate_manifest_and_stage(
            spec, workspace, prereg, run_id, pair, core.pair_results[pair]
        )
        locks[book_id] = staged["publish_lock"]
    finalized = finalize_joint_result(core, candidate_publish_locks=locks)
    _authenticate_current_comparator_a(
        prereg, binding, allow_index_additions=False
    )
    status_dir = workspace / "status"
    status_dir_existed = status_dir.exists() or status_dir.is_symlink()
    status_dir.mkdir(parents=True, exist_ok=True)
    rollback_records: list[tuple[Path, bytes]] = []
    workspace_joint = status_dir / f"joint_replay_{run_id}.json"
    workspace_joint_existed = workspace_joint.exists() or workspace_joint.is_symlink()
    tracked_joint_existed = tracked_joint.exists() or tracked_joint.is_symlink()
    _seal_json_exact(workspace_joint, finalized.joint_result)
    _seal_json_exact(tracked_joint, finalized.joint_result)
    joint_bytes = canonical_bytes(dict(finalized.joint_result))
    _record_new_exact_file(
        rollback_records,
        workspace_joint,
        existed_before=workspace_joint_existed,
        expected=joint_bytes,
    )
    _record_new_exact_file(
        rollback_records,
        tracked_joint,
        existed_before=tracked_joint_existed,
        expected=joint_bytes,
    )
    outputs = [workspace_joint, tracked_joint]
    for pair in spec["pair_order"]:
        pair_value = dict(finalized.pair_results[pair])
        workspace_pair = status_dir / f"{pair}_replay_{run_id}.json"
        tracked_pair = RESULTS_JSON / f"m15_book_refresh_{run_id}_{pair}_replay_result.json"
        workspace_pair_existed = workspace_pair.exists() or workspace_pair.is_symlink()
        tracked_pair_existed = tracked_pair.exists() or tracked_pair.is_symlink()
        _seal_json_exact(workspace_pair, pair_value)
        _seal_json_exact(tracked_pair, pair_value)
        pair_bytes = canonical_bytes(pair_value)
        _record_new_exact_file(
            rollback_records,
            workspace_pair,
            existed_before=workspace_pair_existed,
            expected=pair_bytes,
        )
        _record_new_exact_file(
            rollback_records,
            tracked_pair,
            existed_before=tracked_pair_existed,
            expected=pair_bytes,
        )
        outputs.extend([workspace_pair, tracked_pair])
    status_phase_path = store.path("status_and_publication_lock")
    status_phase_existed = (
        status_phase_path.exists() or status_phase_path.is_symlink()
    )
    status_phase = store.seal(
        "status_and_publication_lock",
        canonical_attempt_id=f"{prereg_id}:status-lock:0",
        input_paths=[workspace / "replay" / f"replay_core_{run_id}.json"],
        output_paths=outputs,
        metadata={
            "run_id": run_id,
            "S": survivor_ids,
            "tracked_joint": repo_relative(tracked_joint),
            "publication_lock_complete": True,
        },
    )
    _record_new_exact_file(
        rollback_records,
        status_phase_path,
        existed_before=status_phase_existed,
        expected=canonical_bytes(status_phase),
    )
    try:
        _authenticate_current_comparator_a(
            prereg, binding, allow_index_additions=False
        )
    except Exception:
        _rollback_new_exact_files(
            rollback_records,
            new_directories=(() if status_dir_existed else (status_dir,)),
        )
        raise
    print(f"[status-lock] run_id={run_id} survivors={len(survivor_ids)}", flush=True)
    return tracked_joint


def _registry_entry_for_candidate(book_id: str, lock: Mapping[str, Any]) -> str:
    return (
        f"- **`{book_id}` — issue #9 retrospective date-refresh survivor.** "
        f"bundle_id={lock['bundle_id']}; prereg_id={lock['prereg_id']}; "
        f"run_id={lock['run_id']}; lifecycle_status=inactive_shadow_candidate; "
        "inactive; no activation. Prospective shadow and fresh refit-CPCV certification remain required."
    )


def publish_candidates(prereg_id: str) -> None:
    """Publish only externally locked S through authenticated, recoverable transitions."""

    workspace, store, prereg = workspace_for(prereg_id)
    binding = _run_binding(workspace, store)
    _authenticate_current_comparator_a(
        prereg, binding, allow_index_additions=True
    )
    run_id = binding["run_id"]
    store.load("status_and_publication_lock")
    joint_path = RESULTS_JSON / f"m15_book_refresh_{run_id}_joint_replay_result.json"
    joint = _read_json(joint_path)
    survivors = list(joint.get("S", []))
    if store.has("candidate_publication"):
        store.load("candidate_publication")
        _authenticate_current_comparator_a(
            prereg, binding, allow_index_additions=True
        )
        return
    publication_dir = workspace / "candidate_publication"
    publication_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for book_id in survivors:
        lock = joint.get("candidate_publish_locks", {}).get(book_id)
        if not isinstance(lock, dict):
            raise RefreshError(f"{book_id}: publication lock missing from joint evidence")
        phase_path = publication_dir / f"{book_id}.json"
        phase_value = {
            "schema": "m15-book-refresh-candidate-publication/v1",
            "status": "sealed",
            "book_id": book_id,
            "bundle_id": lock["bundle_id"],
            "prereg_id": prereg_id,
            "run_id": run_id,
            "lifecycle_status": "inactive_shadow_candidate",
        }

        def comparator_commit_guard() -> None:
            _authenticate_current_comparator_a(
                prereg, binding, allow_index_additions=True
            )

        registry_entry = _registry_entry_for_candidate(book_id, lock)
        destination = REPO_ROOT / "books" / book_id
        common = {
            "repo_root": REPO_ROOT,
            "registry_path": REPO_ROOT / "MODEL_REGISTRY.md",
            "registry_entry": registry_entry,
            "phase_seal_path": phase_path,
            "phase_seal": phase_value,
            "commit_guard": comparator_commit_guard,
        }
        _authenticate_current_comparator_a(
            prereg, binding, allow_index_additions=True
        )
        if destination.exists() or destination.is_symlink():
            manifest.recover_candidate_publication(book_id, joint_path, **common)
        else:
            stages = _partial_stage_for(book_id)
            if len(stages) != 1:
                raise RefreshError(f"{book_id}: expected exactly one authenticated candidate stage")
            manifest.publish_candidate(stages[0], joint_path, **common)
        _authenticate_current_comparator_a(
            prereg, binding, allow_index_additions=True
        )
        payload_names = lock.get("payload")
        if not isinstance(payload_names, dict) or not payload_names:
            raise RefreshError(f"{book_id}: publication payload inventory is malformed")
        outputs.extend(destination / name for name in payload_names)
        outputs.extend(
            [destination / f"{book_id}.publish_receipt.json", phase_path]
        )
    if survivors:
        outputs.extend([REPO_ROOT / "MODEL_REGISTRY.md", REPO_ROOT / "books" / "INDEX.json"])
    publication_phase_path = store.path("candidate_publication")
    publication_phase_existed = (
        publication_phase_path.exists() or publication_phase_path.is_symlink()
    )
    publication_phase = store.seal(
        "candidate_publication",
        canonical_attempt_id=f"{prereg_id}:candidate-publication:0",
        input_paths=[joint_path],
        output_paths=outputs,
        metadata={
            "run_id": run_id,
            "S": survivors,
            "published_inactive_only": True,
            "activation": False,
        },
    )
    publication_rollback: list[tuple[Path, bytes]] = []
    _record_new_exact_file(
        publication_rollback,
        publication_phase_path,
        existed_before=publication_phase_existed,
        expected=canonical_bytes(publication_phase),
    )
    try:
        _authenticate_current_comparator_a(
            prereg, binding, allow_index_additions=True
        )
    except Exception:
        _rollback_new_exact_files(publication_rollback)
        raise
    print(f"[publication] inactive candidates published={len(survivors)}", flush=True)


_PROVIDER_PARITY_FIELDS = {
    "schema", "status", "reason", "feature_schema_parity",
    "feature_value_parity", "runtime_scoring_and_gate_parity",
    "provider_schema_identity", "provider_implementation_identity",
    "provider_source_files", "runtime_implementation_git_sha", "provider_kind",
    "direct_incumbent_builder_parity", "live_feature_builder_parity",
    "runtime_decision_clock_authority",
}


def _validated_shadow_provider_evidence(
    pair: str,
    provider: Mapping[str, Any],
    *,
    spec: Mapping[str, Any],
    implementation_git_sha: str,
    parity_sha256: str,
    evaluator_id_value: str,
) -> tuple[dict[str, Any], bool, str | None]:
    """Authenticate one provider record and return its shadow qualification."""

    if set(provider) != _PROVIDER_PARITY_FIELDS:
        raise RefreshError(f"{pair}: provider parity fields differ")
    if provider.get("schema") != "m15-book-refresh-provider-parity/v1":
        raise RefreshError(f"{pair}: provider parity schema differs")
    status = provider.get("status")
    reason = provider.get("reason")
    if not isinstance(status, str) or not status:
        raise RefreshError(f"{pair}: provider parity status is malformed")
    if status == "PASSED":
        if reason is not None:
            raise RefreshError(f"{pair}: passing provider parity has a failure reason")
    elif not isinstance(reason, str) or not reason:
        raise RefreshError(f"{pair}: non-passing provider parity lacks a reason")
    pass_names = spec["shadow"]["provider_qualification"]["required_passes"]
    if pass_names != [
        "feature_schema_parity", "feature_value_parity",
        "runtime_scoring_and_gate_parity",
    ] or any(type(provider.get(name)) is not bool for name in pass_names):
        raise RefreshError(f"{pair}: provider parity pass fields differ")
    required_passes = all(provider[name] is True for name in pass_names)
    if status == "PASSED" and not required_passes:
        raise RefreshError(f"{pair}: passing provider parity has a failed requirement")
    if not _is_hex(provider.get("provider_schema_identity"), 64):
        raise RefreshError(f"{pair}: provider schema identity is malformed")
    sources = provider.get("provider_source_files")
    if not isinstance(sources, list) or not sources:
        raise RefreshError(f"{pair}: provider source inventory is malformed")
    source_paths: list[str] = []
    for source in sources:
        if (
            not isinstance(source, dict)
            or set(source) != {"path", "bytes", "sha256"}
            or not isinstance(source.get("path"), str)
            or not source["path"]
            or not isinstance(source.get("bytes"), int)
            or isinstance(source.get("bytes"), bool)
            or source["bytes"] <= 0
            or not _is_hex(source.get("sha256"), 64)
        ):
            raise RefreshError(f"{pair}: provider source identity is malformed")
        source_paths.append(source["path"])
    if len(source_paths) != len(set(source_paths)):
        raise RefreshError(f"{pair}: provider source inventory is duplicated")
    expected_source_paths = [
        "scripts/live_features.py",
        "scripts/pipeline.py",
        "scripts/m15_book_refresh_adapters.py",
        "scripts/book_runtime.py",
        "scripts/deriv_hot_daemon.py",
    ]
    if source_paths != expected_source_paths:
        raise RefreshError(f"{pair}: provider source inventory order differs")
    for source in sources:
        if source != asdict(file_identity(REPO_ROOT / source["path"])):
            raise RefreshError(f"{pair}: provider source bytes differ")
    expected_implementation_id = length_prefixed_digest(
        "m15-book-refresh/provider-implementation/v1",
        [canonical_bytes(sources)],
    )
    if provider.get("provider_implementation_identity") != expected_implementation_id:
        raise RefreshError(f"{pair}: provider implementation identity differs")
    if provider.get("runtime_implementation_git_sha") != implementation_git_sha:
        raise RefreshError(f"{pair}: provider runtime implementation differs")
    expected_kind = (
        "offline_xpair_plus_unavailable_live_orderflow"
        if pair == "EURUSD"
        else (
            "public_live_feature_builder_cross_pair"
            if spec["pairs"][pair]["adapter_family"] == "gbpchf_xpair"
            else "public_live_feature_builder_base"
        )
    )
    if provider.get("provider_kind") != expected_kind:
        raise RefreshError(f"{pair}: provider kind differs")
    direct = provider.get("direct_incumbent_builder_parity")
    if not (
        isinstance(direct, dict)
        and direct.get("schema") == "m15-direct-incumbent-builder-parity/v1"
        and direct.get("pair") == pair
        and direct.get("all_equal") is True
        and direct.get("source_to_decision_shift_exact") is True
    ):
        raise RefreshError(f"{pair}: direct incumbent-builder evidence differs")
    runtime_clock = provider.get("runtime_decision_clock_authority")
    daemon_source = next(
        source for source in sources if source["path"] == "scripts/deriv_hot_daemon.py"
    )
    if runtime_clock != {
        "function": "deriv_hot_daemon.score_from_snapshot",
        "field": "candidate_bar_close_utc",
        "mapping": "LiveFeatureBuilder.FeatureRow.timestamp+1minute",
        "source_feature_to_decision_shift_s": 60,
        "adapter_mapping_exact": True,
        "authority_source_sha256": daemon_source["sha256"],
    }:
        raise RefreshError(f"{pair}: runtime decision-clock authority differs")
    live = provider.get("live_feature_builder_parity")
    if pair == "EURUSD":
        if live is not None:
            raise RefreshError("EURUSD cannot claim unavailable live order-flow parity")
    elif not (
        isinstance(live, dict)
        and live.get("schema") == "m15-live-feature-builder-parity/v1"
        and live.get("pair") == pair
        and live.get("all_equal") is True
        and live.get("causal_decision_mapping_exact") is True
        and live.get("ordered_float32_feature_bytes_exact") is True
    ):
        raise RefreshError(f"{pair}: public LiveFeatureBuilder evidence differs")
    if pair == "EURUSD" and (
        status != "BLOCKED_SCHEMA" or reason != "verified_live_OF_provider_absent"
    ):
        raise RefreshError("EURUSD provider must remain blocked without verified live OF")
    if not _is_hex(parity_sha256, 64) or not _is_hex(evaluator_id_value, 64):
        raise RefreshError(f"{pair}: provider binding identity is malformed")
    qualified = status == "PASSED" and required_passes
    exclusion_reason = None if qualified else str(
        reason
        or ("provider_parity_required_pass_failed" if not required_passes
            else "provider_not_qualified")
    )
    evidence = dict(
        provider,
        provider_parity_result_sha256=parity_sha256,
        evaluator_id=evaluator_id_value,
    )
    return evidence, qualified, exclusion_reason


def _expected_deferred_shadow_seed_binding(
    ordered_pairs: Sequence[str], ordered_bundle_ids: Sequence[str]
) -> dict[str, Any]:
    identity_payload = [
        {"pair": pair, "candidate_C_bundle_id": bundle_id}
        for pair, bundle_id in zip(ordered_pairs, ordered_bundle_ids)
    ]
    if (
        not identity_payload
        or len(identity_payload) != len(ordered_pairs)
        or len(ordered_bundle_ids) != len(ordered_pairs)
        or any(not _is_hex(bundle_id, 64) for bundle_id in ordered_bundle_ids)
    ):
        raise RefreshError("deferred shadow seed identities are malformed/misaligned")
    domain = "m15-book-refresh/deferred-shadow-null-seed-identity/v1"
    digest = length_prefixed_digest(domain, [canonical_bytes(identity_payload)])
    digest_bytes = bytes.fromhex(digest)
    words = [
        int.from_bytes(digest_bytes[offset:offset + 4], "big")
        for offset in range(0, len(digest_bytes), 4)
    ]
    return {
        "domain": domain,
        "shadow_phase_code": 1,
        "identity_payload": identity_payload,
        "identity_sha256": digest,
        "identity_digest_u32be": words,
        "seed_sequence_layout": [
            "stats_null_test_seed",
            "shadow_phase_code",
            "identity_digest_u32be[0..7]",
            "fixture_index",
            "helper_code",
            "k_shadow",
            "campaign_index",
            "L_or_0",
            "stream_code",
        ],
    }


def _validate_deferred_shadow_null_artifact(
    path: Path,
    *,
    prereg_id: str,
    run_id: str,
    implementation_git_sha: str,
    ordered_shadow_seal: Mapping[str, Any],
) -> dict[str, Any]:
    """Authenticate the four post-S_shadow binding .541 null cells."""

    from scipy.stats import beta

    spec = load_spec()
    contract = spec["null_calibration"]
    if path.is_symlink() or not path.is_file() or path.stat().st_mode & 0o222:
        raise RefreshError(
            "deferred shadow null artifact must be a read-only regular file"
        )
    artifact = _read_json(path)
    required_fields = {
        "schema", "prereg_id", "run_id", "implementation_git_sha",
        "spec_sha256", "runner_sha256", "statistics_sha256",
        "ordered_S_shadow_seal_sha256", "ordered_shadow_pairs",
        "ordered_shadow_bundle_ids", "ordered_coverages", "seed_binding",
        "contract", "campaign_count",
        "bootstrap_replicates", "block_lengths", "cells", "cells_sha256",
        "accepted", "semantic_replay_outcomes_already_accessed",
        "synthetic_calibration_read_replay_outcomes",
    }
    if set(artifact) != required_fields:
        raise RefreshError("deferred shadow null artifact fields differ")
    try:
        import m15_book_refresh_stats as stats
    except ImportError as exc:
        raise RefreshError("cannot import deferred shadow statistics authority") from exc
    ordered_pairs = list(ordered_shadow_seal.get("ordered_shadow_pairs", []))
    ordered_bundle_ids = [
        row.get("candidate_bundle_id")
        for row in ordered_shadow_seal.get("ordered_candidates", [])
        if isinstance(row, dict)
    ]
    ordered_coverages = list(ordered_shadow_seal.get("ordered_coverages", []))
    expected_seed_binding = _expected_deferred_shadow_seed_binding(
        ordered_pairs,
        ordered_bundle_ids,
    )
    if (
        artifact.get("schema")
        != "m15-book-refresh-deferred-shadow-null-calibration-result/v1"
        or artifact.get("prereg_id") != prereg_id
        or artifact.get("run_id") != run_id
        or artifact.get("implementation_git_sha") != implementation_git_sha
        or artifact.get("spec_sha256") != sha256_file(SPEC_PATH)
        or artifact.get("runner_sha256") != sha256_file(Path(__file__))
        or artifact.get("statistics_sha256") != sha256_file(Path(stats.__file__))
        or artifact.get("ordered_S_shadow_seal_sha256")
        != sha256_bytes(canonical_bytes(ordered_shadow_seal))
        or artifact.get("ordered_shadow_pairs") != ordered_pairs
        or artifact.get("ordered_shadow_bundle_ids") != ordered_bundle_ids
        or artifact.get("ordered_coverages") != ordered_coverages
        or artifact.get("seed_binding") != expected_seed_binding
        or artifact.get("contract") != contract["binding_deferred_shadow"]
        or artifact.get("campaign_count") != contract["campaigns_per_cell"]
        or artifact.get("bootstrap_replicates")
        != contract["bootstrap_replicates"]
        or artifact.get("block_lengths") != contract["block_lengths"]
        or artifact.get("semantic_replay_outcomes_already_accessed") is not True
        or artifact.get("synthetic_calibration_read_replay_outcomes") is not False
    ):
        raise RefreshError("deferred shadow null artifact identity/contract differs")
    cells = artifact.get("cells")
    fixture_order = contract["binding_deferred_shadow"]["fixture_order"]
    if not isinstance(cells, list) or len(cells) != len(fixture_order):
        raise RefreshError("deferred shadow null artifact must contain exactly four cells")
    if artifact.get("cells_sha256") != length_prefixed_digest(
        "m15-book-refresh/deferred-shadow-null-cells/v1",
        [canonical_bytes(cells)],
    ):
        raise RefreshError("deferred shadow null cell digest differs")
    required_cell_fields = {
        "schema", "fixture", "helper", "k_shadow", "ordered_shadow_pairs",
        "accuracy_null", "binding", "calibration_role", "campaign_count",
        "bootstrap_replicates", "block_lengths", "false_positive_count",
        "false_positive_limit", "clopper_pearson_upper_95", "cell_passed",
        "accepted", "deterministic_repeat", "campaign_digest",
        "seed_binding",
    }
    accepted: list[bool] = []
    for fixture, cell in zip(fixture_order, cells):
        if not isinstance(cell, dict):
            raise RefreshError("deferred shadow null cell is malformed")
        count = cell.get("false_positive_count")
        count_valid = (
            isinstance(count, int)
            and not isinstance(count, bool)
            and 0 <= count <= contract["campaigns_per_cell"]
        )
        expected_cp = None
        if count_valid:
            expected_cp = (
                1.0
                if count == contract["campaigns_per_cell"]
                else float(
                    beta.ppf(
                        0.95,
                        count + 1,
                        contract["campaigns_per_cell"] - count,
                    )
                )
            )
        passed = bool(
            count_valid
            and cell.get("deterministic_repeat") is True
            and count <= contract["false_positive_limit_inclusive"]
        )
        exact = (
            set(cell) == required_cell_fields
            and cell.get("schema")
            == "m15-book-refresh-null-calibration-cell/v1"
            and cell.get("fixture") == fixture
            and cell.get("helper") == "shadow"
            and cell.get("k_shadow") == len(ordered_pairs)
            and cell.get("ordered_shadow_pairs") == ordered_pairs
            and cell.get("accuracy_null") == 0.541
            and cell.get("binding") is True
            and cell.get("calibration_role") == "binding_deferred_shadow"
            and cell.get("seed_binding") == expected_seed_binding
            and cell.get("campaign_count") == contract["campaigns_per_cell"]
            and cell.get("bootstrap_replicates")
            == contract["bootstrap_replicates"]
            and cell.get("block_lengths") == contract["block_lengths"]
            and cell.get("false_positive_limit")
            == contract["false_positive_limit_inclusive"]
            and count_valid
            and isinstance(cell.get("clopper_pearson_upper_95"), (int, float))
            and np.isfinite(float(cell["clopper_pearson_upper_95"]))
            and abs(float(cell["clopper_pearson_upper_95"]) - float(expected_cp))
            <= 1e-15
            and type(cell.get("deterministic_repeat")) is bool
            and cell.get("cell_passed") is passed
            and cell.get("accepted") is passed
            and _is_hex(cell.get("campaign_digest"), 64)
        )
        if not exact:
            raise RefreshError(f"deferred shadow null cell differs: {fixture}")
        accepted.append(passed)
    aggregate = bool(all(accepted))
    if artifact.get("accepted") is not aggregate:
        raise RefreshError("deferred shadow null aggregate acceptance differs")
    if not aggregate:
        failures = [
            f"{cell.get('fixture')}={cell.get('false_positive_count')}"
            for cell in cells
            if not cell.get("accepted")
        ]
        raise RefreshError(
            "deferred shadow null calibration blocked T0 handoff: "
            + ", ".join(failures)
        )
    return artifact


def _seal_deferred_shadow_null_calibration(
    workspace: Path,
    *,
    prereg_id: str,
    run_id: str,
    implementation_git_sha: str,
    ordered_shadow_seal_path: Path,
) -> Path:
    """Run once, or authenticate, the binding cells after S_shadow is sealed."""

    import m15_book_refresh_stats as stats

    spec = load_spec()
    ordered_shadow_seal = _read_json(ordered_shadow_seal_path)
    ordered_pairs = list(ordered_shadow_seal.get("ordered_shadow_pairs", []))
    if not ordered_pairs:
        raise RefreshError("deferred shadow calibration requires nonempty sealed S_shadow")
    ordered_bundle_ids = [
        row.get("candidate_bundle_id")
        for row in ordered_shadow_seal.get("ordered_candidates", [])
        if isinstance(row, dict)
    ]
    seed_binding = _expected_deferred_shadow_seed_binding(
        ordered_pairs,
        ordered_bundle_ids,
    )
    if seed_binding != stats.deferred_shadow_seed_binding(
        ordered_shadow_pairs=ordered_pairs,
        ordered_shadow_bundle_ids=ordered_bundle_ids,
    ):
        raise RefreshError("runner/statistics deferred seed derivations diverge")
    sealed_coverages = {
        row.get("pair"): row.get("coverage")
        for row in ordered_shadow_seal.get("ordered_coverages", [])
        if isinstance(row, dict)
    }
    expected_coverages = {
        pair: float(spec["pairs"][pair]["calibration_target_coverage"])
        for pair in ordered_pairs
    }
    if (
        sealed_coverages != expected_coverages
        or {pair: float(stats.PAIR_COVERAGE[pair]) for pair in ordered_pairs}
        != expected_coverages
    ):
        raise RefreshError("deferred shadow calibration coverages differ from sealed spec")
    path = (
        workspace
        / "shadow"
        / f"deferred_shadow_null_calibration_{run_id}.json"
    )
    if path.exists() or path.is_symlink():
        _validate_deferred_shadow_null_artifact(
            path,
            prereg_id=prereg_id,
            run_id=run_id,
            implementation_git_sha=implementation_git_sha,
            ordered_shadow_seal=ordered_shadow_seal,
        )
        return path
    contract = spec["null_calibration"]
    cells = stats.run_deferred_shadow_null_calibration_suite(
        ordered_shadow_pairs=ordered_pairs,
        ordered_shadow_bundle_ids=ordered_bundle_ids,
        campaign_count=int(contract["campaigns_per_cell"]),
        bootstrap_replicates=int(contract["bootstrap_replicates"]),
        block_lengths=tuple(int(value) for value in contract["block_lengths"]),
        verify_repeat=True,
    )
    values = [cell.as_dict() for cell in cells]
    payload = {
        "schema": "m15-book-refresh-deferred-shadow-null-calibration-result/v1",
        "prereg_id": prereg_id,
        "run_id": run_id,
        "implementation_git_sha": implementation_git_sha,
        "spec_sha256": sha256_file(SPEC_PATH),
        "runner_sha256": sha256_file(Path(__file__)),
        "statistics_sha256": sha256_file(Path(stats.__file__)),
        "ordered_S_shadow_seal_sha256": sha256_file(ordered_shadow_seal_path),
        "ordered_shadow_pairs": ordered_pairs,
        "ordered_shadow_bundle_ids": ordered_bundle_ids,
        "ordered_coverages": ordered_shadow_seal["ordered_coverages"],
        "seed_binding": seed_binding,
        "contract": contract["binding_deferred_shadow"],
        "campaign_count": contract["campaigns_per_cell"],
        "bootstrap_replicates": contract["bootstrap_replicates"],
        "block_lengths": contract["block_lengths"],
        "cells": values,
        "cells_sha256": length_prefixed_digest(
            "m15-book-refresh/deferred-shadow-null-cells/v1",
            [canonical_bytes(values)],
        ),
        "accepted": bool(all(cell.accepted for cell in cells)),
        "semantic_replay_outcomes_already_accessed": True,
        "synthetic_calibration_read_replay_outcomes": False,
    }
    _seal_json_exact(path, payload)
    _validate_deferred_shadow_null_artifact(
        path,
        prereg_id=prereg_id,
        run_id=run_id,
        implementation_git_sha=implementation_git_sha,
        ordered_shadow_seal=ordered_shadow_seal,
    )
    return path


def _validate_shadow_handoff_payload(
    payload: Mapping[str, Any], *, spec: Mapping[str, Any]
) -> None:
    """Fail closed on the exact zero-eligible or positive handoff shape."""

    shadow = spec["shadow"]
    identity_fields = set(shadow["completed_retrospective_identity_fields"])
    common_fields = identity_fields | {
        "schema", "status", "S", "S_records", "S_shadow", "k_shadow",
        "exclusions", "provider_parity_result", "provider_parity_result_sha256",
        "activation", "shadow_started",
    }
    zero_fields = common_fields
    positive_fields = common_fields | {
        "eligible_candidates", "T0", "T0_contract", "no_backfill",
        "endpoint_count", "fixed_family_size", "endpoint_order_per_candidate",
        "endpoint_formulas", "prospective_capture", "fixed_look", "statistics",
        "promotion_toward_activation", "activation_prerequisite",
        "family_reset_and_relook", "ordered_S_shadow_seal",
        "deferred_shadow_null_calibration",
    }
    schema = payload.get("schema")
    expected_fields = (
        zero_fields
        if schema == shadow["schemas"]["zero_eligible"]
        else positive_fields
        if schema == shadow["schemas"]["positive_eligible"]
        else None
    )
    if expected_fields is None or set(payload) != expected_fields:
        raise RefreshError("prospective shadow payload fields differ")
    for name, length in (
        ("prereg_id", 64), ("run_id", 64), ("implementation_git_sha", 40),
        ("evaluator_id", 64), ("controls_id", 64), ("run_binding_sha256", 64),
        ("provider_parity_result_sha256", 64),
    ):
        if not _is_hex(payload.get(name), length):
            raise RefreshError(f"prospective shadow {name} is malformed")
    controls_id_value = derive_controls_id(
        spec, payload.get("ordered_control_arms_with_bundle_ids", [])
    )
    run_id_value = derive_run_id(
        spec,
        payload["prereg_id"],
        controls_id_value,
        payload.get("ordered_primary_arms_with_bundle_ids", []),
    )
    if (
        controls_id_value != payload["controls_id"]
        or run_id_value != payload["run_id"]
    ):
        raise RefreshError("prospective shadow completed arm identities differ")
    primary_ids = {
        (row[0], row[1]): row[2]
        for row in payload["ordered_primary_arms_with_bundle_ids"]
    }
    if (
        payload.get("activation") is not False
        or payload.get("shadow_started") is not False
        or not isinstance(payload.get("provider_parity_result"), str)
        or not payload["provider_parity_result"]
    ):
        raise RefreshError("prospective shadow activation/provider binding differs")
    survivors = payload.get("S")
    eligible = payload.get("S_shadow")
    records = payload.get("S_records")
    exclusions = payload.get("exclusions")
    if (
        not isinstance(survivors, list)
        or not isinstance(eligible, list)
        or not isinstance(records, list)
        or not isinstance(exclusions, dict)
        or any(not isinstance(book, str) or not book for book in survivors)
        or any(not isinstance(book, str) or not book for book in eligible)
        or len(survivors) != len(set(survivors))
        or len(eligible) != len(set(eligible))
        or [book for book in survivors if book in set(eligible)] != eligible
        or payload.get("k_shadow") != len(eligible)
        or len(records) != len(survivors)
    ):
        raise RefreshError("prospective shadow survivor/provider sets differ")
    record_by_id: dict[str, Mapping[str, Any]] = {}
    for record in records:
        if (
            not isinstance(record, dict)
            or set(record) != {
                "pair", "candidate_book_id", "candidate_bundle_id", "v1_book_id",
                "v1_bundle_id", "lifecycle_status", "activation", "provider_evidence",
            }
            or record.get("lifecycle_status") != "inactive_shadow_candidate"
            or record.get("activation") is not False
            or not _is_hex(record.get("candidate_bundle_id"), 64)
            or not _is_hex(record.get("v1_bundle_id"), 64)
        ):
            raise RefreshError("prospective shadow survivor record differs")
        book_id = record.get("candidate_book_id")
        pair = record.get("pair")
        if (
            not isinstance(book_id, str)
            or not isinstance(pair, str)
            or pair not in spec["pair_order"]
            or book_id != spec["pairs"][pair]["candidate_book_id"]
            or record.get("v1_book_id") != spec["pairs"][pair]["book_id_A"]
            or record.get("candidate_bundle_id") != primary_ids[(pair, "C")]
            or record.get("v1_bundle_id") != primary_ids[(pair, "A")]
            or book_id in record_by_id
        ):
            raise RefreshError("prospective shadow survivor identity differs")
        evidence = record.get("provider_evidence")
        if not isinstance(evidence, dict) or set(evidence) != (
            _PROVIDER_PARITY_FIELDS | {"provider_parity_result_sha256", "evaluator_id"}
        ):
            raise RefreshError(f"{pair}: prospective provider evidence fields differ")
        if (
            evidence.get("provider_parity_result_sha256")
            != payload["provider_parity_result_sha256"]
            or evidence.get("evaluator_id") != payload["evaluator_id"]
        ):
            raise RefreshError(f"{pair}: prospective provider evidence binding differs")
        _validated_shadow_provider_evidence(
            pair,
            {name: evidence[name] for name in _PROVIDER_PARITY_FIELDS},
            spec=spec,
            implementation_git_sha=payload["implementation_git_sha"],
            parity_sha256=payload["provider_parity_result_sha256"],
            evaluator_id_value=payload["evaluator_id"],
        )
        record_by_id[book_id] = record
    if list(record_by_id) != survivors:
        raise RefreshError("prospective shadow survivor record order differs")
    if set(exclusions) != set(survivors) - set(eligible):
        raise RefreshError("prospective shadow exclusions differ")
    for book_id, exclusion in exclusions.items():
        if (
            not isinstance(exclusion, dict)
            or set(exclusion) != set(record_by_id[book_id]) | {"reason"}
            or {key: exclusion[key] for key in record_by_id[book_id]}
            != record_by_id[book_id]
            or not isinstance(exclusion.get("reason"), str)
            or not exclusion["reason"]
        ):
            raise RefreshError("prospective shadow exclusion record differs")
    if schema == shadow["schemas"]["zero_eligible"]:
        expected_status = (
            shadow["statuses"]["empty_survivors"]
            if not survivors
            else shadow["statuses"]["survivors_but_zero_provider_eligible"]
        )
        if eligible or payload.get("status") != expected_status:
            raise RefreshError("zero-eligible prospective shadow status differs")
        return
    ordered_ref = payload.get("ordered_S_shadow_seal")
    deferred_ref = payload.get("deferred_shadow_null_calibration")
    if (
        not isinstance(ordered_ref, dict)
        or set(ordered_ref) != {"artifact", "sha256"}
        or not isinstance(deferred_ref, dict)
        or set(deferred_ref)
        != {
            "artifact", "sha256", "accepted", "cells", "accuracy_null",
            "false_positive_limit_inclusive",
        }
        or ordered_ref.get("artifact") is None
        or deferred_ref.get("artifact") is None
    ):
        raise RefreshError("positive shadow null-calibration references differ")
    ordered_path = REPO_ROOT / str(ordered_ref["artifact"])
    deferred_path = REPO_ROOT / str(deferred_ref["artifact"])
    for name, reference, path in (
        ("ordered S_shadow", ordered_ref, ordered_path),
        ("deferred shadow null", deferred_ref, deferred_path),
    ):
        if (
            path.is_symlink()
            or not path.is_file()
            or path.stat().st_mode & 0o222
            or repo_relative(path) != reference["artifact"]
            or not _is_hex(reference.get("sha256"), 64)
            or sha256_file(path) != reference["sha256"]
        ):
            raise RefreshError(f"{name} evidence reference differs")
    ordered_seal = _read_json(ordered_path)
    expected_pairs = [record_by_id[book]["pair"] for book in eligible]
    expected_coverages = [
        {
            "pair": pair,
            "coverage": float(spec["pairs"][pair]["calibration_target_coverage"]),
        }
        for pair in expected_pairs
    ]
    expected_candidates = [
        {
            "pair": record_by_id[book]["pair"],
            "candidate_book_id": book,
            "candidate_bundle_id": record_by_id[book]["candidate_bundle_id"],
            "v1_book_id": record_by_id[book]["v1_book_id"],
            "v1_bundle_id": record_by_id[book]["v1_bundle_id"],
            "coverage": float(
                spec["pairs"][record_by_id[book]["pair"]][
                    "calibration_target_coverage"
                ]
            ),
        }
        for book in eligible
    ]
    if (
        set(ordered_seal)
        != {
            "schema", "prereg_id", "run_id", "implementation_git_sha",
            "evaluator_id", "S_shadow", "k_shadow", "ordered_shadow_pairs",
            "ordered_coverages", "ordered_candidates", "joint_result_sha256",
            "run_binding_sha256", "provider_parity_result_sha256",
            "semantic_replay_outcomes_already_accessed",
        }
        or ordered_seal.get("schema")
        != "m15-book-refresh-ordered-S-shadow-seal/v1"
        or ordered_seal.get("prereg_id") != payload["prereg_id"]
        or ordered_seal.get("run_id") != payload["run_id"]
        or ordered_seal.get("implementation_git_sha")
        != payload["implementation_git_sha"]
        or ordered_seal.get("evaluator_id") != payload["evaluator_id"]
        or ordered_seal.get("S_shadow") != eligible
        or ordered_seal.get("k_shadow") != len(eligible)
        or ordered_seal.get("ordered_shadow_pairs") != expected_pairs
        or ordered_seal.get("ordered_coverages") != expected_coverages
        or ordered_seal.get("ordered_candidates") != expected_candidates
        or ordered_seal.get("joint_result_sha256")
        != sha256_file(
            RESULTS_JSON
            / f"m15_book_refresh_{payload['run_id']}_joint_replay_result.json"
        )
        or ordered_seal.get("run_binding_sha256")
        != payload["run_binding_sha256"]
        or ordered_seal.get("provider_parity_result_sha256")
        != payload["provider_parity_result_sha256"]
        or ordered_seal.get("semantic_replay_outcomes_already_accessed") is not True
    ):
        raise RefreshError("ordered S_shadow seal differs from positive handoff")
    deferred = _validate_deferred_shadow_null_artifact(
        deferred_path,
        prereg_id=payload["prereg_id"],
        run_id=payload["run_id"],
        implementation_git_sha=payload["implementation_git_sha"],
        ordered_shadow_seal=ordered_seal,
    )
    if deferred_ref != {
        "artifact": repo_relative(deferred_path),
        "sha256": sha256_file(deferred_path),
        "accepted": True,
        "cells": 4,
        "accuracy_null": 0.541,
        "false_positive_limit_inclusive": 32,
    } or deferred.get("accepted") is not True:
        raise RefreshError("deferred shadow null handoff binding differs")
    if (
        not eligible
        or payload.get("status") != shadow["statuses"]["positive_eligible_handoff"]
        or payload.get("eligible_candidates") != [record_by_id[book] for book in eligible]
        or payload.get("T0") is not None
        or payload.get("T0_contract")
        != shadow["positive_eligible_handoff"]["T0_contract"]
        or payload.get("no_backfill") is not True
        or payload.get("endpoint_count") != shadow["endpoint_count"]
        or payload.get("fixed_family_size") != 3 * len(eligible)
        or payload.get("endpoint_order_per_candidate")
        != shadow["endpoint_order_per_candidate"]
        or payload.get("endpoint_formulas") != shadow["endpoint_formulas"]
        or payload.get("prospective_capture") != shadow["prospective_capture"]
        or payload.get("fixed_look") != shadow["fixed_look"]
        or payload.get("statistics") != shadow["statistics"]
        or payload.get("promotion_toward_activation")
        != shadow["promotion_toward_activation"]
        or payload.get("activation_prerequisite") != shadow["activation_prerequisite"]
        or payload.get("family_reset_and_relook") != shadow["family_reset_and_relook"]
    ):
        raise RefreshError("positive prospective shadow handoff differs")


def seal_shadow_spec(prereg_id: str) -> Path:
    """Emit a provider-evidence-qualified, no-start prospective handoff."""

    spec = load_spec()
    shadow = spec["shadow"]
    workspace, store, prereg = workspace_for(prereg_id)
    binding = _run_binding(workspace, store)
    _authenticate_current_comparator_a(
        prereg, binding, allow_index_additions=True
    )
    run_id = binding["run_id"]
    store.load("candidate_publication")
    parity_path = workspace / "parity" / "parity.json"
    store.load("adapter_parity")
    parity = _read_json(parity_path)
    if (
        parity.get("schema") != "m15-book-refresh-parity/v1"
        or parity.get("prereg_id") != prereg_id
        or parity.get("implementation_git_sha") != prereg["implementation_git_sha"]
    ):
        raise RefreshError("shadow provider parity evidence differs from preregistration")
    binding_path = workspace / "arm_seal" / f"run_binding_{run_id}.json"
    tracked_joint = RESULTS_JSON / f"m15_book_refresh_{run_id}_joint_replay_result.json"
    joint = _read_json(tracked_joint)
    survivors = list(joint.get("S", []))
    if len(survivors) != len(set(survivors)):
        raise RefreshError("shadow survivor set is duplicated")
    pair_by_candidate = {
        str(spec["pairs"][pair]["candidate_book_id"]): pair
        for pair in spec["pair_order"]
    }
    if any(book_id not in pair_by_candidate for book_id in survivors):
        raise RefreshError("shadow survivor set contains an unknown candidate")
    primary_ids = {(row[0], row[1]): row[2] for row in binding["primary_arms"]}
    completed_identities = {
        "prereg_id": prereg_id,
        "run_id": run_id,
        "implementation_git_sha": prereg["implementation_git_sha"],
        "evaluator_id": binding["evaluator_id"],
        "controls_id": binding["controls_id"],
        "ordered_primary_arms_with_bundle_ids": binding["primary_arms"],
        "ordered_control_arms_with_bundle_ids": binding["control_arms"],
        "run_binding_sha256": sha256_file(binding_path),
    }
    if list(completed_identities) != shadow["completed_retrospective_identity_fields"]:
        raise RefreshError("completed retrospective shadow identity order differs")
    parity_sha = sha256_file(parity_path)
    survivor_records: list[dict[str, Any]] = []
    eligible: list[str] = []
    exclusions: dict[str, dict[str, Any]] = {}
    for book_id in survivors:
        pair = pair_by_candidate[book_id]
        provider = parity.get("pairs", {}).get(pair, {}).get("provider_parity")
        if not isinstance(provider, dict):
            raise RefreshError(f"{pair}: provider parity evidence is absent")
        provider_evidence, qualified, exclusion_reason = (
            _validated_shadow_provider_evidence(
                pair,
                provider,
                spec=spec,
                implementation_git_sha=prereg["implementation_git_sha"],
                parity_sha256=parity_sha,
                evaluator_id_value=binding["evaluator_id"],
            )
        )
        record = {
            "pair": pair,
            "candidate_book_id": book_id,
            "candidate_bundle_id": primary_ids[(pair, "C")],
            "v1_book_id": spec["pairs"][pair]["book_id_A"],
            "v1_bundle_id": primary_ids[(pair, "A")],
            "lifecycle_status": "inactive_shadow_candidate",
            "activation": False,
            "provider_evidence": provider_evidence,
        }
        survivor_records.append(record)
        if qualified:
            eligible.append(book_id)
        else:
            exclusions[book_id] = {
                **record,
                "reason": exclusion_reason,
            }
    _authenticate_current_comparator_a(
        prereg, binding, allow_index_additions=True
    )
    shadow_dir = workspace / "shadow"
    shadow_dir_existed = shadow_dir.exists() or shadow_dir.is_symlink()
    shadow_dir.mkdir(parents=True, exist_ok=True)
    rollback_records: list[tuple[Path, bytes]] = []
    ordered_shadow_path: Path | None = None
    deferred_null_path: Path | None = None
    if not eligible:
        status = (
            shadow["statuses"]["empty_survivors"]
            if not survivors
            else shadow["statuses"]["survivors_but_zero_provider_eligible"]
        )
        payload = {
            "schema": shadow["schemas"]["zero_eligible"],
            "status": status,
            **completed_identities,
            "S": survivors,
            "S_records": survivor_records,
            "S_shadow": [],
            "k_shadow": 0,
            "exclusions": exclusions,
            "provider_parity_result": repo_relative(parity_path),
            "provider_parity_result_sha256": parity_sha,
            "activation": False,
            "shadow_started": False,
        }
    else:
        eligible_records = [
            record for record in survivor_records
            if record["candidate_book_id"] in set(eligible)
        ]
        ordered_pairs = [record["pair"] for record in eligible_records]
        expected_order = [
            pair for pair in spec["pair_order"] if pair in set(ordered_pairs)
        ]
        if ordered_pairs != expected_order:
            raise RefreshError("S_shadow does not preserve the sealed target order")
        ordered_coverages = [
            {
                "pair": pair,
                "coverage": float(
                    spec["pairs"][pair]["calibration_target_coverage"]
                ),
            }
            for pair in ordered_pairs
        ]
        ordered_candidates = [
            {
                "pair": record["pair"],
                "candidate_book_id": record["candidate_book_id"],
                "candidate_bundle_id": record["candidate_bundle_id"],
                "v1_book_id": record["v1_book_id"],
                "v1_bundle_id": record["v1_bundle_id"],
                "coverage": float(
                    spec["pairs"][record["pair"]][
                        "calibration_target_coverage"
                    ]
                ),
            }
            for record in eligible_records
        ]
        ordered_shadow = {
            "schema": "m15-book-refresh-ordered-S-shadow-seal/v1",
            "prereg_id": prereg_id,
            "run_id": run_id,
            "implementation_git_sha": prereg["implementation_git_sha"],
            "evaluator_id": binding["evaluator_id"],
            "S_shadow": eligible,
            "k_shadow": len(eligible),
            "ordered_shadow_pairs": ordered_pairs,
            "ordered_coverages": ordered_coverages,
            "ordered_candidates": ordered_candidates,
            "joint_result_sha256": sha256_file(tracked_joint),
            "run_binding_sha256": sha256_file(binding_path),
            "provider_parity_result_sha256": parity_sha,
            "semantic_replay_outcomes_already_accessed": True,
        }
        ordered_shadow_path = (
            shadow_dir / f"ordered_S_shadow_{run_id}.json"
        )
        ordered_shadow_existed = (
            ordered_shadow_path.exists() or ordered_shadow_path.is_symlink()
        )
        # This immutable authority is created before the deferred suite.  The
        # statistics API receives only the exact target order read back from it.
        _seal_json_exact(ordered_shadow_path, ordered_shadow)
        _record_new_exact_file(
            rollback_records,
            ordered_shadow_path,
            existed_before=ordered_shadow_existed,
            expected=canonical_bytes(ordered_shadow),
        )
        if _read_json(ordered_shadow_path) != ordered_shadow:
            raise RefreshError("ordered S_shadow seal bytes do not round-trip")
        expected_deferred_path = (
            shadow_dir / f"deferred_shadow_null_calibration_{run_id}.json"
        )
        deferred_existed = (
            expected_deferred_path.exists() or expected_deferred_path.is_symlink()
        )
        deferred_null_path = _seal_deferred_shadow_null_calibration(
            workspace,
            prereg_id=prereg_id,
            run_id=run_id,
            implementation_git_sha=prereg["implementation_git_sha"],
            ordered_shadow_seal_path=ordered_shadow_path,
        )
        if deferred_null_path != expected_deferred_path:
            raise RefreshError("deferred shadow null output path differs")
        _record_new_exact_file(
            rollback_records,
            deferred_null_path,
            existed_before=deferred_existed,
            expected=deferred_null_path.read_bytes(),
        )
        payload = {
            "schema": shadow["schemas"]["positive_eligible"],
            "status": shadow["statuses"]["positive_eligible_handoff"],
            **completed_identities,
            "S": survivors,
            "S_records": survivor_records,
            "S_shadow": eligible,
            "k_shadow": len(eligible),
            "eligible_candidates": eligible_records,
            "exclusions": exclusions,
            "provider_parity_result": repo_relative(parity_path),
            "provider_parity_result_sha256": parity_sha,
            "T0": None,
            "T0_contract": shadow["positive_eligible_handoff"]["T0_contract"],
            "activation": False,
            "shadow_started": False,
            "no_backfill": True,
            "endpoint_count": shadow["endpoint_count"],
            "fixed_family_size": 3 * len(eligible),
            "endpoint_order_per_candidate": shadow["endpoint_order_per_candidate"],
            "endpoint_formulas": shadow["endpoint_formulas"],
            "prospective_capture": shadow["prospective_capture"],
            "fixed_look": shadow["fixed_look"],
            "statistics": shadow["statistics"],
            "promotion_toward_activation": shadow["promotion_toward_activation"],
            "activation_prerequisite": shadow["activation_prerequisite"],
            "family_reset_and_relook": shadow["family_reset_and_relook"],
            "ordered_S_shadow_seal": {
                "artifact": repo_relative(ordered_shadow_path),
                "sha256": sha256_file(ordered_shadow_path),
            },
            "deferred_shadow_null_calibration": {
                "artifact": repo_relative(deferred_null_path),
                "sha256": sha256_file(deferred_null_path),
                "accepted": True,
                "cells": 4,
                "accuracy_null": 0.541,
                "false_positive_limit_inclusive": 32,
            },
        }
    try:
        _authenticate_current_comparator_a(
            prereg, binding, allow_index_additions=True
        )
    except Exception:
        _rollback_new_exact_files(
            rollback_records,
            new_directories=(() if shadow_dir_existed else (shadow_dir,)),
        )
        raise
    _validate_shadow_handoff_payload(payload, spec=spec)
    workspace_path = shadow_dir / f"shadow_spec_{run_id}.json"
    tracked = RESULTS_JSON / f"m15_book_refresh_{run_id}_shadow_spec_result.json"
    workspace_path_existed = workspace_path.exists() or workspace_path.is_symlink()
    tracked_existed = tracked.exists() or tracked.is_symlink()
    _seal_json_exact(workspace_path, payload)
    _seal_json_exact(tracked, payload)
    payload_bytes = canonical_bytes(payload)
    _record_new_exact_file(
        rollback_records,
        workspace_path,
        existed_before=workspace_path_existed,
        expected=payload_bytes,
    )
    _record_new_exact_file(
        rollback_records,
        tracked,
        existed_before=tracked_existed,
        expected=payload_bytes,
    )
    phase_outputs = [workspace_path, tracked]
    if ordered_shadow_path is not None and deferred_null_path is not None:
        phase_outputs.extend([ordered_shadow_path, deferred_null_path])
    shadow_phase_path = store.path("shadow_spec")
    shadow_phase_existed = shadow_phase_path.exists() or shadow_phase_path.is_symlink()
    shadow_phase = store.seal(
        "shadow_spec",
        canonical_attempt_id=f"{prereg_id}:shadow-spec:0",
        input_paths=[tracked_joint, binding_path, parity_path],
        output_paths=phase_outputs,
        metadata={
            "run_id": run_id,
            "status": payload["status"],
            "k_shadow": payload["k_shadow"],
            "activation": False,
            "deferred_shadow_null_accepted": bool(
                payload.get("deferred_shadow_null_calibration", {}).get("accepted")
            ),
        },
    )
    _record_new_exact_file(
        rollback_records,
        shadow_phase_path,
        existed_before=shadow_phase_existed,
        expected=canonical_bytes(shadow_phase),
    )
    try:
        _authenticate_current_comparator_a(
            prereg, binding, allow_index_additions=True
        )
    except Exception:
        _rollback_new_exact_files(
            rollback_records,
            new_directories=(() if shadow_dir_existed else (shadow_dir,)),
        )
        raise
    print(f"[shadow-spec] status={payload['status']} k_shadow={payload['k_shadow']}", flush=True)
    return tracked


def _copy_json(value: Any) -> Any:
    return json.loads(json.dumps(value))


def smoke() -> None:
    """Exercise infrastructure without reading labels, outcomes, or real books."""

    spec = load_spec()
    eid = evaluator_id(spec)
    if not _is_hex(eid, 64):
        raise RefreshError("smoke evaluator identity is malformed")
    source_sha = "1" * 40
    payload = {"schema": "smoke-prereg/v1", "falsifier": "synthetic-only"}
    pid = derive_prereg_id(spec, payload, source_sha)
    if pid != derive_prereg_id(spec, payload, source_sha):
        raise RefreshError("identity is not deterministic")

    with tempfile.TemporaryDirectory(prefix="m15-refresh-smoke-") as temporary:
        root = Path(temporary)
        workspace = root / "logs" / pid
        store = PhaseStore(workspace)
        sentinel = workspace / "sentinel.json"
        atomic_json_new(sentinel, {"schema": "smoke-sentinel/v1"})
        store.seal(
            "source_preregistration",
            canonical_attempt_id="smoke:source:0",
            output_paths=[sentinel],
            metadata={"semantic_outcomes_read": False},
        )
        blocked_reader = SemanticAccessFence(
            store, lambda: (_ for _ in ()).throw(AssertionError("outcome reader called"))
        )
        try:
            blocked_reader()
        except RefreshError:
            pass
        else:
            raise RefreshError("semantic outcome reader was not fenced before arm seal")

        # Seal empty synthetic predecessors.  The reader remains an exploding
        # fixture, proving smoke never calls it even after infrastructure setup.
        for phase in ("derived_data", "adapter_parity", "fit_and_repeat", "arm_seal"):
            store.seal(phase, canonical_attempt_id=f"smoke:{phase}:0")

        clock = np.arange(100, 110, dtype="int64") * 1_000_000_000
        if not np.array_equal(checked_seconds_from_ns(clock, name="smoke clock"), np.arange(100, 110)):
            raise RefreshError("checked timestamp conversion failed")
        for bad in (clock[::-1], np.array([1, 1], dtype="int64")):
            try:
                assert_strict_clock(bad, name="bad smoke clock")
            except RefreshError:
                pass
            else:
                raise RefreshError("timestamp precondition accepted invalid input")

        # Candidate publication runs against a complete temporary repo shape.
        repo = root / "repo"
        books = repo / "books"
        books.mkdir(parents=True)
        (books / "INDEX.json").write_text('{"schema":"book-index/v1","books":[]}\n')
        artifact_dir = root / "artifacts"
        artifact_dir.mkdir()
        strategy = artifact_dir / "SMOKE_strategy.json"
        model = artifact_dir / "SMOKE_s0_lgb.txt"
        strategy.write_bytes(canonical_bytes({"pair": "SMOKE", "conf_thr": 0.1}))
        model.write_bytes(b"tree\n")
        resolved_without_id = {
            "schema": "m15-book-refresh-resolved-bundle-spec/v1",
            "book_id": "SMOKE.m15.v2",
            "prereg_id": pid,
            "coverage": 0.1,
            "threshold": 0.1,
            "feature_cols": ["x"],
            "feature_dtype": "float32",
            "seed_order": [0],
            "probability_tie_rule": "p>=0.5_is_UP",
            "implementation_git_sha": source_sha,
            "lifecycle_status": "inactive_shadow_candidate",
        }
        bid = derive_bundle_id(
            spec, resolved_without_id, strategy.name, strategy.read_bytes(), [(model.name, model.read_bytes())]
        )
        resolved = dict(resolved_without_id, bundle_id=bid)
        candidate_manifest = manifest.build_candidate_manifest(
            "SMOKE.m15.v2", repo_root=repo, currency="SMOKE", timeframe="15m",
            side="combined", role="direction", lifecycle_status="inactive_shadow_candidate",
            bundle_id=bid, prereg_id=pid, run_id="2" * 64,
            implementation_git_sha=source_sha, source_script="scripts/m15_book_refresh.py",
            determinism={"seeds": [0], "num_threads": 1, "deterministic": True},
            artifacts=[strategy, model], strategy_json=strategy.name,
            summary="synthetic smoke candidate", metrics={}, created_utc="smoke",
        )
        staged = manifest.stage_candidate(
            candidate_manifest, resolved, [strategy, model], repo_root=repo
        )
        result_dir = repo / "results" / "json"
        result_dir.mkdir(parents=True)
        joint_path = result_dir / f"m15_book_refresh_{'2' * 64}_joint_replay_result.json"
        atomic_json_new(
            joint_path,
            {
                "schema": "m15-book-refresh-joint-replay-result/v1",
                "prereg_id": pid,
                "run_id": "2" * 64,
                "S": ["SMOKE.m15.v2"],
                "statuses": {
                    "SMOKE": {"status": "PROMOTE_TO_SHADOW", "reasons": ["synthetic"]}
                },
                "candidate_statuses": {
                    "SMOKE.m15.v2": {
                        "pair": "SMOKE", "status": "PROMOTE_TO_SHADOW", "reasons": ["synthetic"]
                    }
                },
                "candidate_publish_locks": {
                    "SMOKE.m15.v2": staged["publish_lock"],
                },
            },
        )
        registry = repo / "MODEL_REGISTRY.md"
        registry.write_text("# synthetic registry\n")
        registry_entry = (
            "SMOKE.m15.v2 bundle_id=" + bid + " prereg_id=" + pid
            + " run_id=" + "2" * 64
            + " lifecycle_status=inactive_shadow_candidate; inactive; no activation"
        )
        phase_path = (
            repo / "logs" / "m15_book_refresh" / pid
            / "candidate_publication" / "SMOKE.m15.v2.json"
        )
        phase_path.parent.mkdir(parents=True)
        phase_seal = {
            "schema": "m15-book-refresh-candidate-publication/v1",
            "status": "sealed",
            "book_id": "SMOKE.m15.v2",
            "bundle_id": bid,
            "prereg_id": pid,
            "run_id": "2" * 64,
            "lifecycle_status": "inactive_shadow_candidate",
        }
        manifest.publish_candidate(
            staged["stage_dir"], joint_path, repo_root=repo,
            registry_path=registry, registry_entry=registry_entry,
            phase_seal_path=phase_path, phase_seal=phase_seal,
        )
        destination = books / "SMOKE.m15.v2"
        if not destination.is_dir():
            raise RefreshError("synthetic candidate was not published")
        try:
            from book_runtime import require_active_book_lifecycle
            index_entry = manifest.candidate_index_entry(candidate_manifest)
            require_active_book_lifecycle("SMOKE.m15.v2", index_entry, candidate_manifest)
        except Exception as exc:
            if "inactive" not in str(exc).lower():
                raise RefreshError("inactive candidate did not fail for lifecycle reason") from exc
        else:
            raise RefreshError("inactive candidate passed active lifecycle guard")

    print(
        f"[smoke] PASS spec={spec['schema']} evaluator_id={eid} "
        "synthetic_only=true semantic_outcomes_read=false books_untouched=true",
        flush=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--smoke", action="store_true", help="synthetic/pre-April infrastructure smoke")
    action.add_argument(
        "--null-calibration",
        action="store_true",
        help="run and seal the exact eight-cell synthetic prerequisite",
    )
    action.add_argument("--preregister", action="store_true", help="seal a clean measured preregistration")
    action.add_argument("--validate-spec", action="store_true", help="parse and validate the sealed spec")
    action.add_argument("--build-snapshot", action="store_true", help="build the blind isolated 2026 snapshot")
    action.add_argument("--adapter-parity", action="store_true", help="seal pre-April adapter/policy parity")
    action.add_argument("--fit-one", action="store_true", help="fit one isolated arm attempt")
    action.add_argument("--run-fits", action="store_true", help="fit and repeat every sealed arm serially")
    action.add_argument("--seal-arms", action="store_true", help="seal blind arm behavior and final run identity")
    action.add_argument("--run-replay", action="store_true", help="run the single sealed retrospective look")
    action.add_argument("--lock-results", action="store_true", help="seal statuses and candidate publication locks")
    action.add_argument("--publish-candidates", action="store_true", help="publish only locked inactive survivors")
    action.add_argument("--shadow-spec", action="store_true", help="seal the prospective-shadow handoff")
    parser.add_argument("--prereg-id")
    parser.add_argument("--pair")
    parser.add_argument("--arm")
    parser.add_argument("--attempt")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.smoke:
        smoke()
    elif args.null_calibration:
        run_null_calibration_gate()
    elif args.preregister:
        preregister()
    elif args.validate_spec:
        spec = load_spec()
        print(f"[spec] PASS {spec['schema']} evaluator_id={evaluator_id(spec)}")
    elif args.build_snapshot:
        if not args.prereg_id:
            raise RefreshError("--build-snapshot requires --prereg-id")
        build_blind_snapshot(args.prereg_id)
    elif args.adapter_parity:
        if not args.prereg_id:
            raise RefreshError("--adapter-parity requires --prereg-id")
        run_adapter_parity(args.prereg_id)
    elif args.fit_one:
        if not all((args.prereg_id, args.pair, args.arm, args.attempt)):
            raise RefreshError("--fit-one requires --prereg-id, --pair, --arm, and --attempt")
        fit_one(args.prereg_id, args.pair, args.arm, args.attempt)
    elif args.run_fits:
        if not args.prereg_id:
            raise RefreshError("--run-fits requires --prereg-id")
        run_all_fits(args.prereg_id)
    elif args.seal_arms:
        if not args.prereg_id:
            raise RefreshError("--seal-arms requires --prereg-id")
        seal_arms(args.prereg_id)
    elif args.run_replay:
        if not args.prereg_id:
            raise RefreshError("--run-replay requires --prereg-id")
        run_replay(args.prereg_id)
    elif args.lock_results:
        if not args.prereg_id:
            raise RefreshError("--lock-results requires --prereg-id")
        lock_results_and_candidates(args.prereg_id)
    elif args.publish_candidates:
        if not args.prereg_id:
            raise RefreshError("--publish-candidates requires --prereg-id")
        publish_candidates(args.prereg_id)
    elif args.shadow_spec:
        if not args.prereg_id:
            raise RefreshError("--shadow-spec requires --prereg-id")
        seal_shadow_spec(args.prereg_id)
    else:
        print("No action selected. Use --smoke or --validate-spec for an unmeasured check.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
