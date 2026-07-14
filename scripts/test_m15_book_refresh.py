#!/usr/bin/env python3
"""Standalone, outcome-blind evidence gate for the issue-#9 refresh runner.

The default invocation runs the exact preregistered eight-cell null-calibration
suite (400 campaigns, 1,000 bootstrap draws, block lengths 1/2/3, deterministic
repeat).  ``--bounded-null`` is only a development convenience; its output is
explicitly not an acceptance artifact.

Run from the repository root::

    ~/binary-algo-venv/bin/python scripts/test_m15_book_refresh.py
    ~/binary-algo-venv/bin/python scripts/test_m15_book_refresh.py --bounded-null

Every filesystem mutation is confined to temporary directories.  Fixtures are
synthetic or dated before April 2026 and no replay label/outcome source is read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence
from unittest import mock

import numpy as np
import pandas as pd

import book_runtime
import deriv_floor_resolver
import live_features
import m15_book_refresh as refresh
import m15_book_refresh_adapters as adapters
import m15_book_refresh_replay as replay
import m15_book_refresh_stats as stats
import m15_phase_zero_behavior_authority as phase_zero
import manifest
from min1_production import nonoverlap_chrono as canonical_scheduler
from min1_production import wc_ret as canonical_wc_ret


REPO_ROOT = Path(__file__).resolve().parents[1]
GOLDEN_EVALUATOR_ID = "2e113bbf3d457bf62892f0a4779e510eed6730de846edbfedddf9bf509699124"
GOLDEN_PREREG_ID = "4c849280be2bab341d42093fd0b9f37d3b9cd99a0b8ff1752035bf2dc7a6bf84"
GOLDEN_BUNDLE_ID = "fb1b7e35e5ac1ec0517e753b5853b7a62ebf2282fc4b489e0ef2385d93110ed1"
GOLDEN_CONTROLS_ID = "81e76cea275ba9665863cb4e34fcae2bbb9dbe26b5aa3159c3c28facb73bd0b4"
GOLDEN_RUN_ID = "019328e349336f1215e8a837e7577343e53df65a08b2f2afc6942e8b9df531ee"
GOLDEN_RECEIPT_SHA256 = "1d2b804fddbc1a4a00ca3ae1be2323c5341c8558404c86649cb9ddd770ecf360"
GOLDEN_LOCK_SHA256 = "7caa8f96ef765c201163ab328df89eb97a8b3d4b8d03c1d7f0f30743435e46b2"


class GateFailure(AssertionError):
    """A focused evidence-gate invariant failed."""


class InjectedCrash(RuntimeError):
    pass


def check(condition: Any, message: str) -> None:
    if not bool(condition):
        raise GateFailure(message)


def equal(actual: Any, expected: Any, message: str) -> None:
    if actual != expected:
        raise GateFailure(f"{message}: actual={actual!r} expected={expected!r}")


@contextmanager
def expect_raises(error: type[BaseException], contains: str | None = None) -> Iterator[None]:
    try:
        yield
    except error as exc:
        if contains is not None and contains.lower() not in str(exc).lower():
            raise GateFailure(
                f"{error.__name__} did not contain {contains!r}: {exc}"
            ) from exc
    else:
        raise GateFailure(f"expected {error.__name__}")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_section(name: str, function: Callable[[], None]) -> float:
    started = time.monotonic()
    function()
    elapsed = time.monotonic() - started
    print(f"[test] PASS {name} ({elapsed:.2f}s)", flush=True)
    return elapsed


def test_identity_goldens() -> None:
    spec = refresh.load_spec()
    equal(refresh.evaluator_id(spec), GOLDEN_EVALUATOR_ID, "evaluator golden")

    prereg_payload = {
        "schema": "golden-prereg/v1",
        "falsifier": "synthetic-only",
        "order": [1, 2, 3],
    }
    source_sha = "1" * 40
    prereg_id = refresh.derive_prereg_id(spec, prereg_payload, source_sha)
    equal(prereg_id, GOLDEN_PREREG_ID, "prereg golden")
    changed_payload = dict(prereg_payload, order=[1, 3, 2])
    check(
        refresh.derive_prereg_id(spec, changed_payload, source_sha) != prereg_id,
        "prereg identity ignored ordered payload bytes",
    )
    with expect_raises(refresh.RefreshError, "exclude derived prereg_id"):
        refresh.derive_prereg_id(spec, dict(prereg_payload, prereg_id="0" * 64), source_sha)

    resolved = {
        "schema": "resolved-bundle-spec/v1",
        "book_id": "TEST.m15.v2",
        "prereg_id": "2" * 64,
        "run_id": "3" * 64,
        "coverage": 0.02,
        "threshold": 0.125,
        "feature_cols": ["x", "y2"],
        "feature_dtype": "float32",
        "seed_order": [0, 1],
        "probability_tie_rule": "p>=0.5_is_UP",
        "implementation_git_sha": source_sha,
    }
    models = [("model_s0.txt", b"model-zero\n"), ("model_s1.txt", b"model-one\n")]
    bundle_id = refresh.derive_bundle_id(
        spec, resolved, "strategy.json", b'{"strategy":1}', models
    )
    equal(bundle_id, GOLDEN_BUNDLE_ID, "bundle golden")
    check(
        refresh.derive_bundle_id(
            spec, resolved, "strategy.json", b'{"strategy":1}', list(reversed(models))
        )
        != bundle_id,
        "bundle identity ignored declared model order",
    )

    control_entries = [
        [family, arm, hashlib.sha256(f"control:{index}".encode()).hexdigest()]
        for index, (family, arm) in enumerate(spec["control_order"])
    ]
    controls_id = refresh.derive_controls_id(spec, control_entries)
    equal(controls_id, GOLDEN_CONTROLS_ID, "controls golden")
    with expect_raises(refresh.RefreshError, "order"):
        refresh.derive_controls_id(spec, list(reversed(control_entries)))
    for mutation in ("truncate", "reorder", "substitute", "representative"):
        mutated_spec = json.loads(json.dumps(spec))
        if mutation == "truncate":
            mutated_spec["control_order"] = mutated_spec["control_order"][:-2]
        elif mutation == "reorder":
            mutated_spec["control_order"][0], mutated_spec["control_order"][1] = (
                mutated_spec["control_order"][1],
                mutated_spec["control_order"][0],
            )
        elif mutation == "substitute":
            mutated_spec["control_order"][-1] = ["own_pair", "B_perm"]
        else:
            mutated_spec["control_family"]["representatives"]["own_pair"] = "AUDUSD"
        with tempfile.TemporaryDirectory(prefix=f"m15-control-{mutation}-") as raw:
            mutated_path = Path(raw) / "spec.json"
            mutated_path.write_text(json.dumps(mutated_spec), encoding="utf-8")
            with expect_raises(refresh.RefreshError, "control"):
                refresh.load_spec(mutated_path)
        with expect_raises(refresh.RefreshError, "control identity"):
            refresh.derive_controls_id(mutated_spec, control_entries)

    with mock.patch.object(stats, "NULL_TEST_SEED", 9006):
        with expect_raises(refresh.RefreshError, "executable statistics/null"):
            refresh.load_spec()
    with mock.patch.object(
        stats, "NULL_HELPER_CODES", {"main": 1, "control": 2, "shadow": 4}
    ):
        with expect_raises(refresh.RefreshError, "executable statistics/null"):
            refresh.load_spec()
    with mock.patch.object(stats, "PAIR_ORDER", tuple(reversed(stats.PAIR_ORDER))):
        with expect_raises(refresh.RefreshError, "executable statistics/null"):
            refresh.load_spec()
    mutated_minutes = json.loads(json.dumps(spec))
    mutated_minutes["null_calibration"]["row_minutes"] = (
        "09:00_through_16:59_America/New_York"
    )
    with tempfile.TemporaryDirectory(prefix="m15-null-row-minutes-") as raw:
        mutated_path = Path(raw) / "spec.json"
        mutated_path.write_text(json.dumps(mutated_minutes), encoding="utf-8")
        with expect_raises(refresh.RefreshError, "row-minute"):
            refresh.load_spec(mutated_path)

    arm_keys = [(pair, arm) for pair in spec["pair_order"] for arm in spec["arm_order"]]
    arm_entries = [
        [pair, arm, hashlib.sha256(f"arm:{index}".encode()).hexdigest()]
        for index, (pair, arm) in enumerate(arm_keys)
    ]
    run_id = refresh.derive_run_id(spec, "2" * 64, controls_id, arm_entries)
    equal(run_id, GOLDEN_RUN_ID, "run golden")
    with expect_raises(refresh.RefreshError, "order"):
        refresh.derive_run_id(spec, "2" * 64, controls_id, list(reversed(arm_entries)))
    with expect_raises(refresh.RefreshError):
        refresh.derive_run_id(spec, "2" * 64, controls_id, arm_entries[:-1])
    with expect_raises(refresh.RefreshError):
        refresh.derive_run_id(
            spec,
            "2" * 64,
            controls_id,
            arm_entries
            + [["USDCAD", "A", hashlib.sha256(b"excluded-arm").hexdigest()]],
        )

    equal(
        refresh.canonical_bytes({"b": 1, "a": "µ"}),
        b'{"a":"\xc2\xb5","b":1}',
        "canonical JSON bytes",
    )
    with expect_raises(manifest.CandidatePublicationError, "not canonical-JSON encodable"):
        refresh.canonical_bytes({"not_finite": float("nan")})


def test_target_and_feature_source_contract() -> None:
    """Campaign membership must never shrink the cross-pair source universe."""

    targets = (
        "EURUSD",
        "USDJPY",
        "GBPUSD",
        "USDCHF",
        "AUDUSD",
        "NZDUSD",
    )
    sources = (
        "EURUSD",
        "GBPUSD",
        "AUDUSD",
        "NZDUSD",
        "USDJPY",
        "USDCHF",
        "USDCAD",
    )
    spec = refresh.load_spec()
    equal(tuple(spec["pair_order"]), targets, "exact six-target order")
    equal(
        tuple(spec["feature_source_pair_order"]),
        sources,
        "exact seven-source order",
    )
    equal(tuple(spec["pairs"]), targets, "target contract key order")
    equal(tuple(stats.PAIR_ORDER), targets, "statistics target order")
    equal(tuple(adapters.ALL_PAIRS), sources, "adapter source universe")
    equal(set(adapters.CONTRACTS), set(sources), "adapter source-contract set")
    equal(tuple(live_features.PAIRS), sources, "public provider source universe")
    check("USDCAD" not in spec["pairs"], "USDCAD retained a campaign contract")
    check(
        book_runtime.TARGET_BOOKS["USDCAD"] == "USDCAD.m15ny_seedens.v1"
        and "USDCAD.m15ny_seedens.v1" in book_runtime.LEGACY_ACTIVE_BOOK_IDS,
        "six-target relaunch shrank the production USDCAD contract",
    )


def test_loaded_book_batch_behavior() -> None:
    """Bind batch/scalar policy behavior without binding reduction-layout bits."""

    books = book_runtime.load_target_books()
    equal(tuple(books), tuple(book_runtime.TARGET_BOOKS), "loaded-book order")
    for pair, book in books.items():
        rows: list[np.ndarray] = []
        for row_index in range(3):
            values = np.asarray(
                [
                    (((column_index + 1) * 37 + row_index * 19) % 211 - 105)
                    / 31.0
                    for column_index in range(len(book.feature_cols))
                ],
                dtype="float64",
            )
            rows.append(values)
        frame = pd.DataFrame(rows, columns=book.feature_cols, dtype="float64")
        legacy_probabilities: list[float] = []
        for values in rows:
            singleton = pd.DataFrame([values], columns=book.feature_cols)
            predictions = np.array(
                [float(booster.predict(singleton)[0]) for booster in book.boosters],
                dtype=float,
            )
            legacy_probabilities.append(float(predictions.mean()))
        legacy = np.asarray(legacy_probabilities, dtype="float64")
        batch = book.predict_probabilities(frame)
        # Batch/singleton probability bits are diagnostic only; the exact
        # scalar delegation and all discrete policy behavior remain binding.
        components = book.gate_components(frame, batch)
        for position, values in enumerate(rows):
            row = pd.Series(values, index=book.feature_cols, dtype="float64")
            score = book.score(row)
            check(
                np.float64(score.proba).view("uint64")
                == legacy[position].view("uint64"),
                f"{pair}: scalar scorer no longer delegates bitwise to batch",
            )
            legacy_reasons: list[str] = []
            confidence = abs(legacy[position] - 0.5)
            if confidence < book.conf_thr:
                legacy_reasons.append(
                    f"confidence {confidence:.8f} < threshold {book.conf_thr:.8f}"
                )
            bb_threshold = book_runtime._first_number(
                book.strategy, ("bb_width_thr", "bbw_thr")
            )
            strategy_gate = book.strategy.get("gate")
            if bb_threshold is None and isinstance(strategy_gate, dict):
                bb_threshold = book_runtime._first_number(
                    strategy_gate, ("bb_width_thr", "bbw_thr")
                )
            if bb_threshold is not None:
                if "15m_bb_width" not in row.index:
                    legacy_reasons.append(
                        "missing 15m_bb_width for strategy compression gate"
                    )
                elif float(row["15m_bb_width"]) > bb_threshold:
                    legacy_reasons.append(
                        f"15m_bb_width {float(row['15m_bb_width']):.8f} > "
                        f"threshold {bb_threshold:.8f}"
                    )
            equal(score.gate_reasons, legacy_reasons, f"{pair}: scalar gate reasons")
            equal(
                score.gate_passed,
                bool(components.gate_passed[position]),
                f"{pair}: scalar/batch combined gate",
            )
            equal(
                score.direction,
                "UP" if legacy[position] >= 0.5 else "DOWN",
                f"{pair}: direction tie rule",
            )

    class ScalarFixtureBook:
        feature_cols = ["p", "15m_bb_width", "reason_marker"]
        book_id = "TEST.scalar-parity.v1"
        pair = "TEST"
        strategy = {"bb_width_thr": 0.5}
        conf_thr = 0.25

        def __init__(
            self,
            toward: float | None,
            *,
            forced_direction: str | None = None,
            force_gate_passed: bool | None = None,
            singleton_override: Any = None,
        ) -> None:
            self.toward = toward
            self.forced_direction = forced_direction
            self.force_gate_passed = force_gate_passed
            self.singleton_override = singleton_override

        def gate_components(
            self, frame: pd.DataFrame, probabilities: np.ndarray
        ) -> Any:
            if self.singleton_override is not None and len(frame) == 1:
                return self.singleton_override
            p = np.asarray(probabilities, dtype="float64")
            confidence = ~(np.abs(p - 0.5) < self.conf_thr)
            structural = ~(
                frame["15m_bb_width"].to_numpy(dtype="float64") > 0.5
            )
            return book_runtime.BookGateComponents(confidence, structural)

        def score(self, feature_row: pd.Series) -> book_runtime.BookScore:
            row = pd.Series(feature_row, dtype="float64")
            probability = np.float64(row["p"])
            if self.toward is not None:
                probability = np.nextafter(
                    probability, np.float64(self.toward)
                )
            confidence_passed = not (
                abs(float(probability) - 0.5) < self.conf_thr
            )
            structural_passed = not (float(row["15m_bb_width"]) > 0.5)
            reasons: list[str] = []
            if not confidence_passed:
                reasons.append("confidence synthetic")
            if not structural_passed:
                marker = " marker" if row["reason_marker"] > 0 else ""
                reasons.append(f"15m_bb_width synthetic{marker}")
            direction = "UP" if probability >= 0.5 else "DOWN"
            if self.forced_direction is not None:
                direction = self.forced_direction
            gate_passed = not reasons
            if self.force_gate_passed is not None:
                gate_passed = self.force_gate_passed
            return book_runtime.BookScore(
                pair=self.pair,
                book_id=self.book_id,
                proba=float(probability),
                confidence=float(abs(probability - 0.5)),
                threshold=self.conf_thr,
                direction=direction,
                gate_passed=gate_passed,
                gate_reasons=reasons,
            )

    def scalar_report(
        fixture_book: ScalarFixtureBook,
        cached_matrix: np.ndarray,
        rebuilt_matrix: np.ndarray,
        cached_probability: np.ndarray,
        rebuilt_probability: np.ndarray,
        *,
        cached_components: book_runtime.BookGateComponents | None = None,
        rebuilt_components: book_runtime.BookGateComponents | None = None,
    ) -> dict[str, Any]:
        if cached_components is None:
            cached_components = fixture_book.gate_components(
                pd.DataFrame(cached_matrix, columns=fixture_book.feature_cols),
                cached_probability,
            )
        if rebuilt_components is None:
            rebuilt_components = fixture_book.gate_components(
                pd.DataFrame(rebuilt_matrix, columns=fixture_book.feature_cols),
                rebuilt_probability,
            )
        return refresh._loaded_book_scalar_parity_report(
            fixture_book,
            cached_matrix,
            rebuilt_matrix,
            cached_probability,
            rebuilt_probability,
            cached_components,
            rebuilt_components,
        )

    class RawComponents:
        def __init__(self, confidence: Any, structural: Any, combined: Any) -> None:
            self.confidence_passed = np.asarray(confidence)
            self.structural_passed = np.asarray(structural)
            self.gate_passed = np.asarray(combined)

    benign_matrix = np.asarray(
        [[0.6, 0.4, 0.0], [0.4, 0.4, 0.0], [0.9, 0.4, 0.0]],
        dtype="float64",
    )
    benign_probability = benign_matrix[:, 0].copy()
    benign = scalar_report(
        ScalarFixtureBook(1.0),
        benign_matrix,
        benign_matrix.copy(),
        benign_probability,
        benign_probability.copy(),
    )
    check(
        benign["scalar_batch_probability_bits_exact_diagnostic"] is False
        and benign["scalar_batch_probability_bits_binding"] is False
        and benign["behavior_exact"] is True,
        "nonbinding scalar/batch ULP drift did not preserve exact behavior",
    )

    changed_rebuilt = benign_matrix.copy()
    changed_rebuilt[0, 0] = np.nextafter(changed_rebuilt[0, 0], 1.0)
    changed_scalar = scalar_report(
        ScalarFixtureBook(None),
        benign_matrix,
        changed_rebuilt,
        benign_probability,
        benign_probability.copy(),
    )
    check(
        changed_scalar["cached_raw_probability_bits_exact"] is False
        and changed_scalar["behavior_exact"] is False,
        "cached/rebuilt scalar probability-bit drift did not fail",
    )

    direction_matrix = np.asarray([[0.5, 0.4, 0.0]], dtype="float64")
    direction_probability = np.asarray([0.5], dtype="float64")
    direction_crossing = scalar_report(
        ScalarFixtureBook(float("-inf")),
        direction_matrix,
        direction_matrix.copy(),
        direction_probability,
        direction_probability.copy(),
    )
    check(
        direction_crossing["scalar_batch_direction_exact"] is False
        and direction_crossing["behavior_exact"] is False,
        "scalar/batch direction crossing did not fail",
    )
    with expect_raises(refresh.RefreshError, "score is inconsistent"):
        scalar_report(
            ScalarFixtureBook(
                float("-inf"), forced_direction="UP"
            ),
            direction_matrix,
            direction_matrix.copy(),
            direction_probability,
            direction_probability.copy(),
        )

    confidence_matrix = np.asarray([[0.75, 0.6, 0.0]], dtype="float64")
    confidence_probability = np.asarray([0.75], dtype="float64")
    confidence_crossing = scalar_report(
        ScalarFixtureBook(0.5),
        confidence_matrix,
        confidence_matrix.copy(),
        confidence_probability,
        confidence_probability.copy(),
    )
    check(
        confidence_crossing["scalar_batch_confidence_gate_exact"] is False
        and confidence_crossing["scalar_batch_combined_gate_exact"] is True
        and confidence_crossing["behavior_exact"] is False,
        "hidden scalar/batch confidence crossing did not fail",
    )

    structural_matrix = np.asarray([[0.6, 0.4, 0.0]], dtype="float64")
    structural_probability = np.asarray([0.6], dtype="float64")
    hidden_structural = book_runtime.BookGateComponents(
        np.asarray([False]), np.asarray([False])
    )
    structural_crossing = scalar_report(
        ScalarFixtureBook(None),
        structural_matrix,
        structural_matrix.copy(),
        structural_probability,
        structural_probability.copy(),
        cached_components=hidden_structural,
        rebuilt_components=hidden_structural,
    )
    check(
        structural_crossing["scalar_batch_structural_gate_exact"] is False
        and structural_crossing["scalar_batch_combined_gate_exact"] is True
        and structural_crossing["behavior_exact"] is False,
        "hidden scalar/batch structural crossing did not fail",
    )

    reason_cached = np.asarray([[0.6, 0.6, 0.0]], dtype="float64")
    reason_rebuilt = np.asarray([[0.6, 0.6, 1.0]], dtype="float64")
    reason_probability = np.asarray([0.6], dtype="float64")
    reason_crossing = scalar_report(
        ScalarFixtureBook(None),
        reason_cached,
        reason_rebuilt,
        reason_probability,
        reason_probability.copy(),
    )
    check(
        reason_crossing["cached_raw_gate_exact"] is False
        and reason_crossing["behavior_exact"] is False,
        "cached/rebuilt scalar gate-reason drift did not fail",
    )
    with expect_raises(refresh.RefreshError, "score is inconsistent"):
        scalar_report(
            ScalarFixtureBook(None, force_gate_passed=True),
            reason_cached,
            reason_cached.copy(),
            reason_probability,
            reason_probability.copy(),
        )

    empty_components = book_runtime.BookGateComponents(
        np.asarray([], dtype=bool), np.asarray([], dtype=bool)
    )
    with expect_raises(refresh.RefreshError, "inputs are malformed"):
        refresh._loaded_book_scalar_parity_report(
            ScalarFixtureBook(None),
            np.empty((0, 3), dtype="float64"),
            np.empty((0, 3), dtype="float64"),
            np.empty(0, dtype="float64"),
            np.empty(0, dtype="float64"),
            empty_components,
            empty_components,
        )
    benign_components = ScalarFixtureBook(None).gate_components(
        pd.DataFrame(benign_matrix, columns=ScalarFixtureBook.feature_cols),
        benign_probability,
    )
    with expect_raises(refresh.RefreshError, "inputs are malformed"):
        refresh._loaded_book_scalar_parity_report(
            ScalarFixtureBook(None),
            benign_matrix,
            benign_matrix.copy(),
            np.asarray(0.6),
            benign_probability.copy(),
            benign_components,
            benign_components,
        )
    for malformed_singleton in (
        RawComponents([1], [1], [1]),
        RawComponents([True, False], [True, False], [True, False]),
        RawComponents([True], [True], [False]),
    ):
        with expect_raises(refresh.RefreshError, "components are malformed"):
            scalar_report(
                ScalarFixtureBook(
                    None, singleton_override=malformed_singleton
                ),
                structural_matrix,
                structural_matrix.copy(),
                structural_probability,
                structural_probability.copy(),
                cached_components=ScalarFixtureBook(None).gate_components(
                    pd.DataFrame(
                        structural_matrix,
                        columns=ScalarFixtureBook.feature_cols,
                    ),
                    structural_probability,
                ),
                rebuilt_components=ScalarFixtureBook(None).gate_components(
                    pd.DataFrame(
                        structural_matrix,
                        columns=ScalarFixtureBook.feature_cols,
                    ),
                    structural_probability,
                ),
            )
    inconsistent_full = RawComponents([False], [False], [True])
    with expect_raises(refresh.RefreshError, "components are malformed"):
        scalar_report(
            ScalarFixtureBook(None),
            structural_matrix,
            structural_matrix.copy(),
            structural_probability,
            structural_probability.copy(),
            cached_components=inconsistent_full,
        )
    for malformed_probability in (
        benign_probability.astype("float32"),
        benign_probability.astype("int64"),
    ):
        with expect_raises(refresh.RefreshError, "inputs are malformed"):
            refresh._loaded_book_scalar_parity_report(
                ScalarFixtureBook(None),
                benign_matrix,
                benign_matrix.copy(),
                malformed_probability,
                benign_probability.copy(),
                benign_components,
                benign_components,
            )
    with expect_raises(refresh.RefreshError, "inputs are malformed"):
        refresh._loaded_book_scalar_parity_report(
            ScalarFixtureBook(None),
            benign_matrix.astype("int64"),
            benign_matrix.copy(),
            benign_probability,
            benign_probability.copy(),
            benign_components,
            benign_components,
        )
    nonfinite_matrix = benign_matrix.copy()
    nonfinite_matrix[0, 0] = np.nan
    with expect_raises(refresh.RefreshError, "inputs are malformed"):
        refresh._loaded_book_scalar_parity_report(
            ScalarFixtureBook(None),
            nonfinite_matrix,
            benign_matrix.copy(),
            benign_probability,
            benign_probability.copy(),
            benign_components,
            benign_components,
        )

    # A combined-mask comparison alone cannot see these two policy crossings:
    # cached passes confidence/fails structure, rebuilt does the reverse.
    synthetic = book_runtime.LoadedBook(
        pair="TEST",
        book_id="TEST.m15.v1",
        book_dir=Path("."),
        strategy_path=Path("strategy.json"),
        model_paths=[Path("model.txt")],
        strategy={"bb_width_thr": 0.5},
        feature_cols=["x", "15m_bb_width"],
        boosters=[],
        conf_thr=0.2,
        content_id="synthetic",
    )
    cached = synthetic.gate_components(
        pd.DataFrame({"x": [1.0], "15m_bb_width": [0.6]}),
        np.array([0.8], dtype="float64"),
    )
    rebuilt = synthetic.gate_components(
        pd.DataFrame({"x": [1.0], "15m_bb_width": [0.4]}),
        np.array([0.55], dtype="float64"),
    )
    check(
        np.array_equal(cached.gate_passed, rebuilt.gate_passed),
        "component-crossing fixture did not hide behind an equal combined mask",
    )
    check(
        not np.array_equal(cached.confidence_passed, rebuilt.confidence_passed)
        and not np.array_equal(cached.structural_passed, rebuilt.structural_passed),
        "separate gate components failed to expose hidden crossings",
    )


def test_canonical_helpers_and_boundaries() -> None:
    base = int(pd.Timestamp("2025-01-15T12:00:00Z").timestamp())
    observations = base + np.arange(0, 2_001, 10, dtype="int64")
    closes = 1.0 + np.arange(len(observations), dtype="float64") * 1e-5
    direct_ret, direct_valid = canonical_wc_ret(observations, closes, 900, 10, 1)
    wrapped_ret, wrapped_valid = stats.canonical_wc_ret(observations, closes)
    check(np.array_equal(direct_valid, wrapped_valid), "wc_ret validity wrapper drift")
    check(
        np.array_equal(direct_ret, wrapped_ret, equal_nan=True),
        "wc_ret value wrapper drift",
    )
    decisions = observations[::6]
    subset_ret, subset_valid = stats.canonical_settlement_on_decision_grid(
        decisions, observations, closes
    )
    positions = np.searchsorted(observations, decisions)
    check(np.array_equal(subset_valid, direct_valid[positions]), "settlement subset mask drift")
    check(
        np.array_equal(subset_ret, direct_ret[positions], equal_nan=True),
        "settlement subset values drift",
    )
    detail_ret, detail_valid, detail_exit = (
        stats.canonical_settlement_with_exit_on_decision_grid(
            decisions, observations, closes
        )
    )
    check(np.array_equal(detail_valid, subset_valid), "settlement-detail mask drift")
    check(
        np.array_equal(detail_ret, subset_ret, equal_nan=True),
        "settlement-detail values drift",
    )
    check(
        np.array_equal(detail_exit[detail_valid], decisions[detail_valid] + 900),
        "canonical actual exits are not the last 10s observations before +901s",
    )

    # A close that is available only after the canonical exit must not affect
    # the settled return.  On this right-edge 10-second clock the decision at
    # ``base`` enters at +10s and exits at +900s; +910s is lookahead.
    guarded_closes = np.ones(len(observations), dtype="float64")
    guarded_closes[np.searchsorted(observations, base + 900)] = 1.25
    guarded_closes[np.searchsorted(observations, base + 910)] = 9.0
    guarded_ret, guarded_valid, guarded_exit = (
        stats.canonical_settlement_with_exit_on_decision_grid(
            np.array([base], dtype="int64"), observations, guarded_closes
        )
    )
    check(guarded_valid[0], "synthetic canonical settlement unexpectedly invalid")
    equal(int(guarded_exit[0]), base + 900, "canonical exit off by one 10s bar")
    check(
        np.isclose(guarded_ret[0], 0.25),
        "post-exit close leaked into canonical settlement return",
    )
    with expect_raises(stats.StatisticsInputError, "exact subset"):
        stats.canonical_settlement_on_decision_grid(
            np.concatenate(([decisions[0] - 60], decisions)), observations, closes
        )
    with expect_raises(stats.StatisticsInputError, "completed-minute boundaries"):
        stats.canonical_settlement_on_decision_grid(
            decisions + 1, observations, closes
        )

    # Exercise replay's strict actual-exit boundary directly.  The regular
    # right-edge clock gives exits at boundary-60s, boundary, and boundary+60s;
    # only the first may enter results.  Runtime and model scoring are mocked
    # because this fixture isolates clock semantics and contains no outcomes
    # from the measured April-May replay.
    spec = refresh.load_spec()
    boundary_s = int(pd.Timestamp(spec["splits"]["replay"]["settlement_exit_before"]).timestamp())
    replay_decisions_s = boundary_s + np.array([-960, -900, -840], dtype="int64")
    replay_entries_ns = replay_decisions_s * np.int64(1_000_000_000)
    feature_cols = tuple(f"fixture_{index}" for index in range(239))
    score_rows = adapters.ScoreRows(
        pair="USDJPY",
        feature_cols=feature_cols,
        X=np.ones((3, len(feature_cols)), dtype="float32"),
        entry_ns=replay_entries_ns,
        source_feature_ns=replay_entries_ns - np.int64(60 * 1_000_000_000),
    )
    policies = {
        arm: replay.ArmPolicy(
            arm=arm,
            bundle_id=hashlib.sha256(f"boundary:{arm}".encode()).hexdigest(),
            feature_cols=feature_cols,
            model_paths=(Path(f"unused-{arm}.txt"),),
            model_sha256=(hashlib.sha256(f"unused-{arm}".encode()).hexdigest(),),
            confidence_threshold=0.0,
            target_coverage=0.02,
        )
        for arm in replay.PRIMARY_ARMS
    }

    class BoundaryBook:
        def __init__(self) -> None:
            self.pair = "USDJPY"
            self.book_id = "USDJPY.synthetic.boundary.v1"
            self.feature_cols = list(feature_cols)
            self.model_paths = list(policies["A"].model_paths)
            self.conf_thr = 0.0
            self.strategy: dict[str, Any] = {}

        @staticmethod
        def predict_probabilities(frame: pd.DataFrame) -> np.ndarray:
            return np.full(len(frame), 0.9, dtype="float64")

        @staticmethod
        def gate_components(
            frame: pd.DataFrame, _probabilities: np.ndarray
        ) -> book_runtime.BookGateComponents:
            return book_runtime.BookGateComponents(
                np.ones(len(frame), dtype=bool),
                np.ones(len(frame), dtype=bool),
            )

    boundary_book = BoundaryBook()
    with mock.patch.object(
        replay,
        "_policy_model_identity",
        return_value=(1, 2, 3, 4, policies["A"].model_sha256[0]),
    ):
        a_probability, a_confidence, a_structural, a_selected = (
            replay._score_comparator_a_canonical(
                score_rows, policies["A"], boundary_book
            )
        )
    boundary_a_repeat = replay.ComparatorARepeatability.from_mapping(
        replay._comparator_a_replay_record(
            score_rows,
            policies["A"],
            boundary_book,
            a_probability,
            a_confidence,
            a_structural,
            a_selected,
        )
    )
    replay_input = replay.PairReplayInput(
        pair="USDJPY",
        score_rows=score_rows,
        observation_paths=(Path("unused-observation.parquet"),),
        primary_policies=policies,
        comparator_a_book=boundary_book,
        comparator_a_repeatability=boundary_a_repeat,
    )
    boundary_observations = np.arange(
        boundary_s - 1_000, boundary_s + 1_001, 10, dtype="int64"
    )
    boundary_closes = 1.0 + np.arange(len(boundary_observations)) * 1e-6
    observation = replay.ObservationSeries(
        boundary_observations,
        boundary_closes,
        ({"fixture": "synthetic-pre-April-clock"},),
    )
    with (
        mock.patch.object(
            replay,
            "canonical_runtime_ny_mask",
            side_effect=lambda values: np.ones(len(values), dtype=bool),
        ),
        mock.patch.object(
            replay,
            "score_policy",
            side_effect=lambda rows, _policy: (
                np.full(len(rows), 0.9, dtype="float64"),
                np.ones(len(rows), dtype=bool),
            ),
        ),
        mock.patch.object(
            replay,
            "_policy_model_identity",
            return_value=(1, 2, 3, 4, policies["A"].model_sha256[0]),
        ),
    ):
        boundary_result = replay.evaluate_pair(
            spec, replay_input, observation=observation
        )
    check(
        np.array_equal(boundary_result.timestamps_s, replay_decisions_s[:1]),
        "replay did not exclude actual exits equal to/after settlement_end",
    )
    boundary_grid = boundary_result.report["grid"]
    equal(boundary_grid["canonical_settlement_valid_rows"], 3, "boundary valid count")
    equal(boundary_grid["settlement_boundary_rejected_rows"], 2, "boundary rejection count")
    equal(boundary_grid["settlement_valid_rows"], 1, "bounded settlement count")
    check(boundary_grid["all_actual_exits_before_boundary"], "exit-boundary evidence false")

    # A single probability-bit drift against the arm-seal A record must fail
    # before the observation loader can open any close value.
    drifted_repeat = replace(
        boundary_a_repeat,
        probability_f64le_sha256="f" * 64,
    )
    drifted_input = replace(
        replay_input,
        comparator_a_repeatability=drifted_repeat,
    )
    observation_calls: list[str] = []

    def forbidden_observation(*_args: Any, **_kwargs: Any) -> replay.ObservationSeries:
        observation_calls.append("opened")
        raise GateFailure("observation close opened before comparator-A validation")

    with mock.patch.object(
        replay,
        "canonical_runtime_ny_mask",
        side_effect=lambda values: np.ones(len(values), dtype=bool),
    ), mock.patch.object(
        replay, "load_observation_series", side_effect=forbidden_observation
    ), mock.patch.object(
        replay,
        "_policy_model_identity",
        return_value=(1, 2, 3, 4, policies["A"].model_sha256[0]),
    ), expect_raises(replay.ReplayInputError, "arm-seal evidence"):
        replay.evaluate_pair(spec, drifted_input)
    equal(observation_calls, [], "A drift opened observation closes")

    with mock.patch.object(
        stats,
        "canonical_runtime_ny_mask",
        side_effect=lambda values: np.ones(len(values), dtype=bool),
    ):
        blind_positions, _blind_seconds = refresh._blind_score_window(score_rows)
    check(
        np.array_equal(blind_positions, np.arange(3)),
        "blind arm seal is not the full entry/runtime superset",
    )

    schedule_ts = base + np.array([0, 300, 899, 900, 1_200, 1_800], dtype="int64")
    schedule_mask = np.array([True, True, True, True, True, True])
    direct_take = canonical_scheduler(schedule_ts, schedule_mask, 900)
    wrapped_take = stats.canonical_nonoverlap_chrono(schedule_ts, schedule_mask)
    check(np.array_equal(direct_take, wrapped_take), "canonical scheduler wrapper drift")
    check(np.array_equal(wrapped_take, np.array([0, 3, 5])), "first-come schedule changed")

    arm = stats.scheduled_arm(
        schedule_ts,
        np.array([0.5, 0.9, 0.1, 0.8, 0.2, 0.7]),
        np.zeros(len(schedule_ts)),
        threshold=0.0,
    )
    check(arm.direction_up[0], "p=0.5 did not resolve to UP")
    check(not arm.correct.any(), "settlement ties were not charged as losses")
    check(np.all(arm.utility[arm.selected_mask] == -1.0), "tie utility was not -1")

    local_strings = [
        "2025-01-15 08:00:00",
        "2025-01-15 16:34:00",
        "2025-01-15 16:35:00",
        "2025-01-15 17:00:00",
        "2025-07-15 08:00:00",
        "2025-07-15 16:34:00",
        "2025-07-15 16:35:00",
        "2025-07-15 17:00:00",
        "2025-07-19 10:00:00",
    ]
    local = pd.DatetimeIndex(
        [pd.Timestamp(value, tz="America/New_York") for value in local_strings]
    )
    runtime_ts = (
        local.tz_convert("UTC").as_unit("ns").asi8 // 1_000_000_000
    ).astype("int64")
    runtime_mask = stats.canonical_runtime_ny_mask(runtime_ts)
    check(
        np.array_equal(
            runtime_mask,
            np.array([True, True, False, False, True, True, False, False, False]),
        ),
        "DST/session/16:35/weekend runtime boundary changed",
    )
    equal(local[0].tz_convert("UTC").hour, 13, "winter NY open UTC hour")
    equal(local[4].tz_convert("UTC").hour, 12, "summer NY open UTC hour")

    seconds = np.array([base, base + 1], dtype="int64")
    nanoseconds = stats.epoch_seconds_to_nanoseconds(seconds)
    check(
        np.array_equal(stats.epoch_nanoseconds_to_seconds(nanoseconds), seconds),
        "exact seconds/ns round trip failed",
    )
    with expect_raises(stats.StatisticsInputError, "duplicate"):
        stats.validate_epoch_seconds(np.array([base, base], dtype="int64"))
    with expect_raises(stats.StatisticsInputError, "strictly increasing"):
        stats.validate_epoch_seconds(np.array([base + 1, base], dtype="int64"))
    with expect_raises(stats.StatisticsInputError, "out-of-range timestamp"):
        stats.validate_epoch_seconds(np.array([2**63], dtype="uint64"))
    with expect_raises(stats.StatisticsInputError, "exactly representable"):
        stats.epoch_nanoseconds_to_seconds(nanoseconds + 1)


def _synthetic_base_columns() -> tuple[str, ...]:
    return tuple(f"1m_synthetic_{index:03d}" for index in range(239))


def _synthetic_contract_columns(pair: str) -> tuple[str, ...]:
    contract = adapters.contract_for(pair)
    prefix: tuple[str, ...] = ()
    if contract.structural_column is not None:
        prefix = (contract.structural_column,)
    return prefix + tuple(
        f"{pair.lower()}_synthetic_{index:03d}"
        for index in range(contract.n_features - len(prefix))
    )


def _synthetic_adapter_rows(
    pair: str,
    *,
    purpose: str,
    entry_ns: np.ndarray,
    moved: np.ndarray,
    y: np.ndarray,
    fit_session: np.ndarray,
    calibration_session: np.ndarray | None = None,
    structural_values: np.ndarray | None = None,
) -> adapters.AdapterRows:
    columns = _synthetic_contract_columns(pair)
    matrix = np.zeros((len(entry_ns), len(columns)), dtype="float32")
    contract = adapters.contract_for(pair)
    if structural_values is not None:
        check(contract.structural_column is not None, f"{pair} has no structural column")
        column = columns.index(contract.structural_column)
        matrix[:, column] = np.asarray(structural_values, dtype="float32")
    if calibration_session is None:
        calibration_session = fit_session
    return adapters.AdapterRows(
        pair=pair,
        purpose=purpose,
        feature_cols=columns,
        X=matrix,
        y=np.asarray(y, dtype="uint8"),
        moved=np.asarray(moved, dtype=bool),
        fit_row_id=np.asarray(entry_ns, dtype="int64"),
        label_exit_ns=np.asarray(entry_ns, dtype="int64")
        + adapters.HORIZON_SECONDS * adapters.NS_PER_SECOND,
        fit_session_mask=np.asarray(fit_session, dtype=bool),
        calibration_session_mask=np.asarray(calibration_session, dtype=bool),
        replay_session_mask=np.asarray(fit_session, dtype=bool),
        source_feature_ns=np.asarray(entry_ns, dtype="int64")
        - adapters.FEATURE_DECISION_SHIFT_SECONDS * adapters.NS_PER_SECOND,
        builder_stride=(contract.fit_stride if purpose == "fit" else 1),
    )


def _write_frame(path: Path, index: pd.DatetimeIndex, values: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(values, index=index).to_parquet(path)


def test_adapters_cap_and_import_safety() -> None:
    spec = refresh.load_spec()
    for pair in spec["pair_order"]:
        contract = adapters.CONTRACTS[pair]
        row = spec["pairs"][pair]
        equal(contract.family, row["adapter_family"], f"{pair} adapter family")
        equal(contract.n_features, row["feature_count"], f"{pair} feature count")
        equal(contract.fit_stride, row["training_stride"], f"{pair} stride")
        equal(contract.stride_phase, row["stride_phase"], f"{pair} stride phase")
        equal(list(contract.seeds), row["seed_order"], f"{pair} seed order")
        equal(contract.target_coverage, row["calibration_target_coverage"], f"{pair} coverage")
        equal(contract.fit_session, row["fit_session"], f"{pair} fit session")
        equal(
            contract.calibration_session,
            row["calibration_session"],
            f"{pair} calibration session",
        )
        equal(contract.fit_ties, row["fit_ties"], f"{pair} fit ties")
        equal(
            contract.early_stopping_ties,
            row["early_stopping_ties"],
            f"{pair} early-stopping ties",
        )
        equal(contract.feature_dtype, row["feature_dtype"], f"{pair} feature dtype")
        equal(contract.num_leaves, row["model"]["num_leaves"], f"{pair} leaves")
        equal(contract.max_rounds, row["model"]["n_estimators"], f"{pair} rounds")
        equal(
            contract.early_stopping_rounds,
            row["model"]["early_stopping_rounds"],
            f"{pair} early-stopping rounds",
        )
        structural = row["structural_gate"]
        if structural["type"] == "none":
            check(
                contract.structural_column is None
                and contract.structural_quantile is None,
                f"{pair} unexpected structural gate",
            )
        else:
            equal(
                contract.structural_column,
                structural["column"],
                f"{pair} structural column",
            )
            equal(
                contract.structural_quantile,
                structural["quantile"],
                f"{pair} structural quantile",
            )
        parameters = adapters.model_parameters(pair, contract.seeds[0])
        equal(parameters["n_jobs"], 16, f"{pair} thread count")
        check(parameters["deterministic"] and parameters["force_col_wise"], f"{pair} determinism")

    check(np.array_equal(adapters.capped_fit_ranks(5, cap=5), np.arange(5)), "uncapped ranks")
    capped = adapters.capped_fit_ranks(150_001)
    equal(len(capped), 150_000, "cap size")
    equal(int(capped[0]), 0, "cap first rank")
    equal(int(capped[-1]), 150_000, "cap last rank")
    check(np.all(np.diff(capped) > 0), "cap ranks are not unique/increasing")
    check(
        np.array_equal(capped, adapters.timestamp_cap_selector(150_001)),
        "timestamp cap alias drift",
    )

    columns = _synthetic_base_columns()
    # The prior nominal 1999 source deliberately spills into the exact 2000
    # split.  Filename years are evidence metadata, never row-selection policy.
    index = pd.date_range("1999-12-31T23:58:00Z", periods=8, freq="min")
    with tempfile.TemporaryDirectory(prefix="m15-adapter-test-") as temporary:
        root = Path(temporary)
        feature_dir = root / "features"
        values = {column: np.arange(len(index), dtype="float32") for column in columns}
        # Poison/meta fields exist in the source but the blind API must never request them.
        values.update(
            y=np.array(["DO_NOT_READ"] * len(index), dtype=object),
            fwd_ret=np.full(len(index), np.nan),
            valid=np.zeros(len(index), dtype=bool),
            close=np.linspace(1.0, 1.1, len(index), dtype="float32"),
        )
        _write_frame(feature_dir / "USDJPY_1999.parquet", index, values)
        original_read = adapters.pd.read_parquet
        projections: list[tuple[str, ...] | None] = []

        def guarded_read(*args: Any, **kwargs: Any) -> pd.DataFrame:
            projected = kwargs.get("columns")
            projections.append(None if projected is None else tuple(projected))
            if projected:
                forbidden = {"y", "fwd_ret", "valid", "_fwd"}.intersection(projected)
                if forbidden:
                    raise GateFailure(f"blind adapter requested outcome columns {forbidden}")
            return original_read(*args, **kwargs)

        with mock.patch.object(adapters.pd, "read_parquet", side_effect=guarded_read):
            rows = adapters.build_score_rows(
                "USDJPY", ["1999"], columns, feature_dir=feature_dir
            )
        equal(rows.X.shape, (len(index), 239), "blind own-pair matrix shape")
        equal(rows.X.dtype, np.dtype("float32"), "blind matrix dtype")
        equal(rows.entry_ns.dtype, np.dtype("int64"), "blind clock dtype")
        equal(len(rows), len(index), "blind builder future-filtered rows")
        check(
            rows.entry_ns[-1] >= pd.Timestamp("2000-01-01T00:00:00Z").value,
            "prior nominal-file spill was silently calendar-filtered",
        )
        exact_2000 = adapters.slice_score_rows(
            rows,
            entry_at_or_after="2000-01-01T00:00:00Z",
            entry_before="2000-01-01T00:06:00Z",
        )
        equal(len(exact_2000), 6, "exact UTC mask did not admit prior-file spill")
        check(any(call == columns for call in projections), "ordered covariates were not projected")

        # Decoded drift is diagnostic when scoreability and exact policy output agree.
        drift = rows.X.copy()
        drift[0, 0] = np.nextafter(drift[0, 0], np.float32(np.inf), dtype=np.float32)
        rebuilt = adapters.ScoreRows(
            rows.pair,
            rows.feature_cols,
            drift,
            rows.entry_ns.copy(),
            rows.source_feature_ns.copy(),
        )
        probability = np.linspace(0.1, 0.9, len(rows), dtype="float64")
        report = adapters.assert_pre_april_score_parity(
            rows,
            rebuilt,
            reference_probabilities=probability,
            rebuilt_probabilities=probability.copy(),
            confidence_threshold=0.2,
        )
        check(not report["feature_bytes_equal"], "decoded drift fixture did not drift")
        check(report["probability_bitwise_equal"], "identical policy bytes failed parity")
        changed_probability = probability.copy()
        changed_probability[0] = np.nextafter(changed_probability[0], 1.0)
        with expect_raises(adapters.AdapterError, "probability_bitwise_equal"):
            adapters.assert_pre_april_score_parity(
                rows,
                rebuilt,
                reference_probabilities=probability,
                rebuilt_probabilities=changed_probability,
                confidence_threshold=0.2,
            )
        permuted = adapters.ScoreRows(
            rows.pair,
            tuple(reversed(rows.feature_cols)),
            np.ascontiguousarray(rows.X[:, ::-1]),
            rows.entry_ns.copy(),
            rows.source_feature_ns.copy(),
        )
        with expect_raises(adapters.AdapterError, "schema_equal"):
            adapters.assert_pre_april_score_parity(rows, permuted)

        bad_dir = root / "bad"
        bad_index = pd.DatetimeIndex([index[0], index[2], index[1]])
        bad_values = {column: np.ones(3, dtype="float32") for column in columns}
        _write_frame(bad_dir / "USDJPY_2000.parquet", bad_index, bad_values)
        with expect_raises(adapters.AdapterError, "strictly increasing"):
            adapters.build_score_rows("USDJPY", ["2000"], columns, feature_dir=bad_dir)

        # Cross-pair scoring must call the public live builder, never copied formulas.
        xp_columns = tuple(f"xp_synthetic_{index:03d}" for index in range(101))
        gbp_columns = xp_columns + columns
        cross_dir = root / "cross"
        for offset, pair in enumerate(adapters.ALL_PAIRS):
            source = {"close": np.linspace(1.0 + offset, 1.1 + offset, len(index), dtype="float32")}
            if pair == "GBPUSD":
                source.update({column: np.ones(len(index), dtype="float32") for column in columns})
            _write_frame(cross_dir / f"{pair}_2000.parquet", index, source)

        calls: list[tuple[str, tuple[str, ...]]] = []

        def fake_public_builder(
            pair: str, closes: pd.DataFrame, expected_cols: list[str]
        ) -> pd.DataFrame:
            calls.append((pair, tuple(expected_cols)))
            check(tuple(closes.columns) == adapters.ALL_PAIRS, "cross close order changed")
            return pd.DataFrame(
                {column: np.full(len(closes), index + 1, dtype="float32")
                 for index, column in enumerate(xp_columns)},
                index=closes.index,
            )

        with mock.patch.object(
            live_features, "build_cross_pair_features", side_effect=fake_public_builder
        ):
            cross_rows = adapters.build_score_rows(
                "GBPUSD", ["2000"], gbp_columns, feature_dir=cross_dir
            )
        equal(calls, [("GBPUSD", gbp_columns)], "public xpair delegation")
        equal(cross_rows.X.shape, (len(index), 340), "cross-pair matrix shape")

    # The feature cache is left-labeled: source 07:59 becomes the causal 08:00
    # decision.  Runtime session/cutoff policy therefore consumes decision_s,
    # never the source label.
    ny_sources = pd.DatetimeIndex(
        [
            pd.Timestamp("2025-01-15 07:59:00", tz="America/New_York"),
            pd.Timestamp("2025-01-15 16:34:00", tz="America/New_York"),
        ]
    ).tz_convert("UTC")
    ny_source_ns = ny_sources.as_unit("ns").asi8.astype("int64")
    ny_decision_ns = ny_source_ns + (
        adapters.FEATURE_DECISION_SHIFT_SECONDS * adapters.NS_PER_SECOND
    )
    ny_decision_s = ny_decision_ns // adapters.NS_PER_SECOND
    replay_session = np.asarray(adapters.session_mask(ny_decision_s, "ny"), dtype=bool)
    causal_rows = adapters.AdapterRows(
        pair="USDJPY",
        purpose="replay",
        feature_cols=columns,
        X=np.zeros((2, len(columns)), dtype="float32"),
        y=np.array([1, 0], dtype="uint8"),
        moved=np.ones(2, dtype=bool),
        fit_row_id=ny_decision_ns,
        label_exit_ns=ny_decision_ns
        + adapters.HORIZON_SECONDS * adapters.NS_PER_SECOND,
        fit_session_mask=replay_session.copy(),
        calibration_session_mask=replay_session.copy(),
        replay_session_mask=replay_session,
        source_feature_ns=ny_source_ns,
        builder_stride=1,
    )
    check(causal_rows.replay_session_mask[0], "source 07:59 -> decision 08:00 was not admitted")
    runtime_mask = stats.canonical_runtime_ny_mask(ny_decision_s)
    check(runtime_mask[0], "08:00 causal decision failed runtime session")
    check(
        not runtime_mask[1],
        "source 16:34 -> decision 16:35 bypassed the last-start cutoff",
    )
    causal_score = adapters.ScoreRows(
        "USDJPY",
        columns,
        np.zeros((2, len(columns)), dtype="float32"),
        ny_decision_ns.copy(),
        ny_source_ns.copy(),
    )
    check(
        np.array_equal(
            causal_score.entry_ns,
            causal_score.source_feature_ns
            + adapters.FEATURE_DECISION_SHIFT_SECONDS * adapters.NS_PER_SECOND,
        ),
        "ScoreRows causal source-to-decision shift drift",
    )
    with expect_raises(adapters.AdapterError, "+60s"):
        adapters.ScoreRows(
            "USDJPY",
            columns,
            np.zeros((2, len(columns)), dtype="float32"),
            ny_decision_ns.copy(),
            ny_source_ns + adapters.NS_PER_SECOND,
        )
    with expect_raises(adapters.AdapterError, "explicit score source_feature_ns"):
        adapters.ScoreRows(
            "USDJPY",
            columns,
            np.zeros((2, len(columns)), dtype="float32"),
            ny_decision_ns.copy(),
        )

    midnight_source_ns = np.array(
        [pd.Timestamp("2021-12-31T23:44:00Z").value], dtype="int64"
    )
    midnight_decision_ns = midnight_source_ns + 60 * adapters.NS_PER_SECOND
    midnight_rows = adapters.AdapterRows(
        pair="USDJPY",
        purpose="fit",
        feature_cols=columns,
        X=np.zeros((1, len(columns)), dtype="float32"),
        y=np.ones(1, dtype="uint8"),
        moved=np.ones(1, dtype=bool),
        fit_row_id=midnight_decision_ns,
        label_exit_ns=midnight_decision_ns
        + adapters.HORIZON_SECONDS * adapters.NS_PER_SECOND,
        fit_session_mask=np.ones(1, dtype=bool),
        calibration_session_mask=np.ones(1, dtype=bool),
        replay_session_mask=np.zeros(1, dtype=bool),
        source_feature_ns=midnight_source_ns,
        builder_stride=adapters.contract_for("USDJPY").fit_stride,
    )
    check(
        not adapters.window_mask(
            midnight_rows,
            label_exit_before="2022-01-01T00:00:00Z",
        )[0],
        "source 23:44 -> decision 23:45 -> exit 00:00 crossed strict split",
    )
    with expect_raises(adapters.AdapterError, "+60s"):
        replace(
            midnight_rows,
            source_feature_ns=midnight_source_ns + adapters.NS_PER_SECOND,
        )
    with expect_raises(adapters.AdapterError, "explicit source_feature_ns"):
        replace(midnight_rows, source_feature_ns=None)
    with expect_raises(adapters.AdapterError, "builder_stride"):
        replace(midnight_rows, builder_stride=None)

    # Reject an unsigned timestamp before any downstream session conversion can
    # wrap it into signed space.
    unsigned_clock = np.array([2**63], dtype="uint64")
    with expect_raises(adapters.AdapterError, "signed-int64"):
        adapters.ScoreRows(
            "USDJPY",
            columns,
            np.zeros((1, len(columns)), dtype="float32"),
            unsigned_clock,
            np.array([0], dtype="int64"),
        )
    xpair_columns = _synthetic_contract_columns("GBPUSD")
    fake_xp = mock.Mock(FEAT="sentinel")
    fake_frozen = mock.Mock(XP=fake_xp)
    fake_frozen.build_mat.return_value = (
        np.zeros((1, len(xpair_columns)), dtype="float32"),
        np.zeros(1, dtype="float64"),
        unsigned_clock,
        xpair_columns,
    )
    with (
        mock.patch.object(adapters, "_validate_source_files"),
        mock.patch.object(adapters.importlib, "import_module", return_value=fake_frozen),
        mock.patch.object(
            adapters,
            "session_mask",
            side_effect=GateFailure("invalid timestamp reached session_mask"),
        ),
        expect_raises(adapters.AdapterError, "signed-int64 nanosecond range"),
    ):
        adapters.build_xpair_rows("GBPUSD", ["2000"])

    tie_entry = pd.date_range(
        "2021-01-15T13:00:00Z", periods=4, freq="15min"
    ).as_unit("ns").asi8.astype("int64")
    moved = np.array([True, False, True, False])
    down_on_tie = np.array([1, 0, 1, 0], dtype="uint8")
    include_as_down = {"GBPUSD", "USDCHF"}
    for pair in adapters.CONTRACTS:
        fit_session = (
            np.ones(4, dtype=bool)
            if pair == "EURUSD"
            else np.array([True, True, False, True])
        )
        family_rows = _synthetic_adapter_rows(
            pair,
            purpose="fit",
            entry_ns=tie_entry,
            moved=moved,
            y=down_on_tie,
            fit_session=fit_session,
        )
        fit_positions = adapters.fitting_indices(
            family_rows,
            entry_at_or_after="2021-01-15T13:00:00Z",
            label_exit_before="2021-01-16T00:00:00Z",
            cap=10,
        )
        early_positions = adapters.early_stopping_indices(
            family_rows,
            entry_at_or_after="2021-01-15T13:00:00Z",
            label_exit_before="2021-01-16T00:00:00Z",
        )
        expected_fit = np.flatnonzero(fit_session & moved)
        expected_early = np.flatnonzero(
            fit_session if pair in include_as_down else fit_session & moved
        )
        check(
            np.array_equal(fit_positions, expected_fit),
            f"{pair} fitting did not drop ties",
        )
        check(
            np.array_equal(early_positions, expected_early),
            f"{pair} early-stopping tie rule drift",
        )
        if pair in include_as_down:
            check(
                not family_rows.y[~family_rows.moved].any(),
                f"{pair} early-stopping ties were not labeled DOWN",
            )
            bad_labels = down_on_tie.copy()
            bad_labels[1] = 1
            with expect_raises(adapters.AdapterError, "DOWN label"):
                _synthetic_adapter_rows(
                    pair,
                    purpose="fit",
                    entry_ns=tie_entry,
                    moved=moved,
                    y=bad_labels,
                    fit_session=fit_session,
                )
            if pair == "GBPUSD":
                canonical_validation_rows = replace(
                    family_rows,
                    purpose="calibration",
                    builder_stride=1,
                )
                control_rows, permutation_evidence = refresh._permuted_rows(
                    canonical_validation_rows,
                    np.array([0, 1], dtype="int64"),
                    np.random.default_rng(3),
                )
                check(
                    set(permutation_evidence) == {
                        "row_count",
                        "ordered_row_id_sha256",
                        "input_label_u8_sha256",
                        "permutation_index_i64le_sha256",
                        "permuted_label_u8_sha256",
                    },
                    "negative-control permutation evidence fields drifted",
                )
                check(
                    control_rows.sealed_control_label_permutation,
                    "negative-control permutation marker was not sealed",
                )
                check(
                    control_rows.y[1] == 1 and not control_rows.moved[1],
                    "full early-stop permutation did not move UP onto a tie",
                )
                with expect_raises(adapters.AdapterError, "DOWN label"):
                    replace(
                        control_rows,
                        sealed_control_label_permutation=False,
                    )

    # EURUSD's q33 structural threshold sees all label-valid rows in the exact
    # UTC window, including rows outside its legacy fixed-UTC session.  Only
    # after q33 is frozen does the session+q33 confidence population form.
    eur_entry = pd.date_range(
        "2023-01-02T00:00:00Z", periods=6, freq="15min"
    ).as_unit("ns").asi8.astype("int64")
    widths = np.array([1.0, 2.0, 3.0, 4.0, 100.0, -1000.0], dtype="float32")
    fixed_utc_session = np.array([True, True, True, True, False, True])
    eur_rows = _synthetic_adapter_rows(
        "EURUSD",
        purpose="calibration",
        entry_ns=eur_entry,
        moved=np.ones(6, dtype=bool),
        y=np.array([1, 0, 1, 0, 1, 0], dtype="uint8"),
        fit_session=np.ones(6, dtype=bool),
        calibration_session=fixed_utc_session,
        structural_values=widths,
    )
    calibration_stop = "2023-01-02T01:16:00Z"
    structural_scope = adapters.window_mask(
        eur_rows,
        entry_at_or_after="2023-01-02T00:00:00Z",
        label_exit_before=calibration_stop,
    )
    calibration_positions = adapters.calibration_indices(
        eur_rows,
        entry_at_or_after="2023-01-02T00:00:00Z",
        label_exit_before=calibration_stop,
    )
    admitted = np.zeros(6, dtype=bool)
    admitted[calibration_positions] = True
    eur_probability = np.array([0.1, 0.2, 0.9, 0.8, 0.51, 0.0])
    calibration = adapters.calibrate_policy(
        eur_rows,
        eur_probability,
        admitted,
        target_coverage=0.5,
        structural_scope_mask=structural_scope,
    )
    expected_q33 = float(np.quantile(widths[:5], 0.33, method="linear"))
    session_only_q33 = float(np.quantile(widths[:4], 0.33, method="linear"))
    check(not np.isclose(expected_q33, session_only_q33), "q33 fixture is not discriminating")
    check(
        np.isclose(calibration.structural_threshold, expected_q33),
        "EURUSD q33 did not use the full exact label-valid window before session",
    )
    equal(calibration.structural_rows, 5, "EURUSD full-window q33 rows")
    equal(calibration.calibration_rows, 4, "EURUSD fixed-UTC calibration rows")
    equal(calibration.confidence_rows, 2, "EURUSD fixed-UTC+q33 confidence rows")
    check(
        np.isclose(calibration.confidence_threshold, 0.35),
        "EURUSD confidence quantile population drift",
    )

    split_index = pd.date_range("2021-12-31T23:15:00Z", periods=4, freq="15min")
    split_entry = split_index.as_unit("ns").asi8.astype("int64")
    split_rows = adapters.AdapterRows(
        pair="USDJPY",
        purpose="fit",
        feature_cols=columns,
        X=np.zeros((4, len(columns)), dtype="float32"),
        y=np.array([1, 0, 1, 0], dtype="uint8"),
        moved=np.array([True, False, True, True]),
        fit_row_id=split_entry,
        label_exit_ns=split_entry + 900 * 1_000_000_000,
        fit_session_mask=np.ones(4, dtype=bool),
        calibration_session_mask=np.ones(4, dtype=bool),
        replay_session_mask=np.ones(4, dtype=bool),
        source_feature_ns=split_entry
        - adapters.FEATURE_DECISION_SHIFT_SECONDS * adapters.NS_PER_SECOND,
        builder_stride=adapters.contract_for("USDJPY").fit_stride,
    )
    fit_positions = adapters.fitting_indices(
        split_rows,
        entry_at_or_after="2021-12-31T23:15:00Z",
        label_exit_before="2022-01-01T00:00:00Z",
        cap=10,
    )
    check(
        np.array_equal(fit_positions, np.array([0], dtype="int64")),
        "fit split/tie or strict label-exit boundary drift",
    )

    # Position 3 is a 2022 timestamp carried by the prior nominal 2021 source.
    # The exact UTC calibration mask admits it even though its filename year is
    # outside the calibration years proper.
    spill_rows = adapters.AdapterRows(
        pair="USDJPY",
        purpose="calibration",
        feature_cols=columns,
        X=split_rows.X.copy(),
        y=split_rows.y.copy(),
        moved=split_rows.moved.copy(),
        fit_row_id=split_rows.fit_row_id.copy(),
        label_exit_ns=split_rows.label_exit_ns.copy(),
        fit_session_mask=split_rows.fit_session_mask.copy(),
        calibration_session_mask=split_rows.calibration_session_mask.copy(),
        replay_session_mask=split_rows.replay_session_mask.copy(),
        source_feature_ns=split_rows.source_feature_ns.copy(),
        builder_stride=1,
    )
    spill_positions = adapters.calibration_indices(
        spill_rows,
        entry_at_or_after="2022-01-01T00:00:00Z",
        label_exit_before="2022-01-02T00:00:00Z",
    )
    check(
        np.array_equal(spill_positions, np.array([3], dtype="int64")),
        "exact UTC split rejected the prior nominal-file spill row",
    )

    permuted_fit_rows = adapters.AdapterRows(
        pair="USDJPY",
        purpose="calibration",
        feature_cols=tuple(reversed(split_rows.feature_cols)),
        X=np.ascontiguousarray(split_rows.X[:, ::-1]),
        y=split_rows.y.copy(),
        moved=split_rows.moved.copy(),
        fit_row_id=split_rows.fit_row_id.copy(),
        label_exit_ns=split_rows.label_exit_ns.copy(),
        fit_session_mask=split_rows.fit_session_mask.copy(),
        calibration_session_mask=split_rows.calibration_session_mask.copy(),
        replay_session_mask=split_rows.replay_session_mask.copy(),
        source_feature_ns=split_rows.source_feature_ns.copy(),
        builder_stride=1,
    )
    with tempfile.TemporaryDirectory(prefix="m15-feature-order-test-") as raw:
        with expect_raises(adapters.AdapterError, "feature order mismatch"):
            adapters.fit_seed_checkpoint(
                "USDJPY",
                0,
                split_rows,
                np.array([0], dtype="int64"),
                permuted_fit_rows,
                np.array([0], dtype="int64"),
                Path(raw) / "must-not-fit.txt",
            )

    commands = (
        "import sys; sys.argv=['adapter-import','--foreign-flag']; "
        "from gbpusd_15m_xpair import build_xp_gbp; assert callable(build_xp_gbp)",
        "import sys; sys.argv=['adapter-import','--foreign-flag']; "
        "from gbpusd_15m_xpair_frozen import build_mat; assert callable(build_mat)",
        "import sys; sys.argv=['adapter-import','--foreign-flag']; "
        "from usdchf_15m_xpair_frozen import build_mat; assert callable(build_mat)",
    )
    environment = dict(os.environ, PYTHONPATH=str(REPO_ROOT / "scripts"))
    for command in commands:
        completed = subprocess.run(
            [sys.executable, "-c", command],
            cwd=REPO_ROOT,
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if completed.returncode:
            raise GateFailure(
                f"foreign-argument import failed ({completed.returncode}): {completed.stderr}"
            )


def test_direct_incumbent_builder_parity() -> None:
    source_index = pd.date_range(
        "2021-01-15T13:00:00Z", periods=5, freq="min"
    )
    source_s = (
        source_index.as_unit("ns").asi8 // adapters.NS_PER_SECOND
    ).astype("int64")
    selection_bounds = {
        "entry_at_or_after": "2021-01-15T00:00:00Z",
        "label_exit_before": "2021-01-16T00:00:00Z",
        "cap": 2,
    }

    def assert_complete(report: dict[str, Any], pair: str) -> None:
        check(report["all_equal"], f"{pair} direct builder parity failed: {report}")
        check(report["source_feature_clock_equal"], f"{pair} source clock parity")
        check(report["decision_clock_equal"], f"{pair} decision clock parity")
        check(report["feature_order_equal"], f"{pair} ordered feature parity")
        check(report["feature_dtype_equal"], f"{pair} float32 feature parity")
        check(report["feature_bytes_equal"], f"{pair} feature byte parity")
        check(report["labels_equal"] and report["moved_equal"], f"{pair} label parity")
        check(
            report["legacy_fit_session_mask_equal"]
            and report["legacy_calibration_session_mask_equal"],
            f"{pair} legacy session parity",
        )
        check(report["early_stopping_mask_equal"], f"{pair} early-stop parity")
        check(report["stride_equal"] and report["purpose_equal"], f"{pair} build policy")
        equal(report["direct_capped_selected_rows"], 2, f"{pair} capped rows")
        check(report["capped_selected_row_ids_equal"], f"{pair} capped row IDs")

    eur_columns = _synthetic_contract_columns("EURUSD")
    eur_values = {
        column: np.full(len(source_index), index + 0.25, dtype="float32")
        for index, column in enumerate(eur_columns)
    }
    eur_values.update(
        _ts=source_s,
        _fwd=np.array([0.1, -0.2, 0.3, -0.4, 0.5]),
        sess_ny=np.array([1.0, 1.0, 0.0, 1.0, 1.0]),
    )
    eur_frame = pd.DataFrame(eur_values, index=source_index)
    eur_calls: list[tuple[tuple[str, ...], int]] = []
    eur_builder = mock.Mock(FEAT="sentinel-feat", OFDIR="sentinel-of")

    def eur_build(years: list[str], stride: int) -> pd.DataFrame:
        eur_calls.append((tuple(years), stride))
        return eur_frame.copy(deep=True)

    eur_builder.build_xp.side_effect = eur_build
    eur_builder.xp_cols.side_effect = lambda frame: []
    eur_builder.augment.side_effect = lambda frame, years, mode: frame
    eur_builder.feat_cols.side_effect = (
        lambda mode, frame, xpair_cols: list(eur_columns)
    )
    with (
        mock.patch.object(adapters, "_validate_source_files"),
        mock.patch.object(
            adapters, "_import_m15_eur_builder", return_value=eur_builder
        ),
    ):
        eur_rows = adapters.build_rows(
            "EURUSD",
            ["2000"],
            purpose="fit",
            feature_dir="/synthetic/features",
            orderflow_dir="/synthetic/features_of",
        )
        eur_report = adapters.direct_incumbent_builder_parity_report(
            eur_rows,
            ["2000"],
            purpose="fit",
            feature_dir="/synthetic/features",
            orderflow_dir="/synthetic/features_of",
            **selection_bounds,
        )
    assert_complete(eur_report, "EURUSD")
    equal(
        eur_calls,
        [(('2000',), adapters.contract_for("EURUSD").fit_stride)] * 2,
        "EURUSD direct builder was not independently reinvoked",
    )

    for pair in ("GBPUSD", "USDCHF"):
        columns = _synthetic_contract_columns(pair)
        matrix = np.arange(
            len(source_index) * len(columns), dtype="float32"
        ).reshape(len(source_index), len(columns))
        fwd = np.array([0.1, 0.0, -0.2, 0.3, -0.4], dtype="float64")
        calls: list[tuple[tuple[str, ...], int]] = []
        frozen = mock.Mock(XP=mock.Mock(FEAT="sentinel"))

        def build_mat(
            years: list[str], stride: int, *, _calls: list[Any] = calls
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[str, ...]]:
            _calls.append((tuple(years), stride))
            return matrix.copy(), fwd.copy(), source_s.copy(), columns

        frozen.build_mat.side_effect = build_mat
        with (
            mock.patch.object(adapters, "_validate_source_files"),
            mock.patch.object(
                adapters.importlib, "import_module", return_value=frozen
            ),
        ):
            family_rows = adapters.build_rows(
                pair,
                ["2000"],
                purpose="fit",
                feature_dir="/synthetic/features",
            )
            report = adapters.direct_incumbent_builder_parity_report(
                family_rows,
                ["2000"],
                purpose="fit",
                feature_dir="/synthetic/features",
                **selection_bounds,
            )
        assert_complete(report, pair)
        equal(
            calls,
            [(('2000',), adapters.contract_for(pair).fit_stride)] * 2,
            f"{pair} frozen builder was not independently reinvoked",
        )

    own_pairs = ("USDJPY", "USDCAD", "AUDUSD", "NZDUSD")
    for pair in own_pairs:
        columns = _synthetic_contract_columns(pair)
        matrix = np.arange(
            len(source_index) * len(columns), dtype="float32"
        ).reshape(len(source_index), len(columns))
        frame = pd.DataFrame(matrix, index=source_index, columns=columns)
        y = np.array([1, 0, 1, 0, 1], dtype="uint8")
        moved = np.array([True, False, True, True, True])
        calls: list[tuple[tuple[str, ...], int]] = []
        base = mock.Mock(FEAT="sentinel")

        def build_base(
            years: list[str], stride: int, *, _calls: list[Any] = calls
        ) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
            _calls.append((tuple(years), stride))
            return frame.copy(deep=True), y.copy(), moved.copy(), source_s.copy()

        base.build.side_effect = build_base
        with (
            mock.patch.object(adapters, "_validate_source_files"),
            mock.patch.object(
                adapters.importlib, "import_module", return_value=base
            ),
            mock.patch.object(adapters.harness, "feature_cols", return_value=columns),
        ):
            family_rows = adapters.build_rows(
                pair,
                ["2000"],
                purpose="fit",
                feature_dir="/synthetic/features",
            )
            report = adapters.direct_incumbent_builder_parity_report(
                family_rows,
                ["2000"],
                purpose="fit",
                feature_dir="/synthetic/features",
                **selection_bounds,
            )
            if pair == "USDJPY":
                drift = family_rows.X.copy()
                drift[0, 0] = np.nextafter(
                    drift[0, 0], np.float32(np.inf), dtype=np.float32
                )
                drift_report = adapters.direct_incumbent_builder_parity_report(
                    replace(family_rows, X=drift),
                    ["2000"],
                    purpose="fit",
                    feature_dir="/synthetic/features",
                    **selection_bounds,
                )
                check(
                    not drift_report["all_equal"]
                    and not drift_report["feature_bytes_equal"],
                    "direct builder parity ignored feature-byte drift",
                )
                policy_report = adapters.direct_incumbent_builder_parity_report(
                    family_rows,
                    ["2000"],
                    purpose="replay",
                    feature_dir="/synthetic/features",
                    **selection_bounds,
                )
                check(
                    not policy_report["all_equal"]
                    and not policy_report["purpose_equal"]
                    and not policy_report["stride_equal"],
                    "direct builder parity ignored purpose/stride drift",
                )
        assert_complete(report, pair)
        expected_calls = 4 if pair == "USDJPY" else 2
        equal(len(calls), expected_calls, f"{pair} direct builder invocation count")
        equal(
            calls[:2],
            [(('2000',), adapters.contract_for(pair).fit_stride)] * 2,
            f"{pair} base builder was not independently reinvoked",
        )


def _feature_route_fixture(root: Path) -> tuple[Path, Path, Path]:
    workspace = root / "workspace"
    targets = workspace / "feature_snapshot" / "features"
    targets.mkdir(parents=True)
    first = targets / "USDJPY_2026.parquet"
    second = targets / "USDCAD_2026.parquet"
    first.write_bytes(b"sealed-route-one")
    second.write_bytes(b"sealed-route-two")
    views = workspace / "feature_views"
    for name in refresh.FEATURE_VIEW_NAMES:
        (views / name).mkdir(parents=True)
    os.symlink(first.resolve(), views / "features" / first.name)
    os.symlink(second.resolve(), views / "features" / second.name)
    (views / "fit_features" / first.name).write_bytes(b"truncated-regular")
    refresh.atomic_json_new(
        views / "routing_manifest.json",
        refresh._feature_view_routing_payload(workspace),
    )
    return workspace, first, second


def test_feature_view_routing() -> None:
    with tempfile.TemporaryDirectory(prefix="m15-route-repoint-") as raw:
        workspace, first, second = _feature_route_fixture(Path(raw))
        refresh._verify_feature_view_routes(workspace, verify_target_bytes=True)
        routed = workspace / "feature_views" / "features" / first.name
        routed.unlink()
        os.symlink(second.resolve(), routed)
        with expect_raises(refresh.RefreshError, "routing changed"):
            refresh._verify_feature_view_routes(workspace)

    with tempfile.TemporaryDirectory(prefix="m15-route-bytes-") as raw:
        workspace, first, _second = _feature_route_fixture(Path(raw))
        before = first.stat()
        # Preserve size and mtime to prove the phase-boundary byte verifier is
        # cryptographic rather than relying only on the cheap per-builder stat.
        first.write_bytes(b"forged-route-one")
        os.utime(first, ns=(before.st_atime_ns, before.st_mtime_ns))
        with expect_raises(refresh.RefreshError, "routing changed"):
            refresh._verify_feature_view_routes(
                workspace, verify_target_bytes=True
            )

    with tempfile.TemporaryDirectory(prefix="m15-route-unexpected-") as raw:
        workspace, _first, _second = _feature_route_fixture(Path(raw))
        unexpected = workspace / "feature_views" / "features" / "EXTRA_2026.parquet"
        unexpected.write_bytes(b"unexpected")
        with expect_raises(refresh.RefreshError, "count changed"):
            refresh._verify_feature_view_routes(workspace)

    with tempfile.TemporaryDirectory(prefix="m15-route-scoped-") as raw:
        workspace, first, second = _feature_route_fixture(Path(raw))
        first_route = workspace / "feature_views" / "features" / first.name
        second_route = workspace / "feature_views" / "features" / second.name
        real_sha256_file = refresh.sha256_file
        hashed: list[Path] = []

        def hash_spy(path: str | Path) -> str:
            hashed.append(Path(path))
            return real_sha256_file(path)

        with mock.patch.object(refresh, "sha256_file", side_effect=hash_spy):
            with refresh.verified_feature_view_access(workspace, [first_route]):
                pass
        equal(hashed, [first_route, first_route], "scoped route hash set")
        check(second_route not in hashed, "unrelated route was cryptographically hashed")

        before = first.stat()
        with expect_raises(refresh.RefreshError, "scoped feature-view"):
            with refresh.verified_feature_view_access(workspace, [first_route]):
                first.write_bytes(b"forged-route-one")
                os.utime(first, ns=(before.st_atime_ns, before.st_mtime_ns))

    with tempfile.TemporaryDirectory(prefix="m15-route-shared-corrupt-") as raw:
        workspace, _first, _second = _feature_route_fixture(Path(raw))
        route_manifest = workspace / "feature_views/routing_manifest.json"
        route_manifest.chmod(0o600)
        route_manifest.write_bytes(b'{"schema":"corrupt-shared-authority"}\n')
        route_manifest.chmod(0o444)
        try:
            refresh._load_feature_view_routing_manifest(workspace)
        except refresh.PairRouteError as exc:
            raise GateFailure(
                "corrupt shared routing authority was misclassified as pair-local"
            ) from exc
        except refresh.RefreshError:
            pass
        else:
            raise GateFailure("corrupt shared routing authority did not abort")


def test_runner_replay_hardening() -> None:
    """Exercise phase-zero, identity, one-look, route, and replay fail-closed gates."""

    spec = refresh.load_spec()
    equal(
        spec["splits"]["phase_zero"],
        {
            "entry_at_or_after": "2026-01-01T00:00:00Z",
            "entry_before": "2026-04-01T00:00:00Z",
        },
        "phase-zero split must bind the outcome-blind 2026-Q1 authority window",
    )
    with mock.patch.dict(os.environ, {"PYTHONNOUSERSITE": "0"}):
        with expect_raises(refresh.RefreshError, "PYTHONNOUSERSITE=1"):
            refresh.verified_runtime_version_contract()
    with tempfile.TemporaryDirectory(prefix="m15-phase-zero-create-only-") as raw:
        preserved = Path(raw) / "preserved.json"
        preserved.write_bytes(b"preserved-stopped-diagnostic\n")
        before = preserved.read_bytes()
        with expect_raises(phase_zero.AuthorityError, "refusing overwrite"):
            phase_zero._atomic_create(preserved, b"replacement\n")
        equal(preserved.read_bytes(), before, "stopped diagnostic was overwritten")

    # The checked-in two-pair diagnostic is intentionally not an acceptance
    # artifact and therefore cannot authorize preregistration.
    with expect_raises(refresh.RefreshError, "phase-zero authority"):
        refresh._validate_phase_zero_authority(
            REPO_ROOT
            / "results/json/m15_book_refresh_2026q1_phase_zero_behavior_authority.json"
        )

    # A synthetic acceptance artifact binds six target reports, the exact
    # seven-source input/rebuild inventory, and every final implementation byte.
    with tempfile.TemporaryDirectory(prefix="m15-phase-zero-contract-") as raw:
        repo = Path(raw) / "repo"
        (repo / "scripts").mkdir(parents=True)
        (repo / "features").mkdir()
        (repo / "features_of").mkdir()
        (repo / "books").mkdir()
        (repo / "processed").mkdir()
        sources: list[Path] = []
        for index, name in enumerate(
            ("live_features.py", "source_1.py", "source_2.py")
        ):
            source = repo / "scripts" / name
            source.write_bytes(f"source-{index}\n".encode())
            sources.append(source)
        orderflow_source = repo / "scripts" / "orderflow.py"
        orderflow_source.write_bytes(b"synthetic-orderflow-recipe\n")
        sources.append(orderflow_source)
        index_file = repo / "books" / "INDEX.json"
        index_file.write_bytes(b'{"schema":"synthetic-book-index/v1"}\n')

        def identity(path: Path, declared: str) -> dict[str, Any]:
            return {
                "path": declared,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }

        def evidence_digest(label: str) -> str:
            return hashlib.sha256(label.encode()).hexdigest()

        spec = refresh.load_spec()
        target_order = list(spec["pair_order"])
        source_order = list(spec["feature_source_pair_order"])
        raw_schema = ["open", "high", "low", "close", "volume", "datetime_utc"]
        cached_by_pair: dict[str, Path] = {}
        processed_by_pair: dict[str, list[Path]] = {}
        for pair in source_order:
            cached = repo / "features" / f"{pair}_2026.parquet"
            cached.write_bytes(f"cached-{pair}".encode())
            cached_by_pair[pair] = cached
            pair_processed = repo / "processed" / pair
            pair_processed.mkdir()
            processed_by_pair[pair] = [
                pair_processed / f"{pair}_10s_2026-01-01.parquet",
                pair_processed / f"{pair}_10s_2026-01-02.parquet",
            ]
            for index, processed in enumerate(processed_by_pair[pair]):
                processed.write_bytes(f"processed-{pair}-{index}".encode())
        cached_of = repo / "features_of" / "EURUSD_2026.parquet"
        cached_of.write_bytes(b"cached-orderflow")
        pair_rows: dict[str, Any] = {}
        incumbent_contracts: dict[str, dict[str, Any]] = {}
        for pair in target_order:
            book_id = spec["pairs"][pair]["book_id_A"]
            book_root = repo / "books" / book_id
            book_root.mkdir()
            manifest = repo / "books" / f"{book_id}.manifest.json"
            manifest.write_bytes(f"manifest-{pair}".encode())
            strategy = book_root / f"{pair}.strategy.json"
            strategy.write_bytes(f"strategy-{pair}".encode())
            models = [
                book_root / f"{pair}.s{seed}.model.txt" for seed in (0, 1)
            ]
            for seed, model in enumerate(models):
                model.write_bytes(f"model-{pair}-{seed}".encode())
            projection = [
                f"covariate_{index}"
                for index in range(spec["pairs"][pair]["feature_count"])
            ]
            if pair == "EURUSD":
                projection[0] = "OF_synthetic"
            content_id = f"synthetic-{pair}"
            incumbent_contracts[pair] = {
                "book_id": book_id,
                "content_id": content_id,
                "feature_cols": projection,
                "model_order": [
                    model.relative_to(repo).as_posix() for model in models
                ],
                "ordered_files": (
                    index_file,
                    manifest,
                    strategy,
                    *models,
                ),
            }
        cached_source_rows = [
            {
                "pair": pair,
                "file": identity(cached_by_pair[pair], f"features/{pair}_2026.parquet"),
                "projection_in_order": ["close"],
            }
            for pair in source_order
        ]
        processed_source_rows = [
            {
                "pair": pair,
                "projection_in_order": raw_schema,
                "ordered_files": [
                    identity(path, str(path.resolve()))
                    for path in processed_by_pair[pair]
                ],
            }
            for pair in source_order
        ]
        cached_source_by_pair = {row["pair"]: row for row in cached_source_rows}
        processed_source_by_pair = {
            row["pair"]: row for row in processed_source_rows
        }
        incumbent_source_rows = [
            {
                "book_id": incumbent_contracts[pair]["book_id"],
                "content_id": incumbent_contracts[pair]["content_id"],
                "model_order": incumbent_contracts[pair]["model_order"],
                "ordered_files": [
                    identity(path, path.relative_to(repo).as_posix())
                    for path in incumbent_contracts[pair]["ordered_files"]
                ],
            }
            for pair in target_order
        ]
        source_inputs = {
            "cached_feature_sources_in_feature_source_order": cached_source_rows,
            "processed_raw_source_groups_in_feature_source_order": processed_source_rows,
            "eurusd_cached_orderflow_comparator": {
                "file": identity(cached_of, "features_of/EURUSD_2026.parquet"),
                "projection_in_order": ["OF_synthetic"],
            },
            "eurusd_rebuilt_orderflow_authority": {
                "source_pair": "EURUSD",
                "processed_raw_source_group_sha256": refresh.sha256_bytes(
                    refresh.canonical_bytes(processed_source_by_pair["EURUSD"])
                ),
                "recipe": "orderflow.build_of_features(orderflow.of_1m(raw_10s))",
                "recipe_source": identity(
                    orderflow_source, "scripts/orderflow.py"
                ),
                "projection_in_order": ["OF_synthetic"],
            },
            "incumbent_books_in_target_pair_order": incumbent_source_rows,
            "forbidden_semantic_columns_read": [],
        }
        source_inputs_sha256 = refresh.sha256_bytes(
            refresh.canonical_bytes(source_inputs)
        )
        verdict_checks = {
            name: True
            for name in (
                "outcome_free_score_adapter_exact",
                "finite_eligibility_exact",
                "common_probability_bitwise_exact",
                "common_direction_exact",
                "loadedbook_scalar_probes_exact",
                "common_gate_components_exact",
                "common_ordered_pre_schedule_exact",
                "full_population_ordered_pre_schedule_exact",
                "provider_state_exact",
            )
        }
        minute_clock_hash = evidence_digest("minute-clock/common")
        empty_array_hash = refresh.sha256_bytes(b"")
        for pair in target_order:
            family = spec["pairs"][pair]["adapter_family"]
            used_sources = source_order if family != "own_pair" else [pair]
            cross_inputs = (
                [cached_source_by_pair[source]["file"] for source in source_order]
                if family != "own_pair"
                else []
            )
            if family == "eur_xpair_of":
                cross_inputs.append(
                    source_inputs["eurusd_cached_orderflow_comparator"]["file"]
                )
            pair_inputs = {
                "cached_feature_file": cached_source_by_pair[pair]["file"],
                "cached_parquet_projection_in_order": incumbent_contracts[pair][
                    "feature_cols"
                ],
                "processed_parquet_projection_in_order": raw_schema,
                "processed_parquet_required_schema_in_order": raw_schema,
                "forbidden_semantic_columns_read": [],
                "processed_raw_files_in_loader_order": processed_source_by_pair[pair][
                    "ordered_files"
                ],
                "cross_pair_inputs_actually_used": cross_inputs,
                "cross_pair_inputs_used": family != "own_pair",
                "incumbent_book": incumbent_source_rows[target_order.index(pair)],
                "cached_source_projections_in_feature_source_order": [
                    {
                        "pair": source,
                        "projection_in_order": ["close"],
                    }
                    for source in used_sources
                ],
                "processed_raw_source_groups_actually_used": [
                    processed_source_by_pair[source] for source in used_sources
                ],
                "common_input_inventory_sha256": source_inputs_sha256,
            }
            if pair == "EURUSD":
                provider = {
                    "status": "BLOCKED_SCHEMA",
                    "reason": "verified_live_OF_provider_absent",
                    "live_activation_eligible": False,
                    "acceptance_semantics": (
                        "verified_offline_behavior_pass_live_activation_blocked"
                    ),
                    "provider_state_exact": True,
                    "requested_orderflow_columns_in_order": ["OF_synthetic"],
                    "public_provider_missing_orderflow_columns_in_order": [
                        "OF_synthetic"
                    ],
                    "public_feature_row_rejected_missing_schema": True,
                    "public_feature_row_rejection": "synthetic missing OF schema",
                    "public_live_feature_builder_parity": None,
                }
            else:
                provider = {
                    "status": "PASSED",
                    "reason": None,
                    "live_activation_eligible": True,
                    "acceptance_semantics": (
                        "public_live_feature_builder_parity_required"
                    ),
                    "provider_state_exact": True,
                    "requested_orderflow_columns_in_order": [],
                    "public_provider_missing_orderflow_columns_in_order": [],
                    "public_feature_row_rejected_missing_schema": False,
                    "public_feature_row_rejection": None,
                    "public_live_feature_builder_parity": {
                        "schema": "m15-live-feature-builder-parity/v1",
                        "pair": pair,
                        "public_class": "live_features.LiveFeatureBuilder",
                        "public_method_exercised": "feature_row",
                        "provider_rows": 2,
                        "compared_rows": 2,
                        "feature_count": len(
                            incumbent_contracts[pair]["feature_cols"]
                        ),
                        "feature_order_exact": True,
                        "source_clock_subset_exact": True,
                        "causal_decision_mapping_exact": True,
                        "finite_mask_exact": True,
                        "ordered_float32_feature_bytes_exact": True,
                        "decoded_diagnostics": {
                            "mismatched_cells": 0,
                            "finite_nonfinite_disagreements": 0,
                            "columns": {},
                        },
                        "public_latest_row_exact": True,
                        "public_latest_source_feature_ns": int(
                            pd.Timestamp("2026-03-31T23:58:00Z").value
                        ),
                        "all_equal": True,
                        "behavior_all_equal": True,
                        "ordered_finite_float32_values_exact": True,
                        "nonfinite_nan_payload_bits_binding": False,
                    },
                }
            feature_hash = evidence_digest(f"features/{pair}")
            finite_hash = evidence_digest(f"finite/{pair}")
            probability_hash = evidence_digest(f"probability/{pair}")
            selected_hash = evidence_digest(f"selected/{pair}")
            pair_rows[pair] = {
                "verdict": "PASS_BEHAVIOR_AUTHORITY",
                "verdict_checks": verdict_checks,
                "inputs": pair_inputs,
                "rows": 2,
                "feature_count": len(incumbent_contracts[pair]["feature_cols"]),
                "feature_dtype": "float32",
                "source_feature_clock_i64le_sha256": minute_clock_hash,
                "decision_clock_i64le_sha256": evidence_digest(
                    f"decision-clock/{pair}"
                ),
                "causal_decision_mapping_exact": True,
                "ordered_float32_features": {
                    "cached_sha256": feature_hash,
                    "raw_rebuilt_sha256": feature_hash,
                    "bitwise_equal": True,
                    "behavior_neutral_decoded_drift": True,
                    "decoded_diagnostics": {
                        "mismatched_cells": 0,
                        "finite_nonfinite_disagreements": 0,
                        "columns": {},
                    },
                },
                "finite_eligibility": {
                    "cached_count": 2,
                    "raw_rebuilt_count": 2,
                    "common_count": 2,
                    "mismatch_count": 0,
                    "cached_u8_sha256": finite_hash,
                    "raw_rebuilt_u8_sha256": finite_hash,
                },
                "scoreability_mismatch_source": {
                    "columns": [],
                    "rows": 0,
                    "cached_finite_raw_nonfinite": 0,
                    "cached_nonfinite_raw_finite": 0,
                    "first_decision_utc": None,
                    "last_decision_utc": None,
                    "decision_i64le_sha256": empty_array_hash,
                },
                "common_loadedbook_probability": {
                    "population": "common_finite_rows_only",
                    "scorer": "book_runtime.LoadedBook.predict_probabilities",
                    "rows": 2,
                    "bit_mismatch_count": 0,
                    "direction_mismatch_count": 0,
                    "max_abs_difference": 0.0,
                    "cached_f64le_sha256": probability_hash,
                    "raw_rebuilt_f64le_sha256": probability_hash,
                    "loadedbook_scalar_parity": {
                        "probe_rows": 2,
                        "cached_raw_probability_bits_exact": True,
                        "cached_raw_direction_exact": True,
                        "cached_raw_gate_exact": True,
                        "scalar_batch_direction_exact": True,
                        "scalar_batch_confidence_gate_exact": True,
                        "scalar_batch_structural_gate_exact": True,
                        "scalar_batch_combined_gate_exact": True,
                        "scalar_batch_gate_exact": True,
                        "scalar_batch_probability_bits_exact_diagnostic": True,
                        "scalar_batch_probability_bits_binding": False,
                        "behavior_exact": True,
                    },
                },
                "gate_components": {
                    component: {
                        "cached_pass_count": 1,
                        "raw_rebuilt_pass_count": 1,
                        "mismatch_count": 0,
                    }
                    for component in ("confidence", "structural", "combined")
                },
                "ordered_pre_schedule": {
                    "population": "common_finite_rows_only",
                    "cached_count": 1,
                    "raw_rebuilt_count": 1,
                    "exact_equal": True,
                    "cached_i64le_sha256": selected_hash,
                    "raw_rebuilt_i64le_sha256": selected_hash,
                },
                "full_population_ordered_pre_schedule": {
                    "population": "each_side_full_finite_population",
                    "cached_eligible_rows": 2,
                    "raw_rebuilt_eligible_rows": 2,
                    "cached_count": 1,
                    "raw_rebuilt_count": 1,
                    "exact_equal": True,
                    "actual_pre_schedule_behavior_changed": False,
                    "symmetric_difference_count": 0,
                    "cached_i64le_sha256": selected_hash,
                    "raw_rebuilt_i64le_sha256": selected_hash,
                    "cached_probability_f64le_sha256": probability_hash,
                    "raw_rebuilt_probability_f64le_sha256": probability_hash,
                },
                "provider_qualification": provider,
            }

        schema_fields = [
            {"name": name, "type": field_type, "nullable": True}
            for name, field_type in zip(
                raw_schema,
                [
                    "double",
                    "double",
                    "double",
                    "double",
                    "double",
                    "timestamp[ns, tz=UTC]",
                ],
            )
        ]

        def strict_clock(pair: str, label: str) -> dict[str, Any]:
            rows = 24 if label == "global" else 12
            return {
                "rows": rows,
                "first_utc": "2026-01-01T00:00:00Z",
                "last_utc": "2026-01-01T00:01:50Z",
                "clock_i64le_sha256": evidence_digest(f"raw-clock/{pair}"),
                "strictly_increasing": True,
                "unique": True,
                "ten_second_lattice_exact": True,
                "all_gaps_positive_multiples_of_10s": True,
                "gap_gt_10s_count": 0,
                "minimum_gap_seconds": 10,
                "maximum_gap_seconds": 10,
            }

        raw_rebuilds: dict[str, Any] = {}
        for pair in source_order:
            processed_identities = processed_source_by_pair[pair]["ordered_files"]
            full_minute_clock_hash = evidence_digest(
                f"full-minute-clock/{pair}"
            )
            full_minute_values_hash = evidence_digest(f"full-minute-values/{pair}")
            bounded_minute_values_hash = evidence_digest(
                f"bounded-minute-values/{pair}"
            )
            raw_rebuilds[pair] = {
                "strict_processed_source": {
                    "loader": (
                        "m15_phase_zero_behavior_authority."
                        "_load_processed_q1_strict"
                    ),
                    "projection_in_order": raw_schema,
                    "schema_policy": "exact_schema_only_no_extra_columns",
                    "sort_or_dedup_performed": False,
                    "semantic_or_outcome_columns_read": False,
                    "ordered_files": [
                        {
                            "path": processed_identity["path"],
                            "schema_in_order": schema_fields,
                            **strict_clock(pair, "file"),
                        }
                        for processed_identity in processed_identities
                    ],
                    "global_clock": strict_clock(pair, "global"),
                },
                "causal_resampled_close_availability": {
                    "rows": 3,
                    "raw_bar_label_semantics": (
                        "left_edge_of_[label,label+10s)_bar"
                    ),
                    "close_availability_mapping": "raw_bar_label_plus_10_seconds",
                    "decision_mapping": "minute_source_label_plus_60_seconds",
                    "every_close_available_no_later_than_decision": True,
                    "minute_source_clock_i64le_sha256": full_minute_clock_hash,
                    "last_raw_label_i64le_sha256": evidence_digest(
                        f"last-raw-label/{pair}"
                    ),
                    "close_availability_i64le_sha256": evidence_digest(
                        f"close-availability/{pair}"
                    ),
                    "decision_clock_i64le_sha256": evidence_digest(
                        f"causal-decision/{pair}"
                    ),
                    "resampled_close_f64le_sha256": evidence_digest(
                        f"resampled-close/{pair}"
                    ),
                    "minimum_availability_slack_seconds": 0,
                    "maximum_availability_slack_seconds": 0,
                },
                "processed_raw_derived_identity": {
                    "observation_rows": 24,
                    "observation_clock_i64le_sha256": evidence_digest(
                        f"raw-clock/{pair}"
                    ),
                    "one_minute_rows": 3,
                    "one_minute_clock_i64le_sha256": full_minute_clock_hash,
                    "one_minute_ohlcv_gap_f64le_sha256": full_minute_values_hash,
                },
                "raw_rebuilt_q1_feature_view": {
                    "projection_in_order": ["close"],
                    "rows": 2,
                    "source_clock_i64le_sha256": minute_clock_hash,
                    "ordered_float32_values_sha256": evidence_digest(
                        f"raw-values/{pair}"
                    ),
                },
                "cached_q1_feature_view": {
                    "projection_in_order": ["close"],
                    "rows": 2,
                    "source_clock_i64le_sha256": minute_clock_hash,
                    "ordered_float32_values_sha256": evidence_digest(
                        f"cached-values/{pair}"
                    ),
                },
                "live_provider_rolling_store": {
                    "rows": 2,
                    "source_clock_i64le_sha256": minute_clock_hash,
                    "ordered_ohlcv_gap_f64le_sha256": bounded_minute_values_hash,
                },
                "raw_source_loaded_and_rebuilt_once": True,
                "semantic_or_outcome_columns_read": False,
            }
        runtime_expected = {
            distribution: f"synthetic-pin-{index}"
            for index, distribution in enumerate(refresh.RUNTIME_DISTRIBUTIONS)
        }
        runtime_contract = {
            "authority": "ENVIRONMENT_libs.txt",
            "authority_file": identity(index_file, "ENVIRONMENT_libs.txt"),
            "python_environment": {
                "executable": "/synthetic/venv/bin/python",
                "prefix": "/synthetic/venv",
                "base_prefix": "/synthetic/base",
                "expected_executable": "/synthetic/venv/bin/python",
                "expected_prefix": "/synthetic/venv",
                "PYTHONNOUSERSITE": "1",
                "no_user_site_flag": 1,
                "user_site_enabled": False,
            },
            "expected_exact": runtime_expected,
            "actual_exact": runtime_expected,
            "all_exact": True,
        }
        artifact = {
            "schema": "m15-book-refresh-phase-zero-acceptance/v1",
            "acceptance_artifact": True,
            "all_six_targets_pass": True,
            "artifact_serialization": {
                "encoding": "UTF-8",
                "json_key_order": "lexicographic_recursive",
                "json_separators": [",", ":"],
                "allow_nan": False,
                "terminal_newline_bytes_hex": "0a",
            },
            "classification": "outcome_blind_pre_april_acceptance",
            "generation": {
                "working_directory": str(repo),
                "exact_executable_command": (
                    "/synthetic/venv/bin/python "
                    "scripts/m15_phase_zero_behavior_authority.py --mode acceptance"
                ),
                "python_executable": "/synthetic/venv/bin/python",
                "generator": "scripts/m15_phase_zero_behavior_authority.py",
                "action": "generate_and_atomically_create_canonical_json",
            },
            "implementation_sha256": {
                path.relative_to(repo).as_posix(): sha256(path) for path in sources
            },
            "target_pair_order": target_order,
            "feature_source_pair_order": source_order,
            "source_inputs": source_inputs,
            "source_inputs_sha256": source_inputs_sha256,
            "source_input_identities_pre_post_exact": True,
            "raw_source_rebuilds": raw_rebuilds,
            "eurusd_orderflow_behavior": {
                "cached_comparator": {
                    "source_kind": "cached_orderflow_comparator_projection",
                    "schema_in_order": [
                        {"name": "OF_synthetic", "type": "float", "nullable": True},
                        {
                            "name": "datetime_utc",
                            "type": "timestamp[ns, tz=UTC]",
                            "nullable": True,
                        },
                    ],
                    "projection_in_order": ["OF_synthetic"],
                    "semantic_or_outcome_columns_read": False,
                    "rows": 2,
                    "source_clock_i64le_sha256": minute_clock_hash,
                    "ordered_float32_values_sha256": evidence_digest(
                        "orderflow-values"
                    ),
                },
                "raw_rebuilt_from_processed": {
                    "source_kind": "rebuilt_from_strict_processed_eurusd_10s",
                    "recipe": "orderflow.build_of_features(orderflow.of_1m(raw_10s))",
                    "recipe_source": source_inputs[
                        "eurusd_rebuilt_orderflow_authority"
                    ]["recipe_source"],
                    "projection_in_order": ["OF_synthetic"],
                    "semantic_or_outcome_columns_read": False,
                    "rows": 2,
                    "source_clock_i64le_sha256": minute_clock_hash,
                    "ordered_float32_values_sha256": evidence_digest(
                        "orderflow-values"
                    ),
                    "same_strict_raw_load_as_eurusd_base_rebuild": True,
                },
                "source_clock_exact": True,
                "finite_mask_exact": True,
                "ordered_float32_values_exact": True,
            },
            "pairs": pair_rows,
            "recreate": {
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
            },
            "runtime_versions": {
                "python": phase_zero.sys.version.split()[0],
                **runtime_expected,
            },
            "runtime_version_contract": runtime_contract,
            "semantic_labels_or_outcomes_accessed": False,
            "unmodified_head_live_provider_counterfactual": {
                "production_provider_semantics_changed": True,
                "verification": (
                    "exact baseline/current source hashes plus one exact "
                    "close-conversion expression in each source"
                ),
                "baseline_unmodified_head": {
                    "commit": "b74a549f45bddfd6310d80c1f6c130de0076a95f",
                    "path": "scripts/live_features.py",
                    "git_blob_sha1": "bb55b4f17bd6e4f514435624bec2117a92ae7dad",
                    "sha256": (
                        "4903217812213cbeb6902a02552f7bddcc741fa665e8e700aee10c2b1716feb4"
                    ),
                    "log_return_close_dtype": "float64",
                },
                "working_tree_provider": {
                    "path": "scripts/live_features.py",
                    "sha256": sha256(repo / "scripts/live_features.py"),
                    "log_return_close_dtype": "float32",
                },
                "source_semantic_delta": {
                    "baseline_close_conversion": "Python float / NumPy float64",
                    "working_tree_close_conversion": "NumPy float32",
                    "behavioral_claims": "none_in_this_source_only_record",
                },
            },
            "window": {
                "clock": "left_labeled_source_feature_timestamp_plus_60_seconds",
                "entry_at_or_after": "2026-01-01T00:00:00Z",
                "entry_before": "2026-04-01T00:00:00Z",
            },
        }
        with mock.patch.object(phase_zero, "REPO_ROOT", repo), mock.patch.object(
            refresh, "source_files", return_value=sources
        ):
            equal(
                phase_zero._implementation_identities(),
                artifact["implementation_sha256"],
                "phase-zero generator/validator implementation inventory drift",
            )
        authority = repo / "phase-zero.json"
        refresh.atomic_json_new(authority, artifact)

        def validate_authority(
            path: Path,
            *,
            corrected_provider_sha256: str | None = None,
        ) -> dict[str, Any]:
            expected_provider_sha = corrected_provider_sha256 or sha256(
                repo / "scripts/live_features.py"
            )
            with mock.patch.object(refresh, "REPO_ROOT", repo), mock.patch.object(
                refresh, "PROCESSED_ROOT", repo / "processed"
            ), mock.patch.object(
                refresh, "source_files", return_value=sources
            ), mock.patch.object(
                refresh,
                "_phase_zero_current_incumbents",
                return_value=incumbent_contracts,
            ), mock.patch.object(
                refresh,
                "verified_runtime_version_contract",
                return_value=runtime_contract,
            ), mock.patch.object(
                refresh,
                "PHASE_ZERO_FLOAT32_LIVE_FEATURES_SHA256",
                expected_provider_sha,
            ):
                return refresh._validate_phase_zero_authority(path)

        accepted = validate_authority(authority)
        check(accepted["acceptance_artifact"], "six-target phase-zero contract rejected")
        with expect_raises(refresh.RefreshError, "corrected live provider"):
            validate_authority(
                authority,
                corrected_provider_sha256="0" * 64,
            )

        omitted_behavior = json.loads(json.dumps(artifact))
        omitted_behavior["pairs"]["USDJPY"].pop("gate_components")
        omitted_behavior_path = repo / "phase-zero-omitted-behavior.json"
        refresh.atomic_json_new(omitted_behavior_path, omitted_behavior)
        with expect_raises(refresh.RefreshError, "pair behavior fields"):
            validate_authority(omitted_behavior_path)

        forged_finite = json.loads(json.dumps(artifact))
        forged_finite["pairs"]["USDJPY"]["finite_eligibility"][
            "mismatch_count"
        ] = 1
        forged_finite_path = repo / "phase-zero-forged-finite.json"
        refresh.atomic_json_new(forged_finite_path, forged_finite)
        with expect_raises(refresh.RefreshError, "finite-eligibility equality"):
            validate_authority(forged_finite_path)

        truncated_scalar = json.loads(json.dumps(artifact))
        truncated_scalar["pairs"]["USDJPY"]["common_loadedbook_probability"][
            "loadedbook_scalar_parity"
        ]["probe_rows"] = 1
        truncated_scalar_path = repo / "phase-zero-truncated-scalar.json"
        refresh.atomic_json_new(truncated_scalar_path, truncated_scalar)
        with expect_raises(refresh.RefreshError, "scalar behavior"):
            validate_authority(truncated_scalar_path)

        diagnostic_scalar_bits = json.loads(json.dumps(artifact))
        diagnostic_scalar_bits["pairs"]["USDJPY"][
            "common_loadedbook_probability"
        ]["loadedbook_scalar_parity"][
            "scalar_batch_probability_bits_exact_diagnostic"
        ] = False
        diagnostic_scalar_bits_path = repo / "phase-zero-diagnostic-scalar-bits.json"
        refresh.atomic_json_new(
            diagnostic_scalar_bits_path, diagnostic_scalar_bits
        )
        validate_authority(diagnostic_scalar_bits_path)

        binding_scalar_bits = json.loads(json.dumps(artifact))
        binding_scalar_bits["pairs"]["USDJPY"][
            "common_loadedbook_probability"
        ]["loadedbook_scalar_parity"][
            "scalar_batch_probability_bits_binding"
        ] = True
        binding_scalar_bits_path = repo / "phase-zero-binding-scalar-bits.json"
        refresh.atomic_json_new(binding_scalar_bits_path, binding_scalar_bits)
        with expect_raises(refresh.RefreshError, "scalar behavior"):
            validate_authority(binding_scalar_bits_path)

        for field in (
            "cached_raw_probability_bits_exact",
            "cached_raw_direction_exact",
            "cached_raw_gate_exact",
            "scalar_batch_direction_exact",
            "scalar_batch_confidence_gate_exact",
            "scalar_batch_structural_gate_exact",
            "scalar_batch_combined_gate_exact",
            "scalar_batch_gate_exact",
            "behavior_exact",
        ):
            forged_scalar = json.loads(json.dumps(artifact))
            forged_scalar["pairs"]["USDJPY"][
                "common_loadedbook_probability"
            ]["loadedbook_scalar_parity"][field] = False
            forged_scalar_path = repo / f"phase-zero-forged-scalar-{field}.json"
            refresh.atomic_json_new(forged_scalar_path, forged_scalar)
            with expect_raises(refresh.RefreshError, "scalar behavior"):
                validate_authority(forged_scalar_path)

        missing_scalar_field = json.loads(json.dumps(artifact))
        missing_scalar_field["pairs"]["USDJPY"][
            "common_loadedbook_probability"
        ]["loadedbook_scalar_parity"].pop("scalar_batch_combined_gate_exact")
        missing_scalar_field_path = repo / "phase-zero-missing-scalar-field.json"
        refresh.atomic_json_new(missing_scalar_field_path, missing_scalar_field)
        with expect_raises(refresh.RefreshError, "scalar evidence fields"):
            validate_authority(missing_scalar_field_path)

        forged_batch_probability = json.loads(json.dumps(artifact))
        forged_batch_probability["pairs"]["USDJPY"][
            "common_loadedbook_probability"
        ]["bit_mismatch_count"] = 1
        forged_batch_probability_path = repo / "phase-zero-forged-batch.json"
        refresh.atomic_json_new(
            forged_batch_probability_path, forged_batch_probability
        )
        with expect_raises(refresh.RefreshError, "probability equality"):
            validate_authority(forged_batch_probability_path)

        contradictory_features = json.loads(json.dumps(artifact))
        feature_name = incumbent_contracts["USDJPY"]["feature_cols"][0]
        contradictory_features["pairs"]["USDJPY"]["ordered_float32_features"][
            "decoded_diagnostics"
        ] = {
            "mismatched_cells": 1,
            "finite_nonfinite_disagreements": 0,
            "columns": {
                feature_name: {
                    "mismatched_cells": 1,
                    "finite_nonfinite_disagreements": 0,
                    "max_abs_difference": 1.0,
                    "max_ulp_difference": 1,
                }
            },
        }
        contradictory_features_path = repo / "phase-zero-contradictory-features.json"
        refresh.atomic_json_new(
            contradictory_features_path, contradictory_features
        )
        with expect_raises(refresh.RefreshError, "contradicts decoded drift"):
            validate_authority(contradictory_features_path)

        contradictory_gate = json.loads(json.dumps(artifact))
        contradictory_gate["pairs"]["USDJPY"]["gate_components"]["confidence"][
            "cached_pass_count"
        ] = 0
        contradictory_gate["pairs"]["USDJPY"]["gate_components"]["confidence"][
            "raw_rebuilt_pass_count"
        ] = 0
        contradictory_gate_path = repo / "phase-zero-contradictory-gate.json"
        refresh.atomic_json_new(contradictory_gate_path, contradictory_gate)
        with expect_raises(refresh.RefreshError, "exceeds its components"):
            validate_authority(contradictory_gate_path)

        contradictory_probability = json.loads(json.dumps(artifact))
        full_probability = contradictory_probability["pairs"]["USDJPY"][
            "full_population_ordered_pre_schedule"
        ]
        full_probability["cached_probability_f64le_sha256"] = "f" * 64
        full_probability["raw_rebuilt_probability_f64le_sha256"] = "f" * 64
        contradictory_probability_path = (
            repo / "phase-zero-contradictory-probability.json"
        )
        refresh.atomic_json_new(
            contradictory_probability_path, contradictory_probability
        )
        with expect_raises(refresh.RefreshError, "full-population behavior"):
            validate_authority(contradictory_probability_path)

        post_april_provider = json.loads(json.dumps(artifact))
        post_april_provider["pairs"]["USDJPY"]["provider_qualification"][
            "public_live_feature_builder_parity"
        ]["public_latest_source_feature_ns"] = int(
            pd.Timestamp("2026-04-01T00:00:00Z").value
        )
        post_april_provider_path = repo / "phase-zero-post-april-provider.json"
        refresh.atomic_json_new(post_april_provider_path, post_april_provider)
        with expect_raises(refresh.RefreshError, "provider behavior"):
            validate_authority(post_april_provider_path)

        omitted_strict = json.loads(json.dumps(artifact))
        omitted_strict["raw_source_rebuilds"]["EURUSD"][
            "strict_processed_source"
        ].pop("global_clock")
        omitted_strict_path = repo / "phase-zero-omitted-strict.json"
        refresh.atomic_json_new(omitted_strict_path, omitted_strict)
        with expect_raises(refresh.RefreshError, "strict processed-source"):
            validate_authority(omitted_strict_path)

        forged_orderflow = json.loads(json.dumps(artifact))
        forged_orderflow["eurusd_orderflow_behavior"]["raw_rebuilt_from_processed"][
            "ordered_float32_values_sha256"
        ] = "f" * 64
        forged_orderflow_path = repo / "phase-zero-forged-orderflow.json"
        refresh.atomic_json_new(forged_orderflow_path, forged_orderflow)
        with expect_raises(refresh.RefreshError, "order-flow equality"):
            validate_authority(forged_orderflow_path)

        forged = json.loads(json.dumps(artifact))
        forged["pairs"]["USDJPY"]["inputs"]["incumbent_book"][
            "ordered_files"
        ][-1] = identity(
            sources[0], sources[0].relative_to(repo).as_posix()
        )
        forged_path = repo / "phase-zero-forged-incumbent.json"
        refresh.atomic_json_new(forged_path, forged)
        with expect_raises(refresh.RefreshError, "file order/path differs"):
            validate_authority(forged_path)

        reordered = json.loads(json.dumps(artifact))
        reordered_files = reordered["pairs"]["USDJPY"]["inputs"][
            "incumbent_book"
        ]["ordered_files"]
        reordered_files[-1], reordered_files[-2] = (
            reordered_files[-2],
            reordered_files[-1],
        )
        reordered_path = repo / "phase-zero-reordered-incumbent.json"
        refresh.atomic_json_new(reordered_path, reordered)
        with expect_raises(refresh.RefreshError, "file order/path differs"):
            validate_authority(reordered_path)

        reordered_models = json.loads(json.dumps(artifact))
        model_order = reordered_models["pairs"]["USDJPY"]["inputs"][
            "incumbent_book"
        ]["model_order"]
        model_order[0], model_order[1] = model_order[1], model_order[0]
        reordered_models_path = repo / "phase-zero-reordered-models.json"
        refresh.atomic_json_new(reordered_models_path, reordered_models)
        with expect_raises(refresh.RefreshError, "incumbent identity differs"):
            validate_authority(reordered_models_path)

        omitted = json.loads(json.dumps(artifact))
        omitted["source_inputs"][
            "processed_raw_source_groups_in_feature_source_order"
        ][4]["ordered_files"].pop()
        omitted["source_inputs_sha256"] = refresh.sha256_bytes(
            refresh.canonical_bytes(omitted["source_inputs"])
        )
        omitted_path = repo / "phase-zero-omitted-q1-source.json"
        refresh.atomic_json_new(omitted_path, omitted)
        with expect_raises(refresh.RefreshError, "processed source authority"):
            validate_authority(omitted_path)

        reordered_sources = json.loads(json.dumps(artifact))
        cached_sources = reordered_sources["source_inputs"][
            "cached_feature_sources_in_feature_source_order"
        ]
        cached_sources[-1], cached_sources[-2] = cached_sources[-2], cached_sources[-1]
        reordered_sources["source_inputs_sha256"] = refresh.sha256_bytes(
            refresh.canonical_bytes(reordered_sources["source_inputs"])
        )
        reordered_sources_path = repo / "phase-zero-reordered-sources.json"
        refresh.atomic_json_new(reordered_sources_path, reordered_sources)
        with expect_raises(refresh.RefreshError, "preserve source order"):
            validate_authority(reordered_sources_path)

        rejected = json.loads(json.dumps(artifact))
        rejected["pairs"].pop("NZDUSD")
        rejected_path = repo / "phase-zero-missing-pair.json"
        refresh.atomic_json_new(rejected_path, rejected)
        with expect_raises(refresh.RefreshError, "exactly all six"):
            validate_authority(rejected_path)

        runtime_drift = json.loads(json.dumps(artifact))
        runtime_drift["runtime_versions"]["numpy"] = "0.0.synthetic-drift"
        runtime_drift_path = repo / "phase-zero-runtime-drift.json"
        refresh.atomic_json_new(runtime_drift_path, runtime_drift)
        with expect_raises(refresh.RefreshError, "runtime"):
            validate_authority(runtime_drift_path)

        python_runtime_drift = json.loads(json.dumps(artifact))
        python_runtime_drift["runtime_versions"]["python"] = "0.0.synthetic"
        python_runtime_drift_path = repo / "phase-zero-python-runtime-drift.json"
        refresh.atomic_json_new(python_runtime_drift_path, python_runtime_drift)
        with expect_raises(refresh.RefreshError, "runtime"):
            validate_authority(python_runtime_drift_path)

    # Cached pre-April parity may request only the sealed covariate projection.
    with tempfile.TemporaryDirectory(prefix="m15-parity-projection-") as raw:
        temporary = Path(raw)
        workspace = temporary / "workspace"
        (workspace / "feature_snapshot").mkdir(parents=True)
        (temporary / "features").mkdir()
        (temporary / "features_of").mkdir()
        refresh.atomic_json_new(
            workspace / "feature_snapshot" / "data_lock.json",
            {
                "schema": "m15-book-refresh-data-lock/v1",
                "semantic_replay_outcomes_accessed": False,
                "base": {"USDJPY": {"columns": ["covariate", "close"]}},
                "orderflow": {"EURUSD": {"columns": ["OF_covariate"]}},
            },
        )
        clock = pd.DatetimeIndex(
            [
                "2025-12-31T23:58:00Z",
                "2025-12-31T23:59:00Z",
                "2026-03-31T23:58:00Z",
                "2026-03-31T23:59:00Z",
                "2026-04-02T00:00:00Z",
            ],
            name="datetime_utc",
        ).as_unit("ns")
        pd.DataFrame(
            {
                "covariate": np.arange(len(clock), dtype="float32"),
                "close": np.ones(len(clock), dtype="float32"),
                "y": ["poison"] * len(clock),
                "fwd_ret": np.arange(len(clock), dtype="float64"),
                "valid": np.ones(len(clock), dtype=bool),
            },
            index=clock,
        ).to_parquet(temporary / "features" / "USDJPY_2026.parquet")
        pd.DataFrame(
            {
                "OF_covariate": np.arange(len(clock), dtype="float32"),
                "target": ["poison"] * len(clock),
            },
            index=clock,
        ).to_parquet(temporary / "features_of" / "EURUSD_2026.parquet")
        real_read_parquet = refresh.pd.read_parquet
        requested: list[dict[str, Any]] = []

        def projected(path: Path, **kwargs: Any) -> pd.DataFrame:
            requested.append({"path": Path(path), **kwargs})
            check(kwargs.get("engine") == "pyarrow", "cached parity engine")
            filters = kwargs.get("filters")
            check(isinstance(filters, list) and len(filters) == 2, "cached Q1 filters")
            equal(filters[0][:2], ("datetime_utc", ">="), "cached lower predicate")
            equal(filters[1][:2], ("datetime_utc", "<"), "cached upper predicate")
            return real_read_parquet(path, **kwargs)

        with mock.patch.object(refresh, "REPO_ROOT", temporary), mock.patch.object(
            refresh,
            "load_spec",
            return_value={
                "pair_order": ["USDJPY"],
                "feature_source_pair_order": ["USDJPY"],
            },
        ), mock.patch.object(refresh.pd, "read_parquet", side_effect=projected):
            refresh._make_cached_parity_view(workspace)
        equal(
            [tuple(row["columns"]) for row in requested],
            [("covariate", "close"), ("OF_covariate",)],
            "cached parity did not use exact sealed projections",
        )
        check(
            not any(
                column in refresh.FORBIDDEN_SEMANTIC_COLUMNS
                for row in requested
                for column in row["columns"]
            ),
            "cached parity requested a semantic column",
        )
        for output in (
            workspace / "cached_pre_april_parity/features/USDJPY_2026.parquet",
            workspace / "cached_pre_april_parity/features_of/EURUSD_2026.parquet",
        ):
            decoded = pd.read_parquet(output)
            equal(len(decoded), 2, "cached parity output escaped exact causal Q1")
            check(
                decoded.index.min() == pd.Timestamp("2025-12-31T23:59:00Z")
                and decoded.index.max() == pd.Timestamp("2026-03-31T23:58:00Z"),
                "cached parity output clock",
            )

        def ignored_predicate(path: Path, **kwargs: Any) -> pd.DataFrame:
            columns = list(kwargs["columns"])
            return pd.DataFrame(
                {column: np.ones(1, dtype="float32") for column in columns},
                index=pd.DatetimeIndex(
                    ["2026-04-01T00:00:00Z"], name="datetime_utc"
                ),
            )

        with mock.patch.object(
            refresh.pd, "read_parquet", side_effect=ignored_predicate
        ), expect_raises(refresh.RefreshError, "exact causal Q1 clock"):
            refresh._read_cached_parity_projection(
                temporary / "features/USDJPY_2026.parquet",
                ["covariate", "close"],
            )

    # Exact inventory mode detects removal; historical subset verification
    # remains intentionally available for narrowed year calls.
    fake_paths = [Path("model-a"), Path("model-b")]
    fake_rows = {
        path: refresh.FileIdentity(path.name, 1, hashlib.sha256(path.name.encode()).hexdigest())
        for path in fake_paths
    }
    prereg = {
        "comparator_A_files": [
            vars(fake_rows[path]) for path in fake_paths
        ]
    }
    with mock.patch.object(
        refresh, "file_identity", side_effect=lambda path: fake_rows[Path(path)]
    ):
        with expect_raises(refresh.RefreshError, "exact path/count/order"):
            refresh._verify_preregistered_inventory(
                prereg,
                "comparator_A_files",
                fake_paths[:1],
                exact=True,
            )
        refresh._verify_preregistered_inventory(
            prereg, "comparator_A_files", fake_paths[:1]
        )

    # The measured source gate is the exact full sealed 2026 inventory, not
    # merely whichever support-window subset remains discoverable.
    with tempfile.TemporaryDirectory(prefix="m15-processed-exact-") as raw:
        processed_root = Path(raw) / "processed"
        source_pairs = refresh.load_spec()["feature_source_pair_order"]
        for pair in source_pairs:
            pair_root = processed_root / pair
            pair_root.mkdir(parents=True)
            for date in ("2026-01-01", "2026-05-31"):
                (pair_root / f"{pair}_10s_{date}.parquet").write_bytes(
                    f"{pair}/{date}".encode()
                )
        with mock.patch.object(refresh, "PROCESSED_ROOT", processed_root):
            processed_prereg = {
                "processed_source_files": {
                    pair: refresh._source_file_identities(pair)
                    for pair in source_pairs
                }
            }
            refresh._verify_all_processed_source_identities(processed_prereg)
            (processed_root / "NZDUSD/NZDUSD_10s_2026-05-31.parquet").unlink()
            with expect_raises(refresh.RefreshError, "path/order changed"):
                refresh._verify_all_processed_source_identities(processed_prereg)

    # Read-only evidence is already mode 0444 when its destination link appears.
    with tempfile.TemporaryDirectory(prefix="m15-atomic-mode-") as raw:
        destination = Path(raw) / "sealed.json"
        real_link = os.link
        observed_modes: list[int] = []

        def guarded_link(source: str | Path, target: str | Path) -> None:
            observed_modes.append(Path(source).stat().st_mode & 0o777)
            real_link(source, target)

        with mock.patch.object(refresh.os, "link", side_effect=guarded_link):
            refresh.atomic_json_new(destination, {"schema": "born-read-only/v1"})
        equal(observed_modes, [0o444], "evidence inode was writable at publication")
        equal(destination.stat().st_mode & 0o777, 0o444, "sealed destination mode")

    # Canonical/audit fits must repeat both the bundle identity and the exact
    # ordered best-iteration vector before any arm can be sealed.
    fit_identity = {
        "bundle_id": "a" * 64,
        "best_iterations": [11, 13, 17],
    }
    refresh._assert_fit_repeat_identity(
        fit_identity,
        dict(fit_identity),
        pair="USDJPY",
        arm="B",
    )
    with expect_raises(refresh.RefreshError, "bundle IDs differ"):
        refresh._assert_fit_repeat_identity(
            fit_identity,
            dict(fit_identity, bundle_id="b" * 64),
            pair="USDJPY",
            arm="B",
        )
    with expect_raises(refresh.RefreshError, "best iterations differ"):
        refresh._assert_fit_repeat_identity(
            fit_identity,
            dict(fit_identity, best_iterations=[11, 13, 19]),
            pair="USDJPY",
            arm="B",
        )

    # The semantic look marker is durable and makes every unsealed restart fail
    # closed, including unknown residue and a symlinked replay directory.
    def marker_workspace(root: Path) -> tuple[Path, dict[str, Any], dict[str, Any]]:
        workspace = root / "workspace"
        (workspace / "arm_seal").mkdir(parents=True)
        (workspace / "phases").mkdir()
        run_id = "a" * 64
        prereg_value = {"prereg_id": "b" * 64, "implementation_git_sha": "c" * 40}
        binding_value = {"run_id": run_id, "evaluator_id": "d" * 64}
        refresh.atomic_json_new(
            workspace / "arm_seal" / f"run_binding_{run_id}.json",
            binding_value,
        )
        refresh.atomic_json_new(
            workspace / "phases" / "04_arm_seal.json",
            {"schema": "synthetic-arm-seal/v1"},
        )
        return workspace, prereg_value, binding_value

    with tempfile.TemporaryDirectory(prefix="m15-replay-marker-") as raw:
        workspace, prereg_value, binding_value = marker_workspace(Path(raw))
        marker = refresh._seal_replay_start_marker(
            workspace, prereg_value, binding_value
        )
        equal(marker.stat().st_mode & 0o777, 0o444, "replay marker mode")
        with expect_raises(refresh.RefreshError, "quarantined"):
            refresh._seal_replay_start_marker(workspace, prereg_value, binding_value)
    with tempfile.TemporaryDirectory(prefix="m15-replay-unknown-") as raw:
        workspace, prereg_value, binding_value = marker_workspace(Path(raw))
        (workspace / "replay").mkdir()
        (workspace / "replay" / "foreign.partial").write_bytes(b"foreign")
        with expect_raises(refresh.RefreshError, "unknown partial"):
            refresh._seal_replay_start_marker(workspace, prereg_value, binding_value)
    with tempfile.TemporaryDirectory(prefix="m15-replay-symlink-") as raw:
        root = Path(raw)
        workspace, prereg_value, binding_value = marker_workspace(root)
        outside = root / "outside"
        outside.mkdir()
        os.symlink(outside, workspace / "replay")
        with expect_raises(refresh.RefreshError, "regular directory"):
            refresh._seal_replay_start_marker(workspace, prereg_value, binding_value)

    # The replay reader authenticates the exact arm JSON bytes it parses, and
    # a persistent arm-seal mutation after the one-look marker is caught again
    # before replay output can be sealed.  The marker remains quarantining.
    with tempfile.TemporaryDirectory(prefix="m15-arm-seal-toctou-") as raw:
        root = Path(raw)
        workspace, prereg_value, binding_value = marker_workspace(root)
        run_id = binding_value["run_id"]
        binding_path = workspace / "arm_seal" / f"run_binding_{run_id}.json"
        repeat_path = workspace / "arm_seal" / f"repeatability_{run_id}.json"
        refresh.atomic_json_new(
            repeat_path,
            {"schema": "synthetic-repeatability", "run_id": run_id},
        )
        arm_hashes = {
            f"workspace:arm_seal/{binding_path.name}": sha256(binding_path),
            f"workspace:arm_seal/{repeat_path.name}": sha256(repeat_path),
        }
        equal(
            replay._read_arm_seal_json(workspace, repeat_path, arm_hashes)["run_id"],
            run_id,
            "arm-seal exact-byte reader",
        )

        class SyntheticArmStore:
            def has(self, _phase: str) -> bool:
                return False

            def load(self, phase: str) -> dict[str, Any]:
                equal(phase, "arm_seal", "unexpected synthetic phase load")
                if sha256(repeat_path) != arm_hashes[
                    f"workspace:arm_seal/{repeat_path.name}"
                ]:
                    raise refresh.RefreshError(
                        "phase hash verification failed: synthetic repeatability"
                    )
                return {"output_hashes": dict(arm_hashes)}

            def seal(self, *_args: Any, **_kwargs: Any) -> None:
                raise GateFailure("mutated replay unexpectedly reached phase sealing")

        synthetic_store = SyntheticArmStore()

        def mutate_arm_seal(*_args: Any, **_kwargs: Any) -> object:
            repeat_path.chmod(0o600)
            repeat_path.write_bytes(b'{"schema":"persistently-mutated"}\n')
            repeat_path.chmod(0o444)
            return object()

        with mock.patch.object(
            refresh,
            "workspace_for",
            return_value=(workspace, synthetic_store, prereg_value),
        ), mock.patch.object(
            refresh, "_run_binding", return_value=binding_value
        ), mock.patch.object(
            book_runtime, "load_target_books", return_value={}
        ), mock.patch.object(
            book_runtime, "load_index", return_value={}
        ), mock.patch.object(
            refresh, "_verify_loaded_comparator_binding"
        ), mock.patch.object(
            refresh, "_verify_all_processed_source_identities"
        ), mock.patch.object(
            replay, "run_workspace_replay", side_effect=mutate_arm_seal
        ), expect_raises(refresh.RefreshError, "phase hash verification failed"):
            refresh.run_replay(prereg_value["prereg_id"])
        marker = workspace / "replay" / f"replay_started_{run_id}.json"
        check(marker.is_file(), "arm mutation failure lost the durable replay marker")
        with expect_raises(refresh.RefreshError, "quarantined"):
            refresh._seal_replay_start_marker(
                workspace, prereg_value, binding_value
            )

    # Recompute every run-binding identity and predecessor hash before scoring.
    with tempfile.TemporaryDirectory(prefix="m15-binding-") as raw:
        workspace = Path(raw) / "workspace"
        (workspace / "arm_seal").mkdir(parents=True)
        (workspace / "phases").mkdir()
        spec = refresh.load_spec()
        prereg_id = "1" * 64
        implementation = "2" * 40
        evaluator = refresh.evaluator_id(spec)
        control_arms = [
            [family, arm, hashlib.sha256(f"control/{family}/{arm}".encode()).hexdigest()]
            for family, arm in spec["control_order"]
        ]
        controls_id = refresh.derive_controls_id(spec, control_arms)
        primary_arms = [
            [pair, arm, hashlib.sha256(f"primary/{pair}/{arm}".encode()).hexdigest()]
            for pair in spec["pair_order"]
            for arm in spec["arm_order"]
        ]
        run_id = refresh.derive_run_id(spec, prereg_id, controls_id, primary_arms)
        refresh.atomic_json_new(
            workspace / "source_preregistration.json",
            {
                "prereg_id": prereg_id,
                "implementation_git_sha": implementation,
                "evaluator_id": evaluator,
                "canonical_encoding": spec["identity"],
            },
        )
        predecessor_hashes: dict[str, str] = {}
        for index, name in enumerate(
            ("source_preregistration", "derived_data", "adapter_parity", "fit_and_repeat")
        ):
            path = workspace / "phases" / f"{index:02d}_{name}.json"
            refresh.atomic_json_new(path, {"phase": name})
            predecessor_hashes[name] = sha256(path)
        binding = {
            "schema": "m15-book-refresh-run-binding/v1",
            "run_id": run_id,
            "prereg_id": prereg_id,
            "implementation_git_sha": implementation,
            "evaluator_id": evaluator,
            "controls_id": controls_id,
            "control_arms": control_arms,
            "primary_arms": primary_arms,
            "canonical_encoding": spec["identity"],
            "predecessor_manifest_hashes": predecessor_hashes,
            "semantic_replay_outcomes_accessed": False,
        }
        replay._validate_run_binding(
            spec,
            workspace,
            binding,
            requested_prereg_id=prereg_id,
            requested_run_id=run_id,
        )
        mutations = []
        wrong_primary = json.loads(json.dumps(binding))
        wrong_primary["primary_arms"][0], wrong_primary["primary_arms"][1] = (
            wrong_primary["primary_arms"][1], wrong_primary["primary_arms"][0]
        )
        mutations.append(wrong_primary)
        mutations.append(dict(binding, controls_id="f" * 64))
        mutations.append(dict(binding, run_id="e" * 64))
        mutations.append(dict(binding, implementation_git_sha="3" * 40))
        for changed in mutations:
            with expect_raises(replay.ReplayInputError):
                replay._validate_run_binding(
                    spec,
                    workspace,
                    changed,
                    requested_prereg_id=None,
                    requested_run_id=None,
                )

    # Synthetic processed sources are projected to clock+close, trimmed to the
    # exact support window, and authenticated both before and after the read.
    with tempfile.TemporaryDirectory(prefix="m15-observation-window-") as raw:
        processed = Path(raw) / "processed" / "USDJPY"
        processed.mkdir(parents=True)

        def write_raw(
            path: Path,
            clock: pd.DatetimeIndex,
            *,
            pandas_index_metadata: bool = False,
            datetime_unit: str = "ns",
            open_dtype: str = "float64",
            timezone_aware: bool = True,
        ) -> None:
            count = len(clock)
            stored_clock = clock if timezone_aware else clock.tz_localize(None)
            frame = pd.DataFrame(
                {
                    "open": np.ones(count, dtype=open_dtype),
                    "high": np.ones(count, dtype="float64"),
                    "low": np.ones(count, dtype="float64"),
                    "close": 1.0 + np.arange(count, dtype="float64") * 1e-5,
                    "volume": np.ones(count, dtype="float64"),
                    "datetime_utc": stored_clock.as_unit(datetime_unit),
                }
            )
            if pandas_index_metadata:
                frame = frame.set_index("datetime_utc")
            frame.to_parquet(path, index=pandas_index_metadata)

        inside = processed / "USDJPY_10s_2025-01-01.parquet"
        inside_us = processed / "USDJPY_10s_2025-01-02.parquet"
        later = processed / "USDJPY_10s_2025-01-03.parquet"
        write_raw(
            inside,
            pd.DatetimeIndex(
                [
                    "2024-12-31T23:59:50Z",
                    "2025-01-01T00:00:00Z",
                    "2025-01-01T23:59:40Z",
                ]
            ),
            pandas_index_metadata=True,
        )
        write_raw(
            inside_us,
            pd.DatetimeIndex(
                [
                    "2025-01-02T00:00:00Z",
                    "2025-01-02T23:59:40Z",
                ]
            ),
            pandas_index_metadata=True,
            datetime_unit="us",
        )
        write_raw(later, pd.DatetimeIndex(["2025-01-03T00:00:00Z"]))
        real_read_table = replay.pq.read_table
        discovery_calls: list[dict[str, Any]] = []

        def discovery_spy(*args: Any, **kwargs: Any) -> Any:
            discovery_calls.append(dict(kwargs))
            return real_read_table(*args, **kwargs)

        with mock.patch.object(replay.pq, "read_table", side_effect=discovery_spy):
            selected = replay.discover_observation_files(
                "USDJPY",
                processed_root=Path(raw) / "processed",
                nominal_year="2025",
                observation_at_or_after="2025-01-01T00:00:00Z",
                observation_before="2025-01-03T00:00:00Z",
            )
        equal(
            selected,
            (inside, inside_us),
            "mixed-unit processed sources were not selected in sealed order",
        )
        check(
            all(call.get("columns") == ["datetime_utc"] for call in discovery_calls),
            "observation discovery requested a semantic value column",
        )
        check(
            all(call.get("filters") is None for call in discovery_calls),
            "observation discovery unexpectedly filtered the physical clock",
        )
        check(
            all(call.get("use_threads") is False for call in discovery_calls),
            "observation discovery enabled nondeterministic threaded reads",
        )
        check(
            all(
                call.get("use_pandas_metadata") is False
                for call in discovery_calls
            ),
            "observation discovery enabled pandas metadata reconstruction",
        )
        full_inventory = [
            {
                "path": f"processed/USDJPY/{path.name}",
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in (inside, inside_us, later)
        ]
        selected_inventory = replay._verify_observation_inventory(
            "USDJPY", selected, full_inventory
        )
        load_calls: list[dict[str, Any]] = []

        def read_spy(*args: Any, **kwargs: Any) -> Any:
            load_calls.append(dict(kwargs))
            return real_read_table(*args, **kwargs)

        with mock.patch.object(replay.pq, "read_table", side_effect=read_spy):
            observation = replay.load_observation_series(
                selected,
                expected_inventory=selected_inventory,
                pair="USDJPY",
                observation_at_or_after="2025-01-01T00:00:00Z",
                observation_before="2025-01-03T00:00:00Z",
            )
        check(
            all(call.get("columns") == ["datetime_utc", "close"] for call in load_calls),
            "observation loader requested columns beyond clock+close",
        )
        check(all(call.get("filters") for call in load_calls), "observation filter absent")
        check(
            all(call.get("use_threads") is False for call in load_calls),
            "observation projection enabled nondeterministic threaded reads",
        )
        check(
            all(call.get("use_pandas_metadata") is False for call in load_calls),
            "observation projection enabled pandas metadata reconstruction",
        )
        lower_s = int(pd.Timestamp("2025-01-01T00:00:00Z").timestamp())
        upper_s = int(pd.Timestamp("2025-01-03T00:00:00Z").timestamp())
        check(
            np.all(
                (observation.timestamps_s >= lower_s)
                & (observation.timestamps_s < upper_s)
            ),
            "observation rows escaped sealed support window",
        )

        def mutate_after_read(*args: Any, **kwargs: Any) -> Any:
            table = real_read_table(*args, **kwargs)
            with inside.open("ab") as handle:
                handle.write(b"mutation")
            return table

        with mock.patch.object(
            replay.pq, "read_table", side_effect=mutate_after_read
        ), expect_raises(replay.ReplayInputError, "path/size differs"):
            replay.load_observation_series(
                selected,
                expected_inventory=selected_inventory,
                pair="USDJPY",
                observation_at_or_after="2025-01-01T00:00:00Z",
                observation_before="2025-01-03T00:00:00Z",
            )

        invalid_clock_ms = processed / "invalid_clock_ms.parquet"
        invalid_clock_naive = processed / "invalid_clock_naive.parquet"
        invalid_value = processed / "invalid_value.parquet"
        one_clock = pd.DatetimeIndex(["2025-01-01T00:00:00Z"])
        write_raw(invalid_clock_ms, one_clock, datetime_unit="ms")
        write_raw(
            invalid_clock_naive,
            one_clock,
            datetime_unit="us",
            timezone_aware=False,
        )
        write_raw(invalid_value, one_clock, open_dtype="float32")
        for invalid_path in (invalid_clock_ms, invalid_clock_naive, invalid_value):
            with expect_raises(replay.ReplayInputError, "physical types differ"):
                replay._validate_observation_schema(invalid_path)

        source_only_dir = Path(raw) / "processed" / "USDCAD"
        source_only_dir.mkdir()
        source_only = source_only_dir / "USDCAD_10s_2025-01-01.parquet"
        write_raw(source_only, one_clock, pandas_index_metadata=True)
        equal(
            replay.discover_observation_files(
                "USDCAD",
                processed_root=Path(raw) / "processed",
                nominal_year="2025",
            ),
            (source_only,),
            "source-only feature pair was rejected by processed discovery",
        )
        with expect_raises(replay.ReplayInputError, "unknown pair"):
            replay.discover_observation_files(
                "EURGBP",
                processed_root=Path(raw) / "processed",
                nominal_year="2025",
            )

    # Pair-isolated defects reserve every main/control endpoint slot and yield
    # sealed INVALID records; shared inference code still runs over fixed m0.
    spec = refresh.load_spec()
    representatives = {
        pair: family
        for family, pair in spec["control_family"]["representatives"].items()
    }
    invalid_inputs: dict[str, replay.InvalidPairReplayInput] = {}
    for pair in stats.PAIR_ORDER:
        family = spec["pairs"][pair]["adapter_family"]
        invalid_inputs[pair] = replay.InvalidPairReplayInput(
            pair=pair,
            adapter_family=family,
            primary_bundle_ids={
                arm: hashlib.sha256(f"{pair}/{arm}".encode()).hexdigest()
                for arm in replay.PRIMARY_ARMS
            },
            control_bundle_ids=(
                {
                    arm: hashlib.sha256(f"{family}/{arm}".encode()).hexdigest()
                    for arm in replay.CONTROL_ARMS
                }
                if pair in representatives
                else {}
            ),
            invalid_reasons=("pair_input_defect:synthetic",),
        )

    class FakeEndpointInference:
        def __init__(self, endpoint: stats.DateRatioEndpoint):
            self.name = endpoint.name
            self.computable = False
            self.lower = None
            self.upper = None

        def as_dict(self) -> dict[str, Any]:
            return {
                "name": self.name,
                "computable": False,
                "lower": None,
                "upper": None,
            }

    class FakeFamilyInference:
        def __init__(self, endpoints: Sequence[stats.DateRatioEndpoint]):
            self.endpoints = tuple(FakeEndpointInference(row) for row in endpoints)

        def as_dict(self) -> dict[str, Any]:
            return {"endpoint_count": len(self.endpoints), "computable": False}

    with mock.patch.object(
        replay,
        "master_calendar_from_endpoints",
        return_value=np.array([20260401], dtype="int64"),
    ), mock.patch.object(
        replay,
        "infer_main_family",
        side_effect=lambda endpoints, **_kwargs: FakeFamilyInference(endpoints),
    ), mock.patch.object(
        replay,
        "infer_control_family",
        side_effect=lambda endpoints, **_kwargs: FakeFamilyInference(endpoints),
    ):
        invalid_campaign = replay.run_common_replay(spec, invalid_inputs)
    equal(
        invalid_campaign.joint_result["main_inference"]["endpoint_count"],
        stats.MAIN_FAMILY_SIZE,
        "pair INVALID shrank main family",
    )
    equal(
        invalid_campaign.joint_result["control_inference"]["endpoint_count"],
        9,
        "pair INVALID shrank control family",
    )
    for pair in stats.PAIR_ORDER:
        equal(
            invalid_campaign.joint_result["statuses"][pair]["status"],
            "INVALID",
            f"{pair} pair defect did not seal INVALID",
        )

    # Even valid-looking pair inputs complete the whole six-target model/schema
    # preflight pass before the first pair enters evaluation (which is the only
    # place an observation close can be opened).
    class StubPolicy:
        def __init__(self, pair: str, arm: str):
            self.arm = arm
            self.bundle_id = hashlib.sha256(f"preflight/{pair}/{arm}".encode()).hexdigest()

    class StubPairInput:
        def __init__(self, pair: str):
            family = spec["pairs"][pair]["adapter_family"]
            self.pair = pair
            self.primary_policies = {
                arm: StubPolicy(pair, arm) for arm in replay.PRIMARY_ARMS
            }
            self.control_policies = (
                {
                    arm: StubPolicy(pair, arm) for arm in replay.CONTROL_ARMS
                }
                if pair in representatives
                else {}
            )
            self.adapter_family = family

    preflight_inputs = {
        pair: StubPairInput(pair) for pair in stats.PAIR_ORDER
    }
    events: list[str] = []

    def preflight_all(pair_input: StubPairInput) -> None:
        events.append(f"preflight:{pair_input.pair}")

    def precompute_all(
        _spec: dict[str, Any], pair_input: StubPairInput
    ) -> dict[str, replay.ComparatorAScoreCache]:
        equal(
            len([event for event in events if event.startswith("preflight:")]),
            len(stats.PAIR_ORDER),
            "all-arm scoring began before all-pair model preflight completed",
        )
        events.append(f"all_scores:{pair_input.pair}")
        return {
            "A": replay.ComparatorAScoreCache(
                entry_ns_sha256="a" * 64,
                probabilities=np.array([0.9], dtype="float64"),
                pre_schedule_mask=np.array([True], dtype=bool),
            )
        }

    def evaluate_after_preflight(
        _spec: dict[str, Any],
        pair_input: StubPairInput,
        **_kwargs: Any,
    ) -> replay.PairComputation:
        equal(
            len([event for event in events if event.startswith("preflight:")]),
            len(stats.PAIR_ORDER),
            "evaluation began before all-pair preflight completed",
        )
        equal(
            len([event for event in events if event.startswith("all_scores:")]),
            len(stats.PAIR_ORDER),
            "evaluation began before all-pair all-arm scoring completed",
        )
        events.append(f"evaluate:{pair_input.pair}")
        raise replay.ReplayInputError("synthetic post-preflight pair defect")

    with mock.patch.object(
        replay, "_preflight_pair_policy_model_schemas", side_effect=preflight_all
    ), mock.patch.object(
        replay, "_precompute_pair_policy_scores", side_effect=precompute_all
    ), mock.patch.object(
        replay, "evaluate_pair", side_effect=evaluate_after_preflight
    ), mock.patch.object(
        replay,
        "master_calendar_from_endpoints",
        return_value=np.array([20260401], dtype="int64"),
    ), mock.patch.object(
        replay,
        "infer_main_family",
        side_effect=lambda endpoints, **_kwargs: FakeFamilyInference(endpoints),
    ), mock.patch.object(
        replay,
        "infer_control_family",
        side_effect=lambda endpoints, **_kwargs: FakeFamilyInference(endpoints),
    ):
        replay.run_common_replay(spec, preflight_inputs)  # type: ignore[arg-type]
    equal(
        events[: len(stats.PAIR_ORDER)],
        [f"preflight:{pair}" for pair in stats.PAIR_ORDER],
        "global model/schema preflight order",
    )
    equal(
        events[len(stats.PAIR_ORDER) : 2 * len(stats.PAIR_ORDER)],
        [f"all_scores:{pair}" for pair in stats.PAIR_ORDER],
        "global all-arm pre-outcome score order",
    )

    # A late-pair model mutation is encountered during the full all-arm cache
    # pass, before any earlier pair can enter evaluation/observation access.
    mutation_events: list[str] = []

    def precompute_with_last_pair_mutation(
        _spec: dict[str, Any], pair_input: StubPairInput
    ) -> dict[str, replay.ComparatorAScoreCache]:
        mutation_events.append(f"all_scores:{pair_input.pair}")
        if pair_input.pair == stats.PAIR_ORDER[-1]:
            raise replay.ReplayInputError("synthetic sealed model mutation")
        return {
            "A": replay.ComparatorAScoreCache(
                entry_ns_sha256="b" * 64,
                probabilities=np.array([0.9], dtype="float64"),
                pre_schedule_mask=np.array([True], dtype=bool),
            )
        }

    def evaluate_only_after_all_scores(
        _spec: dict[str, Any], pair_input: StubPairInput, **_kwargs: Any
    ) -> replay.PairComputation:
        equal(
            len(
                [
                    event
                    for event in mutation_events
                    if event.startswith("all_scores:")
                ]
            ),
            len(stats.PAIR_ORDER),
            "pair evaluation preceded late-pair model mutation detection",
        )
        mutation_events.append(f"evaluate:{pair_input.pair}")
        raise replay.ReplayInputError("synthetic pair outcome-input defect")

    with mock.patch.object(
        replay, "_preflight_pair_policy_model_schemas"
    ), mock.patch.object(
        replay,
        "_precompute_pair_policy_scores",
        side_effect=precompute_with_last_pair_mutation,
    ), mock.patch.object(
        replay, "evaluate_pair", side_effect=evaluate_only_after_all_scores
    ), mock.patch.object(
        replay,
        "infer_main_family",
        side_effect=lambda endpoints, **_kwargs: FakeFamilyInference(endpoints),
    ), mock.patch.object(
        replay,
        "infer_control_family",
        side_effect=lambda endpoints, **_kwargs: FakeFamilyInference(endpoints),
    ):
        replay.run_common_replay(spec, preflight_inputs)  # type: ignore[arg-type]
    equal(
        mutation_events[: len(stats.PAIR_ORDER)],
        [f"all_scores:{pair}" for pair in stats.PAIR_ORDER],
        "late-pair mutation was not detected in the global score pass",
    )

    # Shared evaluator/statistics defects are never downgraded to one pair's
    # INVALID placeholder; they hard-abort the common replay.
    with mock.patch.object(
        replay, "_preflight_pair_policy_model_schemas"
    ), mock.patch.object(
        replay, "_precompute_pair_policy_scores", side_effect=precompute_all
    ), mock.patch.object(
        replay,
        "evaluate_pair",
        side_effect=stats.StatisticsInputError("synthetic shared evaluator defect"),
    ), expect_raises(stats.StatisticsInputError, "shared evaluator defect"):
        replay.run_common_replay(spec, preflight_inputs)  # type: ignore[arg-type]


def test_post_replay_comparator_fences() -> None:
    """A post-replay A mutation blocks every status/publication handoff."""

    with tempfile.TemporaryDirectory(prefix="m15-comparator-reauth-") as raw:
        repo = Path(raw) / "repo"
        books_root = repo / "books"
        books_root.mkdir(parents=True)
        index_path = books_root / "INDEX.json"
        manifest_path = books_root / "PAIR.m15.v1.manifest.json"
        book_root = books_root / "PAIR.m15.v1"
        book_root.mkdir()
        strategy_path = book_root / "PAIR_strategy.json"
        model_path = book_root / "PAIR_s0_lgb.txt"
        base_index_entry = {
            "id": "PAIR.m15.v1",
            "manifest": "books/PAIR.m15.v1.manifest.json",
        }
        candidate_entry = {
            "id": "PAIR.m15_refresh_2026q1.candidate-v2",
            "manifest": (
                "books/PAIR.m15_refresh_2026q1.candidate-v2/"
                "PAIR.m15_refresh_2026q1.candidate-v2.manifest.json"
            ),
            "lifecycle_status": "inactive_shadow_candidate",
        }
        base_manifest = {"id": "PAIR.m15.v1", "schema": "book-manifest/v1"}
        base_strategy = {"feature_cols": ["feature_0"]}
        base_model = b"synthetic-lightgbm-model\n"

        def write_index(entries: list[dict[str, Any]]) -> None:
            index_path.write_bytes(
                refresh.canonical_bytes(
                    {"schema": "book-index/v1", "books": entries}
                )
            )

        def restore_assets(*, with_candidate: bool) -> None:
            write_index(
                [base_index_entry, candidate_entry]
                if with_candidate
                else [base_index_entry]
            )
            manifest_path.write_bytes(refresh.canonical_bytes(base_manifest))
            strategy_path.write_bytes(refresh.canonical_bytes(base_strategy))
            model_path.write_bytes(base_model)

        class SyntheticBook:
            pair = "PAIR"
            book_id = "PAIR.m15.v1"
            content_id = "synthetic-v1-content"
            feature_cols = ["feature_0"]
            coverage = 0.05
            conf_thr = 0.2
            strategy = base_strategy
            model_paths = [model_path]

        synthetic_book = SyntheticBook()
        synthetic_book.strategy_path = strategy_path
        synthetic_spec = {
            "pair_order": ["PAIR"],
            "pairs": {
                "PAIR": {
                    "book_id_A": "PAIR.m15.v1",
                    "adapter_family": "own_pair",
                    "builder": "synthetic_builder",
                }
            },
            "identity": {"domains": {"bundle_id": "synthetic/bundle/v1"}},
        }
        prereg = {
            "implementation_git_sha": "1" * 40,
            "prereg_id": "2" * 64,
        }

        def load_index() -> dict[str, dict[str, Any]]:
            decoded = json.loads(index_path.read_bytes())
            return {entry["id"]: entry for entry in decoded["books"]}

        def load_manifest(
            _book_id: str, index_entry: dict[str, Any]
        ) -> tuple[Path, dict[str, Any]]:
            path = repo / index_entry["manifest"]
            return path, json.loads(path.read_bytes())

        restore_assets(with_candidate=False)
        with mock.patch.object(refresh, "REPO_ROOT", repo), mock.patch.object(
            refresh, "load_spec", return_value=synthetic_spec
        ), mock.patch.object(
            book_runtime, "load_index", side_effect=load_index
        ), mock.patch.object(
            book_runtime,
            "load_target_books",
            return_value={"PAIR": synthetic_book},
        ), mock.patch.object(
            book_runtime, "load_registered_manifest", side_effect=load_manifest
        ):
            prereg["comparator_A_files"] = [
                vars(refresh.file_identity(path, root=repo))
                for path in (index_path, manifest_path, strategy_path, model_path)
            ]
            expected_bundle = refresh._comparator_a_bundle_id(
                synthetic_spec,
                prereg,
                "PAIR",
                synthetic_book,
                base_index_entry,
                manifest_path,
            )
            binding = {"primary_arms": [["PAIR", "A", expected_bundle]]}
            refresh._authenticate_current_comparator_a(
                prereg, binding, allow_index_additions=False
            )

            restore_assets(with_candidate=True)
            refresh._authenticate_current_comparator_a(
                prereg, binding, allow_index_additions=True
            )
            with expect_raises(refresh.RefreshError, "exact path/count/order/bytes"):
                refresh._authenticate_current_comparator_a(
                    prereg, binding, allow_index_additions=False
                )

            mutations: list[tuple[str, Callable[[], None]]] = [
                (
                    "model",
                    lambda: model_path.write_bytes(base_model + b"mutation\n"),
                ),
                (
                    "strategy",
                    lambda: strategy_path.write_bytes(
                        refresh.canonical_bytes(
                            {"feature_cols": ["feature_0"], "mutation": True}
                        )
                    ),
                ),
                (
                    "manifest",
                    lambda: manifest_path.write_bytes(
                        refresh.canonical_bytes(
                            {
                                "id": "PAIR.m15.v1",
                                "schema": "book-manifest/v1",
                                "mutation": True,
                            }
                        )
                    ),
                ),
                (
                    "v1 INDEX entry",
                    lambda: write_index(
                        [dict(base_index_entry, mutation=True), candidate_entry]
                    ),
                ),
            ]
            for name, mutate in mutations:
                restore_assets(with_candidate=True)
                mutate()
                with expect_raises(refresh.RefreshError):
                    refresh._authenticate_current_comparator_a(
                        prereg, binding, allow_index_additions=True
                    )
            restore_assets(with_candidate=True)

    class FenceStore:
        def __init__(self, *, replay_sealed: bool = False):
            self.replay_sealed = replay_sealed
            self.load_calls: list[str] = []
            self.seal_calls = 0

        def has(self, phase: str) -> bool:
            return phase == "replay" and self.replay_sealed

        def load(self, phase: str) -> dict[str, Any]:
            self.load_calls.append(phase)
            if phase == "arm_seal":
                return {"output_hashes": {}}
            return {}

        def seal(self, *_args: Any, **_kwargs: Any) -> None:
            self.seal_calls += 1
            raise GateFailure("comparator fence unexpectedly reached phase sealing")

    prereg = {"prereg_id": "3" * 64, "implementation_git_sha": "4" * 40}
    binding = {"run_id": "5" * 64}
    with tempfile.TemporaryDirectory(prefix="m15-comparator-fences-") as raw:
        root = Path(raw)
        cases = (
            ("replay", refresh.run_replay, FenceStore(replay_sealed=True)),
            ("status", refresh.lock_results_and_candidates, FenceStore()),
            ("publication", refresh.publish_candidates, FenceStore()),
            ("shadow", refresh.seal_shadow_spec, FenceStore()),
        )
        for name, entrypoint, store in cases:
            workspace = root / name
            authenticate = mock.Mock(
                side_effect=refresh.RefreshError("synthetic comparator A drift")
            )
            with mock.patch.object(
                refresh,
                "workspace_for",
                return_value=(workspace, store, prereg),
            ), mock.patch.object(
                refresh, "_run_binding", return_value=binding
            ), mock.patch.object(
                refresh, "_authenticate_current_comparator_a", authenticate
            ), mock.patch.object(
                replay, "run_workspace_replay"
            ) as semantic_reader, mock.patch.object(
                refresh, "_seal_json_exact"
            ) as seal_json, mock.patch.object(
                refresh.manifest, "publish_candidate"
            ) as publish, mock.patch.object(
                refresh.manifest, "recover_candidate_publication"
            ) as recover, expect_raises(
                refresh.RefreshError, "comparator A drift"
            ):
                entrypoint(prereg["prereg_id"])
            equal(authenticate.call_count, 1, f"{name} comparator fence count")
            check(
                authenticate.call_args.kwargs["allow_index_additions"] is True,
                f"{name} comparator fence did not permit candidate-only INDEX additions",
            )
            check(not semantic_reader.called, f"{name} reached semantic reader")
            check(not seal_json.called, f"{name} wrote status/shadow JSON")
            check(not publish.called, f"{name} published a candidate")
            check(not recover.called, f"{name} recovered a candidate publication")
            equal(store.seal_calls, 0, f"{name} sealed a phase")
            check(not workspace.exists(), f"{name} created workspace output")

        class ResumeStore(FenceStore):
            def __init__(self, sealed_phase: str):
                super().__init__(replay_sealed=sealed_phase == "replay")
                self.sealed_phase = sealed_phase

            def has(self, phase: str) -> bool:
                return phase == self.sealed_phase

        resume_cases = (
            ("replay", refresh.run_replay, ResumeStore("replay")),
            (
                "status",
                refresh.lock_results_and_candidates,
                ResumeStore("status_and_publication_lock"),
            ),
            (
                "publication",
                refresh.publish_candidates,
                ResumeStore("candidate_publication"),
            ),
        )
        for name, entrypoint, store in resume_cases:
            workspace = root / f"{name}-resume"
            authenticate = mock.Mock(
                side_effect=[
                    None,
                    refresh.RefreshError(f"synthetic {name} resume drift"),
                ]
            )
            with mock.patch.object(
                refresh,
                "workspace_for",
                return_value=(workspace, store, prereg),
            ), mock.patch.object(
                refresh, "_run_binding", return_value=binding
            ), mock.patch.object(
                refresh, "_authenticate_current_comparator_a", authenticate
            ), mock.patch.object(
                refresh, "_read_json", return_value={"S": []}
            ), mock.patch.object(
                refresh, "_seal_json_exact"
            ) as seal_json, mock.patch.object(
                refresh.manifest, "publish_candidate"
            ) as publish, mock.patch.object(
                refresh.manifest, "recover_candidate_publication"
            ) as recover, expect_raises(
                refresh.RefreshError, f"{name} resume drift"
            ):
                entrypoint(prereg["prereg_id"])
            equal(authenticate.call_count, 2, f"{name} resume fence count")
            check(not seal_json.called, f"{name} resume wrote JSON")
            check(not publish.called, f"{name} resume published")
            check(not recover.called, f"{name} resume recovered")
            equal(store.seal_calls, 0, f"{name} resume sealed phase")

        # The runner passes the same authenticated closure into both the fresh
        # publish and partial-recovery transactions, where it can reject from
        # inside the repository lock before registration commits.
        for recovering in (False, True):
            case_root = root / ("recover-guard" if recovering else "publish-guard")
            (case_root / "books").mkdir(parents=True)
            workspace = case_root / "workspace"
            store = FenceStore()
            candidate_id = "PAIR.m15.candidate-v2"
            destination = case_root / "books" / candidate_id
            if recovering:
                destination.mkdir()
            lock = {
                "bundle_id": "8" * 64,
                "prereg_id": prereg["prereg_id"],
                "run_id": binding["run_id"],
                "payload": {"artifact.txt": "9" * 64},
            }
            joint = {
                "S": [candidate_id],
                "candidate_publish_locks": {candidate_id: lock},
            }
            authenticate = mock.Mock(
                side_effect=[
                    None,
                    None,
                    refresh.RefreshError("synthetic in-transaction drift"),
                ]
            )
            observed_guard: list[Callable[[], None]] = []

            def invoke_commit_guard(
                *_args: Any, **kwargs: Any
            ) -> Path:
                guard = kwargs.get("commit_guard")
                check(callable(guard), "runner omitted publication commit guard")
                observed_guard.append(guard)
                guard()
                raise GateFailure("rejected commit guard unexpectedly returned")

            with mock.patch.object(
                refresh,
                "workspace_for",
                return_value=(workspace, store, prereg),
            ), mock.patch.object(
                refresh, "_run_binding", return_value=binding
            ), mock.patch.object(
                refresh, "_authenticate_current_comparator_a", authenticate
            ), mock.patch.object(
                refresh, "_read_json", return_value=joint
            ), mock.patch.object(
                refresh, "REPO_ROOT", case_root
            ), mock.patch.object(
                refresh, "RESULTS_JSON", case_root / "results"
            ), mock.patch.object(
                refresh, "_partial_stage_for", return_value=[case_root / "stage"]
            ), mock.patch.object(
                refresh.manifest,
                "publish_candidate",
                side_effect=invoke_commit_guard,
            ) as publish, mock.patch.object(
                refresh.manifest,
                "recover_candidate_publication",
                side_effect=invoke_commit_guard,
            ) as recover, expect_raises(
                refresh.RefreshError, "in-transaction drift"
            ):
                refresh.publish_candidates(prereg["prereg_id"])
            equal(len(observed_guard), 1, "publication guard wiring count")
            equal(authenticate.call_count, 3, "in-transaction auth count")
            equal(publish.call_count, int(not recovering), "fresh publish branch")
            equal(recover.call_count, int(recovering), "recovery branch")
            equal(store.seal_calls, 0, "rejected publication sealed phase")

    class TransactionStore:
        def __init__(self, workspace: Path):
            self.workspace = workspace

        def has(self, _phase: str) -> bool:
            return False

        def load(self, _phase: str) -> dict[str, Any]:
            return {}

        def path(self, phase: str) -> Path:
            return self.workspace / "phases" / f"{phase}.json"

        def seal(self, phase: str, **_kwargs: Any) -> dict[str, Any]:
            value = {"schema": "synthetic-phase/v1", "phase": phase}
            refresh.atomic_json_new(self.path(phase), value)
            return value

    # The enclosing publication phase seal is invocation-owned and is removed
    # if the final post-transaction Comparator-A check rejects.
    with tempfile.TemporaryDirectory(prefix="m15-publication-phase-rollback-") as raw:
        root = Path(raw)
        workspace = root / "workspace"
        store = TransactionStore(workspace)
        authenticate = mock.Mock(
            side_effect=[
                None,
                refresh.RefreshError("synthetic late publication drift"),
            ]
        )
        with mock.patch.object(
            refresh,
            "workspace_for",
            return_value=(workspace, store, prereg),
        ), mock.patch.object(
            refresh, "_run_binding", return_value=binding
        ), mock.patch.object(
            refresh, "_authenticate_current_comparator_a", authenticate
        ), mock.patch.object(
            refresh, "_read_json", return_value={"S": []}
        ), mock.patch.object(
            refresh, "RESULTS_JSON", root / "results"
        ), expect_raises(refresh.RefreshError, "late publication drift"):
            refresh.publish_candidates(prereg["prereg_id"])
        equal(authenticate.call_count, 2, "publication final commit guard count")
        check(
            not store.path("candidate_publication").exists(),
            "publication rollback left enclosing phase seal",
        )

    # A mutation discovered at the final status checkpoint rolls back every
    # status/result/phase file created by that commit attempt.
    with tempfile.TemporaryDirectory(prefix="m15-status-guard-rollback-") as raw:
        root = Path(raw)
        workspace = root / "workspace"
        results_root = root / "results"
        store = TransactionStore(workspace)
        spec = refresh.load_spec()

        class Campaign:
            joint_result = {"S": [], "candidate_statuses": {}}
            pair_results = {
                pair: {"pair": pair, "status": "synthetic"}
                for pair in spec["pair_order"]
            }

        final = Campaign()
        authenticate = mock.Mock(
            side_effect=[
                None,
                None,
                None,
                refresh.RefreshError("synthetic late status drift"),
            ]
        )
        with mock.patch.object(
            refresh,
            "workspace_for",
            return_value=(workspace, store, prereg),
        ), mock.patch.object(
            refresh, "_run_binding", return_value=binding
        ), mock.patch.object(
            refresh, "RESULTS_JSON", results_root
        ), mock.patch.object(
            refresh, "_load_replay_campaign", return_value=Campaign()
        ), mock.patch.object(
            replay, "finalize_joint_result", return_value=final
        ), mock.patch.object(
            refresh, "_authenticate_current_comparator_a", authenticate
        ), mock.patch.object(
            refresh,
            "repo_relative",
            side_effect=lambda path, **_kwargs: Path(path).name,
        ), expect_raises(refresh.RefreshError, "late status drift"):
            refresh.lock_results_and_candidates(prereg["prereg_id"])
        equal(authenticate.call_count, 4, "status final commit guard count")
        check(not (workspace / "status").exists(), "status rollback left workspace output")
        check(
            not store.path("status_and_publication_lock").exists(),
            "status rollback left phase seal",
        )
        check(
            not results_root.exists() or not any(results_root.iterdir()),
            "status rollback left tracked output",
        )

    # The same late-check rollback covers the zero-survivor shadow handoff,
    # including its tracked result and phase seal.
    with tempfile.TemporaryDirectory(prefix="m15-shadow-guard-rollback-") as raw:
        root = Path(raw)
        workspace = root / "workspace"
        results_root = root / "results"
        results_root.mkdir()
        store = TransactionStore(workspace)
        shadow_binding = {
            "run_id": binding["run_id"],
            "evaluator_id": "6" * 64,
            "controls_id": "7" * 64,
            "primary_arms": [],
            "control_arms": [],
        }
        parity_path = workspace / "parity" / "parity.json"
        parity_path.parent.mkdir(parents=True)
        refresh.atomic_json_new(
            parity_path,
            {
                "schema": "m15-book-refresh-parity/v1",
                "prereg_id": prereg["prereg_id"],
                "implementation_git_sha": prereg["implementation_git_sha"],
                "pairs": {},
            },
        )
        binding_path = (
            workspace
            / "arm_seal"
            / f"run_binding_{shadow_binding['run_id']}.json"
        )
        refresh.atomic_json_new(binding_path, shadow_binding)
        tracked_joint = (
            results_root
            / f"m15_book_refresh_{shadow_binding['run_id']}_joint_replay_result.json"
        )
        refresh.atomic_json_new(tracked_joint, {"S": []})
        authenticate = mock.Mock(
            side_effect=[
                None,
                None,
                None,
                refresh.RefreshError("synthetic late shadow drift"),
            ]
        )
        with mock.patch.object(
            refresh,
            "workspace_for",
            return_value=(workspace, store, prereg),
        ), mock.patch.object(
            refresh, "_run_binding", return_value=shadow_binding
        ), mock.patch.object(
            refresh, "RESULTS_JSON", results_root
        ), mock.patch.object(
            refresh, "_authenticate_current_comparator_a", authenticate
        ), mock.patch.object(
            refresh, "_validate_shadow_handoff_payload"
        ), mock.patch.object(
            refresh,
            "repo_relative",
            side_effect=lambda path, **_kwargs: Path(path).name,
        ), expect_raises(refresh.RefreshError, "late shadow drift"):
            refresh.seal_shadow_spec(prereg["prereg_id"])
        equal(authenticate.call_count, 4, "shadow final commit guard count")
        check(not (workspace / "shadow").exists(), "shadow rollback left workspace output")
        check(
            not store.path("shadow_spec").exists(),
            "shadow rollback left phase seal",
        )
        tracked_shadow = (
            results_root
            / f"m15_book_refresh_{shadow_binding['run_id']}_shadow_spec_result.json"
        )
        check(not tracked_shadow.exists(), "shadow rollback left tracked output")


def test_live_feature_builder_provider_parity() -> None:
    import pipeline

    pair = "USDJPY"
    columns = tuple(harness_column for harness_column in adapters.harness.feature_cols(pair))
    count = 16_000
    index = pd.date_range("2025-01-01T00:00:00Z", periods=count, freq="min")
    rng = np.random.default_rng(20260713)
    close = 1.0 + np.cumsum(rng.normal(0.0, 1e-4, count))
    m1 = pd.DataFrame(
        {
            "open": close,
            "high": close + 2e-5,
            "low": close - 2e-5,
            "close": close,
            "volume": np.ones(count, dtype="float64"),
            "gap_prev": np.ones(count, dtype="float64"),
        },
        index=index,
    )
    cached = pipeline.build_features(m1).replace([np.inf, -np.inf], np.nan)
    source_ns = cached.index.as_unit("ns").asi8.astype("int64")
    decision_ns = source_ns + 60 * adapters.NS_PER_SECOND
    score_rows = adapters.ScoreRows(
        pair=pair,
        feature_cols=columns,
        X=np.ascontiguousarray(
            cached.loc[:, list(columns)].to_numpy(dtype="float32", copy=True)
        ),
        entry_ns=decision_ns,
        source_feature_ns=source_ns,
    )
    with tempfile.TemporaryDirectory(prefix="m15-live-provider-test-") as raw:
        store = Path(raw)
        m1.to_parquet(store / f"{pair}.parquet")
        builder = live_features.LiveFeatureBuilder(
            client=None,
            history_minutes=count + 1,
            tick_volume_count=0,
            store_dir=store,
            stale_seconds=10**9,
        )
        report = refresh._live_feature_builder_parity_report(
            pair,
            columns,
            score_rows,
            builder,
        )
        provider_frame = builder._joined_row_frame(pair, list(columns))
        duplicated = pd.concat([provider_frame.iloc[:2], provider_frame.iloc[1:]])
        with mock.patch.object(builder, "_joined_row_frame", return_value=duplicated), \
                expect_raises(refresh.RefreshError, "strictly increasing and unique"):
            refresh._live_feature_builder_parity_report(
                pair, columns, score_rows, builder
            )
        descending = provider_frame.iloc[::-1]
        with mock.patch.object(builder, "_joined_row_frame", return_value=descending), \
                expect_raises(refresh.RefreshError, "strictly increasing and unique"):
            refresh._live_feature_builder_parity_report(
                pair, columns, score_rows, builder
            )

        nan_position = tuple(np.argwhere(np.isnan(score_rows.X))[0])
        payload_matrix = score_rows.X.copy()
        payload_matrix.view("uint32")[nan_position] ^= np.uint32(1 << 31)
        payload_report = refresh._live_feature_builder_parity_report(
            pair,
            columns,
            replace(score_rows, X=np.ascontiguousarray(payload_matrix)),
            builder,
        )

        finite_position = tuple(np.argwhere(np.isfinite(score_rows.X))[0])
        finite_matrix = score_rows.X.copy()
        finite_matrix.view("uint32")[finite_position] += np.uint32(1)
        finite_report = refresh._live_feature_builder_parity_report(
            pair,
            columns,
            replace(score_rows, X=np.ascontiguousarray(finite_matrix)),
            builder,
        )

        mask_matrix = score_rows.X.copy()
        mask_matrix[finite_position] = np.float32(np.nan)
        mask_report = refresh._live_feature_builder_parity_report(
            pair,
            columns,
            replace(score_rows, X=np.ascontiguousarray(mask_matrix)),
            builder,
        )

        infinity_matrix = score_rows.X.copy()
        infinity_matrix[finite_position] = np.float32(np.inf)
        with expect_raises(refresh.RefreshError, "contains an infinity"):
            refresh._live_feature_builder_parity_report(
                pair,
                columns,
                replace(score_rows, X=np.ascontiguousarray(infinity_matrix)),
                builder,
            )
    check(report["all_equal"], f"actual LiveFeatureBuilder parity failed: {report}")
    equal(
        report["public_method_exercised"],
        "feature_row",
        "public LiveFeatureBuilder method evidence",
    )
    check(
        report["ordered_float32_feature_bytes_exact"]
        and report["causal_decision_mapping_exact"],
        "public provider values/decision mapping were not exact",
    )
    check(
        payload_report["ordered_float32_feature_bytes_exact"] is False
        and payload_report["all_equal"] is False
        and payload_report["ordered_finite_float32_values_exact"] is True
        and payload_report["finite_mask_exact"] is True
        and payload_report["nonfinite_nan_payload_bits_binding"] is False
        and payload_report["behavior_all_equal"] is True,
        "NaN payload-only drift did not preserve exact provider behavior",
    )
    check(
        finite_report["ordered_finite_float32_values_exact"] is False
        and finite_report["decoded_diagnostics"]["mismatched_cells"] == 1
        and finite_report["behavior_all_equal"] is False,
        "finite one-ULP provider drift did not fail behavior parity",
    )
    check(
        mask_report["finite_mask_exact"] is False
        and mask_report["ordered_finite_float32_values_exact"] is False
        and mask_report["decoded_diagnostics"][
            "finite_nonfinite_disagreements"
        ] == 1
        and mask_report["behavior_all_equal"] is False,
        "finite-to-NaN provider drift did not fail behavior parity",
    )


def _endpoint_family(size: int, calendar: np.ndarray, prefix: str) -> list[stats.DateRatioEndpoint]:
    phase = np.arange(len(calendar), dtype="float64")
    endpoints = []
    for index in range(size):
        numerator = np.sin(phase * 0.31 + index * 0.17) + (index % 3 - 1) * 0.03
        denominator = np.full(len(calendar), 20.0 + index)
        endpoints.append(
            stats.DateRatioEndpoint(
                f"{prefix}:{index:02d}", calendar, numerator, denominator
            )
        )
    return endpoints


def _status_inputs() -> dict[str, Any]:
    return {
        "statistics_computable": True,
        "control_estimable": True,
        "side_trade_counts": {
            arm: {"UP": 60, "DOWN": 60} for arm in ("A", "B", "C")
        },
        "combined_trade_counts": {"A": 120, "B": 120, "C": 120},
        "promotion_lower_bounds": {
            "C_minus_B_yield": 0.01,
            "C_minus_A_yield": 0.01,
            "C_UP_accuracy_minus_0_5": 0.01,
            "C_DOWN_accuracy_minus_0_5": 0.01,
        },
        "retention_upper_bounds": {
            "C_minus_A_yield": 0.02,
            "C_UP_accuracy_minus_0_5": 0.02,
            "C_DOWN_accuracy_minus_0_5": 0.02,
        },
        "monthly_c_side_counts": {
            month: {"UP": 1, "DOWN": 1} for month in ("April", "May")
        },
        "monthly_c_whole_book_yield": {"April": 0.01, "May": 0.01},
        "whole_book_pass": True,
    }


def test_statistics_and_statuses() -> None:
    calendar = np.arange(20250101, 20250131, dtype="int64")
    main_endpoints = _endpoint_family(stats.MAIN_FAMILY_SIZE, calendar, "main")
    main = stats.infer_main_family(
        main_endpoints,
        master_calendar=calendar,
        bootstrap_seed=41,
        block_lengths=(1, 2, 3),
        bootstrap_replicates=64,
    )
    main_repeat = stats.infer_main_family(
        main_endpoints,
        master_calendar=calendar,
        bootstrap_seed=41,
        block_lengths=(1, 2, 3),
        bootstrap_replicates=64,
    )
    equal(main.as_dict(), main_repeat.as_dict(), "main inference determinism")
    equal(main.family_size, stats.MAIN_FAMILY_SIZE, "main reserved family size")
    equal(main.tail, "two_sided", "main tail")
    check(all(row.computable for row in main.endpoints), "dense main endpoint noncomputable")
    for row in main.endpoints:
        equal(row.lower, min(cell.lower for cell in row.by_length), "lower envelope")
        equal(row.upper, max(cell.upper for cell in row.by_length), "upper envelope")

    control = stats.infer_control_family(
        _endpoint_family(9, calendar, "control"),
        master_calendar=calendar,
        bootstrap_seed=42,
        block_lengths=(1, 2, 3),
        bootstrap_replicates=64,
    )
    equal(control.family_size, 9, "control reserved family size")
    equal(control.tail, "lower_only", "control tail")
    check(all(row.upper is None for row in control.endpoints), "lower-only family emitted upper")

    shadow = stats.infer_shadow_family(
        _endpoint_family(6, calendar, "shadow"),
        master_calendar=calendar,
        k_shadow=2,
        bootstrap_seed=43,
        block_lengths=(1, 2, 3),
        bootstrap_replicates=64,
    )
    equal(shadow.family_size, 6, "shadow 3k family size")
    with expect_raises(stats.StatisticsInputError, "received"):
        stats.infer_shadow_family(
            _endpoint_family(3, calendar, "bad-shadow"),
            master_calendar=calendar,
            k_shadow=2,
            bootstrap_replicates=8,
        )

    sparse = _endpoint_family(stats.MAIN_FAMILY_SIZE, calendar, "sparse")
    sparse[0] = stats.DateRatioEndpoint(
        "sparse:00", calendar, np.zeros(len(calendar)), np.zeros(len(calendar))
    )
    sparse_result = stats.infer_main_family(
        sparse,
        master_calendar=calendar,
        block_lengths=(1, 2, 3),
        bootstrap_replicates=32,
    )
    check(not sparse_result.endpoints[0].computable, "zero-denominator slot became computable")
    check(
        "zero_or_undefined_observed_denominator"
        in sparse_result.endpoints[0].noncomputable_reasons,
        "sparsity reason missing",
    )
    zero_se = _endpoint_family(stats.MAIN_FAMILY_SIZE, calendar, "zero-se")
    zero_se[0] = stats.DateRatioEndpoint(
        "zero-se:00", calendar, np.ones(len(calendar)), np.ones(len(calendar))
    )
    zero_result = stats.infer_main_family(
        zero_se,
        master_calendar=calendar,
        block_lengths=(1, 2, 3),
        bootstrap_replicates=32,
    )
    check(not zero_result.endpoints[0].computable, "genuine zero SE was treated as computable")
    check(
        any("zero_or_nonfinite_standard_error" in reason
            for reason in zero_result.endpoints[0].noncomputable_reasons),
        "zero-SE reason missing",
    )

    promote_inputs = _status_inputs()
    equal(
        stats.assign_terminal_status(**promote_inputs).status,
        "PROMOTE_TO_SHADOW",
        "promotion status",
    )
    invalid_inputs = dict(promote_inputs, invalid_reasons=["clock_defect"], statistics_computable=False)
    equal(stats.assign_terminal_status(**invalid_inputs).status, "INVALID", "invalid precedence")
    noncomputable = dict(promote_inputs, statistics_computable=False)
    equal(
        stats.assign_terminal_status(**noncomputable).status,
        "INCONCLUSIVE",
        "noncomputable precedence",
    )
    retain = json.loads(json.dumps(promote_inputs))
    retain["promotion_lower_bounds"]["C_minus_A_yield"] = -0.01
    retain["retention_upper_bounds"]["C_minus_A_yield"] = -0.001
    equal(stats.assign_terminal_status(**retain).status, "RETAIN_V1", "retention status")
    remaining = json.loads(json.dumps(promote_inputs))
    remaining["promotion_lower_bounds"]["C_minus_A_yield"] = -0.01
    equal(
        stats.assign_terminal_status(**remaining).status,
        "INCONCLUSIVE",
        "remaining status",
    )

    no_candidates = stats.shadow_empty_set_status([], [])
    equal(no_candidates["status"], "no_candidates", "empty survivor shadow status")
    no_eligible = stats.shadow_empty_set_status(
        ["A.v2", "B.v2"], [], exclusions={"A.v2": "provider", "B.v2": "schema"}
    )
    equal(
        no_eligible["status"],
        "no_shadow_eligible_candidates",
        "nonempty survivor zero-eligible shadow status",
    )
    check("T0" not in no_eligible, "zero-eligible shadow status invented T0")

    no_repeat = stats.run_null_calibration_cell(
        "F0",
        helper="main",
        campaign_count=1,
        bootstrap_replicates=8,
        block_lengths=(1, 2, 3),
        verify_repeat=False,
    )
    check(
        no_repeat.deterministic_repeat is False and no_repeat.accepted is False,
        "skipping the calibration repeat falsely asserted determinism/acceptance",
    )


def _candidate_fixture(
    temporary: Path,
) -> tuple[Path, dict[str, Any], dict[str, Any], list[Path], dict[str, Any]]:
    repo = temporary / "repo"
    books = repo / "books"
    books.mkdir(parents=True)
    (books / "INDEX.json").write_text('{"schema":"book-index/v1","books":[]}\n')
    (repo / "results" / "json").mkdir(parents=True)
    (repo / "logs" / "m15_book_refresh" / ("b" * 64) / "candidate_publication").mkdir(
        parents=True
    )
    (repo / "MODEL_REGISTRY.md").write_text("# Synthetic registry\n")
    artifacts_dir = temporary / "artifacts"
    artifacts_dir.mkdir()
    strategy = artifacts_dir / "TEST_strategy.json"
    model = artifacts_dir / "TEST_s0_lgb.txt"
    strategy.write_bytes(b'{"pair":"TEST","conf_thr":0.125}')
    model.write_bytes(b"tree\n")
    manifest_value = manifest.build_candidate_manifest(
        "TEST.m15.v2",
        repo_root=repo,
        currency="TEST",
        timeframe="15m",
        side="combined",
        role="direction",
        lifecycle_status="inactive_shadow_candidate",
        bundle_id="a" * 64,
        prereg_id="b" * 64,
        run_id="c" * 64,
        implementation_git_sha="d" * 40,
        source_script="scripts/m15_book_refresh.py",
        determinism={
            "seeds": [0],
            "num_threads": 1,
            "deterministic": True,
            "force_col_wise": True,
        },
        artifacts=[strategy, model],
        strategy_json=strategy.name,
        summary="synthetic golden candidate",
        metrics={"coverage": 0.02},
        created_utc="2026-01-01T00:00:00Z",
    )
    resolved = {
        "schema": "m15-book-refresh-resolved-bundle-spec/v1",
        "book_id": "TEST.m15.v2",
        "bundle_id": "a" * 64,
        "prereg_id": "b" * 64,
        "lifecycle_status": "inactive_shadow_candidate",
        "coverage": 0.02,
        "threshold": 0.125,
        "feature_cols": ["x"],
        "feature_dtype": "float32",
        "seed_order": [0],
        "probability_tie_rule": "p>=0.5_is_UP",
        "implementation_git_sha": "d" * 40,
    }
    return repo, manifest_value, resolved, [strategy, model], {
        "model": model,
        "strategy": strategy,
    }


def _write_joint_result(
    repo: Path,
    lock: dict[str, Any],
    *,
    status: str = "PROMOTE_TO_SHADOW",
) -> Path:
    joint = {
        "schema": "m15-book-refresh-joint-replay-result/v1",
        "run_id": "c" * 64,
        "prereg_id": "b" * 64,
        "S": ["TEST.m15.v2"],
        "statuses": {"TEST": {"status": status}},
        "candidate_statuses": {
            "TEST.m15.v2": {"pair": "TEST", "status": status}
        },
        "candidate_publish_locks": {"TEST.m15.v2": lock},
    }
    path = (
        repo / "results" / "json"
        / f"m15_book_refresh_{'c' * 64}_joint_replay_result.json"
    )
    path.write_bytes(manifest.canonical_json_bytes(joint))
    path.chmod(0o444)
    return path


def _forge_self_consistent_stage(stage: Path, model_name: str) -> dict[str, Any]:
    stage.chmod(0o755)
    for path in stage.iterdir():
        path.chmod(0o644)
    model_path = stage / model_name
    model_path.write_bytes(b"forged-tree\n")
    manifest_path = next(stage.glob("*.manifest.json"))
    manifest_value = json.loads(manifest_path.read_bytes())
    for row in manifest_value["artifacts"]:
        if row["file"] == model_name:
            row["bytes"] = model_path.stat().st_size
            row["sha256"] = sha256(model_path)
    manifest_path.write_bytes(manifest.canonical_json_bytes(manifest_value))
    receipt_path = next(stage.glob("*.publish_receipt.json"))
    receipt = json.loads(receipt_path.read_bytes())
    receipt["payload"] = {
        path.name: sha256(path)
        for path in sorted(stage.iterdir(), key=lambda row: row.name)
        if path != receipt_path
    }
    receipt_path.write_bytes(manifest.canonical_json_bytes(receipt))
    for path in stage.iterdir():
        path.chmod(0o444)
    stage.chmod(0o555)
    return manifest.candidate_publish_lock(stage)


def _publication_registration_material() -> tuple[str, dict[str, Any], str]:
    registry_entry = (
        f"TEST.m15.v2 bundle={'a' * 64} prereg={'b' * 64} run={'c' * 64} "
        "lifecycle_status=inactive_shadow_candidate inactive no activation"
    )
    phase_seal = {
        "schema": "m15-book-refresh-candidate-publication/v1",
        "status": "sealed",
        "lifecycle_status": "inactive_shadow_candidate",
        "book_id": "TEST.m15.v2",
        "bundle_id": "a" * 64,
        "prereg_id": "b" * 64,
        "run_id": "c" * 64,
    }
    phase_rel = (
        f"logs/m15_book_refresh/{'b' * 64}/candidate_publication/"
        "TEST.m15.v2.json"
    )
    return registry_entry, phase_seal, phase_rel


def test_lifecycle_and_publication() -> None:
    expected_legacy_ids = frozenset(
        {
            "EURUSD.m15xp.v1",
            "USDCHF.m15ny_xpair_seedens.v1",
            "GBPUSD.m15ny_xpair_seedens8.v1",
            "USDJPY.m15ny_seedens.v1",
            "USDCAD.m15ny_seedens.v1",
            "AUDUSD.m15ny_seedens.v1",
            "NZDUSD.m15ny_seedens.v1",
        }
    )
    equal(
        book_runtime.LEGACY_ACTIVE_BOOK_IDS,
        expected_legacy_ids,
        "literal seven-book legacy lifecycle whitelist",
    )
    legacy_id = next(iter(sorted(expected_legacy_ids)))
    legacy_index = {"id": legacy_id}
    legacy_manifest = {"id": legacy_id, "schema": "book-manifest/v1"}
    equal(
        book_runtime.require_active_book_lifecycle(
            legacy_id, legacy_index, legacy_manifest
        ),
        "legacy-active",
        "legacy lifecycle whitelist",
    )
    with expect_raises(book_runtime.BookRuntimeError, "non-whitelisted"):
        book_runtime.require_active_book_lifecycle(
            "UNKNOWN.m15.v1",
            {"id": "UNKNOWN.m15.v1"},
            {"id": "UNKNOWN.m15.v1", "schema": "book-manifest/v1"},
        )
    active_index = {
        "id": "ACTIVE.m15.v2",
        "lifecycle_status": "active",
    }
    active_manifest = dict(active_index, schema="book-manifest/v1")
    equal(
        book_runtime.require_active_book_lifecycle(
            "ACTIVE.m15.v2", active_index, active_manifest
        ),
        "active",
        "active lifecycle",
    )
    inactive_index = {
        "id": "ACTIVE.m15.v2",
        "lifecycle_status": "inactive_shadow_candidate",
        "bundle_id": "e" * 64,
        "prereg_id": "a" * 64,
        "run_id": "b" * 64,
    }
    inactive_manifest = dict(inactive_index, schema="book-manifest/candidate-v1")
    activated_candidate_index = dict(inactive_index, lifecycle_status="active")
    activated_candidate_manifest = dict(
        inactive_manifest, lifecycle_status="active"
    )
    equal(
        book_runtime.require_active_book_lifecycle(
            "ACTIVE.m15.v2",
            activated_candidate_index,
            activated_candidate_manifest,
        ),
        "active",
        "immutable candidate activated in place",
    )
    lifecycle_rejections = (
        (
            "explicit null",
            dict(active_index, lifecycle_status=None),
            dict(active_manifest, lifecycle_status=None),
        ),
        ("inactive candidate", inactive_index, inactive_manifest),
        (
            "candidate bundle disagreement",
            inactive_index,
            dict(inactive_manifest, bundle_id="f" * 64),
        ),
        (
            "candidate prereg disagreement",
            inactive_index,
            dict(inactive_manifest, prereg_id="f" * 64),
        ),
        (
            "candidate run disagreement",
            inactive_index,
            dict(inactive_manifest, run_id="f" * 64),
        ),
        (
            "candidate one-sided bundle",
            inactive_index,
            {key: value for key, value in inactive_manifest.items() if key != "bundle_id"},
        ),
        (
            "candidate malformed bundle",
            dict(inactive_index, bundle_id="not-a-digest"),
            dict(inactive_manifest, bundle_id="not-a-digest"),
        ),
        (
            "one-sided lifecycle",
            {"id": "ACTIVE.m15.v2"},
            active_manifest,
        ),
        (
            "lifecycle disagreement",
            active_index,
            inactive_manifest,
        ),
        (
            "unknown lifecycle",
            dict(active_index, lifecycle_status="retired"),
            dict(active_manifest, lifecycle_status="retired"),
        ),
        (
            "active candidate missing identity",
            active_index,
            dict(active_manifest, schema="book-manifest/candidate-v1"),
        ),
        (
            "active candidate bundle disagreement",
            activated_candidate_index,
            dict(activated_candidate_manifest, bundle_id="f" * 64),
        ),
        (
            "active candidate prereg disagreement",
            activated_candidate_index,
            dict(activated_candidate_manifest, prereg_id="f" * 64),
        ),
        (
            "active candidate malformed run",
            dict(activated_candidate_index, run_id="not-a-digest"),
            dict(activated_candidate_manifest, run_id="not-a-digest"),
        ),
        (
            "malformed inactive schema",
            inactive_index,
            dict(inactive_manifest, schema="book-manifest/v1"),
        ),
        (
            "explicit active carrying bundle metadata",
            dict(active_index, bundle_id="e" * 64),
            dict(active_manifest, bundle_id="e" * 64),
        ),
    )
    for case, malformed_index, malformed_manifest in lifecycle_rejections:
        with expect_raises(book_runtime.BookRuntimeError):
            book_runtime.require_active_book_lifecycle(
                "ACTIVE.m15.v2", malformed_index, malformed_manifest
            )
    for case, malformed_index, malformed_manifest in (
        (
            "legacy bundle contamination",
            dict(legacy_index, bundle_id="e" * 64),
            legacy_manifest,
        ),
        (
            "malformed legacy schema",
            legacy_index,
            dict(legacy_manifest, schema="book-manifest/candidate-v1"),
        ),
    ):
        with expect_raises(book_runtime.BookRuntimeError):
            book_runtime.require_active_book_lifecycle(
                legacy_id, malformed_index, malformed_manifest
            )
    for identity_field in ("prereg_id", "run_id"):
        with expect_raises(book_runtime.BookRuntimeError, "candidate identity fields"):
            book_runtime.require_active_book_lifecycle(
                "ACTIVE.m15.v2",
                dict(active_index, **{identity_field: "e" * 64}),
                active_manifest,
            )
        with expect_raises(book_runtime.BookRuntimeError, "candidate identity fields"):
            book_runtime.require_active_book_lifecycle(
                legacy_id,
                legacy_index,
                dict(legacy_manifest, **{identity_field: "e" * 64}),
            )

    with tempfile.TemporaryDirectory(prefix="m15-publication-test-") as raw:
        temporary = Path(raw)
        repo, candidate_manifest, resolved, artifacts, named = _candidate_fixture(temporary)
        staged = manifest.stage_candidate(candidate_manifest, resolved, artifacts, repo_root=repo)
        stage = Path(staged["stage_dir"])
        lock = staged["publish_lock"]
        receipt = Path(staged["receipt_path"])
        equal(sha256(receipt), GOLDEN_RECEIPT_SHA256, "receipt golden")
        equal(
            hashlib.sha256(manifest.canonical_json_bytes(lock)).hexdigest(),
            GOLDEN_LOCK_SHA256,
            "external lock golden",
        )
        receipt_value = json.loads(receipt.read_bytes())
        check(receipt.name not in receipt_value["payload"], "receipt listed itself")
        equal(set(receipt_value["payload"]), {
            "TEST.m15.v2.manifest.json",
            "TEST.m15.v2.resolved_bundle_spec.json",
            "TEST_s0_lgb.txt",
            "TEST_strategy.json",
        }, "complete non-receipt payload")

        forged = manifest.stage_candidate(candidate_manifest, resolved, artifacts, repo_root=repo)
        forged_stage = Path(forged["stage_dir"])
        forged_lock = _forge_self_consistent_stage(forged_stage, named["model"].name)
        check(forged_lock != lock, "self-consistent forgery did not alter external lock")
        joint_path = _write_joint_result(repo, lock)
        registry_entry, phase_seal, phase_rel = _publication_registration_material()
        index_path = repo / "books" / "INDEX.json"
        registry_path = repo / "MODEL_REGISTRY.md"
        index_mode_before = index_path.stat().st_mode & 0o7777
        registry_mode_before = registry_path.stat().st_mode & 0o7777
        with expect_raises(manifest.CandidatePublicationError, "sealed external joint-result"):
            manifest.publish_candidate(
                stage,
                lock,
                repo_root=repo,
                registry_path="MODEL_REGISTRY.md",
                registry_entry=registry_entry,
                phase_seal_path=phase_rel,
                phase_seal=phase_seal,
            )
        with expect_raises(manifest.CandidatePublicationError, "external publication lock"):
            manifest.publish_candidate(
                forged_stage,
                joint_path,
                repo_root=repo,
                registry_path="MODEL_REGISTRY.md",
                registry_entry=registry_entry,
                phase_seal_path=phase_rel,
                phase_seal=phase_seal,
            )
        check(not (repo / "books" / "TEST.m15.v2").exists(), "failed pre-rename publish created destination")

        def crash_after_rename(_: Path) -> None:
            raise InjectedCrash("post-rename/pre-registration")

        with expect_raises(InjectedCrash, "post-rename"):
            manifest.publish_candidate(
                stage,
                joint_path,
                repo_root=repo,
                registry_path="MODEL_REGISTRY.md",
                registry_entry=registry_entry,
                phase_seal_path=phase_rel,
                phase_seal=phase_seal,
                after_rename=crash_after_rename,
            )
        destination = repo / "books" / "TEST.m15.v2"
        check(destination.is_dir(), "post-rename crash lost published directory")
        index_before = json.loads((repo / "books" / "INDEX.json").read_bytes())
        equal(index_before["books"], [], "post-rename crash registered INDEX early")

        recovered = manifest.recover_candidate_publication(
            "TEST.m15.v2",
            joint_path,
            repo_root=repo,
            registry_path="MODEL_REGISTRY.md",
            registry_entry=registry_entry,
            phase_seal_path=phase_rel,
            phase_seal=phase_seal,
        )
        equal(recovered, destination, "recovery destination")
        phase_path = repo / phase_rel
        index_bytes = index_path.read_bytes()
        registry_bytes = registry_path.read_bytes()
        phase_bytes = phase_path.read_bytes()
        equal(len(json.loads(index_bytes)["books"]), 1, "recovery INDEX registration")
        equal(
            registry_bytes.splitlines().count(registry_entry.encode()),
            1,
            "recovery exact whole-line registry registration",
        )
        equal(
            index_path.stat().st_mode & 0o7777,
            index_mode_before,
            "INDEX replacement preserved file mode",
        )
        equal(
            registry_path.stat().st_mode & 0o7777,
            registry_mode_before,
            "MODEL_REGISTRY replacement preserved file mode",
        )
        equal(json.loads(phase_bytes), phase_seal, "recovery phase seal")
        check(not phase_path.stat().st_mode & 0o222, "recovery phase seal remained writable")

        # Exact recovery is idempotent and changes no registration bytes.
        manifest.recover_candidate_publication(
            "TEST.m15.v2",
            joint_path,
            repo_root=repo,
            registry_path="MODEL_REGISTRY.md",
            registry_entry=registry_entry,
            phase_seal_path=phase_rel,
            phase_seal=phase_seal,
        )
        equal(index_path.read_bytes(), index_bytes, "idempotent INDEX recovery")
        equal(registry_path.read_bytes(), registry_bytes, "idempotent registry recovery")
        equal(phase_path.read_bytes(), phase_bytes, "idempotent phase recovery")

        forged_joint = temporary / "forged_joint_result.json"
        forged_joint_value = json.loads(joint_path.read_bytes())
        forged_joint_value["candidate_publish_locks"]["TEST.m15.v2"][
            "receipt_sha256"
        ] = "0" * 64
        forged_joint.write_bytes(manifest.canonical_json_bytes(forged_joint_value))
        forged_joint.chmod(0o444)
        with expect_raises(manifest.CandidatePublicationError, "exact tracked"):
            manifest.recover_candidate_publication(
                "TEST.m15.v2",
                forged_joint,
                repo_root=repo,
                registry_path="MODEL_REGISTRY.md",
                registry_entry=registry_entry,
                phase_seal_path=phase_rel,
                phase_seal=phase_seal,
            )

        index_entry = manifest.candidate_index_entry(candidate_manifest)
        with expect_raises(book_runtime.BookRuntimeError, "inactive"):
            book_runtime.require_active_book_lifecycle(
                "TEST.m15.v2", index_entry, candidate_manifest
            )
        with expect_raises(deriv_floor_resolver.FloorResolutionError, "inactive"):
            deriv_floor_resolver.resolve_book(
                "TEST",
                "TEST.m15.v2",
                index_entry,
                candidate_manifest,
                {},
                "synthetic-strategy.json",
                ticksettle_dir=temporary / "no-ticksettle",
            )

    # A Comparator-A commit-guard rejection at every publish mutation boundary
    # restores the exact stage and all three registration preimages.
    for reject_at in range(1, 7):
        with tempfile.TemporaryDirectory(
            prefix=f"m15-publication-guard-{reject_at}-"
        ) as raw:
            temporary = Path(raw)
            repo, candidate_manifest, resolved, artifacts, _ = _candidate_fixture(
                temporary
            )
            staged = manifest.stage_candidate(
                candidate_manifest, resolved, artifacts, repo_root=repo
            )
            stage = Path(staged["stage_dir"])
            joint_path = _write_joint_result(repo, staged["publish_lock"])
            registry_entry, phase_seal, phase_rel = (
                _publication_registration_material()
            )
            index_path = repo / "books" / "INDEX.json"
            registry_path = repo / "MODEL_REGISTRY.md"
            phase_path = repo / phase_rel
            index_before = index_path.read_bytes()
            registry_before = registry_path.read_bytes()
            calls = 0

            def reject_publish_commit() -> None:
                nonlocal calls
                calls += 1
                if calls == reject_at:
                    raise InjectedCrash(f"publish commit guard {reject_at}")

            with expect_raises(InjectedCrash, f"guard {reject_at}"):
                manifest.publish_candidate(
                    stage,
                    joint_path,
                    repo_root=repo,
                    registry_path="MODEL_REGISTRY.md",
                    registry_entry=registry_entry,
                    phase_seal_path=phase_rel,
                    phase_seal=phase_seal,
                    commit_guard=reject_publish_commit,
                )
            equal(calls, reject_at, f"publish guard {reject_at} call count")
            check(stage.is_dir(), f"publish guard {reject_at} lost restored stage")
            check(
                not (repo / "books" / "TEST.m15.v2").exists(),
                f"publish guard {reject_at} left destination",
            )
            equal(
                index_path.read_bytes(),
                index_before,
                f"publish guard {reject_at} changed INDEX",
            )
            equal(
                registry_path.read_bytes(),
                registry_before,
                f"publish guard {reject_at} changed registry",
            )
            check(
                not phase_path.exists(),
                f"publish guard {reject_at} left phase seal",
            )

    # Recovery keeps the preexisting authenticated destination but rolls back
    # only registration writes made before a late guard rejection.
    for reject_at in range(1, 5):
        with tempfile.TemporaryDirectory(
            prefix=f"m15-recovery-guard-{reject_at}-"
        ) as raw:
            temporary = Path(raw)
            repo, candidate_manifest, resolved, artifacts, _ = _candidate_fixture(
                temporary
            )
            staged = manifest.stage_candidate(
                candidate_manifest, resolved, artifacts, repo_root=repo
            )
            stage = Path(staged["stage_dir"])
            joint_path = _write_joint_result(repo, staged["publish_lock"])
            registry_entry, phase_seal, phase_rel = (
                _publication_registration_material()
            )

            def stop_after_rename(_destination: Path) -> None:
                raise InjectedCrash("prepare recovery destination")

            with expect_raises(InjectedCrash, "prepare recovery"):
                manifest.publish_candidate(
                    stage,
                    joint_path,
                    repo_root=repo,
                    registry_path="MODEL_REGISTRY.md",
                    registry_entry=registry_entry,
                    phase_seal_path=phase_rel,
                    phase_seal=phase_seal,
                    after_rename=stop_after_rename,
                )
            destination = repo / "books" / "TEST.m15.v2"
            index_path = repo / "books" / "INDEX.json"
            registry_path = repo / "MODEL_REGISTRY.md"
            phase_path = repo / phase_rel
            index_before = index_path.read_bytes()
            registry_before = registry_path.read_bytes()
            calls = 0

            def reject_recovery_commit() -> None:
                nonlocal calls
                calls += 1
                if calls == reject_at:
                    raise InjectedCrash(f"recovery commit guard {reject_at}")

            with expect_raises(InjectedCrash, f"guard {reject_at}"):
                manifest.recover_candidate_publication(
                    "TEST.m15.v2",
                    joint_path,
                    repo_root=repo,
                    registry_path="MODEL_REGISTRY.md",
                    registry_entry=registry_entry,
                    phase_seal_path=phase_rel,
                    phase_seal=phase_seal,
                    commit_guard=reject_recovery_commit,
                )
            equal(calls, reject_at, f"recovery guard {reject_at} call count")
            check(
                destination.is_dir(),
                f"recovery guard {reject_at} removed preexisting destination",
            )
            equal(
                index_path.read_bytes(),
                index_before,
                f"recovery guard {reject_at} changed INDEX",
            )
            equal(
                registry_path.read_bytes(),
                registry_before,
                f"recovery guard {reject_at} changed registry",
            )
            check(
                not phase_path.exists(),
                f"recovery guard {reject_at} left phase seal",
            )

    registry_entry, phase_seal, phase_rel = _publication_registration_material()
    conflicting_registry_rows = (
        (
            "substring",
            b"# Synthetic registry\n" + registry_entry.encode() + b" EXTRA-CONFLICTING-TEXT\n",
        ),
        (
            "duplicate",
            b"# Synthetic registry\n"
            + registry_entry.encode()
            + b"\n"
            + registry_entry.encode()
            + b"\n",
        ),
    )
    for case, conflicting_registry in conflicting_registry_rows:
        with tempfile.TemporaryDirectory(prefix=f"m15-publication-{case}-test-") as raw:
            temporary = Path(raw)
            repo, candidate_manifest, resolved, artifacts, _ = _candidate_fixture(temporary)
            staged = manifest.stage_candidate(
                candidate_manifest, resolved, artifacts, repo_root=repo
            )
            stage = Path(staged["stage_dir"])
            joint_path = _write_joint_result(repo, staged["publish_lock"])
            index_path = repo / "books" / "INDEX.json"
            registry_path = repo / "MODEL_REGISTRY.md"
            registry_path.write_bytes(conflicting_registry)
            index_before = index_path.read_bytes()
            registry_before = registry_path.read_bytes()
            phase_path = repo / phase_rel
            with expect_raises(manifest.CandidatePublicationError, "registry"):
                manifest.publish_candidate(
                    stage,
                    joint_path,
                    repo_root=repo,
                    registry_path="MODEL_REGISTRY.md",
                    registry_entry=registry_entry,
                    phase_seal_path=phase_rel,
                    phase_seal=phase_seal,
                )
            equal(index_path.read_bytes(), index_before, f"{case} registry conflict changed INDEX")
            equal(
                registry_path.read_bytes(),
                registry_before,
                f"{case} registry conflict changed registry",
            )
            check(not phase_path.exists(), f"{case} registry conflict created phase seal")
            check(not (repo / "books" / "TEST.m15.v2").exists(), f"{case} conflict published destination")
            check(stage.is_dir(), f"{case} conflict removed authenticated stage")

    with tempfile.TemporaryDirectory(prefix="m15-publication-index-conflict-test-") as raw:
        temporary = Path(raw)
        repo, candidate_manifest, resolved, artifacts, _ = _candidate_fixture(temporary)
        staged = manifest.stage_candidate(candidate_manifest, resolved, artifacts, repo_root=repo)
        stage = Path(staged["stage_dir"])
        joint_path = _write_joint_result(repo, staged["publish_lock"])
        index_path = repo / "books" / "INDEX.json"
        registry_path = repo / "MODEL_REGISTRY.md"
        conflict_entry = manifest.candidate_index_entry(candidate_manifest)
        conflict_entry["bundle_id"] = "f" * 64
        index_path.write_text(
            json.dumps({"schema": "book-index/v1", "books": [conflict_entry]}, indent=1)
            + "\n"
        )
        index_before = index_path.read_bytes()
        registry_before = registry_path.read_bytes()
        phase_path = repo / phase_rel
        with expect_raises(manifest.CandidatePublicationError, "conflicting INDEX"):
            manifest.publish_candidate(
                stage,
                joint_path,
                repo_root=repo,
                registry_path="MODEL_REGISTRY.md",
                registry_entry=registry_entry,
                phase_seal_path=phase_rel,
                phase_seal=phase_seal,
            )
        equal(index_path.read_bytes(), index_before, "conflicting INDEX registration was mutated")
        equal(registry_path.read_bytes(), registry_before, "INDEX conflict changed registry")
        check(not phase_path.exists(), "INDEX conflict created phase seal")
        check(not (repo / "books" / "TEST.m15.v2").exists(), "INDEX conflict published destination")
        check(stage.is_dir(), "INDEX conflict removed authenticated stage")

    with tempfile.TemporaryDirectory(prefix="m15-publication-unknown-destination-test-") as raw:
        temporary = Path(raw)
        repo, candidate_manifest, resolved, artifacts, _ = _candidate_fixture(temporary)
        staged = manifest.stage_candidate(candidate_manifest, resolved, artifacts, repo_root=repo)
        joint_path = _write_joint_result(repo, staged["publish_lock"])
        destination = repo / "books" / "TEST.m15.v2"
        destination.mkdir()
        unknown = destination / "unknown.txt"
        unknown.write_bytes(b"unowned destination\n")
        unknown.chmod(0o444)
        destination.chmod(0o555)
        index_path = repo / "books" / "INDEX.json"
        registry_path = repo / "MODEL_REGISTRY.md"
        index_before = index_path.read_bytes()
        registry_before = registry_path.read_bytes()
        phase_path = repo / phase_rel
        with expect_raises(manifest.CandidatePublicationError, "publish receipt"):
            manifest.recover_candidate_publication(
                "TEST.m15.v2",
                joint_path,
                repo_root=repo,
                registry_path="MODEL_REGISTRY.md",
                registry_entry=registry_entry,
                phase_seal_path=phase_rel,
                phase_seal=phase_seal,
            )
        equal(index_path.read_bytes(), index_before, "unknown destination changed INDEX")
        equal(registry_path.read_bytes(), registry_before, "unknown destination changed registry")
        check(not phase_path.exists(), "unknown destination created phase seal")
        equal(unknown.read_bytes(), b"unowned destination\n", "unknown destination was altered")

    with tempfile.TemporaryDirectory(prefix="m15-publication-mismatched-destination-test-") as raw:
        temporary = Path(raw)
        repo, candidate_manifest, resolved, artifacts, _ = _candidate_fixture(temporary)
        staged = manifest.stage_candidate(candidate_manifest, resolved, artifacts, repo_root=repo)
        joint_path = _write_joint_result(repo, staged["publish_lock"])
        mismatched = repo / "books" / "OTHER.m15.v2"
        Path(staged["stage_dir"]).rename(mismatched)
        index_path = repo / "books" / "INDEX.json"
        registry_path = repo / "MODEL_REGISTRY.md"
        index_before = index_path.read_bytes()
        registry_before = registry_path.read_bytes()
        phase_path = repo / phase_rel
        with expect_raises(manifest.CandidatePublicationError, "requested recovery"):
            manifest.recover_candidate_publication(
                "OTHER.m15.v2",
                joint_path,
                repo_root=repo,
                registry_path="MODEL_REGISTRY.md",
                registry_entry=registry_entry,
                phase_seal_path=phase_rel,
                phase_seal=phase_seal,
            )
        equal(index_path.read_bytes(), index_before, "mismatched destination changed INDEX")
        equal(registry_path.read_bytes(), registry_before, "mismatched destination changed registry")
        check(not phase_path.exists(), "mismatched destination created phase seal")
        check(mismatched.is_dir(), "mismatched destination was deleted")

    # Authority aliases and contradictory phase-seal fields must fail before
    # publication, even when the symlink resolves to the exact same bytes.
    with tempfile.TemporaryDirectory(prefix="m15-publication-authority-alias-test-") as raw:
        temporary = Path(raw)
        repo, candidate_manifest, resolved, artifacts, _ = _candidate_fixture(temporary)
        staged = manifest.stage_candidate(candidate_manifest, resolved, artifacts, repo_root=repo)
        stage = Path(staged["stage_dir"])
        joint_path = _write_joint_result(repo, staged["publish_lock"])
        alias = temporary / "repo-alias"
        alias.symlink_to(repo, target_is_directory=True)
        common = {
            "repo_root": repo,
            "registry_path": "MODEL_REGISTRY.md",
            "registry_entry": registry_entry,
            "phase_seal_path": phase_rel,
            "phase_seal": phase_seal,
        }
        with expect_raises(manifest.CandidatePublicationError, "exact repo books"):
            manifest.publish_candidate(
                alias / "books" / stage.name,
                joint_path,
                **common,
            )
        with expect_raises(manifest.CandidatePublicationError, "exact tracked"):
            manifest.publish_candidate(
                stage,
                alias / "results" / "json" / joint_path.name,
                **common,
            )
        with expect_raises(manifest.CandidatePublicationError, "exact repo-root"):
            manifest.publish_candidate(
                stage,
                joint_path,
                **dict(common, registry_path=alias / "MODEL_REGISTRY.md"),
            )
        contradictory_phase = dict(phase_seal, activation=True)
        with expect_raises(manifest.CandidatePublicationError, "no-activation schema"):
            manifest.publish_candidate(
                stage,
                joint_path,
                **dict(common, phase_seal=contradictory_phase),
            )
        check(stage.is_dir(), "authority rejection consumed the authenticated stage")
        check(
            not (repo / "books" / "TEST.m15.v2").exists(),
            "authority rejection published a destination",
        )

    # A mutation injected after the final stage authentication but inside the
    # rename boundary must be caught on the destination before registration.
    with tempfile.TemporaryDirectory(prefix="m15-publication-post-rename-auth-test-") as raw:
        temporary = Path(raw)
        repo, candidate_manifest, resolved, artifacts, _ = _candidate_fixture(temporary)
        staged = manifest.stage_candidate(candidate_manifest, resolved, artifacts, repo_root=repo)
        stage = Path(staged["stage_dir"])
        joint_path = _write_joint_result(repo, staged["publish_lock"])
        index_path = repo / "books" / "INDEX.json"
        registry_path = repo / "MODEL_REGISTRY.md"
        index_before = index_path.read_bytes()
        registry_before = registry_path.read_bytes()
        original_rename = manifest._rename_noreplace
        injected = False

        def mutate_inside_rename(source: Path, destination: Path) -> None:
            nonlocal injected
            if not injected and source.is_dir():
                injected = True
                source.chmod(0o755)
                model_path = source / "TEST_s0_lgb.txt"
                model_path.chmod(0o644)
                model_path.write_bytes(b"post-auth mutation\n")
                model_path.chmod(0o444)
                source.chmod(0o555)
            original_rename(source, destination)

        manifest._rename_noreplace = mutate_inside_rename
        try:
            with expect_raises(manifest.CandidatePublicationError, "payload hashes"):
                manifest.publish_candidate(
                    stage,
                    joint_path,
                    repo_root=repo,
                    registry_path="MODEL_REGISTRY.md",
                    registry_entry=registry_entry,
                    phase_seal_path=phase_rel,
                    phase_seal=phase_seal,
                )
        finally:
            manifest._rename_noreplace = original_rename
        destination = repo / "books" / "TEST.m15.v2"
        check(destination.is_dir(), "failed post-rename authentication lost evidence")
        equal(index_path.read_bytes(), index_before, "post-rename mutation changed INDEX")
        equal(
            registry_path.read_bytes(),
            registry_before,
            "post-rename mutation changed registry",
        )
        check(not (repo / phase_rel).exists(), "post-rename mutation created phase seal")

    # The exchange protocol must restore a non-cooperating write that lands
    # after the optimistic read instead of silently replacing it.
    with tempfile.TemporaryDirectory(prefix="m15-publication-cas-test-") as raw:
        target = Path(raw) / "authority.txt"
        target.write_bytes(b"old")
        original_exchange = manifest._rename_exchange
        exchange_calls = 0

        def mutate_before_exchange(source: Path, destination: Path) -> None:
            nonlocal exchange_calls
            if exchange_calls == 0:
                destination.write_bytes(b"concurrent")
            exchange_calls += 1
            original_exchange(source, destination)

        manifest._rename_exchange = mutate_before_exchange
        try:
            with expect_raises(manifest.CandidatePublicationError, "competing bytes restored"):
                manifest._atomic_replace_bytes(
                    target,
                    b"ours",
                    expected_current=b"old",
                )
        finally:
            manifest._rename_exchange = original_exchange
        equal(target.read_bytes(), b"concurrent", "CAS conflict overwrote competing bytes")
        equal(exchange_calls, 2, "CAS conflict did not perform one exchange and rollback")

    # Same-thread recovery from the deterministic post-rename hook must neither
    # deadlock nor make the outer publication report failure after success.
    with tempfile.TemporaryDirectory(prefix="m15-publication-reentrant-recovery-test-") as raw:
        temporary = Path(raw)
        repo, candidate_manifest, resolved, artifacts, _ = _candidate_fixture(temporary)
        staged = manifest.stage_candidate(candidate_manifest, resolved, artifacts, repo_root=repo)
        joint_path = _write_joint_result(repo, staged["publish_lock"])
        common = {
            "repo_root": repo,
            "registry_path": "MODEL_REGISTRY.md",
            "registry_entry": registry_entry,
            "phase_seal_path": phase_rel,
            "phase_seal": phase_seal,
        }

        def recover_inside_hook(_: Path) -> None:
            manifest.recover_candidate_publication(
                "TEST.m15.v2",
                joint_path,
                **common,
            )

        destination = manifest.publish_candidate(
            staged["stage_dir"],
            joint_path,
            after_rename=recover_inside_hook,
            **common,
        )
        check(destination.is_dir(), "reentrant recovery lost destination")
        index_rows = json.loads((repo / "books" / "INDEX.json").read_bytes())["books"]
        equal(len(index_rows), 1, "reentrant recovery duplicated INDEX registration")
        registry_lines = (repo / "MODEL_REGISTRY.md").read_bytes().splitlines()
        equal(
            registry_lines.count(registry_entry.encode()),
            1,
            "reentrant recovery duplicated text registration",
        )

    # Tick-settlement authority must be reached through real directory
    # components and a direct regular result file.
    with tempfile.TemporaryDirectory(prefix="m15-floor-authority-path-test-") as raw:
        temporary = Path(raw)
        real_results = temporary / "results" / "json"
        real_results.mkdir(parents=True)
        valid_result = {
            "mean_tick_minus_bar_COMB": -0.004,
            "verdict": {"tick_settlement_preserves_edge": True},
        }
        (real_results / "test_15m_ticksettle_result.json").write_text(
            json.dumps(valid_result)
        )
        alias_results = temporary / "results-alias"
        alias_results.symlink_to(real_results, target_is_directory=True)
        with expect_raises(deriv_floor_resolver.FloorResolutionError, "symlink"):
            deriv_floor_resolver.recompute_fallback_max(alias_results)

        symlink_results = temporary / "symlink-result-dir"
        symlink_results.mkdir()
        external_result = temporary / "external-result.json"
        external_result.write_text(json.dumps(valid_result))
        (symlink_results / "test_15m_ticksettle_result.json").symlink_to(
            external_result
        )
        with expect_raises(deriv_floor_resolver.FloorResolutionError, "symlinked"):
            deriv_floor_resolver.resolve_haircut(
                "TEST",
                "TEST.fixture",
                {},
                {},
                {},
                ticksettle_dir=symlink_results,
                proxy_table={},
            )

        invalid_result = real_results / "invalid_15m_ticksettle_result.json"
        for invalid_value in (True, float("nan"), float("inf")):
            invalid_result.write_text(
                json.dumps(
                    {
                        "mean_tick_minus_bar_COMB": invalid_value,
                        "verdict": {"tick_settlement_preserves_edge": "false"},
                    }
                )
            )
            with expect_raises(
                deriv_floor_resolver.FloorResolutionError,
                "finite non-boolean",
            ):
                deriv_floor_resolver._load_ticksettle(invalid_result)
        string_verdict = real_results / "string-verdict_15m_ticksettle_result.json"
        string_verdict.write_text(
            json.dumps(
                {
                    "mean_tick_minus_bar_COMB": -0.004,
                    "verdict": {"tick_settlement_preserves_edge": "true"},
                }
            )
        )
        _, validated = deriv_floor_resolver._load_ticksettle(string_verdict)
        check(not validated, "truthy non-boolean tick-settlement verdict was accepted")


def test_phase_resume_and_access_fence() -> None:
    with tempfile.TemporaryDirectory(prefix="m15-phase-test-") as raw:
        workspace = Path(raw) / "workspace"
        store = refresh.PhaseStore(workspace)
        sentinel = workspace / "synthetic.json"
        refresh.atomic_json_new(sentinel, {"schema": "synthetic-pre-April/v1"})
        source = store.seal(
            "source_preregistration",
            canonical_attempt_id="synthetic:source:0",
            output_paths=[sentinel],
            metadata={"semantic_outcomes_read": False},
        )
        resumed = store.seal(
            "source_preregistration",
            canonical_attempt_id="synthetic:source:0",
            output_paths=[sentinel],
            metadata={"semantic_outcomes_read": False},
        )
        equal(resumed, source, "sealed phase exact resume")
        with expect_raises(refresh.RefreshError, "differs on resume"):
            store.seal(
                "source_preregistration",
                canonical_attempt_id="synthetic:source:1",
                output_paths=[sentinel],
                metadata={"semantic_outcomes_read": False},
            )

        calls: list[str] = []
        fence = refresh.SemanticAccessFence(store, lambda: calls.append("called") or "synthetic")
        with expect_raises(refresh.RefreshError, "arm_seal"):
            fence()
        equal(calls, [], "semantic reader ran before arm seal")
        for phase in ("derived_data", "adapter_parity", "fit_and_repeat", "arm_seal"):
            store.seal(phase, canonical_attempt_id=f"synthetic:{phase}:0")
        equal(fence(), "synthetic", "semantic fence after synthetic arm seal")
        equal(calls, ["called"], "synthetic reader call count")

        with expect_raises(refresh.RefreshError, "already exists"):
            refresh.atomic_json_new(sentinel, {"schema": "overwrite/v1"})
        sentinel.chmod(0o644)
        sentinel.write_bytes(b"{}")
        with expect_raises(refresh.RefreshError, "hash verification"):
            store.load("source_preregistration")

        null_artifact = workspace / "null-calibration.json"
        refresh.atomic_json_new(null_artifact, {"schema": "synthetic-null/v1"})
        authority_artifact = workspace / "phase-zero-authority.json"
        refresh.atomic_json_new(
            authority_artifact, {"schema": "synthetic-phase-zero/v1"}
        )
        prereg_inputs = refresh._preregistration_phase_inputs(
            null_artifact, authority_artifact
        )
        resolved_inputs = [path.resolve() for path in prereg_inputs]
        equal(
            len(resolved_inputs), len(set(resolved_inputs)),
            "preregistration phase input paths are not unique",
        )
        equal(
            sum(path == refresh.SPEC_PATH.resolve() for path in resolved_inputs),
            1,
            "refresh spec was duplicated in source phase inputs",
        )

        corrupt_workspace = Path(raw) / "corrupt-workspace"
        corrupt_store = refresh.PhaseStore(corrupt_workspace)
        corrupt_store.seal(
            "source_preregistration", canonical_attempt_id="synthetic:corrupt:0"
        )
        corrupt_manifest = corrupt_store.path("source_preregistration")
        corrupt_manifest.chmod(0o644)
        corrupt_manifest.write_bytes(b"not-json")
        corrupt_manifest.chmod(0o444)
        with expect_raises(refresh.RefreshError, "cannot read phase manifest"):
            corrupt_store.has("source_preregistration")

        predecessor_workspace = Path(raw) / "predecessor-workspace"
        predecessor_store = refresh.PhaseStore(predecessor_workspace)
        predecessor_store.seal(
            "source_preregistration", canonical_attempt_id="synthetic:chain:0"
        )
        predecessor_store.seal(
            "derived_data", canonical_attempt_id="synthetic:chain:1"
        )
        source_manifest_path = predecessor_store.path("source_preregistration")
        source_manifest = predecessor_store.load("source_preregistration")
        source_manifest["metadata"] = {"tampered_after_seal": True}
        source_manifest_path.chmod(0o644)
        source_manifest_path.write_bytes(refresh.canonical_bytes(source_manifest))
        source_manifest_path.chmod(0o444)
        predecessor_store.load("source_preregistration")
        with expect_raises(refresh.RefreshError, "predecessor manifest hash"):
            predecessor_store.load("derived_data")


def _frozen_contract_fixture() -> tuple[
    dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]
]:
    spec = refresh.load_spec()
    pair, arm = "USDJPY", "B"
    pair_spec = spec["pairs"][pair]
    seeds = pair_spec["seed_order"]
    feature_cols = [f"f{index}" for index in range(pair_spec["feature_count"])]
    calibration = {
        "pair": pair,
        "target_coverage": pair_spec["calibration_target_coverage"],
        "confidence_threshold": 0.2,
        "calibration_rows": 100,
        "confidence_rows": 100,
        "structural_rows": None,
        "structural_column": None,
        "structural_quantile": None,
        "structural_threshold": None,
    }
    common = {
        "book_id": "workspace:USDJPY:B",
        "pair": pair,
        "arm": arm,
        "adapter_family": pair_spec["adapter_family"],
        "builder": pair_spec["builder"],
        "implementation_git_sha": "2" * 40,
        "prereg_id": "1" * 64,
        "lifecycle_status": "workspace_only_nonpublishable",
    }
    resolved = {
        "schema": "m15-book-refresh-resolved-bundle-spec/v1",
        **common,
        "feature_cols": feature_cols,
        "feature_dtype": "float32",
        "fit_ties": pair_spec["fit_ties"],
        "fit_session": pair_spec["fit_session"],
        "calibration_session": pair_spec["calibration_session"],
        "training_stride": pair_spec["training_stride"],
        "stride_phase": pair_spec["stride_phase"],
        "cap_selector": spec["cap_selector"],
        "fit_row_count": 150_000,
        "fit_row_id_sha256": "3" * 64,
        "early_stopping_row_count": 100,
        "early_stopping_row_id_sha256": "4" * 64,
        "calibration_row_count": 100,
        "calibration_row_id_sha256": "5" * 64,
        "fit_label_u8_sha256": "6" * 64,
        "early_stopping_label_u8_sha256": "7" * 64,
        "calibration_label_u8_sha256": "8" * 64,
        "split": spec["splits"][arm],
        "model_parameters_by_seed": [
            adapters.model_parameters(
                pair, seed, n_jobs=pair_spec["model"]["num_threads"]
            )
            for seed in seeds
        ],
        "seed_order": seeds,
        "best_iterations": [100] * len(seeds),
        "early_stopping_rule": {
            "rounds": pair_spec["model"]["early_stopping_rounds"],
            "metric": "auc",
            "rows": 100,
            "ties": pair_spec["early_stopping_ties"],
        },
        "nominal_source_years": {
            "fit": [str(year) for year in range(2012, 2022)],
            "validation_and_calibration": ["2021", "2022", "2023"],
        },
        "nominal_file_spill_audit": {"fixture": True},
        "calibration": calibration,
        "probability_tie_rule": "p>=0.5_is_UP",
        "control_permutation_seed": None,
        "control_permutation": None,
        "bundle_id": "9" * 64,
    }
    strategy = {
        "schema": "m15-book-refresh-strategy/v1",
        **common,
        "feature_cols": feature_cols,
        "n_features": len(feature_cols),
        "feature_dtype": "float32",
        "seed_order": seeds,
        "ensemble": "arithmetic_mean_probability_in_seed_order",
        "coverage": pair_spec["calibration_target_coverage"],
        "conf_thr": calibration["confidence_threshold"],
        "gate": {
            "confidence": "abs(p-0.5)>=conf_thr",
            "probability_tie": "p>=0.5_is_UP",
            "structural_column": None,
            "structural_quantile": None,
            "structural_threshold": None,
        },
        "horizon_seconds": 900,
        "outer_evaluator_owned": [
            "ny_session", "last_start_16:35", "wc_ret", "nonoverlap_chrono"
        ],
    }
    result = {
        "schema": "m15-book-refresh-fit-result/v1",
        "prereg_id": common["prereg_id"],
        "pair": pair,
        "arm": arm,
        "attempt": "canonical",
        "bundle_id": resolved["bundle_id"],
        "fit_row_count": resolved["fit_row_count"],
        "fit_row_id_sha256": resolved["fit_row_id_sha256"],
        "early_stopping_row_count": resolved["early_stopping_row_count"],
        "early_stopping_row_id_sha256": resolved["early_stopping_row_id_sha256"],
        "calibration_row_count": resolved["calibration_row_count"],
        "calibration_row_id_sha256": resolved["calibration_row_id_sha256"],
        "fit_label_u8_sha256": resolved["fit_label_u8_sha256"],
        "early_stopping_label_u8_sha256": resolved["early_stopping_label_u8_sha256"],
        "calibration_label_u8_sha256": resolved["calibration_label_u8_sha256"],
        "split": resolved["split"],
        "nominal_file_spill_audit": resolved["nominal_file_spill_audit"],
        "best_iterations": resolved["best_iterations"],
        "calibration": calibration,
        "feature_cols": feature_cols,
        "strategy": "strategy.json",
        "resolved_bundle_spec": "resolved.json",
        "models": [f"model-{seed}.txt" for seed in seeds],
        "model_sha256": ["a" * 64 for _seed in seeds],
        "seed_checkpoint_manifests": [f"checkpoint-{seed}.json" for seed in seeds],
        "seed_checkpoint_manifest_sha256": ["b" * 64 for _seed in seeds],
        "control_permutation_seed": None,
        "control_permutation": None,
    }
    return spec, resolved, strategy, result


def test_frozen_fit_contract_authentication() -> None:
    spec, resolved, strategy, result = _frozen_contract_fixture()
    refresh._validate_frozen_fit_contract(
        spec,
        pair="USDJPY",
        arm="B",
        resolved=resolved,
        strategy=strategy,
        result=result,
    )

    mutations: list[tuple[str, Callable[[dict[str, Any], dict[str, Any]], None]]] = [
        ("coverage", lambda r, s: r["calibration"].update(target_coverage=0.03)),
        ("stride", lambda r, s: r.update(training_stride=5)),
        ("stride phase", lambda r, s: r.update(stride_phase=1)),
        ("fit ties", lambda r, s: r.update(fit_ties="include_as_down")),
        ("tie rule", lambda r, s: r.update(probability_tie_rule="p>0.5_is_UP")),
        ("cap", lambda r, s: r["cap_selector"].update(cap=149_999)),
        ("fit session", lambda r, s: r.update(fit_session="all")),
        (
            "calibration session",
            lambda r, s: r.update(calibration_session="all"),
        ),
        (
            "model parameter",
            lambda r, s: r["model_parameters_by_seed"][0].update(num_leaves=255),
        ),
        ("ensemble", lambda r, s: s.update(ensemble="median_probability")),
    ]
    for name, mutate in mutations:
        changed_resolved = json.loads(json.dumps(resolved))
        changed_strategy = json.loads(json.dumps(strategy))
        mutate(changed_resolved, changed_strategy)
        with expect_raises(refresh.RefreshError):
            refresh._validate_frozen_fit_contract(
                spec,
                pair="USDJPY",
                arm="B",
                resolved=changed_resolved,
                strategy=changed_strategy,
                result=result,
            )


def test_fit_orchestration_integrity() -> None:
    """Exercise policy authentication, control seals, seed resume, and phase bytes."""

    def h(value: str) -> str:
        return hashlib.sha256(value.encode("ascii")).hexdigest()

    primary = {
        "feature_cols": ["f"],
        "split": {"name": "B"},
        "fit_row_count": 3,
        "fit_row_id_sha256": h("fit-rows"),
        "fit_label_u8_sha256": h("fit-labels"),
        "early_stopping_row_count": 2,
        "early_stopping_row_id_sha256": h("early-rows"),
        "early_stopping_label_u8_sha256": h("early-labels"),
        "calibration_row_count": 2,
        "calibration_row_id_sha256": h("cal-rows"),
    }
    control = dict(
        primary,
        fit_label_u8_sha256=h("permuted-fit-labels"),
        early_stopping_label_u8_sha256=h("permuted-early-labels"),
    )
    control["control_permutation"] = {
        "partition_rows": {
            "capped_fit_rows": 3,
            "early_stopping_rows": 2,
        },
        "partitions": {
            "capped_fit_rows": {
                "row_count": 3,
                "ordered_row_id_sha256": primary["fit_row_id_sha256"],
                "input_label_u8_sha256": primary["fit_label_u8_sha256"],
                "permutation_index_i64le_sha256": h("fit-permutation"),
                "permuted_label_u8_sha256": control["fit_label_u8_sha256"],
            },
            "early_stopping_rows": {
                "row_count": 2,
                "ordered_row_id_sha256": primary["early_stopping_row_id_sha256"],
                "input_label_u8_sha256": primary["early_stopping_label_u8_sha256"],
                "permutation_index_i64le_sha256": h("early-permutation"),
                "permuted_label_u8_sha256": control[
                    "early_stopping_label_u8_sha256"
                ],
            },
        }
    }
    identity = refresh._assert_control_partition_identity(
        primary, control, pair="EURUSD", control_arm="B_perm"
    )
    check(identity["partition_identity_exact"], "control partition identity not sealed")
    bad_control = dict(control, calibration_row_id_sha256=h("different-cal-rows"))
    with expect_raises(refresh.RefreshError, "non-label partition"):
        refresh._assert_control_partition_identity(
            primary, bad_control, pair="EURUSD", control_arm="B_perm"
        )

    with tempfile.TemporaryDirectory(prefix="m15-seed-resume-") as raw:
        workspace = Path(raw)
        model_dir = workspace / "models"
        fit_calls: list[int] = []
        feature_rows = mock.Mock(feature_cols=("f", "g"))

        def fake_fit(
            pair: str,
            seed: int,
            fit_rows: Any,
            fit_indices: np.ndarray,
            validation_rows: Any,
            validation_indices: np.ndarray,
            path: Path,
            *,
            n_jobs: int,
        ) -> adapters.SeedCheckpoint:
            del fit_rows, fit_indices, validation_rows, validation_indices, n_jobs
            fit_calls.append(seed)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"model-{seed}".encode("ascii"))
            return adapters.SeedCheckpoint(
                pair=pair,
                seed=seed,
                best_iteration=seed + 11,
                model_path=path,
                sha256=refresh.sha256_file(path),
            )

        def fake_validate(
            manifest_path: Path, model_path: Path, **expected: Any
        ) -> adapters.SeedCheckpoint:
            value = json.loads(manifest_path.read_bytes())
            equal(value["seed"], expected["seed"], "checkpoint seed binding")
            equal(
                value["fit_row_id_sha256"], expected["fit_row_id_sha256"],
                "checkpoint fit-row binding",
            )
            equal(
                value["fit_label_u8_sha256"], expected["fit_label_sha256"],
                "checkpoint fit-label binding",
            )
            equal(
                value["model_sha256"], refresh.sha256_file(model_path),
                "checkpoint model hash",
            )
            return adapters.SeedCheckpoint(
                pair=expected["pair"],
                seed=expected["seed"],
                best_iteration=value["best_iteration"],
                model_path=model_path,
                sha256=value["model_sha256"],
            )

        helper_kwargs = {
            "workspace": workspace,
            "prereg_id": "1" * 64,
            "implementation_git_sha": "2" * 40,
            "pair": "EURUSD",
            "arm": "B",
            "attempt": "canonical",
            "fit_rows": feature_rows,
            "fit_indices": np.array([0], dtype="int64"),
            "validation_rows": feature_rows,
            "validation_indices": np.array([0], dtype="int64"),
            "model_dir": model_dir,
            "seeds": [0, 1],
            "parameters_by_seed": [{"seed": 0}, {"seed": 1}],
            "n_jobs": 1,
            "fit_row_id_sha256": h("fit-rows"),
            "early_stopping_row_id_sha256": h("early-rows"),
            "fit_label_sha256": h("fit-labels"),
            "early_stopping_label_sha256": h("early-labels"),
            "fit_seed": fake_fit,
        }
        with mock.patch.object(
            refresh, "_validate_seed_checkpoint_manifest", side_effect=fake_validate
        ):
            first, manifests = refresh._fit_or_resume_seed_checkpoints(**helper_kwargs)
            equal(fit_calls, [0, 1], "initial per-seed fit order")
            seed_zero_hash = first[0].sha256
            first[1].model_path.unlink()
            manifests[1].unlink()
            fit_calls.clear()
            resumed, _ = refresh._fit_or_resume_seed_checkpoints(**helper_kwargs)
        equal(fit_calls, [1], "resume refit more than the missing seed")
        equal(resumed[0].sha256, seed_zero_hash, "resume replaced valid seed zero")

    _test_authenticated_replay_policy()
    _test_fit_phase_full_inventory()


def _test_authenticated_replay_policy() -> None:
    with tempfile.TemporaryDirectory(prefix="m15-policy-auth-") as raw:
        workspace = Path(raw)
        spec = refresh.load_spec()
        pair, arm = "EURUSD", "B"
        model_path = workspace / "EURUSD_B_s0_lgb.txt"
        model_path.write_bytes(b"synthetic-model-bytes")
        common = {
            "book_id": "workspace:EURUSD:B",
            "pair": pair,
            "arm": arm,
            "adapter_family": spec["pairs"][pair]["adapter_family"],
            "builder": spec["pairs"][pair]["builder"],
            "implementation_git_sha": "2" * 40,
            "prereg_id": "1" * 64,
            "seed_order": spec["pairs"][pair]["seed_order"],
            "lifecycle_status": "workspace_only_nonpublishable",
        }
        calibration = {
            "target_coverage": 0.1,
            "confidence_threshold": 0.2,
            "structural_column": None,
            "structural_quantile": None,
            "structural_threshold": None,
        }
        resolved_without_id = {
            "schema": "m15-book-refresh-resolved-bundle-spec/v1",
            **common,
            "feature_cols": ["f"],
            "feature_dtype": "float32",
            "calibration": calibration,
        }
        strategy = {
            "schema": "m15-book-refresh-strategy/v1",
            **common,
            "feature_cols": ["f"],
            "n_features": 1,
            "feature_dtype": "float32",
            "coverage": 0.1,
            "conf_thr": 0.2,
            "gate": {
                "structural_column": None,
                "structural_quantile": None,
                "structural_threshold": None,
            },
        }
        strategy_path = workspace / "EURUSD_B_strategy.json"
        strategy_path.write_bytes(refresh.canonical_bytes(strategy))
        bundle_id = refresh.derive_bundle_id(
            spec,
            resolved_without_id,
            strategy_path.name,
            strategy_path.read_bytes(),
            [(model_path.name, model_path.read_bytes())],
        )
        resolved_path = workspace / "resolved_bundle_spec.json"
        resolved_path.write_bytes(
            refresh.canonical_bytes(dict(resolved_without_id, bundle_id=bundle_id))
        )
        result = {
            "pair": pair,
            "arm": arm,
            "attempt": "canonical",
            "bundle_id": bundle_id,
            "resolved_bundle_spec": resolved_path.name,
            "strategy": strategy_path.name,
            "models": [model_path.name],
            "model_sha256": [sha256(model_path)],
            "feature_cols": ["malicious-duplicate"],
            "calibration": {"target_coverage": 0.9, "confidence_threshold": 9.0},
        }
        policy = replay._policy_from_fit_result(
            workspace,
            pair,
            arm,
            result,
            spec=spec,
            expected_bundle_id=bundle_id,
            expected_prereg_id="1" * 64,
            expected_implementation_git_sha="2" * 40,
        )
        equal(policy.feature_cols, ("f",), "replay consumed fit-result feature duplicate")
        equal(policy.confidence_threshold, 0.2, "replay consumed fit-result calibration")
        strategy_path.write_bytes(strategy_path.read_bytes() + b" ")
        with expect_raises(replay.ReplayInputError, "bundle bytes"):
            replay._policy_from_fit_result(
                workspace,
                pair,
                arm,
                result,
                spec=spec,
                expected_bundle_id=bundle_id,
                expected_prereg_id="1" * 64,
                expected_implementation_git_sha="2" * 40,
            )


def _test_fit_phase_full_inventory() -> None:
    with tempfile.TemporaryDirectory(prefix="m15-fit-phase-inventory-") as raw:
        workspace = Path(raw)
        store = refresh.PhaseStore(workspace)
        for phase in ("source_preregistration", "derived_data", "adapter_parity"):
            sentinel = workspace / f"{phase}.bin"
            sentinel.write_bytes(phase.encode("ascii"))
            store.seal(
                phase,
                canonical_attempt_id=f"test:{phase}",
                output_paths=[sentinel],
            )
        attempt = workspace / "fits" / "EURUSD" / "B" / "canonical"
        attempt.mkdir(parents=True)
        result_path = attempt / "fit_result.json"
        strategy_path = attempt / "strategy.json"
        resolved_path = attempt / "resolved.json"
        model_path = attempt / "model.txt"
        checkpoint_path = attempt / "model.txt.checkpoint.json"
        for path, data in (
            (strategy_path, b"strategy"),
            (resolved_path, b"resolved"),
            (model_path, b"model"),
            (checkpoint_path, b"checkpoint"),
            (attempt / "fit_row_id.i64le", b"fit-row"),
            (attempt / "early_stopping_row_id.i64le", b"early-row"),
            (attempt / "calibration_row_id.i64le", b"cal-row"),
        ):
            path.write_bytes(data)
        inventory_result = {
            "strategy": strategy_path.relative_to(workspace).as_posix(),
            "resolved_bundle_spec": resolved_path.relative_to(workspace).as_posix(),
            "models": [model_path.relative_to(workspace).as_posix()],
            "seed_checkpoint_manifests": [
                checkpoint_path.relative_to(workspace).as_posix()
            ],
        }
        result_path.write_bytes(refresh.canonical_bytes(inventory_result))
        inventory = refresh._fit_artifact_inventory(workspace, result_path, inventory_result)
        equal(len(inventory), 8, "fit phase artifact inventory size")
        store.seal(
            "fit_and_repeat",
            canonical_attempt_id="test:fit",
            output_paths=inventory,
        )
        model_path.write_bytes(b"corrupt-model")
        with expect_raises(refresh.RefreshError, "phase hash verification"):
            store.load("fit_and_repeat")


def _accepted_null_artifact_fixture(implementation_git_sha: str) -> dict[str, Any]:
    from dataclasses import asdict
    from scipy.stats import beta

    spec = refresh.load_spec()
    contract = spec["null_calibration"]
    count = 0
    cp_upper = float(beta.ppf(0.95, count + 1, contract["campaigns_per_cell"] - count))
    cells = [
        {
            "schema": "m15-book-refresh-null-calibration-cell/v1",
            "fixture": fixture,
            "helper": helper,
            "k_shadow": k_shadow,
            "ordered_shadow_pairs": [],
            "accuracy_null": 0.5,
            "binding": True,
            "calibration_role": "binding_current",
            "campaign_count": contract["campaigns_per_cell"],
            "bootstrap_replicates": contract["bootstrap_replicates"],
            "block_lengths": contract["block_lengths"],
            "false_positive_count": count,
            "false_positive_limit": contract["false_positive_limit_inclusive"],
            "clopper_pearson_upper_95": cp_upper,
            "cell_passed": True,
            "accepted": True,
            "deterministic_repeat": True,
            "campaign_digest": hashlib.sha256(
                f"synthetic-structure-only/{fixture}/{helper}/{k_shadow}".encode()
            ).hexdigest(),
        }
        for fixture, helper, k_shadow in refresh._expected_null_cell_keys(spec)
    ]
    return {
        "schema": "m15-book-refresh-null-calibration-result/v1",
        "implementation_git_sha": implementation_git_sha,
        "spec_sha256": refresh.sha256_file(refresh.SPEC_PATH),
        "source_files": [asdict(refresh.file_identity(path)) for path in refresh.source_files()],
        "contract": contract,
        "cells": cells,
        "cells_sha256": refresh.length_prefixed_digest(
            "m15-book-refresh/null-calibration-cells/v1",
            [refresh.canonical_bytes(cells)],
        ),
        "accepted": True,
        "semantic_replay_outcomes_accessed": False,
    }


def _write_read_only_json(path: Path, value: Any) -> Path:
    return refresh.atomic_json_new(path, value, read_only=True)


def _provider_fixture(
    pair: str,
    *,
    implementation_git_sha: str,
    passed: bool,
) -> dict[str, Any]:
    from dataclasses import asdict

    spec = refresh.load_spec()
    incumbent = book_runtime.load_book(pair, spec["pairs"][pair]["book_id_A"])
    feature_cols = list(incumbent.feature_cols)
    sources = [
        asdict(refresh.file_identity(REPO_ROOT / "scripts" / name))
        for name in (
            "live_features.py",
            "pipeline.py",
            "m15_book_refresh_adapters.py",
            "book_runtime.py",
            "deriv_hot_daemon.py",
        )
    ]
    if pair == "EURUSD":
        status = "BLOCKED_SCHEMA"
        reason: str | None = "verified_live_OF_provider_absent"
    elif passed:
        status = "PASSED"
        reason = None
    else:
        status = "FAILED_PARITY"
        reason = "synthetic_provider_parity_failure"
    return {
        "schema": "m15-book-refresh-provider-parity/v1",
        "status": status,
        "reason": reason,
        "feature_schema_parity": True,
        "feature_value_parity": passed,
        "runtime_scoring_and_gate_parity": passed,
        "provider_schema_identity": refresh.sha256_bytes(
            refresh.canonical_bytes(feature_cols)
        ),
        "provider_implementation_identity": refresh.length_prefixed_digest(
            "m15-book-refresh/provider-implementation/v1",
            [refresh.canonical_bytes(sources)],
        ),
        "provider_source_files": sources,
        "runtime_implementation_git_sha": implementation_git_sha,
        "direct_incumbent_builder_parity": {
            "schema": "m15-direct-incumbent-builder-parity/v1",
            "pair": pair,
            "all_equal": True,
            "source_to_decision_shift_exact": True,
        },
        "live_feature_builder_parity": (
            None
            if pair == "EURUSD"
            else {
                "schema": "m15-live-feature-builder-parity/v1",
                "pair": pair,
                "public_class": "live_features.LiveFeatureBuilder",
                "public_method_exercised": "feature_row",
                "provider_rows": 100,
                "compared_rows": 80,
                "feature_count": len(feature_cols),
                "feature_order_exact": True,
                "source_clock_subset_exact": True,
                "causal_decision_mapping_exact": True,
                "finite_mask_exact": True,
                "ordered_float32_feature_bytes_exact": False,
                "ordered_finite_float32_values_exact": True,
                "nonfinite_nan_payload_bits_binding": False,
                "decoded_diagnostics": {
                    "mismatched_cells": 0,
                    "finite_nonfinite_disagreements": 0,
                    "columns": {},
                },
                "public_latest_row_exact": True,
                "public_latest_source_feature_ns": int(
                    pd.Timestamp("2026-03-31T23:58:00Z").value
                ),
                "all_equal": False,
                "behavior_all_equal": True,
            }
        ),
        "runtime_decision_clock_authority": {
            "function": "deriv_hot_daemon.score_from_snapshot",
            "field": "candidate_bar_close_utc",
            "mapping": "LiveFeatureBuilder.FeatureRow.timestamp+1minute",
            "source_feature_to_decision_shift_s": 60,
            "adapter_mapping_exact": True,
            "authority_source_sha256": next(
                source["sha256"]
                for source in sources
                if source["path"] == "scripts/deriv_hot_daemon.py"
            ),
        },
        "provider_kind": (
            "offline_xpair_plus_unavailable_live_orderflow"
            if pair == "EURUSD"
            else (
                "public_live_feature_builder_cross_pair"
                if spec["pairs"][pair]["adapter_family"] == "gbpchf_xpair"
                else "public_live_feature_builder_base"
            )
        ),
    }


def _shadow_record_fixture(
    pair: str,
    *,
    evidence: dict[str, Any],
) -> dict[str, Any]:
    pair_spec = refresh.load_spec()["pairs"][pair]
    return {
        "pair": pair,
        "candidate_book_id": pair_spec["candidate_book_id"],
        "candidate_bundle_id": hashlib.sha256(f"candidate/{pair}".encode()).hexdigest(),
        "v1_book_id": pair_spec["book_id_A"],
        "v1_bundle_id": hashlib.sha256(f"v1/{pair}".encode()).hexdigest(),
        "lifecycle_status": "inactive_shadow_candidate",
        "activation": False,
        "provider_evidence": evidence,
    }


def test_integrity_artifacts_cli_and_shadow() -> None:
    implementation_sha = "a" * 40
    with tempfile.TemporaryDirectory(prefix="m15-integrity-test-") as raw:
        temporary = Path(raw)
        recoverable = temporary / "recoverable-tracked-result.json"
        refresh._seal_json_exact(recoverable, {"schema": "synthetic-partial/v1"})
        refresh._seal_json_exact(recoverable, {"schema": "synthetic-partial/v1"})
        with expect_raises(refresh.RefreshError, "differs from reconstructed bytes"):
            refresh._seal_json_exact(recoverable, {"schema": "foreign-partial/v1"})

        valid = _accepted_null_artifact_fixture(implementation_sha)
        valid_path = _write_read_only_json(temporary / "valid-null.json", valid)
        accepted = refresh._validate_null_calibration_artifact(
            valid_path,
            implementation_git_sha=implementation_sha,
            require_accepted=True,
        )
        equal(
            len(accepted["cells"]),
            len(refresh._expected_null_cell_keys(refresh.load_spec())),
            "authenticated null cell count",
        )

        boundary = json.loads(json.dumps(valid))
        boundary_count = boundary["contract"]["false_positive_limit_inclusive"]
        from scipy.stats import beta
        boundary_cp = float(
            beta.ppf(
                0.95,
                boundary_count + 1,
                boundary["contract"]["campaigns_per_cell"] - boundary_count,
            )
        )
        for cell in boundary["cells"]:
            cell["false_positive_count"] = boundary_count
            cell["false_positive_limit"] = boundary_count
            cell["clopper_pearson_upper_95"] = boundary_cp
            cell["accepted"] = True
        boundary["cells_sha256"] = refresh.length_prefixed_digest(
            "m15-book-refresh/null-calibration-cells/v1",
            [refresh.canonical_bytes(boundary["cells"])],
        )
        boundary_path = _write_read_only_json(
            temporary / "boundary-null.json", boundary
        )
        refresh._validate_null_calibration_artifact(
            boundary_path,
            implementation_git_sha=implementation_sha,
            require_accepted=True,
        )

        extra = json.loads(json.dumps(valid))
        extra["unsealed_extra"] = True
        extra_path = _write_read_only_json(temporary / "extra-null.json", extra)
        with expect_raises(refresh.RefreshError, "fields differ"):
            refresh._validate_null_calibration_artifact(
                extra_path,
                implementation_git_sha=implementation_sha,
                require_accepted=True,
            )

        semantic = json.loads(json.dumps(valid))
        semantic["semantic_replay_outcomes_accessed"] = True
        semantic_path = _write_read_only_json(temporary / "semantic-null.json", semantic)
        with expect_raises(refresh.RefreshError, "identity/contract"):
            refresh._validate_null_calibration_artifact(
                semantic_path,
                implementation_git_sha=implementation_sha,
                require_accepted=True,
            )

        cell_tamper = json.loads(json.dumps(valid))
        cell_tamper["cells"][0]["campaign_digest"] = "f" * 64
        cell_tamper_path = _write_read_only_json(
            temporary / "cell-tamper-null.json", cell_tamper
        )
        with expect_raises(refresh.RefreshError, "cell payload digest"):
            refresh._validate_null_calibration_artifact(
                cell_tamper_path,
                implementation_git_sha=implementation_sha,
                require_accepted=True,
            )

        recomputed_tamper = json.loads(json.dumps(valid))
        rejected_count = recomputed_tamper["contract"]["false_positive_limit"] + 1
        recomputed_tamper["cells"][0]["false_positive_count"] = rejected_count
        recomputed_tamper["cells"][0]["clopper_pearson_upper_95"] = float(
            beta.ppf(
                0.95,
                rejected_count + 1,
                recomputed_tamper["contract"]["campaigns_per_cell"] - rejected_count,
            )
        )
        recomputed_tamper["cells"][0]["cell_passed"] = False
        recomputed_tamper["cells"][0]["accepted"] = False
        recomputed_tamper["cells_sha256"] = refresh.length_prefixed_digest(
            "m15-book-refresh/null-calibration-cells/v1",
            [refresh.canonical_bytes(recomputed_tamper["cells"])],
        )
        recomputed_tamper["accepted"] = False
        recomputed_path = _write_read_only_json(
            temporary / "recomputed-tamper-null.json", recomputed_tamper
        )
        with expect_raises(refresh.RefreshError, "did not accept"):
            refresh._validate_null_calibration_artifact(
                recomputed_path,
                implementation_git_sha=implementation_sha,
                require_accepted=True,
            )

        aggregate_tamper = json.loads(json.dumps(valid))
        aggregate_tamper["accepted"] = False
        aggregate_path = _write_read_only_json(
            temporary / "aggregate-tamper-null.json", aggregate_tamper
        )
        with expect_raises(refresh.RefreshError, "aggregate acceptance"):
            refresh._validate_null_calibration_artifact(
                aggregate_path,
                implementation_git_sha=implementation_sha,
                require_accepted=True,
            )

        partial_pid = "2" * 64
        expected_partial = (
            "?? results/json/"
            f"m15_book_refresh_{partial_pid}_preregister_result.json\n"
        )
        fake_store = mock.MagicMock()
        synthetic_runtime_contract = {
            "authority": "synthetic",
            "expected_exact": {},
            "actual_exact": {},
            "all_exact": True,
        }
        with mock.patch.object(refresh, "git_porcelain", return_value=expected_partial) as dirty, \
                mock.patch.object(refresh, "git_head", return_value=implementation_sha), \
                mock.patch.object(
                    refresh,
                    "verified_runtime_version_contract",
                    return_value=synthetic_runtime_contract,
                ), \
                mock.patch.object(refresh, "_null_calibration_path", return_value=REPO_ROOT / "README.md"), \
                mock.patch.object(
                    refresh,
                    "_validate_null_calibration_artifact",
                    return_value={
                        "accepted": True,
                        "cells": [
                            {} for _ in refresh._expected_null_cell_keys(refresh.load_spec())
                        ],
                    },
                ), \
                mock.patch.object(
                    refresh,
                    "_validate_phase_zero_authority",
                    return_value={
                        "acceptance_artifact": True,
                        "pairs": {pair: {} for pair in stats.PAIR_ORDER},
                        "feature_source_pair_order": list(
                            stats.FEATURE_SOURCE_PAIR_ORDER
                        ),
                    },
                ), \
                mock.patch.object(
                    refresh,
                    "PHASE_ZERO_AUTHORITY_PATH",
                    REPO_ROOT / "REPO_MAP.md",
                ), \
                mock.patch.object(refresh, "issue_body", return_value="synthetic issue body"), \
                mock.patch.object(refresh, "comparator_a_files", return_value=[]), \
                mock.patch.object(refresh, "historical_feature_files", return_value=[]), \
                mock.patch.object(refresh, "_source_file_identities", return_value=[]), \
                mock.patch.object(refresh, "derive_prereg_id", return_value=partial_pid), \
                mock.patch.object(refresh, "WORK_ROOT", temporary / "prereg-work"), \
                mock.patch.object(refresh, "_seal_json_exact") as seal_exact, \
                mock.patch.object(refresh, "PhaseStore", return_value=fake_store):
            equal(refresh.preregister(), partial_pid, "exact partial preregistration resume")
            equal(seal_exact.call_count, 2, "partial preregistration exact evidence seals")
            check(fake_store.seal.called, "partial preregistration phase was not resumed")
            dirty.return_value = (
                "?? results/json/"
                f"m15_book_refresh_{'3' * 64}_preregister_result.json\n"
            )
            with expect_raises(refresh.RefreshError, "foreign partial"):
                refresh.preregister()

    dispatches = [
        ("--smoke", "smoke", []),
        ("--null-calibration", "run_null_calibration_gate", []),
        ("--preregister", "preregister", []),
        ("--build-snapshot", "build_blind_snapshot", ["p" * 64]),
        ("--adapter-parity", "run_adapter_parity", ["p" * 64]),
        ("--run-fits", "run_all_fits", ["p" * 64]),
        ("--seal-arms", "seal_arms", ["p" * 64]),
        ("--run-replay", "run_replay", ["p" * 64]),
        ("--lock-results", "lock_results_and_candidates", ["p" * 64]),
        ("--publish-candidates", "publish_candidates", ["p" * 64]),
        ("--shadow-spec", "seal_shadow_spec", ["p" * 64]),
    ]
    for flag, function_name, expected_args in dispatches:
        argv = [flag]
        if expected_args:
            argv.extend(["--prereg-id", expected_args[0]])
        with mock.patch.object(refresh, function_name) as function:
            equal(refresh.main(argv), 0, f"{flag} CLI return")
            function.assert_called_once_with(*expected_args)
    with mock.patch.object(refresh, "fit_one") as fit_one:
        equal(
            refresh.main([
                "--fit-one", "--prereg-id", "p" * 64, "--pair", "EURUSD",
                "--arm", "B", "--attempt", "primary",
            ]),
            0,
            "--fit-one CLI return",
        )
        fit_one.assert_called_once_with("p" * 64, "EURUSD", "B", "primary")
    with expect_raises(refresh.RefreshError, "requires --prereg-id"):
        refresh.main(["--run-replay"])
    with mock.patch.object(refresh, "load_spec", return_value={"schema": "synthetic/v1"}), \
            mock.patch.object(refresh, "evaluator_id", return_value="e" * 64):
        equal(refresh.main(["--validate-spec"]), 0, "--validate-spec CLI return")

    spec = refresh.load_spec()
    parity_sha = "b" * 64
    evaluator = "c" * 64
    passing_provider = _provider_fixture(
        "USDJPY", implementation_git_sha=implementation_sha, passed=True
    )
    passing_evidence, qualified, reason = refresh._validated_shadow_provider_evidence(
        "USDJPY",
        passing_provider,
        spec=spec,
        implementation_git_sha=implementation_sha,
        parity_sha256=parity_sha,
        evaluator_id_value=evaluator,
        expected_compared_rows=80,
    )
    check(qualified and reason is None, "passing provider was not shadow-qualified")
    check(
        passing_provider["live_feature_builder_parity"][
            "ordered_float32_feature_bytes_exact"
        ]
        is False,
        "shadow qualification fixture did not exercise non-binding NaN bytes",
    )
    bad_schema = json.loads(json.dumps(passing_provider))
    bad_schema["provider_schema_identity"] = "0" * 64
    with expect_raises(refresh.RefreshError, "schema identity differs"):
        refresh._validated_shadow_provider_evidence(
            "USDJPY",
            bad_schema,
            spec=spec,
            implementation_git_sha=implementation_sha,
            parity_sha256=parity_sha,
            evaluator_id_value=evaluator,
            expected_compared_rows=80,
        )
    bad_provider = json.loads(json.dumps(passing_provider))
    bad_provider["provider_implementation_identity"] = "d" * 64
    with expect_raises(refresh.RefreshError, "implementation identity"):
        refresh._validated_shadow_provider_evidence(
            "USDJPY",
            bad_provider,
            spec=spec,
            implementation_git_sha=implementation_sha,
            parity_sha256=parity_sha,
            evaluator_id_value=evaluator,
            expected_compared_rows=80,
        )
    misleading_provider = json.loads(json.dumps(passing_provider))
    misleading_provider["feature_value_parity"] = False
    with expect_raises(refresh.RefreshError, "failed requirement"):
        refresh._validated_shadow_provider_evidence(
            "USDJPY",
            misleading_provider,
            spec=spec,
            implementation_git_sha=implementation_sha,
            parity_sha256=parity_sha,
            evaluator_id_value=evaluator,
            expected_compared_rows=80,
        )
    behavior_forgery = json.loads(json.dumps(passing_provider))
    behavior_forgery["live_feature_builder_parity"]["behavior_all_equal"] = False
    with expect_raises(refresh.RefreshError, "LiveFeatureBuilder evidence"):
        refresh._validated_shadow_provider_evidence(
            "USDJPY",
            behavior_forgery,
            spec=spec,
            implementation_git_sha=implementation_sha,
            parity_sha256=parity_sha,
            evaluator_id_value=evaluator,
            expected_compared_rows=80,
        )

    def reject_live_forgery(mutator: Callable[[dict[str, Any]], None]) -> None:
        forged = json.loads(json.dumps(passing_provider))
        mutator(forged["live_feature_builder_parity"])
        with expect_raises(refresh.RefreshError):
            refresh._validated_shadow_provider_evidence(
                "USDJPY",
                forged,
                spec=spec,
                implementation_git_sha=implementation_sha,
                parity_sha256=parity_sha,
                evaluator_id_value=evaluator,
                expected_compared_rows=80,
            )

    reject_live_forgery(lambda live: live.pop("public_class"))
    reject_live_forgery(lambda live: live.__setitem__("unknown_field", True))
    reject_live_forgery(lambda live: live.__setitem__("provider_rows", 1))
    reject_live_forgery(
        lambda live: live.update(provider_rows=1, compared_rows=1)
    )
    reject_live_forgery(
        lambda live: live.__setitem__(
            "public_latest_source_feature_ns",
            int(pd.Timestamp("2026-04-01T00:00:00Z").value),
        )
    )
    reject_live_forgery(lambda live: live.__setitem__("all_equal", True))
    reject_live_forgery(lambda live: live.__setitem__("finite_mask_exact", False))
    reject_live_forgery(
        lambda live: live.__setitem__("nonfinite_nan_payload_bits_binding", True)
    )
    reject_live_forgery(
        lambda live: live.__setitem__(
            "decoded_diagnostics",
            {
                "mismatched_cells": 1,
                "finite_nonfinite_disagreements": 0,
                "columns": {},
            },
        )
    )

    prereg_id = "d" * 64
    control_arms = [
        [family, arm, hashlib.sha256(f"control/{family}/{arm}".encode()).hexdigest()]
        for family, arm in spec["control_order"]
    ]
    controls_id = refresh.derive_controls_id(spec, control_arms)
    primary_arms = [
        [pair, arm, hashlib.sha256(f"primary/{pair}/{arm}".encode()).hexdigest()]
        for pair in spec["pair_order"]
        for arm in spec["arm_order"]
    ]
    run_id = refresh.derive_run_id(spec, prereg_id, controls_id, primary_arms)
    completed = {
        "prereg_id": prereg_id,
        "run_id": run_id,
        "implementation_git_sha": implementation_sha,
        "evaluator_id": evaluator,
        "controls_id": controls_id,
        "ordered_primary_arms_with_bundle_ids": primary_arms,
        "ordered_control_arms_with_bundle_ids": control_arms,
        "run_binding_sha256": "1" * 64,
    }
    common = {
        **completed,
        "provider_parity_result": "logs/m15_book_refresh/synthetic/parity.json",
        "provider_parity_result_sha256": parity_sha,
        "activation": False,
        "shadow_started": False,
    }
    zero = {
        "schema": spec["shadow"]["schemas"]["zero_eligible"],
        "status": spec["shadow"]["statuses"]["empty_survivors"],
        **common,
        "S": [],
        "S_records": [],
        "S_shadow": [],
        "k_shadow": 0,
        "exclusions": {},
    }
    refresh._validate_shadow_handoff_payload(zero, spec=spec)
    zero_with_t0 = dict(zero, T0=None)
    with expect_raises(refresh.RefreshError, "fields differ"):
        refresh._validate_shadow_handoff_payload(zero_with_t0, spec=spec)

    passing_record = _shadow_record_fixture("USDJPY", evidence=passing_evidence)
    primary_id_map = {(row[0], row[1]): row[2] for row in primary_arms}
    passing_record["candidate_bundle_id"] = primary_id_map[("USDJPY", "C")]
    passing_record["v1_bundle_id"] = primary_id_map[("USDJPY", "A")]
    passing_id = passing_record["candidate_book_id"]
    with tempfile.TemporaryDirectory(prefix="m15-deferred-shadow-contract-") as raw:
        evidence_root = Path(raw)
        results_json = evidence_root / "results" / "json"
        shadow_root = evidence_root / "shadow"
        results_json.mkdir(parents=True)
        shadow_root.mkdir()
        parity_path = (
            evidence_root / "logs" / "m15_book_refresh" / "synthetic" / "parity.json"
        )
        parity_path.parent.mkdir(parents=True)
        _write_read_only_json(
            parity_path,
            {
                "schema": "m15-book-refresh-parity/v1",
                "prereg_id": prereg_id,
                "implementation_git_sha": implementation_sha,
                "pairs": {
                    "USDJPY": {
                        "rows": 80,
                        "provider_parity": passing_provider,
                    }
                },
            },
        )
        parity_sha = sha256(parity_path)
        common["provider_parity_result"] = parity_path.relative_to(
            evidence_root
        ).as_posix()
        common["provider_parity_result_sha256"] = parity_sha
        passing_evidence["provider_parity_result_sha256"] = parity_sha
        passing_record["provider_evidence"] = passing_evidence
        joint_path = (
            results_json / f"m15_book_refresh_{run_id}_joint_replay_result.json"
        )
        _write_read_only_json(joint_path, {"schema": "synthetic-joint/v1"})
        coverage = float(spec["pairs"]["USDJPY"]["calibration_target_coverage"])
        ordered_seal = {
            "schema": "m15-book-refresh-ordered-S-shadow-seal/v1",
            "prereg_id": prereg_id,
            "run_id": run_id,
            "implementation_git_sha": implementation_sha,
            "evaluator_id": evaluator,
            "S_shadow": [passing_id],
            "k_shadow": 1,
            "ordered_shadow_pairs": ["USDJPY"],
            "ordered_coverages": [{"pair": "USDJPY", "coverage": coverage}],
            "ordered_candidates": [
                {
                    "pair": "USDJPY",
                    "candidate_book_id": passing_id,
                    "candidate_bundle_id": passing_record["candidate_bundle_id"],
                    "v1_book_id": passing_record["v1_book_id"],
                    "v1_bundle_id": passing_record["v1_bundle_id"],
                    "coverage": coverage,
                }
            ],
            "joint_result_sha256": sha256(joint_path),
            "run_binding_sha256": completed["run_binding_sha256"],
            "provider_parity_result_sha256": parity_sha,
            "semantic_replay_outcomes_already_accessed": True,
        }
        ordered_path = _write_read_only_json(
            shadow_root / "ordered-S-shadow.json", ordered_seal
        )
        from scipy.stats import beta

        deferred_count = 0
        deferred_cp = float(
            beta.ppf(
                0.95,
                deferred_count + 1,
                spec["null_calibration"]["campaigns_per_cell"] - deferred_count,
            )
        )
        deferred_bundle_ids = [passing_record["candidate_bundle_id"]]
        deferred_seed_binding = refresh._expected_deferred_shadow_seed_binding(
            ["USDJPY"], deferred_bundle_ids
        )
        deferred_cells = [
            {
                "schema": "m15-book-refresh-null-calibration-cell/v1",
                "fixture": fixture,
                "helper": "shadow",
                "k_shadow": 1,
                "ordered_shadow_pairs": ["USDJPY"],
                "accuracy_null": 0.541,
                "binding": True,
                "calibration_role": "binding_deferred_shadow",
                "seed_binding": deferred_seed_binding,
                "campaign_count": spec["null_calibration"]["campaigns_per_cell"],
                "bootstrap_replicates": spec["null_calibration"][
                    "bootstrap_replicates"
                ],
                "block_lengths": spec["null_calibration"]["block_lengths"],
                "false_positive_count": deferred_count,
                "false_positive_limit": spec["null_calibration"][
                    "false_positive_limit_inclusive"
                ],
                "clopper_pearson_upper_95": deferred_cp,
                "cell_passed": True,
                "accepted": True,
                "deterministic_repeat": True,
                "campaign_digest": hashlib.sha256(
                    f"synthetic-deferred/{fixture}/USDJPY".encode()
                ).hexdigest(),
            }
            for fixture in spec["null_calibration"]["binding_deferred_shadow"][
                "fixture_order"
            ]
        ]
        deferred_payload = {
            "schema": (
                "m15-book-refresh-deferred-shadow-null-calibration-result/v1"
            ),
            "prereg_id": prereg_id,
            "run_id": run_id,
            "implementation_git_sha": implementation_sha,
            "spec_sha256": refresh.sha256_file(refresh.SPEC_PATH),
            "runner_sha256": refresh.sha256_file(Path(refresh.__file__)),
            "statistics_sha256": refresh.sha256_file(Path(stats.__file__)),
            "ordered_S_shadow_seal_sha256": refresh.sha256_bytes(
                refresh.canonical_bytes(ordered_seal)
            ),
            "ordered_shadow_pairs": ["USDJPY"],
            "ordered_shadow_bundle_ids": deferred_bundle_ids,
            "ordered_coverages": ordered_seal["ordered_coverages"],
            "seed_binding": deferred_seed_binding,
            "contract": spec["null_calibration"]["binding_deferred_shadow"],
            "campaign_count": spec["null_calibration"]["campaigns_per_cell"],
            "bootstrap_replicates": spec["null_calibration"][
                "bootstrap_replicates"
            ],
            "block_lengths": spec["null_calibration"]["block_lengths"],
            "cells": deferred_cells,
            "cells_sha256": refresh.length_prefixed_digest(
                "m15-book-refresh/deferred-shadow-null-cells/v1",
                [refresh.canonical_bytes(deferred_cells)],
            ),
            "accepted": True,
            "semantic_replay_outcomes_already_accessed": True,
            "synthetic_calibration_read_replay_outcomes": False,
        }
        deferred_path = _write_read_only_json(
            shadow_root / "deferred-shadow-null.json", deferred_payload
        )
        accepted_deferred = refresh._validate_deferred_shadow_null_artifact(
            deferred_path,
            prereg_id=prereg_id,
            run_id=run_id,
            implementation_git_sha=implementation_sha,
            ordered_shadow_seal=ordered_seal,
        )
        check(accepted_deferred["accepted"], "valid deferred shadow null rejected")
        for count, should_accept in ((32, True), (33, False)):
            boundary_deferred = json.loads(json.dumps(deferred_payload))
            boundary_cp = float(
                beta.ppf(
                    0.95,
                    count + 1,
                    spec["null_calibration"]["campaigns_per_cell"] - count,
                )
            )
            for cell in boundary_deferred["cells"]:
                cell["false_positive_count"] = count
                cell["clopper_pearson_upper_95"] = boundary_cp
                cell["cell_passed"] = should_accept
                cell["accepted"] = should_accept
            boundary_deferred["cells_sha256"] = refresh.length_prefixed_digest(
                "m15-book-refresh/deferred-shadow-null-cells/v1",
                [refresh.canonical_bytes(boundary_deferred["cells"])],
            )
            boundary_deferred["accepted"] = should_accept
            boundary_deferred_path = _write_read_only_json(
                shadow_root / f"deferred-boundary-{count}.json",
                boundary_deferred,
            )
            if should_accept:
                refresh._validate_deferred_shadow_null_artifact(
                    boundary_deferred_path,
                    prereg_id=prereg_id,
                    run_id=run_id,
                    implementation_git_sha=implementation_sha,
                    ordered_shadow_seal=ordered_seal,
                )
            else:
                with expect_raises(refresh.RefreshError, "blocked T0"):
                    refresh._validate_deferred_shadow_null_artifact(
                        boundary_deferred_path,
                        prereg_id=prereg_id,
                        run_id=run_id,
                        implementation_git_sha=implementation_sha,
                        ordered_shadow_seal=ordered_seal,
                    )
        wrong_accuracy = json.loads(json.dumps(deferred_payload))
        wrong_accuracy["cells"][0]["accuracy_null"] = 0.5
        wrong_accuracy["cells_sha256"] = refresh.length_prefixed_digest(
            "m15-book-refresh/deferred-shadow-null-cells/v1",
            [refresh.canonical_bytes(wrong_accuracy["cells"])],
        )
        wrong_accuracy_path = _write_read_only_json(
            shadow_root / "wrong-accuracy.json", wrong_accuracy
        )
        with expect_raises(refresh.RefreshError, "cell differs"):
            refresh._validate_deferred_shadow_null_artifact(
                wrong_accuracy_path,
                prereg_id=prereg_id,
                run_id=run_id,
                implementation_git_sha=implementation_sha,
                ordered_shadow_seal=ordered_seal,
            )

        # The artifact's pair-aligned bundle order is authenticated against
        # the already sealed survivor authority, not merely self-consistent.
        second_bundle_id = hashlib.sha256(b"candidate-C/GBPUSD").hexdigest()
        ordered_seal_two = json.loads(json.dumps(ordered_seal))
        ordered_seal_two["ordered_shadow_pairs"] = ["USDJPY", "GBPUSD"]
        ordered_seal_two["ordered_coverages"] = [
            {"pair": "USDJPY", "coverage": coverage},
            {
                "pair": "GBPUSD",
                "coverage": float(
                    spec["pairs"]["GBPUSD"]["calibration_target_coverage"]
                ),
            },
        ]
        ordered_seal_two["ordered_candidates"].append(
            {
                "pair": "GBPUSD",
                "candidate_book_id": spec["pairs"]["GBPUSD"][
                    "candidate_book_id"
                ],
                "candidate_bundle_id": second_bundle_id,
                "v1_book_id": spec["pairs"]["GBPUSD"]["book_id_A"],
                "v1_bundle_id": hashlib.sha256(b"v1/GBPUSD").hexdigest(),
                "coverage": float(
                    spec["pairs"]["GBPUSD"]["calibration_target_coverage"]
                ),
            }
        )
        bundle_ids_two = [deferred_bundle_ids[0], second_bundle_id]
        seed_two = refresh._expected_deferred_shadow_seed_binding(
            ordered_seal_two["ordered_shadow_pairs"], bundle_ids_two
        )
        deferred_two = json.loads(json.dumps(deferred_payload))
        deferred_two["ordered_S_shadow_seal_sha256"] = refresh.sha256_bytes(
            refresh.canonical_bytes(ordered_seal_two)
        )
        deferred_two["ordered_shadow_pairs"] = ordered_seal_two[
            "ordered_shadow_pairs"
        ]
        deferred_two["ordered_shadow_bundle_ids"] = bundle_ids_two
        deferred_two["ordered_coverages"] = ordered_seal_two["ordered_coverages"]
        deferred_two["seed_binding"] = seed_two
        for cell in deferred_two["cells"]:
            cell["k_shadow"] = 2
            cell["ordered_shadow_pairs"] = ordered_seal_two[
                "ordered_shadow_pairs"
            ]
            cell["seed_binding"] = seed_two
        deferred_two["cells_sha256"] = refresh.length_prefixed_digest(
            "m15-book-refresh/deferred-shadow-null-cells/v1",
            [refresh.canonical_bytes(deferred_two["cells"])],
        )
        deferred_two_path = _write_read_only_json(
            shadow_root / "two-candidate-deferred.json", deferred_two
        )
        refresh._validate_deferred_shadow_null_artifact(
            deferred_two_path,
            prereg_id=prereg_id,
            run_id=run_id,
            implementation_git_sha=implementation_sha,
            ordered_shadow_seal=ordered_seal_two,
        )
        reordered_ids = json.loads(json.dumps(deferred_two))
        reordered_ids["ordered_shadow_bundle_ids"].reverse()
        reordered_seed = refresh._expected_deferred_shadow_seed_binding(
            reordered_ids["ordered_shadow_pairs"],
            reordered_ids["ordered_shadow_bundle_ids"],
        )
        reordered_ids["seed_binding"] = reordered_seed
        for cell in reordered_ids["cells"]:
            cell["seed_binding"] = reordered_seed
        reordered_ids["cells_sha256"] = refresh.length_prefixed_digest(
            "m15-book-refresh/deferred-shadow-null-cells/v1",
            [refresh.canonical_bytes(reordered_ids["cells"])],
        )
        reordered_ids_path = _write_read_only_json(
            shadow_root / "reordered-bundle-ids.json", reordered_ids
        )
        with expect_raises(refresh.RefreshError, "identity/contract"):
            refresh._validate_deferred_shadow_null_artifact(
                reordered_ids_path,
                prereg_id=prereg_id,
                run_id=run_id,
                implementation_git_sha=implementation_sha,
                ordered_shadow_seal=ordered_seal_two,
            )

        positive = {
            "schema": spec["shadow"]["schemas"]["positive_eligible"],
            "status": spec["shadow"]["statuses"]["positive_eligible_handoff"],
            **common,
            "S": [passing_id],
            "S_records": [passing_record],
            "S_shadow": [passing_id],
            "k_shadow": 1,
            "eligible_candidates": [passing_record],
            "exclusions": {},
            "T0": None,
            "T0_contract": spec["shadow"]["positive_eligible_handoff"][
                "T0_contract"
            ],
            "no_backfill": True,
            "endpoint_count": spec["shadow"]["endpoint_count"],
            "fixed_family_size": 3,
            "endpoint_order_per_candidate": spec["shadow"][
                "endpoint_order_per_candidate"
            ],
            "endpoint_formulas": spec["shadow"]["endpoint_formulas"],
            "prospective_capture": spec["shadow"]["prospective_capture"],
            "fixed_look": spec["shadow"]["fixed_look"],
            "statistics": spec["shadow"]["statistics"],
            "promotion_toward_activation": spec["shadow"][
                "promotion_toward_activation"
            ],
            "activation_prerequisite": spec["shadow"]["activation_prerequisite"],
            "family_reset_and_relook": spec["shadow"]["family_reset_and_relook"],
            "ordered_S_shadow_seal": {
                "artifact": ordered_path.relative_to(evidence_root).as_posix(),
                "sha256": sha256(ordered_path),
            },
            "deferred_shadow_null_calibration": {
                "artifact": deferred_path.relative_to(evidence_root).as_posix(),
                "sha256": sha256(deferred_path),
                "accepted": True,
                "cells": 4,
                "accuracy_null": 0.541,
                "false_positive_limit_inclusive": 32,
            },
        }
        with mock.patch.object(refresh, "REPO_ROOT", evidence_root), mock.patch.object(
            refresh, "RESULTS_JSON", results_json
        ), mock.patch.object(
            refresh,
            "repo_relative",
            side_effect=lambda path, **_kwargs: Path(path)
            .resolve(strict=True)
            .relative_to(evidence_root.resolve(strict=True))
            .as_posix(),
        ), mock.patch.object(
            refresh,
            "_validated_shadow_provider_evidence",
            return_value=(passing_evidence, True, None),
        ):
            refresh._validate_shadow_handoff_payload(positive, spec=spec)
            forged_provider_copy = json.loads(json.dumps(positive))
            forged_live = forged_provider_copy["S_records"][0][
                "provider_evidence"
            ]["live_feature_builder_parity"]
            forged_live.update(provider_rows=1, compared_rows=1)
            with expect_raises(refresh.RefreshError, "provider copy differs"):
                refresh._validate_shadow_handoff_payload(
                    forged_provider_copy, spec=spec
                )
            wrong_family = dict(positive, fixed_family_size=6)
            with expect_raises(refresh.RefreshError, "positive prospective"):
                refresh._validate_shadow_handoff_payload(wrong_family, spec=spec)

    blocked_provider = _provider_fixture(
        "EURUSD", implementation_git_sha=implementation_sha, passed=False
    )
    blocked_provider["feature_schema_parity"] = True
    blocked_evidence, blocked_qualified, blocked_reason = (
        refresh._validated_shadow_provider_evidence(
            "EURUSD",
            blocked_provider,
            spec=spec,
            implementation_git_sha=implementation_sha,
            parity_sha256="e" * 64,
            evaluator_id_value=evaluator,
            expected_compared_rows=80,
        )
    )
    check(not blocked_qualified, "EURUSD unavailable live OF provider qualified")
    with tempfile.TemporaryDirectory(prefix="m15-zero-provider-contract-") as raw:
        evidence_root = Path(raw)
        for source in blocked_provider["provider_source_files"]:
            copied_source = evidence_root / source["path"]
            copied_source.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(REPO_ROOT / source["path"], copied_source)
            copied_source.chmod(0o444)
        parity_path = (
            evidence_root / "logs" / "m15_book_refresh" / "synthetic" / "parity.json"
        )
        parity_path.parent.mkdir(parents=True)
        _write_read_only_json(
            parity_path,
            {
                "schema": "m15-book-refresh-parity/v1",
                "prereg_id": prereg_id,
                "implementation_git_sha": implementation_sha,
                "pairs": {
                    "EURUSD": {
                        "rows": 80,
                        "provider_parity": blocked_provider,
                    }
                },
            },
        )
        blocked_parity_sha = sha256(parity_path)
        blocked_evidence["provider_parity_result_sha256"] = blocked_parity_sha
        blocked_record = _shadow_record_fixture("EURUSD", evidence=blocked_evidence)
        blocked_record["candidate_bundle_id"] = primary_id_map[("EURUSD", "C")]
        blocked_record["v1_bundle_id"] = primary_id_map[("EURUSD", "A")]
        blocked_id = blocked_record["candidate_book_id"]
        nonempty_zero = {
            **zero,
            "provider_parity_result": parity_path.relative_to(
                evidence_root
            ).as_posix(),
            "provider_parity_result_sha256": blocked_parity_sha,
            "status": spec["shadow"]["statuses"][
                "survivors_but_zero_provider_eligible"
            ],
            "S": [blocked_id],
            "S_records": [blocked_record],
            "exclusions": {
                blocked_id: {**blocked_record, "reason": blocked_reason},
            },
        }
        with mock.patch.object(refresh, "REPO_ROOT", evidence_root), mock.patch.object(
            refresh,
            "repo_relative",
            side_effect=lambda path, **_kwargs: Path(path)
            .resolve(strict=True)
            .relative_to(evidence_root.resolve(strict=True))
            .as_posix(),
        ):
            refresh._validate_shadow_handoff_payload(nonempty_zero, spec=spec)


def test_null_surface_partition() -> None:
    """Keep current, mechanics-only, and deferred shadow null gates disjoint."""

    bounded = {
        "campaign_count": 1,
        "bootstrap_replicates": 8,
        "block_lengths": (1, 2, 3),
        "verify_repeat": True,
    }
    with expect_raises(stats.StatisticsInputError, "dedicated mechanics or deferred"):
        stats.run_null_calibration_cell(
            "F0", helper="shadow", k_shadow=1, **bounded
        )

    mechanics = stats.run_shadow_null_mechanics_suite(**bounded)
    equal(len(mechanics), 28, "nonbinding prefix mechanics cell count")
    expected_mechanics = [
        (fixture.name, stats.FEATURE_SOURCE_PAIR_ORDER[:k_shadow])
        for fixture in stats.NULL_FIXTURES
        for k_shadow in range(1, len(stats.FEATURE_SOURCE_PAIR_ORDER) + 1)
    ]
    equal(
        [
            (cell.fixture, cell.ordered_shadow_pairs)
            for cell in mechanics
        ],
        expected_mechanics,
        "nonbinding prefix mechanics order",
    )
    check(
        all(
            cell.helper == "shadow"
            and cell.k_shadow == len(cell.ordered_shadow_pairs)
            and cell.accuracy_null == 0.5
            and cell.binding is False
            and cell.calibration_role == "nonbinding_shadow_mechanics"
            and cell.accepted is False
            and cell.deterministic_repeat is True
            for cell in mechanics
        ),
        "prefix mechanics acquired binding acceptance semantics",
    )
    binding_keys = set(refresh._expected_null_cell_keys(refresh.load_spec()))
    mechanics_keys = {
        (cell.fixture, cell.helper, cell.k_shadow) for cell in mechanics
    }
    check(
        binding_keys.isdisjoint(mechanics_keys),
        "prefix mechanics leaked into the current binding null artifact",
    )

    ordered_shadow_pairs = ("USDJPY", "GBPUSD", "NZDUSD")
    ordered_shadow_bundle_ids = tuple(
        hashlib.sha256(f"candidate-C/{pair}".encode()).hexdigest()
        for pair in ordered_shadow_pairs
    )
    deferred = stats.run_deferred_shadow_null_calibration_suite(
        ordered_shadow_pairs=ordered_shadow_pairs,
        ordered_shadow_bundle_ids=ordered_shadow_bundle_ids,
        **bounded,
    )
    equal(len(deferred), 4, "deferred .541 shadow cell count")
    equal(
        [cell.fixture for cell in deferred],
        [fixture.name for fixture in stats.NULL_FIXTURES],
        "deferred .541 fixture order",
    )
    check(
        all(
            cell.helper == "shadow"
            and cell.k_shadow == len(ordered_shadow_pairs)
            and cell.ordered_shadow_pairs == ordered_shadow_pairs
            and cell.accuracy_null == 0.541
            and cell.binding is True
            and cell.calibration_role == "binding_deferred_shadow"
            and cell.deterministic_repeat is True
            and cell.seed_binding is not None
            and cell.seed_binding["identity_payload"]
            == [
                {"pair": pair, "candidate_C_bundle_id": bundle_id}
                for pair, bundle_id in zip(
                    ordered_shadow_pairs, ordered_shadow_bundle_ids
                )
            ]
            for cell in deferred
        ),
        "deferred .541 calibration did not bind the exact ordered S_shadow",
    )
    changed_bundle_ids = (
        hashlib.sha256(b"changed-candidate-C/USDJPY").hexdigest(),
        *ordered_shadow_bundle_ids[1:],
    )
    changed = stats.run_deferred_shadow_null_calibration_suite(
        ordered_shadow_pairs=ordered_shadow_pairs,
        ordered_shadow_bundle_ids=changed_bundle_ids,
        **bounded,
    )
    check(
        deferred[0].seed_binding["identity_sha256"]
        != changed[0].seed_binding["identity_sha256"]
        and deferred[0].campaign_digest != changed[0].campaign_digest,
        "candidate-C bundle change did not change deferred seed/result identity",
    )
    for malformed, fragment in (
        (("USDJPY", "USDJPY"), "duplicate"),
        (("GBPUSD", "USDJPY"), "sealed pair order"),
        (("USDJPY", "USDCAD"), "unknown pairs"),
    ):
        with expect_raises(stats.StatisticsInputError, fragment):
            stats.run_deferred_shadow_null_calibration_suite(
                ordered_shadow_pairs=malformed,
                ordered_shadow_bundle_ids=ordered_shadow_bundle_ids[: len(malformed)],
                **bounded,
            )
    for malformed_ids, fragment in (
        ((ordered_shadow_bundle_ids[0],), "pair-aligned"),
        ((ordered_shadow_bundle_ids[0],) * 3, "unique"),
        (("F" * 64, *ordered_shadow_bundle_ids[1:]), "lowercase SHA-256"),
    ):
        with expect_raises(stats.StatisticsInputError, fragment):
            stats.run_deferred_shadow_null_calibration_suite(
                ordered_shadow_pairs=ordered_shadow_pairs,
                ordered_shadow_bundle_ids=malformed_ids,
                **bounded,
            )


def test_null_calibration(*, bounded: bool) -> None:
    if bounded:
        campaign_count = 1
        bootstrap_replicates = 8
        print(
            "[test] bounded null mode: 1 campaign/cell, 8 draws, repeat=true; "
            "NOT an acceptance artifact",
            flush=True,
        )
    else:
        campaign_count = 400
        bootstrap_replicates = 1_000
        equal(stats.NULL_CAMPAIGNS_PER_CELL, 400, "sealed null campaign count")
        equal(stats.NULL_BOOTSTRAP_REPLICATES, 1_000, "sealed null bootstrap draws")
        equal(stats.NULL_BLOCK_LENGTHS, (1, 2, 3), "sealed null block lengths")
        # Fail fast on the first sealed cell. Once it accepts, the complete
        # public eight-cell suite below remains the acceptance authority.
        preflight = stats.run_null_calibration_cell(
            "F0",
            helper="main",
            campaign_count=campaign_count,
            bootstrap_replicates=bootstrap_replicates,
            block_lengths=(1, 2, 3),
            verify_repeat=True,
        )
        check(
            preflight.accepted,
            "exact null calibration is blocked at F0/main before the remaining "
            f"7 cells: {preflight.as_dict()}",
        )
    cells = stats.run_null_calibration_suite(
        campaign_count=campaign_count,
        bootstrap_replicates=bootstrap_replicates,
        block_lengths=(1, 2, 3),
        verify_repeat=True,
    )
    equal(len(cells), 8, "null suite cell count")
    expected_order = [
        (fixture.name, helper, k_shadow)
        for fixture in stats.NULL_FIXTURES
        for helper, k_shadow in (("main", 0), ("control", 0))
    ]
    equal(
        [(cell.fixture, cell.helper, cell.k_shadow) for cell in cells],
        expected_order,
        "null suite sealed cell order",
    )
    check(all(cell.deterministic_repeat for cell in cells), "null suite repeatability failed")
    check(
        all(cell.campaign_count == campaign_count for cell in cells),
        "null suite campaign count drift",
    )
    check(
        all(cell.bootstrap_replicates == bootstrap_replicates for cell in cells),
        "null suite bootstrap draw drift",
    )
    check(all(cell.block_lengths == (1, 2, 3) for cell in cells), "null block drift")
    if not bounded:
        failed = [cell.as_dict() for cell in cells if not cell.accepted]
        check(not failed, f"exact null calibration rejected cells: {failed}")
        check(
            all(
                cell.false_positive_count
                <= refresh.load_spec()["null_calibration"]["false_positive_limit"]
                for cell in cells
            ),
            "exact null false-positive count exceeded sealed limit",
        )
    summary = ", ".join(
        f"{cell.fixture}/{cell.helper}{cell.k_shadow or ''}={cell.false_positive_count}"
        for cell in cells
    )
    print(f"[test] null false positives: {summary}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bounded-null",
        action="store_true",
        help="development-only 1-campaign/8-draw null suite; default is exact acceptance suite",
    )
    parser.add_argument(
        "--skip-null",
        action="store_true",
        help="run focused non-null sections only (not a complete evidence gate)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.bounded_null and args.skip_null:
        raise GateFailure("--bounded-null and --skip-null are mutually exclusive")
    started = time.monotonic()
    timings = {
        "identity_goldens": run_section("identity goldens", test_identity_goldens),
        "target_source_contract": run_section(
            "six-target/seven-source contract",
            test_target_and_feature_source_contract,
        ),
        "loaded_book_batch": run_section(
            "LoadedBook batch/scalar/component parity",
            test_loaded_book_batch_behavior,
        ),
        "canonical_helpers": run_section(
            "canonical helpers/clocks/DST/cutoff/ties/scheduling",
            test_canonical_helpers_and_boundaries,
        ),
        "adapters": run_section(
            "cap/adapters/import safety", test_adapters_cap_and_import_safety
        ),
        "direct_builder_parity": run_section(
            "direct incumbent-builder parity", test_direct_incumbent_builder_parity
        ),
        "feature_view_routing": run_section(
            "sealed feature-view routing", test_feature_view_routing
        ),
        "runner_replay_hardening": run_section(
            "phase-zero/identity/one-look/window/INVALID hardening",
            test_runner_replay_hardening,
        ),
        "post_replay_comparator_fences": run_section(
            "post-replay comparator-A mutation fences",
            test_post_replay_comparator_fences,
        ),
        "live_provider_parity": run_section(
            "public LiveFeatureBuilder parity",
            test_live_feature_builder_provider_parity,
        ),
        "statistics": run_section("statistics/status/sparsity", test_statistics_and_statuses),
        "publication": run_section(
            "lifecycle/receipt/external-lock/publication recovery",
            test_lifecycle_and_publication,
        ),
        "phases": run_section("phase resume/access fence", test_phase_resume_and_access_fence),
        "fit_orchestration": run_section(
            "fit policy/checkpoint/control/phase integrity",
            test_fit_orchestration_integrity,
        ),
        "frozen_fit_contract": run_section(
            "frozen fit-contract authentication",
            test_frozen_fit_contract_authentication,
        ),
        "integrity": run_section(
            "artifact authentication/CLI/shadow handoff",
            test_integrity_artifacts_cli_and_shadow,
        ),
        "null_surface_partition": run_section(
            "current/mechanics/deferred null partition",
            test_null_surface_partition,
        ),
    }
    if not args.skip_null:
        timings["null_calibration"] = run_section(
            "bounded null calibration" if args.bounded_null else "EXACT 8-cell null calibration",
            lambda: test_null_calibration(bounded=args.bounded_null),
        )
    elapsed = time.monotonic() - started
    mode = "INCOMPLETE_SKIP_NULL" if args.skip_null else (
        "BOUNDED_DEVELOPMENT" if args.bounded_null else "FULL_ACCEPTANCE"
    )
    print(
        f"[test] PASS mode={mode} total={elapsed:.2f}s "
        f"timings={json.dumps(timings, sort_keys=True)} "
        "synthetic_pre_April_only=true replay_semantics_read=false real_books_mutated=false",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
