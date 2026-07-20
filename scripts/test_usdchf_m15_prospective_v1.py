#!/usr/bin/env python3
"""Synthetic focused gate for the issue-#19 prospective USDCHF observer.

All mutable state is confined to temporary directories.  This gate never runs
the live preflight command, starts collection, opens the live stores, analyzes
a real packet, connects to Deriv, or creates a trade request.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import signal
import sqlite3
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator, Mapping
from unittest import mock
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

import evidence_store as evidence
import usdchf_m15_prospective_v1 as prospective


REPO_ROOT = Path(__file__).resolve().parents[1]
NY = ZoneInfo("America/New_York")
UTC = timezone.utc


class GateFailure(AssertionError):
    pass


def check(condition: Any, message: str) -> None:
    if not bool(condition):
        raise GateFailure(message)


def equal(actual: Any, expected: Any, message: str) -> None:
    if actual != expected:
        raise GateFailure(f"{message}: actual={actual!r} expected={expected!r}")


@contextmanager
def expect_raises(
    error: type[BaseException], contains: str | None = None
) -> Iterator[BaseException]:
    try:
        yield GateFailure("unreachable")
    except error as exc:
        if contains is not None and contains not in str(exc):
            raise GateFailure(
                f"exception did not contain {contains!r}: {type(exc).__name__}: {exc}"
            ) from exc
        yield_value = exc
    else:
        raise GateFailure(f"expected {error.__name__}")
    del yield_value


def run_section(name: str, function: Any) -> float:
    print(f"[test] START {name}", flush=True)
    started = time.monotonic()
    function()
    elapsed = time.monotonic() - started
    print(f"[test] PASS {name} ({elapsed:.2f}s)", flush=True)
    return elapsed


def _epoch(local: datetime) -> int:
    return int(local.astimezone(UTC).timestamp())


def _spec() -> dict[str, Any]:
    return json.loads((REPO_ROOT / prospective.SPEC_PATH).read_text(encoding="utf-8"))


def _preflight_fixture() -> dict[str, Any]:
    selected_date = datetime(2026, 7, 16, tzinfo=NY).date()
    session_start = _epoch(datetime(2026, 7, 16, 8, 0, tzinfo=NY))
    session_end = _epoch(datetime(2026, 7, 16, 16, 50, tzinfo=NY)) - 1
    session_seconds = session_end - session_start + 1
    admitted = session_seconds - 1
    feature_epoch = int(
        datetime(2026, 7, 17, 20, 33, tzinfo=UTC).timestamp()
    )
    decision_epoch = feature_epoch + 60
    return {
        "schema": prospective.PREFLIGHT_SCHEMA,
        "pair": prospective.PAIR,
        "host": os.uname().nodename,
        "spent_session_date_ny": selected_date.isoformat(),
        "generated_utc": "2026-07-17T20:34:20Z",
        "provider": {
            "endpoint_class": "public_Deriv_market_stream_store",
            "candle_store": "deriv_data/candles_1m_daemon",
            "tick_store": "deriv_data/ticks_1s/_pages/USDCHF",
            "tick_descriptor": {
                "source": "deriv_data/ticks_1s/_pages/USDCHF",
                "progress_path": "deriv_data/ticks_1s/USDCHF_progress.json",
                "progress_sha256": "1" * 64,
                "last_persisted_epoch": 1_768_000_000,
                "shards_seen": 1,
                "shard_metadata_sha256": "2" * 64,
            },
        },
        "tick_provenance": {
            "session_start_epoch": session_start,
            "session_end_epoch": session_end,
            "rows": session_seconds,
            "admitted_live_rows": admitted,
            "recovered_or_malformed_rows_rejected": 1,
            "finite_second_coverage": admitted / session_seconds,
            "admission_rule": "finite_positive_quote_bid_ask",
        },
        "scoring_parity": {
            "feature_timestamp_utc": "2026-07-17T20:33:00Z",
            "decision_close_utc": "2026-07-17T20:34:00Z",
            "decision_close_epoch": decision_epoch,
            "feature_sha256": "3" * 64,
            "feature_dtype": "float32",
            "feature_count": 337,
            "row_source": "synthetic-public-store",
            "warnings": [],
            "candidate": {
                "book_id": "USDCHF.m15ny_xpair_seedens.r202605.v1",
                "content_id": "candidate-content",
                "probability": 0.6,
                "confidence": abs(0.6 - 0.5),
                "threshold": 0.04310172144482041,
                "direction": "UP",
                "confidence_gate_passed": True,
                "structural_gate_passed": True,
                "gate_passed": True,
                "gate_reasons": [],
            },
            "incumbent": {
                "book_id": "USDCHF.m15ny_xpair_seedens.v1",
                "content_id": "incumbent-content",
                "probability": 0.4,
                "confidence": abs(0.4 - 0.5),
                "threshold": 0.16933929389708485,
                "direction": "DOWN",
                "confidence_gate_passed": False,
                "structural_gate_passed": True,
                "gate_passed": False,
                "gate_reasons": ["confidence"],
            },
        },
        "outcome_fields_present": False,
        "activation": False,
    }


def _score_fixture(
    *, candidate_probability: float = 0.6, incumbent_probability: float = 0.4
) -> dict[str, Any]:
    def policy(book_id: str, probability: float) -> dict[str, Any]:
        return {
            "book_id": book_id,
            "content_id": f"{book_id}-content",
            "probability": probability,
            "confidence": abs(probability - 0.5),
            "threshold": 0.01,
            "direction": "UP" if probability >= 0.5 else "DOWN",
            "confidence_gate_passed": True,
            "structural_gate_passed": True,
            "gate_passed": True,
            "gate_reasons": [],
        }

    return {
        "feature_timestamp_utc": "2026-07-17T12:00:00Z",
        "decision_close_utc": "2026-07-17T12:01:00Z",
        "decision_close_epoch": 1_768_000_060,
        "feature_sha256": "4" * 64,
        "feature_dtype": "float32",
        "feature_count": 337,
        "row_source": "synthetic-public-store",
        "warnings": [],
        "candidate": policy(
            "USDCHF.m15ny_xpair_seedens.r202605.v1", candidate_probability
        ),
        "incumbent": policy(
            "USDCHF.m15ny_xpair_seedens.v1", incumbent_probability
        ),
    }


def _decision(
    prereg_id: str,
    decision_epoch: int,
    *,
    candidate_probability: float = 0.6,
    incumbent_probability: float = 0.4,
    common: bool = True,
) -> dict[str, Any]:
    return prospective._decision_payload(
        prereg_id=prereg_id,
        decision_epoch=decision_epoch,
        recorded_epoch=decision_epoch + 20,
        provider={"class": "public_Deriv_persisted_store"},
        scoring=(
            _score_fixture(
                candidate_probability=candidate_probability,
                incumbent_probability=incumbent_probability,
            )
            if common
            else None
        ),
        exclusion_reason="" if common else "UNSCORED_SYNTHETIC_GAP",
    )


def _prereg_fixture(t0_epoch: int) -> evidence.EvidenceEnvelope:
    return evidence.EvidenceEnvelope.create(
        kind="research.usdchf_m15_prospective.prereg/v1",
        payload={
            "t0": {
                "utc": prospective._canonical_utc(datetime.fromtimestamp(t0_epoch, UTC)),
                "ny": datetime.fromtimestamp(t0_epoch, UTC)
                .astimezone(NY)
                .isoformat(timespec="seconds"),
                "epoch": t0_epoch,
            }
        },
    )


class _FakeBook:
    def __init__(
        self,
        book_id: str,
        probability: float,
        *,
        batch_probability: float | None = None,
        batch_gate: bool | None = None,
    ) -> None:
        self.book_id = book_id
        self.content_id = f"{book_id}-content"
        self.feature_cols = [f"f{index:03d}" for index in range(337)]
        self.conf_thr = 0.05
        self._probability = probability
        self._batch_probability = (
            probability if batch_probability is None else batch_probability
        )
        self._batch_gate = (
            abs(probability - 0.5) >= self.conf_thr
            if batch_gate is None
            else batch_gate
        )

    def score(self, row: pd.Series) -> Any:
        del row
        probability = self._probability
        gate = abs(probability - 0.5) >= self.conf_thr
        return SimpleNamespace(
            proba=probability,
            confidence=abs(probability - 0.5),
            threshold=self.conf_thr,
            direction="UP" if probability >= 0.5 else "DOWN",
            gate_passed=gate,
            gate_reasons=() if gate else ("confidence",),
        )

    def predict_probabilities(self, frame: pd.DataFrame) -> np.ndarray:
        equal(list(frame.columns), self.feature_cols, "batch feature order")
        equal(frame.dtypes.nunique(), 1, "batch feature dtype consistency")
        return np.asarray([self._batch_probability], dtype="float64")

    def gate_components(self, frame: pd.DataFrame, probabilities: Any) -> Any:
        del frame, probabilities
        return SimpleNamespace(
            confidence_passed=np.asarray([self._batch_gate], dtype=bool),
            structural_passed=np.asarray([True], dtype=bool),
            gate_passed=np.asarray([self._batch_gate], dtype=bool),
        )


def test_public_surface_and_buy_incapability() -> None:
    source = (REPO_ROOT / prospective.SCRIPT_PATH).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    forbidden_modules = {
        "deriv_trade_queue",
        "deriv_trade_executor",
        "deriv_runtime_supervisor",
        "deriv_demo_executor",
    }
    check(not (imported & forbidden_modules), "observer imports an execution module")

    called_names = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    check(
        not {"buy", "purchase", "proposal", "enqueue_trade"} & called_names,
        "observer exposes a direct execution call",
    )
    socket_calls = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "socket"
    }
    equal(socket_calls, set(), "observer socket capability")
    check("socket" not in imported, "observer imports the socket module")
    check("os.uname" in source, "observer lacks the narrow host-identity call")
    with expect_raises(prospective.ProspectiveError, "store-only"):
        prospective.StoreOnlySentinel().request_history()

    service = (REPO_ROOT / prospective.SERVICE_PATH).read_text(encoding="utf-8")
    check("EnvironmentFile=" not in service, "service loads an environment file")
    check("RestrictAddressFamilies=AF_UNIX" in service, "service is not socket-isolated")
    check("ExecStart=" in service and " collect" in service, "service is not collector-only")
    for name in (*prospective.FORBIDDEN_ENVIRONMENT, "MX_HOR"):
        check(name in service, f"service does not unset {name}")
    for capability in (
        "NoNewPrivileges=true",
        "ProtectSystem=strict",
        "ProtectHome=read-only",
        "ReadOnlyPaths=",
        "ReadWritePaths=",
    ):
        check(capability in service, f"service lacks {capability}")

    process = subprocess.run(
        [sys.executable, str(REPO_ROOT / prospective.SCRIPT_PATH), "--help"],
        cwd=REPO_ROOT,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={"PATH": os.environ.get("PATH", "")},
    )
    equal(process.returncode, 0, f"CLI help failed: {process.stderr}")
    help_text = process.stdout
    commands = {"preflight", "preregister", "collect", "status", "analyze", "verify"}
    for command in commands:
        check(command in help_text, f"CLI help omits {command}")
    first_choice = next(
        line.strip() for line in help_text.splitlines() if line.strip().startswith("{")
    )
    advertised = set(first_choice.split("}", 1)[0].strip("{").split(","))
    equal(advertised, commands, "CLI command surface")
    for forbidden in ("supply", "backfill", "reset", "relook", "buy", "trade"):
        check(forbidden not in advertised, f"CLI exposes forbidden command {forbidden}")


def test_spec_preflight_embedding_and_environment() -> None:
    raw = (REPO_ROOT / prospective.SPEC_PATH).read_bytes()
    equal(hashlib.sha256(raw).hexdigest(), prospective.SPEC_SHA256, "fixed spec hash")
    spec = _spec()
    equal(spec["activation"], False, "spec activation")
    equal(spec["terminal_statuses"], list(prospective.TERMINAL_STATUSES), "statuses")
    preflight_path = spec["paths"]["preflight_result"]
    ignored = subprocess.run(
        ["git", "check-ignore", "--quiet", preflight_path], cwd=REPO_ROOT, check=False
    )
    equal(ignored.returncode, 0, "preflight result is not ignored")

    preflight = _preflight_fixture()
    validated = prospective._validate_preflight_value(preflight)
    equal(validated, preflight, "preflight value validation")
    payload = prospective._prereg_payload(
        spec,
        h0="a" * 40,
        t0={"utc": "2026-07-20T12:00:00Z", "ny": "2026-07-20T08:00:00-04:00", "epoch": 1},
        preflight_path=preflight_path,
        preflight_payload=preflight,
        runtime_environment={"synthetic": "fixed"},
    )
    equal(payload["preflight"]["payload"], preflight, "embedded preflight payload")
    equal(
        payload["preflight"]["sha256"],
        hashlib.sha256(evidence.canonical_json_bytes(preflight)).hexdigest(),
        "embedded preflight identity",
    )
    equal(
        payload["runtime_environment"],
        {"synthetic": "fixed"},
        "embedded runtime environment",
    )
    runtime = prospective._verified_runtime_environment(
        REPO_ROOT,
        subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            stdout=subprocess.PIPE,
            text=True,
        ).stdout.strip(),
    )
    equal(
        set(runtime["distributions"]),
        set(prospective.RUNTIME_DISTRIBUTIONS),
        "runtime distribution binding",
    )
    bound = prospective._bound_paths(REPO_ROOT, spec)
    check(preflight_path not in bound, "ignored preflight was bound as an H0 artifact")

    prospective._require_public_environment({})
    prospective._require_public_environment({"MX_HOR": "15"})
    for value in ("", "1", "5", "60", "fifteen"):
        with expect_raises(prospective.ProspectiveError, "MX_HOR"):
            prospective._require_public_environment({"MX_HOR": value})
    for name in prospective.FORBIDDEN_ENVIRONMENT:
        with expect_raises(prospective.ProspectiveError, name):
            prospective._require_public_environment({name: ""})

    changed = json.loads(json.dumps(preflight))
    changed["tick_provenance"]["finite_second_coverage"] = 0.994
    with expect_raises(prospective.ProspectiveError, "tick-provenance"):
        prospective._validate_preflight_value(changed)
    changed = json.loads(json.dumps(preflight))
    changed["tick_provenance"]["recovered_or_malformed_rows_rejected"] = 0
    changed["tick_provenance"]["admitted_live_rows"] = changed["tick_provenance"][
        "rows"
    ]
    changed["tick_provenance"]["finite_second_coverage"] = 1.0
    with expect_raises(prospective.ProspectiveError, "tick-provenance"):
        prospective._validate_preflight_value(changed)
    changed = json.loads(json.dumps(preflight))
    changed["generated_utc"] = "2026-07-17T21:00:00Z"
    with expect_raises(prospective.ProspectiveError, "score clock"):
        prospective._validate_preflight_value(changed)
    changed = json.loads(json.dumps(preflight))
    changed["host"] = ""
    with expect_raises(prospective.ProspectiveError, "preflight host"):
        prospective._validate_preflight_value(changed)
    with (
        mock.patch.object(prospective, "_validated_root", return_value=REPO_ROOT),
        mock.patch.object(prospective, "_load_spec", return_value=spec),
        mock.patch.object(prospective, "_load_preflight", return_value=preflight),
        expect_raises(prospective.ProspectiveError, "within 24 hours"),
    ):
        prospective.preregister(
            t0_utc="2026-07-27T12:00:00Z",
            t0_ny="2026-07-27T08:00:00-04:00",
            preflight_path=preflight_path,
            user_approved_t0=True,
            repo_root=REPO_ROOT,
            now=prospective._utc(preflight["generated_utc"], "fixture generated")
            + timedelta(seconds=prospective.PREFLIGHT_REGISTRATION_MAX_AGE_SECONDS + 1),
        )
    changed = json.loads(json.dumps(preflight))
    changed["outcome_fields_present"] = True
    with expect_raises(prospective.ProspectiveError, "lifecycle"):
        prospective._validate_preflight_value(changed)


def test_preflight_generation_and_strict_new_validation() -> None:
    spec = _spec()
    for current, expected in (
        ("2026-07-17", "2026-07-16"),
        ("2026-07-19", "2026-07-17"),
        ("2026-07-20", "2026-07-17"),
        ("2026-07-21", "2026-07-20"),
        ("2026-03-09", "2026-03-06"),
        ("2026-11-02", "2026-10-30"),
    ):
        equal(
            prospective._latest_prior_weekday(
                datetime.fromisoformat(current).date()
            ).isoformat(),
            expected,
            f"latest prior weekday for {current}",
        )
    fixture = _preflight_fixture()
    fixture["scoring_parity"]["incumbent"]["content_id"] = spec["incumbent"][
        "loaded_content_id"
    ]
    start_epoch = fixture["tick_provenance"]["session_start_epoch"]
    end_epoch = fixture["tick_provenance"]["session_end_epoch"]
    epochs = np.arange(start_epoch, end_epoch + 1, dtype="int64")
    ticks = pd.DataFrame(
        {
            "epoch": epochs,
            "quote": np.ones(len(epochs), dtype="float64"),
            "bid": np.ones(len(epochs), dtype="float64"),
            "ask": np.ones(len(epochs), dtype="float64"),
        }
    )
    ticks.loc[len(ticks) // 2, "bid"] = np.nan
    descriptor = fixture["provider"]["tick_descriptor"]
    first_clock = datetime(2026, 7, 17, 20, 34, 59, tzinfo=UTC)
    generated_clock = datetime(2026, 7, 17, 20, 35, 1, tzinfo=UTC)
    events: list[str] = []

    class SequencedDateTime(datetime):
        @classmethod
        def now(cls, tz: Any = None) -> datetime:
            events.append("now")
            value = first_clock if events.count("now") == 1 else generated_clock
            return value.astimezone(tz) if tz is not None else value.replace(tzinfo=None)

    def score_after_observations(*args: Any, **kwargs: Any) -> dict[str, Any]:
        del args, kwargs
        events.append("score")
        return fixture["scoring_parity"]

    def read_ticks(*args: Any, **kwargs: Any) -> tuple[pd.DataFrame, dict[str, Any]]:
        del args, kwargs
        events.append("ticks")
        return ticks, descriptor

    with (
        mock.patch.object(prospective, "datetime", SequencedDateTime),
        mock.patch.object(prospective, "_require_public_environment"),
        mock.patch.object(prospective, "_validated_root", return_value=REPO_ROOT),
        mock.patch.object(
            prospective,
            "authenticate_policies",
            return_value=(SimpleNamespace(), SimpleNamespace(), spec),
        ),
        mock.patch.object(
            prospective,
            "_read_tick_rows",
            side_effect=read_ticks,
        ),
        mock.patch.object(
            prospective,
            "_score_latest_store_row",
            side_effect=score_after_observations,
        ),
    ):
        built = prospective.build_preflight(
            fixture["spent_session_date_ny"], repo_root=REPO_ROOT
        )
    equal(
        events,
        ["now", "ticks", "score", "now"],
        "preflight generation clock ordering",
    )
    equal(
        built["generated_utc"],
        prospective._canonical_utc(generated_clock),
        "preflight generation timestamp",
    )

    stale = json.loads(json.dumps(built))
    stale["spent_session_date_ny"] = "2026-07-15"
    with expect_raises(prospective.ProspectiveError, "newest prior"):
        prospective._validate_preflight_value(stale, spec)

    invalid = json.loads(json.dumps(built))
    invalid["tick_provenance"]["recovered_or_malformed_rows_rejected"] = 0
    invalid["tick_provenance"]["admitted_live_rows"] = invalid["tick_provenance"][
        "rows"
    ]
    invalid["tick_provenance"]["finite_second_coverage"] = 1.0
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        relative = spec["paths"]["preflight_result"]
        destination = root / relative
        with (
            mock.patch.object(prospective, "_validated_root", return_value=root),
            mock.patch.object(prospective, "_load_spec", return_value=spec),
            mock.patch.object(prospective, "build_preflight", return_value=invalid),
            expect_raises(prospective.ProspectiveError, "tick-provenance"),
        ):
            prospective.write_preflight(
                built["spent_session_date_ny"], relative, repo_root=root
            )
        check(not destination.exists(), "invalid fresh preflight consumed its destination")
        check(
            not destination.parent.exists(),
            "invalid fresh preflight created its output directory",
        )


def test_score_feature_clock_and_session_parity() -> None:
    columns = [f"f{index:03d}" for index in range(337)]
    values = np.linspace(-1.0, 1.0, 337, dtype="float32")
    row = pd.Series(values[::-1], index=columns[::-1])
    first = prospective._feature_hash(row, columns, 1_768_000_000)
    reordered = pd.Series(values, index=columns)
    equal(
        first,
        prospective._feature_hash(reordered, columns, 1_768_000_000),
        "feature hash did not bind requested order",
    )
    check(
        first != prospective._feature_hash(reordered, columns, 1_768_000_060),
        "feature hash omitted timestamp",
    )
    bad = reordered.copy()
    bad.iloc[5] = np.nan
    with expect_raises(prospective.ProspectiveError, "finite float32"):
        prospective._feature_hash(bad, columns, 1_768_000_000)

    candidate = _FakeBook("candidate", 0.61)
    incumbent = _FakeBook("incumbent", 0.39)
    equal(prospective._score_payload(candidate, reordered)["direction"], "UP", "UP score")
    equal(prospective._score_payload(incumbent, reordered)["direction"], "DOWN", "DOWN score")
    with expect_raises(prospective.ProspectiveError, "scalar/batch"):
        prospective._score_payload(
            _FakeBook("mismatch", 0.61, batch_probability=0.62), reordered
        )
    with expect_raises(prospective.ProspectiveError, "scalar/batch"):
        prospective._score_payload(_FakeBook("mismatch", 0.61, batch_gate=False), reordered)

    check(
        prospective._is_transient_feature_gap(
            prospective.LiveFeatureError("USDCHF: latest feature row is stale: synthetic")
        ),
        "stale availability gap was classified as malformed",
    )
    check(
        not prospective._is_transient_feature_gap(
            prospective.LiveFeatureError(
                "USDCHF: rolling store missing columns ['close']"
            )
        ),
        "malformed feature schema was classified as transient",
    )
    with tempfile.TemporaryDirectory(prefix="usdchf-prospective-features-") as raw:
        root = Path(raw)
        (root / "candles").mkdir()
        scoring_spec = {
            "paths": {"candle_store": "candles"},
            "runtime": {"decision_shift_seconds": 60},
        }
        feature_timestamp = pd.Timestamp(
            datetime(2026, 7, 13, 7, 59, tzinfo=NY)
        ).tz_convert("UTC")
        feature_row = SimpleNamespace(
            timestamp=feature_timestamp,
            row=reordered,
            source="synthetic-stable-public-store",
            warnings=[],
        )
        builder = mock.Mock()
        builder.feature_row.return_value = feature_row
        with (
            mock.patch.object(prospective, "LiveFeatureBuilder", return_value=builder) as factory,
            mock.patch.object(
                prospective,
                "_validate_candle_store_snapshot",
                return_value={"USDCHF": (1, 2, 3, 4)},
            ),
            mock.patch.object(
                prospective,
                "_assert_candle_store_unchanged",
                side_effect=[
                    prospective.TransientFeatureGap(
                        "USDCHF: rolling store changed while scoring"
                    ),
                    None,
                ],
            ),
        ):
            retried = prospective._score_latest_store_row(
                root, scoring_spec, candidate, incumbent
            )
        equal(factory.call_count, 2, "atomic candle replacement retry count")
        equal(retried["decision_close_epoch"], _epoch(datetime(2026, 7, 13, 8, 0, tzinfo=NY)), "retried feature clock")

        malformed_builder = mock.Mock()
        malformed_builder.feature_row.side_effect = prospective.LiveFeatureError(
            "USDCHF: rolling store missing columns ['close']"
        )
        with (
            mock.patch.object(
                prospective, "LiveFeatureBuilder", return_value=malformed_builder
            ),
            mock.patch.object(
                prospective,
                "_validate_candle_store_snapshot",
                return_value={"USDCHF": (1, 2, 3, 4)},
            ),
            expect_raises(prospective.ProspectiveError, "malformed live feature source"),
        ):
            prospective._score_latest_store_row(
                root, scoring_spec, candidate, incumbent
            )

    winter = datetime(2026, 1, 12, 8, 0, tzinfo=NY)
    summer = datetime(2026, 7, 13, 8, 0, tzinfo=NY)
    winter_last = datetime(2026, 1, 12, 16, 34, tzinfo=NY)
    winter_cutoff = datetime(2026, 1, 12, 16, 35, tzinfo=NY)
    epochs = prospective._expected_decision_epochs(
        _epoch(winter) - 60, _epoch(winter_cutoff) + 60
    )
    check(_epoch(winter) in epochs, "winter 08:00 missing")
    check(_epoch(winter_last) in epochs, "winter 16:34 missing")
    check(_epoch(winter_cutoff) not in epochs, "16:35 admitted")
    summer_epochs = prospective._expected_decision_epochs(
        _epoch(summer), _epoch(summer) + 60
    )
    equal(int(summer_epochs[0]), _epoch(summer), "summer DST mapping")
    equal(int(winter.utcoffset().total_seconds()), -18_000, "winter UTC offset")
    equal(int(summer.utcoffset().total_seconds()), -14_400, "summer UTC offset")
    weekend = datetime(2026, 7, 18, 12, 0, tzinfo=NY)
    equal(
        len(prospective._expected_decision_epochs(_epoch(weekend), _epoch(weekend))),
        0,
        "weekend decision admitted",
    )


def test_fixed_policy_authentication_and_inactive_loading() -> None:
    spec = _spec()
    incumbent = prospective._load_incumbent(REPO_ROOT, spec)
    equal(incumbent.book_id, spec["incumbent"]["book_id"], "incumbent ID")
    equal(incumbent.content_id, spec["incumbent"]["loaded_content_id"], "incumbent content")
    columns = list(incumbent.feature_cols)
    equal(len(columns), 337, "incumbent feature count")
    source_spec = {"evidence_kinds": {"terminal": "synthetic.refit.terminal/v1"}}
    terminal = evidence.EvidenceEnvelope.create(
        kind="synthetic.refit.terminal/v1",
        payload={"status": "INACTIVE_UNTESTED_PROSPECTIVE"},
    )
    verification = {
        "terminal_id": spec["candidate"]["terminal_id"],
        "h1_git_sha": spec["candidate"]["h1_git_sha"],
        "status": "INACTIVE_UNTESTED_PROSPECTIVE",
    }
    package_stub = {"manifest_sha256": spec["candidate"]["manifest_sha256"]}
    with (
        mock.patch.object(prospective.current_refit, "verify", return_value=verification),
        mock.patch.object(prospective.current_refit, "_load_spec", return_value=source_spec),
        mock.patch.object(
            prospective.current_refit, "_validate_package", return_value=package_stub
        ) as package_validator,
        mock.patch.object(prospective.evidence, "verify_object", return_value=terminal),
    ):
        equal(
            prospective._candidate_package(REPO_ROOT, spec),
            package_stub,
            "candidate package authentication",
        )
        equal(
            package_validator.call_args.args[2:],
            (spec["candidate"]["seal_id"], spec["candidate"]["h1_git_sha"]),
            "candidate sealed lineage",
        )
    invalid_verification = dict(verification)
    invalid_verification["status"] = "ACTIVE"
    with (
        mock.patch.object(
            prospective.current_refit, "verify", return_value=invalid_verification
        ),
        expect_raises(prospective.ProspectiveError, "identity differs"),
    ):
        prospective._candidate_package(REPO_ROOT, spec)

    with tempfile.TemporaryDirectory(prefix="usdchf-prospective-loader-") as raw:
        directory = Path(raw)
        models = []
        for seed in (0, 1, 2):
            filename = f"model-{seed}.txt"
            (directory / filename).write_text(f"synthetic-{seed}\n", encoding="utf-8")
            models.append({"seed": seed, "file": filename})
        strategy_path = directory / "m15xpny_USDCHF_r202605_strategy.json"
        strategy_path.write_text("{}\n", encoding="utf-8")
        strategy = {
            "artifact_id": spec["candidate"]["artifact_id"],
            "pair": "USDCHF",
            "side": "combined",
            "ensemble": "arithmetic_mean_in_seed_order",
            "feature_cols": columns,
            "feature_dtype": "float32",
            "models": models,
            "confidence_threshold": spec["candidate"]["confidence_threshold"],
            "coverage": 0.01,
            "seed_order": [0, 1, 2],
            "decision_shift_seconds": 60,
            "horizon_seconds": 900,
            "activation": False,
        }
        package = {"directory": str(directory), "strategy": strategy}

        class FakeBooster:
            def __init__(self, *, model_file: str) -> None:
                self.model_file = model_file

            def num_feature(self) -> int:
                return 337

        with (
            mock.patch.object(prospective, "_candidate_package", return_value=package),
            mock.patch.object(prospective.lgb, "Booster", FakeBooster),
        ):
            loaded = prospective._load_candidate(REPO_ROOT, spec)
        equal(loaded.feature_cols, columns, "candidate feature order")
        equal([Path(item.model_file).name for item in loaded.boosters], ["model-0.txt", "model-1.txt", "model-2.txt"], "candidate seed order")
        equal(loaded.strategy["activation"], False, "inactive candidate activation")
        equal(loaded.coverage_source, "sealed_candidate_strategy", "candidate loader owner")

        active = json.loads(json.dumps(strategy))
        active["activation"] = True
        with (
            mock.patch.object(
                prospective,
                "_candidate_package",
                return_value={"directory": str(directory), "strategy": active},
            ),
            expect_raises(prospective.ProspectiveError, "feature/model order differs"),
        ):
            prospective._load_candidate(REPO_ROOT, spec)

    equal(incumbent.feature_cols, columns, "candidate/incumbent feature order")
    source_spec = prospective.current_refit._load_spec(REPO_ROOT)
    actual_package = prospective.current_refit._validate_package(
        REPO_ROOT,
        source_spec,
        spec["candidate"]["seal_id"],
        spec["candidate"]["h1_git_sha"],
    )
    with mock.patch.object(
        prospective, "_candidate_package", return_value=actual_package
    ):
        actual_candidate = prospective._load_candidate(REPO_ROOT, spec)
    equal(actual_candidate.feature_cols, incumbent.feature_cols, "real common feature order")
    equal(len(actual_candidate.boosters), 3, "real candidate booster count")
    real_row = pd.Series(
        np.linspace(-0.1, 0.1, len(columns), dtype="float32"),
        index=columns,
    )
    candidate_score = prospective._score_payload(actual_candidate, real_row)
    incumbent_score = prospective._score_payload(incumbent, real_row)
    equal(candidate_score["book_id"], spec["candidate"]["artifact_id"], "real candidate score")
    equal(incumbent_score["book_id"], spec["incumbent"]["book_id"], "real incumbent score")
    check(
        candidate_score["direction"] in {"UP", "DOWN"}
        and incumbent_score["direction"] in {"UP", "DOWN"},
        "real policy directions",
    )


def test_tick_settlement_boundaries_ties_and_scheduling() -> None:
    t = 1_768_000_020

    def rows(entry: int, exit_epoch: int, *, tie: bool = False) -> pd.DataFrame:
        entry_quote = 1.25
        exit_quote = entry_quote if tie else 1.26
        return pd.DataFrame(
            {
                "epoch": [entry, exit_epoch],
                "quote": [entry_quote, exit_quote],
                "bid": [1.24, 1.25],
                "ask": [1.26, 1.27],
            }
        )

    exact = rows(t + 1, t + 901)
    canonical_owner = (
        prospective.refresh_stats.canonical_settlement_with_exit_on_decision_grid
    )

    def timestamp_only_owner(
        decision_times: Any,
        observation_times: Any,
        closes: Any,
        **kwargs: Any,
    ) -> Any:
        check(np.asarray(closes).tolist() == [1.0, 1.0, 1.0], "pre-access prices reached return owner")
        return canonical_owner(decision_times, observation_times, closes, **kwargs)

    with mock.patch.object(
        prospective.refresh_stats,
        "canonical_settlement_with_exit_on_decision_grid",
        side_effect=timestamp_only_owner,
    ):
        selected = prospective._select_settlement(
            t, exact, prospective._live_tick_mask(exact)
        )
    equal(selected["status"], "SETTLED", "exact settlement")
    equal((selected["entry_epoch"], selected["exit_epoch"]), (t + 1, t + 901), "targets")
    manifest_row = {"prereg_id": "a" * 64, "decision_close_epoch": t, **selected}
    changed_prices = dict(manifest_row)
    changed_prices["entry_quote"] = 999.0
    changed_prices["exit_quote"] = 0.001
    equal(
        prospective._selection_manifest([manifest_row]),
        prospective._selection_manifest([changed_prices]),
        "outcome-blind selection manifest included prices",
    )

    boundary = rows(t + 11, t + 891)
    equal(
        prospective._select_settlement(t, boundary, prospective._live_tick_mask(boundary))["status"],
        "SETTLED",
        "inclusive tolerance boundary",
    )
    late = rows(t + 12, t + 901)
    equal(
        prospective._select_settlement(t, late, prospective._live_tick_mask(late))["reason"],
        "ENTRY_LATE",
        "late entry",
    )
    early = rows(t + 1, t + 890)
    equal(
        prospective._select_settlement(t, early, prospective._live_tick_mask(early))["reason"],
        "EXIT_EARLY",
        "early exit",
    )

    recovered = exact.copy()
    recovered.loc[0, "bid"] = np.nan
    equal(
        prospective._select_settlement(
            t, recovered, prospective._live_tick_mask(recovered)
        )["reason"],
        "ENTRY_LATE",
        "recovered-history row admitted",
    )
    malformed = exact.copy()
    malformed.loc[1, "quote"] = -1.0
    equal(int(prospective._live_tick_mask(malformed).sum()), 1, "negative tick admitted")
    duplicate = pd.concat([exact.iloc[[0]], exact.iloc[[0]], exact.iloc[[1]]], ignore_index=True)
    with expect_raises(prospective.ProspectiveError, "strict"):
        prospective._select_settlement(
            t, duplicate, prospective._live_tick_mask(duplicate)
        )

    tie = prospective._settled_correctness("UP", 1.25, 1.25)
    equal(tie, False, "UP tie did not lose")
    equal(prospective._settled_correctness("DOWN", 1.25, 1.25), False, "DOWN tie")
    equal(prospective._settled_correctness("UP", 1.25, 1.26), True, "UP correctness")
    equal(prospective._settled_correctness("DOWN", 1.25, 1.24), True, "DOWN correctness")

    prereg_id = "a" * 64
    decisions = [
        _decision(prereg_id, t, candidate_probability=0.6, incumbent_probability=0.4),
        _decision(prereg_id, t + 60, candidate_probability=0.4, incumbent_probability=0.6),
        _decision(prereg_id, t + 900, candidate_probability=0.4, incumbent_probability=0.6),
        _decision(prereg_id, t + 960, candidate_probability=0.6, incumbent_probability=0.4),
    ]
    settlement = [
        {"prereg_id": prereg_id, "decision_close_epoch": row["decision_close_epoch"], "status": "SETTLED", "reason": ""}
        for row in decisions
    ]
    counts = prospective._scheduled_counts(decisions, settlement)
    equal(counts["common_settled"], 4, "common settled count")
    equal(counts["candidate"], {"combined": 2, "UP": 1, "DOWN": 1}, "candidate schedule")
    equal(counts["incumbent"], {"combined": 2, "UP": 1, "DOWN": 1}, "incumbent schedule")
    seal_payload = prospective._seal_payload(
        prereg_id,
        {
            "cutoff_epoch": t + 10_000,
            "week": 8,
            "trigger": "COUNT_TRIGGER",
            "counts": counts,
            "selection": prospective._selection_manifest(settlement),
        },
        settlement_sha256="a" * 64,
        sealed_epoch=t + 10_659,
    )
    encoded_seal = json.dumps(seal_payload, sort_keys=True).lower()
    for forbidden in ("entry_quote", "exit_quote", "correctness", "accuracy", "yield", "profit", "money"):
        check(forbidden not in encoded_seal, f"outcome-blind seal contains {forbidden}")


def test_tick_store_is_public_strict_and_unrepaired() -> None:
    with tempfile.TemporaryDirectory(prefix="usdchf-prospective-ticks-") as raw:
        root = Path(raw)
        spec = _spec()
        spec["paths"]["tick_shards"] = "ticks/pages"
        spec["paths"]["tick_progress"] = "ticks/progress.json"
        pages = root / "ticks/pages"
        pages.mkdir(parents=True)
        progress = root / "ticks/progress.json"
        progress.write_text(
            json.dumps({"last_persisted_epoch": 40, "pages": 2}),
            encoding="utf-8",
        )
        first = pd.DataFrame(
            {
                "epoch": [10, 20],
                "quote": [1.0, 1.1],
                "bid": [np.nan, 1.09],
                "ask": [np.nan, 1.11],
            }
        )
        second = pd.DataFrame(
            {
                "epoch": [30, 40],
                "quote": [1.2, 1.3],
                "bid": [1.19, 1.29],
                "ask": [1.21, 1.31],
            }
        )
        first.to_parquet(pages / "000000_10_20.parquet", index=False)
        second_path = pages / "000001_30_40.parquet"
        second.to_parquet(second_path, index=False)
        rows, descriptor = prospective._read_tick_rows(root, spec)
        equal(rows["epoch"].tolist(), [10, 20, 30, 40], "tick source order")
        equal(descriptor["last_persisted_epoch"], 40, "tick progress boundary")
        equal(
            prospective._live_tick_mask(rows).tolist(),
            [False, True, True, True],
            "recovered-history distinction",
        )

        original_read_parquet = pd.read_parquet
        appended = False

        def read_while_appending(*args: Any, **kwargs: Any) -> pd.DataFrame:
            nonlocal appended
            frame = original_read_parquet(*args, **kwargs)
            if not appended:
                appended = True
                pd.DataFrame(
                    {
                        "epoch": [50],
                        "quote": [1.4],
                        "bid": [1.39],
                        "ask": [1.41],
                    }
                ).to_parquet(pages / "000002_50_50.parquet", index=False)
                progress.write_text(
                    json.dumps({"last_persisted_epoch": 50, "pages": 3}),
                    encoding="utf-8",
                )
            return frame

        with mock.patch.object(pd, "read_parquet", side_effect=read_while_appending):
            append_safe_rows, append_descriptor = prospective._read_tick_rows(
                root, spec
            )
        equal(
            append_safe_rows["epoch"].tolist(),
            [10, 20, 30, 40],
            "append-safe snapshot admitted a later page",
        )
        equal(append_descriptor["last_persisted_epoch"], 40, "snapshot cursor")
        (pages / "000002_50_50.parquet").unlink()
        progress.write_text(
            json.dumps({"last_persisted_epoch": 40, "pages": 2}),
            encoding="utf-8",
        )
        interval_rows, _ = prospective._read_tick_rows(
            root, spec, required_intervals=[(19, 31)]
        )
        equal(interval_rows["epoch"].tolist(), [20, 30], "interval shard filter")

        # The provider appends a new live tick before older recovered gap rows.
        # Reject the recovered rows, but preserve raw order and require only the
        # admitted live clock to remain strict.
        pd.DataFrame(
            {
                "epoch": [40, 30],
                "quote": [1.3, 1.2],
                "bid": [1.29, np.nan],
                "ask": [1.31, np.nan],
            }
        ).to_parquet(second_path, index=False)
        recovered_rows, _ = prospective._read_tick_rows(root, spec)
        equal(
            recovered_rows["epoch"].tolist(),
            [10, 20, 40, 30],
            "provider order was repaired",
        )
        equal(
            recovered_rows.loc[
                prospective._live_tick_mask(recovered_rows), "epoch"
            ].tolist(),
            [20, 40],
            "recovered provider row was admitted",
        )

        second.iloc[::-1].to_parquet(second_path, index=False)
        with expect_raises(prospective.ProspectiveError, "not strict"):
            prospective._read_tick_rows(root, spec)
        second_path.unlink()
        second_path = pages / "000001_15_30.parquet"
        pd.DataFrame(
            {
                "epoch": [15, 30],
                "quote": [1.0, 1.2],
                "bid": [0.99, 1.19],
                "ask": [1.01, 1.21],
            }
        ).to_parquet(second_path, index=False)
        with expect_raises(prospective.ProspectiveError, "out of order"):
            prospective._read_tick_rows(root, spec)
        second_path.unlink()
        second_path = pages / "000001_30_41.parquet"
        pd.DataFrame(
            {
                "epoch": [30, 41],
                "quote": [1.2, 1.3],
                "bid": [1.19, 1.29],
                "ask": [1.21, 1.31],
            }
        ).to_parquet(second_path, index=False)
        with expect_raises(prospective.ProspectiveError, "progress cursor"):
            prospective._read_tick_rows(root, spec)
        second_path.unlink()
        second_path = pages / "000001_30_40.parquet"
        pd.DataFrame(
            {
                "epoch": [30.9, 40.0],
                "quote": [1.2, 1.3],
                "bid": [1.19, 1.29],
                "ask": [1.21, 1.31],
            }
        ).to_parquet(second_path, index=False)
        with expect_raises(prospective.ProspectiveError, "exact integer"):
            prospective._read_tick_rows(root, spec)


def test_sqlite_immutability_lock_and_packet() -> None:
    prereg_id = "b" * 64
    with tempfile.TemporaryDirectory(prefix="usdchf-prospective-sqlite-") as raw:
        root = Path(raw)
        trial_dir = root / "trial"
        trial_dir.mkdir()
        database = trial_dir / "trial.sqlite"
        lock = trial_dir / "trial.lock"
        packet = trial_dir / "sealed_source.sqlite"
        paths = {
            "dir": trial_dir,
            "database": database,
            "lock": lock,
            "packet": packet,
            "packet_stage": trial_dir / ".sealed_source.building.sqlite",
        }
        with prospective._trial_lock(lock):
            with expect_raises(prospective.ProspectiveError, "another prospective writer"):
                with prospective._trial_lock(lock):
                    raise GateFailure("second writer acquired lock")

        conn = prospective._open_trial(database)
        try:
            pragmas = {
                "journal_mode": str(conn.execute("PRAGMA journal_mode").fetchone()[0]).lower(),
                "foreign_keys": int(conn.execute("PRAGMA foreign_keys").fetchone()[0]),
                "synchronous": int(conn.execute("PRAGMA synchronous").fetchone()[0]),
            }
            equal(pragmas, {"journal_mode": "wal", "foreign_keys": 1, "synchronous": 2}, "trial pragmas")
            metadata = {
                "schema": prospective.TRIAL_METADATA_SCHEMA,
                "prereg_id": prereg_id,
                "outcome_fields_present": False,
                "activation": False,
            }
            prospective._initialize_trial(conn, metadata, created_epoch=1)
            prospective._initialize_trial(conn, metadata, created_epoch=2)
            foreign_database_link = trial_dir / "foreign-trial.sqlite"
            os.link(database, foreign_database_link)
            with expect_raises(prospective.ProspectiveError, "hard link"):
                prospective._open_trial(database, create=False)
            foreign_database_link.unlink()
            changed = dict(metadata)
            changed["activation"] = True
            with expect_raises(prospective.ProspectiveError, "metadata conflicts"):
                prospective._initialize_trial(conn, changed, created_epoch=1)

            decision_epoch = _epoch(datetime(2026, 7, 20, 8, 0, tzinfo=NY))
            decision = _decision(prereg_id, decision_epoch)
            equal(prospective._insert_decision_once(conn, decision), True, "first decision")
            equal(prospective._insert_decision_once(conn, decision), False, "decision retry")
            conflict = dict(decision)
            conflict["recorded_epoch"] += 1
            with expect_raises(prospective.ProspectiveError, "decision retry conflicts"):
                prospective._insert_decision_once(conn, conflict)
            gap = _decision(prereg_id, decision_epoch + 60, common=False)
            equal(prospective._insert_decision_once(conn, gap), True, "explicit gap row")
            loaded_rows = prospective._load_decisions(conn, prereg_id)
            equal(len(loaded_rows), 2, "complete decision-grid logging")
            check(
                loaded_rows[1]["common_scoreable"] is False
                and loaded_rows[1]["candidate"] is None
                and loaded_rows[1]["incumbent"] is None
                and loaded_rows[1]["exclusion_reason"],
                "gap row retained a score or omitted its reason",
            )
            for statement in (
                "UPDATE decisions SET exclusion_reason='changed'",
                "DELETE FROM decisions",
                "UPDATE trial_metadata SET schema='changed'",
                "DELETE FROM trial_metadata",
            ):
                with expect_raises(sqlite3.IntegrityError, "immutable"):
                    conn.execute(statement)

            seal = {
                "schema": "usdchf-m15-prospective-trial-seal/v1",
                "prereg_id": prereg_id,
                "cutoff_epoch": decision_epoch + 960,
                "week": 8,
                "trigger": "COUNT_TRIGGER",
                "sealed_epoch": decision_epoch + 1_619,
                "counts": {"candidate": {"UP": 100, "DOWN": 100}},
                "selection": {"rows": 2},
                "settlement_sha256": "a" * 64,
                "outcome_fields_present": False,
                "activation": False,
            }
            equal(prospective._insert_trial_seal_once(conn, seal), True, "first seal")
            equal(prospective._insert_trial_seal_once(conn, seal), False, "seal retry")
            with expect_raises(prospective.ProspectiveError, "after the trial seal"):
                prospective._insert_decision_once(
                    conn, _decision(prereg_id, decision_epoch + 1_020)
                )
            for statement in (
                "UPDATE trial_seal SET trigger='WEEK_26_CAP'",
                "DELETE FROM trial_seal",
            ):
                with expect_raises(sqlite3.IntegrityError, "immutable"):
                    conn.execute(statement)

            columns = {
                row[1]
                for table in prospective.TRIAL_TABLES
                for row in conn.execute(f"PRAGMA table_info({table})")
            }
            for token in ("return", "correct", "accuracy", "yield", "profit", "payout", "stake", "money"):
                check(not any(token in name.lower() for name in columns), f"live schema contains {token}")
            check(Path(str(database) + "-wal").exists(), "committed rows were not resident in WAL")

            settlement = [
                {
                    "prereg_id": prereg_id,
                    "decision_close_epoch": decision["decision_close_epoch"],
                    "status": "SETTLED",
                    "reason": "",
                    "entry_epoch": decision["decision_close_epoch"] + 1,
                    "entry_quote": 1.25,
                    "entry_bid": 1.24,
                    "entry_ask": 1.26,
                    "exit_epoch": decision["decision_close_epoch"] + 901,
                    "exit_quote": 1.26,
                    "exit_bid": 1.25,
                    "exit_ask": 1.27,
                },
                {
                    "prereg_id": prereg_id,
                    "decision_close_epoch": gap["decision_close_epoch"],
                    "status": "NOT_COMMON_SCOREABLE",
                    "reason": gap["exclusion_reason"],
                },
            ]
            reference = prospective._build_packet(
                root=root,
                source_conn=conn,
                paths=paths,
                prereg_id=prereg_id,
                settlement=settlement,
            )
            equal(reference.path, "trial/sealed_source.sqlite", "packet path")
            equal(reference.sha256, prospective._sha256_file(packet), "packet hash")
            check(packet.is_file() and not packet.is_symlink(), "packet missing")
            check(not packet.stat().st_mode & 0o222, "packet is writable")
            for suffix in ("-wal", "-shm", "-journal"):
                check(not Path(str(packet) + suffix).exists(), f"packet retained {suffix}")
            packet_conn = prospective._open_packet_readonly(packet)
            try:
                prospective._assert_packet_content(packet_conn, prereg_id, 2)
                equal(
                    packet_conn.execute("PRAGMA integrity_check").fetchone()[0],
                    "ok",
                    "packet integrity",
                )
                equal(
                    packet_conn.execute("SELECT COUNT(*) FROM settlement_rows").fetchone()[0],
                    2,
                    "packet settlement rows",
                )
            finally:
                packet_conn.close()
            os.link(packet, paths["packet_stage"])
            equal(packet.stat().st_nlink, 2, "packet link-window fixture")
            recovered_link = prospective._build_packet(
                root=root,
                source_conn=conn,
                paths=paths,
                prereg_id=prereg_id,
                settlement=settlement,
            )
            equal(recovered_link, reference, "packet link-window recovery")
            check(not paths["packet_stage"].exists(), "packet recovery stage remains")
            equal(packet.stat().st_nlink, 1, "packet recovery retained hard link")
            adopted = prospective._build_packet(
                root=root,
                source_conn=conn,
                paths=paths,
                prereg_id=prereg_id,
                settlement=settlement,
            )
            equal(adopted, reference, "packet recovery adoption")

            prelink_paths = {
                **paths,
                "packet": trial_dir / "prelink_source.sqlite",
                "packet_stage": trial_dir / ".prelink_source.building.sqlite",
            }
            prelink_reference = prospective._build_packet(
                root=root,
                source_conn=conn,
                paths=prelink_paths,
                prereg_id=prereg_id,
                settlement=settlement,
            )
            prelink_paths["packet"].replace(prelink_paths["packet_stage"])
            recovered_prelink = prospective._build_packet(
                root=root,
                source_conn=conn,
                paths=prelink_paths,
                prereg_id=prereg_id,
                settlement=settlement,
            )
            equal(
                recovered_prelink.sha256,
                prelink_reference.sha256,
                "finalized pre-link packet recovery",
            )
            check(
                not prelink_paths["packet_stage"].exists(),
                "finalized pre-link stage remains",
            )

            altered = json.loads(json.dumps(settlement))
            altered[0]["exit_quote"] = 1.27
            alternate_paths = {
                **paths,
                "packet": trial_dir / "altered_source.sqlite",
                "packet_stage": trial_dir / ".altered_source.building.sqlite",
            }
            prospective._build_packet(
                root=root,
                source_conn=conn,
                paths=alternate_paths,
                prereg_id=prereg_id,
                settlement=altered,
            )
            with expect_raises(prospective.ProspectiveError, "mechanical selection"):
                prospective._build_packet(
                    root=root,
                    source_conn=conn,
                    paths=alternate_paths,
                    prereg_id=prereg_id,
                    settlement=settlement,
                )

            invalid = prospective._protocol_invalid_payload(
                prereg_id,
                detected_epoch=decision_epoch + 2_000,
                stage="SYNTHETIC_PROTOCOL",
                reason="fixed parity mismatch",
            )
            equal(
                prospective._insert_protocol_invalid_once(conn, invalid),
                True,
                "first protocol INVALID",
            )
            equal(
                prospective._insert_protocol_invalid_once(conn, invalid),
                False,
                "exact protocol INVALID retry",
            )
            divergent = dict(invalid)
            divergent["reason"] = "different fixed failure"
            with expect_raises(prospective.ProspectiveError, "conflicts"):
                prospective._insert_protocol_invalid_once(conn, divergent)
            with expect_raises(prospective.ProspectiveError, "protocol INVALID"):
                prospective._insert_decision_once(
                    conn, _decision(prereg_id, decision_epoch + 2_100)
                )
            for statement in (
                "UPDATE trial_invalid SET reason='changed'",
                "DELETE FROM trial_invalid",
            ):
                with expect_raises(sqlite3.IntegrityError, "immutable"):
                    conn.execute(statement)
        finally:
            conn.close()


def test_t0_and_fixed_one_look() -> None:
    t0_ny = datetime(2026, 7, 20, 8, 0, tzinfo=NY)
    t0_epoch = _epoch(t0_ny)
    t0 = prospective._validate_t0(
        "2026-07-20T12:00:00Z",
        "2026-07-20T08:00:00-04:00",
        now=datetime(2026, 7, 19, 12, 0, tzinfo=UTC),
    )
    equal(t0["epoch"], t0_epoch, "T0 epoch")
    for utc_value, ny_value, contains in (
        ("2026-07-20T12:00:00Z", "2026-07-20T08:00:00-05:00", "America/New_York"),
        ("2026-07-21T12:00:00Z", "2026-07-21T08:00:00-04:00", "Monday"),
        ("2026-07-20T13:00:00Z", "2026-07-20T08:00:00-04:00", "representations"),
    ):
        with expect_raises(prospective.ProspectiveError, contains):
            prospective._validate_t0(
                utc_value,
                ny_value,
                now=datetime(2026, 7, 19, 12, 0, tzinfo=UTC),
            )
    with expect_raises(prospective.ProspectiveError, "future"):
        prospective._validate_t0(
            "2026-07-20T12:00:00Z",
            "2026-07-20T08:00:00-04:00",
            now=datetime(2026, 7, 21, 12, 0, tzinfo=UTC),
        )

    spec = _spec()
    prereg = _prereg_fixture(t0_epoch)
    boundaries = prospective._week_boundaries(prereg, spec)
    equal(len(boundaries), 26, "fixed boundary count")
    equal(boundaries[0][0], 1, "first week label")
    equal(
        datetime.fromtimestamp(boundaries[0][1], UTC).astimezone(NY),
        datetime(2026, 7, 24, 17, 0, tzinfo=NY),
        "first Friday boundary",
    )
    equal(boundaries[-1][0], 26, "cap week label")

    decisions: list[dict[str, Any]] = []
    dummy_settlement: list[dict[str, Any]] = []
    ready = {
        "common_settled": 200,
        "candidate": {"combined": 200, "UP": 100, "DOWN": 100},
        "incumbent": {"combined": 200, "UP": 100, "DOWN": 100},
    }
    low = {
        "common_settled": 20,
        "candidate": {"combined": 20, "UP": 10, "DOWN": 10},
        "incumbent": {"combined": 20, "UP": 10, "DOWN": 10},
    }
    with (
        mock.patch.object(prospective, "_settlement_rows", return_value=dummy_settlement),
        mock.patch.object(prospective, "_scheduled_counts", return_value=ready),
        mock.patch.object(
            prospective,
            "_selection_manifest",
            return_value={"rows": 0, "settled": 0},
        ),
    ):
        equal(
            prospective._cutoff_decision(
                prereg,
                spec,
                decisions,
                pd.DataFrame(columns=["epoch", "quote", "bid", "ask"]),
                now_epoch=boundaries[6][1],
            ),
            None,
            "look occurred before eight complete weeks",
        )
        at_eight = prospective._cutoff_decision(
            prereg,
            spec,
            decisions,
            pd.DataFrame(columns=["epoch", "quote", "bid", "ask"]),
            now_epoch=boundaries[7][1],
        )
        equal(at_eight["week"], 8, "first allowed look week")
        equal(at_eight["trigger"], "COUNT_TRIGGER", "count trigger")

    with (
        mock.patch.object(prospective, "_settlement_rows", return_value=dummy_settlement),
        mock.patch.object(prospective, "_scheduled_counts", return_value=low),
        mock.patch.object(
            prospective,
            "_selection_manifest",
            return_value={"rows": 0, "settled": 0},
        ),
    ):
        equal(
            prospective._cutoff_decision(
                prereg,
                spec,
                decisions,
                pd.DataFrame(columns=["epoch", "quote", "bid", "ask"]),
                now_epoch=boundaries[24][1],
            ),
            None,
            "low counts stopped before cap",
        )
        at_cap = prospective._cutoff_decision(
            prereg,
            spec,
            decisions,
            pd.DataFrame(columns=["epoch", "quote", "bid", "ask"]),
            now_epoch=boundaries[25][1],
        )
        equal(at_cap["week"], 26, "cap week")
        equal(at_cap["trigger"], "WEEK_26_CAP", "cap trigger")

    connection = mock.Mock()
    connection.execute.return_value.fetchone.return_value = None
    equal(
        prospective._capture_due(
            REPO_ROOT,
            prereg,
            spec,
            connection,
            mock.Mock(),
            mock.Mock(),
            now_epoch=t0_epoch - 1,
        ),
        0,
        "collector admitted a pre-T0 row",
    )

    restart_epoch = t0_epoch + 3 * 86_400
    capture_boundary = t0_epoch + 3_600
    with (
        mock.patch.object(
            prospective,
            "_expected_decision_epochs",
            return_value=np.asarray([t0_epoch], dtype="int64"),
        ),
        mock.patch.object(
            prospective,
            "_provider_identity",
            return_value={"class": "synthetic-public-store"},
        ),
        mock.patch.object(prospective, "_assert_write_authority"),
        mock.patch.object(
            prospective, "_insert_decision_once", return_value=True
        ) as insert_decision,
    ):
        equal(
            prospective._capture_due(
                REPO_ROOT,
                prereg,
                spec,
                connection,
                mock.Mock(),
                mock.Mock(),
                now_epoch=restart_epoch,
                capture_through_epoch=capture_boundary,
            ),
            1,
            "delayed capture row count",
        )
    delayed_decision = insert_decision.call_args.args[1]
    equal(delayed_decision["recorded_epoch"], restart_epoch, "delayed capture write clock")
    equal(
        delayed_decision["exclusion_reason"],
        "UNSCORED_MISSED_BEFORE_OBSERVATION",
        "delayed capture disposition",
    )

    collection = evidence.EvidenceEnvelope.create(
        kind=spec["evidence_kinds"]["collection_access_receipt"],
        payload={"prereg_id": prereg.object_id},
        dependencies=[prereg.object_id],
    )
    sealed_stub = evidence.EvidenceEnvelope.create(
        kind=spec["evidence_kinds"]["sealed_source"],
        payload={"prereg_id": prereg.object_id},
    )
    seal_boundary = boundaries[25][1]
    seal_restart = seal_boundary + 3 * 86_400
    with (
        mock.patch.object(prospective, "_load_trial_seal", return_value=None),
        mock.patch.object(prospective, "_load_decisions", return_value=[]),
        mock.patch.object(
            prospective,
            "_read_tick_rows",
            return_value=(
                pd.DataFrame(columns=["epoch", "quote", "bid", "ask"]),
                {},
            ),
        ) as read_ticks,
        mock.patch.object(
            prospective,
            "_cutoff_decision",
            return_value={
                "cutoff_epoch": seal_boundary,
                "week": 26,
                "trigger": "WEEK_26_CAP",
                "counts": low,
                "selection": prospective._selection_manifest([]),
            },
        ) as choose_cutoff,
        mock.patch.object(prospective, "_assert_write_authority"),
        mock.patch.object(
            prospective, "_insert_trial_seal_once", return_value=True
        ) as insert_seal,
        mock.patch.object(
            prospective, "_complete_seal", return_value=sealed_stub
        ),
    ):
        equal(
            prospective._seal_if_due(
                root=REPO_ROOT,
                prereg=prereg,
                collection=collection,
                spec=spec,
                conn=connection,
                paths={},
                now_epoch=seal_restart,
                evaluate_through_epoch=seal_boundary,
            ),
            sealed_stub,
            "delayed seal completion",
        )
    equal(read_ticks.call_args.kwargs["end_epoch"], seal_boundary, "delayed seal source cutoff")
    equal(
        choose_cutoff.call_args.kwargs["now_epoch"],
        seal_boundary,
        "delayed seal look clock",
    )
    delayed_seal = insert_seal.call_args.args[1]
    equal(delayed_seal["cutoff_epoch"], seal_boundary, "delayed seal cutoff")
    equal(delayed_seal["sealed_epoch"], seal_restart, "delayed seal write clock")


def test_exact_h1_boundary() -> None:
    h0 = "a" * 40
    h1 = "b" * 40
    prereg = evidence.EvidenceEnvelope.create(
        kind="research.usdchf_m15_prospective.prereg/v1",
        payload={"h0_git_sha": h0},
    )
    collection = evidence.EvidenceEnvelope.create(
        kind="research.usdchf_m15_prospective.collection_access_receipt/v1",
        payload={"prereg_id": prereg.object_id},
        dependencies=[prereg.object_id],
    )
    exact_paths = "\n".join(
        [
            f"evidence/objects/{prereg.object_id}.json",
            f"evidence/objects/{collection.object_id}.json",
        ]
    )

    def git_result(root: Path, *arguments: str, check: bool = True) -> str:
        del root, check
        values = {
            ("branch", "--show-current"): "main",
            ("rev-parse", "HEAD"): h1,
            ("rev-parse", "origin/main"): h1,
            ("status", "--porcelain=v1", "--untracked-files=all"): "",
            ("rev-list", "--parents", "-n", "1", h1): f"{h1} {h0}",
            ("diff", "--name-only", h0, h1): exact_paths,
        }
        return values[arguments]

    with mock.patch.object(prospective, "_git", side_effect=git_result):
        equal(
            prospective._require_h1(Path("/synthetic"), prereg, collection),
            h1,
            "exact H1 boundary",
        )

    def extra_path(root: Path, *arguments: str, check: bool = True) -> str:
        value = git_result(root, *arguments, check=check)
        if arguments == ("diff", "--name-only", h0, h1):
            return value + "\nderiv_data/prospective/preflight/receipt.json"
        return value

    with (
        mock.patch.object(prospective, "_git", side_effect=extra_path),
        expect_raises(prospective.ProspectiveError, "exactly"),
    ):
        prospective._require_h1(Path("/synthetic"), prereg, collection)


def test_status_is_count_only() -> None:
    spec = _spec()
    t0_epoch = _epoch(datetime(2026, 7, 20, 8, 0, tzinfo=NY))
    prereg = _prereg_fixture(t0_epoch)
    collection = evidence.EvidenceEnvelope.create(
        kind=spec["evidence_kinds"]["collection_access_receipt"],
        payload={"prereg_id": prereg.object_id},
        dependencies=[prereg.object_id],
    )
    counts = {
        "common_settled": 1,
        "candidate": {"combined": 1, "UP": 1, "DOWN": 0},
        "incumbent": {"combined": 1, "UP": 0, "DOWN": 1},
    }
    selection = {
        "rows": 1,
        "settled": 1,
        "unsettled": 0,
        "not_common_scoreable": 0,
        "timestamp_manifest_sha256": "1" * 64,
    }
    with tempfile.TemporaryDirectory(prefix="usdchf-prospective-status-") as raw:
        root = Path(raw)
        paths = prospective._trial_paths(root, spec, prereg.object_id)
        conn = prospective._open_trial(paths["database"])
        try:
            prospective._initialize_trial(
                conn,
                {
                    "schema": prospective.TRIAL_METADATA_SCHEMA,
                    "prereg_id": prereg.object_id,
                    "outcome_fields_present": False,
                    "activation": False,
                },
                created_epoch=t0_epoch,
            )
            prospective._insert_decision_once(
                conn, _decision(prereg.object_id, t0_epoch)
            )
            prospective._insert_trial_seal_once(
                conn,
                {
                    "schema": "usdchf-m15-prospective-trial-seal/v1",
                    "prereg_id": prereg.object_id,
                    "cutoff_epoch": t0_epoch + 1,
                    "week": 26,
                    "trigger": "WEEK_26_CAP",
                    "sealed_epoch": t0_epoch + 660,
                    "counts": counts,
                    "selection": selection,
                    "settlement_sha256": "a" * 64,
                    "outcome_fields_present": False,
                    "activation": False,
                },
            )
        finally:
            conn.close()
        with (
            mock.patch.object(prospective, "_validated_root", return_value=root),
            mock.patch.object(prospective, "_load_spec", return_value=spec),
            mock.patch.object(
                prospective, "_command_prereg_id", return_value=prereg.object_id
            ),
            mock.patch.object(
                prospective,
                "_validate_prereg",
                return_value=(prereg, collection, spec),
            ),
            mock.patch.object(prospective, "_domain_objects", return_value={}),
            mock.patch.object(prospective, "_matching_objects", return_value=[]),
        ):
            report = prospective.status(
                prereg.object_id,
                repo_root=root,
                now_epoch=t0_epoch + 700,
            )
        conn = prospective._open_trial(paths["database"], create=False)
        try:
            invalid = prospective._protocol_invalid_payload(
                prereg.object_id,
                detected_epoch=t0_epoch + 701,
                stage="SYNTHETIC_PROTOCOL",
                reason="bound score parity changed",
            )
            prospective._insert_protocol_invalid_once(conn, invalid)
        finally:
            conn.close()
        with (
            mock.patch.object(prospective, "_validated_root", return_value=root),
            mock.patch.object(prospective, "_load_spec", return_value=spec),
            mock.patch.object(
                prospective, "_command_prereg_id", return_value=prereg.object_id
            ),
            mock.patch.object(
                prospective,
                "_validate_prereg",
                return_value=(prereg, collection, spec),
            ),
            mock.patch.object(prospective, "_domain_objects", return_value={}),
            mock.patch.object(prospective, "_matching_objects", return_value=[]),
            mock.patch.object(
                prospective,
                "_read_tick_rows",
                side_effect=GateFailure("INVALID status read tick values"),
            ),
        ):
            invalid_report = prospective.status(
                prereg.object_id,
                repo_root=root,
                now_epoch=t0_epoch + 702,
            )
    equal(report["state"], "SEALED", "sealed status state")
    equal(report["schedule_counts"], counts, "status schedule counts")
    equal(
        report["settlement_availability"],
        {"settled": 1, "unsettled": 0, "not_common_scoreable": 0},
        "status settlement availability",
    )
    equal(report["store_freshness"], None, "sealed status freshness")
    equal(invalid_report["state"], "INVALID", "durable INVALID status")
    equal(invalid_report["stage"], "SYNTHETIC_PROTOCOL", "INVALID status stage")
    encoded = json.dumps(report, sort_keys=True).lower()
    for forbidden in (
        "entry_quote",
        "exit_quote",
        "correctness",
        "accuracy",
        "endpoint",
        "confidence_bound",
        "yield",
        "profit",
    ):
        check(forbidden not in encoded, f"status exposed {forbidden}")


def test_registered_inference_golden_and_terminal_truth_table() -> None:
    spec = _spec()
    names = list(spec["statistics"]["endpoint_order"])
    equal(
        names,
        [
            "candidate_minus_v1_yield",
            "candidate_UP_accuracy_minus_0.541",
            "candidate_DOWN_accuracy_minus_0.541",
        ],
        "registered family order",
    )
    calendar = np.arange(20260101, 20260131, dtype="int64")
    phase = np.arange(len(calendar), dtype="float64")
    endpoints = [
        prospective.refresh_stats.DateRatioEndpoint(
            name,
            calendar,
            np.sin(phase * 0.19 + index * 0.4) + 0.15 + index * 0.02,
            np.full(len(calendar), 10.0 + index),
        )
        for index, name in enumerate(names)
    ]
    inference = prospective.refresh_stats.infer_shadow_family(
        endpoints,
        master_calendar=calendar,
        k_shadow=1,
        bootstrap_seed=spec["statistics"]["bootstrap_seed"],
        block_lengths=spec["statistics"]["block_lengths"],
        bootstrap_replicates=spec["statistics"]["bootstrap_replicates"],
        alpha=spec["statistics"]["alpha"],
    )
    inference_repeat = prospective.refresh_stats.infer_shadow_family(
        endpoints,
        master_calendar=calendar,
        k_shadow=1,
        bootstrap_seed=9003,
        block_lengths=[5, 10, 20],
        bootstrap_replicates=10_000,
        alpha=0.05,
    )
    equal(inference.as_dict(), inference_repeat.as_dict(), "registered inference determinism")
    equal(
        hashlib.sha256(evidence.canonical_json_bytes(inference.as_dict())).hexdigest(),
        "bb07a767ac3806d4146762901fb8a1a606dc902dca1cfd495293e021ee3c24bb",
        "registered inference golden",
    )

    counts = {
        "common_settled": 250,
        "candidate": {"combined": 200, "UP": 100, "DOWN": 100},
        "incumbent": {"combined": 240, "UP": 120, "DOWN": 120},
    }
    positive = {
        name: {"point_estimate": 0.1, "simultaneous_lower_bound": 0.01}
        for name in names
    }
    months = [
        {
            "month": "2026-08",
            "candidate_UP": 5,
            "candidate_DOWN": 5,
            "candidate_combined": 10,
            "candidate_whole_book_yield": 0.0,
        }
    ]
    survivor = prospective._classify_result(
        spec,
        endpoint_results=positive,
        counts=counts,
        complete_months=months,
    )
    equal(survivor["terminal_status"], "SHADOW_SURVIVOR_INACTIVE", "survivor")
    equal(
        survivor["side_statuses"],
        {"UP": "SUPPORTED_SHADOW", "DOWN": "SUPPORTED_SHADOW"},
        "supported sides",
    )

    invalid = prospective._classify_result(
        spec,
        endpoint_results=positive,
        counts=counts,
        complete_months=months,
        protocol_errors=["synthetic clock defect"],
    )
    equal(invalid["terminal_status"], "INVALID", "invalid terminal")
    equal(invalid["side_statuses"], {"UP": "INVALID", "DOWN": "INVALID"}, "invalid sides")

    rejected_endpoints = json.loads(json.dumps(positive))
    rejected_endpoints[names[1]] = {
        "point_estimate": 0.0,
        "simultaneous_lower_bound": -0.1,
    }
    rejected = prospective._classify_result(
        spec,
        endpoint_results=rejected_endpoints,
        counts=counts,
        complete_months=months,
    )
    equal(rejected["terminal_status"], "SHADOW_REJECTED", "rejected terminal")
    equal(rejected["side_statuses"]["UP"], "REJECTED", "rejected UP side")

    bounded = json.loads(json.dumps(positive))
    bounded[names[0]]["simultaneous_lower_bound"] = 0.0
    inconclusive = prospective._classify_result(
        spec,
        endpoint_results=bounded,
        counts=counts,
        complete_months=months,
    )
    equal(inconclusive["terminal_status"], "INCONCLUSIVE", "bound inconclusive")

    sparse_counts = json.loads(json.dumps(counts))
    sparse_counts["candidate"]["DOWN"] = 99
    sparse_counts["candidate"]["combined"] = 199
    sparse = prospective._classify_result(
        spec,
        endpoint_results=positive,
        counts=sparse_counts,
        complete_months=months,
    )
    equal(sparse["terminal_status"], "INCONCLUSIVE", "sparse terminal")
    equal(sparse["side_statuses"]["DOWN"], "INCONCLUSIVE", "sparse DOWN")

    low_activity = json.loads(json.dumps(counts))
    low_activity["common_settled"] = 300
    low_activity["incumbent"] = {"combined": 251, "UP": 126, "DOWN": 125}
    equal(
        prospective._classify_result(
            spec,
            endpoint_results=positive,
            counts=low_activity,
            complete_months=months,
        )["terminal_status"],
        "INCONCLUSIVE",
        "activity screen",
    )
    missing_month_side = json.loads(json.dumps(months))
    missing_month_side[0]["candidate_DOWN"] = 0
    equal(
        prospective._classify_result(
            spec,
            endpoint_results=positive,
            counts=counts,
            complete_months=missing_month_side,
        )["terminal_status"],
        "INCONCLUSIVE",
        "complete-month screen",
    )
    noncomputable = json.loads(json.dumps(positive))
    noncomputable[names[2]] = {
        "point_estimate": None,
        "simultaneous_lower_bound": None,
    }
    equal(
        prospective._classify_result(
            spec,
            endpoint_results=noncomputable,
            counts=counts,
            complete_months=months,
        )["terminal_status"],
        "INCONCLUSIVE",
        "noncomputable endpoint",
    )
    reversed_family = dict(reversed(list(positive.items())))
    with expect_raises(prospective.ProspectiveError, "order"):
        prospective._classify_result(
            spec,
            endpoint_results=reversed_family,
            counts=counts,
            complete_months=months,
        )


def test_packet_reducer_endpoint_definitions() -> None:
    spec = _spec()
    prereg_id = "c" * 64
    t0_epoch = _epoch(datetime(2026, 7, 20, 8, 0, tzinfo=NY))
    prereg = _prereg_fixture(t0_epoch)
    # The reducer keys database rows by the envelope ID, not by an injected ID.
    prereg_id = prereg.object_id
    represented_dates = pd.bdate_range("2026-07-20", periods=20)
    clocks = np.asarray(
        [
            _epoch(
                datetime(
                    int(day.year),
                    int(day.month),
                    int(day.day),
                    8,
                    15 * slot,
                    tzinfo=NY,
                )
            )
            for day in represented_dates
            for slot in range(4)
        ],
        dtype="int64",
    )
    candidate_probabilities = (0.6, 0.4, 0.6, 0.4) * 20
    incumbent_probabilities = (0.3, 0.7, 0.3, 0.7) * 20
    exit_quotes = (1.01, 0.99, 1.00, 1.01) * 20
    decisions = [
        _decision(
            prereg_id,
            int(clock),
            candidate_probability=candidate,
            incumbent_probability=incumbent,
        )
        for clock, candidate, incumbent in zip(
            clocks,
            candidate_probabilities,
            incumbent_probabilities,
            strict=True,
        )
    ]
    for decision in decisions:
        decision["candidate"]["threshold"] = spec["candidate"]["confidence_threshold"]
        decision["incumbent"]["threshold"] = spec["incumbent"]["confidence_threshold"]
    settlement = [
        {
            "prereg_id": prereg_id,
            "decision_close_epoch": int(clock),
            "status": "SETTLED",
            "reason": "",
            "entry_epoch": int(clock) + 1,
            "entry_quote": 1.00,
            "entry_bid": 0.99,
            "entry_ask": 1.01,
            "exit_epoch": int(clock) + 901,
            "exit_quote": exit_quote,
            "exit_bid": exit_quote - 0.01,
            "exit_ask": exit_quote + 0.01,
        }
        for clock, exit_quote in zip(clocks, exit_quotes, strict=True)
    ]
    counts = {
        "common_settled": 80,
        "candidate": {"combined": 80, "UP": 40, "DOWN": 40},
        "incumbent": {"combined": 80, "UP": 40, "DOWN": 40},
    }
    selection = prospective._selection_manifest(settlement)
    cutoff_epoch = int(clocks[-1]) + 60
    seal = {
        "schema": "usdchf-m15-prospective-trial-seal/v1",
        "prereg_id": prereg_id,
        "cutoff_epoch": cutoff_epoch,
        "week": 26,
        "trigger": "WEEK_26_CAP",
        "sealed_epoch": cutoff_epoch + 659,
        "counts": counts,
        "selection": selection,
        "settlement_sha256": prospective._settlement_rows_sha256(settlement),
        "outcome_fields_present": False,
        "activation": False,
    }
    tampered_settlement = json.loads(json.dumps(settlement))
    tampered_settlement[0]["exit_quote"] = 0.98
    equal(
        prospective._selection_manifest(tampered_settlement),
        selection,
        "timestamp-only manifest unexpectedly bound prices",
    )
    check(
        prospective._settlement_rows_sha256(tampered_settlement)
        != seal["settlement_sha256"],
        "immutable settlement digest omitted prices",
    )
    sealed_source = evidence.EvidenceEnvelope.create(
        kind=spec["evidence_kinds"]["sealed_source"],
        payload={
            "cutoff_epoch": cutoff_epoch,
            "week": 26,
            "trigger": "WEEK_26_CAP",
            "outcome_blind_counts": counts,
            "selection": selection,
        },
    )
    conn = sqlite3.connect(":memory:", isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA application_id=1431119105")
    conn.execute("PRAGMA user_version=1")
    prospective._create_trial_schema(conn)
    try:
        prospective._initialize_trial(
            conn,
            {
                "schema": prospective.TRIAL_METADATA_SCHEMA,
                "prereg_id": prereg_id,
                "outcome_fields_present": False,
                "activation": False,
            },
            created_epoch=t0_epoch - 1,
        )
        for decision in decisions:
            prospective._insert_decision_once(conn, decision)
        prospective._insert_trial_seal_once(conn, seal)
        prospective._create_settlement_table(conn)
        conn.executemany(
            """
            INSERT INTO settlement_rows (
                prereg_id, decision_close_epoch, status, reason,
                entry_epoch, entry_quote, entry_bid, entry_ask,
                exit_epoch, exit_quote, exit_bid, exit_ask
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [prospective._settlement_values(row) for row in settlement],
        )
        with mock.patch.object(
            prospective,
            "_expected_decision_epochs",
            return_value=clocks,
        ):
            result = prospective._reduce_packet(
                conn,
                spec=spec,
                prereg=prereg,
                sealed_source=sealed_source,
            )
            repeated = prospective._reduce_packet(
                conn,
                spec=spec,
                prereg=prereg,
                sealed_source=sealed_source,
            )
    finally:
        conn.close()
    equal(result, repeated, "packet reduction determinism")
    equal(list(result["endpoint_results"]), spec["statistics"]["endpoint_order"], "endpoint order")
    endpoints = result["endpoint_results"]
    equal(endpoints["candidate_minus_v1_yield"]["point_estimate"], 0.5, "control endpoint")
    check(
        abs(endpoints["candidate_UP_accuracy_minus_0.541"]["point_estimate"] + 0.041)
        < 1e-12,
        "UP endpoint definition",
    )
    check(
        abs(endpoints["candidate_DOWN_accuracy_minus_0.541"]["point_estimate"] + 0.041)
        < 1e-12,
        "DOWN endpoint definition",
    )
    equal(result["schedule_counts"], counts, "separate schedules")
    equal(result["status"], "INCONCLUSIVE", "below-count cap result")
    equal(result["analysis_access_receipt_id"], None, "reducer invented access receipt")
    check(result["money"] is None and result["activation"] is False, "claim boundary")


def _access_fixture(spec: Mapping[str, Any]) -> tuple[
    evidence.EvidenceEnvelope,
    evidence.EvidenceEnvelope,
    evidence.EvidenceEnvelope,
    evidence.ArtifactRef,
]:
    t0_epoch = _epoch(datetime(2026, 7, 20, 8, 0, tzinfo=NY))
    prereg = _prereg_fixture(t0_epoch)
    collection = evidence.EvidenceEnvelope.create(
        kind=spec["evidence_kinds"]["collection_access_receipt"],
        payload={"prereg_id": prereg.object_id},
        dependencies=[prereg.object_id],
    )
    source_ref = evidence.ArtifactRef(
        "deriv_data/prospective/synthetic/sealed_source.sqlite",
        len(b"packet"),
        hashlib.sha256(b"packet").hexdigest(),
    )
    counts = {
        "common_settled": 0,
        "candidate": {"combined": 0, "UP": 0, "DOWN": 0},
        "incumbent": {"combined": 0, "UP": 0, "DOWN": 0},
    }
    selection = {
        "rows": 0,
        "settled": 0,
        "unsettled": 0,
        "not_common_scoreable": 0,
        "timestamp_manifest_sha256": "0" * 64,
    }
    sealed = evidence.EvidenceEnvelope.create(
        kind=spec["evidence_kinds"]["sealed_source"],
        payload={
            "schema": spec["evidence_kinds"]["sealed_source"],
            "prereg_id": prereg.object_id,
            "collection_access_receipt_id": collection.object_id,
            "source_artifact": source_ref.as_dict(),
            "cutoff_epoch": t0_epoch + 1,
            "trigger": "WEEK_26_CAP",
            "week": 26,
            "outcome_blind_counts": counts,
            "selection": selection,
            "tick_source_descriptor": {},
            "contains_prices": True,
            "efficacy_computed": False,
            "money": None,
            "activation": False,
        },
        dependencies=[collection.object_id],
    )
    return prereg, collection, sealed, source_ref


def test_analysis_receipt_before_read_and_completed_zero_source_read() -> None:
    spec = _spec()
    spec["paths"]["result"] = "results/synthetic-prospective-result.json"
    prereg, collection, sealed, source_ref = _access_fixture(spec)
    reduced = prospective._invalid_result(
        spec, prereg, sealed, "synthetic fixed-analysis defect"
    )
    events: list[str] = []

    def isolated_identity(
        reference: evidence.ArtifactRef,
        *,
        repo_root: str | os.PathLike[str] = REPO_ROOT,
        allow_missing: bool = False,
    ) -> tuple[int, int]:
        del repo_root, allow_missing
        equal(reference, source_ref, "isolated wrong artifact")
        events.append("source_identity")
        return (1, 2)

    def read_for_analysis(
        reference: evidence.ArtifactRef,
        *,
        repo_root: str | os.PathLike[str] = REPO_ROOT,
    ) -> bytes:
        del repo_root
        equal(reference, source_ref, "analysis read wrong artifact")
        events.append("source_read")
        return b"packet"

    def publish(
        envelope: evidence.EvidenceEnvelope,
        *,
        repo_root: str | os.PathLike[str] = REPO_ROOT,
        require_new: bool = False,
    ) -> None:
        del repo_root
        check(require_new, "analysis evidence publication was not strict-new")
        events.append(f"publish:{envelope.kind}")

    fake_conn = mock.Mock()

    def reduce_packet(*args: Any, **kwargs: Any) -> dict[str, Any]:
        del args, kwargs
        events.append("reduce")
        return dict(reduced)

    with tempfile.TemporaryDirectory(prefix="usdchf-prospective-access-") as raw:
        root = Path(raw)
        (root / "results").mkdir()
        with (
            mock.patch.object(prospective, "_validated_root", return_value=root),
            mock.patch.object(prospective, "_load_spec", return_value=spec),
            mock.patch.object(
                prospective, "_command_prereg_id", return_value=prereg.object_id
            ),
            mock.patch.object(
                prospective,
                "_validate_prereg",
                return_value=(prereg, collection, spec),
            ),
            mock.patch.object(prospective, "_domain_objects", return_value={}),
            mock.patch.object(prospective, "_matching_objects", return_value=[]),
            mock.patch.object(
                prospective,
                "_validated_sealed_source",
                return_value=(sealed, source_ref),
            ),
            mock.patch.object(
                evidence.ArtifactRef,
                "isolated_identity",
                new=isolated_identity,
            ),
            mock.patch.object(
                evidence.ArtifactRef,
                "read_verified",
                new=read_for_analysis,
            ),
            mock.patch.object(prospective.evidence, "publish", side_effect=publish),
            mock.patch.object(
                prospective, "_deserialize_packet", return_value=fake_conn
            ),
            mock.patch.object(
                prospective, "_reduce_packet", side_effect=reduce_packet
            ),
        ):
            analyzed = prospective.analyze(prereg.object_id, repo_root=root)
        check(fake_conn.close.called, "analysis did not close the in-memory packet")
        equal(analyzed["status"], "INVALID", "synthetic analysis terminal status")
        analysis_kind = spec["evidence_kinds"]["analysis_access_receipt"]
        terminal_kind = spec["evidence_kinds"]["terminal"]
        equal(
            events,
            [
                "source_identity",
                f"publish:{analysis_kind}",
                "source_read",
                "reduce",
                f"publish:{terminal_kind}",
            ],
            "protected analysis access order",
        )
        result_bytes = (root / spec["paths"]["result"]).read_bytes()
        result = evidence.decode_canonical_json(result_bytes)
        receipt = prospective._expected_analysis_receipt(
            spec, prereg, sealed, source_ref
        )
        equal(
            result["analysis_access_receipt_id"],
            receipt.object_id,
            "result access receipt",
        )
        altered_context = json.loads(json.dumps(result))
        altered_context["look"]["cutoff_epoch"] += 60
        with expect_raises(prospective.ProspectiveError, "context differs"):
            prospective._validate_result_value(
                altered_context,
                spec,
                prereg.object_id,
                sealed.object_id,
                receipt.object_id,
                prereg=prereg,
                sealed=sealed,
            )
        result_path = root / spec["paths"]["result"]
        result_stage = result_path.parent / f".{result_path.name}.building"
        os.link(result_path, result_stage)
        equal(result_path.stat().st_nlink, 2, "result link-window fixture")
        with mock.patch.object(prospective.evidence, "publish") as recovered_publish:
            prospective._publish_result_terminal(
                root,
                spec,
                result,
                require_new_result=False,
                dependency_id=receipt.object_id,
            )
        check(recovered_publish.called, "result link recovery omitted terminal")
        check(not result_stage.exists(), "result recovery stage remains")
        equal(result_path.stat().st_nlink, 1, "result recovery retained hard link")

        result_ref = evidence.ArtifactRef(
            spec["paths"]["result"],
            len(result_bytes),
            hashlib.sha256(result_bytes).hexdigest(),
        )
        terminal = prospective._expected_terminal(spec, result, result_ref)
        verification_events: list[str] = []

        def verify_read(
            reference: evidence.ArtifactRef,
            *,
            repo_root: str | os.PathLike[str] = REPO_ROOT,
        ) -> bytes:
            del repo_root
            if reference.path == source_ref.path:
                raise GateFailure("completed verify read the source packet")
            equal(reference, result_ref, "completed verify read wrong artifact")
            verification_events.append("result_read")
            return result_bytes

        with (
            mock.patch.object(prospective, "_validated_root", return_value=root),
            mock.patch.object(prospective, "_load_spec", return_value=spec),
            mock.patch.object(
                prospective, "_command_prereg_id", return_value=prereg.object_id
            ),
            mock.patch.object(
                prospective,
                "_validate_prereg",
                return_value=(prereg, collection, spec),
            ),
            mock.patch.object(prospective, "_domain_objects", return_value={}),
            mock.patch.object(
                prospective,
                "_validated_sealed_source",
                return_value=(sealed, source_ref),
            ),
            mock.patch.object(
                prospective,
                "_validated_analysis_receipt",
                return_value=receipt,
            ),
            mock.patch.object(
                prospective, "_matching_objects", return_value=[terminal]
            ),
            mock.patch.object(
                evidence.ArtifactRef, "read_verified", new=verify_read
            ),
            mock.patch.object(
                evidence.ArtifactRef,
                "isolated_identity",
                side_effect=GateFailure("completed verify inspected source identity"),
            ),
        ):
            report = prospective.verify_prospective(
                prereg.object_id, repo_root=root
            )
        equal(verification_events, ["result_read"], "completed verification reads")
        equal(report["source_packet_read"], False, "verification source-read claim")
        equal(report["terminal_id"], terminal.object_id, "verified terminal")

    receipt = prospective._expected_analysis_receipt(spec, prereg, sealed, source_ref)
    with tempfile.TemporaryDirectory(prefix="usdchf-prospective-spent-") as raw:
        root = Path(raw)
        (root / "results").mkdir()
        recovery_publications: list[str] = []

        def publish_recovery(
            envelope: evidence.EvidenceEnvelope,
            *,
            repo_root: str | os.PathLike[str] = REPO_ROOT,
            require_new: bool = False,
        ) -> None:
            del repo_root
            check(require_new, "recovery terminal publication was not strict-new")
            recovery_publications.append(envelope.kind)

        with (
            mock.patch.object(prospective, "_validated_root", return_value=root),
            mock.patch.object(prospective, "_load_spec", return_value=spec),
            mock.patch.object(
                prospective, "_command_prereg_id", return_value=prereg.object_id
            ),
            mock.patch.object(
                prospective,
                "_validate_prereg",
                return_value=(prereg, collection, spec),
            ),
            mock.patch.object(prospective, "_domain_objects", return_value={}),
            mock.patch.object(
                prospective, "_persisted_protocol_invalid", return_value=None
            ),
            mock.patch.object(
                prospective,
                "_matching_objects",
                side_effect=[[], [receipt]],
            ),
            mock.patch.object(
                prospective,
                "_validated_sealed_source",
                return_value=(sealed, source_ref),
            ),
            mock.patch.object(
                prospective,
                "_validated_analysis_receipt",
                return_value=receipt,
            ),
            mock.patch.object(
                prospective.evidence, "publish", side_effect=publish_recovery
            ),
            mock.patch.object(
                evidence.ArtifactRef,
                "read_verified",
                side_effect=GateFailure("spent analysis reread source"),
            ),
        ):
            recovered = prospective.analyze(prereg.object_id, repo_root=root)
        equal(recovered["status"], "INVALID", "spent receipt recovery status")
        check(
            "source packet was not reread" in recovered["protocol_errors"][0],
            "spent receipt recovery reason",
        )
        equal(
            recovery_publications,
            [spec["evidence_kinds"]["terminal"]],
            "spent receipt recovery publications",
        )

    invalid = prospective._protocol_invalid_payload(
        prereg.object_id,
        detected_epoch=int(prereg.payload["t0"]["epoch"]) + 1,
        stage="SYNTHETIC_PROTOCOL",
        reason="fixed provider identity changed",
    )
    with tempfile.TemporaryDirectory(prefix="usdchf-prospective-no-source-") as raw:
        root = Path(raw)
        (root / "results").mkdir()
        published: list[evidence.EvidenceEnvelope] = []

        def publish_no_source(
            envelope: evidence.EvidenceEnvelope,
            *,
            repo_root: str | os.PathLike[str] = REPO_ROOT,
            require_new: bool = False,
        ) -> None:
            del repo_root
            check(require_new, "no-source terminal publication was not strict-new")
            published.append(envelope)

        with (
            mock.patch.object(prospective, "_validated_root", return_value=root),
            mock.patch.object(prospective, "_load_spec", return_value=spec),
            mock.patch.object(
                prospective, "_command_prereg_id", return_value=prereg.object_id
            ),
            mock.patch.object(
                prospective,
                "_validate_prereg",
                return_value=(prereg, collection, spec),
            ),
            mock.patch.object(prospective, "_domain_objects", return_value={}),
            mock.patch.object(prospective, "_matching_objects", return_value=[]),
            mock.patch.object(
                prospective, "_persisted_protocol_invalid", return_value=invalid
            ),
            mock.patch.object(
                prospective.evidence, "publish", side_effect=publish_no_source
            ),
            mock.patch.object(
                evidence.ArtifactRef,
                "read_verified",
                side_effect=GateFailure("no-source INVALID read a source packet"),
            ),
        ):
            no_source_result = prospective.analyze(prereg.object_id, repo_root=root)
        equal(no_source_result["status"], "INVALID", "no-source INVALID status")
        equal(no_source_result["sealed_source_id"], None, "no-source sealed ID")
        equal(no_source_result["analysis_access_receipt_id"], None, "no-source access ID")
        equal(len(published), 1, "no-source terminal publication count")
        no_source_terminal = published[0]
        no_source_bytes = (root / spec["paths"]["result"]).read_bytes()
        no_source_ref = evidence.ArtifactRef(
            spec["paths"]["result"],
            len(no_source_bytes),
            hashlib.sha256(no_source_bytes).hexdigest(),
        )

        def no_source_matches(
            groups: Mapping[str, Any], kind: str, prereg_id: str
        ) -> list[evidence.EvidenceEnvelope]:
            del groups, prereg_id
            return (
                [no_source_terminal]
                if kind == spec["evidence_kinds"]["terminal"]
                else []
            )

        def no_source_verify_read(
            reference: evidence.ArtifactRef,
            *,
            repo_root: str | os.PathLike[str] = REPO_ROOT,
        ) -> bytes:
            del repo_root
            equal(reference, no_source_ref, "no-source verify read wrong artifact")
            return no_source_bytes

        with (
            mock.patch.object(prospective, "_validated_root", return_value=root),
            mock.patch.object(prospective, "_load_spec", return_value=spec),
            mock.patch.object(
                prospective, "_command_prereg_id", return_value=prereg.object_id
            ),
            mock.patch.object(
                prospective,
                "_validate_prereg",
                return_value=(prereg, collection, spec),
            ),
            mock.patch.object(prospective, "_domain_objects", return_value={}),
            mock.patch.object(
                prospective, "_matching_objects", side_effect=no_source_matches
            ),
            mock.patch.object(
                prospective, "_persisted_protocol_invalid", return_value=invalid
            ),
            mock.patch.object(
                evidence.ArtifactRef, "read_verified", new=no_source_verify_read
            ),
        ):
            no_source_report = prospective.verify_prospective(
                prereg.object_id, repo_root=root
            )
        equal(no_source_report["source_packet_read"], False, "no-source verify claim")
        equal(no_source_report["sealed_source_id"], None, "no-source verify sealed ID")
        equal(
            no_source_report["analysis_access_receipt_id"],
            None,
            "no-source verify access ID",
        )


def test_collector_lifecycle_recovery_and_no_source_terminal() -> None:
    spec = _spec()
    t0_epoch = _epoch(datetime(2026, 7, 20, 8, 0, tzinfo=NY))
    preflight = _preflight_fixture()
    prereg = evidence.EvidenceEnvelope.create(
        kind=spec["evidence_kinds"]["prereg"],
        payload={
            "h0_git_sha": "a" * 40,
            "t0": {
                "utc": "2026-07-20T12:00:00Z",
                "ny": "2026-07-20T08:00:00-04:00",
                "epoch": t0_epoch,
            },
            "source_paths": {
                "candles": spec["paths"]["candle_store"],
                "tick_shards": spec["paths"]["tick_shards"],
                "tick_progress": spec["paths"]["tick_progress"],
            },
            "preflight": {
                "path": spec["paths"]["preflight_result"],
                "sha256": prospective._sha256_bytes(
                    evidence.canonical_json_bytes(preflight)
                ),
                "payload": preflight,
            },
            "runtime_environment": {"synthetic": "fixed"},
        },
    )
    collection = evidence.EvidenceEnvelope.create(
        kind=spec["evidence_kinds"]["collection_access_receipt"],
        payload={"prereg_id": prereg.object_id},
        dependencies=[prereg.object_id],
    )
    candidate = SimpleNamespace(content_id=preflight["scoring_parity"]["candidate"]["content_id"])
    incumbent = SimpleNamespace(content_id=preflight["scoring_parity"]["incumbent"]["content_id"])
    h1 = "b" * 40

    def signal_controller() -> tuple[dict[int, Any], Any, Any]:
        handlers: dict[int, Any] = {}

        def swap(signum: int, handler: Any) -> Any:
            previous = handlers.get(signum, signal.SIG_DFL)
            handlers[signum] = handler
            return previous

        def stop_sleep(_seconds: float) -> None:
            handlers[signal.SIGTERM](signal.SIGTERM, None)

        return handlers, swap, stop_sleep

    with tempfile.TemporaryDirectory(prefix="usdchf-prospective-lifecycle-") as raw:
        root = Path(raw)
        (root / "results").mkdir()
        paths = prospective._trial_paths(root, spec, prereg.object_id)

        for launch_time in (t0_epoch - 120, t0_epoch - 60):
            _handlers, swap, stop_sleep = signal_controller()
            with (
                mock.patch.object(prospective, "_validated_root", return_value=root),
                mock.patch.object(prospective, "_load_spec", return_value=spec),
                mock.patch.object(
                    prospective, "_command_prereg_id", return_value=prereg.object_id
                ),
                mock.patch.object(
                    prospective,
                    "_assert_write_authority",
                    return_value=(prereg, collection, spec, h1),
                ),
                mock.patch.object(
                    prospective,
                    "authenticate_policies",
                    return_value=(candidate, incumbent, spec),
                ),
                mock.patch.object(prospective.time, "time", return_value=launch_time),
                mock.patch.object(prospective.time, "sleep", side_effect=stop_sleep),
                mock.patch.object(prospective.signal, "signal", side_effect=swap),
                mock.patch.object(
                    prospective,
                    "_read_tick_rows",
                    side_effect=GateFailure("pre-T0 collector read outcomes"),
                ),
            ):
                stopped = prospective.collect(repo_root=root)
            equal(stopped["state"], "STOPPED_BEFORE_SEAL", "pre-T0 stop")

        conn = prospective._open_trial_readonly(paths["database"])
        try:
            created_epoch = int(
                conn.execute(
                    "SELECT created_epoch FROM trial_metadata WHERE singleton=1"
                ).fetchone()[0]
            )
        finally:
            conn.close()
        equal(created_epoch, t0_epoch - 120, "restart rewrote trial metadata")

        boundaries = prospective._week_boundaries(prereg, spec)
        boundary8, boundary9 = boundaries[7][1], boundaries[8][1]
        after_boundary9 = boundary9 + 3 * 86_400
        events: list[tuple[str, int]] = []
        sealed_stub = evidence.EvidenceEnvelope.create(
            kind=spec["evidence_kinds"]["sealed_source"],
            payload={"prereg_id": prereg.object_id},
        )

        def capture_boundary(
            *args: Any,
            now_epoch: int,
            capture_through_epoch: int | None = None,
            **kwargs: Any,
        ) -> int:
            del args, kwargs
            events.append(("capture", now_epoch, capture_through_epoch))
            return 0

        def seal_boundary(
            *args: Any,
            now_epoch: int,
            evaluate_through_epoch: int | None = None,
            **kwargs: Any,
        ) -> Any:
            del args, kwargs
            events.append(("seal", now_epoch, evaluate_through_epoch))
            return sealed_stub if evaluate_through_epoch == boundary9 else None

        _handlers, swap, _stop_sleep = signal_controller()
        with (
            mock.patch.object(prospective, "_validated_root", return_value=root),
            mock.patch.object(prospective, "_load_spec", return_value=spec),
            mock.patch.object(
                prospective, "_command_prereg_id", return_value=prereg.object_id
            ),
            mock.patch.object(
                prospective,
                "_assert_write_authority",
                return_value=(prereg, collection, spec, h1),
            ),
            mock.patch.object(
                prospective,
                "authenticate_policies",
                return_value=(candidate, incumbent, spec),
            ),
            mock.patch.object(prospective.time, "time", return_value=after_boundary9),
            mock.patch.object(
                prospective.time,
                "sleep",
                side_effect=GateFailure("collector slept past due cutoff"),
            ),
            mock.patch.object(prospective.signal, "signal", side_effect=swap),
            mock.patch.object(
                prospective, "_capture_due", side_effect=capture_boundary
            ),
            mock.patch.object(
                prospective, "_seal_if_due", side_effect=seal_boundary
            ),
        ):
            sealed_report = prospective.collect(repo_root=root)
        equal(
            events,
            [
                ("capture", after_boundary9, boundary8),
                ("seal", after_boundary9, boundary8),
                ("capture", after_boundary9, boundary9),
                ("seal", after_boundary9, boundary9),
            ],
            "downtime recovery admitted rows beyond the earliest unresolved boundary",
        )
        equal(
            sealed_report["sealed_source_id"],
            sealed_stub.object_id,
            "cutoff recovery seal",
        )

        zero_counts = {
            "common_settled": 0,
            "candidate": {"combined": 0, "UP": 0, "DOWN": 0},
            "incumbent": {"combined": 0, "UP": 0, "DOWN": 0},
        }
        zero_selection = prospective._selection_manifest([])
        cap_boundary = boundaries[25][1]
        seal_resume_time = cap_boundary + 1
        logical_seal = prospective._seal_payload(
            prereg.object_id,
            {
                "cutoff_epoch": cap_boundary,
                "week": 26,
                "trigger": "WEEK_26_CAP",
                "counts": zero_counts,
                "selection": zero_selection,
            },
            settlement_sha256=prospective._settlement_rows_sha256([]),
            sealed_epoch=cap_boundary,
        )
        conn = prospective._open_trial(paths["database"], create=False)
        try:
            prospective._insert_trial_seal_once(conn, logical_seal)
            descriptor = {
                "source": spec["paths"]["tick_shards"],
                "progress_path": spec["paths"]["tick_progress"],
                "progress_sha256": "1" * 64,
                "last_persisted_epoch": cap_boundary,
                "shards_seen": 0,
                "shard_metadata_sha256": "2" * 64,
            }
            source_was_published = False

            def publish_then_interrupt(**kwargs: Any) -> Any:
                nonlocal source_was_published
                del kwargs
                source_was_published = True
                raise RuntimeError("crash after sealed-source publication")

            with (
                mock.patch.object(
                    prospective,
                    "_read_tick_rows",
                    return_value=(
                        pd.DataFrame(columns=["epoch", "quote", "bid", "ask"]),
                        descriptor,
                    ),
                ),
                mock.patch.object(prospective, "_settlement_rows", return_value=[]),
                mock.patch.object(
                    prospective,
                    "_publish_sealed_source",
                    side_effect=publish_then_interrupt,
                ),
                expect_raises(RuntimeError, "after sealed-source publication"),
            ):
                prospective._complete_seal(
                    root=root,
                    prereg=prereg,
                    collection=collection,
                    spec=spec,
                    conn=conn,
                    paths=paths,
                    seal=logical_seal,
                )
            check(source_was_published, "source-publish crash fixture did not run")
            check(paths["packet"].exists(), "source-publish crash lost final packet")
            with (
                mock.patch.object(
                    prospective,
                    "_read_tick_rows",
                    return_value=(
                        pd.DataFrame(columns=["epoch", "quote", "bid", "ask"]),
                        descriptor,
                    ),
                ),
                mock.patch.object(prospective, "_settlement_rows", return_value=[]),
                mock.patch.object(
                    prospective, "_publish_sealed_source", return_value=sealed_stub
                ),
            ):
                recovered_source = prospective._complete_seal(
                    root=root,
                    prereg=prereg,
                    collection=collection,
                    spec=spec,
                    conn=conn,
                    paths=paths,
                    seal=logical_seal,
                )
            equal(recovered_source, sealed_stub, "sealed-source publication recovery")

            zero_packet_ref = evidence.ArtifactRef(
                prospective._expected_packet_relative(spec, prereg.object_id),
                1,
                hashlib.sha256(b"x").hexdigest(),
            )
            zero_source = evidence.EvidenceEnvelope.create(
                kind=spec["evidence_kinds"]["sealed_source"],
                payload={
                    "schema": spec["evidence_kinds"]["sealed_source"],
                    "prereg_id": prereg.object_id,
                    "collection_access_receipt_id": collection.object_id,
                    "source_artifact": zero_packet_ref.as_dict(),
                    "cutoff_epoch": cap_boundary,
                    "trigger": "WEEK_26_CAP",
                    "week": 26,
                    "outcome_blind_counts": zero_counts,
                    "selection": zero_selection,
                    "tick_source_descriptor": descriptor,
                    "contains_prices": True,
                    "efficacy_computed": False,
                    "money": None,
                    "activation": False,
                },
                dependencies=[collection.object_id],
            )
            validated_zero, validated_zero_ref = prospective._validated_sealed_source(
                root,
                prereg,
                collection,
                spec,
                groups={spec["evidence_kinds"]["sealed_source"]: [zero_source]},
            )
            equal(validated_zero, zero_source, "zero-shard cap source envelope")
            equal(validated_zero_ref, zero_packet_ref, "zero-shard cap packet ref")
        finally:
            conn.close()

        _handlers, swap, _stop_sleep = signal_controller()
        with (
            mock.patch.object(prospective, "_validated_root", return_value=root),
            mock.patch.object(prospective, "_load_spec", return_value=spec),
            mock.patch.object(
                prospective, "_command_prereg_id", return_value=prereg.object_id
            ),
            mock.patch.object(
                prospective,
                "_assert_write_authority",
                return_value=(prereg, collection, spec, h1),
            ),
            mock.patch.object(
                prospective,
                "authenticate_policies",
                return_value=(candidate, incumbent, spec),
            ),
            mock.patch.object(prospective.time, "time", return_value=seal_resume_time),
            mock.patch.object(prospective.signal, "signal", side_effect=swap),
            mock.patch.object(
                prospective,
                "_seal_if_due",
                side_effect=RuntimeError("crash after logical seal"),
            ),
            expect_raises(
                prospective.RetryableSealInterruption,
                "restart to resume",
            ),
        ):
            prospective.collect(repo_root=root)
        conn = prospective._open_trial_readonly(paths["database"])
        try:
            equal(
                prospective._load_protocol_invalid(conn, prereg.object_id),
                None,
                "retryable seal interruption became durable INVALID",
            )
        finally:
            conn.close()
        with mock.patch.object(
            prospective,
            "collect",
            side_effect=prospective.RetryableSealInterruption("synthetic retry"),
        ):
            equal(prospective.main(["collect"]), 1, "retryable main exit status")

        _handlers, swap, _stop_sleep = signal_controller()
        with (
            mock.patch.object(prospective, "_validated_root", return_value=root),
            mock.patch.object(prospective, "_load_spec", return_value=spec),
            mock.patch.object(
                prospective, "_command_prereg_id", return_value=prereg.object_id
            ),
            mock.patch.object(
                prospective,
                "_assert_write_authority",
                return_value=(prereg, collection, spec, h1),
            ),
            mock.patch.object(
                prospective,
                "authenticate_policies",
                return_value=(candidate, incumbent, spec),
            ),
            mock.patch.object(prospective.time, "time", return_value=seal_resume_time),
            mock.patch.object(prospective.signal, "signal", side_effect=swap),
            mock.patch.object(
                prospective, "_seal_if_due", return_value=sealed_stub
            ),
        ):
            resumed_seal = prospective.collect(repo_root=root)
        equal(resumed_seal["state"], "SEALED_SOURCE_READY", "seal retry state")

        def recovery_candidates(
            _root: Path,
            _spec_value: Mapping[str, Any] | None,
            _prereg_id: str | None,
        ) -> list[Any]:
            conn = prospective._open_trial_readonly(paths["database"])
            try:
                invalid = prospective._load_protocol_invalid(conn, prereg.object_id)
            finally:
                conn.close()
            return [(prereg, collection, invalid)]

        drift_time = seal_resume_time + 1
        fatal_source = GateFailure("startup recovery touched a protected source")
        with (
            mock.patch.object(prospective, "_validated_root", return_value=root),
            mock.patch.object(
                prospective,
                "_load_spec",
                side_effect=prospective.ProspectiveError("prospective spec bytes differ"),
            ),
            mock.patch.object(
                prospective, "_metadata_trial_candidates", side_effect=recovery_candidates
            ),
            mock.patch.object(prospective, "_load_h0_spec", return_value=spec),
            mock.patch.object(prospective.time, "time", return_value=drift_time),
            expect_raises(
                prospective.RetryableSealInterruption,
                "after the logical seal",
            ),
        ):
            prospective.collect(repo_root=root)
        conn = prospective._open_trial_readonly(paths["database"])
        try:
            equal(
                prospective._load_protocol_invalid(conn, prereg.object_id),
                None,
                "startup drift invalidated a committed seal",
            )
        finally:
            conn.close()

        prereg = evidence.EvidenceEnvelope.create(
            kind=prereg.kind,
            payload={**prereg.payload, "h0_git_sha": "d" * 40},
        )
        collection = evidence.EvidenceEnvelope.create(
            kind=collection.kind,
            payload={"prereg_id": prereg.object_id},
            dependencies=[prereg.object_id],
        )
        paths = prospective._trial_paths(root, spec, prereg.object_id)
        conn = prospective._open_trial(paths["database"])
        try:
            prospective._initialize_trial(
                conn,
                prospective._metadata_payload(prereg, collection, spec, h1),
                created_epoch=t0_epoch - 120,
            )
        finally:
            conn.close()

        with (
            mock.patch.object(prospective, "_validated_root", return_value=root),
            mock.patch.object(
                prospective,
                "_load_spec",
                side_effect=prospective.ProspectiveError("prospective spec bytes differ"),
            ),
            mock.patch.object(
                prospective, "_metadata_trial_candidates", side_effect=recovery_candidates
            ),
            mock.patch.object(prospective, "_load_h0_spec", return_value=spec),
            mock.patch.object(prospective.time, "time", return_value=drift_time),
            mock.patch.object(prospective, "_read_tick_rows", side_effect=fatal_source),
            mock.patch.object(
                prospective, "authenticate_policies", side_effect=fatal_source
            ),
            mock.patch.object(
                prospective.evidence, "verify_object", side_effect=fatal_source
            ),
            expect_raises(prospective.ProspectiveError, "durably INVALID"),
        ):
            prospective.collect(repo_root=root)
        conn = prospective._open_trial_readonly(paths["database"])
        try:
            invalid = prospective._load_protocol_invalid(conn, prereg.object_id)
        finally:
            conn.close()
        check(invalid is not None, "startup drift did not persist INVALID")
        invalid_before = evidence.canonical_json_bytes(invalid)

        with (
            mock.patch.object(prospective, "_validated_root", return_value=root),
            mock.patch.object(
                prospective,
                "_load_spec",
                side_effect=prospective.ProspectiveError("prospective spec bytes differ"),
            ),
            mock.patch.object(
                prospective, "_metadata_trial_candidates", side_effect=recovery_candidates
            ),
            mock.patch.object(prospective.time, "time", return_value=drift_time + 10),
            expect_raises(prospective.ProspectiveError, "durably INVALID"),
        ):
            prospective.collect(repo_root=root)
        equal(
            evidence.canonical_json_bytes(recovery_candidates(root, None, None)[0][2]),
            invalid_before,
            "startup INVALID retry changed immutable state",
        )

        with (
            mock.patch.object(prospective, "_validated_root", return_value=root),
            mock.patch.object(
                prospective,
                "_load_spec",
                side_effect=prospective.ProspectiveError("prospective spec bytes differ"),
            ),
            mock.patch.object(
                prospective, "_metadata_trial_candidates", side_effect=recovery_candidates
            ),
            mock.patch.object(prospective, "_read_tick_rows", side_effect=fatal_source),
        ):
            invalid_status = prospective.status(
                repo_root=root, now_epoch=drift_time + 10
            )
        equal(invalid_status["state"], "INVALID", "drift status")
        equal(invalid_status["full_authority_verified"], False, "drift authority")
        for forbidden in ("entry_quote", "exit_quote", "correctness", "accuracy", "yield"):
            check(
                forbidden not in json.dumps(invalid_status).lower(),
                f"drift status exposed {forbidden}",
            )

        analysis_spec = json.loads(json.dumps(spec))
        analysis_spec["paths"]["result"] = "results/lifecycle-invalid-result.json"
        published: list[evidence.EvidenceEnvelope] = []

        def publish_terminal(
            envelope: evidence.EvidenceEnvelope,
            *,
            repo_root: str | os.PathLike[str] = REPO_ROOT,
            require_new: bool = False,
        ) -> None:
            del repo_root
            check(require_new, "lifecycle terminal publication was not strict-new")
            published.append(envelope)

        with (
            mock.patch.object(prospective, "_validated_root", return_value=root),
            mock.patch.object(prospective, "_load_spec", return_value=analysis_spec),
            mock.patch.object(
                prospective, "_command_prereg_id", return_value=prereg.object_id
            ),
            mock.patch.object(
                prospective,
                "_validate_prereg",
                return_value=(prereg, collection, analysis_spec),
            ),
            mock.patch.object(prospective, "_domain_objects", return_value={}),
            mock.patch.object(prospective, "_matching_objects", return_value=[]),
            mock.patch.object(
                prospective.evidence, "publish", side_effect=publish_terminal
            ),
            mock.patch.object(
                evidence.ArtifactRef, "read_verified", side_effect=fatal_source
            ),
        ):
            no_source = prospective.analyze(prereg.object_id, repo_root=root)
        equal(no_source["status"], "INVALID", "durable INVALID terminal status")
        equal(no_source["sealed_source_id"], None, "durable INVALID source ID")
        equal(len(published), 1, "durable INVALID terminal count")
        terminal = published[0]
        result_bytes = (root / analysis_spec["paths"]["result"]).read_bytes()

        def matching_terminal(
            _groups: Mapping[str, Any], kind: str, _prereg_id: str
        ) -> list[evidence.EvidenceEnvelope]:
            return [terminal] if kind == analysis_spec["evidence_kinds"]["terminal"] else []

        def read_result(
            _reference: evidence.ArtifactRef,
            *,
            repo_root: str | os.PathLike[str] = REPO_ROOT,
        ) -> bytes:
            del repo_root
            return result_bytes

        def discover_for_resume(
            value: str | None,
            *,
            root: Path,
            spec: Mapping[str, Any],
            state: str,
        ) -> str:
            del root, spec
            if value is None and state == "incomplete":
                raise prospective.ProspectiveError("no incomplete trial")
            return prereg.object_id

        with (
            mock.patch.object(prospective, "_validated_root", return_value=root),
            mock.patch.object(prospective, "_load_spec", return_value=analysis_spec),
            mock.patch.object(
                prospective, "_command_prereg_id", side_effect=discover_for_resume
            ),
            mock.patch.object(
                prospective,
                "_validate_prereg",
                return_value=(prereg, collection, analysis_spec),
            ),
            mock.patch.object(prospective, "_domain_objects", return_value={}),
            mock.patch.object(
                prospective, "_matching_objects", side_effect=matching_terminal
            ),
            mock.patch.object(evidence.ArtifactRef, "read_verified", new=read_result),
        ):
            resumed = prospective.analyze(repo_root=root)
        equal(resumed["status"], "INVALID", "bare completed analyze recovery")
        equal(resumed["source_packet_read"], False, "bare recovery source read")

        empty_prereg = _prereg_fixture(t0_epoch)
        empty_collection = evidence.EvidenceEnvelope.create(
            kind=spec["evidence_kinds"]["collection_access_receipt"],
            payload={"prereg_id": empty_prereg.object_id},
            dependencies=[empty_prereg.object_id],
        )
        empty_paths = prospective._trial_paths(root, spec, empty_prereg.object_id)
        prospective._open_trial(empty_paths["database"]).close()
        with (
            mock.patch.object(prospective, "_validated_root", return_value=root),
            mock.patch.object(prospective, "_load_spec", return_value=spec),
            mock.patch.object(
                prospective, "_command_prereg_id", return_value=empty_prereg.object_id
            ),
            mock.patch.object(
                prospective,
                "_validate_prereg",
                return_value=(empty_prereg, empty_collection, spec),
            ),
            mock.patch.object(prospective, "_domain_objects", return_value={}),
            mock.patch.object(prospective, "_matching_objects", return_value=[]),
            mock.patch.object(prospective, "_read_tick_rows", side_effect=fatal_source),
        ):
            empty_status = prospective.status(
                empty_prereg.object_id,
                repo_root=root,
                now_epoch=t0_epoch + 1,
            )
        equal(empty_status["state"], "MISSED_T0_UNSTARTED", "empty DB status")

        crossing_prereg = evidence.EvidenceEnvelope.create(
            kind=prereg.kind,
            payload={**prereg.payload, "h0_git_sha": "c" * 40},
        )
        crossing_collection = evidence.EvidenceEnvelope.create(
            kind=collection.kind,
            payload={"prereg_id": crossing_prereg.object_id},
            dependencies=[crossing_prereg.object_id],
        )
        crossing_paths = prospective._trial_paths(root, spec, crossing_prereg.object_id)
        _handlers, swap, _stop_sleep = signal_controller()
        with (
            mock.patch.object(prospective, "_validated_root", return_value=root),
            mock.patch.object(prospective, "_load_spec", return_value=spec),
            mock.patch.object(
                prospective,
                "_command_prereg_id",
                return_value=crossing_prereg.object_id,
            ),
            mock.patch.object(
                prospective,
                "_assert_write_authority",
                return_value=(crossing_prereg, crossing_collection, spec, h1),
            ),
            mock.patch.object(
                prospective,
                "authenticate_policies",
                return_value=(candidate, incumbent, spec),
            ),
            mock.patch.object(
                prospective.time,
                "time",
                side_effect=[t0_epoch - 1, t0_epoch - 1, t0_epoch, t0_epoch],
            ),
            mock.patch.object(prospective.signal, "signal", side_effect=swap),
            expect_raises(prospective.ProspectiveError, "did not complete before T0"),
        ):
            prospective.collect(repo_root=root)
        check(
            not crossing_paths["database"].exists(),
            "T0-crossing initialization left a database",
        )


def main() -> int:
    started = time.monotonic()
    timings = {
        "surface": run_section("six-command buy-incapable surface", test_public_surface_and_buy_incapability),
        "spec": run_section("spec, embedded preflight, and environment", test_spec_preflight_embedding_and_environment),
        "preflight_write": run_section("fail-closed fresh preflight write", test_preflight_generation_and_strict_new_validation),
        "parity": run_section("feature, score, clock, and session parity", test_score_feature_clock_and_session_parity),
        "identity": run_section("fixed package identities and private inactive loading", test_fixed_policy_authentication_and_inactive_loading),
        "settlement": run_section("settlement boundaries, ties, and scheduling", test_tick_settlement_boundaries_ties_and_scheduling),
        "ticks": run_section("strict public tick store without repair", test_tick_store_is_public_strict_and_unrepaired),
        "sqlite": run_section("SQLite immutability, lock, WAL backup, and sidecars", test_sqlite_immutability_lock_and_packet),
        "look": run_section("future T0 and fixed 8/100/26 look", test_t0_and_fixed_one_look),
        "h1": run_section("exact two-object H1 boundary", test_exact_h1_boundary),
        "status": run_section("outcome-blind count-only status", test_status_is_count_only),
        "inference": run_section("registered inference golden and truth table", test_registered_inference_golden_and_terminal_truth_table),
        "reducer": run_section("packet reducer endpoint definitions", test_packet_reducer_endpoint_definitions),
        "access": run_section("analysis receipt ordering and zero-source verify", test_analysis_receipt_before_read_and_completed_zero_source_read),
        "lifecycle": run_section("collector lifecycle, recovery, and durable INVALID", test_collector_lifecycle_recovery_and_no_source_terminal),
    }
    print(
        f"[test] PASS total={time.monotonic() - started:.2f}s "
        f"timings={json.dumps(timings, sort_keys=True)} "
        "temporary_mutations_only=true preflight_run=false collection_run=false "
        "real_source_analysis_run=false network=false demo_buys=false real_money=false",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
