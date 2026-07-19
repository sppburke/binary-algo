#!/usr/bin/env python3
"""Synthetic focused gate for the inactive USDCHF current-refit operator.

All writes are confined to temporary repositories.  This gate never opens a
real feature parquet, fits a model, or invokes the live ``seal`` command.
"""

from __future__ import annotations

import builtins
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from contextlib import ExitStack, contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Iterator, Mapping, Sequence
from unittest import mock

import numpy as np
import pandas as pd

import evidence_store as evidence
import m15_book_refresh_adapters as adapters
import usdchf_m15_current_refit_v1 as refit


REPO_ROOT = Path(__file__).resolve().parents[1]
PROTECTED_FIXTURES = {
    "MODEL_REGISTRY.md": b"protected registry\n",
    "books/INDEX.json": b'{"protected":true}\n',
    "results/USDCHF_RESULTS.md": b"protected Tier-2 ledger\n",
    "results/json/m15_book_refresh_joint_replay_result.json": b'{"protected":true}\n',
}
SYNTHETIC_LGB_MODEL = b"""tree
version=v4
num_class=1
num_tree_per_iteration=1
label_index=0
max_feature_idx=0
objective=binary sigmoid:1
feature_names=x
feature_infos=none
tree_sizes=0

Tree=0
num_leaves=1
num_cat=0
split_feature=
split_gain=
threshold=
decision_type=
left_child=
right_child=
leaf_value=0
leaf_weight=0
leaf_count=1
internal_value=
internal_weight=
internal_count=
is_linear=0
shrinkage=1


end of trees

feature_importances:

parameters:
[boosting: gbdt]
[objective: binary]

end of parameters

pandas_categorical:null
"""


class GateFailure(AssertionError):
    pass


class InjectedCrash(RuntimeError):
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
) -> Iterator[None]:
    try:
        yield
    except error as exc:
        if contains is not None and contains.lower() not in str(exc).lower():
            raise GateFailure(
                f"{error.__name__} did not contain {contains!r}: {exc}"
            ) from exc
    else:
        raise GateFailure(f"expected {error.__name__}")


@contextmanager
def _suppress_native_stderr() -> Iterator[None]:
    saved = os.dup(2)
    try:
        with open(os.devnull, "wb", buffering=0) as sink:
            os.dup2(sink.fileno(), 2)
            yield
    finally:
        os.dup2(saved, 2)
        os.close(saved)


def run_section(name: str, function: Callable[[], None]) -> float:
    started = time.monotonic()
    function()
    elapsed = time.monotonic() - started
    print(f"[test] PASS {name} ({elapsed:.2f}s)", flush=True)
    return elapsed


def _git(root: Path, *arguments: str) -> str:
    process = subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if process.returncode:
        raise GateFailure(
            f"git {' '.join(arguments)} failed: {process.stderr.strip()}"
        )
    return process.stdout.strip()


def _make_repo(root: Path, *, include_shared: bool = False) -> None:
    for relative in (
        "scripts",
        "results/json",
        "results/artifacts",
        "evidence/objects",
        "logs",
    ):
        (root / relative).mkdir(parents=True, exist_ok=True)
    shutil.copyfile(REPO_ROOT / refit.SCRIPT_PATH, root / refit.SCRIPT_PATH)
    shutil.copyfile(REPO_ROOT / refit.SPEC_PATH, root / refit.SPEC_PATH)
    shutil.copyfile(REPO_ROOT / "ENVIRONMENT_libs.txt", root / "ENVIRONMENT_libs.txt")
    if include_shared:
        repository_spec = json.loads((REPO_ROOT / refit.SPEC_PATH).read_text())
        for relative in repository_spec["shared_identity_paths"]:
            source = REPO_ROOT / relative
            destination = root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
    for relative, raw in PROTECTED_FIXTURES.items():
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(raw)
    shutil.copyfile(REPO_ROOT / ".gitignore", root / ".gitignore")
    _git(root, "init", "-b", "main")
    _git(root, "config", "user.name", "Synthetic Gate")
    _git(root, "config", "user.email", "synthetic@example.invalid")
    _git(root, "add", ".")
    _git(root, "commit", "-m", "fixture H0")


def _protected_snapshot(root: Path) -> dict[str, tuple[int, str]]:
    return {
        relative: (path.stat().st_mode, hashlib.sha256(path.read_bytes()).hexdigest())
        for relative in PROTECTED_FIXTURES
        for path in [root / relative]
    }


def _custom_spec(h0: str = "0" * 40) -> dict[str, Any]:
    spec = json.loads((REPO_ROOT / refit.SPEC_PATH).read_text(encoding="utf-8"))
    spec["authority"]["reviewed_repo_head"] = h0
    spec["shared_identity_paths"] = []
    return spec


def _runtime_contract(root: Path, h0: str) -> dict[str, Any]:
    return refit._verified_runtime_version_contract(root, h0)


def _source_manifest(spec: Mapping[str, Any], h0: str) -> dict[str, Any]:
    rows = [
        {
            "path": relative,
            "bytes": 1,
            "sha256": hashlib.sha256(relative.encode()).hexdigest(),
            "rows": 1,
            "min_timestamp": "2020-01-01T00:00:00Z",
            "max_timestamp": "2020-01-01T00:00:00Z",
            "required_fields": list(refit.REQUIRED_SOURCE_FIELDS),
        }
        for relative in refit._source_paths(spec)
    ]
    return {
        "schema": refit.SOURCE_SCHEMA,
        "artifact_id": spec["artifact_id"],
        "h0_git_sha": h0,
        "source_count": len(rows),
        "total_bytes": len(rows),
        "fit_year_order": list(spec["fit_year_order"]),
        "calibration_year_order": list(spec["calibration_year_order"]),
        "source_order": rows,
        "raw_sources_tracked": False,
        "raw_sources_are_artifact_refs": False,
    }


def _write_canonical(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(refit._canonical(value))


def test_spec_identity_and_paths() -> None:
    raw = (REPO_ROOT / refit.SPEC_PATH).read_bytes()
    equal(hashlib.sha256(raw).hexdigest(), refit.SPEC_SHA256, "spec SHA-256")
    equal(raw, evidence.canonical_json_bytes(evidence.decode_canonical_json(raw)), "spec canonical bytes")
    spec = refit._load_spec(REPO_ROOT)
    equal((spec["pair"], spec["side"]), ("USDCHF", "combined"), "artifact key")
    check(spec["activation"] is False, "spec gained activation authority")
    for good in ("a", "a/b", "results/json/x.json"):
        equal(refit._canonical_relative(good), good, "canonical path")
    for bad in ("", "/a", "a/../b", "a\\b", "a//b"):
        with expect_raises(refit.RefitError, "canonical relative"):
            refit._canonical_relative(bad)
    h0 = refit._git_head(REPO_ROOT)
    runtime = refit._verified_runtime_version_contract(REPO_ROOT, h0)
    drifted = json.loads(json.dumps(runtime))
    drifted["distributions"]["numpy"] = "0.0.synthetic-drift"
    with expect_raises(refit.RefitError, "runtime version contract"):
        refit._validate_runtime_contract(REPO_ROOT, h0, drifted)
    with tempfile.TemporaryDirectory(prefix="usdchf-refit-directory-") as raw:
        root = Path(raw) / "root"
        outside = Path(raw) / "outside"
        root.mkdir()
        outside.mkdir()
        (root / "logs").symlink_to(outside, target_is_directory=True)
        with expect_raises(refit.RefitError, "symlinked"):
            refit._ensure_real_directory(root, "logs/work/seal")


def _adapter_rows(
    *, purpose: str, timestamps: Sequence[str], moved: Sequence[bool]
) -> adapters.AdapterRows:
    contract = adapters.contract_for("USDCHF")
    entry = pd.DatetimeIndex(timestamps).as_unit("ns").asi8.astype("int64")
    columns = tuple(f"synthetic_{index:03d}" for index in range(contract.n_features))
    return adapters.AdapterRows(
        pair="USDCHF",
        purpose=purpose,
        feature_cols=columns,
        X=np.zeros((len(entry), len(columns)), dtype="float32"),
        y=np.asarray([1 if flag else 0 for flag in moved], dtype="uint8"),
        moved=np.asarray(moved, dtype=bool),
        fit_row_id=entry,
        label_exit_ns=entry + adapters.HORIZON_SECONDS * adapters.NS_PER_SECOND,
        fit_session_mask=np.ones(len(entry), dtype=bool),
        calibration_session_mask=np.ones(len(entry), dtype=bool),
        replay_session_mask=np.ones(len(entry), dtype=bool),
        source_feature_ns=entry
        - adapters.FEATURE_DECISION_SHIFT_SECONDS * adapters.NS_PER_SECOND,
        builder_stride=contract.fit_stride if purpose == "fit" else 1,
    )


def test_exact_partition_boundaries() -> None:
    spec = _custom_spec()
    fit_rows = _adapter_rows(
        purpose="fit",
        timestamps=(
            "2026-03-31T23:29:00Z",
            "2026-03-31T23:45:00Z",
            "2026-04-01T00:00:00Z",
        ),
        moved=(True, True, True),
    )
    calibration_rows = _adapter_rows(
        purpose="calibration",
        timestamps=(
            "2026-03-31T23:45:00Z",
            "2026-04-01T00:00:00Z",
            "2026-05-08T23:44:00Z",
            "2026-05-08T23:45:00Z",
        ),
        moved=(True, False, True, True),
    )
    with (
        mock.patch.object(adapters, "build_fit_rows", return_value=fit_rows),
        mock.patch.object(
            adapters, "build_calibration_rows", return_value=calibration_rows
        ),
        mock.patch.object(refit, "_validate_usdchf_horizon_contract") as horizon,
    ):
        data = refit._build_rows_and_partitions(Path("/not-read"), spec)
    equal(
        [call.kwargs for call in horizon.call_args_list],
        [
            {"require_loaded": False},
            {"require_loaded": True},
            {"require_loaded": True},
        ],
        "horizon checks around builder resolution",
    )
    check(np.array_equal(data["fit_idx"], np.array([0])), "fit strict exit boundary")
    check(
        np.array_equal(data["early_idx"], np.array([1, 2])),
        "early-stopping exact window or tie inclusion",
    )
    check(
        np.array_equal(data["calibration_idx"], np.array([1, 2])),
        "calibration exact window",
    )
    equal(data["partition"]["fit_rows"], 1, "fit row count")
    equal(data["partition"]["calibration_rows"], 2, "calibration row count")
    with (
        mock.patch.dict(os.environ, {"MX_HOR": "5"}),
        mock.patch.object(
            adapters,
            "build_fit_rows",
            side_effect=GateFailure("builder ran under a non-15m horizon"),
        ),
        expect_raises(refit.RefitError, "canonical value 15"),
    ):
        refit._build_rows_and_partitions(Path("/not-read"), spec)
    with (
        mock.patch.dict(
            sys.modules,
            {"usdchf_15m_xpair": SimpleNamespace(HOR=5, GAP_S=300)},
        ),
        expect_raises(refit.RefitError, "non-15m label horizon"),
    ):
        refit._validate_usdchf_horizon_contract(require_loaded=False)


def test_source_manifest_row_contract() -> None:
    spec = _custom_spec()
    h0 = "1" * 40
    valid = _source_manifest(spec, h0)
    refit._validate_source_manifest(valid, spec, h0)

    def rejected(change: Callable[[dict[str, Any]], None], name: str) -> None:
        candidate = json.loads(json.dumps(valid))
        change(candidate)
        with expect_raises(refit.RefitError):
            refit._validate_source_manifest(candidate, spec, h0)

    rejected(lambda value: value["source_order"][0].pop("sha256"), "missing row field")
    rejected(
        lambda value: value["source_order"][0].__setitem__("extra", True),
        "extra row field",
    )
    rejected(
        lambda value: value["source_order"][0].__setitem__("bytes", True),
        "boolean byte count",
    )
    rejected(
        lambda value: value["source_order"][0].__setitem__("sha256", "A" * 64),
        "noncanonical digest",
    )
    rejected(
        lambda value: value["source_order"][0].__setitem__("rows", 0),
        "empty source row",
    )
    rejected(
        lambda value: value["source_order"][0].__setitem__(
            "min_timestamp", "2020-01-02T00:00:00Z"
        ),
        "reversed timestamp range",
    )
    rejected(
        lambda value: value["source_order"][0].__setitem__(
            "max_timestamp", "2020-01-01T00:00:00+00:00"
        ),
        "noncanonical timestamp",
    )
    rejected(
        lambda value: value["source_order"][0].__setitem__(
            "required_fields", list(reversed(refit.REQUIRED_SOURCE_FIELDS))
        ),
        "required-field order",
    )
    rejected(lambda value: value.__setitem__("source_count", True), "boolean count")
    rejected(lambda value: value.__setitem__("total_bytes", True), "boolean total")


def test_seal_crash_recovery() -> None:
    with tempfile.TemporaryDirectory(prefix="usdchf-refit-seal-") as raw:
        root = Path(raw)
        _make_repo(root)
        h0 = _git(root, "rev-parse", "HEAD")
        spec = _custom_spec(h0)
        manifest = _source_manifest(spec, h0)
        protected = _protected_snapshot(root)

        patches = (
            mock.patch.object(refit, "_load_spec", return_value=spec),
            mock.patch.object(refit, "_require_main_at_origin", return_value=h0),
            mock.patch.object(refit, "_is_ancestor", return_value=True),
            mock.patch.object(refit, "_build_source_manifest", return_value=manifest),
        )
        with patches[0], patches[1], patches[2], patches[3]:
            def crash_after_manifest(point: str) -> None:
                if point == "after_source_manifest":
                    raise InjectedCrash(point)

            with expect_raises(InjectedCrash, "after_source_manifest"):
                refit.seal(repo_root=root, fault=crash_after_manifest)
            equal(
                set(refit._git_status(root)),
                {spec["source_manifest_path"]},
                "recoverable post-manifest state",
            )

            def crash_after_seal(point: str) -> None:
                if point == "after_execution_seal":
                    raise InjectedCrash(point)

            with expect_raises(InjectedCrash, "after_execution_seal"):
                refit.seal(repo_root=root, fault=crash_after_seal)
            evidence_names = list((root / "evidence/objects").glob("*.json"))
            equal(len(evidence_names), 1, "recoverable post-seal object count")
            real_import = builtins.__import__

            def guarded_import(name: str, *args: Any, **kwargs: Any) -> Any:
                if name in {
                    "m15_book_refresh_adapters",
                    "usdchf_15m_xpair_frozen",
                    "usdchf_15m_xpair",
                }:
                    raise GateFailure(f"seal imported label-bearing builder: {name}")
                return real_import(name, *args, **kwargs)

            with mock.patch("builtins.__import__", side_effect=guarded_import):
                envelope = refit.seal(repo_root=root)
            recovered = refit.seal(repo_root=root)
        equal(recovered.object_id, envelope.object_id, "seal retry identity")
        equal(
            set(refit._git_status(root)),
            {
                spec["source_manifest_path"],
                f"evidence/objects/{envelope.object_id}.json",
            },
            "exact recovered pre-H1 state",
        )
        equal(_protected_snapshot(root), protected, "seal modified protected paths")


def _pre_h1_snapshot(root: Path, spec: Mapping[str, Any]) -> dict[str, Any]:
    source = root / str(spec["source_manifest_path"])
    return {
        "source": source.read_bytes() if source.exists() else None,
        "objects": {
            path.name: path.read_bytes()
            for path in sorted((root / "evidence/objects").glob("*"))
        },
        "status": refit._git_status(root),
        "protected": _protected_snapshot(root),
    }


def test_seal_foreign_states_fail_without_mutation() -> None:
    for case in ("foreign_source", "foreign_object", "staged_source"):
        with tempfile.TemporaryDirectory(prefix=f"usdchf-refit-seal-{case}-") as raw:
            root = Path(raw)
            _make_repo(root)
            h0 = _git(root, "rev-parse", "HEAD")
            spec = _custom_spec(h0)
            manifest = _source_manifest(spec, h0)
            if case in {"foreign_source", "staged_source"}:
                path = root / spec["source_manifest_path"]
            else:
                path = root / "evidence/objects" / f"{'f' * 64}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(
                refit._canonical(manifest)
                if case == "staged_source"
                else b"foreign-pre-H1-bytes"
            )
            path.chmod(0o444)
            if case == "staged_source":
                _git(root, "add", spec["source_manifest_path"])
            before = _pre_h1_snapshot(root, spec)
            with (
                mock.patch.object(refit, "_load_spec", return_value=spec),
                mock.patch.object(refit, "_require_main_at_origin", return_value=h0),
                mock.patch.object(refit, "_is_ancestor", return_value=True),
                mock.patch.object(
                    refit, "_build_source_manifest", return_value=manifest
                ),
                expect_raises(refit.RefitError),
            ):
                refit.seal(repo_root=root)
            equal(
                _pre_h1_snapshot(root, spec),
                before,
                f"{case} rejection mutated pre-H1 state",
            )


def _create_sealed_h1(
    root: Path, *, extra_h1_path: str | None = None
) -> tuple[dict[str, Any], evidence.EvidenceEnvelope, str]:
    _make_repo(root)
    h0 = _git(root, "rev-parse", "HEAD")
    spec = _custom_spec(h0)
    source = _source_manifest(spec, h0)
    _write_canonical(root / spec["source_manifest_path"], source)
    seal = refit._expected_seal(root, spec, source, h0, _runtime_contract(root, h0))
    evidence.publish(seal, repo_root=root, require_new=True)
    paths = [
        spec["source_manifest_path"],
        f"evidence/objects/{seal.object_id}.json",
    ]
    if extra_h1_path is not None:
        extra = root / extra_h1_path
        extra.parent.mkdir(parents=True, exist_ok=True)
        extra.write_text("foreign H1 path\n", encoding="utf-8")
        paths.append(extra_h1_path)
    _git(root, "add", *paths)
    _git(root, "commit", "-m", "fixture H1")
    h1 = _git(root, "rev-parse", "HEAD")
    _git(root, "update-ref", "refs/remotes/origin/main", h1)
    return spec, seal, h1


def test_h1_authentication() -> None:
    with tempfile.TemporaryDirectory(prefix="usdchf-refit-h1-exact-") as raw:
        root = Path(raw)
        spec, seal, h1 = _create_sealed_h1(root)
        equal(
            refit._authenticate_h1(root, spec, seal),
            (h1, str(seal.payload["h0_git_sha"])),
            "exact H1 authentication",
        )
    with tempfile.TemporaryDirectory(prefix="usdchf-refit-h1-foreign-") as raw:
        root = Path(raw)
        spec, seal, _ = _create_sealed_h1(root, extra_h1_path="foreign.txt")
        with expect_raises(refit.RefitError, "paths differ"):
            refit._authenticate_h1(root, spec, seal)


def test_source_availability_states() -> None:
    with tempfile.TemporaryDirectory(prefix="usdchf-refit-sources-") as raw:
        root = Path(raw)
        spec = _custom_spec()
        source = _source_manifest(spec, "1" * 40)
        equal(refit._source_status(root, spec, source), "unavailable", "absent sources")

        first = root / source["source_order"][0]["path"]
        first.parent.mkdir(parents=True, exist_ok=True)
        first.write_bytes(b"synthetic")
        with expect_raises(refit.RefitError, "partially present"):
            refit._source_status(root, spec, source)

        for row in source["source_order"][1:]:
            path = root / row["path"]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"synthetic")
        with mock.patch.object(refit, "_build_source_manifest", return_value=source):
            equal(refit._source_status(root, spec, source), "match", "matching sources")
        changed = json.loads(json.dumps(source))
        changed["source_order"][0]["sha256"] = "e" * 64
        with (
            mock.patch.object(refit, "_build_source_manifest", return_value=changed),
            expect_raises(refit.RefitError, "identity differs"),
        ):
            refit._source_status(root, spec, source)


def test_cross_process_research_lock() -> None:
    with tempfile.TemporaryDirectory(prefix="usdchf-refit-lock-") as raw:
        root = Path(raw)
        (root / "logs").mkdir()
        spec = {"lock_path": "logs/research_campaign/run_campaign.lock"}
        child = """
import pathlib, sys
import usdchf_m15_current_refit_v1 as refit
spec = {"lock_path": "logs/research_campaign/run_campaign.lock"}
try:
    with refit._research_lock(pathlib.Path(sys.argv[1]), spec):
        raise SystemExit(3)
except refit.RefitError as exc:
    if "another repository research operation is active" not in str(exc):
        raise
"""
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(REPO_ROOT / "scripts")
        with refit._research_lock(root, spec):
            process = subprocess.run(
                [sys.executable, "-c", child, str(root)],
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=environment,
            )
            equal(process.returncode, 0, "concurrent lock rejection")
        lock = root / spec["lock_path"]
        equal(lock.stat().st_mode & 0o777, 0o600, "lock mode")
        with refit._research_lock(root, spec):
            pass


def _phase_spec(root: Path) -> dict[str, Any]:
    spec = _custom_spec()
    spec["workspace_path"] = "logs/work"
    spec["source_manifest_path"] = "results/json/source.json"
    spec["model"]["seed_order"] = [0]
    spec["package_files"] = [
        "s0.txt",
        "unused-s1.txt",
        "unused-s2.txt",
        "strategy.json",
        "resolved.json",
        "manifest.json",
    ]
    _write_canonical(root / spec["source_manifest_path"], {"source": "synthetic"})
    return spec


def test_checkpoint_orphan_and_adoption() -> None:
    with tempfile.TemporaryDirectory(prefix="usdchf-refit-checkpoint-") as raw:
        root = Path(raw)
        spec = _phase_spec(root)
        rows = SimpleNamespace(
            feature_cols=("x",),
            fit_row_id=np.array([1], dtype="int64"),
            X=np.zeros((1, 1), dtype="float32"),
        )
        partition = {
            "fit_row_id_sha256": "1" * 64,
            "early_stopping_row_id_sha256": "2" * 64,
            "fit_label_u8_sha256": "3" * 64,
            "early_stopping_label_u8_sha256": "4" * 64,
        }
        model = root / "logs/work" / ("a" * 64) / "canonical/models/USDCHF_canonical_s0_lgb.txt"
        model.parent.mkdir(parents=True)
        model.write_bytes(b"not-a-lightgbm-model")
        with _suppress_native_stderr():
            with expect_raises(refit.RefitError, "not authentic"):
                refit._authenticate_checkpoint_orphan(
                    model,
                    is_model=True,
                    expected_context={},
                    feature_count=1,
                )
        model.write_bytes(SYNTHETIC_LGB_MODEL)
        refit._authenticate_checkpoint_orphan(
            model,
            is_model=True,
            expected_context={},
            feature_count=1,
        )
        fit_calls: list[int] = []

        def fake_fit(
            pair: str,
            seed: int,
            fit_rows: Any,
            fit_idx: Any,
            validation_rows: Any,
            early_idx: Any,
            model_path: Path,
            **kwargs: Any,
        ) -> adapters.SeedCheckpoint:
            del fit_rows, fit_idx, validation_rows, early_idx, kwargs
            equal(pair, "USDCHF", "checkpoint pair")
            check(not model_path.exists(), "owned orphan was not removed before refit")
            model_path.write_bytes(b"synthetic-model")
            digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
            fit_calls.append(seed)
            return adapters.SeedCheckpoint(pair, seed, 7, model_path, digest)

        def fake_validate(
            model_path: Path,
            sidecar_path: Path,
            *,
            expected_context: Mapping[str, Any],
            feature_count: int,
        ) -> adapters.SeedCheckpoint:
            check(sidecar_path.exists(), "checkpoint sidecar missing")
            equal(feature_count, 1, "checkpoint feature count")
            return adapters.SeedCheckpoint(
                "USDCHF",
                int(expected_context["seed"]),
                7,
                model_path,
                hashlib.sha256(model_path.read_bytes()).hexdigest(),
            )

        calibration = SimpleNamespace(
            target_coverage=0.01,
            confidence_threshold=0.4,
            calibration_rows=1,
            confidence_rows=1,
            structural_column=None,
            structural_quantile=None,
            structural_threshold=None,
        )
        shared = (
            mock.patch.object(adapters, "model_parameters", return_value={"seed": 0}),
            mock.patch.object(adapters, "fit_seed_checkpoint", side_effect=fake_fit),
            mock.patch.object(
                adapters, "predict_checkpoint_mean", return_value=np.array([0.9])
            ),
            mock.patch.object(adapters, "calibrate_policy", return_value=calibration),
            mock.patch.object(refit, "_validate_checkpoint", side_effect=fake_validate),
        )
        with shared[0], shared[1], shared[2], shared[3], shared[4]:
            first = refit._fit_phase(
                root=root,
                spec=spec,
                seal_id="a" * 64,
                h1="b" * 40,
                phase="canonical",
                fit_rows=rows,
                fit_idx=np.array([0]),
                validation_rows=rows,
                early_idx=np.array([0]),
                calibration_idx=np.array([0]),
                partition=partition,
                fault=None,
            )
        equal(fit_calls, [0], "orphan refit count")
        equal(first["checkpoints"][0].model_path.read_bytes(), b"synthetic-model", "checkpoint bytes")

        with (
            mock.patch.object(adapters, "model_parameters", return_value={"seed": 0}),
            mock.patch.object(
                adapters,
                "fit_seed_checkpoint",
                side_effect=GateFailure("adopted checkpoint was refit"),
            ),
            mock.patch.object(
                adapters, "predict_checkpoint_mean", return_value=np.array([0.9])
            ),
            mock.patch.object(adapters, "calibrate_policy", return_value=calibration),
            mock.patch.object(refit, "_validate_checkpoint", side_effect=fake_validate),
        ):
            second = refit._fit_phase(
                root=root,
                spec=spec,
                seal_id="a" * 64,
                h1="b" * 40,
                phase="canonical",
                fit_rows=rows,
                fit_idx=np.array([0]),
                validation_rows=rows,
                early_idx=np.array([0]),
                calibration_idx=np.array([0]),
                partition=partition,
                fault=None,
            )
        equal(second["strategy_bytes"], first["strategy_bytes"], "checkpoint adoption")


def _synthetic_fit_data() -> dict[str, Any]:
    rows = SimpleNamespace(
        feature_cols=("x0", "x1"),
        fit_row_id=np.array([1, 2, 3], dtype="int64"),
        X=np.zeros((3, 2), dtype="float32"),
    )
    return {
        "fit_rows": rows,
        "fit_idx": np.array([0, 1], dtype="int64"),
        "validation_rows": rows,
        "early_idx": np.array([0, 1, 2], dtype="int64"),
        "calibration_idx": np.array([0, 1, 2], dtype="int64"),
        "partition": {
            "fit_rows": 2,
            "early_stopping_rows": 3,
            "calibration_rows": 3,
            "fit_row_id_sha256": "1" * 64,
            "early_stopping_row_id_sha256": "2" * 64,
            "calibration_row_id_sha256": "2" * 64,
            "fit_label_u8_sha256": "3" * 64,
            "early_stopping_label_u8_sha256": "4" * 64,
            "calibration_label_u8_sha256": "4" * 64,
        },
    }


@contextmanager
def _mocked_public_fit(
    spec: Mapping[str, Any], *, mismatch: bool = False
) -> Iterator[list[tuple[str, int]]]:
    calls: list[tuple[str, int]] = []
    data = _synthetic_fit_data()

    def parameters(pair: str, seed: int, *, n_jobs: int) -> dict[str, Any]:
        equal((pair, n_jobs), ("USDCHF", spec["model"]["num_threads"]), "fit parameters")
        return {"seed": seed, "num_threads": n_jobs}

    def fit_seed(
        pair: str,
        seed: int,
        fit_rows: Any,
        fit_idx: Any,
        validation_rows: Any,
        validation_idx: Any,
        model_path: Path,
        **kwargs: Any,
    ) -> adapters.SeedCheckpoint:
        del fit_rows, fit_idx, validation_rows, validation_idx, kwargs
        equal(pair, "USDCHF", "public fit pair")
        phase = "canonical" if "_canonical_" in model_path.name else "audit"
        calls.append((phase, seed))
        suffix = b"-mismatch" if mismatch and phase == "audit" and seed == 1 else b""
        model_path.write_bytes(f"synthetic-seed-{seed}".encode() + suffix)
        digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
        return adapters.SeedCheckpoint(pair, seed, seed + 1, model_path, digest)

    def validate_checkpoint(
        model_path: Path,
        sidecar_path: Path,
        *,
        expected_context: Mapping[str, Any],
        feature_count: int,
    ) -> adapters.SeedCheckpoint:
        check(sidecar_path.is_file(), "public-fit checkpoint sidecar missing")
        equal(feature_count, 2, "public-fit feature count")
        seed = int(expected_context["seed"])
        return adapters.SeedCheckpoint(
            "USDCHF",
            seed,
            seed + 1,
            model_path,
            hashlib.sha256(model_path.read_bytes()).hexdigest(),
        )

    calibration = SimpleNamespace(
        target_coverage=0.01,
        confidence_threshold=0.4,
        calibration_rows=3,
        confidence_rows=1,
        structural_column=None,
        structural_quantile=None,
        structural_threshold=None,
    )
    with ExitStack() as stack:
        stack.enter_context(mock.patch.object(refit, "_load_spec", return_value=spec))
        source_status = stack.enter_context(
            mock.patch.object(refit, "_source_status", return_value="match")
        )
        stack.enter_context(
            mock.patch.object(refit, "_build_rows_and_partitions", return_value=data)
        )
        stack.enter_context(
            mock.patch.object(refit, "_validate_checkpoint", side_effect=validate_checkpoint)
        )
        stack.enter_context(mock.patch.object(adapters, "model_parameters", side_effect=parameters))
        stack.enter_context(mock.patch.object(adapters, "fit_seed_checkpoint", side_effect=fit_seed))
        stack.enter_context(
            mock.patch.object(
                adapters,
                "predict_checkpoint_mean",
                return_value=np.array([0.9, 0.8, 0.7]),
            )
        )
        stack.enter_context(
            mock.patch.object(adapters, "calibrate_policy", return_value=calibration)
        )
        yield calls
        check(
            source_status.call_count >= 2 and source_status.call_count % 2 == 0,
            "source identity was not checked before and after each matrix build",
        )


def test_public_fit_serial_success_and_mismatch() -> None:
    expected_calls = [
        (phase, seed) for phase in ("canonical", "audit") for seed in (0, 1, 2)
    ]
    for mismatch, expected_status in (
        (False, refit.SUCCESS),
        (True, refit.NO_CANDIDATE),
    ):
        with tempfile.TemporaryDirectory(
            prefix=f"usdchf-refit-fit-{'mismatch' if mismatch else 'equal'}-"
        ) as raw:
            root = Path(raw)
            spec, seal, h1 = _create_sealed_h1(root)
            protected = _protected_snapshot(root)
            with _mocked_public_fit(spec, mismatch=mismatch) as calls:
                terminal = refit.fit(seal.object_id, repo_root=root)
            equal(calls, expected_calls, "three-seed canonical/audit serial order")
            equal(terminal.payload["status"], expected_status, "fit terminal status")
            result = refit._load_result(root, spec)
            equal(result["h1_git_sha"], h1, "fit result H1")
            equal(result["no_efficacy_measurement"], True, "fit claim limit")
            equal(_protected_snapshot(root), protected, "fit modified protected paths")
            package = root / spec["package_path"]
            equal(package.exists(), not mismatch, "fit package outcome")
            if mismatch:
                equal(
                    result["comparisons"],
                    {
                        "model_bytes_equal": [True, False, True],
                        "strategy_bytes_equal": False,
                        "resolved_spec_bytes_equal": False,
                    },
                    "audit mismatch evidence",
                )

    with tempfile.TemporaryDirectory(prefix="usdchf-refit-foreign-workspace-") as raw:
        root = Path(raw)
        spec, seal, _ = _create_sealed_h1(root)
        phase = root / spec["workspace_path"] / seal.object_id / "canonical"
        (phase / "models").mkdir(parents=True)
        foreign = phase / "foreign.txt"
        foreign.write_bytes(b"foreign")
        with _mocked_public_fit(spec) as calls:
            with expect_raises(refit.RefitError, "foreign state"):
                refit.fit(seal.object_id, repo_root=root)
        equal(calls, [], "foreign ignored workspace state allowed fitting")
        equal(foreign.read_bytes(), b"foreign", "foreign workspace state was mutated")

    for case in ("future_seed", "audit_before_canonical"):
        with tempfile.TemporaryDirectory(prefix=f"usdchf-refit-reordered-{case}-") as raw:
            root = Path(raw)
            spec, seal, _ = _create_sealed_h1(root)
            if case == "future_seed":
                models = (
                    root
                    / spec["workspace_path"]
                    / seal.object_id
                    / "canonical/models"
                )
                models.mkdir(parents=True)
                name = "USDCHF_canonical_s1_lgb.txt"
                (models / name).write_bytes(b"future-model")
                (models / f"{name}.checkpoint.json").write_bytes(b"future-sidecar")
            else:
                models = (
                    root / spec["workspace_path"] / seal.object_id / "audit/models"
                )
                models.mkdir(parents=True)
            before = {
                path.relative_to(root).as_posix(): path.read_bytes()
                for path in models.iterdir()
                if path.is_file()
            }
            with _mocked_public_fit(spec) as calls:
                with expect_raises(refit.RefitError):
                    refit.fit(seal.object_id, repo_root=root)
            equal(calls, [], f"{case} checkpoint state allowed fitting")
            after = {
                path.relative_to(root).as_posix(): path.read_bytes()
                for path in models.iterdir()
                if path.is_file()
            }
            equal(after, before, f"{case} checkpoint state was mutated")

    with tempfile.TemporaryDirectory(prefix="usdchf-refit-foreign-stage-") as raw:
        root = Path(raw)
        spec, seal, _ = _create_sealed_h1(root)
        stage = refit._package_stage_path(root, spec, seal.object_id)
        stage.mkdir(parents=True)
        planted = stage / spec["package_files"][0]
        planted.write_bytes(b"foreign-stage-bytes")
        with (
            mock.patch.object(refit, "_load_spec", return_value=spec),
            mock.patch.object(
                refit,
                "_source_status",
                side_effect=GateFailure("foreign stage reached source access"),
            ),
            expect_raises(refit.RefitError, "foreign"),
        ):
            refit.fit(seal.object_id, repo_root=root)
        equal(planted.read_bytes(), b"foreign-stage-bytes", "foreign stage was mutated")


def test_public_fit_crash_recovery_boundaries() -> None:
    expected_calls = [
        (phase, seed) for phase in ("canonical", "audit") for seed in (0, 1, 2)
    ]
    fault_points = [
        *(f"after_canonical_seed_{seed}" for seed in (0, 1, 2)),
        *(f"after_audit_seed_{seed}" for seed in (0, 1, 2)),
        "after_audit",
        "after_package_install",
        "after_result",
        "after_terminal",
    ]
    for target in fault_points:
        with tempfile.TemporaryDirectory(prefix=f"usdchf-refit-crash-{target}-") as raw:
            root = Path(raw)
            spec, seal, _ = _create_sealed_h1(root)

            def crash(point: str) -> None:
                if point == target:
                    raise InjectedCrash(point)

            with _mocked_public_fit(spec) as calls:
                with expect_raises(InjectedCrash, target):
                    refit.fit(seal.object_id, repo_root=root, fault=crash)
                terminal = refit.fit(seal.object_id, repo_root=root)
            equal(calls, expected_calls, f"{target} refit or ordering drift")
            equal(terminal.payload["status"], refit.SUCCESS, f"{target} recovery")
            check((root / spec["result_path"]).is_file(), f"{target} result missing")
            check((root / spec["package_path"]).is_dir(), f"{target} package missing")

    with tempfile.TemporaryDirectory(prefix="usdchf-refit-lost-proof-") as raw:
        root = Path(raw)
        spec, seal, _ = _create_sealed_h1(root)

        def crash_after_package(point: str) -> None:
            if point == "after_package_install":
                raise InjectedCrash(point)

        with _mocked_public_fit(spec) as calls:
            with expect_raises(InjectedCrash, "after_package_install"):
                refit.fit(seal.object_id, repo_root=root, fault=crash_after_package)
            shutil.rmtree(root / spec["workspace_path"])
            with expect_raises(refit.RefitError, "post-fit recovery lacks"):
                refit.fit(seal.object_id, repo_root=root)
        equal(calls, expected_calls, "installed package triggered a forbidden refit")

    with tempfile.TemporaryDirectory(prefix="usdchf-refit-symlinked-proof-") as raw:
        root = Path(raw)
        spec, seal, _ = _create_sealed_h1(root)

        def crash_after_package(point: str) -> None:
            if point == "after_package_install":
                raise InjectedCrash(point)

        with _mocked_public_fit(spec) as calls:
            with expect_raises(InjectedCrash, "after_package_install"):
                refit.fit(seal.object_id, repo_root=root, fault=crash_after_package)
            workspace = root / spec["workspace_path"]
            outside = root / "logs/outside-refit-proof"
            workspace.rename(outside)
            workspace.symlink_to(outside, target_is_directory=True)
            with expect_raises(refit.RefitError, "post-fit recovery lacks"):
                refit.fit(seal.object_id, repo_root=root)
        equal(calls, expected_calls, "symlinked proof workspace triggered a refit")

    with tempfile.TemporaryDirectory(prefix="usdchf-refit-ignored-stage-") as raw:
        root = Path(raw)
        spec, seal, _ = _create_sealed_h1(root)

        def crash_after_audit(point: str) -> None:
            if point == "after_audit":
                raise InjectedCrash(point)

        with _mocked_public_fit(spec) as calls:
            with expect_raises(InjectedCrash, "after_audit"):
                refit.fit(seal.object_id, repo_root=root, fault=crash_after_audit)
            stage = refit._package_stage_path(root, spec, seal.object_id)
            stage.mkdir(parents=True)
            ignored = stage / "foreign.pt"
            ignored.write_bytes(b"ignored-foreign-stage")
            with expect_raises(refit.RefitError, "pre-existing package stage"):
                refit.fit(seal.object_id, repo_root=root)
        equal(calls, expected_calls, "ignored stage triggered a forbidden refit")
        equal(ignored.read_bytes(), b"ignored-foreign-stage", "ignored stage was mutated")


def _package_fixture(
    root: Path, spec: Mapping[str, Any], seal_id: str, h1: str
) -> dict[str, Any]:
    workspace = root / str(spec["workspace_path"]) / seal_id / "canonical/models"
    workspace.mkdir(parents=True, exist_ok=True)
    checkpoints = []
    model_rows = []
    for position, name in enumerate(spec["package_files"][:3]):
        path = workspace / f"canonical-{position}.txt"
        path.write_bytes(f"synthetic-model-{position}\n".encode())
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        checkpoints.append(
            adapters.SeedCheckpoint("USDCHF", position, position + 1, path, digest)
        )
        model_rows.append(
            {
                "file": name,
                "seed": position,
                "bytes": path.stat().st_size,
                "sha256": digest,
                "best_iteration": position + 1,
            }
        )
    strategy = {
        "schema": refit.STRATEGY_SCHEMA,
        "artifact_id": spec["artifact_id"],
        "seal_id": seal_id,
        "h1_git_sha": h1,
        "activation": False,
        "models": model_rows,
    }
    resolved = {
        "schema": refit.RESOLVED_SCHEMA,
        "artifact_id": spec["artifact_id"],
        "seal_id": seal_id,
        "h1_git_sha": h1,
        "activation": False,
        "source_manifest_sha256": hashlib.sha256(b"source").hexdigest(),
        "partition": {"fit_rows": 1},
        "models": model_rows,
        "strategy_sha256": hashlib.sha256(refit._canonical(strategy)).hexdigest(),
        "calibration": {"target_coverage": 0.01},
    }
    return {
        "checkpoints": checkpoints,
        "strategy": strategy,
        "strategy_bytes": refit._canonical(strategy),
        "resolved": resolved,
        "resolved_bytes": refit._canonical(resolved),
    }


def test_package_noreplace_and_terminal_semantics() -> None:
    with tempfile.TemporaryDirectory(prefix="usdchf-refit-package-") as raw:
        root = Path(raw)
        (root / "results/json").mkdir(parents=True)
        (root / "results/artifacts").mkdir(parents=True)
        spec = _custom_spec()
        spec["workspace_path"] = "logs/work"
        spec["package_path"] = "results/artifacts/candidate"
        spec["result_path"] = "results/json/result.json"
        spec["package_files"] = [
            "s0.txt", "s1.txt", "s2.txt", "strategy.json", "resolved.json", "manifest.json"
        ]
        seal_id, h1 = "a" * 64, "b" * 40
        phase = _package_fixture(root, spec, seal_id, h1)
        package = refit._install_package(
            root=root,
            spec=spec,
            seal_id=seal_id,
            h1=h1,
            canonical_phase=phase,
        )
        equal(
            package["directory"].stat().st_mode & 0o777,
            0o555,
            "package publication mode",
        )
        first_inode = package["directory"].stat().st_ino
        retried = refit._install_package(
            root=root,
            spec=spec,
            seal_id=seal_id,
            h1=h1,
            canonical_phase=phase,
        )
        equal(retried["directory"].stat().st_ino, first_inode, "package retry replaced destination")
        result = refit._success_result(
            spec=spec, seal_id=seal_id, h1=h1, package=package
        )
        _write_canonical(root / spec["result_path"], result)
        terminal = refit._expected_terminal(
            root=root, spec=spec, seal_id=seal_id, h1=h1, result=result
        )
        equal(
            [item.path for item in terminal.artifacts],
            [spec["result_path"]]
            + [f"{spec['package_path']}/{name}" for name in spec["package_files"]],
            "success terminal artifact order",
        )
        check(terminal.payload["activation"] is False, "success terminal activation")

        model = root / spec["package_path"] / spec["package_files"][0]
        model.chmod(0o644)
        model.write_bytes(b"conflict")
        with expect_raises(refit.RefitError, "manifest differs"):
            refit._install_package(
                root=root,
                spec=spec,
                seal_id=seal_id,
                h1=h1,
                canonical_phase=phase,
            )

    with tempfile.TemporaryDirectory(prefix="usdchf-refit-terminal-fail-") as raw:
        root = Path(raw)
        (root / "results/json").mkdir(parents=True)
        spec = _custom_spec()
        spec["package_path"] = "results/artifacts/absent"
        spec["result_path"] = "results/json/result.json"
        comparisons = {
            "model_bytes_equal": [True, False, True],
            "strategy_bytes_equal": True,
            "resolved_spec_bytes_equal": True,
        }
        result = refit._failure_result(
            spec=spec,
            seal_id="c" * 64,
            h1="d" * 40,
            comparisons=comparisons,
        )
        _write_canonical(root / spec["result_path"], result)
        terminal = refit._expected_terminal(
            root=root,
            spec=spec,
            seal_id="c" * 64,
            h1="d" * 40,
            result=result,
        )
        equal(len(terminal.artifacts), 1, "NO_CANDIDATE artifact count")
        package_destination = root / spec["package_path"]
        package_destination.parent.mkdir(parents=True)
        package_destination.symlink_to(root / "missing-package", target_is_directory=True)
        with expect_raises(refit.RefitError, "package destination"):
            refit._expected_terminal(
                root=root,
                spec=spec,
                seal_id="c" * 64,
                h1="d" * 40,
                result=result,
            )
        package_destination.unlink()
        all_equal = refit._failure_result(
            spec=spec,
            seal_id="c" * 64,
            h1="d" * 40,
            comparisons={
                "model_bytes_equal": [True, True, True],
                "strategy_bytes_equal": True,
                "resolved_spec_bytes_equal": True,
            },
        )
        _write_canonical(root / spec["result_path"], all_equal)
        with expect_raises(refit.RefitError, "exact completed mismatch"):
            refit._expected_terminal(
                root=root,
                spec=spec,
                seal_id="c" * 64,
                h1="d" * 40,
                result=all_equal,
            )


def _complete_synthetic_lineage(root: Path) -> tuple[dict[str, Any], str, str]:
    _make_repo(root, include_shared=True)
    h0 = _git(root, "rev-parse", "HEAD")
    spec = refit._load_spec(root)
    source = _source_manifest(spec, h0)
    _write_canonical(root / spec["source_manifest_path"], source)
    seal = refit._expected_seal(root, spec, source, h0, _runtime_contract(root, h0))
    evidence.publish(seal, repo_root=root, require_new=True)
    _git(
        root,
        "add",
        spec["source_manifest_path"],
        f"evidence/objects/{seal.object_id}.json",
    )
    _git(root, "commit", "-m", "fixture H1")
    h1 = _git(root, "rev-parse", "HEAD")

    phase = _package_fixture(root, spec, seal.object_id, h1)
    package = refit._install_package(
        root=root,
        spec=spec,
        seal_id=seal.object_id,
        h1=h1,
        canonical_phase=phase,
    )
    result = refit._success_result(
        spec=spec, seal_id=seal.object_id, h1=h1, package=package
    )
    _write_canonical(root / spec["result_path"], result)
    terminal = refit._expected_terminal(
        root=root,
        spec=spec,
        seal_id=seal.object_id,
        h1=h1,
        result=result,
    )
    evidence.publish(terminal, repo_root=root, require_new=True)
    _git(root, "add", spec["result_path"], spec["package_path"], f"evidence/objects/{terminal.object_id}.json")
    _git(root, "commit", "-m", "fixture terminal")
    return spec, seal.object_id, terminal.object_id


def test_verify_without_sources_or_adapter_import() -> None:
    with tempfile.TemporaryDirectory(prefix="usdchf-refit-verify-") as raw:
        source_root = Path(raw) / "source"
        clone_root = Path(raw) / "fresh-clone"
        _, seal_id, terminal_id = _complete_synthetic_lineage(source_root)
        _git(source_root, "clone", "--local", str(source_root), str(clone_root))
        check(not (clone_root / "features").exists(), "fresh clone unexpectedly has sources")
        check(not (clone_root / "logs").exists(), "fresh clone unexpectedly has ignored logs")
        protected = _protected_snapshot(clone_root)
        real_import = builtins.__import__

        def guarded_import(name: str, *args: Any, **kwargs: Any) -> Any:
            if name in {
                "m15_book_refresh_adapters",
                "usdchf_15m_xpair_frozen",
                "usdchf_15m_xpair",
            }:
                raise GateFailure(f"verify imported fit source: {name}")
            return real_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=guarded_import):
            report = refit.verify(seal_id, repo_root=clone_root)
        equal(report["terminal_id"], terminal_id, "verified terminal identity")
        equal(report["source_status"], "unavailable", "source-free verification")
        check(report["activation"] is False, "verification activation")
        process = subprocess.run(
            [
                sys.executable,
                str(clone_root / refit.SCRIPT_PATH),
                "--repo-root",
                str(clone_root),
                "verify",
                "--seal-id",
                seal_id,
            ],
            cwd=clone_root,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        equal(process.returncode, 0, f"fresh-clone verify CLI: {process.stderr}")
        cli_report = json.loads(process.stdout)
        equal(cli_report["terminal_id"], terminal_id, "fresh-clone CLI terminal")
        equal(cli_report["source_status"], "unavailable", "fresh-clone CLI sources")
        equal(_protected_snapshot(clone_root), protected, "verify modified protected paths")


def main() -> int:
    started = time.monotonic()
    timings = {
        "spec": run_section("spec identity and canonical paths", test_spec_identity_and_paths),
        "partition": run_section("synthetic exact partition boundaries", test_exact_partition_boundaries),
        "source_manifest": run_section("strict source-manifest rows", test_source_manifest_row_contract),
        "seal": run_section("seal crash recovery", test_seal_crash_recovery),
        "seal_foreign": run_section("foreign pre-H1 rejection without mutation", test_seal_foreign_states_fail_without_mutation),
        "h1": run_section("exact H1 authentication", test_h1_authentication),
        "source_states": run_section("source availability states", test_source_availability_states),
        "lock": run_section("cross-process research lock", test_cross_process_research_lock),
        "checkpoint": run_section("checkpoint orphan recovery and adoption", test_checkpoint_orphan_and_adoption),
        "fit": run_section("public serial fit success and mismatch", test_public_fit_serial_success_and_mismatch),
        "fit_recovery": run_section("public fit crash recovery boundaries", test_public_fit_crash_recovery_boundaries),
        "package": run_section("package no-replace and terminal semantics", test_package_noreplace_and_terminal_semantics),
        "verify": run_section("fresh-clone source-free zero-adapter verification", test_verify_without_sources_or_adapter_import),
    }
    print(
        f"[test] PASS total={time.monotonic() - started:.2f}s "
        f"timings={json.dumps(timings, sort_keys=True)} "
        "temporary_mutations_only=true real_features_read=false models_fitted=false "
        "demo_buys=false real_money=false",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
