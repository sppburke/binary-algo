#!/usr/bin/env python3
"""Focused behavioral gate for the ROS-1 evidence spine.

Run from the repository root::

    ~/binary-algo-venv/bin/python scripts/test_evidence_store.py

All mutations are confined to temporary directories.  Real legacy fixtures
are read and copied only; no model, replay, protected look, or result write is
performed.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator
from unittest import mock

import evidence_store as evidence


REPO_ROOT = Path(__file__).resolve().parents[1]
GOLDEN_OBJECT_ID = "04388458e40eefe9c520277fc694ac18af44e947f2d873e99f88cdef98bd6248"
EXPECTED_IDS = {
    "ROS-1",
    "ROS-2A",
    "ROS-2B",
    "ROS-3",
    "ROS-3A",
    "ROS-4",
    "ROS-5",
    "ROS-6",
    "ROS-7",
    *(f"ROSD-{index:02d}" for index in range(1, 17)),
    "ROSR-01",
}
EXPECTED_STATUSES = {
    "ROS-1": "accepted",
    "ROS-2A": "deferred_until_trigger",
    "ROS-2B": "deferred_until_trigger",
    "ROS-3": "planned",
    "ROS-3A": "planned",
    "ROS-4": "deferred_until_trigger",
    "ROS-5": "deferred_until_trigger",
    "ROS-6": "superseded_by:ROS-3",
    "ROS-7": "deferred_until_trigger",
    **{f"ROSD-{index:02d}": "deferred_until_trigger" for index in range(1, 17)},
    "ROSR-01": "resolved_non_goal",
}
MODERN_PATHS = [
    "results/json/m15_book_refresh_53547498c599f2877a2f6616ae725f8f27e11070e3b550e82eb26ec039fb23f6_joint_replay_result.json",
    "results/json/m15_book_refresh_53547498c599f2877a2f6616ae725f8f27e11070e3b550e82eb26ec039fb23f6_shadow_spec_result.json",
]
PRE_SCHEMA_PATHS = ["results/json/gbpusd_15m_cpcv_xpair_ny_seedens8_result.json"]
BOOK_ROOT = "books/GBPUSD.m15ny_xpair_seedens8.v1"
BOOK_PATHS = [f"{BOOK_ROOT}/m15xpny_GBPUSD_k8_s{index}.txt" for index in range(8)] + [
    f"{BOOK_ROOT}/m15xpny_GBPUSD_k8_strategy.json",
    "books/GBPUSD.m15ny_xpair_seedens8.v1.manifest.json",
]


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


def run_section(name: str, function: Callable[[], None]) -> float:
    started = time.monotonic()
    function()
    elapsed = time.monotonic() - started
    print(f"[test] PASS {name} ({elapsed:.2f}s)", flush=True)
    return elapsed


def make_repo(root: Path) -> Path:
    (root / "evidence" / "objects").mkdir(parents=True)
    return root


def test_identity_and_strict_contract() -> None:
    envelope = evidence.EvidenceEnvelope.create(
        kind=evidence.LEGACY_SIDECAR_KIND,
        payload={"alpha": 1, "ordered": ["a", "b"]},
    )
    equal(envelope.object_id, GOLDEN_OBJECT_ID, "object identity golden")
    equal(evidence.derive_object_id(envelope), GOLDEN_OBJECT_ID, "public ID derivation")
    equal(
        evidence._decode_canonical_json(envelope.canonical_bytes()),
        envelope.as_dict(),
        "canonical round trip",
    )

    fresh_code = (
        "import sys;sys.path.insert(0,'scripts');import evidence_store as e;"
        "print(e.EvidenceEnvelope.create(kind=e.LEGACY_SIDECAR_KIND,"
        "payload={'alpha':1,'ordered':['a','b']}).object_id)"
    )
    fresh = subprocess.check_output(
        [sys.executable, "-c", fresh_code], cwd=REPO_ROOT, text=True
    ).strip()
    equal(fresh, GOLDEN_OBJECT_ID, "fresh-process identity")

    for mutation in (
        lambda value: value["payload"].update(alpha=2),
        lambda value: value.update(kind="research.changed/v1"),
        lambda value: value.update(object_id="f" * 64),
    ):
        changed = envelope.as_dict()
        mutation(changed)
        with expect_raises(evidence.EvidenceError):
            evidence.EvidenceEnvelope.from_dict(changed)

    changed_schema = envelope.as_dict()
    changed_schema["schema"] = "research-evidence-envelope/v2"
    with expect_raises(evidence.EvidenceError, "schema"):
        evidence.EvidenceEnvelope.from_dict(changed_schema)

    missing = envelope.as_dict()
    missing.pop("kind")
    with expect_raises(evidence.EvidenceError, "fields"):
        evidence.EvidenceEnvelope.from_dict(missing)
    unknown = envelope.as_dict()
    unknown["extra"] = True
    with expect_raises(evidence.EvidenceError, "fields"):
        evidence.EvidenceEnvelope.from_dict(unknown)

    with expect_raises(evidence.EvidenceError, "duplicate"):
        evidence._decode_canonical_json(b'{"a":1,"a":2}')
    with expect_raises(evidence.EvidenceError, "non-finite"):
        evidence._decode_canonical_json(b'{"a":NaN}')
    with expect_raises(evidence.EvidenceError, "canonical"):
        evidence._decode_canonical_json(b'{"a": 1}')
    with expect_raises(evidence.EvidenceError, "canonical"):
        evidence._decode_canonical_json(envelope.canonical_bytes() + b"\n")
    with expect_raises(evidence.EvidenceError, "non-finite"):
        evidence.EvidenceEnvelope.create(
            kind=evidence.LEGACY_SIDECAR_KIND, payload={"bad": float("inf")}
        )

    for malformed_id in ("short", "G" * 64):
        malformed = envelope.as_dict()
        malformed["object_id"] = malformed_id
        with expect_raises(evidence.EvidenceError, "object_id"):
            evidence.EvidenceEnvelope.from_dict(malformed)
        with expect_raises(evidence.EvidenceError, "object ID"):
            evidence.verify_object(malformed_id, repo_root=REPO_ROOT)

    first = evidence.ArtifactRef("a.bin", 1, "1" * 64)
    second = evidence.ArtifactRef("b.bin", 2, "2" * 64)
    ordered = evidence.EvidenceEnvelope.create(
        kind=evidence.LEGACY_SIDECAR_KIND,
        payload={},
        artifacts=[first, second],
        dependencies=["3" * 64, "4" * 64],
    )
    reversed_artifacts = evidence.EvidenceEnvelope.create(
        kind=evidence.LEGACY_SIDECAR_KIND,
        payload={},
        artifacts=[second, first],
        dependencies=["3" * 64, "4" * 64],
    )
    reversed_dependencies = evidence.EvidenceEnvelope.create(
        kind=evidence.LEGACY_SIDECAR_KIND,
        payload={},
        artifacts=[first, second],
        dependencies=["4" * 64, "3" * 64],
    )
    check(
        ordered.object_id != reversed_artifacts.object_id,
        "artifact order was normalized",
    )
    check(
        ordered.object_id != reversed_dependencies.object_id,
        "dependency order was normalized",
    )
    changed_domain = evidence._derive_object_id(
        schema=ordered.schema,
        kind=ordered.kind,
        payload=ordered.payload,
        artifacts=ordered.artifacts,
        dependencies=ordered.dependencies,
        domain=evidence.OBJECT_ID_DOMAIN + "/changed",
    )
    check(changed_domain != ordered.object_id, "identity domain was not bound")

    with expect_raises(evidence.EvidenceError, "bytes"):
        evidence.ArtifactRef("a.bin", True, "0" * 64)
    with expect_raises(evidence.EvidenceError, "sha256"):
        evidence.ArtifactRef("a.bin", 0, "A" * 64)
    malformed_dependencies = envelope.as_dict()
    malformed_dependencies["dependencies"] = "not-a-list"
    with expect_raises(evidence.EvidenceError, "list"):
        evidence.EvidenceEnvelope.from_dict(malformed_dependencies)
    for malformed_id in ("short", "G" * 64):
        malformed_dependencies = envelope.as_dict()
        malformed_dependencies["dependencies"] = [malformed_id]
        with expect_raises(evidence.EvidenceError, "dependencies"):
            evidence.EvidenceEnvelope.from_dict(malformed_dependencies)


def test_artifacts_dependencies_and_stateless_verification() -> None:
    with tempfile.TemporaryDirectory(prefix="evidence-artifacts-") as raw:
        root = make_repo(Path(raw))
        source = root / "data" / "source.bin"
        source.parent.mkdir()
        source.write_bytes(b"bound source bytes")
        reference = evidence.ArtifactRef.capture("data/source.bin", repo_root=root)

        real_stat = os.stat

        def fail_child_stat(path: os.PathLike[str], *args: Any, **kwargs: Any) -> Any:
            if path == "data" and kwargs.get("dir_fd") is not None:
                raise FileNotFoundError("injected child-stat race")
            return real_stat(path, *args, **kwargs)

        fd_before = len(os.listdir("/proc/self/fd"))
        with mock.patch.object(evidence.os, "stat", side_effect=fail_child_stat):
            for _ in range(5):
                with expect_raises(evidence.EvidenceError, "real directory"):
                    evidence.ArtifactRef.capture("data/source.bin", repo_root=root)
        equal(
            len(os.listdir("/proc/self/fd")),
            fd_before,
            "child-stat race leaked directory descriptors",
        )

        for invalid in (
            "/absolute",
            "../escape",
            "a/../escape",
            "a//b",
            "./a",
            "a/./b",
            "a\\b",
        ):
            with expect_raises(evidence.EvidenceError, "path"):
                evidence.ArtifactRef(invalid, 0, "0" * 64)
        with expect_raises(evidence.EvidenceError, "cannot open"):
            evidence.ArtifactRef.capture("data/missing.bin", repo_root=root)

        alias = root / "alias"
        alias.symlink_to(source.parent, target_is_directory=True)
        with expect_raises(evidence.EvidenceError, "real directory"):
            evidence.ArtifactRef.capture("alias/source.bin", repo_root=root)
        final_link = source.parent / "source-link.bin"
        final_link.symlink_to(source)
        with expect_raises(evidence.EvidenceError, "non-symlink"):
            evidence.ArtifactRef.capture("data/source-link.bin", repo_root=root)

        outside = root / "outside"
        outside.mkdir()
        (outside / "source.bin").write_bytes(b"outside bytes")
        held = root / "data-held"
        real_read = evidence._read_regular_at

        def swap_parent_then_read(
            parent_descriptor: int, filename: str, **kwargs: Any
        ) -> bytes:
            source.parent.rename(held)
            source.parent.symlink_to(outside, target_is_directory=True)
            raw = real_read(parent_descriptor, filename, **kwargs)
            equal(raw, b"bound source bytes", "artifact read escaped held parent")
            return raw

        try:
            with mock.patch.object(
                evidence, "_read_regular_at", side_effect=swap_parent_then_read
            ):
                with expect_raises(evidence.EvidenceError, "changed during"):
                    evidence.ArtifactRef.capture("data/source.bin", repo_root=root)
        finally:
            if source.parent.is_symlink():
                source.parent.unlink()
            if held.exists():
                held.rename(source.parent)

        with expect_raises(evidence.EvidenceError, "size"):
            evidence.ArtifactRef(
                reference.path, reference.bytes + 1, reference.sha256
            ).verify(repo_root=root)
        with expect_raises(evidence.EvidenceError, "hash"):
            evidence.ArtifactRef(reference.path, reference.bytes, "0" * 64).verify(
                repo_root=root
            )

        parent = evidence.EvidenceEnvelope.create(
            kind=evidence.LEGACY_SIDECAR_KIND,
            payload={"name": "parent"},
            artifacts=[reference],
        )
        evidence.publish(parent, repo_root=root)
        child = evidence.EvidenceEnvelope.create(
            kind=evidence.LEGACY_SIDECAR_KIND,
            payload={"name": "child"},
            dependencies=[parent.object_id],
        )
        evidence.publish(child, repo_root=root)
        equal(
            evidence.verify_object(child.object_id, repo_root=root).object_id,
            child.object_id,
            "dependency verification",
        )
        equal(
            evidence.verify_store(repo_root=root),
            tuple(sorted([parent.object_id, child.object_id])),
            "flat store verification",
        )

        os.utime(source, (1, 1))
        evidence.verify_object(child.object_id, repo_root=root)
        source.write_bytes(b"changed source bytes")
        with expect_raises(evidence.EvidenceError):
            evidence.verify_object(child.object_id, repo_root=root)
        source.write_bytes(b"bound source bytes")

        parent_path = root / "evidence" / "objects" / f"{parent.object_id}.json"
        original_parent = parent_path.read_bytes()
        parent_path.chmod(0o644)
        changed_parent = parent.as_dict()
        changed_parent["payload"]["name"] = "changed"
        parent_path.write_bytes(evidence._canonical_json_bytes(changed_parent))
        parent_path.chmod(0o444)
        with expect_raises(evidence.EvidenceError):
            evidence.verify_object(child.object_id, repo_root=root)
        parent_path.chmod(0o644)
        parent_path.write_bytes(original_parent)
        parent_path.chmod(0o444)

        dangling = evidence.EvidenceEnvelope.create(
            kind=evidence.LEGACY_SIDECAR_KIND,
            payload={},
            dependencies=["9" * 64],
        )
        with expect_raises(evidence.EvidenceError):
            evidence.publish(dangling, repo_root=root)

        for command in (
            ["verify-object", child.object_id],
            ["verify-store"],
        ):
            completed = subprocess.run(
                [
                    sys.executable,
                    "scripts/evidence_store.py",
                    "--repo-root",
                    str(root),
                    *command,
                ],
                cwd=REPO_ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
            equal(completed.returncode, 0, f"CLI {' '.join(command)}")
            check("verified" in completed.stdout, "CLI omitted verification receipt")

        unexpected = root / "evidence" / "objects" / ".partial"
        unexpected.write_bytes(b"partial")
        with expect_raises(evidence.EvidenceError, "unexpected"):
            evidence.verify_store(repo_root=root)
        unexpected.unlink()


def test_immutable_publication() -> None:
    envelope = evidence.EvidenceEnvelope.create(
        kind=evidence.LEGACY_SIDECAR_KIND,
        payload={"publication": "exact"},
    )
    with tempfile.TemporaryDirectory(prefix="evidence-publish-") as raw:
        root = make_repo(Path(raw))
        destination = evidence.publish(envelope, repo_root=root)
        first_inode = destination.stat().st_ino
        equal(
            evidence.publish(envelope, repo_root=root), destination, "idempotent path"
        )
        equal(
            destination.stat().st_ino,
            first_inode,
            "idempotent publication replaced inode",
        )
        check(not destination.stat().st_mode & 0o222, "published object is writable")
        equal(destination.read_bytes(), envelope.canonical_bytes(), "published bytes")
        destination.chmod(0o644)
        equal(
            evidence.verify_object(envelope.object_id, repo_root=root).object_id,
            envelope.object_id,
            "Git-portable verification",
        )
        with expect_raises(evidence.EvidenceError, "writable"):
            evidence.publish(envelope, repo_root=root)
        destination.chmod(0o444)

    with tempfile.TemporaryDirectory(prefix="evidence-conflict-") as raw:
        root = make_repo(Path(raw))
        destination = root / "evidence" / "objects" / f"{envelope.object_id}.json"
        destination.write_bytes(b"different pre-existing bytes")
        destination.chmod(0o444)
        before = destination.read_bytes()
        with expect_raises(evidence.EvidenceError, "differs"):
            evidence.publish(envelope, repo_root=root)
        equal(destination.read_bytes(), before, "conflicting destination was modified")

    with tempfile.TemporaryDirectory(prefix="evidence-symlink-store-") as raw:
        root = Path(raw)
        (root / "evidence").mkdir()
        outside = root / "outside"
        outside.mkdir()
        (root / "evidence" / "objects").symlink_to(outside, target_is_directory=True)
        with expect_raises(evidence.EvidenceError, "real directory"):
            evidence.publish(envelope, repo_root=root)

    with tempfile.TemporaryDirectory(prefix="evidence-store-swap-") as raw:
        root = make_repo(Path(raw))
        objects = root / "evidence" / "objects"
        held = root / "evidence" / "objects-held"
        outside = root / "outside"
        outside.mkdir()
        real_create = evidence._create_temp_at

        def swap_store_then_create(
            store_descriptor: int, object_id: str
        ) -> tuple[int, str]:
            objects.rename(held)
            objects.symlink_to(outside, target_is_directory=True)
            return real_create(store_descriptor, object_id)

        try:
            with mock.patch.object(
                evidence, "_create_temp_at", side_effect=swap_store_then_create
            ):
                with expect_raises(evidence.EvidenceError, "changed during"):
                    evidence.publish(envelope, repo_root=root)
            equal(list(outside.iterdir()), [], "store swap redirected publication")
        finally:
            if objects.is_symlink():
                objects.unlink()
            if held.exists():
                held.rename(objects)
        equal(
            evidence.verify_object(envelope.object_id, repo_root=root).object_id,
            envelope.object_id,
            "held-store publication integrity",
        )

    with tempfile.TemporaryDirectory(prefix="evidence-interrupt-before-") as raw:
        root = make_repo(Path(raw))
        destination = root / "evidence" / "objects" / f"{envelope.object_id}.json"
        with mock.patch.object(
            evidence.os, "link", side_effect=OSError("injected before")
        ):
            with expect_raises(evidence.EvidenceError, "publish"):
                evidence.publish(envelope, repo_root=root)
        check(
            not destination.exists(), "interrupted pre-install exposed a public object"
        )
        equal(list((root / "evidence" / "objects").iterdir()), [], "temporary leaked")

    with tempfile.TemporaryDirectory(prefix="evidence-interrupt-after-") as raw:
        root = make_repo(Path(raw))
        destination = root / "evidence" / "objects" / f"{envelope.object_id}.json"
        real_link = os.link

        def link_then_fail(
            source: os.PathLike[str], target: os.PathLike[str], **kwargs: Any
        ) -> None:
            real_link(source, target, **kwargs)
            raise OSError("injected after")

        with mock.patch.object(evidence.os, "link", side_effect=link_then_fail):
            with expect_raises(evidence.EvidenceError, "publish"):
                evidence.publish(envelope, repo_root=root)
        check(
            destination.exists(),
            "post-install interruption lost complete public object",
        )
        equal(
            destination.read_bytes(), envelope.canonical_bytes(), "post-install bytes"
        )
        equal(
            evidence.publish(envelope, repo_root=root), destination, "interrupted retry"
        )

    with tempfile.TemporaryDirectory(prefix="evidence-concurrent-") as raw:
        root = make_repo(Path(raw))
        code = (
            "import sys;sys.path.insert(0,'scripts');import evidence_store as e;"
            f"x=e.EvidenceEnvelope.create(kind={evidence.LEGACY_SIDECAR_KIND!r},"
            "payload={'publication':'exact'});"
            f"print(e.publish(x,repo_root={str(root)!r}))"
        )
        processes = [
            subprocess.Popen(
                [sys.executable, "-c", code],
                cwd=REPO_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            for _ in range(4)
        ]
        for process in processes:
            stdout, stderr = process.communicate(timeout=20)
            equal(process.returncode, 0, f"concurrent publisher: {stdout} {stderr}")
        equal(
            evidence.verify_store(repo_root=root),
            (envelope.object_id,),
            "concurrent store",
        )


def _real_sidecars() -> dict[str, evidence.EvidenceEnvelope]:
    sidecars: dict[str, evidence.EvidenceEnvelope] = {}
    for object_id in evidence.verify_store(repo_root=REPO_ROOT):
        envelope = evidence.verify_object(object_id, repo_root=REPO_ROOT)
        fixture_class = envelope.payload["fixture_class"]
        check(fixture_class not in sidecars, f"duplicate fixture class {fixture_class}")
        sidecars[fixture_class] = envelope
    return sidecars


def test_legacy_sidecars() -> None:
    sidecars = _real_sidecars()
    equal(
        set(sidecars),
        {"modern_issue_9_evidence", "heterogeneous_pre_schema_result", "frozen_book"},
        "fixture classes",
    )
    for envelope in sidecars.values():
        equal(envelope.kind, evidence.LEGACY_SIDECAR_KIND, "legacy sidecar kind")
        equal(envelope.dependencies, (), "legacy sidecar dependency list")
        equal(
            set(envelope.payload["facts"]),
            {"authoritative", "derived", "unmappable"},
            "fact classifications",
        )

    expected_paths = {
        "modern_issue_9_evidence": MODERN_PATHS,
        "heterogeneous_pre_schema_result": PRE_SCHEMA_PATHS,
        "frozen_book": BOOK_PATHS,
    }
    for fixture_class, envelope in sidecars.items():
        equal(
            [artifact.path for artifact in envelope.artifacts],
            expected_paths[fixture_class],
            f"{fixture_class} ArtifactRefs",
        )
        check(
            all(
                artifact.path not in {"books/INDEX.json", "MODEL_REGISTRY.md"}
                for artifact in envelope.artifacts
            ),
            f"{fixture_class} bound mutable registry state",
        )

    book_facts = sidecars["frozen_book"].payload["facts"]
    equal(
        book_facts["authoritative"]["manifest"]["strategy_json"],
        "models/m15xpny_GBPUSD_k8_strategy.json",
        "historical strategy pointer",
    )
    relation = book_facts["unmappable"][
        "legacy_strategy_pointer_to_bound_frozen_strategy"
    ]
    equal(relation["value"], "unmappable", "legacy strategy relationship")

    joint = json.loads((REPO_ROOT / MODERN_PATHS[0]).read_bytes())
    shadow = json.loads((REPO_ROOT / MODERN_PATHS[1]).read_bytes())
    modern_facts = sidecars["modern_issue_9_evidence"].payload["facts"]
    for name, source, fields in (
        (
            "joint",
            joint,
            ("schema", "run_id", "prereg_id", "S", "publication_lock_complete"),
        ),
        (
            "shadow",
            shadow,
            (
                "schema",
                "run_id",
                "prereg_id",
                "status",
                "S_shadow",
                "k_shadow",
                "activation",
                "shadow_started",
            ),
        ),
    ):
        equal(
            modern_facts["authoritative"][name],
            {field: source[field] for field in fields},
            f"{name} authoritative facts",
        )
    equal(joint["run_id"], shadow["run_id"], "derived shared run ID")
    equal(joint["prereg_id"], shadow["prereg_id"], "derived shared prereg ID")

    pre_schema = json.loads((REPO_ROOT / PRE_SCHEMA_PATHS[0]).read_bytes())
    pre_facts = sidecars["heterogeneous_pre_schema_result"].payload["facts"]
    equal(
        pre_facts["derived"]["top_level_schema_present"]["value"],
        "schema" in pre_schema,
        "derived schema presence",
    )
    equal(
        pre_facts["derived"]["path_count"]["value"],
        len(pre_schema["paths"]),
        "derived path count",
    )

    manifest = json.loads(
        (REPO_ROOT / "books/GBPUSD.m15ny_xpair_seedens8.v1.manifest.json").read_bytes()
    )
    strategy = json.loads((REPO_ROOT / BOOK_PATHS[-2]).read_bytes())
    book_sidecar = sidecars["frozen_book"]
    for artifact, stated in zip(
        book_sidecar.artifacts[:8], manifest["artifacts"], strict=True
    ):
        equal(Path(artifact.path).name, stated["file"], "manifest artifact basename")
        equal(artifact.bytes, stated["bytes"], "manifest artifact byte count")
        equal(artifact.sha256, stated["sha256"], "manifest artifact hash")
    equal(manifest["id"], strategy["book_id"], "derived book identity equality")

    with tempfile.TemporaryDirectory(prefix="evidence-sidecars-") as raw:
        root = make_repo(Path(raw))
        for envelope in sidecars.values():
            source_object = (
                REPO_ROOT / "evidence" / "objects" / f"{envelope.object_id}.json"
            )
            shutil.copyfile(
                source_object,
                root / "evidence" / "objects" / source_object.name,
            )
            for artifact in envelope.artifacts:
                source = REPO_ROOT.joinpath(*Path(artifact.path).parts)
                target = root.joinpath(*Path(artifact.path).parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
        equal(len(evidence.verify_store(repo_root=root)), 3, "copied sidecar store")
        for fixture_class, envelope in sidecars.items():
            mutated = root.joinpath(*Path(envelope.artifacts[0].path).parts)
            original = mutated.read_bytes()
            mutated.write_bytes(original + b"mutated")
            with expect_raises(evidence.EvidenceError):
                evidence.verify_object(envelope.object_id, repo_root=root)
            mutated.write_bytes(original)
            equal(
                evidence.verify_object(envelope.object_id, repo_root=root).object_id,
                envelope.object_id,
                f"{fixture_class} mutation restoration",
            )


def _expected_registry(value: Any) -> None:
    ids = evidence.validate_capability_registry(value)
    equal(set(ids), EXPECTED_IDS, "capability IDs")
    equal(len(ids), len(EXPECTED_IDS), "capability ID count")
    equal(
        {record["id"]: record["status"] for record in value["capabilities"]},
        EXPECTED_STATUSES,
        "capability statuses",
    )


def test_capability_registry() -> None:
    path = REPO_ROOT / "docs" / "research_os_capabilities.json"
    registry = json.loads(
        path.read_bytes(),
        object_pairs_hook=evidence._duplicate_rejecting_object,
        parse_constant=evidence._reject_json_constant,
    )
    _expected_registry(registry)
    for record in registry["capabilities"]:
        equal(set(record), set(evidence.CAPABILITY_FIELDS), f"{record['id']} fields")

    for status in evidence.LITERAL_CAPABILITY_STATUSES:
        changed = json.loads(json.dumps(registry))
        changed["capabilities"][0]["status"] = status
        evidence.validate_capability_registry(changed)
    triggered = json.loads(json.dumps(registry))
    triggered["capabilities"][1]["status"] = "triggered_pending_decision"
    evidence.validate_capability_registry(triggered)

    duplicate = json.loads(json.dumps(registry))
    duplicate["capabilities"].append(duplicate["capabilities"][0])
    with expect_raises(evidence.EvidenceError, "duplicate"):
        evidence.validate_capability_registry(duplicate)
    missing_field = json.loads(json.dumps(registry))
    missing_field["capabilities"][0].pop("decision")
    with expect_raises(evidence.EvidenceError, "fields"):
        evidence.validate_capability_registry(missing_field)
    extra_field = json.loads(json.dumps(registry))
    extra_field["capabilities"][0]["extra"] = True
    with expect_raises(evidence.EvidenceError, "fields"):
        evidence.validate_capability_registry(extra_field)

    for status in (
        "unknown",
        "superseded_by:ROS-404",
        "superseded_by:ROS-6",
        "superseded_by:",
    ):
        changed = json.loads(json.dumps(registry))
        target_index = 7 if status == "superseded_by:ROS-6" else 0
        changed["capabilities"][target_index]["status"] = status
        with expect_raises(evidence.EvidenceError):
            evidence.validate_capability_registry(changed)

    missing_id = json.loads(json.dumps(registry))
    missing_id["capabilities"].pop()
    with expect_raises(GateFailure, "IDs"):
        _expected_registry(missing_id)
    extra_id = json.loads(json.dumps(registry))
    added = dict(extra_id["capabilities"][0])
    added["id"] = "ROS-999"
    extra_id["capabilities"].append(added)
    with expect_raises(GateFailure, "IDs"):
        _expected_registry(extra_id)


def main() -> int:
    started = time.monotonic()
    timings = {
        "identity": run_section(
            "identity and strict contract", test_identity_and_strict_contract
        ),
        "verification": run_section(
            "artifact/dependency/stateless verification",
            test_artifacts_dependencies_and_stateless_verification,
        ),
        "publication": run_section("immutable publication", test_immutable_publication),
        "sidecars": run_section("legacy sidecars", test_legacy_sidecars),
        "registry": run_section("capability registry", test_capability_registry),
    }
    print(
        f"[test] PASS total={time.monotonic() - started:.2f}s "
        f"timings={json.dumps(timings, sort_keys=True)} "
        "temporary_mutations_only=true models_fitted=false replay_invoked=false",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
