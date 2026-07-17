#!/usr/bin/env python3
"""Focused, temp-only acceptance gate for the ROS-3 vertical slice.

Run from the repository root::

    ~/binary-algo-venv/bin/python scripts/test_protected_evaluation.py
"""

from __future__ import annotations

import hashlib
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

import book_runtime
import evidence_store as evidence
import manifest
import protected_evaluation as protected
from m15_book_refresh_stats import shadow_empty_set_status


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "ros3.synthetic.shadow-empty.v1"
HYPOTHESIS = "A registered survivor remains inactive when shadow eligibility is empty."
FALSIFIER = "Reject unregistered, reordered, hidden, or shadow-eligible outcome state."
ARMS = ["candidate.synthetic.v1", "candidate.control.v1"]
PROTECTED_PATH = "protected/outcome.json"
SIDECAR_ID = "21434855a883fea8ff3bdf72da479534798f6f8653057ea12d95fba608186972"
BASELINE_HASHES = {
    "scripts/m15_book_refresh.py": "b7b7b451abaa0fe96f34ec2f6b874c22ed61ed437c3c7e55d791f182a57505ed",
    "scripts/m15_book_refresh_stats.py": "175fb2fc677903ba07944f9c88d625f548863816076862dd42b472e395848930",
    "scripts/m15_book_refresh_replay.py": "83526abf8f7fb3b5cc576452b30064bd880bdcaa9fc9899b5af61eba2789a5e4",
    "scripts/test_m15_book_refresh.py": "bbbd03196b4b6050b7c77c633ee83287049d889900fb52b7392bd640610d615a",
    "scripts/book_runtime.py": "fb299d8adc8b585e0c19294a362c32811912eadf5d87f5a92397c7aba77e916d",
    "books/INDEX.json": "62f0b37afac091bb9821687dce4c0423ffe3bbf7318ac99f4c2d93416f9e7d26",
    "MODEL_REGISTRY.md": "73a405c16e524987759c47a62e72b0f126a91a0f1e038db287d0bdcdf8c96aac",
    "results/json/m15_book_refresh_53547498c599f2877a2f6616ae725f8f27e11070e3b550e82eb26ec039fb23f6_joint_replay_result.json": "d37c90db846cad07c2e22111237f16d0548bbcf0d06b7948182c6776b2a02b52",
    "results/json/m15_book_refresh_53547498c599f2877a2f6616ae725f8f27e11070e3b550e82eb26ec039fb23f6_shadow_spec_result.json": "ee46c8a8d07912a1f6c865d3934841562c1039d09a065926d6cfb74118a7b22e",
}


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


def inactive_outcome() -> dict[str, Any]:
    return {
        "schema": protected.OUTCOME_SCHEMA,
        "survivors": [ARMS[0]],
        "shadow_eligible": [],
        "exclusions": {ARMS[0]: "synthetic_not_shadow_eligible"},
    }


def make_repo(root: Path, raw: bytes) -> tuple[Path, evidence.ArtifactRef]:
    (root / "evidence" / "objects").mkdir(parents=True)
    evaluator = root / protected.EVALUATOR_PATH
    evaluator.parent.mkdir(parents=True)
    shutil.copyfile(REPO_ROOT / protected.EVALUATOR_PATH, evaluator)
    outcome = root / PROTECTED_PATH
    outcome.parent.mkdir(parents=True)
    outcome.write_bytes(raw)
    reference = evidence.ArtifactRef(
        path=PROTECTED_PATH,
        bytes=len(raw),
        sha256=hashlib.sha256(raw).hexdigest(),
    )
    return root, reference


def canonical(value: Any) -> bytes:
    return manifest.canonical_json_bytes(value)


def run(root: Path, reference: evidence.ArtifactRef) -> evidence.EvidenceEnvelope:
    return protected.run_protected_evaluation(
        repo_root=root,
        experiment_id=EXPERIMENT_ID,
        hypothesis=HYPOTHESIS,
        falsifier=FALSIFIER,
        ordered_arms=ARMS,
        protected_input=reference,
    )


def domain_objects(root: Path) -> list[evidence.EvidenceEnvelope]:
    objects: list[evidence.EvidenceEnvelope] = []
    for object_id in evidence.verify_store(repo_root=root):
        envelope = evidence.verify_object(object_id, repo_root=root)
        if envelope.kind in protected.DOMAIN_KINDS:
            objects.append(envelope)
    return objects


def test_vertical_resume_and_determinism() -> None:
    with tempfile.TemporaryDirectory(prefix="ros3-vertical-a-") as raw_a:
        root_a, reference_a = make_repo(Path(raw_a), canonical(inactive_outcome()))
        real_read = evidence.ArtifactRef.read_verified
        real_evaluator = protected.shadow_empty_set_status
        protected_reads = 0
        evaluator_calls = 0

        def checked_read(
            self: evidence.ArtifactRef, *, repo_root: str | os.PathLike[str]
        ) -> bytes:
            nonlocal protected_reads
            if self.path == PROTECTED_PATH:
                protected_reads += 1
                kinds = {
                    item.kind
                    for item in evidence.inspect_store_metadata(repo_root=root_a)
                    if item.kind in protected.DOMAIN_KINDS
                }
                equal(
                    kinds,
                    {protected.SPEC_KIND, protected.FAMILY_KIND, protected.RECEIPT_KIND},
                    "durable state before protected read",
                )
            return real_read(self, repo_root=repo_root)

        def checked_evaluator(*args: Any, **kwargs: Any) -> dict[str, Any]:
            nonlocal evaluator_calls
            evaluator_calls += 1
            return real_evaluator(*args, **kwargs)

        with mock.patch.object(
            evidence.ArtifactRef, "read_verified", new=checked_read
        ), mock.patch.object(
            protected, "shadow_empty_set_status", new=checked_evaluator
        ):
            terminal_a = run(root_a, reference_a)
        equal(protected_reads, 1, "protected read count")
        equal(evaluator_calls, 1, "evaluator call count")
        equal(terminal_a.kind, protected.TERMINAL_KIND, "terminal kind")
        equal(terminal_a.payload["decision"], "inactive_only", "terminal decision")
        equal(terminal_a.payload["activation"], False, "activation authority")
        equal(
            terminal_a.payload["candidates"],
            [
                {
                    "candidate_id": ARMS[0],
                    "lifecycle_status": "inactive_shadow_candidate",
                }
            ],
            "inactive candidate reference",
        )
        equal(len(domain_objects(root_a)), 5, "five-object vertical")

        (root_a / PROTECTED_PATH).unlink()

        def bomb_protected_read(
            self: evidence.ArtifactRef, *, repo_root: str | os.PathLike[str]
        ) -> bytes:
            if self.path == PROTECTED_PATH:
                raise GateFailure("completed resume read protected input")
            return real_read(self, repo_root=repo_root)

        with mock.patch.object(
            evidence.ArtifactRef, "read_verified", new=bomb_protected_read
        ):
            resumed = run(root_a, reference_a)
        equal(resumed.as_dict(), terminal_a.as_dict(), "completed resume terminal")

        with tempfile.TemporaryDirectory(prefix="ros3-vertical-b-") as raw_b:
            root_b, reference_b = make_repo(
                Path(raw_b), canonical(inactive_outcome())
            )
            terminal_b = run(root_b, reference_b)
            equal(terminal_b.object_id, terminal_a.object_id, "terminal ID determinism")
            equal(
                terminal_b.canonical_bytes(),
                terminal_a.canonical_bytes(),
                "terminal byte determinism",
            )
            equal(
                [item.object_id for item in domain_objects(root_b)],
                [item.object_id for item in domain_objects(root_a)],
                "five-object ID determinism",
            )

    empty = {
        "schema": protected.OUTCOME_SCHEMA,
        "survivors": [],
        "shadow_eligible": [],
        "exclusions": {},
    }
    with tempfile.TemporaryDirectory(prefix="ros3-no-candidate-") as raw:
        root, reference = make_repo(Path(raw), canonical(empty))
        terminal = run(root, reference)
        equal(terminal.payload["decision"], "no_candidate", "empty terminal")
        equal(terminal.payload["candidates"], [], "empty candidate references")

    with tempfile.TemporaryDirectory(prefix="ros3-metadata-scan-") as raw:
        root, reference = make_repo(Path(raw), canonical(inactive_outcome()))
        unrelated = evidence.EvidenceEnvelope.create(
            kind=evidence.LEGACY_SIDECAR_KIND,
            payload={"fixture": "declared protected path"},
            artifacts=[reference],
        )
        evidence.publish(unrelated, repo_root=root)
        real_read = evidence.ArtifactRef.read_verified
        protected_reads = 0

        def receipt_guarded_read(
            self: evidence.ArtifactRef, *, repo_root: str | os.PathLike[str]
        ) -> bytes:
            nonlocal protected_reads
            if self.path == PROTECTED_PATH:
                protected_reads += 1
                kinds = {
                    item.kind
                    for item in evidence.inspect_store_metadata(repo_root=root)
                }
                check(
                    protected.RECEIPT_KIND in kinds,
                    "metadata discovery read protected input before receipt",
                )
            return real_read(self, repo_root=repo_root)

        with mock.patch.object(
            evidence.ArtifactRef, "read_verified", new=receipt_guarded_read
        ):
            terminal = run(root, reference)
        equal(terminal.payload["decision"], "inactive_only", "metadata scan terminal")
        equal(protected_reads, 1, "metadata scan protected reads")


def test_resume_state_table() -> None:
    with tempfile.TemporaryDirectory(prefix="ros3-evaluation-only-") as raw:
        root, reference = make_repo(Path(raw), canonical(inactive_outcome()))
        real_publish = evidence.publish

        def fail_terminal(
            envelope: evidence.EvidenceEnvelope,
            *,
            repo_root: str | os.PathLike[str],
            require_new: bool = False,
        ) -> Path:
            if envelope.kind == protected.TERMINAL_KIND:
                raise evidence.EvidenceError("injected before terminal publication")
            return real_publish(
                envelope, repo_root=repo_root, require_new=require_new
            )

        with mock.patch.object(evidence, "publish", new=fail_terminal):
            with expect_raises(evidence.EvidenceError, "injected"):
                run(root, reference)
        equal(
            [item.kind for item in domain_objects(root)].count(protected.EVALUATION_KIND),
            1,
            "evaluation-only state",
        )
        real_read = evidence.ArtifactRef.read_verified

        def bomb_read(
            self: evidence.ArtifactRef, *, repo_root: str | os.PathLike[str]
        ) -> bytes:
            if self.path == PROTECTED_PATH:
                raise GateFailure("evaluation-only resume reread protected input")
            return real_read(self, repo_root=repo_root)

        with mock.patch.object(evidence.ArtifactRef, "read_verified", new=bomb_read):
            terminal = run(root, reference)
        equal(terminal.payload["decision"], "inactive_only", "evaluation-only resume")

    with tempfile.TemporaryDirectory(prefix="ros3-receipt-only-") as raw:
        root, reference = make_repo(Path(raw), canonical(inactive_outcome()))
        real_read = evidence.ArtifactRef.read_verified

        def fail_first_read(
            self: evidence.ArtifactRef, *, repo_root: str | os.PathLike[str]
        ) -> bytes:
            if self.path == PROTECTED_PATH:
                raise evidence.EvidenceError("injected protected read failure")
            return real_read(self, repo_root=repo_root)

        with mock.patch.object(
            evidence.ArtifactRef, "read_verified", new=fail_first_read
        ):
            with expect_raises(evidence.EvidenceError, "injected"):
                run(root, reference)
        receipt_only = [item.kind for item in domain_objects(root)]
        equal(len(receipt_only), 3, "receipt-only object count")
        equal(
            set(receipt_only),
            {protected.SPEC_KIND, protected.FAMILY_KIND, protected.RECEIPT_KIND},
            "receipt-only state",
        )
        retry_reads = 0

        def count_retry(
            self: evidence.ArtifactRef, *, repo_root: str | os.PathLike[str]
        ) -> bytes:
            nonlocal retry_reads
            if self.path == PROTECTED_PATH:
                retry_reads += 1
                raise GateFailure("spent retry read protected input")
            return real_read(self, repo_root=repo_root)

        with mock.patch.object(evidence.ArtifactRef, "read_verified", new=count_retry):
            with expect_raises(protected.ProtectedEvaluationError, "BLOCKED"):
                run(root, reference)
        equal(retry_reads, 0, "receipt-only retry reads")

        chain = {item.kind: item for item in domain_objects(root)}
        forged_input = {
            "schema": protected.OUTCOME_SCHEMA,
            "survivors": [],
            "shadow_eligible": [],
            "exclusions": {},
        }
        forged_evaluation = evidence.EvidenceEnvelope.create(
            kind=protected.EVALUATION_KIND,
            payload={
                "schema": protected.EVALUATION_KIND,
                "experiment_id": EXPERIMENT_ID,
                "spec_id": chain[protected.SPEC_KIND].object_id,
                "family_id": chain[protected.FAMILY_KIND].object_id,
                "receipt_id": chain[protected.RECEIPT_KIND].object_id,
                "evaluator": {
                    "path": protected.EVALUATOR_PATH,
                    "function": protected.EVALUATOR_FUNCTION,
                },
                "normalized_input": forged_input,
                "output": shadow_empty_set_status([], []),
            },
            dependencies=[chain[protected.RECEIPT_KIND].object_id],
        )
        evidence.publish(forged_evaluation, repo_root=root)
        with mock.patch.object(
            evidence.ArtifactRef, "read_verified", new=count_retry
        ), mock.patch.object(
            protected,
            "shadow_empty_set_status",
            side_effect=GateFailure("forged evaluation reached evaluator"),
        ):
            with expect_raises(
                protected.ProtectedEvaluationError, "sealed protected bytes"
            ):
                run(root, reference)
        equal(retry_reads, 0, "forged evaluation retry reads")

    with tempfile.TemporaryDirectory(prefix="ros3-link-ambiguity-") as raw:
        root, reference = make_repo(Path(raw), canonical(inactive_outcome()))
        spec, _ = protected._build_spec(
            repo_root=root,
            experiment_id=EXPERIMENT_ID,
            hypothesis=HYPOTHESIS,
            falsifier=FALSIFIER,
            protected_input=reference,
        )
        family = protected._build_family(spec, ARMS)
        receipt = protected._build_receipt(spec, family)
        evidence.publish(spec, repo_root=root)
        evidence.publish(family, repo_root=root)
        real_link = evidence.os.link

        def install_receipt_then_fail(
            source: os.PathLike[str], target: os.PathLike[str], **kwargs: Any
        ) -> None:
            real_link(source, target, **kwargs)
            if str(target) == f"{receipt.object_id}.json":
                raise OSError("injected post-install ambiguity")

        with mock.patch.object(
            evidence.os, "link", side_effect=install_receipt_then_fail
        ):
            with expect_raises(evidence.EvidenceError, "cannot publish"):
                run(root, reference)
        retry_reads = 0
        real_read = evidence.ArtifactRef.read_verified

        def post_install_retry(
            self: evidence.ArtifactRef, *, repo_root: str | os.PathLike[str]
        ) -> bytes:
            nonlocal retry_reads
            if self.path == PROTECTED_PATH:
                retry_reads += 1
                raise GateFailure("post-install ambiguity reread protected input")
            return real_read(self, repo_root=repo_root)

        with mock.patch.object(
            evidence.ArtifactRef, "read_verified", new=post_install_retry
        ):
            with expect_raises(protected.ProtectedEvaluationError, "BLOCKED"):
                run(root, reference)
        equal(retry_reads, 0, "post-install ambiguity retry reads")


def test_invalid_outcomes_are_spent() -> None:
    for forbidden_path in (
        protected.EVALUATOR_PATH,
        "evidence/objects/protected.json",
    ):
        forbidden = evidence.ArtifactRef(forbidden_path, 0, "0" * 64)
        with tempfile.TemporaryDirectory(prefix="ros3-alias-") as temporary:
            with mock.patch.object(
                evidence.ArtifactRef,
                "capture",
                side_effect=GateFailure("forbidden alias reached evaluator capture"),
            ):
                with expect_raises(protected.ProtectedEvaluationError, "aliases"):
                    run(Path(temporary), forbidden)

    with tempfile.TemporaryDirectory(prefix="ros3-hard-link-") as temporary:
        root, _ = make_repo(Path(temporary), canonical(inactive_outcome()))
        protected_path = root / PROTECTED_PATH
        protected_path.unlink()
        evaluator_path = root / protected.EVALUATOR_PATH
        os.link(evaluator_path, protected_path)
        evaluator_raw = evaluator_path.read_bytes()
        hard_link = evidence.ArtifactRef(
            PROTECTED_PATH,
            len(evaluator_raw),
            hashlib.sha256(evaluator_raw).hexdigest(),
        )
        with mock.patch.object(
            evidence.ArtifactRef,
            "capture",
            side_effect=GateFailure("hard-link alias reached evaluator capture"),
        ):
            with expect_raises(evidence.EvidenceError, "hard link"):
                run(root, hard_link)

    cases: dict[str, bytes] = {
        "unknown field": canonical(dict(inactive_outcome(), hidden=True)),
        "duplicate survivor": canonical(
            dict(inactive_outcome(), survivors=[ARMS[0], ARMS[0]])
        ),
        "reordered survivor": canonical(
            {
                **inactive_outcome(),
                "survivors": [ARMS[1], ARMS[0]],
                "exclusions": {ARMS[0]: "x", ARMS[1]: "x"},
            }
        ),
        "unregistered survivor": canonical(
            {
                **inactive_outcome(),
                "survivors": ["candidate.unregistered.v1"],
                "exclusions": {"candidate.unregistered.v1": "x"},
            }
        ),
        "missing exclusion": canonical(dict(inactive_outcome(), exclusions={})),
        "unknown exclusion": canonical(
            dict(inactive_outcome(), exclusions={ARMS[0]: "x", ARMS[1]: "x"})
        ),
        "empty exclusion": canonical(
            dict(inactive_outcome(), exclusions={ARMS[0]: ""})
        ),
        "non-string arm": canonical(dict(inactive_outcome(), survivors=[7])),
        "nonempty eligible": canonical(
            dict(inactive_outcome(), shadow_eligible=[ARMS[0]], exclusions={})
        ),
        "duplicate JSON key": (
            b'{"exclusions":{},"schema":"'
            + protected.OUTCOME_SCHEMA.encode()
            + b'","shadow_eligible":[],"survivors":[],"survivors":[]}'
        ),
        "noncanonical JSON": b'{"schema": "not-canonical"}',
    }
    for name, raw in cases.items():
        with tempfile.TemporaryDirectory(prefix="ros3-invalid-") as temporary:
            root, reference = make_repo(Path(temporary), raw)

            def evaluator_bomb(*args: Any, **kwargs: Any) -> dict[str, Any]:
                raise GateFailure(f"{name}: malformed input reached evaluator")

            with mock.patch.object(
                protected, "shadow_empty_set_status", new=evaluator_bomb
            ):
                with expect_raises(evidence.EvidenceError):
                    run(root, reference)
            kinds = [item.kind for item in domain_objects(root)]
            equal(len(kinds), 3, f"{name}: spent object count")
            equal(
                set(kinds),
                {protected.SPEC_KIND, protected.FAMILY_KIND, protected.RECEIPT_KIND},
                f"{name}: spent state",
            )
            retry_reads = 0
            real_read = evidence.ArtifactRef.read_verified

            def retry_bomb(
                self: evidence.ArtifactRef, *, repo_root: str | os.PathLike[str]
            ) -> bytes:
                nonlocal retry_reads
                if self.path == PROTECTED_PATH:
                    retry_reads += 1
                    raise GateFailure(f"{name}: retry reread")
                return real_read(self, repo_root=repo_root)

            with mock.patch.object(
                evidence.ArtifactRef, "read_verified", new=retry_bomb
            ):
                with expect_raises(protected.ProtectedEvaluationError, "BLOCKED"):
                    run(root, reference)
            equal(retry_reads, 0, f"{name}: retry read count")


def test_concurrent_single_reader() -> None:
    with tempfile.TemporaryDirectory(prefix="ros3-concurrent-") as raw:
        root, reference = make_repo(Path(raw), canonical(inactive_outcome()))
        spec, _ = protected._build_spec(
            repo_root=root,
            experiment_id=EXPERIMENT_ID,
            hypothesis=HYPOTHESIS,
            falsifier=FALSIFIER,
            protected_input=reference,
        )
        family = protected._build_family(spec, ARMS)
        evidence.publish(spec, repo_root=root)
        evidence.publish(family, repo_root=root)
        ready = root / "ready"
        reads = root / "reads"
        ready.mkdir()
        reads.mkdir()
        start = root / "start"
        code = "\n".join(
            (
                "import os, pathlib, sys, time",
                "sys.path.insert(0, 'scripts')",
                "import evidence_store as e",
                "import protected_evaluation as p",
                f"root = pathlib.Path({str(root)!r})",
                f"reference = e.ArtifactRef.from_dict({reference.as_dict()!r})",
                "real_read = e.ArtifactRef.read_verified",
                "def tracked(self, *, repo_root):",
                f"    if self.path == {PROTECTED_PATH!r}:",
                "        (root / 'reads' / str(os.getpid())).write_bytes(b'read')",
                "        time.sleep(0.4)",
                "    return real_read(self, repo_root=repo_root)",
                "e.ArtifactRef.read_verified = tracked",
                "(root / 'ready' / str(os.getpid())).write_bytes(b'ready')",
                "while not (root / 'start').exists(): time.sleep(0.01)",
                "try:",
                "    p.run_protected_evaluation(",
                "        repo_root=root,",
                f"        experiment_id={EXPERIMENT_ID!r},",
                f"        hypothesis={HYPOTHESIS!r},",
                f"        falsifier={FALSIFIER!r},",
                f"        ordered_arms={ARMS!r},",
                "        protected_input=reference,",
                "    )",
                "except e.EvidenceError:",
                "    raise SystemExit(3)",
            )
        )
        processes = [
            subprocess.Popen([sys.executable, "-c", code], cwd=REPO_ROOT)
            for _ in range(4)
        ]
        deadline = time.monotonic() + 10
        while len(list(ready.iterdir())) < len(processes):
            if time.monotonic() > deadline:
                raise GateFailure("concurrent workers did not reach barrier")
            time.sleep(0.01)
        start.write_bytes(b"start")
        returncodes = [process.wait(timeout=20) for process in processes]
        equal(returncodes.count(0), 1, "concurrent completed worker count")
        equal(returncodes.count(3), 3, "concurrent rejected worker count")
        equal(len(list(reads.iterdir())), 1, "concurrent protected reader count")
        equal(len(domain_objects(root)), 5, "concurrent terminal chain")
        equal(
            [path.name for path in (root / "evidence" / "objects").iterdir() if path.name.startswith(".")],
            [],
            "concurrent staging cleanup",
        )


def test_hidden_state_and_inactive_loading() -> None:
    with tempfile.TemporaryDirectory(prefix="ros3-hidden-") as raw:
        root, reference = make_repo(Path(raw), canonical(inactive_outcome()))
        terminal = run(root, reference)
        forged_payload = dict(terminal.payload)
        forged_payload["hidden"] = True
        forged = evidence.EvidenceEnvelope.create(
            kind=protected.TERMINAL_KIND,
            payload=forged_payload,
            dependencies=terminal.dependencies,
        )
        evidence.publish(forged, repo_root=root)
        (root / PROTECTED_PATH).unlink()
        with expect_raises(protected.ProtectedEvaluationError, "fields differ"):
            run(root, reference)

        candidate = terminal.payload["candidates"][0]
        identity = {
            "id": candidate["candidate_id"],
            "lifecycle_status": candidate["lifecycle_status"],
            "bundle_id": terminal.payload["evaluation_id"],
            "prereg_id": terminal.payload["spec_id"],
            "run_id": terminal.payload["receipt_id"],
        }
        index_entry = dict(identity)
        manifest_value = dict(identity, schema="book-manifest/candidate-v1")
        with mock.patch.object(
            book_runtime, "load_index", return_value={identity["id"]: index_entry}
        ), mock.patch.object(
            book_runtime,
            "load_registered_manifest",
            return_value=(Path("unused"), manifest_value),
        ), mock.patch.object(
            book_runtime,
            "_checked_book_dir",
            side_effect=GateFailure("inactive load reached model path"),
        ):
            with expect_raises(book_runtime.BookRuntimeError, "inactive"):
                book_runtime.load_book("SYNTHETIC", identity["id"])


def test_issue9_parity_registry_and_baselines() -> None:
    sidecar = evidence.verify_object(SIDECAR_ID, repo_root=REPO_ROOT)
    facts = sidecar.payload["facts"]["authoritative"]
    equal(facts["joint"]["S"], [], "issue-9 joint S")
    equal(
        {
            key: facts["shadow"][key]
            for key in ("status", "S_shadow", "k_shadow", "activation")
        },
        {
            "status": "no_candidates",
            "S_shadow": [],
            "k_shadow": 0,
            "activation": False,
        },
        "issue-9 shadow facts",
    )
    helper = shadow_empty_set_status([], [])
    equal(
        {key: helper[key] for key in ("status", "S_shadow", "k_shadow")},
        {"status": "no_candidates", "S_shadow": [], "k_shadow": 0},
        "issue-9 evaluator parity",
    )
    evidence.verify_store(repo_root=REPO_ROOT)
    registry = json.loads(
        (REPO_ROOT / "docs" / "research_os_capabilities.json").read_bytes()
    )
    evidence.validate_capability_registry(registry)
    ros3 = next(row for row in registry["capabilities"] if row["id"] == "ROS-3")
    equal(ros3["status"], "accepted", "ROS-3 capability state")
    equal(ros3["owner_issue"], "#12", "ROS-3 capability owner")
    for path, expected in BASELINE_HASHES.items():
        actual = hashlib.sha256((REPO_ROOT / path).read_bytes()).hexdigest()
        equal(actual, expected, f"protected baseline {path}")


def main() -> int:
    started = time.monotonic()
    timings = {
        "vertical": run_section(
            "vertical, resume, and determinism", test_vertical_resume_and_determinism
        ),
        "resume": run_section("resume state table", test_resume_state_table),
        "invalid": run_section(
            "invalid outcomes are spent", test_invalid_outcomes_are_spent
        ),
        "concurrency": run_section(
            "concurrent single reader", test_concurrent_single_reader
        ),
        "hidden": run_section(
            "hidden state and inactive loading", test_hidden_state_and_inactive_loading
        ),
        "parity": run_section(
            "issue-9 parity, registry, and baselines",
            test_issue9_parity_registry_and_baselines,
        ),
    }
    print(
        f"[test] PASS total={time.monotonic() - started:.2f}s "
        f"timings={json.dumps(timings, sort_keys=True)} "
        "temporary_mutations_only=true protected_reads_bounded=true "
        "models_fitted=false replay_invoked=false activation=false",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
