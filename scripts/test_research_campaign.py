#!/usr/bin/env python3
"""Temp-only acceptance gate for the research-campaign v1 control plane.

Run from the repository root::

    ~/binary-algo-venv/bin/python scripts/test_research_campaign.py
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

import evidence_store as evidence
import research_campaign_v1 as campaign


REPO_ROOT = Path(__file__).resolve().parents[1]
ADAPTER_PATH = "scripts/test_campaign_adapter_v1.py"
INPUT_PATH = "inputs/synthetic.json"
PROTECTED_PATHS = (
    "scripts/evidence_store.py",
    "scripts/protected_evaluation.py",
    "scripts/deriv_economics_ledger.py",
    "scripts/deriv_economics_ledger_v2.py",
    "scripts/m15_book_refresh.py",
    "docs/research_os_capabilities.json",
    "MODEL_REGISTRY.md",
    "books/INDEX.json",
)
PROTECTED_HASHES = {
    path: hashlib.sha256((REPO_ROOT / path).read_bytes()).hexdigest()
    for path in PROTECTED_PATHS
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


def canonical(value: Any) -> bytes:
    return evidence.canonical_json_bytes(value)


ADAPTER_SOURCE = r'''from __future__ import annotations

import json
import os
import time
from pathlib import Path

EVENTS = []


def _root():
    return Path(__file__).resolve().parents[1]


def _receipt(campaign, arm, status="completed", reasons=None, protected=False):
    return {
        "schema": "research-campaign-adapter-receipt/v1",
        "campaign_id": campaign["campaign_id"],
        "arm_id": arm["arm_id"],
        "native_result_contract": campaign["adapter"]["native_result_contract"],
        "status": status,
        "reasons": list(reasons or []),
        "native_result_path": arm["native_result_path"],
        "input_evidence_ids": list(arm["input_evidence_ids"]),
        "data_use": campaign["data_use"],
        "data_use_basis_evidence_ids": list(arm["data_use_basis_evidence_ids"]),
        "protected_access": protected,
    }


def _read_native(path, campaign, arm):
    value = json.loads(path.read_bytes())
    expected = {
        "arm_id": arm["arm_id"],
        "campaign_id": campaign["campaign_id"],
        "contract": campaign["adapter"]["native_result_contract"],
        "data_use": campaign["data_use"],
        "status": "synthetic_complete",
    }
    if value != expected:
        raise ValueError("native contract differs")


def preflight_arm(*, campaign, arm, input_evidence):
    EVENTS.append(("preflight", arm["arm_id"]))
    kinds = []
    for path in (_root() / "evidence" / "objects").glob("*.json"):
        kinds.append(json.loads(path.read_bytes())["kind"])
    if "research.campaign.spec/v1" not in kinds:
        raise ValueError("specification was not published before preflight")
    if any((_root() / item).exists() for item in [arm["native_result_path"]]):
        raise ValueError("native output existed during preflight")
    input_ids = [item["object_id"] for item in input_evidence]
    basis_by_id = {item["object_id"]: item for item in input_evidence}
    basis = [basis_by_id.get(item) for item in arm["data_use_basis_evidence_ids"]]
    inputs_match = input_ids == list(arm["input_evidence_ids"])
    classification_matches = all(
        item is not None and item["payload"].get("data_use") == campaign["data_use"]
        for item in basis
    )
    protected = campaign["question"] == "protected-preflight"
    admissible = (
        campaign["question"] != "inadmissible"
        and inputs_match
        and classification_matches
    )
    reasons = []
    if campaign["question"] == "inadmissible":
        reasons.append("fixture_preflight_rejection")
    if not inputs_match:
        reasons.append("input_identity_mismatch")
    if not classification_matches:
        reasons.append("data_use_basis_mismatch")
    reported_data_use = campaign["data_use"]
    if campaign["question"] == "bad-preflight-data-use":
        reported_data_use = (
            "retrospective" if reported_data_use == "synthetic" else "synthetic"
        )
    return {
        "campaign_id": campaign["campaign_id"],
        "arm_id": arm["arm_id"],
        "data_use": reported_data_use,
        "data_use_basis_evidence_ids": list(arm["data_use_basis_evidence_ids"]),
        "protected_access": protected,
        "admissible": admissible,
        "reasons": reasons,
    }


def run_arm(*, campaign, arm, temporary_output_path):
    if (_root() / "logs" / "forbid-arm-calls").exists():
        raise RuntimeError("arm call forbidden after completion")
    EVENTS.append(("run", arm["arm_id"]))
    run_log = _root() / "logs" / "adapter-runs.log"
    with run_log.open("a", encoding="utf-8") as handle:
        handle.write(campaign["campaign_id"] + ":" + arm["arm_id"] + "\n")
    if campaign["question"] == "slow":
        (_root() / "logs" / "adapter-active").write_bytes(b"active")
        time.sleep(0.8)
    native = {
        "arm_id": arm["arm_id"],
        "campaign_id": campaign["campaign_id"],
        "contract": campaign["adapter"]["native_result_contract"],
        "data_use": campaign["data_use"],
        "status": "synthetic_complete",
    }
    if campaign["question"] == "hardlink-temp":
        os.link(_root() / "inputs" / "synthetic.json", temporary_output_path)
        return _receipt(campaign, arm)
    temporary_output_path.write_text(
        json.dumps(native, sort_keys=True, separators=(",", ":")), encoding="utf-8"
    )
    if campaign["question"] == "fail-once":
        marker = _root() / "logs" / "failed-once"
        if not marker.exists():
            marker.write_bytes(b"failed")
            raise RuntimeError("injected run failure")
    protected = campaign["question"] == "protected-receipt"
    receipt = _receipt(campaign, arm, protected=protected)
    if campaign["question"] == "bad-receipt":
        receipt["input_evidence_ids"] = []
    if campaign["question"] == "bad-receipt-data-use":
        receipt["data_use"] = (
            "retrospective" if receipt["data_use"] == "synthetic" else "synthetic"
        )
    if campaign["question"] == "bad-receipt-status":
        receipt["status"] = []
    return receipt


def validate_arm_result(*, campaign, arm, native_result_path):
    if (_root() / "logs" / "forbid-arm-calls").exists():
        raise RuntimeError("validation forbidden after completion")
    EVENTS.append(("validate", arm["arm_id"]))
    _read_native(native_result_path, campaign, arm)
    return _receipt(campaign, arm)


def evaluate_family(*, campaign, receipts, native_results):
    EVENTS.append(("evaluate", campaign["campaign_id"]))
    if len(receipts) != len(campaign["ordered_arms"]):
        raise ValueError("family is incomplete")
    verdicts = []
    candidates = []
    for arm, receipt in zip(campaign["ordered_arms"], receipts):
        if receipt["status"] == "not_computable":
            status, reasons = "not_computable", ["adapter_not_computable"]
        elif arm["role"] == "candidate":
            status, reasons = "candidate", []
            candidates.append(arm["arm_id"])
        else:
            status, reasons = "killed", ["synthetic_control"]
        verdicts.append({"arm_id": arm["arm_id"], "status": status, "reasons": reasons})
    decision = "inactive_candidate" if candidates else "no_candidate"
    if campaign["question"] == "bad-decision":
        candidates = list(reversed(candidates)) + ["hidden"]
    if campaign["question"] == "bad-verdict-status":
        verdicts[0]["status"] = []
    return {
        "schema": "research-campaign-family-decision/v1",
        "campaign_id": campaign["campaign_id"],
        "verdicts": verdicts,
        "ordered_candidates": candidates,
        "decision": decision,
        "reasons": [],
    }
'''


def make_repo(
    root: Path,
    *,
    adapter_source: str = ADAPTER_SOURCE,
    data_use: str = "synthetic",
) -> tuple[Path, str]:
    for directory in ("scripts", "results/json", "evidence/objects", "logs", "inputs"):
        (root / directory).mkdir(parents=True, exist_ok=True)
    for relative in (
        campaign.RUNNER_PATH,
        "scripts/evidence_store.py",
        "scripts/manifest.py",
    ):
        shutil.copyfile(REPO_ROOT / relative, root / relative)
    (root / ADAPTER_PATH).write_text(adapter_source, encoding="utf-8")
    input_path = root / INPUT_PATH
    input_path.write_bytes(canonical({"data_use": data_use, "rows": 3}))
    input_ref = evidence.ArtifactRef.capture(INPUT_PATH, repo_root=root)
    input_envelope = evidence.EvidenceEnvelope.create(
        kind="test.research_campaign.input/v1",
        payload={"schema": "test.research_campaign.input/v1", "data_use": data_use},
        artifacts=[input_ref],
    )
    evidence.publish(input_envelope, repo_root=root)
    return root, input_envelope.object_id


def spec_value(
    input_id: str,
    *,
    campaign_id: str = "synthetic.campaign.v1",
    question: str = "normal",
    adapter_path: str = ADAPTER_PATH,
    result_suffix: str = "",
) -> dict[str, Any]:
    arms = []
    for arm_id, role in (("candidate.v1", "candidate"), ("control.v1", "control")):
        slug = arm_id.replace(".", "_")
        arms.append(
            {
                "arm_id": arm_id,
                "role": role,
                "native_result_path": f"results/json/{slug}{result_suffix}_result.json",
                "input_evidence_ids": [input_id],
                "data_use_basis_evidence_ids": [input_id],
            }
        )
    return {
        "schema": campaign.INPUT_SCHEMA,
        "campaign_id": campaign_id,
        "key": {
            "pair": "SYNTHETIC",
            "timeframe": "1m",
            "target": "direction",
            "side": "UP",
        },
        "question": question,
        "hypothesis": "The synthetic candidate exercises the campaign control plane.",
        "falsifier": "Reject any unsealed, reordered, protected, or non-reproducible state.",
        "data_use": "synthetic",
        "ordered_arms": arms,
        "adapter": {
            "path": adapter_path,
            "run_function": "run_arm",
            "preflight_function": "preflight_arm",
            "validate_function": "validate_arm_result",
            "evaluate_function": "evaluate_family",
            "native_result_contract": "test-native-result/v1",
        },
        "stopping_rule": "Evaluate the complete sealed two-arm family exactly once.",
        "terminal_rule": {
            "allowed_decisions": [
                "no_candidate",
                "inactive_candidate",
                "capability_deferred",
            ],
            "activation": False,
        },
    }


def write_spec(root: Path, value: dict[str, Any], name: str = "campaign.json") -> Path:
    path = root / name
    path.write_bytes(canonical(value))
    return path


def domain_objects(root: Path) -> list[evidence.EvidenceEnvelope]:
    out = []
    for item in evidence.inspect_store_metadata(repo_root=root):
        if item.kind in campaign.DOMAIN_KINDS:
            out.append(evidence.verify_object(item.object_id, repo_root=root))
    return out


def run_count(root: Path, campaign_id: str | None = None) -> int:
    path = root / "logs" / "adapter-runs.log"
    if not path.exists():
        return 0
    lines = path.read_text(encoding="utf-8").splitlines()
    if campaign_id is not None:
        lines = [line for line in lines if line.startswith(campaign_id + ":")]
    return len(lines)


def test_end_to_end_resume_and_successor() -> None:
    with tempfile.TemporaryDirectory(prefix="research-campaign-a-") as raw:
        root, input_id = make_repo(Path(raw))
        spec = spec_value(input_id)
        path = write_spec(root, spec)
        fsynced_results: list[str] = []
        real_fsync_regular = campaign._fsync_regular

        def track_result_fsync(result_path: Path) -> None:
            real_fsync_regular(result_path)
            fsynced_results.append(result_path.name)

        with mock.patch.object(
            campaign, "_fsync_regular", new=track_result_fsync
        ):
            terminal = campaign.run_campaign(path, repo_root=root)
        equal(terminal.kind, campaign.TERMINAL_KIND, "terminal kind")
        equal(
            terminal.payload["family_decision"]["decision"],
            "inactive_candidate",
            "inactive decision",
        )
        equal(terminal.payload["activation"], False, "terminal activation")
        equal(run_count(root), 2, "one run per arm")
        equal(len(fsynced_results), 2, "one durable file fsync per installed arm")
        kinds = [item.kind for item in domain_objects(root)]
        equal(kinds.count(campaign.SPEC_KIND), 1, "spec count")
        equal(kinds.count(campaign.ATTEMPT_KIND), 2, "attempt count")
        equal(kinds.count(campaign.RESULT_KIND), 2, "result count")
        equal(kinds.count(campaign.TERMINAL_KIND), 1, "terminal count")

        (root / "logs" / "forbid-arm-calls").write_bytes(b"forbid")
        verified = campaign.verify_campaign(spec["campaign_id"], repo_root=root)
        equal(verified.as_dict(), terminal.as_dict(), "lock-free terminal verification")
        resumed = campaign.run_campaign(path, repo_root=root)
        equal(resumed.as_dict(), terminal.as_dict(), "completed run resume")
        equal(run_count(root), 2, "completed resume arm calls")
        (root / "logs" / "forbid-arm-calls").unlink()

        stale = campaign._temporary_path(root, spec["campaign_id"], spec["ordered_arms"][0])
        stale.write_bytes(b"stale")
        with expect_raises(campaign.ResearchCampaignError, "hidden temporary"):
            campaign.verify_campaign(spec["campaign_id"], repo_root=root)
        stale.unlink()

        successor_path = "scripts/test_campaign_adapter_v2.py"
        (root / successor_path).write_text(
            ADAPTER_SOURCE + "\nADAPTER_VERSION = 2\n", encoding="utf-8"
        )
        successor = spec_value(
            input_id,
            campaign_id="synthetic.campaign.v2",
            adapter_path=successor_path,
            result_suffix="_v2",
        )
        successor_terminal = campaign.run_campaign(
            write_spec(root, successor, "successor.json"), repo_root=root
        )
        equal(successor_terminal.payload["activation"], False, "successor inactive")
        evidence.verify_store(repo_root=root)
        equal(
            campaign.verify_campaign(spec["campaign_id"], repo_root=root).object_id,
            terminal.object_id,
            "old campaign after successor",
        )

        clone = root.parent / f"{root.name}-clone"
        shutil.copytree(root, clone)
        completed = subprocess.run(
            [
                sys.executable,
                "scripts/evidence_store.py",
                "--repo-root",
                str(clone),
                "verify-store",
            ],
            cwd=clone,
            check=False,
            capture_output=True,
            text=True,
        )
        equal(completed.returncode, 0, "fresh clone verify-store")
        check("verified store" in completed.stdout, "fresh clone verification receipt")
        shutil.rmtree(clone)

    with tempfile.TemporaryDirectory(prefix="research-campaign-b-") as raw:
        root, input_id = make_repo(Path(raw))
        terminal_b = campaign.run_campaign(
            write_spec(root, spec_value(input_id)), repo_root=root
        )
        equal(terminal_b.object_id, terminal.object_id, "terminal determinism")

    with tempfile.TemporaryDirectory(prefix="research-campaign-retrospective-") as raw:
        root, input_id = make_repo(Path(raw), data_use="retrospective")
        retrospective = spec_value(
            input_id, campaign_id="retrospective.campaign.v1"
        )
        retrospective["data_use"] = "retrospective"
        retrospective_terminal = campaign.run_campaign(
            write_spec(root, retrospective), repo_root=root
        )
        equal(
            retrospective_terminal.payload["data_use"],
            "retrospective",
            "retrospective terminal classification",
        )
        equal(
            retrospective_terminal.payload["family_decision"]["decision"],
            "inactive_candidate",
            "retrospective candidate remains inactive",
        )
        equal(
            retrospective_terminal.payload["activation"],
            False,
            "retrospective terminal activation",
        )

    with tempfile.TemporaryDirectory(prefix="research-campaign-nested-") as raw:
        root, input_id = make_repo(Path(raw))
        nested = spec_value(input_id, campaign_id="synthetic.nested.v1")
        nested["ordered_arms"][0]["native_result_path"] = (
            "results/json/nested/candidate_result.json"
        )
        nested_terminal = campaign.run_campaign(
            write_spec(root, nested), repo_root=root
        )
        equal(nested_terminal.payload["activation"], False, "nested inactive")
        check(
            (root / "results/json/nested/candidate_result.json").is_file(),
            "nested result was not installed",
        )


def test_recovery_table() -> None:
    with tempfile.TemporaryDirectory(prefix="research-campaign-retry-") as raw:
        root, input_id = make_repo(Path(raw))
        spec = spec_value(input_id, question="fail-once")
        path = write_spec(root, spec)
        with expect_raises(campaign.ResearchCampaignError, "injected run failure"):
            campaign.run_campaign(path, repo_root=root)
        kinds = [item.kind for item in domain_objects(root)]
        equal(kinds.count(campaign.ATTEMPT_KIND), 1, "attempt-only state")
        equal(kinds.count(campaign.RESULT_KIND), 0, "attempt-only result count")
        check(
            any((root / "results" / "json").glob(".research-campaign-*.tmp")),
            "missing partial temp",
        )
        terminal = campaign.run_campaign(path, repo_root=root)
        equal(terminal.payload["activation"], False, "retry inactive")
        equal(run_count(root), 3, "attempt retry plus second arm")
        equal(
            list((root / "results" / "json").glob(".research-campaign-*.tmp")),
            [],
            "temp cleanup",
        )

    with tempfile.TemporaryDirectory(prefix="research-campaign-adopt-") as raw:
        root, input_id = make_repo(Path(raw))
        spec = spec_value(input_id)
        path = write_spec(root, spec)
        real_publish = evidence.publish
        failed = False

        def stop_before_result(
            envelope: evidence.EvidenceEnvelope,
            *,
            repo_root: str | os.PathLike[str],
            require_new: bool = False,
        ) -> Path:
            nonlocal failed
            if envelope.kind == campaign.RESULT_KIND and not failed:
                failed = True
                raise evidence.EvidenceError("injected before result publication")
            return real_publish(envelope, repo_root=repo_root, require_new=require_new)

        with mock.patch.object(evidence, "publish", new=stop_before_result):
            with expect_raises(evidence.EvidenceError, "injected"):
                campaign.run_campaign(path, repo_root=root)
        equal(run_count(root), 1, "pre-adoption run count")
        check(
            (root / spec["ordered_arms"][0]["native_result_path"]).exists(),
            "installed result absent",
        )
        terminal = campaign.run_campaign(path, repo_root=root)
        equal(terminal.payload["activation"], False, "adopted inactive")
        equal(run_count(root), 2, "adoption reran an installed arm")

    with tempfile.TemporaryDirectory(prefix="research-campaign-malformed-adopt-") as raw:
        root, input_id = make_repo(Path(raw))
        spec = spec_value(input_id)
        path = write_spec(root, spec)
        real_publish = evidence.publish
        failed = False

        def stop_for_malformed_adoption(
            envelope: evidence.EvidenceEnvelope,
            *,
            repo_root: str | os.PathLike[str],
            require_new: bool = False,
        ) -> Path:
            nonlocal failed
            if envelope.kind == campaign.RESULT_KIND and not failed:
                failed = True
                raise evidence.EvidenceError("injected before malformed adoption")
            return real_publish(envelope, repo_root=repo_root, require_new=require_new)

        with mock.patch.object(evidence, "publish", new=stop_for_malformed_adoption):
            with expect_raises(evidence.EvidenceError, "injected"):
                campaign.run_campaign(path, repo_root=root)
        final = root / spec["ordered_arms"][0]["native_result_path"]
        final.chmod(0o644)
        final.write_bytes(b"{}")
        with expect_raises(campaign.ResearchCampaignError, "native contract differs"):
            campaign.run_campaign(path, repo_root=root)
        equal(run_count(root), 1, "malformed adoption reran workflow")

    with tempfile.TemporaryDirectory(prefix="research-campaign-terminal-") as raw:
        root, input_id = make_repo(Path(raw))
        spec = spec_value(input_id)
        path = write_spec(root, spec)
        real_publish = evidence.publish

        def stop_before_terminal(
            envelope: evidence.EvidenceEnvelope,
            *,
            repo_root: str | os.PathLike[str],
            require_new: bool = False,
        ) -> Path:
            if envelope.kind == campaign.TERMINAL_KIND:
                raise evidence.EvidenceError("injected before terminal publication")
            return real_publish(envelope, repo_root=repo_root, require_new=require_new)

        with mock.patch.object(evidence, "publish", new=stop_before_terminal):
            with expect_raises(evidence.EvidenceError, "injected"):
                campaign.run_campaign(path, repo_root=root)
        equal(run_count(root), 2, "all-results run count")
        equal(
            [item.kind for item in domain_objects(root)].count(campaign.TERMINAL_KIND),
            0,
            "terminal unexpectedly published",
        )
        terminal = campaign.run_campaign(path, repo_root=root)
        equal(run_count(root), 2, "all-results resume reran arms")
        equal(terminal.payload["activation"], False, "resumed terminal inactive")

    with tempfile.TemporaryDirectory(prefix="research-campaign-foreign-") as raw:
        root, input_id = make_repo(Path(raw))
        spec = spec_value(input_id)
        final = root / spec["ordered_arms"][0]["native_result_path"]
        final.write_bytes(b"{}")
        with expect_raises(campaign.ResearchCampaignError, "without the exact prior attempt"):
            campaign.run_campaign(write_spec(root, spec), repo_root=root)
        equal(run_count(root), 0, "foreign result invoked adapter")


def test_strict_inputs_and_fail_closed_outputs() -> None:
    mutations: dict[str, Callable[[dict[str, Any], Path], None]] = {
        "unknown field": lambda value, root: value.update(hidden=True),
        "fresh oos": lambda value, root: value.update(data_use="fresh_oos"),
        "malformed data use": lambda value, root: value.update(data_use=[]),
        "invalid id": lambda value, root: value.update(campaign_id="Bad ID"),
        "oversized id": lambda value, root: value.update(campaign_id="a" * 129),
        "path escape": lambda value, root: value["ordered_arms"][0].update(
            native_result_path="../escape_result.json"
        ),
        "duplicate arm": lambda value, root: value["ordered_arms"].__setitem__(
            1, dict(value["ordered_arms"][0])
        ),
        "duplicate result": lambda value, root: value["ordered_arms"][1].update(
            native_result_path=value["ordered_arms"][0]["native_result_path"]
        ),
        "duplicate input": lambda value, root: value["ordered_arms"][0].update(
            input_evidence_ids=[input_id, input_id]
        ),
        "wrong function": lambda value, root: value["adapter"].update(run_function="other"),
        "activation": lambda value, root: value["terminal_rule"].update(activation=True),
    }
    for name, mutate in mutations.items():
        with tempfile.TemporaryDirectory(prefix="research-campaign-invalid-") as raw:
            root, input_id = make_repo(Path(raw))
            value = spec_value(input_id)
            mutate(value, root)
            with expect_raises(evidence.EvidenceError):
                campaign.run_campaign(write_spec(root, value), repo_root=root)
            equal(domain_objects(root), [], f"{name}: invalid spec published state")
            equal(run_count(root), 0, f"{name}: adapter run count")

    with tempfile.TemporaryDirectory(prefix="research-campaign-noncanonical-") as raw:
        root, input_id = make_repo(Path(raw))
        path = root / "campaign.json"
        path.write_text(json.dumps(spec_value(input_id), indent=2), encoding="utf-8")
        with expect_raises(evidence.EvidenceError, "canonical"):
            campaign.run_campaign(path, repo_root=root)

    with tempfile.TemporaryDirectory(prefix="research-campaign-nonfinite-") as raw:
        root, input_id = make_repo(Path(raw))
        raw_spec = canonical(spec_value(input_id)).replace(
            b'"question":"normal"', b'"question":NaN'
        )
        path = root / "campaign.json"
        path.write_bytes(raw_spec)
        with expect_raises(evidence.EvidenceError, "non-finite"):
            campaign.run_campaign(path, repo_root=root)

    with tempfile.TemporaryDirectory(prefix="research-campaign-symlink-") as raw:
        root, input_id = make_repo(Path(raw))
        real_adapter = root / ADAPTER_PATH
        alternate = root / "scripts" / "adapter-real.py"
        real_adapter.rename(alternate)
        real_adapter.symlink_to(alternate.name)
        with expect_raises(evidence.EvidenceError, "symlink"):
            campaign.run_campaign(write_spec(root, spec_value(input_id)), repo_root=root)

    for question, expected, expected_runs in (
        ("inadmissible", "preflight rejected", 0),
        ("protected-receipt", "protected_access", 1),
        ("bad-receipt", "input_evidence_ids", 1),
        ("bad-decision", "ordered_candidates", 2),
        ("hardlink-temp", "singly linked", 1),
        ("bad-preflight-data-use", "preflight data_use", 0),
        ("bad-receipt-data-use", "receipt data_use", 1),
        ("bad-receipt-status", "receipt status", 1),
        ("bad-verdict-status", "verdict status", 2),
    ):
        with tempfile.TemporaryDirectory(prefix="research-campaign-adapter-invalid-") as raw:
            root, input_id = make_repo(Path(raw))
            value = spec_value(input_id, question=question)
            with expect_raises(evidence.EvidenceError, expected):
                campaign.run_campaign(write_spec(root, value), repo_root=root)
            equal(
                [item.kind for item in domain_objects(root)].count(campaign.TERMINAL_KIND),
                0,
                f"{question}: invalid terminal count",
            )
            equal(run_count(root), expected_runs, f"{question}: adapter run count")

    with tempfile.TemporaryDirectory(prefix="research-campaign-basis-mismatch-") as raw:
        root, input_id = make_repo(Path(raw))
        value = spec_value(input_id)
        value["data_use"] = "retrospective"
        with expect_raises(campaign.ResearchCampaignError, "preflight rejected"):
            campaign.run_campaign(write_spec(root, value), repo_root=root)
        equal(run_count(root), 0, "classification mismatch invoked adapter run")

    with tempfile.TemporaryDirectory(prefix="research-campaign-drift-") as raw:
        root, input_id = make_repo(Path(raw))
        value = spec_value(input_id, question="fail-once")
        path = write_spec(root, value)
        with expect_raises(campaign.ResearchCampaignError, "injected"):
            campaign.run_campaign(path, repo_root=root)
        (root / INPUT_PATH).write_bytes(b"changed")
        with expect_raises(evidence.EvidenceError, "mismatch"):
            campaign.run_campaign(path, repo_root=root)

    with tempfile.TemporaryDirectory(prefix="research-campaign-source-drift-") as raw:
        root, input_id = make_repo(Path(raw))
        value = spec_value(input_id, question="fail-once")
        path = write_spec(root, value)
        with expect_raises(campaign.ResearchCampaignError, "injected"):
            campaign.run_campaign(path, repo_root=root)
        (root / ADAPTER_PATH).write_text(ADAPTER_SOURCE + "\n# changed\n", encoding="utf-8")
        with expect_raises(evidence.EvidenceError, "mismatch"):
            campaign.run_campaign(path, repo_root=root)


def test_hidden_state_and_concurrency() -> None:
    with tempfile.TemporaryDirectory(prefix="research-campaign-malformed-arm-") as raw:
        root, input_id = make_repo(Path(raw))
        value = spec_value(input_id)
        path = write_spec(root, value)
        sealed_spec = campaign._build_spec(root, value)
        evidence.publish(sealed_spec, repo_root=root)
        first_arm = value["ordered_arms"][0]
        preflight = {
            "campaign_id": value["campaign_id"],
            "arm_id": first_arm["arm_id"],
            "data_use": value["data_use"],
            "data_use_basis_evidence_ids": first_arm["data_use_basis_evidence_ids"],
            "protected_access": False,
            "admissible": True,
            "reasons": [],
        }
        valid_attempt = campaign._build_attempt(sealed_spec, 0, preflight)
        malformed_payload = dict(valid_attempt.payload)
        malformed_payload["arm_id"] = [first_arm["arm_id"]]
        malformed_attempt = evidence.EvidenceEnvelope.create(
            kind=campaign.ATTEMPT_KIND,
            payload=malformed_payload,
            dependencies=valid_attempt.dependencies,
        )
        evidence.publish(malformed_attempt, repo_root=root)
        with expect_raises(campaign.ResearchCampaignError, "malformed"):
            campaign.run_campaign(path, repo_root=root)
        equal(run_count(root), 0, "malformed descendant invoked adapter run")

    with tempfile.TemporaryDirectory(prefix="research-campaign-future-arm-") as raw:
        root, input_id = make_repo(Path(raw))
        value = spec_value(input_id)
        path = write_spec(root, value)
        sealed_spec = campaign._build_spec(root, value)
        evidence.publish(sealed_spec, repo_root=root)
        future_arm = value["ordered_arms"][1]
        preflight = {
            "campaign_id": value["campaign_id"],
            "arm_id": future_arm["arm_id"],
            "data_use": value["data_use"],
            "data_use_basis_evidence_ids": future_arm["data_use_basis_evidence_ids"],
            "protected_access": False,
            "admissible": True,
            "reasons": [],
        }
        evidence.publish(
            campaign._build_attempt(sealed_spec, 1, preflight), repo_root=root
        )
        with expect_raises(campaign.ResearchCampaignError, "next sealed arm"):
            campaign.run_campaign(path, repo_root=root)
        equal(run_count(root), 0, "future attempt invoked adapter run")

    with tempfile.TemporaryDirectory(prefix="research-campaign-hidden-") as raw:
        root, input_id = make_repo(Path(raw))
        value = spec_value(input_id)
        terminal = campaign.run_campaign(write_spec(root, value), repo_root=root)
        forged_payload = dict(terminal.payload)
        forged_payload["claim_limit"] = "forged but structurally valid claim"
        forged = evidence.EvidenceEnvelope.create(
            kind=campaign.TERMINAL_KIND,
            payload=forged_payload,
            dependencies=terminal.dependencies,
        )
        evidence.publish(forged, repo_root=root)
        with expect_raises(campaign.ResearchCampaignError, "hidden or conflicting terminal"):
            campaign.verify_campaign(value["campaign_id"], repo_root=root)

    with tempfile.TemporaryDirectory(prefix="research-campaign-concurrent-") as raw:
        root, input_id = make_repo(Path(raw))
        first = spec_value(input_id, question="slow")
        first_path = write_spec(root, first, "first.json")
        second = spec_value(
            input_id,
            campaign_id="synthetic.concurrent.second.v1",
            result_suffix="_second",
        )
        second_path = write_spec(root, second, "second.json")
        code = "\n".join(
            (
                "import pathlib, sys",
                "sys.path.insert(0, 'scripts')",
                "import research_campaign_v1 as c",
                "c.run_campaign("
                f"pathlib.Path({str(first_path)!r}), "
                f"repo_root=pathlib.Path({str(root)!r}))",
            )
        )
        process = subprocess.Popen([sys.executable, "-c", code], cwd=REPO_ROOT)
        deadline = time.monotonic() + 10
        active = root / "logs" / "adapter-active"
        while not active.exists():
            if time.monotonic() > deadline:
                process.kill()
                raise GateFailure("concurrent adapter did not become active")
            time.sleep(0.01)
        for path in (first_path, second_path):
            with expect_raises(campaign.ResearchCampaignError, "already active"):
                campaign.run_campaign(path, repo_root=root)
        started = time.monotonic()
        with expect_raises(campaign.ResearchCampaignError, "incomplete"):
            campaign.verify_campaign(first["campaign_id"], repo_root=root)
        check(time.monotonic() - started < 0.5, "verify_campaign waited on mutating lock")
        equal(process.wait(timeout=10), 0, "concurrent primary process")
        equal(run_count(root, first["campaign_id"]), 2, "concurrent first arm count")
        equal(run_count(root, second["campaign_id"]), 0, "concurrent second arm count")


def test_repository_protected_surfaces() -> None:
    evidence.verify_store(repo_root=REPO_ROOT)
    after = {
        path: hashlib.sha256((REPO_ROOT / path).read_bytes()).hexdigest()
        for path in PROTECTED_PATHS
    }
    equal(after, PROTECTED_HASHES, "protected repository surface hashes")


def main() -> int:
    started = time.monotonic()
    timings = {
        "vertical": run_section(
            "end-to-end, resume, determinism, and successor",
            test_end_to_end_resume_and_successor,
        ),
        "recovery": run_section("crash recovery table", test_recovery_table),
        "invalid": run_section(
            "strict inputs and fail-closed adapter outputs",
            test_strict_inputs_and_fail_closed_outputs,
        ),
        "concurrency": run_section(
            "hidden state and repository-wide concurrency",
            test_hidden_state_and_concurrency,
        ),
        "protected": run_section(
            "protected repository surfaces", test_repository_protected_surfaces
        ),
    }
    print(
        f"[test] PASS total={time.monotonic() - started:.2f}s "
        f"timings={json.dumps(timings, sort_keys=True)} "
        "temporary_mutations_only=true models_fitted=false "
        "strategy_evaluated=false protected_reads=0 activation=false "
        "external_actions=false",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
