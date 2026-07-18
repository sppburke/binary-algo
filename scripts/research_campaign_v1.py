#!/usr/bin/env python3
"""Adapter-driven, resumable control plane for unprotected research campaigns.

The runner seals exact campaign, code, input, and arm identities before it calls
an adapter.  Scientific meaning remains adapter-owned; this module only owns
ordering, recovery, immutable evidence bindings, and an inactive terminal.
"""

from __future__ import annotations

import argparse
import errno
import fcntl
import hashlib
import os
import re
import stat
import sys
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from types import MappingProxyType, ModuleType
from typing import Any, Iterator, Mapping, Sequence

import evidence_store as evidence


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = "scripts/research_campaign_v1.py"
INPUT_SCHEMA = "research-campaign-spec-input/v1"
SPEC_KIND = "research.campaign.spec/v1"
ATTEMPT_KIND = "research.campaign.arm_attempt/v1"
RESULT_KIND = "research.campaign.arm_result/v1"
TERMINAL_KIND = "research.campaign.terminal/v1"
RECEIPT_SCHEMA = "research-campaign-adapter-receipt/v1"
DECISION_SCHEMA = "research-campaign-family-decision/v1"
DOMAIN_KINDS = (SPEC_KIND, ATTEMPT_KIND, RESULT_KIND, TERMINAL_KIND)
AUTHORIZATION_STATE = "authorized_unprotected_retryable"
CLAIM_LIMIT = (
    "The runner established no fresh OOS result, certification, strategy edge, "
    "activation, deployment, purchase, or ledger/book/registry authority."
)
CAMPAIGN_ID = re.compile(r"[a-z0-9](?:[a-z0-9._-]{0,127})")
HEX64 = re.compile(r"[0-9a-f]{64}")
SPEC_FIELDS = {
    "schema",
    "campaign_id",
    "key",
    "question",
    "hypothesis",
    "falsifier",
    "data_use",
    "ordered_arms",
    "adapter",
    "stopping_rule",
    "terminal_rule",
}
KEY_FIELDS = {"pair", "timeframe", "target", "side"}
ARM_FIELDS = {
    "arm_id",
    "role",
    "native_result_path",
    "input_evidence_ids",
    "data_use_basis_evidence_ids",
}
ADAPTER_FIELDS = {
    "path",
    "run_function",
    "preflight_function",
    "validate_function",
    "evaluate_function",
    "native_result_contract",
}
TERMINAL_RULE_FIELDS = {"allowed_decisions", "activation"}
PREFLIGHT_FIELDS = {
    "campaign_id",
    "arm_id",
    "data_use",
    "data_use_basis_evidence_ids",
    "protected_access",
    "admissible",
    "reasons",
}
RECEIPT_FIELDS = {
    "schema",
    "campaign_id",
    "arm_id",
    "native_result_contract",
    "status",
    "reasons",
    "native_result_path",
    "input_evidence_ids",
    "data_use",
    "data_use_basis_evidence_ids",
    "protected_access",
}
DECISION_FIELDS = {
    "schema",
    "campaign_id",
    "verdicts",
    "ordered_candidates",
    "decision",
    "reasons",
}
VERDICT_FIELDS = {"arm_id", "status", "reasons"}
SPEC_PAYLOAD_FIELDS = {"schema", "campaign_id", "campaign"}
ATTEMPT_PAYLOAD_FIELDS = {
    "schema",
    "campaign_id",
    "spec_id",
    "arm_index",
    "arm_id",
    "final_path",
    "preflight",
    "authorization_state",
}
RESULT_PAYLOAD_FIELDS = {
    "schema",
    "campaign_id",
    "spec_id",
    "attempt_id",
    "arm_index",
    "arm_id",
    "receipt",
}
TERMINAL_PAYLOAD_FIELDS = {
    "schema",
    "campaign_id",
    "spec_id",
    "family_decision",
    "evaluator",
    "ordered_candidates",
    "data_use",
    "activation",
    "claim_limit",
}


class ResearchCampaignError(evidence.EvidenceError):
    """The research-campaign contract failed closed."""


def _exact_fields(value: Any, fields: set[str], name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ResearchCampaignError(f"{name} must be a JSON object")
    actual = set(value)
    if actual != fields:
        raise ResearchCampaignError(
            f"{name} fields differ: "
            f"missing={sorted(fields - actual)} unknown={sorted(actual - fields)}"
        )
    return value


def _text(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value.strip() != value
        or any(ord(character) < 0x20 for character in value)
    ):
        raise ResearchCampaignError(f"{name} must be a non-empty canonical string")
    return value


def _campaign_id(value: Any) -> str:
    value = _text(value, "campaign_id")
    if CAMPAIGN_ID.fullmatch(value) is None or value in {".", ".."}:
        raise ResearchCampaignError("campaign_id must be a 1-128 character lowercase slug")
    return value


def _hex_id(value: Any, name: str) -> str:
    if not isinstance(value, str) or HEX64.fullmatch(value) is None:
        raise ResearchCampaignError(f"{name} must be lowercase 64-hex")
    return value


def _canonical_path(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise ResearchCampaignError(f"{name} must be a canonical POSIX path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or value != path.as_posix()
        or value in {".", ".."}
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ResearchCampaignError(f"{name} is not canonical repo-relative POSIX")
    return value


def _json_copy(value: Any) -> Any:
    return evidence.decode_canonical_json(evidence.canonical_json_bytes(value))


def _ordered_text(
    value: Any, name: str, *, allow_empty: bool = False, unique: bool = False
) -> list[str]:
    if type(value) is not list:
        raise ResearchCampaignError(f"{name} must be an ordered JSON list")
    out = [_text(item, f"{name}[{index}]") for index, item in enumerate(value)]
    if not allow_empty and not out:
        raise ResearchCampaignError(f"{name} must not be empty")
    if unique and len(out) != len(set(out)):
        raise ResearchCampaignError(f"{name} contains duplicates")
    return out


def _ordered_ids(value: Any, name: str, *, allow_empty: bool = False) -> list[str]:
    out = _ordered_text(value, name, allow_empty=allow_empty, unique=True)
    for index, item in enumerate(out):
        _hex_id(item, f"{name}[{index}]")
    return out


def _normalize_spec(value: Any) -> dict[str, Any]:
    raw = _exact_fields(value, SPEC_FIELDS, "campaign specification")
    if raw["schema"] != INPUT_SCHEMA:
        raise ResearchCampaignError(f"unsupported campaign input schema {raw['schema']!r}")
    campaign = _campaign_id(raw["campaign_id"])
    key = _exact_fields(raw["key"], KEY_FIELDS, "campaign key")
    normalized_key = {
        field: _text(key[field], f"key.{field}")
        for field in ("pair", "timeframe", "target", "side")
    }
    if normalized_key["target"] == "direction":
        if normalized_key["side"] not in {"UP", "DOWN"}:
            raise ResearchCampaignError("direction target requires side=UP|DOWN")
    elif normalized_key["target"] == "magnitude":
        if normalized_key["side"] != "NA":
            raise ResearchCampaignError("magnitude target requires side=NA")
    else:
        raise ResearchCampaignError("target must be direction or magnitude")
    if normalized_key["side"] == "COMBINED":
        raise ResearchCampaignError("COMBINED is not a campaign result key")
    data_use = raw["data_use"]
    if type(data_use) is not str or data_use not in {"synthetic", "retrospective"}:
        raise ResearchCampaignError("data_use must be synthetic or retrospective")

    if type(raw["ordered_arms"]) is not list or not raw["ordered_arms"]:
        raise ResearchCampaignError("ordered_arms must be a non-empty JSON list")
    arms: list[dict[str, Any]] = []
    arm_ids: set[str] = set()
    result_paths: set[str] = set()
    for index, item in enumerate(raw["ordered_arms"]):
        arm = _exact_fields(item, ARM_FIELDS, f"ordered_arms[{index}]")
        arm_id = _text(arm["arm_id"], f"ordered_arms[{index}].arm_id")
        if arm_id in arm_ids:
            raise ResearchCampaignError("ordered_arms contains duplicate arm_id")
        arm_ids.add(arm_id)
        result_path = _canonical_path(
            arm["native_result_path"], f"ordered_arms[{index}].native_result_path"
        )
        result_parts = PurePosixPath(result_path).parts
        if (
            len(result_parts) < 3
            or result_parts[:2] != ("results", "json")
            or not result_path.endswith("_result.json")
        ):
            raise ResearchCampaignError(
                "native_result_path must be beneath results/json and end in _result.json"
            )
        if result_path in result_paths:
            raise ResearchCampaignError("ordered_arms contains duplicate native_result_path")
        result_paths.add(result_path)
        input_ids = _ordered_ids(
            arm["input_evidence_ids"], f"ordered_arms[{index}].input_evidence_ids"
        )
        basis_ids = _ordered_ids(
            arm["data_use_basis_evidence_ids"],
            f"ordered_arms[{index}].data_use_basis_evidence_ids",
        )
        input_positions = {item_id: position for position, item_id in enumerate(input_ids)}
        try:
            positions = [input_positions[item_id] for item_id in basis_ids]
        except KeyError as exc:
            raise ResearchCampaignError(
                "data_use_basis_evidence_ids must be a subset of input_evidence_ids"
            ) from exc
        if positions != sorted(positions):
            raise ResearchCampaignError(
                "data_use_basis_evidence_ids must preserve input_evidence_ids order"
            )
        arms.append(
            {
                "arm_id": arm_id,
                "role": _text(arm["role"], f"ordered_arms[{index}].role"),
                "native_result_path": result_path,
                "input_evidence_ids": input_ids,
                "data_use_basis_evidence_ids": basis_ids,
            }
        )

    adapter = _exact_fields(raw["adapter"], ADAPTER_FIELDS, "adapter")
    adapter_path = _canonical_path(adapter["path"], "adapter.path")
    adapter_parts = PurePosixPath(adapter_path).parts
    if (
        len(adapter_parts) != 2
        or adapter_parts[0] != "scripts"
        or not adapter_parts[1].endswith(".py")
    ):
        raise ResearchCampaignError("adapter.path must be a direct scripts/*.py file")
    normalized_adapter = {
        "path": adapter_path,
        "run_function": adapter["run_function"],
        "preflight_function": adapter["preflight_function"],
        "validate_function": adapter["validate_function"],
        "evaluate_function": adapter["evaluate_function"],
        "native_result_contract": _text(
            adapter["native_result_contract"], "adapter.native_result_contract"
        ),
    }
    expected_functions = {
        "run_function": "run_arm",
        "preflight_function": "preflight_arm",
        "validate_function": "validate_arm_result",
        "evaluate_function": "evaluate_family",
    }
    for field, expected in expected_functions.items():
        if normalized_adapter[field] != expected:
            raise ResearchCampaignError(f"adapter.{field} must equal {expected}")

    terminal_rule = _exact_fields(
        raw["terminal_rule"], TERMINAL_RULE_FIELDS, "terminal_rule"
    )
    allowed = _ordered_text(
        terminal_rule["allowed_decisions"], "terminal_rule.allowed_decisions", unique=True
    )
    expected_allowed = ["no_candidate", "inactive_candidate", "capability_deferred"]
    if allowed != expected_allowed or terminal_rule["activation"] is not False:
        raise ResearchCampaignError(
            "terminal_rule must contain the exact inactive v1 decision order"
        )
    return _json_copy(
        {
            "schema": INPUT_SCHEMA,
            "campaign_id": campaign,
            "key": normalized_key,
            "question": _text(raw["question"], "question"),
            "hypothesis": _text(raw["hypothesis"], "hypothesis"),
            "falsifier": _text(raw["falsifier"], "falsifier"),
            "data_use": data_use,
            "ordered_arms": arms,
            "adapter": normalized_adapter,
            "stopping_rule": _text(raw["stopping_rule"], "stopping_rule"),
            "terminal_rule": {
                "allowed_decisions": expected_allowed,
                "activation": False,
            },
        }
    )


def _load_spec(path: str | os.PathLike[str]) -> dict[str, Any]:
    candidate = Path(path)
    if candidate.is_symlink():
        raise ResearchCampaignError("campaign specification must not be a symlink")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(candidate, flags)
    except OSError as exc:
        raise ResearchCampaignError(f"cannot open campaign specification: {exc}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ResearchCampaignError("campaign specification must be a regular file")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1 << 20)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if (
        (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino)
        or before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
        or before.st_ctime_ns != after.st_ctime_ns
    ):
        raise ResearchCampaignError("campaign specification changed while being read")
    return _normalize_spec(evidence.decode_canonical_json(b"".join(chunks)))


def _validated_root(repo_root: str | os.PathLike[str]) -> Path:
    candidate = Path(repo_root)
    if candidate.is_symlink():
        raise ResearchCampaignError("repository root must not be a symlink")
    try:
        root = candidate.resolve(strict=True)
    except OSError as exc:
        raise ResearchCampaignError("repository root does not exist") from exc
    if not root.is_dir():
        raise ResearchCampaignError("repository root must be a directory")
    for relative in ("scripts", "results/json", "evidence/objects", "logs"):
        _require_real_directory(root, root / relative, relative)
    return root


def _require_real_directory(root: Path, directory: Path, name: str) -> Path:
    try:
        relative = directory.relative_to(root)
    except ValueError as exc:
        raise ResearchCampaignError(f"{name} must remain beneath repository root") from exc
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink() or not current.is_dir():
            raise ResearchCampaignError(f"{name} must contain only real directories")
    return current


def _artifact(root: Path, path: str) -> evidence.ArtifactRef:
    try:
        return evidence.ArtifactRef.capture(path, repo_root=root)
    except evidence.EvidenceError as exc:
        raise ResearchCampaignError(str(exc)) from exc


def _dependency_union(spec: Mapping[str, Any]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for arm in spec["ordered_arms"]:
        for object_id in arm["input_evidence_ids"]:
            if object_id not in seen:
                seen.add(object_id)
                out.append(object_id)
    return out


def _build_spec(root: Path, spec: dict[str, Any]) -> evidence.EvidenceEnvelope:
    runner = _artifact(root, RUNNER_PATH)
    adapter = _artifact(root, spec["adapter"]["path"])
    runner.isolated_identity(repo_root=root)
    adapter.isolated_identity(repo_root=root)
    return evidence.EvidenceEnvelope.create(
        kind=SPEC_KIND,
        payload={
            "schema": SPEC_KIND,
            "campaign_id": spec["campaign_id"],
            "campaign": spec,
        },
        artifacts=[runner, adapter],
        dependencies=_dependency_union(spec),
    )


def _normalize_preflight(
    value: Any, spec: Mapping[str, Any], arm: Mapping[str, Any]
) -> dict[str, Any]:
    raw = _exact_fields(value, PREFLIGHT_FIELDS, "adapter preflight")
    reasons = _ordered_text(raw["reasons"], "preflight.reasons", allow_empty=True)
    if type(raw["admissible"]) is not bool or type(raw["protected_access"]) is not bool:
        raise ResearchCampaignError("preflight booleans must be exact booleans")
    expected = {
        "campaign_id": spec["campaign_id"],
        "arm_id": arm["arm_id"],
        "data_use": spec["data_use"],
        "data_use_basis_evidence_ids": list(arm["data_use_basis_evidence_ids"]),
        "protected_access": False,
    }
    for field, expected_value in expected.items():
        if raw[field] != expected_value:
            raise ResearchCampaignError(f"preflight {field} differs from sealed state")
    if raw["admissible"] and reasons:
        raise ResearchCampaignError("admissible preflight reasons must be empty")
    if not raw["admissible"] and not reasons:
        raise ResearchCampaignError("inadmissible preflight requires reasons")
    normalized = dict(expected, admissible=raw["admissible"], reasons=reasons)
    evidence.canonical_json_bytes(normalized)
    return _json_copy(normalized)


def _normalize_receipt(
    value: Any, spec: Mapping[str, Any], arm: Mapping[str, Any]
) -> dict[str, Any]:
    raw = _exact_fields(value, RECEIPT_FIELDS, "adapter receipt")
    if raw["schema"] != RECEIPT_SCHEMA:
        raise ResearchCampaignError("adapter receipt schema differs")
    if type(raw["status"]) is not str or raw["status"] not in {
        "completed",
        "not_computable",
    }:
        raise ResearchCampaignError("adapter receipt status differs")
    reasons = _ordered_text(raw["reasons"], "receipt.reasons", allow_empty=True)
    if raw["status"] == "completed" and reasons:
        raise ResearchCampaignError("completed receipt reasons must be empty")
    if raw["status"] == "not_computable" and not reasons:
        raise ResearchCampaignError("not_computable receipt requires reasons")
    expected = {
        "campaign_id": spec["campaign_id"],
        "arm_id": arm["arm_id"],
        "native_result_contract": spec["adapter"]["native_result_contract"],
        "native_result_path": arm["native_result_path"],
        "input_evidence_ids": list(arm["input_evidence_ids"]),
        "data_use": spec["data_use"],
        "data_use_basis_evidence_ids": list(arm["data_use_basis_evidence_ids"]),
        "protected_access": False,
    }
    for field, expected_value in expected.items():
        if raw[field] != expected_value:
            raise ResearchCampaignError(f"receipt {field} differs from sealed state")
    normalized = dict(
        {"schema": RECEIPT_SCHEMA},
        **{field: expected[field] for field in expected},
        status=raw["status"],
        reasons=reasons,
    )
    evidence.canonical_json_bytes(normalized)
    return _json_copy(normalized)


def _normalize_decision(value: Any, spec: Mapping[str, Any]) -> dict[str, Any]:
    raw = _exact_fields(value, DECISION_FIELDS, "family decision")
    if raw["schema"] != DECISION_SCHEMA or raw["campaign_id"] != spec["campaign_id"]:
        raise ResearchCampaignError("family decision identity differs")
    if type(raw["verdicts"]) is not list:
        raise ResearchCampaignError("family decision verdicts must be a JSON list")
    arms = spec["ordered_arms"]
    if len(raw["verdicts"]) != len(arms):
        raise ResearchCampaignError("family decision must contain one verdict per arm")
    verdicts: list[dict[str, Any]] = []
    candidates: list[str] = []
    for index, (value_verdict, arm) in enumerate(zip(raw["verdicts"], arms)):
        verdict = _exact_fields(value_verdict, VERDICT_FIELDS, f"verdicts[{index}]")
        if verdict["arm_id"] != arm["arm_id"]:
            raise ResearchCampaignError("family verdict order or arm differs")
        if type(verdict["status"]) is not str or verdict["status"] not in {
            "killed",
            "candidate",
            "not_computable",
        }:
            raise ResearchCampaignError("family verdict status differs")
        reasons = _ordered_text(
            verdict["reasons"], f"verdicts[{index}].reasons", allow_empty=True
        )
        if verdict["status"] == "candidate":
            if reasons:
                raise ResearchCampaignError("candidate verdict reasons must be empty")
            candidates.append(arm["arm_id"])
        elif not reasons:
            raise ResearchCampaignError("non-candidate verdict requires reasons")
        verdicts.append(
            {"arm_id": arm["arm_id"], "status": verdict["status"], "reasons": reasons}
        )
    ordered_candidates = _ordered_text(
        raw["ordered_candidates"], "ordered_candidates", allow_empty=True, unique=True
    )
    expected_order = [item for item in candidates if item in set(ordered_candidates)]
    if ordered_candidates != expected_order:
        raise ResearchCampaignError("ordered_candidates is not an ordered candidate subset")
    decision = raw["decision"]
    if decision not in spec["terminal_rule"]["allowed_decisions"]:
        raise ResearchCampaignError("family decision is not allowed by terminal_rule")
    reasons = _ordered_text(raw["reasons"], "family decision reasons", allow_empty=True)
    if decision == "no_candidate" and ordered_candidates:
        raise ResearchCampaignError("no_candidate requires an empty candidate list")
    if decision == "inactive_candidate" and not ordered_candidates:
        raise ResearchCampaignError("inactive_candidate requires candidates")
    if decision == "capability_deferred":
        if not reasons or ordered_candidates:
            raise ResearchCampaignError(
                "capability_deferred requires reasons and no candidate claim"
            )
    normalized = {
        "schema": DECISION_SCHEMA,
        "campaign_id": spec["campaign_id"],
        "verdicts": verdicts,
        "ordered_candidates": ordered_candidates,
        "decision": decision,
        "reasons": reasons,
    }
    evidence.canonical_json_bytes(normalized)
    return _json_copy(normalized)


def _build_attempt(
    spec_envelope: evidence.EvidenceEnvelope,
    arm_index: int,
    preflight: dict[str, Any],
) -> evidence.EvidenceEnvelope:
    spec = spec_envelope.payload["campaign"]
    arm = spec["ordered_arms"][arm_index]
    return evidence.EvidenceEnvelope.create(
        kind=ATTEMPT_KIND,
        payload={
            "schema": ATTEMPT_KIND,
            "campaign_id": spec["campaign_id"],
            "spec_id": spec_envelope.object_id,
            "arm_index": arm_index,
            "arm_id": arm["arm_id"],
            "final_path": arm["native_result_path"],
            "preflight": preflight,
            "authorization_state": AUTHORIZATION_STATE,
        },
        dependencies=[spec_envelope.object_id],
    )


def _build_result(
    spec_envelope: evidence.EvidenceEnvelope,
    attempt: evidence.EvidenceEnvelope,
    arm_index: int,
    receipt: dict[str, Any],
    artifact: evidence.ArtifactRef,
) -> evidence.EvidenceEnvelope:
    spec = spec_envelope.payload["campaign"]
    arm = spec["ordered_arms"][arm_index]
    return evidence.EvidenceEnvelope.create(
        kind=RESULT_KIND,
        payload={
            "schema": RESULT_KIND,
            "campaign_id": spec["campaign_id"],
            "spec_id": spec_envelope.object_id,
            "attempt_id": attempt.object_id,
            "arm_index": arm_index,
            "arm_id": arm["arm_id"],
            "receipt": receipt,
        },
        artifacts=[artifact],
        dependencies=[attempt.object_id],
    )


def _evaluator_identity(spec_envelope: evidence.EvidenceEnvelope) -> dict[str, Any]:
    spec = spec_envelope.payload["campaign"]
    return {
        "adapter": spec_envelope.artifacts[1].as_dict(),
        "function": spec["adapter"]["evaluate_function"],
    }


def _build_terminal(
    spec_envelope: evidence.EvidenceEnvelope,
    results: Sequence[evidence.EvidenceEnvelope],
    decision: dict[str, Any],
) -> evidence.EvidenceEnvelope:
    spec = spec_envelope.payload["campaign"]
    return evidence.EvidenceEnvelope.create(
        kind=TERMINAL_KIND,
        payload={
            "schema": TERMINAL_KIND,
            "campaign_id": spec["campaign_id"],
            "spec_id": spec_envelope.object_id,
            "family_decision": decision,
            "evaluator": _evaluator_identity(spec_envelope),
            "ordered_candidates": decision["ordered_candidates"],
            "data_use": spec["data_use"],
            "activation": False,
            "claim_limit": CLAIM_LIMIT,
        },
        dependencies=[item.object_id for item in results],
    )


def _domain_campaign(envelope: evidence.EvidenceEnvelope) -> str:
    fields = {
        SPEC_KIND: SPEC_PAYLOAD_FIELDS,
        ATTEMPT_KIND: ATTEMPT_PAYLOAD_FIELDS,
        RESULT_KIND: RESULT_PAYLOAD_FIELDS,
        TERMINAL_KIND: TERMINAL_PAYLOAD_FIELDS,
    }[envelope.kind]
    payload = _exact_fields(envelope.payload, fields, f"{envelope.kind} payload")
    if payload["schema"] != envelope.kind:
        raise ResearchCampaignError(f"{envelope.kind} payload schema differs")
    return _campaign_id(payload["campaign_id"])


def _groups(root: Path, campaign_id: str) -> dict[str, list[evidence.EvidenceEnvelope]]:
    groups = {kind: [] for kind in DOMAIN_KINDS}
    for envelope in evidence.inspect_store_metadata(repo_root=root):
        if envelope.kind not in groups:
            continue
        if _domain_campaign(envelope) == campaign_id:
            groups[envelope.kind].append(envelope)
    return groups


def _freeze(value: Any) -> Any:
    if type(value) is dict:
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if type(value) is list:
        return tuple(_freeze(item) for item in value)
    return value


def _adapter_call(function: Any, name: str, **kwargs: Any) -> Any:
    try:
        return function(**kwargs)
    except ResearchCampaignError:
        raise
    except Exception as exc:
        raise ResearchCampaignError(f"adapter {name} failed: {exc}") from exc


def _load_adapter(
    root: Path, spec_envelope: evidence.EvidenceEnvelope
) -> ModuleType:
    verified = evidence.verify_object(spec_envelope.object_id, repo_root=root)
    if verified.as_dict() != spec_envelope.as_dict():
        raise ResearchCampaignError("sealed specification changed before adapter import")
    spec = spec_envelope.payload["campaign"]
    reference = spec_envelope.artifacts[1]
    if reference.path != spec["adapter"]["path"]:
        raise ResearchCampaignError("sealed adapter artifact path differs")
    source = reference.read_verified(repo_root=root)
    module_name = f"research_campaign_adapter_{reference.sha256}"
    module = ModuleType(module_name)
    module.__file__ = str(root / reference.path)
    module.__package__ = ""
    scripts = str(root / "scripts")
    inserted = scripts not in sys.path
    previous_module = sys.modules.get(module_name)
    if inserted:
        sys.path.insert(0, scripts)
    sys.modules[module_name] = module
    try:
        exec(compile(source, module.__file__, "exec"), module.__dict__)
    except Exception as exc:
        raise ResearchCampaignError(f"adapter import failed: {exc}") from exc
    finally:
        if previous_module is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous_module
        if inserted:
            sys.path.remove(scripts)
    for field in (
        "preflight_function",
        "run_function",
        "validate_function",
        "evaluate_function",
    ):
        name = spec["adapter"][field]
        if not callable(getattr(module, name, None)):
            raise ResearchCampaignError(f"adapter callable {name} is absent")
    reference.verify(repo_root=root)
    return module


def _input_envelopes(
    root: Path, arm: Mapping[str, Any]
) -> tuple[evidence.EvidenceEnvelope, ...]:
    return tuple(
        evidence.verify_object(object_id, repo_root=root)
        for object_id in arm["input_evidence_ids"]
    )


def _call_preflight(
    module: ModuleType,
    spec: dict[str, Any],
    arm: dict[str, Any],
    inputs: Sequence[evidence.EvidenceEnvelope],
) -> dict[str, Any]:
    value = _adapter_call(
        module.preflight_arm,
        "preflight_arm",
        campaign=_freeze(_json_copy(spec)),
        arm=_freeze(_json_copy(arm)),
        input_evidence=tuple(_freeze(item.as_dict()) for item in inputs),
    )
    normalized = _normalize_preflight(value, spec, arm)
    if not normalized["admissible"]:
        raise ResearchCampaignError(
            "BLOCKED: adapter preflight rejected the arm: " + "; ".join(normalized["reasons"])
        )
    return normalized


def _call_run(
    module: ModuleType,
    spec: dict[str, Any],
    arm: dict[str, Any],
    temporary: Path,
) -> dict[str, Any]:
    value = _adapter_call(
        module.run_arm,
        "run_arm",
        campaign=_freeze(_json_copy(spec)),
        arm=_freeze(_json_copy(arm)),
        temporary_output_path=temporary,
    )
    return _normalize_receipt(value, spec, arm)


def _call_validate(
    module: ModuleType,
    spec: dict[str, Any],
    arm: dict[str, Any],
    installed: Path,
) -> dict[str, Any]:
    value = _adapter_call(
        module.validate_arm_result,
        "validate_arm_result",
        campaign=_freeze(_json_copy(spec)),
        arm=_freeze(_json_copy(arm)),
        native_result_path=installed,
    )
    return _normalize_receipt(value, spec, arm)


def _call_evaluator(
    module: ModuleType,
    spec: dict[str, Any],
    results: Sequence[evidence.EvidenceEnvelope],
) -> dict[str, Any]:
    receipts = tuple(_freeze(item.payload["receipt"]) for item in results)
    references = tuple(_freeze(item.artifacts[0].as_dict()) for item in results)
    value = _adapter_call(
        module.evaluate_family,
        "evaluate_family",
        campaign=_freeze(_json_copy(spec)),
        receipts=receipts,
        native_results=references,
    )
    return _normalize_decision(value, spec)


def _real_parent(root: Path, relative: str, name: str) -> Path:
    parts = PurePosixPath(relative).parts
    return _require_real_directory(root, root.joinpath(*parts[:-1]), name)


def _result_parent(root: Path, relative: str, *, create: bool) -> Path | None:
    parts = PurePosixPath(relative).parts[:-1]
    current = root
    for part in parts:
        current = current / part
        if current.exists() or current.is_symlink():
            if current.is_symlink() or not current.is_dir():
                raise ResearchCampaignError(
                    "native result parent must contain only real directories"
                )
            continue
        if not create:
            return None
        try:
            current.mkdir(mode=0o755)
        except FileExistsError:
            if current.is_symlink() or not current.is_dir():
                raise ResearchCampaignError(
                    "native result parent appeared as a non-directory"
                )
        _fsync_directory(current.parent)
    return current


def _path_exists(root: Path, relative: str, name: str) -> bool:
    parent = _result_parent(root, relative, create=False)
    if parent is None:
        return False
    path = parent / PurePosixPath(relative).name
    if path.is_symlink():
        raise ResearchCampaignError(f"{name} must not be a symlink")
    if path.exists() and not path.is_file():
        raise ResearchCampaignError(f"{name} must be a regular file")
    return path.exists()


def _temporary_path(root: Path, campaign_id: str, arm: Mapping[str, Any]) -> Path:
    final = arm["native_result_path"]
    parent = root.joinpath(*PurePosixPath(final).parts[:-1])
    digest = hashlib.sha256(
        (campaign_id + "\x00" + arm["arm_id"]).encode("utf-8")
    ).hexdigest()
    return parent / f".research-campaign-{digest}.tmp"


def _remove_temporary(path: Path) -> None:
    if not path.exists() and not path.is_symlink():
        return
    if path.is_symlink() or not path.is_file():
        raise ResearchCampaignError("BLOCKED: campaign temporary output is not regular")
    path.unlink()
    _fsync_directory(path.parent)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_regular(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ResearchCampaignError(f"cannot open native result for fsync: {exc}") from exc
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1:
            raise ResearchCampaignError(
                "native result must remain a singly linked regular file during fsync"
            )
        os.fsync(descriptor)
        named = path.stat(follow_symlinks=False)
        if (opened.st_dev, opened.st_ino) != (named.st_dev, named.st_ino):
            raise ResearchCampaignError("native result path changed during fsync")
    finally:
        os.close(descriptor)


def _install_native_result(
    root: Path, temporary: Path, final_relative: str
) -> evidence.ArtifactRef:
    parent = _real_parent(root, final_relative, "native result parent")
    final = parent / PurePosixPath(final_relative).name
    if final.exists() or final.is_symlink():
        raise ResearchCampaignError("BLOCKED: native result destination already exists")
    if temporary.parent != parent or temporary.is_symlink() or not temporary.is_file():
        raise ResearchCampaignError("adapter did not produce the exact regular temporary output")
    temporary_stat = temporary.stat(follow_symlinks=False)
    if not stat.S_ISREG(temporary_stat.st_mode) or temporary_stat.st_nlink != 1:
        raise ResearchCampaignError(
            "adapter temporary output must be a singly linked regular file"
        )
    temporary.chmod(0o444)
    _fsync_regular(temporary)
    temporary_relative = temporary.relative_to(root).as_posix()
    captured = _artifact(root, temporary_relative)
    captured.isolated_identity(repo_root=root)
    try:
        os.link(temporary, final, follow_symlinks=False)
    except FileExistsError as exc:
        raise ResearchCampaignError("BLOCKED: native result appeared concurrently") from exc
    except OSError as exc:
        raise ResearchCampaignError(f"cannot atomically install native result: {exc}") from exc
    temporary.unlink()
    _fsync_directory(parent)
    installed = _artifact(root, final_relative)
    installed.isolated_identity(repo_root=root)
    if installed.bytes != captured.bytes or installed.sha256 != captured.sha256:
        raise ResearchCampaignError("BLOCKED: native result changed during installation")
    return installed


def _validate_attempt(
    root: Path,
    envelope: evidence.EvidenceEnvelope,
    spec_envelope: evidence.EvidenceEnvelope,
    arm_index: int,
) -> evidence.EvidenceEnvelope:
    verified = evidence.verify_object(envelope.object_id, repo_root=root)
    spec = spec_envelope.payload["campaign"]
    arm = spec["ordered_arms"][arm_index]
    preflight = _normalize_preflight(verified.payload["preflight"], spec, arm)
    if not preflight["admissible"]:
        raise ResearchCampaignError("stored attempt has inadmissible preflight")
    expected = _build_attempt(spec_envelope, arm_index, preflight)
    if verified.as_dict() != expected.as_dict():
        raise ResearchCampaignError("BLOCKED: arm attempt differs from sealed state")
    return verified


def _validate_result(
    root: Path,
    envelope: evidence.EvidenceEnvelope,
    spec_envelope: evidence.EvidenceEnvelope,
    attempt: evidence.EvidenceEnvelope,
    arm_index: int,
) -> evidence.EvidenceEnvelope:
    verified = evidence.verify_object(envelope.object_id, repo_root=root)
    spec = spec_envelope.payload["campaign"]
    arm = spec["ordered_arms"][arm_index]
    receipt = _normalize_receipt(verified.payload["receipt"], spec, arm)
    if len(verified.artifacts) != 1 or verified.artifacts[0].path != arm["native_result_path"]:
        raise ResearchCampaignError("arm result artifact binding differs")
    expected = _build_result(
        spec_envelope, attempt, arm_index, receipt, verified.artifacts[0]
    )
    if verified.as_dict() != expected.as_dict():
        raise ResearchCampaignError("BLOCKED: arm result differs from sealed state")
    return verified


def _indexed_state(
    root: Path,
    spec_envelope: evidence.EvidenceEnvelope,
) -> tuple[
    dict[int, evidence.EvidenceEnvelope],
    dict[int, evidence.EvidenceEnvelope],
    evidence.EvidenceEnvelope | None,
]:
    spec = spec_envelope.payload["campaign"]
    groups = _groups(root, spec["campaign_id"])
    if len(groups[SPEC_KIND]) != 1 or groups[SPEC_KIND][0].as_dict() != spec_envelope.as_dict():
        raise ResearchCampaignError("BLOCKED: hidden or conflicting campaign specification")
    arm_by_id = {arm["arm_id"]: index for index, arm in enumerate(spec["ordered_arms"])}
    attempts: dict[int, evidence.EvidenceEnvelope] = {}
    results: dict[int, evidence.EvidenceEnvelope] = {}
    for kind, destination in ((ATTEMPT_KIND, attempts), (RESULT_KIND, results)):
        for envelope in groups[kind]:
            arm_id = envelope.payload["arm_id"]
            arm_index = envelope.payload["arm_index"]
            if not isinstance(arm_id, str) or type(arm_index) is not int:
                raise ResearchCampaignError(f"BLOCKED: malformed {kind} arm identity")
            if arm_id not in arm_by_id:
                raise ResearchCampaignError(f"BLOCKED: hidden {kind} for unknown arm")
            index = arm_by_id[arm_id]
            if arm_index != index or index in destination:
                raise ResearchCampaignError(f"BLOCKED: duplicate or reordered {kind}")
            destination[index] = envelope
    for index in sorted(attempts):
        attempts[index] = _validate_attempt(root, attempts[index], spec_envelope, index)
    for index in sorted(results):
        if index not in attempts:
            raise ResearchCampaignError("BLOCKED: arm result has no exact attempt")
        results[index] = _validate_result(
            root, results[index], spec_envelope, attempts[index], index
        )
        temporary = _temporary_path(root, spec["campaign_id"], spec["ordered_arms"][index])
        if temporary.exists() or temporary.is_symlink():
            raise ResearchCampaignError(
                "BLOCKED: committed arm result has a hidden temporary output"
            )
    completed = set(range(len(results)))
    if set(results) != completed:
        raise ResearchCampaignError(
            "BLOCKED: arm results do not form the sealed ordered prefix"
        )
    allowed_attempts = completed.copy()
    if len(results) < len(spec["ordered_arms"]):
        allowed_attempts.add(len(results))
    if frozenset(attempts) not in {
        frozenset(completed),
        frozenset(allowed_attempts),
    }:
        raise ResearchCampaignError(
            "BLOCKED: arm attempts advance beyond the next sealed arm"
        )
    terminals = groups[TERMINAL_KIND]
    if len(terminals) > 1:
        raise ResearchCampaignError("BLOCKED: hidden or conflicting terminal state")
    return attempts, results, terminals[0] if terminals else None


def _verify_or_publish_terminal(
    root: Path,
    spec_envelope: evidence.EvidenceEnvelope,
    module: ModuleType,
    results: dict[int, evidence.EvidenceEnvelope],
    stored_terminal: evidence.EvidenceEnvelope | None,
    *,
    publish: bool,
) -> evidence.EvidenceEnvelope:
    spec = spec_envelope.payload["campaign"]
    if set(results) != set(range(len(spec["ordered_arms"]))):
        raise ResearchCampaignError("campaign terminal is unavailable: arm family is incomplete")
    ordered = [results[index] for index in range(len(spec["ordered_arms"]))]
    decision = _call_evaluator(module, spec, ordered)
    expected = _build_terminal(spec_envelope, ordered, decision)
    if stored_terminal is not None:
        verified = evidence.verify_object(stored_terminal.object_id, repo_root=root)
        if verified.as_dict() != expected.as_dict():
            raise ResearchCampaignError("BLOCKED: terminal decision is not reproducible")
        return verified
    if not publish:
        raise ResearchCampaignError("campaign terminal is absent")
    evidence.publish(expected, repo_root=root)
    return expected


@contextmanager
def _campaign_lock(root: Path) -> Iterator[None]:
    logs = _require_real_directory(root, root / "logs", "logs")
    directory = logs / "research_campaign"
    if not directory.exists() and not directory.is_symlink():
        try:
            directory.mkdir(mode=0o700)
            _fsync_directory(logs)
        except FileExistsError:
            pass
    _require_real_directory(root, directory, "research campaign lock parent")
    path = directory / "run_campaign.lock"
    if path.is_symlink():
        raise ResearchCampaignError("campaign lock must not be a symlink")
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as exc:
        raise ResearchCampaignError(f"cannot open campaign lock: {exc}") from exc
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or stat.S_IMODE(opened.st_mode) != 0o600
        ):
            raise ResearchCampaignError("campaign lock must be private and singly linked")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if exc.errno in {errno.EACCES, errno.EAGAIN}:
                raise ResearchCampaignError("campaign runner is already active") from exc
            raise ResearchCampaignError(f"cannot acquire campaign lock: {exc}") from exc
        named = path.stat(follow_symlinks=False)
        if (opened.st_dev, opened.st_ino) != (named.st_dev, named.st_ino):
            raise ResearchCampaignError("campaign lock path changed during acquisition")
        _fsync_directory(directory)
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _run_locked(root: Path, spec: dict[str, Any]) -> evidence.EvidenceEnvelope:
    evidence.verify_store(repo_root=root)
    expected_spec = _build_spec(root, spec)
    groups = _groups(root, spec["campaign_id"])
    if groups[SPEC_KIND]:
        if len(groups[SPEC_KIND]) != 1 or groups[SPEC_KIND][0].as_dict() != expected_spec.as_dict():
            raise ResearchCampaignError("BLOCKED: campaign ID has a divergent specification")
    else:
        if any(groups[kind] for kind in DOMAIN_KINDS if kind != SPEC_KIND):
            raise ResearchCampaignError(
                "BLOCKED: campaign descendants exist without a specification"
            )
        evidence.publish(expected_spec, repo_root=root)
    spec_envelope = evidence.verify_object(expected_spec.object_id, repo_root=root)
    module = _load_adapter(root, spec_envelope)
    attempts, results, terminal = _indexed_state(root, spec_envelope)
    if terminal is not None:
        return _verify_or_publish_terminal(
            root, spec_envelope, module, results, terminal, publish=False
        )

    for index, arm in enumerate(spec["ordered_arms"]):
        if index in results:
            continue
        final_relative = arm["native_result_path"]
        final = root / final_relative
        temporary = _temporary_path(root, spec["campaign_id"], arm)
        attempt = attempts.get(index)
        if attempt is None:
            if _path_exists(root, final_relative, "native result"):
                raise ResearchCampaignError(
                    "BLOCKED: final native result exists without the exact prior attempt"
                )
            if temporary.exists() or temporary.is_symlink():
                raise ResearchCampaignError(
                    "BLOCKED: campaign temporary output exists without an exact attempt"
                )
            inputs = _input_envelopes(root, arm)
            preflight = _call_preflight(module, spec, arm, inputs)
            evidence.verify_object(spec_envelope.object_id, repo_root=root)
            _input_envelopes(root, arm)
            if _path_exists(root, final_relative, "native result"):
                raise ResearchCampaignError(
                    "BLOCKED: final native result appeared before attempt publication"
                )
            if temporary.exists() or temporary.is_symlink():
                raise ResearchCampaignError(
                    "BLOCKED: temporary output appeared during preflight"
                )
            attempt = _build_attempt(spec_envelope, index, preflight)
            evidence.publish(attempt, repo_root=root)
            attempts[index] = attempt
        else:
            evidence.verify_object(spec_envelope.object_id, repo_root=root)
            _input_envelopes(root, arm)

        _result_parent(root, final_relative, create=True)
        if _path_exists(root, final_relative, "native result"):
            if temporary.exists() or temporary.is_symlink():
                if temporary.is_symlink() or not temporary.is_file():
                    raise ResearchCampaignError("BLOCKED: invalid campaign temporary output")
                try:
                    same = temporary.samefile(final)
                except OSError as exc:
                    raise ResearchCampaignError("cannot compare interrupted native output") from exc
                if not same:
                    raise ResearchCampaignError(
                        "BLOCKED: temporary and final native results conflict"
                    )
                _remove_temporary(temporary)
            before = _artifact(root, final_relative)
            before_identity = before.isolated_identity(repo_root=root)
            receipt = _call_validate(module, spec, arm, final)
            after = _artifact(root, final_relative)
            after_identity = after.isolated_identity(repo_root=root)
            if before != after or before_identity != after_identity:
                raise ResearchCampaignError(
                    "BLOCKED: adopted native result changed during validation"
                )
            artifact = after
        else:
            _remove_temporary(temporary)
            receipt = _call_run(module, spec, arm, temporary)
            artifact = _install_native_result(root, temporary, final_relative)
        result = _build_result(spec_envelope, attempt, index, receipt, artifact)
        evidence.publish(result, repo_root=root)
        results[index] = result

    attempts, results, terminal = _indexed_state(root, spec_envelope)
    return _verify_or_publish_terminal(
        root, spec_envelope, module, results, terminal, publish=True
    )


def run_campaign(
    spec_path: str | os.PathLike[str],
    *,
    repo_root: str | os.PathLike[str] = REPO_ROOT,
) -> evidence.EvidenceEnvelope:
    """Run or deterministically resume one sealed unprotected campaign."""

    root = _validated_root(repo_root)
    with _campaign_lock(root):
        spec = _load_spec(spec_path)
        return _run_locked(root, spec)


def verify_campaign(
    campaign_id: str,
    *,
    repo_root: str | os.PathLike[str] = REPO_ROOT,
) -> evidence.EvidenceEnvelope:
    """Verify a completed campaign and recompute its pure terminal lock-free."""

    root = _validated_root(repo_root)
    identifier = _campaign_id(campaign_id)
    groups = _groups(root, identifier)
    if len(groups[SPEC_KIND]) != 1:
        raise ResearchCampaignError("campaign must have exactly one sealed specification")
    stored_spec = evidence.verify_object(groups[SPEC_KIND][0].object_id, repo_root=root)
    spec = _normalize_spec(stored_spec.payload["campaign"])
    expected = _build_spec(root, spec)
    if stored_spec.as_dict() != expected.as_dict():
        raise ResearchCampaignError("BLOCKED: sealed campaign specification is not reproducible")
    module = _load_adapter(root, stored_spec)
    _, results, terminal = _indexed_state(root, stored_spec)
    return _verify_or_publish_terminal(
        root, stored_spec, module, results, terminal, publish=False
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="run or resume a sealed campaign")
    run.add_argument("--spec", required=True, type=Path)
    verify = commands.add_parser("verify", help="verify a completed campaign")
    verify.add_argument("--campaign-id", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        terminal = (
            run_campaign(args.spec)
            if args.command == "run"
            else verify_campaign(args.campaign_id)
        )
    except (ResearchCampaignError, evidence.EvidenceError) as exc:
        print(f"research campaign failed: {exc}", file=sys.stderr)
        return 2
    print(
        f"campaign terminal {terminal.object_id} "
        f"decision={terminal.payload['family_decision']['decision']} activation=false"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
