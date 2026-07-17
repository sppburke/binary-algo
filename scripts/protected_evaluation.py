#!/usr/bin/env python3
"""Minimal seal-before-read adapter for one protected evaluation vertical.

This module owns domain validation and access ordering only.  It delegates the
terminal empty-shadow decision to the existing issue-#9 statistics helper and
never fits, scores, activates, or loads a model.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Sequence

import evidence_store as evidence
from m15_book_refresh_stats import StatisticsInputError, shadow_empty_set_status


REPO_ROOT = Path(__file__).resolve().parents[1]
SPEC_KIND = "research.protected_evaluation.sealed_spec/v1"
FAMILY_KIND = "research.protected_evaluation.family/v1"
RECEIPT_KIND = "research.protected_evaluation.access_receipt/v1"
EVALUATION_KIND = "research.protected_evaluation.evaluation_reference/v1"
TERMINAL_KIND = "research.protected_evaluation.terminal_decision/v1"
OUTCOME_SCHEMA = "research.protected_evaluation.shadow_outcome/v1"
DOMAIN_KINDS = (
    SPEC_KIND,
    FAMILY_KIND,
    RECEIPT_KIND,
    EVALUATION_KIND,
    TERMINAL_KIND,
)
EVALUATOR_PATH = "scripts/m15_book_refresh_stats.py"
EVALUATOR_FUNCTION = "shadow_empty_set_status"
ACCESS_SPENT = "spent_may_have_been_read"


class ProtectedEvaluationError(evidence.EvidenceError):
    """The protected-evaluation contract failed closed."""


def _exact_fields(value: Any, fields: set[str], name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ProtectedEvaluationError(f"{name} must be a JSON object")
    actual = set(value)
    if actual != fields:
        raise ProtectedEvaluationError(
            f"{name} fields differ: "
            f"missing={sorted(fields - actual)} unknown={sorted(actual - fields)}"
        )
    return value


def _text(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value.strip() != value
        or any(ord(char) < 0x20 for char in value)
    ):
        raise ProtectedEvaluationError(f"{name} must be a non-empty canonical string")
    return value


def _ordered_ids(value: Any, name: str, *, allow_empty: bool = False) -> list[str]:
    if type(value) not in {list, tuple}:
        raise ProtectedEvaluationError(f"{name} must be an ordered list")
    out = [_text(item, f"{name}[{index}]") for index, item in enumerate(value)]
    if not allow_empty and not out:
        raise ProtectedEvaluationError(f"{name} must not be empty")
    if len(out) != len(set(out)):
        raise ProtectedEvaluationError(f"{name} contains duplicate arms")
    return out


def _evaluator_payload() -> dict[str, str]:
    return {"path": EVALUATOR_PATH, "function": EVALUATOR_FUNCTION}


def _build_spec(
    *,
    repo_root: Path,
    experiment_id: str,
    hypothesis: str,
    falsifier: str,
    protected_input: evidence.ArtifactRef,
) -> tuple[evidence.EvidenceEnvelope, tuple[int, int] | None]:
    if type(protected_input) is not evidence.ArtifactRef:
        raise ProtectedEvaluationError("protected_input must be an ArtifactRef")
    if protected_input.path == EVALUATOR_PATH or protected_input.path.startswith(
        "evidence/objects/"
    ):
        raise ProtectedEvaluationError(
            "protected_input path aliases a required pre-access read surface"
        )
    experiment_id = _text(experiment_id, "experiment_id")
    protected_identity = protected_input.isolated_identity(
        repo_root=repo_root, allow_missing=True
    )
    evaluator = evidence.ArtifactRef.capture(EVALUATOR_PATH, repo_root=repo_root)
    if (
        protected_identity is not None
        and protected_input.isolated_identity(repo_root=repo_root)
        != protected_identity
    ):
        raise ProtectedEvaluationError("protected_input metadata changed before sealing")
    return (
        evidence.EvidenceEnvelope.create(
            kind=SPEC_KIND,
            payload={
                "schema": SPEC_KIND,
                "experiment_id": experiment_id,
                "hypothesis": _text(hypothesis, "hypothesis"),
                "falsifier": _text(falsifier, "falsifier"),
                "protected_input": protected_input.as_dict(),
                "evaluator": _evaluator_payload(),
                "terminal_rule": {
                    "allowed_statuses": [
                        "no_candidates",
                        "no_shadow_eligible_candidates",
                    ],
                    "activation": False,
                },
            },
            artifacts=[evaluator],
        ),
        protected_identity,
    )


def _build_family(
    spec: evidence.EvidenceEnvelope, ordered_arms: Sequence[str]
) -> evidence.EvidenceEnvelope:
    arms = _ordered_ids(ordered_arms, "ordered_arms")
    return evidence.EvidenceEnvelope.create(
        kind=FAMILY_KIND,
        payload={
            "schema": FAMILY_KIND,
            "experiment_id": spec.payload["experiment_id"],
            "spec_id": spec.object_id,
            "ordered_arms": arms,
        },
        dependencies=[spec.object_id],
    )


def _build_receipt(
    spec: evidence.EvidenceEnvelope, family: evidence.EvidenceEnvelope
) -> evidence.EvidenceEnvelope:
    return evidence.EvidenceEnvelope.create(
        kind=RECEIPT_KIND,
        payload={
            "schema": RECEIPT_KIND,
            "experiment_id": spec.payload["experiment_id"],
            "spec_id": spec.object_id,
            "family_id": family.object_id,
            "protected_input": spec.payload["protected_input"],
            "access_state": ACCESS_SPENT,
        },
        dependencies=[family.object_id],
    )


def _normalize_outcome(value: Any, ordered_arms: Sequence[str]) -> dict[str, Any]:
    outcome = _exact_fields(
        value,
        {"schema", "survivors", "shadow_eligible", "exclusions"},
        "protected outcome",
    )
    if outcome["schema"] != OUTCOME_SCHEMA:
        raise ProtectedEvaluationError(
            f"unsupported protected outcome schema {outcome['schema']!r}"
        )
    family = list(ordered_arms)
    survivors = _ordered_ids(outcome["survivors"], "survivors", allow_empty=True)
    eligible = _ordered_ids(
        outcome["shadow_eligible"], "shadow_eligible", allow_empty=True
    )
    family_set = set(family)
    unknown = [arm for arm in survivors if arm not in family_set]
    expected_order = [arm for arm in family if arm in set(survivors)]
    if unknown:
        raise ProtectedEvaluationError(f"survivors contain unregistered arms: {unknown}")
    if survivors != expected_order:
        raise ProtectedEvaluationError("survivors do not preserve sealed family order")
    if not isinstance(outcome["exclusions"], dict):
        raise ProtectedEvaluationError("exclusions must be a JSON object")
    exclusions: dict[str, str] = {}
    for arm, reason in outcome["exclusions"].items():
        exclusions[_text(arm, "exclusion arm")] = _text(
            reason, f"exclusion reason for {arm!r}"
        )
    survivor_set = set(survivors)
    unknown_eligible = [arm for arm in eligible if arm not in survivor_set]
    expected_eligible = [arm for arm in survivors if arm in set(eligible)]
    if unknown_eligible:
        raise ProtectedEvaluationError(
            f"shadow_eligible contains non-survivors: {unknown_eligible}"
        )
    if eligible != expected_eligible:
        raise ProtectedEvaluationError(
            "shadow_eligible does not preserve survivor order"
        )
    if eligible:
        raise ProtectedEvaluationError(
            "this vertical supports only an empty shadow-eligible set"
        )
    missing_exclusions = [arm for arm in survivors if arm not in exclusions]
    unknown_exclusions = [arm for arm in exclusions if arm not in survivor_set]
    if missing_exclusions or unknown_exclusions:
        raise ProtectedEvaluationError(
            "exclusions must give one reason per survivor: "
            f"missing={missing_exclusions} unknown={unknown_exclusions}"
        )
    try:
        output = shadow_empty_set_status(
            survivors, eligible, exclusions=exclusions
        )
    except StatisticsInputError as exc:
        raise ProtectedEvaluationError(f"protected outcome is invalid: {exc}") from exc
    return {
        "schema": OUTCOME_SCHEMA,
        "survivors": survivors,
        "shadow_eligible": eligible,
        "exclusions": exclusions,
        "expected_output": output,
    }


def _require_sealed_input(value: Any, spec: evidence.EvidenceEnvelope) -> None:
    raw = evidence.canonical_json_bytes(value)
    declared = evidence.ArtifactRef.from_dict(spec.payload["protected_input"])
    if len(raw) != declared.bytes or hashlib.sha256(raw).hexdigest() != declared.sha256:
        raise ProtectedEvaluationError(
            "normalized evaluator input does not match the sealed protected bytes"
        )


def _build_evaluation(
    *,
    spec: evidence.EvidenceEnvelope,
    family: evidence.EvidenceEnvelope,
    receipt: evidence.EvidenceEnvelope,
    normalized: dict[str, Any],
) -> evidence.EvidenceEnvelope:
    evaluator_input = {
        key: normalized[key]
        for key in ("schema", "survivors", "shadow_eligible", "exclusions")
    }
    _require_sealed_input(evaluator_input, spec)
    return evidence.EvidenceEnvelope.create(
        kind=EVALUATION_KIND,
        payload={
            "schema": EVALUATION_KIND,
            "experiment_id": spec.payload["experiment_id"],
            "spec_id": spec.object_id,
            "family_id": family.object_id,
            "receipt_id": receipt.object_id,
            "evaluator": _evaluator_payload(),
            "normalized_input": evaluator_input,
            "output": normalized["expected_output"],
        },
        dependencies=[receipt.object_id],
    )


def _build_terminal(
    *,
    spec: evidence.EvidenceEnvelope,
    family: evidence.EvidenceEnvelope,
    receipt: evidence.EvidenceEnvelope,
    evaluation: evidence.EvidenceEnvelope,
) -> evidence.EvidenceEnvelope:
    output = evaluation.payload["output"]
    status = output["status"]
    if status == "no_candidates":
        decision = "no_candidate"
        candidates: list[dict[str, str]] = []
    elif status == "no_shadow_eligible_candidates":
        decision = "inactive_only"
        candidates = [
            {
                "candidate_id": arm,
                "lifecycle_status": "inactive_shadow_candidate",
            }
            for arm in output["S"]
        ]
    else:  # pragma: no cover - the delegated helper has exactly two states
        raise ProtectedEvaluationError(f"unsupported evaluator status {status!r}")
    return evidence.EvidenceEnvelope.create(
        kind=TERMINAL_KIND,
        payload={
            "schema": TERMINAL_KIND,
            "experiment_id": spec.payload["experiment_id"],
            "spec_id": spec.object_id,
            "family_id": family.object_id,
            "receipt_id": receipt.object_id,
            "evaluation_id": evaluation.object_id,
            "decision": decision,
            "activation": False,
            "candidates": candidates,
        },
        dependencies=[evaluation.object_id],
    )


def _domain_experiment(envelope: evidence.EvidenceEnvelope) -> str:
    fields = {
        SPEC_KIND: {
            "schema",
            "experiment_id",
            "hypothesis",
            "falsifier",
            "protected_input",
            "evaluator",
            "terminal_rule",
        },
        FAMILY_KIND: {"schema", "experiment_id", "spec_id", "ordered_arms"},
        RECEIPT_KIND: {
            "schema",
            "experiment_id",
            "spec_id",
            "family_id",
            "protected_input",
            "access_state",
        },
        EVALUATION_KIND: {
            "schema",
            "experiment_id",
            "spec_id",
            "family_id",
            "receipt_id",
            "evaluator",
            "normalized_input",
            "output",
        },
        TERMINAL_KIND: {
            "schema",
            "experiment_id",
            "spec_id",
            "family_id",
            "receipt_id",
            "evaluation_id",
            "decision",
            "activation",
            "candidates",
        },
    }[envelope.kind]
    payload = _exact_fields(envelope.payload, fields, f"{envelope.kind} payload")
    if payload["schema"] != envelope.kind:
        raise ProtectedEvaluationError(
            f"payload schema {payload['schema']!r} != kind {envelope.kind!r}"
        )
    return _text(payload["experiment_id"], "experiment_id")


def _matching_domain_objects(
    repo_root: Path, experiment_id: str
) -> dict[str, list[evidence.EvidenceEnvelope]]:
    groups = {kind: [] for kind in DOMAIN_KINDS}
    for envelope in evidence.inspect_store_metadata(repo_root=repo_root):
        if envelope.kind not in groups:
            continue
        if _domain_experiment(envelope) == experiment_id:
            groups[envelope.kind].append(envelope)
    return groups


def _one_expected(
    groups: dict[str, list[evidence.EvidenceEnvelope]],
    expected: evidence.EvidenceEnvelope,
) -> None:
    matches = groups[expected.kind]
    if len(matches) != 1 or matches[0].as_dict() != expected.as_dict():
        raise ProtectedEvaluationError(
            f"hidden or conflicting {expected.kind} state for "
            f"{expected.payload['experiment_id']}"
        )


def _optional_expected(
    groups: dict[str, list[evidence.EvidenceEnvelope]],
    expected: evidence.EvidenceEnvelope,
) -> evidence.EvidenceEnvelope | None:
    matches = groups[expected.kind]
    if not matches:
        return None
    if len(matches) != 1 or matches[0].as_dict() != expected.as_dict():
        raise ProtectedEvaluationError(
            f"hidden or conflicting {expected.kind} state for "
            f"{expected.payload['experiment_id']}"
        )
    return matches[0]


def _validated_evaluation(
    envelope: evidence.EvidenceEnvelope,
    *,
    spec: evidence.EvidenceEnvelope,
    family: evidence.EvidenceEnvelope,
    receipt: evidence.EvidenceEnvelope,
) -> evidence.EvidenceEnvelope:
    _domain_experiment(envelope)
    if envelope.dependencies != (receipt.object_id,) or envelope.artifacts:
        raise ProtectedEvaluationError("evaluation reference has invalid bindings")
    payload = envelope.payload
    if (
        payload["spec_id"] != spec.object_id
        or payload["family_id"] != family.object_id
        or payload["receipt_id"] != receipt.object_id
        or payload["evaluator"] != _evaluator_payload()
    ):
        raise ProtectedEvaluationError("evaluation reference ancestry differs")
    _require_sealed_input(payload["normalized_input"], spec)
    normalized = _normalize_outcome(
        payload["normalized_input"], family.payload["ordered_arms"]
    )
    expected = _build_evaluation(
        spec=spec, family=family, receipt=receipt, normalized=normalized
    )
    if envelope.as_dict() != expected.as_dict():
        raise ProtectedEvaluationError("evaluation reference is not reproducible")
    return envelope


def _resume_or_spent(
    *,
    repo_root: Path,
    spec: evidence.EvidenceEnvelope,
    family: evidence.EvidenceEnvelope,
    receipt: evidence.EvidenceEnvelope,
) -> evidence.EvidenceEnvelope | None:
    groups = _matching_domain_objects(repo_root, spec.payload["experiment_id"])
    _one_expected(groups, spec)
    _one_expected(groups, family)
    stored_receipt = _optional_expected(groups, receipt)
    evaluations = groups[EVALUATION_KIND]
    terminals = groups[TERMINAL_KIND]
    if len(evaluations) > 1 or len(terminals) > 1:
        raise ProtectedEvaluationError("hidden or conflicting evaluation descendants")
    if evaluations and stored_receipt is None:
        raise ProtectedEvaluationError("evaluation reference has no access receipt")
    if terminals and not evaluations:
        raise ProtectedEvaluationError("terminal decision has no evaluation reference")
    if evaluations:
        evaluation = _validated_evaluation(
            evaluations[0], spec=spec, family=family, receipt=receipt
        )
        expected_terminal = _build_terminal(
            spec=spec,
            family=family,
            receipt=receipt,
            evaluation=evaluation,
        )
        if terminals:
            if terminals[0].as_dict() != expected_terminal.as_dict():
                raise ProtectedEvaluationError("terminal decision is not reproducible")
            return terminals[0]
        evidence.publish(expected_terminal, repo_root=repo_root)
        return expected_terminal
    if terminals:
        raise ProtectedEvaluationError("terminal decision ancestry is incomplete")
    if stored_receipt is not None:
        raise ProtectedEvaluationError(
            "BLOCKED: access receipt exists without an evaluation reference; "
            "the protected look is ambiguously spent"
        )
    return None


def run_protected_evaluation(
    *,
    repo_root: str | Path = REPO_ROOT,
    experiment_id: str,
    hypothesis: str,
    falsifier: str,
    ordered_arms: Sequence[str],
    protected_input: evidence.ArtifactRef,
) -> evidence.EvidenceEnvelope:
    """Run or deterministically resume one sealed empty-shadow evaluation.

    Precondition: ``protected_input`` metadata is declared without this adapter
    capturing or reading the protected path.
    """

    root = Path(repo_root)
    spec, protected_identity = _build_spec(
        repo_root=root,
        experiment_id=experiment_id,
        hypothesis=hypothesis,
        falsifier=falsifier,
        protected_input=protected_input,
    )
    family = _build_family(spec, ordered_arms)
    receipt = _build_receipt(spec, family)
    evidence.publish(spec, repo_root=root)
    evidence.publish(family, repo_root=root)
    resumed = _resume_or_spent(
        repo_root=root, spec=spec, family=family, receipt=receipt
    )
    if resumed is not None:
        return resumed
    if protected_identity is None:
        raise ProtectedEvaluationError(
            "protected_input is unavailable before access-receipt publication"
        )

    evidence.publish(receipt, repo_root=root, require_new=True)
    raw = protected_input.read_verified(repo_root=root)
    try:
        outcome = evidence.decode_canonical_json(raw)
    except evidence.EvidenceError as exc:
        raise ProtectedEvaluationError(f"protected outcome bytes are invalid: {exc}") from exc
    normalized = _normalize_outcome(outcome, family.payload["ordered_arms"])
    evaluation = _build_evaluation(
        spec=spec, family=family, receipt=receipt, normalized=normalized
    )
    evidence.publish(evaluation, repo_root=root)
    terminal = _build_terminal(
        spec=spec,
        family=family,
        receipt=receipt,
        evaluation=evaluation,
    )
    evidence.publish(terminal, repo_root=root)
    return terminal
