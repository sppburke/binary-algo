#!/usr/bin/env python3
"""Immutable successor for the ROS-3A source-key variants.

The failed v1 acquisition specification binds ``deriv_economics_ledger.py``
byte-for-byte.  This module therefore owns the v2 acquisition authority and
orchestration explicitly while reusing only v1 helpers whose behavior is
independent of acquisition identity and source-key admission.
"""

from __future__ import annotations

import argparse
import copy
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any, Sequence

import deriv_economics_ledger as v1
import evidence_store as evidence


REPO_ROOT = Path(__file__).resolve().parents[1]
PREDECESSOR_SPEC_ID = "a0ba3d8dfdb187b2abc9952fa897c8bdf4f9f2815d825757dd19665740c532d4"
FROZEN_V1_SHA256 = "7163a502ccda0cdcbf6b4ec9eee3665a0e503bd51cbd5be0fd4382a10a00de1e"
EXPORTER_PATH = "scripts/deriv_economics_ledger_v2.py"
BOUND_ARTIFACTS = (
    EXPORTER_PATH,
    v1.EXPORTER_PATH,
    v1.EVIDENCE_PATH,
    v1.MANIFEST_PATH,
)

SOURCE_KEY_VARIANTS = {
    event: (fields,) for event, fields in v1.SOURCE_KEYS.items()
}
SOURCE_KEY_VARIANTS["executor_startup"] = (
    v1.SOURCE_KEYS["executor_startup"],
    v1.SOURCE_KEYS["executor_startup"] | {"allocation", "reconciliation"},
)
SOURCE_KEY_VARIANTS["proposal_selected"] = (
    v1.SOURCE_KEYS["proposal_selected"],
    v1.SOURCE_KEYS["proposal_selected"] | {"net_edge"},
)

DerivEconomicsError = v1.DerivEconomicsError
ACQUISITION_KIND = v1.ACQUISITION_KIND
SEALED_KIND = v1.SEALED_KIND
ACCESS_KIND = v1.ACCESS_KIND
NORMALIZED_KIND = v1.NORMALIZED_KIND
PACKET_FIELDS = v1.PACKET_FIELDS
EVIDENCE_CLASSES = v1.EVIDENCE_CLASSES
ACCESS_STATE = v1.ACCESS_STATE
DB_LOGICAL_PATH = v1.DB_LOGICAL_PATH
LOG_LOGICAL_PATH = v1.LOG_LOGICAL_PATH
settle_direction = v1.settle_direction
evidence_class_result = v1.evidence_class_result
_decimal = v1._decimal
_domain_objects = v1._domain_objects


def acquisition_payload() -> dict[str, Any]:
    payload = copy.deepcopy(v1.acquisition_payload())
    payload["acquisition_id"] = "ROS-3A-USDJPY-UP-2026-07-06-v2"
    payload["predecessor"] = {
        "acquisition_spec_id": PREDECESSOR_SPEC_ID,
        "state": "blocked_before_packet",
        "reason": "observed exact source-key variants contradicted the v1 whitelist",
    }
    payload["source_row_keys"] = {
        event: [sorted(fields) for fields in variants]
        for event, variants in SOURCE_KEY_VARIANTS.items()
    }
    payload["falsifiers"].append("any source-row key set outside the exact v2 variants")
    return payload


def build_acquisition_spec(repo_root: Path) -> evidence.EvidenceEnvelope:
    artifacts = [
        evidence.ArtifactRef.capture(path, repo_root=repo_root)
        for path in BOUND_ARTIFACTS
    ]
    if artifacts[1].sha256 != FROZEN_V1_SHA256:
        raise DerivEconomicsError("frozen v1 exporter bytes differ")
    return evidence.EvidenceEnvelope.create(
        kind=ACQUISITION_KIND,
        payload=acquisition_payload(),
        artifacts=artifacts,
        dependencies=[PREDECESSOR_SPEC_ID],
    )


def seal_acquisition(repo_root: Path = REPO_ROOT) -> evidence.EvidenceEnvelope:
    predecessor = evidence.verify_object(PREDECESSOR_SPEC_ID, repo_root=repo_root)
    if predecessor.kind != ACQUISITION_KIND:
        raise DerivEconomicsError("predecessor acquisition kind differs")
    envelope = build_acquisition_spec(repo_root)
    evidence.publish(envelope, repo_root=repo_root)
    return evidence.verify_object(envelope.object_id, repo_root=repo_root)


def _validate_acquisition_spec(
    spec_id: str, *, repo_root: Path
) -> evidence.EvidenceEnvelope:
    try:
        spec = evidence.verify_object(spec_id, repo_root=repo_root)
    except evidence.EvidenceError as exc:
        raise DerivEconomicsError(f"v2 acquisition authority does not verify: {exc}") from exc
    if spec.kind != ACQUISITION_KIND or spec.payload != acquisition_payload():
        raise DerivEconomicsError("v2 acquisition authority kind or payload differs")
    expected = tuple(
        evidence.ArtifactRef.capture(path, repo_root=repo_root)
        for path in BOUND_ARTIFACTS
    )
    if expected[1].sha256 != FROZEN_V1_SHA256:
        raise DerivEconomicsError("frozen v1 exporter bytes differ")
    if spec.artifacts != expected or spec.dependencies != (PREDECESSOR_SPEC_ID,):
        raise DerivEconomicsError("v2 acquisition authority bindings differ")
    return spec


def _decode_log(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DerivEconomicsError(f"invalid JSONL row {number}") from exc
            if not isinstance(row, dict):
                raise DerivEconomicsError(f"JSONL row {number} is not an object")
            v1._reject_secret_keys(row, location=f"log[{number}]")
            event = row.get("event")
            if event in SOURCE_KEY_VARIANTS:
                actual = frozenset(row)
                if actual not in SOURCE_KEY_VARIANTS[event]:
                    raise DerivEconomicsError(
                        f"{event} row fields do not match an exact v2 variant"
                    )
                if "url" in row:
                    raise DerivEconomicsError("url source key is forbidden")
                events.append(row)
    return events


def export_packet(
    *,
    spec_id: str,
    authority_root: Path,
    db_path: Path,
    log_path: Path,
    output_dir: Path,
    enforce_remote_paths: bool = False,
) -> dict[str, Any]:
    spec = _validate_acquisition_spec(spec_id, repo_root=authority_root)
    expected_exporter = next(item for item in spec.artifacts if item.path == EXPORTER_PATH)
    expected_db = v1.REMOTE_ROOT / v1.DB_LOGICAL_PATH
    expected_log = v1.REMOTE_ROOT / v1.LOG_LOGICAL_PATH
    if enforce_remote_paths and (
        db_path != expected_db
        or log_path != expected_log
        or output_dir != v1.REMOTE_ROOT / "deriv_data/venue_evidence"
    ):
        raise DerivEconomicsError("remote source paths differ from the sealed v2 contract")

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, isolation_level=None)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA query_only=ON")
        if conn.execute("PRAGMA query_only").fetchone()[0] != 1:
            raise DerivEconomicsError("SQLite query_only did not engage")
        conn.execute("BEGIN")
        db_rows = v1._eligible_db_rows(conn)
        events = _decode_log(log_path)
        candidates: list[dict[str, Any]] = []
        errors: list[str] = []
        for row in db_rows:
            try:
                candidates.append(v1._linked_candidate(row, events))
            except DerivEconomicsError as exc:
                errors.append(str(exc))
        conn.execute("COMMIT")
    except Exception:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()
    if not candidates:
        raise DerivEconomicsError(
            f"no complete eligible retained chain under v2: rejected={len(errors)}"
        )
    selected = min(candidates, key=lambda item: item["order"])["packet"]
    selected["source"]["acquisition_spec_id"] = spec.object_id
    selected["source"]["exporter_sha256"] = expected_exporter.sha256
    v1._reject_packet_secrets(selected)
    destination, raw, digest = v1._atomic_packet(output_dir, selected)
    descriptor_root = v1.REMOTE_ROOT if enforce_remote_paths else authority_root
    return {
        "relative_path": destination.relative_to(descriptor_root).as_posix(),
        "bytes": len(raw),
        "sha256": digest,
        "fixed_contract": {
            "source_date": v1.SOURCE_DATE,
            "pair": v1.SOURCE_PAIR,
            "side": v1.SOURCE_SIDE,
            "contract_type": "CALL",
            "duration": 15,
            "duration_unit": "m",
            "environment": "DEMO",
            "real_money": False,
        },
        "source_clocks": selected["source"]["selected_source_clocks"],
    }


def _normalize_packet(
    packet: Any,
    *,
    source_ref: evidence.ArtifactRef,
    spec: evidence.EvidenceEnvelope,
    descriptor: dict[str, Any],
) -> dict[str, Any]:
    packet = v1._exact_fields(packet, PACKET_FIELDS, "packet")
    if packet["schema"] != v1.PACKET_SCHEMA:
        raise DerivEconomicsError("packet schema differs")
    v1._reject_packet_secrets(packet)
    source = v1._exact_fields(
        packet["source"],
        {
            "acquisition_spec_id", "exporter_sha256", "database", "log",
            "source_date", "selector", "selected_source_clocks",
            "recorded_query_time_runtime_revision", "runtime_revision_at_trade",
        },
        "packet source",
    )
    exporter = next(item for item in spec.artifacts if item.path == EXPORTER_PATH)
    if (
        source["acquisition_spec_id"] != spec.object_id
        or source["exporter_sha256"] != exporter.sha256
        or source["database"] != v1.DB_LOGICAL_PATH
        or source["log"] != v1.LOG_LOGICAL_PATH
        or source["source_date"] != v1.SOURCE_DATE
        or source["selector"]
        != "earliest_complete_by_contract_created_utc_then_contract_id"
        or source["recorded_query_time_runtime_revision"] != v1.RUNTIME_REVISION
        or source["runtime_revision_at_trade"] != "not_retained"
    ):
        raise DerivEconomicsError("packet source provenance differs")
    clocks = v1._exact_fields(
        source["selected_source_clocks"],
        {
            "proposal_utc", "submission_recorded_utc", "confirmation_recorded_utc",
            "database_terminal_utc", "terminal_update_utc", "contract_closed_utc",
        },
        "selected source clocks",
    )
    ordered_clocks = [
        v1._utc(clocks[key], key)
        for key in (
            "proposal_utc", "submission_recorded_utc", "confirmation_recorded_utc",
            "database_terminal_utc", "terminal_update_utc", "contract_closed_utc",
        )
    ]
    if ordered_clocks != sorted(ordered_clocks):
        raise DerivEconomicsError("packet source clocks are not ordered")
    expected_fixed = {
        "source_date": v1.SOURCE_DATE, "pair": v1.SOURCE_PAIR, "side": v1.SOURCE_SIDE,
        "contract_type": "CALL", "duration": 15, "duration_unit": "m",
        "environment": "DEMO", "real_money": False,
    }
    if descriptor["fixed_contract"] != expected_fixed or descriptor["source_clocks"] != clocks:
        raise DerivEconomicsError("non-outcome descriptor differs from packet")

    account = v1._exact_fields(
        packet["account"], {"environment", "funding", "real_money", "provenance"},
        "account",
    )
    expected_account = {
        "environment": "DEMO", "funding": "VIRTUAL", "real_money": False,
        "provenance": {
            "same_executor_epoch_authenticated": True,
            "meaning": "recorded_query_time_code_marks_authenticated_true_only_for_demo_connections",
        },
    }
    if account != expected_account:
        raise DerivEconomicsError("packet is not the admitted demo provenance")
    linkage = v1._exact_fields(
        packet["linkage"], {"signal_id", "proposal_id", "contract_id"}, "linkage"
    )
    for key in linkage:
        v1._text(linkage[key], f"linkage.{key}")
    contract = v1._exact_fields(
        packet["contract"],
        {
            "pair", "side", "recorded_query_time_runtime_contract",
            "observed_behavior", "reference_semantics",
        },
        "contract",
    )
    runtime = contract["recorded_query_time_runtime_contract"]
    if (
        contract["pair"] != v1.SOURCE_PAIR
        or contract["side"] != v1.SOURCE_SIDE
        or runtime
        != {
            "contract_type": "CALL", "basis": "stake", "currency": "USD",
            "duration": 15, "duration_unit": "m",
        }
    ):
        raise DerivEconomicsError("packet product contract differs")
    behavior = contract["observed_behavior"]
    if (
        not isinstance(behavior, dict)
        or set(behavior) != {"terminal_status", "early_sale"}
        or behavior.get("terminal_status") not in {"won", "lost"}
        or behavior.get("early_sale") is not False
    ):
        raise DerivEconomicsError("packet terminal behavior differs")
    if contract["reference_semantics"] != {"hold_to_expiry": True, "equality": "loses"}:
        raise DerivEconomicsError("packet reference semantics differ")

    proposal = v1._exact_fields(
        packet["proposal"],
        {"observed_utc", "proposal_id", "proposal_source", "ask", "payout"},
        "proposal",
    )
    buy = v1._exact_fields(
        packet["buy"],
        {
            "submission_recorded_utc", "confirmation_recorded_utc", "proposal_id",
            "contract_id", "recorded_buy_price_projection", "stake",
        },
        "buy",
    )
    terminal = v1._exact_fields(
        packet["terminal"],
        {
            "observed_utc", "contract_id", "status", "sell_price",
            "recorded_profit", "payout", "terminal_utc", "raw_hash16",
        },
        "terminal",
    )
    if (
        proposal["proposal_id"] != linkage["proposal_id"]
        or buy["proposal_id"] != linkage["proposal_id"]
        or buy["contract_id"] != linkage["contract_id"]
        or terminal["contract_id"] != linkage["contract_id"]
    ):
        raise DerivEconomicsError("packet linkage differs")
    if (
        terminal["status"] != behavior["terminal_status"]
        or v1.HEX16.fullmatch(v1._text(terminal["raw_hash16"], "raw_hash16")) is None
    ):
        raise DerivEconomicsError("packet terminal status or marker differs")
    ask = v1._decimal(proposal["ask"], "proposal ask")
    proposal_payout = v1._decimal(proposal["payout"], "proposal payout")
    buy_price = v1._decimal(
        buy["recorded_buy_price_projection"], "recorded buy-price projection"
    )
    stake = v1._decimal(buy["stake"], "stake")
    sell = v1._decimal(terminal["sell_price"], "terminal sell price")
    recorded_profit = v1._decimal(terminal["recorded_profit"], "terminal recorded profit")
    terminal_payout = v1._decimal(terminal["payout"], "terminal payout")
    if (
        ask <= 0 or proposal_payout <= 0 or buy_price <= 0 or stake <= 0
        or sell < 0 or terminal_payout <= 0
    ):
        raise DerivEconomicsError("packet money signs differ")
    implied_cost = sell - recorded_profit
    realized = sell - implied_cost
    if not (implied_cost == buy_price == stake == ask and realized == recorded_profit):
        raise DerivEconomicsError("terminal economics do not reconcile")
    if terminal["status"] == "won":
        if recorded_profit <= 0 or sell != terminal_payout:
            raise DerivEconomicsError("winning terminal economics differ")
    elif sell != v1.Decimal("0.00") or recorded_profit != -implied_cost:
        raise DerivEconomicsError("losing terminal economics differ")
    expected_retention = {
        "request_payload_retained": False,
        "raw_response_retained": False,
        "buy_response_values_retained": False,
        "buy_response_hash_retained": False,
        "runtime_revision_at_trade_retained": False,
        "entry_tick_retained": False,
        "exit_tick_retained": False,
    }
    if packet["retention"] != expected_retention:
        raise DerivEconomicsError("packet retention boundary differs")
    classification = v1.evidence_class_result(
        "VENUE_SETTLED", realized_profit=format(realized, ".2f")
    )
    return {
        "schema": NORMALIZED_KIND,
        "evidence_class": classification["evidence_class"],
        "source": {"artifact": source_ref.as_dict(), "acquisition_spec_id": spec.object_id},
        "observation": packet,
        "derivation": {
            "currency": "USD",
            "quoted_win_profit": format(proposal_payout - ask, ".2f"),
            "quoted_loss_profit": format(-ask, ".2f"),
            "terminal_implied_cost": format(implied_cost, ".2f"),
            "realized_profit": classification["realized_profit"],
            "breakeven_fraction": {
                "numerator": format(ask, ".2f"),
                "denominator": format(proposal_payout, ".2f"),
            },
            "proposal_and_terminal_payout_fields_preserved_separately": True,
            "entry_exit_direction": "unverified_not_retained",
        },
        "claim_limits": {
            "request_response_bytes": "not_retained",
            "runtime_revision_at_trade": "not_retained",
            "entry_exit_ticks": "not_retained",
            "real_account_parity": "not_claimed",
            "strategy_edge": "not_claimed",
        },
        "activation": False,
    }


def reduce_packet(
    *,
    repo_root: Path,
    spec_id: str,
    source_ref: evidence.ArtifactRef,
    descriptor: dict[str, Any],
) -> evidence.EvidenceEnvelope:
    spec = _validate_acquisition_spec(spec_id, repo_root=repo_root)
    v1._exact_fields(
        descriptor,
        {"relative_path", "bytes", "sha256", "fixed_contract", "source_clocks"},
        "descriptor",
    )
    if (
        descriptor.get("relative_path") != source_ref.path
        or descriptor.get("bytes") != source_ref.bytes
        or descriptor.get("sha256") != source_ref.sha256
    ):
        raise DerivEconomicsError("source descriptor differs from declared ArtifactRef")
    source_ref.isolated_identity(repo_root=repo_root)
    sealed = evidence.EvidenceEnvelope.create(
        kind=SEALED_KIND,
        payload={
            "schema": SEALED_KIND,
            "acquisition_spec_id": spec.object_id,
            "source_artifact": source_ref.as_dict(),
            "descriptor": descriptor,
        },
        dependencies=[spec.object_id],
    )
    receipt = evidence.EvidenceEnvelope.create(
        kind=ACCESS_KIND,
        payload={
            "schema": ACCESS_KIND,
            "acquisition_spec_id": spec.object_id,
            "sealed_source_id": sealed.object_id,
            "source_artifact": source_ref.as_dict(),
            "access_state": ACCESS_STATE,
        },
        dependencies=[sealed.object_id],
    )
    groups = v1._domain_objects(repo_root, spec.object_id)
    if any(len(items) > 1 for items in groups.values()):
        raise DerivEconomicsError("hidden or conflicting v2 economics descendants")
    if groups[SEALED_KIND] and groups[SEALED_KIND][0].as_dict() != sealed.as_dict():
        raise DerivEconomicsError("conflicting sealed source exists")
    if groups[ACCESS_KIND] and groups[ACCESS_KIND][0].as_dict() != receipt.as_dict():
        raise DerivEconomicsError("conflicting access receipt exists")
    if groups[NORMALIZED_KIND]:
        normalized = groups[NORMALIZED_KIND][0]
        if not groups[ACCESS_KIND] or normalized.dependencies != (receipt.object_id,):
            raise DerivEconomicsError("normalized receipt ancestry differs")
        if normalized.payload.get("source", {}).get("artifact") != source_ref.as_dict():
            raise DerivEconomicsError("normalized receipt source differs")
        expected_payload = _normalize_packet(
            normalized.payload["observation"],
            source_ref=source_ref,
            spec=spec,
            descriptor=descriptor,
        )
        if normalized.payload != expected_payload:
            raise DerivEconomicsError("normalized receipt is not reproducible")
        evidence.verify_object(normalized.object_id, repo_root=repo_root)
        return normalized
    if groups[ACCESS_KIND]:
        raise DerivEconomicsError(
            "BLOCKED: v2 source access receipt exists without normalized receipt"
        )
    evidence.publish(sealed, repo_root=repo_root)
    evidence.publish(receipt, repo_root=repo_root, require_new=True)
    raw = source_ref.read_verified(repo_root=repo_root)
    try:
        packet = evidence.decode_canonical_json(raw)
    except evidence.EvidenceError as exc:
        raise DerivEconomicsError(f"source packet bytes are invalid: {exc}") from exc
    normalized_payload = _normalize_packet(
        packet, source_ref=source_ref, spec=spec, descriptor=descriptor
    )
    normalized = evidence.EvidenceEnvelope.create(
        kind=NORMALIZED_KIND,
        payload=normalized_payload,
        dependencies=[receipt.object_id],
    )
    evidence.publish(normalized, repo_root=repo_root)
    return normalized


def _descriptor(value: str) -> dict[str, Any]:
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError as exc:
        raise DerivEconomicsError("descriptor is not JSON") from exc
    return v1._exact_fields(
        decoded,
        {"relative_path", "bytes", "sha256", "fixed_contract", "source_clocks"},
        "descriptor",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("seal-acquisition")
    export = commands.add_parser("export-one")
    export.add_argument("--acquisition-spec-id", required=True)
    reduce = commands.add_parser("reduce")
    reduce.add_argument("--acquisition-spec-id", required=True)
    reduce.add_argument("--source-path", required=True)
    reduce.add_argument("--source-bytes", required=True, type=int)
    reduce.add_argument("--source-sha256", required=True)
    reduce.add_argument("--descriptor-json", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = args.repo_root.resolve()
    try:
        if args.command == "seal-acquisition":
            spec = seal_acquisition(root)
            print(json.dumps({"acquisition_spec_id": spec.object_id}, sort_keys=True))
        elif args.command == "export-one":
            descriptor = export_packet(
                spec_id=args.acquisition_spec_id,
                authority_root=root,
                db_path=v1.REMOTE_ROOT / v1.DB_LOGICAL_PATH,
                log_path=v1.REMOTE_ROOT / v1.LOG_LOGICAL_PATH,
                output_dir=v1.REMOTE_ROOT / "deriv_data/venue_evidence",
                enforce_remote_paths=True,
            )
            print(json.dumps(descriptor, sort_keys=True, separators=(",", ":")))
        else:
            source_ref = evidence.ArtifactRef(
                path=args.source_path, bytes=args.source_bytes, sha256=args.source_sha256
            )
            normalized = reduce_packet(
                repo_root=root,
                spec_id=args.acquisition_spec_id,
                source_ref=source_ref,
                descriptor=_descriptor(args.descriptor_json),
            )
            print(json.dumps({"normalized_receipt_id": normalized.object_id}, sort_keys=True))
    except (DerivEconomicsError, evidence.EvidenceError, OSError, sqlite3.Error) as exc:
        print(f"Deriv economics ledger v2 failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
