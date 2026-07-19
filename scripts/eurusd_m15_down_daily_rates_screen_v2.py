#!/usr/bin/env python3
"""Recovery wrapper for the pre-attempt v1 frozen-container mismatch.

V1 sealed its source snapshot and campaign specification, then rejected the
runner-frozen input envelope before attempt publication.  This successor binds
the preserved v1 implementation as an input artifact and changes only JSON
tuple/mapping thawing at the adapter boundary.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Sequence

import eurusd_m15_down_daily_rates_screen_v1 as impl


ROOT = Path(__file__).resolve().parents[1]
V1_SCRIPT_PATH = "scripts/eurusd_m15_down_daily_rates_screen_v1.py"
impl.CAMPAIGN_ID = "eurusd.m15.down.daily-rates-screen.v2"
impl.SCRIPT_PATH = "scripts/eurusd_m15_down_daily_rates_screen_v2.py"
impl.SPEC_PATH = "results/json/eurusd_m15_down_daily_rates_screen_v2_campaign_spec.json"
impl.RESULT_PATH = "results/json/eurusd_m15_down_daily_rates_screen_v2_result.json"
impl.INPUT_SCHEMA = "research.eurusd_m15_down_daily_rates.input/v2"
_v1_input_payload = impl._input_payload


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _input_payload() -> dict[str, Any]:
    value = _v1_input_payload()
    value["ordered_artifact_paths"] = [impl.SNAPSHOT_PATH, impl.SELECTION_PATH, V1_SCRIPT_PATH]
    return value


def _validate_input_envelope(value: Mapping[str, Any], expected_id: str) -> Mapping[str, Any]:
    raw = impl._exact(
        _thaw(value),
        {"schema", "object_id", "kind", "payload", "artifacts", "dependencies"},
        "input envelope",
    )
    expected_paths = [impl.SNAPSHOT_PATH, impl.SELECTION_PATH, V1_SCRIPT_PATH]
    payload = impl._exact(
        raw["payload"],
        {
            "schema", "key", "data_use", "ordered_artifact_paths", "selection_sha256",
            "source_snapshot_schema", "measurement", "protected_access", "claim_limit",
        },
        "input payload",
    )
    if (
        raw["object_id"] != expected_id
        or raw["kind"] != impl.INPUT_SCHEMA
        or list(raw["dependencies"]) != [impl.ISSUE15_TERMINAL]
        or payload != _input_payload()
        or [item.get("path") for item in raw["artifacts"]] != expected_paths
    ):
        raise impl.ScreenError("input envelope identity/dependency differs")
    return raw


impl._input_payload = _input_payload
impl._validate_input_envelope = _validate_input_envelope


def _prepare_and_run() -> None:
    import research_campaign_v1 as runner

    snapshot_path = ROOT / impl.SNAPSHOT_PATH
    if not snapshot_path.exists():
        raise impl.ScreenError("v2 requires the already-durable v1 source snapshot")
    impl._validate_snapshot(impl._canonical_read(snapshot_path))
    artifacts = [
        impl.evidence.ArtifactRef.capture(path, repo_root=ROOT)
        for path in (impl.SNAPSHOT_PATH, impl.SELECTION_PATH, V1_SCRIPT_PATH)
    ]
    input_envelope = impl.evidence.EvidenceEnvelope.create(
        kind=impl.INPUT_SCHEMA,
        payload=_input_payload(),
        artifacts=artifacts,
        dependencies=[impl.ISSUE15_TERMINAL],
    )
    impl.evidence.publish(input_envelope, repo_root=ROOT)
    spec = impl._spec(input_envelope.object_id)
    spec_path = ROOT / impl.SPEC_PATH
    if spec_path.exists():
        if impl._canonical_read(spec_path) != spec:
            raise impl.ScreenError("v2 campaign spec path contains divergent bytes")
    else:
        impl._canonical_write(spec_path, spec)
    terminal = runner.run_campaign(spec_path, repo_root=ROOT)
    print(f"[daily-rates-v2] terminal={terminal.object_id}")


def _verify_completed() -> None:
    import research_campaign_v1 as runner

    terminal = runner.verify_campaign(impl.CAMPAIGN_ID, repo_root=ROOT)
    spec = impl._canonical_read(ROOT / impl.SPEC_PATH)
    impl.validate_arm_result(
        campaign=spec,
        arm=spec["ordered_arms"][0],
        native_result_path=ROOT / impl.RESULT_PATH,
    )
    print(f"[daily-rates-v2] verified terminal={terminal.object_id}")


def _self_test() -> None:
    frozen = ({"dependencies": (impl.ISSUE15_TERMINAL,)},)
    assert _thaw(frozen) == [{"dependencies": [impl.ISSUE15_TERMINAL]}]
    assert _input_payload()["ordered_artifact_paths"][-1] == V1_SCRIPT_PATH
    assert impl._spec("0" * 64)["campaign_id"].endswith(".v2")
    assert impl._spec("0" * 64)["adapter"]["path"].endswith("_v2.py")
    impl._self_test()
    print("[daily-rates-v2] recovery self-test PASS")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("self-test", "run", "verify"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    command = _parser().parse_args(argv).command
    if command == "self-test":
        _self_test()
    elif command == "run":
        _prepare_and_run()
    else:
        _verify_completed()
    return 0


preflight_arm = impl.preflight_arm
run_arm = impl.run_arm
validate_arm_result = impl.validate_arm_result
evaluate_family = impl.evaluate_family


if __name__ == "__main__":
    raise SystemExit(main())
