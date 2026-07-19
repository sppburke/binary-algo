#!/usr/bin/env python3
"""Sealed historical EURUSD 15m DOWN selector audit (campaign v1).

This is a retrospective bar-proxy methodology diagnostic.  It deliberately
cannot certify, activate, freeze, deploy, trade, or consume 2026 outcomes.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
import re
import sys
import zlib
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping, Sequence

import evidence_store as evidence


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_ID = "eurusd.m15.down.selection-audit.v1"
ARM_ID = "historical_selection_audit.v1"
SCRIPT_PATH = "scripts/eurusd_m15_down_selection_audit_v1.py"
RESULT_PATH = "results/json/eurusd_m15_down_selection_audit_v1_result.json"
SPEC_PATH = "results/json/eurusd_m15_down_selection_audit_v1_campaign_spec.json"
INPUT_SCHEMA = "research.eurusd_m15_down_selection_audit.input/v1"
RESULT_SCHEMA = "eurusd-m15-down-selection-audit-result/v1"
RECEIPT_SCHEMA = "research-campaign-adapter-receipt/v1"
DECISION_SCHEMA = "research-campaign-family-decision/v1"
PAIRS = ["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDJPY", "USDCHF", "USDCAD"]
YEARS = list(range(2012, 2026))
Q_VALUES = [10, 20, 33]
COVERAGES = [0.05, 0.10]
BREAKEVEN = 0.541
LEGACY_P10 = 0.5742
N_GROUPS = 6
SUB_FIT = 150_000
HORIZON_MINUTES = 15
GAP_SECONDS = 900
LEFT_EXCLUSION_SECONDS = 1800
RIGHT_EXCLUSION_SECONDS = 900
MIN_HALF_DOWN = 30
MIN_TEST_DOWN = 60
MIN_FORWARD_DOWN = 50
MODEL_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.03,
    "num_leaves": 127,
    "min_child_samples": 400,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.5,
    "reg_lambda": 20,
    "n_estimators": 800,
    "random_state": 7,
    "deterministic": True,
    "force_col_wise": True,
    "n_jobs": 1,
    "verbosity": -1,
}
PINNED = {
    "lightgbm": "4.6.0",
    "numpy": "2.4.6",
    "pandas": "3.0.3",
    "scikit-learn": "1.8.0",
    "pyarrow": "24.0.0",
}
CLAIM_LIMIT = (
    "Historical pre-2026 bar-proxy selector diagnostic only; not true Deriv settlement, "
    "current strategy evidence, fresh OOS, certification, activation, or deployment authority."
)
ALLOWED_INPUT_PATHS = [
    *(f"features/{pair}_{year}.parquet" for pair in PAIRS for year in YEARS),
    *(f"features_of/EURUSD_{year}.parquet" for year in YEARS),
    "scripts/m5_xpair.py",
    "scripts/harness.py",
    "scripts/m15_xpair_cpcv.py",
    "ENVIRONMENT_libs.txt",
    "results/json/m15_xpair_cpcv_result.json",
    "books/EURUSD.m15xp.v1.manifest.json",
    "books/EURUSD.m15xp.v1/m15xp_EURUSD_primary_lgb.txt",
    "books/EURUSD.m15xp.v1/m15xp_EURUSD_strategy.json",
]
RESULT_FIELDS = {
    "schema", "campaign_id", "arm_id", "input_evidence_ids", "status", "reasons",
    "protected_access", "claim_limit", "measurement", "environment", "data_diagnostics",
    "paths", "summaries", "deltas", "frozen_forward", "outcome", "diagnostic_cap",
}
PATH_FIELDS = {
    "path_index", "test_groups", "test_year_span", "fit_rows_before_cap", "fit_rows",
    "validation_rows", "test_rows", "left_exclusion_seconds", "right_exclusion_seconds",
    "fit_cap_indices", "validation_mask", "test_mask", "selectors",
}
SELECTOR_FIELDS = {"choice", "validation", "test"}
CHOICE_FIELDS = {"compression_q", "coverage", "compression_threshold", "confidence_threshold"}
TEST_FIELDS = {"down_n", "down_win_rate", "trades"}
TRADE_FIELDS = {"timestamp", "probability", "proxy_direction", "down_win"}
SUMMARY_FIELDS = {"n_paths", "mean", "p10", "min", "max", "frac_clear_breakeven"}


class AuditError(ValueError):
    """The audit contract is malformed."""


class NotComputable(RuntimeError):
    """A preregistered computability gate failed."""


def _exact(value: Any, fields: set[str], name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        actual = set(value) if isinstance(value, Mapping) else set()
        raise AuditError(
            f"{name} fields differ: missing={sorted(fields - actual)} "
            f"unknown={sorted(actual - fields)}"
        )
    return value


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise AuditError(f"{name} must be finite numeric")
    return float(value)


def _canonical_read(path: Path) -> Any:
    return evidence.decode_canonical_json(path.read_bytes())


def _canonical_write(path: Path, value: Any) -> None:
    path.write_bytes(evidence.canonical_json_bytes(value))


def _array_blob(values: Any, *, dtype: str, kind: str) -> dict[str, Any]:
    import numpy as np

    array = np.asarray(values, dtype=np.dtype(dtype))
    raw = array.tobytes(order="C")
    return {
        "schema": "audit-array-zlib/v1",
        "kind": kind,
        "dtype": np.dtype(dtype).str,
        "count": int(array.size),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "data": base64.b64encode(zlib.compress(raw, 9)).decode("ascii"),
    }


def _mask_blob(mask: Any) -> dict[str, Any]:
    import numpy as np

    array = np.asarray(mask, dtype=bool)
    packed = np.packbits(array, bitorder="little")
    raw = packed.tobytes()
    return {
        "schema": "audit-mask-zlib/v1",
        "kind": "boolean_mask",
        "count": int(array.size),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "data": base64.b64encode(zlib.compress(raw, 9)).decode("ascii"),
    }


def _validate_blob(value: Any, name: str, *, expected_count: int, mask: bool) -> Any:
    import numpy as np

    fields = {"schema", "kind", "count", "sha256", "data"}
    if not mask:
        fields.add("dtype")
    raw = _exact(value, fields, name)
    expected_schema = "audit-mask-zlib/v1" if mask else "audit-array-zlib/v1"
    if raw["schema"] != expected_schema or raw["count"] != expected_count:
        raise AuditError(f"{name} identity/count differs")
    try:
        decoded = zlib.decompress(base64.b64decode(raw["data"], validate=True))
    except Exception as exc:
        raise AuditError(f"{name} encoding differs") from exc
    if hashlib.sha256(decoded).hexdigest() != raw["sha256"]:
        raise AuditError(f"{name} digest differs")
    expected_bytes = (expected_count + 7) // 8 if mask else expected_count * 8
    if len(decoded) != expected_bytes:
        raise AuditError(f"{name} decoded size differs")
    if mask:
        return np.unpackbits(np.frombuffer(decoded, dtype="uint8"), bitorder="little")[:expected_count].astype(bool)
    if raw["dtype"] != np.dtype("int64").str:
        raise AuditError(f"{name} dtype differs")
    return np.frombuffer(decoded, dtype="int64")


def _summary(values: Sequence[float]) -> dict[str, Any]:
    import numpy as np

    a = np.asarray(values, dtype=float)
    if len(a) != 15 or not np.all(np.isfinite(a)):
        raise NotComputable("complete_15_path_summary_unavailable")
    return {
        "n_paths": 15,
        "mean": float(a.mean()),
        "p10": float(np.percentile(a, 10)),
        "min": float(a.min()),
        "max": float(a.max()),
        "frac_clear_breakeven": float((a >= BREAKEVEN).mean()),
    }


def _candidate_key(score: float, minimum_n: int, q: int, coverage: float) -> tuple[Any, ...]:
    return (score, minimum_n, coverage == 0.10, -Q_VALUES.index(q))


def _choose_candidate(candidates: Sequence[dict[str, Any]]) -> dict[str, Any]:
    eligible = [item for item in candidates if item["eligible"]]
    if not eligible:
        raise NotComputable("no_worst_half_candidate_with_30_independent_down_trades_per_half")
    return max(
        eligible,
        key=lambda item: _candidate_key(
            item["score"], item["minimum_half_n"], item["compression_q"], item["coverage"]
        ),
    )


def _outcome(worst_summary: Mapping[str, Any], *, computable: bool) -> str:
    if not computable:
        return "not_computable"
    if worst_summary["p10"] < BREAKEVEN or worst_summary["frac_clear_breakeven"] < 0.8:
        return "historical_selection_rule_falsified"
    return "historical_selection_rule_not_falsified"


def _receipt(campaign: Mapping[str, Any], arm: Mapping[str, Any], status: str, reasons: Sequence[str]) -> dict[str, Any]:
    return {
        "schema": RECEIPT_SCHEMA,
        "campaign_id": campaign["campaign_id"],
        "arm_id": arm["arm_id"],
        "native_result_contract": campaign["adapter"]["native_result_contract"],
        "status": status,
        "reasons": list(reasons),
        "native_result_path": arm["native_result_path"],
        "input_evidence_ids": list(arm["input_evidence_ids"]),
        "data_use": campaign["data_use"],
        "data_use_basis_evidence_ids": list(arm["data_use_basis_evidence_ids"]),
        "protected_access": False,
    }


def _validate_campaign(campaign: Mapping[str, Any], arm: Mapping[str, Any]) -> None:
    if (
        campaign["campaign_id"] != CAMPAIGN_ID
        or campaign["data_use"] != "retrospective"
        or campaign["key"] != {"pair": "EURUSD", "timeframe": "15m", "target": "direction", "side": "DOWN"}
        or arm["arm_id"] != ARM_ID
        or arm["native_result_path"] != RESULT_PATH
        or campaign["adapter"]["path"] != SCRIPT_PATH
        or campaign["adapter"]["native_result_contract"] != RESULT_SCHEMA
        or list(arm["input_evidence_ids"]) != list(arm["data_use_basis_evidence_ids"])
        or len(arm["input_evidence_ids"]) != 1
    ):
        raise AuditError("sealed campaign/arm identity differs")


def _validate_input_envelope(value: Mapping[str, Any], expected_id: str) -> None:
    envelope = _exact(value, {"schema", "object_id", "kind", "payload", "artifacts", "dependencies"}, "input envelope")
    if envelope["object_id"] != expected_id or envelope["kind"] != INPUT_SCHEMA or list(envelope["dependencies"]) != []:
        raise AuditError("input envelope identity differs")
    payload = _exact(
        envelope["payload"],
        {
            "schema", "key", "data_use", "years", "pair_order", "ordered_artifact_paths",
            "target_definition", "incumbent_id", "registered_proxy_claim", "excluded_year",
            "protected_access", "measurement_contract", "claim_limit", "semantic_authority",
        },
        "input payload",
    )
    paths = list(payload["ordered_artifact_paths"])
    artifact_paths = [item["path"] for item in envelope["artifacts"]]
    forbidden = [path for path in paths if re.search(r"(?:^|_)2026(?:\.|_)", path) or "m15_book_refresh_" in path]
    if (
        payload["schema"] != INPUT_SCHEMA
        or payload["data_use"] != "retrospective"
        or list(payload["years"]) != YEARS
        or list(payload["pair_order"]) != PAIRS
        or payload["excluded_year"] != 2026
        or payload["protected_access"] is not False
        or paths != ALLOWED_INPUT_PATHS
        or artifact_paths != ALLOWED_INPUT_PATHS
        or len(paths) != 120
        or forbidden
    ):
        raise AuditError("input inventory or protected boundary differs")


def preflight_arm(*, campaign: Mapping[str, Any], arm: Mapping[str, Any], input_evidence: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    _validate_campaign(campaign, arm)
    reasons: list[str] = []
    try:
        if len(input_evidence) != 1:
            raise AuditError("exactly one input envelope is required")
        _validate_input_envelope(input_evidence[0], arm["input_evidence_ids"][0])
    except AuditError as exc:
        reasons.append(str(exc))
    return {
        "campaign_id": campaign["campaign_id"],
        "arm_id": arm["arm_id"],
        "data_use": campaign["data_use"],
        "data_use_basis_evidence_ids": list(arm["data_use_basis_evidence_ids"]),
        "protected_access": False,
        "admissible": not reasons,
        "reasons": reasons,
    }


def _environment_versions() -> dict[str, str]:
    import lightgbm
    import numpy
    import pandas
    import pyarrow
    import sklearn

    actual = {
        "lightgbm": lightgbm.__version__,
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
        "scikit-learn": sklearn.__version__,
        "pyarrow": pyarrow.__version__,
    }
    if actual != PINNED:
        raise NotComputable(f"environment_versions_differ:{actual}")
    return actual


def _target_diagnostic(year: int) -> dict[str, Any]:
    import numpy as np
    import pandas as pd

    closes = {}
    for pair in PAIRS:
        frame = pd.read_parquet(ROOT / f"features/{pair}_{year}.parquet", columns=["close"])
        frame = frame[~frame.index.duplicated(keep="last")]
        closes[pair] = frame["close"]
    joined = pd.DataFrame(closes).dropna()
    seconds = joined.index.values.astype("datetime64[s]").astype("int64")
    log_close = np.log(joined["EURUSD"].to_numpy())
    contiguous = seconds[HORIZON_MINUTES:] - seconds[:-HORIZON_MINUTES] == GAP_SECONDS
    forward = log_close[HORIZON_MINUTES:] - log_close[:-HORIZON_MINUTES]
    valid = contiguous & np.isfinite(forward)
    values = forward[valid]
    moved = values[values != 0]
    up_rate = float((moved > 0).mean()) if len(moved) else float("nan")
    if not math.isfinite(up_rate) or not 0.47 <= up_rate <= 0.53:
        raise NotComputable(f"moved_up_rate_tripwire_failed:{year}:{up_rate}")
    return {
        "year": year,
        "clock_rows": int(len(joined)),
        "contiguous_targets": int(len(values)),
        "zero_returns": int((values == 0).sum()),
        "zero_return_rate": float((values == 0).mean()),
        "moved_targets": int(len(moved)),
        "moved_up_rate": up_rate,
    }


def _build_pooled(MX: Any) -> tuple[Any, Any, Any, Any, Any, list[str], list[dict[str, Any]]]:
    import numpy as np

    reference = MX.augment(MX.build_xp(["2020"]), ["2020"], "xpof")
    cols = MX.feat_cols("xpof", reference, MX.xp_cols(reference))
    if "15m_bb_width" not in reference or "sess_ny" not in reference:
        raise NotComputable("required_gate_columns_absent")
    del reference
    xs, ys, tss, widths, sessions, diagnostics = [], [], [], [], [], []
    for year in YEARS:
        diagnostics.append(_target_diagnostic(year))
        frame = MX.build_xp([str(year)], stride=2)
        frame = MX.augment(frame, [str(year)], "xpof")
        for column in cols:
            if column not in frame:
                frame[column] = np.nan
        xs.append(frame[cols].to_numpy(dtype="float32"))
        ys.append(frame["_y"].to_numpy(dtype="int8"))
        tss.append(frame["_ts"].to_numpy(dtype="int64"))
        widths.append(frame["15m_bb_width"].to_numpy(dtype="float32"))
        sessions.append(frame["sess_ny"].to_numpy(dtype="float32") > 0.5)
        del frame
    X = np.concatenate(xs)
    y = np.concatenate(ys)
    ts = np.concatenate(tss)
    bbw = np.concatenate(widths)
    ny = np.concatenate(sessions)
    order = np.argsort(ts, kind="stable")
    X, y, ts, bbw, ny = X[order], y[order], ts[order], bbw[order], ny[order]
    if len(ts) == 0 or np.any(np.diff(ts) <= 0):
        raise NotComputable("pooled_clock_not_strictly_increasing_unique")
    return X, y, ts, bbw, ny, cols, diagnostics


def _trades(ts: Any, probabilities: Any, y: Any, selected: Any) -> list[dict[str, Any]]:
    rows = []
    for index in selected:
        direction = "UP" if int(y[index]) == 1 else "DOWN"
        rows.append({
            "timestamp": int(ts[index]),
            "probability": float(probabilities[index]),
            "proxy_direction": direction,
            "down_win": bool(int(y[index]) == 0),
        })
    return rows


def _selector_test(MX: Any, choice: Mapping[str, Any], p: Any, y: Any, ts: Any, bbw: Any, ny: Any) -> dict[str, Any]:
    import numpy as np

    gate = (
        (bbw <= choice["compression_threshold"])
        & ny
        & (np.abs(p - 0.5) >= choice["confidence_threshold"])
    )
    independent = MX.nonoverlap_chrono(ts, gate, GAP_SECONDS)
    down = independent[p[independent] < 0.5]
    wins = y[down] == 0
    return {
        "down_n": int(len(down)),
        "down_win_rate": float(wins.mean()) if len(down) else None,
        "trades": _trades(ts, p, y, down),
    }


def _path_selectors(MX: Any, model: Any, fit: Any, val: Any, te: Any, y: Any, ts: Any, bbw: Any, ny: Any, X: Any) -> dict[str, Any]:
    import numpy as np

    pv = model.predict_proba(X[val])[:, 1]
    yv, tsv, bwv, nyv = y[val], ts[val], bbw[val], ny[val]
    midpoint = len(val) // 2
    halves = [np.arange(len(val)) < midpoint, np.arange(len(val)) >= midpoint]
    legacy_best = None
    worst_candidates = []
    for q in Q_VALUES:
        bthr = float(np.nanpercentile(bbw[fit], q))
        in_gate = (bwv <= bthr) & nyv
        if int(in_gate.sum()) == 0:
            continue
        for coverage in COVERAGES:
            cthr = float(np.quantile(np.abs(pv[in_gate] - 0.5), 1 - coverage))
            selected = in_gate & (np.abs(pv - 0.5) >= cthr)
            if int(selected.sum()) == 0:
                continue
            legacy_score = float(((pv[selected] > 0.5).astype(int) == yv[selected]).mean())
            choice = {
                "compression_q": q,
                "coverage": coverage,
                "compression_threshold": bthr,
                "confidence_threshold": cthr,
            }
            if legacy_best is None or legacy_score > legacy_best[0]:
                legacy_best = (legacy_score, choice)
            half_rates, half_ns = [], []
            for half in halves:
                mask = selected & half & (pv < 0.5)
                indices = MX.nonoverlap_chrono(tsv, mask, GAP_SECONDS)
                half_ns.append(int(len(indices)))
                half_rates.append(float((yv[indices] == 0).mean()) if len(indices) else None)
            eligible = min(half_ns) >= MIN_HALF_DOWN and all(rate is not None for rate in half_rates)
            worst_candidates.append({
                **choice,
                "eligible": eligible,
                "score": min(half_rates) if eligible else -1.0,
                "minimum_half_n": min(half_ns),
                "half_down_n": half_ns,
                "half_down_win_rate": half_rates,
            })
    if legacy_best is None:
        raise NotComputable("legacy_selector_has_no_candidate")
    worst = _choose_candidate(worst_candidates)
    legacy_choice = legacy_best[1]
    pt = model.predict_proba(X[te])[:, 1]
    selectors = {}
    for name, choice, validation in (
        ("legacy_full_val", legacy_choice, {"combined_accuracy": legacy_best[0]}),
        (
            "worst_val_half",
            {key: worst[key] for key in CHOICE_FIELDS},
            {
                "worst_down_win_rate": worst["score"],
                "minimum_half_n": worst["minimum_half_n"],
                "half_down_n": worst["half_down_n"],
                "half_down_win_rate": worst["half_down_win_rate"],
            },
        ),
    ):
        test = _selector_test(MX, choice, pt, y[te], ts[te], bbw[te], ny[te])
        selectors[name] = {"choice": dict(choice), "validation": validation, "test": test}
    if selectors["worst_val_half"]["test"]["down_n"] < MIN_TEST_DOWN:
        raise NotComputable("worst_half_test_path_has_fewer_than_60_independent_down_trades")
    return selectors


def _run_cpcv(MX: Any) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]], list[str]]:
    import lightgbm as lgb
    import numpy as np

    X, y, ts, bbw, ny, cols, diagnostics = _build_pooled(MX)
    n = len(y)
    edges = np.linspace(0, n, N_GROUPS + 1).astype(int)
    groups = [(edges[index], edges[index + 1]) for index in range(N_GROUPS)]
    paths = []
    for path_index, test_groups in enumerate(combinations(range(N_GROUPS), 2)):
        te_mask = np.zeros(n, dtype=bool)
        for group in test_groups:
            start, stop = groups[group]
            te_mask[start:stop] = True
        tr_mask = ~te_mask
        for group in test_groups:
            start, stop = groups[group]
            low, high = ts[start], ts[stop - 1]
            tr_mask &= ~(
                (ts >= low - LEFT_EXCLUSION_SECONDS)
                & (ts <= high + RIGHT_EXCLUSION_SECONDS)
            )
        tri = np.flatnonzero(tr_mask)
        if len(tri) < 5000:
            raise NotComputable(f"path_{path_index}_purged_training_too_small")
        cut_index = tri[int(len(tri) * 0.8)]
        cut_time = ts[cut_index]
        fit_full = tri[ts[tri] < cut_time]
        val = tri[ts[tri] >= cut_time]
        fit = fit_full
        if len(fit) > SUB_FIT:
            fit = fit[np.linspace(0, len(fit) - 1, SUB_FIT).astype(int)]
        if len(fit) != len(np.unique(fit)) or np.any(np.diff(fit) <= 0):
            raise NotComputable(f"path_{path_index}_fit_cap_indices_not_unique_ordered")
        model = lgb.LGBMClassifier(**MODEL_PARAMS)
        model.fit(X[fit], y[fit])
        te = np.flatnonzero(te_mask)
        selectors = _path_selectors(MX, model, fit, val, te, y, ts, bbw, ny, X)
        path = {
            "path_index": path_index,
            "test_groups": list(test_groups),
            "test_year_span": [
                int(str(np.datetime64(int(ts[te[0]]), "s"))[:4]),
                int(str(np.datetime64(int(ts[te[-1]]), "s"))[:4]),
            ],
            "fit_rows_before_cap": int(len(fit_full)),
            "fit_rows": int(len(fit)),
            "validation_rows": int(len(val)),
            "test_rows": int(len(te)),
            "left_exclusion_seconds": LEFT_EXCLUSION_SECONDS,
            "right_exclusion_seconds": RIGHT_EXCLUSION_SECONDS,
            "fit_cap_indices": _array_blob(fit, dtype="int64", kind="global_row_indices"),
            "validation_mask": _mask_blob(np.isin(np.arange(n), val, assume_unique=True)),
            "test_mask": _mask_blob(te_mask),
            "selectors": selectors,
        }
        paths.append(path)
        print(
            f"[audit] path {path_index + 1:02d}/15 groups={test_groups} "
            f"legacy={selectors['legacy_full_val']['test']['down_win_rate']:.4f} "
            f"worst={selectors['worst_val_half']['test']['down_win_rate']:.4f}",
            flush=True,
        )
        del model
    summaries = {
        selector: _summary([path["selectors"][selector]["test"]["down_win_rate"] for path in paths])
        for selector in ("legacy_full_val", "worst_val_half")
    }
    return paths, summaries, diagnostics, cols


def _build_year(MX: Any, year: int, cols: Sequence[str]) -> tuple[Any, Any, Any, Any, Any]:
    import numpy as np

    frame = MX.augment(MX.build_xp([str(year)]), [str(year)], "xpof")
    for column in cols:
        if column not in frame:
            frame[column] = np.nan
    return (
        frame[list(cols)].to_numpy(dtype="float32"),
        frame["_y"].to_numpy(dtype="int8"),
        frame["_ts"].to_numpy(dtype="int64"),
        frame["15m_bb_width"].to_numpy(dtype="float32"),
        frame["sess_ny"].to_numpy(dtype="float32") > 0.5,
    )


def _bootstrap(wins: Any) -> list[float]:
    import numpy as np

    values = np.asarray(wins, dtype=float)
    rng = np.random.default_rng(7)
    samples = np.array([values[rng.integers(0, len(values), len(values))].mean() for _ in range(5000)])
    return [float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))]


def _frozen_forward(MX: Any, cols: Sequence[str], diagnostics: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    import lightgbm as lgb
    import numpy as np

    strategy = json.loads((ROOT / "books/EURUSD.m15xp.v1/m15xp_EURUSD_strategy.json").read_bytes())
    if {key: strategy[key] for key in ("pair", "mode", "MX_HOR")} != {"pair": "EURUSD", "mode": "xpof", "MX_HOR": 15}:
        raise NotComputable("frozen_strategy_identity_differs")
    primary = list(strategy["primary_feats"])
    if primary != list(cols):
        raise NotComputable("frozen_primary_feature_order_differs")
    booster = lgb.Booster(model_file=str(ROOT / "books/EURUSD.m15xp.v1/m15xp_EURUSD_primary_lgb.txt"))
    val_parts = []
    for year in (2022, 2023):
        X, y, ts, bbw, ny = _build_year(MX, year, primary)
        val_parts.append((booster.predict(X), y, ts, bbw, ny))
        del X
    pv = np.concatenate([part[0] for part in val_parts])
    yv = np.concatenate([part[1] for part in val_parts])
    tsv = np.concatenate([part[2] for part in val_parts])
    bwv = np.concatenate([part[3] for part in val_parts])
    nyv = np.concatenate([part[4] for part in val_parts])
    midpoint = len(tsv) // 2
    halves = [np.arange(len(tsv)) < midpoint, np.arange(len(tsv)) >= midpoint]
    candidates = []
    for q in Q_VALUES:
        bthr = float(np.nanpercentile(bwv, q))
        in_gate = (bwv <= bthr) & nyv
        for coverage in COVERAGES:
            cthr = float(np.quantile(np.abs(pv[in_gate] - 0.5), 1 - coverage))
            selected = in_gate & (np.abs(pv - 0.5) >= cthr)
            rates, ns = [], []
            for half in halves:
                indices = MX.nonoverlap_chrono(tsv, selected & half & (pv < 0.5), GAP_SECONDS)
                ns.append(int(len(indices)))
                rates.append(float((yv[indices] == 0).mean()) if len(indices) else None)
            eligible = min(ns) >= MIN_HALF_DOWN and all(rate is not None for rate in rates)
            candidates.append({
                "compression_q": q, "coverage": coverage,
                "compression_threshold": bthr, "confidence_threshold": cthr,
                "eligible": eligible, "score": min(rates) if eligible else -1.0,
                "minimum_half_n": min(ns), "half_down_n": ns, "half_down_win_rate": rates,
            })
    selected = _choose_candidate(candidates)
    choice = {key: selected[key] for key in CHOICE_FIELDS}
    years = {}
    diag_by_year = {item["year"]: item for item in diagnostics}
    for year in (2024, 2025):
        X, y, ts, bbw, ny = _build_year(MX, year, primary)
        prediction = booster.predict(X)
        test = _selector_test(MX, choice, prediction, y, ts, bbw, ny)
        if test["down_n"] < MIN_FORWARD_DOWN:
            raise NotComputable(f"frozen_forward_{year}_has_fewer_than_50_down_trades")
        wins = [row["down_win"] for row in test["trades"]]
        years[str(year)] = {
            **test,
            "bootstrap_ci95": _bootstrap(wins),
            "zero_return_rate": diag_by_year[year]["zero_return_rate"],
            "moved_up_rate": diag_by_year[year]["moved_up_rate"],
        }
        del X
    return {
        "book": "EURUSD.m15xp.v1",
        "selection_years": [2022, 2023],
        "selector": "worst_val_half_down",
        "choice": choice,
        "validation": {
            "worst_down_win_rate": selected["score"],
            "minimum_half_n": selected["minimum_half_n"],
            "half_down_n": selected["half_down_n"],
            "half_down_win_rate": selected["half_down_win_rate"],
        },
        "years": years,
    }


def _not_computable_result(campaign: Mapping[str, Any], arm: Mapping[str, Any], reason: str, environment: Mapping[str, str] | None = None) -> dict[str, Any]:
    return {
        "schema": RESULT_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "arm_id": ARM_ID,
        "input_evidence_ids": list(arm["input_evidence_ids"]),
        "status": "not_computable",
        "reasons": [reason],
        "protected_access": False,
        "claim_limit": CLAIM_LIMIT,
        "measurement": _measurement_contract(),
        "environment": dict(environment or {}),
        "data_diagnostics": [],
        "paths": [],
        "summaries": {},
        "deltas": {},
        "frozen_forward": {},
        "outcome": "not_computable",
        "diagnostic_cap": None,
    }


def _measurement_contract() -> dict[str, Any]:
    return {
        "years": YEARS,
        "pair_order": PAIRS,
        "horizon_minutes": HORIZON_MINUTES,
        "target": "contiguous_900_second_EURUSD_bar_close_direction_moved_only",
        "side": "DOWN_probability_below_0.5",
        "groups": 6,
        "test_groups_per_path": 2,
        "paths": 15,
        "left_exclusion_seconds": LEFT_EXCLUSION_SECONDS,
        "right_exclusion_seconds": RIGHT_EXCLUSION_SECONDS,
        "fit_fraction": 0.8,
        "sub_fit": SUB_FIT,
        "candidate_order": [[q, coverage] for q in Q_VALUES for coverage in COVERAGES],
        "session_gate": "sess_ny>0.5",
        "breakeven": BREAKEVEN,
        "minimum_validation_half_down": MIN_HALF_DOWN,
        "minimum_test_down": MIN_TEST_DOWN,
        "model_parameters": MODEL_PARAMS,
        "protected_year_excluded": 2026,
    }


def run_arm(*, campaign: Mapping[str, Any], arm: Mapping[str, Any], temporary_output_path: Path) -> dict[str, Any]:
    _validate_campaign(campaign, arm)
    if temporary_output_path.exists() or temporary_output_path.is_symlink():
        raise AuditError("temporary output must not pre-exist")
    environment: dict[str, str] = {}
    try:
        input_object = evidence.verify_object(arm["input_evidence_ids"][0], repo_root=ROOT)
        _validate_input_envelope(input_object.as_dict(), arm["input_evidence_ids"][0])
        environment = _environment_versions()
        os.environ["MX_HOR"] = "15"
        if os.environ.get("MX_HOR") != "15" or "m5_xpair" in sys.modules:
            raise NotComputable("MX_HOR_or_import_order_differs")
        import m5_xpair as MX

        if MX.HOR != 15 or MX.GAP_S != 900:
            raise NotComputable("scientific_import_horizon_differs")
        paths, summaries, diagnostics, cols = _run_cpcv(MX)
        frozen = _frozen_forward(MX, cols, diagnostics)
        worst = summaries["worst_val_half"]
        outcome = _outcome(worst, computable=True)
        result = {
            "schema": RESULT_SCHEMA,
            "campaign_id": CAMPAIGN_ID,
            "arm_id": ARM_ID,
            "input_evidence_ids": list(arm["input_evidence_ids"]),
            "status": "completed",
            "reasons": [],
            "protected_access": False,
            "claim_limit": CLAIM_LIMIT,
            "measurement": _measurement_contract(),
            "environment": environment,
            "data_diagnostics": diagnostics,
            "paths": paths,
            "summaries": summaries,
            "deltas": {
                key: summaries["worst_val_half"][key] - summaries["legacy_full_val"][key]
                for key in ("mean", "p10", "min", "max", "frac_clear_breakeven")
            },
            "frozen_forward": frozen,
            "outcome": outcome,
            "diagnostic_cap": min(worst["p10"], LEGACY_P10) if outcome == "historical_selection_rule_not_falsified" else None,
        }
    except NotComputable as exc:
        result = _not_computable_result(campaign, arm, str(exc), environment)
    _canonical_write(temporary_output_path, result)
    return validate_arm_result(campaign=campaign, arm=arm, native_result_path=temporary_output_path)


def _validate_trade_rows(test: Mapping[str, Any], name: str) -> float | None:
    raw = _exact(test, TEST_FIELDS, name)
    if not isinstance(raw["trades"], list) or raw["down_n"] != len(raw["trades"]):
        raise AuditError(f"{name} trade count differs")
    wins = []
    previous = None
    for index, item in enumerate(raw["trades"]):
        trade = _exact(item, TRADE_FIELDS, f"{name}.trades[{index}]")
        timestamp = trade["timestamp"]
        probability = _finite(trade["probability"], "trade probability")
        if not isinstance(timestamp, int) or probability >= 0.5 or trade["proxy_direction"] not in {"UP", "DOWN"}:
            raise AuditError(f"{name} trade semantics differ")
        if previous is not None and timestamp - previous < GAP_SECONDS:
            raise AuditError(f"{name} trades overlap")
        previous = timestamp
        expected = trade["proxy_direction"] == "DOWN"
        if trade["down_win"] is not expected:
            raise AuditError(f"{name} down outcome differs")
        wins.append(expected)
    recomputed = sum(wins) / len(wins) if wins else None
    if recomputed != raw["down_win_rate"]:
        raise AuditError(f"{name} win rate differs from trades")
    return recomputed


def _validate_result(value: Any, campaign: Mapping[str, Any], arm: Mapping[str, Any]) -> dict[str, Any]:
    _validate_campaign(campaign, arm)
    raw = _exact(value, RESULT_FIELDS, "native result")
    if (
        raw["schema"] != RESULT_SCHEMA
        or raw["campaign_id"] != CAMPAIGN_ID
        or raw["arm_id"] != ARM_ID
        or raw["input_evidence_ids"] != list(arm["input_evidence_ids"])
        or raw["protected_access"] is not False
        or raw["claim_limit"] != CLAIM_LIMIT
        or raw["measurement"] != _measurement_contract()
        or raw["environment"] not in ({}, PINNED)
    ):
        raise AuditError("native result identity, authority, or measurement differs")
    if raw["status"] == "not_computable":
        if (
            not isinstance(raw["reasons"], list) or not raw["reasons"]
            or raw["outcome"] != "not_computable" or raw["paths"] != []
            or raw["summaries"] != {} or raw["deltas"] != {}
            or raw["frozen_forward"] != {} or raw["diagnostic_cap"] is not None
        ):
            raise AuditError("not-computable result semantics differ")
        return dict(raw)
    if raw["status"] != "completed" or raw["reasons"] != [] or raw["environment"] != PINNED:
        raise AuditError("completed result status differs")
    expected_groups = list(combinations(range(6), 2))
    if not isinstance(raw["paths"], list) or len(raw["paths"]) != 15:
        raise AuditError("native result must contain all 15 paths")
    rates = {"legacy_full_val": [], "worst_val_half": []}
    total_rows = None
    for index, item in enumerate(raw["paths"]):
        path = _exact(item, PATH_FIELDS, f"paths[{index}]")
        if path["path_index"] != index or tuple(path["test_groups"]) != expected_groups[index]:
            raise AuditError("path order/identity differs")
        if path["left_exclusion_seconds"] != 1800 or path["right_exclusion_seconds"] != 900:
            raise AuditError("path exclusion geometry differs")
        inferred_total = path["validation_mask"].get("count")
        if not isinstance(inferred_total, int) or inferred_total <= 0:
            raise AuditError("path mask count differs")
        if total_rows is None:
            total_rows = inferred_total
        elif inferred_total != total_rows or path["test_mask"].get("count") != total_rows:
            raise AuditError("path mask domains differ")
        fit_indices = _validate_blob(path["fit_cap_indices"], "fit cap", expected_count=path["fit_rows"], mask=False)
        validation_mask = _validate_blob(path["validation_mask"], "validation mask", expected_count=total_rows, mask=True)
        test_mask = _validate_blob(path["test_mask"], "test mask", expected_count=total_rows, mask=True)
        if (
            len(set(fit_indices.tolist())) != len(fit_indices)
            or (len(fit_indices) > 1 and not all(fit_indices[pos] < fit_indices[pos + 1] for pos in range(len(fit_indices) - 1)))
            or int(validation_mask.sum()) != path["validation_rows"]
            or int(test_mask.sum()) != path["test_rows"]
            or bool((validation_mask & test_mask).any())
        ):
            raise AuditError("path masks or fit-cap indices differ")
        selectors = _exact(path["selectors"], {"legacy_full_val", "worst_val_half"}, "selectors")
        for selector in rates:
            selected = _exact(selectors[selector], SELECTOR_FIELDS, selector)
            choice = _exact(selected["choice"], CHOICE_FIELDS, f"{selector}.choice")
            if choice["compression_q"] not in Q_VALUES or choice["coverage"] not in COVERAGES:
                raise AuditError("selector candidate outside sealed family")
            rate = _validate_trade_rows(selected["test"], f"{selector}.test")
            if rate is None:
                raise AuditError("selector test is empty")
            rates[selector].append(rate)
        if selectors["worst_val_half"]["test"]["down_n"] < MIN_TEST_DOWN:
            raise AuditError("worst-half path power differs")
    recomputed = {selector: _summary(values) for selector, values in rates.items()}
    for selector, expected in recomputed.items():
        observed = _exact(raw["summaries"][selector], SUMMARY_FIELDS, f"summaries.{selector}")
        if any(not math.isclose(_finite(observed[key], key), expected[key], rel_tol=0, abs_tol=1e-15) for key in SUMMARY_FIELDS if key != "n_paths") or observed["n_paths"] != 15:
            raise AuditError(f"{selector} summary differs from trades")
    expected_outcome = _outcome(recomputed["worst_val_half"], computable=True)
    expected_cap = min(recomputed["worst_val_half"]["p10"], LEGACY_P10) if expected_outcome == "historical_selection_rule_not_falsified" else None
    if raw["outcome"] != expected_outcome or raw["diagnostic_cap"] != expected_cap:
        raise AuditError("historical falsifier outcome differs")
    frozen = raw["frozen_forward"]
    if frozen.get("selection_years") != [2022, 2023] or set(frozen.get("years", {})) != {"2024", "2025"}:
        raise AuditError("frozen-forward years differ")
    for year in ("2024", "2025"):
        test = frozen["years"][year]
        core = {key: test[key] for key in TEST_FIELDS}
        _validate_trade_rows(core, f"frozen_forward.{year}")
        if test["down_n"] < MIN_FORWARD_DOWN or len(test["bootstrap_ci95"]) != 2:
            raise AuditError("frozen-forward power/CI differs")
    evidence.canonical_json_bytes(raw)
    return dict(raw)


def validate_arm_result(*, campaign: Mapping[str, Any], arm: Mapping[str, Any], native_result_path: Path) -> dict[str, Any]:
    if native_result_path.is_symlink() or not native_result_path.is_file():
        raise AuditError("native result must be a regular non-symlink file")
    value = _canonical_read(native_result_path)
    result = _validate_result(value, campaign, arm)
    reasons = result["reasons"] if result["status"] == "not_computable" else []
    return _receipt(campaign, arm, result["status"], reasons)


def evaluate_family(*, campaign: Mapping[str, Any], receipts: Sequence[Mapping[str, Any]], native_results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(receipts) != 1 or len(native_results) != 1:
        raise AuditError("one complete arm result is required")
    arm = campaign["ordered_arms"][0]
    reference = evidence.ArtifactRef.from_dict(dict(native_results[0]))
    if reference.path != RESULT_PATH:
        raise AuditError("native result reference path differs")
    value = evidence.decode_canonical_json(reference.read_verified(repo_root=ROOT))
    result = _validate_result(value, campaign, arm)
    if result["status"] == "not_computable":
        verdict_status, verdict_reasons = "not_computable", list(result["reasons"])
        decision, candidates, reasons = "capability_deferred", [], list(result["reasons"])
    elif result["outcome"] == "historical_selection_rule_falsified":
        verdict_status = "killed"
        verdict_reasons = ["historical_selection_rule_falsified"]
        decision, candidates, reasons = "no_candidate", [], ["historical_selection_rule_falsified"]
    else:
        verdict_status, verdict_reasons = "candidate", []
        decision, candidates, reasons = "inactive_candidate", [ARM_ID], []
    return {
        "schema": DECISION_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "verdicts": [{"arm_id": ARM_ID, "status": verdict_status, "reasons": verdict_reasons}],
        "ordered_candidates": candidates,
        "decision": decision,
        "reasons": reasons,
    }


def _input_payload() -> dict[str, Any]:
    return {
        "schema": INPUT_SCHEMA,
        "key": {"pair": "EURUSD", "timeframe": "15m", "target": "direction", "side": "DOWN"},
        "data_use": "retrospective",
        "years": YEARS,
        "pair_order": PAIRS,
        "ordered_artifact_paths": ALLOWED_INPUT_PATHS,
        "target_definition": "EURUSD close[t+15m]-close[t] on exact 900-second contiguous cross-pair clock; moved bars only",
        "incumbent_id": "EURUSD.m15xp.v1",
        "registered_proxy_claim": {"metric": "DOWN.refit_CPCV.p10", "value": LEGACY_P10, "authority": "provenance_only"},
        "excluded_year": 2026,
        "protected_access": False,
        "measurement_contract": _measurement_contract(),
        "claim_limit": CLAIM_LIMIT,
        "semantic_authority": {
            "legacy_result": ["DOWN.p10", "provenance_identity"],
            "manifest": ["artifact_identities"],
            "strategy": ["pair", "mode", "MX_HOR", "primary_feats"],
        },
    }


def _spec(input_id: str) -> dict[str, Any]:
    return {
        "schema": "research-campaign-spec-input/v1",
        "campaign_id": CAMPAIGN_ID,
        "key": {"pair": "EURUSD", "timeframe": "15m", "target": "direction", "side": "DOWN"},
        "question": "Does worst-validation-half selection preserve the historical pre-2026 EURUSD 15m DOWN bar-proxy threshold?",
        "hypothesis": "The historical DOWN selection rule remains at or above 0.541 under worst-validation-half selection.",
        "falsifier": "Worst-half DOWN path p10 below 0.541 or fewer than 12 of 15 paths clearing 0.541.",
        "data_use": "retrospective",
        "ordered_arms": [{
            "arm_id": ARM_ID,
            "role": "candidate",
            "native_result_path": RESULT_PATH,
            "input_evidence_ids": [input_id],
            "data_use_basis_evidence_ids": [input_id],
        }],
        "adapter": {
            "path": SCRIPT_PATH,
            "run_function": "run_arm",
            "preflight_function": "preflight_arm",
            "validate_function": "validate_arm_result",
            "evaluate_function": "evaluate_family",
            "native_result_contract": RESULT_SCHEMA,
        },
        "stopping_rule": "Run the exact one-arm family once; stop on completion or any preregistered not-computable gate.",
        "terminal_rule": {
            "allowed_decisions": ["no_candidate", "inactive_candidate", "capability_deferred"],
            "activation": False,
        },
    }


def _prepare_and_run() -> None:
    import research_campaign_v1 as campaign_runner

    if (ROOT / RESULT_PATH).exists():
        return_terminal = campaign_runner.run_campaign(ROOT / SPEC_PATH, repo_root=ROOT)
        print(f"[audit] resumed terminal={return_terminal.object_id}")
        return
    artifacts = [evidence.ArtifactRef.capture(path, repo_root=ROOT) for path in ALLOWED_INPUT_PATHS]
    input_envelope = evidence.EvidenceEnvelope.create(
        kind=INPUT_SCHEMA,
        payload=_input_payload(),
        artifacts=artifacts,
    )
    evidence.publish(input_envelope, repo_root=ROOT)
    spec = _spec(input_envelope.object_id)
    if (ROOT / SPEC_PATH).exists():
        if _canonical_read(ROOT / SPEC_PATH) != spec:
            raise AuditError("campaign spec path contains divergent bytes")
    else:
        _canonical_write(ROOT / SPEC_PATH, spec)
    terminal = campaign_runner.run_campaign(ROOT / SPEC_PATH, repo_root=ROOT)
    print(f"[audit] terminal={terminal.object_id}")


def _verify_completed() -> None:
    import research_campaign_v1 as campaign_runner

    terminal = campaign_runner.verify_campaign(CAMPAIGN_ID, repo_root=ROOT)
    spec = _canonical_read(ROOT / SPEC_PATH)
    arm = spec["ordered_arms"][0]
    validate_arm_result(campaign=spec, arm=arm, native_result_path=ROOT / RESULT_PATH)
    print(f"[audit] verified terminal={terminal.object_id}")


def _self_test() -> None:
    synthetic = [
        {"compression_q": 10, "coverage": 0.05, "eligible": True, "score": 0.6, "minimum_half_n": 40},
        {"compression_q": 20, "coverage": 0.10, "eligible": True, "score": 0.6, "minimum_half_n": 40},
    ]
    assert _choose_candidate(synthetic)["coverage"] == 0.10
    assert _outcome({"p10": 0.55, "frac_clear_breakeven": 0.8}, computable=True) == "historical_selection_rule_not_falsified"
    assert _outcome({"p10": 0.5409, "frac_clear_breakeven": 1.0}, computable=True) == "historical_selection_rule_falsified"
    assert _outcome({}, computable=False) == "not_computable"
    assert not any("2026" in path for path in ALLOWED_INPUT_PATHS)
    assert _measurement_contract()["protected_year_excluded"] == 2026
    assert _spec("0" * 64)["terminal_rule"]["activation"] is False
    invalid = _not_computable_result(_spec("0" * 64), _spec("0" * 64)["ordered_arms"][0], "synthetic")
    invalid["unknown"] = True
    try:
        _exact(invalid, RESULT_FIELDS, "synthetic result")
    except AuditError:
        pass
    else:
        raise AssertionError("strict result schema accepted an unknown field")
    sample = [0.54 + index / 1000 for index in range(15)]
    summary = _summary(sample)
    assert math.isclose(summary["p10"], __import__("numpy").percentile(sample, 10))
    print("[audit] self-test PASS")


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


if __name__ == "__main__":
    raise SystemExit(main())
