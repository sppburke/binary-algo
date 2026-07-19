#!/usr/bin/env python3
"""Sealed current-vintage daily-rates screen for EURUSD 15m DOWN.

The experiment is a retrospective procurement diagnostic.  It cannot establish
an intraday rates lead, certify a strategy, activate anything, or buy data.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import io
import math
import subprocess
import sys
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence
from zoneinfo import ZoneInfo

import evidence_store as evidence


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_ID = "eurusd.m15.down.daily-rates-screen.v1"
ARM_ID = "daily_rates_regime_screen.v1"
SCRIPT_PATH = "scripts/eurusd_m15_down_daily_rates_screen_v1.py"
SNAPSHOT_PATH = "results/json/eurusd_m15_down_daily_rates_source_snapshot_v1.json"
SPEC_PATH = "results/json/eurusd_m15_down_daily_rates_screen_v1_campaign_spec.json"
RESULT_PATH = "results/json/eurusd_m15_down_daily_rates_screen_v1_result.json"
SELECTION_PATH = "results/json/eurusd_m15_down_selection_audit_v1_result.json"
SELECTION_SPEC_PATH = "results/json/eurusd_m15_down_selection_audit_v1_campaign_spec.json"
SELECTION_SHA256 = "819f22899a2b83983df2b76be04339c096cb74500addcbc1a6208692fbe4fdef"
ISSUE15_TERMINAL = "4c3b0d20013b9b6301f965ac45aae9161fa6db4fc8e5bf7f1c123f2fd0b76b8d"
INPUT_SCHEMA = "research.eurusd_m15_down_daily_rates.input/v1"
SNAPSHOT_SCHEMA = "eurusd-m15-down-daily-rates-source-snapshot/v1"
RESULT_SCHEMA = "eurusd-m15-down-daily-rates-screen-result/v1"
RECEIPT_SCHEMA = "research-campaign-adapter-receipt/v1"
DECISION_SCHEMA = "research-campaign-family-decision/v1"
NY = ZoneInfo("America/New_York")
FRED_SERIES = "DGS2"
ECB_SERIES = "YC.B.U2.EUR.4F.G_N_C.SV_C_YM.PY_2Y"
START_DATE = "2012-01-01"
END_DATE = "2025-12-31"
WINDOW = 5
MIN_PATH_DATES = 20
MIN_PATH_TRADES = 60
MIN_FROZEN_DATES = 20
MIN_FROZEN_TRADES = 50
MIN_P10_LIFT = 0.005
MIN_POSITIVE_PATHS = 12
CLAIM_LIMIT = (
    "Calendar-lagged current-vintage retrospective association and procurement diagnostic only; "
    "not point-in-time intraday evidence, strategy certification, activation, or purchase authority."
)

FRED_URL = (
    "https://fred.stlouisfed.org/graph/fredgraph.csv?"
    f"id={FRED_SERIES}&cosd={START_DATE}&coed={END_DATE}"
)
ECB_URL = (
    "https://data-api.ecb.europa.eu/service/data/YC/"
    "B.U2.EUR.4F.G_N_C.SV_C_YM.PY_2Y?"
    f"startPeriod={START_DATE}&endPeriod={END_DATE}&format=csvdata"
)
SOURCE_URLS = {
    "fred_dgs2": FRED_URL,
    "ecb_euro_area_2y_par": ECB_URL,
    "databento_glbx": "https://databento.com/datasets/GLBX.MDP3",
    "databento_xeur": "https://databento.com/datasets/XEUR.EOBI",
    "databento_zt_catalog": "https://databento.com/catalog/cme/GLBX.MDP3/futures/ZT",
    "databento_fgbs_catalog": "https://databento.com/catalog/xeur/XEUR.EOBI/futures/FGBS",
    "databento_pricing": "https://databento.com/pricing",
    "eurex_fgbs": (
        "https://www.eurex.com/ex-en/markets/int/long-term-interest-rates/fix/"
        "government-bonds/Euro-Schatz-Futures-137256"
    ),
    "deutsche_boerse_marketplace": (
        "https://console.marketplace.deutsche-boerse.com/products/"
        "934b93d2-4d35-4cf4-b700-a5e2b9b4333c"
    ),
    "deutsche_boerse_product_sheet": (
        "https://www.mds.deutsche-boerse.com/resource/blob/1335658/"
        "ce521aadb5904de6dce017e2afae8081/data/"
        "Product%20sheet_Eurex%20Order%20by%20Order%20Historical%20Data.pdf"
    ),
    "deutsche_boerse_terms": (
        "https://console.marketplace.deutsche-boerse.com/assets/GTC_Marketplace.pdf"
    ),
    "cme_datamine": "https://www.cmegroup.com/datamine.html",
}

SNAPSHOT_FIELDS = {
    "schema", "retrieved_at_utc", "date_bounds", "rate_sources", "vendor_sources",
    "checked", "unknown", "needed",
}
RESULT_FIELDS = {
    "schema", "campaign_id", "arm_id", "input_evidence_ids", "status", "reasons",
    "protected_access", "claim_limit", "measurement", "source", "paths", "summaries",
    "frozen_forward", "legacy_diagnostic", "outcome", "procurement", "evidence",
}
CELL_FIELDS = {"date_n", "trade_n", "equal_date_down_win_rate"}
VIEW_FIELDS = {
    "favorable", "adverse", "excluded_trade_n", "excluded_date_n", "lift",
}
SUMMARY_FIELDS = {"n_paths", "mean", "p10", "min", "max", "fraction_positive"}


class ScreenError(ValueError):
    """The sealed screen contract or bytes differ."""


class NotComputable(RuntimeError):
    """A preregistered computability gate failed."""


def _exact(value: Any, fields: set[str], name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        actual = set(value) if isinstance(value, Mapping) else set()
        raise ScreenError(
            f"{name} fields differ: missing={sorted(fields - actual)} "
            f"unknown={sorted(actual - fields)}"
        )
    return value


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ScreenError(f"{name} must be finite numeric")
    return float(value)


def _canonical_read(path: Path) -> Any:
    return evidence.decode_canonical_json(path.read_bytes())


def _canonical_write(path: Path, value: Any) -> None:
    path.write_bytes(evidence.canonical_json_bytes(value))


def _fetch(url: str) -> tuple[bytes, int, str]:
    import requests

    response = requests.get(
        url,
        timeout=60,
        headers={"User-Agent": "binary-algo-research/1.0 (+private internal research)"},
    )
    return response.content, int(response.status_code), response.headers.get("content-type", "")


def _source_record(source_id: str, url: str, body: bytes, status: int, content_type: str) -> dict[str, Any]:
    return {
        "source_id": source_id,
        "url": url,
        "status": status,
        "bytes": len(body),
        "sha256": hashlib.sha256(body).hexdigest(),
        "content_type": content_type,
    }


def _parse_fred(body: bytes) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    try:
        rows = list(csv.DictReader(io.StringIO(body.decode("utf-8-sig"))))
    except (UnicodeDecodeError, csv.Error) as exc:
        raise ScreenError(f"FRED CSV parse failed: {exc}") from exc
    if not rows or set(rows[0]) != {"observation_date", FRED_SERIES}:
        raise ScreenError("FRED series/schema differs")
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    missing = 0
    for row in rows:
        label = row["observation_date"]
        date.fromisoformat(label)
        if label in seen:
            raise ScreenError("FRED contains duplicate date")
        seen.add(label)
        raw = row[FRED_SERIES].strip()
        value = None if raw in {"", "."} else float(raw)
        if value is not None and not math.isfinite(value):
            raise ScreenError("FRED contains non-finite value")
        missing += value is None
        normalized.append({"date": label, "value": value})
    if normalized[0]["date"] < START_DATE or normalized[-1]["date"] > END_DATE or len(normalized) < 3600:
        raise ScreenError("FRED date bounds/coverage differ")
    return normalized, {
        "series": FRED_SERIES,
        "frequency": "daily_business_observations_with_calendar_rows",
        "unit": "percent",
        "row_n": len(normalized),
        "non_missing_n": len(normalized) - missing,
        "missing_n": missing,
        "first_date": normalized[0]["date"],
        "last_date": normalized[-1]["date"],
        "current_vintage": True,
        "availability_note": "H.15 is posted at 16:15 America/New_York; revisions are not reconstructed.",
    }


def _parse_ecb(body: bytes) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    try:
        rows = list(csv.DictReader(io.StringIO(body.decode("utf-8-sig"))))
    except (UnicodeDecodeError, csv.Error) as exc:
        raise ScreenError(f"ECB CSV parse failed: {exc}") from exc
    required = {
        "KEY", "FREQ", "REF_AREA", "CURRENCY", "INSTRUMENT_FM", "DATA_TYPE_FM",
        "TIME_PERIOD", "OBS_VALUE", "OBS_STATUS", "TITLE", "TITLE_COMPL", "UNIT",
    }
    if not rows or not required.issubset(rows[0]):
        raise ScreenError("ECB schema differs")
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        if (
            row["KEY"] != ECB_SERIES or row["FREQ"] != "B" or row["REF_AREA"] != "U2"
            or row["CURRENCY"] != "EUR" or row["INSTRUMENT_FM"] != "G_N_C"
            or row["DATA_TYPE_FM"] != "PY_2Y" or row["UNIT"] != "PCPA"
            or row["OBS_STATUS"] != "A"
        ):
            raise ScreenError("ECB series metadata differs")
        label = row["TIME_PERIOD"]
        date.fromisoformat(label)
        if label in seen:
            raise ScreenError("ECB contains duplicate date")
        seen.add(label)
        value = float(row["OBS_VALUE"])
        if not math.isfinite(value):
            raise ScreenError("ECB contains non-finite value")
        normalized.append({"date": label, "value": value})
    normalized.sort(key=lambda item: item["date"])
    if normalized[0]["date"] < START_DATE or normalized[-1]["date"] > END_DATE or len(normalized) < 3500:
        raise ScreenError("ECB date bounds/coverage differ")
    first = rows[0]
    return normalized, {
        "series": ECB_SERIES,
        "frequency": "daily_businessweek",
        "unit": "percent_per_annum",
        "row_n": len(normalized),
        "non_missing_n": len(normalized),
        "missing_n": 0,
        "first_date": normalized[0]["date"],
        "last_date": normalized[-1]["date"],
        "title": first["TITLE"],
        "title_complement": first["TITLE_COMPL"],
        "current_vintage": True,
        "availability_note": "Previous TARGET-day curve released at noon CET/CEST; revisions are not reconstructed.",
    }


def _pdf_text(body: bytes) -> str:
    try:
        completed = subprocess.run(
            ["pdftotext", "-", "-"], input=body, capture_output=True, check=True, timeout=30
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ScreenError(f"official PDF extraction failed: {exc}") from exc
    return completed.stdout.decode("utf-8", "replace")


def _vendor_observations(bodies: Mapping[str, bytes]) -> tuple[list[str], list[str], list[str]]:
    checked: list[str] = []
    glbx = bodies.get("databento_glbx", b"").decode("utf-8", "replace")
    if "temporalCoverage\":\"2010-06-06/.." in glbx and "Trades, OHLCV-1s, OHLCV-1m" in glbx:
        checked.append("Databento GLBX.MDP3 declares coverage from 2010-06-06 and Trades/OHLCV-1m schemas.")
    xeur = bodies.get("databento_xeur", b"").decode("utf-8", "replace")
    if "2025-03-10T00:00:00.000000000Z" in xeur and "Euro-Schatz" in xeur and "ohlcv-1m" in xeur:
        checked.append("Databento XEUR.EOBI declares a 2025-03-10 start, Euro-Schatz coverage, and OHLCV-1m/trades schemas.")
    eurex = bodies.get("eurex_fgbs", b"").decode("utf-8", "replace")
    if "Euro-Schatz Futures (FGBS)" in eurex and "1.75 to 2.25" in eurex:
        checked.append("Eurex identifies FGBS as Euro-Schatz futures with 1.75-to-2.25-year remaining term.")
    sheet = _pdf_text(bodies["deutsche_boerse_product_sheet"])
    if "starting end of November 2013" in sheet and "trade price and traded" in sheet:
        checked.append("Deutsche Boerse's historical order-by-order sheet declares monthly availability from end-November 2013 and includes executed-trade price/quantity.")
    terms = _pdf_text(bodies["deutsche_boerse_terms"])
    if "solely offered to enterprises" in terms and "make the Historical Data available to third parties" in terms:
        checked.append("Deutsche Boerse Marketplace terms restrict licensing to enterprises and prohibit third-party availability absent order permission.")
    unknown = [
        "Exact authenticated Databento ZT trades and OHLCV-1m cost for the research interval.",
        "Current FGBS-only trades or one-minute subset availability and all-in Deutsche Boerse price.",
        "Whether the intended purchaser qualifies for and accepts Deutsche Boerse enterprise licensing.",
        "Product-specific ZT and FGBS downloadable sample availability without login or trial terms.",
        "Exact continuous/raw-contract roll construction offered by each vendor export.",
    ]
    needed = [
        "Authenticated non-ordering Databento get_cost estimates for ZT trades and OHLCV-1m.",
        "A Deutsche Boerse FGBS-only subset quote, license terms, and sample/trial manifest.",
        "User approval of exact price and terms before any account, contact form, license, or order.",
        "A successor preregistered paid-intraday experiment after sample timestamp/schema checks.",
    ]
    return checked, unknown, needed


def _acquire_snapshot() -> dict[str, Any]:
    rate_sources: list[dict[str, Any]] = []
    normalized_rates: dict[str, list[dict[str, Any]]] = {}
    raw_bodies: dict[str, bytes] = {}
    for source_id in ("fred_dgs2", "ecb_euro_area_2y_par"):
        body, status, content_type = _fetch(SOURCE_URLS[source_id])
        if status != 200:
            raise ScreenError(f"{source_id} HTTP status {status}")
        rows, metadata = _parse_fred(body) if source_id == "fred_dgs2" else _parse_ecb(body)
        record = _source_record(source_id, SOURCE_URLS[source_id], body, status, content_type)
        record.update({"metadata": metadata, "rows": rows})
        rate_sources.append(record)
    vendor_sources: list[dict[str, Any]] = []
    for source_id, url in SOURCE_URLS.items():
        if source_id in {"fred_dgs2", "ecb_euro_area_2y_par"}:
            continue
        try:
            body, status, content_type = _fetch(url)
        except Exception as exc:
            vendor_sources.append({
                "source_id": source_id, "url": url, "status": None, "bytes": 0,
                "sha256": None, "content_type": "", "retrieval_error": type(exc).__name__,
            })
            continue
        raw_bodies[source_id] = body
        record = _source_record(source_id, url, body, status, content_type)
        record["retrieval_error"] = None
        vendor_sources.append(record)
    required_vendor = {
        "databento_glbx", "databento_xeur", "eurex_fgbs",
        "deutsche_boerse_product_sheet", "deutsche_boerse_terms",
    }
    missing = sorted(required_vendor - set(raw_bodies))
    if missing:
        raise ScreenError(f"required official vendor evidence unavailable: {missing}")
    vendor_checked, vendor_unknown, vendor_needed = _vendor_observations(raw_bodies)
    if len(vendor_checked) < 5:
        raise ScreenError("required vendor observations were not supported by retrieved official bytes")
    checked = [
        "FRED DGS2 and ECB all-ratings two-year par current-vintage rows were fetched from official endpoints before outcome access.",
        "No source leg was independently forward-filled or interpolated.",
        *vendor_checked,
    ]
    return {
        "schema": SNAPSHOT_SCHEMA,
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "date_bounds": {"start": START_DATE, "end": END_DATE, "excluded_year": 2026},
        "rate_sources": rate_sources,
        "vendor_sources": vendor_sources,
        "checked": checked,
        "unknown": [
            "Historically visible point-in-time vintages for replaced FRED/ECB observations.",
            "An exact German Schatz yield in the free euro-area proxy.",
            *vendor_unknown,
        ],
        "needed": [
            "Paid exchange-timestamped ZT/FGBS minute or trade data to test an intraday lead.",
            *vendor_needed,
        ],
    }


def _validate_snapshot(value: Any) -> dict[str, Any]:
    raw = dict(_exact(value, SNAPSHOT_FIELDS, "source snapshot"))
    if raw["schema"] != SNAPSHOT_SCHEMA or raw["date_bounds"] != {
        "start": START_DATE, "end": END_DATE, "excluded_year": 2026
    }:
        raise ScreenError("snapshot identity/date bounds differ")
    try:
        stamp = datetime.fromisoformat(raw["retrieved_at_utc"])
    except (TypeError, ValueError) as exc:
        raise ScreenError("snapshot retrieval timestamp differs") from exc
    if stamp.tzinfo is None:
        raise ScreenError("snapshot timestamp lacks timezone")
    if not isinstance(raw["rate_sources"], list) or len(raw["rate_sources"]) != 2:
        raise ScreenError("snapshot rate source inventory differs")
    for item in raw["rate_sources"]:
        source_id = item.get("source_id")
        rows = item.get("rows")
        metadata = item.get("metadata")
        if source_id not in {"fred_dgs2", "ecb_euro_area_2y_par"} or not isinstance(rows, list) or not isinstance(metadata, Mapping):
            raise ScreenError("snapshot rate source differs")
        previous = None
        for row in rows:
            _exact(row, {"date", "value"}, f"{source_id} row")
            date.fromisoformat(row["date"])
            if previous is not None and row["date"] <= previous:
                raise ScreenError("snapshot rate chronology differs")
            previous = row["date"]
            if row["value"] is not None:
                _finite(row["value"], "rate value")
        expected_n = len(rows)
        if metadata.get("row_n") != expected_n or metadata.get("non_missing_n") != sum(row["value"] is not None for row in rows):
            raise ScreenError("snapshot rate counts differ")
    if not isinstance(raw["vendor_sources"], list) or len(raw["vendor_sources"]) != len(SOURCE_URLS) - 2:
        raise ScreenError("snapshot vendor source inventory differs")
    if not all(isinstance(raw[field], list) and raw[field] for field in ("checked", "unknown", "needed")):
        raise ScreenError("snapshot evidence fields differ")
    evidence.canonical_json_bytes(raw)
    return raw


def _measurement() -> dict[str, Any]:
    return {
        "pair": "EURUSD", "timeframe": "15m", "side": "DOWN",
        "selector": "worst_val_half", "diagnostic_selector": "legacy_full_val",
        "rate_proxy": "DGS2 minus ECB all-ratings 2-year par yield",
        "rate_window_common_observations": WINDOW,
        "trade_date_timezone": "America/New_York",
        "calendar_lag": "latest common date on or before the second preceding New York weekday",
        "favorable_rule": "delta5 > 0", "adverse_rule": "delta5 < 0",
        "zero_or_unavailable": "excluded_and_counted",
        "aggregation": "equal weight per New York trade date within regime",
        "path_power": {"minimum_dates_per_regime": MIN_PATH_DATES, "minimum_trades_per_regime": MIN_PATH_TRADES},
        "frozen_power": {"minimum_dates_per_regime": MIN_FROZEN_DATES, "minimum_trades_per_regime": MIN_FROZEN_TRADES},
        "falsifier": {
            "minimum_path_p10_lift": MIN_P10_LIFT,
            "minimum_strictly_positive_paths": MIN_POSITIVE_PATHS,
            "frozen_2024_positive": True,
            "frozen_2025_positive": True,
        },
        "current_vintage": True, "protected_year_excluded": 2026,
    }


def _validate_campaign(campaign: Mapping[str, Any], arm: Mapping[str, Any]) -> None:
    if (
        campaign.get("campaign_id") != CAMPAIGN_ID or campaign.get("data_use") != "retrospective"
        or campaign.get("key") != {"pair": "EURUSD", "timeframe": "15m", "target": "direction", "side": "DOWN"}
        or arm.get("arm_id") != ARM_ID or arm.get("native_result_path") != RESULT_PATH
        or campaign.get("adapter", {}).get("path") != SCRIPT_PATH
        or campaign.get("adapter", {}).get("native_result_contract") != RESULT_SCHEMA
        or arm.get("input_evidence_ids") != arm.get("data_use_basis_evidence_ids")
        or len(arm.get("input_evidence_ids", [])) != 1
    ):
        raise ScreenError("sealed campaign/arm identity differs")


def _validate_input_envelope(value: Mapping[str, Any], expected_id: str) -> Mapping[str, Any]:
    raw = _exact(value, {"schema", "object_id", "kind", "payload", "artifacts", "dependencies"}, "input envelope")
    if raw["object_id"] != expected_id or raw["kind"] != INPUT_SCHEMA or raw["dependencies"] != [ISSUE15_TERMINAL]:
        raise ScreenError("input envelope identity/dependency differs")
    payload = _exact(raw["payload"], {
        "schema", "key", "data_use", "ordered_artifact_paths", "selection_sha256",
        "source_snapshot_schema", "measurement", "protected_access", "claim_limit",
    }, "input payload")
    if (
        payload["schema"] != INPUT_SCHEMA
        or payload["key"] != {"pair": "EURUSD", "timeframe": "15m", "target": "direction", "side": "DOWN"}
        or payload["data_use"] != "retrospective"
        or payload["ordered_artifact_paths"] != [SNAPSHOT_PATH, SELECTION_PATH]
        or payload["selection_sha256"] != SELECTION_SHA256
        or payload["source_snapshot_schema"] != SNAPSHOT_SCHEMA
        or payload["measurement"] != _measurement()
        or payload["protected_access"] is not False or payload["claim_limit"] != CLAIM_LIMIT
    ):
        raise ScreenError("input payload differs")
    if not isinstance(raw["artifacts"], list) or [item.get("path") for item in raw["artifacts"]] != [SNAPSHOT_PATH, SELECTION_PATH]:
        raise ScreenError("input artifacts differ")
    return raw


def preflight_arm(
    *, campaign: Mapping[str, Any], arm: Mapping[str, Any], input_evidence: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    _validate_campaign(campaign, arm)
    reasons: list[str] = []
    try:
        if len(input_evidence) != 1:
            raise ScreenError("exactly one input envelope is required")
        _validate_input_envelope(input_evidence[0], arm["input_evidence_ids"][0])
    except ScreenError as exc:
        reasons.append(str(exc))
    return {
        "campaign_id": CAMPAIGN_ID, "arm_id": ARM_ID, "data_use": "retrospective",
        "data_use_basis_evidence_ids": list(arm["data_use_basis_evidence_ids"]),
        "protected_access": False, "admissible": not reasons, "reasons": reasons,
    }


def _verified_inputs(input_id: str) -> tuple[dict[str, Any], Mapping[str, Any]]:
    envelope = evidence.verify_object(input_id, repo_root=ROOT)
    raw = _validate_input_envelope(envelope.as_dict(), input_id)
    refs = [evidence.ArtifactRef.from_dict(dict(item)) for item in raw["artifacts"]]
    snapshot = _validate_snapshot(evidence.decode_canonical_json(refs[0].read_verified(repo_root=ROOT)))
    selection_bytes = refs[1].read_verified(repo_root=ROOT)
    if hashlib.sha256(selection_bytes).hexdigest() != SELECTION_SHA256:
        raise ScreenError("issue #15 result digest differs")
    selection = evidence.decode_canonical_json(selection_bytes)
    import eurusd_m15_down_selection_audit_v1 as prior

    prior_spec = _canonical_read(ROOT / SELECTION_SPEC_PATH)
    prior._validate_result(selection, prior_spec, prior_spec["ordered_arms"][0])
    return snapshot, selection


def _rate_map(snapshot: Mapping[str, Any]) -> tuple[list[date], list[float]]:
    sources = {item["source_id"]: item for item in snapshot["rate_sources"]}
    fred = {date.fromisoformat(row["date"]): row["value"] for row in sources["fred_dgs2"]["rows"] if row["value"] is not None}
    ecb = {date.fromisoformat(row["date"]): row["value"] for row in sources["ecb_euro_area_2y_par"]["rows"]}
    dates = sorted(set(fred) & set(ecb))
    if len(dates) < 3400:
        raise NotComputable("common_non_missing_rate_dates_below_3400")
    spreads = [float(fred[label] - ecb[label]) for label in dates]
    return dates, spreads


def _second_preceding_weekday(label: date) -> date:
    current = label
    for _ in range(2):
        current -= timedelta(days=1)
        while current.weekday() >= 5:
            current -= timedelta(days=1)
    return current


def _regime(timestamp: int, dates: Sequence[date], spreads: Sequence[float]) -> tuple[str | None, str]:
    trade_date = datetime.fromtimestamp(timestamp, timezone.utc).astimezone(NY).date()
    cutoff = _second_preceding_weekday(trade_date)
    index = bisect.bisect_right(dates, cutoff) - 1
    if index < WINDOW:
        return None, trade_date.isoformat()
    delta = spreads[index] - spreads[index - WINDOW]
    if delta > 0:
        return "favorable", trade_date.isoformat()
    if delta < 0:
        return "adverse", trade_date.isoformat()
    return None, trade_date.isoformat()


def _view(trades: Sequence[Mapping[str, Any]], dates: Sequence[date], spreads: Sequence[float]) -> dict[str, Any]:
    grouped: dict[str, dict[str, list[bool]]] = {
        "favorable": defaultdict(list), "adverse": defaultdict(list), "excluded": defaultdict(list)
    }
    for trade in trades:
        if not isinstance(trade.get("timestamp"), int) or not isinstance(trade.get("down_win"), bool):
            raise NotComputable("issue15_trade_schema_differs")
        if datetime.fromtimestamp(trade["timestamp"], timezone.utc).year >= 2026:
            raise NotComputable("issue15_contains_2026_timestamp")
        regime, trade_date = _regime(trade["timestamp"], dates, spreads)
        grouped[regime or "excluded"][trade_date].append(trade["down_win"])

    def cell(name: str) -> dict[str, Any]:
        day_values = [sum(wins) / len(wins) for _, wins in sorted(grouped[name].items())]
        return {
            "date_n": len(day_values),
            "trade_n": sum(len(wins) for wins in grouped[name].values()),
            "equal_date_down_win_rate": sum(day_values) / len(day_values) if day_values else None,
        }

    favorable, adverse = cell("favorable"), cell("adverse")
    lift = None
    if favorable["equal_date_down_win_rate"] is not None and adverse["equal_date_down_win_rate"] is not None:
        lift = favorable["equal_date_down_win_rate"] - adverse["equal_date_down_win_rate"]
    return {
        "favorable": favorable,
        "adverse": adverse,
        "excluded_trade_n": sum(len(wins) for wins in grouped["excluded"].values()),
        "excluded_date_n": len(grouped["excluded"]),
        "lift": lift,
    }


def _power_reasons(view: Mapping[str, Any], *, frozen: bool, name: str) -> list[str]:
    minimum_dates = MIN_FROZEN_DATES if frozen else MIN_PATH_DATES
    minimum_trades = MIN_FROZEN_TRADES if frozen else MIN_PATH_TRADES
    reasons = []
    for regime in ("favorable", "adverse"):
        cell = view[regime]
        if cell["date_n"] < minimum_dates or cell["trade_n"] < minimum_trades:
            reasons.append(f"{name}_{regime}_below_power")
    if view["lift"] is None:
        reasons.append(f"{name}_lift_unavailable")
    return reasons


def _summary(values: Sequence[float]) -> dict[str, Any]:
    import numpy as np

    array = np.asarray(values, dtype=float)
    if len(array) != 15 or not bool(np.all(np.isfinite(array))):
        raise NotComputable("complete_15_path_lift_summary_unavailable")
    return {
        "n_paths": 15,
        "mean": float(array.mean()),
        "p10": float(np.percentile(array, 10, method="linear")),
        "min": float(array.min()),
        "max": float(array.max()),
        "fraction_positive": float((array > 0).mean()),
    }


def _measure(selection: Mapping[str, Any], snapshot: Mapping[str, Any]) -> tuple[Any, ...]:
    dates, spreads = _rate_map(snapshot)
    paths: list[dict[str, Any]] = []
    legacy: list[dict[str, Any]] = []
    reasons: list[str] = []
    primary_lifts: list[float] = []
    legacy_lifts: list[float] = []
    if not isinstance(selection.get("paths"), list) or len(selection["paths"]) != 15:
        raise NotComputable("issue15_path_inventory_differs")
    for index, path in enumerate(selection["paths"]):
        if path.get("path_index") != index:
            raise NotComputable("issue15_path_order_differs")
        primary_view = _view(path["selectors"]["worst_val_half"]["test"]["trades"], dates, spreads)
        legacy_view = _view(path["selectors"]["legacy_full_val"]["test"]["trades"], dates, spreads)
        reasons.extend(_power_reasons(primary_view, frozen=False, name=f"path_{index}"))
        if primary_view["lift"] is not None:
            primary_lifts.append(float(primary_view["lift"]))
        if legacy_view["lift"] is not None:
            legacy_lifts.append(float(legacy_view["lift"]))
        paths.append({"path_index": index, "test_groups": path["test_groups"], **primary_view})
        legacy.append({"path_index": index, "test_groups": path["test_groups"], **legacy_view})
    frozen: dict[str, Any] = {}
    for year in ("2024", "2025"):
        view = _view(selection["frozen_forward"]["years"][year]["trades"], dates, spreads)
        reasons.extend(_power_reasons(view, frozen=True, name=f"frozen_{year}"))
        frozen[year] = view
    summaries = _summary(primary_lifts) if len(primary_lifts) == 15 else {}
    legacy_summary = _summary(legacy_lifts) if len(legacy_lifts) == 15 else {}
    reasons = sorted(set(reasons))
    if reasons:
        outcome = "not_computable"
    elif (
        summaries["p10"] < MIN_P10_LIFT
        or summaries["fraction_positive"] < MIN_POSITIVE_PATHS / 15
        or frozen["2024"]["lift"] <= 0
        or frozen["2025"]["lift"] <= 0
    ):
        outcome = "daily_rate_regime_falsified"
    else:
        outcome = "daily_rate_regime_not_falsified"
    return paths, summaries, {"years": frozen}, {"paths": legacy, "summary": legacy_summary}, reasons, outcome


def _procurement(snapshot: Mapping[str, Any], outcome: str) -> dict[str, Any]:
    if outcome == "daily_rate_regime_falsified":
        recommendation = "do_not_recommend_purchase"
        showed = ["The preregistered daily rate-regime screen was falsified; this lowers the prior for purchasing the paid history."]
    elif outcome == "daily_rate_regime_not_falsified":
        recommendation = "purchase_decision_deferred"
        showed = ["The daily regime association survived, but exact price, sample, and product-specific rights remain unavailable."]
    else:
        recommendation = "purchase_decision_deferred"
        showed = ["The daily screen was not computable, so it cannot support a purchase recommendation."]
    return {
        "recommendation": recommendation,
        "checked": list(snapshot["checked"]),
        "showed": showed,
        "unknown": list(snapshot["unknown"]),
        "needed": list(snapshot["needed"]),
        "purchase_authorized": False,
        "vendor_contacted": False,
        "account_created": False,
        "terms_accepted": False,
        "sample_downloaded": False,
    }


def _result(
    campaign: Mapping[str, Any], arm: Mapping[str, Any], snapshot: Mapping[str, Any],
    selection: Mapping[str, Any], environment: Mapping[str, Any],
) -> dict[str, Any]:
    paths, summaries, frozen, legacy, reasons, outcome = _measure(selection, snapshot)
    showed = []
    if outcome == "daily_rate_regime_falsified":
        showed.append("The fixed calendar-lagged current-vintage daily regime failed its procurement falsifier.")
    elif outcome == "daily_rate_regime_not_falsified":
        showed.append("The fixed calendar-lagged current-vintage daily regime was not falsified by the procurement screen.")
    else:
        showed.append("The daily regime screen was not computable under its preregistered power gates.")
    return {
        "schema": RESULT_SCHEMA, "campaign_id": CAMPAIGN_ID, "arm_id": ARM_ID,
        "input_evidence_ids": list(arm["input_evidence_ids"]),
        "status": "not_computable" if reasons else "completed", "reasons": reasons,
        "protected_access": False, "claim_limit": CLAIM_LIMIT, "measurement": _measurement(),
        "source": {
            "snapshot_schema": snapshot["schema"],
            "snapshot_retrieved_at_utc": snapshot["retrieved_at_utc"],
            "selection_sha256": SELECTION_SHA256,
            "environment": dict(environment),
        },
        "paths": paths, "summaries": summaries, "frozen_forward": frozen,
        "legacy_diagnostic": legacy, "outcome": outcome,
        "procurement": _procurement(snapshot, outcome),
        "evidence": {
            "checked": [
                "The result recomputes from the sealed source/procurement snapshot and exact issue #15 bytes.",
                "All 15 worst-half paths and frozen 2024/2025 controls were evaluated without refitting.",
            ],
            "showed": showed,
            "unknown": [
                "Whether historically point-in-time daily vintages would reproduce this association.",
                "Whether minute ZT-minus-FGBS repricing leads EURUSD at 15 minutes.",
            ],
            "needed": [
                "Exact vendor price/rights/sample evidence before any paid-data decision.",
                "A separately preregistered paid intraday experiment to test the distinct lead-lag mechanism.",
            ],
        },
    }


def _receipt(campaign: Mapping[str, Any], arm: Mapping[str, Any], result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": RECEIPT_SCHEMA, "campaign_id": CAMPAIGN_ID, "arm_id": ARM_ID,
        "native_result_contract": campaign["adapter"]["native_result_contract"],
        "status": result["status"], "reasons": list(result["reasons"]),
        "native_result_path": arm["native_result_path"],
        "input_evidence_ids": list(arm["input_evidence_ids"]), "data_use": "retrospective",
        "data_use_basis_evidence_ids": list(arm["data_use_basis_evidence_ids"]),
        "protected_access": False,
    }


def run_arm(
    *, campaign: Mapping[str, Any], arm: Mapping[str, Any], temporary_output_path: Path,
    environment: Mapping[str, Any],
) -> dict[str, Any]:
    _validate_campaign(campaign, arm)
    try:
        snapshot, selection = _verified_inputs(arm["input_evidence_ids"][0])
        result = _result(campaign, arm, snapshot, selection, environment)
    except (ScreenError, NotComputable) as exc:
        result = {
            "schema": RESULT_SCHEMA, "campaign_id": CAMPAIGN_ID, "arm_id": ARM_ID,
            "input_evidence_ids": list(arm["input_evidence_ids"]), "status": "not_computable",
            "reasons": [str(exc)], "protected_access": False, "claim_limit": CLAIM_LIMIT,
            "measurement": _measurement(), "source": {}, "paths": [], "summaries": {},
            "frozen_forward": {}, "legacy_diagnostic": {}, "outcome": "not_computable",
            "procurement": {}, "evidence": {},
        }
    _canonical_write(temporary_output_path, result)
    return validate_arm_result(campaign=campaign, arm=arm, native_result_path=temporary_output_path)


def _validate_cell(value: Any, name: str) -> None:
    raw = _exact(value, CELL_FIELDS, name)
    if not isinstance(raw["date_n"], int) or not isinstance(raw["trade_n"], int):
        raise ScreenError(f"{name} counts differ")
    if raw["equal_date_down_win_rate"] is not None:
        rate = _finite(raw["equal_date_down_win_rate"], f"{name} rate")
        if not 0 <= rate <= 1:
            raise ScreenError(f"{name} rate range differs")


def _validate_view(value: Any, name: str) -> None:
    raw = _exact(value, VIEW_FIELDS, name)
    _validate_cell(raw["favorable"], f"{name}.favorable")
    _validate_cell(raw["adverse"], f"{name}.adverse")
    if not isinstance(raw["excluded_trade_n"], int) or not isinstance(raw["excluded_date_n"], int):
        raise ScreenError(f"{name} excluded counts differ")
    if raw["lift"] is not None:
        expected = raw["favorable"]["equal_date_down_win_rate"] - raw["adverse"]["equal_date_down_win_rate"]
        if not math.isclose(_finite(raw["lift"], f"{name}.lift"), expected, rel_tol=0, abs_tol=1e-15):
            raise ScreenError(f"{name} lift differs")


def _validate_summary(value: Any, name: str) -> None:
    raw = _exact(value, SUMMARY_FIELDS, name)
    if raw["n_paths"] != 15:
        raise ScreenError(f"{name} path count differs")
    for field in SUMMARY_FIELDS - {"n_paths"}:
        _finite(raw[field], f"{name}.{field}")


def _validate_result(value: Any, campaign: Mapping[str, Any], arm: Mapping[str, Any]) -> dict[str, Any]:
    _validate_campaign(campaign, arm)
    raw = dict(_exact(value, RESULT_FIELDS, "native result"))
    if (
        raw["schema"] != RESULT_SCHEMA or raw["campaign_id"] != CAMPAIGN_ID or raw["arm_id"] != ARM_ID
        or raw["input_evidence_ids"] != list(arm["input_evidence_ids"])
        or raw["protected_access"] is not False or raw["claim_limit"] != CLAIM_LIMIT
        or raw["measurement"] != _measurement()
    ):
        raise ScreenError("native result identity/authority differs")
    if raw["status"] == "not_computable" and raw["paths"] == []:
        if not raw["reasons"] or raw["outcome"] != "not_computable":
            raise ScreenError("empty not-computable semantics differ")
        return raw
    snapshot, selection = _verified_inputs(arm["input_evidence_ids"][0])
    expected = _result(campaign, arm, snapshot, selection, raw["source"]["environment"])
    if raw != expected:
        raise ScreenError("native result does not recompute from sealed inputs")
    if len(raw["paths"]) != 15 or len(raw["legacy_diagnostic"]["paths"]) != 15:
        raise ScreenError("path inventory differs")
    for index, path in enumerate(raw["paths"]):
        view = {field: path[field] for field in VIEW_FIELDS}
        _validate_view(view, f"paths[{index}]")
    for index, path in enumerate(raw["legacy_diagnostic"]["paths"]):
        view = {field: path[field] for field in VIEW_FIELDS}
        _validate_view(view, f"legacy[{index}]")
    if raw["summaries"]:
        _validate_summary(raw["summaries"], "summaries")
        _validate_summary(raw["legacy_diagnostic"]["summary"], "legacy summary")
    for year in ("2024", "2025"):
        _validate_view(raw["frozen_forward"]["years"][year], f"frozen {year}")
    if raw["procurement"]["purchase_authorized"] is not False or raw["procurement"]["vendor_contacted"] is not False:
        raise ScreenError("procurement authority differs")
    evidence.canonical_json_bytes(raw)
    return raw


def validate_arm_result(
    *, campaign: Mapping[str, Any], arm: Mapping[str, Any], native_result_path: Path
) -> dict[str, Any]:
    if native_result_path.is_symlink() or not native_result_path.is_file():
        raise ScreenError("native result must be a regular non-symlink file")
    result = _validate_result(_canonical_read(native_result_path), campaign, arm)
    return _receipt(campaign, arm, result)


def evaluate_family(
    *, campaign: Mapping[str, Any], receipts: Sequence[Mapping[str, Any]],
    native_results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if len(receipts) != 1 or len(native_results) != 1:
        raise ScreenError("one complete arm result is required")
    arm = campaign["ordered_arms"][0]
    reference = evidence.ArtifactRef.from_dict(dict(native_results[0]))
    result = _validate_result(
        evidence.decode_canonical_json(reference.read_verified(repo_root=ROOT)), campaign, arm
    )
    if result["status"] == "not_computable":
        status, decision, candidates, reasons = "not_computable", "capability_deferred", [], list(result["reasons"])
    elif result["outcome"] == "daily_rate_regime_falsified":
        status, decision, candidates, reasons = "killed", "no_candidate", [], ["daily_rate_regime_falsified"]
    else:
        status, decision, candidates, reasons = "candidate", "inactive_candidate", [ARM_ID], []
    return {
        "schema": DECISION_SCHEMA, "campaign_id": CAMPAIGN_ID,
        "verdicts": [{"arm_id": ARM_ID, "status": status, "reasons": reasons}],
        "ordered_candidates": candidates, "decision": decision, "reasons": reasons,
    }


def _input_payload() -> dict[str, Any]:
    return {
        "schema": INPUT_SCHEMA,
        "key": {"pair": "EURUSD", "timeframe": "15m", "target": "direction", "side": "DOWN"},
        "data_use": "retrospective", "ordered_artifact_paths": [SNAPSHOT_PATH, SELECTION_PATH],
        "selection_sha256": SELECTION_SHA256, "source_snapshot_schema": SNAPSHOT_SCHEMA,
        "measurement": _measurement(), "protected_access": False, "claim_limit": CLAIM_LIMIT,
    }


def _spec(input_id: str) -> dict[str, Any]:
    return {
        "schema": "research-campaign-spec-input/v1", "campaign_id": CAMPAIGN_ID,
        "key": {"pair": "EURUSD", "timeframe": "15m", "target": "direction", "side": "DOWN"},
        "question": "Is the existing historical EURUSD 15m DOWN signal stronger when the calendar-lagged current-vintage U.S.-euro-area two-year proxy widens?",
        "hypothesis": "The fixed favorable daily-rate regime improves equal-date DOWN win rate consistently enough to justify paid-data review.",
        "falsifier": "Path-lift p10 below +0.005, fewer than 12/15 positive paths, or non-positive frozen 2024 or 2025 lift.",
        "data_use": "retrospective",
        "ordered_arms": [{
            "arm_id": ARM_ID, "role": "candidate", "native_result_path": RESULT_PATH,
            "input_evidence_ids": [input_id], "data_use_basis_evidence_ids": [input_id],
        }],
        "adapter": {
            "path": SCRIPT_PATH, "run_function": "run_arm", "preflight_function": "preflight_arm",
            "validate_function": "validate_arm_result", "evaluate_function": "evaluate_family",
            "native_result_contract": RESULT_SCHEMA,
        },
        "stopping_rule": "Run the exact one-arm screen once; stop on completion or any preregistered computability gate.",
        "terminal_rule": {
            "allowed_decisions": ["no_candidate", "inactive_candidate", "capability_deferred"],
            "activation": False,
        },
    }


def _prepare_and_run() -> None:
    import research_campaign_v1 as runner

    snapshot_path = ROOT / SNAPSHOT_PATH
    if snapshot_path.exists():
        snapshot = _validate_snapshot(_canonical_read(snapshot_path))
    else:
        snapshot = _acquire_snapshot()
        _validate_snapshot(snapshot)
        _canonical_write(snapshot_path, snapshot)
    # Outcome bytes are first opened only after the complete source/procurement snapshot is durable.
    artifacts = [
        evidence.ArtifactRef.capture(path, repo_root=ROOT)
        for path in (SNAPSHOT_PATH, SELECTION_PATH)
    ]
    input_envelope = evidence.EvidenceEnvelope.create(
        kind=INPUT_SCHEMA, payload=_input_payload(), artifacts=artifacts,
        dependencies=[ISSUE15_TERMINAL],
    )
    evidence.publish(input_envelope, repo_root=ROOT)
    spec = _spec(input_envelope.object_id)
    spec_path = ROOT / SPEC_PATH
    if spec_path.exists():
        if _canonical_read(spec_path) != spec:
            raise ScreenError("campaign spec path contains divergent bytes")
    else:
        _canonical_write(spec_path, spec)
    terminal = runner.run_campaign(spec_path, repo_root=ROOT)
    print(f"[daily-rates] terminal={terminal.object_id}")


def _verify_completed() -> None:
    import research_campaign_v1 as runner

    terminal = runner.verify_campaign(CAMPAIGN_ID, repo_root=ROOT)
    spec = _canonical_read(ROOT / SPEC_PATH)
    validate_arm_result(campaign=spec, arm=spec["ordered_arms"][0], native_result_path=ROOT / RESULT_PATH)
    print(f"[daily-rates] verified terminal={terminal.object_id}")


def _self_test() -> None:
    assert _second_preceding_weekday(date(2025, 1, 6)) == date(2025, 1, 2)
    dates = [date(2024, 12, 20) + timedelta(days=index) for index in range(20)]
    spreads = [float(index) for index in range(20)]
    monday = int(datetime(2025, 1, 6, 14, tzinfo=timezone.utc).timestamp())
    assert _regime(monday, dates, spreads)[0] == "favorable"
    synthetic = []
    for day_index in range(20):
        stamp = int(datetime(2025, 2, 3 + day_index, 14, tzinfo=timezone.utc).timestamp())
        synthetic.extend({"timestamp": stamp + offset * 900, "down_win": bool(day_index % 2)} for offset in range(3))
    # Exact schema rejects unknown fields.
    sample = {field: None for field in RESULT_FIELDS}
    sample["unknown"] = True
    try:
        _exact(sample, RESULT_FIELDS, "synthetic")
    except ScreenError:
        pass
    else:
        raise AssertionError("strict result schema accepted an unknown field")
    summary = _summary([0.005] * 15)
    assert summary["p10"] == 0.005 and summary["fraction_positive"] == 1.0
    assert _spec("0" * 64)["terminal_rule"]["activation"] is False
    assert "2026" not in FRED_URL and "2026" not in ECB_URL
    assert _measurement()["current_vintage"] is True
    print("[daily-rates] self-test PASS")


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
