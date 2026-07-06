"""Replay the July 6 Deriv queue pileup fixture under allocator policies."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from deriv_trade_queue import usd_factor_direction, write_gate_result


DEFAULT_FIXTURE = Path("deriv_data/replay_fixtures/deriv_trade_allocator_2026-07-06")
DEFAULT_OUT = Path("results/json/deriv_trade_allocator_replay_2026-07-06_result.json")
DEFAULT_CUTOFF = "2026-07-06T18:58:27.399362+00:00"
EXPECTED_SOURCE = {
    "queue_status": {"bought": 171, "signal_expired": 1, "terminal_skip": 33},
    "contract_status": {"open": 171},
    "bought_pair_side": {
        "AUDUSD DOWN": 82,
        "GBPUSD DOWN": 15,
        "USDCAD UP": 26,
        "USDJPY UP": 48,
    },
    "executor_events": {
        "buy_confirmed": 171,
        "payout_gate_failed": 33,
        "signal_expired": 1,
    },
}


def parse_utc(value: str) -> datetime:
    ts = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def count_by(rows: list[sqlite3.Row], *keys: str) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        label = " ".join(str(row[k]) for k in keys)
        counts[label] += 1
    return dict(sorted(counts.items()))


def load_rows(db_path: Path, cutoff: datetime) -> tuple[list[sqlite3.Row], list[sqlite3.Row], dict[str, list[str]]]:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    schema = {
        "signals": [row["name"] for row in conn.execute("PRAGMA table_info(signals)")],
        "contracts": [row["name"] for row in conn.execute("PRAGMA table_info(contracts)")],
    }
    cutoff_s = cutoff.isoformat()
    signals = conn.execute(
        """
        SELECT * FROM signals
        WHERE created_utc <= ? AND updated_utc <= ?
        ORDER BY created_utc, signal_id
        """,
        (cutoff_s, cutoff_s),
    ).fetchall()
    contracts = conn.execute(
        """
        SELECT * FROM contracts
        WHERE created_utc <= ?
        ORDER BY created_utc, contract_id
        """,
        (cutoff_s,),
    ).fetchall()
    conn.close()
    return signals, contracts, schema


def read_jsonl(path: Path, cutoff: datetime) -> tuple[Counter[str], dict[str, dict[str, Any]], dict[str, Any]]:
    counts: Counter[str] = Counter()
    proposal_by_signal: dict[str, dict[str, Any]] = {}
    meta = {"lines": 0, "bad_json": 0, "asof_lines": 0}
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            meta["lines"] += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                meta["bad_json"] += 1
                continue
            raw_ts = row.get("timestamp_utc")
            if raw_ts and parse_utc(raw_ts) > cutoff:
                continue
            meta["asof_lines"] += 1
            event = row.get("event", "<missing>")
            counts[event] += 1
            if event == "proposal_selected" and row.get("signal_id"):
                proposal_by_signal[str(row["signal_id"])] = row
    return counts, proposal_by_signal, meta


def overlap_metrics(rows: list[sqlite3.Row]) -> dict[str, dict[str, Any]]:
    by_key: dict[str, list[datetime]] = defaultdict(list)
    for row in rows:
        if row["status"] != "bought":
            continue
        by_key[f"{row['pair']} {row['side']}"].append(parse_utc(row["signal_close_utc"]))
    out: dict[str, dict[str, Any]] = {}
    for key, vals in sorted(by_key.items()):
        vals.sort()
        gaps = [(b - a).total_seconds() for a, b in zip(vals, vals[1:])]
        lt900 = [g for g in gaps if g < 900]
        out[key] = {
            "bought": len(vals),
            "gaps_lt_900s": len(lt900),
            "min_gap_s": min(gaps) if gaps else None,
        }
    return out


def policy_replay(
    signals: list[sqlite3.Row],
    proposal_by_signal: dict[str, dict[str, Any]],
    *,
    max_usd_factor_open: int,
    disabled_pair_sides: set[tuple[str, str]],
) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    missing_proposal_context = 0
    for row in signals:
        pair = str(row["pair"]).upper()
        side = str(row["side"]).upper()
        if row["status"] == "signal_expired":
            counts["signal_expired"] += 1
            continue
        if row["status"] == "terminal_skip":
            counts[f"payout_skip:{row['terminal_reason'] or 'terminal_skip'}"] += 1
            continue
        if row["status"] != "bought":
            counts[f"unknown_source_status:{row['status']}"] += 1
            continue
        prop = proposal_by_signal.get(str(row["signal_id"]))
        if prop is None or prop.get("live_breakeven") is None:
            missing_proposal_context += 1
            live_breakeven = None
            net_edge = float(row["effective_floor"])
        else:
            live_breakeven = float(prop["live_breakeven"])
            net_edge = float(row["effective_floor"]) - live_breakeven
        candidates.append({
            "signal_id": row["signal_id"],
            "pair": pair,
            "side": side,
            "close": parse_utc(row["signal_close_utc"]),
            "effective_floor": float(row["effective_floor"]),
            "live_breakeven": live_breakeven,
            "net_edge": net_edge,
        })
    active: list[dict[str, Any]] = []
    buys: list[dict[str, Any]] = []
    skips: list[dict[str, Any]] = []
    max_same_pair_open = 0
    max_usd_open = 0
    for cand in sorted(candidates, key=lambda item: (item["close"], -item["net_edge"], item["signal_id"])):
        active = [item for item in active if item["expiry"] > cand["close"]]
        usd_dir = usd_factor_direction(cand["pair"], cand["side"])
        if (cand["pair"], cand["side"]) in disabled_pair_sides:
            reason = "allocation_disabled_pair_side"
        elif any(item["pair"] == cand["pair"] for item in active):
            reason = "allocation_same_pair_overlap"
        elif max_usd_factor_open > 0 and sum(1 for item in active if item["usd_dir"] == usd_dir) >= max_usd_factor_open:
            reason = "allocation_usd_factor_cap"
        else:
            reason = None
        if reason:
            counts[reason] += 1
            skips.append({"signal_id": cand["signal_id"], "pair": cand["pair"], "side": cand["side"], "reason": reason})
            continue
        item = {
            **cand,
            "usd_dir": usd_dir,
            "expiry": cand["close"] + timedelta(seconds=900),
        }
        active.append(item)
        buys.append(cand)
        counts["bought"] += 1
        max_same_pair_open = max(max_same_pair_open, max(sum(1 for item in active if item["pair"] == pair) for pair in {x["pair"] for x in active}))
        max_usd_open = max(max_usd_open, max(sum(1 for item in active if item["usd_dir"] == direction) for direction in {x["usd_dir"] for x in active}))
    bought_rows = [
        {"pair": item["pair"], "side": item["side"], "signal_close_utc": item["close"].isoformat(), "status": "bought"}
        for item in buys
    ]
    return {
        "max_usd_factor_open": max_usd_factor_open if max_usd_factor_open > 0 else "off",
        "disabled_pair_sides": sorted(f"{p}:{s}" for p, s in disabled_pair_sides),
        "counts": dict(sorted(counts.items())),
        "bought_pair_side": dict(sorted(Counter(f"{item['pair']} {item['side']}" for item in buys).items())),
        "bought_total": len(buys),
        "skipped_total": len(skips),
        "missing_proposal_context": missing_proposal_context,
        "max_same_pair_open": max_same_pair_open,
        "max_usd_factor_observed": max_usd_open,
        "overlap_metrics": overlap_metrics([FakeRow(item) for item in bought_rows]),
    }


class FakeRow(dict):
    def __getitem__(self, key: str) -> Any:
        return dict.__getitem__(self, key)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-dir", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--cutoff", default=DEFAULT_CUTOFF)
    args = parser.parse_args(argv)

    fixture = args.fixture_dir
    cutoff = parse_utc(args.cutoff)
    manifest_path = fixture / "capture_manifest.json"
    db_path = fixture / "trade_queue.sqlite"
    executor_log = fixture / "trade_executor_jsonl_2026-07-06.jsonl"
    supervisor_log = fixture / "supervisor_jsonl_2026-07-06.jsonl"
    required = [manifest_path, db_path, executor_log, supervisor_log]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit(f"missing fixture files: {missing}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    signals, contracts, schema = load_rows(db_path, cutoff)
    executor_counts, proposal_by_signal, executor_meta = read_jsonl(executor_log, cutoff)
    supervisor_counts, _, supervisor_meta = read_jsonl(supervisor_log, cutoff)
    source = {
        "cutoff_utc": cutoff.isoformat(),
        "queue_status": dict(sorted(Counter(row["status"] for row in signals).items())),
        "contract_status": dict(sorted(Counter(row["contract_status"] for row in contracts).items())),
        "bought_pair_side": count_by([row for row in signals if row["status"] == "bought"], "pair", "side"),
        "overlap_metrics": overlap_metrics(signals),
        "executor_event_counts": dict(sorted(executor_counts.items())),
        "supervisor_event_counts": dict(sorted(supervisor_counts.items())),
        "executor_log_meta": executor_meta,
        "supervisor_log_meta": supervisor_meta,
        "schema_columns": schema,
    }
    cap_sensitivity = {
        "off": policy_replay(signals, proposal_by_signal, max_usd_factor_open=0, disabled_pair_sides=set()),
        "1": policy_replay(signals, proposal_by_signal, max_usd_factor_open=1, disabled_pair_sides=set()),
        "2": policy_replay(signals, proposal_by_signal, max_usd_factor_open=2, disabled_pair_sides=set()),
    }
    deploy_policy = policy_replay(
        signals,
        proposal_by_signal,
        max_usd_factor_open=2,
        disabled_pair_sides={("AUDUSD", "DOWN")},
    )
    source_matches = (
        source["queue_status"] == EXPECTED_SOURCE["queue_status"]
        and source["contract_status"] == EXPECTED_SOURCE["contract_status"]
        and source["bought_pair_side"] == EXPECTED_SOURCE["bought_pair_side"]
        and all(executor_counts.get(k) == v for k, v in EXPECTED_SOURCE["executor_events"].items())
    )
    policy_two = cap_sensitivity["2"]
    doc = {
        "gate": "issue#7 deriv_trade_allocator replay",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "fixture": {
            "dir": str(fixture),
            "capture_manifest": manifest,
            "fixture_sha256": {path.name: sha256(path) for path in required},
            "gitignore_proof": "repo .gitignore contains deriv_data/; verified during capture",
        },
        "source_facts": source,
        "replay": {
            "cap_sensitivity": cap_sensitivity,
            "deploy_policy": deploy_policy,
        },
        "falsifiers": {
            "source_facts_reproduced": source_matches,
            "cap_sensitivity_present": set(cap_sensitivity) == {"off", "1", "2"},
            "cap2_max_usd_factor_holds": policy_two["max_usd_factor_observed"] <= 2,
            "cap2_no_same_pair_overlap": all(v["gaps_lt_900s"] == 0 for v in policy_two["overlap_metrics"].values()),
            "deploy_disabled_audusd_down": deploy_policy["bought_pair_side"].get("AUDUSD DOWN", 0) == 0,
            "all_source_rows_accounted": sum(source["queue_status"].values()) == (
                policy_two["bought_total"] + policy_two["skipped_total"] + source["queue_status"].get("terminal_skip", 0) + source["queue_status"].get("signal_expired", 0)
            ),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_gate_result(args.out, doc)
    ok = all(doc["falsifiers"].values())
    print(f"{'REPLAY PASS' if ok else 'REPLAY FAIL'} -> {args.out}")
    print(json.dumps(doc["falsifiers"], indent=2, sort_keys=True))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
