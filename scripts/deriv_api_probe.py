"""Phase-0 Deriv API probe (issue #4): evidence lock for the hot 5s runtime build.

Unauthenticated, public endpoints only, proposal-only — this script never calls
`buy` (and uses public sockets, on which `deriv_client.buy` refuses anyway).

Probes (letters match issue #4 Phase 0):
  a  candle granularity enum (sub-60s rejected; 60s spacing exact)
  b  1-second tick history, all six enabled symbols
  c  live `ticks` subscription: cadence, fields, six-symbol multiplex, per endpoint
  d  tick-history retention depth (binary search) + paging mechanics (3 pages)
  e  proposal fields + proposal subscription support + contracts_for metadata
  f  sustainable request budget: worst-case 18 req / 5s cycle soak
  g  near-close proposal behavior (~16:44-17:00 NY) — GATED: refuses outside window
  h  one-shot executor baseline wall/RSS — GATED: needs --baseline-store-dir
  i  wall-clock candle publication latency + candle subscription support

Results merge per-probe into deriv_api_probe_result.json (CWD; archive to
results/json/ per repo convention). Writes are atomic (tmp + os.replace,
deriv_backfill discipline); re-runs update only the probes requested, so g/h
can be added later on the VPS without clobbering the unattended probes.

Run from repo root:
    ~/binary-algo-venv/bin/python scripts/deriv_api_probe.py                # a,b,c,d,e,f,i
    ~/binary-algo-venv/bin/python scripts/deriv_api_probe.py --probes g    # 16:35-17:05 NY only
    ~/binary-algo-venv/bin/python scripts/deriv_api_probe.py --probes h --baseline-store-dir deriv_data/candles_1m
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import websocket

from deriv_backfill import DEFAULT_ENABLED_PAIRS, DERIV_LEGACY_WS_URL, PAIR_TO_SYMBOL
from deriv_client import PUBLIC_WS_URL, DerivAPIError, DerivOptionsClient

RESULT_PATH = Path("deriv_api_probe_result.json")
ISSUE = "sppburke/binary-algo#4"
PROBE_VERSION = 1
REQUEST_SLEEP_S = 0.25  # syn_collect.py precedent: stay under the 220 req/min general cap
SYMBOLS = {pair: PAIR_TO_SYMBOL[pair] for pair in DEFAULT_ENABLED_PAIRS}
ENDPOINTS = {"options": PUBLIC_WS_URL, "legacy": DERIV_LEGACY_WS_URL}
RECV_TIMEOUT_EXCS = (websocket.WebSocketTimeoutException, socket.timeout, TimeoutError)
EPOCH_2000 = 946684800  # binary-search floor for retention depth
UNATTENDED_PROBES = "a,b,c,d,e,f,i"


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def ny_now() -> datetime:
    # Reuse the executor's session definition; lazy import keeps --help light.
    from deriv_demo_executor import NY_TZ

    return datetime.now(NY_TZ)


def in_ny_session_now() -> bool:
    from deriv_demo_executor import is_ny_session

    return is_ny_session(datetime.now(timezone.utc))


def probe_context() -> dict[str, Any]:
    return {
        "started_utc": now_utc_iso(),
        "host": socket.gethostname(),
        "ny_time": ny_now().isoformat(timespec="seconds"),
        "in_ny_session": in_ny_session_now(),
    }


def delta_stats(epochs: list[int]) -> dict[str, Any]:
    if len(epochs) < 2:
        return {"n": len(epochs)}
    deltas = [b - a for a, b in zip(epochs, epochs[1:])]
    hist: dict[str, int] = {}
    for d in deltas:
        key = str(d) if d <= 5 else ">5"
        hist[key] = hist.get(key, 0) + 1
    return {
        "n": len(epochs),
        "delta_min_s": min(deltas),
        "delta_p50_s": statistics.median(deltas),
        "delta_max_s": max(deltas),
        "delta_hist": dict(sorted(hist.items())),
        "span_s": epochs[-1] - epochs[0],
    }


def pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    xs = sorted(values)
    return round(xs[min(len(xs) - 1, int(q * (len(xs) - 1)))], 1)


def history_times(resp: dict[str, Any]) -> list[int]:
    return [int(t) for t in ((resp.get("history") or {}).get("times") or [])]


def atomic_write(path: Path, obj: dict[str, Any]) -> None:
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n")
    os.replace(tmp, path)


def merge_result(path: Path, probe_key: str, record: dict[str, Any]) -> None:
    doc: dict[str, Any] = {}
    if path.exists():
        try:
            doc = json.loads(path.read_text())
        except json.JSONDecodeError:
            doc = {}
    doc.setdefault("probe", "deriv_api_phase0")
    doc["issue"] = ISSUE
    doc["phase"] = 0
    doc["script"] = "scripts/deriv_api_probe.py"
    doc["probe_version"] = PROBE_VERSION
    doc["updated_utc"] = now_utc_iso()
    doc.setdefault("runs", {})[probe_key] = record
    doc["decisions"] = recompute_decisions(doc["runs"])
    atomic_write(path, doc)


def recompute_decisions(runs: dict[str, Any]) -> dict[str, Any]:
    """Phase 1/4/5 consumers read this block; every value traces to a probe run."""

    def obs(key: str) -> dict[str, Any]:
        rec = runs.get(key) or {}
        return rec.get("observations") or {} if rec.get("status") == "done" else {}

    a, b, c = obs("a_granularity_enum"), obs("b_tick_history_1s"), obs("c_ticks_subscription")
    d, e, f = obs("d_retention_paging"), obs("e_proposal"), obs("f_rate_budget")
    g, h, i = obs("g_near_close"), obs("h_baseline"), obs("i_candle_latency")

    per_pair_b = b.get("per_pair") or {}
    retention = d.get("per_pair_retention_days") or {}
    return {
        "sub_60s_candles_rejected": a.get("granularity_30_rejected"),
        "tick_1s_contiguous_all_pairs": (
            all(v.get("delta_max_s", 99) <= 2 for v in per_pair_b.values()) if per_pair_b else None
        ),
        "six_symbol_multiplex_one_socket": {
            ep: (v.get("symbols_with_ticks") == len(SYMBOLS)) for ep, v in (c.get("per_endpoint") or {}).items()
        } or None,
        "proposal_subscription_supported": e.get("subscription_supported"),
        "rate_budget_ok_5s_cadence": f.get("sustainable_5s_cadence"),
        "rate_budget_error_rate": f.get("error_rate"),
        "tick_retention_days_min": min(retention.values()) if retention else None,
        "tick_retention_is_lower_bound": bool(d.get("search_floor_hit")) if d else None,
        "candle_publication_latency_p95_s": i.get("poll_latency_p95_s"),
        "candle_subscribe_supported": i.get("subscribe_supported") or None,
        "last_start_cutoff_ny": g.get("cutoff_verdict") or "16:44:59 default stands (probe g pending)",
        "rss_baseline_bytes": h.get("max_rss_bytes"),
    }


def run_probe(key: str, fn: Callable[[argparse.Namespace], dict[str, Any]], args: argparse.Namespace) -> None:
    try:
        ctx = probe_context()
    except Exception as exc:  # e.g. executor import chain unavailable on a minimal host
        ctx = {"started_utc": now_utc_iso(), "context_error": f"{type(exc).__name__}: {exc}"}
    print(f"[{key}] start {ctx['started_utc']} (NY {ctx.get('ny_time', '?')})")
    try:
        record = fn(args)
    except Exception as exc:  # persist the failure; later probes still run
        record = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
    record["context"] = ctx
    record["finished_utc"] = now_utc_iso()
    merge_result(args.out, key, record)
    print(f"[{key}] {record['status']}")


# ---------------------------------------------------------------- probe a


def probe_a(args: argparse.Namespace) -> dict[str, Any]:
    sym = SYMBOLS["USDJPY"]
    per_endpoint: dict[str, Any] = {}
    for ep, url in ENDPOINTS.items():
        with DerivOptionsClient(url, timeout=args.timeout) as client:
            try:
                client.ticks_history(sym, style="candles", granularity=30, count=10)
                rejected, err = False, None
            except DerivAPIError as exc:
                rejected, err = True, str(exc)
            time.sleep(REQUEST_SLEEP_S)
            resp = client.ticks_history(sym, style="candles", granularity=60, count=10)
            epochs = [int(cd["epoch"]) for cd in resp.get("candles") or []]
            spacing = sorted({b - a for a, b in zip(epochs, epochs[1:])})
        per_endpoint[ep] = {
            "granularity_30_rejected": rejected,
            "granularity_30_error": err,
            "granularity_60_n_candles": len(epochs),
            "granularity_60_spacing_s": spacing,
        }
        time.sleep(REQUEST_SLEEP_S)
    return {
        "status": "done",
        "observations": {
            "symbol": sym,
            "per_endpoint": per_endpoint,
            "granularity_30_rejected": all(v["granularity_30_rejected"] for v in per_endpoint.values()),
        },
    }


# ---------------------------------------------------------------- probe b


def probe_b(args: argparse.Namespace) -> dict[str, Any]:
    per_pair: dict[str, Any] = {}
    with DerivOptionsClient.public(timeout=args.timeout) as client:
        for pair, sym in SYMBOLS.items():
            resp = client.ticks_history(sym, style="ticks", count=300, granularity=None)
            times = history_times(resp)
            per_pair[pair] = {"response_keys": sorted(resp.keys()), **delta_stats(times)}
            time.sleep(REQUEST_SLEEP_S)
    return {"status": "done", "observations": {"count_requested": 300, "per_pair": per_pair}}


# ---------------------------------------------------------------- probe c


def probe_c(args: argparse.Namespace) -> dict[str, Any]:
    per_endpoint: dict[str, Any] = {}
    for ep, url in ENDPOINTS.items():
        client = DerivOptionsClient(url, timeout=args.timeout)
        sub_ids: list[str] = []
        sub_errors: dict[str, str] = {}
        tick_epochs: dict[str, list[float]] = {s: [] for s in SYMBOLS.values()}
        tick_fields: dict[str, list[str]] = {}
        other_frames: dict[str, int] = {}
        forget_errors = 0
        capture_error = None
        try:
            for sym in SYMBOLS.values():
                try:
                    resp = client.ticks(sym, subscribe=True)
                    sub_id = (resp.get("subscription") or {}).get("id")
                    if sub_id:
                        sub_ids.append(sub_id)
                    tick = resp.get("tick") or {}
                    if tick.get("symbol") in tick_epochs:
                        tick_epochs[tick["symbol"]].append(float(tick.get("epoch", 0)))
                        tick_fields.setdefault(tick["symbol"], sorted(tick.keys()))
                except DerivAPIError as exc:
                    sub_errors[sym] = str(exc)
                time.sleep(REQUEST_SLEEP_S)
            deadline = time.monotonic() + args.sub_seconds
            while time.monotonic() < deadline:
                try:
                    frame = client.recv(timeout=2.0)
                except RECV_TIMEOUT_EXCS:
                    continue
                if frame.get("msg_type") == "tick" and "tick" in frame:
                    tick = frame["tick"]
                    sym = tick.get("symbol")
                    if sym in tick_epochs:
                        tick_epochs[sym].append(float(tick.get("epoch", 0)))
                        tick_fields.setdefault(sym, sorted(tick.keys()))
                else:
                    other_frames[str(frame.get("msg_type"))] = other_frames.get(str(frame.get("msg_type")), 0) + 1
            for sub_id in sub_ids:
                try:
                    client.forget(sub_id)
                except (DerivAPIError, *RECV_TIMEOUT_EXCS):
                    forget_errors += 1
        except (websocket.WebSocketException, json.JSONDecodeError, DerivAPIError) as exc:
            # keep this endpoint's partial capture and the other endpoint's run
            capture_error = f"{type(exc).__name__}: {exc}"[:300]
        finally:
            client.close()
        per_sym = {
            sym: delta_stats([int(e) for e in epochs]) for sym, epochs in tick_epochs.items()
        }
        per_endpoint[ep] = {
            "capture_seconds": args.sub_seconds,
            "subscribed": len(sub_ids),
            "subscribe_errors": sub_errors,
            "symbols_with_ticks": sum(1 for v in tick_epochs.values() if v),
            "per_symbol": per_sym,
            "tick_fields": tick_fields,
            "other_frames": other_frames,
            "forget_errors": forget_errors,
            "capture_error": capture_error,
        }
    return {"status": "done", "observations": {"per_endpoint": per_endpoint}}


# ---------------------------------------------------------------- probe d


def _has_ticks_at_or_before(client: DerivOptionsClient, sym: str, epoch: int) -> bool:
    # A failed request must NOT read as "no data" — that would silently corrupt
    # the binary search into an understated retention depth. Retry once, then
    # propagate (run_probe records the probe as failed).
    for attempt in (0, 1):
        try:
            resp = client.ticks_history(sym, style="ticks", count=10, end=str(epoch), granularity=None)
            break
        except DerivAPIError:
            if attempt:
                raise
            time.sleep(1.0)
    return bool(history_times(resp))


def probe_d(args: argparse.Namespace) -> dict[str, Any]:
    retention_days: dict[str, float] = {}
    oldest_epochs: dict[str, int] = {}
    floor_hit: list[str] = []
    with DerivOptionsClient.public(timeout=args.timeout) as client:
        for pair, sym in SYMBOLS.items():
            newest = history_times(client.ticks_history(sym, style="ticks", count=10, granularity=None))[-1]
            time.sleep(REQUEST_SLEEP_S)
            lo, hi = EPOCH_2000, newest
            if _has_ticks_at_or_before(client, sym, lo):
                oldest_epochs[pair] = lo  # retention deeper than the search floor: LOWER BOUND, not depth
                floor_hit.append(pair)
            else:
                while hi - lo > 3600:
                    mid = (lo + hi) // 2
                    time.sleep(REQUEST_SLEEP_S)
                    if _has_ticks_at_or_before(client, sym, mid):
                        hi = mid
                    else:
                        lo = mid
                time.sleep(REQUEST_SLEEP_S)
                confirm = history_times(
                    client.ticks_history(sym, style="ticks", count=5000, end=str(hi), granularity=None)
                )
                oldest_epochs[pair] = confirm[0] if confirm else hi
            retention_days[pair] = round((newest - oldest_epochs[pair]) / 86400.0, 1)
            time.sleep(REQUEST_SLEEP_S)

        # Paging mechanics: three consecutive pages must be sorted, disjoint, descending in time.
        sym = SYMBOLS["USDJPY"]
        pages: list[list[int]] = []
        end: str = "latest"
        for _ in range(3):
            times = history_times(client.ticks_history(sym, style="ticks", count=5000, end=end, granularity=None))
            pages.append(times)
            if not times:
                break
            end = str(times[0] - 1)  # syn_collect.py paging: end = oldest_epoch - 1
            time.sleep(REQUEST_SLEEP_S)
    sorted_ok = all(p == sorted(p) for p in pages)
    disjoint_ok = all(pages[k + 1][-1] < pages[k][0] for k in range(len(pages) - 1) if pages[k] and pages[k + 1])
    return {
        "status": "done",
        "observations": {
            "method": "binary search on ticks_history emptiness (1h resolution) + oldest-tick confirm; "
            "full page-to-exhaustion infeasible at 5000 ticks/page over multi-day 1s retention",
            "per_pair_retention_days": retention_days,
            "per_pair_oldest_epoch": oldest_epochs,
            "search_floor_epoch": EPOCH_2000,
            "search_floor_hit": floor_hit,
            "retention_note": "floor-hit pairs: ticks exist at/before the 2000-01-01 search floor; recorded days are a LOWER BOUND",
            "paging_symbol": sym,
            "paging_count_requested": 5000,
            "paging_page_sizes": [len(p) for p in pages],
            "paging_sorted_ascending": sorted_ok,
            "paging_disjoint_ordered": disjoint_ok,
        },
    }


# ---------------------------------------------------------------- probe e


def probe_e(args: argparse.Namespace) -> dict[str, Any]:
    sym = SYMBOLS["USDJPY"]
    with DerivOptionsClient.public(timeout=args.timeout) as client:
        try:
            cf = client.contracts_for(sym)
            avail = (cf.get("contracts_for") or {}).get("available") or []
            contracts_meta = {
                "n_available": len(avail),
                "contract_types": sorted({c.get("contract_type") for c in avail if isinstance(c, dict)}),
                "min_durations": sorted({str(c.get("min_contract_duration")) for c in avail if isinstance(c, dict)}),
                "max_durations": sorted({str(c.get("max_contract_duration")) for c in avail if isinstance(c, dict)}),
                "response_keys": sorted(cf.keys()),
            }
        except DerivAPIError as exc:
            contracts_meta = {"error": str(exc)}
        time.sleep(REQUEST_SLEEP_S)

        resp = client.proposal(symbol=sym, contract_type="CALL", amount=1.0)
        prop = resp.get("proposal") or {}
        proposal_fields = {
            "keys": sorted(prop.keys()),
            "has_id": bool(prop.get("id")),
            "ask_price": prop.get("ask_price"),
            "payout": prop.get("payout"),
        }
        time.sleep(REQUEST_SLEEP_S)

        # Mirror deriv_client.proposal's exact Options-schema payload + subscribe.
        sub_payload = {
            "proposal": 1,
            "amount": 1.0,
            "basis": "stake",
            "contract_type": "CALL",
            "currency": "USD",
            "duration": 15,
            "duration_unit": "m",
            "underlying_symbol": sym,
            "subscribe": 1,
        }
        try:
            sub_resp = client.request(sub_payload)
            sub_id = (sub_resp.get("subscription") or {}).get("id")
            updates = 0
            deadline = time.monotonic() + 15.0
            while time.monotonic() < deadline:
                try:
                    frame = client.recv(timeout=2.0)
                except RECV_TIMEOUT_EXCS:
                    continue
                if frame.get("msg_type") == "proposal":
                    updates += 1
            # forget failure must not overturn the measured capture verdict
            forget_error = None
            if sub_id:
                try:
                    client.forget(sub_id)
                except (DerivAPIError, *RECV_TIMEOUT_EXCS) as exc:
                    forget_error = str(exc)[:200]
            subscription = {
                "subscription_supported": True,
                "subscription_id_present": bool(sub_id),
                "updates_in_15s": updates,
                "forget_error": forget_error,
            }
        except DerivAPIError as exc:
            subscription = {"subscription_supported": False, "error": str(exc)}
    return {
        "status": "done",
        "observations": {"symbol": sym, "contracts_for": contracts_meta, "proposal": proposal_fields, **subscription},
    }


# ---------------------------------------------------------------- probe f


def probe_f(args: argparse.Namespace) -> dict[str, Any]:
    """Worst-case daemon cycle: 12 proposals (6 pairs x CALL/PUT) + 6 candle fetches per 5s."""
    latencies_ms: list[float] = []
    cycle_wall_s: list[float] = []
    errors: list[str] = []
    ok = 0
    consecutive_errors = 0
    aborted = None
    with DerivOptionsClient.public(timeout=args.timeout) as client:
        for _ in range(args.soak_cycles):
            cycle_start = time.monotonic()
            for sym in SYMBOLS.values():
                for contract_type in ("CALL", "PUT"):
                    t0 = time.monotonic()
                    try:
                        client.proposal(symbol=sym, contract_type=contract_type, amount=1.0)
                        ok += 1
                        consecutive_errors = 0
                    except DerivAPIError as exc:
                        errors.append(f"proposal {sym} {contract_type}: {exc}"[:300])
                        consecutive_errors += 1
                    latencies_ms.append((time.monotonic() - t0) * 1000)
            for sym in SYMBOLS.values():
                t0 = time.monotonic()
                try:
                    client.ticks_history(sym, style="candles", granularity=60, count=3)
                    ok += 1
                    consecutive_errors = 0
                except DerivAPIError as exc:
                    errors.append(f"candles {sym}: {exc}"[:300])
                    consecutive_errors += 1
                latencies_ms.append((time.monotonic() - t0) * 1000)
            elapsed = time.monotonic() - cycle_start
            cycle_wall_s.append(round(elapsed, 2))
            if consecutive_errors >= 6:
                aborted = f"aborted after {consecutive_errors} consecutive errors"
                break
            time.sleep(max(0.0, args.soak_cadence - elapsed))
    total = ok + len(errors)
    err_rate = len(errors) / total if total else 1.0
    return {
        "status": "done",
        "observations": {
            "cycles_completed": len(cycle_wall_s),
            "requests_total": total,
            "requests_ok": ok,
            "error_count": len(errors),
            "error_rate": round(err_rate, 4),
            "errors_sample": errors[:5],
            "aborted": aborted,
            "request_latency_ms": {"p50": pct(latencies_ms, 0.5), "p95": pct(latencies_ms, 0.95), "max": pct(latencies_ms, 1.0)},
            "cycle_wall_s": {"p50": pct(cycle_wall_s, 0.5), "p95": pct(cycle_wall_s, 0.95), "max": pct(cycle_wall_s, 1.0)},
            "transport_note": "serial sync client, one connection (deriv_client.py); the async Phase-1 client parallelizes",
            "sustainable_5s_cadence": bool(err_rate < 0.01 and aborted is None and (pct(cycle_wall_s, 0.95) or 99) <= args.soak_cadence),
        },
    }


# ---------------------------------------------------------------- probe g


def probe_g(args: argparse.Namespace) -> dict[str, Any]:
    now = ny_now()
    hour = now.hour + now.minute / 60.0
    # Entry must be BEFORE 17:00 so at least one pre-close attempt exists; a
    # verdict backed by zero attempts must never read as measured.
    if not (16 + 35 / 60 <= hour < 17.0):
        return {
            "status": "pending",
            "reason": f"near-close window is 16:35-16:59 NY; now {now.strftime('%H:%M:%S')} NY",
            "rerun": "~/binary-algo-venv/bin/python scripts/deriv_api_probe.py --probes g --out results/json/deriv_api_probe_result.json",
        }
    sym = SYMBOLS["USDJPY"]
    attempts: list[dict[str, Any]] = []
    with DerivOptionsClient.public(timeout=args.timeout) as client:
        while True:
            t = ny_now()
            if t.hour >= 17 and t.minute >= 1:
                break
            try:
                resp = client.proposal(symbol=sym, contract_type="CALL", amount=1.0)
                attempts.append({"ny_time": t.strftime("%H:%M:%S"), "ok": True, "payout": (resp.get("proposal") or {}).get("payout")})
            except DerivAPIError as exc:
                attempts.append({"ny_time": t.strftime("%H:%M:%S"), "ok": False, "error": str(exc)[:300]})
            time.sleep(30.0)
    if not attempts:
        return {"status": "pending", "reason": "no proposal attempts completed before 17:01 NY; rerun earlier in the window"}
    rejected = [a for a in attempts if not a["ok"]]
    verdict = (
        f"first rejection at {rejected[0]['ny_time']} NY" if rejected else "no rejection observed through 17:00 NY; 16:44:59 default stands (model-consistency rule)"
    )
    return {"status": "done", "observations": {"symbol": sym, "attempts": attempts, "cutoff_verdict": verdict}}


# ---------------------------------------------------------------- probe h


def probe_h(args: argparse.Namespace) -> dict[str, Any]:
    if not args.baseline_store_dir:
        return {
            "status": "pending",
            "reason": "run on the VPS during NY session with a fresh store",
            "rerun": "~/binary-algo-venv/bin/python scripts/deriv_api_probe.py --probes h --baseline-store-dir $DERIV_DEMO_STORE_DIR --out results/json/deriv_api_probe_result.json",
        }
    cmd = [
        "/usr/bin/time", "-v", sys.executable, "scripts/deriv_demo_executor.py",
        "--pairs", "all-enabled", "--once", "--store-dir", args.baseline_store_dir,
        "--log-dir", "logs/paper_trades",
    ]
    t0 = time.monotonic()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    wall_s = round(time.monotonic() - t0, 1)
    rss_kb = None
    for line in proc.stderr.splitlines():
        if "Maximum resident set size" in line:
            rss_kb = int(line.rsplit(":", 1)[1].strip())
    return {
        "status": "done" if proc.returncode == 0 and rss_kb else "failed",
        "observations": {
            "command": " ".join(cmd),
            "returncode": proc.returncode,
            "wall_s": wall_s,
            "max_rss_bytes": rss_kb * 1024 if rss_kb else None,
            "in_ny_session": in_ny_session_now(),
            "stderr_tail": proc.stderr.strip().splitlines()[-3:],
        },
    }


# ---------------------------------------------------------------- probe i


def probe_i(args: argparse.Namespace) -> dict[str, Any]:
    sym = SYMBOLS["USDJPY"]
    poll_latencies: list[float] = []
    finality: list[dict[str, Any]] = []
    with DerivOptionsClient(DERIV_LEGACY_WS_URL, timeout=args.timeout) as client:
        for _ in range(args.latency_minutes):
            boundary = (int(time.time()) // 60 + 1) * 60
            time.sleep(max(0.0, boundary - time.time() - 1.0))
            seen: dict[str, Any] | None = None
            deadline = boundary + 15.0
            while time.time() < deadline:
                resp = client.ticks_history(sym, style="candles", granularity=60, count=3)
                candles = resp.get("candles") or []
                if any(int(cd["epoch"]) == boundary for cd in candles):
                    seen = next((cd for cd in candles if int(cd["epoch"]) == boundary - 60), None)
                    poll_latencies.append(round(time.time() - boundary, 2))
                    break
                time.sleep(0.25)
            if seen is not None:
                time.sleep(5.0)
                resp = client.ticks_history(sym, style="candles", granularity=60, count=3)
                later = next((cd for cd in (resp.get("candles") or []) if int(cd["epoch"]) == boundary - 60), None)
                finality.append({
                    "boundary_epoch": boundary,
                    "close_at_first_sight": seen.get("close"),
                    "close_5s_later": later.get("close") if later else None,
                    "changed": bool(later and later.get("close") != seen.get("close")),
                })

    # Candle subscription support, both endpoints.
    subscribe_supported: dict[str, Any] = {}
    for ep, url in ENDPOINTS.items():
        client = DerivOptionsClient(url, timeout=args.timeout)
        try:
            payload = {"ticks_history": sym, "style": "candles", "granularity": 60, "count": 3,
                       "end": "latest", "adjust_start_time": 1, "subscribe": 1}
            resp = client.request(payload)
            sub_id = (resp.get("subscription") or {}).get("id")
            ohlc_frames = 0
            deadline = time.monotonic() + 15.0
            while time.monotonic() < deadline:
                try:
                    frame = client.recv(timeout=2.0)
                except RECV_TIMEOUT_EXCS:
                    continue
                if frame.get("msg_type") == "ohlc":
                    ohlc_frames += 1
            # forget failure must not overturn the measured capture verdict
            forget_error = None
            if sub_id:
                try:
                    client.forget(sub_id)
                except (DerivAPIError, *RECV_TIMEOUT_EXCS) as exc:
                    forget_error = str(exc)[:200]
            subscribe_supported[ep] = {"supported": True, "ohlc_frames_in_15s": ohlc_frames, "forget_error": forget_error}
        except DerivAPIError as exc:
            subscribe_supported[ep] = {"supported": False, "error": str(exc)[:300]}
        except (websocket.WebSocketException, json.JSONDecodeError) as exc:
            # transport loss is UNKNOWN support, not "unsupported"
            subscribe_supported[ep] = {"supported": None, "transport_error": f"{type(exc).__name__}: {exc}"[:300]}
        finally:
            client.close()
        time.sleep(REQUEST_SLEEP_S)
    return {
        "status": "done",
        "observations": {
            "symbol": sym,
            "poll_endpoint": "legacy",
            "boundaries_attempted": args.latency_minutes,
            "boundaries_missed_15s_deadline": args.latency_minutes - len(poll_latencies),
            "poll_latency_samples_s": poll_latencies,
            "poll_latency_p95_s": pct(poll_latencies, 0.95),
            "completed_bar_finality": finality,
            "subscribe_supported": subscribe_supported,
        },
    }


# ---------------------------------------------------------------- main


PROBES: dict[str, tuple[str, Callable[[argparse.Namespace], dict[str, Any]]]] = {
    "a": ("a_granularity_enum", probe_a),
    "b": ("b_tick_history_1s", probe_b),
    "c": ("c_ticks_subscription", probe_c),
    "d": ("d_retention_paging", probe_d),
    "e": ("e_proposal", probe_e),
    "f": ("f_rate_budget", probe_f),
    "g": ("g_near_close", probe_g),
    "h": ("h_baseline", probe_h),
    "i": ("i_candle_latency", probe_i),
}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--probes", default=UNATTENDED_PROBES, help=f"comma list of {','.join(PROBES)} (default: unattended set)")
    p.add_argument("--out", type=Path, default=RESULT_PATH)
    p.add_argument("--timeout", type=float, default=15.0)
    p.add_argument("--sub-seconds", type=float, default=90.0, help="probe c capture window per endpoint")
    p.add_argument("--soak-cycles", type=int, default=24, help="probe f cycles (24 x 5s = 2 min)")
    p.add_argument("--soak-cadence", type=float, default=5.0)
    p.add_argument("--latency-minutes", type=int, default=4, help="probe i minute boundaries to sample")
    p.add_argument("--baseline-store-dir", default=None, help="probe h: store dir for the timed one-shot")
    args = p.parse_args()

    requested = [s.strip() for s in args.probes.split(",") if s.strip()]
    unknown = [s for s in requested if s not in PROBES]
    if unknown:
        p.error(f"unknown probes: {unknown}")
    args.out.parent.mkdir(parents=True, exist_ok=True)  # --out results/json/... works from repo root
    for letter in requested:
        key, fn = PROBES[letter]
        run_probe(key, fn, args)
    print(f"result: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
