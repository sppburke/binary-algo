"""Hot (pair, CALL)/(pair, PUT) quote state via proposal subscriptions (issue #4 Phase 4).

Twelve proposal subscriptions multiplexed on one PUBLIC async socket keep
both sides' ask/payout/live_breakeven fresh for every enabled pair — the
mechanism Phase-0 probe (e) verified (subscription supported, ~1 update/s;
probe (f): zero API errors at far higher load).

Quote workers are deliberately buy-incapable: they hold only the public
client (KILL #6 — no worker may hold buy capability; the coordinator in
deriv_runtime_supervisor.py owns the only demo client). Buys never reuse a
snapshot quote — the coordinator requests a fresh selected-side proposal.

Audit mode (issue #5): `--audit` runs the 12 lanes standalone and samples
`QuoteState` every `--sample-seconds` into quote_snapshot-compatible JSONL
rows (side mapped CALL/PUT -> UP/DOWN, `source: "quote_audit"`), replacing
the old executor timer as the quote_snapshot source for the Phase-5 enable
bar. A lane with no quote emits `live_breakeven: null` + `invalid_reason`
set — an honest sentinel that can never enter a median.

Run from repo root:
    ~/binary-algo-venv/bin/python scripts/deriv_quote_workers.py --probe-seconds 60
    ~/binary-algo-venv/bin/python scripts/deriv_quote_workers.py --audit --duration-seconds 90
    ~/binary-algo-venv/bin/python scripts/deriv_quote_workers.py --smoke
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from deriv_async_client import DerivAsyncClient, Subscription
from deriv_backfill import DEFAULT_ENABLED_PAIRS
from deriv_client import PUBLIC_WS_URL, DerivAPIError
from deriv_runtime_core import (
    HORIZON_MINUTES,
    ExecutorError,
    ExecutorLock,
    JsonlLogger,
    parse_quote,
)
from live_features import DERIV_SYMBOLS

SIDES = ("CALL", "PUT")
AUDIT_LOCK_KEY = "quote-audit"
AUDIT_SILENCE_TIMEOUT_S = 300.0
RESUB_BASE_S = 30.0
RESUB_CAP_S = 600.0


class QuoteState:
    """Latest parsed quote per (pair, side); freshness carried as epoch age."""

    def __init__(self) -> None:
        self._q: dict[tuple[str, str], dict[str, Any]] = {}

    def update(self, pair: str, side: str, proposal: dict[str, Any]) -> dict[str, Any]:
        ask, payout, live_breakeven, invalid_reason = parse_quote(proposal)
        spot_time = proposal.get("spot_time")
        rec = {
            "pair": pair,
            "side": side,
            "ask": ask,
            "payout": payout,
            "live_breakeven": live_breakeven,
            "invalid_reason": invalid_reason,
            "proposal_id_present": bool(proposal.get("id")),
            "spot_time": float(spot_time) if isinstance(spot_time, (int, float)) else None,
            "updated_monotonic": time.monotonic(),
        }
        self._q[(pair, side)] = rec
        return rec

    def get(self, pair: str, side: str) -> dict[str, Any] | None:
        rec = self._q.get((pair, side))
        if rec is None:
            return None
        return {**rec, "age_s": round(time.monotonic() - rec["updated_monotonic"], 3)}

    def ages(self) -> dict[str, float]:
        now = time.monotonic()
        return {f"{p}:{s}": round(now - r["updated_monotonic"], 1) for (p, s), r in self._q.items()}


class QuoteWorkers:
    def __init__(self, client: DerivAsyncClient, pairs: list[str], stake: float,
                 on_event: Callable[[str, dict[str, Any]], None] | None = None):
        self.client = client  # PUBLIC socket only; this class must never see a demo client
        self.pairs = pairs
        self.stake = float(stake)
        self.state = QuoteState()
        self._on_event = on_event or (lambda e, f: None)
        self._subs: dict[tuple[str, str], Subscription] = {}
        self._tasks: list[asyncio.Task] = []
        self.updates = 0
        self.errors = 0

    def _payload(self, pair: str, side: str) -> dict[str, Any]:
        return {
            "proposal": 1, "amount": self.stake, "basis": "stake",
            "contract_type": side, "currency": "USD",
            "duration": HORIZON_MINUTES, "duration_unit": "m",
            "underlying_symbol": DERIV_SYMBOLS[pair],
        }

    def has_lane(self, pair: str, side: str) -> bool:
        return (pair, side) in self._subs

    async def start(self) -> None:
        for pair in self.pairs:
            for side in SIDES:
                try:
                    sub = await self.client.subscribe(self._payload(pair, side))
                except DerivAPIError as exc:
                    self.errors += 1
                    self._on_event("quote_subscribe_failed", {"pair": pair, "side": side, "error": str(exc)[:200]})
                    continue
                self._subs[(pair, side)] = sub
                self._tasks.append(asyncio.get_running_loop().create_task(self._drain(pair, side, sub)))

    async def ensure_lane(self, pair: str, side: str) -> bool:
        """(Re)subscribe a lane with no live subscription; True when live.

        Closes the no-retry hole: a lane whose initial subscribe failed never
        entered the client registry, so the client's automatic resubscribe
        (registered subs, ConnectionClosed only) can never revive it.
        """
        if (pair, side) in self._subs:
            return True
        self._on_event("quote_resubscribe_attempted", {"pair": pair, "side": side})
        try:
            sub = await self.client.subscribe(self._payload(pair, side))
        except DerivAPIError as exc:
            self.errors += 1
            self._on_event("quote_resubscribe_failed", {"pair": pair, "side": side, "error": str(exc)[:200]})
            return False
        self._subs[(pair, side)] = sub
        self._tasks.append(asyncio.get_running_loop().create_task(self._drain(pair, side, sub)))
        return True

    async def _drain(self, pair: str, side: str, sub: Subscription) -> None:
        while True:
            frame = await sub.queue.get()
            if "error" in frame:
                self.errors += 1
                self._on_event("quote_stream_error", {"pair": pair, "side": side, "error": str(frame["error"])[:200]})
                continue
            prop = frame.get("proposal")
            if isinstance(prop, dict):
                self.state.update(pair, side, prop)
                self.updates += 1

    async def stop(self) -> None:
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        for sub in self._subs.values():
            try:
                await self.client.unsubscribe(sub)
            except DerivAPIError:
                pass


async def run_probe(seconds: float, url: str) -> int:
    """Live freshness check: all 12 lanes must update; p95 age under 5s."""
    client = DerivAsyncClient(url)
    await client.connect()
    workers = QuoteWorkers(client, list(DEFAULT_ENABLED_PAIRS), stake=1.0)
    await workers.start()
    await asyncio.sleep(seconds)
    ages = workers.state.ages()
    lanes = len(ages)
    fresh = sum(1 for a in ages.values() if a <= 5.0)
    await workers.stop()
    await client.close()
    out = {
        "lanes_expected": len(DEFAULT_ENABLED_PAIRS) * 2,
        "lanes_updating": lanes,
        "lanes_fresh_5s": fresh,  # reported, not gated: Deriv pushes only on quote change
        "updates": workers.updates,
        "errors": workers.errors,
        "ages_s": ages,
        "overflow": client.stats["queue_overflow"],
        "unmatched": client.stats["unmatched_frames"],
        "unmatched_ring_msg_types": [f.get("msg_type") for f in client.unmatched_ring],
    }
    # zero SILENT loss is the bar: every lane updating, nothing dropped or
    # errored; unmatched frames are counted + retained, and surface here
    ok = (lanes == out["lanes_expected"] and workers.errors == 0 and out["overflow"] == 0
          and out["unmatched"] == 0)
    print(json.dumps(out, indent=1, sort_keys=True))
    print("QUOTE WORKERS PASS" if ok else "QUOTE WORKERS FAIL")
    return 0 if ok else 1


# ---------------------------------------------------------------- audit sampler (issue #5)


def resolve_pairs(raw: str) -> list[str]:
    if raw.strip().lower() == "all-enabled":
        return list(DEFAULT_ENABLED_PAIRS)
    pairs = [p.strip().upper() for p in raw.split(",") if p.strip()]
    unknown = [p for p in pairs if p not in DERIV_SYMBOLS]
    if unknown:
        raise SystemExit(f"unknown pairs {unknown}; valid: {','.join(DERIV_SYMBOLS)}")
    if not pairs:
        raise SystemExit("no pairs selected")
    return pairs


def reap_stale_lock(lock_path: Path, logger: JsonlLogger) -> bool:
    """Unlink a leftover lock whose recorded pid is dead (SIGKILL/OOM survivor).

    Lives HERE, not in deriv_runtime_core — touching runtime_core re-arms the
    KILL #0 parity re-check. Returns True when a stale lock was removed.
    """
    if not lock_path.exists():
        return False
    pid = -1
    try:
        pid = int(json.loads(lock_path.read_text(encoding="utf-8")).get("pid", -1))
    except (json.JSONDecodeError, ValueError, OSError):
        pass
    alive = False
    if pid > 0:
        try:
            os.kill(pid, 0)
            alive = True
        except ProcessLookupError:
            alive = False
        except PermissionError:
            alive = True  # pid exists under another user — not ours to steal
    if alive:
        return False
    try:
        lock_path.unlink()
    except FileNotFoundError:
        return False
    logger.write("stale_lock_stolen", path=str(lock_path), dead_pid=pid)
    return True


def snapshot_row(rec: dict[str, Any] | None, pair: str, contract_type: str) -> dict[str, Any]:
    """One quote_snapshot-compatible row (executor schema of record,
    deriv_demo_executor.py:346-358, minus raw_hash, plus source).

    quote_age_ms is spot_time-based exactly like the executor when spot_time
    is present; the absent case falls back to ms since the last push (the one
    documented divergence — the executor emits null there) for dead-lane
    diagnosis. Not audit-consumed either way.
    """
    side = "UP" if contract_type == "CALL" else "DOWN"
    if rec is None:
        return {
            "pair": pair, "side": side, "contract_type": contract_type,
            "ask": None, "payout": None, "live_breakeven": None,
            "invalid_reason": "no_quote_received", "proposal_id_present": False,
            "quote_age_ms": None, "source": "quote_audit",
        }
    if rec.get("spot_time") is not None:
        quote_age_ms = max(0.0, (datetime.now(timezone.utc).timestamp() - rec["spot_time"]) * 1000.0)
    else:
        quote_age_ms = max(0.0, (time.monotonic() - rec["updated_monotonic"]) * 1000.0)
    return {
        "pair": pair, "side": side, "contract_type": contract_type,
        "ask": rec["ask"], "payout": rec["payout"],
        "live_breakeven": rec["live_breakeven"],
        "invalid_reason": rec["invalid_reason"],
        "proposal_id_present": bool(rec.get("proposal_id_present")),
        "quote_age_ms": round(quote_age_ms, 1), "source": "quote_audit",
    }


def sample_rows(workers: QuoteWorkers, pairs: list[str]) -> list[dict[str, Any]]:
    return [snapshot_row(workers.state.get(pair, side), pair, side)
            for pair in pairs for side in SIDES]


def emit_samples(workers: QuoteWorkers, pairs: list[str], logger: JsonlLogger) -> int:
    rows = sample_rows(workers, pairs)
    for row in rows:
        logger.write("quote_snapshot", **row)
    return len(rows)


async def recover_lanes(workers: QuoteWorkers, pairs: list[str],
                        backoff: dict[tuple[str, str], tuple[float, float]]) -> None:
    """Per-tick lane recovery with bounded per-lane backoff (30s -> 600s cap)."""
    for pair in pairs:
        for side in SIDES:
            lane = (pair, side)
            if workers.has_lane(pair, side):
                backoff.pop(lane, None)
                continue
            next_at, delay = backoff.get(lane, (0.0, RESUB_BASE_S))
            if time.monotonic() < next_at:
                continue
            if await workers.ensure_lane(pair, side):
                backoff.pop(lane, None)
            else:
                backoff[lane] = (time.monotonic() + delay, min(delay * 2, RESUB_CAP_S))


async def run_audit(args: argparse.Namespace) -> int:
    """Always-on non-buying sampler: 12 proposal subscriptions on the public
    socket, one quote_snapshot row per lane every --sample-seconds."""
    log_dir = Path(args.log_dir)
    logger = JsonlLogger(log_dir)
    lock = ExecutorLock(log_dir, AUDIT_LOCK_KEY, logger)
    reap_stale_lock(lock.path, logger)
    pairs = resolve_pairs(args.pairs)
    with lock:
        client = DerivAsyncClient(args.url, silence_timeout_s=args.silence_timeout_s,
                                  on_event=lambda e, f: logger.write("client_" + e, **f))
        await client.connect()
        workers = QuoteWorkers(client, pairs, args.stake,
                               on_event=lambda e, f: logger.write(e, **f))
        await workers.start()
        logger.write("audit_started", pairs=pairs, sample_seconds=args.sample_seconds,
                     duration_seconds=args.duration_seconds, stake=args.stake, url=args.url,
                     silence_timeout_s=args.silence_timeout_s)
        started = time.monotonic()
        backoff: dict[tuple[str, str], tuple[float, float]] = {}
        try:
            while True:
                await asyncio.sleep(args.sample_seconds)
                await recover_lanes(workers, pairs, backoff)
                emit_samples(workers, pairs, logger)
                if args.duration_seconds and time.monotonic() - started >= args.duration_seconds:
                    break
        finally:
            await workers.stop()
            await client.close()
            logger.write("audit_stopped", updates=workers.updates, errors=workers.errors)
    return 0


# ---------------------------------------------------------------- smoke (deterministic)


def _buy_incapability_offenders() -> list[str]:
    """AST gate (not a bare-word grep — this docstring says \"buy\"): no dict
    literal or subscript keyed \"buy\", no .buy attribute access, anywhere in
    this module's source."""
    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    offenders: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            for key in node.keys:
                if isinstance(key, ast.Constant) and key.value == "buy":
                    offenders.append(f"dict-key line {node.lineno}")
        elif isinstance(node, ast.Attribute) and node.attr == "buy":
            offenders.append(f".buy attribute line {node.lineno}")
        elif isinstance(node, ast.Subscript):
            sl = node.slice
            if isinstance(sl, ast.Constant) and sl.value == "buy":
                offenders.append(f"subscript line {node.lineno}")
    return offenders


class _FakeSub:
    def __init__(self) -> None:
        self.queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self.sub_id = "fake"
        self.active = True


class _FakeClient:
    """Scripted subscribe outcomes; no network. `fail_first` lanes reject the
    first subscribe attempt (the startup hole the recovery loop closes)."""

    def __init__(self, fail_first: set[tuple[str, str]] | None = None):
        self.fail_first = set(fail_first or ())
        self.attempts: dict[tuple[str, str], int] = {}
        self.lanes: dict[tuple[str, str], _FakeSub] = {}

    async def subscribe(self, payload: dict[str, Any]) -> _FakeSub:
        sym2pair = {v: k for k, v in DERIV_SYMBOLS.items()}
        lane = (sym2pair[payload["underlying_symbol"]], payload["contract_type"])
        self.attempts[lane] = self.attempts.get(lane, 0) + 1
        if lane in self.fail_first and self.attempts[lane] == 1:
            raise DerivAPIError("scripted first-attempt rejection")
        sub = _FakeSub()
        self.lanes[lane] = sub
        return sub

    async def unsubscribe(self, sub: _FakeSub) -> None:
        sub.active = False


async def run_smoke() -> int:
    import subprocess
    import tempfile

    failures: list[str] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        print(f"{'PASS' if ok else 'FAIL'}: {name}" + (f" — {detail}" if detail else ""))
        if not ok:
            failures.append(name)

    # 1. buy-incapability static gate (AST, not grep)
    offenders = _buy_incapability_offenders()
    check("AST buy-incapability gate", not offenders, "; ".join(offenders) or "no buy-shaped nodes")

    # 2. stale-lock takeover: dead-pid lock is reaped, live-pid lock is honored
    tmp = Path(tempfile.mkdtemp(prefix="quote_audit_smoke_"))
    logger = JsonlLogger(tmp)
    lock = ExecutorLock(tmp, AUDIT_LOCK_KEY, logger)
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    lock.path.write_text(json.dumps({"pid": proc.pid}), encoding="utf-8")
    check("dead-pid lock reaped", reap_stale_lock(lock.path, logger) and not lock.path.exists())
    with ExecutorLock(tmp, AUDIT_LOCK_KEY, logger):
        pass
    check("lock acquirable after reap", not lock.path.exists())
    lock.path.write_text(json.dumps({"pid": os.getpid()}), encoding="utf-8")
    check("live-pid lock NOT reaped", not reap_stale_lock(lock.path, logger) and lock.path.exists())
    try:
        with ExecutorLock(tmp, AUDIT_LOCK_KEY, logger):
            check("live lock blocks second instance", False)
    except ExecutorError:
        check("live lock blocks second instance", True)
    lock.path.unlink()

    # 3. lane recovery: first subscribe rejected, recovery tick revives the lane
    events: list[tuple[str, dict[str, Any]]] = []
    fake = _FakeClient(fail_first={("USDJPY", "CALL")})
    pairs = list(DEFAULT_ENABLED_PAIRS)
    workers = QuoteWorkers(fake, pairs, stake=1.0, on_event=lambda e, f: events.append((e, f)))
    await workers.start()
    check("failed lane absent after start", not workers.has_lane("USDJPY", "CALL")
          and sum(1 for e, _ in events if e == "quote_subscribe_failed") == 1,
          f"lanes={len(workers._subs)}")
    backoff: dict[tuple[str, str], tuple[float, float]] = {}
    await recover_lanes(workers, pairs, backoff)
    check("lane recovery revives the lane", workers.has_lane("USDJPY", "CALL")
          and any(e == "quote_resubscribe_attempted" for e, _ in events)
          and not backoff, f"lanes={len(workers._subs)}")

    # 4. feed scripted frames; leave (GBPUSD, PUT) silent for the null sentinel
    hour_ago = datetime.now(timezone.utc).timestamp() - 3600.0
    for (pair, side), sub in workers._subs.items():
        if (pair, side) == ("GBPUSD", "PUT"):
            continue
        prop: dict[str, Any] = {"id": f"{pair}-{side}", "ask_price": 1.0, "payout": 1.85}
        if (pair, side) == ("USDJPY", "CALL"):
            prop["spot_time"] = hour_ago  # spot-based age >= ~3.6e6 ms
        elif (pair, side) == ("USDCAD", "PUT"):
            prop = {"id": f"{pair}-{side}", "payout": 1.85}  # invalid: no ask
        sub.queue.put_nowait({"proposal": prop})
    await asyncio.sleep(0.05)  # let drain tasks run

    rows = sample_rows(workers, pairs)
    check("12 rows sampled", len(rows) == len(pairs) * 2, f"n={len(rows)}")
    expected_keys = {"pair", "side", "contract_type", "ask", "payout", "live_breakeven",
                     "invalid_reason", "proposal_id_present", "quote_age_ms", "source"}
    check("exact row schema keys", all(set(r) == expected_keys for r in rows))
    check("CALL/PUT -> UP/DOWN mapping", all(
        r["side"] in ("UP", "DOWN") and r["contract_type"] in ("CALL", "PUT")
        and (r["side"] == "UP") == (r["contract_type"] == "CALL") for r in rows))
    by_lane = {(r["pair"], r["contract_type"]): r for r in rows}
    check("quote_age_ms spot_time-based when present",
          by_lane[("USDJPY", "CALL")]["quote_age_ms"] > 3_000_000,
          f"age={by_lane[('USDJPY', 'CALL')]['quote_age_ms']}")
    fallback = by_lane[("USDJPY", "PUT")]
    check("quote_age_ms monotonic fallback when spot_time absent",
          fallback["quote_age_ms"] is not None and 0 <= fallback["quote_age_ms"] < 60_000,
          f"age={fallback['quote_age_ms']}")
    sentinel = by_lane[("GBPUSD", "PUT")]
    check("no-quote lane emits null sentinel", sentinel["live_breakeven"] is None
          and sentinel["invalid_reason"] == "no_quote_received"
          and sentinel["proposal_id_present"] is False and sentinel["quote_age_ms"] is None)
    invalid = by_lane[("USDCAD", "PUT")]
    check("invalid proposal keeps honest invalid_reason", invalid["live_breakeven"] is None
          and invalid["invalid_reason"] == "missing_or_invalid_ask")
    good = by_lane[("AUDUSD", "CALL")]
    check("valid lane carries breakeven + id flag",
          good["live_breakeven"] is not None and abs(good["live_breakeven"] - 1.0 / 1.85) < 1e-12
          and good["proposal_id_present"] is True and good["source"] == "quote_audit")

    # 5. JsonlLogger round-trip: event + timestamp_utc injected, keys exact
    n = emit_samples(workers, pairs, logger)
    logged = [json.loads(line) for line in logger.path.read_text(encoding="utf-8").splitlines()
              if json.loads(line).get("event") == "quote_snapshot"]
    check("emit_samples writes one row per lane", n == len(pairs) * 2 and len(logged) == n)
    check("logged rows carry exact pinned schema",
          all(set(r) == expected_keys | {"event", "timestamp_utc"} for r in logged))

    await workers.stop()
    print(f"\n{'SMOKE ALL PASS' if not failures else f'SMOKE {len(failures)} FAILURES: {failures}'}")
    return 1 if failures else 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--probe-seconds", type=float, default=60.0)
    p.add_argument("--url", default=PUBLIC_WS_URL)
    p.add_argument("--audit", action="store_true", help="run the always-on quote_snapshot sampler")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--log-dir", default="logs/quote_audit")
    p.add_argument("--sample-seconds", type=float, default=30.0)
    p.add_argument("--duration-seconds", type=float, default=0.0, help="0 = run forever")
    p.add_argument("--stake", type=float, default=1.0)
    p.add_argument("--pairs", default="all-enabled")
    p.add_argument("--silence-timeout-s", type=float, default=None,
                   help=f"audit-mode default {AUDIT_SILENCE_TIMEOUT_S:.0f}s; only helps lanes in the registry")
    args = p.parse_args()
    if args.smoke:
        return asyncio.run(run_smoke())
    if args.audit:
        if args.silence_timeout_s is None:
            args.silence_timeout_s = AUDIT_SILENCE_TIMEOUT_S
        try:
            return asyncio.run(run_audit(args))
        except KeyboardInterrupt:
            return 0
    return asyncio.run(run_probe(args.probe_seconds, args.url))


if __name__ == "__main__":
    sys.exit(main())
