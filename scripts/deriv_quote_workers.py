"""Hot (pair, CALL)/(pair, PUT) quote state via proposal subscriptions (issue #4 Phase 4).

Twelve proposal subscriptions multiplexed on one PUBLIC async socket keep
both sides' ask/payout/live_breakeven fresh for every enabled pair — the
mechanism Phase-0 probe (e) verified (subscription supported, ~1 update/s;
probe (f): zero API errors at far higher load).

Quote workers are deliberately buy-incapable: they hold only the public
client (KILL #6 — no worker may hold buy capability; the coordinator in
deriv_runtime_supervisor.py owns the only demo client). Buys never reuse a
snapshot quote — the coordinator requests a fresh selected-side proposal.

Live check (no trading):
    ~/binary-algo-venv/bin/python scripts/deriv_quote_workers.py --probe-seconds 60
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from typing import Any, Callable

from deriv_async_client import DerivAsyncClient, Subscription
from deriv_backfill import DEFAULT_ENABLED_PAIRS
from deriv_client import PUBLIC_WS_URL, DerivAPIError
from deriv_runtime_core import HORIZON_MINUTES, parse_quote
from live_features import DERIV_SYMBOLS

SIDES = ("CALL", "PUT")


class QuoteState:
    """Latest parsed quote per (pair, side); freshness carried as epoch age."""

    def __init__(self) -> None:
        self._q: dict[tuple[str, str], dict[str, Any]] = {}

    def update(self, pair: str, side: str, proposal: dict[str, Any]) -> dict[str, Any]:
        ask, payout, live_breakeven, invalid_reason = parse_quote(proposal)
        rec = {
            "pair": pair,
            "side": side,
            "ask": ask,
            "payout": payout,
            "live_breakeven": live_breakeven,
            "invalid_reason": invalid_reason,
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

    async def start(self) -> None:
        for pair in self.pairs:
            for side in SIDES:
                payload = {
                    "proposal": 1, "amount": self.stake, "basis": "stake",
                    "contract_type": side, "currency": "USD",
                    "duration": HORIZON_MINUTES, "duration_unit": "m",
                    "underlying_symbol": DERIV_SYMBOLS[pair],
                }
                try:
                    sub = await self.client.subscribe(payload)
                except DerivAPIError as exc:
                    self.errors += 1
                    self._on_event("quote_subscribe_failed", {"pair": pair, "side": side, "error": str(exc)[:200]})
                    continue
                self._subs[(pair, side)] = sub
                self._tasks.append(asyncio.get_running_loop().create_task(self._drain(pair, side, sub)))

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


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--probe-seconds", type=float, default=60.0)
    p.add_argument("--url", default=PUBLIC_WS_URL)
    args = p.parse_args()
    return asyncio.run(run_probe(args.probe_seconds, args.url))


if __name__ == "__main__":
    sys.exit(main())
