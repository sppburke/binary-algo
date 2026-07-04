"""Central risk/buy coordinator + runtime assembly (issue #4 Phase 4).

The coordinator is the ONLY buy-capable object in the hot runtime (KILL #6):
quote workers and market-data workers hold the public socket; the coordinator
alone may hold a demo socket, and `execute()` requires the admit token issued
by `admit()` — a worker cannot bypass the gate battery.

Gate battery (order matters; every reject logs a stable reason):
  kill_switch -> outside_ny_session -> after_last_start_cutoff (shared
  conservative issue-#6 cutoff; see deriv_runtime_core.last_start_cutoff_info) ->
  lane-scoped freshness (wall: latest completed store bar <= 60s old;
  shifted: bar age <= 2s AND tick age <= 2s) -> shifted-lane enablement
  (DERIV_ALLOW_EXPERIMENTAL_SHIFTED_DEMO_BUY=1 is only the global master
  switch; the (pair, side) must be marked "passed" in the referenced audit
  verdict JSON) -> duplicate (pair, side, offset_id, signal_close_utc) ->
  max_open (ACCOUNT-WIDE, all pairs/offsets/sides) -> max_trades_day ->
  per-pair 15m cooldown across offsets and sides -> payout gate on a FRESH
  selected-side proposal (parse_quote + effective_floor > live_breakeven +
  live_breakeven <= max_breakeven; snapshot quotes are never bought).

Contract monitoring is correlated by contract_id (fixes the executor's
key-presence-only filter) and reconciles fail-closed on stream silence.

Phase-4 gates:
    ~/binary-algo-venv/bin/python scripts/deriv_runtime_supervisor.py --smoke
    ~/binary-algo-venv/bin/python scripts/deriv_runtime_supervisor.py --run --duration-minutes 540   # proposal-only
    (live $1 demo-buy sessions: --run --demo-buy on the VPS, >=2 NY sessions,
     activity floor >=20 payout-gate signals and >=1 confirmed buy)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import secrets
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from deriv_async_client import DerivAsyncClient, Subscription
from deriv_client import PUBLIC_WS_URL, DerivAPIError, DerivEnv, request_demo_ws_url
from deriv_runtime_core import (
    HORIZON_MINUTES,
    NY_TZ,
    ExecutorError,
    ExecutorLock,
    JsonlLogger,
    after_last_start_cutoff,
    check_kill_switch,
    empty_state,
    last_start_cutoff_info,
    load_state,
    parse_quote,
    pending_contracts,
    record_open_contract,
    resolve_pairs,
    save_state,
    update_contract_state,
)
from deriv_trade_queue import (
    DEFAULT_DB_PATH,
    DEFAULT_WAKEUP_PATH,
    PRODUCER_QUEUE_MODE_FANOUT_LIMIT,
    connect_queue,
    enqueue_signal,
    send_wakeup,
)
from live_features import DERIV_SYMBOLS

SHIFTED_FLAG_ENV = "DERIV_ALLOW_EXPERIMENTAL_SHIFTED_DEMO_BUY"
WALL_BAR_MAX_AGE_S = 60.0
SHIFTED_BAR_MAX_AGE_S = 2.0
SHIFTED_TICK_MAX_AGE_S = 2.0


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Coordinator:
    """Owns every gate and the only buy-capable client. Workers submit signals
    via admit() and may execute ONLY with the token admit() issued."""

    def __init__(
        self,
        args: argparse.Namespace,
        logger: JsonlLogger,
        state_path: Path,
        *,
        demo_client: DerivAsyncClient | None,
        audit_verdicts: dict[str, Any] | None,
    ):
        self.args = args
        self.logger = logger
        self.state_path = state_path
        self.state = load_state(state_path) if state_path.exists() else empty_state()
        self.demo_client = demo_client  # None in proposal-only mode
        self.audit_verdicts = audit_verdicts or {}
        self.kill_path = Path(args.log_dir) / "KILL"
        self.seen_keys: set[tuple[str, str, int, str]] = set()
        self.last_pair_buy_monotonic: dict[str, float] = {}
        self._admitted: dict[str, dict[str, Any]] = {}
        self.counters: dict[str, int] = {}
        self.monitor_tasks: set[asyncio.Task] = set()

    def _trades_today(self, now: datetime) -> int:
        return int((self.state.get("trades_by_day") or {}).get(now.date().isoformat(), 0))

    def _record_trade_today(self, now: datetime) -> None:
        day = now.date().isoformat()
        by_day = self.state.setdefault("trades_by_day", {})
        by_day[day] = int(by_day.get(day, 0)) + 1
        for old in [d for d in by_day if d < day]:  # keep only today (UTC roll)
            del by_day[old]

    async def reconcile(self) -> None:
        """Fail-closed startup reconciliation (review M2): every persisted open
        contract is queried; terminal ones are closed out, live ones keep
        max_open honest. A query failure refuses demo-buy mode."""
        client = self.demo_client
        pending = pending_contracts(self.state)
        if not pending:
            return
        if client is None:
            raise ExecutorError(f"{len(pending)} persisted open contracts but no demo client to reconcile")
        for contract_id in list(pending):
            frame = await client.request({"proposal_open_contract": 1, "contract_id": contract_id})
            terminal = update_contract_state(self.state, contract_id, frame)
            save_state(self.state_path, self.state)
            self.logger.write("reconcile_contract", contract_id=contract_id, terminal=terminal)

    # ------------------------------------------------------------ admit

    def _count(self, reason: str) -> None:
        self.counters[reason] = self.counters.get(reason, 0) + 1

    def admit(self, signal: dict[str, Any], now_utc: datetime | None = None) -> tuple[str | None, str]:
        """Returns (admit_token, reason). Token None => rejected with reason."""
        now = now_utc or datetime.now(timezone.utc)
        pair, side, lane = signal["pair"], signal["side"], signal.get("lane", "wall")
        try:
            check_kill_switch(self.kill_path, self.logger)
        except ExecutorError:
            return self._reject(signal, "kill_switch_triggered")
        ny = now.astimezone(NY_TZ)
        if ny.weekday() >= 5 or not (8.0 <= ny.hour + ny.minute / 60.0 < 17.0):
            return self._reject(signal, "outside_ny_session")
        if after_last_start_cutoff(now):
            return self._reject(signal, "after_last_start_cutoff")
        if lane == "wall":
            if signal.get("bar_age_s", 1e9) > WALL_BAR_MAX_AGE_S or not signal.get("is_latest_bar", False):
                return self._reject(signal, "stale_wall_bar")
        elif lane == "shifted":
            if signal.get("bar_age_s", 1e9) > SHIFTED_BAR_MAX_AGE_S:
                return self._reject(signal, "stale_shifted_bar")
            if signal.get("tick_age_s", 1e9) > SHIFTED_TICK_MAX_AGE_S:
                return self._reject(signal, "stale_shifted_tick")
            if os.environ.get(SHIFTED_FLAG_ENV) != "1":
                return self._reject(signal, "shifted_lane_disabled")
            verdict = ((self.audit_verdicts.get("verdicts") or {}).get(pair) or {}).get(side)
            if verdict != "passed":
                return self._reject(signal, "shifted_lane_not_passed")
        else:
            return self._reject(signal, "unknown_lane")
        key = (pair, side, int(signal.get("offset_id", 0)), str(signal["signal_close_utc"]))
        if key in self.seen_keys:
            return self._reject(signal, "duplicate_signal")
        # Risk caps are OPT-IN (0 = unconstrained): demo trades every firing signal
        if self.args.max_open and len(pending_contracts(self.state)) >= self.args.max_open:
            return self._reject(signal, "max_open")
        # The daily cap counts from persisted state, NOT the JSONL: the log
        # path is pinned to the process start date and logrotate copytruncate
        # would reset a log-derived count mid-day — fail-open (review M1)
        if self.args.max_trades_day and self._trades_today(now) >= self.args.max_trades_day:
            return self._reject(signal, "max_trades_day")
        if self.args.pair_cooldown_seconds:
            last = self.last_pair_buy_monotonic.get(pair)
            if last is not None and time.monotonic() - last < self.args.pair_cooldown_seconds:
                return self._reject(signal, "pair_cooldown")
        self.seen_keys.add(key)
        token = secrets.token_hex(8)
        self._admitted[token] = dict(signal)
        self._count("admitted")
        self.logger.write("signal_admitted", token=token, **signal)
        return token, "admitted"

    def _reject(self, signal: dict[str, Any], reason: str) -> tuple[None, str]:
        self._count(reason)
        self.logger.write("signal_skipped", reason=reason, **signal)
        return None, reason

    # ------------------------------------------------------------ execute (buy path)

    async def execute(self, token: str) -> dict[str, Any]:
        """Fresh proposal -> payout gate -> buy -> correlated monitor.
        Requires a valid admit token (KILL #6) and a demo client."""
        signal = self._admitted.pop(token, None)
        if signal is None:
            raise ExecutorError("execute without valid admit token (coordinator bypass)")
        pair, side = signal["pair"], signal["side"]
        contract_type = "CALL" if side == "UP" else "PUT"
        client = self.demo_client
        if client is None or "/demo" not in client.url:
            self._count("demo_buy_not_enabled")
            self.logger.write("signal_skipped", reason="demo_buy_not_enabled", **signal)
            return {"bought": False, "reason": "demo_buy_not_enabled"}
        resp = await client.request({
            "proposal": 1, "amount": float(self.args.stake), "basis": "stake",
            "contract_type": contract_type, "currency": "USD",
            "duration": HORIZON_MINUTES, "duration_unit": "m",
            "underlying_symbol": DERIV_SYMBOLS[pair],
        })
        prop = resp.get("proposal") or {}
        ask, payout, live_breakeven, invalid_reason = parse_quote(prop)
        if invalid_reason is not None:
            self._count(invalid_reason)
            self.logger.write("signal_skipped", reason=invalid_reason, **signal)
            return {"bought": False, "reason": invalid_reason}
        if signal["effective_floor"] <= live_breakeven:
            self._count("edge_not_positive")
            self.logger.write("signal_skipped", reason="edge_not_positive",
                              live_breakeven=live_breakeven, **signal)
            return {"bought": False, "reason": "edge_not_positive"}
        if live_breakeven > self.args.max_breakeven:
            self._count("breakeven_too_high")
            self.logger.write("signal_skipped", reason="breakeven_too_high",
                              live_breakeven=live_breakeven, **signal)
            return {"bought": False, "reason": "breakeven_too_high"}
        proposal_id = prop.get("id")
        if not proposal_id:  # never send the literal string "None" as a buy id (review M3)
            raise ExecutorError(f"{pair}: proposal missing id despite valid quote")
        check_kill_switch(self.kill_path, self.logger)  # last look before money moves
        self.logger.write("buy_submitted", pair=pair, side=side, ask=ask, payout=payout,
                          live_breakeven=live_breakeven, **{k: v for k, v in signal.items() if k not in ("pair", "side")})
        buy = await client.request({"buy": str(proposal_id), "price": float(ask)})
        contract_id_raw = (buy.get("buy") or {}).get("contract_id")
        if not contract_id_raw:
            raise ExecutorError(f"{pair}: buy response missing contract_id: {sorted(buy)}")
        contract_id = str(contract_id_raw)
        self.last_pair_buy_monotonic[pair] = time.monotonic()
        self._record_trade_today(datetime.now(timezone.utc))
        record_open_contract(
            self.state, pair=pair, proposal_id=str(prop.get("id")), contract_id=contract_id,
            stake=float(self.args.stake), buy_price=float(ask),
            expected_expiry=datetime.now(timezone.utc) + timedelta(minutes=HORIZON_MINUTES))
        self.state["open_contracts"][contract_id].update(
            side=side, offset_id=signal.get("offset_id", 0), lane=signal.get("lane", "wall"))
        save_state(self.state_path, self.state)
        self.logger.write("buy_confirmed", pair=pair, contract_id=contract_id)
        # Monitor CONCURRENTLY (review F3): an inline await serialized the
        # account to ~one open trade despite unconstrained caps. Errors are
        # contained; state keeps the contract for reconcile().
        task = asyncio.get_running_loop().create_task(self._monitor_contained(contract_id))
        self.monitor_tasks.add(task)
        task.add_done_callback(self.monitor_tasks.discard)
        return {"bought": True, "contract_id": contract_id, "monitor_task": task}

    async def _monitor_contained(self, contract_id: str) -> None:
        try:
            await self.monitor(contract_id)
        except (ExecutorError, DerivAPIError) as exc:
            self.logger.write("monitor_error", contract_id=contract_id, error=str(exc)[:300])

    # ------------------------------------------------------------ monitor (contract_id-correlated)

    async def monitor(self, contract_id: str) -> None:
        client = self.demo_client
        assert client is not None
        sub = await client.subscribe({"proposal_open_contract": 1, "contract_id": contract_id})
        deadline = time.monotonic() + self.args.monitor_timeout_seconds
        try:
            while time.monotonic() < deadline:
                try:
                    frame = await asyncio.wait_for(sub.queue.get(), timeout=self.args.monitor_interval_seconds)
                except asyncio.TimeoutError:
                    # stream silent: fail-closed poll for exactly this contract
                    frame = await client.request({"proposal_open_contract": 1, "contract_id": contract_id})
                poc = frame.get("proposal_open_contract")
                if not isinstance(poc, dict):
                    continue
                if str(poc.get("contract_id")) != contract_id:
                    self.logger.write("monitor_frame_ignored", expected=contract_id,
                                      got=str(poc.get("contract_id")))
                    continue
                terminal = update_contract_state(self.state, contract_id, frame)
                save_state(self.state_path, self.state)
                self.logger.write("contract_update", contract_id=contract_id,
                                  status=self.state["open_contracts"].get(contract_id, {}).get("status"))
                if terminal:
                    self.logger.write("contract_closed", contract_id=contract_id)
                    return
            raise ExecutorError(f"contract {contract_id}: monitor timeout before terminal status")
        finally:
            try:
                await client.unsubscribe(sub)
            except (DerivAPIError, KeyError):
                pass


# ================================================================ fake clients for the smoke


class FakeAsyncClient:
    """Scripted DerivAsyncClient stand-in: request() pops scripted responses;
    subscribe() hands out a queue preloaded by the test."""

    def __init__(self, responses: list[dict[str, Any]], sub_frames: list[dict[str, Any]] | None = None,
                 url: str = "wss://fake.example/demo"):
        self.url = url
        self.responses = list(responses)
        self.sub_frames = list(sub_frames or [])
        self.requests: list[dict[str, Any]] = []

    async def request(self, payload: dict[str, Any], timeout: float | None = None) -> dict[str, Any]:
        self.requests.append(payload)
        if not self.responses:
            raise DerivAPIError("fake client out of scripted responses")
        return self.responses.pop(0)

    async def subscribe(self, payload: dict[str, Any], queue_maxsize: int | None = None) -> Subscription:
        sub = Subscription(payload, 100)
        sub.sub_id = "fake-sub"
        for frame in self.sub_frames:
            sub.queue.put_nowait(frame)
        return sub

    async def unsubscribe(self, sub: Subscription) -> None:
        sub.active = False


# ================================================================ smoke


def _sig(**over: Any) -> dict[str, Any]:
    base = {"pair": "USDJPY", "side": "UP", "offset_id": 0,
            "signal_close_utc": "2026-07-01T18:00:00+00:00", "effective_floor": 0.58,
            "lane": "wall", "bar_age_s": 5.0, "is_latest_bar": True}
    base.update(over)
    return base


async def run_smoke() -> int:
    import tempfile

    failures: list[str] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        print(f"{'PASS' if ok else 'FAIL'}: {name}" + (f" — {detail}" if detail else ""))
        if not ok:
            failures.append(name)

    tmp = Path(tempfile.mkdtemp(prefix="deriv_sup_smoke_"))
    args = argparse.Namespace(log_dir=tmp, stake=1.0, max_breakeven=0.60, payout_edge_margin=0.005,
                              max_open=1, max_trades_day=3, pair_cooldown_seconds=900,
                              monitor_timeout_seconds=5.0, monitor_interval_seconds=0.01)
    logger = JsonlLogger(tmp)
    in_session = datetime(2026, 7, 1, 18, 0, tzinfo=timezone.utc)   # Wed 14:00 NY
    late = datetime(2026, 7, 1, 20, 50, tzinfo=timezone.utc)        # Wed 16:50 NY
    edge_ok = datetime(2026, 7, 1, 20, 34, 59, tzinfo=timezone.utc)  # 16:34:59 NY
    edge_cut = datetime(2026, 7, 1, 20, 35, 0, tzinfo=timezone.utc)  # 16:35:00 NY
    pre_open = datetime(2026, 7, 1, 10, 0, tzinfo=timezone.utc)     # 06:00 NY

    def fresh_coord(**kw: Any) -> Coordinator:
        return Coordinator(args, logger, tmp / "state.json", demo_client=kw.get("demo_client"),
                           audit_verdicts=kw.get("audit_verdicts"))

    c = fresh_coord()
    # 1-2. session + cutoff battery
    check("in-session signal admitted", c.admit(_sig(), in_session)[0] is not None)
    check("16:50 NY rejected after_last_start_cutoff",
          c.admit(_sig(signal_close_utc="x1"), late)[1] == "after_last_start_cutoff")
    check("16:34:59 NY admitted", c.admit(_sig(signal_close_utc="x2"), edge_ok)[0] is not None)
    check("16:35:00 NY rejected", c.admit(_sig(signal_close_utc="x3"), edge_cut)[1] == "after_last_start_cutoff")
    check("pre-open rejected outside_ny_session", c.admit(_sig(signal_close_utc="x4"), pre_open)[1] == "outside_ny_session")

    # 3. lane-scoped staleness (fail-closed both lanes)
    check("wall bar 61s stale rejected", c.admit(_sig(signal_close_utc="x5", bar_age_s=61.0), in_session)[1] == "stale_wall_bar")
    check("wall non-latest bar rejected", c.admit(_sig(signal_close_utc="x6", is_latest_bar=False), in_session)[1] == "stale_wall_bar")
    check("shifted bar 3s stale rejected",
          c.admit(_sig(signal_close_utc="x7", lane="shifted", bar_age_s=3.0, tick_age_s=1.0), in_session)[1] == "stale_shifted_bar")
    check("shifted tick 3s stale rejected",
          c.admit(_sig(signal_close_utc="x8", lane="shifted", bar_age_s=1.0, tick_age_s=3.0), in_session)[1] == "stale_shifted_tick")

    # 4. shifted-lane enablement: env flag is master switch only; per-lane verdict required
    os.environ.pop(SHIFTED_FLAG_ENV, None)
    check("shifted lane rejected without env flag",
          c.admit(_sig(signal_close_utc="x9", lane="shifted", bar_age_s=1.0, tick_age_s=1.0), in_session)[1] == "shifted_lane_disabled")
    os.environ[SHIFTED_FLAG_ENV] = "1"
    check("shifted lane rejected without passed verdict",
          c.admit(_sig(signal_close_utc="xa", lane="shifted", bar_age_s=1.0, tick_age_s=1.0), in_session)[1] == "shifted_lane_not_passed")
    c2 = fresh_coord(audit_verdicts={"verdicts": {"USDJPY": {"UP": "passed"}}})
    check("shifted lane admitted with env flag AND passed verdict",
          c2.admit(_sig(signal_close_utc="xb", lane="shifted", bar_age_s=1.0, tick_age_s=1.0), in_session)[0] is not None)
    check("shifted DOWN lane still rejected (per-side verdict)",
          c2.admit(_sig(signal_close_utc="xc", side="DOWN", lane="shifted", bar_age_s=1.0, tick_age_s=1.0), in_session)[1] == "shifted_lane_not_passed")
    os.environ.pop(SHIFTED_FLAG_ENV, None)

    # 5. duplicate key (cross-offset keys differ; same key rejected)
    c3 = fresh_coord()
    c3.admit(_sig(signal_close_utc="dup"), in_session)
    check("duplicate signal rejected", c3.admit(_sig(signal_close_utc="dup"), in_session)[1] == "duplicate_signal")

    # 6. pair cooldown across offsets and sides
    c3.last_pair_buy_monotonic["USDJPY"] = time.monotonic()
    check("cross-offset buy inside cooldown rejected",
          c3.admit(_sig(signal_close_utc="cd", offset_id=25, side="DOWN"), in_session)[1] == "pair_cooldown")

    # 7. max_open account-wide
    c4 = fresh_coord()
    c4.state["open_contracts"]["c-1"] = {"pair": "GBPUSD", "status": "open"}
    check("second contract account-wide rejected",
          c4.admit(_sig(signal_close_utc="mo"), in_session)[1] == "max_open")

    # 7b. daily cap from persisted state (survives log rotation)
    c4b = fresh_coord()
    c4b.state["trades_by_day"] = {in_session.date().isoformat(): 3}
    check("max_trades_day from state rejects",
          c4b.admit(_sig(signal_close_utc="mt"), in_session)[1] == "max_trades_day")

    # 8. KILL file
    (tmp / "KILL").touch()
    c5 = fresh_coord()
    check("KILL file rejects", c5.admit(_sig(signal_close_utc="k"), in_session)[1] == "kill_switch_triggered")
    (tmp / "KILL").unlink()

    # 9. coordinator bypass (KILL #6): execute without a valid token raises
    c6 = fresh_coord(demo_client=FakeAsyncClient([]))
    try:
        await c6.execute("forged-token")
        check("bypass raises", False)
    except ExecutorError:
        check("bypass raises ExecutorError", True)

    # 10. payout gate on the execute path (fresh proposal, never snapshots)
    async def run_exec(responses: list[dict[str, Any]], sub_frames: list[dict[str, Any]] | None = None,
                       floor: float = 0.58) -> dict[str, Any]:
        cx = fresh_coord(demo_client=FakeAsyncClient(responses, sub_frames))
        token, _ = cx.admit(_sig(signal_close_utc=f"e{len(logger.path.read_text())}", effective_floor=floor), in_session)
        r = await cx.execute(token)
        if r.get("monitor_task") is not None:
            await r["monitor_task"]  # smoke assertions need the monitor to finish
        return r

    r = await run_exec([{"proposal": {"payout": 1.9}}])
    check("missing ask fails closed", r["reason"] == "missing_or_invalid_ask")
    r = await run_exec([{"proposal": {"id": "p1", "ask_price": 1.0, "payout": 1.6}}], floor=0.55)
    check("floor <= breakeven rejected edge_not_positive", r["reason"] == "edge_not_positive", str(r))
    r = await run_exec([{"proposal": {"id": "p1", "ask_price": 1.0, "payout": 1.5}}], floor=0.70)
    check("breakeven > max rejected", r["reason"] == "breakeven_too_high")

    # 11. good buy + contract_id-correlated monitor (wrong-id frame ignored)
    poc_wrong = {"proposal_open_contract": {"contract_id": "OTHER", "status": "open"}}
    poc_open = {"proposal_open_contract": {"contract_id": "C77", "status": "open"}}
    poc_won = {"proposal_open_contract": {"contract_id": "C77", "status": "won", "profit": 0.9}}
    r = await run_exec(
        [{"proposal": {"id": "p1", "ask_price": 1.0, "payout": 1.9}}, {"buy": {"contract_id": "C77"}}],
        sub_frames=[poc_wrong, poc_open, poc_won], floor=0.58)
    check("good signal buys and reaches terminal", r.get("bought") is True and r.get("contract_id") == "C77", str(r))
    log_text = logger.path.read_text()
    check("wrong-contract frame was ignored (correlation fix)", '"monitor_frame_ignored"' in log_text and '"OTHER"' in log_text)

    # 12. monitor stream silence falls back to fail-closed poll
    r = await run_exec(
        [{"proposal": {"id": "p1", "ask_price": 1.0, "payout": 1.9}}, {"buy": {"contract_id": "C88"}},
         {"proposal_open_contract": {"contract_id": "C88", "status": "sold"}}],
        sub_frames=[], floor=0.58)
    check("silent stream reconciles via poll to terminal", r.get("bought") is True, str(r))

    print(f"\n{'SMOKE ALL PASS' if not failures else f'SMOKE {len(failures)} FAILURES: {failures}'}")
    return 1 if failures else 0


# ================================================================ live assembly


async def run_live(args: argparse.Namespace) -> int:
    """Wall-clock assembly: hot daemon scoring -> coordinator; quote workers hot.
    Proposal-only unless --demo-buy (which needs DERIV_* env on the VPS)."""
    from deriv_hot_daemon import HotDaemon  # late import: heavy chain

    logger = JsonlLogger(Path(args.log_dir))
    queue_conn = None
    if args.queue_mode:
        if args.demo_buy:
            raise ExecutorError("--queue-mode is producer-only; run scripts/deriv_trade_executor.py for demo buys")
        leaked = [k for k in ("DERIV_PAT", "DERIV_ACCOUNT_ID", "DERIV_SUPERVISOR_MODE_ARGS") if os.environ.get(k)]
        if leaked:
            raise ExecutorError(f"producer queue mode received buy credential/mode env vars: {leaked}")
        queue_conn = connect_queue(Path(args.queue_db))
    demo_client = None
    if args.demo_buy and not args.queue_mode:
        demo_client = DerivAsyncClient(request_demo_ws_url(DerivEnv.from_env()))
        await demo_client.connect()
    audit_verdicts = json.loads(Path(args.audit_json).read_text()) if args.audit_json else None
    coordinator = Coordinator(args, logger, Path(args.log_dir) / "supervisor_state.json",
                              demo_client=demo_client, audit_verdicts=audit_verdicts)
    public = DerivAsyncClient(args.url, on_event=lambda e, f: logger.write("client_" + e, **f))
    await public.connect()
    from deriv_quote_workers import QuoteWorkers
    workers = QuoteWorkers(public, resolve_pairs(args.pairs), args.stake,
                           on_event=lambda e, f: logger.write(e, **f))
    await workers.start()

    daemon_args = argparse.Namespace(
        pairs=args.pairs, once=False, duration_minutes=args.duration_minutes, parity_replay=None,
        store_dir=args.store_dir, snapshot_dir=args.snapshot_dir, snapshot_keep_minutes=240,
        poll_seconds=2.0, store_min_required_rows=14000, store_max_stale_seconds=180,
        store_wait_poll_seconds=args.store_wait_poll_seconds, store_wait_max_seconds=0.0,
        payout_edge_margin=args.payout_edge_margin, max_breakeven=args.max_breakeven,
        stake=args.stake, log_dir=args.log_dir, url=args.url,
        queue_mode=args.queue_mode,
        producer_queue_mode_fanout_limit=args.producer_queue_mode_fanout_limit)
    daemon = HotDaemon(daemon_args)

    if not args.queue_mode:
        await coordinator.reconcile()  # fail-closed before any signal flows
    else:
        logger.write("producer_queue_mode_started", queue_db=str(args.queue_db),
                     queue_wakeup_socket=str(args.queue_wakeup_socket),
                     cutoff=last_start_cutoff_info())

    async def route(signal: dict[str, Any]) -> None:
        if args.queue_mode:
            assert queue_conn is not None
            result = enqueue_signal(queue_conn, signal)
            close_ts = datetime.fromisoformat(str(signal["signal_close_utc"]).replace("Z", "+00:00"))
            if close_ts.tzinfo is None:
                close_ts = close_ts.replace(tzinfo=timezone.utc)
            candidate_ms = round((datetime.now(timezone.utc) - close_ts.astimezone(timezone.utc)).total_seconds() * 1000.0, 1)
            if result.inserted:
                woke = send_wakeup(Path(args.queue_wakeup_socket),
                                   payload={"event": "queue_wakeup", "signal_id": result.signal_id})
                logger.write("signal_queued", signal_id=result.signal_id, wakeup_sent=woke,
                             candidate_close_to_enqueue_ms=candidate_ms, **signal)
            else:
                logger.write("duplicate_signal", signal_id=result.signal_id,
                             conflict_signal_id=result.conflict_signal_id,
                             conflict_reason=result.conflict_reason,
                             candidate_close_to_enqueue_ms=candidate_ms, **signal)
            return
        token, reason = coordinator.admit(signal)
        if token is not None:
            if demo_client is None:
                coordinator._admitted.pop(token, None)
                logger.write("signal_skipped", reason="dry_run_proposal_only", **signal)
            else:
                try:
                    await coordinator.execute(token)
                except (ExecutorError, DerivAPIError) as exc:
                    # survive and reconcile rather than crash-loop mid-contract
                    # (review M2); state keeps the contract for reconcile()
                    logger.write("execute_error", error=str(exc)[:300], **signal)

    daemon.signal_sink = route  # HotDaemon calls this on edge_gate_passed when set
    lock_ctx = ExecutorLock(Path(args.log_dir), "supervisor-demo" if args.demo_buy else "supervisor-dry", logger)
    with lock_ctx:
        rc = await daemon.run()
        if coordinator.monitor_tasks:  # let open-contract monitors finish before teardown
            await asyncio.gather(*coordinator.monitor_tasks, return_exceptions=True)
    await workers.stop()
    await public.close()
    if demo_client is not None:
        await demo_client.close()
    if queue_conn is not None:
        queue_conn.close()
    logger.write("supervisor_rollup", counters=coordinator.counters,
                 quote_updates=workers.updates, quote_errors=workers.errors)
    print(json.dumps({"coordinator": coordinator.counters,
                      "quote_updates": workers.updates, "quote_errors": workers.errors}, indent=1))
    return rc


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--run", action="store_true")
    p.add_argument("--demo-buy", action="store_true")
    p.add_argument("--pairs", default="all-enabled")
    p.add_argument("--duration-minutes", type=float, default=0.0)
    p.add_argument("--store-dir", default="deriv_data/candles_1m_daemon")
    p.add_argument("--snapshot-dir", default="deriv_data/hot_snapshots")
    p.add_argument("--store-wait-poll-seconds", type=float, default=60.0,
                   help="when the daemon store is stale (market closed) re-check every N seconds instead of crash-looping")
    p.add_argument("--queue-mode", action="store_true",
                   help="producer-only mode: enqueue selected-side candidates into SQLite; no credentials/buy path")
    p.add_argument("--queue-db", default=str(DEFAULT_DB_PATH))
    p.add_argument("--queue-wakeup-socket", default=str(DEFAULT_WAKEUP_PATH))
    p.add_argument("--producer-queue-mode-fanout-limit", type=int, default=PRODUCER_QUEUE_MODE_FANOUT_LIMIT)
    p.add_argument("--audit-json", default=None, help="per-(pair,side) shifted-lane verdict JSON (Phase 5)")
    p.add_argument("--stake", type=float, default=1.0)
    p.add_argument("--payout-edge-margin", type=float, default=0.005)
    p.add_argument("--max-breakeven", type=float, default=0.60)
    p.add_argument("--max-open", type=int, default=0, help="0 = unlimited concurrent open contracts, account-wide")
    p.add_argument("--max-trades-day", type=int, default=0, help="0 = unlimited (demo default: trade every signal)")
    p.add_argument("--pair-cooldown-seconds", type=int, default=0, help="0 = none")
    p.add_argument("--monitor-timeout-seconds", type=float, default=1200.0)
    p.add_argument("--monitor-interval-seconds", type=float, default=2.0)
    p.add_argument("--log-dir", default="logs/deriv_supervisor")
    p.add_argument("--url", default=PUBLIC_WS_URL)
    args = p.parse_args()
    # negative caps would be truthy and reject everything (review footgun)
    args.max_open = max(0, args.max_open)
    args.max_trades_day = max(0, args.max_trades_day)
    args.pair_cooldown_seconds = max(0, args.pair_cooldown_seconds)
    if args.smoke:
        return asyncio.run(run_smoke())
    if args.run:
        return asyncio.run(run_live(args))
    p.error("pass --smoke or --run")
    return 2


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ExecutorError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)
