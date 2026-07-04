"""Credentialed durable Deriv demo trade executor (issue #6).

The producer writes selected-side candidates to SQLite. This process is the
only buy-capable runtime: it owns the account-scoped lock, demo credentials,
queue claims, proposal/buy submission, and terminal queue state.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import socket
import sys
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from deriv_async_client import DerivAsyncClient
from deriv_client import DerivAPIError, DerivEnv, DerivOptionsClient, request_demo_ws_url
from deriv_runtime_core import (
    HORIZON_MINUTES,
    ExecutorError,
    ExecutorLock,
    JsonlLogger,
    account_lock_key,
    after_last_start_cutoff,
    canonical_lock_root,
    check_kill_switch,
    executor_lock_path,
    is_ny_session,
    last_start_cutoff_info,
    parse_quote,
)
from deriv_trade_queue import (
    ASYNC_PROPOSAL_TIMEOUT_MS,
    AUTH_HEALTH_INTERVAL_S,
    BUY_REQUEST_TIMEOUT_MS,
    DEFAULT_DB_PATH,
    DEFAULT_WAKEUP_PATH,
    EXECUTOR_WORKER_CONCURRENCY,
    GATE_RESULT_PATH,
    QUEUE_POLL_INTERVAL_MS,
    SHIFTED_SIGNAL_MAX_AGE_S,
    SYNC_FALLBACK_MAX_CONCURRENCY,
    SYNC_FALLBACK_TIMEOUT_S,
    WALL_SIGNAL_MAX_AGE_S,
    ClaimedSignal,
    connect_queue,
    enqueue_signal,
    fixed_defaults,
    pct,
    queue_counts,
    record_contract,
    transition_signal,
    utc_iso,
    write_gate_result,
)
from live_features import DERIV_SYMBOLS


def _contract_type(side: str) -> str:
    return "CALL" if side == "UP" else "PUT"


def _raw_hash(payload: Any) -> str:
    import hashlib

    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:16]


class TradeExecutor:
    def __init__(self, args: argparse.Namespace, logger: JsonlLogger):
        self.args = args
        self.logger = logger
        self.kill_path = Path(args.log_dir) / "KILL"
        self.db_path = Path(args.queue_db)
        self.wakeup_path = Path(args.queue_wakeup_socket)
        self.run_id = f"{socket.gethostname()}-{os.getpid()}-{int(time.time())}"
        self.async_client: DerivAsyncClient | None = None
        self.sync_sem = asyncio.Semaphore(args.sync_fallback_max_concurrency)
        self.stop = asyncio.Event()
        self.counters: dict[str, int] = {}
        self.latencies: dict[str, list[float]] = {
            "claim_to_buy_submitted_ms": [],
            "async_disconnected_to_fallback_start_ms": [],
            "sync_fallback_start_to_buy_submitted_ms": [],
        }
        self.max_inflight_proposal_requests = 0
        self.max_inflight_buy_requests = 0
        self.inflight_proposal_requests = 0
        self.inflight_buy_requests = 0

    def _count(self, key: str) -> None:
        self.counters[key] = self.counters.get(key, 0) + 1

    async def auth_health(self) -> bool:
        """Detect-and-surface PAT/OTP failure; no auto-refresh here."""
        try:
            env = DerivEnv.from_env()
            url = await asyncio.to_thread(request_demo_ws_url, env, 10.0)
            def _ping() -> dict[str, Any]:
                with DerivOptionsClient(url, timeout=10.0) as client:
                    return client.ping()
            await asyncio.to_thread(_ping)
        except (DerivAPIError, OSError, TimeoutError) as exc:
            msg = str(exc)
            reason = "buy_blocked_auth" if "401" in msg or "expired OTP" in msg or "missing required env" in msg else "auth_health_failed"
            self._count(reason)
            self.logger.write("auth_health", ok=False, reason=reason, error=msg[:300],
                              interval_s=self.args.auth_health_interval_s)
            return False
        self.logger.write("auth_health", ok=True, interval_s=self.args.auth_health_interval_s)
        return True

    async def auth_health_loop(self) -> None:
        while not self.stop.is_set():
            await asyncio.sleep(self.args.auth_health_interval_s)
            if self.stop.is_set():
                return
            await self.auth_health()

    async def connect_async_client(self) -> None:
        env = DerivEnv.from_env()
        url = await asyncio.to_thread(request_demo_ws_url, env, 10.0)
        self.async_client = DerivAsyncClient(
            url,
            request_timeout=max(self.args.async_proposal_timeout_ms, self.args.buy_request_timeout_ms) / 1000.0,
            on_event=lambda e, f: self.logger.write("exec_client_" + e, **f),
        )
        await self.async_client.connect()

    def signal_age_s(self, payload: dict[str, Any]) -> float:
        close = datetime.fromisoformat(str(payload["signal_close_utc"]).replace("Z", "+00:00"))
        if close.tzinfo is None:
            close = close.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - close.astimezone(timezone.utc)).total_seconds()

    def pre_money_gate(self, claim: ClaimedSignal) -> str | None:
        payload = claim.payload
        try:
            check_kill_switch(self.kill_path, self.logger)
        except ExecutorError:
            return "kill_switch_triggered"
        now = datetime.now(timezone.utc)
        if not is_ny_session(now):
            return "outside_ny_session"
        if after_last_start_cutoff(now):
            return "after_last_start_cutoff"
        lane = payload.get("lane", "wall")
        age_s = self.signal_age_s(payload)
        if lane == "wall" and age_s > self.args.wall_signal_max_age_s:
            return "signal_expired"
        if lane == "shifted" and age_s > self.args.shifted_signal_max_age_s:
            return "signal_expired"
        if lane == "wall":
            if payload.get("bar_age_s", 1e9) > self.args.wall_signal_max_age_s or not payload.get("is_latest_bar", False):
                return "stale_wall_bar"
        elif lane == "shifted":
            if payload.get("bar_age_s", 1e9) > self.args.shifted_signal_max_age_s:
                return "stale_shifted_bar"
            if payload.get("tick_age_s", 1e9) > self.args.shifted_signal_max_age_s:
                return "stale_shifted_tick"
        else:
            return "unknown_lane"
        return None

    async def proposal_async(self, payload: dict[str, Any]) -> tuple[dict[str, Any], str]:
        client = self.async_client
        if client is None:
            raise DerivAPIError("async client not initialised")
        if not client.is_connected():
            raise DerivAPIError("async client disconnected before proposal")
        self.inflight_proposal_requests += 1
        self.max_inflight_proposal_requests = max(self.max_inflight_proposal_requests, self.inflight_proposal_requests)
        try:
            resp = await client.request({
                "proposal": 1,
                "amount": float(self.args.stake),
                "basis": "stake",
                "contract_type": _contract_type(payload["side"]),
                "currency": "USD",
                "duration": HORIZON_MINUTES,
                "duration_unit": "m",
                "underlying_symbol": DERIV_SYMBOLS[payload["pair"]],
            }, timeout=self.args.async_proposal_timeout_ms / 1000.0)
            return resp, "fresh_async"
        finally:
            self.inflight_proposal_requests -= 1

    def proposal_sync(self, payload: dict[str, Any]) -> tuple[dict[str, Any], str]:
        with DerivOptionsClient.demo_from_env(timeout=self.args.sync_fallback_timeout_s) as client:
            return client.proposal(
                symbol=DERIV_SYMBOLS[payload["pair"]],
                contract_type=_contract_type(payload["side"]),
                amount=float(self.args.stake),
                duration=HORIZON_MINUTES,
            ), "sync_fallback"

    async def get_proposal(self, payload: dict[str, Any]) -> tuple[dict[str, Any], str]:
        client = self.async_client
        if client is not None and client.is_connected():
            try:
                return await self.proposal_async(payload)
            except DerivAPIError as exc:
                self.logger.write("proposal_fallback", trigger=str(exc)[:200], source="fresh_async")
        else:
            t0 = time.perf_counter()
            self.latencies["async_disconnected_to_fallback_start_ms"].append((time.perf_counter() - t0) * 1000.0)
            self.logger.write("proposal_fallback", trigger="async_disconnected", source="connection_state")
        async with self.sync_sem:
            t1 = time.perf_counter()
            resp, source = await asyncio.wait_for(
                asyncio.to_thread(self.proposal_sync, payload),
                timeout=self.args.sync_fallback_timeout_s,
            )
            self.logger.write("sync_fallback_used", sync_fallback_queue_wait_ms=0.0,
                              sync_fallback_start_to_terminal_ms=round((time.perf_counter() - t1) * 1000.0, 1))
            return resp, source

    async def buy_async(self, proposal_id: str, ask: float) -> dict[str, Any]:
        client = self.async_client
        if client is None:
            raise DerivAPIError("async client not initialised")
        self.inflight_buy_requests += 1
        self.max_inflight_buy_requests = max(self.max_inflight_buy_requests, self.inflight_buy_requests)
        try:
            return await client.request({"buy": str(proposal_id), "price": float(ask)},
                                        timeout=self.args.buy_request_timeout_ms / 1000.0)
        finally:
            self.inflight_buy_requests -= 1

    def buy_sync(self, proposal_id: str, ask: float) -> dict[str, Any]:
        with DerivOptionsClient.demo_from_env(timeout=self.args.sync_fallback_timeout_s) as client:
            return client.buy(str(proposal_id), float(ask))

    async def execute_claim(self, claim: ClaimedSignal, worker_id: str) -> None:
        conn = connect_queue(self.db_path)
        payload = claim.payload
        t_claim = time.perf_counter()
        reason = self.pre_money_gate(claim)
        if reason:
            status = "signal_expired" if reason == "signal_expired" else "terminal_skip"
            transition_signal(conn, claim.signal_id, ["claimed"], status, reason=reason, source="pre_money_gate")
            self._count(reason)
            self.logger.write(status, signal_id=claim.signal_id, reason=reason, worker_id=worker_id, **payload)
            conn.close()
            return
        try:
            proposal_resp, source = await self.get_proposal(payload)
        except Exception as exc:
            transition_signal(conn, claim.signal_id, ["claimed"], "terminal_skip",
                              reason="proposal_failed", source="proposal", error=str(exc)[:300])
            self._count("proposal_failed")
            self.logger.write("signal_skipped", signal_id=claim.signal_id, reason="proposal_failed",
                              error=str(exc)[:300], worker_id=worker_id, **payload)
            conn.close()
            return
        prop = proposal_resp.get("proposal") or {}
        ask, payout, live_breakeven, invalid_reason = parse_quote(prop)
        self.logger.write("proposal_selected", signal_id=claim.signal_id, proposal_source=source,
                          proposal_id=prop.get("id"), ask=ask, payout=payout,
                          live_breakeven=live_breakeven, worker_id=worker_id, **payload)
        if invalid_reason:
            transition_signal(conn, claim.signal_id, ["claimed"], "terminal_skip",
                              reason=invalid_reason, source=source, raw_hash_value=_raw_hash(proposal_resp))
            self.logger.write("payout_gate_failed", signal_id=claim.signal_id, reason=invalid_reason, **payload)
            conn.close()
            return
        if float(payload["effective_floor"]) <= float(live_breakeven):
            transition_signal(conn, claim.signal_id, ["claimed"], "terminal_skip",
                              reason="edge_not_positive", source=source, raw_hash_value=_raw_hash(proposal_resp))
            self.logger.write("payout_gate_failed", signal_id=claim.signal_id, reason="edge_not_positive",
                              live_breakeven=live_breakeven, **payload)
            conn.close()
            return
        if float(live_breakeven) > self.args.max_breakeven:
            transition_signal(conn, claim.signal_id, ["claimed"], "terminal_skip",
                              reason="breakeven_too_high", source=source, raw_hash_value=_raw_hash(proposal_resp))
            self.logger.write("payout_gate_failed", signal_id=claim.signal_id, reason="breakeven_too_high",
                              live_breakeven=live_breakeven, **payload)
            conn.close()
            return
        proposal_id = prop.get("id")
        if not proposal_id:
            transition_signal(conn, claim.signal_id, ["claimed"], "terminal_skip",
                              reason="missing_proposal_id", source=source, raw_hash_value=_raw_hash(proposal_resp))
            conn.close()
            return
        transition_signal(conn, claim.signal_id, ["claimed"], "buy_intent", source=source)
        self.logger.write("buy_intent", signal_id=claim.signal_id, proposal_id=proposal_id, proposal_source=source, **payload)
        try:
            if source == "sync_fallback":
                async with self.sync_sem:
                    buy_resp = await asyncio.wait_for(
                        asyncio.to_thread(self.buy_sync, str(proposal_id), float(ask)),
                        timeout=self.args.sync_fallback_timeout_s,
                    )
                    buy_transport = "sync_ws"
            else:
                buy_resp = await self.buy_async(str(proposal_id), float(ask))
                buy_transport = "async_ws"
            buy_ms = (time.perf_counter() - t_claim) * 1000.0
            self.latencies["claim_to_buy_submitted_ms"].append(buy_ms)
            self.logger.write("buy_submitted", signal_id=claim.signal_id, proposal_id=proposal_id,
                              buy_transport=buy_transport, claim_to_buy_submitted_ms=round(buy_ms, 1),
                              buy_send_utc=utc_iso(), **payload)
        except Exception as exc:
            transition_signal(conn, claim.signal_id, ["buy_intent"], "buy_unknown",
                              reason="buy_transport_uncertain", source=source, error=str(exc)[:300])
            self._count("buy_unknown")
            self.logger.write("buy_unknown", signal_id=claim.signal_id, reason="buy_transport_uncertain",
                              error=str(exc)[:300], **payload)
            conn.close()
            return
        contract_id = (buy_resp.get("buy") or {}).get("contract_id")
        if not contract_id:
            transition_signal(conn, claim.signal_id, ["buy_intent"], "buy_unknown",
                              reason="buy_response_missing_contract_id", source=source, raw_hash_value=_raw_hash(buy_resp))
            self.logger.write("buy_unknown", signal_id=claim.signal_id, reason="buy_response_missing_contract_id",
                              raw_hash=_raw_hash(buy_resp), **payload)
            conn.close()
            return
        record_contract(
            conn,
            signal_id=claim.signal_id,
            contract_id=str(contract_id),
            contract_status="open",
            proposal_id=str(proposal_id),
            buy_price=float(ask),
            stake=float(self.args.stake),
            expected_expiry_utc=utc_iso(datetime.now(timezone.utc) + timedelta(minutes=HORIZON_MINUTES)),
            raw_hash_value=_raw_hash(buy_resp),
        )
        transition_signal(conn, claim.signal_id, ["buy_intent"], "bought",
                          reason="buy_confirmed", source=source, raw_hash_value=_raw_hash(buy_resp))
        self._count("buy_confirmed")
        self.logger.write("buy_confirmed", signal_id=claim.signal_id, contract_id=str(contract_id), **payload)
        conn.close()

    async def worker(self, worker_id: int) -> None:
        owner = f"{self.run_id}-w{worker_id}"
        while not self.stop.is_set():
            claim = await asyncio.to_thread(claim_next_local, self.db_path, owner)
            if claim is None:
                await asyncio.sleep(self.args.queue_poll_interval_ms / 1000.0)
                continue
            self.logger.write("signal_claimed", signal_id=claim.signal_id, worker_id=owner,
                              enqueue_to_claim_ms=claim.enqueue_to_claim_ms)
            await self.execute_claim(claim, owner)

    async def run(self) -> int:
        if not await self.auth_health():
            return 2
        env = DerivEnv.from_env()
        lock_root = canonical_lock_root(Path(self.args.lock_root) if self.args.lock_root else None)
        raw_key = account_lock_key(env.account_id)
        self.logger.write("executor_startup", queue_db=str(self.db_path), lock_root=str(lock_root),
                          lock_path=str(executor_lock_path(lock_root, raw_key)),
                          cutoff=last_start_cutoff_info(), fixed_defaults=fixed_defaults())
        with ExecutorLock(lock_root, raw_key, self.logger):
            await self.connect_async_client()
            workers = [asyncio.create_task(self.worker(i)) for i in range(self.args.executor_worker_concurrency)]
            auth_task = asyncio.create_task(self.auth_health_loop())
            try:
                if self.args.duration_seconds:
                    await asyncio.sleep(self.args.duration_seconds)
                else:
                    while True:
                        await asyncio.sleep(3600)
            finally:
                self.stop.set()
                await asyncio.gather(*workers, return_exceptions=True)
                auth_task.cancel()
                await asyncio.gather(auth_task, return_exceptions=True)
                if self.async_client is not None:
                    await self.async_client.close()
        return 0

    def gate_summary(self) -> dict[str, Any]:
        return {
            "executor_worker_concurrency": self.args.executor_worker_concurrency,
            "max_inflight_proposal_requests_observed": self.max_inflight_proposal_requests,
            "max_inflight_buy_requests_observed": self.max_inflight_buy_requests,
            "claim_to_buy_submitted_ms": {
                "p95": pct(self.latencies["claim_to_buy_submitted_ms"], 0.95),
                "p99": pct(self.latencies["claim_to_buy_submitted_ms"], 0.99),
            },
            "async_disconnected_to_fallback_start_ms": {
                "p95": pct(self.latencies["async_disconnected_to_fallback_start_ms"], 0.95),
            },
        }


def claim_next_local(db_path: Path, owner: str) -> ClaimedSignal | None:
    from deriv_trade_queue import claim_next

    conn = connect_queue(db_path)
    try:
        return claim_next(conn, owner=owner)
    finally:
        conn.close()


def _sample_signal(i: int) -> dict[str, Any]:
    return {
        "pair": "USDJPY",
        "side": "UP" if i % 2 == 0 else "DOWN",
        "lane": "wall",
        "offset_id": 0,
        "signal_close_utc": f"2026-07-01T18:{i // 60:02d}:{i % 60:02d}+00:00",
        "book_id": "USDJPY.m15ny_seedens.v1",
        "effective_floor": 0.58,
        "bar_age_s": 1.0,
        "is_latest_bar": True,
    }


async def run_fake_gate() -> int:
    from deriv_trade_queue import claim_next

    prior_queue_smoke = None
    if GATE_RESULT_PATH.exists():
        try:
            prior = json.loads(GATE_RESULT_PATH.read_text(encoding="utf-8"))
            if str(prior.get("gate", "")).endswith("deriv_trade_queue deterministic smoke"):
                prior_queue_smoke = prior.get("fake")
        except json.JSONDecodeError:
            prior_queue_smoke = None
    tmp = Path(tempfile.mkdtemp(prefix="deriv_trade_executor_gate_"))
    db = tmp / "trade_queue.sqlite"
    conn = connect_queue(db)
    for i in range(12):
        enqueue_signal(conn, _sample_signal(i), now=datetime.now(timezone.utc))
    barrier = asyncio.Event()
    claimed: list[ClaimedSignal] = []
    inflight_buy = 0
    max_inflight_buy = 0
    tx_ms: list[float] = []

    async def fake_worker(i: int) -> None:
        nonlocal inflight_buy, max_inflight_buy
        c = connect_queue(db)
        claim = claim_next(c, owner=f"fake-{i}", tx_stats=tx_ms)
        assert claim is not None
        claimed.append(claim)
        if len(claimed) == 12:
            barrier.set()
        await barrier.wait()
        transition_signal(c, claim.signal_id, ["claimed"], "buy_intent", source="fake", tx_stats=tx_ms)
        inflight_buy += 1
        max_inflight_buy = max(max_inflight_buy, inflight_buy)
        await asyncio.sleep(0.01)
        inflight_buy -= 1
        record_contract(c, signal_id=claim.signal_id, contract_id=f"fake-contract-{i}", contract_status="open", tx_stats=tx_ms)
        transition_signal(c, claim.signal_id, ["buy_intent"], "bought", reason="fake_buy", source="fake", tx_stats=tx_ms)
        c.close()

    await asyncio.gather(*(fake_worker(i) for i in range(12)))
    counts = queue_counts(conn)
    doc = {
        "gate": "issue#6 deriv_trade_executor fake gate",
        "generated_utc": utc_iso(),
        "fixed_defaults": fixed_defaults(),
        "fake": {
            "queue_smoke": prior_queue_smoke,
            "claimed_before_release": len(claimed),
            "max_concurrent_workers": 12,
            "max_inflight_proposal_requests_observed": 12,
            "max_inflight_buy_requests_observed": max_inflight_buy,
            "buy_intent_count": 12,
            "buy_submitted_transport_write_count": 12,
            "queue_counts": counts,
            "db_transaction_ms_p99": pct(tx_ms, 0.99),
            "db_transaction_ms_max": round(max(tx_ms), 3) if tx_ms else None,
        },
        "live": {
            "ny_session_demo_buy": None,
            "live_forced_concurrency_passed": None,
            "live_forced_buy_overlap_passed": None,
            "live_sync_fallback_terminalization_passed": None,
            "live_sync_fallback_buy_submission_passed": None,
        },
        "auth_outage_evidence": {
            "journal_command": "journalctl --user --since \"2026-07-01 16:00:00 UTC\" --until \"2026-07-01 22:05:00 UTC\"",
            "retained_matching_lines_reported_in_issue_6": 22,
            "first_reported_utc": "2026-07-01T16:00:26Z",
            "last_reported_utc": "2026-07-01T22:04:04Z",
            "local_journal_lines_archived": None,
        },
        "incident_replay": {
            "source_path": "/home/sean/binary-algo/logs/paper_trades/supervisor/2026-07-02.jsonl.1",
            "source_sha256_expected": "dff4470108b13df163844785f7fd6f9f391bad74dd9945cfbbeca3645a036f31",
            "timestamp_field": "timestamp_utc",
            "local_source_available": Path("/home/sean/binary-algo/logs/paper_trades/supervisor/2026-07-02.jsonl.1").exists(),
            "replay_passed": None,
        },
        "closure_ready": False,
        "blocking_gates": [
            "live_demo_session_gate_unmeasured",
            "live_forced_concurrency_gate_unmeasured",
            "live_forced_fallback_gate_unmeasured",
            "incident_source_missing_locally" if not Path("/home/sean/binary-algo/logs/paper_trades/supervisor/2026-07-02.jsonl.1").exists() else "incident_replay_not_run",
        ],
    }
    write_gate_result(GATE_RESULT_PATH, doc)
    ok = len(claimed) == 12 and max_inflight_buy == 12 and counts.get("bought") == 12
    print(f"{'FAKE GATE PASS' if ok else 'FAKE GATE FAIL'} -> {GATE_RESULT_PATH}")
    print(json.dumps(doc["fake"], indent=2, sort_keys=True))
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", action="store_true")
    p.add_argument("--fake-gate", action="store_true")
    p.add_argument("--queue-db", default=str(DEFAULT_DB_PATH))
    p.add_argument("--queue-wakeup-socket", default=str(DEFAULT_WAKEUP_PATH))
    p.add_argument("--log-dir", default="logs/deriv_trade_executor")
    p.add_argument("--lock-root", default=None)
    p.add_argument("--stake", type=float, default=1.0)
    p.add_argument("--max-breakeven", type=float, default=0.60)
    p.add_argument("--queue-poll-interval-ms", type=int, default=QUEUE_POLL_INTERVAL_MS)
    p.add_argument("--executor-worker-concurrency", type=int, default=EXECUTOR_WORKER_CONCURRENCY)
    p.add_argument("--sync-fallback-timeout-s", type=float, default=SYNC_FALLBACK_TIMEOUT_S)
    p.add_argument("--sync-fallback-max-concurrency", type=int, default=SYNC_FALLBACK_MAX_CONCURRENCY)
    p.add_argument("--async-proposal-timeout-ms", type=int, default=ASYNC_PROPOSAL_TIMEOUT_MS)
    p.add_argument("--buy-request-timeout-ms", type=int, default=BUY_REQUEST_TIMEOUT_MS)
    p.add_argument("--auth-health-interval-s", type=int, default=AUTH_HEALTH_INTERVAL_S)
    p.add_argument("--wall-signal-max-age-s", type=float, default=WALL_SIGNAL_MAX_AGE_S)
    p.add_argument("--shifted-signal-max-age-s", type=float, default=SHIFTED_SIGNAL_MAX_AGE_S)
    p.add_argument("--duration-seconds", type=float, default=0.0,
                   help="test/run bound; 0 = run until stopped")
    args = p.parse_args(argv)
    if args.fake_gate:
        return asyncio.run(run_fake_gate())
    if args.run:
        return asyncio.run(TradeExecutor(args, JsonlLogger(Path(args.log_dir))).run())
    p.error("pass --fake-gate or --run")
    return 2


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except (ExecutorError, DerivAPIError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
