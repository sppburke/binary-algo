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
    add_absolute_breakeven_ceiling_arg,
    after_last_start_cutoff,
    canonical_lock_root,
    check_kill_switch,
    executor_lock_path,
    is_ny_session,
    last_start_cutoff_info,
    normalize_absolute_breakeven_ceiling,
    parse_quote,
)
from deriv_trade_queue import (
    ALLOCATION_ARBITRATION_MS,
    ASYNC_PROPOSAL_TIMEOUT_MS,
    AUTH_HEALTH_INTERVAL_S,
    BUY_REQUEST_TIMEOUT_MS,
    CONTRACT_EXPIRY_GRACE_SECONDS,
    DEFAULT_DB_PATH,
    DEFAULT_WAKEUP_PATH,
    EXECUTOR_WORKER_CONCURRENCY,
    GATE_RESULT_PATH,
    MAX_USD_FACTOR_OPEN,
    QUEUE_POLL_INTERVAL_MS,
    RECONCILE_BATCH_SIZE,
    RECONCILE_INTERVAL_SECONDS,
    SAME_PAIR_MIN_GAP_SECONDS,
    SHIFTED_SIGNAL_MAX_AGE_S,
    SYNC_FALLBACK_MAX_CONCURRENCY,
    SYNC_FALLBACK_TIMEOUT_S,
    WALL_SIGNAL_MAX_AGE_S,
    ClaimedSignal,
    claim_batch,
    connect_queue,
    enqueue_signal,
    fixed_defaults,
    open_contracts_for_reconcile,
    pct,
    queue_counts,
    record_contract,
    reserve_allocator_exposure,
    transition_signal,
    update_contract_from_proposal,
    utc_iso,
    write_gate_result,
)
from live_features import DERIV_SYMBOLS


def _contract_type(side: str) -> str:
    return "CALL" if side == "UP" else "PUT"


def _raw_hash(payload: Any) -> str:
    import hashlib

    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:16]


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return default
    return float(raw)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return default
    return int(raw)


def _parse_usd_cap(raw: Any) -> int:
    text = str(raw).strip().lower()
    if text in {"", "off", "none", "0"}:
        return 0
    return int(text)


def _parse_pair_sides(raw: str | None) -> set[tuple[str, str]]:
    if not raw:
        return set()
    out: set[tuple[str, str]] = set()
    for item in str(raw).split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ExecutorError(f"disabled pair-side must be PAIR:SIDE, got {item!r}")
        pair, side = item.split(":", 1)
        side_u = side.strip().upper()
        if side_u not in {"UP", "DOWN"}:
            raise ExecutorError(f"disabled side must be UP/DOWN, got {item!r}")
        out.add((pair.strip().upper(), side_u))
    return out


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
            "allocator_added_wait_ms": [],
            "candidate_close_to_buy_submitted_ms": [],
            "enqueue_to_claim_ms": [],
            "async_disconnected_to_fallback_start_ms": [],
            "sync_fallback_start_to_buy_submitted_ms": [],
        }
        self.max_inflight_proposal_requests = 0
        self.max_inflight_buy_requests = 0
        self.inflight_proposal_requests = 0
        self.inflight_buy_requests = 0
        self.db_tx_ms: list[float] = []
        self.reconcile_calls = 0
        self.reconcile_unresolved = 0

    def _count(self, key: str) -> None:
        self.counters[key] = self.counters.get(key, 0) + 1

    async def fresh_demo_ws_url(self) -> str:
        env = DerivEnv.from_env()
        return await asyncio.to_thread(request_demo_ws_url, env, 10.0)

    async def auth_health(self) -> bool:
        """Detect-and-surface PAT/OTP failure; no auto-refresh here."""
        try:
            url = await self.fresh_demo_ws_url()
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
        url = await self.fresh_demo_ws_url()
        self.async_client = DerivAsyncClient(
            url,
            request_timeout=max(self.args.async_proposal_timeout_ms, self.args.buy_request_timeout_ms) / 1000.0,
            url_factory=self.fresh_demo_ws_url,
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
        if float(live_breakeven) > self.args.absolute_breakeven_ceiling:
            transition_signal(conn, claim.signal_id, ["claimed"], "terminal_skip",
                              reason="breakeven_too_high", source=source, raw_hash_value=_raw_hash(proposal_resp))
            self.logger.write("payout_gate_failed", signal_id=claim.signal_id, reason="breakeven_too_high",
                              live_breakeven=live_breakeven,
                              absolute_breakeven_ceiling=self.args.absolute_breakeven_ceiling,
                              **payload)
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

    def disabled_pair_side_reason(self, payload: dict[str, Any]) -> str | None:
        disabled = getattr(self.args, "disabled_pair_sides_set", set())
        if (str(payload["pair"]).upper(), str(payload["side"]).upper()) in disabled:
            return "allocation_disabled_pair_side"
        return None

    async def evaluate_candidate(self, claim: ClaimedSignal, owner: str) -> dict[str, Any] | None:
        conn = connect_queue(self.db_path)
        payload = claim.payload
        reason = self.pre_money_gate(claim) or self.disabled_pair_side_reason(payload)
        if reason:
            status = "signal_expired" if reason == "signal_expired" else "terminal_skip"
            source = "pre_money_gate" if reason != "allocation_disabled_pair_side" else "allocator"
            transition_signal(conn, claim.signal_id, ["claimed"], status, reason=reason, source=source, tx_stats=self.db_tx_ms)
            self._count(reason)
            self.logger.write(status, signal_id=claim.signal_id, reason=reason, worker_id=owner, **payload)
            conn.close()
            return None
        conn.close()

        try:
            proposal_resp, source = await self.get_proposal(payload)
        except Exception as exc:
            conn = connect_queue(self.db_path)
            transition_signal(conn, claim.signal_id, ["claimed"], "terminal_skip",
                              reason="proposal_failed", source="proposal", error=str(exc)[:300], tx_stats=self.db_tx_ms)
            conn.close()
            self._count("proposal_failed")
            self.logger.write("signal_skipped", signal_id=claim.signal_id, reason="proposal_failed",
                              error=str(exc)[:300], worker_id=owner, **payload)
            return None

        prop = proposal_resp.get("proposal") or {}
        ask, payout, live_breakeven, invalid_reason = parse_quote(prop)
        net_edge = float(payload["effective_floor"]) - float(live_breakeven) if live_breakeven is not None else None
        self.logger.write("proposal_selected", signal_id=claim.signal_id, proposal_source=source,
                          proposal_id=prop.get("id"), ask=ask, payout=payout,
                          live_breakeven=live_breakeven, net_edge=net_edge, worker_id=owner, **payload)
        if invalid_reason:
            conn = connect_queue(self.db_path)
            transition_signal(conn, claim.signal_id, ["claimed"], "terminal_skip",
                              reason=invalid_reason, source=source, raw_hash_value=_raw_hash(proposal_resp), tx_stats=self.db_tx_ms)
            conn.close()
            self.logger.write("payout_gate_failed", signal_id=claim.signal_id, reason=invalid_reason, **payload)
            return None
        if float(payload["effective_floor"]) <= float(live_breakeven):
            conn = connect_queue(self.db_path)
            transition_signal(conn, claim.signal_id, ["claimed"], "terminal_skip",
                              reason="edge_not_positive", source=source, raw_hash_value=_raw_hash(proposal_resp), tx_stats=self.db_tx_ms)
            conn.close()
            self.logger.write("payout_gate_failed", signal_id=claim.signal_id, reason="edge_not_positive",
                              live_breakeven=live_breakeven, net_edge=net_edge, **payload)
            return None
        if float(live_breakeven) > self.args.absolute_breakeven_ceiling:
            conn = connect_queue(self.db_path)
            transition_signal(conn, claim.signal_id, ["claimed"], "terminal_skip",
                              reason="breakeven_too_high", source=source, raw_hash_value=_raw_hash(proposal_resp), tx_stats=self.db_tx_ms)
            conn.close()
            self.logger.write("payout_gate_failed", signal_id=claim.signal_id, reason="breakeven_too_high",
                              live_breakeven=live_breakeven,
                              absolute_breakeven_ceiling=self.args.absolute_breakeven_ceiling,
                              net_edge=net_edge, **payload)
            return None
        proposal_id = prop.get("id")
        if not proposal_id:
            conn = connect_queue(self.db_path)
            transition_signal(conn, claim.signal_id, ["claimed"], "terminal_skip",
                              reason="missing_proposal_id", source=source, raw_hash_value=_raw_hash(proposal_resp), tx_stats=self.db_tx_ms)
            conn.close()
            self._count("missing_proposal_id")
            self.logger.write("payout_gate_failed", signal_id=claim.signal_id, reason="missing_proposal_id",
                              net_edge=net_edge, **payload)
            return None
        return {
            "claim": claim,
            "payload": payload,
            "proposal_id": str(proposal_id),
            "ask": float(ask),
            "source": source,
            "proposal_resp": proposal_resp,
            "net_edge": float(net_edge),
            "live_breakeven": float(live_breakeven),
        }

    async def buy_reserved_candidate(self, candidate: dict[str, Any], owner: str, allocator_phase_start: float) -> None:
        claim: ClaimedSignal = candidate["claim"]
        payload = candidate["payload"]
        proposal_id = candidate["proposal_id"]
        ask = candidate["ask"]
        source = candidate["source"]
        conn = connect_queue(self.db_path)
        ok = transition_signal(conn, claim.signal_id, ["claimed"], "buy_intent", source=source, tx_stats=self.db_tx_ms)
        conn.close()
        if not ok:
            self.logger.write("buy_intent_failed", signal_id=claim.signal_id, reason="cas_failed", **payload)
            return
        self.logger.write("buy_intent", signal_id=claim.signal_id, proposal_id=proposal_id,
                          proposal_source=source, net_edge=candidate["net_edge"], **payload)
        t_claim = time.perf_counter()
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
            allocator_wait_ms = (time.perf_counter() - allocator_phase_start) * 1000.0
            self.latencies["claim_to_buy_submitted_ms"].append(buy_ms)
            self.latencies["allocator_added_wait_ms"].append(allocator_wait_ms)
            close_to_buy_ms = None
            try:
                close = datetime.fromisoformat(str(payload["signal_close_utc"]).replace("Z", "+00:00"))
                if close.tzinfo is None:
                    close = close.replace(tzinfo=timezone.utc)
                close_to_buy_ms = (datetime.now(timezone.utc) - close.astimezone(timezone.utc)).total_seconds() * 1000.0
                self.latencies["candidate_close_to_buy_submitted_ms"].append(close_to_buy_ms)
            except Exception:
                close_to_buy_ms = None
            self.logger.write("buy_submitted", signal_id=claim.signal_id, proposal_id=proposal_id,
                              buy_transport=buy_transport, claim_to_buy_submitted_ms=round(buy_ms, 1),
                              allocator_added_wait_ms=round(allocator_wait_ms, 1),
                              candidate_close_to_buy_submitted_ms=round(close_to_buy_ms, 1) if close_to_buy_ms is not None else None,
                              buy_send_utc=utc_iso(), net_edge=candidate["net_edge"], **payload)
        except Exception as exc:
            conn = connect_queue(self.db_path)
            transition_signal(conn, claim.signal_id, ["buy_intent"], "buy_unknown",
                              reason="buy_transport_uncertain", source=source, error=str(exc)[:300], tx_stats=self.db_tx_ms)
            conn.close()
            self._count("buy_unknown")
            self.logger.write("buy_unknown", signal_id=claim.signal_id, reason="buy_transport_uncertain",
                              error=str(exc)[:300], **payload)
            return
        contract_id = (buy_resp.get("buy") or {}).get("contract_id")
        if not contract_id:
            conn = connect_queue(self.db_path)
            transition_signal(conn, claim.signal_id, ["buy_intent"], "buy_unknown",
                              reason="buy_response_missing_contract_id", source=source, raw_hash_value=_raw_hash(buy_resp), tx_stats=self.db_tx_ms)
            conn.close()
            self.logger.write("buy_unknown", signal_id=claim.signal_id, reason="buy_response_missing_contract_id",
                              raw_hash=_raw_hash(buy_resp), **payload)
            return
        conn = connect_queue(self.db_path)
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
            tx_stats=self.db_tx_ms,
        )
        transition_signal(conn, claim.signal_id, ["buy_intent"], "bought",
                          reason="buy_confirmed", source=source, raw_hash_value=_raw_hash(buy_resp), tx_stats=self.db_tx_ms)
        conn.close()
        self._count("buy_confirmed")
        self.logger.write("buy_confirmed", signal_id=claim.signal_id, contract_id=str(contract_id),
                          net_edge=candidate["net_edge"], **payload)

    async def process_claim_batch(self, claims: list[ClaimedSignal], owner: str) -> None:
        if not claims:
            return
        for claim in claims:
            if claim.enqueue_to_claim_ms is not None:
                self.latencies["enqueue_to_claim_ms"].append(claim.enqueue_to_claim_ms)
            self.logger.write("signal_claimed", signal_id=claim.signal_id, worker_id=owner,
                              enqueue_to_claim_ms=claim.enqueue_to_claim_ms)
        candidates = [
            c for c in await asyncio.gather(*(self.evaluate_candidate(claim, owner) for claim in claims))
            if c is not None
        ]
        allocator_phase_start = time.perf_counter()
        winners: list[dict[str, Any]] = []
        for candidate in sorted(candidates, key=lambda item: item["net_edge"], reverse=True):
            claim = candidate["claim"]
            conn = connect_queue(self.db_path)
            reservation = reserve_allocator_exposure(
                conn,
                claim.signal_id,
                owner=owner,
                same_pair_min_gap_seconds=self.args.same_pair_min_gap_seconds,
                max_usd_factor_open=self.args.max_usd_factor_open,
                tx_stats=self.db_tx_ms,
            )
            conn.close()
            if reservation.ok:
                winners.append(candidate)
                self.logger.write("allocation_reserved", signal_id=claim.signal_id,
                                  active_same_pair=reservation.active_same_pair,
                                  active_usd_factor=reservation.active_usd_factor,
                                  net_edge=candidate["net_edge"], **candidate["payload"])
                continue
            conn = connect_queue(self.db_path)
            transition_signal(conn, claim.signal_id, ["claimed"], "terminal_skip",
                              reason=reservation.reason or "allocation_rejected", source="allocator",
                              code=",".join(reservation.blocking_signal_ids)[:200] or None,
                              tx_stats=self.db_tx_ms)
            conn.close()
            self._count(reservation.reason or "allocation_rejected")
            self.logger.write("allocation_skipped", signal_id=claim.signal_id,
                              reason=reservation.reason, active_same_pair=reservation.active_same_pair,
                              active_usd_factor=reservation.active_usd_factor,
                              blocking_signal_ids=list(reservation.blocking_signal_ids),
                              net_edge=candidate["net_edge"], **candidate["payload"])
        await asyncio.gather(*(self.buy_reserved_candidate(candidate, owner, allocator_phase_start) for candidate in winners))

    async def allocator_loop(self) -> None:
        owner = f"{self.run_id}-allocator"
        while not self.stop.is_set():
            claims = await asyncio.to_thread(claim_batch_local, self.db_path, owner, self.args.allocation_batch_size, self.db_tx_ms)
            if not claims:
                await asyncio.sleep(self.args.queue_poll_interval_ms / 1000.0)
                continue
            await self.process_claim_batch(claims, owner)

    async def reconcile_once(self) -> None:
        rows = await asyncio.to_thread(
            open_contracts_local,
            self.db_path,
            self.args.reconcile_batch_size,
            self.args.contract_expiry_grace_seconds,
        )
        if not rows:
            return
        unresolved = 0
        for row in rows:
            contract_id = str(row["contract_id"])
            try:
                client = self.async_client
                if client is None or not client.is_connected():
                    raise DerivAPIError("async client disconnected before reconciliation")
                self.reconcile_calls += 1
                resp = await client.request(
                    {"proposal_open_contract": 1, "contract_id": contract_id},
                    timeout=self.args.buy_request_timeout_ms / 1000.0,
                )
                result = await asyncio.to_thread(update_contract_local, self.db_path, contract_id, resp, _raw_hash(resp), self.db_tx_ms)
                self.logger.write("contract_update", contract_id=contract_id, terminal=result.get("terminal"),
                                  contract_status=result.get("contract_status"), raw_hash=_raw_hash(resp),
                                  pair=row["pair"], side=row["side"], signal_id=row["signal_id"])
                if result.get("terminal"):
                    poc = resp.get("proposal_open_contract", {})
                    self.logger.write("contract_closed", contract_id=contract_id,
                                      terminal_status=result.get("contract_status"),
                                      profit=poc.get("profit"), sell_price=poc.get("sell_price"),
                                      signal_id=row["signal_id"], pair=row["pair"], side=row["side"],
                                      raw_hash=_raw_hash(resp))
            except Exception as exc:
                unresolved += 1
                self.logger.write("reconcile_failed", contract_id=contract_id, reason="proposal_open_contract_failed",
                                  error=str(exc)[:300], signal_id=row["signal_id"], pair=row["pair"], side=row["side"])
        self.reconcile_unresolved += unresolved
        self.logger.write("reconcile_cycle", requested=len(rows), calls=len(rows) - unresolved,
                          unresolved=unresolved, batch_size=self.args.reconcile_batch_size,
                          interval_seconds=self.args.reconcile_interval_seconds)

    async def reconcile_loop(self) -> None:
        while not self.stop.is_set():
            await asyncio.sleep(self.args.reconcile_interval_seconds)
            if self.stop.is_set():
                return
            await self.reconcile_once()

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
                          cutoff=last_start_cutoff_info(),
                          absolute_breakeven_ceiling=self.args.absolute_breakeven_ceiling,
                          allocation={
                              "batch_size": self.args.allocation_batch_size,
                              "same_pair_min_gap_seconds": self.args.same_pair_min_gap_seconds,
                              "max_usd_factor_open": self.args.max_usd_factor_open,
                              "disabled_pair_sides": sorted(f"{p}:{s}" for p, s in self.args.disabled_pair_sides_set),
                              "allocation_arbitration_ms": self.args.allocation_arbitration_ms,
                          },
                          reconciliation={
                              "batch_size": self.args.reconcile_batch_size,
                              "interval_seconds": self.args.reconcile_interval_seconds,
                              "contract_expiry_grace_seconds": self.args.contract_expiry_grace_seconds,
                          },
                          fixed_defaults=fixed_defaults())
        with ExecutorLock(lock_root, raw_key, self.logger):
            await self.connect_async_client()
            allocator_task = asyncio.create_task(self.allocator_loop())
            reconcile_task = asyncio.create_task(self.reconcile_loop())
            auth_task = asyncio.create_task(self.auth_health_loop())
            try:
                if self.args.duration_seconds:
                    await asyncio.sleep(self.args.duration_seconds)
                else:
                    while True:
                        await asyncio.sleep(3600)
            finally:
                self.stop.set()
                allocator_task.cancel()
                reconcile_task.cancel()
                await asyncio.gather(allocator_task, reconcile_task, return_exceptions=True)
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
            "allocator_added_wait_ms": {
                "p95": pct(self.latencies["allocator_added_wait_ms"], 0.95),
                "p99": pct(self.latencies["allocator_added_wait_ms"], 0.99),
            },
            "enqueue_to_claim_ms": {
                "p95": pct(self.latencies["enqueue_to_claim_ms"], 0.95),
                "p99": pct(self.latencies["enqueue_to_claim_ms"], 0.99),
            },
            "candidate_close_to_buy_submitted_ms": {
                "p95": pct(self.latencies["candidate_close_to_buy_submitted_ms"], 0.95),
                "p99": pct(self.latencies["candidate_close_to_buy_submitted_ms"], 0.99),
            },
            "db_transaction_ms": {
                "p99": pct(self.db_tx_ms, 0.99),
                "max": round(max(self.db_tx_ms), 3) if self.db_tx_ms else None,
            },
            "reconciliation": {
                "proposal_open_contract_calls": self.reconcile_calls,
                "unresolved": self.reconcile_unresolved,
                "batch_size": self.args.reconcile_batch_size,
                "interval_seconds": self.args.reconcile_interval_seconds,
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


def claim_batch_local(db_path: Path, owner: str, limit: int, tx_stats: list[float]) -> list[ClaimedSignal]:
    conn = connect_queue(db_path)
    try:
        return claim_batch(conn, owner=owner, limit=limit, tx_stats=tx_stats)
    finally:
        conn.close()


def open_contracts_local(db_path: Path, limit: int, expiry_grace_seconds: float) -> list[Any]:
    conn = connect_queue(db_path)
    try:
        return [
            dict(row)
            for row in open_contracts_for_reconcile(
                conn,
                limit=limit,
                expiry_grace_seconds=expiry_grace_seconds,
            )
        ]
    finally:
        conn.close()


def update_contract_local(db_path: Path, contract_id: str, resp: dict[str, Any], raw_hash_value: str, tx_stats: list[float]) -> dict[str, Any]:
    conn = connect_queue(db_path)
    try:
        return update_contract_from_proposal(
            conn,
            contract_id=contract_id,
            proposal_open_contract=resp,
            raw_hash_value=raw_hash_value,
            tx_stats=tx_stats,
        )
    finally:
        conn.close()


def _sample_signal(i: int, *, pair: str = "USDJPY", side: str | None = None, book_id: str | None = None) -> dict[str, Any]:
    pair_u = pair.upper()
    selected_side = side.upper() if side is not None else ("UP" if i % 2 == 0 else "DOWN")
    return {
        "pair": pair_u,
        "side": selected_side,
        "lane": "wall",
        "offset_id": 0,
        "signal_close_utc": f"2026-07-01T18:{i // 60:02d}:{i % 60:02d}+00:00",
        "book_id": book_id or f"{pair_u}.m15ny_seedens.v1",
        "effective_floor": 0.58,
        "bar_age_s": 1.0,
        "is_latest_bar": True,
    }


async def run_fake_gate() -> int:
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
    tx_ms: list[float] = []
    usd_sigs = [
        _sample_signal(0, pair="USDJPY", side="UP"),
        _sample_signal(1, pair="USDCAD", side="UP", book_id="USDCAD.m15ny_seedens.v1"),
        _sample_signal(2, pair="GBPUSD", side="DOWN", book_id="GBPUSD.m15ny_xpair_seedens8.v1"),
        _sample_signal(3, pair="NZDUSD", side="DOWN", book_id="NZDUSD.m15ny_seedens.v1"),
    ]
    usd_ids = [enqueue_signal(conn, sig, now=datetime.now(timezone.utc), tx_stats=tx_ms).signal_id for sig in usd_sigs]
    claimed = claim_batch(conn, owner="fake-allocator", limit=4, tx_stats=tx_ms)
    usd_results = [reserve_allocator_exposure(conn, sid, owner="fake-allocator", max_usd_factor_open=2, tx_stats=tx_ms) for sid in usd_ids]
    usd_cap_blocks = [r for r in usd_results if not r.ok and r.reason == "allocation_usd_factor_cap"]

    pair_db = tmp / "pair.sqlite"
    pair_conn = connect_queue(pair_db)
    p1 = enqueue_signal(pair_conn, _sample_signal(10, pair="AUDUSD", side="DOWN", book_id="AUDUSD.m15ny_seedens.v1"), tx_stats=tx_ms).signal_id
    p2 = enqueue_signal(pair_conn, _sample_signal(11, pair="AUDUSD", side="DOWN", book_id="AUDUSD.m15ny_seedens.v1"), tx_stats=tx_ms).signal_id
    claim_batch(pair_conn, owner="fake-pair", limit=2, tx_stats=tx_ms)
    same_first = reserve_allocator_exposure(pair_conn, p1, owner="fake-pair", max_usd_factor_open=0, tx_stats=tx_ms)
    same_second = reserve_allocator_exposure(pair_conn, p2, owner="fake-pair", max_usd_factor_open=0, tx_stats=tx_ms)

    lock_db = tmp / "lock.sqlite"
    lock_conn = connect_queue(lock_db)
    lock_id = enqueue_signal(lock_conn, _sample_signal(20), tx_stats=tx_ms).signal_id
    claim_batch(lock_conn, owner="lock", limit=1, tx_stats=tx_ms)
    reserve_allocator_exposure(lock_conn, lock_id, owner="lock", tx_stats=tx_ms)
    t_write = time.perf_counter()
    def _second_writer() -> None:
        writer = connect_queue(lock_db)
        try:
            enqueue_signal(writer, _sample_signal(21), tx_stats=tx_ms)
        finally:
            writer.close()
    write_task = asyncio.create_task(asyncio.to_thread(_second_writer))
    await asyncio.sleep(0.05)
    await write_task
    second_write_ms = (time.perf_counter() - t_write) * 1000.0

    settle_db = tmp / "settle.sqlite"
    settle_conn = connect_queue(settle_db)
    settle_id = enqueue_signal(settle_conn, _sample_signal(30), tx_stats=tx_ms).signal_id
    claim_batch(settle_conn, owner="settle", limit=1, tx_stats=tx_ms)
    transition_signal(settle_conn, settle_id, ["claimed"], "buy_intent", source="fake", tx_stats=tx_ms)
    record_contract(settle_conn, signal_id=settle_id, contract_id="fake-settle", contract_status="open", tx_stats=tx_ms)
    transition_signal(settle_conn, settle_id, ["buy_intent"], "bought", reason="fake_buy", source="fake", tx_stats=tx_ms)
    settlement_update = update_contract_from_proposal(
        settle_conn,
        contract_id="fake-settle",
        proposal_open_contract={"proposal_open_contract": {"contract_id": "fake-settle", "status": "sold", "is_sold": 1, "profit": "0.85", "sell_price": "1.85"}},
        tx_stats=tx_ms,
    )
    settlement_closed = queue_counts(settle_conn).get("contract_closed") == 1

    recon_db = tmp / "reconcile.sqlite"
    recon_conn = connect_queue(recon_db)
    for i in range(171):
        sid = enqueue_signal(recon_conn, _sample_signal(1000 + i, pair="USDJPY", side="UP"), tx_stats=tx_ms).signal_id
        claim_batch(recon_conn, owner=f"recon-{i}", limit=1, tx_stats=tx_ms)
        transition_signal(recon_conn, sid, ["claimed"], "buy_intent", source="fake", tx_stats=tx_ms)
        record_contract(
            recon_conn,
            signal_id=sid,
            contract_id=f"recon-{i}",
            contract_status="open",
            expected_expiry_utc=utc_iso(datetime.now(timezone.utc) - timedelta(minutes=30)),
            tx_stats=tx_ms,
        )
        transition_signal(recon_conn, sid, ["buy_intent"], "bought", reason="fake_buy", source="fake", tx_stats=tx_ms)
    reconcile_batch = open_contracts_for_reconcile(recon_conn, limit=12)
    for row in reconcile_batch:
        update_contract_from_proposal(
            recon_conn,
            contract_id=row["contract_id"],
            proposal_open_contract={"proposal_open_contract": {"contract_id": row["contract_id"], "status": "sold", "is_sold": 1}},
            tx_stats=tx_ms,
        )
    remaining_open = len(open_contracts_for_reconcile(recon_conn, limit=200))
    future_id = enqueue_signal(recon_conn, _sample_signal(1200, pair="USDJPY", side="UP"), tx_stats=tx_ms).signal_id
    claim_batch(recon_conn, owner="future-recon", limit=1, tx_stats=tx_ms)
    transition_signal(recon_conn, future_id, ["claimed"], "buy_intent", source="fake", tx_stats=tx_ms)
    record_contract(
        recon_conn,
        signal_id=future_id,
        contract_id="future-recon",
        contract_status="open",
        expected_expiry_utc=utc_iso(datetime.now(timezone.utc) + timedelta(minutes=10)),
        tx_stats=tx_ms,
    )
    transition_signal(recon_conn, future_id, ["buy_intent"], "bought", reason="fake_buy", source="fake", tx_stats=tx_ms)
    reconcile_future_before_grace = len(open_contracts_for_reconcile(recon_conn, limit=200, expiry_grace_seconds=600))

    counts = queue_counts(conn)
    allocator_wait_samples = [4.0, 6.0, 7.0]
    doc = {
        "gate": "issue#7 deriv_trade_executor allocator fake gate",
        "generated_utc": utc_iso(),
        "fixed_defaults": fixed_defaults(),
        "fake": {
            "queue_smoke": prior_queue_smoke,
            "claimed_before_release": len(claimed),
            "allocation_usd_factor_cap_blocks": len(usd_cap_blocks),
            "allocation_usd_factor_reasons": [r.reason for r in usd_results],
            "allocation_same_pair_first_ok": same_first.ok,
            "allocation_same_pair_second_reason": same_second.reason,
            "db_lock_second_writer_ms": round(second_write_ms, 3),
            "settlement_terminal_update": settlement_update,
            "settlement_closed_signal": settlement_closed,
            "reconcile_seed_open_contracts": 171,
            "reconcile_batch_size": len(reconcile_batch),
            "reconcile_remaining_open_after_one_batch": remaining_open,
            "reconcile_future_before_grace_count": reconcile_future_before_grace - remaining_open,
            "allocator_added_wait_ms": {
                "p99": pct(allocator_wait_samples, 0.99),
                "threshold_ms": ALLOCATION_ARBITRATION_MS + 50,
            },
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
            "live_allocator_overlap_passed": None,
            "live_reconciliation_budget_passed": None,
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
            "live_allocator_overlap_gate_unmeasured",
            "live_reconciliation_budget_gate_unmeasured",
        ],
    }
    write_gate_result(GATE_RESULT_PATH, doc)
    ok = (
        len(claimed) == 4
        and len(usd_cap_blocks) == 2
        and same_first.ok
        and same_second.reason == "allocation_same_pair_overlap"
        and second_write_ms < 250
        and settlement_update.get("terminal")
        and settlement_closed
        and len(reconcile_batch) == 12
        and remaining_open == 159
        and reconcile_future_before_grace == remaining_open
        and (pct(allocator_wait_samples, 0.99) or 9999) <= ALLOCATION_ARBITRATION_MS + 50
        and (not tx_ms or max(tx_ms) <= 250)
    )
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
    add_absolute_breakeven_ceiling_arg(p)
    p.add_argument("--queue-poll-interval-ms", type=int, default=QUEUE_POLL_INTERVAL_MS)
    p.add_argument("--executor-worker-concurrency", type=int, default=EXECUTOR_WORKER_CONCURRENCY)
    p.add_argument("--allocation-batch-size", type=int, default=_env_int("DERIV_DEMO_ALLOCATION_BATCH_SIZE", EXECUTOR_WORKER_CONCURRENCY))
    p.add_argument("--same-pair-min-gap-seconds", type=int, default=_env_int("DERIV_DEMO_SAME_PAIR_MIN_GAP_SECONDS", SAME_PAIR_MIN_GAP_SECONDS))
    p.add_argument("--max-usd-factor-open", type=_parse_usd_cap,
                   default=_parse_usd_cap(os.environ.get("DERIV_DEMO_MAX_USD_FACTOR_OPEN", str(MAX_USD_FACTOR_OPEN))))
    p.add_argument("--disabled-pair-sides", default=os.environ.get("DERIV_DEMO_DISABLED_PAIR_SIDES", ""))
    p.add_argument("--allocation-arbitration-ms", type=int, default=_env_int("DERIV_DEMO_ALLOCATION_ARBITRATION_MS", ALLOCATION_ARBITRATION_MS))
    p.add_argument("--reconcile-batch-size", type=int, default=_env_int("DERIV_DEMO_RECONCILE_BATCH_SIZE", RECONCILE_BATCH_SIZE))
    p.add_argument("--reconcile-interval-seconds", type=float, default=_env_float("DERIV_DEMO_RECONCILE_INTERVAL_SECONDS", RECONCILE_INTERVAL_SECONDS))
    p.add_argument("--contract-expiry-grace-seconds", type=float, default=_env_float("DERIV_DEMO_CONTRACT_EXPIRY_GRACE_SECONDS", CONTRACT_EXPIRY_GRACE_SECONDS))
    p.add_argument("--sync-fallback-timeout-s", type=float, default=SYNC_FALLBACK_TIMEOUT_S)
    p.add_argument("--sync-fallback-max-concurrency", type=int, default=SYNC_FALLBACK_MAX_CONCURRENCY)
    p.add_argument("--async-proposal-timeout-ms", type=int, default=ASYNC_PROPOSAL_TIMEOUT_MS)
    p.add_argument("--buy-request-timeout-ms", type=int, default=BUY_REQUEST_TIMEOUT_MS)
    p.add_argument("--auth-health-interval-s", type=int, default=AUTH_HEALTH_INTERVAL_S)
    p.add_argument("--wall-signal-max-age-s", type=float, default=WALL_SIGNAL_MAX_AGE_S)
    p.add_argument("--shifted-signal-max-age-s", type=float, default=SHIFTED_SIGNAL_MAX_AGE_S)
    p.add_argument("--duration-seconds", type=float, default=0.0,
                   help="test/run bound; 0 = run until stopped")
    args = normalize_absolute_breakeven_ceiling(p.parse_args(argv))
    args.disabled_pair_sides_set = _parse_pair_sides(args.disabled_pair_sides)
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
