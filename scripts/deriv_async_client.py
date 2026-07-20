"""Async Deriv WebSocket client (issue #4 Phase 1).

`websockets`-based transport for the hot runtime: req_id-correlated requests,
a subscription registry, reconnect with automatic resubscribe, protocol-level
heartbeat, and a fail-closed error surface — pending requests are rejected on
disconnect, stream errors are marked on the subscription, and unroutable
frames are counted and retained in a ring buffer, never silently dropped.

Raises `deriv_client.DerivAPIError` so existing callers' error handling
(`call_deriv`-style) works unchanged. `scripts/deriv_client.py` (sync) remains
the one-shot fallback transport.

Built-in Phase-1 gates (run from repo root):
    ~/binary-algo-venv/bin/python scripts/deriv_async_client.py --smoke
    ~/binary-algo-venv/bin/python scripts/deriv_async_client.py --soak-minutes 30 --inject-disconnects 3

The smoke is deterministic (in-process fake server, no network). The soak is
the live half of the Phase-1 gate: resubscribe < 5s after each injected
disconnect, zero mis-correlated frames, zero silent frame loss (unmatched ==
0, queue overflow == 0); result JSON written for archive to results/json/.
"""

from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import socket
import sys
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable
from urllib.parse import urlsplit

import websockets
import websockets.exceptions

from deriv_client import PUBLIC_WS_URL, DerivAPIError

SOAK_RESULT_PATH = Path("deriv_async_soak_result.json")
UrlFactory = Callable[[], str | Awaitable[str]]


def _safe_endpoint(url: str) -> str:
    parts = urlsplit(url)
    if not parts.scheme or not parts.netloc:
        return "<invalid-websocket-url>"
    return f"{parts.scheme}://{parts.netloc}{parts.path}"


class Subscription:
    """A live stream registered with the client; survives reconnects.

    Frames arrive on `queue`. `overflow` counts frames dropped because the
    consumer lagged (the gate asserts 0). `error` carries the first stream
    error frame's text (fail-closed marker; the frame is also queued).
    """

    def __init__(self, payload: dict[str, Any], queue_maxsize: int):
        self.payload = dict(payload)
        self.queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=queue_maxsize)
        self.sub_id: str | None = None
        self.overflow = 0
        self.resubscribes = 0
        self.error: str | None = None
        self.active = True


class DerivAsyncClient:
    def __init__(
        self,
        url: str = PUBLIC_WS_URL,
        *,
        request_timeout: float = 15.0,
        open_timeout: float = 10.0,  # a hung connect attempt blocks recovery for this long
        ping_interval: float = 15.0,
        reconnect_base_s: float = 0.5,
        reconnect_cap_s: float = 10.0,
        reconnect_max_attempts: int | None = None,
        queue_maxsize: int = 4096,
        silence_timeout_s: float | None = None,
        url_factory: UrlFactory | None = None,
        on_event: Callable[[str, dict[str, Any]], None] | None = None,
    ):
        self.url = url
        self._url_factory = url_factory
        self.request_timeout = request_timeout
        self.open_timeout = open_timeout
        self.ping_interval = ping_interval
        self.reconnect_base_s = reconnect_base_s
        self.reconnect_cap_s = reconnect_cap_s
        self.reconnect_max_attempts = reconnect_max_attempts
        self.queue_maxsize = queue_maxsize
        # Opt-in liveness watchdog: with live subscriptions a dead socket can
        # only be detected by the protocol ping (~ping_interval+ping_timeout,
        # 45s of silent stream loss). Subscription traffic is ~1 frame/s per
        # sub, so N seconds of TOTAL silence is pathological — force-close and
        # let the reconnect path take over.
        self.silence_timeout_s = silence_timeout_s
        self._last_frame_at = time.monotonic()
        self._watchdog_task: asyncio.Task | None = None
        self._on_event = on_event
        self._ws: Any = None
        self._reader_task: asyncio.Task | None = None
        self._pending: dict[int, asyncio.Future] = {}
        self._subs: dict[str, Subscription] = {}
        self._registry: list[Subscription] = []
        self._req_id = 0
        self._closing = False
        self._dead: str | None = None
        self._connected = asyncio.Event()
        self._background_failed = asyncio.Event()
        self._background_error: str | None = None
        self.stats: dict[str, Any] = {
            "frames": 0,
            "request_frames": 0,
            "stream_frames": 0,
            "unmatched_frames": 0,
            "error_frames": 0,
            "decode_errors": 0,
            "reconnects": 0,
            "url_refreshes": 0,
            "queue_overflow": 0,
            "pre_sub_buffered": 0,
            "resubscribe_latency_s": [],
        }
        self.unmatched_ring: deque[dict[str, Any]] = deque(maxlen=20)
        # Stream frames can beat subscribe()'s registration of the new sub id
        # (the reader keeps dispatching while the subscribe coroutine waits to
        # be scheduled). They buffer here and drain on registration — a first
        # -frame loss window would otherwise exist.
        self._pre_sub: dict[str, list[dict[str, Any]]] = {}

    # ------------------------------------------------------------ lifecycle

    async def connect(self) -> None:
        if self._ws is not None:
            return
        self._ws = await self._open()
        self._connected.set()
        self._reader_task = asyncio.create_task(self._read_loop(), name="deriv-reader")
        self._watch_background_task("reader", self._reader_task)
        if self.silence_timeout_s:
            self._watchdog_task = asyncio.create_task(self._silence_watchdog(), name="deriv-watchdog")
            self._watch_background_task("watchdog", self._watchdog_task)
        self._emit("connected", endpoint=_safe_endpoint(self.url), authenticated="/demo" in self.url)

    async def close(self) -> None:
        self._closing = True
        self._connected.clear()
        if self._watchdog_task is not None:
            self._watchdog_task.cancel()
            try:
                await self._watchdog_task
            except (asyncio.CancelledError, Exception):
                pass
            self._watchdog_task = None
        if self._reader_task is not None:
            self._reader_task.cancel()
            try:
                await self._reader_task
            except (asyncio.CancelledError, Exception):
                pass
            self._reader_task = None
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None
        self._fail_pending(DerivAPIError("client closed"))

    async def __aenter__(self) -> "DerivAsyncClient":
        await self.connect()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    async def _refresh_url(self) -> None:
        if self._url_factory is None:
            return
        maybe_url = self._url_factory()
        if inspect.isawaitable(maybe_url):
            maybe_url = await maybe_url
        if not isinstance(maybe_url, str) or not maybe_url.startswith(("ws://", "wss://")):
            raise DerivAPIError("url_factory returned an invalid WebSocket URL")
        self.url = maybe_url
        self.stats["url_refreshes"] += 1
        self._emit("url_refreshed", endpoint=_safe_endpoint(self.url), authenticated="/demo" in self.url)

    async def _open(self, *, refresh_url: bool = False) -> Any:
        if refresh_url:
            await self._refresh_url()
        return await websockets.connect(
            self.url,
            compression=None,
            open_timeout=self.open_timeout,
            ping_interval=self.ping_interval,
            ping_timeout=self.ping_interval * 2,
            max_size=2**24,
        )

    def _emit(self, event: str, **fields: Any) -> None:
        if self._on_event is not None:
            try:
                self._on_event(event, fields)
            except Exception:
                pass

    def is_connected(self) -> bool:
        return self._connected.is_set() and self._ws is not None and not self._closing and self._dead is None

    def connection_state(self) -> dict[str, Any]:
        return {
            "connected": self.is_connected(),
            "closing": self._closing,
            "dead": self._dead,
            "pending_requests": len(self._pending),
            "subscriptions": len(self._registry),
            "reconnects": self.stats.get("reconnects", 0),
            "last_frame_age_s": round(time.monotonic() - self._last_frame_at, 3),
        }

    async def wait_for_background_failure(self) -> None:
        """Raise when an essential client background task stops unexpectedly."""
        await self._background_failed.wait()
        raise DerivAPIError(self._background_error or "client background task failed")

    def _watch_background_task(self, worker: str, task: asyncio.Task) -> None:
        def completed(done: asyncio.Task) -> None:
            if done.cancelled():
                error_class = "CancelledError"
                error = "cancelled unexpectedly"
            else:
                exc = done.exception()  # retrieve it even during intentional close
                error_class = type(exc).__name__ if exc is not None else "UnexpectedReturn"
                error = str(exc) if exc is not None else "returned unexpectedly"
            if self._dead and error_class == "UnexpectedReturn":
                error = self._dead
            self._latch_background_failure(worker, error_class, error, mark_dead=True)

        task.add_done_callback(completed)

    def _latch_background_failure(
        self, worker: str, error_class: str, error: str, *, mark_dead: bool = False
    ) -> None:
        if self._closing:
            return
        detail = f"client {worker} {error_class}: {error}"
        if mark_dead:
            self._dead = detail
            self._connected.clear()
            self._fail_pending(DerivAPIError(detail))
        if self._background_failed.is_set():
            return
        self._background_error = detail
        self._emit("background_failed", worker=worker, error_class=error_class, error=error[:200])
        self._background_failed.set()

    # ------------------------------------------------------------ requests

    async def request(self, payload: dict[str, Any], timeout: float | None = None) -> dict[str, Any]:
        if self._dead:
            raise DerivAPIError(f"client dead: {self._dead}")
        if self._closing:
            raise DerivAPIError("client closed")
        try:
            await asyncio.wait_for(self._connected.wait(), timeout or self.request_timeout)
        except asyncio.TimeoutError:
            # consumers catch DerivAPIError only — a raw TimeoutError here
            # silently killed drain/refresher tasks (review finding H1)
            raise DerivAPIError("not connected within request timeout") from None
        self._req_id += 1
        rid = self._req_id
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        self._pending[rid] = fut
        msg = dict(payload)
        msg["req_id"] = rid
        try:
            await self._ws.send(json.dumps(msg))
            return await asyncio.wait_for(fut, timeout or self.request_timeout)
        except asyncio.TimeoutError:
            raise DerivAPIError(f"timeout waiting for req_id={rid} payload={payload}") from None
        except websockets.exceptions.ConnectionClosed as exc:
            raise DerivAPIError(f"connection lost during request: {exc}") from exc
        finally:
            self._pending.pop(rid, None)

    async def subscribe(self, payload: dict[str, Any], queue_maxsize: int | None = None) -> Subscription:
        sub_payload = dict(payload)
        sub_payload["subscribe"] = 1
        sub = Subscription(sub_payload, queue_maxsize or self.queue_maxsize)
        resp = await self.request(sub_payload)
        sub_id = (resp.get("subscription") or {}).get("id")
        if not sub_id:
            raise DerivAPIError(f"no subscription id in response keys={sorted(resp)}")
        sub.sub_id = sub_id
        self._subs[sub_id] = sub
        self._registry.append(sub)
        self._offer(sub, resp)  # the initial response carries the first data frame
        for frame in self._pre_sub.pop(sub_id, []):  # frames that beat registration
            self._offer(sub, frame)
        return sub

    async def unsubscribe(self, sub: Subscription) -> None:
        sub.active = False
        if sub in self._registry:
            self._registry.remove(sub)
        if sub.sub_id:
            self._subs.pop(sub.sub_id, None)
            await self.request({"forget": sub.sub_id})

    # ------------------------------------------------------------ dispatch

    def _offer(self, sub: Subscription, frame: dict[str, Any]) -> None:
        try:
            sub.queue.put_nowait(frame)
        except asyncio.QueueFull:
            sub.overflow += 1
            self.stats["queue_overflow"] += 1
            self._emit("queue_overflow", sub_id=sub.sub_id)

    async def _silence_watchdog(self) -> None:
        interval = min(2.0, (self.silence_timeout_s or 2.0) / 2)
        while not self._closing:
            await asyncio.sleep(interval)
            if not self._subs or not self._connected.is_set():
                self._last_frame_at = time.monotonic()
                continue
            silent = time.monotonic() - self._last_frame_at
            if silent > self.silence_timeout_s:
                self.stats["watchdog_trips"] = self.stats.get("watchdog_trips", 0) + 1
                self._emit("silence_watchdog_tripped", silent_s=round(silent, 1))
                self._last_frame_at = time.monotonic()  # no re-trip while reconnecting
                try:
                    await self._ws.close(code=1011, reason="silence watchdog")
                except Exception:
                    pass

    def _dispatch(self, frame: dict[str, Any]) -> None:
        self._last_frame_at = time.monotonic()
        self.stats["frames"] += 1
        rid = frame.get("req_id")
        fut = self._pending.get(rid) if rid is not None else None
        if fut is not None and not fut.done():
            if "error" in frame:
                self.stats["error_frames"] += 1
                error = frame["error"]
                code = error.get("code") if isinstance(error, dict) else None
                fut.set_exception(DerivAPIError(f"Deriv error for req_id={rid}: {error}", code=code))
            else:
                fut.set_result(frame)
            self.stats["request_frames"] += 1
            return
        sub_id = (frame.get("subscription") or {}).get("id")
        if sub_id is None:
            body = frame.get(str(frame.get("msg_type")))
            if isinstance(body, dict):
                sub_id = body.get("id")
        sub = self._subs.get(sub_id) if sub_id else None
        if sub is not None:
            if "error" in frame:
                self.stats["error_frames"] += 1
                sub.error = str(frame["error"])
                self._emit("stream_error", sub_id=sub_id, error=sub.error)
                self._latch_background_failure("subscription", "StreamError", sub.error)
            self._offer(sub, frame)
            self.stats["stream_frames"] += 1
            return
        if sub_id is not None:
            buf = self._pre_sub.setdefault(sub_id, [])
            if len(buf) < 256:
                buf.append(frame)
                self.stats["pre_sub_buffered"] += 1
                return
        if "error" in frame:
            self.stats["error_frames"] += 1
        self.stats["unmatched_frames"] += 1
        self.unmatched_ring.append(frame)
        self._emit("unmatched_frame", msg_type=frame.get("msg_type"))

    def _fail_pending(self, exc: DerivAPIError) -> None:
        for fut in list(self._pending.values()):
            if not fut.done():
                fut.set_exception(exc)

    # ------------------------------------------------------------ reader + reconnect

    async def _read_loop(self) -> None:
        while not self._closing:
            try:
                raw = await self._ws.recv()
            except websockets.exceptions.ConnectionClosed as exc:
                if self._closing:
                    return
                if not await self._reconnect_and_resubscribe(exc):
                    return
                continue
            except asyncio.CancelledError:
                return
            try:
                frame = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                self.stats["decode_errors"] += 1
                self._emit("decode_error", raw_prefix=str(raw)[:80])
                continue
            self._dispatch(frame)

    async def _reconnect_and_resubscribe(self, exc: Exception) -> bool:
        """Reconnect with backoff and re-register every active subscription.

        Runs inside the read loop, so resubscribe responses are correlated
        inline (`_inline_request`) — other frames arriving meanwhile are
        dispatched normally. Returns False when the client is giving up
        (closing, or reconnect_max_attempts exhausted → dead, fail-closed).
        """
        t0 = time.monotonic()
        self._connected.clear()
        self.stats["reconnects"] += 1
        self._fail_pending(DerivAPIError(f"connection lost: {exc}"))
        self._emit("disconnected", reason=str(exc))
        delay = self.reconnect_base_s
        attempts = 0
        while not self._closing:
            attempts += 1
            if self.reconnect_max_attempts is not None and attempts > self.reconnect_max_attempts:
                self._dead = f"reconnect exhausted after {attempts - 1} attempts"
                self._fail_pending(DerivAPIError(self._dead))
                self._emit("reconnect_exhausted", attempts=attempts - 1)
                return False
            try:
                self._ws = await self._open(refresh_url=self._url_factory is not None)
                break
            except Exception as open_exc:
                self._emit("reconnect_failed", attempt=attempts, error=str(open_exc)[:120])
                await asyncio.sleep(delay)
                delay = min(delay * 2, self.reconnect_cap_s)
        if self._closing:
            return False
        t_connected = time.monotonic()
        self._subs.clear()
        self._pre_sub.clear()  # old subscription ids died with the connection
        # Batched resubscribe: send every payload first, then correlate on one
        # recv pump. Deriv subscribe calls run ~0.85s each — done serially six
        # subscriptions blow the 5s gate; batched they cost ~one round trip.
        pending_subs: dict[int, Subscription] = {}
        transport_failed = False
        try:
            for sub in list(self._registry):
                if not sub.active:
                    continue
                self._req_id += 1
                rid = self._req_id
                msg = dict(sub.payload)
                msg["req_id"] = rid
                await self._ws.send(json.dumps(msg))
                pending_subs[rid] = sub
            deadline = time.monotonic() + self.request_timeout
            while pending_subs:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                raw = await asyncio.wait_for(self._ws.recv(), timeout=remaining)
                try:
                    frame = json.loads(raw)
                except (json.JSONDecodeError, TypeError):
                    self.stats["decode_errors"] += 1
                    continue
                rid = frame.get("req_id")
                if rid in pending_subs:
                    sub = pending_subs.pop(rid)
                    if "error" in frame:
                        sub.error = f"resubscribe failed: {frame['error']}"
                        self._emit("resubscribe_failed", payload=sub.payload, error=sub.error[:200])
                        continue
                    new_id = (frame.get("subscription") or {}).get("id")
                    if not new_id:
                        sub.error = "resubscribe returned no subscription id"
                        continue
                    sub.sub_id = new_id
                    sub.resubscribes += 1
                    sub.error = None  # a recovered stream is healthy; stale errors fail gates falsely
                    self._subs[new_id] = sub
                    self._offer(sub, frame)
                else:
                    self._dispatch(frame)
        except (asyncio.TimeoutError, websockets.exceptions.ConnectionClosed) as exc:
            transport_failed = isinstance(exc, websockets.exceptions.ConnectionClosed)
            event = "resubscribe_transport_failure" if transport_failed else "resubscribe_timeout"
            self._emit(event, error=str(exc)[:200])
        if not transport_failed:
            for sub in pending_subs.values():
                sub.error = "resubscribe timeout"
                self._emit("resubscribe_failed", payload=sub.payload, error=sub.error)
        # Breakdown: connect time is bounded by SERVER availability (attempts x
        # open_timeout + backoff); the client-controlled recovery work is the
        # post-connect resubscribe. Both recorded; gates judge what each
        # party controls.
        latency = round(time.monotonic() - t0, 3)
        resubscribe_s = round(time.monotonic() - t_connected, 3)
        self.stats["resubscribe_latency_s"].append(latency)
        self.stats.setdefault("reconnect_breakdown", []).append(
            {"total_s": latency, "connect_attempts": attempts, "connect_s": round(t_connected - t0, 3),
             "resubscribe_s": resubscribe_s})
        if transport_failed:
            return True  # read-loop recv observes the closed transport and retries the whole transaction
        self._connected.set()
        self._emit("resubscribed", latency_s=latency, resubscribe_s=resubscribe_s,
                   connect_attempts=attempts, subs=len(self._subs))
        active_subs = [sub for sub in self._registry if sub.active]
        failed_subs = [sub for sub in active_subs if sub not in self._subs.values()]
        if failed_subs:
            self._latch_background_failure(
                "subscriptions", "ResubscribeFailed",
                f"{len(failed_subs)} of {len(active_subs)} active lanes did not recover",
            )
        return True


# ================================================================ fake-server smoke


async def _fake_server_handler(conn: Any, state: dict[str, Any]) -> None:
    async def stream_ticks(sym: str, sub_id: str) -> None:
        try:
            while sub_id in state["streams"]:
                state["epoch"] += 1
                await conn.send(json.dumps({
                    "msg_type": "tick",
                    "tick": {"symbol": sym, "epoch": state["epoch"], "quote": 1.0, "id": sub_id},
                }))
                await asyncio.sleep(0.02)
        except Exception:
            pass

    try:
        async for raw in conn:
            msg = json.loads(raw)
            rid = msg.get("req_id")
            if "ping" in msg:
                await conn.send(json.dumps({"msg_type": "ping", "ping": "pong", "req_id": rid}))
            elif "echo_marker" in msg:
                marker = msg["echo_marker"]

                async def delayed_echo(marker: int, rid: int) -> None:
                    await asyncio.sleep((marker % 7) * 0.005)  # deterministic shuffle
                    await conn.send(json.dumps({"msg_type": "echo", "echo": marker, "req_id": rid}))

                asyncio.get_running_loop().create_task(delayed_echo(marker, rid))
            elif "ticks" in msg and msg.get("subscribe") == 1:
                symbol = msg["ticks"]
                if symbol in state.get("reject_symbols_once", set()):
                    state["reject_symbols_once"].remove(symbol)
                    await conn.send(json.dumps({
                        "msg_type": "tick",
                        "error": {"code": "InjectedSubscribeFailure", "message": "deterministic smoke"},
                        "req_id": rid,
                    }))
                    continue
                if symbol == state.get("close_subscribe_once"):
                    state["close_subscribe_once"] = None
                    await conn.close(code=1011, reason="injected mid-resubscribe drop")
                    return
                state["sub_n"] += 1
                sub_id = f"sub-{state['sub_n']}"
                state["streams"][sub_id] = symbol
                state["epoch"] += 1
                await conn.send(json.dumps({
                    "msg_type": "tick",
                    "tick": {"symbol": symbol, "epoch": state["epoch"], "quote": 1.0, "id": sub_id},
                    "subscription": {"id": sub_id},
                    "req_id": rid,
                }))
                asyncio.get_running_loop().create_task(stream_ticks(symbol, sub_id))
            elif "forget" in msg:
                state["streams"].pop(msg["forget"], None)
                await conn.send(json.dumps({"msg_type": "forget", "forget": 1, "req_id": rid}))
            elif "bogus_frame" in msg:
                await conn.send(json.dumps({"msg_type": "mystery", "noise": True}))
                await conn.send(json.dumps({"msg_type": "bogus_ack", "req_id": rid}))
            elif "mute_now" in msg:
                # silent death: streams stop but the connection stays open
                state["streams"].clear()
                await conn.send(json.dumps({"msg_type": "mute_ack", "req_id": rid}))
            elif "drop_now" in msg:
                state["streams"].clear()
                await conn.close(code=1011, reason="injected drop")
                return
    except websockets.exceptions.ConnectionClosed:
        pass


async def run_smoke() -> int:
    """Deterministic Phase-1 smoke against an in-process fake server."""
    failures: list[str] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        print(f"{'PASS' if ok else 'FAIL'}: {name}" + (f" — {detail}" if detail else ""))
        if not ok:
            failures.append(name)

    state: dict[str, Any] = {
        "sub_n": 0,
        "epoch": 0,
        "streams": {},
        "reject_symbols_once": set(),
        "close_subscribe_once": None,
    }

    async def handler(conn: Any) -> None:
        await _fake_server_handler(conn, state)

    async def observed_failure(client: DerivAsyncClient, timeout: float = 2.0) -> str | None:
        try:
            await asyncio.wait_for(client.wait_for_background_failure(), timeout)
        except DerivAPIError as exc:
            return str(exc)
        except asyncio.TimeoutError:
            return None
        return None

    async with websockets.serve(handler, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        client = DerivAsyncClient(f"ws://127.0.0.1:{port}", request_timeout=5.0, reconnect_base_s=0.05,
                                  silence_timeout_s=0.5)
        await client.connect()

        # 1. correlation under concurrency: 20 requests with server-side delay shuffle
        resps = await asyncio.gather(*(client.request({"echo_marker": k}) for k in range(20)))
        matched = sum(1 for k, r in enumerate(resps) if r.get("echo") == k)
        check("correlation 20/20 under shuffled delays", matched == 20, f"{matched}/20")
        check("zero unmatched after correlation burst", client.stats["unmatched_frames"] == 0)

        # Structured response codes survive the request exception boundary so
        # callers never classify a free-form message as a MarketIsClosed code.
        state["reject_symbols_once"] = {"CODEUSD"}
        registry_before = len(client._registry)
        propagated_code = None
        try:
            await client.subscribe({"ticks": "CODEUSD"})
        except DerivAPIError as exc:
            propagated_code = exc.code
        check("initial subscription error preserves exact response code",
              propagated_code == "InjectedSubscribeFailure"
              and len(client._registry) == registry_before)

        # 2. subscription stream: consecutive epochs prove zero first-frame loss
        # (the fake server's 20ms stream beats subscribe() registration — the
        # pre-sub buffer must hand every early frame over in order)
        sub = await client.subscribe({"ticks": "FAKEUSD"})
        epochs = [(await asyncio.wait_for(sub.queue.get(), 2.0))["tick"]["epoch"] for _ in range(10)]
        check("10 stream ticks consecutive (no first-frame loss)", epochs == list(range(epochs[0], epochs[0] + 10)), str(epochs))

        # 3. injected drop: pending request fails closed; auto resubscribe < 5s; stream resumes
        healthy_waiter = asyncio.create_task(client.wait_for_background_failure())
        try:
            await client.request({"drop_now": 1})
            check("pending request fails closed on drop", False, "request unexpectedly succeeded")
        except DerivAPIError:
            check("pending request fails closed on drop", True)
        t0 = time.monotonic()
        frame = await asyncio.wait_for(sub.queue.get(), 10.0)
        resume_s = time.monotonic() - t0
        check("stream resumes after reconnect", frame.get("msg_type") == "tick")
        check("resubscribe < 5s", resume_s < 5.0, f"{resume_s:.2f}s")
        check("resubscribe counted once", sub.resubscribes == 1, str(sub.resubscribes))
        check("successful reconnect does not signal background failure", not healthy_waiter.done())
        healthy_waiter.cancel()
        await asyncio.gather(healthy_waiter, return_exceptions=True)
        post = [(await asyncio.wait_for(sub.queue.get(), 2.0))["tick"]["epoch"] for _ in range(5)]
        check("post-reconnect epochs continue increasing", min(post) > max(epochs), f"{max(epochs)} -> {min(post)}")

        # 3a. authenticated sockets get single-use URLs. A reconnect must
        # refresh the URL before opening the next transport, and emitted events
        # must not leak the raw URL/query token.
        refresh_calls = 0
        auth_events: list[dict[str, Any]] = []

        async def fresh_url() -> str:
            nonlocal refresh_calls
            refresh_calls += 1
            return f"ws://127.0.0.1:{port}/?otp=fresh-secret-{refresh_calls}"

        auth_client = DerivAsyncClient(
            f"ws://127.0.0.1:{port}/?otp=initial-secret",
            request_timeout=5.0,
            reconnect_base_s=0.05,
            url_factory=fresh_url,
            on_event=lambda e, f: auth_events.append({"event": e, **f}),
        )
        await auth_client.connect()
        auth_sub = await auth_client.subscribe({"ticks": "AUTHUSD"})
        try:
            await auth_client.request({"drop_now": 1})
        except DerivAPIError:
            pass
        auth_frame = await asyncio.wait_for(auth_sub.queue.get(), 10.0)
        redacted_events = [
            ev for ev in auth_events
            if ev.get("event") in {"connected", "url_refreshed"}
        ]
        redacted_text = json.dumps(redacted_events, sort_keys=True)
        check("url_factory called once for reconnect", refresh_calls == 1, str(refresh_calls))
        check("auth stream resumes with refreshed URL", auth_frame.get("msg_type") == "tick")
        check("connection events redact raw URL tokens",
              all("url" not in ev for ev in redacted_events) and "secret" not in redacted_text,
              redacted_text)
        await auth_client.close()

        # 3b. concurrent multi-subscription resubscribe: routing purity after drop
        subs3 = {sym: await client.subscribe({"ticks": sym}) for sym in ("FAKEA", "FAKEB", "FAKEC")}
        try:
            await client.request({"drop_now": 1})
        except DerivAPIError:
            pass
        pure = True
        for sym, s3 in subs3.items():
            frames = [await asyncio.wait_for(s3.queue.get(), 10.0) for _ in range(3)]
            pure = pure and all(f["tick"]["symbol"] == sym for f in frames)
        check("batched resubscribe: every queue receives only its own symbol", pure)
        check("batched resubscribe counted on all three", all(s.resubscribes == 1 for s in subs3.values()))
        for s3 in subs3.values():
            await client.unsubscribe(s3)

        # 3c. silence watchdog: server goes mute WITHOUT closing -> forced
        # reconnect + resubscribe, stream resumes (the live 52s silent-death
        # signature, detected in ~silence_timeout instead of the ping timeout)
        trips_before = client.stats.get("watchdog_trips", 0)
        await client.request({"mute_now": 1})
        await asyncio.sleep(0.1)  # let in-flight pre-mute ticks settle
        while not sub.queue.empty():
            sub.queue.get_nowait()  # drain buffered pre-mute ticks: the next frame must be post-revival
        t0 = time.monotonic()
        frame = await asyncio.wait_for(sub.queue.get(), 10.0)
        check("watchdog revives muted stream", frame.get("msg_type") == "tick",
              f"{time.monotonic() - t0:.2f}s")
        check("watchdog trip counted", client.stats.get("watchdog_trips", 0) > trips_before)

        # 4. unmatched frame surfaced, never silently dropped
        await client.request({"bogus_frame": 1})
        await asyncio.sleep(0.1)
        check("unmatched frame counted", client.stats["unmatched_frames"] == 1, str(client.stats["unmatched_frames"]))
        check("unmatched frame retained in ring", any(f.get("msg_type") == "mystery" for f in client.unmatched_ring))

        # 5. unsubscribe stops the stream
        await client.unsubscribe(sub)
        while not sub.queue.empty():
            sub.queue.get_nowait()
        await asyncio.sleep(0.2)
        check("no frames after unsubscribe", sub.queue.empty())

        # 6. fail-closed after close
        await client.close()
        try:
            await client.request({"ping": 1})
            check("request after close raises", False)
        except DerivAPIError:
            check("request after close raises", True)
        check("zero queue overflow", client.stats["queue_overflow"] == 0)
        check("zero decode errors", client.stats["decode_errors"] == 0)

        async def rejected_round(symbols: tuple[str, ...], rejected: set[str]) -> tuple[str | None, str | None]:
            rejected_client = DerivAsyncClient(
                f"ws://127.0.0.1:{port}", request_timeout=1.0, reconnect_base_s=0.01)
            await rejected_client.connect()
            rejected_subs = [await rejected_client.subscribe({"ticks": sym}) for sym in symbols]
            for rejected_sub in rejected_subs:
                await asyncio.wait_for(rejected_sub.queue.get(), 1.0)
            state["reject_symbols_once"] = set(rejected)
            try:
                await rejected_client.request({"drop_now": 1})
            except DerivAPIError:
                pass
            first = await observed_failure(rejected_client)
            latched = await observed_failure(rejected_client, timeout=0.1)
            await rejected_client.close()
            return first, latched

        # 7. A stable partial resubscribe rejection is terminal and latched.
        partial_failure, partial_latched = await rejected_round(
            ("PARTIALA", "PARTIALB"), {"PARTIALB"})
        check("partial stable resubscribe rejection signals failure",
              partial_failure is not None and "1 of 2" in partial_failure, str(partial_failure))
        check("first client failure remains latched", partial_latched == partial_failure)

        # 8. Total stable rejection is also terminal (no healthy lane can mask it).
        total_failure, _ = await rejected_round(("TOTALA", "TOTALB"), {"TOTALA", "TOTALB"})
        check("all-lane stable resubscribe rejection signals failure",
              total_failure is not None and "2 of 2" in total_failure, str(total_failure))

        # 9. A transport loss during resubscribe retries the entire transaction;
        # it is not a stable lane failure and every queue routes again.
        mid = DerivAsyncClient(f"ws://127.0.0.1:{port}", request_timeout=1.0, reconnect_base_s=0.01)
        await mid.connect()
        mid_subs = {sym: await mid.subscribe({"ticks": sym}) for sym in ("MIDA", "MIDB")}
        for mid_sub in mid_subs.values():
            await asyncio.wait_for(mid_sub.queue.get(), 1.0)
        mid_waiter = asyncio.create_task(mid.wait_for_background_failure())
        state["close_subscribe_once"] = "MIDB"
        try:
            await mid.request({"drop_now": 1})
        except DerivAPIError:
            pass
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and not (mid.is_connected()
                                                    and all(s.resubscribes >= 1 for s in mid_subs.values())):
            await asyncio.sleep(0.01)
        for mid_sub in mid_subs.values():
            while not mid_sub.queue.empty():
                mid_sub.queue.get_nowait()
        routed = {
            sym: (await asyncio.wait_for(mid_sub.queue.get(), 1.0)).get("tick", {}).get("symbol")
            for sym, mid_sub in mid_subs.items()
        }
        check("mid-resubscribe disconnect recovers routing for every lane",
              routed == {sym: sym for sym in mid_subs}, str(routed))
        check("mid-resubscribe disconnect does not signal stable failure", not mid_waiter.done())
        mid_waiter.cancel()
        await asyncio.gather(mid_waiter, return_exceptions=True)
        await mid.close()

        # 10. Reconnect exhaustion terminates the reader and wakes the public latch.
        exhausted = DerivAsyncClient(f"ws://127.0.0.1:{port}", reconnect_max_attempts=0)
        await exhausted.connect()
        await exhausted._ws.close(code=1011, reason="injected exhaustion")
        exhausted_failure = await observed_failure(exhausted)
        check("reader reconnect exhaustion signals failure",
              exhausted_failure is not None and "reconnect exhausted" in exhausted_failure,
              str(exhausted_failure))
        await exhausted.close()

        # 11. Intentional close must not look like a background failure.
        intentional = DerivAsyncClient(f"ws://127.0.0.1:{port}")
        await intentional.connect()
        intentional_waiter = asyncio.create_task(intentional.wait_for_background_failure())
        await intentional.close()
        await asyncio.sleep(0)
        check("intentional close does not signal background failure", not intentional_waiter.done())
        intentional_waiter.cancel()
        await asyncio.gather(intentional_waiter, return_exceptions=True)

    print(f"\n{'SMOKE ALL PASS' if not failures else f'SMOKE {len(failures)} FAILURES: {failures}'}")
    return 1 if failures else 0


# ================================================================ live soak (Phase-1 gate)


async def run_soak(minutes: float, disconnects: int, url: str) -> int:
    """Live public soak: N symbols subscribed, K injected disconnects.

    Gate: every resubscribe < 5s; zero mis-correlated frames (request
    correlation is future-keyed — mis-correlation manifests as error/timeout;
    additionally every stream frame must route: unmatched == 0); zero silent
    frame loss (queue overflow == 0, decode errors == 0).
    """
    from deriv_backfill import DEFAULT_ENABLED_PAIRS, PAIR_TO_SYMBOL

    symbols = [PAIR_TO_SYMBOL[p] for p in DEFAULT_ENABLED_PAIRS]
    events: list[dict[str, Any]] = []
    started_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
    client = DerivAsyncClient(url, on_event=lambda e, f: events.append({"event": e, **f}))
    await client.connect()
    subs = {sym: await client.subscribe({"ticks": sym}) for sym in symbols}
    tick_counts: dict[str, int] = {s: 0 for s in symbols}
    segment_counts: list[dict[str, int]] = [dict.fromkeys(symbols, 0)]

    async def drain(sub: Subscription, sym: str) -> None:
        while True:
            frame = await sub.queue.get()
            if frame.get("msg_type") == "tick":
                tick_counts[sym] += 1
                segment_counts[-1][sym] += 1

    drains = [asyncio.get_running_loop().create_task(drain(sub, sym)) for sym, sub in subs.items()]
    interval = minutes * 60.0 / (disconnects + 1)
    for k in range(disconnects + 1):
        await asyncio.sleep(interval)
        if k < disconnects:
            segment_counts.append(dict.fromkeys(symbols, 0))
            await client._ws.close(code=1011, reason="injected soak disconnect")
    for task in drains:
        task.cancel()
    stats = dict(client.stats)
    per_sub = {sym: {"resubscribes": sub.resubscribes, "overflow": sub.overflow, "error": sub.error}
               for sym, sub in subs.items()}
    await client.close()

    lat = stats["resubscribe_latency_s"]
    breakdown = stats.get("reconnect_breakdown", [])
    client_side = [b["resubscribe_s"] for b in breakdown]
    gate = {
        "resubscribes_observed": len(lat),
        "resubscribe_total_s": lat,
        "reconnect_breakdown": breakdown,
        # The gate judges the CLIENT-controlled recovery (post-connect
        # resubscribe); total gap includes server connect availability, which
        # is reported but cannot be bounded client-side — the runtime's
        # staleness gates fail closed during any such gap.
        "all_client_resubscribes_under_5s": bool(client_side and max(client_side) < 5.0
                                                 and len(client_side) >= disconnects),
        "total_gap_max_s": max(lat) if lat else None,
        "unmatched_frames": stats["unmatched_frames"],
        "queue_overflow": stats["queue_overflow"],
        "decode_errors": stats["decode_errors"],
        "stream_errors": [s for s, v in per_sub.items() if v["error"]],
        "every_symbol_ticked_every_segment": all(all(seg[s] > 0 for s in symbols) for seg in segment_counts),
    }
    gate["pass"] = bool(
        gate["all_client_resubscribes_under_5s"]
        and gate["unmatched_frames"] == 0
        and gate["queue_overflow"] == 0
        and gate["decode_errors"] == 0
        and not gate["stream_errors"]
        and gate["every_symbol_ticked_every_segment"]
    )
    result = {
        "gate": "issue#4 Phase 1 live soak",
        "endpoint": _safe_endpoint(url),
        "started_utc": started_utc,
        "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "host": socket.gethostname(),
        "minutes": minutes,
        "injected_disconnects": disconnects,
        "tick_counts": tick_counts,
        "segment_counts": segment_counts,
        "stats": {k: v for k, v in stats.items() if k != "resubscribe_latency_s"},
        "per_subscription": per_sub,
        "gate_result": gate,
        "events_tail": events[-20:],
    }
    tmp = SOAK_RESULT_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    tmp.replace(SOAK_RESULT_PATH)
    print(json.dumps(gate, indent=2, sort_keys=True))
    print(f"{'SOAK PASS' if gate['pass'] else 'SOAK FAIL'} -> {SOAK_RESULT_PATH}")
    return 0 if gate["pass"] else 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--smoke", action="store_true", help="deterministic fake-server smoke (no network)")
    p.add_argument("--soak-minutes", type=float, default=0.0, help="live public soak duration")
    p.add_argument("--inject-disconnects", type=int, default=3)
    p.add_argument("--url", default=PUBLIC_WS_URL)
    args = p.parse_args()
    if args.smoke:
        return asyncio.run(run_smoke())
    if args.soak_minutes > 0:
        return asyncio.run(run_soak(args.soak_minutes, args.inject_disconnects, args.url))
    p.error("nothing to do: pass --smoke or --soak-minutes")
    return 2


if __name__ == "__main__":
    sys.exit(main())
