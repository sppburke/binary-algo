"""Deriv market-data workers: tick store + shifted bars (issue #4 Phase 2).

Per-pair live tick subscriptions (via `deriv_async_client`), warmup from tick
history, in-memory ring buffers, an atomic parquet tick store under
`deriv_data/ticks_1s/` (backfill discipline: `_pages` shards, `_progress.json`
cursor, tmp+replace, `ensure_deriv_data_path` guard), the 12-offset shifted
aggregator, and the daemon candle-store refresher.

Pinned constants (issue #4 r3): offsets {0,5,...,55}s; shifted-window validity
= exactly 60 one-second ticks in (end-60s, end], deduped keep-last, NEVER
forward-filled; aggregation open=first / high=max / low=min / close=last
(`pipeline.resample_1m` semantics on right-closed windows). Store ownership:
the production one-shot timer owns `deriv_data/candles_1m/`; this refresher
writes `deriv_data/candles_1m_daemon/` and a cross-store consistency report.

Data plane only: no scoring, no trading. Phase-2 gate (>=3 full NY sessions):
per-pair 1s coverage >= 99.5% of session seconds, 12/12 window rate >= 99%,
no ring overflow, store-writer lag p95 <= 30s / max <= 120s, zero ticks lost
between ring and store. Each run appends its rollup to
`deriv_market_stream_gate_result.json` (archive to results/json/).

Run from repo root:
    ~/binary-algo-venv/bin/python scripts/deriv_market_stream.py --smoke
    ~/binary-algo-venv/bin/python scripts/deriv_market_stream.py --duration-minutes 60
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import shutil
import socket
import sys
import tempfile
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

import pandas as pd

from deriv_async_client import DerivAsyncClient
from deriv_backfill import (
    DEFAULT_ENABLED_PAIRS,
    PAIR_TO_SYMBOL,
    PageResult,
    candles_frame,
    ensure_deriv_data_path,
    merge_existing,
    parse_pairs,
    write_page_shard,
    write_progress,
)
from deriv_client import PUBLIC_WS_URL, DerivAPIError

OFFSETS = tuple(range(0, 60, 5))
WINDOW_S = 60
GATE_RESULT_PATH = Path("deriv_market_stream_gate_result.json")
DEFAULT_TICK_STORE = Path("deriv_data/ticks_1s")
DEFAULT_CANDLE_STORE = Path("deriv_data/candles_1m_daemon")
PRODUCTION_CANDLE_STORE = Path("deriv_data/candles_1m")
MARKET_CLOSED_RETRY_SECONDS = 60.0
STREAM_QUEUE_POLL_SECONDS = 5.0


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    xs = sorted(values)
    return round(xs[min(len(xs) - 1, int(q * (len(xs) - 1)))], 3)


def is_market_closed_error(exc: DerivAPIError) -> bool:
    return exc.code == "MarketIsClosed"


# ---------------------------------------------------------------- shifted aggregator (pure)


def shifted_bar(ticks: list[tuple[int, float, float, float]], end_epoch: int) -> tuple[dict[str, Any] | None, int]:
    """Build one shifted 60s bar ending at `end_epoch` from (epoch, quote, bid, ask).

    Pinned validity: exactly 60 one-second ticks in (end-60, end], deduped
    keep-last; returns (None, n_ticks) otherwise — incomplete windows are never
    forward-filled. Aggregation mirrors pipeline.resample_1m on the
    right-closed window: open=first, high=max, low=min, close=last.
    """
    window: dict[int, tuple[float, float, float]] = {}
    for epoch, quote, bid, ask in ticks:
        if end_epoch - WINDOW_S < epoch <= end_epoch:
            window[epoch] = (quote, bid, ask)  # keep-last dedup
    n = len(window)
    if n != WINDOW_S:
        return None, n
    epochs = sorted(window)
    quotes = [window[e][0] for e in epochs]

    def clean(x: float) -> float | None:
        # warmup history ticks carry no bid/ask (NaN); emit None so the JSONL
        # stays strict-JSON and consumers see an honest "unavailable"
        return None if isinstance(x, float) and math.isnan(x) else x

    return {
        "offset_id": end_epoch % 60,
        "end_epoch": end_epoch,
        "bar_start_utc": datetime.fromtimestamp(end_epoch - WINDOW_S, timezone.utc).isoformat(timespec="seconds"),
        "bar_close_utc": datetime.fromtimestamp(end_epoch, timezone.utc).isoformat(timespec="seconds"),
        "open": quotes[0],
        "high": max(quotes),
        "low": min(quotes),
        "close": quotes[-1],
        "bid_first": clean(window[epochs[0]][1]),
        "bid_last": clean(window[epochs[-1]][1]),
        "ask_first": clean(window[epochs[0]][2]),
        "ask_last": clean(window[epochs[-1]][2]),
        "n_ticks": n,
        "price_basis": "quote_mid",
    }, n


# ---------------------------------------------------------------- per-pair state


class PairState:
    def __init__(self, pair: str, ring_seconds: int, page_n: int):
        self.pair = pair
        self.ring: deque[tuple[int, float, float, float]] = deque(maxlen=ring_seconds)
        self.last_epoch = 0
        self.received = 0  # live post-dedup ticks (warmup excluded)
        self.dupes = 0
        self.gap_max_s = 0
        self.gaps_over_3s = 0
        self.write_buf: list[tuple[int, float, float, float]] = []
        self.buf_depth_max = 0
        self.page_n = page_n
        self.recovered = 0  # gap ticks backfilled from history (store + window heal)
        # True ticks fetched from history for live-skipped seconds; merged into
        # windows at aggregation so a healed window scores COMPLETE (real data,
        # never ffill). Live ring ticks win on duplicate epochs.
        self.recovered_map: dict[int, tuple[float, float, float]] = {}
        self.windows_healed = 0
        self.persisted = 0
        self.persisted_last_epoch = 0
        self.writer_lag_samples: list[float] = []
        self.warm_at_epoch: int | None = None
        self.windows_expected = 0
        self.windows_complete = 0
        self.windows_retry_completed = 0
        self.window_incomplete_counts: dict[str, int] = {}

    def add_tick(self, epoch: int, quote: float, bid: float, ask: float, *, persist: bool) -> None:
        if epoch <= self.last_epoch:
            self.dupes += 1
            return
        if self.last_epoch and persist:
            gap = epoch - self.last_epoch
            self.gap_max_s = max(self.gap_max_s, gap)
            if gap > 3:
                self.gaps_over_3s += 1
        self.last_epoch = epoch
        self.ring.append((epoch, quote, bid, ask))
        if persist:
            self.received += 1
            self.write_buf.append((epoch, quote, bid, ask))
            self.buf_depth_max = max(self.buf_depth_max, len(self.write_buf))

    def window_ticks(self, end_epoch: int) -> list[tuple[int, float, float, float]]:
        # recovered ticks FIRST: shifted_bar dedups keep-last, so a live ring
        # tick wins over a recovered one at the same epoch
        out: list[tuple[int, float, float, float]] = [
            (e, *self.recovered_map[e])
            for e in range(end_epoch - WINDOW_S + 1, end_epoch + 1)
            if e in self.recovered_map
        ]
        tail: list[tuple[int, float, float, float]] = []
        for item in reversed(self.ring):
            if item[0] <= end_epoch - WINDOW_S:
                break
            if item[0] <= end_epoch:
                tail.append(item)
        tail.reverse()
        return out + tail


# ---------------------------------------------------------------- store writer


def flush_pair(store_dir: Path, st: PairState) -> int:
    """Persist the pair's buffered ticks as one atomic shard + progress update."""
    if not st.write_buf:
        return 0
    buf, st.write_buf = st.write_buf, []
    try:
        frame = pd.DataFrame(buf, columns=["epoch", "quote", "bid", "ask"])
        write_page_shard(store_dir, st.pair, st.page_n, frame)
    except Exception:
        # The drain may append while the writer is in a thread. Restore the
        # detached batch ahead of those newer ticks so final flush can retry it.
        st.write_buf[:0] = buf
        raise
    st.page_n += 1
    st.persisted += len(buf)
    st.persisted_last_epoch = max(b[0] for b in buf)
    write_progress(store_dir, st.pair, {
        "updated_utc": now_utc_iso(),
        "live_ticks_received": st.received,
        "live_ticks_persisted": st.persisted,
        "last_persisted_epoch": st.persisted_last_epoch,
        "pages": st.page_n,
    })
    return len(buf)


def compact_pair(store_dir: Path, pair: str) -> int:
    """Merge `_pages/<pair>/*.parquet` shards into <pair>.parquet (audit substrate)."""
    page_dir = store_dir / "_pages" / pair
    shards = sorted(page_dir.glob("*.parquet")) if page_dir.exists() else []
    if not shards:
        return 0
    frame = pd.concat([pd.read_parquet(p) for p in shards], axis=0)
    frame = frame.drop_duplicates(subset="epoch", keep="last").sort_values("epoch").reset_index(drop=True)
    out = store_dir / f"{pair}.parquet"
    tmp = out.with_suffix(".parquet.tmp")
    frame.to_parquet(tmp, index=False)
    tmp.replace(out)
    return len(frame)


# ---------------------------------------------------------------- candle refresher helpers


def upsert_candles(path: Path, pair: str, payload: dict[str, Any], *, endpoint_label: str) -> pd.DataFrame:
    rows = payload.get("candles") or []
    page = PageResult(payload=payload, row_count=len(rows),
                      oldest_epoch=int(rows[0]["epoch"]) if rows else None,
                      newest_epoch=int(rows[-1]["epoch"]) if rows else None)
    frame = candles_frame(pair=pair, symbol=PAIR_TO_SYMBOL[pair], page=page, granularity=60,
                          endpoint=endpoint_label, drop_open_candle=True)
    merged = merge_existing(path, frame, replace=False)
    tmp = path.with_suffix(".parquet.tmp")
    merged.to_parquet(tmp, index=True)
    tmp.replace(path)
    return merged


def xstore_report(daemon_dir: Path, production_dir: Path, pairs: list[str], n_compare: int = 30) -> dict[str, Any]:
    """Cross-store consistency: daemon store vs production store completed bars.

    Divergence here is a store bug, tracked separately from decision parity
    (issue #4 r3 store-ownership rule).
    """
    report: dict[str, Any] = {}
    for pair in pairs:
        d_path, p_path = daemon_dir / f"{pair}.parquet", production_dir / f"{pair}.parquet"
        if not d_path.exists() or not p_path.exists():
            report[pair] = {"status": "missing", "daemon": d_path.exists(), "production": p_path.exists()}
            continue
        d = pd.read_parquet(d_path)[["epoch", "close"]].set_index("epoch")
        p = pd.read_parquet(p_path)[["epoch", "close"]].set_index("epoch")
        common = d.index.intersection(p.index).sort_values()[-n_compare:]
        mismatch = int((d.loc[common, "close"] != p.loc[common, "close"]).sum()) if len(common) else None
        report[pair] = {
            "daemon_rows": len(d), "production_rows": len(p),
            "daemon_last_epoch": int(d.index.max()), "production_last_epoch": int(p.index.max()),
            "n_compared": len(common), "n_close_mismatch": mismatch,
        }
    return report


# ---------------------------------------------------------------- stream runtime


class MarketStream:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.pairs = parse_pairs(args.pairs) if args.pairs != "all-enabled" else list(DEFAULT_ENABLED_PAIRS)
        # The candle store must cover ALL pairs incl. EURUSD: the xpair books'
        # live feature join reads every pair's closes (live_features.PAIRS),
        # exactly like the production store (backfill --pairs all).
        self.candle_pairs = list(PAIR_TO_SYMBOL)
        self.tick_store = ensure_deriv_data_path(Path(args.tick_store_dir), purpose="tick store")
        self.candle_store = ensure_deriv_data_path(Path(args.candle_store_dir), purpose="daemon candle store")
        self.tick_store.mkdir(parents=True, exist_ok=True)
        self.candle_store.mkdir(parents=True, exist_ok=True)
        self.log_dir = Path(args.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_fh: TextIO = (self.log_dir / f"stream_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.jsonl").open("a")
        self.states: dict[str, PairState] = {}
        for pair in self.pairs:
            page_dir = self.tick_store / "_pages" / pair
            page_n = len(list(page_dir.glob("*.parquet"))) if page_dir.exists() else 0
            self.states[pair] = PairState(pair, args.ring_seconds, page_n)
        # 6 tick subs at ~1/s: >8s of total silence is a dead socket — the
        # watchdog cuts the detection latency from ~45s (protocol ping) to ~8s
        self.client = DerivAsyncClient(args.url, silence_timeout_s=8.0,
                                       on_event=lambda e, f: self.log("client_" + e, **f))
        self.stop = asyncio.Event()
        self.started_utc = now_utc_iso()
        self.t0 = time.time()
        self.refresher_updates = 0
        self.refresher_errors = 0

    def log(self, event: str, **fields: Any) -> None:
        self.log_fh.write(json.dumps({"ts": now_utc_iso(), "event": event, **fields}, default=str) + "\n")
        self.log_fh.flush()

    # ---------------- warmup

    async def warmup_pair(self, pair: str) -> None:
        st = self.states[pair]
        sym = PAIR_TO_SYMBOL[pair]
        target_oldest = int(time.time()) - self.args.warmup_seconds
        pages: list[list[tuple[int, float]]] = []
        end: str = "latest"
        for _ in range(64):  # bound: 64 pages x 1000 ticks
            resp = await self.client.request({"ticks_history": sym, "style": "ticks", "count": 1000,
                                              "end": end, "adjust_start_time": 1})
            h = resp.get("history") or {}
            times = [int(t) for t in h.get("times") or []]
            prices = [float(x) for x in h.get("prices") or []]
            if not times:
                break
            pages.append(list(zip(times, prices)))
            if times[0] <= target_oldest:
                break
            end = str(times[0] - 1)
        ticks = sorted({e: q for page in pages for e, q in page}.items())
        for epoch, quote in ticks:
            if epoch >= target_oldest:
                st.add_tick(epoch, quote, math.nan, math.nan, persist=False)
        st.warm_at_epoch = st.last_epoch or None
        self.log("warmup_done", pair=pair, ticks=len(ticks), oldest=ticks[0][0] if ticks else None)

    # ---------------- live drains

    async def drain_pair(self, pair: str) -> None:
        st = self.states[pair]
        while not self.stop.is_set():
            try:
                sub = await self.client.subscribe({"ticks": PAIR_TO_SYMBOL[pair]})
                break
            except DerivAPIError as exc:
                if not is_market_closed_error(exc):
                    raise
                self.log(
                    "subscribe_market_closed",
                    pair=pair,
                    error_class=type(exc).__name__,
                    error=str(exc)[:200],
                    retry_seconds=MARKET_CLOSED_RETRY_SECONDS,
                )
                try:
                    await asyncio.wait_for(self.stop.wait(), timeout=MARKET_CLOSED_RETRY_SECONDS)
                    return
                except asyncio.TimeoutError:
                    pass
        else:
            return
        recovering = False
        while not self.stop.is_set():
            try:
                frame = await asyncio.wait_for(sub.queue.get(), timeout=STREAM_QUEUE_POLL_SECONDS)
            except asyncio.TimeoutError:
                continue
            if "error" in frame:
                raise DerivAPIError(f"{pair} tick subscription error: {frame['error']}")
            tick = frame.get("tick")
            if not isinstance(tick, dict):
                continue
            prev_last = st.last_epoch
            try:
                epoch = int(tick["epoch"])
                st.add_tick(epoch, float(tick["quote"]),
                            float(tick.get("bid", math.nan)), float(tick.get("ask", math.nan)), persist=True)
            except (KeyError, TypeError, ValueError):
                self.log("tick_parse_error", pair=pair, keys=sorted(tick))
                continue
            if prev_last and epoch - prev_last > 1 and not recovering:
                # Deriv retains (and consolidates) ticks the live stream
                # skipped — even 2s micro-gaps, which void 12 overlapping
                # shifted windows each. Backfill every gap into the STORE
                # (audit substrate). The ring stays live-only — windows over
                # the gap were already honestly incomplete.
                recovering = True
                try:
                    await self.recover_gap(pair, prev_last, epoch)
                finally:
                    recovering = False

    async def recover_gap(self, pair: str, start_epoch: int, end_epoch: int) -> None:
        st = self.states[pair]
        sym = PAIR_TO_SYMBOL[pair]
        lo, end = start_epoch + 1, end_epoch - 1
        fetched: dict[int, float] = {}
        for _ in range(8):  # bound: 8 pages x 1000 ticks
            if end < lo:
                break
            try:
                resp = await self.client.request({"ticks_history": sym, "style": "ticks",
                                                  "count": min(1000, end - lo + 1), "end": str(end),
                                                  "adjust_start_time": 1})
            except DerivAPIError as exc:
                self.log("gap_recovery_failed", pair=pair, error=str(exc)[:200])
                break
            h = resp.get("history") or {}
            times = [int(t) for t in h.get("times") or []]
            prices = [float(x) for x in h.get("prices") or []]
            if not times:
                break
            for e, q in zip(times, prices):
                if lo <= e <= end_epoch - 1:
                    fetched[e] = q
            if times[0] <= lo:
                break
            end = times[0] - 1
        if fetched:
            st.write_buf.extend((e, q, math.nan, math.nan) for e, q in sorted(fetched.items()))
            for e, q in fetched.items():
                st.recovered_map[e] = (q, math.nan, math.nan)
            st.recovered += len(fetched)
            cutoff = st.last_epoch - self.args.ring_seconds
            for stale in [e for e in st.recovered_map if e < cutoff]:
                del st.recovered_map[stale]
            self.log("gap_recovered", pair=pair, n=len(fetched), gap_s=end_epoch - start_epoch)

    # ---------------- aggregator

    async def aggregator(self) -> None:
        while not self.stop.is_set():
            now = time.time()
            bound = (int(now) // 5 + 1) * 5
            try:
                await asyncio.wait_for(self.stop.wait(), timeout=max(0.05, bound + 1.0 - now))
                return
            except asyncio.TimeoutError:
                pass
            pending: list[PairState] = []
            for st in self.states.values():
                if st.warm_at_epoch is None:
                    continue
                st.windows_expected += 1
                bar, n = shifted_bar(st.window_ticks(bound), bound)
                if bar is not None:
                    st.windows_complete += 1
                    if any(bound - WINDOW_S < e <= bound for e in st.recovered_map):
                        st.windows_healed += 1
                    if bound % 60 == 0:
                        self.log("window_complete_sample", pair=st.pair, **bar)
                else:
                    pending.append(st)
            if pending:
                try:
                    await asyncio.wait_for(self.stop.wait(), timeout=1.0)
                    return
                except asyncio.TimeoutError:
                    pass
                for st in pending:  # single retry at boundary+2s (staleness bound)
                    bar, n = shifted_bar(st.window_ticks(bound), bound)
                    if bar is not None:
                        st.windows_complete += 1
                        st.windows_retry_completed += 1
                    else:
                        st.window_incomplete_counts[str(n)] = st.window_incomplete_counts.get(str(n), 0) + 1
                        self.log("window_incomplete", pair=st.pair, end_epoch=bound, n_ticks=n)

    # ---------------- writer

    async def writer(self) -> None:
        while not self.stop.is_set():
            try:
                await asyncio.wait_for(self.stop.wait(), timeout=self.args.flush_seconds)
            except asyncio.TimeoutError:
                pass
            for st in self.states.values():
                n = await asyncio.to_thread(flush_pair, self.tick_store, st)
                if n and st.last_epoch:
                    st.writer_lag_samples.append(float(st.last_epoch - st.persisted_last_epoch))

    # ---------------- candle refresher

    async def refresher(self) -> None:
        # startup warmup: page each daemon store up to the health floor
        for pair in self.candle_pairs:
            path = self.candle_store / f"{pair}.parquet"
            rows = len(pd.read_parquet(path)) if path.exists() else 0
            end: str = "latest"
            while rows < self.args.candle_min_rows and not self.stop.is_set():
                resp = await self.client.request({"ticks_history": PAIR_TO_SYMBOL[pair], "style": "candles",
                                                  "granularity": 60, "count": 5000, "end": end,
                                                  "adjust_start_time": 1}, timeout=30.0)
                candles = resp.get("candles") or []
                if not candles:
                    break
                merged = await asyncio.to_thread(upsert_candles, path, pair, resp, endpoint_label="options-async")
                rows = len(merged)
                end = str(int(candles[0]["epoch"]) - 60)
            self.log("candle_warmup_done", pair=pair, rows=rows)
        last_xstore = 0.0
        while not self.stop.is_set():
            cycle_start = time.time()
            for pair in self.candle_pairs:
                try:
                    resp = await self.client.request({"ticks_history": PAIR_TO_SYMBOL[pair], "style": "candles",
                                                      "granularity": 60, "count": 3, "end": "latest",
                                                      "adjust_start_time": 1})
                    await asyncio.to_thread(upsert_candles, self.candle_store / f"{pair}.parquet", pair, resp,
                                            endpoint_label="options-async")
                    self.refresher_updates += 1
                except DerivAPIError as exc:
                    self.refresher_errors += 1
                    self.log("refresher_error", pair=pair, error=str(exc)[:200])
            if time.time() - last_xstore > 300 and Path(self.args.production_store_dir).exists():
                last_xstore = time.time()
                report = await asyncio.to_thread(
                    xstore_report, self.candle_store, Path(self.args.production_store_dir), self.candle_pairs)
                self.log("xstore_report", **report)
            elapsed = time.time() - cycle_start
            try:
                await asyncio.wait_for(self.stop.wait(), timeout=max(0.0, self.args.refresh_cadence - elapsed))
                return
            except asyncio.TimeoutError:
                pass

    # ---------------- rollup

    def rollup(self) -> dict[str, Any]:
        elapsed = max(1.0, time.time() - self.t0)
        per_pair: dict[str, Any] = {}
        for pair, st in self.states.items():
            loss = st.received + st.recovered - st.persisted - len(st.write_buf)
            lag = st.writer_lag_samples
            win_rate = st.windows_complete / st.windows_expected if st.windows_expected else None
            coverage_live = st.received / elapsed
            coverage = (st.received + st.recovered) / elapsed  # store completeness incl. gap recovery
            per_pair[pair] = {
                "live_ticks_received": st.received,
                "recovered_ticks": st.recovered,
                "dupes": st.dupes,
                "coverage_live": round(coverage_live, 4),
                "coverage_vs_elapsed": round(coverage, 4),
                "gap_max_s": st.gap_max_s,
                "gaps_over_3s": st.gaps_over_3s,
                "windows_expected": st.windows_expected,
                "windows_complete": st.windows_complete,
                "windows_healed_by_recovery": st.windows_healed,
                "windows_retry_completed": st.windows_retry_completed,
                "window_rate": round(win_rate, 4) if win_rate is not None else None,
                "window_incomplete_by_n": st.window_incomplete_counts,
                "persisted": st.persisted,
                "unpersisted_at_stop": len(st.write_buf),
                "ticks_lost_ring_to_store": loss,
                "writer_lag_s": {"p50": pct(lag, 0.5), "p95": pct(lag, 0.95), "max": pct(lag, 1.0)},
                "write_buf_depth_max": st.buf_depth_max,
                "pages": st.page_n,
                "gate_pass": bool(
                    coverage >= 0.995 and win_rate is not None and win_rate >= 0.99 and loss == 0
                    and (not lag or (pct(lag, 0.95) <= 30.0 and max(lag) <= 120.0))
                ),
            }
        return {
            "started_utc": self.started_utc,
            "finished_utc": now_utc_iso(),
            "host": socket.gethostname(),
            "elapsed_s": round(elapsed, 1),
            "coverage_denominator": "run_elapsed_seconds (gate sessions must run session-aligned)",
            "pairs": per_pair,
            "refresher": {"updates": self.refresher_updates, "errors": self.refresher_errors},
            "client_stats": {k: v for k, v in self.client.stats.items() if k != "resubscribe_latency_s"},
            "client_resubscribe_latency_s": self.client.stats["resubscribe_latency_s"],
        }

    async def run(self) -> int:
        # SIGINT is the systemd KillSignal: without handlers the final flush
        # and gate rollup never run and buffered ticks are silently lost on
        # every service stop (review finding H2)
        import signal as _signal

        loop = asyncio.get_running_loop()
        for signum in (_signal.SIGINT, _signal.SIGTERM):
            try:
                loop.add_signal_handler(signum, self.stop.set)
            except NotImplementedError:
                pass
        await self.client.connect()
        self.log("start", pairs=self.pairs, args={k: str(v) for k, v in vars(self.args).items()})
        await asyncio.gather(*(self.warmup_pair(p) for p in self.pairs))
        workers: dict[asyncio.Task, str] = {}
        failure: dict[str, str] | None = None

        def remember_failure(worker: str, error_class: str, error: str) -> None:
            nonlocal failure
            if failure is not None:
                return
            failure = {"worker": worker, "error_class": error_class, "error": error[:200]}
            self.stop.set()
            try:
                self.log("worker_failed", **failure)
            except Exception:
                pass

        def start_worker(name: str, coro: Any) -> asyncio.Task:
            task = asyncio.create_task(coro, name=f"market-stream:{name}")
            workers[task] = name

            def completed(done: asyncio.Task) -> None:
                stopping = self.stop.is_set()
                if done.cancelled():
                    if not stopping:
                        remember_failure(name, "CancelledError", "cancelled before stop")
                    return
                exc = done.exception()
                if exc is not None:
                    remember_failure(name, type(exc).__name__, str(exc))
                elif not stopping:
                    remember_failure(name, "UnexpectedReturn", "returned before stop")

            task.add_done_callback(completed)
            return task

        writer_task = start_worker("writer", self.writer())
        for pair in self.pairs:
            start_worker(f"drain:{pair}", self.drain_pair(pair))
        start_worker("aggregator", self.aggregator())
        start_worker("refresher", self.refresher())
        start_worker("client-background", self.client.wait_for_background_failure())

        if self.args.duration_minutes > 0:
            try:
                await asyncio.wait_for(self.stop.wait(), timeout=self.args.duration_minutes * 60.0)
            except asyncio.TimeoutError:
                pass
        else:
            await self.stop.wait()
        self.stop.set()

        # The writer exits via `stop` and is AWAITED, never cancelled: a
        # cancelled to_thread flush keeps running and races the final flush
        # on the same PairState (review finding M4, empirically confirmed).
        writer_result = (await asyncio.gather(writer_task, return_exceptions=True))[0]
        if isinstance(writer_result, BaseException):
            remember_failure("writer", type(writer_result).__name__, str(writer_result))

        non_writer_tasks = [task for task in workers if task is not writer_task]
        for task in non_writer_tasks:
            if not task.done():
                task.cancel()
        results = await asyncio.gather(*non_writer_tasks, return_exceptions=True)
        for task, result in zip(non_writer_tasks, results):
            if isinstance(result, BaseException) and not isinstance(result, asyncio.CancelledError):
                remember_failure(workers[task], type(result).__name__, str(result))

        for pair, st in self.states.items():  # final flush: zero-loss accounting
            try:
                flush_pair(self.tick_store, st)
            except Exception as exc:
                remember_failure(f"final-flush:{pair}", type(exc).__name__, str(exc))

        run_record: dict[str, Any] | None = None
        gate_written = False
        try:
            run_record = self.rollup()
        except Exception as exc:
            remember_failure("rollup", type(exc).__name__, str(exc))
        try:
            await self.client.close()
        except Exception as exc:
            remember_failure("client-close", type(exc).__name__, str(exc))
        if run_record is not None:
            try:
                self.log(
                    "rollup",
                    runtime_gate_pass=failure is None,
                    **{p: v["gate_pass"] for p, v in run_record["pairs"].items()},
                )
            except Exception as exc:
                remember_failure("rollup-log", type(exc).__name__, str(exc))
        try:
            self.log_fh.close()
        except Exception as exc:
            remember_failure("log-close", type(exc).__name__, str(exc))

        if run_record is not None:
            run_record["runtime_gate"] = {"pass": failure is None, "failure": failure}
        try:
            if run_record is None:
                raise RuntimeError("rollup unavailable")
            doc = {"gate": "issue#4 Phase 2 market stream", "runs": []}
            if GATE_RESULT_PATH.exists():
                try:
                    doc = json.loads(GATE_RESULT_PATH.read_text())
                except json.JSONDecodeError:
                    pass
            doc.setdefault("runs", []).append(run_record)
            tmp = GATE_RESULT_PATH.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
            tmp.replace(GATE_RESULT_PATH)
            gate_written = True
        except Exception as exc:
            remember_failure("gate-record", type(exc).__name__, str(exc))

        ok = bool(
            run_record is not None
            and gate_written
            and all(v["gate_pass"] for v in run_record["pairs"].values())
            and run_record["runtime_gate"]["pass"]
        )
        summary = {} if run_record is None else {
            p: {k: v[k] for k in ("live_ticks_received", "coverage_vs_elapsed", "window_rate",
                                  "ticks_lost_ring_to_store", "gate_pass")}
            for p, v in run_record["pairs"].items()
        }
        print(json.dumps(summary, indent=1, sort_keys=True))
        print(f"{'RUN GATE PASS' if ok else 'RUN GATE FAIL'} -> {GATE_RESULT_PATH}")
        return 0 if ok else 1


# ---------------------------------------------------------------- smoke


async def _run_liveness_smoke(check: Any, root: Path) -> None:
    """Exercise closed-market recovery and fail-visible runtime cleanup offline."""
    global GATE_RESULT_PATH, MARKET_CLOSED_RETRY_SECONDS, STREAM_QUEUE_POLL_SECONDS, write_page_shard

    class SmokeSubscription:
        def __init__(self, frame: dict[str, Any] | None = None):
            self.queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
            if frame is not None:
                self.queue.put_nowait(frame)

    class SmokeClient:
        def __init__(self, mode: str = "ok"):
            self.mode = mode
            self.subscribe_calls = 0
            self.closed = False
            self.stats: dict[str, Any] = {"resubscribe_latency_s": []}
            self._never = asyncio.Event()

        async def connect(self) -> None:
            return None

        async def close(self) -> None:
            self.closed = True

        async def subscribe(self, _payload: dict[str, Any]) -> SmokeSubscription:
            self.subscribe_calls += 1
            if self.mode == "market_closed_once" and self.subscribe_calls == 1:
                raise DerivAPIError("Deriv error: {'code': 'MarketIsClosed'}", code="MarketIsClosed")
            if self.mode == "market_closed_once":
                return SmokeSubscription({"tick": {"epoch": 1000, "quote": 1.25}})
            if self.mode == "market_closed_always":
                raise DerivAPIError("Deriv error: {'code': 'MarketIsClosed'}", code="MarketIsClosed")
            if self.mode == "market_closed_mentioned":
                raise DerivAPIError(
                    "Deriv error: {'code': 'OtherFailure', "
                    "'message': \"upstream said 'code': 'MarketIsClosed'\"}",
                    code="OtherFailure",
                )
            if self.mode == "queued_error":
                return SmokeSubscription({"error": {"code": "InjectedStreamFailure"}})
            return SmokeSubscription()

        async def wait_for_background_failure(self) -> None:
            if self.mode == "background_failure":
                await asyncio.sleep(0)
                raise DerivAPIError("latched stable client failure")
            await self._never.wait()

    def bare_stream(client: SmokeClient) -> MarketStream:
        stream = object.__new__(MarketStream)
        stream.states = {"USDCHF": PairState("USDCHF", ring_seconds=120, page_n=0)}
        stream.client = client
        stream.stop = asyncio.Event()
        stream.events = []
        stream.log = lambda event, **fields: stream.events.append({"event": event, **fields})
        return stream

    old_gate = GATE_RESULT_PATH
    old_retry = MARKET_CLOSED_RETRY_SECONDS
    old_poll = STREAM_QUEUE_POLL_SECONDS
    old_write_page_shard = write_page_shard
    GATE_RESULT_PATH = root / "gate.json"
    MARKET_CLOSED_RETRY_SECONDS = 0.01
    STREAM_QUEUE_POLL_SECONDS = 0.01
    try:
        restored = PairState("USDCHF", ring_seconds=120, page_n=0)
        restored.add_tick(1000, 1.25, math.nan, math.nan, persist=True)
        restore_calls = 0

        def fail_with_new_tick(*_args: Any, **_kwargs: Any) -> Path:
            nonlocal restore_calls
            restore_calls += 1
            restored.add_tick(1001, 1.26, math.nan, math.nan, persist=True)
            raise OSError("injected pre-commit shard failure")

        write_page_shard = fail_with_new_tick
        try:
            flush_pair(root / "buffer-restore", restored)
        except OSError:
            pass
        write_page_shard = old_write_page_shard
        restored_epochs = [row[0] for row in restored.write_buf]
        flush_pair(root / "buffer-restore", restored)
        check(
            "failed shard write restores detached ticks ahead of newer ticks",
            restore_calls == 1
            and restored_epochs == [1000, 1001]
            and restored.persisted == restored.received == 2,
        )

        # Initial MarketIsClosed waits in-process, then consumes and persists the
        # first recovered tick instead of depending on a systemd crash loop.
        recovering_client = SmokeClient("market_closed_once")
        recovering = bare_stream(recovering_client)
        drain = asyncio.create_task(recovering.drain_pair("USDCHF"))
        deadline = time.monotonic() + 0.2
        while recovering.states["USDCHF"].received == 0 and time.monotonic() < deadline:
            await asyncio.sleep(0.002)
        recovering.stop.set()
        await asyncio.wait_for(drain, 0.1)
        recovering_store = root / "closed-recovery"
        flush_pair(recovering_store, recovering.states["USDCHF"])
        progress = json.loads((recovering_store / "USDCHF_progress.json").read_text())
        check(
            "MarketIsClosed retry recovers and persists the first tick",
            recovering_client.subscribe_calls == 2
            and recovering.states["USDCHF"].persisted == 1
            and progress["last_persisted_epoch"] == 1000,
        )

        # Continuous closure is bounded by the fixed wait and remains promptly
        # cancellable when systemd sends SIGINT/SIGTERM.
        closed_client = SmokeClient("market_closed_always")
        closed = bare_stream(closed_client)
        attempts_started = time.monotonic()
        drain = asyncio.create_task(closed.drain_pair("USDCHF"))
        deadline = time.monotonic() + 0.1
        while closed_client.subscribe_calls < 2 and time.monotonic() < deadline:
            await asyncio.sleep(0.001)
        second_attempt_s = time.monotonic() - attempts_started
        stopped_at = time.monotonic()
        closed.stop.set()
        await asyncio.wait_for(drain, 0.05)
        check(
            "continuous MarketIsClosed is bounded and stop-aware",
            2 <= closed_client.subscribe_calls <= 3
            and second_attempt_s >= 0.008
            and time.monotonic() - stopped_at < 0.05,
            f"attempts={closed_client.subscribe_calls}",
        )

        mentioned_client = SmokeClient("market_closed_mentioned")
        mentioned = bare_stream(mentioned_client)
        propagated = False
        try:
            await mentioned.drain_pair("USDCHF")
        except DerivAPIError as exc:
            propagated = exc.code == "OtherFailure"
        check(
            "MarketIsClosed text under another code propagates",
            propagated and mentioned_client.subscribe_calls == 1,
        )

        class SmokeLog:
            def __init__(self) -> None:
                self.closed_observed = False

            def close(self) -> None:
                self.closed_observed = True

        class SmokeRuntime(MarketStream):
            def __init__(self, case: str, case_root: Path, *, preload: bool = False):
                self.args = argparse.Namespace(duration_minutes=0.0005, flush_seconds=0.01)
                self.pairs = ["USDCHF"]
                self.tick_store = case_root / "ticks"
                self.tick_store.mkdir(parents=True)
                self.client = SmokeClient("background_failure" if case == "client" else
                                          "queued_error" if case == "drain" else "ok")
                self.stop = asyncio.Event()
                self.states = {"USDCHF": PairState("USDCHF", ring_seconds=120, page_n=0)}
                self.case = case
                self.events: list[dict[str, Any]] = []
                self.log_fh = SmokeLog()
                if preload:
                    self.states["USDCHF"].add_tick(1000, 1.25, math.nan, math.nan, persist=True)

            def log(self, event: str, **fields: Any) -> None:
                self.events.append({"event": event, **fields})

            async def warmup_pair(self, pair: str) -> None:
                self.states[pair].warm_at_epoch = self.states[pair].last_epoch or 1

            async def writer(self) -> None:
                if self.case == "writer":
                    self.states["USDCHF"].add_tick(1001, 1.26, math.nan, math.nan, persist=True)
                    await asyncio.to_thread(flush_pair, self.tick_store, self.states["USDCHF"])
                    return
                if self.case == "ordered":
                    raise RuntimeError("injected first worker failure")
                await self.stop.wait()

            async def aggregator(self) -> None:
                if self.case == "aggregator":
                    return
                if self.case == "ordered":
                    raise RuntimeError("injected second worker failure")
                await self.stop.wait()

            async def refresher(self) -> None:
                await self.stop.wait()

            def rollup(self) -> dict[str, Any]:
                return {
                    "pairs": {
                        "USDCHF": {
                            "live_ticks_received": self.states["USDCHF"].received,
                            "coverage_vs_elapsed": 1.0,
                            "window_rate": 1.0,
                            "ticks_lost_ring_to_store": 0,
                            "gate_pass": True,
                        }
                    }
                }

        async def run_case(case: str, *, preload: bool = False) -> tuple[int, SmokeRuntime]:
            global write_page_shard
            stream = SmokeRuntime(case, root / case, preload=preload)
            stream.write_attempts = 0
            if case != "writer":
                return await stream.run(), stream

            def fail_first_write(*args: Any, **kwargs: Any) -> Path:
                stream.write_attempts += 1
                if stream.write_attempts == 1:
                    raise OSError("injected writer shard failure")
                return old_write_page_shard(*args, **kwargs)

            write_page_shard = fail_first_write
            try:
                return await stream.run(), stream
            finally:
                write_page_shard = old_write_page_shard

        failure_cases = (
            ("drain", "drain:USDCHF", "queued subscription error"),
            ("writer", "writer", "writer failure"),
            ("aggregator", "aggregator", "unexpected non-writer return"),
            ("client", "client-background", "latched client failure"),
            ("ordered", "writer", "ordered simultaneous failures"),
        )
        for case, worker, label in failure_cases:
            rc, stream = await run_case(case, preload=case != "writer")
            runtime_gate = json.loads(GATE_RESULT_PATH.read_text())["runs"][-1]["runtime_gate"]
            worker_events = [e for e in stream.events if e.get("event") == "worker_failed"]
            first_event = worker_events[0] if worker_events else {}
            recorded_failure = {k: first_event.get(k) for k in ("worker", "error_class", "error")}
            check(
                f"{label} exits nonzero after flush and close",
                rc == 1
                and stream.states["USDCHF"].persisted == 1
                and stream.client.closed
                and stream.log_fh.closed_observed
                and first_event.get("worker") == worker
                and runtime_gate == {"pass": False, "failure": recorded_failure}
                and (case != "writer" or stream.write_attempts == 2),
            )

        rc, stream = await run_case("graceful", preload=True)
        runtime_gate = json.loads(GATE_RESULT_PATH.read_text())["runs"][-1]["runtime_gate"]
        check(
            "graceful duration stop flushes and closes without worker failure",
            rc == 0
            and stream.states["USDCHF"].persisted == 1
            and stream.client.closed
            and stream.log_fh.closed_observed
            and not any(e["event"] == "worker_failed" for e in stream.events)
            and runtime_gate == {"pass": True, "failure": None},
        )
    finally:
        GATE_RESULT_PATH = old_gate
        MARKET_CLOSED_RETRY_SECONDS = old_retry
        STREAM_QUEUE_POLL_SECONDS = old_poll
        write_page_shard = old_write_page_shard


def run_smoke() -> int:
    """Deterministic Phase-2 smoke: aggregator semantics, never-ffill, writer atomicity."""
    failures: list[str] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        print(f"{'PASS' if ok else 'FAIL'}: {name}" + (f" — {detail}" if detail else ""))
        if not ok:
            failures.append(name)

    def q(e: int) -> float:
        return round(1.0 + 0.01 * math.sin(e), 8)

    ticks = [(e, q(e), q(e) - 0.0001, q(e) + 0.0001) for e in range(1000, 1300)]

    # 1. OHLC semantics on right-closed windows, all offsets
    ok_all = True
    for end in (1060, 1065, 1090, 1115, 1170, 1200, 1255, 1299):
        bar, n = shifted_bar(ticks, end)
        exp = [q(e) for e in range(end - 59, end + 1)]
        good = (bar is not None and n == 60 and bar["offset_id"] == end % 60
                and bar["open"] == exp[0] and bar["close"] == exp[-1]
                and bar["high"] == max(exp) and bar["low"] == min(exp))
        ok_all = ok_all and good
    check("shifted_bar OHLC == manual first/max/min/last on (end-60, end]", ok_all)

    # 2. pandas cross-check (closed='right', label='right' == the pinned window)
    s = pd.Series([q(e) for e in range(1000, 1300)],
                  index=pd.to_datetime(range(1000, 1300), unit="s", utc=True))
    r = pd.DataFrame({
        "open": s.resample("60s", closed="right", label="right").first(),
        "high": s.resample("60s", closed="right", label="right").max(),
        "low": s.resample("60s", closed="right", label="right").min(),
        "close": s.resample("60s", closed="right", label="right").last(),
    })
    ok_pd = True
    for ts, row in r.iterrows():
        end = int(ts.timestamp())
        bar, n = shifted_bar(ticks, end)
        if bar is None:
            continue  # partial edge windows
        ok_pd = ok_pd and all(abs(bar[k] - row[k]) < 1e-12 for k in ("open", "high", "low", "close"))
    check("shifted_bar == pandas resample(closed=right,label=right) agg", ok_pd)

    # 3. missing tick -> invalid, never ffilled
    holed = [t for t in ticks if t[0] != 1023]
    bar, n = shifted_bar(holed, 1060)
    check("59/60 window invalid (never ffilled)", bar is None and n == 59, f"n={n}")

    # 4. duplicate epoch dedup keep-last
    dup = ticks + [(1050, 9.9, 9.89, 9.91)]
    bar, n = shifted_bar(dup, 1060)
    check("duplicate epoch keep-last (high reflects late dup)", bar is not None and bar["high"] == 9.9)

    # 4b. ring-healing: a live-skipped second recovered from history heals the
    # window with the TRUE tick (complete, never ffilled); live wins on dup
    st_heal = PairState("USDJPY", ring_seconds=7200, page_n=0)
    for e, quote, bid, ask in ticks:
        if e != 1023:
            st_heal.add_tick(e, quote, bid, ask, persist=False)
    bar, n = shifted_bar(st_heal.window_ticks(1060), 1060)
    check("hole window incomplete before heal", bar is None and n == 59)
    st_heal.recovered_map[1023] = (q(1023), math.nan, math.nan)
    bar, n = shifted_bar(st_heal.window_ticks(1060), 1060)
    check("healed window complete with the true tick", bar is not None and n == 60 and bar["close"] == q(1060))
    st_heal.recovered_map[1040] = (99.9, math.nan, math.nan)  # dup epoch: live ring tick must win
    bar, _ = shifted_bar(st_heal.window_ticks(1060), 1060)
    check("live tick wins over recovered on duplicate epoch", bar is not None and bar["high"] != 99.9)

    # 5. writer atomicity + zero-loss accounting (smoke dir under deriv_data, cleaned up)
    smoke_dir = ensure_deriv_data_path(Path("deriv_data/ticks_1s_smoke"), purpose="smoke tick store")
    if smoke_dir.exists():
        shutil.rmtree(smoke_dir)
    smoke_dir.mkdir(parents=True)
    try:
        st = PairState("USDJPY", ring_seconds=7200, page_n=0)
        for e, quote, bid, ask in ticks[:120]:
            st.add_tick(e, quote, bid, ask, persist=True)
        n1 = flush_pair(smoke_dir, st)
        for e, quote, bid, ask in ticks[120:150]:
            st.add_tick(e, quote, bid, ask, persist=True)
        n2 = flush_pair(smoke_dir, st)
        shards = sorted((smoke_dir / "_pages" / "USDJPY").glob("*.parquet"))
        tmps = list(smoke_dir.rglob("*.tmp"))
        reloaded = pd.concat([pd.read_parquet(p) for p in shards])
        prog = json.loads((smoke_dir / "USDJPY_progress.json").read_text())
        check("two atomic shards, no tmp files", len(shards) == 2 and not tmps)
        check("persisted == received == reloaded rows",
              st.persisted == st.received == len(reloaded) == 150 and n1 + n2 == 150,
              f"persisted={st.persisted} reloaded={len(reloaded)}")
        check("progress cursor matches", prog["live_ticks_persisted"] == 150 and prog["last_persisted_epoch"] == 1149)
        rows = compact_pair(smoke_dir, "USDJPY")
        check("compaction dedups to 150 sorted rows", rows == 150)

        # 6. candle upsert: open candle dropped, overlap keeps last
        now_min = (int(time.time()) // 60) * 60
        payload1 = {"candles": [{"epoch": now_min - 180, "open": 1, "high": 2, "low": 0.5, "close": 1.5},
                                {"epoch": now_min - 120, "open": 1, "high": 2, "low": 0.5, "close": 1.6},
                                {"epoch": now_min, "open": 1, "high": 1, "low": 1, "close": 1}]}
        payload2 = {"candles": [{"epoch": now_min - 120, "open": 1, "high": 2, "low": 0.5, "close": 1.7}]}
        cpath = smoke_dir / "USDJPY_candles.parquet"
        m1 = upsert_candles(cpath, "USDJPY", payload1, endpoint_label="smoke")
        m2 = upsert_candles(cpath, "USDJPY", payload2, endpoint_label="smoke")
        check("open candle dropped on upsert", int(m1["epoch"].max()) <= now_min - 60, f"max={int(m1['epoch'].max())}")
        check("overlapping upsert keeps last", float(m2.loc[m2["epoch"] == now_min - 120, "close"].iloc[0]) == 1.7)
    finally:
        shutil.rmtree(smoke_dir, ignore_errors=True)

    with tempfile.TemporaryDirectory(prefix="deriv-market-stream-smoke-") as tmp_dir:
        asyncio.run(_run_liveness_smoke(check, Path(tmp_dir)))

    print(f"\n{'SMOKE ALL PASS' if not failures else f'SMOKE {len(failures)} FAILURES: {failures}'}")
    return 1 if failures else 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--compact", action="store_true", help="merge tick shards into per-pair parquet, then exit")
    p.add_argument("--pairs", default="all-enabled")
    p.add_argument("--duration-minutes", type=float, default=0.0, help="0 = run until interrupted")
    p.add_argument("--url", default=PUBLIC_WS_URL)
    p.add_argument("--tick-store-dir", default=str(DEFAULT_TICK_STORE))
    p.add_argument("--candle-store-dir", default=str(DEFAULT_CANDLE_STORE))
    p.add_argument("--production-store-dir", default=str(PRODUCTION_CANDLE_STORE))
    p.add_argument("--warmup-seconds", type=int, default=3600)
    p.add_argument("--ring-seconds", type=int, default=7200)
    p.add_argument("--flush-seconds", type=float, default=10.0)
    p.add_argument("--refresh-cadence", type=float, default=5.0)
    p.add_argument("--candle-min-rows", type=int, default=14400)
    p.add_argument("--log-dir", default="logs/deriv_market_stream")
    args = p.parse_args()
    if args.smoke:
        return run_smoke()
    if args.compact:
        store = ensure_deriv_data_path(Path(args.tick_store_dir), purpose="tick store")
        pairs = parse_pairs(args.pairs) if args.pairs != "all-enabled" else list(DEFAULT_ENABLED_PAIRS)
        for pair in pairs:
            print(f"{pair}: {compact_pair(store, pair)} rows")
        return 0
    return asyncio.run(MarketStream(args).run())


if __name__ == "__main__":
    sys.exit(main())
