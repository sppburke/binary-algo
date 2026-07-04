"""Hot wall-clock scoring daemon (issue #4 Phase 3): offset 0, proposal-only.

Books, floors, and the Deriv connection are loaded ONCE per process — the
point of the daemon — then every newly completed wall-clock minute is scored
through the SAME chain the one-shot executor uses: `LiveFeatureBuilder`
(store path) -> `book_runtime.LoadedBook.score` -> `deriv_runtime_core.
effective_floor_for` / `parse_quote` / `is_ny_session`. Parity with the
executor is by construction at this shared layer (KILL #0); this file adds no
decision logic of its own and contains NO buy path whatsoever.

Snapshot-replay parity harness (issue #4 r3): before scoring minute M the
daemon copies the candle-store files it is about to read into
`<snapshot-dir>/<M>/`, then scores FROM that snapshot — the archived bytes
are the scored input by construction. `--parity-replay` re-runs the shared
chain over each archived snapshot offline (relaxed staleness, no live calls)
and diffs the deterministic tuple `(pair, candidate_bar_close_utc,
feature_sha256, proba, selected_side, effective_floor)` against the daemon's
logged tuples — 100% or the divergences are listed.

Latency events per scored minute: `candidate_bar_closed`, `feature_row_ready`,
`book_scored`, `quote_state_updated`, `edge_gate_passed` / skip reasons.
Gate rollups (KILL #1 signal->proposal p95 <= 2000ms; KILL #2
candidate-close->score p95 <= 1000ms) append to
`deriv_hot_daemon_gate_result.json`.

Run from repo root:
    ~/binary-algo-venv/bin/python scripts/deriv_hot_daemon.py --once
    ~/binary-algo-venv/bin/python scripts/deriv_hot_daemon.py --duration-minutes 540
    ~/binary-algo-venv/bin/python scripts/deriv_hot_daemon.py --parity-replay logs/deriv_hot_daemon/<run>.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import shutil
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from book_runtime import LoadedBook, load_target_books
from deriv_async_client import DerivAsyncClient
from deriv_backfill import PAIR_TO_SYMBOL, ensure_deriv_data_path, store_health
from deriv_client import PUBLIC_WS_URL, DerivAPIError, DerivOptionsClient
from deriv_runtime_core import (
    HORIZON_MINUTES,
    JsonlLogger,
    ExecutorError,
    effective_floor_for,
    is_ny_session,
    parse_quote,
    resolve_floor_resolutions,
    resolve_pairs,
)
from live_features import DERIV_SYMBOLS, LiveFeatureBuilder, LiveFeatureError, PAIRS

GATE_RESULT_PATH = Path("deriv_hot_daemon_gate_result.json")
PARITY_RESULT_PATH = Path("deriv_parity_result.json")
ONCE_STORE_WAIT_MAX_S = 90.0  # --once self-bounds the stale-store wait so a one-shot never hangs


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    xs = sorted(values)
    return round(xs[min(len(xs) - 1, int(q * (len(xs) - 1)))], 1)


def feature_sha(row: pd.Series, cols: list[str]) -> str:
    return hashlib.sha256(row[cols].astype("float64").to_numpy().tobytes()).hexdigest()[:16]


def newest_completed_epoch(store_dir: Path, pair: str) -> int | None:
    path = store_dir / f"{pair}.parquet"
    if not path.exists():
        return None
    frame = pd.read_parquet(path, columns=["epoch"])
    return int(frame["epoch"].max()) if len(frame) else None


def snapshot_store(store_dir: Path, snap_dir: Path, pairs: list[str]) -> Path:
    """Byte-copy the store files about to be scored; scoring reads FROM here."""
    snap_dir.mkdir(parents=True, exist_ok=True)
    for pair in pairs:
        src = store_dir / f"{pair}.parquet"
        if src.exists():
            shutil.copy2(src, snap_dir / f"{pair}.parquet")
    return snap_dir


def prune_snapshots(root: Path, keep: int) -> None:
    snaps = sorted(p for p in root.iterdir() if p.is_dir()) if root.exists() else []
    for old in snaps[:-keep] if keep else []:
        shutil.rmtree(old, ignore_errors=True)


def score_from_snapshot(
    snap_dir: Path,
    pair: str,
    book: LoadedBook,
    resolution: Any,
    payout_edge_margin: float,
    *,
    stale_seconds: int,
) -> dict[str, Any]:
    """The shared deterministic chain — used identically by the live loop and
    the parity replay. Returns the parity tuple + the feature row timestamp."""
    builder = LiveFeatureBuilder(
        DerivOptionsClient.public(),  # store path never issues a request
        store_dir=snap_dir,
        stale_seconds=stale_seconds,
    )
    row = builder.feature_row(pair, book.feature_cols)
    score = book.score(row.row)
    return {
        "pair": pair,
        "candidate_bar_close_utc": (row.timestamp + pd.Timedelta(minutes=1)).isoformat(),
        "feature_sha256": feature_sha(row.row, book.feature_cols),
        "proba": round(score.proba, 12),
        "selected_side": score.direction,
        "effective_floor": round(effective_floor_for(resolution, score.direction, payout_edge_margin), 12),
        "gate_passed": score.gate_passed,
        "gate_reasons": score.gate_reasons,
        "confidence": round(score.confidence, 12),
        "threshold": score.threshold,
        "row_source": row.source,
    }


class HotDaemon:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.pairs = resolve_pairs(args.pairs)
        self.store_dir = ensure_deriv_data_path(Path(args.store_dir), purpose="daemon candle store")
        self.snap_root = ensure_deriv_data_path(Path(args.snapshot_dir), purpose="hot snapshots")
        self.snap_root.mkdir(parents=True, exist_ok=True)
        self.logger = JsonlLogger(Path(args.log_dir))
        self.books: dict[str, LoadedBook] = {}
        self.resolutions: dict[str, Any] = {}
        self.client = DerivAsyncClient(args.url, on_event=lambda e, f: self.logger.write("client_" + e, **f))
        self.last_scored_epoch: dict[str, int] = {}
        self.score_ms: list[float] = []
        self.proposal_ms: list[float] = []
        self.scored = 0
        self.skips: dict[str, int] = {}
        self.started_utc = now_utc_iso()
        # Phase-4 hook: the supervisor routes edge-passed signals through the
        # coordinator. Unset (Phase 3) the daemon stays pure proposal-only.
        self.signal_sink: Any = None

    def load_once(self) -> None:
        self.books = load_target_books(self.pairs)
        self.resolutions = resolve_floor_resolutions(self.pairs, self.books, self.logger)
        self.logger.write("daemon_loaded", pairs=self.pairs,
                          books={p: b.book_id for p, b in self.books.items()})

    def check_store(self) -> dict[str, Any]:
        return store_health(self.store_dir, enabled_pairs=self.pairs,
                            min_required_rows=self.args.store_min_required_rows,
                            max_stale_seconds=self.args.store_max_stale_seconds)

    def _resolve_wait_max(self) -> float:
        """Effective stale-store wait bound: an explicit --store-wait-max-seconds
        wins; else 0 (wait forever) for the service, ONCE_STORE_WAIT_MAX_S for --once."""
        explicit = getattr(self.args, "store_wait_max_seconds", 0.0)
        if explicit:
            return float(explicit)
        return ONCE_STORE_WAIT_MAX_S if self.args.once else 0.0

    async def wait_for_store(self) -> dict[str, Any]:
        """Idle until the candle store is fresh instead of crash-looping.

        A stale store means the FX market is closed (weekend/holiday — the
        market-stream refresher has no new bars) or is still warming at boot.
        Either way we WAIT and re-check rather than fail closed: the old
        review-M5 behaviour raised and relied on systemd restarts "until the
        store is warm", which loops forever whenever the market is closed
        (confirmed on the VPS: 25 store_unhealthy crashes, restart counter
        16+). `await asyncio.sleep` is cancellable, so a SIGINT from
        `systemctl stop` unwinds cleanly. Unbounded by default so the service
        auto-resumes at market open; a positive --store-wait-max-seconds
        re-raises (fail-fast) and --once self-bounds so a one-shot never hangs.
        """
        poll = getattr(self.args, "store_wait_poll_seconds", 60.0)
        max_s = self._resolve_wait_max()
        deadline = time.monotonic() + max_s if max_s else None
        waits = 0
        while True:
            health = self.check_store()
            if health.get("passes"):
                if waits:
                    self.logger.write("store_ready_resumed", waited_polls=waits)
                return health
            stale = max((v.get("stale_seconds", 0) for v in health.get("per_pair", {}).values()),
                        default=None)
            self.logger.write("store_unhealthy", waiting=True, waited_polls=waits,
                              stale_seconds=stale, detail=json.dumps(health, default=str)[:300])
            if deadline is not None and time.monotonic() >= deadline:
                raise ExecutorError(
                    "daemon candle store still stale after --store-wait-max-seconds; "
                    "is deriv-market-stream running?")
            waits += 1
            await asyncio.sleep(poll)

    async def score_minute(self, pair: str, bar_epoch: int, snap_dir: Path) -> None:
        t_detect = time.monotonic()
        bar_close_utc = datetime.fromtimestamp(bar_epoch + 60, timezone.utc)
        self.logger.write("candidate_bar_closed", pair=pair, bar_epoch=bar_epoch,
                          bar_close_utc=bar_close_utc.isoformat())
        try:
            tup = await asyncio.to_thread(
                score_from_snapshot, snap_dir, pair, self.books[pair], self.resolutions[pair],
                self.args.payout_edge_margin, stale_seconds=self.args.store_max_stale_seconds)
        except (LiveFeatureError, ExecutorError, DerivAPIError, Exception) as exc:
            self.skips["feature_row_failed"] = self.skips.get("feature_row_failed", 0) + 1
            self.logger.write("signal_skipped", pair=pair, reason="feature_row_failed", error=str(exc)[:300])
            return
        score_done_ms = (time.monotonic() - t_detect) * 1000
        self.score_ms.append(score_done_ms)
        self.scored += 1
        self.logger.write("book_scored", pair=pair, score_ms=round(score_done_ms, 1))
        self.logger.write("parity_tuple", snapshot=str(snap_dir), **tup)
        if not is_ny_session(datetime.now(timezone.utc)):
            self.skips["outside_ny_session"] = self.skips.get("outside_ny_session", 0) + 1
            self.logger.write("signal_skipped", pair=pair, reason="outside_ny_session")
            return
        if not tup["gate_passed"]:
            self.skips["book_gate"] = self.skips.get("book_gate", 0) + 1
            self.logger.write("signal_skipped", pair=pair, reason="book_gate", gate_reasons=tup["gate_reasons"])
            return
        # proposal-only quote for the selected side (KILL #1 to-proposal); NO buy path exists here
        contract_type = "CALL" if tup["selected_side"] == "UP" else "PUT"
        try:
            resp = await self.client.request({
                "proposal": 1, "amount": float(self.args.stake), "basis": "stake",
                "contract_type": contract_type, "currency": "USD",
                "duration": HORIZON_MINUTES, "duration_unit": "m",
                "underlying_symbol": DERIV_SYMBOLS[pair],
            })
        except DerivAPIError as exc:
            self.skips["proposal_failed"] = self.skips.get("proposal_failed", 0) + 1
            self.logger.write("signal_skipped", pair=pair, reason="proposal_failed", error=str(exc)[:200])
            return
        prop_ms = (time.monotonic() - t_detect) * 1000
        self.proposal_ms.append(prop_ms)
        ask, payout, live_breakeven, invalid_reason = parse_quote(resp.get("proposal") or {})
        self.logger.write("quote_state_updated", pair=pair, side=tup["selected_side"], ask=ask,
                          payout=payout, live_breakeven=live_breakeven, proposal_ms=round(prop_ms, 1))
        if invalid_reason is not None:
            self.skips[invalid_reason] = self.skips.get(invalid_reason, 0) + 1
            self.logger.write("signal_skipped", pair=pair, reason=invalid_reason)
        elif tup["effective_floor"] <= live_breakeven:
            self.skips["edge_not_positive"] = self.skips.get("edge_not_positive", 0) + 1
            self.logger.write("signal_skipped", pair=pair, reason="edge_not_positive",
                              effective_floor=tup["effective_floor"], live_breakeven=live_breakeven)
        elif live_breakeven > self.args.max_breakeven:
            self.skips["breakeven_too_high"] = self.skips.get("breakeven_too_high", 0) + 1
            self.logger.write("signal_skipped", pair=pair, reason="breakeven_too_high",
                              live_breakeven=live_breakeven)
        else:
            self.logger.write("edge_gate_passed", pair=pair, side=tup["selected_side"],
                              effective_floor=tup["effective_floor"], live_breakeven=live_breakeven,
                              net_edge=round(tup["effective_floor"] - live_breakeven, 6),
                              mode="proposal_only_phase3" if self.signal_sink is None else "routed_to_coordinator")
            if self.signal_sink is not None:
                await self.signal_sink({
                    "pair": pair, "side": tup["selected_side"], "offset_id": 0,
                    "signal_close_utc": tup["candidate_bar_close_utc"],
                    "effective_floor": tup["effective_floor"], "lane": "wall",
                    "bar_age_s": round(time.time() - (bar_epoch + 60), 1),
                    "is_latest_bar": True,
                })

    async def run(self) -> int:
        self.load_once()
        await self.wait_for_store()  # idle (not crash) while the market is closed / store warming
        await self.client.connect()
        deadline = time.monotonic() + self.args.duration_minutes * 60 if self.args.duration_minutes else None
        once_deadline = time.monotonic() + 90.0 if self.args.once else None  # loud failure, never a silent hang
        while True:
            fresh: list[tuple[str, int]] = []
            for pair in self.pairs:
                epoch = await asyncio.to_thread(newest_completed_epoch, self.store_dir, pair)
                if epoch is not None and epoch > self.last_scored_epoch.get(pair, 0):
                    fresh.append((pair, epoch))
            if fresh:
                minute_tag = max(e for _, e in fresh)
                snap_dir = await asyncio.to_thread(
                    snapshot_store, self.store_dir, self.snap_root / str(minute_tag), list(PAIRS))
                for pair, epoch in fresh:
                    await self.score_minute(pair, epoch, snap_dir)
                    self.last_scored_epoch[pair] = epoch
                await asyncio.to_thread(prune_snapshots, self.snap_root, self.args.snapshot_keep_minutes)
            if self.args.once and self.scored + sum(self.skips.values()) >= len(self.pairs):
                break
            if once_deadline is not None and time.monotonic() > once_deadline:
                self.logger.write("once_timeout", scored=self.scored, skips=self.skips)
                print(f"ERROR: --once timed out; scored={self.scored} skips={self.skips}", file=sys.stderr)
                await self.client.close()
                self.rollup()
                return 1
            if deadline is not None and time.monotonic() > deadline:
                break
            await asyncio.sleep(self.args.poll_seconds)
        await self.client.close()
        return self.rollup()

    def rollup(self) -> int:
        run = {
            "started_utc": self.started_utc,
            "finished_utc": now_utc_iso(),
            "host": socket.gethostname(),
            "pairs": self.pairs,
            "minutes_scored": self.scored,
            "skips": self.skips,
            "candidate_close_to_score_done_ms": {"p50": pct(self.score_ms, 0.5), "p95": pct(self.score_ms, 0.95), "max": pct(self.score_ms, 1.0)},
            "signal_to_proposal_ms": {"p50": pct(self.proposal_ms, 0.5), "p95": pct(self.proposal_ms, 0.95), "max": pct(self.proposal_ms, 1.0)},
            # None = unmeasured this run (no scored minutes / no proposals) — never a fake verdict
            "kill2_score_p95_under_1000": (pct(self.score_ms, 0.95) <= 1000.0) if self.score_ms else None,
            "kill1_proposal_p95_under_2000": (pct(self.proposal_ms, 0.95) <= 2000.0) if self.proposal_ms else None,
        }
        doc = {"gate": "issue#4 Phase 3 hot daemon", "runs": []}
        if GATE_RESULT_PATH.exists():
            try:
                doc = json.loads(GATE_RESULT_PATH.read_text())
            except json.JSONDecodeError:
                pass
        doc.setdefault("runs", []).append(run)
        tmp = GATE_RESULT_PATH.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
        tmp.replace(GATE_RESULT_PATH)
        print(json.dumps(run, indent=1, sort_keys=True))
        return 0


def parity_replay(daemon_log: Path, args: argparse.Namespace) -> int:
    """KILL #0a harness: recompute every logged parity tuple from its archived
    snapshot through the shared chain; 100% match or list divergences."""
    tuples = []
    for line in daemon_log.read_text().splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("event") == "parity_tuple":
            tuples.append(row)
    if not tuples:
        print("no parity_tuple events in log")
        return 1
    pairs = sorted({t["pair"] for t in tuples})
    books = load_target_books(pairs)
    logger = JsonlLogger(Path(args.log_dir))
    resolutions = resolve_floor_resolutions(pairs, books, logger)
    checked = 0
    divergences: list[dict[str, Any]] = []
    keys = ("candidate_bar_close_utc", "feature_sha256", "proba", "selected_side", "effective_floor")
    for t in tuples:
        snap = Path(t["snapshot"])
        if not snap.exists():
            divergences.append({"pair": t["pair"], "snapshot": str(snap), "reason": "snapshot_pruned"})
            continue
        replayed = score_from_snapshot(snap, t["pair"], books[t["pair"]], resolutions[t["pair"]],
                                       args.payout_edge_margin, stale_seconds=10**9)
        checked += 1
        diff = {k: {"daemon": t[k], "replay": replayed[k]} for k in keys if t[k] != replayed[k]}
        if diff:
            divergences.append({"pair": t["pair"], "snapshot": str(snap), "diff": diff})
    result = {
        "gate": "issue#4 KILL #0a snapshot-replay parity",
        "daemon_log": str(daemon_log),
        "replayed_utc": now_utc_iso(),
        "tuples_logged": len(tuples),
        "tuples_checked": checked,
        "tuples_skipped_pruned": len([d for d in divergences if d.get("reason") == "snapshot_pruned"]),
        "divergences": [d for d in divergences if "diff" in d],
        "parity_100pct": checked > 0 and not any("diff" in d for d in divergences),
    }
    tmp = PARITY_RESULT_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    tmp.replace(PARITY_RESULT_PATH)
    print(json.dumps({k: result[k] for k in ("tuples_logged", "tuples_checked", "parity_100pct")}, indent=1))
    print(f"{'PARITY PASS' if result['parity_100pct'] else 'PARITY FAIL'} -> {PARITY_RESULT_PATH}")
    return 0 if result["parity_100pct"] else 1


def _smoke_events(log_dir: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for f in log_dir.glob("*.jsonl"):
        for line in f.read_text().splitlines():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


async def run_smoke() -> int:
    """Deterministic idle-until-reopen smoke (no network, no real store). Scripts
    check_store() to prove wait_for_store idles on a stale store, resumes when it
    turns fresh, logs the wait, and fail-fasts under a bounded max / --once."""
    import tempfile

    checks: list[tuple[str, bool]] = []

    def check(name: str, ok: bool) -> None:
        checks.append((name, bool(ok)))
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")

    tmp = Path(tempfile.mkdtemp(prefix="hot_daemon_smoke_"))
    FRESH = {"passes": True, "per_pair": {}}
    STALE = {"passes": False, "per_pair": {"USDJPY": {"stale_seconds": 5733}}}

    def daemon(idx: int, seq: Any, **over: Any) -> HotDaemon:
        # bypass __init__ (ensure_deriv_data_path + DerivAsyncClient); wait_for_store
        # needs only .args, .logger, and check_store, which we script here.
        d = object.__new__(HotDaemon)
        d.args = argparse.Namespace(
            log_dir=str(tmp / f"logs_{idx}"), once=over.get("once", False),
            store_wait_poll_seconds=over.get("store_wait_poll_seconds", 0.01),
            store_wait_max_seconds=over.get("store_wait_max_seconds", 0.0))
        d.logger = JsonlLogger(Path(d.args.log_dir))
        if isinstance(seq, list):
            d.check_store = lambda s=seq: s.pop(0)  # type: ignore[method-assign]
        else:
            d.check_store = lambda h=seq: h  # type: ignore[method-assign]
        return d

    # 1. idles on a stale store, resumes when it turns fresh, never raises
    d1 = daemon(1, [dict(STALE), dict(STALE), dict(FRESH)])
    h1 = await d1.wait_for_store()
    ev1 = _smoke_events(Path(d1.args.log_dir))
    waiting = [e for e in ev1 if e.get("event") == "store_unhealthy" and e.get("waiting")]
    resumed = [e for e in ev1 if e.get("event") == "store_ready_resumed"]
    check("idles 2 polls then resumes fresh, no raise",
          bool(h1.get("passes")) and len(waiting) == 2 and len(resumed) == 1)
    check("logs stale_seconds while waiting", bool(waiting) and all(e.get("stale_seconds") == 5733 for e in waiting))

    # 2. fresh immediately -> returns with no waiting / resume events
    d2 = daemon(2, dict(FRESH))
    await d2.wait_for_store()
    ev2 = _smoke_events(Path(d2.args.log_dir))
    check("fresh store returns with no waiting events",
          not any(e.get("event") in ("store_unhealthy", "store_ready_resumed") for e in ev2))

    # 3. persistent stale + explicit bounded max -> fail-fast ExecutorError
    raised = False
    try:
        await daemon(3, dict(STALE), store_wait_max_seconds=0.03).wait_for_store()
    except ExecutorError:
        raised = True
    check("explicit --store-wait-max-seconds re-raises on persistent stale", raised)

    # 4. --once self-bounds; service (once=False, no max) waits forever (bound == 0)
    check("--once self-bounds the wait", daemon(4, dict(FRESH), once=True)._resolve_wait_max() == ONCE_STORE_WAIT_MAX_S)
    check("service waits unbounded by default", daemon(5, dict(FRESH), once=False)._resolve_wait_max() == 0.0)

    shutil.rmtree(tmp, ignore_errors=True)
    passed = sum(1 for _, v in checks if v)
    ok = passed == len(checks)
    print(f"{'SMOKE PASS' if ok else 'SMOKE FAIL'} ({passed}/{len(checks)})")
    return 0 if ok else 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pairs", default="all-enabled")
    p.add_argument("--once", action="store_true", help="score the latest completed bar per pair, then exit")
    p.add_argument("--duration-minutes", type=float, default=0.0)
    p.add_argument("--parity-replay", type=Path, default=None, metavar="DAEMON_JSONL")
    p.add_argument("--store-dir", default="deriv_data/candles_1m_daemon")
    p.add_argument("--snapshot-dir", default="deriv_data/hot_snapshots")
    p.add_argument("--snapshot-keep-minutes", type=int, default=240)
    p.add_argument("--poll-seconds", type=float, default=2.0)
    p.add_argument("--store-min-required-rows", type=int, default=14000)
    p.add_argument("--store-max-stale-seconds", type=int, default=180)
    p.add_argument("--store-wait-poll-seconds", type=float, default=60.0,
                   help="when the store is stale (market closed/warming) re-check every N seconds instead of crashing")
    p.add_argument("--store-wait-max-seconds", type=float, default=0.0,
                   help="0 = wait indefinitely and auto-resume at market open; >0 re-raises after N stale seconds (fail-fast)")
    p.add_argument("--smoke", action="store_true", help="deterministic in-process idle-wait smoke (no network, no store)")
    p.add_argument("--payout-edge-margin", type=float, default=0.005)
    p.add_argument("--max-breakeven", type=float, default=0.60)
    p.add_argument("--stake", type=float, default=1.0)
    p.add_argument("--log-dir", default="logs/deriv_hot_daemon")
    p.add_argument("--url", default=PUBLIC_WS_URL)
    args = p.parse_args()
    if args.smoke:
        return asyncio.run(run_smoke())
    if args.parity_replay is not None:
        return parity_replay(args.parity_replay, args)
    return asyncio.run(HotDaemon(args).run())


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ExecutorError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)
