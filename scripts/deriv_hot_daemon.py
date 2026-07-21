"""Hot offset-0 proposal-only daemon and issue-#20 KILL #0 parity harness.

Books, floors, and the public Deriv connection are loaded once. Each newly
completed minute is scored from a uniquely named byte snapshot through the
shared feature, book, canonical-floor, session, and payout owners. This module
contains no buy call or credential/account access.

The full-session replay is deliberately fail-closed. It replays every logged
attempt through both the hot scorer and `deriv_demo_executor.process_pair`,
checks every recorded quote through the active `TradeExecutor` payout wrapper
inside a temporary queue, validates immutable snapshot/run/repository/runtime
identities, and requires the exact six-pair 08:00–16:59 New York grid. Network,
credential, persistent-queue, account, and buy methods are hard sentinels.

Latency events per scored minute: `candidate_bar_closed`, `feature_row_ready`,
`book_scored`, `quote_state_updated`, `edge_gate_passed` / skip reasons.
Gate rollups (KILL #1 signal->proposal p95 <= 2000ms; KILL #2
candidate-close->score p95 <= 1000ms) append to
`deriv_hot_daemon_gate_result.json`.

Run from repo root:
    ~/binary-algo-venv/bin/python scripts/deriv_hot_daemon.py --once
    ~/binary-algo-venv/bin/python scripts/deriv_hot_daemon.py --duration-minutes 540
    ~/binary-algo-venv/bin/python scripts/deriv_hot_daemon.py --parity-replay <run>.jsonl --require-full-ny-session YYYY-MM-DD
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from collections import Counter
from datetime import date, datetime, time as datetime_time, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd

from book_runtime import LoadedBook, load_target_books
from deriv_async_client import DerivAsyncClient
from deriv_backfill import PAIR_TO_SYMBOL, ensure_deriv_data_path, store_health
from deriv_client import PUBLIC_WS_URL, DerivAPIError, DerivOptionsClient
from deriv_runtime_core import (
    ENABLED_PAIRS,
    HORIZON_MINUTES,
    JsonlLogger,
    ExecutorError,
    NY_TZ,
    add_absolute_breakeven_ceiling_arg,
    effective_floor_for,
    evaluate_payout_gate,
    feature_row_sha256,
    is_ny_session,
    normalize_absolute_breakeven_ceiling,
    resolve_floor_resolutions,
    resolve_pairs,
)
from live_features import DERIV_SYMBOLS, LiveFeatureBuilder, LiveFeatureError, PAIRS

GATE_RESULT_PATH = Path("deriv_hot_daemon_gate_result.json")
PARITY_RESULT_PATH = Path("deriv_parity_result.json")
ONCE_STORE_WAIT_MAX_S = 90.0  # --once self-bounds the stale-store wait so a one-shot never hangs
PARITY_TRACKED_PATHS = (
    "scripts/book_runtime.py",
    "scripts/live_features.py",
    "scripts/deriv_floor_resolver.py",
    "scripts/deriv_runtime_core.py",
    "scripts/deriv_demo_executor.py",
    "scripts/deriv_hot_daemon.py",
    "scripts/deriv_trade_executor.py",
    "scripts/deriv_trade_queue.py",
    "books/INDEX.json",
)


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    xs = sorted(values)
    return round(xs[min(len(xs) - 1, int(q * (len(xs) - 1)))], 1)


def feature_sha(row: pd.Series, cols: list[str]) -> str:
    return feature_row_sha256(row, cols)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot_tree_sha256(root: Path) -> str:
    """Hash snapshot names, sizes, and bytes so replay detects any overwrite."""
    digest = hashlib.sha256()
    files = sorted(path for path in root.rglob("*") if path.is_file())
    for path in files:
        rel = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(rel).to_bytes(4, "big"))
        digest.update(rel)
        digest.update(path.stat().st_size.to_bytes(8, "big"))
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _git_value(*args: str) -> str | None:
    try:
        proc = subprocess.run(
            ["git", *args], cwd=Path(__file__).resolve().parents[1],
            check=True, capture_output=True, text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return proc.stdout.strip()


def parity_runtime_identity(
    books: dict[str, LoadedBook],
    *,
    payout_edge_margin: float,
    absolute_breakeven_ceiling: float,
    queue_mode: bool,
    public_endpoint_exact: bool,
) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    versions: dict[str, str | None] = {}
    for package in ("numpy", "pandas", "pyarrow", "lightgbm", "scikit-learn", "joblib"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    status = _git_value("status", "--porcelain", "--untracked-files=no")
    return {
        "repo_head": _git_value("rev-parse", "HEAD"),
        "origin_main": _git_value("rev-parse", "origin/main"),
        "tracked_worktree_clean": status == "" if status is not None else None,
        "tracked_file_sha256": {
            rel: _file_sha256(root / rel) if (root / rel).is_file() else None
            for rel in PARITY_TRACKED_PATHS
        },
        "books": {
            pair: {"book_id": book.book_id, "content_id": book.content_id}
            for pair, book in sorted(books.items())
        },
        "runtime": {
            "python_version": sys.version,
            "python_executable": sys.executable,
            "packages": versions,
        },
        "config": {
            "payout_edge_margin": payout_edge_margin,
            "absolute_breakeven_ceiling": absolute_breakeven_ceiling,
            "queue_mode": queue_mode,
            "public_endpoint_exact": public_endpoint_exact,
            "credential_environment_present": {
                name: bool(os.environ.get(name))
                for name in ("DERIV_APP_ID", "DERIV_PAT", "DERIV_ACCOUNT_ID")
            },
        },
    }


def _identity_sha256(identity: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode("utf-8")).hexdigest()


def parity_attempt_id(run_id: str, snapshot_id: str, pair: str, bar_epoch: int) -> str:
    return hashlib.sha256(
        f"{run_id}|{snapshot_id}|{pair}|{bar_epoch}".encode("utf-8")
    ).hexdigest()


def newest_completed_epoch(store_dir: Path, pair: str) -> int | None:
    path = store_dir / f"{pair}.parquet"
    if not path.exists():
        return None
    frame = pd.read_parquet(path, columns=["epoch"])
    return int(frame["epoch"].max()) if len(frame) else None


def snapshot_store(store_dir: Path, snap_dir: Path, pairs: list[str]) -> Path:
    """Byte-copy the store files about to be scored; scoring reads FROM here."""
    if snap_dir.exists():
        raise ExecutorError(f"snapshot dir already exists; refusing to overwrite replay evidence: {snap_dir}")
    missing = [pair for pair in pairs if not (store_dir / f"{pair}.parquet").is_file()]
    if missing:
        raise ExecutorError(f"snapshot source is incomplete; missing pairs: {missing}")
    snap_dir.mkdir(parents=True, exist_ok=True)
    for pair in pairs:
        src = store_dir / f"{pair}.parquet"
        shutil.copy2(src, snap_dir / f"{pair}.parquet")
    return snap_dir


def prune_snapshots(root: Path, keep: int) -> None:
    if keep < 0:
        raise ExecutorError("--snapshot-keep-minutes must be >= 0")
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
        self.deferred: dict[str, int] = {}
        self.started_utc = now_utc_iso()
        self.run_id = f"{socket.gethostname()}-{os.getpid()}-{time.time_ns()}"
        self.capture_identity: dict[str, Any] = {}
        self.capture_identity_sha256 = ""
        # Phase-4 hook: the supervisor routes edge-passed signals through the
        # coordinator. Unset (Phase 3) the daemon stays pure proposal-only.
        self.signal_sink: Any = None

    def load_once(self) -> None:
        self.books = load_target_books(self.pairs)
        self.resolutions = resolve_floor_resolutions(self.pairs, self.books, self.logger)
        self.capture_identity = parity_runtime_identity(
            self.books,
            payout_edge_margin=self.args.payout_edge_margin,
            absolute_breakeven_ceiling=self.args.absolute_breakeven_ceiling,
            queue_mode=bool(getattr(self.args, "queue_mode", False)),
            public_endpoint_exact=self.args.url == PUBLIC_WS_URL,
        )
        self.capture_identity_sha256 = _identity_sha256(self.capture_identity)
        self.logger.write("daemon_loaded", pairs=self.pairs,
                          books={p: b.book_id for p, b in self.books.items()})
        self.logger.write(
            "parity_run_started",
            run_id=self.run_id,
            capture_identity_sha256=self.capture_identity_sha256,
            capture_identity=self.capture_identity,
            pairs=self.pairs,
        )

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

    async def score_minute(
        self,
        pair: str,
        bar_epoch: int,
        snap_dir: Path,
        *,
        observed_utc: datetime | None = None,
        snapshot_sha256_value: str | None = None,
    ) -> bool:
        t_detect = time.monotonic()
        observed = observed_utc or datetime.now(timezone.utc)
        if observed.tzinfo is None:
            observed = observed.replace(tzinfo=timezone.utc)
        observed = observed.astimezone(timezone.utc)
        bar_close_utc = datetime.fromtimestamp(bar_epoch + 60, timezone.utc)
        run_id = getattr(self, "run_id", "smoke-run")
        identity_sha = getattr(self, "capture_identity_sha256", "smoke-identity")
        snapshot_digest = snapshot_sha256_value
        if snapshot_digest is None:
            snapshot_digest = snapshot_tree_sha256(snap_dir) if snap_dir.is_dir() else "unavailable"
        attempt_id = parity_attempt_id(run_id, snap_dir.name, pair, bar_epoch)
        correlation = {
            "run_id": run_id,
            "capture_identity_sha256": identity_sha,
            "attempt_id": attempt_id,
            "snapshot_id": snap_dir.name,
            "snapshot": str(snap_dir),
            "snapshot_sha256": snapshot_digest,
            "pair": pair,
            "offset_id": 0,
            "detected_bar_epoch": bar_epoch,
            "detected_bar_close_utc": bar_close_utc.isoformat(),
        }
        self.logger.write("candidate_bar_closed", pair=pair, bar_epoch=bar_epoch,
                          bar_close_utc=bar_close_utc.isoformat())
        try:
            tup = await asyncio.to_thread(
                score_from_snapshot, snap_dir, pair, self.books[pair], self.resolutions[pair],
                self.args.payout_edge_margin, stale_seconds=self.args.store_max_stale_seconds)
        except (LiveFeatureError, ExecutorError, DerivAPIError, Exception) as exc:
            self.skips["feature_row_failed"] = self.skips.get("feature_row_failed", 0) + 1
            self.logger.write(
                "parity_attempt_failed", attempt_status="feature_row_failed",
                error=str(exc)[:300], **correlation,
            )
            self.logger.write("signal_skipped", pair=pair, reason="feature_row_failed", error=str(exc)[:300])
            return True
        score_done_ms = (time.monotonic() - t_detect) * 1000
        expected_close = bar_close_utc.isoformat()
        actual_close = str(tup["candidate_bar_close_utc"])
        if actual_close != expected_close:
            self.deferred["feature_row_lagged"] = self.deferred.get("feature_row_lagged", 0) + 1
            self.logger.write("parity_tuple", **{**correlation, **tup, "attempt_status": "lagged"})
            self.logger.write("signal_deferred", pair=pair, reason="feature_row_lagged",
                              bar_epoch=bar_epoch,
                              expected_candidate_bar_close_utc=expected_close,
                              actual_candidate_bar_close_utc=actual_close,
                              feature_sha256=tup.get("feature_sha256"),
                              gate_passed=tup.get("gate_passed"),
                              selected_side=tup.get("selected_side"),
                              confidence=tup.get("confidence"),
                              threshold=tup.get("threshold"),
                              snapshot=str(snap_dir))
            return False
        self.score_ms.append(score_done_ms)
        self.scored += 1
        self.logger.write("book_scored", pair=pair, score_ms=round(score_done_ms, 1))
        self.logger.write("parity_tuple", **{**correlation, **tup, "attempt_status": "accepted"})
        if not is_ny_session(observed):
            self.skips["outside_ny_session"] = self.skips.get("outside_ny_session", 0) + 1
            self.logger.write("signal_skipped", pair=pair, reason="outside_ny_session")
            return True
        if not tup["gate_passed"]:
            self.skips["book_gate"] = self.skips.get("book_gate", 0) + 1
            self.logger.write("signal_skipped", pair=pair, reason="book_gate", gate_reasons=tup["gate_reasons"])
            return True
        signal = {
            "pair": pair, "side": tup["selected_side"], "offset_id": 0,
            "signal_close_utc": tup["candidate_bar_close_utc"],
            "book_id": self.books[pair].book_id,
            "effective_floor": tup["effective_floor"], "lane": "wall",
            "bar_age_s": round((observed - bar_close_utc).total_seconds(), 1),
            "is_latest_bar": True,
        }
        if getattr(self.args, "queue_mode", False):
            self.logger.write("queue_candidate_ready", pair=pair, side=tup["selected_side"],
                              effective_floor=tup["effective_floor"], mode="queued_to_executor")
            if self.signal_sink is not None:
                await self.signal_sink(signal)
            else:
                self.skips["queue_sink_missing"] = self.skips.get("queue_sink_missing", 0) + 1
                self.logger.write("signal_skipped", pair=pair, reason="queue_sink_missing")
            return True
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
            return True
        prop_ms = (time.monotonic() - t_detect) * 1000
        self.proposal_ms.append(prop_ms)
        prop = resp.get("proposal") or {}
        decision = evaluate_payout_gate(
            prop,
            effective_floor=tup["effective_floor"],
            absolute_breakeven_ceiling=self.args.absolute_breakeven_ceiling,
        )
        ask = decision.ask
        payout = decision.payout
        live_breakeven = decision.live_breakeven
        quote_input = {
            "ask_price_present": "ask_price" in prop,
            "ask_price": prop.get("ask_price"),
            "payout_present": "payout" in prop,
            "payout": prop.get("payout"),
        }
        self.logger.write(
            "parity_quote", **correlation,
            candidate_bar_close_utc=tup["candidate_bar_close_utc"],
            selected_side=tup["selected_side"],
            quote_gate_input=quote_input,
            ask=ask,
            payout=payout,
            live_breakeven=live_breakeven,
            effective_floor=tup["effective_floor"],
            absolute_breakeven_ceiling=self.args.absolute_breakeven_ceiling,
            payout_gate_passed=decision.passed,
            payout_gate_reason=decision.reason,
        )
        self.logger.write(
            "quote_state_updated", side=tup["selected_side"], ask=ask,
            payout=payout, live_breakeven=live_breakeven, proposal_ms=round(prop_ms, 1),
            candidate_bar_close_utc=tup["candidate_bar_close_utc"],
            effective_floor=tup["effective_floor"],
            absolute_breakeven_ceiling=self.args.absolute_breakeven_ceiling,
            payout_gate_passed=decision.passed, payout_gate_reason=decision.reason,
            **correlation,
        )
        if decision.reason in {"missing_or_invalid_ask", "missing_or_invalid_payout"}:
            self.skips[decision.reason] = self.skips.get(decision.reason, 0) + 1
            self.logger.write("signal_skipped", pair=pair, reason=decision.reason)
        elif decision.reason == "edge_not_positive":
            self.skips["edge_not_positive"] = self.skips.get("edge_not_positive", 0) + 1
            self.logger.write("signal_skipped", pair=pair, reason="edge_not_positive",
                              effective_floor=tup["effective_floor"], live_breakeven=live_breakeven)
        elif decision.reason == "breakeven_too_high":
            self.skips["breakeven_too_high"] = self.skips.get("breakeven_too_high", 0) + 1
            self.logger.write("signal_skipped", pair=pair, reason="breakeven_too_high",
                              live_breakeven=live_breakeven,
                              absolute_breakeven_ceiling=self.args.absolute_breakeven_ceiling)
        else:
            self.logger.write("edge_gate_passed", pair=pair, side=tup["selected_side"],
                              effective_floor=tup["effective_floor"], live_breakeven=live_breakeven,
                              net_edge=round(decision.net_edge, 6),
                              mode="proposal_only_phase3" if self.signal_sink is None else "routed_to_coordinator")
            if self.signal_sink is not None:
                await self.signal_sink(signal)
        return True

    async def run(self) -> int:
        self.load_once()
        await self.wait_for_store()  # idle (not crash) while the market is closed / store warming
        if not getattr(self.args, "queue_mode", False):
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
                snap_name = f"{minute_tag}_{time.time_ns()}"
                snap_dir = await asyncio.to_thread(
                    snapshot_store, self.store_dir, self.snap_root / snap_name, list(PAIRS))
                snapshot_digest = await asyncio.to_thread(snapshot_tree_sha256, snap_dir)
                if getattr(self.args, "queue_mode", False):
                    sem = asyncio.Semaphore(max(1, int(getattr(self.args, "producer_queue_mode_fanout_limit", 6))))

                    async def score_one(p: str, e: int) -> tuple[str, int, bool]:
                        async with sem:
                            return p, e, await self.score_minute(
                                p, e, snap_dir, snapshot_sha256_value=snapshot_digest,
                            )

                    results = await asyncio.gather(*(score_one(pair, epoch) for pair, epoch in fresh))
                    for pair, epoch, completed in results:
                        if completed:
                            self.last_scored_epoch[pair] = epoch
                else:
                    for pair, epoch in fresh:
                        if await self.score_minute(
                            pair, epoch, snap_dir, snapshot_sha256_value=snapshot_digest,
                        ):
                            self.last_scored_epoch[pair] = epoch
                await asyncio.to_thread(prune_snapshots, self.snap_root, self.args.snapshot_keep_minutes)
            if self.args.once and self.scored + sum(self.skips.values()) >= len(self.pairs):
                break
            if once_deadline is not None and time.monotonic() > once_deadline:
                self.logger.write("once_timeout", scored=self.scored, skips=self.skips, deferred=self.deferred)
                print(
                    f"ERROR: --once timed out; scored={self.scored} skips={self.skips} deferred={self.deferred}",
                    file=sys.stderr,
                )
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
            "deferred": self.deferred,
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


class _MemoryLogger:
    def __init__(self, path: Path):
        self.path = path
        self.rows: list[dict[str, Any]] = []

    def write(self, event: str, **fields: Any) -> None:
        self.rows.append({"event": event, **fields})

    def last(self, event: str) -> dict[str, Any] | None:
        return next((row for row in reversed(self.rows) if row.get("event") == event), None)


class _ReplayProposalClient:
    """Synchronous proposal adapter; every authority-bearing method is fatal."""

    def __init__(self, proposal: dict[str, Any]):
        self.proposal_payload = dict(proposal)
        self.proposal_calls = 0
        self.forbidden_calls: list[str] = []

    def proposal(self, **_: Any) -> dict[str, Any]:
        self.proposal_calls += 1
        return {"proposal": dict(self.proposal_payload)}

    def _forbidden(self, name: str) -> None:
        self.forbidden_calls.append(name)
        raise ExecutorError(f"parity replay attempted forbidden {name}")

    def request(self, *_: Any, **__: Any) -> dict[str, Any]:
        self._forbidden("network_request")

    def ticks_history(self, *_: Any, **__: Any) -> dict[str, Any]:
        self._forbidden("network_ticks_history")

    def buy(self, *_: Any, **__: Any) -> dict[str, Any]:
        self._forbidden("buy")


class _RecordingBuilder:
    def __init__(self, inner: LiveFeatureBuilder):
        self.inner = inner
        self.last_row: Any = None

    def feature_row(self, pair: str, expected_cols: list[str]) -> Any:
        self.last_row = self.inner.feature_row(pair, expected_cols)
        return self.last_row


class _RecordingBook:
    def __init__(self, inner: LoadedBook):
        self.inner = inner
        self.last_score: Any = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)

    def score(self, row: pd.Series) -> Any:
        self.last_score = self.inner.score(row)
        return self.last_score


def _proposal_from_quote(row: dict[str, Any]) -> dict[str, Any]:
    raw = row.get("quote_gate_input") or {}
    prop: dict[str, Any] = {}
    if raw.get("ask_price_present"):
        prop["ask_price"] = raw.get("ask_price")
    if raw.get("payout_present"):
        prop["payout"] = raw.get("payout")
    return prop


def _one_shot_reference(
    snap: Path,
    pair: str,
    book: LoadedBook,
    resolution: Any,
    args: argparse.Namespace,
    proposal: dict[str, Any],
    scratch: Path,
) -> tuple[dict[str, Any], dict[str, Any] | None, list[str]]:
    """Exercise process_pair itself with snapshot data and no external authority."""
    from deriv_demo_executor import empty_state, process_pair

    client = _ReplayProposalClient(proposal)
    builder = _RecordingBuilder(
        LiveFeatureBuilder(client, store_dir=snap, stale_seconds=10**9)  # type: ignore[arg-type]
    )
    wrapped_book = _RecordingBook(book)
    logger = _MemoryLogger(scratch / "one_shot.jsonl")
    ref_args = SimpleNamespace(
        max_trades_day=0,
        max_open=0,
        pair_cooldown_seconds=0,
        payout_edge_margin=args.payout_edge_margin,
        stake=args.stake,
        absolute_breakeven_ceiling=args.absolute_breakeven_ceiling,
        max_retries=0,
        retry_base_s=0.0,
        dry_run_proposal_only=True,
        demo_buy=False,
    )
    # A fixed known weekday opens only the reference wrapper's session gate.
    # The archived snapshot remains the sole feature/score input.
    pinned = datetime(2026, 6, 15, 12, 0, tzinfo=NY_TZ).astimezone(timezone.utc)
    process_pair(
        pair,
        wrapped_book,  # type: ignore[arg-type]
        resolution,
        client,  # type: ignore[arg-type]
        builder,  # type: ignore[arg-type]
        logger,  # type: ignore[arg-type]
        ref_args,
        scratch / "KILL",
        empty_state(),
        scratch / "state.json",
        trades_today=0,
        open_count=0,
        now_utc=pinned,
    )
    book_score = logger.last("book_score")
    if book_score is None:
        raise ExecutorError(f"{pair}: one-shot process_pair produced no book_score")
    reference = {
        "pair": pair,
        "candidate_bar_close_utc": book_score["candidate_bar_close_utc"],
        "feature_sha256": book_score["feature_sha256"],
        "proba": round(float(book_score["proba"]), 12),
        "selected_side": book_score["direction"],
        "effective_floor": book_score["effective_floor"],
        "gate_passed": book_score["gate_passed"],
        "gate_reasons": book_score["gate_reasons"],
    }
    return reference, logger.last("payout_gate_decision"), client.forbidden_calls


def _parse_close(value: Any) -> datetime:
    if not isinstance(value, str):
        raise ValueError("timestamp is not a string")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("timestamp is timezone-naive")
    parsed = parsed.astimezone(timezone.utc)
    if parsed.second or parsed.microsecond:
        raise ValueError("timestamp is not minute-aligned")
    return parsed


def required_ny_session_keys(session_date: date) -> set[tuple[str, str]]:
    start = datetime.combine(session_date, datetime_time(8, 0), tzinfo=NY_TZ)
    closes = [(start + timedelta(minutes=i)).astimezone(timezone.utc).isoformat() for i in range(540)]
    return {(pair, close) for pair in ENABLED_PAIRS for close in closes}


def validate_full_ny_session(rows: list[dict[str, Any]], session_text: str) -> dict[str, Any]:
    errors: list[str] = []
    try:
        session_date = date.fromisoformat(session_text)
    except ValueError:
        session_date = date.min
        errors.append("invalid_session_date")
    if session_date.weekday() >= 5:
        errors.append("session_date_is_weekend")
    expected = required_ny_session_keys(session_date) if session_date != date.min else set()
    accepted: list[tuple[str, str]] = []
    seen_pairs: set[str] = set()
    malformed: list[dict[str, Any]] = []
    wrong_dates: list[dict[str, Any]] = []
    for row in rows:
        pair = row.get("pair")
        if pair not in ENABLED_PAIRS:
            errors.append(f"unexpected_pair:{pair}")
            continue
        seen_pairs.add(str(pair))
        try:
            close = _parse_close(row.get("candidate_bar_close_utc"))
        except (TypeError, ValueError) as exc:
            malformed.append({"attempt_id": row.get("attempt_id"), "error": str(exc)})
            continue
        local = close.astimezone(NY_TZ)
        if local.date() != session_date:
            wrong_dates.append({"attempt_id": row.get("attempt_id"), "close": close.isoformat()})
            continue
        if row.get("attempt_status") == "accepted" and datetime_time(8, 0) <= local.time() < datetime_time(17, 0):
            accepted.append((str(pair), close.isoformat()))
        elif row.get("attempt_status") not in {"accepted", "lagged"}:
            errors.append(f"invalid_attempt_status:{row.get('attempt_status')}")
    counts = Counter(accepted)
    present = set(counts)
    missing = sorted(expected - present)
    duplicates = [
        {"pair": pair, "candidate_bar_close_utc": close, "count": count}
        for (pair, close), count in sorted(counts.items()) if count != 1
    ]
    if seen_pairs != set(ENABLED_PAIRS):
        errors.append("pair_set_mismatch")
    if malformed:
        errors.append("malformed_candidate_time")
    if wrong_dates:
        errors.append("wrong_ny_date")
    bounds_date = session_date if session_date != date.min else date(2000, 1, 3)
    local_start = datetime.combine(bounds_date, datetime_time(8, 0), tzinfo=NY_TZ)
    local_end = datetime.combine(bounds_date, datetime_time(16, 59), tzinfo=NY_TZ)
    passed = not errors and not missing and not duplicates and len(present) == len(expected) == 3240
    return {
        "date_ny": session_text,
        "bounds_local": [local_start.isoformat(), local_end.isoformat()],
        "bounds_utc": [local_start.astimezone(timezone.utc).isoformat(), local_end.astimezone(timezone.utc).isoformat()],
        "pairs": list(ENABLED_PAIRS),
        "expected_keys": len(expected),
        "present_keys": len(present),
        "missing_keys": [{"pair": pair, "candidate_bar_close_utc": close} for pair, close in missing],
        "duplicate_keys": duplicates,
        "malformed_rows": malformed,
        "wrong_date_rows": wrong_dates,
        "errors": sorted(set(errors)),
        "passed": passed,
    }


def _read_jsonl_strict(path: Path) -> tuple[list[dict[str, Any]], list[str], str]:
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for lineno, line in enumerate(raw.splitlines(), start=1):
        try:
            row = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            errors.append(f"line_{lineno}:malformed_json:{exc}")
            continue
        if not isinstance(row, dict):
            errors.append(f"line_{lineno}:not_an_object")
            continue
        rows.append(row)
    return rows, errors, digest


async def _trade_wrapper_replay(
    quotes: list[dict[str, Any]],
    ceiling: float,
    scratch: Path,
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """Exercise active evaluate_candidate against an isolated temporary queue."""
    import deriv_trade_executor as trade_executor_module
    from deriv_trade_executor import TradeExecutor
    from deriv_trade_queue import claim_batch, connect_queue, enqueue_signal

    class ReplayTradeExecutor(TradeExecutor):
        def __init__(self, namespace: argparse.Namespace, memory_logger: _MemoryLogger):
            super().__init__(namespace, memory_logger)  # type: ignore[arg-type]
            self.proposals: dict[str, dict[str, Any]] = {}
            self.forbidden: list[str] = []

        def pre_money_gate(self, claim: Any) -> str | None:
            return None

        async def get_proposal(self, payload: dict[str, Any]) -> tuple[dict[str, Any], str]:
            attempt_id = str(payload["parity_attempt_id"])
            return {"proposal": dict(self.proposals[attempt_id])}, "parity_recorded"

        async def fresh_demo_ws_url(self) -> str:
            self.forbidden.append("credential_or_account")
            raise ExecutorError("parity replay attempted credential/account access")

        async def connect_async_client(self) -> None:
            self.forbidden.append("network_connect")
            raise ExecutorError("parity replay attempted network access")

        async def proposal_async(self, *_: Any, **__: Any) -> tuple[dict[str, Any], str]:
            self.forbidden.append("network_proposal_async")
            raise ExecutorError("parity replay attempted network access")

        def proposal_sync(self, *_: Any, **__: Any) -> tuple[dict[str, Any], str]:
            self.forbidden.append("network_proposal_sync")
            raise ExecutorError("parity replay attempted network access")

        async def buy_async(self, *_: Any, **__: Any) -> dict[str, Any]:
            self.forbidden.append("buy_async")
            raise ExecutorError("parity replay attempted buy")

        def buy_sync(self, *_: Any, **__: Any) -> dict[str, Any]:
            self.forbidden.append("buy_sync")
            raise ExecutorError("parity replay attempted buy")

    db = scratch / "parity_trade_queue.sqlite"
    logger = _MemoryLogger(scratch / "trade.jsonl")
    namespace = argparse.Namespace(
        queue_db=str(db),
        queue_wakeup_socket=str(scratch / "wakeup.sock"),
        sync_fallback_max_concurrency=1,
        absolute_breakeven_ceiling=ceiling,
        disabled_pair_sides_set=set(),
    )
    executor = ReplayTradeExecutor(namespace, logger)
    conn = connect_queue(db)
    try:
        for quote in quotes:
            attempt_id = str(quote["attempt_id"])
            executor.proposals[attempt_id] = _proposal_from_quote(quote)
            enqueue_signal(conn, {
                "pair": quote["pair"],
                "side": quote["selected_side"],
                "lane": "wall",
                "offset_id": 0,
                "signal_close_utc": quote["candidate_bar_close_utc"],
                "book_id": "parity-replay",
                "effective_floor": quote["effective_floor"],
                "bar_age_s": 0.0,
                "is_latest_bar": True,
                "parity_attempt_id": attempt_id,
            })
        claims = claim_batch(conn, owner="parity-replay", limit=max(1, len(quotes)))
    finally:
        conn.close()
    production_connect = trade_executor_module.connect_queue

    def guarded_connect(path: Path | str) -> Any:
        if Path(path).resolve() != db.resolve():
            executor.forbidden.append(f"persistent_queue:{path}")
            raise ExecutorError("parity replay attempted a non-temporary queue")
        return production_connect(path)

    trade_executor_module.connect_queue = guarded_connect
    try:
        for claim in claims:
            await executor.evaluate_candidate(claim, "parity-replay")
    finally:
        trade_executor_module.connect_queue = production_connect
    decisions = {
        str(row["parity_attempt_id"]): row
        for row in logger.rows
        if row.get("event") == "payout_gate_decision"
        and row.get("wrapper") == "deriv_trade_executor.TradeExecutor.evaluate_candidate"
    }
    return decisions, executor.forbidden


def _component_pass(logged: int, checked: int, divergences: list[Any], *, pruned: int = 0) -> bool:
    return logged == checked and checked > 0 and pruned == 0 and not divergences


def _has_single_event_value(rows: list[dict[str, Any]], field: str) -> bool:
    if not rows or any(not isinstance(row.get(field), str) or not row.get(field) for row in rows):
        return False
    values = {row[field] for row in rows}
    return len(values) == 1


def _same_unique_attempt_ids(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> bool:
    left_ids = [row.get("attempt_id") for row in left]
    right_ids = [row.get("attempt_id") for row in right]
    return (
        all(isinstance(value, str) and value for value in [*left_ids, *right_ids])
        and len(left_ids) == len(set(left_ids))
        and len(right_ids) == len(set(right_ids))
        and set(left_ids) == set(right_ids)
    )


def _payout_decisions_match(
    expected: dict[str, Any],
    one_shot: dict[str, Any],
    trade_executor: dict[str, Any],
) -> bool:
    return one_shot == expected and trade_executor == expected


def parity_replay(daemon_log: Path, args: argparse.Namespace) -> int:
    """Fail-closed KILL #0 replay across hot, one-shot, and production wrappers."""
    rows, parse_errors, source_sha_before = _read_jsonl_strict(daemon_log)
    tuples = [row for row in rows if row.get("event") == "parity_tuple"]
    quotes = [row for row in rows if row.get("event") == "parity_quote"]
    quote_states = [row for row in rows if row.get("event") == "quote_state_updated"]
    failed_attempts = [row for row in rows if row.get("event") == "parity_attempt_failed"]
    starts = [row for row in rows if row.get("event") == "parity_run_started"]
    correlation_errors: list[str] = list(parse_errors)
    forbidden_capture_events = sorted({
        str(row.get("event")) for row in rows
        if str(row.get("event", "")).startswith("buy_")
        or row.get("event") in {
            "auth_health", "contract_update", "contract_closed", "executor_startup",
            "queue_candidate_ready",
        }
    })
    if forbidden_capture_events:
        correlation_errors.append("authority_bearing_capture_event")

    tuple_ids = [str(row.get("attempt_id")) for row in tuples]
    quote_ids = [str(row.get("attempt_id")) for row in quotes]
    if len(tuple_ids) != len(set(tuple_ids)):
        correlation_errors.append("duplicate_tuple_attempt_id")
    if len(quote_ids) != len(set(quote_ids)):
        correlation_errors.append("duplicate_quote_attempt_id")
    if not _same_unique_attempt_ids(quotes, quote_states):
        correlation_errors.append("parity_quote_state_attempt_mismatch")
    tuple_by_id = {str(row.get("attempt_id")): row for row in tuples}
    for row in tuples:
        attempt_id = str(row.get("attempt_id"))
        try:
            bar_epoch = int(row["detected_bar_epoch"])
            expected_close = datetime.fromtimestamp(bar_epoch + 60, timezone.utc).isoformat()
            expected_attempt = parity_attempt_id(
                str(row["run_id"]), str(row["snapshot_id"]), str(row["pair"]), bar_epoch,
            )
        except (KeyError, TypeError, ValueError, OSError) as exc:
            correlation_errors.append(f"tuple_correlation_malformed:{attempt_id}:{exc}")
            continue
        if row.get("attempt_id") != expected_attempt:
            correlation_errors.append(f"attempt_id_mismatch:{attempt_id}")
        if Path(str(row.get("snapshot"))).name != row.get("snapshot_id"):
            correlation_errors.append(f"snapshot_id_mismatch:{attempt_id}")
        if row.get("detected_bar_close_utc") != expected_close:
            correlation_errors.append(f"detected_close_mismatch:{attempt_id}")
        aligned = row.get("candidate_bar_close_utc") == expected_close
        if (row.get("attempt_status") == "accepted") != aligned:
            correlation_errors.append(f"attempt_status_mismatch:{attempt_id}")
        if row.get("offset_id") != 0:
            correlation_errors.append(f"offset_id_mismatch:{attempt_id}")
    for quote in quotes:
        attempt_id = str(quote.get("attempt_id"))
        owner = tuple_by_id.get(attempt_id)
        if owner is None:
            correlation_errors.append(f"quote_without_tuple:{attempt_id}")
            continue
        for field in (
            "run_id", "capture_identity_sha256", "snapshot_id", "snapshot",
            "snapshot_sha256", "pair", "detected_bar_close_utc", "candidate_bar_close_utc",
            "selected_side", "effective_floor",
        ):
            if quote.get(field) != owner.get(field):
                correlation_errors.append(f"quote_correlation_mismatch:{attempt_id}:{field}")
    quote_state_by_id = {str(row.get("attempt_id")): row for row in quote_states}
    for quote in quotes:
        attempt_id = str(quote.get("attempt_id"))
        diagnostic = quote_state_by_id.get(attempt_id)
        if diagnostic is None:
            continue
        for field in (
            "run_id", "capture_identity_sha256", "snapshot_id", "snapshot", "snapshot_sha256",
            "pair", "detected_bar_close_utc", "candidate_bar_close_utc", "effective_floor",
            "absolute_breakeven_ceiling", "ask", "payout", "live_breakeven",
            "payout_gate_passed", "payout_gate_reason",
        ):
            if quote.get(field) != diagnostic.get(field):
                correlation_errors.append(f"quote_state_correlation_mismatch:{attempt_id}:{field}")
        if quote.get("selected_side") != diagnostic.get("side"):
            correlation_errors.append(f"quote_state_correlation_mismatch:{attempt_id}:side")

    run_ids = {
        str(row.get("run_id")) for row in [*tuples, *quotes, *failed_attempts]
        if row.get("run_id") is not None
    }
    if len(starts) != 1:
        correlation_errors.append("expected_exactly_one_parity_run_started")
    start = starts[0] if len(starts) == 1 else {}
    if not _has_single_event_value([*tuples, *quotes, *failed_attempts], "run_id") or (
        run_ids and start.get("run_id") not in run_ids
    ):
        correlation_errors.append("mixed_run_identity")
    identity_shas = {
        str(row.get("capture_identity_sha256")) for row in [*tuples, *quotes, *failed_attempts]
        if row.get("capture_identity_sha256") is not None
    }
    if not _has_single_event_value([*tuples, *quotes, *failed_attempts], "capture_identity_sha256") or (
        identity_shas and start.get("capture_identity_sha256") not in identity_shas
    ):
        correlation_errors.append("mixed_capture_identity")

    expected_pairs = list(ENABLED_PAIRS)
    event_pairs = {row.get("pair") for row in tuples}
    invalid_pairs = sorted(str(pair) for pair in event_pairs if pair not in ENABLED_PAIRS)
    if invalid_pairs:
        correlation_errors.append(f"invalid_pairs:{invalid_pairs}")

    books: dict[str, LoadedBook] = {}
    resolutions: dict[str, Any] = {}
    replay_identity: dict[str, Any] = {}
    if tuples and not invalid_pairs:
        books = load_target_books(expected_pairs)
        memory_logger = _MemoryLogger(Path("parity-floor-resolution.jsonl"))
        resolutions = resolve_floor_resolutions(expected_pairs, books, memory_logger)  # type: ignore[arg-type]
        replay_identity = parity_runtime_identity(
            books,
            payout_edge_margin=args.payout_edge_margin,
            absolute_breakeven_ceiling=args.absolute_breakeven_ceiling,
            queue_mode=bool(getattr(args, "queue_mode", False)),
            public_endpoint_exact=args.url == PUBLIC_WS_URL,
        )
    capture_identity = start.get("capture_identity") if isinstance(start.get("capture_identity"), dict) else {}
    capture_identity_sha = _identity_sha256(capture_identity) if capture_identity else None
    replay_identity_sha = _identity_sha256(replay_identity) if replay_identity else None
    identity_errors: list[str] = []
    if not capture_identity:
        identity_errors.append("capture_identity_missing")
    if capture_identity_sha != start.get("capture_identity_sha256"):
        identity_errors.append("capture_identity_hash_mismatch")
    if capture_identity != replay_identity:
        identity_errors.append("capture_replay_identity_drift")
    if capture_identity.get("tracked_worktree_clean") is not True:
        identity_errors.append("capture_tracked_worktree_not_clean")
    if capture_identity.get("repo_head") != capture_identity.get("origin_main"):
        identity_errors.append("capture_head_not_origin_main")
    if start.get("pairs") != expected_pairs:
        identity_errors.append("capture_pair_list_mismatch")
    capture_config = capture_identity.get("config") or {}
    if capture_config.get("queue_mode") is not False:
        identity_errors.append("capture_queue_mode_not_disabled")
    if capture_config.get("public_endpoint_exact") is not True:
        identity_errors.append("capture_endpoint_not_public_default")
    if any((capture_config.get("credential_environment_present") or {}).values()):
        identity_errors.append("capture_credential_environment_present")
    capture_ceiling = (capture_identity.get("config") or {}).get("absolute_breakeven_ceiling")
    if any(quote.get("absolute_breakeven_ceiling") != capture_ceiling for quote in quotes):
        identity_errors.append("quote_ceiling_mismatch")

    deterministic_divergences: list[dict[str, Any]] = []
    one_shot_decisions: dict[str, dict[str, Any]] = {}
    one_shot_forbidden_calls: list[str] = []
    tuples_checked = 0
    tuples_pruned = 0
    snapshot_digests: dict[str, str] = {}
    tuple_keys = (
        "pair", "candidate_bar_close_utc", "feature_sha256", "proba",
        "selected_side", "effective_floor", "gate_passed", "gate_reasons",
    )
    quote_by_id = {str(row.get("attempt_id")): row for row in quotes}
    with tempfile.TemporaryDirectory(prefix="deriv_parity_replay_") as scratch_text:
        scratch = Path(scratch_text)
        for row in tuples:
            attempt_id = str(row.get("attempt_id"))
            pair = str(row.get("pair"))
            snap = Path(str(row.get("snapshot")))
            if not snap.is_dir():
                tuples_pruned += 1
                deterministic_divergences.append({"attempt_id": attempt_id, "reason": "snapshot_pruned"})
                continue
            if str(snap) not in snapshot_digests:
                snapshot_digests[str(snap)] = snapshot_tree_sha256(snap)
            actual_digest = snapshot_digests[str(snap)]
            if actual_digest != row.get("snapshot_sha256"):
                deterministic_divergences.append({
                    "attempt_id": attempt_id,
                    "reason": "snapshot_identity_drift",
                    "logged": row.get("snapshot_sha256"),
                    "replay": actual_digest,
                })
                continue
            if pair not in books:
                deterministic_divergences.append({"attempt_id": attempt_id, "reason": "book_missing"})
                continue
            proposal = _proposal_from_quote(quote_by_id[attempt_id]) if attempt_id in quote_by_id else {
                "ask_price": 1.0, "payout": 2.0,
            }
            try:
                hot_replay = score_from_snapshot(
                    snap, pair, books[pair], resolutions[pair],
                    args.payout_edge_margin, stale_seconds=10**9,
                )
                one_shot, one_shot_decision, forbidden = _one_shot_reference(
                    snap, pair, books[pair], resolutions[pair], args, proposal, scratch,
                )
            except Exception as exc:
                if "forbidden" in str(exc).lower():
                    one_shot_forbidden_calls.append(str(exc)[:300])
                deterministic_divergences.append({
                    "attempt_id": attempt_id, "reason": "replay_error", "error": str(exc)[:300],
                })
                continue
            if forbidden:
                one_shot_forbidden_calls.extend(forbidden)
                deterministic_divergences.append({
                    "attempt_id": attempt_id, "reason": "one_shot_forbidden_call", "calls": forbidden,
                })
                continue
            tuples_checked += 1
            diff: dict[str, Any] = {}
            for key in tuple_keys:
                values = {"captured": row.get(key), "hot_replay": hot_replay.get(key), "one_shot": one_shot.get(key)}
                if len({json.dumps(value, sort_keys=True, default=str) for value in values.values()}) != 1:
                    diff[key] = values
            if diff:
                deterministic_divergences.append({"attempt_id": attempt_id, "diff": diff})
            if one_shot_decision is not None:
                one_shot_decisions[attempt_id] = one_shot_decision

        trade_decisions: dict[str, dict[str, Any]] = {}
        trade_forbidden: list[str] = []
        if quotes:
            try:
                trade_decisions, trade_forbidden = asyncio.run(
                    _trade_wrapper_replay(quotes, args.absolute_breakeven_ceiling, scratch)
                )
            except Exception as exc:
                correlation_errors.append(f"trade_wrapper_replay_error:{str(exc)[:300]}")

    quote_divergences: list[dict[str, Any]] = []
    quotes_checked = 0
    if trade_forbidden:
        quote_divergences.append({"reason": "trade_wrapper_forbidden_call", "calls": trade_forbidden})
    for quote in quotes:
        attempt_id = str(quote.get("attempt_id"))
        one = one_shot_decisions.get(attempt_id)
        trade = trade_decisions.get(attempt_id)
        if one is None or trade is None:
            quote_divergences.append({"attempt_id": attempt_id, "reason": "wrapper_decision_missing"})
            continue
        quotes_checked += 1
        expected = {
            "passed": quote.get("payout_gate_passed"),
            "reason": quote.get("payout_gate_reason"),
        }
        observed = {
            "one_shot": {"passed": one.get("passed"), "reason": one.get("reason")},
            "trade_executor": {"passed": trade.get("passed"), "reason": trade.get("reason")},
        }
        if not _payout_decisions_match(expected, observed["one_shot"], observed["trade_executor"]):
            quote_divergences.append({"attempt_id": attempt_id, "expected": expected, **observed})

    deterministic_passed = _component_pass(
        len(tuples), tuples_checked, deterministic_divergences,
        pruned=tuples_pruned,
    ) and not failed_attempts
    payout_passed = _component_pass(len(quotes), quotes_checked, quote_divergences)
    if args.require_full_ny_session is None:
        session = {
            "date_ny": None,
            "expected_keys": 3240,
            "present_keys": 0,
            "errors": ["--require-full-ny-session is required for KILL #0 admission"],
            "passed": False,
        }
    else:
        session = validate_full_ny_session(tuples, args.require_full_ny_session)

    source_sha_after = _file_sha256(daemon_log)
    if source_sha_after != source_sha_before:
        correlation_errors.append("source_log_mutated_during_replay")
    overall = (
        deterministic_passed
        and payout_passed
        and session["passed"]
        and not correlation_errors
        and not identity_errors
    )
    result = {
        "schema": "deriv-kill0-parity/v2",
        "gate": "issue#20 KILL #0 deterministic AND payout-wrapper parity",
        "replayed_utc": now_utc_iso(),
        "daemon_log": {
            "path": str(daemon_log),
            "sha256_before": source_sha_before,
            "sha256_after": source_sha_after,
        },
        "capture_identity": capture_identity,
        "capture_identity_sha256": capture_identity_sha,
        "replay_identity": replay_identity,
        "replay_identity_sha256": replay_identity_sha,
        "identity_errors": identity_errors,
        "correlation_errors": sorted(set(correlation_errors)),
        "authority": {
            "capture_forbidden_events": forbidden_capture_events,
            "one_shot_forbidden_calls": one_shot_forbidden_calls,
            "trade_executor_forbidden_calls": trade_forbidden,
            "ephemeral_trade_queue_removed": True,
            "passed": not forbidden_capture_events and not one_shot_forbidden_calls and not trade_forbidden,
        },
        "session": session,
        "deterministic": {
            "tuples_logged": len(tuples),
            "tuples_checked": tuples_checked,
            "tuples_skipped_pruned": tuples_pruned,
            "failed_attempts": len(failed_attempts),
            "divergences": deterministic_divergences,
            "passed": deterministic_passed,
        },
        "payout": {
            "quotes_logged": len(quotes),
            "quotes_checked": quotes_checked,
            "divergences": quote_divergences,
            "passed": payout_passed,
        },
        "overall_pass": overall,
    }
    tmp = PARITY_RESULT_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    tmp.replace(PARITY_RESULT_PATH)
    print(json.dumps({
        "tuples": [len(tuples), tuples_checked],
        "quotes": [len(quotes), quotes_checked],
        "session_keys": [session.get("present_keys"), session.get("expected_keys")],
        "overall_pass": overall,
    }, indent=1))
    print(f"{'PARITY PASS' if overall else 'PARITY FAIL'} -> {PARITY_RESULT_PATH}")
    return 0 if overall else 1


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

    # 5. xpair/common-row lag: do not enqueue an older feature row for the
    # newly detected bar; leave the epoch unadvanced so the next poll retries.
    original_score_from_snapshot = globals()["score_from_snapshot"]
    emitted: list[dict[str, Any]] = []
    fake_tuples = [
        {
            "pair": "GBPUSD",
            "candidate_bar_close_utc": "2026-07-06T15:59:00+00:00",
            "feature_sha256": "lagged",
            "proba": 0.35,
            "selected_side": "DOWN",
            "effective_floor": 0.6267,
            "gate_passed": True,
            "gate_reasons": [],
            "confidence": 0.15,
            "threshold": 0.13,
            "row_source": "fake",
        },
        {
            "pair": "GBPUSD",
            "candidate_bar_close_utc": "2026-07-06T16:00:00+00:00",
            "feature_sha256": "fresh",
            "proba": 0.35,
            "selected_side": "DOWN",
            "effective_floor": 0.6267,
            "gate_passed": True,
            "gate_reasons": [],
            "confidence": 0.15,
            "threshold": 0.13,
            "row_source": "fake",
        },
        {
            "pair": "GBPUSD",
            "candidate_bar_close_utc": "2026-07-06T16:00:00+00:00",
            "feature_sha256": "fresh-off-session",
            "proba": 0.35,
            "selected_side": "DOWN",
            "effective_floor": 0.6267,
            "gate_passed": True,
            "gate_reasons": [],
            "confidence": 0.15,
            "threshold": 0.13,
            "row_source": "fake",
        },
    ]

    def fake_score_from_snapshot(*_: Any, **__: Any) -> dict[str, Any]:
        return fake_tuples.pop(0)

    async def fake_sink(signal: dict[str, Any]) -> None:
        emitted.append(signal)

    d6 = object.__new__(HotDaemon)
    d6.args = argparse.Namespace(payout_edge_margin=0.005, store_max_stale_seconds=180, queue_mode=True)
    d6.logger = JsonlLogger(tmp / "logs_6")
    d6.books = {"GBPUSD": argparse.Namespace(book_id="GBPUSD.fake.v1")}
    d6.resolutions = {"GBPUSD": object()}
    d6.score_ms = []
    d6.proposal_ms = []
    d6.scored = 0
    d6.skips = {}
    d6.deferred = {}
    d6.signal_sink = fake_sink
    d6.run_id = "smoke-run"
    d6.capture_identity_sha256 = "smoke-identity"
    in_session = datetime(2026, 7, 6, 16, 0, tzinfo=timezone.utc)
    off_session = datetime(2026, 7, 6, 4, 0, tzinfo=timezone.utc)
    globals()["score_from_snapshot"] = fake_score_from_snapshot
    try:
        done1 = await d6.score_minute(
            "GBPUSD", 1783353540, tmp / "snap_lagged",
            observed_utc=in_session, snapshot_sha256_value="lagged-sha",
        )
        emitted_after_first = len(emitted)
        done2 = await d6.score_minute(
            "GBPUSD", 1783353540, tmp / "snap_fresh",
            observed_utc=in_session, snapshot_sha256_value="fresh-sha",
        )
        done3 = await d6.score_minute(
            "GBPUSD", 1783353540, tmp / "snap_off_session",
            observed_utc=off_session, snapshot_sha256_value="off-sha",
        )
    finally:
        globals()["score_from_snapshot"] = original_score_from_snapshot
    ev6 = _smoke_events(tmp / "logs_6")
    check("lagged feature row defers and does not enqueue",
          done1 is False and d6.deferred.get("feature_row_lagged") == 1
          and any(e.get("event") == "signal_deferred" for e in ev6)
          and emitted_after_first == 0)
    check("fresh retry emits exactly one queued signal with matching close",
          done2 is True and len(emitted) == 1
          and emitted[0]["signal_close_utc"] == "2026-07-06T16:00:00+00:00")
    check("pinned off-session clock skips without consulting host time",
          done3 is True and len(emitted) == 1 and d6.skips.get("outside_ny_session") == 1)

    # 6. replay evidence must be immutable; retry snapshots use unique dirs.
    snap_src = tmp / "store"
    snap_src.mkdir()
    pd.DataFrame({"epoch": [1], "open": [1.0], "high": [1.0], "low": [1.0], "close": [1.0]}).to_parquet(snap_src / "GBPUSD.parquet")
    snap_dst = tmp / "snapshot_once"
    snapshot_store(snap_src, snap_dst, ["GBPUSD"])
    overwrite_blocked = False
    try:
        snapshot_store(snap_src, snap_dst, ["GBPUSD"])
    except ExecutorError:
        overwrite_blocked = True
    check("snapshot_store refuses to overwrite replay evidence", overwrite_blocked)
    snapshot_digest_before = snapshot_tree_sha256(snap_dst)
    with (snap_dst / "mutation-sentinel").open("wb") as fh:
        fh.write(b"changed")
    check("snapshot byte drift changes its replay identity",
          snapshot_tree_sha256(snap_dst) != snapshot_digest_before)

    # 7. zero means retain every snapshot; positive N keeps exactly N.
    retain_root = tmp / "retention"
    retain_root.mkdir()
    for idx in range(541):
        (retain_root / f"{idx:04d}").mkdir()
    prune_snapshots(retain_root, 0)
    check("snapshot retention 0 keeps more than a full 540-minute session",
          len(list(retain_root.iterdir())) == 541)
    prune_snapshots(retain_root, 7)
    check("bounded snapshot retention keeps exactly N", len(list(retain_root.iterdir())) == 7)
    negative_rejected = False
    try:
        prune_snapshots(retain_root, -1)
    except ExecutorError:
        negative_rejected = True
    check("negative snapshot retention is rejected", negative_rejected)

    # 8. exact DST-aware full-session grid and its fail-closed mutations.
    session_day = date(2026, 7, 6)
    session_rows = [
        {
            "attempt_id": f"{pair}-{close}",
            "pair": pair,
            "candidate_bar_close_utc": close,
            "attempt_status": "accepted",
        }
        for pair, close in sorted(required_ny_session_keys(session_day))
    ]
    full = validate_full_ny_session(session_rows, session_day.isoformat())
    check("full-session validator accepts the exact 3,240-key grid",
          full["passed"] and full["present_keys"] == 3240)
    winter_keys = required_ny_session_keys(date(2026, 1, 5))
    check("session grid follows New York DST in summer and winter",
          ("USDJPY", "2026-07-06T12:00:00+00:00") in required_ny_session_keys(session_day)
          and ("USDJPY", "2026-01-05T13:00:00+00:00") in winter_keys)
    for label, index in (("first", 0), ("middle", len(session_rows) // 2), ("last", -1)):
        missing = validate_full_ny_session(
            session_rows[:index] + session_rows[index + 1:] if index >= 0 else session_rows[:-1],
            session_day.isoformat(),
        )
        check(f"full-session validator rejects missing {label} key", not missing["passed"])
    without_pair = [row for row in session_rows if row["pair"] != ENABLED_PAIRS[0]]
    check("full-session validator rejects a missing pair",
          not validate_full_ny_session(without_pair, session_day.isoformat())["passed"])
    check("full-session validator rejects a duplicate accepted key",
          not validate_full_ny_session(session_rows + [dict(session_rows[0])], session_day.isoformat())["passed"])
    check("full-session validator rejects the wrong date",
          not validate_full_ny_session(session_rows, "2026-07-07")["passed"])
    weekend_day = date(2026, 7, 11)
    weekend_rows = [
        {"attempt_id": f"w-{pair}-{close}", "pair": pair,
         "candidate_bar_close_utc": close, "attempt_status": "accepted"}
        for pair, close in required_ny_session_keys(weekend_day)
    ]
    check("full-session validator rejects weekends",
          not validate_full_ny_session(weekend_rows, weekend_day.isoformat())["passed"])
    malformed_rows = [dict(row) for row in session_rows]
    malformed_rows[0]["candidate_bar_close_utc"] = "not-a-time"
    check("full-session validator rejects malformed time",
          not validate_full_ny_session(malformed_rows, session_day.isoformat())["passed"])

    # 9. payout boundaries and overall count gates are non-vacuous.
    payout_cases = [
        ({"payout": 2.0}, 0.58, 0.60, "missing_or_invalid_ask"),
        ({"ask_price": "bad", "payout": 2.0}, 0.58, 0.60, "missing_or_invalid_ask"),
        ({"ask_price": 1.0}, 0.58, 0.60, "missing_or_invalid_payout"),
        ({"ask_price": 1.0, "payout": 0.0}, 0.58, 0.60, "missing_or_invalid_payout"),
        ({"ask_price": 0.58, "payout": 1.0}, 0.58, 0.60, "edge_not_positive"),
        ({"ask_price": 0.59, "payout": 1.0}, 0.58, 0.60, "edge_not_positive"),
        ({"ask_price": 0.61, "payout": 1.0}, 0.70, 0.60, "breakeven_too_high"),
        ({"ask_price": 0.50, "payout": 1.0}, 0.58, 0.60, None),
    ]
    decisions_ok = all(
        evaluate_payout_gate(
            prop, effective_floor=floor, absolute_breakeven_ceiling=ceiling,
        ).reason == reason
        for prop, floor, ceiling, reason in payout_cases
    )
    check("payout decision table preserves invalid/equality/edge/ceiling/pass order", decisions_ok)
    check("zero tuples cannot pass", not _component_pass(0, 0, []))
    check("one pruned snapshot cannot pass", not _component_pass(2, 1, [], pruned=1))
    check("one tuple mismatch cannot pass", not _component_pass(1, 1, [{"diff": True}]))
    check("absent quote component cannot pass", not _component_pass(0, 0, []))
    check("one quote mismatch cannot pass",
          not _payout_decisions_match(
              {"passed": True, "reason": None},
              {"passed": True, "reason": None},
              {"passed": False, "reason": "edge_not_positive"},
          ))
    check("mixed run identities are rejected",
          not _has_single_event_value([{"run_id": "a"}, {"run_id": "b"}], "run_id"))
    check("missing identity fields are rejected",
          not _has_single_event_value(
              [{"capture_identity_sha256": "a"}, {"other": "field"}],
              "capture_identity_sha256",
          ))
    check("orphan diagnostic quote events are rejected",
          not _same_unique_attempt_ids(
              [{"attempt_id": "quote-a"}],
              [{"attempt_id": "quote-a"}, {"attempt_id": "orphan"}],
          ))

    malformed_log = tmp / "malformed.jsonl"
    malformed_log.write_text('{"event":"ok"}\nnot-json\n', encoding="utf-8")
    _, strict_errors, original_sha = _read_jsonl_strict(malformed_log)
    malformed_log.write_text('{"event":"ok"}\n', encoding="utf-8")
    check("strict JSONL parsing rejects malformed lines", bool(strict_errors))
    check("source-log mutation changes its replay identity", _file_sha256(malformed_log) != original_sha)

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
    p.add_argument(
        "--require-full-ny-session", default=None, metavar="YYYY-MM-DD",
        help="require the exact six-pair 08:00-16:59 America/New_York grid during parity replay",
    )
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
    p.add_argument("--queue-mode", action="store_true",
                   help="emit selected-side candidates after book gate; executor owns payout/buy gates")
    p.add_argument("--producer-queue-mode-fanout-limit", type=int, default=6)
    p.add_argument("--smoke", action="store_true", help="deterministic in-process idle-wait smoke (no network, no store)")
    p.add_argument("--payout-edge-margin", type=float, default=0.005)
    add_absolute_breakeven_ceiling_arg(p)
    p.add_argument("--stake", type=float, default=1.0)
    p.add_argument("--log-dir", default="logs/deriv_hot_daemon")
    p.add_argument("--url", default=PUBLIC_WS_URL)
    args = normalize_absolute_breakeven_ceiling(p.parse_args())
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
