"""Demo-only executor for the six deployable 15m Deriv FX books."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from book_runtime import TARGET_BOOKS, LoadedBook, book_inventory, load_target_books
from deriv_backfill import DEFAULT_ENABLED_PAIRS, DERIV_DATA_ROOT, store_health
from deriv_client import DerivAPIError, DerivEnv, DerivOptionsClient
from deriv_floor_resolver import (
    FloorResolutionError,
    PairFloorResolution,
    inventory_rows,
    recompute_fallback_max,
    resolve_book,
    resolve_enabled_pairs,
    resolve_haircut,
)
from live_features import (
    DERIV_SYMBOLS,
    FeatureRow,
    LiveFeatureBuilder,
    LiveFeatureError,
    candles_to_m1,
    latest_replay_row,
    pair_symbol,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG_DIR = REPO_ROOT / "logs" / "paper_trades"
HORIZON_MINUTES = 15
ENABLED_PAIRS = list(DEFAULT_ENABLED_PAIRS)
EURUSD_DISABLED_REASON = "EURUSD disabled: verified live OF_* provider is unavailable"
TERMINAL_STATUSES = {"sold", "won", "lost", "expired", "cancelled"}
NY_TZ = ZoneInfo("America/New_York")


class ExecutorError(RuntimeError):
    pass


class JsonlLogger:
    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.log_dir / f"{datetime.now(timezone.utc).date().isoformat()}.jsonl"

    def write(self, event: str, **fields: Any) -> None:
        row = {
            "event": event,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            **fields,
        }
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, sort_keys=True, default=_json_default) + "\n")


class ExecutorLock:
    def __init__(self, log_dir: Path, account_key: str, logger: "JsonlLogger | None" = None):
        digest = hashlib.sha256(account_key.encode("utf-8")).hexdigest()[:12]
        self.path = log_dir / f"executor_{digest}.lock"
        self.fd: int | None = None
        self.logger = logger

    def __enter__(self) -> "ExecutorLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(
            {
                "pid": os.getpid(),
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "path": str(self.path),
            },
            sort_keys=True,
        )
        try:
            self.fd = os.open(str(self.path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError as exc:
            if self.logger is not None:
                self.logger.write("lock_acquire_failed", reason="lock_exists", path=str(self.path))
            raise ExecutorError(f"live executor lock already exists: {self.path}") from exc
        os.write(self.fd, payload.encode("utf-8"))
        os.close(self.fd)
        self.fd = None
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"cannot JSON encode {type(obj).__name__}")


def raw_hash(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:16]


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pairs", default="all-enabled", help="comma-separated pair list or all-enabled")
    p.add_argument("--stake", type=float, default=1.0)
    p.add_argument("--max-trades-day", type=int, default=3)
    p.add_argument("--max-open", type=int, default=1)
    p.add_argument("--demo-buy", action="store_true", help="submit demo-account buys; never enabled by default")
    p.add_argument("--dry-run-proposal-only", action="store_true", help="request proposals but do not buy")
    p.add_argument("--once", action="store_true")
    p.add_argument("--loop", action="store_true")
    p.add_argument("--interval-seconds", type=float, default=60.0)
    p.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    p.add_argument("--state-file", type=Path, default=None)
    p.add_argument("--store-dir", type=Path, default=None, help="rolling deriv_data store for feature rows")
    p.add_argument("--history-minutes", type=int, default=14000)
    p.add_argument("--tick-volume-count", type=int, default=5000)
    p.add_argument(
        "--max-breakeven",
        type=float,
        default=0.60,
        help="secondary absolute breakeven ceiling; the side-floor edge gate is the authoritative guard",
    )
    p.add_argument(
        "--payout-edge-margin",
        type=float,
        default=0.005,
        help="policy margin subtracted from the side floor (effective_floor = side_p10 - haircut_abs - margin)",
    )
    p.add_argument(
        "--quote-snapshots",
        choices=["all", "off"],
        default="all",
        help="all: snapshot CALL+PUT quotes for every enabled pair each cycle (default); off: reduced-cadence override",
    )
    p.add_argument("--max-retries", type=int, default=3)
    p.add_argument("--retry-base-s", type=float, default=1.0)
    p.add_argument("--monitor-timeout-seconds", type=float, default=1200.0)
    p.add_argument("--monitor-interval-seconds", type=float, default=2.0)
    p.add_argument("--store-min-required-rows", type=int, default=14000)
    p.add_argument("--store-max-stale-seconds", type=int, default=180)
    p.add_argument("--check-books", action="store_true")
    p.add_argument("--replay-schema-check", action="store_true")
    p.add_argument("--public-smoke", action="store_true")
    p.add_argument("--auth-smoke", action="store_true")
    p.add_argument("--fake-client-smoke", action="store_true")
    p.add_argument("--reconcile-only", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    logger = JsonlLogger(args.log_dir)

    if args.fake_client_smoke:
        run_fake_client_smoke()
        return 0

    pairs = resolve_pairs(args.pairs)
    snapshots_per_cycle = len(pairs) * 2 if args.quote_snapshots == "all" else 0
    logger.write(
        "startup_config",
        pairs=pairs,
        enabled_pairs=ENABLED_PAIRS,
        stake=args.stake,
        dry_run=args.dry_run_proposal_only,
        demo_buy=args.demo_buy,
        store_dir=args.store_dir,
        store_min_required_rows=args.store_min_required_rows,
        store_max_stale_seconds=args.store_max_stale_seconds,
        side_floor_gating=True,
        payout_edge_margin=args.payout_edge_margin,
        max_breakeven_ceiling=args.max_breakeven,
        quote_snapshot_mode=args.quote_snapshots,
        interval_seconds=args.interval_seconds,
        monitor_interval_seconds=args.monitor_interval_seconds,
        monitor_timeout_seconds=args.monitor_timeout_seconds,
        max_trades_day=args.max_trades_day,
        max_open=args.max_open,
        api_call_budget={
            "quote_snapshots_per_cycle": snapshots_per_cycle,
            "max_fresh_buy_quotes_per_cycle": len(pairs),
            "monitor_poll_interval_seconds": args.monitor_interval_seconds,
            "note": "verify against Deriv per-app/per-connection rate limits before enabling all-enabled snapshots",
        },
    )

    books = load_target_books(pairs)
    for row in book_inventory(pairs):
        logger.write("book_loaded", **row)
    resolutions = resolve_floor_resolutions(pairs, books, logger)

    if args.check_books:
        print(json.dumps(book_inventory(pairs), indent=2))
        print(json.dumps(inventory_rows(pairs), indent=2))
    if args.replay_schema_check:
        run_replay_schema_check(books)
    if args.public_smoke:
        run_public_smoke(pairs, books, logger, args)
    if args.auth_smoke:
        run_auth_smoke()
    if args.check_books or args.replay_schema_check or args.public_smoke or args.auth_smoke:
        return 0

    if args.demo_buy and args.dry_run_proposal_only:
        raise ExecutorError("--demo-buy and --dry-run-proposal-only are mutually exclusive")
    if not args.demo_buy and not args.dry_run_proposal_only:
        args.dry_run_proposal_only = True

    kill_path = args.log_dir / "KILL"
    check_kill_switch(kill_path, logger)

    state_path = args.state_file or args.log_dir / "executor_state.json"
    state = load_state(state_path)
    account_key = DerivEnv.from_env().account_id if args.demo_buy else "public"

    with ExecutorLock(args.log_dir, account_key, logger):
        client = DerivOptionsClient.demo_from_env() if args.demo_buy else DerivOptionsClient.public()
        if args.demo_buy and not client.is_demo:
            raise ExecutorError("refusing buy: authenticated WebSocket URL is not demo")
        with client:
            verify_contracts(client, pairs, logger, args)
            if args.demo_buy:
                reconcile_open_state(client, state, state_path, logger, args)
            if args.reconcile_only:
                return 0
            while True:
                run_cycle(pairs, books, resolutions, client, logger, args, kill_path, state, state_path)
                if args.once or not args.loop:
                    break
                time.sleep(args.interval_seconds)
    return 0


def resolve_floor_resolutions(
    pairs: list[str],
    books: dict[str, LoadedBook],
    logger: JsonlLogger,
) -> dict[str, PairFloorResolution]:
    """Resolve and log side floors/haircuts for every pair; fail closed on any gap.

    Floors are refit-CPCV per-era floors (frozen vintages decay); the gate
    certifies the documented floor under the periodic-retrain deploy policy.
    """
    try:
        resolutions = resolve_enabled_pairs(pairs)
    except FloorResolutionError as exc:
        logger.write("floor_resolution_failed", reason=str(exc))
        raise ExecutorError(f"floor resolution failed closed: {exc}") from exc
    for pair, res in resolutions.items():
        book = books[pair]
        mismatch = (
            abs(res.confidence_threshold - book.conf_thr) > 1e-12
            or book.coverage is None
            or abs(res.coverage - book.coverage) > 1e-12
        )
        if mismatch:
            logger.write(
                "floor_resolution_failed",
                pair=pair,
                reason="resolver_book_mismatch",
                resolver_coverage=res.coverage,
                book_coverage=book.coverage,
                resolver_conf_thr=res.confidence_threshold,
                book_conf_thr=book.conf_thr,
            )
            raise ExecutorError(f"{pair}: resolver coverage/conf_thr does not match loaded book")
        logger.write("floor_resolution", **res.as_dict())
    return resolutions


def resolve_pairs(value: str) -> list[str]:
    raw = value.strip()
    if raw in {"all-enabled", "all"}:
        return list(ENABLED_PAIRS)
    pairs = [p.strip().upper() for p in raw.split(",") if p.strip()]
    if not pairs:
        raise ExecutorError("no pairs selected")
    unknown = [p for p in pairs if p not in TARGET_BOOKS]
    if unknown:
        raise ExecutorError(f"unsupported pairs: {unknown}")
    if "EURUSD" in pairs:
        raise ExecutorError(EURUSD_DISABLED_REASON)
    disabled = [p for p in pairs if p not in ENABLED_PAIRS]
    if disabled:
        raise ExecutorError(f"pairs not enabled for demo executor: {disabled}")
    return pairs


def run_replay_schema_check(books: dict[str, LoadedBook]) -> None:
    print("Check: replay frozen book schemas on latest 2026 feature row")
    for pair, book in books.items():
        row = latest_replay_row(pair, book.feature_cols)
        score = book.score(row.row)
        print(
            f"PASS: {pair} {book.book_id} replay row {row.timestamp} "
            f"cols={len(book.feature_cols)} proba={score.proba:.8f} conf={score.confidence:.8f}"
        )


def run_public_smoke(
    pairs: list[str],
    books: dict[str, LoadedBook],
    logger: JsonlLogger,
    args: argparse.Namespace,
) -> None:
    print("Check: public market-data smoke, contract availability, store health, and live feature schema")
    store_dir = args.store_dir
    if store_dir is not None:
        require_store_health(store_dir, pairs, logger, args)
    with DerivOptionsClient.public() as client:
        verify_contracts(client, pairs, logger, args)
        builder = LiveFeatureBuilder(
            client,
            history_minutes=args.history_minutes,
            tick_volume_count=args.tick_volume_count,
            store_dir=store_dir,
            stale_seconds=args.store_max_stale_seconds,
        )
        for pair in pairs:
            row = builder.feature_row(pair, books[pair].feature_cols)
            score = books[pair].score(row.row)
            print(
                f"PASS: {pair} live row {row.timestamp} cols={len(books[pair].feature_cols)} "
                f"source={row.source} proba={score.proba:.8f} conf={score.confidence:.8f} gate={score.gate_passed}"
            )


def run_auth_smoke() -> None:
    print("Check: env/auth/demo assertion without printing secrets")
    env = DerivEnv.from_env()
    client = DerivOptionsClient.demo_from_env()
    if not client.is_demo:
        raise ExecutorError("FAIL: OTP URL is not demo")
    acct_hash = hashlib.sha256(env.account_id.encode("utf-8")).hexdigest()[:12]
    print(f"PASS: DERIV_APP_ID set, DERIV_PAT set, DERIV_ACCOUNT_ID sha256:{acct_hash}, demo OTP URL asserted")


def require_store_health(
    store_dir: Path,
    pairs: list[str],
    logger: JsonlLogger,
    args: argparse.Namespace,
) -> dict[str, Any]:
    health = store_health(
        out_dir=store_dir,
        enabled_pairs=pairs,
        min_required_rows=args.store_min_required_rows,
        max_stale_seconds=args.store_max_stale_seconds,
    )
    logger.write("store_health", **health)
    if not health["passes"]:
        raise ExecutorError(f"rolling store health failed: {json.dumps(health, sort_keys=True, default=str)}")
    print(f"PASS: rolling store health pairs={pairs} latest rows meet min/stale contract")
    return health


def verify_contracts(
    client: DerivOptionsClient,
    pairs: list[str],
    logger: JsonlLogger,
    args: argparse.Namespace | None = None,
) -> None:
    for pair in pairs:
        symbol = pair_symbol(pair)
        if args is None:
            resp = client.contracts_for(symbol)
        else:
            resp = call_deriv(client, logger, args, "contracts_for", client.contracts_for, symbol)
        available = resp.get("contracts_for", {}).get("available", [])
        ok = _has_15m_call_put(available, symbol)
        logger.write("symbol_contracts_verified", pair=pair, symbol=symbol, ok=ok, raw_hash=raw_hash(resp))
        if not ok:
            raise ExecutorError(f"{pair} {symbol}: contracts_for lacks 15m CALL/PUT")


def _has_15m_call_put(contracts: list[dict[str, Any]], symbol: str) -> bool:
    found = set()
    for c in contracts:
        if c.get("underlying_symbol") not in (None, symbol):
            continue
        if c.get("contract_type") not in {"CALL", "PUT"}:
            continue
        if _duration_allows_15m(str(c.get("min_contract_duration", "")), str(c.get("max_contract_duration", ""))):
            found.add(c["contract_type"])
    return {"CALL", "PUT"}.issubset(found)


def _duration_allows_15m(min_d: str, max_d: str) -> bool:
    def minutes(s: str, default: float) -> float:
        if not s:
            return default
        try:
            value = float(s[:-1])
            unit = s[-1].lower()
        except Exception:
            return default
        return value * {"t": 0.0, "s": 1 / 60, "m": 1, "h": 60, "d": 1440}.get(unit, default)

    return minutes(min_d, 0.0) <= 15 <= minutes(max_d, math.inf)


def _positive_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value) if value > 0 else None
    if isinstance(value, str):
        try:
            parsed = float(value)
        except ValueError:
            return None
        return parsed if parsed > 0 else None
    return None


def parse_quote(prop: dict[str, Any]) -> tuple[float | None, float | None, float | None, str | None]:
    """Return (ask, payout, live_breakeven, invalid_reason), failing closed.

    Both ask_price and payout are required in the Options proposal response
    schema; there is deliberately no display_value/stake fallback (it can
    fabricate a breakeven from the stake default).
    """
    ask = _positive_number(prop.get("ask_price"))
    payout = _positive_number(prop.get("payout"))
    if ask is None:
        return None, payout, None, "missing_or_invalid_ask"
    if payout is None:
        return ask, None, None, "missing_or_invalid_payout"
    return ask, payout, ask / payout, None


def snapshot_quotes(
    pairs: list[str],
    client: DerivOptionsClient,
    logger: JsonlLogger,
    args: argparse.Namespace,
) -> None:
    """Observability-only CALL+PUT quote snapshots; never reused for buys."""
    if args.quote_snapshots != "all":
        return
    for pair in pairs:
        symbol = DERIV_SYMBOLS[pair]
        for contract_type in ("CALL", "PUT"):
            try:
                resp = call_deriv(
                    client,
                    logger,
                    args,
                    "proposal_snapshot",
                    client.proposal,
                    symbol=symbol,
                    contract_type=contract_type,
                    amount=args.stake,
                    duration=HORIZON_MINUTES,
                )
            except ExecutorError as exc:
                logger.write("quote_snapshot_failed", pair=pair, contract_type=contract_type, error=str(exc))
                continue
            prop = resp.get("proposal", {})
            ask, payout, breakeven, invalid_reason = parse_quote(prop)
            spot_time = prop.get("spot_time")
            quote_age_ms = None
            if isinstance(spot_time, (int, float)):
                quote_age_ms = max(0.0, (datetime.now(timezone.utc).timestamp() - float(spot_time)) * 1000.0)
            logger.write(
                "quote_snapshot",
                pair=pair,
                side="UP" if contract_type == "CALL" else "DOWN",
                contract_type=contract_type,
                ask=ask,
                payout=payout,
                live_breakeven=breakeven,
                invalid_reason=invalid_reason,
                proposal_id_present=bool(prop.get("id")),
                quote_age_ms=quote_age_ms,
                raw_hash=raw_hash(resp),
            )


def run_cycle(
    pairs: list[str],
    books: dict[str, LoadedBook],
    resolutions: dict[str, PairFloorResolution],
    client: DerivOptionsClient,
    logger: JsonlLogger,
    args: argparse.Namespace,
    kill_path: Path,
    state: dict[str, Any],
    state_path: Path,
) -> None:
    check_kill_switch(kill_path, logger)
    if args.store_dir is not None:
        require_store_health(args.store_dir, pairs, logger, args)
    snapshot_quotes(pairs, client, logger, args)
    builder = LiveFeatureBuilder(
        client,
        history_minutes=args.history_minutes,
        tick_volume_count=args.tick_volume_count,
        store_dir=args.store_dir,
        stale_seconds=args.store_max_stale_seconds,
    )
    trades_today = count_events(logger.path, "buy_confirmed")
    open_count = len(pending_contracts(state))
    for pair in pairs:
        bought = process_pair(
            pair, books[pair], resolutions[pair], client, builder, logger, args,
            kill_path, state, state_path, trades_today, open_count,
        )
        if bought:
            trades_today += 1
            open_count += 1


def process_pair(
    pair: str,
    book: LoadedBook,
    resolution: PairFloorResolution,
    client: DerivOptionsClient,
    builder: LiveFeatureBuilder,
    logger: JsonlLogger,
    args: argparse.Namespace,
    kill_path: Path,
    state: dict[str, Any],
    state_path: Path,
    trades_today: int,
    open_count: int,
    now_utc: datetime | None = None,
) -> bool:
    check_kill_switch(kill_path, logger)
    now = now_utc or datetime.now(timezone.utc)
    if not is_ny_session(now):
        logger.write("signal_skipped", pair=pair, reason="outside_ny_session")
        return False
    try:
        row = builder.feature_row(pair, book.feature_cols)
        score = book.score(row.row)
    except (LiveFeatureError, ValueError) as exc:
        logger.write("feature_row_failed", pair=pair, error=str(exc))
        raise ExecutorError(str(exc)) from exc

    logger.write(
        "book_score",
        pair=pair,
        symbol=DERIV_SYMBOLS[pair],
        book_id=book.book_id,
        feature_timestamp=row.timestamp,
        feature_source=row.source,
        warnings=row.warnings,
        proba=score.proba,
        conf=score.confidence,
        threshold=score.threshold,
        direction=score.direction,
        gate_passed=score.gate_passed,
        gate_reasons=score.gate_reasons,
    )
    if not score.gate_passed:
        logger.write("signal_skipped", pair=pair, reason="book_gate", gate_reasons=score.gate_reasons)
        return False
    if trades_today >= args.max_trades_day:
        logger.write("signal_skipped", pair=pair, reason="max_trades_day", trades_today=trades_today)
        return False
    if open_count >= args.max_open:
        logger.write("signal_skipped", pair=pair, reason="max_open", open_count=open_count)
        return False
    last_buy = last_pair_buy_seconds(logger.path, pair)
    if last_buy is not None and time.time() - last_buy < HORIZON_MINUTES * 60:
        logger.write("signal_skipped", pair=pair, reason="pair_cooldown", last_buy_epoch=last_buy)
        return False

    contract_type = "CALL" if score.direction == "UP" else "PUT"
    symbol = DERIV_SYMBOLS[pair]
    # Side-specific documented floor for the model-selected side; raw proba is
    # never used as a calibrated win probability.
    side_floor = resolution.side_floor(score.direction)
    effective_floor = side_floor - resolution.haircut_abs - args.payout_edge_margin
    logger.write(
        "proposal_requested",
        pair=pair,
        symbol=symbol,
        contract_type=contract_type,
        direction=score.direction,
        stake=args.stake,
        side_refit_p10=side_floor,
        haircut_abs=resolution.haircut_abs,
        payout_edge_margin=args.payout_edge_margin,
        effective_floor=effective_floor,
    )
    # Fresh selected-side quote for the buy decision; snapshots are never reused.
    proposal = call_deriv(
        client,
        logger,
        args,
        "proposal",
        client.proposal,
        symbol=symbol,
        contract_type=contract_type,
        amount=args.stake,
        duration=HORIZON_MINUTES,
    )
    prop = proposal.get("proposal", {})
    proposal_id = prop.get("id")
    ask, payout, live_breakeven, invalid_reason = parse_quote(prop)
    net_edge = effective_floor - live_breakeven if live_breakeven is not None else None
    logger.write(
        "proposal_received",
        pair=pair,
        symbol=symbol,
        book_id=book.book_id,
        strategy_file=str(book.strategy_path),
        model_files=[str(p) for p in book.model_paths],
        content_id=book.content_id,
        book_created_utc=resolution.book_created_utc,
        feature_timestamp=row.timestamp,
        proba=score.proba,
        conf=score.confidence,
        threshold=score.threshold,
        direction=score.direction,
        contract_type=contract_type,
        proposal_id=proposal_id,
        ask=ask,
        payout=payout,
        live_breakeven=live_breakeven,
        side_refit_p10=side_floor,
        haircut_source=resolution.haircut_source,
        haircut_raw=resolution.haircut_raw,
        haircut_abs=resolution.haircut_abs,
        payout_edge_margin=args.payout_edge_margin,
        effective_floor=effective_floor,
        net_edge=net_edge,
        max_breakeven_ceiling=args.max_breakeven,
        raw_hash=raw_hash(proposal),
    )
    if invalid_reason is not None:
        logger.write("signal_skipped", pair=pair, reason=invalid_reason, ask=ask, payout=payout)
        return False
    if effective_floor <= live_breakeven:
        logger.write(
            "signal_skipped",
            pair=pair,
            reason="edge_not_positive",
            effective_floor=effective_floor,
            live_breakeven=live_breakeven,
            net_edge=net_edge,
        )
        return False
    if live_breakeven > args.max_breakeven:
        logger.write(
            "signal_skipped",
            pair=pair,
            reason="breakeven_too_high",
            live_breakeven=live_breakeven,
            max_breakeven_ceiling=args.max_breakeven,
        )
        return False
    if args.dry_run_proposal_only:
        logger.write("signal_skipped", pair=pair, reason="dry_run_proposal_only")
        return False
    if not args.demo_buy:
        logger.write("signal_skipped", pair=pair, reason="demo_buy_not_enabled")
        return False
    if not proposal_id:
        raise ExecutorError(f"{pair}: proposal response missing id")

    check_kill_switch(kill_path, logger)
    buy = call_deriv(client, logger, args, "buy", client.buy, str(proposal_id), ask)
    buy_payload = buy.get("buy", {})
    contract_id = buy_payload.get("contract_id")
    if contract_id is None:
        raise ExecutorError(f"{pair}: buy response missing contract_id")
    expected_expiry = now + timedelta(minutes=HORIZON_MINUTES)
    record_open_contract(
        state,
        pair=pair,
        proposal_id=str(proposal_id),
        contract_id=str(contract_id),
        stake=args.stake,
        buy_price=ask,
        expected_expiry=expected_expiry,
    )
    save_state(state_path, state)
    decision_context = {
        "pair": pair,
        "direction": score.direction,
        "contract_type": contract_type,
        "ask": ask,
        "payout": payout,
        "live_breakeven": live_breakeven,
        "effective_floor": effective_floor,
        "net_edge": net_edge,
        "proposal_id": str(proposal_id),
        "expected_expiry_utc": expected_expiry.isoformat(),
    }
    logger.write(
        "buy_confirmed",
        symbol=symbol,
        contract_id=str(contract_id),
        buy_price=ask,
        stake=args.stake,
        raw_hash=raw_hash(buy),
        **decision_context,
    )
    monitor_contract(client, str(contract_id), state, state_path, logger, args, decision_context)
    return True


def monitor_contract(
    client: DerivOptionsClient,
    contract_id: str,
    state: dict[str, Any],
    state_path: Path,
    logger: JsonlLogger,
    args: argparse.Namespace,
    decision_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    print(f"Check: monitor contract_id={contract_id} until terminal status")
    ctx = decision_context or {}
    sub_id: str | None = None
    first = call_deriv(client, logger, args, "proposal_open_contract", client.proposal_open_contract, contract_id, True)
    sub_id = (first.get("subscription") or {}).get("id")
    terminal = update_contract_state(state, contract_id, first)
    logger.write("contract_update", contract_id=contract_id, terminal=terminal, raw_hash=raw_hash(first), **ctx)
    deadline = time.monotonic() + args.monitor_timeout_seconds
    latest = first
    while not terminal:
        if time.monotonic() > deadline:
            save_state(state_path, state)
            logger.write("monitor_timeout", contract_id=contract_id, timeout_seconds=args.monitor_timeout_seconds, **ctx)
            raise ExecutorError(f"contract {contract_id}: monitor timeout before terminal status")
        try:
            latest = client.recv(timeout=args.monitor_interval_seconds)
        except Exception:
            latest = call_deriv(client, logger, args, "proposal_open_contract", client.proposal_open_contract, contract_id, False)
            time.sleep(args.monitor_interval_seconds)
        if "proposal_open_contract" not in latest:
            continue
        terminal = update_contract_state(state, contract_id, latest)
        logger.write("contract_update", contract_id=contract_id, terminal=terminal, raw_hash=raw_hash(latest), **ctx)
        save_state(state_path, state)
    if sub_id:
        call_deriv(client, logger, args, "forget", client.forget, sub_id)
        logger.write("subscription_forgotten", contract_id=contract_id, subscription_id=sub_id)
    poc = latest.get("proposal_open_contract", {})
    logger.write(
        "contract_closed",
        contract_id=contract_id,
        terminal_status=poc.get("status"),
        profit=poc.get("profit"),
        sell_price=poc.get("sell_price"),
        raw_hash=raw_hash(latest),
        **ctx,
    )
    print(f"PASS: contract_id={contract_id} reached terminal status")
    return latest


def call_deriv(
    client: DerivOptionsClient,
    logger: JsonlLogger,
    args: argparse.Namespace,
    op: str,
    fn: Callable[..., dict[str, Any]],
    *fn_args: Any,
    **fn_kwargs: Any,
) -> dict[str, Any]:
    attempts = int(getattr(args, "max_retries", 0)) + 1
    base = float(getattr(args, "retry_base_s", 0.0))
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            return fn(*fn_args, **fn_kwargs)
        except (DerivAPIError, OSError, TimeoutError) as exc:
            last = exc
            logger.write("deriv_call_failed", op=op, attempt=attempt + 1, max_attempts=attempts, error=str(exc))
            try:
                client.close()
            except Exception:
                pass
            if attempt + 1 >= attempts:
                break
            time.sleep(base * (2**attempt))
    raise ExecutorError(f"{op} failed after {attempts} attempts: {last}") from last


def is_ny_session(ts_utc: datetime) -> bool:
    if ts_utc.tzinfo is None:
        ts_utc = ts_utc.replace(tzinfo=timezone.utc)
    local = ts_utc.astimezone(NY_TZ)
    if local.weekday() >= 5:
        return False
    hour = local.hour + local.minute / 60.0
    return 8.0 <= hour < 17.0


def check_kill_switch(kill_path: Path, logger: JsonlLogger) -> None:
    if kill_path.exists():
        logger.write("kill_switch_triggered", path=str(kill_path))
        raise ExecutorError(f"kill switch present: {kill_path}")


def empty_state() -> dict[str, Any]:
    return {"schema": 1, "open_contracts": {}, "closed_contracts": []}


def load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return empty_state()
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ExecutorError(f"cannot load executor state {path}: {exc}") from exc
    if not isinstance(state, dict) or not isinstance(state.get("open_contracts"), dict):
        raise ExecutorError(f"state file is not reconcilable: {path}")
    state.setdefault("schema", 1)
    state.setdefault("closed_contracts", [])
    return state


def save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    tmp.replace(path)


def pending_contracts(state: dict[str, Any]) -> dict[str, Any]:
    return dict(state.get("open_contracts", {}))


def record_open_contract(
    state: dict[str, Any],
    *,
    pair: str,
    proposal_id: str,
    contract_id: str,
    stake: float,
    buy_price: float,
    expected_expiry: datetime,
) -> None:
    state.setdefault("open_contracts", {})[contract_id] = {
        "pair": pair,
        "proposal_id": proposal_id,
        "contract_id": contract_id,
        "stake": float(stake),
        "buy_price": float(buy_price),
        "expected_expiry_utc": expected_expiry.astimezone(timezone.utc).isoformat(),
        "status": "open",
        "last_platform_update_utc": datetime.now(timezone.utc).isoformat(),
    }


def update_contract_state(state: dict[str, Any], contract_id: str, response: dict[str, Any]) -> bool:
    poc = response.get("proposal_open_contract", response)
    item = state.setdefault("open_contracts", {}).setdefault(str(contract_id), {"contract_id": str(contract_id)})
    status = str(poc.get("status") or item.get("status") or "open").lower()
    item["status"] = status
    item["last_platform_update_utc"] = datetime.now(timezone.utc).isoformat()
    for key in ("sell_price", "profit", "payout", "entry_tick", "exit_tick"):
        if key in poc:
            item[key] = poc[key]
    terminal = bool(poc.get("is_sold")) or status in TERMINAL_STATUSES
    if terminal:
        item["terminal_utc"] = datetime.now(timezone.utc).isoformat()
        state.setdefault("closed_contracts", []).append(item)
        state.get("open_contracts", {}).pop(str(contract_id), None)
    return terminal


def reconcile_open_state(
    client: DerivOptionsClient,
    state: dict[str, Any],
    state_path: Path,
    logger: JsonlLogger,
    args: argparse.Namespace,
) -> None:
    unresolved: list[str] = []
    for contract_id in list(pending_contracts(state)):
        resp = call_deriv(client, logger, args, "proposal_open_contract", client.proposal_open_contract, contract_id, False)
        terminal = update_contract_state(state, contract_id, resp)
        logger.write("state_reconciled_contract", contract_id=contract_id, terminal=terminal, raw_hash=raw_hash(resp))
        if not terminal:
            unresolved.append(contract_id)
    save_state(state_path, state)
    if unresolved:
        logger.write("reconcile_failed", reason="unresolved_open_contracts", unresolved=unresolved)
        raise ExecutorError(f"pending/open local state not terminal after Deriv reconciliation: {unresolved}")


def count_events(path: Path, event: str) -> int:
    if not path.exists():
        return 0
    count = 0
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("event") == event:
                count += 1
    return count


def last_pair_buy_seconds(path: Path, pair: str) -> float | None:
    if not path.exists():
        return None
    latest: float | None = None
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("event") != "buy_confirmed" or row.get("pair") != pair:
                continue
            ts = row.get("timestamp_utc")
            if not isinstance(ts, str):
                continue
            try:
                value = datetime.fromisoformat(ts).timestamp()
            except ValueError:
                continue
            latest = value if latest is None else max(latest, value)
    return latest


@dataclass
class FakeScore:
    pair: str
    book_id: str = "fake.v1"
    proba: float = 0.62
    confidence: float = 0.12
    threshold: float = 0.10
    direction: str = "UP"
    gate_passed: bool = True
    gate_reasons: list[str] | None = None


class FakeBook:
    pair = "USDJPY"
    book_id = "USDJPY.fake.v1"
    strategy_path = Path("fake_strategy.json")
    model_paths = [Path("fake_lgb.txt")]
    content_id = "fake"
    feature_cols = ["x"]

    def __init__(self, direction: str = "UP"):
        self.direction = direction

    def score(self, feature_row: pd.Series | dict[str, Any]) -> FakeScore:
        proba = 0.62 if self.direction == "UP" else 0.38
        return FakeScore(pair=self.pair, proba=proba, direction=self.direction, gate_reasons=[])


def fixture_resolution(
    pair: str = "USDJPY",
    up_floor: float = 0.6005,
    down_floor: float = 0.5738,
    haircut_abs: float = 0.0036,
) -> PairFloorResolution:
    return PairFloorResolution(
        pair=pair,
        book_id=f"{pair}.fake.v1",
        strategy_path="fixture",
        coverage=0.02,
        coverage_source="fixture",
        confidence_threshold=0.10,
        up_floor=up_floor,
        down_floor=down_floor,
        floor_source="fixture",
        haircut_raw=-haircut_abs,
        haircut_abs=haircut_abs,
        haircut_source="fixture",
        haircut_source_detail="fixture",
        book_created_utc="2026-06-08",
        vintage_source="fixture",
        registry_content_id="fixture",
    )


class FakeBuilder:
    def feature_row(self, pair: str, expected_cols: list[str]) -> FeatureRow:
        ts = pd.Timestamp("2026-06-15T14:00:00Z")
        return FeatureRow(pair=pair, timestamp=ts, row=pd.Series({c: 1.0 for c in expected_cols}), source="fake", warnings=[])


class RecordingOptionsClient(DerivOptionsClient):
    def __init__(self, is_demo: bool = False):
        self.url = "wss://example.test/demo" if is_demo else "wss://example.test/public"
        self.timeout = 1.0
        self.ws = None
        self._req_id = 0
        self.payloads: list[dict[str, Any]] = []
        self.calls: list[str] = []
        self.recv_queue: list[dict[str, Any]] = []

    def request(self, payload: dict[str, Any], timeout: float | None = None) -> dict[str, Any]:
        self.payloads.append(dict(payload))
        if "contracts_for" in payload:
            return {
                "contracts_for": {
                    "available": [
                        {"underlying_symbol": payload["contracts_for"], "contract_type": "CALL", "min_contract_duration": "1m", "max_contract_duration": "365d"},
                        {"underlying_symbol": payload["contracts_for"], "contract_type": "PUT", "min_contract_duration": "1m", "max_contract_duration": "365d"},
                    ]
                }
            }
        if "ticks_history" in payload:
            return {"history": {"times": []}}
        return {}


class FakeTradeClient(RecordingOptionsClient):
    def __init__(
        self,
        terminal: bool = True,
        ask: float | None = 1.0,
        payout: float | None = 2.0,
        include_proposal_id: bool = True,
    ):
        super().__init__(is_demo=True)
        self.terminal = terminal
        self.ask = ask
        self.payout = payout
        self.include_proposal_id = include_proposal_id

    def proposal(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append("proposal")
        self.payloads.append({"proposal": 1, **kwargs})
        prop: dict[str, Any] = {}
        if self.include_proposal_id:
            prop["id"] = "proposal-1"
        if self.ask is not None:
            prop["ask_price"] = self.ask
        if self.payout is not None:
            prop["payout"] = self.payout
        return {"proposal": prop}

    def buy(self, proposal_id: str, price: float) -> dict[str, Any]:
        self.calls.append("buy")
        return {"buy": {"contract_id": "contract-1", "buy_price": price}}

    def proposal_open_contract(self, contract_id: str | int, subscribe: bool = False) -> dict[str, Any]:
        self.calls.append("proposal_open_contract")
        status = "sold" if self.terminal and not subscribe else "open"
        return {
            "proposal_open_contract": {"contract_id": str(contract_id), "status": status, "is_sold": int(status == "sold")},
            "subscription": {"id": "sub-1"} if subscribe else {},
        }

    def recv(self, timeout: float | None = None) -> dict[str, Any]:
        return {"proposal_open_contract": {"contract_id": "contract-1", "status": "sold", "is_sold": 1}}

    def forget(self, subscription_id: str) -> dict[str, Any]:
        self.calls.append("forget")
        return {"forget": subscription_id}


def _last_event(path: Path, event: str) -> dict[str, Any] | None:
    latest: dict[str, Any] | None = None
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("event") == event:
                latest = row
    return latest


def _require_fields(row: dict[str, Any] | None, fields: list[str], label: str) -> dict[str, Any]:
    if row is None:
        raise ExecutorError(f"FAIL: expected {label} event was not logged")
    missing = [f for f in fields if f not in row]
    if missing:
        raise ExecutorError(f"FAIL: {label} event missing fields {missing}")
    return row


def _smoke_args(**overrides: Any) -> SimpleNamespace:
    base = {
        "stake": 1.0,
        "max_breakeven": 0.60,
        "payout_edge_margin": 0.005,
        "quote_snapshots": "all",
        "dry_run_proposal_only": True,
        "demo_buy": False,
        "max_trades_day": 3,
        "max_open": 1,
        "max_retries": 0,
        "retry_base_s": 0.0,
        "monitor_timeout_seconds": 5.0,
        "monitor_interval_seconds": 0.01,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def _smoke_process(client: FakeTradeClient, logger: JsonlLogger, args: SimpleNamespace,
                   log_dir: Path, resolution: PairFloorResolution, book: FakeBook,
                   state: dict[str, Any] | None = None, state_path: Path | None = None) -> bool:
    return process_pair(
        "USDJPY",
        book,  # type: ignore[arg-type]
        resolution,
        client,  # type: ignore[arg-type]
        FakeBuilder(),  # type: ignore[arg-type]
        logger,
        args,
        log_dir / "KILL",
        state if state is not None else empty_state(),
        state_path if state_path is not None else log_dir / "state.json",
        trades_today=0,
        open_count=0,
        now_utc=datetime(2026, 6, 15, 14, 0, tzinfo=timezone.utc),
    )


def _smoke_payload_and_candles() -> None:
    # Scope-1 assertion at the payload-construction layer: interception is
    # RecordingOptionsClient.request, so this exercises DerivOptionsClient.proposal
    # itself (FakeTradeClient.proposal returns a canned dict and would be vacuous).
    client = RecordingOptionsClient()
    client.proposal(symbol="frxUSDJPY", contract_type="CALL", amount=1.0)
    payload = client.payloads[-1]
    if payload.get("underlying_symbol") != "frxUSDJPY":
        raise ExecutorError("FAIL: proposal payload missing underlying_symbol")
    if "symbol" in payload:
        raise ExecutorError("FAIL: proposal payload still includes legacy symbol field")
    client.contracts_for("frxUSDJPY")
    if "currency" in client.payloads[-1]:
        raise ExecutorError("FAIL: contracts_for payload still includes currency")
    client.ticks_history("frxUSDJPY", style="candles", count=5, granularity=60)
    if client.payloads[-1].get("adjust_start_time") != 1:
        raise ExecutorError("FAIL: ticks_history payload missing adjust_start_time=1")

    now = pd.Timestamp.utcnow().floor("min") - pd.Timedelta(minutes=600)
    candles = [
        {"epoch": int((now + pd.Timedelta(minutes=i)).timestamp()), "open": 1.0, "high": 1.1, "low": 0.9, "close": 1.0}
        for i in range(501)
    ]
    m1 = candles_to_m1(candles)
    if "volume" not in m1.columns or not np.isfinite(m1["volume"]).all() or not m1.attrs.get("volume_warning"):
        raise ExecutorError("FAIL: missing-volume candle conversion did not produce explicit synthetic volume warning")


def _smoke_resolver() -> None:
    # Real-registry resolution must reproduce the issue-#3 floor/haircut table
    # exactly, with source rules; xpair books must resolve cov1 (no "0.02" fallback).
    expected = {
        "USDJPY": (0.02, 0.12624352649499532, 0.6005, 0.5738, 0.0036, "direct_ticksettle_json"),
        "USDCAD": (0.02, 0.06189305393772768, 0.5968, 0.577, 0.0035, "documented_proxy"),
        "AUDUSD": (0.03, 0.06624753336867155, 0.5755, 0.5901, 0.0035, "direct_ticksettle_json"),
        "NZDUSD": (0.02, 0.0835073173947228, 0.5749, 0.5803, 0.014, "structured"),
        "USDCHF": (0.01, 0.16933929389708485, 0.6935, 0.6682, 0.0035, "structured"),
        "GBPUSD": (0.01, 0.13209766255390926, 0.6552, 0.6395, 0.0078, "structured"),
    }
    rows = resolve_enabled_pairs()
    for pair, (cov, thr, up, down, haircut_abs, rule) in expected.items():
        res = rows[pair]
        got = (res.coverage, res.confidence_threshold, res.up_floor, res.down_floor, res.haircut_abs, res.haircut_source)
        if got != (cov, thr, up, down, haircut_abs, rule):
            raise ExecutorError(f"FAIL: {pair} resolution {got} != expected {(cov, thr, up, down, haircut_abs, rule)}")
        if res.haircut_raw > 0 or abs(res.haircut_raw) != res.haircut_abs:
            raise ExecutorError(f"FAIL: {pair} signed raw haircut {res.haircut_raw} inconsistent with abs {res.haircut_abs}")
        if res.book_created_utc in (None, "", "vintage_date_unavailable"):
            raise ExecutorError(f"FAIL: {pair} missing book vintage date")

    # Synthetic xpair fixture: deploy_cov must drive both conf_thr and floors to cov1.
    xp_strategy = {
        "gates": {"0.02": {"conf_thr": 0.145}, "0.01": {"conf_thr": 0.169}},
        "deploy": {
            "refit_floor_p10_cov1": {"UP": 0.69, "DOWN": 0.66},
            "refit_floor_p10_cov2": {"UP": 0.65, "DOWN": 0.64},
            "barclose_haircut": -0.0035,
        },
    }
    xp_entry = {"metrics": {"deploy_cov": 0.01, "cov": 0.01, "barclose_haircut": -0.0035, "content_id": "fixture"}}
    xp_manifest = {"created_utc": "2026-06-11", "metrics": {"refit_cpcv_p10_cov1_UP": 0.69, "refit_cpcv_p10_cov1_DOWN": 0.66}}
    res = resolve_book("USDCHF", "USDCHF.fixture", xp_entry, xp_manifest, xp_strategy, "fixture")
    if res.confidence_threshold != 0.169 or res.coverage != 0.01:
        raise ExecutorError("FAIL: xpair fixture did not select the cov1 gates entry (0.02 fallback used)")
    if (res.up_floor, res.down_floor) != (0.69, 0.66):
        raise ExecutorError("FAIL: xpair fixture floors not resolved at active cov1 coverage")

    # Rule-4 fallback: synthetic no-source pair resolves via the recomputed max rule.
    raw, abs_applied, rule, detail = resolve_haircut("FAKEPAIR", "FAKEPAIR.fixture", {}, {}, {}, proxy_table={})
    live_max, contributor, _ = recompute_fallback_max()
    if rule != "fallback_max_validated_fx_ticksettle" or abs_applied != live_max or contributor not in detail:
        raise ExecutorError("FAIL: rule-4 fallback did not use the recomputed validated-ticksettle max")
    with tempfile.TemporaryDirectory() as tmp:
        synth = Path(tmp) / "fakepair2_15m_ticksettle_result.json"
        synth.write_text(json.dumps({
            "mean_tick_minus_bar_COMB": -0.05,
            "verdict": {"tick_settlement_preserves_edge": True},
        }))
        raw2, abs2, rule2, _ = resolve_haircut("NOPE", "NOPE.fixture", {}, {}, {}, ticksettle_dir=Path(tmp), proxy_table={})
        if rule2 != "fallback_max_validated_fx_ticksettle" or abs(abs2 - 0.05) > 1e-12:
            raise ExecutorError("FAIL: rule-4 fallback is not recomputed from the ticksettle dir (frozen constant?)")


def run_fake_client_smoke() -> None:
    print(
        "Check: fake-client payloads, candle conversion, floor/haircut resolver, payout gate, "
        "quote snapshots, no-buy mode, monitoring, reconciliation, lock, EURUSD fail-closed"
    )
    _smoke_payload_and_candles()
    _smoke_resolver()

    with tempfile.TemporaryDirectory() as tmp:
        log_dir = Path(tmp)
        logger = JsonlLogger(log_dir)
        args = _smoke_args()
        resolution = fixture_resolution()  # eff floor = .6005 - .0036 - .005 = .5919

        # floor_resolution structured-log fields (real resolver output).
        logger.write("floor_resolution", **resolve_enabled_pairs(["USDJPY"])["USDJPY"].as_dict())
        _require_fields(
            _last_event(logger.path, "floor_resolution"),
            ["pair", "book_id", "strategy_path", "coverage", "confidence_threshold", "up_floor",
             "down_floor", "floor_source", "haircut_source", "haircut_raw", "haircut_abs",
             "book_created_utc"],
            "floor_resolution",
        )

        # Positive edge (be .50 < eff .5919) reaches the proposal-only skip; no buy.
        dry_client = FakeTradeClient()
        bought = _smoke_process(dry_client, logger, args, log_dir, resolution, FakeBook())
        if bought or "buy" in dry_client.calls or count_events(logger.path, "buy_confirmed"):
            raise ExecutorError("FAIL: proposal-only mode bought or emitted buy_confirmed")
        skip = _require_fields(_last_event(logger.path, "signal_skipped"), ["reason"], "signal_skipped")
        if skip["reason"] != "dry_run_proposal_only":
            raise ExecutorError(f"FAIL: positive-edge dry run skipped for {skip['reason']}, not dry_run_proposal_only")
        _require_fields(
            _last_event(logger.path, "proposal_received"),
            ["pair", "contract_type", "direction", "ask", "payout", "live_breakeven", "side_refit_p10",
             "haircut_source", "haircut_raw", "haircut_abs", "payout_edge_margin", "effective_floor",
             "net_edge", "max_breakeven_ceiling", "book_created_utc", "proposal_id", "raw_hash"],
            "proposal_received",
        )

        # Negative edge (payout 1.65 -> be .6061 > eff .5919) skips before buy.
        neg_client = FakeTradeClient(payout=1.65)
        bought = _smoke_process(neg_client, logger, args, log_dir, resolution, FakeBook())
        skip = _last_event(logger.path, "signal_skipped")
        if bought or "buy" in neg_client.calls or skip is None or skip["reason"] != "edge_not_positive":
            raise ExecutorError("FAIL: negative-edge quote did not skip with edge_not_positive before buy")

        # Missing/zero payout and missing/zero ask each fail closed with explicit reasons.
        for client, expected_reason in (
            (FakeTradeClient(payout=None), "missing_or_invalid_payout"),
            (FakeTradeClient(payout=0.0), "missing_or_invalid_payout"),
            (FakeTradeClient(ask=None), "missing_or_invalid_ask"),
            (FakeTradeClient(ask=0.0), "missing_or_invalid_ask"),
        ):
            bought = _smoke_process(client, logger, args, log_dir, resolution, FakeBook())
            skip = _last_event(logger.path, "signal_skipped")
            if bought or "buy" in client.calls or skip is None or skip["reason"] != expected_reason:
                raise ExecutorError(f"FAIL: invalid quote did not fail closed with {expected_reason}")

        # Selected-side gating: DOWN selections gate on the DOWN floor (PUT),
        # UP selections on the UP floor (CALL). Floors chosen so using the wrong
        # side's floor would flip the decision.
        for direction, floors in (("DOWN", {"up_floor": 0.99, "down_floor": 0.55}),
                                  ("UP", {"up_floor": 0.55, "down_floor": 0.99})):
            side_client = FakeTradeClient(payout=1.8)  # be .5556 > eff .5414 for the .55 floor
            bought = _smoke_process(side_client, logger, args, log_dir,
                                    fixture_resolution(**floors), FakeBook(direction))
            prop_row = _last_event(logger.path, "proposal_received")
            skip = _last_event(logger.path, "signal_skipped")
            expected_ct = "CALL" if direction == "UP" else "PUT"
            if bought or skip is None or skip["reason"] != "edge_not_positive":
                raise ExecutorError(f"FAIL: {direction} selected-side gate did not use its own floor")
            if prop_row is None or prop_row["contract_type"] != expected_ct or prop_row["side_refit_p10"] != 0.55:
                raise ExecutorError(f"FAIL: {direction} proposal did not gate on the {direction} floor / {expected_ct}")

        # Quote snapshots record both sides without causing a buy.
        snap_logger = JsonlLogger(log_dir / "snap")
        snap_client = FakeTradeClient()
        snapshot_quotes(["USDJPY"], snap_client, snap_logger, args)  # type: ignore[arg-type]
        if count_events(snap_logger.path, "quote_snapshot") != 2 or "buy" in snap_client.calls:
            raise ExecutorError("FAIL: quote snapshots did not record both sides without buying")
        snap = _require_fields(
            _last_event(snap_logger.path, "quote_snapshot"),
            ["pair", "side", "contract_type", "ask", "payout", "live_breakeven", "proposal_id_present",
             "quote_age_ms", "raw_hash"],
            "quote_snapshot",
        )
        if {snap["contract_type"]} - {"CALL", "PUT"}:
            raise ExecutorError("FAIL: quote_snapshot contract_type invalid")

        # Demo-buy path with positive edge: buy, monitor to terminal, forget; logs
        # preserve the decision context end-to-end.
        buy_args = _smoke_args(dry_run_proposal_only=False, demo_buy=True)
        buy_state = empty_state()
        buy_logger = JsonlLogger(log_dir / "buy")
        buy_client = FakeTradeClient()
        bought = _smoke_process(buy_client, buy_logger, buy_args, log_dir, resolution, FakeBook(),
                                state=buy_state, state_path=log_dir / "buy_state.json")
        if not bought or "forget" not in buy_client.calls or pending_contracts(buy_state):
            raise ExecutorError("FAIL: demo-buy monitor did not reach terminal state and forget subscription")
        _require_fields(
            _last_event(buy_logger.path, "buy_confirmed"),
            ["pair", "direction", "contract_type", "ask", "payout", "live_breakeven", "effective_floor",
             "net_edge", "proposal_id", "contract_id", "buy_price", "expected_expiry_utc", "stake", "raw_hash"],
            "buy_confirmed",
        )
        _require_fields(
            _last_event(buy_logger.path, "contract_update"),
            ["contract_id", "terminal", "pair", "direction", "contract_type", "effective_floor",
             "live_breakeven", "net_edge"],
            "contract_update",
        )
        _require_fields(
            _last_event(buy_logger.path, "contract_closed"),
            ["contract_id", "terminal_status", "pair", "contract_type"],
            "contract_closed",
        )

        # Reconciliation fail-closed + reconcile_failed event.
        unresolved = empty_state()
        record_open_contract(
            unresolved,
            pair="USDJPY",
            proposal_id="proposal-2",
            contract_id="contract-open",
            stake=1.0,
            buy_price=1.0,
            expected_expiry=datetime(2026, 6, 15, 14, 15, tzinfo=timezone.utc),
        )
        try:
            reconcile_open_state(FakeTradeClient(terminal=False), unresolved, log_dir / "unresolved.json", logger, buy_args)  # type: ignore[arg-type]
        except ExecutorError:
            pass
        else:
            raise ExecutorError("FAIL: unresolved local state did not fail closed")
        _require_fields(_last_event(logger.path, "reconcile_failed"), ["reason", "unresolved"], "reconcile_failed")

        # Lock contention fails closed and logs lock_acquire_failed.
        with ExecutorLock(log_dir, "demo-account", logger):
            try:
                with ExecutorLock(log_dir, "demo-account", logger):
                    raise ExecutorError("FAIL: second executor lock unexpectedly acquired")
            except ExecutorError:
                pass
        _require_fields(_last_event(logger.path, "lock_acquire_failed"), ["reason", "path"], "lock_acquire_failed")

        try:
            resolve_pairs("EURUSD")
        except ExecutorError as exc:
            if "OF_*" not in str(exc):
                raise
        else:
            raise ExecutorError("FAIL: EURUSD did not fail closed")
    print(
        "PASS: fake-client smoke covered payloads, candle conversion, resolver table + cov1 fixtures + "
        "rule-4 recompute, payout gate fail-closed paths, selected-side floors, snapshots, no-buy, "
        "monitor context, reconciliation, lock, EURUSD fail-closed"
    )


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ExecutorError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
