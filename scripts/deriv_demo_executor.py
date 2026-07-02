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
from deriv_runtime_core import (
    ENABLED_PAIRS,
    ExecutorError,
    ExecutorLock,
    effective_floor_for,
    FakeBook,
    FakeBuilder,
    FakeTradeClient,
    HORIZON_MINUTES,
    JsonlLogger,
    RecordingOptionsClient,
    _has_15m_call_put,
    _last_event,
    _require_fields,
    _smoke_args,
    call_deriv,
    check_kill_switch,
    count_events,
    empty_state,
    fixture_resolution,
    is_ny_session,
    last_pair_buy_seconds,
    load_state,
    monitor_contract,
    parse_quote,
    pending_contracts,
    raw_hash,
    reconcile_open_state,
    record_open_contract,
    resolve_floor_resolutions,
    resolve_pairs,
    save_state,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG_DIR = REPO_ROOT / "logs" / "paper_trades"


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pairs", default="all-enabled", help="comma-separated pair list or all-enabled")
    p.add_argument("--stake", type=float, default=1.0)
    p.add_argument("--max-trades-day", type=int, default=0, help="0 = unlimited (demo default: trade every signal)")
    p.add_argument("--max-open", type=int, default=0, help="0 = unlimited concurrent open contracts, account-wide")
    p.add_argument("--pair-cooldown-seconds", type=int, default=0, help="0 = none; per-pair wait after a buy")
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
    p.add_argument("--blocking-monitor", action="store_true",
                   help="block after each buy until the contract settles (serializes the account to one open trade)")
    p.add_argument("--store-min-required-rows", type=int, default=14000)
    p.add_argument("--store-max-stale-seconds", type=int, default=180)
    p.add_argument("--check-books", action="store_true")
    p.add_argument("--replay-schema-check", action="store_true")
    p.add_argument("--public-smoke", action="store_true")
    p.add_argument("--auth-smoke", action="store_true")
    p.add_argument("--fake-client-smoke", action="store_true")
    p.add_argument("--reconcile-only", action="store_true")
    args = p.parse_args(argv)
    # negative caps would be truthy and reject everything (review footgun)
    args.max_open = max(0, args.max_open)
    args.max_trades_day = max(0, args.max_trades_day)
    args.pair_cooldown_seconds = max(0, args.pair_cooldown_seconds)
    return args


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
        pair_cooldown_seconds=args.pair_cooldown_seconds,
        blocking_monitor=args.blocking_monitor,
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
    if args.demo_buy:
        # Per-cycle reconciliation: with detached monitoring this is what
        # settles open contracts and lets open_count fall (review F1 — the
        # detached design is incoherent without it in --loop mode).
        reconcile_open_state(client, state, state_path, logger, args)
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
    # Risk caps are OPT-IN (0 = unconstrained): demo trades every firing signal
    if args.max_trades_day and trades_today >= args.max_trades_day:
        logger.write("signal_skipped", pair=pair, reason="max_trades_day", trades_today=trades_today)
        return False
    if args.max_open and open_count >= args.max_open:
        logger.write("signal_skipped", pair=pair, reason="max_open", open_count=open_count)
        return False
    if args.pair_cooldown_seconds:
        last_buy = last_pair_buy_seconds(logger.path, pair)
        if last_buy is not None and time.time() - last_buy < args.pair_cooldown_seconds:
            logger.write("signal_skipped", pair=pair, reason="pair_cooldown", last_buy_epoch=last_buy)
            return False

    contract_type = "CALL" if score.direction == "UP" else "PUT"
    symbol = DERIV_SYMBOLS[pair]
    # Side-specific documented floor for the model-selected side; raw proba is
    # never used as a calibrated win probability.
    side_floor = resolution.side_floor(score.direction)
    # Shared parity-critical formula: both the executor and the hot daemon
    # must compute the payout-gate floor through deriv_runtime_core.
    effective_floor = effective_floor_for(resolution, score.direction, args.payout_edge_margin)
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
    if getattr(args, "blocking_monitor", False):
        monitor_contract(client, str(contract_id), state, state_path, logger, args, decision_context)
    else:
        # Detached (default): the contract is recorded in state; each cycle's
        # reconciliation refreshes it to terminal. Blocking here serialized the
        # whole account to one open contract at a time.
        logger.write("monitor_detached", contract_id=str(contract_id))
    return True


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

        # Detached monitoring (the DEFAULT path): buy returns immediately, the
        # contract is carried open in state, and the next cycle's
        # reconciliation settles it to terminal.
        det_args = _smoke_args(dry_run_proposal_only=False, demo_buy=True, blocking_monitor=False)
        det_state = empty_state()
        det_logger = JsonlLogger(log_dir / "detached")
        det_client = FakeTradeClient()
        bought = _smoke_process(det_client, det_logger, det_args, log_dir, resolution, FakeBook(),
                                state=det_state, state_path=log_dir / "det_state.json")
        if not bought or not pending_contracts(det_state):
            raise ExecutorError("FAIL: detached buy did not record an open contract in state")
        _require_fields(_last_event(det_logger.path, "monitor_detached"), ["contract_id"], "monitor_detached")
        reconcile_open_state(FakeTradeClient(), det_state, log_dir / "det_state.json", det_logger, det_args)  # type: ignore[arg-type]
        if pending_contracts(det_state):
            raise ExecutorError("FAIL: reconciliation did not settle the detached contract to terminal")

        # Reconciliation: a KNOWN-open contract is refreshed and carried (multi
        # -open concurrency is normal); no refusal, no raise.
        carried = empty_state()
        record_open_contract(
            carried,
            pair="USDJPY",
            proposal_id="proposal-2",
            contract_id="contract-open",
            stake=1.0,
            buy_price=1.0,
            expected_expiry=datetime.now(timezone.utc) + timedelta(minutes=15),  # in-flight, not overdue
        )
        reconcile_open_state(FakeTradeClient(terminal=False), carried, log_dir / "carried.json", logger, buy_args)  # type: ignore[arg-type]
        if "contract-open" not in pending_contracts(carried):
            raise ExecutorError("FAIL: known-open contract was not carried through reconciliation")
        _require_fields(_last_event(logger.path, "reconcile_carried_open"), ["contracts"], "reconcile_carried_open")

        # Reconciliation fail-closed on an OVERDUE open contract (still "open"
        # long past expected expiry = unknown exposure).
        overdue = empty_state()
        record_open_contract(
            overdue,
            pair="USDJPY",
            proposal_id="proposal-4",
            contract_id="contract-overdue",
            stake=1.0,
            buy_price=1.0,
            expected_expiry=datetime.now(timezone.utc) - timedelta(hours=1),
        )
        try:
            reconcile_open_state(FakeTradeClient(terminal=False), overdue, log_dir / "overdue.json", logger, buy_args)  # type: ignore[arg-type]
        except ExecutorError:
            pass
        else:
            raise ExecutorError("FAIL: overdue-open contract did not fail closed")
        _require_fields(_last_event(logger.path, "reconcile_overdue_open"), ["contract_id"], "reconcile_overdue_open")

        # Reconciliation fail-closed on UNRESOLVABLE state (query failure).
        class _FailingClient(FakeTradeClient):
            def proposal_open_contract(self, contract_id: str | int, subscribe: bool = False) -> dict[str, Any]:
                raise DerivAPIError("poc query down")

        unresolved = empty_state()
        record_open_contract(
            unresolved,
            pair="USDJPY",
            proposal_id="proposal-3",
            contract_id="contract-unknown",
            stake=1.0,
            buy_price=1.0,
            expected_expiry=datetime(2026, 6, 15, 14, 15, tzinfo=timezone.utc),
        )
        try:
            reconcile_open_state(_FailingClient(terminal=False), unresolved, log_dir / "unresolved.json", logger, buy_args)  # type: ignore[arg-type]
        except ExecutorError:
            pass
        else:
            raise ExecutorError("FAIL: unresolvable local state did not fail closed")
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
        "monitor context, detached-monitor lifecycle (buy -> carried open -> reconciled terminal), "
        "reconciliation carry/fail-closed, lock, EURUSD fail-closed"
    )


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ExecutorError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
