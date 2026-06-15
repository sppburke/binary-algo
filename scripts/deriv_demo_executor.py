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
    def __init__(self, log_dir: Path, account_key: str):
        digest = hashlib.sha256(account_key.encode("utf-8")).hexdigest()[:12]
        self.path = log_dir / f"executor_{digest}.lock"
        self.fd: int | None = None

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
    p.add_argument("--max-breakeven", type=float, default=0.60)
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
    logger.write(
        "startup_config",
        pairs=pairs,
        enabled_pairs=ENABLED_PAIRS,
        stake=args.stake,
        dry_run=args.dry_run_proposal_only,
        demo_buy=args.demo_buy,
        store_dir=args.store_dir,
    )

    books = load_target_books(pairs)
    for row in book_inventory(pairs):
        logger.write("book_loaded", **row)

    if args.check_books:
        print(json.dumps(book_inventory(pairs), indent=2))
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

    with ExecutorLock(args.log_dir, account_key):
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
                run_cycle(pairs, books, client, logger, args, kill_path, state, state_path)
                if args.once or not args.loop:
                    break
                time.sleep(args.interval_seconds)
    return 0


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
    print(f"PASS: DERIV_APP_ID set, DERIV_PAT set, DERIV_ACCOUNT_ID={env.account_id}, demo OTP URL asserted")


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


def run_cycle(
    pairs: list[str],
    books: dict[str, LoadedBook],
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
        bought = process_pair(pair, books[pair], client, builder, logger, args, kill_path, state, state_path, trades_today, open_count)
        if bought:
            trades_today += 1
            open_count += 1


def process_pair(
    pair: str,
    book: LoadedBook,
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
    logger.write("proposal_requested", pair=pair, symbol=symbol, contract_type=contract_type, stake=args.stake)
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
    ask = float(prop.get("ask_price", prop.get("display_value", args.stake)))
    payout = float(prop.get("payout", 0.0))
    breakeven = ask / payout if payout > 0 else math.inf
    logger.write(
        "proposal_received",
        pair=pair,
        symbol=symbol,
        book_id=book.book_id,
        strategy_file=str(book.strategy_path),
        model_files=[str(p) for p in book.model_paths],
        content_id=book.content_id,
        feature_timestamp=row.timestamp,
        proba=score.proba,
        conf=score.confidence,
        threshold=score.threshold,
        direction=score.direction,
        proposal_id=proposal_id,
        ask=ask,
        implied_breakeven=breakeven,
        raw_hash=raw_hash(proposal),
    )
    if breakeven > args.max_breakeven:
        logger.write("signal_skipped", pair=pair, reason="breakeven_too_high", implied_breakeven=breakeven)
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
    logger.write(
        "buy_confirmed",
        pair=pair,
        symbol=symbol,
        contract_id=str(contract_id),
        proposal_id=str(proposal_id),
        buy_price=ask,
        raw_hash=raw_hash(buy),
    )
    monitor_contract(client, str(contract_id), state, state_path, logger, args)
    return True


def monitor_contract(
    client: DerivOptionsClient,
    contract_id: str,
    state: dict[str, Any],
    state_path: Path,
    logger: JsonlLogger,
    args: argparse.Namespace,
) -> dict[str, Any]:
    print(f"Check: monitor contract_id={contract_id} until terminal status")
    sub_id: str | None = None
    first = call_deriv(client, logger, args, "proposal_open_contract", client.proposal_open_contract, contract_id, True)
    sub_id = (first.get("subscription") or {}).get("id")
    terminal = update_contract_state(state, contract_id, first)
    logger.write("contract_update", contract_id=contract_id, terminal=terminal, raw_hash=raw_hash(first))
    deadline = time.monotonic() + args.monitor_timeout_seconds
    latest = first
    while not terminal:
        if time.monotonic() > deadline:
            save_state(state_path, state)
            raise ExecutorError(f"contract {contract_id}: monitor timeout before terminal status")
        try:
            latest = client.recv(timeout=args.monitor_interval_seconds)
        except Exception:
            latest = call_deriv(client, logger, args, "proposal_open_contract", client.proposal_open_contract, contract_id, False)
            time.sleep(args.monitor_interval_seconds)
        if "proposal_open_contract" not in latest:
            continue
        terminal = update_contract_state(state, contract_id, latest)
        logger.write("contract_update", contract_id=contract_id, terminal=terminal, raw_hash=raw_hash(latest))
        save_state(state_path, state)
    if sub_id:
        call_deriv(client, logger, args, "forget", client.forget, sub_id)
        logger.write("subscription_forgotten", contract_id=contract_id, subscription_id=sub_id)
    logger.write("contract_closed", contract_id=contract_id, raw_hash=raw_hash(latest))
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

    def score(self, feature_row: pd.Series | dict[str, Any]) -> FakeScore:
        return FakeScore(pair=self.pair, gate_reasons=[])


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
    def __init__(self, terminal: bool = True):
        super().__init__(is_demo=True)
        self.terminal = terminal

    def proposal(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append("proposal")
        return {"proposal": {"id": "proposal-1", "ask_price": 1.0, "payout": 2.0}}

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


def run_fake_client_smoke() -> None:
    print("Check: fake-client payloads, candle conversion, no-buy mode, monitoring, reconciliation, lock, EURUSD fail-closed")
    client = RecordingOptionsClient()
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

    with tempfile.TemporaryDirectory() as tmp:
        log_dir = Path(tmp)
        logger = JsonlLogger(log_dir)
        state_path = log_dir / "state.json"
        state = empty_state()
        args = SimpleNamespace(
            stake=1.0,
            max_breakeven=0.60,
            dry_run_proposal_only=True,
            demo_buy=False,
            max_trades_day=3,
            max_open=1,
            max_retries=0,
            retry_base_s=0.0,
            monitor_timeout_seconds=5.0,
            monitor_interval_seconds=0.01,
        )
        dry_client = FakeTradeClient()
        bought = process_pair(
            "USDJPY",
            FakeBook(),  # type: ignore[arg-type]
            dry_client,  # type: ignore[arg-type]
            FakeBuilder(),  # type: ignore[arg-type]
            logger,
            args,
            log_dir / "KILL",
            state,
            state_path,
            trades_today=0,
            open_count=0,
            now_utc=datetime(2026, 6, 15, 14, 0, tzinfo=timezone.utc),
        )
        if bought or "buy" in dry_client.calls or count_events(logger.path, "buy_confirmed"):
            raise ExecutorError("FAIL: proposal-only mode bought or emitted buy_confirmed")

        buy_args = SimpleNamespace(**{**args.__dict__, "dry_run_proposal_only": False, "demo_buy": True})
        buy_state = empty_state()
        buy_logger = JsonlLogger(log_dir / "buy")
        buy_client = FakeTradeClient()
        bought = process_pair(
            "USDJPY",
            FakeBook(),  # type: ignore[arg-type]
            buy_client,  # type: ignore[arg-type]
            FakeBuilder(),  # type: ignore[arg-type]
            buy_logger,
            buy_args,
            log_dir / "KILL",
            buy_state,
            log_dir / "buy_state.json",
            trades_today=0,
            open_count=0,
            now_utc=datetime(2026, 6, 15, 14, 0, tzinfo=timezone.utc),
        )
        if not bought or "forget" not in buy_client.calls or pending_contracts(buy_state):
            raise ExecutorError("FAIL: demo-buy monitor did not reach terminal state and forget subscription")

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

        with ExecutorLock(log_dir, "demo-account"):
            try:
                with ExecutorLock(log_dir, "demo-account"):
                    raise ExecutorError("FAIL: second executor lock unexpectedly acquired")
            except ExecutorError:
                pass
        try:
            resolve_pairs("EURUSD")
        except ExecutorError as exc:
            if "OF_*" not in str(exc):
                raise
        else:
            raise ExecutorError("FAIL: EURUSD did not fail closed")
    print("PASS: fake-client smoke covered payloads, candle conversion, no-buy, monitor, reconciliation, lock, EURUSD fail-closed")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ExecutorError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
