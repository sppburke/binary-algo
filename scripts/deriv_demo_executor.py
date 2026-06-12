"""Demo-buy executor for the seven certified 15m Deriv FX books."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import sessions
from book_runtime import TARGET_BOOKS, BookRuntimeError, LoadedBook, book_inventory, load_target_books
from deriv_client import DerivAPIError, DerivEnv, DerivOptionsClient
from live_features import DERIV_SYMBOLS, LiveFeatureBuilder, LiveFeatureError, latest_replay_row, pair_symbol


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG_DIR = REPO_ROOT / "logs" / "paper_trades"
HORIZON_MINUTES = 15


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


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"cannot JSON encode {type(obj).__name__}")


def raw_hash(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:16]


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pairs", default=",".join(TARGET_BOOKS), help="comma-separated pair list")
    p.add_argument("--stake", type=float, default=1.0)
    p.add_argument("--max-trades-day", type=int, default=3)
    p.add_argument("--max-open", type=int, default=1)
    p.add_argument("--dry-run-proposal-only", action="store_true", help="request proposals but do not buy")
    p.add_argument("--demo-buy", action="store_true", help="submit demo-account buys; never enabled by default")
    p.add_argument("--once", action="store_true")
    p.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    p.add_argument("--history-minutes", type=int, default=14000)
    p.add_argument("--tick-volume-count", type=int, default=5000)
    p.add_argument("--max-breakeven", type=float, default=0.60)
    p.add_argument("--check-books", action="store_true")
    p.add_argument("--replay-schema-check", action="store_true")
    p.add_argument("--public-smoke", action="store_true")
    p.add_argument("--auth-smoke", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    pairs = [p.strip().upper() for p in args.pairs.split(",") if p.strip()]
    unknown = [p for p in pairs if p not in TARGET_BOOKS]
    if unknown:
        raise ExecutorError(f"unsupported pairs: {unknown}")
    logger = JsonlLogger(args.log_dir)
    logger.write("startup_config", pairs=pairs, stake=args.stake, dry_run=args.dry_run_proposal_only, demo_buy=args.demo_buy)

    books = load_target_books(pairs)
    for inv in book_inventory(pairs):
        logger.write("book_loaded", **inv)
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
    if kill_path.exists():
        logger.write("kill_switch_triggered", path=str(kill_path))
        raise ExecutorError(f"kill switch present: {kill_path}")

    client = DerivOptionsClient.demo_from_env() if args.demo_buy else DerivOptionsClient.public()
    if args.demo_buy and not client.is_demo:
        raise ExecutorError("refusing buy: authenticated WebSocket URL is not demo")
    with client:
        verify_contracts(client, pairs, logger)
        builder = LiveFeatureBuilder(client, history_minutes=args.history_minutes, tick_volume_count=args.tick_volume_count)
        trades_today = count_events(logger.path, "buy_confirmed")
        open_count = 0
        for pair in pairs:
            bought = process_pair(pair, books[pair], client, builder, logger, args, kill_path, trades_today, open_count)
            if bought:
                trades_today += 1
            if args.once:
                break
    return 0


def run_replay_schema_check(books: dict[str, LoadedBook]) -> None:
    print("Check: replay schema and model scoring for selected books")
    for pair, book in books.items():
        row = latest_replay_row(pair, book.feature_cols)
        score = book.score(row.row)
        print(
            f"PASS: {pair} {book.book_id} replay row {row.timestamp} "
            f"cols={len(book.feature_cols)} proba={score.proba:.8f} conf={score.confidence:.8f}"
        )


def run_public_smoke(pairs: list[str], books: dict[str, LoadedBook], logger: JsonlLogger, args: argparse.Namespace) -> None:
    print("Check: public market-data smoke, contract availability, and live feature schema")
    with DerivOptionsClient.public() as client:
        verify_contracts(client, pairs, logger)
        builder = LiveFeatureBuilder(client, history_minutes=args.history_minutes, tick_volume_count=args.tick_volume_count)
        for pair in pairs:
            row = builder.feature_row(pair, books[pair].feature_cols)
            score = books[pair].score(row.row)
            print(
                f"PASS: {pair} live row {row.timestamp} cols={len(books[pair].feature_cols)} "
                f"proba={score.proba:.8f} conf={score.confidence:.8f} gate={score.gate_passed}"
            )


def run_auth_smoke() -> None:
    print("Check: env/auth/demo assertion without printing secrets")
    env = DerivEnv.from_env()
    client = DerivOptionsClient.demo_from_env()
    if not client.is_demo:
        raise ExecutorError("FAIL: OTP URL is not demo")
    print(f"PASS: DERIV_APP_ID set, DERIV_PAT set, DERIV_ACCOUNT_ID={env.account_id}, demo OTP URL asserted")


def verify_contracts(client: DerivOptionsClient, pairs: list[str], logger: JsonlLogger) -> None:
    for pair in pairs:
        symbol = pair_symbol(pair)
        resp = client.contracts_for(symbol)
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


def process_pair(
    pair: str,
    book: LoadedBook,
    client: DerivOptionsClient,
    builder: LiveFeatureBuilder,
    logger: JsonlLogger,
    args: argparse.Namespace,
    kill_path: Path,
    trades_today: int,
    open_count: int,
) -> bool:
    if kill_path.exists():
        logger.write("kill_switch_triggered", path=str(kill_path))
        return False
    row = builder.feature_row(pair, book.feature_cols)
    score = book.score(row.row)
    logger.write(
        "feature_row_ready",
        pair=pair,
        symbol=DERIV_SYMBOLS[pair],
        book_id=book.book_id,
        feature_timestamp=row.timestamp,
        warnings=row.warnings,
        proba=score.proba,
        conf=score.confidence,
        threshold=score.threshold,
        direction=score.direction,
        gate_passed=score.gate_passed,
        gate_reasons=score.gate_reasons,
    )
    ts_s = int(row.timestamp.timestamp())
    if not bool(sessions.session_mask([ts_s], "ny")[0]):
        logger.write("signal_skipped", pair=pair, reason="outside_ny_session", feature_timestamp=row.timestamp)
        return False
    if not score.gate_passed:
        logger.write("signal_skipped", pair=pair, reason="strategy_gate", details=score.gate_reasons)
        return False
    if trades_today >= args.max_trades_day:
        logger.write("signal_skipped", pair=pair, reason="max_trades_day", max_trades_day=args.max_trades_day)
        return False
    if open_count >= args.max_open:
        logger.write("signal_skipped", pair=pair, reason="max_open", max_open=args.max_open)
        return False
    last_buy = last_pair_buy_seconds(logger.path, pair)
    if last_buy is not None and time.time() - last_buy < HORIZON_MINUTES * 60:
        logger.write("signal_skipped", pair=pair, reason="pair_cooldown", last_buy_epoch=last_buy)
        return False

    contract_type = "CALL" if score.direction == "UP" else "PUT"
    symbol = DERIV_SYMBOLS[pair]
    if kill_path.exists():
        logger.write("kill_switch_triggered", path=str(kill_path))
        return False
    logger.write("proposal_requested", pair=pair, symbol=symbol, contract_type=contract_type, stake=args.stake)
    proposal = client.proposal(symbol=symbol, contract_type=contract_type, amount=args.stake, duration=HORIZON_MINUTES)
    prop = proposal.get("proposal", {})
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
        proposal_id=prop.get("id"),
        ask=ask,
        payout=payout,
        implied_breakeven=breakeven,
        raw_hash=raw_hash(proposal),
    )
    if breakeven > args.max_breakeven:
        logger.write("signal_skipped", pair=pair, reason="breakeven_limit", implied_breakeven=breakeven)
        return False
    if args.dry_run_proposal_only:
        logger.write("signal_skipped", pair=pair, reason="dry_run_proposal_only")
        return False
    if not args.demo_buy:
        logger.write("signal_skipped", pair=pair, reason="demo_buy_not_enabled")
        return False
    if kill_path.exists():
        logger.write("kill_switch_triggered", path=str(kill_path))
        return False
    proposal_id = prop.get("id")
    if not proposal_id:
        raise ExecutorError("proposal response lacks buyable proposal id")
    logger.write("buy_submitted", pair=pair, proposal_id=proposal_id, price=ask)
    buy = client.buy(str(proposal_id), ask)
    bought = buy.get("buy", {})
    contract_id = bought.get("contract_id")
    logger.write("buy_confirmed", pair=pair, contract_id=contract_id, raw_hash=raw_hash(buy))
    if contract_id is None:
        raise ExecutorError("buy response lacks contract_id")
    monitor_contract(client, logger, pair, contract_id)
    return True


def monitor_contract(client: DerivOptionsClient, logger: JsonlLogger, pair: str, contract_id: str | int) -> None:
    resp = client.proposal_open_contract(contract_id, subscribe=True)
    sub_id = resp.get("subscription", {}).get("id")
    deadline = time.monotonic() + 25 * 60
    while True:
        payload = resp if resp.get("proposal_open_contract") else client.recv(timeout=30.0)
        poc = payload.get("proposal_open_contract", {})
        status = poc.get("status")
        logger.write("contract_update", pair=pair, contract_id=contract_id, platform_status=status, profit_loss=poc.get("profit"), raw_hash=raw_hash(payload))
        if status in {"won", "lost", "sold", "expired"} or poc.get("is_expired") or poc.get("is_sold"):
            logger.write("contract_closed", pair=pair, contract_id=contract_id, platform_status=status, profit_loss=poc.get("profit"), raw_hash=raw_hash(payload))
            if sub_id:
                client.forget(sub_id)
            return
        if time.monotonic() > deadline:
            raise ExecutorError(f"proposal_open_contract did not reach terminal status for {contract_id}")
        resp = {}


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


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ExecutorError, BookRuntimeError, DerivAPIError, LiveFeatureError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
