"""Shared Deriv runtime core (issue #4 Phase 3).

Gate/monitor/reconcile/state helpers, pair + floor resolution, and the fake
test-harness classes extracted verbatim from scripts/deriv_demo_executor.py so
the one-shot executor and the hot daemon share one implementation — parity by
construction. The executor re-imports everything here; its behaviour is
unchanged and covered by its own smokes (--fake-client-smoke, --check-books,
--replay-schema-check).

The only non-verbatim addition is `effective_floor_for` — the payout-gate
formula previously inlined in process_pair (effective_floor = side_floor -
haircut_abs - payout_edge_margin); both binaries must compute it through this
one function.
"""

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

from book_runtime import TARGET_BOOKS, LoadedBook
from deriv_backfill import DEFAULT_ENABLED_PAIRS
from deriv_client import DerivAPIError, DerivOptionsClient
from deriv_floor_resolver import (
    FloorResolutionError,
    PairFloorResolution,
    resolve_enabled_pairs,
)
from live_features import FeatureRow


def effective_floor_for(resolution: PairFloorResolution, direction: str, payout_edge_margin: float) -> float:
    """The authoritative payout-gate floor: side_refit_p10 - haircut - margin."""
    return resolution.side_floor(direction) - resolution.haircut_abs - payout_edge_margin


HORIZON_MINUTES = 15

ENABLED_PAIRS = list(DEFAULT_ENABLED_PAIRS)

EURUSD_DISABLED_REASON = "EURUSD disabled: verified live OF_* provider is unavailable"

TERMINAL_STATUSES = {"sold", "won", "lost", "expired", "cancelled"}

NY_TZ = ZoneInfo("America/New_York")

# A contract still "open" this long past its expected expiry is an anomaly
# (platform settlement stall / clock skew) — reconciliation fails closed on it
# rather than carrying unknown exposure forever (review F2).
OVERDUE_OPEN_GRACE_S = 600

# Shared issue-#6 cutoff, now evidence-validated by probe-g. The fold in
# results/json/deriv_api_probe_result.json pinned the boundary: last accepted
# start 16:34:40 NY, first rejection 16:35:10 NY. The (16, 35, 0) cutoff sits in
# that gap, so it admits the last accepted 16:34:40 start and blocks the first
# rejected 16:35:10 start, still failing closed before the 16:50 NY blackout can
# catch a 15m expiry.
LAST_START_NY = (16, 35, 0)
LAST_START_CUTOFF_SOURCE = (
    "results/json/deriv_api_probe_result.json decisions.last_start_cutoff_ny="
    "'last accepted start 16:34:40 NY; first rejection 16:35:10 NY'; "
    "probe-g-pinned issue-#6 cutoff (admits 16:34:40, blocks 16:35:10)"
)

DEFAULT_RUNTIME_LOCK_ROOT = Path("deriv_data/runtime/locks")
DEFAULT_ABSOLUTE_BREAKEVEN_CEILING = 0.60
ABSOLUTE_BREAKEVEN_CEILING_ENV = "DERIV_DEMO_ABSOLUTE_BREAKEVEN_CEILING"
DEPRECATED_MAX_BREAKEVEN_ENV = "DERIV_DEMO_MAX_BREAKEVEN"

class ExecutorError(RuntimeError):
    pass

def absolute_breakeven_ceiling_default(environ: dict[str, str] | None = None) -> float:
    """Resolve the secondary payout sanity ceiling from env, preserving the old name."""
    env = os.environ if environ is None else environ
    for key in (ABSOLUTE_BREAKEVEN_CEILING_ENV, DEPRECATED_MAX_BREAKEVEN_ENV):
        raw = env.get(key)
        if raw is None or str(raw).strip() == "":
            continue
        try:
            return float(raw)
        except ValueError as exc:
            raise ExecutorError(f"{key} must be numeric, got {raw!r}") from exc
    return DEFAULT_ABSOLUTE_BREAKEVEN_CEILING

def add_absolute_breakeven_ceiling_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--absolute-breakeven-ceiling",
        "--max-breakeven",
        dest="absolute_breakeven_ceiling",
        type=float,
        default=None,
        help=(
            "secondary absolute live breakeven ceiling; --max-breakeven is a "
            "deprecated alias, and the side-floor edge gate remains authoritative"
        ),
    )

def normalize_absolute_breakeven_ceiling(args: argparse.Namespace) -> argparse.Namespace:
    if args.absolute_breakeven_ceiling is None:
        args.absolute_breakeven_ceiling = absolute_breakeven_ceiling_default()
    return args

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

def last_start_cutoff_info() -> dict[str, Any]:
    return {
        "last_start_ny": f"{LAST_START_NY[0]:02d}:{LAST_START_NY[1]:02d}:{LAST_START_NY[2]:02d}",
        "reject_rule": "reject when NY wall-clock >= last_start_ny",
        "source": LAST_START_CUTOFF_SOURCE,
    }

def after_last_start_cutoff(ts_utc: datetime) -> bool:
    if ts_utc.tzinfo is None:
        ts_utc = ts_utc.replace(tzinfo=timezone.utc)
    ny = ts_utc.astimezone(NY_TZ)
    return (ny.hour, ny.minute, ny.second) >= LAST_START_NY

def account_lock_key(account_id: str) -> str:
    raw = str(account_id).strip()
    if not raw:
        raise ExecutorError("DERIV_ACCOUNT_ID is required for an account-scoped buy lock")
    return f"deriv-demo-account:{raw}"

def canonical_lock_root(root: Path | str | None = None) -> Path:
    """Create/verify the shared account lock root for every buy-capable runtime.

    Lock identity is (canonical root, raw account key). Do not pre-hash the key:
    ExecutorLock hashes it once for the filename.
    """
    selected = root or os.environ.get("DERIV_RUNTIME_LOCK_ROOT") or DEFAULT_RUNTIME_LOCK_ROOT
    path = Path(selected)
    if path.exists() and path.is_symlink():
        raise ExecutorError(f"lock root is a symlink, refusing: {path}")
    path.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ExecutorError(f"lock root is a symlink, refusing: {path}")
    st = path.stat()
    if st.st_uid != os.getuid():
        raise ExecutorError(f"lock root is not owned by current user: {path}")
    mode = st.st_mode & 0o777
    if mode != 0o700:
        path.chmod(0o700)
        mode = path.stat().st_mode & 0o777
    if mode != 0o700:
        raise ExecutorError(f"lock root is not 0700: {path} mode={oct(mode)}")
    return path

def executor_lock_path(lock_root: Path | str, raw_account_key: str) -> Path:
    digest = hashlib.sha256(raw_account_key.encode("utf-8")).hexdigest()[:12]
    return Path(lock_root) / f"executor_{digest}.lock"

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
    # Fail-closed on UNKNOWN state (a query failure raises via call_deriv);
    # KNOWN-open contracts are normal with unconstrained concurrency — they are
    # refreshed and carried, never a reason to refuse trading.
    unresolved: list[str] = []
    still_open: list[str] = []
    for contract_id in list(pending_contracts(state)):
        try:
            resp = call_deriv(client, logger, args, "proposal_open_contract", client.proposal_open_contract, contract_id, False)
        except ExecutorError:
            unresolved.append(contract_id)
            continue
        terminal = update_contract_state(state, contract_id, resp)
        logger.write("state_reconciled_contract", contract_id=contract_id, terminal=terminal, raw_hash=raw_hash(resp))
        if not terminal:
            expiry_raw = (state.get("open_contracts", {}).get(contract_id) or {}).get("expected_expiry_utc")
            overdue = True  # missing/unparseable expiry on an open contract is itself anomalous
            if expiry_raw:
                try:
                    overdue = datetime.now(timezone.utc) > (
                        datetime.fromisoformat(expiry_raw) + timedelta(seconds=OVERDUE_OPEN_GRACE_S))
                except ValueError:
                    pass
            if overdue:
                logger.write("reconcile_overdue_open", contract_id=contract_id, expected_expiry_utc=expiry_raw)
                unresolved.append(contract_id)
            else:
                still_open.append(contract_id)
    save_state(state_path, state)
    if still_open:
        logger.write("reconcile_carried_open", contracts=still_open)
    if unresolved:
        logger.write("reconcile_failed", reason="unresolved_open_contracts", unresolved=unresolved)
        raise ExecutorError(f"open local state UNRESOLVABLE against Deriv (query failed): {unresolved}")

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
        "absolute_breakeven_ceiling": DEFAULT_ABSOLUTE_BREAKEVEN_CEILING,
        "payout_edge_margin": 0.005,
        "quote_snapshots": "all",
        "dry_run_proposal_only": True,
        "demo_buy": False,
        "max_trades_day": 3,
        "max_open": 1,
        "pair_cooldown_seconds": 900,
        "blocking_monitor": True,
        "max_retries": 0,
        "retry_base_s": 0.0,
        "monitor_timeout_seconds": 5.0,
        "monitor_interval_seconds": 0.01,
    }
    base.update(overrides)
    return SimpleNamespace(**base)
