"""One-command VPS readiness & read-only Deriv Options/v3 proposal smoke check.

Never sends `buy`. Verifies:
1. Python runtime & required numerical/WebSocket libraries.
2. Absence of legacy buy-capable processes (`deriv_no_legacy_buy_guard.py`) and `KILL` switch.
3. Account lock directory (`0700` user-owned) and SQLite WAL durability on queue DB.
4. `books/SOTA_INDEX.json` + SHA-256, content_id, and G1-G6 certificates of all enabled books.
5. Optional `--live-smoke`: verifies `DERIV_APP_ID`, `DERIV_PAT`, `DERIV_ACCOUNT_ID`, fail-closed
   account mode (`demo` VRTC vs `live` CR + `DERIV_ALLOW_LIVE_REAL_MONEY=1`), obtains an
   authenticated WS connection (acquiring `/mnt/data/deriv/.connection.lock` non-blocking if
   present on a research host), requests a $1.00 read-only `proposal` per enabled book, and
   evaluates `evaluate_sota_payout_gate`.
"""

from __future__ import annotations

import argparse
import asyncio
import fcntl
import importlib
import json
import math
import os
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from deriv_async_client import DerivAsyncClient
from deriv_client import DerivEnv
from deriv_no_legacy_buy_guard import offender, cmdline
from deriv_runtime_core_v2 import (
    SOTA_DERIV_SYMBOLS,
    assert_non_us_deriv_host,
    canonical_lock_root,
    contract_type_for,
    evaluate_sota_multiplier_gate,
    evaluate_sota_payout_gate,
    finite_positive_number,
    format_barrier_levels,
    validate_account_mode,
)
from deriv_sota_executor import (
    connect_sota_queue,
    default_queue_db_path,
    request_authenticated_ws_url,
)
from sota_book_runtime import REPO_ROOT, load_enabled_sota_books, load_sota_index, parse_book_ids_spec

LOCAL_HOST_CONN_LOCK = Path("/mnt/data/deriv/.connection.lock")


@contextmanager
def maybe_host_connection_lock() -> Iterator[None]:
    """Respect `/mnt/data/deriv/.connection.lock` when running on the research host."""
    if not LOCAL_HOST_CONN_LOCK.exists():
        yield
        return
    fd = os.open(str(LOCAL_HOST_CONN_LOCK), os.O_RDWR)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"host connection lock busy: {LOCAL_HOST_CONN_LOCK}") from exc
        yield
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def check_no_legacy_buyers() -> None:
    self_pid = os.getpid()
    for item in Path("/proc").iterdir():
        if item.name.isdigit() and int(item.name) != self_pid:
            parts = cmdline(item)
            is_v1_trade_exec = any("deriv_trade_executor.py" in tok for tok in parts) and (
                "--run" in parts or "--once" in parts
            )
            if offender(parts) or is_v1_trade_exec:
                raise RuntimeError(f"legacy buy-capable process running (pid={item.name}): {' '.join(parts)}")


async def run_network_proposal_smoke(account_mode: str, books: list[Any], stake: float) -> list[dict[str, Any]]:
    assert_non_us_deriv_host()
    env = DerivEnv.from_env()
    ws_url, transport_kind = await asyncio.to_thread(request_authenticated_ws_url, env, account_mode)
    validate_account_mode(account_mode=account_mode, account_id=env.account_id, ws_url=ws_url)
    results: list[dict[str, Any]] = []
    async with DerivAsyncClient(ws_url, silence_timeout_s=20.0) as client:
        if transport_kind == "v3_authorize":
            try:
                auth_resp = await client.request({"authorize": env.pat}, timeout=10.0)
            except Exception as exc:
                safe_msg = str(exc).replace(env.pat, "[REDACTED_PAT]") if env.pat else str(exc)
                raise RuntimeError(f"v3_authorize failed: {safe_msg}") from None
            info = auth_resp.get("authorize") or {}
            loginid = str(info.get("loginid") or "").strip()
            if not loginid or loginid != env.account_id:
                raise RuntimeError(f"PAT authorized loginid={loginid!r} != DERIV_ACCOUNT_ID {env.account_id!r}")
            if str(info.get("currency") or "").strip().upper() != "USD":
                raise RuntimeError(f"account currency must be USD, got {info.get('currency')!r}")
            is_virt = bool(info.get("is_virtual"))
            if account_mode == "live" and is_virt:
                raise RuntimeError("live mode refused: token authorized virtual account")
            if account_mode == "demo" and not is_virt:
                raise RuntimeError("demo mode refused: token authorized real-money account")
        else:
            bal_resp = await client.request({"balance": 1}, timeout=10.0)
            binfo = bal_resp.get("balance") or {}
            loginid = str(binfo.get("loginid") or "").strip()
            if loginid and loginid != env.account_id:
                raise RuntimeError(f"OTP socket loginid={loginid!r} != DERIV_ACCOUNT_ID {env.account_id!r}")
            if str(binfo.get("currency") or "").strip().upper() != "USD":
                raise RuntimeError(f"OTP account currency must be USD, got {binfo.get('currency')!r}")

        sym_key = "symbol" if transport_kind == "v3_authorize" else "underlying_symbol"
        for b in books:
            side = b.allowed_sides[0]
            bct = b.barrier_contract_type
            ctype = contract_type_for(side, family=b.contract_family, barrier_contract_type=bct)
            try:
                if b.contract_family == "multiplier":
                    mult = int(b.multiplier or 100)
                    req: dict[str, Any] = {
                        "proposal": 1,
                        "amount": float(stake),
                        "basis": "stake",
                        "contract_type": ctype,
                        "currency": "USD",
                        "multiplier": mult,
                        sym_key: SOTA_DERIV_SYMBOLS[b.pair],
                    }
                elif b.contract_family == "barrier_1d":
                    spot_probe = await client.request(
                        {
                            "proposal": 1,
                            "amount": float(stake),
                            "basis": "stake",
                            "contract_type": "CALL",
                            "currency": "USD",
                            "duration": 15,
                            "duration_unit": "m",
                            sym_key: SOTA_DERIV_SYMBOLS[b.pair],
                        },
                        timeout=8.0,
                    )
                    spot_now = finite_positive_number((spot_probe.get("proposal") or {}).get("spot"))
                    if spot_now is None:
                        raise RuntimeError(f"{b.book_id}: spot probe returned no valid positive spot")
                    hi_s, lo_s = format_barrier_levels(
                        b.pair,
                        spot_now,
                        float(b.barrier_rel_half_width or 0.005),
                        ctype,
                        side=side,
                    )
                    req = {
                        "proposal": 1,
                        "amount": float(stake),
                        "basis": "stake",
                        "contract_type": ctype,
                        "currency": "USD",
                        "duration": 1,
                        "duration_unit": "d",
                        "barrier": hi_s,
                        sym_key: SOTA_DERIV_SYMBOLS[b.pair],
                    }
                    if lo_s is not None:
                        req["barrier2"] = lo_s
                else:
                    req = {
                        "proposal": 1,
                        "amount": float(stake),
                        "basis": "stake",
                        "contract_type": ctype,
                        "currency": "USD",
                        "duration": int(b.horizon_minutes),
                        "duration_unit": "m",
                        sym_key: SOTA_DERIV_SYMBOLS[b.pair],
                    }
                resp = await client.request(req, timeout=8.0)
                prop = resp.get("proposal") or {}
                if b.contract_family == "multiplier":
                    exp_ev = (float(b.ev_floors[side]) - float(b.payout_edge_margin)) if (b.ev_floors and side in b.ev_floors) else 0.01
                    mdec = evaluate_sota_multiplier_gate(
                        prop,
                        expected_ev=exp_ev,
                        side=side,
                        certified_commission_frac=b.certified_commission_frac,
                        max_commission_frac=b.max_commission_frac,
                        min_stop_out_rel_dist=b.min_stop_out_rel_dist,
                    )
                    structural_err = mdec.reason in {
                        "missing_or_invalid_ask",
                        "missing_or_invalid_spot",
                        "missing_or_invalid_stop_out",
                        "invalid_stop_out_side",
                        "missing_or_invalid_commission",
                        "non_usd_proposal_currency",
                    }
                    quote_ok = (
                        bool(prop.get("id"))
                        and not structural_err
                        and mdec.ask is not None
                        and mdec.stop_out_rel_dist is not None
                        and math.isfinite(mdec.stop_out_rel_dist)
                    )
                    results.append({
                        "book_id": b.book_id,
                        "pair": b.pair,
                        "horizon_minutes": b.horizon_minutes,
                        "side": side,
                        "proposal_received": bool(quote_ok),
                        "ask": mdec.ask,
                        "commission": mdec.commission,
                        "stop_out_rel_dist": mdec.stop_out_rel_dist,
                        "gate_passed_at_floor": mdec.passed,
                        "gate_reason": mdec.reason,
                    })
                else:
                    dec = evaluate_sota_payout_gate(
                        prop,
                        effective_floor=b.effective_floor(side),
                        absolute_breakeven_ceiling=0.60,
                        calibrated_win_prob=b.side_floor(side),
                        ev_margin=0.005,
                        min_payout_ratio=b.min_payout_ratio,
                    )
                    quote_ok = (
                        bool(prop.get("id"))
                        and dec.ask is not None
                        and dec.payout is not None
                        and dec.payout_ratio is not None
                        and math.isfinite(dec.payout_ratio)
                        and dec.payout_ratio > 1.0
                        and dec.live_breakeven is not None
                        and math.isfinite(dec.live_breakeven)
                        and dec.reason != "non_usd_proposal_currency"
                    )
                    results.append({
                        "book_id": b.book_id,
                        "pair": b.pair,
                        "horizon_minutes": b.horizon_minutes,
                        "side": side,
                        "proposal_received": bool(quote_ok),
                        "ask": dec.ask,
                        "payout": dec.payout,
                        "payout_ratio": dec.payout_ratio,
                        "live_breakeven": dec.live_breakeven,
                        "gate_passed_at_floor": dec.passed,
                        "gate_reason": dec.reason,
                    })
            except Exception as exc:
                results.append({
                    "book_id": b.book_id,
                    "pair": b.pair,
                    "horizon_minutes": b.horizon_minutes,
                    "side": side,
                    "proposal_received": False,
                    "error": str(exc)[:240],
                })
            await asyncio.sleep(2.6)
    return results


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--account-mode", choices=("demo", "live"), default=os.environ.get("DERIV_ACCOUNT_MODE", "demo"))
    ap.add_argument("--book-ids", default=os.environ.get("DERIV_SOTA_BOOK_IDS", ""))
    ap.add_argument("--offline-only", action="store_true")
    ap.add_argument("--live-smoke", action="store_true")
    ap.add_argument("--allow-empty-index", action="store_true")
    ap.add_argument("--repo-root", default=str(REPO_ROOT))
    ap.add_argument("--log-dir", default=os.environ.get("DERIV_SOTA_LOG_DIR", "logs/sota_runtime"))
    ap.add_argument("--lock-root", default=os.environ.get("DERIV_RUNTIME_LOCK_ROOT", ""))
    ap.add_argument("--queue-db", default=os.environ.get("DERIV_SOTA_QUEUE_DB", ""))
    ap.add_argument("--stake", type=float, default=float(os.environ.get("DERIV_SOTA_STAKE", "1.0")))
    args = ap.parse_args()

    repo_root = Path(args.repo_root).resolve()
    checks: list[dict[str, Any]] = []
    failed = False

    def record(name: str, ok: bool, **detail: Any) -> None:
        nonlocal failed
        if not ok:
            failed = True
        checks.append({"check": name, "passed": ok, **detail})

    # 1. Python & required libs
    missing_libs = []
    for mod in ("numpy", "pandas", "pyarrow", "lightgbm", "websockets", "websocket"):
        try:
            importlib.import_module(mod)
        except Exception as exc:
            missing_libs.append(f"{mod}:{exc}")
    record("python_dependencies", len(missing_libs) == 0, python=sys.version.split()[0], missing=missing_libs)

    # 2. Legacy buyer & KILL switch check
    try:
        check_no_legacy_buyers()
        record("no_legacy_buyers", True)
    except Exception as exc:
        record("no_legacy_buyers", False, error=str(exc))

    kill_path = Path(args.log_dir) / "KILL"
    record("kill_switch_absent", not kill_path.exists(), kill_path=str(kill_path))

    # 3. Lock root & SQLite WAL verification
    try:
        lroot = canonical_lock_root(Path(args.lock_root) if args.lock_root else None)
        record("lock_root_0700", True, path=str(lroot))
    except Exception as exc:
        record("lock_root_0700", False, error=str(exc))

    try:
        if args.queue_db:
            qpath = Path(args.queue_db)
            conn = connect_sota_queue(qpath)
            conn.close()
            record("sqlite_wal_queue", True, db_path=str(qpath))
        else:
            with tempfile.TemporaryDirectory() as td:
                qpath = default_queue_db_path(args.account_mode, runtime_dir=Path(td))
                conn = connect_sota_queue(qpath)
                conn.close()
                record("sqlite_wal_queue", True, db_path="ephemeral_wal_verified")
    except Exception as exc:
        record("sqlite_wal_queue", False, error=str(exc))

    # 4. SOTA book index & cryptographic certificate verification
    books = []
    try:
        idx = load_sota_index(repo_root=repo_root)
        req_book_ids = parse_book_ids_spec(args.book_ids)
        books = load_enabled_sota_books(repo_root=repo_root, account_mode=args.account_mode, book_ids=req_book_ids)
        ok_books = len(books) > 0 or bool(args.allow_empty_index)
        record(
            "certified_sota_books",
            ok_books,
            registered_total=len(idx),
            enabled_for_mode=len(books),
            book_ids=[b.book_id for b in books],
        )
    except Exception as exc:
        record("certified_sota_books", False, error=str(exc))

    # 5. Optional read-only network proposal smoke (never buys)
    if args.live_smoke and not args.offline_only and not failed:
        try:
            with maybe_host_connection_lock():
                smoke = asyncio.run(run_network_proposal_smoke(args.account_mode, books, args.stake))
            all_rx = all(x.get("proposal_received") for x in smoke)
            record("network_proposal_smoke", all_rx, proposals=smoke)
        except Exception as exc:
            record("network_proposal_smoke", False, error=str(exc))

    report = {
        "schema": "deriv-sota-vps-preflight/v1",
        "account_mode": args.account_mode,
        "passed": not failed,
        "checks": checks,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
