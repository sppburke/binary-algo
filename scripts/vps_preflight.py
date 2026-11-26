"""One-command VPS readiness & read-only Deriv Options/v3 proposal smoke check (Phase 1 hardened).

Never sends `buy` or `sell` (`SotaDerivAsyncClient(..., allow_buy=False)`).
Supports:
- Mutually exclusive `--offline-only` and `--live-smoke` (spec §3.6).
- Non-bypassable `resolve_and_validate_sota_runtime_config()` under `--phase1-policy`.
- Test-only `--fixture-home` + `--fixture-repo-root` (admitted strictly on `--offline-only` and
  verified outside real `<PW_HOME>`, `/home/sean/git/binary-algo*`, and `/mnt/data/deriv*`;
  unconditionally refused on `--live-smoke`).
- Strict per-book `--live-smoke` quote-floor success predicate (exact rational `1.76`/`1.78`
  defeating `1.7799999999`, exact `100`-cent stake, `USD`, `effective_floor`, reporting
  `row_calibrated_evidence: false, decision5_satisfied: false`).
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
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from deriv_client import DerivEnv
from deriv_no_legacy_buy_guard import cmdline, offender
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
    SotaDerivAsyncClient,
    acquire_ordered_locks,
    check_exact_rational_quote_floor,
    connect_sota_queue,
    default_queue_db_path,
    is_safety_latch_set,
    open_sota_queue_readonly,
    parse_exact_whole_cents,
    request_authenticated_ws_url,
    resolve_and_validate_sota_runtime_config,
    verify_v3_queue_binding,
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


def evaluate_live_smoke_proposal_result(
    book: Any,
    prop: dict[str, Any],
    *,
    expected_stake_cents: int = 100,
    absolute_breakeven_ceiling: float = 0.60,
    ev_margin: float = 0.005,
) -> dict[str, Any]:
    """Strict per-book --live-smoke success predicate (spec §3.6 + B9)."""
    side = book.allowed_sides[0]
    if book.contract_family == "multiplier":
        exp_ev = (float(book.ev_floors[side]) - float(book.payout_edge_margin)) if (book.ev_floors and side in book.ev_floors) else 0.01
        mdec = evaluate_sota_multiplier_gate(
            prop,
            expected_ev=exp_ev,
            side=side,
            certified_commission_frac=book.certified_commission_frac,
            max_commission_frac=book.max_commission_frac,
            min_stop_out_rel_dist=book.min_stop_out_rel_dist,
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
        return {
            "book_id": book.book_id,
            "pair": book.pair,
            "horizon_minutes": book.horizon_minutes,
            "side": side,
            "proposal_received": bool(quote_ok),
            "ask": mdec.ask,
            "commission": mdec.commission,
            "stop_out_rel_dist": mdec.stop_out_rel_dist,
            "gate_passed_at_floor": bool(mdec.passed),
            "gate_reason": mdec.reason,
            "smoke_passed": bool(quote_ok and mdec.passed),
            "row_calibrated_evidence": False,
            "decision5_satisfied": False,
        }

    # Exact rational quote-floor check (rejects 1.7799999999 vs 1.78!) + exact 100-cent stake + USD!
    ask_f, pay_f, rat_f, rat_err = check_exact_rational_quote_floor(prop, min_payout_ratio=book.min_payout_ratio)
    stake_ok = False
    if ask_f is not None:
        try:
            stake_ok = parse_exact_whole_cents(ask_f, allow_zero=False, allow_negative=False) == int(expected_stake_cents)
        except Exception:
            stake_ok = False

    dec = evaluate_sota_payout_gate(
        prop,
        effective_floor=book.effective_floor(side),
        absolute_breakeven_ceiling=absolute_breakeven_ceiling,
        calibrated_win_prob=book.side_floor(side),
        ev_margin=ev_margin,
        min_payout_ratio=book.min_payout_ratio,
    )
    has_id = bool(str(prop.get("id") or "").strip())
    # MAJOR #3: Require explicit USD currency evidence (reject missing/None currency!)
    ccy_ok = str(prop.get("currency") or "").strip().upper() == "USD"
    quote_ok = (
        has_id
        and ccy_ok
        and dec.ask is not None
        and dec.payout is not None
        and dec.payout_ratio is not None
        and math.isfinite(dec.payout_ratio)
        and dec.payout_ratio > 1.0
        and dec.live_breakeven is not None
        and math.isfinite(dec.live_breakeven)
        and dec.reason != "non_usd_proposal_currency"
    )
    gate_ok = bool(ccy_ok and rat_err is None and stake_ok and dec.passed)
    reason = (None if ccy_ok else "missing_or_non_usd_proposal_currency") or rat_err or (None if stake_ok else "ask_price_mismatch_configured_stake") or dec.reason
    return {
        "book_id": book.book_id,
        "pair": book.pair,
        "horizon_minutes": book.horizon_minutes,
        "side": side,
        "proposal_received": bool(quote_ok),
        "ask": dec.ask,
        "payout": dec.payout,
        "payout_ratio": dec.payout_ratio,
        "live_breakeven": dec.live_breakeven,
        "gate_passed_at_floor": gate_ok,
        "gate_reason": reason,
        "smoke_passed": bool(quote_ok and gate_ok),
        "row_calibrated_evidence": False,
        "decision5_satisfied": False,
    }


async def run_network_proposal_smoke(
    account_mode: str,
    books: list[Any],
    stake: float,
    *,
    absolute_breakeven_ceiling: float = 0.60,
    ev_margin: float = 0.005,
    handoff_lock_path: Path | None = None,
    queue_db_path: Path | None = None,
    kill_path: Path | None = None,
    smoke_timeout_s: float = 10.0,
    client_override: SotaDerivAsyncClient | None = None,
    transport_kind_override: str | None = None,
    env_override: DerivEnv | None = None,
    pace_s: float = 2.6,
) -> list[dict[str, Any]]:
    """Read-only proposal smoke through SotaDerivAsyncClient(allow_buy=False) under ONE bounded deadline covering OTP -> teardown."""
    expected_stake_cents = parse_exact_whole_cents(stake, allow_zero=False, allow_negative=False)

    def _smoke_handoff_guard(_primary: str, _payload: dict[str, Any]) -> None:
        if kill_path is not None and kill_path.exists():
            raise RuntimeError("live-smoke aborted under handoff lock: KILL present")
        if queue_db_path is not None and queue_db_path.is_file():
            ro_g = open_sota_queue_readonly(queue_db_path)
            try:
                if is_safety_latch_set(ro_g) is not None:
                    raise RuntimeError("live-smoke aborted under handoff lock: phase1_safety_latch active")
            finally:
                ro_g.close()

    deadline_mono = time.monotonic() + float(smoke_timeout_s)

    async def _run_inner() -> list[dict[str, Any]]:
        if client_override is None:
            assert_non_us_deriv_host()
            env = env_override or DerivEnv.from_env()
            ws_url, transport_kind = await asyncio.to_thread(request_authenticated_ws_url, env, account_mode)
            validate_account_mode(account_mode=account_mode, account_id=env.account_id, ws_url=ws_url)
            client = SotaDerivAsyncClient(
                ws_url,
                allow_buy=False,
                expected_account_id=env.account_id,
                handoff_lock_path=handoff_lock_path,
                pre_send_guard_fn=_smoke_handoff_guard,
                silence_timeout_s=20.0,
            )
            owns_ctx = True
        else:
            env = env_override or DerivEnv(app_id="test", pat="test", account_id=client_override.expected_account_id or "VRTC999999")
            transport_kind = transport_kind_override or "options_otp"
            client = client_override
            if client.allow_buy:
                raise RuntimeError("live smoke client MUST have allow_buy=False")
            owns_ctx = False

        results: list[dict[str, Any]] = []
        try:
            if owns_ctx:
                await client.connect()
            if not client.is_identity_verified():
                await client.establish_identity(
                    account_id=env.account_id,
                    account_mode=account_mode,
                    transport_kind=transport_kind,
                    pat=env.pat,
                    timeout=min(10.0, smoke_timeout_s),
                )
            sym_key = "symbol" if transport_kind == "v3_authorize" else "underlying_symbol"
            for idx_b, b in enumerate(books):
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
                        spot_probe = await client.request_proposal(
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
                            timeout=min(8.0, smoke_timeout_s),
                        )
                        spot_now = finite_positive_number((spot_probe.get("proposal") or {}).get("spot"))
                        if spot_now is None:
                            raise RuntimeError(f"{b.book_id}: spot probe returned no valid positive spot")
                        hi_s, lo_s = format_barrier_levels(
                            b.pair, spot_now, float(b.barrier_rel_half_width or 0.005), ctype, side=side
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
                    resp = await client.request_proposal(req, timeout=min(8.0, smoke_timeout_s))
                    prop = resp.get("proposal") or {}
                    results.append(
                        evaluate_live_smoke_proposal_result(
                            b,
                            prop,
                            expected_stake_cents=expected_stake_cents,
                            absolute_breakeven_ceiling=absolute_breakeven_ceiling,
                            ev_margin=ev_margin,
                        )
                    )
                except Exception as exc:
                    results.append(
                        {
                            "book_id": b.book_id,
                            "pair": b.pair,
                            "horizon_minutes": b.horizon_minutes,
                            "side": side,
                            "proposal_received": False,
                            "gate_passed_at_floor": False,
                            "smoke_passed": False,
                            "row_calibrated_evidence": False,
                            "decision5_satisfied": False,
                            "error": str(exc)[:240],
                        }
                    )
                if idx_b + 1 < len(books) and pace_s > 0:
                    await asyncio.sleep(pace_s)
        finally:
            if owns_ctx:
                rem_close = max(0.0, deadline_mono - time.monotonic())
                if rem_close > 0.005:
                    try:
                        await asyncio.wait_for(client.close(), timeout=min(1.0, rem_close))
                    except Exception:
                        pass
                else:
                    # Deadline already expired: schedule non-blocking close without extending past smoke_timeout_s!
                    try:
                        asyncio.get_running_loop().create_task(client.close())
                    except Exception:
                        pass
        return results

    return await asyncio.wait_for(_run_inner(), timeout=float(smoke_timeout_s))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase1-policy", action="store_true")
    ap.add_argument("--require-explicit-book-ids", action="store_true")
    ap.add_argument("--enforce-3book-cutover", action="store_true")
    ap.add_argument("--account-mode", choices=("demo", "live"), default=os.environ.get("DERIV_ACCOUNT_MODE", "demo"))
    ap.add_argument("--book-ids", default=os.environ.get("DERIV_SOTA_BOOK_IDS", ""))
    ap.add_argument("--offline-only", action="store_true")
    ap.add_argument("--live-smoke", action="store_true")
    ap.add_argument("--allow-empty-index", action="store_true")
    ap.add_argument("--allow-unprovisioned-queue", action="store_true")
    ap.add_argument("--repo-root", default=str(REPO_ROOT))
    ap.add_argument("--fixture-home", default=None)
    ap.add_argument("--fixture-repo-root", default=None)
    ap.add_argument("--log-dir", default=os.environ.get("DERIV_SOTA_LOG_DIR", "logs/sota_runtime"))
    ap.add_argument("--lock-root", default=os.environ.get("DERIV_RUNTIME_LOCK_ROOT", ""))
    ap.add_argument("--queue-db", default=os.environ.get("DERIV_SOTA_QUEUE_DB", ""))
    ap.add_argument("--stake", default=os.environ.get("DERIV_SOTA_STAKE", "1.0"))
    ap.add_argument("--max-stake-usd", type=float, default=float(os.environ.get("DERIV_SOTA_MAX_STAKE_USD", "5.0")))
    ap.add_argument("--max-daily-loss-usd", type=float, default=float(os.environ.get("DERIV_SOTA_MAX_DAILY_LOSS_USD", "25.0")))
    ap.add_argument("--max-drawdown-usd", type=float, default=float(os.environ.get("DERIV_SOTA_MAX_DRAWDOWN_USD", "50.0")))
    ap.add_argument("--max-open-contracts", type=int, default=int(os.environ.get("DERIV_SOTA_MAX_OPEN_CONTRACTS", "3")))
    ap.add_argument("--max-usd-factor-open", type=int, default=int(os.environ.get("DERIV_SOTA_MAX_USD_FACTOR_OPEN", "3")))
    ap.add_argument("--max-jpy-factor-open", type=int, default=int(os.environ.get("DERIV_SOTA_MAX_JPY_FACTOR_OPEN", "3")))
    ap.add_argument("--absolute-breakeven-ceiling", type=float, default=float(os.environ.get("DERIV_SOTA_BREAKEVEN_CEILING", "0.60")))
    ap.add_argument("--ev-margin", type=float, default=float(os.environ.get("DERIV_SOTA_EV_MARGIN", "0.005")))
    ap.add_argument("--max-buy-dispatch-delay-s", type=float, default=float(os.environ.get("DERIV_SOTA_MAX_BUY_DISPATCH_DELAY_S", "4.0")))
    ap.add_argument("--credential-isolation-receipt", default=os.environ.get("DERIV_SOTA_CREDENTIAL_ISOLATION_RECEIPT", ""))
    ap.add_argument("--smoke-timeout-s", type=float, default=10.0)
    args = ap.parse_args(argv)

    # Mutually exclusive mode check (spec §3.6); require explicit --offline-only or --live-smoke
    if args.offline_only and args.live_smoke:
        print(json.dumps({"passed": False, "error": "--offline-only and --live-smoke are mutually exclusive"}))
        return 2
    if not args.offline_only and not args.live_smoke:
        print(json.dumps({"passed": False, "error": "must specify either --offline-only or --live-smoke"}))
        return 2

    # BLOCKER #1: Non-empty fixture paths required AND only admitted with --offline-only!
    has_fh = args.fixture_home is not None
    has_fr = args.fixture_repo_root is not None
    using_fixture = has_fh or has_fr
    if using_fixture:
        if not args.offline_only:
            print(json.dumps({"passed": False, "error": "--fixture-home/--fixture-repo-root admitted ONLY with --offline-only"}))
            return 2
        if not (has_fh and has_fr and str(args.fixture_home).strip() and str(args.fixture_repo_root).strip()):
            print(json.dumps({"passed": False, "error": "both non-empty --fixture-home and --fixture-repo-root are required together"}))
            return 2
    if args.allow_unprovisioned_queue and not using_fixture:
        print(json.dumps({"passed": False, "error": "--allow-unprovisioned-queue admitted ONLY inside validated fixture root"}))
        return 2
    if Path(args.repo_root).resolve() != REPO_ROOT.resolve():
        print(json.dumps({"passed": False, "error": "--repo-root override is forbidden"}))
        return 2

    checks: list[dict[str, Any]] = []
    failed = False
    reconciliation_only = False

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

    # 2. BLOCKER #1: Unconditional Phase 1 non-bypassable config validation (policy markers cannot relax checks!)
    cfg = None
    try:
        cfg = resolve_and_validate_sota_runtime_config(
            account_mode=args.account_mode,
            account_id=os.environ.get("DERIV_ACCOUNT_ID", ""),
            book_ids_raw=args.book_ids,
            stake_raw=args.stake,
            max_stake_usd=args.max_stake_usd,
            max_daily_loss_usd=args.max_daily_loss_usd,
            max_drawdown_usd=args.max_drawdown_usd,
            max_open_contracts=args.max_open_contracts,
            max_usd_factor_open=args.max_usd_factor_open,
            max_jpy_factor_open=args.max_jpy_factor_open,
            absolute_breakeven_ceiling=args.absolute_breakeven_ceiling,
            ev_margin=args.ev_margin,
            max_buy_dispatch_delay_s=args.max_buy_dispatch_delay_s,
            queue_db_raw=args.queue_db,
            log_dir_raw=args.log_dir,
            credential_isolation_receipt_raw=args.credential_isolation_receipt,
            enforce_3book_cutover=bool(args.enforce_3book_cutover),
            require_explicit_book_ids=True,
            fixture_home=Path(args.fixture_home) if using_fixture else None,
            fixture_repo_root=Path(args.fixture_repo_root) if using_fixture else None,
            allow_fixture=bool(args.offline_only and using_fixture),
        )
        record("phase1_runtime_config", True, account_id=cfg.account_id, book_ids=list(cfg.book_ids))
    except Exception as exc:
        record("phase1_runtime_config", False, error=str(exc))

    if cfg is not None:
        # Activation hold check
        hold_absent = not cfg.lock_paths["hold"].exists()
        record("activation_hold_absent", hold_absent, hold_path=str(cfg.lock_paths["hold"]))

        # Legacy buyers check (supplementary /proc scan)
        try:
            check_no_legacy_buyers()
            record("no_legacy_buyers", True)
        except Exception as exc:
            record("no_legacy_buyers", False, error=str(exc))

        # Queue & locks check (read-only snapshot; NEVER mutates canonical DB!)
        kill_present = (cfg.log_dir / "KILL").exists()
        try:
            if args.allow_unprovisioned_queue and using_fixture and not cfg.queue_db.exists():
                record("sqlite_v3_queue", True, status="unprovisioned_fixture_allowed", db_path=str(cfg.queue_db))
            else:
                lock_seq = [(cfg.lock_paths["queue"], False, True), (cfg.lock_paths["handoff"], False, True)]
                with acquire_ordered_locks(lock_seq):
                    ro = open_sota_queue_readonly(cfg.queue_db)
                    try:
                        binding = verify_v3_queue_binding(ro, expected_account_id=cfg.account_id, expected_account_mode="demo")
                        latch = is_safety_latch_set(ro)
                        if latch is not None or kill_present:
                            reconciliation_only = True
                            if args.live_smoke:
                                raise RuntimeError("live-smoke refused on latched or KILL queue")
                        record("sqlite_v3_queue", True, db_path=str(cfg.queue_db), bound_account=binding["account_id"], latched=latch is not None)
                    finally:
                        ro.close()
        except Exception as exc:
            record("sqlite_v3_queue", False, error=str(exc))

        # Load & verify exact requested books AFTER config admission (B7!)
        books: list[Any] = []
        try:
            idx = load_sota_index(repo_root=cfg.repo_root)
            books = load_enabled_sota_books(repo_root=cfg.repo_root, account_mode="demo", book_ids=list(cfg.book_ids))
            record(
                "certified_sota_books",
                len(books) == len(cfg.book_ids),
                registered_total=len(idx),
                loaded_book_ids=[b.book_id for b in books],
            )
        except Exception as exc:
            record("certified_sota_books", False, error=str(exc))

        # Live smoke (never buys/sells; strict exact rational quote floor; holds LOCK_SH|LOCK_NB sota_queue!)
        if args.live_smoke and not failed:
            try:
                with maybe_host_connection_lock(), acquire_ordered_locks([(cfg.lock_paths["queue"], False, True)]):
                    smoke = asyncio.run(
                        run_network_proposal_smoke(
                            "demo",
                            books,
                            cfg.stake_usd,
                            absolute_breakeven_ceiling=cfg.absolute_breakeven_ceiling,
                            ev_margin=cfg.ev_margin,
                            handoff_lock_path=cfg.lock_paths["handoff"],
                            queue_db_path=cfg.queue_db,
                            kill_path=cfg.log_dir / "KILL",
                            smoke_timeout_s=args.smoke_timeout_s,
                        )
                    )
                all_ok = bool(smoke) and all(x.get("smoke_passed") is True for x in smoke)
                record("network_proposal_smoke", all_ok, proposals=smoke)
            except Exception as exc:
                record("network_proposal_smoke", False, error=str(exc))

    report = {
        "schema": "deriv-sota-vps-preflight/v1",
        "account_mode": args.account_mode,
        "reconciliation_only": reconciliation_only,
        "passed": not failed,
        "checks": checks,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
