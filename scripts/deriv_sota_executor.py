"""Durable multi-model, multi-horizon, EV-gated Deriv SOTA VPS executor.

Sibling to scripts/deriv_trade_executor.py and scripts/deriv_trade_queue.py (both kept
byte-frozen for PARITY_TRACKED_PATHS and historical seals).

Features:
- Loads certified books from books/SOTA_INDEX.json (`sota_book_runtime.py`).
- Scores live 1s-tick features (`deriv_sota_tick_stream.py`) at every minute close (watermark-gated in external mode).
- Keys durable SQLite queue on `(pair, horizon_minutes, side, lane, offset_id, signal_close_utc)`.
- Enforces currency-leg concentration caps (`max_usd_factor_open` AND `max_jpy_factor_open`) and halts on any unresolved `buy_intent`/`buy_unknown` state.
- Enforces dual floor + calibrated row-EV payout gate on fresh proposals (`deriv_runtime_core_v2.py`).
- Defaults fail-closed to Demo (`VRTC*`, `/demo` WS, `sota_trade_queue_demo.sqlite`).
- Supports explicit opt-in Real-Money (`CR*`, `--account-mode live` + `DERIV_ALLOW_LIVE_REAL_MONEY=1`,
  `sota_trade_queue_live.sqlite`, server-side `currency == "USD"` & `loginid` verification across reconnects,
  and persisted SQLite circuit breakers: `max_daily_loss_usd`, `max_drawdown_usd`, `max_stake_usd`,
  `max_open_contracts`, overdue-contract & 5-error automatic `KILL` tripwire).
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import sqlite3
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from deriv_async_client import DerivAsyncClient
from deriv_client import API_BASE, PUBLIC_WS_URL, DerivAPIError, DerivEnv
from deriv_runtime_core_v2 import (
    SOTA_DERIV_SYMBOLS,
    ExecutorError,
    ExecutorLock,
    JsonlLogger,
    SotaPayoutDecision,
    after_last_start_cutoff_v2,
    canonical_lock_root,
    check_kill_switch,
    contract_type_for,
    currency_leg_exposure,
    evaluate_sota_payout_gate,
    finite_positive_number,
    is_ny_16h_haircut_window,
    is_ny_session,
    is_venue_sub5h_open,
    raw_hash,
    sota_account_lock_key,
    validate_account_mode,
)
from deriv_sota_tick_stream import (
    DEFAULT_SOTA_TICK_STORE,
    LiveTickBuffer,
    SotaFeatureError,
    extract_decision_feature_row,
    read_tick_store_watermark,
)
from sota_book_runtime import load_enabled_sota_books

CLASSIC_V3_WS_TEMPLATE = "wss://ws.derivws.com/websockets/v3?app_id={app_id}"
TERMINAL_CONTRACT_STATUSES = frozenset({"won", "lost", "sold", "expired", "cancelled"})
OVERDUE_OPEN_GRACE_S = 600.0


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_iso(ts: datetime | None = None) -> str:
    return (ts or utc_now()).astimezone(timezone.utc).isoformat()


def _parse_iso_utc(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(timezone.utc)
    except Exception:
        return None


def request_authenticated_ws_url(env: DerivEnv, account_mode: str, timeout: float = 10.0) -> tuple[str, str]:
    """Obtain authenticated WebSocket URL via Options v1 OTP endpoint (or classic v3 token transport if configured)."""
    transport = os.environ.get("DERIV_WS_TRANSPORT", "options_otp").strip().lower()
    if transport == "v3_authorize":
        ws_url = CLASSIC_V3_WS_TEMPLATE.format(app_id=env.app_id)
        validate_account_mode(account_mode=account_mode, account_id=env.account_id, ws_url=ws_url)
        return ws_url, "v3_authorize"

    url = f"{API_BASE}/trading/v1/options/accounts/{env.account_id}/otp"
    req = urllib.request.Request(
        url,
        method="POST",
        headers={
            "Deriv-App-ID": env.app_id,
            "Authorization": f"Bearer {env.pat}",
            "Content-Type": "application/json",
        },
        data=b"{}",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise DerivAPIError(f"OTP request failed: HTTP {exc.code}: {body}") from exc
    except urllib.error.URLError as exc:
        raise DerivAPIError(f"OTP request failed: {exc}") from exc

    data = payload.get("data", payload)
    ws_url = data.get("url") or data.get("websocket_url")
    if not isinstance(ws_url, str) or not ws_url.startswith("wss://"):
        raise DerivAPIError(f"OTP response missing wss:// URL: {payload!r}")
    validate_account_mode(account_mode=account_mode, account_id=env.account_id, ws_url=ws_url)
    return ws_url, "options_otp"


# ---------------------------------------------------------------------------
# Durable SQLite Queue + Persisted Risk Circuit-Breaker Ledger
# ---------------------------------------------------------------------------


def default_queue_db_path(account_mode: str, runtime_dir: Path = Path("deriv_data/runtime")) -> Path:
    mode = str(account_mode).strip().lower()
    if mode not in {"demo", "live"}:
        raise ExecutorError(f"invalid account_mode {account_mode!r}")
    return runtime_dir / f"sota_trade_queue_{mode}.sqlite"


def connect_sota_queue(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), timeout=5.0, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("PRAGMA foreign_keys=ON")
    mode = str(conn.execute("PRAGMA journal_mode=WAL").fetchone()[0]).lower()
    conn.execute("PRAGMA synchronous=FULL")
    if mode != "wal":
        raise ExecutorError(f"SQLite journal_mode did not become WAL: {mode!r}")
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS sota_signals (
            signal_id TEXT PRIMARY KEY,
            book_id TEXT NOT NULL,
            pair TEXT NOT NULL,
            horizon_minutes INTEGER NOT NULL,
            side TEXT NOT NULL CHECK (side IN ('UP','DOWN')),
            lane TEXT NOT NULL DEFAULT 'wall',
            offset_id INTEGER NOT NULL DEFAULT 0,
            signal_close_utc TEXT NOT NULL,
            effective_floor REAL NOT NULL,
            calibrated_win_prob REAL,
            payload_json TEXT NOT NULL,
            status TEXT NOT NULL CHECK (status IN (
                'queued','claimed','buy_intent','bought','contract_closed',
                'terminal_skip','signal_expired','buy_blocked_risk','buy_unknown'
            )),
            reason TEXT,
            created_utc TEXT NOT NULL,
            updated_utc TEXT NOT NULL,
            UNIQUE(pair, horizon_minutes, side, lane, offset_id, signal_close_utc)
        );

        CREATE TABLE IF NOT EXISTS sota_contracts (
            contract_id TEXT PRIMARY KEY,
            signal_id TEXT NOT NULL UNIQUE REFERENCES sota_signals(signal_id),
            book_id TEXT NOT NULL,
            pair TEXT NOT NULL,
            horizon_minutes INTEGER NOT NULL,
            side TEXT NOT NULL CHECK (side IN ('UP','DOWN')),
            proposal_id TEXT NOT NULL,
            buy_price_cents INTEGER NOT NULL,
            payout_cents INTEGER NOT NULL,
            sell_price_cents INTEGER NOT NULL DEFAULT 0,
            profit_cents INTEGER NOT NULL DEFAULT 0,
            status TEXT NOT NULL CHECK (status IN ('open','won','lost','sold','expired','cancelled','unknown')),
            opened_utc TEXT NOT NULL,
            expected_expiry_utc TEXT NOT NULL,
            closed_utc TEXT,
            raw_hash TEXT NOT NULL,
            signal_close_utc TEXT
        );

        CREATE TABLE IF NOT EXISTS sota_risk_state (
            key TEXT PRIMARY KEY,
            value_json TEXT NOT NULL,
            updated_utc TEXT NOT NULL
        );
        """
    )
    cols = {str(r["name"]) for r in conn.execute("PRAGMA table_info(sota_contracts)").fetchall()}
    if "signal_close_utc" not in cols:
        conn.execute("ALTER TABLE sota_contracts ADD COLUMN signal_close_utc TEXT")
    return conn


def sota_signal_id(signal: dict[str, Any]) -> str:
    parts = [
        str(signal["book_id"]),
        str(signal["pair"]).upper(),
        str(int(signal["horizon_minutes"])),
        str(signal["side"]).upper(),
        str(signal.get("lane", "wall")),
        str(int(signal.get("offset_id", 0))),
        str(signal["signal_close_utc"]),
    ]
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


def enqueue_sota_signal(conn: sqlite3.Connection, signal: dict[str, Any]) -> tuple[bool, str]:
    sid = sota_signal_id(signal)
    now = utc_iso()
    conn.execute("BEGIN IMMEDIATE")
    try:
        cur = conn.execute(
            """
            INSERT OR IGNORE INTO sota_signals (
                signal_id, book_id, pair, horizon_minutes, side, lane, offset_id,
                signal_close_utc, effective_floor, calibrated_win_prob, payload_json,
                status, created_utc, updated_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'queued', ?, ?)
            """,
            (
                sid,
                str(signal["book_id"]),
                str(signal["pair"]).upper(),
                int(signal["horizon_minutes"]),
                str(signal["side"]).upper(),
                str(signal.get("lane", "wall")),
                int(signal.get("offset_id", 0)),
                str(signal["signal_close_utc"]),
                float(signal["effective_floor"]),
                float(signal["calibrated_win_prob"]) if signal.get("calibrated_win_prob") is not None else None,
                json.dumps(signal, sort_keys=True),
                now,
                now,
            ),
        )
        conn.execute("COMMIT")
        return cur.rowcount == 1, sid
    except Exception:
        conn.execute("ROLLBACK")
        raise


@dataclass(frozen=True)
class RiskBreakerConfig:
    max_stake_usd: float = 5.0
    max_daily_loss_usd: float = 25.0
    max_drawdown_usd: float = 50.0
    max_open_contracts: int = 4
    max_usd_factor_open: int = 2
    max_jpy_factor_open: int = 2
    same_book_min_gap_seconds: int = 900


def check_risk_and_exposure(
    conn: sqlite3.Connection,
    *,
    book_id: str,
    pair: str,
    horizon_minutes: int,
    side: str,
    stake_usd: float,
    cfg: RiskBreakerConfig,
    now_utc: datetime | None = None,
    signal_close_utc: str | None = None,
) -> str | None:
    """Return rejection reason if any persisted risk breaker, unresolved buy, or currency-leg cap is breached."""
    now = (now_utc or utc_now()).astimezone(timezone.utc)
    if not math.isfinite(stake_usd) or stake_usd <= 0.0:
        return "invalid_stake"
    if stake_usd > cfg.max_stake_usd + 1e-9:
        return "stake_exceeds_max_stake_usd"

    # 0. Fail-closed halt on ANY unresolved buy_intent / buy_unknown signal or unknown contract!
    uncertain_sigs = int(
        conn.execute(
            "SELECT COUNT(*) FROM sota_signals WHERE status IN ('buy_intent', 'buy_unknown')"
        ).fetchone()[0]
    )
    unknown_contracts = int(
        conn.execute("SELECT COUNT(*) FROM sota_contracts WHERE status = 'unknown'").fetchone()[0]
    )
    if uncertain_sigs > 0 or unknown_contracts > 0:
        return "unresolved_buy_uncertainty_halt"

    # 1. Check open contract count, bar-close nonoverlap cooldown, and currency-leg exposure (USD + JPY)
    open_rows = conn.execute(
        "SELECT book_id, pair, horizon_minutes, side, opened_utc FROM sota_contracts WHERE status = 'open'"
    ).fetchall()
    if cfg.max_open_contracts > 0 and len(open_rows) >= cfg.max_open_contracts:
        return "max_open_contracts_reached"

    # Exact bar-close de-overlap (`signal_close_utc` diff < gap, matching cpcv_deriv.nonoverlap_chrono)
    min_gap = max(int(horizon_minutes) * 60, int(cfg.same_book_min_gap_seconds))
    if min_gap > 0:
        ref_dt = _parse_iso_utc(signal_close_utc) or now
        recent_rows = conn.execute(
            """
            SELECT c.opened_utc, COALESCE(c.signal_close_utc, s.signal_close_utc) AS sig_close
            FROM sota_contracts c
            LEFT JOIN sota_signals s ON s.signal_id = c.signal_id
            WHERE c.pair = ? AND c.horizon_minutes = ?
            ORDER BY c.opened_utc DESC LIMIT 20
            """,
            (pair.upper(), int(horizon_minutes)),
        ).fetchall()
        for r in recent_rows:
            prev_dt = _parse_iso_utc(r["sig_close"]) or _parse_iso_utc(r["opened_utc"])
            if prev_dt is not None and 0.0 <= (ref_dt - prev_dt).total_seconds() < float(min_gap) - 1e-6:
                return "same_pair_horizon_cooldown"

    want_legs = currency_leg_exposure(pair, side)
    leg_counts: dict[str, int] = {}
    for r in open_rows:
        r_legs = currency_leg_exposure(str(r["pair"]), str(r["side"]))
        for ccy, sign in r_legs.items():
            if want_legs.get(ccy) == sign:
                leg_counts[ccy] = leg_counts.get(ccy, 0) + 1

    if cfg.max_usd_factor_open > 0 and "USD" in want_legs and leg_counts.get("USD", 0) >= cfg.max_usd_factor_open:
        return "max_usd_factor_open_reached"
    if cfg.max_jpy_factor_open > 0 and "JPY" in want_legs and leg_counts.get("JPY", 0) >= cfg.max_jpy_factor_open:
        return "max_jpy_factor_open_reached"

    # 2. Check persisted daily realized loss + worst-case open stake at risk
    day_prefix = now.strftime("%Y-%m-%d")
    daily_profit_cents = int(
        conn.execute(
            """
            SELECT COALESCE(SUM(profit_cents), 0) FROM sota_contracts
            WHERE status IN ('won','lost','sold','expired','cancelled') AND closed_utc LIKE ?
            """,
            (f"{day_prefix}%",),
        ).fetchone()[0]
    )
    open_risk_cents = int(
        conn.execute(
            "SELECT COALESCE(SUM(buy_price_cents), 0) FROM sota_contracts WHERE status = 'open'"
        ).fetchone()[0]
    )
    new_stake_cents = int(round(stake_usd * 100))
    worst_daily_loss_cents = -daily_profit_cents + open_risk_cents + new_stake_cents
    if cfg.max_daily_loss_usd > 0 and worst_daily_loss_cents > int(round(cfg.max_daily_loss_usd * 100)):
        return "daily_loss_circuit_breaker"

    # 3. Check persisted peak-to-trough cumulative drawdown (including worst-case open stake)
    settled_profits = [
        int(r[0])
        for r in conn.execute(
            """
            SELECT profit_cents FROM sota_contracts
            WHERE status IN ('won','lost','sold','expired','cancelled')
            ORDER BY closed_utc ASC, contract_id ASC
            """
        ).fetchall()
    ]
    cum = 0
    peak = 0
    for p_c in settled_profits:
        cum += p_c
        if cum > peak:
            peak = cum
    worst_drawdown_cents = (peak - cum) + open_risk_cents + new_stake_cents
    if cfg.max_drawdown_usd > 0 and worst_drawdown_cents > int(round(cfg.max_drawdown_usd * 100)):
        return "max_drawdown_circuit_breaker"

    return None


# ---------------------------------------------------------------------------
# SOTA Executor Engine
# ---------------------------------------------------------------------------


class SotaTradeExecutor:
    def __init__(
        self,
        *,
        account_mode: str,
        stake_usd: float,
        db_path: Path,
        log_dir: Path,
        risk_cfg: RiskBreakerConfig,
        absolute_breakeven_ceiling: float = 0.60,
        ev_margin: float = 0.005,
        dry_run_proposal_only: bool = False,
        max_signal_age_s: float = 45.0,
        env: DerivEnv | None = None,
    ):
        self.account_mode = str(account_mode).strip().lower()
        self.stake_usd = float(stake_usd)
        self.db_path = Path(db_path)
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.kill_path = self.log_dir / "KILL"
        self.logger = JsonlLogger(self.log_dir / f"sota_executor_{self.account_mode}")
        self.risk_cfg = risk_cfg
        self.absolute_breakeven_ceiling = float(absolute_breakeven_ceiling)
        self.ev_margin = float(ev_margin)
        self.dry_run_proposal_only = bool(dry_run_proposal_only)
        self.max_signal_age_s = float(max_signal_age_s)
        self.env = env
        self.client: DerivAsyncClient | None = None
        self.transport_kind: str = "options_otp"
        self._authorized_reconnect_epoch: int = -1
        self.consecutive_reconcile_errors = 0

    def trip_kill_switch(self, reason: str, **detail: Any) -> None:
        self.kill_path.write_text(json.dumps({"tripped_utc": utc_iso(), "reason": reason, **detail}) + "\n", encoding="utf-8")
        self.logger.write("kill_switch_tripped", reason=reason, **detail)

    async def ensure_authenticated(self) -> None:
        """Verify server-side account loginid, USD currency, and virtual/real mode on startup AND after any WS reconnect."""
        if self.client is None or self.env is None:
            return
        reconnects = int(getattr(self.client, "stats", {}).get("reconnects", 0))
        if reconnects == self._authorized_reconnect_epoch:
            return

        if self.transport_kind == "v3_authorize":
            auth_resp = await self.client.request({"authorize": self.env.pat}, timeout=10.0)
            auth_info = auth_resp.get("authorize") or {}
            loginid = str(auth_info.get("loginid") or "").strip()
            ccy = str(auth_info.get("currency") or "").strip().upper()
            is_virt = bool(auth_info.get("is_virtual"))
            if not loginid or loginid != self.env.account_id:
                raise ExecutorError(f"PAT authorized loginid={loginid!r} != DERIV_ACCOUNT_ID {self.env.account_id!r}")
            if ccy != "USD":
                raise ExecutorError(f"SOTA executor requires USD account currency, got {ccy!r}")
            if self.account_mode == "live" and is_virt:
                raise ExecutorError("live mode refused: token authorized a virtual account")
            if self.account_mode == "demo" and not is_virt:
                raise ExecutorError("demo mode refused: token authorized a non-virtual account")
        else:
            bal_resp = await self.client.request({"balance": 1}, timeout=10.0)
            bal_info = bal_resp.get("balance") or {}
            loginid = str(bal_info.get("loginid") or "").strip()
            ccy = str(bal_info.get("currency") or "").strip().upper()
            if loginid and loginid != self.env.account_id:
                raise ExecutorError(f"OTP socket loginid={loginid!r} != DERIV_ACCOUNT_ID {self.env.account_id!r}")
            if ccy != "USD":
                raise ExecutorError(f"SOTA executor requires USD account currency on OTP socket, got {ccy!r}")

        self._authorized_reconnect_epoch = reconnects

    def pre_proposal_gate(self, payload: dict[str, Any], now_utc: datetime | None = None) -> str | None:
        now = (now_utc or utc_now()).astimezone(timezone.utc)
        try:
            check_kill_switch(self.kill_path, self.logger)
        except ExecutorError:
            return "kill_switch_triggered"
        if not is_ny_session(now):
            return "outside_ny_session"
        horizon = int(payload["horizon_minutes"])
        if not is_venue_sub5h_open(now, horizon):
            return "venue_sub5h_blackout_21_24_utc"
        if after_last_start_cutoff_v2(now, horizon):
            return "after_last_start_cutoff"
        if payload.get("drop_ny_16h") and is_ny_16h_haircut_window(now):
            return "excluded_ny_16h_payout_haircut_window"

        close_dt = _parse_iso_utc(str(payload.get("signal_close_utc", "")))
        if close_dt is None:
            return "invalid_signal_close_utc"
        age_s = (now - close_dt).total_seconds()
        if age_s < -1.0 or age_s > self.max_signal_age_s:
            return "signal_expired"
        return None

    async def process_one_signal(
        self,
        conn: sqlite3.Connection,
        signal_id: str,
        payload: dict[str, Any],
        *,
        now_utc: datetime | None = None,
    ) -> dict[str, Any]:
        now = (now_utc or utc_now()).astimezone(timezone.utc)
        pre_reason = self.pre_proposal_gate(payload, now_utc=now)
        if pre_reason is not None:
            status = "signal_expired" if pre_reason == "signal_expired" else "terminal_skip"
            conn.execute(
                "UPDATE sota_signals SET status=?, reason=?, updated_utc=? WHERE signal_id=?",
                (status, pre_reason, utc_iso(now), signal_id),
            )
            self.logger.write("signal_rejected_pre_proposal", **{**payload, "signal_id": signal_id, "reason": pre_reason})
            return {"signal_id": signal_id, "status": status, "reason": pre_reason}

        risk_reason = check_risk_and_exposure(
            conn,
            book_id=str(payload["book_id"]),
            pair=str(payload["pair"]),
            horizon_minutes=int(payload["horizon_minutes"]),
            side=str(payload["side"]),
            stake_usd=self.stake_usd,
            cfg=self.risk_cfg,
            now_utc=now,
            signal_close_utc=str(payload["signal_close_utc"]),
        )
        if risk_reason is not None:
            conn.execute(
                "UPDATE sota_signals SET status='buy_blocked_risk', reason=?, updated_utc=? WHERE signal_id=?",
                (risk_reason, utc_iso(now), signal_id),
            )
            self.logger.write("signal_blocked_risk", **{**payload, "signal_id": signal_id, "reason": risk_reason})
            return {"signal_id": signal_id, "status": "buy_blocked_risk", "reason": risk_reason}

        assert self.client is not None
        await self.ensure_authenticated()
        pair = str(payload["pair"]).upper()
        side = str(payload["side"]).upper()
        horizon = int(payload["horizon_minutes"])
        family = str(payload.get("contract_family", "risefall"))
        ctype = contract_type_for(side, family=family)
        sym_key = "symbol" if self.transport_kind == "v3_authorize" else "underlying_symbol"
        prop_req = {
            "proposal": 1,
            "amount": float(self.stake_usd),
            "basis": "stake",
            "contract_type": ctype,
            "currency": "USD",
            "duration": horizon,
            "duration_unit": "m",
            sym_key: SOTA_DERIV_SYMBOLS[pair],
        }
        try:
            prop_resp = await self.client.request(prop_req, timeout=4.0)
        except Exception as exc:
            conn.execute(
                "UPDATE sota_signals SET status='terminal_skip', reason='proposal_failed', updated_utc=? WHERE signal_id=?",
                (utc_iso(now), signal_id),
            )
            self.logger.write("proposal_failed", **{**payload, "signal_id": signal_id, "error": str(exc)[:240]})
            return {"signal_id": signal_id, "status": "terminal_skip", "reason": "proposal_failed"}

        prop = prop_resp.get("proposal") or {}
        decision: SotaPayoutDecision = evaluate_sota_payout_gate(
            prop,
            effective_floor=float(payload["effective_floor"]),
            absolute_breakeven_ceiling=self.absolute_breakeven_ceiling,
            calibrated_win_prob=payload.get("calibrated_win_prob"),
            ev_margin=self.ev_margin,
            min_payout_ratio=payload.get("min_payout_ratio"),
        )
        self.logger.write(
            "payout_gate_decision",
            **{
                **payload,
                "signal_id": signal_id,
                "passed": decision.passed,
                "reason": decision.reason,
                "ask": decision.ask,
                "payout": decision.payout,
                "payout_ratio": decision.payout_ratio,
                "live_breakeven": decision.live_breakeven,
                "net_floor_edge": decision.net_floor_edge,
                "calibrated_win_prob": decision.calibrated_win_prob,
                "calibrated_ev": decision.calibrated_ev,
                "dry_run": self.dry_run_proposal_only,
            },
        )
        if not decision.passed:
            conn.execute(
                "UPDATE sota_signals SET status='terminal_skip', reason=?, updated_utc=? WHERE signal_id=?",
                (decision.reason, utc_iso(now), signal_id),
            )
            return {"signal_id": signal_id, "status": "terminal_skip", "reason": decision.reason}

        proposal_id = prop.get("id")
        if not proposal_id:
            conn.execute(
                "UPDATE sota_signals SET status='terminal_skip', reason='missing_proposal_id', updated_utc=? WHERE signal_id=?",
                (utc_iso(now), signal_id),
            )
            return {"signal_id": signal_id, "status": "terminal_skip", "reason": "missing_proposal_id"}

        if self.dry_run_proposal_only:
            conn.execute(
                "UPDATE sota_signals SET status='terminal_skip', reason='dry_run_proposal_passed', updated_utc=? WHERE signal_id=?",
                (utc_iso(now), signal_id),
            )
            return {"signal_id": signal_id, "status": "terminal_skip", "reason": "dry_run_proposal_passed"}

        assert decision.ask is not None and decision.payout is not None
        conn.execute(
            "UPDATE sota_signals SET status='buy_intent', reason=?, updated_utc=? WHERE signal_id=?",
            (f"proposal_id:{proposal_id}", utc_iso(now), signal_id),
        )
        try:
            buy_resp = await self.client.request({"buy": str(proposal_id), "price": float(decision.ask)}, timeout=5.0)
        except Exception as exc:
            conn.execute(
                "UPDATE sota_signals SET status='buy_unknown', reason='buy_transport_uncertain', updated_utc=? WHERE signal_id=?",
                (utc_iso(now), signal_id),
            )
            self.logger.write("buy_unknown", **{**payload, "signal_id": signal_id, "proposal_id": str(proposal_id), "error": str(exc)[:240]})
            return {"signal_id": signal_id, "status": "buy_unknown", "reason": "buy_transport_uncertain"}

        buy_obj = buy_resp.get("buy") or {}
        contract_id = buy_obj.get("contract_id")
        if not contract_id:
            conn.execute(
                "UPDATE sota_signals SET status='buy_unknown', reason='missing_contract_id', updated_utc=? WHERE signal_id=?",
                (utc_iso(now), signal_id),
            )
            return {"signal_id": signal_id, "status": "buy_unknown", "reason": "missing_contract_id"}

        buy_price = finite_positive_number(buy_obj.get("buy_price")) or float(decision.ask)
        payout_val = finite_positive_number(buy_obj.get("payout")) or float(decision.payout)
        exp_utc = utc_iso(now + timedelta(minutes=horizon))
        sig_close = str(payload["signal_close_utc"])
        conn.execute("BEGIN IMMEDIATE")
        try:
            conn.execute(
                """
                INSERT INTO sota_contracts (
                    contract_id, signal_id, book_id, pair, horizon_minutes, side,
                    proposal_id, buy_price_cents, payout_cents, status,
                    opened_utc, expected_expiry_utc, raw_hash, signal_close_utc
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'open', ?, ?, ?, ?)
                """,
                (
                    str(contract_id),
                    signal_id,
                    str(payload["book_id"]),
                    pair,
                    horizon,
                    side,
                    str(proposal_id),
                    int(round(buy_price * 100)),
                    int(round(payout_val * 100)),
                    utc_iso(now),
                    exp_utc,
                    raw_hash(buy_resp),
                    sig_close,
                ),
            )
            conn.execute(
                "UPDATE sota_signals SET status='bought', reason='buy_confirmed', updated_utc=? WHERE signal_id=?",
                (utc_iso(now), signal_id),
            )
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise

        self.logger.write(
            "buy_confirmed",
            **{
                **payload,
                "signal_id": signal_id,
                "contract_id": str(contract_id),
                "buy_price": buy_price,
                "payout": payout_val,
                "calibrated_ev": decision.calibrated_ev,
                "account_mode": self.account_mode,
            },
        )
        return {"signal_id": signal_id, "status": "bought", "contract_id": str(contract_id)}

    async def _reconcile_uncertain_signals(self, conn: sqlite3.Connection, now: datetime) -> int:
        """Recover any crashed `buy_intent` or `buy_unknown` signal via Deriv `portfolio`, or count toward KILL."""
        assert self.client is not None
        uncertain = conn.execute(
            """
            SELECT signal_id, book_id, pair, horizon_minutes, side, signal_close_utc, payload_json, reason, updated_utc
            FROM sota_signals WHERE status IN ('buy_intent', 'buy_unknown')
            """
        ).fetchall()
        if not uncertain:
            return 0

        errors = 0
        try:
            port_resp = await self.client.request({"portfolio": 1}, timeout=6.0)
            contracts = (port_resp.get("portfolio") or {}).get("contracts") or []
        except Exception as exc:
            self.logger.write("portfolio_reconcile_error", error=str(exc)[:240])
            return len(uncertain)

        for row in uncertain:
            sid = str(row["signal_id"])
            pair = str(row["pair"]).upper()
            horizon = int(row["horizon_minutes"])
            side = str(row["side"]).upper()
            sig_dt = _parse_iso_utc(str(row["signal_close_utc"])) or now
            payload = json.loads(str(row["payload_json"]))
            want_sym = SOTA_DERIV_SYMBOLS[pair]
            want_ctype = contract_type_for(side, family=str(payload.get("contract_family", "risefall")))
            matched_c: dict[str, Any] | None = None
            for c in contracts:
                c_sym = str(c.get("symbol") or c.get("underlying_symbol") or "")
                c_type = str(c.get("contract_type") or "")
                dstart = int(c.get("date_start") or 0)
                if c_sym == want_sym and c_type == want_ctype and abs(dstart - int(sig_dt.timestamp())) <= 90:
                    matched_c = c
                    break

            if matched_c is not None and matched_c.get("contract_id"):
                cid = str(matched_c["contract_id"])
                bp = finite_positive_number(matched_c.get("buy_price")) or self.stake_usd
                po = finite_positive_number(matched_c.get("payout")) or round(bp * 1.80, 2)
                exp_epoch = int(matched_c.get("expiry_time") or (int(sig_dt.timestamp()) + horizon * 60))
                exp_iso = datetime.fromtimestamp(exp_epoch, tz=timezone.utc).isoformat()
                conn.execute("BEGIN IMMEDIATE")
                try:
                    conn.execute(
                        """
                        INSERT OR IGNORE INTO sota_contracts (
                            contract_id, signal_id, book_id, pair, horizon_minutes, side,
                            proposal_id, buy_price_cents, payout_cents, status,
                            opened_utc, expected_expiry_utc, raw_hash, signal_close_utc
                        ) VALUES (?, ?, ?, ?, ?, ?, 'recovered_portfolio', ?, ?, 'open', ?, ?, ?, ?)
                        """,
                        (
                            cid,
                            sid,
                            str(row["book_id"]),
                            pair,
                            horizon,
                            side,
                            int(round(bp * 100)),
                            int(round(po * 100)),
                            utc_iso(now),
                            exp_iso,
                            raw_hash(matched_c),
                            str(row["signal_close_utc"]),
                        ),
                    )
                    conn.execute(
                        "UPDATE sota_signals SET status='bought', reason='recovered_from_portfolio', updated_utc=? WHERE signal_id=?",
                        (utc_iso(now), sid),
                    )
                    conn.execute("COMMIT")
                except Exception:
                    conn.execute("ROLLBACK")
                    raise
                self.logger.write("buy_unknown_recovered", signal_id=sid, contract_id=cid)
            else:
                errors += 1
                self.logger.write("buy_unknown_unmatched", signal_id=sid, pair=pair, horizon_minutes=horizon)
        return errors

    async def reconcile_open_contracts(self, conn: sqlite3.Connection, *, now_utc: datetime | None = None) -> None:
        assert self.client is not None
        await self.ensure_authenticated()
        now = (now_utc or utc_now()).astimezone(timezone.utc)
        errors = await self._reconcile_uncertain_signals(conn, now)

        open_rows = conn.execute(
            "SELECT contract_id, buy_price_cents, expected_expiry_utc FROM sota_contracts WHERE status = 'open'"
        ).fetchall()
        if not open_rows and errors == 0:
            self.consecutive_reconcile_errors = 0
            return

        for row in open_rows:
            cid = str(row["contract_id"])
            try:
                resp = await self.client.request({"proposal_open_contract": 1, "contract_id": cid}, timeout=6.0)
                poc = resp.get("proposal_open_contract") or {}
                status_raw = str(poc.get("status") or "open").lower()
                is_terminal = bool(poc.get("is_sold")) or status_raw in TERMINAL_CONTRACT_STATUSES
                if is_terminal:
                    raw_sp = poc.get("sell_price")
                    sell_cents = int(round(float(raw_sp) * 100)) if raw_sp is not None else 0
                    buy_cents = int(row["buy_price_cents"])
                    raw_pf = poc.get("profit")
                    profit_cents = (
                        int(round(float(raw_pf) * 100))
                        if raw_pf is not None
                        else (sell_cents - buy_cents)
                    )
                    if status_raw in {"expired", "cancelled"} and profit_cents == 0 and sell_cents == 0:
                        final_st = status_raw
                    else:
                        final_st = "won" if profit_cents > 0 else ("lost" if profit_cents < 0 else "sold")
                    conn.execute("BEGIN IMMEDIATE")
                    try:
                        conn.execute(
                            """
                            UPDATE sota_contracts
                            SET status=?, sell_price_cents=?, profit_cents=?, closed_utc=?, raw_hash=?
                            WHERE contract_id=?
                            """,
                            (final_st, sell_cents, profit_cents, utc_iso(now), raw_hash(resp), cid),
                        )
                        conn.execute(
                            """
                            UPDATE sota_signals SET status='contract_closed', updated_utc=?
                            WHERE signal_id=(SELECT signal_id FROM sota_contracts WHERE contract_id=?)
                            """,
                            (utc_iso(now), cid),
                        )
                        conn.execute("COMMIT")
                    except Exception:
                        conn.execute("ROLLBACK")
                        raise
                    self.logger.write("contract_settled", contract_id=cid, status=final_st, profit_usd=profit_cents / 100.0)
                else:
                    exp_dt = _parse_iso_utc(str(row["expected_expiry_utc"]))
                    if exp_dt is not None and (now - exp_dt).total_seconds() > OVERDUE_OPEN_GRACE_S:
                        errors += 1
                        self.logger.write("contract_overdue_open", contract_id=cid, expected_expiry_utc=str(row["expected_expiry_utc"]))
            except Exception as exc:
                errors += 1
                self.logger.write("reconcile_error", contract_id=cid, error=str(exc)[:240])

        if errors:
            self.consecutive_reconcile_errors += 1
            if self.consecutive_reconcile_errors >= 5:
                self.trip_kill_switch("consecutive_reconcile_errors", count=self.consecutive_reconcile_errors)
        else:
            self.consecutive_reconcile_errors = 0


async def run_integrated_vps_loop(args: argparse.Namespace) -> int:
    books = load_enabled_sota_books(account_mode=args.account_mode)
    if not books:
        print("FAIL-CLOSED: 0 certified SOTA books registered in books/SOTA_INDEX.json for account_mode=" + args.account_mode)
        return 2

    env = DerivEnv.from_env()
    ws_url, transport_kind = await asyncio.to_thread(request_authenticated_ws_url, env, args.account_mode)
    db_path = Path(args.queue_db) if args.queue_db else default_queue_db_path(args.account_mode)
    risk_cfg = RiskBreakerConfig(
        max_stake_usd=args.max_stake_usd,
        max_daily_loss_usd=args.max_daily_loss_usd,
        max_drawdown_usd=args.max_drawdown_usd,
        max_open_contracts=args.max_open_contracts,
        max_usd_factor_open=args.max_usd_factor_open,
        max_jpy_factor_open=args.max_jpy_factor_open,
    )
    executor = SotaTradeExecutor(
        account_mode=args.account_mode,
        stake_usd=args.stake,
        db_path=db_path,
        log_dir=Path(args.log_dir),
        risk_cfg=risk_cfg,
        absolute_breakeven_ceiling=args.absolute_breakeven_ceiling,
        ev_margin=args.ev_margin,
        dry_run_proposal_only=args.dry_run_proposal_only,
        env=env,
    )
    executor.transport_kind = transport_kind
    lock_root = canonical_lock_root(Path(args.lock_root) if args.lock_root else None)
    lock_key = sota_account_lock_key(env.account_id, args.account_mode)

    pairs_needed = sorted({b.pair for b in books})
    tick_store_dir = Path(args.tick_store_dir)
    tick_buf = LiveTickBuffer(pairs=pairs_needed, store_dir=tick_store_dir)
    tick_buf.load_persisted()
    external_ticks = str(args.tick_feed_mode).lower() == "external"

    async def _async_refresh_auth_url() -> str:
        fresh_url, _ = await asyncio.to_thread(request_authenticated_ws_url, env, args.account_mode)
        return fresh_url

    with ExecutorLock(lock_root, lock_key, executor.logger):
        conn = connect_sota_queue(db_path)
        from contextlib import AsyncExitStack
        async with AsyncExitStack() as stack:
            pub_client = (
                None
                if external_ticks
                else await stack.enter_async_context(DerivAsyncClient(PUBLIC_WS_URL, silence_timeout_s=45.0))
            )
            auth_client = await stack.enter_async_context(
                DerivAsyncClient(
                    ws_url,
                    url_factory=_async_refresh_auth_url,
                )
            )
            executor.client = auth_client
            await executor.ensure_authenticated()
            await executor.reconcile_open_contracts(conn)
            subs = {}
            if not external_ticks:
                assert pub_client is not None
                await tick_buf.warmup_from_public_ws(pub_client, lookback_hours=36)
                subs = {
                    p: await pub_client.subscribe({"ticks": SOTA_DERIV_SYMBOLS[p], "subscribe": 1})
                    for p in tick_buf.pairs
                }
            last_scored_minute = int(time.time() // 60) * 60
            last_reconcile = time.monotonic()
            deadline = (time.monotonic() + args.duration_seconds) if args.duration_seconds else None

            while deadline is None or time.monotonic() < deadline:
                if auth_client._background_failed.is_set():
                    raise ExecutorError(f"auth WS background failure: {auth_client._background_error}")
                if not external_ticks:
                    assert pub_client is not None
                    if pub_client._background_failed.is_set():
                        raise ExecutorError(f"public tick WS background failure: {pub_client._background_error}")
                    for p, sub in subs.items():
                        if sub.error is not None:
                            raise ExecutorError(f"public tick subscription error on {p}: {sub.error}")
                        while not sub.queue.empty():
                            fr = sub.queue.get_nowait()
                            tk = fr.get("tick")
                            if isinstance(tk, dict) and "epoch" in tk and "quote" in tk:
                                tick_buf.ingest_ticks(p, [int(tk["epoch"])], [float(tk["quote"])])

                now_wall = time.time()
                minute_close = (int(now_wall) // 60) * 60
                if minute_close > last_scored_minute and (now_wall - minute_close) >= 1.2:
                    ready_to_score = True
                    if external_ticks:
                        wm = read_tick_store_watermark(tick_store_dir)
                        wm_min = int(wm.get("minute_boundary_epoch") or 0)
                        if wm_min < minute_close:
                            if (now_wall - minute_close) < 4.5:
                                ready_to_score = False  # wait for deriv_sota_tick_stream boundary flush
                            else:
                                last_scored_minute = minute_close
                                ready_to_score = False
                                executor.logger.write(
                                    "external_watermark_lag_skip",
                                    minute_close=minute_close,
                                    watermark_minute=wm_min,
                                )
                        else:
                            tick_buf.load_persisted()

                    if ready_to_score:
                        last_scored_minute = minute_close
                        # Reconcile just-expired contracts BEFORE checking open exposure & scoring!
                        await executor.reconcile_open_contracts(conn)
                        last_reconcile = time.monotonic()
                        snap = tick_buf.snapshot_ticks()
                        for book in books:
                            try:
                                frow = extract_decision_feature_row(
                                    book.pair,
                                    snap,
                                    minute_close,
                                    book.feature_cols,
                                )
                                sc = book.score_row(frow.row, minute_close)
                                executor.logger.write(
                                    "book_scored",
                                    book_id=book.book_id,
                                    pair=book.pair,
                                    horizon_minutes=book.horizon_minutes,
                                    decision_utc=frow.decision_utc,
                                    proba_up=round(sc.proba_up_raw, 6),
                                    confidence=round(sc.confidence, 6),
                                    direction=sc.direction,
                                    calibrated_win_prob=sc.calibrated_win_prob,
                                    gate_passed=sc.gate_passed,
                                    gate_reasons=sc.gate_reasons,
                                )
                                if sc.gate_passed:
                                    sig = {
                                        "book_id": book.book_id,
                                        "pair": book.pair,
                                        "horizon_minutes": book.horizon_minutes,
                                        "side": sc.direction,
                                        "lane": "wall",
                                        "offset_id": 0,
                                        "signal_close_utc": frow.decision_utc,
                                        "effective_floor": sc.effective_floor,
                                        "calibrated_win_prob": sc.calibrated_win_prob,
                                        "min_payout_ratio": book.min_payout_ratio,
                                        "drop_ny_16h": book.drop_ny_16h,
                                        "contract_family": book.contract_family,
                                    }
                                    inserted, sid = enqueue_sota_signal(conn, sig)
                                    if inserted:
                                        await executor.process_one_signal(conn, sid, sig)
                            except SotaFeatureError as exc:
                                executor.logger.write("feature_skip", book_id=book.book_id, pair=book.pair, reason=str(exc))

                        if not external_ticks:
                            tick_buf.persist_snapshot(minute_boundary_epoch=minute_close)

                if time.monotonic() - last_reconcile >= 5.0:
                    await executor.reconcile_open_contracts(conn)
                    last_reconcile = time.monotonic()

                await asyncio.sleep(0.15)
        conn.close()
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--account-mode", choices=("demo", "live"), default=os.environ.get("DERIV_ACCOUNT_MODE", "demo"))
    ap.add_argument(
        "--tick-feed-mode",
        choices=("integrated", "external"),
        default=os.environ.get("DERIV_SOTA_TICK_FEED_MODE", "integrated"),
    )
    ap.add_argument("--stake", type=float, default=float(os.environ.get("DERIV_SOTA_STAKE", "1.0")))
    ap.add_argument("--max-stake-usd", type=float, default=float(os.environ.get("DERIV_SOTA_MAX_STAKE_USD", "5.0")))
    ap.add_argument("--max-daily-loss-usd", type=float, default=float(os.environ.get("DERIV_SOTA_MAX_DAILY_LOSS_USD", "25.0")))
    ap.add_argument("--max-drawdown-usd", type=float, default=float(os.environ.get("DERIV_SOTA_MAX_DRAWDOWN_USD", "50.0")))
    ap.add_argument("--max-open-contracts", type=int, default=int(os.environ.get("DERIV_SOTA_MAX_OPEN_CONTRACTS", "4")))
    ap.add_argument("--max-usd-factor-open", type=int, default=int(os.environ.get("DERIV_SOTA_MAX_USD_FACTOR_OPEN", "2")))
    ap.add_argument("--max-jpy-factor-open", type=int, default=int(os.environ.get("DERIV_SOTA_MAX_JPY_FACTOR_OPEN", "2")))
    ap.add_argument("--absolute-breakeven-ceiling", type=float, default=float(os.environ.get("DERIV_SOTA_BREAKEVEN_CEILING", "0.60")))
    ap.add_argument("--ev-margin", type=float, default=float(os.environ.get("DERIV_SOTA_EV_MARGIN", "0.005")))
    ap.add_argument("--dry-run-proposal-only", action="store_true")
    ap.add_argument("--queue-db", default=os.environ.get("DERIV_SOTA_QUEUE_DB", ""))
    ap.add_argument("--tick-store-dir", default=os.environ.get("DERIV_SOTA_TICK_STORE_DIR", str(DEFAULT_SOTA_TICK_STORE)))
    ap.add_argument("--log-dir", default=os.environ.get("DERIV_SOTA_LOG_DIR", "logs/sota_runtime"))
    ap.add_argument("--lock-root", default=os.environ.get("DERIV_RUNTIME_LOCK_ROOT", ""))
    ap.add_argument("--duration-seconds", type=float, default=0.0)
    return ap


def main() -> int:
    args = build_parser().parse_args()
    if not args.run:
        build_parser().print_help()
        return 0
    return asyncio.run(run_integrated_vps_loop(args))


if __name__ == "__main__":
    raise SystemExit(main())
