"""Deriv SOTA v2 runtime core: finite-quote math, multi-horizon cutoffs, calibrated EV gating,
currency-leg exposure vectors, and fail-closed Demo (VRTC) / opt-in Live (CR) account guards.

Sibling to scripts/deriv_runtime_core.py (which remains byte-frozen for evidence/objects seals).
"""

from __future__ import annotations

import hashlib
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlsplit

from deriv_runtime_core import (
    NY_TZ,
    ExecutorError,
    ExecutorLock,
    JsonlLogger,
    account_lock_key as _v1_account_lock_key,
    canonical_lock_root,
    check_kill_switch,
    executor_lock_path,
    is_ny_session,
    raw_hash,
)

LIVE_CONFIRM_ENV = "DERIV_ALLOW_LIVE_REAL_MONEY"
SUPPORTED_HORIZONS_MINUTES = (15, 30, 60)
SUPPORTED_CONTRACT_FAMILIES = ("risefall", "equal")

SOTA_DERIV_SYMBOLS: dict[str, str] = {
    "EURUSD": "frxEURUSD",
    "GBPUSD": "frxGBPUSD",
    "USDJPY": "frxUSDJPY",
    "USDCHF": "frxUSDCHF",
    "USDCAD": "frxUSDCAD",
    "AUDUSD": "frxAUDUSD",
    "NZDUSD": "frxNZDUSD",
    "EURJPY": "frxEURJPY",
    "GBPJPY": "frxGBPJPY",
    "NZDJPY": "frxNZDJPY",
    "CHFJPY": "frxCHFJPY",
    "AUDJPY": "frxAUDJPY",
    "CADJPY": "frxCADJPY",
}


def contract_type_for(side: str, family: str = "risefall") -> str:
    side_u = str(side).upper()
    if side_u not in {"UP", "DOWN"}:
        raise ExecutorError(f"unsupported side {side!r}")
    if family == "risefall":
        return "CALL" if side_u == "UP" else "PUT"
    if family == "equal":
        return "CALLE" if side_u == "UP" else "PUTE"
    raise ExecutorError(f"unsupported contract family {family!r}")


def sota_account_lock_key(account_id: str, account_mode: str = "demo") -> str:
    value = account_id.strip()
    if not value:
        raise ExecutorError("missing Deriv account id for executor lock")
    if account_mode == "demo":
        return _v1_account_lock_key(value)
    if account_mode == "live":
        return f"deriv-live-account:{value}"
    raise ExecutorError(f"unknown account_mode {account_mode!r}")


def validate_account_mode(
    *,
    account_mode: str,
    account_id: str,
    ws_url: str,
    env_live_flag: str | None = None,
) -> None:
    """Fail-closed boundary separating VRTC demo sockets from CR real-money sockets.
    Pins exact Deriv hostnames and paths via urlsplit (never substring matching).
    """
    mode = str(account_mode).strip().lower()
    acct = str(account_id).strip()
    url = str(ws_url).strip()
    flag = (os.environ.get(LIVE_CONFIRM_ENV, "") if env_live_flag is None else str(env_live_flag)).strip()

    if not acct:
        raise ExecutorError("missing DERIV_ACCOUNT_ID")
    parts = urlsplit(url)
    if parts.scheme != "wss" or not parts.hostname or parts.username is not None or parts.password is not None:
        raise ExecutorError(f"Deriv WebSocket URL must be clean wss:// host URL, got {url!r}")

    host = parts.hostname.lower()
    norm_path = parts.path.rstrip("/")
    is_v3 = host == "ws.derivws.com" and norm_path == "/websockets/v3"
    is_opt_demo = host == "api.derivws.com" and norm_path == "/trading/v1/options/ws/demo"
    is_opt_live = host == "api.derivws.com" and norm_path == "/trading/v1/options/ws/live"

    if mode == "demo":
        if flag not in {"", "0", "false", "False"}:
            raise ExecutorError(
                f"demo mode refuses startup while {LIVE_CONFIRM_ENV}={flag!r} is set"
            )
        if not acct.upper().startswith("VRTC"):
            raise ExecutorError(
                f"demo mode requires a VRTC* demo account id, got {acct[:6]!r}"
            )
        if not (is_opt_demo or is_v3):
            raise ExecutorError(
                f"demo mode requires wss://api.derivws.com/trading/v1/options/ws/demo or wss://ws.derivws.com/websockets/v3, got {url!r}"
            )
        return

    if mode == "live":
        if flag != "1":
            raise ExecutorError(
                f"live real-money mode requires {LIVE_CONFIRM_ENV}=1 explicitly set"
            )
        if not acct.upper().startswith("CR") and not acct.upper().startswith("MF"):
            raise ExecutorError(
                f"live real-money mode requires a CR*/MF* real account id, got {acct[:6]!r}"
            )
        if not (is_opt_live or is_v3):
            raise ExecutorError(
                f"live real-money mode requires wss://api.derivws.com/trading/v1/options/ws/live or wss://ws.derivws.com/websockets/v3, got {url!r}"
            )
        return

    raise ExecutorError(f"unsupported account_mode {account_mode!r}; expected 'demo' or 'live'")


def finite_positive_number(value: Any) -> float | None:
    """Return finite float > 0 or None (rejects bool, NaN, +inf, -inf, <= 0)."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        parsed = float(value)
        return parsed if (math.isfinite(parsed) and parsed > 0.0) else None
    if isinstance(value, str):
        try:
            parsed = float(value)
        except ValueError:
            return None
        return parsed if (math.isfinite(parsed) and parsed > 0.0) else None
    return None


def parse_quote_v2(prop: dict[str, Any]) -> tuple[float | None, float | None, float | None, str | None]:
    """Return (ask, payout, live_breakeven, invalid_reason) failing closed on non-finite values."""
    if not isinstance(prop, dict):
        return None, None, None, "missing_or_invalid_ask"
    ccy = prop.get("currency")
    if ccy is not None and str(ccy).upper() != "USD":
        return None, None, None, "non_usd_proposal_currency"
    ask = finite_positive_number(prop.get("ask_price"))
    payout = finite_positive_number(prop.get("payout"))
    if ask is None:
        return None, payout, None, "missing_or_invalid_ask"
    if payout is None:
        return ask, None, None, "missing_or_invalid_payout"
    breakeven = ask / payout
    if not math.isfinite(breakeven) or breakeven <= 0.0:
        return ask, payout, None, "missing_or_invalid_payout"
    return ask, payout, breakeven, None


def after_last_start_cutoff_v2(ts_utc: datetime, horizon_minutes: int = 15) -> bool:
    """Deriv rejects FX Rise/Fall purchases whose expiry lands in 16:50-18:00 America/New_York.
    Starts in [16:50 - horizon_minutes, 18:00) NY are untradeable.
    """
    if horizon_minutes <= 0:
        raise ExecutorError(f"horizon_minutes must be positive, got {horizon_minutes}")
    if ts_utc.tzinfo is None:
        ts_utc = ts_utc.replace(tzinfo=timezone.utc)
    local = ts_utc.astimezone(NY_TZ)
    minute_of_day = local.hour * 60 + local.minute + local.second / 60.0
    cutoff_start = 16 * 60 + 50 - int(horizon_minutes)
    return cutoff_start <= minute_of_day < 18 * 60


def is_venue_sub5h_open(ts_utc: datetime, horizon_minutes: int) -> bool:
    """Owner Decision 14: Deriv refuses FX contracts < 5 hours from 21:00:00 to 23:59:59 UTC."""
    if horizon_minutes >= 300:
        return True
    if ts_utc.tzinfo is None:
        ts_utc = ts_utc.replace(tzinfo=timezone.utc)
    u = ts_utc.astimezone(timezone.utc)
    return u.hour < 21


def is_ny_16h_haircut_window(ts_utc: datetime) -> bool:
    """True when America/New_York local time lies in [16:00, 16:35), where 15m payouts drop to ~1.58."""
    if ts_utc.tzinfo is None:
        ts_utc = ts_utc.replace(tzinfo=timezone.utc)
    local = ts_utc.astimezone(NY_TZ)
    mod = local.hour * 60 + local.minute
    return 960 <= mod < 995


def currency_leg_exposure(pair: str, side: str) -> dict[str, int]:
    """Return signed (+1 long / -1 short) currency exposure for a 6-letter FX pair and UP/DOWN side."""
    p = str(pair).strip().upper()
    s = str(side).strip().upper()
    if len(p) != 6 or not p.isalpha():
        raise ExecutorError(f"pair must be a 6-letter FX code, got {pair!r}")
    if s not in {"UP", "DOWN"}:
        raise ExecutorError(f"side must be UP or DOWN, got {side!r}")
    base, quote = p[:3], p[3:]
    sign = 1 if s == "UP" else -1
    return {base: sign, quote: -sign}


@dataclass(frozen=True)
class SotaPayoutDecision:
    ask: float | None
    payout: float | None
    payout_ratio: float | None
    live_breakeven: float | None
    net_floor_edge: float | None
    calibrated_win_prob: float | None
    calibrated_ev: float | None
    reason: str | None

    @property
    def passed(self) -> bool:
        return self.reason is None


def evaluate_sota_payout_gate(
    prop: dict[str, Any],
    *,
    effective_floor: float,
    absolute_breakeven_ceiling: float,
    calibrated_win_prob: float | None = None,
    ev_margin: float = 0.0,
    min_payout_ratio: float | None = None,
    require_calibrated_ev: bool = True,
) -> SotaPayoutDecision:
    """Dual mathematical gate (fail-closed):
    1. Certified refit-CPCV p10 floor clears live_breakeven (ask / payout) and ceiling.
    2. Optional minimum gross payout ratio floor (payout / ask >= min_payout_ratio).
    3. Mandatory calibrated row-level expected value: p_cal * (payout / ask) - 1 >= ev_margin.
    """
    ask, payout, live_breakeven, reason = parse_quote_v2(prop)
    payout_ratio = (payout / ask) if (ask is not None and payout is not None) else None
    net_floor_edge = (effective_floor - live_breakeven) if live_breakeven is not None else None
    cal_ev: float | None = None

    if calibrated_win_prob is not None:
        if (
            isinstance(calibrated_win_prob, bool)
            or not math.isfinite(float(calibrated_win_prob))
            or not (0.0 <= float(calibrated_win_prob) <= 1.0)
        ):
            return SotaPayoutDecision(
                ask=ask,
                payout=payout,
                payout_ratio=payout_ratio,
                live_breakeven=live_breakeven,
                net_floor_edge=net_floor_edge,
                calibrated_win_prob=None,
                calibrated_ev=None,
                reason="invalid_calibrated_probability",
            )
        if payout_ratio is not None:
            cal_ev = float(calibrated_win_prob) * payout_ratio - 1.0

    if reason is None and (not math.isfinite(effective_floor) or effective_floor <= live_breakeven):  # type: ignore[operator]
        reason = "edge_not_positive"
    if reason is None and live_breakeven > absolute_breakeven_ceiling:  # type: ignore[operator]
        reason = "breakeven_too_high"
    if reason is None and min_payout_ratio is not None and payout_ratio < float(min_payout_ratio) - 1e-9:  # type: ignore[operator]
        reason = "payout_below_floor"
    if reason is None and require_calibrated_ev and calibrated_win_prob is None:
        reason = "missing_calibrated_probability"
    if reason is None and calibrated_win_prob is not None:
        assert cal_ev is not None
        if not math.isfinite(cal_ev) or cal_ev < float(ev_margin):
            reason = "calibrated_ev_below_margin"

    return SotaPayoutDecision(
        ask=ask,
        payout=payout,
        payout_ratio=payout_ratio,
        live_breakeven=live_breakeven,
        net_floor_edge=net_floor_edge,
        calibrated_win_prob=float(calibrated_win_prob) if calibrated_win_prob is not None else None,
        calibrated_ev=cal_ev,
        reason=reason,
    )


def feature_vector_sha256(values: dict[str, float], cols: list[str]) -> str:
    payload = ",".join(f"{c}={float(values[c]):.10g}" for c in cols)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
