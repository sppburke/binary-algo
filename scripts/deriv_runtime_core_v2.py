"""Deriv SOTA v2 runtime core: finite-quote math, multi-horizon cutoffs, calibrated EV gating,
currency-leg exposure vectors, and fail-closed Demo (VRTC) / opt-in Live (CR) account guards.

Sibling to scripts/deriv_runtime_core.py (which remains byte-frozen for evidence/objects seals).
"""

from __future__ import annotations

import hashlib
import math
import os
import socket
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
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
JURISDICTION_ENV = "DERIV_HOST_JURISDICTION"
REQUIRED_JURISDICTION_VALUE = "mexico_vps"
BLOCKED_US_HOSTNAMES = frozenset({"quantum"})
SUPPORTED_HORIZONS_MINUTES = (5, 15, 30, 60, 120, 240, 1440)
SUPPORTED_CONTRACT_FAMILIES = ("risefall", "equal", "multiplier", "barrier_1d")
TWO_BARRIER_TYPES = frozenset({"RANGE", "UPORDOWN", "EXPIRYRANGE", "EXPIRYMISS"})
SINGLE_BARRIER_TYPES = frozenset({"NOTOUCH", "ONETOUCH"})
ALL_BARRIER_TYPES = TWO_BARRIER_TYPES | SINGLE_BARRIER_TYPES


def assert_non_us_deriv_host(
    *,
    hostname: str | None = None,
    jurisdiction_flag: str | None = None,
    us_lab_marker: Path = Path("/mnt/data/deriv-sota-lab"),
) -> None:
    """Fail-closed jurisdiction fence: block ALL direct Deriv server connections from USA hosts (`quantum`).
    Only the Mexico VPS (`DERIV_HOST_JURISDICTION=mexico_vps`) may contact Deriv servers directly.
    """
    host = (socket.gethostname() if hostname is None else str(hostname)).strip().lower()
    flag = (os.environ.get(JURISDICTION_ENV, "") if jurisdiction_flag is None else str(jurisdiction_flag)).strip().lower()
    if host in BLOCKED_US_HOSTNAMES or us_lab_marker.exists():
        raise ExecutorError(
            f"JURISDICTION BLOCK: direct Deriv network connections are forbidden on US workstation ({host!r}); "
            "run all Deriv WebSocket/REST interactions exclusively on the Mexico VPS."
        )
    if flag != REQUIRED_JURISDICTION_VALUE:
        raise ExecutorError(
            f"JURISDICTION BLOCK: direct Deriv network access requires {JURISDICTION_ENV}={REQUIRED_JURISDICTION_VALUE!r} "
            f"on the Mexico VPS (got {flag!r})."
        )

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
    "EURGBP": "frxEURGBP",
    "EURAUD": "frxEURAUD",
    "EURNZD": "frxEURNZD",
    "EURCAD": "frxEURCAD",
    "EURCHF": "frxEURCHF",
    "GBPAUD": "frxGBPAUD",
    "GBPNZD": "frxGBPNZD",
    "GBPCAD": "frxGBPCAD",
    "GBPCHF": "frxGBPCHF",
    "AUDCAD": "frxAUDCAD",
    "AUDCHF": "frxAUDCHF",
    "AUDNZD": "frxAUDNZD",
    "USDMXN": "frxUSDMXN",
    "USDPLN": "frxUSDPLN",
    "XAUUSD": "frxXAUUSD",
    "XAGUSD": "frxXAGUSD",
    "XPDUSD": "frxXPDUSD",
    "XPTUSD": "frxXPTUSD",
    "BTCUSD": "cryBTCUSD",
    "ETHUSD": "cryETHUSD",
    "WLDUSD": "WLDUSD",
    "WLDEUR": "WLDEUR",
    "WLDGBP": "WLDGBP",
    "WLDAUD": "WLDAUD",
    "WLDXAU": "WLDXAU",
    "OTC_SPC": "OTC_SPC",
    "OTC_NDX": "OTC_NDX",
    "OTC_DJI": "OTC_DJI",
    "OTC_FTSE": "OTC_FTSE",
    "OTC_GDAXI": "OTC_GDAXI",
    "OTC_N225": "OTC_N225",
    "OTC_FCHI": "OTC_FCHI",
    "OTC_SX5E": "OTC_SX5E",
    "OTC_SSMI": "OTC_SSMI",
    "OTC_AEX": "OTC_AEX",
    "OTC_AS51": "OTC_AS51",
    "OTC_HSI": "OTC_HSI",
}


def contract_type_for(
    side: str,
    family: str = "risefall",
    *,
    barrier_contract_type: str | None = None,
) -> str:
    side_u = str(side).upper()
    if family == "barrier_1d":
        bct = str(barrier_contract_type or "").strip().upper()
        if bct not in ALL_BARRIER_TYPES:
            raise ExecutorError(f"unsupported barrier_contract_type {barrier_contract_type!r}")
        if side_u not in {"UP", "DOWN", "NEUTRAL"}:
            raise ExecutorError(f"unsupported barrier side {side!r}")
        return bct
    if side_u not in {"UP", "DOWN"}:
        raise ExecutorError(f"unsupported side {side!r}")
    if family == "risefall":
        return "CALL" if side_u == "UP" else "PUT"
    if family == "equal":
        return "CALLE" if side_u == "UP" else "PUTE"
    if family == "multiplier":
        return "MULTUP" if side_u == "UP" else "MULTDOWN"
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
    """Deriv rejects intraday (< 5h) FX Rise/Fall purchases whose expiry lands in 16:50-18:00 America/New_York.
    Starts in [16:50 - horizon_minutes, 18:00) NY are untradeable for sub-5h contracts.
    """
    if horizon_minutes <= 0:
        raise ExecutorError(f"horizon_minutes must be positive, got {horizon_minutes}")
    if horizon_minutes >= 300:
        return False
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
    """Return signed (+1 long / -1 short) currency or asset-leg exposure for risk concentration caps."""
    p = str(pair).strip().upper()
    s = str(side).strip().upper()
    if s == "NEUTRAL":
        if not p:
            raise ExecutorError(f"invalid pair {pair!r}")
        return {f"VOL_{p}": 1}
    if s not in {"UP", "DOWN"}:
        raise ExecutorError(f"side must be UP, DOWN, or NEUTRAL, got {side!r}")
    sign = 1 if s == "UP" else -1
    if p.startswith("WLD") and len(p) == 6 and p[3:].isalpha():
        return {p[3:]: sign}
    if p.startswith("OTC_"):
        return {p: sign}
    if len(p) != 6 or not p.isalpha():
        raise ExecutorError(f"pair must be a 6-letter FX/Metal/Crypto code, WLD*, or OTC_*, got {pair!r}")
    base, quote = p[:3], p[3:]
    return {base: sign, quote: -sign}


def format_barrier_levels(pair: str, spot: float, rel_half_width: float, contract_type: str, side: str = "NEUTRAL") -> tuple[str, str | None]:
    """Format exact Deriv `barrier` (and `barrier2` for two-barrier 1d contracts) around live `spot`."""
    spot_f = finite_positive_number(spot)
    rw_f = finite_positive_number(rel_half_width)
    if spot_f is None or rw_f is None or rw_f >= 0.25:
        raise ExecutorError(f"invalid barrier spot={spot!r} or rel_half_width={rel_half_width!r}")
    ct = str(contract_type).strip().upper()
    if ct not in ALL_BARRIER_TYPES:
        raise ExecutorError(f"unsupported barrier contract_type {contract_type!r}")
    p = str(pair).strip().upper()
    if "JPY" in p:
        decimals = 3
    elif p.startswith("WLD"):
        decimals = 3
    elif "XAG" in p:
        decimals = 4
    elif any(k in p for k in ("XAU", "XPD", "XPT", "BTC", "ETH")) or p.startswith("OTC_"):
        decimals = 2
    else:
        decimals = 5
    spot_r = round(spot_f, decimals)
    hi = f"{spot_f * (1.0 + rw_f):.{decimals}f}"
    lo = f"{spot_f * (1.0 - rw_f):.{decimals}f}"
    if float(hi) <= spot_r or float(lo) >= spot_r or float(hi) <= float(lo):
        raise ExecutorError(f"degenerate rounded barrier levels around spot={spot_f}: hi={hi}, lo={lo}")
    if ct in TWO_BARRIER_TYPES:
        return hi, lo
    s_u = str(side).strip().upper()
    if s_u not in {"UP", "DOWN"}:
        raise ExecutorError(f"single-barrier contract {ct!r} requires explicit UP or DOWN side, got {side!r}")
    chosen = lo if s_u == "DOWN" else hi
    return chosen, None


@dataclass(frozen=True)
class SotaMultiplierDecision:
    ask: float | None
    commission: float | None
    commission_frac_stake: float | None
    spot: float | None
    stop_out: float | None
    stop_out_rel_dist: float | None
    expected_ev: float | None
    reason: str | None

    @property
    def passed(self) -> bool:
        return self.reason is None


def evaluate_sota_multiplier_gate(
    prop: dict[str, Any],
    *,
    expected_ev: float | None,
    side: str | None = None,
    ev_margin: float = 0.005,
    certified_commission_frac: float | None = None,
    max_commission_frac: float = 0.20,
    min_stop_out_rel_dist: float = 0.0008,
) -> SotaMultiplierDecision:
    """Fail-closed quote & cost gate for `MULTUP` / `MULTDOWN` proposals."""
    if not isinstance(prop, dict):
        return SotaMultiplierDecision(None, None, None, None, None, None, None, "missing_or_invalid_ask")
    bounds_to_check = [ev_margin, max_commission_frac, min_stop_out_rel_dist]
    if certified_commission_frac is not None:
        bounds_to_check.append(certified_commission_frac)
    for bound in bounds_to_check:
        if isinstance(bound, bool) or not isinstance(bound, (int, float)) or not math.isfinite(float(bound)) or float(bound) < 0.0:
            return SotaMultiplierDecision(None, None, None, None, None, None, None, "invalid_policy_bounds")
    ccy = prop.get("currency")
    if ccy is not None and str(ccy).upper() != "USD":
        return SotaMultiplierDecision(None, None, None, None, None, None, None, "non_usd_proposal_currency")
    ask = finite_positive_number(prop.get("ask_price"))
    if ask is None:
        return SotaMultiplierDecision(None, None, None, None, None, None, None, "missing_or_invalid_ask")
    raw_comm = prop.get("commission")
    if isinstance(raw_comm, bool) or raw_comm is None:
        return SotaMultiplierDecision(ask, None, None, None, None, None, None, "missing_or_invalid_commission")
    try:
        comm = float(raw_comm)
    except (TypeError, ValueError):
        return SotaMultiplierDecision(ask, None, None, None, None, None, None, "missing_or_invalid_commission")
    if not math.isfinite(comm) or comm < 0.0:
        return SotaMultiplierDecision(ask, None, None, None, None, None, None, "missing_or_invalid_commission")
    comm_frac = comm / ask
    spot = finite_positive_number(prop.get("spot"))
    lim_ord = prop.get("limit_order")
    so_dict = lim_ord.get("stop_out") if isinstance(lim_ord, dict) else None
    so_raw = so_dict.get("value") if isinstance(so_dict, dict) else None
    so_val = finite_positive_number(so_raw)
    if spot is None or so_val is None:
        return SotaMultiplierDecision(ask, comm, comm_frac, spot, so_val, None, None, "missing_or_invalid_stop_out")
    if side is not None:
        s_u = str(side).strip().upper()
        if s_u == "UP" and so_val >= spot:
            return SotaMultiplierDecision(ask, comm, comm_frac, spot, so_val, None, None, "invalid_stop_out_side")
        if s_u == "DOWN" and so_val <= spot:
            return SotaMultiplierDecision(ask, comm, comm_frac, spot, so_val, None, None, "invalid_stop_out_side")
    so_rel = abs(so_val - spot) / spot
    if comm_frac > float(max_commission_frac) + 1e-9:
        return SotaMultiplierDecision(ask, comm, comm_frac, spot, so_val, so_rel, None, "commission_above_ceiling")
    if so_rel < float(min_stop_out_rel_dist) - 1e-9:
        return SotaMultiplierDecision(ask, comm, comm_frac, spot, so_val, so_rel, None, "stop_out_too_tight")
    if expected_ev is None or isinstance(expected_ev, bool) or not math.isfinite(float(expected_ev)):
        return SotaMultiplierDecision(ask, comm, comm_frac, spot, so_val, so_rel, None, "missing_expected_ev")
    extra_comm = comm_frac if certified_commission_frac is None else max(0.0, comm_frac - float(certified_commission_frac))
    ev_f = float(expected_ev) - extra_comm
    if ev_f < float(ev_margin):
        return SotaMultiplierDecision(ask, comm, comm_frac, spot, so_val, so_rel, ev_f, "expected_ev_below_margin")
    return SotaMultiplierDecision(ask, comm, comm_frac, spot, so_val, so_rel, ev_f, None)


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
    if reason is None and (not math.isfinite(float(absolute_breakeven_ceiling)) or live_breakeven > float(absolute_breakeven_ceiling)):  # type: ignore[operator]
        reason = "breakeven_too_high"
    if reason is None and min_payout_ratio is not None and (not math.isfinite(float(min_payout_ratio)) or payout_ratio < float(min_payout_ratio) - 1e-9):  # type: ignore[operator]
        reason = "payout_below_floor"
    if reason is None and require_calibrated_ev and calibrated_win_prob is None:
        reason = "missing_calibrated_probability"
    if reason is None and calibrated_win_prob is not None:
        assert cal_ev is not None
        if not math.isfinite(float(ev_margin)) or not math.isfinite(cal_ev) or cal_ev < float(ev_margin):
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
