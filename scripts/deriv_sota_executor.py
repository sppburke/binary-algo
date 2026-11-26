"""Durable multi-model, multi-horizon, EV-gated Deriv SOTA VPS executor (Phase 1 v3 hardened).

Sibling to scripts/deriv_trade_executor.py and scripts/deriv_trade_queue.py (both kept
byte-frozen for PARITY_TRACKED_PATHS and historical seals).

Phase 1 hardening (spec §§3.3–3.6):
- Exact wire decimal decoding (`_WireDecimal`, duplicate-key & non-finite rejection) and exact
  rational `Fraction` quote-floor enforcement (`1.76` m15 / `1.78` m30, defeating `1.7799999999`),
  exact `100`-cent submitted stake, whole-cent settlement, and unbounded Python `int` risk sums.
- `SotaDerivAsyncClient(DerivAsyncClient)`: single `_exclusive_send_text(send_class, ...)` gate
  forbidding generic `request()`, enforcing 5 capability classes (`public_read`, `identity` with
  both `options_otp` bootstrap `balance` and `v3_authorize`, `account_read`, `proposal`,
  `trade_mutation`), invalidating generation on reconnect, and blocking BOTH `buy` AND `sell`
  with 0 bytes sent whenever `allow_buy=False`.
- Canonical `pwd.getpwuid` absolute queue and 3-flock namespace (`sota_acct -> sota_queue ->
  sota_handoff`), read-only vs writable PRAGMAs outside `BEGIN`, atomic single-transaction `v3`
  SQLite DDL (`UNIQUE(book_id, pair, horizon_minutes, side, lane, offset_id, signal_close_utc)`,
  `sota_contracts.signal_id UNIQUE`, venue provenance columns, `user_version = 3`).
- Pre-export coherent SQLite backup, 6-shape predecessor adoption with `install_intent.json`
  recovery-first dispatch (recognizing exact single-file installed `v3` + re-syncing both rename
  dirs, bottom-up ancestry fsync before canonical unlinks, syncing removals before
  `renameat2(RENAME_NOREPLACE)`, migrated multiplier `payout_cents = 0 -> NULL`, and absent/null
  `signal_close_utc` backfill from linked `sota_signals`).
- Non-bypassable `resolve_and_validate_sota_runtime_config()`, once-per-pair feature caching,
  prospective `G6` cohort anchor registration, stale-lock cleanup, and `--request-stop-latch`.
"""

from __future__ import annotations

import argparse
import asyncio
import fcntl
import hashlib
import json
import math
import os
import pwd
import re
import shutil
import sqlite3
import stat
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from contextlib import AsyncExitStack, contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Iterator

import websockets
import websockets.exceptions

from deriv_async_client import DerivAsyncClient, Subscription
from deriv_client import API_BASE, PUBLIC_WS_URL, DerivAPIError, DerivEnv
from deriv_no_legacy_buy_guard import cmdline as proc_cmdline, offender as proc_offender
from deriv_runtime_core_v2 import (
    SOTA_DERIV_SYMBOLS,
    ExecutorError,
    ExecutorLock,
    JsonlLogger,
    SotaMultiplierDecision,
    SotaPayoutDecision,
    after_last_start_cutoff_v2,
    assert_non_us_deriv_host,
    canonical_lock_root,
    check_kill_switch,
    contract_type_for,
    currency_leg_exposure,
    evaluate_sota_multiplier_gate,
    evaluate_sota_payout_gate,
    executor_lock_path,
    finite_positive_number,
    format_barrier_levels,
    is_ny_16h_haircut_window,
    is_ny_session,
    is_valid_demo_account_id,
    is_venue_sub5h_open,
    raw_hash,
    sota_account_lock_key,
    validate_account_mode,
)
from deriv_sota_tick_stream import (
    DEFAULT_SOTA_TICK_STORE,
    DEFAULT_WARMUP_MIN_BARS,
    LiveTickBuffer,
    SotaFeatureError,
    build_pair_decision_snapshot,
    extract_decision_feature_row,
    read_tick_store_watermark,
)
from sota_book_freeze import (
    atomic_write_file_bytes,
    fsync_ancestry,
    fsync_dir,
    fsync_file,
    rename_noreplace,
)
from sota_book_runtime import (
    REPO_ROOT as LEXICAL_REPO_ROOT,
    SotaBookError,
    load_enabled_sota_books,
    load_sota_book,
    parse_book_ids_spec,
)

CLASSIC_V3_WS_TEMPLATE = "wss://ws.derivws.com/websockets/v3?app_id={app_id}"
TERMINAL_CONTRACT_STATUSES = frozenset({"won", "lost", "sold", "expired", "cancelled"})
OVERDUE_OPEN_GRACE_S = 600.0
PHASE1_ORDERED_BOOK_IDS: tuple[str, ...] = (
    "USDJPY.sota_ens5_m30_up.v1",
    "USDJPY.sota_stack05_m30_up.v1",
    "USDJPY.sota_ens5_m15_up.v1",
)
SQLITE_INT64_MIN = -(1 << 63)
SQLITE_INT64_MAX = (1 << 63) - 1
_STRICT_DEC_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d+)?|\.\d+)$")
CREDENTIAL_RECEIPT_SCHEMA = "sota-credential-isolation-receipt/v1"
FIRST_USE_RECORD_SCHEMA = "sota-first-use-record/v1"
CREDENTIAL_ISOLATION_MODE_REVOKED = "revoked"
CREDENTIAL_ISOLATION_MODE_SINGLE_HOST_MASKED = "single_host_units_masked"
LEGACY_MASKED_UNIT_NAMES: tuple[str, ...] = (
    "deriv-trade-executor.service",
    "deriv-runtime-supervisor.service",
    "deriv-demo-executor.service",
    "deriv-demo-executor.timer",
)
DEMO_ENV_ACCOUNT_SENTINEL = "VRTC_FROM_ENV"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_iso(ts: datetime | None = None) -> str:
    return (ts or utc_now()).astimezone(timezone.utc).isoformat()


def _parse_iso_utc(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        dt_v = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if dt_v.tzinfo is None:
            return None
        return dt_v.astimezone(timezone.utc)
    except Exception:
        return None


def _wire_contract_id(cid: str | int) -> int | str:
    s = str(cid).strip()
    return int(s) if s.isdigit() else s


def _safe_int_epoch(value: Any, fallback: int) -> int:
    if value is None or isinstance(value, bool):
        return fallback
    try:
        fv = float(value)
        if not math.isfinite(fv):
            return fallback
        ep = int(fv)
        return ep if (0 < ep <= 4_102_444_800) else fallback
    except (TypeError, ValueError, OverflowError):
        return fallback


# ---------------------------------------------------------------------------
# Exact Wire Decimal Decoding & Rational Money Enforcement (spec §§3.3-3.4)
# ---------------------------------------------------------------------------


class _WireDecimal(str):
    """Raw JSON decimal token preserving exact lexical precision alongside float compatibility."""

    def __float__(self) -> float:
        return float(str(self))


def strict_json_loads_wire(raw: str | bytes) -> Any:
    """Decode inbound WS JSON rejecting duplicate keys & non-finite constants and preserving _WireDecimal."""
    text = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else str(raw)

    def _no_dup_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for k, v in pairs:
            if k in out:
                raise DerivAPIError(f"duplicate JSON key rejected on wire: {k!r}")
            out[k] = v
        return out

    def _reject_const(tok: str) -> Any:
        raise DerivAPIError(f"non-finite JSON constant rejected on wire: {tok!r}")

    return json.loads(
        text,
        object_pairs_hook=_no_dup_pairs,
        parse_float=_WireDecimal,
        parse_constant=_reject_const,
    )


def parse_exact_money_fraction(
    raw: Any,
    *,
    allow_zero: bool = False,
    allow_negative: bool = False,
) -> Fraction:
    """Parse strict decimal text, int, or _WireDecimal into an exact Fraction within SQLite int64 cent bounds."""
    if isinstance(raw, bool) or raw is None:
        raise ExecutorError(f"invalid money value: {raw!r}")
    if isinstance(raw, Fraction):
        fr = raw
    elif isinstance(raw, int):
        fr = Fraction(raw, 1)
    elif isinstance(raw, (_WireDecimal, str)):
        s = str(raw)
        if not _STRICT_DEC_RE.fullmatch(s):
            raise ExecutorError(f"malformed decimal money token: {s!r}")
        fr = Fraction(s)
    elif isinstance(raw, float):
        if not math.isfinite(raw):
            raise ExecutorError(f"non-finite float money value: {raw!r}")
        s = repr(raw)
        if not _STRICT_DEC_RE.fullmatch(s):
            raise ExecutorError(f"non-plain float money value: {raw!r}")
        fr = Fraction(s)
    else:
        raise ExecutorError(f"unsupported money type {type(raw).__name__}")

    if not allow_negative and fr < 0:
        raise ExecutorError(f"negative money value refused: {raw!r}")
    if not allow_zero and fr == 0:
        raise ExecutorError(f"zero money value refused: {raw!r}")
    cents_f = fr * 100
    if cents_f < SQLITE_INT64_MIN or cents_f > SQLITE_INT64_MAX:
        raise ExecutorError(f"money value out of signed 64-bit SQLite cent range: {raw!r}")
    return fr


def parse_exact_whole_cents(
    raw: Any,
    *,
    allow_zero: bool = False,
    allow_negative: bool = False,
) -> int:
    fr = parse_exact_money_fraction(raw, allow_zero=allow_zero, allow_negative=allow_negative)
    cents_f = fr * 100
    if cents_f.denominator != 1:
        raise ExecutorError(f"fractional-cent money value refused where whole cents required: {raw!r}")
    return int(cents_f.numerator)


def check_exact_rational_quote_floor(
    prop: dict[str, Any],
    *,
    min_payout_ratio: float | str | Fraction | None,
) -> tuple[Fraction | None, Fraction | None, Fraction | None, str | None]:
    """Exact rational payout/ask >= min_payout_ratio pre-check defeating float tolerance (e.g. 1.7799999999 < 1.78)."""
    if not isinstance(prop, dict):
        return None, None, None, "missing_or_invalid_ask"
    ccy = prop.get("currency")
    if ccy is not None and str(ccy).upper() != "USD":
        return None, None, None, "non_usd_proposal_currency"
    try:
        ask_f = parse_exact_money_fraction(prop.get("ask_price"), allow_zero=False, allow_negative=False)
    except ExecutorError:
        return None, None, None, "missing_or_invalid_ask"
    try:
        pay_f = parse_exact_money_fraction(prop.get("payout"), allow_zero=False, allow_negative=False)
    except ExecutorError:
        return ask_f, None, None, "missing_or_invalid_payout"
    ratio = pay_f / ask_f
    if min_payout_ratio is not None:
        floor_f = (
            min_payout_ratio
            if isinstance(min_payout_ratio, Fraction)
            else Fraction(str(min_payout_ratio))
        )
        if ratio < floor_f:
            return ask_f, pay_f, ratio, "payout_below_exact_rational_floor"
    return ask_f, pay_f, ratio, None


# ---------------------------------------------------------------------------
# Canonical passwd-home Absolute Queue, Locks & Fixture Isolation (spec §§3.4, 3.6)
# ---------------------------------------------------------------------------


def canonical_pw_home(*, fixture_home: Path | None = None, allow_fixture: bool = False) -> Path:
    real_home = Path(pwd.getpwuid(os.getuid()).pw_dir).resolve(strict=True)
    if fixture_home is None:
        return real_home
    if not allow_fixture:
        raise ExecutorError("--fixture-home is forbidden on production actions")
    fh = Path(fixture_home)
    if not fh.is_absolute() or fh.is_symlink() or not fh.is_dir():
        raise ExecutorError(f"invalid --fixture-home: {fixture_home}")
    res = fh.resolve(strict=True)
    _assert_outside_protected_roots(res, real_home)
    return res


def _assert_outside_protected_roots(candidate: Path, real_home: Path) -> None:
    forbidden = [
        real_home,
        Path("/home/sean/git/binary-algo"),
        Path("/home/sean/git/binary-algo-vps"),
        Path("/mnt/data/deriv"),
        Path("/mnt/data/deriv-sota-lab"),
    ]
    for fb in forbidden:
        try:
            fb_res = fb.resolve()
        except OSError:
            fb_res = fb
        try:
            candidate.relative_to(fb_res)
            raise ExecutorError(f"fixture path {candidate} must lie strictly outside protected root {fb_res}")
        except ValueError:
            pass


def validate_fixture_roots(fixture_home: Path | None, fixture_repo_root: Path | None, *, allow_fixture: bool) -> tuple[Path | None, Path | None]:
    if fixture_home is None and fixture_repo_root is None:
        return None, None
    if not allow_fixture:
        raise ExecutorError("--fixture-home/--fixture-repo-root are strictly forbidden on production actions")
    if fixture_home is None or fixture_repo_root is None:
        raise ExecutorError("--fixture-home and --fixture-repo-root must be provided together")
    real_home = Path(pwd.getpwuid(os.getuid()).pw_dir).resolve(strict=True)
    fh = canonical_pw_home(fixture_home=fixture_home, allow_fixture=True)
    fr = Path(fixture_repo_root)
    if not fr.is_absolute() or fr.is_symlink() or not fr.is_dir():
        raise ExecutorError(f"invalid --fixture-repo-root: {fixture_repo_root}")
    fr_res = fr.resolve(strict=True)
    _assert_outside_protected_roots(fr_res, real_home)
    return fh, fr_res


def _nofollow_ensure_dir(path: Path, mode: int = 0o700) -> Path:
    parts = list(path.parts)
    if not parts or parts[0] != "/":
        raise ExecutorError(f"expected absolute path, got {path}")
    cur = Path("/")
    for comp in parts[1:]:
        cur = cur / comp
        if cur.is_symlink():
            raise ExecutorError(f"refusing symlink in canonical lock/queue path: {cur}")
        if not cur.exists():
            cur.mkdir(mode=mode)
    return path


def default_queue_db_path(account_mode: str, runtime_dir: Path | None = None) -> Path:
    mode = str(account_mode).strip().lower()
    if mode not in {"demo", "live"}:
        raise ExecutorError(f"invalid account_mode {account_mode!r}")
    rdir = runtime_dir if runtime_dir is not None else (LEXICAL_REPO_ROOT / "deriv_data" / "runtime")
    return Path(rdir) / f"sota_trade_queue_{mode}.sqlite"


def canonical_sota_queue_path(account_mode: str, *, repo_root: Path = LEXICAL_REPO_ROOT) -> Path:
    return default_queue_db_path(account_mode, runtime_dir=Path(repo_root) / "deriv_data" / "runtime")


def canonical_sota_lock_paths(queue_path: Path, *, pw_home: Path | None = None) -> dict[str, Path]:
    home = pw_home if pw_home is not None else canonical_pw_home()
    locks_dir = home / ".local" / "state" / "binary-algo" / "locks"
    q_norm = os.path.abspath(str(queue_path))
    sha12 = hashlib.sha256(q_norm.encode("utf-8")).hexdigest()[:12]
    return {
        "dir": locks_dir,
        "acct": locks_dir / f"sota_acct_{sha12}.flock",
        "queue": locks_dir / f"sota_queue_{sha12}.flock",
        "handoff": locks_dir / f"sota_handoff_{sha12}.flock",
        "hold": locks_dir / "sota_activation.hold",
    }


@contextmanager
def acquire_sota_flock(lock_path: Path, *, exclusive: bool = True, nonblocking: bool = True) -> Iterator[int]:
    """Acquire persistent 0600 flock without unlinking on release (O_NOFOLLOW)."""
    _nofollow_ensure_dir(lock_path.parent, mode=0o700)
    if lock_path.is_symlink():
        raise ExecutorError(f"refusing symlinked flock file: {lock_path}")
    fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        os.fchmod(fd, 0o600)
        flags = (fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH) | (fcntl.LOCK_NB if nonblocking else 0)
        try:
            fcntl.flock(fd, flags)
        except BlockingIOError as exc:
            raise ExecutorError(f"lock busy: {lock_path}") from exc
        yield fd
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


@contextmanager
def acquire_ordered_locks(lock_specs: list[tuple[Path, bool, bool]]) -> Iterator[None]:
    """Acquire locks strictly in order (e.g. sota_acct -> sota_queue -> sota_handoff)."""
    if not lock_specs:
        yield
        return
    p, ex, nb = lock_specs[0]
    with acquire_sota_flock(p, exclusive=ex, nonblocking=nb):
        with acquire_ordered_locks(lock_specs[1:]):
            yield


def _scrub_secret_text(text: str, *, pat: str = "", account_id: str = "") -> str:
    safe = str(text)
    if pat:
        safe = safe.replace(pat, "[REDACTED_PAT]")
    if account_id:
        safe = safe.replace(account_id, "[REDACTED_ACCT]")
    return re.sub(r"\b(?:(?:VRTC|VRW|CR|MF|MLT|MX)[A-Za-z0-9_-]*|D[A-Z]{2}[0-9]+)\b", "[REDACTED_ACCT]", safe, flags=re.IGNORECASE)


def redact_account_id(account_id: str) -> str:
    """Return a non-reversible SHA-256 prefix fingerprint for logs/CLI output (never prints raw account_id)."""
    raw = str(account_id or "").strip()
    return f"sha256:{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:12]}"


def _has_symlink_ancestor(path: Path) -> bool:
    """Return True if path contains '..', is a symlink, or traverses any symlinked ancestor component."""
    raw = Path(path)
    if ".." in raw.parts:
        return True
    full = raw if raw.is_absolute() else (Path.cwd() / raw)
    if ".." in full.parts:
        return True
    cur = Path(full.parts[0])
    if cur.is_symlink():
        return True
    for part in full.parts[1:]:
        cur = cur / part
        if cur.is_symlink():
            return True
    return False


def query_systemd_user_unit_states(units: tuple[str, ...]) -> dict[str, dict[str, str]]:
    """Query the live systemd --user manager for effective LoadState/UnitFileState/ActiveState."""
    systemctl_bin = shutil.which("systemctl") or "/usr/bin/systemctl"
    try:
        proc = subprocess.run(
            [systemctl_bin, "--user", "show", *units, "-p", "Id", "-p", "LoadState", "-p", "UnitFileState", "-p", "ActiveState"],
            capture_output=True,
            text=True,
            timeout=5.0,
            check=False,
        )
    except Exception as exc:
        raise ExecutorError(f"failed to query systemd --user manager for masked legacy units: {exc}") from exc
    if proc.returncode != 0:
        raise ExecutorError(f"systemctl --user show failed (rc={proc.returncode})")
    states: dict[str, dict[str, str]] = {}
    for block in proc.stdout.strip().split("\n\n"):
        rec: dict[str, str] = {}
        for line in block.splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                rec[k.strip()] = v.strip()
        uid = rec.get("Id")
        if uid:
            states[uid] = rec
    return states


def validate_credential_isolation_receipt(
    receipt_path: Path,
    *,
    expected_account_id: str,
    expected_account_mode: str = "demo",
    pw_home: Path | None = None,
    systemd_state_probe: Callable[[tuple[str, ...]], dict[str, dict[str, str]]] | None = None,
) -> dict[str, Any]:
    """Authenticate non-secret sota-credential-isolation-receipt/v1 (never contains tokens).

    Supports two explicit modes:
    - `"revoked"` (default): requires exact matching `account_id` and `legacy_credentials_revoked=true`.
    - `"single_host_units_masked"` (Demo VRTC* ONLY): requires `legacy_credentials_revoked=false`,
      exact 4 legacy unit names in `legacy_units_masked`, verified `/dev/null` mask symlinks in
      `<pw_home>/.config/systemd/user/`, effective `systemd --user` `LoadState=masked` /
      `UnitFileState=masked` / `ActiveState=inactive` on real host (or via `systemd_state_probe`),
      and either exact `account_id` or `"VRTC_FROM_ENV"`.
    """
    exp_acct = str(expected_account_id or "").strip()
    if not exp_acct or exp_acct.upper() == DEMO_ENV_ACCOUNT_SENTINEL:
        raise ExecutorError("expected_account_id must be a concrete account_id, not empty or sentinel")
    p = Path(receipt_path)
    if _has_symlink_ancestor(p) or not p.is_file():
        raise ExecutorError(f"missing or symlinked credential isolation receipt (or symlink ancestor): {p}")
    try:
        obj = strict_json_loads_wire(p.read_bytes())
    except Exception:
        raise ExecutorError(f"malformed JSON in credential isolation receipt: {p.name}") from None
    if not isinstance(obj, dict) or obj.get("schema") != CREDENTIAL_RECEIPT_SCHEMA:
        raise ExecutorError(f"invalid credential isolation receipt schema at {p}")
    # Guard against accidentally storing secrets inside the receipt
    for forbidden_k in ("pat", "token", "deriv_pat", "secret", "password"):
        if forbidden_k in {str(k).lower() for k in obj.keys()}:
            raise ExecutorError(f"forbidden secret key {forbidden_k!r} inside non-secret receipt")
    if type(obj.get("account_mode")) is not str or obj["account_mode"].lower() != expected_account_mode.lower():
        raise ExecutorError(f"receipt account_mode does not match expected mode {expected_account_mode!r}")
    attested_by = obj.get("attested_by")
    if type(attested_by) is not str or not attested_by.strip() or _parse_iso_utc(obj.get("attested_utc")) is None:
        raise ExecutorError("credential isolation receipt missing string attested_by or valid attested_utc")

    if "credential_isolation_mode" in obj:
        mode_raw = obj["credential_isolation_mode"]
        if type(mode_raw) is not str:
            raise ExecutorError("credential_isolation_mode must be a string when present")
        iso_mode = mode_raw
    else:
        iso_mode = CREDENTIAL_ISOLATION_MODE_REVOKED

    rcpt_acct_raw = obj.get("account_id")
    if type(rcpt_acct_raw) is not str or not rcpt_acct_raw.strip():
        raise ExecutorError("credential isolation receipt missing string account_id")
    rcpt_acct = rcpt_acct_raw

    if iso_mode == CREDENTIAL_ISOLATION_MODE_REVOKED:
        if expected_account_mode.lower() == "demo" and not is_valid_demo_account_id(exp_acct):
            raise ExecutorError("demo credential isolation receipt requires a valid concrete Deriv Demo account_id")
        if rcpt_acct != exp_acct:
            raise ExecutorError("receipt account_id does not match expected account_id")
        if obj.get("legacy_credentials_revoked") is not True:
            raise ExecutorError("credential isolation receipt does not attest legacy_credentials_revoked=true")
        return obj

    if iso_mode == CREDENTIAL_ISOLATION_MODE_SINGLE_HOST_MASKED:
        if expected_account_mode.lower() != "demo" or not is_valid_demo_account_id(exp_acct):
            raise ExecutorError("single_host_units_masked credential mode is permitted ONLY for Demo accounts")
        if rcpt_acct not in (exp_acct, DEMO_ENV_ACCOUNT_SENTINEL):
            raise ExecutorError(f"receipt account_id does not match expected Demo account or {DEMO_ENV_ACCOUNT_SENTINEL!r}")
        if obj.get("legacy_credentials_revoked") is not False:
            raise ExecutorError("single_host_units_masked receipt must explicitly set legacy_credentials_revoked=false")
        masked_decl = obj.get("legacy_units_masked")
        if (
            not isinstance(masked_decl, list)
            or not all(type(x) is str for x in masked_decl)
            or len(masked_decl) != len(LEGACY_MASKED_UNIT_NAMES)
            or set(masked_decl) != set(LEGACY_MASKED_UNIT_NAMES)
        ):
            raise ExecutorError(
                f"single_host_units_masked receipt must declare exact legacy_units_masked={list(LEGACY_MASKED_UNIT_NAMES)}"
            )
        real_home = canonical_pw_home()
        home_base = Path(pw_home) if pw_home is not None else real_home
        is_real_host = (pw_home is None) or (os.path.abspath(str(home_base)) == os.path.abspath(str(real_home)))
        if is_real_host:
            xdg_cfg = os.environ.get("XDG_CONFIG_HOME")
            if xdg_cfg and xdg_cfg != str(home_base / ".config"):
                raise ExecutorError("single_host_units_masked refuses redirected XDG_CONFIG_HOME")
        systemd_dir = home_base / ".config" / "systemd" / "user"
        for ancestor in (home_base / ".config", home_base / ".config" / "systemd", systemd_dir):
            if ancestor.is_symlink() or not ancestor.is_dir():
                raise ExecutorError(f"single_host_units_masked requires real non-symlink directory {ancestor}")
        for unit_name in LEGACY_MASKED_UNIT_NAMES:
            u_path = systemd_dir / unit_name
            if not u_path.is_symlink() or os.readlink(u_path) != "/dev/null":
                raise ExecutorError(f"single_host_units_masked requires masked /dev/null symlink at {u_path}")
        if is_real_host or systemd_state_probe is not None:
            probe_fn = systemd_state_probe if systemd_state_probe is not None else query_systemd_user_unit_states
            states = probe_fn(LEGACY_MASKED_UNIT_NAMES)
            for unit_name in LEGACY_MASKED_UNIT_NAMES:
                st = states.get(unit_name) or {}
                if (
                    st.get("LoadState") != "masked"
                    or st.get("UnitFileState") != "masked"
                    or st.get("ActiveState") != "inactive"
                ):
                    raise ExecutorError(
                        f"single_host_units_masked requires systemd --user LoadState=masked, UnitFileState=masked, "
                        f"ActiveState=inactive for {unit_name}; got LoadState={st.get('LoadState')!r}, "
                        f"UnitFileState={st.get('UnitFileState')!r}, ActiveState={st.get('ActiveState')!r}"
                    )
        return obj

    raise ExecutorError("unsupported credential_isolation_mode")


# ---------------------------------------------------------------------------
# Non-Bypassable Shared Phase 1 Runtime Config Validation (spec §3.6)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ValidatedSotaRuntimeConfig:
    account_mode: str
    account_id: str
    book_ids: tuple[str, ...]
    stake_cents: int
    stake_usd: float
    max_stake_usd: float
    max_daily_loss_usd: float
    max_drawdown_usd: float
    max_open_contracts: int
    max_usd_factor_open: int
    max_jpy_factor_open: int
    absolute_breakeven_ceiling: float
    ev_margin: float
    max_buy_dispatch_delay_s: float
    queue_db: Path
    log_dir: Path
    credential_isolation_receipt: Path
    repo_root: Path
    pw_home: Path
    lock_paths: dict[str, Path]


def resolve_and_validate_sota_runtime_config(
    *,
    account_mode: str,
    account_id: str,
    book_ids_raw: str | None,
    stake_raw: Any,
    max_stake_usd: float,
    max_daily_loss_usd: float,
    max_drawdown_usd: float,
    max_open_contracts: int,
    max_usd_factor_open: int,
    max_jpy_factor_open: int,
    absolute_breakeven_ceiling: float,
    ev_margin: float,
    max_buy_dispatch_delay_s: float,
    queue_db_raw: str | None,
    log_dir_raw: str,
    credential_isolation_receipt_raw: str | None,
    enforce_3book_cutover: bool = False,
    require_explicit_book_ids: bool = True,
    fixture_home: Path | None = None,
    fixture_repo_root: Path | None = None,
    allow_fixture: bool = False,
) -> ValidatedSotaRuntimeConfig:
    """Single non-bypassable Phase 1 Demo runtime configuration validator (spec §3.6)."""
    fh, fr = validate_fixture_roots(fixture_home, fixture_repo_root, allow_fixture=allow_fixture)
    pw_home = fh if fh is not None else canonical_pw_home()
    repo_root = fr if fr is not None else LEXICAL_REPO_ROOT

    mode = str(account_mode).strip().lower()
    if mode != "demo":
        raise ExecutorError(f"Phase 1 runtime policy permits account_mode='demo' only, got {account_mode!r}")
    transport = os.environ.get("DERIV_WS_TRANSPORT", "options_otp").strip().lower()
    if transport not in {"options_otp", "v3_authorize"}:
        raise ExecutorError(f"unsupported DERIV_WS_TRANSPORT: {transport!r}")
    acct = str(account_id or "").strip()
    if not is_valid_demo_account_id(acct, allow_v3_only=(transport == "v3_authorize")):
        raise ExecutorError("Phase 1 Demo mode requires a concrete Deriv Demo account_id compatible with DERIV_WS_TRANSPORT")

    parsed_books = parse_book_ids_spec(book_ids_raw)
    if require_explicit_book_ids and not parsed_books:
        raise ExecutorError("explicit non-empty --book-ids is required")
    assert parsed_books is not None
    allowed_set = set(PHASE1_ORDERED_BOOK_IDS)
    for b in parsed_books:
        if b not in allowed_set:
            raise ExecutorError(f"unauthorized Phase 1 book_id: {b!r}")
    # Preserve exact priority order among selected authorized books
    idx_pos = [PHASE1_ORDERED_BOOK_IDS.index(b) for b in parsed_books]
    if idx_pos != sorted(idx_pos):
        raise ExecutorError(f"--book-ids violates Phase 1 priority order {PHASE1_ORDERED_BOOK_IDS}: {parsed_books}")
    if enforce_3book_cutover:
        if tuple(parsed_books) != PHASE1_ORDERED_BOOK_IDS:
            raise ExecutorError(
                f"--enforce-3book-cutover requires exact ordered 3 books {PHASE1_ORDERED_BOOK_IDS}, got {tuple(parsed_books)}"
            )
        if (max_open_contracts, max_usd_factor_open, max_jpy_factor_open) != (3, 3, 3):
            raise ExecutorError(
                f"--enforce-3book-cutover requires 3/3/3 caps, got {(max_open_contracts, max_usd_factor_open, max_jpy_factor_open)}"
            )

    stake_cents = parse_exact_whole_cents(stake_raw, allow_zero=False, allow_negative=False)
    if stake_cents != 100:
        raise ExecutorError(f"Phase 1 flat stake must be exact 100 USD cents ($1.00), got {stake_cents} cents ({stake_raw!r})")

    if not (1.0 <= float(max_stake_usd) <= 5.0):
        raise ExecutorError(f"max_stake_usd={max_stake_usd} outside [1, 5]")
    if not (1.0 <= float(max_daily_loss_usd) <= 25.0):
        raise ExecutorError(f"max_daily_loss_usd={max_daily_loss_usd} outside [1, 25]")
    if not (1.0 <= float(max_drawdown_usd) <= 50.0):
        raise ExecutorError(f"max_drawdown_usd={max_drawdown_usd} outside [1, 50]")
    for cap_name, cap_val in (
        ("max_open_contracts", max_open_contracts),
        ("max_usd_factor_open", max_usd_factor_open),
        ("max_jpy_factor_open", max_jpy_factor_open),
    ):
        if isinstance(cap_val, bool) or not (1 <= int(cap_val) <= 3):
            raise ExecutorError(f"{cap_name}={cap_val} outside [1, 3]")
    if not (0.0 < float(absolute_breakeven_ceiling) <= 0.60):
        raise ExecutorError(f"absolute_breakeven_ceiling={absolute_breakeven_ceiling} > 0.60")
    if not (0.005 <= float(ev_margin) < 1.0):
        raise ExecutorError(f"ev_margin={ev_margin} < 0.005")
    if not (0.0 < float(max_buy_dispatch_delay_s) <= 4.0):
        raise ExecutorError(f"max_buy_dispatch_delay_s={max_buy_dispatch_delay_s} outside (0, 4.0]")

    canon_q = canonical_sota_queue_path("demo", repo_root=repo_root)
    if queue_db_raw:
        q_in = Path(queue_db_raw)
        q_abs = q_in if q_in.is_absolute() else (repo_root / q_in)
        if os.path.abspath(str(q_abs)) != os.path.abspath(str(canon_q)):
            raise ExecutorError(f"non-canonical --queue-db refused: {q_abs} != {canon_q}")
    q_path = canon_q

    if not credential_isolation_receipt_raw:
        raise ExecutorError("missing --credential-isolation-receipt")
    rcpt_path = Path(credential_isolation_receipt_raw)
    validate_credential_isolation_receipt(rcpt_path, expected_account_id=acct, expected_account_mode="demo", pw_home=pw_home)

    log_dir = Path(log_dir_raw) if Path(log_dir_raw).is_absolute() else (repo_root / log_dir_raw)
    lock_paths = canonical_sota_lock_paths(q_path, pw_home=pw_home)

    return ValidatedSotaRuntimeConfig(
        account_mode="demo",
        account_id=acct,
        book_ids=tuple(parsed_books),
        stake_cents=100,
        stake_usd=1.0,
        max_stake_usd=float(max_stake_usd),
        max_daily_loss_usd=float(max_daily_loss_usd),
        max_drawdown_usd=float(max_drawdown_usd),
        max_open_contracts=int(max_open_contracts),
        max_usd_factor_open=int(max_usd_factor_open),
        max_jpy_factor_open=int(max_jpy_factor_open),
        absolute_breakeven_ceiling=float(absolute_breakeven_ceiling),
        ev_margin=float(ev_margin),
        max_buy_dispatch_delay_s=float(max_buy_dispatch_delay_s),
        queue_db=q_path,
        log_dir=log_dir,
        credential_isolation_receipt=rcpt_path,
        repo_root=repo_root,
        pw_home=pw_home,
        lock_paths=lock_paths,
    )


# ---------------------------------------------------------------------------
# 5-Class Generation-Bound Async Client (spec §3.3)
# ---------------------------------------------------------------------------

_CLASS_ALLOWED_PRIMARY_KEYS: dict[str, frozenset[str]] = {
    "public_read": frozenset({"ticks", "ticks_history", "forget"}),
    "identity": frozenset({"authorize", "balance", "time"}),
    "account_read": frozenset({"portfolio", "statement", "profit_table", "proposal_open_contract", "balance"}),
    "proposal": frozenset({"proposal"}),
    "trade_mutation": frozenset({"buy", "sell"}),
}
_ALL_MUTATION_KEYS = frozenset({"buy", "sell"})


class SotaDerivAsyncClient(DerivAsyncClient):
    """5-class capability & generation-bound WebSocket client forbidding untyped generic request()."""

    def __init__(
        self,
        url: str = PUBLIC_WS_URL,
        *,
        allow_buy: bool = False,
        expected_account_id: str | None = None,
        handoff_lock_path: Path | None = None,
        pre_send_guard_fn: Callable[[str, dict[str, Any]], None] | None = None,
        ws_opener: Callable[[str], Any] | None = None,
        **kwargs: Any,
    ):
        super().__init__(url=url, **kwargs)
        self.allow_buy = bool(allow_buy)
        self.expected_account_id = expected_account_id
        self.handoff_lock_path = handoff_lock_path
        self.pre_send_guard_fn = pre_send_guard_fn
        self._ws_opener = ws_opener
        self._generation: int = 0
        self._verified_generation: int | None = None
        self._verified_account_id: str | None = None
        self._bootstrap_balance_used_gen: int | None = None
        self._send_lock = asyncio.Lock()
        self.outbound_log: list[tuple[str, dict[str, Any]]] = []

    @property
    def generation(self) -> int:
        return self._generation

    @property
    def verified_account_id(self) -> str | None:
        return self._verified_account_id if self._verified_generation == self._generation else None

    def is_identity_verified(self) -> bool:
        return (
            self._ws is not None
            and self._verified_generation is not None
            and self._verified_generation == self._generation
            and bool(self._verified_account_id)
        )

    def _invalidate_identity(self) -> None:
        self._verified_generation = None
        self._verified_account_id = None
        self._bootstrap_balance_used_gen = None

    async def _open(self, *, refresh_url: bool = False) -> Any:
        self._invalidate_identity()
        self._generation += 1
        if refresh_url:
            await self._refresh_url()
        if self._ws_opener is not None:
            ws = self._ws_opener(self.url)
            if asyncio.iscoroutine(ws):
                ws = await ws
            return ws
        conn = websockets.connect(
            self.url,
            compression=None,
            open_timeout=self.open_timeout,
            ping_interval=self.ping_interval,
            ping_timeout=self.ping_interval * 2,
            max_size=2**24,
        )
        # Never follow HTTP 3xx redirects after URL validation (prevents /demo -> /live redirect bypass!)
        conn.process_redirect = lambda exc: exc  # type: ignore[method-assign]
        return await conn

    async def close(self) -> None:
        self._invalidate_identity()
        await super().close()

    async def request(self, payload: dict[str, Any], timeout: float | None = None) -> dict[str, Any]:
        raise ExecutorError("generic request() is forbidden on SotaDerivAsyncClient; use typed capability methods")

    def _validate_send_capability(
        self,
        send_class: str,
        payload: dict[str, Any],
        *,
        expected_generation: int | None = None,
    ) -> None:
        if send_class not in _CLASS_ALLOWED_PRIMARY_KEYS:
            raise ExecutorError(f"unknown send_class {send_class!r}")
        allowed_keys = _CLASS_ALLOWED_PRIMARY_KEYS[send_class]
        present = [k for k in payload.keys() if k in allowed_keys]
        if len(present) != 1:
            raise ExecutorError(f"payload keys {sorted(payload.keys())} invalid for send_class {send_class!r}")
        primary = present[0]
        # Reject mixed cross-class primary keys
        all_known = set().union(*_CLASS_ALLOWED_PRIMARY_KEYS.values())
        other_primary = [k for k in payload.keys() if k in all_known and k != primary]
        if other_primary:
            raise ExecutorError(f"mixed capability keys rejected: {primary} + {other_primary}")

        if expected_generation is not None and expected_generation != self._generation:
            raise ExecutorError(f"stale transport generation {expected_generation} != current {self._generation}")

        if send_class == "public_read":
            return

        if send_class == "identity":
            if primary == "time" and not self.is_identity_verified():
                raise ExecutorError("venue time request requires verified identity on current generation")
            if primary == "balance" and self._bootstrap_balance_used_gen == self._generation:
                raise ExecutorError("options_otp bootstrap balance already consumed for current generation")
            return

        # account_read, proposal, trade_mutation all require currently verified (ws, generation, account_id)
        if not self.is_identity_verified():
            raise ExecutorError(f"send_class {send_class!r} requires verified identity on current generation")
        if self.expected_account_id and self._verified_account_id != self.expected_account_id:
            raise ExecutorError(
                f"verified account {redact_account_id(str(self._verified_account_id or ''))} != expected {redact_account_id(self.expected_account_id)}"
            )

        if send_class == "proposal":
            if self.pre_send_guard_fn is not None:
                self.pre_send_guard_fn(primary, payload)
        elif send_class == "trade_mutation":
            if not self.allow_buy:
                raise ExecutorError(f"trade_mutation ({primary}) refused with 0 bytes sent because allow_buy=False")
            if self.pre_send_guard_fn is not None:
                self.pre_send_guard_fn(primary, payload)

    async def _exclusive_send_text(
        self,
        send_class: str,
        payload: dict[str, Any],
        *,
        timeout: float | None = None,
        expected_generation: int | None = None,
        expect_response: bool = True,
        send_guard_fn: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> dict[str, Any] | int:
        """Single gated outbound transport funnel enforcing capability class, generation & handoff lock."""
        if self._dead:
            raise DerivAPIError(f"client dead: {self._dead}")
        if self._closing:
            raise DerivAPIError("client closed")
        if expect_response:
            try:
                await asyncio.wait_for(self._connected.wait(), timeout or self.request_timeout)
            except asyncio.TimeoutError:
                raise DerivAPIError("not connected within request timeout") from None

        async with self._send_lock:
            self._validate_send_capability(send_class, payload, expected_generation=expected_generation)
            if send_class == "identity" and "balance" in payload:
                self._bootstrap_balance_used_gen = self._generation

            ws = self._ws
            if ws is None:
                raise DerivAPIError("websocket closed before send")
            self._req_id += 1
            rid = self._req_id
            msg = dict(payload)
            msg["req_id"] = rid
            # Reject non-finite values at outbound serialization too
            raw_text = json.dumps(msg, allow_nan=False)

            fut: asyncio.Future | None = None
            if expect_response:
                fut = asyncio.get_running_loop().create_future()
                self._pending[rid] = fut

            need_handoff_lock = send_class in ("proposal", "trade_mutation") and self.handoff_lock_path is not None
            try:
                if need_handoff_lock:
                    assert self.handoff_lock_path is not None
                    with acquire_sota_flock(self.handoff_lock_path, exclusive=False, nonblocking=True):
                        # Re-check capability, client guard & per-call guard INSIDE handoff lock before sending a single byte!
                        self._validate_send_capability(send_class, payload, expected_generation=expected_generation)
                        if send_guard_fn is not None:
                            send_guard_fn(send_class, payload)
                        self.outbound_log.append((send_class, dict(msg)))
                        await ws.send(raw_text)
                else:
                    if send_guard_fn is not None:
                        send_guard_fn(send_class, payload)
                    self.outbound_log.append((send_class, dict(msg)))
                    await ws.send(raw_text)
            except Exception as exc:
                if expect_response:
                    self._pending.pop(rid, None)
                if isinstance(exc, (ExecutorError, DerivAPIError)):
                    raise
                raise DerivAPIError(f"transport send failed: {exc}") from exc

        if not expect_response:
            return rid

        assert fut is not None
        try:
            return await asyncio.wait_for(fut, timeout or self.request_timeout)
        except asyncio.TimeoutError:
            raise DerivAPIError(f"timeout waiting for req_id={rid} class={send_class}") from None
        except websockets.exceptions.ConnectionClosed as exc:
            raise DerivAPIError(f"connection lost during request: {exc}") from exc
        finally:
            self._pending.pop(rid, None)

    async def request_public_read(self, payload: dict[str, Any], timeout: float | None = None) -> dict[str, Any]:
        res = await self._exclusive_send_text("public_read", payload, timeout=timeout, expect_response=True)
        assert isinstance(res, dict)
        return res

    async def request_identity(self, payload: dict[str, Any], timeout: float | None = None) -> dict[str, Any]:
        res = await self._exclusive_send_text("identity", payload, timeout=timeout, expect_response=True)
        assert isinstance(res, dict)
        return res

    async def request_account_read(
        self,
        payload: dict[str, Any],
        timeout: float | None = None,
        *,
        expected_generation: int | None = None,
    ) -> dict[str, Any]:
        res = await self._exclusive_send_text(
            "account_read", payload, timeout=timeout, expected_generation=expected_generation, expect_response=True
        )
        assert isinstance(res, dict)
        return res

    async def request_proposal(
        self,
        payload: dict[str, Any],
        timeout: float | None = None,
        *,
        expected_generation: int | None = None,
    ) -> dict[str, Any]:
        res = await self._exclusive_send_text(
            "proposal", payload, timeout=timeout, expected_generation=expected_generation, expect_response=True
        )
        assert isinstance(res, dict)
        return res

    async def request_trade_mutation(
        self,
        payload: dict[str, Any],
        timeout: float | None = None,
        *,
        expected_generation: int | None = None,
        send_guard_fn: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        res = await self._exclusive_send_text(
            "trade_mutation",
            payload,
            timeout=timeout,
            expected_generation=expected_generation,
            expect_response=True,
            send_guard_fn=send_guard_fn,
        )
        assert isinstance(res, dict)
        return res

    async def fetch_venue_time_epoch(self, *, timeout: float = 5.0, expected_generation: int | None = None) -> int:
        """Fetch fresh same-generation venue time epoch on a verified transport."""
        time_resp = await self._exclusive_send_text(
            "identity", {"time": 1}, timeout=timeout, expected_generation=expected_generation, expect_response=True
        )
        assert isinstance(time_resp, dict)
        t_raw = time_resp.get("time")
        if isinstance(t_raw, bool) or t_raw is None:
            raise ExecutorError(f"invalid venue time response: {time_resp!r}")
        try:
            venue_ep = int(str(t_raw))
        except Exception as exc:
            raise ExecutorError(f"non-integer venue time: {t_raw!r}") from exc
        if venue_ep <= 0:
            raise ExecutorError(f"non-positive venue time: {venue_ep}")
        return venue_ep

    async def establish_identity(
        self,
        *,
        account_id: str,
        account_mode: str,
        transport_kind: str,
        pat: str = "",
        timeout: float = 10.0,
    ) -> int:
        """Establish generation-bound identity via v3_authorize OR options_otp bootstrap balance, then venue time."""
        gen = self._generation
        if transport_kind == "v3_authorize":
            try:
                auth_resp = await self.request_identity({"authorize": pat}, timeout=timeout)
            except Exception as exc:
                safe_msg = _scrub_secret_text(str(exc), pat=pat, account_id=account_id)
                raise ExecutorError(f"v3_authorize failed: {safe_msg}") from None
            info = auth_resp.get("authorize") or {}
            loginid = str(info.get("loginid") or "").strip()
            ccy = str(info.get("currency") or "").strip().upper()
            is_virt = info.get("is_virtual")
            if not loginid or loginid != account_id:
                raise ExecutorError(
                    f"PAT authorized account mismatch: got={redact_account_id(loginid)} != expected={redact_account_id(account_id)}"
                )
            if ccy != "USD":
                raise ExecutorError(f"account currency must be USD, got {ccy!r}")
            if account_mode == "demo" and is_virt not in {1, True}:
                raise ExecutorError("demo mode refused: token authorized non-virtual account")
            if account_mode == "demo" and not is_valid_demo_account_id(loginid, allow_v3_only=True):
                raise ExecutorError(f"v3_authorize non-demo account ({redact_account_id(loginid)}) in demo mode")
            if account_mode == "live" and is_virt in {1, True}:
                raise ExecutorError("live mode refused: token authorized virtual account")
        elif transport_kind == "options_otp":
            try:
                bal_resp = await self.request_identity({"balance": 1}, timeout=timeout)
            except Exception as exc:
                safe_msg = _scrub_secret_text(str(exc), pat=pat, account_id=account_id)
                raise ExecutorError(f"OTP bootstrap balance failed: {safe_msg}") from None
            binfo = bal_resp.get("balance") or {}
            loginid = str(binfo.get("loginid") or "").strip()
            ccy = str(binfo.get("currency") or "").strip().upper()
            is_virt_otp = binfo.get("is_virtual")
            if not loginid or loginid != account_id:
                raise ExecutorError(
                    f"OTP bootstrap balance missing or mismatched account: got={redact_account_id(loginid)} != expected={redact_account_id(account_id)}"
                )
            if account_mode == "demo" and not is_valid_demo_account_id(loginid):
                raise ExecutorError(f"OTP bootstrap balance non-demo account ({redact_account_id(loginid)}) in demo mode")
            # MAJOR #1: reject conflicting virtual-account evidence on OTP bootstrap balance!
            if is_virt_otp is not None:
                if account_mode == "demo" and is_virt_otp not in {1, True}:
                    raise ExecutorError(f"OTP bootstrap conflicting is_virtual={is_virt_otp!r} in demo mode")
                if account_mode == "live" and is_virt_otp in {1, True}:
                    raise ExecutorError(f"OTP bootstrap conflicting is_virtual={is_virt_otp!r} in live mode")
            if ccy != "USD":
                raise ExecutorError(f"OTP account currency must be USD, got {ccy!r}")
        else:
            raise ExecutorError(f"unsupported transport_kind {transport_kind!r}")

        if gen != self._generation:
            self._invalidate_identity()
            raise ExecutorError("transport rotated during identity establishment")
        self._verified_generation = gen
        self._verified_account_id = account_id
        # Fetch verified venue clock on same generation; clear verification immediately on ANY failure (MAJOR #1)!
        try:
            return await self.fetch_venue_time_epoch(timeout=timeout, expected_generation=gen)
        except Exception:
            self._invalidate_identity()
            raise

    async def subscribe(self, payload: dict[str, Any], queue_maxsize: int | None = None) -> Subscription:
        sub_payload = dict(payload)
        sub_payload["subscribe"] = 1
        sub = Subscription(sub_payload, queue_maxsize or self.queue_maxsize)
        resp = await self.request_public_read(sub_payload)
        sub_id = (resp.get("subscription") or {}).get("id")
        if not sub_id:
            raise DerivAPIError(f"no subscription id in response keys={sorted(resp)}")
        sub.sub_id = sub_id
        self._subs[sub_id] = sub
        self._registry.append(sub)
        self._offer(sub, resp)
        for frame in self._pre_sub.pop(sub_id, []):
            self._offer(sub, frame)
        return sub

    async def unsubscribe(self, sub: Subscription) -> None:
        sub.active = False
        if sub in self._registry:
            self._registry.remove(sub)
        if sub.sub_id:
            self._subs.pop(sub.sub_id, None)
            await self.request_public_read({"forget": sub.sub_id})

    async def _read_loop(self) -> None:
        while not self._closing:
            try:
                raw = await self._ws.recv()
            except websockets.exceptions.ConnectionClosed as exc:
                self._invalidate_identity()
                if self._closing:
                    return
                if not await self._reconnect_and_resubscribe(exc):
                    return
                continue
            except asyncio.CancelledError:
                return
            try:
                frame = strict_json_loads_wire(raw)
            except (json.JSONDecodeError, DerivAPIError, TypeError) as exc:
                self.stats["decode_errors"] += 1
                self._emit("decode_error", raw_prefix=str(raw)[:80], error=str(exc)[:120])
                continue
            self._dispatch(frame)

    async def _reconnect_and_resubscribe(self, exc: Exception) -> bool:
        t0 = time.monotonic()
        self._invalidate_identity()
        self._connected.clear()
        self.stats["reconnects"] += 1
        self._fail_pending(DerivAPIError(f"connection lost: {exc}"))
        self._emit("disconnected", reason=str(exc))
        delay = self.reconnect_base_s
        attempts = 0
        while not self._closing:
            attempts += 1
            if self.reconnect_max_attempts is not None and attempts > self.reconnect_max_attempts:
                self._dead = f"reconnect exhausted after {attempts - 1} attempts"
                self._fail_pending(DerivAPIError(self._dead))
                self._emit("reconnect_exhausted", attempts=attempts - 1)
                return False
            try:
                self._ws = await self._open(refresh_url=self._url_factory is not None)
                break
            except Exception as open_exc:
                self._emit("reconnect_failed", attempt=attempts, error=str(open_exc)[:120])
                await asyncio.sleep(delay)
                delay = min(delay * 2, self.reconnect_cap_s)
        if self._closing:
            return False
        self._subs.clear()
        self._pre_sub.clear()
        pending_subs: dict[int, Subscription] = {}
        try:
            for sub in list(self._registry):
                if not sub.active:
                    continue
                rid = await self._exclusive_send_text("public_read", dict(sub.payload), expect_response=False)
                assert isinstance(rid, int)
                pending_subs[rid] = sub
            deadline = time.monotonic() + self.request_timeout
            while pending_subs:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                raw = await asyncio.wait_for(self._ws.recv(), timeout=remaining)
                try:
                    frame = strict_json_loads_wire(raw)
                except (json.JSONDecodeError, DerivAPIError, TypeError):
                    self.stats["decode_errors"] += 1
                    continue
                rid = frame.get("req_id")
                if rid in pending_subs:
                    sub = pending_subs.pop(rid)
                    if "error" in frame:
                        sub.error = f"resubscribe failed: {frame['error']}"
                        continue
                    new_id = (frame.get("subscription") or {}).get("id")
                    if not new_id:
                        sub.error = "resubscribe returned no subscription id"
                        continue
                    sub.sub_id = new_id
                    sub.resubscribes += 1
                    sub.error = None
                    self._subs[new_id] = sub
                    self._offer(sub, frame)
                else:
                    self._dispatch(frame)
        except (asyncio.TimeoutError, websockets.exceptions.ConnectionClosed) as sub_exc:
            self._emit("resubscribe_transport_failure", error=str(sub_exc)[:200])
        self._last_frame_at = time.monotonic()
        self._connected.set()
        self.stats["resubscribe_latency_s"].append(round(time.monotonic() - t0, 3))
        return True


def request_authenticated_ws_url(env: DerivEnv, account_mode: str, timeout: float = 10.0) -> tuple[str, str]:
    """Obtain authenticated WebSocket URL via Options v1 OTP endpoint (or classic v3 token transport if configured)."""
    assert_non_us_deriv_host()
    transport = os.environ.get("DERIV_WS_TRANSPORT", "options_otp").strip().lower()
    if transport == "v3_authorize":
        ws_url = CLASSIC_V3_WS_TEMPLATE.format(app_id=env.app_id)
        validate_account_mode(
            account_mode=account_mode, account_id=env.account_id, ws_url=ws_url, expected_transport="v3_authorize"
        )
        return ws_url, "v3_authorize"
    if transport != "options_otp":
        raise ExecutorError(f"unsupported DERIV_WS_TRANSPORT: {transport!r}")

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
            raw_payload = resp.read()
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        safe_body = _scrub_secret_text(re.sub(r"otp=[^&\s\"']+", "otp=[REDACTED]", body), pat=env.pat, account_id=env.account_id)[:200]
        raise DerivAPIError(f"OTP request failed: HTTP {exc.code}: {safe_body}") from None
    except urllib.error.URLError as exc:
        safe_err = _scrub_secret_text(re.sub(r"otp=[^&\s\"']+", "otp=[REDACTED]", str(exc)), pat=env.pat, account_id=env.account_id)[:200]
        raise DerivAPIError(f"OTP request failed: {safe_err}") from None
    try:
        payload = strict_json_loads_wire(raw_payload)
    except Exception:
        raise DerivAPIError("OTP response malformed JSON") from None

    data = payload.get("data", payload) if isinstance(payload, dict) else {}
    ws_url = data.get("url") or data.get("websocket_url") if isinstance(data, dict) else None
    if not isinstance(ws_url, str) or not ws_url.startswith("wss://"):
        raise DerivAPIError("OTP response missing wss:// URL")
    validate_account_mode(
        account_mode=account_mode, account_id=env.account_id, ws_url=ws_url, expected_transport="options_otp"
    )
    return ws_url, "options_otp"


# ---------------------------------------------------------------------------
# Account-Bound v3 SQLite Queue, Read-Only vs Writable PRAGMAs & Adoption (spec §3.4)
# ---------------------------------------------------------------------------

V3_SIGNALS_DDL = """
CREATE TABLE sota_signals (
    signal_id TEXT PRIMARY KEY,
    book_id TEXT NOT NULL,
    pair TEXT NOT NULL,
    horizon_minutes INTEGER NOT NULL,
    side TEXT NOT NULL CHECK(side IN ('UP','DOWN','NEUTRAL')),
    lane TEXT NOT NULL DEFAULT 'wall',
    offset_id INTEGER NOT NULL DEFAULT 0,
    signal_close_utc TEXT NOT NULL,
    effective_floor REAL NOT NULL,
    calibrated_win_prob REAL,
    payload_json TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN (
        'queued','buy_intent','buy_unknown','bought',
        'buy_refused_pre_send','rejected','expired','cancelled'
    )),
    proposal_id TEXT,
    contract_id TEXT,
    reason TEXT,
    created_utc TEXT NOT NULL,
    updated_utc TEXT NOT NULL,
    UNIQUE(book_id, pair, horizon_minutes, side, lane, offset_id, signal_close_utc)
)
"""

V3_CONTRACTS_DDL = """
CREATE TABLE sota_contracts (
    contract_id TEXT PRIMARY KEY,
    signal_id TEXT NOT NULL UNIQUE REFERENCES sota_signals(signal_id),
    book_id TEXT NOT NULL,
    book_content_id TEXT,
    account_id TEXT NOT NULL,
    account_mode TEXT NOT NULL CHECK(account_mode IN ('demo','live')),
    pair TEXT NOT NULL,
    horizon_minutes INTEGER NOT NULL,
    side TEXT NOT NULL CHECK(side IN ('UP','DOWN')),
    contract_family TEXT NOT NULL DEFAULT 'risefall',
    is_migrated_legacy INTEGER NOT NULL DEFAULT 0 CHECK(is_migrated_legacy IN (0,1)),
    cost_unknown INTEGER NOT NULL DEFAULT 0 CHECK(cost_unknown IN (0,1)),
    buy_price_cents INTEGER CHECK(buy_price_cents IS NULL OR buy_price_cents > 0),
    payout_cents INTEGER CHECK(payout_cents IS NULL OR payout_cents > 0),
    sell_price_cents INTEGER CHECK(sell_price_cents IS NULL OR sell_price_cents >= 0),
    profit_cents INTEGER,
    stake_authoritative INTEGER NOT NULL DEFAULT 0 CHECK(stake_authoritative IN (0,1)),
    venue_start_epoch INTEGER,
    venue_end_epoch INTEGER,
    venue_terminal_status TEXT,
    venue_underlying_symbol TEXT,
    venue_contract_type TEXT,
    signal_close_utc TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('open','unknown','won','lost','sold','cancelled','expired')),
    proposal_id TEXT,
    expected_expiry_utc TEXT,
    raw_hash TEXT,
    sell_ack_cents INTEGER,
    opened_utc TEXT NOT NULL,
    closed_utc TEXT,
    updated_utc TEXT NOT NULL,
    CHECK(is_migrated_legacy = 1 OR (book_content_id IS NOT NULL AND length(book_content_id) > 0)),
    CHECK(status != 'expired' OR is_migrated_legacy = 1),
    CHECK(
        status NOT IN ('won','lost','sold')
        OR (
            cost_unknown = 0
            AND buy_price_cents IS NOT NULL
            AND sell_price_cents IS NOT NULL
            AND profit_cents IS NOT NULL
            AND (sell_price_cents - buy_price_cents) = profit_cents
        )
    )
)
"""

V3_RISK_STATE_DDL = """
CREATE TABLE sota_risk_state (
    key TEXT PRIMARY KEY,
    value_json TEXT NOT NULL,
    updated_utc TEXT NOT NULL
)
"""


def open_sota_queue_readonly(db_path: Path) -> sqlite3.Connection:
    """Open SQLite source strictly read-only without ANY writable PRAGMAs (zero file mutations)."""
    p = Path(db_path)
    if not p.is_file() or p.is_symlink():
        raise ExecutorError(f"missing or symlinked SQLite file: {p}")
    conn = sqlite3.connect(f"file:{p.resolve()}?mode=ro", uri=True, isolation_level=None)
    conn.row_factory = sqlite3.Row
    return conn


def _configure_writable_pragmas_outside_txn(conn: sqlite3.Connection) -> None:
    """Configure and assert writable SQLite 3.45.1 PRAGMAs OUTSIDE any transaction (spec §3.4)."""
    if conn.in_transaction:
        raise ExecutorError("writable PRAGMAs must be configured outside BEGIN")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("PRAGMA foreign_keys=ON")
    fk = int(conn.execute("PRAGMA foreign_keys").fetchone()[0])
    if fk != 1:
        raise ExecutorError(f"PRAGMA foreign_keys did not become 1 (got {fk})")
    jm = str(conn.execute("PRAGMA journal_mode=WAL").fetchone()[0]).lower()
    if jm != "wal":
        raise ExecutorError(f"PRAGMA journal_mode did not become 'wal' (got {jm!r})")
    conn.execute("PRAGMA synchronous=FULL")


def create_v3_queue_schema_atomic(
    conn: sqlite3.Connection,
    *,
    account_id: str,
    account_mode: str = "demo",
    reason: str = "first_use",
) -> None:
    """Create full v3 DDL + queue_account_binding + user_version=3 inside ONE BEGIN IMMEDIATE (zero executescript)."""
    if conn.in_transaction:
        raise ExecutorError("cannot start v3 DDL transaction while already in transaction")
    now = utc_iso()
    conn.execute("BEGIN IMMEDIATE")
    try:
        conn.execute(V3_SIGNALS_DDL)
        conn.execute(V3_CONTRACTS_DDL)
        conn.execute(V3_RISK_STATE_DDL)
        binding = {
            "account_id": account_id,
            "account_mode": account_mode,
            "bound_utc": now,
            "reason": reason,
            "schema_version": 3,
        }
        conn.execute(
            "INSERT INTO sota_risk_state (key, value_json, updated_utc) VALUES ('queue_account_binding', ?, ?)",
            (json.dumps(binding, sort_keys=True), now),
        )
        conn.execute("PRAGMA user_version = 3")
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise


def verify_v3_queue_binding(
    conn: sqlite3.Connection,
    *,
    expected_account_id: str,
    expected_account_mode: str = "demo",
) -> dict[str, Any]:
    uv = int(conn.execute("PRAGMA user_version").fetchone()[0])
    if uv != 3:
        raise ExecutorError(f"queue DB is not bound v3 (user_version={uv})")
    row = conn.execute("SELECT value_json FROM sota_risk_state WHERE key='queue_account_binding'").fetchone()
    if row is None:
        raise ExecutorError("v3 queue missing queue_account_binding")
    binding = json.loads(str(row[0]))
    if binding.get("account_id") != expected_account_id or str(binding.get("account_mode") or "").lower() != expected_account_mode.lower():
        raw_mode = str(binding.get("account_mode") or "").lower()
        safe_mode = raw_mode if raw_mode in ("demo", "live") else "<invalid>"
        raise ExecutorError(
            f"queue account binding mismatch: got=({redact_account_id(str(binding.get('account_id') or ''))}, {safe_mode!r}) "
            f"!= expected=({redact_account_id(expected_account_id)}, {expected_account_mode!r})"
        )
    return binding


def connect_sota_queue_v3(
    db_path: Path,
    *,
    expected_account_id: str,
    expected_account_mode: str = "demo",
) -> sqlite3.Connection:
    """Open an already-bound writable v3 queue with PRAGMAs asserted outside BEGIN."""
    p = Path(db_path)
    if not p.is_file() or p.is_symlink():
        raise ExecutorError(f"missing or symlinked v3 queue DB: {p}")
    conn = sqlite3.connect(str(p), timeout=5.0, isolation_level=None)
    conn.row_factory = sqlite3.Row
    try:
        _configure_writable_pragmas_outside_txn(conn)
        verify_v3_queue_binding(conn, expected_account_id=expected_account_id, expected_account_mode=expected_account_mode)
        return conn
    except Exception:
        conn.close()
        raise


def _table_names(conn: sqlite3.Connection) -> set[str]:
    return {str(r[0]) for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}


def _is_empty_predecessor_db(conn: sqlite3.Connection) -> bool:
    uv = int(conn.execute("PRAGMA user_version").fetchone()[0])
    if uv != 0:
        return False
    tables = _table_names(conn) - {"sqlite_sequence"}
    if not tables:
        return True
    for t in tables:
        if int(conn.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]) > 0:
            return False
    return True


def _validate_first_use_record(
    first_use_record_path: Path,
    *,
    account_id: str,
    account_mode: str,
    rcpt_obj: dict[str, Any],
) -> dict[str, Any]:
    fur_p = Path(first_use_record_path)
    if _has_symlink_ancestor(fur_p) or not fur_p.is_file():
        raise ExecutorError(f"missing or symlinked first_use record JSON: {fur_p}")
    try:
        fur = strict_json_loads_wire(fur_p.read_bytes())
    except Exception:
        raise ExecutorError(f"malformed JSON in first_use record: {fur_p.name}") from None
    if not isinstance(fur, dict):
        raise ExecutorError("invalid first_use record: expected JSON object")
    for forbidden_k in ("pat", "token", "deriv_pat", "secret", "password"):
        if forbidden_k in {str(k).lower() for k in fur.keys()}:
            raise ExecutorError(f"forbidden secret key {forbidden_k!r} inside first_use record")
    allow_env_sentinel = (
        rcpt_obj.get("credential_isolation_mode") == CREDENTIAL_ISOLATION_MODE_SINGLE_HOST_MASKED
        and account_mode == "demo"
        and is_valid_demo_account_id(account_id)
    )
    fur_acct = fur.get("account_id")
    fur_acct_ok = (type(fur_acct) is str) and (
        fur_acct == account_id or (allow_env_sentinel and fur_acct == DEMO_ENV_ACCOUNT_SENTINEL)
    )
    if (
        fur.get("schema") != FIRST_USE_RECORD_SCHEMA
        or not fur_acct_ok
        or type(fur.get("account_mode")) is not str
        or fur["account_mode"].lower() != account_mode
    ):
        raise ExecutorError("invalid or mismatched first_use record")
    return fur


def init_v3_queue_first_use(
    *,
    queue_db: Path,
    account_id: str,
    account_mode: str,
    first_use_record_path: Path,
    receipt_path: Path,
    pw_home: Path | None = None,
) -> dict[str, Any]:
    """--init-queue --reason first_use: creates persistent 3 flocks + single-txn v3 DB (idempotent on empty v3)."""
    if account_mode != "demo" or not is_valid_demo_account_id(account_id):
        raise ExecutorError("Phase 1 --init-queue requires a concrete demo account")
    rcpt_obj = validate_credential_isolation_receipt(
        receipt_path, expected_account_id=account_id, expected_account_mode=account_mode, pw_home=pw_home
    )
    _validate_first_use_record(
        first_use_record_path, account_id=account_id, account_mode=account_mode, rcpt_obj=rcpt_obj
    )

    locks = canonical_sota_lock_paths(queue_db, pw_home=pw_home)
    with acquire_ordered_locks([(locks["acct"], True, True), (locks["queue"], True, True)]):
        # Ensure persistent handoff lockfile exists (0600)
        with acquire_sota_flock(locks["handoff"], exclusive=True, nonblocking=True):
            pass
        queue_db.parent.mkdir(parents=True, exist_ok=True)
        if queue_db.exists():
            ro = open_sota_queue_readonly(queue_db)
            try:
                uv = int(ro.execute("PRAGMA user_version").fetchone()[0])
                if uv == 3:
                    verify_v3_queue_binding(ro, expected_account_id=account_id, expected_account_mode=account_mode)
                    n_sig = int(ro.execute("SELECT COUNT(*) FROM sota_signals").fetchone()[0])
                    n_con = int(ro.execute("SELECT COUNT(*) FROM sota_contracts").fetchone()[0])
                    if n_sig != 0 or n_con != 0:
                        raise ExecutorError("refusing first_use init on non-empty v3 queue")
                    return {"status": "idempotent_empty_v3", "queue_db": str(queue_db), "account_id": account_id}
                if not _is_empty_predecessor_db(ro):
                    raise ExecutorError("refusing first_use init on non-empty legacy predecessor DB; use adoption")
            finally:
                ro.close()
            # Empty user_version=0 interrupted-init state: drop any empty tables and create v3 atomically
            conn = sqlite3.connect(str(queue_db), timeout=5.0, isolation_level=None)
            conn.row_factory = sqlite3.Row
            try:
                _configure_writable_pragmas_outside_txn(conn)
                tables = sorted(_table_names(conn) - {"sqlite_sequence"})
                conn.execute("BEGIN IMMEDIATE")
                try:
                    for t in tables:
                        conn.execute(f'DROP TABLE "{t}"')
                    conn.execute(V3_SIGNALS_DDL)
                    conn.execute(V3_CONTRACTS_DDL)
                    conn.execute(V3_RISK_STATE_DDL)
                    now = utc_iso()
                    binding = {"account_id": account_id, "account_mode": account_mode, "bound_utc": now, "reason": "first_use", "schema_version": 3}
                    conn.execute(
                        "INSERT INTO sota_risk_state (key, value_json, updated_utc) VALUES ('queue_account_binding', ?, ?)",
                        (json.dumps(binding, sort_keys=True), now),
                    )
                    conn.execute("PRAGMA user_version = 3")
                    conn.execute("COMMIT")
                except Exception:
                    conn.execute("ROLLBACK")
                    raise
            finally:
                conn.close()
            fsync_dir(queue_db.parent)
            return {"status": "initialized_from_empty_v0", "queue_db": str(queue_db), "account_id": account_id}

        conn = sqlite3.connect(str(queue_db), timeout=5.0, isolation_level=None)
        conn.row_factory = sqlite3.Row
        try:
            _configure_writable_pragmas_outside_txn(conn)
            create_v3_queue_schema_atomic(conn, account_id=account_id, account_mode=account_mode, reason="first_use")
        finally:
            conn.close()
        fsync_dir(queue_db.parent)
        return {"status": "initialized_v3", "queue_db": str(queue_db), "account_id": account_id}


def detect_predecessor_shape(conn: sqlite3.Connection) -> str:
    """Classify one of the 6 source-derived user_version==0 predecessor shapes (spec §3.4)."""
    uv = int(conn.execute("PRAGMA user_version").fetchone()[0])
    if uv != 0:
        raise ExecutorError(f"predecessor must have user_version=0, got {uv}")
    tables = _table_names(conn)
    if not {"sota_signals", "sota_contracts"}.issubset(tables):
        raise ExecutorError(f"unrecognized predecessor tables: {sorted(tables)}")
    # BLOCKER #4: reject unexpected extra tables beyond known legacy queue schema!
    extra_tables = tables - {"sota_signals", "sota_contracts", "sota_risk_state", "sqlite_sequence"}
    if extra_tables:
        raise ExecutorError(f"unexpected extra tables in predecessor DB: {sorted(extra_tables)}")
    sig_sql = str(conn.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='sota_signals'").fetchone()[0] or "")
    con_cols = {str(r["name"]) for r in conn.execute("PRAGMA table_info(sota_contracts)").fetchall()}
    has_neutral = "'NEUTRAL'" in sig_sql
    has_sc = "signal_close_utc" in con_cols
    has_fam = "contract_family" in con_cols
    has_sack = "sell_ack_cents" in con_cols
    has_am = "account_mode" in con_cols
    has_bcid = "book_content_id" in con_cols
    if not has_neutral and not has_sc and not has_fam:
        return "shape1_earliest_v0"
    if not has_neutral and has_sc and not has_fam:
        return "shape2_signal_close_only"
    if has_neutral and has_fam and not has_sack:
        return "shape3_neutral_family"
    if has_neutral and has_fam and has_sack and not has_am:
        return "shape4_sell_ack_no_mode"
    if has_neutral and has_fam and has_sack and has_am and not has_bcid:
        return "shape5_cdd6475_no_bcid"
    if has_neutral and has_fam and has_sack and has_am and has_bcid:
        return "shape6_adf4ee2_latest_v0"
    raise ExecutorError(f"unrecognized predecessor column combination: {sorted(con_cols)}")


def _copy_sqlite_family_readonly(src_db: Path, dst_dir: Path) -> Path:
    """Copy canonical sqlite + optional -wal/-shm read-only into a private dir (0 bytes written to canonical)."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst_db = dst_dir / src_db.name
    for suffix in ("", "-wal", "-shm"):
        sp = Path(f"{src_db}{suffix}")
        if sp.exists():
            if sp.is_symlink() or not sp.is_file():
                raise ExecutorError(f"non-regular sqlite family file refused: {sp}")
            dp = Path(f"{dst_db}{suffix}")
            dp.write_bytes(sp.read_bytes())
    return dst_db


def prepare_coherent_predecessor_backup(
    canonical_db: Path,
    quarantine_dir: Path,
    *,
    durable_ancestor: Path | None = None,
) -> dict[str, Any]:
    """Step 1: Take atomic coherent SQLite online backup BEFORE export without mutating canonical files."""
    if not canonical_db.is_file() or canonical_db.is_symlink():
        raise ExecutorError(f"missing or symlinked canonical predecessor DB: {canonical_db}")
    qdir = Path(quarantine_dir)
    qdir.mkdir(parents=True, exist_ok=True)
    anc_stop = durable_ancestor if durable_ancestor is not None else Path(os.path.commonpath([str(qdir.resolve()), str(canonical_db.parent.resolve())]))
    backup_path = qdir / "predecessor_coherent_backup.sqlite"
    if backup_path.exists():
        # Verify existing backup integrity and return deterministically
        ro = open_sota_queue_readonly(backup_path)
        try:
            ic = str(ro.execute("PRAGMA integrity_check").fetchone()[0]).lower()
            if ic != "ok":
                raise ExecutorError(f"existing backup failed integrity_check: {ic}")
        finally:
            ro.close()
        raw = backup_path.read_bytes()
        fsync_file(backup_path)
        fsync_ancestry(qdir, stop_at=anc_stop)
        return {"backup_path": str(backup_path), "backup_sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}

    tmp_bak = qdir / ".predecessor_coherent_backup.sqlite.tmp"
    if tmp_bak.exists():
        tmp_bak.unlink()
    # Read canonical predecessor strictly read-only into a private scratch dir first so SQLite .backup()
    # replays uncheckpointed -wal frames coherently with ZERO writes/mutations to canonical files!
    with tempfile.TemporaryDirectory(prefix=".bak_stage_", dir=qdir) as stage_td:
        staged_src = _copy_sqlite_family_readonly(canonical_db, Path(stage_td))
        src_conn = sqlite3.connect(str(staged_src), timeout=5.0, isolation_level=None)
        dst_conn = sqlite3.connect(str(tmp_bak), timeout=5.0, isolation_level=None)
        try:
            src_conn.backup(dst_conn)
            ic = str(dst_conn.execute("PRAGMA integrity_check").fetchone()[0]).lower()
            if ic != "ok":
                raise ExecutorError(f"coherent backup failed integrity_check: {ic}")
            dst_conn.close()
            dst_conn = None  # type: ignore[assignment]
        finally:
            src_conn.close()
            if dst_conn is not None:
                dst_conn.close()
    fsync_file(tmp_bak)
    rename_noreplace(tmp_bak, backup_path)
    fsync_ancestry(qdir, stop_at=anc_stop)
    raw = backup_path.read_bytes()
    return {"backup_path": str(backup_path), "backup_sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}


def _read_predecessor_snapshot_rows(conn: sqlite3.Connection) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]]]:
    shape = detect_predecessor_shape(conn)
    sigs = [dict(r) for r in conn.execute("SELECT * FROM sota_signals ORDER BY signal_id ASC").fetchall()]
    cons = [dict(r) for r in conn.execute("SELECT * FROM sota_contracts ORDER BY contract_id ASC").fetchall()]
    # Refuse undrained states and legacy contract 'NEUTRAL'
    for s in sigs:
        if str(s.get("status") or "") in ("buy_intent", "buy_unknown"):
            raise ExecutorError(f"undrained signal {s.get('signal_id')!r} with status={s.get('status')!r}")
    for c in cons:
        if str(c.get("status") or "") in ("open", "unknown"):
            raise ExecutorError(f"undrained contract {c.get('contract_id')!r} with status={c.get('status')!r}")
        if str(c.get("side") or "").upper() == "NEUTRAL":
            raise ExecutorError(f"contract {c.get('contract_id')!r} has disallowed side='NEUTRAL'")
    return shape, sigs, cons


def export_legacy_ownership_preimage(quarantine_dir: Path) -> dict[str, Any]:
    """Step 2: Read strictly read-only prepared coherent backup (0 bytes written to source)."""
    backup_path = Path(quarantine_dir) / "predecessor_coherent_backup.sqlite"
    if not backup_path.is_file() or backup_path.is_symlink():
        raise ExecutorError(f"prepared coherent backup missing: {backup_path}")
    b_sha = hashlib.sha256(backup_path.read_bytes()).hexdigest()
    ro = open_sota_queue_readonly(backup_path)
    try:
        shape, _, cons = _read_predecessor_snapshot_rows(ro)
    finally:
        ro.close()
    contracts_map = [
        {
            "contract_id": str(c["contract_id"]),
            "signal_id": str(c["signal_id"]),
            "book_id": str(c["book_id"]),
            "pair": str(c["pair"]).upper(),
            "horizon_minutes": int(c["horizon_minutes"]),
            "side": str(c["side"]).upper(),
            "status": str(c["status"]),
            "buy_price_cents": int(c["buy_price_cents"]),
            "sell_price_cents": int(c["sell_price_cents"]),
            "profit_cents": int(c["profit_cents"]),
        }
        for c in cons
    ]
    return {"backup_sha256": b_sha, "shape": shape, "contracts": contracts_map}


def _file_fingerprint(p: Path) -> dict[str, Any] | None:
    if not p.exists():
        return None
    if p.is_symlink() or not p.is_file():
        raise ExecutorError(f"non-regular predecessor file refused: {p}")
    raw = p.read_bytes()
    return {"size": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def _canonical_committed_signature(db_path: Path) -> str:
    """Deterministic digest of full sqlite_master DDL + all committed rows from a predecessor DB (read-only snapshot)."""
    with tempfile.TemporaryDirectory(prefix="sota_sig_") as td:
        staged = _copy_sqlite_family_readonly(db_path, Path(td))
        conn = sqlite3.connect(str(staged), timeout=5.0, isolation_level=None)
        conn.row_factory = sqlite3.Row
        try:
            shape, sigs, cons = _read_predecessor_snapshot_rows(conn)
            ddl = [
                (str(r[0]), str(r[1]), str(r[2] or ""))
                for r in conn.execute("SELECT type, name, sql FROM sqlite_master ORDER BY type ASC, name ASC").fetchall()
            ]
            risk_rows = (
                [dict(r) for r in conn.execute("SELECT * FROM sota_risk_state ORDER BY key ASC").fetchall()]
                if "sota_risk_state" in _table_names(conn)
                else []
            )
            payload = {"shape": shape, "ddl": ddl, "signals": sigs, "contracts": cons, "risk_state": risk_rows}
            return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()
        finally:
            conn.close()


_LEGACY_SIGNAL_STATUS_MAP: dict[str, str] = {
    "queued": "queued",
    "claimed": "cancelled",
    "bought": "bought",
    "contract_closed": "bought",
    "terminal_skip": "rejected",
    "buy_blocked_risk": "rejected",
    "signal_expired": "expired",
}


def adopt_legacy_account(
    *,
    canonical_db: Path,
    quarantine_dir: Path,
    staged_v3_dir: Path,
    ownership_proof_path: Path,
    receipt_path: Path,
    account_id: str,
    account_mode: str = "demo",
    pw_home: Path | None = None,
    durable_ancestor: Path | None = None,
    crash_hook: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Steps 3 & 4: Intent-first crash-safe 6-shape predecessor adoption (spec §3.4 + B5 + B6 + D3 + B4)."""
    if account_mode != "demo" or not is_valid_demo_account_id(account_id):
        raise ExecutorError("Phase 1 adoption requires a concrete demo account")
    # B5: Receipt & credential isolation verified FIRST!
    validate_credential_isolation_receipt(
        receipt_path, expected_account_id=account_id, expected_account_mode=account_mode, pw_home=pw_home
    )

    qdir = Path(quarantine_dir)
    sdir = Path(staged_v3_dir)
    backup_path = qdir / "predecessor_coherent_backup.sqlite"
    if not backup_path.is_file() or backup_path.is_symlink():
        raise ExecutorError(f"missing prepared coherent backup: {backup_path}")
    backup_sha = hashlib.sha256(backup_path.read_bytes()).hexdigest()

    # Authenticate venue ownership proof (backup_sha256 + 1-to-1 contract_ownership_records bijection)
    proof_p = Path(ownership_proof_path)
    if not proof_p.is_file() or proof_p.is_symlink():
        raise ExecutorError(f"missing ownership proof JSON: {proof_p}")
    proof = strict_json_loads_wire(proof_p.read_bytes())
    if proof.get("backup_sha256") != backup_sha or proof.get("account_id") != account_id:
        raise ExecutorError("ownership proof backup_sha256 or account_id mismatch")
    proof_recs = proof.get("contract_ownership_records")
    if not isinstance(proof_recs, list):
        raise ExecutorError("ownership proof missing contract_ownership_records list")

    ro_bak = open_sota_queue_readonly(backup_path)
    try:
        shape, bak_sigs, bak_cons = _read_predecessor_snapshot_rows(ro_bak)
    finally:
        ro_bak.close()

    proof_cids = [str(r.get("contract_id") or "") for r in proof_recs]
    if len(set(proof_cids)) != len(proof_cids):
        raise ExecutorError("duplicate contract_id in ownership proof")
    bak_cids = [str(c["contract_id"]) for c in bak_cons]
    if set(proof_cids) != set(bak_cids):
        raise ExecutorError(f"ownership proof bijection failed (proof={len(proof_cids)} vs backup={len(bak_cids)})")
    for rec in proof_recs:
        if rec.get("account_id") != account_id or rec.get("verified_against_venue") is not True:
            raise ExecutorError(f"unverified ownership proof record: {rec!r}")

    locks = canonical_sota_lock_paths(canonical_db, pw_home=pw_home)
    intent_path = sdir / "install_intent.json"
    staged_db_path = sdir / "staged_v3.sqlite"
    wal_p = Path(f"{canonical_db}-wal")
    shm_p = Path(f"{canonical_db}-shm")

    with acquire_ordered_locks([(locks["acct"], True, True), (locks["queue"], True, True)]):
        with acquire_sota_flock(locks["handoff"], exclusive=True, nonblocking=True):
            pass

        def _finish_install_from_intent(intent: dict[str, Any]) -> dict[str, Any]:
            if intent.get("backup_sha256") != backup_sha or intent.get("staged_v3_user_version") != 3:
                raise ExecutorError("install_intent.json backup_sha256 or user_version mismatch")
            want_v3_sha = str(intent["staged_v3_sha256"])

            # Step 4(a) (D3): Inspect canonical main file FIRST — recognize exact installed SINGLE-FILE v3
            # ONLY when canonical matches staged_v3_sha256 AND user_version==3 AND BOTH -wal/-shm are absent!
            if canonical_db.exists():
                if canonical_db.is_symlink() or not canonical_db.is_file():
                    raise ExecutorError("canonical DB path is not a regular file; holding without unlinking")
                cur_sha = hashlib.sha256(canonical_db.read_bytes()).hexdigest()
                if cur_sha == want_v3_sha:
                    # Verify user_version == 3 read-only
                    ro_c = open_sota_queue_readonly(canonical_db)
                    try:
                        uv_c = int(ro_c.execute("PRAGMA user_version").fetchone()[0])
                    finally:
                        ro_c.close()
                    if uv_c != 3:
                        raise ExecutorError(f"canonical file matches sha but user_version={uv_c} != 3; holding")
                    # D3: Hold on ANY unexpected sidecar without unlinking the v3!
                    if wal_p.exists() or shm_p.exists():
                        raise ExecutorError(
                            "unexpected canonical -wal/-shm sidecar coexists with installed v3; holding without unlinking v3"
                        )
                    # D3: Re-fsync BOTH rename directories before reporting completion!
                    fsync_dir(sdir)
                    fsync_dir(canonical_db.parent)
                    return {
                        "status": "adopted_installed_v3_verified",
                        "queue_db": str(canonical_db),
                        "staged_v3_sha256": want_v3_sha,
                        "shape": shape,
                    }

            # Step 4(b) (B4 + D3 + BLOCKER #4): Validate same-filesystem rename feasibility BEFORE any canonical unlink!
            if not sdir.is_dir() or os.stat(sdir).st_dev != os.stat(canonical_db.parent).st_dev:
                raise ExecutorError(
                    f"staged_v3_dir ({sdir}) is not on the same filesystem as canonical queue ({canonical_db.parent}); refusing EXDEV before unlinking"
                )
            anc_q = durable_ancestor if durable_ancestor is not None else Path(os.path.commonpath([str(qdir.resolve()), str(canonical_db.parent.resolve())]))
            anc_s = durable_ancestor if durable_ancestor is not None else Path(os.path.commonpath([str(sdir.resolve()), str(canonical_db.parent.resolve())]))
            fsync_file(backup_path)
            fsync_ancestry(qdir, stop_at=anc_q)
            fsync_file(intent_path)
            fsync_ancestry(sdir, stop_at=anc_s)

            # Authenticate finalized staged v3 file before touching ANY canonical file
            if not staged_db_path.is_file() or staged_db_path.is_symlink():
                raise ExecutorError(f"staged v3 file missing during recovery: {staged_db_path}")
            if Path(f"{staged_db_path}-wal").exists() or Path(f"{staged_db_path}-shm").exists():
                raise ExecutorError("staged v3 file is not single-file (sidecars present)")
            if hashlib.sha256(staged_db_path.read_bytes()).hexdigest() != want_v3_sha:
                raise ExecutorError("staged v3 file digest mismatch vs install_intent.json; holding")

            # Authenticate EVERY remaining canonical predecessor file against install_intent.predecessor_files!
            pred_map = intent["predecessor_files"]
            targets = [("sqlite", canonical_db), ("wal", wal_p), ("shm", shm_p)]
            to_unlink: list[Path] = []
            for key, path_t in targets:
                if not path_t.exists():
                    continue
                exp_fp = pred_map.get(key)
                if exp_fp is None:
                    raise ExecutorError(f"unexpected canonical file {path_t} absent from install_intent; holding")
                got_fp = _file_fingerprint(path_t)
                if got_fp != exp_fp:
                    raise ExecutorError(f"unknown/modified canonical file {path_t} != install_intent; holding without unlinking")
                to_unlink.append(path_t)

            # Remove ONLY authenticated remaining predecessor files
            for idx_u, path_u in enumerate(to_unlink):
                path_u.unlink()
                if crash_hook is not None and idx_u == 0:
                    crash_hook("mid_canonical_removal")
            for _, path_t in targets:
                if path_t.exists():
                    raise ExecutorError(f"canonical predecessor name still exists after unlink: {path_t}")

            # D3: Fsync deriv_data/runtime/ to make predecessor removals durable BEFORE installing v3!
            fsync_dir(canonical_db.parent)
            if crash_hook is not None:
                crash_hook("after_sync_predecessor_removals")

            # Install finalized single-file v3 via atomic renameat2(RENAME_NOREPLACE)
            rename_noreplace(staged_db_path, canonical_db)
            if crash_hook is not None:
                crash_hook("after_rename_before_dir_fsync")
            fsync_dir(sdir)
            fsync_dir(canonical_db.parent)
            return {
                "status": "adopted_v3_installed",
                "queue_db": str(canonical_db),
                "staged_v3_sha256": want_v3_sha,
                "shape": shape,
            }

        # B5: FIRST check whether install_intent.json already exists — dispatch recovery BEFORE fresh predecessor check!
        if intent_path.is_file():
            existing_intent = strict_json_loads_wire(intent_path.read_bytes())
            return _finish_install_from_intent(existing_intent)

        # Fresh adoption path: compare canonical committed schema & rows against prepared backup
        if not canonical_db.is_file():
            raise ExecutorError(f"fresh adoption requires existing canonical predecessor DB: {canonical_db}")
        if _canonical_committed_signature(canonical_db) != _canonical_committed_signature(backup_path):
            raise ExecutorError("canonical predecessor drifted from prepared coherent backup; refusing adoption")

        # Stage new v3 DB inside same-filesystem staged_v3_dir (BLOCKER #4: check st_dev before staging!)
        sdir.mkdir(parents=True, exist_ok=True)
        if os.stat(sdir).st_dev != os.stat(canonical_db.parent).st_dev:
            raise ExecutorError(
                f"staged_v3_dir ({sdir}) must reside on the same filesystem as canonical queue ({canonical_db.parent})"
            )
        if staged_db_path.exists():
            staged_db_path.unlink()
        for sc in (Path(f"{staged_db_path}-wal"), Path(f"{staged_db_path}-shm")):
            if sc.exists():
                sc.unlink()

        sig_by_id = {str(s["signal_id"]): s for s in bak_sigs}
        now = utc_iso()
        s_conn = sqlite3.connect(str(staged_db_path), timeout=5.0, isolation_level=None)
        s_conn.row_factory = sqlite3.Row
        try:
            _configure_writable_pragmas_outside_txn(s_conn)
            s_conn.execute("BEGIN IMMEDIATE")
            try:
                s_conn.execute(V3_SIGNALS_DDL)
                s_conn.execute(V3_CONTRACTS_DDL)
                s_conn.execute(V3_RISK_STATE_DDL)
                binding = {
                    "account_id": account_id,
                    "account_mode": account_mode,
                    "bound_utc": now,
                    "reason": f"adopted_{shape}",
                    "backup_sha256": backup_sha,
                    "schema_version": 3,
                }
                s_conn.execute(
                    "INSERT INTO sota_risk_state (key, value_json, updated_utc) VALUES ('queue_account_binding', ?, ?)",
                    (json.dumps(binding, sort_keys=True), now),
                )
                for s in bak_sigs:
                    old_st = str(s["status"])
                    if old_st not in _LEGACY_SIGNAL_STATUS_MAP:
                        raise ExecutorError(f"unmapped legacy signal status {old_st!r}")
                    new_st = _LEGACY_SIGNAL_STATUS_MAP[old_st]
                    s_conn.execute(
                        """
                        INSERT INTO sota_signals (
                            signal_id, book_id, pair, horizon_minutes, side, lane, offset_id,
                            signal_close_utc, effective_floor, calibrated_win_prob, payload_json,
                            status, proposal_id, contract_id, reason, created_utc, updated_utc
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, ?, ?, ?)
                        """,
                        (
                            str(s["signal_id"]),
                            str(s["book_id"]),
                            str(s["pair"]).upper(),
                            int(s["horizon_minutes"]),
                            str(s["side"]).upper(),
                            str(s.get("lane") or "wall"),
                            int(s.get("offset_id") or 0),
                            str(s["signal_close_utc"]),
                            float(s["effective_floor"]),
                            float(s["calibrated_win_prob"]) if s.get("calibrated_win_prob") is not None else None,
                            str(s["payload_json"]),
                            new_st,
                            s.get("reason"),
                            str(s["created_utc"]),
                            str(s["updated_utc"]),
                        ),
                    )
                for c in bak_cons:
                    st_c = str(c["status"])
                    if st_c not in TERMINAL_CONTRACT_STATUSES:
                        raise ExecutorError(f"non-terminal legacy contract status {st_c!r}")
                    fam = str(c.get("contract_family") or "risefall")
                    # B6: For proven migrated multiplier history ONLY, map no-fixed-payout sentinel payout_cents=0 -> NULL!
                    raw_pay = c.get("payout_cents")
                    if fam == "multiplier" and raw_pay is not None and int(raw_pay) == 0:
                        pay_c: int | None = None
                    elif raw_pay is None:
                        pay_c = None
                    else:
                        pay_c = int(raw_pay)
                        if pay_c <= 0:
                            raise ExecutorError(f"non-multiplier contract {c['contract_id']!r} has invalid payout_cents={pay_c}")

                    # B6: Backfill absent or NULL legacy contract signal_close_utc from linked legacy signal!
                    linked_sig = sig_by_id.get(str(c["signal_id"]))
                    raw_c_sc = c.get("signal_close_utc")
                    sig_sc = str(linked_sig["signal_close_utc"]) if linked_sig is not None else None
                    if raw_c_sc is None or str(raw_c_sc).strip() == "":
                        if not sig_sc:
                            raise ExecutorError(f"contract {c['contract_id']!r} missing signal_close_utc and linked signal clock")
                        sc_utc = sig_sc
                    else:
                        sc_utc = str(raw_c_sc)
                        if sig_sc and sig_sc != sc_utc:
                            raise ExecutorError(f"contract {c['contract_id']!r} signal_close_utc conflicts with linked signal")

                    bcid = str(c["book_content_id"]).strip() if c.get("book_content_id") else None
                    if bcid == "":
                        bcid = None
                    s_conn.execute(
                        """
                        INSERT INTO sota_contracts (
                            contract_id, signal_id, book_id, book_content_id, account_id, account_mode,
                            pair, horizon_minutes, side, contract_family, is_migrated_legacy,
                            cost_unknown, buy_price_cents, payout_cents, sell_price_cents, profit_cents,
                            stake_authoritative, venue_start_epoch, venue_end_epoch, venue_terminal_status,
                            venue_underlying_symbol, venue_contract_type, signal_close_utc, status,
                            proposal_id, expected_expiry_utc, raw_hash, sell_ack_cents,
                            opened_utc, closed_utc, updated_utc
                        ) VALUES (
                            ?, ?, ?, ?, ?, 'demo', ?, ?, ?, ?, 1,
                            0, ?, ?, ?, ?, 0, NULL, NULL, NULL,
                            NULL, NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?
                        )
                        """,
                        (
                            str(c["contract_id"]),
                            str(c["signal_id"]),
                            str(c["book_id"]),
                            bcid,
                            account_id,
                            str(c["pair"]).upper(),
                            int(c["horizon_minutes"]),
                            str(c["side"]).upper(),
                            fam,
                            int(c["buy_price_cents"]),
                            pay_c,
                            int(c["sell_price_cents"]),
                            int(c["profit_cents"]),
                            sc_utc,
                            st_c,
                            str(c.get("proposal_id") or ""),
                            c.get("expected_expiry_utc"),
                            c.get("raw_hash"),
                            c.get("sell_ack_cents"),
                            str(c["opened_utc"]),
                            c.get("closed_utc"),
                            now,
                        ),
                    )
                s_conn.execute("PRAGMA user_version = 3")
                s_conn.execute("COMMIT")
            except Exception:
                s_conn.execute("ROLLBACK")
                raise

            # Finalize staged v3 into ONE single file (checkpoint TRUNCATE, DELETE journal mode, integrity + FK checks)
            s_conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            s_conn.execute("PRAGMA journal_mode=DELETE")
            ic = str(s_conn.execute("PRAGMA integrity_check").fetchone()[0]).lower()
            if ic != "ok":
                raise ExecutorError(f"staged v3 failed integrity_check: {ic}")
            fk_errs = s_conn.execute("PRAGMA foreign_key_check").fetchall()
            if fk_errs:
                raise ExecutorError(f"staged v3 failed foreign_key_check: {fk_errs[:5]}")
        finally:
            s_conn.close()

        if Path(f"{staged_db_path}-wal").exists() or Path(f"{staged_db_path}-shm").exists():
            raise ExecutorError("staged v3 still has -wal/-shm sidecars after finalization")
        fsync_file(staged_db_path)
        staged_sha = hashlib.sha256(staged_db_path.read_bytes()).hexdigest()

        # Persist dual-digest install_intent.json BEFORE unlinking any canonical name!
        intent = {
            "schema": "sota-v3-install-intent/v1",
            "backup_sha256": backup_sha,
            "predecessor_files": {
                "sqlite": _file_fingerprint(canonical_db),
                "wal": _file_fingerprint(wal_p),
                "shm": _file_fingerprint(shm_p),
            },
            "staged_v3_sha256": staged_sha,
            "staged_v3_user_version": 3,
        }
        atomic_write_file_bytes(intent_path, (json.dumps(intent, indent=2, sort_keys=True) + "\n").encode("utf-8"), sync_parent=True)
        if crash_hook is not None:
            crash_hook("after_intent_before_unlink")
        return _finish_install_from_intent(intent)


# ---------------------------------------------------------------------------
# Backwards-Compatible Legacy Connect Helper (Preserves Existing Tests)
# ---------------------------------------------------------------------------


def _migrate_queue_schema_if_needed(conn: sqlite3.Connection) -> None:
    uv = int(conn.execute("PRAGMA user_version").fetchone()[0])
    if uv == 3:
        return
    conn.execute("PRAGMA foreign_keys=OFF")
    conn.execute("BEGIN IMMEDIATE")
    try:
        sig_sql_row = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='sota_signals'"
        ).fetchone()
        con_sql_row = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='sota_contracts'"
        ).fetchone()
        need_sig = sig_sql_row is not None and "'NEUTRAL'" not in str(sig_sql_row[0] or "")
        con_cols = {str(r["name"]) for r in conn.execute("PRAGMA table_info(sota_contracts)").fetchall()} if con_sql_row else set()
        need_con = con_sql_row is not None and (
            "'NEUTRAL'" not in str(con_sql_row[0] or "")
            or "contract_family" not in con_cols
            or "signal_close_utc" not in con_cols
            or "sell_ack_cents" not in con_cols
            or "account_mode" not in con_cols
            or "book_content_id" not in con_cols
        )
        if not (need_sig or need_con):
            conn.execute("COMMIT")
            return
        if need_sig:
            conn.execute(
                """
                CREATE TABLE sota_signals_v2 (
                    signal_id TEXT PRIMARY KEY,
                    book_id TEXT NOT NULL,
                    pair TEXT NOT NULL,
                    horizon_minutes INTEGER NOT NULL,
                    side TEXT NOT NULL CHECK (side IN ('UP','DOWN','NEUTRAL')),
                    lane TEXT NOT NULL DEFAULT 'wall',
                    offset_id INTEGER NOT NULL DEFAULT 0,
                    signal_close_utc TEXT NOT NULL,
                    effective_floor REAL NOT NULL,
                    calibrated_win_prob REAL,
                    payload_json TEXT NOT NULL,
                    status TEXT NOT NULL CHECK (status IN (
                        'queued','claimed','buy_intent','bought','contract_closed',
                        'terminal_skip','signal_expired','buy_blocked_risk','buy_unknown',
                        'buy_refused_pre_send','rejected','expired','cancelled'
                    )),
                    reason TEXT,
                    created_utc TEXT NOT NULL,
                    updated_utc TEXT NOT NULL,
                    UNIQUE(book_id, pair, horizon_minutes, side, lane, offset_id, signal_close_utc)
                )
                """
            )
            conn.execute(
                """
                INSERT INTO sota_signals_v2 (
                    signal_id, book_id, pair, horizon_minutes, side, lane, offset_id,
                    signal_close_utc, effective_floor, calibrated_win_prob, payload_json,
                    status, reason, created_utc, updated_utc
                )
                SELECT signal_id, book_id, pair, horizon_minutes, side, lane, offset_id,
                       signal_close_utc, effective_floor, calibrated_win_prob, payload_json,
                       status, reason, created_utc, updated_utc
                FROM sota_signals
                """
            )
            conn.execute("DROP TABLE sota_signals")
            conn.execute("ALTER TABLE sota_signals_v2 RENAME TO sota_signals")
        if need_con:
            sig_close_expr = "signal_close_utc" if "signal_close_utc" in con_cols else "NULL"
            fam_expr = "contract_family" if "contract_family" in con_cols else "'risefall'"
            sack_expr = "sell_ack_cents" if "sell_ack_cents" in con_cols else "NULL"
            am_expr = "account_mode" if "account_mode" in con_cols else "'demo'"
            bcid_expr = "book_content_id" if "book_content_id" in con_cols else "NULL"
            conn.execute(
                """
                CREATE TABLE sota_contracts_v2 (
                    contract_id TEXT PRIMARY KEY,
                    signal_id TEXT NOT NULL UNIQUE REFERENCES sota_signals(signal_id),
                    book_id TEXT NOT NULL,
                    pair TEXT NOT NULL,
                    horizon_minutes INTEGER NOT NULL,
                    side TEXT NOT NULL CHECK (side IN ('UP','DOWN','NEUTRAL')),
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
                    signal_close_utc TEXT,
                    contract_family TEXT NOT NULL DEFAULT 'risefall',
                    sell_ack_cents INTEGER DEFAULT NULL,
                    account_mode TEXT NOT NULL DEFAULT 'demo',
                    book_content_id TEXT DEFAULT NULL
                )
                """
            )
            conn.execute(
                f"""
                INSERT INTO sota_contracts_v2 (
                    contract_id, signal_id, book_id, pair, horizon_minutes, side,
                    proposal_id, buy_price_cents, payout_cents, sell_price_cents, profit_cents,
                    status, opened_utc, expected_expiry_utc, closed_utc, raw_hash,
                    signal_close_utc, contract_family, sell_ack_cents, account_mode, book_content_id
                )
                SELECT contract_id, signal_id, book_id, pair, horizon_minutes, side,
                       proposal_id, buy_price_cents, payout_cents, sell_price_cents, profit_cents,
                       status, opened_utc, expected_expiry_utc, closed_utc, raw_hash,
                       {sig_close_expr}, {fam_expr}, {sack_expr}, {am_expr}, {bcid_expr}
                FROM sota_contracts
                """
            )
            conn.execute("DROP TABLE sota_contracts")
            conn.execute("ALTER TABLE sota_contracts_v2 RENAME TO sota_contracts")
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise
    finally:
        conn.execute("PRAGMA foreign_keys=ON")


def connect_sota_queue(db_path: Path) -> sqlite3.Connection:
    """Connect to existing v3 or legacy test queue (PRAGMAs configured outside BEGIN, zero executescript)."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), timeout=5.0, isolation_level=None)
    conn.row_factory = sqlite3.Row
    _configure_writable_pragmas_outside_txn(conn)
    uv = int(conn.execute("PRAGMA user_version").fetchone()[0])
    if uv == 3:
        return conn
    _migrate_queue_schema_if_needed(conn)
    conn.execute("BEGIN IMMEDIATE")
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sota_signals (
                signal_id TEXT PRIMARY KEY,
                book_id TEXT NOT NULL,
                pair TEXT NOT NULL,
                horizon_minutes INTEGER NOT NULL,
                side TEXT NOT NULL CHECK (side IN ('UP','DOWN','NEUTRAL')),
                lane TEXT NOT NULL DEFAULT 'wall',
                offset_id INTEGER NOT NULL DEFAULT 0,
                signal_close_utc TEXT NOT NULL,
                effective_floor REAL NOT NULL,
                calibrated_win_prob REAL,
                payload_json TEXT NOT NULL,
                status TEXT NOT NULL CHECK (status IN (
                    'queued','claimed','buy_intent','bought','contract_closed',
                    'terminal_skip','signal_expired','buy_blocked_risk','buy_unknown',
                    'buy_refused_pre_send','rejected','expired','cancelled'
                )),
                reason TEXT,
                created_utc TEXT NOT NULL,
                updated_utc TEXT NOT NULL,
                UNIQUE(book_id, pair, horizon_minutes, side, lane, offset_id, signal_close_utc)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sota_contracts (
                contract_id TEXT PRIMARY KEY,
                signal_id TEXT NOT NULL UNIQUE REFERENCES sota_signals(signal_id),
                book_id TEXT NOT NULL,
                pair TEXT NOT NULL,
                horizon_minutes INTEGER NOT NULL,
                side TEXT NOT NULL CHECK (side IN ('UP','DOWN','NEUTRAL')),
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
                signal_close_utc TEXT,
                contract_family TEXT NOT NULL DEFAULT 'risefall',
                sell_ack_cents INTEGER DEFAULT NULL,
                account_mode TEXT NOT NULL DEFAULT 'demo',
                book_content_id TEXT DEFAULT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sota_risk_state (
                key TEXT PRIMARY KEY,
                value_json TEXT NOT NULL,
                updated_utc TEXT NOT NULL
            )
            """
        )
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise
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
        exists = conn.execute("SELECT 1 FROM sota_signals WHERE signal_id=?", (sid,)).fetchone()
        if exists is not None:
            conn.execute("COMMIT")
            return False, sid
        cur = conn.execute(
            """
            INSERT INTO sota_signals (
                signal_id, book_id, pair, horizon_minutes, side, lane, offset_id,
                signal_close_utc, effective_floor, calibrated_win_prob, payload_json,
                status, created_utc, updated_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'queued', ?, ?)
            ON CONFLICT(book_id, pair, horizon_minutes, side, lane, offset_id, signal_close_utc) DO NOTHING
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


# ---------------------------------------------------------------------------
# Safety Latch, Cohort Anchor, Stale Lock & Risk Breakers (spec §§3.4-3.6)
# ---------------------------------------------------------------------------


def is_safety_latch_set(conn: sqlite3.Connection) -> dict[str, Any] | None:
    row = conn.execute("SELECT value_json FROM sota_risk_state WHERE key='phase1_safety_latch'").fetchone()
    if row is None:
        return None
    try:
        return json.loads(str(row[0]))
    except Exception:
        return {"raw": str(row[0])}


def commit_safety_latch(conn: sqlite3.Connection, reason: str, **detail: Any) -> dict[str, Any]:
    now = utc_iso()
    val = {"latched_utc": now, "reason": str(reason), **detail}
    conn.execute("BEGIN IMMEDIATE")
    try:
        conn.execute(
            """
            INSERT INTO sota_risk_state (key, value_json, updated_utc)
            VALUES ('phase1_safety_latch', ?, ?)
            ON CONFLICT(key) DO UPDATE SET value_json=excluded.value_json, updated_utc=excluded.updated_utc
            """,
            (json.dumps(val, sort_keys=True), now),
        )
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise
    return val


def request_stop_latch(
    *,
    queue_db: Path,
    account_id: str,
    account_mode: str,
    reason: str,
    log_dir: Path | None = None,
    pw_home: Path | None = None,
) -> dict[str, Any]:
    """--request-stop-latch: works while --run is alive & even if log_dir is unwritable (LOCK_EX sota_handoff)."""
    if account_mode != "demo":
        raise ExecutorError("--request-stop-latch refuses live mode in Phase 1")
    if not str(reason or "").strip():
        raise ExecutorError("--request-stop-latch requires non-empty --reason")
    locks = canonical_sota_lock_paths(queue_db, pw_home=pw_home)
    with acquire_sota_flock(locks["handoff"], exclusive=True, nonblocking=False):
        conn = connect_sota_queue_v3(queue_db, expected_account_id=account_id, expected_account_mode=account_mode)
        try:
            latched = commit_safety_latch(conn, reason.strip())
        finally:
            conn.close()
    # Best-effort KILL marker (must NEVER fail if log_dir is unwritable!)
    if log_dir is not None:
        try:
            Path(log_dir).mkdir(parents=True, exist_ok=True)
            (Path(log_dir) / "KILL").write_text(json.dumps(latched, sort_keys=True) + "\n", encoding="utf-8")
        except OSError:
            pass
    return latched


def register_cohort_anchor(
    *,
    queue_db: Path,
    account_id: str,
    account_mode: str,
    book_id: str,
    cohort_start_utc: str,
    venue_time_epoch: int,
    receipt_path: Path,
    repo_root: Path = LEXICAL_REPO_ROOT,
    pw_home: Path | None = None,
) -> dict[str, Any]:
    """--register-cohort-anchor: binds registered_utc to verified venue time epoch (spec §3.5)."""
    validate_credential_isolation_receipt(
        receipt_path, expected_account_id=account_id, expected_account_mode=account_mode, pw_home=pw_home
    )
    book = load_sota_book(book_id, repo_root=repo_root, allow_demo_canary=(account_mode == "demo"))
    start_dt = _parse_iso_utc(cohort_start_utc)
    if start_dt is None or start_dt.microsecond != 0:
        raise ExecutorError(f"cohort_start_utc must be whole-second RFC3339 UTC, got {cohort_start_utc!r}")
    start_ep = int(start_dt.timestamp())
    reg_ep = int(venue_time_epoch)
    reg_iso = datetime.fromtimestamp(reg_ep, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    start_iso = datetime.fromtimestamp(start_ep, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    anchor_key = f"g6_cohort_anchor:{account_id}:{book.book_id}:{book.content_id}"
    locks = canonical_sota_lock_paths(queue_db, pw_home=pw_home)
    with acquire_ordered_locks([(locks["acct"], True, True), (locks["queue"], True, True)]):
        conn = connect_sota_queue_v3(queue_db, expected_account_id=account_id, expected_account_mode=account_mode)
        try:
            ex_row = conn.execute("SELECT value_json FROM sota_risk_state WHERE key=?", (anchor_key,)).fetchone()
            if ex_row is not None:
                ex = json.loads(str(ex_row[0]))
                if (
                    ex.get("cohort_start_utc") == start_iso
                    and ex.get("required_stake_cents") == 100
                    and ex.get("account_id") == account_id
                    and ex.get("book_id") == book.book_id
                    and ex.get("book_content_id") == book.content_id
                ):
                    return {"status": "idempotent_existing_anchor", "anchor": ex}
                raise ExecutorError(f"conflicting existing cohort anchor for book={book.book_id} content_id={book.content_id}")

            if start_ep <= reg_ep:
                raise ExecutorError(f"cohort_start_utc ({start_iso}) must be strictly > verified venue time ({reg_iso})")
            unresolved = int(
                conn.execute(
                    "SELECT COUNT(*) FROM sota_signals WHERE book_id=? AND status IN ('buy_intent','buy_unknown')",
                    (book.book_id,),
                ).fetchone()[0]
            )
            if unresolved > 0:
                raise ExecutorError(f"cannot register cohort anchor while {unresolved} unresolved signals exist for {book.book_id}")
            ambig = int(
                conn.execute(
                    """
                    SELECT COUNT(*) FROM sota_contracts
                    WHERE book_id=? AND (
                        status IN ('open','unknown')
                        OR (venue_start_epoch IS NOT NULL AND venue_start_epoch >= ?)
                    )
                    """,
                    (book.book_id, start_ep),
                ).fetchone()[0]
            )
            if ambig > 0:
                raise ExecutorError(f"cannot register cohort anchor with {ambig} open/unknown/cohort contracts for {book.book_id}")

            anchor = {
                "account_id": account_id,
                "book_id": book.book_id,
                "book_content_id": book.content_id,
                "pair": book.pair,
                "horizon_minutes": book.horizon_minutes,
                "side": book.allowed_sides[0],
                "contract_family": book.contract_family,
                "cohort_start_utc": start_iso,
                "registered_utc": reg_iso,
                "required_stake_cents": 100,
            }
            conn.execute("BEGIN IMMEDIATE")
            try:
                conn.execute(
                    "INSERT INTO sota_risk_state (key, value_json, updated_utc) VALUES (?, ?, ?)",
                    (anchor_key, json.dumps(anchor, sort_keys=True), reg_iso),
                )
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise
            return {"status": "registered", "anchor": anchor}
        finally:
            conn.close()


def clear_stale_legacy_lock(*, account_id: str, account_mode: str = "demo", lock_root: Path | None = None) -> dict[str, Any]:
    """--clear-stale-legacy-lock: verifies dead PID and unlinks stale legacy ExecutorLock (refuses live PID)."""
    lroot = canonical_lock_root(lock_root)
    lkey = sota_account_lock_key(account_id, account_mode)
    lpath = executor_lock_path(lroot, lkey)
    if not lpath.exists():
        return {"cleared": False, "reason": "not_present", "path": str(lpath)}
    try:
        meta = json.loads(lpath.read_text(encoding="utf-8"))
        pid = int(meta.get("pid", -1))
    except Exception as exc:
        raise ExecutorError(f"cannot parse legacy lockfile {lpath}: {exc}") from exc
    if pid <= 1:
        raise ExecutorError(f"refusing invalid PID {pid} in legacy lockfile {lpath}")
    # Verify PID is dead (os.kill(pid, 0) raises ProcessLookupError)
    try:
        os.kill(pid, 0)
        raise ExecutorError(f"refusing to clear legacy lock held by live PID {pid}: {lpath}")
    except ProcessLookupError:
        pass
    except PermissionError as exc:
        raise ExecutorError(f"refusing to clear legacy lock held by live PID {pid} (EPERM): {lpath}") from exc
    lpath.unlink()
    return {"cleared": True, "dead_pid": pid, "path": str(lpath)}


@dataclass(frozen=True)
class RiskBreakerConfig:
    max_stake_usd: float = 5.0
    max_daily_loss_usd: float = 25.0
    max_drawdown_usd: float = 50.0
    max_open_contracts: int = 4
    max_usd_factor_open: int = 2
    max_jpy_factor_open: int = 2
    max_vol_pair_open: int = 1
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
    venue_time_epoch: int | None = None,
) -> str | None:
    """Return rejection reason if latch, unresolved buy, cooldown, currency-leg cap, or unbounded-int risk cap trips."""
    now = (now_utc or utc_now()).astimezone(timezone.utc)
    if is_safety_latch_set(conn) is not None:
        return "phase1_safety_latch_active"
    if not math.isfinite(stake_usd) or stake_usd <= 0.0:
        return "invalid_stake"
    if stake_usd > cfg.max_stake_usd + 1e-9:
        return "stake_exceeds_max_stake_usd"

    uncertain_sigs = int(
        conn.execute("SELECT COUNT(*) FROM sota_signals WHERE status IN ('buy_intent', 'buy_unknown')").fetchone()[0]
    )
    unknown_contracts = int(
        conn.execute("SELECT COUNT(*) FROM sota_contracts WHERE status = 'unknown'").fetchone()[0]
    )
    if uncertain_sigs > 0 or unknown_contracts > 0:
        return "unresolved_buy_uncertainty_halt"

    open_rows = conn.execute(
        "SELECT book_id, pair, horizon_minutes, side, opened_utc FROM sota_contracts WHERE status = 'open'"
    ).fetchall()
    if cfg.max_open_contracts > 0 and len(open_rows) >= cfg.max_open_contracts:
        return "max_open_contracts_reached"

    # Direct same-generation venue-time cooldown (spec §3.4: venue_epoch > venue_end_epoch) + bar-close nonoverlap
    con_cols = {str(r["name"]) for r in conn.execute("PRAGMA table_info(sota_contracts)").fetchall()}
    if venue_time_epoch is not None and "venue_end_epoch" in con_cols:
        active_ve = conn.execute(
            """
            SELECT MAX(venue_end_epoch) FROM sota_contracts
            WHERE book_id = ? AND venue_end_epoch IS NOT NULL
            """,
            (book_id,),
        ).fetchone()[0]
        if active_ve is not None and int(venue_time_epoch) <= int(active_ve):
            return "venue_clock_cooldown_active"

    min_gap = max(int(horizon_minutes) * 60, int(cfg.same_book_min_gap_seconds))
    if min_gap > 0:
        ref_dt = _parse_iso_utc(signal_close_utc) or now
        recent_rows = conn.execute(
            """
            SELECT c.opened_utc, COALESCE(c.signal_close_utc, s.signal_close_utc) AS sig_close
            FROM sota_contracts c
            LEFT JOIN sota_signals s ON s.signal_id = c.signal_id
            WHERE c.book_id = ? AND c.pair = ? AND c.horizon_minutes = ?
            ORDER BY c.opened_utc DESC LIMIT 20
            """,
            (book_id, pair.upper(), int(horizon_minutes)),
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
    if cfg.max_vol_pair_open > 0:
        for ccy in want_legs:
            if ccy.startswith("VOL_") and leg_counts.get(ccy, 0) >= cfg.max_vol_pair_open:
                return "max_vol_pair_open_reached"

    # Unbounded Python int risk sums (immune to SQLite 64-bit SUM overflow!)
    day_prefix = now.strftime("%Y-%m-%d")
    daily_profit_cents = sum(
        int(r[0])
        for r in conn.execute(
            """
            SELECT profit_cents FROM sota_contracts
            WHERE status IN ('won','lost','sold','expired','cancelled')
              AND closed_utc LIKE ? AND profit_cents IS NOT NULL
            """,
            (f"{day_prefix}%",),
        ).fetchall()
    )
    open_risk_cents = sum(
        int(r[0])
        for r in conn.execute(
            "SELECT buy_price_cents FROM sota_contracts WHERE status = 'open' AND buy_price_cents IS NOT NULL"
        ).fetchall()
    )
    new_stake_cents = int(round(stake_usd * 100))
    worst_daily_loss_cents = -daily_profit_cents + open_risk_cents + new_stake_cents
    if cfg.max_daily_loss_usd > 0 and worst_daily_loss_cents > int(round(cfg.max_daily_loss_usd * 100)):
        return "daily_loss_circuit_breaker"

    settled_profits = [
        int(r[0])
        for r in conn.execute(
            """
            SELECT profit_cents FROM sota_contracts
            WHERE status IN ('won','lost','sold','expired','cancelled') AND profit_cents IS NOT NULL
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
        max_buy_dispatch_delay_s: float = 4.0,
        env: DerivEnv | None = None,
    ):
        self.account_mode = str(account_mode).strip().lower()
        self.stake_usd = float(stake_usd)
        self.db_path = Path(db_path)
        self.log_dir = Path(log_dir)
        try:
            self.log_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass
        self.kill_path = self.log_dir / "KILL"
        # MAJOR #2: Graceful fallback when log_dir is unwritable so latched reconciliation-only startup NEVER fails!
        self.logger: JsonlLogger | None = None
        try:
            self.logger = JsonlLogger(self.log_dir / f"sota_executor_{self.account_mode}")
        except OSError:
            self.logger = None
        self.risk_cfg = risk_cfg
        self.absolute_breakeven_ceiling = float(absolute_breakeven_ceiling)
        self.ev_margin = float(ev_margin)
        self.dry_run_proposal_only = bool(dry_run_proposal_only)
        self.max_signal_age_s = float(max_signal_age_s)
        self.max_buy_dispatch_delay_s = float(max_buy_dispatch_delay_s)
        self.env = env
        self.client: Any = None
        self.transport_kind: str = "options_otp"
        self._authorized_reconnect_epoch: int = -1
        self.last_venue_time_epoch: int | None = None
        self.consecutive_reconcile_errors = 0

    def _safe_log(self, event: str, **fields: Any) -> None:
        if self.logger is None:
            return
        try:
            self.logger.write(event, **fields)
        except OSError:
            pass

    def trip_kill_switch(self, reason: str, **detail: Any) -> None:
        try:
            self.kill_path.write_text(json.dumps({"tripped_utc": utc_iso(), "reason": reason, **detail}) + "\n", encoding="utf-8")
        except OSError:
            pass
        self._safe_log("kill_switch_tripped", reason=reason, **detail)

    async def _call_typed(
        self,
        send_class: str,
        payload: dict[str, Any],
        *,
        timeout: float = 6.0,
        expected_generation: int | None = None,
        send_guard_fn: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        assert self.client is not None
        if isinstance(self.client, SotaDerivAsyncClient):
            if send_class == "account_read":
                return await self.client.request_account_read(payload, timeout=timeout, expected_generation=expected_generation)
            if send_class == "proposal":
                return await self.client.request_proposal(payload, timeout=timeout, expected_generation=expected_generation)
            if send_class == "trade_mutation":
                return await self.client.request_trade_mutation(
                    payload, timeout=timeout, expected_generation=expected_generation, send_guard_fn=send_guard_fn
                )
            if send_class == "identity":
                return await self.client.request_identity(payload, timeout=timeout)
            return await self.client.request_public_read(payload, timeout=timeout)
        return await self.client.request(payload, timeout=timeout)

    async def ensure_authenticated(self) -> None:
        """Verify server-side account loginid, USD currency, virtual/real mode & venue clock across reconnects."""
        if self.client is None or self.env is None:
            return
        if isinstance(self.client, SotaDerivAsyncClient):
            if self.client.is_identity_verified():
                return
            self.last_venue_time_epoch = await self.client.establish_identity(
                account_id=self.env.account_id,
                account_mode=self.account_mode,
                transport_kind=self.transport_kind,
                pat=self.env.pat,
                timeout=10.0,
            )
            return

        reconnects = int(getattr(self.client, "stats", {}).get("reconnects", 0))
        if reconnects == self._authorized_reconnect_epoch:
            return
        if self.transport_kind == "v3_authorize":
            try:
                auth_resp = await self.client.request({"authorize": self.env.pat}, timeout=10.0)
            except Exception as exc:
                safe_msg = str(exc).replace(self.env.pat, "[REDACTED_PAT]") if self.env.pat else str(exc)
                raise ExecutorError(f"v3_authorize request failed: {safe_msg}") from None
            auth_info = auth_resp.get("authorize") or {}
            loginid = str(auth_info.get("loginid") or "").strip()
            ccy = str(auth_info.get("currency") or "").strip().upper()
            is_virt = bool(auth_info.get("is_virtual"))
            if not loginid or loginid != self.env.account_id:
                raise ExecutorError(
                    f"PAT authorized account mismatch: got={redact_account_id(loginid)} != expected={redact_account_id(self.env.account_id)}"
                )
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
                raise ExecutorError(
                    f"OTP socket account mismatch: got={redact_account_id(loginid)} != expected={redact_account_id(self.env.account_id)}"
                )
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

    def _is_v3_conn(self, conn: sqlite3.Connection) -> bool:
        return int(conn.execute("PRAGMA user_version").fetchone()[0]) == 3

    def _reject_status_for(self, conn: sqlite3.Connection, default_legacy: str) -> str:
        if not self._is_v3_conn(conn):
            return default_legacy
        return "expired" if default_legacy == "signal_expired" else "rejected"

    async def process_one_signal(
        self,
        conn: sqlite3.Connection,
        signal_id: str,
        payload: dict[str, Any],
        *,
        now_utc: datetime | None = None,
    ) -> dict[str, Any]:
        mono_start = time.monotonic()
        now = (now_utc or utc_now()).astimezone(timezone.utc)
        is_v3 = self._is_v3_conn(conn)

        def _set_sig_status(st_val: str, rsn: str, ts_dt: datetime, prop_id: str | None = None) -> None:
            if is_v3:
                conn.execute(
                    "UPDATE sota_signals SET status=?, reason=?, proposal_id=COALESCE(?, proposal_id), updated_utc=? WHERE signal_id=?",
                    (st_val, rsn, prop_id, utc_iso(ts_dt), signal_id),
                )
            else:
                conn.execute(
                    "UPDATE sota_signals SET status=?, reason=?, updated_utc=? WHERE signal_id=?",
                    (st_val, rsn, utc_iso(ts_dt), signal_id),
                )

        pre_reason = self.pre_proposal_gate(payload, now_utc=now)
        if pre_reason is not None:
            status = self._reject_status_for(conn, "signal_expired" if pre_reason == "signal_expired" else "terminal_skip")
            _set_sig_status(status, pre_reason, now)
            self._safe_log("signal_rejected_pre_proposal", **{**payload, "signal_id": signal_id, "reason": pre_reason})
            return {"signal_id": signal_id, "status": status, "reason": pre_reason}

        assert self.client is not None
        await self.ensure_authenticated()
        expected_gen = self.client.generation if isinstance(self.client, SotaDerivAsyncClient) else None
        # MAJOR #1: Refresh same-generation venue clock on v3 before cooldown admission so cooldown never goes stale!
        if is_v3 and isinstance(self.client, SotaDerivAsyncClient):
            try:
                self.last_venue_time_epoch = await self.client.fetch_venue_time_epoch(timeout=4.0, expected_generation=expected_gen)
            except Exception as exc:
                st_f = self._reject_status_for(conn, "terminal_skip")
                _set_sig_status(st_f, "venue_time_refresh_failed", now)
                self._safe_log("venue_time_refresh_failed", signal_id=signal_id, error=str(exc)[:240])
                return {"signal_id": signal_id, "status": st_f, "reason": "venue_time_refresh_failed"}

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
            venue_time_epoch=self.last_venue_time_epoch,
        )
        if risk_reason is not None:
            st_r = self._reject_status_for(conn, "buy_blocked_risk")
            _set_sig_status(st_r, risk_reason, now)
            self._safe_log("signal_blocked_risk", **{**payload, "signal_id": signal_id, "reason": risk_reason})
            return {"signal_id": signal_id, "status": st_r, "reason": risk_reason}
        pair = str(payload["pair"]).upper()
        side = str(payload["side"]).upper()
        horizon = int(payload["horizon_minutes"])
        family = str(payload.get("contract_family", "risefall"))
        bct = payload.get("barrier_contract_type")
        try:
            ctype = contract_type_for(side, family=family, barrier_contract_type=bct)
            sym_key = "symbol" if self.transport_kind == "v3_authorize" else "underlying_symbol"
            if family == "multiplier":
                raw_mult = payload.get("multiplier")
                if isinstance(raw_mult, bool) or raw_mult is None or int(raw_mult) < 1:
                    raise ExecutorError(f"invalid multiplier {raw_mult!r}")
                mult = int(raw_mult)
                prop_req: dict[str, Any] = {
                    "proposal": 1,
                    "amount": float(self.stake_usd),
                    "basis": "stake",
                    "contract_type": ctype,
                    "currency": "USD",
                    "multiplier": mult,
                    sym_key: SOTA_DERIV_SYMBOLS[pair],
                }
            elif family == "barrier_1d":
                if horizon != 1440:
                    raise ExecutorError(f"barrier_1d requires horizon_minutes=1440, got {horizon}")
                if payload.get("spot_ref") is None or payload.get("barrier_rel_half_width") is None:
                    raise ExecutorError("barrier_1d requires spot_ref and barrier_rel_half_width")
                hi_str, lo_str = format_barrier_levels(
                    pair, float(payload["spot_ref"]), float(payload["barrier_rel_half_width"]), ctype, side=side
                )
                prop_req = {
                    "proposal": 1,
                    "amount": float(self.stake_usd),
                    "basis": "stake",
                    "contract_type": ctype,
                    "currency": "USD",
                    "duration": 1,
                    "duration_unit": "d",
                    "barrier": hi_str,
                    sym_key: SOTA_DERIV_SYMBOLS[pair],
                }
                if lo_str is not None:
                    prop_req["barrier2"] = lo_str
            else:
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
        except Exception as exc:
            st_f = self._reject_status_for(conn, "terminal_skip")
            _set_sig_status(st_f, "invalid_family_payload", now)
            self._safe_log("invalid_family_payload", **{**payload, "signal_id": signal_id, "error": str(exc)[:240]})
            return {"signal_id": signal_id, "status": st_f, "reason": "invalid_family_payload"}

        try:
            prop_resp = await self._call_typed("proposal", prop_req, timeout=4.0, expected_generation=expected_gen)
        except Exception as exc:
            st_f = self._reject_status_for(conn, "terminal_skip")
            _set_sig_status(st_f, "proposal_failed", now)
            self._safe_log("proposal_failed", **{**payload, "signal_id": signal_id, "error": str(exc)[:240]})
            return {"signal_id": signal_id, "status": st_f, "reason": "proposal_failed"}

        prop = prop_resp.get("proposal") or {}
        payout_exact_frac: Fraction | None = None
        if family == "multiplier":
            cert_cf = payload.get("certified_commission_frac")
            m_dec: SotaMultiplierDecision = evaluate_sota_multiplier_gate(
                prop,
                expected_ev=payload.get("expected_ev"),
                side=side,
                ev_margin=self.ev_margin,
                certified_commission_frac=float(cert_cf) if cert_cf is not None else None,
                max_commission_frac=float(payload.get("max_commission_frac", 0.20)),
                min_stop_out_rel_dist=float(payload.get("min_stop_out_rel_dist", 0.0008)),
            )
            self._safe_log("multiplier_gate_decision", **{**payload, "signal_id": signal_id, "passed": m_dec.passed, "reason": m_dec.reason, "dry_run": self.dry_run_proposal_only})
            if not m_dec.passed:
                st_f = self._reject_status_for(conn, "terminal_skip")
                _set_sig_status(st_f, str(m_dec.reason), now)
                return {"signal_id": signal_id, "status": st_f, "reason": m_dec.reason}
            ask_approved = m_dec.ask
            payout_approved: float | None = 0.0
            cal_ev_logged = m_dec.expected_ev
        else:
            # Exact rational quote-floor + exact whole-cent ask check BEFORE protected evaluate_sota_payout_gate!
            # Defeats BOTH 1.7799999999 < 1.78 AND 1.0000000000000001 != 100 cents (BLOCKER #2)!
            ask_exact_frac, payout_exact_frac, _, rat_err = check_exact_rational_quote_floor(
                prop, min_payout_ratio=payload.get("min_payout_ratio")
            )
            if rat_err is None and is_v3:
                try:
                    want_stake_cents = parse_exact_whole_cents(self.stake_usd, allow_zero=False, allow_negative=False)
                    got_ask_cents = parse_exact_whole_cents(prop.get("ask_price"), allow_zero=False, allow_negative=False)
                    if got_ask_cents != want_stake_cents:
                        rat_err = f"ask_price_cents_mismatch:{got_ask_cents}!={want_stake_cents}"
                except ExecutorError:
                    rat_err = "ask_price_not_exact_whole_cents"
            if rat_err is not None:
                st_f = self._reject_status_for(conn, "terminal_skip")
                _set_sig_status(st_f, rat_err, now)
                self._safe_log("payout_gate_decision", **{**payload, "signal_id": signal_id, "passed": False, "reason": rat_err, "dry_run": self.dry_run_proposal_only})
                return {"signal_id": signal_id, "status": st_f, "reason": rat_err}

            decision: SotaPayoutDecision = evaluate_sota_payout_gate(
                prop,
                effective_floor=float(payload["effective_floor"]),
                absolute_breakeven_ceiling=self.absolute_breakeven_ceiling,
                calibrated_win_prob=payload.get("calibrated_win_prob"),
                ev_margin=self.ev_margin,
                min_payout_ratio=payload.get("min_payout_ratio"),
            )
            self._safe_log(
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
                st_f = self._reject_status_for(conn, "terminal_skip")
                _set_sig_status(st_f, str(decision.reason), now)
                return {"signal_id": signal_id, "status": st_f, "reason": decision.reason}
            ask_approved = decision.ask
            payout_approved = decision.payout
            cal_ev_logged = decision.calibrated_ev

        proposal_id = prop.get("id")
        if not proposal_id:
            st_f = self._reject_status_for(conn, "terminal_skip")
            _set_sig_status(st_f, "missing_proposal_id", now)
            return {"signal_id": signal_id, "status": st_f, "reason": "missing_proposal_id"}

        if self.dry_run_proposal_only:
            st_f = self._reject_status_for(conn, "terminal_skip")
            _set_sig_status(st_f, "dry_run_proposal_passed", now, prop_id=str(proposal_id))
            return {"signal_id": signal_id, "status": st_f, "reason": "dry_run_proposal_passed"}

        assert ask_approved is not None and payout_approved is not None
        post_now = (
            self._post_quote_clock_fn().astimezone(timezone.utc)
            if getattr(self, "_post_quote_clock_fn", None) is not None
            else (
                utc_now().astimezone(timezone.utc)
                if now_utc is None
                else (now + timedelta(seconds=max(0.0, time.monotonic() - mono_start)))
            )
        )
        post_pre_reason = self.pre_proposal_gate(payload, now_utc=post_now)
        if post_pre_reason is not None:
            status = self._reject_status_for(conn, "signal_expired" if post_pre_reason == "signal_expired" else "terminal_skip")
            _set_sig_status(status, post_pre_reason, post_now)
            self._safe_log("signal_rejected_post_quote", **{**payload, "signal_id": signal_id, "reason": post_pre_reason})
            return {"signal_id": signal_id, "status": status, "reason": post_pre_reason}

        post_quote_risk = check_risk_and_exposure(
            conn,
            book_id=str(payload["book_id"]),
            pair=pair,
            horizon_minutes=horizon,
            side=side,
            stake_usd=float(ask_approved),
            cfg=self.risk_cfg,
            now_utc=post_now,
            signal_close_utc=str(payload["signal_close_utc"]),
            venue_time_epoch=self.last_venue_time_epoch,
        )
        if post_quote_risk is not None:
            st_r = self._reject_status_for(conn, "buy_blocked_risk")
            _set_sig_status(st_r, post_quote_risk, post_now)
            self._safe_log("quote_ask_blocked_risk", **{**payload, "signal_id": signal_id, "ask": ask_approved, "reason": post_quote_risk})
            return {"signal_id": signal_id, "status": st_r, "reason": post_quote_risk}

        # Atomically commit durable buy_intent BEFORE transport handoff!
        conn.execute("BEGIN IMMEDIATE")
        try:
            _set_sig_status("buy_intent", f"proposal_id:{proposal_id}", post_now, prop_id=str(proposal_id))
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise

        # Pre-handoff refusal check (0 bytes sent -> buy_refused_pre_send on v3!)
        if isinstance(self.client, SotaDerivAsyncClient):
            pre_refusal: str | None = None
            if not self.client.allow_buy:
                pre_refusal = "allow_buy_false"
            elif is_safety_latch_set(conn) is not None or self.kill_path.exists():
                pre_refusal = "latch_or_kill_pre_handoff"
            elif expected_gen is not None and self.client.generation != expected_gen:
                pre_refusal = "generation_rotated_pre_handoff"
            elif (time.monotonic() - mono_start) > self.max_buy_dispatch_delay_s:
                pre_refusal = "buy_dispatch_delay_exceeded"
            if pre_refusal is not None:
                st_pre = "buy_refused_pre_send" if is_v3 else "terminal_skip"
                _set_sig_status(st_pre, pre_refusal, post_now, prop_id=str(proposal_id))
                self._safe_log("buy_refused_pre_send", signal_id=signal_id, reason=pre_refusal)
                return {"signal_id": signal_id, "status": st_pre, "reason": pre_refusal}

        # BLOCKER #2: Recheck live latch, KILL, dispatch deadline, exact 100-cent ask & quote binding INSIDE handoff lock!
        buy_price_wire = (
            parse_exact_whole_cents(self.stake_usd, allow_zero=False, allow_negative=False) / 100.0
            if (is_v3 and family != "multiplier")
            else float(ask_approved)
        )

        def _handoff_buy_guard(_cls: str, _pay: dict[str, Any]) -> None:
            if is_safety_latch_set(conn) is not None or self.kill_path.exists():
                raise ExecutorError("latch_or_kill_detected_under_handoff_lock")
            if (time.monotonic() - mono_start) > self.max_buy_dispatch_delay_s:
                raise ExecutorError("buy_dispatch_delay_exceeded_under_handoff_lock")
            if expected_gen is not None and isinstance(self.client, SotaDerivAsyncClient) and self.client.generation != expected_gen:
                raise ExecutorError("generation_rotated_under_handoff_lock")
            if is_v3 and family != "multiplier":
                if parse_exact_whole_cents(prop.get("ask_price"), allow_zero=False, allow_negative=False) != parse_exact_whole_cents(self.stake_usd, allow_zero=False, allow_negative=False):
                    raise ExecutorError("exact_ask_cents_mismatch_under_handoff_lock")

        try:
            buy_resp = await self._call_typed(
                "trade_mutation",
                {"buy": str(proposal_id), "price": buy_price_wire, "passthrough": {"sota_signal_id": signal_id}},
                timeout=5.0,
                expected_generation=expected_gen,
                send_guard_fn=_handoff_buy_guard,
            )
        except ExecutorError as exc:
            # Pre-handoff client capability/guard refusal -> 0 bytes sent!
            st_pre = "buy_refused_pre_send" if is_v3 else "terminal_skip"
            _set_sig_status(st_pre, f"pre_send_refused:{str(exc)[:120]}", post_now, prop_id=str(proposal_id))
            self._safe_log("buy_refused_pre_send", signal_id=signal_id, error=str(exc)[:240])
            return {"signal_id": signal_id, "status": st_pre, "reason": "pre_send_refused"}
        except Exception as exc:
            _set_sig_status("buy_unknown", f"proposal_id:{proposal_id}|buy_transport_uncertain", post_now, prop_id=str(proposal_id))
            self._safe_log("buy_unknown", **{**payload, "signal_id": signal_id, "proposal_id": str(proposal_id), "error": str(exc)[:240]})
            return {"signal_id": signal_id, "status": "buy_unknown", "reason": "buy_transport_uncertain"}

        buy_obj = buy_resp.get("buy") or {}
        contract_id = buy_obj.get("contract_id")
        if not contract_id:
            _set_sig_status("buy_unknown", f"proposal_id:{proposal_id}|missing_contract_id", post_now, prop_id=str(proposal_id))
            return {"signal_id": signal_id, "status": "buy_unknown", "reason": "missing_contract_id"}

        # Parse buy_price & payout; on v3 detect differing venue buy cost -> persist & trip phase1_safety_latch!
        raw_bp = buy_obj.get("buy_price")
        cost_unknown_flag = 0
        stake_auth_flag = 1
        latch_reason: str | None = None
        if is_v3:
            try:
                buy_cents = parse_exact_whole_cents(raw_bp, allow_zero=False, allow_negative=False)
                expected_cents = parse_exact_whole_cents(ask_approved, allow_zero=False, allow_negative=False)
                if buy_cents != expected_cents:
                    latch_reason = f"venue_buy_price_differs:{buy_cents}!={expected_cents}"
            except ExecutorError:
                buy_cents = int(round(float(ask_approved) * 100))
                cost_unknown_flag = 1
                stake_auth_flag = 0
                latch_reason = f"venue_buy_price_unparseable:{raw_bp!r}"
            # Fractional-cent proposal payout admitted with payout_cents=NULL on v3!
            if family == "multiplier":
                payout_cents_v3: int | None = None
            else:
                raw_po = buy_obj.get("payout") if buy_obj.get("payout") is not None else payout_exact_frac
                try:
                    payout_cents_v3 = parse_exact_whole_cents(raw_po, allow_zero=False, allow_negative=False)
                except ExecutorError:
                    payout_cents_v3 = None
            buy_price = buy_cents / 100.0
            payout_val = (payout_cents_v3 / 100.0) if payout_cents_v3 is not None else float(payout_approved)
        else:
            buy_price = finite_positive_number(raw_bp) or float(ask_approved)
            payout_val = 0.0 if family == "multiplier" else (finite_positive_number(buy_obj.get("payout")) or float(payout_approved))
            buy_cents = int(round(buy_price * 100))
            payout_cents_v3 = int(round(payout_val * 100))

        fallback_start = int(post_now.timestamp())
        start_ep = _safe_int_epoch(buy_obj.get("start_time"), _safe_int_epoch(buy_obj.get("date_start"), fallback_start))
        exp_ep = (
            start_ep + horizon * 60
            if family == "multiplier"
            else _safe_int_epoch(buy_obj.get("expiry_time"), start_ep + horizon * 60)
        )
        # On v3, NEVER populate authoritative venue_* columns from local clock/config fallbacks (populated strictly from venue responses/POC)!
        venue_start_v3 = _safe_int_epoch(buy_obj.get("start_time"), _safe_int_epoch(buy_obj.get("date_start"), 0)) or None
        venue_end_v3 = _safe_int_epoch(buy_obj.get("expiry_time"), _safe_int_epoch(buy_obj.get("date_expiry"), 0)) or None
        exp_utc = datetime.fromtimestamp(exp_ep, tz=timezone.utc).isoformat()
        sig_close = str(payload["signal_close_utc"])
        book_cid_opt = str(payload["book_content_id"]) if payload.get("book_content_id") else None
        acct_id = self.env.account_id if self.env is not None else "VRTC_LOCAL"

        conn.execute("BEGIN IMMEDIATE")
        try:
            if is_v3:
                conn.execute(
                    """
                    INSERT INTO sota_contracts (
                        contract_id, signal_id, book_id, book_content_id, account_id, account_mode,
                        pair, horizon_minutes, side, contract_family, is_migrated_legacy,
                        cost_unknown, buy_price_cents, payout_cents, stake_authoritative,
                        venue_start_epoch, venue_end_epoch, venue_underlying_symbol, venue_contract_type,
                        signal_close_utc, status, proposal_id, expected_expiry_utc, raw_hash,
                        opened_utc, updated_utc
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?, ?, ?, ?, ?, NULL, NULL, ?, 'open', ?, ?, ?, ?, ?)
                    """,
                    (
                        str(contract_id),
                        signal_id,
                        str(payload["book_id"]),
                        book_cid_opt,
                        acct_id,
                        self.account_mode,
                        pair,
                        horizon,
                        side,
                        family,
                        cost_unknown_flag,
                        buy_cents,
                        payout_cents_v3,
                        stake_auth_flag,
                        venue_start_v3,
                        venue_end_v3,
                        sig_close,
                        str(proposal_id),
                        exp_utc,
                        raw_hash(buy_resp),
                        utc_iso(post_now),
                        utc_iso(post_now),
                    ),
                )
                conn.execute(
                    "UPDATE sota_signals SET status='bought', reason='buy_confirmed', contract_id=?, updated_utc=? WHERE signal_id=?",
                    (str(contract_id), utc_iso(post_now), signal_id),
                )
                if latch_reason is not None:
                    now_l = utc_iso(post_now)
                    conn.execute(
                        """
                        INSERT INTO sota_risk_state (key, value_json, updated_utc)
                        VALUES ('phase1_safety_latch', ?, ?)
                        ON CONFLICT(key) DO UPDATE SET value_json=excluded.value_json, updated_utc=excluded.updated_utc
                        """,
                        (json.dumps({"latched_utc": now_l, "reason": latch_reason, "contract_id": str(contract_id)}, sort_keys=True), now_l),
                    )
            else:
                conn.execute(
                    """
                    INSERT INTO sota_contracts (
                        contract_id, signal_id, book_id, pair, horizon_minutes, side,
                        proposal_id, buy_price_cents, payout_cents, status,
                        opened_utc, expected_expiry_utc, raw_hash, signal_close_utc, contract_family,
                        account_mode, book_content_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'open', ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(contract_id),
                        signal_id,
                        str(payload["book_id"]),
                        pair,
                        horizon,
                        side,
                        str(proposal_id),
                        buy_cents,
                        int(round(payout_val * 100)),
                        utc_iso(post_now),
                        exp_utc,
                        raw_hash(buy_resp),
                        sig_close,
                        family,
                        self.account_mode,
                        book_cid_opt,
                    ),
                )
                conn.execute(
                    "UPDATE sota_signals SET status='bought', reason='buy_confirmed', updated_utc=? WHERE signal_id=?",
                    (utc_iso(post_now), signal_id),
                )
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise

        self._safe_log(
            "buy_confirmed",
            **{
                **payload,
                "signal_id": signal_id,
                "contract_id": str(contract_id),
                "buy_price": buy_price,
                "payout": payout_val,
                "calibrated_ev": cal_ev_logged,
                "account_mode": self.account_mode,
                "latch_reason": latch_reason,
            },
        )
        return {"signal_id": signal_id, "status": "bought", "contract_id": str(contract_id)}

    async def _reconcile_uncertain_signals(self, conn: sqlite3.Connection, now: datetime) -> int:
        assert self.client is not None
        is_v3 = self._is_v3_conn(conn)
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
            port_resp = await self._call_typed("account_read", {"portfolio": 1}, timeout=6.0)
            contracts = (port_resp.get("portfolio") or {}).get("contracts") or []
        except Exception as exc:
            self._safe_log("portfolio_reconcile_error", error=str(exc)[:240])
            contracts = []

        owned_cids = {str(r[0]) for r in conn.execute("SELECT contract_id FROM sota_contracts").fetchall()}
        stmt_txs: list[dict[str, Any]] | None = None
        acct_id = self.env.account_id if self.env is not None else "VRTC_LOCAL"

        for row in uncertain:
            sid = str(row["signal_id"])
            pair = str(row["pair"]).upper()
            horizon = int(row["horizon_minutes"])
            side = str(row["side"]).upper()
            sig_dt = _parse_iso_utc(str(row["signal_close_utc"])) or now
            payload = json.loads(str(row["payload_json"]))
            family = str(payload.get("contract_family", "risefall"))
            reason_str = str(row["reason"] or "")
            want_prop_id = reason_str.split("|", 1)[0][len("proposal_id:") :] if reason_str.startswith("proposal_id:") else ""
            # BLOCKER #3: Collect ALL matching candidates across portfolio AND statement (refuse ambiguous duplicates!).
            matches_by_cid: dict[str, dict[str, Any]] = {}
            for c in contracts:
                cid_cand = str(c.get("contract_id") or "")
                if not cid_cand or cid_cand in owned_cids:
                    continue
                pt_obj = c.get("passthrough") if isinstance(c.get("passthrough"), dict) else {}
                c_prop = str(c.get("proposal_id") or "")
                if pt_obj.get("sota_signal_id") == sid or (want_prop_id and c_prop == want_prop_id):
                    matches_by_cid[cid_cand] = c

            if stmt_txs is None:
                try:
                    st_resp = await self._call_typed("account_read", {"statement": 1, "description": 1, "limit": 25}, timeout=6.0)
                    stmt_txs = ((st_resp.get("statement") or {}).get("transactions")) or []
                except Exception:
                    stmt_txs = []
            for tx in stmt_txs:
                if str(tx.get("action_type") or "").lower() != "buy":
                    continue
                cid_tx = str(tx.get("contract_id") or "")
                if not cid_tx or cid_tx in owned_cids:
                    continue
                pt_tx = tx.get("passthrough") if isinstance(tx.get("passthrough"), dict) else {}
                tx_prop = str(tx.get("proposal_id") or "")
                if pt_tx.get("sota_signal_id") == sid or (want_prop_id and tx_prop == want_prop_id):
                    if cid_tx not in matches_by_cid:
                        matches_by_cid[cid_tx] = {
                            "contract_id": cid_tx,
                            "buy_price": abs(float(tx.get("amount") or self.stake_usd)),
                            "payout": tx.get("payout"),
                            "date_start": _safe_int_epoch(tx.get("transaction_time"), int(sig_dt.timestamp())),
                        }

            if len(matches_by_cid) > 1:
                errors += 1
                cand_list = sorted(matches_by_cid.keys())
                if is_v3:
                    commit_safety_latch(conn, f"ambiguous_uncertain_buy_candidates:{','.join(cand_list)}", signal_id=sid)
                self._safe_log("buy_unknown_ambiguous_candidates", signal_id=sid, candidates=cand_list)
                continue

            matched_c: dict[str, Any] | None = next(iter(matches_by_cid.values())) if len(matches_by_cid) == 1 else None

            if matched_c is not None and matched_c.get("contract_id"):
                cid = str(matched_c["contract_id"])
                if is_v3:
                    # BLOCKER #3: On v3, NEVER clear uncertainty using inferred/rounded cost! Fetch authoritative POC FIRST!
                    try:
                        poc_r = await self._call_typed(
                            "account_read", {"proposal_open_contract": 1, "contract_id": _wire_contract_id(cid)}, timeout=6.0
                        )
                        poc_m = poc_r.get("proposal_open_contract") or {}
                    except Exception as exc:
                        errors += 1
                        self._safe_log("buy_unknown_poc_verify_failed", signal_id=sid, contract_id=cid, error=str(exc)[:240])
                        continue
                    if not poc_m or str(poc_m.get("contract_id") or "") != cid:
                        errors += 1
                        self._safe_log("buy_unknown_poc_incomplete", signal_id=sid, contract_id=cid)
                        continue
                    rec_latch: str | None = None
                    rec_cost_unknown = 0
                    rec_stake_auth = 1
                    want_c = parse_exact_whole_cents(self.stake_usd, allow_zero=False, allow_negative=False)
                    raw_poc_bp = poc_m.get("buy_price")
                    try:
                        bp_cents_v3: int | None = parse_exact_whole_cents(raw_poc_bp, allow_zero=False, allow_negative=False)
                        if bp_cents_v3 != want_c:
                            rec_stake_auth = 0
                            rec_latch = f"recovered_buy_price_differs:{bp_cents_v3}!={want_c}"
                    except ExecutorError:
                        bp_cents_v3 = None
                        rec_cost_unknown = 1
                        rec_stake_auth = 0
                        rec_latch = f"recovered_buy_price_unknown:{raw_poc_bp!r}"
                    raw_poc_po = poc_m.get("payout")
                    try:
                        po_cents_v3: int | None = None if (family == "multiplier" or raw_poc_po is None) else parse_exact_whole_cents(raw_poc_po, allow_zero=False, allow_negative=False)
                    except ExecutorError:
                        po_cents_v3 = None
                    v_start_rec = _safe_int_epoch(poc_m.get("date_start"), 0) or None
                    v_end_rec = _safe_int_epoch(poc_m.get("date_expiry"), _safe_int_epoch(poc_m.get("sell_time"), 0)) or None
                    exp_epoch = (v_start_rec or int(sig_dt.timestamp())) + horizon * 60
                    exp_iso = datetime.fromtimestamp(exp_epoch, tz=timezone.utc).isoformat()
                    book_cid_rec = str(payload["book_content_id"]) if payload.get("book_content_id") else None
                    conn.execute("BEGIN IMMEDIATE")
                    try:
                        cur = conn.execute(
                            """
                            INSERT INTO sota_contracts (
                                contract_id, signal_id, book_id, book_content_id, account_id, account_mode,
                                pair, horizon_minutes, side, contract_family, is_migrated_legacy,
                                cost_unknown, buy_price_cents, payout_cents, stake_authoritative,
                                venue_start_epoch, venue_end_epoch, signal_close_utc, status,
                                proposal_id, expected_expiry_utc, raw_hash, opened_utc, updated_utc
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?, ?, ?, ?, ?, ?, 'open', ?, ?, ?, ?, ?)
                            ON CONFLICT(contract_id) DO NOTHING
                            """,
                            (
                                cid,
                                sid,
                                str(row["book_id"]),
                                book_cid_rec,
                                acct_id,
                                self.account_mode,
                                pair,
                                horizon,
                                side,
                                family,
                                rec_cost_unknown,
                                bp_cents_v3,
                                po_cents_v3,
                                rec_stake_auth,
                                v_start_rec,
                                v_end_rec,
                                str(row["signal_close_utc"]),
                                want_prop_id or "recovered_verified",
                                exp_iso,
                                raw_hash(poc_r),
                                utc_iso(now),
                                utc_iso(now),
                            ),
                        )
                        if cur.rowcount == 1:
                            conn.execute(
                                "UPDATE sota_signals SET status='bought', reason='recovered_from_portfolio_poc', contract_id=?, updated_utc=? WHERE signal_id=?",
                                (cid, utc_iso(now), sid),
                            )
                            if rec_latch is not None:
                                now_rl = utc_iso(now)
                                conn.execute(
                                    """
                                    INSERT INTO sota_risk_state (key, value_json, updated_utc)
                                    VALUES ('phase1_safety_latch', ?, ?)
                                    ON CONFLICT(key) DO UPDATE SET value_json=excluded.value_json, updated_utc=excluded.updated_utc
                                    """,
                                    (json.dumps({"latched_utc": now_rl, "reason": rec_latch, "contract_id": cid}, sort_keys=True), now_rl),
                                )
                            owned_cids.add(cid)
                            conn.execute("COMMIT")
                            self._safe_log("buy_unknown_recovered", signal_id=sid, contract_id=cid, latch_reason=rec_latch)
                        else:
                            conn.execute("ROLLBACK")
                            errors += 1
                            self._safe_log("buy_unknown_recovery_conflict", signal_id=sid, contract_id=cid)
                    except Exception:
                        conn.execute("ROLLBACK")
                        raise
                    continue

                bp = finite_positive_number(matched_c.get("buy_price")) or self.stake_usd
                raw_rec_po = finite_positive_number(matched_c.get("payout"))
                po = 0.0 if family == "multiplier" else (raw_rec_po or round(bp * 1.80, 2))
                dstart_ep = _safe_int_epoch(matched_c.get("date_start"), int(sig_dt.timestamp()))
                exp_epoch = (
                    dstart_ep + horizon * 60
                    if family == "multiplier"
                    else _safe_int_epoch(matched_c.get("expiry_time"), dstart_ep + horizon * 60)
                )
                exp_iso = datetime.fromtimestamp(exp_epoch, tz=timezone.utc).isoformat()
                book_cid_rec = str(payload["book_content_id"]) if payload.get("book_content_id") else None
                conn.execute("BEGIN IMMEDIATE")
                try:
                    cur = conn.execute(
                        """
                        INSERT INTO sota_contracts (
                            contract_id, signal_id, book_id, pair, horizon_minutes, side,
                            proposal_id, buy_price_cents, payout_cents, status,
                            opened_utc, expected_expiry_utc, raw_hash, signal_close_utc, contract_family,
                            account_mode, book_content_id
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'open', ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(contract_id) DO NOTHING
                        """,
                        (
                            cid,
                            sid,
                            str(row["book_id"]),
                            pair,
                            horizon,
                            side,
                            want_prop_id or "recovered_verified",
                            int(round(bp * 100)),
                            int(round(po * 100)),
                            utc_iso(now),
                            exp_iso,
                            raw_hash(matched_c),
                            str(row["signal_close_utc"]),
                            family,
                            self.account_mode,
                            book_cid_rec,
                        ),
                    )
                    if cur.rowcount == 1:
                        conn.execute(
                            "UPDATE sota_signals SET status='bought', reason='recovered_from_portfolio', updated_utc=? WHERE signal_id=?",
                            (utc_iso(now), sid),
                        )
                        owned_cids.add(cid)
                        conn.execute("COMMIT")
                        self._safe_log("buy_unknown_recovered", signal_id=sid, contract_id=cid)
                    else:
                        conn.execute("ROLLBACK")
                        errors += 1
                        self._safe_log("buy_unknown_recovery_conflict", signal_id=sid, contract_id=cid)
                except Exception:
                    conn.execute("ROLLBACK")
                    raise
            else:
                errors += 1
                self._safe_log("buy_unknown_unmatched", signal_id=sid, pair=pair, horizon_minutes=horizon)
        return errors

    async def reconcile_open_contracts(self, conn: sqlite3.Connection, *, now_utc: datetime | None = None) -> None:
        assert self.client is not None
        await self.ensure_authenticated()
        is_v3 = self._is_v3_conn(conn)
        now = (now_utc or utc_now()).astimezone(timezone.utc)
        errors = await self._reconcile_uncertain_signals(conn, now)

        open_rows = conn.execute(
            (
                "SELECT contract_id, buy_price_cents, payout_cents, sell_price_cents, sell_ack_cents, expected_expiry_utc, contract_family, cost_unknown FROM sota_contracts WHERE status = 'open'"
                if is_v3
                else "SELECT contract_id, buy_price_cents, payout_cents, sell_price_cents, sell_ack_cents, expected_expiry_utc, contract_family, 0 AS cost_unknown FROM sota_contracts WHERE status = 'open'"
            )
        ).fetchall()
        if not open_rows and errors == 0:
            self.consecutive_reconcile_errors = 0
            return

        for row in open_rows:
            cid = str(row["contract_id"])
            wire_cid = _wire_contract_id(cid)
            family = str(row["contract_family"] or "risefall")
            sold_for_cents: int | None = int(row["sell_ack_cents"]) if row["sell_ack_cents"] is not None else None
            try:
                resp = await self._call_typed("account_read", {"proposal_open_contract": 1, "contract_id": wire_cid}, timeout=6.0)
                poc = resp.get("proposal_open_contract") or {}
                # BLOCKER #5: Require POC contract_id to match the requested contract_id (reject mismatched/empty v3 POC)!
                poc_cid = str(poc.get("contract_id") or "")
                if (is_v3 and poc_cid != cid) or (poc_cid and poc_cid != cid):
                    errors += 1
                    self._safe_log("poc_contract_id_mismatch", expected_contract_id=cid, got_contract_id=poc_cid)
                    continue
                status_raw = str(poc.get("status") or "open").lower()
                is_terminal = (poc.get("is_sold") in {1, True, "1"}) or (status_raw in TERMINAL_CONTRACT_STATUSES)
                exp_dt = _parse_iso_utc(str(row["expected_expiry_utc"]))
                if (
                    not is_terminal
                    and sold_for_cents is None
                    and not self.dry_run_proposal_only
                    and family == "multiplier"
                    and exp_dt is not None
                    and now >= exp_dt
                ):
                    if poc.get("is_valid_to_sell") in {1, True, "1"}:
                        sell_resp = await self._call_typed("trade_mutation", {"sell": wire_cid, "price": 0}, timeout=6.0)
                        sell_obj = sell_resp.get("sell") or {}
                        if "sold_for" in sell_obj and sell_obj["sold_for"] is not None:
                            sold_for_cents = int(round(float(sell_obj["sold_for"]) * 100))
                            conn.execute(
                                "UPDATE sota_contracts SET sell_ack_cents=?, sell_price_cents=? WHERE contract_id=? AND status='open'",
                                (sold_for_cents, sold_for_cents, cid),
                            )
                            self._safe_log("multiplier_timed_sell_sent", contract_id=cid, sold_for=sold_for_cents / 100.0)
                            resp = await self._call_typed("account_read", {"proposal_open_contract": 1, "contract_id": wire_cid}, timeout=6.0)
                            poc = resp.get("proposal_open_contract") or {}
                            poc_cid2 = str(poc.get("contract_id") or "")
                            if (is_v3 and poc_cid2 != cid) or (poc_cid2 and poc_cid2 != cid):
                                errors += 1
                                self._safe_log("poc_contract_id_mismatch", expected_contract_id=cid, got_contract_id=poc_cid2)
                                continue
                            status_raw = str(poc.get("status") or "open").lower()
                            is_terminal = (poc.get("is_sold") in {1, True, "1"}) or (status_raw in TERMINAL_CONTRACT_STATUSES)

                # BLOCKER #5: On v3, require explicit authoritative sell_price + profit from POC (never infer 0 or carry forward!)
                if is_v3:
                    has_authoritative_money = (poc.get("profit") is not None and poc.get("sell_price") is not None)
                else:
                    has_authoritative_money = (
                        poc.get("profit") is not None
                        or poc.get("sell_price") is not None
                        or (is_terminal and sold_for_cents is not None)
                    )
                if is_terminal and has_authoritative_money:
                    raw_sp = poc.get("sell_price")
                    if raw_sp is not None:
                        sell_cents = parse_exact_whole_cents(raw_sp, allow_zero=True, allow_negative=False) if is_v3 else int(round(float(raw_sp) * 100))
                    elif sold_for_cents is not None:
                        sell_cents = sold_for_cents
                    else:
                        sell_cents = 0
                    row_cost_unknown = int(row["cost_unknown"] or 0)
                    buy_cents_existing: int | None = int(row["buy_price_cents"]) if row["buy_price_cents"] is not None else None
                    resolved_buy_cents = buy_cents_existing
                    resolved_cost_unknown = row_cost_unknown
                    if is_v3 and (row_cost_unknown == 1 or buy_cents_existing is None):
                        raw_poc_bp = poc.get("buy_price")
                        if raw_poc_bp is not None:
                            resolved_buy_cents = parse_exact_whole_cents(raw_poc_bp, allow_zero=False, allow_negative=False)
                            resolved_cost_unknown = 0
                    buy_cents = resolved_buy_cents if resolved_buy_cents is not None else 0
                    raw_pf = poc.get("profit")
                    profit_cents = (
                        (parse_exact_whole_cents(raw_pf, allow_zero=True, allow_negative=True) if is_v3 else int(round(float(raw_pf) * 100)))
                        if raw_pf is not None
                        else (sell_cents - buy_cents)
                    )
                    if status_raw in {"expired", "cancelled"} and profit_cents == 0 and sell_cents == 0 and not is_v3:
                        final_st = status_raw
                    else:
                        final_st = "won" if profit_cents > 0 else ("lost" if profit_cents < 0 else "sold")

                    conn.execute("BEGIN IMMEDIATE")
                    try:
                        if is_v3:
                            v_start = _safe_int_epoch(poc.get("date_start"), 0) or None
                            v_end = _safe_int_epoch(poc.get("date_expiry"), _safe_int_epoch(poc.get("sell_time"), 0)) or None
                            v_sym = str(poc.get("underlying") or poc.get("underlying_symbol") or "") or None
                            v_ct = str(poc.get("contract_type") or "") or None
                            v_term = status_raw if status_raw in {"won", "lost", "sold"} else None
                            conn.execute(
                                """
                                UPDATE sota_contracts
                                SET status=?, buy_price_cents=?, cost_unknown=?, sell_price_cents=?, profit_cents=?, closed_utc=?, raw_hash=?,
                                    venue_start_epoch=?,
                                    venue_end_epoch=?,
                                    venue_terminal_status=?,
                                    venue_underlying_symbol=?,
                                    venue_contract_type=?,
                                    updated_utc=?
                                WHERE contract_id=?
                                """,
                                (final_st, resolved_buy_cents, resolved_cost_unknown, sell_cents, profit_cents, utc_iso(now), raw_hash(resp), v_start, v_end, v_term, v_sym, v_ct, utc_iso(now), cid),
                            )
                        else:
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
                    self._safe_log("contract_settled", contract_id=cid, status=final_st, profit_usd=profit_cents / 100.0)
                else:
                    if exp_dt is not None and (now - exp_dt).total_seconds() > OVERDUE_OPEN_GRACE_S:
                        errors += 1
                        self._safe_log("contract_overdue_open", contract_id=cid, expected_expiry_utc=str(row["expected_expiry_utc"]))
            except Exception as exc:
                errors += 1
                self._safe_log("reconcile_error", contract_id=cid, error=str(exc)[:240])

        if errors:
            self.consecutive_reconcile_errors += 1
            if self.consecutive_reconcile_errors >= 5:
                self.trip_kill_switch("consecutive_reconcile_errors", count=self.consecutive_reconcile_errors)
        else:
            self.consecutive_reconcile_errors = 0


def _check_no_legacy_buyers_proc() -> None:
    self_pid = os.getpid()
    for item in Path("/proc").iterdir():
        if item.name.isdigit() and int(item.name) != self_pid:
            parts = proc_cmdline(item)
            is_v1 = any("deriv_trade_executor.py" in tok for tok in parts) and ("--run" in parts or "--once" in parts)
            if proc_offender(parts) or is_v1:
                raise ExecutorError(f"legacy buy-capable process running (pid={item.name}): {' '.join(parts)}")


async def run_integrated_vps_loop(args: argparse.Namespace) -> int:
    env = DerivEnv.from_env()
    if getattr(args, "fixture_home", None) or getattr(args, "fixture_repo_root", None):
        raise ExecutorError("--fixture-home/--fixture-repo-root are forbidden on deriv_sota_executor.py")

    # BLOCKER #1: Unconditional Phase 1 non-bypassable config validation (omitting --phase1-policy can NEVER bypass!)
    cfg = resolve_and_validate_sota_runtime_config(
        account_mode=args.account_mode,
        account_id=env.account_id,
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
        allow_fixture=False,
    )
    if cfg.lock_paths["hold"].exists():
        raise ExecutorError(f"activation hold present: {cfg.lock_paths['hold']}")
    _check_no_legacy_buyers_proc()
    books = load_enabled_sota_books(repo_root=cfg.repo_root, account_mode="demo", book_ids=list(cfg.book_ids))
    db_path = cfg.queue_db
    log_dir = cfg.log_dir
    lock_paths = cfg.lock_paths
    stake_usd = cfg.stake_usd
    risk_cfg = RiskBreakerConfig(
        max_stake_usd=cfg.max_stake_usd,
        max_daily_loss_usd=cfg.max_daily_loss_usd,
        max_drawdown_usd=cfg.max_drawdown_usd,
        max_open_contracts=cfg.max_open_contracts,
        max_usd_factor_open=cfg.max_usd_factor_open,
        max_jpy_factor_open=cfg.max_jpy_factor_open,
    )

    ws_url, transport_kind = await asyncio.to_thread(request_authenticated_ws_url, env, args.account_mode)
    pairs_needed = sorted({b.pair for b in books})
    tick_store_dir = Path(args.tick_store_dir)
    external_ticks = str(args.tick_feed_mode).lower() == "external"

    async def _async_refresh_auth_url() -> str:
        fresh_url, _ = await asyncio.to_thread(request_authenticated_ws_url, env, args.account_mode)
        return fresh_url

    async def _async_public_ws_url() -> str:
        assert_non_us_deriv_host()
        return PUBLIC_WS_URL

    # MAJOR #2: Acquire locks, open bound v3 queue & determine recovery mode BEFORE initializing logging/tick_buf!
    with acquire_ordered_locks([(lock_paths["acct"], True, True), (lock_paths["queue"], True, True)]):
        conn = connect_sota_queue_v3(db_path, expected_account_id=env.account_id, expected_account_mode=args.account_mode)
        latched_recovery_only = is_safety_latch_set(conn) is not None or (log_dir / "KILL").exists()
        executor = SotaTradeExecutor(
            account_mode=args.account_mode,
            stake_usd=stake_usd,
            db_path=db_path,
            log_dir=log_dir,
            risk_cfg=risk_cfg,
            absolute_breakeven_ceiling=cfg.absolute_breakeven_ceiling,
            ev_margin=cfg.ev_margin,
            dry_run_proposal_only=args.dry_run_proposal_only,
            max_buy_dispatch_delay_s=cfg.max_buy_dispatch_delay_s,
            env=env,
        )
        executor.transport_kind = transport_kind

        # BLOCKER #2: Wire live pre_send_guard_fn inside sota_handoff lock on production auth_client!
        def _prod_handoff_guard(_primary: str, _payload: dict[str, Any]) -> None:
            if is_safety_latch_set(conn) is not None or executor.kill_path.exists():
                raise ExecutorError("production send refused under handoff lock: latch or KILL active")

        async with AsyncExitStack() as stack:
            pub_client: SotaDerivAsyncClient | None = None
            tick_buf: LiveTickBuffer | None = None
            if not latched_recovery_only:
                tick_buf = LiveTickBuffer(pairs=pairs_needed, store_dir=tick_store_dir)
                tick_buf.load_persisted()
                if not external_ticks:
                    assert_non_us_deriv_host()
                    pub_client = await stack.enter_async_context(
                        SotaDerivAsyncClient(
                            PUBLIC_WS_URL,
                            allow_buy=False,
                            silence_timeout_s=45.0,
                            url_factory=_async_public_ws_url,
                        )
                    )
            auth_client = await stack.enter_async_context(
                SotaDerivAsyncClient(
                    ws_url,
                    allow_buy=(not args.dry_run_proposal_only and not latched_recovery_only),
                    expected_account_id=env.account_id,
                    handoff_lock_path=lock_paths["handoff"],
                    pre_send_guard_fn=_prod_handoff_guard,
                    url_factory=_async_refresh_auth_url,
                )
            )
            executor.client = auth_client
            await executor.ensure_authenticated()
            await executor.reconcile_open_contracts(conn)

            subs: dict[str, Subscription] = {}
            if not latched_recovery_only and not external_ticks:
                assert pub_client is not None and tick_buf is not None

                async def _warmup_reconcile_bg() -> None:
                    while True:
                        await asyncio.sleep(5.0)
                        await executor.reconcile_open_contracts(conn)

                bg_rec = asyncio.create_task(_warmup_reconcile_bg())
                try:
                    await tick_buf.warmup_from_public_ws(
                        pub_client,
                        lookback_hours=36,
                        min_bars=DEFAULT_WARMUP_MIN_BARS,
                        history_request_fn=pub_client.request_public_read,
                    )
                finally:
                    bg_rec.cancel()
                    try:
                        await bg_rec
                    except asyncio.CancelledError:
                        pass
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
                # Re-check latch/KILL dynamically: enter log/feed-independent recovery-only mode immediately
                if is_safety_latch_set(conn) is not None or executor.kill_path.exists():
                    auth_client.allow_buy = False
                    latched_recovery_only = True

                if not latched_recovery_only and not external_ticks and pub_client is not None and tick_buf is not None:
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
                if not latched_recovery_only and tick_buf is not None and minute_close > last_scored_minute and (now_wall - minute_close) >= 1.2:
                    ready_to_score = True
                    if external_ticks:
                        wm = read_tick_store_watermark(tick_store_dir)
                        wm_min = int(wm.get("minute_boundary_epoch") or 0)
                        if wm_min < minute_close:
                            if (now_wall - minute_close) < 4.5:
                                ready_to_score = False
                            else:
                                last_scored_minute = minute_close
                                ready_to_score = False
                                executor._safe_log("external_watermark_lag_skip", minute_close=minute_close, watermark_minute=wm_min)
                        else:
                            tick_buf.load_persisted()

                    if ready_to_score:
                        last_scored_minute = minute_close
                        await executor.reconcile_open_contracts(conn)
                        last_reconcile = time.monotonic()
                        snap = tick_buf.snapshot_ticks()
                        # Cache ONE causal snapshot-or-error per (pair, minute_close) across books in priority order!
                        pair_cache: dict[tuple[str, int], tuple[Any | None, Exception | None]] = {}
                        for book in books:
                            cache_key = (book.pair, minute_close)
                            if cache_key not in pair_cache:
                                union_cols = sorted({c for b in books if b.pair == book.pair for c in b.feature_cols})
                                try:
                                    frow_c = build_pair_decision_snapshot(
                                        book.pair,
                                        snap,
                                        minute_close,
                                        union_cols,
                                        min_active_bars=DEFAULT_WARMUP_MIN_BARS,
                                    )
                                    pair_cache[cache_key] = (frow_c, None)
                                except Exception as exc:
                                    pair_cache[cache_key] = (None, exc)
                            frow, f_err = pair_cache[cache_key]
                            if f_err is not None or frow is None:
                                executor._safe_log("feature_or_book_skip", book_id=book.book_id, pair=book.pair, reason=str(f_err))
                                continue
                            try:
                                sc = book.score_row(frow.row, minute_close)
                                executor._safe_log(
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
                                        "multiplier": book.multiplier,
                                        "expected_ev": sc.expected_ev,
                                        "certified_commission_frac": book.certified_commission_frac,
                                        "max_commission_frac": book.max_commission_frac,
                                        "min_stop_out_rel_dist": book.min_stop_out_rel_dist,
                                        "barrier_contract_type": book.barrier_contract_type,
                                        "barrier_rel_half_width": book.barrier_rel_half_width,
                                        "spot_ref": frow.spot_ref,
                                        "book_content_id": book.content_id,
                                    }
                                    inserted, sid = enqueue_sota_signal(conn, sig)
                                    if inserted:
                                        await executor.process_one_signal(conn, sid, sig)
                            except (SotaFeatureError, SotaBookError) as exc:
                                executor._safe_log("feature_or_book_skip", book_id=book.book_id, pair=book.pair, reason=str(exc))

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
    # Mutually exclusive actions (spec §3.6 9-action matrix)
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--init-queue", action="store_true")
    ap.add_argument("--export-legacy-ownership-preimage", action="store_true")
    ap.add_argument("--adopt-legacy-account", action="store_true")
    ap.add_argument("--register-cohort-anchor", action="store_true")
    ap.add_argument("--clear-stale-legacy-lock", action="store_true")
    ap.add_argument("--request-stop-latch", action="store_true")

    ap.add_argument("--phase1-policy", action="store_true")
    ap.add_argument("--require-explicit-book-ids", action="store_true")
    ap.add_argument("--enforce-3book-cutover", action="store_true")
    ap.add_argument("--account-mode", choices=("demo", "live"), default=os.environ.get("DERIV_ACCOUNT_MODE", "demo"))
    ap.add_argument("--book-ids", default=os.environ.get("DERIV_SOTA_BOOK_IDS", ""))
    ap.add_argument("--book-id", default="")
    ap.add_argument("--tick-feed-mode", choices=("integrated", "external"), default=os.environ.get("DERIV_SOTA_TICK_FEED_MODE", "integrated"))
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
    ap.add_argument("--dry-run-proposal-only", action="store_true")
    ap.add_argument("--queue-db", default=os.environ.get("DERIV_SOTA_QUEUE_DB", ""))
    ap.add_argument("--tick-store-dir", default=os.environ.get("DERIV_SOTA_TICK_STORE_DIR", str(DEFAULT_SOTA_TICK_STORE)))
    ap.add_argument("--log-dir", default=os.environ.get("DERIV_SOTA_LOG_DIR", "logs/sota_runtime"))
    ap.add_argument("--lock-root", default=os.environ.get("DERIV_RUNTIME_LOCK_ROOT", ""))
    ap.add_argument("--credential-isolation-receipt", default=os.environ.get("DERIV_SOTA_CREDENTIAL_ISOLATION_RECEIPT", ""))
    ap.add_argument("--reason", default="")
    ap.add_argument("--first-use-record-json", default="")
    ap.add_argument("--maintenance-quarantine-dir", default="")
    ap.add_argument("--staged-v3-dir", default="")
    ap.add_argument("--ownership-proof-json", default="")
    ap.add_argument("--cohort-start-utc", default="")
    ap.add_argument("--duration-seconds", type=float, default=0.0)
    # Explicitly parse fixture flags solely to refuse them fail-closed on production CLI!
    ap.add_argument("--fixture-home", default=None)
    ap.add_argument("--fixture-repo-root", default=None)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.fixture_home is not None or args.fixture_repo_root is not None:
        raise ExecutorError("--fixture-home and --fixture-repo-root are strictly forbidden on deriv_sota_executor.py")

    actions = [
        name
        for name, on in (
            ("--run", args.run),
            ("--init-queue", args.init_queue),
            ("--export-legacy-ownership-preimage", args.export_legacy_ownership_preimage),
            ("--adopt-legacy-account", args.adopt_legacy_account),
            ("--register-cohort-anchor", args.register_cohort_anchor),
            ("--clear-stale-legacy-lock", args.clear_stale_legacy_lock),
            ("--request-stop-latch", args.request_stop_latch),
        )
        if on
    ]
    if len(actions) > 1:
        raise ExecutorError(f"mutually exclusive actions specified: {actions}")
    if len(actions) == 0:
        build_parser().print_help()
        return 0

    acct_id = os.environ.get("DERIV_ACCOUNT_ID", "").strip()
    q_db = canonical_sota_queue_path(args.account_mode)

    if args.run:
        return asyncio.run(run_integrated_vps_loop(args))

    if args.export_legacy_ownership_preimage:
        if not acct_id:
            raise ExecutorError("missing DERIV_ACCOUNT_ID")
        if not args.maintenance_quarantine_dir:
            raise ExecutorError("--export-legacy-ownership-preimage requires --maintenance-quarantine-dir")
        _check_no_legacy_buyers_proc()
        locks_e = canonical_sota_lock_paths(q_db)
        with acquire_ordered_locks([(locks_e["acct"], True, True), (locks_e["queue"], True, True)]):
            out = export_legacy_ownership_preimage(Path(args.maintenance_quarantine_dir))
        print(json.dumps(out, indent=2, sort_keys=True))
        return 0

    if args.request_stop_latch:
        if not acct_id:
            raise ExecutorError("missing DERIV_ACCOUNT_ID")
        out = request_stop_latch(
            queue_db=q_db,
            account_id=acct_id,
            account_mode=args.account_mode,
            reason=args.reason,
            log_dir=Path(args.log_dir) if args.log_dir else None,
        )
        print(json.dumps(out, indent=2, sort_keys=True))
        return 0

    if args.clear_stale_legacy_lock:
        if not acct_id:
            raise ExecutorError("missing DERIV_ACCOUNT_ID")
        _check_no_legacy_buyers_proc()
        locks_c = canonical_sota_lock_paths(q_db)
        with acquire_ordered_locks([(locks_c["acct"], True, True), (locks_c["queue"], True, True)]):
            out = clear_stale_legacy_lock(account_id=acct_id, account_mode=args.account_mode)
        print(json.dumps(out, indent=2, sort_keys=True))
        return 0

    if args.init_queue:
        if args.reason != "first_use" or not args.first_use_record_json or not args.credential_isolation_receipt:
            raise ExecutorError("--init-queue requires --reason first_use, --first-use-record-json, and --credential-isolation-receipt")
        env = DerivEnv.from_env()
        if args.account_mode != "demo" or not is_valid_demo_account_id(env.account_id):
            raise ExecutorError("Phase 1 --init-queue requires a concrete demo account")
        rcpt_pre = validate_credential_isolation_receipt(
            Path(args.credential_isolation_receipt),
            expected_account_id=env.account_id,
            expected_account_mode=args.account_mode,
        )
        _validate_first_use_record(
            Path(args.first_use_record_json),
            account_id=env.account_id,
            account_mode=args.account_mode,
            rcpt_obj=rcpt_pre,
        )
        async def _verify_venue() -> int:
            ws_url, tk = await asyncio.to_thread(request_authenticated_ws_url, env, args.account_mode)
            async with SotaDerivAsyncClient(ws_url, allow_buy=False, expected_account_id=env.account_id) as cli:
                return await cli.establish_identity(account_id=env.account_id, account_mode=args.account_mode, transport_kind=tk, pat=env.pat)
        asyncio.run(_verify_venue())
        out = init_v3_queue_first_use(
            queue_db=q_db,
            account_id=env.account_id,
            account_mode=args.account_mode,
            first_use_record_path=Path(args.first_use_record_json),
            receipt_path=Path(args.credential_isolation_receipt),
        )
        pub_out = {k: v for k, v in out.items() if k != "account_id"}
        if "account_id" in out:
            pub_out["account_id_sha256"] = redact_account_id(str(out["account_id"]))
        print(json.dumps(pub_out, indent=2, sort_keys=True))
        return 0

    if args.adopt_legacy_account:
        if not all((args.maintenance_quarantine_dir, args.staged_v3_dir, args.ownership_proof_json, args.credential_isolation_receipt)):
            raise ExecutorError("--adopt-legacy-account requires quarantine dir, staged dir, proof JSON, and receipt")
        _check_no_legacy_buyers_proc()
        env = DerivEnv.from_env()
        if args.account_mode != "demo" or not is_valid_demo_account_id(env.account_id):
            raise ExecutorError("Phase 1 --adopt-legacy-account requires a concrete demo account")
        validate_credential_isolation_receipt(
            Path(args.credential_isolation_receipt),
            expected_account_id=env.account_id,
            expected_account_mode=args.account_mode,
        )
        async def _verify_venue_ad() -> int:
            ws_url, tk = await asyncio.to_thread(request_authenticated_ws_url, env, args.account_mode)
            async with SotaDerivAsyncClient(ws_url, allow_buy=False, expected_account_id=env.account_id) as cli:
                return await cli.establish_identity(account_id=env.account_id, account_mode=args.account_mode, transport_kind=tk, pat=env.pat)
        asyncio.run(_verify_venue_ad())
        out = adopt_legacy_account(
            canonical_db=q_db,
            quarantine_dir=Path(args.maintenance_quarantine_dir),
            staged_v3_dir=Path(args.staged_v3_dir),
            ownership_proof_path=Path(args.ownership_proof_json),
            receipt_path=Path(args.credential_isolation_receipt),
            account_id=env.account_id,
            account_mode=args.account_mode,
        )
        print(json.dumps(out, indent=2, sort_keys=True))
        return 0

    if args.register_cohort_anchor:
        if not all((args.book_id, args.cohort_start_utc, args.credential_isolation_receipt)):
            raise ExecutorError("--register-cohort-anchor requires --book-id, --cohort-start-utc, and --credential-isolation-receipt")
        env = DerivEnv.from_env()
        if args.account_mode != "demo" or not is_valid_demo_account_id(env.account_id):
            raise ExecutorError("Phase 1 --register-cohort-anchor requires a concrete demo account")
        validate_credential_isolation_receipt(
            Path(args.credential_isolation_receipt),
            expected_account_id=env.account_id,
            expected_account_mode=args.account_mode,
        )
        async def _get_vtime() -> int:
            ws_url, tk = await asyncio.to_thread(request_authenticated_ws_url, env, args.account_mode)
            async with SotaDerivAsyncClient(ws_url, allow_buy=False, expected_account_id=env.account_id) as cli:
                return await cli.establish_identity(account_id=env.account_id, account_mode=args.account_mode, transport_kind=tk, pat=env.pat)
        v_ep = asyncio.run(_get_vtime())
        out = register_cohort_anchor(
            queue_db=q_db,
            account_id=env.account_id,
            account_mode=args.account_mode,
            book_id=args.book_id,
            cohort_start_utc=args.cohort_start_utc,
            venue_time_epoch=v_ep,
            receipt_path=Path(args.credential_isolation_receipt),
        )
        pub_out = dict(out)
        if isinstance(pub_out.get("anchor"), dict):
            anc = {k: v for k, v in pub_out["anchor"].items() if k != "account_id"}
            if "account_id" in pub_out["anchor"]:
                anc["account_id_sha256"] = redact_account_id(str(pub_out["anchor"]["account_id"]))
            pub_out["anchor"] = anc
        print(json.dumps(pub_out, indent=2, sort_keys=True))
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
