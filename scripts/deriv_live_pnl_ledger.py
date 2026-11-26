"""Per-book exact-cent VENUE_SETTLED P&L ledger and strictly prospective G6 cohort promotion gate.

Sibling to scripts/deriv_economics_ledger_v2.py (which stays byte-frozen for ROS-3A seals).
Supports:
- Strictly prospective Phase-1 G6 cohort evaluation (spec §3.5): canonical queue enforcement
  (`--db` diagnostic alias sets `promotion_eligible: False` on non-canonical paths), stored anchor
  matching (`registered_utc < cohort_start_utc <= eval_now_utc`) and frozen book manifest agreement,
  strict `venue_start_epoch >= cohort_start_epoch` membership, full contract-spec & Deriv
  `proposal_open_contract` venue-field agreement, closed-interval & signal-gap non-overlap checks,
  and exact `Fraction(s[124] + 39 * s[125], 40) / (n * 100) > 0` bootstrap over 5,000 default_rng(7) draws.
- Backwards-compatible legacy summary mode when invoked without prospective cohort parameters.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

from deriv_runtime_core_v2 import SOTA_DERIV_SYMBOLS, contract_type_for

SETTLED_STATUSES = frozenset({"won", "lost", "sold", "expired", "cancelled"})
G6_ELIGIBLE_SETTLED_STATUSES = frozenset({"won", "lost", "sold"})
PROMOTION_HARD_MIN_TRADES = 50
REPO_ROOT = Path(__file__).resolve().parents[1]


def _parse_iso_utc(val: str | None) -> datetime | None:
    if not val or not isinstance(val, str):
        return None
    try:
        dt_v = datetime.fromisoformat(val.replace("Z", "+00:00"))
        if dt_v.tzinfo is None:
            return None
        return dt_v.astimezone(timezone.utc)
    except Exception:
        return None


def _parse_whole_second_epoch(val: str | None) -> int | None:
    dt_v = _parse_iso_utc(val)
    if dt_v is None or dt_v.microsecond != 0:
        return None
    return int(dt_v.timestamp())


def _load_registered_books(repo_root: Path | None) -> dict[str, dict[str, Any]]:
    if repo_root is None:
        return {}
    idx_path = repo_root / "books" / "SOTA_INDEX.json"
    if not idx_path.is_file():
        return {}
    try:
        obj = json.loads(idx_path.read_text(encoding="utf-8"))
        out: dict[str, dict[str, Any]] = {}
        for b in obj.get("books") or []:
            if isinstance(b, dict) and b.get("id") and b.get("content_id"):
                bid = str(b["id"])
                ent = dict(b)
                man_path = repo_root / "books" / bid / f"{bid}.manifest.json"
                strat_path = repo_root / "books" / bid / f"{bid}_strategy.json"
                if man_path.is_file():
                    ent["_manifest"] = json.loads(man_path.read_text(encoding="utf-8"))
                if strat_path.is_file():
                    ent["_strategy"] = json.loads(strat_path.read_text(encoding="utf-8"))
                out[bid] = ent
        return out
    except Exception:
        return {}


def _bootstrap_ci95_float(values: np.ndarray, n_boot: int = 5000, seed: int = 7) -> tuple[float, float]:
    if len(values) < 5:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    n = len(values)
    means = np.array([values[rng.integers(0, n, n)].mean() for _ in range(n_boot)], dtype=np.float64)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def exact_fraction_bootstrap_ci95(profits_cents: list[int], stake_cents: int = 100, *, n_boot: int = 5000, seed: int = 7) -> tuple[Fraction, Fraction]:
    """Exact Fraction linear 2.5th & 97.5th percentile bootstrap (pure Python ints — zero int64 overflow!)."""
    n = len(profits_cents)
    if n < 1 or stake_cents <= 0:
        raise ValueError("exact_fraction_bootstrap_ci95 requires n >= 1 and stake_cents > 0")
    py_profits = [int(x) for x in profits_cents]
    rng = np.random.default_rng(seed)
    # Sum profits strictly in unbounded Python ints per bootstrap draw (immune to NumPy int64 overflow!)
    sums: list[int] = [sum(py_profits[i] for i in rng.integers(0, n, n).tolist()) for _ in range(n_boot)]
    sums.sort()
    # Linear percentile index for 2.5% of 5000: 0.025 * 4999 = 124.975 = 124 + 39/40 -> (s[124] + 39*s[125]) / 40
    # Linear percentile index for 97.5% of 5000: 0.975 * 4999 = 4874.025 = 4874 + 1/40 -> (39*s[4874] + s[4875]) / 40
    denom = n * int(stake_cents)
    if n_boot == 5000:
        lo = Fraction(sums[124] + 39 * sums[125], 40 * denom)
        hi = Fraction(39 * sums[4874] + sums[4875], 40 * denom)
    else:
        pos_lo = Fraction(25, 1000) * (n_boot - 1)
        i_lo = int(pos_lo)
        f_lo = pos_lo - i_lo
        lo = (Fraction(sums[i_lo]) * (1 - f_lo) + Fraction(sums[min(i_lo + 1, n_boot - 1)]) * f_lo) / denom
        pos_hi = Fraction(975, 1000) * (n_boot - 1)
        i_hi = int(pos_hi)
        f_hi = pos_hi - i_hi
        hi = (Fraction(sums[i_hi]) * (1 - f_hi) + Fraction(sums[min(i_hi + 1, n_boot - 1)]) * f_hi) / denom
    return lo, hi


def _open_readonly_snapshot(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{db_path.resolve()}?mode=ro", uri=True, isolation_level=None)
    conn.row_factory = sqlite3.Row
    return conn


def summarize_settled_queue(
    db_path: Path,
    *,
    min_settled_for_promotion: int = 50,
    book_id: str | None = None,
    expected_content_id: str | None = None,
    repo_root: Path | None = REPO_ROOT,
    require_registered_book: bool = False,
    cohort_start_utc: str | None = None,
    expected_account_id: str | None = None,
    expected_stake_usd: str | float | None = None,
    eval_now_utc: str | None = None,
    canonical_db_path: Path | None = None,
    enforce_canonical_db: bool = False,
) -> dict[str, Any]:
    """Evaluate per-book settled queue; enforces full prospective G6 contract when cohort_start_utc or v3 anchor mode is requested."""
    prospective_mode = bool(
        cohort_start_utc is not None
        or expected_account_id is not None
        or expected_stake_usd is not None
        or eval_now_utc is not None
        or enforce_canonical_db
    )
    schema = "deriv-sota-live-pnl-ledger/v3" if prospective_mode else "deriv-sota-live-pnl-ledger/v2"
    db_p = Path(db_path)
    if not db_p.exists():
        return {
            "schema": schema,
            "db_path": str(db_p),
            "book_id": book_id,
            "settled_contracts": 0,
            "open_contracts": 0,
            "unresolved_signals": 0,
            "promotion_eligible_books": [],
            "promotion_eligible": False if book_id else None,
            "reasons": ["queue_db_missing"],
            "by_book": {},
        }

    registered_books = _load_registered_books(repo_root)
    registered_cids = {k: str(v["content_id"]) for k, v in registered_books.items()}

    # Open strictly read-only (zero writes, zero writable PRAGMAs) & read ONE consistent snapshot!
    conn = _open_readonly_snapshot(db_p)
    try:
        conn.execute("BEGIN")
        user_ver = int(conn.execute("PRAGMA user_version").fetchone()[0])
        con_cols = {str(r["name"]) for r in conn.execute("PRAGMA table_info(sota_contracts)").fetchall()}
        sig_cols = {str(r["name"]) for r in conn.execute("PRAGMA table_info(sota_signals)").fetchall()}
        risk_tables = {str(r[0]) for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}

        def _col(name: str, alias: str | None = None) -> str:
            al = alias or name
            return f"c.{name} AS {al}" if name in con_cols else f"NULL AS {al}"

        sig_join_fields = (
            "s.book_id AS sig_book_id, s.pair AS sig_pair, s.horizon_minutes AS sig_horizon_minutes, s.side AS sig_side, s.signal_close_utc AS sig_signal_close_utc"
            if {"book_id", "pair", "horizon_minutes", "side", "signal_close_utc"}.issubset(sig_cols)
            else "NULL AS sig_book_id, NULL AS sig_pair, NULL AS sig_horizon_minutes, NULL AS sig_side, NULL AS sig_signal_close_utc"
        )
        rows = conn.execute(
            f"""
            SELECT c.contract_id, c.signal_id, c.book_id, c.pair, c.horizon_minutes, c.side,
                   c.buy_price_cents, c.payout_cents, c.sell_price_cents, c.profit_cents,
                   c.status, c.opened_utc, c.closed_utc,
                   {_col('account_mode')}, {_col('account_id')}, {_col('book_content_id')},
                   {_col('contract_family')}, {_col('is_migrated_legacy')}, {_col('cost_unknown')},
                   {_col('stake_authoritative')}, {_col('venue_start_epoch')}, {_col('venue_end_epoch')},
                   {_col('venue_terminal_status')}, {_col('venue_underlying_symbol')},
                   {_col('venue_contract_type')}, {_col('signal_close_utc')},
                   {sig_join_fields}
            FROM sota_contracts c
            LEFT JOIN sota_signals s ON s.signal_id = c.signal_id
            ORDER BY c.opened_utc ASC, c.contract_id ASC
            """
        ).fetchall()
        sig_rows = conn.execute(
            """
            SELECT signal_id, book_id, status FROM sota_signals
            WHERE status IN ('buy_intent', 'buy_unknown')
            """
        ).fetchall()
        risk_state: dict[str, Any] = {}
        if "sota_risk_state" in risk_tables:
            for rr in conn.execute("SELECT key, value_json FROM sota_risk_state").fetchall():
                try:
                    risk_state[str(rr["key"])] = json.loads(str(rr["value_json"]))
                except Exception:
                    risk_state[str(rr["key"])] = None
        conn.execute("COMMIT")
    finally:
        conn.close()

    all_contracts = [dict(r) for r in rows]
    unresolved_sigs = [dict(r) for r in sig_rows]

    # ------------------------------------------------------------------
    # Strictly Prospective G6 Mode (spec §3.5)
    # ------------------------------------------------------------------
    if prospective_mode:
        eval_dt = _parse_iso_utc(eval_now_utc) if eval_now_utc is not None else datetime.now(timezone.utc)
        blockers_top: list[str] = []
        if eval_dt is None:
            blockers_top.append("invalid_eval_now_utc")
        if user_ver != 3:
            blockers_top.append(f"queue_not_v3:user_version={user_ver}")
        if enforce_canonical_db:
            canon = canonical_db_path if canonical_db_path is not None else (
                (repo_root or REPO_ROOT) / "deriv_data" / "runtime" / "sota_trade_queue_demo.sqlite"
            )
            try:
                if db_p.resolve(strict=True) != canon.resolve():
                    blockers_top.append("non_canonical_db_diagnostic_only")
            except OSError:
                blockers_top.append("non_canonical_db_diagnostic_only")

        if not book_id:
            blockers_top.append("missing_book_id")
            return {
                "schema": schema,
                "db_path": str(db_p),
                "book_id": None,
                "promotion_eligible_books": [],
                "promotion_eligible": False,
                "reasons": blockers_top,
                "by_book": {},
            }

        # Validate expected_stake_usd == 1.00 (100 exact cents)
        expected_stake_cents = 100
        if expected_stake_usd is not None:
            try:
                sf = Fraction(str(expected_stake_usd).strip())
                if sf * 100 != 100:
                    blockers_top.append(f"invalid_expected_stake_usd:{expected_stake_usd}")
            except Exception:
                blockers_top.append(f"invalid_expected_stake_usd:{expected_stake_usd}")

        cohort_ep = _parse_whole_second_epoch(cohort_start_utc)
        if cohort_ep is None:
            blockers_top.append("invalid_or_missing_cohort_start_utc")
        eval_ep = int(eval_dt.timestamp()) if eval_dt is not None else None

        reg_ent = registered_books.get(book_id)
        reg_cid = registered_cids.get(book_id)
        if require_registered_book and not reg_ent:
            blockers_top.append("unregistered_book_id")
        want_cid = reg_cid or expected_content_id
        if reg_cid and expected_content_id and expected_content_id != reg_cid:
            blockers_top.append(f"expected_content_id_conflicts_with_index:{expected_content_id}!={reg_cid}")
        if not want_cid:
            blockers_top.append("missing_expected_content_id")

        binding = risk_state.get("queue_account_binding")
        acct_id = expected_account_id or (binding.get("account_id") if isinstance(binding, dict) else None)
        if not isinstance(binding, dict) or not acct_id or binding.get("account_id") != acct_id or str(binding.get("account_mode") or "").lower() != "demo":
            blockers_top.append("queue_account_binding_missing_or_mismatch")

        # Validate stored prospective cohort anchor
        anchor_key = f"g6_cohort_anchor:{acct_id}:{book_id}:{want_cid}" if (acct_id and want_cid) else ""
        anchor = risk_state.get(anchor_key) if anchor_key else None
        if not isinstance(anchor, dict):
            blockers_top.append("missing_stored_cohort_anchor")
        else:
            reg_ep = _parse_whole_second_epoch(anchor.get("registered_utc"))
            anc_start_ep = _parse_whole_second_epoch(anchor.get("cohort_start_utc"))
            if reg_ep is None or anc_start_ep is None:
                blockers_top.append("invalid_anchor_clocks")
            elif cohort_ep is not None and anc_start_ep != cohort_ep:
                blockers_top.append("cohort_start_utc_mismatch_vs_anchor")
            elif eval_ep is not None and not (reg_ep < anc_start_ep <= eval_ep):
                blockers_top.append("anchor_clock_ordering_invalid")
            if anchor.get("required_stake_cents") != expected_stake_cents:
                blockers_top.append("anchor_required_stake_cents_mismatch")
            if anchor.get("account_id") != acct_id or anchor.get("book_id") != book_id or anchor.get("book_content_id") != want_cid:
                blockers_top.append("anchor_identity_mismatch")

        # Expected manifest spec for contract-spec verification
        exp_pair = None
        exp_horizon = None
        exp_sides: set[str] | None = None
        exp_family = None
        if reg_ent:
            man = reg_ent.get("_manifest") or reg_ent
            strat = reg_ent.get("_strategy") or {}
            exp_pair = str(man.get("pair") or "").upper() or None
            exp_horizon = int(man["horizon_minutes"]) if man.get("horizon_minutes") is not None else None
            exp_sides = {str(s).upper() for s in (man.get("allowed_sides") or [])} or None
            exp_family = str(strat.get("contract_family", "risefall"))
            if isinstance(anchor, dict):
                if (
                    str(anchor.get("pair") or "").upper() != exp_pair
                    or anchor.get("horizon_minutes") != exp_horizon
                    or (exp_sides is not None and str(anchor.get("side") or "").upper() not in exp_sides)
                    or str(anchor.get("contract_family") or "") != exp_family
                ):
                    blockers_top.append("anchor_spec_mismatch_vs_manifest")

        b_contracts = [c for c in all_contracts if str(c["book_id"]) == book_id]
        b_uncertain = [s for s in unresolved_sigs if str(s["book_id"]) == book_id]
        if b_uncertain:
            blockers_top.append(f"unresolved_buy_signals:{len(b_uncertain)}")

        effective_min = max(PROMOTION_HARD_MIN_TRADES, int(min_settled_for_promotion))
        if int(min_settled_for_promotion) < PROMOTION_HARD_MIN_TRADES:
            blockers_top.append(f"min_trades_arg_below_hard_floor:{min_settled_for_promotion}<{PROMOTION_HARD_MIN_TRADES}")

        cohort_rows: list[dict[str, Any]] = []
        excluded_pre_anchor = 0
        excluded_post_eval = 0
        seen_cids: set[str] = set()
        seen_sids: set[str] = set()
        anomalies: list[str] = []

        for c in b_contracts:
            cid = str(c["contract_id"])
            sid = str(c["signal_id"])
            st = str(c["status"] or "")
            is_mig = int(c["is_migrated_legacy"] or 0)
            v_start = c["venue_start_epoch"]
            v_end = c["venue_end_epoch"]
            closed_dt = _parse_iso_utc(c["closed_utc"])

            # Validate clocks & joined signal identity/clocks BEFORE pre-anchor exclusion (BLOCKER #5)!
            if is_mig == 1 and v_start is None:
                if closed_dt is None:
                    anomalies.append(f"{cid}:migrated_missing_closed_utc")
                    continue
                if cohort_ep is not None and int(closed_dt.timestamp()) < cohort_ep:
                    excluded_pre_anchor += 1
                    continue
                anomalies.append(f"{cid}:migrated_row_not_pre_anchor")
                continue

            if v_start is None or isinstance(v_start, bool) or not isinstance(v_start, int) or v_start <= 0:
                anomalies.append(f"{cid}:missing_or_invalid_venue_start_epoch")
                continue
            if st in G6_ELIGIBLE_SETTLED_STATUSES and (v_end is None or isinstance(v_end, bool) or not isinstance(v_end, int) or v_end < v_start):
                anomalies.append(f"{cid}:missing_or_invalid_venue_end_epoch")
                continue
            if v_end is not None and (isinstance(v_end, bool) or not isinstance(v_end, int) or v_end < v_start):
                anomalies.append(f"{cid}:invalid_venue_end_epoch")
                continue
            if c["sig_book_id"] is None or str(c["sig_book_id"]) != str(c["book_id"]):
                anomalies.append(f"{cid}:linked_signal_book_id_mismatch")
                continue
            c_sc_raw = c["signal_close_utc"]
            s_sc_raw = c["sig_signal_close_utc"]
            if not c_sc_raw or not s_sc_raw or str(c_sc_raw) != str(s_sc_raw) or _parse_iso_utc(c_sc_raw) is None:
                anomalies.append(f"{cid}:signal_close_utc_mismatch_or_invalid")
                continue

            # Pre-anchor exclusion (including boundary-crossing contracts started < cohort_start_epoch!)
            if cohort_ep is not None and v_start < cohort_ep:
                excluded_pre_anchor += 1
                continue

            # Non-excluded contract in cohort window [cohort_start_epoch, ...)
            if st not in G6_ELIGIBLE_SETTLED_STATUSES:
                anomalies.append(f"{cid}:non_settled_or_ineligible_status:{st}")
                continue
            if closed_dt is None:
                anomalies.append(f"{cid}:missing_or_invalid_closed_utc")
                continue
            if eval_dt is not None and closed_dt > eval_dt:
                excluded_post_eval += 1
                continue

            if is_mig != 0:
                anomalies.append(f"{cid}:migrated_legacy_in_cohort")
                continue
            if cid in seen_cids or sid in seen_sids:
                anomalies.append(f"{cid}:duplicate_contract_or_signal")
                continue
            seen_cids.add(cid)
            seen_sids.add(sid)

            # Full provenance, economics & contract-spec verification!
            if c["cost_unknown"] != 0:
                anomalies.append(f"{cid}:cost_unknown")
                continue
            if c["stake_authoritative"] != 1:
                anomalies.append(f"{cid}:stake_not_authoritative")
                continue
            if c["payout_cents"] is None or int(c["payout_cents"]) <= 0:
                anomalies.append(f"{cid}:missing_or_nonpositive_payout_cents")
                continue
            bp = c["buy_price_cents"]
            sp_c = c["sell_price_cents"]
            pf = c["profit_cents"]
            if bp != expected_stake_cents or sp_c is None or pf is None or (int(sp_c) - int(bp) != int(pf)):
                anomalies.append(f"{cid}:economics_mismatch")
                continue
            if c["account_id"] != acct_id or str(c["account_mode"] or "").lower() != "demo":
                anomalies.append(f"{cid}:account_mismatch")
                continue
            if want_cid and c["book_content_id"] != want_cid:
                anomalies.append(f"{cid}:book_content_id_mismatch")
                continue
            if v_end is None or isinstance(v_end, bool) or not isinstance(v_end, int) or v_end < v_start:
                anomalies.append(f"{cid}:invalid_venue_end_epoch")
                continue
            hm = int(c["horizon_minutes"])
            if (v_end - v_start) != hm * 60:
                anomalies.append(f"{cid}:venue_duration_mismatch:{v_end - v_start}!={hm * 60}")
                continue
            if str(c["venue_terminal_status"] or "").lower() != st.lower():
                anomalies.append(f"{cid}:venue_terminal_status_mismatch")
                continue

            c_pair = str(c["pair"] or "").upper()
            c_side = str(c["side"] or "").upper()
            c_fam = str(c["contract_family"] or "")
            # Linked signal agreement
            if (
                str(c["sig_pair"] or "").upper() != c_pair
                or c["sig_horizon_minutes"] != hm
                or str(c["sig_side"] or "").upper() != c_side
            ):
                anomalies.append(f"{cid}:linked_signal_spec_mismatch")
                continue
            # Manifest agreement
            if (
                (exp_pair is not None and c_pair != exp_pair)
                or (exp_horizon is not None and hm != exp_horizon)
                or (exp_sides is not None and c_side not in exp_sides)
                or (exp_family is not None and c_fam != exp_family)
            ):
                anomalies.append(f"{cid}:manifest_spec_mismatch")
                continue
            # Venue symbol & contract_type agreement
            want_sym = SOTA_DERIV_SYMBOLS.get(c_pair)
            try:
                want_ctype = contract_type_for(c_side, family=c_fam)
            except Exception:
                want_ctype = None
            if c["venue_underlying_symbol"] != want_sym or str(c["venue_contract_type"] or "").upper() != want_ctype:
                anomalies.append(f"{cid}:venue_symbol_or_contract_type_mismatch")
                continue
            sig_close_dt = _parse_iso_utc(c["signal_close_utc"] or c["sig_signal_close_utc"])
            if sig_close_dt is None:
                anomalies.append(f"{cid}:invalid_signal_close_utc")
                continue

            cohort_rows.append(
                {
                    "contract_id": cid,
                    "v_start": v_start,
                    "v_end": v_end,
                    "sig_close_ep": int(sig_close_dt.timestamp()),
                    "horizon_s": hm * 60,
                    "buy_price_cents": int(bp),
                    "payout_cents": int(c["payout_cents"]),
                    "profit_cents": int(pf),
                }
            )

        # Closed-interval overlap & signal-gap < horizon_minutes * 60 check
        cohort_rows.sort(key=lambda r: (r["v_start"], r["v_end"], r["contract_id"]))
        for a, b in zip(cohort_rows, cohort_rows[1:]):
            if b["v_start"] <= a["v_end"]:
                anomalies.append(f"closed_interval_overlap:{a['contract_id']}:{b['contract_id']}")
        by_sig = sorted(cohort_rows, key=lambda r: (r["sig_close_ep"], r["contract_id"]))
        for a, b in zip(by_sig, by_sig[1:]):
            if (b["sig_close_ep"] - a["sig_close_ep"]) < a["horizon_s"]:
                anomalies.append(f"signal_gap_below_horizon:{a['contract_id']}:{b['contract_id']}")

        blockers = list(blockers_top)
        blockers.extend(anomalies)
        n_coh = len(cohort_rows)
        if n_coh < effective_min:
            blockers.append(f"insufficient_prospective_trades:{n_coh}<{effective_min}")

        ci_lo_str = None
        ci_hi_str = None
        ci_lo_f = None
        ci_hi_f = None
        roi_f = None
        wins = sum(1 for r in cohort_rows if r["profit_cents"] > 0)
        stake_sum = sum(r["buy_price_cents"] for r in cohort_rows)
        profit_sum = sum(r["profit_cents"] for r in cohort_rows)
        if n_coh > 0:
            roi_f = float(Fraction(profit_sum, stake_sum))
            ci_lo_frac, ci_hi_frac = exact_fraction_bootstrap_ci95(
                [r["profit_cents"] for r in cohort_rows], stake_cents=expected_stake_cents, n_boot=5000, seed=7
            )
            ci_lo_str = f"{ci_lo_frac.numerator}/{ci_lo_frac.denominator}"
            ci_hi_str = f"{ci_hi_frac.numerator}/{ci_hi_frac.denominator}"
            ci_lo_f = float(ci_lo_frac)
            ci_hi_f = float(ci_hi_frac)
            if ci_lo_frac <= 0:
                blockers.append("ev_ci95_lo_not_positive")

        ok = len(blockers) == 0
        b_summary = {
            "n": n_coh,
            "wins": wins,
            "win_rate": round(wins / n_coh, 4) if n_coh > 0 else None,
            "excluded_pre_anchor": excluded_pre_anchor,
            "excluded_post_eval": excluded_post_eval,
            "stake_usd": round(stake_sum / 100.0, 2),
            "net_profit_usd": round(profit_sum / 100.0, 2),
            "realized_ev_per_stake": round(roi_f, 6) if roi_f is not None else None,
            "ev_ci95": [round(ci_lo_f, 6) if ci_lo_f is not None else None, round(ci_hi_f, 6) if ci_hi_f is not None else None],
            "ev_ci95_exact_fraction": [ci_lo_str, ci_hi_str],
            "promotion_eligible": ok,
            "reasons": blockers,
        }
        return {
            "schema": schema,
            "db_path": str(db_p),
            "book_id": book_id,
            "cohort_start_utc": cohort_start_utc,
            "eval_now_utc": eval_dt.isoformat() if eval_dt is not None else eval_now_utc,
            "promotion_scope": "single_book_prospective_g6",
            "settled_contracts": n_coh,
            "open_contracts": sum(1 for c in b_contracts if c["status"] == "open"),
            "unresolved_signals": len(b_uncertain),
            "by_book": {book_id: b_summary},
            "promotion_eligible_books": [book_id] if ok else [],
            "promotion_eligible": ok,
            "reasons": blockers,
        }

    # ------------------------------------------------------------------
    # Backwards-compatible v2 summary path
    # ------------------------------------------------------------------
    settled_all = [x for x in all_contracts if x["status"] in SETTLED_STATUSES]
    open_all = [x for x in all_contracts if x["status"] == "open"]
    observed_books = sorted(
        {str(x["book_id"]) for x in all_contracts}
        | {str(x["book_id"]) for x in unresolved_sigs}
        | ({book_id} if book_id else set())
    )

    def _agg_book(bid: str) -> dict[str, Any]:
        b_settled = [x for x in settled_all if str(x["book_id"]) == bid]
        b_open = [x for x in open_all if str(x["book_id"]) == bid]
        b_unknown_c = [x for x in all_contracts if str(x["book_id"]) == bid and x["status"] == "unknown"]
        b_uncertain_s = [x for x in unresolved_sigs if str(x["book_id"]) == bid]

        blockers: list[str] = []
        if b_open:
            blockers.append(f"open_contracts:{len(b_open)}")
        if b_unknown_c:
            blockers.append(f"unknown_contracts:{len(b_unknown_c)}")
        if b_uncertain_s:
            blockers.append(f"unresolved_buy_signals:{len(b_uncertain_s)}")
        effective_min = max(PROMOTION_HARD_MIN_TRADES, int(min_settled_for_promotion))
        if int(min_settled_for_promotion) < PROMOTION_HARD_MIN_TRADES:
            blockers.append(f"min_trades_arg_below_hard_floor:{min_settled_for_promotion}<{PROMOTION_HARD_MIN_TRADES}")
        if len(b_settled) < effective_min:
            blockers.append(f"insufficient_settled_trades:{len(b_settled)}<{effective_min}")
        non_demo = [x for x in b_settled if str(x.get("account_mode") or "").strip().lower() != "demo"]
        if non_demo:
            blockers.append(f"non_demo_provenance_rows:{len(non_demo)}")
        missing_cid = [x for x in b_settled if not str(x.get("book_content_id") or "").strip()]
        if missing_cid:
            blockers.append(f"missing_book_content_id_rows:{len(missing_cid)}")
        cids_seen = sorted({str(x["book_content_id"]).strip() for x in b_settled if str(x.get("book_content_id") or "").strip()})
        if len(cids_seen) > 1:
            blockers.append(f"mixed_book_content_ids:{cids_seen}")
        reg_cid = registered_cids.get(bid)
        if require_registered_book and not reg_cid:
            blockers.append("unregistered_book_id")
        want_cid = reg_cid if reg_cid else (expected_content_id if (book_id == bid and expected_content_id) else None)
        if reg_cid and expected_content_id and book_id == bid and expected_content_id != reg_cid:
            blockers.append(f"expected_content_id_conflicts_with_index:{expected_content_id}!={reg_cid}")
        if want_cid and cids_seen and cids_seen != [want_cid]:
            blockers.append(f"book_content_id_mismatch:{cids_seen}!={want_cid}")

        if not b_settled:
            return {
                "n": 0,
                "open_contracts": len(b_open),
                "unresolved_signals": len(b_uncertain_s),
                "promotion_eligible": False,
                "reasons": blockers,
            }

        bad_bp = [x for x in b_settled if x["buy_price_cents"] is None or int(x["buy_price_cents"]) <= 0]
        if bad_bp:
            blockers.append(f"invalid_nonpositive_buy_price:{len(bad_bp)}")
        valid_settled = [x for x in b_settled if x["buy_price_cents"] is not None and int(x["buy_price_cents"]) > 0]
        if not valid_settled:
            return {
                "n": len(b_settled),
                "open_contracts": len(b_open),
                "unresolved_signals": len(b_uncertain_s),
                "promotion_eligible": False,
                "reasons": blockers,
            }

        stake_cents = sum(int(x["buy_price_cents"]) for x in valid_settled)
        profit_cents = sum(int(x["profit_cents"] or 0) for x in valid_settled)
        wins = sum(1 for x in valid_settled if int(x["profit_cents"] or 0) > 0)
        unit_returns = np.array(
            [float(Fraction(int(x["profit_cents"] or 0), int(x["buy_price_cents"]))) for x in valid_settled],
            dtype=np.float64,
        )
        ci_lo, ci_hi = _bootstrap_ci95_float(unit_returns)
        if not (np.isfinite(ci_lo) and ci_lo > 0.0):
            blockers.append("ev_ci95_lo_not_positive")

        roi_exact = Fraction(profit_cents, stake_cents) if stake_cents > 0 else Fraction(0, 1)
        fixed_pay_rows = [x for x in valid_settled if x["payout_cents"] is not None and int(x["payout_cents"]) > 0]
        mean_pay = (
            round(float(np.mean([int(x["payout_cents"]) / int(x["buy_price_cents"]) for x in fixed_pay_rows])), 4)
            if fixed_pay_rows
            else None
        )
        ok = len(blockers) == 0
        return {
            "n": len(b_settled),
            "wins": wins,
            "win_rate": round(wins / len(valid_settled), 4),
            "open_contracts": len(b_open),
            "unknown_contracts": len(b_unknown_c),
            "unresolved_signals": len(b_uncertain_s),
            "stake_usd": round(stake_cents / 100.0, 2),
            "net_profit_usd": round(profit_cents / 100.0, 2),
            "mean_payout_ratio": mean_pay,
            "realized_ev_per_stake": round(float(roi_exact), 4),
            "ev_ci95": [round(ci_lo, 4) if np.isfinite(ci_lo) else None, round(ci_hi, 4) if np.isfinite(ci_hi) else None],
            "promotion_eligible": bool(ok),
            "reasons": blockers,
        }

    by_book = {bid: _agg_book(bid) for bid in observed_books}
    eligible_books = [bid for bid, stats in by_book.items() if stats.get("promotion_eligible") is True]
    total_stake_cents = sum(int(x["buy_price_cents"] or 0) for x in settled_all)
    total_profit_cents = sum(int(x["profit_cents"] or 0) for x in settled_all)

    target_eligible: bool | None = None
    reasons: list[str] = []
    if book_id is not None:
        b_res = by_book.get(book_id, {"promotion_eligible": False, "reasons": ["book_not_observed"]})
        target_eligible = bool(b_res.get("promotion_eligible", False))
        reasons = list(b_res.get("reasons", []))

    return {
        "schema": schema,
        "db_path": str(db_p),
        "book_id": book_id,
        "promotion_scope": "single_book" if book_id else "per_book_only_never_pooled",
        "settled_contracts": len(settled_all),
        "open_contracts": len(open_all),
        "unresolved_signals": len(unresolved_sigs),
        "portfolio_accounting_totals": {
            "stake_usd": round(total_stake_cents / 100.0, 2),
            "net_profit_usd": round(total_profit_cents / 100.0, 2),
            "note": "Accounting sum only; win rates and bootstrap CIs are never pooled across books/timeframes.",
        },
        "by_book": by_book,
        "promotion_eligible_books": eligible_books,
        "promotion_eligible": target_eligible,
        "reasons": reasons,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=None, help="Diagnostic alias; non-canonical path always yields promotion_eligible=False in prospective mode")
    ap.add_argument("--book-id", default=None, help="Evaluate promotion gate for one specific SOTA book_id")
    ap.add_argument("--cohort-start-utc", default=None, help="Prospective G6 cohort start timestamp (RFC3339 whole-second UTC)")
    ap.add_argument("--expected-account-id", default=None)
    ap.add_argument("--expected-stake-usd", default=None)
    ap.add_argument("--eval-now-utc", default=None)
    ap.add_argument("--min-trades", type=int, default=50)
    args = ap.parse_args(argv)
    canon_db = REPO_ROOT / "deriv_data" / "runtime" / "sota_trade_queue_demo.sqlite"
    db_path = Path(args.db) if args.db else canon_db
    # BLOCKER #5: CLI ALWAYS evaluates strictly in canonical prospective G6 mode (never promoting retrospective/non-canonical DBs!)
    report = summarize_settled_queue(
        db_path,
        min_settled_for_promotion=args.min_trades,
        book_id=args.book_id,
        repo_root=REPO_ROOT,
        require_registered_book=True,
        cohort_start_utc=args.cohort_start_utc,
        expected_account_id=args.expected_account_id,
        expected_stake_usd=args.expected_stake_usd,
        eval_now_utc=args.eval_now_utc,
        canonical_db_path=canon_db,
        enforce_canonical_db=True,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
