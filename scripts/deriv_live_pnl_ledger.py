"""Per-book exact-cent VENUE_SETTLED P&L ledger and Demo -> Live promotion gate.

Sibling to scripts/deriv_economics_ledger_v2.py (which stays byte-frozen for ROS-3A seals).
Computes exact Fraction cent returns per `book_id` in the SOTA SQLite queue (never pooling
win rates or bootstrap CIs across books or timeframes, per AGENTS.md) and enforces the
mathematical promotion gate (`n >= 50` settled contracts, two-sided 95% bootstrap CI lower
bound `ev_ci95[0] > 0` [97.5% one-sided confidence], and zero `open`/`unknown`/`buy_unknown`
records) before authorizing Real-Money (`CR`) promotion for a specific book.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

SETTLED_STATUSES = frozenset({"won", "lost", "sold", "expired", "cancelled"})
PROMOTION_HARD_MIN_TRADES = 50
REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_registered_content_ids(repo_root: Path | None) -> dict[str, str]:
    if repo_root is None:
        return {}
    idx_path = repo_root / "books" / "SOTA_INDEX.json"
    if not idx_path.is_file():
        return {}
    try:
        obj = json.loads(idx_path.read_text(encoding="utf-8"))
        return {
            str(b["id"]): str(b["content_id"])
            for b in (obj.get("books") or [])
            if isinstance(b, dict) and b.get("id") and b.get("content_id")
        }
    except Exception:
        return {}


def _bootstrap_ci95(values: np.ndarray, n_boot: int = 5000, seed: int = 7) -> tuple[float, float]:
    if len(values) < 5:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    n = len(values)
    means = np.array([values[rng.integers(0, n, n)].mean() for _ in range(n_boot)], dtype=np.float64)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def summarize_settled_queue(
    db_path: Path,
    *,
    min_settled_for_promotion: int = 50,
    book_id: str | None = None,
    expected_content_id: str | None = None,
    repo_root: Path | None = REPO_ROOT,
    require_registered_book: bool = False,
) -> dict[str, Any]:
    if not db_path.exists():
        return {
            "schema": "deriv-sota-live-pnl-ledger/v2",
            "db_path": str(db_path),
            "book_id": book_id,
            "settled_contracts": 0,
            "open_contracts": 0,
            "unresolved_signals": 0,
            "promotion_eligible_books": [],
            "promotion_eligible": False if book_id else None,
            "reasons": ["queue_db_missing"],
            "by_book": {},
        }

    registered_cids = _load_registered_content_ids(repo_root)
    conn = sqlite3.connect(str(db_path), isolation_level=None)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("BEGIN")
        con_cols = {str(r["name"]) for r in conn.execute("PRAGMA table_info(sota_contracts)").fetchall()}
        am_expr = "c.account_mode" if "account_mode" in con_cols else "NULL AS account_mode"
        cid_expr = "c.book_content_id" if "book_content_id" in con_cols else "NULL AS book_content_id"
        rows = conn.execute(
            f"""
            SELECT c.contract_id, c.book_id, c.pair, c.horizon_minutes, c.side,
                   c.buy_price_cents, c.payout_cents, c.sell_price_cents, c.profit_cents,
                   c.status, c.opened_utc, c.closed_utc, {am_expr}, {cid_expr}
            FROM sota_contracts c
            ORDER BY c.opened_utc ASC
            """
        ).fetchall()
        sig_rows = conn.execute(
            """
            SELECT book_id, status FROM sota_signals
            WHERE status IN ('buy_intent', 'buy_unknown')
            """
        ).fetchall()
        conn.execute("COMMIT")
    finally:
        conn.close()

    all_contracts = [dict(r) for r in rows]
    unresolved_sigs = [dict(r) for r in sig_rows]
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

        bad_bp = [x for x in b_settled if int(x["buy_price_cents"]) <= 0]
        if bad_bp:
            blockers.append(f"invalid_nonpositive_buy_price:{len(bad_bp)}")
        valid_settled = [x for x in b_settled if int(x["buy_price_cents"]) > 0]
        if not valid_settled:
            return {
                "n": len(b_settled),
                "open_contracts": len(b_open),
                "unresolved_signals": len(b_uncertain_s),
                "promotion_eligible": False,
                "reasons": blockers,
            }

        stake_cents = sum(int(x["buy_price_cents"]) for x in valid_settled)
        profit_cents = sum(int(x["profit_cents"]) for x in valid_settled)
        wins = sum(1 for x in valid_settled if int(x["profit_cents"]) > 0)
        unit_returns = np.array(
            [float(Fraction(int(x["profit_cents"]), int(x["buy_price_cents"]))) for x in valid_settled],
            dtype=np.float64,
        )
        ci_lo, ci_hi = _bootstrap_ci95(unit_returns)
        if not (np.isfinite(ci_lo) and ci_lo > 0.0):
            blockers.append("ev_ci95_lo_not_positive")

        roi_exact = Fraction(profit_cents, stake_cents) if stake_cents > 0 else Fraction(0, 1)
        fixed_pay_rows = [x for x in valid_settled if int(x["payout_cents"]) > 0]
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

    total_stake_cents = sum(int(x["buy_price_cents"]) for x in settled_all)
    total_profit_cents = sum(int(x["profit_cents"]) for x in settled_all)

    target_eligible: bool | None = None
    reasons: list[str] = []
    if book_id is not None:
        b_res = by_book.get(book_id, {"promotion_eligible": False, "reasons": ["book_not_observed"]})
        target_eligible = bool(b_res.get("promotion_eligible", False))
        reasons = list(b_res.get("reasons", []))

    return {
        "schema": "deriv-sota-live-pnl-ledger/v2",
        "db_path": str(db_path),
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="deriv_data/runtime/sota_trade_queue_demo.sqlite")
    ap.add_argument("--book-id", default=None, help="Evaluate promotion gate for one specific SOTA book_id")
    ap.add_argument("--min-trades", type=int, default=50)
    args = ap.parse_args()
    report = summarize_settled_queue(
        Path(args.db),
        min_settled_for_promotion=args.min_trades,
        book_id=args.book_id,
        repo_root=REPO_ROOT,
        require_registered_book=True,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
