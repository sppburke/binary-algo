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

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT c.contract_id, c.book_id, c.pair, c.horizon_minutes, c.side,
                   c.buy_price_cents, c.payout_cents, c.sell_price_cents, c.profit_cents,
                   c.status, c.opened_utc, c.closed_utc
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
        if len(b_settled) < min_settled_for_promotion:
            blockers.append(f"insufficient_settled_trades:{len(b_settled)}<{min_settled_for_promotion}")

        if not b_settled:
            return {
                "n": 0,
                "open_contracts": len(b_open),
                "unresolved_signals": len(b_uncertain_s),
                "promotion_eligible": False,
                "reasons": blockers,
            }

        stake_cents = sum(int(x["buy_price_cents"]) for x in b_settled)
        profit_cents = sum(int(x["profit_cents"]) for x in b_settled)
        wins = sum(1 for x in b_settled if int(x["profit_cents"]) > 0)
        unit_returns = np.array(
            [float(Fraction(int(x["profit_cents"]), int(x["buy_price_cents"]))) for x in b_settled],
            dtype=np.float64,
        )
        ci_lo, ci_hi = _bootstrap_ci95(unit_returns)
        if not (np.isfinite(ci_lo) and ci_lo > 0.0):
            blockers.append("ev_ci95_lo_not_positive")

        roi_exact = Fraction(profit_cents, stake_cents) if stake_cents > 0 else Fraction(0, 1)
        mean_pay = float(
            np.mean([int(x["payout_cents"]) / int(x["buy_price_cents"]) for x in b_settled])
        )
        ok = len(blockers) == 0
        return {
            "n": len(b_settled),
            "wins": wins,
            "win_rate": round(wins / len(b_settled), 4),
            "open_contracts": len(b_open),
            "unknown_contracts": len(b_unknown_c),
            "unresolved_signals": len(b_uncertain_s),
            "stake_usd": round(stake_cents / 100.0, 2),
            "net_profit_usd": round(profit_cents / 100.0, 2),
            "mean_payout_ratio": round(mean_pay, 4),
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
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
