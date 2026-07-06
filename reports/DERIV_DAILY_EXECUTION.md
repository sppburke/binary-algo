> **SCOPE: GENERIC operational report** (Deriv demo execution day-over-day). This is live-execution bookkeeping, not model certification evidence.

# Deriv Daily Execution Report

Purpose: track demo trading outcomes by NY trading date so runtime/code-policy changes can be compared against the next session's realized execution.

## Daily Summary

| NY date | Runtime policy bucket | Code boundary | Settled W-L | Win % | Demo P&L | ROI on stake | Open at report | Non-contract skips | Source |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| 2026-07-06 | Pre-allocator buys; post-allocator reconciliation only | Last buy `2026-07-06T18:59:05Z`; allocator executor startup `2026-07-06T20:45:34Z` after the 16:35 NY last-start cutoff | 84-88 | 48.837% | -25.94 | -15.081% | 0 | 34 | VPS query `2026-07-06T21:34:44Z` |

## Pair Results

| NY date | Pair | Side(s) traded | Won | Lost | Settled | Win % | Demo P&L | ROI on stake | Open |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 2026-07-06 | AUDUSD | DOWN | 33 | 50 | 83 | 39.759% | -25.15 | -30.301% | 0 |
| 2026-07-06 | GBPUSD | DOWN | 8 | 7 | 15 | 53.333% | -1.28 | -8.533% | 0 |
| 2026-07-06 | USDCAD | UP | 17 | 9 | 26 | 65.385% | 3.44 | 13.231% | 0 |
| 2026-07-06 | USDJPY | UP | 26 | 22 | 48 | 54.167% | -2.95 | -6.146% | 0 |

## Execution Context

### 2026-07-06

- Classification: pre-allocator baseline for buys. The issue-#7 allocator/reconciliation code was deployed after the last buy and after the configured 16:35 NY last-start cutoff, so this row should be compared against the next NY session as "before allocator".
- Settlement state at report time: all contracts settled; no open pair-side exposure remained.
- Non-contract rows: `terminal_skip=33` (`edge_not_positive`), `signal_expired=1`.
- Deployed allocator config observed in executor startup after the trading window: `batch_size=12`, `same_pair_min_gap_seconds=900`, `max_usd_factor_open=2`, `disabled_pair_sides=[AUDUSD:DOWN]`, `allocation_arbitration_ms=250`.
- Reconciliation config observed in executor startup: `batch_size=12`, `interval_seconds=5.0`, `contract_expiry_grace_seconds=600.0`.

## Source Evidence

Source was the VPS live queue DB and executor JSONL, queried from `/home/sean/git/binary-algo` on host `64.177.80.63` at `2026-07-06T21:34:44Z`. VPS repo HEAD for the runtime source at query time: `c57512a`.

```json
{
  "target_ny_date": "2026-07-06",
  "source": {
    "host": "64.177.80.63",
    "repo": "/home/sean/git/binary-algo",
    "git_head": "c57512a",
    "queue_db": "deriv_data/runtime/trade_queue.sqlite",
    "executor_log": "logs/paper_trades/trade_executor/2026-07-06.jsonl"
  },
  "totals": {
    "won": 84,
    "lost": 88,
    "open": 0,
    "settled": 172,
    "win_pct": 48.837,
    "profit": -25.94,
    "stake": 172.0,
    "roi_pct": -15.081
  },
  "by_pair": [
    {"key": "AUDUSD", "won": 33, "lost": 50, "settled": 83, "win_pct": 39.759, "profit": -25.15, "stake": 83.0, "roi_pct": -30.301, "open": 0},
    {"key": "GBPUSD", "won": 8, "lost": 7, "settled": 15, "win_pct": 53.333, "profit": -1.28, "stake": 15.0, "roi_pct": -8.533, "open": 0},
    {"key": "USDCAD", "won": 17, "lost": 9, "settled": 26, "win_pct": 65.385, "profit": 3.44, "stake": 26.0, "roi_pct": 13.231, "open": 0},
    {"key": "USDJPY", "won": 26, "lost": 22, "settled": 48, "win_pct": 54.167, "profit": -2.95, "stake": 48.0, "roi_pct": -6.146, "open": 0}
  ],
  "non_contract_signal_status_counts": {
    "signal_expired": 1,
    "terminal_skip": 33
  },
  "terminal_reason_counts": {
    "edge_not_positive": 33,
    "signal_expired": 1
  },
  "contract_status_counts": {
    "lost": 88,
    "won": 84
  },
  "executor_event_counts_subset": {
    "buy_confirmed": 172,
    "contract_closed": 172,
    "allocation_reserved": 0,
    "allocation_skipped": 0,
    "reconcile_cycle": 15
  },
  "last_buy_confirmed_utc": "2026-07-06T18:59:05.494867+00:00",
  "last_executor_startup_utc": "2026-07-06T20:45:34.290367+00:00",
  "last_reconcile_cycle": {
    "timestamp_utc": "2026-07-06T20:47:12.694058+00:00",
    "requested": 4,
    "calls": 4,
    "unresolved": 0,
    "batch_size": 12,
    "interval_seconds": 5.0
  }
}
```

## Next Row Template

| NY date | Runtime policy bucket | Code boundary | Settled W-L | Win % | Demo P&L | ROI on stake | Open at report | Non-contract skips | Source |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| YYYY-MM-DD | Post-allocator session | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
