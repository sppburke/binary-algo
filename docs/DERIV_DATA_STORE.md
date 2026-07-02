# Deriv Data Store

Deriv-native market data is stored under `deriv_data/`, which is gitignored.
This directory is intentionally separate from the existing Dukascopy-derived
feature store (`features/`, `features_of/`, `features_tick*/`) and from frozen
deployable books (`books/`).

## Backfill

Run from the repo root:

```bash
~/binary-algo-venv/bin/python scripts/deriv_backfill.py --pairs all --granularity 60
```

The default output path is:

```text
deriv_data/candles_1m/
```

Each pair writes:

- `<PAIR>.parquet` — Deriv OHLC candles with timestamp, epoch, pair, symbol,
  granularity, endpoint, open, high, low, close.
- `<PAIR>_metadata.json` — page count, row count, first/last timestamp, stop
  reason, and API errors if Deriv stops the backward walk.
- `<PAIR>_progress.json` — the latest page cursor and status, updated after
  each fetched page.
- `_pages/<PAIR>/*.parquet` — per-page checkpoint shards written immediately
  during long backfills.
- `_last_run_summary.json` — one summary for the most recent backfill command.

## Isolation Rules

- Do not write Deriv backfills into `features/`.
- Do not overwrite or mutate existing frozen books in `books/`.
- Do not put Deriv experiment models in the generic `models/` directory.
- Any Deriv-trained experimental artifact must be named with a `deriv_` prefix
  and written under `deriv_data/models/`.

The smallest Deriv candle bar requested by this repo is 60 seconds. Raw ticks
are a separate feed shape and are not treated as bars — the `ticks_1s` store
below captures them for the hot runtime's shifted lane and offline audit.

## Tick Store (`ticks_1s`) and Daemon Candle Store (issue #4 Phase 2)

`scripts/deriv_market_stream.py` (shadow-only: no scoring, no trading) writes
two additional gitignored stores:

- `deriv_data/ticks_1s/_pages/<PAIR>/*.parquet` — live 1-second tick shards
  (`epoch, quote, bid, ask`; quote = mid), one atomic shard per writer flush
  (default 10s), with `<PAIR>_progress.json` recording received/persisted
  counts and the last persisted epoch. `--compact` merges shards into
  `<PAIR>.parquet` (the Phase-5 offline-audit substrate). Warmup history ticks
  feed only the in-memory ring — the store holds live-received ticks, so
  `received == persisted` is the zero-loss gate check.
- `deriv_data/candles_1m_daemon/<PAIR>.parquet` — the daemon candle store,
  all seven pairs including EURUSD (the xpair books' live feature join reads
  every pair's closes, exactly like the production store), refreshed every 5s
  (Phase-0 probe (i): completed candles are fetchable ~0.5s after the
  boundary). Same schema and open-candle-drop discipline as
  the production store (`candles_frame`/`merge_existing` from
  `deriv_backfill.py`). **Store ownership:** the production one-shot timer
  keeps sole write ownership of `deriv_data/candles_1m/`; the daemon lane
  reads only its own store. A cross-store consistency report (last-30
  completed closes per pair) is logged every 5 minutes — divergence there is
  a store bug, tracked separately from decision parity.

The shifted aggregator emits 12-offset 60s bars (offsets {0,5,...,55}s;
validity = exactly 60 one-second ticks in `(end-60s, end]`, deduped
keep-last, never forward-filled) to the runtime only — shifted bars are not
persisted as a feature store. Per-run gate rollups append to
`deriv_market_stream_gate_result.json` (archived in `results/json/`); the
Phase-2 gate needs >=3 full NY sessions passing coverage >= 99.5%, window
rate >= 99%, writer lag p95 <= 30s / max <= 120s, zero ring-to-store loss.

## Experimental Deriv Models

Run from the repo root after the candle backfill exists:

```bash
~/binary-algo-venv/bin/python scripts/deriv_train.py --pairs all --horizon 15 --holdout-months 2
```

This script is deliberately experimental. It trains from `deriv_data/candles_1m/`
and writes only under:

```text
deriv_data/models/
```

Every model artifact and manifest uses a `deriv_` prefix and status
`EXPERIMENTAL_NOT_CERTIFIED`. The most recent two calendar months of available
Deriv data are held out by default and are not used for fitting or threshold
selection. Train/validation rows are also dropped when their full forward-label
horizon would cross into the next split. Evaluation is Deriv-faithful for
Rise/Fall settlement: exact ties are counted as losses. The manifest reports
both overlapping selected-decision metrics and first-come non-overlapping
selected-trade metrics for the configured expiry horizon.

The trainer refuses to run on capped `--max-pages` backfills, open-candle
backfills, still-running backfills, or output paths outside `deriv_data/`. If
Deriv stops the backward walk with an API boundary error after usable progress,
training requires the explicit `--allow-api-error-boundary` flag.

## Health

The default rolling-store health surface is:

```bash
~/binary-algo-venv/bin/python scripts/deriv_backfill.py health --out-dir deriv_data/candles_1m --pairs all
```

For `--pairs all`, health reports the six executor-enabled pairs: `USDJPY`, `USDCAD`, `AUDUSD`, `NZDUSD`, `USDCHF`, `GBPUSD`. `EURUSD` remains backfillable for data parity but is not executor-enabled until verified live `OF_*` columns exist.

The JSON includes `enabled_pairs`, per-pair `row_count`, `latest_completed_utc`, `stale_seconds`, `monotonic_utc`, `duplicate_timestamps`, `gap_count`, `open_candle_excluded`, `min_required_rows`, and `passes`. It also reports cross-pair `common_close_rows`, `latest_common_completed_utc`, `stale_seconds`, `min_required_rows`, and `passes` for `USDCHF` and `GBPUSD`; those common-close checks require all seven Deriv close series for parity, including EURUSD close data, but not EURUSD `OF_*` features.

Executor runs with `--store-dir` consume the same health contract before scoring store-backed feature rows.

**Refresh dependency:** health requires freshness within `--max-stale-seconds` (default 180s), so a scheduled executor must be preceded by a store refresh each run. The `ops/deriv-demo-executor.service` template does this via `ExecStartPre` (`deriv_backfill.py backfill --max-pages 2`, an idempotent warm-store top-up); a standalone deployment needs an equivalent refresh unit/timer or documented runbook step before proposal-only/demo-buy runs.
