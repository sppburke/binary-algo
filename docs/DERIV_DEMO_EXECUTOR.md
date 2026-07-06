> **SCOPE: GENERIC** (Deriv demo runtime / venue integration). Model numbers live in the pair ledgers and `books/INDEX.json`.

# Deriv Demo Executor

`scripts/deriv_demo_executor.py` runs the six deployable non-EURUSD 15m books against Deriv Options demo infrastructure:

```text
USDJPY, USDCAD, AUDUSD, NZDUSD, USDCHF, GBPUSD
```

`EURUSD` is disabled until a verified live provider can produce finite `OF_*` order-flow columns. `--pairs all-enabled` means the six-pair set above. `--pairs EURUSD --public-smoke` fails closed.

This is operational demo tooling. It does not certify Deriv-native edge, does not enable real-money execution, and does not replace the frozen-book registry.

## Issue #6 Producer/Executor Split

The current service path is split:

- `scripts/deriv_runtime_supervisor.py --queue-mode` is a producer. It holds no Deriv demo credentials, scores fresh selected-side book candidates, and commits them to `deriv_data/runtime/trade_queue.sqlite`.
- `scripts/deriv_trade_executor.py --run` is the only credentialed demo buyer. It owns the account-scoped process lock, auth health, queue claims, payout/buy gates, proposal fallback, buy submission, and contract rows.
- `scripts/deriv_trade_queue.py` is the SQLite queue of record. Unix-socket wake-ups are a latency hint only; polling/startup scans remain the correctness fallback.

Queue status values are separate from Deriv contract statuses. Signal rows use `queued`, `claimed`, `buy_intent`, `bought`, `contract_closed`, `terminal_skip`, `signal_expired`, `buy_blocked_auth`, and `buy_unknown`. Contract rows use Deriv lifecycle statuses (`open`, `sold`, `won`, `lost`, `expired`, `cancelled`, `unknown`). Every terminal skip records reason/source/code fields.

The selected-side natural key is `(pair, lane, offset_id, signal_close_utc)` and intentionally excludes both `side` and `book_id`; a selected-side flip or metadata change for one pair/bar cannot double-buy. `signal_id` keeps provenance with `sha256(pair|side|lane|offset_id|signal_close_utc|book_id)`.

The shared last-start cutoff now lives in `scripts/deriv_runtime_core.py`: reject `>= 16:35:00 America/New_York`. Source: `results/json/deriv_api_probe_result.json` proves rejection by `16:43:10 NY` but does not pin the last accepted boundary, so issue #6 uses a conservative default until a committed accepted-boundary capture supersedes it.

The shared buy lock root is `deriv_data/runtime/locks` by default (`DERIV_RUNTIME_LOCK_ROOT` may override). It is verified as non-symlink, current-user-owned, and `0700`. The raw account key is `deriv-demo-account:{DERIV_ACCOUNT_ID}`; callers must not pre-hash it because `ExecutorLock` hashes once for the filename.

The legacy `scripts/deriv_demo_executor.py --demo-buy` and `--auth-smoke` paths are hard-disabled before credential reads. Auth health moved to `scripts/deriv_trade_executor.py`; invalid/expired PAT/OTP is surfaced as `buy_blocked_auth` and does not trigger blind retry. The retained runbook item for PAT rotation is: stop `deriv-trade-executor.service`, re-mint/update `DERIV_PAT` in the private env file, run the executor auth-health/fake gate, then restart the executor.

Deriv demo WebSocket URLs from the OTP endpoint are single-use. The credentialed executor's async client therefore refreshes the OTP URL before each reconnect attempt; otherwise a previously-used URL replays as HTTP 401 while periodic auth health still passes with a newly-minted URL. Client connection events log only the redacted endpoint (`scheme://host/path`), never the raw URL/query token.

**Current closure status:** fake queue/executor gates write `deriv_trade_executor_gate_result.json` and archive under `results/json/`. Live NY-session demo-buy, live forced concurrency, and live forced fallback gates are still required before issue #6 can be closed.

## Book Model and Payout-Aware Edge Gate

Runtime loads **one combined direction book per pair** (not separate UP/DOWN artifacts): it averages the book's seed probabilities, selects `UP` when `p >= 0.5` else `DOWN`, and requests `CALL` or `PUT`. The registry's UP/DOWN numbers are **side-specific measured floors for the selected side** (refit-CPCV p10 conditional on the model selecting that side at the coverage gate) and drive payout gating.

At startup `scripts/deriv_floor_resolver.py` resolves, per enabled pair and fail-closed: active coverage, confidence threshold, side floors, and the settlement haircut with its evidentiary source (precedence: structured registry/manifest/strategy-deploy metadata → the pair's own validated ticksettle result JSON → a ledger-documented trusted proxy → `fallback_max_validated_fx_ticksettle`, recomputed at resolution time from `results/json/*_15m_ticksettle_result.json`, never a frozen constant). The xpair books (USDCHF/GBPUSD) resolve **cov1** thresholds+floors via `deploy_cov` in `books/INDEX.json` — the old silent `"0.02"` gates fallback is a startup failure now. Every resolution is logged as a `floor_resolution` JSONL event including the signed raw haircut, the applied absolute haircut, and the book freeze/vintage date.

For a candidate trade on the selected side:

```text
live_breakeven  = ask_price / payout        # fail closed on missing/non-positive either
effective_floor = side_refit_p10 - haircut_abs - payout_edge_margin   # margin default 0.005
buy only when    effective_floor > live_breakeven
```

`--absolute-breakeven-ceiling` (default 0.60, or `DERIV_DEMO_ABSOLUTE_BREAKEVEN_CEILING`) remains a secondary absolute ceiling; the side-floor edge gate is the authoritative guard. The old `--max-breakeven` / `DERIV_DEMO_MAX_BREAKEVEN` spelling is accepted only as a compatibility alias. Raw model `proba` is never used as a calibrated win probability. There is no `display_value`/stake fallback for `ask`.

**Refit-vintage disclosure:** floors are refit-CPCV **per-era** floors. The frozen vintages the executor loads decay forward (frozen-forward 2026 COMBINED: AUDUSD 0.5082, NZDUSD 0.5194; USDCAD DOWN 0.5258). The gate certifies the documented floor under the periodic-retrain deploy policy, not the frozen artifact's current forward win rate; this stays **demo-only** until a retrain cadence exists. Haircuts are per-book means; worst year/side cells are materially larger (e.g. NZDUSD 2024 DOWN −0.0388) — revisit worst-year haircuts and/or a larger margin before any real-money step.

## Quote Snapshots and API Call Budget

Default legacy mode (`--quote-snapshots all`) snapshots **both CALL and PUT quotes for every enabled pair each cycle** for observability, logged as `quote_snapshot` events. In issue-#6 queue mode the producer no longer waits on public selected-side proposals before enqueue; final payout gating moves to the authenticated executor. Cached authenticated proposal lanes are bounded by the pre-registered hot-proposal max buy-age and fall back to a fresh proposal when stale/missing/consumed.

**Since the issue-#5 cutover the standing `quote_snapshot` source is `scripts/deriv_quote_workers.py --audit`** (`ops/deriv-quote-audit.service`): a buy-incapable sampler that logs one row per (pair, side) lane every 30s under `${DERIV_DEMO_LOG_DIR}/quote_audit/`. Sampler rows carry `source: "quote_audit"`; executor rows have no `source` field (absence ⇒ executor). The Phase-5 enable bar admits each source per lane only with ≥ 3 NY sessions of coverage and takes the MAXIMUM of per-source medians (fail-closed; see the amended falsifier in `scripts/deriv_offset_audit.py`). A lane with no quote emits `live_breakeven: null` + `invalid_reason` — nulls can never enter a median.

Worst-case API call budget per ~60s cycle at all-enabled defaults: **12 snapshot proposals + up to 6 fresh buy-quote proposals + `proposal_open_contract` polling at `--monitor-interval-seconds` (default 2.0s) while a contract is open**. The budget is written to the `startup_config` event. The empirical answer to the rate-limit caveat below is in the Phase-0 probe findings: 432/432 requests succeeded at 3.6 req/s sustained (probe f); `call_deriv` retry/backoff handles transient errors but is not a rate-limit budget.

## Phase-0 API Probe Findings (issue #4)

Measured 2026-07-02 ~04:25–04:38 UTC (00:25–00:38 NY, **off-session**; market open) by `scripts/deriv_api_probe.py`, unauthenticated public endpoints, proposal-only. Evidence: `results/json/deriv_api_probe_result.json` (per-probe records + a `decisions` block consumed by Phases 1–5). Re-runs merge per probe; the JSON is the citable artifact, not this prose.

- **(a) Candle floor:** `granularity=30` rejected on **both** endpoints with `InputValidationFailed` (`Not in enum list: 60, 120, ...`); `granularity=60` returns exact 60s spacing. Sub-60s candles do not exist; shifted bars must be tick-built.
- **(b) 1s ticks:** history is perfectly contiguous for all six pairs (300/300 ticks at exactly 1s deltas each).
- **(c) Tick subscription:** supported on **both** the options and legacy endpoints; **all six symbols multiplex on one socket**; tick fields `ask, bid, epoch, id, pip_size, quote, symbol`; ~1 tick/s with occasional gaps ≤6s in the off-session capture (NY-session coverage is measured at the Phase-2 gate, not here).
- **(d) Retention:** ticks exist at/before the **2000-01-01 search floor for all six pairs** — retention ≥ ~26.5 years (recorded as a lower bound). The Phase-5 offline audit is **not retention-bound**; no audit-scope shrink needed. Paging via `end = oldest − 1` returns sorted, disjoint, backward-ordered pages; the server caps tick pages at **1000 ticks/page** (5000 requested).
- **(e) Proposals:** unauthenticated proposals return `id, ask_price, payout, spot, date_start, date_expiry, ...`; **proposal subscription is supported** (`subscribe: 1` → ~16 updates/15s, forgettable); `contracts_for` lists CALL/PUT with a 15m minimum duration.
- **(f) Rate budget:** the worst-case daemon cycle (12 proposals + 6 candle fetches per 5s) ran 24 cycles, **432/432 OK, zero errors** — no API-side rate limit at this load. But cycle wall was p50 5.4s / p95 6.7s on the **serial sync client** (`deriv_client.py`, request latency p50 130ms), so a 5s cadence is **not achievable by serial polling**. KILL #8 resolution: hot CALL+PUT quote state rides **proposal subscriptions** (supported per (e)) and/or the Phase-1 async client's parallel dispatch — the budgeted-polling descope is unnecessary.
- **(i) Candle publication:** a completed 1m candle is fetchable **p95 0.5s** after the minute boundary (legacy poll, 4 boundaries); completed closes are final (unchanged 5s later, 4/4); **candle subscription (`style=candles, subscribe=1`) is supported on both endpoints** (~1 `ohlc` frame/s). The 5s refresher cadence is confirmed viable; the wall-clock freshness target (publication latency + 10s) lands at ~10.5s, well inside the ≤60s coordinator bar.
- **(g) Near-close: PARTIAL (2026-07-02)** — the run captured Deriv's daily close/rollover blackout: 15m FX proposals are rejected (`ContractBuyValidationError`, blackout 16:50–18:00 NY) when their expiry lands in it, so the effective last start is **~16:35 NY**. A later committed probe record proves rejection by `16:43:10 NY` but still does not pin the last accepted boundary. The issue-#6 shared runtime cutoff therefore rejects `>= 16:35:00 NY` until a committed accepted-boundary capture supersedes it.
- **(h) Baseline: RETIRED (issue #5)** — this probe timed `deriv_demo_executor.py --once` for a peak-RSS baseline, but #5 deprecated that executor. Live-runtime RSS is owned by the #5 cutover acceptance, and an ongoing ceiling is enforced by systemd `MemoryHigh`/`MemoryMax` on the ops units (measured 2026-07-02: quote-audit ~200 MB, candle-refresh ~481 MB peak). `--probes h` now returns `status: retired`.

## Environment

Required only for the credentialed executor:

```bash
export DERIV_APP_ID=...
export DERIV_PAT=...
export DERIV_ACCOUNT_ID=...
```

The PAT must have Options API trading permission. The executor obtains an authenticated demo WebSocket URL through Deriv's OTP endpoint and refuses to buy unless the returned URL is a demo endpoint.

## Install

Preferred reproducible sync:

```bash
uv pip sync requirements-deriv-demo.txt --python ~/binary-algo-venv/bin/python
~/binary-algo-venv/bin/python -m pip check
```

Fallback:

```bash
~/binary-algo-venv/bin/python -m pip install -r requirements-deriv-demo.txt
~/binary-algo-venv/bin/python -m pip check
```

## Local Smokes

Book and replay schema:

```bash
~/binary-algo-venv/bin/python scripts/deriv_demo_executor.py --check-books --replay-schema-check
```

Deterministic fake-client regression, no live credentials:

```bash
~/binary-algo-venv/bin/python scripts/deriv_demo_executor.py --fake-client-smoke
```

Rolling store health:

```bash
~/binary-algo-venv/bin/python scripts/deriv_backfill.py health --out-dir deriv_data/candles_1m --pairs all --min-required-rows 14000 --max-stale-seconds 180
```

Public market-data and live feature check:

```bash
~/binary-algo-venv/bin/python scripts/deriv_demo_executor.py --pairs all-enabled --public-smoke
~/binary-algo-venv/bin/python scripts/deriv_demo_executor.py --pairs EURUSD --public-smoke
```

The second command is expected to fail closed with the missing verified `OF_*` provider message.

Executor auth health with VPS secrets:

```bash
~/binary-algo-venv/bin/python scripts/deriv_trade_executor.py --run --duration-seconds 1
```

Proposal-only dry run, the default execution mode:

```bash
~/binary-algo-venv/bin/python scripts/deriv_demo_executor.py --pairs USDCHF --dry-run-proposal-only --once
```

Legacy `deriv_demo_executor.py --demo-buy` is disabled. Demo buys are executor-owned:

```bash
~/binary-algo-venv/bin/python scripts/deriv_trade_executor.py --run --stake 1
```

## Runtime Guards

The executor requires 15m `CALL` and `PUT` availability from `contracts_for`, a finite feature row matching the frozen book schema, the 08:00-17:00 America/New_York session gate, the frozen strategy confidence gate at the **registry-resolved active coverage**, a resolved side floor + haircut for every pair (fail-closed at startup), the payout-aware edge gate above, no `logs/paper_trades/KILL` file, and a healthy rolling store when `--store-dir` is set.

Every skip logs a stable `reason` (`outside_ny_session`, `feature_row_failed`, `book_gate`, `max_trades_day`, `max_open`, `pair_cooldown`, `missing_or_invalid_ask`, `missing_or_invalid_payout`, `edge_not_positive`, `breakeven_too_high`, `dry_run_proposal_only`, `demo_buy_not_enabled`, plus `lock_acquire_failed` / `reconcile_failed` / `kill_switch_triggered` events). Buy and monitor events carry the full decision context (side, contract type, ask/payout, effective floor, live breakeven, net edge, ids, expiry, terminal status, profit). Logs are JSONL- and secret-safe: raw responses are hashed, and the auth smoke prints only a SHA-256 prefix of the account id.

`deriv_data/runtime/trade_queue.sqlite` stores queued signals and linked contract rows. On executor startup/auth health it can claim queue rows independently of producer/store health; unresolved exposure remains counted until account/contract evidence or expiry-plus-grace reconciliation clears it.

**Risk caps are opt-in** (demo default: trade every firing signal): `--max-open`, `--max-trades-day`, and `--pair-cooldown-seconds` all default to 0 = unconstrained; set them explicitly to re-impose limits. Post-buy monitoring is **detached** by default (the contract settles via per-cycle reconciliation); `--blocking-monitor` restores the old block-until-terminal behavior, which serializes the account to one open trade at a time.

The credentialed executor uses the shared account lock root from `deriv_runtime_core.canonical_lock_root()`, not a per-service log directory. A second buy-capable process for the same account fails closed on the same lock path.

Demo-buy mode monitors `proposal_open_contract` until terminal status, writes `contract_update` and `contract_closed` JSONL events, then forgets the subscription. Proposal-only mode may request proposals but must not call `buy` or emit `buy_submitted` / `buy_confirmed`.

## VPS Files

Templates live under `ops/`:

```text
ops/deriv-demo-executor.env.example
ops/deriv-demo-executor.service              (DEPRECATED — issue #5, retained disabled)
ops/deriv-demo-executor.timer                (DEPRECATED — issue #5, retained disabled)
ops/deriv-trade-executor.service             (issue #6 credentialed queue executor)
ops/deriv-demo-logrotate
ops/deriv-market-stream.service
ops/deriv-runtime-supervisor.service
ops/deriv-production-candle-refresh.service  (issue #5)
ops/deriv-production-candle-refresh.timer    (issue #5)
ops/deriv-quote-audit.service                (issue #5)
ops/deriv-probe-g-nearclose.service          (issue #4 — Phase-0 probe g capture)
ops/deriv-probe-g-nearclose.timer            (issue #4 — fires 16:19 NY weekdays)
```

**Probe g near-close scheduler** (`ops/deriv-probe-g-nearclose.{service,timer}`): fires `scripts/deriv_api_probe.py --probes g --near-close-start 16:20` at **16:19 NY on weekdays** (the `OnCalendar=... America/New_York` TZ suffix is load-bearing — hosts on other timezones would otherwise misfire), writing gitignored `logs/probe_g_scheduled_capture.json`. Fully unattended: on a clean accepted→rejected capture the service `ExecStartPost` runs `scripts/fold_probe_g_capture.py`, which merges the `g_near_close` record into `results/json/deriv_api_probe_result.json` and **commits + pushes it** (fail-safe: rebase-first, never force-push, retry), then the timer **self-disables**. A non-clean run (holiday/closed market) is a no-op that leaves the timer armed to retry the next weekday. Manual stop if needed: `systemctl --user disable --now deriv-probe-g-nearclose.timer`. (`loginctl enable-linger` — no sudo needed for self — makes the user timer survive a full logout/reboot; enabled on the dev host 2026-07-02.)

Copy the env template outside git, fill only VPS-local secrets, then install the units with paths adjusted if the checkout or venv differs. Emergency stop is `touch <log-dir>/KILL` (per log dir).
The absolute live-breakeven sanity ceiling is configured with `DERIV_DEMO_ABSOLUTE_BREAKEVEN_CEILING`; existing private env files that still contain `DERIV_DEMO_MAX_BREAKEVEN` continue to work, but new installs should use the clearer name.

The old per-minute executor service/timer are **deprecated** (issue #5): `--dry-run-proposal-only` skips only the buy, so every 60s run still loaded books, built features, and scored (~69s wall, 1.2–1.4 G peak, operator-reported). Its two surviving support roles moved to purpose-built non-buying jobs:

- **`deriv-production-candle-refresh.timer`** — oneshot `deriv_backfill.py backfill --max-pages 2 --prune-shards --reset-stale-progress` every 180s; sole writer of `deriv_data/candles_1m/`. `TimeoutStartSec=150` kills a hung run inside one cycle, so worst-case staleness is 330s steady-state / 540s after one failed-then-reset cycle — the production-store health contract is therefore `health --max-stale-seconds 600`.
- **`deriv-quote-audit.service`** — long-running `deriv_quote_workers.py --audit`: 12 proposal-subscription lanes on the PUBLIC socket, one `quote_snapshot` row per lane every 30s, per-lane resubscribe recovery (`quote_resubscribe_*` events), stale-lock reap, `UnsetEnvironment=DERIV_PAT DERIV_ACCOUNT_ID` (the sampler never sees the credential that mints a demo client).

Secrets, account IDs, JSONL logs, and `deriv_data/` artifacts must stay out of git.

## Hot Runtime (issue #4, Phases 1-6)

The hot 5s runtime shares this executor's decision chain — `deriv_runtime_core.py`
holds the extracted gate/monitor/reconcile/state helpers, pair + floor
resolution, `effective_floor_for`, and the fake test harness; both binaries
import it (parity by construction). Modules:

- `deriv_async_client.py` — websockets transport: req_id futures, subscription
  registry, batched auto-resubscribe (<1s live), fail-closed error surface.
  Gates: `--smoke` (deterministic), `--soak-minutes 30 --inject-disconnects 3`.
- `deriv_market_stream.py` — tick store + 12-offset shifted aggregator +
  daemon candle refresher (see `docs/DERIV_DATA_STORE.md`). **Must run before
  the supervisor** — the daemon store health gate fails closed when stale.
- `deriv_hot_daemon.py` — offset-0 wall-clock scoring, books loaded once,
  per-minute store snapshots + `--parity-replay` (KILL #0a: the deterministic
  tuple recomputed from archived snapshots must match 100%).
- `deriv_quote_workers.py` — hot CALL/PUT quote state via proposal
  subscriptions on the PUBLIC socket (buy-incapable by construction; the
  `--smoke` AST gate proves no buy-shaped node exists in the module).
  `--audit` runs the standalone quote-audit sampler (issue #5). Gates:
  `--smoke` (deterministic), `--probe-seconds 60` (live).
- `deriv_runtime_supervisor.py` — producer in issue-#6 queue mode: no demo
  credentials, no buy path, selected-side candidates committed to SQLite after
  the book gate. Legacy non-queue proposal-only mode remains for comparison.
- `deriv_trade_executor.py` — credentialed queue executor: account lock,
  auth-health, pre-money gates, final payout gate, buy submission, contract
  rows, and fake/live gate JSON.
- `deriv_offset_audit.py` — the pre-registered Phase-5 shifted-lane audit
  (falsifier-first; the audit RUN is a `strategy-eval` session).

**Deploy order (VPS):** install `ops/deriv-market-stream.service` +
`ops/deriv-runtime-supervisor.service`; start the market stream, wait for
daemon-store health, then the supervisor (proposal-only). Production-store
freshness and `quote_snapshot` coverage are owned by the issue-#5 support
jobs (`deriv-production-candle-refresh.timer` + `deriv-quote-audit.service`);
the old executor timer is deprecated — cut over per the runbook below.
Buy exclusivity: do not run legacy demo-buy modes. Cutover verifies
`systemctl --user disable --now deriv-demo-executor.service deriv-demo-executor.timer`,
`pgrep -f deriv_demo_executor` is empty, and
`pgrep -af 'deriv_runtime_supervisor.py .*--demo-buy'` is empty before
`deriv-trade-executor.service` starts. Emergency stop:
`touch <log-dir>/KILL` (per log dir).

**Cutover runbook (issue #5; executed on the VPS by the operator/agent):**

1. `git pull`; install the two new units + the updated logrotate config.
2. **Enable + start** the sampler: `systemctl --user enable --now
   deriv-quote-audit.service` (`Restart=` survives crashes, not reboots — an
   unenabled sampler silently stops accruing ≥3-NY-session coverage at the
   first reboot); verify 12-lane rows are flowing. Coexists with the old
   timer; executor-era rows keep accruing coverage toward their own
   ≥3-session admission — overlap length does not gate validity, the MAX
   rule does.
3. `systemctl --user disable --now deriv-demo-executor.timer` AND
   `systemctl --user stop deriv-demo-executor.service`; **wait until
   `systemctl --user is-active deriv-demo-executor.service` reports
   inactive** — stopping the timer does not stop an in-flight ~69s run, and
   a concurrent second backfill is corrupting, not just racy (both processes
   write the same fixed `{pair}.parquet.tmp` paths).
4. `systemctl --user enable --now deriv-production-candle-refresh.timer` —
   fires an immediate first run (OnBootSec already past on a long-booted
   VPS); safe only once step 3 is complete.
5. Confirm the hot services are untouched, no scheduled
   `deriv_demo_executor.py` remains (`systemctl --user list-timers`;
   `pgrep -f deriv_demo_executor`), and `deriv_backfill.py health --out-dir
   ${DERIV_DEMO_STORE_DIR} --max-stale-seconds 600` exits 0.
**Parity re-check rule:** any change to the registry, a book,
`live_features.py`, or `deriv_runtime_core.py` re-arms the 1-session KILL #0
re-check before the next demo-buy session.

Phase gate status lives in issue #4 and the per-gate result JSONs under
`results/json/` (`deriv_async_soak_result.json`,
`deriv_market_stream_gate_result.json`, `deriv_hot_daemon_gate_result.json`,
`deriv_parity_result.json`, `deriv_offset_audit_falsifier.json`).
