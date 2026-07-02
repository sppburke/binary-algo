# Deriv Demo Executor

`scripts/deriv_demo_executor.py` runs the six deployable non-EURUSD 15m books against Deriv Options demo infrastructure:

```text
USDJPY, USDCAD, AUDUSD, NZDUSD, USDCHF, GBPUSD
```

`EURUSD` is disabled until a verified live provider can produce finite `OF_*` order-flow columns. `--pairs all-enabled` means the six-pair set above. `--pairs EURUSD --public-smoke` fails closed.

This is operational demo tooling. It does not certify Deriv-native edge, does not enable real-money execution, and does not replace the frozen-book registry.

## Book Model and Payout-Aware Edge Gate

Runtime loads **one combined direction book per pair** (not separate UP/DOWN artifacts): it averages the book's seed probabilities, selects `UP` when `p >= 0.5` else `DOWN`, and requests `CALL` or `PUT`. The registry's UP/DOWN numbers are **side-specific measured floors for the selected side** (refit-CPCV p10 conditional on the model selecting that side at the coverage gate) and drive payout gating.

At startup `scripts/deriv_floor_resolver.py` resolves, per enabled pair and fail-closed: active coverage, confidence threshold, side floors, and the settlement haircut with its evidentiary source (precedence: structured registry/manifest/strategy-deploy metadata → the pair's own validated ticksettle result JSON → a ledger-documented trusted proxy → `fallback_max_validated_fx_ticksettle`, recomputed at resolution time from `results/json/*_15m_ticksettle_result.json`, never a frozen constant). The xpair books (USDCHF/GBPUSD) resolve **cov1** thresholds+floors via `deploy_cov` in `books/INDEX.json` — the old silent `"0.02"` gates fallback is a startup failure now. Every resolution is logged as a `floor_resolution` JSONL event including the signed raw haircut, the applied absolute haircut, and the book freeze/vintage date.

For a candidate trade on the selected side:

```text
live_breakeven  = ask_price / payout        # fail closed on missing/non-positive either
effective_floor = side_refit_p10 - haircut_abs - payout_edge_margin   # margin default 0.005
buy only when    effective_floor > live_breakeven
```

`--max-breakeven` (default 0.60) remains a secondary absolute ceiling; the side-floor edge gate is the authoritative guard. Raw model `proba` is never used as a calibrated win probability. There is no `display_value`/stake fallback for `ask`.

**Refit-vintage disclosure:** floors are refit-CPCV **per-era** floors. The frozen vintages the executor loads decay forward (frozen-forward 2026 COMBINED: AUDUSD 0.5082, NZDUSD 0.5194; USDCAD DOWN 0.5258). The gate certifies the documented floor under the periodic-retrain deploy policy, not the frozen artifact's current forward win rate; this stays **demo-only** until a retrain cadence exists. Haircuts are per-book means; worst year/side cells are materially larger (e.g. NZDUSD 2024 DOWN −0.0388) — revisit worst-year haircuts and/or a larger margin before any real-money step.

## Quote Snapshots and API Call Budget

Default mode (`--quote-snapshots all`) snapshots **both CALL and PUT quotes for every enabled pair each cycle** for observability, logged as `quote_snapshot` events; snapshots are never reused for buys — any buy uses a fresh selected-side proposal. `--quote-snapshots off` is the reduced-cadence override so the one-minute timer cannot create uncontrolled proposal traffic.

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
- **(g) Near-close: PARTIAL (2026-07-02)** — the run captured Deriv's daily close/rollover blackout: 15m FX proposals are rejected (`ContractBuyValidationError`, blackout 16:50–18:00 NY) when their expiry lands in it, so the effective last start is **~16:35 NY** — the `16:44:59` default is too permissive. But that run sampled 16:43–17:00 NY, entirely inside the blackout, so it never saw an accepted→rejected *transition*. A widened window is now configurable (`--near-close-start`, default 16:20); a weekday-16:19-NY scheduler re-runs to pin the exact boundary (writes gitignored `logs/probe_g_scheduled_capture.json`; fold into the record via dev-cycle). Until pinned, the `16:44:59` default stands.
- **(h) Baseline: RETIRED (issue #5)** — this probe timed `deriv_demo_executor.py --once` for a peak-RSS baseline, but #5 deprecated that executor. Live-runtime RSS is owned by the #5 cutover acceptance, and an ongoing ceiling is enforced by systemd `MemoryHigh`/`MemoryMax` on the ops units (measured 2026-07-02: quote-audit ~200 MB, candle-refresh ~481 MB peak). `--probes h` now returns `status: retired`.

## Environment

Required only for `--demo-buy`, `--auth-smoke`, and `--reconcile-only`:

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

Auth smoke with VPS secrets:

```bash
~/binary-algo-venv/bin/python scripts/deriv_demo_executor.py --auth-smoke
```

Proposal-only dry run, the default execution mode:

```bash
~/binary-algo-venv/bin/python scripts/deriv_demo_executor.py --pairs USDCHF --dry-run-proposal-only --once
```

Tiny opt-in demo buy:

```bash
~/binary-algo-venv/bin/python scripts/deriv_demo_executor.py --pairs USDCHF --demo-buy --stake 1 --once
```

## Runtime Guards

The executor requires 15m `CALL` and `PUT` availability from `contracts_for`, a finite feature row matching the frozen book schema, the 08:00-17:00 America/New_York session gate, the frozen strategy confidence gate at the **registry-resolved active coverage**, a resolved side floor + haircut for every pair (fail-closed at startup), the payout-aware edge gate above, no `logs/paper_trades/KILL` file, and a healthy rolling store when `--store-dir` is set.

Every skip logs a stable `reason` (`outside_ny_session`, `feature_row_failed`, `book_gate`, `max_trades_day`, `max_open`, `pair_cooldown`, `missing_or_invalid_ask`, `missing_or_invalid_payout`, `edge_not_positive`, `breakeven_too_high`, `dry_run_proposal_only`, `demo_buy_not_enabled`, plus `lock_acquire_failed` / `reconcile_failed` / `kill_switch_triggered` events). Buy and monitor events carry the full decision context (side, contract type, ask/payout, effective floor, live breakeven, net edge, ids, expiry, terminal status, profit). Logs are JSONL- and secret-safe: raw responses are hashed, and the auth smoke prints only a SHA-256 prefix of the account id.

`logs/paper_trades/executor_state.json` stores open demo contracts with pair, proposal id, contract id, stake, expected expiry, status, and last platform update. On `--demo-buy` startup the executor queries Deriv for every pending local contract; known-open contracts are refreshed and **carried** (concurrent open contracts are normal), and it fails closed only on UNRESOLVABLE state (the status query itself fails).

**Risk caps are opt-in** (demo default: trade every firing signal): `--max-open`, `--max-trades-day`, and `--pair-cooldown-seconds` all default to 0 = unconstrained; set them explicitly to re-impose limits. Post-buy monitoring is **detached** by default (the contract settles via per-cycle reconciliation); `--blocking-monitor` restores the old block-until-terminal behavior, which serializes the account to one open trade at a time.

A single-instance lock is created under the log directory per account/log-dir key. A second executor with the same key fails closed.

Demo-buy mode monitors `proposal_open_contract` until terminal status, writes `contract_update` and `contract_closed` JSONL events, then forgets the subscription. Proposal-only mode may request proposals but must not call `buy` or emit `buy_submitted` / `buy_confirmed`.

## VPS Files

Templates live under `ops/`:

```text
ops/deriv-demo-executor.env.example
ops/deriv-demo-executor.service              (DEPRECATED — issue #5, retained disabled)
ops/deriv-demo-executor.timer                (DEPRECATED — issue #5, retained disabled)
ops/deriv-demo-logrotate
ops/deriv-market-stream.service
ops/deriv-runtime-supervisor.service
ops/deriv-production-candle-refresh.service  (issue #5)
ops/deriv-production-candle-refresh.timer    (issue #5)
ops/deriv-quote-audit.service                (issue #5)
ops/deriv-probe-g-nearclose.service          (issue #4 — Phase-0 probe g capture)
ops/deriv-probe-g-nearclose.timer            (issue #4 — fires 16:19 NY weekdays)
```

**Probe g near-close scheduler** (`ops/deriv-probe-g-nearclose.{service,timer}`): fires `scripts/deriv_api_probe.py --probes g --near-close-start 16:20` at **16:19 NY on weekdays** (the `OnCalendar=... America/New_York` TZ suffix is load-bearing — hosts on other timezones would otherwise misfire), writing gitignored `logs/probe_g_scheduled_capture.json`. It **self-disables** once a run captures a clean accepted→rejected transition (service `ExecStartPost`), retrying the next weekday otherwise (holiday/closed-market tolerant). Fold the captured `g_near_close` record into `results/json/deriv_api_probe_result.json` via dev-cycle. Manual stop if needed: `systemctl --user disable --now deriv-probe-g-nearclose.timer`. Note: a *user* timer needs `loginctl enable-linger` to survive a full logout/reboot.

Copy the env template outside git, fill only VPS-local secrets, then install the units with paths adjusted if the checkout or venv differs. Emergency stop is `touch <log-dir>/KILL` (per log dir).

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
- `deriv_runtime_supervisor.py` — the coordinator: sole buy-capable object,
  admit-token gate battery (session, 16:44:59 NY last-start cutoff,
  lane-scoped staleness, shifted-lane per-(pair,side) verdict enforcement,
  dedup, opt-in risk caps (max_open / max_trades_day / pair cooldown, 0 =
  unconstrained default), payout gate on a fresh proposal), contract_id-correlated
  monitoring. Gates: `--smoke`; live `--run` (proposal-only default).
- `deriv_offset_audit.py` — the pre-registered Phase-5 shifted-lane audit
  (falsifier-first; the audit RUN is a `strategy-eval` session).

**Deploy order (VPS):** install `ops/deriv-market-stream.service` +
`ops/deriv-runtime-supervisor.service`; start the market stream, wait for
daemon-store health, then the supervisor (proposal-only). Production-store
freshness and `quote_snapshot` coverage are owned by the issue-#5 support
jobs (`deriv-production-candle-refresh.timer` + `deriv-quote-audit.service`);
the old executor timer is deprecated — cut over per the runbook below.
Buy exclusivity: before setting `DERIV_SUPERVISOR_MODE_ARGS=--demo-buy`,
drop `--demo-buy` from `DERIV_DEMO_MODE_ARGS`. Emergency stop:
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
