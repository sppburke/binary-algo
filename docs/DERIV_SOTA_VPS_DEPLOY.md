> **SCOPE: GENERIC** — Deriv SOTA certified multi-horizon runtime deployment, queue adoption, monitoring, and promotion runbook.

# Deriv SOTA Certified Strategy VPS Deployment & Promotion Runbook

> [!CAUTION]
> **MANDATORY JURISDICTION BOUNDARY (MEXICO VPS ONLY — ZERO USA DERIV CONNECTIONS):**
> **Never run any live or public Deriv WebSocket / REST command (`deriv_sota_executor.py`, `deriv_sota_tick_stream.py`, `vps_preflight.py --live-smoke`, or ad-hoc `*.derivws.com` scripts) from the USA (`quantum`).** Every direct connection to Deriv servers must originate **exclusively from the Mexico VPS (`sean@64.177.80.63`, hostname `vultr`)** with `DERIV_HOST_JURISDICTION=mexico_vps` set in `~/.config/binary-algo/deriv-sota-executor.env`. The runtime enforces `assert_non_us_deriv_host()` fail-closed before opening any network socket.

> [!IMPORTANT]
> **PHASE 1 DEPLOYMENT & DECISION 5 BOUNDARY:**
> Local inactive packaging on `refs/heads/vps-certified-strategies` (`lifecycle_status: "demo_venue_settled_canary"`) does **NOT** authorize Mexico VPS (`vultr`) deployment or activation. Any `vultr` git fetch/move, sole-buyer transition, or unit activation requires **separate explicit owner authorization** under `deriv-runtime-ops` + `git-workflow`, including:
> 1. Outcome-blind `scripts/usdchf_m15_prospective_v1.py status` BEFORE any `git fetch` or checkout move (requiring terminal or positively absent state);
> 2. Explicit commit transfer disposition (`origin/main` `ff-only` or explicit non-`main` disposition);
> 3. Explicit legacy KILL #0 parity re-check disposition (Phase 1 leaves all 9 `PARITY_TRACKED_PATHS` byte-identical);
> 4. **Explicit Owner Decision 5 payout deployment disposition** — per-trade payout-aware path evidence for `[USDJPY·30m·UP]` OR a separately owner-approved amendment expressly authorizing Demo deployment without it (`--live-smoke` proves current proposal admission ONLY and never satisfies Decision 5).

---

## 1. Safety & Certification Architecture

1. **Zero Uncertified Execution:** `scripts/sota_book_runtime.py` refuses any book missing cryptographic SHA-256 verification, exact `content_id`, ordered LightGBM `feature_name() == feature_cols` (for `nystroem-lgb-stack`), or passed `G1_cpcv`–`G5_payout` certificates.
2. **Phase 1 Priority & Exact-Money Policy:** Explicit priority order `USDJPY.sota_ens5_m30_up.v1 -> USDJPY.sota_stack05_m30_up.v1 -> USDJPY.sota_ens5_m15_up.v1`, flat `100` authoritative USD cents (`--stake 1.0`), exact rational quote floors (`1.76` m15 / `1.78` m30, defeating `1.7799999999`), and `SotaDerivAsyncClient` 5-class generation-bound transport supporting both `options_otp` (bootstrap `balance` + venue `time`) and `v3_authorize`, with `allow_buy=False` blocking both `buy` and `sell` (`0` bytes sent).
3. **Account-Bound `v3` SQLite Queue & Locks:** Canonical passwd-home absolute locks (`~/.local/state/binary-algo/locks/sota_{acct,queue,handoff}_<sha12>.flock`, acquired strictly in order `sota_acct -> sota_queue -> sota_handoff`) and `user_version = 3` queue (`deriv_data/runtime/sota_trade_queue_demo.sqlite`) with `UNIQUE(book_id, pair, horizon_minutes, side, lane, offset_id, signal_close_utc)`, `sota_contracts.signal_id UNIQUE`, durable pre-send `buy_intent`, and log/feed-independent latched recovery (`--request-stop-latch`).
4. **Exclusivity Controls:** Revoking legacy Deriv credentials (`sota-credential-isolation-receipt/v1`), masking all 4 legacy units, and canonical locks are the **primary** exclusivity controls; startup `/proc` cmdline scanning is supplementary evidence only (`deriv-runtime-ops:105`).

---

## 2. Separately Authorized Mexico VPS Cutover Sequence (`vultr` Only)

Run these steps **strictly on `vultr`** after separate explicit owner authorization:

### Step 1 — Outcome-Blind Prospective Shadow Check (Before Any `git fetch` or Move)
```bash
cd ~/git/binary-algo
~/binary-algo-venv/bin/python scripts/usdchf_m15_prospective_v1.py status
```
Require terminal or positively absent state; stop immediately on open or unclear state.

### Step 2 — Drain, Stop, Disable & Mask All 4 Legacy Units by Role + Current SOTA Unit
Mask `deriv-sota-executor.service` and all **4** legacy units by their distinct operational roles:
- `deriv-trade-executor.service` — legacy sole credentialed buyer (`deriv_trade_executor.py --run`)
- `deriv-runtime-supervisor.service` — legacy queue producer (also carries a manual `--demo-buy` path under a separate lock)
- `deriv-demo-executor.service` & `deriv-demo-executor.timer` — deprecated one-shot & minute timer (disabled for store-write and service conflicts)

```bash
systemctl --user stop deriv-sota-executor.service deriv-trade-executor.service deriv-runtime-supervisor.service deriv-demo-executor.service deriv-demo-executor.timer
systemctl --user disable deriv-sota-executor.service deriv-trade-executor.service deriv-runtime-supervisor.service deriv-demo-executor.service deriv-demo-executor.timer
# If regular unit files already exist under ~/.config/systemd/user/, move them aside before masking:
mkdir -p ~/.config/systemd/user/.legacy_disabled && mv -n ~/.config/systemd/user/deriv-{trade-executor,runtime-supervisor,demo-executor}.service ~/.config/systemd/user/deriv-demo-executor.timer ~/.config/systemd/user/.legacy_disabled/ 2>/dev/null || true
systemctl --user daemon-reload
systemctl --user mask deriv-sota-executor.service deriv-trade-executor.service deriv-runtime-supervisor.service deriv-demo-executor.service deriv-demo-executor.timer
```

### Step 3 — Establish Demo Credentials & Non-Secret Receipt FIRST
Establish Demo (`VRTC*` / `VRW*` on WS v3; `VRTC*` / `VRW*` / `D[A-Z]{2}[0-9]+` on Options v1 `/options/ws/demo`, never `CR*`/`MF*`/live) credentials and the non-secret credential isolation receipt **BEFORE** any first-use or adoption command:
- Use relative `DERIV_SOTA_QUEUE_DB=deriv_data/runtime/sota_trade_queue_demo.sqlite` in `~/.config/binary-algo/deriv-sota-executor.env` so canonical queue resolution matches even when `~/git/binary-algo` is a symlink (as on `vultr`).
- **Mode A (Strict dedicated PAT — `"credential_isolation_mode": "revoked"`, default):** Revoke legacy Deriv PATs in the Deriv dashboard, populate `DERIV_APP_ID`/`DERIV_PAT`/`DERIV_ACCOUNT_ID` in `~/.config/binary-algo/deriv-sota-executor.env` (`chmod 600`), do **NOT** install `ops/deriv-sota-executor.single-host-demo.conf`, and write `~/.config/binary-algo/sota-credential-isolation-receipt.json` with `"legacy_credentials_revoked": true` and exact Demo `account_id`.
- **Mode B (Opt-in owner-authorized single-host Demo reuse — `"credential_isolation_mode": "single_host_units_masked"`, Demo accounts ONLY):** Comment out the 3 credential lines in `~/.config/binary-algo/deriv-sota-executor.env`, install the opt-in drop-in `~/.config/systemd/user/deriv-sota-executor.service.d/10-single-host-demo.conf` from `ops/deriv-sota-executor.single-host-demo.conf` (zero secret copying; default Mode A never loads `deriv-demo-executor.env`), and write:
  - `~/.config/binary-algo/sota-credential-isolation-receipt.json` with `"schema": "sota-credential-isolation-receipt/v1"`, `"credential_isolation_mode": "single_host_units_masked"`, `"account_mode": "demo"`, `"account_id": "VRTC_FROM_ENV"`, `"legacy_credentials_revoked": false`, exact `legacy_units_masked`, non-empty string `"attested_by"`, and RFC3339 UTC `"attested_utc"` (verified at runtime against both `/dev/null` symlinks in `~/.config/systemd/user/` AND live `systemctl --user show` reporting `LoadState=masked`, `UnitFileState=masked`, `ActiveState=inactive` for all 4 units).
  - `~/.config/binary-algo/sota-first-use-record.json` (`schema: "sota-first-use-record/v1"`, `"account_mode": "demo"`, `"account_id": "VRTC_FROM_ENV"` — sentinel admitted ONLY under Mode B Demo).
  Live (`CR*`) accounts unconditionally refuse Mode B. For manual Mode B commands in Steps 4–5 (while the service is masked/inactive), run them via `systemd-run --user --wait --pipe --collect --working-directory="$HOME/git/binary-algo" --property=EnvironmentFile="$HOME/.config/binary-algo/deriv-demo-executor.env" --property=EnvironmentFile="$HOME/.config/binary-algo/deriv-sota-executor.env" -- ...` (using absolute `$HOME` paths, since transient `systemd-run` `EnvironmentFile=` properties reject `%h`) so credentials stay inside systemd's `EnvironmentFile=` boundary without shell sourcing or printing.

### Step 4 — Inspect Predecessor Queue & Branch Cleanly to First Use or Coherent-Backup Adoption
Perform sanitized schema/row-state and positive venue identity/settlement inspection on `vultr`:
- **First Use (no existing predecessor DB):**
  ```bash
  ~/binary-algo-venv/bin/python -B -E -s scripts/deriv_sota_executor.py \
    --init-queue --reason first_use \
    --first-use-record-json ~/.config/binary-algo/sota-first-use-record.json \
    --account-mode demo --credential-isolation-receipt ~/.config/binary-algo/sota-credential-isolation-receipt.json
  ```
- **Staged Adoption (existing drained `user_version == 0` predecessor DB):**
  1. Prepare coherent SQLite online backup `<quarantine_dir>/predecessor_coherent_backup.sqlite` (`fsync` + `integrity_check`) FIRST;
  2. Export preimage strictly from that prepared backup (`0` bytes written to source):
     ```bash
     ~/binary-algo-venv/bin/python -B -E -s scripts/deriv_sota_executor.py \
       --export-legacy-ownership-preimage --maintenance-quarantine-dir "$QUARANTINE_DIR" --account-mode demo
     ```
  3. Build venue ownership proof (`backup_sha256` + 1-to-1 `contract_ownership_records` bijection against `statement`/`profit_table` + `proposal_open_contract`);
  4. Run `--adopt-legacy-account` (dispatches existing `install_intent.json` crash recovery BEFORE fresh predecessor checks, authenticates backup/proof/receipt, verifies directory ancestry durability before unlinking predecessor files, installs single-file `v3` via `renameat2(RENAME_NOREPLACE)`, leaves `deriv_data/runtime/locks/sota_index.lock` and `<quarantine_dir>` untouched):
     ```bash
     ~/binary-algo-venv/bin/python -B -E -s scripts/deriv_sota_executor.py \
       --adopt-legacy-account --account-mode demo \
       --maintenance-quarantine-dir "$QUARANTINE_DIR" \
       --staged-v3-dir "$STAGED_V3_DIR" \
       --ownership-proof-json "$PROOF_JSON" \
       --credential-isolation-receipt ~/.config/binary-algo/sota-credential-isolation-receipt.json
     ```
  5. Register strictly prospective `G6` cohort anchors per book via `--register-cohort-anchor` (binding `registered_utc` to verified venue `time` epoch).

### Step 5 — Manual Masked Preflight (`--offline-only` & `--live-smoke`)
While `deriv-sota-executor.service` stays `/dev/null`-masked and `sota_activation.hold` is absent, verify `--offline-only` and `--live-smoke` (every selected book must pass strict exact rational quote floor `1.76` m15 / `1.78` m30, `0` buy/sell frames sent).

### Step 6 — Activation Hold, Unit Install, Public Directive Audit & Gated Enablement
```bash
mkdir -p ~/.local/state/binary-algo/locks && chmod 700 ~/.local/state/binary-algo/locks
touch ~/.local/state/binary-algo/locks/sota_activation.hold
systemctl --user unmask deriv-sota-executor.service
cp ~/git/binary-algo/ops/deriv-sota-executor.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user show deriv-sota-executor.service -p ExecStartPre,ExecStart,Environment,EnvironmentFiles,UnsetEnvironment,DropInPaths,ConditionPathExists,Conflicts
# Only after owner authorizes Decision 5 / Demo activation:
rm -f ~/.local/state/binary-algo/locks/sota_activation.hold
systemctl --user enable --now deriv-sota-executor.service
```

---

## 3. Prospective `G6` Cohort Ledger & Safe Rollback

- **Prospective `G6` Evaluation (Per-Book Only, Never Pooled):**
  ```bash
  ~/binary-algo-venv/bin/python -B -E -s scripts/deriv_live_pnl_ledger.py \
    --book-id USDJPY.sota_stack05_m30_up.v1 \
    --cohort-start-utc <RFC3339_UTC> \
    --expected-account-id VRTC... \
    --expected-stake-usd 1.0 \
    --min-trades 50
  ```
  `--db` is diagnostic only (`promotion_eligible: false` on non-canonical path). Eligibility requires verified anchor match, `venue_start_epoch >= cohort_start_epoch`, full `proposal_open_contract` venue-field and contract-spec agreement, zero anomalies, `n >= max(50, min_trades)`, and exact `Fraction` 95% bootstrap lower bound `> 0`.
- **Safe Rollback / Emergency Stop:**
  Trigger `<log_dir>/KILL` OR run `--request-stop-latch --reason <text>` (commits `phase1_safety_latch` atomically in SQLite under `LOCK_EX` on `sota_handoff` even if `log_dir` is unwritable). Keep the service in log/feed-independent reconciliation-only mode until `0` open/unknown contracts and `0` `buy_intent`/`buy_unknown` signals remain, then stop the unit while keeping `v3` risk history and latch intact.

---

## 4. Separately Authorized 4-Book Rolling-Refit Expansion (`USDJPY.sota_roll14d_m60_up.v1`, Demo-Only — Owner Decision 15)

> [!IMPORTANT]
> **INACTIVE PACKAGING & SEPARATE AUTHORIZATION BOUNDARY:**
> Tracked policy `books/USDJPY.sota_roll14d_m60_up.v1/` (`books/SOTA_ROLLING_INDEX.json`, `policy_content_id = c0bc05ed06e288c1`, `lifecycle_status: "demo_rolling_refit_canary"`) and systemd templates `ops/deriv-sota-4book-cutover.conf`, `ops/deriv-sota-tick-archive.{service,timer}`, and `ops/deriv-sota-rolling-refit.{service,timer}` are **inactive** on `main`. They do **NOT** authorize starting archive backfill, running refits, or cutting over the running 3-book Demo cohort on `vultr`. Each VPS activation step requires **separate explicit owner authorization** under `deriv-runtime-ops`.
> - **Parity & Custody Disposition:** All 5 static frozen books in `books/SOTA_INDEX.json`, all 15 `PHASE1_SOURCE_CLOSURE_FILES`, and all 9 `PARITY_TRACKED_PATHS` remain **byte-identical** (legacy KILL #0 parity preserved unchanged).
> - **Demo-Only Boundary:** `USDJPY.sota_roll14d_m60_up.v1` is strictly Demo-only (`live_real_money_eligible: false`, unconditionally refused in `live` mode).

### Stage A — Non-Credentialed Rolling Tick Archive Bootstrap (Can Coexist With Live 3-Book Demo)
1. Vendored Archive Downloader 1.7 (`scripts/deriv_ticks_download_vps.py`) connects strictly from `vultr` (`DERIV_HOST_JURISDICTION=mexico_vps`), uses pinned `wss://api.derivws.com/trading/v1/options/ws/public` (zero credentials, zero `EnvironmentFile=`), atomically creates/acquires `~/.local/state/binary-algo/deriv_public_connection.{lock,admission.json}` (`O_NOFOLLOW`, `0o600`), configures `--window 1000` and `--per-minute 20.0` in `ops/deriv-sota-tick-archive.service` (CLI hard ceiling `24.0 req/min`), fetches `--rolling-days 145` (`120` train + `7` warmup + up to `14` cadence-lag + `4` buffer calendar days) into repo-relative `deriv_data/rolling_ticks_archive/`, and refuses `/mnt/data/deriv*` research roots and symlinks.
2. Resource & timeout bounds on `vultr` (`3.3 GiB` RAM, `~34 GiB` disk free): `ops/deriv-sota-tick-archive.service` pins `Nice=10`, `MemoryHigh=512M`, `MemoryMax=1G`, and `TimeoutStopSec=1920` (covering WebSocket teardown plus full `900s` cooldown and `900s` failed-persistence quiet drain). One-time 145-day 7-pair bootstrap takes `~61,500` base requests (`~51h` at the configured `20 req/min` before retries); scheduled top-ups run every `6h` via `ops/deriv-sota-tick-archive.timer` (`OnCalendar=*-*-* 00,06,12,18:10:00 UTC` with `RandomizedDelaySec=120`), with a full Monday–Thursday 7-pair weekday top-up taking `~609` base requests (`~30.5 min` at `20 req/min` before retries).
3. Separate owner authorization installs `ops/deriv-sota-tick-archive.{service,timer}` into `~/.config/systemd/user/` without touching `deriv-sota-executor.service` or `deriv_data/ticks_1s/`.

### Stage B — Offline Causal 14-Day Rolling Refit Verification (`deriv-sota-rolling-refit.{service,timer}`)
1. `scripts/deriv_sota_rolling_refit.py` runs **strictly offline** (zero Deriv sockets; `Nice=15`, `MemoryHigh=1G`, `MemoryMax=1536M`; repo-relative `--archive-dir deriv_data/rolling_ticks_archive --vintage-root deriv_data/runtime/rolling_vintages` safe under `vultr`'s `~/git/binary-algo` symlink layout), acquires shared `LOCK_SH | LOCK_NB` on `<archive_dir>/.download.lock` (exiting `75` if a download is writing), audits Downloader 1.7 `complete == true` / `gaps_reconfirmed` sidecars (`symbol`, `date`, `clipped_by_*`, `parquet_sha256`, unfinished `parquet_sha256_next` refusal), freshness, and weekday gap budgets, streams the 6 non-target pairs day-by-day (`< 1 GiB` peak RSS), builds causal `core66` features with explicit `cutoff_utc = scheduled_refit_epoch` and `horizon_s = 3600`, verifies `< 1e-11` reload parity via `load_verified_vintage_book()`, and atomically replaces `transitions.jsonl`, `writer_watermark.json`, and `state.json` under `.transition.flock` (sampling publication time under the lock and verifying the journal head did not change during fit).
2. When trailing-120d validation produces `platt_coef <= 0`, the refitter commits `status = "abstained"` locked across the entire 14-day block (`--force`, `--revoke-reason`, and `--resume-after-revoke` never override a block abstention), immediately preventing trading until the next scheduled 14-day refit clears `platt_coef > 0`.

### Stage C — Clean Drain-to-4-Book Cutover Sequence (`vultr` Only, After Separate Owner Approval)
1. Confirm `0` open/unknown contracts and `0` `buy_intent`/`buy_unknown` signals on the 3-book Demo cohort (wait for any open contract to settle cleanly).
2. Stop `deriv-sota-executor.service` cleanly (`systemctl --user stop deriv-sota-executor.service`) **without** creating `<log_dir>/KILL` or setting `phase1_safety_latch` (an unlatched drained queue is required for `--live-smoke`). Pause `deriv-sota-tick-archive.service` if running so `~/.local/state/binary-algo/deriv_public_connection.lock` is free for `--live-smoke`.
3. Durably seal the original 3-book (`3/3/3`) cohort observation cutoff in `sota_risk_state` (`cutoff_utc <= now`, whole-second RFC3339 UTC; required before 4-book cutover startup or preflight will admit 4-book buys) so post-cutover 4-book trades never retroactively contaminate the 3-book MVT cohort (pass `deriv-sota-executor.env` for Mode A; add `--property=EnvironmentFile="$HOME/.config/binary-algo/deriv-demo-executor.env"` only on Mode B hosts):
   ```bash
   systemd-run --user --wait --pipe --collect --working-directory="$HOME/git/binary-algo" \
     --property=EnvironmentFile="$HOME/.config/binary-algo/deriv-sota-executor.env" -- \
     ~/binary-algo-venv/bin/python -B -E -s scripts/deriv_sota_executor.py \
       --seal-3book-cohort-cutoff \
       --cutoff-utc <RFC3339_WHOLE_SECOND_UTC> \
       --reason "pre_4book_cutover_owner_decision_15" \
       --account-mode demo \
       --credential-isolation-receipt ~/.config/binary-algo/sota-credential-isolation-receipt.json
   ```
4. Run strict 4-book cutover readiness preflight (`--enforce-4book-cutover --strict-rolling-vintage-state --live-smoke`) with 4-book `--book-ids`, `--rolling-vintage-dir deriv_data/runtime/rolling_vintages`, and `4/4/4` caps (auto-scales `--smoke-timeout-s` to `20.0s` covering `3 * 2.6s` pacing plus OTP/connect/proposal allowance; verifies `phase1_3book_cohort_cutoff` is sealed and no host rate-limit cooldown is active; passes if the rolling vintage is active OR currently `abstained` within `[scheduled_refit_epoch, next_refit_epoch)` and `[published_epoch, next_refit_epoch)`; fails closed if missing/expired/future/corrupt/tampered).
5. Register the prospective rotating-vintage `G6` cohort anchor for `USDJPY.sota_roll14d_m60_up.v1` via `--register-cohort-anchor` (`vintage_policy: "rotating_vintages"`).
6. Keep `~/.config/binary-algo/deriv-sota-executor.env` untouched at its 3-book/`3/3/3` baseline! Install `ops/deriv-sota-4book-cutover.conf` to `~/.config/systemd/user/deriv-sota-executor.service.d/20-4book-cutover.conf` (the drop-in pins the 4-book `--book-ids`, `--rolling-vintage-dir`, and `4/4/4` CLI arguments directly in `ExecStartPre`/`ExecStart`, overriding `EnvironmentFile=` cleanly while keeping `.env` untouched), install `ops/deriv-sota-rolling-refit.{service,timer}` into `~/.config/systemd/user/`, run `systemctl --user daemon-reload`, inspect `systemctl --user show deriv-sota-executor.service -p ExecStartPre,ExecStart,DropInPaths`, start `deriv-sota-executor.service`, and enable `deriv-sota-tick-archive.timer` and `deriv-sota-rolling-refit.timer`.

### Stage D — 4-Book Demo MVT Evaluation & Audited Rollback
- **Read-Only 4-Book Demo MVT Report:**
  ```bash
  ~/binary-algo-venv/bin/python -B -E -s scripts/deriv_live_pnl_ledger.py \
    --portfolio-mvt-report \
    --book-ids USDJPY.sota_ens5_m30_up.v1,USDJPY.sota_stack05_m30_up.v1,USDJPY.sota_ens5_m15_up.v1,USDJPY.sota_roll14d_m60_up.v1
  ```
  Requires a valid sealed `phase1_3book_cohort_cutoff`, admits ONLY `portfolio_mode == "4book"` fills started at/after the sealed cutover boundary into the 4-book portfolio pool (so neither pre-cutover 3-book static fills nor post-rollback 3-book fills can contaminate 4-book MVT, while keeping full per-book historical G6 evidence), and emits strictly Demo-only verdicts (`DEMO_4BOOK_FULL_G6_REVIEW_PENDING_OWNER`, `DEMO_4BOOK_MVT_EV_REVIEW_PENDING_OWNER`, `DEMO_4BOOK_MVT_MICRO_REVIEW_PENDING_OWNER`, `HOLD_COLLECTING`, `HALT_INVESTIGATE`, `DIAGNOSTIC_ONLY_NOT_PORTFOLIO_READY`) with `all_books_strict_g6_eligible: false` and `rolling_book_live_real_money_eligible: false`. Running the default 3-book `--portfolio-mvt-report` automatically caps evaluation at `phase1_3book_cohort_cutoff`.
- **Zero-Downtime Rolling Book Revocation / Full Rollback to 3-Book:**
  - Immediate durable 4th-book halt (zero restart of the 3 static books; remains halted across scheduled `--if-due` and `--force` runs until an explicit `--resume-after-revoke` refit is invoked):
    `~/binary-algo-venv/bin/python -B -E -s scripts/deriv_sota_rolling_refit.py --book-id USDJPY.sota_roll14d_m60_up.v1 --revoke-reason <reason>`
  - Full atomic revert to 3-book (`3/3/3`): stop `deriv-sota-rolling-refit.timer` & `deriv-sota-tick-archive.timer`, remove `~/.config/systemd/user/deriv-sota-executor.service.d/20-4book-cutover.conf` (which restores the untouched 3-book/`3/3/3` base `.env` and `ExecStart` atomically), `systemctl --user daemon-reload`, and restart `deriv-sota-executor.service` once open contracts settle.
