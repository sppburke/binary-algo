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
systemctl --user mask deriv-sota-executor.service deriv-trade-executor.service deriv-runtime-supervisor.service deriv-demo-executor.service deriv-demo-executor.timer
```

### Step 3 — Revoke Legacy Credentials & Establish Dedicated Demo Credentials + Receipt FIRST
Establish dedicated Demo credentials and the human-attested non-secret credential isolation receipt **BEFORE** any first-use or adoption command:
```bash
mkdir -p ~/.config/binary-algo && chmod 700 ~/.config/binary-algo
cp ops/deriv-sota-executor.env.example ~/.config/binary-algo/deriv-sota-executor.env
chmod 600 ~/.config/binary-algo/deriv-sota-executor.env
# Populate dedicated VRTC* credentials (never print or share private env file)
# Revoke legacy Deriv PATs in the Deriv dashboard, then write non-secret receipt:
# ~/.config/binary-algo/sota-credential-isolation-receipt.json (schema: sota-credential-isolation-receipt/v1)
```

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
systemctl --user show deriv-sota-executor.service -p ExecStartPre,ExecStart,Environment,UnsetEnvironment,DropInPaths,ConditionPathExists
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
