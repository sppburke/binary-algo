> **SCOPE: GENERIC** — Deriv SOTA certified multi-horizon runtime deployment, monitoring, and promotion runbook.

# Deriv SOTA Certified Strategy VPS Deployment & Promotion Runbook

> [!CAUTION]
> **MANDATORY JURISDICTION BOUNDARY (MEXICO VPS ONLY — ZERO USA DERIV CONNECTIONS):**
> **Never run any live or public Deriv WebSocket / REST command (`deriv_sota_executor.py`, `deriv_sota_tick_stream.py`, `vps_preflight.py --live-smoke`, `run_lh16_yencross_certifier.py --download`, or ad-hoc `*.derivws.com` scripts) from the USA (`quantum`).** Every direct connection to Deriv servers must originate **exclusively from the Mexico VPS (`sean@64.177.80.63`, hostname `vultr`, accessed via `ssh -i ~/.ssh/id_personal -o IdentitiesOnly=yes -o BatchMode=yes sean@64.177.80.63`)** with `DERIV_HOST_JURISDICTION=mexico_vps` set in `~/.config/binary-algo/deriv-sota-executor.env`. The runtime enforces `assert_non_us_deriv_host()` fail-closed before opening any network socket.

This runbook covers deploying, verifying, monitoring, and promoting certified SOTA binary-options books on the Mexico Linux VPS (`sean@64.177.80.63`, `vultr`, `3.3 GiB` RAM — note `MemoryMax` in `ops/deriv-sota-executor.service` is capped at `1G` so it fits alongside OS buffers comfortably).

---

## 1. Safety & Certification Architecture

1. **Zero Uncertified Execution:** `scripts/sota_book_runtime.py` refuses to load any book missing cryptographic SHA-256 verification or passed `G1_cpcv`, `G2_placebo`, `G3_entry_delay`, `G4_sign_null`, and `G5_payout` (`PAYOUT-CLEARED`) certificates in `books/<BOOK_ID>/<BOOK_ID>.manifest.json`.
2. **Lifecycle Isolation (`demo` vs. `live`):**
   - Books with `lifecycle_status = "demo_venue_settled_canary"` (e.g., `[USDJPY·15m]` / `[USDJPY·30m]` in `results/USDJPY_RESULTS.md` while Look 2 remains sealed) execute **only** in `--account-mode demo` (`VRTC*`).
   - `--account-mode live` (`CR*` real money) loads **only** books with `lifecycle_status = "active"` and `G6_holdout_or_prospective.passed == true`.
3. **Dual Mathematical Proposal Gate (`deriv_runtime_core_v2.py`):**
   - `parse_quote_v2` rejects `NaN`, `±inf`, `<= 0`, booleans, or missing `ask_price`/`payout`.
   - Every live Deriv proposal must clear **both** the certified refit-CPCV `p10` floor (`effective_floor > ask / payout`) **and** row-calibrated expected value (`p_cal * (payout / ask) - 1 >= ev_margin`).
4. **Session, Venue Blackout & Haircut Guards:**
   - Gated to `America/New_York` `08:00–17:00` (`ZoneInfo("America/New_York")` handles EDT/EST transitions automatically).
   - Blocks `21:00:00–23:59:59 UTC` sub-5h FX venue blackout (`Owner Decision 14`).
   - Blocks `16:50 - H` NY rollover (`16:35:00 NY` for `15m`, `16:20:00 NY` for `30m`).
   - Drops `16:00–16:35 NY` payout-haircut bars (`~1.58` payout) when `drop_ny_16h = true`.
5. **Currency-Leg Concentration & Persisted Circuit Breakers (`deriv_sota_executor.py`):**
   - Enforces independent open-contract caps on **both** `USD` (`max_usd_factor_open=2`) and `JPY` (`max_jpy_factor_open=2`), preventing simultaneous correlated JPY exposure across `USDJPY`, `EURJPY`, `NZDJPY`, and `GBPJPY`.
   - Persists all contracts and realized cent P&L in WAL-mode SQLite (`deriv_data/runtime/sota_trade_queue_{demo,live}.sqlite`), enforcing `max_stake_usd`, `max_daily_loss_usd`, `max_drawdown_usd`, `max_open_contracts`, and automatic `KILL` switch tripwire on 5 consecutive reconciliation errors.

---

## 2. VPS Host Setup

```bash
# 1. Verify NTP clock sync (mandatory for exact minute-close bar alignment)
timedatectl status | grep -E "System clock synchronized|NTP service"

# 2. Clone repository and create isolated Python 3.12 virtualenv
mkdir -p ~/git
git clone git@github.com:sppburke/binary-algo.git ~/git/binary-algo
cd ~/git/binary-algo

python3 -m venv ~/binary-algo-venv
~/binary-algo-venv/bin/pip install --upgrade pip
~/binary-algo-venv/bin/pip install -r ENVIRONMENT_libs.txt

# 3. Configure private credentials outside the git tree
mkdir -p ~/.config/binary-algo && chmod 700 ~/.config/binary-algo
cp ops/deriv-sota-executor.env.example ~/.config/binary-algo/deriv-sota-executor.env
chmod 600 ~/.config/binary-algo/deriv-sota-executor.env
nano ~/.config/binary-algo/deriv-sota-executor.env
```

---

## 3. Preflight & Proposal-Only Smoke Verification

Always run offline tests, 1s-tick parity, and a read-only proposal check **before** enabling automatic buying:

```bash
cd ~/git/binary-algo

# 1. Run full positive + negative runtime safety suite
~/binary-algo-venv/bin/python scripts/test_deriv_sota_runtime.py

# 2. Run offline + read-only live proposal preflight (never sends buy)
set -a && source ~/.config/binary-algo/deriv-sota-executor.env && set +a
~/binary-algo-venv/bin/python scripts/vps_preflight.py --account-mode demo --live-smoke

# 3. Optional 15-minute dry-run proposal-only session during NY hours
~/binary-algo-venv/bin/python scripts/deriv_sota_executor.py \
  --run --account-mode demo --tick-feed-mode integrated \
  --dry-run-proposal-only --duration-seconds 900
```

---

## 4. Enabling Systemd User Services (Demo `VENUE_SETTLED` Stage)

```bash
# Enable lingering so user services stay active across SSH logouts and reboots
sudo loginctl enable-linger "$USER"

mkdir -p ~/.config/systemd/user
cp ~/git/binary-algo/ops/deriv-sota-executor.service ~/.config/systemd/user/
cp ~/git/binary-algo/ops/deriv-sota-tick-stream.service ~/.config/systemd/user/
systemctl --user daemon-reload

# Default integrated mode runs public 1s tick stream + credentialed executor in one loop:
systemctl --user enable --now deriv-sota-executor.service
# (Optional: if DERIV_SOTA_TICK_FEED_MODE=external is set, also enable deriv-sota-tick-stream.service)

# Inspect live status and structured JSONL logs
systemctl --user status deriv-sota-executor.service
tail -f ~/git/binary-algo/logs/sota_runtime/sota_executor_demo/*.jsonl
```

---

## 5. Monitoring Settled P&L & Promoting to Real-Money (`CR`)

### Step 5.1 — Check Per-Book Prospective Demo `VENUE_SETTLED` Gate

Promotion is evaluated strictly **per book** (`--book-id <BOOK_ID>`, never pooled across timeframes or books):

```bash
cd ~/git/binary-algo
~/binary-algo-venv/bin/python scripts/deriv_live_pnl_ledger.py \
  --db deriv_data/runtime/sota_trade_queue_demo.sqlite \
  --book-id USDJPY.sota_e49nys30_m30_up.v1 \
  --min-trades 50
```

Wait until `"promotion_eligible": true` (`n >= 50` settled trades for that specific `book_id`, two-sided 95% bootstrap lower bound `ev_ci95[0] > 0.0`, and zero `open`/`unknown`/`unresolved` records). For any `demo_venue_settled_canary` book (such as `[USDJPY·15m]` or `[USDJPY·30m]` in `results/USDJPY_RESULTS.md`), record the per-book settled ledger output as its `G6_holdout_or_prospective` certificate and update that book's lifecycle to `"active"` via `sota_book_freeze.py`.

### Step 5.2 — Switch Executor to Opt-In Real-Money (`CR`) Mode

1. Stop the demo service cleanly:
   ```bash
   systemctl --user stop deriv-sota-executor.service
   ```
2. Update `~/.config/binary-algo/deriv-sota-executor.env`:
   - `DERIV_ACCOUNT_ID=CRxxxxxxx` (USD real-money account)
   - `DERIV_PAT=<real_account_pat_token>`
   - `DERIV_ACCOUNT_MODE=live`
   - `DERIV_ALLOW_LIVE_REAL_MONEY=1`
   - Start with conservative caps (`DERIV_SOTA_STAKE=1.0`, `DERIV_SOTA_MAX_DAILY_LOSS_USD=25.0`, `DERIV_SOTA_MAX_DRAWDOWN_USD=50.0`).
3. Run the live read-only preflight check (zero buys):
   ```bash
   set -a && source ~/.config/binary-algo/deriv-sota-executor.env && set +a
   ~/binary-algo-venv/bin/python scripts/vps_preflight.py --account-mode live --live-smoke
   ```
4. Start the live executor:
   ```bash
   systemctl --user start deriv-sota-executor.service
   ```

### Emergency Stop (`KILL` Switch)

To immediately halt all new proposals and buys while letting open contracts settle cleanly:

```bash
touch ~/git/binary-algo/logs/sota_runtime/KILL
```
