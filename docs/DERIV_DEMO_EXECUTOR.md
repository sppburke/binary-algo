# Deriv Demo Executor

`scripts/deriv_demo_executor.py` runs the six deployable non-EURUSD 15m books against Deriv Options demo infrastructure:

```text
USDJPY, USDCAD, AUDUSD, NZDUSD, USDCHF, GBPUSD
```

`EURUSD` is disabled until a verified live provider can produce finite `OF_*` order-flow columns. `--pairs all-enabled` means the six-pair set above. `--pairs EURUSD --public-smoke` fails closed.

This is operational demo tooling. It does not certify Deriv-native edge, does not enable real-money execution, and does not replace the frozen-book registry.

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

The executor requires 15m `CALL` and `PUT` availability from `contracts_for`, a finite feature row matching the frozen book schema, the 08:00-17:00 America/New_York session gate, the frozen strategy confidence gate, no `logs/paper_trades/KILL` file, and a healthy rolling store when `--store-dir` is set.

`logs/paper_trades/executor_state.json` stores open demo contracts with pair, proposal id, contract id, stake, expected expiry, status, and last platform update. On `--demo-buy` startup the executor queries Deriv for every pending local contract and refuses new buys unless all pending state reconciles to a terminal platform status.

A single-instance lock is created under the log directory per account/log-dir key. A second executor with the same key fails closed.

Demo-buy mode monitors `proposal_open_contract` until terminal status, writes `contract_update` and `contract_closed` JSONL events, then forgets the subscription. Proposal-only mode may request proposals but must not call `buy` or emit `buy_submitted` / `buy_confirmed`.

## VPS Files

Templates live under `ops/`:

```text
ops/deriv-demo-executor.env.example
ops/deriv-demo-executor.service
ops/deriv-demo-executor.timer
ops/deriv-demo-logrotate
```

Copy the env template outside git, fill only VPS-local secrets, then install the service/timer with paths adjusted if the checkout or venv differs. Stop with `systemctl --user stop deriv-demo-executor.timer deriv-demo-executor.service`. Emergency stop is `touch logs/paper_trades/KILL`.

Secrets, account IDs, JSONL logs, and `deriv_data/` artifacts must stay out of git.
