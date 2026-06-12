# Deriv Demo Executor

`scripts/deriv_demo_executor.py` runs the seven certified 15m FX books against Deriv Options demo infrastructure. It is demo-buy only; no real-money path exists in this repo.

## Environment

Required for `--demo-buy` and `--auth-smoke`:

```bash
export DERIV_APP_ID=...
export DERIV_PAT=...
export DERIV_ACCOUNT_ID=...
```

The PAT must have Options API trading permission. The executor obtains an authenticated demo WebSocket URL through Deriv's OTP endpoint and refuses to buy unless the returned URL is a demo endpoint.

## Commands

Book and replay schema check:

```bash
~/binary-algo-venv/bin/python scripts/deriv_demo_executor.py --check-books --replay-schema-check
```

Public market-data and live feature check:

```bash
~/binary-algo-venv/bin/python scripts/deriv_demo_executor.py --pairs USDCHF --public-smoke
```

Proposal-only dry run, the default execution mode:

```bash
~/binary-algo-venv/bin/python scripts/deriv_demo_executor.py --pairs USDCHF --dry-run-proposal-only --once
```

Opt-in demo buy with a tiny stake:

```bash
~/binary-algo-venv/bin/python scripts/deriv_demo_executor.py --pairs USDCHF --demo-buy --stake 1 --once
```

## Safety Gates

Before any buy, the executor requires:

- a demo WebSocket URL from the Deriv OTP flow;
- live `contracts_for` availability for 15m `CALL` and `PUT` on the mapped FX symbol;
- a finite feature row matching the frozen book's exact model columns and LightGBM feature count;
- the DST-correct NY session gate from `scripts/sessions.py`;
- the frozen strategy confidence/compression gate;
- append-only JSONL logging under `logs/paper_trades/YYYY-MM-DD.jsonl`;
- no `logs/paper_trades/KILL` file.

The executor logs skipped signals and proposal errors. Platform-settled `proposal_open_contract` status is the outcome of record for demo buys; demo logs are not performance claims.

## Known Live-Feature Constraint

The EURUSD xpair book includes `OF_*` order-flow features from the frozen strategy. The live feature builder only scores that book when those columns can be produced as finite values. If Deriv market data cannot supply enough tick detail to populate them, the executor fails closed instead of substituting a silent approximation.

