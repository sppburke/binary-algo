# results/

## Per-pair result ledgers

Each `{PAIR}_RESULTS.md` is the complete record of truth for that currency pair — every experiment run, the verdict (CERTIFIED / KILLED / SUBSUMED), the key numbers, and the deployment spec for the certified book.

| File | Pair | Sweep status |
|---|---|---|
| `EURUSD_RESULTS.md` | EURUSD | All tf closed |
| `USDJPY_RESULTS.md` | USDJPY | 15m closed |
| `AUDUSD_RESULTS.md` | AUDUSD | 15m closed |
| `GBPUSD_RESULTS.md` | GBPUSD | 15m closed |
| `USDCAD_RESULTS.md` | USDCAD | 15m closed |
| `USDCHF_RESULTS.md` | USDCHF | 15m closed |
| `NZDUSD_RESULTS.md` | NZDUSD | 15m closed |

## json/

Raw `*_result.json` files produced by experiment scripts. Each JSON is the machine-readable output of one script run — the corresponding `_RESULTS.md` file summarises the finding in human-readable form.

New experiment runs write their JSON to the **repo root** (scripts write to CWD). Move them here when archiving:
```bash
mv *_result.json results/json/
```
