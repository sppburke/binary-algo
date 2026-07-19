# scripts/

All Python experiment scripts and shell runners. Flat structure — no subdirectories — because scripts import each other directly (e.g. `import harness as H`, `from sessions import session_mask`).

**Run from the repo root, not from this directory:**
```bash
cd /home/sean/git/binary-algo
python scripts/usdchf_15m_cpcv_xpair.py
```

---

## Shared Modules (imported by many scripts)

| File | Purpose |
|---|---|
| `harness.py` | Train/val/test/OOS split logic, feature loading from `features/` |
| `sessions.py` | NY / LDN / Asia session mask (`session_mask(df, "ny")`) |
| `dataset.py` | Low-level data utilities |
| `cpcv_certify.py` | CPCV certification harness (15 purged paths, per-fold refit) |
| `crosspair.py` | EUR-bloc cross-pair feature construction |
| `manifest.py` | Book manifest creation and content_id hashing |
| `build_panel.py` | Builds `features/{PAIR}_{year}.parquet` from raw OHLCV |
| `build_books.py` | Freezes a trained model into `books/` |
| `fwd_holdout.py` | Forward holdout evaluation utilities |
| `research_campaign_v1.py` | Sealed, serial, resumable control plane for adapter-owned synthetic/retrospective campaigns; emits inactive evidence terminals only |

Run or verify a campaign from the repository root:

```bash
~/binary-algo-venv/bin/python scripts/research_campaign_v1.py run --spec <canonical-json-path>
~/binary-algo-venv/bin/python scripts/research_campaign_v1.py verify --campaign-id <id>
```

### USDCHF 15m current-vintage refit

`usdchf_m15_current_refit_v1.py` seals, fits, and verifies one fixed
latest-on-disk combined-policy refit. Its output belongs under
`results/artifacts/` as a tracked inactive research package, not under
`books/`; it publishes no efficacy measurement and performs no routing,
deployment, or activation. See
`docs/USDCHF_M15_CURRENT_REFIT.md` for the exact boundaries and lifecycle.

```bash
~/binary-algo-venv/bin/python scripts/usdchf_m15_current_refit_v1.py seal
~/binary-algo-venv/bin/python scripts/usdchf_m15_current_refit_v1.py fit --seal-id <id>
~/binary-algo-venv/bin/python scripts/usdchf_m15_current_refit_v1.py verify --seal-id <id>
```

---

## Script Naming Convention

### Pair-specific 15m scripts (the main body of work)
```
{pair}_15m_{method}.py
```
Examples:
- `usdchf_15m_cpcv_xpair.py` — USDCHF 15m cross-pair CPCV certification
- `gbpusd_15m_cpcv_xpair.py` — GBPUSD 15m cross-pair CPCV
- `audusd_15m_cpcv_session.py` — AUDUSD 15m session-segmented CPCV
- `nzdusd_15m_ticksettle.py` — NZDUSD tick settlement validation

Each script produces `{pair}_15m_{method}_result.json` in the repo root (move to `results/json/` when done).

### EURUSD horizon-specific scripts
```
m{minutes}_{method}.py   or   min{minutes}_{method}.py
```
Examples:
- `m5_xpair.py` — EURUSD 5m cross-pair
- `m10_cpcv_side.py` — EURUSD 10m per-side CPCV
- `m30_session.py` — EURUSD 30m session
- `min1_production.py` — EURUSD 1m production module (imported by other scripts)
- `min2_production.py` — EURUSD 2m production module

### Shared/generic scripts
```
{topic}.py
```
- `barcnn_*.py` — CNN bar-image experiments (all KILLED, do not rerun)
- `decomp_dir.py` — DLinear/Autoformer/FEDformer direction (KILLED)
- `nbeats_nhits_dir.py` — N-BEATS/N-HiTS direction (KILLED)
- `spectral_dir.py` — DWT/SSA direction (KILLED)
- `exp_*.py` — early EURUSD experiments (archived, superseded)

### Shell scripts
- `run_*.sh` — queue runners for long experiment batches
- `session_xpair_*.sh` — session cross-pair chain runners
- `kronos_*.sh` — Kronos-specific orchestration (deprecated)

---

## Common Patterns

### Per-fold refit CPCV (the gold standard)
```python
# Most pair-specific certification scripts follow this shape:
python scripts/{pair}_15m_cpcv_session.py ny 2 0.02 3
#                                          ^  ^  ^    ^
#                                     session  stride  cov  n_seeds
```

### Confidence curve (coverage vs win-rate tradeoff)
```python
python scripts/{pair}_15m_cpcv_{variant}.py
# Produces a multicov_result.json with p10/mean at cov 0.5/1/2/3/5%
```

### Seed ensemble
Appending `seedens{K}` to the script name or passing `K` as an argument runs K seeds and averages probabilities before applying the confidence gate.

---

## What to Ignore

These families were exhaustively tested and produced null results — do not attempt to improve or rerun:

- `barcnn_*.py` — CNN bar images (84 arms, all KILLED)
- `decomp_dir.py`, `nbeats_nhits_dir.py`, `spectral_dir.py` — neural forecasters (84 arms, all KILLED)
- `exp_v*.py`, `exp_15m_v*.py` — early iteration experiments (superseded)
- `*_xpair*.py` for USDJPY/USDCAD/AUDUSD/NZDUSD — pooling confirmed null for own-pair family
- `*_gru.py`, `*_optuna.py` — GRU sequences and hyperparameter tuning (KILLED universally)

See `docs/DIRECTION_FINDINGS.md` and the relevant `results/{PAIR}_RESULTS.md` for full verdicts.
