# binary-algo — FX Binary Direction Prediction

Predicts **up/down price direction** for 7 USD FX pairs at the 15-minute horizon, targeting Deriv FX Rise/Fall binaries (even-payout, breakeven = **0.541**). All models are trained on 10s OHLCV bars + tick-derived features, 2012–2026.

---

## Certified Model Leaderboard

All 7 pairs certified as of 2026-06-11. All sweeps closed. Models gate on **NY session only** (08:00–17:00 America/New_York, DST-correct) and are **refit-dependent** — periodic retraining is required; frozen vintage books decay toward breakeven by 2026 OOS.

**Win-rate p10** = worst-path 10th-percentile across 15 CPCV refit paths. Breakeven = 0.541.

### Tier 1 — EUR-bloc (cross-pair pooling; AUC ~.547–.558)

| Pair | UP p10 | DOWN p10 | Cov gate | Book |
|---|---|---|---|---|
| **USDCHF** | **.6935** @cov1 / .6533 @cov2 | **.6682** @cov1 / .6440 @cov2 | 1% best | `books/USDCHF.m15ny_xpair_seedens.v1` `eb44d999` |
| **GBPUSD** | **.6552** @cov1 / .6251 @cov2 | **.6395** @cov1 / .6255 @cov2 | 1% best | `books/GBPUSD.m15ny_xpair_seedens8.v1` `9ee9a4634a6b73e0` |
| **EURUSD** | .5845 @~cov5 NY | .5712 @~cov5 NY | ~5% NY | `books/EURUSD.m15xp.v1` |

CHF and GBP use EUR-bloc cross-pair pooling (EURUSD + GBPUSD + USDCHF features trained jointly; USDCHF features are sign-flipped since CHF ≈ −EUR). EURUSD is the base case, cross-pair USD-residual features.

### Tier 2 — Own-pair family (AUC ~.531–.539; >65% floor requires external data)

| Pair | UP p10 | DOWN p10 | Cov gate | Book |
|---|---|---|---|---|
| **USDJPY** | .6005 @cov2 | .5738 @cov2 | 2% | `books/USDJPY.m15ny_seedens.v1` |
| **USDCAD** | .604 @cov1 / .5968 @cov2 | .5814 @cov5 / .5770 @cov2 | 1–2% | `books/USDCAD.m15ny_seedens.v1` `ddb4a78c` |
| **AUDUSD** | .596 @cov2 | .596 @cov2 | 2% | `books/AUDUSD.m15ny_seedens.v1` `9b0e0ed3` |
| **NZDUSD** | .5749 @cov2 | .5803 @cov2 | 2% | `books/NZDUSD.m15ny_seedens.v1` `f599708e` |

These pairs use own-pair features only — cross-pair pooling was tested and confirmed null for each.

**Kelly sizing:** 1/8-Kelly recommended. At the p10 floor: EUR-bloc @cov1 ≈ 3–4% per trade; own-pair @cov2 ≈ 1% per trade.

---

## What's in `books/`

Each book directory is a **frozen, self-contained, deployable model**:

- `*_lgb.txt` — saved LightGBM model weights (one file per seed)
- `*_strategy.json` — confidence threshold, session gate, coverage rule, content_id

All 7 certified 15m books have saved weights. Load without retraining:

```python
import lightgbm as lgb, json, numpy as np

book_dir = "books/USDCHF.m15ny_xpair_seedens.v1"
strategy = json.load(open(f"{book_dir}/m15ny_xpny_USDCHF_seedens_strategy.json"))

# Load all seeds and average probabilities
seed_files = sorted(glob.glob(f"{book_dir}/*_lgb.txt"))
models = [lgb.Booster(model_file=f) for f in seed_files]
proba = np.mean([m.predict(X) for m in models], axis=0)

# Apply coverage gate from strategy.json
conf = np.abs(proba - 0.5)
gate = strategy["conf_threshold"]   # fit on VAL NY worst-half
trade = conf >= gate                 # only trade when confident
direction = proba[trade] > 0.5      # True = UP (Rise), False = DOWN (Fall)
```

See `results/{PAIR}_RESULTS.md` for the deployment spec and full provenance for each book.

---

## Directory Structure

```
binary-algo/
│
├── README.md               ← this file — start here
├── MODEL_REGISTRY.md       ← all frozen books, content_ids, status (active/superseded)
├── SWEEP_MATRIX.md         ← all (pair, tf, side) combinations and sweep status
├── REPO_MAP.md             ← file-naming conventions and organisation rules
├── ENVIRONMENT_libs.txt    ← pip dependency list
│
├── books/                  ← frozen certified models (one subdir per book)
│   ├── USDCHF.m15ny_xpair_seedens.v1/   ← best: >65% both sides
│   ├── GBPUSD.m15ny_xpair_seedens8.v1/  ← EUR-bloc K=8
│   ├── USDJPY.m15ny_seedens.v1/
│   ├── USDCAD.m15ny_seedens.v1/
│   ├── AUDUSD.m15ny_seedens.v1/
│   ├── NZDUSD.m15ny_seedens.v1/
│   ├── EURUSD.m15xp.v1/
│   └── ... (older/superseded EURUSD books for 1m/2m/5m/10m/30m)
│
├── scripts/                ← all Python scripts and shell runners (446 files, flat)
│   ├── harness.py          ← SHARED: train/val/test/OOS split + feature loading
│   ├── sessions.py         ← SHARED: NY/LDN/Asia session mask
│   ├── dataset.py          ← SHARED: data utilities
│   ├── cpcv_certify.py     ← SHARED: CPCV certification harness
│   ├── crosspair.py        ← SHARED: EUR-bloc cross-pair feature construction
│   ├── manifest.py         ← SHARED: book manifest / content_id
│   ├── build_panel.py      ← builds feature parquets from raw OHLCV
│   ├── build_books.py      ← freezes trained models into books/
│   ├── {pair}_15m_*.py     ← pair-specific experiment scripts (audusd_, gbpusd_, etc.)
│   ├── m5_*.py             ← EURUSD 5m experiments
│   ├── m10_*.py            ← EURUSD 10m experiments
│   ├── m30_*.py            ← EURUSD 30m experiments
│   ├── min1_*.py           ← EURUSD 1m experiments
│   ├── exp_*.py            ← early EURUSD experiments (archived, do not rerun)
│   ├── barcnn_*.py         ← CNN image-based experiments (all KILLED)
│   └── *.sh                ← shell orchestration / queue runners
│
├── results/                ← experiment outputs and pair result ledgers
│   ├── EURUSD_RESULTS.md   ← full results ledger: every experiment, verdict, numbers
│   ├── USDJPY_RESULTS.md
│   ├── AUDUSD_RESULTS.md
│   ├── GBPUSD_RESULTS.md
│   ├── USDCAD_RESULTS.md
│   ├── USDCHF_RESULTS.md
│   ├── NZDUSD_RESULTS.md
│   └── json/               ← raw *_result.json files from every experiment run
│
├── sweeps/                 ← ordered experiment ledgers per (pair, timeframe)
│   ├── EURUSD_15m.md       ← step-by-step experiment log with verdicts
│   ├── EURUSD_15m_backlog.md
│   └── ... (one .md + one _backlog.md per pair+tf)
│
├── research/               ← literature review (16 topic files + index)
│   └── 00-INDEX.md
│
├── docs/                   ← methodology, findings, and campaign notes
│   ├── DIRECTION_FINDINGS.md     ← distilled direction-edge findings across all pairs
│   ├── MAGNITUDE_FINDINGS.md     ← magnitude prediction findings
│   ├── EXPERIMENT_LEDGER.md      ← master ledger of all experiments
│   ├── METHODS_CATALOG.md        ← all methods tried with verdicts
│   ├── THEORY.md                 ← theoretical grounding and signal limits
│   ├── DERIV_MAGNITUDE_MONETIZATION.md  ← magnitude monetisation analysis
│   ├── SYNTHETIC_RNG_FINDINGS.md ← Deriv synthetics are IID — do not re-chase
│   └── ...
│
├── logs/                   ← experiment stdout logs (*.log, *.txt) and checkpoints (*.pt)
│
└── [data dirs at root — absolute paths hardcoded in scripts, do not move]
    ├── macro_calendar.parquet  ← macro event calendar (both absolute + relative refs in scripts)
    ├── features/           ← main feature parquets: {PAIR}_{year}.parquet (2012–2026)
    ├── features_of/        ← order-flow features
    ├── features_tick/      ← tick-derived features
    ├── features_tick_cks/  ← CKS signed order-flow
    ├── features_tick_whale/← large-order features
    └── syn_data/           ← synthetic pair data
```

---

## Running Scripts

**Always run from the repo root.** Scripts use absolute paths for `features/` data and write `*_result.json` outputs to the current working directory:

```bash
cd /home/sean/git/binary-algo

# Run any experiment script
python scripts/usdchf_15m_cpcv_xpair.py

# Result lands at: ./usdchf_15m_cpcv_xpair_ny_multicov_result.json
# Move to results/json/ when done: mv *_result.json results/json/
```

Python imports work automatically — `python scripts/foo.py` adds `scripts/` to `sys.path[0]`, so `import harness`, `import sessions`, etc. resolve correctly.

---

## Data Requirements

Feature parquets at `features/{PAIR}_{year}.parquet` (2012–2026) are pre-built and present. To rebuild from raw source:

```bash
# Source 10s OHLCV bars at: /home/sean/git/processed/{PAIR}/
python scripts/build_panel.py
```

Raw sub-second tick files are **not currently on disk** — the 13 tick-level scripts (`*_ticksettle.py` etc.) are dormant until raw ticks are restored to disk.

---

## Key Methodology

| Concept | Detail |
|---|---|
| Label | Fixed 15m sign of return from bar close; ties LOSE (Deriv-faithful) |
| Evaluation | Refit-CPCV: 15 purged paths, per-fold model refit |
| Certification | p10 ≥ 0.541 AND ≥ 80% of 15 paths clear breakeven |
| Session gate | NY only (America/New_York 08:00–17:00, DST-correct) |
| Coverage gate | Confidence = \|p − 0.5\|; trade only top-N% confident bars |
| Leakage checks | Label-shuffle null + frozen-forward adversarial + tick settlement |
| Deployment | Deriv FX Rise/Fall; 15m = minimum expiry; even payout |

Full methodology: `docs/DIRECTION_FINDINGS.md`, `docs/THEORY.md`.
Per-pair narratives and full result tables: `results/{PAIR}_RESULTS.md`.
