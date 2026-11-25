# binary-algo — FX Binary Direction Prediction

> **CRITICAL JURISDICTION & NETWORK RULE (DO NOT VIOLATE):**
> **NEVER contact any Deriv server (`*.derivws.com`, `*.deriv.com`, public or authenticated WebSockets, or REST endpoints) directly from the USA / local research workstation (`quantum`).**
> **ALL direct network interaction with Deriv servers—including public tick/candle downloads, quote surveys, contract catalog queries, proposal smoke checks, Demo (`VRTC*`) execution, and Live (`CR*`) trading—MUST run EXCLUSIVELY on the Mexico VPS (`sean@64.177.80.63`, hostname `vultr`, via `ssh -i ~/.ssh/id_personal -o IdentitiesOnly=yes -o BatchMode=yes sean@64.177.80.63`, with `DERIV_HOST_JURISDICTION=mexico_vps`).** Local US machines are strictly for offline research, training, and certification on archived data synced from the Mexico VPS.

Predicts **up/down price direction** on Deriv FX Rise/Fall binaries (ties-lose). For current Deriv-native 1-second tick research, certified zero-pickle SOTA books, gap-closure audits, and Mexico-VPS deployment, start with the **2026-11 Deriv-Feed SOTA Certification** section below; the 2026-06 Dukascopy 10s-bar leaderboard is retained beneath it as historical v1 reference.

---

## 2026-11 Deriv-Feed SOTA Certification, Gap-Closure & Profitability Ranking

Full cross-contract ranking, falsification ledger, compounding study, and Mexico-VPS runbook live in [docs/DERIV_PROFITABILITY_RANKING.md](docs/DERIV_PROFITABILITY_RANKING.md), [SOTA_MODEL_REGISTRY.md](SOTA_MODEL_REGISTRY.md), [books/SOTA_INDEX.json](books/SOTA_INDEX.json), and [results/USDJPY_RESULTS.md](results/USDJPY_RESULTS.md). All DEV evaluation uses native Deriv 1-second ticks `< 2026-08-01T00:00:00Z` (sealed holdout `>= 2026-08-01T00:00:00Z` untouched) with zero-pickle NumPy runtime hardened in `scripts/deriv_sota_executor.py`, `scripts/deriv_sota_tick_stream.py`, `scripts/sota_book_runtime.py`, `scripts/vps_preflight.py`, and `scripts/deriv_live_pnl_ledger.py`.

### Certified Frozen Demo Canary Books (`books/SOTA_INDEX.json`, `G1–G5` DEV Pass Under Drift Amendment, `G6` Pending)

| Rank | Book ID (`content_id`) | Key & Contract | Trades/Yr | Payout | `p10_wr` / `p10_ev` | Path-`p10` Floor Log-Growth (`$/Yr` per `$1k`) | Tier-2 Ledger & Dossier |
|---:|---|---|---:|---:|---|---|---|
| **`#1`** | `USDJPY.sota_ens5_m30_up.v1` (`c8d0b13768a4c649`) | `[USDJPY·30m]` `CALL` (`risefall`, `cov=0.03`, 5-seed RBF Nyström) | `316` | `1.8200` (flat `0.00` skew) | `0.6031` / `+0.0976` | **`+0.6495/yr`** (`+$914.67/yr`, `maxDD 31.56%`) | [USDJPY_RESULTS.md](results/USDJPY_RESULTS.md), [SOTA_MODEL_REGISTRY.md](SOTA_MODEL_REGISTRY.md), [5-gate JSON](results/json/deriv_5gate_USDJPY_sota_ens5_m30_up_v1_result.json) |
| **`#4`** | `USDJPY.sota_ens5_m15_up.v1` (`51b8c7fd6d179b3f`) | `[USDJPY·15m]` `CALL` (`risefall`, `cov=0.015`, 5-seed RBF Nyström) | `274` | `1.7764` (NY mean) | `0.5763` / `+0.0238` | **`+0.0494/yr`** (`+$50.63/yr`, `maxDD 8.01%`) | [USDJPY_RESULTS.md](results/USDJPY_RESULTS.md), [SOTA_MODEL_REGISTRY.md](SOTA_MODEL_REGISTRY.md), [5-gate JSON](results/json/deriv_5gate_USDJPY_sota_ens5_m15_up_v1_result.json) |

*(Legacy 1-seed baselines `USDJPY.sota_e49nys30_m30_up.v1` and `USDJPY.sota_c2nysabs_m15_up.v1` remain in `books/SOTA_INDEX.json` for reproducibility and are superseded by the two 5-seed books above; deploy with `--book-ids USDJPY.sota_ens5_m30_up.v1,USDJPY.sota_ens5_m15_up.v1`.)*

### Closed Real-Market Sweeps, Conditional Candidate & Compounding Study

- **Exhausted contract families & instruments (`0` certified survivors):**
  - Continuous Multipliers (`MULTUP`/`MULTDOWN`, `879` scored rows across FX, Metals, Crypto, `WLD*` baskets): `0` survivors (`results/json/deriv_priority123_grand_sweep_result.json`, `deriv_novel_levers_and_mult_tp_result.json`, `deriv_round2_discovery_and_jpy_basket_result.json`, `deriv_multiasset_candles_and_crypto_mult_result.json`).
  - `1d` Volatility Barriers (`RANGE`, `EXPIRYRANGE`, `ONETOUCH`, `NOTOUCH`, `EXPIRYMISS`, `UPORDOWN`, `450` exact broker-IV evaluations + `12` Gold/Silver touch cells) and `1d` `HIGHER`/`LOWER` offset barriers (`30` cells): `0` survivors (`deriv_barrier_1d_exact_quoterv_audit_result.json`, `deriv_gap_closure_barriers_hl_result.json`, `deriv_mexico_vps_gap_probe_result.json`).
  - Non-JPY FX majors (`126` rows), `16` G10 crosses (`[EURJPY·30m/60m]` and `[AUDNZD·120m]` falsified on native venue ticks; `[GBPJPY·120m]` blocked on `< 120` cached venue tick days), `12` OTC stock indices (`144` cash-session rows), non-NY FX sessions (`174` Tokyo/London/All-Open rows), `240m`/`1d` FX & Metals, and `4h`/`8h` `WLD*` baskets: `0` certified (`deriv_gap_closure_sweep_result.json`, `deriv_gap_closure_falsifiers_result.json`).
- **Conditional unfrozen candidate (`[USDJPY·60m]` `CALL` @ `1.82`):** Failed static `2025 -> 2026` freeze (`EV -0.0563`), passed all `4` start-date shifts under bi-weekly (`14d`) trailing-`120d` refitting (`wr = 0.6230–0.6931`, `ev = +0.1339..+0.2614`, rank `9/9` vs week-shift nulls in `results/json/deriv_gap_closure_falsifiers_result.json`; Tier-2 [USDJPY_RESULTS.md](results/USDJPY_RESULTS.md)). Adding `[USDJPY·60m]` refit paths alongside `[USDJPY·30m]` + `[USDJPY·15m]` across all `15` shared CPCV paths at `0.25×` Kelly (`jpy_cap = 3`) lifts portfolio frequency from `316` to `760` tr/yr and floor log-growth from `+0.6504` to `+0.9024/yr` (`maxDD 33.0%`; `[USDJPY·30m]` + `[USDJPY·60m]` at `jpy_cap = 2` reaches `+0.9539/yr` at `maxDD 28.1%`, `results/json/deriv_compounding_and_frequency_study_result.json`).
- **Post-G6 frequency & compounding study:** Archived in [deriv_compounding_and_frequency_study_result.json](results/json/deriv_compounding_and_frequency_study_result.json) and summarized in [docs/DERIV_PROFITABILITY_RANKING.md](docs/DERIV_PROFITABILITY_RANKING.md) §2 (Kelly fraction scaling, 3-book portfolio expansion, staggered `15m` `0.65×` lanes, `14d` rolling-refit frequency lift, and coverage sensitivity).

---

## Certified Model Leaderboard (Legacy 2026-06 Dukascopy 10s-Bar v1 Reference)

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
import lightgbm as lgb, json, glob, numpy as np

book_dir = "books/USDCHF.m15ny_xpair_seedens.v1"
# strategy.json filename varies per book — glob for it (one per book dir)
strategy = json.load(open(glob.glob(f"{book_dir}/*strategy*.json")[0]))

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
├── MODEL_REGISTRY.md       ← legacy v1 frozen books (byte-frozen for v1 cryptographic seals)
├── SOTA_MODEL_REGISTRY.md  ← Deriv-feed 1s-tick SOTA certified books & cross-contract summary
├── SWEEP_MATRIX.md         ← all (pair, tf, side) combinations and sweep status
├── REPO_MAP.md             ← file-naming conventions and organisation rules
├── ENVIRONMENT_libs.txt    ← research-stack dependency record
├── requirements-deriv-demo.txt ← pinned Deriv demo executor runtime deps
│
├── books/                  ← frozen certified models (one subdir per book)
│   ├── SOTA_INDEX.json                  ← active Deriv-feed zero-pickle SOTA book index
│   ├── USDJPY.sota_ens5_m30_up.v1/      ← SOTA #1 frozen Demo canary (30m flat 1.82)
│   ├── USDJPY.sota_ens5_m15_up.v1/      ← SOTA #4 frozen Demo canary (15m cov=0.015)
│   ├── USDCHF.m15ny_xpair_seedens.v1/   ← legacy Dukascopy v1 books below
│   ├── GBPUSD.m15ny_xpair_seedens8.v1/
│   ├── USDJPY.m15ny_seedens.v1/
│   ├── USDCAD.m15ny_seedens.v1/
│   ├── AUDUSD.m15ny_seedens.v1/
│   ├── NZDUSD.m15ny_seedens.v1/
│   ├── EURUSD.m15xp.v1/
│   └── ... (older/superseded EURUSD books for 1m/2m/5m/10m/30m)
│
├── scripts/                ← all Python scripts and shell runners (flat)
│   ├── deriv_sota_executor.py    ← Mexico-VPS SOTA Demo/Live executor (EV + concurrency gated)
│   ├── deriv_sota_tick_stream.py ← 1s tick stream & bit-exact feature parity builder
│   ├── sota_book_runtime.py      ← zero-pickle NumPy SOTA inference loader
│   ├── deriv_live_pnl_ledger.py  ← per-book G6 prospective Demo promotion gate
│   ├── vps_preflight.py          ← offline + Mexico-VPS live-smoke preflight checker
│   ├── build_certified_sota_books.py ← deterministic SOTA book freezer (cutoff < 2026-08-01)
│   ├── harness.py          ← SHARED: train/val/test/OOS split + feature loading
│   ├── sessions.py         ← SHARED: NY/LDN/Asia session mask
│   ├── dataset.py          ← SHARED: data utilities
│   ├── cpcv_certify.py     ← SHARED: CPCV + deflation experiment (8 groups, 28 paths)
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

## Package Manager / Pinned Runtime

The package-manager path for the Deriv demo executor is `uv` with a pip-compatible pinned requirements file:

```bash
# Tooling versions verified in this checkout on 2026-06-12:
# uv==0.11.7
# pip==24.0 inside ~/binary-algo-venv

uv venv ~/binary-algo-venv --python 3.12
uv pip sync requirements-deriv-demo.txt --python ~/binary-algo-venv/bin/python
```

Fallback if `uv` is unavailable:

```bash
~/binary-algo-venv/bin/python -m pip install -r requirements-deriv-demo.txt
```

`requirements-deriv-demo.txt` pins the Deriv runtime packages used by `scripts/deriv_client.py`, `scripts/book_runtime.py`, `scripts/live_features.py`, `scripts/deriv_demo_executor.py`, `scripts/deriv_backfill.py`, and `scripts/deriv_train.py`. `ENVIRONMENT_libs.txt` remains the broader research-stack dependency record.

For VPS demo deployment, see `docs/DERIV_DEMO_EXECUTOR.md` and the placeholder-only `ops/` systemd/logrotate templates. The old per-minute executor timer is deprecated (issue #5): production-store freshness is owned by `ops/deriv-production-candle-refresh.timer` (health contract `--max-stale-seconds 600`) and `quote_snapshot` coverage by `ops/deriv-quote-audit.service` (`deriv_quote_workers.py --audit`, buy-incapable). The executor default pair set is `all-enabled`, which excludes EURUSD until verified live `OF_*` columns exist. Runtime uses one combined direction book per pair with side-specific payout floors: a buy requires `side_refit_p10 − haircut_abs − payout_edge_margin > ask/payout`, with coverage, floors, and haircuts resolved fail-closed from the registry by `scripts/deriv_floor_resolver.py` (floors are refit-CPCV per-era floors; frozen vintages decay, so demo-only until a retrain cadence exists).

---

## Deriv-Native Data

Deriv candle backfills are isolated under gitignored `deriv_data/`; they do not
replace or mutate the existing Dukascopy-derived `features/` store or any frozen
book in `books/`.

Run on the Mexico VPS only; Deriv connections never originate from the workstation (`AGENTS.md`). The default path
is the production store the refresh timer owns, so run this before that timer is enabled or pass a separate `--out-dir`:

```bash
~/binary-algo-venv/bin/python scripts/deriv_backfill.py --pairs all --granularity 60
```

The default store is `deriv_data/candles_1m/`. See
`docs/DERIV_DATA_STORE.md` for the artifact layout and isolation rules. Any
experimental Deriv-trained model must be clearly named with a `deriv_` prefix
and written under `deriv_data/models/`. Deriv experiment tooling holds out the
latest two calendar months of Deriv data by default:

```bash
~/binary-algo-venv/bin/python scripts/deriv_train.py --pairs all --horizon 15 --holdout-months 2
```

To attempt a schema-faithful incumbent-book replication on Deriv candle data:

```bash
~/binary-algo-venv/bin/python scripts/book_deriv_replicate.py --pairs all --cpcv candidates
```

That run writes tracked evidence to `results/json/book_deriv_replicate_*_result.json`
and ignored model artifacts under `deriv_data/models/`.

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
