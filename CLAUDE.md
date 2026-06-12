# CLAUDE.md — agent entry point

You are working in **binary-algo**, a research repo that predicts **up/down price direction**
(and magnitude) for 7 USD FX pairs at the 15-minute horizon, targeting Deriv FX Rise/Fall
binaries. Read this first, then the files it points to. Keep it accurate as the repo evolves.

## Read in this order
1. **`README.md`** — the certified-model leaderboard (which model wins for each pair/side,
   where its weights live, how to load one), directory structure, and how to run scripts.
2. **`REPO_MAP.md`** — the filing convention (Tier-1 GENERIC vs Tier-2 KEY-SPECIFIC) that keeps
   the bookkeeping sane. Follow it when you add or move anything.
3. **`MODEL_REGISTRY.md`** + **`books/INDEX.json`** — every frozen book, keyed `<PAIR>.<book>.v1`,
   with provenance and recreate notes. `books/INDEX.json` is the machine-readable incumbent index.
4. The **`strategy-eval`** skill (`.claude/skills/strategy-eval/SKILL.md`) — invoke it for ANY task
   that evaluates / reverse-engineers / retargets / freezes a strategy, or writes a result into a
   `results/<PAIR>_RESULTS.md`. It wires the registry + methods catalog + evaluation discipline +
   results-writing protocol into one workflow.

## Where things live (2026-06-12 reorganisation)
| What | Where |
|---|---|
| Frozen certified models (deployable) | `books/<PAIR>.<book>.v1/` — `*_lgb.txt` weights + `*strategy.json` |
| Per-pair results ledgers (record of truth) | `results/<PAIR>_RESULTS.md` |
| Raw experiment JSON outputs | `results/json/` |
| All Python + shell scripts (flat) | `scripts/` |
| Methodology / findings / campaign docs | `docs/` |
| Sweep ledgers + backlogs per (pair, tf) | `sweeps/` |
| Literature corpus | `research/` |
| Logs / checkpoints | `logs/` |
| Feature data + `macro_calendar.parquet` | **repo root** (absolute paths hardcoded — do not move) |

## Operational rules (non-negotiable)
- **Run scripts from the repo root**, never from `scripts/`: `python scripts/<name>.py`. This is what
  makes `import harness`, `from sessions import ...` resolve (Python adds `scripts/` to `sys.path[0]`)
  and what makes the hardcoded `features/` absolute paths line up.
- Scripts write `*_result.json` to the **current working directory** (repo root). Move finished ones
  to `results/json/` when archiving.
- **Do not move** the root data dirs (`features*/`, `syn_data/`) or `macro_calendar.parquet` — scripts
  reference them by absolute (and sometimes relative) path. Moving them breaks the harness.
- Python env: `~/binary-algo-venv`; pinned deps in `ENVIRONMENT_libs.txt`.

## Research discipline (from REPO_MAP — keep results trustworthy)
- **Evidence-first.** Every factual claim traces to primary-source evidence (an on-disk result JSON /
  ledger line / code citation), never prose or memory. The full binding standard — tiered primary
  sources (T1–T4), pre-committed falsifiers, forbidden hedging, and the Blocked format — is
  **`docs/_EVIDENCE-FIRST.md`**; read and follow it.
- Every number traces to an on-disk result JSON / ledger line, never prose.
- **Tier-1 GENERIC docs must not bake in one key's numbers.** If a generic doc cites a result, tag it
  inline `[PAIR·tf]` and point to the Tier-2 file of record. Per-key numbers live only in
  `results/<PAIR>_RESULTS.md` / `sweeps/<PAIR>_<tf>.md`.
- **Never pool across timeframes**, and never copy one key's number into another key.
- Certification = refit-CPCV (15 purged paths, per-fold refit): **p10 ≥ 0.541** AND ≥ 80% of paths
  clear breakeven. Models gate on **NY session** (America/New_York 08:00–17:00, DST-correct).
  Breakeven **0.541** (Deriv even-payout). All books are **refit-dependent** (frozen vintages decay).

## Already exhausted — do NOT re-mine or rerun (all KILLED/NULL, Tier-1 confirmed)
- Neural + spectral direction forecasters (N-BEATS/N-HiTS/Autoformer/DLinear/FEDformer/TFT/DWT/SSA),
  bar-image CNNs, GRU/ESN, Kronos/Chronos TSFMs, Optuna tuning — see `scripts/README.md` "What to Ignore".
- Cross-pair POOLING for the own-pair family (USDJPY/USDCAD/AUDUSD/NZDUSD) — confirmed null per pair.
- Deriv synthetic indices — IID-Gaussian / memoryless, no edge (`docs/SYNTHETIC_RNG_FINDINGS.md`).
- All 7 pairs' 15m sweeps are CLOSED. Pushing any pair past its ceiling needs **external data**
  (macro/rates/flow), not more on-disk feature engineering. See each `results/<PAIR>_RESULTS.md`.

## When you finish a piece of work
Record per-key results in `results/<PAIR>_RESULTS.md` (+ the sweep ledger/backlog); record generic
methods/ideas in `docs/METHODS_CATALOG.md` / `docs/IDEAS_LOG.md` / `SWEEP_MATRIX.md`; freeze survivors
as a new `books/<PAIR>.<book>.v1` and register them in `MODEL_REGISTRY.md` + `books/INDEX.json`.
