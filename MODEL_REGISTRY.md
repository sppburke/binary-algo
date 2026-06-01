# Model Registry — frozen books, manifests, and how to recreate them verbatim

This registry makes every deliverable model **concrete, uniquely referenceable, and recreatable verbatim**. It is the resolution target for the `(currency, timeframe, side)` keys in `EURUSD_RESULTS.md`.

> **Future agents:** to evaluate/reverse-engineer a strategy for any (currency, timeframe) and record it correctly, invoke the **`strategy-eval`** skill (`.claude/skills/strategy-eval/SKILL.md`). It wires this registry + `METHODS_CATALOG.md` + the evaluation discipline + the `<PAIR>_RESULTS.md` writing protocol into one workflow.

## Latest additions (2026-06-01 — EURUSD 5m sweep + edge-improvement loop)
- **`EURUSD.m5xp.v1` — the (5m,UP) certified edge** (side-split of the frozen cross-pair book). UP forward .605/.577/.615; survives the FULL per-fold-refit CPCV at the ~5% operating gate (cov0.05 p10 **0.553**, 96% of purged folds clear) — the first sub-15m direction edge to survive the full refit. (5m,DOWN) = marginal/dead (refit p10 .542 barely, forward 2025 .538 fails). Deployable ~0.55–0.57, ⅛-Kelly. `m5_updown_result.json`, `m5_refit_tightcov_result.json`, `sweeps/EURUSD_5m.md`.
- **`EURUSD.m5xp_aci.v1` — NEW edge-IMPROVEMENT (the win from digging deeper).** Same m5xp primary+meta bytes (`depends_on: EURUSD.m5xp.v1`) + an ONLINE **adaptive-conformal (ACI) gate** targeting win-rate w*=0.57 (causal; updates the threshold on past outcomes only). Beats the fixed gate on the binding 2025 year (**.584 at n764 vs .579 at n618 — better win AND coverage**), ~36% more total trades, tighter cross-regime spread → better EV/time + regime-robustness. The improvement is in the GATE POLICY, not new weights. Status: within-experiment vs fixed; CPCV-validation of the adaptive policy is the open follow-up. `m5_conformal.py`, `m5_conformal_result.json`. (Other levers: EXP-1 magnitude-weighting rebalanced to two-sided ~.56, no UP win; EXP-2 seed-ensemble MLP decorrelated corr .694 but 50/50 blend ≈ GBM — info-bound.)

## The core decision: FREEZE-the-artifact + RECORD-a-manifest (not config-driven re-derivation)

The frozen books are gradient-boosted ensembles (LightGBM/XGBoost/CatBoost) trained **multithreaded (`n_jobs/thread_count=20`) with no seed on the estimators**. That makes them **NOT bit-for-bit retrainable** — re-running the training script yields a *different* model. Therefore:

- **The persisted artifact IS the model** — the only verbatim copy. We version-control it.
- **The manifest is the model's birth certificate** — a RECORD (written at freeze time), not a DRIVER (read by training). It binds the stable id to its provenance so you know exactly what produced it and can verify the bytes.
- **Config files were rejected** as the primary mechanism: a config pins *inputs to training*, not the *output model*; for non-deterministic GBMs that does not give verbatim recreation. (Determinism flags — `deterministic=True`, `force_row_wise`, fixed seed/threads — are worth adding to *future* trainers, but only as a bonus on top of persistence, never as a substitute.)

## Layout

```
books/
  INDEX.json                         # registry index: id -> {timeframe, side, role, script, metrics, manifest}
  EURUSD.m15.v1.manifest.json        # birth certificate (record)
  EURUSD.m15.v1/                     # the frozen artifact bytes (verbatim model)
    m15_EURUSD_direction_lgb.txt
    m15_EURUSD_direction_xgb.json
    m15_EURUSD_direction_cat.cbm
    m15_EURUSD_strategy.json
  ... (one manifest + dir per book)
manifest.py                          # reusable helper: git sha, sha256, data fingerprint, env capture, build/freeze
build_books.py                       # one-time retrofit that froze the existing deliverable books
```

`books/` is version-controlled (un-ignored in `.gitignore`), so a clone gets the models — fixing the prior single-point-of-failure (artifacts lived only on the external drive). The duplicate copy also acts as an on-drive frozen snapshot independent of the working `models/` dir.

## Identity scheme (semantic name + content hashes, layered)

- **Semantic id** = `<currency>.<book>.v<n>` (e.g. `EURUSD.m15.v1`, `EURUSD.min1.v1`, `EURUSD.tick5.v1`). The handle you cite; matches how we already talk (m15/min1/m30).
- **Content hashes** in the manifest (`artifacts[].sha256`, `source.git_sha`, `feature_fingerprint.sha256_of_listing`) prove uniqueness/integrity. The name is the handle; the hash is the proof. Cut a new `vN+1` whenever the artifact bytes change.

## What a manifest captures (`books/<id>.manifest.json`, schema `book-manifest/v1`)

| Field | What |
|---|---|
| `id`, `currency`, `timeframe`, `side`, `role` | the unique key + model role (direction / magnitude / stack-meta) |
| `summary`, `metrics` | one-liner + the headline OOS numbers (oos_2026, combined, cpcv, breakeven) |
| `source.{script, git_sha, git_dirty}` | the producing script + commit. **`script @ git_sha` is AUTHORITATIVE for the full training config** (hyperparams are inline there). |
| `hyperparams` | best-effort summary of the ensemble + headline knobs (convenience; not exhaustive) |
| `artifacts[].{file, bytes, sha256}` | the verbatim model bytes — the recreation IS loading these |
| `strategy_json` | path to the frozen gate/feature spec (thresholds, feature_names, splits) |
| `feature_fingerprint` | fast change-detector of the feature parquet set that produced the book (n_files, total_bytes, sha256 of the size+mtime listing) |
| `env.key_libs` | the ACTUAL training-venv versions (lightgbm/xgboost/catboost/numpy/pandas/...), auto-captured — supersedes the hand-maintained `ENVIRONMENT_libs.txt`, which had drifted (it said pandas 3.0.3; the venv ran 2.3.3) |
| `determinism` | honest record: estimator seed `unset`, threads 20, `bitwise_retrainable: false` |
| `depends_on` | parent books a composite (e.g. the 5m stack consumes the 15m parent) |

## How to RECREATE a book verbatim

1. `git checkout <commit>` (or the tag `book/<id>` — see below) to get the code + the artifact bytes.
2. Load the artifacts from `books/<id>/` (LightGBM `Booster(model_file=...)`, XGBoost `load_model`, CatBoost `load_model`, joblib `load`) + read the `strategy.json` for the gate/threshold spec.
3. Verify integrity: recompute each `artifacts[].sha256` and compare to the manifest. Same hash ⇒ same model.
4. Run inference with the producing `source.script`'s load/predict path (e.g. `m15_production.py` `_load()` + `_blend()`), under the deriv-faithful settlement in `harness.py`/the script.

To **re-verify a result number** (not just reload): also confirm `feature_fingerprint` matches the current `features/` (data unchanged) and `env.key_libs` matches the interpreter. If either differs, the number may move — flag it.

## How to ADD a new book (going forward)

At the end of a production `train()`, call the helper (no retrofit needed):
```python
import manifest
man = manifest.build(book_id="EURUSD.m15.v2", timeframe="15m", side="combined", role="direction",
                     script="m15_production.py", summary="...", metrics={"oos_2026": ...},
                     artifacts=[art("direction_lgb.txt"), art("direction_xgb.json"), art("direction_cat.cbm")],
                     hyperparams={...}, strategy_json="models/m15_EURUSD_strategy.json",
                     feature_fingerprint=manifest.dir_fingerprint(FEAT_DIR, ("EURUSD_*.parquet",)))
manifest.freeze(man, artifacts_src=[...])     # copies bytes into books/<id>/ + writes the manifest
```
Then `git add books/EURUSD.m15.v2*`, commit, and `git tag book/EURUSD.m15.v2`.

## Git tags

Each frozen book is tagged `book/<id>` at the commit that contains it, so `git checkout book/EURUSD.m15.v1` gives you the exact code+artifact state.

## Scope

Only the **frozen deliverable books** (the survivors referenced in `EURUSD_RESULTS.md`) are registered. The ~50 killed/null experiments are NOT — they are adequately captured by their `*_result.json` + pre-registered falsifiers and do not need verbatim recreation.

## Provenance caveat

The manifests were retrofitted onto already-trained weights (trained 2026-05-29..31). `source.git_sha` records the commit at which they were FROZEN (with `git_dirty: true` for the build files), not necessarily the commit that trained them. The `artifacts[].sha256` are what guarantee verbatim identity regardless of that nuance.
