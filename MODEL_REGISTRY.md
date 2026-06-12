> **SCOPE: GENERIC registry** (multi-currency). Every entry is keyed `<PAIR>.<book>.v1` with a provenance manifest. See REPO_MAP.md.

# Model Registry — frozen books, manifests, and how to recreate them verbatim

This registry makes every deliverable model **concrete, uniquely referenceable, and recreatable verbatim**. It is the resolution target for the `(currency, timeframe, side)` keys in `results/EURUSD_RESULTS.md`.

> **Future agents:** to evaluate/reverse-engineer a strategy for any (currency, timeframe) and record it correctly, invoke the **`strategy-eval`** skill (`.claude/skills/strategy-eval/SKILL.md`). It wires this registry + `docs/METHODS_CATALOG.md` + the evaluation discipline + the `results/<PAIR>_RESULTS.md` writing protocol into one workflow.

## Latest additions (2026-06-04 — EURUSD 10m BOTH sides certified)
- **`EURUSD.m10xp.v1` — the (10m,UP)+(10m,DOWN) certified leader** (cross-pair USD-residual+OF primary, m5xp source @MX_HOR=10, lgb 3000-tree; gate `5m_bb_width ≤ q × sess_ny × |p−0.5| ≥ conf_thr`). Per-side full-refit CPCV (15 purged paths): **UP p10 .5863 / DOWN p10 .5683, 15/15 paths each** — both improve the base-book side floors (`EURUSD.m10.v1` side-split UP .561 / DOWN .552, `m10_cpcv_side.py`). Deploy cov10% (DOWN better EV at cov5%: fwd-2025 .630, EV .166); durable number = refit p10 (frozen-forward overstates). **11 improve/discover levers (magweight, GMADL, residual-relabel, ACI, specialist, calibration, cross-horizon blend [pred corr .957], lagged-leadlag, queue, intraday-momentum, base) ALL KILLED/subsumed** — edge is the gated raw cross-pair sign, info-bound by the 2025-USD-regime; loops dry (K=2 rounds). Confirms the cross-pair gradient **none@60s→UP@5m→BOTH@10m&15m**. **10m < deriv 15m forex min → research/synthetic-index horizon; deployable sibling = `EURUSD.m15xp.v1`.** `m10_xpair_cpcv.py`/`_result.json`, `m10_xpair_freeze.py`+`m10_xpair_regate.py`, `sweeps/EURUSD_10m.md`, `results/EURUSD_RESULTS.md` §10m. Base `EURUSD.m10.v1` = lower certified fallback. git-tagged `book/EURUSD.m10xp.v1`.

## Latest additions (2026-06-02 — DOWN book freeze)
- **`EURUSD.m5xp_magw_down.v1` — the (5m,DOWN) MARGINAL certified edge** (magnitude-weighted primary, POW=0.5). Nested-refit CPCV DOWN cov0.05: p10=**0.5441**, 89.3% of 28 purged paths clear breakeven 0.541 — CERTIFIED but MARGINAL (regime-dependent, 2025 is weakest). Gate: NY session & pr<0.5 & top-5% confidence cover; fixed threshold 0.0917 derived from val 2022-2023. Trained on all 2012-2026 pooled data (stride=6, n_est=700, sample_weight=(|fwd|/med)^0.5). `m5_magweight_freeze.py`, `m5_magweight_cpcv_result.json` (B1 run, git HEAD~4).

## Previous additions (2026-06-01 — EURUSD 5m sweep + edge-improvement loop)
- **`EURUSD.m5xp.v1` — the (5m,UP) certified edge** (side-split of the frozen cross-pair book). UP forward .605/.577/.615; survives the FULL per-fold-refit CPCV at the ~5% operating gate (cov0.05 p10 **0.553**, 96% of purged folds clear) — the first sub-15m direction edge to survive the full refit. (5m,DOWN) = marginal/dead (refit p10 .542 barely, forward 2025 .538 fails). Deployable ~0.55–0.57, ⅛-Kelly. `m5_updown_result.json`, `m5_refit_tightcov_result.json`, `sweeps/EURUSD_5m.md`.
- **`EURUSD.m5xp_aci.v1` — ⚠ DOWNGRADED 2026-06-01 (single-split mirage; does NOT survive nested-refit CPCV).** Same m5xp primary+meta bytes (`depends_on: EURUSD.m5xp.v1`) + an ONLINE **adaptive-conformal (ACI) gate** (w*=0.57). On the forward 2024-26 split it LOOKED like a win (binding 2025 .584@n764 vs fixed .579@n618, +36% trades). The open follow-up — **CPCV-validation of the adaptive policy — RAN and KILLED it**: the nested-refit CPCV (`m5_aci_cpcv.py`, refits BOTH primary+meta on each of 28 purged paths, replays ACI online per fold, n=876k) gives ACI **p10 0.521–0.523, only 54–61% of paths clear**, at LOWER coverage (med_n 1231–1663) — strictly WORSE than the fixed meta gate (p10 0.5376, 78.6%, med_n 2247). The 2025 "win" was a forward-regime artifact of one chronological split. **DO NOT deploy ACI as an improvement.** Moreover the nested-refit shows the DEPLOYED meta gate itself is only a near-miss (p10 .5376/78.6%) and that selecting by PRIMARY CONFIDENCE (cov0.05 p10 .553/96%) is MORE robust at equal selectivity — so the certified deployable UP point is the **tight primary-confidence cover**, not the meta gate. The book artifact remains version-controlled for provenance but is NOT a recommended deliverable. `m5_aci_cpcv.py`, `m5_aci_cpcv_result.json`. (Other levers: EXP-1 magnitude-weighting rebalanced to two-sided ~.56, no UP win; EXP-2 seed-ensemble MLP decorrelated corr .694 but 50/50 blend ≈ GBM — info-bound.)

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

Only the **frozen deliverable books** (the survivors referenced in `results/EURUSD_RESULTS.md`) are registered. The ~50 killed/null experiments are NOT — they are adequately captured by their `*_result.json` + pre-registered falsifiers and do not need verbatim recreation. **The 2026-06-07 novel-methods campaign added NO new books** (HAR magnitude sub-bar; FFD/D1/D6/D7 direction all killed forward) — see `docs/CAMPAIGN_2026-06-07_FACTS.md` / `docs/EXPERIMENT_LEDGER.md` #156-160. **The 2026-06-07 post-campaign Brier-advantage audit (`brier_audit.py`, docs/EXPERIMENT_LEDGER.md #161) added no books either but VALIDATED the 4 cross-pair direction books** (`EURUSD.m{5,10,15,30}xp.v1`): each beats a naive persistence baseline in calibrated Brier score in every forward year 2024/25/26 (all-bars + bet-tail) → none is "ranking-only". Audit-only, no artifact/cert changed; honest caveat = the flat-0.5 margin is razor-thin and the skill is tail-concentrated (see `brier_audit_result.json` `verdict`).

## Provenance caveat

The manifests were retrofitted onto already-trained weights (trained 2026-05-29..31). `source.git_sha` records the commit at which they were FROZEN (with `git_dirty: true` for the build files), not necessarily the commit that trained them. The `artifacts[].sha256` are what guarantee verbatim identity regardless of that nuance.
