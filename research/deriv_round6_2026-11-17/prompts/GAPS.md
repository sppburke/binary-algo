You are a completeness critic for a research planning round. Work read-only: read files, think, and write your answer
as your final message. Do not modify any file and do not open any network connection. Do NOT read holdout data or
holdout results: never open anything under /mnt/data/deriv-sota-lab/holdout/, /mnt/data/deriv-sota-lab/cache/dataset_full_*,
/mnt/data/deriv-sota-lab/registry/, the files /mnt/data/deriv-sota-lab/PROGRESS.md,
/home/sean/git/binary-algo/results/USDJPY_RESULTS.md, /home/sean/git/binary-algo/docs/DERIV_SOTA_PROGRAM.md and
/home/sean/git/binary-algo/results/json/deriv_lab_registry.jsonl, any file under /home/sean/git/binary-algo/results/json/
whose name contains "look1", "look2" or "holdout", or any Deriv tick, bar or sidecar file dated 2026-08-01 or later.
Do not run git. Do not start heavy computation (no model fits); reading and light parsing only (python3 is available).

## The program (context)

Goal: the best demonstrable out-of-sample predictor of the direction of seven FX pairs as quoted by the broker Deriv
(AUDUSD, EURUSD, GBPUSD, NZDUSD, USDCAD, USDCHF, USDJPY) for its Rise/Fall binary contract (durations from 15 minutes
to 1 day on FX; UP and DOWN are separate keys; ties lose; breakeven win rate 0.541). Decisions are made at the close of
every one-minute bar. Certification is per-fold-refit combinatorial purged cross-validation (6 chronological groups, 15
two-group test paths), New York 08:00-17:00; a side passes iff at least 12 of 15 paths have >= 25 trades and a win rate
>= 0.541, and the 10th percentile of path win rates is >= 0.541. Trades are the most confident validation-quantile
scores per side, de-overlapped by one horizon. Data in use: only Deriv 1-second mid ticks for the seven pairs,
development window 2025-08-11 .. 2026-07-31 (Deriv serves about 365 days of tick history). About 110 experiment tags
have been run; no pair except USDJPY has a passing side; the path-mean AUC of every configuration is at most 0.52
outside USDJPY, and lower-variance models (kernel, logistic) have matched or beaten LightGBM on most keys. Hardware: 20
CPU threads, 62 GB RAM, an idle 8 GB NVIDIA GPU (PyTorch CUDA, XGBoost CUDA, CatBoost GPU). Installed: lightgbm,
xgboost, catboost, scikit-learn, torch, pytorch_tabnet, sktime, darts, chronos, river, hmmlearn, arch, pywt,
statsmodels, numba, polars, optuna.

The owner asked for a FULLY COMPREHENSIVE set of configurations to test - every machine-learning variant worth trying,
in the owner's words "any combination of ensemble, pre-processing + ML, certain labels, etc, ANYTHING". Twenty-one
lenses have proposed candidates. Your job is to find what is still MISSING.

## Inputs (all under the current directory, /mnt/data/deriv-sota-lab/discovery/round6)

- experiments_inventory.md - every experiment already run, with per-pair results.
- merge/items_brief.tsv - one line for every one of the 353 proposals the lenses made in this round: id, group, kind,
  side, needs, prior, title, first 160 characters of the spec. Read every line. Details when you need them:
  merge/items.jsonl (id, title, prior, side, kind, needs, cost, mechanism, spec, nearest_existing, fast_kill; long
  fields cut with " [...]") and the full records in results/<LENS>.txt (R6-K4-02 is in results/K4.txt).
- The harness options: /mnt/data/deriv-sota-lab/tools/cpcv_deriv_x.py (read the argparse section) and the feature and
  label builders beside it (deriv_dataset.py, deriv_features_*.py, deriv_labels_*.py).
- Method menus of the repository: /home/sean/git/binary-algo/SWEEP_MATRIX.md, /home/sean/git/binary-algo/docs/METHODS_CATALOG.md,
  /home/sean/git/binary-algo/docs/IDEAS_LOG.md.

## What to do

1. Write down your own systematic taxonomy of the design space BEFORE reading the proposals, so that the lenses do not
   anchor you. Cover at least: target and contract (horizon 15 m .. 1 d, side, tie handling, entry timing, which
   minutes of the hour and which sessions to trade), information set (own-pair tick microstructure, cross-pair, derived
   crosses and currency indices, calendar and clock, regime states), feature transformations and preprocessing
   (scaling, denoising, decomposition, dimensionality reduction, discretization, learned representations) each paired
   with a model class, model classes (linear, kernel, tree ensembles of every growth and sampling regime, additive
   models, rule lists, nearest neighbours, Bayesian models, state-space and regime-switching models, neural tabular,
   neural sequence, pretrained time-series models, online learners), labels and losses, sample weighting and training
   population (pooling across pairs, horizons and sessions; window length; recency; uniqueness), hyper-parameter
   search protocol, ensembling and stacking, calibration and the decision layer (thresholds, coverage, abstention,
   gating, cooldowns, per-hour rules), validation and multiple-testing control, and deployment realism (latency,
   refit cadence, drift monitoring).
2. Read experiments_inventory.md and every line of merge/items_brief.tsv and mark each cell of your taxonomy as TESTED
   (name the experiment tags), PROPOSED (name the proposal ids) or GAP.
3. For every GAP that is worth one bounded test, write a proposal. Do not pad: a gap is worth listing only if you can
   state a mechanism by which it could change the confident-tail win rate or the validity of the conclusions, and it is
   not a small variation of something TESTED or PROPOSED. Combination gaps count (for example a preprocessing step that
   was only ever paired with one model class, or a label never combined with the best model on that pair).

## Output format (final message; JSON lines only, no prose outside JSON, no code fences)

First the coverage table, one line per taxonomy cell:
{"cell": "<axis> / <sub-axis>", "status": "TESTED|PROPOSED|GAP|PARTIAL", "tested_by": ["e30nystroem", ...],
 "proposed_by": ["R6-K4-02", ...], "note": "<=160 chars"}

Then one line per gap proposal:
{"id": "R6-GAP-01", "title": "<=90 chars", "kind": "feature|model|label|loss|training|ensemble|gate|decision|validation|
infrastructure|data", "sign_or_size": "SIGN|SIZE|N/A", "needs": "new_code|gpu|reopen_closed_family|owner_data_decision",
 "prior": "low|medium|high", "prior_why": "<=200 chars", "mechanism": "<=300 chars", "spec": "<=700 chars, bounded and
 concrete: inputs, model, hyper-parameters or the small grid, how it is selected without touching test data",
 "nearest_existing": "<=200 chars: closest experiment tags and proposal ids, and what differs", "fast_kill_falsifier":
 "<=200 chars", "cost": "<build S/M/L; compute>"}

Last line:
{"summary": "<=1200 chars: how complete the union of experiments and proposals is, the most important gaps, and what
 you looked for and did not find missing>"}
