> **SCOPE: GENERIC** (program protocol for the Deriv-feed state-of-the-art search; currency/timeframe-agnostic).
> Per-key numbers live only in `results/<PAIR>_RESULTS.md` (DERIV-FEED PROGRAM sections) and `sweeps/<PAIR>_15m.md`
> (DERIV-FEED PROGRAM LEDGER). See REPO_MAP.md.

# Deriv-feed state-of-the-art program (opened 2026-11-15)

**Goal (owner's /goal).** Build, certify and keep improving the best demonstrable out-of-sample predictor for every
Deriv FX instrument we hold data for (AUDUSD, EURUSD, GBPUSD, NZDUSD, USDCAD, USDCHF, USDJPY), UP and DOWN as separate
keys; magnitude (`|ret| >= Q`) only as a gate. A side is state of the art when it beats every incumbent combination in
`books/INDEX.json` on the binding held-out period with CI95-lower clear of breakeven, and is certified by full
per-fold-refit CPCV at its operating gate (15 purged paths, p10 >= 0.541, >= 80% of paths clear, America/New_York
08:00–17:00 with DST).

**Owner's constraints.** Deriv data only — no Dukascopy-derived data, not even auxiliary history. Horizons start at
15 minutes (Deriv's minimum for these contracts): 15m, 30m and longer. Every experiment is isolated, bounded, local,
recorded and cleaned, and runs under `/mnt/data/deriv-sota-lab/`.

## Data and windows

- Source: the Deriv tick archive `/mnt/data/deriv/ticks` (one mid price per second, true UTC), read-only.
- DEV window **2025-08-11 .. 2026-07-31**: all training, selection and CPCV.
- HOLDOUT **2026-08-01 .. 2026-11-14**: sealed; opened once per finalized candidate (and for the incumbents on the
  same window). Per-calendar-year held-out evaluation is not applicable to ~15 months of Deriv-only data.

## Labels (Deriv-faithful settlement)

Decision at the close of each one-minute bar (minute start + 60 s). Entry = first tick at or after decision + 1 s
(within 30 s at 15m, 60 s at 30m); exit = last tick at or before entry target + H (same tolerance); return = exit /
entry − 1; ties are valid and lose (`scripts/min1_production.wc_ret` semantics). Breakeven 0.541.

## Datasets (scripts are byte-identical copies of the lab tools; result JSONs record their SHA-256)

| Script | Builds |
|---|---|
| `scripts/deriv_dataset.py` | Deriv-native features v0.1: own-pair wall-clock returns, volatility, range position, EMA distance, activity (share of seconds with a price change), clock; other pairs' returns joined as-of (backward only) with a staleness column; labels at 15m and 30m |
| `scripts/deriv_features_b239.py` | the incumbent books' schemas rebuilt from Deriv ticks with the repository's own recipe code (`pipeline.build_features`, `live_features.build_cross_pair_features`); volume = ticks per minute (the live path's proxy); EURUSD order-flow columns cannot be built from mid ticks |

## Certification harness

Harness builds: `cpcv_deriv/0.2` (SHA-256 prefix `6705cf9f49a0dcd8`) differs from the committed `0.2.1`
(`7a3032a6a3ad1011`) only by a defensive copy of the feature matrix (`to_numpy(..., copy=True)`, needed under pandas 3
copy-on-write when a column subset is selected) and the version string; results are unaffected.

`scripts/cpcv_deriv.py` mirrors `scripts/audusd_15m_cpcv_session.py`: session gate on the decision time
(`sessions.session_mask`), 6 equal-count groups of in-session decisions, all 15 pairs of groups as test paths, purge
and embargo H around each test block, training on moved in-session unpurged rows, a random 15% validation subset
(early stopping, threshold source), fit pool <= 150,000 rows, LightGBM with the repository's `mk_lgb` parameters,
threshold = validation quantile of |p − 0.5| at the coverage, side filter then first-come de-overlap (gap H), wins
need the right sign and a move, a path counts with >= 25 trades; CERTIFIED iff p10 >= 0.541 and >= 80% of counted
paths >= 0.541. The falsifier is written to the result JSON before any path is fitted. Validity exactly as the
repository certifier applies it: dataset in-session moved up-rate in [0.47, 0.53] and every path in [0.45, 0.55].
Diagnostics (not part of the rule): per-path moved AUC; always-UP / always-DOWN baselines on the same de-overlapped
decisions; a **forward walk** (train only on groups before group g, test group g, g = 2..5) guarding against
flanked-fold memorization of slow features (leakage trap #9).

`scripts/score_frozen_books.py` forward-tests the frozen books as deployed: exact models, columns, deploy coverage and
threshold via `book_runtime`, DST-correct NY session, strategy compression gate, rows with all-finite book features
(the live path fails closed otherwise); SUPPORTED iff n >= 50 and CI95-lower >= 0.541 per side.

## Experiment record

Lab registry `/mnt/data/deriv-sota-lab/registry/tests.jsonl` (append-only: start, run, finish, clean for every test;
snapshot committed as `results/json/deriv_lab_registry.jsonl`). Result JSONs: `results/json/deriv_<exp>_<PAIR>_<H>m_result.json`
and `results/json/deriv_r1_frozen_books_<window>_result.json`. Per-key tables are rendered from those JSONs.

## Closures respected

The repository closures in AGENTS.md stand for this program unless the owner reopens them: no re-mining of the
neural/spectral, image, GRU/ESN, TSFM or Optuna nulls, and no cross-pair pooling for the USDJPY/USDCAD/AUDUSD/NZDUSD
own-pair family. Note for the owner: those closures were measured on the Dukascopy-derived stores, whose timestamps
are not true UTC (verified 2026-11-14: stored = true UTC − 5 h for files labelled 2012-01-02..2025-08-22, + 7 h in US
summer / + 8 h in US winter from 2025-08-25; evidence and reference correction in
`/mnt/data/deriv/checks/clock_corrected_agreement.py`, outside this repository). Their session-gated numbers therefore
describe different true hours than the live Deriv feed. The closures are kept, not re-litigated, unless reopened.
