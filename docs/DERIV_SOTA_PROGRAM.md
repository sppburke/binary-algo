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
entry − 1; ties are valid and lose (`scripts/min1_production.wc_ret` semantics). Breakeven 0.541 (the goal's binding
rule; `docs/THEORY.md` §3 assumes payout R≈1.85).

**Payout sensitivity.** The only venue-recorded trade on disk (`evidence/objects/e2512a4a…json`: a 15m CALL on
2026-07-06, ask 1.00, payout 1.77) implies breakeven 1/1.77 ≈ 0.565, above 0.541. One observation does not set the
rule, but every Deriv-feed table also reports the share of CPCV paths clearing 0.565. Deriv also lists Rise/Fall
**Equal** contracts (CALLE/PUTE, ties win) at 15m–1d (`docs/deriv_frxEURUSD_contracts_for.json`); their payout is not on
disk and they are not evaluated.

## Datasets (scripts are byte-identical copies of the lab tools; result JSONs record their SHA-256)

| Script | Builds |
|---|---|
| `scripts/deriv_dataset.py` | Deriv-native features v0.1: own-pair wall-clock returns, volatility, range position, EMA distance, activity (share of seconds with a price change), clock; other pairs' returns joined as-of (backward only) with a staleness column; labels at 15m and 30m |
| `scripts/deriv_features_b239.py` | the incumbent books' schemas rebuilt from Deriv ticks with the repository's own recipe code (`pipeline.build_features`, `live_features.build_cross_pair_features`); volume = ticks per minute (the live path's proxy); EURUSD order-flow columns cannot be built from mid ticks |
| `scripts/deriv_features_family.py` | 52 causal SWEEP_MATRIX family features in 8 prefixed blocks (tick-rule flow, Lévy areas, Kalman, RMT residual, fractional differencing, complexity/regime, tick-clock returns, price action); merged onto v0.1 as `dataset_dev_v0.3` |
| `scripts/deriv_features_ccm.py` | the C4 convergent cross-mapping block (`cm_`); merged onto v0.3 as `dataset_dev_v0.4` |
| `scripts/deriv_labels_resid.py` | the F4 residualized TRAINING label (own settled return minus beta × USD-basket return; training target only) |

## Certification harness

Harness builds: `cpcv_deriv/0.2` (SHA-256 prefix `6705cf9f49a0dcd8`) differs from `0.2.1` (`7a3032a6a3ad1011`, commit
a2f5ed8) only by a defensive copy of the feature matrix and the version string; results are unaffected. `0.3` added the
venue tradeability filter; `0.4` the review fixes below (each result JSON records its harness version and SHA-256).
Variant harnesses: `cpcv_deriv_x.py` (fit session, local NY windows `nylate` 13–22 / `nypm` 13–17 / `nyeve` 17–22 /
`nyam` 8–13, time-block validation, triple-barrier training label, magnitude gate, model knobs; 0.4 adds SWEEP_MATRIX
family blocks `--families`, the residualized training label, model classes `--model ens3|logit|nystroem`, side
specialists, a compression gate, |return| weights, an adaptive-conformal gate and `--dump-preds`; defaults reproduce the
base mirror), `cpcv_deriv_pool.py` (pooled training for the EURUSD/GBPUSD/USDCHF targets) and `cpcv_deriv_online.py`
(prequential online learner scored on the same groups and paths; forward-only by construction).

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

## Venue tradeability (binding from harness cpcv_deriv 0.3, 2026-11-15)

Deriv rejects an FX Rise/Fall purchase whose expiry lands in **16:50–18:00 New York** (daily close/rollover; for 15m
the last accepted start was 16:34:40 NY and the first rejection 16:35:10 NY — `docs/DERIV_DEMO_EXECUTOR.md` (g)).
Every Deriv-feed evaluation therefore excludes decisions with start in **[16:50 − H, 18:00) NY** from training,
validation and test (`cpcv_deriv.tradeable`; 15m: [16:35, 18:00); 30m: [16:20, 18:00)). Results produced before the
filter (`e1`, `r2`, `r1`, `r1b`) are superseded by their tradeable re-runs (`e1t`, `r2t`, `r1t`, `r1bt`).

## Where the incumbents' "NY" edges came from (2026-11-15)

The Dukascopy-derived stores were stamped true UTC − 5 h before 2025-08-22, so `session_mask(ts, 'ny')` on them
selected **true New York 13:00–22:00** (all year: the shift is constant in UTC, the gate converts with DST). Scoring the
frozen books on Deriv features built on that training clock (`deriv_features_b239.py --clock-shift-h=-5`; only clock
features and 4h bar boundaries move) in true 13:00–22:00 ET gives very high win rates — but a per-trade audit
(`audit_trades.py`) shows most of them come from decisions inside the untradeable rollover blackout, where the feed's
behavior is predictable but no contract can be bought. On tradeable decisions only, a smaller edge remains for some
books, concentrated in the NY afternoon (13:00–16:35). Per-key numbers: the DERIV-FEED PROGRAM sections of
`results/<PAIR>_RESULTS.md` (R1b, R1bt) and `results/json/deriv_audit001_r1b_tradeable_result.json`.
Generic lesson (a new mirage class): **an edge measured across the daily rollover is not tradeable on Deriv**; any
session-gated certification must exclude the venue's purchase blackout before it is believed.

## Independent foundation review (2026-11-15) and fixes

A read-only Codex review of the dataset, settlement, CPCV mirror and scorer found no feature look-ahead and confirmed
settlement matches `wc_ret`; it reproduced fidelity defects that were fixed in `cpcv_deriv 0.4` (with `cpcv_deriv_x
0.3`, `cpcv_deriv_pool 0.3`, `holdout_eval 0.2`, `score_frozen_books 0.4`): verdicts now aggregate unrounded path win
rates (rounding could have counted a 0.54098 path as clearing 0.541); the subsampled fit pool keeps the reference's
sampled order (no sort); an invalid run certifies nothing in any field; feature groups (`own`, `noclock`) understand
every schema; `settle` requires a finite return; the frozen-book scorer applies the runtime's `15m_bb_width`
compression column by default. A regression re-ran all 15 E1 USDJPY paths with 0.4 and reproduced every per-path value.
Documented, not changed: research features use the full history while the live builder keeps 14,000 minutes (EMA
state differs slightly — a deployment-parity item); the de-overlapped side streams are the repository's evaluation
discipline, not a replay of the live executor's concurrency; v0.1 cross-pair staleness measures bar age, not tick age.

## Deriv history depth (data ceiling)

Deriv serves tick history for about **365 days** (archive README `/mnt/data/deriv/README.md`: on 2026-11-14, Friday
2025-11-14 18:00 UTC was no longer served, Monday 2025-11-17 00:00 UTC was). The "retention >= 26.5 years" recorded by
`scripts/deriv_api_probe.py` probe (d) (`results/json/deriv_api_probe_result.json` `decisions.tick_retention_days_min`,
quoted in `docs/DERIV_DEMO_EXECUTOR.md` (d)) is an artifact: its `_has_ticks_at_or_before` requests `ticks_history` with
`end` and no `start` and only tests that the reply is non-empty, while the API ignores `end` without `start` and returns
recent ticks (archive README, reply-shape check of 2026-11-15). The lab archive (from 2025-08-11) is therefore the
maximum Deriv-only history; it grows only forward. Deriv-only models are data-limited (about 15 months of DEV); this is
the main structural constraint behind the null results of 15-month models in hours where 10-year frozen books still
carry signal (R1bt). That docs/DERIV_DEMO_EXECUTOR.md (d) needs a correction is recorded here; the edit itself is a
separate dev-cycle change.

## Holdout discipline (registered 2026-11-15, before any holdout data is opened)

Every DEV-window variant tried for a key belongs to that key's search family (all logged in the lab registry). At most
**two finalized candidates per key** are opened on the holdout, chosen on DEV evidence only (CPCV verdict, forward walk,
no-clock ablation) and named in the key's sweep ledger before the holdout is scored. Reading per candidate
(`holdout_eval.py`): PRIMARY = walk-forward monthly refit (train on everything before each holdout month, purged),
CONTEXT = one model frozen at the end of DEV. A side is SOTA-eligible iff the PRIMARY reading has n >= 50 and its
CI95-lower (Bonferroni-adjusted for the number of candidates opened for that key, i.e. a 97.5% interval when two are
opened) is >= 0.541, and its point win rate beats every incumbent book of the key scored on the same window as
deployed. Then the full per-fold-refit CPCV at the operating gate certifies it.

## Experiment record

Lab registry `/mnt/data/deriv-sota-lab/registry/tests.jsonl` (append-only: start, run, finish, clean for every test;
snapshot committed as `results/json/deriv_lab_registry.jsonl`). Result JSONs: `results/json/deriv_<exp>_<PAIR>_<H>m_result.json`
and `results/json/deriv_r1_frozen_books_<window>_result.json`. Per-key tables are rendered from those JSONs.

## SWEEP_MATRIX family coverage on the Deriv feed (2026-11-15)

Every SWEEP_MATRIX family is mapped once to a Deriv-feed experiment or to the reason it is not run. Family feature
blocks are built causally by `scripts/deriv_features_family.py` (truncation test: rebuilding on a longer window changes
no earlier value of any of the 52 features) and merged onto v0.1 as `dataset_dev_v0.3`; each block is evaluated as
"E1t + block" with `cpcv_deriv_x.py --families`. Model-class families use `cpcv_deriv_x.py --model / --specialist /
--comp-gate`; C5 uses the prequential `cpcv_deriv_online.py`. Results per key: the DERIV-FEED tables of
`results/<PAIR>_RESULTS.md`.

| Family | Deriv-feed experiment or disposition |
|---|---|
| A1 three-GBM ensemble | `e31ens3` (LightGBM + XGBoost + CatBoost; capped fast-KILL screen) |
| A2 compression × session × coverage gate | `e33comp33` (rv_60m <= fit-pool 33% quantile); sessions `e1t`, `e14`–`e17`; every run reports coverages 0.02/0.03/0.05 |
| A3 compression-release × reversion specialist | `e38release` (the A8 specialists trained and traded only after a volatility burst, volratio_15_240 >= fit-pool 75% quantile) |
| A4 meta-labeler | `e37meta` (out-of-fold primary from two purged chronological halves of each fit pool; meta gate keeps the top 50%) |
| A5 cross-horizon stack | the family's mechanism (a longer-trained parent book) needs the Dukascopy-trained frozen parents, excluded by the Deriv-only constraint (open owner decision 2); a Deriv-trained parent sees the child's rows and features; the 30m horizon itself is `e5h30` |
| A6 cross-pair residual / lead-lag | v0.1 cross-pair as-of block in every run; incumbent cross-pair blocks `e3union`; RMT residual `e22rm`; Lévy area vs the USD basket `e20sg` |
| A7 walk-forward retrain | forward walk in every CPCV run; holdout PRIMARY reading is a monthly walk-forward refit |
| A8 side filter vs specialist | filter: every run's UP/DOWN split; specialist `e32spec` |
| A9 DST-correct sessions | `e1t` (08:00–16:35), `e14nylate`, `e15nypm`, `e16nyeve`, `e17nyam` |
| B1 tick microstructure ensemble (seconds horizons) | out of scope: horizons below 15 minutes |
| B2 order-book imbalance, B3 CKS event-OFI, B4 cross-impact OFI | not computable: Deriv's feed is a 1-second mid (no book, sizes or trades) |
| B5 per-side signed flow | `e19tf` (tick-rule imbalance of mid changes) |
| B6 kernel SVM | `e30nystroem` |
| B7 path signature | `e20sg` |
| C1 HMM regime | `e24cx` (two-state volatility-regime forward filter, sign-invariant) |
| C2 Kalman | `e21kf` |
| C3 RMT eigen-residual | `e22rm` |
| C4 CCM coupling gate | `e39cm` (`scripts/deriv_features_ccm.py`: rolling CCM skill of each other pair onto the target from a 240-bar shadow manifold, every 15 bars, plus the coupled drivers' 15-minute push; truncation-tested; merged as `dataset_dev_v0.4`) |
| C5 online drift learner | `e34online` (river ARF + ADWIN, prequential) |
| C6 SSA / fractional differencing / particle filter | `e23fd` (fractional differencing); ESN closed |
| D1–D3, D6–D9 (CNN/GRU/NCDE/TabNet, TSFM/Kronos, bar-image CNN) | closed (AGENTS.md neural/spectral, image, GRU/ESN, TSFM nulls) |
| D4–D5 deep RL (DQN, IQN) | closed with the neural family (AGENTS.md); reopen only by owner decision |
| E1 magnitude classifier | a gate only (goal scope): `e13mag50` |
| E2 direction conditioned on magnitude | `e13mag50` |
| E3 complexity gates | `e24cx` |
| E4 information bars | `e25tk` (tick-clock returns); volume/dollar bars not computable (no volume) |
| F1 macro-release impulse | excluded: non-Deriv input (`macro_calendar.parquet`; open owner decision 4) |
| F2 structural / price action | `e26pa`; round-number and same-slot seasonality in `e10v02` |
| F3 external cross-asset lead-lag | excluded: needs data acquisition (no ES/NQ on Deriv) |
| F4 residualized target | `e28resid` |
| Low-capacity models (data-ceiling response) | `e18lowcap` (LightGBM), `e29logit` (L2 logistic) |

Tier-I improvers named by the goal: seed ensemble `e4seed3`; |return|-weighted loss `e35absret`; adaptive-conformal gate
`e36aci` (causal: the threshold moves only when a taken trade has settled; with step 0 it reproduces the plain scorer
exactly); pooling where not closed `e6pool`; capped tuning `e18lowcap`; calibration does not change which decisions clear
a symmetric confidence quantile, so it is folded into the ACI gate; the DL+GBM stack is closed with the neural family.
Harness regression for every option added to `cpcv_deriv_x` 0.4: with defaults, CPCV path 1 of `e1t` USDJPY is
reproduced exactly (trade counts and unrounded win rates at every coverage and side).

**Drift-neutral diagnostic.** Every CPCV result stores, per side, the excess of the path win rate over the always-side
baseline on the same path (`excess_vs_always`). `analyze_drift_placebo.py` (run by `run_placebo.sh`) re-runs a model with
`--dump-preds` and compares its trades with always-side baselines matched on New York hour and on hour × rv_60m tercile
within each path; its reading (skill beyond timing the drift iff the hour × volatility excess is positive on >= 12
counted paths with mean >= 0.02) is written into the output before the predictions are opened.

## Closures respected

The repository closures in AGENTS.md stand for this program unless the owner reopens them: no re-mining of the
neural/spectral, image, GRU/ESN, TSFM or Optuna nulls, and no cross-pair pooling for the USDJPY/USDCAD/AUDUSD/NZDUSD
own-pair family. Note for the owner: those closures were measured on the Dukascopy-derived stores, whose timestamps
are not true UTC (verified 2026-11-14: stored = true UTC − 5 h for files labelled 2012-01-02..2025-08-22, + 7 h in US
summer / + 8 h in US winter from 2025-08-25; evidence and reference correction in
`/mnt/data/deriv/checks/clock_corrected_agreement.py`, outside this repository). Their session-gated numbers therefore
describe different true hours than the live Deriv feed. The closures are kept, not re-litigated, unless reopened.

## Open owner decisions

1. **Validity band vs genuine drift.** strategy-eval §2 treats a moved up-rate outside [0.47, 0.53] as a fake-flat mirage
   ("invalid regardless of accuracy"). On tick-settled Deriv labels (no forward fill) the band also fires on genuine
   drift (USDJPY +6.8% over DEV: `results/USDJPY_RESULTS.md`, drift diagnosis). The program keeps the rule; a possible
   amendment is mirage diagnostics (tie rate, forward-fill share) plus a drift-neutral requirement (each side beating its
   always-side baseline on every counted path).
2. **Frozen books as an operational option.** The only DEV-supported edges so far are frozen Dukascopy-trained USDJPY
   books scored on their training clock in true NY 13:00–16:35 / 18:00–22:00. They are benchmarks under the Deriv-only
   constraint, not deliverables, unless the owner decides otherwise.
3. **Repository closures** measured on mis-clocked data (neural/spectral/GRU/TSFM/Optuna nulls; pooling for the
   JPY/CAD/AUD/NZD family): kept unless reopened.
4. **Non-Deriv inputs** (e.g. `macro_calendar.parquet`): excluded unless approved.
5. **Payout**: one recorded venue trade implies breakeven ≈ 0.565; the goal's rule uses 0.541.
