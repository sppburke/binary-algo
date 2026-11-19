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
- Completeness: from `deriv_dataset` 0.2 (2026-11-15) every builder reads only days whose downloader sidecar says
  `complete: true`. The DEV datasets were built before that and also read the tick files of the only two DEV days marked
  incomplete, identical for all 7 pairs: 2025-12-24 and 2025-12-31, holiday early closes whose gap is the last 80
  minutes of the UTC day (17:40-19:00 ET). Checked on the cached datasets: no decision falls inside either gap, and no
  valid 15m or 30m label of any tradeable decision (NY 08:00-16:35, nylate, nyeve, nyam, nypm, all hours) crosses one;
  features after a gap see a closure, as after a weekend. `DERIV_ALLOW_INCOMPLETE=1` reproduces the old selection.
- DEV window **2025-08-11 .. 2026-07-31**: all training, selection and CPCV.
- HOLDOUT **2026-08-01 .. 2026-11-14**: sealed; opened once per finalized candidate (and for the incumbents on the
  same window). Per-calendar-year held-out evaluation is not applicable to ~15 months of Deriv-only data.

## Labels (Deriv-faithful settlement)

Decision at the close of each one-minute bar (minute start + 60 s). Entry = first tick at or after decision + 1 s
(within 30 s at 15m, 60 s at 30m); exit = last tick at or before entry target + H (same tolerance); return = exit /
entry − 1; ties are valid and lose (`scripts/min1_production.wc_ret` semantics). Breakeven 0.541 (the goal's binding
rule; `docs/THEORY.md` §3 assumes payout R≈1.85).

**Payout sensitivity.** 0.541 is the breakeven of a payout of 1.85. Deriv's quoted payout is not a constant: it moves
with the price path and the clock (section "Venue payouts" below; in July 2026 the 15-minute quotes averaged about
1.80 from 08:00 to 16:00 New York, the breakeven of which is 0.556, and 1.79 over the whole 08:00–16:35 window). Every
Deriv-feed CPCV table therefore also reports the share of paths clearing 0.565 (the breakeven of the venue-recorded
trade in `evidence/objects/e2512a4a…json`: a 15m CALL on 2026-07-06, ask 1.00, payout 1.77), and a payout-aware reading
prices a candidate's trades one by one. Deriv also lists Rise/Fall **Equal** contracts (CALLE/PUTE) at 15m–1d
(`docs/deriv_frxEURUSD_contracts_for.json`); the list states neither their tie rule nor a payout, no quote for them is
in the archive, and they are not evaluated.

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
(prequential online learner scored on the same groups and paths; forward-only by construction). `cpcv_deriv_x` 0.5
lets the logit, extra-trees and kernel models use training weights (before 0.5 they ignored `--weight`) and fits the
meta gate's half-pool primaries with the selected model; 0.6 / 0.6.1 add the literature options (shrunk stack, ordinal
bins, HAR size gate, bootstrap calibration-uncertainty gate, recency weights, multi-horizon training); 0.7 / 0.7.1
add the discovery options (drift fixed effects, entry-delay agreement label, play-operator and shock-curvature
blocks, monotone catch-up constraints); 0.8 adds discovery round 3 (`--signed-only` same-input controls, `--drift-fe`
for LightGBM with profiled stratum offsets, the `nysagree` agreement gate, the quote-update order block `qo_` and its
permuted control `qp_` from `scripts/deriv_features_qorder.py`); 0.9 adds discovery round 4 (`--model nystail
--tail-aux`: the kernel features fitted by full-batch L-BFGS with a fixed-mass soft loss of the operating confidence
tail; `--tail-aux 0` is its matched control); 0.10 adds discovery round 5 (`--blend-absret SIDE`: the unweighted
and the |return|-weighted LightGBM blended with a weight in {0, 0.5, 1} chosen on the worse chronological half of the
validation trades; `--rule-gate`: a zero-parameter fixing rule's offers ranked by a fitted model, restricted to the
audited rule columns). Each of 0.5 .. 0.10 reproduced path 1 of a stored 0.4 LightGBM result and a stored 0.4 kernel
result exactly before it was used (lab registry tests `x-regress-06` .. `x-regress-010`; 0.8 also reproduced the drift
fixed-effect runs `e67drfe` and `e67drfenys` and `c2nysabs`; 0.9 also `c6nyscomp`, `e76drfelgb` and `e78nysagree`, and
checked its new gradients against finite differences; 0.10 also the round-5 parents, both blend endpoints and the
rule offers against an independent computation). A label-based choice made on the default random validation rows
measures in-sample fit: they are minute neighbours of fit rows with overlapping labels. Such choices use
`--val-mode block`.

Round 6, stage 1 (installed 2026-11-19 11:21 UTC; the replaced files, with their SHA-256, are in
`tools_backups/e003_e020_20261119T112141Z/`): `cpcv_deriv` 0.5 and `cpcv_deriv_x` 0.11, with `cpcv_deriv_pool` 0.3.1,
`cpcv_deriv_online` 0.1.1, `cpcv_rule` 0.1.1 and `fork_jobs.py` (SHA-256 prefixes 45c6e4705320, 4cdb6f15f2c5,
a6230154fc19, 4032af0a08b3, e3eb129dc44b and 667ba85523cc). Test E-003 (thirteen review rounds with two independent
reviewers each, then three delta reviews of the evidence alone by one of them; the reviews, the dispositions, the
evidence and the build scripts, with SHA-256, are the record in
`/mnt/data/deriv-sota-lab/registry/round6_stage1/e003/`):

- Support: a path whose validation population is empty, holds a score that is not finite, or holds no row every
  gate kept gets no threshold and zero strategy trades, with the reason in `insufficient_support` (listed in
  `paths_insufficient_support`). A split without local support gets the same explicit row, for example an empty
  partition or out-of-fold half; fit labels the selected model class cannot fit; no signed feature column for a
  model that fits on the signed coordinates; no feature the Brier objective can split or CatBoost can use. A
  specialist side without fit or validation rows, or whose fit labels its estimator cannot fit, is skipped and its
  rows excluded, and the other side's specialist still trades; any other refusal of a specialist's fit (no feature
  the Brier objective can split, for example) refuses the whole split.
- Validity: a score's validity is read where the model produces it, before any transformation (a sigmoid, the
  agreement gate's sign comparisons, the specialists' clip, the meta gate's input, the drift-neutral booster's
  offsets). A score that is not finite never trades and no diagnostic reads it. In the CPCV harnesses one in a
  validation population gives the explicit row, and one in a test population makes only its own row ineligible; in
  the online harness it is also left out of the later threshold windows, which count valid confidences only (a
  window needs 2,000).
- Masks: a row any gate excluded trades on no side, COMBINED included, at any threshold. Before, a gate's rows at
  p = 0.5 could trade COMBINED where the threshold was 0; 5 of the 31,476 stored thresholds (paths and forward walks,
  one per coverage) are 0.
- Classes: each estimator's own requirement, so every supported case keeps its thresholds and trades. LightGBM needs
  two fit rows and fits one class, but refuses an early-stopping label its fit labels lack; the ordinal model refuses
  only an unseen validation bin and a layout whose sorted bins from the third on are not exactly the up bins; the
  scikit-learn, XGBoost and CatBoost models need two fit classes; validation labels need none.
- Reports: a model without tree gain reports no gain ranking; result files are standard JSON (a missing number is
  null) and carry `run_status`.
- Options: every numeric option is checked before anything is loaded, at the domain of the estimator that consumes
  it (each library's limits probed one per process); an option the run does not consume is not checked.
- Continuing policies (owner decision 13: forward diagnostics only): `--aci-gamma` is refused, since ACI moves its
  threshold with settled test outcomes, which E-003's stop rule excludes. The online harness and any stored result
  whose configuration sets ACI are never certified: aggregation, the result writer and the printed summary apply one
  veto (the writer and the printer to a copy), and its recorded reason keeps a reread result uncertified. None of
  the 7 stored online and 12 stored ACI results is certified, so no stored reading changes.

Owner decision 14 binds from this build on: in all five harnesses, decisions under five hours that start from 21:00:00
to 23:59:59 UTC are excluded from training, validation and test, with `--include-blackout` as well (section "Venue
tradeability"); with `--july-filter` the harnesses keep the July 2026 tradeability filter instead, for regression
against stored results only (`regression_only` true, never a candidate).

E-020: `--path-jobs N` fits the paths of one run in N forked workers (owner decision 7: 8 workers of 2 threads). In
every configuration its regression covers (listed in its log in the record: among them LightGBM and its variants,
logistic and kernel models, the base harness with three seeds, five gated configurations, the prediction dumps, all
hours and paths without support) the parallel run's results equal the sequential run's, floats bit for bit, except
the Brier-objective booster (within 1e-6 of its stored result; lab test `x-brier-threads` found two sequential runs
with the same arguments differing in their last bits) and gains summed over several boosters (within 1e-9). A
configuration the regression does not cover is not shown equal. A parallel run needs `--n-jobs` of at least 1; a
sequential run takes any `--n-jobs` the estimators accept.

Regression (lab tests in `registry/tests.jsonl`): `e003-regress13b`, 206 checks, none failed: the synthetic and
nested cases against the lab's harness; the online harness against an independent causal oracle; 107 mutants of the
evidence; the stored configurations reproduced with `--july-filter`, three in full and eleven at path 1, every
compared field equal (a missing number NaN in the stored file, null in the new one); the venue window as masks; the
arguments. `e020-regress13`, 44 checks, none failed: parallel runs against sequential and stored ones (for stored
results of older harness versions the measured fields are compared and descriptive differences listed). The
evidence the later delta reviews added (114 mutants with the final online test) ran on the same build; its outputs
are in the record above (`evidence_final/`).

F1 (installed 2026-11-19 17:39:11 UTC by `install_f1.py`, in the lab and in this repository's mirror in one step; the
replaced files, with their SHA-256, are in `tools_backups/f1_20261119T173911Z/`): `cpcv_deriv` 0.6 and `certificate`
0.5 (SHA-256 prefixes 4ef12e795e7e and 53d696fbc42c) complete the E-003 veto. The veto writes any recorded
noncertifying reason verbatim and keeps the forward-diagnostic wording of owner decision 13 only for the two
continuing policies (the online harness and ACI), whose reasons are shared harness constants; it also clears the
survival flags of the three payout-aware readings. The light reader honours a recorded noncertifying marker: such a
side reads "not certified" with the reason "never certified: <reason>" while its path statistics are still computed,
and both formatters print the policy reason beside any incomplete, invalid or unreadable flag. Apart from the reader's
own version field (`tool`, now `certificate/0.5`), rereading the stored results changes no certification: the 557
results with a strict field match it on all 1,114 sides, as with 0.4; of the 605 harness results only sides of marked
(continuing-policy) results read differently (32 stay "not certified" with the new reason, 6 read "not certified"
where 0.4 read "invalid"), and no unmarked result changes any field; the AUDUSD ledger rendered with each reader
differs only in the rows of its three continuing-policy results, whose "no" gains the reason. Evidence (record
`/mnt/data/deriv-sota-lab/registry/round6_stage1/f1/`): two review rounds of three lenses each, the second approve,
approve, and approve with one evidence-only revision (a percentile fixture), made and verified by the reviewer's own
mutant without a further round under the owner's stopping rule of 2026-11-19; `test_f1.py`, 57,669 assertions and 35
executed wrong implementations rejected; lab test `f1-regress-e10v02`: the stored `e10v02` AUDUSD configuration rerun
with `--july-filter --no-forward --path-jobs 8 --n-jobs 2` equals the stored 0.4.1 result in every field of all 15
paths (the forward walks were not run, so their summaries differ, and so does the gain ranking, which sums over every
fitted booster).

The neural adapter `cpcv_neural` 0.3 (stages 9 and 10; lab only, no repository mirror; installed 2026-11-19 20:08:23
UTC by `install_neural03e.py`, the replaced 0.2 in `tools_backups/neural03_20261119T200823Z/`; SHA-256 prefix
e4ec2824463a): for datasets of the lab's builder each row's dependency start follows the builder's semantics (own
windows in observed bars with the predecessor bar, both as-of endpoints of every peer return on the peer's own grid,
the USD aggregate from all six peers, capped ages, the temporal recipes' 129 minute buckets, the first day as the
start of short prefixes); S is the longest span on the complete dataset; purge P = 60H + LAG + 60 and embargo
E = 60H + LAG + 1 + S (LAG, the manifest's 1-second settlement lag), outside and inside the inner partition. On
`dataset_dev_v0.1` at H = 15: S = 363,840 s (AUDUSD 363,900 s), E = 364,742 s (AUDUSD 364,802 s), P = 961 s, against
H for the incumbent harness. The six recursive EMA inputs are refused (no finite memory), so the default input set
is 60 of the plan's 66, an amendment recorded before any run produced a result; the nine pandas rolling-moment
inputs are recomputed from `ret_1m` with each row a function of its own window (on the development data close to the
stored float32 columns, not equal: absolute differences at most 2.4e-10 for the volatilities, 4.8e-7 for the two
ratios, 1.02e-6 for the skew and 1.53e-5 for the kurtosis, with the same missing-value masks; record
`realdata_moments_compare.json`); the integer-count inputs need the raw `ticks_per_bar_15m` finite in [0, 2^40];
stale ages are validated before conversion; a refused load writes a failure record. Evidence (record `/mnt/data/deriv-sota-lab/registry/round6_stage9_10/
neural03/`): five review rounds (the last a delta review: causal safety approve with no finding, test evidence and
failure semantics approve with one Minor each, the first fixed in the tests, the second, a gap in failure records for
direct invocations, deferred because `run_batch.sh` records failures itself); 81 suite groups and 88 of 88 wrong
implementations rejected, rerun independently (90 of 90 after the tests' follow-up for the first Minor, the harness
unchanged); the 182 pilot and A-068 arms reached their first fit on the development data with the P, E and inputs
above.

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

**The venue window (binding from harness cpcv_deriv 0.5, owner decision 14 of 2026-11-18).** With New York on
standard time the venue also refused every contract under five hours from 21:00:00 UTC, which is 16:00 New York:
"Trading on forex contracts with duration less than 5 hours is not available from 21:00:00 to 23:59:59" (one day,
met by the commissioning of the quote survey; section "Venue payouts", "Quote survey"). From stage 1 on, every
evaluation also excludes, from training, validation and test, the decisions under five hours that start from
21:00:00 to 23:59:59 UTC (`cpcv_deriv.venue_open`, recorded as `venue_window_rule`), with `--include-blackout` as
well; only `--july-filter` (regression against stored results) leaves them in. With New York on daylight time the
window lies after the 17:00 New York end of the session. Stored results stay as they are; look 2 of the two USDJPY
finalists binds on the starts the venue sells, and both readings are reported.

## Venue payouts (2026-11-17)

What Deriv pays for a 15-minute Rise/Fall contract is quoted per purchase and varies. The evidence is a quote archive
written on the trading server by `scripts/deriv_quote_workers.py --audit` and copied to this machine on 2026-09-28
(`/mnt/data/binary-alpha-research/next-steps/binary-algo-study/dev-logs/quote_audit/`, outside this repository;
29 non-empty files, SHA-256 of their concatenation `a59fc6f6…8046a`). Its sampler logs one Rise and one Fall proposal
at stake 1.00 per pair every 30 s for six pairs (no EURUSD lane): 1,007,892 snapshots in July 2026, of which 362,990
are fresh (quote age under 5 s) and valid, 574,578 stale and 70,324 invalid; 19 New York dates have session quotes.
Everything below is development data. The tools read an archive directory only if every `.jsonl` file in it is a
regular file named `YYYY-MM-DD.jsonl[.N]` for a day up to 2026-07-31 (the archive's are named 2026-07-02 to
2026-07-20), skip and count any quote dated at or after 2026-08-01T00:00:00Z (the archive has none), open tick files
dated up to 2026-07-31 only, and count no contract that would expire at or after that instant as settled. A file name
does not show what a file holds: the archive is taken as a development export, and the row-level skip is the check
on that.

**The venue study** `[six pairs pooled·15m]` (`scripts/venue_quote_study.py`, lab test `v1-quotestudy-jul26`,
`results/json/deriv_venue_quote_study_2026-07_result.json`; rules fixed in the tool before it read an outcome - the
result records what had been seen before that: outcome-free payout tables of this archive and the owner's July test
T92; settlement by `deriv_dataset.settle` with the decision at the quote's sample time, ties lose). 171,148 moments
have a fresh quote on both sides; 81,301 of them fall in weekday New York 08:00–16:35 and 81,278 of those, on 19
dates, settle in development.

- *Level.* In that window the mean payout is 1.789 (Rise) and 1.788 (Fall) and the mean breakeven 0.560 per side, so
  the two implied probabilities sum to 1.12. By New York hour the mean payout is about 1.80 on each side from 08:00
  to 16:00 and 1.61 (Rise) / 1.65 (Fall) in the hour from 16:00 (`results/json/deriv_venue_quote_model_v1_result.json`,
  `date_blocked_out_of_fold_by_new_york_hour`). The quote model prices a weekday × five-minute clock cell only if the
  archive has at least 30 fresh matched quotes in it; no cell does between 16:35 and 20:00 New York (20:30 on
  Tuesdays, 21:15 on Wednesdays, none after 16:35 on Fridays; `not_offered_new_york_clock_by_weekday`). From 16:35 to
  18:00 that agrees with the venue's rejection of contracts expiring in 16:50–18:00 (section "Venue tradeability").
  From 18:00 to 20:00 the tradeability filter admits decisions that the archive cannot price, and it does not show
  whether a contract was on offer then: the frxEURUSD contract list of 2026-06-03 blacks out forward starts from
  20:00:00 to 23:59:59 UTC (16:00–20:00 New York in summer; `docs/deriv_frxEURUSD_contracts_for.json`,
  `forward_starting_options`), and whether that holds for a purchase at the current price, and how either window
  moves when New York leaves daylight saving, is not on disk; the quote survey below is to show it (owner
  decision 9). Its commissioning met the winter close once, on 2026-11-17 ("Quote survey" below; open decision
  14).
- *The skew between the two sides follows the price path and fails its forecast test.* Among session quotes with
  unequal payouts and a non-zero past move, the side Deriv pays less (the "favoured" side) has the sign of the past
  900 s move only 0.154 of the time (300 s: 0.206, 60 s: 0.363): in 0.846 of them the lower-paid side is the one that
  fades the last 15 minutes. The lower-paid side is paid 1.73 on average and the higher-paid side 1.85. As a forecast
  the skew fails its registered test: AUC of the implied probability of Rise for a rise, among outcomes that moved,
  0.5091, date-cluster CI95 [0.4874, 0.5343] (K1 KILLED; an interval that covers 0.50 does not show that the skew
  carries no information). The favoured side won 0.4985 against an implied 0.5161 (K3 NOT CALIBRATED).
- *None of the four registered quote-only policies pays* (K2 SUPPORTED): expected value per unit staked of always
  buying the favoured side −0.138 [−0.164, −0.114], the other side −0.108 [−0.137, −0.078], Rise −0.112, Fall −0.134.
  Where the two payouts differ most (0.18 or more) the better-paid side won 0.4881 at a mean payout of 1.90 (−0.072
  [−0.154, +0.007]) and the other 0.4927 at 1.69 (−0.170).
- *The quote can be rebuilt from the price path* (B RECONSTRUCTIBLE): out of fold on date blocks, a model of the
  pair's own ticks and the clock explains 0.99 of the variance of the skew and 0.95 of the overround, and lands within
  0.02 of the quoted payout for 0.91 of quotes; 0.90 of the skew model's gain comes from one feature, the position of
  the price in its trailing 900 s range.
- *The books run in July selected the lower-paid side.* Of the demo executor's 1,492 selected signals
  (`proposal_selected` events, not purchase confirmations), 821 have a quote within 30 s; the selected side was the
  lower-paid one in 0.949 of them (own payout 1.73, opposite 1.86; no outcomes used).

**The quote model** (`scripts/venue_quote_model.py`, lab test `v2-quotemodel-jul26`,
`results/json/deriv_venue_quote_model_v1_result.json`; model files in the lab at `cache/venue_quote_model_v1/`,
checksums in the result). Two LightGBM regressors, skew and overround, fitted on the 168,335 fresh matched weekday
quotes with no pair identity among the features, so that a pair without a quote lane can be priced. Date-blocked out
of fold on session quotes: R² of the skew 0.978, of the overround 0.939, payout within 0.02 for 0.89 of quotes and
within 0.05 for 0.986, mean payout within 0.001 of the quoted mean. Leaving one pair out: R² of the skew 0.967 to
0.990, favoured side named correctly 0.992 to 0.997, mean payout off by at most 0.0085. Its weakest cell is the hour
from 16:00 New York (R² of the skew 0.83 and of the overround −1.52, payout within 0.02 for about two thirds of
quotes, mean payout 0.003 to 0.004 too high).

**The payout-aware reading** (`scripts/analyze_payout_aware.py`, lab runner `tools/run_payout.sh`). A recorded
15-minute candidate is re-run with `--dump-preds`; its trades are selected with the harness's own rule (side,
confidence at or above the path's validation threshold, first-come de-overlap), and the tool checks that their count
and win rate equal the harness's and the recorded result's on every path - a necessary check, not a comparison of
trade identities. Each trade is priced with the payout the quote model reconstructs for its side at the decision; a
trade whose weekday × 5-minute clock cell the model does not price (fewer than 30 fresh matched quotes in the archive)
is left out and counted, which is not a record that no contract was on offer. Per path: expected value per unit
staked = mean(win × payout) − 1. Reading, fixed in the tool before a dump is opened: PAYOUT-CLEARED iff at least 12 of
the 15 paths have 25 or more priced trades, the expected value is above 0 on at least 12 of them and the 10th
percentile of their expected values is at least 0. It is an estimate, not a venue record and not a certificate, and
rests on three assumptions that nothing on disk can check: (1) July's pricing rule held on the other development
dates; (2) the daily pattern (lower payouts in the hour from 16:00 New York, no priced cell after 16:35) follows New
York time in winter — the tool also reports the reading with the pattern held at July's UTC times instead; (3) EURUSD,
which has no quotes in the archive, is priced by the other six pairs' rule. The 30-minute keys cannot be priced: the
archive holds 15-minute quotes only. A side whose trades do not reproduce on all 15 paths is reported as not readable.

Fourteen stored 15-minute candidates were read this way, both sides of each: first one candidate per pair and two
for USDJPY (lab tests `a16pay` to `a23pay`), then the leader of every key side those had not covered (`a24pay` to
`a29pay`). Two side leaders cannot be read: GBPUSD DOWN `r4fixldn` (a rule harness that writes no prediction dump)
and USDCAD DOWN `e36aci` (an adaptive threshold the tool does not replay). Of the 28 sides, 27 are NOT CLEARED AT THE
QUOTED PAYOUT and one is PAYOUT-CLEARED: `c2nysabs` `[USDJPY·15m·UP]` (`results/USDJPY_RESULTS.md`). That reading
is a development-data estimate of a candidate picked from the whole search and is not a certificate: `c2nysabs` is
not one of the registered holdout finalists and nothing here is scored on holdout data. Per-key readings are in
`results/<PAIR>_RESULTS.md` (DERIV-FEED notes, "Payout-aware reading"); their averages are unweighted means over the
counted paths, and their trade totals count a decision once for each path it sits in.

**Tool builds.** The study, the quote model and the fourteen readings above were produced by build 0.1 of the three
tools (commit `282718c`). Build 0.2 rewrites no stored result and changes none of the fourteen readings; it adds
refusals and one counting rule. The reading tool reports a side as NOT READABLE unless its frozen trades reproduce the
harness result and the recorded candidate on every path, each of which must hold the 15 CPCV paths, each once and no
other. It refuses a `--last` after 2026-07-31, a horizon other than 15 minutes, an adaptive-threshold run, a dump
without those 15 paths, any tick, decision or contract expiry at or after the cutoff, and an output file that exists
(it writes through the file it created and stops if the path no longer names that file). It counts a path on its
priced trades alone, as the registered reading is worded; 0.1 also required a priced always-side decision on the path,
which all 348 counted paths of the fourteen readings have. The study refuses an archive directory as described above,
runs its LightGBM fits with the deterministic setting and names the `corr_skew_with_returns` keys `z_*`; `payouts()`
of the quote model refuses a tick or decision time that is not a finite time before the cutoff.
`scripts/update_records.py` renders readings of both builds, prints no number of a NOT READABLE side and stops unless
each side holds the 15 CPCV paths, each once.

Regression before the swap (lab test `x-regress-venue-02e`, 3,120 s, 104 checks, none failed): on fresh harness runs
with the recorded arguments, the 0.2 reading tool gives `sides`, `quote_model` and `decisions_priced` equal to the
stored reading for 14 of 14; two runs of study 0.2 are equal to each other and to the stored 0.1 result outside its
part B, and inside part B one fitted number differs, by 3.99e-07; the model build gives both model files with the
stored checksums; every refusal and NOT READABLE case fires. Four independent read-only reviews preceded the swap: the
first two ended in reject, the last two in approve with revisions, with minor findings only.

Two facts about build 0.1 found on the way. Its study is not bit-reproducible: a re-run of its fit code differed from
the stored result in two fitted numbers of part B at the seventh decimal (lab test `x-regress-venue-02`), with no
verdict changed. Its reading tool imported `cpcv_deriv` first, which puts this repository's `scripts/` ahead of the
lab's tool directory on the import path, so its other modules came from `scripts/` whenever a file of that name
existed there. The lab copies were last modified before the first reading started; the `scripts/` copies of the three
tools were created while the readings ran, were byte-identical to the lab copies and stayed unmodified until build
0.2 replaced them; and build 0.2, which imports `cpcv_deriv` last, reproduces all fourteen readings.

**What follows for the program.** 0.541 is about the breakeven of the higher-paid side only (mean payout 1.846, the
breakeven of which is 0.542). The lower-paid side is paid 1.73 on average (breakeven 0.578), and in 0.85 of quotes it
is the side that fades the last 15 minutes; a trade from 16:00 to 16:35 New York is paid about 1.63 (0.613). A win
rate is therefore not comparable across candidates that buy different sides of the skew, and the quantity to
maximise is the expected value at the quoted payout. The 0.541 path rule stays the certificate of record (it is the
goal's). Since owner decision 5 (2026-11-17) a deploy recommendation also needs PAYOUT-CLEARED of the payout-aware
reading, and every run reports both.

**Venue-price tests** (candidates raised by this evidence after discovery round 6 was recorded; none is
pre-registered or run, each adds to the search-wide multiple-testing count, and none is in the round-6 matrix counts;
the last column is where each stands after the owner's answers of 2026-11-17):

| id | test | status |
|---|---|---|
| V-03 | The harness reports the payout-aware reading beside the 0.541 rule for every 15-minute run (no new fit) | owner decision 5: every run reports both |
| V-04 | Selection by expected value: calibrate the probability on the validation block, trade the side whose probability × reconstructed payout − 1 is highest, threshold = validation quantile at the coverage; compare with confidence-ranked selection at equal coverage | released by owner decision 8, after stage 1 |
| V-05 | The reconstructed skew, overround and own-side payouts as model inputs, so a model can learn where the price paid is wrong | released by owner decision 8, after stage 1 |
| V-06 | Profit-shaped training: weights or targets built from win × payout − 1 per side instead of the sign | released by owner decision 8, after stage 1 |
| V-07 | Better-paid-side specialist: the label is whether the better-paid side wins, on decisions where the two payouts differ by 0.05 or more | released by owner decision 8, after stage 1 |
| V-08 | The last New York hour (16:00–16:35) dropped from training and evaluation; re-read of stored candidates without it | released by owner decision 8, after stage 1 |
| V-09 | A payout floor chosen on validation (1.75, 1.80 or 1.85) as a gate on the trades of stored models | released by owner decision 8, after stage 1 |
| V-10 | Rise/Fall Equal contracts (if a tie wins, as the name says and the contract list does not state; tie share 0.019 in the study's window): needs their quotes and tie rule | quotes: the quote survey; the tie rule stays unverified without a contract record (open decision 9) |
| V-11 | 30-, 60- and 120-minute quotes, to price the 30-minute keys and longer horizons | the quote survey, which also takes 240 minutes |
| V-12 | A quote sample taken now on all seven pairs: tests whether July's rule still holds and which clock the daily pattern follows in winter | the quote survey; its reading needs a procedure pre-registered for it |
| V-13 | The server's quotes after 2026-07-31, to price holdout-window trades at a registered look only | not decided (open decision 9), and the holdout register |

Cluster `C-075` of the round-6 record ("reject variable-odds decorrelation losses for the fixed-payout target") rests
on a fixed payout; with a quoted payout that premise does not hold, and V-04 and V-06 take its place.

**Quote survey** (owner decisions 9 and 10 of 2026-11-17; `scripts/deriv_quote_survey.py`, build
`deriv_quote_survey/0.7`; its rows and README are in `/mnt/data/deriv/quote_survey_2026-11/`, outside this
repository). One public connection without a login. Every API request the process sends leaves through one gate at
the socket. The gate passes three kinds of request, each with a fixed set of fields (a price request for CALL, PUT,
CALLE or PUTE at a stake of 1, with or without a subscription; a contract list; the trading times), and stops the
process before anything else is sent. Requests leave one at a time and no sooner than 2.5 s after the one before: at
most 24 a minute, the pace of the tick backfill, with nothing saved up and no exemption for lane subscriptions or
metadata. The pace holds from one process to the next: a process sends its first request no sooner than 2.5 s after
it took the connection lock, and at its end it keeps the lock until its next request could have left. No request is
let through once the survey is over (a request let through before the end can still leave after it if the process is
held up before the socket sends it, or the connection holds it back). It has two ends, the same moment on two
clocks: the fixed end time by the wall clock, and a reading of the boot clock, which goes on through a suspension
and cannot be set. Both are in one record in the survey directory (`survey_record.json`), which is made once, before
the first process is started, and both are on the command line of every process, so neither a wall clock that is set
back, nor a process that replaces another, nor a later start by hand makes the survey run longer. A row that holds
what the venue said (a quote, a contract list, the trading times, or the refusal of one of them) gets its time at
the moment it is judged against both ends, and none has a time at or after the end; a diagnostic event row (a failed
resubscription, a stream error) is not held to the end and can carry the venue's error text after it. Nothing leaves
for 75 s after the venue rejected a request as too frequent: the rest begins as the rejection is read, whatever it
answers. For at most seven days (the record holds the first start and the end; an end more than seven days ahead is
refused) it records:

- `quote_snapshot`: the 15-minute Rise and Fall quote of all seven pairs at stake 1.00, as proposal subscriptions
  sampled every 30 s by the code that wrote the July archive, so the two samples compare row for row (V-12) and
  EURUSD gets the lane the archive lacks. The sampler reads memory only and keeps its schedule whatever the network
  does; a row repeats the lane's last quote, and `quote_age_ms` says how old that quote is;
- `quote_poll`: unsubscribed requests, the two sides of a contract one after the other, for Rise/Fall and Rise/Fall
  Equal at 15, 30, 60, 120 and 240 minutes on every pair at stake 1.00 (V-10, V-11 and the lengths of owner
  decision 10). A cycle is 70 such units and takes 350 s; the first cycle of a process takes about 385 s, because
  it also subscribes the 14 lanes, and the one after it 370 s, because the eight metadata requests come before it;
- `contracts_for` of each pair and `trading_times`, after the first cycle of a process and then every six hours.

A row is the record of an outcome: a request that was in flight when a process was killed has none. The count of
the requests that left, and the smallest time between two of them, are the gate's own figures in the state rows.

One connection, never repaired. A process connects once. It ends with exit code 1 when the connection cannot be
opened or is lost, when a subscription request ends without a clear answer (the venue may then hold a stream that
nothing reads), when three requests in a row get no answer, or when a lane is dead: it has had no update for 300 s
although the poll that has just priced its contract shows a later tick. A lane in a market without ticks is quiet,
not dead. The rule is a heuristic that rests on a live lane getting an update with every tick. In a ten-minute run
with the market open (2026-11-17, 20:48 to 20:58 UTC) each of the 14 lanes had one every second, 5,205 of 7,833 of
them without a change in spot, ask or payout; other hours were not measured. systemd starts a new process 90 s
after an exit with code 1, and that process subscribes every lane afresh. A lane is subscribed when a poll has
priced its contract, and not before. A cycle without one priced side (the market is closed) switches to one unit a
minute, going through the ten kinds of contract and the pairs in turn, until a side is priced again. The lanes are
not checked while their contract is not priced, so a reopening can cost one restart. The 15-minute quote at stakes
10 and 50 is not in the owner's option: the tool does not have it and refuses any stake other than 1 (decision 9
below).

One connection to Deriv at a time. The survey takes `/mnt/data/deriv/.connection.lock` itself, once it has checked
its arguments and that its end time has not passed, and holds it until its process ends. The tick backfill service
and the tick downloader take the same file: the downloader from version 1.5 also when it is started by hand, and
from version 1.6 whatever `--data-dir` it writes. No option of any of the three names another lock, and each ends
with exit code 75 while another holds it. The survey runs as a transient systemd unit that starts it again after
90 s on exit code 75 or 1 and not on 0 (the survey is over, a stop request) or 2 (a command line that cannot
work). Both ends are on the unit's command line, so no restart extends the survey. A second transient unit, a
timer on systemd's monotonic clock, stops the survey's unit 300 s after the earlier of the two ends, as a further
line. A script in the survey directory (`start/start_survey.sh`, with one helper) is the only way the survey is
started. It takes both ends from the record, and only when every start receipt and every start row in the
directory is of that record (a receipt of each boot's reading is written before the record holds it, and a
boot that a receipt has and the record lacks is taken back from that receipt); holds the
connection lock itself for 90 s before it starts anything, which is longer than the rest of 75 s that a killed
process may have left unfinished, and then looks at all its conditions again, the checkout and the launcher
among them; starts the limit, at a moment of the monotonic clock that it fixed before, and then the survey with
a nonce of this start in its note; and stops both again when the start row with that nonce, the checkout, the
launcher, the timer or the lock is not as intended. The tick backfill waits until the survey has ended, and
then for its own follow-up (below); the
archive's last tick day is 2026-11-13. The limits that remain are in the survey's README: a suspended machine
stops the survey for as long as it is suspended; a restart of the machine ends both units, and the survey is then
started again with the same script, which takes the end from the record and a reading of the new boot's clock for
it, and only from a wall clock that the system reports as synchronized, so that across a restart of the machine
the end is as good as the wall clock; a link that stalls, or a change of the machine's default route, ends a
process, and the next one starts 90 s later; a process that is killed cannot keep the lock through a rest, which
the restart delay of 90 s covers; and a rate-limit rejection that the venue sent and the process never read is
unknown to it.

The week lies after the development window. Its usage rule is the owner's: payout levels and availability only; no
candidate is scored on those dates and nothing is selected or tuned on them. A reading of the sample needs a
procedure pre-registered for it; until then only operational checks are made (row counts, error codes, latency).
`venue_quote_study.py` refuses the survey directory as it stands, its files being dated after 2026-07-31.

Reviews. Six builds, and then the start files of the last in ten more rounds, went to independent read-only
reviewers, each before any commit. Build 0.1 was rejected with
ten findings. The blocking one: polls, lane repairs, metadata and reconnects shared no request budget (46 requests
possible in the first minute, and 1,400 requests over 100 reconnects in the reviewer's in-memory probe). Build 0.2
put a budget and an allowlist at the socket and was rejected by both of its two reviewers, with sixteen findings
between them. The blocking one, found by both: the budget let two requests be saved up and sent the
resubscriptions of a new connection at once, charging them afterwards, so that 25 requests left in under a minute
of ordinary running and 36 in a minute around reconnects, and it knew no pause after a rate-limit rejection.
Build 0.3 was rejected by one of its two reviewers and approved with revisions by the other, with fifteen findings
between them, all of which held against the code. The two blocking ones: an option let a caller name another lock
file, so that two tools could each hold a lock; and the end of the run was read from the wall clock alone, so that
a clock set back extended it. The others of weight: neither the pace nor the rest after a rate-limit rejection
carried over from one process to the next; a rate-limit rejection that arrived after its request had timed out was
not seen; a quiet lane was taken for a dead one; an answer that arrived after the end could be stored as a quote;
and the README let a missing row be read as a request that was not made. Build 0.4 was rejected by both of its
two reviewers, with ten findings between them, all of which held against the code. The blocking one, found by
both: the limit on the boot clock began anew with every process, so that after the wall clock was set back a
process that replaced the first ran past the end. The others on the survey's side: a stop could erase a
rate-limit rejection that had already been received; a sample of 14 rows could cross the end; the two clock
readings behind the limit were not taken together; an answer in the instant of its timeout was kept without end;
the wait at the end of a run did not count the writing of the last row; and three statements of the README said
more than the code or the evidence did. Build 0.5 put both ends on the command line of every process and was
rejected by both of its two reviewers, with fourteen findings between them, all of which held against the code.
The blocking one, found by both: the start script took a new reading of the boot clock at every start, so that
after the wall clock was set back a later start by hand gave the survey that much more time. The others of
weight: a priced row could carry a time after the end, because the end was looked at before the row's time was
taken; a torn start row made the check after a start fail, which stopped a healthy survey; a start by hand a
minute after a killed process could break that process's rest; and the start script had no check of its own.
Build 0.6 is the rework: the record, a row's time from the gate, the 90 s under the lock, and a rehearsal of the
start script. The author's own reading of it found one kind of row that the rework had left out (a contract list
that the venue refused was written with the writer's time); build 0.7 is build 0.6 with that put right, and is
what is described above. One finding of the fifth review is declined: the test for a rate-limit rejection is the
tick downloader's and goes by the text of the error, so an error that only reads like one starts a rest of 75 s.
It errs toward resting, the row keeps the venue's code and text, and the test changes for all tools at once in
the follow-up below. One finding of the third review stays declined: importing the quote helpers imports
modelling code that makes a directory before the arguments are checked, and the shared modules are not changed
here. Build 0.7 was rejected by both of its two reviewers, with twelve findings between them, all of which held
against the code and none of which needed a change of the survey file: the blocking one, found by both, was that the
start helper took a record of another survey although the start rows in the directory named this one, which could
have given the survey more than seven days. Two of its findings are declined with their conditions written into the
README: a rest that a rate-limit rejection sets between a request's turn and its leaving, which cannot happen in the
survey's process while its interpreter (Python 3.12.3) and the shared client are unchanged, and a start row later
than 30 s, which undoes the start. The start files then went through ten more rounds. The seventh review (ten
findings) found that a boot whose start ended before its start row, a directory whose only start row was torn, and a
boot entry added by hand were bound by nothing but the record: start receipts written before the record, a nonce for
every start, and the limit's moment on the monotonic clock followed. The eighth (nine findings) found that a boot
taken back from its receipt lost the receipt's time, that a receipt cut before its newline could be followed by a
differing one, and that a row file cut and grown again hid the start row: receipts' times are kept and counted, a
receipt without its newline is completed, the start row is found by the start's nonce and the process's number in
whole reads within 30 s, and a rule for the week keeps the loaded code from being changed and put back. The ninth
(six findings) found that the clock check, the survey's reading of the end and the receipt's time were not one
reading of the clocks, so that a wall clock set back in between could, through a stopped start and a start on
another boot, end the survey late, and that a receipt could be relied on before it was synced: the three now share
one reading of the clocks, a new reading must match it, the receipts are synced before anything is chosen, and the
look at the start row is stopped after 40 s. The tenth (four findings, found by both reviewers) found that a reading
of the survey's up to 2 s later than that one reading of the clocks was still kept, that a retry could rely on a
record that was not synced, and that after a failed check the units were stopped only after other checks without a
bound: the stored reading is now never later than the end by the checked clocks, the record is synced before a start
relies on it, and the first failed check after the start stops both units at once. The eleventh (five findings)
found that the times of the clock checks were written to the second, so that a wall clock less than a second behind
a check could pass, that the cut-down end was computed in floats, and that not every look after the start was
bounded as a whole or had its status checked: the times are kept to the microsecond, the end is computed exactly,
every look after the start is bounded as a whole with a kill after its time, a failed look undoes the start, and the
record is written whole or not at all; a first start that stops between its receipt and its record stays a repair by
hand. The twelfth (four findings, found by both reviewers) found that a first start that stopped after it linked the
record left a second name through which a later write could empty it, that a survey reading earlier than the checked
end was kept, that the journal and the making of the units were not checked like the other looks, and that across a
restart the end rests on the synchronized wall clock: the record is written into a new file every time, the stored
end is always the end by the checked clocks, the making of the units is bounded and a doubtful one undone with the
limit's timer kept until the survey's unit is seen stopped, and the journal is a look like the others; the wall
clock across a restart stays the stated limit. The thirteenth (two findings, found by both reviewers, who both
approved with revisions) found that a later boot's end on its clock can be up to about 0.051 s later than an earlier
boot's, though no request is let through after the end, and that two comments still said that the stored end is
taken from the survey: the comments are corrected, and the difference between boots is declined and stated in the
README. The fourteenth (two findings, found by both reviewers, who both approved with revisions) found that the
precision of the end on a later boot was stated too tightly and that the output of the checks on the earlier helper
lacked that helper: both are corrected. The fifteenth and sixteenth, of one reviewer each and on words and evidence
only, found that a probe did not show all it was said to, that the README said nothing leaves once the survey is
over where a request let through before the end can leave after it, and that diagnostic event rows can carry the
venue's error text after the end: the probe is made decisive, and the README says what the gate guarantees and
states the rest as limits. One weakness stays open and is not the survey's to fix: the shared quote parser takes a
payout of infinity for a price (`docs/DERIV_DEMO_EXECUTOR.md`, under the parity re-check rule); the survey stores
such a quote as `non_finite_quote`.

Deferred, with a rule. Two findings of the fourth review are on the side of the tick downloader and the backfill
service (archive tools 1.6). The downloader's `time` request is outside its pace (25 to 26 requests in a rolling
minute when it takes over from a survey), it accepts a pace of up to 28 a minute, and it does not keep the lock
through its own pause after a rate-limit rejection; and two downloaders that inherit one lock descriptor from a
parent would both be admitted (no shipped path starts two). Neither can occur as long as neither tool is
started. That rule is what holds, not the lock: a survey process gives the lock back when it ends, and the next
one takes it 90 s later. From one survey process to the next 24 a minute holds. **Neither the tick downloader
nor the backfill service is started again before one owner of connection admission for the archive tools is
built and reviewed**: the time of the last request and the time before which nothing may leave kept under the
connection lock, every request of the downloader paced, 24 a minute for every tool, no borrowed lock descriptor,
and a rate-limit rejection told by the venue's code before its text. That follow-up comes before the next tick
top-up. Until then, by hand: another tool no sooner than 90 seconds after one has ended, and no sooner than 15
minutes after a downloader that was rate-limited.

Checks. `--smoke` runs whole surveys on a virtual clock against an in-memory venue, in a process that can reach
neither the network nor the archive's lock file: 92 checks. Each of 114 deliberate breaks of the script, one at a
time, fails the check it is paired with. The launcher's check has 27 checks; it runs the checkout's survey inside
a network namespace without a route and a mount namespace in which the archive directory is an empty stand-in,
and refuses to run it anywhere else. Twenty-three broken launchers, surveys and places each fail it. The backfill
service's suite has 296 checks; six broken variants of the tools' lock handling each fail it, and a copy of the
suite that forgets its fixture lock is stopped by the suite's guard on the archive's lock. Under systemd, with
stand-ins: the unit's four exit codes; the limit timer stopping a running unit and a unit between two starts; and
the start script itself in 61 cases (the conditions under which it starts nothing, also when the checkout, the
launcher or the commit changes during its wait; a first and a later start; a start after the wall clock was set
back and one after a restart of the machine; a torn start row, and start rows written around a wall clock that
was set back; a unit that does not start; records and ends that cannot be started, among them a record of
another directory, one of another survey, and records put right by hand after a start that ended before its
start row; and a start row, a survey file, a checkout that git cannot read, a timer or a lock that is not as
intended, a start row that comes late, a look at the start row, the checkout or the limit that hangs, a query
of the service manager that fails, a lock or a journal that cannot be looked at, or a survey unit whose making
fails or does not end, after which nothing of the survey runs; a survey unit that cannot be stopped, whose
stop does not end or whose state cannot be seen, which keeps its limit timer; and a wall clock set back before
the survey's reading of the end, which starts nothing).
Twenty-nine checks of the start helper run without a service manager, each check added after the ninth to
twelfth reviews failing on a helper that lacks what it checks; the same rehearsal on a script that takes the survey
file's checksum after the launch fails the one case that is to catch that.

An unplanned run. When the deliberate breaks of build 0.2 were first run, the break that removes the lock let the
smoke go on into the survey itself: 2026-11-17 16:38:18 to 16:40:39 UTC on the public endpoint, one connection
without a login, 58 requests in 141.6 s (the survey's pace) and 44 of 44 polled sides priced, while no tick
download was running. That broke the rule that checks use no network. Since then the smoke calls nothing that can
start the survey and its process refuses every name lookup, every connection and the archive's lock file, which a
check holds in place; every deliberate break and every broken variant runs with neither a network nor the archive
directory; and the launcher's check runs the survey only in such a place. The rows are kept with the commissioning
files.

Trials of builds 0.6 and 0.7 on the public endpoint and on the archive's lock file, 2026-11-17 23:36 to 2026-11-18
00:59 UTC: the survey as a transient unit with the properties of the real one, with a record, the limit unit and the
start helper's checks, and with a kill, a stop and a start by hand on the way (the driver, its output and
operational readings of the rows: `commissioning_2026-11-17/trial_0.6/`, `trial_0.7_stopped_early/` and `trial_0.7/`
in the survey directory). Build 0.6 ran as six processes that wrote 1,635 rows, build 0.7 as four that wrote 761. In
both, the last process ended by itself at the end with exit code 0, a process started after the end sent nothing,
and the limit found nothing left to stop. Every row of a quote, a contract list or the trading times has a time
before the end (1,552 and 705 rows). No request was rejected as too frequent, no two requests were less than 2.5 s
apart, and at most 24 polls left in any 60 s. A killed process was replaced 93.5 s and 91.2 s after the kill. The
trial of build 0.6 ran through the end of the venue's daily close (below). Not seen in any trial: a process that is
in closed mode when the venue reopens, and lanes that were subscribed before a close.

Two things went wrong in the trials, neither in the survey. The author's driver started a process by hand 2.8 s
after it had stopped one, against the rule of 90 s between tools (the stopped process had ended in order,
without a rejection and with under 0.05 s of its own wait left, and a new process waits 2.5 s before its first
request: the venue was not asked faster than the pace). A first trial of
build 0.7 with the same driver was stopped by hand after four and a half minutes; the driver was corrected and
the trial run again, with 92.3 s at that step. And from 23:53:43 UTC the machine was on a wireless link on which,
every two to twelve minutes, requests had no answer for 15 to 30 s: of the nine processes that ran on it, two
ended for a subscription request without an answer and one could not open its connection, and each was replaced
after 90 s. From 00:48:46 UTC a wired network was the machine's default route; in the one process that ran on
it none of the 68 polls timed out, and no answer took longer than 0.6 s. With fourteen lanes the connection
received about 10 kB a second: about 0.9 GB in 24 hours of open market.

Commissioning of build 0.4 on the public endpoint and on the archive's lock file, 2026-11-17 20:48:48 to 21:06:51
UTC: two processes one after the other, of ten and of eight minutes (readings and check outputs:
`commissioning_2026-11-17/` in the survey directory, with a README). Both ended with exit code 0 at their end
time, each on one connection, without a rate-limit rejection or a late answer. The gate counted 239 and 191
requests (217 and 169 polls, and in each 14 lane subscriptions and 8 metadata requests) and measured no two less
than 2.5 s apart. The first cycle took 385.5 s and 385.2 s. The second process wrote its start row 2.4 s after
the last row of the first and sent its first request 4.9 s after it. The largest resident memory was 206,040 kB.
A wrapper counted the stream updates of the first process (the figures above). The second ran through the
venue's daily close (below): its lanes were silent for the last seven minutes, no poll priced their contract,
and it ended at its end time.

Build 0.3 had been run the same way from 18:57:03 to 19:07:00 UTC (exit code 0 at the end time, 7,837 stream
updates over the 14 lanes, the first cycle of 70 units in 382.9 s with 140 of 140 sides priced, 239 requests),
and build 0.2 from 17:03:48 to 17:11:17 UTC; a second start of the survey and a start of the tick backfill
service, both tried during that run, ended with exit code 75, and the refused start wrote no row.

Build 0.1 had been run from 15:18:56 to 15:27:57 UTC: one cycle of 84 units in 454 s with 154 of 168 sides priced.
The 14 unpriced sides were a stake-100 rung, refused with "Minimum stake of 0.50 and maximum payout of 100.00".
What that run shows about the venue, with what the later runs add:

- with the market open, Deriv quoted both contract families at all five lengths on all seven pairs, EURUSD
  included, and every priced
  contract ran exactly its length from `date_start` (182 of 182 priced rows of the build 0.1 run; 158 of 158,
  217 of 217 and 154 of 154 of the three later ones);
- the venue sells no contract under five hours for three hours a day, and says so. On 2026-11-17, with New York
  on standard time, every polled contract was refused from 21:00:03 UTC until that run ended at 21:06:49 (all
  five lengths, both families, all seven pairs) with "Trading on forex contracts with duration less than 5 hours
  is not available from 21:00:00 to 23:59:59". From 20:51:46 the 60- and 120-minute contracts had been refused
  with "Trading is not available from 21:50:00 to 23:00:00.", their expiry falling in that window, while the
  15-, 30- and 240-minute ones were priced until 20:59:55. The second text is the rollover blackout of "Venue
  tradeability" one hour later by UTC: the same New York time, 16:50–18:00. The first begins at 16:00 New York
  on that day, inside the 08:00–17:00 session and 35 minutes before the first start that the tradeability
  filter excludes for a 15-minute contract. It does not follow New York time: the July archive has priced
  quotes from 16:00 to 16:35 New York. If it is fixed by UTC, it explains why that archive prices nothing from
  18:00 to 20:00 New York, the question left open above; the texts do not name their clock. The trial of build
  0.6 ran through the end of that window the same night: the last of 253 refusals at 23:59:57 UTC, the first
  priced poll at 00:00:00.377 UTC on 2026-11-18, and no refusal after it. This is one day. The week's refusal
  rows are to show when the window holds, and what follows for the filter is with the owner (open decision 14);
- `trading_times` is answered on this endpoint and gives the seven pairs one schedule: open 00:00:00 to 23:59:59
  from Monday to Friday, an event "Closes early (at 20:55)" on "Fridays", and the holidays 2026-12-25 and
  2027-01-01. The answer does not name its clock;
- `contracts_for` lists CALL, PUT, CALLE and PUTE for every pair as intraday contracts of 15 minutes to one day (and
  daily ones of 1 to 365 days). Its entries carry no forward-start field, unlike the 2026-06-03 list cited above, so
  the question of the 18:00–20:00 New York window stays with the week's quotes. NZDUSD is classed under minor pairs
  and lists these four contract types only (8 entries against 20 for each other pair);
- at stake 1.00 no ask or payout in the rows of the build 0.2, 0.3 and 0.4 runs has more than two decimals. That
  is the resolution that was seen, not a rounding rule. At a payout of 1.85 one cent is about 0.29 percentage
  points of breakeven: the breakeven of a single row is known to about ±0.15 points if the venue rounds to the
  nearest cent, which is an assumption, and to within 0.29 points if it rounds in one direction;
- nine minutes are not a payout finding. The reading of the build 0.1 run printed payout levels before any
  procedure for them was registered (15-minute Rise/Fall 1.69 to 1.92 with a median of 1.82; Rise/Fall at the four
  longer lengths 1.82 on both sides in all 56 rows; the Rise/Fall Equal means are in the same output). They are
  recorded as what the tool wrote, and nothing is concluded from them. The readings made since print no quote
  value.

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
2025-11-14 18:00 UTC was no longer served, Monday 2025-11-17 00:00 UTC was). The "retention >= 26.5 years" of
`scripts/deriv_api_probe.py` probe (d) is an artifact; the mechanism and the corrected reading are in
`docs/DERIV_DEMO_EXECUTOR.md` (d). The lab archive (from 2025-08-11) is therefore the maximum Deriv-only history; it
grows only forward: about 12 months of DEV and about 15 months across DEV and the holdout. Whether this short history
explains the null results in hours where 10-year frozen books still score on the Deriv feed (R1bt; per-key results
files) is a hypothesis that no experiment here has isolated.

## Holdout discipline (registered 2026-11-15, before any holdout data is opened)

Every DEV-window variant tried for a key belongs to that key's search family (all logged in the lab registry). At most
**two finalized candidates per key** are opened on the holdout, chosen on DEV evidence only (CPCV verdict, forward walk,
no-clock ablation) and named in the key's sweep ledger before the holdout is scored. Reading per candidate
(`holdout_eval.py`): PRIMARY = walk-forward monthly refit (train on everything before each holdout month, purged),
CONTEXT = one model frozen at the end of DEV. As registered before owner decision 6 (which now governs the opened keys,
below), a side was SOTA-eligible iff the PRIMARY reading had n >= 50, it passed the Holm step-down across the key's
finalists (one-sided family alpha 0.025 on `holdout_eval.py` `p_boot_le_breakeven`: the j-th smallest of k passes iff it
and every smaller one are <= 0.025 / (k - j + 1), so its CI95 lower bound is >= 0.541 too), its point win rate beats
every incumbent book of the key scored on the same window as deployed, and - for a key under owner decision 1 - its
drift-matched excess passes. Then the full per-fold-refit CPCV at the operating gate certifies it. (Holm replaced the
first registration's single-step Bonferroni interval at 2026-11-15 13:21 UTC, before any holdout data was read; lab
`holdout/REGISTER.md`. `holdout_eval.py` still reports the Bonferroni reading as `bonferroni_context_eligible`, context
only.)

**Two-look sequential holdout (owner decision 6, 2026-11-15; supersedes the single look and the Holm step-down for the
opened keys).** A DEV-only power estimate showed that one look at the 15-week window would pass a real DEV-sized edge
only about 26-29% of the time for the USDJPY 15m kernel and 15-53% for the 30m kernel (estimates: DEV trade rate and win
rate, normal approximation). The owner chose one finalist per key - USDJPY 15m UP `e30nystroem`, USDJPY 30m UP
`e49nys30` - and a two-look group-sequential test with one-sided alpha 0.025 per key, on the cumulative PRIMARY wins
from 2026-08-01: look 1 on the window to 2026-11-14, look 2 on the window to 2027-02-26 (Deriv ticks backfilled after
that date). The boundaries are exact binomial win counts at the null win rate 0.541 (look 1 spends 0.00258, look 2 the
rest given the realized trade counts), with the owner's quoted z >= 2.80 and z >= 1.98 as floors that are never relaxed;
the exact family-wise error is at most 0.025 by construction (0.018-0.0225 on a grid of trade counts from 50/100 to
400/800; e.g. 128 wins of 199 trades at look 1, 236 of 398 at look 2). Certification at a look also needs PRIMARY n >=
50, the drift-matched test and a win fraction above every incumbent book's on the same window; otherwise the test
continues to look 2. The null model - nonoverlapping trade outcomes independent given the past, the same assumption as
the single-look CI95 rule - was checked on the finalists' DEV trades (day- and week-clustered variance inflation
0.71-0.89, no overdispersion). Both looks run frozen, hash-registered copies of the evaluation code (their whole import
closure is checked), wins are counted from the evaluator's trade list, and at look 2 the trades inside the look-1 window
must equal the look-1 trades exactly. The exact rule, finalist commands and the decision tool
(`scripts/seq_decision.py`, lab `holdout/seq_decision.py`) are in lab `holdout/REGISTER.md`.

Look 1 ran on 2026-11-16 at 10:00 UTC (lab test `hold-look1`, frozen code). Neither finalist crossed its boundary, so
neither is certified, and both continue to look 2. The numbers are in `results/USDJPY_RESULTS.md` (Deriv-feed notes,
"Holdout look 1") and lab `holdout/REGISTER.md`.

**Data after the look-1 window.** Since 2026-11-17 the quote survey (section "Venue payouts") stores quotes of later
dates, each with the spot it was quoted at. Its usage rule covers them: no candidate is scored on them, nothing is
selected or tuned on them and no look opens them. One exposure is recorded: a check of the commissioning rows printed
the default barriers of the EURUSD and NZDUSD contract lists, which show the price level of those two pairs on
2026-11-17. No outcome, score or USDJPY value was shown, and checks of survey rows print field names and counts
only since then (lab `PROGRESS.md`).

## Experiment record

Lab registry `/mnt/data/deriv-sota-lab/registry/tests.jsonl` (append-only: start, run, finish, clean for every test;
snapshot committed as `results/json/deriv_lab_registry.jsonl`). Result JSONs: `results/json/deriv_<exp>_<PAIR>_<H>m_result.json`
and `results/json/deriv_r1_frozen_books_<window>_result.json`. Per-key tables are rendered from those JSONs. The
venue study is `results/json/deriv_venue_quote_study_2026-07_result.json`, the quote model
`results/json/deriv_venue_quote_model_v1_result.json`, and the estimated payout-aware readings
`results/json/deriv_a<nn>pay_<PAIR>_15m_result.json` (one per re-scored candidate; they hold no certificate, so the
leaderboards skip them and the DERIV-FEED notes carry one line each).

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

## Literature (phase 1, 2026-11-15)

All 13,846 corpus documents were read into structured records; 21,270 levers were synthesized into 4,059 candidates
and consolidated into 2,420 items mapped against this program's runs (`research/deriv_literature_2026-11-15/`: README
with dispositions, compressed records and backlog). The ranked testable direction levers enter as `e51fx`, `e52ck`,
`e53ckcompact`, `e54ckcompactlogit`, `e55rl` / `e56rs` (with a shifted-level placebo), `e40forest`, `e41pool` and the
zero-parameter rules `r4fixldn` / `r5fixecb` (`scripts/cpcv_rule.py`: a rule scored on the CPCV paths with nothing
fitted). The high-prior items are all validation upgrades; the sign-null replay (`scripts/run_nullreplay.sh`) is a
pre-registered label-shift diagnostic of the selected configuration only - it does not replay the search, as the
literature's "full-search null replay" would. Literature blocks: `scripts/deriv_features_lit.py`
(truncation-tested), merged as `dataset_dev_v0.5` (a superset of v0.3 and v0.4, which were removed).

Literature feature blocks, measured (2026-11-15; per key in `results/<PAIR>_RESULTS.md`; paired against `e1t`, same
rows and splits). Every key-side is KILLED by the goal rule. The fixing-window block `e51fx` is the only block with a
consistent gain: the median paired path moved-AUC gain is positive on all seven pairs but between +0.001 and +0.003,
below the pre-registered 0.005 line. The kernel blocks `e52ck` / `e53ckcompact` / `e54ckcompactlogit` give no gain.
The round-level block `e55rl` beats its shifted-grid placebo `e56rs` on paths clear and p10 on only 4 of 14
key-sides; where it lifts AUC (AUDUSD), the placebo lifts it as much, so there is no round-level effect.

## Discovery loop (2026-11-15 .. 2026-11-16)

Five lens-driven rounds proposed, vetted and pre-registered new candidates on development data only. They are in lab
`PROGRESS.md`, as SWEEP_MATRIX rows N60-N74, and their lessons are in `docs/IDEAS_LOG.md`; results per key are in
`results/<PAIR>_RESULTS.md`. Rounds 1-3 were not dry under their pre-registered readings. Rounds 4 and 5 were dry: no
candidate improved on its key-sides, and none passed the goal path test on a VALID key-side. Under the dryness rule
amended on 2026-11-16 at 06:07 UTC (two consecutive dry rounds, counted from round 3), discovery ended on 2026-11-16 at
13:39 UTC. The queue's remaining sweep rows still run, and the final report follows them.

## Discovery round 6: the comprehensive test matrix (2026-11-17)

The owner reopened discovery on 2026-11-17, asking for a comprehensive set of configurations and for the GPU and the
code's speed to be examined. Round 6 is a planning round. Its record is `research/deriv_round6_2026-11-17/`: the
README, a matrix of 292 tests in 14 stages with a kill rule and a cost each, the 353 lens proposals behind it, the
110 configurations already run, the scripts and logs of its checks, and the script and inputs that rebuild the
matrix. Nothing in it is pre-registered or run: each test is pre-registered in lab `PROGRESS.md` before its first run
and adds to the search-wide multiple-testing burden. Stages 1-6 (200 tests: the evaluation contract and a prediction
cache first, then tie and participation policies, compact models, price-path blocks, ensembles and losses,
magnitude) stay inside the existing research scope (no venue call, new key, closed family or new data source); four
stage-3 tests whose specifications name the configuration race of stage 10 run without it, on a short fixed list of
settings (the README's Ranking paragraph). Release state after the owner's answers of 2026-11-17 (section "Owner
decisions (resolved)"): stage 1, then the venue-price tests V-04 to V-09, then stages 2-4, each stage only if its
gate is met and with a report at each stage boundary. Stages 7, 8, 9 and 10 are released each only when the stages
it builds on are done and its own gate is met; stage 9 builds on stage 6 and stage 6 on stage 5, and stages 5 and 6
are not released: they are put to the owner at the stage-4 report. Stages 11-14 are not released, nor the parts of
four stage 5-6 tests that update on the evaluated path (`A-039`, `D-033`, `D-035`, `D-036`), which wait for the
rest of decision 13. Of the three tests that waited for decision 7, the answer covers `E-020` and the thread
budgets of `E-018`; the part of `E-019` that runs independent jobs side by side is held, because one run at a time
stays the rule. The matrix was built before those answers and gives, for each test, the decisions its own
specification needs (`program_decisions`) and those it needs through the stages it builds on
(`upstream_decisions`). All 19 high-prior items are validation, infrastructure or venue-reconciliation work; no
predictive candidate carries a high prior.

Two findings of the round shape the rest of the program (scripts and logs in
`research/deriv_round6_2026-11-17/checks/`; the README states their assumptions and limits):

- **At about 60 trades per path the goal rule needs a true win rate near 0.60 for an even chance of passing.** The
  rule's 10th percentile is taken over the counted paths; when all 15 are counted, 12 clear paths cannot pass.
  Assuming 15 counted paths with equal trade counts and one win probability, an even chance of passing needs a true
  win rate of 0.60 to 0.61 at 60 trades per path, 0.59 to 0.60 at 100 and 0.57 to 0.58 at 200
  (`pass_probability.py`; the range is two dependence scenarios, not proven bounds). The stored 15-minute runs
  outside USDJPY have a median of 59 trades per counted path at their operating coverage, and the frequency of raw
  passes among them is compatible with chance under those scenarios (`tail_distribution.py`). That is not a
  calibrated null and does not show that no side has an edge. Under the same assumptions a side with a true rate of
  0.56 to 0.58 would still fail the rule in most runs, so a kill at this support says little about such an edge, and
  the support a key offers is counted first (`E-010` and `E-009`, stage 1).
- **On the one benchmark fit, the measured speed-up for the models in use is to run the paths of a run side by
  side, not the GPU.** LightGBM and CatBoost were slower on the GPU than on the CPU; XGBoost was faster on CUDA, but
  as a different model; the two neural benchmarks ran much faster per row on CUDA, though both CPU fits completed; a
  GPU kernel transform is untested (`results/json/deriv_b1bench_compute_result.json`: one path of `e1t`
  `[USDJPY·15m]`, timed while other projects used most of the CPU). The staged harness (`cpcv_deriv_x` 0.11,
  `cpcv_deriv` 0.5, option `--path-jobs`) equalled stored results in 24 of 25 comparisons; the one difference, on the
  Brier-objective LightGBM, is below 1e-6 and of the kind that two runs of the live harness show between themselves
  (`harness_regression_011b.log`). Owner decision 7 (2026-11-17) allows it for the path workers of one run; it has
  not been swapped into the lab yet.

*Corrected after the record was committed (2026-11-17).* The round's dispositions were written without two sources
already on this machine: the July 2026 quote archive (section "Venue payouts") and the frxEURUSD contract list
(`docs/deriv_frxEURUSD_contracts_for.json`). The record's text was corrected where it said that contract durations
were not on record or relied on single quotes (the README's addendum lists each place); no stage, count, decision
list or disposition changed. The matrix's tests are scored against the fixed 0.541 breakeven; the venue-price
tests V-03 to V-13 above are outside it.

## Closures respected

The repository closures in AGENTS.md stand for this program unless the owner reopens them: no re-mining of the
neural/spectral, image, GRU/ESN, TSFM or Optuna nulls, and no cross-pair pooling for the USDJPY/USDCAD/AUDUSD/NZDUSD
own-pair family. The owner reopened them on 2026-11-17 for three bounded stages of round 6 on the Deriv feed only
(stages 8, 9 and 10, each behind its gate, fixed recipes, no open-ended architecture search; decisions 3 and 11
under "Owner decisions (resolved)"); on 2026-11-18 the owner released stages 9 and 10 to run now, ahead of their
parent stages ("All neural stages now"); everything else stays closed, and the Dukascopy-store results are not
re-mined. Note for the owner: those closures were measured on the Dukascopy-derived stores, whose timestamps
are not true UTC (verified 2026-11-14: stored = true UTC − 5 h for files labelled 2012-01-02..2025-08-22, + 7 h in US
summer / + 8 h in US winter from 2025-08-25; evidence and reference correction in
`/mnt/data/deriv/checks/clock_corrected_agreement.py`, outside this repository). Their session-gated numbers therefore
describe different true hours than the live Deriv feed. The closures are kept, not re-litigated, unless reopened.

## Owner decisions (resolved)

1. **Validity band vs genuine drift — drift amendment adopted by the owner on 2026-11-15.** The owner selected, verbatim
   (about 18:05 UTC): "Outside the band, certify only if the side also beats its New-York-hour × volatility matched
   always-side baseline: DEV placebo positive on >= 12/15 paths AND holdout drift-matched excess CI95-lower > 0, plus
   mirage checks (tie rate, no forward-fill)." Asked separately (between 18:15 and 18:30 UTC), the owner confirmed
   that the amendment also covers the per-path up-rate tripwire [0.45, 0.55] - the same drift seen path by path
   (USDJPY paths reach 0.5515). Operational definitions, recorded before the holdout is opened:
   - *Mirage check* (an observation, not a fitted threshold): the pair's NY in-session tradeable tie share at the
     configuration's horizon is compared with the other six pairs' range, and labels must be tick-settled with no
     forward fill. USDJPY 15m: 0.0119 (1,542 / 129,284) against 0.0109-0.0182 for the other six pairs; USDJPY 30m:
     0.0081 (1,016 / 125,528) against 0.0073-0.0131 (`dataset_dev_v0.1`); labels from `scripts/deriv_dataset.py`
     settle, no forward fill.
   - *DEV placebo*: `analyze_drift_placebo.py` on exactly that configuration (same pair, horizon and operating coverage;
     the lab registry records the command), baseline matched on New York hour × rv_60m tercile within each path. Its
     pre-registered verdict SKILL BEYOND DRIFT TIMING - excess > 0 on >= 12 counted paths AND mean excess >= 0.02 - is
     retained; the mean floor is a pre-registered program requirement, stricter than the owner's words.
   - *Holdout*: the drift-matched excess with the baseline matched on New York hour × rv_60m tercile over the holdout
     rows (`holdout_eval.py` `drift_matched`, from version 0.5) must have a CI95 lower bound > 0. Version 0.6 adds
     `--weight absret`, so a |return|-weighted configuration such as `c2nysabs` is scored as configured; the test is
     unavailable, and never passes, unless the rv_60m tercile cut points are finite and distinct.
   - *Staging*: "DEV-certified under the amendment" = the goal path test + mirage check + DEV placebo (generated in
     `results/<PAIR>_RESULTS.md` by `scripts/render_ledger.py` `amended_verdict`). Final certification additionally
     needs the holdout drift test and the holdout conditions (owner decision 6's two-look test, PRIMARY n >= 50,
     beating every incumbent book; Holdout discipline above) and then the full per-fold-refit CPCV. A configuration without its own
     placebo run stays INVALID. This is an owner exception for this program; strategy-eval §2 is unchanged elsewhere.

**Answers of 2026-11-17 (14:23 to 14:38 UTC).** After the owner's "if you need anything from me, ask me questions,
give me responses, and your recommendation", seven questions were put with options, and the owner chose the option
marked recommended in each. The labels are verbatim; the rest is what the option said. The numbers are those the
decisions had under "Open owner decisions".

- **Decision 5 (payout): "0.541 rule + payout gate (Recommended)".** The 0.541 path rule stays the certificate. A
  deploy recommendation also needs PAYOUT-CLEARED of the payout-aware reading: expected value above 0 at the quoted
  payout on at least 12 of the 15 paths, and a 10th percentile that is not negative. Every run reports both, which
  also settles the first item of decision 13.
- **Decision 7 (parallel fits): "8 path workers (Recommended)".** Still one run at a time; only the 15 paths of that
  run are fitted by 8 workers of 2 threads each, inside the same 14 GB memory cap (the staged harness peaked at
  7.15 GB with 8 kernel workers, lab test `x-pj-probe2`). This is an owner exception to the AGENTS.md rule that
  heavy fits run one at a time, for the path workers of a single run only.
- **Decision 8 (release policy): "Stage 1, payout tests, 2-4 (Recommended)".** Stage 1 first (15 tests, cap 5,200
  fits), then the six payout tests V-04 to V-09 on all seven pairs, then stages 2-4 in ranked order, each only if
  its gate is met. Cap for the stages: 20,700 fits and 6 GPU hours. A report at each stage boundary. Stages 5-14
  are not released by this answer.
- **Decision 9 (venue access), in part: "Sample quotes for a week (Recommended)".** One public connection (no login,
  cannot buy) records Rise/Fall and Rise/Fall Equal payout quotes at 15 and 30 minutes on all seven pairs for 7
  days, plus contract lists and trading hours; a background service that stops itself; stored on `/mnt/data` under
  the same Deriv API terms as the tick backfill; used only to check and extend the payout model: no candidate is
  scored on that week and nothing is selected or tuned on it. The answer to decision 10 added the 60-, 120- and
  240-minute quotes. Not in the option's words, and so not run: the 15-minute quote at stakes 10 and 50, which
  would show whether the payout per unit staked depends on the stake. The tool does not have it and refuses any
  stake other than 1; it was put to the owner on 2026-11-17 and waits for a yes (section "Venue payouts", "Quote
  survey").
- **Decision 10 (new keys), in part: "Add 60 min, longer if it holds (Recommended)".** 60-minute keys first, after a
  check that enough independent trades exist; 120 and 240 minutes only if that check passes (stage 7: 6 tests, cap
  1,200 fits). The quote week also records 60-, 120- and 240-minute payouts. London and Asian sessions are not
  added.
- **Decisions 3 and 11 (closed families), in part: "Bounded reopen, gated (Recommended)".** In the plan's order, each
  only when the stages it builds on are done and its own gate is met: stronger shrinkage and pooling (stage 8: 4
  tests, cap 2,000 fits), small neural controls on the GPU (stage 9: 26 tests, cap 2,500 fits and 120 GPU hours),
  representations and pretraining (stage 10: 15 tests, cap 1,200 fits and 100 GPU hours). Fixed recipes, no
  open-ended architecture search. Stage 9 builds on stage 6 and stage 6 on stage 5; stages 5 and 6 are inside the
  research scope but were not released by the answer to decision 8, so they are put to the owner at the stage-4
  report.
- **Decision 12 (additional observations), other Deriv instruments only: "Yes, after the quote week
  (Recommended)".** First a probe of which instruments Deriv serves and how far back; then gold, oil, the main stock
  indices and independently quoted crosses with the existing backfill tool, on one connection at its normal pace
  and under the same Deriv API terms as the tick archive. Collection only: tests on them stay behind stage 11's
  gate. Non-Deriv sources stay excluded.

In force as defaults until the owner rules otherwise (not asked; the ranking's recommendation and the stricter choice
in each case): continuing (adaptive) policies count as forward diagnostics only, and the complete search is not
replayed under a null without a raw survivor and adequate power. Under the first default the extended harness
refuses ACI from `cpcv_deriv_x` 0.11 on (Certification harness, above).

**Answers of 2026-11-18.** Put with options after the report of the quote survey's start (about 14:10 UTC) and later
in the day (about 17:00 UTC); the labels are verbatim.

- **Decision 14 (the tradeability filter in winter): "Exclude them from stage 1 on (Recommended)".** Starts from
  21:00:00 to 23:59:59 UTC are excluded from the harness build stage 1 uses onward; look 2 of the two USDJPY
  finalists binds on the starts the venue sells; both readings are reported; stored results stay as they are
  (section "Venue tradeability").
- **The order of the released work: "Core checks, then configs (Recommended)".** `E-004`, `E-002` and `E-003` first;
  then the configurations (`V-04` to `V-09`, then stages 2 to 4) while the rest of stage 1 is built alongside;
  nothing is promoted until `E-009` (the search-wide correction) and `E-010` (the power count) are recorded. This
  replaces the order of decision 8.
- **Sleep during the quote week: "Add a week-long block (Recommended)".** A user unit blocks sleep until the week
  ends; nothing else about the power settings changed.
- **Speed (the owner asked whether the work uses the GPU).** For the LightGBM and CatBoost models in use the GPU was
  slower (lab benchmark `b1bench`); the measured speed-up is the path-parallel harness (`E-020`, 10.8 times the
  single 16-thread fit), so its acceptance was moved into stage 1 and built with `E-003`.
- **Neural families: "All neural stages now".** Stages 9 and 10 (41 tests) are built and run as soon as possible,
  ahead of their parent stages, inside their caps (2,500 + 1,200 training fits, 120 + 100 GPU hours), fixed recipes
  as the plan states them, one GPU job at a time under its own memory cap (a GPU job may run beside a CPU job).
  Every arm is registered before it runs, each neural result is read against the plan's same-input controls, and
  nothing is promoted before `E-009` and `E-010`.

**Answers of 2026-11-19.** Put with options after the owner asked whether the work could be sped up (about 15:15
UTC); the labels are verbatim.

- **Review stopping rule: "Yes, adopt it (Recommended)".** Once the reviewers approve the code itself, a Minor
  finding that is evidence-only (a gap in the tests, not the code) is fixed and verified by the reviewer's own mutant
  or probe failing, without a further review round, and disclosed; a Major or Blocking finding still gets a fresh
  review.
- **Reviewers per round: "3–4 lenses per round (Recommended)".** Three or four independent reviewer lenses per round
  instead of two.
- **Cores: "10 × 2 when idle (Recommended)".** Heavy CPU runs may use `--path-jobs 10 --n-jobs 2` (all 20 cores) when
  nothing else runs on the machine; 8 × 2 (decision 7) otherwise; one heavy job at a time either way.
- **Coverage of the currencies: "Both families, all V tests".** Put with options after the owner asked whether every
  currency is tested (18:30 UTC; the question and the per-key reading given are in lab `PROGRESS.md`, "Owner
  decisions (2026-11-19)"). On DEV only two keys pass the path test: USDCHF 15m UP (`e20sg`, whose strict certificate
  passed and which then failed its survivor check, so it is not a finalist) and USDJPY 15m UP (invalid under the
  moved up-rate band); no DOWN key passes. The first answer, "Targeted transfer (Recommended)", was replaced when the
  owner asked to be asked again: the configurations of `e20sg` (USDCHF) and `e30nystroem` (USDJPY) also run on each
  of the six other pairs with the V baseline and V-04 to V-09, 84 arms for campaign addition 2, which is written
  before any of them runs.

## Open owner decisions

(Decision 1 is resolved above. Its background: strategy-eval §2 treats a moved up-rate outside [0.47, 0.53] as a
fake-flat mirage; on tick-settled Deriv labels the band also fires on genuine drift - USDJPY +6.8% over DEV,
`results/USDJPY_RESULTS.md` drift diagnosis. The USDJPY 15m UP kernel `e30nystroem` (15/15 paths, p10 0.5587) also
passed the pre-registered 16-shift sign-null diagnostic `a5null` for this one selected configuration (labels shifted
3..48 weeks, keeping drift and calendar structure; real p10 rank 17 of 17; none of the shifts met the goal path test;
`results/json/deriv_a5null_USDJPY_15m_result.json`); it does not replay the search, so it is not a search-adjusted
significance test. The sealed holdout remains the binding test.)

Decisions 5, 7, 8 and 14 are resolved above and decision 6 (the two-look sequential holdout) in Holdout discipline.
Decisions 3, 9, 10, 11, 12 and 13 are resolved in part; what stays open of each is listed here. Decisions 7 to 13
were raised by discovery round 6 on 2026-11-17, and the ranking agent's options and recommendations for them are in
`research/deriv_round6_2026-11-17/plan.json` (`owner_decisions`). Decision 14 was raised on the same day by the
commissioning of the quote survey and answered on 2026-11-18.

- **Decision 2 — Frozen books as an operational option.** Before owner decision 1, the only DEV-supported edges were
  frozen Dukascopy-trained USDJPY books scored on their training clock in true NY 13:00–16:35 / 18:00–22:00. They
  are benchmarks under the Deriv-only constraint, not deliverables, unless the owner decides otherwise.
- **Decision 3 — Repository closures** measured on mis-clocked data (neural/spectral/GRU/TSFM/Optuna nulls; pooling
  for the JPY/CAD/AUD/NZD family): reopened for stages 8, 9 and 10 on the Deriv feed (above); otherwise kept unless
  reopened.
- **Decision 4 — Non-Deriv inputs** (e.g. `macro_calendar.parquet`): excluded unless approved.
- **Decision 9 — Venue access, the rest.** The quote survey is to supply what the July archive lacks: the contract
  list of every pair; quotes for EURUSD, for 30 minutes and longer and for Rise/Fall Equal; and quotes outside
  July, which show whether the pricing rule still holds, which clock its daily pattern and the venue's daily close
  follow in winter (the 16:35 New York cut-off of the tradeability filter was measured in July only) and whether a
  contract is quoted between 18:00 and 20:00 New York. Its commissioning gave the first winter evidence
  (decision 14). Not decided: any contact with the trading server; the
  server's quotes after 2026-07-31, which belong to the holdout window (V-13); and one demo contract whose record
  keeps the venue's start and expiry times and its entry and exit ticks (the one retained receipt,
  `evidence/objects/e2512a4a…json`, keeps the proposal, purchase and terminal-record clocks only), which is what
  would check the label's settlement rule, and the tie rule of Rise/Fall Equal, against the venue. The demo
  contract is put to the owner when stage 7 is due.
- **Decision 10 — New keys, the rest.** Not asked: horizons of eight hours and one day, and contract types that pay
  on the size of a move. London and Asian decision sessions as separate keys are not added (the answer above).
- **Decision 11 — Closed families, the rest.** Not asked: frozen external checkpoints (stage 12) and a continuation
  model for admission (stage 13). The record treats a choice among a few listed settings inside each fold (`E-014`,
  and four stage-3 tests in that form) as outside the closed hyperparameter search; the owner can rule otherwise
  here.
- **Decision 12 — Additional observations, the rest.** The tests on other Deriv instruments (stage 11) are not
  released; only their collection is. For the seven pairs Deriv served about 365 days of ticks on 2026-11-14 (Deriv
  history depth above; the probe's longer retention reading is an artifact, `docs/DERIV_DEMO_EXECUTOR.md` probe
  (d)). Whether other instruments have the same limit is not verified; the probe that precedes the collection is to
  show it, and if they do, every day of delay shortens their overlap with the development window by one day.
  Everything from outside Deriv stays excluded: a repaired macro calendar, rates, options, signed executions, order
  books, news and the legacy-vendor history as a prior. The external checkpoints of stage 12 were pretrained
  outside the Deriv archive, so that stage waits for this decision as well as for decision 11.
- **Decision 13 — Evidence claims, the rest.** Reporting at quoted payouts beside the 0.541 certificate is settled
  by decision 5. Two items rest on the defaults above until the owner rules: continuing (adaptive) policies count
  only as forward diagnostics, so the parts of four stage 5-6 tests that update on the evaluated path with matured
  outcomes (`A-039`, `D-033`, `D-035`, `D-036`) stay held; and the complete search is not replayed under a null
  without a raw survivor and adequate power. Not asked yet: one registered forward look, at the whole basket and at
  `c2nysabs`, on data collected after the development window. It is put to the owner with numbers after the power
  count `E-010`; no candidate has been scored on any date after 2026-11-14, so nothing is lost by waiting.
- **Release of stages 5 and 6.** Inside the research scope and not released; stages 9 and 10 run ahead of them
  (answer of 2026-11-18, "All neural stages now"). Put to the owner at the stage-4 report.
