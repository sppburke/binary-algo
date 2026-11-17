> **SCOPE: GENERIC** (discovery round 6 of the Deriv-feed program: what has been run, what is proposed and in what
> order; a plan, not results. Results of record live in `results/<PAIR>_RESULTS.md`; a per-key citation here is tagged
> `[PAIR·tf]` and points there). See `docs/DERIV_SOTA_PROGRAM.md`.

# Deriv-feed discovery round 6: the comprehensive test matrix (2026-11-17)

**Why.** Discovery had ended under the dryness rule on 2026-11-16. The owner reopened it on 2026-11-17: "check all of
the previous configs ran as well as all configs that have yet to run. think deeply about all of the machine learning
variants you can come up with in order to make sure we have a fully a comprehensive set of configs to tests as i want
to find the global optimum ML trading strategy. [...] feel free to look at the relevant papers in
~/git/academic-papers to recommend any combination of ensemble, pre-processing + ML, certain labels, etc, ANYTHING
that you think we should test because i want to FIND THE VERY BEST ALGO TO TRADE. also, can we be doing this on the
GPU to make it faster? optimize the code?" Round 6 is a planning round. **Nothing in it is pre-registered or run.**
Each test is pre-registered in the lab log (`/mnt/data/deriv-sota-lab/PROGRESS.md`) before its first run, and every
test that runs adds to the search-wide multiple-testing burden.

**What had been run.** `tested_configs.tsv` lists every configuration by tag without any result number (columns
`tag`, `horizon_min`, `session`, `operating_cov`, `nseed`, `features`, `n_features`, `dataset`, `harness`,
`pairs_run`, `knobs_not_default`, `extra`); `checks/tested_configs.py` prints its tallies
(`checks/tested_configs.log`). At the inventory the lenses read (02:30 UTC): 110 tags, 542 results. A tag is a
completed evaluation: a CPCV run with a verdict.

| Axis | What the 110 tags cover |
|---|---|
| Model | LightGBM default recipe 69, and once more fitted on the rows of several pairs (`e6pool`); RBF-kernel (Nyström) logistic 24; logistic 5; zero-parameter rule 2; kernel with a tail auxiliary 2; one each of an online adaptive random forest (`e34online`), Brier-objective LightGBM, extra-trees forest, equal-weight pool, simplex stack, ordinal LightGBM, kernel agreement |
| Horizon | 15 minutes 107; 30 minutes 3 |
| Decision session | New York windows only: `ny` 105, `nylate` 2, `nypm` 1, `nyeve` 1, `nyam` 1 |
| Seeds | one seed 100; three seeds 8; none (rules) 2 |
| Inputs | the full feature set 76; a named feature list 8; one added feature family 21 tags (18 families and one union of eight), among them tick-derived blocks (`qo` quote order on the one-second grid, `tk` tick-clock returns, `qp` a permuted control) |
| Gates | compression 17; magnitude 4; adaptive-conformal 4; bootstrap 4; meta-label 3; rule 1 |
| Labels, weights | training label `agree3` 2, `tb2` 1, `resid` 1; weights `absret` 6, `recency` 1 |
| Training population | fit session 6; block validation 4; drift fixed effect 4; reflection 3; specialist 2; signed-only inputs 2; multi-horizon rows 1; monotone constraints 1; blend 1 |
| Hyperparameters | four one-off tags (`e18lowcap`, `e46nysg05`, `e47nysg2`, `e48nysc1000`); no search |

No completed evaluation among those tags has: a neural network of any kind; XGBoost or CatBoost (the three-model mean
`e31ens3` was in the lab queue, not yet run, when this was written); nearest-neighbour, additive or spline, and
discriminant models; a nested hyperparameter search; a horizon beyond 30 minutes; a decision session outside New York
hours (the harness accepts `--session ldn|asia|overlap|all`, `scripts/sessions.py`); separate thresholds for the two
sides (one threshold serves both, `scripts/cpcv_deriv.py`); any contract type other than Rise/Fall; any data source
other than the Deriv tick archive of the seven pairs (the features come from those ticks and from the clock); a
forward test of the whole basket. The compute benchmark below did fit an MLP, a one-dimensional CNN, XGBoost and
CatBoost once on Deriv data, for timing only: one path, no verdict.

**Lenses.** 21 read-only Codex lenses (prompts in `prompts/`, launched through multicodex). Each was instructed not
to read the holdout, and none of the 950 shell commands in their logs names a holdout path except to exclude it from a
file listing (`checks/lens_log_scan.py`, which reads the lab's logs; a command that reached such a file without naming
it would not be caught). Each prompt carries the program state the lenses were told, a snapshot; the records of truth
are the per-pair results files. 13 lenses worked the consolidated literature backlog
(`research/deriv_literature_2026-11-15/`) by kind:
C closed families (17 proposals), E ensembles (24), F1 and F2 features (12, 15), G1 and G2 gates (5, 9), L losses and
labels (22), M1 and M2 models (27, 27), V1 and V2 validation (7, 11), X1 and X2 other data (15, 15). Four independent
completeness critics K1-K4 (18, 15, 14, 22) looked for what the backlog lacks. Three lenses P1-P3 (17, 19, 22) re-read
the paper corpus in full text for targeted questions, and S1 (20) reviewed the harness for speed. Total 353 proposals
(`proposals.jsonl.gz`: each as its lens returned it, with `lens`, `id`, `title`, `kind`, `sign_or_size`, `source`,
`spec`, `mechanism`, `implementation`, `nearest_existing`, `prior`, `prior_why`, `cost`, `fast_kill_falsifier`,
`leakage_risks`, `needs`, `batch`). Lens priors: low 287, medium 36, high 30.

**Clustering.** Five agents (`prompts/CLUSTER_template.md`), one per axis, merged duplicates and judged each cluster:
A model classes 82 proposals -> 77 clusters; B features and data 79 -> 67; C labels, losses and training populations
80 -> 75; D ensembles, gates and decision rules 62 -> 56; E validation and infrastructure 50 -> 41. Total 316
clusters (`clusters.jsonl.gz`): 290 to test, 26 rejected with a recorded reason. A script checked that every group's
clusters partition its proposal ids.

**Gap check.** One agent (`prompts/GAPS.md`) read the clusters against the tested configurations and named seven
missing tests; the primary agent added three (contract types that pay on move size, London and Asian decision
sessions, one forward look at the whole basket). All ten are accepted; two are narrowed and lowered from a medium to a
low prior (`plan.json` `gaps`).

**Ranking.** One agent (`prompts/RANK.md`) ordered the 316 clusters into stages by information value per unit of
build and compute: 282 staged, 7 absorbed into 5 others, 1 not run (`A-076`, LightGBM on OpenCL, already measured
slower), and the 26 rejections kept on record. With the ten gaps: **292 tests in 14 stages**. Stages 1-6 (200 tests)
stay inside the existing research scope: no venue call, new key, closed family or new data source (the ranking
agent's `can_start_now`). That holds for four stage-3 tests only in a narrowed form. The specifications of `A-027`,
`A-028`, `A-030` and `A-031` choose their settings through the configuration race of proposal `R6-K4-05`, which that
proposal says reopens the closed hyperparameter search and which is `C-069` in stage 10. Stage 3 runs them without the
race, in the capped form of the ranking agent's notes: a short list of settings fixed before the first fit and chosen
inside each fold by the stage-1 selector `E-014` (matrix field `primary_agent_note`, which replaces what the
specifications and cost fields say about the race). This record does not name those settings. They are entered in the
lab log before the first fit of stage 3, at most eight per test from the levels its specification lists (for `A-030`
its six additive cells), and the four tests are not runnable until then. Choosing among a few listed
settings on inner rows of a fold is treated here as outside that closure: the repository names it as Optuna tuning
(`scripts/README.md`, "`*_optuna.py` ... hyperparameter tuning"; `SWEEP_MATRIX.md` row I6), and the harness already
chooses the number of trees and the coverage threshold on inner rows (`scripts/cpcv_deriv.py`). That reading is the
primary agent's; the owner can draw the line elsewhere under decision 11.

Nine more tests of stages 1-6 lean on something held for a later decision, and each carries a note saying how it
runs. `E-007` (stage 1) and `E-011` (stage 3) name `E-001`, which the ranking agent merged into the venue test `B-002`
of stage 7 while placing its local maturity checks in `E-007`: both use that development-data part only and make no
venue call. `E-012` (stage 3) starts "after verifying the settlement contract": in stage 3 that is the same
development-data audit, and its replay uses the dataset's own settlement function. `E-009` (stage 1) leaves the null
refits to `E-030` of stage 14 and labels its own inference as conditional. `A-039` (stage 5) and `D-033`, `D-035` and
`D-036` (stage 6) have parts that update on the evaluated path with outcomes that have matured there. Stage 13 holds
policies of that kind for decision 13, so those parts wait for decision 13 as well as 8; `D-050` (stage 6) starts from
the residuals of `D-035` and uses the fit-side ones only. The harness already has one element of this kind, the
adaptive-conformal gate, which moves its threshold when a trade of the evaluated path settles
(`scripts/cpcv_deriv_x.py`, `aci_side`); holding the new parts back is the cautious reading and the primary agent's
choice.

The builder stops if a specification names a test of a stage that its own stage does not build on, or a test that
waits for a decision which the naming test neither lists nor needs through earlier stages, and the naming test has no
note. Those checks are structural: the builder does not read what a note says, and it cannot see a dependency that
names no test. `E-012` and the path-updating parts are of that kind and were found by reading the specifications (the
first by the third review of this record), which is not a mechanical check. The lists themselves are the primary
agent's input. The builder tests each only against the test's `needs` field (a reopened closed family needs decision
11, an owner data decision one of 9, 12 and 13), so a wrong list that satisfies that passes; the last two reviews
each passed the builder with a changed list.

None of this is a release. Every test waits for open owner decisions of `docs/DERIV_SOTA_PROGRAM.md`. The decisions
its own specification needs are the matrix field `program_decisions`; the table gives each stage's list, and Owner
decisions below maps them. Stages 1-6 wait for the release policy (8), three of their tests also for parallel fits (7)
and four, in the parts that update on the evaluated path, for the evidence-claims decision (13). Stages 7-14 (92
tests) wait for the decisions on venue access, new keys, closed families, additional observations and evidence
claims. A test also needs what its stage builds on; the decisions that adds are the matrix field
`upstream_decisions` and the table's fifth column.

| Stage | Name | Tests | Waits for program decisions | Through earlier stages | Cap (model fits; GPU hours) | Build, engineer-days (ranking agent's estimate) |
|---:|---|---:|---|---|---|---|
| 1 | Freeze validity, selection, caching and execution | 15 | 8; `E-019`, `E-020`: 7, 8 | none | 5,200; 0 | 5-10 |
| 2 | Cheap tie, participation and path-information tests | 22 | 8 | none | 2,500; 0 | 3-6 |
| 3 | Shared compact models, geometry and fast producers | 48 | 8; `E-018`: 7, 8 | none | 8,000; 6 | 8-15 |
| 4 | One causal price-path block and historical libraries | 29 | 8 | none | 5,000; 0 | 8-15 |
| 5 | Reuse honest teachers for ensembles and loss tests | 53 | 8; `A-039`: 8, 13 | none | 10,000; 0 | 15-25 |
| 6 | Magnitude features and distributions behind fixed sign | 33 | 8; `D-033`, `D-035`, `D-036`: 8, 13 | none | 7,000; 0 | 10-20 |
| 7 | Contract probe and longer expiries | 6 | 9, 10; `B-002`: 9, 13 | 8 | 1,200; 0 | 2-4 |
| 8 | Stronger shrinkage and partial pooling | 4 | 11 | 8 | 2,000; 0 | 5-10 |
| 9 | Bounded supervised neural coverage | 26 | 11 | 8 | 2,500; 120 | 10-20 |
| 10 | Representations and pretraining | 15 | 11 | 8 | 1,200; 100 | 10-20 |
| 11 | Genuinely additional observations | 21 | 12 | 8 | 4,000; 0 | 20-40 |
| 12 | Frozen external checkpoints | 5 | 11, 12 | 8 | 400; 60 | 3-6 |
| 13 | Continuing and adaptive policies | 11 | 13; `C-065`, `R6-GAP-07`: 11, 13 | 8, 11 | 2,000; 20 | 6-12 |
| 14 | Full-search confirmation and null refits | 4 | 13 | 8 | 1,800; 0 | 5-10 |

In the fourth column the numbers before a semicolon are the list of every test in the stage; a test named after it
waits for its own list instead (the column is generated, `plan_tables.md`; the reason for each named test is its
`primary_agent_note`). Stage 11 is released source by source and stage 12 checkpoint by checkpoint; stage 12 waits
for decision 12 as well as 11 because its checkpoints were pretrained outside the Deriv archive. Stage 14 runs only
after a raw survivor.

The fifth column is what a stage needs beyond its own list because it builds on earlier stages: the lists of those
stages (`plan.json` `depends_on_stages`, followed through; every later stage builds on stage 1; the column is
generated). The ranking agent states that dependency per stage, not per test. Stage 13 builds on stage 9, the neural
stage, so decision 11 stands there for the whole stage; `C-065` is its only test whose specification names a stage-9
test, and whether the others need anything from stage 9 is not recorded. An earlier stage counts at its stage list.
The tests it holds back for a further decision (parallel fits in stages 1 and 3, the path-updating parts in stages 5
and 6) are not counted as prerequisites of later stages; where another specification names such a test, the builder
asks for a note (`D-050` is the one case).

The caps are the ranking agent's conditional ceilings, not forecasts; they include control and inner fits: 37,700
fits and 6 GPU hours for stages 1-6, 52,800 fits and 306 GPU hours for all fourteen. They exclude the ten gap tests,
which were added after the ranking; each gap's cost is in its matrix row (`R6-GAP-09`, for example, is 28 runs of 19
fits). A replay of the ranked search under two nulls (stage 14) would add up to 21,067,200 fits and 122,094 GPU hours
(399 times the ordinary caps, again without the gap tests) and runs only if a raw survivor exists; its ceiling is an
owner decision.

What the 292 tests are (`matrix.jsonl.gz`, field `axis`): model classes 67; information for the sign (features and
data) 46; training population 34; ensembles 26; validation 25; losses 23; information for move size 23; decision
layer 15; infrastructure 13; labels 12; contract and horizon, data decisions and others 8. By target: sign 205, size
49, neither 38. By what they need: new code 213; a closed family reopened 48; an owner data decision 26; a GPU back
end 3; nothing 2. By prior: low 250, medium 23, high 19. **All 19 high priors are validation, infrastructure or
venue-reconciliation items** (14 of them in stage 1); no predictive candidate carries a high prior.

**First 25, in run order** (`plan.json` `first_25`, each with what runs, its kill rule and cost): freeze the search
and evidence contract (`E-004`); strict certification reporting (`E-002`); reject empty support and degenerate gates
(`E-003`); exact execution and split references (`E-005`); causal prefixes and outcome maturity (`E-007`); the nested
selector (`E-014`); reuse of raw arrays (`B-001`); the reusable prediction cache (`E-006`); synchronized daily
evidence (`E-008`); support and detectable effect sizes (`E-010`); the joint search-wide inference guard (`E-009`);
global resource accounting (`E-019`); acceptance of the parallel path harness (`E-020`); participation and
concentration audit (`E-024`); deterministic early stopping (`E-021`); separate side coverages (`D-001`); a learned
tie-risk discount (`C-004`); tie-inclusive targets (`C-005`); recovery of tuning rows for the final fit (`C-002`); a
raw minute-path linear control (`A-068`); pending peer shocks (`B-006`); within-minute order (`B-008`); a stale-input
veto (`D-003`); causal logit smoothing (`D-004`); consecutive same-side confidence (`D-005`).

**What the stored results say about the ceiling.** `checks/tail_distribution.py` (log beside it) reads every result
JSON in `results/json/`: 553 files with paths at 05:41 UTC (through `r2t` AUDUSD). It describes the outcomes of the
search at 15 minutes; it is not a performance estimate for any key, and 30-minute results are never mixed in. Among
VALID runs outside USDJPY at their operating coverage, 707 sides have at least 12 counted paths (all 15 counted on
484 of them). Their mean path win rate has median 0.5070, 90th percentile 0.5434 and maximum 0.6071, at a median of
59 trades per counted path (10th percentile 39, 90th 171); within a side, the counted path with the most trades holds
a median of 2.7 times as many as the one with the fewest. Eight sides have at least 12 paths clear of 0.541 and one
meets the goal path test, all of one key (`[USDCHF·15m]` UP; `results/USDCHF_RESULTS.md` records its best run as a
goal-path pass that failed its survivor check). Over every coverage it is 28 and 4 of 1,940 sides.

`checks/pass_probability.py` (log beside it; a calculation that reads no data) gives the true win rate the goal rule
needs under stated assumptions. The rule (`scripts/cpcv_deriv.py`, result field `verdict_goal`) counts a path when it
holds at least 25 trades, and asks for at least 12 counted paths clear of 0.541 and for the 10th percentile of the
counted paths' win rates to reach 0.541. How hard the percentile is depends on how many paths are counted: with 12 or
13 counted, 12 clear paths pass; with all 15 counted, 12 clear paths cannot pass, and 13 pass only when the
third-lowest path exceeds 0.541 by at least 1.5 times the second-lowest's shortfall (log section 1). The table assumes
that all 15 paths are counted and hold the same number of trades, and that every trade wins with one probability p.
The 15 paths are the pairs of six time groups. How much the trades of two paths that share a group overlap has not
been measured, so two scenarios are computed: independent paths, and paths that share all of their groups' trades.
They are scenarios, not proven bounds.

| Trades per path | p for a 0.5 chance of passing (independent / shared) | p for a 0.8 chance |
|---:|---|---|
| 60 | 0.614 / 0.604 | 0.634 / 0.640 |
| 100 | 0.596 / 0.589 | 0.612 / 0.616 |
| 200 | 0.580 / 0.574 | 0.591 / 0.594 |
| 500 | 0.566 / 0.562 | 0.573 / 0.575 |

Under those assumptions, at 60 trades per path the rule passes with probability 0.0005 to 0.066 at the breakeven
rate itself (p 0.541), 0.008 to 0.146 at p 0.56 and 0.07 to 0.28 at p 0.58 (log section 3; the range is the two
scenarios). With a win probability that differs between the six time groups at the same average, the pass probability
was lower by more than three standard errors (a conservative bound on the error of the difference) in 135 of the 144
cases computed, higher by that criterion in none, and within it in 9 (log section 4).

What this shows and what it does not. A side with p 0.507 (the median above) and 60 trades meets the goal path test
with probability 0.00000 to 0.0102 in the two scenarios and has 12 paths clear with probability 0.00008 to 0.031; the
observed 1 and 8 of 707 (0.0014 and 0.0113) lie inside those ranges. The frequency of raw passes seen so far is
therefore compatible with chance under these scenarios. It is not a calibrated null: the 707 sides are different
configurations run on the same six pairs' rows, so they are not independent; 223 of them have fewer than 15 counted
paths; trade counts differ between paths; and no true rate has been identified for any side. It does not show that no
side has an edge, nor that more trades per path would produce a pass. What it supports is narrower: at about 60
trades per path, a side whose true rate is 0.56 to 0.58 would still fail the rule in 72% to 99% of runs in these
scenarios, so a kill at that support says little about whether such an edge exists, and the support a key offers
should be counted before more model variety is spent on it. `E-010` (support and detectable effect sizes) and
`E-009` (the search-wide inference guard) in stage 1 are the planned checks.

**Speed: the GPU and the code.** Measured on one fit of one path of the default recipe (`e1t` path 1, `[USDJPY·15m]`,
72,331 fit rows; `results/json/deriv_b1bench_compute_result.json`, lab test `b1bench`; a compute benchmark, not a
result of record). Other projects used 12 to 24 threads' worth of CPU on the 20-thread machine while it ran (median
16; the file records the load before each measurement), so the CPU timings are measurements under contention:
- LightGBM was slowed, not sped up, by more threads: 800 trees took 86.59 s with 16 threads and 19.97 s with 6, with
  identical predictions; 200 trees were fastest at 4 threads (5.77 s) and took 19.6 to 21.0 s at 16 to 20. Eight
  2-thread fits side by side finished 27.31 fits per minute against 2.53 for one 16-thread fit.
- LightGBM, the tree model of the completed evaluations, was slower on the GPU: 200 trees took 12.16 s on OpenCL
  (with 6 CPU threads) against 8.45 s on 6 CPU threads alone and 5.77 s on 4. CatBoost (so far only in the queued
  `e31ens3`) took 11.79 to 12.19 s on the CPU and 14.11 s on the GPU for 400 trees. XGBoost (also only in `e31ens3`)
  took 168.75 s (6 threads) or 660 s (16) on the CPU and 6.78 s on CUDA for 400 trees, but the CUDA model's
  predictions differ from the CPU model's, so it is a new recipe to be tested as such (`A-031`, `A-075`), not a
  faster copy of an existing one.
- The two neural benchmarks ran much faster per row on CUDA: a small MLP at 723,789 rows per second against 14,898
  on 6 CPU threads, a one-dimensional CNN at 40,430 against 261 (its CPU run used 9,041 of the 72,331 rows, so that
  ratio compares rates, not equal workloads). Both CPU fits completed: the GPU is the practical choice for the
  neural tests, not a requirement. A GPU kernel transform was not benchmarked (`A-077` proposes one).
- The exact speed-up is to run the 15 paths of a run side by side. The staged harness (`cpcv_deriv_x` 0.11 and
  `cpcv_deriv` 0.5 with `fork_jobs.py`, option `--path-jobs`) passed 31 of 32 regression checks
  (`checks/harness_regression_011b.log`): 24 of 25 comparisons with stored results were equal in every measured
  value, among them eleven full runs with 8 path workers of 2 threads each (the descriptive `model` string, which
  lists knobs added since the stored run, is reported and not compared); 2 checks compared parallel with sequential
  output; 5 tested argument and worker-failure handling. In that session the default LightGBM recipe took 235 s
  sequentially and 92 s with path workers, and the kernel recipe 160 s and 91 s (`e1t` and `e30nystroem`,
  `[USDJPY·15m]`; timings, not results). The failing comparison is the Brier-objective LightGBM (`e59brier`, first
  three paths): one path AUC differs by 2.6e-07 and thresholds by up to 1.4e-15, with equal trade counts and win
  rates. Two runs of the live harness with identical arguments also differ from each other, in that AUC by 2.6e-07
  and in seven thresholds by at most 1.1e-16 (lab test `x-brier-threads`; `checks/brier_threads_run.log`, last
  comparison, and `checks/brier_threads_leaves.log`). That check therefore compares counts exactly and floats within
  1e-6 (lab test `x-regress-011c`; `checks/harness_regression_011c.log`), and the exactness rules of `E-005`, `E-018`
  and `E-020` carry that exception (matrix field `primary_agent_note`).
  The staged files are not yet in use: AGENTS.md says heavy fits run one at a time, and relaxing that is the owner's
  decision (program decision 7).

**Claims the agents asked to have verified** (`plan.json` `claim_dispositions`, with the evidence for each): the
realized-volatility columns are standard deviations of the log returns between adjacent one-minute bars over a count
of bars, so the square-root-of-horizon scaling that `A-056` proposes is dimensionally right where bars are contiguous
(`scripts/deriv_dataset.py`, `own_features`; a gap counts as one step, which `R6-GAP-01` tests); CatBoost Ordered
boosting takes the row order as its one permutation when `has_time=True`, so `A-032` must sort rows chronologically
(`checks/library_behaviour_fixture.py`, a synthetic fixture); the three members of `ens3` are trained differently
(only LightGBM receives sample weights), so GPU producers are new recipes (`A-075`); the durations Deriv offers per
contract type are not on record and need one metadata call per pair (`C-009`); the Deriv tick archive holds the seven
pairs only and no approved same-asset reference archive with receipt timestamps is on disk (`B-038`; other disks were
not searched); the legacy-vendor clock correction is checked against Deriv ticks only for 2025-08-11 to 2026-06-06
(`B-042`; `/mnt/data/deriv/checks/clock_corrected_agreement_result.json`, outside the repository).

**Owner decisions.** The ranking names 17 (`plan.json` `owner_decisions`, each with options and a recommendation).
`docs/DERIV_SOTA_PROGRAM.md` carries them, with what the primary agent added, as open owner decisions 7-13
(`plan.json` `decision_map`; the Gates column is generated from the per-test lists):

| Program decision | Ranking decisions it carries (position in `owner_decisions`) | Added by the primary agent | Gates |
|---|---|---|---|
| 7 Parallel fits | none | the staged path-parallel harness, measured after the ranking; independent jobs side by side (`E-019`) are a further step | `E-019`, `E-020` (stage 1); `E-018` (stage 3) |
| 8 Release policy | 1 | none | stages 1-6 |
| 9 Venue access | 2 | the contract list by pair and contract type, which `R6-GAP-08` and `R6-GAP-09` need and on which `C-009`, `C-010` and `R6-GAP-02` are conditional | stage 7 |
| 10 New keys | 4 | contract types that pay on move size (`R6-GAP-08`); London and Asian decision sessions (`R6-GAP-09`) | `C-009`, `C-010`, `R6-GAP-02`, `R6-GAP-08`, `R6-GAP-09` (stage 7) |
| 11 Closed families | 5, 15 | the continuation model `R6-GAP-07`; the configuration race `C-069` is the hyperparameter search it covers | stages 8-10, 12; `C-065`, `R6-GAP-07` (stage 13) |
| 12 Additional observations | 6 to 14 | the external checkpoints of stage 12, pretrained outside the Deriv archive | stages 11, 12 |
| 13 Evidence claims | 3, 16, 17 | one registered forward look at the whole basket (`R6-GAP-10`); the parts of four stage 5-6 tests that update on the evaluated path | stages 13, 14; `A-039` (stage 5); `D-033`, `D-035`, `D-036` (stage 6); `B-002` (stage 7) |

The ranking's eight disagreements between lenses and how each was settled are in `plan.json` `disagreements`.

**Per-key numbers in this folder.** This README owns none. The prompts (the program state the lenses were told on
2026-11-17, untagged), the agents' records (`proposals.jsonl.gz`, `clusters.jsonl.gz`, `build/`, `plan.json`,
`plan_tables.md`, `matrix.jsonl.gz`) and the check logs quote per-key numbers as context. Those are archival, are not
results of record and are not updated. The record for a key is its results file: `[AUDUSD·tf]`
`results/AUDUSD_RESULTS.md`, `[EURUSD·tf]` `results/EURUSD_RESULTS.md`, `[GBPUSD·tf]` `results/GBPUSD_RESULTS.md`,
`[NZDUSD·tf]` `results/NZDUSD_RESULTS.md`, `[USDCAD·tf]` `results/USDCAD_RESULTS.md`, `[USDCHF·tf]`
`results/USDCHF_RESULTS.md`, `[USDJPY·tf]` `results/USDJPY_RESULTS.md`. Timings and library behaviour measured on one
key's data (`b1bench`, the regression logs, the Brier reproducibility logs) are infrastructure measurements, not
strategy results.

**The ranking agent's judgment, not a measurement.** Probability of at least one credible pass outside USDJPY at 15 or
30 minutes on this fixed history: about 8% (range 3-15%; 15 minutes alone 3-5%). Most model, loss, ensemble and
magnitude branches are expected to be null. The one change it names to raise the odds is new information: an archive
of signed executions, timestamped as received and aligned to Deriv.

## Files

| File | Content |
|---|---|
| `plan.json` | stages (ids, purpose, budget, expectation, dependencies; `can_start_now` is the ranking agent's "inside the existing scope", `program_decisions` and `waits_for` say what the specifications of the stage's tests need, `upstream_decisions` what the stage needs through the stages it builds on), `first_25`, `merged`, `gaps`, `not_run`, `owner_decisions`, `decision_map`, `test_decisions`, `claims_to_verify`, `claim_dispositions`, `disagreements`, `counts`, `summary`, `primary_agent_note` |
| `plan_tables.md` | the same as tables: stages, first 25, ids by stage, gap dispositions, claims, owner decisions and their mapping to program decisions 7-13, notes on single tests, merges, not run, disagreements |
| `matrix.jsonl.gz` | 326 rows, one per cluster (316) and gap (10): `id`, `group`, `axis`, `title`, `side`, `needs`, `prior`, `verdict`, `disposition` (`stage` 292, `absorbed` 7, `not_run` 27), `stage`, `stage_name`, `program_decisions`, `upstream_decisions`, `first_25_rank`, `rank_note`, `primary_agent_note`, `spec`, `fast_kill`, `cost`, `members`, `n_lenses`, `absorbed_into`, `not_run_why`, and for gaps `mechanism`, `nearest_existing`, `primary_agent_why`, `primary_agent_evidence`, `prior_as_proposed` |
| `clusters.jsonl.gz` | the 316 clusters as the clustering agents wrote them |
| `proposals.jsonl.gz` | the 353 lens proposals in full, each with its lens |
| `tested_configs.tsv` | the 110 configurations already run, without result numbers |
| `prompts/` | the 21 lens prompts, the clustering template, the gap and ranking prompts |
| `build/` | `build_round6_record.py` and its inputs beside `clusters.jsonl.gz`: `RANK_result.txt` (the ranking agent's output), `gaps.jsonl` (the gap-finder's proposals), `primary.json` (the primary agent's dispositions, including the one list of program decisions per test, `test_decisions`). Run as its docstring says, it rewrites `plan.json`, `plan_tables.md` and `matrix.jsonl.gz` byte for byte. It stops if a test's list contradicts its `needs` field, or if a specification names a test of a stage that its own stage does not build on, or a test that waits for a decision the naming test neither lists nor needs through earlier stages, and the naming test has no note |
| `checks/` | `tested_configs.py`, `tail_distribution.py`, `pass_probability.py`, `library_behaviour_fixture.py`, `lens_log_scan.py`, each with its log and its command from the repository root in its docstring; outputs of three lab tests of the staged harness, whose scripts stay in the lab: `harness_regression_011b.log` (`x-regress-011b`), `brier_threads_run.log` and `brier_threads_leaves.log` (`x-brier-threads`: the runs, and the differing values by field), `harness_regression_011c.log` (`x-regress-011c`) |

What stays in the lab (`/mnt/data/deriv-sota-lab/discovery/round6/`) and is needed to redo the earlier steps: the
lens outputs as returned (their summaries and the text around the proposal records), the lenses' command logs, the
clustering agents' raw outputs, the per-key inventory the lenses read, from which `tested_configs.tsv` was
written, and the scripts of the three lab tests whose outputs are in `checks/` (`lab_test_scripts/` there, and
`/mnt/data/deriv-sota-lab/tools/regress_v011.py`, the regression script as amended with the tolerance).
`tail_distribution.log` depends on the result files present when it ran; its first line counts them.
