You are the final ranking step of a research planning round. Work read-only: read files, think, and write your answer
as your final message. Do not modify any file and do not open any network connection. Do NOT read holdout data or
holdout results: never open anything under /mnt/data/deriv-sota-lab/holdout/, /mnt/data/deriv-sota-lab/cache/dataset_full_*,
/mnt/data/deriv-sota-lab/registry/, the files /mnt/data/deriv-sota-lab/PROGRESS.md,
/home/sean/git/binary-algo/results/USDJPY_RESULTS.md, /home/sean/git/binary-algo/docs/DERIV_SOTA_PROGRAM.md and
/home/sean/git/binary-algo/results/json/deriv_lab_registry.jsonl, any file under /home/sean/git/binary-algo/results/json/
whose name contains "look1", "look2" or "holdout", or any Deriv tick, bar or sidecar file dated 2026-08-01 or later.
Do not run git. Do not start heavy computation (no model fits); reading and light parsing only (python3 is available).

## The program (context)

Goal: the best demonstrable out-of-sample predictor of the 15-minute (also 30-minute and longer) direction of seven FX
pairs as quoted by the broker Deriv (AUDUSD, EURUSD, GBPUSD, NZDUSD, USDCAD, USDCHF, USDJPY) for its Rise/Fall binary
contract; UP and DOWN are separate keys; ties lose; breakeven win rate 0.541. Certification is per-fold-refit
combinatorial purged cross-validation (6 chronological groups, 15 two-group test paths), New York 08:00-17:00; a side
passes iff at least 12 of 15 paths have >= 25 trades and a win rate >= 0.541, and the 10th percentile of path win
rates is >= 0.541. Data: Deriv 1-second mid-price ticks for the seven pairs, development window 2025-08-11 ..
2026-07-31 (about 8,600 non-overlapping 15-minute outcomes per pair in total). About 110 experiment tags (547 result
files) have been run. No pair except USDJPY (which drifted, and is judged against a placebo) has a passing side. Per
pair, the median path-mean AUC over all 15-minute runs is 0.498 to 0.511 outside USDJPY (USDJPY 0.526) and the 90th
percentile is at most 0.514; the few higher values (0.52 to 0.53) come from narrow-window rule runs and sub-session runs
that still failed the path test. A higher AUC has not meant a better confident tail. Only one 30-minute configuration
has been run on six of the seven pairs; no horizon beyond 30 minutes has been run.

## Facts verified in the code and data after the lenses wrote (treat as settled; do not list them as claims to verify)

- Labels: entry target = decision + 1 s; entry price = first tick at or after it; exit price = last tick at or before
  entry target + H; a zero return is a valid label and loses. The venue's published wording (the contract runs for its
  duration from the start time; entry spot = next tick after the start; exit spot = latest tick at or before the end)
  agrees with this label. Several lenses read a discrepancy ("expiry should be actual entry tick + H"); that reading
  came from shorthand in their prompt. What remains open is a comparison with one real contract's timestamps, which
  needs a venue connection and is an owner decision. Do not plan a label migration; plan the one-contract check.
- Fit rows and validation rows exclude ties (moved rows only); test rows include ties, which lose. Ties are about 1.6%
  of valid 15-minute labels and 1.1% of 30-minute labels (EURUSD: 5,586 of 351,114 and 3,745 of 350,364).
- One threshold serves both sides: the coverage quantile of |p - 0.5| over the validation rows.
- Validation = a random 15% of the in-session training rows (or the chronologically last 15% in block mode). Those rows
  set early stopping and the threshold and never enter the final fit.
- The prediction dump keeps only the post-gate test predictions and the thresholds of each path: no validation
  predictions, no per-member predictions.
- The bootstrap gate resamples rows, not calendar blocks. The stack's weights are fitted on the same validation rows
  that set early stopping and the threshold.
- `--max-paths N` takes the first N of the 15 combinations (all containing group 0 for N <= 5).
- Features are built on 1-minute bars of the 1-second ticks (66 base columns; later blocks up to 340). Peer staleness is
  measured to the peer's last bar close. The tick archive has two columns, time and mid price: no bid, ask, size or
  aggressor side.
- The macro calendar on disk has 8,838 rows from 2012-01-03 to 2026-05-29 (505 inside the development window, none after
  2026-05-29), with actual, consensus and previous values and no receipt or revision timestamps.
- Payout is not constant. Recorded venue quotes for a 15-minute USDJPY contract range from 1.52 to 1.90 by time of day
  (breakeven 0.526 to 0.66); one real trade paid 1.77 (breakeven 0.565). The payout by pair, hour, duration and contract
  type (including the ties-win variant) is not on disk.
- Reporting: sweep ledgers now print the strict rule's verdict; the batch runner's one-line note still prints the older
  field.

## Measured compute facts

This machine: 6 performance cores (12 threads) plus 8 efficiency cores, 62 GB RAM, an idle 8 GB NVIDIA GPU; other
projects use part of the CPU. One default LightGBM fit (800 trees, about 72,000 rows, 66 to 340 features) takes about 5
to 40 s on 2 to 8 threads; a certification run is 19 fits per seed (15 paths plus 4 forward folds). Eight 2-thread fits
side by side gave about 10 times the throughput of one 16-thread fit under load. A path-level parallel harness is
staged and reproduces stored results exactly: a 19-fit LightGBM run took 86 to 95 s with 8 workers against 186 to 217 s
before (the earlier timings were taken under a different machine load); the kernel model (500 Nystroem components) completes 19 fits in 29 to 95 s. LightGBM on the GPU was slower than
one CPU thread. XGBoost on the GPU fitted in 7 s what took 170 s on 6 CPU threads (a slightly different model from the
CPU one). A small MLP ran about 50 times faster on the GPU and a 1-D CNN about 150 times. So: a cached-decision test
costs seconds; a tree or linear test costs 1 to 5 minutes per pair; a GPU neural screen costs minutes to a few GPU hours.

The owner asked for a fully comprehensive set of configurations to test and wants the best strategy that can be
demonstrated. Twenty-one lenses proposed 353 candidates; five clustering agents merged them by group into 316 clusters
(290 TEST, 26 REJECTED); a completeness critic listed what is still missing.

## Inputs (all under the current directory, /mnt/data/deriv-sota-lab/discovery/round6)

- merge/clusters_all.jsonl - every cluster from the five groups (A model classes; B information: features and data;
  C labels, losses, training populations; D ensembles, gates, decision rules; E validation and infrastructure): cid,
  axis, title, members, n_lenses, side, needs, verdict (TEST | REJECTED), duplicate_of, new_vs_existing,
  same_test_elsewhere, representative, spec, fast_kill, cost, needs_cache, prior, prior_why, tier.
- merge/group_notes.jsonl - each group's owner_decisions, claims_to_verify, disagreements, counts and summary.
- merge/gaps.jsonl - the completeness critic's coverage table and gap proposals (ids R6-GAP-nn), and its summary. The
  critic may still be running: if this file is absent or empty, write {"gaps": []} and plan the clusters only; its
  proposals will then be placed by the primary agent.
- merge/items_brief.tsv - one line per original proposal (id, group, kind, side, needs, prior, title, spec start);
  full records in results/<LENS>.txt (R6-K4-02 is in results/K4.txt).
- experiments_inventory.md - every experiment already run, with per-pair results.

## What to produce

1. CROSS-GROUP MERGE. Using `same_test_elsewhere` and your own reading, list clusters from different groups that are
   the same test; name the cid that survives.
2. GAPS. For each gap proposal decide: accept as a test (give it a stage), or decline (already covered by a cluster or
   an inventory experiment - name it - or not worth one bounded test).
3. ONE ORDERED PLAN over every surviving cluster with verdict TEST and every accepted gap. Order by expected value per
   unit of compute and build effort toward a side that passes the certification rule. Respect dependencies: shared
   infrastructure (a prediction cache with validation and member predictions, parallel paths, GPU back ends) and
   validity guards (a frozen search manifest, selection of the whole procedure inside each fold, a search-wide null)
   come before the trials that need them, because every added trial raises the chance of a false pass. Group clusters
   that share one build (one feature block, one label builder, one cached prediction set, one neural trainer) into the
   same stage so that the build is paid once. Say plainly which stages you expect to be null. Keep stages that need an
   owner decision (a closed family reopened, external data, a venue probe, a change of certification scope) separate
   from stages that can start now, and say which is which.
4. A compute and build budget per stage (model fits, GPU hours, build effort S/M/L), using the measured facts above.
5. THE FIRST 25 TESTS TO RUN, in order, each written so that it can be started without reading anything else.
6. Every decision only the owner can take, consolidated, each with options and your recommendation.
7. Your honest odds.

## Output format (final message; JSON lines only, no prose outside JSON, no code fences)

{"merged": [{"keep": "A-012", "absorb": ["C-003"], "why": "<=160 chars"}]}
{"gaps": [{"id": "R6-GAP-01", "decision": "accept|decline", "stage_or_covered_by": "...", "why": "<=160 chars"}]}
then one line per stage, in run order (at most 14 stages):
{"stage": 1, "name": "<=60 chars", "can_start_now": true, "owner_decision": "<none, or which decision gates it>",
 "purpose": "<=220 chars", "ids": ["<cid or R6-GAP-nn>", "... in run order ..."], "notes": {"<cid>": "<=120 chars,
 only where the test should be narrowed, capped or changed from its cluster spec>"}, "depends_on_stages": [..],
 "budget": "<total fits, GPU hours and build effort for the stage>", "expected": "<=220 chars: what you expect to
 learn; write 'expected null' when that is your view>"}
then, one line each:
{"first_25": [{"rank": 1, "id": "<cid or R6-GAP-nn>", "title": "<=80 chars", "what_runs": "<=320 chars: the bounded
 test, its grid, the pairs and horizons, the control it is compared with>", "kill_rule": "<=160 chars", "cost":
 "<fits or GPU hours; build S|M|L>", "why_now": "<=140 chars"}]}
{"not_run": [{"id": "<cid>", "why": "REJECTED|dominated by <cid>|<other reason, <=100 chars>"}]}
{"owner_decisions": [{"decision": "...", "options": ["..."], "recommendation": "...", "unlocks": ["<stage or cid>"],
 "why": "<=220 chars"}]}
{"claims_to_verify": [{"claim": "...", "by": "<id>", "how": "<file or check>"}]}
{"disagreements": [{"topic": "...", "positions": "<=300 chars", "your_view": "<=160 chars"}]}
{"counts": {"clusters_in": N, "test_clusters_in": N, "absorbed_by_cross_merge": N, "tests_planned": N, "not_run": N,
 "gaps_accepted": N, "total_model_fits": "...", "total_gpu_hours": "..."}}
{"summary": "<=1800 chars: what the plan is, what would have to be true for a non-USDJPY side to pass, your
 probability that any does on this fixed data, and the single change that would raise that probability the most>"}

Check before you answer: every cid with verdict TEST in merge/clusters_all.jsonl appears exactly once: in one stage's
`ids`, or in `not_run`, or as an `absorb` entry of `merged`. Every cid with verdict REJECTED appears in `not_run`.
State the counts. Write plain complete sentences inside the JSON strings; no arrows, no invented abbreviations.
