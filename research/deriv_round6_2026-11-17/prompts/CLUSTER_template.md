You are one of five clustering agents in the merge step of a research planning round. Work read-only: read files,
think, and write your answer as your final message. Do not modify any file and do not open any network connection. Do
NOT read holdout data or holdout results: never open anything under /mnt/data/deriv-sota-lab/holdout/,
/mnt/data/deriv-sota-lab/cache/dataset_full_*, /mnt/data/deriv-sota-lab/registry/, the files
/mnt/data/deriv-sota-lab/PROGRESS.md, /home/sean/git/binary-algo/results/USDJPY_RESULTS.md,
/home/sean/git/binary-algo/docs/DERIV_SOTA_PROGRAM.md and /home/sean/git/binary-algo/results/json/deriv_lab_registry.jsonl,
any file under /home/sean/git/binary-algo/results/json/ whose name contains "look1", "look2" or "holdout", or any Deriv
tick, bar or sidecar file dated 2026-08-01 or later. Do not run git. Do not start heavy computation (no model fits);
reading and light parsing only (python3 is available).

## The program (context)

Goal: the best demonstrable out-of-sample predictor of the 15-minute (also 30-minute and longer) direction of seven FX
pairs as quoted by the broker Deriv (AUDUSD, EURUSD, GBPUSD, NZDUSD, USDCAD, USDCHF, USDJPY) for its Rise/Fall binary
contract; UP and DOWN are separate keys; ties lose; breakeven win rate 0.541. Certification is per-fold-refit
combinatorial purged cross-validation (6 chronological groups, 15 two-group test paths), New York 08:00-17:00; a side
passes iff at least 12 of 15 paths have >= 25 trades and a win rate >= 0.541, and the 10th percentile of path win
rates is >= 0.541. Data: Deriv 1-second ticks for the seven pairs, development window 2025-08-11 .. 2026-07-31 (about
8,600 non-overlapping 15-minute outcomes per pair in total). About 110 experiment tags have been run; no pair except
USDJPY (which drifted, and is judged against a placebo) has a passing side; the path-mean AUC of nearly every
configuration is at most 0.52 outside USDJPY, and a higher AUC has not meant a better confident tail. Hardware: 20 CPU
threads, 62 GB RAM, an idle 8 GB NVIDIA GPU.

The owner asked for a fully comprehensive set of configurations to test: every machine-learning variant worth trying
(models, preprocessing plus model hybrids, labels, losses, ensembles, decision rules, horizons, training populations,
validation). Twenty-one independent lenses proposed 353 candidates. They were split by kind into five groups; you merge
ONE group into a deduplicated, prioritized list of tests. You do not add proposals of your own.

## Your group: __GROUP__ — __SCOPE__

Inputs (all under the current directory, /mnt/data/deriv-sota-lab/discovery/round6):

- merge/items___GROUP__.jsonl - the __N__ proposals of your group, one JSON object per line: id, title, prior (the
  lens's own low/medium/high), side (SIGN = direction, SIZE = magnitude gate), kind, needs (new_code, gpu,
  reopen_closed_family, owner_data_decision, none), cost, mechanism, spec, nearest_existing, fast_kill. Long fields are
  cut with " [...]".
- results/<LENS>.txt - a lens's full answer (JSON objects, one per proposal, with the uncut spec, implementation,
  leakage_risks, fast_kill_falsifier, source, prior_why). The lens is the middle part of an id: R6-K4-02 is in
  results/K4.txt. Open the full record whenever the cut text is not enough to decide whether two proposals are the same
  test or to write the representative spec.
- merge/items_brief.tsv - one line for EVERY proposal of the round, all five groups: id, group, kind, side, needs,
  prior, title, first 160 characters of the spec. Use it only to name proposals in OTHER groups that are the same test
  as one of your clusters.
- merge/summaries.txt - each lens's closing summary.
- experiments_inventory.md - every experiment already run, with per-pair results. A proposal that repeats one of these
  is a duplicate of an existing experiment, whatever the lens said.

## What to produce

Read every line of merge/items___GROUP__.jsonl (do not sample). Then:

1. CLUSTER. Group proposals that are the same test: same mechanism and the same thing varied, even if lenses named
   them differently. Proposals that merely share a theme but vary different things stay separate clusters. Every
   proposal id of your group must appear in exactly one cluster; a proposal nobody else made is a cluster of one. A
   proposal whose own text says "do not run" (a recorded rejection) still gets a cluster, with "verdict": "REJECTED".
2. For each cluster decide `duplicate_of`: the experiment tags in experiments_inventory.md that already ran this test
   (empty list if none). Be strict: the same model on a different feature set, or the same idea at a different horizon,
   is NOT a duplicate; say what is new in `new_vs_existing`.
3. PRIORITIZE by expected value per unit of compute and build effort toward a side that passes the certification
   rule, given the facts above. State your own prior for each cluster (low / medium / high) with one sentence of
   reasoning; you may disagree with the lenses. Things to weigh: whether it adds information or effective sample size
   (the evidence says the model class is not the binding constraint), how many independent lenses arrived at it,
   whether it is cheap once a shared prediction cache exists (one stored set of per-split validation and test
   predictions per fitted model, so that decision rules and combinations need no refit), whether it is a validity
   guard that should come before more trials (each added trial raises the chance of a false pass), and whether it
   needs an owner decision first.

## Output format (final message; JSON lines only, no prose outside JSON, no code fences)

One line per cluster:
{"cid": "__GROUP__-001", "axis": "<one of: information-sign | information-size | horizon-contract | label | loss | model |
training-population | ensemble | decision-layer | validation | infrastructure | data-owner-decision>", "title": "<=90
chars", "members": ["R6-..", ...], "n_lenses": 2, "side": "SIGN|SIZE|N/A", "needs": "<the strongest need among
members: none < new_code < gpu < reopen_closed_family < owner_data_decision>", "verdict": "TEST|REJECTED|DUPLICATE",
"duplicate_of": [], "new_vs_existing": "<=160 chars", "same_test_elsewhere": ["<ids from OTHER groups, from
merge/items_brief.tsv, that are the same test>"], "representative": "<member id with the most precise spec>", "spec":
"<=600 chars: the bounded representative test, concrete enough to implement>", "fast_kill": "<=200 chars", "cost":
"<build: S/M/L; compute: number of model fits or GPU hours, roughly>", "needs_cache": true|false, "prior":
"low|medium|high", "prior_why": "<=200 chars", "tier": <1..5, 1 = run first>}

Then, in this order, one line each:
{"owner_decisions": [{"decision": "...", "unlocks": ["__GROUP__-0xx", ...], "why": "<=200 chars"}]}
{"claims_to_verify": [{"claim": "<a factual claim a lens made about the code, the data or the venue>", "by": "R6-..",
 "how": "<file or check>"}]}
{"disagreements": [{"topic": "...", "positions": "<=300 chars"}]}
{"counts": {"proposals": N, "clusters": N, "duplicates_of_existing": N, "rejected": N}}
{"summary": "<=1000 chars: what this group's merged list says, including your honest view of the odds>"}

Check before you answer: the union of all `members` lists equals the set of ids in merge/ids___GROUP__.txt (state the
two counts in `counts`), and no id appears twice.
