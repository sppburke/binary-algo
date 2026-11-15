# Synthesis lens — Deriv FX direction program (one lever family)

You are one of several synthesis readers. Thousands of literature records were extracted, one per document, by
earlier readers from a 13,846-document corpus. Your input is the subset of extracted **levers** of ONE kind (see
"Your input" below): a JSONL file, one lever per line with fields `doc` (document id), `title`, `rel` (document
relevance 0-3), `grade` (evaluation grade A-D of the document's own evidence), `lever`, `mechanism`, `kind`, `scope`,
`horizon`, `data_needed`, `evidence` (strong / moderate / weak), `falsifier`. The full per-document records (with
verbatim quotes and page numbers) are in `/mnt/data/deriv-sota-lab/literature/work/<shard>/records.jsonl`; the page-marked
source text of every document is at the path in the shard file `/mnt/data/deriv-sota-lab/literature/shards/<shard>.jsonl`
(field `text`), lines prefixed `[P<page>]`.

## The program (what the levers are for)

- Instruments: 7 FX majors on Deriv (EURUSD GBPUSD USDJPY USDCHF USDCAD AUDUSD NZDUSD). Product: Rise/Fall binary
  options, 15-minute minimum expiry (also 30m and longer). UP and DOWN are separate keys; ties lose; breakeven win
  rate 0.541. Magnitude (|return| >= Q) only as a gate.
- Data: ONLY Deriv's own 1-second mid-price ticks (no bid/ask, no volume, no order book), 2025-08-11 onward (~15 months).
  No Dukascopy, no external data unless the owner later approves a specific source.
- Current models: LightGBM on causal minute features (wall-clock returns, volatility, range, EMA distance, activity,
  clock, other pairs' as-of returns, multi-timeframe technical indicators), NY-session gate, per-fold-refit CPCV
  certification (15 purged paths, p10 >= 0.541, >= 80% of paths clear) and a sealed forward holdout.
- Measured so far on Deriv data: direction AUC about 0.50-0.53 on moved bars; one pair near the certification bar.
  Direction at these horizons is close to efficient; most levers will be null. Statistics such as entropy,
  order-flow proxies, Hurst exponents, HMM states, Kalman filters and complexity measures gate move SIZE, not SIGN
  (sign-invariance) — classify such ideas as magnitude/gating, not direction.
- Closed families (do not propose re-running unless a lever is materially different, and say why): neural/spectral
  forecasters (N-BEATS/N-HiTS, DLinear/Autoformer/FEDformer/TFT, DWT/SSA), bar-image CNNs, GRU/ESN/reservoir
  sequence models, time-series foundation models (TSFM/Kronos), Optuna hyperparameter search, and cross-pair POOLED
  training for the USDJPY/USDCAD/AUDUSD/NZDUSD family.

## Your task

1. Read every lever in your input file. Cluster near-duplicates (same mechanism and test) into one **candidate**;
   keep the strongest supporting documents (doc ids) for each.
2. For each candidate decide: `testable_now` (Deriv ticks only, 15m+ horizon, implementable on minute bars or ticks),
   `needs_external_data` (name the data), `closed_family`, `out_of_scope` (horizon < 15 min, other products), or
   `already_done` (the program already does it: say where).
3. For `testable_now` candidates give: a precise implementation spec on our data (inputs, formula or model change,
   where it enters: features / label / loss / gate / validation / ensemble), the mechanism and whether it can carry
   SIGN (direction) or only SIZE (magnitude/gate), a pre-registered **fast-kill falsifier** in our terms (e.g. "KILL if
   moved AUC gain over the base < 0.005 on the 15 CPCV paths, or no forward-walk improvement"), a prior (low / medium
   / high) with one line of justification grounded in the cited evidence, and the cost (hours to build + compute).
4. Verify the load-bearing claim of each candidate you rank HIGH or MEDIUM: open the cited document's text, find the
   supporting passage and quote it (<= 40 words) with its page. If you cannot find support, downgrade it and say so.
5. Rank. Put the most promising testable direction levers first, then gating/magnitude levers, then validation
   upgrades that change our certification decisions.

## Output

Write `candidates.jsonl` in your working directory: one JSON object per candidate with fields
`id` (short slug), `title`, `kind`, `status` (one of the five above), `docs` (list of doc ids), `spec`, `enters`
(features|label|loss|gate|validation|ensemble|model), `sign_or_size`, `mechanism`, `falsifier`, `prior`,
`prior_why`, `cost`, `verified_quote` ({doc, page, quote} or null), `notes`.
Then reply with a short Markdown summary: counts by status, and the top 10 testable candidates (one line each).
Never invent numbers, pages or quotes; if a field is unknown, write null.
