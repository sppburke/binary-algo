# Consolidation lens — Deriv FX direction program (all lever families)

Several synthesis readers each clustered ONE family of literature levers into candidates. Your job is to merge their
outputs into one deduplicated, ranked, program-aware backlog.

## Inputs (read-only)

- `/mnt/data/deriv-sota-lab/literature/synthesis/work/*/candidates.jsonl` — one candidate per line, fields `id`, `title`,
  `kind`, `status` (testable_now / needs_external_data / closed_family / out_of_scope / already_done), `docs`, `spec`,
  `enters`, `sign_or_size`, `mechanism`, `falsifier`, `prior`, `prior_why`, `cost`, `verified_quote`, `notes`. The
  directory name is the chunk (`<family>__<k>of<n>`).
- `/mnt/data/deriv-sota-lab/literature/synthesis/results/*.md` — each reader's summary.
- `/mnt/data/deriv-sota-lab/literature/program_experiments.md` — every experiment the program has run or queued, with
  the SWEEP_MATRIX family coverage map. Treat it as the authority on what is already done.
- Source texts for quote checks: shard files `/mnt/data/deriv-sota-lab/literature/shards/<shard>.jsonl` (field `text`,
  pages marked `[P<page>]`); per-document records in `/mnt/data/deriv-sota-lab/literature/work/<shard>/records.jsonl`.

## The program (constraints that decide status)

Deriv Rise/Fall on 7 FX majors, 15-minute minimum expiry (also 30m+), UP and DOWN separate keys, ties lose, breakeven
0.541; data = Deriv 1-second mid ticks only (no bid/ask, volume or book), 2025-08-11 onward; per-fold-refit CPCV
certification (15 purged paths, p10 >= .541, >= 80% of paths clear, NY session) plus a sealed holdout. Closed families:
neural/spectral forecasters, bar-image CNNs, GRU/ESN/reservoir, time-series foundation models, Optuna search, pooled
training for the USDJPY/USDCAD/AUDUSD/NZDUSD family. Entropy / Hurst / HMM / Kalman / complexity / order-flow-proxy
statistics gate SIZE not SIGN (sign-invariance). Measured so far: direction AUC about 0.50-0.53 on moved decisions; the
data ceiling is about 15 months, so low-capacity and data-efficient levers matter.

## Task

1. Merge candidates that are the same mechanism and the same test across chunks into one consolidated item; keep the
   union of doc ids and the best verified quote.
2. For each item decide the status against `program_experiments.md`: `already_done:<experiment id(s)>` only when an
   existing run tests the same mechanism the same way; `variant_of:<experiment id>` when it is a materially different
   variant of a run (say exactly what differs); otherwise keep the readers' status (testable_now / needs_external_data
   / closed_family / out_of_scope). Never mark something testable_now if it needs data the program does not have.
3. For testable_now and variant_of items: one precise implementation spec on our data (what enters where: features,
   label, loss, gate, validation, ensemble, model), sign_or_size, a pre-registered fast-kill falsifier in program terms,
   prior (low / medium / high) with a one-line justification grounded in the cited evidence, and cost.
4. Rank testable items by expected value for this program (prior x relevance to 15m FX direction on ~15 months of data /
   cost). For the top 25, confirm the verified quote exists in the source text (open the shard text at that page); if
   it does not, set `quote_check` to `failed` and downgrade the prior one step.

## Output

Write `consolidated.jsonl` in your working directory: one JSON object per item with fields `id`, `title`, `families`
(chunk families merged), `status`, `docs`, `spec`, `enters`, `sign_or_size`, `mechanism`, `falsifier`, `prior`,
`prior_why`, `cost`, `verified_quote` ({doc, page, quote} or null), `quote_check` (ok / failed / not_checked), `rank`
(integer for testable items, null otherwise), `merged_from` (list of `<chunk>:<candidate id>`).
Then reply with a Markdown summary: counts by status, the top 25 testable items (one line each: rank, id, enters,
sign_or_size, prior, cost, falsifier), and the items you marked already_done with the experiment ids.
Never invent numbers, pages or quotes; if a field is unknown, write null.
