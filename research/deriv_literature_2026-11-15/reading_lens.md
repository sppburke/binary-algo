# Literature lens: read one shard of a research library and record every document

You are one of many independent readers covering a 13,846-document quant research library. Your
shard is a list of documents. Your job is to READ each one and write one structured record per
document, so that a later synthesis can find every idea that could improve short-horizon price
prediction for the instruments described below. Accuracy matters more than enthusiasm: record
what each document actually shows, with page-cited quotes for every number, and say plainly when
a result is weak, in-sample only, or irrelevant to us.

## Hard limits

- Library root (READ ONLY): `/home/sean/git/academic-papers`. Never modify, move or delete
  anything there. Read only the files named in your shard (and, if a document cites one that is
  load-bearing, you may look it up in `catalog.jsonl` and read it too; say so in `notes`).
- Write ONLY inside your current working directory: your output file is `records.jsonl` there.
  Small helper scripts there are fine. Nothing else anywhere.
- No network access, no installs. Use `/home/sean/binary-algo-venv/bin/python` or `python3`.

## Who will use your records (context for judging relevance)

We predict deriv.com "Rise/Fall" binary contracts on seven FX majors (AUDUSD, EURUSD, GBPUSD,
NZDUSD, USDCAD, USDCHF, USDJPY): will the price at expiry be above the entry price? Settlement is
Deriv's own 1-second mid price; entry is the next tick after the order, exit the last tick at or
before expiry; a tie (no move) LOSES; the payout makes 54.1% the break-even win rate. Horizons:
15 minutes is the shortest contract we can trade; 15m, 30m, 1h up to about a day matter. (Methods
shown at shorter horizons can still transfer — record them, and say at which horizon they were
shown.) We also model MAGNITUDE (will |return| over the horizon exceed a threshold) as a gate.

Our ONLY data: Deriv's own 1-second mid ticks for the 7 pairs, 2025-08-11 onward (about 15
months, true UTC; no volume, no bid/ask, no order book). Anything else — other venues' prices,
order flow, depth, options/implied volatility, rates, positioning, news, macro calendars — would be
new data that needs the owner's approval: still record such levers, with data_needed="external: …".
Ideas that work from a single mid-price series (and the other six pairs' mid series on the same
clock) are the most immediately usable.

Earlier in-house work (on another data feed with a clock error, so only a weak prior): models on
price-derived features converged to ~0.52 AUC at 1-5 minutes; a ~0.58 direction edge at 15
minutes and a strong magnitude edge (AUC 0.71-0.81) were reported; tuned gradient-boosted trees
matched or beat deep nets on tabular features; seed-ensembling, calibration, adaptive conformal
gating and |return|-weighted losses were used as edge improvers. Deriv's synthetic indices are
certified RNG output (no edge possible) — out of scope. A known theorem here: entropy /
order-flow / complexity / HMM / Hurst / Kalman statistics predict move SIZE, not SIGN — if a
"direction" claim rests on one of these, note it. Evaluation standard we hold results to:
out-of-sample by period, purged/embargoed combinatorial CV, no look-ahead, costs or ties charged,
multiple-testing awareness. With only ~15 months of data, sample efficiency, regularisation,
pooling across pairs and honest small-sample validation matter a lot — note ideas about these.

## How to read each document

Text files carry page markers: every line starts with `[P<n>]` (page n). Read pages with e.g.
`grep -E '^\[P(1|2|3|4)\] ' <file>`; `grep -c '^\[P' <file>` gives the page count.

1. Triage EVERY document: first ~4 pages (title, abstract, introduction) and the last ~2-3 pages
   before the references (conclusion). Decide relevance:
   - 3 = directly about short-horizon (seconds to hours) direction/return/volatility prediction,
     binary-outcome pricing or calibration, FX microstructure, or an evaluation/validation method
     that would change how we judge a model;
   - 2 = a model, feature, loss, ensembling, calibration, regime, data-source or validation idea
     plausibly transferable to our problem, or strong evidence for/against such an idea;
   - 1 = finance/ML but tangential (long-horizon asset pricing, portfolio theory, etc.);
   - 0 = not useful to us.
2. Relevance 2-3: read the WHOLE document (for books and documents over ~80 pages: table of
   contents plus every relevant chapter). Extract the load-bearing claims with exact numbers and a
   verbatim quote (<= 40 words) and its page. Grade the evaluation (A = proper out-of-sample with
   costs/realistic constraints and robustness; B = out-of-sample but thin or no costs; C =
   in-sample, single split, or cherry-picked; D = no real evaluation / anecdotal).
3. Turn every testable idea into a `lever` (see schema). Be concrete about the mechanism and what
   data it needs. Never invent numbers — write "not stated".

Budget: keep moving; do not spend more than ~10 minutes on any single document. If a text file is
missing or unreadable, still write a record with `"depth": "unreadable"`.

## Record schema — ONE JSON object per line in records.jsonl, one per document, in shard order

Relevance 0-1 (short record):
{"id": "...", "title": "...", "depth": "triage", "pages_read": "1-4,31-33", "relevance": 0, "what": "<= 25 words"}

Relevance 2-3 (full record):
{"id": "...", "title": "...", "depth": "full" | "partial", "pages_read": "...", "relevance": 2,
 "what": "<= 25 words",
 "findings": [{"claim": "...", "number": "the exact figure or 'not stated'", "quote": "verbatim <= 40 words", "page": 7}],
 "method": "model/technique in one or two sentences",
 "data": "assets, frequency, horizon, period, sample size",
 "evaluation_grade": "A|B|C|D", "evaluation_why": "one sentence",
 "levers": [{"lever": "what we would do", "mechanism": "why it could work",
             "kind": "direction|magnitude|calibration|selection-gating|loss|ensemble|features|labels|validation|data|sizing|regime",
             "scope": "fx|synthetic|commodity|index|crypto|all", "horizon": "seconds-10m|15-30m|1h-4h|day|any",
             "data_needed": "deriv-ticks|external: <what>",
             "evidence": "strong|moderate|weak", "falsifier": "the result that would kill it"}],
 "notes": "caveats, contradictions with other work, look-ahead or overfitting risks, references worth chasing"}

Append each record as soon as you finish the document (e.g. with a tiny Python helper using
json.dumps), so partial progress survives. Before finishing, verify: the number of lines equals
the number of documents in the shard, every line parses as JSON, ids match the shard, every
`number` other than "not stated" has a quote and page.

## Final message

Reply with a short Markdown summary: documents read by relevance level; the 10 most promising
levers in this shard (lever, document id, kind, evidence, data needed); any document you could not
read; anything in this shard that contradicts the in-house priors above.
