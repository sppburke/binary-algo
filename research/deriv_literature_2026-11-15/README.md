> **SCOPE: GENERIC** (literature fan-out for the Deriv-feed program; no per-key numbers — results live in
> `results/<PAIR>_RESULTS.md`). See `docs/DERIV_SOTA_PROGRAM.md`.

# Deriv-feed literature fan-out (2026-11-15)

**What was read.** Every document of the `~/git/academic-papers` corpus (13,846 documents, `catalog.jsonl`; page-marked
text), split into 254 shards and read by Codex lenses (`reading_lens.md`): one structured record per document —
relevance 0-3, reading depth, pages read, what it says, and every testable lever with mechanism, data need, horizon,
evidence grade and a falsifier, with page-cited quotes. Records: `document_records.jsonl.gz` (13,846 lines, one per
document); per-shard summaries: `shard_summaries.md.gz`.

**Synthesis.** The 21,270 extracted levers were grouped by kind (validation 6,852; features 3,264; magnitude 1,989;
calibration 1,806; ensemble 1,668; selection-gating 1,234; regime 1,099; data 955; loss 788; labels 510; direction 483;
sizing 192; levers below 15 minutes 430, kept for the record only) and read by 33 synthesis lenses
(`synthesis_lens.md`, one per chunk of at most 700 levers): near-duplicates clustered into 4,059 candidates, each with a
status, an implementation spec on Deriv data, sign-or-size, a fast-kill falsifier, a prior and a page-cited quote check
for every medium or high prior (`synthesis_summaries.md.gz`).

**Consolidation.** One lens (`consolidation_lens.md`) merged the 4,059 candidates across chunks into 2,420 items, mapped
each against the program's run or queued experiments, ranked the testable ones and re-checked the top 25 quotes against
the source text (`consolidation_summary.md`; full items `consolidated_backlog.jsonl.gz`, fields `id`, `title`,
`families`, `status`, `docs`, `spec`, `enters`, `sign_or_size`, `mechanism`, `falsifier`, `prior`, `prior_why`, `cost`,
`verified_quote`, `quote_check`, `rank`, `merged_from`).

| Status | Items | Disposition |
|---|---:|---|
| testable_now | 1,507 | ranked; worked in rank order (table below); the long tail is low prior |
| variant_of an existing run | 41 | recorded against that run |
| already_done | 30 | mapped to the run ids (`e1t`, `e13mag50`, `e18lowcap`, `e29logit`, the session runs, ...) |
| needs_external_data | 611 | proposals for the owner (no acquisition without approval); most need news / macro / sentiment / volume / bid-ask / options data |
| closed_family | 108 | repository closures (neural/spectral, image, GRU/ESN, TSFM, Optuna, JPY/CAD/AUD/NZD pooling) |
| out_of_scope | 123 | horizons below 15 minutes or other products |

**Reading of the literature, in one line.** Direction at 15-30 minutes on a mid-only feed is close to efficient; the
evidence-backed levers are regularization / ensembling, a handful of session-structure features (fixings, round
levels), size (magnitude) gates, and validation upgrades that make certification decisions harder to fool. No
document supplies a high-prior direction lever that the program could test with Deriv data alone.

## Ranked testable items and their disposition

| Rank | Item | Enters | Sign or size | Prior | Disposition (experiment ids; results per key in `results/<PAIR>_RESULTS.md`) |
|---:|---|---|---|---|---|
| 1 | fixing-window response | features | SIGN | medium | `e51fx` (block), `r4fixldn` / `r5fixecb` (zero-parameter post-fix reversal rules) |
| 2 | compact return kernels | features | SIGN | medium | `e52ck` (block), `e53ckcompact` (compact dictionary alone) |
| 3 | round-level response vs shifted placebo | features | SIGN | medium | `e55rl` (real grid) vs `e56rs` (half-grid shifted placebo); `e10v02` |
| 4 | Brier-loss sign training | loss | SIGN | medium | `e59brier` (LightGBM on the Brier score) |
| 5 | heterogeneous equal-weight pool | ensemble | SIGN | medium | `e41pool` |
| 6 | shrunk simplex stack | ensemble | SIGN | medium | pending (equal weights = its large-shrinkage limit, `e41pool`) |
| 7 | recency-weighted training | loss | SIGN | low | pending (walk-forward form belongs to the holdout reading) |
| 8 | within-pair sign reflection | features | SIGN | low | `e57reflect` (LightGBM), `e58nysreflect` (kernel) |
| 9 | group-fused lag penalty | loss | SIGN | low | pending |
| 10 | USD-factor residual | features | SIGN | low | covered by `e21kf` (time-varying basket beta and residual) and `e22rm` (eigen-residual) |
| 11 | compact sign logistic | model | SIGN | low | `e54ckcompactlogit` |
| 12-15 | HAR / multi-decay / filtered-historical / clock-scale size models | gate | SIZE | medium | magnitude gate `e13mag50` (learned on features that include multi-window realized volatility); specific variants pending |
| 16 | calibration-uncertainty gate | gate | SIGN selection | medium | adaptive-conformal gate `e36aci`; bootstrap-map variant pending |
| 17-22 | synchronized-block inference, trial registry, step-down FWER, conditional calibration, episode influence, peer audit | validation | — | medium-high | trial registry: the lab registry (every test event); full-search null replay of the selected configuration: `run_nullreplay.sh` (`a5null`); drift placebo and entry-delay audits (`analyze_drift_placebo.py`, `analyze_entry_delay.py`); step-down FWER pending |
| 23 | affine-logit calibration map | gate | SIGN probabilities | medium | a monotone map does not change which decisions clear a symmetric confidence quantile; folded into the ACI gate |
| 24-25 | shared-horizon tree partitions; ordinal signed bins | loss / label | SIGN | low | pending |
