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
| 1 | fixing-window response | features | SIGN | medium | `e51fx` (block): KILLED on every key (median paired path AUC gain +0.001..+0.003, below 0.005); `r4fixldn` / `r5fixecb` (zero-parameter post-fix reversal rules): KILLED |
| 2 | compact return kernels | features | SIGN | medium | `e52ck` (block), `e53ckcompact` (compact dictionary alone): KILLED, no AUC gain |
| 3 | round-level response vs shifted placebo | features | SIGN | medium | `e55rl` (real grid) vs `e56rs` (half-grid shifted placebo): KILLED (real beats placebo on 4 of 14 key-sides; lifts match the placebo's); `e10v02` |
| 4 | Brier-loss sign training | loss | SIGN | medium | `e59brier` (LightGBM on the Brier score): no VALID key-side passes the goal path test; USDJPY UP passes it (13/15, p10 0.5458) but is INVALID by drift with no placebo run, so it certifies nothing |
| 5 | heterogeneous equal-weight pool | ensemble | SIGN | medium | `e41pool`: no VALID key-side passes; USDJPY UP 14/15, p10 0.5479 (INVALID, no placebo run) |
| 6 | shrunk simplex stack | ensemble | SIGN | medium | `e60stack`: no VALID key-side passes; USDJPY UP 15/15, p10 0.5536 (INVALID, no placebo run); equal weights = its large-shrinkage limit, `e41pool` |
| 7 | recency-weighted training | loss | SIGN | low | `e65recency` (read on the forward walk only, since a CPCV fit pool can postdate its test groups): beats E1t on that reading only on AUDUSD UP and USDCAD DOWN, each on a single forward fold with at least 25 trades (33 and 41 trades); no key-side passes the goal path test (best VALID EURUSD DOWN 7/13 clear, p10 0.434) - KILLED |
| 8 | within-pair sign reflection | features | SIGN | low | `e57reflect` (LightGBM): KILLED - no key-side passes the goal path test (best USDCAD UP 9/15 clear, p10 0.505); `e58nysreflect` (kernel): KILLED - no key-side passes (best USDCAD UP 9/15, p10 0.498) |
| 9 | group-fused lag penalty | loss | SIGN | low | subsumed by `e29logit`: on the nested trailing returns an L2 penalty on the cumulative-return coefficients is a fused ridge on adjacent block coefficients (weighted by the standardisation scales; lab PROGRESS.md 13:20 UTC note) |
| 10 | USD-factor residual | features | SIGN | low | covered by `e21kf` (time-varying basket beta and residual) and `e22rm` (eigen-residual) |
| 11 | compact sign logistic | model | SIGN | low | `e54ckcompactlogit`: KILLED |
| 12-15 | HAR / multi-decay / filtered-historical / clock-scale size models | gate | SIZE | medium | magnitude gate `e13mag50` (learned on features that include multi-window realized volatility); HAR size gate `e62har`: KILLED - no key-side passes the goal path test (best USDJPY UP 11/14 clear, p10 0.466, INVALID) |
| 16 | calibration-uncertainty gate | gate | SIGN selection | medium | adaptive-conformal gate `e36aci`: no VALID key-side passes; USDJPY UP 13/15, p10 0.5440 (INVALID, no placebo run); bootstrap-map variants `e63bootnys` (kernel) / `e64boot` (LightGBM): no VALID key-side passes (best VALID USDCHF UP 10/15 clear, p10 0.482); USDJPY UP passes the path test (13/15, p10 0.5489; 14/15, p10 0.5493) but is INVALID by drift with no placebo run, below the finalist e30nystroem's DEV p10 0.5587 |
| 17-22 | synchronized-block inference, trial registry, step-down FWER, conditional calibration, episode influence, peer audit | validation | — | medium-high | trial registry: the lab registry (every test event); label-shift null replay of the selected configuration - a pre-registered diagnostic of that one configuration, not a replay of the search: `run_nullreplay.sh` (`a5null`); drift placebo and entry-delay audits (`analyze_drift_placebo.py`, `analyze_entry_delay.py`); step-down FWER: superseded for the opened keys by one finalist per key under a two-look group-sequential test with exact binomial boundaries (owner decision 6; `docs/DERIV_SOTA_PROGRAM.md`, Holdout discipline) |
| 23 | affine-logit calibration map | gate | SIGN probabilities | medium | a monotone map does not change which decisions clear a symmetric confidence quantile; folded into the ACI gate |
| 24-25 | shared-horizon tree partitions; ordinal signed bins | loss / label | SIGN | low | `e66mh` (30-minute rows with an indicator): KILLED - no key-side passes the goal path test (best VALID USDCHF UP 3/6 clear; USDJPY UP 11 of 12 counted paths clear, INVALID); `e61ord4` (4 ordinal bins): KILLED - no key-side passes (best VALID USDCHF UP 10/15 clear, p10 0.479; USDJPY UP 13/15, p10 0.5365, INVALID) |
