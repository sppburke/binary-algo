Wrote [consolidated.jsonl](/mnt/data/deriv-sota-lab/literature/consolidation/work/consolidated.jsonl): **2,420 items**, accounting for all **4,059 inputs exactly once**.

| Status | Count |
|---|---:|
| testable_now | 1,507 |
| variant_of:* | 41 |
| already_done:* | 30 |
| needs_external_data | 611 |
| closed_family | 108 |
| out_of_scope | 123 |

The backlog has **1,548 ranked items**. Among the top 25, **20 quotes passed**. Ranks **7, 8, 11, 24 and 25** lacked a supplied quote/page and were marked `failed`; their priors remain at the low floor.

Costs below are estimated build hours; complete compute estimates and specifications are in the JSONL. ΔAUC means paired moved-decision AUC gain. Trading promotion retains the unchanged CPCV and sealed-holdout requirements. High validation priors concern certification quality.

| Rank | ID | Enters | sign_or_size | Prior | Build h | Fast-kill falsifier |
|---:|---|---|---|---|---|---|
| 1 | c-fixing-window-response | features | SIGN | medium | 4–8 | Median ΔAUC < .005 or no forward Brier improvement |
| 2 | c-compact-return-kernels | features | SIGN | medium | 5–10 | Median ΔAUC < .005 or forward gain ≤ 0 |
| 3 | c-features-round-level-response | features | SIGN | medium | 4–8 | Median ΔAUC < .005 or effect disappears against shifted levels |
| 4 | c-brier-sign-training | loss | SIGN | medium | 3–6 | No proper-score/calibration and matched-coverage utility improvement |
| 5 | c-heterogeneous-equal-pool | ensemble | SIGN | medium | null | Median ΔAUC < .005, fewer than 12/15 improve, or no forward gain |
| 6 | c-shrunk-simplex-stack | ensemble | SIGN | medium | null | Median ΔAUC < .005, fewer than 12/15 improve, or no forward gain |
| 7 | c-recency-training-loss | loss | SIGN | low | 2–4 | Mean ΔAUC < .005 or no forward utility gain |
| 8 | c-within-pair-sign-reflection | features | SIGN | low | 4–8 | Median ΔAUC < .005, no forward gain, or invalid reflection |
| 9 | c-group-fused-lag-penalty | loss | SIGN | low | 5–8 | Median ΔAUC < .005 or nonpositive forward gain |
| 10 | c-usd-factor-residual | features | SIGN | low | 8–16 | Median ΔAUC < .005 or nonpositive forward gain |
| 11 | c-compact-sign-logistic | model | SIGN | low | 4–8 | Median ΔAUC < .005 or no forward probability/utility improvement |
| 12 | c-har-size | gate | SIZE | medium | 3–5 | No magnitude-Brier improvement or no matched-coverage strict-win gain |
| 13 | c-multi-decay-variance | gate | SIZE | medium | 2–4 | Magnitude ΔAUC < .005 or log-loss reduction < 1% |
| 14 | c-filtered-historical-size | gate | SIZE | medium | 3–6 | Proper-score improvement on fewer than 12/15 paths or selected-win gain < .005 |
| 15 | c-clock-scale-normalization | features | SIZE | medium | 4–7 | Median magnitude-Brier reduction < .5% or fewer than 12/15 improve |
| 16 | c-calibration-uncertainty-gate | gate | SIGN selection | medium | 4–7 | Proper scores fail on 12/15 paths or selected-win gain < .005 |
| 17 | c-synchronized-block-inference | validation | VALIDATION | high | 4–8 | No repeatable improvement in certification or uncertainty decisions |
| 18 | c-search-trial-registry | validation | VALIDATION | high | 4–8 | Missing attempted trials or trial IDs cannot reproduce predictions |
| 19 | c-search-stepdown-fwer | validation | VALIDATION | high | 6–12 | Adjusted advantage fails .05 significance or null rejection exceeds budget |
| 20 | c-conditional-calibration-audit | validation | VALIDATION | medium | 3–6 | No repeatable reliability issue or decision value |
| 21 | c-episode-influence-audit | validation | VALIDATION | medium | 3–6 | Removing one cluster eliminates improvement or breaks certification |
| 22 | c-peer-sign-size-audit | validation | VALIDATION | medium | 8–16 | Injected defect is missed or retention decisions become unstable |
| 23 | c-affine-logit-map | gate | SIGN probabilities | medium | 3–6 | Median Brier reduction < .5% or fewer than 12/15 improve |
| 24 | c-shared-horizon-tree-partitions | loss | SIGN | low | 6–10 | Mean ΔAUC < .005 or no forward utility gain |
| 25 | c-ordinal-signed-bins | label | SIGN | low | 6–10 | Median ΔAUC < .005, nonpositive forward gain, or worse proper loss |

The following items were marked already done. Item names below omit their common `c-` prefix.

| Experiment IDs | Consolidated item IDs |
|---|---|
| e13mag50 | direct-magnitude-classifier; existing-fold-thresholds; existing-sign-size-separation; magnitude-direct-event-vs-remote-es |
| e18lowcap | existing-low-capacity-trees |
| e29logit | l2-logistic-current-features |
| e1t | existing-activity-features; existing-asof-crosspair; existing-causal-features; existing-clock-features; existing-contract-labels; existing-cpcv-forward |
| e1t | existing-deriv-feed; existing-feature-retention; existing-fold-selection; existing-lightgbm; existing-no-holdout-selection; existing-physical-event-probability |
| e1t | existing-purge-maturity; existing-refit-transforms; existing-return-relative-target; existing-sealed-holdout; existing-separate-pair-models; existing-strict-utility-validation; existing-vol-range-activity |
| e1t, e10v02 | existing-clock-scale |
| e1t, e14nylate, e15nypm, e16nyeve, e17nyam | existing-ny-session |
| e1t, e19tf | existing-mid-input-limit |
| e1t, e26pa | existing-return-technical |
| e21kf, e24cx | existing-filtered-state |