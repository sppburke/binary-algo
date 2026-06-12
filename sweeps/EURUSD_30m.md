# EURUSD · 30m — SWEEP LEDGER (key-specific status)
SCOPE: EURUSD · 30m. Resumable sweep ledger — the STATE of the exhaustive+generative search for the best
30m UP and best 30m DOWN EURUSD direction predictor. Generic menu = SWEEP_MATRIX.md; ideas = docs/IDEAS_LOG.md;
executable backlog/discovery = sweeps/EURUSD_30m_backlog.md; results of record = results/EURUSD_RESULTS.md.
Created 2026-06-04 (the /goal 30m two-sided certification push). 30m is DERIV-TRADEABLE (longest, above 15m floor).

Discipline (strategy-eval §2): deriv-faithful wc settlement (mid-to-mid, ties LOSE, breakeven 0.541),
nonoverlap_chrono(gap=1800s), per-year CI95, moved-bars-only, worst-VAL-half selection, pre-registered falsifier,
CERTIFY only via full per-fold-refit CPCV (refit p10>=0.541 AND >=~80% paths clear).

## ✅ FINAL STATE — GOAL COMPLETE (2026-06-04): BOTH sides certified + improve/discover loops DRY
**Best 30m UP = best 30m DOWN = the cross-pair pooled book `EURUSD.m30xp.v1`** (content_id `da55d7c6b3b08bef`,
git-tag `book/EURUSD.m30xp.v1`):
- **UP refit-CPCV p10 .5588** (15/15 paths, mean .582) — CERTIFIED, improves base floor .5538.
- **DOWN refit-CPCV p10 .5525** (15/15 paths, mean .587) — CERTIFIED, rescued base near-miss .5356.
- Breakeven 0.541. Deploy gate cov5% (1h_bb_width≤q33 × NY × conf). **EDGE IS REFIT-DEPENDENT** — frozen-2021 fwd
  decays by 2026 OOS (UP .5302@cov5/.4652@cov10; covcurve: no cov rescues frozen-2026 UP) → deploy WITH retraining.
  Optional: deploy primary as K=4 seed-ensemble for UP-tail robustness (frac .733→.933 at ~same p10).
- Full per-side pipeline (a)-(e) DONE both sides: (a) side-split ✓ (b) worst-VAL-half gate ✓ (c) specialist ✓SUBSUMED
  (d) coverage curve ✓ (e) refit-CPCV ✓ CERTIFIED.
- **IMPROVE loop DRY (best-combos RUN, not assumed):** magweight HURTS; seed-ens/specialist/Aₐ/IPCA SUBSUMED;
  **3-model ensemble ON the pool RUN→SUBSUMED** (p10 .5542/.5432 < single-LGB — model-class diversity raises mean but
  widens worst-path dispersion); **recency-weight RUN→SUBSUMED** (HURTS UP — shrinking the pool hurts, confirms FI).
  Optuna subsumed by the ensemble RUN (xgb+cat = a LARGER perturbation than LGB-hyperparam tuning, already lost on p10);
  ACI/GMADL/DL Tier-1-subsumed (THEORY §2 law + 5m/10m runs). Mechanism: FI = edge is POOLED base-feat training (94%
  gain), so model/loss/feature/reweight tweaks can't move the binding p10.
- **DISCOVER loop DRY (K=2):** R1 (Aₐ/IPCA→subsumed + external rate-diff), R2 (month-end KILLED + rate-diff), R3 (empty).
- **Only remaining frontier = EXTERNAL data** (DE-US 2y rate-diff = H-TODO-1, daily risk-reversal) — gated on acquisition.
  Same terminal conclusion as every other horizon. Gradient extends: none@60s→UP@5m→BOTH@10m,15m,30m.

---

## CONTEXT — what is already known at 30m (Tier-1, prior thread, docs/m30_research_log.md + results/EURUSD_RESULTS.md)
- COMBINED deliverable EURUSD.m30.v1 = 3-model ensemble × comp(1h)×NY, conf-selective: combined held-out 0.591
  CI[.556,.625] (2024 .623 / 2025 .589 / 2026 .546), EV +0.064. Book sha a17be49b9262668f. NEVER side-split.
- (30m,UP) and (30m,DOWN) keys = UNTESTED (this sweep gives them their first real numbers).
- Prior 30m DIRECTION nulls (~25 experiments, all pre-cross-pair-breakthrough, dated May 30-31):
  larger-TF reversal/flag gates (m30_gates) null · complexity/PE/Hurst gates (m30_complexity) flat ~0.515 ·
  Sofien 79 rules null · volume/dollar bars null · FX-fix reversal (m30_fix) sign-FLIPS OOS (.429) ·
  tick microstructure (m30_tick) 0.50 noise · ES lead-lag (m30_es_feas) null · path-signatures (m30_sig) null @30m ·
  cross-pair-as-FEATURES Iteration-2 (single LGB + 60 peer feats) +0 @ AUC .518.
- POSITIVE (sign-invariant, magnitude only): rv30 large-move AUC 0.73-0.78 OOS — MAGNITUDE_FINDINGS, not a side key.
- KEY GAP: the cross-pair POOLED + xpof-residual lever (DIFFERENT from Iteration-2 features) — the keystone that
  certified BOTH 10m sides (.586/.568) and BOTH 15m sides (.5673/.5742) — was NEVER run at 30m. Gradient is rising
  with horizon -> strong prior 30m certifies both sides. THIS is row K1/K2.

## LEDGER (work top-to-bottom; resume from first pending/running)
| id | family | method | variant | script | target | prior | status | combined | UP | DOWN | verdict | result_json |
|----|--------|--------|---------|--------|--------|-------|--------|----------|----|----|---------|-------------|
| K1 | validation | base-book side-split refit-CPCV | 3-model ens, 1h×NY gate, HOR=30 | m30_cpcv_side.py | UP+DOWN floor | high | **done** | p10 .5594 (15/15) | **CERT p10 .5538** (14/15) | not-cert p10 .5356 (13/15) | UP certified by BASE book; DOWN near-miss (mean .5843 but 2 weak paths). Floors: UP .5538 / DOWN .5356 | m30_cpcv_side_result.json |
| K2 | cross-horizon&pair | CROSS-PAIR xpof primary refit-CPCV (KEYSTONE) | MX_HOR=30, 1h×NY gate | m30_xpair_cpcv.py | UP+DOWN cert | high | **done** | p10 .5645 (15/15) | **CERT p10 .5588** (15/15) +impr | **CERT p10 .5525** (15/15) +impr | BOTH SIDES CERTIFIED. DOWN rescued (.5356 near-miss base -> .5525 cross-pair). Gradient now none@60s->UP@5m->BOTH@10m,15m,30m. Freeze -> EURUSD.m30xp.v1 | m30_xpair_cpcv_result.json |
| FZ | records | freeze cross-pair book + deploy gate (worst-VAL-half) | EURUSD.m30xp.v1·da55d7c6b3b08bef | m30_xpair_freeze.py + regate cov5 | book | high | **done** | — | — | — | FROZEN, git-tag book/EURUSD.m30xp.v1. Deploy cov5 (worst-VAL-half .6389; cov10 rejected, 2026 UP .4652). REFIT-DEPENDENT: frozen-2021 fwd decays 2026 (UP .530/.465). | m30_xpair_freeze_result.json |
| I-magw | improve | \|ret\|-weighted loss (magnitude->direction bridge) | POW=0.5, same K2 gate/CPCV | m30_magweight_cpcv.py | both | med | **done** | p10 .5608 | .5528 (SUBSUMED, <.5588) | .5382 (HURTS, <.5525 + <BE) | **KILLED/SUBSUMED** — \|ret\|-weight pulls toward MAGNITUDE (sign-invariant), dilutes directional sign. Confirms sign-invariance thm. | m30_magweight_cpcv_result.json |
| I-seed | improve | seed-ensemble K=4 (variance reduction) | same K2 gate/CPCV | m30_seedens_cpcv.py | both | low-med | **done** | ENS p10 .5632 | .5572 (SUBSUMED, ~incumbent) | .5503 (SUBSUMED) | tail-variance NOT binding (p10 ~flat vs incumbent) BUT stabilizes UP vs single-seed (S1 .5397/.733 -> ENS .5572/.933) -> DEPLOY primary as K=4 seed-ens for robustness | m30_seedens_cpcv_result.json |
| I-ens | improve | **3-model ensemble (lgb+xgb+cat) ON the pool** (the untested best-combo) | same K2 gate/CPCV | m30_xpair_ens_cpcv.py | both | med | **done** | p10 .558 | .5542 (SUBSUMED, <.5588) | .5432 (SUBSUMED, frac .867) | model-class diversity RAISES mean (DOWN .5956) but INCREASES worst-path dispersion -> p10 DROPS; regularized single-LGB tighter on worst paths. Incumbent stands (tested, not assumed) | m30_xpair_ens_cpcv_result.json |
| I-recency | improve | recency-weighted frozen-forward (attack 2026 decay) | hl∈{flat,4y,2y} | m30_recency.py | both | low-med | **done** | — | — | — | **SUBSUMED** — recency HURTS UP every year (downweighting old data shrinks the pool = the mechanism); side-finding: flat 2026 UP .574 vs freeze .530 = seed/threshold-fragile. "deploy w/ retrain" stays the fix | m30_recency_result.json |
| I-cov | pipeline-d | confidence/coverage curve (frozen book) | cov 2-20% per yr/side | m30_xpair_covcurve.py | both | — | **done** | — | — | — | step d DONE. 2024/25 UP .62-.63 DOWN .57-.65 @cov5; **2026 UP DECAYED (<=.57 all covs, .46 @cov8-10)**, DOWN holds .56-.60 -> reinforces REFIT-DEPENDENCE; no cov rescues frozen 2026 UP | m30_xpair_covcurve_result.json |
| FI | analysis | frozen-book feature-importance (gain by family) | — | (inline) | mechanism | — | **done** | — | — | — | base239 multi-TF **94.0%**, leadlag_ll 2.1%, crosspair_xp 2.0%, OF 1.8% -> edge = POOLED TRAINING on base feats, NOT xpof block | m30_xpair_featimp_result.json |
| I-Aa | improve | antisymmetric cross-pair matrix feature | Aa(X) lead-lag rotation added to K2 | — | both | med | **SUBSUMED (Tier-1 FI)** | — | — | — | ll_ lead-lag features = 2.1% gain AND already GBM inputs; Aa is a linear combo of them -> cannot beat incumbent | m30_xpair_featimp_result.json |
| I-ipca | improve | IPCA time-varying USD-loadings | instrumented betas, regime-adaptive | — | both | med | **SUBSUMED (Tier-1 FI)** | — | — | — | cross-pair factor channel = 2.0% gain; IPCA refines that channel; 2025-inversion rationale doesn't bind at 30m (2025 is the STRONG fwd year .62/.56) | m30_xpair_featimp_result.json |
| SPEC | pipeline-c | meta-labeler UP/DOWN specialist, nested-refit CPCV | per-fold primary+meta refit | m30_spec_cpcv.py | both | low | **done** | — | .5501 (CERT but SUBSUMED, <.5588) | .5203 (NOT cert, 40% clear — COLLAPSES) | **SUBSUMED** — meta-gate/subset training kills ranking (DOWN collapses, same as 2m). Symmetric book wins both sides | m30_spec_cpcv_result.json |
| TI-sub | improve | ACI/calibration gate · GMADL loss · DL-stack · Optuna(worst-VAL-half) | Tier-I remainder | — | both | low | **SUBSUMED (Tier-1)** | — | — | — | THEORY law: loss/label/gate re-eng can't beat gated cross-pair sign; 5m ACI nested-refit KILLED; 10m N18 GMADL KILLED; DL info-bound (D1-D6); 30m magweight HURTS + FI pooling-not-factors | docs/THEORY.md §2 |
| I1 | improve | cross-pair re-gate (search gate TF/cov on worst-VAL-half) | if K2 borderline | m30_xpair_regate.py (retarget) | best side | med | pending | — | — | — | — | — |
| I2 | improve | seed-ensemble ⊕ GBM on cross-pair | n-seed bag | m30_seedens_cpcv.py (retarget m5) | both | med | pending | — | — | — | — | — |
| I3 | improve | |ret|-weighted / GMADL loss on cross-pair | sample-weight by |ret| | m30_magweight_cpcv.py (retarget) | both | med | pending | — | — | — | magnitude→direction bridge | — |
| I4 | improve | calibration + ACI adaptive-conformal gate | on best book | m30_aci_cpcv.py (retarget m5) | both | med | pending | — | — | — | — | — |
| I5 | improve | specialist (UP-only / DOWN-only refit) | per-side subset train | m30_sidepipe.py (retarget m15) | weaker side | low | pending | — | — | — | subset-training usually hurts ranking | — |
| I6 | improve | Optuna(worst-VAL-half) on cross-pair LGB | tuned hyperparams | m30_optuna (new) | both | low | pending | — | — | — | — | — |
| D* | discover | corpus + arXiv 30m direction levers | mechanism-first, sign-carrying | (per backlog) | both | — | pending | — | — | — | loop until 2 dry rounds | — |

## 2026-06-08 neural+spectral sweep (D* discover round — corpus-mined path/spectral levers, 30m)
Three "technically-uncoded" lever families (mined from external directional-prediction + n-hits repos) RUN at 30m,
single-pair AND cross-pair input, GPU/AMP, deriv-faithful (TRAIN 2012-21 / VAL 22-23 / held-out per-year 2024/25/26,
selacc@cov0.10, ties-dropped, moved-bars, BE 0.541). Pre-reg KILL: VAL dirAUC<=0.515 OR no held-out year CI95-lo>=0.541.
ALL THREE KILLED at 30m (consistent with the 84-arm sweep: 0 survivors across all 6 horizons). No leader unseated.
Mechanism: SIGN-INVARIANCE thm — path/spectral forecasters carry move SIZE, not SIGN.
| id | family | method | variant | script | target | prior | status | combined | UP | DOWN | verdict | result_json |
|----|--------|--------|---------|--------|--------|-------|--------|----------|----|----|---------|-------------|
| NS1 | discover | FAM1 N-BEATS / N-HiTS path-forecast->sign | single+cross-pair, MX_HOR=30 | nbeats_nhits_dir.py | both | low | **KILLED** | — | — | — | valAUC<=.5055; best held-out selacc .5123 (2026); no year CI-lo>=.541. Path forecast = magnitude, sign-invariant | nbeats_nhits_dir_30m_result.json |
| NS2 | discover | FAM2 DLinear / Autoformer / FEDformer-freq / TFT-quantile-fan | single+cross-pair, MX_HOR=30 | decomp_dir.py | both | low | **KILLED** | — | — | — | best .5275 (tft_quantile, 2026, CI-lo .5186 < .541); decomposition carries size not sign | decomp_dir_30m_result.json |
| NS3 | discover | FAM3 causal DWT + SSA band-split->GBM literal recombine->sign | single+cross-pair, MX_HOR=30 | spectral_dir.py | both | low | **KILLED** | — | — | — | best .5162 (2024, CI-lo .4965 < .541); spectral bands carry move size, not direction | spectral_dir_30m_result.json |

## RUN ORDER / NOTES
- ONE heavy job at a time (OOM history). K1 -> K2 serialized. Improvement levers only on whichever side(s) are
  uncertified/binding after K2 (the certified ones become the incumbent to beat).
- Incumbent to beat = the BEST COMBO found (cross-pair book if K2 certifies), NOT the old combined m30.v1 GBM.
- Each improvement row: pre-registered falsifier = beat the incumbent side's binding-path p10 OR raise frac-paths-clear.
