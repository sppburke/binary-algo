# EURUSD · 30m — SWEEP LEDGER (key-specific status)
SCOPE: EURUSD · 30m. Resumable sweep ledger — the STATE of the exhaustive+generative search for the best
30m UP and best 30m DOWN EURUSD direction predictor. Generic menu = SWEEP_MATRIX.md; ideas = IDEAS_LOG.md;
executable backlog/discovery = sweeps/EURUSD_30m_backlog.md; results of record = EURUSD_RESULTS.md.
Created 2026-06-04 (the /goal 30m two-sided certification push). 30m is DERIV-TRADEABLE (longest, above 15m floor).

Discipline (strategy-eval §2): deriv-faithful wc settlement (mid-to-mid, ties LOSE, breakeven 0.541),
nonoverlap_chrono(gap=1800s), per-year CI95, moved-bars-only, worst-VAL-half selection, pre-registered falsifier,
CERTIFY only via full per-fold-refit CPCV (refit p10>=0.541 AND >=~80% paths clear).

## CONTEXT — what is already known at 30m (Tier-1, prior thread, m30_research_log.md + EURUSD_RESULTS.md)
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
| I-magw | improve | \|ret\|-weighted loss (magnitude->direction bridge) | POW=0.5, same K2 gate/CPCV | m30_magweight_cpcv.py | both | med | **pending** | — | — | — | must beat incumbent UP .5588 / DOWN .5525; else SUBSUMED | m30_magweight_cpcv_result.json |
| I1 | improve | cross-pair re-gate (search gate TF/cov on worst-VAL-half) | if K2 borderline | m30_xpair_regate.py (retarget) | best side | med | pending | — | — | — | — | — |
| I2 | improve | seed-ensemble ⊕ GBM on cross-pair | n-seed bag | m30_seedens_cpcv.py (retarget m5) | both | med | pending | — | — | — | — | — |
| I3 | improve | |ret|-weighted / GMADL loss on cross-pair | sample-weight by |ret| | m30_magweight_cpcv.py (retarget) | both | med | pending | — | — | — | magnitude→direction bridge | — |
| I4 | improve | calibration + ACI adaptive-conformal gate | on best book | m30_aci_cpcv.py (retarget m5) | both | med | pending | — | — | — | — | — |
| I5 | improve | specialist (UP-only / DOWN-only refit) | per-side subset train | m30_sidepipe.py (retarget m15) | weaker side | low | pending | — | — | — | subset-training usually hurts ranking | — |
| I6 | improve | Optuna(worst-VAL-half) on cross-pair LGB | tuned hyperparams | m30_optuna (new) | both | low | pending | — | — | — | — | — |
| D* | discover | corpus + arXiv 30m direction levers | mechanism-first, sign-carrying | (per backlog) | both | — | pending | — | — | — | loop until 2 dry rounds | — |

## RUN ORDER / NOTES
- ONE heavy job at a time (OOM history). K1 -> K2 serialized. Improvement levers only on whichever side(s) are
  uncertified/binding after K2 (the certified ones become the incumbent to beat).
- Incumbent to beat = the BEST COMBO found (cross-pair book if K2 certifies), NOT the old combined m30.v1 GBM.
- Each improvement row: pre-registered falsifier = beat the incumbent side's binding-path p10 OR raise frac-paths-clear.
