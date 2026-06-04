> **SCOPE: EURUSD × 10m** (key-specific sweep LEDGER — resumable state of the exhaustive two-sided sweep). Generic menu: `SWEEP_MATRIX.md`. Backlog/queue: `sweeps/EURUSD_10m_backlog.md`. Results of record: `EURUSD_RESULTS.md`. Methods: `METHODS_CATALOG.md`.

# EURUSD × 10m — SWEEP LEDGER

**Goal:** best (10m,UP) + best (10m,DOWN) EURUSD binary-direction predictor, OOS-verified, each certified-or-honestly-exhausted. Deriv settlement (mid-to-mid, +1s entry, ties LOSE), breakeven **0.541**. NOTE: 10m < deriv's 15m forex-Rise/Fall minimum → research/synthetic-index horizon (not natively deployable; the deployable neighbor is 15m). Certify ONLY via full per-fold-refit CPCV at the operating gate (per-side p10≥0.541 AND ≥~80% paths clear). Selection on worst-VAL-half (never VAL-acc-max; corr(VAL,OOS)=−0.54).

## Model of the edge (THE ENGINE — read before each experiment; update after each)
- **Incumbent at 10m = `EURUSD.m10.v1`** (native-10 3-model ensemble × `5m_bb_width≤q20(VAL) × sess_ny @cov10%`): COMBINED t24 .614 / t25 .579 / 2026 .594, comb .602 [.582,.621], floor .579. **Never side-split → both (10m,UP) and (10m,DOWN) keys were UNTESTED.** (`m10_freeze_honest.py`, `m10_research_log.md`.)
- **Cross-pair USD-common-factor horizon gradient (the governing prior):** none@60s → UP@5m (p10 .553) → **BOTH@15m** (.5673/.5742, 15/15). The keystone pooling lever strengthens with horizon. 10m sits between 5m and 15m → **expect 10m UP to certify; 10m DOWN is the crux** (dead@5m = down-moves jump/informed-dominated → magnitude not sign; certified@15m where mean-reversion/common-factor dominates). Mechanism: longer horizon averages out the informed-jump component of down-moves → sign becomes forecastable. The transition is somewhere in (5m,15m]; this sweep locates it.
- **2025 is the binding regime** everywhere (USD-regime). The 10m ES→EURUSD lead-lag flipped +0.02(2024)→−0.05(2025) — the mechanistic key to the 2025 wall.
- **What's already killed COMBINED at 10m** (do NOT relitigate blindly; side-split is the open part): native-GBM VAL-max .667 (artifact), honest gate sweep (oracle .599 — no gate clears .65 even cheating), cross-horizon stack .598, cross-pair *probe* .599 (but NOT the pooling+refit-CPCV pipeline — that's row A6c below), walk-forward .609, ES lead-lag null, direction-on-magnitude (mag AUC .81/.74/.71, dir flat .51–.53 → 10m direction is info-bound, the >.65 is magnitude).

## Side-completion gate (a row's SIDE is "done" only after the full pipeline ran & SHOWED the wall)
(a) side-split · (b) gate on worst-VAL-half · (c) purpose-built specialist · (d) confidence/coverage curve · (e) full-refit CPCV at operating gate. Dead = (a)-(d) ran AND the wall is shown.

## LEDGER (work top→bottom; ROI/prior order; never skip; never re-run a `done` row)

| id | family | method / variant | script | tgt | prior | status | combined | UP | DOWN | verdict | result_json |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **L0** | base | m10 base 3-model ensemble side-split refit-CPCV (step a+e) | `m10_cpcv_side.py` | U/Dn | high | **DONE** | p10 .563 (15/15) | **p10 .5611 (15/15)** | **p10 .5519 (14/15)** | base book certifies BOTH sides by rule; UP robust, DOWN regime-dependent (2021–26 path .528). FLOOR for A6c. | `m10_cpcv_side_result.json` |
| **A6c** | cross-pair | **KEYSTONE** xpof USD-residual+OF primary side refit-CPCV @MX_HOR=10 | `m10_xpair_cpcv.py` | U/Dn | **HIGH** (cert@5m-UP/15m-both; 10m between) | **DONE ✅ BOTH CERTIFIED** | p10 .579 (15/15) | **p10 .5863 (15/15)** | **p10 .5683 (15/15)** | CERTIFIES + IMPROVES base both sides; DOWN min .5556 (vs base .528) — fixes recent-era weakness. **NEW (10m,UP)+(10m,DOWN) leader.** Freeze→`EURUSD.m10xp.v1`. | `m10_xpair_cpcv_result.json` |
| I1 | improve | ACI adaptive-conformal gate on certified edge | `m5_conformal.py`@MX_HOR=10 | gate | high (WIN@5m) | pending | — | — | — | apply iff A6c certifies a side | — |
| I3 | improve | \|ret\|-weighted / GMADL loss (POW sweep) | `m5_magweight.py`@MX_HOR=10 | D | med (rescued 5m DOWN) | pending | — | — | — | esp. for DOWN if it doesn't certify unweighted | — |
| I2 | improve | seed-ensemble net ⊕ GBM decorrelated stack | `m5_deep_ens.py`@MX_HOR=10 | D | low-med | pending | — | — | — | — | — |
| I4 | improve | calibration (temp/Venn-Abers) + re-gate | wrap book | gate | low (nearly free) | pending | — | — | — | required wrapper for any conf-gated edge | — |
| I5 | improve | cross-pair POOLING weight-shared net (Sirignano-Cont) | new | D | low (struct) | pending | — | — | — | run only if I2 shows life | — |
| I6 | improve | Optuna TPE worst-VAL-half tuning | wrap | tune | low | pending | — | — | — | — | — |
| A8c | side | purpose-built UP/DOWN specialist + meta-labeler (step c) | `m15_updown.py`/spec @10m | U/Dn | med (filter), low (spec) | pending | — | — | — | per side not yet certified/dead | — |
| Ad  | side | confidence/coverage curve (step d) per side | `m15_sidepipe.py`@10m | U/Dn | — | pending | — | — | — | — | — |
| N17 | discover | lead-lag transfer-entropy + RFF virtue-of-complexity | `m5_leadlag_te_rff.py`@10m | D | ~13% | pending | — | — | — | strictly-lagged cross-pair + directed-info gate | — |
| N18 | discover | signed-payoff GMADL/MADL + RRL diff-Sharpe head | `m5_signedpayoff_torch.py`@10m | D | ~10% | pending | — | — | — | sign-coupled payoff objective | — |
| N16 | discover | Cont–deLarrard P(up\|queue) + tick-sign reversal | (per N16 spec)@10m | D | low | pending | — | — | — | — | — |
| N19 | discover | H=10 risk-residual RELABEL + DOWN-split | `m5_residlabel_*`@10m | D | low-med | pending | — | — | — | — | — |

### Already-run at 10m COMBINED (status `done`/`killed` — recorded in EURUSD_RESULTS.md, NOT re-run; side-split is the open work)
| id | method | result (combined) | verdict | source |
|---|---|---|---|---|
| A1 | native-10 3-model GBM, VAL-acc-max | .667 (oos n57) | killed (VAL-max artifact, trap #5) | `m10_production.py` |
| A2 | honest gate sweep 200 cfg | floor .549 / comb .576; oracle .599 | done (no gate clears .65 even cheating) | `m10_gate_sweep.py` |
| A5 | cross-horizon stack | comb .598, oracle floor .560 | done (~.59) | `m10_stack.py` |
| A6(probe) | cross-pair MX_HOR=10 (xp probe, NOT pooling-cpcv) | comb .599 [.579,.620] | done; XP lift weaker at 10m combined → **A6c re-tests as pooling+side-refit-CPCV** | `m5_xpair.py` |
| A7 | walk-forward retrain | t24 .668/t25 .573/oos .574/comb .609 | done (t25 +.017 only) | `m10_walkforward.py` |
| A0 | **HONEST freeze** native-10 × 5m_bb_width-NY | comb .602 [.582,.621] floor .579 | ✅ honest deliverable (combined) = incumbent | `m10_freeze_honest.py` |
| F3 | ES→EURUSD lead-lag | +0.02(2024)→−0.05(2025) | killed (null; mechanistic key to 2025 wall) | `m10_xasset_probe.py` |
| E1 | direction-on-magnitude | mag AUC .813/.741/.706; dir flat .51–.53 | done (dir null / magnitude → MAGNITUDE_FINDINGS) | `m10_magdir.py` |

## Discovery rounds (loop until K=2 dry)
- Round 0: pending — kick off after L0/A6c resolve (build on the closest result). Retarget genuinely-distinct Tier-N direction levers (N16–N19) + scan corpus for 10m-specific levers.

## Status log
- 2026-06-04: ledger created. L0 (base side-CPCV) launched. A6c (keystone) scripted, blocked on L0 floor. Both sides currently UNTESTED → no certified 10m direction edge yet.
- 2026-06-04: **L0 DONE — surprise: the base m10 book already carries a two-sided refit-CPCV edge.** UP p10 **.5611** (15/15 clear, robust across recent paths), DOWN p10 **.5519** (14/15, but regime-dependent: 2021–26 path **.528** < BE). Both ABOVE the 15m base floors (.5475/.5486). Updated model of the edge: at 10m the cross-pair common factor is ALREADY embedded enough in the base 239 features that even the plain ensemble certifies both sides by rule — consistent with the none@60s→UP@5m→both@15m gradient (10m≈15m-like for the base book). **Open questions now:** (1) does the dedicated cross-pair primary (A6c) beat .561/.552, esp. lifting DOWN's recent-era robustness? (2) can IMPROVE levers firm DOWN's 2021–26 .528? A6c running.
- 2026-06-04: **A6c DONE — KEYSTONE CERTIFIES BOTH SIDES & IMPROVES base.** Cross-pair xpof primary @MX_HOR=10: UP p10 **.5863** (15/15, min .564), DOWN p10 **.5683** (15/15, min .5556). All paths healthy n (UP 638–3576, DOWN 442–2399; no thin mirage). **Recent-era DOWN fixed:** 2019–26 .575 / 2021–26 .570 (base book had failing .528). So 10m DOWN behaves like 15m (cross-pair carries DOWN sign), NOT like 5m (dead). **Edge model updated: the 10m direction edge is the cross-pair USD-common-factor, two-sided, the same mechanism as 15m. Both keys now have a certified leader (cross-pair primary), improving the base book.** Freezing `EURUSD.m10xp.v1`. Next: IMPROVE cross-product (ACI/magweight/seed-ens/calibration/pooling) — esp. can anything lift DOWN's p10 .568 further / firm the 2021–26 tail; then (b)(c)(d) per side; then DISCOVER.
