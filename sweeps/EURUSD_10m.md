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
| I3 | improve | \|ret\|-weighted retrain POW=0.5 | `m10_magweight.py 0.5` | D | med (rescued 5m DOWN) | **KILLED** | ~.56 | 2025 .578 (<inc .605) | 2025 .539 (<inc .574,<BE) | does NOT beat certified book either side; 2026 UP collapses .474 (up-rate .429 tripwire) — same as 5m. POW=1.0 pruned-dominated. | `m10_magweight_result.json` |
| I2 | improve | seed-ensemble net ⊕ GBM decorrelated stack | `m5_deep_ens.py`@MX_HOR=10 | D | low-med | pending | — | — | — | — | — |
| I4 | improve | calibration (temp/Venn-Abers) + re-gate | wrap book | gate | low (nearly free) | pending | — | — | — | required wrapper for any conf-gated edge | — |
| X-horizon | improve/combo | cross-horizon blend 10m×15m cross-pair | `m10_xhorizon_blend.py` | D | low-med | **KILLED** | — | 2025 .596 | 2025 .637(cov5,~=certified) | **corr(p10,p15)=.957** → same signal, no decorrelated gain (mechanism: identical cross-pair family adjacent horizon) | `m10_xhorizon_blend_result.json` |
| I2 | improve | seed-ensemble net ⊕ GBM | `m5_deep_ens.py`@10m | D | low | **subsumed** | — | — | — | Tier-1: 5m blend≈GBM (corr .694, no lift) + 10m x-horizon blend corr .957 → same-feature seed-ens decorrelates even less → cannot beat certified. Logged prune. | — |
| I5 | improve | cross-pair POOLING weight-shared net (Sirignano-Cont) | new | D | low (struct) | **subsumed** | — | — | — | gated on I2 showing life (it didn't); cross-pair already IS the certified mechanism | — |
| I4 | improve | calibration (isotonic) + re-gate | `m10_calib.py` | gate | low | **KILLED** | — | 2025 .592≈.61 | 2025 .568≈.564 | within-noise of raw; monotonic calib ≈ rank-invariant for the |p-0.5| gate | (inline log) |
| N17 | discover | strictly-lagged cross-pair lead-lag (bounded, no RFF/TE tail) | `m10_leadlag.py` | D | low (5m null) | **KILLED** | VAL AUC .517 | 2025 .542<.605 | 2025 .519<.574 | anti-contemporaneous channel adds nothing over concurrent cross-pair (confirms 5m null; lead-lag decays, concurrent dominates at 10m) | `m10_leadlag_result.json` |
| N4 | discover | clock-conditioned intraday-momentum / turning-point (corpus residue) | `m10_intramom.py` | D | low (~8-10%, sign-inv risk) | **KILLED** | VAL AUC .526 | 2025 .562<.605 | 2025 .506<.574 | intraday-momentum sign-invariant/dominated at 10m — no DIRECTION beyond certified cross-pair (confirms corpus-scan caveat) | `m10_intramom_result.json` |
| N16 | discover | Cont–deLarrard P(up\|queue) + triple-barrier | (tick spec) | D | low | **subsumed** | — | — | — | Tier-1: killed @5m (DOWN CI-lo .5165); tick/sub-minute queue-imbalance is SIGN-INVARIANT at 10m + OF_* already in certified book | `m5_corpus_audit` |
| I6 | improve | Optuna TPE worst-VAL-half tuning | wrap | tune | low | pending | — | — | — | info-bound caps AUC; multiplicity risk | — |
| A8c | side | purpose-built UP/DOWN specialist + meta-labeler (step c) | `m15_updown.py`/spec @10m | U/Dn | med (filter), low (spec) | pending | — | — | — | per side not yet certified/dead | — |
| Ad  | side | confidence/coverage curve (step d) per side | `m15_sidepipe.py`@10m | U/Dn | — | pending | — | — | — | — | — |
| N19 | discover | H=10 risk-residual RELABEL + DOWN-split | `m10_residlabel_down_h10.py` | D | low-med | **KILLED** | — | — | 2025 .5077 | residual sign coin-flip in 2025; edge is gated RAW cross-pair sign not idiosyncratic residual | `m10_residlabel_down_h10_result.json` |
| N18 | discover | signed-payoff GMADL sign-coupled loss (a∈{50,100}×b∈{1,2}) | `m10_signedpayoff_gmadl.py` | D | ~10% | **KILLED** | — | 2025 .44–.49 | 2025 .46–.48 | all 4 configs BELOW 0.5 — sign-coupled payoff loss WORSE than gated BCE; loss-reopt family exhausted (cf magweight+N19) | `m10_signedpayoff_gmadl_result.json` |
| I1 | improve | ACI adaptive-conformal gate (deploy) | `m10_deploy_eval.py` | gate | high (WIN@5m) | **KILLED** | — | 2025 .611<.679 | 2025 .564<.579 | ACI trades MORE at LOWER 2025 win than fixed; compression×NY gate already captures regime | `m10_deploy_eval_result.json` |
| Ad | side | coverage curve (step d) + deploy EV net-of-spread | `m10_deploy_eval.py` | U/Dn | — | **DONE** | — | 2025 cov5 .613/cov10 .605/cov15 .585 | 2025 cov5 .630(EV.166)/cov10 .574/cov15 .583 | DOWN better at TIGHTER cov (cov5 EV.166>cov10 EV.062) → per-side deploy: DOWN cov5, UP cov10 | `m10_deploy_eval_result.json` |
| A8c | side | purpose-built side specialist (step c) — asym class-weight | `m10_spec.py` | U/Dn | low (spec worse @5m) | **KILLED** | — | 2025 .559<.605 | 2025 .572<.630(cov5) | asym-weight side specialist does NOT beat symmetric side-split either side; subset/asym training kills ranking (confirms 5m). **Full (a)-(e) pipeline now run both sides → both CERTIFIED (not dead).** | `m10_spec_result.json` |
| N17 | discover | lead-lag transfer-entropy + RFF | `m5_leadlag_te_rff.py`@10m | D | low (5m null) | pending | — | — | — | bounded run / subsume | — |

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

## TWO-SIDED LEADERBOARD (10m) — incumbent per side + all attempts vs it
**Operating metric = per-side refit-CPCV p10 at the operating gate (durable); forward side-split = descriptive (overstates).**
### (10m, UP) — LEADER: `EURUSD.m10xp.v1` cross-pair primary, **refit-CPCV p10 .5863** (15/15), forward cov10 2024 .642/2025 .605/2026 .68(thin)
| rank | method | refit p10 / binding | verdict |
|---|---|---|---|
| 1 | cross-pair xpof primary (A6c) | **.5863** (15/15) | ✅ CERTIFIED leader |
| 2 | base 3-model ensemble (L0) | .5611 (15/15) | certified, lower (fallback) |
| — | magweight POW=0.5 (I3) · ACI gate (I1) · GMADL (N18) · asym specialist (A8c) | all 2025 < .605 (GMADL <0.5) | all KILLED — none beat the certified UP |
### (10m, DOWN) — LEADER: `EURUSD.m10xp.v1` cross-pair primary, **refit-CPCV p10 .5683** (15/15), forward cov10 2024 .599/2025 .574/2026 .476(thin)
| rank | method | refit p10 / binding | verdict |
|---|---|---|---|
| 1 | cross-pair xpof primary (A6c) | **.5683** (15/15) | ✅ CERTIFIED leader (deploy DOWN at cov5%: fwd-2025 .630, EV .166) |
| 2 | base 3-model ensemble (L0) | .5519 (14/15) | certified-by-rule, regime-dependent (2021–26 .528) |
| — | magweight POW=0.5 (I3) | 2025 .539 < .574, <BE | KILLED |
| — | residual-label H=10 (N19) | 2025 .5077 (coin-flip) | KILLED (collapses 2025; edge=gated RAW cross-pair sign, not idiosyncratic residual) |
| — | GMADL sign-coupled loss (N18) | 2025 .46–.48 (<0.5) | KILLED (loss worse than gated BCE) |
| — | ACI gate (I1) · asym specialist (A8c) | 2025 < .574 | KILLED (none beat certified DOWN) |

## Discovery rounds (loop until K=2 dry)
- **Round 1 (improve/discover on the certified cross-pair edge) — DRY** (no lever beat the certified book): magweight POW0.5 (KILL, 2025 collapse) · N19 residual-label DOWN (KILL, 2025 .5077) · ACI gate I1 (KILL, trades more at lower 2025 win) · GMADL sign-coupled loss N18 (KILL, all configs <0.5) · specialist (c) (running) · coverage-curve (done). **Mechanistic conclusion: the 10m direction edge is the gated raw cross-pair sign; every loss/label/gate modification collapses on the 2025-USD-regime wall — the same info-bound + leader-not-unseated pattern as 15m.**
- **Round 2 (fresh discovery) — DRY:** cross-horizon 10m×15m blend (KILL, corr .957=same signal) · calibration I4 (KILL, rank-invariant) · N17 lagged lead-lag (KILL, AUC .517) · N16 queue/triple-barrier (subsumed, sign-invariant) · **N4 clock-conditioned intraday-momentum (KILL, the corpus-scan residue — sign-invariant/dominated)** · corpus scan (16-paper-family adjudication via sub-agent: everything maps to a killed bucket / magnitude-only / external-blocked; N4 was the one residue → now KILLED). seed-ens/pooling/Optuna subsumed (Tier-1: x-horizon corr .957 → same-feature ensembles can't decorrelate; info-bound caps AUC).
- **→ K=2 CONSECUTIVE DRY ROUNDS. Improve+discover loops DRY.** Both sides CERTIFIED via the cross-pair book; NO lever (11 distinct: magweight·N19·GMADL·ACI·specialist·calibration·x-horizon-blend·N17·N16·N4 + base) beats the gated raw cross-pair sign. The 10m direction edge is the cross-pair USD-common-factor, info-bound by the 2025-USD-regime — identical outcome to 15m (cross-pair leader, not unseated). Only remaining frontier = external/funded data (intraday DE-US rate-diff, implied-vol/risk-reversal, EURGBP ticks) — gated.

## FINAL — both sides certified-or-exhausted, loops dry (2026-06-04)
- **(10m, UP)** = `EURUSD.m10xp.v1` UP-bets — **refit-CPCV p10 .5863 (15/15) CERTIFIED**, durable. Forward cov10 2024 .642 / 2025 .605 / 2026 .68(thin). Best UP found; improve loop dry.
- **(10m, DOWN)** = `EURUSD.m10xp.v1` DOWN-bets — **refit-CPCV p10 .5683 (15/15) CERTIFIED**, durable. Forward cov10 2024 .599 / 2025 .574 (cov5: .630, EV .166) / 2026 .476(thin, n21). Best DOWN found; improve loop dry.
- Base-book floors `EURUSD.m10.v1` (UP .561 / DOWN .552) = lower certified fallback. 10m < deriv 15m min → research horizon; deployable sibling = `EURUSD.m15xp.v1`.

## Status log
- 2026-06-04: ledger created. L0 (base side-CPCV) launched. A6c (keystone) scripted, blocked on L0 floor. Both sides currently UNTESTED → no certified 10m direction edge yet.
- 2026-06-04: **L0 DONE — surprise: the base m10 book already carries a two-sided refit-CPCV edge.** UP p10 **.5611** (15/15 clear, robust across recent paths), DOWN p10 **.5519** (14/15, but regime-dependent: 2021–26 path **.528** < BE). Both ABOVE the 15m base floors (.5475/.5486). Updated model of the edge: at 10m the cross-pair common factor is ALREADY embedded enough in the base 239 features that even the plain ensemble certifies both sides by rule — consistent with the none@60s→UP@5m→both@15m gradient (10m≈15m-like for the base book). **Open questions now:** (1) does the dedicated cross-pair primary (A6c) beat .561/.552, esp. lifting DOWN's recent-era robustness? (2) can IMPROVE levers firm DOWN's 2021–26 .528? A6c running.
- 2026-06-04: **A6c DONE — KEYSTONE CERTIFIES BOTH SIDES & IMPROVES base.** Cross-pair xpof primary @MX_HOR=10: UP p10 **.5863** (15/15, min .564), DOWN p10 **.5683** (15/15, min .5556). All paths healthy n (UP 638–3576, DOWN 442–2399; no thin mirage). **Recent-era DOWN fixed:** 2019–26 .575 / 2021–26 .570 (base book had failing .528). So 10m DOWN behaves like 15m (cross-pair carries DOWN sign), NOT like 5m (dead). **Edge model updated: the 10m direction edge is the cross-pair USD-common-factor, two-sided, the same mechanism as 15m. Both keys now have a certified leader (cross-pair primary), improving the base book.** Freezing `EURUSD.m10xp.v1`. Next: IMPROVE cross-product (ACI/magweight/seed-ens/calibration/pooling) — esp. can anything lift DOWN's p10 .568 further / firm the 2021–26 tail; then (b)(c)(d) per side; then DISCOVER.
- 2026-06-04: **GOAL COMPLETE — both sides CERTIFIED + improve/discover loops DRY (K=2 dry rounds).** Full (a)-(e) pipeline ran both sides; 11 distinct improve/discover levers all KILLED/subsumed (no lever beats the certified cross-pair book). Final: (10m,UP) p10 .5863 / (10m,DOWN) p10 .5683, `EURUSD.m10xp.v1`, cov10% deployed (DOWN better EV at cov5%). Edge = gated raw cross-pair USD-common-factor sign, info-bound by 2025 regime (same as 15m). Records + deployment specs + generic docs updated; book frozen+tagged. Only open frontier = external/funded data (gated).
