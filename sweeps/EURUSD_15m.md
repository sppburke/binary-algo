SCOPE: KEY-SPECIFIC — EURUSD · 15m sweep LEDGER (status/state). Generic menu: `SWEEP_MATRIX.md`. Results of
record: `EURUSD_RESULTS.md`. Backlog/queue: `sweeps/EURUSD_15m_backlog.md`. Ideas: `IDEAS_LOG.md`.

# EURUSD · 15m — EXHAUSTIVE SWEEP LEDGER
Goal: best (15m,UP) and best (15m,DOWN) EURUSD binary-direction predictor, certified-or-honestly-exhausted.
15m is the **deriv-tradeable** binary (forex Rise/Fall min duration = 15m). Breakeven 0.541 (R≈0.85).
Incumbent COMBINED book = `EURUSD.m15.v1` (CPCV-faithful 0.579, p10 0.557; recent 0.647). Incumbent per-side
= the side-split below (UP binding .607 / DOWN binding .559, both NOT yet certified).

## MODEL OF THE EDGE (updated each iteration — THE ENGINE; read before designing the next row)
- **2026-06-03 (side-split a):** 15m edge is **confidence-driven, not side-driven**. Gate-only (comp×NY, no conf
  selection) ~.50–.54 both sides; the entire lift comes from the top-~2% confidence selection. The selected
  edge **rotates by regime** (gate-only UP strong 2024-25 / dead 2026 .500; DOWN weak 2024-25 / strong 2026
  .566). Combined book captures whichever side is right per regime → a *fixed* side-only filter fights a
  rotating edge. Both sides clear breakeven in POINT estimate at the operating point (UP .725/.607/.674; DOWN
  .653/.559/.650); binding year 2025 for both; UP>.DOWN. The open question is **CPCV power at thin coverage**,
  not edge-absence. → Next rows: coverage curve (find operating point maximizing binding-year CI-lo) + full-refit
  CPCV per side; then |ret|-weight (5m DOWN rescue mechanism) and ACI gate (5m WIN lever).
- The book selected its gate via **VAL-acc-max** (m15_production.train: "maximizing VAL accuracy") — the trap the
  skill warns against (corr(VAL,OOS)=−0.54). A worst-VAL-half re-selection of the gate is a free improvement lever.
- **2026-06-03 (coverage curve d, `m15_sidepipe_result.json`):** Swept conf threshold (cov 100%→2%) with VAL
  thresholds. **Monotone confidence→accuracy on BOTH sides** (UP 2024 .528→.725, 2025 .543→.607; DOWN 2024
  .521→.653) — the signature of a genuine but THIN edge (noise wouldn't sort by confidence). BUT binding-year
  single-bootstrap CI-lo **never clears .541** at any coverage (best UP .513@cov2%, DOWN .506@cov50%): tight cov →
  strong point est but n≤120/yr/side → wide CI; loose cov → n large but point decays to ~.53. This is a POWER
  limit of single-year bootstrap on a HALF-sized sample, exactly why the COMBINED book (full n) certified (p10
  .557) while its halves don't. → The powered arbiter is the per-side **refit CPCV** (pools across 15 paths),
  NOT single-year CI-lo. Binding year = 2025 for UP; for DOWN the weak year rotates (2025 weakest at tight cov).
  Mechanism update: the edge is a thin, regime-rotating, confidence-concentrated directional signal; both sides
  share it; certification hinges on whether per-side CPCV p10 clears with the n available.
- **2026-06-03 (CROSS-PAIR fwd-POSITIVE — the first improvement lever to beat the book):** the m5xp cross-pair
  USD-residual+OF primary, retargeted MX_HOR=15 (`m15_xpair_blend.py`), beats the frozen book on the BINDING year
  forward, BOTH sides: xpair 2025 UP .570/DOWN .591 vs book .541/.517; blend 2025 UP .581/DOWN .568. Corr(book,
  xpair)=0.83 (high) yet xpair's operating point is better. MECHANISM: 2025 was binding BECAUSE of a USD-factor
  regime shift; the cross-pair model directly captures the USD factor → robust exactly there. This is mechanism-
  backed, not forward noise → escalated to per-side refit CPCV (A6c). If it certifies > book floor, it's the new
  (15m) deliverable / a blend book. NOTE: a frozen-book+frozen-xpair BLEND is a legit DEPLOYMENT combo (both
  frozen on 2012-21, evaluated forward) even if only the standalone xpair gets the per-fold-refit cert.
- **2026-06-03 (improve round 2 NULL):** N16 3-class relabel + I2 seed-ensemble both fail to beat the lgb control
  on binding 2025 (~.52-.53). I4 calibration subsumed (monotonic ⇒ same selection). Binding-2025 wall holds for
  same-feature levers; only the cross-pair (NEW features) breaks it.
- **2026-06-03 (CERT + I3 magweight KILLED):** Per-side refit CPCV CERTIFIED both sides (UP p10 .5475 / DOWN .5486).
  `|ret|`-weight retrain (I3, POW=0.5) does NOT improve 15m — fwd binding-2025 UP .562 (<frozen .607), DOWN **.513
  (<breakeven)**; VAL AUC dropped .528→.522. The 5m DOWN-rescue mechanism (up-weight large moves) is WRONG for 15m:
  15m DOWN already lives on diffusive/trend moves; up-weighting large (jump-driven, sign-unpredictable) moves
  degrades it. CONFIRMS the horizon-transfer reasoning. **Methodological note:** reduced-fidelity (600/700/700)
  forward retrains at thin auto-cov2% are NOISE-DOMINATED (POW=0 control hit n=11-36 thin pockets) — improve levers
  must report at FIXED cov5%/10% with a matched-fidelity control; only forward-winners get the expensive CPCV. The
  certified frozen full-fidelity book is a strong incumbent that reduced-fidelity retrains start behind.

## LEDGER (ROI order; work top→bottom; never repeat a `done` row)
| id | family | method / variant | script | tgt | prior | status | combined | UP | DOWN | result_json |
|----|--------|------------------|--------|-----|-------|--------|----------|----|----|-------------|
| A8a | side-split | FILTER: split frozen m15 book by predicted side (operating + gate-only) | `m15_updown.py` | U/Dn | — | **done** | repro .689/.582/.663 | binding .607 CI[.513,.692] | binding .559 CI[.472,.646] | `m15_updown_result.json` |
| A8b-up | side-pipeline | (15m,UP) coverage curve (d) — VAL-worst-half threshold sweep | `m15_sidepipe.py` | U | high | **done** | — | binding-CI-lo best .513 @cov2% (never ≥.541) | — | `m15_sidepipe_result.json` |
| A8b-dn | side-pipeline | (15m,DOWN) coverage curve (d) | `m15_sidepipe.py` | Dn | high | **done** | — | — | binding-CI-lo best .506 @cov50% (never ≥.541) | `m15_sidepipe_result.json` |
| A8c-up | side-pipeline | (15m,UP) full-refit CPCV (e) — 15 purged paths, per-side split | `m15_cpcv_side.py` | U | high | **done ✅ CERTIFIED** | — | **p10 .5475** mean .5771 frac .933 | — | `m15_cpcv_side_result.json` |
| A8c-dn | side-pipeline | (15m,DOWN) full-refit CPCV (e) | `m15_cpcv_side.py` | Dn | high | **done ✅ CERTIFIED** | — | — | **p10 .5486** mean .5805 frac 1.0 | `m15_cpcv_side_result.json` |
| A8d-up | specialist | (15m,UP) purpose-built specialist (subset/weighted retrain MX_HOR=15) | `m15_spec.py` | U | med | pending | | | | |
| A8d-dn | specialist | (15m,DOWN) specialist | `m15_spec.py` | Dn | med | pending | | | | |
| I3 | improve | \|return\|-weighted retrain (POW=0.5) — 5m DOWN-rescue, RE-TEST @15m | `m15_magweight.py` | D | **high@DOWN** | **killed** | .54/.51/.57 | fwd 2025 .562<frozen .607 | fwd 2025 **.513<BE** | `m15_magweight_result_pow0.5.json` |
| I1 | improve | Adaptive-conformal (ACI) gate on the m15 book — 5m WIN lever | `m15_conformal.py` | gate | **high** | pending | | | | |
| A2 | gate | compression×session×coverage gate re-sweep (worst-VAL-half, fix VAL-acc-max trap) | `m15_gate_sweep.py` | G+D | med | pending | | | | |
| I2 | improve | seed-ensemble K=3 lgb avg (variance reduction) | `m15_improve.py` | D | med | **killed** | — | 2025 .520 ≤ ctrl .526 | 2025 .519 ≤ ctrl .534 | `m15_improve_result.json` |
| N16 | improve | 3-class up/flat/down deadband relabel (trend hypo) | `m15_improve.py` | D | ~.15 | **killed** | — | 2025 .533 ≈ ctrl | 2025 .530 ≤ ctrl .534 | `m15_improve_result.json` |
| I4 | improve | temperature/Platt calibration + selective gate | wrap | gate | low | **subsumed** | — | — | — | monotonic transform ⇒ preserves \|p−0.5\| ranking ⇒ SAME cov-gated selection ⇒ same acc (Tier-1 logic) |
| I4 | improve | calibration (temp/Venn-Abers) + re-derived selective gate | wrap | gate | med | pending | | | | |
| I6 | improve | Optuna TPE on worst-VAL-half (logged multiplicity) | wrap | tune | low | pending | | | | |
| A5 | x-horizon | cross-horizon stack: 30m/10m parent → 15m front-load | `m15_stack.py` | D | med | pending | | | | |
| A6 | x-pair | cross-pair USD-residual+OF PRIMARY (m5xp source) retargeted MX_HOR=15 + blend w/ book | `m15_xpair_blend.py` | D | **high** | **fwd-positive → CPCV** | — | xpair fwd 2025 .570 > book .541 | xpair fwd 2025 .591 > book .517 | `m15_xpair_blend_result.json` |
| A6c | x-pair | (A6) per-side full-refit CPCV — certify the cross-pair improvement | `m15_xpair_cpcv.py` | D | high | **done ✅ CERTIFIED+IMPROVES** | mean .5804 p10 .5687 | **p10 .5673** (15/15, vs book .5475) | **p10 .5742** (15/15, vs book .5486) | `m15_xpair_cpcv_result.json` |
| A6f | x-pair | FREEZE the cross-pair 15m book → `EURUSD.m15xp.v1` (cov10% deploy gate) | `m15_xpair_freeze.py`+`m15_xpair_regate.py` | D | **done (frozen)** | fwd cov10 .605/.546/.543 | fwd .602/.561/.539 | fwd .610/.531/.552 | `m15_xpair_freeze_result.json` |
| N2 | discovered | triangular USD-canceling residual (EUR-vs-GBP) — top-prior, retarget @15m | `m15_triresid.py` | D | ~15% | pending | | | | |
| N16 | discovered | redefined TRAIN label (triple-barrier/trend-scan/jump-filter) — RE-TEST @15m (more trend) | `m15_labels.py` | D | ~15% | pending | | | | |
| N18 | discovered | sign-coupled payoff objective (GMADL/RRL diff-Sharpe head) | `m15_signedpayoff.py` | D | ~10% | pending | | | | |
| Q1 | backlog | magnitude×direction gate on the 15m book (`sweeps/EURUSD_15m_backlog.md`) | `m15_magdir.py` | D | ~10% | pending | | | | |

## TIER B–F COVERAGE PASS (run once each @15m, fast-KILL; nulls are horizon-specific — confirm, don't assume)
Most are null at 60s/5m and corpus-audit-subsumed for 5m, but 15m regime composition differs (more trend, less
microstructure noise). Fast-KILL falsifier: KILL unless VAL dirAUC>0.515 AND some held-out year moved-acc CI-lo>.541.
| id | family | method | script | prior | status | note |
|----|--------|--------|--------|-------|--------|------|
| B-pass | microstructure/OFI | B3 CKS-OFI, B4 x-OFI, B5 per-side flow @15m | min1_* (MX_HOR=15) | ~null | **subsumed** | Tier-1: microstructure DECAYS monotonically with H — null already by 60s across ~24 channels (xOFI VAL .5015, CCM slope .0066, whale/queue/Cont-deLarrard all coin-flip @5m); at 15m the tick-scale signal is gone a fortiori. Sign-invariance: most OFI gates MAGNITUDE. Running = confirm-the-null, near-zero ROI. |
| C-pass | state-space | C1 HMM-gate, C2 Kalman, C3 RMT, C4 CCM @15m | min1_* | ~null | **subsumed** | Tier-1: HMM/Kalman/RMT/CCM all null @60s; sign-invariance theorem (arXiv:2512.15720) — these gate SIZE not SIGN. The base book ALREADY uses a vol-regime gate (comp×NY); an HMM regime gate is a variant of that, no new direction info. |
| D-pass | seq/deep | D1 CNN/GRU, D2 NeuralCDE, D3 TabNet @15m | exp_seq/m_cnn | ~null | **subsumed** | Tier-1: info-bound caps raw AUC at ALL horizons (5m DL review: no economic gain; 15m raw AUC .528). Deep nets can't exceed the info content the GBM already extracts; D5 meta-labeler @15m already 0.615<parent. |
| E/Q1 | magnitude×dir | magnitude regime gate on 15m direction (the queued backlog Q1; the ONE distinct on-disk test) | `m15_magdir.py` | ~.10 | **pending (run)** | sign-invariance prior = null, but 15m move-composition differs from the 5m null → RUN to confirm. Magnitude itself (sign-invariant) → MAGNITUDE_FINDINGS, no UP/DOWN key. |
| F-pass | exog | F1 news, F2 price-action, F3 ES/NQ lead-lag, F4 resid-target @15m | m5_*/m30_* | ~null | **subsumed (on-disk) / →Tier-G** | Tier-1: news NULL for ≤5m FX direction (macro-calendar memory); price-action rules (RSI2/BB%b) already in the 239 base features the GBM uses; ES/NQ + macro = EXTERNAL → Tier-G frontier (backlog). |

## DISCOVERY (loop-until-dry; see task 5 + CORPUS_LEVER_INVENTORY.md)
**Round 1 done (2026-06-03):** parsed CORPUS_LEVER_INVENTORY.md (550 levers) programmatically → **106 untested
sign-aware direction/labeling levers**, ranked by prior, clustered into 3 families. Horizon-reasoned transfer:
- **Loss/labeling (HIGH @15m, horizon-agnostic objective changes on the GBM):** MADL training loss (.18),
  GMADL diff loss (.16), triple-barrier path-dependent train labels (.15), 3-class up/flat/down deadband (.13),
  squared/power-MADL (.12). → consolidated into rows **I3 / N16 / N18** (variant menu enriched below). The 5m
  |ret|-weight result ("rebalances to two-sided ~.56, collapses 2026 UP") is the SAME family — at 15m the move
  composition (more trend) differs → RE-TEST, especially for the DOWN side which is alive at 15m.
- **Cross-pair cross-sectional (MED @15m, USD-factor is slower → arguably more relevant than 5m):** Kozak-Nagel-
  Santosh PC-shrinkage (.18), IPCA latent factor (.17), cross-pair learning-to-rank RankNet/LambdaMART (.16),
  DeltaLag adaptive lead-lag (.13), FinGAT graph-attention (.13). → new row **N20** (D7 exog was null @15m but
  these are RANKING/factor constructions, not exog features; N11/N12 lambdarank killed @5m but not @15m). Needs
  7-major cross-section (on-disk).
- **Microstructure/OFI/LOB (LOW @15m — decay; null by 60s):** Cont-de Larrard, Sirignano-Cont pooled, Taranto
  OF-surprise, depth-norm OFI, SPDE. → DEPRIORITIZED but confirm-and-kill via the B-pass (one fast row), do NOT
  silently skip (nulls are horizon-specific; the point is to confirm at 15m).
**Round 2 done (2026-06-03):** mined the corpus's external-blocked + lower-prior untested direction levers.
Findings: (1) the remaining on-disk direction levers (prior≥.10) are ALL the **cross-pair/cross-sectional family**
(PC-shrinkage, IPCA, learning-to-rank, DeltaLag, FinGAT, THGNN, RFF) — represented by the cross-pair primary now
under CPCV (A6c); elaborate variants are lower-marginal-prior follow-ups (N11/N12 lambdarank already killed @5m).
(2) The genuine NEW-information frontier = **external data** (10 external-blocked levers): VIX/implied-vol risk-off,
GARCH-MIDAS equity-vol, DE-US 2y/10y rate-diff, cross-asset S&P/oil/gold leads — SEVERAL are DOWN-side gates
(risk-off USD-bid → EURUSD down). → queued as Tier-G in `sweeps/EURUSD_15m_backlog.md`, gated on user "go".
**Discovery converging: 1 live on-disk idea (cross-pair, under test); rest subsumed or external. K=2 rounds, the
only survivable on-disk idea is being certified now → loop near-dry pending A6c.**

### Tier-N detail (15m discovery rows)
| id | lever | variants to run | prior | falsifier |
|----|-------|-----------------|-------|-----------|
| N20 | cross-pair cross-sectional rank/factor | learning-to-rank (LambdaMART) on 7-major fwd-15m rank · PC-shrinkage residual · IPCA latent | ~.16 | KILL unless binding-2025 side win-rate CI-lo > frozen side incumbent |
| I3+ | loss/labeling retrain (consolidated) | MADL · GMADL(a,b) · \|ret\|^p p∈{.5,1,2} sample-weight · triple-barrier(λ) 3-class · deadband-tertile | ~.16 | KILL unless DOWN binding-2025 CI-lo>.541 OR UP beats frozen, AND 2026 not collapse, up-rate∈[.47,.53] |

## ★★ MILESTONE 2 2026-06-03: CROSS-PAIR IMPROVES BOTH SIDES (refit-CPCV). New floors UP .5673 / DOWN .5742.
The cross-pair USD-residual+OF primary (m5xp source) retargeted MX_HOR=15 CERTIFIES on the SAME refit-CPCV
harness at UP p10 **.5673** / DOWN p10 **.5742**, frac-clear **1.0 both sides** (all 15 purged paths clear) —
a +1.98pt (UP) / +2.56pt (DOWN) improvement over the base-book floors, turning the DOWN side from a thin +.8pt
edge into a solid +3.3pt one. Even the WORST cross-pair path (UP .5507/DOWN .5684) beats the book p10. MECHANISM
confirmed: modeling the USD common factor directly is robust in the USD-regime binding years that capped the base
book. **NEW LEADER both keys = the cross-pair 15m model.** The base book remains a valid (lower) certified fallback.
NEXT: freeze the cross-pair 15m book (A6f) as the deliverable; then Q1 magnitude (last on-disk test) + finalize.

**ADVERSARIAL VERIFICATION of the cross-pair (2026-06-03, `m15_xpair_freeze_result.json`):** froze `EURUSD.m15xp.v1`
and stress-tested the DEPLOYABLE frozen-2021 single model. Finding: refit-CPCV cert (.5673/.5742) is SOUND (folds
well-powered, n~1000-2000), but the frozen single-model FORWARD is marginal in binding years (@cov10%, n≥50 all
yrs: UP .602/.561/.539, DOWN .610/.531/.552; binding CI-lo<BE). The gap is explained: the cross-pair edge is
**REFIT-DEPENDENT** — the USD common factor is non-stationary, so a model frozen on 2012-2021 is staler for 2026
than the per-fold-refit CPCV models; the base book (own-MTF features) is less time-sensitive and frozen-forward-
stabler. ALSO: cov5% (worst-VAL-half's pick) is thin-2026-fragile (n<50 trap-6, seed-variance .486-.541) → deploy
gate set to cov10% (`m15_xpair_regate.py`). **Honest verdict:** cross-pair = CERTIFIED LEADER by the mandated
refit-CPCV criterion (.5673/.5742 > book .5475/.5486), BEST DOWN edge in program; DEPLOY with periodic retrain
(not frozen-forever), size on the refit floor. Base book = frozen-stable lower fallback. Vocab: refit-certified +
regime/refit-dependent (NOT "frozen-forward-positive", which it is only in 2024).

## ★ MILESTONE 2026-06-03: BOTH SIDES CERTIFIED (refit-CPCV). Honest floors UP .5475 / DOWN .5486 (base book).
The side-pipeline (a)-(e) is COMPLETE for both sides. Integrity: per-side combined reproduces the independent
min15_cpcv (mean .5787, p10 .557) → harness faithful. **15m is the first EURUSD horizon with BOTH UP and DOWN
certified, and the only deriv-deployable one.** DOWN marginally MORE robust (frac-clear 1.0 vs UP .933). Forward
operating point (.607/.559) was OPTIMISTIC selection — durable figure = the refit p10 floor (.5475 / .5486).
Margins are THIN (~+0.6-0.7 pt over breakeven .541); the IMPROVE phase tries to widen them.

## NEXT (improve, don't stop at first cert): incumbent to beat = UP refit-p10 .5475 / DOWN .5486 under SAME refit-CPCV.
Cheap retrains first (1 fit each, ~6-10 min), CPCV only survivors: I3 magweight(POW 0/0.5/1) → relabel family
(N16 triple-barrier/3-class) → N20 cross-pair rank → gate worst-VAL-half (A2) → seed-ens (I2). Then B-F fast-KILL
pass + discovery round 2. FREEZE the certified side books now (below).
