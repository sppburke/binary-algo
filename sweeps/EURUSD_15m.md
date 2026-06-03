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
| I3 | improve | \|return\|-weighted / GMADL retrain (POW=0.5) — 5m DOWN-rescue mechanism, RE-TEST @15m | `m15_magweight.py` | D | **high@DOWN** | pending | | | | |
| I1 | improve | Adaptive-conformal (ACI) gate on the m15 book — 5m WIN lever | `m15_conformal.py` | gate | **high** | pending | | | | |
| A2 | gate | compression×session×coverage gate re-sweep (worst-VAL-half, fix VAL-acc-max trap) | `m15_gate_sweep.py` | G+D | med | pending | | | | |
| I2 | improve | seed-ensemble MLP ⊕ GBM decorrelated stack | `m15_deep_ens.py` | D | med | pending | | | | |
| I4 | improve | calibration (temp/Venn-Abers) + re-derived selective gate | wrap | gate | med | pending | | | | |
| I6 | improve | Optuna TPE on worst-VAL-half (logged multiplicity) | wrap | tune | low | pending | | | | |
| A5 | x-horizon | cross-horizon stack: 30m/10m parent → 15m front-load | `m15_stack.py` | D | med | pending | | | | |
| N2 | discovered | triangular USD-canceling residual (EUR-vs-GBP) — top-prior, retarget @15m | `m15_triresid.py` | D | ~15% | pending | | | | |
| N16 | discovered | redefined TRAIN label (triple-barrier/trend-scan/jump-filter) — RE-TEST @15m (more trend) | `m15_labels.py` | D | ~15% | pending | | | | |
| N18 | discovered | sign-coupled payoff objective (GMADL/RRL diff-Sharpe head) | `m15_signedpayoff.py` | D | ~10% | pending | | | | |
| Q1 | backlog | magnitude×direction gate on the 15m book (`sweeps/EURUSD_15m_backlog.md`) | `m15_magdir.py` | D | ~10% | pending | | | | |

## TIER B–F COVERAGE PASS (run once each @15m, fast-KILL; nulls are horizon-specific — confirm, don't assume)
Most are null at 60s/5m and corpus-audit-subsumed for 5m, but 15m regime composition differs (more trend, less
microstructure noise). Fast-KILL falsifier: KILL unless VAL dirAUC>0.515 AND some held-out year moved-acc CI-lo>.541.
| id | family | method | script | prior | status | note |
|----|--------|--------|--------|-------|--------|------|
| B-pass | microstructure/OFI | B3 CKS-OFI, B4 x-OFI, B5 per-side flow — at 15m (decay prior says dead) | min1_* (MX_HOR=15) | ~null | pending | tick→15m decay; expect dead |
| C-pass | state-space | C1 HMM-gate, C2 Kalman, C3 RMT, C4 CCM @15m | min1_* | ~null | pending | sign-invariance: likely magnitude |
| D-pass | seq/deep | D1 CNN/GRU, D2 NeuralCDE, D3 TabNet @15m | exp_seq/m_cnn | ~null | pending | info-bound caps AUC |
| E-pass | magnitude | E1 |ret|≥Q @15m → `MAGNITUDE_FINDINGS.md` (sign-invariant, no UP/DOWN key) | m*_magnitude | high(mag) | pending | size edge, not direction |
| F-pass | exog | F1 news, F2 price-action rules, F3 ES/NQ lead-lag, F4 resid-target @15m | m5_*/m30_* | ~null | pending | external mostly |

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
- Round 2 (arXiv/SSRN fresh + novel combinations) queued after the on-disk rows run. Stop after K=2 dry rounds.

### Tier-N detail (15m discovery rows)
| id | lever | variants to run | prior | falsifier |
|----|-------|-----------------|-------|-----------|
| N20 | cross-pair cross-sectional rank/factor | learning-to-rank (LambdaMART) on 7-major fwd-15m rank · PC-shrinkage residual · IPCA latent | ~.16 | KILL unless binding-2025 side win-rate CI-lo > frozen side incumbent |
| I3+ | loss/labeling retrain (consolidated) | MADL · GMADL(a,b) · \|ret\|^p p∈{.5,1,2} sample-weight · triple-barrier(λ) 3-class · deadband-tertile | ~.16 | KILL unless DOWN binding-2025 CI-lo>.541 OR UP beats frozen, AND 2026 not collapse, up-rate∈[.47,.53] |

## ★ MILESTONE 2026-06-03: BOTH SIDES CERTIFIED (refit-CPCV). Honest floors UP .5475 / DOWN .5486.
The side-pipeline (a)-(e) is COMPLETE for both sides. Integrity: per-side combined reproduces the independent
min15_cpcv (mean .5787, p10 .557) → harness faithful. **15m is the first EURUSD horizon with BOTH UP and DOWN
certified, and the only deriv-deployable one.** DOWN marginally MORE robust (frac-clear 1.0 vs UP .933). Forward
operating point (.607/.559) was OPTIMISTIC selection — durable figure = the refit p10 floor (.5475 / .5486).
Margins are THIN (~+0.6-0.7 pt over breakeven .541); the IMPROVE phase tries to widen them.

## NEXT (improve, don't stop at first cert): incumbent to beat = UP refit-p10 .5475 / DOWN .5486 under SAME refit-CPCV.
Cheap retrains first (1 fit each, ~6-10 min), CPCV only survivors: I3 magweight(POW 0/0.5/1) → relabel family
(N16 triple-barrier/3-class) → N20 cross-pair rank → gate worst-VAL-half (A2) → seed-ens (I2). Then B-F fast-KILL
pass + discovery round 2. FREEZE the certified side books now (below).
