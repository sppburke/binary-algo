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

## LEDGER (ROI order; work top→bottom; never repeat a `done` row)
| id | family | method / variant | script | tgt | prior | status | combined | UP | DOWN | result_json |
|----|--------|------------------|--------|-----|-------|--------|----------|----|----|-------------|
| A8a | side-split | FILTER: split frozen m15 book by predicted side (operating + gate-only) | `m15_updown.py` | U/Dn | — | **done** | repro .689/.582/.663 | binding .607 CI[.513,.692] | binding .559 CI[.472,.646] | `m15_updown_result.json` |
| A8b-up | side-pipeline | (15m,UP) coverage curve (d) — VAL-worst-half threshold sweep, per-year binding CI-lo | `m15_sidepipe.py` | U | high | pending | | | | |
| A8b-dn | side-pipeline | (15m,DOWN) coverage curve (d) — symmetric | `m15_sidepipe.py` | Dn | high | pending | | | | |
| A8c-up | side-pipeline | (15m,UP) full-refit CPCV (e) at operating gate (refit p10≥.541 & ≥80% folds) | `m15_cpcv_side.py` | U | high | pending | | | | |
| A8c-dn | side-pipeline | (15m,DOWN) full-refit CPCV (e) | `m15_cpcv_side.py` | Dn | high | pending | | | | |
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
Round 0: pending — fan-out readers over CORPUS_LEVER_INVENTORY.md (190 sign-aware direction levers) filtered to
15m-relevant + sign-carrying + on-disk. Append survivors as Tier-N rows here + IDEAS_LOG. Stop after K=2 dry rounds.

## STATUS: A8a done (side-split). NEXT: A8b coverage curve (both sides) → A8c CPCV.
