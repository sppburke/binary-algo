# EURUSD — Results Ledger (unique key: currency × timeframe × side)

**Pair tag: `EURUSD`.** (Other currencies get their own `<PAIR>_RESULTS.md`.)

**The unique result key is `(currency, timeframe, side)` where side ∈ {UP, DOWN}.** Every key has its own result — the result for one timeframe is NOT the result for another, and the UP result is NOT the DOWN result. Where a key has not been measured yet, it is marked **`UNTESTED`** (an open backlog item) rather than inheriting another key's number.

A note on "combined / symmetric" books: most models trade BOTH sides and report a single pooled accuracy. That pooled number is shown per timeframe as **context only** (labelled `COMBINED`); it is decomposed into its UP-side and DOWN-side accuracy ONLY where that decomposition was actually run (so far: 60s only). A pooled number does NOT populate the UP or DOWN key.

**Conventions.** Deriv-faithful settlement (mid-to-mid, next-tick entry +1s, ties LOSE), breakeven **0.541**. Splits: bars train 2012-21 / val 2022-23 / test 2024 / test 2025 / oos 2026; tick train 2021-23 / val 2024-H1 / test 2024.09-2025.11 / oos 2026. Selection on VAL worst-half (never VAL-acc-max; `corr(VAL,OOS)=−0.54`). Moved-bars-only; per-year CI95. Deriv EURUSD forex Rise/Fall **minimum expiry = 15m** → 1s/1m/5m/10m are research/synthetic-index horizons; 15m/30m are deriv-tradeable. Last updated 2026-06-01.

---

## MASTER KEY TABLE — one row per (EURUSD, timeframe, side)

For each key: the **model ensemble / methodology** that produced the best OOS result, and **that best OOS number**. `OOS = 2026` (the strict held-out year; test = 2024–2025). "MEASURED" keys are side-split results; "by-side UNTESTED" keys show the best COMBINED book (both sides pooled) as the current best-available proxy + its OOS — the side-specific split has not been run.

| Currency | Timeframe | Side | Best model ensemble / methodology (file) | **Best OOS result (2026)** | Key status |
|---|---|---|---|---|---|
| EURUSD | **1–5s** | UP | Tick microstructure GBM ensemble, lgb+xgb+cat (`m_tick_prod.py`) — *combined, not side-split* | **~0.65** (combined; TEST+OOS robust) | by-side UNTESTED |
| EURUSD | **1–5s** | DOWN | Tick microstructure GBM ensemble, lgb+xgb+cat (`m_tick_prod.py`) — *combined, not side-split* | **~0.65** (combined) | by-side UNTESTED |
| EURUSD | **60s** | **UP** | **Up-only filter on the 3-model GBM ensemble (lgb+xgb+cat) + compression-release specialist, reversion-filtered** (`min1_updown.py` / `min1_production.py`) | **0.613** ✅ | MEASURED |
| EURUSD | **60s** | **DOWN** | Same 3-model ensemble, **down-side predictions** (`min1_updown.py`) | **0.516** ❌ dead | MEASURED |
| EURUSD | **5m** | UP | Cross-horizon soft stack: 15m-parent ensemble front-loaded into 5m, meta-gated (`m5_stack2.py`) — *combined* | **~0.571** (combined; 3-yr headline 0.613) | by-side UNTESTED |
| EURUSD | **5m** | DOWN | Cross-horizon soft stack (`m5_stack2.py`) — *combined* | **~0.571** (combined) | by-side UNTESTED |
| EURUSD | **10m** | UP | Native-10 3-model GBM ensemble, gated by 5m_bb_width × NY (`m10_freeze_honest.py`) — *combined* | **0.594** (combined 3-yr 0.602) | by-side UNTESTED |
| EURUSD | **10m** | DOWN | Native-10 ensemble × 5m_bb_width-NY gate (`m10_freeze_honest.py`) — *combined* | **0.594** (combined) | by-side UNTESTED |
| EURUSD | **15m** | UP | 3-model GBM ensemble (lgb+xgb+cat) gated by 15m compression(bb_width) × NY (`m15_production.py`) — *combined* | **0.663** (combined recent 0.647; CPCV-faithful 0.579) | by-side UNTESTED |
| EURUSD | **15m** | DOWN | 3-model ensemble × compression×NY (`m15_production.py`) — *combined* | **0.663** (combined) | by-side UNTESTED |
| EURUSD | **30m** | UP | 3-model GBM ensemble gated by 1h-compression × NY (`m30_production.py`) — *combined* | **0.546** (combined 3-yr 0.591) | by-side UNTESTED |
| EURUSD | **30m** | DOWN | 3-model ensemble × 1h-compression-NY (`m30_production.py`) — *combined* | **0.546** (combined) | by-side UNTESTED |

**Only 2 of the 12 side-keys are MEASURED** (60s UP 0.613, 60s DOWN 0.516). Every "by-side UNTESTED" key shows the best COMBINED book + its OOS as the current best-available number; the true side-specific OOS is unknown until that book's trades are split by predicted side. **Do NOT treat the combined OOS as the UP or DOWN number, and do NOT copy 60s's asymmetry to another timeframe** — at 60s the UP edge is real in 2025–26 but a wash in 2024, i.e. horizon- AND regime-specific.

*Provenance of the OOS numbers:* 60s UP/DOWN — `min1_research_log.md:104-113` (2026 column). 5m 0.571 — `m5_research_log.md` (m5_stack2 q0.98 oos n163). 10m 0.594 — `m10_freeze_honest.py` / `m10_research_log.md`. 15m 0.663 — `m15_production` 2026 (`DIRECTION_FINDINGS.md:15`, `m15_walkforward.py` frozen map). 30m 0.546 — `m30_production.py` (`m30_research_log.md`). 1–5s ~0.65 — `m_tick_prod.py:4-5` (no 2026-only split on disk; TEST+OOS-robust).

**Magnitude** (|ret|≥Q) is sign-invariant by construction → it has NO up/down key. It is the size edge (AUC 0.71–0.81), tracked separately in `MAGNITUDE_FINDINGS.md`, not here.

---

# Per-timeframe detail
Each timeframe is fully self-contained: its COMBINED-book experiments, then its explicit UP and DOWN key results.

---

## EURUSD × 1–5s (tick)
DIRECTION at the tick→seconds scale. Tick/seconds-expiry venues only (deriv forex floor = 15m).

### Combined-book experiments
| Method (file) | Result | Verdict |
|---|---|---|
| **Tick microstructure ensemble @3–5s** (`m_tick_prod.py`) | **~0.65–0.66** selective; 3s **0.657 / 0.667**, n=377–1124 @cov0.5–2% | ✅ the one genuine >0.65 directional edge — latency-critical |
| Tick LGBM ensemble @3s/5s (`tick_ensemble.py`) | ~0.65 @1–5s band | clears ~0.65 not 0.75 |
| Horizon-frontier sweep (`tick_horizon_sweep.py`, `tickhz.py`) | ~0.65 @1–5s → **~0.50 AUC by ≥60s** | edge is seconds-local, decays fast |
| Raw imbalance/microprice decay (`rawtick_decay.py`) | next-tick 0.553 → +10t 0.509 → 1min 0.501 | mechanistic proof of the sub-minute wall |
| Path-signature / Lévy area (`m30_sig.py` @5s) | top-5 feature @5s (null @30m) | price↔flow rotation is a seconds signal |

### Key results
| Key | Result | Status |
|---|---|---|
| **(EURUSD, 1–5s, UP)** | `UNTESTED` | Combined ~0.65 measured, never decomposed by side. Open: split tick-ensemble trades by predicted side. |
| **(EURUSD, 1–5s, DOWN)** | `UNTESTED` | Same — open. |

Citations: `m_tick_prod.py:4-5`, `DIRECTION_FINDINGS.md:142,148,178,199`. (No tick result-JSON; figures from docstrings + findings docs; models gitignored.)

---

## EURUSD × 60s (1-minute)
Breakeven 0.541. **60s direction is near-efficient: ~0.50–0.51 AUC across every model class.** This is the ONLY timeframe with a measured up/down split.

### Combined-book experiments
| # | Method (file) | Result (2024 / 2025 / 2026, AUC, CI) | Verdict |
|---|---|---|---|
| 1 | **Frozen production** reversion×compression (`min1_production.py`) | **TEST 0.539 / OOS 0.550** (CI incl. 0.50); per-year symmetric ~0.548 (2025) / 0.550 (2026) | reference baseline |
| 2 | Cross-horizon stack (`min1_stack.py`) | worst-VAL-half 0.586–0.594; verifiable oos 0.53–0.55 | ❌ |
| 3 | Cross-pair USD-residual (`m5_xpair.py` MX_HOR=1) | VAL AUC 0.516; test25 0.534 CI[.518,.550] | ❌ |
| 4 | Hurst/VR persistence switch (`min1_hurst.py`) | 2024 .518 / 2025 .548 / 2026 .513, floor 0.513 | ❌ |
| 5 | HMM regime (`min1_hmm.py`) | U2 engine-switch 2024 .600 / 2025 .605 / 2026 .613 (CI[.519,.708], n≈106–162) | ❌ thin, CI spans breakeven |
| 6 | Online ARF+ADWIN (`min1_online.py`) | AUC 0.503–0.508 every window | ❌ NULL → **2025 wall is genuine efficiency** |
| 7 | Macro 60s impulse (`min1_news60.py`) | HIGH-vol 0.373/0.526/0.517; &\|z\|≥1 → 0.167/0.364/**0.000** | ❌ bigger surprise = more wrong |
| 8 | **Cross-impact OFI matrix** (`min1_xofi.py`) | VAL dirAUC **0.5015**; 78.9% gain off-diagonal yet coin-flip | ❌ KILLED (`min1_xofi_result.json`) |
| 9 | CKS event-OFI standalone (`min1_cksofi.py`) | VAL dirAUC **0.4993** | ❌ KILLED (`min1_cksofi_result.json`) |
| 10 | Kalman (`min1_kalman.py`) | channel 0.499 / velocity 0.492 / β-resid 0.504 | ❌ NULL |
| 11 | Kernel-SVM (`min1_kernel.py`) | VAL AUC 0.502, train 0.529 (can't memorize) | ❌ NULL — data is the ceiling |
| 12 | RMT eigen-residual (`min1_rmt.py`) | 2025 selective 0.508–0.520 | ❌ NULL — degenerate w/ USD factor |
| 13 | Neural-CDE irregular Δt (`min1_ncde.py`) | irregular VAL AUC **0.498** ≤ grid 0.509 | ❌ NULL (`min1_ncde_results.json`) |
| 14 | DRL DQN/IQN (`min1_drl.py`) | DQN 0.482/0.484/0.483; IQN+CVaR 0.489/0.494/0.496 | ❌ NULL — below 0.50 |
| 15 | Ordinal irreversibility (`min1_irrev.py`) | VAL AUC 0.503; 0.479/0.488/0.489 | ❌ KILLED (`min1_irrev_result.json`) |
| 16 | Residualized TARGET (`min1_residtarget.py`) | H1 resid-sign 2026 0.534; agree 2025 0.522 | ❌ KILLED (`min1_residtarget_result.json`) |
| 17 | **CCM coupling-gate** (`min1_ccm.py`) | all 7 drivers slope≤0.0066; gated 2024≈0.51/2025≈0.503/2026≈0.52 | ❌ KILLED (`min1_ccm_result.json`) |

Adversarial controls: informational ceiling (`_redteam_magdir60.py`) 2025 cond-acc 0.512@10%→0.517@0.5%cov (gap to 0.65 is informational, not coverage); 60s **dirAUC 0.510 vs magAUC 0.787** identical data (sign-invariance).

### Key results (MEASURED)
| Key | Result (2024 / 2025 / 2026) | n / floor | Method | Status |
|---|---|---|---|---|
| **(EURUSD, 60s, UP)** 🏆 | **0.520 / 0.584 / 0.613** | floor 0.520 | up-only FILTER on symmetric ensemble (`min1_updown.py`) | ✅ genuine improvement over symmetric (0.548/0.550); regime-dependent; 2024 a wash; **does not clear 0.65** |
| **(EURUSD, 60s, DOWN)** | **0.522 / 0.522 / 0.516** | — | symmetric ensemble, down-predictions (`min1_updown.py`) | ❌ dead — no down-side edge in any year |

**Side-specific TRAINING (controls — both worse than the filter):**
| Approach | 2024 / 2025 / 2026 | Why it failed |
|---|---|---|
| Dedicated UP-specialist (trained only on dip-buy bars) (`min1_upspec.py`) | 0.498 / 0.537 / 0.519 | subset-training destroys the confidence ranking (WORSE than the filter) |
| Dedicated DOWN-specialist (control) | 0.472 / 0.521 / 0.481 | dead — no down-side signal to sharpen |
| Up-only production-gate filter (refined config) (`min1_best.py`) | sags / 0.581 / 0.600 | floor 0.518; no freeze (2024 sags) |

**Mechanism:** the reversion lever buys dips / sells rallies; dip-buying worked while rally-selling didn't over the EUR-up 2025-26 regime → the UP-key carries and the DOWN-key is dead. The asymmetry VANISHES in 2024 (UP 0.520 ≈ DOWN 0.522) → it's a regime artifact, not structural. Setup base-rates ~coin-flip (dip-bounce 0.507, rally-down 0.514). Citations: `min1_research_log.md:104-113,139`.

---

## EURUSD × 5m
Research horizon (deriv forex floor 15m). Raw direction AUC ≈ 0.52 (EMH/noise floor).

### Combined-book experiments
| Method (file) | Result | Verdict |
|---|---|---|
| OHLCV ensemble + cross-pair + OF (`m5_production.py`, `m5_xpair.py`) | VAL AUC ~0.523; book 0.586–0.594 | base ~0.52 |
| Meta-labeler orthogonal axes (`m5_meta.py`) | t24 .643/t25 .580/oos .615/**comb 0.612** CI[.596,.628] | best honest ~0.61 |
| **PRODUCTION freeze** (`m5_xpair_production.py`) | t24 .607/t25 .555/oos .606/**COMB 0.583** CI[.570,.595], EV +0.078 | frozen deliverable |
| **Cross-horizon soft STACK** (`m5_stack2.py`) | q0.98 **comb 0.613** (frozen) / q0.99 0.648 (oos n45) | ✅ strongest 5m (0.613 verifiable) |
| dir15→5m transfer (`m5_xhorizon.py`) | comb 0.597 | 15m edge front-loads |
| Walk-forward (`m5_walkforward.py`) | t24 .676/t25 .557/oos .565/**comb 0.600** | +0.015 only |
| Sofien rules / macro / tick (`m5_sofien*.py`, `m5_news*.py`, `m5_tick.py`) | 0.50–0.535 / null / ~0.53 | ❌ NULL |

### Key results
| Key | Result | Status |
|---|---|---|
| **(EURUSD, 5m, UP)** | `UNTESTED` | Combined best 0.613 (cross-horizon stack), never split by side. Open: split `m5_stack2.py`/`m5_xpair_production.py` trades by predicted side. |
| **(EURUSD, 5m, DOWN)** | `UNTESTED` | Same — open. |

Conclusion (combined): frontier 0.566→0.648 across session; **0.613 verifiable / 0.648 thin**; capped by the 15m parent; >0.65 not achieved (binding window 2025). Citations: `m5_research_log.md:142-162`, `DIRECTION_FINDINGS.md:18`.

---

## EURUSD × 10m
Raw direction AUC ≈ 0.525. Honest combined ceiling ≈ 0.60.

### Combined-book experiments
| Method (file) | Result | Verdict |
|---|---|---|
| Native-10, VAL-acc-max (`m10_production.py`) | combined 0.667 (t25 0.591, oos n57) | ❌ VAL-max artifact |
| Honest gate sweep 200 cfg (`m10_gate_sweep.py`) | honest floor 0.549/comb 0.576; oracle 0.599 | no gate clears 0.65 even cheating |
| Cross-horizon stack (`m10_stack.py`) | dir10 comb 0.598, oracle floor 0.560 | ~0.59 |
| Cross-pair MX_HOR=10 (`m5_xpair.py`) | comb 0.599 CI[.579,.620] | XP lift weaker at 10m |
| Walk-forward (`m10_walkforward.py`) | t24 .668/t25 .573/oos .574/**comb 0.609** | t25 +0.017 only |
| **HONEST freeze** native-10 × 5m_bb_width-NY (`m10_freeze_honest.py`) | t24 .614/t25 .579/oos .594/**COMB 0.602** CI[.582,.621], floor 0.579 | ✅ honest deliverable |
| ES→EURUSD lead-lag (`m10_xasset_probe.py`) | lagged corr +0.02 (2024) → **−0.05 (2025)** | ❌ null + **mechanistic key to 2025 wall** |
| Direction-on-magnitude (`m10_magdir.py`) | magnitude AUC 0.813/0.741/0.706; direction flat 0.51–0.53 | direction null / magnitude clears 0.65-equiv |

### Key results
| Key | Result | Status |
|---|---|---|
| **(EURUSD, 10m, UP)** | `UNTESTED` | Combined best 0.602, never split. Open: split `m10_freeze_honest.py` trades by predicted side. |
| **(EURUSD, 10m, DOWN)** | `UNTESTED` | Same — open. |

Conclusion (combined): 9 converging nulls; deliverable 0.602 (floor 0.579); the >0.65 at 10m is magnitude. Citations: `m10_research_log.md:115-191`.

---

## EURUSD × 15m (deriv-tradeable)
The best COMBINED direction book in the program. Raw AUC ≈ 0.528.

### Combined-book experiments
| # | Method (file) | Result | Verdict |
|---|---|---|---|
| D1 | **Production ensemble × comp×NY** (`m15_production.py`) | combined **0.647** (n677, CI[.612,.684]); 2024 0.689 / 2025 0.582 / 2026 0.663 | ✅ real, regime-dependent, tradeable |
| D2 | **FAITHFUL CPCV of same book** (`min15_cpcv.py`) | gated-sel **mean 0.5787**, p10 0.557, min 0.538, max 0.633; 14/15 paths >0.541 | ✅ durable cross-era ~0.58 |
| D3 | Walk-forward adaptive (`m15_walkforward.py`) | 2024 .543 / 2025 .536 / 2026 .604 — all BELOW frozen | ❌ adaptation WORSE → frozen optimal |
| D4 | CPCV weaker re-impl (`cpcv_certify.py`) | selective mean 0.5455, p10 0.531 | ⚠ not a faithful test |
| D5 | Meta-labeler (`m15_meta.py`) | 0.615 < 0.647 | ❌ can't exceed parent |
| D6 | Specialist+reversion (`min15_v2.py`) | 0.642 (2024 .691/2025 .590/2026 .592) | no lift |
| D7 | Cross-pair exog (`exp_15m_v3.py`) | +0 AUC | ❌ dead at 15m |

### Key results
| Key | Result | Status |
|---|---|---|
| **(EURUSD, 15m, UP)** | `UNTESTED` | Combined 0.647 recent / 0.579 CPCV-faithful, never split by side. ⚠ Earlier docs mis-attributed the 60s up/down numbers (0.584/0.613) to 15m — that was an error; 15m has NO measured up/down split. Open: split `m15_production.py` trades by predicted side. |
| **(EURUSD, 15m, DOWN)** | `UNTESTED` | Same — open. |

Reconciling the combined number: **0.647** = recent chronological split (favorable end); **0.579** = faithful 15-path CPCV (durable headline); walk-forward is worse (frozen optimal). The 0.545 from `cpcv_certify.py` is a weaker re-implementation, not a refutation. Citations: `min15_cpcv_result.json`, `m15_walkforward_result.json`, `DIRECTION_FINDINGS.md:15`.

---

## EURUSD × 30m (deriv-tradeable)
Raw direction AUC ≈ 0.52. Home of the one CPCV-deflation-certified edge (magnitude — sign-invariant, no up/down key).

### Combined-book DIRECTION experiments
| Method (file) | Result | Verdict |
|---|---|---|
| **Production comp-1h×NY** (`m30_production.py`) | combined **0.591** CI[.556,.625]; 2024 .623/2025 .589/2026 .546; EV +0.064 | ✅ honest deliverable |
| Baseline + gates (`m30_lab.py`) | comp1h_ny cov2% oos 0.639 (n83) | non-transferring |
| Ensemble + agreement (`m30_ens.py`) | best pocket ~0.60 (oos 0.621, n124) | ceiling ~0.60–0.64 |
| FX fixing reversal (`m30_fix.py`) | t24 .580/t25 .549/**oos26 .429 (sign FLIPS)** | ❌ sign-flips OOS |
| Complexity gates (`m30_complexity.py`) | flat ~0.515 all bins | ❌ sign-invariant |
| Tick microstructure (`m30_tick.py`) | AUC 0.501–0.503 | ❌ PURE NOISE |
| ES lead-lag (`m30_es_feas.py`) | 30m lead corr +0.001/−0.041/+0.011 | ❌ null (real external data) |

### Magnitude (the certified edge — NO up/down key, sign-invariant)
| Method (file) | Result | Verdict |
|---|---|---|
| **CPCV-certified large-move** (`cpcv_certify.py`) | **AUC mean 0.744**, p10 0.718, deflated 0.712; 28/28 paths clear; lift 5.0× | ✅✅ deflation-proof |
| rv30 large-move (`m30_magnitude.py`) | AUC test24 0.783 / test25 0.729 / oos 0.729; PE null | ✅ realized vol is the predictor, not entropy |

### Key results
| Key | Result | Status |
|---|---|---|
| **(EURUSD, 30m, UP)** | `UNTESTED` | Combined direction 0.591, never split by side. Open: split `m30_production.py` trades by predicted side. |
| **(EURUSD, 30m, DOWN)** | `UNTESTED` | Same — open. |

Conclusion: 9+ converging direction nulls; combined deliverable 0.591; magnitude AUC 0.73–0.78 (every purged path clears). Citations: `cpcv_certify_result.json`, `magnitude_verified.json`, `m30_research_log.md`.

---

## Why most (timeframe, side) keys are UNTESTED, and how to fill them
The up/down decomposition requires taking a frozen book's **independent trades and splitting them by predicted side**, then scoring each side per-year. This was only done at 60s (`min1_updown.py`). To populate any other key:
1. Load the frozen book for that timeframe (`m{5,10,15,30}_*production*` / tick).
2. Run its backtest, keep the non-overlap-chrono independent trades + predictions.
3. Split by `pred==UP` vs `pred==DOWN`; compute per-year (2024/2025/2026) moved-bars accuracy + CI95.
4. Write the two rows into this file's MASTER KEY TABLE and that timeframe's "Key results".
**Do not** assume another timeframe's asymmetry transfers — at 60s the UP edge is real in 2025-26 but a wash in 2024, i.e. horizon- AND regime-specific.

## How to maintain this file
1. Each experiment → add a row to the timeframe's combined-book table (method, file, per-year stats + CI, verdict).
2. If side-split, update that timeframe's **Key results** AND the **MASTER KEY TABLE** (one row per side). Unseat a key's leader only if the new method beats the incumbent's binding-year stat (worst held-out year) with CI95-lower clearing it, under deriv-faithful moved-bars discipline; move the old leader to a "previous" line.
3. Keep every number traceable to a result JSON or research-log line (Tier-1). Flag thin-coverage (n<50) and VAL-acc-max numbers as non-robust.
4. Magnitude (|ret|) is sign-invariant → it has no up/down key; record it in `MAGNITUDE_FINDINGS.md`.
5. New currencies get their own `<PAIR>_RESULTS.md` with this exact structure and key.
