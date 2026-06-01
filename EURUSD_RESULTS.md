# EURUSD — Experiment Results by Horizon + Up/Down Leaderboard

**Pair tag: `EURUSD`.** (Other currencies will get their own `<PAIR>_RESULTS.md`.) This is the results ledger: every direction/magnitude experiment, organized by prediction horizon, with stats and verdict. The companion `METHODS_CATALOG.md` describes *how/why* each methodology works; this file records *what it scored on EURUSD*.

**Running up/down leaderboard** (§ "UP vs DOWN Leaderboard" below) tracks the best methodology for UP prediction and for DOWN prediction with stats — **update it whenever a new experiment unseats a leader.**

**Conventions.** Deriv-faithful settlement (mid-to-mid, next-tick entry +1s, ties LOSE), breakeven **0.541**. Splits: bars train 2012-21 / val 2022-23 / test 2024 / test 2025 / oos 2026; tick train 2021-23 / val 2024-H1 / test 2024.09-2025.11 / oos 2026. Selection on VAL worst-half (never VAL-acc-max; `corr(VAL,OOS)=−0.54`). Moved-bars-only; per-year CI95. Deriv EURUSD forex Rise/Fall **minimum expiry = 15m** → 1s/1m/5m/10m are research/synthetic-index horizons; 15m/30m are deriv-tradeable. Last updated 2026-06-01.

---

## Horizon summary (best honest book per horizon)

| Horizon | Best honest DIRECTION book | Stat | >0.65 robust? | Deriv-tradeable? |
|---|---|---|---|---|
| **1–5s (tick)** | tick microstructure ensemble (`m_tick_prod.py`) | **~0.65** (3s 0.657/0.667) | borderline, needs tick venue | ❌ (15m floor) |
| **60s (1m)** | HMM vol-state reversion / up-only filter | ~0.55 frozen, ~0.58–0.60 thin/regime | ❌ | ❌ |
| **5m** | cross-horizon soft stack (`m5_stack2.py`) | **0.613** verifiable / 0.648 thin | ❌ | ❌ |
| **10m** | native-10 × 5m_bb_width-NY (`m10_freeze_honest.py`) | **0.602** (floor 0.579) | ❌ | ❌* |
| **15m** | compression×NY ensemble (`m15_production.py`) | **0.647** recent / **0.579** CPCV-faithful | ❌ | ✅ |
| **30m** | compression-1h×NY ensemble (`m30_production.py`) | **0.591** (CI[.556,.625]) | ❌ | ✅ |
| **any** | **MAGNITUDE \|ret\|≥Q (the one certified edge)** | **AUC 0.71–0.81** (not up/down) | ✅ (as size, not sign) | Touch/Range/Straddle |

\*10m "deriv" tag conflicts with the 15m forex floor (`research_log.md:701-702`); treat as tradeable only if a sub-15m EURUSD product exists at your venue.

---

## Tick / 1-second
DIRECTION at the tick→seconds scale. Tick/seconds-expiry venues only (deriv forex floor = 15m).

| Method (file) | Result | Verdict |
|---|---|---|
| **Tick microstructure ensemble @3–5s** (`m_tick_prod.py`) | **~0.65–0.66** selective; 3s **0.657 / 0.667**, n=377–1124 @cov0.5–2% | ✅ the one genuine >0.65 directional edge — latency-critical |
| Tick LGBM ensemble @3s/5s (`tick_ensemble.py`) | ~0.65 @1–5s band | clears ~0.65 not 0.75 |
| Horizon-frontier sweep (`tick_horizon_sweep.py`, `tickhz.py`) | ~0.65 @1–5s → **~0.50 AUC by ≥60s** | mapping: edge is seconds-local, decays fast |
| Raw imbalance/microprice decay (`rawtick_decay.py`) | next-tick 0.553 → +10t 0.509 → 1min 0.501 → 5min 0.501 | mechanistic proof of the sub-minute wall |
| Path-signature / Lévy area (`m30_sig.py` @5s) | top-5 feature @5s (null @30m) | the price↔flow rotation is a seconds signal |

**Conclusion.** The only place EURUSD direction clears ~0.60 is the 1–5s tick scale (~0.65, robust TEST+OOS), driven by order-flow imbalance/microprice. It decays to noise by 60s. Citations: `m_tick_prod.py:4-5`, `DIRECTION_FINDINGS.md:142,148,178,199`. (No tick result-JSON on disk; figures are from script docstrings + findings docs; backing models gitignored.)

---

## 60-second (1-minute)
Breakeven 0.541. **60s direction is near-efficient: ~0.50–0.51 AUC across every model class.** Magnitude IS forecastable (AUC ~0.79).

| # | Method (file) | Result (2024/2025/2026, AUC, CI) | Verdict |
|---|---|---|---|
| 1 | **Frozen production** reversion×compression (`min1_production.py`) | **TEST 0.539 / OOS 0.550** (CI incl. 0.50) | reference baseline |
| 2 | Cross-horizon stack (`min1_stack.py`) | worst-VAL-half 0.586–0.594; verifiable oos 0.53–0.55 | ❌ |
| 3 | Cross-pair USD-residual (`m5_xpair.py` MX_HOR=1) | VAL AUC 0.516; test25 0.534 CI[.518,.550] | ❌ |
| 4 | Hurst/VR persistence switch (`min1_hurst.py`) | floor 0.513, oracle 0.555 | ❌ |
| 5 | HMM regime (`min1_hmm.py`) | U1 0.565 / **U2 engine-switch 0.600** (2024 .600/2025 .605/2026 .613, CI[.519,.708], n≈106–162) / U3 0.487 | ❌ best refinement but thin, CI spans breakeven |
| 6 | Online ARF+ADWIN (`min1_online.py`) | AUC 0.503–0.508 every window | ❌ NULL → **2025 wall is genuine efficiency** |
| 7 | Macro 60s impulse (`min1_news60.py`) | HIGH-vol 0.373/0.526/0.517; &\|z\|≥1 → 0.167/0.364/**0.000** | ❌ bigger surprise = more wrong |
| 8 | **Cross-impact OFI matrix** (`min1_xofi.py`) | VAL dirAUC **0.5015**; 78.9% gain off-diagonal yet coin-flip; no sign-flip | ❌ KILLED (`min1_xofi_result.json`) |
| 9 | CKS event-OFI standalone (`min1_cksofi.py`) | VAL dirAUC **0.4993** | ❌ KILLED (`min1_cksofi_result.json`) |
| 10 | Kalman (`min1_kalman.py`) | channel 0.499 / velocity 0.492 / β-resid 0.504 | ❌ NULL |
| 11 | Kernel-SVM (`min1_kernel.py`) | VAL AUC 0.502, train 0.529 (can't memorize) | ❌ NULL — data is the ceiling |
| 12 | RMT eigen-residual (`min1_rmt.py`) | 2025 selective 0.508–0.520 | ❌ NULL — degenerate w/ USD factor |
| 13 | Neural-CDE irregular Δt (`min1_ncde.py`) | irregular VAL AUC **0.498** ≤ grid 0.509; 0.490–0.507/yr | ❌ NULL (`min1_ncde_results.json`) |
| 14 | DRL DQN/IQN (`min1_drl.py`) | DQN 0.482/0.484/0.483; IQN+CVaR 0.489/0.494/0.496 | ❌ NULL — below 0.50 |
| 15 | Ordinal irreversibility (`min1_irrev.py`) | VAL AUC 0.503; 0.479/0.488/0.489, CI<0.50 | ❌ KILLED (`min1_irrev_result.json`) |
| 16 | Residualized TARGET (`min1_residtarget.py`) | H1 resid-sign 2026 0.534; agree 2025 0.522 | ❌ KILLED (`min1_residtarget_result.json`) |
| 17 | **CCM coupling-gate** (`min1_ccm.py`) | all 7 drivers slope≤0.0066 (no convergence); gated 2024≈0.51/2025≈0.503/2026≈0.52 | ❌ KILLED (`min1_ccm_result.json`) — even EUR-own-OFI doesn't cross-map its own 60s return |

**Adversarial controls.** Informational ceiling (`_redteam_magdir60.py`): 2025 conditional-acc 0.512@10%→0.517@0.5%cov (n11,283) — the gap to 0.65 is **informational, not coverage**; 60s **dirAUC 0.510 vs magAUC 0.787** on identical data (sign-invariance). Per-side raw flow (`_adj_perside_flow.py`): VAL dirAUC 0.5086, null. Caught false-positive AUC 0.728 = ffill-flat mirage (collapses to 0.49 on moved bars).

**Conclusion (`min1_research_log.md:66`):** *"A 1-minute EURUSD up/down model with >0.65 OOS-stable accuracy is NOT achievable on this data."* Best honest book ≈ 0.55 frozen / ~0.58–0.60 at thin coverage (regime-dependent). The 2025 wall is genuine efficiency (the online learner can't adapt past it; macro shocks give 0.17–0.53).

---

## 5-minute
Research horizon (deriv forex floor 15m). Raw direction AUC ≈ **0.52** (EMH/noise floor).

| Method (file) | Result | Verdict |
|---|---|---|
| OHLCV ensemble + cross-pair + OF (`m5_production.py`, `m5_xpair.py`) | VAL AUC ~0.523; book combined 0.586–0.594 | base ~0.52, book to 0.59 |
| Meta-labeler orthogonal axes (`m5_meta.py`) | thr0.588: t24 .643/t25 .580/oos .615/**comb 0.612** CI[.596,.628] | best honest ~0.61 |
| **PRODUCTION freeze** (`m5_xpair_production.py`) | t24 .607/t25 .555/oos .606/**COMB 0.583** CI[.570,.595], EV +0.078 | frozen deliverable, profitable, not 0.65 |
| **Cross-horizon soft STACK** (`m5_stack2.py`) | q0.98 **comb 0.613** (frozen) / q0.99 **0.648** (oos n45) | ✅ STRONGEST 5m method (0.613 verifiable) |
| dir15→5m transfer (`m5_xhorizon.py`) | comb 0.597 (beats native 0.583) | 15m edge front-loads |
| Walk-forward (`m5_walkforward.py`) | t24 .676/t25 .557/oos .565/**comb 0.600** | +0.015 only; regime fundamental |
| Sofien indicators/setups/confluence (`m5_sofien*.py`, `m5_patterns.py`) | 0 AUC added; rules 0.50–0.535 | ❌ NULL |
| Macro-surprise rule + model (`m5_news*.py`) | large-surprise NY 0.459/0.475/0.513 | ❌ NULL (direction) |
| Tick microstructure HS=300 (`m5_tick.py`) | ~0.53 best selective | ❌ below OHLCV |

**Conclusion.** 5m frontier improved 0.566 → 0.648 across the session; **0.613 verifiable / 0.648 thin** (cross-horizon stack). Profitable vs 0.541 but **robust >0.65 NOT achieved** — binding window is 2025 (~0.55–0.60 even under hindsight oracle). 5m is **capped by the 15m parent** (~0.647). Citations: `m5_research_log.md:142-162`, `DIRECTION_FINDINGS.md:18`.

---

## 10-minute
Raw direction AUC ≈ **0.525**. Honest ceiling ≈ 0.60–0.61.

| Method (file) | Result | Verdict |
|---|---|---|
| Native-10 ensemble, VAL-acc-max (`m10_production.py`) | combined **0.667** (t25 0.591, oos n57) | ❌ artifact — VAL-max trap |
| **Honest gate sweep** 200 configs (`m10_gate_sweep.py`) | honest floor 0.549/comb 0.576; **oracle max-floor 0.599** | no gate clears 0.65 even cheating |
| Cross-horizon stack dir15/dir10 (`m10_stack.py`) | dir10 comb 0.598, oracle floor 0.560 | ~0.59, no edge created |
| Cross-pair USD-resid+OF MX_HOR=10 (`m5_xpair.py`) | comb 0.599 CI[.579,.620] (covcurve mirage) | XP lift weaker at 10m |
| Walk-forward 1yr-gap (`m10_walkforward.py`) | t24 .668/t25 .573/oos .574/**comb 0.609** | t25 +0.017 only |
| **HONEST freeze** native-10 × 5m_bb_width-NY (`m10_freeze_honest.py`) | t24 .614/t25 .579/oos .594/**COMB 0.602** CI[.582,.621], floor 0.579, EV +0.113 | ✅ honest deliverable ~0.60 |
| ES→EURUSD lead-lag (`m10_xasset_probe.py`) | lagged corr +0.02 (2024) → **−0.05 (2025)** sign-flip | ❌ null + **mechanistic key to the 2025 wall** |
| **Direction-on-magnitude** (`m10_magdir.py`) | magnitude AUC **0.813/0.741/0.706**; direction flat 0.51–0.53 across all mag quartiles | direction null / **magnitude clears 0.65-equiv** |

**Conclusion.** 9 converging Tier-1 nulls; >0.65 OOS-verified not achievable, every method capped by 2025. Deliverable = `m10_freeze_honest.py` (0.602, floor 0.579). The genuine >0.65 at 10m is **magnitude** (AUC 0.71–0.81, stable all 3 years). Citations: `m10_research_log.md:115-191`.

---

## 15-minute (deriv-tradeable)
**The best up/down book in the program.** Raw direction AUC ≈ 0.528.

| # | Method (file) | Result | Verdict |
|---|---|---|---|
| D1 | **Production ensemble × comp×NY selective** (`m15_production.py`) | combined **0.647** (n677, CI[.612,.684]); **2024 0.689 / 2025 0.582 / 2026 0.663** | ✅ real, regime-dependent, tradeable |
| D2 | **FAITHFUL CPCV of the same book** (`min15_cpcv.py`) | gated-sel **mean 0.5787**, p10 0.5574, min 0.538, max 0.633; **14/15 paths > 0.541** | ✅ durable cross-era ~0.58 (not a mirage) |
| D3 | Walk-forward adaptive (`m15_walkforward.py`) | 2024 .543 / 2025 .536 / 2026 .604 — all BELOW frozen | ❌ adaptation makes it WORSE → frozen is optimal |
| D4 | CPCV weaker re-impl (single LGBM, q33) (`cpcv_certify.py`) | selective mean 0.5455, p10 0.5306 | ⚠ NOT a faithful test — flags regime-dependence only |
| D5 | Meta-labeler (`m15_meta.py`) | 0.615 < 0.647 | ❌ can't exceed parent |
| D6 | Specialist+reversion (`min15_v2.py`) | baseline 0.642 (2024 .691/2025 .590/2026 .592) | no lift |
| D7 | Cross-pair exog (`exp_15m_v3.py`) | +0 AUC | ❌ dead at 15m |
| D8 | exp_15m v1→v13 arc | converges on comp×NY ≈ 0.642–0.647 | the frozen pipeline's lineage |

**Reconciling 0.647 vs 0.58 vs walk-forward:** all three test the *same* book. **0.647** = recent chronological split (favorable-regime end). **0.579** = faithful 15-path CPCV across all eras (durable headline). Walk-forward = adaptive retrain, *worse* (frozen is optimal; 2025 not fixable by adaptation). The 0.545 from `cpcv_certify.py` is a weaker re-implementation (single LGBM/q33/all-era), not a refutation. **Net: real, regime-dependent, robustly profitable; honest headline 0.579 (recent 0.647).** Magnitude at 15m: never built (interpolates 0.73–0.79). Citations: `min15_cpcv_result.json`, `m15_walkforward_result.json`, `DIRECTION_FINDINGS.md:15`.

---

## 30-minute (deriv-tradeable)
Raw direction AUC ≈ 0.52. **Home of the one CPCV-deflation-certified edge (magnitude).**

### Direction
| Method (file) | Result | Verdict |
|---|---|---|
| **Production comp-1h×NY selective** (`m30_production.py`) | combined **0.591** CI[.556,.625]; 2024 .623/2025 .589/2026(oos) .546; EV +0.064 | ✅ honest deliverable |
| Baseline + regime gates (`m30_lab.py`) | comp1h_ny cov2% oos 0.639 (n83); cov1% spikes = small-n mirage | ~0.60–0.64 non-transferring |
| Ensemble + agreement (`m30_ens.py`) | best stable pocket ~0.60 (oos 0.621, n124) | ceiling ~0.60–0.64 |
| FX fixing reversal (`m30_fix.py`) | WMR reversal t24 .580/t25 .549/**oos26 .429 (sign FLIPS)** | ❌ real in-sample, sign-flips OOS |
| Complexity gates PE/Hurst (`m30_complexity.py`) | direction flat ~0.515 all bins | ❌ sign-invariant |
| Tick microstructure (`m30_tick.py`) | AUC 0.501–0.503 | ❌ PURE NOISE |
| ES lead-lag (`m30_es_feas.py`) | 30m lead corr +0.001/−0.041/+0.011 (sign-flips) | ❌ null (real external data) |
| Path-sig + Hawkes (`m30_sig.py`) | null @30m | ❌ (seconds-only signal) |

### Magnitude — the certified edge
| Method (file) | Result | Verdict |
|---|---|---|
| **CPCV-certified large-move** (`cpcv_certify.py`) | **AUC mean 0.744**, p10 0.718, deflated 0.712; 28/28 paths clear; **decile lift 5.0×** | ✅✅ THE deflation-proof edge |
| rv30 large-move (`m30_magnitude.py`) | AUC test24 0.783 / test25 0.729 / oos 0.729; PE null (~0.50) | ✅ verified — realized vol is the predictor, NOT entropy |

**Conclusion.** 9+ converging direction nulls; honest direction deliverable = `m30_production.py` ~0.591. **Magnitude (|ret30|) AUC 0.73–0.78, every purged path clears** — tradeable on Touch/Range/Straddle/VRP, not Rise/Fall. Citations: `cpcv_certify_result.json`, `magnitude_verified.json`, `m30_research_log.md`.

---

## UP vs DOWN Leaderboard (running — update when a leader is unseated)

**The directional edge is one-sided and lives on the UP (dip-buy) side.** Treat side asymmetry as a **trade FILTER on the symmetric ensemble**, never as separately-trained side models (subset-training kills the confidence ranking — reproduced at v4/v6/v13/up-spec).

### Current leaders

| Side | Best methodology | Horizon | Stats (2024 / 2025 / 2026) | File | Caveat |
|---|---|---|---|---|---|
| **UP** 🏆 | **Up-only filter on symmetric ensemble** (dip-buy: keep predicted-up trades) | **15m** | up-predictions **2025 0.584 / 2026 0.613** (2024 wash ~0.52) | `min1_updown.py` diag; `DIRECTION_FINDINGS.md:60-61` | regime-dependent; vanishes in 2024; does NOT clear 0.65 |
| **UP** (60s) | Up-only production-gate filter | 60s | 2025 0.581 / 2026 0.600, floor 0.518 | `min1_best.py`, `min1_upspec.py` | 2024 sags; no freeze |
| **DOWN** | *(no methodology beats coin-flip on the down side at any horizon)* | — | down-predictions ~0.516–0.522 every year | `min1_updown.py` | **DOWN side is unsolved** — rally-selling is dead in the 2024-26 EUR-up regime |

### Failed side-specific approaches (controls)
| Approach | Result | Why it failed |
|---|---|---|
| Dedicated UP-specialist (trained only on dip-buy bars) | 0.498 / 0.537 / 0.519 | subset-training destroys the confidence ranking (WORSE than the filter) |
| Dedicated DOWN-specialist | 0.472 / 0.521 / 0.481 | dead — no down-side signal to sharpen |

### Notes & open gaps
- **Mechanism:** dip-buying worked while rally-selling didn't over the EUR-up 2025-26 regime; the asymmetry is a *regime artifact*, not a structural edge (it disappears in 2024). Setup base-rates are ~coin-flip (dip-bounce 0.507, rally-down 0.514), so a specialist has little to sharpen.
- **Coverage gap:** up/down asymmetry has only been tested at **60s and 15m** (`min1_*`). **It has NOT been tested at 5m, 10m, or 30m** — those are open backlog items. When run, add rows here.
- **Magnitude** is sign-invariant by construction (|ret|) → carries no up/down split; it is the size edge, orthogonal to this leaderboard.
- **To unseat a leader:** a new methodology must beat the incumbent's binding-year stat (currently UP@15m 2025 0.584) with CI95-lower clearing it on the worst held-out year, under deriv-faithful moved-bars discipline. Record the file + per-year stats + verdict and move the old leader to a "previous leaders" line.

---

## How to maintain this file
1. After each experiment, add a row to the relevant horizon table: method, file, per-year stats + CI, verdict.
2. If it produces a side-specific (up or down) result, also update the **UP vs DOWN Leaderboard** — unseat the leader if it beats the binding-year stat under discipline; otherwise log it as a control.
3. Update the **Horizon summary** table if the best honest book for a horizon changes.
4. Keep every number traceable to a result JSON or research-log line (Tier-1). Flag thin-coverage (n<50) and VAL-acc-max numbers as non-robust.
5. New currencies get their own `<PAIR>_RESULTS.md` with the same structure.
