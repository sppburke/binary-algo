> **SCOPE: EURUSD** (key-specific — all timeframes × sides; the results of record + UP/DOWN leaderboard). Generic methods: METHODS_CATALOG.md. See REPO_MAP.md.

# EURUSD — Results Ledger (unique key: currency × timeframe × side)

**Pair tag: `EURUSD`.** (Other currencies get their own `<PAIR>_RESULTS.md`.)

**The unique result key is `(currency, timeframe, side)` where side ∈ {UP, DOWN}.** Every key has its own result — the result for one timeframe is NOT the result for another, and the UP result is NOT the DOWN result. Where a key has not been measured yet, it is marked **`UNTESTED`** (an open backlog item) rather than inheriting another key's number.

A note on "combined / symmetric" books: most models trade BOTH sides and report a single pooled accuracy. That pooled number is shown per timeframe as **context only** (labelled `COMBINED`); it is decomposed into its UP-side and DOWN-side accuracy ONLY where that decomposition was actually run (so far: 60s only). A pooled number does NOT populate the UP or DOWN key.

**Conventions.** Deriv-faithful settlement (mid-to-mid, next-tick entry +1s, ties LOSE), breakeven **0.541**. Splits: bars train 2012-21 / val 2022-23 / test 2024 / test 2025 / oos 2026; tick train 2021-23 / val 2024-H1 / test 2024.09-2025.11 / oos 2026. Selection on VAL worst-half (never VAL-acc-max; `corr(VAL,OOS)=−0.54`). Moved-bars-only; per-year CI95. Deriv EURUSD forex Rise/Fall **minimum expiry = 15m** → 1s/1m/5m/10m are research/synthetic-index horizons; 15m/30m are deriv-tradeable. Last updated 2026-06-01.

---

## MASTER KEY TABLE — one row per (EURUSD, timeframe, side)

One row per key = **best OOS % | model id · content_id | description**. `OOS = 2026` (the strict held-out year; test = 2024–2025). The model id resolves to `books/<id>.manifest.json` (verbatim-recreatable; `content_id` = byte-unique hash of the artifact hashes). "MEASURED" = side-split actually run; "untested" = only the COMBINED book exists (shown as the best-available proxy — its OOS is NOT a true side number).

| Key (currency · timeframe · side) | **Best OOS % (2026)** | Model id · content_id | Description | Status |
|---|---|---|---|---|
| EURUSD · **1–5s** · UP | **~0.657** | `EURUSD.tick3.v1` · `e6bbc74604746959` | Tick microstructure 3s GBM ensemble (lgb+xgb+cat) | untested (combined) |
| EURUSD · **1–5s** · DOWN | **~0.657** | `EURUSD.tick3.v1` · `e6bbc74604746959` | Tick microstructure 3s GBM ensemble (lgb+xgb+cat) | untested (combined) |
| EURUSD · **60s** · UP | **0.613** ✅ | `EURUSD.min1.v1` · `d8b2c32c63ada163` (up-filter) | Up-only filter on 3-model ensemble + compression-release specialist, reversion-gated | **MEASURED** |
| EURUSD · **60s** · DOWN | **0.516** ❌ | `EURUSD.min1.v1` · `d8b2c32c63ada163` (down-preds) | Same 3-model ensemble, down-side predictions (dead) | **MEASURED** |
| EURUSD · **2m** · UP | **0.555** ⚠ uncertified | `EURUSD.min2.v1` · `ace4d5c6befbda45` (up-preds) | 3-model ensemble + comp-release specialist, reversion-gated; up-preds. **Best point estimate but NOT a certified edge** (see ‡) | MEASURED, uncertified |
| EURUSD · **2m** · DOWN | **0.540** ❌ | `EURUSD.min2.v1` · `ace4d5c6befbda45` (down-preds) | Same ensemble, down-preds — **sub-breakeven in ALL 3 years** (.540/.512/.540 < 0.541) | MEASURED, dead |

‡ **Adversarial verification (3-agent audit, 2026-06-01) downgraded the 2m UP claim from "robust" to "uncertified":** no per-year UP CI95-lower clears breakeven (.486/.536/.508); pooled UP binomial p=0.053 (fails 5%); ties-strict UP = **0.545** (the 0.555 drops 8 ties = 1.8%) with CI-lower 0.499; the UP side is `pred≡1` so `acc ≡ up-rate of bet-up bars` — a conditional **base-rate/drift, not two-sided skill**; 2025 combined up-rate **0.534 breaches the [0.47,0.53] mirage tripwire** in the highest-n year; and the structurally-identical 15m pipeline deflated 0.647→0.5455 (p10 0.531, FAIL) under faithful CPCV. The online-ARF keystone (an adaptive model at ~0.50 AUC every year) confirms 2m direction is **genuinely efficient**. **CPCV CONFIRMATION** (`min2_cpcv_result.json`, ties-strict purged-combinatorial + block-bootstrap on the frozen UP trades): pooled **0.5445**, block-boot CI95 **[0.521, 0.569]**, 28-path **p10 0.524**, only **57% of paths clear 0.541**, per-year-strict 2024 **0.517** (below breakeven). **NOT CERTIFIED.** **Honest verdict: no robustly-certified 2m direction edge exists; min2 UP ~0.55 is the best point estimate but fails certification — marginal/uncertified.** (No leakage found; numbers re-derived exact.) See `min2_cpcv_result.json`, `sweeps/EURUSD_2m.md`.
| EURUSD · **5m** · UP | **~0.55–0.57** ✅ refit-certified @gate (fwd .605/.577/.615) | `EURUSD.m5xp.v1` · `39e4fedbb43e1d24` (up-preds) | Cross-pair USD-residual + OF + meta, UP-only, gate `sess_ny & meta≥0.5738` (~5% NY cov). **CERTIFIED under FULL per-fold refit CPCV at the operating gate**: cov0.05 p10 **0.553**, 96% of 28 purged-refit folds clear 0.541 (cov0.10 .544/89%; cov0.15 .541/89%); fails only at loose cov0.30. The BEST 5m algo + first sub-15m direction edge to survive the full refit. Deploy ⅛-Kelly, size on the .553 floor; skip win-rate kill-switch. ⚠ **ACI DOWNGRADED (2026-06-01):** the gold-standard NESTED-REFIT CPCV (`m5_aci_cpcv.py`, refits BOTH primary+meta per fold, n=876k) shows `EURUSD.m5xp_aci.v1` does NOT survive — ACI p10 .521–.523, only 54–61% of 28 paths clear, WORSE than fixed on the tail; and the DEPLOYED meta gate (q0.95) is itself only a near-miss (p10 .5376, 78.6%). The certified deployable point is the **TIGHT PRIMARY-CONFIDENCE cover** (cov0.05 p10 .553), NOT the meta gate or ACI; the earlier ACI 2025 .584 was a single-split forward-regime artifact. See `sweeps/EURUSD_5m.md`. ⚠ **V1 MULTIPLE-TESTING HAIRCUT (2026-06-02, `m5_mt_haircut_result.json`):** procedure VALID (all kills flagged non-sig); the POOLED edge (t=5.81, Šidák-p@N̂ 7.3e-08, HLZ t≥3) survives deflation decisively and the refit-CPCV cert is untouched — BUT the **single-binding-2025 headline .577 is multiplicity-inflated** (BY-adj p **0.158**>0.05, Holm .137, fails HLZ t≥3 with t=2.67; robust to M). **Use the refit-CPCV floor .553 as the durable UP figure, not the .577 binding headline.** ⚠ **V2 CSCV/PBO (2026-06-02, `m5_cscv_pbo_result.json`):** agrees with V1 — PBO **0.242**>0.05 and DSR@N̂ **0.934**<0.95 trigger the falsifier, but the driver is operating-point SELECTION instability (OOS-degradation slope **−0.505** = the corr(VAL,OOS)=−0.54 anti-transfer), NOT edge-absence: the IS-best clears breakeven in **96.9%** of CSCV splits (IS-best OOS win-rate p10 **0.589**), MinBTL passes (n153>118). Frozen tight-cover pools to .647 (overstated). **Both validation levers converge: edge is real + robustly profitable; size on the refit floor .553, never the optimistic operating point.** | **MEASURED — UP refit-CPCV-certified (durable floor .553); headline .577/.647 selection/multiplicity-inflated (V1+V2); ACI/meta-gate downgraded** |
| EURUSD · **5m** · DOWN | **~0.54** ⚠ marginal — best via POW=0.5 magweight (.605/**.559**/.549) | `EURUSD.m5xp.v1` down-preds → **POW=0.5 magweight rescue** (book freeze pending) | Symmetric pipeline (side-split, A8b, D3, conf-curve, A8c specialist) left DOWN dead: unweighted refit cov0.05 p10 .5421 mechanically certifies BUT forward binding 2025 **.538 fails**. **B1 |return|-weighted retrain (POW=0.5, `m5_magweight_cpcv.py`, 2026-06-01) is the first DOWN to BOTH CPCV-certify AND point-clear the forward binding year:** DOWN cov0.05 p10 **.5441**/89.3%, mean .5569; forward 2025 DOWN **.559**[.533,.586] (vs incumbent's failing .533), 2024 .605[.577,.634], up-rate tripwire clean. RAZOR-THIN — certifies only at cov0.05 (fails .10/.15), forward 2025 CI-lo .533 grazes below breakeven. Deployable only as a thin tight-cover edge, NOT robust. `m5_magweight_cpcv_result.json` ⚠ **§3 IMPROVE APPLIED + FAILED (2026-06-02, `m5_magweight_seedens_result.json`):** seed-ensemble (K=4) on the magweight-DOWN — the highest-EV variance-reduction lever — does NOT certify DOWN: forward 2025 ~**.52** at every cov (CI-lo .498), same 2025-regime wall. DOWN genuinely exhausted on-disk EMPIRICALLY; the binding 2025 USD-regime is the wall; only remaining lever (risk-reversal sign) is verified-PAYWALLED. **NO certified/deployable DOWN edge exists.** | **MEASURED, marginal (magweight); §3 improve applied+failed → exhausted, non-deployable** |
| EURUSD · **10m** · UP | **0.594** | `EURUSD.m10.v1` · `05fd0a85e07b50fb` | Native-10 3-model ensemble × 5m_bb_width-NY gate | untested (combined) |
| EURUSD · **10m** · DOWN | **0.594** | `EURUSD.m10.v1` · `05fd0a85e07b50fb` | Native-10 3-model ensemble × 5m_bb_width-NY gate | untested (combined) |
| EURUSD · **15m** · UP | **0.663** | `EURUSD.m15.v1` · `67370590c2293e0d` | 3-model ensemble × 15m compression(bb_width)×NY | untested (combined) |
| EURUSD · **15m** · DOWN | **0.663** | `EURUSD.m15.v1` · `67370590c2293e0d` | 3-model ensemble × 15m compression(bb_width)×NY | untested (combined) |
| EURUSD · **30m** · UP | **0.546** | `EURUSD.m30.v1` · `a17be49b9262668f` | 3-model ensemble × 1h-compression×NY | untested (combined) |
| EURUSD · **30m** · DOWN | **0.546** | `EURUSD.m30.v1` · `a17be49b9262668f` | 3-model ensemble × 1h-compression×NY | untested (combined) |

Robustness caveats on the single 2026 number (the program selects on per-year+CI, not one year): **15m** 0.663 is the 2026 slice; the cross-era CPCV-faithful headline is **0.579** (p10 0.557) — use 0.579 as the durable figure. **5m** `m5xp` 0.606 beats the cross-horizon stack `EURUSD.m5stack.v1` (`cf20bf9f16bf0f64`, 2026 0.571) on OOS-2026, though the stack has a higher 3-year *combined* (0.613). **1–5s** ~0.657 is from script docstrings/findings (no result-JSON; not deriv-tradeable, needs a tick venue). Only the two **60s** rows are true side-split numbers; every "untested (combined)" row's OOS is the both-sides book, not a measured UP or DOWN. See `MODEL_REGISTRY.md` + `books/INDEX.json`.

**6 of the 12 side-keys are now MEASURED** (60s UP 0.613 / DOWN 0.516; 2m UP 0.555 uncertified / DOWN 0.540 dead; **5m UP 0.577 CERTIFIED / DOWN 0.533 dead**). Every "by-side UNTESTED" key shows the best COMBINED book + its OOS as the current best-available number; the true side-specific OOS is unknown until that book's trades are split by predicted side. **Do NOT treat the combined OOS as the UP or DOWN number, and do NOT copy one timeframe's asymmetry to another** — the UP-only-edge pattern (dip-buy) recurs (60s 2025-26, 5m all-yrs, 15m) but is horizon- AND regime-specific; at 60s it was a wash in 2024 while at 5m it clears all three years.

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
| **(EURUSD, 5m, UP)** | **0.577 binding ✅ CERTIFIED** (.605/.577/.615; CPCV p10 .576) | Side-split of the frozen `EURUSD.m5xp.v1` book (`m5_updown.py`). All 3 yrs moved-acc CI95-lo clear 0.541 (2024 [.579,.629]/2025 [.552,.603] n1379/2026 [.556,.675]); up-rate clean .504/.523/.529; COMBINED reproduces book 0.607/0.555/0.606 exactly. CPCV (`m5_cpcv.py`): 28/28 purged-combinatorial paths clear 0.541, p10 .576, block-boot CI-lo .575 — **survives the exact test that downgraded 2m UP to uncertified**. `m5_updown_result.json`, `m5_cpcv_m5xp_result.json`. ⚠ **V1 haircut (2026-06-02):** pooled edge t=5.81 survives MT-deflation decisively, but the **binding-2025 .577 alone is multiplicity-inflated** (BY-adj p 0.158, fails HLZ t≥3); the durable single-year figure is the refit floor **.553** under full-family multiplicity, not the .577 headline. `m5_mt_haircut_result.json`. |
| **(EURUSD, 5m, DOWN)** | **0.559 ⚠ marginally rescued** (POW=0.5 magweight: .605/.559/.549; CPCV cov0.05 p10 .5441/89%) | Unweighted book DOWN was dead (binding 2025 .533 < breakeven). **B1 |return|-weighted retrain (POW=0.5, `m5_magweight_cpcv.py`) is the new DOWN leader** — first to BOTH CPCV-certify at cov0.05 (p10 .5441, 89% paths) AND point-clear the forward binding year (.559 vs .533), up-rate clean. RAZOR-THIN: cov0.05-only, forward 2025 CI-lo .533 grazes breakeven. Best DOWN to date; not robust. `m5_magweight_cpcv_result.json`, `m5_magweight_result.json`. |

Conclusion (combined): frontier 0.566→0.648; **0.613 verifiable / 0.648 thin**; capped by the 15m parent; >0.65 not achieved. **Side-split + EXHAUSTIVE SWEEP (2026-06-01): the (5m,UP) side of `EURUSD.m5xp.v1` is the BEST 5m algo and the first sub-15m direction edge to survive the FULL per-fold refit CPCV (the test that deflated 15m).** DOWN is dead (2025 0.533). Full deployment spec, sizing, kill-switch analysis, equity path → `sweeps/EURUSD_5m.md` FINAL CONCLUSION.

**The verdict was corrected twice (I over-claimed, then over-downgraded); final, evidence-locked:** the frozen-trade CPCV (p10 0.576, 28/28) does NOT refit → blind to selection overfitting (over-stated). The cov-0.30 full-refit (p10 0.534) is too loose → the model has no edge at 30% coverage (under-stated). The **tight-cov full-refit at the book's actual ~5% operating gate CERTIFIES: p10 0.553, 96% of 28 purged-refit folds clear 0.541** (cov0.10 .544/89%, cov0.15 .541/89%). So the edge is real and refit-robust **in the high-confidence tail**, magnitude DEFLATED from the single-split: **deployable ≈ 0.55–0.57 (robust floor 0.553), optimistic 0.58–0.61.**

**Sweep findings:** (a) Keystone (online-ARF single-pair TA) ~0.51 every year; the UP edge lives ONLY in cross-pair+NY+meta+UP structure. (b) 76-candidate signed-lag family probe ~0.51 standalone → **the UP edge is the nonlinear GBM COMBINATION** of cross-pair signed features, none predictive alone (pre-killed N10/N11/N14/N15/N16/N17). (c) DOWN dead: m5xp .533, A8b .558 (CI-lo .516), D3 USD-strength .526 — down-moves are jump/informed-dominated (magnitude not sign). (d) All else null/subsumed (B3a CKS .50, B5a per-side flow .51, F3a ES sign-flip, N12/N13 irrev .51; 24 audit-subsumed rows). (e) **Deployment:** gate is the win-rate knee + Kelly growth peak + lowest gate where the binding year clears breakeven; size ⅛-Kelly on the .553 floor; the win-rate kill-switch costs more than it saves (fractional stake is the real protection). Citations: `m5_refit_tightcov_result.json`, `m5_cpcv_refit_result.json`, `m5_kelly_result.json`, `m5_equity_result.json`, `m5_confcurve_result.json`, `m5_online_result.json`, `m5_legsign_result.json`, `m5_downcond_result.json`.

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
**Future agents: invoke the `strategy-eval` skill** (`.claude/skills/strategy-eval/SKILL.md`) — it encodes the mandatory deriv-faithful evaluation discipline, the 7 leakage traps, the registry-freeze step, and this results-writing protocol as one workflow. Then:
1. Each experiment → add a row to the timeframe's combined-book table (method, file, per-year stats + CI, verdict).
2. If side-split, update that timeframe's **Key results** AND the **MASTER KEY TABLE** (one row per side). Unseat a key's leader only if the new method beats the incumbent's binding-year stat (worst held-out year) with CI95-lower clearing it, under deriv-faithful moved-bars discipline; move the old leader to a "previous" line.
3. Keep every number traceable to a result JSON or research-log line (Tier-1). Flag thin-coverage (n<50) and VAL-acc-max numbers as non-robust.
4. Magnitude (|ret|) is sign-invariant → it has no up/down key; record it in `MAGNITUDE_FINDINGS.md`.
5. New currencies get their own `<PAIR>_RESULTS.md` with this exact structure and key.

---

## DEPLOYMENT SPEC — EURUSD 5m, both sides (FINAL, 2026-06-02, post V1/V2 honest correction)

**⚠ VENUE (verified): the 5m EURUSD edge is NOT directly tradeable on deriv FOREX.** Deriv forex Rise/Fall
minimum duration = 15m (repo-documented + prior-verified; the live deriv glossary confirms duration is
per-trade-type/asset and the authoritative check is the `contracts_for(frxEURUSD)` API — synthetics allow 1-tick
but are NOT EURUSD). **So 5m is RESEARCH-grade / horizon-mapping; it needs a ≤5m-expiry EURUSD binary venue
before any live trade. The deriv-deployable EURUSD direction keys are 15m/30m.** Spec below assumes a ≤5m venue
at R≈0.85 / breakeven 0.541; confirm the venue first. Sources: deriv.com glossary + /markets/forex (fetched 2026-06-02).

### UP — `EURUSD.m5xp.v1` up-preds (the deliverable; refit-CPCV-CERTIFIED)
- **Operating gate:** `sess_ny & meta≥0.5738`, then the TIGHT primary-confidence cover **cov0.05** (NOT the meta
  gate alone, NOT ACI — `m5xp_aci.v1` ACI is DOWNGRADED: fails nested-refit CPCV, p10 .521–.523).
- **Robust floor (SIZE ON THIS):** refit-CPCV **p10 = 0.553**, 96% of 28 purged-refit folds clear 0.541. Forward
  per-year .605/.577/.615 and the binding-year .577 / frozen-tight-cover .647 are **multiplicity/selection-inflated
  (V1 BY-adj p .158, fails HLZ t≥3; V2 PBO .242)** — do NOT size on them. Pooled edge is real (V1 t=5.81).
- **Coverage:** ~cov0.05 of NY 5m bars ≈ **1.4 trades/NY-hr (~3–4/session/day)**; med_n 2126 over test+oos.
- **Confidence/coverage curve** (`m5_kelly_result.json`): Kelly-positive only at the tight cover (thr≳0.569,
  bind_win .573, ¼-Kelly qf .018); looser covers → bind_win<breakeven → qf=0 (no bet). Tighter = higher win, lower n.
- **Fractional-Kelly on the floor (0.553, R=0.85):** full-Kelly f*=(pR−(1−p))/R = **2.7%**; deploy **⅛-Kelly ≈ 0.34%/bet**
  (¼-Kelly ≈ 0.68% if aggressive). EV/bet at the floor ≈ **+2.3% of stake**. (The earlier `m5_equity` 1.07%/bet was
  sized on the optimistic 0.58 belief — the honest floor sizing is ~⅓ of that.)
- **Equity path / risk (`m5_equity_result.json`, real frozen trades @ optimistic stake):** forward end 25.9×, **max-DD
  22%**, **longest losing streak 8**; STRESS-regime max-DD 35.8%. Floor-sized (⅓ stake) → max-DD ≈ **7–12%**, streak 8
  (streak is sizing-independent). 2025 is the binding regime.
- **Kill-switch — TESTED, DO NOT USE:** win-rate kill-switch (window150/kill .52/resume .55) HURTS — end 19.4× vs 25.9×
  no-KS, 6.8% flat, 4 kills (`m5_equity_result.json`). Tested, not assumed; deploy WITHOUT it.

### DOWN — `EURUSD.m5xp_magw_down.v1` (POW=0.5 magweight; MARGINAL, NOT robustly deployable)
- Refit-CPCV cov0.05 p10 **0.5441** (89% folds clear) — barely above breakeven 0.541; forward 2025 .559 but CI-lo
  **.533 grazes below breakeven**. V1/V2 imply DOWN is even more multiplicity-fragile (found via a POW×cov sub-search).
- **Floor (0.5441) sits AT breakeven → near-zero Kelly edge → effectively NON-deployable.** Honest verdict: a real but
  razor-thin marginal edge; ship as "measured, marginal," not as a deployable book. Rally-selling DOWN remains the open gap.
- **§3 loss-reoptimization exhausted (2026-06-03, `m5_lossbatch_result.json`):** focal-γ2 / quantile-median (distribution-head)
  / asymmetric class-weight (×1.5,×2) ALL fail — no DOWN binding-year CI-lo clears .541 (best DOWN-2025 .5407, CI-lo .524).
  The objective is not the binding constraint; loss family joins magweight/seed-ens/rankloss. DOWN exhaustion now spans
  features, model class, ensembling, calibration+ACI, hyperparameter tuning (Optuna), AND loss design — all hit the 2025 wall.

### Registry notes (to reconcile)
- `EURUSD.m5xp_magw_down.v1` is git-tagged but MISSING from `books/INDEX.json` — added below.
- `EURUSD.m5xp_aci.v1` is registered with ACI .584 metrics but **ACI is DOWNGRADED** (nested-refit fail); the durable
  UP figure is the m5xp refit floor **0.553**, not ACI. Registry metric annotated.
