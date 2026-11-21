> **SCOPE: EURUSD** (key-specific — all timeframes × sides; the results of record + UP/DOWN leaderboard). Generic methods: docs/METHODS_CATALOG.md. See REPO_MAP.md.

# EURUSD — Results Ledger (unique key: currency × timeframe × side)

**Pair tag: `EURUSD`.** (Other currencies get their own `<PAIR>_RESULTS.md`.)

**The unique result key is `(currency, timeframe, side)` where side ∈ {UP, DOWN}.** Every key has its own result — the result for one timeframe is NOT the result for another, and the UP result is NOT the DOWN result. Where a key has not been measured yet, it is marked **`UNTESTED`** (an open backlog item) rather than inheriting another key's number.

A note on "combined / symmetric" books: most models trade BOTH sides and report a single pooled accuracy. That pooled number is shown per timeframe as **context only** (labelled `COMBINED`); it is decomposed into its UP-side and DOWN-side accuracy ONLY where that decomposition was actually run (so far: 60s only). A pooled number does NOT populate the UP or DOWN key.

**Conventions.** Deriv-faithful settlement (mid-to-mid, next-tick entry +1s, ties LOSE), breakeven **0.541**. Splits: bars train 2012-21 / val 2022-23 / test 2024 / test 2025 / oos 2026; tick train 2021-23 / val 2024-H1 / test 2024.09-2025.11 / oos 2026. Selection on VAL worst-half (never VAL-acc-max; `corr(VAL,OOS)=−0.54`). Moved-bars-only; per-year CI95. Deriv EURUSD forex Rise/Fall **minimum expiry = 15m** → 1s/1m/2m/5m/10m are research/synthetic-index horizons; 15m/30m are deriv-tradeable. Last updated 2026-06-07 (**Novel-methods campaign**: HAR/realized-vol magnitude canon tested as additive arms on the certified base — REAL-but-SUB-BAR, all arms <+0.005 fwd, best +all +0.0021 3/3 deployable but economically negligible; the certified base magnitude edge RE-VALIDATED on a clean forward holdout with NO decay; fractional-differentiation (FFD, T2) direction into the cross-pair book KILLED at 15m AND 30m. **No leader unseated.** See the 2026-06-07 dated section below. Prior 2026-06-06 entry: DST-correct **session segmentation** — the certified ≥10m cross-pair direction edge is decisively **NY-session-concentrated** — strict session-only NY recertifies BOTH sides at 10m/15m/30m and BEATS the legacy fixed-UTC gate, while LDN/Asia certify at NONE; corrected Kronos is NULL at every horizon/session; full-suite audit found the Kronos look-forward bug ISOLATED to 2 scripts and invalidates NO certified book. Prior 2026-06-04 entry: 30m two-sided certification — cross-pair keystone CERTIFIES BOTH 30m sides, refit-CPCV UP p10 .5588 / DOWN p10 .5525, 15/15, `EURUSD.m30xp.v1`; refit-dependent. Gradient now none@60s→UP@5m→BOTH@10m,15m,30m).

---

## MASTER KEY TABLE — one row per (EURUSD, timeframe, side)

> **Deriv-feed program (2026-11-15):** the numbers in this table are Dukascopy-era evidence. On the Deriv feed (DEV 2025-08-11..2026-07-31) none of this pair's frozen 15m/30m books is SUPPORTED as deployed — see "EURUSD — DERIV-FEED PROGRAM" at the end of this file.

One row per key = **best OOS % | model id · content_id | description**. `OOS = 2026` (the strict held-out year; test = 2024–2025). The model id resolves to `books/<id>.manifest.json` (verbatim-recreatable; `content_id` = byte-unique hash of the artifact hashes). "MEASURED" = side-split actually run; "untested" = only the COMBINED book exists (shown as the best-available proxy — its OOS is NOT a true side number).

| Key (currency · timeframe · side) | **Best OOS % (2026)** | Model id · content_id | Description | Status |
|---|---|---|---|---|
| EURUSD · **1–5s** · UP | **~0.657** | `EURUSD.tick3.v1` · `e6bbc74604746959` | Tick microstructure 3s GBM ensemble (lgb+xgb+cat) | untested (combined) |
| EURUSD · **1–5s** · DOWN | **~0.657** | `EURUSD.tick3.v1` · `e6bbc74604746959` | Tick microstructure 3s GBM ensemble (lgb+xgb+cat) | untested (combined) |
| EURUSD · **60s** · UP | **0.613** ⚠ uncertified | `EURUSD.min1.v1` · `d8b2c32c63ada163` (up-filter) | Up-only filter on 3-model ensemble + comp-release specialist, reversion-gated. **Best AVAILABLE 60s UP but NOT CPCV-certified** (`min1_cpcv.py`: ties-strict .573, block-boot CI-lo .530, path p10 .524, 2024 .520<breakeven) — regime-dependent dip-buy, near-miss like 2m UP | **MEASURED, uncertified** |
| EURUSD · **60s** · DOWN | **0.516** ❌ | `EURUSD.min1.v1` · `d8b2c32c63ada163` (down-preds) | Same ensemble, down-preds — **dead + honestly EXHAUSTED on-disk** (CPCV p10 .474; D3a/magweight/mag-bridge killed + 2 dry discovery rounds, rows 18-21); external (risk-reversal/rate-diff) only redirect | **MEASURED, dead/exhausted** |
| EURUSD · **2m** · UP | **0.555** ⚠ uncertified (best AVAILABLE) / honestly EXHAUSTED | `EURUSD.min2.v1` · `ace4d5c6befbda45` (up-preds) | 3-model ensemble + comp-release specialist, reversion-gated; up-preds. Best point estimate, NOT a certified edge (see ‡). **EXHAUSTED 2026-06-04 under full (a)-(e):** every cert-grade test fails — frozen-book CPCV p10 .524, **cross-pair refit-CPCV p10 .5096** (1/15), **RFF .4997**, **specialist nested-refit .4957** (0/15). | MEASURED, uncertified, **exhausted** |
| EURUSD · **2m** · DOWN | **~0.52** ❌ dead + honestly EXHAUSTED | `EURUSD.min2.v1` · `ace4d5c6befbda45` (down-preds) | Same ensemble, down-preds — sub-breakeven ALL years (.540/.512/.540). **EXHAUSTED 2026-06-04:** cross-pair refit-CPCV p10 **.5124** (0/15), specialist nested-refit **.4912** (0/15), frozen DOWN specialist killed. No DOWN edge at any cover. | MEASURED, dead/**exhausted** |

‡ **Adversarial verification (3-agent audit, 2026-06-01) downgraded the 2m UP claim from "robust" to "uncertified":** no per-year UP CI95-lower clears breakeven (.486/.536/.508); pooled UP binomial p=0.053 (fails 5%); ties-strict UP = **0.545** (the 0.555 drops 8 ties = 1.8%) with CI-lower 0.499; the UP side is `pred≡1` so `acc ≡ up-rate of bet-up bars` — a conditional **base-rate/drift, not two-sided skill**; 2025 combined up-rate **0.534 breaches the [0.47,0.53] mirage tripwire** in the highest-n year; and the structurally-identical 15m pipeline deflated 0.647→0.5455 (p10 0.531, FAIL) under faithful CPCV. The online-ARF keystone (an adaptive model at ~0.50 AUC every year) confirms 2m direction is **genuinely efficient**. **CPCV CONFIRMATION** (`min2_cpcv_result.json`, ties-strict purged-combinatorial + block-bootstrap on the frozen UP trades): pooled **0.5445**, block-boot CI95 **[0.521, 0.569]**, 28-path **p10 0.524**, only **57% of paths clear 0.541**, per-year-strict 2024 **0.517** (below breakeven). **NOT CERTIFIED.** **Honest verdict: no robustly-certified 2m direction edge exists; min2 UP ~0.55 is the best point estimate but fails certification — marginal/uncertified.** (No leakage found; numbers re-derived exact.) See `min2_cpcv_result.json`, `sweeps/EURUSD_2m.md`.

‡‡ **2026-06-04 — KEYSTONE GAP CLOSED + both sides EXHAUSTED under the full (a)-(e) pipeline.** The prior closure never actually ran **cross-pair feature-row POOLING** (the lever that certified 5m UP and BOTH 15m sides) at 2m — it was wrongly subsumed under the *different* cross-leg-sign-lead mechanism. Now RUN: (1) **cross-pair TIES-STRICT refit-CPCV** (`min2_xpair_cpcv_result.json`, 15 purged paths, per-fold gate refit) — UP p10 **.5096** (1/15 clear), DOWN p10 **.5124** (0/15), moved-AUC .517, up-rates [.482,.524]; the probe's frozen-VAL gate gave .6126/.5603 — an **overfit-gate artifact that dissolved under per-fold refit**. (2) **RFF virtue-of-complexity SDF** (`min2_rff_result.json`, 4th model class, P~T random nonlinear basis) — VAL AUC **.4997** (coin-flip): the channel is empty across FOUR model classes, not a capacity artifact. (3) **purpose-built UP/DOWN meta-labeler specialist** (step c, `min2_spec_result.json` + nested-refit `min2_spec_cpcv_result.json`) — DOWN frozen-killed; UP frozen-survived (.575/.5525/.6009) then **collapsed to p10 .4957 / .4912 (0/15 both sides) under nested refit** (overfit-gate). (4) mag×direction bridge already RUN at 120s (`docs/EXPERIMENT_LEDGER.md` 41-42) — HURTS. **Both sides honestly EXHAUSTED on-disk; the cross-pair edge gradient none@60s→UP@5m→both@15m does NOT reach back to 2m (USD-common-factor needs ≥5m). Only redirect = external data (Tier-G, gated). Note: 120s is below deriv's 15m forex minimum → research horizon, not deployable.**
| EURUSD · **5m** · UP | **~0.55–0.57** ✅ refit-certified @gate (fwd .605/.577/.615) | `EURUSD.m5xp.v1` · `39e4fedbb43e1d24` (up-preds) | Cross-pair USD-residual + OF + meta, UP-only, gate `sess_ny & meta≥0.5738` (~5% NY cov). **CERTIFIED under FULL per-fold refit CPCV at the operating gate**: cov0.05 p10 **0.553**, 96% of 28 purged-refit folds clear 0.541 (cov0.10 .544/89%; cov0.15 .541/89%); fails only at loose cov0.30. The BEST 5m algo + first sub-15m direction edge to survive the full refit. Deploy ⅛-Kelly, size on the .553 floor; skip win-rate kill-switch. ⚠ **ACI DOWNGRADED (2026-06-01):** the gold-standard NESTED-REFIT CPCV (`m5_aci_cpcv.py`, refits BOTH primary+meta per fold, n=876k) shows `EURUSD.m5xp_aci.v1` does NOT survive — ACI p10 .521–.523, only 54–61% of 28 paths clear, WORSE than fixed on the tail; and the DEPLOYED meta gate (q0.95) is itself only a near-miss (p10 .5376, 78.6%). The certified deployable point is the **TIGHT PRIMARY-CONFIDENCE cover** (cov0.05 p10 .553), NOT the meta gate or ACI; the earlier ACI 2025 .584 was a single-split forward-regime artifact. See `sweeps/EURUSD_5m.md`. ⚠ **V1 MULTIPLE-TESTING HAIRCUT (2026-06-02, `m5_mt_haircut_result.json`):** procedure VALID (all kills flagged non-sig); the POOLED edge (t=5.81, Šidák-p@N̂ 7.3e-08, HLZ t≥3) survives deflation decisively and the refit-CPCV cert is untouched — BUT the **single-binding-2025 headline .577 is multiplicity-inflated** (BY-adj p **0.158**>0.05, Holm .137, fails HLZ t≥3 with t=2.67; robust to M). **Use the refit-CPCV floor .553 as the durable UP figure, not the .577 binding headline.** ⚠ **V2 CSCV/PBO (2026-06-02, `m5_cscv_pbo_result.json`):** agrees with V1 — PBO **0.242**>0.05 and DSR@N̂ **0.934**<0.95 trigger the falsifier, but the driver is operating-point SELECTION instability (OOS-degradation slope **−0.505** = the corr(VAL,OOS)=−0.54 anti-transfer), NOT edge-absence: the IS-best clears breakeven in **96.9%** of CSCV splits (IS-best OOS win-rate p10 **0.589**), MinBTL passes (n153>118). Frozen tight-cover pools to .647 (overstated). **Both validation levers converge: edge is real + robustly profitable; size on the refit floor .553, never the optimistic operating point.** | **MEASURED — UP refit-CPCV-certified (durable floor .553); headline .577/.647 selection/multiplicity-inflated (V1+V2); ACI/meta-gate downgraded** |
| EURUSD · **5m** · DOWN | **~0.54** ⚠ marginal — best via POW=0.5 magweight (.605/**.559**/.549) | `EURUSD.m5xp.v1` down-preds → **POW=0.5 magweight rescue** (book freeze pending) | Symmetric pipeline (side-split, A8b, D3, conf-curve, A8c specialist) left DOWN dead: unweighted refit cov0.05 p10 .5421 mechanically certifies BUT forward binding 2025 **.538 fails**. **B1 |return|-weighted retrain (POW=0.5, `m5_magweight_cpcv.py`, 2026-06-01) is the first DOWN to BOTH CPCV-certify AND point-clear the forward binding year:** DOWN cov0.05 p10 **.5441**/89.3%, mean .5569; forward 2025 DOWN **.559**[.533,.586] (vs incumbent's failing .533), 2024 .605[.577,.634], up-rate tripwire clean. RAZOR-THIN — certifies only at cov0.05 (fails .10/.15), forward 2025 CI-lo .533 grazes below breakeven. Deployable only as a thin tight-cover edge, NOT robust. `m5_magweight_cpcv_result.json` ⚠ **§3 IMPROVE APPLIED + FAILED (2026-06-02, `m5_magweight_seedens_result.json`):** seed-ensemble (K=4) on the magweight-DOWN — the highest-EV variance-reduction lever — does NOT certify DOWN: forward 2025 ~**.52** at every cov (CI-lo .498), same 2025-regime wall. DOWN genuinely exhausted on-disk EMPIRICALLY; the binding 2025 USD-regime is the wall; only remaining lever (risk-reversal sign) is verified-PAYWALLED. **NO certified/deployable DOWN edge exists.** | **MEASURED, marginal (magweight); §3 improve applied+failed → exhausted, non-deployable** |
| EURUSD · **10m** · UP | **0.586** refit-CPCV p10 (15/15) ✅ **CERTIFIED** | `EURUSD.m10xp.v1` (cross-pair); base `EURUSD.m10.v1` .561 fallback | Cross-pair USD-residual+OF primary @MX_HOR=10 × 5m_bb_width-NY-conf gate, UP-bets | **CERTIFIED 2026-06-04** (`m10_xpair_cpcv`); improves base; IMPROVE levers pending |
| EURUSD · **10m** · DOWN | **0.568** refit-CPCV p10 (15/15) ✅ **CERTIFIED** | `EURUSD.m10xp.v1` (cross-pair); base `EURUSD.m10.v1` .552 fallback | Cross-pair USD-residual+OF primary @MX_HOR=10 × 5m_bb_width-NY-conf gate, DOWN-bets | **CERTIFIED 2026-06-04** (`m10_xpair_cpcv`; recent-era robust); IMPROVE levers pending |
| EURUSD · **15m** · UP | **.5673 refit-CPCV ✅ CERTIFIED** (cross-pair; 15/15 clear; base-book .5475 fallback) | cross-pair @15m (freeze pending `EURUSD.m15xp.v1`); base `EURUSD.m15.v1`·`67370590c2293e0d` | cross-pair USD-residual+OF primary @MX_HOR=15, UP bets; +1.98pt over base book | **MEASURED — CERTIFIED (refit p10 .5673), cross-pair LEADER; deriv-deployable** |
| EURUSD · **15m** · DOWN | **.5742 refit-CPCV ✅ CERTIFIED** (cross-pair; 15/15 clear; base-book .5486 fallback) | cross-pair @15m (freeze pending `EURUSD.m15xp.v1`); base `EURUSD.m15.v1` | cross-pair primary @MX_HOR=15, DOWN bets; +2.56pt over base book | **MEASURED — CERTIFIED (refit p10 .5742); BEST certified DOWN edge in program; deriv-deployable** |
| EURUSD · **30m** · UP | **.5588 refit-CPCV ✅ CERTIFIED** (cross-pair; 15/15 clear; base-book .5538 fallback) | `EURUSD.m30xp.v1`·`da55d7c6b3b08bef` (up-bets); base `EURUSD.m30.v1`·`a17be49b9262668f` | cross-pair USD-residual+OF primary @MX_HOR=30 × 1h_bb_width-NY-conf gate, UP bets; +0.50pt over base. **REFIT-DEPENDENT** (frozen-2021 fwd decays: 2026 UP .5302@cov5/.4652@cov10) | **MEASURED — CERTIFIED (refit p10 .5588), cross-pair leader; deriv-deployable w/ retrain** |
| EURUSD · **30m** · DOWN | **.5525 refit-CPCV ✅ CERTIFIED** (cross-pair; 15/15 clear; base near-miss .5356) | `EURUSD.m30xp.v1`·`da55d7c6b3b08bef` (down-bets); base `EURUSD.m30.v1` | cross-pair primary @MX_HOR=30, DOWN bets; rescued base near-miss .5356→.5525 (+1.69pt). **REFIT-DEPENDENT** (frozen fwd 2026 DOWN .5778 n45-thin) | **MEASURED — CERTIFIED (refit p10 .5525); deriv-deployable w/ retrain** |

Robustness caveats on the single 2026 number (the program selects on per-year+CI, not one year): **15m** 0.663 is the 2026 slice; the cross-era CPCV-faithful headline is **0.579** (p10 0.557) — use 0.579 as the durable figure. **5m** `m5xp` 0.606 beats the cross-horizon stack `EURUSD.m5stack.v1` (`cf20bf9f16bf0f64`, 2026 0.571) on OOS-2026, though the stack has a higher 3-year *combined* (0.613). **1–5s** ~0.657 is from script docstrings/findings (no result-JSON; not deriv-tradeable, needs a tick venue). Only the two **60s** rows are true side-split numbers; every "untested (combined)" row's OOS is the both-sides book, not a measured UP or DOWN. See `MODEL_REGISTRY.md` + `books/INDEX.json`.

### 2026-06-08 — Neural-forecaster + spectral DIRECTION sweep (all tf): ALL KILLED, no leader unseated

Executed the three "technically-uncoded" lever families flagged when mining external repos (`directional-prediction`,
`n-hits`) at **every** EURUSD timeframe 1/2/5/10/15/30m, on GPU, deriv-faithful (forward-holdout, per-year 2024/25/26,
selacc@cov0.10, ties-dropped, moved-bars, breakeven 0.541, 2026 OOS; KILL if VAL dirAUC≤0.515 OR no yr CI95-lo≥0.541).
**84 arms across the 3 families × 6 horizons → 0 survivors.** Numbers (Tier-1, `neural_spectral_dir_sweep_result.json`
+ per-`*_dir_*m_result.json`):
- **N-BEATS / N-HiTS** path-forecast→sign, single+cross-pair (`nbeats_nhits_dir.py`): valAUC 0.5035–0.5080 all tf; best held-out selacc **0.5153** (15m); every CI-lo <0.541. KILLED 1/2/5/10/15/30m.
- **DLinear / Autoformer / FEDformer-freq / TFT-quantile**, single+cross-pair (`decomp_dir.py`): valAUC 0.5090–0.5193; best selacc **0.5330** (10m, CI-lo .524). KILLED all tf.
- **causal DWT + SSA** band-split→GBM/literal recombine→sign (`spectral_dir.py`): valAUC 0.4996–0.5155; flashiest point 0.5455 (10m/2026, n847) has **CI-lo 0.5112** (thin-coverage mirage). KILLED all tf.

Confirms the sign-invariance prior (path/spectral forecasters carry SIZE not SIGN) and the prior `binary_alpha`
N-HiTS@5m=0.515 null. `CORPUS_LEVER_INVENTORY.md` rows 333/356/357/359 flipped UNTESTED→tested-killed. **No UP/DOWN
key changed; the deriv-deployable leaders (15m/30m cross-pair) stand.** The mined external repos add nothing net-new
at any horizon — the only remaining DIRECTION frontier stays EXTERNAL data (rate-diff / vol / EURGBP ticks).

**All 12 side-keys are now MEASURED** (60s UP 0.613 **uncertified** [CPCV p10 .524, regime-dependent] / DOWN 0.516 **dead+exhausted** [CPCV p10 .474, 2 dry discovery rounds]; 2m UP 0.555 uncertified+**exhausted** [cross-pair refit p10 .5096, RFF .4997, specialist nested-refit .4957] / DOWN dead+**exhausted** [cross-pair refit p10 .5124, specialist .4912]; **5m UP 0.577 CERTIFIED / DOWN 0.533 dead**; **10m UP .5863 CERTIFIED / DOWN .5683 CERTIFIED — cross-pair leader `EURUSD.m10xp.v1` (refit-CPCV, 15/15 paths both sides), improving base-book side floors .561/.552; like 15m, a BOTH-SIDES-certified horizon (research horizon, 10m<deriv 15m min)**; **15m UP .5673 CERTIFIED / DOWN .5742 CERTIFIED — cross-pair leader (refit-CPCV, 15/15 paths both sides), improving the base-book floors .5475/.5486; the first BOTH-SIDES-certified horizon, deriv-deployable**; **30m UP .5588 CERTIFIED / DOWN .5525 CERTIFIED — cross-pair leader `EURUSD.m30xp.v1` (refit-CPCV, 15/15 paths both sides), improving base floors .5538/.5356 (DOWN rescued from near-miss); the LONGEST deriv-tradeable horizon, REFIT-DEPENDENT (frozen-2021 fwd decays in 2026 OOS)**). Every "by-side UNTESTED" key shows the best COMBINED book + its OOS as the current best-available number; the true side-specific OOS is unknown until that book's trades are split by predicted side. **Do NOT treat the combined OOS as the UP or DOWN number, and do NOT copy one timeframe's asymmetry to another** — the UP-only-edge pattern (dip-buy) recurs at 60s/5m but **does NOT hold at 10m or 15m** (where DOWN certifies as robustly as UP, frac-clear 1.0): asymmetry is horizon- AND regime-specific, confirmed by direct measurement, never copied. The cross-pair USD-common-factor direction edge **strengthens with horizon: none@60s → UP@5m → BOTH@10m, 15m & 30m** (the informed-jump component of moves, esp. DOWN, averages out as the horizon lengthens, so sign becomes forecastable ≥10m). At 30m (longest deriv horizon) both sides certify under refit-CPCV (p10 .5588/.5525, 15/15) but the edge is **REFIT-DEPENDENT** — the frozen-2021 book's forward edge decays by the 2026 OOS year (UP .5302@cov5/.4652@cov10), so deploy with periodic retraining; the refit-CPCV per-era floor is the durable figure.

*Provenance of the OOS numbers:* 60s UP/DOWN — `docs/min1_research_log.md:104-113` (2026 column). 5m 0.571 — `docs/m5_research_log.md` (m5_stack2 q0.98 oos n163). 10m 0.594 — `m10_freeze_honest.py` / `docs/m10_research_log.md`. 15m 0.663 — `m15_production` 2026 (`docs/DIRECTION_FINDINGS.md:15`, `m15_walkforward.py` frozen map). 30m 0.546 — `m30_production.py` (`docs/m30_research_log.md`). 1–5s ~0.65 — `m_tick_prod.py:4-5` (no 2026-only split on disk; TEST+OOS-robust).

**Magnitude** (|ret|≥Q) is sign-invariant by construction → it has NO up/down key. It is the size edge (AUC 0.71–0.81), tracked separately in `docs/MAGNITUDE_FINDINGS.md`, not here.

---

## 2026-06-07 — Novel-methods campaign (HAR/realized-vol magnitude canon + FFD direction + Phase 4 cross-sectional direction)

This session executed the magnitude HAR/realized-vol canon, the fractional-differentiation (FFD) direction transform, AND the Phase 4 cross-sectional direction families (D1/D6/D7) from NOVEL_METHODS_RESEARCH §7, every arm gated by the deployment-faithful FROZEN-PAST FORWARD HOLDOUT (train≤2023 → per-year 2024/25/26), NOT pooled CPCV. **Nothing new certifies; no leader moves.** Every direction family is KILLED forward; the one positive is that the certified base MAGNITUDE edge was RE-VALIDATED on a clean forward holdout with NO decay. New reusable infra committed: `build_panel.py` (clean 7-pair USD return panel, construction-faithful vs certified `build_xp`), `fwd_holdout.py` (the mandatory forward-holdout gate), `surrogate_null.py` (phase-randomize/IAAFT gate), `xsec_direction.py` (Phase 4 cross-sectional harness w/ mechanism-shuffle controls). Full verified facts: `docs/CAMPAIGN_2026-06-07_FACTS.md`.

### MAGNITUDE — HAR/realized-vol canon as additive arms on the certified base (REAL-but-SUB-BAR; base RE-VALIDATED, no decay)
`mag_har.py` + `mag_har_result.json` (recorded `docs/MAGNITUDE_FINDINGS.md` §6g). Arms added to the certified base `[-pe,rv30,rv120]`, forward holdout, horizons 10/15/30m, target |ret_H|≥train-Q75. Falsifier = +0.005 AUC in ≥2 forward years AND ≥2 horizons. **ALL ARMS FAIL — none clears +0.005:**
| Arm | mean fwd ΔAUC | deployable | verdict |
|---|---|---|---|
| +har (multiscale RV) | +0.0010 | 2/3 | sub-bar |
| +jump (bipower split) | +0.0003 | 1/3 | null |
| +semivar (RS⁺/RS⁻/signed-jump) | −0.0000 | 1/3 | null |
| +harq (realized quarticity) | +0.0004 | 2/3 | null |
| **+all (stacked)** | **+0.0021** | **3/3 (no decay)** | forward-CONSISTENT but economically negligible |

Two findings: (1) UNLIKE the §6f time-of-day arm, the stacked additions are forward-CONSISTENT (+all positive in all 9 year×horizon cells) — genuine but tiny; base rv already extracts ~all magnitude (signed-jump SJ120 is the only orthogonal channel and carries nothing). (2) **RE-VALIDATES the certified magnitude edge on a clean deployment-faithful holdout: base AUC .799/.750/.750 (10m) and .791/.739/.737 (30m), 4–5.6× decile lift, NO decay.** VERDICT: KILLED as a deployable upgrade; magnitude path EXHAUSTED on-disk. (Magnitude is sign-invariant → NO up/down key; recorded here as context only, full detail in `docs/MAGNITUDE_FINDINGS.md`.)

### DIRECTION — fractional differentiation (T2) into the cross-pair book (KILLED at 15m AND 30m)
`frac_diff.py` (hand-rolled FFD, fixed-width weights + ADF d*-selection; self-check: random walk needs d*=0.1 to pass ADF while retaining 98.8% level-memory) + `frac_direction.py`. FFD USD factor/residual/lead-lag fed into the certified cross-pair direction book, forward holdout, NY cov0.10 selacc. Per-pair d*: EURUSD/NZD/CHF=0.1, GBP/AUD/JPY/CAD=0.2; thresh 1e-4. Arms {base, +ffd, ffdonly}:
| Horizon | base selacc 2024/2025/2026 (pooled) | +ffd selacc 2024/2025/2026 (pooled) | +ffd deployable | ffdonly | verdict |
|---|---|---|---|---|---|
| **15m** | .5900 / .5520 / .5254 (.5642) | .5792 / .5301 / .5403 (.5537) | false (DECAYS Δ −0.0108/−0.0219/+0.0149) | pooled .5001 = coin flip | ❌ KILLED (`frac_direction_15m_result.json`) |
| **30m** | .5842 / .5483 / .5007 (.5558) | .5546 / .5135 / .5064 (.5308) | false (DECAYS Δ −0.0296/−0.0348/+0.0057) | pooled .4958 (below coin flip) | ❌ KILLED (harder) (`frac_direction_30m_result.json`) |

FFD level-memory adds nothing to direction and HURTS the recent years; standalone FFD is a coin flip. **Note (consistent with documented refit-dependence):** the base cross-pair book's own per-year forward selacc DECAYS **.59 → .55 → .525** at 15m (and .584 → .548 → .501 at 30m) — this is the known non-stationarity of the cross-pair USD-factor edge (frozen-2021 forward decays by the OOS year), exactly why the `EURUSD.m{15,30}xp.v1` certs are flagged REFIT-DEPENDENT (deploy with periodic retrain). FFD direction is DEAD at BOTH 15m and 30m.

### DIRECTION — Phase 4 cross-sectional families (D1 lead-lag signature, D6 HAVOK, D7 signed-semivariance) — ALL KILLED forward
`xsec_direction.py` (+ `xsec_direction_{sig,havok}_{15,30}m_result.json`). Unified harness on the certified panel, forward holdout (direction, NY cov0.10 selacc), arms {base=certified xp book, base+fam, famonly, fam_shuffle (mechanism-specificity control)}. The base book DECAYS forward (15m .5895/.5518/.5234; 30m .5827/.5505/.5055 — the documented refit-dependence) and NOTHING adds to it:
| Family | Mechanism | famonly pooled (15m / 30m) | Mechanism null | +fam vs base | Verdict |
|---|---|---|---|---|---|
| **D1 signature lead-lag** | depth-2 iterated integrals S^{ij}=∫∫dX^i dX^j + Lévy area between EURUSD & each peer (signed lead-lag / quad-covariation) | .5243 / .5283 | ❌ FAILS — sigonly .5243 ≈ sig_shuf .5236 (rotation surrogate doesn't degrade it) | decays (15m Δ −.0043/−.0099/+.0032; 30m −.0094/−.0043/−.0133) | ❌ **KILLED both horizons** — NOT genuine lead-lag (fails mechanism-specificity falsifier) |
| **D6 frozen-basis HAVOK** (Hankel-Koopman) | delay-embed (q=60) USD-factor trend, freeze r=8 SVD modes on TRAIN, causal project to v1..v7 + intermittent FORCING v_r precursor | .5148 / .5197 | barely beats phase-randomized surrogate (.5148 vs .5063) | decays (15m Δ −.003/−.0024/+.0014) | ❌ **KILLED both horizons** — sub-breakeven (far below base .5624), forcing never clears .541 |
| **D7 signed-semivariance** (Patton-Sheppard good/bad vol) | realized signed semivariance RS⁺/RS⁻ asymmetry as directional content | .5258 / .5382 | ✅ **PASSES** — semivaronly beats semivar_shuf (sign-flip control) by +.015 (15m) / +.030 (30m) → signed-vol asymmetry genuinely carries DIRECTION | decays vs base (15m Δ −.0154/−.0029/−.0107; 30m −.0081/+.0062/−.0139) | ⚠ **REAL-but-SUB-BREAKEVEN, KILLED** — genuine signed-vol direction content but pooled .5258/.5382 < .541 (only 2024@30m .5516 clears), DECAYS, non-additive |

**D7 is the distinction that matters:** it is the ONLY direction shot that PASSES its mechanism null (the signed-vol good/bad-vol asymmetry genuinely carries directional content — unlike D1/D6 which are null-on-mechanism), yet it is sub-breakeven (.5258/.5382 < .541), decaying, and already subsumed by the cross-pair book (+semivar does not add). It is the DIRECTION analog of magnitude's "real-but-sub-bar": genuine content, sub-deployable. **CONVERGENT VERDICT:** FFD (T2) + D1 signature lead-lag + D6 HAVOK + D7 signed-semivariance ALL killed forward — no cross-sectional direction family beats or matches the certified book, and the base book itself decays forward. This corroborates the standing program conclusion: **direction beyond the engineered cross-pair book is EFFICIENT on existing data; the only frontier is EXTERNAL data** (rate-diff / vol / EURGBP ticks).

### Remaining slate — evidence-based scope decisions (post-Phase-4)
After 4 convergent direction nulls (FFD / D1 / D6 / D7) on the deployment-faithful gate, with the base book itself decaying forward, the remaining slate is scoped by evidence so the next session resumes deliberately:
- **D5 causal lead-lag (Granger/PCMCI/structural-VAR)** — **REASONED-SKIP.** The certified base book ALREADY contains every peer's lagged lead-lag feature (`ll_<pair><k>`, k∈{1,3,5,10,15,30}) which the GBM weights; D1 proved signature lead-lag content does NOT survive a rotation null. A Granger feature-SELECTION on top of an already-lead-lag-saturated GBM has near-zero marginal prior. Re-open only with EXTERNAL leaders (rate-diff), not more EURUSD-panel selection.
- **D2 untruncated signature kernel / D4 FASCL contrastive** — **BLOCKED:** sigkernel/KeOps + the FASCL encoder need GPU and (for sigkernel) a C-extension build that fails here (no `Python.h`). Deferred to a GPU+headers environment; low prior given D1 (signatures) already null.
- **D8 quantile-direction baseline** — a CONTROL, not an edge candidate; only needed to ablate a TSFM-quantile claim this campaign does not make. **Skip.**
- **Phase 5 gates G1 (TDA corr-cloud, needs gudhi=unbuildable) / G2 (BOCPD, ruptures available) / G3 (windowed-DMD)** — **MOOT:** a gate conditions a SURVIVING signal; no direction signal survived the forward holdout, so there is nothing to gate. Re-open only if a future (external-data) signal clears breakeven first.
- **Phase 3 / T1 information-driven bars** — needs per-pair 1s/tick data; only EURUSD 1s on disk (2021+). The cross-pair-synchronization value (the whole point) requires the other 6 pairs' ticks = EXTERNAL data acquisition. DEFERRED behind T2.
- **Transforms T3 vol-time subordination / T4 cross-pair whitening / T5 Hilbert phase** — low prior: T4/T5 are direction transforms (direction shown efficient); T3 is a magnitude transform but mag_har showed base rv already extracts ~all magnitude. Catalogued as QUEUED; not run this campaign.
- **Magnitude exotica M4 (TDA-Wasserstein, gudhi-blocked) / M5 (multiscale-ECC) / M6 (MOMENT)** — low prior after the HAR canon (§6g) showed the realized-vol family adds only a forward-consistent sliver over base rv. M4 gudhi-blocked; M5/M6 QUEUED.

### Leaderboard impact
**No leader changes — the UP/DOWN LEADERBOARD is UNCHANGED; nothing was unseated.** Every tested lever this campaign is KILLED or REAL-but-SUB-BAR/SUB-BREAKEVEN: the HAR/RV magnitude arms are sub-bar (best +all +0.0021, economically negligible); FFD direction is KILLED at 15m and 30m; and ALL Phase 4 cross-sectional DIRECTION families are killed forward — D1 signature lead-lag (fails its rotation/mechanism null), D6 HAVOK (sub-breakeven, never clears .541), and D7 signed-semivariance (REAL signed-vol direction content — passes its shuffle null — but pooled .5258/.5382 < breakeven .541, decaying, non-additive to the base book). The base cross-pair book itself decays forward (.59→.55→.52 @15m), confirming the certs are refit-dependent. The certified direction keys (10m UP .586/DOWN .568, 15m UP .5673/DOWN .5742, 30m UP .5588/DOWN .5525) and the certified magnitude edge all STAND; magnitude was additionally re-validated forward with no decay. **No new deployable edge was created; convergent conclusion: direction beyond the engineered cross-pair book is EFFICIENT on existing data — external data is the only frontier.**

### Post-campaign — BRIER-ADVANTAGE audit of the 4 certified direction books (probabilistic-skill-vs-ranking)
Our certs rank on AUC + deriv-faithful selective accuracy (both reward RANKING, not the quality of the probabilities). The one transferable lever from the 2026-06-07 11-repo Polymarket scour (`evan-kolberg/prediction-market-backtesting`) is probabilistic-skill scoring; Tier-1 verified absent (`brier` token nowhere in-repo). Added opt-in `brier=True` to `fwd_holdout.py` (observability-safe trailing-persistence baseline + train-only OOF-isotonic calibration; self-checked) and ran `brier_audit.py` over m5xp/m10xp/m15xp/m30xp (certified gate config, train≤2023, NY, per forward year). **ALL 4 books PASS the pre-registered falsifier:** `badv_persist > 0` in EVERY forward year 2024/25/26, all-bars AND bet-tail, CALIBRATED AND RAW → **none of the 4 direction certifications is "ranking-only"**; each carries genuine probabilistic skill over a naive persistence baseline (badv_persist grows with horizon: 5m ~+0.002 → 30m ~+0.012). **Honest nuance:** the persistence baseline is itself worse-than-flat-0.5 (persist_brier 0.252–0.262 > 0.25 → intraday EURUSD mildly anti-persistent); vs the harsher flat-0.5 the all-bars edge is razor-thin (`badv_flat` ≤ +0.0013 in 2024, slightly NEGATIVE in 2026 at 15/30m) and the genuine skill is concentrated in the **selective bet-tail** (`badv_persist_sel` +0.005…+0.017) — exactly where the books operate. Net: certs survive an audit they'd never faced; interpretation tightened to thin/tail-concentrated/refit-dependent; **no key changes**. Audit-only — can tighten, never inflate. `brier_audit_result.json`, docs/EXPERIMENT_LEDGER.md #161, docs/DIRECTION_FINDINGS.md §2026-06-07, METHODS_CATALOG validation 9.8.

---

## 2026-06-06 — DST-correct session re-campaign + Kronos look-forward fix + full-suite audit

This session re-tested every freq×method **segmented by trading session** under DST-correct local-tz masks (`sessions.py`: NY 8–17 America/New_York, LDN 8–16 Europe/London, Asia 9–18 Asia/Tokyo; `session_mask` = local-tz hour applied per-day across train+val+test+oos). It also CAUGHT + FIXED a 1-bar look-forward misalignment in the Kronos direction eval, and ran a full-suite correctness audit. **No new key unseats a leader; the headline is WHEN the certified ≥10m direction edge lives (NY only) and a corrected Kronos null.**

### DIRECTION — session-segmented cross-pair (the certified ≥10m lever, strict session-only, DST-correct)
`session_xpair.py` runs the certified cross-pair USD-residual+OF book STRICT session-only (filter restricts decision rows in train+val+test+oos; causal-continuous features = rows-only), gate `{10:5m_bb_width, 15:15m_bb_width, 30:1h_bb_width}×conf`. **NY certifies BOTH sides at every horizon; LDN and Asia certify at NONE.** NY also **BEATS the legacy fixed-UTC gate** (10m legacy .586/.568, 30m legacy .559/.553) — the direction edge is decisively NY-concentrated.

| Horizon | NY UP (p10) | NY DOWN (p10) | LDN UP / DOWN | Asia UP / DOWN | NY verdict |
|---|---|---|---|---|---|
| **2m** | **0.564** | **0.560** | .514 / .506 | .499 / .497 | ✅ BOTH certify (NY); LDN/Asia null — **NEW: legacy EURUSD 2m was DEAD/uncertified** |
| **5m** | **0.596** | **0.588** | .526 / .516 | .510 / .504 | ✅ BOTH certify (NY); LDN/Asia null — **NEW: legacy was UP-ONLY (.553); NY adds the DOWN side** |
| **10m** | **0.6053** (15/15) | **0.5896** (15/15) | .523 / .524 | .511 / .516 | ✅ BOTH certify (NY); LDN/Asia null |
| **15m** | **0.5845** | **0.5712** | .527 / .520 | .519 / .496 | ✅ BOTH certify (NY); LDN/Asia null |
| **30m** | **0.5681** | **0.5639** | .520 / .514 | .487 / .509 | ✅ BOTH certify (NY); LDN/Asia null |

Files: `session_xpair_{2,5,10,15,30}m_{ny,ldn,asia}_result.json`. **NY certifies BOTH sides at every horizon 2m→30m** (the two-sided NY cross-pair range now extends down to 2m). These NY numbers are the **strongest DIRECTION result of the session re-campaign** and refine (do not replace) the existing fixed-UTC `EURUSD.m{10,15,30}xp.v1` certifications: the cert is real but its edge is concentrated in the NY session. The 2m & 5m NY two-sided certs are *new* (legacy EURUSD 2m was dead; legacy 5m was UP-only at .553).

### DIRECTION — session-segmented base GBM / tick GBM (all NULL or KILLED, every session)
- **1m tick GBM** (`session_1m.py`): DIRECTION NULL all sessions (pooled ~.504, p10 .500–.504, frac_clear 0.0). `session_1m_dir_{ny,ldn,asia}_result.json`.
- **2m tick GBM** (`session_2m.py`): DIRECTION NULL all (p10 .496–.500). `session_2m_dir_{ny,ldn,asia}_result.json`.
- **5m/10m/15m/30m base bar GBM** (`session_bars.py`): DIRECTION KILLED all sessions (NY strongest, e.g. 10m NY p10 .525; 15m NY p10 .520 / LDN .515 / Asia .512, all KILLED). `session_{5,10,15,30}m_dir_{ny,ldn,asia}_result.json`.

### KRONOS direction — CORRECTED (look-forward bug fixed) → NULL at every horizon & session
A user-caught **1-bar look-forward misalignment** in `kronos_dir.py`/`kronos_ft.py` (decision-bar context scored against the move INTO the entry, disjoint from the forward deriv label — failure-mode FM-F) was fixed in `kronos_mtf.py` (context ends AT bar i, predict H forward steps, `Pup = pred_close(+H) > C[i]`, pred-side contiguity + nonoverlap gap; forward label agrees next-bar sign 92.3%, n=233,950). `kronos_dir.py` is now gated behind `KRONOS_DIR_LEGACY=1`; the legacy result is superseded. **Corrected verdict: Kronos direction is NULL at every horizon 1/5/10/15/30m, zero-shot AND fine-tuned, all sessions** (pooled .50–.51, CPCV p10 .489–.500, all KILLED, up-rates in-band). Even at NY≥10m where cross-pair certifies .57–.61, Kronos reads ~.50 — it ingests only EURUSD's OWN OHLCV, not the 7-pair USD cross-section that carries the edge; fine-tune did not help direction. Files: `kronos_dir_mtf_*_result.json`. **The user's two multi-TF ideas were tested directly and also fail:** (a) FINE-mode "use a finer grid up the chain" (1m→2/5/10m, 5m→10m) all KILLED (pooled .490/.509/.495/.503, CPCV p10 .481/.499/.475/.497, frac_clear 0.0); (b) multi-TF vote-ensemble (`kronos_ensemble.py`) — 5m KILLED (pooled .531, p10 .428), 10m ABORT (only 12 common decision bars across 3 grids). Finer/combined single-pair views add no sign — consistent with the missing ingredient being the 7-pair cross-section, not more views of EURUSD. The per-session bar-image CNN (4th model class) is also null (NY VAL .510/test .503/oos .505; LDN .509/.499/.503; Asia .501/.506/.500). New scripts: `kronos_ft.py`, `kronos_mtf.py`, `kronos_bars.py`, `kronos_ensemble.py`; `barcnn_run.py` gained a per-session SESSION arg.

### MAGNITUDE (context — sign-invariant, no up/down key; full detail in `docs/MAGNITUDE_FINDINGS.md`)
Session-segmented magnitude CERTIFIES every session at every freq: 1m magAUC NY .675/LDN .728/Asia .717 (p10 .60–.71); 2m p10 .58–.70; 5m/10m/30m base bar p10 .72–.80. Files `session_{1m,2m}_mag_*`, `session_{5,10,30}m_mag_*`. (Recorded here for completeness only; magnitude has no UP/DOWN key.)

### FULL-SUITE CORRECTNESS AUDIT — no certified book invalidated
A 19-agent audit + lead Tier-1 proofs confirmed the look-forward bug is **ISOLATED to the Kronos family (2 scripts), NOT systemic** — across all 308 scripts + a dedicated forecast-derivation sweep, NO second instance was found. Every other forecast-derivation script predicts the FORWARD quantity over the SAME horizon as the label at the SAME bar (correctly aligned); GBM/CNN classifiers train directly on the label → structurally immune. Proven clean (Tier-1 empirical): TICK substrate (features causal max|full−trunc|=0.0; label forward 0/4000 mismatch; up-rate .5006), BAR substrate (0/2000 mismatch at H=5/10/30; up-rate .498–.506), XPAIR substrate (0/2000 at H=10), and all 239 bar features (truncation max|full−trunc|=0.0). **NO certified direction/magnitude book is invalidated.** Bounded FM-E/FM-A flags exist (`barcnn_mag.py`, `barcnn_regime.py`, `usdjpy_2m_cpcv2.py` max-p10 cell .5185 <<.541 already CERTIFIED=false, `min2_mim.py`/`min2_legsign.py` KILL screens, superseded pre-v3 cohort) but none invalidate a cert.

### Leaderboard impact
No leader changes. The certified ≥10m cross-pair DIRECTION keys (10m UP .586/DOWN .568, 15m UP .5673/DOWN .5742, 30m UP .5588/DOWN .5525) stand; this session **localizes their edge to the NY session** (NY recert UP/DOWN: 10m .605/.590, 15m .585/.571, 30m .568/.564 — NY beats the fixed-UTC gate) and adds the corrected Kronos null as a closed channel at all horizons/sessions.

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

Citations: `m_tick_prod.py:4-5`, `docs/DIRECTION_FINDINGS.md:142,148,178,199`. (No tick result-JSON; figures from docstrings + findings docs; models gitignored.)

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
| 18 | **USD-strength-conditioned DOWN** (D3 retarget, `min1_downcond.py` MX_HOR=1) | USD-strong DOWN 2024 .500/2025 **.494**/2026 .497; coverage curve FALLS to .475(25)/.464(26) at the top-2% most-USD-strong tail; USD-strong≈USD-weak (no separation) | ❌ KILLED (`min1_downcond_result.json`) — 60s DOWN efficient even USD-gated; STRENGTHENS 5m D3 kill |
| 19 | **\|ret\|-weighted magweight retrain** (POW=0.5, `min1_magweight.py`, tick substrate, deriv-faithful) | DOWN cov0.05 2024 .513/2025 **.507**/2026 .512 (CI-lo all <.50); UP worse than filter (2024 .42-.45). **best_iter=8/4000** — no learnable direction signal to weight | ❌ KILLED (`min1_magweight_result.json`) — loss-reweighting doesn't unlock 60s DOWN; 5m razor-thin edge doesn't transfer to more-efficient 60s |
| 20 | **magnitude→direction bridge** (`min1_magdir.py`, frozen mag+dir, VAL-frozen magp gate) | DOWN FLAT ~.50 across EVERY predicted-move-size gate: magq0.0 .500/magq0.7 **.502**/magq0.95 (biggest moves) **.498** in 2025; UP same ~.50-.51. Magnitude finds big moves (magAUC .787) but they carry ZERO direction | ❌ KILLED (`min1_magdir_result.json`) — textbook sign-invariance at the operating point; the 60s direction edge does NOT hide on large moves |
| 21 | **Faithful CPCV of the UP-filter + DOWN** (`min1_cpcv.py`, frozen-book, ties-strict, 28 purged-comb paths + block-boot) | **UP NOT certified**: pooled-strict .573, block-boot CI-lo **.530**, path p10 **.524**, 75% paths clear, 2024-strict **.520**<breakeven. DOWN NOT certified: pooled .519, p10 .474, 29% clear | ⚠/❌ formal (e)-step (`min1_cpcv_result.json`) — UP is a regime-dependent filter, NOT a robust edge (near-miss, same as 2m .5445 FAIL); DOWN dead. No certified 60s direction edge |
| 22 | **BAR-IMAGE 2-D CNN** (Sezer CNN-BI, the last non-subsumed bar/candlestick sub-lever; `barcnn_bars.py`→1m OHLCV, `barcnn_run.py`→render+CNN, `barcnn_cpcv.py`→CPCV). 3 variants × 30-bar window, per-window min-max norm, MNIST-class CNN, 60s wc_ret label (ties-strict), train 80k subsample | **hist** (Sezer close-histogram 1×30×30) VAL dirAUC **.4992** test .5029 oos .5030; **ohlc** (3ch wick+up/down-body) VAL **.5028** test .5011 oos .5018; **gaf** (GASF+GADF) VAL **.5020** test .5006 oos .4997. **CPCV (28 purged paths, all 3 variants, cov{100,10,5,2}%): path_p10 .484–.499, frac_paths_clear_0.541 = 0.0 everywhere**; every held-out year CI95-lo < 0.519 | ❌ **KILLED** (`barcnn_{hist,ohlc,gaf}_result.json` + `barcnn_cpcv_*_result.json`) — image geometry of past bars carries magnitude not next-60s sign (even GADF antisymmetric sign-field is null); confirms sign-invariance @60s. **Bar/candlestick 2-D-image family now RUN + EXHAUSTED on-disk** for DIRECTION; method in `docs/METHODS_CATALOG.md` §5.5. **The SAME bar-image CNN pointed at MAGNITUDE clears >65% CPCV-certified (absolute-scale image, magAUC .699/.714/.686, all 28 paths ≥0.65 at cov≤0.2 every year) → see `docs/MAGNITUDE_FINDINGS.md` §3 (60s bar-image-CNN row).** Kronos deprioritized (RankIC/magnitude not direction; fine-tune deteriorates arXiv:2511.18578) |

Adversarial controls: informational ceiling (`_redteam_magdir60.py`) 2025 cond-acc 0.512@10%→0.517@0.5%cov (gap to 0.65 is informational, not coverage); 60s **dirAUC 0.510 vs magAUC 0.787** identical data (sign-invariance).

### Key results (MEASURED)
| Key | Result (2024 / 2025 / 2026) | n / floor | Method | Status |
|---|---|---|---|---|
| **(EURUSD, 60s, UP)** 🏆 | **0.520 / 0.584 / 0.613** | floor 0.520 | up-only FILTER on symmetric ensemble (`min1_updown.py`) | ✅ genuine improvement over symmetric (0.548/0.550); regime-dependent; 2024 a wash; **does not clear 0.65**. ⚠ **NOT CPCV-certified (2026-06-03, `min1_cpcv.py`):** ties-strict pooled .573, block-boot CI-lo **.530**, 28-path **p10 .524**, only **75%** of paths clear, 2024-strict **.520**<breakeven — a regime-dependent filter, NOT a robust edge (near-miss; same story as 2m UP .5445 FAIL). Best available 60s UP point-estimate; uncertified |
| **(EURUSD, 60s, DOWN)** | **0.522 / 0.522 / 0.516** | — | symmetric ensemble, down-predictions (`min1_updown.py`) | ❌ dead — no down-side edge in any year; **USD-driver (D3) + \|ret\|-magweight + mag→dir bridge all KILLED @60s** (rows 18-20); **CPCV pooled .519 p10 .474** (`min1_cpcv.py`); **discovery round 1 DRY** (12 microstructure-sign levers vetted, all subsumed/sign-invariant). DOWN honestly exhausted on-disk; external (risk-reversal/rate-diff) only redirect |

**Side-specific TRAINING (controls — both worse than the filter):**
| Approach | 2024 / 2025 / 2026 | Why it failed |
|---|---|---|
| Dedicated UP-specialist (trained only on dip-buy bars) (`min1_upspec.py`) | 0.498 / 0.537 / 0.519 | subset-training destroys the confidence ranking (WORSE than the filter) |
| Dedicated DOWN-specialist (control) | 0.472 / 0.521 / 0.481 | dead — no down-side signal to sharpen |
| Up-only production-gate filter (refined config) (`min1_best.py`) | sags / 0.581 / 0.600 | floor 0.518; no freeze (2024 sags) |

**Mechanism:** the reversion lever buys dips / sells rallies; dip-buying worked while rally-selling didn't over the EUR-up 2025-26 regime → the UP-key carries and the DOWN-key is dead. The asymmetry VANISHES in 2024 (UP 0.520 ≈ DOWN 0.522) → it's a regime artifact, not structural. Setup base-rates ~coin-flip (dip-bounce 0.507, rally-down 0.514). Citations: `docs/min1_research_log.md:104-113,139`.

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
| **(EURUSD, 5m, UP)** | **0.577 binding ✅ CERTIFIED** (.605/.577/.615; CPCV p10 .576) | Side-split of the frozen `EURUSD.m5xp.v1` book (`m5_updown.py`). All 3 yrs moved-acc CI95-lo clear 0.541 (2024 [.579,.629]/2025 [.552,.603] n1379/2026 [.556,.675]); up-rate clean .504/.523/.529; COMBINED reproduces book 0.607/0.555/0.606 exactly. CPCV (`m5_cpcv.py`): 28/28 purged-combinatorial paths clear 0.541, p10 .576, block-boot CI-lo .575 — **survives the exact test that downgraded 2m UP to uncertified**. `m5_updown_result.json`, `m5_cpcv_m5xp_result.json`. ⚠ **V1 haircut (2026-06-02):** pooled edge t=5.81 survives MT-deflation decisively, but the **binding-2025 .577 alone is multiplicity-inflated** (BY-adj p 0.158, fails HLZ t≥3); the durable single-year figure is the refit floor **.553** under full-family multiplicity, not the .577 headline. `m5_mt_haircut_result.json`. ⚠ **DEFENDED 2026-06-03:** corpus audit (N16–N19 family-killers, all KILLED) + 2,874-combination sweep + meta-gate capstone produced NO UP challenger beating the .553 floor / .577 binding (best UP combo `blend(pxp,p30)` .5414, CI-lo .497). Leader STANDS. `m5_combo_sweep_result.json`, `m5_combo_metagate_result.json`. |
| **(EURUSD, 5m, DOWN)** | **0.559 ⚠ marginally rescued** (POW=0.5 magweight: .605/.559/.549; CPCV cov0.05 p10 .5441/89%) | Unweighted book DOWN was dead (binding 2025 .533 < breakeven). **B1 |return|-weighted retrain (POW=0.5, `m5_magweight_cpcv.py`) is the new DOWN leader** — first to BOTH CPCV-certify at cov0.05 (p10 .5441, 89% paths) AND point-clear the forward binding year (.559 vs .533), up-rate clean. RAZOR-THIN: cov0.05-only, forward 2025 CI-lo .533 grazes breakeven. Best DOWN to date; not robust. `m5_magweight_cpcv_result.json`, `m5_magweight_result.json`. ⚠ **DEFENDED 2026-06-03:** family-killers (incl. N19 H=5 risk-residual relabel, 2026 collapses .5124) + 2,874-combination sweep + meta-gate produced NO DOWN challenger clearing .541 CI-lo (closest `blend(pxp,p10)` binding-2025 .5529, CI-lo .5277). Leader STANDS; DOWN remains marginal / non-deployable. `m5_combo_sweep_result.json`, `m5_combo_metagate_result.json`. |

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

### Side-split (step a+e) — base book refit-CPCV — `m10_cpcv_side.py` → `m10_cpcv_side_result.json` (2026-06-04)
First side-split of the 10m base book (both keys were UNTESTED). Faithful clone of `m15_cpcv_side.py` (the keystone-cert harness): 3-model ensemble REFIT per fold, 6 groups / C(6,2)=15 purged paths, purge+embargo=1 horizon, gate `5m_bb_width≤q{10,20,33} × NY × conf-cover` tuned per fold, test trades split by predicted side, deriv-faithful (ties-lose, nonoverlap 600s). Pooled n=5.29M, up-rate 0.5018 ✓.
| Side | refit-CPCV p10 | mean | min | frac paths clear 0.541 | verdict |
|---|---|---|---|---|---|
| **UP** | **0.5611** | 0.5861 | 0.5572 | **15/15 (1.0)** | ✅ certifies by rule; robust across recent paths |
| **DOWN** | **0.5519** | 0.5735 | 0.5277 | 14/15 (0.933) | ✅ certifies by rule, but **regime-dependent** (2021–2026 path .528 < BE; p10 leans on pre-2019 eras) |
Binding recent-era paths: 2019–26 UP .557/DOWN .565; 2021–26 UP .560/**DOWN .528**. These are the **base-book floors the cross-pair keystone (`m10_xpair_cpcv.py`, A6c) must beat**; above the 15m base floors (UP .5475 / DOWN .5486).

### ★ KEYSTONE cross-pair side certification (step a+e) — `m10_xpair_cpcv.py` → `m10_xpair_cpcv_result.json` (2026-06-04)
Cross-pair USD-residual+OF primary (m5xp source, MX_HOR=10), single lgb refit per fold, 15 purged paths, gate `5m_bb_width × NY × conf-cover`, side-split by predicted side. The exact pipeline that certified 5m-UP and BOTH 15m sides.
| Side | refit-CPCV p10 | mean | min | frac clear 0.541 | base floor | verdict |
|---|---|---|---|---|---|---|
| **UP** | **0.5863** | 0.5985 | 0.564 | **15/15 (1.0)** | .5611 | ✅ **CERTIFIED + IMPROVES base**; robust |
| **DOWN** | **0.5683** | 0.6001 | 0.5556 | **15/15 (1.0)** | .5519 | ✅ **CERTIFIED + IMPROVES base**; recent-era FIXED (2019–26 .575 / 2021–26 .570 vs base .528) |
All paths healthy n (UP 638–3576, DOWN 442–2399 → no thin-coverage mirage). **This is a refit-CPCV cert (not a frozen-forward overstatement); up-rate 0.5018 ✓.** 10m DOWN behaves like 15m (cross-pair carries DOWN sign) NOT like 5m (dead). Frozen → `EURUSD.m10xp.v1`.

### Key results
| Key | Result | Status |
|---|---|---|
| **(EURUSD, 10m, UP)** | **0.586** refit-CPCV p10 (15/15) ✅ | **CERTIFIED — cross-pair leader `EURUSD.m10xp.v1`** (improves base floor .561). `m10_xpair_cpcv_result.json` |
| **(EURUSD, 10m, DOWN)** | **0.568** refit-CPCV p10 (15/15) ✅ | **CERTIFIED — cross-pair leader `EURUSD.m10xp.v1`** (improves base floor .552; recent-era robust). `m10_xpair_cpcv_result.json` |

Conclusion: **10m is a BOTH-SIDES refit-CPCV-certified direction edge via the cross-pair USD-common-factor — the same two-sided mechanism as 15m.** Leader = `EURUSD.m10xp.v1` (UP p10 .586 / DOWN p10 .568, 15/15 paths each), improving the base-book floors (.561/.552) which themselves beat the 15m base. Combined honest deliverable .602/floor .579 remains the fallback; the >0.65 at 10m is magnitude. Citations: `m10_xpair_cpcv_result.json`, `m10_cpcv_side_result.json`, `docs/m10_research_log.md:115-191`.

### DEPLOYMENT SPEC — EURUSD 10m, both sides (FINAL, 2026-06-04; improve+discover loops dry, K=2 dry rounds)
**Book:** `EURUSD.m10xp.v1` (cross-pair USD-residual+OF primary, lgb 3000-tree, m5xp source @MX_HOR=10). **Gate:** `5m_bb_width ≤ q-thr × sess_ny(13–22 UTC) × |p−0.5| ≥ conf_thr`. **Settlement:** deriv Rise/Fall mid-to-mid, +1s entry, ties LOSE, breakeven 0.541, nonoverlap 600s. **Durable number = refit-CPCV p10** (well-powered, 15/15 paths); frozen-forward overstates (selection on easy years) — size on the refit floor.
- **(10m, UP) — `EURUSD.m10xp.v1` UP-bets (p>0.5).** Certified refit-CPCV p10 **.5863** (mean .5985, min .564, 15/15). Deploy gate cov10% (conf_thr in strategy.json); forward 2024 .642 / 2025 .605. Size ⅛-Kelly on the **.586** floor. Robust across recent CPCV paths (no regime fragility).
- **(10m, DOWN) — `EURUSD.m10xp.v1` DOWN-bets (p<0.5).** Certified refit-CPCV p10 **.5683** (mean .6001, min .5556, 15/15). Deploy gate **cov5%** for DOWN (forward 2025 .630, EV/bet .166@R0.85 — materially better than cov10's .574/EV .062); cov10% also clears. Size ⅛-Kelly on the **.568** floor.
- **Caveat (deployability):** 10m < deriv's **15m forex Rise/Fall minimum** → this is a RESEARCH / synthetic-index horizon; for live deriv forex use the deployable sibling `EURUSD.m15xp.v1` (same mechanism, 15m, deriv-tradeable). 2026 per-side n is thin (partial year, n~20–25) → the refit-CPCV cert (pooled, well-powered) is the operative evidence, not the 2026 forward slice.
- **Exhaustion (both sides):** full (a)-(e) pipeline ran + 11 distinct improve/discover levers (magweight, residual-label N19, GMADL N18, ACI, asym-specialist, calibration, cross-horizon-blend, lagged-leadlag N17, queue N16, intraday-momentum N4, base) — ALL KILLED/subsumed; none beats the gated raw cross-pair sign. Edge is info-bound by the 2025-USD-regime. Only open frontier = external/funded data (gated). See `sweeps/EURUSD_10m{,_backlog}.md`.

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
| D8 | issue #9 date-only retrospective replay: A=current v1; B=clean capped legacy-date refit; C=clean capped date-refresh refit | Apr–May 2026 standardized whole-book correctness: A .5223 (n=157, yield .000316); B .5227 (n=176, .000361); C .5127 (n=316, .000361). C UP .5135 (n=185); DOWN .5115 (n=131). | ⚪ **INCONCLUSIVE — `control_not_estimable`.** Simultaneous promotion lower bounds: C−A yield −.002113, C−B yield −.001595, C-UP accuracy−.5 −.085977, C-DOWN accuracy−.5 −.127017. No survivor/candidate; incumbent unchanged. Pair: `results/json/m15_book_refresh_53547498c599f2877a2f6616ae725f8f27e11070e3b550e82eb26ec039fb23f6_EURUSD_replay_result.json`; joint/shadow: `results/json/m15_book_refresh_53547498c599f2877a2f6616ae725f8f27e11070e3b550e82eb26ec039fb23f6_joint_replay_result.json`, `results/json/m15_book_refresh_53547498c599f2877a2f6616ae725f8f27e11070e3b550e82eb26ec039fb23f6_shadow_spec_result.json`. |

### Key results
| Key | Result | Status |
|---|---|---|
| **(EURUSD, 15m, UP)** | **✅ CERTIFIED — refit-CPCV p10 .5673** (cross-pair; mean .5793, **15/15** paths clear, +1.98pt vs base book) | **LEADER = cross-pair USD-residual+OF primary @MX_HOR=15** — frozen `EURUSD.m15xp.v1` (`m15_xpair_freeze.py`; m5xp source retargeted). Certifies on the SAME refit-CPCV harness ABOVE the base book (book up-bets p10 .5475, also certified, **frozen-stable fallback**). Mechanism: models the USD common factor → robust in USD-regime binding years. **DEPLOY caveat (adversarial-verified):** the edge is **refit-dependent** (USD factor non-stationary) — frozen-2021 single-model forward @cov10% is marginal in binding years (UP .602/.561/**.539**, CI-lo<BE); the CPCV advantage needs **periodic retrain** to realize. Size on refit floor .5673. `m15_xpair_cpcv_result.json`, `m15_xpair_freeze_result.json`. |
| **(EURUSD, 15m, DOWN)** | **✅ CERTIFIED — refit-CPCV p10 .5742** (cross-pair; mean .5818, **15/15** paths clear, +2.56pt vs base book) | **LEADER = cross-pair primary @MX_HOR=15** (`EURUSD.m15xp.v1`, same model). **Best certified DOWN-side direction edge in the program** (+3.3pt over breakeven on the refit-CPCV vs 5m/2m/60s DOWN all dead). Base-book down-bets p10 .5486 = frozen-stable fallback. Same DEPLOY caveat: frozen-2021 forward @cov10% marginal in binding 2025 (.610/**.531**/.552); deploy with periodic retrain, size on refit floor .5742. `m15_xpair_cpcv_result.json`, `m15_xpair_freeze_result.json`. |

Reconciling the combined number: **0.647** = recent chronological split (favorable end); **0.579** = faithful 15-path CPCV (durable headline); walk-forward is worse (frozen optimal). The 0.545 from `cpcv_certify.py` is a weaker re-implementation, not a refutation. Citations: `min15_cpcv_result.json`, `m15_walkforward_result.json`, `docs/DIRECTION_FINDINGS.md:15`.

### Side-split (step a) — `m15_updown.py` → `m15_updown_result.json` (2026-06-03)
Integrity check ✓ operating COMBINED reproduces the frozen book per-year (2024 .689 / 2025 .582 / 2026 .663). No mirage: operating ALL moved-up-rate .529, gate-only .502 (both ≈[.47,.53]).
| view | side | 2024 | 2025 (binding) | 2026 |
|---|---|---|---|---|
| operating (cov~2%, deployed) | UP | .725 (n171) | **.607** [.513,.692] n117 | .674 [.531,.796] n49 |
| operating | DOWN | .653 (n173) | **.559** [.472,.646] n127 | .650 [.500,.800] n40 |
| gate-only (n powered) | UP | .528 (n2380) | .543 [.518,.570] n1382 | **.500** [.461,.539] n644 |
| gate-only | DOWN | .521 (n2340) | .522 [.495,.547] n1438 | **.566** [.521,.610] n472 |

**What this overturns / what I learned:** (1) **15m DOWN is NOT dead** — unlike 5m (DOWN binding .533, dead), 15m DOWN operating binding-2025 point **.559 > breakeven**. The 5m UP-only asymmetry does NOT transfer (docs warned). (2) **The edge is confidence-driven, not side-driven** — gate-only is ~.50–.54 both sides (barely above coin); all the lift comes from the top-2% confidence selection, roughly symmetric across sides at the operating point. (3) **Regime flip at the gate level** — UP strong 2024-25 / dead 2026 (.500); DOWN weak 2024-25 / strong 2026 (.566). The confidence selection picks whichever side is right per regime → why the combined book works and why a *fixed* side-only filter fights a rotating edge. Binding year 2025 for both; UP (.607) > DOWN (.559). Neither binding CI-lo clears yet (thin-n power problem at cov 2%) → run coverage curve (d) + full-refit CPCV (e) before any dead/certified verdict.

### Coverage curve (step d) — `m15_sidepipe.py` → `m15_sidepipe_result.json` (2026-06-03)
Swept conf threshold (cov 100%→2%, thresholds picked on VAL). **Monotone confidence→accuracy on BOTH sides** (UP 2024 .528→.725, 2025 .543→.607; DOWN 2024 .521→.653) — the signature of a real-but-thin edge (noise wouldn't sort by confidence). But binding-year single-bootstrap CI-lo **never clears .541** at any coverage (best UP .513@cov2%, DOWN .506@cov50%): tight cov → strong point/wide CI (n≤120/yr/side), loose cov → big n but point decays to ~.53. This is a POWER limit of the single-year bootstrap on a **half-sized** sample — exactly why the COMBINED book (full n) certified while its halves don't under that test. → the powered arbiter is the per-side refit CPCV, not single-year CI-lo.

### ★ Side certification (step e) — `m15_cpcv_side.py` → `m15_cpcv_side_result.json` (2026-06-03)
**Full per-fold refit CPCV** (15 purged-combinatorial paths, 3-model ensemble refit + gate re-tuned per fold, purge+embargo = 1 horizon). Integrity: per-side **combined reproduces the independent `min15_cpcv`** (mean .5787, p10 .557) → harness faithful. Per-side path-accuracy distributions:
| side | p10 | mean | min | max | frac paths ≥.541 | median path n | verdict |
|---|---|---|---|---|---|---|---|
| **UP** | **.5475** | .5771 | .5269 | .630 | 0.933 (14/15) | 949 | ✅ **CERTIFIED** |
| **DOWN** | **.5486** | .5805 | .543 | .639 | **1.000 (15/15)** | 845 | ✅ **CERTIFIED** |
| combined | .5574 | .5787 | .538 | .633 | 0.933 | — | (reproduces min15_cpcv) |
**Both sides CERTIFIED** (p10 ≥ .541 AND ≥80% paths clear). 15m is the **first EURUSD horizon with BOTH UP & DOWN certified under refit-CPCV**, and the only deriv-deployable one (min duration 15m). DOWN is marginally MORE robust (frac-clear 1.0 vs UP .933). Adversarial checks: refit-CPCV IS the gold-standard (controls selection+refit overfit); purge+embargo applied; modern-era (group-5) paths all clear both sides (no drift collapse, DOWN-modern min .543 / UP-modern min .548); per-side n healthy (median ~850-950/path). **Honest durable figures = the refit p10 floors .5475 / .5486** (the frozen-forward operating point .607/.559 was optimistic selection — refit < forward ⇒ trust refit). Margins THIN (+.006/+.008 over breakeven) → IMPROVE phase tries to widen them.

---

## EURUSD × 30m (deriv-tradeable)
Raw direction AUC ≈ 0.52. **BOTH sides now refit-CPCV-CERTIFIED** via the cross-pair lever (`EURUSD.m30xp.v1`, UP p10 .5588 / DOWN p10 .5525, 15/15 — REFIT-DEPENDENT). Also home of the one CPCV-deflation-certified MAGNITUDE edge (sign-invariant, no up/down key).

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
| **Cross-pair xpof refit-CPCV** (`m30_xpair_cpcv.py`) | combined refit p10 **.5645** (15/15); UP **.5588** / DOWN **.5525** both cert; AUC ~.529 | ✅✅ **BOTH SIDES CERTIFIED** → `EURUSD.m30xp.v1` (the prior "cross-pair +0" tested peer-FEATURES-on-1-LGB, not pooled xpof) |
| Base-book side-split refit-CPCV (`m30_cpcv_side.py`) | combined p10 .5594; UP .5538 ✅cert / DOWN .5356 near-miss (13/15) | base floor; cross-pair improves both + rescues DOWN |
| **IMPROVE levers** (all vs cross-pair incumbent UP .5588/DOWN .5525) | magweight UP .5528/DOWN .5382 (HURTS); seed-ens .5572/.5503; 3-model-ens-on-pool .5542/.5432; specialist .5501/.5203(collapses); recency HURTS UP; Aₐ/IPCA — | **ALL SUBSUMED** — feature-importance (`m30_xpair_featimp`): edge = POOLED base-feat training (94% gain), NOT xpof factors (~6%) → model/loss/feature/reweight tweaks can't move the binding p10. `m30_{magweight,seedens,xpair_ens,spec,recency}_cpcv_result.json` |
| Discovery (3 rounds, K=2 dry) | month-end rebalancing KILLED (no stable bias); rate-diff/risk-reversal external-gated | only frontier = external data, same as all horizons |

### Magnitude (the certified edge — NO up/down key, sign-invariant)
| Method (file) | Result | Verdict |
|---|---|---|
| **CPCV-certified large-move** (`cpcv_certify.py`) | **AUC mean 0.744**, p10 0.718, deflated 0.712; 28/28 paths clear; lift 5.0× | ✅✅ deflation-proof |
| rv30 large-move (`m30_magnitude.py`) | AUC test24 0.783 / test25 0.729 / oos 0.729; PE null | ✅ realized vol is the predictor, not entropy |

### Key results
| Key | Result | Status |
|---|---|---|
| **(EURUSD, 30m, UP)** | **.5588 refit-CPCV ✅ CERTIFIED** (cross-pair `EURUSD.m30xp.v1`·`da55d7c6b3b08bef`, 15/15; base .5538) | Cross-pair USD-residual+OF @MX_HOR=30 × 1h_bb_width×NY×conf, UP-bets. Fwd@cov5 2024 .629/2025 .618/2026 .530. **REFIT-DEPENDENT** (deploy w/ retrain; frozen-2021 decays 2026). `m30_xpair_cpcv_result.json` |
| **(EURUSD, 30m, DOWN)** | **.5525 refit-CPCV ✅ CERTIFIED** (cross-pair, 15/15; rescued base near-miss .5356) | Same book, DOWN-bets. Fwd@cov5 2024 .649/2025 .564/2026 .578(n45-thin). **REFIT-DEPENDENT**. Improves base +1.69pt. `m30_xpair_cpcv_result.json` |

Conclusion (UPDATED 2026-06-04): the prior "9+ converging direction nulls / not predictable" verdict **predated the cross-pair lever and chased >75%**; its "cross-pair +0" was peer-FEATURES-on-single-LGB (Iteration-2), NOT the pooled USD-common-factor xpof mechanism. That mechanism — the keystone that certified 10m & 15m — **CERTIFIES BOTH 30m sides under per-fold-refit CPCV** (UP p10 .5588 / DOWN p10 .5525, 15/15; DOWN rescued from base near-miss). Edge is **REFIT-DEPENDENT** (frozen-2021 forward decays in 2026 OOS, esp UP .5302→.4652). Best UP+DOWN deliverable = `EURUSD.m30xp.v1` (deploy w/ periodic retrain); base `EURUSD.m30.v1` combined .591 = lower fallback; magnitude AUC 0.73–0.78 (sign-invariant, separate). Citations: `m30_xpair_cpcv_result.json`, `m30_cpcv_side_result.json`, `m30_xpair_freeze_result.json`, `cpcv_certify_result.json`.

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
4. Magnitude (|ret|) is sign-invariant → it has no up/down key; record it in `docs/MAGNITUDE_FINDINGS.md`.
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
- **VENUE & SETTLEMENT — VERIFIED LIVE (2026-06-03, `m5_venue_feasibility_result.json` + `deriv_frxEURUSD_contracts_for.json`):**
  ⚠ **5m EURUSD is NOT a tradeable fixed-payout binary at any verified regulated venue.** deriv `contracts_for` API (live,
  spot 1.15991) → forex Rise/Fall **MIN duration 15m** (intraday 15m–1d / daily 1d–365d; callput + callputequal, both
  start types). Nadex — the historical ≤5m US-regulated FX-binary venue (5-min EUR/USD binaries, mid-TWAP settlement) —
  **CLOSED 2025-12-20** (→ Crypto.com CDNA, prediction-markets pivot, no confirmed 5m FX binary). So the certified (5m,UP)
  edge is a **certified PREDICTOR, research-grade for binary deployment at 5m — NOT "the best 5m algo to *trade*".** Honest
  tradeable expressions: **(a)** the deriv **15m** book `EURUSD.m15.v1` (min 15m matches — the 5m research supports it);
  **(b)** SPOT FX (OANDA/IB) 5m hold, but RE-DERIVE under spot settlement (P&L = 5m move − ~0.1–0.2 pip round-trip spread;
  the .541 binary breakeven does NOT apply; a ~.55 hit-rate must be re-mapped to spread-aware, magnitude-dependent spot
  expectancy — likely marginal at 5m); **(c)** offshore ≤5m binary brokers — high counterparty risk, NOT recommended. The
  R≈0.85 / breakeven-0.541 used to CERTIFY the 5m edge is REAL only at ≥15m; at 5m it is a HYPOTHETICAL payout.
- **SPOT-FX DEPLOYMENT SPEC — the real venue-available expression (2026-06-03, `m5_spot_deploy_result.json`):** EURUSD spot
  (OANDA/IB, no min duration) IS a real venue; re-settled the certified gated trades as spot (P&L = signed 5m move − round-trip
  spread). Gated dir-acc reproduces the cert (.605/.581/.614). **avg 5m |move| = 1.76 pip → spread-critical.** Per-trade
  expectancy, binding (worst-year, CI-lo) by spread: **@0.1 pip raw/prime → +0.258 pip, CI-lo +0.037, all 3 yrs positive →
  DEPLOYABLE** (Sharpe/trade .12, maxDD(1×) <1%, streak 7; lever to target return, DD scales linearly); **@0.2 pip ECN →
  +0.158 pip but binding CI-lo −0.06 → NOT robustly deployable** (positive-EV, thin); **@≥0.4 pip retail → negative → NOT
  deployable.** ⇒ **UP is deployable as SPOT FX only on a raw/prime account (~≤0.1 pip EURUSD spread)**; not at retail spreads;
  binary 5m venue-unavailable. **DOWN: NOT deployable at any spread** (binding-2025 spot expectancy negative even @0.1 pip,
  consistent with the marginal .5441 binary edge). This is the honest "deployable-spec'd" answer: UP = spot-deployable@raw,
  DOWN = non-deployable.
- **MAGNITUDE × DIRECTION for SPOT — RUN, does NOT rescue (2026-06-03, `m5_magspot_result.json`):** the one combination
  skipped in the combo sweep (pmag disabled for OOM). Trained a move-size predictor on the m5xp features and gated the
  certified UP/DOWN spot trades to high-predicted-move bars (q sweep). **Magnitude-gating WORSENS spot:** as the gate
  tightens, avg move rises 2.21→2.75→3.31→**4.28 pip** but binding-year spot CI-lo FALLS +0.037(ungated)→−0.117→−0.111→
  **−0.259** — the larger-move gain is more than offset by **lower directional hit-rate on large/jump-driven moves**. UP
  stays deployable ONLY ungated@≤0.1 pip; DOWN stays non-deployable. **Mechanistic insight: the 5m UP edge lives in
  SMALL/diffusive moves (mean-reverting dip-buy), NOT large moves** (which are jump/informed → sign-unpredictable). Magnitude
  confirmed non-rescuing for direction even via the spot channel (consistent with sign-invariance + m5_magdyn).
- **CROSS-PAIR PORTFOLIO (#3) — edge-generalization test RAN, #3 NOT BUILT (2026-06-03, `m5_xpair_generalize_result.json`):**
  Built the m5xp cross-pair construction TARGETED on each of the 7 USD majors (basket sign-aligned to each pair's own
  up-direction; label = that pair's own next-5m sign), primary GBM UP-gated @NY·pr>.5·cov0.05, per-year 24/25/26 CI95.
  Pre-registered falsifier: GENERALIZES iff ≥2 NON-EUR majors binding-year UP CI95-lo > .53. **FAILED — 0/6** (binding CI-lo:
  GBP .490 / AUD .502 / NZD .472 / JPY .506 / CHF .478 / CAD .471; EURUSD control .491). **The script auto-labeled this
  "EURUSD-specific" — CORRECTED (adversarial verify): the EURUSD CONTROL ALSO failed (.4913), so that label is unsupportable.**
  Honest read: the dip-buy edge GENERALIZES in **MECHANISM** — every major shows the identical shape (strong 2024 .55–.62 →
  decaying to ~.51–.55 by 2026, a generic USD-major mean-reversion) — but is **UNIFORMLY THIN** (this reduced single-GBM/cov0.05
  transfer harness under-powers vs the certified `EURUSD.m5xp_magw` ensemble's .553 floor; it reproduces the shape but certifies
  no single pair, incl. EURUSD). **#3 VERDICT: DEAD for deployment.** It is not a route to a new certifiable single-pair edge
  (none exists); the only surviving thesis (aggregate N decorrelated thin edges, 5m return-corr 0.054, into one book) runs into
  the spread gate — EURUSD is the TIGHTEST major and already spot-deployable only @≤0.1 pip; the other 6 carry WIDER spreads, so
  each leg is MORE spot-marginal, and the 2026 slices decay *coherently in time* (shared regime risk) despite decorrelated
  per-bar returns. Recorded as a mechanism finding; do NOT build the portfolio.

### DOWN — `EURUSD.m5xp_magw_down.v1` (POW=0.5 magweight; MARGINAL, NOT robustly deployable)
- Refit-CPCV cov0.05 p10 **0.5441** (89% folds clear) — barely above breakeven 0.541; forward 2025 .559 but CI-lo
  **.533 grazes below breakeven**. V1/V2 imply DOWN is even more multiplicity-fragile (found via a POW×cov sub-search).
- **Floor (0.5441) sits AT breakeven → near-zero Kelly edge → effectively NON-deployable.** Honest verdict: a real but
  razor-thin marginal edge; ship as "measured, marginal," not as a deployable book. Rally-selling DOWN remains the open gap.
- **§3 loss-reoptimization exhausted (2026-06-03, `m5_lossbatch_result.json`):** focal-γ2 / quantile-median (distribution-head)
  / asymmetric class-weight (×1.5,×2) ALL fail — no DOWN binding-year CI-lo clears .541 (best DOWN-2025 .5407, CI-lo .524).
  The objective is not the binding constraint; loss family joins magweight/seed-ens/rankloss. DOWN exhaustion now spans
  features, model class, ensembling, calibration+ACI, hyperparameter tuning (Optuna), AND loss design — all hit the 2025 wall.
- **FULL CLOSURE (2026-06-03) — corpus audit + 4 family-killers + 2,874-combination sweep + meta-gate, ALL exhausted:**
  A 16-agent corpus audit (`m5_corpus_audit_result.json`) triaged 363 untested levers → 4 genuinely-distinct direction
  family-killers, ALL KILLED: N16 redefined-label (`m5_labels`, best DOWN tb-2025 CI-lo .5165), N17 lead-lag+TE+RFF
  (`m5_leadlag_te_rff`, all null), N18 sign-coupled GMADL/RRL (`m5_signedpayoff_torch`, 0 survivors), N19 H=5 risk-residual
  DOWN-relabel (`m5_residlabel_down_h5`, strongest DOWN 2024 .5685/2025 .544 but 2026 collapses .5124). Then a **2,874-
  combination model-combination sweep** (`m5_combo_sweep`: blends/cross-horizon-gates/consensus/cascades/regime-switch/
  horizon-stacks reusing all frozen per-horizon+cross-pair+stack producers) + a **meta-gate capstone** (`m5_combo_metagate`):
  **NO survivor.** Best DOWN combo `blend(pxp,p10)` reached binding-2025 .5529 (CI-lo .5277) — closest DOWN has come, still
  < .541. **The model-combination cross-product is exhausted; the wall is information/regime (the 2025 USD-factor inversion),
  not model architecture or combination structure.** The sole remaining DOWN lever is the verified-PAYWALLED funded
  risk-reversal feed. No on-disk DOWN edge exists.
- **DOWN DEPLOYMENT DECISION-SHEET — DO NOT DEPLOY, documented (2026-06-03, `m5_down_deploy_result.json`):** operating gate
  `sess_ny & pr<0.5 & |pr−0.5|≥0.0917` (~2.7–5% NY cov; ~1175/1876/444 trades 2024/25/26). **Robust (refit-CPCV) floor p10
  .5441 ≈ breakeven → full-Kelly only 0.775%** (near-zero edge, no margin); refit forward-2025 binding DOWN .559 **CI-lo .533
  < .541 → negative-EV in the binding year**. Kill-switch N/A. Venue: no 5m binary (deriv 15m / Nadex closed); spot EV
  marginal-to-negative on the honest edge. ⚠ **Adversarial-verify catch:** a first pass showed frozen-book forward
  .69/.65/.73 — but moved up-rate .31/.35/.27 (far outside [.47,.53]) + frozen ≫ refit .5441 + the unweighted m5xp control
  reproducing the established .536 expose it as a **FROZEN OVERSTATEMENT** (selection/overfit, not skill); rejected. The
  do-not-deploy stands on the honest refit floor. Disqualifiers: floor at breakeven · binding-year negative-EV · no venue.

### Registry notes (to reconcile)
- `EURUSD.m5xp_magw_down.v1` is git-tagged but MISSING from `books/INDEX.json` — added below.
- `EURUSD.m5xp_aci.v1` is registered with ACI .584 metrics but **ACI is DOWNGRADED** (nested-refit fail); the durable
  UP figure is the m5xp refit floor **0.553**, not ACI. Registry metric annotated.

## Issue #15 historical selector audit — 2026-07-18

- **Outcome: `historical_selection_rule_not_falsified` (inactive diagnostic only).** On the exact 2012–2025 bar-proxy harness, worst-validation-half DOWN selection produced 15/15 complete purged paths with mean **.5821**, p10 **.5622**, minimum **.5573**, maximum **.6143**, and **15/15** paths clearing .541. The preregistered falsifier (p10 < .541 or fewer than 12/15 clearing) did not fire. The allowed diagnostic comparison is capped at **.5622**, below the registered legacy .5742 claim.
- The matched legacy full-VAL selector produced mean **.5800**, p10 **.5603**, minimum **.5515**, and 15/15 clearing; worst-half minus legacy was **+.0021 mean**, **+.0019 p10**, and **+.0057 minimum**. These are historical proxy diagnostics, not replacement incumbent metrics.
- The frozen `EURUSD.m15xp.v1` context, selected on 2022–2023 by the same worst-half DOWN rule, was 2024 **.6096** (n=607, bootstrap CI95 [.5717,.6474]) and 2025 **.5306** (n=571, CI95 [.4904,.5709]). The weak 2025 row is mandatory context, not a second pass/fail rule or an outcome-informed selector.
- Primary source: `results/json/eurusd_m15_down_selection_audit_v1_result.json`; sealed terminal `4c3b0d20013b9b6301f965ac45aae9161fa6db4fc8e5bf7f1c123f2fd0b76b8d` has `decision=inactive_candidate` and `activation=false`.
- **Authority unchanged.** This used pre-2026 moved-bar close-direction proxies, not true next-tick Deriv settlement or fresh OOS. It does not recertify, upgrade, downgrade, activate, or alter `EURUSD.m15xp.v1`, its registered .5742 DOWN floor, any book/registry/index/runtime mapping, or the closed sweep.

## Issue #16 local-tick settlement audit — 2026-07-19

- **Outcome: `not_computable`; terminal `capability_deferred`.** The exact issue #15 trade rows and the sealed local EURUSD bid/ask archive were authenticated, but the fixed 30-second entry/exit tolerance settled only **18,752/21,086 unique decisions (88.93%)**. Every one of the 15 worst-half and 15 legacy path views missed the preregistered 95% completeness gate; worst-half path coverage ranged **81.14%–93.41%**. The audit therefore produced no pass/fail evidence about historical tick-settlement preservation.
- **Diagnostic complete cases only:** worst-half tick p10 **.5463** (mean .5647, min .5436, 15/15 point paths ≥.541) versus same-valid-subset proxy p10 **.5624** (mean .5792), a p10 delta of **−.0160**. Legacy tick p10 was **.5413** (mean .5619, 13/15 point paths ≥.541) versus same-subset proxy p10 **.5545**, delta **−.0132**. These subset estimates cannot rescue the failed completeness gate or replace the registered .5742 proxy claim.
- Missing settlement was broad: **1,227** unique decisions had entry delay >30 seconds and **1,107** had exit staleness >30 seconds; 195 valid decisions tied and correctly lost. Frozen context was likewise incomplete: 2024 valid **483/607 (79.57%)**, tick .5859 versus proxy .6128; 2025 valid **525/571 (91.94%)**, tick .5162 versus proxy .5371. Bootstrap CIs were intentionally not produced after the coverage gate failed.
- Primary source: `results/json/eurusd_m15_down_tick_settlement_audit_v2_result.json`; sealed terminal `257a78861ceb574752a89e777bcb3a82bc2fc2ca1be2da8187b12029ecd240a7` has `decision=capability_deferred` and `activation=false`. The source seal bound 77,255 files / 4,122,905,970 bytes under tree SHA-256 `e7e803ecab4030778d99f3cd2c5296f444d862713ea9c8d5c07ead704c7e5f96`; archive provenance remains unknown and this is not Deriv-feed evidence.
- The sealed v1 campaign blocked in preflight on runner-frozen list normalization before any attempt or scientific read. Its bytes remain immutable; v2 changed only JSON-shape normalization and completed the identical scientific contract.
- **Authority unchanged.** This result neither validates nor falsifies venue settlement, and it does not recertify, upgrade, downgrade, activate, or alter `EURUSD.m15xp.v1`, its registered .5742 DOWN floor, any book/registry/index/runtime mapping, or the closed sweep.

## Issue #17 daily two-year rate-regime procurement screen — 2026-07-19

- **Outcome: `daily_rate_regime_falsified`; terminal `no_candidate`.** The one preregistered feature was the five-common-observation change in current-vintage `DGS2 − ECB euro-area all-ratings 2y par yield`, calendar-lagged to the latest common date on or before the second preceding New York weekday. On issue #15 worst-half DOWN trades, equal-date favorable-minus-adverse lift was positive on **12/15 paths**, with mean **+.0208**, but p10 **−.0250** (minimum −.0741, maximum +.0831). The procurement rule required p10 **≥+.005**, at least 12/15 positive paths, and positive frozen 2024/2025 lifts; the p10 gate failed.
- Frozen context was directionally favorable but cannot rescue the path-tail failure: 2024 favorable **.6820** (104 dates, n311) versus adverse **.6416** (90 dates, n296), lift **+.0404**; 2025 favorable **.6041** (77 dates, n291) versus adverse **.4981** (93 dates, n280), lift **+.1060**. The matched legacy-selector diagnostic also had 12/15 positive paths but p10 **−.0157** (mean +.0245).
- The source snapshot contains 2012–2025 official current-vintage FRED `DGS2` and ECB `YC.B.U2.EUR.4F.G_N_C.SV_C_YM.PY_2Y` rows plus public vendor evidence. It is a slow-regime association only—not a point-in-time vintage reconstruction, exact German Schatz spread, or 15-minute lead-lag test.
- Vendor evidence checked: Databento GLBX declares CME history from 2010-06-06 with trades/OHLCV-1m; XEUR declares 2025-03-10 and is too short for the historical FGBS leg; Eurex identifies FGBS as 1.75–2.25-year Euro-Schatz; Deutsche Boerse's official sheet declares monthly order/trade history from end-November 2013. Exact authenticated ZT cost, FGBS-only subset price/rights, and downloadable samples remain unavailable. No account, vendor contact, terms acceptance, sample download, or purchase occurred.
- **Procurement decision: `do_not_recommend_purchase` on this evidence.** A true intraday underreaction mechanism remains scientifically unknown rather than disproved, but the preregistered free screen does not earn the paid-data spend. Any override requires a new plan and explicit user approval of exact quote/terms after sample checks.
- Primary source: `results/json/eurusd_m15_down_daily_rates_screen_v3_result.json`; sealed terminal `fad191e8c764c4004a4e4d08e1c44a8bae4fb6b809d68dcefb2e53422820947c` has `decision=no_candidate` and `activation=false`. V1 sealed the source/spec then blocked before attempt on frozen-container normalization; v2 fixed that boundary, published an attempt, and failed before native output on an unsupported run-call argument; v3 changed only the runner boundary and binds both preserved predecessors.
- **Authority unchanged.** This does not alter `EURUSD.m15xp.v1`, its registered .5742 DOWN floor, any book/registry/index/runtime mapping, or the closed sweep.

## EURUSD — DERIV-FEED PROGRAM (Deriv-only data; opened 2026-11-15)

> **SCOPE:** this key on the Deriv feed only — Deriv 1-second mid ticks (`/mnt/data/deriv`), no Dukascopy data. Labels settle on Deriv ticks (entry first tick >= decision + 1 s, exit last tick <= entry target + H, ties lose); DEV window 2025-08-11..2026-07-31 for training, selection and CPCV; HOLDOUT 2026-08-01..2026-11-14 sealed (opened once per finalized candidate). Protocol and tools: `docs/DERIV_SOTA_PROGRAM.md`. Tables below are rendered from the result JSONs by `scripts/render_ledger.py` (do not edit by hand).

<!-- DERIV-FEED-NOTES:BEGIN -->
- **Incumbent benchmark on TRADEABLE decisions, books as trained (R1bt, 13:00-16:35 + 18:00-22:00 ET):** UP best `EURUSD.m15.v1` 0.5000 [0.4253, 0.5747] n=174 (NOT SUPPORTED); DOWN best `EURUSD.m15.v1` 0.5204 [0.4286, 0.6224] n=98 (NOT SUPPORTED).
- **Books as deployed today (R1t, true clock, NY 08:00-16:35, tradeable):** UP best `EURUSD.m15.v1` 0.5802 [0.4691, 0.6914] n=81 (NOT SUPPORTED); DOWN best `EURUSD.m30.v1` 0.5385 [0.4038, 0.6731] n=52 (NOT SUPPORTED).
- **DEV-CPCV certified sides, strict certificate (>= 12 of 15 paths counted and clear, p10 >= .541; valid, tradeable runs):** none yet. DEV certification is necessary, not sufficient: the sealed holdout decides (docs/DERIV_SOTA_PROGRAM.md).
- Certified only under the legacy rule, not under the strict certificate (its reading in each item): e15nypm 15m UP (cov 0.03: no (1/15, 1 clear); forward 0/0); r6v-v07-e17nyam 15m UP (cov 0.03: no (1/15, 1 clear); forward 0/0).
- **DEV leaderboard 15m UP** (closest to the goal rule at each run's operating coverage): `r6v-v05-e17nyam` 10/14 clear, p10 0.477, excess +0.076; `r6n-a005-f` 9/15 clear, p10 0.504, excess +0.058; `e17nyam` 9/13 clear, p10 0.500, excess +0.080.
- **DEV leaderboard 15m DOWN** (closest to the goal rule at each run's operating coverage): `e87fixrank` 10/15 clear, p10 0.504, excess +0.051; `e82tailaux` 10/14 clear, p10 0.436, excess +0.057; `r4fixldn` 9/15 clear, p10 0.520, excess +0.038.
- **DEV leaderboard 30m UP** (closest to the goal rule at each run's operating coverage): `e5h30` 3/7 clear, p10 0.382, excess +0.003.
- **DEV leaderboard 30m DOWN** (closest to the goal rule at each run's operating coverage): `e5h30` 2/11 clear, p10 0.419, excess -0.026.
- **Deployment status (Deriv feed):** DO NOT DEPLOY - no candidate has passed both the DEV-CPCV goal rule (or the drift amendment) and a sealed-holdout look, and deployment needs both.
- **Payout-aware reading of `e87fixrank` 15m** (`deriv_a23pay_EURUSD_15m_result.json`, cov 0.03; the candidate's trades frozen, each priced with the payout reconstructed for its decision from the July 2026 quote archive - an estimate resting on the assumptions in docs/DERIV_SOTA_PROGRAM.md "Venue payouts", not a venue record; this pair has no quotes in the archive and is priced by the other six pairs' rule; averages are over the counted paths): UP NOT CLEARED AT THE QUOTED PAYOUT: the candidate takes no UP trade; DOWN NOT CLEARED AT THE QUOTED PAYOUT, NEUTRAL: expected value per unit staked above 0 on 5 of 15 counted paths, p10 -0.1133, mean -0.0225; win rate 0.5516 at mean payout 1.7762 (mean breakeven faced 0.5636; always-DOWN payout 1.7925, expected value -0.1037); the lower-paid side was bought on 0.676 of the trades; 0 of 1824 trades over the 15 paths not priced (clock cells with too few quotes in the archive).
- **Payout-aware reading of `e17nyam` 15m** (`deriv_a27pay_EURUSD_15m_result.json`, cov 0.03; the candidate's trades frozen, each priced with the payout reconstructed for its decision from the July 2026 quote archive - an estimate resting on the assumptions in docs/DERIV_SOTA_PROGRAM.md "Venue payouts", not a venue record; this pair has no quotes in the archive and is priced by the other six pairs' rule; averages are over the counted paths): UP NOT CLEARED AT THE QUOTED PAYOUT, ADVERSELY PRICED: expected value per unit staked above 0 on 6 of 13 counted paths, p10 -0.1249, mean -0.0079; win rate 0.5723 at mean payout 1.7358 (mean breakeven faced 0.5765; always-UP payout 1.8016, expected value -0.1141); the lower-paid side was bought on 0.893 of the trades; 0 of 567 trades over the 15 paths not priced (clock cells with too few quotes in the archive); DOWN NOT CLEARED AT THE QUOTED PAYOUT, ADVERSELY PRICED: expected value per unit staked above 0 on 3 of 15 counted paths, p10 -0.2507, mean -0.1211; win rate 0.5042 at mean payout 1.7425 (mean breakeven faced 0.5743; always-DOWN payout 1.8019, expected value -0.0986); the lower-paid side was bought on 0.881 of the trades; 0 of 717 trades over the 15 paths not priced (clock cells with too few quotes in the archive).
- Superseded: `e1` and `r2` ran before the venue tradeability filter (they include untradeable blackout decisions); their tradeable re-runs are `e1t` and `r2t`.
<!-- DERIV-FEED-NOTES:END -->

<!-- DERIV-FEED-TABLES:BEGIN -->
**Frozen books on the Deriv feed — true clock, ny (America/New_York 08:00-17:00, DST-correct, sessions.session_mask), ALL decisions incl. the untradeable blackout** (`deriv_r1_frozen_books_dev_result.json`; strategy gate, tick-settled, ties lose; SUPPORTED iff n >= 50 and CI95-lo >= .541):

| Book | AUC (moved) | gate rate | UP n · WR [CI95] | DOWN n · WR [CI95] | COMB n · WR | verdict UP / DOWN |
|---|---|---|---|---|---|---|
| `EURUSD.m15.v1` | .5226 | .0042 | 84 · .5833 [.4762, .6905] | 39 · .5128 [.3590, .6667] | 120 · .5583 | NOT SUPPORTED / NOT SUPPORTED |
| `EURUSD.m30.v1` | .5256 | .0253 | 247 · .4858 [.4251, .5466] | 52 · .5385 [.4038, .6731] | 298 · .4966 | NOT SUPPORTED / NOT SUPPORTED |
| `EURUSD.m30xp.v1` | — | — | — | — | — | SCHEMA_UNAVAILABLE: columns not computable from Deriv mid ticks (EURUSD order-flow needs a volume feed) |
| `EURUSD.m15xp.v1` | — | — | — | — | — | SCHEMA_UNAVAILABLE: columns not computable from Deriv mid ticks (EURUSD order-flow needs a volume feed) |

**Frozen books on the Deriv feed — training clock (true UTC - 5 h), nylate (America/New_York 13:00-22:00, DST-correct), ALL decisions incl. the untradeable blackout** (`deriv_r1b_frozen_books_trainclock_nylate_dev_result.json`; strategy gate, tick-settled, ties lose; SUPPORTED iff n >= 50 and CI95-lo >= .541):

| Book | AUC (moved) | gate rate | UP n · WR [CI95] | DOWN n · WR [CI95] | COMB n · WR | verdict UP / DOWN |
|---|---|---|---|---|---|---|
| `EURUSD.m15.v1` | .5370 | .0209 | 282 · .5887 [.5319, .6489] | 208 · .5817 [.5144, .6442] | 464 · .5862 | NOT SUPPORTED / NOT SUPPORTED |
| `EURUSD.m30.v1` | .5384 | .0209 | 174 · .5345 [.4598, .6092] | 46 · .5217 [.3696, .6522] | 216 · .5231 | NOT SUPPORTED / NOT SUPPORTED |
| `EURUSD.m30xp.v1` | — | — | — | — | — | SCHEMA_UNAVAILABLE: columns not computable from Deriv mid ticks (EURUSD order-flow needs a volume feed) |
| `EURUSD.m15xp.v1` | — | — | — | — | — | SCHEMA_UNAVAILABLE: columns not computable from Deriv mid ticks (EURUSD order-flow needs a volume feed) |

**Frozen books on the Deriv feed — training clock (true UTC - 5 h), nylate (America/New_York 13:00-22:00, DST-correct), tradeable decisions only (Deriv purchase blackout excluded)** (`deriv_r1bt_frozen_books_trainclock_nylate_dev_tradeable_result.json`; strategy gate, tick-settled, ties lose; SUPPORTED iff n >= 50 and CI95-lo >= .541):

| Book | AUC (moved) | gate rate | UP n · WR [CI95] | DOWN n · WR [CI95] | COMB n · WR | verdict UP / DOWN |
|---|---|---|---|---|---|---|
| `EURUSD.m15.v1` | .5210 | .0132 | 174 · .5000 [.4253, .5747] | 98 · .5204 [.4286, .6224] | 271 · .5055 | NOT SUPPORTED / NOT SUPPORTED |
| `EURUSD.m30.v1` | .5215 | .0154 | 119 · .4538 [.3697, .5462] | 26 · .3846 [.1923, .5769] | 145 · .4414 | NOT SUPPORTED / NOT SUPPORTED |
| `EURUSD.m30xp.v1` | — | — | — | — | — | SCHEMA_UNAVAILABLE: columns not computable from Deriv mid ticks (EURUSD order-flow needs a volume feed) |
| `EURUSD.m15xp.v1` | — | — | — | — | — | SCHEMA_UNAVAILABLE: columns not computable from Deriv mid ticks (EURUSD order-flow needs a volume feed) |

**Frozen books on the Deriv feed — true clock, ny (America/New_York 08:00-17:00, DST-correct, sessions.session_mask), tradeable decisions only (Deriv purchase blackout excluded)** (`deriv_r1t_frozen_books_dev_tradeable_result.json`; strategy gate, tick-settled, ties lose; SUPPORTED iff n >= 50 and CI95-lo >= .541):

| Book | AUC (moved) | gate rate | UP n · WR [CI95] | DOWN n · WR [CI95] | COMB n · WR | verdict UP / DOWN |
|---|---|---|---|---|---|---|
| `EURUSD.m15.v1` | .5193 | .0044 | 81 · .5802 [.4691, .6914] | 39 · .5128 [.3590, .6667] | 117 · .5556 | NOT SUPPORTED / NOT SUPPORTED |
| `EURUSD.m30.v1` | .5185 | .0259 | 234 · .4786 [.4145, .5427] | 52 · .5385 [.4038, .6731] | 285 · .4912 | NOT SUPPORTED / NOT SUPPORTED |
| `EURUSD.m30xp.v1` | — | — | — | — | — | SCHEMA_UNAVAILABLE: columns not computable from Deriv mid ticks (EURUSD order-flow needs a volume feed) |
| `EURUSD.m15xp.v1` | — | — | — | — | — | SCHEMA_UNAVAILABLE: columns not computable from Deriv mid ticks (EURUSD order-flow needs a volume feed) |

**Per-fold-refit CPCV on the Deriv feed** (DEV 2025-08-11..2026-07-31, NY DST session, 15 paths; p10/frac = path win-rate 10th percentile / share of paths >= .541; fwd = forward-walk paths clearing .541 and their mean):

| Run | Model / features | H | cov | AUC mean (min) | UP p10 · frac · mean | UP fwd | DOWN p10 · frac · mean | DOWN fwd | always UP / DOWN | legacy verdict UP / DOWN | strict certificate UP / DOWN (paths counted) | drift amendment UP / DOWN (owner decision 1) | frac @.565 UP / DOWN | result JSON |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| e10v02 | e10v02: all (82), nseed 1 | 15m | 0.03 | .5109 (.4997) | .4231 · .455 · .5260 | 0/0 · — | .4184 · .000 · .4703 | 1/2 · .5040 | .4872 / .5006 | KILLED / KILLED | no (11/15, 5 clear) / no (14/15, 0 clear) | — / — | 0.45 / 0.00 | `deriv_e10v02_EURUSD_15m_result.json` |
| e13mag50 | e13mag50: all (66), nseed 1 | 15m | 0.03 | .5089 (.4999) | .3720 · .111 · .4750 | 0/0 · — | .4334 · .154 · .4912 | 0/1 · .4545 | .4872 / .5006 | KILLED / KILLED | no (9/15, 1 clear) / no (13/15, 2 clear) | — / — | 0.11 / 0.08 | `deriv_e13mag50_EURUSD_15m_result.json` |
| e14nylate | e14nylate: all (66), nseed 1 | 15m | 0.03 | .5127 (.5011) | .4343 · .750 · .5525 | 0/1 · .4194 | .4751 · .467 · .5421 | 1/2 · .5427 | .4825 / .4990 | KILLED / KILLED | no (12/15, 9 clear) / no (15/15, 7 clear) | — / — | 0.58 / 0.40 | `deriv_e14nylate_EURUSD_15m_result.json` |
| e15nypm | e15nypm: all (66), nseed 1 | 15m | 0.03 | .5041 (.4794) | .5862 · 1.0 · .5862 | 0/0 · — | .4551 · .417 · .5184 | 0/0 · — | .4820 / .5008 | CERTIFIED / KILLED | no (1/15, 1 clear) / no (12/15, 5 clear) | — / — | 1.00 / 0.25 | `deriv_e15nypm_EURUSD_15m_result.json` |
| e16nyeve | e16nyeve: all (66), nseed 1 | 15m | 0.03 | .5163 (.5033) | .4243 · .500 · .5409 | 0/0 · — | .3826 · .400 · .5196 | 0/0 · — | .4828 / .4981 | KILLED / KILLED | no (6/15, 3 clear) / no (10/15, 4 clear) | — / — | 0.50 / 0.40 | `deriv_e16nyeve_EURUSD_15m_result.json` |
| e17nyam | e17nyam: all (66), nseed 1 | 15m | 0.03 | .5130 (.4984) | .5000 · .692 · .5723 | 0/0 · — | .4282 · .200 · .5042 | 0/2 · .4003 | .4909 / .5006 | KILLED / KILLED | no (13/15, 9 clear) / no (15/15, 3 clear) | — / — | 0.62 / 0.20 | `deriv_e17nyam_EURUSD_15m_result.json` |
| e18lowcap | e18lowcap: all (66), nseed 1 | 15m | 0.03 | .5058 (.4981) | .4163 · .200 · .4942 | 0/4 · .4801 | .4520 · .067 · .4939 | 0/4 · .4901 | .4872 / .5006 | KILLED / KILLED | no (15/15, 3 clear) / no (15/15, 1 clear) | — / — | 0.13 / 0.07 | `deriv_e18lowcap_EURUSD_15m_result.json` |
| e19tf | e19tf: all+fam:tf (74), nseed 1 | 15m | 0.03 | .5095 (.5019) | .4519 · .467 · .5169 | 0/2 · .5250 | .4488 · .267 · .5165 | 1/4 · .4897 | .4872 / .5006 | KILLED / KILLED | no (15/15, 7 clear) / no (15/15, 4 clear) | — / — | 0.33 / 0.27 | `deriv_e19tf_EURUSD_15m_result.json` |
| e1 | e1: all (66), nseed 1 | 15m | 0.03 | .5116 (.5044) | .4500 · .143 · .4912 | 1/2 · .5333 | .5000 · .400 · .5360 | 2/4 · .5481 | .4869 / .5006 | KILLED / KILLED | no (14/15, 2 clear) / no (15/15, 6 clear) | — / — | 0.14 / 0.20 | `deriv_e1_EURUSD_15m_result.json` |
| e1t | e1t: all (66), nseed 1 | 15m | 0.03 | .5089 (.4999) | .4162 · .071 · .4719 | 0/1 · .4444 | .4636 · .333 · .5143 | 1/4 · .4944 | .4872 / .5006 | KILLED / KILLED | no (14/15, 1 clear) / no (15/15, 5 clear) | — / — | 0.07 / 0.07 | `deriv_e1t_EURUSD_15m_result.json` |
| e20sg | e20sg: all+fam:sg (71), nseed 1 | 15m | 0.03 | .5099 (.4980) | .4404 · .071 · .5081 | 0/1 · .3462 | .4521 · .333 · .5282 | 1/4 · .5237 | .4872 / .5006 | KILLED / KILLED | no (14/15, 1 clear) / no (15/15, 5 clear) | — / — | 0.07 / 0.27 | `deriv_e20sg_EURUSD_15m_result.json` |
| e21kf | e21kf: all+fam:kf (73), nseed 1 | 15m | 0.03 | .5103 (.5017) | .4200 · .429 · .5138 | 0/0 · — | .4394 · .333 · .5124 | 1/4 · .4811 | .4872 / .5006 | KILLED / KILLED | no (14/15, 6 clear) / no (15/15, 5 clear) | — / — | 0.43 / 0.13 | `deriv_e21kf_EURUSD_15m_result.json` |
| e22rm | e22rm: all+fam:rm (70), nseed 1 | 15m | 0.03 | .5092 (.5003) | .3908 · .214 · .4826 | 0/0 · — | .4672 · .333 · .5234 | 1/4 · .5270 | .4872 / .5006 | KILLED / KILLED | no (14/15, 3 clear) / no (15/15, 5 clear) | — / — | 0.00 / 0.13 | `deriv_e22rm_EURUSD_15m_result.json` |
| e23fd | e23fd: all+fam:fd (68), nseed 1 | 15m | 0.03 | .5088 (.5007) | .4107 · .143 · .4830 | 0/1 · .5385 | .4418 · .400 · .5199 | 1/4 · .5406 | .4872 / .5006 | KILLED / KILLED | no (14/15, 2 clear) / no (15/15, 6 clear) | — / — | 0.00 / 0.27 | `deriv_e23fd_EURUSD_15m_result.json` |
| e24cx | e24cx: all+fam:cx (74), nseed 1 | 15m | 0.03 | .5096 (.4983) | .4000 · .214 · .4920 | 0/1 · .4706 | .4453 · .267 · .5110 | 2/4 · .5377 | .4872 / .5006 | KILLED / KILLED | no (14/15, 3 clear) / no (15/15, 4 clear) | — / — | 0.14 / 0.13 | `deriv_e24cx_EURUSD_15m_result.json` |
| e25tk | e25tk: all+fam:tk (72), nseed 1 | 15m | 0.03 | .5096 (.5018) | .4083 · .143 · .4911 | 0/0 · — | .4666 · .133 · .5012 | 1/4 · .5247 | .4872 / .5006 | KILLED / KILLED | no (14/15, 2 clear) / no (15/15, 2 clear) | — / — | 0.00 / 0.07 | `deriv_e25tk_EURUSD_15m_result.json` |
| e26pa | e26pa: all+fam:pa (78), nseed 1 | 15m | 0.03 | .5103 (.5040) | .3939 · .182 · .4845 | 0/0 · — | .3725 · .286 · .4975 | 0/2 · .4712 | .4872 / .5006 | KILLED / KILLED | no (11/15, 2 clear) / no (14/15, 4 clear) | — / — | 0.09 / 0.21 | `deriv_e26pa_EURUSD_15m_result.json` |
| e27famall | e27famall: all+fam:tf,sg,kf,rm,fd,cx,tk,pa (118), nseed 1 | 15m | 0.03 | .5102 (.5042) | .4194 · .091 · .4796 | 0/0 · — | .4590 · .538 · .5349 | 0/2 · .4387 | .4872 / .5006 | KILLED / KILLED | no (11/15, 1 clear) / no (13/15, 7 clear) | — / — | 0.09 / 0.31 | `deriv_e27famall_EURUSD_15m_result.json` |
| e28resid | e28resid: all (66), nseed 1 | 15m | 0.03 | .5110 (.4982) | .4433 · .200 · .5040 | 0/4 · .4988 | .4484 · .267 · .5082 | 0/4 · .4880 | .4872 / .5006 | KILLED / KILLED | no (15/15, 3 clear) / no (15/15, 4 clear) | — / — | 0.13 / 0.13 | `deriv_e28resid_EURUSD_15m_result.json` |
| e29logit | e29logit: all (66), nseed 1 | 15m | 0.03 | .5100 (.5021) | .4666 · .267 · .5205 | 2/4 · .5264 | .4773 · .267 · .5148 | 0/4 · .4957 | .4872 / .5006 | KILLED / KILLED | no (15/15, 4 clear) / no (15/15, 4 clear) | — / — | 0.13 / 0.00 | `deriv_e29logit_EURUSD_15m_result.json` |
| e2noclock | e2noclock: noclock (63), nseed 1 | 15m | 0.03 | .5085 (.4983) | .4509 · .462 · .5218 | 0/1 · .4667 | .4614 · .400 · .5268 | 1/4 · .4799 | .4872 / .5006 | KILLED / KILLED | no (13/15, 6 clear) / no (15/15, 6 clear) | — / — | 0.31 / 0.13 | `deriv_e2noclock_EURUSD_15m_result.json` |
| e30nystroem | e30nystroem: all (66), nseed 1 | 15m | 0.03 | .5038 (.4897) | .4447 · .077 · .4869 | 0/3 · .4383 | .4497 · .133 · .5036 | 1/4 · .5350 | .4872 / .5006 | KILLED / KILLED | no (13/15, 1 clear) / no (15/15, 2 clear) | — / — | 0.00 / 0.07 | `deriv_e30nystroem_EURUSD_15m_result.json` |
| e31ens3 | e31ens3: all (66), nseed 1 | 15m | 0.03 | .5102 (.5025) | .5000 · .429 · .5465 | 0/0 · — | .4516 · .385 · .5251 | 0/0 · — | .4872 / .5006 | KILLED / KILLED | no (14/15, 6 clear) / no (13/15, 5 clear) | — / — | 0.43 / 0.08 | `deriv_e31ens3_EURUSD_15m_result.json` |
| e32spec | e32spec: all (66), nseed 1 | 15m | 0.03 | .5064 (.5011) | .4275 · .067 · .4791 | 1/4 · .4818 | .4566 · .333 · .5179 | 1/4 · .5052 | .4872 / .5006 | KILLED / KILLED | no (15/15, 1 clear) / no (15/15, 5 clear) | — / — | 0.00 / 0.20 | `deriv_e32spec_EURUSD_15m_result.json` |
| e33comp33 | e33comp33: all (66), nseed 1 | 15m | 0.1 | .5089 (.4999) | .4416 · .182 · .5001 | 2/2 · .5872 | .4713 · .333 · .5296 | 3/3 · .5849 | .4872 / .5006 | KILLED / KILLED | no (11/15, 2 clear) / no (15/15, 5 clear) | — / — | 0.18 / 0.20 | `deriv_e33comp33_EURUSD_15m_result.json` |
| e34online | e34online: all (66), nseed 1 | 15m | 0.03 | .5050 (.4972) | .4652 · .000 · .4809 | 0/0 · — | .4773 · .133 · .5050 | 0/0 · — | .4872 / .5006 | KILLED / KILLED | no (15/15, 0 clear; never certified: a continuing policy: the online learner updates on settled outcomes during each test path) / no (15/15, 2 clear; never certified: a continuing policy: the online learner updates on settled outcomes during each test path) | — / — | 0.00 / 0.00 | `deriv_e34online_EURUSD_15m_result.json` |
| e35absret | e35absret: all (66), nseed 1 | 15m | 0.03 | .5102 (.5004) | .4267 · .167 · .5027 | 0/0 · — | .4610 · .462 · .5377 | 1/4 · .4967 | .4872 / .5006 | KILLED / KILLED | no (12/15, 2 clear) / no (13/15, 6 clear) | — / — | 0.17 / 0.38 | `deriv_e35absret_EURUSD_15m_result.json` |
| e36aci | e36aci: all (66), nseed 1 | 15m | 0.03 | .5089 (.4999) | .4000 · .071 · .4492 | 0/1 · .4444 | .4737 · .200 · .5123 | 1/3 · .5008 | .4872 / .5006 | KILLED / KILLED | no (14/15, 1 clear; never certified: an adaptive threshold: ACI moves it with settled test outcomes) / no (15/15, 3 clear; never certified: an adaptive threshold: ACI moves it with settled test outcomes) | — / — | 0.07 / 0.07 | `deriv_e36aci_EURUSD_15m_result.json` |
| e37meta | e37meta: all (66), nseed 1 | 15m | 0.06 | .5089 (.4999) | .4444 · .286 · .5069 | 0/2 · .4380 | .4564 · .267 · .5063 | 2/4 · .5636 | .4872 / .5006 | KILLED / KILLED | no (14/15, 4 clear) / no (15/15, 4 clear) | — / — | 0.21 / 0.00 | `deriv_e37meta_EURUSD_15m_result.json` |
| e38release | e38release: all (66), nseed 1 | 15m | 0.03 | .5041 (.5026) | .4750 · .400 · .5294 | 1/4 · .5053 | .4705 · .333 · .5117 | 0/4 · .4854 | .4872 / .5006 | KILLED / KILLED | no (15/15, 6 clear) / no (15/15, 5 clear) | — / — | 0.27 / 0.13 | `deriv_e38release_EURUSD_15m_result.json` |
| e39cm | e39cm: all+fam:cm (75), nseed 1 | 15m | 0.03 | .5125 (.5006) | .4474 · .273 · .5119 | 0/0 · — | .5000 · .615 · .5664 | 2/2 · .6298 | .4872 / .5006 | KILLED / KILLED | no (11/15, 3 clear) / no (13/15, 8 clear) | — / — | 0.09 / 0.62 | `deriv_e39cm_EURUSD_15m_result.json` |
| e3union | e3union: all (380), nseed 1 | 15m | 0.03 | .5148 (.5078) | .4181 · .250 · .5132 | 0/0 · — | .4571 · .364 · .5431 | 0/0 · — | .4872 / .5006 | KILLED / KILLED | no (8/15, 2 clear) / no (11/15, 4 clear) | — / — | 0.25 / 0.36 | `deriv_e3union_EURUSD_15m_result.json` |
| e40forest | e40forest: all (66), nseed 1 | 15m | 0.03 | .5095 (.5036) | .4514 · .182 · .4956 | 0/2 · .5063 | .5013 · .067 · .5209 | 1/4 · .5343 | .4872 / .5006 | KILLED / KILLED | no (11/15, 2 clear) / no (15/15, 1 clear) | — / — | 0.00 / 0.07 | `deriv_e40forest_EURUSD_15m_result.json` |
| e41pool | e41pool: all (66), nseed 1 | 15m | 0.03 | .5094 (.5035) | .4200 · .267 · .4884 | 1/4 · .4541 | .4880 · .333 · .5266 | 2/4 · .5738 | .4872 / .5006 | KILLED / KILLED | no (15/15, 4 clear) / no (15/15, 5 clear) | — / — | 0.13 / 0.13 | `deriv_e41pool_EURUSD_15m_result.json` |
| e4seed3 | e4seed3: all (66), nseed 3 | 15m | 0.03 | .5096 (.5004) | .4155 · .143 · .4747 | 0/1 · .4815 | .4615 · .200 · .5062 | 1/4 · .4881 | .4872 / .5006 | KILLED / KILLED | no (14/15, 2 clear) / no (15/15, 3 clear) | — / — | 0.00 / 0.13 | `deriv_e4seed3_EURUSD_15m_result.json` |
| e51fx | e51fx: all+fam:fx (82), nseed 1 | 15m | 0.03 | .5109 (.5019) | .4226 · .267 · .5015 | 0/1 · .4857 | .4369 · .133 · .4883 | 1/3 · .5272 | .4872 / .5006 | KILLED / KILLED | no (15/15, 4 clear) / no (15/15, 2 clear) | — / — | 0.20 / 0.00 | `deriv_e51fx_EURUSD_15m_result.json` |
| e52ck | e52ck: all+fam:ck (77), nseed 1 | 15m | 0.03 | .5085 (.4994) | .3794 · .067 · .4627 | 0/2 · .4807 | .4564 · .200 · .5100 | 0/4 · .5068 | .4872 / .5006 | KILLED / KILLED | no (15/15, 1 clear) / no (15/15, 3 clear) | — / — | 0.07 / 0.07 | `deriv_e52ck_EURUSD_15m_result.json` |
| e53ckcompact | e53ckcompact: list:compact_kernels.json (14), nseed 1 | 15m | 0.03 | .4988 (.4901) | .4437 · .067 · .4799 | 0/4 · .4757 | .4574 · .067 · .4885 | 1/4 · .4929 | .4872 / .5006 | KILLED / KILLED | no (15/15, 1 clear) / no (15/15, 1 clear) | — / — | 0.00 / 0.00 | `deriv_e53ckcompact_EURUSD_15m_result.json` |
| e54ckcompactlogit | e54ckcompactlogit: list:compact_kernels.json (14), nseed 1 | 15m | 0.03 | .5098 (.4942) | .4682 · .467 · .5397 | 1/4 · .5096 | .5122 · .533 · .5427 | 1/4 · .5303 | .4872 / .5006 | KILLED / KILLED | no (15/15, 7 clear) / no (15/15, 8 clear) | — / — | 0.33 / 0.13 | `deriv_e54ckcompactlogit_EURUSD_15m_result.json` |
| e55rl | e55rl: all+fam:rl (72), nseed 1 | 15m | 0.03 | .5102 (.4977) | .4649 · .692 · .5556 | 0/0 · — | .4418 · .200 · .5037 | 1/2 · .5153 | .4872 / .5006 | KILLED / KILLED | no (13/15, 9 clear) / no (15/15, 3 clear) | — / — | 0.46 / 0.13 | `deriv_e55rl_EURUSD_15m_result.json` |
| e56rs | e56rs: all+fam:rs (72), nseed 1 | 15m | 0.03 | .5098 (.5024) | .4106 · .300 · .5071 | 0/1 · .4483 | .4022 · .143 · .4708 | 1/3 · .5229 | .4872 / .5006 | KILLED / KILLED | no (10/15, 3 clear) / no (14/15, 2 clear) | — / — | 0.20 / 0.07 | `deriv_e56rs_EURUSD_15m_result.json` |
| e57reflect | e57reflect: all (66), nseed 1 | 15m | 0.03 | .5079 (.4997) | .3538 · .067 · .4595 | 0/2 · .4626 | .4312 · .533 · .5463 | 0/2 · .4978 | .4872 / .5006 | KILLED / KILLED | no (15/15, 1 clear) / no (15/15, 8 clear) | — / — | 0.00 / 0.47 | `deriv_e57reflect_EURUSD_15m_result.json` |
| e58nysreflect | e58nysreflect: all (66), nseed 1 | 15m | 0.03 | .5102 (.5011) | .4496 · .000 · .4856 | 2/4 · .5039 | .4575 · .231 · .5170 | 0/2 · .5234 | .4872 / .5006 | KILLED / KILLED | no (13/15, 0 clear) / no (13/15, 3 clear) | — / — | 0.00 / 0.08 | `deriv_e58nysreflect_EURUSD_15m_result.json` |
| e59brier | e59brier: all (66), nseed 1 | 15m | 0.03 | .5089 (.4997) | .3946 · .286 · .4915 | 0/0 · — | .4587 · .267 · .5070 | 1/4 · .5224 | .4872 / .5006 | KILLED / KILLED | no (14/15, 4 clear) / no (15/15, 4 clear) | — / — | 0.21 / 0.07 | `deriv_e59brier_EURUSD_15m_result.json` |
| e5h30 | e5h30: all (66), nseed 1 | 30m | 0.03 | .5042 (.4917) | .3819 · .429 · .4903 | 0/0 · — | .4194 · .182 · .4859 | 0/0 · — | .4833 / .5104 | KILLED / KILLED | no (7/15, 3 clear) / no (11/15, 2 clear) | — / — | 0.14 / 0.18 | `deriv_e5h30_EURUSD_30m_result.json` |
| e60stack | e60stack: all (66), nseed 1 | 15m | 0.03 | .5091 (.5012) | .4155 · .200 · .5013 | 0/2 · .3740 | .4433 · .400 · .5189 | 3/4 · .5502 | .4872 / .5006 | KILLED / KILLED | no (15/15, 3 clear) / no (15/15, 6 clear) | — / — | 0.13 / 0.27 | `deriv_e60stack_EURUSD_15m_result.json` |
| e61ord4 | e61ord4: all (66), nseed 1 | 15m | 0.03 | .5085 (.4983) | .3813 · .333 · .4698 | 0/1 · .5200 | .4395 · .400 · .5116 | 2/4 · .5376 | .4872 / .5006 | KILLED / KILLED | no (12/15, 4 clear) / no (15/15, 6 clear) | — / — | 0.25 / 0.27 | `deriv_e61ord4_EURUSD_15m_result.json` |
| e62har | e62har: all (66), nseed 1 | 15m | 0.03 | .5089 (.4999) | .3930 · .200 · .4844 | 0/0 · — | .4012 · .250 · .4803 | 0/1 · .4688 | .4872 / .5006 | KILLED / KILLED | no (10/15, 2 clear) / no (12/15, 3 clear) | — / — | 0.20 / 0.08 | `deriv_e62har_EURUSD_15m_result.json` |
| e64boot | e64boot: all (66), nseed 1 | 15m | 0.03 | .5082 (.5002) | .4508 · .133 · .4906 | 0/1 · .3871 | .4519 · .267 · .4989 | 2/4 · .5655 | .4872 / .5006 | KILLED / KILLED | no (15/15, 2 clear) / no (15/15, 4 clear) | — / — | 0.00 / 0.00 | `deriv_e64boot_EURUSD_15m_result.json` |
| e65recency | e65recency: all (66), nseed 1 | 15m | 0.03 | .5076 (.4995) | .4800 · .364 · .5310 | 0/0 · — | .4344 · .538 · .5379 | 0/2 · .4524 | .4872 / .5006 | KILLED / KILLED | no (11/15, 4 clear) / no (13/15, 7 clear) | — / — | 0.36 / 0.31 | `deriv_e65recency_EURUSD_15m_result.json` |
| e66mh | e66mh: all (66), nseed 1 | 15m | 0.03 | .5092 (.4999) | .3608 · .167 · .4367 | 0/0 · — | .3842 · .222 · .4869 | 0/0 · — | .4872 / .5006 | KILLED / KILLED | no (6/15, 1 clear) / no (9/15, 2 clear) | — / — | 0.00 / 0.22 | `deriv_e66mh_EURUSD_15m_result.json` |
| e67drfe | e67drfe: all (66), nseed 1 | 15m | 0.03 | .5168 (.5078) | .4920 · .467 · .5409 | 2/4 · .5474 | .4324 · .267 · .5106 | 1/4 · .4809 | .4872 / .5006 | KILLED / KILLED | no (15/15, 7 clear) / no (15/15, 4 clear) | — / — | 0.20 / 0.13 | `deriv_e67drfe_EURUSD_15m_result.json` |
| e68agree | e68agree: all (66), nseed 1 | 15m | 0.03 | .5089 (.4998) | .4153 · .214 · .4913 | 0/1 · .4375 | .4498 · .200 · .5013 | 3/4 · .5537 | .4872 / .5006 | KILLED / KILLED | no (14/15, 3 clear) / no (15/15, 3 clear) | — / — | 0.07 / 0.13 | `deriv_e68agree_EURUSD_15m_result.json` |
| e6pool | e6pool: own (41), nseed 1, pooled +6 pairs | 15m | 0.03 | .5064 (.4906) | .4489 · .267 · .5155 | 1/4 · .5073 | .4727 · .400 · .5278 | 4/4 · .5976 | .4872 / .5006 | KILLED / KILLED | no (15/15, 4 clear) / no (15/15, 6 clear) | — / — | 0.13 / 0.13 | `deriv_e6pool_EURUSD_15m_result.json` |
| e70play | e70play: all+fam:hy (75), nseed 1 | 15m | 0.03 | .5091 (.4999) | .4297 · .214 · .5005 | 0/1 · .4815 | .4557 · .400 · .5160 | 1/4 · .4986 | .4872 / .5006 | KILLED / KILLED | no (14/15, 3 clear) / no (15/15, 6 clear) | — / — | 0.21 / 0.20 | `deriv_e70play_EURUSD_15m_result.json` |
| e71playreset | e71playreset: all+fam:hr (75), nseed 1 | 15m | 0.03 | .5094 (.4998) | .3958 · .214 · .4898 | 0/1 · .5000 | .4458 · .333 · .5161 | 2/4 · .5219 | .4872 / .5006 | KILLED / KILLED | no (14/15, 3 clear) / no (15/15, 5 clear) | — / — | 0.21 / 0.27 | `deriv_e71playreset_EURUSD_15m_result.json` |
| e72shock | e72shock: all+fam:d2 (67), nseed 1 | 15m | 0.03 | .5096 (.5009) | .3640 · .214 · .4686 | 0/1 · .5000 | .4398 · .467 · .5067 | 2/3 · .5670 | .4872 / .5006 | KILLED / KILLED | no (14/15, 3 clear) / no (15/15, 7 clear) | — / — | 0.07 / 0.07 | `deriv_e72shock_EURUSD_15m_result.json` |
| e73mono | e73mono: all (66), nseed 1 | 15m | 0.03 | .5101 (.5007) | .3869 · .071 · .4644 | 1/1 · .5676 | .4318 · .333 · .5044 | 2/4 · .5293 | .4872 / .5006 | KILLED / KILLED | no (14/15, 1 clear) / no (15/15, 5 clear) | — / — | 0.07 / 0.07 | `deriv_e73mono_EURUSD_15m_result.json` |
| e74logitsig | e74logitsig: all (66), nseed 1 | 15m | 0.03 | .5143 (.5083) | .4482 · .267 · .5116 | 2/4 · .5385 | .4733 · .200 · .5157 | 0/4 · .4958 | .4872 / .5006 | KILLED / KILLED | no (15/15, 4 clear) / no (15/15, 3 clear) | — / — | 0.20 / 0.07 | `deriv_e74logitsig_EURUSD_15m_result.json` |
| e75lgbsig | e75lgbsig: all (66), nseed 1 | 15m | 0.03 | .5086 (.5023) | .4597 · .400 · .5221 | 2/4 · .4983 | .4787 · .533 · .5394 | 2/4 · .5697 | .4872 / .5006 | KILLED / KILLED | no (15/15, 6 clear) / no (15/15, 8 clear) | — / — | 0.20 / 0.27 | `deriv_e75lgbsig_EURUSD_15m_result.json` |
| e76drfelgb | e76drfelgb: all (66), nseed 1 | 15m | 0.03 | .5113 (.5034) | .4860 · .400 · .5296 | 1/4 · .5095 | .4778 · .400 · .5369 | 1/4 · .5223 | .4872 / .5006 | KILLED / KILLED | no (15/15, 6 clear) / no (15/15, 6 clear) | — / — | 0.13 / 0.33 | `deriv_e76drfelgb_EURUSD_15m_result.json` |
| e79qorder | e79qorder: all+fam:qo (88), nseed 1 | 15m | 0.03 | .5107 (.5020) | .4000 · .214 · .4761 | 0/1 · .4483 | .4824 · .333 · .5493 | 1/2 · .5492 | .4872 / .5006 | KILLED / KILLED | no (14/15, 3 clear) / no (15/15, 5 clear) | — / — | 0.14 / 0.20 | `deriv_e79qorder_EURUSD_15m_result.json` |
| e7fitall | e7fitall: all (66), nseed 1 | 15m | 0.03 | .5073 (.4963) | .5038 · .400 · .5369 | 1/2 · .5658 | .4250 · .133 · .4917 | 2/4 · .5094 | .4872 / .5006 | KILLED / KILLED | no (15/15, 6 clear) / no (15/15, 2 clear) | — / — | 0.27 / 0.07 | `deriv_e7fitall_EURUSD_15m_result.json` |
| e80qperm | e80qperm: all+fam:qp (88), nseed 1 | 15m | 0.03 | .5093 (.5019) | .4081 · .308 · .4888 | 0/0 · — | .4131 · .429 · .5058 | 0/3 · .4745 | .4872 / .5006 | KILLED / KILLED | no (13/15, 4 clear) / no (14/15, 6 clear) | — / — | 0.08 / 0.29 | `deriv_e80qperm_EURUSD_15m_result.json` |
| e81tail0 | e81tail0: all (66), nseed 1 | 15m | 0.03 | .5044 (.4921) | .4212 · .125 · .4971 | 0/1 · .4545 | .4552 · .467 · .5228 | 2/3 · .5764 | .4872 / .5006 | KILLED / KILLED | no (8/15, 1 clear) / no (15/15, 7 clear) | — / — | 0.12 / 0.27 | `deriv_e81tail0_EURUSD_15m_result.json` |
| e82tailaux | e82tailaux: all (66), nseed 1 | 15m | 0.03 | .5014 (.4879) | .3863 · .000 · .4710 | 0/4 · .4390 | .4355 · .714 · .5576 | 0/0 · — | .4872 / .5006 | KILLED / KILLED | no (10/15, 0 clear) / no (14/15, 10 clear) | — / — | 0.00 / 0.43 | `deriv_e82tailaux_EURUSD_15m_result.json` |
| e87fixrank | e87fixrank: list:compact_kernels.json (14), nseed 1 | 15m | 0.03 | .5034 (.5007) | nan · .000 · — | 0/0 · — | .5036 · .667 · .5516 | 1/4 · .5004 | .4872 / .5006 | KILLED / KILLED | no (0/15, 0 clear) / no (15/15, 10 clear) | — / — | — / 0.27 | `deriv_e87fixrank_EURUSD_15m_result.json` |
| e8tb2 | e8tb2: all (66), nseed 1 | 15m | 0.03 | .5062 (.4940) | .4179 · .385 · .5253 | 0/1 · .3889 | .4773 · .333 · .5269 | 1/3 · .5049 | .4872 / .5006 | KILLED / KILLED | no (13/15, 5 clear) / no (15/15, 5 clear) | — / — | 0.23 / 0.27 | `deriv_e8tb2_EURUSD_15m_result.json` |
| e9blockval | e9blockval: all (66), nseed 1 | 15m | 0.03 | .5112 (.5025) | .4850 · .600 · .5358 | 2/4 · .5512 | .4369 · .200 · .5034 | 2/4 · .5546 | .4872 / .5006 | KILLED / KILLED | no (15/15, 9 clear) / no (15/15, 3 clear) | — / — | 0.20 / 0.00 | `deriv_e9blockval_EURUSD_15m_result.json` |
| r2t | r2t: list:EURUSD.m15xp.v1.json (313), nseed 1 | 15m | 0.1 | .5134 (.5016) | .4046 · .071 · .4591 | 1/3 · .5300 | .4896 · .400 · .5310 | 1/4 · .5089 | .4872 / .5006 | KILLED / KILLED | no (14/15, 1 clear) / no (15/15, 6 clear) | — / — | 0.00 / 0.27 | `deriv_r2t_EURUSD_15m_result.json` |
| r3 | r3: list:EURUSD.m15xp.v1.json (313), nseed 1 | 15m | 0.1 | .5214 (.5141) | .4421 · .267 · .5245 | 0/2 · .4460 | .4418 · .200 · .5040 | 1/4 · .5300 | .4828 / .4986 | KILLED / KILLED | no (15/15, 4 clear) / no (15/15, 3 clear) | — / — | 0.27 / 0.13 | `deriv_r3_EURUSD_15m_result.json` |
| r4fixldn | r4fixldn: rule:fx_ldn_prefix_u60 (1), nseed 0 | 15m | 1.0 | .5315 (.5144) | .4705 · .200 · .5060 | 0/0 · — | .5202 · .600 · .5381 | 0/0 · — | .4872 / .5006 | KILLED / KILLED | no (15/15, 3 clear) / no (15/15, 9 clear) | — / — | 0.00 / 0.07 | `deriv_r4fixldn_EURUSD_15m_result.json` |
| r5fixecb | r5fixecb: rule:fx_ecb_prefix_u60 (1), nseed 0 | 15m | 1.0 | .5271 (.5107) | .4950 · .000 · .5098 | 0/0 · — | .5052 · .533 · .5398 | 0/0 · — | .4872 / .5006 | KILLED / KILLED | no (15/15, 0 clear) / no (15/15, 8 clear) | — / — | 0.00 / 0.20 | `deriv_r5fixecb_EURUSD_15m_result.json` |
| r6n-a003-p | r6n-a003-p: all (60), nseed 1 | 15m | 0.03 | .5013 (.4971) | .4714 · .000 · .4909 | 0/0 · — | .4956 · .000 · .5031 | 0/0 · — | .4863 / .5018 | PILOT (three balanced paths) / PILOT (three balanced paths) | no (3/3, 0 clear; incomplete: 3 of 15 paths evaluated; never certified: PILOT (three balanced paths)) / no (3/3, 0 clear; incomplete: 3 of 15 paths evaluated; never certified: PILOT (three balanced paths)) | — / — | 0.00 / 0.00 | `deriv_r6n-a003-p_EURUSD_15m_result.json` |
| r6n-a003fixed-f | r6n-a003fixed-f: all (60), nseed 1 | 15m | 0.03 | .5090 (.4993) | .4309 · .500 · .5004 | 0/4 · .4989 | .5000 · .333 · .5270 | 1/3 · .5365 | .4863 / .5018 | KILLED / KILLED | no (14/15, 7 clear) / no (15/15, 5 clear) | — / — | 0.07 / 0.13 | `deriv_r6n-a003fixed-f_EURUSD_15m_result.json` |
| r6n-a003fixed-p | r6n-a003fixed-p: all (60), nseed 1 | 15m | 0.03 | .5046 (.4993) | .4301 · .000 · .4429 | 0/0 · — | .5156 · .333 · .5314 | 0/0 · — | .4863 / .5018 | PILOT (three balanced paths) / PILOT (three balanced paths) | no (3/3, 0 clear; incomplete: 3 of 15 paths evaluated; never certified: PILOT (three balanced paths)) / no (3/3, 1 clear; incomplete: 3 of 15 paths evaluated; never certified: PILOT (three balanced paths)) | — / — | 0.00 / 0.00 | `deriv_r6n-a003fixed-p_EURUSD_15m_result.json` |
| r6n-a003linear-p | r6n-a003linear-p: all (60), nseed 1 | 15m | 0.03 | .5023 (.4984) | .4902 · .000 · .5115 | 0/0 · — | .4869 · .000 · .5065 | 0/0 · — | .4863 / .5018 | PILOT (three balanced paths) / PILOT (three balanced paths) | no (3/3, 0 clear; incomplete: 3 of 15 paths evaluated; never certified: PILOT (three balanced paths)) / no (3/3, 0 clear; incomplete: 3 of 15 paths evaluated; never certified: PILOT (three balanced paths)) | — / — | 0.00 / 0.00 | `deriv_r6n-a003linear-p_EURUSD_15m_result.json` |
| r6n-a003permuted-f | r6n-a003permuted-f: all (60), nseed 1 | 15m | 0.03 | .5059 (.4967) | .4807 · .067 · .5058 | 1/4 · .5249 | .4933 · .267 · .5260 | 0/4 · .4942 | .4863 / .5018 | KILLED / KILLED | no (15/15, 1 clear) / no (15/15, 4 clear) | — / — | 0.00 / 0.00 | `deriv_r6n-a003permuted-f_EURUSD_15m_result.json` |
| r6n-a003permuted-p | r6n-a003permuted-p: all (60), nseed 1 | 15m | 0.03 | .5009 (.4967) | .4756 · .000 · .4944 | 0/0 · — | .5257 · .333 · .5377 | 0/0 · — | .4863 / .5018 | PILOT (three balanced paths) / PILOT (three balanced paths) | no (3/3, 0 clear; incomplete: 3 of 15 paths evaluated; never certified: PILOT (three balanced paths)) / no (3/3, 1 clear; incomplete: 3 of 15 paths evaluated; never certified: PILOT (three balanced paths)) | — / — | 0.00 / 0.00 | `deriv_r6n-a003permuted-p_EURUSD_15m_result.json` |
| r6n-a005-f | r6n-a005-f: all (60), nseed 1 | 15m | 0.03 | .5075 (.4985) | .5041 · .600 · .5445 | 1/3 · .5640 | .4653 · .067 · .5077 | 0/4 · .5092 | .4863 / .5018 | KILLED / KILLED | no (15/15, 9 clear) / no (15/15, 1 clear) | — / — | 0.40 / 0.07 | `deriv_r6n-a005-f_EURUSD_15m_result.json` |
| r6n-a005-p | r6n-a005-p: all (60), nseed 1 | 15m | 0.03 | .5009 (.4985) | .5019 · .667 · .5424 | 0/0 · — | .4738 · .000 · .5008 | 0/0 · — | .4863 / .5018 | PILOT (three balanced paths) / PILOT (three balanced paths) | no (3/3, 2 clear; incomplete: 3 of 15 paths evaluated; never certified: PILOT (three balanced paths)) / no (3/3, 0 clear; incomplete: 3 of 15 paths evaluated; never certified: PILOT (three balanced paths)) | — / — | 0.67 / 0.00 | `deriv_r6n-a005-p_EURUSD_15m_result.json` |
| r6n-a005conv-f | r6n-a005conv-f: all (60), nseed 1 | 15m | 0.03 | .5064 (.4942) | .4414 · .231 · .4933 | 0/3 · .5289 | .4700 · .133 · .5100 | 0/3 · .5099 | .4863 / .5018 | KILLED / KILLED | no (13/15, 3 clear) / no (15/15, 2 clear) | — / — | 0.15 / 0.07 | `deriv_r6n-a005conv-f_EURUSD_15m_result.json` |
| r6n-a005conv-p | r6n-a005conv-p: all (60), nseed 1 | 15m | 0.03 | .5002 (.4942) | .3691 · .000 · .4211 | 0/0 · — | .4189 · .000 · .4853 | 0/0 · — | .4863 / .5018 | PILOT (three balanced paths) / PILOT (three balanced paths) | no (3/3, 0 clear; incomplete: 3 of 15 paths evaluated; never certified: PILOT (three balanced paths)) / no (3/3, 0 clear; incomplete: 3 of 15 paths evaluated; never certified: PILOT (three balanced paths)) | — / — | 0.00 / 0.00 | `deriv_r6n-a005conv-p_EURUSD_15m_result.json` |
| r6n-a005gru-f | r6n-a005gru-f: all (60), nseed 1 | 15m | 0.03 | .5066 (.4946) | .4455 · .000 · .4922 | 0/3 · .4372 | .4876 · .286 · .5213 | 1/4 · .5284 | .4863 / .5018 | KILLED / KILLED | no (12/15, 0 clear) / no (14/15, 4 clear) | — / — | 0.00 / 0.14 | `deriv_r6n-a005gru-f_EURUSD_15m_result.json` |
| r6n-a005gru-p | r6n-a005gru-p: all (60), nseed 1 | 15m | 0.03 | .4968 (.4946) | .4536 · .000 · .4862 | 0/0 · — | .4675 · .000 · .4900 | 0/0 · — | .4863 / .5018 | PILOT (three balanced paths) / PILOT (three balanced paths) | no (3/3, 0 clear; incomplete: 3 of 15 paths evaluated; never certified: PILOT (three balanced paths)) / no (3/3, 0 clear; incomplete: 3 of 15 paths evaluated; never certified: PILOT (three balanced paths)) | — / — | 0.00 / 0.00 | `deriv_r6n-a005gru-p_EURUSD_15m_result.json` |
| r6n-a005linear-f | r6n-a005linear-f: all (60), nseed 1 | 15m | 0.03 | .5095 (.4984) | .5002 · .333 · .5252 | 2/4 · .5108 | .4934 · .333 · .5331 | 0/4 · .4997 | .4863 / .5018 | KILLED / KILLED | no (15/15, 5 clear) / no (15/15, 5 clear) | — / — | 0.07 / 0.13 | `deriv_r6n-a005linear-f_EURUSD_15m_result.json` |
| r6n-a005linear-p | r6n-a005linear-p: all (60), nseed 1 | 15m | 0.03 | .5024 (.4984) | .5177 · .333 · .5390 | 0/0 · — | .4816 · .000 · .5017 | 0/0 · — | .4863 / .5018 | PILOT (three balanced paths) / PILOT (three balanced paths) | no (3/3, 1 clear; incomplete: 3 of 15 paths evaluated; never certified: PILOT (three balanced paths)) / no (3/3, 0 clear; incomplete: 3 of 15 paths evaluated; never certified: PILOT (three balanced paths)) | — / — | 0.33 / 0.00 | `deriv_r6n-a005linear-p_EURUSD_15m_result.json` |
| r6n-a016-p | r6n-a016-p: all (60), nseed 1 | 15m | 0.03 | .4833 (.4732) | .3820 · .000 · .4232 | 0/0 · — | .4648 · .000 · .4897 | 0/0 · — | .4863 / .5018 | PILOT (three balanced paths) / PILOT (three balanced paths) | no (3/3, 0 clear; incomplete: 3 of 15 paths evaluated; never certified: PILOT (three balanced paths)) / no (3/3, 0 clear; incomplete: 3 of 15 paths evaluated; never certified: PILOT (three balanced paths)) | — / — | 0.00 / 0.00 | `deriv_r6n-a016-p_EURUSD_15m_result.json` |
| r6n-a069-f | r6n-a069-f: all (60), nseed 1 | 15m | 0.03 | .5068 (.4958) | .4623 · .182 · .5122 | 0/4 · .4965 | .4965 · .067 · .5168 | 0/4 · .4968 | .4863 / .5018 | KILLED / KILLED | no (11/15, 2 clear) / no (15/15, 1 clear) | — / — | 0.18 / 0.00 | `deriv_r6n-a069-f_EURUSD_15m_result.json` |
| r6n-a069-p | r6n-a069-p: all (60), nseed 1 | 15m | 0.03 | .5013 (.4958) | .4772 · .333 · .5429 | 0/0 · — | .5056 · .000 · .5216 | 0/0 · — | .4863 / .5018 | PILOT (three balanced paths) / PILOT (three balanced paths) | no (3/3, 1 clear; incomplete: 3 of 15 paths evaluated; never certified: PILOT (three balanced paths)) / no (3/3, 0 clear; incomplete: 3 of 15 paths evaluated; never certified: PILOT (three balanced paths)) | — / — | 0.33 / 0.00 | `deriv_r6n-a069-p_EURUSD_15m_result.json` |
| r6n-a070-p | r6n-a070-p: all (60), nseed 1 | 15m | 0.03 | .4964 (.4959) | .4399 · .000 · .4805 | 0/0 · — | .4848 · .000 · .4971 | 0/0 · — | .4863 / .5018 | PILOT (three balanced paths) / PILOT (three balanced paths) | no (3/3, 0 clear; incomplete: 3 of 15 paths evaluated; never certified: PILOT (three balanced paths)) / no (3/3, 0 clear; incomplete: 3 of 15 paths evaluated; never certified: PILOT (three balanced paths)) | — / — | 0.00 / 0.00 | `deriv_r6n-a070-p_EURUSD_15m_result.json` |
| r6n-a071-p | r6n-a071-p: all (60), nseed 1 | 15m | 0.03 | .4997 (.4951) | .4851 · .000 · .4880 | 0/0 · — | .4589 · .333 · .4985 | 0/0 · — | .4863 / .5018 | PILOT (three balanced paths) / PILOT (three balanced paths) | no (2/3, 0 clear; incomplete: 3 of 15 paths evaluated; never certified: PILOT (three balanced paths)) / no (3/3, 1 clear; incomplete: 3 of 15 paths evaluated; never certified: PILOT (three balanced paths)) | — / — | 0.00 / 0.00 | `deriv_r6n-a071-p_EURUSD_15m_result.json` |
| r6n-lgb2h-p | r6n-lgb2h-p: all (60), nseed 1 | 15m | 0.03 | .5004 (.4962) | .4439 · .000 · .4692 | 0/0 · — | .4861 · .000 · .4923 | 0/0 · — | .4863 / .5018 | PILOT (three balanced paths) / PILOT (three balanced paths) | no (3/3, 0 clear; incomplete: 3 of 15 paths evaluated; never certified: PILOT (three balanced paths)) / no (2/3, 0 clear; incomplete: 3 of 15 paths evaluated; never certified: PILOT (three balanced paths)) | — / — | 0.00 / 0.00 | `deriv_r6n-lgb2h-p_EURUSD_15m_result.json` |
| r6n-logit2h-p | r6n-logit2h-p: all (60), nseed 1 | 15m | 0.03 | .5026 (.4992) | .4916 · .000 · .5056 | 0/0 · — | .4814 · .000 · .4881 | 0/0 · — | .4863 / .5018 | PILOT (three balanced paths) / PILOT (three balanced paths) | no (3/3, 0 clear; incomplete: 3 of 15 paths evaluated; never certified: PILOT (three balanced paths)) / no (3/3, 0 clear; incomplete: 3 of 15 paths evaluated; never certified: PILOT (three balanced paths)) | — / — | 0.00 / 0.00 | `deriv_r6n-logit2h-p_EURUSD_15m_result.json` |
| r6n-mlp2h-p | r6n-mlp2h-p: all (60), nseed 1 | 15m | 0.03 | .5043 (.5022) | .4729 · .000 · .4885 | 0/0 · — | .4755 · .000 · .5000 | 0/0 · — | .4863 / .5018 | PILOT (three balanced paths) / PILOT (three balanced paths) | no (3/3, 0 clear; incomplete: 3 of 15 paths evaluated; never certified: PILOT (three balanced paths)) / no (3/3, 0 clear; incomplete: 3 of 15 paths evaluated; never certified: PILOT (three balanced paths)) | — / — | 0.00 / 0.00 | `deriv_r6n-mlp2h-p_EURUSD_15m_result.json` |
| r6n-mlp2hmod-p | r6n-mlp2hmod-p: all (60), nseed 1 | 15m | 0.03 | .5044 (.5025) | .4746 · .000 · .4856 | 0/0 · — | .4726 · .333 · .4987 | 0/0 · — | .4863 / .5018 | PILOT (three balanced paths) / PILOT (three balanced paths) | no (3/3, 0 clear; incomplete: 3 of 15 paths evaluated; never certified: PILOT (three balanced paths)) / no (3/3, 1 clear; incomplete: 3 of 15 paths evaluated; never certified: PILOT (three balanced paths)) | — / — | 0.00 / 0.00 | `deriv_r6n-mlp2hmod-p_EURUSD_15m_result.json` |
| r6n-mlp32x2-p | r6n-mlp32x2-p: all (60), nseed 1 | 15m | 0.03 | .4992 (.4916) | .4683 · .000 · .4928 | 0/0 · — | .4652 · .000 · .4797 | 0/0 · — | .4863 / .5018 | PILOT (three balanced paths) / PILOT (three balanced paths) | no (3/3, 0 clear; incomplete: 3 of 15 paths evaluated; never certified: PILOT (three balanced paths)) / no (3/3, 0 clear; incomplete: 3 of 15 paths evaluated; never certified: PILOT (three balanced paths)) | — / — | 0.00 / 0.00 | `deriv_r6n-mlp32x2-p_EURUSD_15m_result.json` |
| r6n-p102-f | r6n-p102-f: all (60), nseed 1 | 15m | 0.03 | .5076 (.5003) | .4753 · .267 · .5205 | 1/3 · .5161 | .5032 · .333 · .5292 | 0/4 · .4788 | .4863 / .5018 | KILLED / KILLED | no (15/15, 4 clear) / no (15/15, 5 clear) | — / — | 0.07 / 0.07 | `deriv_r6n-p102-f_EURUSD_15m_result.json` |
| r6n-p10266-f | r6n-p10266-f: all (60), nseed 1 | 15m | 0.03 | .5072 (.4992) | .4361 · .000 · .4740 | 1/2 · .5333 | .4839 · .067 · .5104 | 0/4 · .4954 | .4863 / .5018 | KILLED / KILLED | no (13/15, 0 clear) / no (15/15, 1 clear) | — / — | 0.00 / 0.07 | `deriv_r6n-p10266-f_EURUSD_15m_result.json` |
| r6n-tabm-p | r6n-tabm-p: all (60), nseed 1 | 15m | 0.03 | .4989 (.4963) | .4580 · .000 · .4904 | 0/0 · — | .4707 · .000 · .4993 | 0/0 · — | .4863 / .5018 | PILOT (three balanced paths) / PILOT (three balanced paths) | no (3/3, 0 clear; incomplete: 3 of 15 paths evaluated; never certified: PILOT (three balanced paths)) / no (3/3, 0 clear; incomplete: 3 of 15 paths evaluated; never certified: PILOT (three balanced paths)) | — / — | 0.00 / 0.00 | `deriv_r6n-tabm-p_EURUSD_15m_result.json` |
| r6n-tabmemb-p | r6n-tabmemb-p: all (60), nseed 1 | 15m | 0.03 | .5049 (.5010) | .5004 · .000 · .5085 | 0/0 · — | .5266 · .667 · .5505 | 0/0 · — | .4863 / .5018 | PILOT (three balanced paths) / PILOT (three balanced paths) | no (3/3, 0 clear; incomplete: 3 of 15 paths evaluated; never certified: PILOT (three balanced paths)) / no (3/3, 2 clear; incomplete: 3 of 15 paths evaluated; never certified: PILOT (three balanced paths)) | — / — | 0.00 / 0.33 | `deriv_r6n-tabmemb-p_EURUSD_15m_result.json` |
| r6v-base-e17nyam | r6v-base-e17nyam: all (66), nseed 1 | 15m | 0.03 | .5130 (.4984) | .5000 · .692 · .5723 | 0/0 · — | .4282 · .200 · .5042 | 0/2 · .4003 | .4909 / .5006 | KILLED / KILLED | no (13/15, 9 clear) / no (15/15, 3 clear) | — / — | 0.62 / 0.20 | `deriv_r6v-base-e17nyam_EURUSD_15m_result.json` |
| r6v-base-e87fixrank | r6v-base-e87fixrank: list:compact_kernels.json (14), nseed 1 | 15m | 0.03 | .5034 (.5002) | — · .000 · — | 0/0 · — | .5112 · .467 · .5473 | 2/4 · .5170 | .4863 / .5018 | KILLED / KILLED | no (0/15, 0 clear) / no (15/15, 7 clear) | — / — | — / 0.27 | `deriv_r6v-base-e87fixrank_EURUSD_15m_result.json` |
| r6v-v04-e17nyam | r6v-v04-e17nyam: all (66), nseed 1 | 15m | 0.03 | .5144 (.4969) | .4599 · .000 · .4895 | 0/2 · .4762 | .4471 · .167 · .4903 | 2/3 · .5435 | .4909 / .5006 | KILLED / KILLED | no (11/15, 0 clear) / no (6/15, 1 clear) | — / — | 0.00 / 0.17 | `deriv_r6v-v04-e17nyam_EURUSD_15m_result.json` |
| r6v-v04-e87fixrank | r6v-v04-e87fixrank: list:compact_kernels.json (14), nseed 1 | 15m | 0.03 | .5034 (.5002) | — · .000 · — | 0/0 · — | .4836 · .133 · .5190 | 0/4 · .4872 | .4863 / .5018 | KILLED / KILLED | no (0/15, 0 clear) / no (15/15, 2 clear) | — / — | — / 0.13 | `deriv_r6v-v04-e87fixrank_EURUSD_15m_result.json` |
| r6v-v05-e17nyam | r6v-v05-e17nyam: all (70), nseed 1 | 15m | 0.03 | .5114 (.4959) | .4771 · .714 · .5671 | 0/1 · .4815 | .3892 · .429 · .4981 | 0/2 · .4591 | .4909 / .5006 | KILLED / KILLED | no (14/15, 10 clear) / no (14/15, 6 clear) | — / — | 0.57 / 0.29 | `deriv_r6v-v05-e17nyam_EURUSD_15m_result.json` |
| r6v-v05-e87fixrank | r6v-v05-e87fixrank: list:compact_kernels.json (18), nseed 1 | 15m | 0.03 | .5034 (.5002) | — · .000 · — | 0/0 · — | .5153 · .600 · .5579 | 2/4 · .5309 | .4863 / .5018 | KILLED / KILLED | no (0/15, 0 clear) / no (15/15, 9 clear) | — / — | — / 0.27 | `deriv_r6v-v05-e87fixrank_EURUSD_15m_result.json` |
| r6v-v06-e17nyam | r6v-v06-e17nyam: all (66), nseed 1 | 15m | 0.03 | .5125 (.4995) | .5000 · .727 · .5623 | 0/0 · — | .3844 · .250 · .4824 | 0/2 · .4405 | .4909 / .5006 | KILLED / KILLED | no (11/15, 8 clear) / no (12/15, 3 clear) | — / — | 0.55 / 0.08 | `deriv_r6v-v06-e17nyam_EURUSD_15m_result.json` |
| r6v-v06-e87fixrank | r6v-v06-e87fixrank: list:compact_kernels.json (14), nseed 1 | 15m | 0.03 | .5033 (.5002) | — · .000 · — | 0/0 · — | .4691 · .400 · .5274 | 0/4 · .4694 | .4863 / .5018 | KILLED / KILLED | no (0/15, 0 clear) / no (15/15, 6 clear) | — / — | — / 0.33 | `deriv_r6v-v06-e87fixrank_EURUSD_15m_result.json` |
| r6v-v07-e17nyam | r6v-v07-e17nyam: all (66), nseed 1 | 15m | 0.03 | .5127 (.5035) | .5667 · 1.0 · .5667 | 0/0 · — | .5020 · .000 · .5100 | 0/0 · — | .4909 / .5006 | CERTIFIED / KILLED | no (1/15, 1 clear) / no (2/15, 0 clear) | — / — | 1.00 / 0.00 | `deriv_r6v-v07-e17nyam_EURUSD_15m_result.json` |
| r6v-v08-e17nyam | r6v-v08-e17nyam: all (66), nseed 1 | 15m | 0.03 | .5130 (.4984) | .5000 · .692 · .5723 | 0/0 · — | .4282 · .200 · .5042 | 0/2 · .4003 | .4909 / .5006 | KILLED / KILLED | no (13/15, 9 clear) / no (15/15, 3 clear) | — / — | 0.62 / 0.20 | `deriv_r6v-v08-e17nyam_EURUSD_15m_result.json` |
| r6v-v08-e87fixrank | r6v-v08-e87fixrank: list:compact_kernels.json (14), nseed 1 | 15m | 0.03 | .5037 (.5004) | — · .000 · — | 0/0 · — | .5162 · .533 · .5502 | 2/4 · .5220 | .4881 / .5002 | KILLED / KILLED | no (0/15, 0 clear) / no (15/15, 8 clear) | — / — | — / 0.27 | `deriv_r6v-v08-e87fixrank_EURUSD_15m_result.json` |
| r6v-v09-e17nyam | r6v-v09-e17nyam: all (66), nseed 1 | 15m | 0.03 | .5130 (.4984) | — · .000 · — | 0/0 · — | .3323 · .000 · .4158 | 0/0 · — | .4909 / .5006 | KILLED / KILLED | no (0/15, 0 clear) / no (3/15, 0 clear) | — / — | — / 0.00 | `deriv_r6v-v09-e17nyam_EURUSD_15m_result.json` |
| r6v-v09-e87fixrank | r6v-v09-e87fixrank: list:compact_kernels.json (14), nseed 1 | 15m | 0.03 | .5034 (.5002) | — · .000 · — | 0/0 · — | .4382 · .333 · .5146 | 0/2 · .4371 | .4863 / .5018 | KILLED / KILLED | no (0/15, 0 clear) / no (15/15, 5 clear) | — / — | — / 0.13 | `deriv_r6v-v09-e87fixrank_EURUSD_15m_result.json` |
| r6vx-base-e20sg | r6vx-base-e20sg: all+fam:sg (71), nseed 1 | 15m | 0.03 | .5098 (.4985) | .3888 · .071 · .4656 | 0/1 · .5000 | .4325 · .533 · .5235 | 2/4 · .5330 | .4863 / .5018 | KILLED / KILLED | no (14/15, 1 clear) / no (15/15, 8 clear) | — / — | 0.07 / 0.33 | `deriv_r6vx-base-e20sg_EURUSD_15m_result.json` |
| r6vx-base-e30nystroem | r6vx-base-e30nystroem: all (66), nseed 1 | 15m | 0.03 | .5035 (.4934) | .3846 · .000 · .4600 | 1/3 · .5111 | .4547 · .067 · .4970 | 2/4 · .5306 | .4863 / .5018 | KILLED / KILLED | no (12/15, 0 clear) / no (15/15, 1 clear) | — / — | 0.00 / 0.00 | `deriv_r6vx-base-e30nystroem_EURUSD_15m_result.json` |
| r6vx-v04-e20sg | r6vx-v04-e20sg: all+fam:sg (71), nseed 1 | 15m | 0.03 | .5107 (.4995) | .5098 · .000 · .5098 | 1/2 · .5441 | .4553 · .000 · .4787 | 1/3 · .4972 | .4863 / .5018 | KILLED / KILLED | no (1/15, 0 clear) / no (15/15, 0 clear) | — / — | 0.00 / 0.00 | `deriv_r6vx-v04-e20sg_EURUSD_15m_result.json` |
| r6vx-v04-e30nystroem | r6vx-v04-e30nystroem: all (66), nseed 1 | 15m | 0.03 | .5033 (.4948) | .4779 · .167 · .5046 | 1/2 · .5271 | .4488 · .000 · .4803 | 1/3 · .5048 | .4863 / .5018 | KILLED / KILLED | no (6/15, 1 clear) / no (15/15, 0 clear) | — / — | 0.00 / 0.00 | `deriv_r6vx-v04-e30nystroem_EURUSD_15m_result.json` |
| r6vx-v05-e20sg | r6vx-v05-e20sg: all+fam:sg (75), nseed 1 | 15m | 0.03 | .5091 (.4972) | .4110 · .308 · .5081 | 0/0 · — | .4513 · .200 · .5094 | 1/4 · .5248 | .4863 / .5018 | KILLED / KILLED | no (13/15, 4 clear) / no (15/15, 3 clear) | — / — | 0.23 / 0.20 | `deriv_r6vx-v05-e20sg_EURUSD_15m_result.json` |
| r6vx-v05-e30nystroem | r6vx-v05-e30nystroem: all (70), nseed 1 | 15m | 0.03 | .5020 (.4888) | .4231 · .000 · .4615 | 0/2 · .4667 | .4831 · .067 · .5136 | 2/4 · .5275 | .4863 / .5018 | KILLED / KILLED | no (11/15, 0 clear) / no (15/15, 1 clear) | — / — | 0.00 / 0.00 | `deriv_r6vx-v05-e30nystroem_EURUSD_15m_result.json` |
| r6vx-v06-e20sg | r6vx-v06-e20sg: all+fam:sg (71), nseed 1 | 15m | 0.03 | .5081 (.4973) | .3680 · .167 · .4554 | 0/0 · — | .4612 · .400 · .5337 | 2/3 · .5738 | .4863 / .5018 | KILLED / KILLED | no (12/15, 2 clear) / no (15/15, 6 clear) | — / — | 0.00 / 0.40 | `deriv_r6vx-v06-e20sg_EURUSD_15m_result.json` |
| r6vx-v06-e30nystroem | r6vx-v06-e30nystroem: all (66), nseed 1 | 15m | 0.03 | .4962 (.4811) | .4082 · .071 · .4637 | 1/4 · .5036 | .4429 · .143 · .4942 | 1/3 · .5157 | .4863 / .5018 | KILLED / KILLED | no (14/15, 1 clear) / no (14/15, 2 clear) | — / — | 0.00 / 0.07 | `deriv_r6vx-v06-e30nystroem_EURUSD_15m_result.json` |
| r6vx-v07-e20sg | r6vx-v07-e20sg: all+fam:sg (71), nseed 1 | 15m | 0.03 | .5045 (.4988) | .3903 · .000 · .4129 | 0/0 · — | .4220 · .000 · .4554 | 0/0 · — | .4863 / .5018 | KILLED / KILLED | no (2/15, 0 clear) / no (3/15, 0 clear) | — / — | 0.00 / 0.00 | `deriv_r6vx-v07-e20sg_EURUSD_15m_result.json` |
| r6vx-v07-e30nystroem | r6vx-v07-e30nystroem: all (66), nseed 1 | 15m | 0.03 | .4984 (.4940) | .4434 · .267 · .4981 | 1/4 · .5023 | .4266 · .067 · .4748 | 0/4 · .4876 | .4863 / .5018 | KILLED / KILLED | no (15/15, 4 clear) / no (15/15, 1 clear) | — / — | 0.07 / 0.00 | `deriv_r6vx-v07-e30nystroem_EURUSD_15m_result.json` |
| r6vx-v08-e20sg | r6vx-v08-e20sg: all+fam:sg (71), nseed 1 | 15m | 0.03 | .5085 (.4979) | .4012 · .267 · .4980 | 0/0 · — | .4273 · .133 · .4938 | 1/4 · .5266 | .4881 / .5002 | KILLED / KILLED | no (15/15, 4 clear) / no (15/15, 2 clear) | — / — | 0.20 / 0.13 | `deriv_r6vx-v08-e20sg_EURUSD_15m_result.json` |
| r6vx-v08-e30nystroem | r6vx-v08-e30nystroem: all (66), nseed 1 | 15m | 0.03 | .5016 (.4959) | .3938 · .154 · .4696 | 1/3 · .5358 | .4704 · .133 · .5001 | 0/4 · .5035 | .4881 / .5002 | KILLED / KILLED | no (13/15, 2 clear) / no (15/15, 2 clear) | — / — | 0.08 / 0.00 | `deriv_r6vx-v08-e30nystroem_EURUSD_15m_result.json` |
| r6vx-v09-e20sg | r6vx-v09-e20sg: all+fam:sg (71), nseed 1 | 15m | 0.03 | .5098 (.4985) | — · .000 · — | 0/0 · — | .3846 · .000 · .3846 | 0/0 · — | .4863 / .5018 | KILLED / KILLED | no (0/15, 0 clear) / no (1/15, 0 clear) | — / — | — / 0.00 | `deriv_r6vx-v09-e20sg_EURUSD_15m_result.json` |
| r6vx-v09-e30nystroem | r6vx-v09-e30nystroem: all (66), nseed 1 | 15m | 0.03 | .5035 (.4934) | .4391 · .000 · .4916 | 0/0 · — | .4276 · .111 · .4897 | 0/2 · .5350 | .4863 / .5018 | KILLED / KILLED | no (4/15, 0 clear) / no (9/15, 1 clear) | — / — | 0.00 / 0.11 | `deriv_r6vx-v09-e30nystroem_EURUSD_15m_result.json` |
<!-- DERIV-FEED-TABLES:END -->
