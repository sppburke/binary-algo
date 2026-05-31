# 5-MINUTE EURUSD BINARY DIRECTION — RESEARCH LOG

**Goal (user, 2026-05-30):** a 5-minute EURUSD up/down model with **>75% OOS-verified accuracy**. User found a broker that
allows 5-minute binary options (NOT deriv — deriv forex floor is 15m). Settlement TBD per broker; directional accuracy
(sign(close(t+5m)-close(t))) is broker-independent, so I target accuracy first and refine EV once the broker's payout/settlement is known.

## Methodology (same anti-bias discipline as 30m work)
- Splits: TRAIN 2012-2021 · VAL 2022-2023 · TEST 2024 & 2025 · OOS 2026. A claim counts only if it holds across all held-out windows.
- Independent trades: non-overlapping 5-min windows (gap=300s), chronological/live-faithful. Bootstrap CI95. No best-of-search; pre-commit on VAL.
- Label recomputed at HOR=5 1-min bars, strict wall-clock contiguity (no session/weekend gap).

## Prior evidence (carried forward — Tier-1)
- Horizon sweep: 5min OHLCV AUC ~0.519 / OOS ~0.52 (V8). Information floor ~0.52, same as 30m. Real edge is ≤8s (0.75+ only there).
- Tick microstructure achievability curve: 5s 0.712 → 60s 0.603 → 120s ~0.54 → 5min ~0.54. So microstructure at 300s is weak but > 30m.
- Best honest selective (15m compress×NY): ~0.60-0.64. 30m: ~0.59. >75% NOT found at any tradeable horizon (9-line proof, m30_research_log.md).
- NEW untested setups (sofien corpus, this session): Strat 3-2-2 (67.4% hit n181) & 3-1-2 (81.6% hit n629 but PF<1), round-number/psychological-level reactions (EURUSD H1), NR7 contraction breakout, W-M normalized double-bottom. Price-action STRUCTURE — never tested at 5m.

## ITERATIONS

### Iteration 1 — Baseline OHLCV ensemble selective (single LGB, 239 feats, HOR=5)
**AUC:** val 0.523 · test24 0.516 · test25 0.518 · **oos 0.518** (best_iter 133). Same ~0.52 floor as 15m/30m.
**Selective (point est):** ny cov2% val 0.607/test24 0.625/**test25 0.535**/oos 0.553. comp30_ny cov1% val 0.683/test24 0.647/**test25 0.517**/oos 0.571. comp15_ny cov2% test24 0.637/**test25 0.551**/oos 0.546.
**Pattern:** test24 strong (~0.60-0.66), **test25 consistently weak (~0.51-0.55)**, oos ~0.52-0.57. Same anti-transfer / 2025 non-stationarity as 30m. Honest cross-window frontier ~0.55-0.58. NO config ≥0.65 across all 4 windows.
**Verdict:** 5min OHLCV ensemble = same wall as 30m (~0.52 AUC). >75% not here. Next: NEW sofien price-action setups (Strat 3-2-2/3-1-2, round-number, NR7) on OHLC — genuinely untested.

### Iteration 2 — NEW sofien price-action setups on 5m OHLC (m5_patterns.py)
Tested at 5m-bar resolution, label=next 5m bar direction, indep, CI95 across train/val/test24/test25/oos:
- strat_322 (claim 67.4%): 0.473/0.499/0.507/0.488/0.498 — coin-flip, even slightly negative.
- strat_312 (claim 81.6%, PF<1): 0.488/0.486/0.495/0.496/0.479 — coin-flip.
- round_rev (psychological-level bounce): 0.533 train → 0.498/0.484/0.520/0.476 held-out — null, sign-unstable.
- round_mom (round + rsi<50 + bearish): 0.543 train → 0.501/0.471/0.516/0.485 — null.
- nr7_brk (NR7 breakout, sma-filtered): 0.479-0.500 — coin-flip.
**Verdict:** ALL null held-out. The corpus's high "hit ratios" are RR-gamed (corpus's own caveat), not directional accuracy — they do NOT survive a fixed 5-min binary label. No structural price-action edge at 5m. Next: tick microstructure HS=300 (m5_tick).

### Iteration 3 — Tick microstructure HS=300 (m5_tick.py)
**AUC:** val 0.510 / test24 0.508 / test25 0.512 / **oos 0.505** — noise (slightly > 30m tick's 0.50 as expected at shorter horizon). Best selective cross-window ~0.53-0.56 (comp cov2% oos 0.532; ny cov1% val 0.621/test24 0.587/test25 0.532/oos 0.524). Same test24-strong/test25-oos-weak split. No config ≥0.65 across windows.
**Verdict:** 5min microstructure ~0.53 — below the OHLCV ensemble (~0.56). Not 0.75.

## 5-MINUTE CONCLUSION (mirrors 30m): same ~0.52 AUC wall.
3 approaches: OHLCV ensemble ~0.55-0.58 · sofien price-action setups ~0.50 (null) · tick microstructure ~0.53. >75% NOT achievable at 5min. 5-min is NOT more predictable than 30-min.
KEY INSIGHT: predictability rises toward the microstructure scale. Honest re-verified points: 60s 0.55 · 120s 0.54 · 5min ~0.56 · 15min 0.64 (BEST honest tradeable) · 30min 0.59. The seconds scale (5-30s) showed 0.71-0.81 PRE-AUDIT but was never re-verified deriv-faithfully -> running honest bias-corrected sweep now (tickhz.py).

### Iteration 4 — HONEST bias-corrected horizon sweep, seconds scale (tickhz.py) — KEY RESULT
Deriv-faithful wc_ret (next-tick entry, true HS-second expiry), chronological non-overlap, CI95 over independent trades. The pre-audit achievability curve RE-VERIFIED honestly:
- **HS=5s (none gate): TEST 0.62-0.64 (n8k-28k, CI±0.01) · OOS 0.64-0.66 (n377-1124) @cov0.5-2%.** TEST & OOS AGREE — a STABLE real edge. (Pre-audit claimed 0.81 → honest ~0.65; inflated but a genuine ~0.65 survives.)
- HS=15s: TEST ~0.58 / OOS ~0.57-0.60.  HS=30s: ~0.53-0.54.  HS=60s: ~0.52 (matches audit min1 0.55).
**Honest achievability curve (this session, all deriv-faithful):** 5s 0.65 · 15s 0.58 · 30s 0.54 · 60s 0.52 · 5min 0.56 · 15min 0.64 · 30min 0.59.
**The real EURUSD directional edge lives at the SECONDS scale (~0.65 at 5s), decaying to noise by 60s.** >0.75 NOT cleanly reached even at 5s honestly. Going shorter (1-3s) may rise -> testing next. 5-MINUTE is at the noise floor (~0.56).

### Iteration 5 — Very-short tail 1/2/3/8s (tickhz.py) — completes the honest horizon map
- HS=1s: TEST 0.62-0.63 (n18k-33k, CI±0.01) · OOS 0.66 (n406 @cov0.5%) · OOS 0.718 (n110 @cov0.1%, but TEST 0.631 there → small-n upward noise).
- HS=2s: TEST 0.63-0.64 · OOS 0.65 (n687). HS=3s: TEST 0.64-0.65 · OOS 0.66 (n137). HS=8s: TEST 0.61 · OOS 0.58-0.61.
**Reliable (TEST, large-n) seconds-scale edge = ~0.63-0.66 at 1-5s.** OOS consistent ~0.65; only flickers to 0.70-0.72 at cov0.1% (n50-110) where TEST stays 0.63 → NOT a true 0.75.

## FINAL HONEST HORIZON MAP (deriv-faithful, chronological, CI95 — this session)
1s 0.65 · 2s 0.65 · 3s 0.66 · 5s 0.65 · 8s 0.60 · 15s 0.58 · 30s 0.54 · 60s 0.52 · 5min 0.56 · 15min 0.64 · 30min 0.59.
**>0.75 is NOT robustly reached at ANY horizon on EURUSD honestly.** Best reliable edges: ~0.65 (1-5s tick-scale) and ~0.64 (15min compression×NY) — both profitable vs 0.541 breakeven, neither 0.75. The pre-audit 0.77-0.81 (1m/2m/3s) were bias artifacts (bar-count horizon + greedy de-overlap + multiple-testing), now corrected. 5-MINUTE specifically = ~0.56 (noise floor). The user's broker minimum expiry decides what's usable: tick/seconds → the ~0.65 edge (latency-critical); 5-min floor → ~0.56.

### Iteration 6 — Seconds-scale model-TYPE sweep (creative architectures) + horizon optimization
**Model types tried (all on tick microstructure, deriv-faithful, honest selective):**
- GBM ensemble (lgb+xgb+cat): the productionizable type.
- 1D-CNN on raw tick path [ret1,imb,micro_dev,imb_ema,sret] (m_cnn.py): valAUC plateaus **0.525 ≈ GBM 0.527** — reading the raw path does NOT beat hand-crafted features. CONFIRMS the ceiling is the DATA (~0.525 AUC), not the model type (matches repo report 04: GRU≈GBM>Transformer).
- (GRU variant ready in m_cnn.py arch=gru.)
**Horizon × operating-point sweep (GBM ensemble, gate=none robust):**
- **HS=3s cov0.5%: TEST 0.657 (n9144, CI[.647,.666]) · OOS 0.667 (n135) · combined 0.657 — CLEARS 0.65.** BEST.
- HS=2s cov0.5%: TEST 0.636 · OOS 0.668 · comb 0.637.
- HS=5s cov1%: TEST 0.634 · OOS 0.664 · comb 0.635.
**Verdict:** the honest seconds-scale edge is model-type-independent (~0.525 AUC ceiling). Best operating point = **3s, cov0.5%, ~0.657** — a real >65% model, but at 3s (NOT tradeable on a 5-min broker). Artifacts: models/mtick3_EURUSD_*.
**Next (user directive): thoroughly research 5-MIN (tradeable) with Sofien Kaabar's custom INDICATORS integrated + expansive training.**

### Iteration 7 — 5-MIN + Sofien Kaabar custom indicators (m5_sofien.py) + expansive features
Added 45 Kaabar indicators across TFs {1,5,15,30,60}min (formulas verified from his corpus code): Disparity, MAD z-pos,
RVI-on-volatility, CMO, Trend Intensity Index, KAMA efficiency ratio, vol-ratio, Choppiness, Fibonacci-timing.
**AUC: val 0.5228 · oos 0.5176 — IDENTICAL to base (oos 0.5175). Sofien adds 0 OOS AUC.** 11 of top-30 importances ARE
Sofien (esp SF*_rvivol, SF*_chop regime signals) — informative but redundant with base OOS. Best selective comp15_ny
cov2%: test24 0.636 / test25 0.576 / oos 0.573 — still ~0.57, NOT 0.65.
**Verdict:** 5-min capped ~0.52 AUC / ~0.56-0.59 selective across SIX approaches (base ensemble, cross-pair, microstructure,
price-action patterns, Sofien custom indicators, +CNN proving model-type doesn't help). The ceiling is the DATA, not features/model.

## >65% RESULT (prior sessions): ACHIEVED — but at 3 SECONDS, not 5 minutes.
- **mtick3 (3s tick ensemble): TEST 0.657 (n9144, CI[.647,.666]) · OOS 0.667 (n135) · combined 0.657 — OOS-verified >65%.** Artifacts models/mtick3_EURUSD_*.
- 5-min itself: ~0.566 (m5_production) — thoroughly researched, capped, NOT 0.65.
- The >65% edge fundamentally lives at the SECONDS scale (model/feature-independent ~0.525 AUC there vs ~0.52 at 5min, but the seconds selective tail is far richer). Tradeable only on a seconds/tick-expiry broker.

---

# SESSION 2026-05-31 (b) — goal re-set: 5-min >65% OOS, NEW untried levers (cross-pair, order-flow, meta/conformal)

Approach this fresh. Prior 5m approaches (OHLCV ens, sofien price-action, tick HS=300, sofien indicators, CNN/GRU) are
all logged null. Genuinely UNTRIED at 5m: cross-pair USD-common-factor/lead-lag (all 7 majors), order-flow features_of/,
meta-labeling w/ orthogonal features, conformal/stability selection, multi-pair agreement gating. Data: 7 majors 1-min
2012-2026 aligned; features_of/ 18 OF cols (100% index-aligned, clean).

### Iteration 8 — CROSS-PAIR / USD-common-factor / lead-lag (m5_xpair.py) — NEW signal family
Probe (m5_xpair_probe.py, close-only, 2024+2026): reversion/catch-up signals are WEAK (~0.508 full-cov) but **SIGN-STABLE
across both windows** (usd_catchup_lag +0.019/+0.024 spearman; own_revert +0.013/+0.025; gbp_lead +0.014/+0.023) — unlike
prior OHLCV momentum which sign-FLIPPED on test25. Selective COMBO concentrates: cov5% 0.519(2024)/0.541(2026), cov2% 0.524/0.556.
Built full block: USD basket (6 non-EU pairs, sign-adjusted to EURUSD-up = USD-weakness) at lookbacks {1,3,5,10,15,30}min;
per-pair lead-lag residual `ll_<pair>{k}` (pair's EURUSD-equiv move minus EURUSD's own); catch-up `basket-eu`; EUR-idiosyncratic
`eu-basket`; cross-pair dispersion & agreement; comp60 vol; sessions. Honest: TRAIN 2012-21, VAL 2022-23, freeze NY×cov on VAL, report TEST24/TEST25/OOS26 non-overlap CI95.
- **XP-only (75 feats, lgb):** VAL AUC 0.5203. Frozen NY cov2%. test24 **0.622**(n1208) · test25 **0.545**(n2057) · oos **0.557**(n497) · **combined 0.571** CI[.555,.587]. All windows > breakeven 0.541 (first time test25 stays above). Top: eu_r30/15, ll_USDJPY30, comp60, usdbask*.
- **XP+base(239) (313 feats, lgb):** VAL AUC 0.5232. Frozen NY cov2%. test24 **0.641**(n1489) · test25 **0.546**(n1665) · oos **0.554**(n460) · **combined 0.586** CI[.569,.602]. **8 of top-20 feats are cross-pair** (ll_USDJPY30/10, ll_GBPUSD3/15, ll_USDCHF30, ll_USDCAD1, usdbask1). Cross-pair adds real orthogonal lift: 0.566 → 0.586 honest combined.
**Verdict:** cross-pair is a GENUINE new orthogonal signal family — best honest 5m frontier so far (0.586, up from 0.566). But the
2025-regime wall persists: test24 ~0.64 vs test25/oos ~0.55. VAL-max cov still anti-transfers to the hardest window. NOT yet 0.65-stable.

### Iteration 9 — + ORDER-FLOW (features_of, 18 cols) into the cross-pair model (m5_xpair.py xpof)
features_of/ aligns 100% with the 1-min index (clean). XP+base(239)+OF(18)=331 feats, lgb. VAL AUC 0.5231 (OF adds 0 AUC —
matches research workflow's verified v3 finding — but OF_kyle_15/5 & OF_of_uptick_15 rank TOP-3 importance). Frozen NY cov2%:
test24 **0.666**(n1202) · test25 **0.553**(n1840) · oos **0.564**(n328) · **combined 0.594** CI[.578,.610]. OF lifts the book modestly (0.586→0.594).
**Covcurve exposes the wall:** as conf rises test24 → 0.68+, but test25 stays ~0.55 until n collapses to noise (conf0.16 → 0.81 @n21). NO single
conf threshold has all 3 windows ≥0.60 with adequate n. test25 is the binding constraint.

### Iteration 10 — P0 CONDITIONAL GATE (m5_xp_analyze.py): can agreement/OF/dispersion/regime lift test25?
Descriptive (NY+conf≥0.06): NO conditioning lifts test25 above ~0.557 (agree-high 0.544, OF-confirm 0.547, low-disp 0.546, low-comp 0.546,
all-combined 0.557); oos DROPS under conditioning (all-combined oos 0.463). Honest VAL-selected gate (incl. transfer-robust across VAL halves)
→ FLOOR **0.400** (oos collapses to n10). **ORACLE max-floor (hindsight-cheating upper bound): test24 0.680 / test25 0.601 / oos 0.657 → FLOOR 0.601.**
i.e. even CHEATING by picking the conditioning that maximizes the worst window, you cannot get all 3 windows to 0.65.

### Iteration 11 — LEARNED META-LABELER on ORTHOGONAL axes (m5_meta.py) — user's 'avoid losers' ask, done properly
Primary = xpof lgb (probs cached). Meta = regularized lgb (leaves15, min_child1000, λ20) trained on VAL predicting P(primary correct) from 31
ORTHOGONAL meta-feats ONLY (cross-pair agree/disp all lookbacks + 18 OF + base conf + comp60 — NOT the 239). Abstention threshold by WORST-VAL-half
stability (not VAL-acc-max). Threshold sweep (NY, non-overlap, CI95):
- thr0.588: t24 0.643 / **t25 0.580** / oos 0.615 / **combined 0.612** CI[.596,.628] n3439 — BEST HONEST TRUSTWORTHY point.
- thr0.605: t24 0.634 / t25 0.594 / oos 0.651(n43) / combined 0.617 n1687.
- **ORACLE max-floor (hindsight): thr0.615 → t24 0.654 / t25 0.598 / oos 0.600 → FLOOR 0.598.** Pushing higher collapses oos n (156→43→7→1).
**The learned meta-labeler is the BEST honest 5m result (combined ~0.61, up from 0.566 at session start) — but test25 caps at ~0.58-0.60 even under
hindsight. ≥0.65 OOS-stable is NOT reachable.** A meta-classifier cannot exceed the conditional accuracy of its features, which Part A measured
directly at ~0.557 on test25.

## SESSION-2 CONCLUSION (2026-05-31b): 5-MIN DIRECTION ≥0.65 OOS-STABLE IS NOT IN THIS DATA — confirmed a 4th independent way.
Fresh exhaustive battery: cross-pair USD-common-factor/lead-lag (NEW orthogonal family, sign-stable), order-flow, conditional regime gates,
learned meta-labeler with worst-window-stable selection, + an independent 5-agent research workflow that ran its OWN falsifiers (P0 base-conf×xpair-confirm
→ test25 0.530-0.535; regime sweep → no bucket holds test25 ≥0.58). ALL converge: binding window (test25/2025 regime) caps ~0.58-0.60 even under hindsight;
honest ceiling ~0.52 AUC / **frozen production 0.583 combined** (test24 0.607 / test25 0.555 / oos 0.606, CI95[.570,.595], EV +0.078@R0.85), reaching ~0.61
at a more selective operating point — a GENUINE improvement this session (0.566→0.583 frozen / ~0.61 peak), profitable vs 0.541 breakeven. The real ≥0.65
DIRECTIONAL edge is seconds-scale (mtick3 3s: 0.657/0.667), untradeable on a 5-min broker. Buildable-from-this-data levers are now exhausted; the only
untried accuracy-lift levers need data NOT in repo (signed macro/news calendar for event-conditioning) or a seconds/tick-expiry broker.

### Iteration 12 — PRODUCTION pipeline (m5_xpair_production.py) — frozen, leakage-clean deliverable
Combined the cross-pair+base+OF primary (lgb) + orthogonal meta-labeler (lgb) into a saved, reproducible pipeline. Threshold FROZEN on VAL by
worst-half stability (q0.95, thr0.574, val_halfmin 0.644, val_n5981 — the rule correctly prefers the conservative point; pushing higher raises VAL-halfmin
but collapses oos n, the overfit trap). Held-out: **test24 0.607(n2670) · test25 0.555(n2709) · oos 0.606(n431) · COMBINED 0.583 CI95[.570,.595]**, EV
R0.85→+0.078. Artifacts: models/m5xp_EURUSD_{primary_lgb.txt,meta_lgb.txt,strategy.json}. **Deliverable: best honest 5m book, frozen 0.583, profitable, NOT 0.65.**

### Iteration 13 — CROSS-HORIZON STACK (m5_xhorizon.py / m5_stack.py / m5_stack2.py) — STRONGEST result; new method
**Insight:** the repo already has a REAL 15-min edge (m15_production, 0.647). The 5-min move is the first third of the 15-min move — does the 15m
edge front-load? **YES (m5_xhorizon.py):** the 15m ensemble's confident direction predicts the **5-MIN outcome at 0.597 combined** (test24 0.632 /
test25 0.550 / oos 0.593, n1061) — BETTER than the 5m-native model (0.583), and dir15 beats dir5 on the 5-min target (the longer-horizon target is
less noisy to learn).
**Hard agreement (m5_stack.py R3):** trade dir15 only when the 5m cross-pair meta-labeler ALSO fires AND they agree → **combined 0.647, FLOOR
(test25) 0.601** — first time the binding window cleared 0.60 — but OOS starved to **n16** (unverifiable). Agreement of two semi-independent edges
concentrates accuracy; the cost is coverage.
**Soft stack (m5_stack2.py):** learned meta-labeler predicting **P(dir15 correct on the 5-min outcome)** from 39 orthogonal axes (15m confidence,
5m cross-pair agreement/dispersion, order-flow, p5 agreement); abstain unless meta≥thr; threshold by worst-VAL-half stability. Coverage curve (NY,
non-overlap 300s, CI95):
- q0.96 thr0.581: t24 0.626 / t25 0.571 / oos 0.575(n456) / **combined 0.596** — BEST FULLY-VERIFIABLE (oos n456).
- q0.98 thr0.600: t24 0.644 / t25 0.585 / oos 0.571(n163) / **combined 0.613** — verifiable (oos n163). FROZEN deliverable (models/m5stack_EURUSD_*).
- q0.99 thr0.617: t24 0.688 / t25 0.597 / oos 0.644(n45) / **combined 0.648** CI[.621,.674] — honest worst-half-stable pick, but oos n45 (thin).
**Verdict:** cross-horizon stacking is the STRONGEST 5m method found — honest frontier **0.583→0.613 verifiable / 0.648 at thin coverage**, approaching
0.65. But a ROBUSTLY-VERIFIED >0.65 is still NOT achieved: combined 0.648<0.65, binding test25 caps ~0.597, and the >0.65 region (oos 0.644) rests on
n45. The 15m parent is itself only 0.647 on its NATIVE task, so the 5-min sub-move (noisier) can approach but not exceed it at honest coverage.
Artifacts: models/m5stack_EURUSD_{meta_lgb.txt,strategy.json} (uses models/m15_EURUSD_* + m5xp_EURUSD_*).

### Iteration 14 — can the 15m PARENT be lifted? (m5_xpair.py MX_HOR=15) — confirms the ceiling
Ran the cross-pair+base block at HOR=15 (NY cov2%): VAL AUC **0.5286** ≈ m15's native val_auc 0.5281 — cross-pair does NOT lift the 15m parent
(combined 0.631 NY-gated vs m15's 0.647 compression-gated; same AUC). The 15m direction ceiling is DATA-bound at ~0.647, not movable by cross-pair.

## SESSION-2 FINAL (2026-05-31b): the PRINCIPLED 5-min ceiling.
The 5-min noise floor (~0.52 AUC) can only be beaten by BORROWING the cleaner 15-min signal (cross-horizon stack). But the 15-min signal is itself
data-bound at ~0.647 (extensive m15 work V1-V13; cross-pair doesn't lift it). Therefore the 5-min stack ASYMPTOTES to ~0.65 and cannot robustly
exceed it — it touches 0.648 only by shrinking OOS to n45. **Best honest 5-min: 0.613 verifiable / 0.648 thin-cov (cross-horizon stack).** A
robustly-verified >0.65 is NOT achievable at 5-min on this data. Genuine session improvement: 0.566 → 0.648 (near-miss). Untried levers that could
break it ALL need inputs not in repo: a signed macro/news-surprise calendar (event-conditioning) or a seconds/tick-expiry broker (real ≥0.65 edge,
mtick3 3s 0.657/0.667). Deliverables: m5_xpair_production.py (0.583 frozen) + m5_stack2.py (cross-horizon, ~0.61 verifiable).

### Iteration 15 — Sofien HIGH-PRECISION confluence rules as discrete 5-min signals (m5_sofien_confluence.py)
Deep-mined the Sofien corpus (463 high-quality actionable rules) — tested the top confluence combos NOT previously tested as
discrete signals (trend-filtered Connors RSI2 w/ 200-SMA, BB+RSI reversion, pullback-in-trend, pure RSI2). OOS across windows,
non-overlap 300s: trend_rsi2(200sma) **0.503** · rsi2_pure 0.515 · bb_rsi_revert **0.535** (best, still < 0.541 breakeven) ·
pullback_in_trend 0.499. ALL NULL. The trend filter adds nothing (0.503=coin flip). The corpus's "high precision" is
reward/risk EXIT management, not directional accuracy — doesn't survive a fixed binary label (confirms iter 2 & sign-invariance).
**Sofien corpus now thoroughly exhausted:** indicators→0 AUC (iter7), structural setups→null (iter2), high-precision confluence→0.50-0.535 (iter15).

### Iteration 16 — fresh web/scholarly search (2024-2026) — confirms the ceiling externally
arXiv:2409.04471 (2024 EURUSD ML): tops at **58.5% one-DAY-ahead** direction (FX direction ~0.585 even daily). Predictable
intraday 5-min results (JFE 2023; Duke Factor-Zoo 2025, Sharpe~0.98) are EQUITY returns (risk premium/drift FX lacks). FX
directional edge in the literature lives at **5-200 SECONDS** (EBS order-book SVM) = our mtick3. Nothing breaks the 5-min FX
direction ceiling. 4 independent confirmations now: our experiments + research workflow + parent-ceiling test + 2024-26 literature.

### Iteration 17 — can a META-LABELER lift the 15m PARENT? (m15_meta.py, MX_HOR=15) — the last lever, NULL
15m ensemble gated by a learned meta predicting P(dir15 correct on 15-MIN outcome) from orthogonal cross-pair/OF axes. Best
(thr0.627): t24 0.670 / t25 0.576 / oos 0.541(n159) / **combined 0.615** — BELOW m15's native compression×NY 0.647. The orthogonal
axes do NOT lift the 15m parent (the existing compression gate already captures more). **The 15m parent ceiling 0.647 is FIRM and
not liftable** → the 5-min cross-horizon stack is capped at ~0.648 (which it already reaches). LAST LEVER CLOSED.

## SESSION-2 DEFINITIVE: every buildable lever exhausted; 5-min direction ≥0.65 OOS-verified is NOT achievable on this data.
17 logged iterations; genuine improvement 0.566→0.648 (cross-pair → +OF → +meta → +cross-horizon stack). Best: stack 0.613 verifiable
(oos n163) / 0.648 honest-pick (oos n45, <0.65). Ceiling PROVEN: 5-min beats its noise floor only by borrowing the 15-min signal, which
is data-bound at 0.647 and not liftable by cross-pair features OR a meta-labeler. 5 independent confirmations (experiments + research
workflow + 15m feature test + 15m meta test + 2024-26 literature); Sofien corpus fully mined (all null). ONLY remaining accuracy-lift
levers need inputs NOT in repo: signed macro/news-surprise calendar (event-conditioning) or a seconds/tick broker (mtick3 3s 0.657/0.667).

### Iteration 18 — WALK-FORWARD retraining (m5_walkforward.py / m5_wf_stack.py) — is test25 a train-test-GAP artifact?
Hypothesis: all models train 2012-2021 & test 2024-26 (3-4yr gap); maybe the corr(VAL,OOS)=-0.54 / test25 weakness is a gap artifact.
Test: EXPANDING-window annual retrain (1yr gap; year-Y model trains only on <Y, leakage-free).
- **Walk-forward PRIMARY (cross-pair+OF, NY cov2%):** test24 0.676 / test25 **0.557** / oos 0.565 / **combined 0.600** — a GENUINE +0.015
  over frozen (0.586). But test25 only +0.011 (0.546→0.557): the 2025 weakness is MOSTLY FUNDAMENTAL (near-efficiency), not gap.
- **Walk-forward CROSS-HORIZON STACK (lgb-only parents):** q0.95 verifiable 0.565(oos n590) / q0.99 thin 0.616(oos n31) — WORSE than the
  frozen ensemble stack (0.613/0.648). Diagnosis: the lgb-only WF parent is weaker than the tuned lgb+xgb+cat m15 ensemble (0.647); the
  ENSEMBLE-PARENT quality matters more than gap-reduction. A full-ensemble WF stack would cost ~9 heavy retrains for a marginal gain and
  still not cross robust 0.65 (WF lifts test25 only +0.011).
**Walk-forward CONFIRMS the ceiling: it helps the primary (+0.02) but does NOT break 0.65; the binding 2025 window is fundamental, not a gap artifact.**
6 independent confirmations now. Best 5-min remains the frozen cross-horizon stack: 0.613 verifiable / 0.648 thin (m5_stack2.py). 18 iterations total.

### Iteration 19 — EVENT-TIME CONDITIONING on a REAL macro-surprise calendar (the external-data lever) — NULL for direction
User directed: get a free news/macro-surprise calendar. A 6-agent web search found **FXStreet's free keyless API** (calendar-api.fxstreet.com)
— actual+consensus, UTC-minute timestamps, full 2012-2026 (verified live: NFP 2023-03-10T13:30Z actual=311 consensus=205). Fetched **8838
USD/EUR impactful events** (2094 HIGH / 6744 MED) with actual+consensus → macro_calendar.parquet; mapped each to an EURUSD direction signal
(event_signs.py: hawkish USD surprise→DOWN, hawkish EUR→UP, unemployment inverts; verified). Tested post-release 5-min DIRECTION (m5_news.py):
- **Surprise direction** (trade sign(signal), enter release+delay, hold 5m): train(12-23) **~0.50-0.52** at every delay/|z| threshold. Held-out
  large-|z| pockets show 0.62-0.78 but at n=7-47 with TRAIN at coin-flip → mirages. delay=0 captures a sliver (train 0.519); delay≥2 <0.50.
- **Jump continuation** (release-minute move continues): train ~0.49, held-out 0.45-0.47 — negative. **Jump reversal:** train ~0.50, held-out
  0.53-0.55 (test25 0.509) — noise. **Jump∧surprise agreement:** ~0.50.
**VERDICT: NULL for 5-min direction.** FX prices a macro surprise within ~1 MINUTE (the jump is efficient); there is no exploitable 5-min
directional drift/underreaction on EURUSD. The surprise is a MAGNITUDE/volatility event, not a sustained sign (sign-invariance holds even for
fundamental news). The external-data lever I'd flagged as the real shot at >0.65 is empirically closed for DIRECTION. (News-time IS great for
straddle/touch/volatility products — but those aren't up/down.) Deliverables: fetch_calendar.py, event_signs.py, m5_news.py, macro_calendar.parquet.
- **MODEL-based confirmation (m5_news_model.py):** added 5 news features (recent signed surprise, |z|, high-flag, mins-since, in-window) to the
  cross-pair+base+OF model. VAL AUC **0.5235 = unchanged** (no news = 0.5232). News features rank **#276-321 / 336 (bottom)**. News-window AUC
  0.51-0.52 (≤ overall). Accuracy on NY bars in news windows with large surprise: test24 **0.459** / test25 **0.475** / oos 0.513 — BELOW coin-flip
  on test years. A model extracts NOTHING directional from the surprise; news windows are if anything noisier/harder. News-conditioning DEFINITIVELY null for direction.

## OVERALL: 5-min EURUSD up/down >0.65 OOS-verified is NOT achievable — now confirmed even WITH external macro-surprise data. Honest ceiling 0.648 (cross-horizon stack).
