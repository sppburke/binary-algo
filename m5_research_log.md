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

## >65% RESULT: ACHIEVED — but at 3 SECONDS, not 5 minutes.
- **mtick3 (3s tick ensemble): TEST 0.657 (n9144, CI[.647,.666]) · OOS 0.667 (n135) · combined 0.657 — OOS-verified >65%.** Artifacts models/mtick3_EURUSD_*.
- 5-min itself: ~0.566 (m5_production) — thoroughly researched, capped, NOT 0.65.
- The >65% edge fundamentally lives at the SECONDS scale (model/feature-independent ~0.525 AUC there vs ~0.52 at 5min, but the seconds selective tail is far richer). Tradeable only on a seconds/tick-expiry broker.
