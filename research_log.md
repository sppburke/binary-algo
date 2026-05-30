# Research Log — 5-Minute Binary Option Direction Prediction

**Goal:** Predict whether spot price will be **up or down 5 minutes from now** for FX pairs
(starting EURUSD), achieving **≥75% correct prediction rate**, verified on held-out data with
2026 kept fully out-of-sample.

---

## Problem framing & honest priors (read before every new variant)

- **Data:** 10-second OHLCV bars, 2012-01-02 → 2026-05-08, ~8640 bars/day, ~261 days/yr.
  Columns: open, high, low, close, volume. tz-aware UTC index. Pairs: AUDUSD, EURUSD, GBPUSD,
  NZDUSD, USDCAD, USDCHF, USDJPY.
- **Prior attempts (binary_alpha / tick_data):** N-HiTS + 58 TA features → EURUSD 5-min
  **accuracy 0.515, AUC 0.518** (`feature_evaluation_results_*.json`). This is the
  efficient-market wall: predicting *every* 5-min bar's direction from price ≈ coin flip.
- **External evidence (Reddit r/PredictionsMarkets BTC up/down builder):** best 5m AUC 0.51,
  15m AUC 0.57. Confirms 5m is "a different beast." Advice taken: question-based features (not
  raw indicators), watch lookahead bias from smoothing, ensemble of GBMs, speed via Polars.
- **Key realization:** 75% accuracy on *every* bar is not achievable from price/volume on liquid
  FX. The viable path is **SELECTIVE PREDICTION (abstention):** only bet when the model is
  highly confident. We measure **accuracy@coverage** and seek a confidence threshold where
  TEST + 2026-OOS accuracy ≥ 75% at meaningful coverage. This is exactly how a real binary-option
  edge works — you wait for rare high-probability setups, you don't trade every period.
- User guidance: **examine timeseries trends/characteristics across MANY timeframes.**

## Methodology guardrails (non-negotiable, to avoid fooling ourselves)

1. **No lookahead:** every feature uses only data ≤ decision time t. Label uses close(t+5m) vs
   close(t). Verify each indicator is causal (esp. centered smoothing).
2. **Splits (time-ordered):** TRAIN 2012–2021 · VAL 2022–2023 · TEST 2024–2025 · OOS 2026 (never
   touched until final confirmation). Purge+embargo around boundaries (5-min label overlap).
3. **Selective metric:** report AUC, full-coverage accuracy, and accuracy at coverage thresholds.
   A 75% claim must hold on TEST and be confirmed on 2026-OOS at the *same* threshold chosen on VAL.
4. **Costs/ties:** exact ties dropped. Be mindful real binary payout needs strict > / <.
5. **One variant at a time; record learnings here BEFORE starting the next.**

---

## Experiment ledger

| # | Variant | Hypothesis | Result (val/test/OOS) | Verdict | Lesson |
|---|---------|-----------|----------------------|---------|--------|
| V1 | LightGBM, 239 MTF feats, full sample | rich multi-TF feats beat coinflip | AUC 0.523/0.517/0.517; acc@full 0.515/0.511/0.513 | ❌ wall | 5-min direction on all samples ≈ coinflip; selective 75% threshold from VAL collapses to 61/63% on TEST/OOS. Top feats: hour-of-day, return-autocorr across TFs. |
| EDA3 | Confluence-fade hand rule (multi-TF stretch + OF + low-vol) | stacking orthogonal reversion conditions hits 75% pocket | dev 0.645 → TEST 0.60 → OOS 0.43 (n=14) | ❌ overfit | Complex stacked rules OVERFIT dev period; collapse OOS. Order-flow proxy conditions add nothing. |
| YBY | Year-by-year reversion stability | did the edge decay? | fade_q99: 0.50–0.55 every yr; **2026=0.555 (strongest)** | ✅ edge real+stable | Simple reversion (fade extreme move) generalizes at ~52–55%; the edge is NOT gone — it's just SMALL. 75% is the wrong altitude for price-only. |
| V3-A | GBM base only (ablation) | reproduce wall | AUC 0.5225/0.5174/0.5171 | ❌ wall | matches V1 exactly. |
| V3-B | GBM base + cross-pair lead-lag (6 peers) | peer moves add orthogonal signal | AUC 0.5234/0.5199/0.5191 | ➕ tiny real lift | cross-pair lifts AUC ~+0.002–0.003 (TEST 0.5174→0.5199). Real but small; 75% VAL thr → TEST 0.69/OOS noisy. Confirms ceiling. |
| V3-C | GBM base + cross + self order-flow proxy | signed-volume OFI adds short-horizon signal | AUC 0.5233/0.5193/0.5199 | ✖ no lift | OFI proxy from 10s volume adds ~0 over cross-pair (we lack true LOB; proxy too weak). Matches literature: >0.75 at <10min needs LOB imbalance. |
| V3-D | GBM base + cross + self + PEER order-flow (425 feat) | other pairs' flow = cross-asset info | AUC 0.5235/0.5201/0.5208 | ➕ hair | Peer-OF best OOS AUC (0.5208) but within noise; 75% VAL thr → TEST 0.677/OOS 0.531. All 4 ablations cluster AUC≈0.52. **Wall is absolute.** |
| V4 | Ensemble LGBM+XGB+CatBoost + blend (Reddit author's method) | model diversity squeezes more | lgb 0.5186/0.5208, xgb 0.5188/0.5207, cat 0.5190/0.5204, **blend 0.5191/0.5207** | ✖ no lift | 3 independent model families + blend agree to 3rd decimal. VAL top-1% acc only 0.598. **The limit is the data, not the algorithm.** |
| V5 | Extreme-event specialist (train only on top-decile moves) + selective | a model (not hand-rule) finds generalizing high-precision fade pocket | EVENT AUC ~0.51/0.51; OOS selective ~0.52–0.56; 75/70/65% all unreachable on VAL-events | ❌ | Confirms eda3 collapse was the SIGNAL's limit, not just hand-rule overfit. Even in the strongest-reversion regime the ceiling holds. |
| AP | Per-pair models (base+cross+OF), all 7 pairs | some pair more predictable | AUC 0.517–0.526; top-1% selective OOS 0.56–0.59 (USDCHF/NZDUSD best) | ❌ | Wall is universal across pairs; none reaches 75% generalizing. |

---

## Detailed entries

### Setup notes
- venv: `~/binary-algo-venv` (uv-managed). Libs: pandas 3.0.3, numpy 2.4, pyarrow, polars,
  scikit-learn 1.8, xgboost 3.2, lightgbm 4.6, catboost 1.2.10, scipy, matplotlib.
- Workspace: `/media/sean/CORSAIR/binary-algo`. Data (read-only): `/media/sean/CORSAIR/tick_data/processed/{PAIR}/`.
- Data quality: 24h coverage, zero-volume (flat) bars 3–24% depending on year; duplicate/loose
  timestamps possible → dedup + reindex to regular grid in loader.

<!-- Append a dated entry per variant below this line -->

### STRUCTURAL CEILING (2026-05-29) — why 75% is the wrong altitude for price-only
- Lag-1 autocorrelation of EURUSD **5-min returns ≈ −0.03 every year** (2014 −0.046 … 2025 −0.023 …
  2026 −0.029). Negative = mean reversion; magnitude tiny; strongest at the 5-min horizon (good)
  but only −0.03 (hard).
- For linear reversion, directional accuracy ≈ 0.5 + |ρ|/π ≈ **0.51** unconditional; conditioning on
  extreme moves lifts the *fade* edge to ~0.53–0.55 (matches all experiments + every pair).
- **Therefore price-only single-instrument 5-min direction is mathematically capped ≈ 0.51–0.55.**
  Reaching 0.75 would require a LARGE orthogonal signal (cross-pair lead-lag, true order-flow/LOB,
  or news) — limit-order-book imbalance is the only documented route to >0.75 at <10 min, and we do
  not have LOB data (only aggregated 10s volume → weak proxy). Ablations V3/V4 quantify the remaining
  cross-pair + order-flow lift.

### V1 — Baseline LightGBM (2026-05-29)
- **Config:** 239 multi-TF features (1m/5m/15m/30m/1h/4h), train stride=3 (1.22M rows),
  LGBM 255 leaves, lr 0.03, early-stop on VAL AUC. best_iter=95.
- **Result:** VAL AUC 0.5227 (acc 0.515) · TEST AUC 0.5173 (acc 0.511) · OOS AUC 0.5168 (acc 0.513).
  accuracy@coverage: even top-1% confident only ~0.56 acc. VAL→TEST/OOS at "75% threshold":
  VAL 0.75 acc collapses to TEST 0.611 / OOS 0.632 (and at <0.1% coverage = unusable + noisy).
- **Verdict:** ❌ Confirms efficient-market wall (matches prior 0.515 / Reddit 0.51 AUC).
  Selective prediction alone does NOT reach a *generalizing* 75%.
- **Lessons / signal found:**
  1. Top features = `hour_cos/hour_sin` (time-of-day) and `*_autocorr_10` (return autocorrelation
     across all timeframes) + multi-TF volatility. → real structure lives in TIME and in the
     momentum-vs-mean-reversion micro-regime, not in raw level features.
  2. Confidence calibration is weak (AUC 0.52) → need a *better signal source*, not just a better
     threshold. Candidate sources: (a) cross-pair lead-lag (7 USD pairs, unused by prior work),
     (b) regime conditioning (only bet in autocorr/vol regimes with real edge), (c) a more
     predictable target (barrier/magnitude-filtered, or only-strong-move samples).
- **Next:** EDA to locate conditional pockets where P(up) deviates from 0.5 (by hour, by recent
  move strength, by MTF alignment, by cross-pair lead-lag). Then V2 = cross-pair lead-lag features.

### EDA findings (2026-05-29, train+val 2012-2023, N=4.41M)
- **The market MEAN-REVERTS at 5m, it does not trend.** P(up|prev_up)=0.491, P(up|prev_dn)=0.511
  on every timeframe. **Fade (reversion) is the edge; momentum continuation LOSES (~0.49).**
- Edge grows monotonically with move/stretch extremity:
  - Fade |5m_ret|≥q99 → 0.528 reversion acc.
  - Fade MTF-aligned trend (align≥0.95) → P(up)=0.474 (fade up-trend) / 0.529 (fade dn-trend).
  - Composite multi-TF stretch fade: q0.999 → 0.566; coverage 0.07% → 0.564.
  - **Unanimous ALL-TF bb%b>1.0 + fade → 0.645 acc (in-sample) but only 860 samples (0.02% cov).**
- Time-of-day: small effects; h21 P(up)=0.513, h23 P(up)=0.489 (weak alone).
- **Implication:** A *generalizing* 75% needs to ENRICH this rare high-precision reversion pocket
  with ORTHOGONAL confirming signals prior work never used: (a) cross-pair lead-lag (7 USD pairs),
  (b) intrabar order-flow imbalance from 10s volume, (c) volatility/time regime. Then let a
  calibrated GBM/meta-labeler find the rare "perfect-storm" fade configs and bet only those.
- **Honest prior:** price-only single-pair caps ~0.52 AUC / ~0.53-0.56 selective. 75% is only
  plausible at very low coverage IF orthogonal signals stack. Will measure exactly how far it goes
  on TEST + 2026 OOS and report the true frontier.
