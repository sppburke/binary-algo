# 5-Minute Binary Option Direction Prediction — Findings & Strategy

**Scope:** Predict whether spot will be **up or down 5 minutes ahead** for 7 FX pairs
(EURUSD first), targeting a ≥75% correct-prediction rate, verified on held-out data with
**2026 kept fully out-of-sample**. Data: 10-second OHLCV bars, 2012-01-02 → 2026-05-08.

---

## 1. Headline conclusion (evidence-first)

**A 75% directional hit-rate on 5-minute liquid-FX moves is not attainable from this OHLCV
data — not with indicators, not with gradient boosting, not with multi-timeframe features,
cross-pair lead-lag, an order-flow proxy, or a multi-model ensemble.** Every independent line
of evidence converges on a hard ceiling:

| Approach | TEST AUC (2024-25) | 2026 OOS AUC | Full-coverage acc |
|---|---|---|---|
| LightGBM, 239 multi-TF features (V1) | 0.517 | 0.517 | 0.51 |
| + cross-pair lead-lag, 6 USD peers (V3-B) | 0.520 | 0.519 | 0.51 |
| + order-flow proxy (V3-C) | 0.519 | 0.520 | 0.51 |
| + peer order-flow, 425 feats (V3-D) | 0.520 | 0.521 | 0.51 |
| 3-model ensemble LGBM+XGB+CatBoost blend (V4) | 0.519 | 0.521 | 0.51 |
| Extreme-event specialist (V5, top-decile moves only) | 0.519 | 0.513 |
| TabNet attentive deep tabular net (V6) | 0.513 | 0.519 | 0.51 |

Three independent model families (LightGBM, XGBoost, CatBoost) agree to the **third decimal** —
proof the limit is the *data*, not the algorithm.

The **mathematical reason**: the lag-1 autocorrelation of EURUSD 5-minute returns is
**≈ −0.03 in every year of the 15-year sample** (2026 = −0.029). Negative ⇒ mild mean
reversion, but the magnitude is tiny. For linear reversion the directional accuracy ceiling is
`0.5 + |ρ|/π ≈ 0.51`. Conditioning on the largest moves (where reversion is strongest) lifts a
fade strategy to **~0.53–0.55**, and that is the most any selective approach generalizes to.
This matches prior work in `binary_alpha` (acc 0.515 / AUC 0.518) and the external Reddit
builder (5m AUC 0.51). The published route to >75% at <10 minutes requires **limit-order-book /
order-flow imbalance data**, which is not present here (we only have aggregated 10s volume; a
signed-volume proxy added ~0 lift).

**What IS real and verified:** a small, stable mean-reversion edge of **~52% unconditional and
~53–55% when fading extreme 5-minute moves**, present in every pair and every year including
2026. This is genuine but far below 75%.

---

## 2. Why 75% on every/any bar is structurally impossible here

- **Efficient market:** majors (EURUSD etc.) are the most liquid, most-arbitraged instruments on
  earth. 5-minute mid-price changes are dominated by noise; drift ≈ 0.
- **Autocorrelation math** (`research_log.md` → STRUCTURAL CEILING): ρ₁(5m) ≈ −0.03 ⇒ linear
  accuracy ≈ 0.51. Stable 2012→2026 (not decayed, not exploitable beyond ~0.55).
- **Selective prediction doesn't rescue it:** thresholds that hit 75% on validation collapse to
  61–69% on TEST and to noise (n=14–60) on 2026 OOS. The high-precision "pocket" is overfit.
- **Per-pair universality:** fading extreme moves yields 0.51–0.54 on TEST+OOS for ALL seven
  pairs; none approaches 75%.
- **Order-flow proxy:** signed 10s volume (tick-rule OFI) added ~0 AUC. True LOB imbalance — the
  documented >75% signal — is unavailable in this dataset.

---

## 3. Best achievable strategy (the honest deliverable)

**Model:** LightGBM on 239 causal multi-timeframe features (1m/5m/15m/30m/1h/4h: trend vs EMA,
RSI direction, MACD state, Bollinger %b, range position, realized vol, return autocorrelation,
session/time) **+ 60 cross-pair lead-lag features** from the 6 other USD pairs. Order-flow proxy
optional (no harm, no help).

**Decision rule (selective):** bet only when model confidence `|p−0.5|` exceeds a threshold
chosen on validation. Trades a lower coverage for a higher hit-rate.

**Verified frontier (EURUSD, threshold set on 2022-23 VAL, read off held-out data via `verify.py`):**

| Confidence target | TEST cov / acc | 2026 OOS cov / acc (n) |
|---|---|---|
| 75% | unreachable on VAL | — |
| 70% | 0.05% / 0.661 | 0.015% / 0.684 (n=19, noise) |
| 60% | 0.57% / 0.592 | 0.36% / 0.568 (n=463) |
| 58% | 1.66% / 0.561 | 1.31% / 0.546 (n=1681) |
| 56% | 6.79% / 0.544 | 6.10% / 0.534 (n=7847) |

So the **stable, generalizing** rate is ~**56–59% at 0.5–1.5% coverage**. Above that, accuracy keeps
rising (top-1% confident bets → ~57% TEST) but sample size collapses to noise.

**Per-pair (top-1% most-confident bets, VAL-thresholded):**

| Pair | AUC test/oos | top-1% TEST acc | top-1% OOS acc |
|---|---|---|---|
| EURUSD | 0.519 / 0.521 | 0.571 | 0.554 |
| GBPUSD | 0.524 / 0.518 | 0.643 | 0.529 |
| AUDUSD | 0.521 / 0.518 | 0.598 | **0.579** |
| NZDUSD | 0.517 / 0.517 | 0.646 | **0.593** |
| USDCAD | 0.520 / 0.515 | 0.652 | 0.559 |
| USDCHF | 0.526 / 0.517 | 0.687 | 0.570 |
| USDJPY | 0.522 / 0.522 | 0.566 | 0.549 |

The wall is universal: best generalizing pocket is ~**58–59% OOS** (NZDUSD/AUDUSD), not 75%.

- The edge is **real but marginal**; for binary options with typical 70–90% payouts the
  break-even win-rate is ~53–59%, so this strategy is at best marginally profitable and only at
  low coverage — **not a 75% system**.

**Economic reading:** this is the kind of edge that is technically present but too thin and
too capacity-limited to be a reliable money-maker on majors after costs.

---

## 4. What would actually move the needle toward higher accuracy

1. **Longer horizon.** The same builder/Reddit author gets **15m AUC ≈ 0.57** vs 5m 0.51. If the
   product can tolerate 15m, predictability roughly doubles in excess-over-coinflip terms.
2. **True order-flow / limit-order-book data** (bid/ask sizes, trade signs, depth) — the only
   documented path to >75% at sub-10-minute horizons.
3. **Less-efficient instruments.** Deriv synthetic indices (R_10…R_100, volatility indices) and
   exotic crosses have exploitable structure majors lack; the `binary_alpha` project already
   targets Deriv.
4. **Event/news conditioning** with an economic calendar (NFP/FOMC) — directional order-flow
   persists for minutes around releases (momentum, not reversion).
5. **Higher-frequency execution edge** rather than direction prediction (latency arbitrage of the
   cross-pair lead-lag we detected) — a different business than binary options.

---

## 5. Reproduce & verify

```
~/binary-algo-venv/bin/python pipeline.py EURUSD          # build base features (per year cache)
~/binary-algo-venv/bin/python orderflow.py EURUSD         # order-flow proxy cache
~/binary-algo-venv/bin/python exp_v3.py                   # train + selective eval (cross+OF)
~/binary-algo-venv/bin/python exp_allpairs.py             # per-pair models + saved OOS probs
~/binary-algo-venv/bin/python verify.py models/probs_EURUSD.npz   # honest prediction-rate report
```

`verify.py` chooses a confidence threshold **only on VAL** and reports the resulting accuracy +
coverage on TEST and on the fully-held-out **2026** set — the unbiased live prediction rate.
A 75% claim would require both TEST and 2026 to reach 75% at non-trivial sample size; they do not.

Full chronological reasoning, every variant, and its lesson are in **`research_log.md`**.
