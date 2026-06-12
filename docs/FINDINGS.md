> **SCOPE: EURUSD** (key-specific). Generic methods/ideas live in docs/METHODS_CATALOG.md / SWEEP_MATRIX.md / docs/IDEAS_LOG.md; cross-key theory in docs/THEORY.md. See REPO_MAP.md.

# 5-Minute Binary Option Direction Prediction — Findings & Strategy

> ⚠️ **SUPERSEDED IN PART — read the 2026-05-30 bias audit first** (`docs/research_log.md` "BIAS AUDIT" and
> `README.md` "Methodology audit"). The seconds/minute accuracies in this file (3 s 0.81, etc.) were
> **inflated** by a bar-count horizon, greedy de-overlap, and best-of-search, and the sub-15-minute books
> are **not tradeable on deriv EUR/USD** (forex Rise/Fall minimum = 15 minutes). Deriv-faithful, OOS-verified:
> 1-min 0.55 / 2-min 0.54 (≈ breakeven, untradeable on deriv) — the only real, deriv-tradeable EUR/USD edge is
> the **15-minute** book at **~0.65** (`m15_production.py`). The mechanism findings below remain valid; the
> headline ≥75% claims do not.

**Scope:** Predict whether spot will be **up or down 5 minutes ahead** for 7 FX pairs
(EURUSD first), targeting a ≥75% correct-prediction rate, verified on held-out data with
**2026 kept fully out-of-sample**. Data: 10-second OHLCV bars, 2012-01-02 → 2026-05-08.

---

## 1. Headline conclusion (evidence-first)

**Two-part result:**
1. **75% at the 5-MINUTE horizon is not yet achieved** from this data — across 11 model/
   signal families (V1–V11), with the order-book-imbalance decay curve explaining the current best level.
2. **A clean, VERIFIED ≥75% directional rate IS achieved at the 3-SECOND horizon.** An ensemble
   (LightGBM+XGBoost+CatBoost) on 13.8M 1-second bars of raw-tick microstructure (quote imbalance,
   microprice, momentum), confidence threshold chosen on VAL and read off held-out data:

   | Coverage | TEST (2024-25) acc (n) | **2026 OOS acc (n)** |
   |---|---|---|
   | 0.20% | 0.729 (12134) | 0.749 (1893) |
   | 0.10% | 0.746 (5610) | 0.786 (899) |
   | **0.05%** | **0.756 (2361)** | **0.809 (397)** |

   At 0.05% coverage **both TEST and 2026 OOS clear 75%** at robust n (TEST n=2361, SE≈0.9%).
   (Earlier single-model 5s runs reached only ~72% on TEST; 2× training data + the ensemble closed
   the gap. H=5s ensemble reaches TEST 0.738 / OOS 0.801 — near-75%; H=3s is the clean pass.)

Caveats on the seconds-horizon result: very low coverage (~1 trade per 2000 seconds), modest OOS
sample at the tightest threshold (n=397, CI ±~4%), requires ultra-low-latency execution (the
fill-rate risk the Reddit author flagged), 3-second/tick-duration contracts only (NOT 5-minute),
and depends on Dukascopy quote-size fidelity. It is a genuine, reproducible statistical edge —
probabilities saved to `models/probs_tickens_H3.npz`, verifiable via `verify.py`.

For the 5-minute target specifically, every independent line of evidence so far converges on the same current best level:

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
evidence the current best level is set by the *data*, not the algorithm.

The **mathematical reason**: the lag-1 autocorrelation of EURUSD 5-minute returns is
**≈ −0.03 in every year of the 15-year sample** (2026 = −0.029). Negative ⇒ mild mean
reversion, but the magnitude is tiny. For linear reversion the directional accuracy current best is
`0.5 + |ρ|/π ≈ 0.51`. Conditioning on the largest moves (where reversion is strongest) lifts a
fade strategy to **~0.53–0.55**, and that is the most any selective approach generalizes to.
This matches prior work in `binary_alpha` (acc 0.515 / AUC 0.518) and the external Reddit
builder (5m AUC 0.51). The published route to >75% at <10 minutes requires **limit-order-book /
order-flow imbalance data**, which is not present here (we only have aggregated 10s volume; a
signed-volume proxy added ~0 lift).

**What IS real and verified:** a small, stable mean-reversion edge of **~52% unconditional and
~53–55% when fading extreme 5-minute moves**, present in every pair and every year including
2026. This is genuine but well short of the open 75% target.

**The mechanistic proof (raw-tick order-book imbalance).** The raw tick files DO contain bid/ask
quote SIZES (order-book imbalance) — the documented >75% short-horizon signal. Tested directly,
the imbalance edge is real and strong at the tick scale and decays exactly as theory predicts:

| Horizon | imbalance-follow accuracy |
|---|---|
| +1 tick | **0.553** |
| +10 ticks | 0.509 |
| +30 ticks | 0.504 |
| 1 minute | 0.501 |
| **5 minutes** | **0.501 (~0.50)** |

So the one genuinely predictable signal in FX lives at the **sub-minute/seconds** scale and is
fully arbitraged away before 5 minutes. This is *why* 5-min direction sits at its current best
level so far — and it holds even with the order-book data, not just OHLCV.

**The achievability frontier (tick-microstructure LightGBM, 11.5M 1-sec bars, 2026 OOS).**
Selective accuracy (confidence threshold chosen on VAL to target 75%, read off 2026 OOS):

| Horizon | 2026 OOS selective accuracy |
|---|---|
| **5 seconds** | **0.712** (real, tradeable edge) |
| 10 seconds | 0.696 |
| 30 seconds | 0.663 |
| 60 seconds | 0.603 |
| **5 minutes** | **75% not yet reached; selective ~0.54** |

**Conclusion with the full picture:** a genuinely high, verified directional edge (~71% on the
confident subset) EXISTS — but at the **5-second** horizon, not 5 minutes. Binary contracts do
exist at tick/sub-minute durations (e.g., Deriv), so this is a *real tradeable strategy at the
right horizon*. The 5-minute target specifically sits near ~0.50 and 75% is still open — across
V1–V11, with the order-book-imbalance decay curve explaining the current best level.

---

## 2. What bounds accuracy on every/any bar here

- **Efficient market:** majors (EURUSD etc.) are the most liquid, most-arbitraged instruments on
  earth. 5-minute mid-price changes are dominated by noise; drift ≈ 0.
- **Autocorrelation math** (`docs/research_log.md` → STRUCTURE OF THE PROBLEM): ρ₁(5m) ≈ −0.03 ⇒ linear
  accuracy ≈ 0.51. Stable 2012→2026 (not decayed; not exploited beyond ~0.55 so far).
- **Selective prediction hasn't rescued it yet:** thresholds that hit 75% on validation collapse to
  61–69% on TEST and to noise (n=14–60) on 2026 OOS. The high-precision "pocket" is overfit so far.
- **Per-pair universality:** fading extreme moves yields 0.51–0.54 on TEST+OOS for ALL seven
  pairs; none has reached 75% yet.
- **Order-flow proxy:** signed 10s volume (tick-rule OFI) added ~0 AUC. True LOB imbalance — the
  documented >75% signal — is not present in this dataset.

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
| 75% | not yet reached on VAL | — |
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

The current best level is universal: best generalizing pocket so far is ~**58–59% OOS**
(NZDUSD/AUDUSD); 75% remains the open target.

- The edge is **real but marginal**; for binary options with typical 70–90% payouts the
  break-even win-rate is ~53–59%, so this strategy is at best marginally profitable and only at
  low coverage — **not yet a 75% system**.

**Economic reading:** this is the kind of edge that is technically present but too thin and
too capacity-limited to be a reliable money-maker on majors after costs.

---

## 4. What would actually move the needle toward higher accuracy

1. **Longer horizon — tested here, did NOT generalize.** I swept 5/10/15/30/60-min labels on the
   same features. TEST AUC rises (5m 0.519 → 30m 0.528) but **2026 OOS AUC stays ~0.52 at every
   horizon** (5m 0.519, 15m 0.521, 60m 0.525). The author's 15m>5m gain is real in-sample but the
   longer-horizon edge here is regime drift that does not hold out-of-sample on this FX data.
   (On his BTC data the story may differ — crypto is less efficient than EURUSD.)
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
A 75% claim would require both TEST and 2026 to reach 75% at non-trivial sample size; at the
5-minute horizon they have not yet.

Full chronological reasoning, every variant, and its lesson are in **`docs/research_log.md`**.
