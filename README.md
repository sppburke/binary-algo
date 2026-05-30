# binary-algo — FX binary-option direction prediction (research)

Predicting **price direction (up/down from current spot)** for FX pairs from tick/OHLCV data,
with a rigorous, leakage-controlled, out-of-sample methodology. Started from a 5-minute target on
EURUSD and mapped the full **accuracy-vs-horizon frontier** across 17 strategy variants.

> **Data:** 10-second OHLCV bars + raw sub-second ticks (bid/ask + quote sizes), 2012-01-02 →
> 2026-05-08, 7 USD pairs (EURUSD, GBPUSD, AUDUSD, NZDUSD, USDCAD, USDCHF, USDJPY).
> Source dir (read-only): `/media/sean/CORSAIR/tick_data/{processed,raw}`.
> **Splits:** TRAIN 2012–2021 · VAL 2022–2023 · TEST 2024–2025 · **OOS 2026 (held out everywhere)**.

---

## Headline result (two parts, both verified out-of-sample)

| Horizon / data | Best generalizing selective accuracy (2026 OOS) | 75%? |
|---|---|---|
| **3 seconds** — tick order-book microstructure (ensemble) | **TEST 0.756 / OOS 0.809–0.814** @0.05% coverage | ✅ |
| **5 seconds** — same | **TEST 0.760 / OOS 0.814** @0.02% coverage | ✅ |
| 8 seconds — same | OOS 0.753 / TEST 0.702 | borderline |
| 15–30 seconds — same | OOS ~0.70–0.76 / TEST ~0.62–0.65 @0.05% | ✖ not both-sides |
| 1–5 minutes — OHLCV + everything | ~0.55–0.60 selective | ✖ |
| 15 minutes — OHLCV + cross-pair + daily + calendar | ~0.52 global; **EURUSD compress×NY ≈ 0.60–0.64** (4-window-stable) | ✖ |

> **Horizon-frontier sweep (`tick_horizon_sweep.py`):** the longest horizon clearing ≥75% on **both** TEST and
> 2026 OOS (n≥100) is **5 seconds**. The directional edge is an order-book-imbalance microstructure phenomenon
> that decays to coin-flip by ~30–60 s. A >75% binary-direction edge on liquid FX majors is a **seconds-scale**
> effect; it does not reach a binary-tradeable expiry. The best *tradeable-horizon* book is the validated
> EURUSD 15m compression×NY-session selective pocket at ~0.60–0.64 OOS (profitable vs an 0.80 payout, but <75%).

1. **5-minute (and 15-minute) 75% is still open on liquid EURUSD** — explored across 11+ model/
   signal families and *mechanistically explained*: the lag-1 autocorrelation of 5-min returns is
   ≈ −0.03 (current best linear directional accuracy ~0.51), and the only strong signal — order-book
   imbalance — decays from **55.3% at the next tick → ~0.50 by 1 minute → gone by 5 min**.
2. **75% IS achieved at the 3-second horizon.** An LGBM+XGB+CatBoost ensemble on 13.8M 1-second
   bars of tick microstructure reaches **75.6% TEST / 80.9% OOS** on the top-0.05%-confidence
   signals. *Honest bounds:* extreme selectivity only (~1 bet/2000 s), small OOS n (397), latency-
   critical, Dukascopy quote-size fidelity; a less-selective threshold generalizes to ~70%.

**Bottom line:** the predictable directional edge in liquid FX surfaces most clearly at the
**seconds** scale and is much weaker by 5 minutes. Reaching ≥75% has so far come from that short
horizon; other routes — true order-book data, or a structurally less-efficient instrument — remain
open to explore.

---

## Experiment ledger (full detail in [`research_log.md`](research_log.md))

| # | Variant | TEST/OOS AUC | Verdict |
|---|---------|-------------|---------|
| V1 | LightGBM, 239 multi-timeframe features | 0.517 / 0.517 | — best ~0.517, open |
| EDA | reversion vs momentum, stretch fade | — | market mean-reverts ~0.51–0.55 |
| V3-A..D | + cross-pair lead-lag + order-flow proxy + peer-OF | 0.520 / 0.521 | ➕ tiny lift |
| V4 | ensemble LGBM+XGB+CatBoost | 0.519 / 0.521 | — best ~0.521, open |
| V5 | extreme-event specialist | 0.519 / 0.513 | — best ~0.519, open |
| V6 | TabNet (deep tabular) | 0.513 / 0.519 | — best ~0.519, open |
| V7 | stat-arb USD-basket residual | ~0.54 fade | ➕ marginal |
| V8 | horizon sweep 5/10/15/30/60m | OOS ~0.52 all | ➖ no lift yet |
| V9 | GRU sequence net | 0.51 / 0.52 | — best ~0.52, open |
| V10 | **raw-tick order-book imbalance** | 55.3% next-tick → 0.50 @1min | ★ mechanism |
| V11 | tick-microstructure model, 5s→5min frontier | 5s 0.71 → 5min ~0.54 | ★ frontier |
| V12 | optimized 5s model | TEST ~0.72 / OOS ~0.78 | ★ near-75% |
| **V13** | **3s ensemble, 13.8M bars** | **TEST 0.756 / OOS 0.809 @0.05%** | ★★★ **≥75% ✓** |
| V14–V17 | 15m: ensemble, daily context, exogenous peers, calendar | 0.527 / 0.520; sel ~0.63 | ➕ best 0.632 OOS, 75% open |

Each row's hypothesis, config, full result, and lesson are recorded in `research_log.md`.

---

## Repository structure

**Pipelines / features**
- `pipeline.py` — causal multi-timeframe (1m/5m/15m/30m/1h/4h) feature engine (239 features) from
  10s OHLCV; caches `features/{PAIR}_{YEAR}.parquet`.
- `crosspair.py` — cross-pair lead-lag features (6 peer pairs).
- `orderflow.py` — signed-volume order-flow proxy.
- `exog.py` — exogenous longer-horizon peer features + USD-basket / stat-arb residual factors.
- `dailyctx.py` — daily/weekly context (multi-day trend, range position, seasonality).
- `eventtime.py` — compliant economic-calendar proxy (release-window volatility seasonality).
- `rawtick_probe.py`, `rawtick_decay.py` — raw-tick (bid/ask/size) microstructure analysis.
- `tick1s_cache.py` — caches 1-second microstructure bars from raw ticks.

**Harness / evaluation**
- `harness.py` — splits + selective metrics (accuracy@coverage, threshold-for-target).
- `verify.py` — standalone verifier: picks a confidence threshold on VAL, reads accuracy + coverage
  off TEST and **2026 OOS**. `python verify.py models/probs_*.npz`.

**Experiments** — `exp_baseline.py` (V1), `eda*.py`, `exp_v2..v5.py`, `exp_seq.py` (GRU),
`exp_horizon.py` (V8), `statarb.py` (V7), `tickmodel.py`/`tickmodel_opt.py`/`tick5s_final.py`/
`tick_ensemble.py` (V10–V13), `exp_15m*.py` (V14–V17).

**Records** — `research_log.md` (the running ledger + verdicts), `FINDINGS.md` (conclusions,
strategy, what would raise accuracy), `README.md` (this file).

> Large/regenerable artifacts (`features/`, `features_tick/`, `models/`, `*.parquet`, `*.npz`) are
> git-ignored. Regenerate via the pipeline scripts.

---

## Reproduce

```bash
# Environment (uv-managed venv: numpy/pandas/polars/sklearn/xgboost/lightgbm/catboost/torch)
PY=~/binary-algo-venv/bin/python

$PY pipeline.py EURUSD GBPUSD AUDUSD NZDUSD USDCAD USDCHF USDJPY   # base features (per-year cache)
$PY orderflow.py EURUSD ...                                        # order-flow proxy cache
$PY exp_v3.py                                                      # 5m baseline + ablations
$PY tick1s_cache.py && $PY tick_ensemble.py                        # 3s ensemble (the ≥75% result)
$PY exp_15m_v3.py                                                  # 15m with exogenous pairs
$PY verify.py models/probs_tickens_H3.npz                          # verify the 3s result on 2026 OOS
```

## What would raise accuracy further
1. **True limit-order-book / order-flow tick data** — a promising route toward >75% at <10 min.
2. **A structurally less-efficient instrument** (crypto, exotic crosses) for a *longer*-horizon edge — *[deferred — majors-only scope for now]*.
3. Harden the 3-second strategy: calibration, execution/fill modeling, walk-forward over all of 2026.

*Research code. Not financial advice; backtest edges are upper bounds and ignore execution/fill/cost.*
