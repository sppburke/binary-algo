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
| F1 | compound the 3s edge → 15m (Gârleanu–Pedersen) | decays to ~0.50 by 15m | ➖ closed |
| F2 | dollar-neutral cross-sectional rank (7 majors) | idio AUC 0.51 ≈ raw | ➖ closed |
| V18–V19 | 15m conditional pockets + regime gating | vol-compression best gate | ★ found compress×NY |
| V20 | meta-labeling (predict primary correctness) | meta-AUC OOS 0.502 | ➖ closed |
| V21–V22 | honest deflation + pooled 7-major power test | corr(VAL,OOS)=−0.54 | ★ killed multiple-testing illusion |
| V23 | EURUSD reconciliation (4-window stability) | OOS 0.58–0.66, no CI thru 0.50 | ★ edge is real, EURUSD-specific |
| V24–V26 | in-pocket stacking + accuracy-coverage curve | ~0.63 reliable / ~0.70 spike | ★ frontier mapped |
| V25 | **horizon-frontier sweep 3–300 s** | **longest ≥75% = 5 s** | ★ 75% is a seconds effect |
| **V27** | **pre-committed 15m pipeline (frozen 2012–23)** | **0.642 held-out 2024–26** (CI [.624,.659]) | ★★ **reproducible ~64%** |

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

## The 15-minute strategy — reproducible ~64% (`exp_15m_v13_proof.py`)

The honest, **pre-committed** 15-minute up/down binary result: **64.2% accuracy on fully held-out
2024–2026** (n=2788, 95% CI [0.624, 0.659]) — comfortably above the ~0.556 break-even of an 0.80
binary payout. Nothing is tuned on the evaluation years; the gate, coverage, and threshold are all
chosen on VAL (2022–23) and then frozen. This is **not** ≥75% — that lives only at the seconds
horizon (see the headline). It *is* a real, reproducible, profitable-looking 15m selective edge.

**Per-year held-out (frozen pipeline):** 2024 **0.691** (n1421) · 2025 **0.590** (n1041) · 2026
**0.592** (n326) · **combined 0.642** (n2788). The >67% prints in any single year/cell are *not*
reproducible under pre-commitment — that is the whole point of V21–V27.

### Methodology (exactly what produces the 64%)
1. **Target.** Up/down binary = sign of the 15-minute return: `y = 1[close(t+15m) > close(t)]`,
   exact ties (`ret==0`) dropped. EURUSD, 1-minute bars (built from 10s OHLCV by `pipeline.py`).
   Only bars where the full 15-minute forward window is contiguous (no session gap) are labelled.
2. **Features.** The 239 causal multi-timeframe features from `pipeline.py` (RSI / MA / Bollinger
   %b & width / ATR / return-autocorrelation / realized-vol / MACD / range-position / EMA-distance
   across 1m·5m·15m·30m·1h·4h, plus hour sin/cos, day-of-week, session flags). No lookahead.
3. **Splits.** TRAIN 2012–2021 (stride 3 to decorrelate overlapping labels) · VAL 2022–2023 ·
   held-out 2024 / 2025 / 2026 (never consulted in any selection step).
4. **Model.** Equal-weight ensemble of LGBM + XGBoost + CatBoost (binary objective, ~2–3k trees,
   `num_leaves/depth` 255/8, `lr` 0.02, `reg_lambda` 10, early-stopping on VAL AUC). Trained on
   TRAIN only. Ensemble VAL AUC ≈ 0.528 (single pair, all bars — the edge is in *selectivity*).
5. **Regime gate (the lever).** Bet only when **(a) volatility compression** — `15m_bb_width` ≤ the
   chosen TRAIN percentile (q33 was selected on VAL; q10/q20/q33 are candidates) **AND (b) NY
   session** (`sess_ny`). Low-vol NY-session bars are where the 15m direction is most predictable.
6. **Selection, on VAL only.** Over candidates (compression depth × coverage ∈ {10%,5%,2%}), pick the
   `(depth, coverage)` and confidence threshold `thr = quantile(|p−0.5|, 1−coverage)` that **maximize
   VAL accuracy** inside the gate, subject to VAL n ≥ 150. (VAL selected **q33 × 2% coverage**,
   VALacc 0.649.) Freeze `thr`.
7. **Decision rule.** On any new bar: predict only if `gate(bar)` AND `|p−0.5| ≥ thr`; direction =
   `1[p > 0.5]`. Otherwise abstain. Coverage is ~2% of NY-session compression bars.
8. **Evaluation.** Apply the frozen rule to 2024 / 2025 / 2026 untouched; report accuracy + 5000×
   bootstrap CI per year and combined. Result: **0.642 combined, CI [0.624, 0.659]**.

> Why not higher: deeper compression (q10/q20) printed 0.70–0.77 in 2026 specifically, but VAL did
> not support those configs (so honest selection rejects them) and the 2025 leg sits at ~0.59. The
> binding constraint is *number of high-accuracy bets EURUSD history provides*, not the idea — see
> V26/V27 in `research_log.md`. Consolidating a stable ≥0.68 needs more compression-regime history.

## The 1-minute strategy — compression-release selective book (`min1_strategy.py`)

A **1-minute (60-second) up/down binary** on EURUSD, predicted from 1-second tick microstructure. Splits
(tick data): TRAIN 2021-2023 · VAL 2024-H1 · TEST 2024.09-2025.11 · **OOS 2026**. Honest evaluation uses a
**non-overlapping (60s-gap)** selective book so every reported bet is independent/tradeable.

**Headline (frozen pipeline, nothing tuned on the evaluation periods):**

| held-out period | accuracy | n | 95% CI | EV/bet @0.80 payout |
|---|---|---|---|---|
| **OOS 2026** | **0.872** | 47 | **[0.766, 0.957]** | **+0.570** |
| OOS 2026 (2nd half) | 0.857 | 42 | [0.738, 0.952] | — |
| TEST 2024-25 | 0.668 | 804 | [0.634, 0.700] | +0.202 |

**OOS 2026 is verified >75%** (CI lower bound 0.766). *Honest caveat:* the larger held-out sample (TEST
2024-25) is 0.668 — the two held-out periods disagree, so accuracy is **not uniform ≥75%** across all windows;
2026 ran favorable. **Both periods are profitable** vs a typical binary break-even (~0.556 @0.80): EV/bet
+0.20 (TEST) to +0.57 (OOS). Broad-sample (n-weighted) accuracy ≈ 0.68-0.70.

### What works at 60s (and what doesn't) — the mechanism
- **Direction is near-efficient (~0.50-0.51 AUC).** Confirmed across a GBM ensemble, a large-move-trained GBM,
  and a temporal 1D-CNN on the raw 1s path — all land ~0.50. You cannot predict *every* 60s bar's direction.
- **Magnitude IS predictable (AUC 0.68).** A model for "is the next 60s move large?" works well; large moves are
  more directional (small moves are spread/bounce noise).
- **The lever is the volatility COMPRESSION-RELEASE regime:** bet only when the last ~30 min were quiet
  (`bbw1800` bottom tercile) AND short-term vol is now expanding (`rel_ratio = bbw300/bbw1800` high) — a
  directional squeeze-breakout. London-NY overlap is *bad* (too noisy); quiet→release is where 60s direction
  is most predictable. The model is used only to *rank confidence* within this regime, then bet selectively.

### Methodology (exactly what produces it)
1. **Target:** `y = sign(mid(t+60s) - mid(t))`, ties dropped, on the 1s grid.
2. **Features (53):** order-book imbalance (+EMAs/accel/persistence), microprice deviation, spread, trade
   count/size, multi-timeframe returns (5s-3600s), realized vol (30s-1800s), EMA-distance, stretch z-scores,
   range-position, and compression (`bbw`) — all causal, computed from the 1s mid/imbalance/microprice.
3. **Direction model:** LGBM+XGB+CatBoost ensemble on TRAIN (`models/probs_min1_v3.npz`).
4. **Regime gate (fixed on VAL):** `bbw1800 ≤ q33` AND `rel_ratio ≥ p90` (compression-release).
5. **Selection (fixed on VAL):** within the gate, bet the top-10%-confidence (`|p-0.5|`) bars, threshold frozen
   on VAL, with a 60s non-overlap constraint.
6. **Evaluate** frozen on TEST 2024-25 and OOS 2026 (+halves) with bootstrap CIs.
7. *(Optional)* a magnitude model (`models/probs_min1_mag.npz`, AUC 0.68) raises TEST accuracy to ~0.76 but
   thins the 3-month OOS too much to verify simultaneously — the OOS sample size is the binding constraint.

## Reproduce

```bash
# Environment (uv-managed venv: numpy/pandas/polars/sklearn/xgboost/lightgbm/catboost/torch)
PY=~/binary-algo-venv/bin/python

$PY pipeline.py EURUSD GBPUSD AUDUSD NZDUSD USDCAD USDCHF USDJPY   # base 239 features (per-year cache)
$PY exp_15m_v13_proof.py                                          # ★ the ~64% 15m pipeline (frozen, held-out)
$PY exp_15m_v12_max.py                                            # 15m accuracy-vs-coverage curve (ceiling map)
$PY tick1s_cache.py && $PY tick_ensemble.py                       # 3s ensemble (the ≥75% seconds result)
$PY tick_horizon_sweep.py                                         # horizon frontier: longest ≥75% horizon (=5s)
$PY verify.py models/probs_tickens_H3.npz                         # verify the 3s result on 2026 OOS
# --- 1-minute strategy ---
$PY min1_v3.py                                                    # direction ensemble -> models/probs_min1_v3.npz
$PY min1_v11.py                                                   # magnitude model    -> models/probs_min1_mag.npz
$PY min1_strategy.py                                              # ★ compression-release 1-min book (OOS 2026 0.872)
```

## What would raise accuracy further
1. **True limit-order-book / order-flow tick data** — a promising route toward >75% at <10 min.
2. **A structurally less-efficient instrument** (crypto, exotic crosses) for a *longer*-horizon edge — *[deferred — majors-only scope for now]*.
3. Harden the 3-second strategy: calibration, execution/fill modeling, walk-forward over all of 2026.

*Research code. Not financial advice; backtest edges are upper bounds and ignore execution/fill/cost.*
