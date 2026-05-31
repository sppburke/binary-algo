# binary-algo — FX binary-option direction prediction (research)

Predicting **price direction (up/down from current spot)** for FX pairs from tick/OHLCV data,
with a rigorous, leakage-controlled, out-of-sample methodology. Started from a 5-minute target on
EURUSD and mapped the full **accuracy-vs-horizon frontier** across 17 strategy variants.

> **Data:** 10-second OHLCV bars + raw sub-second ticks (bid/ask + quote sizes), 2012-01-02 →
> 2026-05-08, 7 USD pairs (EURUSD, GBPUSD, AUDUSD, NZDUSD, USDCAD, USDCHF, USDJPY).
> Source dir (read-only): `/media/sean/CORSAIR/tick_data/{processed,raw}`.
> **Splits:** TRAIN 2012–2021 · VAL 2022–2023 · TEST 2024–2025 · **OOS 2026 (held out everywhere)**.

---

## ⚠️ Methodology audit & deriv-faithful correction (2026-05-30)

An end-to-end bias audit (multi-agent review + independent Tier-1 re-runs) found the earlier headline
tick numbers (1m OOS 0.777, 2m OOS 0.764, 3s OOS 0.809) were **inflated by methodology errors**, and that
the sub-15-minute books are **not even tradeable on deriv EUR/USD**. All production pipelines were fixed,
retrained, and re-verified out-of-sample. Full record: `research_log.md` ("BIAS AUDIT"); pre-deploy guard:
`audit_leakage.py`. The fixes:

- **True fixed-expiry label** — the old `mid.shift(-HS)` counted *bars* (empty seconds are dropped), so a
  "60 s" horizon was a **median 111 wall-clock seconds** (variable, ~2× nominal) that matches no fixed binary
  expiry and broke trade independence. Now a strict wall-clock expiry (`wc_ret`).
- **Honest trade independence** — de-overlap is now **chronological/live-faithful** (no greedy confidence
  look-ahead, which inflated ~3–9 pts); bootstrap CIs are over genuinely independent trades.
- **Deriv-faithful settlement** — verified vs deriv T&C + live `contracts_for` API: Rise/Fall settles
  **mid-to-mid, tick-to-tick, NO spread**; entry = the **next tick after the order**, exit = last tick ≤ expiry,
  **ties lose**; the broker edge is a **payout deduction** (~15 % → breakeven ~0.541). (An earlier spread penalty
  was wrong and was removed.)
- **No best-of-search** — each production model is one pre-committed config (regime from TRAIN, threshold from
  VAL), judged once on TEST + OOS. The repo's own `corr(VALacc, OOSacc) = −0.54` shows VAL-selection was an
  unreliable OOS predictor — the old headlines were search maxima.

### Deriv tradeability (live-API verified)

deriv `asset_index` / `contracts_for` for `frxEURUSD`: **forex EUR/USD Rise/Fall minimum = 15 minutes** (max
365 days), time units only — **no ticks, no seconds, no sub-15-minute expiries**. Those exist only on **synthetic
indices**, not forex. So the 1 m / 2 m books **cannot be placed on deriv EUR/USD**; only the 15-minute (and longer:
30 m / 1 h / …) Rise/Fall is tradeable. Higher/Lower min = 1 day.

### Corrected, deriv-faithful out-of-sample results

| Production model | Horizon | Deriv EUR/USD tradeable? | TEST 2024-25 | OOS 2026 | Verdict |
|---|---|---|---|---|---|
| `min1_production.py` | 60 s | ❌ below 15 m floor | 0.539 (n1017) | 0.550 (n349, CI[.499,.602]) | ~breakeven — no edge |
| `min2_production.py` | 120 s | ❌ below 15 m floor | 0.528 (n2528) | 0.539 (n710, CI[.503,.576]) | ~coin-flip — no edge |
| **`m15_production.py`** | **15 min** | ✅ **at the floor** | 2024 0.689 / 2025 0.582 | **0.663** (n89, CI[.562,.753]) | **Combined 0.647 (n677, CI[.612,.684]) — the real deriv edge** |

deriv payout-deduction EV (R ≈ 1.85, breakeven 0.541): **m15 combined +0.197** (profitable); min1 OOS +0.018,
min2 OOS −0.002 (marginal/negative — and untradeable anyway).

**Bottom line:** the only genuine, deriv-tradeable, out-of-sample-verified EUR/USD up/down edge is the
**15-minute** volatility-compression × NY Rise/Fall book at **~0.64–0.66** — comfortably profitable against
deriv's payout deduction and not eroded by entry latency (a next-tick entry is negligible at 15 m). The
seconds-to-2-minute microstructure edge is **real but uncapturable on deriv EUR/USD**: it decays within a tick or
two of entry latency and is below the 15-minute forex floor (it would only be tradeable on synthetic indices or
another broker).

> **Everything below this line is the *pre-audit* research record**, kept for provenance. Its tick/seconds
> accuracies (3 s 0.81, 1 m 0.78, 2 m 0.76, etc.) **overstate the tradeable edge** — read them through the audit
> above (they reflect the bar-count horizon, greedy de-overlap, and best-of-search before correction).

---

## Pre-audit research record (superseded by the audit above)

| Horizon / data | Pre-audit selective accuracy (2026 OOS) — *overstated* | 75%? |
|---|---|---|
| 3 seconds — tick order-book microstructure (ensemble) | TEST 0.756 / OOS 0.809–0.814 @0.05% cov | (pre-audit) |
| 5 seconds — same | TEST 0.760 / OOS 0.814 @0.02% cov | (pre-audit) |
| 2 minutes — 1s tick microstructure, compression-release reversion | OOS 0.764 / TEST 0.694 → **deriv-faithful 0.539/0.528** | corrected |
| 1 minute — 1s tick microstructure, compression-release + reversion | OOS 0.777 / TEST 0.739 → **deriv-faithful 0.550/0.539** | corrected |
| 15 minutes — OHLCV + cross-pair + daily + calendar | EURUSD compress×NY ≈ 0.60–0.64 → **holds: 0.647 combined** | ✅ deriv |

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

## The 1-minute strategy — compression-release + reversion book (`min1_production.py` v2)

A **1-minute (60-second) up/down binary** on EURUSD, predicted from 1-second tick microstructure. Splits
(tick data): TRAIN 2021-2023 · VAL 2024-H1 · TEST 2024.09-2025.11 · **OOS 2026**. Honest evaluation uses a
**non-overlapping (60s-gap)** selective book so every reported bet is independent/tradeable.

**v2 upgrade (applied the 2-minute model's lessons — strictly better than v1):** added a **reversion trend
filter** (bet AGAINST the last 5-min move) and a **compression-release direction specialist** blended 50/50 with
the all-bars ensemble. This fixes v1's two weaknesses (TEST only ~0.68; OOS thin and April-concentrated) at once.

**Headline (frozen pipeline, nothing tuned on the evaluation periods):**

| pipeline | TEST 2024-25 | OOS 2026 | OOS month spread |
|---|---|---|---|
| v1 (compression-release only) | 0.682 (n759) | 0.780 (n50) | 44/50 in April |
| **v2 (+ reversion + specialist)** | **0.739 (n1033)** | **0.777 (n184)** | **Feb .892 / Mar .850 / Apr .710** |

**Read this honestly.** v2 lifts the large-sample TEST **+6 points** (0.682→0.739) AND multiplies the OOS trade
count **3.7×** (50→184), now spread across **all three** 2026 months instead of one — OOS CI95 [0.712, 0.837].
Profitable vs typical binary payouts (break-even 0.556 @0.80): EV/bet +0.33 (TEST) to +0.40 (OOS). The OOS-2026
0.777 is a steady, month-distributed result; the all-period rate is ~0.74. Still a low-frequency, opportunistic
book that abstains most of the time.

### What works at 60s (and what doesn't) — the mechanism
- **Direction is near-efficient (~0.50-0.51 AUC).** Confirmed across a GBM ensemble, a large-move-trained GBM,
  and a temporal 1D-CNN on the raw 1s path — all land ~0.50. You cannot predict *every* 60s bar's direction.
- **The lever is the volatility COMPRESSION-RELEASE regime:** bet only when the last ~30 min were quiet
  (`bbw1800` low) AND short-term vol is now expanding (`rel_ratio = bbw300/bbw1800` high) — a squeeze-breakout.
- **REVERSION is the v2 lever (transferred from the 2-min model):** in that regime, **bet against the last 5-min
  move** (`sign(p-0.5) = −sign(ret300)`). A quiet market that just pushed tends to mean-revert over the next
  minute — this is a sub-minute-to-2-minute microstructure effect (it is gone by 15m). Adding it lifted TEST ~6 pts.
- **A regime specialist** (LGBM trained only on compression-release bars) blended 50/50 with the all-bars
  ensemble broadens the confident set (in-regime AUC 0.510→0.519), so the OOS sample spans all 3 months.

### Methodology (exactly what produces it)
1. **Target:** `y = sign(mid(t+60s) - mid(t))`, ties dropped, on the 1s grid; 60s non-overlap.
2. **Features (62):** order-book imbalance (+EMAs/accel/persistence), microprice deviation, spread, trade
   count/size, multi-timeframe returns (5s-3600s), realized vol (30s-3600s), EMA-distance, stretch, range-
   position, compression (`bbw`), `rel_ratio` — all causal, from the 1s mid/imbalance/microprice.
3. **Direction = 0.5·all-bars ensemble (LGBM+XGB+CatBoost) + 0.5·compression-release LGBM specialist.**
4. **Regime gate (fixed on TRAIN/VAL):** `bbw1800 ≤ train_q67` AND `rel_ratio ≥ train_p70`, rel re-tightened to
   the VAL p80, AND reversion: `sign(p−0.5) = −sign(ret300)`.
5. **Selection (fixed on VAL):** 5% confidence coverage, threshold frozen on VAL, 60s non-overlap.
6. **Evaluate** frozen on TEST 2024-25 and OOS 2026 with bootstrap CIs and per-month breakdown.
- *(Note — what did NOT transfer to 15m:* the same reversion+specialist recipe gives no lift at 15 minutes
  (best ~0.62 < V27's 0.642); by 15m the move is efficient and the reversion effect is gone. See `research_log.md`
  "Cross-pollination".)

### Production pipeline (`min1_production.py`) — train, serialize, infer
**Per-pair model (default EURUSD).** All artifacts are **labeled with the pair** so other currencies sit
alongside; each pair is trained separately (the edge is EURUSD-concentrated — see V22 — so do *not* share one
model across pairs). Serialized to `models/`:
- `models/min1_EURUSD_direction_lgb.txt`, `..._xgb.json`, `..._cat.cbm` — the all-bars direction ensemble.
- `models/min1_EURUSD_direction_spec_lgb.txt` — the compression-release LGBM specialist (blended 50/50).
- `models/min1_EURUSD_magnitude.joblib` — the magnitude model (P(|ret60| large)), kept for info.
- `models/min1_EURUSD_strategy.json` — frozen params: pair, 62 feature names, `w_spec`, compression
  `bbw1800_q67`, release `rel_p70`, `rel_tighten`, reversion rule, `conf_thr`, horizon (60s), gap (60s).

```bash
PY=~/binary-algo-venv/bin/python
$PY min1_production.py train            # EURUSD: train 2021-2023, freeze on VAL, write models/min1_EURUSD_* + report
$PY min1_production.py backtest         # replay TEST 2024-25 + OOS 2026 (trades, accuracy, EV)
$PY min1_production.py train GBPUSD     # another pair (needs that pair's 1s cache under features_tick_GBPUSD/)
```

### Production pipeline (`m15_production.py`) — the 15-minute model, same treatment
Per-pair (default EURUSD); 239-feature parquets already exist for all 7 majors, so other pairs need only the
argument. Serializes `models/m15_EURUSD_direction_{lgb.txt,xgb.json,cat.cbm}` + `m15_EURUSD_strategy.json`
(pair, feature names, selected compression depth `comp_q`, `bb_width_thr`, coverage, `conf_thr`).
```bash
$PY m15_production.py train             # EURUSD: train 2012-2021, freeze on VAL, write models/m15_EURUSD_* + per-year backtest (~0.64)
$PY m15_production.py train GBPUSD      # another major (parquets already cached by pipeline.py)
```

**Live inference** (`from min1_production import Min1Strategy`): feed a rolling buffer of ≥3600 recent 1-second
bars `[mid, imb, micro, spread, nt, tsz]`; `strategy.signal(buffer)` returns `{"trade": bool, "direction":
±1, "confidence": float}` — `trade=True` only when the bar is in the compression-release regime **and** the
direction confidence clears the frozen threshold (so it abstains most of the time, by design). Decision uses
only data up to `t`; the binary settles on `mid(t+60s)`. Enforce the 60s non-overlap (one open position).

## The 2-minute strategy — compression-release reversion book (`min2_production.py`) ★ best tradeable horizon

A **2-minute (120-second) up/down binary** on **EURUSD specifically**, from 1-second tick microstructure. Same
splits and non-overlapping (120s-gap) selective evaluation as the 1-min book. **This is the strongest of the
tradeable horizons** — it clears >75% out-of-sample.

**Headline (frozen pipeline, nothing tuned on the evaluation periods):**

| held-out period | accuracy | trades (n) | 95% CI | EV/bet @0.80 payout |
|---|---|---|---|---|
| VAL 2024-H1 | 0.763 | 375 | — | +0.373 |
| TEST 2024-25 | 0.694 | 1591 | [0.671, 0.717] | +0.249 |
| **OOS 2026 (Feb–May)** | **0.764** | **330** | **[0.718, 0.809]** | **+0.375** |

**Read this honestly.** OOS-2026 = **0.764 and is month-consistent** — Feb 0.806 (n93), Mar 0.753 (n77), Apr
0.744 (n160) — so unlike the 1-min book (whose 0.872 was 43/47 trades in a single month), the >75% here is a
**steady, all-three-months result on 330 independent trades.** The honest nuance is the other direction this
time: the larger **TEST 2024-25 sample is 0.694**, so the *all-period* selective rate is ~0.70–0.72 and the
**>75% specifically describes the 2026 out-of-sample period** (verified, pre-committed, CI lower bound 0.718).
**Profitable on both** held-out periods vs an 0.80 payout (break-even 0.556): EV/bet +0.25 (TEST) to +0.375 (OOS).

### Why 120s works where 60s and 15m don't — the mechanism
- **Direction is still near-efficient unconditionally (~0.51 AUC)** — same as every horizon. The edge is regime.
- **vs 1-minute:** at 60s the move is dominated by spread / bid-ask-bounce noise → compression-release direction
  tops out ~0.66. At **120s the *released* volatility produces a move large enough to dominate the spread**, so
  direction-given-release climbs to ~0.72–0.76. The horizon escapes the microstructure noise floor.
- **vs 15-minute:** by 15m the move is fully developed and efficient (no regime edge survives, ~0.52).
- **120s is the sweet spot:** long enough to clear bounce noise, short enough that compression-release still
  predicts the breakout. **A regime-specialist direction model** (trained only on compression-release bars) lifts
  in-regime AUC 0.510→0.531; blended 50/50 with the all-bars ensemble it denoises the in-regime ranking.
- **The trade rule is reversion:** in a compressed→releasing market, **bet AGAINST the last 5-minute move**
  (`sign(p−0.5) = −sign(ret300)`) — a quiet market that just pushed tends to revert over the next 2 minutes.

### Methodology (exactly what produces the 0.764 OOS)
1. **Target:** `y = sign(mid(t+120s) − mid(t))`, ties dropped, on the 1s grid; 120s non-overlap.
2. **Features (62):** the 1-min set + longer-horizon vol/range (`bbw3600`, `rv3600`, `rel_ratio2`, `stretch120`).
3. **Direction = 0.5·all-bars ensemble (LGBM+XGB+CatBoost) + 0.5·compression-release LGBM specialist.** Blend
   weight `W=0.5` fixed by argmax VAL in-regime AUC.
4. **Regime gate (fixed on TRAIN/VAL):** `bbw1800 ≤ train_q67` AND `rel_ratio ≥ train_p70` (compression-release).
5. **Trend filter:** reversion vs `ret300` (bet against the last 5-min move). Selected on VAL from
   {none, rev300, rev900, rev3600}.
6. **Selection (fixed on VAL):** 3% confidence coverage (`conf_thr` = 97th pct of `|p−0.5|` over VAL regime bars).
   Family swept on VAL only; argmax VAL accuracy s.t. n≥350 → `rev300, cov 0.03`. **OOS judged once.**

### Production pipeline (`min2_production.py`) — train, serialize, infer
Per-pair (default EURUSD), **EURUSD-labeled artifacts** (train other currencies with a `PAIR` argument; each
needs its own 1s cache under `features_tick_<PAIR>/`). Serialized to `models/`:
- `models/min2_EURUSD_dir_v1_lgb.txt`, `..._xgb.json`, `..._cat.cbm` — all-bars direction ensemble.
- `models/min2_EURUSD_dir_spec_lgb.txt` — compression-release LGBM specialist.
- `models/min2_EURUSD_magnitude.joblib` — P(|ret120| large), kept for info.
- `models/min2_EURUSD_strategy.json` — frozen params: pair, 62 feature names, `w_spec`, `bbw1800_q67`,
  `rel_p70`, trend rule, `conf_thr`, coverage, horizon (120s), non-overlap gap (120s).

```bash
PY=~/binary-algo-venv/bin/python
$PY min2_production.py train            # EURUSD: train 2021-2023, freeze on VAL, write models/min2_EURUSD_* + report
$PY min2_production.py backtest         # replay TEST 2024-25 + OOS 2026 (trades, accuracy, EV, per-month)
$PY min2_production.py train GBPUSD     # another pair (needs that pair's 1s cache under features_tick_GBPUSD/)
```

**Live inference** (`from min2_production import Min2Strategy`): feed a rolling buffer of ≥3700 recent 1-second
bars `[mid, imb, micro, spread, nt, tsz]`; `strategy.signal(buffer)` returns `{"trade": bool, "direction": ±1,
"confidence": float, "p_up": float, "in_regime": bool}` — `trade=True` only when the bar is in the
compression-release regime, the direction opposes the last 5-min move, **and** confidence clears the frozen
threshold. Decision uses only data ≤ `t`; the binary settles on `mid(t+120s)`. Enforce 120s non-overlap.

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
# --- 1-minute strategy (production) ---
$PY min1_production.py train                                      # ★ train+serialize models/min1_* + held-out report
$PY min1_production.py backtest                                   # replay TEST 2024-25 + OOS 2026 from saved artifacts
# --- 2-minute strategy (production) — best tradeable horizon, ≥75% OOS ---
$PY min2_production.py train                                      # ★ train+serialize models/min2_EURUSD_* + held-out report
$PY min2_production.py backtest                                   # replay TEST 2024-25 (0.694) + OOS 2026 (0.764) per-month
```

## What would raise accuracy further
1. **True limit-order-book / order-flow tick data** — a promising route toward >75% at <10 min.
2. **A structurally less-efficient instrument** (crypto, exotic crosses) for a *longer*-horizon edge — *[deferred — majors-only scope for now]*.
3. Harden the 3-second strategy: calibration, execution/fill modeling, walk-forward over all of 2026.

*Research code. Not financial advice; backtest edges are upper bounds and ignore execution/fill/cost.*
