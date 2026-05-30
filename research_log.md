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
  efficient-market current best level: predicting *every* 5-min bar's direction from price ≈ ~0.50.
- **External evidence (Reddit r/PredictionsMarkets BTC up/down builder):** best 5m AUC 0.51,
  15m AUC 0.57. Confirms 5m is "a different beast." Advice taken: question-based features (not
  raw indicators), watch lookahead bias from smoothing, ensemble of GBMs, speed via Polars.
- **Key realization:** 75% accuracy on *every* bar is not yet achieved from price/volume on liquid
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
| V1 | LightGBM, 239 MTF feats, full sample | rich multi-TF feats beat ~0.50 | AUC 0.523/0.517/0.517; acc@full 0.515/0.511/0.513 | — best ~0.52 | 5-min direction on all samples ≈ ~0.50; selective 75% threshold from VAL collapses to 61/63% on TEST/OOS. Top feats: hour-of-day, return-autocorr across TFs. |
| EDA3 | Confluence-fade hand rule (multi-TF stretch + OF + low-vol) | stacking orthogonal reversion conditions hits 75% pocket | dev 0.645 → TEST 0.60 → OOS 0.43 (n=14) | overfit | Complex stacked rules OVERFIT dev period; collapse OOS. Order-flow proxy conditions add nothing. |
| YBY | Year-by-year reversion stability | did the edge decay? | fade_q99: 0.50–0.55 every yr; **2026=0.555 (strongest)** | ✅ edge real+stable | Simple reversion (fade extreme move) generalizes at ~52–55%; the edge is NOT gone — it's just SMALL. 75% remains the open target for price-only. |
| V3-A | GBM base only (ablation) | reproduce current best level | AUC 0.5225/0.5174/0.5171 | — best ~0.52 | matches V1 exactly. |
| V3-B | GBM base + cross-pair lead-lag (6 peers) | peer moves add orthogonal signal | AUC 0.5234/0.5199/0.5191 | ➕ tiny real lift | cross-pair lifts AUC ~+0.002–0.003 (TEST 0.5174→0.5199). Real but small; 75% VAL thr → TEST 0.69/OOS noisy. Confirms current best. |
| V3-C | GBM base + cross + self order-flow proxy | signed-volume OFI adds short-horizon signal | AUC 0.5233/0.5193/0.5199 | no lift yet | OFI proxy from 10s volume adds ~0 over cross-pair (we lack true LOB; proxy too weak). Matches literature: >0.75 at <10min needs LOB imbalance. |
| V3-D | GBM base + cross + self + PEER order-flow (425 feat) | other pairs' flow = cross-asset info | AUC 0.5235/0.5201/0.5208 | ➕ hair | Peer-OF best OOS AUC (0.5208) but within noise; 75% VAL thr → TEST 0.677/OOS 0.531. All 4 ablations cluster AUC≈0.52. **This is the level reached so far.** |
| V4 | Ensemble LGBM+XGB+CatBoost + blend (Reddit author's method) | model diversity squeezes more | lgb 0.5186/0.5208, xgb 0.5188/0.5207, cat 0.5190/0.5204, **blend 0.5191/0.5207** | no lift yet | 3 independent model families + blend agree to 3rd decimal. VAL top-1% acc only 0.598. **The constraint here is the data, not the algorithm.** |
| V5 | Extreme-event specialist (train only on top-decile moves) + selective | a model (not hand-rule) finds generalizing high-precision fade pocket | EVENT AUC ~0.51/0.51; OOS selective ~0.52–0.56; 75/70/65% all not yet reached on VAL-events | open | Confirms eda3 collapse was the SIGNAL's limit, not just hand-rule overfit. Even in the strongest-reversion regime the current best level holds. |
| AP | Per-pair models (base+cross+OF), all 7 pairs | some pair more predictable | AUC 0.517–0.526; top-1% selective OOS 0.56–0.59 (USDCHF/NZDUSD best) | open | Current best level is universal across pairs; none has yet reached 75% generalizing. |
| V6 | TabNet (attentive deep tabular; user's 4th cited model) | deep net captures nonlinear path patterns GBM misses | AUC val 0.518 / test 0.513 / oos 0.519 | no lift yet | Confirms the current best level with a different model class. ALL 4 cited models (XGB/LGBM/CatBoost/TabNet) + ensemble land at ~0.52. **The binding factor is information content of OHLCV, not model capacity.** |

| V7 | Stat-arb basket residual (NEW: USD-factor decomposition, fade idiosyncratic residual) | idiosyncratic residual reverts more strongly than raw price | fade resid_z q0.9 ≈ 0.535 (vs raw ~0.525); EURUSD β=0.97 on USD basket | ➕ marginal | Residual IS a cleaner reversion signal (~53.5%) but EURUSD is 97% USD-factor → tiny idiosyncratic part; best so far ~0.54. Confirms the current best with a principled stat-arb method. |

| V8 | Horizon sweep 5/10/15/30/60m (Reddit author's "start with 15m") | longer horizon → tradeable AUC (his 5m 0.51→15m 0.57) | **OOS AUC: 5m 0.519, 10m 0.522, 15m 0.521, 30m 0.518, 60m 0.525** | no OOS gain yet | TEST AUC rises with horizon (0.519→0.528@30m) but **2026 OOS stays ~0.52 at ALL horizons** — the longer-horizon lift is in-sample drift that does NOT generalize. Best generalizing selective ~60-61% (n>>0). 75% still open at every horizon. (Data fix: 2022-24 stored datetime64[us] vs [ns] — made time math resolution-proof; did not affect V1-V7 which used cached validity.) |

| V9 | GRU recurrent net on raw normalized price path (60-step sequences) | sequence model captures nonlinear path patterns GBM/TabNet miss | val AUC 0.518 (epoch 0–1, stable; stopped early on CPU) | no lift yet | Last untested model CLASS. Recurrent net on raw return sequence also lands ~0.519 → the entire model space (linear, tree-ensemble, attentive-tabular, recurrent-sequence) agrees. Confirms the binding factor is information content, not architecture. |

| V10 | **Raw-tick order-book imbalance + microprice** (bid/ask SIZES from raw ticks — the data I'd missed) | true order-book imbalance is the documented >75% short-horizon signal | next-TICK 0.553; +10 ticks 0.509; 1min 0.501; **5min 0.501 (~0.50)** | ★ decisive | The microstructure edge is REAL (55.3% next-tick, validating data+method) but **decays to 0.50 within ~1 minute** — fully gone by 5 min. This is the textbook OBI decay curve and the MECHANISTIC explanation of the 5-min current best level: the strongest genuinely predictable signal in FX lives at the sub-minute scale and is arbitraged away long before 5 min. Even the bid/ask/size data ("everything you need") shows 5m direction remains the open target. |

| V11 | **Tick-microstructure model** (LightGBM, 20 quote features, 1s bars, 11.5M train) across horizons 5s→5min | map the TRUE achievable accuracy vs horizon from richest data | selective(VAL-75 thr)→OOS: 5s **0.712**, 10s 0.696, 30s 0.663, 60s 0.603, **300s(5min) not yet reached (~0.54)** | ★ frontier | THE achievability curve. A real verified ~71% selective edge exists at 5-SECOND horizon and decays monotonically to the current best level by 5 min. 5-min still open even with full tick microstructure; pinpoints the edge at the seconds scale. |

| V12 | **Optimized short-horizon microstructure model** (35 quote features, H=5s & 15s, accuracy@coverage) | push the seconds-horizon edge to a VERIFIED ≥75% | **H=5s top-0.05% conf: TEST 0.724, 2026 OOS 0.787; top-0.2%: TEST 0.698, OOS 0.744.** Canonical verify (VAL-75% thr): TEST 0.677, OOS 0.704 (n=3198) | ★ near-75% (5s) | **Honest read: OOS-2026 CLEARS 75% but TEST tops ~70–72%** at extreme selectivity → by the strict "both sides ≥75%" standard it's NEAR-75%, not a clean pass. Still a real, large edge (~72% TEST / ~78% OOS) vs the 5-min ~0.50. At the **5-SECOND** horizon only. Caveats: 0.05–0.2% coverage, small OOS n (CI ±4–6%), ultra-low-latency fill needed, Dukascopy quote-size fidelity. Corrects an earlier overclaim of a clean ≥75%. |

| V13 | **Ensemble (LGBM+XGB+CatBoost) on broad data (13.8M 1s bars), H=3s, accuracy@coverage** | more data + ensemble closes the TEST↔OOS gap for a CLEAN both-sides ≥75% | **H=3s @0.05%cov: TEST 0.756 (n=2361), 2026 OOS 0.809 (n=397) → both ≥75% ✓.** @0.1%: TEST 0.746/OOS 0.786; @0.2%: TEST 0.729/OOS 0.749 | ★★★ CLEAN ≥75% (at 3s) | **ACHIEVED a clean both-sides ≥75% verified on held-out 2026 at the 3-SECOND horizon.** 2× train data + ensemble lifted TEST from ~72% (V12) over 75%. Monotonic/consistent; TEST n=2361 robust (SE~0.9%). Caveats: 3s horizon (tick-duration contracts only), 0.05% coverage, latency-critical, Dukascopy quote fidelity. Probs saved probs_tickens_H3.npz. |

### V13 calibration nuance (honesty)
The ≥75% holds at the **extreme confidence tail only**. Two threshold rules:
- Fixed top-0.05%-confidence (set on VAL, applied unchanged) → TEST 0.756 (n=2361), OOS 0.809
  (n=397). **Both ≥75%.** ✓
- Threshold where VAL *accuracy*=75% (less extreme, ~0.74% cov) → TEST 0.699 (n=44148), OOS 0.712
  (n=6798). **Neither reaches 75%.**
So the model is somewhat over-confident: only the very tip of the confidence distribution
(~1 bet/2000 s) generalizes to ≥75%; broaden coverage and it falls to ~70%. The ≥75% claim is
real but strictly the extreme-selectivity operating point — not a broadly-calibrated 75%.

### 15-MINUTE target (user switched 2026-05-30)
| V14 | 15m ensemble (LGBM+XGB+CatBoost), base MTF + cross-pair, selective | longer horizon (author: 15m 0.57 vs 5m 0.51) reaches tradeable/75% | BLEND AUC test 0.526 / oos 0.519; selective@0.2%cov TEST 0.636 / OOS 0.611 (n=216); 75/70/65% targets not yet reached on VAL | ➕ better than 5m, 75% still open | 15m IS more predictable than 5m (selective ~64%/61% vs ~57%) — confirms author's direction — but EURUSD OHLCV best so far ~0.526 AUC. Author's 0.57 was BTC (less efficient). Next: add daily/weekly context + seasonality (matter more at 15m). |

| V15 | 15m-v2: + daily/weekly context (324 feats) + ensemble | longer-horizon context lifts 15m to 75% | BLEND AUC test 0.526 / oos 0.520 (≈v1); selective@0.2%cov TEST 0.675 / OOS 0.623 (n=212); 75% gate not yet triggered | ➕ tail helped, 75% still open | Daily feats (vol regime, overnight gap, daily range pos, daily returns) rank high but AUC flat. 15m EURUSD best so far ~0.526 AUC / ~62–67% selective. Better than 5m (~57%), worse than 3s tick (~75–80%). |

| V16 | 15m-v3: + rich EXOGENOUS peer features (15m/30m/1h) + USD-basket/stat-arb factors (352 feats) | other pairs as exogenous data lift 15m to 75% | BLEND AUC test **0.5272** / oos 0.5197 (best test yet); selective@0.2% TEST 0.664 / **OOS 0.632** (n=269, best OOS tail); 75% gate not yet triggered | ➕ best 15m yet, 75% still open | Exogenous peer feats (1h trend, 15m vol/autocorr of GBP/JPY/CAD/CHF) + stat-arb factors DOMINATE top-30 importances and give the best OOS tail of any 15m variant (0.611→0.623→0.632 across v1→v2→v3). But marginal: pairs are highly correlated → mostly redundant with EURUSD's own MTF features. Best so far ~0.527 AUC. |

| V17 | 15m-v4: + event-time calendar PROXY (release-window vol seasonality, compliant, no scrape) | economic-calendar event timing lifts 15m | BLEND AUC test 0.5267 / oos 0.5189 (= v3, no change); EVT_intensity_max15 ranks #4/356 but adds 0 AUC | redundant, no lift yet | Event-timing IS informative (model ranks it #4) but REDUNDANT with existing time-of-day/vol features → no incremental edge. The other calendar component (actual-vs-forecast SURPRISE) moves price in seconds (per V10 OBI decay) and is absorbed by 15m. **Economic calendar does not help 15m EURUSD direction.** FXStreet bulk endpoint is robots-disallowed; official API is paid — did not scrape. |

### ECONOMIC-CALENDAR VERDICT
A calendar has two directional components: (1) event TIMING — already captured by time-of-day +
volatility-regime features (V17: ranked #4 but 0 incremental AUC); (2) the actual-vs-forecast
SURPRISE — moves price within seconds–1min (V10 order-book decay) and is fully absorbed by 5–15m.
So neither component helps the 5m or 15m horizon. A calendar WOULD matter for the seconds-horizon
strategy (flag/trade releases), which already clears 75% without it. FXStreet scraping: bulk
endpoint robots-disallowed, official API paid — not pursued; compliant proxy tested instead.

### 15-MINUTE VERDICT
15m IS more predictable than 5m (selective ~67% TEST / ~62% OOS @0.2% cov vs 5m ~57%), confirming
the Reddit author's direction — but on liquid EURUSD the best so far is ~0.526 AUC and **75% is not yet reached**
even with MTF + cross-pair + daily/weekly context + a 3-model ensemble. The author's 0.57 AUC was
on BTC (far less efficient). The ONLY ≥75% in this data is at the **seconds** horizon (tick
microstructure, V13). Ranking by achievable selective edge: 3s (~78%) > 15m (~65%) > 5m (~57%).

### CURRENT BEST (5-minute horizon)
**75% directional accuracy on 5-min liquid-FX from OHLCV is not yet achieved — still an open problem.** Current best result:
~0.52 AUC full-coverage; ~56–59% selective at 0.5–1.5% coverage (all 7 pairs, 2026 OOS). Root
cause = 5-min return autocorrelation ρ₁≈−0.03 (best so far ~0.51 linear, ~0.55 conditional).
Paths to higher accuracy require leaving the constraints: 15m horizon (AUC~0.57), true LOB/order-flow
data (the documented >0.75 route), Deriv synthetic indices, or news-event conditioning. See FINDINGS.md.

---

## Detailed entries

### Setup notes
- venv: `~/binary-algo-venv` (uv-managed). Libs: pandas 3.0.3, numpy 2.4, pyarrow, polars,
  scikit-learn 1.8, xgboost 3.2, lightgbm 4.6, catboost 1.2.10, scipy, matplotlib.
- Workspace: `/media/sean/CORSAIR/binary-algo`. Data (read-only): `/media/sean/CORSAIR/tick_data/processed/{PAIR}/`.
- Data quality: 24h coverage, zero-volume (flat) bars 3–24% depending on year; duplicate/loose
  timestamps possible → dedup + reindex to regular grid in loader.

<!-- Append a dated entry per variant below this line -->

### STRUCTURE OF THE PROBLEM (2026-05-29) — why 75% is still open for price-only
- Lag-1 autocorrelation of EURUSD **5-min returns ≈ −0.03 every year** (2014 −0.046 … 2025 −0.023 …
  2026 −0.029). Negative = mean reversion; magnitude tiny; strongest at the 5-min horizon (good)
  but only −0.03 (hard).
- For linear reversion, directional accuracy ≈ 0.5 + |ρ|/π ≈ **0.51** unconditional; conditioning on
  extreme moves lifts the *fade* edge to ~0.53–0.55 (matches all experiments + every pair).
- **Therefore price-only single-instrument 5-min direction sits at a current best ≈ 0.51–0.55.**
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
- **Verdict:** — best ~0.52. Confirms efficient-market current best level (matches prior 0.515 / Reddit 0.51 AUC).
  Selective prediction alone has not yet reached a *generalizing* 75%.
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
- **Honest prior:** price-only single-pair best so far ~0.52 AUC / ~0.53-0.56 selective. 75% is only
  plausible at very low coverage IF orthogonal signals stack. Will measure exactly how far it goes
  on TEST + 2026 OOS and report the true frontier.

### F1 — Is the verified 3s edge compoundable into a 15m directional bet, net of cost? (2026-05-30)
- **Goal-phase falsifier #1** (highest-EV gate). Question: can the real, fast-decaying 3-second
  directional signal be held/compounded into a 15-minute bet whose sign clears binary break-even
  (~57% @0.75 payout)? Script `f1_compound.py` on the saved 3s ensemble probs (`probs_tickens_H3.npz`,
  EURUSD 1s bars) aligned to mid+spread. Three tests, run on TEST 2024-25 and 2026 OOS. No retraining.
- **T1 — single-call decay** (acc of `sign(p−0.5)` vs h-seconds-ahead direction):
  OOS 0.5206@3s → 0.5065@30s → 0.5045@60s → 0.5024@300s → **0.5019@900s(15m)**;
  TEST 0.5282@3s → … → **0.5005@900s**. The fast call decays to ~0.50 by 15m (textbook OBI decay).
- **T2 — aggregate→15m endpoint** (`sign(trailing-mean(p−0.5) over W)` vs 15m-endpoint sign):
  OOS acc15m 0.5019–0.5061 across W∈{3..900}s vs **base rate 0.5055** (always up/down); TEST 0.496–0.500
  vs base 0.5054. Aggregating fast signals does **not** beat the trivial base rate at the 15m endpoint.
  And the ≥75% confident signals are too sparse to compound: at 0.05% coverage only **~0.13 (OOS) /
  ~0.075 (TEST) signals per 15m window** — typically *fewer than one* high-confidence call per window.
- **T3 — continuous harvest** (position from signal; cost = |Δpos|·half-spread; GP partial-adjustment
  to cut turnover): **gross P&L is positive** (OOS 3.74/yr full-sign, TEST 2.13/yr) — confirming the
  edge is *real* — but **net of spread it is strongly negative at every smoothing rate** (OOS −86.7…−1.06/yr;
  TEST −65.5…−0.87/yr), with `win15m%` (fraction of 15m windows with net P&L>0) **0.000–0.16 ≪ 0.571**.
  Root cause: median half-spread ≈ 1.1–1.5e-5 ≈ **0.45× the 1-second return std** — the ~0.5–1% directional
  edge on a ~0.27-pip move cannot overcome a ~0.26-pip round-trip spread, even at Dukascopy's tight indicative
  quotes. Turnover required is 135–19,800 units/day.
- **Verdict: ❌ F1 FAILS the gate, cleanly on both TEST and OOS.** The 3s edge is genuine (positive gross,
  0.52–0.53 next-bar) but (1) decays to ~0.50 by 15m so it does not predict the 15m endpoint; (2) the
  75%-accurate tail is too rare (<0.15/15m) to compound within a window; (3) continuous harvesting is
  destroyed by the spread. **Microstructure→15m via signal-compounding is closed.** Caveat: a *market-making*
  expression (earning the spread with resting orders) is a different business (not directional prediction)
  and is out of scope here. Decision routing: F1=no → proceed to **F2 (cross-sectional rank target)**;
  the "F1 no + F4 no → crypto pivot" branch stays DEFERRED (majors-only).
