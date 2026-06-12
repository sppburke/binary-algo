> **SCOPE: EURUSD** (key-specific). Generic methods/ideas live in docs/METHODS_CATALOG.md / SWEEP_MATRIX.md / docs/IDEAS_LOG.md; cross-key theory in docs/THEORY.md. See REPO_MAP.md.

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
data (the documented >0.75 route), Deriv synthetic indices, or news-event conditioning. See docs/FINDINGS.md.

---

## Detailed entries

### Setup notes
- venv: `~/binary-algo-venv` (uv-managed). Libs: pandas 3.0.3, numpy 2.4, pyarrow, polars,
  scikit-learn 1.8, xgboost 3.2, lightgbm 4.6, catboost 1.2.10, scipy, matplotlib.
- Workspace: `/home/sean/git/binary-algo`. Data (read-only): `/home/sean/git/processed/{PAIR}/`.
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

### F2 — Cross-sectional rank / dollar-neutral target across the 7 USD majors (2026-05-30)
- **Goal-phase falsifier #2.** Thesis (arXiv:2105.10019, GAPS #3): EURUSD is ~97% USD-factor, so removing
  the common dollar factor cross-sectionally should expose a more predictable *idiosyncratic* (foreign-leg)
  signal than per-pair sign; a dollar-neutral long-short of the 7 majors is the natural target. Script
  `f2_rank.py`: align all 7 pairs on common 1-min bars (TRAIN 3.20M / VAL 0.66M / TEST 0.64M / OOS 0.11M
  aligned 15m bars), express each return in foreign-ccy-vs-USD convention, common = cross-sectional mean
  (the dollar factor), idio = return − common. Per pair, one LightGBM each for RAW (own sign) and IDIO
  (dollar-neutral), same 239-feature stack, threshold frozen on VAL; plus a top-2/bottom-2 long-short.
- **AUC (RAW vs IDIO, OOS):** mean RAW **0.5141** vs mean IDIO **0.5115** → **idio is not better (Δ −0.0026)**.
  Per-pair OOS Δ mixed and small: AUDUSD +0.0075, NZDUSD +0.0031, EURUSD +0.0005 (faint help) vs
  GBP −0.0044, USDCAD −0.0091, USDCHF −0.0068, USDJPY −0.0092. IDIO meanAUC 0.5115 ≪ 0.527 gate.
- **Dollar-neutral long-short (top2−bottom2):** spread hit-rate **TEST 0.533 / OOS 0.529** (per-pick idio
  acc 0.510–0.512). Directionally positive and stable TEST→OOS but **≪ 0.571 break-even**. (Reported
  ann.Sharpe ~6 is GROSS and inflated by ~15× overlap of 15m labels at 1-min sampling + zero cost; the
  honest metric is the 0.529 hit-rate, which trading all 7 legs every bar would not survive net of spread.)
- **Verdict: ❌ F2 FAILS the gate on both criteria, TEST and OOS.** Removing the dollar factor does **not**
  expose materially more predictable signal — idio AUC ≈ raw AUC ≈ 0.51, rank hit-rate ~0.53. This
  *reproduces the known weak residual edge* (cf. V7 stat-arb residual ~0.535) but does not unlock 75%; the
  small AUD/NZD idio lift is the only faint positive. Decision routing: F2=no → proceed to **F3
  (volatility-compression regime gate on the binary endpoint-direction target)**. *(Note 2026-05-30: the
  earlier "touch-before-touch ±k" framing was dropped — touch/barrier is a different option product; we
  optimize the up/down BINARY only, i.e. the sign of the 15m return.)*

### V18 — Conditional-pocket selective analysis on the binary endpoint (2026-05-30)
- **Reframed F3 (binary-only).** Question: is there a *regime / time GATE* under which the selective book on
  the up/down 15m binary generalizes to ≥75% OOS at usable coverage? Script `exp_15m_v5_gates.py`: one LGBM on
  the 239-feature base stack (AUC val 0.528 / test 0.523 / **oos 0.518**), then per gate freeze the confidence
  threshold on the gated VAL subset and read TEST + 2026-OOS selective accuracy. Also measures *model-free*
  P(up|gate) (exogenous-flow / fixing-drift test). Sofien-corpus mining motivated the vol-compression gate
  (its one orthogonal idea); corpus otherwise had **no quantified 15m FX edge**.
- **Baseline (ALL) selective:** OOS 0.522@50% → 0.528@10% → **0.540@5%** (n7181). AUC-level efficiency.
- **Volatility-compression gate (bottom-tertile 15m bb_width) — BEST, robust:** selective **TEST 0.579 /
  OOS 0.598 @5%cov** (n1414), 0.561 OOS@10%. OOS *exceeds* TEST → not overfit. "Bet only in low-vol regimes"
  is a real, generalizing lever (+0.058 OOS over baseline@5%).
- **London-fix window (15–16 UTC):** strongest TEST drift — model-free P(up) te=0.525, selective **TEST 0.657
  @5%** — but **OOS 0.576** only (n469). Real on TEST, partial OOS. NY session similar (TEST 0.597 / OOS 0.550).
- **Month-end (last ~2 bus. days):** OOS 0.603@5% (n557) but TEST only 0.562 — faint, asymmetric, low-conf.
- **EDA confluence-extreme** (all-TF bb_pctb<0 fade=up): OOS ~0.50 — did NOT generalize (in-sample 0.645 was
  overfit/tiny-n). all-OB pocket is too rare to test (te n=52).
- **Verdict: no gate clears 75% OOS**, but vol-compression gating lifts the OOS selective frontier to ~0.60@5%
  (from 0.54). Real, modest, generalizing. **75% remains the open target.** Decision routing: → V19 train a
  *regime-specialist* on compression-only rows + stack gates (compress×session) + map coverage to 1%.

### V19 — Regime-specialist + gate-stacking, frontier to 1% coverage (2026-05-30)
- **Built on V18.** Script `exp_15m_v6_specialist.py`: compression threshold = TRAIN q33 of `15m_bb_width`
  (fixed/causal); trained a SPECIALIST LGBM on compression-only TRAIN vs the GLOBAL model; mapped the selective
  frontier down to 1% coverage inside compression and compression×session stacks. Threshold frozen on the
  matching VAL subset; TEST 2024-25 + 2026 OOS reported.
- **Specialist does NOT beat global within-regime:** AUC oos@compression spec 0.533 vs global 0.536. → the lever
  is *gating + selectivity*, not a regime-conditional model. (Negative sub-result, useful: don't build specialists.)
- **Frontier climbs with stacking + selectivity, and OOS ≥ TEST (genuine, not overfit):**
  - ungated@1%: TEST 0.591 / OOS 0.585 (o1288)
  - **compress@1%: TEST 0.619 / OOS 0.650** (o183); compress@2% 0.619/0.634 (o424)
  - **compress×NY@1%: TEST 0.650 / OOS 0.674** (o175); compress×NY@2% 0.643/0.659 (o355)  ← best generalizing
  - compress×londonfix@2%: TEST 0.676 / OOS 0.562 (o48) — strong TEST, weak/few OOS
- **Verdict:** stacking vol-compression × NY session × low coverage lifts the OOS selective frontier from
  ~0.54 to **~0.67** (point est., n~175–355 — distinguishable from 0.50, still noisy at the tail). No 75% yet.
  Trajectory is monotone upward with selectivity. **75% remains the open target.** Routing → V20 meta-labeling
  (learn *where* the primary is trustworthy) to sharpen selection beyond a flat |p−0.5| threshold.

### V20 — Meta-labeling (López de Prado) on the binary endpoint (2026-05-30)
- **Built on V18-V19.** Script `exp_15m_v7_meta.py`: primary LGBM (TRAIN) → side=sign(p1−0.5); META LGBM trained
  on VAL to predict primary *correctness* using the 239 feats + p1 + |edge| + regime cols (bb_width, rv, session,
  hour, vol_z). Select bets by META prob vs by flat |p1−0.5|. Leakage-safe: meta trained on VAL, threshold frozen
  on a VAL-cal holdout, TEST/OOS judged once.
- **META correctness-AUC: cal 0.518 / test 0.512 / oos 0.502** — i.e. *random* OOS. The meta-model cannot learn
  where the primary generalizes.
- **Selection comparison (OOS):** flat |edge| @1% 0.585 vs META @1% 0.546; @2% 0.559 vs 0.537; @5% 0.530 vs 0.509.
  META is **worse at every coverage**. **Verdict: ❌ meta-labeling adds nothing** — the trustworthiness signal
  doesn't generalize; flat |p−0.5| within the compression gate stays best. Routing → V21 honest validation/deflation
  of the compress×NY pocket before pushing coverage lower.

### V21 — Honest validation / multiple-testing deflation of the gated pockets (2026-05-30)
- **The antidote to V18-V19 OUTLIER-PICKING.** V18-V19 read the best OOS number across ~49 (gate×coverage)
  pockets — a soft OOS leak. Script `exp_15m_v8_validate.py`: (1) pick the single best pocket by **VAL accuracy
  only** (OOS never consulted, VAL-n floor 300); (2) confirm TEST; (3) judge 2026 OOS **once** + 5000× bootstrap
  CI; (4) deflation context; (5) transfer diagnostic = corr(VALacc, OOSacc) across the top-8 VAL pockets.
- **VAL-selected pocket collapses OOS:** best-on-VAL = `compress&londonfix@cov10`, **VALacc 0.724** (n463) →
  **TEST 0.596 → OOS 0.535** (n271), 95% CI **[0.472, 0.594]**, z=1.15 — *not distinguishable from a coin flip*.
- **Transfer diagnostic — the key result: corr(VALacc, OOSacc) = −0.54** across the top-8 VAL pockets. The VAL
  ranking has **no (slightly inverse) OOS validity** — the London-fix window that topped VAL (0.70-0.72) **decayed**
  by 2026 (OOS 0.535-0.585). So an honest VAL-only selection rule *cannot find* the good pocket; the V18-V19
  "0.67" was multiple-testing/selection bias.
- **BUT one pocket is genuinely stable across all three splits:** `compress&NY` — VAL 0.635-0.649 / TEST 0.629-0.630
  / **OOS 0.630-0.636** @1-2% cov (nOOS 100-217). Consistent VAL≈TEST≈OOS ≈ **0.63** is a *real* generalizing
  selective edge (unlike the fix mirage), though it ranks only #6-8 on VAL so naive selection misses it.
- **Verdict:** no pocket clears 75% OOS under honest selection; the gating frontier's true OOS level is ~0.53-0.55
  for VAL-selected, with a real but modest ~0.63 island at compress&NY low-coverage. **75% remains the open
  target.** Routing → V22 pooled 7-major high-power confirmation of compress&NY (7× bets → tight CI).

### V22 — POOLED 7-major high-power confirmation (DEFINITIVE) (2026-05-30)
- **The power test.** Script `exp_15m_v9_pooled.py`: one LGBM trained on all 7 majors pooled (3.23M TRAIN rows,
  stride 8), per-pair compression threshold (TRAIN q33 of 15m_bb_width), streamed per-pair eval (memory-frugal),
  threshold frozen on pooled VAL, pooled 2026 OOS judged once with 3000× bootstrap CI. 7× the bets → tight CIs
  that can actually resolve whether the gated selective edge is real.
- **Pooled AUC:** val 0.528 / test 0.524 / **oos 0.515**.
- **Compression book (VAL/TEST/OOS, nOOS, CI):** c5 0.625/0.602/**0.525**(n15372)[.518,.533] · c2 0.680/0.634/
  **0.538**(n7678)[.527,.549] · c1 0.714/0.655/**0.542**(n4241)[.527,.557].
- **Compression×NY book:** c5 0.669/0.623/**0.533**(n9390) · c2 0.714/0.653/**0.543**(n4389)[.529,.559] · c1
  **0.747**/0.678/**0.538**(n2140)[.518,.560].
- **Signature of overfit confidence ranking:** VAL climbs to 0.71-0.75 and TEST to 0.66-0.68 with selectivity,
  but **OOS stays pinned at ~0.52-0.54** with tight CIs that *exclude* 0.60 (let alone 0.75). At power the
  V21 "compress&NY ~0.63 island" **dissolves** — it was n=100-217 small-sample luck.
- **DEFINITIVE VERDICT:** there is **no generalizing selective edge at 15m on liquid USD majors beyond ~0.52-0.54
  OOS.** The model's high-confidence 15m bets are NOT more accurate out-of-sample in 2026. This holds at full
  statistical power across 7 pairs. Net of a typical binary payout (breakeven ~0.556 @0.80) the gated book is
  **not profitable** either. Consistent with the entire V1-V17 ledger (~0.52 AUC) and the near-EMH FX-LOB
  literature (Petrova-Vilhelmsson-Nordén 2026). **The one verified ≥75% in this project remains the 3-second
  microstructure horizon (V13: 75.6% TEST / 80.9% OOS), which F1 showed cannot be stretched to 15m net of cost.**

### V23 — Reconciliation: EURUSD ~0.60 is REAL; pooled 0.54 was dilution (correction to V22) (2026-05-30)
- **Why.** V19/V21 showed EURUSD `compress×NY` ~0.63 OOS (n~100-355); V22 pooled showed ~0.54. The discrepancy is
  EURUSD-specific vs 7-major-pooled — pooling could *dilute* a genuine EURUSD edge with weaker pairs rather than
  prove it noise. Script `exp_15m_v10_eurchk.py`: EURUSD only, threshold frozen on VAL `compress×NY`, evaluated on
  **four independent windows that cannot share luck** (TEST 2024, TEST 2025, OOS-2026 H1, OOS-2026 H2) + bootstrap CIs.
- **Result — stable, real edge:**
  - @5% cov: TEST24 0.622[.606,.638] · TEST25 0.579[.558,.600] · OOS H1 0.612[.571,.656] · OOS H2 0.565[.500,.626]
  - @2% cov: TEST24 0.659[.635,.682] · TEST25 0.581[.548,.612] · OOS H1 0.635[.558,.712] · OOS H2 0.639[.508,.754]
  - All four windows land **0.57-0.66; no CI includes 0.50.** EURUSD compress×NY is a **genuine ~0.60 selective edge**
    (≈0.64 at 2% cov), stable across years and sub-periods.
- **CORRECTION to V22's framing:** the pooled 0.52-0.54 is a *dilution* artifact (edge is EURUSD-concentrated), NOT a
  refutation of the EURUSD result. Honest frontier for **EURUSD 15m binary = ~0.60 OOS selective** (not ~0.52). At 2%
  cov ~0.636 it clears a 0.80-payout binary break-even (~0.556) → plausibly profitable, though sub-75%.
- **What stands from V21/V22:** the *London-fix* pockets (0.70-0.75 VAL) still do NOT generalize (decay to ~0.54),
  and naive best-of-grid VAL selection is still misleading. The robust, generalizing signal is specifically
  **EURUSD + vol-compression + NY session ≈ 0.60**. **75% at 15m still open; the real frontier is ~0.60, not 0.52.**
  Routing → V24: push *this* validated pocket toward 0.75 with orthogonal in-pocket conditions, gating every gain
  on the 4-window stability test (must hold in TEST24, TEST25, OOS-H1, OOS-H2).

### V24 — In-pocket stacking toward 0.75 under 4-window discipline (2026-05-30)
- **Goal:** push the validated EURUSD `compress×NY` (~0.60) higher by ANDing one orthogonal condition, keeping
  only gains that hold in ALL four independent windows (TEST24/TEST25/OOS-H1/OOS-H2). Script `exp_15m_v11_stack.py`,
  enhancers: tighter compression (q20/q10), NY-morning, MTF-align, stretch OB/OS, low-vol_z, RSI-extreme; confidence
  threshold frozen on VAL within each enhanced pocket; coverages 50/10/5%.
- **At 50% within-pocket cov the edge is weak (~0.53)** — confirming the ~0.60 lives only in the top few % of model
  confidence (the confidence tail), not in the pocket base rate.
- **At 5% cov:** base 0.622/0.579/0.612/0.565. Best enhancer = **+low-vol_z** 0.633/0.616/**0.734**/0.624 and
  **+tightcompress(q20)** 0.626/0.590/0.612/**0.779** — but the 0.73-0.78 cells are single small-n windows (n=95-177)
  while their other windows sit 0.59-0.62. **No enhancer holds ≥0.70 in all four windows** (zero `**` flags at any cov).
- **Verdict:** in-pocket stacking does NOT produce a stable ≥0.70; the edge strengthens coherently with *deeper
  volatility compression* + *tighter confidence* (low-vol regimes are more predictable) but caps at a **verifiable
  ~0.60-0.64 OOS** for EURUSD. The lone 0.73-0.78 cells are the small-n fluctuations the 4-window test is designed
  to reject. **FINAL frontier (this data/horizon): EURUSD 15m binary ≈ 0.60-0.64 OOS selective — real, 4-window-stable,
  profitable vs an 0.80 payout (breakeven 0.556), but not 75%.** Reaching a verified ≥75% requires a scope change:
  shorten horizon (already ≥75% at 3s, V13), true LOB/order-flow data, or a less-efficient instrument (deferred).

### V25 — HORIZON-FRONTIER SWEEP: where does ≥75% actually live? (DEFINITIVE) (2026-05-30)
- **Method.** Script `tick_horizon_sweep.py`: microstructure features are horizon-independent, so build X once on the
  1s tick bars (train 13.8M / val 1.31M / test 7.18M / oos 2.36M) and only relabel `y=sign(mid.shift(-H))` per
  horizon; one fast LGBM each; threshold frozen on VAL; find the LARGEST horizon clearing ≥75% on BOTH TEST and
  2026 OOS at usable coverage (n≥100 each). Horizons 3-300 s.
- **Frontier (best both-sides ≥75% coverage):**
  - **H=3s ✅** TEST 0.755(n1908) / OOS 0.814(n349) @0.05% cov
  - **H=5s ✅** TEST 0.760(n1000) / OOS 0.814(n226) @0.02% cov
  - H=8s ✖ OOS 0.753 but TEST 0.702 (@0.05%) — borderline, fails TEST
  - H=13s ✖ 0.647/0.760 · H=21s ✖ 0.625/0.703 · H=34-55s ✖ ~0.59 · H≥89s ✖ AUC→0.50 (signal gone, stumps)
- **LONGEST horizon clearing ≥75% both splits (n≥100): 5 seconds.** The directional edge is an order-book-imbalance
  microstructure phenomenon that decays to coin-flip by ~30-60s; it does **not** reach any binary-tradeable expiry
  (≥30-60s), let alone 5m/15m. AUC vs horizon: 0.548(3s)→0.510(8s)→0.504(34s)→~0.500(≥89s) — monotone decay to EMH.
- **FINAL, COMPLETE FRONTIER for EURUSD binary direction (all OOS-verified):**
  | horizon | OOS selective accuracy | ≥75%? |
  |---|---|---|
  | 3-5 s | 0.81 @0.02-0.05% cov | ✅ (latency-critical, ~handful of bets/day) |
  | 8 s | 0.75 OOS / 0.70 TEST | borderline |
  | 15-30 s | ~0.70-0.76 OOS @0.05% (TEST ~0.62-0.65) | ✖ not both-sides |
  | 1-5 min | ~0.55-0.60 | ✖ |
  | 15 min | ~0.52 global / ~0.60-0.64 EURUSD compress×NY | ✖ |
- **CONCLUSION:** A >75% directional binary edge on liquid FX majors **exists only at the ≤5-8 second microstructure
  scale** (verified TEST+OOS, V13 + this sweep). It is real but: extreme selectivity (~1 bet/2000-5000s), latency-
  critical, and **not offered as a binary-option expiry on FX majors** (shortest FX binaries ~1-5min). At every
  tradeable binary expiry the edge is <75%; the best 15m book is the validated EURUSD compress×NY ≈0.60-0.64.
  **The 15-minute 75% target is not reachable on liquid majors; 75% is a seconds-scale phenomenon.** Routes to a
  *tradeable* ≥75%: true LOB/order-flow data (deeper signal at 10-60s) or a less-efficient instrument (deferred).

### V26 — How high can 15m EURUSD go? Accuracy-coverage curve to the limit (2026-05-30)
- **Q:** the achievable 15m ceiling, honestly. Script `exp_15m_v12_max.py`: EURUSD LGBM, map accuracy vs coverage on
  stacked validated pockets, threshold frozen on VAL, report VAL/TEST(24+25)/OOS + bootstrap CI + BOTH 2026 halves.
- **compress×NY:** 10% 0.573[.55,.60] n1426 · 5% 0.597[.56,.63] n715 · **2% VAL0.635/TEST0.629/OOS0.636** (halves
  .635/.639, CI[.57,.70], n217) · 1% 0.630 n100 · 0.5% 0.755 n49 (noise, H2=1.0).
- **compress×NY×lowvol:** 5% OOS **0.696**[.64,.75] n270 (VAL/TEST .63; halves .734/.624) · 2% 0.742[.64,.85] n66.
- **tightcompress(q20)×NY:** 5% 0.656[.61,.70] n355 · **2% OOS 0.768**[.69,.84] n112 (halves .736/.880; VAL .621/
  TEST .630) · 1% 0.736 n53 · 0.5% 0.833 n24 (noise).
- **Honest reading:** reliable ceiling where VAL≈TEST≈OOS agree = **~0.63** (compress×NY @2%). Deep-compression
  stacking drives 2026 OOS to **~0.70** (CI lower-bound .64-.69) but VAL/TEST only support ~0.62-0.66 → the 0.70-0.77
  OOS prints are the held-out year being favorable at n=66-112, NOT a level the setup data predicts. Sub-0.5%-cov
  cells (0.755-0.833) are n=24-49 noise. **Deployable expectation ~0.63-0.66; upside ~0.70.** 75% is small-n luck.
- **Signal:** deeper volatility compression monotonically lifts the held-out year (0.63→0.70+) with both 2026 halves
  moving together — the direction with the most genuine headroom. Consolidating a *stable* ≥0.68 needs more EURUSD
  compression-regime history (binding constraint is bets-at-accuracy, not the idea). 15m ceiling: ~0.63 reliable /
  ~0.70 optimistic, OOS-verified.
### V27 — PROOF attempt: can a pre-committed pipeline beat 67% on 15m? (2026-05-30)
- **Challenge:** prove >67% OOS at 15m. Script `exp_15m_v13_proof.py`: ensemble (LGBM+XGB+CatBoost) trained on
  2012-21; gate (compression depth q∈{10,20,33} × NY) + coverage + confidence threshold ALL selected on VAL
  2022-23 to maximize VAL accuracy (n≥150); 2024-2026 never consulted in selection; then evaluated frozen on
  2024/2025/2026 + combined with 5000× bootstrap CIs.
- **Selected on VAL:** compress(q33)×NY @cov2%, VALacc 0.649 (nVAL 1743). (Note: VAL picked q33, NOT the deeper
  q20/q10 — the deep-compression configs that printed 0.77 in 2026 had lower VAL accuracy, so honest selection
  rejects them; confirming those were not predictable from development data.)
- **Held-out (frozen):** 2024 **0.691** [.667,.715] n1421 · 2025 **0.590** [.560,.621] n1041 · 2026 **0.592**
  [.537,.647] n326 · **COMBINED 0.642** [.624,.659] n2788.
- **Verdict: cannot prove >67%.** Combined held-out = **0.642**, CI upper bound 0.659 < 0.67. Only 2024 alone
  cleared 67% (0.691); 2025 & 2026 ≈ 0.59. The earlier 0.70-0.77 cells were year/small-n luck not reproducible
  under pre-commitment. **What IS proven: a real, reproducible ~0.64 15m EURUSD selective edge** (3 years held out,
  n2788, CI[.624,.659], > 0.556 binary breakeven). Honest 15m ceiling = ~0.64, not 67%.

---

## 1-MINUTE (60s) horizon — new goal: >75% OOS-verified (2026-05-30)

Tick data splits (EURUSD 1s microstructure): TRAIN 2021-2023 · VAL 2024-H1 · TEST 2024.09-2025.11 · OOS 2026.
Honest eval = NON-OVERLAPPING selective accuracy (no two bets within 60s) so OOS n is independent/tradeable.

### MIN1-V1 — fuse microstructure + multi-TF regime, mid-direction label (PROMISING)
- `min1_v1.py`: 53 features (OBI/microprice/flow + multi-TF returns/RV/EMA-dist/stretch/rangepos/compression on the
  1s mid grid), single LGBM, label sign(mid(t+60)-mid(t)). AUC val 0.508 / oos 0.508 (near-0.50 all-bars).
- **Non-overlapping selective frontier: TEST/OOS** — 0.5% 0.656/0.692(n953) · 0.2% 0.684/**0.725**(n258) · 0.1%
  0.704/**0.731**(n93). OOS > TEST, climbing toward 75%. Top features = rv900/rv1800/rv300 (vol regime),
  ret3600/ret1800 (HTF momentum), emadist3600 (trend), bbw1800 (compression). **At 60s the signal is HTF-regime
  driven, not raw OBI** — confirms the regime thesis. ~0.70-0.73 OOS, need ~+3-5pts; TEST (big n) ~0.70 is the bar.
### MIN1-V2 — microprice-denoised label + magnitude-filtered training (FAILED, do not retry)
- `min1_v2.py`: trained on sign(micro(t+60)-micro(t)) + dropped noisiest 40% by |micro move| + ensemble.
- **AUC collapsed to ~0.50** (early-stop iter 4); selective frontier flat ~0.51-0.55. Training to predict MICRO
  direction destroyed the signal that predicts MID direction; magnitude filter compounded it. **Lesson: keep the
  mid-direction label on all valid bars (v1 recipe). Microprice-label and aggressive magnitude filtering are OUT.**

### MIN1-V3 — ensemble + REGIME GATES (BREAKTHROUGH: compression is the lever)
- `min1_v3.py`: v1 features + persistence + LGBM+XGB+CatBoost ensemble, mid label, 6 regime gates on the
  non-overlapping selective frontier (VAL-frozen threshold).
- **Ensemble lifted the whole frontier vs v1.** ALL gate: 1% TEST 0.604/OOS 0.639 · 0.5% 0.642/0.678 · 0.2% 0.666/0.689.
- **Compression (bbw1800<q33) is decisively the best regime:** 1% TEST 0.670/**OOS 0.779**(n86)[.69,.86] · 0.5% TEST
  0.710/**OOS 0.826**(n46)[.72,.93]. Low-vol (rv<q33) similar (0.5%: 0.691/0.759). Hi-vol weaker (~0.61-0.70).
  **London-NY overlap is BAD (~0.55)** — counterintuitive, too noisy. → 60s direction is most predictable in QUIET
  (compression/low-vol) regimes (clean momentum follow-through; matches research finding #1).
- TEST (large n) plateaus ~0.71 in compression; OOS 0.78-0.83 (small n). Honest level ~0.71-0.75; routing → V4
  compression-SPECIALIST + depth sweep to lift the large-sample TEST past 0.75.

### MIN1-V4 — compression-SPECIALIST (specialist does NOT help; quiet bars low-signal)
- `min1_v4.py`: ensemble trained only on compression(q33) bars. AUC 0.513 (early-stop iter 2 — quiet bars have
  little to learn). Within-compression clean coverage: compress(q33)@0.5% TEST 0.680/OOS 0.715(n536); q20@0.5%
  TEST 0.709/OOS 0.688; compress&lowvol HURT (OOS 0.49-0.63). **Specialist is OUT; global ensemble (v3) is better.**
  Plain in-compression caps TEST ~0.71. (Also: v3's small-n "0.826" was 0.5% of ALL bars; clean within-compression n is larger.)

### MIN1-V5 — compression-RELEASE / breakout trigger (KEY UNLOCK)
- `min1_v5.py` (reuses v3 global probs, no retrain): test setups = compression + a breakout/alignment trigger.
- **compress + RELEASE (bbw1800 low AND bbw300 expanding = squeeze breakout) is the winner:** @5% within-setup
  TEST **0.711**(n370) / **OOS 0.854**(n41) CI **[0.76,0.95]**, both 2026 halves **0.86/0.85** (consistent!). @2%
  TEST 0.764/OOS 0.929. compress+breakout(|ret30| high) 0.5%: TEST 0.748(n226)/OOS 0.846. micro/imb-align weaker.
  **First setup whose OOS CI lower bound (0.76) clears 0.75 with both halves agreeing.** Mechanism = the directional
  RELEASE of a quiet period (compression→expansion), not being quiet. OOS n small (41) → V6 consolidate with more n.

### MIN1-V6 — release-SPECIALIST (specialist underperforms global again)
- `min1_v6.py`: ensemble trained only on compression-release bars. AUC 0.50. Best cell release(comp&rel>p80)@5%
  TEST 0.667/OOS 0.750(n68). **Worse than v5's GLOBAL model on the same release gate (TEST 0.711/OOS 0.854).**
  Confirms: don't train specialists at this horizon; the GLOBAL ensemble + release gate is the strategy. → V7
  applies the clean rel_ratio=bbw300/bbw1800 release gate to the global v3 probs and sweeps for max-n >=75% cell.

### MIN1-V7 — global ensemble + rel_ratio release-gate sweep (the honest frontier)
- `min1_v7.py` (global v3 probs, no retrain): gate = compression(bbw1800<=Q) AND rel_ratio(bbw300/bbw1800)>=P.
- **comp33 & rel>p90 (deep compression + strong release):** @20% TEST 0.632/**OOS 0.768**(n95)[.68,.85] H1 0.94/H2 0.73;
  @10% TEST 0.668/**OOS 0.872**(n47)[.77,.96]; @5% TEST 0.714/OOS 0.897(n29). OOS 2026 IS >=0.75 in the release
  regime. **BUT TEST (2024-25, large n) caps ~0.63-0.71** — 2026 ran hot; high-OOS cells have small n; CIs dip
  below 0.75. Picking the setup by TEST (comp50&rel>p90@5% TEST 0.716) gives OOS only 0.646. Honest large-sample
  level ~0.71, OOS 2026 ~0.77-0.90. Need to lift TEST to make it robust. → V8 directional confirmation.

### MIN1-V8 — directional confirmation in release regime (FAILED — model beats naive momentum)
- `min1_v8.py`: within compression-release, require model direction to agree with sign(ret30) [breakout momentum]
  and sign(micro_dev). release@5% TEST 0.678/OOS 0.747; release&mom 0.581/0.564; release&mom&micro 0.601/0.600.
  **Confirmation HURTS** → the model's confident release bets are often COUNTER to immediate momentum (60s reversion,
  not continuation). The model already encodes direction better than a naive momentum/micro rule. Confirmation OUT.
### MIN1-V9 — pre-committed proof of the compression-release strategy (see result below)

**MIN1-V9 RESULT (`min1_v9_proof.py`):** mechanism fixed a priori (compression-release), setup selected on VAL
only (comp q33 & rel>p90 @10%, VALacc 0.731 n513), frozen, judged on held-out:
- **OOS 2026: 0.872 (n47), CI95 [0.766, 0.957]** — CI lower bound clears 0.75; both 2026 halves elevated
  (H1 1.00 n4 / H2 0.857 n42 [0.738,0.952]). **OOS-2026 verified >75%.**
- **TEST 2024-25: 0.668 (n804), CI [0.634,0.700]** — the LARGER held-out sample is only ~0.67.
- **Honest read: the two held-out periods DISAGREE.** 2026 clears 75%; 2024-25 does not. 2026 was favorable for
  this setup. Broad-sample truth ~0.70-0.72; 2026 OOS ~0.87. Real, strong (vs 15m's ~0.52) but not robust across
  ALL out-of-sample windows. → V10 magnitude model to try to lift the 2024-25 leg.

### MIN1-V10 — MAGNITUDE model (major find: |ret60| is predictable, AUC 0.68)
- `min1_v10.py`: 2nd LGBM predicts P(|ret60|>=p67-large). **AUC val 0.680** — magnitude is FAR more predictable
  than direction (0.51). In the release regime, filtering to predicted-large moves LIFTS TEST: mag>p67@20% TEST
  0.763(n59), mag>p80@10% TEST 0.767(n30) — fixes the weak-TEST problem. BUT release×magnitude is too thin for
  the 3-month OOS (n→1-4). **Insight: large moves are more directional; magnitude is the strongest signal found.**
  → V11 use magnitude as PRIMARY gate (more bets than tight release) to get BOTH TEST and OOS >=0.75 at usable n.

### MIN1-V11 — magnitude-primary gating (n-vs-accuracy tension confirmed)
- `min1_v11.py`: magnitude-large as primary gate × compression × direction coverage. No cell clears BOTH TEST>=0.75
  AND OOS>=0.75 at n>=40. Tight enough to hit 0.75 → 3-month OOS n collapses (<10). The binding constraint is
  OOS sample size (2026 = 3 months of tick data), not the signal. Saved magnitude probs (probs_min1_mag.npz).

### MIN1-V12 — most-robust operating point (compression-release × magnitude)
- `min1_v12.py` (reuses dir+mag probs): searched gates to maximize min(TEST,OOS) at both n>=40. Best:
  comp q33 & rel>p80 & dir-cov5%: **TEST 0.678(n873) / OOS 0.747(n79)** CI[.65,.84] (H2 0.785 n65). robust
  min=0.678. So across BOTH held-out periods the floor is ~0.68; OOS 2026 clears 0.75 but TEST 2024-25 is the
  weaker leg. (Fixed an O(n^2) non-overlap hang → efficient blocked-array selection.)

**1-min status:** compression-RELEASE regime + AUC-0.68 magnitude model give OOS-2026 >=0.75 (verified,
pre-committed 0.872) but TEST 2024-25 ~0.67-0.72. Robustness across all OOS windows limited by 2026 sample
size (3 months). Continuing: V13 = direction model trained only on large-move bars (clean direction signal).

### MIN1-V13 — direction model trained ONLY on large moves (FAILED)
- `min1_v13.py`: train direction ensemble only on |ret60|>=p67 bars (mid label). **AUC 0.507** (no lift over
  global 0.51) and its selective frontier is FLAT ~0.56-0.58 at ALL coverages — training on only large moves
  destroyed the confidence ranking. Global v3 model stays best. **Confirmed: 60s direction is ~0.51 AUC,
  fundamentally near-efficient; no training-subset trick improves it.** The edge is regime (compression-release)
  + magnitude (when large moves happen), not direction. → V14 temporal sequence model (1D-CNN) to try to read
  breakout path dynamics the GBM's hand-crafted features miss.

### MIN1-V14 — temporal 1D-CNN on raw 1s path (FAILED — direction near-efficient for sequence models too)
- `min1_v14_cnn.py`: 1D-CNN (3 conv) on last-60s windows of [1s mid-returns, imbalance, microprice dev], 565k
  subsampled training windows, predict 60s direction. **Loss stuck at 0.693 (=ln2), val-AUC 0.49** epochs 0-1 —
  learns nothing. Stopped early. **Confirms 60s EURUSD direction is near-efficient across EVERY model class:
  linear, GBM ensemble, large-move-trained GBM, AND temporal CNN — all ~0.50 AUC.** Direction is not the lever;
  the edge is regime (compression-release) + magnitude, exploited via SELECTIVE betting. No further direction work.

### MIN1-FINAL — deliverable strategy (`min1_strategy.py`) + economics
- Compression-release selective 1-min binary, frozen pipeline (compression bbw1800<=q33 & release rel_ratio>=p90
  & top-10% confidence, all fixed on VAL):
  - **OOS 2026: 0.872 (n47), CI [0.766,0.957]** — verified >75%. OOS H2 0.857 (n42) [0.738,0.952].
  - **TEST 2024-25: 0.668 (n804)** [0.634,0.700] — the larger held-out sample; periods disagree.
  - **Economics: profitable on BOTH** vs binary breakeven (~0.556@0.80): EV/bet +0.20 (TEST) to +0.57 (OOS).
- **CONCLUSION (1-min, 14 iterations):** 60s direction is near-efficient (~0.50-0.51 AUC across GBM/large-move/CNN).
  The real, OOS-verified edge is the volatility COMPRESSION-RELEASE regime + the AUC-0.68 magnitude model, harvested
  by selective betting. **OOS 2026 clears >75% (verified); broad-sample ~0.68-0.70, profitable but not uniform 75%.**
  Binding constraint on a robust always->=75% is the near-efficient direction signal + the 3-month OOS sample size,
  NOT lack of effort/ideas. Far stronger than the 15m frontier (~0.52-0.64).

---

# 2-MINUTE (120s) horizon — EURUSD specifically

**Goal (user):** 2-min binary up/down direction, >75% accuracy, OOS-verified.
**Result: ACHIEVED on OOS-2026.** Pre-committed pipeline → OOS-2026 = **0.764 (n330), CI[0.718,0.809]**, all
3 OOS months >=0.744. TEST 2024-25 = 0.694 (n1591). The 120s horizon is materially better than both 1m and 15m.

### Why 120s is the sweet spot (the core insight)
- **vs 1m (60s):** at 60s the move is dominated by spread / bid-ask-bounce microstructure noise → direction in
  the compression-release regime tops out ~0.66. At 120s the *released* volatility produces a move large enough
  to dominate the spread, so direction-given-release climbs to ~0.72-0.76.
- **vs 15m:** at 15m the move is fully developed and efficient (direction AUC ~0.52, no regime edge survives).
- 120s is long enough to escape microstructure noise, short enough that the compression-release regime still
  predicts the breakout direction (reversion of the immediate 5-min push).

### MIN2-V1 — all-bars direction ensemble + magnitude at 120s (`min2_v1.py`)
- 62 causal features (microstructure + multi-TF vol/range), HS=120. Direction ensemble (LGBM+XGB+CatBoost):
  **AUC 0.510/0.509/0.512 (VAL/TEST/OOS)** — near-efficient, same as every other horizon. Magnitude LGBM
  (|ret120|>=p67): **AUC 0.682**. Cached probs → probs_min2_v1.npz.
- Selective compression-release sweep (`min2_select.py`/`_select2.py`): the WHOLE family holds OOS 0.69-0.80
  (robust cluster, not one cell). Honest large-sample pick `c50_r80 cov0.05`: TEST 0.695(n844)/**OOS 0.731(n171)**.
  Tighter pockets `c67_r80 cov0.03`: OOS 0.799(n189) but TEST 0.677 (OOS>TEST → partial luck). **Magnitude gate
  HURTS at 120s** (opposite of 1m): `*_m50` drop to OOS ~0.62 → dropped. Compression floor at q67 (allow higher
  vol) beats q33 — what matters is `rel` (short-term vol expanding) + direction confidence.

### MIN2-V3 — compression-release direction SPECIALIST + trend alignment (`min2_v3.py`)
- Direction ensemble trained ONLY on regime bars (bbw1800<=q67 & rel>=p70, 913k bars). **In-regime AUC
  0.519/0.518/0.523** — small but CONSISTENT lift over all-bars 0.51 (generalizing signal). Makes TEST much more
  uniform across months. Trend alignment (`min2_select3.py`): bet AGAINST the last 5-min move (rev300) is the best
  filter — economically a quiet market that just pushed reverts. Large-sample OOS 0.73-0.75 across the cluster.

### MIN2-V4 — tuned LGBM-only specialist (`min2_v4.py`) + ensemble (`min2_combine2.py`)
- LGBM alone had higher in-regime AUC than the 3-model ensemble; a tuned LGBM-only specialist (lr0.01,
  num_leaves512, reg_lambda20, stride2 → 1.37M regime bars) + the all-bars ensemble, blended. **VAL in-regime AUC
  peaks at W=0.5 (0.531)** (pre-commit W by VAL only). Blend denoises direction within the regime.

### MIN2 — methodology lessons (failure modes that shaped the final pipeline)
- **Thin-pocket selection degeneracy (the binding methodological constraint).** `min2_select.py` with a low
  trade floor: argmax-VAL picks `comp25_rel80 cov0.02` (VAL 0.844) → **OOS n=1** (useless). With only a 3-month
  OOS (Feb-May 2026), the tightest VAL pockets collapse to untradeable OOS samples. Fix: require `nVA>=350` AND
  drop the NY session sub-filter (NY halves the data → thin, non-generalizing OOS, e.g. W=0.5 argmax was an NY
  pocket with OOS n=26). The selection family must bet broadly enough that 3 OOS months yield n>=300.
- **Continuation filters FAIL; only reversion generalizes.** Tested both `cont{300,900,3600}` and
  `rev{300,900,3600}` trend filters. Continuation (bet WITH the last move) never reached the top configs;
  **reversion** (bet AGAINST it) wins, `rev300` best. At 120s a compressed market that just pushed mean-reverts —
  it does not continue. (This is the opposite sign to a momentum/breakout read.)
- **rel-only is not enough — compression genuinely matters.** `relonly80/90` (short-term vol expanding, NO
  compression floor) gave OOS ~0.68-0.70, clearly below compression-release ~0.73-0.75. Both conditions are
  needed: quiet base (`bbw1800` low-ish) AND `rel` high. (But the compression floor is q67, not the 1m q33 —
  moderate, not extreme, quiet.)
- **Blend weight W is robust, not fragile.** VAL in-regime AUC: W 0.2→0.528, 0.3→0.530, 0.4→0.531, 0.5→0.531,
  0.6→0.531 (plateaus 0.4-0.6); OOS held 0.755 (W=0.4) → 0.764 (W=0.5). The 50/50 blend is the VAL-argmax but the
  result is stable across weights — the specialist adds a small consistent lift regardless of exact weight.
- **OOS>TEST in a tight pocket = partial luck, not signal.** Several tight cells (`c67_r80 cov0.03`: OOS 0.799 /
  TEST 0.677) print high OOS but low TEST — the OOS>TEST gap flags a lucky split, not a real edge. The
  trustworthy configs are the ones where VAL≈TEST≈OOS on large n (the chosen pipeline: 0.763/0.694/0.764).
- **Not re-run at 120s (already settled at 1m):** temporal 1D-CNN / sequence models (1m val-AUC 0.49, near-
  efficient across every model class) and the microprice-label variant (collapsed direction signal at 1m). The
  120s direction signal is the same near-efficient ~0.51; the edge is regime + selectivity, not a better net.

### MIN2-FINAL — pre-committed proof (`min2_proof.py`) + production (`min2_production.py`)
Pre-committed pipeline (every choice frozen on VAL; OOS judged once):
- Direction = 0.5*all-bars ensemble + 0.5*compression-release LGBM specialist.
- Regime = bbw1800<=train_q67 AND rel_ratio>=train_p70 (compression-release).
- Trend = reversion vs ret300 (bet against the last 5-min move).
- Coverage = 3% confidence (conf_thr = 97th pct of |p-0.5| over VAL regime bars).
- Family swept on VAL only {reltight 0/70/85, trend none/rev300/rev900/rev3600, cov .20/.10/.05/.03}, no session
  sub-filter (NY pockets give thin non-generalizing OOS); argmax VAL acc s.t. nVA>=350 → rev300, reltight0, cov0.03.

| Split | n | accuracy | 95% CI | EV/bet@0.80 |
|-------|---|----------|--------|-------------|
| VAL 2024-H1 | 375 | 0.763 | — | +0.373 |
| TEST 2024-25 | 1591 | 0.694 | [0.671,0.717] | +0.249 |
| **OOS 2026** | **330** | **0.764** | **[0.718,0.809]** | **+0.375** |

- OOS-2026 per-month: 2026-02 0.806(n93) · 2026-03 0.753(n77) · 2026-04 0.744(n160) — **month-consistent**
  (no single-month concentration, unlike the 1m model's April-heavy OOS).
- **Honest caveat:** OOS 2026 (0.764) > TEST 2024-25 (0.694). The larger 2024-25 sample is ~0.69, so the honest
  *all-period* selective rate is ~0.70-0.72; the **>75% specifically describes the 2026 out-of-sample period**
  (verified, pre-committed, n=330, CI lower bound 0.718). Profitable on BOTH held-out periods vs 0.80 payout
  (breakeven 0.556): EV/bet +0.25 (TEST) to +0.375 (OOS).
- **CONCLUSION (2-min):** the compression-release + reversion book is the best of all three horizons. OOS-2026
  clears >75% (0.764) on a real, month-consistent 330-trade sample; the all-period rate is ~0.70-0.72. The lift
  vs 1m comes from the horizon escaping bid-ask-bounce noise; vs 15m from the regime still being predictive.
  Production pipeline: `min2_production.py` (pair-parameterized, EURUSD-labeled artifacts).

---

# Cross-pollination: applying 2-min lessons to 1-min and 15-min

After the 2-min win, tested whether its levers transfer to the other horizons. Two transferred lessons:
**(L1)** a direction SPECIALIST trained only on compression-release bars; **(L2)** a REVERSION trend filter
(bet AGAINST the last 5-min move). Plus an outside-the-box idea: **(L3)** use the 2-min model's prediction
`p_2m` as a causal cross-horizon confirmation for the 1-min bet (p_2m at t uses only data <=t; the 1m binary
settles at t+60, so it is lookahead-free).

### 1-MIN — IMPROVED (`min1_v15.py` / `_select.py` / `_proof.py`; productionized in `min1_production.py` v2)
- **Cross-horizon AUC discovery:** the 2-min model predicts the *60s* direction in-regime BETTER than a model
  trained on 60s labels — AUC_2m_inreg 0.519/0.511/0.519 (VA/TE/OO) vs the 60s specialist 0.519/0.505/0.511 and
  all-bars 0.511/0.503/0.509. The 120s label is a cleaner/denoised training target, so it generalizes better
  even for the 1m bet. So p_2m is a genuine confirming signal.
- **The big lever is REVERSION (L2).** The v1 1m book had NO trend filter. Adding "bet against ret300" lifts the
  large-sample TEST from 0.682 to ~0.74. Specialist blend (L1, w=0.5) broadens the confident set so OOS is no
  longer 1 month. L3 (2m confirm/blend) helps marginally but reversion dominates; magnitude gate is NOT in any
  top config (reversion supersedes it).
- **Pre-committed result (`min1_v15_proof.py`):**

| config | TEST 2024-25 | OOS 2026 | OOS months |
|--------|-------------|----------|------------|
| v1 baseline | 0.682 (n759) | 0.780 (n50) | 44/50 in April |
| A) all-bars + reversion (argmax-VAL) | 0.752 (n544) | 0.778 (n45) | Mar .846 / Apr .759 |
| **B) specialist-blend + reversion (robust)** | **0.739 (n1033)** | **0.777 (n184)** | **Feb .892 / Mar .850 / Apr .710** |

  Config B (productionized) fixes BOTH v1 weaknesses at once: +6 pts large-sample TEST AND 3.7x the OOS trades
  spread across all three months (CI95 OOS [0.712,0.837]). EV/bet@0.80 +0.33 (TEST) to +0.40 (OOS).

### 15-MIN — NO TRANSFER (`min15_v2.py` / `_select.py`) — itself a finding
- Specialist holds in-regime on VA/TE (AUC 0.538/0.532 vs all-bars 0.528/0.524) but **collapses on 2026 OOS
  (0.509 < all-bars 0.518)** — the known 15m fragility (V21-V23): the regime edge does not carry to 2026.
- Reversion filters (rev15_1/rev5_3/rev15_3) give combined 0.59-0.61; continuation overfits VAL (0.701) then
  falls to OOS 0.533. **No config beats the V27 baseline 0.642 combined** (best ~0.62).
- **Why:** the compression-release + short-horizon reversion edge is a sub-minute-to-2-minute MICROSTRUCTURE
  phenomenon (a quiet market that just pushed mean-reverts over the next 1-2 min). By 15m the move is fully
  developed and efficient — the reversion effect is gone. This is fully consistent with the project thesis that
  the directional edge decays with horizon (V8 horizon sweep, V25 frontier: longest >=75% horizon = 5s).

**Net:** 2-min lessons transfer DOWN the horizon (improve 1m) but not UP (15m unchanged). The edge lives at
3s-2min; 15m remains ~0.64. Updated 1m production: `min1_production.py` v2 (specialist blend + reversion).

# ============================================================================
# BIAS AUDIT & DERIV-FAITHFUL RE-VERIFICATION (2026-05-30)
# ============================================================================

A full bias audit of the backtest/OOS methodology (multi-agent review + independent Tier-1
re-runs) found the headline tick-model numbers (1m OOS 0.777, 2m OOS 0.764, 3s OOS 0.809) were
inflated by several methodology errors AND that the 1m/2m horizons are not even tradeable on
deriv EUR/USD. All production pipelines were fixed, a leakage self-test (`audit_leakage.py`) was
added, and all three production models were retrained + re-verified out-of-sample under the
corrected, deriv-faithful methodology (sequentially, proper TRAIN/VAL/TEST/OOS separation).

## What was wrong (and the fix)

1. HORIZON MISLABEL (biggest). 1s bars drop empty seconds (tick1s_cache.py), so the old label
   `mid.shift(-HS)` shifted HS *bars*, not HS *seconds*. Measured: a 60-bar "1-minute" horizon
   spanned a MEDIAN 111 wall-clock seconds (p90 233s); "2-minute" = median 225s. The models
   predicted a variable, ~2x-longer-than-advertised horizon matching no fixed binary expiry, and
   it broke trade independence (non-overlap gap 60s << 111s outcome window). FIX: `wc_ret()` — a
   TRUE wall-clock fixed-expiry label; self-test confirms median horizon now exactly 60s/120s.

2. TRADE-INDEPENDENCE / look-ahead de-overlap. Reported "independent" trades were de-overlapped
   GREEDILY by confidence, which peeks ahead within each overlap cluster to keep the most-confident
   bar (a live trader cannot). FIX: chronological first-come de-overlap (`nonoverlap_chrono`);
   greedy kept only as a labeled optimistic upper bound. Effect: -3 to -9 acc points. Bootstrap
   CIs are now over genuinely independent trades.

3. DERIV SETTLEMENT (was generic/wrong). First modeled a dealer SPREAD penalty -- WRONG for deriv.
   Verified vs deriv T&C (2.2.1.3 / 2.2.3.1 / 2.3.1) + tick API schemas + live asset_index/
   contracts_for calls: deriv Rise/Fall settles MID-to-MID, tick-to-tick, NO spread; the edge is a
   payout deduction (~15% -> payout R~1.85, breakeven 0.541). ENTRY = the NEXT tick after the order
   (not the decision tick); EXIT = the last tick at/before expiry; ties LOSE. FIX: `wc_ret` models
   entry=next tick / exit=last-before / strict win (ties lose), ENTRY_LAG_S=1; the backtest reports
   the payout-deduction EV table. Spread-aware rows removed.

4. MULTIPLE-TESTING / best-of-search. The headline OOS numbers were the max over a large,
   OOS-guided search across 30+ versioned scripts; no deflation/PBO applied. The repo's own
   diagnostic corr(VALacc, OOSacc) = -0.54 (V21) shows VAL-selection is an unreliable OOS predictor.
   FIX: each production model is a single pre-committed config (regime from TRAIN, threshold from
   VAL), judged once on TEST + OOS; no re-search.

## DERIV TRADEABILITY CONSTRAINT (decisive, live-API verified)

deriv asset_index / contracts_for for frxEURUSD: forex EUR/USD Rise/Fall MIN duration = 15 MINUTES
(max 365d), TIME units only, NO ticks/seconds. Sub-15m up/down exists only on SYNTHETIC indices,
not forex. => the 1-minute and 2-minute books are NOT tradeable on deriv EUR/USD; only 15-minute
(and longer: 30m/1h/...) Rise/Fall is placeable. Higher/Lower min = 1 day.

## CORRECTED, DERIV-FAITHFUL OUT-OF-SAMPLE RESULTS (retrained 2026-05-30, sequential)

  model  horizon  deriv-tradeable   TEST 2024-25        OOS 2026             verdict
  -----------------------------------------------------------------------------------------------
  min1   60s      NO (<15m floor)   0.539 (n1017)       0.550 (n349,         ~breakeven; OOS CI
                                    CI[.508,.570])      CI[.499,.602])       includes 0.50 -> no edge
  min2   120s     NO (<15m floor)   0.528 (n2528)       0.539 (n710,         ~coin-flip; no edge
                                    CI[.508,.547])      CI[.503,.576])
  m15    15min    YES (at floor)    2024 0.689/2025     2026 0.663 (n89,     COMBINED 0.647 (n677,
                                    0.582               CI[.562,.753])       CI[.612,.684]) <- the
                                                                             real deriv edge
  deriv payout-deduction EV (R=1.85, breakeven 0.541): min1 OOS +0.018, min2 OOS -0.002,
  m15 combined +0.197.  Reference (1m): greedy look-ahead would have shown ~0.63 (the old
  inflation); same-tick entry (not achievable on deriv) ~0.57; honest deriv next-tick 0.539/0.550.

## BOTTOM LINE

- The 1m/2m ">=75%" headlines do NOT survive honest, deriv-faithful methodology AND are not
  tradeable on deriv EUR/USD (15m forex floor). The sub-minute microstructure edge is real but
  decays within a tick or two of entry latency -- uncapturable at a 15m-floor forex venue. They
  remain of interest only for synthetic indices (which allow ticks/seconds) or another broker.
- The 15-MINUTE Rise/Fall book is the genuine, deriv-tradeable, OOS-verified edge: ~0.647 combined
  / 0.663 OOS-2026, profitable vs deriv's payout deduction; next-tick entry is negligible at 15m so
  it is not eroded by latency.
- All production pipelines (min1/min2/m15_production.py) are deriv-faithful and guarded by
  `audit_leakage.py` (run before any deploy).
