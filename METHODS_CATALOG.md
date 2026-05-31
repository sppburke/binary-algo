# EURUSD Binary Direction — TIMEFRAME-AGNOSTIC METHODS CATALOG

A reusable catalog of **every technique tried across all sessions** (1-second → 30-minute), framed so it can be re-pointed at
**any** horizon. Use this before starting a new-horizon push: most methods are already implemented and parameterized by horizon —
check the matrix for what's been tested where, and the "untested" cells are your ready-made backlog. Per-horizon detail lives in
`m30_research_log.md`, `m5_research_log.md`, `m10_research_log.md`, `min1_research_log.md`; master summary in `DIRECTION_FINDINGS.md`.

## 0. How to retarget any method to a new horizon

- **Bar-based methods** (cross-pair, stack, walk-forward, gate sweeps) read the horizon from an env var:
  `MX_HOR=<minutes> python m5_xpair.py …` (m5_xpair.py:22 `HOR=int(os.environ.get("MX_HOR","5"))`, GAP auto = HOR×60s, label =
  contiguous next-HOR-minute sign with ties excluded). `MX_HOR=1` gives the 1-min bar label for free. The 239 multi-TF features in
  `features/<PAIR>_<year>.parquet` (2012-2026, 7 majors) are horizon-independent.
- **Tick-based methods** (`min1_production.py`, `min2_production.py`) set the wall-clock expiry `HS` (seconds) + `TOL_S` + entry
  lag; the 1-second microstructure cache is in `features_tick*/`. Splits differ from bar models: tick = train 2021-23 / val 2024-H1 /
  test 2024.09-2025.11 / oos 2026 (tick data only spans 2021-2026); bars = train 2012-21 / val 2022-23 / test 2024 / test 2025 /
  oos 2026.
- **Parent ensembles for cross-horizon stacking** are on disk per horizon: `models/m{5,10,15}_EURUSD_direction_{lgb.txt,xgb.json,
  cat.cbm}` (all use the same 239 base cols). Load via the `_load`/`pf` pattern in `m5_stack2.py` / `min1_stack.py`.
- **Discipline harness** (reuse everywhere): `nonoverlap_chrono(ts,mask,gap)` (live-faithful de-overlap), `boot()` (bootstrap CI95
  over non-overlap trades), `wc_ret()` (deriv mid-to-mid settlement, ties LOSE, next-tick entry). Always: select on VAL by
  **worst-VAL-half stability**, then require the edge on EACH of {2024, 2025, 2026} with CI95 — never VAL-acc-max (corr=−0.54).

## 1. Method × horizon results matrix (honest OOS, deriv-faithful)

`—` = not yet tested at that horizon (← candidate experiment). Numbers are best honest selective accuracy (or AUC where noted).

| Method (script) | 1–5 s | 60 s (1 m) | 5 m | 10 m | 15 m | 30 m |
|---|---|---|---|---|---|---|
| OHLCV/bar 239-feat ensemble | n/a | 0.51 AUC | 0.52 AUC→0.566 | 0.525→0.60 | 0.52→**0.647** | 0.518→0.59 |
| Tick microstructure ensemble (`min1/2_production`) | **~0.65** | 0.550 | 0.56 | — | — | 0.50 AUC |
| 1D-CNN / GRU on raw tick path | edge | 0.49 AUC | — | — | — | 0.525 AUC |
| Compression-release × reversion gate | — | 0.550 | 0.566 | 0.60 | 0.647 | 0.591 |
| Cross-pair USD-residual (contemporaneous) | — | 0.516 AUC | **0.586** | 0.599 | +0 | +0 |
| Cross-pair lead-lag (CCF, time-shifted) | — | null (0.51) | — | — | — | — |
| Per-side bid/ask raw order flow (`tick_data/raw`) | — | null (0.51) | — | — | — | — |
| Order-flow / Kyle-λ (`features_of`) | — | (in meta) | helps book | weak | — | — |
| **Cross-horizon STACK** (parent→child front-load) | — | 0.59/0.53 | **0.613/0.648** | ≤0.56 | n/a(parent) | — |
| Meta-labeler (worst-VAL-half) | — | 0.49–0.60 | 0.583 | ~0.59 | — | — |
| Hurst / variance-ratio switch | — | 0.555 oracle | — | — | — | null |
| **HMM regime** (Gaussian, causal filtered) | — | 0.565/0.60thin | — | — | — | — |
| Online concept-drift (river ARF+ADWIN) | — | 0.50 AUC | — | — | — | — |
| Kalman channel/velocity/β (filter-only) | — | 0.49–0.52 | — | — | — | — |
| Kernel-SVM (RBF Nyström, Fletcher) | — | 0.50 AUC | — | — | — | — |
| Macro-release surprise impulse | — | null/neg | null | — | — | — |
| Up/down asymmetry (filter) | — | up 0.58–0.61 | — | — | — | — |
| External ES/NQ cross-asset lead-lag | — | — | — | null/flip | — | null/flip |
| Walk-forward retraining (1-yr gap) | — | — | — | +0.017 | — | +0.011 |
| Sofien indicators / 79 rules | — | — | 0 AUC | — | — | 0 AUC |
| Path-sig / Hawkes / transfer-entropy / complexity | top@5s | — | — | — | — | null(magnitude) |
| FX fixing-window reversals | — | — | ~0.58 IS, flips OOS | — | — | ~0.58 IS |
| **MAGNITUDE model** (\|ret\|≥Q, the real edge) | — | **0.79 AUC** | — | — | — | **0.73–0.78 AUC** |

**Biggest untested cells (ready experiments for other horizons):** HMM / online-drift / Kalman / kernel-SVM / up-down-asymmetry /
per-side-order-flow were only run at 60 s — none have been tried at 5m/10m/15m/30m. The cross-horizon stack (the single strongest
method) has only been run at 5m and 60s. Given the wall is the same everywhere, priors are low, but these are the cheap fills.

## 2. Method catalog (what each is, the script, the reusable knob)

- **Cross-horizon STACK** (`m5_stack2.py`, `min1_stack.py`) — front-load a longer parent's confident direction into the shorter
  outcome, gated by a meta-labeler on orthogonal axes (cross-pair agreement/dispersion, order-flow, parent confidence). **Strongest
  method anywhere** (5m 0.613/0.648). Retarget: set the child label horizon, load the parent ensemble(s). Caveat: the parent
  front-loads weakly when the child horizon ≪ parent (60s child off a 5m/15m parent front-loads at only ~0.51).
- **Cross-pair USD-residual** (`m5_xpair.py`, env `MX_HOR`) — relative-value residual of EURUSD vs the USD-weakness basket of 6 other
  majors; the reversion residual is the one orthogonal family that's **sign-stable across 2024 & 2026** (momentum agreement is dead).
  Helped at 5m (0.586); weak by 10m; ~0.52 at 60s. `augment(...,"xpof")` adds the 239 base + 18 order-flow cols.
- **HMM regime** (`min1_hmm.py`) — Gaussian HMM on causal emission features; **assign states with the forward FILTER only**
  (`_forward_filter`), never Viterbi/forward-backward (they peek at the future). Three uses: regime-gate, per-state engine-switch,
  soft-posteriors-as-meta-features. Found: states are pure volatility regimes (P(up)≈0.50) → no directional lift. Retarget: change
  the label horizon; reuse `causal_states`. hmmlearn installed.
- **Online concept-drift** (`min1_online.py`) — river `ARFClassifier`+ADWIN, **prequential** (predict-then-learn) in time order; the
  control that proves a wall is genuine efficiency vs stale-model drift (if an adaptive model still hits ~0.50, it's efficiency).
- **Kalman** (`min1_kalman.py`) — local-linear-trend filter (level+slope) and time-varying cross-pair β. **Forward filter only — the
  RTS smoother leaks the future.** Channel-revert ≈ adaptive compression×reversion (magnitude not sign); velocity ≈ smoothed momentum
  (the dead side). filterpy/pykalman installed.
- **Kernel-SVM** (`min1_kernel.py`) — RBF via `Nystroem`+`SGDClassifier` (SVC doesn't scale) on microstructure + signed per-side flow.
  A different model class; confirmed the data, not the model, is the ceiling (train AUC 0.53, val 0.50).
- **Macro-release impulse** (`min1_news60.py`, `m5_news.py`, calendar `macro_calendar.parquet` via `fetch_calendar.py`/`event_signs.py`)
  — predict sign(surprise) in the post-release window. Null at 5m AND at the 60s impulse scale (FX prices a surprise in <60s and
  overshoots-reverts; larger surprises are MORE wrong OOS). News = magnitude, not direction.
- **Up/down asymmetry** (`min1_updown.py`, `min1_upspec.py`) — split predictions by side; the edge can live entirely on one side
  (60s: up/dip-buy carries it, down is dead). Exploit as a trade FILTER on the symmetric model — a separately-TRAINED side specialist
  is WORSE (subset-training kills the confidence ranking). Regime-dependent (the 60s up-edge vanished in 2024).
- **Walk-forward** (`m5_walkforward.py`, `m10_walkforward.py`) — annual expanding-window retrain; isolates regime-shift vs stale-model
  (lifts the binding window only +0.01–0.02 → the weakness is efficiency, not the train-test gap).
- **Magnitude model** (`m30_magnitude.py`, `min1_magdir`/`_redteam_magdir60.py`) — |ret|≥Q classifier; **the genuine forecastable
  edge** (AUC 0.73–0.79, stable across years) at every horizon. Tradeable on touch/range/straddle, NOT up/down.

## 3. Leakage / bias traps (timeframe-agnostic — these recur and inflate results at every horizon)

1. **Sequence-model future-peeking:** HMM Viterbi/forward-backward and the Kalman RTS smoother both condition on the WHOLE series.
   Use the forward FILTER (state/level at t from data ≤t) only. (Cost us nothing because we caught it; the literature's "HMM/Kalman
   trading" results are often inflated by exactly this.)
2. **ffill-flat-window label artifact:** intersecting multiple pairs' timestamps + forward-fill manufactures ~50% fake flat/no-move
   bars; a model then "predicts" no-move and shows AUC 0.7+/CI>0.65 — which evaporates on moved-bars-only (real up-rate ~0.49).
   ALWAYS evaluate on moved bars (|fwd|>0) and check the train up-rate ≈ real up-rate.
3. **Wavelet-denoising lookahead:** denoisers that smooth across the series see the future; inflates daily DL F1 (Lopez Gil 2024).
4. **Greedy-by-confidence de-overlap:** peeking ahead to keep the most-confident bar in an overlap cluster — use
   `nonoverlap_chrono` (first-come, no look-ahead) for the headline number.
5. **Bar-shift horizon mislabel:** on gap-dropped tick data, `shift(-N)` shifts N *bars* not N *seconds* → a variable ~2× horizon and
   broken trade independence. Use a strict wall-clock lookup (`wc_ret`).
6. **VAL-acc-max selection:** corr(VAL_acc, OOS_acc) = −0.54 → maxing VAL anti-transfers. Select by worst-VAL-half stability.
7. **Thin-coverage mirages:** any n<25–50 pocket at 0.65+ is multiple-testing noise; require OOS n≥25 (ideally ≥100) AND CI95 lower
   bound above the bar on EACH window.
8. **Ties:** binary settlement ties (ret==0) LOSE; a ~0.50-AUC model's realized win-rate sits BELOW 0.50 once ties are charged.

## 4. Timeframe-agnostic lessons (the priors that held across 1s→30m, ~17 model classes, 6 research workflows)

1. **Direction ≠ magnitude.** Sign is ~EMH at ≥5m (and ~0.51 AUC at 60s across LGBM/CNN/HMM/kernel/online); |return| is strongly
   forecastable (AUC 0.73–0.79) at every horizon. Sign-invariance theorem (arXiv:2512.15720): entropy/order-flow/complexity gate
   move SIZE, not sign. **The productizable edge is magnitude (touch/range/straddle), not up/down.**
2. **Information, not model class, is the bottleneck.** GBM ≈ CNN ≈ GRU ≈ HMM ≈ kernel-SVM ≈ online-forest at the data ceiling.
3. **The directional edge is at the 1–5 second scale (~0.65) and decays to ~0.51 by 60s.** Longer = more efficient. A real >0.65
   up/down book needs a tick/seconds-expiry venue (`m_tick_prod.py` 3s 0.657/0.667).
4. **The binding constraint is the 2025 regime** at every horizon (every method drives 2024 & 2026 to ~0.60–0.70 but 2025 to
   ~0.50–0.60). It's genuine near-efficiency (the online-adaptive control confirms), driven by a macro risk-on/USD inversion
   (ES→EURUSD lead-lag flips +0.02→−0.05). Walk-forward lifts it only +0.01–0.02.
5. **The cross-horizon stack is the one way to beat a horizon's own noise floor** — borrow the cleaner longer-horizon signal. But the
   child asymptotes to the parent's native ceiling (5m stack → the 15m parent's 0.647) and front-loads weakly when child ≪ parent.
6. **Cross-pair USD-residual reversion** is the one orthogonal signal family that helped (sign-stable 2024 & 2026), strongest at 5m,
   fading by 10m, ~0.52 at 60s.
7. **Published ceiling agrees** (scholarly deep-search, `beats_our_060=False`): best EUR/USD direction is 0.585 DAILY (Castillo 2024);
   honest sub-5-minute sign ceiling ~0.52–0.55 net of costs (Petrova-Vilhelmsson-Nordén, IJF 2025); >0.65 at 60s is unsupported.

## 5. Where the real upside is (for any horizon push)

- **MAGNITUDE productization** (AUC 0.73–0.79 everywhere) — the genuine edge; needs a touch/range/straddle venue.
- **Seconds/tick-expiry venue** — to trade the real ~0.65 1–5s directional edge.
- **External data NOT on disk** (ranked candidates to acquire): options-implied risk-reversal/skew (does FX skew lead spot?),
  intraday US–DE rate-differential futures, full depth-10 LOB volumes, CFTC COT positioning. These are the only inputs that could
  change the directional answer; on-disk data is exhausted across ~17 model classes.
- **15-minute direction (~0.647)** is the best *deriv-tradeable* up/down book today.
