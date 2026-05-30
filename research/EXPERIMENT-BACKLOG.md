# EXPERIMENT BACKLOG — Beating the 15-Minute FX Direction Current Best Level

Synthesized from research vectors 01–16 (`research/01-*.md` … `research/16-*.md`) and the V1–V17
ledger / headline result in `README.md`. Goal: lift 15-minute EURUSD (and USD-major) directional
accuracy toward 75%, strictly OOS (TRAIN 2012–21 · VAL 2022–23 · TEST 2024–25 · **OOS 2026 locked**).

## How to read this backlog

**The single most important meta-finding across all 16 vectors:** *an always-on 75% at 15m on a liquid
FX major is still open — not yet achieved in a credible, reproducible source.* The most rigorous FX-LOB
study (Petrova–Vilhelmsson–Nordén 2026) ran almost exactly our experiment with better data and reported
results near the efficient-markets baseline. Credible honest results so far: ~58.5% daily, lower intraday.
Our own ~0.52 AUC is the current best level on all-bars, the regime we are working to lift.

**Therefore the realistic near-term prize is reframed as: high accuracy at low-but-usable coverage, inside
mechanistically-special states** — while the always-on 75% target stays open. Every P0 below is either (a)
a genuinely orthogonal *directional* signal we have not built, or (b) machinery that makes our existing
selective edge *generalize* OOS. A bet that only "moves AUC 0.002" is explicitly de-prioritized.

**Validation contract for EVERY experiment (non-negotiable):**
- Causal features only; purged + embargoed CV with embargo ≥ the 15m label span at every fold boundary.
- Pick the operating threshold on **VAL 2022–23**, freeze it, then report TEST 2024–25 **and** 2026 OOS
  at that *same* threshold. A result counts only if it holds on both at the VAL-chosen threshold.
- Deflate: compute Deflated Sharpe / PBO given our large (~17+) effective trial count before believing any tail number.
- Cost-realism: re-label / re-score on mid-to-mid **net of half-spread** from our own tick bid/ask
  (and an asymmetric-fill variant) — a 0.63@0.2% edge can be entirely a spread artifact.

Effort: S = <1 day · M = 1–3 days · L = >3 days. Priority: P0 (do first) · P1 · P2.

---

# TOP 5 (priority order)

1. **E1 — Cross-asset USD-factor fair-value-gap (rates + ES + DXY-residual), windowed**
2. **E2 — Signed macro-surprise event-conditioned sub-model (the directional payload, not vol-timing)**
3. **E3 — Mondrian conformal + DtACI selective layer (make the 0.632 generalize OOS, guaranteed floor)**
4. **E4 — Fixing-clock directional drift / reversal overlay (Krohn W-pattern, sign not vol)**
5. **E5 — Walk-forward online learning + meta-labeling on the existing primary**

---

# P0 EXPERIMENTS

## E1 — Cross-asset USD-factor fair-value-gap (rates + ES + DXY-residual), windowed
- **Hypothesis.** EURUSD is ~97% USD-factor (Verdelhan); the informative variable is therefore the
  *USD factor's external drivers*, not peer FX. The US–DE 2y/10y rate differential and ES/risk repricing
  on data often lead spot FX by seconds-to-minutes (Huth–Abergel lead-lag peaks at releases/US open).
  Spot under-reacts short-term to a sharp rate-diff move → the residual mean-reverts toward fair value
  over the next 15m, giving a *signed* direction. This survives because it is exogenous repricing, not
  endogenous microstructure that decays by 5 min.
- **New data.** NOT in hand. Cheapest credible path: **Databento CME 6E + ZN/ZF (Treasury futures) + ES**
  intraday tick (low cost), plus free FRED/ECB daily yields to prototype first. DXY-basket residual is
  buildable **for free** from our existing 7 pairs today.
- **Method.** Causal rolling/expanding ridge regression of the target pair's recent return on
  contemporaneous (i) US-minus-DE 2y & 10y futures-implied yield changes, (ii) free-built USD-basket,
  (iii) ES return / VIX change. Features = residual sign + z-score, plus *driver velocities*
  (Δrate-diff over 30s/1m/5m, ΔES over 1m). Feed to existing LightGBM. **Split the target into
  continuous vs jump (bipower/threshold) and train only on the continuous part** (Aleti–Bollerslev–
  Siggaard: cross-asset leads are non-predictive for the jump component). Interact every driver with a
  news/US-open window dummy — unconditioned cross-asset washed out for us before; the *conditioning* is the new idea.
- **Differs from prior work.** We did cross-*pair* lead-lag (≈0 lift) and USD-basket *residual fade*.
  We have NEVER fed *external* USD-factor drivers (Treasuries, ES) at minute resolution, nor split
  continuous/jump, nor used the *signed realized rate move* (our event-time proxy used "release happened" vol-seasonality only).
- **Expected lift (honest).** Small on all-bars; the credible payoff is a few points of accuracy on the
  *continuous* component concentrated in release / US-open windows. Prototype free (FRED daily + basket)
  to detect *any* signal before paying Databento. Validate per contract; report accuracy per-regime.
- **Effort L · Priority P0.**

## E2 — Signed macro-surprise event-conditioned sub-model
- **Hypothesis.** Scheduled-macro windows are the one place >55–60% directional hit-rates are credibly
  reproducible (Andersen–Bollerslev–Diebold–Vega; ECB WP1901: ~half the move is pre-release drift; sign
  maps to the standardized surprise). This is an *exogenous information shock*, so it escapes the
  microstructure 5-min decay. The directional carrier is `z=(actual−consensus)/σ`, which we never used.
- **New data.** Need a release calendar **with consensus + actual + true wire timestamps** (Trading
  Economics / FXStreet / Finnhub free tier; ForexFactory/Econoday scrape). FRED ALFRED for point-in-time
  vintages to avoid revision leakage. CB text (FOMC/ECB statements) free from Fed/ECB sites for the tone overlay.
- **Method.** Build a *separate* two-headed model that activates only in event windows: features =
  `sign(z)×|z|`-buckets, minutes-to/since-release, first-10–60s post-release return as a state, EURUSD's
  historical beta to that release; pre-window drift sign from the surprise-series trend. Optional CB-tone
  overlay scored with a CB-specific lexicon/LLM (Picault–Renault / Ornithologist), NOT vanilla FinBERT.
  Keep it as a regime overlay — do NOT dilute the all-bars LightGBM (that is why the prior calendar proxy added 0).
- **Differs from prior work.** V17 calendar proxy captured *vol seasonality* (redundant with time-of-day).
  This uses the **signed surprise** — the actual directional payload — and a continuation/fade model on bars 2–3 (5–15m out).
- **Expected lift (honest).** 60–70% on the ~2–5% of bars near Tier-1 releases (NFP/CPI/FOMC/ECB/PMI);
  ~0 elsewhere. A high-precision, low-coverage book. **Must re-validate stationarity on 2024–25 + 2026**
  (Lucca–Moench: pre-FOMC drift largely decayed post-2015 — assume nothing from a 2012–14 backtest).
- **Effort M · Priority P0.**

## E3 — Mondrian conformal + DtACI selective layer (generalize the 0.632 OOS)
- **Hypothesis.** Our selective threshold is over-confident and "collapses OOS" because (a) a fixed
  |p−0.5| cutoff tuned on VAL is selection-biased, and (b) FX violates exchangeability so vanilla
  conformal under-covers exactly at vol spikes. Mondrian (label-conditional) conformal gives a
  *distribution-free finite-sample error floor per predicted side*; DtACI online-adapts the miscoverage
  so realized accuracy tracks target *through* regime shifts instead of collapsing.
- **New data.** None — methodological. Needs a temporally-contiguous, recent calibration block carved
  from train, plus our own tick-derived spread for cost-aware evaluation. Libraries: `crepes`/`MAPIE`
  (Mondrian), `venn-abers`, ACI/DtACI reference code.
- **Method.** (1) Mondrian inductive conformal reject-option: non-conformity `s=1−p_y`, calibrate
  per-class; bet only on singleton prediction sets; ε=0.25 targets ≥75% covered accuracy *by
  construction* if exchangeability held. (2) Wrap in **DtACI** to online-update the effective ε as
  2024–25 regimes shift. (3) Add **Venn–ABERS** for honest probabilities + an orthogonal abstention axis
  (interval width). (4) Per-regime Mondrian buckets {session × vol-regime}. **Do NOT chase Platt/isotonic
  to raise accuracy — calibration is order-preserving and provably cannot move the risk-coverage curve.**
- **Differs from prior work.** We did *ad-hoc* |p−0.5| tail search (optimistically biased). This *fixes
  the error target a priori* and adapts online — the literal fix for "threshold that generalizes."
- **Expected lift (honest).** Will NOT manufacture signal: if true tail edge is ~0.60–0.63, you get a
  *trustworthy, floored* ~0.60–0.63 with no OOS collapse — the prerequisite for safely harvesting any new
  signal (E1/E2/E4) at coverage. Budget wide CIs at low coverage (tail estimates are high-variance).
- **Effort M · Priority P0.**

## E4 — Fixing-clock directional drift / reversal overlay
- **Hypothesis.** Krohn–Mueller–Whelan (J. Finance 2024): USD systematically appreciates into the Tokyo
  / ECB(2:15pm CET) / London(4pm WM-R) fixes and reverses after — a W-shaped, sign-predictable intraday
  pattern, all-G10, 21y, t-stats to 9.2, robust to calendar effects. This is a *direction* signal keyed
  to wall-clock, mechanistically distinct from the time-of-day *vol* seasonality we ruled out.
- **New data.** None new — just precise fix timestamps (public) aligned in exchange-local time with DST
  (ECB fix moved to 14:15 CET in 2016 — handle the regime break). Uses our existing bars.
- **Method.** Event-time features: minutes-to-next-fix, minutes-since-last-fix, cumulative pre-fix drift,
  and a signed "fix-pressure" = sign(time-rel-fix) × historical Krohn drift-sign for that pair. Train a
  *separate* conditional model that predicts only in ±30m fix windows; model the post-fix reversal sign
  explicitly. EUR is the best candidate (only pair positive net of CME costs in Krohn). Interact fix-flag
  × existing realized-vol/OFI.
- **Differs from prior work.** Our event-time work was vol seasonality (redundant). This is the *sign of
  drift / reversal*, never isolated.
- **Expected lift (honest).** Small magnitude (single-digit bps, much eaten by retail spread — net Sharpe
  0.5–0.7 only at institutional spreads, GBP/JPY go negative). The point for *our accuracy metric* is
  whether in-window directional hit-rate clears the bar at usable coverage. Free to test; report accuracy
  *inside windows* and net of spread.
- **Effort M · Priority P0.**

## E5 — Walk-forward online learning + meta-labeling on the existing primary
- **Hypothesis.** In the most analogous public comp (Jane Street 2024), *online learning* (one gradient
  step per new revealed label at inference) gave +0.008 — ~4× the lift of all feature engineering
  combined — because the signal is non-stationary. We have a continuous series with eventually-revealed
  labels and have never tried streaming weight updates. Meta-labeling then formalizes our selective bet.
- **New data.** None — reuses existing pipeline.
- **Method.** (1) Take the best GRU/LightGBM primary; after each 15m label resolves, do a small online
  update (NN: 1 SGD step lr≈1e-4–3e-4 on recent rows; GBDT: periodic warm-start or an online head on
  GBDT-leaf embeddings, e.g. `river`). Evaluate strictly walk-forward on VAL first. (2) Add a meta-label
  layer: triple-barrier (vol-scaled PT/SL + 15–60m vertical barrier, net of our tick spread) → secondary
  model predicts P(primary correct) from primary prob, recent hit-rate, vol regime, session, spread; bet
  only when high. Use average-uniqueness weighting + sequential bootstrap (overlapping labels); **avoid
  return-attribution weighting** (documented to misbehave).
- **Differs from prior work.** Online/streaming updates and triple-barrier meta-labeling are both untried;
  our selective prediction was a naive confidence gate, not a learned second model.
- **Expected lift (honest).** Online learning is the single highest-EV *untried, no-new-data* lever.
  **Pre-test first (~1 hr, P0 gate): bucket existing OOS preds by |p−0.5| × vol-regime × session and
  measure conditional accuracy. If no bucket materially exceeds ~0.55, meta-labeling is unlikely to reach
  0.75 from here** — this cheaply falsifies the meta vector before building it.
- **Effort M · Priority P0.**

---

# P1 EXPERIMENTS

## E6 — Integrated multi-level OFI + microprice from quote sizes
- **Hypothesis.** Our ≈0-lift "signed 10s-volume OFI" was a degraded single-level wall-clock proxy.
  Integrated multi-level OFI (Cont–Cucuringu–Cont: 71%→87% contemporaneous R²) and Stoikov microprice
  are materially better-specified and computable from the quote *sizes* we already have. Expect real lift
  at 1–5m, marginal at 15m — but it sharpens labels/entries and feeds the selective layer.
- **New data.** None (uses tick quote sizes). True L2 depth (LSEG PCAP / CME 6E) would strengthen it but
  is low-ROI at 15m given our own decay finding.
- **Method.** Per-event signed queue-size change → OFI per window (10s/30s/1m/5m/15m) → PCA-integrate →
  normalize by rolling depth (the Kolm–Turiel–Westray stationarity step is the key detail). Add Stoikov
  microprice and use the **15m-aggregated drift of (microprice − mid)** + time-avg imbalance as features,
  and microprice-to-microprice returns as cleaner (bid-ask-bounce-free) labels.
- **Differs from prior work.** Different, better OFI estimator + microprice (never built). Re-opens the
  OFI question cleanly rather than re-proposing the exhausted tick-rule proxy.
- **Expected lift.** Modest at 15m; primary value is label de-noising + a conviction input. **Effort M · P1.**

## E7 — Metaorder-in-progress detector (order-flow long-memory)
- **Hypothesis.** The *instantaneous* imbalance edge decays in seconds, but the *persistence of the flow
  that generates it* has power-law long memory at the 3-minute scale (Lillo–Mike–Farmer; confirmed in FX
  spot by **Gould–Porter–Howison 2016**, H≈0.7) because institutions split metaorders over minutes-to-hours.
  The signal is "is a same-signed metaorder still running, and how much life remains" — a feature family at
  *our* timescale, untried. **Verified caveat (C3):** FX *aggressive market-order* sign decays to null in
  ~2 min (Lallouache–Abergel 2014, EBS); limit/cancel-sign runs to ~5 min. Scope the usable horizon to ~2–5 min,
  not an extended multi-minute trade-sign edge — this tightens E7's target window, it does not kill it.
- **New data.** None (ticks).
- **Method.** Persistence features (rolling sign-autocorr, Hurst of signed flow, run-length, fraction of
  last-K same side); Hawkes residual same-side intensity (2D buy/sell, sum-of-exponential kernels); online
  Bayesian change-point gate to trade only inside a stable persistent-flow regime; square-root-decay-aware
  label (more expected drift early in a run, fade near exhaustion). Classifier on these → 15m direction.
- **Differs from prior work.** Not instantaneous imbalance; a continuation-of-persistent-flow model.
- **Expected lift.** Highest-novelty / highest-risk in the microstructure family; the only one whose
  physics lives at minutes. **Effort L · P1.**

## E8 — Information-driven bars + triple-barrier re-targeting
- **Hypothesis.** Everything we tried was *fixed-time*. Imbalance/run/dollar bars re-clock the series by
  information arrival (returns closer to IID, lower serial corr) and fire precisely when informed flow
  arrives. Features weak at fixed 15m spacing may be informative *at imbalance-bar events*, and "direction
  over next K imbalance bars" may be more predictable than "next 15 minutes."
- **New data.** None (ticks, signed via quote-rule).
- **Method.** Build tick-imbalance bars (AFML EWMA-threshold algo) per pair; re-sample all 239 features at
  bar timestamps; relabel with vol-scaled triple-barrier. Test AUC at bar events vs fixed 15m. Pairs with E5/E6.
- **Differs from prior work.** Changes the *clock and the label*, the one axis never varied. **Effort M · P1.**

## E9 — Rough-path signatures over the 7-pair quote stream
- **Hypothesis.** Signatures encode signed *order-and-area* (Lévy area, covariation-of-increments) of a
  multivariate path — cross-pair lead-lag *rotation* that linear lag features cannot represent. Sig-DNN
  beats raw-LOB DNN on error; the Guo 2025 covariation term is an explicit *directional* indicator.
- **New data.** None.
- **Method.** Path = time- + lead-lag-augmented [mid_i, signed-size_i] over the 7 pairs for a trailing
  window; Generalised Signature Method canonical config (log-signature depth ~3, per-window, causal);
  include cross-pair Lévy-area + covariation terms; feed to LightGBM (`iisignature`) or as a `signatory`
  layer in a net. Strictly causal (window only).
- **Differs from prior work.** Genuinely new representation of the same data; not in V1–V17. **Effort L · P1.**

## E10 — Regime-gated / vol-regime-sign-conditioned modeling
- **Hypothesis.** Our flat −0.03 lag-1 autocorr is an average that *cancels* a positive (momentum) low-vol
  component against a negative (reversal) high-vol component (Engel: regime models help *direction* even
  when they fail on MSE). Conditioning on vol regime should expose non-zero, opposite-sign conditional autocorr.
- **New data.** None.
- **Method.** **First, a ~1-afternoon diagnostic (P0-cheap gate):** bucket 15m bars by trailing-RV tercile,
  compute lag-1 autocorr within each; if top/bottom terciles show opposite-sign non-zero autocorr, a
  conditional momentum/reversal rule falls out. Then: HMM (Gaussian/MS-GARCH) decoded **filtered, not
  smoothed** (smoothing leaks) → train LightGBM per state or feed soft state-probabilities as features; bet
  only in states clearing a VAL threshold. Optional Input-Output HMM with RV-ratio/session covariate-driven transitions.
- **Differs from prior work.** Explicit regime conditioning of the *sign*, not another unconditional model.
- **Expected lift.** Pushes selective accuracy to higher *coverage* (e.g. ~0.60 at 5–15%) rather than raw
  75% all-bars. **Effort M · P1.**

## E11 — Retail order/position-book level-structure (OANDA) as a stop-cluster map
- **Hypothesis.** OANDA's forexlabs Order Book (pending orders) + Position Book (open positions) bucketed
  by price is a *liquidity map* of where retail stops/limits cluster — mechanistically distinct from TA and
  from executed OFI. Stops are magnets (stop-runs); crowded positions flag squeezes. EURUSD-specific 2025
  academic support that intraday retail flow is contrarian and exploitable.
- **New data.** OANDA forexlabs API (`orderbook_data`, `position_book`) — free w/ account, 5min premium /
  15min free. Forward-only (no clean deep history) → prospective experiment.
- **Method.** Per snapshot: stop-cluster pressure (net stop pool above vs below within ±5/10/20 pips),
  position skew (pl−ps)/(pl+ps), order-book asymmetry, and — the real alpha — *Δ over last 1–3 snapshots*.
  **Residualize against existing return/TA features first** (retail positioning is partly lagged returns).
  Lag strictly by publication cadence (no using the bar's own snapshot). Use as conditioning/gating.
- **Differs from prior work.** Pending/resting orders of a slow contrarian crowd, not executed imbalance. **Effort M · P1.**

---

# P2 EXPERIMENTS

## E12 — Dealer-gamma volatility-regime gate + option-expiry magnet sub-model
- **Hypothesis.** FX dealer gamma (Ulmann–Sornette) drives *volatility, not direction* (and the gamma→
  intraday-momentum trade is proven *insignificant in currencies*, Baltussen et al.). But (a) gamma sign
  is a useful trend-amplify-vs-mean-revert *gate*, and (b) the 10am-NY-cut expiry magnet is a genuine
  conditional mean-reversion-toward-strike effect.
- **New data.** Free: CME QuikStrike OI-by-strike, DTCC FX-option tape (messy), ForexLive expiry lists.
- **Method.** Crude GEX-by-strike proxy (assume dealers short gamma) → `gamma_regime` gates whether our
  momentum/micro signals persist. Separate expiry-magnet selective sub-model: predict only when spot within
  ~40 pips of a ≥$0.75bn strike, ≥30min pre-cut, quiet tape. **Do NOT import equity-GEX 75%-winrate blogs.**
- **Expected lift.** Gating + a small high-precision expiry bucket; not an all-bar lift. **Effort L · P2.**

## E13 — Raw multi-channel tick+quote-size sequence encoder (last honest DL attempt)
- **Hypothesis.** The only place DL can beat our GBMs is raw *quote-size* structure that 10s aggregation
  destroys. A causal DeepLOB-style front-end (Conv over price/size → Conv over time → GRU) on the
  un-aggregated tick matrix might find orthogonal 15m structure — base-rate prior says lift is small.
- **New data.** None (our quote-size feed is the scarce ingredient most papers lack).
- **Method.** Channels {bid, ask, bid_size, ask_size, microprice, signed size} resampled to 100ms–1s over
  a 5–15m lookback; causal/dilated TCN (no leakage); 15m head + auxiliary multi-horizon (5/30/60m) +
  smoothed-return heads (multi-task denoising). Clean falsification: self-supervised pretrain on unlabeled
  ticks → linear probe; if it can't beat LightGBM, the DL vector is dead (learned cheaply).
- **Differs from prior work.** Raw size dynamics, not hand OFI; multi-task heads; SSL pretraining — untried.
- **Note.** Do NOT re-implement DeepLOB hoping for 15m (microstructure dead by 5 min — our finding +
  Lucchese 2024 + TLOB all agree); do NOT use zero-shot Chronos/TimesFM (~0.50 on returns). **Effort L · P2.**

## E14 — Fractional-differentiation features (memory without non-stationarity)
- **Hypothesis.** Our 239 features + the −0.03 autocorr are all built on returns (d=1, memoryless).
  Fracdiff finds the min `d*` that is ADF-stationary while preserving maximal memory — exposing long-memory
  in the *price level / spread / USD-basket level / cross-pair-spread residual* that returns destroy.
- **New data.** None.
- **Method.** Binary-search ADF `d*` (FFD fixed-width) on EURUSD mid level, log-spread, basket level, and
  EURUSD−basket residual; feed FFD series + short lags to the model. Causal/expanding only.
- **Differs from prior work.** The one genuinely untried orthogonal feature *source* in the AFML vector. **Effort S · P2.**

## E15 — Combinatorial imbalance + cross-sectional "market-mode" features
- **Hypothesis.** Auction/order-book comps (Optiver, Jane Street) win on systematic doublet `(x−y)/(x+y)`
  / triplet `(max−mid)/(mid−min)` imbalance over price/size columns + `market_urgency = spread × imbalance`,
  and per-timestamp cross-sectional means ("market mode"). We have OFI but not this combinatorial family
  nor basket-residualized *order-flow* (only residualized price).
- **New data.** None.
- **Method.** Generate doublet/triplet/urgency features over {bid/ask price, mid, microprice, bid/ask size}
  per sub-window; add per-bar cross-pair mean/std of returns/spreads/imbalances; express each pair *relative
  to* that cross-sectional mean (extend stat-arb basket from price to flow/imbalance).
- **Expected lift.** Overlaps some ruled-out work → modest, but near-zero cost on existing ticks. **Effort S · P2.**

## E16 — FALLBACK: change the instrument — re-point the pipeline at crypto 15m **[DEFERRED — majors-only scope, 2026-05-30]**
- **Hypothesis.** Our FX current best level is *because EURUSD is ~97% USD-factor and near-efficient*. Crypto is
  peer-reviewed *less efficient at 15/30/60m* (intraday momentum + reversal coexist; alts less efficient
  than BTC), has **real** (not proxy) order flow free from exchange APIs, a dominant lead asset (BTC), and
  orthogonal axes with no FX analog (funding rate, open-interest deltas, liquidation cascades, cross-venue
  dispersion). This is the strongest "less-efficient instrument" path to a longer-lived 15m edge.
- **New data.** Free: Binance/Bybit/OKX 1m OHLCV + L2 + trades; Tardis.dev (1st-of-month free) for
  historical L2; Coinglass funding/OI/liquidations.
- **Method.** Reuse `pipeline.py`/`crosspair.py`(BTC-as-leader)/`orderflow.py` with *real* signed flow +
  book imbalance; add funding/OI/liquidation/dominance/cross-venue features; explicitly test the
  intraday-momentum + reversal decomposition. Same strict OOS contract.
- **Differs from prior work.** Different, structurally-inefficient instrument; real order flow upgrades the
  null-lift FX OFI proxy. **Effort L · P2.**

## E17 — FALLBACK: change the horizon — extend / harvest the clean 3-second edge
- **Hypothesis.** Our one verified ≥75% (3s, 0.756 TEST / 0.809 OOS @0.05% cov) is real. A DeepLOB/TLOB-
  style encoder or microprice/integrated-OFI may *extend* it slightly (3–30s) and increase its coverage,
  feeding a faster execution book — accepting that 15m is the wrong target for microstructure.
- **New data.** None (ticks); CME 6E top-of-book could sharpen via the documented futures-lead-spot effect.
- **Method.** DeepLOB front-end on raw size tensors at 3–30s with LOBFrame-grade evaluation discipline;
  microprice/integrated-OFI features; measure accuracy-vs-coverage vs the existing 3s ensemble.
- **Note.** Reframes the horizon, not the goal; honest given the mechanistic decay. **Effort M · P2.**

---

# Explicitly DE-PRIORITIZED / do-not-repeat (with reason)
- More TA feature families, deeper/fancier architectures, plain Informer/Autoformer/FEDformer, zero-shot
  TSFMs — bottleneck is current best signal level, not the model; beaten by linear/LightGBM on FX.
- Anonymous tick-rule OFI refinements (value is in counterparty *identity*, unbuyable).
- Calibration (Platt/isotonic/temperature) to *raise* selective accuracy — provably order-preserving, cannot.
- COT / CESI / generic social-Twitter sentiment / RR as a 15m *trigger* — wrong cadence/horizon; at most slow conditioners.
- Predicting Deriv synthetic-index *direction* (CSPRNG martingale, ~0.50), unregulated OTC binary brokers (manipulated tape).
- Entropy/Hurst/VPIN/RQA/wavelet-energy as *direction* features — sign-blind by construction (use only as vol/regime gates).
- Buying enterprise spot-FX depth (LSEG/EBS) before the free CME-6E + crypto tests show depth buys 15m persistence.
- Triangular-arb profit, oil→CAD/gold→AUD commodity leads for EURUSD — mirage / ambiguous intraday causality.
