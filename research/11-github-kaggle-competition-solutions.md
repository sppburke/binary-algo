# GitHub Repos & Kaggle/Competition Winning Solutions for Financial Direction Prediction

Research vector: what features, models, target/CV designs actually win direction- and short-horizon-return
prediction in open competitions and open-source repos, what was reproducible vs leaked/overfit, and what
transfers to our 15-minute EURUSD direction goal.

Scope covered: Jane Street Market Prediction (2021), Jane Street Real-Time Market Data Forecasting (2024),
Optiver Realized Volatility Prediction (2021), Optiver "Trading at the Close" (2023-24), G-Research Crypto
Forecasting (2022), Ubiquant Market Prediction (2022), JPX Tokyo Stock Exchange Prediction (2022),
plus de Prado methodology and a representative academic FX-direction paper.

---

## TL;DR (most actionable for our 15m FX goal)

- **The single biggest, most credible lift in any of these comps came from ONLINE LEARNING, not a new feature.**
  The Jane Street 2024 winner's GRU gained **+0.008** (its largest single component, vs +0.001–0.002 for any
  feature/architecture change) purely from doing **one gradient step per day on newly-revealed labels** at inference
  time. We have never tried walk-forward online weight updates. This is the highest-value untried idea here.
  (evgeniavolkova `solution.md`, JS-2024.)

- **Auxiliary/multi-horizon targets help.** The same winner trained the same net to predict several *rolling-average*
  versions of the target (8-day and 60-day smoothings) as auxiliary heads, then combined them linearly → +0.001 CV/LB.
  For us: predict the 15m direction *plus* the 5m, 30m, 1h, and smoothed-return targets jointly (multi-task), which
  denoises the shared representation. Untried in our stack.

- **GBDTs (LightGBM) repeatedly beat or matched deep nets on tabular market data, with near-default hyperparameters.**
  G-Research crypto 2nd/3rd place: plain LightGBM, squared loss, *no regularization/feature-neutralization/ensembling*
  beyond GBDT itself; tuning only n_estimators/num_leaves/lr. This matches our finding that our trees cap ~0.527 AUC —
  the current best level is set by the *signal*, not the model. Stop spending effort on architectures; spend it on targets/CV/online-update.

- **Imbalance ratios are the workhorse feature family for auction/order-book data.** "Doublet" `(x−y)/(x+y)` over all
  price/size pairs, and "triplet" `(max−mid)/(mid−min)` over price/size triples, plus "market urgency"
  `price_spread × liquidity_imbalance`. We have OFI but NOT this systematic combinatorial-imbalance expansion over our
  bid/ask quote *sizes*. Cheap to add to our tick data; Med-priority test.

- **"More training data monotonically helped" in crypto (200 weeks) — but JS-2024 winner found data older than a regime
  break (`date_id<700`) was useless.** Conflicting evidence → for us, the regime-stability of pre-2015 EURUSD vs post-2015
  is an empirical question; test training-window length as a hyperparameter rather than assuming "more is better."

- **The most-cited "1st place" for an order-book comp (Optiver Realized Volatility) WON BY LEAKAGE**, not signal: it
  reverse-engineered the shuffled `time_id` ordering via tick-size→price reconstruction + nearest-neighbor Hamiltonian
  path, then built cross-sample "future-neighbor" features. **Not transferable to live trading.** Treat any
  competition writeup whose key trick is "recover the hidden ordering / index" as a leakage exploit, not alpha.

- **Cross-validation discipline is the real differentiator.** Winners use purged, embargoed, walk-forward, *grouped-by-time*
  CV with a gap between train and test folds (de Prado purged/combinatorial-purged CV). Several writeups explicitly note
  the model "cheats at the start of the test period" when there is no train/test gap. Our 2012-21/22-23/24-25 split is
  already strict; the actionable add is **per-fold gap + embargo** sized to the 15m label horizon.

- **Realistic OOS daily FX direction tops out ~55–59%, intraday lower.** A peer-reviewed-venue EURUSD paper
  (Guyard & Deriaz 2024) gets 58.52% one-day-ahead with stacking+PCA. No credible open source shows sustained
  >75% intraday directional accuracy without leakage. Our 75% target is far above any credible public result;
  the only honest path to it is selective prediction (bet rarely) or a genuinely new data source.

---

## Key findings (each with inline source)

### 1. Jane Street 2024 (Real-Time Market Data Forecasting) — the most relevant comp to us

This is the closest analog to our problem: anonymized high-frequency cross-sectional market features, predict a
weighted return responder, strict no-peek online inference API, scored by zero-mean weighted R². The published
solution by **Evgenia Volkova** (top-of-leaderboard, full reproducible code + `solution.md`) is unusually transparent.

Verbatim findings from `solution.md` (github.com/evgeniavolkova/kagglejanestreet):

- **CV:** time-series CV, 2 folds, validation = 200 dates (matching the public set). Additionally tested fold-0 model
  on the last 200 dates **with a 200-day gap** to simulate the private-set regime gap. Fold scores varied wildly
  (fold 0 = 0.0161, fold 1 = 0.0062) — i.e. signal is unstable across time, exactly our experience.
- **Sample:** dropped all data before `date_id=700` (before the cross-section stabilized at 968 symbols). "Using the
  entire dataset did not improve the score." → older data past a regime change is noise.
- **Features:** all raw features + 16 high-target-correlation features, plus two engineered groups: **(a) market
  averages** (mean per `date_id` and per `time_id` — i.e. cross-sectional/"market mode" features) and **(b) rolling
  mean & std over the last 1000 `time_id`s per symbol**. Plus `time_id` itself. Total feature lift only **+0.002**.
- **Model:** time-series **GRU** with sequence = one trading day. 3-layer GRU and a 1-layer-GRU+2-dense variant.
  "MLP, time-series transformers, cross-symbol attention and embeddings didn't work." (GRU > Transformer here.)
- **Auxiliary targets (multi-task):** trained separate base nets to predict `responder_7`, `responder_8`, and two
  *constructed* longer-rolling-average responders (≈8-day and ≈60-day smoothings of the base target), then a linear
  layer combined them → final prediction. Loss = sum of weighted zero-mean-R² over all responders. **+0.001 CV/LB.**
- **Online learning (the big one):** at inference, when new labeled data arrives, **one forward+backward pass updates
  the weights, lr=0.0003**, base-target loss only. "**This significantly improved performance on CV (+0.008).**"
  Day-level updates for nearly a year kept the model performing without any full retrain.
- **Ensemble:** 2 architectures × 3 seeds, simple unweighted average → LB 0.0112 vs best single 0.0105.

Why this matters: every component except online-learning gave +0.001–0.002. **Online learning gave ~4× the lift of
all feature engineering combined.** We have an analogous setup (continuous EURUSD with eventually-revealed labels) and
have *never* tried online/streaming weight updates.

### 2. Jane Street 2021 (Market Prediction) — supervised autoencoder + MLP

1st place (Yirun Zhang / "Cat's Trading"): a **supervised denoising autoencoder whose bottleneck feeds an MLP**, trained
*jointly* (reconstruction + supervised heads), inspired by "Deep Bottleneck Classifiers in Supervised Dimension
Reduction." Used Gaussian-noise injection (denoising), the multiple `resp` targets at different horizons as multi-label
outputs, and careful CV (purged group k-fold). (Kaggle writeup "Yirun's Solution (1st place): Training Supervised
Autoencoder with MLP".)

The Semper Augustus team's public repo (scaomath/kaggle-jane-street) is a candid record of **how unstable these
leaderboards are**: the same model swung between rank 75/4245 (top 1.77%) and 359/4245 (top 8.46%) across the live
period, finishing 241st. Their notes are a catalog of overfitting traps they hit and reverted ("a new de-noised target —
CV too good but leaderboard bad"; weight-zero-row tricks that improved CV but hurt LB). Direct evidence that
**CV-LB correlation is weak for this kind of data** — exactly the regime we're in.

### 3. Optiver "Trading at the Close" (2023-24) — order-book/auction imbalance features

Predict each stock's 10-min-to-close return *relative to a synthetic index*. The dominant theme across all top
solutions: **feature engineering on imbalance, and index-relative (cross-sectional) features.** LightGBM (often
ensembled with an MLP) was the standard winning model.

Concrete feature families used by strong solutions (fan2goa1 blog; liyiyan128 silver-medal repo; nimashahbazi):

- `liquidity_imbalance = (bid_size−ask_size)/(bid_size+ask_size)`
- `matched_imbalance = (imbalance_size−matched_size)/(matched_size+imbalance_size)`
- **`market_urgency = price_spread × liquidity_imbalance`** — "the strongest feature found in the public kernel."
- **Doublet imbalance:** for every pair `(x,y)` of price/size columns, `(x−y)/(x+y)`.
- **Triplet imbalance:** for every triple, `(max−mid)/(mid−min)` computed row-wise (numba-parallelized).
- **Global/cross-sectional features:** per-`time_id` (per-timestamp) means of features = the "market mode"; predict
  relative to the synthetic index. (Same idea as JS-2024's market-average features.)
- Lagged features of prices/sizes per stock over windows (1,2,3,10).
- "Feature is all you need": baseline LightGBM *without* feature engineering scored worse than predicting constant 0;
  *with* the imbalance features it became competitive. (liyiyan128 README.)

### 4. Optiver Realized Volatility (2021) — CREDIBLE features, but the "1st place" trick was leakage

Legitimate, transferable feature ideas (winning-tier solutions): WAP (weighted average price) from top-2 book levels,
log-returns of WAP, realized vol = sqrt(sum of squared log-returns) over sub-windows, bid-ask spread, depth/size
aggregations, and the same per-`time_id` cross-sectional aggregations.

**The leakage caveat (critical for credibility assessment):** the headline "1st Place Solution — Nearest Neighbors" and
the 7th-place (michaelpoluektov/orvp) solution both won largely by **reverse-engineering the deliberately-shuffled
`time_id` ordering**: reconstruct real prices from normalized prices + tick size, build a graph where edge weight =
L2 distance between end-prices of one window and start-prices of another, approximate a shortest Hamiltonian path
(KNN to 6 nearest nodes + brute force) to recover chronological order, then create features using *neighboring*
(temporally adjacent, partly future) windows. The 7th-place author explicitly states: "**despite the high leaderboard
score, the real-world applications of this submission are limited.**" → This is a competition-artifact exploit, **not a
tradable signal.** Lesson for us: discount any reported result whose core mechanism is recovering a hidden index/order.

### 5. G-Research Crypto Forecasting (2022) — minute-level multi-asset return forecasting (closest market type to FX)

14 crypto assets, forecast short-horizon residualized returns, scored by weighted Pearson correlation. Most relevant
because it's minute-resolution, multi-asset, low-signal — like FX majors. Findings (kaggle.curtischong.me summary of
2nd/3rd/7th/9th solutions + discussion 323098):

- **Model:** plain **LightGBM, squared loss. No ensembling beyond GBDT, no regularization, no augmentation, no feature
  neutralization** ("it didn't help CV"). Only tuned n_estimators, num_leaves, lr.
- **CV:** 6-fold **walk-forward grouped-by-timestamp** CV; 40-week train folds, 40-week test folds, **1-week gap**
  between train and test ("with no gap, a model can cheat at the start of the test period"). Folds overlapped to stay
  long while keeping 6 folds (lower variance). No seed-averaging (CV variance already low).
- **Training window:** "scores just kept improving with longer training data" → trained on the *entire* dataset, no gap
  before final test. (Contrast with JS-2024's pre-700 drop — regime-dependent.)
- **Features:** top teams were deliberately vague (host wouldn't open-source alpha). Disclosed: Hull moving average was a
  top feature; Fibonacci-spaced lag windows [55,210,340,890,3750]; one team built **3 LightGBMs for up/down/stable market
  regimes** and averaged. The 7th-place team added **only one feature: time-of-day** and competed on modeling alone.
- "Timeseries learning is essentially supervised learning that respects causality."

### 6. Ubiquant / JPX — cross-sectional equity ranking (less relevant, but confirms patterns)

- **Ubiquant 1st** ("Our Betting Strategy"): anonymized features, predict per-asset return; large ensemble of FFNN/CNN/
  Transformer; emphasis on **market-average features** for the anonymized set and careful bagging across seeds/time —
  again the cross-sectional "market mode" feature and seed/time ensembling.
- **JPX:** rank ~2000 stocks; winners used straightforward TA (RSI, VWAP, multi-window EMA), **per-sector models**, and
  optimized the *ranking/portfolio* objective (top-200 minus bottom-200 spread) rather than raw direction. Lesson:
  matching the model's loss to the *economic* objective (spread/Sharpe) beat optimizing raw accuracy.

### 7. Methodology backbone: de Prado (Advances in Financial ML)

Repeatedly the implicit standard behind winning CV: **purged k-fold** (drop train obs whose label horizon overlaps the
test window), **embargo** (drop a buffer of obs right after each test fold to kill autocorrelation leakage),
**combinatorial purged CV** (multiple backtest paths for a distribution of OOS scores, not one), **triple-barrier
labeling** (label by which of profit-take / stop-loss / time-out is hit first), and **meta-labeling** (a second model
decides *whether to act* on the primary signal's call — precision filter). (de Prado, "10 Reasons Most ML Funds Fail";
Wikipedia "Purged cross-validation".)

---

## Concrete techniques / features / architectures to try (implementable)

1. **Walk-forward online learning (HIGHEST PRIORITY).** Take our best GRU/LightGBM. After each 15m bar's label
   becomes known (15m later), do a small online update: for a net, one SGD step lr≈1e-4–3e-4 on the new (or last-N)
   labeled rows; for GBDT, periodic warm-start/`refit` or a small online learner (e.g. river/`linear` online head on
   GBDT-leaf embeddings). Evaluate strictly walk-forward on val 2022-23 first. This is the only technique here that gave
   a *large* lift in a directly analogous comp.

2. **Multi-task / auxiliary-horizon targets.** Single shared trunk → multiple heads predicting direction/return at
   5m, 15m, 30m, 1h, plus *smoothed* (rolling-mean) versions of the 15m target. Sum the per-head losses. Use the 15m
   head at inference. Denoises the representation (JS-2024 +0.001, JS-2021 used multi-resp heads).

3. **Supervised denoising autoencoder → classifier** (JS-2021 winner). Train an AE with Gaussian-noise-corrupted inputs
   + a supervised head off the bottleneck, jointly. Use the bottleneck as denoised features into your existing GBDT.
   Targets the "97% USD-factor / low SNR" problem directly via learned denoising rather than hand TA.

4. **Combinatorial imbalance expansion on quote SIZES (we have tick bid/ask sizes).** Systematically generate:
   doublet `(x−y)/(x+y)` over all {bid_price, ask_price, mid, micro-price, bid_size, ask_size} pairs;
   triplet `(max−mid)/(mid−min)`; and **market urgency = spread × (bid_size−ask_size)/(bid_size+ask_size)**, aggregated
   over multiple sub-windows inside the 15m bar. We have OFI but not this combinatorial family or the micro-price.

5. **Cross-sectional "market-mode" features across the 7 USD pairs at each timestamp.** Per-bar mean/std of returns,
   spreads, and imbalances across all 7 pairs; then express each pair's features *relative to* that cross-sectional mean
   (USD-basket residual, which we have, but extend to residualized *order-flow* and *imbalance*, not just price). This is
   the winning "predict relative to the synthetic index" idea (Optiver Close, Ubiquant) applied to our basket.

6. **Regime-specialized expert models + gate.** Three models for high-vol / low-vol / trending regimes (or vol-bucketed),
   averaged or gated by a regime classifier (G-Research crypto). Cheap; may rescue accuracy in specific regimes that our
   single model washes out.

7. **Match the loss to the objective.** If the deployment is "bet only when confident," train with the *selective*
   objective directly (e.g. cost-sensitive / abstain-aware loss, or meta-labeling: model A proposes direction, model B
   predicts P(A is correct) and we trade only when B is high). JPX/de Prado pattern; directly serves our 75%-at-low-
   coverage goal better than maximizing global accuracy.

8. **Purged + embargoed walk-forward CV with a gap sized to the label horizon.** Add a 15m (or longer) embargo between
   train and val/test boundaries inside every fold. Even our clean year-split should embargo around the boundary.

9. **Stop tuning architectures; treat the model as roughly fixed.** Across every comp, LightGBM≈GRU and beat
   Transformers/attention on this data. Our ~0.527 AUC current best is set by the signal, not the model. Reallocate effort to items 1-7.

---

## Reported results & CREDIBILITY assessment

| Source | Reported result | Credible & reproducible? | Leakage / hype risk |
|---|---|---|---|
| JS-2024 (Volkova) | LB R² 0.0112, full code + solution.md | **High** — code public, CV→LB correlation shown, gap-test done | Low. Online-learning lift is honest (uses only past-revealed labels). |
| JS-2021 1st (Zhang) | 1st/4245, supervised AE+MLP | **High** — method well-documented, widely reproduced | Low method; but LB *instability* documented (rank swung 1.8%→8.5%). |
| Optiver Close top solutions | LB ~5.32–5.34, LightGBM+MLP | **High** for the *features* (imbalance/urgency) | Low; features are genuine microstructure. |
| Optiver Realized Vol "1st place NN" | public RMSPE ~0.20 | **Features yes, headline result NO** | **High leakage** — wins by recovering shuffled time_id order; author admits "real-world applications limited." |
| G-Research Crypto 2nd/3rd | top-tier weighted-Pearson | **High** — plain LightGBM, disciplined gapped walk-forward CV | Low; teams candid that signal is weak and tuning minimal. |
| Ubiquant 1st | 1st place | Medium — method described, anonymized data limits transfer | Low leakage; relevance limited. |
| GitHub FX-LSTM repos ("58% profitability", "55-60%") | 55–60% direction | **Low-Medium** — mostly small, weak/absent OOS protocol, MinMaxScaler-before-split leakage common | High overfit/leakage risk; treat as folklore, not evidence. |
| Guyard & Deriaz 2024 (arXiv 2409.04471, MLMI conf) | **58.52%** 1-day EURUSD, 32.48% 2022 return | Medium-High — venue-reviewed, stacking+PCA, daily horizon | Single-year return figure is fragile; accuracy figure plausible as a *daily* best-so-far. |

**Overarching credibility rule learned:** in financial-prediction competitions, the spectacular headline numbers almost
always come from (a) recovering a hidden ordering/index the host accidentally left exploitable (Optiver RV), or
(b) pre-split scaling / target leakage in toy GitHub repos. The *honest* numbers cluster at: order-book direction
55% next-tick decaying fast (matches our own finding), daily FX direction ~55–59%, and minute-level multi-asset return
correlations small-but-positive. **No credible source shows sustained >75% intraday FX direction without leakage.**

---

## Data sources needed

- **None of the credible techniques require new data — they re-use what we already have** (tick bid/ask + sizes for 7
  USD pairs, 10s OHLCV). Online learning, multi-task targets, AE denoising, combinatorial imbalance, cross-sectional
  basket features all run on existing data. This is the cheapest set of experiments available.
- To *replicate the comps themselves* (optional, for benchmarking our pipeline against a known-signal dataset):
  - Jane Street 2024 data: free via Kaggle (`jane-street-real-time-market-data-forecasting`), ~100 GB RAM / 12 GB GPU
    to run Volkova's code. Useful as a sanity check that our online-learning harness actually delivers the +0.008.
  - G-Research crypto, Optiver Close, Ubiquant, JPX datasets: all free on Kaggle. Minute/auction resolution.
- **Reference code to lift directly (free, MIT/Apache-ish):** evgeniavolkova/kagglejanestreet (online learning + GRU
  + multi-task), liyiyan128/optiver-trading-at-the-close and the fan2goa1 blog (numba imbalance/triplet/urgency feature
  code), scaomath/kaggle-jane-street (supervised AE+MLP reference). de Prado: `mlfinlab`-style purged/embargoed CV and
  meta-labeling (open implementations exist).
- Fidelity note: competition data is already cleaned/aligned; our raw tick data is *higher* fidelity than most of these,
  so feature ideas should port up, not down.

---

## Relevance & priority for OUR project

Ranking against what we've already exhausted (239 TA features, cross-pair lead-lag, 10s-OFI, GBDT/NN ensembles capping
~0.527 AUC, stat-arb basket, calendar proxy, selective prediction at ~0.632@0.2%):

**HIGH**
- **Online / walk-forward weight updates** (item 1). Untried, largest credible lift in the most analogous comp, uses
  existing data. Single best bet in this whole vector.
- **Meta-labeling for selective prediction** (item 7). Directly improves our existing best lever (bet-rarely-at-high-
  confidence); reframes "75% accuracy" as a precision-at-coverage problem with a dedicated second model. de Prado-grade.
- **Multi-task auxiliary-horizon + smoothed targets** (item 2). Cheap, additive, denoises; complements (not duplicates)
  our existing features.

**MEDIUM**
- **Supervised denoising autoencoder features** (item 3) — new *representation*, not a new raw feature; may extract
  residual signal our hand-built TA misses. We've done plain NNs but not joint AE+denoising+supervised bottleneck.
- **Combinatorial imbalance + micro-price + market-urgency on quote sizes** (item 4) — we have OFI but not this exact
  family; some overlap with what we ruled out, so expect modest lift, but it's near-zero cost on existing ticks.
- **Cross-sectional basket-residualized order-flow/imbalance** (item 5) — extends our existing stat-arb basket from
  price to flow; partially overlaps ruled-out lead-lag, so Medium not High.
- **Regime-specialized experts + gate** (item 6).
- **Purged/embargoed gap in CV** (item 8) — cheap correctness/robustness fix; won't raise accuracy but prevents us
  fooling ourselves (and we should confirm our current splits embargo the 15m horizon at boundaries).

**LOW**
- More TA features, deeper/fancier architectures (Transformers/attention), more cross-pair lead-lag — all either
  already exhausted or shown across these comps to *not* beat LightGBM/GRU on this data type.
- Chasing the Optiver-RV "nearest-neighbor" trick or GitHub "58%/90%" repos — leakage/overfit, not transferable.

**Read from the evidence:** every credible competition indicates our current best level is set by the *signal*, not the model.
The realistic, honest path to anything near 75% is selective prediction (meta-labeling at low coverage) and online
adaptation — not another feature family. A flat-out >75% *unconditional* 15m direction rate has no credible precedent
in any of these solutions.

---

## Sources (annotated)

1. **Evgenia Volkova — Jane Street 2024 solution** (`solution.md`, github.com/evgeniavolkova/kagglejanestreet, 2024) —
   https://github.com/evgeniavolkova/kagglejanestreet — *Most relevant: transparent code showing online learning gave
   +0.008 (vs +0.001–0.002 for everything else), multi-task auxiliary rolling-average targets, GRU>Transformer.*
2. **Yirun Zhang — Jane Street 2021 1st place: Supervised Autoencoder + MLP** (Kaggle writeup, 2021) —
   https://www.kaggle.com/competitions/jane-street-market-prediction/writeups/cats-trading-yirun-s-solution-1st-place-training-s — *The canonical denoising-AE→MLP architecture for low-SNR market features.*
3. **Semper Augustus (Cao/McBride-Ellis/Zheng) — Jane Street 2021 repo** (github.com/scaomath/kaggle-jane-street) —
   https://github.com/scaomath/kaggle-jane-street — *Honest live-LB log showing rank swinging 1.8%↔8.5%; catalog of
   overfitting traps; evidence CV-LB correlation is weak for this data.*
4. **fan2goa1 (Zifan Zheng) — Optiver Trading at the Close walkthrough** (blog, 2023) —
   https://fan2goa1.github.io/mkdocs-material/blog/2023/12/24/kaggle-optiver---trading-at-the-close/ — *Concrete numba
   code for doublet/triplet imbalance, market_urgency, synthetic-index weights; purged k-fold.*
5. **liyiyan128 — Optiver Close silver-medal repo** (github) — https://github.com/liyiyan128/optiver-trading-at-the-close
   — *"Feature is all you need"; baseline LGBM w/o FE < constant-zero; imbalance feature taxonomy.*
6. **nimashahbazi — Optiver Close LSTM/ConvNet repo** (github) — https://github.com/nimashahbazi/optiver-trading-close —
   *Shows raw-feature NN + minimal imbalance features reaching competitive LB; residual NN block.*
7. **michaelpoluektov — Optiver Realized Vol 7th place (orvp) README** (github) — https://github.com/michaelpoluektov/orvp
   — *Explicitly documents the time_id-reordering leakage exploit and admits "real-world applications limited."*
8. **Optiver Realized Vol "1st place — Nearest Neighbors" discussion** (Kaggle disc. 274970, 2021) —
   https://www.kaggle.com/competitions/optiver-realized-volatility-prediction/discussion/274970 — *Headline winner;
   credible features but leakage-driven headline result.*
9. **Curtis Chong — G-Research Crypto Forecasting solutions summary** (kaggle.curtischong.me, 2022) —
   https://kaggle.curtischong.me/competitions/G-Research-Crypto-Forecasting — *2nd/3rd/7th/9th solutions: plain LightGBM,
   gapped walk-forward grouped CV, longer-data-helps, regime-expert models, minimal feature set.*
10. **G-Research — competition wrap-up** (gresearch.com, 2022) —
    https://www.gresearch.com/news/wrapping-up-the-g-research-crypto-forecasting-competition/ — *Host perspective on a
    minute-level multi-asset low-signal forecasting comp (closest market type to FX).*
11. **k-i-y — Ubiquant Market Prediction 1st place "Our Betting Strategy"** (Kaggle writeup, 2022) —
    https://www.kaggle.com/competitions/ubiquant-market-prediction/writeups/k-i-y-1st-place-solution-our-betting-strategy
    — *Cross-sectional market-average features + seed/time ensembling on anonymized return prediction.*
12. **J-Quants — JPX Tokyo Stock Exchange Prediction winners repo** (github.com/J-Quants/JPXTokyoStockExchangePrediction)
    — https://github.com/J-Quants/JPXTokyoStockExchangePrediction — *Per-sector models, TA features, ranking/spread
    objective; matching loss to economic objective.*
13. **Marcos López de Prado — "The 10 Reasons Most ML Funds Fail"** (GARP whitepaper) —
    https://www.garp.org/hubfs/Whitepapers/a1Z1W0000054x6lUAA.pdf — *Purged/embargoed CV, triple-barrier, meta-labeling,
    combinatorial purged CV — the methodology behind credible OOS validation.*
14. **Purged cross-validation** (Wikipedia) — https://en.wikipedia.org/wiki/Purged_cross-validation — *Definitions of
    purging + embargo for time-dependent labels; directly applicable to our 15m-horizon CV.*
15. **Guyard & Deriaz — "Predicting Foreign Exchange EUR/USD direction using machine learning"** (arXiv 2409.04471,
    MLMI 2024) — https://arxiv.org/abs/2409.04471 — *Venue-reviewed EURUSD direction: 58.52% one-day-ahead via
    stacking+PCA; a realistic credible current-best level for daily FX direction.*
16. **shenrunzhang/forex & assorted FX-LSTM GitHub repos** — https://github.com/shenrunzhang/forex — *Representative of
    the "55–60% direction" GitHub genre; small, weak OOS protocol, common pre-split scaling leakage → folklore, not
    evidence.*
17. **Bojer & Meldgaard — "Learnings from Kaggle's Forecasting Competitions"** (arXiv 2009.07701) —
    https://arxiv.org/pdf/2009.07701 — *Cross-competition synthesis: global models, gapped holdout validation, GBDT
    dominance, leaderboard-overfitting risk.*
