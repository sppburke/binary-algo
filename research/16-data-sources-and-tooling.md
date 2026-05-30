# Data Sources, Vendors & Tooling for the Missing 15-Minute FX Signals

*Research vector 16 — practical, actionable catalog of where to get the orthogonal data our
239-feature TA + cross-pair + 10s-OFI stack does not contain. Goal context: predict EURUSD (and other
USD majors) direction at a 15-minute horizon, >75% OOS. We have 10s OHLCV + raw sub-second top-of-book
ticks (bid/ask + quote sizes) for 7 USD pairs, train 2012-21 / val 2022-23 / test 2024-25 / 2026 held out.*

---

## TL;DR (most actionable findings)

- **True FX depth-of-book is the single biggest data gap, and it is gated by money, not availability.** Our
  Dukascopy/own ticks are *top-of-book only*. Real multi-level LOB for spot FX exists via **LSEG Tick History
  PCAP (FX Matching CLOB, full L1/L2/L3, nanosecond)** and is the gold standard, but it is enterprise-priced.
  **Databento does NOT yet sell EBS/spot-FX order book** — "EBS FX data" sits in the *Considering* bucket of
  their public roadmap, not shipping. So depth-FX is a *paid pilot* decision, not a free download. ([Databento roadmap](https://roadmap.databento.com/roadmap/ebs-fx-data), [LSEG Tick History PCAP](https://www.lseg.com/en/insights/fx/revolutionising-fx-price-transparency-with-tick-history-pcap))
- **The cheapest credible path to a genuine deep-LOB + imbalance signal is a crypto testbed, not FX.**
  Binance/Coinbase give free real-time L2 websockets; **Tardis.dev gives the first day of every month free**
  (full historical L2 incremental + snapshots) and academic plans from ~$300. Use crypto (less efficient,
  deeper public book) to *validate the methodology* of multi-level imbalance at minute horizons before
  paying for FX depth. Same caveat applies though: crypto LOB imbalance predictive power is also
  short-lived and decays with horizon, mirroring our own 3s-vs-15m finding. ([Tardis docs](https://docs.tardis.dev/downloadable-csv-files), [TDS: order-book imbalance crypto](https://towardsdatascience.com/price-impact-of-order-book-imbalance-in-cryptocurrency-markets-bf39695246f6/))
- **FX options risk-reversals / implied-vol skew are the most theoretically orthogonal signal we have not
  used** — they encode *forward-looking* directional risk pricing, not past returns. Free daily proxies are
  thin; real surfaces need LSEG/Bloomberg/Tradition. Evidence that RR predicts spot is real but *weak and
  low-frequency* (forward-bias horizon), so treat as a slow-moving conditioning feature, not a 15m trigger. ([ECB options sentiment](https://www.ecb.europa.eu/pub/pdf/other/mb200305_focus06.en.pdf), [Refinitiv IV forum](https://community.developers.refinitiv.com/questions/81902/how-to-get-historical-implied-volatility-data-of-f.html))
- **CFTC COT is free, clean, and trivially API-accessible (Socrata) — but it is weekly and lagged 3 days,
  so it can only ever be a slow regime/positioning context feature, never a 15m edge.** Low expected lift
  given we already have daily/weekly context. ([CFTC Public Reporting](https://publicreporting.cftc.gov/stories/s/Commitments-of-Traders/r4w3-av2u/))
- **GDELT is genuinely free at scale (BigQuery, 15-min update cadence) and is the most defensible *news/event*
  orthogonal source** — but the headline "Sharpe 5.87 EUR/USD" paper (arXiv 2505.16136) is a **single-author,
  daily-horizon, self-reported backtest** whose Sharpe is implausible for a tradable daily FX strategy. Mine
  the *data source and 15-min cadence*, not the result. ([arXiv 2505.16136](https://arxiv.org/abs/2505.16136))
- **Retail-positioning sentiment (OANDA order/position book, FXSSI, Myfxbook) is a real contrarian signal but
  is broker-scoped, coarse (15-min/5-min refresh), and has no clean free historical API.** Best treated as a
  Med-priority conditioning feature, not a primary signal; the contrarian effect is concentrated at extremes
  and turning points, not continuously at 15m. ([OANDA order book tool](https://www.oanda.com/bvi-en/lab-education/tools/order-book-position-book-tool/))
- **Dealer/customer signed order flow (Evans-Lyons school) is the one signal with the strongest academic claim
  to *cause* exchange-rate moves** — but the predictive flow is *customer* flow held by banks/CLS, essentially
  unavailable to us. Our 10s tick-rule OFI is the retail-accessible shadow of it, and we already found it adds
  ~0 lift. This confirms the gap is *data access*, not technique. ([NBER w23206 CLS](https://www.nber.org/system/files/working_papers/w23206/w23206.pdf), [Chan: FX order flow](http://epchan.blogspot.com/2018/02/fx-order-flow-as-predictor.html))
- **Economic-calendar APIs (Trading Economics, FXStreet, Finnhub free) unlock precise event *timing +
  surprise* (actual vs forecast), which is strictly more than the time-of-day proxy we already ruled out** —
  surprise magnitude is the missing piece. Low-cost, Med priority: most useful as an exclusion/gating filter
  around releases rather than a continuous predictor. ([FXStreet Calendar API](https://docs.fxstreet.com/api/calendar/), [Finnhub economic calendar](https://finnhub.io/docs/api/economic-calendar))

---

## Key findings (detailed, with inline citations)

### 1. True limit-order-book / depth FX data — the core gap

Our own ticks and Dukascopy are **best-bid/ask only**. Dukascopy explicitly *"does not save the full limit
order book history, but saves the history of Best Bid/Ask with the corresponding volumes,"* and depth entries
require orders >100K EUR to even appear and are platform-only, not in the downloadable feed
([Dukascopy market depth docs](https://www.dukascopy.com/wiki/en/development/strategy-api/practices/get-full-market-depth/)).
So we have *quote sizes at top-of-book*, which is a thin slice of a true LOB.

Where real spot-FX depth lives:

- **LSEG (Refinitiv) Tick History — PCAP**: packet-capture for **FX Matching** (LSEG's spot/forwards CLOB),
  delivering *"lossless, top-of-book and full-depth data (Levels 1, 2 and 3)"* with nanosecond timestamps,
  reflecting ~$460bn/day of liquidity. This is the institutional gold standard for spot-FX LOB
  ([LSEG Tick History PCAP](https://www.lseg.com/en/insights/fx/revolutionising-fx-price-transparency-with-tick-history-pcap),
  [Tick History factsheet](https://www.lseg.com/content/dam/data-analytics/en_us/documents/fact-sheets/final_re2664707_ent_tick_history_factsheet_a4_v4_web.pdf)).
  No public price; enterprise contract. Note: LSEG markets FX Matching as **EBS's main competitor venue** — EBS
  (now CME-owned) is the other primary interdealer CLOB.
- **Databento**: excellent for *futures* LOB (CME Globex MDP 3.0, full MBO/MBP-10) including **CME FX futures**
  (6E = EUR/USD future). But **spot-FX/EBS is not yet a product** — "EBS FX data" is in the *Considering(142)*
  list on their roadmap, i.e. requested-but-not-shipping ([Databento roadmap](https://roadmap.databento.com/roadmap/ebs-fx-data),
  [GLBX.MDP3 dataset](https://databento.com/datasets/GLBX.MDP3)). **Actionable nuance: CME 6E futures LOB IS
  buyable from Databento today** and is a legitimate, leakage-free proxy for EUR/USD depth/flow during CME
  hours — this is the cheapest *real multi-level book touching EUR/USD* available to us.
- **Integral / EBS / Reuters Matching**: institutional ECN/CLOB feeds, contract-only, not retail-accessible.
- **LOBSTER**: equities (Nasdaq ITCH) only — **not FX**. Useful only if we wanted an equity LOB testbed.

### 2. Crypto order book — the free, deep-LOB methodology testbed

Because deep FX LOB is paywalled, crypto is the rational place to *prove the method* first:

- **Binance/Coinbase native websockets**: free real-time L2 depth. Binance `@depth` stream is throttled to
  **100ms granularity** — fine for minute-horizon work, not for true microstructure
  ([Binance order book mgmt](https://developers.binance.com/docs/derivatives/usds-margined-futures/websocket-market-streams/How-to-manage-a-local-order-book-correctly)).
- **Tardis.dev**: historical tick-level incremental L2 + snapshots for Binance/Coinbase; **first day of every
  month is free, no API key**; academic/solo plans from a $300 minimum
  ([Tardis CSV docs](https://docs.tardis.dev/downloadable-csv-files),
  [Tardis billing](https://docs.tardis.dev/faq/billing-and-subscriptions)). CoinAPI/Amberdata are paid alternatives.
- **Reality check that mirrors our own finding**: crypto research repeatedly shows *"the price impact of the
  imbalance measure is short-lived and quickly deteriorates with the time horizon"*
  ([TDS](https://towardsdatascience.com/price-impact-of-order-book-imbalance-in-cryptocurrency-markets-bf39695246f6/)).
  This is the same decay we documented (55.3% next-tick → coin-flip by 1 min). So depth-LOB imbalance alone is
  unlikely to survive to 15m even in crypto; the testbed's value is validating *engineered* multi-level
  features (queue dynamics, depth-slope, cancel/replace intensity) that we cannot compute from top-of-book.

### 3. FX options — implied vol, risk-reversals, skew (most orthogonal)

Risk-reversals (IV of OTM calls minus puts) are a *forward-looking, market-priced* directional signal,
structurally independent of our return-history TA. The ECB notes options-based indicators *"assess how the
market sees the balance of risks between large appreciation and depreciation"*
([ECB](https://www.ecb.europa.eu/pub/pdf/other/mb200305_focus06.en.pdf)).
Academic evidence: RR correlates with the **forward bias** and changing expectations, but the documented
predictability is *low-frequency* (the forward-bias / carry horizon), not intraday
([daytrading.com RR](https://www.daytrading.com/risk-reversal),
[BIS vol-correlated currencies](https://www.bis.org/publ/confp01l.pdf)).
Data: Bloomberg/LSEG/Tradition/Data-In-Harmony sell ATM vol + 10Δ/25Δ RR & butterfly surfaces; **no robust
free historical surface** exists. Refinitiv RICs like `EUR3MR25=` give 3M 25Δ RR for those with a terminal
([Refinitiv forum](https://community.developers.refinitiv.com/questions/81902/how-to-get-historical-implied-volatility-data-of-f.html),
[Tradition FX options](https://www.traditiondata.com/products/fx-options/)).
*Closest free proxy:* CME FX options (on Databento/CME) → reconstruct a coarse RR from put/call IV. Verdict:
orthogonal and worth a **daily conditioning feature**, but do not expect a 15m trigger.

### 4. CFTC COT — free positioning, but weekly/lagged

CFTC publishes COT free via a **Socrata public-reporting API** (Legacy / Disaggregated / TFF datasets),
queryable/downloadable, public domain ([CFTC PRE](https://publicreporting.cftc.gov/stories/s/Commitments-of-Traders/r4w3-av2u/),
[CFTC data](https://www.cftc.gov/data)). It covers CME FX futures positioning (large speculators vs
commercials). Fundamental limit: **Tuesday snapshot, released Friday → ~3-day lag, weekly cadence.** It can
only be a slow regime feature. We already have daily/weekly context, so expected *incremental* lift is low.

### 5. News / sentiment — GDELT is the credible free option

- **GDELT 2.0**: free, open, **updates every 15 minutes**, with the GKG (themes, entities, tone). Available as
  15-min CSV chunks on Google Cloud Storage *and* via **BigQuery** (2015→present)
  ([DataResearchTools](https://dataresearchtools.com/gdelt-project-for-news-data-2026-free-alternative-to-newsapi/)).
  The 15-min cadence aligns exactly with our horizon — this is the strongest fit of any news source.
- **Alpha Vantage NEWS_SENTIMENT**: free tier, AI sentiment scores, JSON; coarse and rate-limited but a
  zero-cost starting point ([Alpha Vantage](https://www.alphavantage.co/documentation/)).
- **RavenPack (now "RavenPack Edge")**: institutional-grade, low-latency entity sentiment used by *"70%+ of
  the best-performing quant funds"*; full history + real-time. Quote-only pricing, expensive
  ([RavenPack](https://www.ravenpack.com/products/edge/data/news-analytics)).
- **NewsAPI / Newsdata.io**: cheap REST news but headline-level, weak for structured FX-entity sentiment.

### 6. Dealer / customer order flow — strongest theory, least accessible

The Evans-Lyons microstructure literature is the canonical evidence that **signed customer order flow is the
main proximate driver of exchange rates** and *"customer order flows have forecasting power for future changes
in the exchange rate"* ([Glasgow customer-flow study](https://www.gla.ac.uk/media/Media_125282_smxx.pdf),
[BIS bppdf 02j](https://www.bis.org/publ/bppdf/bispap02j.pdf)). Hasbrouck-Levich's **CLS settlement** work
confirms volume/liquidity structure but the data is bank-settlement-level, not a tradable real-time flow feed
([NBER w23206](https://www.nber.org/system/files/working_papers/w23206/w23206.pdf)). The predictive flow lives
inside banks (disaggregated customer flow); CLS aggregates are settlement, lagged, and not retail-licensable.
**This is the crux:** the signal that the academy says *works* is precisely the one we cannot buy, and our 10s
tick-rule OFI (the only proxy we can build) already added ~0 — consistent with the literature that *interdealer*
flow is far noisier than *customer* flow.

### 7. Economic-calendar APIs — timing + surprise (beyond our time-of-day proxy)

We already ruled out an *event-timing* proxy as redundant with time-of-day. But calendar APIs also deliver
**actual vs forecast vs previous**, i.e. the **surprise**, which time-of-day cannot encode:

- **Trading Economics API**: real-time calendar + millions of historical rows, subscribe-to-updates
  ([TE calendar API](https://docs.tradingeconomics.com/economic_calendar/snapshot/)).
- **FXStreet Calendar API v4**: OAuth2, **webhooks** for release updates, rich FX-relevant event meta
  ([FXStreet docs](https://docs.fxstreet.com/api/calendar/v4/introduction/)).
- **Finnhub**: free-tier economic calendar ([Finnhub](https://finnhub.io/docs/api/economic-calendar)).
- **RapidAPI "Economic Calendar"**: free tier, 5-min refresh
  ([RapidAPI](https://economic-calendar.horizonfx.id/)).

---

## Concrete techniques / features / architectures to try

1. **CME 6E (EUR/USD futures) MBP-10 from Databento → real depth features for spot.** Buy GLBX.MDP3 6E
   history. Engineer features we *cannot* compute from top-of-book: multi-level depth imbalance (L1-L5),
   depth slope (price-to-fill a fixed notional), book-pressure asymmetry, cancel/add intensity, and
   queue-depletion rate. Causally align to spot EURUSD at 15m. This is the only *buyable, real multi-level
   EUR/USD book*. Test whether 5-level imbalance survives longer than the top-of-book imbalance we already
   showed dies by 5 min.
2. **Crypto pre-validation harness (free first-of-month Tardis data).** Before paying for FX depth, replicate
   the deep-LOB feature set on BTC/ETH-USDT, fit the same LightGBM/GRU stack at 1m/5m/15m, and measure how
   feature importance decays with horizon. If multi-level features add nothing over top-of-book even in the
   *less efficient* crypto book, do **not** spend on FX depth — strong negative-result filter.
3. **FX options risk-reversal conditioning layer.** Build daily 25Δ RR + butterfly (skew/convexity) from CME
   FX options IV (cheapest) or a vendor surface. Use as **regime gates / interaction features**, not triggers:
   e.g. only take directional bets when 15m TA signal *agrees* with the sign of the RR-implied skew. Tests
   whether a forward-looking orthogonal prior sharpens our existing selective-prediction (the 0.632@0.2% result).
4. **Calendar *surprise* gating + a short post-release drift feature.** From Trading Economics/FXStreet,
   compute standardized surprise = (actual−forecast)/historical_std for high-impact USD/EUR releases. Add (a)
   an *exclusion* flag (don't predict inside ±N min of a high-impact release — likely improves hit-rate by
   dropping the worst regime) and (b) a *signed post-surprise drift* feature for the 15-60 min window. This is
   strictly richer than the time-of-day proxy already shown redundant.
5. **GDELT 15-min tone deltas as an orthogonal exogenous feature.** From BigQuery, per 15-min bin compute
   EUR/USD/ECB/Fed-themed article count, mean tone, tone dispersion, and Δtone vs prior bins. Lag-align
   causally. Cheap, genuinely orthogonal to price-TA; test for *incremental* AUC over the 239-feature stack.
6. **Retail-positioning extreme flag.** Scrape OANDA/FXSSI long% (15-min refresh) live-forward (no clean
   history, so this is a forward-only experiment). Encode only the *extremes* (>70/<30% one-sided) as a
   contrarian conditioning feature — the literature places the contrarian edge at extremes, not continuously.
7. **COT speculative-positioning regime feature.** Weekly large-spec net position (z-scored) from the CFTC
   Socrata API as a slow context input. Low cost, low expected lift; include only as a cheap ablation.

---

## Reported results & CREDIBILITY assessment

| Claim / source | What's claimed | Credibility | Leakage / overfit risk |
|---|---|---|---|
| **arXiv 2505.16136** (GDELT+FinBERT→XGBoost) | Sharpe **5.87 EUR/USD, 4.65 USD/JPY**, CAGR >50% FX, daily next-day direction, cost-adjusted, 5-fold expanding-window OOS ~2017-2025 ([abs](https://arxiv.org/abs/2505.16136)) | **LOW / HYPE-adjacent.** Single author (personal capacity), self-reported backtest, **not peer-reviewed**, promoted on the GDELT blog. A Sharpe ~5.9 on a *daily* directional FX strategy is far outside anything reproducible in liquid FX — credible daily FX strategies are Sharpe ~0.5-1.5. | **High.** Daily-sentiment→next-day-return setups are notorious for (a) sentiment-timestamp leakage (using same-day news that postdates the entry), (b) expanding-window CV that still leaks via feature scaling/selection, (c) cost models too optimistic. **Mine the DATA SOURCE + 15-min cadence; discard the headline number.** |
| **GDELT-FX macro forecasting** ("8/10 cases beat benchmark") ([arXiv 2009.14281](https://arxiv.org/pdf/2009.14281)) | Filtered GDELT sentiment improves macro forecasts | **MED.** Plausible, modest effect; consistent with sentiment as weak orthogonal signal. | Macro-forecast horizon, not 15m; modest and direction-of-effect only. |
| **FX risk-reversal predicts spot** ([ECB](https://www.ecb.europa.eu/pub/pdf/other/mb200305_focus06.en.pdf), [BIS](https://www.bis.org/publ/confp01l.pdf)) | RR encodes directional risk; correlates with forward bias | **MED-HIGH (but low-freq).** Well-established that RR carries info; predictability is at carry/forward-bias horizons, **not intraday**. | Low leakage risk; just *wrong horizon* for a 15m trigger. Use as slow conditioning. |
| **Customer order flow forecasts FX** (Evans-Lyons; [Glasgow](https://www.gla.ac.uk/media/Media_125282_smxx.pdf), [Chan](http://epchan.blogspot.com/2018/02/fx-order-flow-as-predictor.html)) | Signed *customer* flow forecasts exchange-rate changes | **HIGH (academically robust).** Canonical microstructure result. | Low leakage; **but the predictive flow is bank/customer-level, not retail-buyable.** Our interdealer 10s OFI proxy adding ~0 lift is *consistent* with this — it's the wrong flow, not the wrong method. |
| **Crypto LOB imbalance predicts price** ([TDS](https://towardsdatascience.com/price-impact-of-order-book-imbalance-in-cryptocurrency-markets-bf39695246f6/), [arXiv 2506.05764](https://arxiv.org/html/2506.05764v2)) | Depth imbalance precedes short-horizon mid moves | **MED-HIGH for microstructure, LOW for 15m.** Reproducible at seconds; *"short-lived, deteriorates with horizon."* | Matches our own decay finding — depth signal unlikely to reach 15m. |
| **OANDA/retail contrarian sentiment** ([OANDA](https://www.oanda.com/bvi-en/lab-education/tools/order-book-position-book-tool/), [arXiv 2507.03350](https://arxiv.org/pdf/2507.03350)) | Retail positioning is a contrarian signal | **LOW-MED.** Real effect at extremes/turning points; broker-scoped, coarse, much of the "95% lose" framing is marketing. | Effect concentrated at extremes; weak as a continuous 15m feature. No clean free history → forward-test only. |

**General skepticism note:** every "Sharpe >3 / 90% accuracy" FX-sentiment claim found in this search is
self-reported, non-peer-reviewed, or vendor marketing. None survive the bar set by our own strict OOS protocol.
The *credible, reproducible* findings (Evans-Lyons customer flow; RR forward-bias; crypto imbalance decay) all
point the same way: **the durable orthogonal signal is order flow / positioning, but the version that works is
gated behind institutional data access, and the freely-accessible versions decay before 15m.**

---

## Data sources needed (what / where / free vs paid / fidelity)

| Source | Signal unlocked | Free vs Paid | Latency / fidelity | Integration difficulty |
|---|---|---|---|---|
| **Databento GLBX.MDP3 (CME 6E)** | Real multi-level EUR/USD *futures* LOB (MBO/MBP-10) | Paid (usage-based, modest) | Nanosecond, full book; CME hours only | **Med** — clean API/SDK, but CME≠spot, need session alignment |
| **LSEG Tick History PCAP (FX Matching)** | True spot-FX L1/L2/L3 CLOB | Paid (enterprise) | Nanosecond, lossless, full depth | **High** — contract, onboarding, storage |
| **EBS (CME) / Integral / Reuters** | Interdealer spot CLOB | Paid (enterprise) | Top-tier | **High** |
| **Dukascopy / TrueFX / HistData** | Top-of-book bid/ask (+ Dukascopy sizes) | Free | ms ticks, 2009+/2012+; **no real depth** | **Low** (already have equivalent) |
| **Binance/Coinbase websockets** | Real-time crypto L2 depth | Free | 100ms (Binance depth); live only | **Low-Med** |
| **Tardis.dev** | Historical crypto L2 incremental + snapshots | **1st-of-month free**; plans ≥$300 | tick-level; next-day CSV | **Low-Med** |
| **CFTC COT (Socrata API)** | Spec/commercial FX-futures positioning | Free | Weekly, ~3-day lag | **Low** |
| **GDELT 2.0 (BigQuery / GCS)** | News tone, themes, entities | Free (BigQuery costs only) | **15-min** updates, 2015+ | **Med** (BQ + entity filtering) |
| **Alpha Vantage NEWS_SENTIMENT** | AI news sentiment | Free tier | rate-limited, coarse | **Low** |
| **RavenPack Edge** | Institutional entity sentiment | Paid (quote-only) | low-latency, full history | **High** |
| **FX options surfaces** (LSEG/Bloomberg/Tradition/CME) | ATM IV, 25Δ RR, butterfly (skew) | Paid (CME options cheapest) | daily→intraday | **Med-High** |
| **Trading Economics / FXStreet / Finnhub calendar** | Event timing + **surprise** (actual/forecast) | TE/FXStreet paid; Finnhub/RapidAPI free tier | real-time, webhooks | **Low-Med** |
| **OANDA / FXSSI / Myfxbook positioning** | Retail contrarian sentiment | Free (no clean history) | 5-15 min refresh, broker-scoped | **Med** (scrape; forward-only) |

---

## Relevance & priority for OUR project

Interaction with what we've already ruled out is the key filter. We've exhausted price-derived TA,
cross-pair lead-lag, **interdealer 10s OFI (≈0 lift)**, ensembles (~0.52-0.527 AUC), stat-arb, daily/weekly
context, exogenous peer features, and an **event-*timing* proxy (redundant with time-of-day)**. The durable
microstructure signal we found only lives at **3 seconds**. So the only data worth buying is data that is
*orthogonal to price* and *plausibly persists to minutes-hours*.

**HIGH priority**
- **CME 6E (Databento) deep-LOB features** — the only *buyable real multi-level EUR/USD book*. Directly tests
  whether depth (vs top-of-book) buys persistence beyond our 5-min decay wall. Concrete, leakage-safe, modest cost.
- **Crypto free-tier deep-LOB methodology testbed (Tardis)** — near-zero cost; decides whether deep-LOB is
  worth buying for FX at all. A strong *negative-result gate* before spending.
- **Calendar surprise gating + post-release drift** — strictly richer than the time-of-day proxy we discarded;
  the *surprise magnitude* is genuinely new information, and event-window exclusion likely lifts hit-rate.

**MED priority**
- **FX options RR/skew daily conditioning** — most theoretically orthogonal, but wrong frequency for a trigger;
  worth it as a regime gate on our existing selective-prediction (the 0.632@0.2% lever).
- **GDELT 15-min tone deltas** — free, orthogonal, cadence-matched; test for incremental AUC. Manage expectations.
- **Retail positioning extremes** — forward-only experiment; cheap to wire, contrarian edge only at extremes.

**LOW priority**
- **CFTC COT** — weekly/lagged, redundant with existing weekly context. Cheap ablation only.
- **LSEG/EBS/Integral enterprise spot-FX depth** — gold standard but enterprise cost; only justify *after* the
  6E + crypto tests show multi-level depth actually buys 15m persistence.
- **RavenPack** — expensive; only if GDELT shows a real news signal worth upgrading.

**The honest strategic read:** the academic evidence (Evans-Lyons customer flow) says the signal that works is
*customer* order flow — which we structurally cannot buy — and our null result on interdealer OFI is consistent
with that, not a refutation of the method. The realistic orthogonal bets are therefore (1) **deeper book** (6E,
gated by the crypto test), (2) **forward-looking options skew**, and (3) **event-surprise**, each as a
*conditioning* layer on the existing selective-prediction engine rather than a standalone 15m oracle. Treat any
external ">75%/Sharpe>3" FX claim as overfit until reproduced under our 2024-25 OOS / 2026-held-out protocol.

---

## Sources (annotated)

1. **Databento product roadmap — "EBS FX data"** — Databento, 2025 — https://roadmap.databento.com/roadmap/ebs-fx-data — Confirms spot-FX/EBS LOB is *not yet a product* (Considering bucket); futures LOB is.
2. **Databento GLBX.MDP3 dataset** — Databento — https://databento.com/datasets/GLBX.MDP3 — CME Globex full MBO incl. 6E EUR/USD futures; the buyable multi-level EUR/USD book.
3. **LSEG Tick History PCAP — FX Matching** — LSEG, 2024 — https://www.lseg.com/en/insights/fx/revolutionising-fx-price-transparency-with-tick-history-pcap — Institutional gold-standard spot-FX L1/L2/L3 nanosecond depth.
4. **Dukascopy full market depth docs** — Dukascopy — https://www.dukascopy.com/wiki/en/development/strategy-api/practices/get-full-market-depth/ — Documents that free Dukascopy is best-bid/ask, not full LOB.
5. **Tardis.dev downloadable CSV docs** — Tardis.dev — https://docs.tardis.dev/downloadable-csv-files — First-of-month free crypto L2 incremental + snapshots; the cheap deep-LOB testbed.
6. **Binance order-book management guide** — Binance — https://developers.binance.com/docs/derivatives/usds-margined-futures/websocket-market-streams/How-to-manage-a-local-order-book-correctly — Free L2 websocket; 100ms throttle caveat.
7. **Order-book imbalance in crypto (TDS)** — Towards Data Science — https://towardsdatascience.com/price-impact-of-order-book-imbalance-in-cryptocurrency-markets-bf39695246f6/ — "Impact short-lived, deteriorates with horizon" — mirrors our 3s→15m decay.
8. **CFTC Commitments of Traders (Public Reporting)** — CFTC, 2022+ — https://publicreporting.cftc.gov/stories/s/Commitments-of-Traders/r4w3-av2u/ — Free Socrata API; weekly, 3-day-lagged FX-futures positioning.
9. **arXiv 2505.16136 — Interpretable ML for Macro Alpha** — Y. Zhang, 2025 — https://arxiv.org/abs/2505.16136 — Source of the Sharpe-5.87 GDELT claim; data source useful, headline result not credible.
10. **GDELT as free NewsAPI alternative** — DataResearchTools, 2026 — https://dataresearchtools.com/gdelt-project-for-news-data-2026-free-alternative-to-newsapi/ — GDELT 15-min cadence + BigQuery access.
11. **arXiv 2009.14281 — Macro forecasting via news/emotion/narrative** — 2020 — https://arxiv.org/pdf/2009.14281 — Filtered GDELT sentiment beats benchmark in 8/10 cases (modest, macro horizon).
12. **ECB — currency options indicators for sentiment** — ECB Monthly Bulletin, 2003 — https://www.ecb.europa.eu/pub/pdf/other/mb200305_focus06.en.pdf — RR/skew as directional-risk gauge.
13. **Refinitiv IV/RR historical data forum** — LSEG Dev Community — https://community.developers.refinitiv.com/questions/81902/how-to-get-historical-implied-volatility-data-of-f.html — RR RIC structure (e.g. `EUR3MR25=`), terminal-gated.
14. **NBER w23206 — FX Market Metrics from CLS settlement** — Hasbrouck & Levich, 2017-21 — https://www.nber.org/system/files/working_papers/w23206/w23206.pdf — Settlement-level flow/liquidity; predictive flow not retail-accessible.
15. **Customer order flow in FX (Glasgow)** — https://www.gla.ac.uk/media/Media_125282_smxx.pdf — Evans-Lyons school: *customer* flow forecasts FX changes.
16. **FX order flow as a predictor** — E. Chan blog, 2018 — http://epchan.blogspot.com/2018/02/fx-order-flow-as-predictor.html — Practitioner view on signed-flow predictability and data access limits.
17. **FXStreet Calendar API v4** — FXStreet — https://docs.fxstreet.com/api/calendar/v4/introduction/ — OAuth2 + webhooks; actual/forecast/previous (surprise).
18. **Trading Economics Calendar API** — https://docs.tradingeconomics.com/economic_calendar/snapshot/ — Real-time + historical calendar with surprise data.
19. **Finnhub economic calendar** — https://finnhub.io/docs/api/economic-calendar — Free-tier calendar API.
20. **OANDA Order & Position Book tool** — OANDA — https://www.oanda.com/bvi-en/lab-education/tools/order-book-position-book-tool/ — Retail contrarian positioning; 5-15 min refresh, no clean free history.
21. **RavenPack Edge News Analytics** — RavenPack — https://www.ravenpack.com/products/edge/data/news-analytics — Institutional entity sentiment; expensive, quote-only.
22. **Alpha Vantage API docs** — https://www.alphavantage.co/documentation/ — Free-tier NEWS_SENTIMENT + FX rates.
23. **TrueFX / HistData free tick data** — https://www.truefx.com/truefx-historical-downloads/ , https://www.histdata.com/download-free-forex-data/ — Free top-of-book ms ticks (2009+); no depth.
24. **arXiv 2506.05764 — Microstructural dynamics in crypto LOBs** — 2025 — https://arxiv.org/html/2506.05764v2 — "Better inputs matter more than another hidden layer"; deep-LOB feature engineering guidance for the testbed.
