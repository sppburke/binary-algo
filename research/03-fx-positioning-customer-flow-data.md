# FX Dealer/Customer Order Flow, Positioning & Sentiment as Directional Signals

Research vector for the binary-algo 15-minute EURUSD direction problem. Focus: do real-money customer/dealer order flow, futures positioning (COT), and retail sentiment/order-book data provide an *orthogonal, lagged, out-of-sample* directional edge at minutes-to-hours horizons that we have not already captured with TA + tick microstructure?

Bottom line up front: the order-flow literature is the single most robust source of FX *predictability* known to academia — but almost all of it is (a) **contemporaneous** explanatory power, not lagged forecasting, and (b) at **daily-to-monthly** horizons, not 15 minutes. The genuinely 15m-relevant, accessible, orthogonal datasets are **retail broker positioning/order-book data** (OANDA Order/Position Book, FXSSI/Myfxbook/IG aggregates) and, if budget allows, **CLS settlement flow**. None is a documented path to 75% yet, but retail order-book data is cheap, truly orthogonal to our TA/microstructure stack, and worth a rigorous test.

## TL;DR

- **Order flow explains FX, but mostly contemporaneously and at daily+ horizons.** Evans-Lyons: interbank flow gives R² of 60-80% on *concurrent* 1-day returns; end-user (customer) flow *forecasts* returns mainly at the **1-month** horizon. This is not a 15m signal as-is (Lyons 2003 executive summary, CEPII).
- **The forecasting power is real and not just "flow predicting flow"** — ~2/3 of end-user flow's predictive power relates to subsequent price moves unrelated to subsequent flow (Lyons 2003). But the data that carries it (segmented customer flow) is bank-proprietary and slow.
- **Segment matters and is the key reusable idea:** *financial-institution* flow drives abrupt short-horizon moves; *non-financial corporate* flow drives longer-horizon trends (Lyons 2003). If you can get segmented flow, the financial-segment component is the part with any chance of intraday relevance.
- **COT (CFTC Commitments of Traders) is structurally unusable for 15m:** weekly, as-of Tuesday, released Friday 3pm CT — a 3-day-stale, once-a-week signal. Useful only as a slow positioning/regime feature, not a 15m directional predictor. Do not spend time backtesting it for this horizon.
- **Retail broker positioning/order-book data is the one accessible, intraday, orthogonal dataset.** OANDA's forexlabs API exposes an **Order Book** (pending orders) AND **Position Book** (open positions) bucketed by price level, updated every **5 min (premium) / 15 min (free)** — i.e. a live map of where retail stops/limits and crowded positions sit. This is *not* in our current feature set and is mechanistically distinct from TA.
- **Retail sentiment as a contrarian signal is real but weak and slow-acting.** Academic + practitioner consensus: retail FX crowds are return-contrarian and "uninformed"; extremes (>75% one-sided) mark eventual reversals, but timing is poor and edge is at hours-to-days, not 15m. Best used as a conditioning/regime feature, not a standalone 15m predictor.
- **A March 2025 peer-reviewed study (J. Int. Financial Markets) on intraday retail EURUSD order flow** finds retail investors are return-contrarian and that "simple strategies that exploit retail intraday order flow can be profitable" — the most directly on-point modern result, but its profitability is not demonstrated at a fixed 15m bar-classification accuracy and risks the usual costs/leakage caveats.
- **The credible, reproducible win here is the price-level order-book structure, not the aggregate long/short ratio.** Crowded retail stop clusters above/below spot are a liquidity map that can predict short-horizon *path* (stop-run / mean-revert) better than a single sentiment scalar.

## Key findings

### 1. The Evans-Lyons "order flow" canon: huge R², but contemporaneous and daily+

Evans & Lyons (2002, *Journal of Political Economy*; NBER w7317) is the foundational result: a model of *daily* DM/USD and JPY/USD changes using interdealer order flow produces R² above 50%, vastly beating macro models and the random walk in out-of-sample short-horizon forecasts. Price impact: ~$1bn net dollar purchases moves DM/USD by ~1 pfennig. (NBER w7317; web search summary.)

Critically, Lyons' own 2003 synthesis ("Explaining and Forecasting Exchange Rates with Order Flows", *Économie Internationale* 96, CEPII, fetched and read) lays out the seven lessons precisely — and they are sobering for a 15m goal:

> "Interbank transaction flows have substantial power to **explain concurrent 1-day returns**. R² statistics using direct interbank flows ... in 60-80 percent range (Evans and Lyons 2002c)."
> "End-user flows have substantial power to **forecast subsequent 1-month returns**."
> "Different flow segments affect the exchange rate at different horizons. Longer horizon price moves appear more closely tied to **non-financial corporate flows**. Abrupt short-horizon price moves are more closely tied to **financial institution flows**."
> "Flows' forecasting power is not simply flow forecasting future flow. **Two thirds** of forecasting power relates to subsequent price movements that are **unrelated to subsequent flow**."

Takeaways for us:
- The famous R² numbers are *contemporaneous*. To predict the next 15m you need *lagged* flow, and the lagged/forecasting evidence lives at the **1-month** end-user horizon — far outside our window.
- The only segment tied to *short-horizon* moves is **financial-institution flow**, which is exactly the bank-proprietary, hard-to-get data.
- This is consistent with your own mechanistic finding: true imbalance information decays fast. Lyons' framework says the *persistent* price-relevant flow is monthly corporate hedging/repatriation, not minutes.

### 2. Customer order flow at the dealer level: weak short-run Granger causality

The Glasgow customer-order-flow study and related dealer-perspective work (European Journal of Finance, Vol 17 No 2) find that in cointegration, exchange rates relate positively to *financial* customer orders and negatively to *commercial* orders — but **in the short run, neither financial nor commercial customer order flow Granger-causes returns**, so "dealers may find the practical value of order flow for forecasting purposes to be negligible." (web search summary of Glasgow PDF & EJF abstract.) This is a direct warning that even with real customer flow, *lagged* short-horizon predictability is weak.

### 3. CLS settlement flow: the best "real" flow data, but expensive, segmented, 5-min, and contemporaneous

CLS settles >50% of global FX (~$1.5tn/day, ~25,000 participants, ~500k trades/day). Its **CLSMarketData FX Flow** product gives directional net volume **by counterparty type** (banks, funds, non-bank financials, corporates) and by market-maker vs price-taker ("coreness" algorithm), at frequencies **intraday / daily / monthly** — CLS states data is captured at **5-minute** intervals. (CLS product page, fetched; NBER w23206 Hasbrouck-Levich.)

Caveats for 15m:
- **Latency/settlement lag:** CLS sees *settlement* instructions, not real-time execution; the flow is informative but the public/commercial feeds are not a low-latency execution signal. The academic value (Hasbrouck-Levich) is liquidity/metrics and *contemporaneous* flow-return relations, not lagged 15m forecasting.
- **Cost & access:** institutional, licensed via CLS / Registered Vendor Program (historically distributed via Quandl/Nasdaq Data Link). Not free, likely $$$.
- **Where it could help:** the *segmented* nature (corporate vs fund vs NBFI directional net) is the one thing that could be orthogonal to your microstructure stack. A daily/4h corporate-vs-financial net-flow feature could be a *regime/drift* conditioner, but is unlikely to classify a 15m bar to 75%.

### 4. COT (CFTC Commitments of Traders): too slow, wrong shape for 15m

COT is weekly: positions as-of **Tuesday**, released **Friday 3pm CT** (QuantifiedStrategies, fetched). Legacy + the Traders-in-Financial-Futures (TFF) report split dealer/asset-manager/leveraged-funds/other. It is a genuine positioning dataset and *does* carry slow contrarian/extreme-positioning information for weekly-to-monthly currency-futures direction — but as a 15m predictor it is a single number that updates once a week with a 3-day publication lag. **Do not backtest COT at 15m.** At most it is a weekly regime feature (e.g., "leveraged funds at positioning extreme") that you broadcast onto intraday bars — and even that is mostly redundant with the slow trend/vol-regime features you already have.

### 5. Retail positioning & sentiment: the accessible, orthogonal, intraday-capable angle

This is where the actionable opportunity is, because it is (a) free or cheap, (b) genuinely intraday (5-15 min refresh), and (c) mechanistically distinct from TA/microstructure.

**Mechanism (well-established, contrarian):** retail FX clients are "uninformed," return-contrarian (they fade moves / buy dips and sell rallies), and herd. IG/DailyFX explicitly market their Client Sentiment on this contrarian basis; the consensus across OANDA, IG, FXSSI, earnforex is that extreme one-sidedness (>~75-80% long or short) precedes reversals — but with poor timing and long holding-out in trends. (IG/DailyFX, OANDA, earnforex, web search summaries.)

**Modern academic confirmation (most on-point):** "News and intraday retail investor order flow in foreign exchange markets" (*Journal of International Financial Markets, Institutions & Money*, Vol. 98, March 2025; S1042443125000368). Uses a **proprietary intraday dataset of aggregate retail long/short positions in EUR/USD**. Findings (web search summary; abstract gated behind Cloudflare/RePEc 404 at fetch time):
- Retail investors are **return-contrarian**; contrarian behavior is driven mainly by **lagged returns**, not fundamentals.
- They "lack the skills to interpret fundamental information in scheduled macro news ... rather follow the sentiment in the news."
- **"Simple trading strategies that exploit retail intraday order flow can be profitable."**

This is the single best modern citation that intraday retail flow has exploitable, lagged, directional content in EURUSD specifically — your exact instrument.

**The richer signal — order/position BOOK, not just the ratio.** The aggregate long/short % is a scalar. Far more information is in the **price-level distribution** of pending orders and open positions, which is a liquidity map:
- **OANDA forexlabs API** (`oandapyV20.endpoints.forexlabs`) exposes `OrderbookData` (`labs/v1/orderbook_data`), `HistoricalPositionRatios`, `CommitmentsOfTraders`, `Calendar`, `Spreads`, `Autochartist`. The `OrderbookData` response is bucketed by price with four fields per price point (verified from the API wrapper docs):
  - `ol` = open *orders* long, `os` = open *orders* short, `pl` = open *positions* long, `ps` = open *positions* short (percentages).
  - Example bucket: `"1.23": {"ps":1.2155,"ol":0.3871,"os":0.2615,"pl":0.5633}` with a top-level `rate` and unix timestamp.
- This tells you where retail **stop-losses and limit orders cluster** relative to spot. Stops above spot (shorts' stops) and below spot (longs' stops) are liquidity pools that dealers/algos run toward; concentrated open positions on one side flag squeeze risk. This is a *path/levels* signal, not a drift signal — and it is exactly the kind of thing absent from EMA/RSI/vol features.
- Update cadence: **5 min for premium clients, 15 min for non-premium** (OANDA Order Book Tool page) — matches your horizon.

**Other retail sources** (for cross-broker aggregation, since single-broker books are noisy): IG Client Sentiment (DailyFX), FXSSI current-ratio tool, Myfxbook Community Outlook, FXCM SSI (legacy), Dukascopy/SWFX sentiment, ForexFactory Positions. Combining several reduces single-broker idiosyncrasy.

## Concrete techniques / features / architectures to try

Ranked by orthogonality-to-what-you-have and 15m relevance.

### A. OANDA Order/Position Book level-structure features (HIGH priority)
Pull `orderbook_data` and `position_book` history for EURUSD (+ your 6 other USD pairs). For each snapshot, bucket relative to current spot and engineer:
1. **Stop-cluster pressure:** sum of `os` (short stops, located above spot) within +5/+10/+20 pips minus sum of `ol`-derived long stops within -5/-10/-20 pips. Net "magnetic pull" toward the heavier stop pool. Hypothesis: price drifts toward the larger nearby stop cluster over the next 15m (stop-run).
2. **Position-book skew & crowding:** `(pl - ps)/(pl + ps)` overall and within ±N pips. Extreme one-sidedness → contrarian fade signal; combine with distance-to-nearest-stop-pool.
3. **Order-book asymmetry above vs below spot:** pending limit orders act as support/resistance; net `(orders below) - (orders above)` is a short-horizon barrier feature.
4. **Change features (the real alpha):** Δ in each of the above over the last 1-3 snapshots (5-15 min). *New* stop accumulation or rapid position unwinding is more predictive than the static level. Build deltas, not just levels.
5. Interact these with your existing realized-vol and time-of-day features (stop-runs cluster around London/NY opens and news).

### B. Cross-broker retail sentiment contrarian feature (MED)
Aggregate long/short % across OANDA position ratio + IG + FXSSI + Myfxbook into a consensus retail-positioning z-score per pair. Feature = signed distance from neutral and its rate-of-change. Use as a **conditioning variable** in your existing GBM (interaction terms), not a standalone classifier. Expect edge at hours-to-days; at 15m its value is mostly in *gating* (suppress/boost a TA signal when retail is at an extreme).

### C. Segmented institutional flow drift feature (MED, gated on CLS budget)
If CLS FX Flow is obtained: build daily/4h **net corporate flow** and **net fund/NBFI flow** signs as slow drift conditioners, broadcast onto 15m bars. Per Lyons, corporate = trend, financial = abrupt moves. Test the financial-segment net flow as the only component with a chance of short-horizon lift. Treat as regime, not bar-classifier.

### D. COT as a weekly regime flag only (LOW)
Leveraged-funds net positioning extreme (z-score vs trailing 1-3y) from the TFF report → a weekly categorical "positioning stretched long/short/neutral" broadcast to bars. Cheap to add, almost certainly redundant with your existing vol-regime/seasonality features. Include only as a robustness check.

### E. Modeling architecture
- These are **slowly-varying, level-based** features — feed them to your existing LightGBM/CatBoost as extra columns with explicit *change* and *interaction* terms; do not expect a sequence net to extract more.
- **Critical OOS hygiene:** retail book/sentiment snapshots are timestamped; align strictly to *decision time* and lag by the publication cadence (15 min for free OANDA). Any "use the 15-min book to predict the bar it was measured during" is look-ahead leakage.
- Evaluate as **selective prediction**: the realistic win is "when retail positioning is at an extreme AND a stop cluster sits within N pips, classify the next 15m better than baseline" — i.e. higher accuracy at low coverage, consistent with your current 0.632@0.2% frontier. Target lifting that coverage curve; unconditional 75% remains the open target.

## Reported results & CREDIBILITY assessment

**Credible / reproducible:**
- *Order flow explains contemporaneous FX returns* (Evans-Lyons, R² 50-80% daily). Rock-solid, replicated for decades. BUT it is contemporaneous and daily; it does **not** translate to a lagged 15m classifier. Credibility high; *relevance to our exact problem low*.
- *Retail FX clients are return-contrarian and "uninformed"* (multiple academic + every broker that sells contrarian sentiment). Robust qualitatively. The *magnitude* and *timing* of any tradable edge is where claims get soft.
- *Intraday retail EURUSD order flow has exploitable contrarian content* (JIFMIM March 2025). Peer-reviewed, EURUSD-specific, intraday — the most credible modern on-point result. Caveat: "can be profitable" in a paper rarely survives realistic spread/cost and is not stated as a 15m bar-accuracy number; treat as *promising, not proven for our metric*.

**Hype / overfit risk — discount heavily:**
- Practitioner "OANDA order book → 75% reversal" / "stop-hunt" guides (PriceActionNinja, ForexMentorOnline, etc.) are educational, anecdotal, **no out-of-sample stats, no costs, heavy survivorship/confirmation bias**. The *mechanism* (stop pools are liquidity magnets) is real and worth testing; the *win-rate claims* are unsupported.
- "COT predicts forex direction" backtests (QuantifiedStrategies et al.) — mostly weekly/swing, frequently in-sample, and irrelevant at 15m.
- Any vendor (incl. CLS marketing) claiming flow "improves alpha generation / FX rate forecast accuracy" is selling; the academic CLS work (Hasbrouck-Levich, NBER w23206) is about **liquidity and market metrics**, not a documented lagged directional forecaster.

**Leakage traps specific to this vector:**
- **Contemporaneous-vs-lagged:** the entire order-flow literature's headline R² is contemporaneous. Re-deriving it as "predictability" is the classic leak. Use only strictly-lagged flow.
- **Snapshot timing:** retail book updates every 5-15 min; using the snapshot that overlaps the prediction bar leaks the outcome.
- **Survivorship in retail data:** broker sentiment reflects only that broker's surviving clients; not a clean population.
- **Reverse causality:** retail positioning is *driven by* recent returns (the 2025 paper says so explicitly) — so a naive retail-skew feature is partly a lagged-return feature you already have. Orthogonality must be verified by residualizing against your existing return/TA features before claiming new lift.

## Data sources needed

| Source | What | Frequency / latency | Access | Cost | Fidelity for 15m |
|---|---|---|---|---|---|
| **OANDA forexlabs API** (`orderbook_data`, `position_book`, `HistoricalPositionRatios`) | Price-level open orders + open positions (stop/limit map) for ~16 instruments incl. EURUSD | 5 min (premium) / 15 min (free); historical book available | REST V20, `oandapyV20` Python wrapper; needs OANDA account + token | Free w/ account (premium tier richer) | **Best available** — intraday, level-structured, orthogonal |
| IG Client Sentiment / DailyFX | Aggregate long/short % per pair | ~hourly/daily | Web / DailyFX API | Free | Med (scalar, slow) |
| FXSSI current-ratio | Aggregated multi-broker retail ratio | ~hourly | Web/scrape; paid API for history | Free/low | Med (good for cross-broker consensus) |
| Myfxbook Community Outlook | Long/short popularity + positions | ~minutes | Public API (`/get-community-outlook`) | Free | Med |
| **CLS CLSMarketData FX Flow** | Net directional flow by counterparty segment + MM/price-taker | Intraday (5-min) / daily / monthly; settlement-lagged | CLS Registered Vendor Program / Nasdaq Data Link | $$$ institutional | High info, but contemporaneous + costly + lagged |
| EBS Market Data (CME) | Interdealer spot flow/volume | Tick/intraday | CME DataMine | $$$ | High but interdealer, costly |
| CFTC COT / TFF | Weekly futures positioning by trader class | Weekly (Tue), Fri 3pm CT release | Free CFTC download; OANDA `CommitmentsOfTraders` endpoint | Free | Low for 15m (weekly) |
| State Street / Citi PAIN / JPM flow indices | Institutional risk-appetite & FX flow indices | Daily-weekly | Bank research subscription | $$$ | Low for 15m (slow, gated) |

Start with the **free OANDA book** — it is the only one that is simultaneously intraday, level-structured, orthogonal, and zero-cost.

## Relevance & priority for OUR project

How it interacts with what you've already ruled out:
- Your signed-10s-volume OFI proxy captured *executed interbank* imbalance and added ~0 lift, and you proved book imbalance decays to ~0.50 by 1-5 min. **Retail order/position book is a different object**: it is *pending* orders and *resting* positions of an uninformed, slow, contrarian crowd — a liquidity/levels map, not a fast execution-imbalance signal. It is not the thing you already tested.
- Your existing features have *no* representation of "where retail stops are clustered relative to spot," which is the one genuinely new, mechanistically-motivated, intraday feature this vector offers.
- Aggregate sentiment ratio and COT/institutional indices are largely **redundant** with your existing slow trend/vol-regime/seasonality features and are partly just lagged returns — low expected marginal lift.

Ranking:
- **HIGH — OANDA Order/Position Book level-structure & change features (A).** Cheap, intraday, orthogonal, mechanistically motivated, EURUSD-specific academic support (2025). Most likely source of *any* new selective-prediction lift in this vector. Build it next.
- **MED — Cross-broker retail contrarian z-score as a gating/interaction feature (B).** Worth adding as conditioner; residualize against existing return features first to confirm it is not just lagged returns.
- **MED (budget-gated) — CLS segmented financial-vs-corporate net flow as slow drift conditioner (C).** Only if institutional data budget exists; expect regime-level, not 15m-classifier, value.
- **LOW — COT weekly positioning flag (D)** and **institutional flow indices.** Almost certainly redundant; include only for completeness/robustness.

Honest expectation: this vector has not *single-handedly* reached 75% unconditional 15m accuracy — that stays open. Its realistic contribution is to **improve the selective-prediction frontier** (push your 0.632@0.2% curve up/right) in the specific regime where retail is crowded and a stop pool sits nearby — a genuinely new, orthogonal conditioner you have not yet tried.

## Sources

- Evans, M. & Lyons, R. (2002), "Order Flow and Exchange Rate Dynamics," *Journal of Political Economy* / NBER w7317 — https://www.nber.org/papers/w7317 — foundational result: daily FX R² 50%+ from interdealer order flow; the canon this whole vector rests on.
- Lyons, R. (2003), "Explaining and Forecasting Exchange Rates with Order Flows," *Économie Internationale* 96, CEPII — https://cepii.fr/IE/rev96/rev96lyons.pdf — *read in full*; the seven-lesson executive summary that pins flow predictability to daily-contemporaneous and monthly-forecasting horizons, and segments flow by counterparty.
- "An Investigation of Customer Order Flow in the Foreign Exchange Market" (Glasgow) — https://www.gla.ac.uk/media/Media_125282_smxx.pdf — finds short-run customer flow does NOT Granger-cause returns; key skeptical counterweight.
- "End-user order flow and exchange rate dynamics – a dealer's perspective," *European Journal of Finance* 17(2) — https://www.tandfonline.com/doi/abs/10.1080/13518471003651925 — financial vs commercial customer-flow asymmetry.
- Hasbrouck, J. & Levich, R., "FX Market Metrics: New Findings Based on CLS Bank Settlement Data," NBER w23206 — https://www.nber.org/system/files/working_papers/w23206/w23206.pdf — academic use of CLS settlement data (liquidity/metrics, not lagged forecasting).
- CLS Group, "FX Flow" product page — https://www.cls-group.com/products/data/clsmarketdata/fx-flow/ — directional net flow by counterparty segment, intraday/daily/monthly, "coreness" MM vs price-taker; the premier real customer-flow dataset (paid).
- "News and intraday retail investor order flow in foreign exchange markets," *J. Int. Financial Markets, Inst. & Money* 98 (Mar 2025), S1042443125000368 — https://www.sciencedirect.com/science/article/abs/pii/S1042443125000368 — proprietary intraday EURUSD retail positioning; retail return-contrarian; "simple strategies exploiting retail intraday order flow can be profitable." Most on-point modern result.
- OANDA forexlabs `OrderbookData` API docs (oandapyV20) — https://oanda-api-v20.readthedocs.io/en/latest/endpoints/forexlabs/orderbookdata.html — exact schema: per-price `ol/os/pl/ps` (orders long/short, positions long/short); the implementable data spec for feature A.
- OANDA Order Book / Position Book Tool — https://www.oanda.com/bvi-en/lab-education/tools/order-book-position-book-tool/ — update cadence 5 min (premium) / 15 min (free); confirms intraday availability.
- "How to Measure Retail Forex Market Sentiment" (EarnForex) — https://www.earnforex.com/guides/how-to-measure-retail-forex-market-sentiment/ — survey of retail sentiment sources (OANDA, ForexFactory, DailyFX, Saxo, Dukascopy, cTrader); cross-broker aggregation rationale.
- IG, "Trading on sentiment: using IG client sentiment data" — https://www.ig.com/au/trading-strategies/trading-on-sentiment--using-ig-client-sentiment-data-220930 — broker's own contrarian framing of retail clients as "uninformed."
- "Commitments of Traders Trading Strategies: COT Report And Backtest" (QuantifiedStrategies) — https://www.quantifiedstrategies.com/commitments-of-traders/ — confirms COT is weekly (Tue as-of, Fri 3pm CT release); establishes why it is unusable at 15m.
- CFTC Commitments of Traders — https://www.cftc.gov/MarketReports/CommitmentsofTraders/index.htm — free official source; TFF report splits dealer/asset-mgr/leveraged-funds.
- State Street Markets Institutional Investor Indicators / Risk Appetite — https://globalmarkets.statestreet.com/research/portal/insights/iii — flow-based institutional risk-appetite indices (slow, gated; low 15m relevance).
- "Using Oanda's Orderbook To Trade Stop Hunts" (ForexMentorOnline) — https://forexmentoronline.com/using-oandas-orderbook-to-trade-stop-hunts/ — practitioner mechanism for stop-cluster magnetism (anecdotal, no OOS stats — test, don't trust).
