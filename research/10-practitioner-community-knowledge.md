# Practitioner & Community Knowledge: What Intraday FX Directional Edges Actually Work

*Research vector 10 — mining forums (r/algotrading, r/quant, r/Forex, EliteTrader, Forex Factory, Wilmott, QuantConnect), practitioner blogs, and the credible academic work those communities actually cite, for HONEST reports of intraday FX directional edges and a skeptical teardown of "75%/90% win-rate" binary-option marketing.*

Project target for context: predict EURUSD (and USD majors) **direction at a 15-minute horizon, >75% correct, strict OOS**. We already exhausted 239 causal TA features, cross-pair lead-lag, signed-volume OFI, GBM/XGB/Cat/TabNet/GRU (~0.52–0.527 AUC), stat-arb basket, daily context, calendar event-timing, and selective prediction (best ~0.632 at 0.2% coverage). Clean >75% only found at 3-second horizon. The honest community/academic consensus below is largely a *constraint map* — it tells us which "edges" are mirages and which of the few real ones might survive at 15m.

---

## TL;DR

- **The single most important number for a binary-direction target: at the typical 75–95% broker payout, break-even win rate is ~51–57%, NOT 50% — and a *conventional* (non-binary) 5-minute trade needs ~80% win rate just to cover spread.** Your 75% goal is, in conventional-execution terms, roughly the break-even line for a *5-minute* scalp, not a profit line. This reframes the goal: 75% at 15m is highly ambitious by every honest practitioner benchmark (Financial Hacker; their "Scalping" cost curve) and remains an open target.
- **The one rigorously documented, reproducible, *predictable* intraday FX direction effect is fixing/clock seasonality**: the USD systematically appreciates into the major FX fixes (Tokyo, ECB/London 1:15pm CET, WM/R 4pm London) and reverses after, with daily swings ~2 bps (~5% annualized) and t-stats up to ~9.2 (Krohn et al., *Journal of Finance* 2024). This is the most credible "free" directional signal you have NOT exploited as a structural feature — but it is small and largely arbitraged net of retail spreads.
- **Net-of-cost reality is the killer**: Krohn's fix-reversal strategy earns Sharpe 0.5–0.7 *only at institutional/CME spreads*; at full retail bid-ask it goes **negative** for GBP and JPY (EUR survives best). Every credible source converges on the same verdict: the edge is real, the *exploitable* edge after costs is marginal and pair-specific.
- **Practitioner consensus on retail "75%/90% win-rate" bots/signals is unanimous: scam or martingale-in-disguise.** High win rates are manufactured by hidden tail risk (martingale/grid blow-ups) or by inverting the real statistic (75–90% of retail *traders lose*). Treat any sold "system" claiming >70% as a credibility-zero prior.
- **Best-so-far level for honest ML FX direction is ~53–59% daily**, and published higher numbers almost always carry one of three leaks: forward-filled macro features, single-year test windows, or bid-only (no spread) labels. The much-cited Guyard & Deriaz (2024) "58.52% EURUSD daily" result tests on **2022 only** with forward-filled economic indicators — exactly the leakage pattern to distrust.
- **Robert Carver's "speed limit" is the cleanest practitioner framing of your current best level**: as horizon shortens, cost-per-Sharpe rises hyperbolically; there is an optimal trading speed and intraday FX for retail-grade costs sits *past* it. Faster signals don't help; you need a *cheaper-to-trade* expression or a structurally larger move.
- **The few edges practitioners report as durable at minutes-to-hours are NOT pure direction**: fix/flow seasonality, session-open range behavior conditioned on prior-session range, and *event-window volatility* (not direction — NFP/ECB are ~0.50 on direction with V-shaped reversals). Direction-pure alpha at 15m is the part everyone says is closest to a random walk.
- **Where to actually look next (orthogonal to what you tried)**: (1) fixing-window conditioning as a *first-class regime feature*, not a calendar proxy; (2) cross-asset *level* signals you likely haven't used — US 2y/10y yield ticks, ES/SPX futures returns, and DXY-component dislocation at minute resolution; (3) dealer-positioning / CFTC-derived skew as a slow conditioner of intraday drift; (4) honest re-labeling at the *mid-to-mid net of half-spread* so the model optimizes the thing you can actually trade.

---

## Key findings (each with inline citation)

### 1. The binary-payout math redefines what "75% accuracy" means

The Financial Hacker ("jcl", developer of Zorro) gives the exact break-even formula for binary options:
> W = (1 − Pl) / (1 + Pw − Pl), where Pw = win payout, Pl = loss payout.
> "With 85% win payout and no loss payout, you need a win rate of W = 1/1.85 = **54%**." ([Financial Hacker, *Binary Options: Scam or Opportunity?*](https://financial-hacker.com/binary-options-scam-or-opportunity/))

At a worse 75% payout, break-even is 1/1.75 ≈ **57%**; at a generous 95%, ≈ **51%**. So a binary instrument *lowers* the bar relative to conventional trading — the same article's "Scalping" cost curve shows a conventional broker requires "**almost 80% of five-minute trades** … impossible for a trading system under normal conditions" purely to overcome spread. **Implication for us:** a 75% *correct-direction* rate at 15m, if achievable, would be wildly profitable on binaries (huge edge over the ~57% bar) — which is exactly why honest sources treat it as still open OOS. The goal's difficulty is highlighted, not contradicted, by the binary framing.

### 2. The one robust, *predictable* intraday FX direction effect: fixing / clock seasonality

Krohn, Mueller, Whelan et al., **"Foreign Exchange Fixings and Returns Around the Clock"** (*Journal of Finance*, 2024 — top-tier, peer-reviewed) is the most credible source the FX community cites for genuine intraday *directional* predictability:
> "Up to the Tokyo fix the [dollar] DOL appreciates by ~5.3% per annum (2.1 bps per day) with a t-statistic … ~9.2. … the U.S. dollar appreciates in the run up to foreign exchange fixes" and reverses after; "the intraday seasonality does **not** carry over into close-to-close returns" (i.e., it's a within-day predictable component, not a trend). ([Krohn et al., INSEAD working-paper PDF](https://sites.insead.edu/facultyresearch/research/file.cfm?fid=66802))

Magnitudes: daily swings "around 2 basis points (or over 5% annualized)" with "persistence of reversals." t-stats of 5.5–9.2 are *strong* by FX standards. This is a real, documented, sign-predictable intraday pattern keyed to **wall-clock time and fix windows** — distinct from the time-of-day vol seasonality you already ruled out, because it predicts the **sign of drift**, not just volatility.

A related, older, community-cited result: Breedon & Ranaldo, **"Intraday Patterns in FX Returns and Order Flow"** (SSRN 2099321 / QMUL WP 694) — currencies tend to *depreciate during their own local trading hours and appreciate during foreign hours* (e.g., EURUSD tends to fall in European hours, rise in US hours), driven by order-flow seasonality. (SSRN blocks scraping; summarized via [Mammadov, "Profiting from FX Fixes"](https://defitrading.substack.com/p/profiting-from-fx-fixes), which reproduces and tests both papers.)

### 3. The net-of-cost verdict (this is the whole ballgame)

Krohn et al. are unusually honest about exploitability:
> "ignoring transaction costs, a portfolio … generates extremely large low frequency variation … [but] **not easy to exploit once transaction costs are accounted for**."
> Gross fix-reversal returns: "**13.6%, 11.2% and 12.9% for the euro, pound and yen**"; after **full** transaction costs: "**6.6%, 4.2% and 3.4%**"; "Sharpe ratios are **negative using the full transaction costs**, and positive and very high if transaction costs are ignored. Market participants able to trade at better bid-ask spreads may earn **Sharpe ratios ranging between 0.5 and 0.7**." For the euro specifically, "returns … are extremely large and generate a [high] Sharpe" even after CME spreads, while "returns … are negative for the pound and the yen." ([Krohn et al. PDF](https://sites.insead.edu/facultyresearch/research/file.cfm?fid=66802))

So even the single best documented intraday direction effect is **EUR-favorable, GBP/JPY-negative, after costs** — and only investable at institutional execution. This is the recurring shape of *every* honest intraday-FX finding.

### 4. Honest ML benchmarks: ~53–59% daily direction, with leakage traps

- Di Persio & Honchar (CNN/LSTM on S&P500 direction): **0.536 classification accuracy** — cited across the literature as a typical honest number. (via [Guyard & Deriaz 2024, arXiv 2409.04471](https://arxiv.org/pdf/2409.04471))
- Guyard & Deriaz, **"Predicting Foreign Exchange EURUSD direction using machine learning"** (2024): 21 ML models + meta-stacking + PCA + macro indicators → "**58.52% for one-day ahead forecasts … annual return of 32.48% for the year 2022.**" ([arXiv 2409.04471](https://arxiv.org/pdf/2409.04471)) **Credibility flag (high leakage risk):** test set is **2022 only** (one regime — a strong USD trend year), and economic indicators are **forward-filled** ("for each sample, the value … is the value of the last update"), which mixes look-ahead-prone macro state into daily features. The headline 58.52% should be read as an *optimistic upper bound under favorable conditions*, not a reproducible OOS edge.
- The same paper repeats the widely-circulated folk stat: "only **2% of traders** are successful in predicting Forex market movement correctly."

### 5. Practitioner consensus on scalping/short-horizon direction: spread eats it

The EliteTrader thread "Can you profitably scalp the forex?" frames the canonical retail question — "scalping for 10–20 pips … can you consistently be profitable even with the spread that every forex pair has?" — and the durable forum consensus (mirrored in the Financial Hacker cost curve) is that at retail spreads + slippage the required win rate is punishingly high and most claimed scalp edges are spread-illusions or survivorship. ([EliteTrader thread](https://www.elitetrader.com/et/threads/can-you-profitably-scalp-the-forex.68211/))

Robert Carver (ex-AHL, *Systematic Trading*) gives the rigorous version — the **"speed limit"**: he "avoids strategies that trade too slowly or too quickly … slow instruments have low Sharpe ratios while fast instruments are expensive to trade," measuring cost as annualized-Sharpe loss per round trip. ([qoppac blog, "How fast should we trade?"](https://qoppac.blogspot.com/2020/04/how-fast-should-we-trade.html); [7 Circles summary](https://the7circles.uk/systematic-trading-5-speed/)) For retail-grade FX costs, 15-minute direction sits *past* the optimal speed: even a true edge gets capitalized away by the round-trip cost-to-vol ratio.

### 6. The "75%/90% win-rate" bot/signal claims: how they're manufactured

- **Inverting the real statistic.** "75–90% of retail traders *lose* money with binary options" — which is why they were banned for retail in the EU/UK/etc. A "75% win" claim is statistically the *reciprocal* of the documented loss rate. ([daytrading.com](https://www.daytrading.com/binary-options-scams); [Dukascopy](https://www.dukascopy.com/swiss/english/marketwatch/articles/binary-trading-scams/))
- **Hidden tail risk via martingale/grid.** Forex Factory "holy grail" threads show how a 90%+ win *rate* is trivially produced by averaging-down: "left a martingale EA running overnight … made about $300 … in the morning's London session it hit maximum drawdown and lost $540." Grid/martingale "inevitably blow up … due to exponential drawdowns." High win-rate, negative expectancy. ([Forex Factory grid/martingale threads](https://www.forexfactory.com/thread/278178-bks-grid-ea-martingale))
- **Counterparty/execution games.** Even a genuine edge can be neutralized by **asymmetric slippage** — "execution of a client's order when the slippage favors the broker, but a requote when it favors the client" — and **last look** rejections in fast markets. ([forexop](https://forexop.com/learning/slippage-requotes-and-unfair-price-execution/); [b2broker, *Last Look*](https://b2broker.com/news/last-look-in-forex/)) **Implication:** a 15m direction model validated on mid-price will systematically over-state live edge unless labels and costs use realistic, *asymmetric* fill assumptions.

### 7. Event-window trading is a volatility edge, not a direction edge

The practitioner consensus on NFP/ECB/CPI windows: direction is "**a coin-flip**," and "markets can mimic a **V-shape post-NFP**, where the spike goes one direction then reverses." ([FOREX.com NFP V-shape](https://www.forex.com/en-uk/trading-academy/courses/advanced-strategies/uk-the-non-farm-payrolls-v-shaped-reversal/); [daytrading.com NFP](https://www.daytrading.com/nfp-trading)) This matches your finding that calendar event-timing is redundant with time-of-day for *vol* — it confirms there is little *directional* alpha in the release itself, but possibly in the **reversal of the first spike** (a conditional mean-reversion, not a release-direction bet).

### 8. Overfitting is the dominant failure mode — and the reason most "edges" are fake

The community's most-cited statistical antidote is the **Deflated Sharpe Ratio** (Bailey & López de Prado): correct the observed Sharpe for the *number of trials* and non-normality, because "backtest optimizers search for parameter combinations that maximize simulated performance, leading to backtest overfitting; selection bias combined with overfitting misleads investors into … strategies that systematically lose money." ([SSRN 2460551](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551); [Wikipedia summary](https://en.wikipedia.org/wiki/Deflated_Sharpe_ratio)) QuantPedia's IS/OOS study quantifies the typical decay: out-of-sample Sharpe degrades **~33% (mean) / 44% (median)** vs in-sample. ([QuantPedia, IS vs OOS](https://quantpedia.com/in-sample-vs-out-of-sample-analysis-of-trading-strategies/)) Given the dozens of feature families you have already trialed, your *effective* number of trials is large — any future 15m result near 75% should be discounted hard via DSR before being believed.

---

## Concrete techniques / features / architectures to try

Ranked by orthogonality to what you've already exhausted.

1. **Fixing-clock structural features (NOT a calendar vol proxy).** Build explicit features keyed to the three major fixes in *exchange-local time with DST handling*:
   - minutes-to-next-fix and minutes-since-last-fix (Tokyo 9:55 JST / ECB 13:15 CET / WM-R 16:00 London);
   - a signed "fix-pressure" feature = sign of (time relative to fix) × historical Krohn drift sign for that pair;
   - interaction of fix-window flag × your existing realized-vol and OFI features.
   The Krohn effect is a *sign-of-drift* predictor, distinct from the time-of-day *vol* seasonality you ruled out. Test whether conditioning your existing model on fix-windows lifts accuracy *inside those windows specifically* (it may be a high-coverage-within-window edge even if dilute on average).

2. **Cross-asset *level* features at minute resolution (likely your biggest untried orthogonal source).** You used the 6 other USD pairs for lead-lag, but EURUSD is ~97% USD-factor — so the informative variable is the *USD factor's drivers*, not peer FX. Add, at 10s–1m bars, **synchronized** (causal, lagged ≥1 bar): US 2y & 10y Treasury yield changes (or ZN/ZF futures), ES/SPX-future returns, DXY-basket dislocation (EURUSD vs. its implied value from the other DXY legs), and Brent/Gold for risk-on/off. Minute-level rates and equity-index moves *lead* FX more cleanly than peer-FX lead-lag.

3. **Honest, tradeable labels.** Re-label the target as **direction of mid-to-mid return net of half-spread** (and test an asymmetric-fill variant). If the 0.632@0.2% selective result was computed on bid-only or mid prices, re-running on net labels is the single cheapest way to learn whether your "edge" is real or a spread artifact (per §6 execution games).

4. **Slow conditioner: dealer positioning / CFTC skew.** Use weekly CFTC CoT non-commercial net positioning (and option risk-reversal skew if obtainable) as a *low-frequency regime feature* that gates intraday direction (crowded longs → fade rallies). Slow, but orthogonal to all your fast microstructure — and exactly the kind of "longer-lived" signal your mechanistic note says you need.

5. **Fix-reversal conditional model.** Implement Krohn's actual structure: predict the **post-fix reversal** sign (USD weakens after the fix it strengthened into). Evaluate accuracy *only* in the ±N-minute post-fix window per pair; EUR is the best candidate (only pair positive net of CME costs).

6. **Event-spike fade, not event-direction bet.** For scheduled releases, model the *reversal* of the first 1–5 minute spike (V-shape) rather than the release direction. Direction-of-release is ~0.50 (§7); the conditional fade is the documented structure.

7. **Deflated-Sharpe / PBO gate on every result.** Before believing any 15m number, compute the Deflated Sharpe Ratio (account for your large trial count) and the Probability of Backtest Overfitting. This is a *methodology* fix that prevents the next false positive, given how many families you've already tested.

---

## Reported results & CREDIBILITY assessment

| Claim / result | Source | Credibility | Leakage / caveat |
|---|---|---|---|
| Intraday USD-into-fix appreciation, ~2bps/day, t up to 9.2; reverses after fix | Krohn et al., *J. Finance* 2024 | **High (peer-reviewed, top journal, honest cost accounting)** | Real but small; net Sharpe 0.5–0.7 only at institutional spreads; **negative net** for GBP/JPY |
| Local-hours depreciation / foreign-hours appreciation | Breedon & Ranaldo (QMUL WP 694) | **Medium-High** (cited, reproduced by independent blogger) | Order-flow seasonality; exploitability after costs unproven for retail |
| 58.52% EURUSD daily direction, +32% in 2022 | Guyard & Deriaz 2024 (arXiv) | **Low-Medium** | **Tested on 2022 only** (trend year); forward-filled macro features; daily not 15m |
| ~0.536 S&P direction (CNN/LSTM) | Di Persio & Honchar | **Medium** (representative honest baseline) | Shows the realistic ~53–54% best-so-far level |
| Binary break-even 51–57% at 75–95% payout; 5m conventional needs ~80% | Financial Hacker | **High** (math is verifiable) | None — it's arithmetic; reframes the 75% goal as very hard |
| "75% win-rate" bots / signals | Trader's Union / Dukascopy / daytrading.com reviews | **Hype/Scam (credibility ~0)** | Inverts the loss statistic; martingale tail risk; payment fraud |
| Martingale/grid "90% win" systems | Forex Factory threads | **Negative-expectancy (debunked)** | High win rate by construction; blows up on trend/range |
| IS→OOS Sharpe decay ~33–44% | QuantPedia | **High** | Generic but directly applicable to your many-trial situation |
| Deflated Sharpe Ratio method | Bailey & López de Prado (SSRN) | **High (standard quant methodology)** | Use it as a gate, not a result |

**Bottom line on credibility:** The only intraday *direction* effects that survive peer review and honest cost accounting are **fixing/clock seasonality** — and even those are marginal-to-negative net of retail spreads. Everything claiming >70% directional accuracy for sale is, in the documented community record, either statistic-inversion, martingale tail risk, or single-regime overfit. None of the credible sources reports anything close to a *costed, OOS, multi-regime* 75% at minutes horizon. That absence is itself a strong finding: it is consistent with your own current best level at ~0.52–0.53 AUC and leaves the target open.

---

## Data sources needed

| Data | For which idea | Where | Free / Paid | Fidelity notes |
|---|---|---|---|---|
| Exact fix times w/ DST (Tokyo 9:55 JST, ECB 13:15 CET, WM/R 16:00 London) | Fix-clock features (#1, #5) | ECB site / WM-Refinitiv methodology docs | Free | Must handle DST per zone; you already have the bars to align |
| US Treasury yields / futures (ZN, ZF, ZT) at 1m | Cross-asset levels (#2) | CME via data vendor; FRED for daily; Databento/Polygon for intraday | Free (FRED daily) / Paid (intraday futures) | Intraday rates are the cleanest USD-factor driver; daily won't help at 15m |
| ES/SPX futures 1m | Risk-on/off (#2) | Databento, Polygon, IQFeed | Paid (modest) | Equity index leads FX in risk regimes |
| DXY component quotes (you already have 7 USD pairs) | DXY dislocation (#2) | **Already in your dataset** | Free | Compute EURUSD-implied-from-basket residual |
| CFTC Commitments of Traders | Positioning conditioner (#4) | CFTC.gov (weekly, free) | Free | Weekly, lagged 3 days — slow regime feature only |
| FX option risk-reversal / 25-delta skew | Positioning conditioner (#4) | Bloomberg/Refinitiv (paid); some free EOD | Paid | Optional; risk-reversals encode directional skew |
| Economic-calendar timestamps (you have this) | Event-spike fade (#6) | ForexFactory calendar / Econoday | Free | You already proxied this; reuse for the *fade*, not the direction |
| Tick bid/ask w/ sizes (you have this) | Net-of-spread labels (#3) | **Already have** | — | Use to build realistic asymmetric-fill labels |

The headline gap vs. what you already have is **intraday US rates and ES futures** — the actual drivers of the USD factor that EURUSD is 97% loaded on. Everything else is recombination of data you hold.

---

## Relevance & priority for OUR project

| Idea | Priority | Interaction with what you already ruled out |
|---|---|---|
| **Cross-asset intraday LEVEL features (rates + ES + DXY dislocation)** | **HIGH** | You did peer-FX lead-lag (no lift) and USD-basket *residual* fade. You have NOT fed the *external* USD-factor drivers (Treasuries, equity index) at minute resolution. This is the most genuinely orthogonal untried source, and it targets exactly the "97% USD-factor" structure. |
| **Fixing-clock sign-of-drift features** | **HIGH** | Distinct from your time-of-day *vol* seasonality and your event-timing proxy (which predicted vol, not sign). Krohn's effect predicts *direction*. Even if dilute on average, it may give high accuracy *within fix windows* → pairs naturally with selective prediction. |
| **Net-of-spread / asymmetric-fill labels** | **HIGH (cheap)** | Directly tests whether your 0.632@0.2% selective result is real or a mid-price artifact. Low effort, high information. |
| **Deflated-Sharpe / PBO gate** | **HIGH (methodology)** | Given your large trial count across 17 versions, this prevents the next false 75%. |
| **Event-spike FADE (V-shape reversal)** | **MED** | Re-uses your calendar data for a *conditional reversal* rather than the release direction you implicitly treated as redundant. |
| **CFTC / risk-reversal positioning conditioner** | **MED** | Slow, orthogonal "longer-lived" signal your mechanistic note explicitly asks for; gates intraday direction by crowding. |
| **Fix-reversal conditional model (EUR-only)** | **MED** | The only effect positive net of (institutional) costs; worth a scoped test even if retail-uneconomic, to see if *accuracy* (your metric, not P&L) clears 75% in-window. |
| **Buy retail "75%" bots/signals** | **NONE (avoid)** | Documented scam/martingale; zero information value. |
| **Faster microstructure / more TA families** | **LOW** | Community + your own findings agree: past the speed limit, more of the same won't move 15m. |

**Net read for the project:** Practitioner/community knowledge does **not** yet offer a hidden 75% direction trick — its honest verdict is *consistent with* your current best level, and the target stays open. Its value is (a) one credible, untested orthogonal *direction* effect (fixing/clock seasonality keyed to USD-factor drivers), (b) a strong steer toward **cross-asset intraday levels** rather than more FX-internal features, and (c) discipline (net labels, Deflated Sharpe) to stop chasing leakage. If 75% is to be reached at 15m, the credible record suggests it will be *conditional* (within fix windows / specific regimes) and *selective*, not unconditional — which aligns with your best result being a selective one.

---

## Sources (annotated)

1. **Krohn, Mueller, Whelan, et al. — "Foreign Exchange Fixings and Returns Around the Clock"**, *Journal of Finance* 2024 — [INSEAD PDF](https://sites.insead.edu/facultyresearch/research/file.cfm?fid=66802) / [Wiley](https://onlinelibrary.wiley.com/doi/10.1111/jofi.13306). *The* credible, peer-reviewed evidence of predictable intraday FX *direction* (USD-into-fix), with honest net-of-cost teardown (Sharpe 0.5–0.7 institutional; negative retail GBP/JPY). Most important source here.
2. **Financial Hacker (jcl) — "Binary Options: Scam or Opportunity?"** (2016) — [link](https://financial-hacker.com/binary-options-scam-or-opportunity/). Exact break-even-win-rate math (54% at 85% payout); the "~80% required to scalp 5m conventionally" cost curve. Reframes the 75% goal quantitatively.
3. **Breedon & Ranaldo — "Intraday Patterns in FX Returns and Order Flow"** (QMUL WP 694 / SSRN 2099321) — [RePEc](https://ideas.repec.org/p/qmw/qmwecw/694.html). Local-hours depreciation / foreign-hours appreciation; order-flow seasonality underlying the clock effect.
4. **Mammadov — "Profiting from FX Fixes"** (Substack, *Actionable Trading Strategies*) — [link](https://defitrading.substack.com/p/profiting-from-fx-fixes). Independent practitioner who *reproduces and tests* Krohn and Breedon-Ranaldo on EURUSD — useful for implementation detail and honest "exploitable vs academic" framing.
5. **Guyard & Deriaz — "Predicting Foreign Exchange EURUSD direction using machine learning"** (arXiv 2409.04471, MLMI 2024) — [PDF](https://arxiv.org/pdf/2409.04471). 58.52% daily EURUSD direction — a textbook example of the leakage traps (2022-only test, forward-filled macro) to distrust; also a good literature map of prior FX-ML accuracy (~0.536 typical).
6. **Bailey & López de Prado — "The Deflated Sharpe Ratio"** (SSRN 2460551) — [link](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551) / [Wikipedia](https://en.wikipedia.org/wiki/Deflated_Sharpe_ratio). The standard multiple-testing correction; mandatory gate given your large trial count.
7. **QuantPedia — "In-Sample vs Out-of-Sample Analysis of Trading Strategies"** (2023) — [link](https://quantpedia.com/in-sample-vs-out-of-sample-analysis-of-trading-strategies/). Quantifies typical IS→OOS Sharpe decay (~33% mean / 44% median).
8. **Robert Carver — "How fast should we trade?"** (qoppac blog, 2020) — [link](https://qoppac.blogspot.com/2020/04/how-fast-should-we-trade.html) + [7 Circles "Speed" summary](https://the7circles.uk/systematic-trading-5-speed/). The "speed limit": cost-per-Sharpe explains why retail 15m FX direction sits past the optimal trading speed.
9. **EliteTrader — "Can you profitably scalp the forex?"** — [thread](https://www.elitetrader.com/et/threads/can-you-profitably-scalp-the-forex.68211/). Canonical practitioner debate on whether short-horizon direction survives the spread.
10. **Forex Factory — grid/martingale "holy grail" threads** — [example](https://www.forexfactory.com/thread/278178-bks-grid-ea-martingale). How 90%+ win *rates* are manufactured with hidden tail risk; community debunking.
11. **forexop — "Slippage, Requotes and Unfair Price Execution"** — [link](https://forexop.com/learning/slippage-requotes-and-unfair-price-execution/) + **b2broker — "Last Look in Forex"** — [link](https://b2broker.com/news/last-look-in-forex/). Asymmetric slippage / last look — why mid-price backtests over-state live edge; motivates net/asymmetric labels.
12. **daytrading.com & Dukascopy — binary-options scam guides** — [daytrading](https://www.daytrading.com/binary-options-scams), [Dukascopy](https://www.dukascopy.com/swiss/english/marketwatch/articles/binary-trading-scams/). Source of the "75–90% of retail *lose*" statistic that the marketing claims invert.
13. **FOREX.com — "NFP V-shaped reversal"** + **daytrading.com NFP** — [link](https://www.forex.com/en-uk/trading-academy/courses/advanced-strategies/uk-the-non-farm-payrolls-v-shaped-reversal/). Event direction = ~0.50; the tradeable structure is the spike *fade*, not the release direction.
