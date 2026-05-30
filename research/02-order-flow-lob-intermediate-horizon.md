# Order Flow & LOB Predictability: Extending the Edge from Seconds Toward Minutes

Research vector: can aggregated / persistent order flow, multi-level OFI, metaorder persistence, Hawkes models, VPIN, or microprice produce a *minutes-horizon* directional signal for EURUSD, given that we already know raw next-tick imbalance decays to coin-flip by ~1 min and is gone by 5 min?

Scope note: our target is 15m directional accuracy >75% on a near-efficient FX major that is ~97% USD-factor. This report is deliberately skeptical. The single most important external finding (below) is that the most rigorous, FX-specific, multi-pair LOB study to date concludes predictability at 1min–1h is **low** and consistent with EMH — so the bar for any of these ideas is high.

---

## TL;DR (most actionable for our 15m FX goal)

- **The literature confirms our mechanistic finding, not refutes it.** Order-flow → return predictability in FX is strong at 1-minute, *already weak by 15–30 minutes*. The cleanest FX result (FRB/EBS EUR/USD) finds significant price impact at 1-min that "disappears at 15-minute and 30-minute frequencies." A 2025 multi-pair FX LOB study (Petrova–Vilhelmsson–Norden) tests exactly our horizon band (1min–1h) with PCA/LASSO/RF and concludes predictability is "generally low… supporting EMH." Do not expect a clean 75% from order flow alone at 15m. Treat any single idea below as a marginal-lift hypothesis, not a silver bullet.
- **Multi-level / integrated OFI is the one order-flow upgrade with the strongest evidence — but mostly for *contemporaneous* fit, not forward prediction.** Cont–Cucuringu–Cont (2023) show best-level OFI explains ~71% of contemporaneous return variance; PCA-integrated multi-level OFI raises this to ~87%. Forward (predictive) OFI still "decays quickly through time… up to several minutes." Build integrated OFI from your tick quote sizes; it is the best-justified new *feature*, but expect the lift to live at 1–5m, fading by 15m.
- **Cross-asset OFI adds forward predictive content only at short horizons (≤ several min) and only via *lagged* cross-impact** — this overlaps heavily with your already-tried cross-pair lead-lag features. Low marginal value at 15m.
- **The genuinely novel, minutes-lived angle is metaorder / parent-order persistence (long-memory of order flow).** Order-flow sign has *long memory* (power-law autocorrelation out to 100s of lags, persisting at the 3-minute aggregation scale) because institutions split metaorders over minutes-to-hours. This is the one order-flow phenomenon whose *timescale matches 15m*. The actionable hypothesis: detect that a metaorder is *in progress* (persistent same-sign flow + abnormal trade clustering) and bet on continuation for its expected remaining life. This is NOT the same as instantaneous imbalance and you have not tried it.
- **Microprice (Stoikov) is a better *instantaneous* fair-value than mid, but it is a sub-second estimator** — it sharpens your label/entry, it does not extend horizon. Use it to de-noise the target, not as a 15m signal.
- **VPIN is a *toxicity/volatility* signal, not a *direction* signal, and its predictive value is contested.** Andersen–Bondarenko (2014) show VPIN's apparent predictive power is largely a mechanical artifact of trading intensity and it peaked *after* the flash crash. Use a signed/directional toxicity variant at most as a regime/conviction gate, never as the directional signal.
- **Hawkes processes model order-flow *intensity/clustering* well (60–80% of intensity is self/cross-excited) but published work forecasts OFI itself, not minutes-horizon price direction.** Useful as a feature generator (expected residual same-side intensity = "is the metaorder still running"), not as a standalone predictor.
- **Net priority:** (1) integrated multi-level OFI from your quote sizes, (2) a metaorder-in-progress detector exploiting order-flow long memory, (3) signed-toxicity / Hawkes residual-intensity as conviction gates for selective prediction — which is where your current best result (0.632 @ 0.2% coverage) already lives.

---

## Key findings (each with inline citation)

### 1. FX order flow → return predictability dies before 15 minutes (this is the headline)

- **FRB / EBS EUR/USD study:** at the 1-minute frequency interdealer order flow has significant price impact, but "the significance disappears at 15-minute and 30-minute frequencies" for EUR/USD; contemporaneous *daily* order flow explains ~45% R² of daily EUR/USD returns, but that is contemporaneous, not predictive [Order Flow and Exchange Rate Dynamics in Electronic Brokerage System Data, FRB IFDP 830, https://www.federalreserve.gov/pubs/ifdp/2005/830/revision/ifdp830r.pdf]. This is the most direct external corroboration of your own "gone by 5 min" microstructure finding, on exactly your instrument.
- **Petrova, Vilhelmsson, Norden (2025), *Assessing Cross-Currency Predictability in Forex Markets: Insights from Limit Order Book Data*** — the most relevant paper for us. Uses real multi-pair FX **LOB** data, focuses on **short-term forecasts from 1 minute to 1 hour**, applies unsupervised & supervised PCA factor-augmented regressions, LASSO, and random forest, and explicitly tests cross-currency predictability. Conclusion: "the findings reveal **generally low predictability** across various models, supporting the Efficient Market Hypothesis." [SSRN https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5523878 ; Int. J. Forecasting https://www.sciencedirect.com/science/article/pii/S0169207025001281]. This paper effectively *pre-runs our exact experiment* (FX, LOB, 1min–1h, PCA/LASSO/RF, cross-pair) and finds little. Any claim of 75% at 15m must explain why it beats this.
- **Evans–Lyons (2002, and follow-ups):** order flow has strong *contemporaneous/explanatory* power on FX and some out-of-sample forecastability at the **1–3 week** horizon (driven partly by flow forecasting future flow), but daily major-pair order flow series are themselves close to unpredictable [Evans & Lyons, *Order Flow and Exchange Rate Dynamics*, https://faculty.georgetown.edu/evansm1/wpapers_files/orderflow.pdf]. Takeaway: FX order flow's *predictive* (vs contemporaneous) horizon is either sub-minute or multi-day, with a dead zone right where we want it (minutes-hours).

### 2. Multi-level / integrated OFI: the best-evidenced order-flow upgrade — but contemporaneous, not forward

- Cont, Cucuringu, Cont, *Cross-Impact of Order Flow Imbalance in Equity Markets* (Quantitative Finance 2023) [https://arxiv.org/html/2112.13213v4 ; https://www.tandfonline.com/doi/full/10.1080/14697688.2023.2236159]:
  - Best-level OFI (the Cont–Kukanov–Stoikov 2014 definition) explains **71.16%** of in-sample contemporaneous return variation; their **PCA-integrated multi-level OFI** (first principal component across the top 10 LOB levels) raises in-sample adjusted R² to **87.14%** (std 9.16%) — a large jump. Verbatim: "integrated OFIs provide a higher explanatory power for price movements than the widely-used best-level OFIs."
  - **Contemporaneous cross-asset OFI adds nothing once you use integrated OFI:** "once information from multi-level order flow is incorporated… cross-impact terms do not provide additional explanatory power for contemporaneous impact, compared to a parsimonious model without cross-impact." Integrated *self*-OFI absorbs the cross-asset information.
  - **Forward prediction is the weak part:** "cross-impact terms do provide significant information content for intraday forecasting of future returns over a short horizon of **up to several minutes, but their predictability decays quickly through time**." They forecast at h=1 minute with a rolling 30-minute estimation window and extend to f ∈ {2,3,5,10,20,30} min, where the edge fades. Out-of-sample they note cross-asset OFI "can increase out-of-sample R²" at 1-min, but small and decaying.
  - Implication for us: build **integrated multi-level OFI** from quote sizes — best-justified new feature — but it is a *contemporaneous* state variable. It will sharpen 1–5m signals and may marginally help 15m; do not expect it to be the 75% lever.

### 3. The Cont–Kukanov–Stoikov foundation

- Cont, Kukanov, Stoikov, *The Price Impact of Order Book Events* (J. Financial Econometrics 2014) [https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1712822 ; arXiv https://arxiv.org/abs/1011.6402]: over short intervals, price changes are driven mainly by OFI (net of arrivals/cancellations at best bid/ask), with a **linear** price–OFI relation whose slope is inversely proportional to depth. This is a *contemporaneous price-formation* result (mechanics of how flow moves price), not a forecasting result. It is the reason imbalance "predicts" the next tick and the reason that edge is mechanical and fast-decaying.

### 4. Queue/LOB imbalance is a *one-tick-ahead* predictor — confirms the short horizon ceiling

- Gould & Bonart (2016), *Queue Imbalance as a One-Tick-Ahead Price Predictor* [https://arxiv.org/abs/1512.03492]: strongly significant logistic relationship between queue imbalance and the *next* mid-move direction, and notably the improvement is large only for **large-tick** stocks (where the queue is informative). FX spot majors are effectively large-tick / small-spread, which is favorable for the *tick* horizon — but this is explicitly a one-event-ahead predictor. It re-confirms that LOB imbalance lives at the tick, matching your 3-second clean-signal finding.

### 5. Long memory of order flow / metaorders — the one phenomenon at *our* timescale

- Lillo–Mike–Farmer (LMF) and Bouchaud et al.: market-order sign autocorrelation is **long memory**, C(τ) ∝ τ^(−γ), positive and significant out to **100s of lags**, persisting at the **3-minute aggregation scale** [Why is order flow so persistent?, CFM, https://www.cfm.com/wp-content/uploads/2022/12/315-2011-why-is-order-flow-so-persistant.pdf]. Cause is **order splitting**: institutions break metaorders into child orders executed sequentially over minutes-to-hours [Tóth et al.; Sato & Kanazawa 2023, account-level TSE test of LMF, https://arxiv.org/abs/2301.13505 / PRL 131,197401].
- This long memory **also exists in FX spot** [**Gould, Porter & Howison (2016)**, "The Long Memory of Order Flow in the Foreign Exchange Spot Market," https://people.maths.ox.ac.uk/porterm/papers/long-memory-published.pdf — H≈0.7]. **Caveat (verified, C3):** the *aggressive market-order* sign in FX decays fast — the actual Lallouache & Abergel (2014, EBS) paper finds the market-order sign ACF "null after about 2 minutes"; only limit/cancel-sign persistence runs longer (~5 min). Exploitable trade-sign horizon ≈ 2–5 min, **not** "extended multi-minute."
- **Square-root impact + slow decay:** a metaorder of size Q moves price ∝ √Q, and after completion impact decays slowly (≈ as t^(−1/2)), with roughly **~2/3 of peak impact lingering** [Bouchaud, *The Square-Root Law of Market Impact*, https://bouchaud.substack.com/p/the-square-root-law-of-market-impact ; *Market impacts and the life cycle of investors' orders*, https://arxiv.org/abs/1412.0217]. Crucially, *while a metaorder is executing*, price drifts in its direction over the execution window (minutes-to-hours) — a genuinely minutes-horizon directional tilt.
- **Why this is the live idea:** the *instantaneous* imbalance edge decays in seconds (you proved it), but the *persistence of the flow that generates it* has memory measured in minutes. The signal you want is not "current imbalance" but "is a same-signed metaorder still running, and how much of its life remains." That is a different feature family you have not built.
- Detection is hard from public data: Sato & Kanazawa note clustering of child orders matters more than their size, and metaorder identification from public market data is "challenging" [https://arxiv.org/html/2501.17096]. Bayesian change-point detection has been used to segment order-flow regimes online [*Online learning of order flow and market impact with Bayesian change-point detection*, https://arxiv.org/html/2307.02375].

### 6. VPIN: toxicity/volatility, not direction, and contested

- Easley, López de Prado, O'Hara (2012): VPIN buckets volume on a volume clock and measures one-sidedness as a toxicity gauge; spikes precede *volatility/jumps*, used as a flash-crash early-warning [VPIN paper https://www.quantresearch.org/VPIN.pdf].
- **Critique (must heed):** Andersen & Bondarenko (2014, J. Financial Markets 17) — after controlling for trading **intensity/volume**, VPIN has "no incremental predictive power for future volatility," its content is "primarily a mechanical relation with the underlying trading intensity," and VPIN **peaked *after* the flash crash, not before** [https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1881731 ; rejoinder https://repec.econ.au.dk/repec/creates/rp/13/rp13_42.pdf]. VPIN is **unsigned** (toxicity, not direction). At most use a *signed* variant as a conviction/regime gate.

### 7. Hawkes processes: clustering/intensity modeling, not a direction forecaster (yet)

- State-dependent / queue-reactive Hawkes models explain **60–80% of total order-arrival intensity** via self- and cross-excitation [Morariu-Patrichi & Pakkanen, https://www.tandfonline.com/doi/full/10.1080/14697688.2021.1983199 ; Wu et al. queue-reactive Hawkes https://arxiv.org/pdf/1901.08938].
- Anantha & Jain (2024), *Forecasting high-frequency order flow imbalance using Hawkes processes* [https://arxiv.org/html/2408.03594v1]: forecasts **next-1-minute OFI** (NIFTY futures, one day of tick data), evaluated by negative-log-likelihood / superior-predictive-ability tests on the *OFI distribution* — **not** directional price accuracy, single-day sample. Useful as a method to estimate "expected residual same-side intensity," not as evidence of minutes-horizon direction.

### 8. Deep LOB models: high "accuracy" ≠ tradable, and horizon-limited

- Zhang, Zohren, Roberts, *DeepLOB* (IEEE TSP 2019) [https://arxiv.org/abs/1808.03668]: CNN+LSTM on 10-level LOB; strong mid-price-direction accuracy on FI-2010 / LSE — but at *event/tens-of-events* horizons, and the benchmark is contested for leakage/label construction.
- Critical follow-up — Briola, Bagnara, Zohren et al., *Deep limit order book forecasting: a microstructural guide* (Quant. Finance 2025) [https://www.tandfonline.com/doi/full/10.1080/14697688.2025.2522911 ; LOBFrame code https://ideas.repec.org/p/ehl/lserod/128950.html]: on NASDAQ stocks, "high forecasting power does **not necessarily correspond to actionable trading signals**," standard ML metrics "fail to adequately assess… forecasts in the LOB context," and predictability rate is tied to microstructural properties (tick size, liquidity). This is the most important sobering note for deep-LOB hype.

---

## Concrete techniques / features / architectures to try

Ranked roughly by expected value for our 15m FX goal. All assume we use the raw sub-second bid/ask quotes **with sizes** we already have (this is what unlocks real OFI vs the signed-volume proxy that gave ~0 lift).

### A. Integrated multi-level OFI (best-justified new feature) — HIGH
Implement the Cont–Cucuringu–Cont definition exactly:
1. From quotes, at each LOB level m (m=1..L; you have top-of-book + sizes, use as many levels as your tick data provides), compute the per-event signed queue-size change: `e_m(t)` = (Δbid_size if bid not worse) − (Δask_size if ask not worse), with the standard CKS sign rules for price-level moves.
2. Aggregate `OFI_m` over a bar interval (try 10s, 30s, 1m, 5m, 15m windows — multi-scale).
3. **PCA-integrate**: stack `[OFI_1..OFI_L]`, take the first principal component → `OFI_integrated`. Normalize by rolling depth.
4. Features: `OFI_integrated` at multiple lookback windows; its sign-persistence; its EWMA at several decay constants matched to metaorder timescales (1m, 5m, 15m, 30m).
Expectation: meaningful contemporaneous/1–5m lift; marginal at 15m. This is the correct upgrade of your "signed 10s volume OFI proxy."

### B. Metaorder-in-progress detector (the novel, timescale-matched idea) — HIGH
Goal: estimate "a same-signed institutional metaorder is currently executing; bet continuation over its remaining life."
- **Persistence features:** rolling order-flow-sign autocorrelation; Hurst/long-memory exponent of signed flow over the last N minutes; fraction of last K trades on the same side; run-length of dominant-sign flow.
- **Clustering features (Hawkes residual intensity):** fit a 2D (buy/sell) Hawkes (sum-of-exponentials kernel) on trade arrivals; the feature is *expected residual same-side intensity* = how much more same-side flow the self-excitation predicts. High residual same-side intensity = metaorder likely still running. Use multiple kernel timescales (seconds, minutes).
- **Change-point gate:** run online Bayesian change-point detection on the order-flow sign process (per Tsaknaki/Lillo); only trade *inside* a stable persistent-flow regime, flatten at detected change-points.
- **Square-root-decay-aware label:** condition expected continuation on estimated metaorder maturity — early in a run, more expected drift remains; near exhaustion, fade.
- Train a classifier on these persistence/clustering features to predict 15m direction; evaluate strictly OOS. This is the one feature family whose physics lives at minutes.

### C. Signed toxicity / directional-VPIN as a *conviction gate* — MED
- Build VPIN on a **volume clock** (volume buckets, not wall-clock) but keep it **signed** (net buy−sell fraction per bucket), and bucket-classify trades with tick rule / BVC.
- Do NOT use raw VPIN level as the direction. Use it as: (i) a *gate* (only take directional bets when signed-toxicity conviction is high — feeds directly into your existing selective-prediction framework that already hits 0.632@0.2%), and (ii) a volatility/regime feature.
- Heed Andersen–Bondarenko: always include raw trading-intensity/volume as a control so you're not just re-learning volume.

### D. Microprice for label/entry de-noising — MED (not a horizon extender)
- Compute Stoikov microprice [https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2970694] as the mid adjusted by spread × imbalance toward its martingale fair value.
- Use it to (i) define a cleaner 15m target (microprice-to-microprice return removes bid-ask bounce that adds label noise) and (ii) time entries. It will not by itself extend the predictive horizon, but cleaner labels can raise measured accuracy at fixed signal.

### E. Cross-asset *lagged* integrated OFI — LOW/MED (overlaps prior work)
- A sparse-LASSO regression of EURUSD 1–5m return on lagged integrated OFI of all 7 USD pairs (the cross-impact spec). Literature says this helps only ≤ several minutes and overlaps your cross-pair lead-lag features. Worth a quick confirm at 1–5m; expect little new at 15m.

### F. Deep LOB models — LOW for 15m
- DeepLOB-style CNN-LSTM on raw multi-level quote/size tensors *can* be tried, but (i) it targets event/seconds horizons, (ii) "high accuracy ≠ tradable" (LOBFrame), (iii) high leakage risk. If tried, use LOBFrame's evaluation discipline and target ≤1m, not 15m.

---

## Reported results & CREDIBILITY assessment

**Credible / reproducible (trust, with caveats):**
- Cont–Kukanov–Stoikov linear OFI→price-change law (peer-reviewed JFE; replicated widely incl. Cont–Cucuringu–Cont's 71% contemporaneous R²). **Contemporaneous, sub-minute.** Credible but explains why the edge is fast.
- Cont–Cucuringu–Cont integrated-OFI 71%→87% contemporaneous R² jump and "predictability decays within several minutes" (Quant Finance, temporal-ordered OOS, NASDAQ equities). Credible; the forward-decay statement is exactly the caution we need.
- Gould–Bonart queue-imbalance one-tick-ahead predictor (peer-reviewed, 10 stocks, 1yr). Credible, explicitly one-event horizon.
- Long-memory of order flow / LMF / square-root impact (decades of replication across equities, futures, crypto, **and FX spot**; account-level TSE confirmation Sato–Kanazawa, PRL). Credible. The metaorder *continuation* trade is the plausible but **unproven-for-us** extrapolation.
- **Petrova–Vilhelmsson–Norden 2025 FX-LOB low-predictability result** (Int. J. Forecasting, our exact horizon band and methods). Credible and *adverse* to optimism — treat as the prior to beat.
- Andersen–Bondarenko VPIN critique (J. Financial Markets). Credible; downgrades VPIN to a controlled gate at best.

**Overfit / hype / leakage-prone (discount heavily):**
- "VPIN spike >0.7 ⇒ imminent directional move" practitioner blogs — unsigned toxicity dressed up as direction; contradicted by Andersen–Bondarenko on the predictive claim.
- DeepLOB-family "90%+ accuracy on FI-2010" — horizon is tens of events, FI-2010 labels/normalization are leakage-prone, and LOBFrame (2025) shows accuracy doesn't map to PnL. Any online "75–90% FX direction" claim citing such models is almost certainly horizon-mismatched, leaked, or marketing.
- Single-day Hawkes OFI forecasts (Anantha–Jain) — method is sound, but one trading day, one instrument, forecasts OFI distribution not price direction; not evidence of tradable minutes-horizon direction.

**Leakage red flags to police in our own builds:**
- OFI/microprice computed with the *closing* quote of the bar leaks the move; compute features strictly from quotes *before* the prediction timestamp.
- Volume-clock VPIN can leak future volume if bucket boundaries straddle the prediction point — close buckets only on past volume.
- Cross-pair features must respect synchronized timestamps; asynchronous last-tick carries lookahead.

---

## Data sources needed

- **What we already have is the right data** (and the key advantage): raw sub-second bid/ask quotes **with sizes** for 7 USD pairs. This is what makes *real* multi-level OFI, microprice, queue imbalance, and Hawkes intensity computable — the signed-10s-volume proxy that gave ~0 lift was a degraded substitute. Mine the quote-size ladder.
- **Gap — true multi-level LOB depth:** real OFI/integrated-OFI wants ≥5–10 levels. Spot FX is fragmented (no consolidated tape). If your feed is top-of-book only, integrated OFI is limited to L1–L2.
  - *EBS Market Data / EBS Live* (paid, institutional): genuine interdealer EUR/USD LOB depth — the gold standard used in the FX academic papers. Expensive.
  - *Refinitiv FXall / Dealing* depth, *LMAX Exchange* (CLOB with visible depth, retail-accessible), *Integral*, *Hotspot/CBOE FX* — varying depth, paid.
  - *Dukascopy* (free historical tick bid/ask + volume for majors) — good for tick OFI proxy, limited depth.
- **Trade-direction classification:** with quotes you can apply tick-rule / Lee-Ready / Bulk-Volume-Classification to sign flow for VPIN/Hawkes without a true trade tape.
- **Metaorder ground truth:** unobtainable publicly (it's account-level). The whole point of the LMF line is *inferring* metaorder presence from public statistics — so you don't need it, you estimate it.

---

## Relevance & priority for OUR project

Interaction with what we've already ruled out:
- Our "signed 10s-volume OFI proxy ≈ 0 lift" is **not** a clean rejection of OFI — it tested a degraded, single-level, wall-clock proxy. **Integrated multi-level OFI from quote sizes (A) is a materially different, better-specified feature.** Worth a clean re-test. (HIGH)
- Our cross-pair lead-lag features already capture most of what *cross-asset OFI* (E) offers per the literature → low marginal value. (LOW/MED)
- Our selective-prediction result (0.632 @ 0.2% coverage) is exactly where toxicity/Hawkes-intensity **conviction gates** (C) should plug in — they don't need to predict direction, just to identify the rare high-conviction windows. (MED, synergistic with existing best result)
- The **metaorder-in-progress detector (B)** is the only idea here operating natively at the 15m timescale and is genuinely **not** in our exhausted list. It is also the highest-risk/highest-novelty. (HIGH)
- Microprice (D) improves labels/entries, orthogonal to everything tried. (MED)
- Deep LOB (F): low priority at 15m; horizon-mismatched. (LOW)

**Honest expectation setting:** the FX-specific evidence (FRB EBS EUR/USD 15m impact "disappears"; Petrova et al. 2025 "generally low predictability… EMH") says order flow alone almost certainly will NOT reach 75% at 15m on EURUSD. The realistic win is *marginal lift on selective high-conviction windows*, where ideas A+B+C combine: integrated-OFI persistence (A) + metaorder-running detector (B) feeding a toxicity-gated selective predictor (C). That stacks onto your 0.632@0.2% rather than replacing it.

Ranked:
- **HIGH:** A (integrated multi-level OFI), B (metaorder-in-progress / order-flow long-memory detector).
- **MED:** C (signed-toxicity / Hawkes residual-intensity conviction gate), D (microprice label de-noising).
- **LOW:** E (cross-asset lagged OFI — redundant with cross-pair), F (deep LOB — horizon mismatch).

---

## Sources (annotated)

1. **Order Flow and Exchange Rate Dynamics in Electronic Brokerage System Data** — FRB IFDP 830 (rev.). https://www.federalreserve.gov/pubs/ifdp/2005/830/revision/ifdp830r.pdf — *EUR/USD EBS: order-flow price impact significant at 1-min, vanishes by 15–30 min. The direct FX corroboration of our decay finding.*
2. **Petrova, Vilhelmsson, Norden (2025), Assessing Cross-Currency Predictability in Forex Markets: Insights from LOB Data** — Int. J. Forecasting. https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5523878 / https://www.sciencedirect.com/science/article/pii/S0169207025001281 — *Our exact experiment (FX LOB, 1min–1h, PCA/LASSO/RF, cross-pair) → "generally low predictability, supporting EMH." The prior to beat.*
3. **Cont, Cucuringu, Cont (2023), Cross-Impact of Order Flow Imbalance in Equity Markets** — Quant. Finance. https://arxiv.org/html/2112.13213v4 / https://www.tandfonline.com/doi/full/10.1080/14697688.2023.2236159 — *Integrated multi-level OFI 71%→87% contemporaneous R²; forward predictability "decays within several minutes"; cross-impact absorbed by integrated self-OFI. Blueprint for feature A.*
4. **Cont, Kukanov, Stoikov (2014), The Price Impact of Order Book Events** — J. Fin. Econometrics. https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1712822 / https://arxiv.org/abs/1011.6402 — *The linear OFI→price law; foundation, explains why the edge is mechanical and fast.*
5. **Gould & Bonart (2016), Queue Imbalance as a One-Tick-Ahead Price Predictor** — Market Microstructure & Liquidity. https://arxiv.org/abs/1512.03492 — *LOB imbalance predicts the next mid-move, strongest for large-tick (FX-like) instruments. Confirms tick-horizon ceiling.*
6. **Lillo, Mike, Farmer; Bouchaud et al. — Long memory of order flow / "Why is order flow so persistent?"** https://www.cfm.com/wp-content/uploads/2022/12/315-2011-why-is-order-flow-so-persistant.pdf — *Power-law sign autocorrelation persisting at 3-min scale via order splitting. The physics behind feature B.*
7. **Sato & Kanazawa (2023), quantitative test of the Lillo-Mike-Farmer model (TSE account-level)** — Phys. Rev. Lett. 131,197401. https://arxiv.org/abs/2301.13505 — *Account-level confirmation that order-flow long memory = metaorder splitting; child-order clustering > size for impact.*
8. **Gould, Porter & Howison (2016), The Long Memory of Order Flow in the Foreign Exchange Spot Market** — Market Microstructure & Liquidity (arXiv:1504.04354). https://people.maths.ox.ac.uk/porterm/papers/long-memory-published.pdf — *Long-memory order flow (H≈0.7) exists in FX spot — license to apply feature B to EURUSD. NB: the separate Lallouache–Abergel (2014, EBS) paper finds FX aggressive market-order sign decays to null in ~2 min, so the usable trade-sign horizon is ~2–5 min.*
9. **Bouchaud, The Square-Root Law of Market Impact (+ Market impacts and the life cycle of investors' orders, 2014)** https://bouchaud.substack.com/p/the-square-root-law-of-market-impact / https://arxiv.org/abs/1412.0217 — *√Q impact, slow t^(−1/2) decay, ~2/3 lingering — quantifies the minutes-horizon drift while a metaorder executes.*
10. **Easley, López de Prado, O'Hara (2012), VPIN / Flow Toxicity** https://www.quantresearch.org/VPIN.pdf — *Volume-clock toxicity gauge; precedes volatility/jumps. Unsigned — toxicity not direction.*
11. **Andersen & Bondarenko (2014), VPIN and the Flash Crash** — J. Financial Markets 17. https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1881731 / rejoinder https://repec.econ.au.dk/repec/creates/rp/13/rp13_42.pdf — *VPIN's predictive power is mostly a trading-intensity artifact; peaked AFTER the crash. Demotes VPIN to a controlled gate.*
12. **Morariu-Patrichi & Pakkanen — State-dependent Hawkes processes for LOB** https://www.tandfonline.com/doi/full/10.1080/14697688.2021.1983199 — *60–80% of order-arrival intensity is self/cross-excited; basis for Hawkes residual-intensity feature in B/C.*
13. **Anantha & Jain (2024), Forecasting HF Order Flow Imbalance using Hawkes Processes** https://arxiv.org/html/2408.03594v1 — *Method to forecast next-1-min OFI distribution (NIFTY, 1 day). Feature-generator, not direction evidence.*
14. **Stoikov (2018), The Micro-Price: A High-Frequency Estimator of Future Prices** https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2970694 — *Imbalance-adjusted martingale fair value; better short-term predictor than mid. Use for label/entry de-noising (D).*
15. **Zhang, Zohren, Roberts (2019), DeepLOB** — IEEE TSP. https://arxiv.org/abs/1808.03668 — *CNN-LSTM on 10-level LOB; strong event-horizon accuracy. Horizon-mismatched for 15m; leakage-prone benchmark.*
16. **Briola et al. (2025), Deep Limit Order Book Forecasting: A Microstructural Guide (LOBFrame)** — Quant. Finance. https://www.tandfonline.com/doi/full/10.1080/14697688.2025.2522911 — *"High forecasting power ≠ actionable signals"; standard ML metrics mislead in LOB context. The deep-LOB reality check.*
17. **Tsaknaki, Lillo et al. (2024), Online learning of order flow & market impact with Bayesian change-point detection** https://arxiv.org/html/2307.02375 — *Online segmentation of order-flow regimes; the change-point gate in feature B.*
