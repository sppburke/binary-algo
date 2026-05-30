# Cross-Asset & Macro Lead-Lag Signals for 15-Minute FX Direction

Research vector: intraday lead-lag from other markets (rates/yields, equity index futures, DXY, commodities, VIX/risk sentiment, CIP/cross-currency basis, triangular-arb residuals) into liquid USD FX majors, evaluated against our existing ~0.52 AUC / ~0.527 ceiling and our "true microstructure decays by 1-5 min" finding.

---

## TL;DR

- **The honest academic consensus is against us at 15m.** The most rigorous, recent, FX-specific out-of-sample study (Petrova, Vilhelmsson & Norden 2025, LMAX LOB, 5 pairs, 1-min to 1-hour, LASSO/RF/sPCA, transaction-cost aware) finds **predictability essentially everywhere absent** — the *only* surviving case was AUD/USD at the *1-hour* horizon via random forest / supervised PCA. This independently corroborates our own ceiling and means the bar for any new cross-asset signal is "beat a near-EMH null OOS," not "find correlation."
- **The one robust, repeatedly-confirmed cross-asset lead is FX *futures* (CME) leading interbank *spot*** in price discovery (NY Fed SR262, Chen & Gau / Rosenberg & Traub). This is a *venue/microstructure* lead measured in milliseconds-to-seconds, not a 15-minute macro signal — so it is closer to our already-strong 3-second tick edge than to the 15m gap. Treat it as a faster-execution / data-fidelity idea, not a new 15m alpha.
- **Lead-lag forecasting "works" but plateaus near 55-60% even with the leader's full history** (Huth & Abergel 2014: ~60% next-midquote-move accuracy of the *lagger* using the *leader's* past, futures→stock). That 60% is a high-frequency, single-step, no-spread-cost number — exactly the regime we already exploit at 3s, and it *decays into the noise* by the time you reach minutes, matching our decay finding.
- **The strongest *macro* anchor for EURUSD is the 2-year (and 10-year) US-vs-German yield/rate-futures differential.** It is a genuine, economically-grounded driver (a ~1% differential move ≈ 10-15 "big figures" in EURUSD over time), and it updates intraday through Bund/Schatz and Treasury/SOFR futures *which are themselves liquid, tick-level, and often lead spot FX around data*. This is the single most promising untried orthogonal *level/slow-moving* feature for us.
- **Risk-on/off (VIX, ES/NQ, credit) is real but mostly a contemporaneous co-factor, not a clean lead, for EUR/USD.** It matters far more for JPY/CHF (safe havens) and AUD/CAD/NZD (risk-on commodity FX) than for EURUSD, which is ~97% USD-factor for us. Use risk-state as a *regime gate / conditioning variable*, not a standalone directional predictor.
- **Commodity→currency leads (oil→CAD, gold→AUD) exist but the causality flips and is horizon-dependent**; at intraday frequency several papers find currency→commodity causality dominates, so oil is a weak *lead* for USDCAD intraday. Lower priority than rates.
- **Triangular-arbitrage residuals are largely a "mirage" at executable sizes** (Fenn et al. 2009): apparent mispricings mostly vanish once bid/ask and the synchronization of the three legs are handled. The *useful* version is not arbitrage profit but using which leg adjusts last as a 1-3s direction hint — again microstructure, not 15m.
- **Best credible *new* direction for us: build a real-time cross-asset "fair-value gap" for each USD pair** — a small regression of the pair on contemporaneous rate-differential futures, DXY/dollar-basket, and a risk factor — then predict 15m direction from the *residual's* sign and the *velocity of the driver futures* (which lead spot FX into and around macro releases). This is the one construction that (a) is orthogonal to our TA/order-flow features, (b) is grounded in mechanism, and (c) has not been in our V1-V17 ledger.

---

## Key findings (with inline citations)

### 1. The rigorous OOS verdict on intraday FX predictability is bleak — and that is the most important finding

Petrova, Vilhelmsson & Norden (2025), *"Assessing cross-currency predictability in forex markets: Insights from limit order book data,"* International Journal of Forecasting / SSRN 5523878, use full LOB data on 5 pairs from **LMAX** (a real, low-latency FX ECN), horizons **1 minute to 1 hour**, with factor-augmented regressions, **supervised and unsupervised PCA, LASSO, and random forests**, and crucially evaluate **out-of-sample with transaction costs**. Their headline: *"findings reveal generally low predictability across various models, supporting the Efficient Market Hypothesis. … the only case in which predictability was found is for the AUD/USD currency pair, estimated using random forests and supervised PCA at the one-hour frequency."*
- Why it matters for us: this is the closest published analogue to our setup (LOB micro features, cross-currency, ML, strict OOS, costs) and it lands almost exactly where we did. It tells us (a) our ~0.52-0.527 ceiling is *not* a bug in our pipeline, it is the regime; (b) any cross-asset claim of 70%+ at 15m that does *not* match this rigor should be presumed leaked/overfit; (c) the one positive (AUD, 1h, RF/sPCA) hints that *commodity/risk FX at slightly longer horizons* is where weak signal survives. Source: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5523878 and https://www.sciencedirect.com/science/article/pii/S0169207025001281

The companion observation from the same literature: *"pockets of predictability exist … no predictor is consistently selected over the whole sample,"* i.e. predictability is **time-varying and non-persistent** — which is fatal for a static 75%-always model but compatible with a *selective-prediction / regime-gated* approach (which we already explored to 0.632 at 0.2% coverage).

### 2. Cross-asset / lead-lag forecasting tops out ~55-60% at the fast end and decays to coin-flip at minutes

Huth & Abergel (2014), *"High-frequency lead/lag relationships — Empirical facts,"* J. Empirical Finance (arXiv 1111.7103). Using the Hayashi-Yoshida asynchronous cross-correlation estimator on tick data they report: most-liquid assets (short intertrade duration, narrow spread, high turnover) **lead** less-liquid ones; strongly **asymmetric** cross-correlation, *especially future→stock*; **~60% accuracy forecasting the next midquote move of the lagger using only the leader's past** (vs the lagger's own past). Two caveats they stress and that bite us: *"a naive strategy based on market orders cannot make any profit … because of the bid/ask spread,"* and the effect has **intraday seasonality, peaking at macro releases and the US open**. Source: https://arxiv.org/abs/1111.7103
- For us: the 60% is the *same kind of edge* as our 55.3% next-tick and our clean >75% at 3s — a fast, single-step, pre-cost microstructure number. It confirms the lead exists but **does not extend to a 15-minute horizon**; by minutes it is gone, matching our own decay measurement.

DeltaLag (Zhou et al., 2025, ACM ICAIF; arXiv 2511.00390) is the current SOTA *learned* lead-lag method: a **sparsified cross-attention** model that discovers *pair-specific, time-varying* lags end-to-end and aligns lagged leader features to predict the lagger. It beats temporal/spatio-temporal baselines on IC/annualized-return/Sharpe on equities. Source: https://arxiv.org/abs/2511.00390
- For us: the *architecture* (learned dynamic lag + cross-attention over a panel of "leader" assets) is directly transplantable to a cross-asset FX panel (rate futures, DXY, ES, gold, peer pairs) → target pair. This is a concrete untried model. But note it is validated on *equities at daily-ish frequency with portfolio metrics*, not 15m FX accuracy, so expect attenuation.

### 3. FX *futures* lead interbank *spot* in price discovery — but it is a microsecond-to-second venue lead

NY Fed Staff Report 262 (Rosenberg & Traub), *"Price Discovery in the Foreign Currency Futures and Spot Market,"* finds via **Hasbrouck (1995) information shares** that *"the FX futures market contributes more to price discovery than does the spot market,"* and both futures and spot order flow carry unique information; the lower-transparency spot market *"may respond more slowly to information."* Source: https://www.newyorkfed.org/medialibrary/media/research/staff_reports/sr262.pdf
- Counter-evidence: Cabrera, Wang & Yang (and the broader literature) find **EBS spot can lead** depending on regime, so the lead is venue- and period-dependent, not a free directional signal. EBS data is available at **5 ms delay / 100 ms slices**; CME FX futures tick is similar order. Source (latency facts): https://www.cmegroup.com/markets/ebs/ebs-data-and-analytics.html
- For us: this is a **data-fidelity / execution** insight, not 15m alpha. If our spot feed lags the CME EURUSD future by a few hundred ms around bursts, ingesting the future could *sharpen our existing 3-second edge*. It does **not** create a minutes-horizon signal.

### 4. The rate/yield-differential channel is the strongest *macro* anchor and is genuinely untried by us

Multiple sources converge: the EURUSD market trades *"as a proxy for 2-year sovereign rate differentials far more than any other maturity,"* and a ~1% differential move maps to roughly **10-15 big figures** in EURUSD; Treasury vs Bund yields are repeatedly cited as the dominant intraday/medium-term driver of USDJPY/EURUSD/GBPUSD direction (FXEmpire, MacroMicro US-Germany 10Y spread vs EURUSD). The German bond futures complex (Schatz/Bobl/Bund/Buxl) is **tick-level, deeply liquid** (Bund > 281M contracts in 2025) and well-characterized microstructurally (arXiv 2401.10722). Treasury futures (ZT/ZF/ZN) and SOFR futures are equally liquid intraday. Sources: https://en.macromicro.me/collections/2204/euro-dollar/17798/euro-us-germany-10y-treasury-note-spread , https://arxiv.org/html/2401.10722v1 , https://www.fxempire.com/forecasts/article/interest-rates-forecast-treasury-yield-surge-drives-usdjpy-eurusd-and-gbpusd-1599798
- Mechanism for a *lead*: yields/rate-futures often reprice on macro data *milliseconds to seconds before* spot FX fully adjusts (the Huth-Abergel seasonality peaks exactly at releases). So the **change in the 2y or 10y US-DE differential over the last N seconds/minutes** is a candidate orthogonal predictor of the *next* 15m FX move, especially in the post-release window. This is distinct from our event-time vol-seasonality proxy (which was redundant with time-of-day) because it is the **signed magnitude of the actual rate move**, not just "a release happened."

### 5. Risk-on/off and the dollar factor: co-factors, not clean leads (for EURUSD specifically)

- Brunnermeier, Nagel & Pedersen (carry/crashes) and the KC Fed risk-on/off index work establish that **rising VIX → carry unwind, risk-reversals reprice, high-yield/commodity FX (AUD/NZD/CAD/MXN) sell off, JPY/CHF rally.** Sources: https://www.fmg.ac.uk/sites/default/files/2020-08/M-Brunnermeier.pdf , https://www.kansascityfed.org/documents/10594/rwp24-12charistedmanlundblad.pdf
- Verdelhan (2018) / Lustig-Roussanov-Verdelhan: exchange rates are driven by a **dollar factor + carry factor**; the two HML factors explain **18%-83% of monthly bilateral USD exchange-rate moves**; the dollar factor dominates. Source: https://www.ecb.europa.eu/events/pdf/conferences/130627/2.1a_A.Verdelhan_Paper.pdf
- For us: EURUSD being ~97% USD-factor in our data is *exactly* the Verdelhan dollar-factor result. The implication is that **a tradable EURUSD signal must come from the residual euro-leg or from leading the dollar factor itself**, not from generic risk sentiment. VIX/ES are most useful as a **regime variable** that (a) switches *which* peer leads, and (b) gates when commodity-FX signals are live.

### 6. Commodity→currency leads are real but intraday-causality is ambiguous

- Oil↔USDCAD: ~0.75-0.80 historical correlation, but high-frequency wavelet-Granger work finds *"causal flows are more pronounced at longer time horizons and from currency markets to the crude oil market"* — i.e. at intraday horizons the lead can run **CAD→oil**, not oil→CAD. Source: https://www.sciencedirect.com/science/article/abs/pii/S0140988319303020
- Gold→AUD and gold jumps: gold intraday jumps are rare (~0.43% of intervals) and *"US macroeconomic news predicts 34% of price jumps in gold, with FOMC the dominant driver"* — so gold and AUD largely share a *common* macro driver rather than one cleanly leading the other. Source: https://www.sciencedirect.com/science/article/abs/pii/S1057521925004673
- For us: commodity leads are **lower priority than rates** for USD majors, and for EURUSD specifically nearly irrelevant. They are most relevant if we extend to USDCAD/AUDUSD targets.

### 7. Triangular-arbitrage residuals are mostly a mirage at executable size

Fenn, Howison, McDonald, Williams & Johnson (2009), *"The Mirage of Triangular Arbitrage in the Spot Foreign Exchange Market"* (arXiv 0812.0913): apparent triangular mispricings *largely disappear* once you use executable bid/ask and account for the asynchronicity of the three legs; persistent profitable opportunities are essentially absent. The *useful residue* (per the wavelet-UHF triangular-arb literature, Sci. Direct S0264999318319072) is that **one leg adjusts to new information with a lag**, so the residual sign tells you *which pair is stale* — a 100ms-to-seconds direction hint. Source: https://arxiv.org/abs/0812.0913
- For us: do **not** chase triangular-arb profit. Do consider the **triangular residual (e.g. EURUSD × USDJPY vs EURJPY) as a 1-5s "which leg is stale" feature**, feeding our fast model — not the 15m model.

### 8. Where weak intraday predictability *does* survive (cross-asset, ML, with discipline)

Aleti, Bollerslev & Siggaard (2025, *Management Science* 71(9)), *"Intraday Market Return Predictability Culled from the Factor Zoo"*: using **lagged high-frequency cross-sectional factor returns** + ML regularization + separating continuous vs (non-predictable) jump components, they get **sizeable OOS Sharpe/alpha after costs** for liquid ETFs — but *"most of the superior performance traces to periods of high economic uncertainty and a few factors related to tail risk and liquidity,"* attributing it to **slow-moving capital / gradual information incorporation**. Source: https://public.econ.duke.edu/~boller/Papers/MS_2025.pdf
- For us: two transferable lessons — (1) **split the target return into continuous vs jump**; cross-asset leads only predict the *continuous* part, so don't train on bars containing release jumps. (2) **The edge concentrates in high-uncertainty regimes** → strong argument to *condition/gate* on a realized-vol or VIX regime, consistent with our selective-prediction result.

---

## Concrete techniques / features / architectures to try

All designed to be **causal**, OOS-safe (train 2012-21 / val 2022-23 / test 2024-25), and orthogonal to our existing 239 TA + cross-pair + OFI features.

### A. Cross-asset fair-value-gap (residual) features  — HIGH priority
1. At each 15m decision point, regress (rolling, expanding-window, *causal*) the target pair's recent return on contemporaneous returns of: (i) **2y and 10y US-minus-DE rate-futures-implied yield** (or directly Treasury futures ZF/ZN and Bund/Schatz futures returns), (ii) a **dollar-basket** built from your 7 USD pairs (you already have this from stat-arb), (iii) a **risk factor** (ES or NQ future return, and/or VIX change). Use shrinkage (ridge) for stability.
2. Features = the **residual** (sign + z-score) and the **driver velocities**: change in the rate differential over the last 30s/1m/5m, change in DXY/dollar-basket, ES return over last 1m. Hypothesis: spot FX *under-reacts* short-term to a sharp rate-diff move → residual mean-reverts toward fair value over the next 15m, giving signed direction.
3. This is **not** in V1-V17: our event-time proxy used "release happened" seasonality; this uses the *signed realized rate move*, which carries direction, not just vol.

### B. Driver-leads-spot, release-window model — HIGH priority (conditional)
- Build a **release-window regime flag** (you have the calendar). In the [0, +15m] window after Tier-1 releases (NFP, CPI, FOMC, ECB, PMIs), train a *separate* model whose key features are the **post-release jump in 2y/10y differential and in ES/VIX**. Huth-Abergel shows the lead-lag effect is strongest exactly here. Expect this to be the highest-accuracy sub-population even if all-sample accuracy stays modest — report accuracy *conditional on regime*.

### C. Continuous-vs-jump decomposition (Bollerslev lesson) — MED-HIGH
- Use a bipower-variation / threshold estimator to split each pair's path into continuous and jump parts. Train the cross-asset model to predict the **continuous** component only; exclude or separately model jump bars. Cross-asset leads are theoretically non-predictive for the discontinuous part, so mixing them depresses measured accuracy.

### D. Learned dynamic lead-lag (DeltaLag transplant) — MED
- Implement a **sparsified cross-attention** net over a panel of leaders {ZN, ZF, Bund, ES, NQ, DXY, gold, the 6 peer pairs} → target pair, with **learnable per-leader lag offsets** (Gumbel-softmax over a small lag grid, e.g. 0-300s). Output = 15m direction logit. This generalizes our fixed cross-pair lead-lag features to *learned, time-varying* lags and lets the model pick the leader-of-the-moment. Validate hard against overfitting (the panel is wide).

### E. FX-futures tape ingestion to sharpen the *fast* model — MED (execution, not 15m)
- If feasible, add **CME EURUSD/6E (and 6B/6J/6C) futures top-of-book** to your tick pipeline. NY Fed SR262 says futures lead spot in price discovery; even a few-hundred-ms lead would improve your already-clean 3s edge and your fill simulation. Do *not* expect this to move the 15m number.

### F. Triangular-residual fast feature — LOW-MED
- Compute the synthetic-vs-direct mispricing for triangles among your 7 pairs (e.g. EURUSD·USDJPY vs EURJPY) at tick level; feed sign/size as a **1-5s** staleness feature to the fast model. Per Fenn et al., do not treat it as 15m alpha.

### G. Risk-regime gate (Verdelhan/carry lesson) — MED (as a wrapper, not a feature)
- Use VIX level/term-structure and ES realized vol to define 3 regimes (calm / stressed / unwinding). Apply commodity-FX and carry-residual signals **only** in the matching regime; for EURUSD, expect the dollar-factor/rate-diff signal to dominate in all regimes. This formalizes "which peer leads when."

---

## Reported results & CREDIBILITY assessment

**Credible / reproducible (rigorous OOS or peer-reviewed, methodology transparent):**
- Petrova, Vilhelmsson & Norden (2025) — *most credible and most relevant*: strict OOS + costs, finds near-zero FX intraday predictability except AUD/USD@1h. **Believe it; it bounds our expectations.** Leakage risk: low (they explicitly guard).
- Huth & Abergel (2014, J. Emp. Fin.) — ~60% lagger-next-move accuracy from leader; **pre-cost, single-step, HF**. Credible but *not a 15m or net-of-cost result*; they themselves note spread kills naive PnL.
- NY Fed SR262 / Hasbrouck information-share work — futures lead spot in price discovery. Credible, but a venue/ms-level lead, **not** a 15m directional alpha.
- Verdelhan (2018) / Lustig-Roussanov-Verdelhan (RFS 2011) — dollar+carry factor structure; **explains why EURUSD is ~97% USD-factor for us.** Credible, peer-reviewed; it is a *constraint*, telling us where alpha cannot come from.
- Aleti, Bollerslev & Siggaard (2025, *Management Science*) — cross-sectional factor-zoo predicts intraday *market index* with OOS Sharpe after costs; concentrated in high-uncertainty regimes. Credible; transferable *method* but on equities, not 15m FX.
- Fenn et al. (2009) — triangular arbitrage is mostly illusory at executable size. Credible debunking; **use it to avoid a dead end.**

**Promising but unvalidated for our exact problem (transplant with skepticism):**
- DeltaLag (2025, ICAIF) — strong architecture, but reported on equity portfolios with IC/Sharpe, *not* 15m FX accuracy. Risk: wide leader panel → overfit; the IC gains may not survive at 15m FX with our null.

**Hype / leakage-prone (discount heavily):**
- Generic "EUR/USD ML 70-90% accuracy" blog/Kaggle/marketing claims. The clean peer-reviewed EURUSD ML papers land at **54-59% daily** (Deriaz 2024: 58.52% 1-day; PCA variants 54.98-57.23%) — anything claiming far more almost always has **look-ahead leakage** (using the close to predict the same bar's direction, normalizing with full-sample stats, target encoding leak, or ignoring spread/costs). Treat 75%+ at 15m from any single feature family as a red flag until reproduced OOS with costs. Sources: https://arxiv.org/abs/2409.04471
- Oil→CAD / gold→AUD "correlation = causation, just trade it" practitioner pieces — the academic HF-causality is **ambiguous/reversed intraday**; correlation is contemporaneous co-movement, not a lead.

**Leakage checklist to enforce on every cross-asset feature we build:**
- Align timestamps to *decision time*; the rate-future / ES / DXY value must be **strictly ≤ t**, with realistic feed latency (don't use a quote stamped 200ms in your future).
- Build the fair-value regression with **expanding/rolling causal** windows only; never full-sample betas.
- Net of **spread + commission** in any reported "accuracy→PnL" claim (Huth-Abergel's warning).
- Report accuracy **per regime and per horizon**; a single all-sample number hides where the (small) edge lives.

---

## Data sources needed

| Data | Use | Where | Free/Paid | Fidelity |
|---|---|---|---|---|
| **CME FX futures (6E/6B/6J/6C/6A) tick / top-of-book** | futures-lead-spot; sharpen fast model | CME DataMine / Databento / firstratedata | Paid (Databento pay-as-you-go is cheapest) | tick / MBP-1, ms |
| **Treasury futures (ZT/ZF/ZN/ZB) tick** | US rate-diff leg | CME DataMine / Databento | Paid | tick |
| **SOFR / Fed Funds futures** | short-rate expectations leg | CME / Databento | Paid | tick |
| **Eurex Bund/Bobl/Schatz futures tick** | German rate leg (the other half of US-DE diff) | Eurex / Databento | Paid | tick |
| **ES / NQ E-mini tick** | risk factor / continuous-vs-jump | CME / Databento | Paid | tick |
| **VIX & VX futures (intraday)** | risk regime gate | Cboe DataShop / Databento | Paid (Cboe), some delayed free | 1-min+ |
| **DXY / ICE Dollar Index intraday** | dollar factor (or build from your 7 pairs — already have) | ICE; or synthesize | Free (synthesize) | 10s (yours) |
| **Constant-maturity yields (2y/10y US & DE)** daily/intraday | slow rate-diff level | FRED (US, free), Bundesbank/ECB SDW (DE, free); intraday via futures-implied | Free (daily) / Paid (intraday) | daily free, intraday via futures |
| **Economic calendar w/ actual vs consensus + timestamps** | release-window regime, surprise sign | you have a proxy; upgrade with Econoday/Haver/forexfactory timestamps | mixed | event |
| **Spot FX cross-check feed (EBS/LMAX)** | latency calibration vs your feed | CME EBS / LMAX | Paid | 5-100 ms |

Cheapest credible path: **Databento** for CME (6E, ZN/ZF, ES) and Eurex (Bund) futures tick + **FRED/ECB** for free daily yields to prototype the rate-differential idea *before* paying for intraday rate data. The dollar-basket and triangular residuals you can build *for free* from your existing 7-pair tick data today.

---

## Relevance & priority for OUR project

Ranked, with explicit interaction with what we've already ruled out.

**HIGH**
1. **Rate-differential fair-value-gap feature (A) + release-window model (B).** This is the single most defensible *new* orthogonal signal: grounded in the Verdelhan dollar-factor + 2y-diff mechanism, distinct from our event-time vol proxy (it carries *signed* rate moves, not just "release happened"), and aimed at the *continuous under-reaction* of spot to rate moves. Not in V1-V17. Prototype with free daily yields first, then Databento intraday futures if signal appears.
2. **Continuous-vs-jump decomposition (C) as a preprocessing step for any cross-asset model.** Cheap, principled, likely to *raise* measured accuracy on the predictable sub-population and stop jumps from poisoning training.

**MEDIUM**
3. **DeltaLag-style learned dynamic lead-lag net (D)** over a cross-asset panel — most promising *architecture* upgrade to our existing fixed cross-pair features, but wide panel = overfit risk; gate behind a strict OOS/regime evaluation.
4. **Risk-regime gating (G)** — wraps everything; low cost, formalizes "which peer leads when," consistent with our 0.632@0.2%-coverage selective-prediction result.
5. **FX-futures tape ingestion (E)** — improves the *fast* (3s) model and fill realism, not the 15m number; do it for execution quality, not as 15m alpha.

**LOW**
6. **Triangular residual (F)** — only as a 1-5s fast feature; Fenn et al. says no 15m alpha.
7. **Oil→CAD / gold→AUD commodity leads** — ambiguous intraday causality; near-irrelevant for EURUSD; only if we extend targets to USDCAD/AUDUSD, and even then expect weak.

**Hard truth / interaction with our ceiling:** The most rigorous FX-specific OOS evidence (Petrova et al.) and our own results agree that **a static 75%-always model at 15m is unlikely from cross-asset signals alone.** The realistic upside is (a) a few points of OOS accuracy on the *continuous* component, concentrated in (b) *release windows / high-uncertainty regimes*, harvested via (c) *selective prediction*. That combination — rate-diff residual + jump-aware + regime-gated + bet-only-when-confident — is the coherent, evidence-backed program here, and it is genuinely orthogonal to V1-V17.

---

## Sources (annotated)

1. **Petrova, Vilhelmsson & Norden (2025), "Assessing cross-currency predictability in forex markets: Insights from limit order book data,"** Int. J. Forecasting / SSRN 5523878 — https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5523878 / https://www.sciencedirect.com/science/article/pii/S0169207025001281 — *Most relevant: strict OOS + costs, 5 pairs, 1m-1h; finds near-zero predictability except AUD/USD@1h. Bounds our expectations and validates our ceiling.*
2. **Huth & Abergel (2014), "High-frequency lead/lag relationships — Empirical facts,"** J. Empirical Finance / arXiv 1111.7103 — https://arxiv.org/abs/1111.7103 — *~60% lagger-next-move accuracy from leader; futures→stock asymmetry; effect peaks at releases/US open; spread kills naive PnL. Confirms our fast-edge/decay picture.*
3. **Rosenberg & Traub, NY Fed Staff Report 262, "Price Discovery in the Foreign Currency Futures and Spot Market"** — https://www.newyorkfed.org/medialibrary/media/research/staff_reports/sr262.pdf — *Hasbrouck info-share: FX futures lead spot in price discovery. Execution/fidelity insight, not 15m alpha.*
4. **Aleti, Bollerslev & Siggaard (2025), "Intraday Market Return Predictability Culled from the Factor Zoo,"** Management Science 71(9):7731-7751 — https://public.econ.duke.edu/~boller/Papers/MS_2025.pdf — *Lagged cross-sectional factors predict intraday index OOS after costs; edge concentrated in high-uncertainty regimes; continuous-vs-jump split. Transferable method + regime lesson.*
5. **Verdelhan (2018), "The Share of Systematic Variation in Bilateral Exchange Rates"** (J. Finance) — https://www.ecb.europa.eu/events/pdf/conferences/130627/2.1a_A.Verdelhan_Paper.pdf — *Dollar + carry factors explain 18-83% of monthly bilateral USD FX; dollar factor dominates. Explains our ~97%-USD-factor EURUSD and where alpha cannot come from.*
6. **Lustig, Roussanov & Verdelhan (2011), "Common Risk Factors in Currency Markets,"** RFS / NBER 14082 — https://www.nber.org/system/files/working_papers/w14082/w14082.pdf — *Foundational dollar/carry factor structure underpinning the regime-gating logic.*
7. **Brunnermeier, Nagel & Pedersen (2008), "Carry Trades and Currency Crashes,"** NBER Macro Annual — https://www.fmg.ac.uk/sites/default/files/2020-08/M-Brunnermeier.pdf — *VIX↑ → carry unwind, risk-reversals reprice; safe-haven vs risk-FX asymmetry. Basis for risk-regime conditioning.*
8. **Charis, Tedman & Lundblad (2024), KC Fed RWP 24-12, "Risk-On/Risk-Off,"** — https://www.kansascityfed.org/documents/10594/rwp24-12charistedmanlundblad.pdf — *VIX alone insufficient; multi-asset risk index needed. Informs how to build the risk-regime gate.*
9. **Zhou, Wang, Cucuringu, Zhang et al. (2025), "DeltaLag: Learning Dynamic Lead-Lag Patterns,"** ACM ICAIF / arXiv 2511.00390 — https://arxiv.org/abs/2511.00390 — *Sparsified cross-attention learns time-varying per-pair lags; transplantable architecture for our cross-asset panel.*
10. **Fenn, Howison, McDonald, Williams & Johnson (2009), "The Mirage of Triangular Arbitrage in the Spot FX Market,"** arXiv 0812.0913 — https://arxiv.org/abs/0812.0913 — *Triangular mispricings vanish at executable size; saves us from a dead end. Residual usable only as a fast staleness hint.*
11. **Wavelet-based UHF analysis of triangular currency arbitrage (2018/2019),** Economic Modelling S0264999318319072 — https://www.sciencedirect.com/science/article/abs/pii/S0264999318319072 — *One triangle leg adjusts with a lag; the residual indicates which pair is stale (1-5s feature).*
12. **Causal flows between oil and forex (good/bad volatility), 2019,** Energy Economics S0140988319303020 — https://www.sciencedirect.com/science/article/abs/pii/S0140988319303020 — *Intraday HF wavelet-Granger: causality often currency→oil at intraday horizons; weakens oil→CAD lead thesis.*
13. **"What triggers intraday price jumps and co-jumps in gold?" (2025),** Int. Rev. Fin. Analysis S1057521925004673 — https://www.sciencedirect.com/science/article/abs/pii/S1057521925004673 — *Gold jumps rare (0.43%); 34% driven by US macro news, FOMC dominant → gold & AUD share a common driver rather than clean lead.*
14. **"Stylized Facts and Market Microstructure: German Bond Futures Market" (2024),** arXiv 2401.10722 — https://arxiv.org/html/2401.10722v1 — *Confirms Schatz/Bobl/Bund/Buxl are tick-level, deeply liquid — viable real-time rate-differential leg.*
15. **MacroMicro: US-Germany 10Y spread vs EUR/USD** — https://en.macromicro.me/collections/2204/euro-dollar/17798/euro-us-germany-10y-treasury-note-spread — *Practitioner confirmation of the rate-diff↔EURUSD relationship; visual sanity check.*
16. **FXEmpire, "Treasury Yield Surge Drives USDJPY, EURUSD, GBPUSD"** — https://www.fxempire.com/forecasts/article/interest-rates-forecast-treasury-yield-surge-drives-usdjpy-eurusd-and-gbpusd-1599798 — *Practitioner mapping of ~1% rate-diff ≈ 10-15 big figures; the 2y-as-primary-maturity claim.*
17. **Deriaz et al. (2024), "Predicting Foreign Exchange EUR/USD Direction Using Machine Learning,"** arXiv 2409.04471 / ICMLMI — https://arxiv.org/abs/2409.04471 — *Clean peer-reviewed EURUSD ML: 58.52% 1-day; PCA/meta 54.98-57.23%. Reality benchmark exposing 75-90% hype as leakage.*
18. **CME EBS Data & Analytics (latency facts)** — https://www.cmegroup.com/markets/ebs/ebs-data-and-analytics.html — *EBS 5ms delay / 100ms slices; calibrates feasibility of futures-vs-spot latency arbitrage / fidelity.*
