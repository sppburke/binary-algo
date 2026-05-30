# News & Social-Media Sentiment / NLP for 15-Minute FX Direction

Research vector 08. Target: lift EURUSD (and USD-major) **15-minute** directional
accuracy toward the open 75% out-of-sample target using **news / text / sentiment**
signals that are *orthogonal* to the 239 TA features, cross-pair lead-lag, and decayed
microstructure already explored. Skeptical lens throughout: most published "sentiment predicts FX"
results are at **daily-to-weekly** horizons, not 15-minute, and many headline accuracy
numbers leak.

---

## TL;DR (most actionable for our 15m goal)

- **The single highest-value, most-credible idea in this whole vector is the
  scheduled-macro-announcement window, not generic sentiment.** The price-discovery
  literature (Andersen-Bollerslev-Diebold-Vega; ECB working papers) shows FX reacts to a
  macro *surprise* with the bulk of repricing in the **first 1 minute**, but **elevated
  volatility persisting ~15 minutes** and a **measurable directional drift starting ~30
  min BEFORE** the release. That 30-min-pre / 15-min-post window is exactly our horizon,
  and the sign is predictable from the *surprise* (actual − consensus), which is a hard
  number, not a fuzzy NLP score. **This is the one place where >55-60% directional hit
  rates are plausibly real and reproducible** — but only on the small subset of bars near
  Tier-1 releases (NFP, CPI, FOMC/ECB, PMIs).

- **Generic news/social sentiment predicts FX at DAILY-to-WEEKLY horizons, not 15-min.**
  RavenPack's own FX research reports Information Ratios of ~0.5-0.8 at **4-5 day**
  effective holding periods (macro-news trend-following IR ~0.52-0.80; FX-news
  mean-reversion IR ~0.63-0.79). Nothing in their published material claims a 15-minute
  edge. Treat any 15m sentiment-alpha claim as guilty until proven.

- **The "speed of pricing" challenge is steep at 15m.** Machine-readable news (Bloomberg,
  Reuters, AlphaFlash, RavenPack) is consumed by co-located HFTs in <100µs; the headline
  *direction* of a scheduled release is priced within seconds. So **fast,
  unambiguous, machine-readable news is not yet an edge for us** — beating that latency
  remains open. Our edge, if any, is in **slower-to-interpret text** (nuanced central-bank tone,
  multi-headline narrative, surprise *relative to a richer expectation model*) that takes
  minutes-to-hours to fully price.

- **Central-bank communication tone (Fed/ECB) is the most credible "slow text" signal.**
  Picault & Renault (2017) custom ECB lexicons, and the RBA's "Ornithologist" LLM
  hawkish/dovish reasoning framework (2025), show communication tone shifts *ahead* of
  policy action and moves rates/FX. But these events are **rare** (8 ECB + 8 FOMC
  meetings/yr + speeches) — high per-event signal, near-zero coverage of the 35,000
  15-min bars/year. Useful as a **regime/event overlay**, not a continuous feature.

- **Economic-surprise indices (Citi CESI) are a daily/weekly *level* signal, already
  redundant with our ruled-out calendar work** at 15m. The *gradient* of CESI predicts
  multi-day USD drift, not 15-minute bars. Low priority as-is; the *component* surprises
  (per-release) feeding the announcement-window idea above are what matter.

- **LLM "forecast EURUSD" papers are mostly regression (MAE/RMSE) at daily granularity and
  do NOT report honest OOS *direction* accuracy.** The flagship EUR/USD LLM+news paper
  (Ding et al. 2024, arXiv 2408.13214) reports only a 9-11% MAE/RMSE reduction vs baseline
  — a low-information-content win for a near-random-walk series, and no 15m, no direction
  hit-rate. ChatGPT-headline equity results (Lopez-Lira & Tang) are real but **daily**,
  equities, and partly an artifact of headline-timing. Do not chase "LLM predicts FX."

- **Net recommendation:** Treat this vector as **event-conditional, not continuous.**
  Build (1) a real machine-readable *economic-surprise* feature keyed to release
  timestamps (pre-30m drift + post-15m volatility-sign), and (2) a sparse central-bank
  tone overlay. Expect them to lift accuracy **only on the ~2-5% of bars near events**,
  possibly to 60-70% there, while contributing ~0 on the other 95-98% of bars. That is a
  *selective-prediction* story (which we've already explored) — sentiment's realistic role
  is to **improve the conditioning of when to bet**; raising unconditional 15m AUC stays open.

---

## Key findings (with citations)

### 1. Macro-announcement price discovery: the credible 15m mechanism

The canonical reference is **Andersen, Bollerslev, Diebold & Vega (2003), "Micro Effects
of Macro Announcements: Real-Time Price Discovery in Foreign Exchange," *American Economic
Review* 93(1)** ([AEA](https://www.aeaweb.org/articles?id=10.1257/000282803321455151)).
They show announcement *surprises* (actual minus median forecast, standardized) produce
**conditional mean jumps** in high-frequency FX returns, with sign and size mapping
cleanly to the surprise. Bad news has bigger impact than good (asymmetry), and the
response is concentrated immediately after release.

The timing detail that matters for us comes from the broader high-frequency literature
(summarized across the CFTC "Macro News Announcements and Automated Trading" working paper
[Haynes & Roberts](https://www.cftc.gov/sites/default/files/idc/groups/public/@economicanalysis/documents/file/oce_macroannouncement.pdf)
and ECB WP 1901 "Price drift before U.S. macroeconomic news"
[ECB](https://www.ecb.europa.eu/pub/pdf/scpwps/ecbwp1901.en.pdf)):

- **Bulk of permanent price change occurs in the first ~minute** after a Tier-1 release.
- **Volatility stays substantially elevated for ~15 minutes**, slightly elevated for hours.
- **Pre-announcement drift:** prices begin moving in the "correct" direction **~30 minutes
  before** scheduled releases; this pre-drift accounts on average for **about half** of the
  total adjustment (ECB WP 1901). Attributed to information leakage + superior forecasting.

Implication: the *direction* of the move in the 15-min window straddling a Tier-1 release
is **partially predictable from the surprise sign**, and there is a *pre*-release drift
whose sign also correlates with the eventual surprise. This is the one mechanism in this
entire vector that operates natively at our horizon and is grounded in peer-reviewed
high-frequency evidence.

### 2. Pre-FOMC drift (and its decay)

**Lucca & Moench (2015), "The Pre-FOMC Announcement Drift," *Journal of Finance*** (NY Fed
Staff Report 512, [SR512](https://www.newyorkfed.org/research/staff_reports/sr512.html)):
large excess *equity* returns in the 24h before scheduled FOMC announcements — >80% of the
equity premium over their sample. Crucially for skepticism: **the effect appears in
international equity indices but NOT in US Treasuries or money-market futures**, and a
follow-up ("The disappearing pre-FOMC announcement drift," 2020,
[PubMed](https://pubmed.ncbi.nlm.nih.gov/33013237/)) shows it **largely vanished after
2015**. Lesson: even strong, famous event-window anomalies **decay and get arbitraged** —
any FOMC/ECB window feature must be validated on our 2024-25 test and 2026 holdout
separately, not assumed stationary from a 2012-2014 backtest.

### 3. News/social sentiment predicts FX — at daily-to-weekly horizons

**RavenPack FX research** (the most credible vendor evidence because it's their own data
and they're incentivized to show *some* edge, so treat as an upper bound):

- "Harnessing News Sentiment for FX Futures Strategies" (Hafez et al., 2023; SSRN
  [5560198](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5560198);
  [RavenPack](https://www.ravenpack.com/research/harnessing-news-sentiment-for-fx-futures-strategies)):
  macro-news → **trend-following** long-short G7 FX-futures, IR ~0.52, **5-day** effective
  holding; FX-news → **mean-reversion**, IR ~0.63, **4-day** holding. Daily rebalanced
  variants reach IR ~0.80/0.79 at 5d/4d.
- "Enhancing Currency Risk Premia with RavenPack FX Sentiment Factors" (2024; SSRN
  [5560201](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5560201)): combined
  composite IR ~0.62 before costs; improves Carry/Value/Momentum.

**Every credible number here is multi-day.** No RavenPack publication claims a 15-minute
directional edge. The mechanism (overreaction → 4-day mean reversion) is *intrinsically*
slower than our horizon.

### 4. Central-bank communication NLP

- **Picault & Renault (2017), "Words are not all created equal: A new measure of ECB
  communication," *Journal of International Money and Finance*** — custom field-specific
  ECB lexicons (monetary-policy vs economic-condition tone). Communication tone **shifts
  ahead of policy actions** and significantly affects rates and bank stocks. Foundational
  for credible CB-tone scoring (generic FinBERT mis-scores ECB jargon).
- **"Ornithologist: Towards Trustworthy 'Reasoning' about Central Bank Communications"
  (Jones, RBA Working Paper, 2025; arXiv
  [2505.09083](https://arxiv.org/abs/2505.09083))** — an LLM "reasoning" framework that
  scores hawkish/dovish stance with traceable justifications (addresses the black-box /
  hallucination problem of naive LLM scoring). State-of-the-art *measurement* tool; does
  not itself claim FX trading alpha.
- **Renault, Picault & Gillet, "Investor Attention and Intraday Market Reaction to ECB
  Announcements" (SSRN [4415462](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4415462))**
  — explicitly *intraday* ECB-event reaction, and finds investor attention conditions the
  reaction. Most relevant CB paper to our horizon, but events are sparse.

Caveat: FinBERT was trained on US/English financial text and **mis-scores ECB-specific
language** (multiple sources flagged this corpus mismatch). Use a CB-specific lexicon or a
CB-tuned LLM, not vanilla FinBERT, for central-bank text.

### 5. Twitter / social microblogging for intraday FX

- **Papaioannou et al. (2013), "Can social microblogging be used to forecast intraday
  exchange rates?" (arXiv [1310.5306](https://arxiv.org/pdf/1310.5306))** — uses Twitter
  to model intradaily EUR/USD; reports that microblog info *can* enhance very-short-term
  forecasting. **But:** the sample is short, the gains are marginal, and 2013-era Twitter
  FX chatter has little to do with 2024-26 X. Directionally interesting, not reproducible
  as-is.
- General social-sentiment-intraday studies (Greyling & Rossouw 2022 and the broad
  StockTwits/Twitter literature) cluster at **just above 50%** accuracy and are dominated
  by **equities and crypto**, where retail social flow actually moves price. **FX majors
  are ~97% USD-macro-factor and far less retail-driven**, so social sentiment is weaker for
  EURUSD than for memestocks/crypto. Low priority.

### 6. LLM-as-forecaster: mostly hype at our horizon

- **Ding et al. (2024), "EUR-USD Exchange Rate Forecasting Based on Information Fusion with
  LLMs and Deep Learning" (arXiv [2408.13214](https://arxiv.org/abs/2408.13214))** — IUS
  framework: LLM sentiment polarity + Optuna-Bi-LSTM. Reports **MAE −10.69%, RMSE −9.56%**
  vs best baseline. This is a **regression** metric on a near-random-walk series; a small
  MAE reduction is near-meaningless for *direction*, no 15m horizon, no honest OOS hit
  rate. **Not evidence of a 15m directional edge.**
- **Lopez-Lira & Tang (2023), "Can ChatGPT Forecast Stock Price Movements?"** (arXiv
  2304.07619) — real result, but **daily, US equities, headline-driven**, and the
  "~90% portfolio-day hit rate" figure is widely misread (it's a long-short portfolio sign
  consistency, not per-name direction accuracy). Does not transfer to 15m FX.
- **Gruver et al. (2023), "LLMs Are Zero-Shot Time Series Forecasters" (LLMTime, arXiv
  2310.07820)** — LLMs can extrapolate numeric series zero-shot, but the "exchange_rate"
  benchmark there is daily and the wins are on *level* forecasting, not intraday direction.

### 7. The latency challenge (why "fast news" is not yet our edge)

Machine-readable feeds (Bloomberg B-PIPE, Reuters/Refinitiv, **AlphaFlash**) deliver
structured releases in **milliseconds**; co-located HFTs round-trip in **<100µs**
([AlphaFlash](https://alphaflash.com/)). The literature on "Speed, algorithmic trading,
and market quality around macroeconomic news" (Scholtus, van Dijk, Frijns) confirms the
adjustment window has compressed from **minutes (1990s) → seconds (2000s) → milliseconds
(2010s+)**. **Conclusion:** any signal that is *unambiguous and machine-readable* (the raw
surprise number, a clear hawkish headline) is priced within seconds, before we can act at 15m. The only
text-derived edge plausibly *available* to us at 15m is **interpretation that is genuinely
slow to converge** — nuanced CB tone, narrative aggregation, surprise relative to a
*better-than-consensus* nowcast — i.e., where our model knows something the median fast
reaction missed.

---

## Concrete techniques / features / architectures to try

Ordered by expected value for our specific 15m goal.

### A. Economic-surprise event features (HIGHEST priority, most credible)
Build a per-release **standardized surprise** feature, keyed to exact release timestamps:
1. Get a release calendar with **consensus + actual** for Tier-1 US/EZ items: NFP &
   unemployment, CPI/PCE, FOMC decision + statement, ECB decision, ISM/PMI, GDP, retail
   sales. Standardize each: `z = (actual − consensus) / rolling_std(surprise)`.
2. **Pre-event drift feature:** in the 30 min *before* a scheduled release, add (a)
   minutes-to-release, (b) a sign prior from the *trend of the surprise series* / recent
   CESI gradient, (c) realized order-flow tilt in the pre-window. Test whether the 15m bar
   straddling/leading the release is predictable.
3. **Post-event sign feature:** for the 15m after release, feature = `sign(z) ×
   |z|`-bucketed, interacted with EUR/USD's historical beta to that release. The first-bar
   move is mostly gone, but bars 2-3 (5-15 min out) carry the *continuation/reversal*
   residual that the ABDV asymmetry literature documents.
4. **Critical hygiene:** the *actual* value's timestamp must be the true wire time, and the
   feature must be NaN/zero until the release minute — any earlier and you've leaked. Train
   a separate event-conditioned model; do not dilute the all-bars model.

### B. Central-bank tone overlay (MED priority, sparse but high per-event)
1. Ingest FOMC statements, ECB monetary-policy statements + press-conference transcripts,
   and speeches. Score with a **CB-specific** method (Picault-Renault lexicon, or an LLM
   with the Ornithologist-style reasoning prompt), NOT vanilla FinBERT.
2. Feature = **change in hawkish/dovish score vs the previous same-type document**
   (surprise-in-tone), plus a **statement-vs-press-conference divergence** feature (the
   press conference often walks back or amplifies the statement → second 15m leg).
3. Build a **rate-differential expectation delta**: map tone change → implied policy-path
   change → expected USD direction. Use only as an event-window overlay (the ~16
   meetings/yr + scheduled speeches), with explicit decay-after-2015 caution (finding #2).

### C. "Slow-interpretation" news narrative features (MED/Low, experimental)
The only generic-news angle that respects the latency reality:
1. Aggregate **headline flow intensity + net tone** over rolling 15-60 min windows per
   currency (count, novelty/dedup, source weighting). Hypothesis: *clusters* of
   reinforcing headlines (a developing narrative) price in over minutes-to-hours, unlike a
   single scheduled number that prices in seconds.
2. **Novelty / staleness filter** (à la RavenPack's "event novelty score"): only the
   *first* mention of a story has predictive content; downstream repeats are noise. Feature
   = tone-weighted *novel*-headline flow only.
3. Cross-asset confirmation: require the news-tone tilt to agree with a contemporaneous
   rates/equity move before trusting it (reduces false NLP signals).

### D. Surprise-relative-to-nowcast (Low/experimental, the real long-shot edge)
The latency reality says "consensus surprise is priced within seconds." The escape hatch: build your
*own* nowcast of the release (from higher-frequency data, alt-data, or an LLM digesting
pre-release commentary) and trade the **gap between market-implied expectation and your
nowcast**. This is the construction with the clearest *theoretical* path to beating the fast
reaction — but it's the hardest and most leakage-prone. Defer until A/B are validated.

### Architecture notes
- **Two-headed model:** (1) an always-on TA/microstructure model (existing), (2) an
  event-conditioned model that only activates in release/CB windows, gated by a "near-event"
  flag. Combine via the selective-prediction layer we already have — sentiment's realistic
  contribution is to **raise confidence/coverage specifically near events**.
- Do **not** feed sparse event features into the all-bars LightGBM ensemble — they'll be
  swamped and the model will learn nothing (this is likely *why* our prior "calendar proxy"
  added ~0: a release-window vol-seasonality proxy carries no *direction*, only timing).
  The new ingredient vs our exhausted calendar work is the **signed surprise**, not the
  timing.

---

## Reported results & CREDIBILITY assessment

| Claim / source | Horizon | Reported result | Credibility | Leakage / caveat |
|---|---|---|---|---|
| ABDV (2003) macro surprises → FX jumps | seconds-minutes | Significant signed mean jumps, bad>good asymmetry | **High** (top journal, HF data) | Effect is in first ~1 min; we must capture the *residual* 5-15m, smaller |
| Pre-announcement drift (ECB WP1901) | ~30 min pre | ~half of total move occurs pre-release | **High** | Partly leakage/insider; sign tied to surprise we can't see early |
| Lucca-Moench pre-FOMC drift | 24h pre-FOMC | >80% of equity premium | **High but DECAYED** | Vanished post-2015; equities not Treasuries; FX weaker |
| RavenPack FX sentiment IR ~0.5-0.8 | **4-5 days** | trend IR 0.52-0.80, MR IR 0.63-0.79 | **Med-High** (vendor, but own data) | Multi-day only; net-of-cost lower; no 15m claim |
| Picault-Renault ECB lexicon | event/daily | tone → rates, bank stocks; leads policy | **High** (peer-reviewed) | Sparse events; FX effect indirect |
| Ornithologist CB LLM (RBA 2025) | measurement | trustworthy hawk/dove scoring | **High as a tool** | Not a trading-alpha claim |
| Twitter intraday EURUSD (2013) | intraday | marginal forecasting gain | **Low** | Tiny sample, stale platform, marginal |
| Ding et al. EUR/USD LLM+news (2024) | daily | MAE −10.7%, RMSE −9.6% | **Low for our goal** | Regression not direction; no 15m; near-RW series |
| ChatGPT forecasts stocks (Lopez-Lira) | daily | "~90% hit" portfolio sign | **Med but misread** | Equities; portfolio-level not per-name; headline timing |
| Generic social-sentiment intraday | intraday | ~50-53% accuracy | **Low** | Equities/crypto-centric; FX is macro-driven |

**General leakage red flags to watch in any sentiment-FX backtest** (all present in the
weaker papers above): (1) random train/test split instead of temporal → look-ahead;
(2) news timestamp = *article publication* but feature aligned to *event* time → future
leak; (3) sentiment scored on revised/aggregated daily text but applied intraday;
(4) survivorship in which currencies/events are included; (5) reporting MAE/RMSE on a
near-random-walk and calling small reductions "predictive."

---

## Data sources needed

| Data | Where | Free / Paid | Fidelity for 15m FX |
|---|---|---|---|
| **Economic release calendar + consensus + actual + timestamps** | Trading Economics API, Econoday, Bloomberg, **Refinitiv/LSEG Economic Indicators**, Haver, **AlphaFlash** (ms-stamped) | Mostly paid; some free (TE limited, investpy-style scrapes, ForexFactory calendar) | **Critical.** Need *true wire timestamps* and *consensus* to compute surprise. Free calendars often lack exact times/consensus → leakage risk |
| **Central-bank text** (FOMC statements, ECB statements + presser transcripts, speeches) | Fed & ECB websites (free), BIS central-bankers'-speeches corpus (free) | **Free** | High; sparse events. Need precise release timestamps |
| **Machine-readable news + sentiment** | RavenPack/Bigdata.com, Bloomberg (NLP/B-PIPE), Refinitiv News Analytics (TRNA), Dow Jones DNA | Paid (expensive) | High but priced within seconds; useful for *novelty/flow*; beating fast reaction still open |
| **FinBERT / CB-tuned LLM** | HuggingFace (ProsusAI/finbert), Ornithologist prompt (RBA paper), Picault-Renault lexicon (published) | Free / your own LLM | Use CB-specific, not vanilla FinBERT, for CB text |
| **Citi CESI** | Bloomberg (CESIUSD Index), MacroMicro, Refinitiv | Paid (Bloomberg) | Daily level/gradient only; low 15m value |
| **Twitter/X & Reddit** | X API (now costly), Pushshift/Reddit | Paid/limited | Low for FX majors; better for crypto |

**Cheapest credible starting stack for us:** free CB text + a free/cheap economic calendar
with consensus (ForexFactory/Econoday-style) + our own FX ticks. That alone lets us build
the **announcement-surprise event feature (idea A)** and the **CB-tone overlay (idea B)**
with no expensive vendor feed. Validate before paying for RavenPack/Bloomberg.

---

## Relevance & priority for OUR project

Context interaction with what we've already ruled out:
- We already found the **economic-calendar event-timing proxy redundant with time-of-day**
  (V17). That proxy carried **timing/volatility seasonality but no *direction*.** The
  **new and untried ingredient is the signed standardized surprise** `(actual−consensus)`,
  which is the actual directional payload. This is *not* the same as our ruled-out work.
- Sentiment's realistic role aligns with our **selective-prediction** finding (best 15m =
  0.632 @ 0.2% coverage): sentiment/event features should **improve *which* bars we bet on
  and *how confident*** near events, as one route toward unconditional AUC. Frame near-term success as
  "60-70% on the 2-5% of bars near Tier-1 events," with "75% on all bars" still the open target.
- Microstructure decays in minutes (our finding); **scheduled-event drift is the rare
  exception that lives at 15m** — it's an *exogenous information shock*, not endogenous
  flow, so it doesn't suffer the same decay.

**Ranked ideas:**
- **HIGH — Signed economic-surprise event feature (idea A).** Most credible, cheap data,
  native 15m horizon, directly addresses the "no direction" gap in our prior calendar work.
  Build the two-headed event-conditioned model.
- **HIGH (validation discipline) — Pre/post window timing from ABDV + ECB WP1901.** Use
  their documented 30-min-pre / 15-min-post structure to define feature windows; re-test
  stationarity on 2024-26 (decay risk per Lucca-Moench).
- **MED — Central-bank tone overlay (idea B).** High per-event signal, free data, but
  sparse; CB-specific scoring required; post-2015 decay caution.
- **MED/LOW — Novelty-filtered news-flow narrative (idea C).** Only generic-news angle that
  respects the latency reality; needs a (paid) machine-readable feed to do well; experimental.
- **LOW — Citi CESI level/gradient.** Daily, redundant with ruled-out context at 15m.
- **LOW — Twitter/Reddit/StockTwits sentiment.** FX majors too macro-driven; better for
  crypto/equities.
- **LOW — "LLM forecasts EURUSD" end-to-end.** Hype at our horizon; regression metrics, no
  honest 15m direction OOS. Use LLMs only as *scoring tools* inside ideas A/B, not as the
  forecaster.

---

## Sources (annotated)

1. **Andersen, Bollerslev, Diebold, Vega (2003), "Micro Effects of Macro Announcements,"
   AER 93(1)** — [aeaweb.org](https://www.aeaweb.org/articles?id=10.1257/000282803321455151)
   — Canonical proof that signed macro *surprises* drive HF FX returns; basis for our event feature.
2. **ECB Working Paper 1901, "Price drift before U.S. macroeconomic news"** —
   [ecb.europa.eu](https://www.ecb.europa.eu/pub/pdf/scpwps/ecbwp1901.en.pdf) — Documents
   ~30-min pre-announcement drift = ~half the total move; defines our pre-event window.
3. **Haynes & Roberts (CFTC), "Macro News Announcements and Automated Trading"** —
   [cftc.gov](https://www.cftc.gov/sites/default/files/idc/groups/public/@economicanalysis/documents/file/oce_macroannouncement.pdf)
   — Timing of adjustment (first minute dominant, ~15-min elevated vol); the latency reality.
4. **Lucca & Moench (2015), "The Pre-FOMC Announcement Drift," J. Finance / NY Fed SR512** —
   [newyorkfed.org](https://www.newyorkfed.org/research/staff_reports/sr512.html) — Famous
   pre-event drift; cautionary because it largely **decayed after 2015**.
5. **"The disappearing pre-FOMC announcement drift" (2020)** —
   [PubMed](https://pubmed.ncbi.nlm.nih.gov/33013237/) — Direct evidence event anomalies get
   arbitraged away; mandates OOS re-validation on 2024-26.
6. **RavenPack / Hafez et al. (2023), "Harnessing News Sentiment for FX Futures Strategies,"
   SSRN 5560198** —
   [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5560198) /
   [RavenPack](https://www.ravenpack.com/research/harnessing-news-sentiment-for-fx-futures-strategies)
   — Best vendor evidence; **4-5 day** holding, IR 0.52-0.80; shows sentiment FX edge is multi-day.
7. **RavenPack, "Enhancing Currency Risk Premia with FX Sentiment Factors," SSRN 5560201** —
   [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5560201) — Composite IR ~0.62;
   improves Carry/Value/Momentum; confirms daily-weekly horizon.
8. **Picault & Renault (2017), "Words are not all created equal," JIMF** — custom ECB
   communication lexicon; tone leads policy and moves markets; why generic FinBERT fails on ECB.
9. **Jones (2025), "Ornithologist: Trustworthy Reasoning about Central Bank Communications,"
   RBA WP, arXiv 2505.09083** — [arXiv](https://arxiv.org/abs/2505.09083) — SOTA LLM
   hawk/dove scoring with traceable reasoning; the tool to use for CB tone (idea B).
10. **Renault, Picault, Gillet, "Investor Attention and Intraday Market Reaction to ECB
    Announcements," SSRN 4415462** —
    [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4415462) — Explicitly intraday
    CB-event reaction; closest CB paper to our horizon.
11. **Papaioannou et al. (2013), "Can social microblogging forecast intraday exchange rates?"
    arXiv 1310.5306** — [arXiv](https://arxiv.org/pdf/1310.5306) — Early intraday Twitter-EURUSD;
    marginal, stale, low priority — illustrative of social-sentiment's weak FX edge.
12. **Ding et al. (2024), "EUR-USD Forecasting via Information Fusion with LLMs," arXiv 2408.13214**
    — [arXiv](https://arxiv.org/abs/2408.13214) — LLM+news+Bi-LSTM; MAE −10.7% only, **no 15m, no
    direction hit-rate** — example of hype to avoid.
13. **Lopez-Lira & Tang (2023), "Can ChatGPT Forecast Stock Price Movements?" arXiv 2304.07619**
    — [arXiv](https://arxiv.org/html/2304.07619.pdf) — Real but **daily, equities, portfolio-sign**;
    widely misread "90%"; does not transfer to 15m FX.
14. **Gruver et al. (2023), "LLMs Are Zero-Shot Time Series Forecasters" (LLMTime), arXiv 2310.07820**
    — [arXiv](https://arxiv.org/pdf/2310.07820) — LLM numeric extrapolation; daily exchange-rate
    benchmark, level not intraday direction.
15. **AlphaFlash (Deutsche Börse) machine-readable macro feed** — [alphaflash.com](https://alphaflash.com/)
    — Ms/µs-stamped releases; embodies the latency reality — shows fast unambiguous news is not yet our edge.
16. **Scholtus, van Dijk, Frijns (2014), "Speed, algorithmic trading, and market quality around
    macroeconomic news announcements," J. Banking & Finance** —
    [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0378426613003841) —
    Adjustment compressed minutes→seconds→ms; quantifies why speed is not yet available to us.
17. **Citi Economic Surprise Index overview** — [FP Markets](https://www.fpmarkets.com/education/trading-guides/what-is-the-citigroup-economic-surprise-index/)
    — CESI is *direction/gradient* of aggregate surprises; daily-weekly USD signal, low 15m value.
