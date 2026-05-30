# Microstructure Features, Realized Volatility & Jump Signals for 15-Minute FX Direction

Research vector 05. Goal context: predict 15-minute EURUSD (and other USD majors) **direction**, >75% OOS,
from 10s OHLCV + raw sub-second ticks (bid/ask + quote sizes). We already explored 239 TA features,
cross-pair lead-lag, tick-rule OFI proxy (≈0 lift), tree/deep ensembles (~0.52 AUC), stat-arb, calendar
proxy, selective prediction (best 0.632 @ 0.2% coverage). Clean >75% so far found at the 3-second horizon;
raw imbalance decays to ~0.50 by ~1 min. This report covers what the realized-vol / jump / microstructure
literature actually says, with a hard separation between credible single-asset minutes-horizon evidence and
hype / cross-sectional / wrong-horizon results that do **not** transfer.

---

## TL;DR

- **Realized semivariance & signed jumps are overwhelmingly a VOLATILITY signal, not a direction signal.**
  Patton & Sheppard (2015) show the *sign* of the jump predicts *future volatility* (negative jumps → higher
  future vol), with **near-zero contribution to predicting the next return's sign**. Do not expect semivariance
  to give you direction directly. (Patton-Sheppard, RESTAT 2015; PDF read.)
- **The "semivariance/skewness predicts returns" results are almost all CROSS-SECTIONAL (sort stocks) and
  WEEKLY/MONTHLY**, not single-asset-15-minute. They are not portable to one EURUSD time series at 15m.
  (Amaya-Christoffersen-Jacobs-Vasquez 2015; Bollerslev-Li-Zhao JFQA 2020.)
- **The one credible single-asset, time-series, OOS directional use of semivariance** is Liu, Lu, Li & Wang
  (J. Empirical Finance 2023): semivariance *asymmetry* gates **time-series momentum vs reversal** in commodity
  futures (daily). This is a real, OOS-robust pattern — but it is a *conditioning/regime* signal layered on
  momentum, not a standalone 15m classifier, and it's daily not 15m.
- **FX-specific jump direction is hard.** Mäkinen et al. (2019) report jump-*direction* F-measure ≈ **49%
  (~0.50)** even with CNN-LSTM-Attention, vs ~72% for jump *occurrence*. Liu et al. (2019) confirm:
  predicting whether a jump arrives is much easier than its sign. This matches your own 3s→1min decay finding.
- **The genuinely orthogonal, credible FX signal you have NOT properly tried is true signed ORDER FLOW**
  (Evans-Lyons): real dealer/interbank order flow explains >50% of daily DM/USD and beats a random walk OOS.
  Your tick-rule proxy is a weak shadow of this; the lift opportunity is in *better-signed, depth/size-weighted*
  flow (microprice, multi-level OFI), not in more realized-vol features.
- **Microprice (Stoikov 2018) and multi-level OFI (Cont-Kukanov-Stoikov; Kolm-Turiel-Westray 2023) are the
  strongest microstructure direction signals — but their effective horizon is "≈2 price changes" (seconds).**
  They will not by themselves reach 15m. Realistic use: as a *better short-horizon nowcast* whose aggregate
  drift you bet on selectively, not a 15m point predictor.
- **Macro-news jump timing is the most credible minutes-horizon FX direction lever.** Lee-Wang ("Tales of Tails")
  show FOMC/NFP releases predictably *cause* currency jumps; combine release-window + pre-announcement flow.
  Your "calendar redundant with time-of-day" finding may have under-exploited *flow asymmetry in the
  pre-release window* (different from vol seasonality).
- **Credibility hazard:** most online "75–90% FX accuracy" microstructure claims are leakage (contemporaneous
  OFI used to "predict" the same-bar return), wrong-horizon (seconds dressed as minutes), or cross-sectional
  results misread as time-series. Use purged+embargoed CV (López de Prado) — a leak as small as one 15m bar of
  overlap inflates accuracy.

---

## Key findings (detailed, with inline citations)

### 1. Realized semivariance and signed jumps: a volatility signal, not a direction signal

Barndorff-Nielsen, Kinnebrock & Shephard (2010) decompose realized variance into **upside** and **downside
realized semivariance** by the sign of each high-frequency return:

```
RS+ = Σ r_i² · 1{r_i > 0}      RS- = Σ r_i² · 1{r_i < 0}      RV = RS+ + RS-
Signed jump variation:  ΔJ² = RS+ − RS-
```

In the limit, each semivariance → ½·integrated-variance + the same-signed jump variation, so **ΔJ² isolates the
net signed jump component** (the continuous part cancels). (Patton & Sheppard 2015, eqs. 6–8, PDF read.)

What Patton & Sheppard (REStat 2015, *"Good Volatility, Bad Volatility"*, PDF read directly) actually establish,
on S&P 100 constituents at ~5-min sampling, horizons 1 day–3 months:
- **Negative (downside) semivariance dominates future-volatility prediction;** positive semivariance is "small and
  often insignificant or negative." The asymmetric leverage effect lives in RS-.
- The interaction with the classic Glosten-Jagannathan-Runkle leverage dummy is small once RS- is included —
  RS- *is* the high-frequency leverage effect.
- Crucially for us: their dependent variable is **future realized volatility**, not the future return sign.
  The "good/bad" terminology is about vol persistence, not directional alpha. **There is no claim that ΔJ²
  predicts the next return's direction at the same asset.**

Implication for the 15m goal: building RS+, RS-, ΔJ², HAR-RV-J features will buy you a **better volatility/regime
estimate** (useful for sizing and for selective-prediction thresholds), but the literature gives **no current
evidence of a directional edge** from them on a single FX series. This is consistent with your current best AUC level.

### 2. Where semivariance/skewness *does* predict returns — and why it doesn't transfer to you

- **Amaya, Christoffersen, Jacobs & Vasquez (2015, JFE)**: realized **skewness** computed from 5-min intraday
  returns over the past *week* predicts the *cross-section* of next-week stock returns (lowest-minus-highest
  skew decile ≈ +19 bps/week, t≈3.7). This is (a) **cross-sectional** (a long-short over many stocks), and
  (b) **weekly**. Neither holds for a single EURUSD series at 15m.
- **Bollerslev, Li & Zhao (2020, JFQA)** and Zhao et al.: "good minus bad" volatility / signed jump variation
  prices the **cross-section** of stock returns. Again cross-sectional, weekly+.
- **Liu, Lu, Li & Wang (2023, J. Empirical Finance, "Time Series Momentum and Reversal: Intraday Information
  from Realized Semivariance")** — the *one* single-asset, time-series, OOS-validated result: the **asymmetry of
  RS+/RS-** signals whether **time-series momentum will continue or reverse** in Chinese commodity futures
  (daily). Their rule-based momentum strategy that uses semivariance asymmetry has a significantly higher OOS
  Sharpe than vanilla TSMOM, robust to lookback, vol-scaling, execution lag, costs. Mechanism: positive RS
  captures informed-contrarian behavior near momentum exhaustion. **This is a conditioning signal layered on an
  existing momentum signal, daily horizon — not a standalone 15m classifier.** It is the most promising
  *transferable* idea in this cluster, but as a regime gate, not a direct predictor.

### 3. Jump *direction* is not yet predictable; jump *occurrence* and *timing* are predictable

- **Lee & Mykland (2008, RFS)** give the standard nonparametric intraday jump test: standardize each
  high-frequency return by a local bipower-variation scale; |L_i| above a Gumbel threshold ⇒ jump. This is the
  workhorse detector. (Search + Lee 2011 PDF read.)
- **Liu et al. (2019, arXiv 1912.07165, "Predicting intraday jumps … liquidity measures and technical
  indicators")** — 5-min intervals, level-2 data, 1271 stocks, Random Forest best. Jump *occurrence* F-measure
  ≈ 63–72%. But **direction (up/down/no-jump) is explicitly much harder**; they cite **Mäkinen et al. (2019):
  average direction F-measure ≈ 49% (~0.50)** even with CNN-LSTM-Attention. (PDF read.)
- **Lee & Wang, "Tales of Tails: Jumps in Currency Markets" (JFM)** — directly FX. Currency jumps are
  predictably *triggered* by **scheduled US macro releases, especially FOMC**; jump intensity peaks *before*
  the order-flow peak; strong time-of-day and clustering structure. Predictors are national fundamentals +
  release timing. They build "jump-robust carry trades." **This is about jump arrival/size around news, and the
  directional content is tied to the news surprise sign — not free from a TA feature set.** (Abstract read.)

Net: your own finding (3s edge, 1-min decay, direction ≈ 0.50 at minutes) is *exactly* what the credible
literature finds. Jump features help you know *when* a move is likely, not *which way*, unless you also have the
news-surprise sign or true order flow.

### 4. Order flow is the credible orthogonal FX direction signal — and you've only used a weak proxy

- **Evans & Lyons (2002, J. Political Economy / NBER 7317)**: signed interdealer **order flow** (buy vol − sell
  vol) explains **>50% R² of daily DM/USD changes** and **beats a random walk out-of-sample** at short horizons.
  This is the foundational microstructure-FX result and the strongest evidence that a *direction* signal exists
  in FX flow. (Search; well-established.)
- The catch: Evans-Lyons flow is **true interdealer signed flow** (EBS/Reuters dealing). Your **tick-rule OFI
  from 10s bars is a noisy, sign-error-prone proxy** of this and is exactly why it added ≈0. The gap between
  "tick-rule proxy ≈ 0 lift" and "real flow > 50% R²" is the single largest unexploited lever in this whole
  vector.
- **Cont, Kukanov & Stoikov (2014, "Price Impact of Order Book Events")**: short-horizon price change is a
  **near-linear function of order-flow imbalance (OFI)** at L1, slope ∝ 1/depth. **MLOFI (Xu et al. 2019)**:
  deeper levels add modest explanatory power. **Kolm, Turiel & Westray (2023, Mathematical Finance, "Deep Order
  Flow Imbalance")**: off-the-shelf NNs on *stationary OFI inputs* (not raw books) give state-of-the-art
  multi-horizon return forecasts on 115 Nasdaq names — but **effective horizon ≈ "two average price changes"**
  (i.e., seconds), and "information-rich" names predict better.
- **Stoikov (2018, "The micro-price")**: the **microprice** = mid + adjustment for quote-size imbalance and
  spread; it is a martingale-consistent estimator of the *future* mid and beats the size-weighted mid. **You have
  quote sizes** — this is directly buildable and is a strictly better short-horizon nowcast than mid.

### 5. Liquidity/impact features (Kyle λ, Amihud, Roll, VPIN, Hawkes) — mostly regime, weak direction

- **Kyle's λ** (price per unit signed flow) and **Amihud illiquidity** (|return|/volume) are **magnitude/impact**
  measures. In ML jump/return studies they enter as features that improve *occurrence* and *volatility*
  prediction, not sign. Useful as a *conditioning* feature (impact is high → flow signal stronger).
- **Roll's implied spread** and **effective/quoted spread dynamics** proxy adverse selection; spread *widening*
  often precedes jumps (Liu et al. 2019). Again timing, not direction.
- **VPIN (Easley-López de Prado-O'Hara)**: order-flow *toxicity*. Heavily contested — Andersen & Bondarenko
  show its predictive content is largely a **mechanical function of trading intensity** and it spiked *after*,
  not before, the flash crash. Treat as low-credibility for direction.
- **Hawkes processes**: buy/sell intensity ratio of a bivariate self-/cross-exciting process can be a directional
  *entry* signal and can forecast near-future trade imbalance (Bacry et al. 2013). Credible but again
  seconds-horizon and execution-flavored.

---

## Concrete techniques / features / architectures to try

Ordered by expected value for *your* 15m goal. All assume strict causal computation and purged+embargoed CV.

1. **Microprice + quote-size imbalance features (build from your tick quote sizes).**
   - Per tick: `imbalance I = bid_size/(bid_size+ask_size)`; `weighted_mid = ask·I + bid·(1−I)`.
   - Implement Stoikov's microprice (recursive `G^(n)` adjustment using the empirical transition of
     (imbalance bucket, spread) → future mid). Use the **15m-aggregated drift of (microprice − mid)** and the
     **time-average and end-of-window imbalance** as features. Hypothesis: a persistent quote-size lean carries
     minutes-horizon directional info that the tick-rule sign threw away.
2. **Multi-level OFI (Cont-Kukanov) on your top-of-book size series, aggregated to 15m.**
   - OFI event series → cumulative signed OFI over the bar, and over trailing 1/5/15m windows. Normalize by
     trailing depth (Amihud-style) so it's stationary (this stationarity step is *the* Kolm-Turiel-Westray
     point — feed stationary OFI, not raw sizes).
   - Add **OFI decay/half-life** and **sign-persistence run-length** features.
3. **Semivariance asymmetry as a REGIME GATE on a momentum signal (the Liu et al. 2023 design), adapted to 15m.**
   - Compute RS+, RS-, ΔJ², and the ratio `RS-/RS+` over trailing windows from your 10s bars.
   - Don't feed them as direct direction features. Instead: estimate a *separate* simple momentum/reversal sign,
     and let `ΔJ²` / semivariance-asymmetry **switch** between "trend-continue" and "fade" labeling, then test if
     this conditioning lifts selective-prediction accuracy in the tails. This is the one credible single-asset
     time-series use.
4. **HAR-RV-J volatility nowcast feeding the selective-prediction threshold (not the classifier).**
   - Build HAR-RV (daily/weekly/monthly RV) + jump component (BV vs RV) + signed-jump. Use predicted vol to
     **dynamically set the |p−0.5| betting threshold** and to size. Higher-vol/post-jump windows are where any
     directional edge (flow, news) is largest — concentrate coverage there. This directly extends your 0.632@0.2%
     result toward higher accuracy at matched coverage.
5. **News-window order-flow asymmetry (extends your calendar work).**
   - For the 0–15m after each scheduled US release (FOMC/NFP/CPI), compute **signed flow / OFI in the
     pre-release and first-tick post-release window**. Lee-Wang: jumps cluster here and direction tracks the
     surprise. Even without the macro print, the *immediate post-release flow sign* may be a cleaner directional
     label than time-of-day. Test as a specialist model that only predicts in release windows.
6. **Jump detection (Lee-Mykland) as a feature, not a target.**
   - Flag the last Lee-Mykland jump in the trailing window, its sign, and time-since-jump. Use as interaction
     terms with flow (post-jump continuation vs reversal). Expect occurrence-info, weak direction.
7. **Hawkes buy/sell intensity ratio** (if/when you move to execution-grade horizons) — low priority for 15m.

Architecture note: keep the **direction classifier simple (GBM)**; put the sophistication in **features (microprice/OFI)**
and in the **regime gate + selective-coverage layer**. Kolm-Turiel-Westray's lesson is that *stationary,
well-constructed flow inputs* matter more than network depth.

---

## Reported results & CREDIBILITY assessment

| Result / claim | Horizon | Asset class | Credible & transferable to 15m FX? | Note |
|---|---|---|---|---|
| Evans-Lyons: order flow > 50% R², beats RW OOS | Daily (short-h) | FX (DM/USD) | **High** — *if* you get true signed flow | Your tick-rule proxy is the weak link |
| Stoikov microprice better than mid | Seconds | Equities | **High** (mechanically true) | Horizon is sub-minute; use aggregate drift |
| Cont-Kukanov OFI near-linear price impact | Seconds–minutes | Equities | **Med-High** | Decays fast; matches your decay finding |
| Kolm-Turiel-Westray deep OFI SOTA | "≈2 price changes" | Nasdaq | **Med** | Effective horizon = seconds, not 15m |
| Liu et al. 2023 semivariance gates TSMOM, OOS Sharpe↑ | Daily | Commodity futures | **Med** | Regime gate, daily; adapt cautiously |
| Patton-Sheppard signed jumps → future *vol* | Days–months | Equities | **High but it's a vol signal** | Not direction |
| Amaya et al. realized skew → returns | Weekly | Equity **cross-section** | **Low transfer** | Cross-sectional, not single-series 15m |
| Bollerslev-Li-Zhao good/bad vol prices returns | Weekly+ | Equity **cross-section** | **Low transfer** | Cross-sectional |
| Mäkinen 2019 jump *direction* F≈49% | 5-min | Equities | **High (cautionary)** | Direction ≈ 0.50 — confirms your finding |
| Liu 2019 jump *occurrence* F≈63–72% | 5-min | Equities | **Med** | Occurrence ≠ direction |
| Lee-Wang FX jumps predictable from news | Intraday | FX | **High** | But direction needs the surprise sign |
| VPIN predicts toxicity/crashes | Intraday | Equities | **Low** | Andersen-Bondarenko: mechanical, post-hoc |
| Online "75–90% FX accuracy from OFI" blogs | "minutes" | FX/crypto | **Low / hype** | Usually contemporaneous-OFI leakage |

**Leakage risks to guard against (these produce fake 75%+):**
- **Contemporaneous OFI leakage:** using OFI computed *over the same bar* whose return you predict. OFI and
  same-bar return are mechanically linked (Cont-Kukanov). Your features must end strictly before the bar you
  label. This is the #1 way microstructure backtests fake >70%.
- **Wrong-horizon relabeling:** seconds-horizon LOB edges (DeepLOB, Kolm) reported as "intraday." Their effective
  horizon is seconds; do not read them as 15m evidence.
- **Cross-sectional → time-series confusion:** semivariance/skew long-short alpha is across many assets; it is not
  a per-asset 15m classifier.
- **Overlapping-label / non-purged CV:** 15m labels on 10s bars overlap heavily; without purge+embargo
  (López de Prado) you leak. CPCV recommended.
- **Survivorship/quote-fidelity:** Dukascopy quote sizes are indicative dealer quotes, not a consolidated CLOB —
  the microprice/OFI signal is real but noisier than an equity LOB; validate the sign convention against realized
  next-tick moves before trusting it.

---

## Data sources needed

| Data | For | Where | Free/Paid | Fidelity for our use |
|---|---|---|---|---|
| Raw tick bid/ask + **quote sizes** (you have it) | microprice, OFI, imbalance | Dukascopy (current) | Free-ish | Indicative dealer quotes, not consolidated LOB; usable but validate sign |
| True multi-level FX LOB (depth) | MLOFI, depth slope, Kyle λ | LMAX, Refinitiv FXall, EBS Live, Hotspot | **Paid, expensive** | Highest fidelity; closest to Evans-Lyons flow |
| Interdealer **signed order flow** | the Evans-Lyons signal | CLS, EBS/Reuters aggregates, bank feeds | **Paid/restricted** | The gold standard; hard to obtain |
| Scheduled macro release calendar + **consensus & actual** (surprise) | news-window flow/jumps | Econoday, FXStreet, Bloomberg/Refinitiv, free scrapes | Free–Paid | Need *surprise sign*, not just timing |
| Tick data for jump detection | Lee-Mykland on your series | your existing ticks | Free | Sufficient |

Cheapest high-leverage add: **macro surprise data** (actual − consensus) to pair with your existing release-window
work — turns "vol seasonality" (redundant) into a *directional* signal. Most expensive but highest-upside:
**real depth / signed interdealer flow**.

---

## Relevance & priority for OUR project

**HIGH**
- **Microprice + quote-size imbalance features from your existing tick sizes** (technique 1). Orthogonal to the
  tick-rule sign you already tried; directly buildable; this is the strongest "you have the data, haven't used it
  right" lever. Validate against the documented Stoikov result first.
- **HAR-RV-J vol nowcast driving the selective-prediction threshold** (technique 4). Doesn't need new data,
  directly extends your best result (0.632@0.2%), and uses semivariance/jumps for what they're actually good at.
- **Macro-surprise-signed news-window specialist** (technique 5). Re-opens your "calendar redundant" conclusion
  with a *directional* (surprise-sign) input rather than vol seasonality. Lee-Wang gives the strongest FX
  minutes-horizon directional evidence.

**MEDIUM**
- **Multi-level / depth-normalized OFI** (technique 2), *if* you can get even shallow depth; with only L1 sizes,
  expect partial lift. Stationarity normalization (Kolm) is the key implementation detail.
- **Semivariance-asymmetry regime gate on a momentum signal** (technique 3). The only credible single-asset
  time-series semivariance result, but daily-origin; treat as a coverage/labeling gate, test carefully for
  overfit given small effective N.

**LOW**
- Direct RS+/RS-/ΔJ² as classifier direction features — literature says vol, not direction; expect to
  reproduce the ~0.52 AUC current best level.
- VPIN, raw DeepLOB-style nets for 15m, Hawkes intensity — wrong horizon or contested.
- Cross-sectional skew/semivariance "alpha" — not a single-series 15m signal.

**How it interacts with what you ruled out:** your tick-rule OFI ≈ 0 lift does **not** falsify order-flow as a
signal — it falsifies a *bad sign estimator*. Microprice/size-imbalance and depth-normalized OFI are the
upgraded versions. Your "jump direction is ~0.50 at minutes" intuition is *confirmed* by Mäkinen (F≈49%) —
so don't chase jump *direction*; chase jump *timing × flow* and news-surprise sign. Your calendar = vol
seasonality finding is real but only tested the *magnitude* channel; the *surprise-sign* channel is untested.

---

## Sources (annotated)

1. **Patton & Sheppard (2015), "Good Volatility, Bad Volatility: Signed Jumps and the Persistence of Volatility,"
   Review of Economics and Statistics.** https://public.econ.duke.edu/~ap172/Patton_Sheppard_REStat_2015.pdf
   — *Definitive*: signed jumps/semivariance predict future **volatility**, not return direction. PDF read; eqs. 6–8.
2. **Barndorff-Nielsen, Kinnebrock & Shephard (2010/2008), "Measuring Downside Risk — Realised Semivariance."**
   https://www.nuffield.ox.ac.uk/economics/papers/2008/w2/downside.pdf — Origin of RS+/RS- decomposition.
3. **Liu, Lu, Li & Wang (2023), "Time Series Momentum and Reversal: Intraday Information from Realized
   Semivariance," J. Empirical Finance 72:54-77.** https://centaur.reading.ac.uk/111035/ — The one credible
   single-asset, time-series, OOS use of semivariance asymmetry (regime gate on momentum). PDF read.
4. **Amaya, Christoffersen, Jacobs & Vasquez (2015), "Does Realized Skewness Predict the Cross-Section of Equity
   Returns?" JFE.** https://public.econ.duke.edu/~ap172/ACJV_26Dec2011.pdf — Cross-sectional, weekly; do NOT
   read as single-series 15m.
5. **Bollerslev, Li & Zhao (2020), "Good Volatility, Bad Volatility, and the Cross Section of Stock Returns,"
   JFQA.** https://public.econ.duke.edu/~boller/Papers/jfqa_19.pdf — Cross-sectional pricing of good/bad vol.
6. **Lee & Mykland (2008), "Jumps in Financial Markets: A New Nonparametric Test and Jump Dynamics," RFS.**
   https://academic.oup.com/rfs/article-abstract/21/6/2535/1574138 — Standard intraday jump detector.
7. **Liu et al. (2019), "Predicting intraday jumps in stock prices using liquidity measures and technical
   indicators," arXiv 1912.07165.** https://arxiv.org/pdf/1912.07165 — Occurrence F≈63–72%; **direction ≈ 49%**
   (cites Mäkinen). PDF read. Key cautionary number.
8. **Lee & Wang, "Tales of Tails: Jumps in Currency Markets," J. Financial Markets.**
   https://www.scheller.gatech.edu/directory/research/finance/lee/pdf/jumps_in_currency_markets_jfm5.0.pdf —
   FX jumps predictably triggered by FOMC/NFP; intraday clustering, time-of-day. Abstract read.
9. **Evans & Lyons (2002), "Order Flow and Exchange Rate Dynamics," J. Political Economy / NBER 7317.**
   https://www.nber.org/papers/w7317 — Foundational: signed FX order flow > 50% daily R², beats RW OOS.
10. **Cont, Kukanov & Stoikov (2014), "The Price Impact of Order Book Events," arXiv 1011.6402.**
    https://arxiv.org/abs/1011.6402 — OFI near-linear price impact, slope ∝ 1/depth.
11. **Xu, Gould & Howison (2019), "Multi-Level Order-Flow Imbalance in a Limit Order Book," arXiv 1907.06230.**
    https://arxiv.org/pdf/1907.06230 — Deeper levels add modest explanatory power.
12. **Kolm, Turiel & Westray (2023), "Deep Order Flow Imbalance: Extracting Alpha at Multiple Horizons,"
    Mathematical Finance 33(4).** https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3900141 — Stationary OFI
    inputs → SOTA; effective horizon ≈ 2 price changes (seconds).
13. **Stoikov (2018), "The Micro-Price: A High-Frequency Estimator of Future Prices," Quantitative Finance.**
    https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2970694 — Quote-size + spread → martingale future-mid
    estimator; buildable from your quote sizes.
14. **Deriaz et al. (2024), "Predicting Foreign Exchange EUR/USD direction using machine learning," arXiv
    2409.04471.** https://arxiv.org/pdf/2409.04471 — Daily horizon, best accuracy **58.5%** (meta-estimators);
    a current best FX directional level at daily. PDF read.
15. **Easley, López de Prado & O'Hara (2012), VPIN / flow toxicity** (and Andersen-Bondarenko critique,
    "VPIN and the flash crash," 2014). https://www.quantresearch.org/VPIN.pdf — Contested; treat as low-credibility
    for direction.
16. **Bacry et al. (2013), "Hawkes model for price and trades high-frequency dynamics," arXiv 1301.1135.**
    https://arxiv.org/pdf/1301.1135 — Self-/cross-exciting trade arrivals; buy/sell intensity as directional entry.
17. **López de Prado, "Advances in Financial ML" / purged & embargoed CV, CPCV; "10 Reasons Most ML Funds Fail."**
    https://www.garp.org/hubfs/Whitepapers/a1Z1W0000054x6lUAA.pdf — Anti-leakage methodology; mandatory for any
    microstructure backtest here.
