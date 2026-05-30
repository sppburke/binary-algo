# Binary-Option Economics, Required Hit-Rate, and Less-Efficient Instruments (Deriv, Nadex, Prediction Markets)

> Research vector for the binary-algo project. Goal context: predict EURUSD/major-FX **direction** at a
> **15-minute** horizon with **>75% out-of-sample** accuracy. We have reached a current best level on liquid FX at
> 5m/15m (~0.52 AUC, ~0.63 selective) and so far reached ≥75% only at the **3-second** microstructure horizon.
> This report asks: *what does 75% actually buy you economically, and is there a structurally
> less-efficient instrument where a longer-horizon directional edge is more reachable?*

---

## TL;DR (most actionable findings for our 15m FX goal)

- **You do not need 75% to make money on binaries — but the project's 75% target is the right bar *because of
  payout/cost geometry, not bankroll growth.*** Break-even win-rate is `1/(1+payout)`: **57.1% at 0.75
  payout, 55.6% at 0.80, 52.6% at 0.90, 51.3% at 0.95.** 75% is a large margin over break-even at any
  realistic payout — i.e. 75% is *aspirational headroom*, while the *real* commercial threshold is ~57–59%.
  ([Options Guide / break-even](https://www.theoptionsguide.com/binary-options-win-rate-payout-breakeven.aspx),
  derived below.)
- **At 75% accuracy and an 80% payout, full-Kelly stake is ~44% of bankroll and EV is +$0.35 per $1 risked**
  — enormous, *if real*. The danger is not sizing; it is that any claimed 75% at 15m on liquid FX is almost
  certainly leaked/overfit (our own work shows the linear best-so-far is ~0.51).
- **Deriv synthetic indices are a dead end for *directional* prediction by construction.** Deriv's Head of
  Quants states prices come from a **central CSPRNG broadcast** (geometric-Brownian-style constant-vol random
  walk). A CSPRNG-driven GBM has **independent, zero-drift increments → direction is a martingale →
  unpredictable in principle.** Do not spend modeling effort trying to predict V10/V25/V75 up/down.
  ([Deriv: Do brokers manipulate synthetic indices](https://experts.deriv.com/insights/do-brokers-manipulate-synthetic-indices))
- **BUT synthetic indices have one genuinely exploitable property: the *volatility is known and constant*,
  and Crash/Boom indices have a *known Poisson-like spike intensity*.** This makes **volatility-structured
  binaries (touch / no-touch / range / over-under)** analytically priceable. If Deriv mis-prices a
  vol-contingent contract relative to its own generator's σ, that is a real, *non-directional* edge —
  orthogonal to everything we've tried.
- **[DEFERRED — majors-only scope, 2026-05-30]** **A credible, peer-reviewed orthogonal direction documented in the literature is CRYPTO, not synthetics.** Multiple journals
  (ScienceDirect, Springer, Nature Sci. Reports) document Bitcoin/altcoins **deviating from the random-walk
  hypothesis at 15/30/60-min intraday intervals**, with **intraday momentum + reversal** effects and
  **altcoins markedly less efficient** than majors. This is the strongest "less-efficient instrument"
  signal in the literature and is directly testable with free data. *(Retained as analysis; not a current target under the majors-only scope decision of 2026-05-30.)*
- **Avoid unregulated binary brokers (Pocket Option et al.) entirely as a data/edge source.** SEC/CFTC and
  multiple reviews document **OTC price-feed manipulation and adverse-selection algorithms** ("learn your
  strategy, trade against you"). Any backtest edge there is unrealizable; the counterparty controls the tape.
- **Nadex (the only US-regulated binary venue) shut binary trading on 2025-12-20** — no longer an option.
  Where an edge existed, it was eaten by spread + fee. Regulated *prediction markets* (Kalshi) are the live
  successor but are event-driven, not 15m FX tick markets.
- **Net recommendation for the project:** treat "75% at 15m on liquid FX" as **not yet achieved (open target)**.
  The previously-suggested redirects — **[DEFERRED — majors-only scope, 2026-05-30]** (1) **crypto majors/alts at 15m** (less-efficient, real-data, real-edge),
  and (2) **vol-structured (not directional) binaries on Deriv synthetics** where the generator's known σ is
  the edge — are retained as documented analysis but are **not current targets** under the majors-only scope
  decision of 2026-05-30. (Earlier ranking, for the record: crypto intraday = High, synthetic vol-binaries = Med,
  synthetic direction / unregulated brokers = Do-not-pursue.)

---

## Key findings (detailed, with inline citations)

### 1. The binary payout/break-even/Kelly math — exactly what 75% buys

A binary option pays a fixed fraction `b` (the "payout", typically 0.70–0.95) of stake on a win and loses
the full stake on a loss. With win-rate `w`:

- **Expected value per $1 staked:** `EV = w·b − (1−w)`.
- **Break-even win-rate (EV = 0):** `w* = 1/(1+b)`.
- **Full-Kelly fraction of bankroll:** `f* = (b·w − (1−w)) / b` (the standard `f = (bp−q)/b` with odds
  `b`). ([Kelly criterion — Wikipedia](https://en.wikipedia.org/wiki/Kelly_criterion);
  [arXiv 1411.3615, Kelly for variable payoff](https://arxiv.org/pdf/1411.3615))

Computed from first principles (verified numerically this session):

| Payout `b` | Break-even win-rate `1/(1+b)` | Full-Kelly stake @ w=0.60 | Full-Kelly @ w=0.75 | EV/$1 @ w=0.75 |
|---|---|---|---|---|
| 0.70 | **58.82%** | 2.9% | 39.3% | +0.275 |
| 0.75 | **57.14%** | 6.7% | 41.7% | +0.313 |
| 0.80 | **55.56%** | 10.0% | 43.8% | +0.350 |
| 0.85 | **54.05%** | 12.9% | 45.6% | +0.388 |
| 0.90 | **52.63%** | 15.6% | 47.2% | +0.425 |
| 0.95 | **51.28%** | 17.9% | 48.7% | +0.463 |
| 1.00 | **50.00%** | 20.0% | 50.0% | +0.500 |

**Interpretation for the project:**
- The *economic* threshold is **~57–59%** at realistic payouts, not 75%. Our best 15m selective result
  (0.632 OOS) is **already above break-even at every payout ≤ 0.78** — but only at ~0.2% coverage, so it
  produces almost no bets and the small-n makes the edge statistically fragile.
- 75% is the right *engineering* target because it gives a fat cushion against (a) payout < spot accuracy,
  (b) the *spread/fee* embedded in OTC binaries (which effectively lowers `b` by 5–15 pts), and (c)
  probability-estimation error, which fractional-Kelly users discount by 50–75%
  ([Matthew Downey, fractional Kelly simulations](https://matthewdowney.github.io/uncertainty-kelly-criterion-optimal-bet-size.html)).
- **Payout sensitivity is steep:** a 5-pt payout drop (85%→80%) raises break-even by ~1.5 pts
  ([Binany scalping win-rate](https://learn.binany.com/trading-strategy/scalping-win-rate-for-binary-options/)).
  On OTC brokers the *effective* payout is the quoted payout minus the hidden quote spread, so a "85% payout"
  market can have a true break-even near 60%.

### 2. Deriv synthetic indices — direction is provably unpredictable, volatility is the only edge

Deriv's Head of Quants (Prashant Sinha, 2026-01-20) states directly that synthetic-index prices are produced
by a **single central Cryptographically Secure Pseudo-Random Number Generator (CSPRNG)**, **broadcast
identically to all 2.5M+ clients**, with a seed-based design making it "mathematically and technically
impossible to target your specific trade"
([Deriv experts](https://experts.deriv.com/insights/do-brokers-manipulate-synthetic-indices)). Marketing and
academy pages describe the underlying process as a **continuous random walk with a fixed, engineered
volatility** (V10/V25/V50/V75/V100 = constant 10–100% annualized vol; one tick per 1s or 2s)
([Deriv Academy — volatility indices](https://traders-academy.deriv.com/trading-guides/guide-to-understanding-volatility-indices);
[Deriv synthetic indices](https://deriv.com/markets/derived-indices/synthetic-indices)).

**Mechanistic consequence (Tier-1 derivation):** a CSPRNG-driven geometric Brownian motion has i.i.d.
log-return increments with **zero (or fixed) drift and constant σ**. The conditional expectation of the next
move given all history is the current price → it is a **martingale** → **directional up/down is ~0.50
no matter how much past data you feed a model.** This is the *opposite* of real FX, where (weak) order-flow
and microstructure create exploitable autocorrelation. **Conclusion: any "I predict V75 direction with X%"
claim is either curve-fit noise or a backtest that won't replicate.** Our own next-tick decay finding
(55.3% → ~0.50) is a *real-market* effect; on synthetics even that does not exist.

**The exploitable corner — volatility, not direction:**
- Because **σ is published and constant**, the fair price of any **volatility-contingent binary** (touch /
  no-touch / stays-in-range / over-under / digits) is **computable in closed form** (barrier-hitting
  probabilities for GBM). If Deriv's quoted contract price deviates from the generator-implied probability,
  that mispricing is a pure, *non-directional* statistical edge. ([Touch/No-Touch mechanics](https://www.binaryoptiontrading.com/guides/touch-no-touch/);
  [Deriv synthetic indices](https://deriv.com/markets/derived-indices/synthetic-indices))
- **Crash/Boom indices** add a **Poisson-like spike process**: e.g. "Boom 900" spikes on average every 900
  ticks, so ~144 spikes per 86,400-tick day
  ([Deriv Crash/Boom](https://deriv.com/markets/derived-indices/crash-boom);
  [traders-academy Crash/Boom 150](https://traders-academy.deriv.com/trading-guides/crash-boom-150-derived-indices)).
  Between spikes the drift is *gently adverse* (slow bleed) and spikes are *one-directional*. This is a
  **known hazard-rate process**, so the EV of "no-spike before tick N" contracts is analytically derivable.
  This is the one place a *direction-ish* edge exists on synthetics — but it comes from the **known spike
  asymmetry**, not from predicting the random component.

### 3. Crypto as the credible "less-efficient instrument" — **[DEFERRED — majors-only scope, 2026-05-30]**

Peer-reviewed evidence that crypto is *less efficient than FX* at exactly our horizon:

- **Bitcoin deviates from the random-walk hypothesis at intraday 15/30/60-min intervals**, with efficiency
  varying over time and strong evidence of serial dependence
  ([Sci. Reports, Bitcoin market efficiency](https://www.nature.com/articles/s41598-023-31618-4);
  [Testing RWH in crypto, SAGE 2022](https://journals.sagepub.com/doi/10.1177/23197145221101238)).
- **Intraday momentum AND reversal coexist in crypto**: intraday momentum (late-informed investors) +
  intraday reversal (overreaction) unique to crypto, producing **higher economic value than benchmarks**
  ([Intraday return predictability in crypto, J. Int. Money & Finance](https://www.sciencedirect.com/science/article/abs/pii/S1062940822000833)).
- **Altcoins are markedly less efficient than BTC** (fragmented liquidity, manipulation susceptibility) — so
  the directional edge is *larger* down the cap curve
  ([Efficiency of crypto markets, HFT data](https://www.researchgate.net/publication/395593609_The_Efficiency_of_Crypto_Markets_Evidence_from_High-_Frequency_Trading_Data);
  [Price transmission BTC→altcoins, HFT](https://link.springer.com/article/10.1007/s10690-026-09589-z)).
- **Lead-lag is strong and tradable**: BTC leads alts at high frequency — a *cross-asset* version of the
  cross-pair lead-lag we tried in FX (V3), but with a *bigger* and *slower-decaying* effect because crypto
  microstructure is less arbitraged
  ([BTC→altcoin price transmission](https://link.springer.com/article/10.1007/s10690-026-09589-z)).

**Why this matters vs. what we ruled out:** our FX current best level is *because EURUSD is ~97% USD-factor and ~efficient*.
Crypto breaks both: weaker informational efficiency at 15m, real (not proxy) order-flow available free from
exchange APIs, and a dominant lead asset (BTC) whose moves propagate with a lag. The same 239-feature +
cross-asset + order-flow pipeline *could* be re-pointed at BTC/ETH/large-alt 15m bars — **[DEFERRED — majors-only
scope, 2026-05-30]**: retained as analysis, not a current target.

### 4. Nadex, prediction markets, and unregulated brokers

- **Nadex** (CFTC-regulated, the only legitimate US binary venue) **stopped accepting binary-options trades on
  2025-12-20** and is not onboarding new clients. Even before that, practitioner consensus was that any edge
  was lost to **crossing the spread + transaction fee + hitting a market-maker** ([binaryoptions.net Nadex
  review](https://www.binaryoptions.net/nadex)). **Not a viable venue now.**
- **Kalshi (CFTC-regulated prediction market)** replaced the "regulated binary" niche but trades **event
  contracts via a central limit order book**, not 15m FX tick binaries. Pricing is peer-to-peer (no house
  vig); **taker fee = 0.07·price·(1−price)** (max ~1.75% at $0.50, ~0.6% on tails); documented
  **favorite-longshot bias** offers edge to informed traders
  ([Kalshi fees](https://help.kalshi.com/trading/fees);
  [Math of prediction markets / CLOB](https://navnoorbawa.substack.com/p/the-math-of-prediction-markets-binary)).
  Relevant only if we reframe the target as *event/threshold* contracts (e.g. "EURUSD > X at NY close"),
  which is a different, longer-horizon problem than 15m direction.
- **Pocket Option and similar offshore OTC brokers: do not use as a data source or edge target.** The
  broker *is* the counterparty and *generates the OTC tape*; SEC/CFTC alerts document software that
  "manipulate[s] binary options prices and payouts," and reviews report adverse-selection algorithms that
  learn and fade profitable users, plus withdrawal blocks
  ([CFTC binary-options fraud alert](https://www.cftc.gov/LearnAndProtect/AdvisoriesAndArticles/fraudadv_binaryoptions.html);
  [SEC investor alert PDF](https://www.sec.gov/investor/alerts/ia_binary.pdf)). Any backtested edge there is
  fictional because the price you backtest on is not the price you'll be filled at.

---

## Concrete techniques / features / architectures to try

1. **Re-point the existing 15m pipeline at crypto. [DEFERRED — majors-only scope, 2026-05-30]** *(Documented for later; not a current target. Earlier framing: highest leverage, lowest new code.)*
   - Ingest free 1m/OHLCV + order-book/trade tick data for BTCUSDT, ETHUSDT, and 5–10 liquid alts from
     Binance/Bybit/OKX public APIs (or Kaggle dumps). Build 15m direction labels identically to the FX setup.
   - Reuse `pipeline.py` (239 TA features), `crosspair.py` (now **BTC-as-leader** lead-lag into alts),
     `orderflow.py` — but with **real signed trade flow and book imbalance**, not the 10s tick-rule proxy.
   - **New, crypto-specific orthogonal features:** funding-rate sign/level (perp basis), open-interest
     deltas, liquidation-cascade proximity, BTC-dominance regime, exchange-to-exchange price dispersion
     (cross-venue stat-arb residual). These have *no FX analog* and are where the inefficiency lives.
   - Test the **intraday-momentum + intraday-reversal** decomposition explicitly (first-bar-of-session return
     as a predictor), per the crypto-predictability literature.

2. **Vol-structured binaries on Deriv synthetics (direction-free edge).**
   - Pull the live tick feed via the **Deriv WebSocket API** (free, real-time, documented) for V10/V25/V75
     and Crash/Boom. Estimate the *realized* σ and *empirical* spike hazard, and compare to the published
     constant-vol/spike parameters.
   - Price **touch / no-touch / stays-in / over-under** contracts with the GBM barrier-hitting formulas,
     then compare to Deriv's quoted contract prices. **Trade only the gap** (mispricing of a vol-contingent
     contract), never up/down. This is a clean, *measurable* EV strategy with a known generator.
   - For Crash/Boom: model the inter-spike interval as Poisson/renewal, derive EV of "no spike in next N
     ticks" vs quoted price; the slow adverse bleed between spikes must be netted against spike payoff.

3. **Calibrate the 75% target to the real cost stack.** Before any new model, encode the
   `EV = w·b_eff − (1−w)` constraint where `b_eff = quoted_payout − round_trip_spread_in_payout_terms`.
   Output the **minimum w at the venue's true `b_eff`** and gate the selective-prediction threshold on
   *that*, not on a generic 0.75. (On regulated/exchange instruments `b_eff` is much closer to fair, which
   is another reason to prefer crypto/Kalshi over OTC binaries.)

4. **Fractional-Kelly sizing harness.** Add a sizing layer: `f = κ · (b·p̂ − (1−p̂))/b` with κ ≈ 0.25–0.5
   and `p̂` from the *calibrated* model probability (Platt/isotonic on VAL). Cap per-trade and per-day risk.
   This converts any verified >break-even edge into a growth-optimal but drawdown-bounded strategy and makes
   the economic value of a given accuracy explicit.

5. **(Optional) Kalshi/threshold reframe.** If 15m tick direction stays open, reframe to
   *threshold-at-fixed-time* contracts ("EURUSD ≥ K at 16:00 ET"). These are priceable from our existing vol
   models, trade on a regulated CLOB with transparent ~0.6–1.75% fees, and have a documented
   favorite-longshot bias to fade. Different horizon, but a *real* venue with a *real* edge.

---

## Reported results & CREDIBILITY assessment

**Credible / reproducible:**
- **Break-even and Kelly math** — Tier-1 derivation, verified numerically this session; matches multiple
  independent sources. *No leakage risk; it's arithmetic.*
- **Deriv = CSPRNG broadcast → direction unpredictable** — stated by Deriv's own Head of Quants; consistent
  across Deriv academy/marketing. The *direction-is-a-martingale* conclusion follows by math, not marketing.
  **High confidence.** (Caveat: this is the vendor's self-description; we cannot independently audit the RNG,
  but the broadcast architecture and the absence of any real microstructure make a directional edge
  implausible regardless.)
- **Crypto intraday inefficiency (15/30/60m)** — multiple peer-reviewed journals (Nature Sci. Reports,
  ScienceDirect, SAGE, Springer). **Credible direction-of-effect**, though *effect sizes are small and
  time-varying*; "less efficient than FX" is well supported, "75% accuracy" is **not** claimed by any of
  them. Treat as "where to dig," not "guaranteed 75%."
- **Nadex shutdown (2025-12-20), Kalshi fee formula** — verifiable venue facts.

**Overfit / hype / leakage-risk (treat with skepticism):**
- **Any "predict synthetic-index direction with 70–95%" content** (YouTube/blogs/courses) — *mathematically
  impossible* against a CSPRNG martingale; these are survivorship-biased backtests or outright marketing.
- **"Directional forex forecasting, 8 pairs, ML"** (Springer Discover AI 2025) — uses **daily** bars,
  2018–2023, only **1,565 days**, and 5-fold CV with RandomizedSearch; **no realistic transaction costs or
  binary-payout framing.** Daily ≠ our 15m problem, and small-sample CV accuracy on daily FX direction is a
  known overfit trap. *Low transfer value; cite only as "ML-on-FX-direction exists but at daily horizon."*
- **Pocket Option / OTC broker "strategies"** — edge is unrealizable; the broker controls and can manipulate
  the tape (SEC/CFTC documented). **Reject.**
- **General "5-min equity ML, Sharpe 0.98 after costs"** (Liu/Stentoft; Chinco et al.) — *credible* and
  peer-reviewed, but (a) equities not FX, (b) it's a *continuous return* timing edge (R²≈0.24%), which is far
  below binary-direction 75%; it reinforces that *small* intraday edges are real but *75% directional is not
  the shape of the documented edge.*

**Leakage flags to watch when we test crypto:** look-ahead in funding/OI timestamps, survivorship in
delisted alts, exchange clock skew, and using *settlement* prices that aren't tradable in real time.

---

## Data sources needed

| Data | Where | Free/Paid | Fidelity / notes |
|---|---|---|---|
| Crypto 1m/15m OHLCV (BTC/ETH/alts) | Binance/Bybit/OKX REST + Kaggle dumps | Free | High; years of history; multi-venue |
| Crypto trades + L2 order book (real flow) | Exchange WebSocket; Tardis.dev for historical L2 | Free live / Paid history | **Real** order flow & book imbalance (vs our 10s proxy) |
| Perp funding rate, open interest, liquidations | Exchange APIs; Coinglass | Free/freemium | The *orthogonal* crypto-only features |
| Deriv synthetic tick feed (V10–V100, Crash/Boom) | **Deriv WebSocket API** (`ws.derivws.com`) | Free | Real-time + historical ticks; the generator we'd price against |
| Deriv contract quotes (touch/no-touch/over-under) | Deriv API `proposal` calls | Free (demo account) | Lets us measure quoted-vs-fair mispricing |
| Kalshi event contracts + order book | Kalshi API | Free (account) | Regulated CLOB; threshold-contract reframe |
| Binary break-even/Kelly references | (math, no data) | — | Derive in-repo |

**Do NOT use:** Pocket Option / IFMRRC-only brokers as a data or execution source (manipulated OTC tape).

---

## Relevance & priority for OUR project

**High**
- **Encoding the true break-even/`b_eff`/Kelly economics into the harness.** Cheap, makes the selective
  threshold venue-correct, and reframes whether 0.632 OOS is already commercially useful at some payout.

**Med**
- **Deriv vol-structured (non-directional) binaries.** Real, measurable edge *only* on volatility-contingent
  contracts where the generator's σ/spike intensity is known. Orthogonal to all 17 FX variants. Requires a
  different model class (barrier/hazard pricing, not direction classification) — new work, but well-posed and
  with a *known* data-generating process (rare luxury). Crash/Boom spike-asymmetry is the most direction-like
  sub-case.
- **Kalshi threshold-contract reframe.** Real regulated venue, transparent fees, exploitable favorite-longshot
  bias — but it's a *different horizon/target* (event/threshold), so it's a pivot, not an extension.

**Deferred — majors-only scope (2026-05-30)**
- **Crypto majors/alts at 15m via the existing pipeline. [DEFERRED — majors-only scope, 2026-05-30]** Retained
  as documented analysis, not a current target. (Earlier assessment: best fit for "longer-lived
  orthogonal signal at minutes-to-hours"; directly attacks our root cause — EURUSD is ~efficient,
  ~97% USD-factor — because crypto is *documented* less efficient at 15/30/60m, has *real* order flow, and a
  *lead asset (BTC)*. Reuses `pipeline.py`/`crosspair.py`/`orderflow.py` with stronger inputs, and adds
  genuinely new features (funding/OI/liquidations/cross-venue) with no FX analog.)

**Low / Do-not-pursue**
- **Predicting synthetic-index *direction* (V10–V100).** Provably a martingale; any apparent edge is noise.
  *Explicitly rule out* so no future variant wastes cycles here.
- **Unregulated OTC binary brokers (Pocket Option etc.)** as data or execution. Manipulated counterparty tape.
- **Nadex.** Binary trading discontinued 2025-12-20.
- **Daily-bar FX ML papers** as a template — wrong horizon, small-sample CV, no cost model.

**Interaction with what we already ruled out:** crypto lead-lag is the *stronger* cousin of our FX cross-pair
V3 (BTC lead decays slower than EUR/peer lead); crypto real order flow is the *upgrade* to our null-lift 10s
OFI proxy (V3/V10); funding/OI/liquidations are *new orthogonal axes* we never had in FX. The 75% target
should be re-expressed as "beat `b_eff` break-even with usable coverage," which our 0.632 selective result
already does at low payouts — the open question is **coverage and stability**, not crossing break-even.

---

## Sources (annotated)

1. **Binary Options: Calculating Breakeven Win-Rate for a Given Payout** — The Options Guide.
   https://www.theoptionsguide.com/binary-options-win-rate-payout-breakeven.aspx — canonical `1/(1+payout)`
   break-even formula and payout/win-rate table.
2. **Break Even Ratios in Binary Trading** — BinaryTrading.com.
   https://www.binarytrading.com/break-even-ratios-in-binary-trading/ — `X = OTM%/(ITM%+OTM%)` derivation,
   confirms 57–59% break-even band.
3. **Scalping Win Rate for Binary Options** — Binany.
   https://learn.binany.com/trading-strategy/scalping-win-rate-for-binary-options/ — payout sensitivity
   (5-pt payout drop ≈ +1.5 pt break-even); practitioner framing.
4. **Kelly Criterion** — Wikipedia. https://en.wikipedia.org/wiki/Kelly_criterion — `f=(bp−q)/b`, full vs
   fractional Kelly, halving-before-doubling risk.
5. **Kelly criterion for variable pay-off** — arXiv 1411.3615. https://arxiv.org/pdf/1411.3615 — rigorous
   variable-payoff Kelly (matches binary `b`-to-1 case).
6. **Why fractional Kelly? Simulations under uncertainty** — Matthew Downey.
   https://matthewdowney.github.io/uncertainty-kelly-criterion-optimal-bet-size.html — why 0.25–0.5 Kelly
   under probability-estimation error (directly relevant to our `p̂` noise).
7. **Do brokers manipulate synthetic indices?** — Deriv (Head of Quants), 2026-01-20.
   https://experts.deriv.com/insights/do-brokers-manipulate-synthetic-indices — **primary source**: central
   CSPRNG broadcast → directional martingale; the key "don't predict synthetic direction" evidence.
8. **Guide to understanding Volatility Indices** — Deriv Academy.
   https://traders-academy.deriv.com/trading-guides/guide-to-understanding-volatility-indices — constant-vol
   (10–100%), tick cadence 1s/2s; basis for closed-form vol-binary pricing.
9. **Crash/Boom Indices** — Deriv. https://deriv.com/markets/derived-indices/crash-boom — Poisson-like spike
   intensity (avg ticks between spikes); the known-hazard, direction-asymmetric sub-case.
10. **Crash & Boom 150** — Deriv Traders Academy.
    https://traders-academy.deriv.com/trading-guides/crash-boom-150-derived-indices — concrete spike-count
    arithmetic (ticks/day ÷ spike interval).
11. **Intraday return predictability in crypto: momentum, reversal, or both** — J. Int. Money & Finance
    (ScienceDirect). https://www.sciencedirect.com/science/article/abs/pii/S1062940822000833 — crypto
    intraday momentum + reversal, higher economic value than benchmarks.
12. **Market efficiency of cryptocurrency: evidence from the Bitcoin market** — Nature Sci. Reports 2023.
    https://www.nature.com/articles/s41598-023-31618-4 — Bitcoin deviates from RWH, time-varying efficiency.
13. **Testing of Random Walk Hypothesis in the Cryptocurrency Market** — SAGE 2022.
    https://journals.sagepub.com/doi/10.1177/23197145221101238 — variance-ratio rejection of RWH at intraday
    intervals.
14. **The Efficiency of Crypto Markets: HFT Data** — ResearchGate 2025.
    https://www.researchgate.net/publication/395593609 — BTC more efficient than alts; alts = bigger edge.
15. **Price Transmission from Bitcoin to Altcoins: HF Evidence** — Springer (Asia-Pac. Fin. Markets) 2026.
    https://link.springer.com/article/10.1007/s10690-026-09589-z — BTC→alt lead-lag, tradable at HF.
16. **CFTC Investor Alert: Binary Options and Fraud.**
    https://www.cftc.gov/LearnAndProtect/AdvisoriesAndArticles/fraudadv_binaryoptions.html — OTC software
    price/payout manipulation; why offshore brokers are unusable.
17. **SEC Investor Alert: Binary Options and Fraud (PDF).** https://www.sec.gov/investor/alerts/ia_binary.pdf
    — regulator confirmation of manipulated binary platforms.
18. **Nadex Review 2026** — binaryoptions.net. https://www.binaryoptions.net/nadex — Nadex stopped binary
    trading 2025-12-20; edge eaten by spread+fee.
19. **Kalshi Fees** — Kalshi Help Center. https://help.kalshi.com/trading/fees — `0.07·p·(1−p)` taker fee;
    regulated CLOB successor to binaries.
20. **The Math of Prediction Markets: Binary Options, Kelly, CLOB** — N. Bawa (Substack).
    https://navnoorbawa.substack.com/p/the-math-of-prediction-markets-binary — ties binary payout math,
    Kelly, and CLOB pricing together (practitioner synthesis).
21. **Directional forecasting for eight forex pairs using ML** — Springer Discover AI 2025.
    https://link.springer.com/article/10.1007/s44163-025-00424-4 — *daily*-horizon FX direction ML; cited as
    a credibility *counter-example* (wrong horizon, no costs, small sample).
22. **Intraday Market Predictability: A Machine Learning Approach** — Huddleston/Liu/Stentoft, J. Financial
    Econometrics. https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3726765 — real but *small* intraday ML
    edges (Sharpe ~0.73 after cost), showing 75% directional is not the documented edge shape.
