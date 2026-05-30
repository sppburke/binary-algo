# Gaps, Blind Spots & Skepticism Audit of the 16 Research Reports

*Completeness critic pass over reports 01–16 for the binary-algo 15m FX direction project.
Goal context: EURUSD (and USD-major) up/down at 15m, >75% strict OOS. Already exhausted: 239 causal TA,
cross-pair lead-lag, 10s tick-rule OFI (~0 lift), GBM/NN ensembles (~0.52–0.527 AUC), stat-arb basket,
daily/weekly context, exogenous peer features, calendar event-timing proxy, selective prediction
(best 0.632 @ 0.2% coverage). Clean >75% only at the 3-second horizon; raw imbalance decays to coin-flip
by ~1 min, gone by 5 min. EURUSD ~97% USD-factor; lag-1 autocorr of 5m returns ≈ −0.03.*

**One-line verdict on the 16 reports:** collectively excellent and admirably skeptical, but they *converge*
(by independent rediscovery) on the same five levers — fix-window drift, signed-surprise news, rate-diff
fair-value, meta-labeling/conformal, regime gating — and they systematically **under-cover how to USE the
one edge you actually have (the 3-second signal)**, **what target to predict**, and **which instrument to
predict**. The biggest missing ideas are not new features; they are (a) an optimal-aggregation/execution
theory that turns your fast edge into a slower position, (b) a different prediction *target*, and (c) a
different *instrument*. Below, ranked.

---

## Missing or under-covered angles

### 1. Optimal aggregation of a fast alpha into a slow position — the Gârleanu–Pedersen / Almgren short-term-alpha gap (BIGGEST)
**What:** You have a *real, clean* directional edge at 3 seconds and near-nothing at 15m. Every report treats
these as separate problems and concludes "the 15m signal isn't there." None asks the rigorous question:
**can a strong, fast-decaying signal be optimally compounded/held into a 15m-horizon directional bet under
transaction costs?** This is exactly the problem solved in closed form by **Gârleanu & Pedersen, "Dynamic
Trading with Predictable Returns and Transaction Costs" (J. Finance 2013)** — optimal policy with *multiple
predictors of different mean-reversion (alpha-decay) speeds*: "aim in front of the target, trade partially
toward the aim," and **slower-decaying predictors get more weight**. The companion practical line is **Almgren
/ "Optimal Execution of Portfolio Transactions with Short-Term Alpha"** and **Gârleanu-style optimal-trading-
with-alpha-predictors** — how to convert a short-lived mid-price predictor into a position/trajectory.
**Why it matters:** your 3s edge is the *only* thing in the entire research corpus that is verified to clear
75%. The unexamined hypothesis is that the 15m *bar label* is the wrong object: instead of predicting the 15m
sign directly, you could **stack/Kalman-filter the decaying 3s signal forward** and hold a position whose
*sign over the 15m window* inherits the fast edge, accepting that each 3s prediction is individually short-
lived but their *time-integral* has a non-trivial drift. None of the 16 reports models the signal-to-noise of
the **aggregate of many fast bets** vs the single 15m bet — a basic, answerable question (a 55% edge refreshed
every 3s, even with fast decay and cost, can dominate a 51% edge refreshed every 15m). This is the highest-
value untouched idea and it uses data you already have.
*Starting sources:* Gârleanu & Pedersen, https://nbgarleanu.github.io/DynTrad.pdf ; "Optimal Execution with
Short-Term Alpha," https://arxiv.org/pdf/2502.04284 (alpha-decay + cost multi-period) ; "Optimal trading
without optimal control," https://arxiv.org/pdf/2012.12945 .

### 2. Breedon–Ranaldo intraday *return* seasonality (local-hours depreciation) as a first-class directional signal
**What:** Reports 10 and 01 mention the Krohn fix-reversal heavily, but the **Breedon & Ranaldo, "Intraday
Patterns in FX Returns and Order Flow" (SNB WP 2011-04 / QMUL 694)** effect is given only a one-line aside.
It is distinct and arguably *more* exploitable at 15m: currencies **depreciate during their own local trading
hours and appreciate during USD local hours**, driven by *mechanical corporate conversion flow* (European
importers sell EUR for USD in EU hours → EUR weakness in that window). This is a **signed, time-deterministic,
order-flow-grounded drift** that (a) is not your time-of-day *vol* feature, (b) is not the fix-window effect,
and (c) has a real microstructural cause (regular business-day conversion), so it is more likely to persist
than a statistical anomaly.
**Why it matters:** it is free (you have the bars), zero-leakage (clock-deterministic), orthogonal to TA, and
specifically EUR-relevant. It should be a *separate* directional overlay, tested per-session, not folded into
the all-bars model where it averages away.
*Starting source:* https://www.snb.ch/public/asset/en/www-snb-ch/publications/research/working-papers/2011/working_paper_2011_04/publications0_en/working_paper_2011_04.n.pdf

### 3. Reframe the target: cross-sectional learning-to-rank across the 7 pairs, not per-pair sign
**What:** Every report predicts EURUSD's *own* sign. The project *has 7 USD pairs* but only ever uses them as
lead-lag *features*. The competition meta-report (11) even notes the winning pattern in JPX/Ubiquant/Optiver
was **rank/cross-sectional**, not absolute direction — yet no report proposes a **learning-to-rank** target:
at each 15m bar, rank the 7 pairs by expected next-15m return and bet the **top-minus-bottom spread** (which is
dollar-neutral and strips out the ~97% common USD factor that is killing you). Because EURUSD is 97% USD-factor,
its *idiosyncratic* (euro-leg) signal is buried; a cross-sectional rank **explicitly removes the common factor**
and exposes the residual that may be more predictable. There is direct FX evidence this helps:
**"Enhancing Cross-Sectional Currency Strategies by Context-Aware Learning-to-Rank" (arXiv:2105.10019)**.
**Why it matters:** it directly attacks the stated root cause ("97% USD factor"). Binary direction on one pair
forces you to predict the factor; ranking lets you predict only the residual. Different, well-evidenced target
shape, uses existing data.
*Starting source:* https://arxiv.org/pdf/2105.10019

### 4. Smarter *labels* than next-15m-sign: signed path-dependence and area/touch targets
**What:** Report 14 covers triple-barrier and signatures, and report 06 covers triple-barrier for meta-labels,
but no report tests **alternative directional *targets*** as the primary object. Two concrete, under-covered
ones: (a) **signed path-dependence** — a model-free test/predictor where the *sign of cumulative innovations
over a lookback* predicts the *sign of cumulative innovations over the forecast horizon* ("A Non-parametric
Test and Predictive Model for Signed Path Dependence," Comp. Economics 2020) — reported significant after costs
in equities, never tested on your FX; (b) **"will price touch +k before −k"** (a barrier-touch probability),
which is the *actual* binary-option-relevant question and is often more predictable than end-point sign because
it rewards path/volatility-asymmetry, not just drift.
**Why it matters:** your label ("sign of 15m return") is the least informative target and the one most
contaminated by the coin-flip drift. A touch/area target can be 55–60% predictable when end-sign is 51%,
and it is the economically correct target for binaries. Pure target-engineering, free.
*Starting source:* https://link.springer.com/article/10.1007/s10614-019-09934-7

### 5. Wrong instrument: EM/exotic USD pairs as the less-efficient FX target (report 12 only did this for crypto)
**What:** Report 12 makes the strong case to **re-point the whole pipeline at a less-efficient instrument** —
but only considers crypto. The same logic applies *within FX*: **USDMXN, USDZAR, USDTRY, USDINR** are far less
efficient, more flow-driven, and less HFT-arbitraged than EURUSD, and they are still liquid enough to trade.
The project brief says "EURUSD *and other liquid FX majors*," but the genuinely exploitable directional
predictability in FX is documented to be **larger down the liquidity curve** (cf. the AUD/USD@1h survivor in
Petrova et al., and the general efficiency–liquidity relationship). No report tests whether the *same 239-
feature stack* that caps at 0.52 on EURUSD reaches materially higher on an exotic.
**Why it matters:** it is the cheapest possible "new instrument" test (you may already have some of these
pairs; if not, Dukascopy has them free) and directly tests the report-12 thesis without leaving FX or taking on
crypto's idiosyncrasies. If 75% is reachable anywhere in liquid FX, exotics are the most likely place.
*Starting context:* CME EM FX liquidity, https://www.cmegroup.com/markets/fx/emerging-markets.html ; efficiency–
predictability link, https://arxiv.org/pdf/0712.1624 .

### 6. Strategic/adversarial decay and capacity — the "who is on the other side at 15m" question is never asked
**What:** All 16 reports treat predictability as a static statistical property. None asks the *game-theoretic*
question: **if a 15m EURUSD edge existed, who would have arbitraged it, and what residual is left for a non-
co-located retail-data participant?** The honest implication (foreshadowed by Carver's "speed limit" in report
10 and the post-2015 decay of pre-FOMC drift in reports 08/09) is that **any 15m edge in the most-arbitraged
pair on earth is structurally transient** — it must be re-mined continuously (favoring the *online-learning*
idea from report 11, which is correctly flagged but under-prioritized). The corollary, never stated: the
project's *own 17-version search* is itself an adversary inflating in-sample edge (the DSR/PBO point from
reports 06/10), so the realistic deliverable is a **continuously-adapted, capacity-tiny, regime-conditional**
edge, not a stable 75% model. This reframing should govern how every other idea is evaluated.
*Starting source:* online-learning lift (Jane Street 2024 winner, +0.008 vs +0.001–0.002 for features),
https://github.com/evgeniavolkova/kagglejanestreet ; alpha decay under costs, https://arxiv.org/pdf/2502.04284 .

### 7. Spot-vs-CME-futures clock/latency synchronization as a *data-quality* prerequisite (not just an alpha idea)
**What:** Reports 04/05/16 note CME 6E futures depth is buyable and that futures lead spot; report 07 frames
futures-lead-spot as ms-level. But none treats the **timestamp-alignment of your Dukascopy-grade spot feed to
exchange-clocked futures/rates** as a *correctness* issue that could be silently capping accuracy. Indicative
dealer quotes (Dukascopy) have jitter, last-look, and asynchronous stamps; if your 15m bar boundaries are even
slightly misaligned to true UTC vs the cross-asset features, you get *both* leakage (in backtest) and signal
destruction (live). This is a mundane but high-leverage audit nobody flagged as a gate.

### 8. Validation: CPCV/DSR are recommended but the *adversarial-validation* leak-detector is missing
**What:** Reports 06/10/11/13 correctly push purged+embargoed CV, CPCV, DSR, PBO. But none mentions
**adversarial validation** (train a classifier to distinguish train-rows from test-rows; if it succeeds, your
features have distribution shift / leakage / non-stationarity), which is the standard modern complement to CPCV
for *detecting* the exact "threshold collapses OOS" pathology report 13 describes. Cheap, and it would tell you
whether your 2024–25 collapse is leakage or genuine regime shift.
*Starting source:* backtest-overfitting comparison incl. synthetic controls,
https://www.sciencedirect.com/science/article/abs/pii/S0950705124011110 .

---

## Claims to treat with skepticism (beyond what the reports already flag)

The reports are mostly well-calibrated; these are residual over-optimisms or internal contradictions.

- **"Integrated multi-level OFI is the best-justified new feature" (report 02/05).** Over-sold. The *same
  reports* cite Cont–Cucuringu–Cont showing integrated OFI's lift is **contemporaneous (87% R²) and forward-
  predictive only "up to several minutes"**, and your true-tick OFI already dies by 5 min. Integrated OFI is a
  better *1–5m* feature, **not** a 15m one. Risk: spending real effort building L1–L2 integrated OFI and
  re-confirming the decay you already proved. Build it only if paired with the *aggregation* idea (#1 above),
  not as a standalone 15m feature.

- **The "metaorder-in-progress detector" (report 02, rated HIGH).** Appealing physics (order-flow long memory
  persists to ~3-min scale) but the same report concedes metaorder identification from public top-of-book data
  is "challenging" and you have **no depth and no trade tape** — only indicative dealer quotes with sizes. The
  honest prior is that you cannot reliably detect institutional metaorders from Dukascopy-grade L1. Treat as
  research-grade/low-probability, not HIGH.

- **Pre-FOMC drift as a usable directional signal (reports 08/09).** Reports 08 and 09 both lean on it, but the
  *same reports* document it **largely vanished after 2015** and "appears in equities, not Treasuries/MM
  futures." For an FX project whose test set is **2024–25**, a signal that decayed a decade earlier is a poor
  bet. The pre-FOMC drift should be **down-weighted to near-zero** for the 2024–26 evaluation window; the
  general "signed-surprise post-news drift" (which does not rely on the pre-drift anomaly) is the survivable
  part.

- **RavenPack/news-sentiment IRs ~0.5–0.8 (reports 08/16) as evidence sentiment "works."** Vendor self-research
  on its own data, at **4–5 day** holding — three to four *hundred* times your horizon. It is essentially zero
  evidence for a 15m edge and should not be cited as encouraging for this project. Report 08 mostly says this,
  but report 16 lets the "70%+ of best funds use it" marketing line stand under-challenged.

- **"Re-point the pipeline at crypto" as HIGH (report 12).** The strongest single recommendation in report 12,
  but it is a **change of project, not a solution to the FX problem**, and report 12's own sources note crypto
  LOB imbalance *also* decays fast with horizon. Crypto's documented intraday inefficiency is real but the
  effect sizes are *small and time-varying* — no source claims 75%. It is a reasonable pivot, but it should be
  labeled a *scope change*, and the *within-FX* exotic-pair test (#5) is the cheaper, more honest first probe of
  the same "less-efficient instrument" thesis.

- **SelectiveNet / SAE-denoising / signature features (reports 04/14) as plausible lift sources.** All three are
  i.i.d.-benchmark or error-metric results, not directional-sign-on-nonstationary-FX results. The reports flag
  this, but the *priority rankings* still place them MED/MED-HIGH. Given your finding that "architecture is not
  the bottleneck, signal is" (report 11, strongly evidenced across every competition), these representation-
  learning ideas deserve **LOW** priority until a new signal source exists for them to represent.

- **The 0.632 @ 0.2% coverage result itself (cross-cutting).** Reports 06/13 are right that this is a wide-CI,
  selection-biased point estimate at tiny n. The skepticism should be sharper: with **17+ versions tried**, the
  *effective number of trials* makes 0.632 quite possibly indistinguishable from 0.50 after deflation. Before
  any new vector is funded, this number should be **re-derived under CPCV + DSR**, because several reports
  silently build their "extend the selective frontier" recommendations on top of it as if it were solid.

- **Options/gamma "expiry magnet" selective sub-model (report 15, rated HIGH-within-vector).** The mechanism is
  sound but report 15 itself notes it requires ≥$0.5–1bn notional at strike, spot within 30–50 pips, *quiet
  tape*, dissolves on news — i.e. a handful of bars per week at most, reconstructed from messy free DTCC/CME
  data. The coverage is so thin and the data so noisy that the realistic lift to a 15m program is negligible;
  "HIGH within this vector" risks being read as "HIGH overall."

---

## Recommended additional research before experimenting

Ordered to maximize information per unit effort, and chosen to *falsify cheaply* before any build.

1. **Settle the foundational question first: is the 3-second edge compoundable to 15m?** Before any new feature,
   do the back-of-envelope and then the simulation: model your 3s signal as a noisy predictor with measured
   decay and a realistic per-trade cost (from your own tick bid/ask), and compute the SNR / hit-rate of the
   **sign of the cost-aware optimal position held over 15m** (Gârleanu–Pedersen aim-portfolio with your decay
   rate). If the integrated fast edge clears break-even at 15m, *that is the project* and most of the 16
   reports' feature ideas become irrelevant. If it does not, you have a rigorous, defensible reason the 15m
   goal is unreachable from microstructure. This single experiment dominates the others in expected value.

2. **Run the cheap target/instrument falsifiers in parallel (one afternoon each, existing or free data):**
   (a) **Cross-sectional rank target** across the 7 pairs (top-minus-bottom 15m spread) — does removing the USD
   factor raise residual predictability? (b) **Touch-before-touch (±k) target** instead of end-sign — is it
   more predictable? (c) **Exotic pair** (USDMXN/USDZAR via free Dukascopy) — does the same stack beat 0.52
   there? Each is a clean yes/no that could redirect the entire project; none requires new modeling.

3. **Re-validate the 0.632 result under CPCV + DSR + adversarial validation** *before* building anything on top
   of it. Treat it as the load-bearing number it has become. If it deflates to ~0.50, the "extend the selective
   frontier" program (the implicit thesis of half the reports) needs rethinking.

4. **Build the two free, mechanistic, *directional* overlays as separate sub-models** and measure in-window
   accuracy only: **Breedon–Ranaldo local-hours drift** and **signed-surprise post-news drift** (NOT the decayed
   pre-FOMC drift). These are the two most-credible non-TA directional signals in the whole corpus that you have
   not isolated, both keyed to mechanical flow.

5. **Audit feed synchronization as a correctness gate** (spot bar boundaries vs true UTC vs any cross-asset/CME
   feature; last-look/jitter in Dukascopy quotes). Do this before trusting *any* cross-asset experiment, since
   misalignment both fakes backtest edge and destroys live edge.

6. **Adopt online/walk-forward weight updates as a default, not an experiment** (report 11's best-evidenced
   single lift). Given the adversarial-decay argument (#6 above), a continuously re-fit model is the correct
   *baseline*, and every feature should be evaluated against the online baseline, not a static one.

7. **Deprioritize, explicitly:** more TA/representation learning (SAE, signatures, SelectiveNet, transformers),
   integrated-OFI-as-a-15m-feature, pre-FOMC drift, options-gamma magnets, COT, generic news sentiment. The
   evidence across all 16 reports is consistent that these are either architecture-not-signal, wrong-horizon, or
   decayed. Spending here is negative-EV until items 1–3 change the picture.

**Bottom line:** the 16 reports exhaustively answer "what *features* might help" and correctly conclude "almost
none, at 15m." The three questions they leave open — **how to use the fast edge you have (aggregation/execution
theory), what target to predict (rank / touch / signed-path), and what instrument to predict (exotic FX, or
accept crypto as a pivot)** — are where the remaining probability mass of reaching anything near 75% actually
lives, and all three are testable cheaply with data already in hand.

---

## Sources (this critique's additions)

- Gârleanu & Pedersen, "Dynamic Trading with Predictable Returns and Transaction Costs" (J. Finance 2013) — https://nbgarleanu.github.io/DynTrad.pdf — optimal multi-alpha aggregation by decay speed; the framework for compounding your 3s edge.
- "On the Effect of Alpha Decay and Transaction Costs on the Multi-period Optimal Trading Strategy" (2025) — https://arxiv.org/pdf/2502.04284 — alpha-decay + cost multi-period MDP.
- "Optimal trading without optimal control" (2020) — https://arxiv.org/pdf/2012.12945 — microstructure-alpha → near-optimal trading.
- Breedon & Ranaldo, "Intraday Patterns in FX Returns and Order Flow" (SNB WP 2011-04 / QMUL 694) — https://www.snb.ch/public/asset/en/www-snb-ch/publications/research/working-papers/2011/working_paper_2011_04/publications0_en/working_paper_2011_04.n.pdf — local-hours depreciation, flow-driven, directional.
- "Enhancing Cross-Sectional Currency Strategies by Context-Aware Learning-to-Rank" (arXiv:2105.10019) — https://arxiv.org/pdf/2105.10019 — learning-to-rank target across currencies.
- "A Non-parametric Test and Predictive Model for Signed Path Dependence" (Comp. Economics 2020) — https://link.springer.com/article/10.1007/s10614-019-09934-7 — signed-path-dependence target/predictor.
- CME Group — Emerging Markets FX — https://www.cmegroup.com/markets/fx/emerging-markets.html — liquid-but-less-efficient FX instruments.
- "Hurst exponent and prediction based on weak-form EMH" (arXiv:0712.1624) — https://arxiv.org/pdf/0712.1624 — efficiency–predictability relationship across markets.
- "Backtest Overfitting in the Machine Learning Era … Synthetic Controlled Environment" (KBS 2024) — https://www.sciencedirect.com/science/article/abs/pii/S0950705124011110 — OOS-method comparison; motivates adversarial validation.
- Volkova, Jane Street 2024 solution — https://github.com/evgeniavolkova/kagglejanestreet — online learning as the dominant lift; baseline argument.
