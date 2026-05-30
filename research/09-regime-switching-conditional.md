# Regime-Switching, HMM & Conditional Predictability: When Is 15m FX Direction Predictable?

Research vector 09 for binary-algo. Goal: lift 15-minute EURUSD directional accuracy toward 75% OOS, with emphasis on **orthogonal, regime-conditional** signals and methods not yet exhausted. The central thesis of this vector: **FX directional predictability is not a constant — it is conditional and concentrated in identifiable states.** Our flat ~0.52 AUC is an *average over regimes*; the right move is to find the minority of bars where edge is real and abstain on the rest. This dovetails directly with our existing best result (0.632 OOS @ 0.2% coverage via selective prediction) — regime conditioning is the principled way to *grow that coverage* without collapsing accuracy.

---

## TL;DR (most actionable for our 15m FX goal)

- **Reframe the target as conditional, not unconditional.** Decades of FX literature (Engel-Hamilton, Sarno-Valente, Marcucci) converge on one robust fact: a single model cannot beat a random walk on average, but *regime-conditional* models predict the **direction** (sign) materially better than the mean. The literature explicitly notes Markov-switching helps **direction-of-change** even when it fails on MSE. This is exactly the metric we care about. **[High]**

- **The single most credible orthogonal edge in this vector is the pre-FOMC / pre-scheduled-announcement drift.** It is large, persistent, OOS-robust, *directional*, and predictable from ex-ante volatility/uncertainty. NY Fed staff report: pre-FOMC drift = >80% of the entire equity premium over ~17 years; it produces a positive EUR/USD futures return (USD depreciation) into the announcement. It is a *time-deterministic regime* you can flag with zero lookahead. We tried a generic "event-window vol seasonality" proxy and found it redundant with time-of-day — but **the drift is a directional sign signal, not a vol signal**, which is a different thing we have NOT exploited. **[High]**

- **Use HMM/regime state as a GATE (meta-filter), not as a direction predictor.** The credible practitioner + academic consensus (RobotWealth, QuantInsti, Macrosynergy, hybrid HMM-SVM papers) is that HMMs are weak at *picking direction* but useful at *deciding when to engage* a direction model. Implement: Gaussian/MS-GARCH HMM on returns+RV → label each 15m bar's latent state → train/apply our existing LightGBM **separately per state** and only bet in states where validation accuracy clears threshold. This is regime-conditioned selective prediction. **[High]**

- **Condition on a volatility regime, because the *direction* of autocorrelation flips with vol.** Practitioner + RV-regime evidence: low-vol regimes show shallow intraday reversals and breakout follow-through (momentum), high-vol regimes show sharp violent reversals (mean-reversion). Our flat lag-1 autocorr of -0.03 is an *average that cancels*; split by RV quantile and the conditional autocorr is likely non-zero with opposite sign in the tails. This is a concrete, cheap test we have not run as a *sign-conditioning* experiment. **[High]**

- **Input-Output HMM (IO-HMM) is the right architecture for "predictability varies with covariates."** The Fischer/Krauss-style intraday IO-HMM paper makes the transition matrix a function of side information (realized-vol ratio, intraday seasonality via splines) and reports Sharpe ~1.9 with a 3-state model and *no time-lag at change points*. The transition-matrix-conditioning idea is directly portable to our 15m bars. Credible method, but reported Sharpe needs adversarial re-validation (single-asset, in-sample spline fitting risk). **[Med-High]**

- **Session conditioning is real but mostly a vol/liquidity effect — exploit the London-NY overlap as a *higher-edge regime*, not as a direction signal by itself.** The overlap concentrates >70% of volume and is where lasting intraday trends form. We already have time-of-day features, so the *new* move is to fit a separate model / lower the abstention threshold *during the overlap* rather than adding more session dummies. **[Med]**

- **Change-point detection (BOCPD, CUSUM, PELT) is best used defensively: detect regime breaks to TRIGGER abstention/retraining, not to predict direction.** Online Bayesian change-point detection has credible use for order-flow/impact regimes; for us its value is killing the bet in the bars immediately after a structural break (where stale features mislead). **[Med]**

- **Hurst exponent as a trend/mean-revert meta-filter is theoretically sound but statistically fragile at 15m.** Estimation literature: consistent H estimation needs interval→∞ or noise→0; intraday noise ratio is high (worse near close). Treat H as a weak, slow regime indicator at best; do NOT trust per-bar H. The "only trade trend when H>0.55" rules are practitioner folklore with real overfit risk. **[Low-Med]**

---

## Key findings (each with inline citation)

### 1. The canonical result: regime-switching helps DIRECTION even when it fails on MSE

The foundational FX regime-switching literature (Engel & Hamilton 1990; Engel 1994 "Can the Markov switching model forecast exchange rates?", J. Int. Economics) found Markov-switching models **do not** beat a random walk on mean-squared-error, **but** there is evidence they are **superior at predicting the direction of change** of the exchange rate ([Engel 1994, ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/0022199694900620)). A more recent extension ("Markov switching in exchange rate models: will more regimes help?", Empirical Economics 2020) finds adding regimes improves forecasts over 1–2 regime models but still cannot beat a random walk on level/MSE ([Springer](https://link.springer.com/article/10.1007/s00181-019-01623-6)). The pattern is consistent across 30 years: **the edge of regime models is in the sign, which is precisely our objective.** This is the strongest theoretical justification for our entire approach and reframes our "0.52 AUC ≈ coin flip" result as an artifact of averaging over regimes.

Sarno & Valente (and Abhyankar-Sarno-Valente, SSRN 549142) show the *set of predictors that work changes over time* — implying frequent coefficient shifts and that any static model is mis-specified ([SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=549142); [RePEc Sarno-Valente time-varying cointegration](https://ideas.repec.org/p/iek/wpaper/1302.html)). They also document that the **economic value** of predictability (utility/Sharpe to a trader) can far exceed the statistical value (MSE) — a direct argument that our metric (directional hit-rate / selective accuracy) is the right one and that MSE-based dismissals of FX predictability are too pessimistic.

### 2. Pre-scheduled-announcement drift — the most credible orthogonal directional signal in this vector

- **Equity benchmark (the cleanest evidence):** Lucca & Moench, "The Pre-FOMC Announcement Drift" (NY Fed Staff Report 512; J. Finance 2015): excess returns in the **24 hours before** scheduled FOMC announcements account for **>80% of the equity premium** over ~17 years ([NY Fed SR512 PDF](https://www.newyorkfed.org/medialibrary/media/research/staff_reports/sr512.pdf)). This is a *time-deterministic* effect — the announcement calendar is known years ahead, so flagging the regime has zero lookahead.
- **FX specifically:** Search-surfaced evidence that the pre-FOMC drift produces a **positive return on EUR/USD futures (USD depreciation, EUR appreciation)** into the announcement, and that the strength of the drift is **predictable** from stock-market volatility / monetary-policy uncertainty ([pre-FOMC drift summary, Tandfonline 2024](https://www.tandfonline.com/doi/full/10.1080/00036846.2024.2322573)).
- **Post-announcement reversal (also tradeable, opposite sign):** Lee & Wang, "Jumps and Post-FOMC Announcement Returns in Currency Markets" (SSRN 4386170): post-FOMC currency returns are significantly low and **reverse ~65% of the pre-FOMC drift**, mostly 12–24h after; option-implied skewness/kurtosis measured *before* the meeting **robustly predict post-FOMC returns in- AND out-of-sample** ([SSRN 4386170](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4386170)).
- **Mechanism:** Macroeconomic-news jump-diffusion models for FX (yen/pound/mark futures) show conditional-on-news diffusion+jump models dominate non-conditional ones, and **scheduled-announcement risk is directional in nature** ([Computational Economics, Springer](https://link.springer.com/article/10.1007/BF01299458)).

**Why this is NOT what we already ruled out:** our calendar work was an *event-window volatility-seasonality* proxy, which we (correctly) found redundant with time-of-day. The pre/post-FOMC drift is a **directional sign edge conditioned on a specific scheduled-event regime**, predictable from ex-ante implied-vol moments — an entirely different signal class. There are only ~8 FOMC meetings/year, plus ECB, NFP, CPI — so the coverage is small, but small high-accuracy coverage is exactly our selective-prediction sweet spot.

### 3. HMM regimes work as a GATE, not a direction oracle

- Macrosynergy and RobotWealth (credible practitioner quant sources) both frame regime indicators as **meta-filters**: decide *when to engage* the primary strategy, not the direction ([Macrosynergy Hurst](https://macrosynergy.com/research/detecting-trends-and-mean-reversion-with-the-hurst-exponent/); [RobotWealth Hurst pt2](https://robotwealth.com/demystifying-the-hurst-exponent-part-2/)).
- Hybrid academic models confirm HMM **embeddings as features** for a discriminative classifier beat single-stage models: "Generative-Discriminative ML for High-Frequency Financial Regime Classification" (Methodology & Computing in Applied Probability 2025) — HMM-generated feature embeddings feed an SVM/MKL and **outperform logistic, feed-forward NN, and single-kernel SVM** on trade-direction classification ([Springer](https://link.springer.com/article/10.1007/s11009-025-10148-8)).
- "A Hybrid Learning Approach to Detecting Regime Switches in Financial Markets" (arXiv 2108.05801) and the QuantInsti HMM+Random-Forest tutorial both implement the **regime-then-conditional-model** pattern we should adopt ([arXiv 2108.05801](https://arxiv.org/pdf/2108.05801); [QuantInsti](https://blog.quantinsti.com/regime-adaptive-trading-python/)).
- "Improving Portfolio Performance Using a Novel Method for Predicting Financial Regimes" (arXiv 2310.04536) and "Multi-Period Portfolio Optimisation Using a Regime-Switching Predictive Framework" (arXiv 2308.09263): random-forest regime classifiers achieve high regime-classification scores; gains concentrate in **high-volatility periods** ([arXiv 2310.04536](https://arxiv.org/pdf/2310.04536); [arXiv 2308.09263](https://arxiv.org/pdf/2308.09263)).

### 4. Volatility regime flips the SIGN of short-horizon autocorrelation

Practitioner + RV-regime evidence (Volatility Box; RV-regime trading notes): **low-vol regimes → shallow reversals, breakouts follow through (momentum); high-vol regimes → sharp violent reversals (mean-reversion)** ([Volatility Box regimes](https://volatilitybox.com/research/volatility-regimes-explained/)). This is a direct, testable explanation for why our **unconditional 5m lag-1 autocorr ≈ -0.03 is near zero**: it is a cancellation of a positive (momentum) component in calm regimes and a negative (reversal) component in turbulent regimes. MS-GARCH literature (Marcucci 2005, "Forecasting Stock Market Volatility with Regime-Switching GARCH"; Ardia et al. MSGARCH) confirms two-regime (low/high vol) models **outperform single-regime GARCH at short horizons** ([Marcucci PDF](https://www.greta.it/old/jae/poster/10_2_Marcucci.pdf)). The actionable consequence is not better vol forecasting per se but using the **vol-regime label as the conditioning variable for a sign model**.

### 5. Input-Output HMM with side information (the most directly portable architecture)

"Hidden Markov Models Applied To Intraday Momentum Trading With Side Information" (arXiv 2006.08307): a latent **momentum state** generates noisy returns in a state-space form with **no time-lag at change points** (signal sign flips exactly at the regime change, unlike lagging MAs). Side information — a **ratio of realized volatilities** and **intraday seasonality**, fit with **splines** — is fed into an **Input-Output HMM transition matrix** so transition probabilities depend on covariates. A **3-component HMM strategy reported Sharpe ~1.9** and fast regime identification ([arXiv 2006.08307](https://arxiv.org/abs/2006.08307)). DS³M ("Deep Switching State Space Model", arXiv 2106.02329) and "Deep State Space RNNs" (arXiv 2407.15236) generalize this with neural emissions + covariate-driven transitions ([DS³M](https://arxiv.org/pdf/2106.02329); [Deep SSRNN](https://arxiv.org/html/2407.15236v1)).

### 6. Change-point detection — defensive use

Online Bayesian change-point detection (BOCPD) has credible application to **order-flow and market-impact regimes** (Quantitative Finance 2024 / arXiv 2307.02375), and score-driven BOCPD handles temporal correlation + time-varying params within regimes ([arXiv 2307.02375](https://arxiv.org/abs/2307.02375); [arXiv 2407.16376](https://arxiv.org/pdf/2407.16376)). PELT is used for offline structural-break detection in financial series ([ACM PELT paper](https://dl.acm.org/doi/pdf/10.1145/3773365.3773532)). For our purpose the value is **detecting a break → abstaining / triggering retrain**, because our causal TA features are momentarily stale and misleading right after a structural shift.

### 7. Session conditioning

London-NY overlap concentrates **up to ~70% volume increase / >70% of daily volume**, and is when **lasting intraday trends form** (multiple practitioner sources). This is a vol/liquidity regime, not a free direction signal. Since we already encode time-of-day, the marginal move is **regime-conditional model selection / threshold lowering during the overlap**, not more session dummies.

### 8. Hurst exponent — fragile at our horizon

H<0.5 mean-reverting, =0.5 random walk, >0.5 trending. BUT: consistent estimation of H from noisy data requires observation interval→∞ or noise→0 ([arXiv 2205.11092](https://arxiv.org/pdf/2205.11092)), the intraday noise ratio is high and worsens near close, and H is **non-stationary** (past H does not guarantee future H). The "trade trend if H>0.55 / mean-revert if H<0.45 / stand aside in [0.45,0.55]" rule is practitioner folklore ([QuantNeuralEdge](https://quantneuraledge.com/blog/hurst-exponent-trending-ranging-markets); [QuantifiedStrategies](https://www.quantifiedstrategies.com/hurst-exponent/)). Use only as a slow, daily-scale regime hint.

---

## Concrete techniques / features / architectures to try

1. **Regime-gated selective prediction (highest priority, smallest lift-to-effort).**
   - Fit a 2–3 state HMM (Gaussian or MS-GARCH emissions) on a low-dim feature set: 15m return, realized vol (sum of 10s squared returns over the bar), and signed-volume OFI. Use `hmmlearn` / `MSGARCH` (R) / `pomegranate`.
   - Decode the *causal* (filtered, not smoothed — smoothing leaks future) latent state for each 15m bar.
   - Train our existing LightGBM **once per state** (or add state as a categorical feature + interaction terms). On val, compute per-state accuracy; **only bet in states clearing a threshold (e.g. >0.55)**. Report accuracy vs coverage curve and compare to our 0.632@0.2% baseline. *Falsifier to pre-commit: if per-state val accuracy is flat ~0.52 across all states, regime gating adds nothing.*

2. **Volatility-regime sign-conditioning experiment (cheap, do this first as a diagnostic).**
   - Bucket every 15m bar by trailing RV quantile (e.g. terciles). Compute lag-1 autocorr of 15m returns *within each bucket*. If the bottom and top RV terciles show non-zero autocorr of **opposite sign**, that confirms the cancellation hypothesis and immediately yields a conditional momentum/reversal sign rule. This directly tests why our unconditional -0.03 is uninformative.

3. **Scheduled-event directional overlay (highest-credibility orthogonal signal).**
   - Build a clean *causal* event calendar (FOMC, ECB, NFP, US CPI) with exact release timestamps.
   - Define pre-window (e.g. T-24h to T-5min) and post-window regimes. Test the **directional drift** (sign of EURUSD return) in each window, conditioned on ex-ante implied vol / MOVE / VIX level and option-implied skew if obtainable. Validate strictly OOS per our split.
   - This is a *separate small high-accuracy book* — expect tiny coverage (a few dozen events/year × the bars around them) but potentially well above 75% in the right direction. Do NOT merge it into the generic model; keep it as a regime overlay so it does not get averaged away.

4. **Input-Output HMM / covariate-driven transition matrix.**
   - Port the 2006.08307 design: latent momentum state; transition matrix = softmax(linear/spline of covariates) where covariates = realized-vol ratio (short RV / long RV), session indicator, time-to-next-scheduled-event. Emission = Gaussian on 15m return. Decode filtered state; bet only in the high-momentum state. Compare to HMM-gated LightGBM.
   - Stretch: DS³M (arXiv 2106.02329) for neural emissions if linear emissions cap out.

5. **Change-point abstention layer.**
   - Run BOCPD (e.g. `bayesian_changepoint_detection` lib) or a CUSUM on 15m returns/RV. For N bars after a detected change-point, force-abstain (our features are stale). Measure whether removing post-break bars *raises* accuracy on the remaining bets.

6. **Session-conditional thresholding.**
   - Keep one model; lower the |p-0.5| abstention threshold during London-NY overlap (12:00–16:00 UTC) and raise it in thin Asia hours. Measure coverage/accuracy tradeoff vs a flat threshold.

7. **HMM-embedding-as-feature (per the generative-discriminative paper).**
   - Instead of hard state labels, feed the HMM **posterior state probabilities** (a soft regime vector) as extra features into LightGBM. This is strictly more information than a hard label and is what the HMM-SVM paper found beat baselines.

---

## Reported results & CREDIBILITY assessment

**Credible / reproducible (peer-reviewed, OOS, mechanism-grounded):**
- *Regime models help direction-of-change, not MSE* — Engel 1994 + 2020 replication. Robust, 30-year consistency, peer-reviewed. **Credible.** Caveat: edge is modest, not a free 75%.
- *Pre-FOMC drift* — NY Fed staff report + J. Finance publication; magnitude (>80% of equity premium) is extraordinary but heavily replicated. **Highly credible** for equities; FX extension (EURUSD futures positive drift) is credible but the effect is smaller in FX and partly arbitraged post-2015 ("disappearing pre-FOMC drift", [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC7525326/)) — *this decay is itself a leakage/regime-stability risk you must test on 2024-25.*
- *Post-FOMC reversal predicted by option-implied skew/kurtosis, OOS* — Lee & Wang explicitly report out-of-sample predictability. **Credible**, but requires FX options data we may not have.
- *MS-GARCH beats single GARCH at short horizons* — Marcucci/Ardia, peer-reviewed. **Credible**, but it's a *volatility* result; the directional lift is indirect (via sign-conditioning), not established directly.

**Promising but needs adversarial re-validation:**
- *IO-HMM Sharpe ~1.9* (arXiv 2006.08307) — single-paper, single-asset, **spline side-information fit in-sample is a classic overfit vector**; the no-time-lag claim is appealing but the headline Sharpe should be reproduced on our data before belief. **Medium.**
- *EXFormer "directional accuracy +8.5–22.8% over baselines"* (arXiv 2512.12727) and various transformer FX papers — **treat as hype-adjacent.** "% improvement over baseline" without absolute hit-rate, plus daily/low-frequency data and no transaction costs, are red flags. Improvement over a weak baseline ≠ 75% absolute. **Low credibility for our metric until absolute OOS hit-rate is shown.**
- *Generative-discriminative HMM-SVM beats NN/logistic* — peer-reviewed (2025) but reports relative classification improvement; absolute directional accuracy and cost-after performance unstated. **Medium.**

**Hype / high leakage risk (discount heavily):**
- Blog/marketing "HMM strategy Sharpe 1.9 / 90% accuracy" reproductions on QuantConnect/Medium — many use **smoothed (Viterbi/forward-backward) state decoding that peeks at future data**, the #1 leakage bug in HMM regime backtests. *Always use filtered (online) state probabilities.*
- Hurst "H>0.55 = trade trend" rules — folklore, no OOS validation, estimation unreliable at 15m. **Hype-adjacent.**

**Leakage checklist for everything above:** (1) HMM state must be **filtered, not smoothed**; (2) event calendar timestamps must be release-time, not revised/backfilled; (3) per-regime model thresholds must be set on val, frozen for test; (4) RV-regime buckets must use trailing-only windows; (5) the pre-FOMC drift's post-2015 decay means **fit on 2012-21, but the 2024-25 test is the real judge.**

---

## Data sources needed

| Need | Where | Free/Paid | Fidelity |
|---|---|---|---|
| 15m OHLCV + 10s bars + ticks (have) | in-repo | — | already have; sufficient for HMM/RV/session |
| Causal scheduled-event calendar (FOMC, ECB, NFP, CPI) with exact release timestamps | FRED (FOMC dates), ForexFactory/Econoday scrape, central-bank sites; **investing.com calendar** | Free (scrape) / paid (Econoday) | Must capture *release time to the minute* and avoid revised timestamps — critical for no-lookahead |
| FX option-implied vol / risk-reversal / skew (for pre-FOMC conditioning & post-FOMC reversal) | CME (USD pairs options), Bloomberg/Refinitiv, or DTCC; **EUR/USD 1m ATM vol + 25Δ RR** | Mostly paid; some CME settlement free | Needed only for the option-implied-moment predictors (Lee-Wang). Optional — drift works without it, just weaker |
| MOVE / VIX / monetary-policy-uncertainty index (drift strength predictor) | CBOE (VIX free daily), policyuncertainty.com (EPU free) | Free | Daily granularity fine for conditioning |
| `hmmlearn`, `MSGARCH` (R), `pomegranate`, `bayesian_changepoint_detection`, `ruptures` (PELT/CUSUM) | PyPI/CRAN/GitHub | Free | Standard libs |

No new market data is strictly required to start: items 1–3 (HMM gating, vol-regime sign test, and a scraped event calendar) cover the three highest-priority experiments using data we already have plus a free calendar scrape.

---

## Relevance & priority for OUR project

**How this interacts with what we've ruled out:** Our prior work added features and bigger models to a *single unconditional* predictor and capped at 0.52 AUC. This vector says the ceiling is an averaging artifact — the same features may carry real sign-edge *inside specific regimes*. It also explains our one success (0.632@0.2% selective): selective prediction is implicitly finding a high-edge regime; regime conditioning makes that explicit and lets us grow coverage. Our "calendar event = vol seasonality, redundant" finding does NOT kill the pre-FOMC **directional drift** — that's a different (sign) signal we never tested.

**Ranked ideas:**

- **High:**
  1. RV-regime sign-conditioning diagnostic (idea #2) — one afternoon, directly tests the cancellation hypothesis behind our -0.03 autocorr.
  2. HMM-gated / HMM-soft-probability-as-feature selective prediction (ideas #1, #7) — principled extension of our best result; reuses existing LightGBM.
  3. Pre/post scheduled-event directional overlay (idea #3) — highest-credibility orthogonal signal; small coverage but potentially >75% directional in-window; needs only a free calendar scrape to start.

- **Medium:**
  4. Input-Output HMM with covariate-driven transitions (idea #4) — strong architecture match; build after #1 proves regimes matter.
  5. Session-conditional thresholding (idea #6) — cheap, modest.
  6. Change-point abstention layer (idea #5) — defensive; protects accuracy rather than creating edge.

- **Low:**
  7. Hurst-exponent meta-filter — statistically fragile at 15m; only as a slow daily regime hint, not a per-bar gate.
  8. Transformer FX direction papers (EXFormer etc.) — defer until/unless someone publishes absolute OOS hit-rate with costs; high overfit risk.

**Expected realistic outcome:** This vector is unlikely to yield a *flat* 75% over all bars (the literature is clear FX is ~RW on average). Its credible payoff is **higher selective accuracy at materially higher coverage than 0.2%** — e.g. pushing toward 0.60-0.65 at 5-15% coverage by betting only in identified regimes, plus a separate near-75% **event-overlay book** at tiny coverage. That is the honest, evidence-based target.

---

## Sources (annotated)

1. **Engel, "Can the Markov switching model forecast exchange rates?"** J. Int. Economics 1994 — [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/0022199694900620). *The seminal "helps direction, not MSE" result — core justification for regime-conditional sign models.*
2. **"Markov switching in exchange rate models: will more regimes help?"** Empirical Economics 2020 — [Springer](https://link.springer.com/article/10.1007/s00181-019-01623-6). *Modern replication: more regimes improve forecasts but still don't beat RW on level — calibrates expectations.*
3. **Abhyankar, Sarno & Valente, "Exchange Rates and Fundamentals: Evidence on the Economic Value of Predictability"** — [SSRN 549142](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=549142). *Economic value >> statistical value; predictor set shifts over time → static models mis-specified.*
4. **Lucca & Moench, "The Pre-FOMC Announcement Drift"** NY Fed SR512 / J. Finance 2015 — [PDF](https://www.newyorkfed.org/medialibrary/media/research/staff_reports/sr512.pdf). *>80% of equity premium in the 24h pre-FOMC window; the flagship scheduled-event drift.*
5. **"The disappearing pre-FOMC announcement drift"** — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC7525326/). *Critical caveat: the drift weakened post-publication — test stability on 2024-25.*
6. **Lee & Wang, "Jumps and Post-FOMC Announcement Returns in Currency Markets"** — [SSRN 4386170](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4386170). *FX-specific: post-FOMC reversal; option-implied skew/kurtosis predict it OOS — directional FX edge.*
7. **"Pre-FOMC drift: short-lived or long-lasting?"** Applied Economics 2024 — [Tandfonline](https://www.tandfonline.com/doi/full/10.1080/00036846.2024.2322573). *Drift predictable from vol/policy-uncertainty; EURUSD futures positive drift.*
8. **"Jump-diffusion processes in FX and the release of macroeconomic news"** Computational Economics — [Springer](https://link.springer.com/article/10.1007/BF01299458). *News-conditional jump-diffusion dominates; scheduled-announcement risk is directional.*
9. **"Hidden Markov Models Applied To Intraday Momentum Trading With Side Information"** arXiv 2006.08307 — [abs](https://arxiv.org/abs/2006.08307). *IO-HMM with RV-ratio + intraday-seasonality splines in the transition matrix; Sharpe ~1.9, no lag at change points — most portable architecture.*
10. **"Generative-Discriminative ML for High-Frequency Financial Regime Classification"** Methodology & Computing in Applied Probability 2025 — [Springer](https://link.springer.com/article/10.1007/s11009-025-10148-8). *HMM embeddings → SVM/MKL beat logistic/NN on trade direction — supports HMM-as-feature.*
11. **"A Hybrid Learning Approach to Detecting Regime Switches in Financial Markets"** arXiv 2108.05801 — [PDF](https://arxiv.org/pdf/2108.05801). *Practical regime-then-model pipeline.*
12. **Marcucci, "Forecasting Stock Market Volatility with Regime-Switching GARCH Models"** — [PDF](https://www.greta.it/old/jae/poster/10_2_Marcucci.pdf). *MS-GARCH beats single-regime GARCH at short horizons — basis for vol-regime conditioning.*
13. **"Improving Portfolio Performance Using a Novel Method for Predicting Financial Regimes"** arXiv 2310.04536 — [PDF](https://arxiv.org/pdf/2310.04536). *RF regime classifier; gains concentrate in high-vol regimes.*
14. **"Multi-Period Portfolio Optimisation Using a Regime-Switching Predictive Framework"** arXiv 2308.09263 — [PDF](https://arxiv.org/pdf/2308.09263). *Kalman+regime framework; regime-conditional return estimates.*
15. **"Online Learning of Order Flow and Market Impact with Bayesian Change-Point Detection"** Quant. Finance 2024 / arXiv 2307.02375 — [abs](https://arxiv.org/abs/2307.02375). *BOCPD for real-time regime breaks — defensive abstention trigger.*
16. **"Bayesian Autoregressive Online Change-Point Detection with Time-Varying Parameters"** arXiv 2407.16376 — [PDF](https://arxiv.org/pdf/2407.16376). *Score-driven BOCPD handling within-regime autocorrelation.*
17. **DS³M: "Deep Switching State Space Model for Nonlinear Time Series with Regime Switching"** arXiv 2106.02329 — [PDF](https://arxiv.org/pdf/2106.02329). *Neural generalization of IO-HMM (covariate-driven transitions + neural emissions).*
18. **"Estimation of the Hurst parameter from continuous noisy data"** arXiv 2205.11092 — [PDF](https://arxiv.org/pdf/2205.11092). *Why per-bar 15m Hurst is statistically unreliable — caps Hurst priority.*
19. **Macrosynergy, "Detecting trends and mean reversion with the Hurst exponent"** — [link](https://macrosynergy.com/research/detecting-trends-and-mean-reversion-with-the-hurst-exponent/). *Credible practitioner case for Hurst/regime as meta-filter, not direction signal.*
20. **RobotWealth, "Demystifying the Hurst Exponent (pt 2)"** — [link](https://robotwealth.com/demystifying-the-hurst-exponent-part-2/). *Skeptical, well-tested practitioner view — Hurst as regime gate.*
21. **QuantInsti, "Regime-Adaptive Trading with HMM and Random Forest"** — [link](https://blog.quantinsti.com/regime-adaptive-trading-python/). *Concrete Python implementation of the regime-gated-classifier pattern.*
22. **Volatility Box, "Volatility Regimes Explained"** — [link](https://volatilitybox.com/research/volatility-regimes-explained/). *Low-vol→breakout/momentum vs high-vol→sharp reversal — the sign-flip mechanism behind our -0.03 autocorr.*
