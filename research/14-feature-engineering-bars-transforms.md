# Advanced Feature Engineering for 15m FX Direction: Information-Driven Bars, Transforms, Entropy, Signatures, Pattern Mining

Research vector 14. Scope: alternative bar sampling (tick/volume/dollar/imbalance/run bars), Fourier/wavelet/EMD decomposition, entropy & complexity, Hurst/fractal, recurrence quantification, rough-path signatures, SAX/motif mining, tsfresh/Catch22 automated features, microstructural information measures (VPIN/Kyle lambda). Goal: which of these have *credible* evidence of lifting **intraday directional** accuracy, and which are hype/leakage.

Bottom line up front: almost every public "75–90% directional accuracy" claim in this space is one of three things — (a) **predicting magnitude/volatility, not sign**; (b) **leakage** from decomposing/normalizing the whole series before the train/test split (wavelet, EMD/EEMD, full-series scaling); or (c) **selective-coverage precision** dressed up as accuracy (meta-labeling). The honest, reproducible uses of these tools are as *representations and regime/volatility gates*, not as a new source of raw sign edge. That said, two ideas (rough-path **signatures** as a sequence representation, and **information-driven bars** as a resampling that changes what your existing models see) are genuinely under-explored at your horizon and worth a bounded experiment.

## TL;DR

- **Entropy predicts SIZE, not DIRECTION — provably.** Order-flow entropy is invariant under sign permutation, so it detects informed-trading *presence* without revealing its sign. Direct test on 38.5M SPY trades: conditioning on low entropy raises 5-min absolute returns 2.89x (t=12.4) while directional accuracy stays at **45.0% (p=0.12, ~0.50)**. Same will hold for Hurst, VPIN, RQA, realized-vol-from-wavelets. Use these as volatility/regime gates for *selective prediction*, never as a sign feature. (Singha 2025, arXiv:2512.15720)
- **The wavelet "~76% directional accuracy on forex" results are a leakage artifact.** Decomposing or denoising the *entire* series (including test rows) before splitting injects look-ahead bias; published EEMD/EMD finance results are explicitly documented as inflated by this. If you wavelet/EMD-transform, you MUST recompute the decomposition causally inside each rolling window — which destroys most of the headline accuracy. (Nature Sci Reports 2024 s41598-024-80018-9; ESWA 2017 EMD-bias paper)
- **Information-driven bars (dollar/volume/imbalance/run bars) are the single most defensible "new representation" idea you have not tried.** They re-clock the series by information arrival, giving better-behaved (closer to IID, lower serial correlation, more Gaussian) returns. This won't manufacture sign edge, but it can make your *existing* OFI/momentum/peer features more learnable and gives a natural event-trigger for *when* to predict. Credible (López de Prado, AFML 2018). High priority.
- **Rough-path signatures are the most credible "new transform."** They are a principled, near-universal feature map for *order-and-area* of multivariate paths (your 7-pair quote stream is exactly multivariate). Sig-DNN beats raw-LOB DNN on price-prediction error; sig pair-trading beats classic pair-trading on Sharpe. Evidence is for *return/Sharpe/error*, not a clean 75% sign rate — but as an orthogonal representation of your tick stream it's worth a bounded test. Use `signatory`/`iisignature` with the **Generalised Signature Method** canonical recipe (Morrill et al. 2021). Med-High.
- **tsfresh/Catch22 are commodity feature factories with real leakage traps.** tsfresh has a documented bug where it leaks the row `id`/index as a "feature." Catch22 (22 features) is the safer, faster, interpretable subset. These overlap heavily with your existing 239 TA features; expect marginal lift at best, and only if computed strictly causally per-window. Low-Med.
- **Meta-labeling + triple-barrier is about PRECISION on bets, not raw accuracy.** Reported 48%→55% is accuracy *conditional on a primary model already firing* — i.e. it's your existing "selective prediction" idea formalized. It can improve your 0.632@0.2%-coverage frontier but has not moved the unconditional 52% sign rate so far. Med (you've effectively done a version of this).
- **SAX/motif mining and RQA are descriptive/regime tools.** RQA cleanly detects regime transitions and crises (good gating signal) but the literature is explicit it does **not** predict direction. SAX-GA motif mining shows backtest "profitability" almost always without proper OOS / multiple-testing control — treat as hype until reproduced causally. Low.
- **Hurst is a regime label, not a forecast.** H>0.5 (persistent) vs H<0.5 (mean-reverting) is a useful *conditioning variable* to switch between momentum and fade logic, but estimating H needs thousands of points and is noisy intraday; on its own it does not deliver directional accuracy. Low-Med as a gate.

## Key findings

### 1. The current best level: entropy/complexity measures are sign-blind by construction
The most important paper for your project in this batch is Singha (2025), *"Hidden Order in Trades Predicts the Size of Price Moves"* (arXiv:2512.15720). Real-time order-flow entropy from a 15-state Markov transition matrix at second resolution, on **38.5M SPY trades over 36 days**, with 5-fold walk-forward validation and label-permutation placebo (z=14.4): conditioning on entropy below the 5th percentile **multiplies subsequent 5-min absolute return by 2.89x** (t=12.41, p<0.0001), while **directional accuracy stays at 45.0%, indistinguishable from ~0.50 (p=0.12)**. The author states the mechanism explicitly: *"entropy is invariant under sign permutation, detecting the presence of informed trading without revealing its direction."*

This generalizes. Any feature that is a function of the *distribution* of returns/order-flow but not their *signed order* (Shannon/approximate/sample/permutation entropy, realized vol, Hurst-via-R/S, VPIN, most RQA measures, wavelet *energy* per band) is structurally a volatility/regime variable. This directly matches your own mechanistic finding (true OFI decays to ~0.50 by 1 min; EURUSD ~97% USD factor). Conclusion: **the missing sign edge has not been found in entropy/complexity so far.** Look to them for *when to bet* (gating) and *position sizing*.

Corroborating: research surveys note "entropy values could [not] be used to predict the exact price direction... large entropy meant large risk" (ResearchGate review on approximate/sample entropy in finance), and the ultra-high-frequency entropy-randomness-test literature (arXiv:2312.16637) frames entropy as a *predictability/randomness* diagnostic, not a sign predictor.

### 2. Decomposition methods (wavelet, EMD/EEMD/CEEMDAN) — the leakage trap that fakes 70–90% accuracy
A USD/JPY 5-min wavelet+RNN+ARIMA study reports ~76% directional accuracy (arXiv:2008.06841), and many DWT/EEMD stock papers report similarly glamorous numbers. These are almost universally inflated by **look-ahead bias in the decomposition step**:

- *Research on information leakage in time series prediction based on EMD* (Yang/Li/Jiang, Scientific Reports 2024, s41598-024-80018-9): "after decomposition with division of training and test sets, information leakage from the test set ultimately shows high prediction accuracy that is illusionary." EMD/CEEMDAN sift over the *whole* series, so each IMF at time t embeds future values. Proposed fixes: sliding-window decomposition (SW-EMD), single-train/multi-decompose (STMP), multi-train/multi-decompose (MTMP).
- *Bias effect on predicting market trends with EMD* (ESWA 2017): promising EEMD finance results "were obtained by inadvertently adding look-ahead bias to the testing protocol via pre-processing the entire series with EMD."
- The same applies to **denoising via DWT thresholding** and to **whole-series normalization** ("min, max and standard deviation are known and fixed from the start" — a documented leakage source).

What survives causal evaluation: the **Maximal Overlap DWT (MODWT)** computed *inside* each rolling window, used to build multi-scale realized-volatility / energy features. But per finding #1, those are magnitude features. The honest verdict: wavelet/EMD are fine as causal multi-scale *volatility* descriptors, near-worthless as a "denoise then the trend becomes predictable" sign engine.

### 3. Information-driven bars (AFML) — re-clocking the data, not new information
From López de Prado's *Advances in Financial Machine Learning* (2018), surfaced via multiple practitioner writeups (Sefidian Academy; Gerard Martínez / TDS; RiskLab AI "The Need for Structure"; Alpaca "Alternative Bars"):
- **Tick / volume / dollar bars** sample a new bar every N ticks / N units volume / N dollars traded. Returns from these bars are closer to IID, have **lower serial correlation**, and are more nearly Gaussian than fixed-time bars — better statistical inputs for ML.
- **Imbalance bars (TIB/VIB/DIB)** and **run bars** sample when *signed* order-flow imbalance exceeds its expectation, i.e. they fire precisely when informed flow arrives. The pitch: "early detection of an imbalance change... anticipate a potential change of trend before reaching a new equilibrium."
- A peer-reviewed application — *Algorithmic crypto trading using information-driven bars, triple barrier labeling and deep learning* (Financial Innovation / Springer 2025, s40854-025-00866-w) — combines exactly these three ingredients into a trading pipeline. (It reports strategy-level returns, which carry the usual transaction-cost/OOS caveats; treat the bars+labeling *method* as the credible part, not the headline PnL.)

For you: imbalance/run bars are a way to turn your 10s-OFI (which "added ~0 lift" as a fixed-time feature) into an **event clock**. The hypothesis worth testing: features are weak at fixed 15-min spacing but informative when *sampled at imbalance-bar events*, and the right prediction target is "direction over the next K imbalance bars" rather than "next 15 minutes." This is genuinely orthogonal to everything in your V1–V17 ledger.

### 4. Rough-path signatures — the most principled "new transform"
Signature transform = the sequence of iterated integrals of a path; loosely the Fourier-analog for *order and area* of a multivariate stream, and a near-universal nonlinear feature map (any continuous function of the path is approximable by a linear functional of its signature). Key sources:
- **Generalised Signature Method** (Morrill, Fermanian, Kidger, Lyons 2021, arXiv:2006.00873): unifies the design choices (augmentations, windowing, rescaling, depth, log-signature) and gives a **canonical domain-agnostic recipe** validated on 26 datasets — the practical starting point.
- **Signatory** (Kidger & Lyons, ICLR 2021, arXiv:2001.00706; GitHub patrick-kidger/signatory): differentiable signature/log-signature on CPU/GPU, PyTorch-native, backprop-able → can be a trainable layer in a net. `iisignature` and `esig` are non-differentiable alternatives.
- **Signature Transform of LOB data for stock price prediction** (IEEE 2023, doc 10175387): Sig-DNN beats raw-LOB DNN on prediction error and efficiency (Sig-RF did not beat raw-RF — signatures help neural models more).
- **Signature Decomposition Method for Pair Trading** (Guo et al. 2025, arXiv:2505.05332): decomposes the signature into a path-interactivity indicator (segmented signature) and a **directional indicator (covariation of increments)**; on minute-level futures, beats classic pair trading on return, drawdown, Sharpe. Notably their *directional* component is the covariation of increments — a concrete, interpretable signature term to try as a feature.
- **Signature Trading** (Futter, Horvath, Wiese 2023, arXiv:2308.15135): path-dependent mean-variance with exogenous signals; relevant if you later want sizing/portfolio logic over the 7 pairs.

Caveat: signatures suffer the curse of dimensionality (depth-n on d channels → ~d^n terms); the convolutional-signature line (Digital Finance 2022, s42521-022-00049-7) and log-signatures mitigate. Evidence is for error/Sharpe, *not* a demonstrated 75% sign rate — but as a representation of your raw multi-pair quote+size stream it is the most defensible untried transform.

### 5. tsfresh / Catch22 — automated features with leakage footguns
- **tsfresh** (Christ et al., Neurocomputing 2018): ~794 features + FRESH hypothesis-test selection. **Documented leakage**: GitHub issue #162 — `index_mass_quantile` and friends can be "directly proportional to the id," so classification accuracy depended on *row order*. Any use must (a) drop index-derived features, (b) compute per-window causally, (c) select features on training folds only.
- **Catch22** (Lubba et al. 2019, arXiv:1901.10200): the 22 most discriminative `hctsa` features — autocorrelation (linear/nonlinear), successive differences, distribution/outliers, fluctuation scaling, entropy. Faster, interpretable, far less overfit-prone than tsfresh; aeon benchmarks show tsfresh+rotation-forest strong on generic TSC, but your domain already has 239 hand-built TA features that overlap heavily. Expect overlap, not orthogonality.

### 6. Microstructural information measures (VPIN, Kyle lambda)
VPIN (Easley, López de Prado, O'Hara) — order-flow toxicity over equal-*volume* buckets; Kyle's lambda — price impact per unit signed volume. Both are **volatility/liquidity/toxicity** measures (VPIN forecasts *volatility* ex-ante per ScienceDirect S1044028318302679), i.e. magnitude per finding #1. Use as gates/sizing. Kyle lambda regime (high vs low price-impact) is a plausible *conditioning* variable for whether microstructure signals survive longer than your measured 1-min decay.

### 7. Hurst / fractal & RQA — regime labels, not forecasts
- Hurst H: H∈(0.5,1) persistent/trending, H∈(0,0.5) anti-persistent/mean-reverting; D=2−H. Useful as a *switch* between momentum and fade models, but "a large amount of data (thousands of values) is needed for a reliable estimate" — noisy and slow intraday. Macrosynergy's practitioner study frames it as trend/mean-reversion *detection*, not point-forecasting.
- RQA (recurrence quantification): determinism, laminarity, longest diagonal, trapping time. Literature is consistent that RQA **detects** regime transitions/crises and "can't predict them" in the directional sense. Good as a regime/structural-break gate feeding selective prediction.

## Concrete techniques / features / architectures to try

Ranked roughly by expected value for your specific 15m-FX-direction goal.

1. **Imbalance/run-bar event clock + re-targeted labels (HIGH).**
   - Build dollar bars and tick-imbalance bars (TIB) from your sub-second ticks per pair: accumulate signed tick volume; emit a bar when |cumulative imbalance| exceeds an EWMA of its expected absolute value (AFML algorithm). Libraries: `mlfinlab`-style implementations (now `mlfinpy`), or roll your own (~50 lines).
   - Re-sample ALL your existing 239 features at imbalance-bar timestamps; relabel target as "sign of return over next K imbalance bars" with a **triple-barrier** label (vol-scaled up/down barrier + time barrier).
   - Test whether AUC at imbalance-bar events > AUC at fixed 15m. The win condition is event-conditioned predictability, not unconditional.

2. **Rough-path signature features over the 7-pair quote stream (MED-HIGH).**
   - Path = lead-lag-augmented, time-augmented stream of [mid_i, signed-size_i] for the 7 USD pairs over a trailing window (e.g. last N ticks or last few imbalance bars).
   - Use the **Generalised Signature Method** canonical config (Morrill 2021): time augmentation + lead-lag transform, log-signature depth 3 (start), per-window. Compute with `signatory` (differentiable, for a net) or `iisignature` (for LightGBM features).
   - Specifically include the **covariation-of-increments** signature terms (the Guo 2025 "directional indicator") and cross-pair area (Lévy area) terms — these encode signed lead-lag rotation between pairs, which is *not* in your current cross-pair lag features.
   - Feed signature vector to your existing LightGBM/GRU. Strictly causal: signature of trailing window only.

3. **Causal MODWT multi-scale volatility/energy features as GATES, not sign features (MED).**
   - Inside each rolling window compute MODWT (shift-invariant, no downsampling) band energies → multi-scale realized-vol regime. Use to gate selective prediction (bet only in scale-energy regimes where your model historically scored higher), and to size. Never normalize across the test set.

4. **Meta-labeling layer on your best primary model (MED — formalizes what you've done).**
   - Primary model emits side (your 52% sign model). Secondary classifier predicts P(primary is correct) using vol regime, entropy, VPIN, RQA-determinism, Hurst, signature features. Bet only when secondary P high. This is exactly your "selective prediction" reframed with López de Prado's machinery; it improves precision-on-bets / coverage curve (reported 48%→55% accuracy on *fired* signals); moving the unconditional rate remains open.

5. **Catch22 per-window, causally, as a compact orthogonality probe (LOW-MED).**
   - Cheap to run (`pycatch22`). Add the 22 features per trailing window, check SHAP/importance vs your TA set. If nothing rises above your existing features, drop it. Avoid full tsfresh (leakage + 794-dim multiple-testing).

6. **Hurst & RQA-determinism as 2–3 regime-conditioning columns (LOW-MED).**
   - Rolling H (R/S or DFA) and RQA determinism/laminarity as conditioning inputs to a regime-switching head, or as gates. Expect a small selective-prediction lift, no unconditional sign lift.

7. **SAX symbolization + motif mining (LOW, exploratory).**
   - SAX-encode imbalance-bar returns; mine motifs with a matrix-profile (STUMPY). Only pursue if a motif's *forward* return is tested OOS with multiple-testing correction (deflated Sharpe / White's reality check). Default assumption: in-sample mirage.

## Reported results & CREDIBILITY assessment

| Claim / method | Reported result | Credibility | Leakage / caveat |
|---|---|---|---|
| Order-flow entropy → magnitude (Singha 2512.15720) | 2.89x abs-return, **45% directional (chance)** | **High** — walk-forward + placebo; but 36 days, 1 instrument | Honest *negative* result on direction. Generalize the lesson. |
| Wavelet+RNN+ARIMA USD/JPY 5m (2008.06841) | ~76% directional | **Low** | Classic whole-series decomposition/denoise leakage |
| EMD/EEMD/CEEMDAN finance high accuracy | 70–90%+ | **Low** | Explicitly documented as look-ahead-biased (Nature 2024; ESWA 2017) |
| Information-driven bars (AFML) | Better IID/statistical props | **High (method)** | Resampling ≠ new sign info; PnL claims need cost/OOS scrutiny |
| Crypto info-bars+triple-barrier+DL (Springer 2025) | Strategy returns | **Med** | Bars+labeling solid; PnL has cost/overfit risk |
| Sig-DNN on LOB (IEEE 2023) | Beats raw-LOB DNN on error | **Med-High** | Error metric, not sign rate; Sig-RF did *not* beat |
| Signature pair-trading (Guo 2025) | Higher Sharpe vs classic | **Med** | Pair-trading (relative), futures, minute data |
| Meta-labeling + triple barrier | 48%→55% accuracy | **Med** | Accuracy *conditional on firing*; precision not raw sign |
| SAX-GA motif strategies | "profitable" vs B&H | **Low** | Rare proper OOS / multiple-testing control |
| RQA / Hurst | Regime/crisis detection | **Med (as gate)** | Literature: does NOT predict direction |
| tsfresh/Catch22 generic TSC | Strong on UCR/UEA | **Med** | tsfresh id-leakage bug (#162); overlaps your TA set |

General leakage checklist this vector forces you to honor: (1) any transform/normalization/decomposition computed per-window causally, never whole-series; (2) feature selection on training folds only; (3) purged + embargoed walk-forward CV (López de Prado) because overlapping triple-barrier labels autocorrelate; (4) deflated Sharpe / multiple-testing correction for any mined pattern; (5) realistic spread/slippage before believing any PnL.

## Data sources needed

- **You already have the critical inputs**: sub-second bid/ask quotes **with sizes** for 7 USD pairs + 10s OHLCV. This is exactly what information-driven bars and signature paths need. No new purchase required for the two High-priority ideas.
- **Imbalance/dollar bars**: derived from your ticks (signed via tick-rule or quote-rule using your bid/ask). Free, in-house.
- **Signatures**: `signatory` (PyTorch, GPU, differentiable), `iisignature` (NumPy), `esig` (avoid for high-dim per its known multidim issues). Free, pip.
- **Catch22**: `pycatch22`; **tsfresh**: `tsfresh` (use with care). Free, pip.
- **Matrix profile / motifs**: `stumpy`. Free.
- **RQA**: `pyrqa` (GPU) or `pyunicorn`. **Hurst**: `hurst`, `nolds`. **Wavelets**: `PyWavelets` (use MODWT/`modwt` causally). Free.
- **VPIN / Kyle lambda**: computed from your trades+quotes (need signed volume buckets). Free, in-house. Reference impl: GitHub yt-feng/VPIN.
- Optional **true depth-of-book** (L2/L3) FX from an ECN (paid: e.g. LMAX, Integral, Refinitiv) would strengthen VPIN/imbalance vs your tick-rule proxy — but your earlier work shows true OFI decays by 1 min, so paying for deeper book is **low ROI** at the 15m horizon.

## Relevance & priority for OUR project

Context check against your ledger: you've exhausted fixed-time TA (239 feats), cross-pair lead-lag, 10s-OFI proxy (~0 lift), tree/seq ensembles best so far ~0.52 AUC, stat-arb, daily context, exogenous peer feats, calendar proxy, and selective prediction (best 0.632 @ 0.2% coverage). Your mechanistic truth: real OFI edge dies by ~1 min; clean >75% so far observed at 3s; EURUSD ~97% USD factor.

How this vector interacts:
- **Most of this vector confirms, not contradicts, your current best level.** Entropy/Hurst/VPIN/RQA/wavelet-energy are all *magnitude* tools — they have not supplied the missing sign edge so far, and the papers that claim they can are leakage-driven. This is itself a valuable negative result: stop hunting sign edge in complexity measures.
- **Two genuinely untried, orthogonal levers**: (1) **information-driven bars** change the *clock* and the *label*, not just the features — the one thing you have not varied (everything you tried was fixed-time). (2) **Signatures** encode signed cross-pair order-and-area (Lévy area, covariation of increments) that your linear lead-lag features cannot represent — a real chance at orthogonal information from the same data.

Priority ranking:
- **HIGH**: Imbalance/run-bar resampling + triple-barrier relabeling (re-target what "15m direction" even means).
- **MED-HIGH**: Rough-path signature features (esp. cross-pair Lévy area + covariation-of-increments) into existing models.
- **MED**: Meta-labeling formalization of your selective-prediction frontier; causal MODWT vol-regime gates.
- **LOW-MED**: Catch22 orthogonality probe; Hurst/RQA regime-conditioning columns.
- **LOW**: tsfresh (leakage risk, redundant); SAX/motif mining (mirage-prone); wavelet/EMD *denoising-for-direction* (do not — leakage).

Hard rule for every item: per-window causal computation + purged/embargoed walk-forward + multiple-testing correction, or the result is fiction.

## Sources

- **Singha, M. (2025). "Hidden Order in Trades Predicts the Size of Price Moves." arXiv:2512.15720.** https://arxiv.org/abs/2512.15720 — The decisive evidence that order-flow entropy predicts magnitude (2.89x) but NOT direction (45%, chance); sign-permutation invariance explains why all complexity measures are sign-blind.
- **Yang, Li, Jiang (2024). "Research on information leakage in time series prediction based on EMD." Scientific Reports 14.** https://www.nature.com/articles/s41598-024-80018-9 — Proves EMD/CEEMDAN whole-series decomposition leaks future info and inflates accuracy; gives causal fixes (SW/STMP/MTMP).
- **de Oliveira et al. (2017). "Bias effect on predicting market trends with EMD." Expert Systems with Applications.** https://www.inf.ufpr.br/lesoliveira/download/ESWA2017.pdf — Documents EEMD finance results as look-ahead-biased; why "denoise then predict" is fake edge.
- **López de Prado, M. (2018). Advances in Financial Machine Learning** (info-driven bars, triple barrier, meta-labeling, purged CV). Summaries: https://www.sefidian.com/2021/06/12/introduction-to-advanced-candlesticks-in-finance-tick-bars-dollar-bars-volume-bars-and-imbalance-bars/ and https://www.risklab.ai/research/financial-data-science/financial_data_structures — Canonical reference for dollar/imbalance/run bars and the labeling/CV machinery.
- **Algorithmic crypto trading using information-driven bars, triple barrier labeling and deep learning (2025). Financial Innovation (Springer).** https://link.springer.com/article/10.1186/s40854-025-00866-w — Peer-reviewed end-to-end pipeline combining the three AFML ingredients.
- **Morrill, Fermanian, Kidger, Lyons (2021). "A Generalised Signature Method for Multivariate Time Series Feature Extraction." arXiv:2006.00873.** https://arxiv.org/abs/2006.00873 — Canonical, domain-agnostic recipe for applying signatures (augmentations, windowing, depth, log-sig); your practical starting point.
- **Kidger & Lyons (2021). "Signatory." ICLR. arXiv:2001.00706.** https://arxiv.org/abs/2001.00706 / https://github.com/patrick-kidger/signatory — Differentiable GPU signature/log-signature for PyTorch; use as a trainable layer or feature generator.
- **"A Signature Transform of LOB Data for Stock Price Prediction" (IEEE 2023).** https://ieeexplore.ieee.org/document/10175387/ — Sig-DNN beats raw-LOB DNN on error; evidence signatures help neural models on microstructure.
- **Guo, Jin, Kuang, Qian, Wang (2025). "Signature Decomposition Method Applying to Pair Trading." arXiv:2505.05332.** https://arxiv.org/abs/2505.05332 — Splits signature into path-interactivity + a DIRECTIONAL covariation-of-increments indicator; concrete signature terms to try.
- **Futter, Horvath, Wiese (2023). "Signature Trading." arXiv:2308.15135.** https://arxiv.org/abs/2308.15135 — Path-dependent mean-variance with exogenous signals; relevant for later sizing across 7 pairs.
- **Lubba et al. (2019). "catch22: CAnonical Time-series CHaracteristics." arXiv:1901.10200.** https://ui.adsabs.harvard.edu/abs/2019arXiv190110200L/abstract — 22 interpretable, low-overfit features; preferred over full tsfresh.
- **Christ et al. (2018). tsfresh. Neurocomputing** + leakage issue: https://github.com/blue-yonder/tsfresh/issues/162 — 794-feature factory; documented id/index leakage footgun.
- **Hudson & Thames — "Does Meta-Labeling Add to Signal Efficacy?"** https://hudsonthames.org/does-meta-labeling-add-to-signal-efficacy-triple-barrier-method/ — Meta-labeling raises precision-on-bets (~48%→55%), reframes selective prediction.
- **Easley, López de Prado, O'Hara — VPIN.** https://www.quantresearch.org/VPIN.pdf ; VPIN→volatility evidence: https://www.sciencedirect.com/science/article/abs/pii/S1044028318302679 — Order-flow toxicity as a volatility/liquidity (magnitude) gate.
- **Macrosynergy — "Detecting trends and mean reversion with the Hurst exponent."** https://macrosynergy.com/research/detecting-trends-and-mean-reversion-with-the-hurst-exponent/ — Hurst as a regime/switch variable, not a point forecaster.
- **Bastos & Caiado (2007/2011). Recurrence Quantification Analysis of financial series / crashes.** https://arxiv.org/pdf/1107.5420 — RQA detects regime transitions/crises but does not predict direction.
- **Price predictability at ultra-high frequency: Entropy-based randomness test. arXiv:2312.16637.** https://arxiv.org/pdf/2312.16637 — Entropy as a predictability/randomness diagnostic at UHF, reinforcing the magnitude-vs-direction split.
- **Henderson & Fulcher (2021). "An Empirical Evaluation of Time-Series Feature Sets." arXiv:2110.10914.** https://arxiv.org/pdf/2110.10914 — Benchmarks tsfresh vs Catch22 vs others; informs which feature factory to use.
