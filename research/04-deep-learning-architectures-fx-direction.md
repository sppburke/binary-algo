# Deep-Learning Architectures for Financial Time-Series & FX Direction Prediction

*Research vector 04 — binary-algo. Compiled 2026-05-30. Goal context: predict EURUSD up/down at a 15-minute horizon, target >75% strictly OOS. We have already exhausted GBM/Transformer/GRU tabular ensembles (cap ~0.52–0.527 AUC), 239 TA features, cross-pair lead-lag, 10s OFI, stat-arb, calendar proxy, and selective prediction (best 0.632 acc @ 0.2% coverage). True tick-level OBI is real but decays to coin-flip by 1 min and is gone by 5 min.*

---

## TL;DR

- **No credible, reproducible deep-learning result reaches 75% directional accuracy at a minutes-to-hours horizon on liquid FX/equities.** The honest ceiling for *daily* EUR/USD direction with ML/DL is ~**58.5%** (Castillo et al. 2024, arXiv:2409.04471); intraday is lower. Treat every ">70–90%" intraday claim as leakage/overfit until proven otherwise.
- **The architecture is almost never the bottleneck — the label and the horizon are.** "Are Transformers Effective for Time Series Forecasting?" (Zeng et al., AAAI 2023) showed a one-layer linear model (DLinear) beats Informer/Autoformer/FEDformer on most long-horizon benchmarks. On the *Exchange-Rate* dataset specifically, linear baselines are notoriously strong because daily FX is near a random walk. **Do not expect TFT/Informer/N-HiTS to rescue a signal that GBM cannot already see.**
- **DeepLOB-style CNN-LSTM works only where there is a live limit-order book and only at horizons of tens of events (sub-second to seconds).** Its predictability traces directly to *tick size / spread*, decays with horizon, and largely **evaporates after transaction costs** (Lucchese et al. 2024, "microstructural guide", arXiv:2403.09267; TLOB conclusion, Berti & Kasneci 2025, arXiv:2502.15757). This *agrees with your own mechanistic finding* (3s clean, gone by 5 min) and is **not a path to 15m**.
- **Time-series foundation models (Chronos, TimesFM, Moirai) are a dead end for FX direction zero-shot.** Rahimikia 2025 (arXiv:2511.18578) measured Chronos-large and TimesFM-500M on daily excess returns: **negative OOS R² (−1.4% to −2.8%) and directional accuracy ~49.5–51%** — i.e., a coin flip. They only help *after* pretraining from scratch on financial data.
- **The few legitimately useful DL ideas are about *labeling and representation*, not architecture:** triple-barrier / meta-labeling (López de Prado), self-supervised contrastive pretraining on unlabeled tick data, and multi-horizon/quantile heads that predict *path* statistics (vol, MFE/MAE) rather than a single 15m sign.
- **Where DL has a real, defensible edge vs your GBMs: raw multi-channel sequence input** (full tick/quote-size matrix fed to a CNN/attention front-end) instead of hand-aggregated 10s features. If any orthogonal 15m signal exists in quote sizes, a learned spatio-temporal encoder is more likely to find it than 239 causal TA features — but the base-rate prior says the lift is small.
- **Realistic deliverable from this vector:** not 75% raw accuracy, but a **better-calibrated selective predictor** (improve your 0.632@0.2% frontier) via meta-labeling + a sequence encoder + proper probability calibration. The 75% target is achievable only on a *conditional* subset, not unconditionally.

---

## Key findings (each with inline citation)

### 1. The realistic accuracy ceiling for FX/equity direction is ~52–59%, and it falls as horizon shortens
- Castillo et al., *Predicting Foreign Exchange EUR/USD direction using machine learning* (arXiv:2409.04471, MLMI 2024): best stacked ML model reaches **58.52% accuracy for one-day-ahead** EUR/USD direction (with 32.48% annualized return for 2022 — a single bullish-trend year, so the return is regime-lucky, not the accuracy). This is *daily*; the paper does not claim intraday. **58.5% daily is the credible bar.** [arXiv:2409.04471]
- López Gil et al., *An Evaluation of Deep Learning Models for Stock Market Trend Prediction* (arXiv:2408.12408): across TCN, N-BEATS, TFT, N-HiTS, TiDE, xLSTM-TS, **TFT's accuracy was "not better than random guessing"** on their setup. Best model (xLSTM-TS) reached F1 ~73% on **daily** S&P500/EWZ — but dropped to **63.9–69.4% on hourly**, and the daily numbers rely on **wavelet denoising** (a known lookahead-bias risk; see §4). Pattern: accuracy collapses as you go from daily → hourly → 15m. [arXiv:2408.12408]
- This is the single most important calibration for our project: **the literature's own honest numbers say 15m EUR/USD direction at 75% is not a normal result.** Anything claiming it should be assumed leaked.

### 2. Transformer architectures do NOT reliably beat simple baselines on financial series
- Zeng et al., *Are Transformers Effective for Time Series Forecasting?* (AAAI 2023, arXiv:2205.13504): one-layer **DLinear beats Informer/Autoformer/FEDformer/Pyraformer** on 9 datasets, often by 20–50% MSE. The self-attention permutation-invariance loses temporal order that a linear model keeps. **Directly relevant: the "Exchange-Rate" benchmark is where linear models dominate hardest**, because daily FX ≈ random walk. [arXiv:2205.13504]
- PatchTST (Nie et al., arXiv:2211.14730) later restored Transformer competitiveness (~21% MSE gain vs prior Transformers) via **patching + channel-independence** — but its wins are on *weather/traffic/electricity*, which have strong seasonality FX lacks. No evidence of a directional-accuracy edge on FX. [arXiv:2211.14730]
- Implication: switching from LightGBM to TFT/Informer/FEDformer is **very unlikely** to move your 0.527 AUC. These models win on *smooth, seasonal, high-SNR multivariate forecasting*, not low-SNR sign prediction.

### 3. DeepLOB & descendants are a microstructure story, not a 15m story
- Zhang, Zohren, Roberts, *DeepLOB* (IEEE TSP 2019, arXiv:1808.03668): CNN (spatial LOB structure) + LSTM (temporal), labels = **smoothed mid-price over horizon k = {10,20,50,100} events** (not minutes). Strong, stable OOS on FI-2010 and LSE; generalizes across instruments. But the horizon is **tens of order-book events ≈ sub-second to seconds**. [arXiv:1808.03668]
- TLOB (Berti & Kasneci 2025, arXiv:2502.15757) — current SOTA dual-attention Transformer on LOB. Critical honest findings from its own conclusion: (a) **NASDAQ stocks (TSLA, INTC) are "significantly more challenging" than FI-2010** (the classic benchmark is saturated/easy); (b) **accuracy decreases as horizon increases**; (c) when the trend threshold θ is set to **average spread (i.e., real transaction cost), profitability collapses** — authors state the methods are **"not sufficiently mature for practical deployment in live trading."** [arXiv:2502.15757]
- Lucchese, Magnani, et al., *Deep Limit Order Book Forecasting: a microstructural guide* (arXiv:2403.09267, Quantitative Finance 2025): the **"predictability rate" of DeepLOB is fully explained by tick-size/spread** — large-tick stocks are predictable, small-tick are not — and there is a large **"simulation-to-reality gap."** Predictability is a microstructure artifact at very short horizons, not a tradeable longer-horizon edge. [arXiv:2403.09267]
- **This is exactly your own finding restated by academia: OBI/LOB signal is real at seconds, gone by minutes. DeepLOB will not give you 15m. Do not re-implement it expecting a 15m lift.**

### 4. Many headline accuracies are inflated by leakage; the usual culprits
- **Whole-series normalization / denoising before the train-test split.** FI-2010 ships pre-normalized so the raw LOB can't be reconstructed; the standard practice of z-scoring on global statistics leaks future moments into training (noted in FI-2010 critiques and LOBFrame). Wavelet/DWT denoising of the *entire* series (common in "70–90%" stock papers, incl. the daily xLSTM-TS results above) is a **classic lookahead leak** — the smoothed value at time t uses t+1…t+k. [FI-2010 critiques; arXiv:2408.12408]
- **Overlapping labels** (smoothed mid-price / fixed-horizon labels on sliding windows) inflate apparent skill and break IID assumptions in CV — López de Prado's central warning. Purged/embargoed CV is required.
- **Class-balanced thresholds chosen with test knowledge**, and **non-chronological splits / random k-fold** on autocorrelated series.
- **Takeaway for us:** our pipeline (strict chronological 2012-21 / 22-23 / 24-25, 2026 held out, causal features) is *already* more rigorous than most papers reporting 75–90%. That is *why* we see 0.52 — we are not cheating. Reaching 75% honestly is genuinely hard.

### 5. Foundation models for time series do not help FX direction zero-shot
- Rahimikia, *Re(Visiting) Time Series Foundation Models in Finance* (arXiv:2511.18578, Nov 2025): on daily excess returns across 94 countries / 34 years, **Chronos-large: OOS R² = −1.37%, directional acc just above 51%; TimesFM-500M: R² = −2.80%, acc just below 50%.** Off-the-shelf TSFMs **underperform CatBoost/LightGBM**. Only models **pretrained from scratch on financial data** showed gains. [arXiv:2511.18578]
- Consistent with TimesFM/Chronos/Moirai being trained to minimize point-forecast MSE on smooth series; financial returns are near-martingale low-SNR, so zero-shot transfer is ~useless for sign. **Skip zero-shot TSFM for direction.** A *domain-pretrained* encoder (your data, self-supervised) is the only version worth considering.

### 6. The genuinely transferable ideas are about labels and representations
- **Triple-barrier + meta-labeling** (López de Prado, *Advances in Financial ML* 2018; mlfinlab): label by which of {profit, stop, time} barrier is hit first; train a *primary* model for side and a *secondary* meta-model for "act / don't act." This directly improves the *selective-prediction* frontier you already care about (0.632@0.2%). Crypto study (Tan et al., *Financial Innovation* 2025, s40854-025-00866-w) found event-based bars + triple-barrier + meta-labeling improved DL strategies; **ResNet-LSTM beat the Transformers** there, and Autoformer/FEDformer adapted poorly. [s40854-025-00866-w]
- **Self-supervised / contrastive pretraining on unlabeled ticks** (TF-C, Zhang et al. NeurIPS 2022; Contrastive Asset Embeddings, arXiv:2407.18645): learn representations from your large unlabeled tick history, then fine-tune a small head on the scarce "confident-direction" labels. Plausibly extracts structure from quote-size dynamics that hand features miss.
- **Multi-horizon / quantile heads** (TFT, DeepAR, MQ-Forecaster): instead of one 15m sign, predict the *distribution/path* (quantiles of return, vol, MFE/MAE) over 1–60m. Then derive a *conditional* directional bet only when the predicted quantile spread is asymmetric. This reframes 75% as "75% on the subset where the path is predictable," which is the only realistic way to hit it.

---

## Concrete techniques / architectures to try (implementable)

1. **Raw multi-channel sequence encoder (highest-information-content idea).** Feed the model the *un-aggregated* tick/quote matrix: for each pair, channels = {bid, ask, bid_size, ask_size, micro-price, signed trade size}, resampled to a fine grid (e.g., 100ms–1s) over a 5–15 min lookback window. Front-end = **DeepLOB-style block** (Conv2D across price/size levels → Conv1D across time → LSTM/GRU or a small PatchTST attention head). Output = direction at 15m. Rationale: this is the one place DL beats GBM — it sees structure in raw quote-size flow that your 10s OFI aggregation destroys. Use causal/dilated convolutions (TCN) to guarantee no leakage.
2. **Triple-barrier labeling + meta-labeling layer on top of your existing GBM.** Keep LightGBM as the primary side model. Add a secondary model (GBM or small MLP) trained on triple-barrier outcomes to predict P(primary signal is correct). Bet only when meta-P is high. This is the most direct, low-risk way to push the 0.632@0.2% selective frontier. Use **purged + embargoed k-fold CV** (López de Prado) so overlapping labels don't leak.
3. **PatchTST / N-HiTS as multi-horizon *path* forecasters, not sign classifiers.** Train them to forecast quantiles of cumulative return at 1/5/15/30/60 min. Convert to a directional bet only when the median is far from 0 *and* the interquartile band is narrow (low predicted vol). This exploits the one thing DL forecasts well — conditional volatility/quantiles — to *gate* directional bets.
4. **Self-supervised pretraining on unlabeled ticks, then linear-probe / fine-tune.** Pretrain a TCN or small Transformer with a contrastive (TF-C style time-frequency) or masked-reconstruction objective on all 7 pairs' tick history. Freeze, then fit a logistic head on confident-direction windows. Compare linear-probe AUC vs your GBM; if pretraining adds nothing, abandon the DL vector entirely (clean falsification).
5. **xLSTM-TS** (López Gil et al. 2408.12408 report it as the only model beating naive on hourly): worth a single benchmark run as a sequence backbone, **without** the wavelet denoising they used (that's the leak). If it can't beat LightGBM causally, drop it.
6. **Calibration + conditional-coverage analysis.** Whatever the backbone, run isotonic/Platt calibration and measure accuracy-vs-coverage. The deliverable is a *reliable* selective predictor, not a higher raw AUC.

**Explicitly do NOT spend time on:** plain Informer/Autoformer/FEDformer (beaten by linear baselines on FX), zero-shot Chronos/TimesFM/Moirai (coin-flip on returns), or re-implementing DeepLOB expecting a 15m edge (microstructure decays by 5 min — your own result and Lucchese 2024 both confirm).

---

## Reported results & CREDIBILITY assessment

| Claim / model | Reported | Horizon | Credibility | Leakage risk |
|---|---|---|---|---|
| EUR/USD ML stacked ensemble — 58.52% acc (arXiv:2409.04471) | 58.52% | **Daily** | **Credible** — modest, chronological, honest | Low; 2022-only return is regime-lucky |
| DeepLOB on FI-2010 (arXiv:1808.03668) | ~80%+ F1 at small k | **10–100 events (sub-sec–sec)** | Credible *for that horizon*; not minutes | Medium (FI-2010 pre-normalized, saturated) |
| TLOB SOTA on FI-2010 (arXiv:2502.15757) | beats DeepLOB | events | Credible, but authors say **not deployable after spread cost** | FI-2010 benchmark "easy"; NASDAQ much harder |
| Microstructural guide (arXiv:2403.09267) | predictability ∝ tick size, big sim-to-reality gap | events–seconds | **High credibility (skeptical primary source)** | Explicitly warns of gap |
| xLSTM-TS — F1 ~73% (arXiv:2408.12408) | 73% daily / 64–69% hourly | daily/hourly | **Suspect at face value** | **High — wavelet denoising lookahead** |
| TFT (same paper) | "not better than random" | daily | Credible negative result | Low |
| Chronos/TimesFM zero-shot (arXiv:2511.18578) | acc ~49.5–51%, R²<0 | daily | **High credibility** — careful, debunks hype | Low |
| EXFormer FX — "+8.5–22.8% directional" (arXiv:2512.12727) | up to ~22.8% rel. improvement | daily | **Unverified / treat as hype** — relative gain, unaudited, very recent, no independent replication | Unknown |
| Generic "70–90% intraday accuracy" blog/Kaggle claims | 70–90% | intraday | **Reject** — almost always whole-series normalization, overlapping labels, random CV, or no costs | Severe |

**Bottom line:** every result that survives scrutiny lands at **52–59% for minutes-to-daily horizons**; the only >70% numbers are either (a) sub-second microstructure that dies by 5 min, or (b) daily with denoising-lookahead. **There is no credible 75% at 15m in this literature.**

---

## Data sources needed

- **What you already have is the right data** (sub-second bid/ask quotes + **quote sizes** for 7 USD pairs). Quote *size* is the scarce ingredient most papers lack — this is your one genuine edge for a DL sequence encoder. Most retail FX has no true LOB; you effectively have a top-of-book/level-1+size feed, which is more than the daily-bar papers use.
- **True FX limit-order-book depth** (for a real DeepLOB): EBS / Refinitiv FXall / LMAX Exchange / Hotspot. Paid, expensive, institutional. *Recommendation: don't — microstructure won't reach 15m anyway.*
- **FI-2010** (Ntakaris et al. 2018): free, equities, for *benchmarking your DL plumbing only* — known to be saturated/leaky; never report it as a real result.
- **LOBSTER** (lobsterdata.com): reconstructable NASDAQ message-level LOB (TSLA/INTC etc.), paid academic; for replicating TLOB/DeepLOB if you want to validate code.
- **Pretrained TSFM checkpoints** (free, HuggingFace): Chronos (Amazon), TimesFM (Google), Moirai (Salesforce) — only worth downloading if you pursue *fine-tuning from financial data*, not zero-shot.
- **Self-supervised pretraining corpus:** your own full tick history across all 7 pairs + correlated assets (DXY constituents) — no external data needed; that's the point of SSL.

---

## Relevance & priority for OUR project

| Idea | Priority | Interaction with what we've ruled out |
|---|---|---|
| **Raw tick+quote-size sequence encoder (TCN/DeepLOB-front-end → 15m head)** | **High** | The *only* DL approach that uses information your GBM never saw (raw size dynamics, not 10s OFI). Most likely place for an orthogonal lift, though prior is small. Causal convs avoid leakage. |
| **Triple-barrier + meta-labeling on existing GBM, purged CV** | **High** | Directly upgrades your selective-prediction frontier (0.632@0.2%). Low cost, low risk, attacks the realistic 75%-on-a-subset framing. |
| **Self-supervised contrastive/masked pretraining on unlabeled ticks → linear probe** | **Med-High** | Clean falsification test: if pretraining can't beat GBM, the DL vector is dead and you've learned that cheaply. |
| **Multi-horizon quantile heads (PatchTST/N-HiTS/TFT) to *gate* bets by predicted vol** | **Med** | Reframes the problem usefully; exploits DL's real strength (conditional vol/quantiles) rather than sign. |
| **xLSTM-TS / PatchTST as drop-in sequence backbone (no denoising)** | **Med** | One benchmark run vs LightGBM; abandon if no causal edge. |
| **Plain Informer/Autoformer/FEDformer/DeepAR for direction** | **Low** | Beaten by linear baselines on Exchange-Rate; will not beat your 0.527 AUC. |
| **Zero-shot Chronos/TimesFM/Moirai** | **Low (skip)** | Measured coin-flip on returns. Domain-pretraining only. |
| **Re-implement DeepLOB hoping for 15m** | **Low (skip)** | Microstructure dead by 5 min — your finding + Lucchese 2024 + TLOB all agree. |

**Strategic read:** This vector is unlikely to deliver raw 75% at 15m — the architecture is not your bottleneck, and the honest literature ceiling is ~58% even daily. Its real value is (1) a raw-sequence encoder as a last honest attempt to find orthogonal structure in quote sizes, and (2) labeling/meta-labeling/calibration machinery to push the *conditional* (selective) accuracy where 75% is actually attainable on a thin subset.

---

## Sources (annotated)

1. **DeepLOB: Deep Convolutional Neural Networks for Limit Order Books** — Zhang, Zohren, Roberts, IEEE TSP 2019. https://arxiv.org/abs/1808.03668 — The canonical deep-LOB CNN-LSTM; sets the "predict mid-price over k events" template; horizon is events, not minutes.
2. **Deep Limit Order Book Forecasting: a microstructural guide** — Lucchese et al., Quantitative Finance 2025. https://arxiv.org/html/2403.09267v1 — Skeptical primary source: DeepLOB predictability = tick-size artifact, large simulation-to-reality gap; confirms decay with horizon.
3. **TLOB: Transformer with Dual Attention for Price Trend Prediction** — Berti & Kasneci 2025. https://arxiv.org/abs/2502.15757 — Current LOB SOTA; key honest admissions: NASDAQ ≫ harder than FI-2010, accuracy falls with horizon, **profitability dies at spread-sized thresholds**, "not deployable."
4. **Are Transformers Effective for Time Series Forecasting?** — Zeng et al., AAAI 2023. https://arxiv.org/abs/2205.13504 — DLinear beats Informer/Autoformer/FEDformer; linear baselines dominate on the Exchange-Rate dataset. Architecture is rarely the bottleneck.
5. **A Time Series is Worth 64 Words (PatchTST)** — Nie et al. 2023. https://arxiv.org/abs/2211.14730 — Patching + channel-independence restores Transformer competitiveness; wins on seasonal data, no FX-direction edge shown.
6. **An Evaluation of Deep Learning Models for Stock Market Trend Prediction** — López Gil et al. 2024. https://arxiv.org/html/2408.12408v1 — TCN/N-BEATS/TFT/N-HiTS/TiDE/xLSTM-TS bake-off; TFT ≈ random; daily F1 ~73% but via wavelet denoising (leak); hourly drops to 64–69%.
7. **Predicting Foreign Exchange EUR/USD direction using machine learning** — Castillo et al., MLMI 2024. https://arxiv.org/abs/2409.04471 — The credible bar: **58.52% daily** EUR/USD direction; stacked ML; honest chronological setup.
8. **Re(Visiting) Time Series Foundation Models in Finance** — Rahimikia 2025. https://arxiv.org/abs/2511.18578 — Chronos/TimesFM zero-shot: negative R², ~50% directional, underperform CatBoost/LightGBM; only from-scratch financial pretraining helps.
9. **Advancing Financial Forecasting: N-HiTS vs N-BEATS** — Apte et al. 2024. https://arxiv.org/abs/2409.00480 — Claims N-HiTS/N-BEATS improve financial forecasts; point-forecast framing, no rigorous OOS direction-accuracy proof — treat as suggestive, not load-bearing.
10. **Temporal Fusion Transformers for interpretable multi-horizon forecasting** — Lim et al., IJF 2021. https://www.sciencedirect.com/science/article/pii/S0169207021000637 — The multi-horizon/quantile architecture worth borrowing for *path/vol gating*, not sign classification.
11. **N-HiTS: Neural Hierarchical Interpolation for Time Series** — Challu et al. 2022. https://arxiv.org/pdf/2201.12886 — Efficient multi-rate hierarchical forecaster; candidate quantile path-forecaster backbone.
12. **Self-Supervised Contrastive Pre-Training via Time-Frequency Consistency (TF-C)** — Zhang et al., NeurIPS 2022. https://zitniklab.hms.harvard.edu/projects/TF-C/ — SSL recipe to pretrain on unlabeled ticks before fine-tuning a tiny direction head.
13. **Contrastive Learning of Asset Embeddings from Financial Time Series** — 2024. https://arxiv.org/abs/2407.18645 — Contrastive representation learning specifically for financial series; informs the SSL-on-ticks idea.
14. **Advances in Financial Machine Learning** — López de Prado 2018 (triple-barrier, meta-labeling, purged/embargoed CV). mlfinlab docs: https://mlfinpy.readthedocs.io/en/latest/Labelling.html — The labeling/CV discipline that prevents the leaks behind fake 75–90% results and powers selective prediction.
15. **Algorithmic crypto trading using information-driven bars, triple-barrier labeling and deep learning** — Tan et al., Financial Innovation 2025. https://link.springer.com/article/10.1186/s40854-025-00866-w — Event bars + triple-barrier + meta-labeling help; ResNet-LSTM beat Transformers; Autoformer/FEDformer adapted poorly to direction.
16. **EXFormer: Multi-Scale Trend-Aware Transformer for FX Returns** — Liu et al. 2025. https://arxiv.org/abs/2512.12727 — Claims +8.5–22.8% directional improvement on daily EUR/USD etc.; **unverified, very recent, no independent replication — flagged as hype until reproduced.**
17. **FI-2010 Benchmark Dataset** — Ntakaris et al. 2018. https://arxiv.org/abs/1705.03233 — The standard (and saturated/leaky) LOB benchmark; use only to validate DL plumbing.
