# 00 — Research Index & Orientation

## The goal, the splits, what's ruled out

**Goal.** Predict the *direction* (up/down vs current spot) of EURUSD (and other liquid USD FX majors)
at a **15-minute horizon**, targeting **>75% correct-prediction rate**, verified strictly out-of-sample.
**Splits:** train 2012–2021, validation 2022–2023, test 2024–2025, **2026 fully held out**. We hold
10-second OHLCV bars plus raw sub-second bid/ask ticks **with quote sizes** for 7 USD pairs.

**Already exhausted (do NOT re-recommend as-is):** 239 causal multi-timeframe TA features
(EMA/RSI/MACD/Bollinger/realized-vol/return-autocorr/session); cross-pair lead-lag from the 6 other USD
pairs; signed-10s-volume tick-rule OFI proxy (~0 lift); LightGBM/XGBoost/CatBoost/TabNet/GRU ensembles
(all best so far ~0.52–0.527 AUC); stat-arb USD-basket residual fade; daily/weekly context (vol regime, overnight
gap, range position, seasonality); exogenous longer-horizon peer features; an economic-calendar
*event-timing/vol-seasonality* proxy (found redundant with time-of-day); and selective prediction
(best 15m result **~0.632 OOS accuracy at 0.2% coverage**). **Mechanistic ground truth:** true raw-tick
order-book imbalance is real (55.3% next-tick) but decays to ~0.50 by ~1 min and is gone by ~5 min;
the only clean >75% we ever found is at the **3-second** horizon. EURUSD is ~97% USD-factor; lag-1 autocorr
of 5m returns is ~−0.03. **The single most-cited external reality check** (Petrova–Vilhelmsson–Nordén 2026,
*Int. J. Forecasting*) ran essentially our experiment on *better* data (real FX LOB, 1min–1h, PCA/LASSO/RF,
cross-pair, OOS, cost-aware) and found "generally low predictability… supporting the EMH" — our 0.527
current best level matches the literature's current best level, not a pipeline bug.

---

## Annotated table of contents

| File | Title | Best ideas (2-line summary) |
|---|---|---|
| **01** | Intraday/short-horizon FX predictability literature | Peer-reviewed consensus: 15m FX direction prediction is still open at present; the "79% direction" figure is *equities, 5-second, with an order-flow peek*. The one robust orthogonal pattern is the **fixing-window reversal** (Krohn–Mueller–Whelan, *JoF* 2024); also **surprise-conditioned post-news drift** (not vol-seasonality). |
| **02** | Order flow & LOB predictability seconds→minutes | FX OFI impact "disappears at 15–30 min" (FRB/EBS). Best upgrades: **integrated multi-level OFI** (Cont–Cucuringu–Cont, contemporaneous though) and the genuinely timescale-matched **metaorder-in-progress detector** (order-flow long memory, persistent same-sign flow). |
| **03** | FX positioning, customer & retail flow | Real customer flow is daily/monthly & bank-proprietary. The accessible, orthogonal, intraday lever is the **OANDA Order/Position Book** (price-level stop/limit clusters = a liquidity map); retail is return-contrarian (JIFMIM 2025, EURUSD-specific). |
| **04** | Deep-learning architectures for FX direction | Architecture is *not* the bottleneck — DLinear beats Transformers on Exchange-Rate; zero-shot TSFMs are ~0.50. Transferable: **raw quote-size sequence encoder**, **triple-barrier + meta-labeling**, self-supervised pretraining on ticks. Honest best so far ~58.5% *daily*. |
| **05** | Microstructure features & realized vol | Semivariance/signed jumps predict **volatility, not direction**; jump *direction* ≈ ~0.50 (49%). Real levers: **microprice + quote-size imbalance**, depth-normalized OFI, **macro-surprise-signed news-window** model, HAR-RV-J vol-nowcast to set the selective threshold. |
| **06** | Meta-labeling, triple-barrier, AFML, selective | Meta-labeling raises *precision on bets*, cannot create edge that isn't there (bounded by the ~0.51 primary). Genuinely new: **fractional-differentiation** features (memory that returns destroy); plus the **AFML validation stack** (purged/CPCV + DSR/PBO) to avoid declaring a fluke. |
| **07** | Cross-asset & macro lead-lag | EURUSD is dollar-factor (Verdelhan) → signal must come from the factor's *drivers*. Best untried: **rate-differential (US-DE 2y/10y) fair-value-gap residual** + **release-window driver-leads-spot** model; split target into continuous vs jump. |
| **08** | News / sentiment / NLP | Generic sentiment is daily-to-weekly; the **latency challenge** means fast machine-readable news is priced within seconds at 15m. Native-15m exception: **signed economic surprise** (actual−consensus) + pre-30m drift / post-15m window; central-bank tone overlay (CB-specific scoring, not FinBERT). |
| **09** | Regime-switching & conditional predictability | Reframe target as *conditional*: regime models help **direction-of-sign** even when they fail on MSE (Engel). Use **HMM/vol-regime as a gate**, not a direction oracle; vol regime **flips the sign** of short-horizon autocorr (explains our −0.03); pre-FOMC drift is a directional, scheduled regime. |
| **10** | Practitioner & community knowledge | Honest verdict *matches* our current best level; "75%/90%" bots are statistic-inversion or martingale. The one durable direction effect is **fixing/clock seasonality** (EUR survives costs, GBP/JPY don't). Steer: **cross-asset intraday LEVELS** (rates/ES), net-of-spread labels, Deflated Sharpe gate. |
| **11** | GitHub/Kaggle/competition solutions | Biggest credible lift in an analogous comp (Jane Street 2024) came from **online/walk-forward weight updates** (+0.008 vs +0.001–0.002 for features) — untried by us. Also: multi-task auxiliary-horizon targets, combinatorial imbalance features, supervised denoising AE. Headline >75% comp results were *leakage* (recovering shuffled index). |
| **12** | Binary-option economics & less-efficient instruments | Break-even is **~57%** at 0.75 payout, not 75% → 0.632 is already economically live at low payouts; the problem is *coverage/stability*. Deriv synthetics: direction is a CSPRNG **martingale** (out of scope). The real pivot: **crypto at 15m** (documented less efficient, real order flow, BTC lead). |
| **13** | Calibration, conformal, abstention | **Calibration is order-preserving → cannot move the risk-coverage curve** (key myth-buster). For a *guaranteed* accuracy floor use **Mondrian conformal reject-option**, and **DtACI online adaptation** to stop the threshold collapsing OOS. Makes the thin edge *honest/stable*, doesn't manufacture it. |
| **14** | Feature engineering: bars, transforms, signatures | Entropy/Hurst/VPIN/wavelet are **sign-blind (magnitude, not direction)** — most "76% wavelet" results are decomposition leakage. Two genuinely untried orthogonal levers: **information-driven (imbalance/run) bars** (re-clock + re-label) and **rough-path signatures** (cross-pair Lévy area / covariation-of-increments). |
| **15** | Options-implied & volatility-derived direction | Risk-reversals are a slow positioning gauge; FX dealer gamma drives **vol, not direction**; the gamma→intraday-momentum trade is **insignificant in currencies** (Baltussen). The one intraday mechanism: **option-expiry "magnet" / 10am NY cut** (mean-reversion toward large strikes) as a selective bucket. |
| **16** | Data sources & tooling | Biggest data gap = true FX depth (paywalled). Cheapest *real* multi-level EUR/USD book is **CME 6E futures (Databento)**; pre-validate deep-LOB methodology free on **crypto (Tardis)**. Also: **calendar surprise** APIs, GDELT 15-min tone, OANDA book. Customer flow (the signal that works) is structurally unbuyable. |

See **SYNTHESIS.md** for cross-cutting themes, the ranked list of most-promising avenues, and the honest
assessment of the open target of 75% at 15m.
