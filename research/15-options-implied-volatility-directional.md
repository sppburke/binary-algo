# Options-Implied & Volatility-Derived Directional Information for Intraday FX

Research vector 15 of the binary-algo direction-prediction project. Goal frame: lift **15-minute EURUSD directional accuracy toward 75% OOS** using NEW orthogonal signals. This vector evaluates FX options–implied data: 25-delta risk reversals (skew), butterflies, IV term structure, implied-vs-realized vol spread, dealer gamma exposure ("gamma walls"/pinning), variance risk premium, and option expiry magnets.

**Bottom line up front:** Almost all *credible* options-implied directional predictability in FX lives at **daily-to-monthly horizons**, not 15 minutes. The two mechanisms that *are* genuinely intraday and mechanistic — (a) dealer gamma hedging feedback and (b) option-expiry/strike "magnets" — are well-documented but the peer-reviewed evidence says they move **realized volatility and pinning, not signed direction**, and the one paper that tests intraday *directional* momentum from gamma hedging finds the effect **statistically insignificant specifically in currencies** (significant in equities/bonds/commodities). So this is a Medium-priority vector: one or two narrow, mechanistically-grounded micro-edges (expiry magnet, gamma-regime vol gating) are worth building, but the headline "risk reversals predict EURUSD direction" claim does not survive scrutiny at 15m.

---

## TL;DR (most actionable for our 15m FX goal)

- **Risk reversals are a POSITIONING/sentiment gauge, not a 15m directional predictor.** Credible academic work finds RR predicts currency returns/carry-crash risk only at **weekly–monthly** horizons, and even daily next-day regressions show "negative association but limited explanatory power" (mixed/weak). Do **not** expect RR to give a clean 15m edge; treat it as a slow-moving *regime/state* feature, not a trigger. (Della Corte et al.; Jurek; redalyc EM RR study)
- **Dealer gamma in FX drives VOLATILITY, not signed direction** — this is the single most important finding. Ulmann–Sornette (JIMF 2022), using reconstructed DTCC dealer positioning for EURUSD/USDJPY, show OMMs are **persistently short gamma**, and net short gamma *raises realized vol* (≈ +0.7% in EURUSD per −$1000bn gamma) but does **not** predict the *sign* of the next move. Use gamma as a **volatility-regime gate / target-scaler**, not a direction signal.
- **The "gamma → intraday momentum" trade fails in currencies.** Baltussen, Da, Lammers & Martens (JFE 2021) find strong last-30-min intraday momentum in equity/bond/commodity futures (OOS R² up to 2.88%) but in **currency futures the coefficients are "positive but insignificant"** and the U-shaped volume pattern (the mechanism's prerequisite) is weakest in FX. This is a direct negative result for applying the most-cited gamma-momentum idea to EURUSD.
- **Option-expiry "magnets" (10am NY cut) are the most plausibly-exploitable INTRADAY mechanism**, but the edge is conditional (large notional ≥ $0.5–1bn at a strike, spot within ~30–50 pips, quiet tape) and is fundamentally a **mean-reversion-toward-strike** effect, not a free 75% directional call. Worth building as a feature; capacity and reliability are limited.
- **Implied-minus-realized vol spread (variance risk premium) is a vol-timing and weak return signal at ≥1-month**, with no credible 15m directional content. Useful only as a slow risk-on/off context feature, redundant with vol-regime features you already have.
- **Free/cheap data exists** to build crude dealer-gamma and expiry-magnet features: **CME QuikStrike** FX options open-interest-by-strike (free, daily, EUR/USD weekly+monthly), **DTCC public price dissemination** (free, real-time FX-option trade tape, but messy), and the daily ForexLive/Investing "10am NY cut" expiry lists. Clean historical IV surfaces (RR/BF/ATM by tenor and delta) require **Bloomberg BVOL / OVDV or LSEG/Refinitiv** (paid).
- **Leakage/credibility caveat:** most online "RR predicts FX" and "GEX 75% win-rate" claims are practitioner blogs (StrikeWatch, SpotGamma) extrapolating from *equity* index structure, where skew is structurally negative and dealers are differently positioned. FX skew is a U-shaped smile and dealer positioning is different; do not import equity-GEX intuition wholesale.
- **Net recommendation:** treat options-implied data as **state/gating features** (vol regime, gamma sign, distance-to-expiry-strike) layered onto your existing model, plus **one targeted expiry-magnet conditional model**. Do not expect it to be the orthogonal signal that breaks 75% on its own.

---

## Key findings (with inline citations)

### 1. Risk reversals & skew: a sentiment/positioning gauge, predictive only at slow horizons

- A 25-delta risk reversal = IV(25Δ call) − IV(25Δ put); positive ⇒ market pays up for upside (bullish skew), negative ⇒ pays up for downside protection (bearish/crash-fear skew). It is "the market's best guess about directional bias" and changes signal *changing expectations* (GFMI; Wikipedia; Derivative Engines). This is **interpretation, not demonstrated forecasting power**.
- Crucially, the standard caution from practitioners and academics: "High put demand does not mean the market will fall — it means participants are *hedged* for a fall. The risk reversal does not tell you where price is going; it tells you where the options market is pricing risk." (StrikeWatch 2026). So RR is a *risk/positioning* state, easily contrarian.
- Academic predictability is **horizon-mismatched for us**: Jurek and Brunnermeier/Nagel/Pedersen ("Carry Trades and Currency Crashes," NBER Macro Annual 2008) tie negative conditional skewness and RR to *carry-trade crash risk*, with VIX spikes coinciding with carry unwinds — a **weekly-to-monthly, cross-sectional** phenomenon. Della Corte, Ramadorai & Sarno ("Volatility risk premia and exchange rate predictability," JFE 2016) predict appreciation at the **currency variance risk premium 4–6-month horizon**. None of this is 15m.
- Direct daily RR→next-day-direction regressions report "a negative association between past risk reversal changes and exchange-rate returns, but with limited explanatory power, casting doubt on usefulness for prediction" (search synthesis of EURUSD daily studies; consistent with redalyc EM RR study). i.e. weak even at *daily*, let alone 15m.

### 2. Dealer gamma in FX: a VOLATILITY driver, not a direction predictor (the key result)

- **Ulmann & Sornette (with Anderegg/Wehrli), "The impact of option hedging on the spot market volatility," Journal of International Money and Finance 124 (2022)** — the most rigorous FX-specific gamma study. They reconstruct option-market-maker (OMM) delta/gamma exposure for **EURUSD and USDJPY from DTCC trade data** (Oct 2017–Jun 2018). Findings:
  - OMMs are **short gamma at essentially all times** (they net-sell puts and calls).
  - Spot **volatility rises with negative gamma**: ≈ **+0.7% absolute vol in EURUSD** (and +0.9% in USDJPY) per ≈ −$1000bn OMM gamma. Difference attributed to USDJPY's lower spot liquidity (higher market impact).
  - The model is about **|move| magnitude / realized vol amplification**, NOT the sign of the move. Negative gamma ⇒ dealers chase moves (destabilizing) ⇒ bigger ranges, but **no signed-direction prediction**.
  - Companion: Sornette, Ulmann & Wehrli, "On the Directional Destabilizing Feedback Effects of Option Hedging" (SSRN 4087222) — even the "directional" framing is about feedback amplification of an existing move, not ex-ante sign prediction.
- A separate **S&P 500** master's thesis (DiVA 2024) finds Δ(GEX) is "significantly and positively associated with S&P 500 returns" and improves OOS forecasts — but this is **equities**, where dealer positioning sign and skew structure differ fundamentally from FX. Do not assume it transfers to EURUSD.

### 3. The gamma→intraday-momentum trade is INSIGNIFICANT in currencies (direct negative result)

- **Baltussen, Da, Lammers & Martens, "Hedging demand and market intraday momentum," JFE 142 (2021) 377–403.** Using tick data on 60+ futures (1974–2020) incl. **8 currency futures (Tick Data LLC)**: the last-30-min return is positively predicted by the return-over-rest-of-day (rROD), attributed to short-gamma hedging that trades *with* the move into the close.
  - Equities: highest OOS R² **2.88%**, highly significant.
  - **Currencies (Panel D): the exception.** Direct quote from the paper: *"The currency futures market is again the exception with the coefficients for both rROD and rONFH positive but insignificant."* The authors attribute this to FX having the **least-pronounced U-shaped intraday volume pattern** (no concentrated close auction → the hedging-into-the-close mechanism is weak in 24h FX).
  - One *narrow* positive: Elaut et al. (2018) document intraday momentum in **RUB/USD** since 2005 — an exotic, not EURUSD.
- Implication for us: the most-cited, most-credible gamma-driven *directional* intraday effect **specifically does not work in major USD pairs** — consistent with your own finding that EURUSD is ~97% USD-factor and lag-1 autocorr ≈ 0.

### 4. Option-expiry "magnets" / pinning — the one genuinely intraday, mechanistic, exploitable effect

- Mechanism (practitioner-standard, well-described): near a large vanilla expiry, as spot approaches the strike, dealers long gamma hedge by **selling above / buying below the strike**, pinning spot toward the strike into the **10am New York cut (14:00 GMT)**. After the cut, the pull disappears (ForexLive/Investing daily expiry reports; FinancialSource; ForexFactory).
- Practical rules-of-thumb from desk lore (NOT peer-reviewed, but mechanistically coherent): notional **≥ $500m–$1bn** at a strike to matter; spot within **~30–50 pips**; effect strongest in **quiet** tape and **dissolves on news**. This is fundamentally **mean-reversion toward a known level**, conditional and capacity-limited — not an unconditional directional edge.
- **Barrier options** add a second mechanism: near knock-out barriers dealers must unwind hedges if spot touches, which can *accelerate* moves through the level (Wystup, "Barriers Brake the Spot," MathFinance 2025). This is the *anti-pin* case (momentum through a level), and is real in EURUSD around well-known barrier clusters.

### 5. Implied-vs-realized vol spread / variance risk premium — vol timing, not 15m direction

- IV−RV (the variance risk premium) is the structural short-vol edge; ratio ≫1 ⇒ rich options, ≪1 ⇒ cheap (OAS; MenthorQ). As a *return* predictor it works at **≥1 month** (Della Corte et al. JFE 2016; Bollerslev-style VRP). No credible 15m signed-direction content. For us it is at best a slow risk-on/off context feature, **largely redundant** with the realized-vol-regime and daily-context features you already built.

### 6. NY Fed implied-vol data is dead (don't waste time)

- The NY Fed FX implied-volatility series is **ATM only, monthly, and was discontinued 30 Sep 2013** ("averages of mid-level rates… at 11:00am on the last business day of the month"). Useless for intraday and stale by a decade. (NY Fed Implied Volatility Rates page.)

---

## Concrete techniques / features / architectures to try

Ranked by expected value for the 15m goal. All designed as **causal, point-in-time** features to bolt onto your existing model — not standalone 75% predictors.

### A. Expiry-magnet conditional model (HIGH within this vector)
1. Ingest daily large-expiry strike lists (ForexLive/Investing "10am NY cut", or reconstruct from CME QuikStrike OI-by-strike, or DTCC tape).
2. Features, all causal at bar close: `dist_to_nearest_large_strike_pips`, `notional_at_strike_usd`, `signed_dist = (strike - spot)/ATR`, `minutes_to_1000NY`, `is_quiet_tape = realized_vol_30m < median`, `near_barrier_flag`.
3. **Conditional target**: only predict on bars where spot is within ~40 pips of a ≥$0.75bn strike AND ≥30 min before the 10am NY cut AND tape is quiet. Hypothesis: P(move toward strike) > 0.5 → a *selective-prediction* edge, aligned with your existing selective-prediction approach (you got 0.632 at 0.2% coverage; this is a candidate higher-precision micro-bucket).
4. Train a small separate classifier on this sub-sample; evaluate hit-rate vs coverage. Expect a *real but small-coverage* edge, not 75% across all bars.

### B. Dealer-gamma volatility-regime GATE (HIGH — use as meta-feature, not direction)
1. Reconstruct a crude OMM net-gamma proxy from CME QuikStrike OI-by-strike (free) or DTCC tape: `GEX(S) = Σ_k OI_k · γ_k(S) · sign_dealer`, assuming dealers **short gamma** (Ulmann–Sornette) → mostly negative.
2. Build `gamma_regime = sign(net_gamma)` and `gamma_magnitude`.
3. **Do not use it to predict direction.** Use it to (a) **gate** when your existing momentum/microstructure signals are likely to *persist* (negative gamma = trend-amplifying) vs *mean-revert* (positive gamma = pin/range), and (b) scale position size / abstain. This converts the credible Ulmann–Sornette vol result into a regime switch on your existing features — the mechanistically honest use.

### C. Risk-reversal as a slow STATE feature + its short-horizon *change* (MED)
1. Daily 25Δ RR and 10Δ RR, 1w/1m tenors (from Refinitiv `EUR1MR25=` style RICs or Bloomberg BVOL).
2. Features: `RR_level_zscore` (positioning regime), `RR_1d_change`, `RR - ATM` (skew steepness), `BF_level` (tail demand).
3. Test as **conditioning state** in your gradient-boosted model (interactions with existing momentum), NOT as a standalone trigger. Pre-register the falsifier: if RR-interaction terms add < 0.002 AUC OOS on val 2022–23, drop it (this is the likely outcome given the daily-regression weakness).

### D. IV term-structure & smile dynamics as regime context (LOW–MED)
- `IV_term_slope = IV_1m − IV_1w`, `IV_RV_spread`, `smile_curvature`. Inversion of the term structure (1w > 1m) flags imminent event stress. Use as risk-off gate, expect redundancy with your vol-regime features.

### E. Barrier-proximity momentum feature (MED, niche)
- Flag when spot is within X pips of a well-known round-number/barrier cluster with large OI; in **negative-gamma** regime, condition for *continuation through* the level (Wystup). Pairs naturally with (B).

### Architecture note
None of these are sequence-model upgrades; they are **exogenous state features + a conditional sub-model**. Best integration: (1) add A–E as features to the existing LightGBM/CatBoost ensemble; (2) build the expiry-magnet selective sub-model separately; (3) use gamma-regime as a *gating/abstention* variable on top of your existing selective-prediction layer.

---

## Reported results & CREDIBILITY assessment

| Claim / source | Horizon | Credibility | Notes / leakage risk |
|---|---|---|---|
| Ulmann–Sornette: FX dealer gamma → realized vol (EURUSD/USDJPY, DTCC) | intraday/daily | **High** (peer-reviewed JIMF 2022, real DTCC reconstruction) | Predicts **vol, not direction**. Short ~8-month sample. Honest, mechanistic. |
| Baltussen et al.: gamma→intraday momentum | last 30 min | **High** (JFE 2021) but **NEGATIVE for FX** | Currencies "positive but insignificant." Strong evidence the trade *doesn't* work in EURUSD. |
| Della Corte/Jurek/BNP: RR & variance-risk-premium predict currency returns | **4–6 months / weekly** | **High** but **wrong horizon** | Cross-sectional carry/crash, not 15m EURUSD direction. |
| Expiry "magnet" / 10am NY cut pinning | intraday | **Medium** (desk lore, mechanistically sound, no clean academic OOS) | Conditional, low-capacity, dissolves on news. Real but small. |
| Equity GEX → S&P returns (DiVA thesis, SpotGamma, StrikeWatch) | intraday | **Medium for equities, LOW transfer to FX** | Equity skew/dealer-positioning structure ≠ FX. Importing GEX-75%-winrate intuition to EURUSD is a leakage/regime-transfer error. |
| Blog "75%/90% accuracy" RR/skew signals | varies | **Low / hype** | Marketing; usually in-sample, equity-centric, or no transaction costs. Discard. |

**Leakage red flags to watch in our own build:** (1) using *end-of-day* IV/RR to predict *intraday same-day* moves (look-ahead — IV surfaces are often timestamped at a fixing); (2) survivorship/selection in expiry lists (only large expiries that "mattered" get reported ex-post); (3) reconstructing dealer gamma with the *wrong sign assumption* (Ulmann–Sornette show short-gamma is the FX norm, but it can flip — assuming a constant sign is a subtle bias); (4) RR data vendors sometimes back-fill/revise — pin a point-in-time snapshot.

---

## Data sources needed

| Data | Source | Cost | Fidelity / notes |
|---|---|---|---|
| FX options **open interest by strike** (EUR/USD weekly + monthly) | **CME QuikStrike** (OI Heatmap, OI Profile) + CME daily Volume/OI reports | **Free** | Daily granularity, listed-only (misses OTC). Enough for a crude GEX-by-strike proxy. Best free path to dealer-gamma reconstruction. |
| **DTCC public price dissemination** — OTC FX-option trade tape (strike, notional, tenor, pair) | pddata.dtcc.com (CFTC dashboard) | **Free** | Real-time but **messy**: Clarus documents mis-quoted call/put, inverted strikes, 0-premium garbage, ambiguous ccy terms. Needs heavy cleaning (imply-vol-both-ways to infer C/P). Covers the OTC bulk dealers actually hedge. |
| Daily large-expiry strike lists ("10am NY cut", ≥$100m) | ForexLive / Investing.com / FinancialSource | **Free** | Curated, human-filtered; convenient for the expiry-magnet feature. Selection-biased (only "notable" expiries). |
| **Historical IV surfaces**: ATM, 25Δ/10Δ RR & BF, by tenor (1w–1y) | **Bloomberg BVOL/OVDV** or **LSEG/Refinitiv** (RICs e.g. `EUR1MR25=`, `JPY3MR10=`) | **Paid** | The clean, point-in-time gold standard for RR/skew features. Refinitiv exposes per-delta RR/BF historically via RDP. |
| ATM IV only, monthly, ≤2013 | NY Fed Implied Volatility Rates | Free but **dead/useless** | Discontinued 2013, monthly, ATM-only. Skip. |
| Retail-friendly IV snapshots | Investing.com, MyFXBook, MarketMilk, Barchart (Euro FX futures options) | Free | Snapshot/live only, weak history, not point-in-time clean. OK for sanity checks. |

For a budget build: **CME QuikStrike (gamma-by-strike proxy) + DTCC tape (OTC notional) + ForexLive expiry lists** gets you A, B, E for free. Add Refinitiv/Bloomberg only if RR/skew (C, D) shows lift in cheap prototyping.

---

## Relevance & priority for OUR project

Interaction with what we've already ruled out:
- **Order-flow proxy (signed 10s OFI) added ~0 lift** and **true tick imbalance decays by 1–5 min** → consistent with the FX literature here: the intraday options/gamma effect is *vol-amplifying* and *expiry-localized*, not a persistent directional flow. Options data won't resurrect microstructure direction at 15m.
- **EURUSD ~97% USD-factor, lag-1 autocorr ≈ 0** → directly explains *why* the Baltussen intraday-momentum trade is insignificant in FX. Options-implied data does not add an idiosyncratic directional component here.
- **Selective prediction got 0.632 @ 0.2% coverage** → the **expiry-magnet conditional model (A)** is the most natural extension: a new high-precision, low-coverage bucket defined by an *external mechanistic state*, orthogonal to your TA features.
- **Calendar/event-timing was redundant with time-of-day** → the **10am NY cut** time feature partially overlaps existing session features; the *novel* part is the **strike/notional geometry**, not the time itself. Keep only the geometry.

Priority ranking:
- **HIGH:** (A) Expiry-magnet selective sub-model; (B) Dealer-gamma **volatility-regime gate** (not direction) from free CME/DTCC data. These are orthogonal, mechanistically credible, and free to prototype.
- **MEDIUM:** (C) RR/skew as slow state feature with pre-registered drop criterion; (E) barrier-proximity continuation feature.
- **LOW:** (D) IV term-structure context (redundant with vol-regime); any standalone RR/VRP *directional* predictor; anything imported from equity-GEX 75%-winrate blogs.

Honest expectation: this vector yields **gating/regime/selective-bucket improvements and possibly a small high-precision expiry-magnet edge**, not the orthogonal signal that lifts *all-bar* 15m accuracy to 75%. The strongest, most-cited directional gamma effect is *proven absent in currencies*; the strongest FX gamma effect is *volatility, not direction*.

---

## Sources (annotated)

1. **Ulmann, Anderegg, Sornette — "The impact of option hedging on the spot market volatility," Journal of International Money and Finance 124 (2022)** — https://www.sciencedirect.com/science/article/pii/S0261560622000304 (RePEc: https://ideas.repec.org/a/eee/jimfin/v124y2022ics0261560622000304.html) — *The* FX-specific gamma paper; reconstructs EURUSD/USDJPY dealer gamma from DTCC; shows short-gamma raises **vol** (+0.7% EURUSD), not direction.
2. **SNB Working Paper 2019-03, "Quantification of feedback effects in FX options markets"** — https://www.snb.ch/en/publications/research/working-papers/2019/working_paper_2019_03 — Open working-paper version of the DTCC FX-gamma reconstruction; method + magnitudes.
3. **Sornette, Ulmann, Wehrli — "On the Directional Destabilizing Feedback Effects of Option Hedging," SSRN 4087222** — https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4087222 — Even the "directional" feedback is *amplification of an existing move*, not ex-ante sign prediction.
4. **Baltussen, Da, Lammers, Martens — "Hedging demand and market intraday momentum," JFE 142 (2021) 377–403** — open PDF https://www3.nd.edu/~zda/intramom.pdf — Gamma-hedging intraday momentum; **currencies are the insignificant exception** (direct negative result for FX). Equity OOS R² 2.88%.
5. **Brunnermeier, Nagel, Pedersen — "Carry Trades and Currency Crashes," NBER Macro Annual 23 (2008)** — https://www.journals.uchicago.edu/doi/full/10.1086/593088 — RR/skew ↔ carry-crash risk; monthly, cross-sectional; the canonical FX-skew-direction link (wrong horizon for us).
6. **Della Corte, Ramadorai, Sarno — "Volatility risk premia and exchange rate predictability," JFE 121 (2016)** — https://www.sciencedirect.com/science/article/abs/pii/S0304405X16300150 — Variance-risk-premium predicts FX appreciation at **4–6 months**, not intraday.
7. **"Can implied volatility predict returns on the currency carry trade?" (J. Banking & Finance 2015)** — https://www.sciencedirect.com/science/article/abs/pii/S0378426615001570 — Rising IV predicts negative carry returns next week; weekly, carry-specific.
8. **Wystup — "Barriers Brake the Spot," MathFinance (2025)** — https://www.mathfinance.com/wp-content/uploads/2025/01/2025-01-Barriers-brake-the-spot.pdf — Barrier-option hedging accelerates/pins spot near knock-outs; the anti-pin (momentum-through-level) mechanism.
9. **Clarus FT — "FX Options Data on the SDR"** — https://www.clarusft.com/fx-options-data-on-the-sdr/ — Documents the free DTCC FX-option public tape AND its data-quality landmines (mis-quoted C/P, inverted strikes, 0-premium garbage). Read before reconstructing gamma from it.
10. **CME QuikStrike — Options Open Interest Heatmap / Profile** — https://www.cmegroup.com/tools-information/quikstrike/open-interest-heatmap.html — Free daily EUR/USD OI-by-strike (weekly+monthly options); the practical free input for a listed-option GEX-by-strike proxy.
11. **StrikeWatch EA — "Volatility Skew and the 25-Delta Risk Reversal as a Directional Signal" (2026)** — https://www.strike-watch.com/lab/volatility-skew-25-delta-risk-reversal-directional-signal — Clear practitioner explainer of RR mechanics and the crucial caveat ("RR tells you where risk is priced, not where price is going"). Equity-centric — transfer to FX with skepticism.
12. **SpotGamma — Gamma Exposure (GEX) explainer** — https://spotgamma.com/gamma-exposure-gex/ — Reference for GEX mechanics, positive-vs-negative-gamma regime behavior, pinning. Equities; use only for mechanism intuition.
13. **FinancialSource — "How to Use Option Expiry Levels in FX"** — https://financialsource.co/how-to-use-option-expiry-levels-in-fx — Practitioner rules for 10am-NY-cut expiry magnets (size thresholds, distance, quiet-tape conditionality).
14. **ForexLive — daily "FX option expiries for … 10am New York cut"** — https://investinglive.com/Orders/ (recurring) — Free curated large-expiry strike lists; convenient daily feed for the expiry-magnet feature.
15. **NY Fed — Implied Volatility Rates** — https://www.newyorkfed.org/markets/impliedvolatility.html — Confirms the free official FX IV series is ATM-only, monthly, **discontinued 2013** (don't use).
16. **LSEG/Refinitiv Developer Community — historical FX IV / RR retrieval** — https://community.developers.refinitiv.com/questions/81902/ — Shows RIC structure (`JPY3MR10=`, RDP `get_historical_price_summaries`) for paid point-in-time RR/BF/ATM by tenor and delta.
