# SYNTHESIS — Lifting 15-Minute EURUSD Directional Accuracy

*Synthesis across research vectors 01–16. Evidence-driven and deliberately skeptical: credible findings are
separated from hype, and every "75%/90%" claim in the source literature is treated as overfit/leaked until
proven OOS with costs. The single most-cited external anchor — Petrova–Vilhelmsson–Nordén 2026 (Int. J.
Forecasting), which ran our experiment on better data and found near-EMH predictability at 1min–1h — is the
prior every avenue below must beat.*

---

## Cross-cutting themes

These are the signal sources and methods that **multiple independent reports converge on**. Convergence is
the strongest evidence we have, because each report mined a different literature.

### 1. The same null result, from five different directions
Reports **01, 02, 04, 07, 10, 16** independently land on the *same* peer-reviewed conclusion: liquid FX at
1min–1h is statistically near-random, and the **Petrova–Vilhelmsson–Nordén 2026** FX-LOB study (cited by 01,
02, 04-adjacent, 07) is the recurring "prior to beat." Our ~0.527 AUC ceiling is corroborated, not anomalous.
**Implication:** abandon the search for an *always-on* 15m direction signal. Every credible path is
*conditional* (a minority of bars) or *selective* (bet rarely).

### 2. Reframe from unconditional accuracy → conditional/selective precision
Reports **06, 09, 11, 12, 13** all converge here. The literature (Engel: regime models help *direction* even
when they fail on MSE; AFML meta-labeling; selective classification) says the realistic prize is **high
accuracy at low-to-moderate coverage inside identifiable regimes**, which is exactly the shape of our existing
0.632@0.2% result. The unifying program: find the minority of bars where edge is mechanistic, and abstain on
the rest. Binary economics (report 12) reframes the bar itself: **break-even is ~57% at 0.75 payout, not 75%**
— so 0.632 is already economically live; the open problem is *coverage and stability*, not crossing break-even.

### 3. The mechanistically-special windows are the same handful everywhere
Reports **01, 05, 07, 08, 09, 10, 15** independently nominate the *same* regimes as where directional edge
plausibly exists at 15m:
  - **The FX fixings** (Tokyo / ECB 14:15 CET / WM-R 16:00 London): USD drifts into the fix and reverses after
    — a *signed* pattern (Krohn–Mueller–Whelan, *JoF* 2024), distinct from the vol-seasonality we already ruled
    out. Named by 01, 10 (and implicitly 09).
  - **Scheduled macro releases** conditioned on the **signed surprise** (actual−consensus), not the timing:
    pre-30m drift + post-15m window. Named by 01, 05, 07, 08, 09, 16. This is the explicit re-opening of our
    "calendar redundant with time-of-day" finding — the surprise *sign* is the directional payload we never used.
  - **Option-expiry "magnets" / 10am NY cut** and **gamma-regime** vol gating (report 15).

### 4. Order flow is the signal that works — but the version that works is unbuyable
Reports **01, 02, 03, 05, 16** agree: signed *customer* order flow is the most robust FX driver in academia
(Evans–Lyons), but it is (a) mostly contemporaneous / daily-to-monthly and (b) bank/CLS-proprietary. Our
tick-rule OFI null is therefore *consistent* with the literature (wrong flow, not wrong method), **not** a
refutation. The two accessible upgrades multiple reports endorse: **integrated multi-level OFI / microprice
from quote sizes** (02, 05) and **metaorder-in-progress detection via order-flow long-memory** (02) — the one
order-flow phenomenon whose physics lives at minutes.

### 5. Architecture and feature-factories are not the bottleneck — labels, clock, and validation are
Reports **04, 06, 11, 14** converge hard: LightGBM ≈ GRU and both beat Transformers on this data; zero-shot
foundation models are coin-flip; entropy/Hurst/VPIN/wavelet are sign-blind (magnitude, not direction). What
*does* transfer is **(a)** changing the label (triple-barrier) and the **clock** (information-driven bars),
**(b)** online/walk-forward weight updates (the single largest credible comp lift, report 11), and **(c)** the
AFML validation stack (purged/embargoed CPCV + Deflated Sharpe / PBO) to avoid declaring a fluke after 17+
variants (06, 10, 11).

### 6. Cross-asset must be the dollar-factor's *drivers*, not more FX peers
Reports **07, 10** agree: EURUSD is ~97% dollar-factor (Verdelhan), so peer-FX lead-lag (which we did)
washes out. The untried orthogonal source is the **external USD-factor drivers at minute resolution** — US
2y/10y Treasury / Bund rate differential, ES/SPX futures, DXY-basket dislocation — conditioned on
news/overlap windows where lead-lag concentrates.

### 7. Calibration ≠ better selection (and the threshold must adapt)
Report **13** (reinforced by 06) is the key methodological myth-buster: post-hoc calibration is
order-preserving and **cannot** improve the risk-coverage curve — our OOS "threshold collapse" is a
*non-stationarity* problem, fixed by **online conformal adaptation (DtACI)** and **Mondrian reject-options**,
not by Platt/isotonic. This makes whatever thin edge exists *honest and stable*, but does not create edge.

---

## The most promising avenues for 15m EURUSD (ranked)

Ranked by (expected directional lift × credibility × feasibility given our data). Each entry: thesis · why it
could beat the ~0.527 AUC wall · new data needed · single key risk.

### 1. Signed macro-surprise event model (pre-30m drift + post-15m window)
**Thesis.** Build a *separate* event-conditioned model keyed to exact release timestamps for Tier-1 US/EZ
items, with the standardized **signed surprise** `z=(actual−consensus)/σ` as the core feature, plus pre-release
drift and immediate post-release reaction. (Reports 01, 05, 07, 08, 09, 16.)
**Why it can beat the wall.** Scheduled-event drift is an *exogenous information shock*, not endogenous flow,
so it does **not** suffer the 1–5 min microstructure decay that capped everything we tried. ABDV (2003) and
ECB WP1901 document native-15m directional structure (~half the move drifts in the 30 min *before* release).
Our prior calendar work only captured vol-*timing*; the surprise *sign* is genuinely new directional payload.
**New data.** A calendar with **consensus + actual + true wire timestamps** (Trading Economics / FXStreet /
Econoday; FRED ALFRED for vintage actuals). Cheap.
**Key risk.** Decay/arbitrage: the pre-FOMC drift *largely vanished after 2015* (Lucca–Moench follow-up) — must
re-validate stationarity separately on 2024–25, and the fast machine-readable surprise is priced in seconds
(latency wall), so the edge lives only in the slower 5–15 min continuation/fade residual, which is small.

### 2. Fixing-window directional overlay (Tokyo / ECB / WM-R 4pm)
**Thesis.** Encode minutes-to/since each fix in correct local time (DST-aware) and the cumulative pre-fix drift;
train a conditional model exploiting the documented USD-into-fix drift and post-fix reversal. (Reports 01, 10.)
**Why it can beat the wall.** Krohn–Mueller–Whelan (*JoF* 2024) document a *signed*, around-the-clock W-pattern,
significant across all G10 over 21 years, t-stats up to 9.2 — orthogonal to all 239 TA features and to our
ruled-out vol-seasonality proxy (that captured volatility, this captures *sign of drift*). Free to build.
**New data.** None — just correct fix timestamps + our existing bars (handle the 2016 ECB-fix regime break).
**Key risk.** Magnitudes are single-digit bps and **net-of-retail-spread the effect is positive only for EUR,
negative for GBP/JPY** — so accuracy may clear a bar in-window while being economically marginal; coverage is
small (a few windows/day).

### 3. Selective prediction re-pointed at mechanistic windows + made reliable (Mondrian conformal / DtACI)
**Thesis.** Re-derive coverage from the *union of mechanistically-special states* (fix ±30m, news ±15m,
gamma/vol regime) rather than a black-box |p−0.5| gate, and wrap it in a **Mondrian label-conditional conformal
reject-option** with **DtACI online threshold adaptation**. (Reports 06, 09, 13, 15.)
**Why it can beat the wall.** This is the principled version of our single success (0.632@0.2%). The literature
predicts the edge concentrates exactly in those windows; conformal sets the error floor *a priori* (avoiding
the in-sample threshold-search bias we currently have), and DtACI stops the OOS collapse we observed.
**New data.** None — methodological; needs a temporally-contiguous calibration block + our own tick-derived
spreads for cost-aware evaluation.
**Key risk.** It *cannot manufacture signal* — if true tail edge is ~0.60–0.63 it will reliably realize
~0.60–0.63, not 0.75. At 0.2% coverage tail accuracy is high-variance; honest CIs may show 0.632 isn't
significant. Conformal's guarantee assumes exchangeability, which FX violates (hence DtACI, but it only
promises *long-run* coverage).

### 4. Integrated multi-level OFI + microprice + metaorder-in-progress detector
**Thesis.** Replace the discredited signed-10s-volume proxy with (a) **integrated multi-level OFI** and
**Stoikov microprice** from quote sizes, and (b) a **metaorder-in-progress detector** — rolling order-flow-sign
long-memory / run-length / Hawkes residual same-side intensity — that bets on continuation over a metaorder's
remaining life. (Reports 02, 05.)
**Why it can beat the wall.** Our OFI null tested a *degraded* single-level wall-clock proxy; integrated OFI and
microprice are materially better-specified. Critically, the **metaorder** physics (order-flow long memory,
power-law sign autocorrelation persisting at the 3-min scale, documented in FX spot) is the *only* order-flow
phenomenon whose timescale natively matches 15m — and we have never built it.
**New data.** None for the proxy version (our quote sizes suffice); true depth would help but is paywalled.
**Key risk.** The contemporaneous integrated-OFI lift demonstrably "decays within several minutes"
(Cont–Cucuringu–Cont) — only the metaorder-continuation extrapolation could reach 15m, and that is
**unproven for us** and hard to detect from public top-of-book data. Highest novelty, highest risk.

### 5. Rate-differential / cross-asset fair-value-gap residual (dollar-factor drivers)
**Thesis.** Causally regress EURUSD short return on contemporaneous **US-DE 2y/10y rate-futures**, a dollar
basket, and a risk factor (ES); features = the residual sign/z-score + driver *velocities*; predict 15m
mean-reversion of the residual, especially in release windows. (Reports 07, 10.)
**Why it can beat the wall.** It targets the actual structure that makes EURUSD hard (97% dollar-factor) by
modeling the factor's *drivers* rather than peer FX. Rates often reprice seconds before spot fully adjusts;
the *signed* rate move carries direction (unlike our vol-seasonality proxy).
**New data.** Intraday Treasury/Bund/ES futures (Databento, low cost) — prototype free with FRED/ECB daily
yields first.
**Key risk.** Plain unconditioned cross-asset features wash out the same way cross-*pair* did (we saw this);
the lift, if any, is concentrated in news/overlap windows and on the *continuous* (non-jump) component, so it
overlaps avenue 1 and may add little incremental.

### 6. Information-driven bars + triple-barrier relabeling, and online/walk-forward updates
**Thesis.** Re-clock the series on **imbalance/run bars** (fire when signed flow exceeds expectation), relabel
with vol-scaled **triple barriers**, and add **online weight updates** (one step per newly-revealed label) plus
multi-task auxiliary-horizon heads. (Reports 04, 06, 11, 14.)
**Why it can beat the wall.** Everything we tried was fixed-time, fixed-horizon; the *clock and label* are the
one axis we never varied. Online learning gave ~4× the lift of all feature engineering in the most analogous
competition (Jane Street 2024) and we've never tried it.
**New data.** None — all from existing ticks.
**Key risk.** These are *representation/optimization* upgrades, not new signal; the comp lift was on a dataset
with known signal, and our base SNR may be too low for them to surface anything. Triple-barrier overlap demands
strict purged/embargoed CV or it leaks.

### 7. Fractional-differentiation features + AFML validation stack
**Thesis.** Add **fractionally-differentiated** (memory-preserving, stationary) versions of the price level,
spread, USD-basket level, and basket residual — a feature class our 239 return-based (d=1) features destroy by
construction — and gate every result through **Deflated Sharpe / PBO / CPCV**. (Report 06; validation echoed by
10, 11.)
**Why it can beat the wall.** Returns are memoryless; fracdiff can expose long-memory in *levels* that returns
discard. The validation stack retroactively tells us whether our 0.632 (and 3s result) survive multiple-testing
deflation — high methodological ROI regardless.
**New data.** None.
**Key risk.** Whether fracdiff adds *directional* edge on near-efficient EURUSD is untested and the prior is
weak; most likely a small lift, and the validation work may *downgrade* our existing results.

### Explicitly low-value / avoid (so no future variant wastes cycles)
- **Plain Transformers / Informer / Autoformer / zero-shot TSFMs** for direction (beaten by linear baselines;
  coin-flip on returns). **Re-implementing DeepLOB for 15m** (microstructure dead by 5 min — our finding +
  Lucchese 2024 + TLOB all agree). **Entropy/Hurst/VPIN/wavelet as *sign* features** (provably magnitude, not
  direction; "76% wavelet" results are decomposition leakage). **Risk-reversals / gamma as a direction trigger**
  (FX gamma is vol not direction; gamma→momentum is *insignificant in currencies*). **COT at 15m**
  (weekly/3-day-lagged). **Deriv synthetic-index direction** (CSPRNG martingale — provably unpredictable).
  **Unregulated OTC binary brokers** as a data/edge source (manipulated tape). **More TA families / cross-pair
  lead-lag** (exhausted; comps confirm it won't beat LightGBM).

---

## Honest assessment

**Is 75% at 15m on liquid EURUSD plausible at all?** No — not *unconditionally*, and the evidence is
overwhelming and convergent. Five independent literatures (academic FX-LOB, order-flow, deep-learning, cross-
asset, practitioner) plus our own 0.527 wall all say liquid FX at 15m is near-efficient. The credible honest
ceilings are ~58.5% *daily* (Castillo/Guyard–Deriaz) and *lower* intraday; the only >75% numbers in the entire
corpus are (a) equities at 5-second with an order-flow peek, (b) daily with wavelet-denoising lookahead, (c)
competition leakage (recovering a shuffled index), or (d) marketing/martingale. **No credible, reproducible,
costed, multi-regime source shows always-on 75% at 15m on a major.**

**What is the realistic ceiling?** Two distinct deliverables:
- **A thin, high-precision conditional book.** On the small union of mechanistically-special bars (fix windows,
  signed-surprise news windows, expiry magnets), **plausibly 65–75% directional accuracy at very low coverage
  (~0.5–3%)**. This is the only place 75% is achievable, and it is consistent with our existing 0.632@0.2%
  pointing the same way. The avenues most likely to populate it: #1 (surprise), #2 (fix), #3 (reliable selective).
- **A modestly better all-bars selective frontier.** Realistically **~0.60–0.65 at materially higher coverage
  (5–15%)** than today's 0.2%, via regime-gated + conformal-reliable selection (#3) layered on integrated-OFI /
  metaorder (#4) and online-updated representations (#6). Not 75%, but *deployable and stable*.

**The best fallback if 15m-EURUSD stays capped.** Two, in order of evidence strength:
1. **Re-point the entire pipeline at crypto at 15m** (report 12, supported by 16). Peer-reviewed evidence
   (Nature Sci. Reports; J. Int. Money & Finance) documents BTC/alts *deviating from the random walk at
   15/30/60m*, with real (not proxy) order flow free from exchange APIs, a dominant lead asset (BTC), and
   orthogonal axes with no FX analog (funding rate, open interest, liquidations, cross-venue dispersion). This
   directly attacks our root cause — EURUSD is efficient and 97% dollar-factor; crypto is neither. **This is the
   highest-expected-value pivot if the FX ceiling holds.** (Caveat: crypto LOB imbalance also decays with
   horizon; the edge is the inefficiency + orthogonal features, not faster microstructure.)
2. **Accept the binary-economics reframe and ship the thin conditional book.** Break-even is ~57% at 0.75
   payout; our 0.632 already clears it. With conformal-reliable selection (#3) and cost-aware net-of-spread
   labels, a *stable* ~63–68% at usable (1–5%) coverage is a real, deployable strategy — the goal becomes
   *coverage and stability*, not the 75% headline.

**Bottom line.** Stop hunting an always-on 15m oracle (it isn't there). Spend the next cycles on: **(1)** the
signed-surprise and fix-window conditional models (highest-credibility orthogonal *direction* signals we never
built), **(2)** making selective prediction reliable and mechanistic (conformal/DtACI + regime gates), **(3)**
the integrated-OFI/metaorder upgrade to our discredited flow proxy, and **(4)** a crypto pivot as the fallback
where a longer-lived 15m edge actually exists. Gate every result through Deflated Sharpe / CPCV before
believing it.

---

### Top 5 cross-cutting takeaways
1. **75% all-bars at 15m on EURUSD is not credibly achievable** — five independent literatures + our own wall
   agree it's near-efficient; every external >75% is equities-5s, denoising-leakage, comp-leakage, or marketing.
2. **The reframe is conditional/selective, not unconditional** — and binary economics shows **break-even is
   ~57%, not 75%**, so our 0.632 is already economically live; the real problem is **coverage and stability**.
3. **The same handful of mechanistic windows recur across reports** — **fixings**, **signed macro surprises**
   (not vol-timing), and **expiry magnets** — these are the orthogonal *directional* signals we genuinely never
   built, and the only place 65–75% is plausible (at low coverage).
4. **Our OFI null was a bad estimator, not a dead signal** — integrated multi-level OFI + microprice, and
   especially a **metaorder-in-progress detector** (order-flow long memory, the one phenomenon that lives at
   minutes), are the real untried upgrades; architecture and complexity-feature factories are *not* the bottleneck.
5. **Calibration can't fix selection; online conformal adaptation can** — and the best fallback if FX stays
   capped is **re-pointing the pipeline at crypto at 15m**, which is documented less-efficient with real order
   flow and orthogonal funding/OI/liquidation features that have no FX analog.
