# Probability Calibration, Conformal Prediction & Abstention for Reliable Selective Accuracy at 15m FX

> Research vector 13 for binary-algo. Goal context: we already see ~0.632 OOS directional
> accuracy at ~0.2% coverage on EURUSD 15m, but the selective threshold is over-confident OOS
> and collapses. This report is about making selective prediction *reliable* — i.e. choosing an
> abstention rule whose realized accuracy on 2024–25 (and held-out 2026) matches what we promise
> on val, with finite-sample guarantees where possible.

---

## TL;DR (most actionable findings for our 15m FX goal)

- **Calibration alone will NOT raise your selective accuracy.** Post-hoc calibration (Platt /
  isotonic / temperature / beta) is *order-preserving*: it remaps probabilities monotonically, so
  it cannot change the *ranking* of samples by confidence and therefore cannot change your
  risk–coverage curve or AURC at all (Cattelan & Silva 2023; Feng et al. "Calibrated Selective
  Classification" 2022). It fixes the *number* (your "75%" promise being honest) but not *which*
  bets you take. This is the single most important misconception to retire.

- **The honest tool for a *guaranteed* accuracy floor is Mondrian (label-conditional) conformal
  prediction used as a reject option.** Accept only *singleton* prediction sets, abstain on
  empty/both-label sets. With a calibration set this gives a **distribution-free, finite-sample
  upper bound on the error rate among the bets you actually place** (Johansson et al. 2023,
  "Confidence Classifiers with Guaranteed Accuracy or Precision"; Zhang & Roth 2025, "Classification
  with Reject Option: Distribution-free Error Guarantees via Conformal Prediction"). Set ε=0.25 and,
  *if exchangeability held*, you'd get ≥75% accuracy on the covered set by construction.

- **But exchangeability does NOT hold for FX**, so the conformal guarantee is *not* free — it must
  be earned with a temporally-aware calibration scheme. Use **Adaptive Conformal Inference (ACI)**
  and especially **DtACI** (Gibbs & Candès 2021/2022) or **EnbPI** (Xu & Xie 2021), which maintain
  long-run coverage *under distribution shift* by online-updating the miscoverage level
  α_{t+1}=α_t+γ(α−err_t). This is the correct answer to "a threshold that generalizes OOS instead
  of collapsing": don't pick a fixed threshold — *track* it online.

- **Venn–ABERS predictors** (Vovk & Petej 2014) are the best calibration choice for our binary
  problem: they are the only method giving *provably* calibrated probabilities via two isotonic
  fits, and they emit a probability *interval* [p0,p1] whose *width* is itself an abstention signal
  (wide interval = unreliable region). Use the interval width as a second gate on top of |p−0.5|.

- **A fixed |p−0.5| threshold tuned on val is exactly the failure mode the literature warns about.**
  Under shift, calibration degrades and "selective classification may pass high-confidence errors
  while discarding easy cases" (Cattelan & Silva 2023). The fix is *Mondrian/group-conditional*
  validity (separate thresholds per class, per session/vol-regime bucket) plus *online* threshold
  adaptation, not a single global cutoff.

- **Selective classification has a brutal small-sample variance problem at low coverage** —
  precisely our regime (0.2% coverage). The empirical accuracy at the extreme tail is estimated
  from a handful of samples and is high-variance; the "Overcoming Common Flaws in the Evaluation of
  Selective Classification" (Traub et al., NeurIPS 2024) and AURC finite-sample-estimator papers
  show tail estimates are unstable. Budget for *much* wider confidence intervals on tail accuracy,
  and prefer methods that give the floor *before* you see the test set (conformal) over methods that
  pick the tail *after* (threshold search → optimistic bias / leakage).

- **Train an explicit reject option, don't bolt it on.** SelectiveNet (Geifman & El-Yaniv 2019) and
  cost-based "learning to reject" optimize prediction *and* abstention jointly for a target
  coverage, yielding a better risk–coverage frontier than post-hoc thresholding of a vanilla net.
  Worth trying as a head on the GRU you already have.

- **Realistic expectation:** these methods make your selective accuracy *honest and stable* OOS;
  they do not manufacture signal. If your true edge at the |p−0.5| tail is ~0.60–0.63, conformal
  abstention will let you *reliably realize* ~0.60–0.63 with a guaranteed floor — it will not turn
  it into 0.75. Reaching 0.75 still requires a stronger orthogonal signal (other vectors). The win
  here is *trustworthy* selective deployment and not blowing up live.

---

## Key findings (detailed, with inline citations)

### 1. Calibration is order-preserving → it cannot move the risk–coverage curve

The deepest and most counter-intuitive result for our project: **post-hoc calibration cannot improve
selective classification.** Standard calibrators (Platt/logistic, isotonic, temperature, beta,
Dirichlet) are monotone transforms of the score. Selective performance — the risk–coverage (RC)
curve and its summary AURC — depends only on the *ranking* of samples by confidence. A monotone map
preserves ranking, hence preserves the RC curve and AURC exactly. "Standard calibration … cannot
reduce ranking error because it preserves score ordering" and "AURC … depends on that ranking …
temperature scaling is designed to preserve the original ranking … cannot change the AURC"
(Cattelan & Silva, "How to Fix a Broken Confidence Estimator," arXiv:2305.15508, 2023; AURC
characterization, arXiv:2410.15361, 2024).

Implication for us: our over-confidence OOS is a *calibration* defect (the 0.85 we print is really
0.60), and calibration *will* fix the printed number so we stop over-betting. But if our problem is
*which* trades we select (RC frontier), calibration is the wrong tool — we need a better confidence
*estimator* (different score / feature) or a guaranteed-coverage wrapper.

There is one caveat the same literature raises: calibration *can* help selective performance
indirectly when the base confidence estimator is "broken" (e.g. softmax with bad logit norms);
fixing the estimator (MaxLogit / logit-L2-normalization / "p-norm" tricks) *does* move the RC curve
because it changes ordering, not just scale (Cattelan & Silva 2023). So: try alternative confidence
scores (margin, entropy, Venn–ABERS interval width, ensemble disagreement), not just calibrators.

### 2. Conformal classification as a reject option = a distribution-free accuracy floor

In binary conformal classification you compute, for a new x, a p-value for label 0 and for label 1
against a held-out calibration set of non-conformity scores. At significance ε the prediction *set*
contains each label whose p-value > ε. Three outcomes: **{0}** or **{1}** (singleton → confident
bet), **{0,1}** (both → uncertain → abstain), **{}** (empty → outlier → abstain). "By accepting
only the singleton predictions, we turn CP into a binary classifier with reject option," and one can
"derive theoretical guarantees on the resulting error rate" (Zhang & Roth, "Classification with
Reject Option: Distribution-free Error Guarantees via Conformal Prediction," arXiv:2506.21802 / Mach.
Learn. with Appl. 2025). The error rate among singletons is controlled by ε with finite-sample
validity *under exchangeability*.

The **Mondrian / label-conditional** variant is what gives a *guaranteed accuracy or precision*:
calibrate non-conformity scores *separately per class*, so the error bound holds *within each
predicted class* rather than only marginally (Johansson, Boström, Löfström, "Confidence Classifiers
with Guaranteed Accuracy or Precision," PMLR v204, 2023). This is exactly the right framing for a
directional bet: "guaranteed accuracy" = "of the times I said UP, at most ε were wrong." Set ε=0.25
→ ≥75% accuracy on covered UP-bets, ≥75% on covered DOWN-bets, *by construction* if exchangeability
holds. Coverage (how often you get a singleton) is whatever the data permits — you trade coverage
for the guaranteed floor instead of guessing a |p−0.5| cutoff.

This is strictly more honest than our current approach: we currently *search* for a tail threshold
that gives 0.632 — an in-sample optimization that is optimistically biased. Conformal *sets the
floor first* and lets coverage fall out.

### 3. Exchangeability fails for FX — so use ACI / DtACI / EnbPI, not vanilla split-conformal

The conformal guarantee rests on exchangeability of (calibration, test) scores. "Temporal
dependence, autocorrelation, and nonstationarity violate exchangeability; naïve application of
conformal prediction then loses the coverage guarantees" (Barber, Candès, Ramdas, Tibshirani,
"Conformal prediction beyond exchangeability," Ann. Statist. 2023, arXiv:2202.13415). FX 15m has
volatility clustering and regime shifts → vanilla split-conformal will *under-cover* (realized error
> ε) exactly when it matters (vol spikes / regime change), which is our observed "threshold
collapses OOS" symptom.

Three production-grade fixes:

- **Adaptive Conformal Inference (ACI)** — Gibbs & Candès, NeurIPS 2021 (arXiv:2106.00170). Track a
  single state α_t and update α_{t+1}=α_t+γ(α−err_t) where err_t=1 if the last set missed. This
  drives the *realized long-run miscoverage* to the target α "without any assumptions on the
  data-generating distribution over long-time intervals." It self-corrects when a regime shift makes
  the model over-confident: misses → α shrinks → it abstains more until accuracy recovers.

- **DtACI (dynamically-tuned ACI)** — Gibbs & Candès 2022 ("Conformal Inference for Online
  Prediction with Arbitrary Distribution Shifts," JMLR 2024). Removes the need to hand-pick γ by
  running several ACI experts with different γ in parallel and aggregating them with exponential
  reweighting on pinball loss. **This is the recommended default** — it adapts the adaptation rate
  itself, so you don't overfit γ on val.

- **EnbPI** — Xu & Xie, ICML 2021 (arXiv:2010.09107; code github.com/hamrel-cxu/EnbPI). Bootstrap
  ensemble; uses leave-one-out residuals updated with the most recent errors, "neither data-splitting
  nor refitting," adapts to seasonality/trend, "finite-sample approximately valid marginal coverage
  for time series with strongly mixing errors." Good when you want to reuse all data for training.

Benchmark caveat: a 2026 benchmark (arXiv:2601.18509) finds **no single method dominates**; ACI
reliably hit nominal 90% coverage with tight intervals, while EnbPI and SPCI *under-covered* in some
settings. So validate coverage empirically on *your* series; don't assume the guarantee.

### 4. Venn–ABERS: the right calibrator for binary FX, and a free abstention signal

Venn–ABERS predictors (Vovk & Petej, UAI 2014, arXiv:1211.0025) are the only calibration method with
a *theoretical validity guarantee* for the probabilities themselves (multiprobability prediction):
they fit isotonic regression twice on the calibration set (once assuming the test label is 0, once
assuming 1) and output an interval [p0,p1]. The true calibrated probability is provably bracketed.
Two practical wins for us:

1. **Calibrated probabilities with guarantees**, robust to the "transformers/boosters output extreme
   0/1 probabilities" over-confidence we see (Manokhin, valeman.medium.com; venn-abers GitHub
   ip200/venn-abers).
2. **Interval width = epistemic-uncertainty / abstention signal.** "In precarious scenarios
   Venn–ABERS … sends an 'alert' by expanding the interval width." A wide [p0,p1] flags a region the
   calibration set covered poorly → abstain even if the point probability looks confident. This is an
   *orthogonal* abstention axis to |p−0.5|.

Inductive (IVAP) and cross (CVAP) variants are cheap; CVAP uses cross-folds for stability on smaller
calibration sets (Manokhin 2017, PMLR v60).

### 5. Train the reject option in, don't threshold after

- **SelectiveNet** (Geifman & El-Yaniv, ICML 2019, arXiv:1901.09192): a network with three heads —
  prediction, selection (g(x)∈[0,1]), and auxiliary — trained jointly to minimize selective risk at
  a *target coverage c* with a coverage constraint. Beats post-hoc confidence thresholding on the RC
  frontier because the model *learns where to abstain* rather than reusing a confidence score never
  optimized for rejection. Directly applicable as a new head on our existing GRU.

- **Selective classification theory** (Geifman & El-Yaniv, "Selective Classification for Deep Neural
  Networks," NeurIPS 2017; El-Yaniv & Wiener 2010): the SGR algorithm picks a threshold giving a
  *PAC-style bound* on test risk at a chosen confidence — i.e. it tells you, with a stated
  probability, that test selective risk ≤ target. This is a principled alternative to eyeballing the
  RC curve. Use it to size the honest confidence interval on your tail accuracy.

- **Calibrated Selective Classification** (Fisch/Feng et al., TMLR 2022, arXiv:2208.12084): trains a
  *selective* model to be calibrated *on the accepted region* (not marginally). Relevant because we
  want the printed probability to be honest *specifically on the trades we take* — marginal
  calibration is not enough.

### 6. The low-coverage tail is statistically treacherous (our exact regime)

At 0.2% coverage the tail accuracy is computed from very few samples. "Overcoming Common Flaws in the
Evaluation of Selective Classification Systems" (Traub et al., NeurIPS 2024, arXiv:2407.01032) and the
AURC finite-sample-estimator paper (arXiv:2410.15361, 2024) show that low-coverage RC estimates are
high-variance and that naive AURC estimators are biased. Two consequences:

- Our 0.632 @ 0.2% is a *point estimate with a wide CI*; on a fresh year it can swing materially.
- Choosing the operating point by searching the val RC curve for the best tail = selection bias /
  leakage. Conformal/ACI avoids this by fixing the error target a priori and adapting online.

### 7. Conformal Risk Control generalizes the floor beyond 0/1 error

Angelopoulos, Bates et al., "Conformal Risk Control" (ICLR 2024, arXiv:2208.02814) extends conformal
guarantees from miscoverage to *any monotone loss*. You can calibrate the abstention threshold to
bound, distribution-free, the *expected trading loss* or a custom asymmetric error (a wrong UP costs
differently than a wrong DOWN), not just raw error rate — minimizing abstention subject to that risk
cap. "Selective Conformal Risk Control" (arXiv:2512.12844, 2025) fuses this with selective
classification explicitly.

---

## Concrete techniques / features / architectures to try

Ordered roughly by expected value-for-effort on the 15m FX pipeline.

1. **Mondrian (label-conditional) inductive conformal reject option — primary deliverable.**
   - Hold out a *temporally contiguous, recent* calibration block (e.g. last N weeks of train, or a
     rolling window) — NOT a random split (preserves time order).
   - Non-conformity score for a sample with model prob p and true label y: `s = 1 − p_y` (or use the
     model's margin / logit). Compute *separate* calibration score distributions for y=0 and y=1.
   - For a new x: p-value_0 = rank of (1−p0) among class-0 calib scores; same for class-1. Predict
     {label : p-value > ε}. Bet only on singletons; ε=0.25 targets ≥75% covered accuracy.
   - Implementation: `crepes` (Boström, KTH; supports Mondrian classifiers out of the box) or
     `MAPIE` (`mapie.classification`, supports Mondrian via groups) or `nonconformist`.
   - Report realized coverage AND realized covered-accuracy on val 2022–23, then test 2024–25 — the
     gap between target ε and realized error is your exchangeability-violation tax.

2. **Wrap (1) in DtACI for online threshold adaptation — the OOS-generalization fix.**
   - Instead of fixed ε, run DtACI to online-update the effective miscoverage so realized long-run
     accuracy tracks the 75% target *as regimes change* in 2024–25.
   - Update rule per the ACI family; DtACI auto-tunes γ via expert aggregation. Reference impl:
     `AdaptiveConformal` R package (Susmann et al.) or port from Zaffran et al. code
     (github.com/mzaffran/AdaptiveConformalPredictionsTimeSeries).
   - This is the concrete answer to "a confidence threshold that generalizes": it is *not a constant*;
     it is a tracked state that abstains more during regimes where the model is silently degrading.

3. **Venn–ABERS calibration + interval-width gate.**
   - Fit IVAP/CVAP on the calibration block (`venn-abers` pip package, or MAPIE Venn–ABERS).
   - Bet only when BOTH: (a) |p̄−0.5| above a level, AND (b) interval width (p1−p0) below a level.
     Gate (b) catches regions the calibration set covered poorly — orthogonal to (a).
   - Cheap, drop-in on top of LightGBM/GRU; immediately fixes the over-confident printed probability.

4. **Per-bucket (Mondrian) conditioning on regime, not just class.**
   - Define groups by {session × volatility-regime} (e.g. London/NY/Asia × low/med/high realized
     vol). Calibrate conformal scores *within each group*. This gives group-conditional validity, so
     a single global threshold can't be high-confidence-wrong in, say, high-vol NY where the model is
     worst. We already have session/vol features, so the buckets are free.

5. **SelectiveNet head on the existing GRU.**
   - Add selection head g(x) + auxiliary head; train with the coverage-constrained selective loss for
     a target coverage (sweep 1–10%). Compare RC frontier vs post-hoc |p−0.5|. If it beats
     thresholding, it means the abstention region is learnable from features — a genuine improvement,
     not just relabeling.

6. **Confidence-estimator swap (because calibration won't help, ordering might).**
   - Replace |p−0.5| with: ensemble *disagreement* (variance across LGBM/XGB/Cat/GRU), Venn–ABERS
     width, conformal p-value margin, or logit-norm tricks (MaxLogit) from Cattelan & Silva 2023.
     These *change the ranking* and can move the RC curve where calibration cannot.

7. **Conformal Risk Control with asymmetric / PnL loss.**
   - If costs are asymmetric (spread, slippage), calibrate the abstention threshold to bound expected
     monetary loss rather than error rate (`arXiv:2208.02814`). Lets you abstain *less* when the
     payoff structure tolerates it.

8. **Evaluation discipline (mandatory).**
   - Compute bootstrapped CIs on tail accuracy (Traub et al. 2024). Report AURC with a low-bias
     estimator. Treat 2026 as a true lockbox — run the *frozen* DtACI+Mondrian rule on it exactly
     once.

---

## Reported results & CREDIBILITY assessment

**Credible / reproducible (peer-reviewed, code available, guarantees are theorems not backtests):**

- **Conformal reject-option error bounds** (Zhang & Roth 2025; Johansson et al. 2023; Vovk's CP
  framework): the finite-sample error/coverage guarantee is a *mathematical theorem* under
  exchangeability — not a backtest that can overfit. **High credibility**, but the guarantee is
  *conditional on exchangeability*, which FX violates; the honest version is "guaranteed under
  exchangeability, empirically validated under shift." Do not present the floor as unconditional.

- **ACI / DtACI / EnbPI** (Gibbs & Candès NeurIPS 2021 / JMLR 2024; Xu & Xie ICML 2021): long-run
  coverage under arbitrary distribution shift is proven; widely reproduced; reference code public.
  **High credibility.** Caveat: "long-run" — they don't promise *conditional* (per-timestep)
  coverage, and a 2026 benchmark (arXiv:2601.18509) shows EnbPI/SPCI can under-cover in finite
  samples. Verify on your data.

- **Venn–ABERS validity** (Vovk & Petej 2014; Manokhin 2017): probability validity is a theorem.
  **High credibility.**

- **SelectiveNet / SGR** (Geifman & El-Yaniv 2017, 2019): RC-frontier improvements reproduced on
  standard vision/tabular benchmarks; PAC risk bound is principled. **Credible**, though gains were
  shown on i.i.d. benchmarks, not nonstationary finance — expect smaller, noisier gains for us.

- **Calibration ≠ better selection** (Cattelan & Silva 2023; AURC papers 2024): this is a *proof*
  (monotone maps preserve ranking) plus broad empirics. **High credibility** and directly saves us
  from a dead-end (chasing calibration to fix selection).

**Hype / overfit / leakage-prone (treat with skepticism):**

- **"Trading via Selective Classification" (Chalkidis & Savani, arXiv:2110.14914, 2021):** the
  cleanest *finance* application of selective classification (abstain when low-confidence, trade
  when confident, report accuracy↑ as coverage↓). The *method* is sound and on-thesis. **But**
  finance selective-trading papers are highly prone to: (i) tuning the abstention threshold on the
  same period they report (selection bias → optimistic tail accuracy — exactly our worry); (ii)
  ignoring transaction costs that eat the thin-coverage edge; (iii) survivorship/look-ahead in
  features. I could not fetch the full PDF this session (host interception), so I cannot verify their
  OOS protocol or numbers — **treat its headline accuracy as unverified and likely optimistic until
  the threshold-selection protocol is read.**

- **Blog/Medium "conformal trading" posts (Manokhin et al.):** good *pedagogy* on Venn–ABERS/CP
  mechanics (credible) but trading-PnL claims in practitioner posts are typically illustrative, not
  rigorously OOS-validated. Use for technique, not for expected-return claims.

- **Generic "75%/90% accuracy with conformal" forum claims:** none found that survive scrutiny;
  conformal does not create signal, it quantifies/abstains. Any such claim is either (a) measuring
  *coverage* (90% coverage ≠ 90% directional accuracy — easy to conflate) or (b) leaked. Be explicit
  internally: **conformal gives a guaranteed *error rate among taken bets*, achieved by abstaining;
  it cannot exceed your model's underlying conditional separability.**

**Leakage risks specific to us:**
- Random (not temporal) calibration split → leaks future into calibration → optimistic.
- Re-using val to *both* pick the operating threshold *and* report tail accuracy → selection bias.
  Conformal/ACI mitigates by fixing the target a priori.
- Calibrating on a quiet regime, deploying in a volatile one → silent under-coverage (the ACI fix).

---

## Data sources needed

This vector is **methodological** — it needs essentially no new market data, only correct *use* of
data we already have.

- **A temporally-held-out calibration block** carved from existing train/val (we already have 2012–25
  10s bars + ticks). *Free; already in hand.* Fidelity: must be contiguous and recent relative to the
  prediction point, not random-shuffled.
- **Realistic transaction-cost / spread data** to validate that thin-coverage selective accuracy
  survives costs. We have raw bid/ask quotes + sizes → effective spread is *derivable from our own
  tick data*. *Free; in hand.* This is critical: a 0.63-accuracy edge at 0.2% coverage can be
  entirely consumed by spread.
- **Software (all free, open-source):**
  - `crepes` (Mondrian conformal classifiers; Boström) — github.com/henrikbostrom/crepes
  - `MAPIE` (conformal classification + Venn–ABERS + Mondrian groups) — mapie.readthedocs.io
  - `venn-abers` (IVAP/CVAP) — github.com/ip200/venn-abers
  - `nonconformist` (classic split/cross CP) — github.com/donlnz/nonconformist
  - EnbPI reference — github.com/hamrel-cxu/EnbPI
  - ACI/AgACI time-series — github.com/mzaffran/AdaptiveConformalPredictionsTimeSeries;
    `AdaptiveConformal` R package
- **No paid data required.** No alt-data needed for this vector.

---

## Relevance & priority for OUR project

How this interacts with what we've already ruled out: our prior work *built signals* (239 TA
features, cross-pair, OFI, ensembles, stat-arb, exogenous peers, calendar) and found the edge caps at
~0.52 AUC with a thin reliable tail at 0.632. This vector does **not** add signal — it makes the
*deployment* of whatever thin edge exists **honest, stable, and guaranteed-floored OOS**, and fixes
the specific "over-confident, threshold-collapses-OOS" pathology you flagged. It is the right
*risk-management / reliability* layer, complementary to (not a substitute for) the signal-hunting
vectors.

**HIGH priority**
- **Mondrian label-conditional conformal reject option (technique 1)** — directly replaces the
  ad-hoc |p−0.5| tail search with a guaranteed error floor; low effort with `crepes`/`MAPIE`; fixes
  the selection-bias/leakage in how we currently pick the operating point. *Do this first.*
- **DtACI online threshold adaptation (technique 2)** — the literal fix for "a threshold that
  generalizes OOS instead of collapsing." Medium effort; biggest reliability gain under 2024–25
  regime shifts.
- **Venn–ABERS calibration + width gate (technique 3)** — cheap, drop-in, fixes printed-probability
  over-confidence and adds an orthogonal abstention axis. *Do this alongside (1).*
- **Cost-aware evaluation from our own tick spreads** — non-negotiable sanity gate; many thin-edge
  selective strategies die here.

**MED priority**
- **Per-regime Mondrian buckets (technique 4)** — uses features we already have; meaningful if model
  quality varies by session/vol (it does).
- **SelectiveNet head on the GRU (technique 5)** — could genuinely improve the RC frontier if the
  abstain region is learnable; more engineering, uncertain payoff.
- **Confidence-estimator swap / ensemble disagreement (technique 6)** — the only family that can move
  the RC curve (since calibration can't); worth a focused experiment.

**LOW priority**
- **Conformal Risk Control with custom PnL loss (technique 7)** — elegant, but only worth it after
  the basic floor + cost model are in place and asymmetry is shown to matter.
- **Chasing additional calibration methods (Platt/isotonic/temperature/beta/Dirichlet) to raise
  accuracy** — explicitly **de-prioritized**: proven order-preserving, will *not* raise selective
  accuracy. Use one calibrator (Venn–ABERS) for honesty and stop.

**Bottom line on the 75% goal:** this vector makes a *reliable* selective strategy that *honestly*
realizes your true tail edge with a guaranteed floor and no OOS collapse. If that true edge is
~0.60–0.63, you'll get a *trustworthy* ~0.60–0.63 — valuable for safe deployment, but **not a path to
75% by itself.** 75% still needs a stronger orthogonal signal from the other research vectors;
conformal/abstention is how you'd *safely harvest* it once found.

---

## Sources (annotated)

1. **Cattelan & Silva, "How to Fix a Broken Confidence Estimator: Evaluating Post-hoc Methods for
   Selective Classification with Deep Neural Networks"** (arXiv:2305.15508, 2023) —
   https://arxiv.org/html/2305.15508 — *Proves/shows calibration is order-preserving so it can't help
   selection; fixing the confidence estimator can. The key myth-buster for us.*

2. **"A Novel Characterization of the Population AURC and Rates of Finite Sample Estimators"**
   (arXiv:2410.15361, 2024) — https://arxiv.org/pdf/2410.15361 — *Why AURC depends only on ranking,
   and why low-coverage tail estimates are high-variance/biased — our 0.2%-coverage regime.*

3. **Traub et al., "Overcoming Common Flaws in the Evaluation of Selective Classification Systems"**
   (NeurIPS 2024, arXiv:2407.01032) — https://arxiv.org/abs/2407.01032 — *Correct evaluation of RC
   curves; warns about exactly the optimistic tail-estimate trap we risk.*

4. **Zhang & Roth (al.), "Classification with Reject Option: Distribution-free Error Guarantees via
   Conformal Prediction"** (arXiv:2506.21802; Mach. Learn. with Appl. 2025) —
   https://arxiv.org/abs/2506.21802 — *Singleton-conformal-set = binary classifier with reject
   option; derives the error-rate guarantee we'd use as our accuracy floor.*

5. **Johansson, Boström, Löfström, "Confidence Classifiers with Guaranteed Accuracy or Precision"**
   (PMLR v204, COPA 2023) — https://proceedings.mlr.press/v204/johansson23a/johansson23a.pdf —
   *Mondrian/label-conditional CP to guarantee per-class accuracy or precision by abstaining; the
   precise "guaranteed ≥75% on covered UP-bets" construction.*

6. **Gibbs & Candès, "Adaptive Conformal Inference Under Distribution Shift"** (NeurIPS 2021,
   arXiv:2106.00170) — https://arxiv.org/abs/2106.00170 — *ACI: online α-update for long-run coverage
   under arbitrary shift. The core "threshold that adapts instead of collapsing" method.*

7. **Gibbs & Candès, "Conformal Inference for Online Prediction with Arbitrary Distribution Shifts"
   (DtACI)** (JMLR 2024, arXiv:2208.08401) — https://arxiv.org/abs/2208.08401 — *DtACI auto-tunes the
   learning rate γ via expert aggregation; recommended default so we don't overfit γ on val.*

8. **Xu & Xie, "Conformal Prediction Interval for Dynamic Time-Series" (EnbPI)** (ICML 2021,
   arXiv:2010.09107; code github.com/hamrel-cxu/EnbPI) — https://arxiv.org/abs/2010.09107 —
   *Ensemble batch PI for time series without refitting; the no-data-split alternative to ACI.*

9. **Barber, Candès, Ramdas, Tibshirani, "Conformal Prediction Beyond Exchangeability"** (Ann.
   Statist. 2023, arXiv:2202.13415) — https://arxiv.org/abs/2202.13415 — *Why naive CP loses
   guarantees under temporal dependence; weighting fixes for non-exchangeable FX data.*

10. **"Conformal Prediction Algorithms for Time Series Forecasting: Methods and Benchmark"**
    (arXiv:2601.18509, 2026) — https://arxiv.org/html/2601.18509 — *Head-to-head benchmark: ACI
    robustly hits nominal coverage; EnbPI/SPCI can under-cover. Validate the guarantee on your data.*

11. **Vovk & Petej, "Venn–Abers Predictors"** (UAI 2014, arXiv:1211.0025) —
    https://arxiv.org/abs/1211.0025 — *Provably-calibrated binary probabilities via double isotonic
    regression; the [p0,p1] interval doubles as an abstention signal.*

12. **Manokhin, "Multi-class probabilistic classification using inductive and cross Venn–Abers
    predictors"** (PMLR v60, 2017) — https://proceedings.mlr.press/v60/manokhin17a.html — *IVAP/CVAP
    algorithms and the practitioner case for Venn–ABERS over Platt/isotonic in trading.*

13. **Geifman & El-Yaniv, "SelectiveNet: A Deep Neural Network with an Integrated Reject Option"**
    (ICML 2019, arXiv:1901.09192) — https://arxiv.org/abs/1901.09192 — *Jointly learns prediction +
    abstention for a target coverage; better RC frontier than post-hoc thresholding. Try as a GRU
    head.*

14. **Geifman & El-Yaniv, "Selective Classification for Deep Neural Networks"** (NeurIPS 2017,
    arXiv:1705.08500) — https://arxiv.org/pdf/1705.08500 — *SGR algorithm: PAC-style bound on
    selective test risk at a chosen confidence — principled threshold selection.*

15. **Feng/Fisch et al., "Calibrated Selective Classification"** (TMLR 2022, arXiv:2208.12084) —
    https://arxiv.org/abs/2208.12084 — *Calibrate specifically on the accepted region; marginal
    calibration isn't enough for honest probabilities on the trades you actually take.*

16. **Angelopoulos, Bates et al., "Conformal Risk Control"** (ICLR 2024, arXiv:2208.02814) —
    https://arxiv.org/abs/2208.02814 — *Extends the conformal floor from miscoverage to any monotone
    loss → calibrate abstention to bound expected PnL/asymmetric error.*

17. **Chalkidis & Savani, "Trading via Selective Classification"** (arXiv:2110.14914, 2021) —
    https://arxiv.org/pdf/2110.14914 — *Closest finance application (abstain-when-unsure trading).
    On-thesis method but headline numbers UNVERIFIED this session — scrutinize its threshold-selection
    protocol and cost treatment for optimistic bias before trusting.*

18. **Angelopoulos & Bates, "A Gentle Introduction to Conformal Prediction and Distribution-Free
    Uncertainty Quantification"** (arXiv:2107.07511) — https://arxiv.org/abs/2107.07511 — *Best
    practical primer; implementation recipes for split/Mondrian CP.*

19. **Boström, "Conformal Prediction in Python with crepes"** (PMLR/COPA 2024) —
    https://github.com/henrikbostrom/crepes — *The library to implement Mondrian conformal
    classifiers directly on our existing models.*

20. **MAPIE library** (arXiv:2207.12274; mapie.readthedocs.io) —
    https://mapie.readthedocs.io/en/latest/ — *Conformal classification + Venn–ABERS + Mondrian
    groups + time-series, one toolkit; fastest path to a working prototype.*
