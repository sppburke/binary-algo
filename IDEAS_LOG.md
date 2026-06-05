# IDEAS LOG — transferable prediction levers (GENERIC)
SCOPE: GENERIC (currency/timeframe-agnostic). Idea→experiment backlog of transferable levers + mechanisms +
falsifier TEMPLATES. **No per-key incumbents/numbers here** — those live in the per-key backlogs
`sweeps/<PAIR>_<tf>_backlog.md` and results in `<PAIR>_RESULTS.md`. See `REPO_MAP.md`.
Generic methods: `METHODS_CATALOG.md`; permutation menu: `SWEEP_MATRIX.md`; cross-key theory: `THEORY.md`.

**Tested-on-keys pointers (where these levers were instantiated):**
- EURUSD · 1m  → `sweeps/EURUSD_1m_backlog.md`  (+ `EURUSD_RESULTS.md` §60s) — **CLOSED 2026-06-03: both sides certified-or-exhausted, 2 discovery rounds DRY**
- EURUSD · 5m  → `sweeps/EURUSD_5m_backlog.md`  (+ `EURUSD_RESULTS.md`)
- EURUSD · 2m  → `sweeps/EURUSD_2m_backlog.md`
- EURUSD · 10m → `sweeps/EURUSD_10m_backlog.md`  (+ `EURUSD_RESULTS.md` §10m) — **CLOSED 2026-06-04: BOTH sides certified (cross-pair `EURUSD.m10xp.v1` UP p10 .586 / DOWN p10 .568, 15/15), improve+discover loops DRY (2 rounds, 11 levers)**
- EURUSD · 15m → `sweeps/EURUSD_15m_backlog.md`  (deriv-tradeable binary; Q1 = magnitude×direction gate, user-queued 2026-06-03)
- EURUSD · 30m → `sweeps/EURUSD_30m_backlog.md`

**Cross-pair direction-edge HORIZON GRADIENT (generic, confirmed 2026-06-04):** the cross-pair USD-common-factor sign edge is monotone in horizon — **none@60s → UP-only@5m → BOTH sides@10m & 15m; null <5m (2m)**. Mechanism: the informed/jump component of moves (esp. DOWN) averages out as the horizon lengthens, so the common-factor SIGN becomes forecastable at ≥10m; below 5m the channel is jump/noise-dominated. CONCURRENT cross-pair (windows ending at t) is the carrier; strictly-LAGGED lead-lag is dominated. **For a new (currency, ≥10m) key, the cross-pair refit-CPCV side-split is the #1-prior lever; expect BOTH sides to certify.**

**Loss/label/gate re-engineering does NOT beat the gated cross-pair sign ≥5m (generic, reinforced [EURUSD·10m] 2026-06-04):** with BOTH 10m sides already certified by the plain-BCE cross-pair book, 11 distinct improve/discover levers ALL failed to beat it on the binding 2025 year — magweight |ret|-reweight, GMADL sign-coupled loss, residual-relabel, ACI gate, asym-class-weight specialist, isotonic calibration, cross-horizon 10m×15m blend (pred corr .957=same signal), lagged lead-lag, queue/triple-barrier, intraday-momentum. They collapse on the binding-regime wall: a wrapper/loss/relabel cannot create SIGN the regime has erased. **Lesson — once the cross-pair book certifies a ≥10m key, the improve loop is near-certainly dry; run the cross-product with fast-KILL falsifiers to confirm, but the redirect for a HIGHER number is external data, not another loss/gate variant.** Same as 5m (leader not unseated) + 15m.

**Discovery-round vetting outcome (generic, mechanism-level — 2026-06-03):** two adversarial discovery rounds over the
UNTESTED sign-aware microstructure family (OF-surprise/MRR residual, propagator past-sign, DAR sign-AR, queue-imbalance
p_up, metaorder continuation, depletion-asymmetry, microprice, event-clock, idiosyncratic-residual-flow, long-OF-history)
and the cross-horizon/calendar/UP-cert/recent-lit angles found **every candidate subsumed or sign-invariant** at the
sub-minute horizon. Mechanism: these reparameterize the signed-flow channel that decays to noise by ~60s (raw OFI/CKS/
per-side-flow all null, online-ARF keystone ~0.50) and/or are functions of the return *distribution* not its *signed
order* (sign-invariance theorem [[THEORY]]). Treat the whole microstructure-sign family as **subsumed for ≤2m direction**
unless a genuinely NEW signed representation appears. (Per-key kill evidence: `sweeps/EURUSD_1m_backlog.md` discovery rounds.)

---

Goal: break the ~0.52-AUC time-bar ceiling by (a) grouping data non-temporally and (b) applying math from OUTSIDE
finance, especially **complexity/predictability measures used as GATES** ("avoid losers" — only bet when the regime
is measurably predictable). Honest rule: any winner must hold across TEST24/TEST25/OOS26 (corr(VAL,OOS) was -0.54).

## UNIFYING HYPOTHESIS
Direction is ~0.52 predictable ON AVERAGE, but the average mixes predictable + unpredictable regimes. Many non-financial
math tools MEASURE local predictability/determinism. Gate the bet to LOW-complexity / HIGH-determinism states → refuse
the coin-flip "losers" → raise accuracy on what remains. This directly fuses "avoid losers" + "outside-the-box math".

## A. COMPLEXITY / PREDICTABILITY GAUGES → GATES (highest priority, "avoid losers")
1. **Permutation entropy (Bandt-Pompe 2002)** — ordinal-pattern complexity of a return window. LOW PE = more
   deterministic → tradeable. GATE: bet only when PE < threshold. [implement from scratch, fast]
2. **Weighted permutation entropy / amplitude-aware PE** — PE that accounts for amplitude (Fadlallah 2013).
3. **Recurrence Quantification Analysis (RQA)** — determinism (DET), laminarity (LAM), trapping time from the
   recurrence plot of the phase-space embedding (Marwan). HIGH DET = predictable → gate. [from nonlinear dynamics]
4. **Detrended Fluctuation Analysis / Hurst exponent (Peng 1994)** — H>0.5 trending (momentum tradeable), H<0.5
   mean-reverting (reversal tradeable), H≈0.5 random (skip). GATE + direction selector.
5. **Sample / multiscale entropy (Richman-Moorman)** — regularity gauge; low = predictable.
6. **Lempel-Ziv complexity** — compressibility of the symbolized return sequence; low = predictable.
7. **Largest Lyapunov exponent (Rosenstein)** — local divergence rate; low = predictable horizon.
8. **0-1 test for chaos (Gottwald-Melbourne)** — distinguishes regular vs chaotic dynamics.

## B. INFORMATION FLOW / CAUSALITY (volume→price)
9. **Transfer entropy (Schreiber 2000)** — directed info flow volume→price; high TE(vol→price) → volume leads, gate/direction.
10. **Convergent Cross Mapping (Sugihara 2012, from ecology)** — nonlinear causality detection in dynamical systems.
11. **Partial information decomposition** — synergy/redundancy of multiple drivers.

## C. PATH / GEOMETRY / TOPOLOGY
12. **Path signatures (rough path theory, Terry Lyons)** — truncated signature of the (time,price,volume) path as
    ML features; captures order + signed area (lead-lag) of the path. [iisignature/signatory or compute level-2 by hand]
13. **Topological Data Analysis / persistent homology (Gidea-Katz 2018 applied to crashes)** — persistence landscapes
    of the time-delay embedding; topological change precedes regime shifts. [ripser/gudhi]
14. **Visibility graphs (Lacasa 2008)** — map a return window to a graph; degree distribution / assortativity =
    regime features (random vs fractal vs periodic). [from scratch]
15. **Optimal transport / Wasserstein distance** — distance between recent vs historical return distributions → regime-shift gauge.

## D. POINT PROCESSES / EVENT TIME (fits volume/dollar/imbalance bars)
16. **Hawkes processes (self-exciting, from seismology/neuro)** — model up-tick vs down-tick arrivals; the conditional
    intensity ratio predicts next-move direction. Natural on EVENT bars, not clock bars.
17. **Information-driven bars (López de Prado AFML ch2)** — VOLUME / DOLLAR / tick-imbalance / run bars. Better return
    normality; sample at information arrival. [building now: vbars.py]
18. **VPIN (Easley-LdP-O'Hara)** — order-flow toxicity from volume buckets; high VPIN = informed flow = directional.

## E. SIGNAL PROCESSING / STATE-SPACE
19. **Fractional differentiation (LdP)** — stationary-but-memory-preserving transform of price for features.
20. **Singular Spectrum Analysis (SSA)** — causal trend/oscillation decomposition (no EMD look-ahead).
21. **Particle filter / nonlinear state-space regime** — latent regime posterior as a gate.
22. **Reservoir computing / Echo State Network (from physics)** — chaotic-series predictor; nonlinear memory.

## F. GROUPING / UNSUPERVISED REGIME DISCOVERY
23. **Diffusion maps / UMAP manifold of microstructure state → cluster** → find the predictable cluster.
24. **HMM / change-point (ruptures, BOCPD)** — segment into regimes; condition direction per regime.
25. **Symbolic motif mining** — encode returns as symbols, mine motifs that precede directional moves.

## CRUCIAL THEOREM (orthogonal-math research, 2026-05-31) — why complexity gates are null for DIRECTION
arXiv:2512.15720 (Dec 2025, SPY 38.5M trades): order-flow/permutation ENTROPY is INVARIANT UNDER SIGN PERMUTATION,
so it detects PRESENCE of informed trading = MAGNITUDE/volatility, NOT sign. At entropy<5th pct: |5-min return| x2.89
(t=12.41) BUT directional accuracy 45% (chance). => This is WHY PE/Hurst/SampEn gates were null for 30m DIRECTION:
they gate magnitude, not sign. Volatility is forecastable; SIGN is not. Confirms my null complexity results from theory.
- VOLUME/DOLLAR BARS (vbars.py): AUC val 0.509/test 0.516/oos 0.512, selective OOS ~0.50-0.53 — NULL for direction (matches research: bars improve normality not directional AUC).
- Genuinely DIRECTIONAL orthogonal methods still to test: HAWKES intensity ratio (buy vs sell self-excitation), PATH SIGNATURES (signed area=lead-lag), TRANSFER ENTROPY direction. These are sign-AWARE (not sign-invariant).

## HORIZON-DEPENDENT LEVER TRANSFER (generic, 2026-06-03; example key [EURUSD·15m])
When moving a sweep to a LONGER horizon, lever priors shift by MECHANISM-decay, not uniformly:
- **Microstructure/OFI/LOB levers decay** (null by ~60s) → priors DROP at 5m→15m→30m. Confirm-and-kill, don't re-invest.
- **Loss/labeling objective changes are horizon-AGNOSTIC** (MADL/GMADL sign-weighted loss, triple-barrier path labels, 3-class
  deadband) — they reshape the GBM's training signal regardless of horizon. The `|ret|`-weight DOWN-rescue that worked at 5m
  ([5m·DOWN] rebalanced two-sided ~.56) MUST be re-tested per horizon: move COMPOSITION changes (more trend, less micro-noise at
  longer H) flip whether large moves are sign-predictable. A DOWN side dead at 5m can be alive at 15m (measured: [EURUSD·15m]).
- **Cross-pair cross-sectional factor/ranking levers get MORE relevant at longer H** — the USD common factor evolves slowly, so
  rank/PC-shrinkage/IPCA constructions (≠ raw exog features, which were null) plausibly carry more sign at 15m than 5m.
Source: CORPUS_LEVER_INVENTORY.md families (Lucchese/Michankow MADL-GMADL, Kozak-Nagel-Santosh, IPCA, Sirignano-Cont).

**Corollary — cross-pair pooling is HORIZON-GATED DOWNWARD (measured 2026-06-04):** the cross-pair USD-residual+OF primary
(`m5_xpair.py`, the lever that CERTIFIED [EURUSD·5m·UP] and BOTH [EURUSD·15m] sides under refit-CPCV) is **null below 5m** —
at 2m it KILLED under ties-strict refit-CPCV (UP p10 .5096 / DOWN .5124, 0-7% of 15 paths clear) and at 60s it was never an
edge. The slow USD-common-factor needs ≥5m to overcome microstructure noise; the certification gradient is none@60s → UP@5m →
both@15m and does NOT extend back to sub-5m. **Do not expect cross-pair to rescue a sub-5m direction key** — confirm-and-kill.
**Lesson — ALWAYS refit-CPCV a frozen-gate positive (the overfit-gate dissolution recurs):** at [EURUSD·2m] a frozen-VAL
compression gate (probe) AND a frozen-VAL meta-labeler (specialist) BOTH showed .55-.61 selective acc that **collapsed to
~.50 when the gate was re-tuned per fold** under (nested) refit-CPCV. A frozen-VAL selective-accuracy number is never a
certification — only per-fold-refit p10 ≥ breakeven with ≥80% paths clearing is. (Same lesson as the 5m ACI nested-refit kill.)
**Lesson — a different FUNCTION CLASS does not rescue an empty channel:** RFF virtue-of-complexity (P~T random nonlinear basis,
Kelly-Malamud) on the same [EURUSD·2m] features gave VAL AUC .4997 — the 4th model class (after 3-GBM ensemble, online-ARF,
single GBM) to read ~.50. When ≥3 diverse learners agree on ~.50 AUC, the channel is information-empty, not capacity-limited;
the redirect is NEW DATA (external), not a new model.

**Lesson — cross-pair POOLING lifts the MEAN, NOT the worst-regime p10 (decorrelation ≠ certification) (measured 2026-06-05,
[USDJPY·2m]):** pooling 7 USD-majors' 239 base feats (train each pair on its OWN forward sign, eval the target — distinct from
the cross-pair-FEATURES `m5_xpair` lever above) lifted per-fold-refit-CPCV win-rate MEAN from the single-pair ~.52 to **.546
(>0.541 BE)** — yet the worst-regime **p10 SATURATED ~.531** across the variance-reduction stack (seed-ens K1 .5245 → K3 .5306 →
K5+2×data .531). Pooling = **noise-decorrelation** (more diverse training rows → smoother conditional-mean), NOT a new cross-pair
factor SIGN (which doesn't reach <5m — gradient lesson above). So pooling can make the AVERAGE win-rate tradeable while the WORST
regime-combination stays sub-BE → a *marginal regime-risky* edge, not a certification. **Always read CPCV p10 + frac-clear-BE,
never the mean; variance-reduction levers plateau at the worst-regime signal bound.** **ARF horizon-gradient refinement:** at
[USDJPY·2m] online-ARF reads **.508** (>.50, vs the 1m .50 efficiency floor) → adaptive *detectability* of direction first crosses
.50 between 1m and 2m — a thin REAL edge that is genuinely absent at 1m, but still ~1pp under breakeven. (Magnitude stays the
strong sign-invariant edge: [USDJPY·2m] magAUC ~.78 vs dirAUC ~.52.)

**META-LESSON — when an edge is ~1pp from certification, RUN the Tier-I cross-product; do NOT subsume-by-analogy (the program's
own "2nd-pass exhaustiveness audit" pattern) (2026-06-05, [USDJPY·2m]):** a first pass declared exhaustion after seed-ensemble;
a deep 2nd-pass RAN all 9 previously-argued levers and confirmed the ceiling BY EXPERIMENT. Two textbook gap-closers actively
MADE IT WORSE: (i) **multi-algorithm decorrelation HURTS a thin signal** — lgb+xgb+cat ensemble p10 .519 < lgb seed-ens .531,
because xgb/cat are weaker learners on the residual and drag the average (decorrelation only helps when the members are
individually comparable). (ii) **Optuna worst-VAL-half ANTI-TRANSFERS** — best config scored VAL .550 (passes!) but held-out
WORSE than the untuned default, a clean live demo of corr(VAL_acc,OOS_acc)=−.54. Also confirmed null at 2m: recency-weighting
(overfits the small recent window → monotonically worse), ESN reservoir (no intra-window path order the GBM misses), GMADL/|ret|
loss + mag→dir bridge (sign-invariance holds: big moves are NOT more sign-predictable), compression-gate combine (starves n).
**Generalize:** once ≥3 diverse learners agree near the info floor and variance-reduction has plateaued, tuning/decorrelation/
reweighting/re-representation cannot cross a SIGNAL bound — the only redirect is NEW DATA (external). But you must SHOW this by
running the cross-product, not asserting it.

**Lever — BAR / CANDLESTICK PATTERN recognition (vetted 2026-06-05; corpus papers present):** sources in corpus —
`OA_Sezer_AlgorithmicFinancialTrading_StockBarChartImageCNN` (CNN-BI: render N OHLC bars as a 2-D image → 2-D CNN →
direction), GAF / Markov-Transition-Field → CNN image encodings, Kronos (candlestick/K-line foundation model with a price
tokenizer), SACLSTM. **Vet:** (i) explicit candlestick FEATURES (body/wick/range ratios; doji/engulfing/hammer/pin) are
SUBSUMED — they are deterministic functions of O/H/L/C that a GBM already accesses through `rangepos`/`atr_pct`/`gap_prev`/
`bb_width`/multi-TF returns; the re-representation adds nothing (same kill as ESN/GRU), and named candlestick patterns
classically gate move SIZE not SIGN (`OA_Singha_HiddenOrderEntropy_MagnitudeNotDirection`; most "it works" results die under
deflated/purged CV per the Bailey/Lopez-de-Prado backtest-overfitting corpus). (ii) The ONE genuinely NON-subsumed sub-lever
= a **2-D CNN over rendered OHLC bar IMAGES** (Sezer CNN-BI) / Kronos: a conv learns LOCAL 2-D pattern detectors that neither
a per-bar GBM nor a 1-D sequence net (GRU/ESN, both tested) can represent. Prior LOW (corpus grades the image-CNN/GAF levers
`low`; FX-direction CNNs typically ~.52-.55, ≈ the existing edge). **DATA GATE:** needs O/H/L — on disk only as
`ohlc_cache/EURUSD_5m_*` (+ rebuildable from EURUSD ticks via `vbars.py` source). **NOT computable for any pair whose feature
store keeps only `close` (e.g. USDJPY) without raw tick/OHLC acquisition.** ⇒ runnable today on EURUSD; an external-data lever
for USDJPY. If run: same deriv-faithful CPCV harness, pre-registered p10 ≥ breakeven falsifier, eval vs the GBM baseline.

