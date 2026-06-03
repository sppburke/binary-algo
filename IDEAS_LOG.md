# IDEAS LOG — transferable prediction levers (GENERIC)
SCOPE: GENERIC (currency/timeframe-agnostic). Idea→experiment backlog of transferable levers + mechanisms +
falsifier TEMPLATES. **No per-key incumbents/numbers here** — those live in the per-key backlogs
`sweeps/<PAIR>_<tf>_backlog.md` and results in `<PAIR>_RESULTS.md`. See `REPO_MAP.md`.
Generic methods: `METHODS_CATALOG.md`; permutation menu: `SWEEP_MATRIX.md`; cross-key theory: `THEORY.md`.

**Tested-on-keys pointers (where these levers were instantiated):**
- EURUSD · 5m  → `sweeps/EURUSD_5m_backlog.md`  (+ `EURUSD_RESULTS.md`)
- EURUSD · 2m  → `sweeps/EURUSD_2m_backlog.md`
- EURUSD · 30m → `sweeps/EURUSD_30m_backlog.md`

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

