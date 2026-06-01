# ORTHOGONAL-MATH IDEAS LOG — 30m EURUSD direction (started 2026-05-31)

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

## EXECUTION ORDER (live)
- [RUNNING] Workflow: Sofien gate mining (larger-TF reversal / flag gates).
- [RUNNING] Workflow: non-time bars / volume-price / VPIN / meta-labeling research.
- [TODO] Workflow: orthogonal-math academic research (A-F above), with reported numbers + implementability.
- [BUILDING] vbars.py: volume & dollar bars → 30m label, honest selective.
- [TODO] m30_complexity.py: permutation entropy + Hurst/DFA + RQA-DET + Lempel-Ziv gates on the 30m model; "avoid losers".
- [TODO] signatures as features; Hawkes intensity on event bars; transfer entropy volume→price.

## RESULTS (append as they come — be honest, cross-window-stable only)
(none yet this thread)

## RESULTS LOG (2026-05-31, honest, cross-window)
- **Larger-TF reversal/flag gates** (1h/4h/daily RSI/DeMarker/WillR/CCI/BB/z/Fisher extremes; confluence; chop-gated) — m30_gates.py: ALL null, none >=0.60 stable across held-out. Oscillator extremes at larger TFs do NOT predict 30m reversals.
- **Complexity/predictability gates** (Permutation Entropy d3 W30/60/120; lag-1 autocorr ac20; variance-ratio Hurst) — m30_complexity.py: NULL. Conditional accuracy is FLAT across all bins of all measures: momentum ~0.48, reversion ~0.515. Complexity does NOT isolate a predictable subset. Only signal = uniform weak ~0.515 mean-reversion (below 0.541 breakeven). The "low-entropy = tradeable" hypothesis FAILS for 30m EURUSD.
- IMPLICATION: 30m direction is ~0.515 mean-reverting, uniform, no regime/complexity gate isolates >0.58. Next untested: VOLUME/DOLLAR BARS (non-time grouping), VPIN, Hawkes intensity on event bars, path signatures, transfer entropy vol->price, Sofien's 79 mined rules (pivots/overextension-release/confluence).

## CRUCIAL THEOREM (orthogonal-math research, 2026-05-31) — why complexity gates are null for DIRECTION
arXiv:2512.15720 (Dec 2025, SPY 38.5M trades): order-flow/permutation ENTROPY is INVARIANT UNDER SIGN PERMUTATION,
so it detects PRESENCE of informed trading = MAGNITUDE/volatility, NOT sign. At entropy<5th pct: |5-min return| x2.89
(t=12.41) BUT directional accuracy 45% (chance). => This is WHY PE/Hurst/SampEn gates were null for 30m DIRECTION:
they gate magnitude, not sign. Volatility is forecastable; SIGN is not. Confirms my null complexity results from theory.
- VOLUME/DOLLAR BARS (vbars.py): AUC val 0.509/test 0.516/oos 0.512, selective OOS ~0.50-0.53 — NULL for direction (matches research: bars improve normality not directional AUC).
- Genuinely DIRECTIONAL orthogonal methods still to test: HAWKES intensity ratio (buy vs sell self-excitation), PATH SIGNATURES (signed area=lead-lag), TRANSFER ENTROPY direction. These are sign-AWARE (not sign-invariant).

## POSITIVE FINDING (validated OOS) — the real "different story": MAGNITUDE is forecastable, DIRECTION is not
m30_magnitude.py: target = 30m forward |return|.
- realized vol (rv30): corr with future|ret| = +0.42/+0.44/+0.44/+0.40/+0.38 (train/val/t24/t25/OOS). Large-move (|ret|>=Q75) classification **AUC 0.75/0.75/0.78/0.73/0.73 — STRONG, STABLE, OOS-verified**.
- permutation entropy: corr ~0.01, AUC 0.50, flat |ret| across quintiles — PE does NOT predict magnitude on EURUSD (the SPY-trades result doesn't replicate on 1m FX returns; realized vol is the real magnitude predictor).
=> CONFIRMED: 30m EURUSD MAGNITUDE/volatility is ~0.73-0.78 AUC forecastable; DIRECTION/sign is ~0.52 (EMH). The predictable structure is magnitude, exactly as the sign-invariance theorem predicts.
IMPLICATION for up/down binary: magnitude predictability lets you AVOID small-move/tie losers and size bets, but does NOT raise directional accuracy (sign within large moves is still ~0.515). Tradeable as a volatility/touch/straddle product, not up/down.

## THREAD SUMMARY (this session's orthogonal/creative push, all honest cross-window)
Tested & NULL for 30m DIRECTION: larger-TF reversal gates; complexity gates (PE/WPE/Hurst/ac20); Sofien 79 rules
(pivots/IBS/Connors-RSI2/TD-setup/BB-reentry/confluence); volume & dollar bars; (theory) transfer-entropy/Hawkes are
sub-minute/hourly/trader-resolved or sign-invariant. POSITIVE & VERIFIED: magnitude forecastable (rv, AUC ~0.75).
Untested-but-low-prior (sign-aware): path signatures (CNN-on-raw-path already null 0.525), 4D Bacry-Muzy Hawkes intensity
(needs LOB events, sub-minute). Direction at 30m remains ~0.52-0.515 mean-reverting; no method this thread broke it.

## PATH SIGNATURES (Lévy area, rough-path theory) — last sign-aware orthogonal lever (m30_sig.py)
Lévy area A(price, order-flow imbalance) = signed area = lead-lag ROTATION, computed causally over W={30,60,120,300}s.
- HS=1800 (30m GOAL): AUC val 0.515 / TEST 0.508 / OOS 0.509 — NULL. Model barely fits (1 tree). Signatures don't help 30m direction.
- HS=5s: signature Lévy areas rank TOP-5 importances; selective ~0.63-0.66 (= base tick edge). Order-flow→price path rotation IS directional AT SECONDS, decays to noise by 30m.
=> FINAL: every sign-aware orthogonal method (signatures, and by extension Hawkes/transfer-entropy which capture the same lead-lag) confirms: directional edge lives at the SECONDS scale; 30m direction is EMH (~0.515).

## DEFINITIVE THREAD CONCLUSION (orthogonal/creative push, 2026-05-31)
30m EURUSD DIRECTION is not predictable beyond ~0.515 by ANY method tried (now ~25 experiments + 4 research workflows +
a sign-invariance theorem). The predictable structure is MAGNITUDE (realized vol -> large-move AUC ~0.75 OOS). The only
directional edge is at the SECONDS scale (~0.65, confirmed by signatures). For an up/down 30m binary, >0.65/>0.75 is not
achievable on EURUSD; the honest tradeable edges are: 15m compression×NY ~0.64, 3s tick ~0.66, and the NEW magnitude/vol
model ~0.75 (volatility/touch products, not up/down).

## SESSION-2 (2026-05-31b) — 5-MIN re-push: CROSS-PAIR + ORDER-FLOW + META-LABELER (NEW orthogonal ideas)
Goal re-set to 5-min >65% OOS. New ideas tried (the genuinely-untried-at-5m set):
- **Cross-pair USD-common-factor / lead-lag** (m5_xpair.py): EURUSD = EUR-strength − USD-strength; build USD basket from the
  other 6 majors (sign-aligned: +ret for USD-base JPY/CHF/CAD, −ret for USD-quote GBP/AUD/NZD), per-pair lead-lag residual
  in EURUSD-equivalent terms, catch-up residual (EURUSD owes the basket move), EUR-idiosyncratic residual, cross-pair
  dispersion & agreement. **The relative-value RESIDUAL reversion is the FIRST orthogonal signal SIGN-STABLE across 2024 &
  2026** (spearman +0.015/+0.024) — momentum-continuation agreement is dead (sign-blind). Lifts honest book 0.566→0.586.
- **Order-flow as features + reliability** (features_of/: Kyle λ, OF persist/accel/uptick, normalized imbalance): 0 AUC lift
  (raw OF into a GBM is null), signed-OF flow-FOLLOWING loses (stably negative vs fwd ret) — but helps the book → 0.594.
- **Learned META-LABELER on ORTHOGONAL axes** (m5_meta.py): 2nd lgb predicts P(primary correct) from agreement/dispersion/OF/
  confidence (NOT the 239), abstain unless meta≥thr, threshold by WORST-VAL-HALF stability (kills the corr(VAL,OOS)=−0.54
  trap). Best honest 5-min book **combined ~0.61** (t24 0.64/t25 0.58/oos 0.62). The right way to "not select losers".
**Falsifier (P0 gate, m5_xp_analyze.py):** NO conditioning region (agreement/OF/disp/vol/session) lifts the 2025 window above
~0.557; ORACLE (hindsight) max-floor = 0.598-0.601. A meta-classifier ≤ its features' conditional accuracy → can't reach 0.65.
**5-MIN CONCLUSION:** honest frontier improved 0.566→**~0.61** (genuine, profitable vs 0.541), but **≥0.65 OOS-stable is NOT
reachable** — the 2025 regime is near-efficient for 5-min direction. Buildable-from-repo levers exhausted; remaining levers =
signed macro/news calendar (external) or a seconds/tick broker (real ≥0.65 edge). Deliverable: m5_xpair_production.py.

---
## Discovery round — novel 2m EURUSD DIRECTION ideas (2026-06-01, sub-agent research, for sweeps/EURUSD_2m)
Generated for the EURUSD 2m sweep (SWEEP_MATRIX Tier-N). Each vetted: genuinely new vs catalog, plausible
SIGN mechanism (survives sign-invariance arXiv:2512.15720), on-disk data, fast-KILL falsifier.
- **N2 Triangular USD-canceling residual** (EURUSD vs GBPUSD rolling cointegration; USD factor algebraically removed → immune to the 2025 USD-factor inversion). arXiv:0812.0913. Prior 15%. **TESTED → KILLED** (min2_triangular_result.json: COMB .499/.497/.497, coin-flip; relative-value reverts too slowly for 120s).
- N3 Cross-quantilogram tail-lead (Han-Linton-Oka-Whang arXiv:1402.1937) — tail-conditional sign asymmetry. Prior 12%. PENDING.
- N4 Market-Intraday-Momentum term-structure (Gao-Han-Li-Zhou JFE2018) — lagged clock-interval return → fwd sign. Prior 10%. PENDING.
- N5 PCMCI+ causal-discovery sign-stable lead (Runge SciAdv2019) — conditions out the common USD factor. Prior 8%. PENDING.
- N6 Directed-information on SIGN sequences (Massey; Quinn-Coleman-Kiyavash) — magnitude-blind go/no-go gate; pre-kills N3/N5/N9. Prior 6%. PENDING (run as gate).
- N7 Asymmetric up/down-tick Hawkes intensity imbalance (Bacry-Muzy arXiv:1301.1135) — cross-excitation asymmetry φuu−φdd, mid-tick only. Prior 9%. PENDING.
- N8 Signed-semivariance-skew sign-conditioning on the magnitude model (Patton-Sheppard) — RS⁺−RS⁻ skew is signed. Prior 5%. PENDING.
- N9 Cross-pair signed ordinal transition imbalance (Bandt-Pompe / Neuman-Cohen-Tamir) — cross-series, signed. Prior 5%. PENDING.

## Discovery round 2 — workflow fan-out (5 lenses), 2026-06-01. Novel 2m ideas beyond N2-N9 (priors honestly 4-9%):
- Stoikov microprice-velocity vs mid-velocity slope-divergence, gated reversion sign (within-EURUSD, USD-inversion-immune; signed fair-value lead, not a level). Prior 8%, on-disk. [TOP — test_spec in workflow output]
- Directed horizontal-visibility-graph peak-vs-trough irreversibility (signed by tail decomposition; sign-covariant, unlike the killed scalar I_W). Prior 7%, on-disk.
- PID synergy/redundancy across legs (higher-order info beyond pairwise directed-info, which the cross-leg gate killed). Prior 6%.
- CMI: condition next-2m sign on the surviving relative-value RESIDUAL STATE (not the slow residual itself). Prior 6%.
- Event-localized (not time-averaged) information bursts at CKS-OFI events. Prior 5%.
- WMR-fixing reversal at 2m, FLOW-conditioned (m30_fix found WMR sign-flipped to continuation in 2026 @30m; never tested @2m). Prior 5-6%. Coverage-starved (2026 = 5 month-turns).
- Turn-of-month signed drift book (only ever a 15m coverage gate, never standalone @2m). Prior 5%. Coverage-starved.
- ML: contrastive/SSL pretrain → linear sign-probe; conformal-abstain; focal/ordinal loss on the UP asymmetry. Prior 4-9% (program shows ceiling is DATA not model).
ASSESSMENT: all low-prior; the 13-channel sweep + online-ARF keystone + the UP-uncertified CPCV make these confirmatory. The Stoikov slope-divergence (within-EURUSD, USD-immune, signed) is the single most-worth-testing; tested as a completeness check.
