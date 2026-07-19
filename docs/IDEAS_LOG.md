# IDEAS LOG — transferable prediction levers (GENERIC)
SCOPE: GENERIC (currency/timeframe-agnostic). Idea→experiment backlog of transferable levers + mechanisms +
falsifier TEMPLATES. **No per-key incumbents/numbers here** — those live in the per-key backlogs
`sweeps/<PAIR>_<tf>_backlog.md` and results in `results/<PAIR>_RESULTS.md`. See `REPO_MAP.md`.
Generic methods: `docs/METHODS_CATALOG.md`; permutation menu: `SWEEP_MATRIX.md`; cross-key theory: `docs/THEORY.md`.
**Untried-methodology research (2026-06-07): `docs/NOVEL_METHODS_RESEARCH.md`** — 110-candidate web+academic slate (raw:
`novel_methods_candidates.json`) distilled to ranked runnable experiments + input transforms (frac-diff, information-driven
bars, vol-time subordination, cross-pair whitening) + GARCH/HAR/semivariance/Hawkes/BOCPD/causal-PCMCI gaps. **Every item
gated by the §6f frozen-past forward holdout + a surrogate-null** (pooled CPCV alone is insufficient — leakage trap #9).

## Deferred research-OS capabilities

These capabilities remain valuable but unauthorized. A trigger permits a new
plan; it does not activate the capability. The first bound consumer is the
inactive refit package contract in `docs/USDCHF_M15_CURRENT_REFIT.md`
`[USDCHF·15m]`.

| ID | Value retained | Unlock trigger | Owner / pickup |
|---|---|---|---|
| `REFIT-PROSPECTIVE-1` | Prospective Deriv-demo evaluator with separate UP/DOWN lanes and candidate-specific routing | An exact inactive package exists and a later plan can seal a pre-outcome T0 | Standalone `feature-dev -> plan-review -> dev-cycle/strategy-eval`; no automatic activation |
| `REFIT-PUBLISH-1` | Promotion through the book manifest/index/registry owner | A side-specific prospective trial passes its preregistered venue-economics rule | Separate reviewed publication issue; never infer survivor status from packaging |
| `REFIT-CADENCE-1` | Operator-approved repeatable refit cadence | One policy passes prospective and later canary gates at two vintages | Separate issue; no scheduler assumed |
| `ALPHA-COMPILER-1` | Thin typed hypothesis/operator/rejection layer over the existing campaign/evidence control plane | One authorized point-in-time macro/rates/flow sample exists, or two real adapters repeat manual structure | Separate infrastructure issue; never run on the closed on-disk feature universe |
| `REAL-MONEY-CANARY-1` | Minimum-stake VPS-only canary with a fixed loss budget and kill switch | Demo realized-P&L lower bound exceeds zero and the user separately authorizes exact stake/loss limits | Separate issue with explicit authority; never automatic |

**Tested-on-keys pointers (where these levers were instantiated):**
- EURUSD · 1m  → `sweeps/EURUSD_1m_backlog.md`  (+ `results/EURUSD_RESULTS.md` §60s) — **CLOSED 2026-06-03: both sides certified-or-exhausted, 2 discovery rounds DRY**
- EURUSD · 5m  → `sweeps/EURUSD_5m_backlog.md`  (+ `results/EURUSD_RESULTS.md`)
- EURUSD · 2m  → `sweeps/EURUSD_2m_backlog.md`
- EURUSD · 10m → `sweeps/EURUSD_10m_backlog.md`  (+ `results/EURUSD_RESULTS.md` §10m) — **CLOSED 2026-06-04: BOTH sides certified (cross-pair `EURUSD.m10xp.v1` UP p10 .586 / DOWN p10 .568, 15/15), improve+discover loops DRY (2 rounds, 11 levers)**
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
Source: docs/CORPUS_LEVER_INVENTORY.md families (Lucchese/Michankow MADL-GMADL, Kozak-Nagel-Santosh, IPCA, Sirignano-Cont).

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
> **RAN it [EURUSD·60s] 2026-06-05 → DIRECTION null/KILLED, MAGNITUDE clears >65% (generic conclusion).** Faithful
> Sezer CNN-BI reimplementation (`docs/METHODS_CATALOG.md` §5.5; `barcnn_bars.py`/`barcnn_run.py`/`barcnn_cpcv.py`/
> `barcnn_mag.py`), image encodings — Sezer close-histogram,
> 3-channel OHLC (wick + up-body + down-body), and GAF (GASF+GADF) — each a small MNIST-class 2-D CNN on the 60s
> wc_ret label (ties-strict, moved-bars). ALL three: VAL dirAUC ≈ .50, test/oos AUC ≈ .50, and faithful CPCV (28
> purged-combinatorial paths) **path_p10 .484–.499 with 0.0 of paths clearing breakeven at every coverage**.
> **Generic lesson (adds to the sign-invariance corpus): re-rendering bars as a 2-D image does NOT create 60s
> direction the GBM/GRU couldn't see — even the GADF *antisymmetric sign-bearing* field is null. The bar-image-CNN /
> GAF lever is magnitude, not ≤60s sign — confirming the 60s near-efficiency keystone via a 4th model class (2-D
> conv).** Kronos NOT built (its reported gains are RankIC/magnitude; no FX/60s/direction numbers; fine-tune
> deteriorates, arXiv:2511.18578) — a magnitude/path foundation model, not a direction lever.
> **The flip side — bar patterns DO predict the MAGNITUDE outcome at >65% (`barcnn_mag.py`, `docs/MAGNITUDE_FINDINGS.md`
> §3).** Pointing the same bar-image CNN at large-vs-small move (the deriv Touch/Range/straddle outcome) gives magAUC
> .699/.714/.686 and selective large-call precision .68→.80 with **all 28 CPCV paths ≥0.65 at cov≤0.2 in every held-out
> year** — but ONLY with ABSOLUTE-scale rendering (per-window min-max normalization throws away the vol scale and caps
> at .64). **Generic lesson: a bar IMAGE is a magnitude representation; normalize it min-max and you keep shape but
> lose the size signal — for magnitude, preserve absolute scale; for direction, neither helps (sign isn't there).**

---

## 2026-06-06 — DST-correct session re-campaign + Kronos look-forward fix + full-suite correctness audit

Three generic, cross-key transferable findings from a DST-correct per-session (NY/LDN/Asia) re-test of every freq×method,
a user-caught look-forward bug in the Kronos forecast→direction family, and a 19-agent + lead full-suite audit. New scripts:
`sessions.py` (DST-correct `session_mask`/`SESSIONS`), `session_1m.py`/`session_2m.py` (tick GBM per session),
`session_bars.py` (bar GBM, any H), `session_xpair.py` (cross-pair STRICT session-only, any H), `kronos_ft.py` (single-process
GPU Kronos predictor fine-tune), `kronos_mtf.py` (alignment-CORRECTED multi-TF Kronos direction eval), `kronos_bars.py`
(H-min OHLCV + forward deriv-label builder), `kronos_ensemble.py` (multi-TF vote combine), `barcnn_run.py` gained a SESSION arg.

### (a) Lever — DIRECTION edge is SESSION-CONCENTRATED (NY carries the cross-pair sign); MAGNITUDE is SESSION-INVARIANT
The certified ≥10m cross-pair USD-common-factor SIGN edge is **decisively NY-concentrated**. STRICT session-only DST-correct
cross-pair (`session_xpair.py`, `sessions.py`: NY=8–17 America/New_York, LDN=8–16 Europe/London, Asia=9–18 Asia/Tokyo,
session_mask = local-tz hour applied per-day across train+val+test+oos) certifies **BOTH** sides at every horizon **only in NY**;
LDN and Asia certify at NONE. Files `session_xpair_{2,5,10,15,30}m_{ny,ldn,asia}_result.json` (COMPLETE — NY both sides 2m→30m):
- 2m: NY UP .564 / DOWN .560; LDN .514/.506; Asia .499/.497. **NEW — legacy EURUSD 2m was DEAD/uncertified.**
- 5m: NY UP .596 / DOWN .588; LDN .526/.516; Asia .510/.504. **NEW DOWN side — legacy was UP-ONLY (.553).**
- 10m: NY UP .6053 / DOWN .5896 (15/15 each); LDN .523/.524; Asia .511/.516.
- 15m: NY UP .5845 / DOWN .5712; LDN .527/.520; Asia .519/.496.
- 30m: NY UP .5681 / DOWN .5639; LDN .520/.514; Asia .487/.509.

NY BEATS the legacy fixed-UTC pooled gate at every horizon (legacy 10m .586/.568, 30m .559/.553) — i.e. the all-hours number
was a NY edge DILUTED by two near-coin-flip sessions, not a uniformly-distributed edge. By contrast MAGNITUDE certifies in
**every** session at every freq (tick 1m/2m GBM `session_{1m,2m}_mag_{ny,ldn,asia}_result.json` p10 .58–.71, magAUC NY .675/
LDN .728/Asia .717; bar 5m/10m/30m `session_{5,10,30}m_mag_{sess}_result.json` p10 .72–.80). DIRECTION GBM is null/killed in
**all** sessions at every freq (tick 1m/2m pooled ~.504/.496–.500, frac_clear 0.0; bars killed all, NY strongest e.g. 10m NY
p10 .525 — `session_{1m,2m}_dir_*` / `session_{5,10,30}m_dir_*`). 15m base GBM null all sessions (NY .520/LDN .515/Asia .512). Per-session bar-image CNN also null (NY .510/.503/.505, LDN .509/.499/.503, Asia .501/.506/.500).
> **Generic lesson:** the direction-edge HORIZON gradient (none@60s→UP@5m→BOTH@≥10m) has a SESSION dimension that is just as
> sharp — the cross-sectional USD sign is forecastable **when the US desk is the marginal price-setter (NY hours)** and decays
> to coin-flip in LDN/Asia, while MAGNITUDE (volatility presence, the sign-invariant edge) is session-AGNOSTIC. **For any new
> (currency, ≥10m) key, run the cross-pair side-split GATED ON THE NY SESSION first — it both lifts the number and reveals
> whether an all-hours edge is really a diluted NY edge.** Falsifier template: if NY session-only fails to beat the all-hours
> pooled number at the same horizon, the edge is NOT NY-concentrated for that key — record and fall back to all-hours.

### (b) METHODOLOGY lesson — FM-F forecast-derivation: align the PREDICTED window to the LABEL window
A user-caught 1-bar look-forward MISALIGNMENT in `kronos_dir.py`/`kronos_ft.py` eval: at decision bar `i` they fed context
`slice(i-L, i)` = bars [i-L..i-1], predicted bar `i`, and scored `Pup = pred_close(i) > C[i-1]` — i.e. the move INTO the entry,
window [t[i-1], t[i]]. But the deriv label `y[i]` (`barcnn_bars.labels_at`: entry t[i]+1s, exit t[i]+61s) is the FORWARD window
[t[i]+1, t[i]+61] — DISJOINT from the scored window, off by one bar. **Failure-mode "FM-F forecast-derivation": deriving a binary
signal from a generative price FORECAST scored against a window misaligned with the deriv label.** Critically this direction is
SAFE — **a misaligned forecast yields a FALSE NULL, never a false POSITIVE** (it scores a past/disjoint window uncorrelated with
the forward label). FIX (`kronos_mtf.py`): context ends AT bar `i` (`slice(i-L+1, i+1)`, last close = entry ref `C[i]`), predict
`pred_len = H/GRID` FORWARD steps, `Pup = pred_close(+H) > C[i]`; pred-side contiguity `t[i+Hsteps]-t[i] == Hsteps*step`;
nonoverlap GAP = HS+TOL. Validated: forward label agrees with next-bar sign 92.3% (n=233,950). `kronos_dir.py` now gated
(`KRONOS_DIR_LEGACY=1` to override); legacy result superseded.
> **Generic lesson — when a model FORECASTS price and you DERIVE direction, the predicted window MUST cover the SAME forward
> interval as the deriv label at the SAME decision bar.** Check three things: (i) context ends at (includes) the entry bar so the
> last context close = the entry reference price; (ii) the forecast horizon equals the label horizon in the SAME units; (iii)
> pred-side contiguity + nonoverlap gap on the forecast steps. Falsifier template: confirm the derived label agrees with the raw
> forward next-bar sign at ≥90% before trusting any AUC — if it doesn't, you are scoring a misaligned (often past/disjoint)
> window and any null is uninterpretable. Audit scope: the full-suite audit found this bug ISOLATED to the 2 Kronos scripts;
> every other forecast-derivation script (`usdjpy_{1m,2m}_statespace`, `usdjpy_2m_xhorizon`, `m5_xhorizon`, `m5_lossbatch`,
> `f1_compound`) predicts the FORWARD quantity over the SAME horizon as the label at the SAME bar = correctly aligned, and
> GBM/CNN classifiers trained DIRECTLY on the label are structurally immune to FM-F.

### (c) Lever — single-pair foundation-model (Kronos) DIRECTION = NULL; the edge is CROSS-SECTIONAL not own-history
With the alignment fixed, the corrected Kronos eval (`kronos_mtf.py`) reads NULL at EVERY horizon (1/5/10/15/30m), zero-shot AND
fine-tuned, in ALL sessions (pooled .50–.51, CPCV p10 .489–.500, all KILLED, up-rates in-band). Files
`kronos_dir_mtf_*_result.json`. Both of the user's multi-TF ideas were tested and are also null: FINE-mode "use a finer grid up the chain" (1m→2/5/10m, 5m→10m) all KILLED (pooled .490/.509/.495/.503, CPCV p10 .481/.499/.475/.497); multi-TF vote-ensemble (`kronos_ensemble.py`) — 5m KILLED (pooled .531, p10 .428), 10m ABORT (only 12 common decision bars across 3 grids → nonoverlap-chrono leaves too few shared decision instants). The decisive observation:
even at NY≥10m where `session_xpair` certifies .57–.61, Kronos reads ~.50 — because it ingests only EURUSD's OWN OHLCV candles,
NOT the 7-pair USD cross-section that carries the sign. Fine-tuning the predictor did NOT help direction.
> **Generic lesson — a single-pair candlestick/K-line foundation model is a MAGNITUDE/path model, not a direction lever; the
> ≥10m direction edge is CROSS-SECTIONAL (7-pair USD common factor), so any learner fed only one pair's own OHLCV is blind to
> it by construction — adding model capacity / fine-tuning / multi-TF context cannot create a cross-sectional sign from a
> single-pair input.** This is the 5th+ model class to read ~.50 on own-history direction (after 3-GBM ensemble, online-ARF,
> single GBM, RFF, 1-D GRU/ESN, 2-D bar-image CNN). Falsifier template: if a single-pair foundation model EVER beats the
> cross-pair NY gate at the same (horizon, session), the edge is NOT purely cross-sectional for that key — re-open the own-history
> channel. (Consistent with the bar-image-CNN/Kronos magnitude finding above: Kronos's reported gains are RankIC/magnitude.)

### (d) Lever — Kronos per-path DISPERSION as a forward-vol MAGNITUDE feature = KILLED (2026-06-06)
Tested the top "run-first" lever from the Kronos web-research synthesis: tap the K sample paths Kronos generates and
discards (`kronos.py:467` mean-collapses them), turn their dispersion into 6 forward-looking vol features, and ask if
they beat the certified backward-rv magnitude baseline `[-pe,rv30,rv120]`. `kronos_disp.py` ran a paired CPCV ablation
on 15,041 nonoverlap bars (K=24, pred_len=30, all horizons 1/5/10/15/30m, cpcv_certify's exact target/model/deflation).
**KILLED everywhere:** paired ΔAUC = −.0003 (H=1, zero effect) → −.0070 (H=30, hurts); CI95 excludes 0 below for H≥5.
The dispersion features ARE used by the GBM (non-trivial gain) but correlate .46–.83 with rv30 — a noisier Monte-Carlo
restatement of realized vol that backward rolling-std already captures cleanly. `kronos_disp_disp_main_result.json`,
docs/MAGNITUDE_FINDINGS.md §6c.
> **Generic lesson — a single-pair generative path forecast adds NO magnitude info orthogonal to cheap trailing realized
> vol.** Kronos's per-path spread ≈ predicted forward vol ≈ a re-derivation of rv30/rv120, just with Monte-Carlo noise and
> (at long H) compounding error that anti-transfers. Magnitude gains must come from inputs rv CAN'T see — macro-event
> windows, deseasonalized/semivariance RV, an external implied-vol feed (docs/MAGNITUDE_FINDINGS.md §7), NOT more model capacity
> on own OHLCV. Mirrors the direction finding (d above): Kronos is blind to anything not in one pair's own candles.

### (e) Lever — Kronos `decode_s1` 512-d HIDDEN STATE as a frozen feature = SMALL REAL MAGNITUDE LIFT / direction null (2026-06-06)
Lever 2 of the synthesis. Take the post-norm transformer hidden at the decision bar (`decode_s1` returns `[B,L,512]`,
kronos.py:305-308; one forward pass, no AR/sampling, ~190 win/s). `kronos_embed.py` paired-CPCV-ablates it on 18,077
nonoverlap bars vs the certified rv baseline. **MAGNITUDE: the FIRST feature to beat rv** — paired ΔAUC
+.0067/+.0022/+.0076/+.0024/+.0058 (H=1/5/10/15/30m), **every CI95 excludes 0** (clears the +.005 bar at 1/10/30m); rv
.73→.74. Real but MODEST. **DIRECTION emb-only NULL** (AUC .50–.51). `kronos_embed_embed_main_result.json`,
docs/MAGNITUDE_FINDINGS.md §6d, docs/DIRECTION_FINDINGS.md.
> **Generic lesson — a single-pair LEARNED representation carries a little magnitude info that hand-crafted rv misses,
> but ZERO sign.** The likely source of the magnitude lift is time-of-day vol seasonality (the embedding has an additive
> time-emb; rv30/rv120 don't) → the cheap win is §7 deseasonalized-RV, not a Kronos dependency. Decisive next-check: add
> hour-of-day/day-of-week to the baseline and re-run; if it captures the lift, drop Kronos. Direction stays cross-sectional.
> **RESOLVED 2026-06-07, then RETRACTED — pooled-CPCV artifact.** The 4-way ablation (pooled CPCV) suggested time-of-day
> beats the embedding at 30m (+.0087) and the embedding adds beyond-clock info at 1–10m. **But `deseason_mag.py` (full
> 2012-2026) + `deseason_fwd.py` (frozen-past forward holdout) KILLED it:** the +tod lift is +0.0140 on pooled CPCV but
> decays 2024 +.020 → 2025 −.009 → 2026 −.054 forward (non-stationary seasonal shape). NOT deployable. The whole
> tod/clock magnitude-lift family was pooled-CPCV-only; adversarial review (wwtci9slp) confirmed it's leakage-free but
> forward-fragile. **New leakage trap #9 (pooled-CPCV non-stationary-feature memorization) recorded in METHODS_CATALOG.**
> docs/MAGNITUDE_FINDINGS.md §6f.

### (f) Lever — Chronos-2 GROUP-ATTENTION on the 7-pair USD panel = DIRECTION NULL (2026-06-07)
Lever 3, the cross-sectional bet — the highest-value/most-unsolved need (every single-pair model reads ~.50 on direction;
the certified edge is the 7-pair USD common factor). amazon/chronos-2 (119.5M, the ONLY mainstream TSFM whose
group-attention mixes across variates) fed the L=512 close panel of all 7 USD pairs; EURUSD direction read 3 ways on
44,999 bars (`chronos2_xpair.py`): forecast-sign, embed→GBM (1536-d cross-pair representation), NY/LDN/Asia. **KILLED
every horizon/method:** acc .505–.511, AUC .506–.514, CPCV p10 ~.50 — far below breakeven .541. A FAINT NY-tilt
(NY acc slightly > pooled) + AUC consistently a hair above .50 means the signal is THERE but non-deployable. Forecast
spread → magnitude sub-bar (+.002–.004, < +.005). `chronos2_xpair_c2_main_result.json`, docs/DIRECTION_FINDINGS.md,
docs/MAGNITUDE_FINDINGS.md §6e.
> **Generic lesson — a strong multivariate TSFM's learned cross-attention on raw price LEVELS does NOT recover the
> cross-sectional direction sign.** The deployable ≥10m NY edge (.57–.61, `session_xpair`) lives in the SPECIFIC engineered
> features (USD-residual, basket-catchup, signed eu-equiv lead-lag residuals), NOT in what a foundation model extracts from
> the panel. The cross-sectional edge is FEATURE-engineered, not FM-recoverable (matches arXiv:2511.18578: multivariate
> TSFMs only modestly close the gap to engineered/tree methods on direction). **All 3 web-research Kronos/TSFM levers now
> resolved: L1 per-path dispersion KILLED, L2 decode_s1 embedding (small mag-win / dir-null), L3 Chronos-2 (dir-null).**
> Net: Kronos/TSFMs add a sliver to MAGNITUDE (likely time-of-day, cheaper via deseasonalized RV) and NOTHING deployable
> to DIRECTION. Frontier for direction stays EXTERNAL data (intraday rate-diff, risk-reversals); for magnitude, §7 cheap levers.

**Full-suite audit verdict (generic, reinforces "no certified book invalidated"):** the FM-F bug is ISOLATED to the Kronos
family (2 scripts), NOT systemic — across all 308 scripts + a dedicated forecast-derivation sweep, no second instance was found.
Tier-1 empirical clean proofs on each substrate: TICK (features causal via truncation test max|full-trunc|=0.0, label forward
0/4000 mismatch, corr(y,future)=.486 vs corr(y,past)=−.003, up-rate .5006), BAR (forward 0/2000 mismatch at H=5/10/30,
corr(y,future)~.99 vs ~−.02, up-rate .498–.506), XPAIR (`m5_xpair.build_xp` forward 0/2000 mismatch H=10), BAR-FEATURE causality
(239 features, truncation max|full-trunc|=0.0). Bounded flags that do NOT invalidate any cert (recorded for fix, not redirect):
`barcnn_mag.py:163` + `barcnn_regime.py:69,71` FM-E selective-threshold on pooled test+oos (magnitude/regime target,
CPCV-deflated; fix = use VAL threshold); `usdjpy_2m_cpcv2.py` FM-E max-p10 cell (best .5185≪.541 → CERTIFIED=false anyway);
`min2_mim.py:18` + `min2_legsign.py:25` FM-A shift(−FWD) no contiguity guard (KILL screens, already KILLED); superseded pre-v3
cohort (`min1_v*`/`min2_v*`/`tickmodel*`/`tick5s_final`/`tick_ensemble`, no result.json, replaced by `*_production` books).
> **Generic lesson — a look-forward/alignment bug in one model FAMILY does not invalidate the program; prove each SUBSTRATE
> clean independently (truncation-causality on features + forward-mismatch recompute on labels + corr(y,future)≫corr(y,past) +
> up-rate in-band), and classify every other forecast-derivation script as aligned-or-not by the same FM-F test.** FM-F (false
> NULL) and FM-E (selective-threshold on test, false POSITIVE) are the two recurring forecast/threshold traps to screen.

---

## 2026-06-07 — NOVEL-METHODS campaign (execute ALL of `docs/NOVEL_METHODS_RESEARCH.md` §7), by-experiment status

Verified-facts source of truth: `docs/CAMPAIGN_2026-06-07_FACTS.md`. Every item below is gated by the deployment-faithful
**FROZEN-PAST FORWARD HOLDOUT** (train≤2023 → per-year 2024/25/26), NOT pooled CPCV (leakage trap #9). Campaign certified
NOTHING new — every tested lever is KILLED or REAL-but-SUB-BAR. The ONE positive: the certified MAGNITUDE edge RE-VALIDATED
on a clean forward holdout with NO decay.

### (a) DELIVERED infra — two NEW reusable validation GATES + the cross-pair PANEL (committed 051e525)
Phase-1 substrate for all path-based cross-sectional methods, plus the two mandatory gates `NOVEL §0` requires:
- **`fwd_holdout.py`** — reusable frozen-past forward-holdout gate (magnitude AUC+lift / direction cov-selacc modes). The
  MANDATORY deployment gate. Self-checked: DEPLOYS a stationary signal (+0.10/yr), REJECTS a trap-#9 non-stationary feature
  (decays to −0.015) — the exact deseason-+tod failure mode it exists to catch.
- **`surrogate_null.py`** — phase-randomize / IAAFT surrogate-null gate. Self-checked: linear lag-1 autocorr → NOT
  significant (real .6959 ≈ null_p95 .6961, spectrum preserved); nonlinear |.| vol-clustering → significant (real .0883 ≫
  null_p95 .0199). Separates spectral re-encoding from genuine nonlinear structure.
- **`build_panel.py`** → `features/panel_<year>.parquet` — the clean 7-pair USD return panel (5,345,437 bars 2012-2026;
  per-row `r_<PAIR>` eu-equiv 1-min log-returns, `fac` USD common factor, `e_<PAIR>` residual, `c_eur` raw close). DELIVERED
  + FAITHFULNESS-CERTIFIED via `panel_faithcheck.py`: all continuous channels (`eu_r/usdbask/catchup/eurresid/disp/ll_*`)
  BIT-IDENTICAL (max_abs 0.0) vs certified `build_xp`; only `agree*` differs at 1 row/lookback (warmup off-by-one). Pooled-CPCV
  repro read selacc p10 .5355 / mean .5425 vs certified m15xp .567/.574 — the DOCUMENTED weaker-reimplementation effect (single
  600-tree pooled LGBM vs certified refit), NOT leakage (which would INFLATE); up-rate .5074 inside [.47,.53] tripwire.
  VERDICT: CONSTRUCTION-FAITHFUL, safe for all downstream.
> **Generic lesson — pooled CPCV ALONE is insufficient (leakage trap #9): it can MEMORIZE a non-stationary feature and report
> a clean p10 that DECAYS forward.** The frozen-past forward holdout (`fwd_holdout.py`) is now the MANDATORY gate before any
> "deployable" claim, paired with `surrogate_null.py` to separate genuine nonlinear structure from spectral re-encoding. Both
> are reusable across keys/horizons; build the panel/feature substrate, prove it BIT-IDENTICAL to the certified builder, THEN run.

### (b) MAGNITUDE HAR / realized-vol canon — TESTED → REAL-but-SUB-BAR / KILLED as upgrade (committed 7d45bf8)
`mag_har.py` + `mag_har_result.json`. Arms added to certified base `[-pe,rv30,rv120]`, forward holdout, horizons 10/15/30m,
target |ret_H|≥train-Q75; falsifier = +0.005 AUC in ≥2 forward years AND ≥2 horizons. **ALL ARMS FAIL:** +har (multiscale RV)
mean fwd ΔAUC +0.0010, 2/3 deployable — sub-bar; +jump (bipower split) +0.0003, 1/3 — null; +semivar (RS⁺/RS⁻/signed-jump)
−0.0000, 1/3 — null; +harq (realized quarticity) +0.0004, 2/3 — null; **+all (stacked) +0.0021, 3/3 deployable (no decay) —
forward-CONSISTENT but economically negligible.** Base rv already extracts ~all magnitude (collinearity vs rv120: lRV120 .89,
RS .70, HARQ .58/−.63; only signed-jump SJ120 orthogonal −.04 and carries nothing). **THE positive: RE-VALIDATES the certified
magnitude edge on a clean deployment-faithful holdout** — base AUC .799/.750/.750 (10m) .791/.739/.737 (30m), 4–5.6× decile
lift, NO decay. VERDICT: KILLED as deployable upgrade; magnitude path EXHAUSTED on-disk.
> **Generic lesson — HAR/jump/semivar/quarticity decompositions add only a forward-consistent SLIVER (+.002) over plain
> rolling rv because rolling rv already captures ~all the realized-vol magnitude signal; only signed-jump is orthogonal and it
> carries nothing.** Magnitude gains must come from inputs rv CAN'T see (macro-event windows, external implied-vol feed), not
> finer functions of own realized vol.

### (c) FRACTIONAL DIFFERENTIATION (FFD) direction — TESTED → KILLED (15m AND 30m)
`frac_diff.py` (hand-rolled FFD, fixed-width weights + ADF d*-selection; self-check: random walk needs d*=0.1 to pass ADF while
retaining 98.8% level-memory vs 0.9% for plain returns) + `frac_direction.py`. FFD USD factor/residual/lead-lag into the certified
direction book, forward holdout, NY cov0.10 selacc; per-pair d* 0.1–0.2 (windows ~500 bars), thresh 1e-4.
- **15m KILLED** (`frac_direction_15m_result.json`): base selacc 2024 .5900 / 2025 .5520 / 2026 .5254 (pooled .5642); +ffd
  .5792/.5301/.5403 (pooled .5537) → DECAYS (Δ −0.0108/−0.0219/+0.0149), deployable=false; ffdonly pooled .5001 (coin flip).
- **30m ALSO KILLED, harder** (`frac_direction_30m_result.json`): base 2024 .5842 / 2025 .5483 / 2026 .5007 (pooled .5558);
  +ffd .5546/.5135/.5064 (pooled .5308) → DECAYS (Δ −0.0296/−0.0348/+0.0057), deployable=false; ffdonly pooled .4958 (below coin flip).
Note the base book ITSELF decays over forward years (.59→.55→.525), consistent with the documented refit-dependence of the
cross-pair edge.
> **Generic lesson — FFD level-memory adds NOTHING to direction and HURTS recent years; "stationarity-with-memory" is a
> magnitude/level transform, not a sign creator.** Genuinely-untried ≠ promising: a level-preserving stationary transform of
> price feeds the same cross-sectional channel that already certifies, and re-encoding it as fractionally-differenced memory
> only adds collinear noise that anti-transfers forward. FFD direction DEAD at both 15m and 30m.

### (d) CROSS-SECTIONAL DIRECTION (D1 lead-lag signature, D6 HAVOK, D7 signed-semivariance) — TESTED → ALL KILLED (15m AND 30m)
`xsec_direction.py` (+ `xsec_direction_{sig,havok}_{15,30}m_result.json`). Forward holdout, NY cov0.10 selacc; arms
{base=certified xp book, base+fam, famonly, fam_shuffle(mechanism-specificity control)}. The certified base book ITSELF DECAYS
forward (15m .5895/.5518/.5234; 30m .5827/.5505/.5055 — the documented refit-dependence) and NOTHING adds to it:
- **D1 depth-2 lead-lag SIGNATURE — KILLED both horizons.** 15m: +sig DECAYS (Δ −.0043/−.0099/+.0032), sigonly pooled .5243 ≈
  sig_shuf .5236 → NOT genuine lead-lag (rotation surrogate doesn't degrade it → **FAILS the mechanism-specificity falsifier**).
  30m: +sig DECAYS all 3 years (−.0094/−.0043/−.0133), sigonly pooled .5283 (sub-base).
- **D6 frozen-basis HAVOK (Koopman forcing) — KILLED both horizons.** 15m: +havok DECAYS (Δ −.003/−.0024/+.0014), havokonly
  pooled .5148 (sub-breakeven, far below base .5624); the forcing only marginally beats its phase-randomized surrogate (.5148
  vs .5063) and never clears .541. 30m: +havok DECAYS, havokonly pooled .5197 (sub-base).
- **D7 realized signed-SEMIVARIANCE direction (Patton-Sheppard good/bad vol) — REAL-but-SUB-BREAKEVEN, KILLED (the distinction
  that matters).** This is the ONLY direction shot that PASSES its mechanism null: semivaronly BEATS semivar_shuf (sign-flip
  control) by +.015 (15m) / +.030 (30m) → the signed-vol asymmetry genuinely carries DIRECTIONAL content (NOT null-on-mechanism
  like D1/D6). BUT weak: semivaronly pooled .5258 (15m) / .5382 (30m) — below breakeven .541 (only 2024@30m .5516 clears it),
  DECAYS forward, and +semivar does NOT add to the base book (Δ15m −.0154/−.0029/−.0107, Δ30m −.0081/+.0062/−.0139). The
  DIRECTION analog of magnitude's "real-but-sub-bar": genuine signed content, sub-deployable, already subsumed by the cross-pair book.
> **Generic lesson — 4 convergent direction nulls on the deployment-faithful forward holdout (FFD/D1/D6/D7) reinforce that
> direction beyond the engineered cross-pair book is EFFICIENT on existing data.** No cross-sectional direction family beats or
> matches the certified book; the base book itself decays forward (refit-dependence). The signature/Koopman families fail their
> MECHANISM-specificity null (a rotation/phase surrogate doesn't degrade them → they encode no genuine lead-lag/forcing). D7 is
> the careful exception: signed-semivariance asymmetry DOES carry directional content (clears its sign-flip null, +.015/+.030)
> yet is sub-breakeven, decaying, and non-additive — genuine-but-weak, already subsumed. **Distinguish "null on mechanism"
> (D1/D6 — nothing there) from "real but sub-breakeven" (D7 — something there, undeployable); both are KILLED for deployment but
> only the latter is evidence the channel is non-empty.** The only frontier for direction is EXTERNAL data (intraday rate-diff,
> implied-vol/risk-reversals, EURGBP ticks).

### (e) REMAINING SLATE — reasoned SCOPE DECISIONS (evidence-based skip/blocked/moot/queued; 2026-06-07)
After 4 convergent direction nulls (FFD/D1/D6/D7) on the deployment-faithful gate, with the base book itself decaying forward,
the remaining `docs/NOVEL_METHODS_RESEARCH.md` §7 slate is scoped by EVIDENCE — recorded so the next session resumes deliberately,
not blindly. From `docs/CAMPAIGN_2026-06-07_FACTS.md` "reasoned scope decisions":
- **D5 causal lead-lag (Granger / PCMCI / structural-VAR)** — **REASONED-SKIP.** The certified base book ALREADY contains every
  peer's lagged lead-lag feature (`ll_<pair>k`, k∈{1,3,5,10,15,30}) which the GBM weights; D1 proved signature lead-lag content
  does NOT survive a rotation null. A Granger feature-SELECTION on top of an already-lead-lag-saturated GBM has near-zero
  marginal prior. Re-open only with EXTERNAL leaders (rate-diff), not more EURUSD-panel selection.
- **D2 untruncated signature kernel / D4 FASCL contrastive** — **BLOCKED:** sigkernel/KeOps + the FASCL encoder need GPU and (for
  sigkernel) a C-extension build that fails here (no `Python.h`). Deferred to a GPU+headers environment; LOW prior given D1
  (signatures) already null.
- **D8 quantile-direction baseline** — **SKIP:** a CONTROL, not an edge candidate; only needed to ablate a TSFM-quantile claim,
  which this campaign does not make.
- **Phase 5 gates G1 (TDA corr-cloud, needs gudhi=unbuildable) / G2 (BOCPD, ruptures available) / G3 (windowed-DMD)** — **MOOT:**
  a gate conditions a SURVIVING signal; no direction signal survived the forward holdout, so there is nothing to gate. Re-open
  only if a future (external-data) signal clears breakeven first.
- **T1 information-driven bars** — **BLOCKED on external data:** needs per-pair 1s/tick data; only EURUSD 1s on disk (2021+). The
  cross-pair-synchronization value (the whole point) requires the other 6 pairs' ticks = external acquisition.
- **T3 vol-time subordination / T4 cross-pair whitening / T5 Hilbert phase** — **QUEUED, LOW prior:** T4/T5 are direction
  transforms (direction shown efficient); T3 is a magnitude transform but `mag_har` showed base rv already extracts ~all
  magnitude. Catalogued; not run this campaign.
- **Magnitude exotica M4 (TDA-Wasserstein, gudhi-blocked) / M5 (multiscale-ECC) / M6 (MOMENT)** — **QUEUED, LOW prior:** after the
  HAR canon (§6g) showed the realized-vol family adds only a forward-consistent sliver over base rv. M4 also gudhi-blocked.
> **Durable campaign lesson — the forward-holdout is now the MANDATORY gate; a pooled-CPCV p10 alone certifies NOTHING (trap
> #9, pooled-CPCV non-stationary-feature memorization, recorded in `METHODS_CATALOG`).** Two reusable gates now exist
> (`fwd_holdout.py`, `surrogate_null.py`) and the faithfulness-certified 7-pair `build_panel.py` substrate. Campaign net: 4
> convergent direction nulls (FFD/D1/D6/D7) on the forward holdout reinforce that direction beyond the cross-pair book is
> EFFICIENT — the signature/Koopman families carry no genuine mechanism, signed-semivariance carries real-but-sub-breakeven
> content, and the base book itself decays forward. Magnitude exhausted on-disk (re-validated, no decay). Nothing new
> certified; UP/DOWN leaderboard UNCHANGED. The only frontier for direction is EXTERNAL data; remaining items are scoped
> skip/blocked/moot/queued above, not blindly queued.

---

## 2026-06-08 — NEURAL + SPECTRAL forecast→sign sweep (3 lever families), by-experiment status

Three "technically-uncoded" lever families (mined from external directional-prediction + n-hits repos) RUN at all 6 EURUSD
timeframes, single-pair AND cross-pair input, GPU + deriv-faithful (TRAIN 2012-21 / VAL 22-23 / held-out per-year 2024/25/26,
2026 strict OOS; selective-acc @cov0.10, ties-dropped, moved-bars, breakeven .541). KILL per arm: VAL dirAUC ≤ .515 OR no
held-out year's selective-acc CI95-lower ≥ breakeven. **All KILLED — no leader unseated.** Each family converts a *generative
PATH or SPECTRAL forecast* into a sign — i.e. is structurally exposed to the SIGN-INVARIANCE theorem (`docs/THEORY.md`): a forecaster
that nails move SIZE need carry no move SIGN. Tested-on-keys (no per-key numbers here): EURUSD {1,2,5,10,15,30}m → KILLED;
numbers in `results/EURUSD_RESULTS.md` (§2026-06-08), consolidated `neural_spectral_dir_sweep_result.json` (+ per-horizon
`{nbeats_nhits_dir,decomp_dir,spectral_dir}_<tf>m_result.json`). Retarget knob: env `MX_HOR=<minutes>`.

### (a) Lever — N-BEATS / N-HiTS PATH-forecast → sign (`nbeats_nhits_dir.py`)
Deep interpretable basis-expansion / hierarchical-interpolation forecasters predict the forward price PATH; derive direction
from the sign of the forecast over the label horizon. **Mechanism / sign-invariance:** these minimize a path-reconstruction loss
(MSE/MAE on level), so they learn the trend+seasonality DECOMPOSITION that carries magnitude, not the residual sign the regime
erases — the same null as the prior `binary_alpha` N-HiTS@5m. **Falsifier template:** if a path-forecast → sign EVER clears VAL
dirAUC > .515 AND a held-out year's selective-acc CI95-lower ≥ breakeven, the path model is recovering genuine forward sign for
that key — re-open. **Tested-on-keys:** EURUSD {1,2,5,10,15,30}m, single + cross-pair → KILLED (`nbeats_nhits_dir_<tf>m_result.json`,
`neural_spectral_dir_sweep_result.json`).

### (b) Lever — DECOMPOSITION / FREQUENCY transformers DLinear/Autoformer/FEDformer + TFT-quantile fan (`decomp_dir.py`)
Series-decomposition (trend/seasonal) linear + auto-correlation (Autoformer) + frequency-enhanced (FEDformer, FFT-domain
attention) forecasters, plus a TFT quantile-fan whose median/spread → directional call. **Mechanism / sign-invariance:** the
trend/seasonal/frequency split is a re-expression of the return *distribution* (magnitude/vol structure), not its *signed
order*; the quantile fan models forecast UNCERTAINTY (a magnitude object) — sign-invariant by the same theorem as PE/entropy
gates and the bar-image CNN. **Falsifier template:** a decomposition/freq/quantile forecaster that beats both gates above for a
key means a frequency or quantile channel carries sign there — re-open. **Tested-on-keys:** EURUSD {1,2,5,10,15,30}m, single +
cross-pair → KILLED (`decomp_dir_<tf>m_result.json`, `neural_spectral_dir_sweep_result.json`).

### (c) Lever — causal DWT / SSA SPECTRAL band-split → per-band model → recombine → sign (`spectral_dir.py`)
Causal discrete-wavelet-transform and Singular-Spectrum-Analysis band decomposition (no look-ahead), a GBM per frequency band,
literal recombination → sign. **Mechanism / sign-invariance:** a SPECTRAL re-encoding preserves the power spectrum and so is the
canonical sign-invariant transform (cf. `surrogate_null.py`: a phase-randomized surrogate matching the spectrum is direction-
null). A flashy thin-coverage held-out cell can show a high CI-lower MIRAGE; only the pre-registered per-year CI95-lower at the
fixed coverage certifies. **Falsifier template:** a band-split → sign that clears VAL dirAUC > .515 AND a held-out year's
selective-acc CI95-lower ≥ breakeven at the FIXED coverage (not a thin-n cell) means a frequency band carries sign — re-open.
**Tested-on-keys:** EURUSD {1,2,5,10,15,30}m, single + cross-pair → KILLED (`spectral_dir_<tf>m_result.json`,
`neural_spectral_dir_sweep_result.json`).
> **Generic lesson — path/spectral/quantile FORECASTERS carry move SIZE, not SIGN (sign-invariance theorem, `docs/THEORY.md`).**
> N-BEATS/N-HiTS (path), DLinear/Autoformer/FEDformer + TFT-quantile (decomposition/frequency), and causal DWT/SSA (spectral
> band-split) all reduce to magnitude/distribution re-expressions of own (or cross-pair) price; none recovers the directional
> sign at any of the 6 EURUSD horizons. This adds 3 model families to the ≥5 already at ~.50 on own-history direction and
> confirms the prior N-HiTS@5m null. Frontier for direction stays EXTERNAL data, not a richer forecaster.


---
## Transferable lessons from the AUDUSD 15m direction sweep (2026-06-09) — generic; per-key numbers in `results/AUDUSD_RESULTS.md` + `sweeps/AUDUSD_15m{,_backlog}.md`

> **A commodity/risk USD-major's 15m DIRECTION-sign is still NY-session-concentrated + OWN-PAIR-specific — its "home-session" idiosyncratic info gates MAGNITUDE, not 15m sign.** [AUDUSD·15m] Going in, the strong prior was that AUDUSD's edge lives in the **Asia** session (RBA/China/AU-data/commodity flows). The symmetric session test REFUTED this: NY certifies both sides, Asia + LDN are sub-breakeven. The commodity/China information is real but gates move SIZE (sign-invariance theorem), while the directional sign rides the US-session USD flow — same as EURUSD/USDJPY. **Lesson: test the session symmetrically for every new pair; do NOT assume the home-session carries direction even for a home-driven currency.** Cite: `audusd_15m_cpcv_session_{ny,asia,ldn}_multicov_result.json`.

> **Cross-pair POOLING vs own-pair-specificity is pair-specific and must be measured, not assumed from the USD-major label.** [AUDUSD·15m] AUDUSD is a heavy USD-major yet pooling DILUTES (own-pair base ignores the USD-factor pool + lead-lag + cousin residual + risk-bloc; all rank below top-20) — the USDJPY case, NOT the EURUSD case. **Lesson: the cross-pair keystone certifies EURUSD/EUR-bloc but not JPY/AUD; run A6 per pair.** Cite: `audusd_15m_xpair_xpbase_result.json`, `audusd_15m_orthochan_result.json`.

> **Cousin residual-DIFFERENCE (e.g. AUDNZD = e_AUD−e_NZD) and a commodity-vs-safe-haven RISK factor are SUBSUMED by the own-pair base at 15m** even when mechanistically orthogonal (the base 239 carry no cousin info). famonly ΔAUC ≈ 0. The cross-sectional residual reversion that certifies EURUSD pooling does not transfer to the Antipodean triplet at 15m. Cite: `audusd_15m_orthochan_result.json`.

> **Cross-horizon STACK viability = the parent↔child probability correlation; gate on it before building.** [generic] USDJPY 30m parent was collinear with the 15m child (corr .957 → no orthogonal sign → subsumed); AUDUSD's 30m parent is only ~.72–.81 correlated (genuinely decorrelated) — but the parent is *equally weak* (AUC ≈ child), so blending cuts variance like a seed-ensemble WITHOUT adding directional information (path-means flat; only the p10 order-statistic moves). **Lesson: decorrelation is necessary but NOT sufficient — a cross-horizon parent helps only if it is BOTH decorrelated AND carries information the child lacks (higher or complementary AUC); a decorrelated-but-equally-weak parent is redundant with seed-ensembling.** Cite: `audusd_15m_xhorizon_result.json`, `audusd_15m_xhstack_ny_result.json`.

> **Seed-ensemble (Tier-I I2) reproduces as the one robust own-pair improve-lever** (lifts both p10 AND mean modestly), while triple-barrier label, meta-labeling, symmetric ACI, GMADL-selection, and calibration remain null/subsumed-by-precedent for an own-pair NY book. The honest direction ceiling on existing on-disk data is an information bound (NY moved-AUC ~.53–.54); >65% as a certified floor requires EXTERNAL signed data (intraday AU-US rate differential, risk-on/off VIX/ES, commodity index) — the standing program conclusion, AUD-instantiated. Cite: `audusd_15m_cpcv_session_ny_seedens3_result.json`.

---
## Transferable lessons from the GBPUSD 15m direction sweep (2026-06-10) — generic; per-key numbers in `results/GBPUSD_RESULTS.md` + `sweeps/GBPUSD_15m.md`

> **EUR-bloc discriminator resolved: GBPUSD follows EURUSD (pooling ADDS), not JPY/AUD (own-pair-specific).** [GBPUSD·15m] The program's prior entering GBPUSD was genuinely open — EUR-bloc pooling certifies EURUSD but dilutes JPY+AUD; GBPUSD sits at the EUR end by USD-correlation and FX-structure. The symmetric test confirmed: xpair 340-feat matrix lifts BOTH sides at EVERY coverage (+.007–.032 p10 vs own-pair NY incumbent). **Lesson: EUR-bloc pairs (EURUSD, GBPUSD) are the pooling-wins cases; commodity/JPY/AUD are own-pair cases. Poolable = tight USD-EUR-GBP triangle; separate = commodity/risk/carry.** Cite: `gbpusd_15m_xpair_xpbase_result.json`, `gbpusd_15m_cpcv_xpair_ny_multicov_result.json`.

> **Orthogonal channel families (eurgbp, risk+eurobloc, RS±, carryrank) are ALL SUBSUMED by the 340-feat xpbase matrix — mirroring the AUD precedent exactly.** [GBPUSD·15m] All 4 famonly screens return ADDS=False (best ΔVal .0046, all < .005 threshold; best 2026 cov2 .5362 vs base .4892 = false positive killed by 2026 gate). The xpair book captures the most these channels offer via its cross-pair pooling; bolt-on channel families add nothing once xpair features are in the matrix. **Lesson: when the xpair book is the incumbent, run the famonly screen per §Ortho, but expect ADDS=False if the xpair matrix already spans the cross-pair information space.** Cite: `gbpusd_15m_orthochan_result.json`.

> **Seed-ensemble K=3 is the robust improve-lever for DOWN; UP saturates at K=1.** [GBPUSD·15m] Same pattern as AUD: DOWN gets a genuine lift from seed-ensembling (+.005–.010 p10 AND mean +.010, both criteria); UP's cov1 p10 drops by .0045 (order-stat noise — the ensemble bands the worst-path outcomes rather than systematically lifting them). K=3 = deliverable config; K=8 is a saturation check (sibling precedent: null). **Lesson: in EUR-bloc xpair books, seed-ensemble helps DOWN more than UP (UP has higher variance → the K=1 order-stat p10 can be anomalously high; K=3 regresses it slightly).** Cite: `gbpusd_15m_cpcv_xpair_ny_seedens3_result.json`.

> **The >65% floor remains an information bound on both sides at GBPUSD 15m with on-disk data.** [GBPUSD·15m] UP @cov1 p10 .647 (borderline-65%; seed-config order-stat band .647–.6515, means .68); DOWN @cov1 p10 .6323 (mean .6624). The worst-path fold-autopsy shows the recent era (2024–26) is uniformly weaker than 2018–23 — a regime shift in USD-factor non-stationarity. Era-agnostic levers (more seeds) do not close this. **The standing program conclusion holds: >65% certified floor on existing data = an information bound; net-new external features (intraday news sentiment, real-time FX flows, cross-pair microstructure) are the only lever left.** Cite: `gbpusd_15m_cpcv_xpair_ny_seedens3_result.json`, fold autopsy in `sweeps/GBPUSD_15m.md`.

---
## Discovered idea (untested) — from USDCHF 15m discovery R1 (2026-06-10); per-key result → `sweeps/USDCHF_15m_backlog.md`

> **★ Synthetic-cross LEVEL error-correction band for SNB/policy-MANAGED currencies (a DIRECTION lever, not magnitude).** [generic; first target USDCHF·15m] For a currency whose central bank pins a CROSS rate's LEVEL into a slow band (SNB↔EURCHF; analogously HKMA↔USDHKD, or any soft-peg/target-zone), build the synthetic cross from two on-disk USD legs: logEURCHF = logEURUSD + logUSDCHF. A causal rolling z-score of that LEVEL vs its slow anchor carries a RESTORING-FORCE SIGN (target-zone/ECM mean-reversion): −sign(z) on the cross maps through to a signed prediction on the managed leg (USDCHF). **Why it survives sign-invariance:** it is a conditional-MEAN drift on a policy-stationary LEVEL (Krugman target-zone S-curve), NOT a vol/entropy/flow statistic — so unlike return-residual mean-reversion (corr~0 at 15m) it predicts the SIGN of the next move, not just its size. **Distinct from cross-pair POOLING** (which uses RETURN residuals / lead-lag): this uses the LEVEL, an axis the 239-feat base + xpair return-residuals do NOT span. **Prior low-med; fast-KILL:** add [z, |z|, leg-shares] to the base GBM — KILL if VAL moved-AUC ≤ base OR the z-coefficient sign flips across eras (2024↔2026) OR no held-out year's COMB CI95-lower clears breakeven. Mechanism source: Krugman target-zone (1991) + SNB EURCHF floor regime; corpus + web discovery R1. Build: `usdchf_15m_ecm.py`.

---
## Transferable lesson from the USDCHF 15m direction sweep (2026-06-10) — generic; per-key numbers in `results/USDCHF_RESULTS.md` + `sweeps/USDCHF_15m.md`

> **★ The all-session FROZEN pooling SCREEN can FALSE-NEGATIVE; the NY refit-CPCV is the DECISIVE pooling test — run it even when the screen says IMPROVES=False.** [USDCHF·15m] The frozen all-session xpair screen returned IMPROVES_base=False (VAL AUC lifts .5301→.5330 but frozen-2026 cov2 COMB .5049 < base .5103, DOWN collapses) — by the standard escalation rule this would have KILLED pooling. But the NY-restricted per-fold-refit CPCV showed the OPPOSITE: xpair BEATS the own-pair NY incumbent at EVERY cov on BOTH sides (single-seed pooling-test stage: cov1 p10 UP .6997/DOWN .662 vs own .641/.636; the deployed deliverable is the K=3 seed-ens `USDCHF.m15ny_xpair_seedens.v1` cov1 .6935/.6682, which superseded it), and the matched frozen-forward head-to-head confirmed it (xpair-frozen beats own-pair-frozen in all 6 forward cells, mean +.048 — genuine, not trap#9). **Why the screen lies:** it scores the FROZEN ALL-SESSION 2012-21 vintage; the deliverable is NY + refit. The EUR-bloc cross-pair signal is era-local (USD-factor structure drifts), so the per-fold-refit harness captures it while the frozen all-session book cannot — and the dilution from off-carrier (non-NY) bars further masks it in the screen. **Protocol fix: for any pair with a MECHANISTIC reason to expect pooling (EUR-bloc cousin — EUR/GBP/CHF triangle), escalate to the NY refit-CPCV pooling test regardless of the frozen-screen verdict; the screen's frozen-2026 failure is necessary-not-sufficient evidence against pooling.** **Caveat/follow-up:** USDCAD's pooling was ruled NULL on the SCREEN ONLY (never refit-CPCV-tested) — mechanistically defensible (CAD is commodity, no EUR-bloc cousin) but technically a gap; a USDCAD NY xpair-CPCV retest would close it. Cite: `usdchf_15m_xpair_xpbase_result.json` (screen, IMPROVES=False), `usdchf_15m_cpcv_xpair_ny_multicov_result.json` (refit-CPCV, pooling WINS), `usdchf_15m_xpair_frozen_result.json` + `usdchf_15m_ownpair_frozen_result.json` (matched frozen-forward, xpair wins all cells).

> **USDCHF is the EUR-BLOC case (pooling wins), and a HYBRID carrier (broad + NY-concentrated) — the safe-haven label did NOT predict own-pair-specificity.** [USDCHF·15m] Prior reasoning gave two faces: safe-haven (own-pair, like USDJPY) vs EUR-bloc (pooling-wins, like EURUSD/GBPUSD). USDCHF resolved to EUR-BLOC despite being a premier safe-haven — because CHF≈EUR (SNB-managed, EURCHF tight). Also unlike the pure havens (all-session efficient, NY-only), USDCHF certifies on ALL-SESSION too (EUR-bloc breadth) AND concentrates further in NY (haven). **Lesson: classify a pair's pooling behaviour by its EUR-bloc co-movement (CHF, GBP → pool with EUR), not by its risk archetype (haven/commodity). The session landscape (broad-vs-concentrated) and the pooling-family are SEPARATE axes — measure both.** Cite: `usdchf_15m_cpcv_session_{all,ny}_multicov_result.json`.

> **USDCHF is the FIRST major to CERTIFY >65% on BOTH sides** (cov1 refit-CPCV p10 UP .6935 / DOWN .6682; cov.5 .7303/.7084) — the EUR-bloc xpair-NY **seed-ens K=3** book `USDCHF.m15ny_xpair_seedens.v1` (`eb44d999`), which superseded the single-seed parent (`7505c934`, cov1 .6997/.662). The >65% target, an information bound for the own-pair havens (USDCAD/USDJPY) and borderline for GBPUSD (.647/.632), is CLEARED here because pooling + NY-concentration + the CHF≈EUR tight coupling stack. REFIT-DEPENDENT (deploy w/ periodic retrain). Cite: `usdchf_15m_cpcv_xpair_ny_seedens3_result.json` (deliverable), `usdchf_15m_cpcv_xpair_ny_multicov_result.json` (single-seed parent).

> **The refit-dependent forward decay of an EUR-bloc xpair direction book is an INFORMATION BOUND, not a fixable feature-selection artifact.** [USDCHF·15m, generic] The certified xpair-NY book decays frozen (cov1 .79→.48 2024→26). Hypothesis (IRM / environment-invariant feature-stability filter): drop the features whose directed sign→Y FLIPS across eras and refit the invariant subset → less decay. RESULT: pruning 40.1% of feats (135 era-local sign-flippers) HURT 2024/25 (cov1 −.10/−.02) and COLLAPSED UP in 2026 (.50 vs .63) — only DOWN marginally improved; SURVIVES=False. **The era-local-sign features carry GENUINE in-era signal; the cross-pair USD-factor is non-stationary in a way that is real predictive structure per-era but does not transfer forward. No on-disk feature selection separates "era-local noise" from "era-local signal" — they are the same thing.** Lesson: for refit-dependent pooled-FX books, the deployment answer is periodic RETRAINING (size on the refit floor), NOT a cleverer frozen feature set. Don't spend more compute trying to make a pooled book frozen-deployable. Cite: `usdchf_15m_irm_result.json`.

> **★ Seed-ensemble K=3 LIFTS a POOLED/cross-pair book on BOTH the mean AND the p10 — re-run seed-ens after a pooling win, not just on own-pair books.** [USDCHF·15m, generic] On the EUR-bloc xpair book, K=3 raised path-MEAN in 15/15 cov×side cells AND p10 in 13/15 (only the thin cov0.01/0.005 UP p10 dips, mean still rises) over the single-seed parent — a genuine variance-reduction FLOOR lift, not TB-style redistribution, so it became the certified deliverable. This is STRONGER than the seed-ens effect on own-pair-haven books, where the prior pattern was a modest/asymmetric lift near parity (own-pair UP can saturate at K=1, only DOWN gains). **Mechanism: a pooled/cross-pair feature space gives the per-seed boosters more diverse, decorrelated training rows, so averaging 3 seeds cuts conditional-mean variance MORE than on a single-pair space → the worst-regime p10 rises, not just the mean. Lesson: after a POOLING win certifies a book, always re-run the K=3 seed-ensemble — expect a two-sided mean+p10 floor lift (unlike the ~parity it gives an own-pair-haven book).** Cite: `usdchf_15m_cpcv_xpair_ny_seedens3_result.json`; book `USDCHF.m15ny_xpair_seedens.v1` supersedes single-seed `USDCHF.m15ny_xpair.v1`.

> **★ A TB-first-touch / refit-CPCV "improvement" on the OWN-PAIR book can be REAL yet SUBSUMED by a cross-pair-pooled deliverable — always compare a lever against the BEST combination, not the base.** [USDCHF·15m, generic] Triple-barrier first-touch labelling (TB_K=1.5) certified AND scored IMPROVES=True vs the OWN-PAIR NY base at every cov (a genuine label lift on the weaker own-pair space) — yet the EUR-bloc xpair seed-ens deliverable DOMINATES it at every cov/side by +.008…+.048, so it was SUBSUMED, not promoted. Scored only against the own-pair base it looked like a win; scored against the best standing combination (xpair + seed-ens) it adds nothing. **It was also an un-frozen-forward-gated refit-CPCV positive — the refit-overfit signature already seen on the own-pair havens (USDCAD/USDJPY), so a frozen-forward gate is mandatory before trusting any such "improvement." Lesson: a lever's baseline is the BEST current deliverable, never the bare base book; a real improvement over the base can be entirely inside the gap a stronger orthogonal lever (pooling) already closed.** Cite: `usdchf_15m_cpcv_tbfirsttouch_ny_multicov_result.json` (TB IMPROVES own-pair base), `usdchf_15m_cpcv_xpair_ny_seedens3_result.json` (deliverable dominates TB).
