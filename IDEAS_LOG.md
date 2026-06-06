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
> **RAN it [EURUSD·60s] 2026-06-05 → DIRECTION null/KILLED, MAGNITUDE clears >65% (generic conclusion).** Faithful
> Sezer CNN-BI reimplementation (`METHODS_CATALOG.md` §5.5; `barcnn_bars.py`/`barcnn_run.py`/`barcnn_cpcv.py`/
> `barcnn_mag.py`), image encodings — Sezer close-histogram,
> 3-channel OHLC (wick + up-body + down-body), and GAF (GASF+GADF) — each a small MNIST-class 2-D CNN on the 60s
> wc_ret label (ties-strict, moved-bars). ALL three: VAL dirAUC ≈ .50, test/oos AUC ≈ .50, and faithful CPCV (28
> purged-combinatorial paths) **path_p10 .484–.499 with 0.0 of paths clearing breakeven at every coverage**.
> **Generic lesson (adds to the sign-invariance corpus): re-rendering bars as a 2-D image does NOT create 60s
> direction the GBM/GRU couldn't see — even the GADF *antisymmetric sign-bearing* field is null. The bar-image-CNN /
> GAF lever is magnitude, not ≤60s sign — confirming the 60s near-efficiency keystone via a 4th model class (2-D
> conv).** Kronos NOT built (its reported gains are RankIC/magnitude; no FX/60s/direction numbers; fine-tune
> deteriorates, arXiv:2511.18578) — a magnitude/path foundation model, not a direction lever.
> **The flip side — bar patterns DO predict the MAGNITUDE outcome at >65% (`barcnn_mag.py`, `MAGNITUDE_FINDINGS.md`
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
LDN and Asia certify at NONE. Files `session_xpair_{10,15,30}m_{ny,ldn,asia}_result.json` (5m + 2m **in progress**):
- 10m: NY UP .6053 / DOWN .5896 (15/15 each); LDN .523/.524; Asia .511/.516.
- 15m: NY UP .5845 / DOWN .5712; LDN .527/.520; Asia .519/.496.
- 30m: NY UP .5681 / DOWN .5639; LDN .520/.514; Asia .487/.509.

NY BEATS the legacy fixed-UTC pooled gate at every horizon (legacy 10m .586/.568, 30m .559/.553) — i.e. the all-hours number
was a NY edge DILUTED by two near-coin-flip sessions, not a uniformly-distributed edge. By contrast MAGNITUDE certifies in
**every** session at every freq (tick 1m/2m GBM `session_{1m,2m}_mag_{ny,ldn,asia}_result.json` p10 .58–.71, magAUC NY .675/
LDN .728/Asia .717; bar 5m/10m/30m `session_{5,10,30}m_mag_{sess}_result.json` p10 .72–.80). DIRECTION GBM is null/killed in
**all** sessions at every freq (tick 1m/2m pooled ~.504/.496–.500, frac_clear 0.0; bars killed all, NY strongest e.g. 10m NY
p10 .525 — `session_{1m,2m}_dir_*` / `session_{5,10,30}m_dir_*`). 15m base GBM **in progress**.
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
`kronos_dir_mtf_*_result.json` (FINE-mode "1m→Nm up-the-chain" + multi-TF ensembles **in progress**). The decisive observation:
even at NY≥10m where `session_xpair` certifies .57–.61, Kronos reads ~.50 — because it ingests only EURUSD's OWN OHLCV candles,
NOT the 7-pair USD cross-section that carries the sign. Fine-tuning the predictor did NOT help direction.
> **Generic lesson — a single-pair candlestick/K-line foundation model is a MAGNITUDE/path model, not a direction lever; the
> ≥10m direction edge is CROSS-SECTIONAL (7-pair USD common factor), so any learner fed only one pair's own OHLCV is blind to
> it by construction — adding model capacity / fine-tuning / multi-TF context cannot create a cross-sectional sign from a
> single-pair input.** This is the 5th+ model class to read ~.50 on own-history direction (after 3-GBM ensemble, online-ARF,
> single GBM, RFF, 1-D GRU/ESN, 2-D bar-image CNN). Falsifier template: if a single-pair foundation model EVER beats the
> cross-pair NY gate at the same (horizon, session), the edge is NOT purely cross-sectional for that key — re-open the own-history
> channel. (Consistent with the bar-image-CNN/Kronos magnitude finding above: Kronos's reported gains are RankIC/magnitude.)

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

