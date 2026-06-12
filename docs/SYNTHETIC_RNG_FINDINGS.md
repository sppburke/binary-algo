<!-- Generated 2026-06-09. Empirical audit of Deriv synthetic-index RNG.
     All numbers are Tier-1: computed this session from ticks pulled live off Deriv's public
     WebSocket API (wss://ws.derivws.com, no auth, app_id=1089) and analyzed locally.
     Scripts: syn_collect.py (collector), syn_rng_audit.py (randomness battery),
     syn_spike_audit.py (point-process/hazard). Raw data + per-symbol JSON in syn_data/. -->

# Deriv Synthetic Indices — RNG / Predictability Audit

> **Verdict: NO predictive edge found.** Every test is consistent with Deriv's stated design
> (a cryptographically-secure RNG driving the documented price processes). The pure volatility
> indices are IID-Gaussian constant-vol and pass a NIST-style battery; the engineered indices
> show exactly their documented structure, and the only non-trivial unknown — spike/jump
> **timing** — is memoryless and unpredictable. The "predict the next series" goal is dead;
> this rules it out rigorously rather than on faith. Every public "Deriv prediction bot" is
> scam-adjacent marketing (no credible crack exists).

## 0. What Deriv claims (verified, T3/T4)
- **CSPRNG**, verbatim: *"we use Cryptographically Secure Pseudo-Random Number Generators (CSPRNGs)… the same grade of technology used in banking security."* (`experts.deriv.com/insights/do-brokers-manipulate-synthetic-indices`). Seed → number sequence → {tick direction, magnitude}.
- Vol indices = constant-volatility simulated markets, 10%–300%; standard R_* tick every **2s**, "(1s)" variants every **1s** (`deriv.com/trading-terms-glossary`).
- If the CSPRNG claim holds, prediction from outputs is cryptographically infeasible **by definition**. We did not take it on faith — we tested it.

## 1. Data (Tier-1, collected this session)
Public WS API, no auth. **Tick history is capped at ~1 calendar day** (the `ticks_history` `start` floor); paging via `end` cannot go older. So each series below is one full UTC day at native cadence.

| Symbol | Ticks | Cadence | Notes |
|---|---|---|---|
| R_100, 1HZ100V | 43,196 / 86,380 | 2s / 1s | pure volatility indices |
| CRASH1000, CRASH500, CRASH50 | ~86,383 | 1s | down-spike indices |
| BOOM1000, BOOM50 | ~86,383 | 1s | up-spike indices |
| JD100 (Jump 100) | 86,385 | 1s | symmetric periodic jumps |
| stpRNG (Step Index) | 86,381 | 1s | fixed-step |
| RB100 (Range Break 100) | 86,387 | 1s | bounded + breaks |

Cadence was **perfectly regular** in every series (`dt min = max = median`, 100% regular) — no gaps, confirming the documented tick spacing.

## 2. Program A — Pure volatility indices (R_100, 1HZ100V)

| Test | R_100 | 1HZ100V | Expected if CSPRNG const-vol |
|---|---|---|---|
| Annualized vol implied | **0.994** | **0.999** | ≈ 1.0 (confirms "100%") |
| Return skew / excess kurtosis | +0.017 / −0.021 | +0.010 / −0.038 | ≈ 0 / ≈ 0 (Gaussian) |
| max\|ACF\| returns (lags 1–50) | 0.0093 | 0.0092 | ≤ IID floor (0.0113/0.0118) |
| max\|ACF\| returns² | 0.0128 | 0.0077 | ≤ IID floor (0.0117/0.0084) |
| **corr(rv, future\|r\|)** W30 / W120 | **−0.009 / +0.004** | **+0.023 / +0.011** | ≈ 0 |
| up-fraction / runs p | 0.5021 / 0.71 | 0.5008 / 0.90 | 0.5 / random |
| NIST battery (sign bits / price LSB) | **6/6 · 6/6** | **6/6 · 6/6** | pass |

**Findings:**
1. R_* are **IID-Gaussian constant-volatility random walks**, exactly as advertised. The "100% annualized" reading of "R_100" is empirically confirmed (0.994–0.999).
2. **No serial structure** in direction or magnitude — all autocorrelations sit at/below the IID-Gaussian noise floor; NIST passes 6/6 on both the sign stream and the price-parity (LSB) stream.
3. **The magnitude edge does NOT transfer.** `corr(rv, future|r|) ≈ 0` vs **~+0.40 on FX**. There is **zero volatility clustering** (a constant-vol IID generator has nothing to cluster). The repo's rv30/rv120 magnitude model would have nothing to predict on R_*. → The "retrain the FX magnitude edge on synthetic vol indices" fork is **dead on the pure vol indices.**

## 3. Program B — Engineered indices (documented structure)

### 3a. Gross structure (all confirmed vs Deriv docs)
| Symbol | skew | exkurt | up-frac | implied vol | NIST sign | Reading |
|---|---|---|---|---|---|---|
| CRASH1000 | −45.7 | +2415 | **0.9989** | 0.24 | 0/6 | 99.9% up-drift + rare DOWN crashes |
| CRASH500 | −33.0 | +1282 | 0.9979 | 0.29 | 0/6 | same, more frequent |
| BOOM1000 | +45.2 | +2220 | **0.0009** | 0.18 | 0/6 | 99.9% down-drift + rare UP booms |
| JD100 | −0.15 | +437 | 0.5008 | 1.28 | **6/6** | symmetric drift + periodic jumps |
| stpRNG | −0.001 | **−2.000** | 0.5002 | 0.07 | 6/6 sign | **fixed ±step** (\|r\| const) + coin-flip dir |
| RB100 | −9.1 | +3074 | 0.5001 | 0.17 | runs/apen fail | bounded mean-reversion + rare breaks |

Notable: **Step Index** has excess kurtosis exactly **−2.0** (signature of a two-point ±step) and \|return\| autocorrelation = **1.000** → the magnitude is a *known constant* while direction is a fair coin (NIST 6/6). The worst case for us: size is trivially known to everyone (incl. Deriv's pricing), direction is unpredictable.

### 3b. Spike/jump TIMING — the only non-trivial unknown (point-process test)
The direction, magnitude, and *average* frequency are all documented & priced. The one thing not given away is *when* the next spike fires. Tested via inter-spike gaps: CV (1.0 = memoryless), hazard slope vs age (rising = "due" = exploitable), gap autocorrelation, KS-vs-geometric.

| Symbol | spikes/day | mean gap (vs nominal) | spike mag | CV(gaps) | hazard slope p | gap ACF₁ (noise floor) | **timing** |
|---|---|---|---|---|---|---|---|
| CRASH1000 | 89 | 962 (×0.96) | 1047× | 1.096 | 0.61 (flat) | +0.064 (±0.209) | memoryless |
| BOOM1000 | 74 | 1168 (×1.17) | 976× | 0.992 | 0.31 (flat) | −0.187 (±0.229) | memoryless |
| CRASH500 | 168 | 506 (×1.01) | 468× | 0.998 | 0.13 (flat) | +0.100 (±0.152) | memoryless |
| JD100 | 60 | 1400s ≈ 23min | 38.6× | 0.974 | 0.09 (flat) | +0.166 (±0.255) | memoryless |
| **CRASH50** | **1445** | 59.6 (×1.19) | 50.7× | 0.974 | 0.73 (flat) | **+0.016 (±0.052)** | **memoryless (well-powered)** |
| **BOOM50** | **1554** | 55.6 (×1.11) | 54.9× | 0.935 | 0.04 | **−0.001 (±0.050)** | **memoryless (well-powered)** |

**Findings:**
1. Documented frequencies/magnitudes confirmed: CRASH1000 → 1 per ~960 ticks; CRASH500 → ~506; JD100 → ~23 min & ~39× (doc: ~20 min, ~30×).
2. With the high-frequency CRASH50/BOOM50 (~1,500 spikes → ACF floor ±0.05), **spike timing is memoryless**: flat hazard (no "due" effect), gap autocorrelation ≈ 0. Knowing the full spike history tells you nothing about the next spike — a Bernoulli-per-tick process, exactly what a CSPRNG yields.
3. BOOM50 shows CV 0.935 (<1) and KS-vs-geometric rejected (p<0.001) — a *mild* sub-Poisson regularity (slight minimum-spacing tendency). But with a **flat hazard and zero gap autocorrelation**, this is **not operationally exploitable**: it doesn't let you time the next spike.

## 4. What this means for monetization
- **Pure vol indices (R_*, 1HZ*):** no RNG predictability; no magnitude/vol-clustering edge to harvest with lookbacks/accumulators. Dead for our methods.
- **Crash/Boom/Jump:** direction & magnitude are *given*; the only unknown (timing) is unpredictable. The slow drift exactly offsets the expected spike by design → the buy-and-hold-the-drift "strategy" is zero-EV before costs. No edge.
- **Step:** magnitude known/constant, direction a fair coin → nothing to predict.
- These confirm Deriv's "audited for fairness" claim *behaviourally*: the feed is a single global broadcast and every exploitable quantity is either documented (and priced) or memoryless.

## 5. Honest scope / limitations
- **One day per symbol** (tick-history hard cap). Sufficient for the decisive tests (CRASH50/BOOM50 gave ~1,500 spikes; ACF floors ±0.05; vol-clustering and magnitude-transfer effects on FX are ~10× any residual here). More data would tighten, not overturn.
- A NIST/Dieharder/TestU01 *cryptographic* certification (millions of bits, full battery) was **not** run; passing a battery proves statistical quality, not cryptographic security. Our battery is a subset (monobit, block-freq, runs, DFT spectral, approx-entropy, cusum) — adequate to *falsify* a weak RNG, which it did not.
- **Not exhaustively timing-tested:** Range Break (RB) break-timing, and the newer indices (Skew Step 80/20, Trek directional bias, Drift Switch, DEX, Daily Reset). All share the same CSPRNG architecture and expose only *documented* structure (a known probability split or drift is priced into the product, not a prediction edge), so the expectation is the same; they were not run to ground.

## 6. Next steps (only if pursuing further)
1. **Live forward-collection** of CRASH50/BOOM50 over days/weeks to push the spike sample to 10⁴–10⁵ and tighten the hazard/gap-ACF floors to ±0.01 (the only way past the 1-day history cap).
2. If you still want a *cryptographic* verdict: stream a multi-million-bit file and run full Dieharder/NIST STS/TestU01 (would need the binaries installed).
3. Otherwise: **stop here.** The synthetic route is a rigorously-supported NULL for our prediction methods — redirect effort to the FX magnitude work (`DERIV_MAGNITUDE_MONETIZATION.md` Fork 1: the 1-day FX magnitude model), which at least has a real (vol-clustering) edge to build on.
