# THEORY — cross-key facts (GENERIC)

SCOPE: GENERIC (currency/timeframe-agnostic). The load-bearing theory that governs every key. Per-key
*measurements* that illustrate these facts are tagged `[PAIR·tf]` and their record-of-truth is the Tier-2
file (`<PAIR>_RESULTS.md`, `MAGNITUDE_FINDINGS.md`). See `REPO_MAP.md`.

---

## 1. Sign-invariance theorem (why most "direction" gauges are really MAGNITUDE)

`arXiv:2512.15720` (Dec 2025, SPY 38.5M trades): order-flow / permutation **ENTROPY is invariant under sign
permutation**, so it detects the *presence* of informed trading = **MAGNITUDE / volatility**, NOT **sign**.
Empirically there: at entropy<5th pct, `|5-min return|` ×2.89 (t=12.41) but directional accuracy ≈45% (chance).

**Generalization (the working law of this program):** entropy, order-flow imbalance, complexity (PE/RQA/LZ),
Hurst/DFA, Kalman/HMM **filter statistics**, signature **norms** — all gate move SIZE, not move SIGN.

**Operational test (apply to EVERY candidate before claiming "direction"):** the within-`|ret|`-bin **sign-shuffle
placebo**. Shuffle the sign of the label within magnitude bins; if the "edge" survives, it is a MAGNITUDE edge →
evaluate it as magnitude (`|ret|≥Q`, AUC) and record in `MAGNITUDE_FINDINGS.md`, NOT as direction. A genuine
*direction* mechanism must be **sign-AWARE** (e.g. signed Hawkes up/down cross-excitation asymmetry, signed
lead-lag / path signed-area, tail-conditional sign asymmetry) and must name which side's sign it carries.

**Cleanest single-method demonstration `[EURUSD·60s]` (2026-06-05):** the SAME 2-D bar-image CNN (Sezer CNN-BI /
GAF — a 4th model class beyond GBM/GRU/state-space) is **null on direction** (dirAUC ≈ .50, CPCV 0/28 paths clear
0.541, even the antisymmetric GADF sign-field) yet **clears >65% on magnitude** (large-vs-small move; magAUC
.699/.714/.686, selective precision .68→.80, all 28 CPCV paths ≥0.65 every held-out year). One representation, one
training pipeline, opposite verdicts by target — exactly the theorem. KEY detail: a bar image must keep its
**absolute volatility scale** to carry magnitude (per-window min-max normalization strips it → magAUC .64). See
`MAGNITUDE_FINDINGS.md` §3, `METHODS_CATALOG.md` §5.5, `barcnn_*.py`.

## 2. The direction ceiling (horizon-dependent; magnitude is the durable edge)

Working shape (currency-agnostic hypothesis; numbers below are `[EURUSD]`-measured evidence, see `EURUSD_RESULTS.md`):
- Sub-minute direction is **near-efficient** — AUC ≈0.50–0.51 across ~24 input channels (now incl. a 2-D bar-image
  CNN, the 4th model class — dirAUC ≈.50, CPCV 0/28 clear); `>0.65` OOS-stable is **not** achievable at ≤5m on
  clock-bar data. The best *available* 60s direction is the regime-gated UP dip-buy filter (uncertified: pooled .573,
  CPCV p10 .524, 2024 .520<breakeven); DOWN is dead. The only 60s edge that survives CPCV is **magnitude** (the
  Touch/Range straddle, ~.72–.80 selective precision). `[EURUSD·60s]`
- Short tick horizons (1–5s) carry a genuine `>0.65` directional edge but need a **tick venue** (deriv forex
  min duration is 15m — see §3). `[EURUSD·1-5s]`
- Intermediate horizons (≈10–15m) carry a modest **regime-dependent** direction edge (~0.58 cross-era / ~0.65
  recent). `[EURUSD·15m]` `[EURUSD·10m]`
- **CROSS-PAIR USD-COMMON-FACTOR is THE direction lever, and it is HORIZON-GATED — a gradient: none@60s → UP-only@5m
  → BOTH sides@10m, 15m & 30m; null again <5m.** Mechanism: the informed/jump component of a move (especially DOWN) is
  noise at short horizons but **averages out as the horizon lengthens**, so the slow USD-common-factor SIGN becomes
  forecastable at ≥10m. The carrier is the **CONCURRENT** cross-pair read (windows ending at t); strictly-LAGGED
  lead-lag is dominated. Per-fold-refit CPCV certified: 5m-UP (p10 .553), BOTH 10m sides (.586/.568), BOTH 15m sides
  (.567/.574), BOTH 30m sides (.5588/.5525 — the LONGEST deriv horizon, but REFIT-DEPENDENT: the frozen-2021 book's
  forward edge DECAYS by the 2026 OOS year, esp. UP → deploy with periodic retraining); KILLED at 2m (p10 ~.51).
  For a new (currency, ≥10m) key the cross-pair refit-CPCV side-split is the
  #1-prior lever. `[EURUSD·5m/10m/15m/2m]` (see `EURUSD_RESULTS.md`, `sweeps/EURUSD_{10,15}m.md`, `METHODS_CATALOG.md` A6)
- Once the cross-pair book certifies a ≥10m key, **loss/label/gate re-engineering does NOT beat the gated raw
  cross-pair sign** — magweight, GMADL/sign-coupled loss, residual-relabel, ACI gate, specialist, calibration,
  cross-horizon blend, lagged lead-lag, intraday-momentum all collapse on the binding-regime wall (a wrapper cannot
  create SIGN the regime erased). The redirect for a HIGHER number is **external data**, not another loss/gate variant.
  `[EURUSD·10m]` (11 levers dry, 2 rounds), `[EURUSD·30m]` (magweight HURTS — dilutes sign→magnitude; seed-ens/
  specialist/Aₐ/IPCA subsumed; 2 dry discovery rounds) — same outcome as 15m.
- **The cross-pair edge is carried by POOLED TRAINING on the BASE multi-TF features, NOT the cross-pair-specific
  features** (measured: frozen 30m cross-pair book feature-importance = base multi-TF **94%** gain vs lead-lag 2.1% /
  cross-pair-factor 2.0% / order-flow 1.8%). Pooling 6–7 USD-major rows gives more data + cross-sectional
  regularization of the SAME base features (esp. rescuing DOWN); the xpof factor/lead-lag block is near-zero gain.
  ⇒ factor-refinement levers (IPCA instrumented betas, antisymmetric lead-lag matrix Aₐ) are **subsumed** — they
  refine a ~2%-gain channel. CORRECTS the earlier speculation that factor/IPCA levers grow more relevant at longer H:
  what grows is the *pooling* benefit, not the factor features. `[EURUSD·30m]` (`m30_xpair_featimp_result.json`)
- **MAGNITUDE is the one CPCV-deflation-certified edge** at every horizon tested (large-move AUC ≈0.71–0.81).
  `[EURUSD·30m/60s]`
- Where a direction edge exists it is **regime-/horizon-specific** and may be **one-sided** (dip-buy UP at 60s/5m)
  OR **two-sided** (BOTH sides via cross-pair at 10m & 15m, DOWN as robust as UP) — never assume a side or copy one
  key's asymmetry to another; measure each side directly.

## 3. Deriv.com Rise/Fall settlement & breakeven (platform mechanics — generic)

- Settlement is **wall-clock mid-to-mid**: entry = NEXT tick after the order (+1s lag), exit = last tick ≤ expiry.
  No spread charged on settlement; **ties LOSE** (a flat outcome is a loss).
- **Breakeven win-rate ≈ 0.541** at payout R≈1.85 (R≈0.85 profit). Every certified edge must clear this on the
  binding (worst) held-out year's CI95-lower.
- **Forex minimum contract duration = 15m** on deriv; sub-15m edges need a different venue to be tradeable.
- Faithful evaluation MUST use `wc_ret()` (never `mid.shift(-N)` on gap-dropped bars — that shifts *bars* not
  *seconds*, the biggest historical inflation). See `METHODS_CATALOG.md` (validation family) for the full discipline.

## 4. corr(VAL, OOS) = −0.54 (selection anti-transfers)

Across the configs searched, validation accuracy is **negatively** correlated with OOS accuracy: VAL-maximal
pockets ANTI-transfer. Therefore SELECT thresholds/gates on the **worst-VAL-half**, never VAL-acc-max. This bites
purpose-built specialists hardest. (Generic; the −0.54 figure is `[EURUSD]`-measured.)

## 5. See also
- The 7 recurring leakage traps + the sign-invariance shuffle: `METHODS_CATALOG.md` (validation family) and the
  `strategy-eval` skill (`.claude/skills/strategy-eval/SKILL.md`).
- Multiple-testing deflation (Holm/BHY/HLZ/N̂, CSCV/PBO/DSR): `METHODS_CATALOG.md` validation family; per-key
  results in the key's backlog/results file.
