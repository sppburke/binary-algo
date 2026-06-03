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

## 2. The direction ceiling (horizon-dependent; magnitude is the durable edge)

Working shape (currency-agnostic hypothesis; numbers below are `[EURUSD]`-measured evidence, see `EURUSD_RESULTS.md`):
- Sub-minute direction is **near-efficient** — AUC ≈0.50–0.51 across ~24 input channels; `>0.65` OOS-stable is
  **not** achievable at ≤5m on clock-bar data. `[EURUSD·60s]`
- Short tick horizons (1–5s) carry a genuine `>0.65` directional edge but need a **tick venue** (deriv forex
  min duration is 15m — see §3). `[EURUSD·1-5s]`
- Intermediate horizons (≈15m) carry a modest **regime-dependent** direction edge (~0.58 cross-era / ~0.65
  recent). `[EURUSD·15m]`
- **MAGNITUDE is the one CPCV-deflation-certified edge** at every horizon tested (large-move AUC ≈0.71–0.81).
  `[EURUSD·30m/60s]`
- Where a direction edge exists it tends to be **one-sided** (e.g. dip-buy UP) and **regime-/horizon-specific** —
  never assume a side or copy one key's asymmetry to another.

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
