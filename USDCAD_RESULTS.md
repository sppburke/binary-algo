> **SCOPE: USDCAD** (key-specific — results of record + UP/DOWN leaderboard). Generic methods: METHODS_CATALOG.md. Sweep menu: SWEEP_MATRIX.md. See REPO_MAP.md. Bootstrapped 2026-06-10 (new currency; structure copied from AUDUSD_RESULTS.md / USDJPY_RESULTS.md). Current scope: **15m only** (deriv-FX-deployable floor + program's best direction horizon).

# USDCAD — Results Ledger (unique key: currency × timeframe × side)

**Pair tag: `USDCAD`.** USDCAD is a **USD-BASE / CAD-QUOTE** pair (USDCAD up ⇒ USD strengthens vs CAD ⇒ "USD up / CAD down"). It is **doubly determined** and sits between the program's two precedents:
- a **USD major with USD on the NUMERATOR** (like USDJPY) — loads on the USD common factor, so the **cross-pair POOLING / USD-residual lever (the EURUSD/GBPUSD 15m certifying keystone)** is a candidate. BUT both closest analogs — USDJPY (USD-base) and AUDUSD (commodity) — were **OWN-PAIR-SPECIFIC** (pooling DILUTED). Strong prior: USDCAD is own-pair-specific too → pooling NULL. RUN it (don't argue). **Sign caution:** USDCAD has USD in the numerator, so the USD-residual sign is INVERTED vs AUDUSD/NZDUSD (USD in denominator) — cross-pair construction must be sign-aligned.
- a **COMMODITY / petrocurrency** — CAD is oil-linked (WTI down ⇒ CAD weak ⇒ USDCAD UP). **Oil/WTI is the dominant external signed driver, OFF-DISK** (acquisition frontier, the USDCAD analog of AUDUSD's iron-ore / AU-US rate-diff).
- the **MOST North-American pair** (both legs trade US/Canada hours; BoC + Fed + US/CA data + oil all hit the LDN/NY window) → the **NY-session-concentration prior is STRONGER here than for any prior pair**. Asia is a very low prior for CAD.

**Side-asymmetry prediction (falsifiable).** AUDUSD (AUD base) found **DOWN** (= risk-off / USD-strength, the sharp move) the more forecastable side. USDCAD has USD on the OPPOSITE side of the quote, so the analogous sharp risk-off / oil-down / USD-strength move is USDCAD **UP**. **Prediction: USDCAD UP is the more forecastable/robust side (inverse of AUDUSD).** Tested in the side-split.

**The unique result key is `(currency, timeframe, side)`, side ∈ {UP, DOWN}.** Where a key is unmeasured it is `UNTESTED`. COMBINED is context only, never an UP/DOWN number.

**Conventions.** Deriv-faithful settlement (bar-close approx at 15m, mid-to-mid, next-tick entry +1s, ties LOSE), breakeven **0.541** (deriv payout R≈1.85). Splits: bars train 2012-21 / val 2022-23 / test 2024 / test 2025 / oos 2026 (strict OOS). Selection on VAL worst-half (never VAL-acc-max; `corr(VAL,OOS)=−0.54`). Moved-bars-only; verify moved up-rate ∈ [0.47,0.53] (USDCAD checked clean: .506/.501/.501 for 2018/2024/2026); per-year CI95 (bootstrap). de-overlap = `nonoverlap_chrono` gap=900s. CERTIFY a side ONLY via full per-fold-refit CPCV at the operating gate: p10 ≥ 0.541 AND ≥ ~80% of 15 purged paths clear 0.541; adversarially verify every positive. **Target: >65% accuracy** (achievable, if at all, only at tight coverage — neighbors hit ~.60–.65 at cov1–2%). Bar features on disk 2012–2026 (no tick data for USDCAD). Last updated 2026-06-10 (bootstrap).

---

## MASTER KEY TABLE — one row per (USDCAD, timeframe, side)

| Key (currency · timeframe · side) | **Best OOS % (2026)** | Model id · content_id | Description | Status |
|---|---|---|---|---|
| USDCAD · **15m** · UP | `UNTESTED` | — | Sweep in progress (2026-06-10). Predicted more-forecastable side (risk-off/oil-down/USD-strength). | UNTESTED |
| USDCAD · **15m** · DOWN | `UNTESTED` | — | Sweep in progress (2026-06-10). | UNTESTED |
| USDCAD · 1m/2m/5m/10m/30m · UP/DOWN | `UNTESTED` | — | Out of current scope (goal = 15m). Bar data present (no tick). Bootstrap when scoped. | UNTESTED |

**Magnitude** (|ret|≥Q) is sign-invariant → no up/down key; tracked in `MAGNITUDE_FINDINGS.md` (USDCAD pending).

---

# Per-timeframe detail

## USDCAD × 15m
Breakeven 0.541. Deriv-FX-deployable (15m = forex Rise/Fall minimum expiry) AND the program's strongest direction horizon (EURUSD 15m xpair certified BOTH sides p10 .567/.574; USDJPY 15m NY own-pair seed-ens p10 .60/.57; AUDUSD 15m NY own-pair seed-ens p10 .596/.596; GBPUSD 15m xpair-NY seed-ens K=8 p10 .655/.640). **Mechanistic prior (above): own-pair NY-concentrated is the strong prior (USDCAD = most-NA pair; both USD-base + commodity analogs were own-pair-specific); pooling a low-prior RUN; oil/WTI the off-disk frontier; UP predicted the more forecastable side.**

### Combined-book / single-pair experiments
| # | Method (file) | Result (2024 / 2025 / 2026 moved-AUC; gate cov2% COMB wr) | Verdict |
|---|---|---|---|
| BASE | single-pair LGBM, 239 base feats, 15m own-clock label, all-session (`usdcad_15m_base.py`) | AUC .5229 / .5223 / .5158; frozen-gate cov2% COMB .6496[.622,.677] / .5518[.527,.576] / .5453[.508,.581] | ✅ **SURVIVED** standalone (COMBINED CI-lo clears BE in 2024; 2025/2026 binding sub-BE = refit-dependent decay). up-rate .501/.499/.501 tripwire-clean. VAL moved-AUC .5234 (>.515). Real all-session 15m edge, thin (AUC ~.52), decays forward. `usdcad_15m_base_result.json` |

### Coverage curve (step d) — `usdcad_15m_base.py` covcurve (frozen-2012-21 book; per-year, per-cov thr)
| cov | COMB wr 2024/25/26 | UP wr 2024/25/26 | DOWN wr 2024/25/26 |
|---|---|---|---|
| 0.10 | .552 / .526 / .533 | .557 / .533 / .517 | .545 / .518 / .551 |
| 0.05 | .578 / .541 / .555 | .573 / .543 / .549 | .589 / .538 / .564 |
| 0.03 | .621 / .552 / .545 | .606 / .557 / .541 | .657 / .543 / .551 |
| 0.02 | .639 / .553 / .537 | .617 / .543 / .547 | **.709 / .582** / .513 |
| 0.01 | .661 / .560 / .556 | .644 / .550 / .545 | **.750 / .608 / .654** |

n@cov2 (COMB/UP/DOWN): 2024 1248/953/295, 2025 1142/829/313, 2026 413/300/113 (cov1 ≈ half → DOWN cov1 thin, flag).

**Read:** monotone-in-confidence (real edge), thin (AUC ~.52). 2024 very strong (DOWN .709@cov2 / .750@cov1). **Binding years 2025/2026 hover at/just-below BE frozen** (COMB cov1 .560/.556) — refit-dependent decay (frozen-2021 vintage, same signature as EURUSD/USDJPY/AUDUSD 15m). **Side-asymmetry: DOWN is the stronger side at tight cov in ALL THREE years** (cov1 DOWN .750/.608/.654 vs UP .644/.550/.545) — this **REVERSES the v0 prediction (UP)**. Mechanistic update: USDCAD's forecastable side is **CAD-strength (oil-up / USDCAD-down)**, not USD-strength — plausibly the USD-strength side is shared/diluted across all USD-majors while CAD-strength is oil-idiosyncratic. → certify via per-fold-REFIT CPCV (recovers per-era floor); DOWN looks the more deployable side; confirm both forward (trap#9). Next levers: session concentration (A9, NY strong prior), seed-ensemble, cross-pair (low prior).

_(tables populated as rows complete — see sweep ledger `sweeps/USDCAD_15m.md` for status of record)_

---

## UP/DOWN LEADERBOARD (current best per side, certified-or-best-available)
| Side | Best certified (refit-CPCV p10) | Best available (mean) | Book | Status |
|---|---|---|---|---|
| **15m UP** | `UNTESTED` | — | — | sweep in progress |
| **15m DOWN** | `UNTESTED` | — | — | sweep in progress |

_Provenance: every number traces to a `*_result.json` (Tier-1). Updated as rows complete._
