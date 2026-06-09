> **SCOPE: AUDUSD** (key-specific — results of record + UP/DOWN leaderboard). Generic methods: METHODS_CATALOG.md. Sweep menu: SWEEP_MATRIX.md. See REPO_MAP.md. Bootstrapped 2026-06-09 (new currency; structure copied from EURUSD_RESULTS.md / USDJPY_RESULTS.md). Current scope: **15m only** (deriv-FX-deployable floor + program's best direction horizon).

# AUDUSD — Results Ledger (unique key: currency × timeframe × side)

**Pair tag: `AUDUSD`.** AUDUSD is a **USD-QUOTE** pair (AUDUSD up ⇒ USD down ⇒ AUD up). It is simultaneously:
- a **USD major** — loads heavily on the USD common factor, so the **cross-pair POOLING / USD-residual lever (the EURUSD 15m certifying keystone)** is the strong prior here; and
- a **commodity / risk-on currency** — its idiosyncratic information (RBA policy, China data/PMI, AU CPI & jobs, iron-ore/metals, broad risk-on/off) arrives largely in the **ASIA session** (Sydney+Tokyo+China-open). This is the key contrast with EURUSD/USDJPY, whose certified direction edge is **NY-session-concentrated**. So for AUDUSD the **session question is genuinely open** and is tested symmetrically (ny/ldn/asia/overlap), not assumed-NY.
- NZDUSD is AUDUSD's closest cousin (Antipodean / commodity / risk twin; AUDNZD relative-value is a candidate own-pair direction channel).

**The unique result key is `(currency, timeframe, side)`, side ∈ {UP, DOWN}.** Where a key is unmeasured it is `UNTESTED`. COMBINED is context only, never an UP/DOWN number.

**Conventions.** Deriv-faithful settlement (bar-close approx at 15m, mid-to-mid, next-tick entry +1s, ties LOSE), breakeven **0.541** (deriv payout R≈1.85). Splits: bars train 2012-21 / val 2022-23 / test 2024 / test 2025 / oos 2026 (strict OOS). Selection on VAL worst-half (never VAL-acc-max; `corr(VAL,OOS)=−0.54`). Moved-bars-only; verify moved up-rate ∈ [0.47,0.53]; per-year CI95 (bootstrap). de-overlap = `nonoverlap_chrono` gap=900s. CERTIFY a side ONLY via full per-fold-refit CPCV at the operating gate: p10 ≥ 0.541 AND ≥ ~80% of 15 purged paths clear 0.541; adversarially verify every positive. **Target: >65% accuracy** (achievable, if at all, only at tight coverage — neighbors hit ~.60–.65 at cov2%). Bar features + raw ticks both on disk 2012–2026. Last updated 2026-06-09 (bootstrap).

---

## MASTER KEY TABLE — one row per (AUDUSD, timeframe, side)

| Key (currency · timeframe · side) | **Best OOS % (2026)** | Model id · content_id | Description | Status |
|---|---|---|---|---|
| AUDUSD · **15m** · UP | `UNTESTED` | — | Sweep in progress (base → session → cross-pair → improve). | UNTESTED |
| AUDUSD · **15m** · DOWN | `UNTESTED` | — | Sweep in progress. | UNTESTED |
| AUDUSD · 1m/2m/5m/10m/30m · UP/DOWN | `UNTESTED` | — | Out of current scope (goal = 15m). Bar + tick data present; bootstrap when scoped. | UNTESTED |

**Magnitude** (|ret|≥Q) is sign-invariant → no up/down key; tracked in `MAGNITUDE_FINDINGS.md` (AUDUSD pending).

---

# Per-timeframe detail

## AUDUSD × 15m
Breakeven 0.541. Deriv-FX-deployable (15m = forex Rise/Fall minimum expiry) AND the program's strongest direction horizon (EURUSD 15m cross-pair certified BOTH sides p10 .567/.574; USDJPY 15m NY own-pair seed-ens p10 .60/.57). **Mechanistic prior (above): pooling-keystone is the strong lever; session is the open AUDUSD-specific question (Asia vs NY).**

### Combined-book / single-pair experiments
| # | Method (file) | Result (2024 / 2025 / 2026 moved-AUC; gate cov2% COMB wr) | Verdict |
|---|---|---|---|
| BASE | single-pair LGBM, 239 base feats, 15m own-clock label, all-session (`audusd_15m_base.py`) | AUC .5266 / .5261 / .5186; cov2% COMB .6044[.574,.634] / .5872[.558,.616] / .5482[.503,.591] | ✅ **SURVIVED** standalone (COMBINED CI-lo clears BE in 2024 AND 2025; 2026 binding). up-rate .502/.506/.506 tripwire-clean. Real all-session 15m edge (unlike USDJPY which needed NY). `audusd_15m_base_result.json` |

### Coverage curve (step d) — `audusd_15m_base.py` covcurve (frozen-2012-21 book; per-year)
| cov | COMB wr 2024/25/26 | UP wr 2024/25/26 | DOWN wr 2024/25/26 |
|---|---|---|---|
| 0.10 | .548 / .545 / .521 | .542 / .545 / .519 | .554 / .545 / .521 |
| 0.05 | .568 / .571 / .528 | .560 / .576 / .527 | .572 / .568 / .529 |
| 0.03 | .578 / .572 / .524 | .579 / .581 / .500 | .577 / .568 / .537 |
| 0.02 | .586 / .583 / .559 | **.603 / .595 / .493** | .580 / .578 / **.589** |
| 0.01 | .600 / .569 / .562 | .552 / .584 / .516 | .610 / .567 / .578 |

**Read:** monotone-in-confidence, both sides ~.57–.60 @cov2–3% in 2024/2025. **2026 (binding) is DOWN-only** — DOWN holds (.589@cov2 / .578@cov1) while UP collapses (.493@cov2 / .516@cov1). The frozen-2021 book decays forward (refit-dependent, same as EURUSD/USDJPY 15m); DOWN survives the decay, UP does not. → certify via per-fold-REFIT CPCV (recovers per-era floor); DOWN is the more deployable side; UP is regime-dependent and must be confirmed forward (trap #9).

### All-session refit-CPCV (per-fold-refit, 15 purged paths) — `audusd_15m_cpcv_session.py all`
| cov | UP p10 (frac_clear) | DOWN p10 (frac) | COMB p10 (frac) | CERT |
|---|---|---|---|---|
| 0.05 | .5371 (.87) | .5354 (.73) | .5374 (.73) | ✗ |
| 0.03 | .5342 (.80) | .5405 (.87) | .5385 (.87) | ✗ |
| 0.02 | .5381 (.87) | .5345 (.73) | .5371 (.73) | ✗ |
| 0.01 | .5345 (.80) | **.544 (.93)** | **.542 (.87)** | **DOWN ✓ / COMB ✓** |

AUC mean .5213 (min .5142, max .5252), up-rate tripwire clean. **Read:** a real all-session edge but thin — **DOWN certifies at cov1%** (p10 .544, 14/15 paths; med_n 895/path), COMBINED at cov1% (p10 .542); **UP does NOT certify at any cov** (p10 saturates ~.534–.538, ~0.5–1pp sub-BE). DOWN>UP robustness (consistent with binding-2026 DOWN-only). This is the honest per-era refit floor (deployable w/ retrain). `audusd_15m_cpcv_session_all_multicov_result.json`. **NEXT levers to lift p10 to a usable coverage + rescue UP:** session concentration (A9), cross-pair pooling (A6), seed-ensemble (the USDJPY lever).

### Session segmentation (A9) — `audusd_15m_cpcv_session.py {ny,ldn,asia}` (symmetric)
_running — KEY AUDUSD-specific test: does the edge live in Asia (RBA/China/own-pair) or NY (USD-factor) or both? Does any session lift p10 over all-session's thin cov1% cert to a usable cov2–3%?_

### Cross-pair pooling (A6) — the EURUSD keystone, retargeted to AUDUSD
_pending — `audusd_15m_xpair.py` (USD-common-factor residual + lead-lag, AUDUSD target); + NZDUSD-cousin variant._

---

## UP/DOWN LEADERBOARD (current best per side, certified-or-best-available)
| Side | Best certified (refit-CPCV p10) | Best available (point) | Book | Status |
|---|---|---|---|---|
| **15m UP** | — | — | — | sweep in progress |
| **15m DOWN** | — | — | — | sweep in progress |

_Provenance: every number traces to a `*_result.json` (Tier-1). Updated as rows complete._
