> **SCOPE: GBPUSD** (key-specific — results of record + UP/DOWN leaderboard). Generic methods: METHODS_CATALOG.md. Sweep menu: SWEEP_MATRIX.md. See REPO_MAP.md. Bootstrapped 2026-06-09 (new currency; structure copied from AUDUSD_RESULTS.md / EURUSD_RESULTS.md). Current scope: **15m only** (deriv-FX-deployable floor + program's best direction horizon).

# GBPUSD — Results Ledger (unique key: currency × timeframe × side)

**Pair tag: `GBPUSD`.** GBPUSD ("cable") is a **USD-QUOTE** pair (GBPUSD up ⇒ USD down ⇒ GBP up). It is simultaneously:
- a **USD major in the EUROPEAN bloc** — the tightest EURUSD correlate among the untested majors. Cross-pair POOLING / USD-residual was THE certifying keystone at **EURUSD 15m** (UP p10 .567 / DOWN .574, the one sibling where pooling won) but **DILUTED** own-pair signal at USDJPY and AUDUSD. GBPUSD sits squarely between those precedents → **the pooling lever has a genuinely open prior here** and is run, not argued. The **EURGBP triangular residual** (USD leg algebraically cancelled, SWEEP_MATRIX N2) is uniquely on-disk-runnable for this pair.
- a currency whose **idiosyncratic information (BoE policy, UK CPI/GDP/labour, gilt-market/politics risk premia) arrives largely in the LONDON morning** — yet the certified direction edge was **NY-session-concentrated at ALL THREE siblings** (EURUSD/USDJPY/AUDUSD; LDN+Asia null every time). So the session question is tested symmetrically (ny/ldn/asia/all), not assumed.

**The unique result key is `(currency, timeframe, side)`, side ∈ {UP, DOWN}.** Where a key is unmeasured it is `UNTESTED`. COMBINED is context only, never an UP/DOWN number.

**Conventions.** Deriv-faithful settlement (bar-close approx at 15m, mid-to-mid, next-tick entry +1s, ties LOSE), breakeven **0.541** (deriv payout R≈1.85). Splits: bars train 2012-21 / val 2022-23 / test 2024 / test 2025 / oos 2026 (strict OOS). Selection on VAL worst-half (never VAL-acc-max; `corr(VAL,OOS)=−0.54`). Moved-bars-only; verify moved up-rate ∈ [0.47,0.53]; per-year CI95 (bootstrap). de-overlap = `nonoverlap_chrono` gap=900s. CERTIFY a side ONLY via full per-fold-refit CPCV at the operating gate: p10 ≥ 0.541 AND ≥ ~80% of 15 purged paths clear 0.541; adversarially verify every positive. **Target: >65% accuracy** (achievable, if at all, only at tight coverage — siblings hit ~.60–.65 at cov2%). Bar features (`features/GBPUSD_2012..2026.parquet`, 244 cols) + raw 10s ticks (`/home/sean/git/processed/GBPUSD/`, 2012-01→2026-06) both on disk. Last updated 2026-06-09 (bootstrap).

---

## MASTER KEY TABLE — one row per (GBPUSD, timeframe, side)

| Key (currency · timeframe · side) | **Best OOS % (2026)** | Model id · content_id | Description | Status |
|---|---|---|---|---|
| GBPUSD · **15m** · UP | **all-session refit-CPCV p10 .5532 @cov2 / .5531 @cov3 (14/15)** ✅ CERTIFIED, REFIT-DEPENDENT | all-session own-pair LGBM s2/l255, cov-gate (`gbpusd_15m_cpcv_session.py all 2`); book not yet frozen — session/pooling LIFT tests running | First sibling to certify BOTH sides all-session. Frozen-2021 book dead by 2026 → deploy w/ periodic retrain. | ✅ CERTIFIED (all-session refit-CPCV) 2026-06-09; lift tests in progress |
| GBPUSD · **15m** · DOWN | **all-session refit-CPCV p10 .5527 @cov2 / .5483 @cov3 (14/15, cov3 15/15)** ✅ CERTIFIED, REFIT-DEPENDENT | same harness/book as UP | Confidence-tail-sensitive (cov5/cov1 drop out); certifies cov2-3. | ✅ CERTIFIED (all-session refit-CPCV) 2026-06-09; lift tests in progress |
| GBPUSD · 1m/2m/5m/10m/30m · UP/DOWN | `UNTESTED` | — | Out of current scope (goal = 15m). Bar + tick data present; bootstrap when scoped. | UNTESTED |

**Magnitude** (|ret|≥Q) is sign-invariant → no up/down key; tracked in `MAGNITUDE_FINDINGS.md` (GBPUSD pending).

---

# Per-timeframe detail

## GBPUSD × 15m
Breakeven 0.541. Deriv-FX-deployable (15m = forex Rise/Fall minimum expiry) AND the program's strongest direction horizon (EURUSD 15m cross-pair certified BOTH sides p10 .567/.574; USDJPY 15m NY own-pair p10 .586/.572; AUDUSD 15m NY own-pair seed-ens K=3 p10 .596/.596). **Mechanistic prior (above): pooling is the genuinely open EUR-bloc question; session tested symmetrically (LDN-morning info vs the 3-sibling NY precedent).**

### Combined-book / single-pair experiments
| # | Method (file) | Result (2024 / 2025 / 2026 moved-AUC; gate cov2% COMB wr) | Verdict |
|---|---|---|---|
| BASE | single-pair LGBM, 239 base feats, 15m own-clock label, all-session (`gbpusd_15m_base.py`) | AUC .5262 / .5250 / .5098; cov2% COMB .6489[.624,.673] / .5582[.536,.580] / .5052[.470,.541] | ✅ **SURVIVED** standalone via 2024 (CI-lo .624 ≫ BE — strongest sibling-family 2024 base); 2025 point>BE, CI-lo misses; **2026 frozen book DEAD (~.50 both sides)** = the sibling-universal refit-dependence. up-rate .504/.505/.498 tripwire-clean. VAL moved-AUC .5285, best_iter 33. `gbpusd_15m_base_result.json` |

### Coverage curve (step d) — `gbpusd_15m_base.py` covcurve (frozen-2012-21 book; per-year)
| cov | COMB wr 2024/25/26 | UP wr 2024/25/26 | DOWN wr 2024/25/26 |
|---|---|---|---|
| 0.10 | .556 / .533 / .520 | .558 / .527 / .515 | .553 / .542 / .526 |
| 0.05 | .586 / .559 / .514 | .583 / .558 / .515 | .591 / .560 / .513 |
| 0.03 | .622 / .564 / .508 | .622 / .567 / .515 | .622 / .559 / .500 |
| 0.02 | .653 / .589 / .501 | .656 / .584 / .518 | .650 / .595 / .483 |
| 0.01 | .686 / .599 / .532 | **.720** / .612 / .530 | .647 / .582 / .534 |

**Read:** monotone-in-confidence and SYMMETRIC across sides in 2024/2025 (both ~.58–.66 @cov2–3%); 2024 is the strongest base year any sibling has shown (cov1 UP .720, n385). The frozen-2021 book decays to coin-flip by 2026 at every coverage — pure refit-dependence (USD-factor non-stationarity, same as EURUSD/USDJPY/AUDUSD 15m). → certify via per-fold-REFIT CPCV; session + pooling are the rescue levers.

### All-session refit-CPCV (per-fold-refit, 15 purged paths) — `gbpusd_15m_cpcv_session.py all 2`
| cov | UP p10 (frac_clear) | DOWN p10 (frac) | COMB p10 (frac) | CERT |
|---|---|---|---|---|
| 0.05 | **.5507 (.93)** | .5391 (.73) | **.5443 (.93)** | **UP ✓ / COMB ✓** |
| 0.03 | **.5531 (.93)** | **.5483 (1.0)** | **.5517 (.93)** | **BOTH ✓ / COMB ✓** |
| 0.02 | **.5532 (.93)** | **.5527 (.93)** | **.5557 (.93)** | **BOTH ✓ / COMB ✓** |
| 0.01 | **.5482 (.93)** | .5392 (.80) | **.552 (1.0)** | **UP ✓ / COMB ✓** |

AUC mean .5243 (min .5176, max .529), up-rate tripwire clean, med_n/path cov2: UP 2053 / DOWN 1893 / COMB 4007 (well-powered). **Read: BOTH SIDES CERTIFIED ALL-SESSION at cov2 AND cov3** — the strongest all-session 15m floor of the four pairs (AUDUSD all-session was DOWN-only @cov1 thin; USDJPY UP-only; EURUSD needed pooling). Means run ~.555–.569 (cov2 COMB mean .567). DOWN is confidence-tail-sensitive (drops out at cov5/cov1); UP certifies everywhere probed. The refit floor (~.55) vs the dead frozen-2026 book (~.50) confirms refit-dependence. Session concentration + pooling are LIFT levers on an already-certified floor. `gbpusd_15m_cpcv_session_all_multicov_result.json`.

## UP/DOWN LEADERBOARD (current best per side, certified-or-best-available)
| Side | Best certified (refit-CPCV p10) | Best available (mean) | Book | Status |
|---|---|---|---|---|
| **15m UP** | **.5532 @cov2 / .5531 @cov3** (all-session own-pair, 14/15) | .568 / .561 mean | not yet frozen (session/pooling lift tests running) | ✅ CERTIFIED (all-session refit-CPCV), refit-dependent |
| **15m DOWN** | **.5527 @cov2 / .5483 @cov3** (all-session own-pair, 14/15, cov3 15/15) | .567 / .563 mean | not yet frozen (session/pooling lift tests running) | ✅ CERTIFIED (all-session refit-CPCV), refit-dependent |

_Provenance: every number traces to a `*_result.json` (Tier-1). Updated as rows complete._
