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
| USDCAD · **15m** · UP | **NY seed-ens K=3 refit-CPCV p10 .5968 @cov2 / .604 @cov1 / .5891 @cov3 (15/15)** ✅ **CERTIFIED, REFIT-DEPENDENT** | NY own-pair LGBM **seed-ens K=3**, cov-gate (`usdcad_15m_cpcv_session.py ny 2 .. 3`); book `USDCAD.m15ny_seedens.v1` ✅ FROZEN (content_id ddb4a78c) | NY-concentrated own-pair GBM; all-session sub-BE, NY rescues (USDJPY pattern). UP is the stronger side on the refit floor (vindicates v0 USD-strength/risk-off prediction; frozen DOWN-lead was a 2024 artifact). Seed-ens ~parity vs single-seed (single-seed peak UP .6044@cov2). REFIT-DEPENDENT. | **CERTIFIED (NY seed-ens refit-CPCV) 2026-06-10** |
| USDCAD · **15m** · DOWN | **NY seed-ens K=3 refit-CPCV p10 .5814 @cov5 / .5805 @cov3 / .577 @cov2 (15/15)** ✅ **CERTIFIED, REFIT-DEPENDENT** | NY own-pair LGBM **seed-ens K=3**, cov-gate; book `USDCAD.m15ny_seedens.v1` ✅ FROZEN (content_id ddb4a78c) | NY-concentrated; certifies every cov (cov1 14/15). DOWN peaks at wider cov (cov5 .5814) vs UP at tight cov — slightly below UP on the refit floor. REFIT-DEPENDENT. | **CERTIFIED (NY seed-ens refit-CPCV) 2026-06-10** |
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

### All-session refit-CPCV (per-fold-refit, 15 purged paths) — `usdcad_15m_cpcv_session.py all`
| cov | UP p10 (frac_clear) | DOWN p10 (frac) | COMB p10 (frac) | CERT |
|---|---|---|---|---|
| 0.05 | .5295 (.40) | .5240 (.00) | .5290 (.13) | ✗ |
| 0.03 | .5325 (.53) | .5231 (.13) | .5298 (.33) | ✗ |
| 0.02 | .5278 (.73) | .5202 (.27) | .5305 (.67) | ✗ |
| 0.01 | .5277 (.67) | .5191 (.67) | .5329 (.80) | ✗ |

AUC mean .516 (min .5119, max .5185), up-rate tripwire clean. **Read: NO side certifies at any coverage** — every p10 sub-BE (.519–.533). All-session USDCAD 15m direction is essentially **efficient on-disk** (the per-era refit floor doesn't clear breakeven). This is the **USDJPY case** (all-session weak → needs NY-session restriction), NOT the AUDUSD case (whose DOWN certified all-session @cov1). **On the per-era REFIT floor UP ≥ DOWN** (UP p10 .527–.533 vs DOWN .519–.524) — REVERSING the frozen-book coverage curve where DOWN led; the frozen DOWN-strength was partly a 2024-regime artifact (refit removes it). `usdcad_15m_cpcv_session_all_multicov_result.json`. **NEXT: NY-session restriction** (USDCAD = most-NA pair → strongest concentration prior of any pair; running).

### Session segmentation (A9) — `usdcad_15m_cpcv_session.py ny` (symmetric refit-CPCV)
**RESULT: NY is the carrier — and certifies BOTH sides at every coverage, 15/15 paths.** All-session was sub-BE (efficient); NY-restriction rescues it (USDJPY pattern). USDCAD = most-NA pair (BoC+Fed+US/CA data+oil in the LDN/NY window), so the direction-sign rides US-session flow. (LDN/asia session-landscape confirmation pending.)

**NY-session refit-CPCV (the deliverable, single-seed), all covs, frac_clear=1.0 (15/15) every cov (DOWN cov1 14/15):**
| cov | UP p10 (mean) | DOWN p10 (mean) | COMB p10 (mean) | med_n/path (UP/DN/CB) | CERT |
|---|---|---|---|---|---|
| 0.05 | .5831 (.5985) | .5694 (.5898) | .5817 (.5933) | 1333/1163/2608 | ✅ both |
| 0.03 | .5972 (.6131) | **.5858 (.6057)** | .5942 (.6092) | 776/647/1526 | ✅ both |
| 0.02 | **.6044 (.6286)** | .5791 (.6178) | **.601 (.6228)** | 510/422/1009 | ✅ both |
| 0.01 | .5999 (.6373) | .5727 (.6269, .933) | .5987 (.6306) | 255/191/506 | ✅ both |

AUC mean NY .5319 (.527–.535), up-rate tripwire clean. **Both sides CERTIFIED (refit-CPCV p10 ≥ BE, 15/15 paths, every cov)** — REFIT-DEPENDENT (program-wide 15m signature; frozen-forward verification pending). **UP is the stronger side on the refit floor** (UP p10 .6044 vs DOWN .5791 @cov2) — VINDICATING the v0 mechanistic prediction (UP = USD-strength/risk-off the more forecastable side); the frozen-book coverage curve's DOWN-lead was a 2024-regime artifact the per-fold refit removes. Strong vs neighbors (UP .604 > AUDUSD .596 / USDJPY UP .60). Operating cov2% (best UP+COMB p10). `usdcad_15m_cpcv_session_ny_multicov_result.json`.

### ★ IMPROVE — seed-ensemble K=3 on NY (Tier-I I2) — ~PARITY (FROZEN as deliverable for robustness)
Matched-fold NY refit-CPCV, K=3 seed-average vs single-seed. **p10 deltas mixed; means uniformly slightly up:**
| cov | UP p10 (Δ K1) | DOWN p10 (Δ) | COMB p10 (Δ) | COMB mean (Δ) |
|---|---|---|---|---|
| 0.05 | .5902 (+.0071) | .5814 (+.0120) | .5925 (+.0108) | .6015 (+.008) |
| 0.03 | .5891 (−.0081) | .5805 (−.0053) | .5928 (−.0014) | .6119 (+.003) |
| 0.02 | .5968 (−.0076) | .5770 (−.0021) | .6025 (+.0015) | .6240 (+.001) |
| 0.01 | .6040 (+.0041) | .5536 (−.0191, .933) | .5929 (−.0058) | .6303 (−.000) |

frac_clear=1.0 every cell (DOWN cov1 .933). **Read:** seed-ens LIFTS at cov5 (all sides p10+mean) but is flat-to-slightly-down at the tight cov2-3 operating points (cov2 UP p10 .597 vs single-seed .604); all path-MEANS rise +.001..+.009 (except cov1 DOWN). This is **WEAKER than AUDUSD's clean seed-ens lift** — the p10 wiggles sit within CPCV-path noise; the means are uniformly marginally up. **Verdict: ~PARITY** — not a robust p10 improvement at the tight operating point, but a genuine small variance-reduction (means + cov5). **FROZEN as the deliverable** (`USDCAD.m15ny_seedens.v1`) for deployment robustness (seed-average is not seed-luck-dependent) + program consistency; reported honestly as marginal. `usdcad_15m_cpcv_session_ny_seedens3_result.json`. **K=8 DEFERRED low-EV** (saturation already evident at K=3; 239-feat own-pair saturated at K=3 on AUDUSD+USDJPY; only GBPUSD's 340-feat xpair lifted at K=8). Improve cross-product continues: cross-pair (R1-2), two-speed (R1-1), kNN/double-ortho (R1-3/4) below.

### ★ Adversarial verification — NY FROZEN-PAST forward holdout (trap#9) — `usdcad_15m_freeze.py` (book `USDCAD.m15ny_seedens.v1`, content_id ddb4a78c7ea95439)
Train ONCE on 2012-21 NY (seed-ens K=3), FREEZE, test per-year NY @cov2 (deployment-faithful, NO retrain):
| year | cov2 COMB (CI-lo) | cov2 UP | cov2 DOWN |
|---|---|---|---|
| 2024 | .6931 (.657) | .6658 | .7487 |
| 2025 | .5696 (.538) | .5606 | .5851 |
| **2026** | **.5519 (.506)** | **.5714** | .5258 |

**VERDICT: the refit-CPCV cert is REFIT-DEPENDENT, NOT a frozen-deployable edge** (program-wide 15m signature). The frozen-2012-21 vintage DECAYS forward (.693→.570→.552 COMB) — but this is **NOT trap#9 era-local memorization** (that would be flat ~.50 every year): the edge transfers to 2024 (.69) and 2025 (.57) then decays = USD/oil-factor non-stationarity (identical to EURUSD/USDJPY/AUDUSD 15m). **USDCAD's frozen forward is BETTER than AUDUSD's** (which went sub-BE on both sides by 2026): USDCAD **UP survives frozen to 2026 (.5714 > BE)** and COMB clears marginally (.5519), only DOWN goes sub-BE (.5258) — again confirming UP is the more robust side. The frozen-forward (.571/.526) sits BELOW the refit-CPCV p10 (.597/.581), confirming the cert = the per-era refit floor. **Honest deployment:** deploy NY-only **with periodic retraining**; size on the refit per-era floor (UP ~.59-.60, DOWN ~.58), NOT the frozen book. No USDCAD tick data on disk → true-tick-settlement validation N/A; bar-close proxy validated on USDJPY/AUDUSD (mean tick−bar −.0035, PRESERVED) → trusted at 15m.

### Cross-pair pooling (A6 / R1-2) — the EUR-bloc keystone, retargeted to USDCAD (sign-aligned, USD-numerator) → **POOLING DOES NOT ADD (own-pair-specific, USDJPY/AUDUSD case)**
`usdcad_15m_xpair.py xpbase` (cross-pair USD-residual + lead-lag + commodity-bloc/oil-proxy + risk factor, sign-aligned via aln(p)=equiv_sign(TARGET)·equiv_sign(p) so USDCAD stays in its raw USD-strength frame; all-session screen). VAL AUC **.5244** vs base .5234 (+.0010, negligible). **top-20 features are mostly own-pair base** (hour_cos/sin, 4h_autocorr, bb_width, dist_ema, rangepos, rsi, rv) — though the cross-pair lead-lag (ll_AUDUSD/GBPUSD/EURUSD), catchup1, tgtresid3, cadcommod_resid3 DO reach top-20 (more than AUDUSD where they didn't), they carry no net VAL direction and **HURT 2026**. Per-year cov2%: 2024 COMB .6006/UP .581/DOWN .628; 2025 .557/.559/.555; 2026 **.5112/UP .525/DOWN .496 (collapses, below base .537)**. **VERDICT: IMPROVES_base=False** — the GBM picks up the USD-factor pool but it doesn't carry incremental 15m sign and overfits 2026. **USDCAD is the USDJPY/AUDUSD case (own-pair-specific), NOT the EURUSD/GBPUSD case (poolable).** No escalation to NY refit-CPCV. `usdcad_15m_xpair_xpbase_result.json`. (R1-4 double-orthogonalized oil-proxy residual running as confirm — single-purge showed no DOWN lift, low EV.)

_(tables populated as rows complete — see sweep ledger `sweeps/USDCAD_15m.md` for status of record)_

---

## UP/DOWN LEADERBOARD (current best per side, certified-or-best-available)
| Side | Best certified (refit-CPCV p10) | Best available (mean) | Book | Status |
|---|---|---|---|---|
| **15m UP** | **.5968 @cov2 / .604 @cov1 / .5891 @cov3** (NY seed-ens K=3, 15/15) | .6294 / .6380 mean (cov2/1) | **`USDCAD.m15ny_seedens.v1`** ✅ FROZEN (content_id ddb4a78c) | ✅ CERTIFIED (NY seed-ens refit-CPCV); REFIT-DEPENDENT |
| **15m DOWN** | **.5814 @cov5 / .5805 @cov3 / .577 @cov2** (NY seed-ens K=3, 15/15) | .5972 / .6092 mean (cov5/3) | **`USDCAD.m15ny_seedens.v1`** ✅ FROZEN (content_id ddb4a78c) | ✅ CERTIFIED (NY seed-ens refit-CPCV); REFIT-DEPENDENT |

_Provenance: every number traces to a `*_result.json` (Tier-1). Updated as rows complete._
