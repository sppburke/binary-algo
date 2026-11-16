> **SCOPE: AUDUSD** (key-specific — results of record + UP/DOWN leaderboard). Generic methods: docs/METHODS_CATALOG.md. Sweep menu: SWEEP_MATRIX.md. See REPO_MAP.md. Bootstrapped 2026-06-09 (new currency; structure copied from EURUSD_RESULTS.md / USDJPY_RESULTS.md). Current scope: **15m only** (deriv-FX-deployable floor + program's best direction horizon).

# AUDUSD — Results Ledger (unique key: currency × timeframe × side)

**Pair tag: `AUDUSD`.** AUDUSD is a **USD-QUOTE** pair (AUDUSD up ⇒ USD down ⇒ AUD up). It is simultaneously:
- a **USD major** — loads heavily on the USD common factor, so the **cross-pair POOLING / USD-residual lever (the EURUSD 15m certifying keystone)** is the strong prior here; and
- a **commodity / risk-on currency** — its idiosyncratic information (RBA policy, China data/PMI, AU CPI & jobs, iron-ore/metals, broad risk-on/off) arrives largely in the **ASIA session** (Sydney+Tokyo+China-open). This is the key contrast with EURUSD/USDJPY, whose certified direction edge is **NY-session-concentrated**. So for AUDUSD the **session question is genuinely open** and is tested symmetrically (ny/ldn/asia/overlap), not assumed-NY.
- NZDUSD is AUDUSD's closest cousin (Antipodean / commodity / risk twin; AUDNZD relative-value is a candidate own-pair direction channel).

**The unique result key is `(currency, timeframe, side)`, side ∈ {UP, DOWN}.** Where a key is unmeasured it is `UNTESTED`. COMBINED is context only, never an UP/DOWN number.

**Conventions.** Deriv-faithful settlement (bar-close approx at 15m, mid-to-mid, next-tick entry +1s, ties LOSE), breakeven **0.541** (deriv payout R≈1.85). Splits: bars train 2012-21 / val 2022-23 / test 2024 / test 2025 / oos 2026 (strict OOS). Selection on VAL worst-half (never VAL-acc-max; `corr(VAL,OOS)=−0.54`). Moved-bars-only; verify moved up-rate ∈ [0.47,0.53]; per-year CI95 (bootstrap). de-overlap = `nonoverlap_chrono` gap=900s. CERTIFY a side ONLY via full per-fold-refit CPCV at the operating gate: p10 ≥ 0.541 AND ≥ ~80% of 15 purged paths clear 0.541; adversarially verify every positive. **Target: >65% accuracy** (achievable, if at all, only at tight coverage — neighbors hit ~.60–.65 at cov2%). Bar features + raw ticks both on disk 2012–2026. Last updated 2026-06-09 (bootstrap).

---

## MASTER KEY TABLE — one row per (AUDUSD, timeframe, side)

> **Deriv-feed program (2026-11-15):** the numbers in this table are Dukascopy-era evidence. On the Deriv feed (DEV 2025-08-11..2026-07-31) none of this pair's frozen 15m/30m books is SUPPORTED as deployed — see "AUDUSD — DERIV-FEED PROGRAM" at the end of this file.

| Key (currency · timeframe · side) | **Best OOS % (2026)** | Model id · content_id | Description | Status |
|---|---|---|---|---|
| AUDUSD · **15m** · UP | **NY seed-ens(K=3) refit-CPCV p10 .596 @cov2 / .576 @cov5 (15/15)** ✅ **CERTIFIED, REFIT-DEPENDENT** | NY own-pair LGBM **seed-ens K=3**, cov-gate (`audusd_15m_cpcv_session.py ny 2 .. 3`); book `AUDUSD.m15ny_seedens.v1` ✅ FROZEN (content_id 9b0e0ed3) | NY-concentrated own-pair GBM; all-session UP was sub-BE, NY rescues it; seed-ens lifts p10 (+.013 @cov2). p10 ≥ BE every cov, all 15 paths. **Frozen-2021 fwd decays (.61→.58→.53 cov5), sub-BE by 2026 → deploy w/ periodic retrain, size on refit floor.** | **CERTIFIED (NY seed-ens refit-CPCV), refit-dependent 2026-06-09** |
| AUDUSD · **15m** · DOWN | **NY seed-ens(K=3) refit-CPCV p10 .596 @cov2 / .587 @cov5 (15/15)** ✅ **CERTIFIED, REFIT-DEPENDENT** | NY own-pair LGBM **seed-ens K=3**, cov-gate; book `AUDUSD.m15ny_seedens.v1` ✅ FROZEN (content_id 9b0e0ed3) | NY-concentrated; the more robust side (also all-session @cov1%); seed-ens lifts p10 (+.010 @cov5). p10 ≥ BE every cov, all 15 paths. **Frozen-2021 fwd decays (.60→.56→.53 cov5) → deploy w/ periodic retrain, size on refit floor.** | **CERTIFIED (NY seed-ens refit-CPCV), refit-dependent 2026-06-09** |
| AUDUSD · 1m/2m/5m/10m/30m · UP/DOWN | `UNTESTED` | — | Out of current scope (goal = 15m). Bar + tick data present; bootstrap when scoped. | UNTESTED |

**Magnitude** (|ret|≥Q) is sign-invariant → no up/down key; tracked in `docs/MAGNITUDE_FINDINGS.md` (AUDUSD pending).

---

# Per-timeframe detail

## AUDUSD × 15m
Breakeven 0.541. Deriv-FX-deployable (15m = forex Rise/Fall minimum expiry) AND the program's strongest direction horizon (EURUSD 15m cross-pair certified BOTH sides p10 .567/.574; USDJPY 15m NY own-pair seed-ens p10 .60/.57). **Mechanistic prior (above): pooling-keystone is the strong lever; session is the open AUDUSD-specific question (Asia vs NY).**

### Combined-book / single-pair experiments
| # | Method (file) | Result (2024 / 2025 / 2026 moved-AUC; gate cov2% COMB wr) | Verdict |
|---|---|---|---|
| BASE | single-pair LGBM, 239 base feats, 15m own-clock label, all-session (`audusd_15m_base.py`) | AUC .5266 / .5261 / .5186; cov2% COMB .6044[.574,.634] / .5872[.558,.616] / .5482[.503,.591] | ✅ **SURVIVED** standalone (COMBINED CI-lo clears BE in 2024 AND 2025; 2026 binding). up-rate .502/.506/.506 tripwire-clean. Real all-session 15m edge (unlike USDJPY which needed NY). `audusd_15m_base_result.json` |
| R9 | issue #9 date-only retrospective replay: A=current v1; B=clean capped legacy-date refit; C=clean capped date-refresh refit | Apr–May 2026 standardized whole-book correctness: A .5772 (n=149, yield .001039); B .5846 (n=130, .000993); C .4971 (n=173, −.000045). C emitted UP only: .4971 (n=173); DOWN n=0. | ⚪ **INCONCLUSIVE — `required_endpoint_not_estimable`.** Simultaneous promotion lower bounds: C−A yield −.003117, C−B yield −.002574, C-UP accuracy−.5 −.091468; C-DOWN undefined. No survivor/candidate; incumbent unchanged. Pair: `results/json/m15_book_refresh_53547498c599f2877a2f6616ae725f8f27e11070e3b550e82eb26ec039fb23f6_AUDUSD_replay_result.json`; joint/shadow: `results/json/m15_book_refresh_53547498c599f2877a2f6616ae725f8f27e11070e3b550e82eb26ec039fb23f6_joint_replay_result.json`, `results/json/m15_book_refresh_53547498c599f2877a2f6616ae725f8f27e11070e3b550e82eb26ec039fb23f6_shadow_spec_result.json`. |

### Coverage curve (step d) — `audusd_15m_base.py` covcurve (frozen-2012-21 book; per-year)
| cov | COMB wr 2024/25/26 | UP wr 2024/25/26 | DOWN wr 2024/25/26 |
|---|---|---|---|
| 0.10 | .548 / .545 / .521 | .542 / .545 / .519 | .554 / .545 / .521 |
| 0.05 | .568 / .571 / .528 | .560 / .576 / .527 | .572 / .568 / .529 |
| 0.03 | .578 / .572 / .524 | .579 / .581 / .500 | .577 / .568 / .537 |
| 0.02 | .586 / .583 / .559 | **.603 / .595 / .493** | .580 / .578 / **.589** |
| 0.01 | .600 / .569 / .562 | .552 / .584 / .516 | .610 / .567 / .578 |

**Read:** monotone-in-confidence, both sides ~.57–.60 @cov2–3% in 2024/2025. **2026 (binding) is DOWN-only** — DOWN holds (.589@cov2 / .578@cov1) while UP collapses (.493@cov2 / .516@cov1). The frozen-2021 book decays forward (refit-dependent, same as EURUSD/USDJPY 15m); DOWN survives the decay, UP does not. → certify via per-fold-REFIT CPCV (recovers per-era floor); DOWN is the more deployable side; UP is regime-dependent and must be confirmed forward (trap #9).

### Meta-label 'avoid-losers' gate (fast pre-check) — `audusd_15m_metagate.py` → NULL
Meta-correctness AUC (predict P(primary call correct) from orthogonal axes audnzd/risk/agree/session): **axes-only .5197, axes+conf .5294 → viable=False** (falsifier ≤.53). The orthogonal axes carry no incremental WHEN-CORRECT info — same as EURUSD 15m meta (.502 NULL). Confirms orthochan (axes carry no conditional direction signal). Meta-gate KILLED. `audusd_15m_metagate_result.json`.

### TB first-touch TRAIN-label (confirm-or-kill) — `audusd_15m_cpcv_tbfirsttouch.py`
VAL k-screen picked k=2.0 (VAL fixed-15m AUC **.5389** vs base .5378 = +.0011, marginal; USDJPY was +.005). Single-seed NY CPCV certifies all sides (p10>BE, frac 1.0) but the HONEST **matched-cov** comparison (the script's `IMPROVES` flag compares cross-cov vs the cov5 incumbent — NOT matched, the USDJPY overclaim trap): TB vs single-seed base @cov2 = UP .5826 vs .5833 (−.001 flat), DOWN .5981 vs .5897 (+.008), COMB .5906 vs .5915 (−.001 flat). **vs the seed-ens K=3 DELIVERABLE: TB single-seed is BELOW on UP (.5826≪.596) and COMB (.5906<.591); DOWN ≈ (.5981 vs .5962).** → TB-label certifies-at-level but is **NOT a robust improvement** over the deliverable. **Decisive TB+seed-ens K=3 matched test** (`audusd_15m_cpcv_tbfirsttouch_ny_seedens3_result.json`) vs the matched seed-ens K=3 base, p10: cov5 UP +.004/DOWN −.003/COMB +.001; cov3 UP +.008/DOWN +.002/COMB −.003; cov2 **UP −.016**/DOWN +.007/COMB +.004 — **sign-inconsistent across cov and UP REGRESSES at the cov2 operating point** (the weak side). Path-means rise uniformly ~+.007 but the p10 cert floor is mixed; per the USDJPY lesson a mean nudge with inconsistent/regressing p10 is variance redistribution, not a robust edge. **KILLED as an improvement; symmetric seed-ens K=3 deliverable stands.** `audusd_15m_cpcv_tbfirsttouch_ny_multicov_result.json` (single-seed) + `..._seedens3_result.json`.

### All-session refit-CPCV (per-fold-refit, 15 purged paths) — `audusd_15m_cpcv_session.py all`
| cov | UP p10 (frac_clear) | DOWN p10 (frac) | COMB p10 (frac) | CERT |
|---|---|---|---|---|
| 0.05 | .5371 (.87) | .5354 (.73) | .5374 (.73) | ✗ |
| 0.03 | .5342 (.80) | .5405 (.87) | .5385 (.87) | ✗ |
| 0.02 | .5381 (.87) | .5345 (.73) | .5371 (.73) | ✗ |
| 0.01 | .5345 (.80) | **.544 (.93)** | **.542 (.87)** | **DOWN ✓ / COMB ✓** |

AUC mean .5213 (min .5142, max .5252), up-rate tripwire clean. **Read:** a real all-session edge but thin — **DOWN certifies at cov1%** (p10 .544, 14/15 paths; med_n 895/path), COMBINED at cov1% (p10 .542); **UP does NOT certify at any cov** (p10 saturates ~.534–.538, ~0.5–1pp sub-BE). DOWN>UP robustness (consistent with binding-2026 DOWN-only). This is the honest per-era refit floor (deployable w/ retrain). `audusd_15m_cpcv_session_all_multicov_result.json`. **NEXT levers to lift p10 to a usable coverage + rescue UP:** session concentration (A9), cross-pair pooling (A6), seed-ensemble (the USDJPY lever).

### Session segmentation (A9) — `audusd_15m_cpcv_session.py {ny,ldn,asia}` (symmetric refit-CPCV)
**RESULT: NY is the carrier — and certifies BOTH sides at every coverage, 15/15 paths.** The Asia hypothesis (RBA/China own-pair info) was REFUTED for 15m DIRECTION (Asia/LDN sub-BE) → AUDUSD direction-sign rides the US-session USD flow (like EURUSD/USDJPY); the commodity/China info gates magnitude not 15m sign (sign-invariance holds).

| session | UP p10 @cov5 (frac) | DOWN p10 @cov5 (frac) | COMB p10 | verdict |
|---|---|---|---|---|
| **NY** | **.5713 (1.0)** | **.5773 (1.0)** | **.5809 (1.0)** | ✅ BOTH CERTIFY |
| asia | .5243 (.40) | .5259 (.53) | .5272 | ✗ sub-BE |
| ldn | .5163 (.27) | .5162 (.67) | .5196 | ✗ sub-BE |

**NY-session refit-CPCV (the deliverable), all covs, frac_clear=1.0 (all 15 paths) at every cov:**
| cov | UP p10 (mean, min) | DOWN p10 (mean, min) | COMB p10 (mean) | med_n/path |
|---|---|---|---|---|
| 0.05 | .5713 (.587, .569) | .5773 (.596, .565) | .5809 (.590) | UP 1552 / DN 1286 |
| 0.03 | .5719 (.595, .563) | .5972 (.610, .580) | .5866 (.601) | UP 909 / DN 735 |
| 0.02 | .5833 (.605, .577) | .5897 (.617, .578) | .5915 (.610) | UP 583 / DN 494 |
| 0.01 | .5863 (.612, .579) | .5959 (.625, .566) | .6039 (.618) | UP 301 / DN 242 |

AUC mean NY .5363 (.530–.542), up-rate tripwire clean. **Both sides CERTIFIED (refit-CPCV p10 ≥ BE, 15/15 paths, every cov)** — but **REFIT-DEPENDENT** (see frozen-forward below). NY rescues the all-session UP near-miss (sub-BE → p10 .571–.586). `audusd_15m_cpcv_session_ny_multicov_result.json`.

**★ IMPROVE — seed-ensemble K=3 (Tier-I I2) on NY (the deliverable): genuine modest lift over single-seed.** Matched-fold NY refit-CPCV, K=3 seed-average:
| cov | UP p10 (Δ single) | DOWN p10 (Δ) | COMB p10 (Δ) | COMB mean |
|---|---|---|---|---|
| 0.05 | .576 (+.005) | **.587 (+.010)** | .588 (+.007) | .597 |
| 0.03 | .5755 (+.004) | .590 (−.007) | .596 (+.009) | .609 |
| 0.02 | **.596 (+.013)** | .596 (+.007) | .591 (−.001) | .616 |
| 0.01 | .586 (0) | .590 (−.006) | .589 (−) | .620 |

frac_clear=1.0 every cell. **Both p10 AND mean lift at the operating covs** (cov2–5%) — variance reduction genuinely helps the tail (the mean rising distinguishes this from the USDJPY TB false-positive where only the p10 order-statistic moved on correlated paths). Deliverable = **seed-ens K=3 NY**; cov2% UP p10 .596 / DOWN .596 (mean ~.616), cov5% UP .576 / DOWN .587 (mean .597, med_n 1429/1172). `audusd_15m_cpcv_session_ny_seedens3_result.json`. Still REFIT-DEPENDENT (the frozen-forward decay below is a property of the edge, not the seed count).

**Seed-depth SATURATION — K=8 (`audusd_15m_cpcv_session_ny_seedens8_result.json`):** K=8 vs K=3 p10 is mixed within ±.005–.016, sign-inconsistent across cov/side (cov2 UP .591 vs .596 = −.005; DOWN .612 vs .596 = +.016 but mean only +.007 = worst-path order-stat; COMB .593 vs .591 = +.002). Means marginally up ~+.003–.007 but no robust two-sided lift. **Seed lever saturated past K=3** (DL-review M=8 prescription tested → diminishing returns, as USDJPY/EURUSD). **K=3 retained as the deliverable** (identical certified performance, fewer models). Improve loop CLOSED.

### ★ Adversarial verification — NY FROZEN-PAST forward holdout (trap#9) — `audusd_15m_ny_frozen.py`
Train ONCE on 2012-21 NY, FREEZE, test per-year NY (deployment-faithful, NO retrain):
| year | cov5 COMB (CI-lo) | cov5 UP | cov5 DOWN | cov2 UP | cov2 DOWN |
|---|---|---|---|---|---|
| 2024 | .606 (.578) | .613 | .602 | .651 | .619 |
| 2025 | .569 (.542) | .580 | .560 | .614 | .571 |
| **2026** | **.532 (.489)** | **.530** | **.533** | **.442** | .529 |

**VERDICT: the refit-CPCV cert is REFIT-DEPENDENT, NOT a frozen-deployable edge.** The frozen-2012-21 vintage DECAYS monotonically (.606→.569→.532 COMB cov5), going **sub-BE on BOTH sides by 2026**. This is NOT trap#9 era-local memorization (that would be flat ~.50 every forward year) — the edge genuinely transfers to 2024/2025 (.60/.57) then decays, the signature of **USD-factor non-stationarity** (identical to EURUSD m15xp/m30xp + USDJPY m15ny, all flagged refit-dependent). **Honest deployment claim:** deploy NY-session-only **with periodic retraining**; the durable figure is the refit-CPCV per-era floor (UP .571–.583, DOWN .577–.590), NOT the frozen book (a stale 2021 vintage loses money in 2026). The certified floor is real with retrain; a frozen deployment is not. `audusd_15m_ny_frozen_result.json`.

### Cross-horizon stack viability (R2 lever) — `audusd_15m_xhorizon.py` → **NOT collinear; viable lead**
corr(p30_NY, p15_NY) = **0.8117** (n270k) — BELOW the .90 collinearity kill-threshold (USDJPY 30m parent was .957 → there subsumed; **AUDUSD differs**). 30m NY parent VAL-AUC .5408 (> 15m child .5388), parent cov2 NY VAL-acc .5954 (a real edge). ~19% decorrelated → an agreement-gated cross-horizon stack could lift conditional accuracy. PURSUED (`audusd_15m_xhstack.py`). `audusd_15m_xhorizon_result.json`.

**Cross-horizon STACK refit-CPCV (single-seed) — `audusd_15m_xhstack.py`:** mean fold corr(p15,p30)=**.7223** (genuinely decorrelated). vs the matched single-seed BASE arm @cov2: **blend** COMB .5953 (+.008), DOWN .5966 (+.023), UP .5847 (−.001); agree ≈ base. **But path-MEANS are flat** (blend DOWN mean .6173 vs base .615 = +.002) → the p10 lift is variance reduction, NOT new information (the 30m parent is decorrelated but equally weak, AUC .5408≈.5388 — averaging cuts variance like seed-ens, adds no directional info). blend/agree do NOT beat the seed-ens K=3 incumbent on BOTH sides (UP .5847<.596). `IMPROVES_over_seedens=False`. **Decisive seed-ens K=3 blend test (corr .766): blend/agree are WORSE than the matched seed-ens base** (cov2 blend UP −.01/DOWN −.014/COMB −.014; agree ≈ base) — `IMPROVES_over_seedens=False`. **Confirms the mechanism: the 30m parent is decorrelated but equally weak (AUC≈child), so blending into an already-variance-reduced seed-ens child only dilutes — the horizon route is REDUNDANT with seed-ens.** Cross-horizon stack KILLED; deliverable stands. `audusd_15m_xhstack_ny_result.json`.

### Cross-pair pooling (A6) — the EURUSD keystone, retargeted to AUDUSD → **POOLING DOES NOT ADD (own-pair-specific, like USDJPY)**
`audusd_15m_xpair.py xpbase` (cross-pair USD-residual + lead-lag + **AUDNZD relative value** + **commodity/safe-haven risk factor**, AUDUSD target, all-session screen). VAL AUC **.5244** vs base .5232 (+.0012, negligible). **top-20 features are ALL own-pair base feats** (hour_sin/cos, autocorr, rv, dist_hi/lo, bb_width) — the cross-pair catchup/usdbask/lead-lag, AUDNZD-RV (audnzd_r/dev), and risk-bloc (risk/audrisk_resid) features **don't crack the top 20**. Per-year cov2%: 2024 COMB .621/UP .658/DOWN .607; 2025 .604/.599/.606; 2026 **.5515/UP .4935 (collapses, same as base)/DOWN .576**. **VERDICT: IMPROVES_base=False** — the GBM ignores the USD-factor pool + AUDNZD difference + risk factor; the own-pair base 239 already captures the 15m direction signal. **AUDUSD is the USDJPY case (own-pair-specific), NOT the EURUSD case (poolable).** `audusd_15m_xpair_xpbase_result.json`.

**Orthogonal-channel famonly isolation (NY) — `audusd_15m_orthochan.py` (confirms A6, run-don't-argue):** base+block VAL-AUC vs base .5378: **AUDNZD residual-diff** Δ+.0002, **commodity/safe-haven risk-bloc** Δ−.0009, **signed-semivariance RS+/−** Δ−.0012 — **ALL ADDS=False**. None carries net-new VAL direction signal over the base 239 (the binding-2026 cov2 tail nudges up with audnzd .567 vs base .543, but VAL-AUC — the selection metric — is flat, so not selectable; chasing one small-n year's tail violates discipline). Backlog #3/#4/#11 SUBSUMED by the own-pair base. `audusd_15m_orthochan_result.json`.

---

## FINAL CONCLUSION (AUDUSD 15m direction) — 2026-06-09
**BOTH sides CERTIFIED + frozen + deployable-spec'd + tick-validated.** Deliverable book `AUDUSD.m15ny_seedens.v1` (NY-session own-pair LGBM seed-ensemble K=3): UP p10 **.596** / DOWN p10 **.596** @cov2% (UP .576/DOWN .587 @cov5%), mean ~.62, 15/15 refit-CPCV paths clear at every cov; tick-settlement PRESERVED (−.0035). REFIT-DEPENDENT (deploy NY-only w/ periodic retrain, size on the refit floor).

**Model of the edge (final):** AUDUSD 15m direction-sign is **NY-session-concentrated + own-pair-specific** — the AUD Asia/commodity/China information gates move MAGNITUDE not 15m sign (sign-invariance), the directional sign rides US-session USD flow. The Asia hypothesis was tested symmetrically and **REFUTED** (Asia/LDN sub-BE). It is the **USDJPY case, not the EURUSD case**: cross-pair pooling DILUTES (NULL).

**The >65% target is an INFORMATION BOUND on existing on-disk data, not reached.** NY moved-AUC ceiling ~.536 → certified floor ~.57–.60, mean ~.62 at tight coverage; single forward years touch ~.65 (2024 cov2 UP .65) but that is not a floor. The bound is robust — confirmed across the full improve cross-product:
- **LIFTS (frozen):** seed-ensemble K=3 (the one robust own-pair lever; p10 + mean both up).
- **NULL / subsumed (run, not argued):** cross-pair pooling (own-pair-specific); AUDNZD residual-difference, commodity/safe-haven risk-bloc, signed-semivariance RS± (famonly ΔAUC≈0); meta-label gate (meta-AUC .52); triple-barrier label (UP regresses @cov2, not robust); cross-horizon stack (parent decorrelated .72 but equally weak → variance-redundant with seed-ens, non-additive); state/complexity/magnitude-bridge/symmetric-ACI/calibration (theorem-killed, subsumed from USDJPY exhaust). Cross-horizon stack confirmed KILLED through seed-ens K=3 (decorrelated-but-equally-weak parent = redundant with seed-ens).
- **The only frontier past the bound = EXTERNAL signed data** (intraday AU-US 2y rate differential, risk-on/off VIX/ES, commodity index iron-ore/CRB) — off-disk, acquisition TODO. On-disk discover loop (R1 corpus + R2 repo-docs) converged here.

**Improve + discover loops DRY** on existing on-disk data. The deliverable is the deployable answer; >65% requires external data acquisition (the documented redirect, not a wall).

## UP/DOWN LEADERBOARD (current best per side, certified-or-best-available)
| Side | Best certified (refit-CPCV p10) | Best available (mean) | Book | Status |
|---|---|---|---|---|
| **15m UP** | **.576 @cov5 / .596 @cov2** (NY seed-ens K=3, 15/15) | .593 / .610 mean (cov5/2) | **`AUDUSD.m15ny_seedens.v1`** ✅ FROZEN (content_id 9b0e0ed3) | ✅ CERTIFIED (NY seed-ens refit-CPCV); REFIT-DEPENDENT |
| **15m DOWN** | **.587 @cov5 / .596 @cov2** (NY seed-ens K=3, 15/15) | .604 / .623 mean (cov5/2) | **`AUDUSD.m15ny_seedens.v1`** ✅ FROZEN (content_id 9b0e0ed3) | ✅ CERTIFIED (NY seed-ens refit-CPCV); REFIT-DEPENDENT |

**★ True-tick-settlement validation (integrity gate) — `audusd_15m_ticksettle.py`:** re-settled the frozen book's confident NY trades on raw AUDUSD ticks (entry = first tick ≥ bar-close+1s, exit = last tick ≤ entry+900s, mid-to-mid, ties LOSE). Same-subset bar-WR → tick-WR: 2024 .6051→.5882, 2025 .5837→.5872, 2026 .5068→.5096; **mean(tick−bar) = −0.0035 → PRESERVED** (matches USDJPY −.0036). The certified ~.59 NY win-rate is real on traded prices, NOT a bar-close/bar-shift artifact; bar-close proxy faithful at 15m. `audusd_15m_ticksettle_result.json`.

**Deployment spec (both sides, book `AUDUSD.m15ny_seedens.v1`):** trade AUDUSD 15m Rise/Fall when `ts ∈ NY session` (America/New_York 08:00–17:00, DST-correct) AND `|p̄−0.5| ≥ conf_thr` where `p̄` = mean of K=3 seed probabilities; cov3% gate (thr frozen on VAL NY worst-half). Two-sided via confidence selection (bet UP if p̄>0.5 else DOWN). **REFIT-DEPENDENT: retrain periodically (frozen-2012-21 vintage decays to sub-BE by 2026); size on the refit per-era floor (~.57–.60), 1/8-Kelly, NOT the stale frozen book.** Breakeven 0.541. Deriv-FX-deployable (15m = forex Rise/Fall minimum expiry).

_Provenance: every number traces to a `*_result.json` (Tier-1). Updated as rows complete._

## AUDUSD — DERIV-FEED PROGRAM (Deriv-only data; opened 2026-11-15)

> **SCOPE:** this key on the Deriv feed only — Deriv 1-second mid ticks (`/mnt/data/deriv`), no Dukascopy data. Labels settle on Deriv ticks (entry first tick >= decision + 1 s, exit last tick <= entry target + H, ties lose); DEV window 2025-08-11..2026-07-31 for training, selection and CPCV; HOLDOUT 2026-08-01..2026-11-14 sealed (opened once per finalized candidate). Protocol and tools: `docs/DERIV_SOTA_PROGRAM.md`. Tables below are rendered from the result JSONs by `scripts/render_ledger.py` (do not edit by hand).

<!-- DERIV-FEED-NOTES:BEGIN -->
- **Incumbent benchmark on TRADEABLE decisions, books as trained (R1bt, 13:00-16:35 + 18:00-22:00 ET):** UP best `AUDUSD.m15ny_seedens.v1` 0.4737 [0.3860, 0.5614] n=114 (NOT SUPPORTED); DOWN best `AUDUSD.m15ny_seedens.v1` 0.5137 [0.4510, 0.5765] n=255 (NOT SUPPORTED).
- **Books as deployed today (R1t, true clock, NY 08:00-16:35, tradeable):** UP best `AUDUSD.m15ny_seedens.v1` 0.5106 [0.4468, 0.5745] n=235 (NOT SUPPORTED); DOWN best `AUDUSD.m15ny_seedens.v1` 0.5129 [0.4741, 0.5503] n=696 (NOT SUPPORTED).
- **DEV-CPCV certified sides, GOAL rule (>= 12 of 15 paths counted and clear, p10 >= .541; tradeable runs):** none yet. DEV certification is necessary, not sufficient: the sealed holdout decides (docs/DERIV_SOTA_PROGRAM.md).
- **DEV leaderboard 15m UP** (closest to the goal rule at each run's operating coverage): `e29logit` 8/15 clear, p10 0.492, excess +0.048; `e7fitall` 7/15 clear, p10 0.498, excess +0.057; `e58nysreflect` 7/11 clear, p10 0.496, excess +0.067.
- **DEV leaderboard 15m DOWN** (closest to the goal rule at each run's operating coverage): `e10v02` 11/15 clear, p10 0.447, excess +0.065; `e55rl` 9/15 clear, p10 0.464, excess +0.060; `c9compaci` 8/15 clear, p10 0.449, excess +0.032.
- **DEV leaderboard 30m UP** (closest to the goal rule at each run's operating coverage): `e5h30` 3/8 clear, p10 0.410, excess -0.004.
- **DEV leaderboard 30m DOWN** (closest to the goal rule at each run's operating coverage): `e5h30` 2/8 clear, p10 0.425, excess +0.006.
- **Deployment status (Deriv feed):** DO NOT DEPLOY - no candidate has passed both the DEV-CPCV goal rule (or the drift amendment) and a sealed-holdout look, and deployment needs both.
- Superseded: `e1` and `r2` ran before the venue tradeability filter (they include untradeable blackout decisions); their tradeable re-runs are `e1t` and `r2t`.
<!-- DERIV-FEED-NOTES:END -->

> **AUDUSD 15m — round levels vs placebo (2026-11-15, hand-written; numbers from the cited JSONs).** `e55rl` leads the
> DOWN leaderboard above (9/15 clear, p10 0.464; KILLED), but its gain is not a round-level effect: the median paired
> path moved-AUC gain over `e1t` is +0.0088 for the real grid (`deriv_e55rl_AUDUSD_15m_result.json`) and +0.0081 for
> the half-grid shifted placebo (`deriv_e56rs_AUDUSD_15m_result.json`), so the lift comes from the block's generic
> price-grid structure, not from round numbers.

> **AUDUSD 15m DOWN — Tier-I levers on the gate base (2026-11-15, hand-written; numbers from the cited JSONs).** Base
> `e33comp33` (LightGBM + compression gate 0.33, operating coverage 0.10): 7 of 15 paths clear, p10 0.4707, VALID.
> Pre-registered reading (lab PROGRESS.md, by 13:11 UTC): a lever IMPROVES iff paths clear >= the base's and p10 > the
> base's at the operating coverage on a VALID run. `c7comp3b` (3 seeds) IMPROVES by that reading (7 of 15, p10
> 0.4808) but stays KILLED by the goal rule (7 < 12 paths, p10 < 0.541), so it is not a finalist. No improvement:
> `c9compaci` (8, 0.4491), `c8compabs` (4, 0.4770), `c6nyscompb` (3, 0.4625), `c10compmeta` (coverage doubled to 0.20
> by design; 4, 0.4738). `c11compmag` leaves no path with 25 trades (at most 7 on a path). `c6nyscomp` and `c7comp3`
> ran at coverage 0.03 by a queueing error and are recorded as coverage variants, not comparisons.

<!-- DERIV-FEED-TABLES:BEGIN -->
**Frozen books on the Deriv feed — true clock, ny (America/New_York 08:00-17:00, DST-correct, sessions.session_mask), ALL decisions incl. the untradeable blackout** (`deriv_r1_frozen_books_dev_result.json`; strategy gate, tick-settled, ties lose; SUPPORTED iff n >= 50 and CI95-lo >= .541):

| Book | AUC (moved) | gate rate | UP n · WR [CI95] | DOWN n · WR [CI95] | COMB n · WR | verdict UP / DOWN |
|---|---|---|---|---|---|---|
| `AUDUSD.m15ny_seedens.v1` | .5201 | .0503 | 238 · .5168 [.4538, .5798] | 696 · .5129 [.4741, .5503] | 931 · .5134 | NOT SUPPORTED / NOT SUPPORTED |

**Frozen books on the Deriv feed — training clock (true UTC - 5 h), nylate (America/New_York 13:00-22:00, DST-correct), ALL decisions incl. the untradeable blackout** (`deriv_r1b_frozen_books_trainclock_nylate_dev_result.json`; strategy gate, tick-settled, ties lose; SUPPORTED iff n >= 50 and CI95-lo >= .541):

| Book | AUC (moved) | gate rate | UP n · WR [CI95] | DOWN n · WR [CI95] | COMB n · WR | verdict UP / DOWN |
|---|---|---|---|---|---|---|
| `AUDUSD.m15ny_seedens.v1` | .5409 | .0338 | 243 · .6584 [.6008, .7160] | 468 · .6175 [.5726, .6603] | 683 · .6223 | SUPPORTED / SUPPORTED |

**Frozen books on the Deriv feed — training clock (true UTC - 5 h), nylate (America/New_York 13:00-22:00, DST-correct), tradeable decisions only (Deriv purchase blackout excluded)** (`deriv_r1bt_frozen_books_trainclock_nylate_dev_tradeable_result.json`; strategy gate, tick-settled, ties lose; SUPPORTED iff n >= 50 and CI95-lo >= .541):

| Book | AUC (moved) | gate rate | UP n · WR [CI95] | DOWN n · WR [CI95] | COMB n · WR | verdict UP / DOWN |
|---|---|---|---|---|---|---|
| `AUDUSD.m15ny_seedens.v1` | .5169 | .0210 | 114 · .4737 [.3860, .5614] | 255 · .5137 [.4510, .5765] | 368 · .5000 | NOT SUPPORTED / NOT SUPPORTED |

**Frozen books on the Deriv feed — true clock, ny (America/New_York 08:00-17:00, DST-correct, sessions.session_mask), tradeable decisions only (Deriv purchase blackout excluded)** (`deriv_r1t_frozen_books_dev_tradeable_result.json`; strategy gate, tick-settled, ties lose; SUPPORTED iff n >= 50 and CI95-lo >= .541):

| Book | AUC (moved) | gate rate | UP n · WR [CI95] | DOWN n · WR [CI95] | COMB n · WR | verdict UP / DOWN |
|---|---|---|---|---|---|---|
| `AUDUSD.m15ny_seedens.v1` | .5164 | .0520 | 235 · .5106 [.4468, .5745] | 696 · .5129 [.4741, .5503] | 928 · .5119 | NOT SUPPORTED / NOT SUPPORTED |

**Per-fold-refit CPCV on the Deriv feed** (DEV 2025-08-11..2026-07-31, NY DST session, 15 paths; p10/frac = path win-rate 10th percentile / share of paths >= .541; fwd = forward-walk paths clearing .541 and their mean):

| Run | Model / features | H | cov | AUC mean (min) | UP p10 · frac · mean | UP fwd | DOWN p10 · frac · mean | DOWN fwd | always UP / DOWN | repo-rule verdict UP / DOWN | GOAL rule UP / DOWN (paths counted) | frac @.565 UP / DOWN | result JSON |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| c10compmeta | c10compmeta: all (66), nseed 1 | 15m | 0.2 | .5042 (.4949) | .4527 · .200 · .5010 | 0/3 · .4309 | .4738 · .267 · .5192 | 0/2 · .4454 | .4899 / .4926 | KILLED / KILLED | no (15/15, 3 clear) / no (15/15, 4 clear) | 0.13 / 0.13 | `deriv_c10compmeta_AUDUSD_15m_result.json` |
| c11compmag | c11compmag: all (66), nseed 1 | 15m | 0.1 | .5042 (.4949) | nan · .000 · — | 0/0 · — | nan · .000 · — | 0/0 · — | .4899 / .4926 | KILLED / KILLED | no (0/15, 0 clear) / no (0/15, 0 clear) | — / — | `deriv_c11compmag_AUDUSD_15m_result.json` |
| c6nyscomp | c6nyscomp: all (66), nseed 1 | 15m | 0.03 | .5110 (.5011) | .4552 · .429 · .5227 | 0/0 · — | .4842 · .286 · .5290 | 1/2 · .5434 | .4899 / .4926 | KILLED / KILLED | no (7/15, 3 clear) / no (14/15, 4 clear) | 0.29 / 0.21 | `deriv_c6nyscomp_AUDUSD_15m_result.json` |
| c6nyscompb | c6nyscompb: all (66), nseed 1 | 15m | 0.1 | .5110 (.5011) | .4686 · .250 · .5124 | 1/2 · .5417 | .4625 · .200 · .5129 | 2/4 · .5312 | .4899 / .4926 | KILLED / KILLED | no (12/15, 3 clear) / no (15/15, 3 clear) | 0.00 / 0.13 | `deriv_c6nyscompb_AUDUSD_15m_result.json` |
| c7comp3 | c7comp3: all (66), nseed 3 | 15m | 0.03 | .5040 (.4946) | .4510 · .000 · .4554 | 0/0 · — | .4603 · .250 · .5101 | 0/0 · — | .4899 / .4926 | KILLED / KILLED | no (3/15, 0 clear) / no (4/15, 1 clear) | 0.00 / 0.25 | `deriv_c7comp3_AUDUSD_15m_result.json` |
| c7comp3b | c7comp3b: all (66), nseed 3 | 15m | 0.1 | .5040 (.4946) | .4417 · .333 · .5120 | 0/3 · .4446 | .4808 · .467 · .5479 | 0/2 · .4138 | .4899 / .4926 | KILLED / KILLED | no (12/15, 4 clear) / no (15/15, 7 clear) | 0.25 / 0.47 | `deriv_c7comp3b_AUDUSD_15m_result.json` |
| c8compabs | c8compabs: all (66), nseed 1 | 15m | 0.1 | .5031 (.4891) | .4515 · .214 · .5052 | 0/3 · .4634 | .4770 · .267 · .5159 | 2/3 · .5670 | .4899 / .4926 | KILLED / KILLED | no (14/15, 3 clear) / no (15/15, 4 clear) | 0.14 / 0.07 | `deriv_c8compabs_AUDUSD_15m_result.json` |
| c9compaci | c9compaci: all (66), nseed 1 | 15m | 0.1 | .5042 (.4949) | .4418 · .231 · .5051 | 0/3 · .4075 | .4491 · .533 · .5244 | 0/2 · .4241 | .4899 / .4926 | KILLED / KILLED | no (13/15, 3 clear) / no (15/15, 8 clear) | 0.15 / 0.27 | `deriv_c9compaci_AUDUSD_15m_result.json` |
| e10v02 | e10v02: all (82), nseed 1 | 15m | 0.03 | .5136 (.5033) | .4156 · .250 · .5024 | 0/0 · — | .4468 · .733 · .5579 | 0/1 · .4815 | .4899 / .4926 | KILLED / KILLED | no (12/15, 3 clear) / no (15/15, 11 clear) | 0.17 / 0.47 | `deriv_e10v02_AUDUSD_15m_result.json` |
| e13mag50 | e13mag50: all (66), nseed 1 | 15m | 0.03 | .5042 (.4949) | .4798 · .500 · .5254 | 0/0 · — | .3893 · .500 · .4898 | 0/1 · .4375 | .4899 / .4926 | KILLED / KILLED | no (4/15, 2 clear) / no (4/15, 2 clear) | 0.25 / 0.25 | `deriv_e13mag50_AUDUSD_15m_result.json` |
| e14nylate | e14nylate: all (66), nseed 1 | 15m | 0.03 | .5100 (.4982) | .3945 · .308 · .4936 | 1/2 · .5037 | .4101 · .357 · .4940 | 1/2 · .4643 | .4850 / .4933 | KILLED / KILLED | no (13/15, 4 clear) / no (14/15, 5 clear) | 0.08 / 0.14 | `deriv_e14nylate_AUDUSD_15m_result.json` |
| e15nypm | e15nypm: all (66), nseed 1 | 15m | 0.03 | .5101 (.4906) | .3590 · .222 · .4468 | 0/1 · .3077 | .4000 · .286 · .5026 | 0/1 · .5278 | .4729 / .5013 | KILLED / KILLED | no (9/15, 2 clear) / no (14/15, 4 clear) | 0.11 / 0.14 | `deriv_e15nypm_AUDUSD_15m_result.json` |
| e16nyeve | e16nyeve: all (66), nseed 1 | 15m | 0.03 | .5106 (.4926) | .4948 · .375 · .5348 | 1/1 · .6061 | .4365 · .429 · .5204 | 0/0 · — | .4977 / .4848 | KILLED / KILLED | no (8/15, 3 clear) / no (7/15, 3 clear) | 0.25 / 0.43 | `deriv_e16nyeve_AUDUSD_15m_result.json` |
| e17nyam | e17nyam: all (66), nseed 1 | 15m | 0.03 | .5084 (.4952) | .4516 · .545 · .5274 | 0/0 · — | .4039 · .500 · .5293 | 0/1 · .5172 | .5030 / .4860 | KILLED / KILLED | no (11/15, 6 clear) / no (8/15, 4 clear) | 0.55 / 0.25 | `deriv_e17nyam_AUDUSD_15m_result.json` |
| e18lowcap | e18lowcap: all (66), nseed 1 | 15m | 0.03 | .5020 (.4882) | .4579 · .200 · .5059 | 0/4 · .5098 | .4400 · .200 · .5047 | 1/4 · .4848 | .4899 / .4926 | KILLED / KILLED | no (15/15, 3 clear) / no (15/15, 3 clear) | 0.07 / 0.00 | `deriv_e18lowcap_AUDUSD_15m_result.json` |
| e19tf | e19tf: all+fam:tf (74), nseed 1 | 15m | 0.03 | .5066 (.4975) | .4303 · .429 · .5176 | 1/4 · .4937 | .4081 · .133 · .4834 | 0/1 · .3871 | .4899 / .4926 | KILLED / KILLED | no (14/15, 6 clear) / no (15/15, 2 clear) | 0.29 / 0.13 | `deriv_e19tf_AUDUSD_15m_result.json` |
| e1 | e1: all (66), nseed 1 | 15m | 0.03 | .5099 (.5014) | .4951 · .533 · .5588 | 0/2 · .4579 | .4907 · .600 · .5646 | 0/3 · .4448 | .4881 / .4945 | KILLED / KILLED | no (15/15, 8 clear) / no (15/15, 9 clear) | 0.53 / 0.47 | `deriv_e1_AUDUSD_15m_result.json` |
| e1t | e1t: all (66), nseed 1 | 15m | 0.03 | .5042 (.4949) | .4737 · .214 · .5147 | 0/4 · .4651 | .4643 · .400 · .5216 | 0/3 · .4412 | .4899 / .4926 | KILLED / KILLED | no (14/15, 3 clear) / no (15/15, 6 clear) | 0.14 / 0.13 | `deriv_e1t_AUDUSD_15m_result.json` |
| e20sg | e20sg: all+fam:sg (71), nseed 1 | 15m | 0.03 | .5038 (.4915) | .3986 · .067 · .4669 | 0/3 · .4286 | .4024 · .267 · .4912 | 1/4 · .4645 | .4899 / .4926 | KILLED / KILLED | no (15/15, 1 clear) / no (15/15, 4 clear) | 0.07 / 0.20 | `deriv_e20sg_AUDUSD_15m_result.json` |
| e29logit | e29logit: all (66), nseed 1 | 15m | 0.03 | .5087 (.5005) | .4922 · .533 · .5379 | 1/4 · .5303 | .4470 · .067 · .4898 | 1/3 · .5052 | .4899 / .4926 | KILLED / KILLED | no (15/15, 8 clear) / no (15/15, 1 clear) | 0.27 / 0.00 | `deriv_e29logit_AUDUSD_15m_result.json` |
| e2noclock | e2noclock: noclock (63), nseed 1 | 15m | 0.03 | .5015 (.4947) | .4181 · .267 · .5079 | 2/3 · .5548 | .4192 · .267 · .4930 | 0/3 · .4329 | .4899 / .4926 | KILLED / KILLED | no (15/15, 4 clear) / no (15/15, 4 clear) | 0.20 / 0.07 | `deriv_e2noclock_AUDUSD_15m_result.json` |
| e30nystroem | e30nystroem: all (66), nseed 1 | 15m | 0.03 | .5110 (.5011) | .4636 · .000 · .5102 | 2/2 · .5946 | .4770 · .067 · .5090 | 0/4 · .4879 | .4899 / .4926 | KILLED / KILLED | no (13/15, 0 clear) / no (15/15, 1 clear) | 0.00 / 0.07 | `deriv_e30nystroem_AUDUSD_15m_result.json` |
| e32spec | e32spec: all (66), nseed 1 | 15m | 0.03 | .5041 (.4960) | .4327 · .000 · .4661 | 1/4 · .5224 | .4440 · .067 · .4990 | 1/4 · .4739 | .4899 / .4926 | KILLED / KILLED | no (15/15, 0 clear) / no (15/15, 1 clear) | 0.00 / 0.07 | `deriv_e32spec_AUDUSD_15m_result.json` |
| e33comp33 | e33comp33: all (66), nseed 1 | 15m | 0.1 | .5042 (.4949) | .4405 · .154 · .4969 | 0/3 · .4233 | .4707 · .467 · .5360 | 0/2 · .4327 | .4899 / .4926 | KILLED / KILLED | no (13/15, 2 clear) / no (15/15, 7 clear) | 0.15 / 0.27 | `deriv_e33comp33_AUDUSD_15m_result.json` |
| e34online | e34online: all (66), nseed 1 | 15m | 0.03 | .5050 (.4955) | .4676 · .000 · .4916 | 0/0 · — | .4922 · .000 · .5165 | 0/0 · — | .4899 / .4926 | KILLED / KILLED | no (15/15, 0 clear) / no (15/15, 0 clear) | 0.00 / 0.00 | `deriv_e34online_AUDUSD_15m_result.json` |
| e35absret | e35absret: all (66), nseed 1 | 15m | 0.03 | .5031 (.4891) | .3937 · .222 · .4848 | 1/2 · .5507 | .4208 · .300 · .4960 | 0/1 · .5316 | .4899 / .4926 | KILLED / KILLED | no (9/15, 2 clear) / no (10/15, 3 clear) | 0.22 / 0.10 | `deriv_e35absret_AUDUSD_15m_result.json` |
| e36aci | e36aci: all (66), nseed 1 | 15m | 0.03 | .5042 (.4949) | .4656 · .286 · .5144 | 0/4 · .4628 | .4611 · .267 · .5187 | 0/2 · .4924 | .4899 / .4926 | KILLED / KILLED | no (14/15, 4 clear) / no (15/15, 4 clear) | 0.07 / 0.20 | `deriv_e36aci_AUDUSD_15m_result.json` |
| e37meta | e37meta: all (66), nseed 1 | 15m | 0.06 | .5042 (.4949) | .4529 · .200 · .5028 | 0/4 · .4913 | .4444 · .200 · .4932 | 0/3 · .4242 | .4899 / .4926 | KILLED / KILLED | no (15/15, 3 clear) / no (15/15, 3 clear) | 0.20 / 0.07 | `deriv_e37meta_AUDUSD_15m_result.json` |
| e38release | e38release: all (66), nseed 1 | 15m | 0.03 | .5021 (.4972) | .4655 · .200 · .5080 | 1/4 · .5174 | .4267 · .000 · .4658 | 0/4 · .4499 | .4899 / .4926 | KILLED / KILLED | no (15/15, 3 clear) / no (15/15, 0 clear) | 0.07 / 0.00 | `deriv_e38release_AUDUSD_15m_result.json` |
| e3union | e3union: all (305), nseed 1 | 15m | 0.03 | .5096 (.5022) | nan · .000 · — | 0/0 · — | .4446 · .000 · .4629 | 0/0 · — | .4899 / .4926 | KILLED / KILLED | no (0/15, 0 clear) / no (2/15, 0 clear) | — / 0.00 | `deriv_e3union_AUDUSD_15m_result.json` |
| e40forest | e40forest: all (66), nseed 1 | 15m | 0.03 | .5139 (.5057) | .4800 · .500 · .5296 | 0/2 · .4726 | .4558 · .200 · .5079 | 1/3 · .5339 | .4899 / .4926 | KILLED / KILLED | no (12/15, 6 clear) / no (15/15, 3 clear) | 0.25 / 0.13 | `deriv_e40forest_AUDUSD_15m_result.json` |
| e41pool | e41pool: all (66), nseed 1 | 15m | 0.03 | .5082 (.4992) | .4745 · .467 · .5392 | 1/3 · .4328 | .4322 · .133 · .5005 | 0/4 · .4788 | .4899 / .4926 | KILLED / KILLED | no (15/15, 7 clear) / no (15/15, 2 clear) | 0.40 / 0.13 | `deriv_e41pool_AUDUSD_15m_result.json` |
| e4seed3 | e4seed3: all (66), nseed 3 | 15m | 0.03 | .5040 (.4946) | .4647 · .357 · .5168 | 0/3 · .4294 | .4602 · .333 · .5138 | 0/3 · .4484 | .4899 / .4926 | KILLED / KILLED | no (14/15, 5 clear) / no (15/15, 5 clear) | 0.21 / 0.13 | `deriv_e4seed3_AUDUSD_15m_result.json` |
| e51fx | e51fx: all+fam:fx (82), nseed 1 | 15m | 0.03 | .5064 (.4949) | .4106 · .267 · .5063 | 0/3 · .4732 | .4713 · .267 · .5324 | 1/3 · .5038 | .4899 / .4926 | KILLED / KILLED | no (15/15, 4 clear) / no (15/15, 4 clear) | 0.27 / 0.27 | `deriv_e51fx_AUDUSD_15m_result.json` |
| e52ck | e52ck: all+fam:ck (77), nseed 1 | 15m | 0.03 | .5033 (.4936) | .4303 · .000 · .4806 | 1/4 · .4664 | .4495 · .267 · .5101 | 0/3 · .4154 | .4899 / .4926 | KILLED / KILLED | no (15/15, 0 clear) / no (15/15, 4 clear) | 0.00 / 0.13 | `deriv_e52ck_AUDUSD_15m_result.json` |
| e53ckcompact | e53ckcompact: list:compact_kernels.json (14), nseed 1 | 15m | 0.03 | .5017 (.4899) | .4607 · .000 · .4855 | 1/4 · .4889 | .4432 · .067 · .4954 | 1/4 · .5128 | .4899 / .4926 | KILLED / KILLED | no (15/15, 0 clear) / no (15/15, 1 clear) | 0.00 / 0.00 | `deriv_e53ckcompact_AUDUSD_15m_result.json` |
| e54ckcompactlogit | e54ckcompactlogit: list:compact_kernels.json (14), nseed 1 | 15m | 0.03 | .5093 (.4992) | .4879 · .067 · .5112 | 1/4 · .5342 | .4323 · .067 · .4857 | 1/4 · .5286 | .4899 / .4926 | KILLED / KILLED | no (15/15, 1 clear) / no (15/15, 1 clear) | 0.07 / 0.00 | `deriv_e54ckcompactlogit_AUDUSD_15m_result.json` |
| e55rl | e55rl: all+fam:rl (72), nseed 1 | 15m | 0.03 | .5131 (.5019) | .4574 · .462 · .5392 | 0/2 · .4358 | .4641 · .600 · .5524 | 0/1 · .4000 | .4899 / .4926 | KILLED / KILLED | no (13/15, 6 clear) / no (15/15, 9 clear) | 0.38 / 0.40 | `deriv_e55rl_AUDUSD_15m_result.json` |
| e56rs | e56rs: all+fam:rs (72), nseed 1 | 15m | 0.03 | .5115 (.4993) | .4279 · .357 · .5028 | 0/3 · .4853 | .4447 · .400 · .5252 | 0/1 · .4375 | .4899 / .4926 | KILLED / KILLED | no (14/15, 5 clear) / no (15/15, 6 clear) | 0.21 / 0.33 | `deriv_e56rs_AUDUSD_15m_result.json` |
| e57reflect | e57reflect: all (66), nseed 1 | 15m | 0.03 | .5103 (.5011) | .4372 · .286 · .5211 | 0/1 · .4815 | .4188 · .133 · .4746 | 1/1 · .5769 | .4899 / .4926 | KILLED / KILLED | no (14/15, 4 clear) / no (15/15, 2 clear) | 0.21 / 0.13 | `deriv_e57reflect_AUDUSD_15m_result.json` |
| e58nysreflect | e58nysreflect: all (66), nseed 1 | 15m | 0.03 | .5180 (.5075) | .4963 · .636 · .5566 | 1/3 · .5237 | .4816 · .267 · .5317 | 0/1 · .5405 | .4899 / .4926 | KILLED / KILLED | no (11/15, 7 clear) / no (15/15, 4 clear) | 0.45 / 0.27 | `deriv_e58nysreflect_AUDUSD_15m_result.json` |
| e59brier | e59brier: all (66), nseed 1 | 15m | 0.03 | .5036 (.4967) | .4160 · .333 · .4895 | 0/3 · .4552 | .4402 · .133 · .4937 | 2/3 · .5418 | .4899 / .4926 | KILLED / KILLED | no (15/15, 5 clear) / no (15/15, 2 clear) | 0.20 / 0.07 | `deriv_e59brier_AUDUSD_15m_result.json` |
| e5h30 | e5h30: all (66), nseed 1 | 30m | 0.03 | .5013 (.4850) | .4101 · .375 · .4917 | 1/1 · .5484 | .4249 · .250 · .4943 | 0/0 · — | .5004 / .4869 | KILLED / KILLED | no (8/15, 3 clear) / no (8/15, 2 clear) | 0.12 / 0.12 | `deriv_e5h30_AUDUSD_30m_result.json` |
| e60stack | e60stack: all (66), nseed 1 | 15m | 0.03 | .5053 (.4958) | .4669 · .400 · .5248 | 0/4 · .4563 | .4364 · .267 · .4986 | 1/4 · .4918 | .4899 / .4926 | KILLED / KILLED | no (15/15, 6 clear) / no (15/15, 4 clear) | 0.27 / 0.07 | `deriv_e60stack_AUDUSD_15m_result.json` |
| e61ord4 | e61ord4: all (66), nseed 1 | 15m | 0.03 | .5055 (.4939) | .4063 · .267 · .4985 | 1/3 · .5289 | .4429 · .400 · .5306 | 0/1 · .4314 | .4899 / .4926 | KILLED / KILLED | no (15/15, 4 clear) / no (15/15, 6 clear) | 0.20 / 0.33 | `deriv_e61ord4_AUDUSD_15m_result.json` |
| e62har | e62har: all (66), nseed 1 | 15m | 0.03 | .5042 (.4949) | .4578 · .571 · .5412 | 0/0 · — | .3514 · .333 · .4719 | 0/1 · .3750 | .4899 / .4926 | KILLED / KILLED | no (7/15, 4 clear) / no (6/15, 2 clear) | 0.57 / 0.00 | `deriv_e62har_AUDUSD_15m_result.json` |
| e64boot | e64boot: all (66), nseed 1 | 15m | 0.03 | .5036 (.4910) | .4227 · .200 · .5104 | 1/4 · .4703 | .4427 · .000 · .4863 | 0/3 · .4921 | .4899 / .4926 | KILLED / KILLED | no (15/15, 3 clear) / no (15/15, 0 clear) | 0.07 / 0.00 | `deriv_e64boot_AUDUSD_15m_result.json` |
| e65recency | e65recency: all (66), nseed 1 | 15m | 0.03 | .5024 (.4921) | .3569 · .125 · .4546 | 1/1 · .5455 | .3822 · .417 · .4943 | 1/3 · .4918 | .4899 / .4926 | KILLED / KILLED | no (8/15, 1 clear) / no (12/15, 5 clear) | 0.00 / 0.25 | `deriv_e65recency_AUDUSD_15m_result.json` |
| e66mh | e66mh: all (66), nseed 1 | 15m | 0.03 | .5045 (.4940) | .3283 · .000 · .3710 | 0/0 · — | .4074 · .000 · .4535 | 0/0 · — | .4899 / .4926 | KILLED / KILLED | no (3/15, 0 clear) / no (6/15, 0 clear) | 0.00 / 0.00 | `deriv_e66mh_AUDUSD_15m_result.json` |
| e67drfe | e67drfe: all (66), nseed 1 | 15m | 0.03 | .5162 (.5062) | .4768 · .400 · .5339 | 1/4 · .5393 | .4536 · .267 · .5041 | 2/4 · .5111 | .4899 / .4926 | KILLED / KILLED | no (15/15, 6 clear) / no (15/15, 4 clear) | 0.27 / 0.07 | `deriv_e67drfe_AUDUSD_15m_result.json` |
| e68agree | e68agree: all (66), nseed 1 | 15m | 0.03 | .5053 (.4964) | .4788 · .333 · .5258 | 1/4 · .4534 | .4150 · .200 · .4891 | 0/3 · .4912 | .4899 / .4926 | KILLED / KILLED | no (15/15, 5 clear) / no (15/15, 3 clear) | 0.20 / 0.20 | `deriv_e68agree_AUDUSD_15m_result.json` |
| e70play | e70play: all+fam:hy (75), nseed 1 | 15m | 0.03 | .5039 (.4932) | .4313 · .357 · .5162 | 1/3 · .4976 | .4577 · .200 · .5002 | 0/3 · .4525 | .4899 / .4926 | KILLED / KILLED | no (14/15, 5 clear) / no (15/15, 3 clear) | 0.29 / 0.20 | `deriv_e70play_AUDUSD_15m_result.json` |
| e71playreset | e71playreset: all+fam:hr (75), nseed 1 | 15m | 0.03 | .5037 (.4923) | .4132 · .267 · .4972 | 0/3 · .4567 | .4312 · .333 · .5170 | 0/3 · .4098 | .4899 / .4926 | KILLED / KILLED | no (15/15, 4 clear) / no (15/15, 5 clear) | 0.13 / 0.20 | `deriv_e71playreset_AUDUSD_15m_result.json` |
| e72shock | e72shock: all+fam:d2 (67), nseed 1 | 15m | 0.03 | .5058 (.4953) | .4342 · .231 · .4883 | 0/2 · .4375 | .4473 · .333 · .5099 | 0/1 · .4576 | .4899 / .4926 | KILLED / KILLED | no (13/15, 3 clear) / no (15/15, 5 clear) | 0.15 / 0.13 | `deriv_e72shock_AUDUSD_15m_result.json` |
| e73mono | e73mono: all (66), nseed 1 | 15m | 0.03 | .5041 (.4951) | .3976 · .200 · .4798 | 1/4 · .5367 | .4071 · .200 · .4836 | 1/3 · .5032 | .4899 / .4926 | KILLED / KILLED | no (15/15, 3 clear) / no (15/15, 3 clear) | 0.13 / 0.00 | `deriv_e73mono_AUDUSD_15m_result.json` |
| e74logitsig | e74logitsig: all (66), nseed 1 | 15m | 0.03 | .5125 (.5030) | .4614 · .400 · .5249 | 1/4 · .4824 | .4428 · .200 · .4970 | 0/4 · .4711 | .4899 / .4926 | KILLED / KILLED | no (15/15, 6 clear) / no (15/15, 3 clear) | 0.13 / 0.07 | `deriv_e74logitsig_AUDUSD_15m_result.json` |
| e75lgbsig | e75lgbsig: all (66), nseed 1 | 15m | 0.03 | .5012 (.4952) | .4188 · .133 · .4839 | 1/4 · .5218 | .4444 · .200 · .4971 | 0/4 · .4871 | .4899 / .4926 | KILLED / KILLED | no (15/15, 2 clear) / no (15/15, 3 clear) | 0.07 / 0.07 | `deriv_e75lgbsig_AUDUSD_15m_result.json` |
| e76drfelgb | e76drfelgb: all (66), nseed 1 | 15m | 0.03 | .5053 (.4980) | .4483 · .133 · .5011 | 2/4 · .5505 | .4691 · .200 · .5096 | 1/4 · .4930 | .4899 / .4926 | KILLED / KILLED | no (15/15, 2 clear) / no (15/15, 3 clear) | 0.13 / 0.13 | `deriv_e76drfelgb_AUDUSD_15m_result.json` |
| e79qorder | e79qorder: all+fam:qo (88), nseed 1 | 15m | 0.03 | .5044 (.4920) | .3932 · .429 · .5029 | 0/1 · .4138 | .4167 · .200 · .4967 | 0/1 · .4615 | .4899 / .4926 | KILLED / KILLED | no (14/15, 6 clear) / no (15/15, 3 clear) | 0.36 / 0.07 | `deriv_e79qorder_AUDUSD_15m_result.json` |
| e7fitall | e7fitall: all (66), nseed 1 | 15m | 0.03 | .5037 (.4951) | .4978 · .467 · .5465 | 0/3 · .4684 | .4169 · .267 · .4882 | 0/3 · .4660 | .4899 / .4926 | KILLED / KILLED | no (15/15, 7 clear) / no (15/15, 4 clear) | 0.27 / 0.07 | `deriv_e7fitall_AUDUSD_15m_result.json` |
| e80qperm | e80qperm: all+fam:qp (88), nseed 1 | 15m | 0.03 | .5052 (.4941) | .4553 · .429 · .5378 | 1/2 · .5463 | .4597 · .357 · .5299 | 0/1 · .3673 | .4899 / .4926 | KILLED / KILLED | no (14/15, 6 clear) / no (14/15, 5 clear) | 0.29 / 0.36 | `deriv_e80qperm_AUDUSD_15m_result.json` |
| e81tail0 | e81tail0: all (66), nseed 1 | 15m | 0.03 | .5106 (.4997) | .4187 · .300 · .5166 | 0/1 · .2500 | .4728 · .200 · .5177 | 0/1 · .5000 | .4899 / .4926 | KILLED / KILLED | no (10/15, 3 clear) / no (15/15, 3 clear) | 0.20 / 0.07 | `deriv_e81tail0_AUDUSD_15m_result.json` |
| e82tailaux | e82tailaux: all (66), nseed 1 | 15m | 0.03 | .5083 (.4951) | .4394 · .273 · .5252 | 0/2 · .4614 | .4784 · .500 · .5651 | 0/0 · — | .4899 / .4926 | KILLED / KILLED | no (11/15, 3 clear) / no (14/15, 7 clear) | 0.18 / 0.43 | `deriv_e82tailaux_AUDUSD_15m_result.json` |
| e8tb2 | e8tb2: all (66), nseed 1 | 15m | 0.03 | .5022 (.4895) | .4062 · .538 · .5192 | 0/1 · .3488 | .4349 · .333 · .5011 | 1/4 · .4578 | .4899 / .4926 | KILLED / KILLED | no (13/15, 7 clear) / no (15/15, 5 clear) | 0.54 / 0.20 | `deriv_e8tb2_AUDUSD_15m_result.json` |
| e9blockval | e9blockval: all (66), nseed 1 | 15m | 0.03 | .5062 (.4934) | .4646 · .400 · .5110 | 0/4 · .4910 | .4246 · .200 · .4898 | 1/4 · .4943 | .4899 / .4926 | KILLED / KILLED | no (15/15, 6 clear) / no (15/15, 3 clear) | 0.07 / 0.07 | `deriv_e9blockval_AUDUSD_15m_result.json` |
| r4fixldn | r4fixldn: rule:fx_ldn_prefix_u60 (1), nseed 0 | 15m | 1.0 | .5320 (.5009) | .4937 · .067 · .5108 | 0/0 · — | .5022 · .000 · .5139 | 0/0 · — | .4899 / .4926 | KILLED / KILLED | no (15/15, 1 clear) / no (15/15, 0 clear) | 0.00 / 0.00 | `deriv_r4fixldn_AUDUSD_15m_result.json` |
| r5fixecb | r5fixecb: rule:fx_ecb_prefix_u60 (1), nseed 0 | 15m | 1.0 | .5160 (.4863) | .4731 · .133 · .5146 | 0/0 · — | .4968 · .000 · .5150 | 0/0 · — | .4899 / .4926 | KILLED / KILLED | no (15/15, 2 clear) / no (15/15, 0 clear) | 0.07 / 0.00 | `deriv_r5fixecb_AUDUSD_15m_result.json` |
<!-- DERIV-FEED-TABLES:END -->
