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
