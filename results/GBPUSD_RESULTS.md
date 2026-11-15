> **SCOPE: GBPUSD** (key-specific — results of record + UP/DOWN leaderboard). Generic methods: docs/METHODS_CATALOG.md. Sweep menu: SWEEP_MATRIX.md. See REPO_MAP.md. Bootstrapped 2026-06-09 (new currency; structure copied from AUDUSD_RESULTS.md / EURUSD_RESULTS.md). Current scope: **15m only** (deriv-FX-deployable floor + program's best direction horizon).

# GBPUSD — Results Ledger (unique key: currency × timeframe × side)

**Pair tag: `GBPUSD`.** GBPUSD ("cable") is a **USD-QUOTE** pair (GBPUSD up ⇒ USD down ⇒ GBP up). It is simultaneously:
- a **USD major in the EUROPEAN bloc** — the tightest EURUSD correlate among the untested majors. Cross-pair POOLING / USD-residual was THE certifying keystone at **EURUSD 15m** (UP p10 .567 / DOWN .574, the one sibling where pooling won) but **DILUTED** own-pair signal at USDJPY and AUDUSD. GBPUSD sits squarely between those precedents → **the pooling lever has a genuinely open prior here** and is run, not argued. The **EURGBP triangular residual** (USD leg algebraically cancelled, SWEEP_MATRIX N2) is uniquely on-disk-runnable for this pair.
- a currency whose **idiosyncratic information (BoE policy, UK CPI/GDP/labour, gilt-market/politics risk premia) arrives largely in the LONDON morning** — yet the certified direction edge was **NY-session-concentrated at ALL THREE siblings** (EURUSD/USDJPY/AUDUSD; LDN+Asia null every time). So the session question is tested symmetrically (ny/ldn/asia/all), not assumed.

**The unique result key is `(currency, timeframe, side)`, side ∈ {UP, DOWN}.** Where a key is unmeasured it is `UNTESTED`. COMBINED is context only, never an UP/DOWN number.

**Conventions.** Deriv-faithful settlement (bar-close approx at 15m, mid-to-mid, next-tick entry +1s, ties LOSE), breakeven **0.541** (deriv payout R≈1.85). Splits: bars train 2012-21 / val 2022-23 / test 2024 / test 2025 / oos 2026 (strict OOS). Selection on VAL worst-half (never VAL-acc-max; `corr(VAL,OOS)=−0.54`). Moved-bars-only; verify moved up-rate ∈ [0.47,0.53]; per-year CI95 (bootstrap). de-overlap = `nonoverlap_chrono` gap=900s. CERTIFY a side ONLY via full per-fold-refit CPCV at the operating gate: p10 ≥ 0.541 AND ≥ ~80% of 15 purged paths clear 0.541; adversarially verify every positive. **Target: >65% accuracy** (achievable, if at all, only at tight coverage — siblings hit ~.60–.65 at cov2%). Bar features (`features/GBPUSD_2012..2026.parquet`, 244 cols) + raw 10s ticks (`/home/sean/git/processed/GBPUSD/`, 2012-01→2026-06) both on disk. Last updated 2026-06-09 (bootstrap).

---

## MASTER KEY TABLE — one row per (GBPUSD, timeframe, side)

> **Deriv-feed program (2026-11-15):** the numbers in this table are Dukascopy-era evidence. On the Deriv feed (DEV 2025-08-11..2026-07-31) none of this pair's frozen 15m/30m books is SUPPORTED as deployed — see "GBPUSD — DERIV-FEED PROGRAM" at the end of this file.

| Key (currency · timeframe · side) | **Best OOS % (2026)** | Model id · content_id | Description | Status |
|---|---|---|---|---|
| GBPUSD · **15m** · UP | **xpair-NY K=8 refit-CPCV p10 .6251 @cov2 / .6552 @cov1 (15/15 every cov); means .6535 / .6874** ✅ CERTIFIED, REFIT-DEPENDENT | `GBPUSD.m15ny_xpair_seedens8.v1` `9ee9a4634a6b73e0` (340 feats, K=8 seed-ens, NY session, cov-gate .01); `gbpusd_15m_cpcv_xpair_ny_seedens8_result.json`; frozen `books/GBPUSD.m15ny_xpair_seedens8.v1/` | **K=8 unseats K=3** (+.0082 p10 @cov1; K=8 LIFTS unlike 3/3 own-pair siblings — xpair 340-feat richer diversity). Borderline-65%+ floor at cov1. EUR-bloc discriminator resolved (pooling wins at GBPUSD, same as EURUSD, not JPY/AUD). Refit-dependent confirmed; deploy with periodic retrain; tick haircut ~1pp. K=3 book `df279204d247f618` superseded but preserved. | ✅ CERTIFIED + FROZEN (xpair-NY K=8 refit-CPCV) 2026-06-10 |
| GBPUSD · **15m** · DOWN | **xpair-NY K=8 refit-CPCV p10 .6255 @cov2 / .6395 @cov1 (15/15 every cov); means .6449 / .6660** ✅ CERTIFIED, REFIT-DEPENDENT | same book as UP: `GBPUSD.m15ny_xpair_seedens8.v1` `9ee9a4634a6b73e0`; `gbpusd_15m_cpcv_xpair_ny_seedens8_result.json` | **K=8 unseats K=3** (+.0101 @cov2 — the STRONGEST lift in the K=8 run; +.0072 @cov1). >65% floor open (mean .666; refit-p10 .6395 — recent-regime era binds; improve loop exhausted: orthochan all ADDS=False, K>8 diminishing returns). | ✅ CERTIFIED + FROZEN (xpair-NY K=8 refit-CPCV) 2026-06-10 |
| GBPUSD · 1m/2m/5m/10m/30m · UP/DOWN | `UNTESTED` | — | Out of current scope (goal = 15m). Bar + tick data present; bootstrap when scoped. | UNTESTED |

**Magnitude** (|ret|≥Q) is sign-invariant → no up/down key; tracked in `docs/MAGNITUDE_FINDINGS.md` (GBPUSD pending).

---

# Per-timeframe detail

## GBPUSD × 15m
Breakeven 0.541. Deriv-FX-deployable (15m = forex Rise/Fall minimum expiry) AND the program's strongest direction horizon (EURUSD 15m cross-pair certified BOTH sides p10 .567/.574; USDJPY 15m NY own-pair p10 .586/.572; AUDUSD 15m NY own-pair seed-ens K=3 p10 .596/.596). **Mechanistic prior (above): pooling is the genuinely open EUR-bloc question; session tested symmetrically (LDN-morning info vs the 3-sibling NY precedent).**

### Combined-book / single-pair experiments
| # | Method (file) | Result (2024 / 2025 / 2026 moved-AUC; gate cov2% COMB wr) | Verdict |
|---|---|---|---|
| BASE | single-pair LGBM, 239 base feats, 15m own-clock label, all-session (`gbpusd_15m_base.py`) | AUC .5262 / .5250 / .5098; cov2% COMB .6489[.624,.673] / .5582[.536,.580] / .5052[.470,.541] | ✅ **SURVIVED** standalone via 2024 (CI-lo .624 ≫ BE — strongest sibling-family 2024 base); 2025 point>BE, CI-lo misses; **2026 frozen book DEAD (~.50 both sides)** = the sibling-universal refit-dependence. up-rate .504/.505/.498 tripwire-clean. VAL moved-AUC .5285, best_iter 33. `gbpusd_15m_base_result.json` |
| R9 | issue #9 date-only retrospective replay: A=current v1; B=clean capped legacy-date refit; C=clean capped date-refresh refit | Apr–May 2026 standardized whole-book correctness: A .5603 (n=116, yield .000633); B .5339 (n=118, .000362); C .5393 (n=89, .000317). C UP .5593 (n=59); DOWN .5000 (n=30). | ⚪ **INCONCLUSIVE — `control_not_estimable`.** Simultaneous promotion lower bounds: C−A yield −.001708, C−B yield −.001343, C-UP accuracy−.5 −.157652, C-DOWN accuracy−.5 −.238356. No survivor/candidate; incumbent unchanged. Pair: `results/json/m15_book_refresh_53547498c599f2877a2f6616ae725f8f27e11070e3b550e82eb26ec039fb23f6_GBPUSD_replay_result.json`; joint/shadow: `results/json/m15_book_refresh_53547498c599f2877a2f6616ae725f8f27e11070e3b550e82eb26ec039fb23f6_joint_replay_result.json`, `results/json/m15_book_refresh_53547498c599f2877a2f6616ae725f8f27e11070e3b550e82eb26ec039fb23f6_shadow_spec_result.json`. |
| Q21-DOWN | issue #21 outcome-blind Deriv paired-proposal source-feasibility screen (`gbpusd_m15_down_quote_skew_screen_v1.py`) | The fixed window sealed 633,552 quote rows (52,796 complete 12-lane blocks) and 15,072 opaque tick shards, but shard filename bounds represented only 3 NY dates versus the preregistered minimum 10. | ⚪ **NOT COMPUTABLE — `capability_deferred`.** `tick_source_cannot_represent_minimum_dates:3<10`; settlement and inference are null. No efficacy, alpha, candidate, activation, fill, or P&L claim; incumbent unchanged. Sources: `results/json/gbpusd_m15_down_quote_skew_screen_v1_result.json`, `results/json/gbpusd_m15_down_quote_skew_source_snapshot_v1.json`. |

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

### Session segmentation (A9) — `gbpusd_15m_cpcv_session.py ny` (symmetric refit-CPCV)
**NY-session refit-CPCV: BOTH SIDES CERTIFIED at EVERY coverage, all 15 paths — the strongest 15m certified result in the 4-pair family (single-seed).**
| cov | UP p10 (mean) | DOWN p10 (mean) | COMB p10 (mean) | med_n/path (UP/DN) |
|---|---|---|---|---|
| 0.05 | .5874 (.612) | .5840 (.6031) | .5870 (.6064) | 1880 / 1372 |
| 0.03 | .5953 (.6206) | .5977 (.6196) | .5993 (.619) | 1131 / 823 |
| 0.02 | **.6039 (.629)** | **.5937 (.6227)** | .6013 (.6254) | 752 / 539 |
| 0.01 | **.6192 (.6452)** | .5978 (.6318) | .6089 (.6387) | 374 / 251 |

frac_clear=1.0 every cell. NY AUC mean .5409 (min .5303, max .5472) — family-high (AUD NY .5363). Up-rate tripwire clean. NY lift over the certified all-session floor: +.04–.06 p10 (concentration, not dilution — 4th consecutive sibling). Single-seed NY already beats AUDUSD's seed-ens K=3 deliverable (.596@cov2). `gbpusd_15m_cpcv_session_ny_multicov_result.json`. REFIT-DEPENDENT (frozen-2021 forward decays — adversarial frozen-forward validation queued before freeze).

### ★ Cross-pair pooling (A6) — the EUR-bloc keystone CONFIRMED at GBPUSD → NEW INCUMBENT
**Screen** (`gbpusd_15m_xpair.py xpbase`, all-session, frozen-2012-21): PASSES pre-reg falsifier — VAL AUC .5310 > base .5285; 2026 cov2 COMB .530 > .5012; 2025 cov2 DOWN .5917. `gbpusd_15m_xpair_xpbase_result.json`.

**Escalation — xpair-NY per-fold-refit CPCV** (`gbpusd_15m_cpcv_xpair.py 3`, 340-feat xpbase matrix, ties-strict, 15 purged paths): **BEATS the own-pair NY incumbent on BOTH SIDES at EVERY coverage, frac_clear=1.0 every cell:**
| cov | UP p10 (mean) [vs own-pair] | DOWN p10 (mean) [vs own-pair] | COMB p10 | med_n UP/DN |
|---|---|---|---|---|
| 0.05 | .5962 (.6139) [+.009] | .5908 (.6054) [+.007] | .5946 | 3200 / 2679 |
| 0.03 | .6100 (.6307) [+.015] | .6054 (.6185) [+.008] | .6085 | 2132 / 1675 |
| 0.02 | .6206 (.6477) [+.017] | .6052 (.6309) [+.012] | .6186 | 1497 / 1176 |
| 0.01 | **.6515 (.6766)** [+.032] | **.6229 (.652)** [+.025] | .6408 | 800 / 612 |

AUC mean .5473 (min .5353, max .5539) — family record. Up-rate tripwire clean. **UP @cov1: p10 .6515 single-seed — the >65% floor at K=1; the K=3 replicate reads .6470 (order-stat noise band .647–.6515 → honest claim: borderline-65% floor, means .68).** DOWN @cov1 .6229 (mean .652). GBPUSD follows the EURUSD (pooling) precedent, not JPY/AUD — the EUR-bloc discriminator resolved. `gbpusd_15m_cpcv_xpair_ny_multicov_result.json`. REFIT-DEPENDENT presumed (sibling-universal) — adversarial frozen-forward + tick-settlement pending before freeze.

**★ IMPROVE — seed-ensemble K=3 on xpair-NY (matched folds, `gbpusd_15m_cpcv_xpair_ny_seedens3_result.json`): DOWN genuinely lifts, UP saturates.** DOWN p10 +.005–.010 at every cov WITH mean +.010 (the genuine-lift criterion); UP mixed at the tails (cov1 −.0045 order-stat). K=3: DOWN .5955/.6122/.6154/.6323 @cov5/3/2/1 (means .61–.66), UP .6016/.6211/.6209/.6470 (means .62–.68), COMB .5994/.6145/.6203/.6427, AUC .5488, frac 1.0 every cell.

**★★ IMPROVE — seed-ensemble K=8 on xpair-NY (matched folds, `gbpusd_15m_cpcv_xpair_ny_seedens8_result.json`): UNEXPECTED LIFT at K=8 — the xpair 340-feat model is NOT saturated at K=3 (unlike 3/3 own-pair siblings).** Xpair richer feature diversity gives seed averaging more variance to reduce. DOWN @cov2 +.0101 (strongest lift in the run), UP @cov2 +.0042, cov1 UP +.0082, cov1 DOWN +.0072. **K=8 = new deliverable config.** AUC mean .5492, frac 1.0 every cell, 15/15 paths.
| cov | UP p10 (mean) [Δ vs K3] | DOWN p10 (mean) [Δ vs K3] | COMB p10 (mean) |
|---|---|---|---|
| 0.05 | .6049 (.6244) [+.003] | .5969 (.6168) [+.001] | .6007 (.621) |
| 0.03 | .6163 (.6475) [−.005] | .6156 (.6346) [+.003] | .6144 (.639) |
| 0.02 | **.6251 (.6535)** [+.004] | **.6255 (.6449)** [**+.010**] | .6262 (.645) |
| 0.01 | **.6552 (.6874)** [+.008] | **.6395 (.6660)** [+.007] | .6594 (.684) |

### Orthogonal channel famonly screen (Ortho) — `gbpusd_15m_orthochan.py`
**ALL 4 families ADDS=False — mirror of AUDUSD precedent; xpair 340-feat matrix subsumes all.**
| Family | base VAL AUC | base+block VAL AUC | ΔVal AUC | 2026 cov2 COMB | ADDS |
|---|---|---|---|---|---|
| eurgbp (8 feats) | .5401 | .5447 | +.0046 | .5362 | ❌ |
| risk+eurobloc (18 feats) | .5401 | .5426 | +.0025 | .5282 | ❌ |
| RS± semivar (3 feats) | .5401 | .5408 | +.0007 | .5259 | ❌ |
| carryrank (12 feats) | .5401 | .5442 | +.0041 | .4914 | ❌ |

Falsifier: ADDS iff ΔVal-AUC > .005 AND 2026 cov2 > base (.4892). None pass either gate. **Conclusion: no orthogonal channel family adds net-new signal to the 239-feat own-pair base; the xpair book's cross-pair features already capture the most these channels can offer.** `gbpusd_15m_orthochan_result.json`.

## UP/DOWN LEADERBOARD (current best per side, certified-or-best-available) — SWEEP CLOSED 2026-06-10
| Side | Best certified (refit-CPCV p10) | Best available (mean) | Book | Status |
|---|---|---|---|---|
| **15m UP** | **.6552 @cov1 (K8) / .6251 @cov2** (xpair-NY K=8, 15/15 every cov) | .6874 @cov1 / .6535 @cov2 | `GBPUSD.m15ny_xpair_seedens8.v1` `9ee9a4634a6b73e0` | ✅ CERTIFIED + FROZEN (2026-06-10); borderline-65%+ floor at cov1; tick haircut ~1pp; deploy w/ periodic retrain |
| **15m DOWN** | **.6395 @cov1 / .6255 @cov2** (xpair-NY K=8, 15/15 every cov) | .6660 @cov1 / .6449 @cov2 | same: `GBPUSD.m15ny_xpair_seedens8.v1` `9ee9a4634a6b73e0` | ✅ CERTIFIED + FROZEN (2026-06-10); >65% floor open (mean .666; K=8 strongest lever exhausted — recent era binds) |
| _(superseded)_ xpair-NY K=3 | UP .6470 @cov1 / .6209 @cov2; DOWN .6323 @cov1 / .6154 @cov2 | .6787/.6624 @cov1 | `GBPUSD.m15ny_xpair_seedens.v1` `df279204d247f618` | superseded by K=8 (UP +.0082/@cov1, DOWN +.0101/@cov2 p10) |
| _(superseded)_ own-pair NY K=1 | UP .6039 / DOWN .5937 @cov2 (15/15) | .629 / .623 | — | subsumed by xpair-NY (+.007–.032 p10 every cell) |
| _(superseded)_ all-session own-pair | UP .5532 / DOWN .5527 @cov2 (14/15) | .568 / .567 | — | subsumed by NY |

_Provenance: every number traces to a `*_result.json` (Tier-1). Updated 2026-07-21 (issue #21 source-feasibility handoff; sweep and K=8 deliverable unchanged)._

## GBPUSD — DERIV-FEED PROGRAM (Deriv-only data; opened 2026-11-15)

> **SCOPE:** this key on the Deriv feed only — Deriv 1-second mid ticks (`/mnt/data/deriv`), no Dukascopy data. Labels settle on Deriv ticks (entry first tick >= decision + 1 s, exit last tick <= entry target + H, ties lose); DEV window 2025-08-11..2026-07-31 for training, selection and CPCV; HOLDOUT 2026-08-01..2026-11-14 sealed (opened once per finalized candidate). Protocol and tools: `docs/DERIV_SOTA_PROGRAM.md`. Tables below are rendered from the result JSONs by `scripts/render_ledger.py` (do not edit by hand).

<!-- DERIV-FEED-NOTES:BEGIN -->
- **Incumbent benchmark on TRADEABLE decisions, books as trained (R1bt, 13:00-16:35 + 18:00-22:00 ET):** UP best `GBPUSD.m15ny_xpair_seedens.v1` 0.6300 [0.5300, 0.7200] n=100 (NOT SUPPORTED); DOWN best `GBPUSD.m15ny_xpair_seedens.v1` 0.5303 [0.4091, 0.6515] n=66 (NOT SUPPORTED).
- **Books as deployed today (R1t, true clock, NY 08:00-16:35, tradeable):** UP best `GBPUSD.m15ny_xpair_seedens.v1` 0.5299 [0.4701, 0.5896] n=268 (NOT SUPPORTED); DOWN best `GBPUSD.m15ny_xpair_seedens8.v1` 0.5751 [0.5107, 0.6352] n=233 (NOT SUPPORTED).
- **DEV-CPCV certified sides, GOAL rule (>= 12 of 15 paths counted and clear, p10 >= .541; tradeable runs):** none yet. DEV certification is necessary, not sufficient: the sealed holdout decides (docs/DERIV_SOTA_PROGRAM.md).
- Certified only under the repository rule (too few counted paths for the goal rule): e16nyeve 15m UP (cov 0.03: no (4/15, 4 clear); forward 0/0).
- **DEV leaderboard 15m UP** (closest to the goal rule at each run's operating coverage): `e14nylate` 8/14 clear, p10 0.468, excess +0.045; `e16nyeve` 4/4 clear, p10 0.567, excess +0.095; `e7fitall` 4/15 clear, p10 0.435, excess +0.009.
- **DEV leaderboard 15m DOWN** (closest to the goal rule at each run's operating coverage): `e16nyeve` 4/6 clear, p10 0.471, excess +0.075; `e32spec` 3/15 clear, p10 0.457, excess +0.007; `e6pool` 3/15 clear, p10 0.428, excess -0.012.
- Superseded: `e1` and `r2` ran before the venue tradeability filter (they include untradeable blackout decisions); their tradeable re-runs are `e1t` and `r2t`.
<!-- DERIV-FEED-NOTES:END -->

<!-- DERIV-FEED-TABLES:BEGIN -->
**Frozen books on the Deriv feed — true clock, ny (America/New_York 08:00-17:00, DST-correct, sessions.session_mask), ALL decisions incl. the untradeable blackout** (`deriv_r1_frozen_books_dev_result.json`; strategy gate, tick-settled, ties lose; SUPPORTED iff n >= 50 and CI95-lo >= .541):

| Book | AUC (moved) | gate rate | UP n · WR [CI95] | DOWN n · WR [CI95] | COMB n · WR | verdict UP / DOWN |
|---|---|---|---|---|---|---|
| `GBPUSD.m15ny_xpair_seedens.v1` | .5164 | .0145 | 270 · .5333 [.4741, .5926] | 243 · .5720 [.5062, .6296] | 505 · .5505 | NOT SUPPORTED / NOT SUPPORTED |
| `GBPUSD.m15ny_xpair_seedens8.v1` | .5169 | .0147 | 269 · .5167 [.4572, .5762] | 233 · .5751 [.5107, .6352] | 496 · .5444 | NOT SUPPORTED / NOT SUPPORTED |

**Frozen books on the Deriv feed — training clock (true UTC - 5 h), nylate (America/New_York 13:00-22:00, DST-correct), ALL decisions incl. the untradeable blackout** (`deriv_r1b_frozen_books_trainclock_nylate_dev_result.json`; strategy gate, tick-settled, ties lose; SUPPORTED iff n >= 50 and CI95-lo >= .541):

| Book | AUC (moved) | gate rate | UP n · WR [CI95] | DOWN n · WR [CI95] | COMB n · WR | verdict UP / DOWN |
|---|---|---|---|---|---|---|
| `GBPUSD.m15ny_xpair_seedens.v1` | .5504 | .0177 | 342 · .7339 [.6842, .7807] | 271 · .7491 [.6974, .8007] | 525 · .7295 | SUPPORTED / SUPPORTED |
| `GBPUSD.m15ny_xpair_seedens8.v1` | .5506 | .0181 | 351 · .7379 [.6895, .7807] | 270 · .7333 [.6778, .7852] | 532 · .7199 | SUPPORTED / SUPPORTED |

**Frozen books on the Deriv feed — training clock (true UTC - 5 h), nylate (America/New_York 13:00-22:00, DST-correct), tradeable decisions only (Deriv purchase blackout excluded)** (`deriv_r1bt_frozen_books_trainclock_nylate_dev_tradeable_result.json`; strategy gate, tick-settled, ties lose; SUPPORTED iff n >= 50 and CI95-lo >= .541):

| Book | AUC (moved) | gate rate | UP n · WR [CI95] | DOWN n · WR [CI95] | COMB n · WR | verdict UP / DOWN |
|---|---|---|---|---|---|---|
| `GBPUSD.m15ny_xpair_seedens.v1` | .5255 | .0048 | 100 · .6300 [.5300, .7200] | 66 · .5303 [.4091, .6515] | 162 · .5802 | NOT SUPPORTED / NOT SUPPORTED |
| `GBPUSD.m15ny_xpair_seedens8.v1` | .5255 | .0048 | 101 · .6238 [.5347, .7129] | 67 · .5224 [.4030, .6418] | 165 · .5758 | NOT SUPPORTED / NOT SUPPORTED |

**Frozen books on the Deriv feed — true clock, ny (America/New_York 08:00-17:00, DST-correct, sessions.session_mask), tradeable decisions only (Deriv purchase blackout excluded)** (`deriv_r1t_frozen_books_dev_tradeable_result.json`; strategy gate, tick-settled, ties lose; SUPPORTED iff n >= 50 and CI95-lo >= .541):

| Book | AUC (moved) | gate rate | UP n · WR [CI95] | DOWN n · WR [CI95] | COMB n · WR | verdict UP / DOWN |
|---|---|---|---|---|---|---|
| `GBPUSD.m15ny_xpair_seedens.v1` | .5130 | .0150 | 268 · .5299 [.4701, .5896] | 243 · .5720 [.5062, .6296] | 503 · .5487 | NOT SUPPORTED / NOT SUPPORTED |
| `GBPUSD.m15ny_xpair_seedens8.v1` | .5134 | .0153 | 267 · .5131 [.4532, .5730] | 233 · .5751 [.5107, .6352] | 494 · .5425 | NOT SUPPORTED / NOT SUPPORTED |

**Per-fold-refit CPCV on the Deriv feed** (DEV 2025-08-11..2026-07-31, NY DST session, 15 paths; p10/frac = path win-rate 10th percentile / share of paths >= .541; fwd = forward-walk paths clearing .541 and their mean):

| Run | Model / features | H | cov | AUC mean (min) | UP p10 · frac · mean | UP fwd | DOWN p10 · frac · mean | DOWN fwd | always UP / DOWN | repo-rule verdict UP / DOWN | GOAL rule UP / DOWN (paths counted) | frac @.565 UP / DOWN | result JSON |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| e14nylate | e14nylate: all (66), nseed 1 | 15m | 0.03 | .5023 (.4890) | .4679 · .571 · .5320 | 0/1 · .4333 | .4175 · .214 · .5010 | 2/4 · .5690 | .4876 / .4961 | KILLED / KILLED | no (14/15, 8 clear) / no (14/15, 3 clear) | 0.36 / 0.21 | `deriv_e14nylate_GBPUSD_15m_result.json` |
| e15nypm | e15nypm: all (66), nseed 1 | 15m | 0.03 | .4887 (.4692) | .4292 · .167 · .4628 | 0/0 · — | .3958 · .077 · .4560 | 0/0 · — | .4780 / .5080 | KILLED / KILLED | no (6/15, 1 clear) / no (13/15, 1 clear) | 0.00 / 0.00 | `deriv_e15nypm_GBPUSD_15m_result.json` |
| e16nyeve | e16nyeve: all (66), nseed 1 | 15m | 0.03 | .5150 (.4973) | .5667 · 1.0 · .5870 | 0/0 · — | .4714 · .667 · .5570 | 0/0 · — | .4975 / .4833 | CERTIFIED / KILLED | no (4/15, 4 clear) / no (6/15, 4 clear) | 0.75 / 0.50 | `deriv_e16nyeve_GBPUSD_15m_result.json` |
| e17nyam | e17nyam: all (66), nseed 1 | 15m | 0.03 | .4971 (.4863) | .4474 · .182 · .5043 | 0/0 · — | .3871 · .091 · .4521 | 0/0 · — | .5013 / .4915 | KILLED / KILLED | no (11/15, 2 clear) / no (11/15, 1 clear) | 0.09 / 0.09 | `deriv_e17nyam_GBPUSD_15m_result.json` |
| e18lowcap | e18lowcap: all (66), nseed 1 | 15m | 0.03 | .4956 (.4895) | .4306 · .067 · .4907 | 1/4 · .4728 | .4184 · .200 · .4862 | 0/4 · .4610 | .4908 / .4988 | KILLED / KILLED | no (15/15, 1 clear) / no (15/15, 3 clear) | 0.00 / 0.07 | `deriv_e18lowcap_GBPUSD_15m_result.json` |
| e1 | e1: all (66), nseed 1 | 15m | 0.03 | .5036 (.4956) | .4606 · .267 · .5170 | 1/2 · .5200 | .4359 · .267 · .5084 | 1/3 · .5029 | .4915 / .4980 | KILLED / KILLED | no (15/15, 4 clear) / no (15/15, 4 clear) | 0.07 / 0.13 | `deriv_e1_GBPUSD_15m_result.json` |
| e1t | e1t: all (66), nseed 1 | 15m | 0.03 | .4975 (.4886) | .4248 · .286 · .5026 | 1/2 · .4755 | .3948 · .067 · .4772 | 2/3 · .4921 | .4908 / .4988 | KILLED / KILLED | no (14/15, 4 clear) / no (15/15, 1 clear) | 0.21 / 0.00 | `deriv_e1t_GBPUSD_15m_result.json` |
| e29logit | e29logit: all (66), nseed 1 | 15m | 0.03 | .4966 (.4836) | .4287 · .000 · .4716 | 1/4 · .4814 | .4217 · .000 · .4671 | 0/4 · .4420 | .4908 / .4988 | KILLED / KILLED | no (15/15, 0 clear) / no (15/15, 0 clear) | 0.00 / 0.00 | `deriv_e29logit_GBPUSD_15m_result.json` |
| e30nystroem | e30nystroem: all (66), nseed 1 | 15m | 0.03 | .4927 (.4799) | .4452 · .000 · .4686 | 1/3 · .5580 | .4434 · .067 · .4957 | 0/4 · .4946 | .4908 / .4988 | KILLED / KILLED | no (11/15, 0 clear) / no (15/15, 1 clear) | 0.00 / 0.07 | `deriv_e30nystroem_GBPUSD_15m_result.json` |
| e32spec | e32spec: all (66), nseed 1 | 15m | 0.03 | .5016 (.4926) | .3582 · .000 · .4450 | 0/4 · .4483 | .4566 · .200 · .5057 | 1/4 · .5505 | .4908 / .4988 | KILLED / KILLED | no (15/15, 0 clear) / no (15/15, 3 clear) | 0.00 / 0.07 | `deriv_e32spec_GBPUSD_15m_result.json` |
| e33comp33 | e33comp33: all (66), nseed 1 | 15m | 0.1 | .4975 (.4886) | .3743 · .133 · .4781 | 0/3 · .4144 | .4176 · .200 · .4950 | 1/3 · .5318 | .4908 / .4988 | KILLED / KILLED | no (15/15, 2 clear) / no (15/15, 3 clear) | 0.07 / 0.13 | `deriv_e33comp33_GBPUSD_15m_result.json` |
| e34online | e34online: all (66), nseed 1 | 15m | 0.03 | .5042 (.5010) | .4725 · .000 · .4980 | 0/0 · — | .4688 · .000 · .4898 | 0/0 · — | .4908 / .4988 | KILLED / KILLED | no (15/15, 0 clear) / no (15/15, 0 clear) | 0.00 / 0.00 | `deriv_e34online_GBPUSD_15m_result.json` |
| e6pool | e6pool: own (41), nseed 1, pooled +6 pairs | 15m | 0.03 | .4957 (.4822) | .4160 · .267 · .4839 | 1/4 · .5305 | .4279 · .200 · .4864 | 2/4 · .5241 | .4908 / .4988 | KILLED / KILLED | no (15/15, 4 clear) / no (15/15, 3 clear) | 0.13 / 0.07 | `deriv_e6pool_GBPUSD_15m_result.json` |
| e7fitall | e7fitall: all (66), nseed 1 | 15m | 0.03 | .4960 (.4861) | .4349 · .267 · .4999 | 1/2 · .5241 | .3909 · .133 · .4495 | 0/4 · .4039 | .4908 / .4988 | KILLED / KILLED | no (15/15, 4 clear) / no (15/15, 2 clear) | 0.20 / 0.13 | `deriv_e7fitall_GBPUSD_15m_result.json` |
<!-- DERIV-FEED-TABLES:END -->
