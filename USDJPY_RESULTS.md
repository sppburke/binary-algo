> **SCOPE: USDJPY** (key-specific — all timeframes × sides; results of record + UP/DOWN leaderboard). Generic methods: METHODS_CATALOG.md. Sweep menu: SWEEP_MATRIX.md. See REPO_MAP.md. Bootstrapped 2026-06-04 (new currency; structure copied from EURUSD_RESULTS.md).

# USDJPY — Results Ledger (unique key: currency × timeframe × side)

**Pair tag: `USDJPY`.** USDJPY is a **USD-BASE** pair (USDJPY up ⇒ USD up ⇒ JPY down). Carry/risk-on-off proxy; 2022–2024 BoJ-driven uptrend + MoF intervention spikes → distinct microstructure vs EURUSD.

**The unique result key is `(currency, timeframe, side)`, side ∈ {UP, DOWN}.** Where a key is unmeasured it is `UNTESTED`. COMBINED is context only, never an UP/DOWN number.

**Conventions.** Deriv-faithful settlement (mid-to-mid, ties LOSE), breakeven **0.541**. Splits: bars train 2012-21 / val 2022-23 / test 2024 / test 2025 / oos 2026. Selection on VAL worst-half (never VAL-acc-max). Moved-bars-only; per-year CI95. **No USDJPY 1s tick cache** → 1m is BAR-based (entry≈close[t], exit≈close[t+1]; mild optimistic proxy of a tick-tradeable 60s binary). Deriv forex Rise/Fall minimum expiry = 15m → **1m is a research/synthetic-index horizon** (not deriv-forex-deployable; deployable on synthetic-index 1m or as a research result). Current scope: **1m only**. Last updated 2026-06-04 (bootstrap + baseline).

---

## MASTER KEY TABLE — one row per (USDJPY, timeframe, side)

| Key (currency · timeframe · side) | **Best OOS % (2026)** | Model id · content_id | Description | Status |
|---|---|---|---|---|
| USDJPY · **1m** · UP | **~0.534–0.545** ⚠ **sub-breakeven, honestly EXHAUSTED on-disk** | best-available: s6/l255 LGBM up-preds (`usdjpy_1m_base.py 6 255`) | Faint MONOTONE-in-confidence dip-buy UP edge: cov2% .547/.534/.538, cov1% .538/.540/.545; OOS-persistent but **worst-year CI-lo never clears 0.541**. EXHAUSTED under full (a)-(e): more-data/cross-pair/regime-gate/signed-feats(N1/N2)/OF-residual(N3)/seed-ens/gotobi(N5) all sub-BE; discovery 2 rounds (R2 DRY). The ~1pp gap is informational (efficiency), not model/coverage. NOT certified, NOT deployable. | **MEASURED, sub-breakeven, EXHAUSTED on-disk** |
| USDJPY · **1m** · DOWN | **~0.51** ❌ **dead + honestly EXHAUSTED** | s6/l255 LGBM down-preds | No monotone lift; ≤.51 all covers; every gated/regime/specialist attempt OOS-collapses (<.50 in 2026: regime .495, OF-resid .480, gotobi .481). Sell-rally reversion has no OOS-stable edge. Same (a)-(e) + Tier-N exhaustion as UP. | **MEASURED, dead, EXHAUSTED** |
| USDJPY · **2m** · UP | **CPCV mean .546 / p10 .531** ⚠ **REAL-but-sub-BE (not certified)** | best-available: cross-pair POOLED+seed-ens GBM (`usdjpy_2m_xpair.py pool` + `usdjpy_2m_cpcv.py … 5`) | Cross-pair pooling LIFTS the edge above the 1m near-efficiency floor (CPCV mean .546 vs 1m ~.52) — genuine signal gain, independently confirmed (ARF .508>.50). But seed-ens+2×data lift the mean over BE while p10 SATURATES ~.531 (~1pp sub-BE, worst regime signal-bound). Base/compression-regime/DL/loss all sub-BE-or-subsumed. Frontier = external (tick/EURJPY/rate-diff). | **MEASURED real-but-sub-BE, on-disk EXHAUSTED** |
| USDJPY · **2m** · DOWN | **CPCV mean .546 / p10 .531** ⚠ **REAL-but-sub-BE (not certified)** | best-available: cross-pair POOLED+seed-ens GBM down-preds | Pooling lifts DOWN from 1m-dead (~.51) to a real thin edge (CPCV mean .546, p10 .531) — a genuine gain — but still ~1pp sub-BE; symmetric with UP. Subset specialists destroy ranking; regime-fade dead. Frontier = external data. | **MEASURED real-but-sub-BE, on-disk EXHAUSTED** |
| USDJPY · 5m/10m/15m/30m · UP/DOWN | `UNTESTED` | — | Out of current scope (goal = 2m). Bar data present; bootstrap when scoped. | UNTESTED |

Provenance: `usdjpy_1m_base_result.json`, `usdjpy_2m_base_result.json`, `usdjpy_2m_base_s6_l255_result.json` (Tier-1).

**Magnitude** (|ret|≥Q) is sign-invariant → no up/down key; tracked in `MAGNITUDE_FINDINGS.md` (USDJPY pending).

---

# Per-timeframe detail

## USDJPY × 1m (60-second)
Breakeven 0.541. **Prior: 1m direction near-efficient** (EURUSD 60s ~0.50–0.51 AUC, 24 channels null; cross-pair gradient none@60s). Measured directly for USDJPY below — faintly above EURUSD's floor, with a UP-only tilt.

### Combined-book / single-pair experiments
| # | Method (file) | Result (2024 / 2025 / 2026 moved-AUC; gate cov2% COMB wr) | Verdict |
|---|---|---|---|
| BASE | single-pair LGBM, 239 base feats, 1m own-clock label (`usdjpy_1m_base.py`) | AUC 0.5173 / 0.5142 / 0.5157; COMB wr 0.5218 / 0.5192 / 0.5154 (CI incl. <0.541) | ❌ KILLED as standalone tradeable edge (no year COMBINED CI-lo clears 0.541); but UP-tilt + monotone confidence → redirect, not wall. `usdjpy_1m_base_result.json` |
| A6a | cross-pair USD-residual+OF, USDJPY-target, xpof @MX_HOR=1 (`usdjpy_xpair.py`) | AUC 0.5196 / 0.5164 / 0.5188; cov2% COMB .532 / .526 / .512 | ❌ KILLED — raises AUC + mid-cov win-rate in 2024/25 (top feats: ll_GBPUSD/AUDUSD lead-lag, own_r1, hour, OF_kyle) but **DILUTES the OOS tight tail** (cov1% UP .513 vs baseline .545); the cross-pair lead-lag is regime-dependent (2024-25 only, not 2026). `usdjpy_1m_xpair_xpof_result.json` |
| A1a | more data + capacity: stride6 (592k train) × 255 leaves (`usdjpy_1m_base.py 6 255`) | AUC 0.5234; UP cov2% .547 / .534 / .538 (worst .534) | ⚠ IMPROVES (signal was DATA-STARVED; +~1pt on UP tail) but still sub-breakeven on the worst year. New best base model for downstream gating/improve. `usdjpy_1m_base_s6_l255_result.json` |
| UJ-N1/N2 | + 16 SIGNED reversion/path-state feats (spike×rangepos, signed cumret, run-length, dist-from-extreme) on s6/l255 (`usdjpy_1m_revspec.py`) | VAL AUC 0.5235 (UNCHANGED); UP cov2% .556/.523/.527; DOWN cov2% .518/.530/.503 | ❌ SUBSUMED — new feats rank #38-90 of 254, VAL AUC unmoved; 2024 UP up but 2025/26 DOWN vs base, worst year lower; DOWN OOS .503. Base 239 already capture signed reversion. `usdjpy_1m_revspec_s6_result.json` |
| UJ-N3 | cross-pair OF-basket + RESIDUAL (own signed OF − USD-up-basket OF, the untested cross-pair delta) (`usdjpy_1m_ofresid.py`) | VAL AUC 0.5219; UP cov2% .538/.528/.519; DOWN cov2% .533/.518/.480 | ❌ KILLED — OF feats rank #15-25 but no net VAL gain; UP worst OOS .519 < base, DOWN OOS collapses .480. Cross-pair OF channel as dead as the return channel for OOS direction. `usdjpy_1m_ofresid_s8_result.json` |
| I2 (Tier-I) | seed-ensemble K=5 of s6/l255 base (`usdjpy_1m_improve.py seedens`) | VAL AUC 0.5238; UP cov2% .551/.532/.536 (worst .532); DOWN cov2% .524/.542/.494 | ❌ no lift — variance reduction can't close the ~1pp worst-year gap; confirms the wall is signal-level, not variance. `usdjpy_1m_improve_seedens_result.json` |
| A8b/N5 | gotobi/Tokyo-fix calendar conditioning (`usdjpy_1m_gotobi.py`); pure UP/DOWN specialists subset-trained (`usdjpy_1m_improve.py spec`) | gotobi +.0004 AUC (null); UP-spec cov2% .531/.528/.524 (< filter), DOWN-spec .515/.519/.517 | ❌ gotobi KILLED; specialists KILLED — subset-training destroys ranking (worse than the symmetric filter; EURUSD finding reproduced). `usdjpy_1m_gotobi_s6_result.json`, `usdjpy_1m_improve_spec_result.json` |

### Coverage curve (step d) — `usdjpy_1m_base.py` covcurve
UP-side win-rate is monotone in model confidence and OOS-persistent (the deliverable UP signal):
| cov | UP wr 2024 / 2025 / 2026 | DOWN wr 2024 / 2025 / 2026 |
|---|---|---|
| 0.30 | .513 / .510 / .515 | .501 / .504 / .497 |
| 0.10 | .518 / .515 / .518 | .512 / .514 / .505 |
| 0.05 | .522 / .521 / .518 | .515 / .514 / .506 |
| 0.02 | .536 / .529 / .524 | .511 / .512 / .506 |
| 0.01 | .538 / .540 / **.545** | .504 / .500 / .508 |

DOWN: flat/declining → **dead**. UP: monotone↑, OOS-best at cov1% (.545) but thin-slice CI-lo (≈.507 @ n651) does not clear 0.541.

### Regime-gated filter (steps a+b+c+d) — `usdjpy_1m_regime.py` → `usdjpy_1m_regime_s6_result.json`
Symmetric s6/l255 GBM + dip×compression×Tokyo (UP) / rally (DOWN) regime filter + model-confidence, gate FROZEN on VAL worst-half. **Both KILLED.** UP gate `dip_comp_tk` looked strong on VAL (worst-half .558) but held-out UP = .524 / .529 / **.495** — **OOS collapses below 0.50**. The model's confidence ranking *within* the regime ANTI-TRANSFERS (corr(VAL,OOS)=−0.54); the only OOS-stable thing is the raw regime base-rate (~.52, sub-breakeven). DOWN rally-gate .518/.515/.513 — consistent thin sell-rally reversion, well below breakeven. **Lesson: gating + model-confidence does NOT lift USDJPY 1m direction over breakeven; the signed edge is a raw ~.52 reversion, not model-extractable to a tradeable level.**

### Key results (MEASURED — baseline)
| Key | Result (2024 / 2025 / 2026) | Method | Status |
|---|---|---|---|
| **(USDJPY, 1m, UP)** | .533 / .531 / .518 @cov2%; **.538 / .540 / .545 @cov1%** | up-preds of single-pair LGBM (`usdjpy_1m_base.py`) | ⚠ sub-breakeven; faint monotone-confidence edge, OOS-persistent; **best AVAILABLE, not certified.** Improve levers (cross-pair, more data, gate, mag-bridge) pending |
| **(USDJPY, 1m, DOWN)** | .511 / .511 / .513 @cov2%; ≤.508 @cov1% | down-preds of single-pair LGBM | ❌ dead — no monotone lift; confirming via specialist (c) + discovery before declaring exhausted |

**Mechanism (v1 model of the edge):** USDJPY 1m direction is near-efficient; the only signed structure is a faint UP-autocorrelation/dip-buy tilt (UP carries, DOWN dead — same asymmetry as EURUSD 60s, but UP fades slightly into 2026 rather than strengthening). Base GBM AUC 0.52 caps it below breakeven. The "below-breakeven" cause = weak base signal → attack via: stronger UP channel (cross-pair USD-factor — USDJPY has DIRECT USD exposure + on-disk OF), more training data / tuning, regime gate, and magnitude→direction bridge. Citations: `usdjpy_1m_base_result.json`.

---

## BEST UP / DOWN ACCURACY (2026-06-05) — the headline numbers
Per held-out year **2024 / 2025 / 2026-OOS**, moved-bars-only, deriv-faithful (ties LOSE), gap=60s. **Breakeven = 0.541. None CPCV-certified.** Numbers derived by scanning all 16 `usdjpy_1m_*_result.json` coverage curves (n≥150 floor).

| Side | Honest single fixed model+gate (deployable form) | Most-flattering per-year slice (best model×cov each yr — selection-inflated) |
|---|---|---|
| **UP** | baseline @cov1% **.538 / .540 / .545**; deploy-spec s6/l255 @cov2% **.547 / .534 / .538** | **.567 / .540 / .545** (2024 seed-ens cov1%) |
| **DOWN — dead** | best single fixed model OOS only **~.515** (seed-ens @cov10%) | .547 / .547 / **.515** — the 2024/25 ~.55 is cherry-picked and **collapses to .515 OOS** |

- **UP ≈ 53.5–54.5%** at tradeable coverage — real + OOS-persistent but **at/just-below the 0.541 breakeven, uncertified** (worst-year CI-lo never clears).
- **DOWN ≈ 51–52% OOS — dead** (the higher 2024/25 numbers don't survive to 2026: regime .495 / OF-resid .480 / magweight .486 all collapse).
- Contrast — **magnitude** (move-size, sign-invariant, NOT direction): AUC **0.72–0.79** (`usdjpy_1m_magnitude_result.json`). The online-ARF keystone (adaptive model) sits at AUC **~.50** every year → direction is genuinely near-efficient.

## DEPLOYMENT SPEC / FINAL CONCLUSION — USDJPY 1m, both sides (2026-06-04; improve+discover loops dry; 2nd-pass exhaustiveness audit 2026-06-05)

**Verdict: USDJPY 1-minute direction is NEAR-EFFICIENT. No certified, deployable UP or DOWN edge exists on the available (on-disk) data. Both sides honestly EXHAUSTED.**

**What the edge IS:** a thin, OOS-stable **dip-buy mean-reversion** (after a 5-min dip, next-1min UP-rate ~.515 vs after-rally ~.495; amplified by compression + Tokyo session ~.52), fully captured by the base GBM (dirAUC ~.514–.524 every year). It is structurally **~1pp below the 0.541 deriv breakeven** at the best tradeable coverage and cannot be lifted over it.

**Why it can't be lifted (the wall is SIGNAL-LEVEL / informational, not a modeling deficiency)** — the full (a)–(e) pipeline + Tier-N + Tier-I, all Tier-1:
- (a) side-split: UP carries faintly, DOWN dead — every model.
- (b) VAL worst-half gate: best UP worst-year .532–.534 < .541.
- (c) specialist: regime-gated filter OOS-COLLAPSES (.495); pure subset-trained specialists worse (control `usdjpy_1m_improve.py spec`).
- (d) coverage curve: UP monotone-in-confidence but plateaus ~.54 at cov1%, thin-n CI-lo < .541; DOWN flat/declining.
- (e) full-refit CPCV: **never triggered** — no model produced a worst-year UP/DOWN CI-lo ≥ 0.541 to certify.
- Capacity/data: more data + 255 leaves lifts the tail ~+1pp but still < BE (signal was data-starved, not the wall).
- Cross-pair (returns + OF residual): KILLED — USD-common-factor carries no next-60s sign (the program's `none@60s` gradient holds for USDJPY too); OOS DOWN collapses .477/.480.
- Signed reversion/path-state feats (N1/N2): SUBSUMED (base 239 already encode them).
- Seed-ensemble (Tier-I I2): no lift (variance reduction can't close the gap). ACI/calibration/|ret|-weight: moot/subsumed.
- gotobi/Tokyo-fix calendar (N5): null (+.0004 AUC). Magnitude→direction bridge: sign-invariant (null).
- Discovery: round-1 5 candidates all killed; round-2 DRY (2 independent agents + their own probes). Loop dry.

**The real USDJPY 1m edge is MAGNITUDE** (sign-invariant): magAUC OOS 0.72–0.79, decile |ret| lift ~2.1–2.5× (`usdjpy_1m_magnitude_result.json`, recorded in MAGNITUDE_FINDINGS.md). Not a direction deliverable.

**Best-AVAILABLE (uncertified, NOT deployable) per side** — for the record only:
- UP: `usdjpy_1m_base.py 6 255` up-preds, cov2% .547/.534/.538 (reproduce: `~/binary-algo-venv/bin/python usdjpy_1m_base.py 6 255`). No book frozen (sub-breakeven, not a deliverable; 1m < deriv 15m forex minimum anyway → research/synthetic-index horizon only).
- DOWN: dead — no best-available above ~.51.

**Only frontier = EXTERNAL DATA (Tier-G, gated on user "go"):** intraday US–JP 2y/10y rate-differential (carry sign), intraday Nikkei/risk-asset lead (Tokyo hours), USDJPY 25-delta risk-reversal skew (DOWN-enabler), traded JPY crosses (EURJPY/GBPJPY triangular). None on disk; modeling rows cannot manufacture them. See `sweeps/USDJPY_1m_backlog.md` DATA-BLOCKED.

### 2nd-pass exhaustiveness audit (2026-06-05)
After the first pass leaned on subsume-by-analogy-to-EURUSD, a second pass RAN the previously-argued levers (coverage rule: run each family once even if null elsewhere). All confirm the wall:
- **★ C5 online-ARF keystone** (`usdjpy_1m_statespace.py`): an ADAPTIVE continuously-retraining forest = AUC **.502/.501/.506** every year incl. OOS → the wall is **genuine efficiency**, not a stationary-GBM limitation (the GBM's .52 is marginally *above* it). This is the decisive confirmator.
- **C2 Kalman forward-filter** drift-sign AUC .488/.489/.489 — anti-predictive (trend dead → dip-buy reversion confirmed).
- **I3 |ret|-weighted retrain** (POW .5/1.0): DOWN OOS .486/.509 — the EURUSD-5m DOWN-rescue does NOT transfer.
- **I1 ACI adaptive gate** (EURUSD-5m winner): UP .534/.519/.513, DOWN .520/.505/.493 — base-rate trading, no edge.
- **D1 GRU** (Tier-D): AUC .515 all years, win-rates sub-BE — info-bound caps DL (confirms EURUSD D1–D6).
- **A5 cross-horizon**: USDJPY H=15m UP 2024 .593 (clears!) but 2025/OOS sub-BE → no robust parent to front-load (hook for a future 15m goal).
- **I6 Optuna** (10-trial TPE, worst-VAL-half deploy objective): best worst-VAL-half WR **.5347 < .541**; held-out UP cov2% .548/.528/.527, DOWN sub-BE — tuning can't cross breakeven (signal-bound, as the keystone predicts). `usdjpy_1m_optuna_result.json`.

Remaining families subsumed on Tier-1 grounds (run-don't-argue exception — the dominating variant ran + failed): **C1 HMM / C3 RMT / C4 CCM** (state/complexity gates = sign-invariant by theorem; the adaptive-state-space question is answered by the ARF keystone @.50, the directional state-space by Kalman @anti-predictive); **D2 NCDE / D3 TabNet / D4-D5 DRL** (info-bound caps all DL, established by D1 GRU + EURUSD D1–D6); **E2 dir-cond-on-mag** (magnitude is sign-invariant: E1 strong + dir flat; EURUSD `min1_magdir` null); **E3 complexity / E4 info-bars** (sign-invariant by theorem); **I5 cross-pair pooling** (the cross-pair channel carries no 1m sign — A6a returns + N3 OF both dead); **cks1s tick model** (signed-event = OFI family, dead: `min1_xofi` .5015 + N3 OF-residual .480). **Net: every on-disk method family has now been run-or-subsumed-with-a-Tier-1-cite; the verdict is unchanged and better-supported.**

---

## USDJPY × 2m (120-second)
Breakeven 0.541. **Prior: 2m is the transition horizon** (EURUSD 1m~.50 → 2m UP .555 robust, but EURUSD's 2m edge is TICK-based). USDJPY is BAR-only (no tick cache). Sweep started 2026-06-05; ledger `sweeps/USDJPY_2m.md`, backlog `sweeps/USDJPY_2m_backlog.md`.

### Combined-book / single-pair + cross-pair experiments (combined-context; per-side below)
| # | Method (file) | Result (2024 / 2025 / 2026 moved-AUC; gate UP wr) | Verdict |
|---|---|---|---|
| A1 | single-pair LGBM s24/l127 (`usdjpy_2m_base.py`) | AUC .519/.515/.518; UP cov2% .546/.525/.527 | ❌ KILLED standalone (no yr COMB CI-lo≥.541). `usdjpy_2m_base_result.json` |
| A2 | single-pair LGBM s6/l255 (`usdjpy_2m_base.py 6 255`) | AUC .521/.517/.518; UP cov2% .546/.524/.536 (worst 2025 .524) | ❌ KILLED — more data doesn't cross BE (same all-bars wall as 1m). `usdjpy_2m_base_s6_l255_result.json` |
| D1 | compression-release regime FILTER (`usdjpy_2m_regime.py`) | UP dip×comp gate .548/.521/.542; DOWN .519/.513/.497 | ❌ KILLED both — binding 2025 UP .521; EURUSD-2m mechanism (TICK-based) doesn't transfer to bar-only USDJPY. `usdjpy_2m_regime_s8_result.json` |
| **C1** | **cross-pair POOLED GBM 7-major s42/l255 (`usdjpy_2m_xpair.py pool 42 255`)** | **UP frozen-gate cov2% .547/.546/.570; covcurve cov3% UP .549/.543/.555** | ⚠ **best frozen edge — point-above-BE all 3 yrs incl OOS, but CI-lo n-limited ~.527. POOLING LIFTS binding 2025 (.524→.546).** Numbers in `usdjpy_2m_xpair_pool.log` (Tier-1) |
| C1b | cross-pair POOLED 2x data s21/l255 | UP cov2% .542/.540/.527; COMB .544/.542/.526 | ❌ KILLED on CI-lo; lifts 2024/25 but OOS 2026 drops .527. Thin/tail-sensitive. `usdjpy_2m_xpair_pool_s21_l255_result.json` |
| **G1** | **★ full per-fold-REFIT CPCV of pooled (`usdjpy_2m_cpcv.py`)** | **UP p10 .5245 (mean .536, frac_clear .33); DOWN p10 .5195; COMB p10 .527** | ❌ **NOT CERTIFIED — CORRECTS C1 overstatement (trust refit). Recency: recent regime UP .532/22%-clear vs older .542/50%. Edge REAL (mean .536 > 1m .52) but sub-BE in deployable regime.** `usdjpy_2m_cpcv_result.json` |

### Key results (MEASURED — cross-pair pooled best edge; CPCV-honest)
| Key | Result (CPCV p10 / mean @cov3%) | Method | Status |
|---|---|---|---|
| **(USDJPY, 2m, UP)** | **p10 .5245 / mean .536** (frozen-gate cov2% .547/.546/.570 = overstatement) | up-preds of cross-pair POOLED GBM (`usdjpy_2m_xpair.py pool`) | ⚠ real-but-sub-BE; pooling lifts above 1m level but CPCV p10 < .541; **best AVAILABLE, NOT certified.** Discovery+Tier-I+external open |
| **(USDJPY, 2m, DOWN)** | **p10 .5195 / mean .536** | down-preds of cross-pair POOLED GBM | ⚠ real-but-sub-BE (lifted from 1m-dead by pooling); CPCV p10 < .541. Purpose-built DOWN specialist + discovery pending |

**The 2m story so far:** unlike 1m (near-efficient, dip-buy ~1pp sub-BE), **2m has a REAL cross-pair-pooled direction edge** — pooling 7 majors' base features lifts the CPCV mean to ~.536 (both sides), a genuine gain over the 1m ~.52 floor. But it does **not certify** (refit-CPCV p10 ~.524, ~1.5pp sub-BE; recent regime weaker). The signal is on the right side of the horizon gradient (none@60s → emerging@2m) but **the on-disk bar features fall ~1.5pp short of the deriv breakeven**. Improve (Tier-I) + discovery + external-data (US–JP rate-diff) frontiers remain open before an exhaustion verdict.

### DEPLOYMENT SPEC / FINAL CONCLUSION — USDJPY 2m, both sides (2026-06-05; improve+discover loops dry on-disk)

**Verdict: USDJPY 2-minute direction is a REAL but SUB-BREAKEVEN reversion edge on the available (on-disk, bar-only) data. NOT certified, NOT deployable on deriv. Both sides honestly EXHAUSTED on-disk; the only remaining frontier is EXTERNAL data (esp. USDJPY tick microstructure).**

**What the edge IS:** a thin mean-reversion / cross-pair-decorrelated tilt that the 120s horizon makes *measurably real* (vs the 1m near-efficiency floor). Three independent methods agree on a thin genuine signal:
- **Cross-pair POOLED GBM** (train all 7 USD-majors' 239 base feats → each pair's own 2m sign; eval USDJPY): the best edge. Per-fold-refit CPCV (15 purged paths, ties-strict, all-pair purge): **win-rate mean .546, p50 .545, p10 .531** (both UP and DOWN), frac-clear-BE .60. Pooling = noise-decorrelation (lifts the *mean* over BE), NOT a cross-pair factor signal (the cross-pair SIGN gradient none@60s→UP@5m does not reach 2m).
- **Online-ARF keystone** (adaptive, retrains continuously): AUC **.5068/.5085/.5093** — *above* the 1m .50 efficiency floor → confirms genuine thin 2m signal (not pure efficiency, unlike 1m).
- **Kalman forward-filter drift**: AUC .488 (anti-predictive) → the edge is REVERSION, not trend.

**Why it can't be lifted over 0.541 (the wall is SIGNAL-LEVEL at the worst regime, not modeling)** — full (a)–(e) + Tier-I + discovery, all Tier-1:
- (a) side-split: UP and DOWN both CPCV p10 .531 — symmetric, both sub-BE.
- (b) VAL worst-half gate: built into base + CPCV VAL-tuned threshold; doesn't cross.
- (c) specialist: compression-release regime FILTER (D1) sub-BE (binding 2025 UP .521); subset-trained specialists destroy ranking (established).
- (d) coverage curve: monotone-in-confidence but UP/DOWN plateau ~.55 at cov2% with CI-lo ~.527, n-limited.
- (e) full-refit CPCV: **ran** — UP p10 .5245, DOWN p10 .5195 (cov3%); NOT certified (need p10≥.541 & ≥80% paths clear).
- **Tier-I improve (the decisive test):** seed-ensemble (K=3) lifts UP p10 .5245→.5306 (frac .33→.53); +2× data +K=5 lifts mean to .546 but **p10 SATURATES ~.531** — the worst regime-combination paths are SIGNAL-bound; variance reduction cannot close the residual ~1pp. Diminishing returns prove an information bound on bar data.
- **Deep exhaustive 2nd-pass (2026-06-05) — RAN the full Tier-I cross-product + combinations by experiment (not subsumed):** ESN reservoir (VAL .5228, held-out worse), GMADL/|ret|-loss (lowers AUC), pooled×compression-gate combine (UP .465-.520, n-starved), cov-grid (cov3% best, no lift), magnitude→direction bridge (high-mag bars not more sign-predictable — theorem holds), **multi-algorithm lgb+xgb+cat CPCV (UP p10 .519 < seed-ens .531 — decorrelation HURTS)**, **Optuna worst-VAL-half (VAL .550 but OOS WORSE — the corr(VAL,OOS)=−.54 anti-transfer)**, recency-weighted training (monotonically worse). **ALL killed/no-lift; none crossed 0.541. The on-disk p10 ≈ .531 ceiling is experimentally established, not argued.**
- **Discovery R1 (corpus mining, 67 levers):** factor/SDF/OFI/cross-pair-sign levers DRY (sign doesn't reach 2m; pooling already = decorrelation); sequence/ESN subsumed by GRU (AUC .514 < GBM .524, info-bound); OF-router/depth-OFI subsumed (OF dead at 1m); conformal/calibration/loss subsumed (sign-invariant / no-signal gates).
- **The frozen-gate OVERSTATEMENT, corrected:** the C1 frozen-gate per-year UP .547/.546/.570 looked point-above-BE all 3 years — but that was thin-n SELECTION; the honest per-fold-refit CPCV (p10 .531) is the number of record. Trust the refit.

**Deliverables (best-available, NOT certified):**
- **(USDJPY, 2m, UP)** = up-preds of the cross-pair POOLED + seed-ensemble GBM. CPCV win-rate mean **.546** / p10 **.531**. Real edge, ~1pp sub-BE at worst regime. Reproduce: `usdjpy_2m_xpair.py pool 42 255` (model), `usdjpy_2m_cpcv.py 14 0.03 5` (honest cert test).
- **(USDJPY, 2m, DOWN)** = down-preds of the same pooled model. CPCV mean **.546** / p10 **.531**. Pooling lifts DOWN from 1m-dead (~.51) to a real-but-sub-BE thin edge — a genuine gain, still not tradeable.

**Frontier (the only path over breakeven) — EXTERNAL DATA, in priority order:**
1. **USDJPY 1-second TICK microstructure** (imbalance, micro-price, spread, trade-size) — *the* EURUSD-2m edge driver (EURUSD.min2.v1 UP .555 is tick-based); USDJPY has no tick cache. This is the structurally-right unlock and the single highest-value acquisition.
2. **Triangular EURJPY** (USDJPY = EURJPY/EURUSD dislocation residual) — needs EURJPY 1m bars (not on disk; only 7 USD-majors present). Sign-carrying.
3. **Intraday US–JP 2y rate differential** (carry driver) + **JPY 25-delta risk-reversal** (DOWN-enabler) — slower drift, helps longer horizons more than 2m but structurally USDJPY-specific.

**Magnitude (sign-invariant, separate deliverable):** USDJPY 2m magnitude is STRONG — magAUC Q90 **.827/.772/.785**, Q75 .780/.718/.722, decile lift ~2.1–2.5× (`usdjpy_2m_magnitude_result.json`). Recorded in MAGNITUDE_FINDINGS.md. The sign-invariance signature holds (magAUC ~.78 vs dirAUC ~.52).

---

## How to maintain this file
- One row per key; never copy a COMBINED number into an UP/DOWN key; never copy one timeframe/side's number to another.
- Every number traces to an on-disk result JSON (Tier-1). Flag n<50 + VAL-acc-max as non-robust.
- Unseat a side-leader only if the challenger beats the incumbent's binding (worst held-out) year with CI95-lower clearing it under the discipline, ideally refit-CPCV-certified.
- Update the master table + leaderboard + `sweeps/USDJPY_1m.md` ledger + `sweeps/USDJPY_1m_backlog.md` together; commit often.
