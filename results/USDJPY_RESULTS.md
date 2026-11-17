> **SCOPE: USDJPY** (key-specific — all timeframes × sides; results of record + UP/DOWN leaderboard). Generic methods: docs/METHODS_CATALOG.md. Sweep menu: SWEEP_MATRIX.md. See REPO_MAP.md. Bootstrapped 2026-06-04 (new currency; structure copied from EURUSD_RESULTS.md).

# USDJPY — Results Ledger (unique key: currency × timeframe × side)

**Pair tag: `USDJPY`.** USDJPY is a **USD-BASE** pair (USDJPY up ⇒ USD up ⇒ JPY down). Carry/risk-on-off proxy; 2022–2024 BoJ-driven uptrend + MoF intervention spikes → distinct microstructure vs EURUSD.

**The unique result key is `(currency, timeframe, side)`, side ∈ {UP, DOWN}.** Where a key is unmeasured it is `UNTESTED`. COMBINED is context only, never an UP/DOWN number.

**Conventions.** Deriv-faithful settlement (mid-to-mid, ties LOSE), breakeven **0.541**. Splits: bars train 2012-21 / val 2022-23 / test 2024 / test 2025 / oos 2026. Selection on VAL worst-half (never VAL-acc-max). Moved-bars-only; per-year CI95. **No USDJPY 1s tick cache** → 1m is BAR-based (entry≈close[t], exit≈close[t+1]; mild optimistic proxy of a tick-tradeable 60s binary). Deriv forex Rise/Fall minimum expiry = 15m → **1m is a research/synthetic-index horizon** (not deriv-forex-deployable; deployable on synthetic-index 1m or as a research result). Current scope: **1m only**. Last updated 2026-06-04 (bootstrap + baseline).

---

## MASTER KEY TABLE — one row per (USDJPY, timeframe, side)

> **Deriv-feed program (2026-11-15):** the numbers in this table are Dukascopy-era evidence. On the Deriv feed (DEV 2025-08-11..2026-07-31) none of this pair's frozen 15m/30m books is SUPPORTED as deployed — see "USDJPY — DERIV-FEED PROGRAM" at the end of this file.

| Key (currency · timeframe · side) | **Best OOS % (2026)** | Model id · content_id | Description | Status |
|---|---|---|---|---|
| USDJPY · **1m** · UP | **~0.534–0.545** ⚠ **sub-breakeven, honestly EXHAUSTED on-disk** | best-available: s6/l255 LGBM up-preds (`usdjpy_1m_base.py 6 255`) | Faint MONOTONE-in-confidence dip-buy UP edge: cov2% .547/.534/.538, cov1% .538/.540/.545; OOS-persistent but **worst-year CI-lo never clears 0.541**. EXHAUSTED under full (a)-(e): more-data/cross-pair/regime-gate/signed-feats(N1/N2)/OF-residual(N3)/seed-ens/gotobi(N5) all sub-BE; discovery 2 rounds (R2 DRY). The ~1pp gap is informational (efficiency), not model/coverage. NOT certified, NOT deployable. | **MEASURED, sub-breakeven, EXHAUSTED on-disk** |
| USDJPY · **1m** · DOWN | **~0.51** ❌ **dead + honestly EXHAUSTED** | s6/l255 LGBM down-preds | No monotone lift; ≤.51 all covers; every gated/regime/specialist attempt OOS-collapses (<.50 in 2026: regime .495, OF-resid .480, gotobi .481). Sell-rally reversion has no OOS-stable edge. Same (a)-(e) + Tier-N exhaustion as UP. | **MEASURED, dead, EXHAUSTED** |
| USDJPY · **2m** · UP | **CPCV mean .546 / p10 .531** ⚠ **REAL-but-sub-BE (not certified)** | best-available: cross-pair POOLED+seed-ens GBM (`usdjpy_2m_xpair.py pool` + `usdjpy_2m_cpcv.py … 5`) | Cross-pair pooling LIFTS the edge above the 1m near-efficiency floor (CPCV mean .546 vs 1m ~.52) — genuine signal gain, independently confirmed (ARF .508>.50). But seed-ens+2×data lift the mean over BE while p10 SATURATES ~.531 (~1pp sub-BE, worst regime signal-bound). Base/compression-regime/DL/loss all sub-BE-or-subsumed. Frontier = external (tick/EURJPY/rate-diff). | **MEASURED real-but-sub-BE, on-disk EXHAUSTED** |
| USDJPY · **2m** · DOWN | **CPCV mean .546 / p10 .531** ⚠ **REAL-but-sub-BE (not certified)** | best-available: cross-pair POOLED+seed-ens GBM down-preds | Pooling lifts DOWN from 1m-dead (~.51) to a real thin edge (CPCV mean .546, p10 .531) — a genuine gain — but still ~1pp sub-BE; symmetric with UP. Subset specialists destroy ranking; regime-fade dead. Frontier = external data. | **MEASURED real-but-sub-BE, on-disk EXHAUSTED** |
| USDJPY · **15m** · UP | **NY seed-ens(K=3) refit-CPCV p10 .6005 @cov2% / .593 @cov3% (15/15)** ✅ **CERTIFIED** | own-pair LGBM base, **NY-session**, **seed-ensemble K=3**, cov2% gate (`usdjpy_15m_cpcv_session.py ny 2 0.02 3`) | ✅ **CERTIFIED.** NY-concentrated own-pair base GBM + Tier-I seed-ensemble: UP p10 **.6005** ≥ BE @cov2%, ALL 15 paths clear (single-seed cov3% .586). NY lifts conditional AUC .526→.539; seed-ens lifts worst-path p10. Refit-dependent. | **CERTIFIED (NY seed-ens refit-CPCV) 2026-06-08** |
| USDJPY · **15m** · DOWN | **NY seed-ens(K=3) refit-CPCV p10 .5738 @cov2% / .572 @cov3% (15/15)** ✅ **CERTIFIED** | own-pair LGBM base, **NY-session**, **seed-ensemble K=3**, cov2% down-preds | ✅ **CERTIFIED in NY** — all-session DOWN sub-BE (.538, dragged by LDN/Asia); NY rescues it (USD-flow carries both sides), seed-ens lifts p10 .554→**.5738** @cov2%, all 15 paths clear. First certified USDJPY DOWN edge. | **CERTIFIED (NY seed-ens refit-CPCV) 2026-06-08** |
| USDJPY · 5m/10m/30m · UP/DOWN | `UNTESTED` | — | Out of current scope (goal = 15m). Bar data present; bootstrap when scoped. | UNTESTED |

Provenance: `usdjpy_1m_base_result.json`, `usdjpy_2m_base_result.json`, `usdjpy_2m_base_s6_l255_result.json` (Tier-1).

**Magnitude** (|ret|≥Q) is sign-invariant → no up/down key; tracked in `docs/MAGNITUDE_FINDINGS.md` (USDJPY pending).

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

**The real USDJPY 1m edge is MAGNITUDE** (sign-invariant): magAUC OOS 0.72–0.79, decile |ret| lift ~2.1–2.5× (`usdjpy_1m_magnitude_result.json`, recorded in docs/MAGNITUDE_FINDINGS.md). Not a direction deliverable.

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

**Magnitude (sign-invariant, separate deliverable):** USDJPY 2m magnitude is STRONG — magAUC Q90 **.827/.772/.785**, Q75 .780/.718/.722, decile lift ~2.1–2.5× (`usdjpy_2m_magnitude_result.json`). Recorded in docs/MAGNITUDE_FINDINGS.md. The sign-invariance signature holds (magAUC ~.78 vs dirAUC ~.52).

---

## USDJPY × 15m (900-second) — DERIV-FX-DEPLOYABLE floor; sweep OPEN 2026-06-08
Breakeven 0.541. **Prior: 15m is the program's strongest direction horizon** (EURUSD 15m cross-pair certified BOTH sides refit-CPCV UP p10 .567 / DOWN p10 .574) AND the only deriv-FX-deployable one (deriv forex min expiry = 15m). Sweep ledger `sweeps/USDJPY_15m.md`, backlog `sweeps/USDJPY_15m_backlog.md`. Goal: certify best UP + best DOWN, stretch win-rate >0.70, or honestly exhaust both sides.

### Combined-book / single-pair + cross-pair experiments (combined-context; per-side below)
| # | Method (file) | Result (2024 / 2025 / 2026 moved-AUC; gate per-side wr) | Verdict |
|---|---|---|---|
| A1 | single-pair LGBM s6/l127, 15-min own-clock label (`usdjpy_15m_base.py 6 127`) | AUC .5252/.5131/.5166; cov3% COMB .579/.546/.538, **UP .601/.566/.552**, DOWN .541/.519/.520; up-rate .514/.505/.521 (trip OK) | ⚠ SURVIVED (2024 COMB CI-lo .557≥BE). Real UP-tilted edge, pt-est>BE all yrs; CI-lo clears 2024 only. Best-available baseline. `usdjpy_15m_base_result.json` |
| A1b | single-pair LGBM s3/l255 (2× data + 2× capacity) (`usdjpy_15m_base.py 3 255`) | VAL-AUC .5313 (== s6 .5312); cov3% UP .576/.573/.529, DOWN .578/.534/.555 | ❌ no AUC lift — **signal NOT data-starved; AUC capped ~.531.** More data ROTATES which year/side leads (DOWN 2026 lifts .555, UP 2026 drops). `usdjpy_15m_base_s3_l255_result.json` |
| A6/I5 | cross-pair POOLED 7-major base-GBM s42/l255 (`usdjpy_15m_xpair.py pool 42 255`) | AUC .5252/.5126/.5131; cov2% UP .584/.540/.547, DOWN .576/.517/.485 | ❌ DOES-NOT-BEAT-BASE — pooling DILUTES the binding years (2025/26 < base), DOWN 2026 collapses .485, VAL-AUC .527<base .531. **USDJPY 15m signal is OWN-PAIR-SPECIFIC (opposite of EURUSD 15m where pooling certified).** `usdjpy_15m_xpair_pool_s42_l255_result.json` |
| **G1** | **own-pair (all-session) per-fold-REFIT CPCV — `usdjpy_15m_cpcv_base.py 4 0.03`** | **UP p10 .5582 (15/15); COMB p10 .5516 (14/15); DOWN p10 .5381 mean .555 (10/15)** | ✅ UP+COMB CERTIFIED; DOWN sub-BE (recent-uptrend folds drag it). Per-fold: failing DOWN folds all contain group 4/5 (2022+). AUC mean .526. `usdjpy_15m_cpcv_base_result.json` |
| A9 | NY/LDN/Asia session-concentrated own-pair base (`usdjpy_15m_session.py`) | NY VAL-AUC .544; cov2% COMB .599/.587/.553, UP .589/.579/.580, DOWN .663/.629/.455 | ✅ **NY SURVIVED** (COMB clears 2024+2025; UP stable ~.58 incl 2026); LDN+Asia KILLED. Direction edge NY-concentrated (EURUSD precedent). `usdjpy_15m_session_{ny,ldn,asia}_s6_l127_result.json` |
| **G1-NY** | **★★ NY-session own-pair per-fold-REFIT CPCV — `usdjpy_15m_cpcv_session.py ny 2 0.03`** | **UP p10 .586 mean .602 (15/15); DOWN p10 .572 mean .597 (15/15); COMB p10 .580 (15/15)** | ✅✅ **BOTH SIDES CERTIFIED.** NY concentration lifts conditional AUC .526→.539 and RESCUES DOWN (.538→.572, all paths clear). First two-sided certification of record. `usdjpy_15m_cpcv_session_ny_cov0.03_result.json` |
| G1-NY-SE | I2 seed-ens K=3 ⊕ G1-NY — `usdjpy_15m_cpcv_session.py ny 2 0.02 3` | UP p10 .6005 / DOWN .5738 @cov2% (15/15) | ✅ seed-ens lifts worst-path p10 (variance reduction). Frozen book `USDJPY.m15ny_seedens.v1`. `usdjpy_15m_cpcv_session_ny_seedens3_result.json` |
| TN5-CPCV | TB FIRST-TOUCH train label ⊕ G1-NY (single-seed) — `usdjpy_15m_cpcv_tbfirsttouch.py 2 0.02,0.03 1` | UP p10 .6002/.5875; DOWN .5827/.5765; COMB .6012/.5911 @cov2%/cov3% (15/15) | ✅ CERTIFIED at LEVEL. ⚠ comparison caveat: the script's INC dict benchmarked the SINGLE-MODEL base (.586/.572), not the matched seed-ens. See TN5-COMBO for the apples-to-apples verdict. `usdjpy_15m_cpcv_tbfirsttouch_ny_multicov_result.json` |
| **TN5-COMBO** | **TB first-touch ⊕ seed-ens K=3 — `usdjpy_15m_cpcv_tbfirsttouch.py 2 0.01,0.02,0.03 3`** | UP p10 .6052/.5914; DOWN .5800/.5861; COMB .5953/.5905 @cov2%/cov3% (15/15) | ⚠ **CERTIFIED at LEVEL but NOT a robust IMPROVEMENT over the seed-ens book** (adversarial verify 2026-06-08, SUSPECT). Tier-1 paired per-path vs matched K=3 base: cov2% UP +.0067(t.91)/DOWN **−.0093(t−.89, mean regresses, 7/15)**/COMB +.0026; cov3% COMB +.0093 (t2.73 on CORRELATED CPCV paths→discount). Lift within path noise + sign-inconsistent across cov. **Incumbent NOT unseated.** Book `USDJPY.m15ny_tbft_seedens.v1` kept as alternative. `usdjpy_15m_cpcv_tbfirsttouch_ny_seedens3_result.json` |
| R9 | issue #9 date-only retrospective replay: A=current v1; B=clean capped legacy-date refit; C=clean capped date-refresh refit | Apr–May 2026 standardized whole-book correctness: A .5541 (n=148, yield .000723); B .5455 (n=143, .000587); C .5777 (n=206, .001445). C UP .5743 (n=148); DOWN .5862 (n=58). | ⚪ **INCONCLUSIVE — `control_not_estimable`.** The positive point estimate is not promotable: simultaneous lower bounds are C−A yield −.001243, C−B yield −.000753, C-UP accuracy−.5 −.073627, C-DOWN accuracy−.5 −.116478. No survivor/candidate; incumbent unchanged. Pair: `results/json/m15_book_refresh_53547498c599f2877a2f6616ae725f8f27e11070e3b550e82eb26ec039fb23f6_USDJPY_replay_result.json`; joint/shadow: `results/json/m15_book_refresh_53547498c599f2877a2f6616ae725f8f27e11070e3b550e82eb26ec039fb23f6_joint_replay_result.json`, `results/json/m15_book_refresh_53547498c599f2877a2f6616ae725f8f27e11070e3b550e82eb26ec039fb23f6_shadow_spec_result.json`. |

### Key results (MEASURING — best-available, cert in progress)
| Key | Result (2024 / 2025 / 2026) | Method | Status |
|---|---|---|---|
| **(USDJPY, 15m, UP)** | **NY refit-CPCV p10 .586 / mean .602 (15/15 paths)**; all-session .558 | own-pair LGBM base, NY-session, cov3% gate (`usdjpy_15m_session.py ny` + `usdjpy_15m_cpcv_session.py ny`) | ✅ **CERTIFIED** — p10 .586 ≥ BE, all paths clear. Best USDJPY direction edge. Refit-dependent; size on p10 .586. Tighter-cov + freeze pending |
| **(USDJPY, 15m, DOWN)** | **NY refit-CPCV p10 .572 / mean .597 (15/15 paths)**; all-session sub-BE .538 | own-pair LGBM base, NY-session down-preds, cov3% gate | ✅ **CERTIFIED in NY** — p10 .572, all paths clear. NY rescues DOWN from the all-session uptrend drag. First certified USDJPY DOWN edge |

**Mechanism (v2 model of the edge — BOTH SIDES CERTIFIED in NY):** USDJPY 15m carries a real, two-sided direction edge concentrated in the **NY/US session** (USD-driven flow + carry). Own-pair GBM AUC ~.531 all-session, lifting to ~.539 in NY; the win-rate edge comes from confidence selection. Established (all Tier-1):
- **15m is where USDJPY direction has tradeable signal** — a clean break from 1m (near-efficient) and 2m (sub-BE).
- **The signal is USDJPY-SPECIFIC** — own-pair base BEATS cross-pair pooling (the inverse of EURUSD 15m, where pooling certified); USDJPY's BoJ/MoF/carry microstructure is its own factor, not the shared USD-common-factor.
- **AUC is signal-capped ~.531, data-independent** (2× data + 2× capacity gave identical VAL-AUC) — not a starvation wall.
- **The edge is NY-concentrated** — NY survives, LDN+Asia KILLED (EURUSD precedent reproduced).
- **NY rescues DOWN** — all-session DOWN is sub-BE (p10 .538) because the 2022+ BoJ/carry *uptrend* makes DOWN bets fight the drift; but restricted to NY decision rows, DOWN certifies (p10 .572, all paths clear) — the USD-flow direction signal carries both sides cleanly in NY.
- **>0.70 is not a certified reality** — AUC ~.539 caps selective win-rate ~.58–.62 at tradeable coverage; .70 needs cov<0.5% (thin-n noise). Genuine certified deliverables sit ~.57–.60.

Citations: `usdjpy_15m_base_result.json`, `usdjpy_15m_xpair_pool_s42_l255_result.json`, `usdjpy_15m_cpcv_base_result.json`, `usdjpy_15m_session_ny_s6_l127_result.json`, `usdjpy_15m_cpcv_session_ny_cov0.03_result.json` (Tier-1).

### DEPLOYMENT SPEC — USDJPY 15m, NY-session (both sides certified)
**Coverage→certified-p10 curve** (NY per-fold-refit CPCV, `usdjpy_15m_cpcv_session_ny_multicov_result.json`):

| cov | UP p10 / mean (med n/path) | DOWN p10 / mean (frac clear) | COMBINED p10 / mean |
|---|---|---|---|
| 3% | .586 / .602 (937) | **.572 / .597 (1.0)** | .580 / .599 |
| 2% | .582 / .602 (580) | .554 / .599 (1.0) | .582 / .600 |
| 1% | **.599 / .625 (273)** | .544 / .601 (.87) | .590 / .614 |
| 0.5% | .580 / .631 (130) | .559 / .625 (.93) | .599 / .624 |

- **Best UP operating point: cov1% → certified p10 .599, mean .625** (15/15 paths). **DOWN: cov3% → p10 .572** (most robust; tighter cov thins n and drops frac<1.0).
- **Deployable model:** own-pair base LGBM (239 feats, s6/l127), trade only **NY-session** decision bars (America/New_York 08–17, DST-correct), confidence gate at the chosen cov, ties LOSE, breakeven 0.541. Frozen book `USDJPY.m15ny.v1` (pending freeze).
- **UP is robust frozen-forward** (~.58 all years incl 2026); **DOWN is refit-dependent** (frozen-2012-21 DOWN decays in 2026-oos thin-slice; the refit-CPCV p10 .572 is the certified floor — deploy with periodic retrain, size on the refit floor).
- **>0.70 is NOT achievable as a certified floor** — directional AUC caps ~.539 in NY; certified win-rate tops out ~.60–.625. Beyond cov0.5% the mean rises (~.63) but p10 falls / frac<1.0 (thin-n, uncertifiable). Genuine deployable edge: **UP ~.60, DOWN ~.57**, both clearing the 0.541 deriv breakeven with margin (R≈1.85 → profitable).

**Adversarial verification (all PASS, Tier-1):**
- **Label-shuffle leakage control** (`usdjpy_15m_verify_result.json`): NY model trained on PERMUTED labels collapses to moved-AUC **.503/.504/.512** (~.50, early-stops iter 2) → the .586 edge is NOT a harness/leakage artifact.
- **★ TRUE-TICK-SETTLEMENT validation (2026-06-09, `usdjpy_15m_ticksettle_result.json`):** re-settled the certified seed-ens book's confident NY trades on the raw tick feed with deriv-faithful wc_ret (entry = first tick after bar-close+1s, exit = last tick ≤ entry+900s, mid-to-mid, ties LOSE). Tick-WR **converges to bar-close WR**: 2024 .600 vs .620, 2025 .577 vs .576, 2026 .488 vs .480; **mean Δ −0.0036 → PRESERVED.** The certified win-rate is REAL on actual tick prices (2024/25 clear BE; 2026 = known frozen-vintage decay), NOT a bar-shift artifact — and the bar-close label is faithful at 15m. (Two settler bugs found+fixed en route: broker-local-time tick-file loading; bar indexed at minute-START while close[t]=mid at minute-END → anchor entry at index+60s, else a spurious ~10pp window-misalignment gap, trap#8.)
- **Up-rate tripwire** ∈ [.47,.53] every fold (CPCV ~.509) → no fake-flat mirage.
- **Frozen-forward ≈ refit-CPCV** for UP (.58 ≈ .586) → not a refit overstatement. DOWN frozen-2026 thin-slice weaker → refit-dependent (size on refit floor).
- **Settlement**: deriv-faithful fixed-15m sign, ties LOSE, `nonoverlap_chrono` gap=900s; CPCV purge+embargo=900s; gate selected on VAL worst-half (never VAL-acc-max).
- **AUC ceiling confirmed** across ~15 orthogonal lever classes (data A1b, capacity, cross-pair pooling A6b, avg-label TN1, cointegration ECM TN4, HMM C1, Kalman C2, RMT C3, CCM C4, ARF C5, magdir E2, complexity E3, GRU D1, cross-horizon stack A5, first-touch label TN5) → the ~.539 cap is a genuine directional information bound, not a modeling deficiency.

**FINAL CONCLUSION — USDJPY 15m direction (sweep CLOSED 2026-06-08; thoroughness pass complete):** BOTH sides CERTIFIED and deployable on deriv (15m = FX floor).

**Deliverable = frozen book `USDJPY.m15ny_seedens.v1`** (tag `book/USDJPY.m15ny_seedens.v1`; best) — own-pair LGBM **seed-ensemble (K=3)**, **NY-session** decision rows, **cov2%** confidence gate. **UP** certified refit-CPCV p10 **.6005** / mean .617; **DOWN** certified p10 **.5738** / mean .616; both 15/15 paths clear. (Single-model sibling `USDJPY.m15ny.v1`: UP p10 .586 / DOWN .572 @cov3%.)

- **The edge is NY-concentrated + USDJPY-specific** — cross-pair pooling DILUTES it (inverse of EURUSD 15m); LDN+Asia KILLED.
- **REFIT-DEPENDENT (deploy caveat):** the certified p10 is a *per-fold-refit* number (USD-factor non-stationarity). A frozen single vintage (2012-21) decays — its 2026 NY-cov2% slice runs ~.48–.53 (thin n). **Deploy with periodic retrain on a trailing window; size on the refit-CPCV floor.** Same property as the EURUSD 15m/10m certified books; the frozen artifact is the reproducibility snapshot, not the literal deploy weights.
- **>0.70 is NOT achievable as a certified floor** — directional AUC caps ~.539 (confirmed across ~15 lever classes, listed above); certified win-rate tops ~.60–.625. Both sides clear breakeven 0.541 with margin (R≈1.85 → profitable).
- **Improve loop:** seed-ensemble (I2) was the ONLY robust lift (+.01–.02 p10 → the deliverable). ACI (I1) binding WR .5152→.5216 (still sub-BE, no cert). I3 |ret|-loss / I7 recency RAN at 15m NY refit-CPCV (head-to-head vs control, same folds) — null (AUC means control .5386 > ret .5366 > recency .5354; size/non-stationarity levers, sign-invariant). **I6 Optuna RAN at 15m NY (40-trial TPE, worst-VAL-half obj) → KILLED**: best worst-VAL-half AUC .5437 but the binding 2026-OOS held-out AUC is no better (tuned .5077 < default .5084), cov3 CI-lo clears only 1 yr — hyperparameters are not the constraint; reproduces the 2m anti-transfer. `usdjpy_15m_optuna_result.json`. I4 calibration closed by **rank-invariance** (isotonic/Platt are monotonic in p → they preserve the |p-0.5| rank a coverage gate selects on → cannot change the certified cov-gated win-rate; derivation, not experiment). **All Tier-I levers I1–I7 now RUN/closed.**
- **EXHAUSTIVE 2nd-pass RUNs (2026-06-08, no more subsume-by-argument), all NULL/data-limited at 15m NY:** C1 HMM .4966, C2 Kalman .4829 (anti-predictive), C3 RMT .5072, C4 CCM .4983, C5 ARF ~.50 (leak-corrected from a bogus .85), E2 magdir (flat across mag quartiles), E3 complexity (flat across bins), E4 info-bars (no volume on disk); plus A2 compression gate, A8 specialist (subset-train kills ranking), A5 stack (collinear), D1 GRU (.5164).
- **★ THE TB-LABEL HONESTY CASE (rigor for honesty, not dismissal):** the triple-barrier first-touch TRAIN label (TN5) was the ONE lever to beat base VAL-AUC (.5439>.539) and certified at LEVEL (UP .6052/DOWN .58 @cov2%, 15/15). My first read called it the new best book — **WRONG.** A 6-lens adversarial-refutation workflow (look-ahead, eval-contamination, CPCV-purge, noise, settlement, mechanism) + my Tier-1 paired per-path recompute vs the **matched** seed-ens K=3 base showed: the headline compared the wrong incumbent (single-model, not K3) and cov-cherry-picked. Paired lift is within CPCV-path noise (cov2% UP +.0067 t.91; **DOWN −.0093 t−.89, central tendency REGRESSES, 7/15 wins**; COMB +.0026) and sign-inconsistent across coverage (cov3% COMB +.0093 but t2.73 on highly-correlated CPCV paths → discount). **Verdict: certified-at-level, NOT a robust improvement; incumbent NOT unseated.** Book `USDJPY.m15ny_tbft_seedens.v1` kept as a recorded alternative (honest co-certified config). Mechanism: label denoising lifts a single p10 order statistic of a noisy 15-path distribution without moving global AUC or central-tendency win-rate.
- **Discovery: 3 dry rounds** (R1 TN1/TN4/TN2; R2 corpus+repo scan; R3 exhaustive 2nd pass C1-5+E2-4+TN5). Loop SATURATED on bar data.
- **★ TICK-MICROSTRUCTURE FRONTIER TESTED (2026-06-09; raw ticks relocated to /home/sean/git/raw) → NO deployable 15m-direction lift (trap #9).** Built causal OFI/microprice/spread/intensity/rv/momentum features (60/300/900s) at the NY 15m grid. **A 4-gate investigation, and a model case of CPCV being necessary-not-sufficient:** (1) caught+fixed a broker-local-time file-labeling bug that had staled afternoon-UTC features; (2) corrected CPCV showed a small consistent lift (bars+tk +.0024 AUC, 14/15 folds, cov2% WR .582→.605); (3) leakage controls PASSED (60s-lag gives identical lift → no boundary leak; shuffle AUC .4997); BUT (4) the **frozen-past forward holdout (train ≤2021 → fwd) KILLED it**: per-year COMB Δ = +.004 / +.036 / **−.035** (2026), binding year .526→.491, the model riding spread/intensity *regime* features (sprd_900/flow_900/nt_900) → **pooled-CPCV non-stationary-feature memorization, not a forward edge.** The ~.539 bar bound holds even with full tick OFI/quote-volume data (sign-invariance; microstructure lives at seconds, label is 900s out). `usdjpy_15m_tickmicro{,_verify,_frozen}_result.json`.
- **Remaining frontiers (out of 15m-direction scope):** seconds-horizon (1-5s) tick direction (~.65, needs a non-15m venue), tick MAGNITUDE (sign-invariant, separate key), US–JP rate-diff, traded JPY-crosses, RR skew.

---

## How to maintain this file
- One row per key; never copy a COMBINED number into an UP/DOWN key; never copy one timeframe/side's number to another.
- Every number traces to an on-disk result JSON (Tier-1). Flag n<50 + VAL-acc-max as non-robust.
- Unseat a side-leader only if the challenger beats the incumbent's binding (worst held-out) year with CI95-lower clearing it under the discipline, ideally refit-CPCV-certified.
- Update the master table + leaderboard + `sweeps/USDJPY_1m.md` ledger + `sweeps/USDJPY_1m_backlog.md` together; commit often.

## USDJPY — DERIV-FEED PROGRAM (Deriv-only data; opened 2026-11-15)

> **SCOPE:** this key on the Deriv feed only — Deriv 1-second mid ticks (`/mnt/data/deriv`), no Dukascopy data. Labels settle on Deriv ticks (entry first tick >= decision + 1 s, exit last tick <= entry target + H, ties lose); DEV window 2025-08-11..2026-07-31 for training, selection and CPCV; HOLDOUT 2026-08-01..2026-11-14 sealed (opened once per finalized candidate). Protocol and tools: `docs/DERIV_SOTA_PROGRAM.md`. Tables below are rendered from the result JSONs by `scripts/render_ledger.py` (do not edit by hand).

<!-- DERIV-FEED-NOTES:BEGIN -->
- **Incumbent benchmark on TRADEABLE decisions, books as trained (R1bt, 13:00-16:35 + 18:00-22:00 ET):** UP best `USDJPY.m15ny_seedens.v1` 0.6500 [0.5750, 0.7188] n=160 (SUPPORTED); DOWN best `USDJPY.m15ny_tbft_seedens.v1` 0.6250 [0.5250, 0.7250] n=80 (NOT SUPPORTED).
- **Books as deployed today (R1t, true clock, NY 08:00-16:35, tradeable):** UP best `USDJPY.m15ny_seedens.v1` 0.5253 [0.4822, 0.5666] n=533 (NOT SUPPORTED); DOWN best `USDJPY.m15ny_tbft_seedens.v1` 0.5133 [0.4333, 0.5933] n=150 (NOT SUPPORTED).
- **DEV-CPCV certified sides, GOAL rule (>= 12 of 15 paths counted and clear, p10 >= .541; tradeable runs):** none yet. DEV certification is necessary, not sufficient: the sealed holdout decides (docs/DERIV_SOTA_PROGRAM.md).
- **DEV-CPCV certified under the owner's drift amendment** (decision 1, 2026-11-15: for a genuinely drifting pair the moved up-rate checks - in-session band and per-path tripwire - are replaced by the hour x volatility drift placebo of the same configuration; mirage check: 15m tie share 0.0119 (1,542 / 129,284), the other six pairs 0.0109-0.0182; tick-settled labels, no forward fill; 30m tie share 0.0081 (1,016 / 125,528), the other six pairs 0.0073-0.0131; tick-settled labels, no forward fill): c2nysabs 15m UP (cov 0.03: 15/15 clear, p10 0.5760; DEV-CERT (drift amendment, placebo `a6placebo`)); e1t 15m UP (cov 0.03: 14/15 clear, p10 0.5554; DEV-CERT (drift amendment, placebo `a1placebo`)); e30nystroem 15m UP (cov 0.03: 15/15 clear, p10 0.5587; DEV-CERT (drift amendment, placebo `a3placebo`)); e49nys30 30m UP (cov 0.03: 15/15 clear, p10 0.5984; DEV-CERT (drift amendment, placebo `a8placebo`)). Configurations without a placebo run stay INVALID. Final certification also needs the holdout (drift-matched excess CI95 lower bound > 0 and the holdout conditions).
- **DEV leaderboard 15m UP** (closest to the goal rule at each run's operating coverage): `c2nysabs` 15/15 clear, p10 0.576, excess +0.085, original rule: INVALID (up-rate checks); effective (owner decision 1): DEV-CERT (drift amendment, placebo `a6placebo`); `e19tf` 15/15 clear, p10 0.567, excess +0.086, INVALID (up-rate checks); `e48nysc1000` 15/15 clear, p10 0.565, excess +0.076, INVALID (up-rate checks).
- **DEV leaderboard 15m DOWN** (closest to the goal rule at each run's operating coverage): `e57reflect` 8/15 clear, p10 0.475, excess +0.092, INVALID (up-rate checks); `e75lgbsig` 8/15 clear, p10 0.469, excess +0.072, INVALID (up-rate checks); `e32spec` 7/15 clear, p10 0.459, excess +0.049, INVALID (up-rate checks).
- **DEV leaderboard 30m UP** (closest to the goal rule at each run's operating coverage): `e49nys30` 15/15 clear, p10 0.598, excess +0.095, original rule: INVALID (up-rate checks); effective (owner decision 1): DEV-CERT (drift amendment, placebo `a8placebo`); `e5h30` 12/13 clear, p10 0.567, excess +0.082, INVALID (up-rate checks).
- **DEV leaderboard 30m DOWN** (closest to the goal rule at each run's operating coverage): `e5h30` 2/2 clear, p10 0.631, excess +0.200, INVALID (up-rate checks).
- **Holdout look 1** (`deriv_e30nystroem_look1decision_USDJPY_15m_result.json`; window 2026-08-01..2026-11-14; frozen code; owner decision 6): `e30nystroem` USDJPY.15m UP won 149 of 253 trades (win rate 0.5889; z 1.53) against a boundary of 160 wins (z 2.9178): not crossed. Conditions: n >= 50 True; drift-matched excess (NY hour x rv_60m) 0.071 CI95 [0.01, 0.1302], passes True; incumbents: `USDJPY.m15ny.v1` (ny) 85/158 = 0.5380 beaten; `USDJPY.m15ny_seedens.v1` (ny) 67/116 = 0.5776 beaten; `USDJPY.m15ny_tbft_seedens.v1` (ny) 57/105 = 0.5429 beaten; `USDJPY.m15ny.v1` (nylate) 40/76 = 0.5263 beaten; `USDJPY.m15ny_seedens.v1` (nylate) 29/49 = 0.5918 NOT beaten; `USDJPY.m15ny_tbft_seedens.v1` (nylate) 22/45 = 0.4889 beaten. Decision: continue to look 2.
- **Holdout look 1** (`deriv_e49nys30_look1decision_USDJPY_30m_result.json`; window 2026-08-01..2026-11-14; frozen code; owner decision 6): `e49nys30` USDJPY.30m UP won 59 of 102 trades (win rate 0.5784; z 0.7586) against a boundary of 70 wins (z 2.9443): not crossed. Conditions: n >= 50 True; drift-matched excess (NY hour x rv_60m) 0.0375 CI95 [-0.06, 0.1315], passes False; incumbents: none registered for this key. Decision: continue to look 2.
- **Deployment status (Deriv feed):** USDJPY.15m UP DO NOT DEPLOY yet (`e30nystroem` did not certify at look 1; look 2, on the window to 2027-02-26, decides); USDJPY.30m UP DO NOT DEPLOY yet (`e49nys30` did not certify at look 1; look 2, on the window to 2027-02-26, decides); every other key and side: DO NOT DEPLOY - no candidate has passed both the DEV-CPCV goal rule (or the drift amendment) and a sealed-holdout look, and deployment needs both.
- Superseded: `e1` and `r2` ran before the venue tradeability filter (they include untradeable blackout decisions); their tradeable re-runs are `e1t` and `r2t`.
- Robustness checks, not candidates (excluded from the lists and leaderboards above; tables below): `a12b02`, `a13b02`.
<!-- DERIV-FEED-NOTES:END -->

> **Drift diagnosis for the moved up-rate validity check (2026-11-15, hand-written; evidence re-checkable).** `e1t` (NY
> 08:00-16:35 tradeable) is INVALID under the pre-registered rule because the in-session moved up-rate is 0.5315 (band
> [0.47, 0.53]) and the Apr-Jul 2026 path reaches 0.5515 (tripwire 0.55). The rule exists to catch forward-fill
> fake-flat mirages; here the labels are tick-settled with no forward fill, the tie rate is normal (about 1% in NY
> hours, every month), and the excess is genuine drift: USDJPY rose 6.8% over the DEV window (first tick 2025-08-11
> 147.658, last tick 2026-07-31 157.657; 162.636 on 2026-07-01), with all-hours monthly up-rates above 0.50 every month
> (0.5484 in May 2026). The `e1t` UP win rate exceeds the always-UP baseline on all 15 paths. The verdict stays INVALID
> (rule unchanged); whether to replace the band with mirage diagnostics plus a drift-neutral requirement is an open
> owner decision (docs/DERIV_SOTA_PROGRAM.md). [Superseded 2026-11-15: owner decision 1 adopted the drift amendment;
> the generated lines above give the current status.]

> **USDJPY 15m UP — ablations and drift placebo (2026-11-15, hand-written; numbers from the cited JSONs).** At the
> operating coverage 0.03 (NY 08:00-16:35 tradeable), removing the clock or the slow (480/1440-minute) features or both
> keeps the moved AUC (0.5256 `e1t`, 0.5244 `e2noclock`, 0.5267 `e11noslow`, 0.5248 `e12noslowclock`) and the UP excess
> over always-UP (+0.080 on 15/15 paths, +0.084 on 15/15, +0.062 on 14/15, +0.050 on 14/15); UP p10 0.5554 / 0.5425 /
> 0.5345 / 0.5288 with 14 / 14 / 11 / 12 of 15 paths clearing .541. Forward walks have about 50 trades per fold
> (3/3, 0/3, 4/4, 2/4 counted folds clear) and do not separate the variants. Every run is INVALID by the moved up-rate
> band (drift), so none certified under the rule then in force (superseded 2026-11-15 by owner decision 1). Drift placebo `deriv_a1placebo_USDJPY_15m_result.json` (E1t re-run with its
> predictions dumped; the rebuilt trades match the run exactly): against always-UP baselines matched on New York hour
> and on hour × rv_60m tercile within each path, the UP trades' excess is +0.066 (hour × volatility) with 14 of 15 CPCV
> paths positive (pre-registered reading met: SKILL BEYOND DRIFT TIMING); forward folds +0.045, 3/3 positive. DOWN:
> +0.077 on 8/8 counted paths, below the 12-path reading (too thin). The analysis' first version pooled the forward
> folds into the reading; the JSON records the correction.
>
> **Entry-delay stress (2026-11-15, `deriv_a2delay_USDJPY_15m_result.json`).** The `e1t` UP trades, frozen as selected,
> re-settled from the ticks with the contract started 2 / 4 / 9 / 29 s later (zero extra delay reproduces the dataset
> labels exactly): paths clear 14 / 13 / 13 / 12 / 13 of 15, p10 0.555 / 0.531 / 0.540 / 0.527 / 0.532, mean 0.595 /
> 0.592 / 0.594 / 0.591 / 0.587, excess over always-UP +0.080 / +0.078 / +0.079 / +0.077 / +0.070. Pre-registered
> reading (still >= 12 of 15 paths clear with p10 >= 0.541 at +4 s): FRAGILE, by 0.0014. The selection skill is
> stable under delay; the certification margin is not. Staleness: the excess is positive both for trades whose last mid
> change was <= 2 s old (n 932, +0.076) and older (n 573, +0.083); the run's tercile verdict is void (tied age
> quantiles left one bin empty; correction recorded in the JSON). The incumbent recipe refit on Deriv data on its
> training clock (`r3`, NY 13:00-16:35 + 18:00-22:00) is too thin to read at its deploy coverage 0.02 (1 of 15 paths
> reaches 25 trades) and INVALID by drift.
>
> **Base kernel candidate: `e30nystroem` UP (E1t features, RBF Nystroem features + logistic SGD; 2026-11-15; on DEV p10
> it is now behind `c2nysabs`, note below).** At
> the operating coverage 0.03: p10 0.5587 with 15 of 15 paths clearing .541 (path minimum 0.5488, mean 0.5963, median
> 228 trades per path), excess over always-UP +0.081 on 15/15 paths, forward walk 3 of 4 counted folds clear (mean
> 0.587); coverage 0.02: p10 0.576, 15/15; coverage 0.05: p10 0.539, 13/15. Drift placebo
> `deriv_a3placebo_USDJPY_15m_result.json`: +0.061 over hour × volatility-matched always-UP, 15 of 15 CPCV paths positive
> (forward folds 4/4, +0.048) - SKILL BEYOND DRIFT TIMING. Entry-delay stress `deriv_a4delay_USDJPY_15m_result.json`:
> 15 of 15 paths clear at +0 / 2 / 4 / 9 / 29 s with p10 0.559 / 0.561 / 0.559 / 0.560 / 0.564 - OPERATIONALLY ROBUST;
> excess positive in every rank tercile of feed staleness (+0.091 / +0.077 / +0.061). Verdict under the program's rule:
> INVALID (moved up-rate 0.5315, paths to 0.5515) - it meets every other DEV condition of the goal rule. Owner
> decision 1 (docs/DERIV_SOTA_PROGRAM.md, resolved 2026-11-15: drift amendment adopted) makes it DEV-certified (generated
> line above); the sealed holdout decides whether it holds out of sample.

> **`e30nystroem` UP vs the sign-null replay (2026-11-15, hand-written; `deriv_a5null_USDJPY_15m_result.json`).** The
> same configuration re-run 16 times with every label replaced by the label W weeks later (W = 3, 6, .., 48, circular;
> the moved up-rate stays 0.522-0.532 and the time-of-day / day-of-week structure is kept, the feature-label link is
> broken). Null p10 median 0.491, 95% quantile 0.529, maximum 0.530 (W = 30); the most paths any null cleared is 12/15
> (W = 27, p10 0.529); no null meets the goal path test; null moved AUC 0.497-0.521 vs 0.526 real. Real p10 0.5587
> ranks 17 of 17. Pre-registered reading (in the JSON before any null ran): SIGNIFICANT. Within this diagnostic none
> of the 16 label shifts - which keep the drift and the calendar structure - reproduced the result; it tests this one
> selected configuration, not the selection across the program's search, so it is not a search-adjusted significance
> claim.

> **`c2nysabs` UP: the kernel with |return| training weights (2026-11-15, hand-written; numbers from the cited JSONs).**
> A Tier-I lever on the base kernel, pre-registered in the lab before it ran (lab PROGRESS.md). Operating coverage 0.03:
> 15 of 15 paths clear, p10 0.5760 (base 0.5587), forward walk 3 of 3 counted folds clear (mean 0.5785;
> `deriv_c2nysabs_USDJPY_15m_result.json`). Drift placebo `deriv_a6placebo_USDJPY_15m_result.json`: hour x volatility
> matched excess +0.063 on 15 of 15 paths - SKILL BEYOND DRIFT TIMING (rebuilt trades match the run). Entry-delay stress
> `deriv_a7delay_USDJPY_15m_result.json`: 15 of 15 paths clear at +0 / +2 / +4 s (p10 0.576 / 0.579 / 0.575) -
> OPERATIONALLY ROBUST. It was chosen as the best of about ten kernel levers on DEV p10, so its DEV number carries
> selection inflation. Owner decision 6 opened only `e30nystroem` for this key in the two-look sequential holdout;
> `c2nysabs` stays DEV-certified and unopened.
> The other pre-registered levers on the kernel do not improve it (paths clear >= 15 and p10 > 0.5587 needed):
> `e44nysseed3` (3 seeds, the c1 lever; 15, 0.5579), `e45nysaci` (ACI gate, the c3 lever; 15, 0.5515), `c4nysmeta`
> (meta gate keep 0.5, coverage doubled to 0.06 by design; 15, 0.5582), `c5nysmag` (magnitude gate keep 0.5; 14,
> 0.5433), `c6nyscomp` (compression gate; 11 of 14 counted, 0.5114).

> **USDJPY 30m UP — the kernel at 30 minutes (2026-11-15, hand-written; numbers from the cited JSONs).** `e49nys30`
> (`cpcv_deriv_x.py --horizon 30 --session ny --model nystroem --nseed 1`, operating coverage 0.03, `dataset_dev_v0.1`):
> 15 of 15 paths clear, p10 0.5984 (path minimum 0.5946, mean 0.6342, median 131 trades per path), forward walk 3 of 4
> counted folds clear (mean 0.5835; `deriv_e49nys30_USDJPY_30m_result.json`). INVALID by the moved up-rate checks
> (path up-rates 0.5259-0.5670). Drift placebo `deriv_a8placebo_USDJPY_30m_result.json` (same configuration; reading
> registered before the analysis): hour x volatility matched excess +0.075 on 15 of 15 paths - SKILL BEYOND DRIFT TIMING
> (rebuilt trades match the run). Entry-delay stress `deriv_a9delay_USDJPY_30m_result.json`: 15 of 15 paths clear at
> +0 / +2 / +4 / +9 / +29 s (p10 0.598 / 0.598 / 0.594 / 0.596 / 0.594) - OPERATIONALLY ROBUST, NOT A STALE-FEED
> ARTIFACT; its +0 s reading reproduces the run (p10 0.5984, mean 0.6342). DEV-certified under owner decision 1 (30m
> mirage check in the generated lines above). It is the only 30-minute configuration run on this key, so its DEV number
> carries no within-key selection at 30 minutes, but the horizon was chosen after the 15-minute kernel results; the
> sealed holdout is the test. No frozen book exists for USDJPY 30m, so the incumbent condition is empty for this key.

> **Holdout finalists and their trade dependence (2026-11-15, hand-written; numbers from the cited JSONs).** Owner
> decision 6 opens one finalist per key in a two-look sequential holdout: 15m UP `e30nystroem`, 30m UP `e49nys30`
> (`docs/DERIV_SOTA_PROGRAM.md`, Holdout discipline). The test treats nonoverlapping trade outcomes as independent given
> the past; on the finalists' DEV CPCV trades the day- and week-clustered variance of the win rate is 0.894 / 0.856 of
> the independent-trades variance at 15m (lag-1 autocorrelation -0.024; `deriv_a10dep_USDJPY_15m_result.json`) and
> 0.849 / 0.709 at 30m (-0.050; `deriv_a11dep_USDJPY_30m_result.json`) - no overdispersion, reading SUPPORTED
> (pre-registered in the JSONs). Both finalists' 15 DEV paths are reproduced exactly by the current harness.
> Builder change (2026-11-16): the holdout's full-period dataset uses deriv_dataset 0.2, which skips days whose sidecar
> is not complete; over DEV that drops the 2025-12-24 and 2025-12-31 early closes for every pair (352,032 USDJPY rows vs
> 354,674; lookback features differ near those days; labels identical on 99.99% of common rows). Re-run on the 0.2 DEV
> build, both finalists still pass the goal path test - reading ROBUST, pre-registered: `e30nystroem` 14/15 clear, p10
> 0.5661 (`deriv_a12b02_USDJPY_15m_result.json`); `e49nys30` 15/15, p10 0.6089 (`deriv_a13b02_USDJPY_30m_result.json`).

<!-- DERIV-FEED-TABLES:BEGIN -->
**Frozen books on the Deriv feed — true clock, ny (America/New_York 08:00-17:00, DST-correct, sessions.session_mask), ALL decisions incl. the untradeable blackout** (`deriv_r1_frozen_books_dev_result.json`; strategy gate, tick-settled, ties lose; SUPPORTED iff n >= 50 and CI95-lo >= .541):

| Book | AUC (moved) | gate rate | UP n · WR [CI95] | DOWN n · WR [CI95] | COMB n · WR | verdict UP / DOWN |
|---|---|---|---|---|---|---|
| `USDJPY.m15ny.v1` | .5208 | .0478 | 725 · .5228 [.4855, .5600] | 231 · .4805 [.4156, .5455] | 952 · .5137 | NOT SUPPORTED / NOT SUPPORTED |
| `USDJPY.m15ny_seedens.v1` | .5201 | .0358 | 533 · .5253 [.4822, .5666] | 211 · .5071 [.4408, .5735] | 741 · .5209 | NOT SUPPORTED / NOT SUPPORTED |
| `USDJPY.m15ny_tbft_seedens.v1` | .5217 | .0324 | 517 · .5203 [.4778, .5648] | 150 · .5133 [.4333, .5933] | 665 · .5188 | NOT SUPPORTED / NOT SUPPORTED |

**Frozen books on the Deriv feed — training clock (true UTC - 5 h), nylate (America/New_York 13:00-22:00, DST-correct), ALL decisions incl. the untradeable blackout** (`deriv_r1b_frozen_books_trainclock_nylate_dev_result.json`; strategy gate, tick-settled, ties lose; SUPPORTED iff n >= 50 and CI95-lo >= .541):

| Book | AUC (moved) | gate rate | UP n · WR [CI95] | DOWN n · WR [CI95] | COMB n · WR | verdict UP / DOWN |
|---|---|---|---|---|---|---|
| `USDJPY.m15ny.v1` | .5464 | .0361 | 535 · .6449 [.6037, .6860] | 194 · .6443 [.5773, .7113] | 710 · .6380 | SUPPORTED / SUPPORTED |
| `USDJPY.m15ny_seedens.v1` | .5435 | .0255 | 417 · .6835 [.6379, .7266] | 152 · .6184 [.5395, .6974] | 559 · .6583 | SUPPORTED / NOT SUPPORTED |
| `USDJPY.m15ny_tbft_seedens.v1` | .5468 | .0272 | 426 · .6432 [.5962, .6878] | 135 · .7333 [.6593, .8074] | 543 · .6519 | SUPPORTED / SUPPORTED |

**Frozen books on the Deriv feed — training clock (true UTC - 5 h), nylate (America/New_York 13:00-22:00, DST-correct), tradeable decisions only (Deriv purchase blackout excluded)** (`deriv_r1bt_frozen_books_trainclock_nylate_dev_tradeable_result.json`; strategy gate, tick-settled, ties lose; SUPPORTED iff n >= 50 and CI95-lo >= .541):

| Book | AUC (moved) | gate rate | UP n · WR [CI95] | DOWN n · WR [CI95] | COMB n · WR | verdict UP / DOWN |
|---|---|---|---|---|---|---|
| `USDJPY.m15ny.v1` | .5322 | .0175 | 227 · .6300 [.5683, .6916] | 142 · .5352 [.4577, .6127] | 367 · .5940 | SUPPORTED / NOT SUPPORTED |
| `USDJPY.m15ny_seedens.v1` | .5292 | .0120 | 160 · .6500 [.5750, .7188] | 111 · .5045 [.4144, .6036] | 270 · .5889 | SUPPORTED / NOT SUPPORTED |
| `USDJPY.m15ny_tbft_seedens.v1` | .5331 | .0099 | 150 · .6067 [.5267, .6867] | 80 · .6250 [.5250, .7250] | 229 · .6114 | NOT SUPPORTED / NOT SUPPORTED |

**Frozen books on the Deriv feed — true clock, ny (America/New_York 08:00-17:00, DST-correct, sessions.session_mask), tradeable decisions only (Deriv purchase blackout excluded)** (`deriv_r1t_frozen_books_dev_tradeable_result.json`; strategy gate, tick-settled, ties lose; SUPPORTED iff n >= 50 and CI95-lo >= .541):

| Book | AUC (moved) | gate rate | UP n · WR [CI95] | DOWN n · WR [CI95] | COMB n · WR | verdict UP / DOWN |
|---|---|---|---|---|---|---|
| `USDJPY.m15ny.v1` | .5158 | .0497 | 725 · .5228 [.4855, .5600] | 231 · .4805 [.4156, .5455] | 952 · .5137 | NOT SUPPORTED / NOT SUPPORTED |
| `USDJPY.m15ny_seedens.v1` | .5149 | .0372 | 533 · .5253 [.4822, .5666] | 211 · .5071 [.4408, .5735] | 741 · .5209 | NOT SUPPORTED / NOT SUPPORTED |
| `USDJPY.m15ny_tbft_seedens.v1` | .5173 | .0337 | 517 · .5203 [.4778, .5648] | 150 · .5133 [.4333, .5933] | 665 · .5188 | NOT SUPPORTED / NOT SUPPORTED |

**Per-fold-refit CPCV on the Deriv feed** (DEV 2025-08-11..2026-07-31, NY DST session, 15 paths; p10/frac = path win-rate 10th percentile / share of paths >= .541; fwd = forward-walk paths clearing .541 and their mean):

| Run | Model / features | H | cov | AUC mean (min) | UP p10 · frac · mean | UP fwd | DOWN p10 · frac · mean | DOWN fwd | always UP / DOWN | repo-rule verdict UP / DOWN | GOAL rule UP / DOWN (paths counted) | frac @.565 UP / DOWN | result JSON |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| a12b02 | a12b02: all (66), nseed 1 | 15m | 0.03 | .5260 (.5146) | .5661 · .933 · .5980 | 4/4 · .6291 | .4077 · .000 · .4170 | 1/2 · .5208 | .5151 / .4743 | INVALID (moved up-rate validity check failed) / INVALID (moved up-rate validity check failed) | no (15/15, 14 clear) / no (2/15, 0 clear) | 0.87 / 0.00 | `deriv_a12b02_USDJPY_15m_result.json` |
| a13b02 | a13b02: all (66), nseed 1 | 30m | 0.03 | .5337 (.5168) | .6089 · 1.0 · .6377 | 3/3 · .6163 | .5186 · .500 · .5929 | 0/1 · .4194 | .5381 / .4550 | INVALID (moved up-rate validity check failed) / INVALID (moved up-rate validity check failed) | no (15/15, 15 clear) / no (2/15, 1 clear) | 1.00 / 0.50 | `deriv_a13b02_USDJPY_30m_result.json` |
| c2nysabs | c2nysabs: all (66), nseed 1 | 15m | 0.03 | .5169 (.5045) | .5760 · 1.0 · .6000 | 3/3 · .5785 | .4497 · .250 · .5194 | 0/3 · .4661 | .5151 / .4741 | INVALID (moved up-rate validity check failed) / INVALID (moved up-rate validity check failed) | original rule: no (15/15, 15 clear); effective (owner decision 1): DEV-CERT (drift amendment, placebo `a6placebo`) / no (12/15, 3 clear) | 0.93 / 0.25 | `deriv_c2nysabs_USDJPY_15m_result.json` |
| c4nysmeta | c4nysmeta: all (66), nseed 1 | 15m | 0.06 | .5260 (.5184) | .5582 · 1.0 · .5901 | 4/4 · .6035 | .4795 · .000 · .4795 | 0/1 · .4510 | .5151 / .4741 | INVALID (moved up-rate validity check failed) / INVALID (moved up-rate validity check failed) | no (15/15, 15 clear) / no (1/15, 0 clear) | 0.80 / 0.00 | `deriv_c4nysmeta_USDJPY_15m_result.json` |
| c5nysmag | c5nysmag: all (66), nseed 1 | 15m | 0.03 | .5260 (.5184) | .5433 · .933 · .5739 | 3/4 · .5705 | .4310 · .000 · .4310 | 0/1 · .4872 | .5151 / .4741 | INVALID (moved up-rate validity check failed) / INVALID (moved up-rate validity check failed) | no (15/15, 14 clear) / no (1/15, 0 clear) | 0.47 / 0.00 | `deriv_c5nysmag_USDJPY_15m_result.json` |
| c6nyscomp | c6nyscomp: all (66), nseed 1 | 15m | 0.03 | .5260 (.5184) | .5114 · .786 · .5969 | 3/3 · .6354 | .4524 · .000 · .4524 | 0/0 · — | .5151 / .4741 | INVALID (moved up-rate validity check failed) / INVALID (moved up-rate validity check failed) | no (14/15, 11 clear) / no (1/15, 0 clear) | 0.71 / 0.00 | `deriv_c6nyscomp_USDJPY_15m_result.json` |
| e10v02 | e10v02: all (82), nseed 1 | 15m | 0.03 | .5258 (.5091) | .5352 · .786 · .5835 | 3/3 · .5846 | .4539 · .167 · .5081 | 0/0 · — | .5151 / .4741 | INVALID (moved up-rate validity check failed) / INVALID (moved up-rate validity check failed) | no (14/15, 11 clear) / no (6/15, 1 clear) | 0.57 / 0.17 | `deriv_e10v02_USDJPY_15m_result.json` |
| e11noslow | e11noslow: list:v01_noslow_USDJPY.json (61), nseed 1 | 15m | 0.03 | .5267 (.5195) | .5345 · .733 · .5773 | 4/4 · .5734 | .4543 · .500 · .5304 | 2/3 · .5576 | .5151 / .4741 | INVALID (moved up-rate validity check failed) / INVALID (moved up-rate validity check failed) | no (15/15, 11 clear) / no (14/15, 7 clear) | 0.53 / 0.29 | `deriv_e11noslow_USDJPY_15m_result.json` |
| e12noslowclock | e12noslowclock: list:v01_noslow_noclock_USDJPY.json (58), nseed 1 | 15m | 0.03 | .5248 (.5140) | .5288 · .800 · .5656 | 2/4 · .5628 | .4321 · .400 · .5217 | 0/3 · .4752 | .5151 / .4741 | INVALID (moved up-rate validity check failed) / INVALID (moved up-rate validity check failed) | no (15/15, 12 clear) / no (15/15, 6 clear) | 0.40 / 0.40 | `deriv_e12noslowclock_USDJPY_15m_result.json` |
| e13mag50 | e13mag50: all (66), nseed 1 | 15m | 0.03 | .5256 (.5141) | .5308 · .857 · .6113 | 1/1 · .6207 | .4667 · .333 · .5022 | 0/0 · — | .5151 / .4741 | INVALID (moved up-rate validity check failed) / INVALID (moved up-rate validity check failed) | no (14/15, 12 clear) / no (3/15, 1 clear) | 0.71 / 0.00 | `deriv_e13mag50_USDJPY_15m_result.json` |
| e14nylate | e14nylate: all (66), nseed 1 | 15m | 0.03 | .5234 (.5138) | .5100 · .533 · .5731 | 3/3 · .6004 | .3544 · .500 · .4961 | 0/0 · — | .5193 / .4663 | INVALID (moved up-rate validity check failed) / INVALID (moved up-rate validity check failed) | no (15/15, 8 clear) / no (8/15, 4 clear) | 0.40 / 0.38 | `deriv_e14nylate_USDJPY_15m_result.json` |
| e15nypm | e15nypm: all (66), nseed 1 | 15m | 0.03 | .5276 (.5082) | .5397 · .786 · .6119 | 1/1 · .6452 | nan · .000 · — | 0/0 · — | .5332 / .4520 | INVALID (moved up-rate validity check failed) / INVALID (moved up-rate validity check failed) | no (14/15, 11 clear) / no (0/15, 0 clear) | 0.79 / — | `deriv_e15nypm_USDJPY_15m_result.json` |
| e16nyeve | e16nyeve: all (66), nseed 1 | 15m | 0.03 | .5096 (.4975) | .4111 · .500 · .5369 | 2/2 · .6333 | .4171 · .000 · .4625 | 0/0 · — | .5053 / .4807 | KILLED / KILLED | no (14/15, 7 clear) / no (7/15, 0 clear) | 0.36 / 0.00 | `deriv_e16nyeve_USDJPY_15m_result.json` |
| e17nyam | e17nyam: all (66), nseed 1 | 15m | 0.03 | .5190 (.4969) | .5022 · .714 · .5767 | 2/2 · .6571 | .4600 · .000 · .4960 | 0/2 · .4115 | .5023 / .4898 | KILLED / KILLED | no (14/15, 10 clear) / no (7/15, 0 clear) | 0.57 / 0.00 | `deriv_e17nyam_USDJPY_15m_result.json` |
| e18lowcap | e18lowcap: all (66), nseed 1 | 15m | 0.03 | .5251 (.5163) | .5560 · .933 · .5808 | 3/4 · .6233 | .4418 · .200 · .5283 | 1/3 · .5229 | .5151 / .4741 | INVALID (moved up-rate validity check failed) / INVALID (moved up-rate validity check failed) | no (15/15, 14 clear) / no (5/15, 1 clear) | 0.80 / 0.20 | `deriv_e18lowcap_USDJPY_15m_result.json` |
| e19tf | e19tf: all+fam:tf (74), nseed 1 | 15m | 0.03 | .5280 (.5139) | .5675 · 1.0 · .6013 | 3/3 · .5664 | .4635 · .429 · .5263 | 0/0 · — | .5151 / .4741 | INVALID (moved up-rate validity check failed) / INVALID (moved up-rate validity check failed) | no (15/15, 15 clear) / no (7/15, 3 clear) | 0.93 / 0.29 | `deriv_e19tf_USDJPY_15m_result.json` |
| e1 | e1: all (66), nseed 1 | 15m | 0.03 | .5323 (.5195) | .5646 · 1.0 · .5935 | 2/3 · .5353 | .5317 · .800 · .5935 | 2/3 · .6105 | .5121 / .4771 | CERTIFIED / KILLED | CERTIFIED (15/15, 15 clear) / no (15/15, 12 clear) | 0.80 / 0.47 | `deriv_e1_USDJPY_15m_result.json` |
| e1t | e1t: all (66), nseed 1 | 15m | 0.03 | .5256 (.5141) | .5554 · .933 · .5948 | 3/3 · .5791 | .4636 · .250 · .5336 | 1/1 · .6250 | .5151 / .4741 | INVALID (moved up-rate validity check failed) / INVALID (moved up-rate validity check failed) | original rule: no (15/15, 14 clear); effective (owner decision 1): DEV-CERT (drift amendment, placebo `a1placebo`) / no (8/15, 2 clear) | 0.80 / 0.25 | `deriv_e1t_USDJPY_15m_result.json` |
| e20sg | e20sg: all+fam:sg (71), nseed 1 | 15m | 0.03 | .5271 (.5154) | .5377 · .867 · .6010 | 1/3 · .5404 | .4786 · .250 · .5376 | 0/0 · — | .5151 / .4741 | INVALID (moved up-rate validity check failed) / INVALID (moved up-rate validity check failed) | no (15/15, 13 clear) / no (8/15, 2 clear) | 0.67 / 0.25 | `deriv_e20sg_USDJPY_15m_result.json` |
| e21kf | e21kf: all+fam:kf (73), nseed 1 | 15m | 0.03 | .5275 (.5151) | .5622 · .933 · .6043 | 3/3 · .6036 | .4415 · .250 · .5001 | 0/0 · — | .5151 / .4741 | INVALID (moved up-rate validity check failed) / INVALID (moved up-rate validity check failed) | no (15/15, 14 clear) / no (8/15, 2 clear) | 0.87 / 0.12 | `deriv_e21kf_USDJPY_15m_result.json` |
| e22rm | e22rm: all+fam:rm (70), nseed 1 | 15m | 0.03 | .5275 (.5140) | .5379 · .867 · .5968 | 2/3 · .5401 | .4663 · .000 · .4991 | 0/0 · — | .5151 / .4741 | INVALID (moved up-rate validity check failed) / INVALID (moved up-rate validity check failed) | no (15/15, 13 clear) / no (5/15, 0 clear) | 0.67 / 0.00 | `deriv_e22rm_USDJPY_15m_result.json` |
| e23fd | e23fd: all+fam:fd (68), nseed 1 | 15m | 0.03 | .5262 (.5166) | .5450 · .933 · .5826 | 3/3 · .5932 | .4062 · .417 · .5410 | 1/1 · .6207 | .5151 / .4741 | INVALID (moved up-rate validity check failed) / INVALID (moved up-rate validity check failed) | no (15/15, 14 clear) / no (12/15, 5 clear) | 0.60 / 0.33 | `deriv_e23fd_USDJPY_15m_result.json` |
| e24cx | e24cx: all+fam:cx (74), nseed 1 | 15m | 0.03 | .5310 (.5184) | .5206 · .800 · .5805 | 2/3 · .5392 | .4591 · .600 · .5791 | 0/0 · — | .5151 / .4741 | INVALID (moved up-rate validity check failed) / INVALID (moved up-rate validity check failed) | no (15/15, 12 clear) / no (5/15, 3 clear) | 0.60 / 0.60 | `deriv_e24cx_USDJPY_15m_result.json` |
| e25tk | e25tk: all+fam:tk (72), nseed 1 | 15m | 0.03 | .5263 (.5176) | .5203 · .667 · .5762 | 2/3 · .5844 | .4502 · .250 · .4966 | 1/1 · .6000 | .5151 / .4741 | INVALID (moved up-rate validity check failed) / INVALID (moved up-rate validity check failed) | no (15/15, 10 clear) / no (8/15, 2 clear) | 0.53 / 0.12 | `deriv_e25tk_USDJPY_15m_result.json` |
| e29logit | e29logit: all (66), nseed 1 | 15m | 0.03 | .5210 (.5054) | .5212 · .733 · .5622 | 3/4 · .5563 | .4266 · .000 · .4818 | 1/3 · .4798 | .5151 / .4741 | INVALID (moved up-rate validity check failed) / INVALID (moved up-rate validity check failed) | no (15/15, 11 clear) / no (9/15, 0 clear) | 0.53 / 0.00 | `deriv_e29logit_USDJPY_15m_result.json` |
| e2noclock | e2noclock: noclock (63), nseed 1 | 15m | 0.03 | .5244 (.5150) | .5425 · .933 · .5989 | 0/3 · .4818 | .4242 · .273 · .4982 | 1/1 · .7097 | .5151 / .4741 | INVALID (moved up-rate validity check failed) / INVALID (moved up-rate validity check failed) | no (15/15, 14 clear) / no (11/15, 3 clear) | 0.67 / 0.18 | `deriv_e2noclock_USDJPY_15m_result.json` |
| e30nystroem | e30nystroem: all (66), nseed 1 | 15m | 0.03 | .5260 (.5184) | .5587 · 1.0 · .5963 | 3/4 · .5871 | .4647 · .000 · .4647 | 0/1 · .5179 | .5151 / .4741 | INVALID (moved up-rate validity check failed) / INVALID (moved up-rate validity check failed) | original rule: no (15/15, 15 clear); effective (owner decision 1): DEV-CERT (drift amendment, placebo `a3placebo`) / no (1/15, 0 clear) | 0.80 / 0.00 | `deriv_e30nystroem_USDJPY_15m_result.json` |
| e32spec | e32spec: all (66), nseed 1 | 15m | 0.03 | .5124 (.5008) | .5345 · .867 · .5728 | 3/4 · .6024 | .4588 · .467 · .5235 | 3/4 · .5429 | .5151 / .4741 | INVALID (moved up-rate validity check failed) / INVALID (moved up-rate validity check failed) | no (15/15, 13 clear) / no (15/15, 7 clear) | 0.53 / 0.27 | `deriv_e32spec_USDJPY_15m_result.json` |
| e33comp33 | e33comp33: all (66), nseed 1 | 15m | 0.1 | .5256 (.5141) | .4913 · .643 · .5512 | 3/3 · .5976 | .4684 · .600 · .5798 | 2/3 · .5444 | .5151 / .4741 | INVALID (moved up-rate validity check failed) / INVALID (moved up-rate validity check failed) | no (14/15, 9 clear) / no (5/15, 3 clear) | 0.43 / 0.40 | `deriv_e33comp33_USDJPY_15m_result.json` |
| e34online | e34online: all (66), nseed 1 | 15m | 0.03 | .5070 (.5013) | .5191 · .467 · .5408 | 0/0 · — | .4582 · .067 · .4929 | 0/0 · — | .5151 / .4741 | INVALID (moved up-rate validity check failed) / INVALID (moved up-rate validity check failed) | no (15/15, 7 clear) / no (15/15, 1 clear) | 0.13 / 0.00 | `deriv_e34online_USDJPY_15m_result.json` |
| e35absret | e35absret: all (66), nseed 1 | 15m | 0.03 | .5241 (.5138) | .5623 · 1.0 · .5978 | 1/2 · .5349 | .4106 · .300 · .4781 | 1/1 · .6667 | .5151 / .4741 | INVALID (moved up-rate validity check failed) / INVALID (moved up-rate validity check failed) | no (14/15, 14 clear) / no (10/15, 3 clear) | 0.86 / 0.20 | `deriv_e35absret_USDJPY_15m_result.json` |
| e36aci | e36aci: all (66), nseed 1 | 15m | 0.03 | .5256 (.5141) | .5440 · .867 · .5903 | 3/3 · .5820 | .4773 · .444 · .5490 | 1/1 · .6061 | .5151 / .4741 | INVALID (moved up-rate validity check failed) / INVALID (moved up-rate validity check failed) | no (15/15, 13 clear) / no (9/15, 4 clear) | 0.80 / 0.33 | `deriv_e36aci_USDJPY_15m_result.json` |
| e37meta | e37meta: all (66), nseed 1 | 15m | 0.06 | .5256 (.5141) | .5598 · .933 · .5890 | 3/3 · .6267 | .4428 · .250 · .5122 | 1/2 · .5430 | .5151 / .4741 | INVALID (moved up-rate validity check failed) / INVALID (moved up-rate validity check failed) | no (15/15, 14 clear) / no (12/15, 3 clear) | 0.80 / 0.08 | `deriv_e37meta_USDJPY_15m_result.json` |
| e38release | e38release: all (66), nseed 1 | 15m | 0.03 | .5036 (.4988) | .5300 · .667 · .5570 | 3/4 · .5660 | .4729 · .267 · .5178 | 1/4 · .4958 | .5151 / .4741 | INVALID (moved up-rate validity check failed) / INVALID (moved up-rate validity check failed) | no (15/15, 10 clear) / no (15/15, 4 clear) | 0.40 / 0.13 | `deriv_e38release_USDJPY_15m_result.json` |
| e3union | e3union: all (305), nseed 1 | 15m | 0.03 | .5226 (.5019) | .5313 · .833 · .6169 | 1/1 · .6296 | .4000 · .000 · .4000 | 0/0 · — | .5151 / .4741 | INVALID (moved up-rate validity check failed) / INVALID (moved up-rate validity check failed) | no (12/15, 10 clear) / no (1/15, 0 clear) | 0.67 / 0.00 | `deriv_e3union_USDJPY_15m_result.json` |
| e40forest | e40forest: all (66), nseed 1 | 15m | 0.03 | .5302 (.5141) | .5256 · .733 · .5839 | 4/4 · .6392 | nan · .000 · — | 0/0 · — | .5151 / .4741 | INVALID (moved up-rate validity check failed) / INVALID (moved up-rate validity check failed) | no (15/15, 11 clear) / no (0/15, 0 clear) | 0.67 / — | `deriv_e40forest_USDJPY_15m_result.json` |
| e41pool | e41pool: all (66), nseed 1 | 15m | 0.03 | .5292 (.5153) | .5479 · .933 · .5822 | 3/4 · .5721 | .5185 · .000 · .5185 | 0/0 · — | .5151 / .4741 | INVALID (moved up-rate validity check failed) / INVALID (moved up-rate validity check failed) | no (15/15, 14 clear) / no (1/15, 0 clear) | 0.60 / 0.00 | `deriv_e41pool_USDJPY_15m_result.json` |
| e42nysnoclock | e42nysnoclock: noclock (63), nseed 1 | 15m | 0.03 | .5228 (.5137) | .5369 · .867 · .5631 | 1/3 · .5539 | .4491 · .000 · .4491 | 0/1 · .4706 | .5151 / .4741 | INVALID (moved up-rate validity check failed) / INVALID (moved up-rate validity check failed) | no (15/15, 13 clear) / no (1/15, 0 clear) | 0.47 / 0.00 | `deriv_e42nysnoclock_USDJPY_15m_result.json` |
| e43nysnoslow | e43nysnoslow: list:v01_noslow_USDJPY.json (61), nseed 1 | 15m | 0.03 | .5268 (.5165) | .5639 · 1.0 · .5891 | 2/4 · .5686 | .4729 · .000 · .4849 | 0/1 · .4600 | .5151 / .4741 | INVALID (moved up-rate validity check failed) / INVALID (moved up-rate validity check failed) | no (15/15, 15 clear) / no (2/15, 0 clear) | 0.87 / 0.00 | `deriv_e43nysnoslow_USDJPY_15m_result.json` |
| e44nysseed3 | e44nysseed3: all (66), nseed 3 | 15m | 0.03 | .5260 (.5181) | .5579 · 1.0 · .5907 | 3/4 · .6208 | .4504 · .000 · .4779 | 0/0 · — | .5151 / .4741 | INVALID (moved up-rate validity check failed) / INVALID (moved up-rate validity check failed) | no (15/15, 15 clear) / no (2/15, 0 clear) | 0.87 / 0.00 | `deriv_e44nysseed3_USDJPY_15m_result.json` |
| e45nysaci | e45nysaci: all (66), nseed 1 | 15m | 0.03 | .5260 (.5184) | .5515 · 1.0 · .5819 | 2/3 · .5832 | .4706 · .000 · .4706 | 0/1 · .4750 | .5151 / .4741 | INVALID (moved up-rate validity check failed) / INVALID (moved up-rate validity check failed) | no (15/15, 15 clear) / no (1/15, 0 clear) | 0.73 / 0.00 | `deriv_e45nysaci_USDJPY_15m_result.json` |
| e46nysg05 | e46nysg05: all (66), nseed 1 | 15m | 0.03 | .5260 (.5161) | .5585 · .933 · .5847 | 4/4 · .6186 | .4313 · .000 · .4784 | 0/1 · .4762 | .5151 / .4741 | INVALID (moved up-rate validity check failed) / INVALID (moved up-rate validity check failed) | no (15/15, 14 clear) / no (3/15, 0 clear) | 0.80 / 0.00 | `deriv_e46nysg05_USDJPY_15m_result.json` |
| e47nysg2 | e47nysg2: all (66), nseed 1 | 15m | 0.03 | .5256 (.5180) | .5471 · .933 · .5825 | 4/4 · .6029 | .5068 · .000 · .5068 | 1/2 · .5663 | .5151 / .4741 | INVALID (moved up-rate validity check failed) / INVALID (moved up-rate validity check failed) | no (15/15, 14 clear) / no (1/15, 0 clear) | 0.73 / 0.00 | `deriv_e47nysg2_USDJPY_15m_result.json` |
| e48nysc1000 | e48nysc1000: all (66), nseed 1 | 15m | 0.03 | .5252 (.5157) | .5646 · 1.0 · .5907 | 4/4 · .5981 | .4756 · .000 · .4758 | 0/1 · .5385 | .5151 / .4741 | INVALID (moved up-rate validity check failed) / INVALID (moved up-rate validity check failed) | no (15/15, 15 clear) / no (2/15, 0 clear) | 0.80 / 0.00 | `deriv_e48nysc1000_USDJPY_15m_result.json` |
| e49nys30 | e49nys30: all (66), nseed 1 | 30m | 0.03 | .5321 (.5189) | .5984 · 1.0 · .6342 | 3/4 · .5835 | nan · .000 · — | 0/1 · .3864 | .5396 / .4538 | INVALID (moved up-rate validity check failed) / INVALID (moved up-rate validity check failed) | original rule: no (15/15, 15 clear); effective (owner decision 1): DEV-CERT (drift amendment, placebo `a8placebo`) / no (0/15, 0 clear) | 1.00 / — | `deriv_e49nys30_USDJPY_30m_result.json` |
| e4seed3 | e4seed3: all (66), nseed 3 | 15m | 0.03 | .5261 (.5155) | .5394 · .867 · .5915 | 2/3 · .5754 | .4674 · .250 · .5137 | 0/1 · .5333 | .5151 / .4741 | INVALID (moved up-rate validity check failed) / INVALID (moved up-rate validity check failed) | no (15/15, 13 clear) / no (8/15, 2 clear) | 0.80 / 0.12 | `deriv_e4seed3_USDJPY_15m_result.json` |
| e51fx | e51fx: all+fam:fx (82), nseed 1 | 15m | 0.03 | .5271 (.5171) | .4872 · .733 · .5681 | 1/3 · .5449 | .4763 · .333 · .5402 | 0/1 · .4412 | .5151 / .4741 | INVALID (moved up-rate validity check failed) / INVALID (moved up-rate validity check failed) | no (15/15, 11 clear) / no (9/15, 3 clear) | 0.60 / 0.22 | `deriv_e51fx_USDJPY_15m_result.json` |
| e52ck | e52ck: all+fam:ck (77), nseed 1 | 15m | 0.03 | .5255 (.5155) | .5392 · .867 · .5797 | 2/3 · .5780 | .4804 · .400 · .5233 | 1/2 · .4975 | .5151 / .4741 | INVALID (moved up-rate validity check failed) / INVALID (moved up-rate validity check failed) | no (15/15, 13 clear) / no (10/15, 4 clear) | 0.53 / 0.30 | `deriv_e52ck_USDJPY_15m_result.json` |
| e53ckcompact | e53ckcompact: list:compact_kernels.json (14), nseed 1 | 15m | 0.03 | .5142 (.5064) | .5215 · .467 · .5499 | 3/4 · .5451 | .3973 · .000 · .4643 | 0/3 · .4355 | .5151 / .4741 | INVALID (moved up-rate validity check failed) / INVALID (moved up-rate validity check failed) | no (15/15, 7 clear) / no (15/15, 0 clear) | 0.27 / 0.00 | `deriv_e53ckcompact_USDJPY_15m_result.json` |
| e54ckcompactlogit | e54ckcompactlogit: list:compact_kernels.json (14), nseed 1 | 15m | 0.03 | .5204 (.5076) | .5313 · .733 · .5673 | 2/4 · .5620 | nan · .000 · — | 0/0 · — | .5151 / .4741 | INVALID (moved up-rate validity check failed) / INVALID (moved up-rate validity check failed) | no (15/15, 11 clear) / no (0/15, 0 clear) | 0.53 / — | `deriv_e54ckcompactlogit_USDJPY_15m_result.json` |
| e55rl | e55rl: all+fam:rl (72), nseed 1 | 15m | 0.03 | .5258 (.5144) | .5146 · .733 · .5830 | 3/3 · .5844 | .4361 · .400 · .5043 | 0/0 · — | .5151 / .4741 | INVALID (moved up-rate validity check failed) / INVALID (moved up-rate validity check failed) | no (15/15, 11 clear) / no (5/15, 2 clear) | 0.73 / 0.20 | `deriv_e55rl_USDJPY_15m_result.json` |
| e56rs | e56rs: all+fam:rs (72), nseed 1 | 15m | 0.03 | .5241 (.5131) | .5510 · 1.0 · .6121 | 3/3 · .6020 | .5077 · .333 · .5271 | 0/0 · — | .5151 / .4741 | INVALID (moved up-rate validity check failed) / INVALID (moved up-rate validity check failed) | no (15/15, 15 clear) / no (3/15, 1 clear) | 0.80 / 0.00 | `deriv_e56rs_USDJPY_15m_result.json` |
| e57reflect | e57reflect: all (66), nseed 1 | 15m | 0.03 | .5247 (.5115) | .5294 · .818 · .5936 | 0/0 · — | .4745 · .533 · .5666 | 0/1 · .4889 | .5151 / .4741 | INVALID (moved up-rate validity check failed) / INVALID (moved up-rate validity check failed) | no (11/15, 9 clear) / no (15/15, 8 clear) | 0.64 / 0.40 | `deriv_e57reflect_USDJPY_15m_result.json` |
| e58nysreflect | e58nysreflect: all (66), nseed 1 | 15m | 0.03 | .5177 (.5070) | .4908 · .417 · .5378 | 1/4 · .5444 | .4728 · .214 · .5155 | 0/4 · .4987 | .5151 / .4741 | INVALID (moved up-rate validity check failed) / INVALID (moved up-rate validity check failed) | no (12/15, 5 clear) / no (14/15, 3 clear) | 0.25 / 0.14 | `deriv_e58nysreflect_USDJPY_15m_result.json` |
| e59brier | e59brier: all (66), nseed 1 | 15m | 0.03 | .5247 (.5141) | .5458 · .867 · .5935 | 2/3 · .5490 | .4103 · .545 · .5191 | 1/2 · .5190 | .5151 / .4741 | INVALID (moved up-rate validity check failed) / INVALID (moved up-rate validity check failed) | no (15/15, 13 clear) / no (11/15, 6 clear) | 0.73 / 0.27 | `deriv_e59brier_USDJPY_15m_result.json` |
| e5h30 | e5h30: all (66), nseed 1 | 30m | 0.03 | .5319 (.5237) | .5671 · .923 · .6204 | 2/2 · .5857 | .6312 · 1.0 · .6374 | 0/0 · — | .5396 / .4538 | INVALID (moved up-rate validity check failed) / INVALID (moved up-rate validity check failed) | no (13/15, 12 clear) / no (2/15, 2 clear) | 0.85 / 1.00 | `deriv_e5h30_USDJPY_30m_result.json` |
| e60stack | e60stack: all (66), nseed 1 | 15m | 0.03 | .5270 (.5148) | .5536 · 1.0 · .5936 | 2/3 · .5760 | .5045 · .500 · .5705 | 1/1 · .5517 | .5151 / .4741 | INVALID (moved up-rate validity check failed) / INVALID (moved up-rate validity check failed) | no (15/15, 15 clear) / no (4/15, 2 clear) | 0.73 / 0.50 | `deriv_e60stack_USDJPY_15m_result.json` |
| e61ord4 | e61ord4: all (66), nseed 1 | 15m | 0.03 | .5257 (.5168) | .5365 · .867 · .5860 | 2/3 · .5690 | .4272 · .333 · .5212 | 0/0 · — | .5151 / .4741 | INVALID (moved up-rate validity check failed) / INVALID (moved up-rate validity check failed) | no (15/15, 13 clear) / no (6/15, 2 clear) | 0.73 / 0.33 | `deriv_e61ord4_USDJPY_15m_result.json` |
| e62har | e62har: all (66), nseed 1 | 15m | 0.03 | .5256 (.5141) | .4661 · .786 · .5899 | 1/2 · .5406 | .4343 · .167 · .4788 | 0/0 · — | .5151 / .4741 | INVALID (moved up-rate validity check failed) / INVALID (moved up-rate validity check failed) | no (14/15, 11 clear) / no (6/15, 1 clear) | 0.79 / 0.17 | `deriv_e62har_USDJPY_15m_result.json` |
| e63bootnys | e63bootnys: all (66), nseed 1 | 15m | 0.03 | .5246 (.5163) | .5489 · .867 · .5844 | 3/4 · .6233 | .5252 · .500 · .5517 | 0/0 · — | .5151 / .4741 | INVALID (moved up-rate validity check failed) / INVALID (moved up-rate validity check failed) | no (15/15, 13 clear) / no (2/15, 1 clear) | 0.80 / 0.50 | `deriv_e63bootnys_USDJPY_15m_result.json` |
| e64boot | e64boot: all (66), nseed 1 | 15m | 0.03 | .5259 (.5165) | .5493 · .933 · .5836 | 2/3 · .5454 | .4640 · .444 · .5550 | 1/1 · .6667 | .5151 / .4741 | INVALID (moved up-rate validity check failed) / INVALID (moved up-rate validity check failed) | no (15/15, 14 clear) / no (9/15, 4 clear) | 0.67 / 0.33 | `deriv_e64boot_USDJPY_15m_result.json` |
| e65recency | e65recency: all (66), nseed 1 | 15m | 0.03 | .5252 (.5148) | .5385 · .786 · .6200 | 1/3 · .5238 | .5000 · .400 · .5308 | 1/2 · .5390 | .5151 / .4741 | INVALID (moved up-rate validity check failed) / INVALID (moved up-rate validity check failed) | no (14/15, 11 clear) / no (5/15, 2 clear) | 0.64 / 0.20 | `deriv_e65recency_USDJPY_15m_result.json` |
| e66mh | e66mh: all (66), nseed 1 | 15m | 0.03 | .5257 (.5158) | .5609 · .917 · .6064 | 2/2 · .7139 | .5567 · 1.0 · .5778 | 0/0 · — | .5151 / .4741 | INVALID (moved up-rate validity check failed) / INVALID (moved up-rate validity check failed) | no (12/15, 11 clear) / no (3/15, 3 clear) | 0.83 / 0.67 | `deriv_e66mh_USDJPY_15m_result.json` |
| e67drfe | e67drfe: all (66), nseed 1 | 15m | 0.03 | .5209 (.4990) | .5396 · .867 · .5864 | 3/4 · .6046 | .4735 · .133 · .5064 | 1/4 · .5042 | .5151 / .4741 | INVALID (moved up-rate validity check failed) / INVALID (moved up-rate validity check failed) | no (15/15, 13 clear) / no (15/15, 2 clear) | 0.73 / 0.00 | `deriv_e67drfe_USDJPY_15m_result.json` |
| e67drfenys | e67drfenys: all (66), nseed 1 | 15m | 0.03 | .5265 (.5137) | .5390 · .846 · .5781 | 3/4 · .5944 | .4997 · .200 · .5289 | 0/2 · .5237 | .5151 / .4741 | INVALID (moved up-rate validity check failed) / INVALID (moved up-rate validity check failed) | no (13/15, 11 clear) / no (10/15, 2 clear) | 0.62 / 0.20 | `deriv_e67drfenys_USDJPY_15m_result.json` |
| e68agree | e68agree: all (66), nseed 1 | 15m | 0.03 | .5258 (.5159) | .5471 · .933 · .5862 | 1/3 · .5304 | .4424 · .500 · .5326 | 1/1 · .6250 | .5151 / .4741 | INVALID (moved up-rate validity check failed) / INVALID (moved up-rate validity check failed) | no (15/15, 14 clear) / no (10/15, 5 clear) | 0.73 / 0.40 | `deriv_e68agree_USDJPY_15m_result.json` |
| e69agreenys | e69agreenys: all (66), nseed 1 | 15m | 0.03 | .5261 (.5164) | .5594 · 1.0 · .5943 | 4/4 · .5884 | .4459 · .000 · .4736 | 1/1 · .6364 | .5151 / .4741 | INVALID (moved up-rate validity check failed) / INVALID (moved up-rate validity check failed) | no (15/15, 15 clear) / no (2/15, 0 clear) | 0.87 / 0.00 | `deriv_e69agreenys_USDJPY_15m_result.json` |
| e70play | e70play: all+fam:hy (75), nseed 1 | 15m | 0.03 | .5251 (.5140) | .5331 · .733 · .5766 | 2/3 · .5511 | .4914 · .333 · .5223 | 1/1 · .6207 | .5151 / .4741 | INVALID (moved up-rate validity check failed) / INVALID (moved up-rate validity check failed) | no (15/15, 11 clear) / no (6/15, 2 clear) | 0.67 / 0.17 | `deriv_e70play_USDJPY_15m_result.json` |
| e71playreset | e71playreset: all+fam:hr (75), nseed 1 | 15m | 0.03 | .5264 (.5162) | .5354 · .800 · .5924 | 1/3 · .5373 | .4494 · .625 · .5210 | 1/1 · .6452 | .5151 / .4741 | INVALID (moved up-rate validity check failed) / INVALID (moved up-rate validity check failed) | no (15/15, 12 clear) / no (8/15, 5 clear) | 0.73 / 0.25 | `deriv_e71playreset_USDJPY_15m_result.json` |
| e72shock | e72shock: all+fam:d2 (67), nseed 1 | 15m | 0.03 | .5260 (.5163) | .5631 · 1.0 · .6091 | 3/3 · .6176 | .4764 · .500 · .5783 | 1/1 · .6333 | .5151 / .4741 | INVALID (moved up-rate validity check failed) / INVALID (moved up-rate validity check failed) | no (15/15, 15 clear) / no (6/15, 3 clear) | 0.87 / 0.50 | `deriv_e72shock_USDJPY_15m_result.json` |
| e73mono | e73mono: all (66), nseed 1 | 15m | 0.03 | .5254 (.5148) | .5475 · .933 · .5782 | 3/3 · .5650 | .4361 · .500 · .5297 | 1/1 · .6471 | .5151 / .4741 | INVALID (moved up-rate validity check failed) / INVALID (moved up-rate validity check failed) | no (15/15, 14 clear) / no (10/15, 5 clear) | 0.73 / 0.40 | `deriv_e73mono_USDJPY_15m_result.json` |
| e74logitsig | e74logitsig: all (66), nseed 1 | 15m | 0.03 | .5148 (.4971) | .4980 · .600 · .5466 | 3/4 · .5611 | .3833 · .000 · .4238 | 1/3 · .4661 | .5151 / .4741 | INVALID (moved up-rate validity check failed) / INVALID (moved up-rate validity check failed) | no (15/15, 9 clear) / no (4/15, 0 clear) | 0.40 / 0.00 | `deriv_e74logitsig_USDJPY_15m_result.json` |
| e75lgbsig | e75lgbsig: all (66), nseed 1 | 15m | 0.03 | .5173 (.5051) | .5274 · .667 · .5635 | 2/4 · .5443 | .4691 · .533 · .5461 | 1/3 · .5104 | .5151 / .4741 | INVALID (moved up-rate validity check failed) / INVALID (moved up-rate validity check failed) | no (15/15, 10 clear) / no (15/15, 8 clear) | 0.47 / 0.33 | `deriv_e75lgbsig_USDJPY_15m_result.json` |
| e76drfelgb | e76drfelgb: all (66), nseed 1 | 15m | 0.03 | .5204 (.5081) | .5367 · .800 · .5679 | 2/4 · .5161 | .4783 · .333 · .5241 | 1/4 · .5287 | .5151 / .4741 | INVALID (moved up-rate validity check failed) / INVALID (moved up-rate validity check failed) | no (15/15, 12 clear) / no (15/15, 5 clear) | 0.47 / 0.13 | `deriv_e76drfelgb_USDJPY_15m_result.json` |
| e77drfenysabs | e77drfenysabs: all (66), nseed 1 | 15m | 0.03 | .5204 (.5069) | .5119 · .643 · .5668 | 2/4 · .5741 | .4486 · .333 · .5206 | 1/2 · .5250 | .5151 / .4741 | INVALID (moved up-rate validity check failed) / INVALID (moved up-rate validity check failed) | no (14/15, 9 clear) / no (9/15, 3 clear) | 0.57 / 0.11 | `deriv_e77drfenysabs_USDJPY_15m_result.json` |
| e78nysagree | e78nysagree: all (66), nseed 1 | 15m | 0.03 | .5046 (.4979) | nan · .000 · — | 0/2 · .4793 | .4553 · .000 · .4833 | 0/4 · .4634 | .5151 / .4741 | INVALID (moved up-rate validity check failed) / INVALID (moved up-rate validity check failed) | no (0/15, 0 clear) / no (15/15, 0 clear) | — / 0.00 | `deriv_e78nysagree_USDJPY_15m_result.json` |
| e79qorder | e79qorder: all+fam:qo (88), nseed 1 | 15m | 0.03 | .5250 (.5108) | .5068 · .533 · .5528 | 3/3 · .6002 | .4760 · .333 · .5400 | 0/0 · — | .5151 / .4741 | INVALID (moved up-rate validity check failed) / INVALID (moved up-rate validity check failed) | no (15/15, 8 clear) / no (6/15, 2 clear) | 0.40 / 0.33 | `deriv_e79qorder_USDJPY_15m_result.json` |
| e7fitall | e7fitall: all (66), nseed 1 | 15m | 0.03 | .5275 (.5179) | .5167 · .867 · .5848 | 2/4 · .5575 | .5085 · .750 · .5923 | 0/0 · — | .5151 / .4741 | INVALID (moved up-rate validity check failed) / INVALID (moved up-rate validity check failed) | no (15/15, 13 clear) / no (8/15, 6 clear) | 0.67 / 0.75 | `deriv_e7fitall_USDJPY_15m_result.json` |
| e80qperm | e80qperm: all+fam:qp (88), nseed 1 | 15m | 0.03 | .5237 (.5117) | .5413 · .867 · .5832 | 1/3 · .5177 | .4947 · .667 · .5495 | 1/1 · .7308 | .5151 / .4741 | INVALID (moved up-rate validity check failed) / INVALID (moved up-rate validity check failed) | no (15/15, 13 clear) / no (6/15, 4 clear) | 0.67 / 0.50 | `deriv_e80qperm_USDJPY_15m_result.json` |
| e81tail0 | e81tail0: all (66), nseed 1 | 15m | 0.03 | .5267 (.5194) | .5188 · .857 · .5970 | 3/3 · .6451 | nan · .000 · — | 0/0 · — | .5151 / .4741 | INVALID (moved up-rate validity check failed) / INVALID (moved up-rate validity check failed) | no (14/15, 12 clear) / no (0/15, 0 clear) | 0.79 / — | `deriv_e81tail0_USDJPY_15m_result.json` |
| e82tailaux | e82tailaux: all (66), nseed 1 | 15m | 0.03 | .5259 (.5162) | .5557 · .933 · .6170 | 3/3 · .6644 | .5263 · .000 · .5263 | 0/0 · — | .5151 / .4741 | INVALID (moved up-rate validity check failed) / INVALID (moved up-rate validity check failed) | no (15/15, 14 clear) / no (1/15, 0 clear) | 0.80 / 0.00 | `deriv_e82tailaux_USDJPY_15m_result.json` |
| e8tb2 | e8tb2: all (66), nseed 1 | 15m | 0.03 | .5243 (.5034) | .5575 · 1.0 · .5829 | 2/3 · .5721 | .4304 · .714 · .5223 | 0/0 · — | .5151 / .4741 | INVALID (moved up-rate validity check failed) / INVALID (moved up-rate validity check failed) | no (15/15, 15 clear) / no (7/15, 5 clear) | 0.67 / 0.14 | `deriv_e8tb2_USDJPY_15m_result.json` |
| e9blockval | e9blockval: all (66), nseed 1 | 15m | 0.03 | .5242 (.5111) | .5671 · .933 · .5945 | 4/4 · .5946 | .4000 · .273 · .4926 | 1/4 · .5247 | .5151 / .4741 | INVALID (moved up-rate validity check failed) / INVALID (moved up-rate validity check failed) | no (15/15, 14 clear) / no (11/15, 3 clear) | 0.87 / 0.18 | `deriv_e9blockval_USDJPY_15m_result.json` |
| r2 | r2: list:USDJPY.m15ny_seedens.v1.json (239), nseed 3 | 15m | 0.02 | .5263 (.5047) | .6200 · 1.0 · .6589 | 0/0 · — | .5698 · 1.0 · .5698 | 0/0 · — | .5121 / .4771 | CERTIFIED / CERTIFIED | no (7/15, 7 clear) / no (1/15, 1 clear) | 1.00 / 1.00 | `deriv_r2_USDJPY_15m_result.json` |
| r3 | r3: list:USDJPY.m15ny_seedens.v1.json (239), nseed 3 | 15m | 0.02 | .5222 (.5124) | .6800 · 1.0 · .6800 | 0/0 · — | nan · .000 · — | 0/0 · — | .5196 / .4660 | INVALID (moved up-rate validity check failed) / INVALID (moved up-rate validity check failed) | no (1/15, 1 clear) / no (0/15, 0 clear) | 1.00 / — | `deriv_r3_USDJPY_15m_result.json` |
| r4fixldn | r4fixldn: rule:fx_ldn_prefix_u60 (1), nseed 0 | 15m | 1.0 | .5287 (.4954) | .5022 · .467 · .5396 | 0/0 · — | .4838 · .133 · .5107 | 0/0 · — | .5151 / .4741 | INVALID (moved up-rate validity check failed) / INVALID (moved up-rate validity check failed) | no (15/15, 7 clear) / no (15/15, 2 clear) | 0.27 / 0.00 | `deriv_r4fixldn_USDJPY_15m_result.json` |
| r5fixecb | r5fixecb: rule:fx_ecb_prefix_u60 (1), nseed 0 | 15m | 1.0 | .4858 (.4519) | .5064 · .333 · .5336 | 0/0 · — | .4499 · .000 · .4753 | 0/0 · — | .5151 / .4741 | INVALID (moved up-rate validity check failed) / INVALID (moved up-rate validity check failed) | no (15/15, 5 clear) / no (15/15, 0 clear) | 0.20 / 0.00 | `deriv_r5fixecb_USDJPY_15m_result.json` |
<!-- DERIV-FEED-TABLES:END -->
