> **SCOPE: USDJPY** (key-specific — all timeframes × sides; results of record + UP/DOWN leaderboard). Generic methods: docs/METHODS_CATALOG.md. Sweep menu: SWEEP_MATRIX.md. See REPO_MAP.md. Bootstrapped 2026-06-04 (new currency; structure copied from EURUSD_RESULTS.md).

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
