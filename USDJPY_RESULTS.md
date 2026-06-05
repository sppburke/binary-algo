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
| USDJPY · 5m/10m/15m/30m · UP/DOWN | `UNTESTED` | — | Out of current scope (goal = 1m). Bar data present; bootstrap when scoped. | UNTESTED |

Provenance: `usdjpy_1m_base_result.json` (Tier-1).

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

## DEPLOYMENT SPEC / FINAL CONCLUSION — USDJPY 1m, both sides (2026-06-04; improve+discover loops dry)

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

---

## How to maintain this file
- One row per key; never copy a COMBINED number into an UP/DOWN key; never copy one timeframe/side's number to another.
- Every number traces to an on-disk result JSON (Tier-1). Flag n<50 + VAL-acc-max as non-robust.
- Unseat a side-leader only if the challenger beats the incumbent's binding (worst held-out) year with CI95-lower clearing it under the discipline, ideally refit-CPCV-certified.
- Update the master table + leaderboard + `sweeps/USDJPY_1m.md` ledger + `sweeps/USDJPY_1m_backlog.md` together; commit often.
