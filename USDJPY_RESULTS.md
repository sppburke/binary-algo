> **SCOPE: USDJPY** (key-specific — all timeframes × sides; results of record + UP/DOWN leaderboard). Generic methods: METHODS_CATALOG.md. Sweep menu: SWEEP_MATRIX.md. See REPO_MAP.md. Bootstrapped 2026-06-04 (new currency; structure copied from EURUSD_RESULTS.md).

# USDJPY — Results Ledger (unique key: currency × timeframe × side)

**Pair tag: `USDJPY`.** USDJPY is a **USD-BASE** pair (USDJPY up ⇒ USD up ⇒ JPY down). Carry/risk-on-off proxy; 2022–2024 BoJ-driven uptrend + MoF intervention spikes → distinct microstructure vs EURUSD.

**The unique result key is `(currency, timeframe, side)`, side ∈ {UP, DOWN}.** Where a key is unmeasured it is `UNTESTED`. COMBINED is context only, never an UP/DOWN number.

**Conventions.** Deriv-faithful settlement (mid-to-mid, ties LOSE), breakeven **0.541**. Splits: bars train 2012-21 / val 2022-23 / test 2024 / test 2025 / oos 2026. Selection on VAL worst-half (never VAL-acc-max). Moved-bars-only; per-year CI95. **No USDJPY 1s tick cache** → 1m is BAR-based (entry≈close[t], exit≈close[t+1]; mild optimistic proxy of a tick-tradeable 60s binary). Deriv forex Rise/Fall minimum expiry = 15m → **1m is a research/synthetic-index horizon** (not deriv-forex-deployable; deployable on synthetic-index 1m or as a research result). Current scope: **1m only**. Last updated 2026-06-04 (bootstrap + baseline).

---

## MASTER KEY TABLE — one row per (USDJPY, timeframe, side)

| Key (currency · timeframe · side) | **Best OOS % (2026)** | Model id · content_id | Description | Status |
|---|---|---|---|---|
| USDJPY · **1m** · UP | **~0.518–0.545** ⚠ sub-breakeven (best @cov1%) | single-pair LGBM up-preds (`usdjpy_1m_base.py`) | Faint MONOTONE-in-confidence UP edge: cov2% .533/.531/.518, cov1% .538/.540/**.545**; OOS-persistent but CI-lo never clears 0.541. Base GBM AUC 0.52 = weak. **Sweep in progress** | **MEASURED (baseline), sub-breakeven — improving** |
| USDJPY · **1m** · DOWN | **~0.51** ❌ dead | single-pair LGBM down-preds (`usdjpy_1m_base.py`) | No monotone lift; ≤0.51 all covers, <0.50 at tight coverage / OOS. **Sweep in progress** | **MEASURED (baseline), dead — confirming via (a)-(e)** |
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

### Key results (MEASURED — baseline)
| Key | Result (2024 / 2025 / 2026) | Method | Status |
|---|---|---|---|
| **(USDJPY, 1m, UP)** | .533 / .531 / .518 @cov2%; **.538 / .540 / .545 @cov1%** | up-preds of single-pair LGBM (`usdjpy_1m_base.py`) | ⚠ sub-breakeven; faint monotone-confidence edge, OOS-persistent; **best AVAILABLE, not certified.** Improve levers (cross-pair, more data, gate, mag-bridge) pending |
| **(USDJPY, 1m, DOWN)** | .511 / .511 / .513 @cov2%; ≤.508 @cov1% | down-preds of single-pair LGBM | ❌ dead — no monotone lift; confirming via specialist (c) + discovery before declaring exhausted |

**Mechanism (v1 model of the edge):** USDJPY 1m direction is near-efficient; the only signed structure is a faint UP-autocorrelation/dip-buy tilt (UP carries, DOWN dead — same asymmetry as EURUSD 60s, but UP fades slightly into 2026 rather than strengthening). Base GBM AUC 0.52 caps it below breakeven. The "below-breakeven" cause = weak base signal → attack via: stronger UP channel (cross-pair USD-factor — USDJPY has DIRECT USD exposure + on-disk OF), more training data / tuning, regime gate, and magnitude→direction bridge. Citations: `usdjpy_1m_base_result.json`.

---

## How to maintain this file
- One row per key; never copy a COMBINED number into an UP/DOWN key; never copy one timeframe/side's number to another.
- Every number traces to an on-disk result JSON (Tier-1). Flag n<50 + VAL-acc-max as non-robust.
- Unseat a side-leader only if the challenger beats the incumbent's binding (worst held-out) year with CI95-lower clearing it under the discipline, ideally refit-CPCV-certified.
- Update the master table + leaderboard + `sweeps/USDJPY_1m.md` ledger + `sweeps/USDJPY_1m_backlog.md` together; commit often.
