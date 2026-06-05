> **SCOPE: USDJPY · 1m** (key-specific sweep LEDGER — resumable state of the exhaustive sweep). Generic menu: `SWEEP_MATRIX.md`. Results of record: `USDJPY_RESULTS.md`. Executable backlog: `sweeps/USDJPY_1m_backlog.md`.

# USDJPY × 1m — Sweep Ledger

**Goal:** best (USDJPY,1m,UP) + best (USDJPY,1m,DOWN) binary-direction predictor, OOS-certified. Breakeven **0.541**. Splits: train 2012-21 / val 2022-23 / test 2024 / test 2025 / oos 2026. Settlement: bar-close approx (no USDJPY tick cache), ties LOSE, gap=60s `nonoverlap_chrono`, per-year CI95, moved-bars-only, selection on VAL worst-half.

**Honest prior (from EURUSD program):** 60s/1m direction is **near-efficient** — EURUSD ~0.50–0.51 AUC across ~24 channels; cross-pair edge gradient **none@60s → UP@5m → BOTH@10m,15m,30m**. So design fast-KILL falsifiers. BUT USDJPY is a structurally different pair (carry/risk proxy, BoJ-driven trends 2022-24, MoF intervention spikes) → measure directly; nulls are redirects (ATTITUDE).

## DATA AVAILABILITY (USDJPY) — gates which rows can run
- ✅ **Bar features** `features/USDJPY_2012..2026.parquet` (239 multi-TF feats, 1-min bars). 1-min own-clock moved up-rate ∈ [0.500,0.507] all splits (tripwire clean).
- ✅ **Order-flow bar features** `features_of/USDJPY_*.parquet` (18 OF cols) → `xpof` cross-pair mode runnable.
- ✅ **Cross-pair legs** all 7 majors present in `features/` → USDJPY-as-target cross-pair block buildable.
- ❌ **No `features_tick_USDJPY/`** (1s tick cache) → **DATA-BLOCKED**: Tier-B tick microstructure (B1,B2,B5,B7), tick-substrate Tier-C (the `min1_*` HMM/Kalman/RMT/CCM/online scripts read the 1s cache), Tier-D sequence/DL on raw tick path. These are not modeling rows here — they are data-acquisition prereqs. Bar-level analogs may substitute.

## LEDGER (work top→bottom; one row per method×variant actually run)
| id | family | method / variant | script | tgt | prior | status | combined_oos | up_oos | down_oos | verdict | result_json |
|----|--------|------------------|--------|-----|-------|--------|--------------|--------|----------|---------|-------------|
| BASE | A1 | single-pair LGBM, 239 feats, 1m own-clock label (the foundation) | `usdjpy_1m_base.py` | D | — | **killed** | AUC .516; COMB wr .515 (CI<BE) | .518@cov2 / **.545@cov1** (CI-lo<BE) | .513 dead | KILLED as standalone edge; UP monotone-confidence tilt = redirect | `usdjpy_1m_base_result.json` |
| A1a | A1 | more data + capacity: stride6 (592k) × leaves255 | `usdjpy_1m_base.py 6 255` | D | med | **done (improves, still <BE)** | AUC .523 | cov2% UP .547/.534/.538 (worst .534) | dead | signal was partly DATA-STARVED; more data lifts UP tail ~+1pt, worst year still <0.541 | `usdjpy_1m_base_s6_l255_result.json` |
| A2a | A2 | compression×session×coverage gate sweep | new (bar) | G+D | med | pending | | | | | |
| A3a | A3 | compression-release × reversion specialist | bar analog of `min1_production` levers | D | med | pending | | | | | |
| A4a | A4 | meta-labeler on orthogonal axes | bar analog | G+D | med | pending | | | | | |
| A5a | A5 | cross-horizon stack (USDJPY 5m/10m/15m parent → 1m) | needs USDJPY parents first | D | low@1m | pending | | | | | |
| A6a | A6 | **cross-pair USD-residual, USDJPY-target** (xpof, stride8) | `usdjpy_xpair.py` | D | low@1m | **killed** | AUC .519/.516/.519; cov2% COMB .532/.526/.512 | cov2% UP .534/.526/.519; cov1% .538/.522/.513 | cov2% .531/.520/**.477** | KILLED — lifts AUC + mid-cov 24/25 but DILUTES the OOS tail (cov1% UP .513 < baseline .545); lead-lag regime-dependent | `usdjpy_1m_xpair_xpof_result.json` |
| A8a | A8 | up/down side split (FILTER) of BASE/best book | bar analog of `min1_updown` | U/Dn | filter med | pending | | | | | |
| A8b | A8 | purpose-built UP / DOWN specialist | bar analog of `min1_upspec` | U/Dn | ~null | pending | | | | | |
| E1a | E1 | magnitude \|ret60\|≥Q classifier (sign-invariant; → MAGNITUDE_FINDINGS) | bar analog | M | high (cert elsewhere) | pending | | | | | |
| F2a | F2 | price-action / RSI2 / BB%b / NR7 rules | bar | D | ~null | pending | | | | | |
| F4a | F4 | residualized TARGET (label = USD-residual sign) | bar analog of `min1_residtarget` | D | ~null | pending | | | | | |
| —  | B/C(tick)/D | tick microstructure / state-space-on-tick / seq-DL | — | D | — | **BLOCKED** | — | — | — | no USDJPY tick cache (data-acq prereq) | — |

## Tier-N (discovered) rows — appended by discovery rounds (see backlog)
_(none yet — discovery starts after Tier-A baseline establishes the landscape)_

## Tier-I (improve) — apply to any edge that survives
_(pending — only after an edge is found)_

## Discovery rounds (loop until K=2 dry)
_(pending)_

## STATUS: bootstrap in progress — baseline running. Next: record baseline, instantiate Tier-A heavy rows serially.
