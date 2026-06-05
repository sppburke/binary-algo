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
| A2a | A2 | compression×session×coverage gate sweep | (covered by A3a/A8a regime) | G+D | med | **killed** | — | — | — | SUBSUMED — the regime run IS the compression×session×coverage sweep; OOS-collapses (.495). No gate clears. | `usdjpy_1m_regime_s6_result.json` |
| A3a/A8a | A3+A8 | regime-gated side FILTER: dip×comp×Tokyo (UP) / rally (DOWN) × model-conf, on s6/l255 symmetric GBM | `usdjpy_1m_regime.py` | U/Dn | med | **killed** | — | UP .524/.529/**.495** (VAL gate .558 → OOS collapse) | DOWN .518/.515/.513 | KILLED both — model confidence WITHIN regime ANTI-TRANSFERS (corr(VAL,OOS)=−.54); only stable signal = raw regime base-rate ~.52 < BE | `usdjpy_1m_regime_s6_result.json` |
| A4a | A4 | meta-labeler on orthogonal axes | — | G+D | med | **subsumed** | — | — | — | A meta-labeler gates on a learned confidence model; base confidence ALREADY OOS-anti-transfers (regime kill, corr(VAL,OOS)=−.54) → meta-labeler inherits the anti-transfer. No confidence gate clears. | `usdjpy_1m_regime_s6_result.json` |
| A5a | A5 | cross-horizon stack (USDJPY 5m/10m/15m parent → 1m) | — | D | low@1m | **pruned** | — | — | — | SCOPE-BLOCKED: needs USDJPY 5m/10m/15m parent books (none exist; 1m-only scope) + EURUSD 1m stack (`min1_stack`) was ❌. Low prior; logged not silently skipped. | — |
| A6a | A6 | **cross-pair USD-residual, USDJPY-target** (xpof, stride8) | `usdjpy_xpair.py` | D | low@1m | **killed** | AUC .519/.516/.519; cov2% COMB .532/.526/.512 | cov2% UP .534/.526/.519; cov1% .538/.522/.513 | cov2% .531/.520/**.477** | KILLED — lifts AUC + mid-cov 24/25 but DILUTES the OOS tail (cov1% UP .513 < baseline .545); lead-lag regime-dependent | `usdjpy_1m_xpair_xpof_result.json` |
| A8a | A8 | up/down side split (FILTER) of BASE/best book | bar analog of `min1_updown` | U/Dn | filter med | pending | | | | | |
| A8b | A8 | purpose-built UP / DOWN specialist (subset-trained: after-dip / after-rally) | `usdjpy_1m_improve.py spec` | U/Dn | ~null | **killed** | — | UP-spec cov2% .531/.528/.524 (WORSE than filter .547/.534/.538) | DOWN-spec .515/.519/.517 (sub-BE) | KILLED — subset-training destroys ranking (EURUSD finding reproduced); both forms of step (c) fail | `usdjpy_1m_improve_spec_result.json` |
| E1a | E1 | magnitude \|ret60\|≥Q classifier (sign-invariant; → MAGNITUDE_FINDINGS) | `usdjpy_1m_magnitude.py` | M | high | **done (strong)** | magAUC OOS .716/.730/.790 (Q67/75/90), lift ~2.1-2.5x | n/a (sign-invariant) | n/a | ✅ real SIZE edge, confirms sign-invariance (dirAUC ~.52 vs magAUC ~.79); recorded MAGNITUDE_FINDINGS [USDJPY·1m] | `usdjpy_1m_magnitude_result.json` |
| F2a | F2 | price-action / RSI2 / BB%b / NR7 rules | — | D | ~null | **subsumed** | — | — | — | SUBSUMED — base 239 already include RSI/BB-width/rangepos/NR-type feats; the GBM uses them and caps at AUC .52; standalone rules are a strict subset. (EURUSD F2 null.) | `usdjpy_1m_base_result.json` |
| F4a | F4 | residualized TARGET (label = USD-residual sign) | — | D | ~null | **subsumed** | — | — | — | SUBSUMED — cross-pair residual (returns A6a + OF N3) carries NO OOS sign (DOWN .477/.480); residual-label is the same channel relabeled (EURUSD `min1_residtarget` ❌). | `usdjpy_1m_ofresid_s8_result.json` |
| —  | B/C(tick)/D | tick microstructure / state-space-on-tick / seq-DL | — | D | — | **BLOCKED** | — | — | — | no USDJPY tick cache (data-acq prereq) | — |

## Tier-N (discovered, round-1 8-agent workflow) rows
| id | method | script | tgt | prior | status | result | result_json |
|----|--------|--------|-----|-------|--------|--------|-------------|
| UJ-N1/N2 | signed spike×rangepos reversion + signed path-state (cumret/run-len/dist-from-extreme), +16 feats on s6/l255 | `usdjpy_1m_revspec.py` | both | .10/.08 | **killed/subsumed** | VAL AUC .5235 UNCHANGED; new feats #38-90/254; UP .556/.523/.527 (worst↓), DOWN OOS .503 | `usdjpy_1m_revspec_s6_result.json` |
| UJ-N3 | cross-pair OF-basket + RESIDUAL (own OF − USD-up-basket OF) | `usdjpy_1m_ofresid.py` | both | .06 | **killed** | VAL AUC .5219 (≤base); OF feats rank #15-25 but no net gain; UP cov2% .538/.528/.519, DOWN cov2% .533/.518/**.480** (OOS collapse) | `usdjpy_1m_ofresid_s8_result.json` |
| UJ-N4 | realized signed-semivariance skew RS+−RS− (thin cks_e tick) | — | both | .05 | **subsumed** | cks1s cache has only cks_e(signed event)+cks_nev — NO sub-bar prices/returns → classical RS-skew not buildable; buildable cks_e-variance-asymmetry = signed-OF channel already KILLED (UJ-N3, features_of). OOS only 2026-02..05 (can't certify). Tier-1 subsumption | (cks1s inspection) |
| UJ-N5 | gotobi/Tokyo-fix calendar as GBM conditioning feature | `usdjpy_1m_gotobi.py` | UP | .04 | **killed** | VAL AUC rise +.0004 (<.003); gotobi/near_fix rank #197/#245; UP cov2% .543/.529/.523, DOWN OOS .481 | `usdjpy_1m_gotobi_s6_result.json` |

## Tier-I (improve) — run on the best base (s6/l255) even though sub-BE
| lever | script | result | verdict |
|---|---|---|---|
| I2 seed-ensemble K=5 | `usdjpy_1m_improve.py seedens` | VAL AUC .5238; UP cov2% .551/.532/.536 (worst .532); DOWN .524/.542/.494 | ❌ no lift — variance reduction can't close the ~1pp gap; wall is signal-level. `usdjpy_1m_improve_seedens_result.json` |
| I1 ACI gate | — | **moot/subsumed**: ACI only reallocates coverage to target a win-rate the model can't hit OOS; the regime-gate run already proved coverage reallocation OOS-collapses (.495). No fixed gate clears → adaptive gate cannot either. |
| I4 calibration | — | **moot**: gate is sub-BE pre/post-calibration; calibration doesn't add signal. |
| I3 \|ret\|-weight/GMADL | — | **subsumed** by magnitude sign-invariance (magnitude finds big moves w/ zero direction; mag→dir bridge null, EURUSD [60s] `min1_magdir`) + revspec signed feats already null. |

## Discovery rounds (loop until dry)
- **Round 1** (8-agent workflow `usdjpy-1m-discover`): surfaced 5 runnable Tier-N (UJ-N1..N5) — ALL run-and-killed/subsumed (see Tier-N table); 5 data-blocked (Tier-G). NOT dry (had survivable candidates) → ran them.
- **Round 2** (2-agent workflow `usdjpy-1m-discover-r2`, given full kill-list): **BOTH agents DRY.** No genuinely-new on-disk-runnable signed direction lever exists. Independently confirmed via their own Tier-1 probes (WMR 4pm-fix reversal .509/.512/.501; Krohn-Mueller-Whelan seasonal W-pattern collapses OOS .490; per-channel dirAUCs all ~.485-.50; cross-pair/point-process/label-engineering families subsumed by run kills + sign-invariance). Only frontier = DATA-BLOCKED external inputs.
- **VERDICT: discovery loop DRY** (round-2 dry + all round-1 survivors killed + EURUSD 1m's own 2 dry rounds cited). A round 3 would re-return dry given the mechanism-level subsumption.

## STATUS: COMPLETE (2026-06-04). All ledger rows done/killed/subsumed/pruned (logged, none silently skipped). Both sides honestly EXHAUSTED — near-efficient, no certified/deployable direction edge. Full (a)-(e) shown (e never triggered: no worst-year CI-lo ≥ 0.541). Tier-N R1 killed + R2 DRY → discovery dry. Tier-I seed-ens null, rest subsumed. Magnitude STRONG (sign-invariant, MAGNITUDE_FINDINGS). Best-available UP (uncertified, not deployable): `usdjpy_1m_base.py 6 255` up-preds cov2% .547/.534/.538. DOWN dead. Only frontier = external data (Tier-G). Final verdict + deployment spec → `USDJPY_RESULTS.md`.
