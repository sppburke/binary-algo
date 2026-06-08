> **SCOPE: USDJPY · 15m** (key-specific SWEEP LEDGER — resumable row-by-row state of the exhaustive direction sweep). Generic menu: SWEEP_MATRIX.md. Methods: METHODS_CATALOG.md. Results of record: USDJPY_RESULTS.md. Backlog/discovery: sweeps/USDJPY_15m_backlog.md. Opened 2026-06-08.

# USDJPY × 15m — Direction Sweep Ledger (UP & DOWN, symmetric)

**Goal:** certify the best 15m UP and best 15m DOWN USDJPY binary-direction predictor (deriv-faithful, ties LOSE, breakeven 0.541), OOS-verified; stretch target win-rate >0.70. Two deliverables, each certified-or-honestly-exhausted. Don't stop until both sides certified-or-exhausted and the improve+discover loops are dry.

**Why 15m is the strongest USDJPY direction prospect:**
- 15m is the **deriv-FX-deployable minimum** expiry (deriv forex Rise/Fall floor) AND the program's **strongest direction horizon**: EURUSD 15m cross-pair POOLED certified BOTH sides via refit-CPCV (UP p10 .567 / DOWN p10 .574 — the program's best DOWN edge).
- The cross-pair sign gradient is **horizon-gated: none@60s → UP@5m → BOTH@10m & 15m** (EURUSD keystone). USDJPY 2m pooling lifted the mean above the 1m floor but p10 saturated ~.531 (sub-BE). The gradient predicts **USDJPY 15m pooling should cross BE**.
- Direction edge is **NY-session-concentrated** (EURUSD A9: 15m NY cross-pair .5845/.5712, LDN/Asia null).
- Prior hook: a cross-horizon probe at 1m showed USDJPY H=15m UP 2024 .593 (cleared!) but 2025/OOS sub-BE (`USDJPY_RESULTS.md:116`) — a 15m-specific signal worth chasing directly.

**Discipline (every row):** deriv-faithful settlement (bar-close approx, ties LOSE), `nonoverlap_chrono` gap=900s, train 2012-21 / val 2022-23 / test 2024 / test 2025 / oos 2026, selection on VAL worst-half (never VAL-acc-max), moved-bars only + up-rate tripwire ∈ [0.47,0.53], pre-registered falsifier in result JSON BEFORE OOS, COMBINED + UP-split + DOWN-split scored per-year with CI95. Certify ONLY via per-fold-refit CPCV (p10 ≥ 0.541 AND ≥80% folds clear). One heavy fit at a time. Commit after each row.

**Retarget mechanics:** clones of the proven `usdjpy_2m_*` templates with `HOR=15` (GAP=900s = 15 clean 1-min bars): `usdjpy_15m_base.py` [stride] [leaves], `usdjpy_15m_xpair.py pool [stride] [leaves]`, `usdjpy_15m_cpcv.py [pool_stride] [cov] [nseed]`. Feature set = 239 base feats (`H.feature_cols`).

---

## STATUS LEGEND
`pending` (not run) · `running` · `done` (ran, recorded) · `killed` (ran + failed falsifier) · `subsumed` (Tier-1 dominating variant ran+failed). Every `done`/`killed` row cites its result JSON.

## LEDGER — Tier A → F (ROI-ordered), then Tier-I improve, then Tier-N discovered

| id | family | method | variant | script | target | prior | status | combined_oos | up_oos | down_oos | verdict | result_json |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A1 | GBM core | single-pair LGBM base | s6/l127 | usdjpy_15m_base.py 6 127 | D | med | **done** | .538 [.498,.579] | **.601/.566/.552** (24/25/26) | .541/.519/.520 | ⚠ SURVIVED (2024 COMB CI-lo clears); UP real-but-thin (pt-est >BE all yrs, CI-lo clears only 2024); DOWN ~BE. VAL-AUC .5312, up-rate trip OK. | usdjpy_15m_base_result.json |
| A1b | GBM core | single-pair LGBM base | s3/l255 (more data+capacity) | usdjpy_15m_base.py 3 255 | D | med | pending | — | — | — | — | usdjpy_15m_base_s3_l255_result.json |
| A6/I5 | cross-pair | **POOLED 7-major base-GBM** (THE keystone) | pool s42/l255 | usdjpy_15m_xpair.py pool 42 255 | D | **HIGH** | **done** | .515 [.475,.556] | .584/.540/.547 | .576/.517/**.485** | ⚠ SURVIVED-but-DOES-NOT-BEAT-BASE. Pooling DILUTES USDJPY own signal: binding 2025/26 WEAKER than base, DOWN 2026 collapses .485. VAL-AUC .527 < base .531. **USDJPY 15m signal is OWN-PAIR-SPECIFIC (opposite of EURUSD).** Redirect→own-pair data+capacity. | usdjpy_15m_xpair_pool_s42_l255_result.json |
| A1b | GBM core | single-pair LGBM base | s3/l255 (2× data+capacity) | usdjpy_15m_base.py 3 255 | D | **HIGH (redirect)** | **done** | .540 [.500,.579] | .576/.573/.529 | .578/.534/.555 | ⚠ SURVIVED (2024). **VAL-AUC .5313 = IDENTICAL to s6 .5312 → signal is NOT data-starved; AUC capped ~.531.** Regime rotation (DOWN 2026 lifts .555, UP 2026 drops .529). Cause=AUC ceiling, not data. Redirect→session/regime + tight-cov + CPCV cert. | usdjpy_15m_base_s3_l255_result.json |
| A6b | cross-pair | POOLED 7-major, more data | pool s21/l255 | usdjpy_15m_xpair.py pool 21 255 | D | high | pending | — | — | — | — | usdjpy_15m_xpair_pool_s21_l255_result.json |
| G1-base | validation | **per-fold-REFIT CPCV of OWN-PAIR base** (CERTIFY) | stride4 cov0.03 | usdjpy_15m_cpcv_base.py 4 0.03 1 | D | cert | **done** | **p10 .5516 (14/15)** | **p10 .5582 (15/15) ✓CERT** | p10 .5381 (10/15) | ✅ **UP CERTIFIED** (p10 .558≥BE, ALL folds clear); COMBINED certified (p10 .552, 14/15); DOWN real-but-sub-BE (p10 .538, 10/15). AUC mean .526, trip OK. | usdjpy_15m_cpcv_base_result.json |
| G1-pool | validation | per-fold-REFIT CPCV of POOLED (moot) | — | usdjpy_15m_cpcv.py | D | cert | **subsumed** | — | — | — | Pooling lost to own-pair base (A6 diluted binding yrs); own-pair CPCV is the cert of record. Skip pooled CPCV. | — |
| A9 | session | **NY/LDN/Asia session-concentrated own-pair base** | ny\|ldn\|asia s6/l127 | usdjpy_15m_session.py | D | **HIGH** | **done** | NY .599/.587/.553 | NY .589/.579/.580 | NY .663/.629/.455 | ✅ **NY SURVIVED** (VAL-AUC .544>all-sess .531; COMB clears 2024+2025; UP stable ~.58 all yrs incl 2026). LDN+Asia KILLED. **Direction edge is NY-concentrated (EURUSD precedent confirmed).** NY-CPCV cert running. | usdjpy_15m_session_{ny,ldn,asia}_s6_l127_result.json |
| A2 | gate | compression × session × coverage gate sweep | comp-q × session × cov | usdjpy_15m_regime.py (clone usdjpy_2m_regime) | G+D | med | pending | — | — | — | — | usdjpy_15m_regime_result.json |
| A3 | gate | compression-release × reversion specialist | rel_tighten + w_spec | usdjpy_15m_regime.py spec | D | med | pending | — | — | — | — | usdjpy_15m_regime_spec_result.json |
| A5 | cross-horizon | parent (30m/60m) → 15m front-load stack | soft/hard q-gate | usdjpy_15m_stack.py (new) | D | med | pending | — | — | — | — | usdjpy_15m_stack_result.json |
| A8 | side asym | UP/DOWN FILTER vs separately-trained SPECIALIST | filter / spec | usdjpy_15m_improve.py spec | U/Dn | filter med, spec ~null | pending | — | — | — | — | usdjpy_15m_improve_spec_result.json |
| C5 | state-space | online-ARF keystone (efficiency control) | river ARF+ADWIN | usdjpy_15m_statespace.py (new) | D(control) | keystone | pending | — | — | — | — | usdjpy_15m_arf_result.json |
| C2 | state-space | Kalman forward-filter drift sign | channel/velocity | usdjpy_15m_statespace.py kalman | D | ~null | pending | — | — | — | — | usdjpy_15m_kalman_result.json |
| D1 | sequence/DL | GRU/1D-CNN on path | window 20/40/80 | usdjpy_15m_dl.py | D | ~null (info-bound) | pending | — | — | — | — | usdjpy_15m_dl_result.json |
| E2 | mag→dir | direction-conditioned-on-magnitude | mag-quartile gate | usdjpy_15m_magdir.py | D | ~null (sign-invariant) | pending | — | — | — | — | usdjpy_15m_magdir_result.json |
| B/F | micro/exo | tick microstructure, OFI, news, x-asset | — | — | D | n/a | n/a (no USDJPY tick cache; bar-only) | — | — | — | DATA-BLOCKED (Tier B tick) / low-prior (Tier F) | — |

### Tier-I edge-improvement levers (apply ON the best edge found, evaluate COMBINATIONS)
| id | lever | script | status | note |
|---|---|---|---|---|
| I1 | adaptive-conformal (ACI) gate | usdjpy_15m_improve.py aci | pending | EURUSD-5m WINNER; retarget to 15m best combo |
| I2 | seed-ensemble ⊕ GBM stack | usdjpy_15m_cpcv.py 12 0.03 5 (nseed) + stack | pending | variance reduction; test if it lifts p10 |
| I3 | \|ret\|-weighted / GMADL loss | usdjpy_15m_loss.py | pending | USDJPY-2m KILLED on pooled; re-test @15m |
| I4 | calibration + selective threshold | wrap best book | pending | nearly-free wrapper |
| I6 | Optuna (worst-VAL-half objective) | usdjpy_15m_optuna.py | pending | USDJPY-2m KILLED; re-test @15m |
| I7 | recency-weighted training | usdjpy_15m_recency.py | pending | USDJPY-2m KILLED; re-test @15m |

### Tier-N discovered/novel (append as vetted; see backlog for mechanisms + priors)
| id | candidate | source | mechanism (direction sign) | prior | status |
|---|---|---|---|---|---|
| N16 | triple-barrier / trend-scan TRAIN labels | López de Prado AFML | relabels train target; may sharpen sign | ~20% | pending |
| N2 | triangular JPY-cross residual (EURJPY=EURUSD·USDJPY) | arXiv:0812.0913 | dislocation reversion carries sign | ~15% | pending (needs EURJPY — DATA-CHECK) |
| N10 | TAR-VECM error-correction speed-of-adjustment sign | threshold-VECM | cointegration velocity = directional | ~15% | pending |
| N17 | anti-contemporaneous lead-lag + transfer-entropy gate + RFF | Sirignano-Cont + Schreiber + Kelly-Malamud-Zhou | directed-info lead carries sign | ~13% | pending |

---

## RUN ORDER (first-to-run queue)
1. **A1** base s6/l127 (RUNNING) — floor + up-rate tripwire.
2. **A6/I5** xpair pool s42/l255 — THE keystone single-fit read.
3. **G1** refit-CPCV of pooled — certification gate (only if A6 point-above-BE).
4. **A9** NY-session × cross-pair — the highest-prior combo (EURUSD precedent: NY is where the 15m edge lives).
5. Tier-I levers on the best edge (seed-ens, ACI, calibration) toward >0.70 at tight coverage.
6. A2/A3 regime gates, A5 cross-horizon, A8 specialists; C5 ARF keystone (efficiency control); D1 DL (info-bound check).
7. Discovery rounds (Tier-N) until K=2 dry.

## RESUMABILITY
This ledger IS the state. On resume, continue from the first `pending`/`running` row; never repeat a `done` row. Update USDJPY_RESULTS.md (master table + per-tf + leaderboard) and this ledger together; commit after each row.
