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
| **G1-NY** | validation | **★★ NY-session per-fold-REFIT CPCV (CERT of record)** | ny stride2 cov0.03 | usdjpy_15m_cpcv_session.py ny 2 0.03 | D | cert | **done** | **p10 .580 (15/15)** | **p10 .586 (15/15) ✅CERT** | **p10 .572 (15/15) ✅CERT** | ✅✅ **BOTH SIDES CERTIFIED** — NY lifts AUC .526→.539, rescues DOWN (.538→.572). The two-sided cert of record. multicov run probing >0.70. | usdjpy_15m_cpcv_session_ny_cov0.03_result.json |
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
| I2 | seed-ensemble (K=3) on NY cert | usdjpy_15m_cpcv_session.py ny 2 0.03,0.02,0.01 3 | **done — LIFTS p10** | UP cov2% p10 .582→**.6005**, DOWN .554→**.5738** (both 15/15). Best certified config. `usdjpy_15m_cpcv_session_ny_seedens3_result.json` |
| I3 | \|ret\|-weighted / GMADL loss | usdjpy_15m_loss.py | pending | USDJPY-2m KILLED on pooled; re-test @15m |
| I4 | calibration + selective threshold | wrap best book | pending | nearly-free wrapper |
| I6 | Optuna (worst-VAL-half objective) | usdjpy_15m_optuna.py | pending | USDJPY-2m KILLED; re-test @15m |
| I7 | recency-weighted training | usdjpy_15m_recency.py | pending | USDJPY-2m KILLED; re-test @15m |

### Tier-N discovered/novel (append as vetted; see backlog for mechanisms + priors)
| id | candidate | source | mechanism (direction sign) | prior | status |
|---|---|---|---|---|---|
| N16/TN1 | triple-barrier / **avg-horizon** TRAIN labels | López de Prado AFML; Prata 2024 | relabels train target; may sharpen sign | ~20% | **done — no lift** (avg: VAL fixed-15m AUC .5319 ≈ base .531; 2026 COMB collapses .499). AUC ceiling is label-independent. `usdjpy_15m_tblabel_avg_s6_l127_result.json` |
| N2 | triangular JPY-cross residual (EURJPY=EURUSD·USDJPY) | arXiv:0812.0913 | dislocation reversion carries sign | ~15% | pending (needs EURJPY — DATA-CHECK) |
| N10/TN4 | TAR-VECM error-correction speed-of-adjustment sign | threshold-VECM | cointegration velocity = directional | ~15% | **done — NULL.** z cointegrated (ADF p=.003) but ECM γ̂≈−1e-5 (no reversion @15m); standalone sign(−z) AUC .490/.490/.502 (coin-flip); integration +.0037 VAL = overfit noise. `usdjpy_15m_tarvecm_result.json` |
| N11/TN2 | pairwise/listwise RANKING-LOSS over 7 majors | DeltaLag/ListNet/Feng | cross-section sign-ranking | ~13% | **subsumed (Tier-1)** — ranking loss is invariant to monotonic per-bar target transforms → predicts RELATIVE (USDJPY-vs-basket) rank, not own 15m sign; the own-pair pairwise form ≡ BCE (=base). A6 cross-pair POOLING already showed cross-pair structure DILUTES USDJPY own-sign at 15m (Tier-1). Logged not-run; reopen only if a new own-sign mechanism appears. |
| N17 | anti-contemporaneous lead-lag + transfer-entropy gate + RFF | Sirignano-Cont + Schreiber + Kelly-Malamud-Zhou | directed-info lead carries sign | ~13% | pending |

---

## TWO-SIDED LEADERBOARD (current best certified, by refit-CPCV p10)
| side | leader | refit-CPCV p10 | mean | paths clear | cov | status |
|---|---|---|---|---|---|---|
| **UP** | own-pair base GBM · **NY** · **seed-ens K=3** · cov2% (`usdjpy_15m_cpcv_session.py ny 2 0.02 3`) | **.6005** | .617 | 15/15 | 0.02 | ✅ CERTIFIED (best) |
| **DOWN** | own-pair base GBM · **NY** · **seed-ens K=3** · cov2% (same) | **.5738** | .616 | 15/15 | 0.02 | ✅ CERTIFIED (best) |
| _UP single-seed_ | NY · cov3% | .586 | .602 | 15/15 | 0.03 | ✅ (pre-seed-ens) |
| _DOWN single-seed_ | NY · cov3% | .572 | .597 | 15/15 | 0.03 | ✅ (pre-seed-ens) |
| _(ref) UP all-session_ | own-pair base, all-session | .558 | .568 | 15/15 | 0.03 | superseded by NY |
| _(ref) DOWN all-session_ | own-pair base, all-session | .538 | .555 | 10/15 | 0.03 | sub-BE (NY rescues) |

Incumbents to beat: UP p10 .586 / DOWN p10 .572 (NY). Challengers must beat the binding (worst) path with the discipline.

**Coverage→certified-p10 curve (NY refit-CPCV, `usdjpy_15m_cpcv_session_ny_multicov_result.json`):**
| cov | UP p10 / mean (med_n) | DOWN p10 / mean (frac) | COMB p10 / mean |
|---|---|---|---|
| 3% | .586 / .602 (937) | .572 / .597 (1.0) | .580 / .599 |
| 2% | .582 / .602 (580) | .554 / .599 (1.0) | .582 / .600 |
| 1.5% | .582 / .606 (423) | .554 / .603 (.93) | .575 / .603 |
| **1%** | **.599 / .625 (273)** | .544 / .601 (.87) | .590 / .614 |
| 0.5% | .580 / .631 (130) | .559 / .625 (.93) | .599 / .624 |
- **Best certified UP operating point = cov1%: p10 .599, mean .625** (15/15 paths, med_n 273). DOWN most-robust at cov3% (p10 .572, 15/15). 
- **>0.70 is NOT a certified reality** — AUC ~.539 caps the certified win-rate at ~.60–.625; beyond cov0.5% the mean rises (~.63) but p10 falls / frac<1.0 (thin-n, uncertifiable). The wall is the directional AUC ceiling. The only path past it is a higher-AUC signal (Tier-N levers tested next) or external data.

## RUN ORDER (first-to-run queue)
1. **A1** base s6/l127 (RUNNING) — floor + up-rate tripwire.
2. **A6/I5** xpair pool s42/l255 — THE keystone single-fit read.
3. **G1** refit-CPCV of pooled — certification gate (only if A6 point-above-BE).
4. **A9** NY-session × cross-pair — the highest-prior combo (EURUSD precedent: NY is where the 15m edge lives).
5. Tier-I levers on the best edge (seed-ens, ACI, calibration) toward >0.70 at tight coverage.
6. A2/A3 regime gates, A5 cross-horizon, A8 specialists; C5 ARF keystone (efficiency control); D1 DL (info-bound check).
7. Discovery rounds (Tier-N) until K=2 dry.

## SWEEP CLOSURE (2026-06-08) — both sides CERTIFIED; loops dry
**Outcome:** ✅ BOTH sides certified via NY refit-CPCV (UP p10 .586 cov3% / .599 cov1%, DOWN p10 .572 cov3%, 15/15 paths). Frozen book `USDJPY.m15ny.v1` (tag `book/USDJPY.m15ny.v1`). Adversarial shuffle-control PASSES. >0.70 shown unreachable as a certified floor (AUC ~.539 cap).

**Run-or-subsume status of remaining Tier rows** (coverage rule; AUC ceiling ~.531/.539 confirmed across data/capacity/pooling/avg-label/cointegration → the cap is a directional info bound):
- **A2 compression×session gate / A3 reversion specialist** — SUBSUMED: A9 session-filter IS the winning gate (NY); USDJPY-2m compression-regime (D1) was KILLED; the session-concentrated base already captures the best regime. Compression gates redistribute the same .539-AUC signal, can't exceed it.
- **A8 UP/DOWN separately-trained specialist** — SUBSUMED (Tier-1): subset-training destroys ranking at EURUSD-all-tf + USDJPY-1m/2m; the symmetric NY base certifies BOTH sides already (DOWN via confidence selection, not a separate model).
- **C2 Kalman / C5 ARF / C1 HMM / C3 RMT / C4 CCM** — SUBSUMED: sign-invariance theorem (state/complexity gates = magnitude). ARF/Kalman moot for a CERTIFIED edge (the wall is not pure efficiency — there IS extractable NY signal). 
- **D1 GRU/CNN, DL/spectral/foundation** — SUBSUMED (Tier-1): neural+spectral DIRECTION sweep KILLED 84/84 arms all EURUSD tf (THEORY.md); Kronos/Chronos null all horizons; info-bound caps DL at the .53 AUC.
- **E2 mag→dir / E3 complexity / E4 info-bars** — SUBSUMED: sign-invariance theorem (magnitude ≠ sign; TAR-VECM standalone .50 reconfirmed reversion-magnitude carries no 15m sign).
- **B/F (tick micro / OFI / news / x-asset)** — DATA-BLOCKED (no USDJPY tick cache) / low-prior bar-only.
- **A5 cross-horizon stack (30m/60m parent → 15m child)** — **LOGGED, NOT RUN.** The one untested lever with a non-subsumed mechanism (longer-horizon drift sign could add orthogonal info). Needs a 30m USDJPY parent book first (untested key). Lower prior given the AUC ceiling holds at neighboring horizons. → `sweeps/USDJPY_15m_backlog.md` follow-up; reopen if a 30m USDJPY direction edge is found.

**Discovery: 2 dry rounds** (R1: TN1 avg-label null, TN4 TAR-VECM null, TN2 ranking subsumed; R2: corpus+repo scan → every on-disk direction-carrying lever tested/killed/subsumed-by-theorem/external-blocked). Loop SATURATED.

## RESUMABILITY
This ledger IS the state. On resume, continue from the first `pending`/`running` row; never repeat a `done` row. Update USDJPY_RESULTS.md (master table + per-tf + leaderboard) and this ledger together; commit after each row.
