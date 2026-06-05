# USDJPY × 2m — Sweep Ledger

SCOPE: USDJPY · 2m (120s). KEY-SPECIFIC ledger (status of record). Results → `USDJPY_RESULTS.md`;
backlog/queue → `sweeps/USDJPY_2m_backlog.md`; generic methods → `METHODS_CATALOG.md`/`SWEEP_MATRIX.md`.

currency: USDJPY
timeframe: 2m (120s)
started: 2026-06-05
target: best UP and best DOWN binary-direction predictor at 120s, OOS(2026)-verified, clearing breakeven 0.541
        (deriv-faithful, ties LOSE, bar-close approx; per-year CI95; worst-VAL-half gate; nonoverlap_chrono gap=120s).
incumbent/answer: TBD (sweep in progress).

## MODEL OF THE EDGE (updated each iteration — drives the NEXT experiment; see backlog for reasoning)
- USDJPY **1m** is near-efficient/EXHAUSTED (best UP `usdjpy_1m_base 6 255` cov2% .547/.534/.538; worst-yr .534 < .541 BE;
  DOWN dead). Cause = informational/signal-level wall at 60s, NOT modeling. Magnitude STRONG (sign-invariant).
- **2m single-pair base = sub-BE** (A1/A2 binding yr 2025 UP .524) AND **compression-release regime filter (D1) = sub-BE**
  (binding 2025 UP .521; mechanism doesn't transfer robustly from EURUSD — whose 2m edge is TICK-based, unavailable to USDJPY).
- **★ BREAKTHROUGH (C1): cross-pair POOLED training LIFTS the binding year.** UP frozen-gate cov2% **.547/.546/.570**
  (2024/2025/2026) — point estimates ABOVE BE in ALL 3 years incl strict-OOS 2026; covcurve cov3% UP .549/.543/.555.
  KILLED ONLY on CI-lo width (~.527 @cov2%, n-limited). Confirms the 30m 94%-pooling-lever transfers to USDJPY 2m:
  pooling 7 majors' base feats decorrelates pair-idiosyncratic noise + adds regime diversity → stabilizes 2025.
- **Updated cause model:** USDJPY 2m UP has a REAL thin edge (unlike 1m); the binding constraint is now **estimator
  variance / n at the confident tail**, NOT signal absence. ⇒ Attack with: MORE DATA (C1b), seed-ensemble (F1, variance
  reduction → tighter tail), then **CPCV** (aggregates folds → the right test for a thin-but-real edge). DOWN marginal
  (2025 .548 but 2024/26 ~.52) — still needs work.
- **★ CPCV CORRECTION (G1):** the pooled edge is **NOT CERTIFIED**. Per-fold-refit CPCV (15 purged paths, all-pair purge,
  ties-strict) gives UP p10 **.5245** (mean .536, frac_clear .33), DOWN p10 .5195. The C1 frozen-gate .547/.546/.570 was
  a thin-n SELECTION OVERSTATEMENT — trust the refit. **Recency is the wall, not n**: recent regime (2021-26) UP mean .532
  / 22% paths clear; older (2012-20) .542 / 50% clear. So the deployable-regime edge is ~.53 UP, ~1.5pp SUB-BE, and MORE
  data won't fix a regime/signal deficit. Pooling DID lift the global mean above 1m (.536 vs .52) — real signal gain, not
  certifiable. **Cross-pair-pooled direction path = EXHAUSTED pending Tier-I seed-ens check (G1b).**
- ⇒ Remaining live frontiers: (1) Tier-I seed-ens on pooled CPCV (G1b, low prior — won't close 1.5pp); (2) DISCOVERY levers
  (workflow running — NEW directional mechanisms the base/pool miss); (3) lower-prior ledger rows (loss/DL/statespace/session);
  (4) EXTERNAL data (US-JP rate-diff/carry — the structurally-right USDJPY lever, gated on user "go"). Magnitude (sign-invariant)
  not yet measured at 2m — likely STRONG (like 1m), record separately.

## STATUS: IN PROGRESS (started 2026-06-05). Working Tier A→F; both sides scored symmetrically every row.

## LEDGER
| id | family | method | variant | script | target | prior | status | combined_oos | up_oos | down_oos | verdict | result_json |
|----|--------|--------|---------|--------|--------|-------|--------|--------------|--------|----------|---------|-------------|
| A1 | base | single-pair LGBM | stride24 leaves127 | usdjpy_2m_base.py | 2m | — | done | .541/.521/.516 | .546/.525/.527 | .532/.515/.498 | KILLED (no yr COMB CI-lo≥BE; VAL-AUC .524) | usdjpy_2m_base_result.json |
| A2 | base | single-pair LGBM | stride6 leaves255 (1m-best cfg) | usdjpy_2m_base.py 6 255 | 2m | .35 | done | .538/.522/.523 | .546/.524/.536 | .529/.520/.507 | KILLED (binding yr 2025 UP .524 CI-lo .512<BE; VAL-AUC .526) | usdjpy_2m_base_s6_l255_result.json |
| A3 | base | single-pair LGBM | stride12 leaves191 (capacity mid) | usdjpy_2m_base.py 12 191 | 2m | .15 | pending | | | | | usdjpy_2m_base_s12_l191_result.json |
| B1 | side | UP/DOWN split @frozen gate | (free from A covcurve) | — | 2m | — | pending | | | | | (from A result) |
| B2 | gate | worst-VAL-half gate sweep | per side | usdjpy_2m_base (built-in) | 2m | — | pending | | | | | |
| B3 | spec | UP specialist (subset-trained) | dip-buy/Tokyo | usdjpy_2m_spec.py up | 2m | .15 | pending | | | | | usdjpy_2m_spec_up_result.json |
| B4 | spec | DOWN specialist (subset-trained) | rally-fade | usdjpy_2m_spec.py down | 2m | .12 | pending | | | | | usdjpy_2m_spec_down_result.json |
| C1 | xpair | cross-pair POOLED GBM | 7-major pooled s42/l255, eval USDJPY | usdjpy_2m_xpair.py pool 42 255 | 2m | .40 | **INCUMBENT** | .535/.547/.551 | **.547/.546/.570** | .519/.548/.520 | NOT-killed-as-edge (UP pts ABOVE BE all 3 yrs incl OOS .570; KILLED only on CI-lo width ~.527<BE @cov2%). POOLING LIFTED binding 2025 (.524→.546). → improve+CPCV | usdjpy_2m_xpair_pool.log (Tier-1; JSON clobbered by C1b stub, numbers in log) |
| C1b | xpair | cross-pair POOLED GBM | 2x data s21/l255 | usdjpy_2m_xpair.py pool 21 255 | 2m | .45 | done | .544/.542/.526 | .542/.540/.527 | .548/.544/.524 | KILLED on CI-lo; lifted 2024/25 (COMB pt>BE) but OOS 2026 DROPPED .526. Confirms thin/tail-sensitive edge. C1(s42) keeps better worst-yr → incumbent | usdjpy_2m_xpair_pool_s21_l255_result.json |
| C2 | xpair | cross-pair OF residual | UJ-N3 analog @2m | usdjpy_2m_xpair.py ofresid | 2m | .12 | pending | | | | | usdjpy_2m_xpair_ofresid_result.json |
| C3 | xpair | EURUSD.min2 frozen-parent fwd | transfer to USDJPY | usdjpy_2m_xpair.py transfer | 2m | .10 | pending | | | | | usdjpy_2m_xpair_transfer_result.json |
| D1 | regime | compression-release filter | dip×comp×session, sym-GBM | usdjpy_2m_regime.py 8 | 2m | .30 | killed | — | .548/.521/.542 | .519/.513/.497 | KILLED both (UP binding 2025 .521 CI-lo .503<BE; pt .54+ only in 2024/26; DOWN dead). Mechanism regime-dependent, doesn't transfer robustly to USDJPY | usdjpy_2m_regime_s8_result.json |
| D2 | regime | gotobi / Tokyo-session gate | JPY-specific calendar | usdjpy_2m_regime.py session | 2m | .18 | pending | | | | | usdjpy_2m_regime_session_result.json |
| D3 | regime | vol/regime-route blend | route hi/lo-vol | usdjpy_2m_regime.py route | 2m | .12 | pending | | | | | usdjpy_2m_regime_route_result.json |
| E1 | loss | |ret|-weighted / GMADL objective | on best base | usdjpy_2m_loss.py | 2m | .12 | pending | | | | | usdjpy_2m_loss_result.json |
| E2 | dl | GRU/MLP decorrelated stack | torch | usdjpy_2m_dl.py | 2m | .10 | pending | | | | | usdjpy_2m_dl_result.json |
| E3 | statespace | Kalman/HMM forward-filter gate | sign-invariance check | usdjpy_2m_statespace.py | 2m | .07 | pending | | | | | usdjpy_2m_statespace_result.json |
| F1 | improve | seed-ensemble | on best edge | usdjpy_2m_improve.py seedens | 2m | .15 | pending | | | | | usdjpy_2m_improve_seedens_result.json |
| F2 | improve | ACI adaptive-conformal gate | on best edge | usdjpy_2m_aci.py | 2m | .12 | pending | | | | | usdjpy_2m_aci_result.json |
| F3 | improve | calibration (isotonic/Platt) | on best edge | usdjpy_2m_improve.py calib | 2m | .08 | pending | | | | | usdjpy_2m_improve_calib_result.json |
| F4 | improve | Optuna (worst-VAL-half obj) | on best edge | usdjpy_2m_optuna.py | 2m | .12 | pending | | | | | usdjpy_2m_optuna_result.json |
| G1 | cert | full per-fold-refit CPCV (pooled) | 6grp C(6,2)=15 paths, ties-strict, cov3% | usdjpy_2m_cpcv.py 16 0.03 | 2m | — | **done** | p10 .527 | **p10 .524** (mean .536) | p10 .520 (mean .536) | **NOT CERTIFIED** — UP p10 .5245 frac_clear .33; DOWN p10 .5195 frac_clear .33. CORRECTS C1 overstatement. Recency: recent regime WEAKER (UP .532 vs older .542). Edge real but sub-BE | usdjpy_2m_cpcv_result.json |
| G1b | cert | seed-ensemble pooled CPCV (Tier-I) | K=3 seeds/fold, cov3% | usdjpy_2m_cpcv.py 16 0.03 3 | 2m | .12 | pending | | | | | usdjpy_2m_cpcv_k3_result.json |

Legend: prior = subjective P(survives a pre-registered falsifier), used to size fast-KILL effort. status ∈ {pending,running,done,killed,subsumed}.
Tier-N (discovered) rows appended below as discovery rounds run. CPCV (G1) triggers only when a row's worst held-out year CI-lo ≥ 0.541.

## TIER-N — discovered (appended by discovery rounds)
(none yet — discovery round R1 pending after Tier A–D drain)

## EVENT LOG
- 2026-06-05: ledger created; `usdjpy_2m_base.py` written (H=2 fork of usdjpy_1m_base.py); A1 launched.
- 2026-06-05: A1 (default) + A2 (rich stride6/leaves255) DONE → both KILLED as standalone. All-bars base sub-BE
  (A2 binding yr 2025 UP .524 CI-lo .512 < .541; oos UP .536). UP carries faintly, DOWN weak — SAME all-bars wall
  as 1m. Update to MODEL: the 2m signal lives in the compression-release REGIME, not all-bars → D1 is the decisive test.
- 2026-06-05: `usdjpy_2m_regime.py` written (H=2 fork of usdjpy_1m_regime.py, nonoverlap gap=120); D1 launched (stride8).
