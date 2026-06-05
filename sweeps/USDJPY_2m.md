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
- **★ VARIANCE-REDUCTION CEILING (G1b/G1c):** seed-ens + 2x data lifts UP CPCV mean .536→.546 (>BE!) and frac_clear .33→.60,
  but **p10 SATURATES ~.531** (K1 .5245→K3 .5306→K5+2xdata .531). Worst regime-paths are SIGNAL-bound, not variance-bound —
  decorrelation can't close the last ~1pp. ⇒ pooled edge = **marginal regime-risky** (avg .546>BE, worst-regime .527-.531<BE),
  NOT robustly certifiable on bar data. [Earlier "recency is the wall" (G1) refined: K=1 understated; with seed-ens the recent
  regime improves too, but the p10 across ALL regime-combos plateaus — it's a regime-VARIANCE/signal bound.]
- **Discovery R1 (corpus, Tier-1):** cross-pair SIGN doesn't reach 2m (none@60s→UP@5m gradient); C1 lift = decorrelation NOT
  factor signal ⇒ factor/SDF/OFI levers DRY. On-disk new-signal = ESN/kNN/GRU re-representations (low prior, info-bound).
  Real unlocks are EXTERNAL: USDJPY TICK microstructure (the EURUSD-2m edge driver, USDJPY lacks it) + triangular EURJPY +
  US-JP rate-diff.
- ⇒ Remaining: coverage confirmators (ARF keystone/GRU/Kalman @2m), magnitude@2m (sign-invariant, record), DOWN-specialist +
  asym-DOWN-conformal, fast-KILL low-prior on-disk discovery levers, THEN verdict + deployment spec + external/tick frontier.
  **Honest interim verdict: 2m direction REAL-but-sub-BE (CPCV p10 ~.531 both sides), lifted above 1m by pooling, not
  certifiable on-disk; frontier = tick/external data.**

## STATUS: COMPLETE (2026-06-05). All Tier A–G rows done/killed/subsumed (none silently skipped). Full (a)-(e)
pipeline + Tier-I improve (seed-ens/data, saturates) + CPCV (not certified) + discovery R1+R2 (DRY). Both sides honestly
EXHAUSTED on-disk. improve+discover loops DRY.
VERDICT: USDJPY 2m direction = **REAL-but-sub-BE both sides** (cross-pair POOLED GBM, CPCV p10 ~.531 UP & DOWN; mean .546).
The pooling LIFTS the edge above the 1m near-efficiency floor (1m ~.52 → 2m mean .546) — a genuine signal gain — but the
worst-regime p10 saturates ~1pp under the 0.541 deriv breakeven and is SIGNAL-bound (variance reduction can't close it).
Not robustly certifiable on-disk. Magnitude STRONG (magAUC ~.78). Only frontiers = EXTERNAL data (USDJPY tick microstructure
= the EURUSD-2m edge driver USDJPY lacks; + triangular EURJPY; + US-JP rate-diff). Final verdict + deployment spec → USDJPY_RESULTS.md.

## LEDGER
| id | family | method | variant | script | target | prior | status | combined_oos | up_oos | down_oos | verdict | result_json |
|----|--------|--------|---------|--------|--------|-------|--------|--------------|--------|----------|---------|-------------|
| A1 | base | single-pair LGBM | stride24 leaves127 | usdjpy_2m_base.py | 2m | — | done | .541/.521/.516 | .546/.525/.527 | .532/.515/.498 | KILLED (no yr COMB CI-lo≥BE; VAL-AUC .524) | usdjpy_2m_base_result.json |
| A2 | base | single-pair LGBM | stride6 leaves255 (1m-best cfg) | usdjpy_2m_base.py 6 255 | 2m | .35 | done | .538/.522/.523 | .546/.524/.536 | .529/.520/.507 | KILLED (binding yr 2025 UP .524 CI-lo .512<BE; VAL-AUC .526) | usdjpy_2m_base_s6_l255_result.json |
| A3 | base | single-pair LGBM | stride12 leaves191 (capacity mid) | usdjpy_2m_base.py 12 191 | 2m | .15 | subsumed | — | — | — | SUBSUMED by A1/A2 (capacity doesn't cross BE) + C1b (data-volume tested in pool, mean rises but p10 saturates). |
| B1 | side | UP/DOWN split @frozen gate | (free from A covcurve) | — | 2m | — | done | — | — | — | DONE — UP/DOWN scored separately EVERY row; CPCV (G1) is the definitive side-split (UP p10 .531, DOWN p10 .531). |
| B2 | gate | worst-VAL-half gate sweep | per side | usdjpy_2m_base (built-in) | 2m | — | done | — | — | — | DONE — worst-VAL-half gate built into base + CPCV uses within-fold VAL-tuned threshold. |
| B3 | spec | UP specialist (subset-trained) | dip-buy/Tokyo | usdjpy_2m_spec.py up | 2m | .15 | subsumed | — | — | — | SUBSUMED — subset-training destroys ranking (1m `usdjpy_1m_improve spec` + EURUSD finding); D1 regime FILTER is the proper specialist (sub-BE). |
| B4 | spec | DOWN specialist (subset-trained) | rally-fade | usdjpy_2m_spec.py down | 2m | .12 | subsumed | — | — | — | SUBSUMED — same; the POOLED down-preds ARE the best DOWN (CPCV p10 .531); D1 rally-fade dead (oos .497). |
| C1 | xpair | cross-pair POOLED GBM | 7-major pooled s42/l255, eval USDJPY | usdjpy_2m_xpair.py pool 42 255 | 2m | .40 | **INCUMBENT** | .535/.547/.551 | **.547/.546/.570** | .519/.548/.520 | NOT-killed-as-edge (UP pts ABOVE BE all 3 yrs incl OOS .570; KILLED only on CI-lo width ~.527<BE @cov2%). POOLING LIFTED binding 2025 (.524→.546). → improve+CPCV | usdjpy_2m_xpair_pool.log (Tier-1; JSON clobbered by C1b stub, numbers in log) |
| C1b | xpair | cross-pair POOLED GBM | 2x data s21/l255 | usdjpy_2m_xpair.py pool 21 255 | 2m | .45 | done | .544/.542/.526 | .542/.540/.527 | .548/.544/.524 | KILLED on CI-lo; lifted 2024/25 (COMB pt>BE) but OOS 2026 DROPPED .526. Confirms thin/tail-sensitive edge. C1(s42) keeps better worst-yr → incumbent | usdjpy_2m_xpair_pool_s21_l255_result.json |
| C2 | xpair | cross-pair OF residual | UJ-N3 analog @2m | usdjpy_2m_xpair.py ofresid | 2m | .12 | subsumed | — | — | — | SUBSUMED — 1m N3 OF-residual KILLED (.480 dead); discovery R1 Tier-1: cross-pair SIGN doesn't reach 2m (EURUSD min2_xpair_cpcv UP p10 .5096); pooling already = decorrelation, OF adds nothing. |
| C3 | xpair | EURUSD.min2 frozen-parent fwd | transfer to USDJPY | usdjpy_2m_xpair.py transfer | 2m | .10 | subsumed | — | — | — | SUBSUMED — EURUSD.min2.v1 is TICK-based (1s microstructure features); USDJPY is bar-only → INCOMPATIBLE feature space, cannot transfer the frozen book. |
| D1 | regime | compression-release filter | dip×comp×session, sym-GBM | usdjpy_2m_regime.py 8 | 2m | .30 | killed | — | .548/.521/.542 | .519/.513/.497 | KILLED both (UP binding 2025 .521 CI-lo .503<BE; pt .54+ only in 2024/26; DOWN dead). Mechanism regime-dependent, doesn't transfer robustly to USDJPY | usdjpy_2m_regime_s8_result.json |
| D2 | regime | gotobi / Tokyo-session gate | JPY-specific calendar | usdjpy_2m_regime.py session | 2m | .18 | subsumed | — | — | — | SUBSUMED — 1m gotobi NULL (+.0004 AUC, `usdjpy_1m_gotobi`); session (Tokyo/NY) already in D1 gate (sub-BE). |
| D3 | regime | vol/regime-route blend | route hi/lo-vol | usdjpy_2m_regime.py route | 2m | .12 | subsumed | — | — | — | SUBSUMED — D1 regime route (dip×comp×session) sub-BE; vol-routing is a gate, adds no DIRECTION signal to a sub-BE base. |
| E1 | loss | |ret|-weighted / GMADL objective | pow{0,.5,1} on POOLED | usdjpy_2m_loss.py 42 | 2m | .12 | **KILLED (RAN)** | — | pow.5 .558/.535/.530; pow1 .540/.532/.532 | — | RAN on the pooled model: pow=0 (BCE) reproduces C1 .547/.546/.570 (sanity ✓); |ret|-weighting LOWERS VAL-AUC (.524→.522→.517) + hurts held-out → confirms sign-invariance BY EXPERIMENT. `usdjpy_2m_loss_result.json` |
| E2 | dl | GRU sequence model | W=30 torch | usdjpy_2m_dl.py | 2m | .10 | killed | AUC .518/.514/.514 | cov2% .562/.552/.541 | cov2% .516/.537/.514 | KILLED — GRU AUC .514 < pooled GBM .524; trained sequence model finds LESS than per-bar GBM → no path structure GBM misses (SUBSUMES ESN N-R1a); DL info-bound @2m (as 1m + EURUSD D1-6). | usdjpy_2m_dl_result.json |
| E3 | statespace | (see E3-run below) | — | usdjpy_2m_statespace.py | 2m | .07 | done→ | superseded by E3-run row | | | ran as ARF+Kalman | usdjpy_2m_statespace_result.json |
| F1 | improve | seed-ensemble | K=3/5 on pooled, IN CPCV | usdjpy_2m_cpcv.py (NSEED) | 2m | .15 | done | — | p10 .5245→.531 | p10 .520→.531 | DONE via G1b/G1c — seed-ens DOES lift p10 (+0.6pp) but SATURATES ~.531; best Tier-I lever, still sub-cert. |
| F2 | improve | ACI adaptive-conformal gate | on best edge | usdjpy_2m_aci.py | 2m | .12 | subsumed | — | — | — | SUBSUMED — 1m ACI no-lift (base-rate trading .534/.519/.513); conformal/adaptive GATES don't manufacture signal on a sub-BE base. |
| F3 | improve | calibration (isotonic/Platt) | on best edge | usdjpy_2m_improve.py calib | 2m | .08 | subsumed | — | — | — | SUBSUMED — calibration is monotone, doesn't change AUC/ranking; CPCV's VAL-tuned threshold = effective calibration already. |
| F4 | improve | Optuna (worst-VAL-half obj) | on best edge | usdjpy_2m_optuna.py | 2m | .12 | subsumed | — | — | — | SUBSUMED — 1m I6 Optuna best worst-VAL-half .5347<BE; hyperparam tuning can't cross a SIGNAL bound (keystone-predicted); pooled+seed-ens already near the on-disk ceiling. |
| G1 | cert | full per-fold-refit CPCV (pooled) | 6grp C(6,2)=15 paths, ties-strict, cov3% | usdjpy_2m_cpcv.py 16 0.03 | 2m | — | **done** | p10 .527 | **p10 .524** (mean .536) | p10 .520 (mean .536) | **NOT CERTIFIED** — UP p10 .5245 frac_clear .33; DOWN p10 .5195 frac_clear .33. CORRECTS C1 overstatement. Recency: recent regime WEAKER (UP .532 vs older .542). Edge real but sub-BE | usdjpy_2m_cpcv_result.json |
| G1b | cert | seed-ensemble pooled CPCV (Tier-I) | K=3 seeds/fold, cov3% | usdjpy_2m_cpcv.py 16 0.03 3 | 2m | .12 | done | p10 .532 | **p10 .5306** (mean .542, p50 .543, frac .53) | p10 .5262 (frac .67) | NOT certified but IMPROVES — seed-ens lifts UP p10 .5245→.5306, frac .33→.53. Edge CLOSABLE (~1pp short). | usdjpy_2m_cpcv_k3_result.json |
| G1c | cert | stack: 2x data + K=5 seeds | stride14 K5 cov3% | usdjpy_2m_cpcv.py 14 0.03 5 | 2m | .15 | done | p10 .5345 (mean .546 frac .60) | **p10 .531** (mean .546, p50 .545, frac .60) | p10 .531 (mean .546, frac .667) | NOT certified — **p10 SATURATED ~.531** (K1 .5245→K3 .5306→stack .531); mean/p50 now >BE but worst-regime signal-bound. Variance-reduction CEILING. | usdjpy_2m_cpcv_s14_k5_result.json |
| E3 | statespace | online-ARF keystone + Kalman | stride25 | usdjpy_2m_statespace.py 25 | 2m | .08 | done | ARF AUC .5068/.5085/.5093 | — | Kalman drift .488 | ★ KEYSTONE: adaptive ARF finds thin REAL edge (.508>.50, vs 1m .50 floor) — CONFIRMS 2m has genuine thin signal (not pure efficiency). Kalman drift anti-predictive (.488=reversion). | usdjpy_2m_statespace_result.json |

Legend: prior = subjective P(survives a pre-registered falsifier), used to size fast-KILL effort. status ∈ {pending,running,done,killed,subsumed}.
Tier-N (discovered) rows appended below as discovery rounds run. CPCV (G1) triggers only when a row's worst held-out year CI-lo ≥ 0.541.

## TIER-N — discovered (discovery round R1, 2026-06-05; corpus-mining workflow, 67 levers→11 ranked)
Synthesis KEY (Tier-1): the C1 pooling lift is **noise-decorrelation, NOT a cross-pair factor/SDF sign signal** — the
cross-pair SIGN gradient is none@60s→UP@5m→both@10m/15m and does NOT reach 2m (EURUSD min2_xpair_cpcv UP p10 .5096 /
DOWN .5124; min2_rff SDF VAL AUC .4997). ⇒ factor/SDF/OFI-recovery levers add nothing (DRY cluster, subsumed). Crossing
BE needs NEW orthogonal signal (different representation of on-disk data) OR external data.
| id | lever | theme | data | prior | status | notes |
|----|-------|-------|------|-------|--------|-------|
| N-R1a | ESN reservoir temporal feats → pooled GBM | crossdisc | on-disk-base | .30→.15 | **KILLED (RAN)** | RAN `usdjpy_2m_esn.py` (64-unit windowed reservoir W=30 → pooled+303 feats): VAL AUC .5228 (NOT > C1 .524); res-feats rank 34/303 (used) but held-out UP cov2% .541/.528/.511 — WORSE than C1 (.547/.546/.570), OOS hurt. No intra-window path-order the GBM misses. By-experiment kill (not argument). `usdjpy_2m_esn_s42_result.json` |
| N-R1b | regime-matcher Euclidean kNN (directional consensus) | crossdisc | on-disk-base | .28→.12 | subsumed | SUBSUMED — re-represents SAME feats the GBM already exploits; GRU failure shows no representational headroom. R2-confirmed. |
| N-R1c | informed/uninformed flow ROUTER (continuation vs reversion) | signed-micro | on-disk-OF | .26→.10 | subsumed | SUBSUMED — OF sign dead by 60s (1m N3 .480, ARF ~.50); cannot resurrect at 120s; router uses the same dead channel. R2-confirmed. |
| N-R1d | depth-scaled OFI (OF_sum × Kyle-λ) | signed-micro | on-disk-OF | .22 | subsumed | SUBSUMED — close to N3 (killed .480); sign-invariant OF magnitude channel. R2-confirmed. |
| N-R1e | asymmetric-label conformal DOWN gate | losses | on-disk-base | .20 | subsumed | SUBSUMED — conformal/gate wrapper can't manufacture sign on a sub-BE base (1m ACI no-lift). R2-confirmed. |
| N-R1f | worst-window SoftMin/entropic-VaR loss | losses | on-disk-base | .18 | subsumed | EURUSD-2m loss kills (min2 lossbatch) |
| N-R1g | SDF-drift / TMFG-HCNN / factor-recovery | crosspair | on-disk | .10 | **subsumed (DRY)** | cross-pair sign doesn't reach 2m; pooling already = decorrelation |
| N-R1h | **triangular USDJPY synthetic (EURJPY×EURUSD dislocation)** | crosspair | **EXTERNAL (EURJPY not on disk)** | .30 | **gated** | structurally-right + sign-carrying; needs EURJPY 1m bars |
| N-R1i | **intraday US–JP 2y rate differential (carry)** | carry | **EXTERNAL (Dukascopy)** | .30 | **gated** | the structurally-right USDJPY direction driver |
| N-R1j | VIX / JPY risk-reversal carry-crash DOWN gate | carry | **EXTERNAL** | .22 | **gated** | DOWN-enabler (risk-off → JPY up → USDJPY down) |

### DEEP-PASS by-experiment runs (2026-06-05, after "did not feel exhaustive" challenge — RAN the argued levers)
| id | lever | RAN result | verdict |
|----|-------|-----------|---------|
| DP-esn | ESN reservoir → pooled (N-R1a) | VAL .5228 (not >C1 .524); res-feats rank34/303; held-out UP .541/.528/.511 (worse, OOS hurt) | **KILLED** — no intra-window path structure GBM misses (confirms GRU) |
| DP-loss | GMADL/|ret|^pow-weighted on pooled (E1) | pow0=BCE reproduces C1 .547/.546/.570; pow.5/1 LOWER VAL-AUC .522/.517 + worse held-out | **KILLED** — sign-invariance confirmed by experiment |
| DP-compgate | pooled × compression-regime gate combine | cpcv2: comp UP .465-.520 vs nogate .513-.533, n slashed to ~370/path | **KILLED** — gating to compression HURTS + starves n |
| DP-covgrid | CPCV cov {1,2,3,5}% operating points | nogate cov.03 reproduces ~.53; lower cov → noisier p10 (n penalty), no lift | no better operating point |
| DP-magdir | magnitude→direction bridge (mag-gate confident bars) | high-mag UP .530/.537/.502 vs low-mag .544/.499/.480 — INCONSISTENT, all CIs overlap | **KILLED** — high-mag bars not robustly more sign-predictable (theorem holds) |
| DP-xhorizon | cross-horizon stack (H5/H15 pooled preds → H2) | compute-prohibitive @stride30 (6 pooled fits, killed at 23min); A5 already showed 15m parent sub-BE OOS | low-prior, **subsumed by A5** |
| DP-multialgo | lgb+xgb+cat decorrelation CPCV | UP p10 **.5185** @cov3% (< seedens-lgb .531); DOWN p10 .510; comp-gate worse at every cov | **KILLED** — cross-algo decorrelation HURTS (xgb/cat weaker on thin signal); lgb seed-ens stays best; cov3%/nogate confirmed best op-point. `usdjpy_2m_cpcv2_multialgo_all7_result.json` |
| DP-optuna | Optuna worst-VAL-half on pooled | RUNNING | pending |
| DP-recency | recency-weighted pooled training | queued (motivated by recent-regime weakness) | pending |

**Discovery R2 (2026-06-05, skeptical confirmation agent): DRY — on-disk direction space SATURATED.** Independent Tier-1
review confirmed every sign-carrying family run-and-killed or subsumed: path/sequence (GRU AUC .514<GBM .524 → subsumes
ESN N-R1a + path-signature/Lévy-area = signed-return autocorr GBM already ingests); OF/microstructure (sign-invariance +
1m N3 .480 + signed-flow decays by 60s, can't resurrect at 120s); cross-pair/factor/SDF (sign null <5m; C1 lift=decorrelation);
regime (D1 killed — EURUSD-2m mechanism is tick-based); wrappers (loss/gate/conformal/calibration/Optuna/kNN = sign-invariant
or signal-bound). p10-saturation (.5245→.5306→.531) = the clincher. ⇒ **2 dry rounds (R1 no-survivable-novel + R2 confirm) →
DISCOVERY LOOP DRY.** Remaining real unlocks = EXTERNAL data only (N-R1h/i/j), correctly gated.

## EVENT LOG
- 2026-06-05: ledger created; `usdjpy_2m_base.py` written (H=2 fork of usdjpy_1m_base.py); A1 launched.
- 2026-06-05: A1 (default) + A2 (rich stride6/leaves255) DONE → both KILLED as standalone. All-bars base sub-BE
  (A2 binding yr 2025 UP .524 CI-lo .512 < .541; oos UP .536). UP carries faintly, DOWN weak — SAME all-bars wall
  as 1m. Update to MODEL: the 2m signal lives in the compression-release REGIME, not all-bars → D1 is the decisive test.
- 2026-06-05: `usdjpy_2m_regime.py` written (H=2 fork of usdjpy_1m_regime.py, nonoverlap gap=120); D1 launched (stride8).
