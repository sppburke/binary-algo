SCOPE: AUDUSD · 15m — sweep LEDGER (status of record). Backlog/executable queue: `AUDUSD_15m_backlog.md`. Results of record: `AUDUSD_RESULTS.md`. Generic menu: `SWEEP_MATRIX.md`.

# AUDUSD 15m direction sweep — LEDGER

**Goal:** best AUDUSD 15m UP & best 15m DOWN binary-direction predictor, OOS-verified, certified (refit-CPCV p10≥.541 & ≥80% paths) + deployable-spec'd, >65% target. Both sides certified-or-honestly-exhausted; improve+discover loops dry.

**Mechanistic model of the edge (updated each iteration — THE ENGINE):**
- v0 (prior, pre-data): AUDUSD = USD-major (⇒ cross-pair pooling strong prior, the EURUSD keystone) + commodity/risk currency with **Asia-session** idiosyncratic info (RBA/China/AU data) — UNLIKE EUR/JPY (NY-concentrated). Two live levers: POOLING and SESSION (Asia hypothesis). NZDUSD = closest cousin (AUDNZD relative value candidate). AUC ceiling unknown; neighbors ~.539 (USDJPY) / ~.56 (EURUSD xpair).

| id | family | method | variant | script | target | prior | status | combined_oos | up_oos | down_oos | verdict | result_json |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A1 | baseline | single-pair LGBM | 239 feats, 15m label, all-session | audusd_15m_base.py | floor+AUC | high | DONE | .604/.587/.548 @cov2 | .603/.595/.493 | .580/.578/.589 | ✅ SURVIVED (2yr CI-lo clears BE; 2026 binding, DOWN-only) | audusd_15m_base_result.json |
| A9-all | cert | all-session own-pair refit-CPCV (multicov) | per-fold refit | audusd_15m_cpcv_session.py all | honest floor | high | DONE | cov1 p10 .542 (.87) | cov1 p10 .5345 (.80) | cov1 p10 .544 (.93) | ✅ DOWN+COMB CERT @cov1%; UP sub-BE all cov | audusd_15m_cpcv_session_all_multicov_result.json |
| A9 | session | own-pair session-restricted refit-CPCV | ny / ldn / asia | audusd_15m_cpcv_session.py | which session carries edge | high | RUNNING | — | — | — | — | audusd_15m_cpcv_session_{asia,ny,ldn}_multicov_result.json |
| A6 | cross-pair | USD-common-factor residual + lead-lag pooling | AUDUSD target (EURUSD keystone retarget) | audusd_15m_xpair.py | pooling helps? | high | pending | — | — | — | — | audusd_15m_xpair_result.json |
| A6b | cross-pair | NZDUSD-cousin / Antipodean pool + AUDNZD RV | AUD-NZD relative value | audusd_15m_xpair.py (cousin mode) | own-pair RV | med | pending | — | — | — | — | — |
| A6c | combo | session × cross-pair (the EURUSD certified combo) | best-session + xpair primary | session_xpair-style | combine keystones | high | pending | — | — | — | — | — |
| I2 | improve | seed-ensemble K=3 | on best edge | audusd_15m_cpcv_session.py (nseed) | variance↓ p10↑ | high | pending | — | — | — | — | — |
| I-loss | improve | \|ret\|-weighted / GMADL loss | on best edge | tbd | mag→dir bridge | med | pending | — | — | — | — | — |
| I-cal | improve | calibration + ACI gate | on best edge | tbd | gate quality | med | pending | — | — | — | — | — |
| I-opt | improve | Optuna (worst-VAL-half objective) | on best edge | tbd | capacity | med | pending | — | — | — | — | — |
| A-tb | label | triple-barrier first-touch train label | eval unchanged deriv-sign | tbd | label denoise | med | pending | — | — | — | — | — |
| Disc | discover | corpus-mine + AUDUSD-specific levers | commodity/China/carry/AUDNZD | (workflow) | new levers | — | RUNNING | — | — | — | — | — |

**Discipline reminders (strategy-eval §2):** deriv-faithful settlement, nonoverlap_chrono, per-year CI95, VAL worst-half selection, moved up-rate∈[.47,.53] tripwire, pre-registered falsifier in result JSON BEFORE OOS, full refit-CPCV to certify, adversarially verify every positive, one heavy job at a time (OOM), commit often.

## Run log (newest first)
- 2026-06-09 — **All-session refit-CPCV DONE → DOWN+COMBINED CERT @cov1% (thin), UP sub-BE.** DOWN p10 .544 (14/15 paths), COMB p10 .542 (13/15) @cov1%; UP p10 saturates .534–.538 at every cov (sub-BE). AUC mean .5213. **v2 model:** AUDUSD 15m DOWN is the ROBUST deliverable (certifies refit + survives frozen-2026 forward at .589@cov2); UP is regime-dependent/near-miss (frozen-2026 collapse .493 = the trap#9 forward signal; refit-CPCV agrees, no cert). Mechanism: for a risk-on commodity currency the DOWN side (risk-off / carry-unwind, sharp+directional) is more forecastable than UP (slow grind, drift-fighting) — matches backlog lever #4. NEXT: session concentration (Asia prior) + A6 pooling + seed-ens to lift p10 to a usable cov2–3% and rescue/confirm UP.
- 2026-06-09 — **A1 base DONE → SURVIVED.** All-session single-pair LGBM has a real 15m edge (cov2% COMB .604/.587/.548; 2yr CI-lo clears BE). **v1 model update:** AUDUSD all-session base is stronger than USDJPY all-session (which needed NY). Binding = 2026, where the edge is **DOWN-only** (DOWN .589@cov2, UP .493 dead) — frozen-2021 forward decay, DOWN survives it, UP doesn't. NEXT: all-session refit-CPCV (recovers per-era floor; direct cert path) → then session restriction (does Asia/NY concentrate it?). DOWN is the more robust deliverable; UP is regime-dependent (must confirm forward, trap#9).
- 2026-06-09 — sweep OPEN. Bootstrapped AUDUSD_RESULTS.md + ledger + backlog. Launched A1 base. Corpus-mining/discovery workflow launched in parallel. Mechanistic v0 above.
