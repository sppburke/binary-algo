> **SCOPE: USDJPY · 1m** (key-specific EXECUTABLE backlog — TOP-N first-to-run queue, incumbents-to-beat, discovery rounds). Ledger/status: `sweeps/USDJPY_1m.md`. Results of record: `USDJPY_RESULTS.md`. Generic ideas: `IDEAS_LOG.md` + `SWEEP_MATRIX.md`.

# USDJPY × 1m — Executable Backlog

## Incumbents to beat (current best per side)
- (USDJPY,1m,UP): single-pair LGBM up-preds — **.538/.540/.545 @cov1%** (sub-breakeven; `usdjpy_1m_base_result.json`). Beat = lift a held-out-year UP CI-lo over 0.541 at a tradeable coverage.
- (USDJPY,1m,DOWN): single-pair LGBM down-preds — ~.51, **dead**. Beat = produce ANY monotone UP-of-breakeven DOWN signal.
- Breakeven 0.541. Certify only via full per-fold-refit CPCV at the operating gate (p10 ≥ 0.541, ≥~80% folds clear).

## Model of the edge (updated each iteration — THE ENGINE)
- **v0 (prior, pre-data):** 60s/1m direction near-efficient (EURUSD: 24 channels null; cross-pair none@60s). USDJPY differs structurally → faint UP-autocorrelation plausible; magnitude likely the real edge.
- **v1 (post-baseline, `usdjpy_1m_base_result.json`):** USDJPY 1m direction near-efficient (AUC .514–.520, best_iter only 114 → very little learnable signal). **DOWN dead** (no monotone lift, ≤.51, <.50 at tight cov & OOS — rally-selling has no edge). **UP = faint MONOTONE-in-confidence dip-buy tilt**, OOS-PERSISTENT (cov1% .538/.540/.545; 2026 the strongest, unlike EURUSD 60s where UP faded). Cause of sub-breakeven = weak base signal (AUC .52), NOT coverage. **Attack the cause:**
  1. **Strengthen the UP channel** — USDJPY has DIRECT USD-factor exposure (it IS a USD pair) + on-disk OF (`features_of/USDJPY`). Cross-pair USDJPY-target (A6a) is the most mechanism-distinct lever even though EURUSD cross-pair was none@60s.
  2. **More data / tuning** — baseline subsampled hard (stride24→148k); the faint ranking may sharpen with more train + leaf tuning (A1a).
  3. **Regime gate** — UP edge may concentrate in compression / specific sessions (Tokyo dip-buy?) (A2a/A3a).
  4. **Magnitude→direction bridge for UP** — if large UP moves are more sign-predictable (E1a→I3).
- Next experiment chosen to ATTACK cause #1 (highest mechanism-distinctness) + #2 (cheapest), in parallel-research + serial-fit.
- **v2 (post cross-pair + more-data + raw-conditional analysis):**
  - **Cause #1 (cross-pair, A6a) PARTIAL/KILLED:** xpof lifts AUC (.524) + mid-cov win-rate in 2024-25 (lead-lag ll_GBPUSD/AUDUSD, own_r1, hour, OF_kyle top) but DILUTES the OOS tight tail (cov1% UP .513 < baseline .545); cross-pair lead-lag is regime-dependent (not 2026). Not the UP lever.
  - **Cause #2 (data-starved, A1a) CONFIRMED partial:** stride6+leaves255 (592k train) lifts UP cov2% to .547/.534/.538 (worst .534, +1pt) — signal WAS data-starved; still sub-BE on worst year.
  - **MECHANISM nailed (Tier-1 raw conditional, no model):** USDJPY 1m signed edge = **dip-buy mean-reversion**, OOS-stable: after-dip UP-rate .515/.510/.517 vs after-rally .495/.495/.497; **amplified by compression + Tokyo session** (comp×Tokyo×dip .513/.520/.524, OOS-best). DOWN = after-rally down-rate only ~.505 (weak). This is JPY-home-session importer/carry dip-buying. The GBM tail selects within this.
  - **Now testing:** regime-gated UP filter (dip×comp×Tokyo × model-confidence) on the s6/l255 model — does gating to the signed regime push the worst year UP CI-lo over 0.541? (`usdjpy_1m_regime.py`, running).
  - **If regime insufficient:** the edge is a thin (~.52-.54) dip-buy reversion; remaining shots = Tier-N genuinely-new signed channels (Tokyo fixing/gotobi flow, MoF intervention reversion, JPY-cross triangular residual — from discovery workflow) + magnitude→direction bridge + Tier-I (seed-ens/ACI) variance reduction. Then honest exhaustion if all dry.
- **v3 (post regime-gate, A3a/A8a KILLED):** Regime gating + model confidence does NOT lift either side over breakeven. Decisive finding: the dip×comp×Tokyo UP gate looked strong on VAL (worst-half .558) but **OOS collapsed to .495** — the GBM's confidence ranking *within* the regime anti-transfers (corr(VAL,OOS)=−.54). The ONLY OOS-stable signed structure is the **raw regime base-rate ~.52** (dip-buy reversion, no model), which is **sub-breakeven**. So model-based direction extraction has hit its wall: AUC .52 caps it, and confidence selection is OOS-harmful. **Remaining genuinely-distinct shots:** (1) Tier-N new signed CHANNELS not yet in the feature set (Tokyo fixing/gotobi calendar flow, MoF intervention level-reversion) — these add INFORMATION, not just re-rank existing feats; (2) magnitude E1 (likely real, sign-invariant) + an honest mag→direction bridge check; (3) Tier-I seed-ensemble (variance reduction might stabilize the tail). NOTE: triangular JPY-cross residual is DATA-BLOCKED (no EURJPY/GBPJPY on disk).

## FIRST-TO-RUN queue (ROI order; reasoned, not blind permutation)
1. **BASE** — single-pair LGBM baseline (`usdjpy_1m_base.py`). RUNNING. Establishes: is there ANY 1m signal; UP/DOWN asymmetry; where (if anywhere) a coverage gate clears breakeven.
2. **A8a up/down FILTER + coverage curve** — if BASE shows a UP tilt (as EURUSD 60s did), the deliverable UP predictor is the up-only filter on the symmetric model; map the coverage curve to find where UP CI-lo clears 0.541.
3. **A1a GBM knob sweep** — only if BASE AUC > ~0.515 (signal worth tuning); pick by worst-VAL-half.
4. **A6a cross-pair USDJPY-target (xp/xpbase/xpof)** — USDJPY IS a USD pair (direct USD-factor exposure); even though EURUSD cross-pair was none@60s, USDJPY's own USD loading + OF (features_of/USDJPY exists) is a distinct channel. Low prior at 1m but mechanism-distinct → run once.
5. **E1a magnitude |ret60|≥Q** — the likely-real edge (sign-invariant); record in MAGNITUDE_FINDINGS.md. Establishes whether a magnitude→direction bridge (I3) is even worth trying.
6. **A2a gate sweep / A3a reversion-compression specialist** — regime gating to concentrate the UP edge.
7. **F4a residualized-target / A8b specialist** — controls; expected to confirm subset-training hurts ranking.

## Discovery rounds (loop until K=2 dry) — USDJPY-specific mechanisms to research
- USDJPY carry/risk-on-off lead from cross-asset (Nikkei/JGB/US2Y) — likely external-data-gated.
- MoF/BoJ intervention-zone reversion (price-level conditioning near round numbers / prior intervention bands) — bar-computable; signed reversion mechanism.
- Tokyo-fixing (gotobi 5/10-day) directional flow — calendar gate, bar-computable, signed.
- JPY-cross triangular residual (USDJPY vs EURUSD×EURJPY) — USD-canceling, on-disk if EURJPY present (check).
- (mine `_CORPUS_INDEX.md` + `_extracted_levers.json` for 1m/60s direction levers not yet subsumed.)

## KILLED / subsumed (with Tier-1 cite) — fill as rows complete
_(none yet)_
