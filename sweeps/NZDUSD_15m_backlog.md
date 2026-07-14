SCOPE: NZDUSD · 15m — EXECUTABLE backlog (TOP-N queue, incumbents-to-beat, discovery rounds). Backlog is ALIVE — update after each run. See `NZDUSD_15m.md` for ledger + mechanistic model.

# NZDUSD 15m backlog

## Active queue (priority order)

| # | id | lever | note | prior | status |
|---|-----|-------|------|-------|--------|
| — | FREEZE | nzdusd_15m_freeze_ny.py nseed=3 | ✅ DONE — content_id=f599708e; test24 COMB .622 DOWN .718; test25 COMB .602 DOWN .640; oos .519 (REFIT-DEPENDENT) | high | ✅ DONE |
| 2 | A9-asia | Asia refit-CPCV (completeness) | ❌ **KILLED 2026-06-11** — UP p10=.5154 frac=0.60; DOWN p10=.5199 frac=0.60 @cov3%. AUC mean≈.519. Own-pair Asia pattern confirmed (4th major). | low | ❌ KILLED |
| 3 | A6 | AUD-cousin Antipodean xpair CPCV | ⚠️ PARTIAL 2026-06-11 — DOWN CERT p10=.5772@cov2% (frac=1.0 all covs); UP NOT CERT p10=.5371@cov2% (frac=.733; 2 bad paths). AUC=.5345. NO SUPERSEDE (UP fails + DOWN p10 < .5803). Directional asymmetry: AUDUSD = Pacific risk-off → DOWN only. | low-med | ⚠️ DONE (partial) |
| 3b | A6b | AUD-xpair seed-ens K=3 (271 feats) | ❌ **ERA-STRUCTURAL 2026-06-11** — UP p10@cov2%=.5298 < .541 (falsifier triggered). UP CERT cov3%/cov1% only. Path 15 (g[4,5]) regressed .5462→.5088 under K=3 — recent era UP worsens with averaging. DOWN CERT all covs (p10=.5782). AUC=.5356. Xpair UP avenue EXHAUSTED. | med | ❌ DONE (no supersede) |
| 3c | I1-ACI | ACI adaptive-conformal gate | ❌ **KILLED 2026-06-11** — `nzdusd_15m_aci.py`, 51s. Fixed OOS .509; ACI OOS .521 (both <BE=.541, CI overlap). ACI trades 2× more at lower selectivity; test24 fixed .635 vs ACI .558 (−.077). Dilution not selection. Own-pair family: AUDUSD "both pairs" kill confirmed. `nzdusd_15m_aci_result.json` | low | ❌ KILLED |
| 4 | §8-OFI | signed order-flow screen | ❌ **KILLED (Tier-1 own-pair) 2026-06-11** — `nzdusd_15m_ofi_screen.py`: 18 pre-computed OF features (OF_of_norm/sum 1/3/5/10/15/30m + Kyle's λ 5/15 + uptick 5/15 + accel/persist) added to base 239 = 257 total. VAL AUC .5160 vs base .5219 = **−.0059** (hurts). Dilutes base signal at colsample_bytree=0.5. CKS tick data subsumed by `features_of`. 6th major confirmed: OFI gates SIZE not SIGN. | low | ❌ KILLED (T1) |
| 5 | §8-GRU | DL sequence GRU | ⊘ **SUBSUMED** — DL adds no directional signal over GBM-on-TA for all 5 prior majors; pattern is definitive. | low | ⊘ SUBSUMED |
| 6 | §8-Optuna | Hparam search | ⊘ **SUBSUMED** — AUC bound ~.535 confirmed for every major; USDCHF ran Optuna explicitly and got .5046 vs default .5049. Hparams not the constraint. | low | ⊘ SUBSUMED |
| 7 | §8-TB | triple-barrier TRAIN-label | ⊘ **SUBSUMED** — TB improves own-pair base but subsumed by xpair/seed-ens for majors with pooling; xpair failed so TB added value is moot. | low | ⊘ SUBSUMED |
| I5 | seed-ens K=8 | seed depth escalation | ⊘ **SUBSUMED 2026-06-11** — `audusd_15m_cpcv_session_ny_seedens8_result.json` (own-pair Antipodean Tier-1): K=8 vs K=3 cov2% UP p10=.5908 vs .596 (−.005 REGRESS); DOWN .6124 vs .596 (+.016); sign-inconsistent. Seed lever saturated past K=3 for own-pair space. AUDUSD starts from .596/.596; NZDUSD projected ≤.591/.612 — far below .65 target. | low | ⊘ SUBSUMED |
| — | USDCAD xpair | commodity bloc xpair | ⊘ **VAL SUBSUMED 2026-06-11** — `nzdusd_15m_usdcad_xpair_screen.py`. VAL AUC .5299 vs base .5219 = **+.0080** lift. Sub-.010 escalation threshold. Seed-ens K=3 TRIED 2026-06-11: .5292 (WORSE — seed diversity hurts in 478-feat space; seeds 0/7 find worse optima than seed 42). Dual-xpair AUDUSD+USDCAD TRIED 2026-06-11: .5271 (WORST — dilution; 36%/31%/32% importance split, no additive signal). PATTERN PROVEN: every combination larger than own-pair 239-feat regresses. AUC ceiling confirmed at .5299 (single-seed xpair); own-pair K=3 (.5362) DOMINATES all xpair combinations. | very low | ⊘ EXHAUSTED |
| — | Arch-ens K=9 | architectural diversity ensemble | ⊘ **VAL SUBSUMED 2026-06-11** — `nzdusd_15m_arch_ens_screen.py`. 3 seeds × 3 num_leaves (127/255/511) = 9 models. K=9 full ensemble: .5254. Per-arch K=3: 127=.5258, 255=.5256, 511=.5244. Full K=9 (.5254) WORSE than K=3 standard (.5256) — weaker architectures dilute. Standard 255-leaf K=3 is global optimum for 239-feat space. Architectural diversity adds zero independent signal. | none | ⊘ EXHAUSTED |
| A1-XGB | XGBoost A1 | different GBM algorithm (level-wise vs leaf-wise) | ⊘ **VAL SUBSUMED 2026-06-11** — `nzdusd_15m_xgb_screen.py`. XGB single-seed: .5241; LGB K=3: .5256; best blend α=0.7: .5258. LGB/XGB corr=.8809 — 88% prediction overlap leaves only 12% error decorrelation (+.0002 lift). ZERO INDEPENDENT SIGNAL. A1 fully closed: LGB+XGB+different architectures all SUBSUMED at Tier-1. | none | ⊘ EXHAUSTED |
| D7-semi | D7 RS semivariance | Patton-Sheppard RS+/RS- (Tier N novel) | ⊘ **VAL SUBSUMED 2026-06-11** — `nzdusd_15m_semivar_screen.py`. RS+/RS-/RS_diff at 15/30/60m windows, 248 feats. AUC .5240 vs base .5257 = **−.0017 (HURTS)**. RS features 4.82% importance share but steal colsample from stronger predictors. Raw target corr: ±.007 range. Confirms NOVEL_METHODS_RESEARCH D7 at Tier-1 own-pair. Final NOVEL_METHODS row closed. | none | ⊘ EXHAUSTED |

| D1-feats | D1 daily trend features | d1_ret_1/d1_ret_5/ema20/ema100 | ⊘ **RETRACTED+SUBSUMED 2026-06-11** — `nzdusd_15m_d1feats_clean_result.json`. LOOKAHEAD BUG: `resample("1D").last()` labels bin 00:00 UTC but holds 23:59 UTC close; ffill gave NY session bars TODAY's daily return. Buggy AUC .5471/.5724 INVALID. FIX: shift(1) on daily close. Clean result: d1_ret_1 target corr = −0.0008; AUC .5257 = base (−.0001). NO genuine 1-day momentum signal. Feature confirmed SUBSUMED with Tier-1 clean evidence. | none | ⊘ RETRACTED |

| Moments+AUDNZD | Rolling skewness/kurtosis + synthetic AUDNZD | skew/kurt (30/60/120/240 bar windows) + AUDNZD=AUDUSD/NZDUSD TA | ⊘ **VAL SUBSUMED 2026-06-11** — `nzdusd_15m_moments_screen_result.json`. 12 new feats (8 moments + 4 AUDNZD) added to base 239 = 251 total. Skew raw corr −.025→−.036 (non-trivial; negative skew predicts DOWN-continuation); AUDNZD raw corr +.028→+.033. AUC .5257 vs base .5257 = **0.0000 net lift**. Feature importance 7.79% but colsample dilution neutralizes signal already captured by return/vol features. Both moment types and synthetic AUDNZD SUBSUMED. | none | ⊘ EXHAUSTED |

| Multi-xpair (USDJPY/EURUSD/GBPUSD) | 3 cross-pair feature screens | 239 NZD + 239 cross-pair feats each (478 total); stride-6 single-seed | ⊘ **VAL SUBSUMED 2026-06-11** — `nzdusd_15m_multi_xpair_screen_result.json`. USDJPY: AUC .5276 (+.0057); EURUSD: .5265 (+.0046); GBPUSD: .5276 (+.0057). All three below .5319 escalation threshold. Closes Tier-3 gaps: EUR-bloc features KILLED (Tier-3 via USDJPY precedent I7) now Tier-1 confirmed for NZDUSD. Haven pair (USDJPY) and EUR-bloc pairs all SUBSUMED. AUC ceiling for any single xpair ≤ .5299 (USDCAD). Own-pair K=3 (.5362) DOMINATES all cross-pair feature additions. | none | ⊘ EXHAUSTED |

## Incumbents to beat (p10 floors after each stage)

| stage | incumbent | UP p10 @cov | DOWN p10 @cov | updated |
|-------|-----------|-------------|---------------|---------|
| A9-ny single-seed | NZDUSD.m15ny.v1 (superseded) | .574 @cov2 | .572 @cov2 | 2026-06-11 |
| seed-ens K=3 ← **CURRENT** | NZDUSD.m15ny_seedens.v1 | .5749 @cov2 | .5803 @cov2 | 2026-06-11 |

## Cross-pair / pooling menu

The cross-pair hypothesis for NZDUSD is specific: **Antipodean-cousin (AUDUSD) pooling**, not EUR-bloc. EUR-bloc is demoted (all-session KILLED = no EUR-bloc breadth).

| candidate | basis | action |
|-----------|-------|--------|
| AUDUSD xpair (Antipodean bloc) | r≈.88 with NZDUSD; both commodity/risk-on; NZDUSD≈0.97×AUDUSD+noise historically | After NY own-pair certifies: run NZDUSD+AUDUSD pooled xpair refit-CPCV. Test with 239+239=~400-feat pool, per-fold refit |
| USDCAD xpair (commodity bloc) | Both commodity currencies, but USD on opposite side (USDCAD vs NZDUSD); corr ~0.5–0.6. Weaker | Lower priority; try only if Antipodean pool fails |
| EUR-bloc (EURUSD/GBPUSD/USDCHF) | NZD NOT EUR-related; all-session KILLED demotes this | KILLED on prior — do NOT run unless adversarial evidence resurfaces |

## KILL / promote rules

PROMOTE to full CPCV if frozen-forward screen passes:
- Screen rule: VAL AUC > incumbent (A9-ny single-seed VAL AUC) AND 2025+2026 cov2 COMB wr > incumbent forward floor

KILL rule:
- Per side: p10 < 0.541 OR frac_clear_BE < 0.80 → KILL that side
- Both sides KILLED → KILL the experiment, move to SUBSUMED or KILLED ledger entry

## Discovery round R1 — ✅ CLOSED 2026-06-11

**Status:** CLOSED. On-disk topics covered by A6/A6b; off-disk topics externally blocked.

| topic | status | resolution |
|-------|--------|------------|
| Within-Antipodean residuals (NZDUSD orthogonal to AUDUSD) | ✅ COVERED | `aud_resid_k` (k=1..8) are in the 271-feat xpair pool; A6/A6b ran with full 271 features; UP era-structural, DOWN cert below incumbent |
| Commodity proxy (iron-ore/CRB direction) | ✅ COVERED | AUDUSD cousin A6/A6b captures commodity cross-pair signal (AUD = commodity currency); DOWN cert all covs, UP era-structural; no further commodity xpair on-disk |
| NZD-specific: dairy/GDT auction timing+direction | ⊘ EXTERNAL-BLOCKED | Not in bar features. Off-disk: GDT auction direction + surprise magnitude required. Not acquirable without external data pipeline. |
| NZD-specific: RBNZ tone/surprise magnitude | ⊘ EXTERNAL-BLOCKED | Not in bar features. Off-disk: NLP on RBNZ MPR + press conference or rate-surprise series required. |
| NZ-US rate differential (RBNZ-Fed divergence) | ⊘ **TRIED+KILLED Tier-1 2026-06-11** | `nzdusd_15m_rbnz_ocr_screen.py`: FRED IRSTCI01NZM156N (480 monthly rows) + Yahoo ^IRX (3770 daily rows). VAL AUC .5225 vs base .5219 = **+.0006** → SUBSUMED. Monthly OCR too coarse; already embedded in price. `nzdusd_15m_rbnz_ocr_screen_result.json`. |
| China PMI/trade → AUD/NZD risk-on driver | ⊘ EXTERNAL-BLOCKED | Not in bar features at bar-frequency. AUDUSD cousin A6 serves as the on-disk proxy; UP era-structural. True China PMI surprise series is off-disk. |
| **Global macro daily (VIX/DXY/US10Y/Gold) — accessible external** | ⊘ **SUBSUMED 2026-06-11** | F5 screen `nzdusd_15m_extfeat_screen.py` (259 feats): VAL AUC .5254 vs base .5219 = **+.0035** lift. Far below .010 CPCV escalation threshold. Top features: us10y_ret1d, vix_ret1d, us10y_mom5d — absorbed by 15m price-action signals. `nzdusd_15m_extfeat_screen_result.json`. |
| **Antipodean/regional daily (NZX50, ASX200, AUDNZD, Copper, HSI + F5 = 9 tickers)** | ⊘ **SUBSUMED 2026-06-11** | F6 screen `nzdusd_15m_extfeat2_screen.py` (284 feats): VAL AUC .5259 vs base .5219 = **+.0040** lift. Only +.0005 marginal over F5 despite NZD-specific tickers. AUDNZD_ret1d most informative (957). Accessible daily external data **definitively exhausted**. `nzdusd_15m_extfeat2_screen_result.json`. |

**Conclusion:** On-disk R1 dry. F5 (+.0035) + F6 (9 tickers, +.0040) + **F8 CFTC COT (+.0007)** + **F9 RBNZ OCR/NZ-US spread (+.0006)**: ALL accessible external data exhausted with Tier-1 NZDUSD evidence. **MEGA-COMBO (528 feats, all combined): .5286 — below USDCAD xpair alone (.5299)** — noise dilution proves accessible ceiling ≈ .530, 20bp below .5319 threshold. "NZ-US rate differential" converted from EXTERNAL-BLOCKED to TRIED+KILLED (Tier-1). Remaining truly externally blocked (no free API): dairy/GDT, RBNZ surprise NLP, China PMI. R1+F8+F9+MEGA CLOSED 2026-06-11. ALL accessible external data + combinations CLOSED.

## Discovery round R2 — ✅ CLOSED 2026-06-11

**Status:** CLOSED. 12 arXiv + 10 SS queries (general FX intraday direction terms). Zero applicable findings. Academic literature on NZDUSD 15m direction is effectively empty.

## Discovery round R3 — ✅ CLOSED 2026-06-11

**Status:** CLOSED. 6 NZD-specific queries (dairy/GDT, RBNZ OCR surprise, AUD-NZD correlation regime, commodity carry + COT, antipodean GBM, dairy→NZD 15m). Zero applicable papers found. NZD-specific preprint literature on 15m direction is empty. Confirms: academic frontier exhausted at K=3 dry rounds.

| query | status | finding |
|-------|--------|---------|
| NZD intraday dairy/GDT | 0 arXiv | GDT bi-weekly, not 15m operationalized |
| RBNZ OCR surprise intraday | 2 candidates (AUD, daily EM) | NOT APPLICABLE — not NZDUSD 15m |
| NZD/AUD correlation regime | 2 candidates (stock/FX daily) | NOT APPLICABLE |
| Commodity carry/COT intraday ML | 0 arXiv at intersection | ZERO applicable |
| Antipodean GBM direction prediction | 0 NZD-specific arXiv | ZERO applicable |
| Dairy price NZD 15m forecasting | 0 arXiv | GDT nexus real but no 15m method |

**Conclusion:** R3 DRY — K=3 consecutive dry discovery rounds. Loop closed.

## Completeness notes

**Data verified:** 2012–2026 parquets ✓, 239 feats ✓, 0 missing ✓, 2,667,689 rows ✓
**Session split (moved bars):** exact breakdown TBD after A9-ny (log shows 992,016 in-session NY)
**Up-rate tripwire:** A1 gate (cov5) all years ∈ [0.4995, 0.5016] — clean, no fake-flat mirage
**Tick data:** VERIFIED 2026-06-11 — `features_of/NZDUSD_*.parquet` (18 OF features, 1-min, 2012–2026) + `features_tick_xofi/NZDUSD_*_cks1s.parquet` (11.8M rows, cks_e/cks_nev) both confirmed on-disk. OFI screen run (§8, row 4): KILLED (Tier-1). CKS tick raw data subsumed by features_of aggregates.

## CLOSED measured handoff — issue #9 date refresh (2026-07-13)

- The preregistered Apr–May 2026 retrospective ended `INCONCLUSIVE` because the sparse C-DOWN endpoint was not estimable; pair source: `results/json/m15_book_refresh_53547498c599f2877a2f6616ae725f8f27e11070e3b550e82eb26ec039fb23f6_NZDUSD_replay_result.json`.
- Joint survivor set `S=[]`; no inactive candidate and no prospective shadow were opened. Do not rerun, retune, or reinterpret this spent one-look window. The incumbent and the exhausted accessible-data queue remain unchanged. Joint/shadow sources: `results/json/m15_book_refresh_53547498c599f2877a2f6616ae725f8f27e11070e3b550e82eb26ec039fb23f6_joint_replay_result.json`, `results/json/m15_book_refresh_53547498c599f2877a2f6616ae725f8f27e11070e3b550e82eb26ec039fb23f6_shadow_spec_result.json`.
