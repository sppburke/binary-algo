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
| 4 | §8-OFI | signed order-flow screen | ⊘ **SUBSUMED** — OFI gates SIZE not SIGN for all 5 prior majors (EURUSD/GBPUSD/USDCHF/USDCAD/AUDUSD); definitively not directional at 15m horizon. | low | ⊘ SUBSUMED |
| 5 | §8-GRU | DL sequence GRU | ⊘ **SUBSUMED** — DL adds no directional signal over GBM-on-TA for all 5 prior majors; pattern is definitive. | low | ⊘ SUBSUMED |
| 6 | §8-Optuna | Hparam search | ⊘ **SUBSUMED** — AUC bound ~.535 confirmed for every major; USDCHF ran Optuna explicitly and got .5046 vs default .5049. Hparams not the constraint. | low | ⊘ SUBSUMED |
| 7 | §8-TB | triple-barrier TRAIN-label | ⊘ **SUBSUMED** — TB improves own-pair base but subsumed by xpair/seed-ens for majors with pooling; xpair failed so TB added value is moot. | low | ⊘ SUBSUMED |
| I5 | seed-ens K=8 | seed depth escalation | ⊘ **SUBSUMED 2026-06-11** — `audusd_15m_cpcv_session_ny_seedens8_result.json` (own-pair Antipodean Tier-1): K=8 vs K=3 cov2% UP p10=.5908 vs .596 (−.005 REGRESS); DOWN .6124 vs .596 (+.016); sign-inconsistent. Seed lever saturated past K=3 for own-pair space. AUDUSD starts from .596/.596; NZDUSD projected ≤.591/.612 — far below .65 target. | low | ⊘ SUBSUMED |
| — | USDCAD xpair | commodity bloc xpair | **NOT RUN — LOW PRIOR.** NZDUSD/USDCAD corr~.55 (vs AUD .88); AUDUSD (stronger) failed UP → USDCAD (weaker) near-zero probability of lifting UP above incumbent .5749. >65% needs off-disk external data (dairy/GDT, RBNZ, China PMI). | very low | ⊘ NOT RUN |

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
| NZ-US rate differential (RBNZ-Fed divergence) | ⊘ EXTERNAL-BLOCKED | Not in bar features. Off-disk: NZ 2y yield or RBNZ OCR series required. |
| China PMI/trade → AUD/NZD risk-on driver | ⊘ EXTERNAL-BLOCKED | Not in bar features at bar-frequency. AUDUSD cousin A6 serves as the on-disk proxy; UP era-structural. True China PMI surprise series is off-disk. |

**Conclusion:** On-disk R1 is dry. External data frontier (dairy/GDT, RBNZ, NZ-US rate diff, China PMI) is the only path to AUC lift above the ~.535 on-disk ceiling. R1 CLOSED.

## Completeness notes

**Data verified:** 2012–2026 parquets ✓, 239 feats ✓, 0 missing ✓, 2,667,689 rows ✓
**Session split (moved bars):** exact breakdown TBD after A9-ny (log shows 992,016 in-session NY)
**Up-rate tripwire:** A1 gate (cov5) all years ∈ [0.4995, 0.5016] — clean, no fake-flat mirage
**Tick data:** verify on-disk (check features_of/NZDUSD_* and features_tick_xofi/NZDUSD_* — USDCHF had this; NZDUSD TBD)
