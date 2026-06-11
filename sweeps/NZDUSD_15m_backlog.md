SCOPE: NZDUSD · 15m — EXECUTABLE backlog (TOP-N queue, incumbents-to-beat, discovery rounds). Backlog is ALIVE — update after each run. See `NZDUSD_15m.md` for ledger + mechanistic model.

# NZDUSD 15m backlog

## Active queue (priority order)

| # | id | lever | note | prior | status |
|---|-----|-------|------|-------|--------|
| — | FREEZE | nzdusd_15m_freeze_ny.py nseed=3 | ✅ DONE — content_id=f599708e; test24 COMB .622 DOWN .718; test25 COMB .602 DOWN .640; oos .519 (REFIT-DEPENDENT) | high | ✅ DONE |
| 2 | A9-asia | Asia refit-CPCV (completeness) | ❌ **KILLED 2026-06-11** — UP p10=.5154 frac=0.60; DOWN p10=.5199 frac=0.60 @cov3%. AUC mean≈.519. Own-pair Asia pattern confirmed (4th major). | low | ❌ KILLED |
| 3 | A6 | AUD-cousin Antipodean xpair CPCV | Screen ESCALATE (val_auc=0.5247>0.5219); NY CPCV running PID 3539283 (271 feats) | low-med | 🔄 RUNNING |
| 4 | §8-OFI | signed order-flow screen | OF gates SIZE not SIGN @900s for all majors run; run as confirmatory | low | ⏸ DEFERRED |
| 5 | §8-GRU | DL sequence GRU | DL no sign over GBM-on-TA for all 5 majors tested; confirmatory | low | ⏸ DEFERRED |
| 6 | §8-Optuna | Hparam search | hparams not the constraint (~.535 AUC bound confirmed every major) | low | ⏸ DEFERRED |
| 7 | §8-TB | triple-barrier TRAIN-label | subsumed by xpair/seed-ens for pooling pairs; test after pooling determination | low | ⏸ DEFERRED |

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

## Discovery round R1 (pending — after NY carrier established)

Topics to mine after NY own-pair certifies:
- NZD-specific mechanisms: dairy price cycles (GDT auction timing + direction carry), RBNZ tone/surprise magnitude, NZ-US rate differential
- External alpha: China PMI surprise → AUD/NZD risk-on driver (off-disk, NZD proxies on disk?)
- Within-Antipodean residuals: NZDUSD orthogonal to AUDUSD (pure NZ-specific signal after AUD extracted)
- Commodity proxy: iron-ore / CRB already in the 239 feature set? If not, the AUD-cousin pool captures it via AUDUSD features

## Completeness notes

**Data verified:** 2012–2026 parquets ✓, 239 feats ✓, 0 missing ✓, 2,667,689 rows ✓
**Session split (moved bars):** exact breakdown TBD after A9-ny (log shows 992,016 in-session NY)
**Up-rate tripwire:** A1 gate (cov5) all years ∈ [0.4995, 0.5016] — clean, no fake-flat mirage
**Tick data:** verify on-disk (check features_of/NZDUSD_* and features_tick_xofi/NZDUSD_* — USDCHF had this; NZDUSD TBD)
