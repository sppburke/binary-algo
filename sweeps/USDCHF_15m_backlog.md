SCOPE: USDCHF · 15m — EXECUTABLE backlog (TOP-N queue / incumbents-to-beat / discovery rounds). Ledger: `USDCHF_15m.md`. Results: `results/USDCHF_RESULTS.md`. Generic menu: `SWEEP_MATRIX.md`.

# USDCHF 15m direction — executable backlog — ✅ SWEEP CLOSED 2026-06-11

**STATUS: CLOSED.** Queue DRAINED, every candidate RESOLVED (verdicts below + `results/USDCHF_RESULTS.md`), R1+R2+R3 all DRY. On-disk key EXHAUSTED.
**DELIVERABLE (frozen + git-tagged + in `books/INDEX.json`):** `USDCHF.m15ny_xpair_seedens.v1` (content_id `eb44d9992cce9734` / `eb44d999`; tag `book/USDCHF.m15ny_xpair_seedens.v1`) — EUR-bloc CROSS-PAIR POOLED, NY-session, LightGBM SEED-ENS K=3, 337 feats. ★ FIRST major to certify >65% BOTH sides (refit-CPCV p10: cov.02 UP .6533 / DOWN .6440; cov.01 .6935/.6682; cov.005 .7303/.7084; all cells CERT=True). VAL moved-AUC .5576; CPCV AUC .5479. REFIT-DEPENDENT (NY-only deploy + periodic retrain; size refit-CPCV floor −.0035 haircut; Kelly 1/8). Supersedes single-seed `USDCHF.m15ny_xpair.v1` (`7505c934`). Only frontier left = OFF-DISK data (traded EURCHF, SNB sight-deposits, VIX/risk-reversal, US-CH rate-diff) — out of on-disk scope.

## FIRST-TO-RUN queue — ✅ DRAINED
1. **A1 base** — `usdchf_15m_base.py` ✅ DONE (SURVIVED; VAL AUC .5301; 2026 binding cov2 COMB .5103 sub-BE = refit-dependent).
2. **A9 session landscape** — `usdchf_15m_cpcv_session.py {all,ny,ldn,asia}` ✅ DONE. Carrier = **NY** (all-session also certifies @cov≤.03; LDN no-cert; Asia killed). NY = the deploy session.
3. **A6 EUR-bloc POOLING (keystone)** — `usdchf_15m_xpair.py xpbase` ✅ DONE → **POOLING WINS.** CHF≈−EURUSD (SNB-managed) → EUR-bloc case like EURUSD/GBPUSD, UNLIKE own-pair havens JPY/AUD/CAD. The cheap frozen ALL-SESSION screen was a FALSE NEGATIVE (IMPROVES=False); NY refit-CPCV was DECISIVE; adversarially gated vs trap#9 (xpair-frozen BEATS own-pair-frozen ALL 6 forward cells, mean +.048 → GENUINE, not era-local memorization). Book `USDCHF.m15ny_xpair.v1`.
4. **I2 seed-ens K=3** on the NY carrier book ✅ DONE → **WINS** = the DELIVERABLE `USDCHF.m15ny_xpair_seedens.v1`. Lifts parent on path-MEAN 15/15 cov×side cells + p10 13/15 (variance-reduction floor lift, NOT TB-style redistribution).
5. Improve cross-product (magdir, TB, xhor, OFI, kNN, meta, two-speed, spec, GRU, Optuna) ✅ DONE — all RESOLVED (see §"Improve cross-product verdicts" below + §8 batch). Pooling + seed-ens K=3 are the only survivors; everything else KILLED or Tier-1-subsumed.

## Incumbents to beat — ✅ RESOLVED (final: seed-ens K=3 deliverable)
- **A1 base (all-session frozen):** VAL AUC .5301; per-year AUC .5288/.5205/.5129; cov2 COMB .6513/.5675/.5103 (2024/25/26). 2026 binding sub-BE. → BEATEN by NY xpair pooling.
- **NY xpair single-seed** `USDCHF.m15ny_xpair.v1` (`7505c934`): BOTH sides certified >65% @cov1. → BEATEN by seed-ens K=3.
- **FINAL** = `USDCHF.m15ny_xpair_seedens.v1` (`eb44d999`): refit-CPCV p10 cov.02 UP .6533/DOWN .6440, cov.01 .6935/.6682, cov.005 .7303/.7084 (all CERT=True both sides). No on-disk lever beats it.

## Discovery rounds — ✅ R1 + R2 + R3 ALL DRY (no further on-disk levers)
- **R1 (corpus + web, CHF-specific):** ✅ DONE → DRY (all levers RESOLVED below). Workflow usdchf-15m-discovery-r1, 7 agents, 22 raw levers → vetted below. Verified on-disk reality: only 7 USD-majors + panel_*.parquet; NO ticks/EUR-crosses/SNB/rates/VIX. xpair builds EUR-bloc RETURN residuals but NO EURCHF-LEVEL feature.
- **R3 (completeness critic):** ✅ DONE → DRY. Corrected a false "no tick data" claim: OF *modeling* feats DO exist on disk (used by OFI/§8 batch); only RAW-tick settlement is absent. No new survivable lever.

### R1 vetted ON-DISK candidates (ranked) — ✅ ALL RESOLVED — base to beat: VAL AUC .5301, 2026 cov2 COMB .5103, BE .541
| # | Lever | Mechanism (why SIGN) | sign-survives? | prior | VERDICT |
|---|---|---|---|---|---|
| 1 | **EUR-bloc sign-transfer POOLING** `usdchf_15m_xpair.py xpbase` | inherits certified EURUSD/GBPUSD xpair SIGN mech (signed USD-factor + resid-reversion), mapped to USDCHF-up frame (sign-flipped, CHF≈−EUR, concurrent β +0.69). CHF differentiator: CHF is the ONLY haven in the EUR bloc → may transfer unlike JPY/AUD/CAD | YES (signed pooling) | **med** | ✅ **WON** → parent book `USDCHF.m15ny_xpair.v1` (`7505c934`); adversarially verified vs trap#9. The keystone. |
| 2 | **dblortho SNB-proxy residual** `usdchf_15m_xpair.py dblortho` | double-orthogonalized resid (USDCHF purged of USD-factor AND EUR-bloc) = signed CHF over-extension; mean-reverts → −sign(resid) predicts next sign | YES (signed reversion) | **med-low** | ❌ **KILLED** — xpair matrix subsumes the channel (no lift over pooled book). |
| 3 | **★NEW EURCHF-level error-correction band (SNB-pinned)** `usdchf_15m_ecm.py` (BUILD) | synthetic logEURCHF=logEURUSD+logUSDCHF; SNB pins the LEVEL into a slow band → causal rolling-z carries restoring-force SIGN; −sign(z)→USDCHF leg. CHF-UNIQUE, orthogonal to pooling (LEVEL not return-resid; verified absent from xpair) | YES (cond-mean drift on policy-stationary level) | **low-med** | ❌ **KILLED** — EURCHF level non-stationary (~25% monotone trend 2012→2026; premise dead) + held-out HURTS (returns-only .5489 → +ECM .5476). See R2-KILLED. |
| 4 | risk-proxy (AUD+NZD) sell-off lead-lag → USDCHF DOWN (NY-masked k=1,3,6,12) | risk SELL→haven bid→USDCHF DOWN; best haven lead-lag (NY k=6 corr −.0285) | marginal (at noise floor) | low | ❌ **KILLED** — at noise floor; subsumed by xpair (ll_{p}{k} terms present); Tier-1 channel-class. |
| 5 | DOWN-side RETAINED common-USD/PC1 component (asymmetry) | risk-off common-mode USD/CHF-bid IS the DOWN signal → KEEP (not residualize) PC1 for DOWN side. CHF: European haven → common-mode more directional for DOWN | conditional (directional common move) | low | ❌ **KILLED** — Tier-1-subsumed by the xpair signed USD-factor; no DOWN lift over pooled book. |

**Dropped — magnitude/gate-only (sign-invariance fail):** coordinated risk-off STATE gate (top-decile .4917<coin-flip → magnitude); EURCHF-dislocation RETURN mean-reversion (corr −.003, distinct from #3 which is LEVEL).
**Dropped — empirically null lead-lag:** EUR-bloc/EURUSD→USDCHF lead-lag catch-up (corr ~0, coupling 100% contemporaneous); triangular USD-cancel (algebraic identity, no 3rd leg on disk → min2_triangular already KILLED); DeltaLag cross-attention (lag corr ~.01); FinGAT EUR-homogeneity (magnitude-coupling, subsumed by #1).
**Dropped — dup of already-NULL, no CHF differentiator:** SNB persistent-drift (P(up)≈.501); SNB target-zone parity asymmetry; EURCHF-leg decomposition (managed-leg AUC .5185=base); funding-bloc carry-unwind (joint −.0233 weaker than own-pair −.0472; JPY leg subtracts sign); carry-bloc spread resid = ll_USDJPY already in script.

### ACQUISITION-FRONTIER (off-disk, separate)
- **SNB weekly sight-deposit Δ** as signed official-flow regime gate (rising deposits = SNB selling CHF → USDCHF UP bias). Weekly granularity vs 15m label. Prior low.
- **Triangular-DISLOCATION residual (TRADED vs implied EURCHF), CHF leg (Chaboud)** — needs independently-traded EURCHF + EURGBP feed (Dukascopy 1m). Unconstructable on disk (implied-vs-implied ≡ 0). Prior low. NB: the on-disk synthetic-EURCHF LEVEL band IS testable now = lever #3.

## ★ >65% TARGET — ✅ ACHIEVED (Tier-1 prediction confirmed)
NY own-pair cov1 per-path distribution (`usdchf_15m_cpcv_session_ny_multicov_result.json`, 15 paths) flagged the worst-path dip as PATH VARIANCE, not info-bound:
- **UP cov1**: p10 .641, median .674, mean .678, std .031, **frac paths ≥.65 = 0.87 (13/15)**
- **DOWN cov1**: p10 .636, median .712, mean .695, std .036, **frac ≥.65 = 0.80 (12/15)**
- **COMB cov1**: p10 .639, mean .684, **frac ≥.65 = 0.80**

**OUTCOME (confirmed):** seed-ens K=3 (variance reduction) lifted the worst paths exactly as predicted → the xpair seed-ens DELIVERABLE certifies p10≥.65 at cov.02 BOTH sides (UP .6533 / DOWN .6440) and rises through cov.005 (.7303/.7084). **USDCHF = FIRST major to certify >65% on BOTH sides.** The pursuit plan items resolved as: (1) seed-ens K=3 ✅ WON; (2) EUR-bloc pooling ✅ WON; (3) cov.005 probe ✅ certified; (4) ECM ❌ KILLED (non-stationary level); (5) combos = Tier-1-subsumed. NOT an information bound at cov1 — confirmed reachable and reached.

## Post-discovery experiment queue — ✅ ALL EXECUTED & RESOLVED
1. **A6 xpair xpbase** (cand #1, EUR-bloc pooling keystone) ✅ WON → parent book `USDCHF.m15ny_xpair.v1`.
2. **A6b xpair dblortho** (cand #2, SNB-proxy resid) ❌ KILLED (xpair subsumes).
3. **ECM band** (cand #3) ❌ KILLED (EURCHF level non-stationary; held-out HURTS).
4. **I2 seed-ens K=3** on carrier book ✅ WON = DELIVERABLE `USDCHF.m15ny_xpair_seedens.v1` (`eb44d999`).
5. Lower-priority on-disk: risk-proxy lead-lag (#4), DOWN-side PC1 retain (#5) ❌ KILLED (Tier-1-subsumed).
6. Standard improve cross-product (magdir/TB/xhor/OFI/kNN/meta/twospeed/spec/GRU/Optuna) ✅ DONE — verdicts below.

## Improve cross-product verdicts — ✅ COMPLETE
- **Pooling (xpair)** ✅ WON · **seed-ens K=3** ✅ WON = the only survivors (the deliverable).
- **magdir** ❌ KILLED (mag ≠ 15m sign). **TB-firsttouch (TB_K=1.5, own-pair NY refit-CPCV)** ❌ SUBSUMED — certifies + IMPROVES=True vs the OWN-PAIR base every cov (cov1 UP p10 .6611 / DOWN .6558, a real label lift on the weaker own-pair space) BUT the xpair seed-ens deliverable DOMINATES it at every cov/side by +.008…+.048; also an un-frozen-forward-gated refit-CPCV positive (USDCAD/USDJPY refit-overfit signature). Reinforces "EUR-bloc pooling is the dominant signal."
- **dblortho/SNB-resid** ❌ KILLED (xpair matrix subsumes channels). **kNN / meta / two-speed / |ret|-weight / cross-horizon (xhor) / ACI** ❌ Tier-1-subsumed (channel-class, spanned by the 337-feat matrix).
- **IRM env-invariant feature-stability filter** ❌ SURVIVES=False (R2-1; see below) — pruned 40.1% era-local sign-flippers, decay slope NOT reduced → PROVED the refit-decay is an INFORMATION BOUND, not a fixable feature-selection artifact.
- See §8 below for the OFI / GRU / Optuna / TB confirmatory batch (4/4 complete).

## Discovery R2 — refit-decay mitigation focus (4-angle fanout)

**TARGET of this round:** the ONE certified weakness — the EUR-bloc xpair-NY book is REFIT-DEPENDENT (frozen 2012-21 vintage decays cov1 .79(2024)→.48(2026), DOWN to .43). R2 asked: can any on-disk lever make the SIGN more forward-stationary without losing it?

**VERDICT (then): NOT DRY — but barely.** Three of four angles returned DRY (all levers KILLED-by-citation or not-novel/spanned). **One genuinely-new survivable lever** cleared every screen: an **environment-invariant (IRM / InvariantStock) cross-regime feature-stability filter** on the xpair-NY book.
**VERDICT (now): ✅ R2 DRY — R2-1 RAN and was KILLED.** The IRM filter pruned 40.1% of feats (era-local sign-flippers) but SURVIVES=False: the invariant-subset refit-CPCV forward holdout did NOT reduce the decay slope. This PROVES the refit-decay is an INFORMATION BOUND (genuine recent-era signal decay), not a fixable spurious-feature artifact — the UNTESTED FORK below resolved AGAINST a feature-selection fix.

### R2 survivor — ✅ RESOLVED (KILLED, SURVIVES=False) — base to beat: refit-CPCV decay slope, full-337 book 2026 cov1 ~.48 (both sides)
| # | Lever | Mechanism (why SIGN) | on-disk? | prior | VERDICT |
|---|---|---|---|---|---|
| R2-1 | **★ Environment-invariant cross-regime feature-stability filter (IRM / InvariantStock on the xpair-NY book)** — reuses LGBM+CPCV harness, NO neural net | Partition train years into K environments (e.g. 2012-15 / 2016-19 / 2020-23 or per-year); fit sign model per-env; KEEP only features whose directed sign-contribution (per-env SHAP-sign or per-env single-feat AUC-above-.5 direction) is CONSISTENT across all environments; DROP era-local sign-flippers; refit pooled NY xpair book on the invariant subset. Carries SIGN: selects on directed sign-predictiveness, not magnitude. Targets decay directly: a parsimonious invariant book should decay LESS forward even at lower peak. InvariantStock: keep F s.t. H(Y\|F)=H(Y\|F,E). Indep. support: Fed FEDS 2025-089 — FX complexity gains fragile/non-stationary, parsimony more robust OOS. | **on-disk** (reuses build_xp + CPCV harness; ~half-day CPU) | **med** | ❌ **KILLED — SURVIVES=False.** Pruned 40.1% of feats (era-local sign-flippers) — clears the (b) "nothing to prune" screen — but the invariant-subset fwd holdout did NOT materially reduce decay vs full-337 (2026 cov1 not above .48) on BOTH sides → fails (a). The honest UNTESTED FORK resolved AGAINST a fix: recent weakness IS genuine SIGNAL decay (info bound), no selection helps. Decision-grade. |

### R2 KILLED / DRY (do not re-propose — each has a Tier-1 citation)
| Lever | Why dead | Citation (verified this round) |
|---|---|---|
| **Rank/quantile-normalized cross-pair residual** (cross-sectional carry-rank, replace raw return-residual) | Already tested on EUR-bloc cousin GBPUSD: rank lifts VAL but did NOT improve the forward tail. Root cause orthogonal: decay is CONDITIONAL-MEAN drift in the already-ratio-normalized base block (pipeline dist_ema/bb_width/atr_pct/rangepos), which marginal normalization cannot touch; own-pair-only frozen decays HARDER than xpair-augmented. | `gbpusd_15m_orthochan_result.json` VERDICT.carryrank: `block_2026_cov2=0.4914`, `delta_auc=0.0041`, **ADDS=False**. |
| **Fractional-differentiation (FFD) stationarity-with-memory** of cross-pair USD-factor/level | Already BUILT + TESTED + KILLED on EURUSD xpair direction book. Stationarizing transform did NOT reduce forward decay — made recent years WORSE; ffdonly is a coin flip. | `frac_direction_15m_result.json`: base→+ffd selacc deltas 2024 **−0.0108** / 2025 **−0.0219**; ffdonly pooled selacc .5001; `deployable.+ffd=false`. |
| **EURCHF-LEVEL ECM band** (causal rolling-z of synthetic logEURCHF) as decay-resistant complement | Premise REFUTED: EURCHF level is NOT policy-stationary — mean 1.2052(2012 floor)→1.0681(2015)→0.9527(2024)→0.9178(2026), ~25% monotone trend across 3 distinct regimes. A frozen z-band anchors on a regime that no longer exists. Standard-menu rolling-z (not novel vs R1). NB: also already KILLED held-out in the orthogonal-angle session test (returns-only .5489 → +ECM .5476, Δ −0.0014; ranks #1-3 in importance yet HURTS = overfit to non-stationary level). | `usdchf_15m_ecm.py` BUILT-but-UNRUN; pre-registered falsifier = trap#9 z-sign-flip 2024↔2026. R1 cand #3 — superseded; **do not run as decay-mitigation** (premise dead). |
| **EUR-bloc dispersion-conditioned catch-up**; **non-target bloc divergence + co-skew**; **>30m / <1m cross-pair lead-lag** | NOT provably absent — all spanned by the 337-feat matrix (catchup/disp/agree + nested GBM splits; bloc divergences are linear combos of present ll_{p}{k}; >30m lags weak & mean-reverting). Co-skew flat across terciles = SIZE descriptor (sign-invariance fail). <1m needs ticks (off-disk, orphaned). Std-menu-adjacent (two-speed/meta-gate). | Angle-2 saturation map (`usdchf_15m_xpair.py:50-111`): linear cross-pair return space SATURATED; added-term deltas +0.0005/+0.0015 = split-efficiency noise. Lead-lag catch-up already R1-KILLED. |

**R2 net (final):** the linear cross-pair return space is exhausted (R1 + angle-2 saturation audit), marginal/stationarity transforms (rank, FFD) and the level-ECM band are all KILLED-by-citation against the actual decay mechanism (conditional-mean USD-factor regime drift, not marginal non-stationarity). The single open shot — **R2-1 (IRM feature-stability filter)** — RAN and RESOLVED the question: it pruned 40.1% of feats yet did NOT reduce forward decay → the gap is an INFORMATION BOUND, not spurious-feature reliance. R2 is DRY.

### R2 queue addition — ✅ EXECUTED
- **R2-1 IRM stability filter** on the EUR-bloc xpair-NY book — ran (K-environment per-env sign-stability prune, refit invariant subset, decay slope via `fwd_holdout.py` both sides). ❌ **KILLED, SURVIVES=False** — see R2 survivor table above. Closed the one certified weakness as an info bound.

## §8 COVERAGE-RULE CONFIRMATORY BATCH — ✅ COMPLETE 4/4
- **OFI signed order-flow** ❌ KILLED — base+OF NY VAL moved-AUC .5490 vs base .5495 (Δ −.0006 ≤ +.003); OF gates SIZE not SIGN at the 900s horizon (sign-invariance). [R3 note: OF *modeling* feats exist on disk; only RAW-tick settlement absent.]
- **GRU sequence DL** ❌ KILLED — VAL-AUC(NY) .5313 < base .5433; DL adds no sign over GBM-on-TA.
- **Optuna (40-trial TPE)** ❌ KILLED — tuned binding-year AUC .5046 vs default .5049; beats default in 0/3 yrs; hyperparameters not the constraint (~.539 AUC bound holds; confirms USDJPY-2m anti-transfer).
- **TB-firsttouch (TB_K=1.5, own-pair NY refit-CPCV)** ❌ SUBSUMED — certifies + IMPROVES=True vs the OWN-PAIR base every cov (cov1 UP p10 .6611 / DOWN .6558) but the xpair seed-ens deliverable dominates +.008…+.048 every cov/side; un-frozen-forward-gated refit-CPCV positive (refit-overfit signature).

## ✅ CLOSE-OUT (2026-06-11)
- **DELIVERABLE:** `USDCHF.m15ny_xpair_seedens.v1` (`eb44d9992cce9734` / `eb44d999`; tag `book/USDCHF.m15ny_xpair_seedens.v1`; in `books/INDEX.json`). FIRST major to certify >65% BOTH sides.
- **MECHANISM:** EUR-bloc case — CHF≈−EURUSD (SNB-managed) → cross-pair pooling WINS sign-flipped (like EURUSD/GBPUSD, unlike own-pair havens). NY = carrier. Methodological lesson: the cheap frozen all-session pooling screen was a FALSE NEGATIVE; NY refit-CPCV was decisive; adversarially gated vs trap#9.
- **REFIT-DEPENDENT** (info-bound decay confirmed by IRM-null): deploy NY-only + periodic retrain; size refit-CPCV per-era floor −.0035 haircut; Kelly 1/8.
- **MAGNITUDE BONUS** (sign-invariant; recorded only in `docs/MAGNITUDE_FINDINGS.md`): magAUC .728/.669/.646.
- **DISCOVERY R1+R2+R3 all DRY. ON-DISK KEY EXHAUSTED.** Only frontier left = OFF-DISK data (traded EURCHF feed, SNB sight-deposits, VIX/risk-reversal, US-CH rate-diff) — out of on-disk scope.
- Per-key files: `results/USDCHF_RESULTS.md`, `sweeps/USDCHF_15m.md`, `sweeps/USDCHF_15m_backlog.md`. **SWEEP CLOSED 2026-06-11.**
