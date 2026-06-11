SCOPE: USDCHF · 15m — EXECUTABLE backlog (TOP-N queue / incumbents-to-beat / discovery rounds). Ledger: `USDCHF_15m.md`. Results: `USDCHF_RESULTS.md`. Generic menu: `SWEEP_MATRIX.md`.

# USDCHF 15m direction — executable backlog

## FIRST-TO-RUN queue (highest ROI first)
1. **A1 base** — `usdchf_15m_base.py` ✅ DONE (SURVIVED; VAL AUC .5301; 2026 binding cov2 COMB .5103 sub-BE = refit-dependent).
2. **A9 session landscape** — `usdchf_15m_cpcv_session.py {all,ny,ldn,asia}` 🔄 RUNNING (chain). Find carrier (NY vs LDN — CHF is European). Honest refit floor.
3. **A6 EUR-bloc POOLING (keystone)** — `usdchf_15m_xpair.py xpbase` — THE differentiating test (CHF≈−EURUSD → pooling may WIN unlike havens). Ready to launch when heavy slot frees. Falsifier: IMPROVES iff VAL AUC>.5301 AND 2026 cov2 COMB>.5103.
4. **I2 seed-ens K=3** on the carrier book — deliverable improve lever.
5. Improve cross-product (magdir, TB, xhor, OFI, kNN, meta, two-speed, spec, GRU, Optuna) — each vs BEST COMBO incumbent + frozen-forward adversarial check.

## Incumbents to beat
- **A1 base (all-session frozen):** VAL AUC .5301; per-year AUC .5288/.5205/.5129; cov2 COMB .6513/.5675/.5103 (2024/25/26). 2026 binding sub-BE.
- Carrier-session refit-CPCV floor: pending (A9).

## Discovery rounds
- **R1 (corpus + web, CHF-specific):** ✅ DONE (workflow usdchf-15m-discovery-r1, 7 agents, 22 raw levers → vetted below). Verified on-disk reality: only 7 USD-majors + panel_*.parquet; NO ticks/EUR-crosses/SNB/rates/VIX. xpair builds EUR-bloc RETURN residuals but NO EURCHF-LEVEL feature (→ novel lever #3).

### R1 vetted ON-DISK candidates (ranked) — base to beat: VAL AUC .5301, 2026 cov2 COMB .5103, BE .541
| # | Lever | Mechanism (why SIGN) | sign-survives? | prior | fast-KILL |
|---|---|---|---|---|---|
| 1 | **EUR-bloc sign-transfer POOLING** `usdchf_15m_xpair.py xpbase` | inherits certified EURUSD/GBPUSD xpair SIGN mech (signed USD-factor + resid-reversion), mapped to USDCHF-up frame (sign-flipped, CHF≈−EUR, concurrent β +0.69). CHF differentiator: CHF is the ONLY haven in the EUR bloc → may transfer unlike JPY/AUD/CAD | YES (signed pooling) | **med** | KILL unless VAL AUC>.5301 AND 2026 cov2 COMB>.5103 |
| 2 | **dblortho SNB-proxy residual** `usdchf_15m_xpair.py dblortho` | double-orthogonalized resid (USDCHF purged of USD-factor AND EUR-bloc) = signed CHF over-extension; mean-reverts → −sign(resid) predicts next sign | YES (signed reversion) | **med-low** | standalone .5115<BE → must win inside GBM. KILL if VAL AUC≤.5301 AND no held-out yr COMB CI-lo≥.541 |
| 3 | **★NEW EURCHF-level error-correction band (SNB-pinned)** `usdchf_15m_ecm.py` (BUILD) | synthetic logEURCHF=logEURUSD+logUSDCHF; SNB pins the LEVEL into a slow band → causal rolling-z carries restoring-force SIGN; −sign(z)→USDCHF leg. CHF-UNIQUE, orthogonal to pooling (LEVEL not return-resid; verified absent from xpair) | YES (cond-mean drift on policy-stationary level) | **low-med** | add [z,|z|,leg-shares] to base. KILL if VAL AUC≤.5301 OR z-coef sign flips 2024↔2026 OR no held-out yr COMB CI-lo≥.541 |
| 4 | risk-proxy (AUD+NZD) sell-off lead-lag → USDCHF DOWN (NY-masked k=1,3,6,12) | risk SELL→haven bid→USDCHF DOWN; best haven lead-lag (NY k=6 corr −.0285) | marginal (at noise floor) | low | KILL unless lifts VAL AUC over EUR-bloc carrier AND NY-DOWN top-decile >.52 (measured .4917 → likely DOA) |
| 5 | DOWN-side RETAINED common-USD/PC1 component (asymmetry) | risk-off common-mode USD/CHF-bid IS the DOWN signal → KEEP (not residualize) PC1 for DOWN side. CHF: European haven → common-mode more directional for DOWN | conditional (directional common move) | low | DOWN-only one-shot. KILL if retaining PC1 doesn't lift DOWN win-rate >.52 in ANY held-out yr |

**Dropped — magnitude/gate-only (sign-invariance fail):** coordinated risk-off STATE gate (top-decile .4917<coin-flip → magnitude); EURCHF-dislocation RETURN mean-reversion (corr −.003, distinct from #3 which is LEVEL).
**Dropped — empirically null lead-lag:** EUR-bloc/EURUSD→USDCHF lead-lag catch-up (corr ~0, coupling 100% contemporaneous); triangular USD-cancel (algebraic identity, no 3rd leg on disk → min2_triangular already KILLED); DeltaLag cross-attention (lag corr ~.01); FinGAT EUR-homogeneity (magnitude-coupling, subsumed by #1).
**Dropped — dup of already-NULL, no CHF differentiator:** SNB persistent-drift (P(up)≈.501); SNB target-zone parity asymmetry; EURCHF-leg decomposition (managed-leg AUC .5185=base); funding-bloc carry-unwind (joint −.0233 weaker than own-pair −.0472; JPY leg subtracts sign); carry-bloc spread resid = ll_USDJPY already in script.

### ACQUISITION-FRONTIER (off-disk, separate)
- **SNB weekly sight-deposit Δ** as signed official-flow regime gate (rising deposits = SNB selling CHF → USDCHF UP bias). Weekly granularity vs 15m label. Prior low.
- **Triangular-DISLOCATION residual (TRADED vs implied EURCHF), CHF leg (Chaboud)** — needs independently-traded EURCHF + EURGBP feed (Dukascopy 1m). Unconstructable on disk (implied-vs-implied ≡ 0). Prior low. NB: the on-disk synthetic-EURCHF LEVEL band IS testable now = lever #3.

## ★ >65% TARGET IS IN REACH AT cov1 (path-variance, not info-bound) — Tier-1 analysis
NY own-pair cov1 per-path distribution (`usdchf_15m_cpcv_session_ny_multicov_result.json`, 15 paths):
- **UP cov1**: p10 .641, median .674, mean .678, std .031, **frac paths ≥.65 = 0.87 (13/15)**
- **DOWN cov1**: p10 .636, median .712, mean .695, std .036, **frac ≥.65 = 0.80 (12/15)**
- **COMB cov1**: p10 .639, mean .684, **frac ≥.65 = 0.80**

**Read:** 80–87% of paths ALREADY clear 65%; p10 sits at .64 only because the worst 1–2 paths dip to ~.62. The p10→mean gap (.04) is PATH VARIANCE. **seed-ens K=3 (variance reduction, std~.03) is the precise tool to lift the worst paths → p10 over .65** → USDCHF could be the FIRST major to CERTIFY p10≥.65 at cov1. **>65% PURSUIT PLAN:** (1) seed-ens K=3 (compress worst paths); (2) EUR-bloc pooling if it lifts (running); (3) cov0.005 probe (already added to xpair-CPCV run); (4) ECM orthogonal signal; (5) combos. NOT an information bound at cov1 — actively reachable.

## Post-discovery experiment queue (after session landscape → carrier)
1. **A6 xpair xpbase** (cand #1, EUR-bloc pooling keystone) — ready, queued for heavy slot.
2. **A6b xpair dblortho** (cand #2, SNB-proxy resid).
3. **ECM band** (cand #3, NEW — build usdchf_15m_ecm.py).
4. **I2 seed-ens K=3** on carrier book (deliverable lever).
5. Lower-priority on-disk: risk-proxy lead-lag (#4, DOWN), DOWN-side PC1 retain (#5).
6. Standard improve cross-product (magdir/TB/xhor/OFI/kNN/meta/twospeed/spec/GRU/Optuna) — each forked individually (cross-pair PAIRS-list fix), vs BEST COMBO + frozen-forward.

## Discovery R2 — refit-decay mitigation focus (4-angle fanout)

**TARGET of this round:** the ONE certified weakness — the EUR-bloc xpair-NY book is REFIT-DEPENDENT (frozen 2012-21 vintage decays cov1 .79(2024)→.48(2026), DOWN to .43). R2 asked: can any on-disk lever make the SIGN more forward-stationary without losing it?

**VERDICT: NOT DRY — but barely.** Three of four angles returned DRY (all levers KILLED-by-citation or not-novel/spanned). **One genuinely-new survivable lever** cleared every screen: an **environment-invariant (IRM / InvariantStock) cross-regime feature-stability filter** on the xpair-NY book. It is novel vs R1 and the standard menu, on-disk, and directly attacks the diagnosed decay mechanism. Honest prior: **med** — it has a real fork that could null it (if recent-era weakness is genuine signal/info-bound decay, no feature selection helps; that fork is empirically untested).

### R2 survivor (the only one) — base to beat: refit-CPCV decay slope, full-337 book 2026 cov1 ~.48 (both sides)
| # | Lever | Mechanism (why SIGN) | on-disk? | prior | fast-KILL |
|---|---|---|---|---|---|
| R2-1 | **★ Environment-invariant cross-regime feature-stability filter (IRM / InvariantStock on the xpair-NY book)** — reuses LGBM+CPCV harness, NO neural net | Partition train years into K environments (e.g. 2012-15 / 2016-19 / 2020-23 or per-year); fit sign model per-env; KEEP only features whose directed sign-contribution (per-env SHAP-sign or per-env single-feat AUC-above-.5 direction) is CONSISTENT across all environments; DROP era-local sign-flippers; refit pooled NY xpair book on the invariant subset. Carries SIGN: selects on directed sign-predictiveness, not magnitude. Targets decay directly: a parsimonious invariant book should decay LESS forward even at lower peak. InvariantStock: keep F s.t. H(Y\|F)=H(Y\|F,E). Indep. support: Fed FEDS 2025-089 — FX complexity gains fragile/non-stationary, parsimony more robust OOS. | **on-disk** (reuses build_xp + CPCV harness; ~half-day CPU) | **med** | KILL if (a) invariant-subset refit-CPCV fwd holdout (`fwd_holdout.py`, train≤Y→test Y+1,Y+2) does NOT reduce decay slope vs full-337 (2026 cov1 not materially above .48) on BOTH sides, OR (b) filter removes <~10% of feats / leaves AUC unchanged (=all-invariant, nothing to prune). Honest cap: if recent weakness is genuine SIGNAL decay (info bound) not spurious-feature reliance, no selection helps — UNTESTED FORK. |

### R2 KILLED / DRY (do not re-propose — each has a Tier-1 citation)
| Lever | Why dead | Citation (verified this round) |
|---|---|---|
| **Rank/quantile-normalized cross-pair residual** (cross-sectional carry-rank, replace raw return-residual) | Already tested on EUR-bloc cousin GBPUSD: rank lifts VAL but did NOT improve the forward tail. Root cause orthogonal: decay is CONDITIONAL-MEAN drift in the already-ratio-normalized base block (pipeline dist_ema/bb_width/atr_pct/rangepos), which marginal normalization cannot touch; own-pair-only frozen decays HARDER than xpair-augmented. | `gbpusd_15m_orthochan_result.json` VERDICT.carryrank: `block_2026_cov2=0.4914`, `delta_auc=0.0041`, **ADDS=False**. |
| **Fractional-differentiation (FFD) stationarity-with-memory** of cross-pair USD-factor/level | Already BUILT + TESTED + KILLED on EURUSD xpair direction book. Stationarizing transform did NOT reduce forward decay — made recent years WORSE; ffdonly is a coin flip. | `frac_direction_15m_result.json`: base→+ffd selacc deltas 2024 **−0.0108** / 2025 **−0.0219**; ffdonly pooled selacc .5001; `deployable.+ffd=false`. |
| **EURCHF-LEVEL ECM band** (causal rolling-z of synthetic logEURCHF) as decay-resistant complement | Premise REFUTED: EURCHF level is NOT policy-stationary — mean 1.2052(2012 floor)→1.0681(2015)→0.9527(2024)→0.9178(2026), ~25% monotone trend across 3 distinct regimes. A frozen z-band anchors on a regime that no longer exists. Standard-menu rolling-z (not novel vs R1). NB: also already KILLED held-out in the orthogonal-angle session test (returns-only .5489 → +ECM .5476, Δ −0.0014; ranks #1-3 in importance yet HURTS = overfit to non-stationary level). | `usdchf_15m_ecm.py` BUILT-but-UNRUN; pre-registered falsifier = trap#9 z-sign-flip 2024↔2026. R1 cand #3 — superseded; **do not run as decay-mitigation** (premise dead). |
| **EUR-bloc dispersion-conditioned catch-up**; **non-target bloc divergence + co-skew**; **>30m / <1m cross-pair lead-lag** | NOT provably absent — all spanned by the 337-feat matrix (catchup/disp/agree + nested GBM splits; bloc divergences are linear combos of present ll_{p}{k}; >30m lags weak & mean-reverting). Co-skew flat across terciles = SIZE descriptor (sign-invariance fail). <1m needs ticks (off-disk, orphaned). Std-menu-adjacent (two-speed/meta-gate). | Angle-2 saturation map (`usdchf_15m_xpair.py:50-111`): linear cross-pair return space SATURATED; added-term deltas +0.0005/+0.0015 = split-efficiency noise. Lead-lag catch-up already R1-KILLED. |

**R2 net:** the linear cross-pair return space is exhausted (R1 + angle-2 saturation audit), marginal/stationarity transforms (rank, FFD) and the level-ECM band are all KILLED-by-citation against the actual decay mechanism (conditional-mean USD-factor regime drift, not marginal non-stationarity). The single open shot is **R2-1 (IRM feature-stability filter)** — it is the only lever that asks WHICH features are sign-INVARIANT across regimes rather than which matter (R1) or which reduce variance era-agnostically (std menu). It either closes the forward-decay gap or confirms the gap is an information bound; both outcomes are decision-grade.

### R2 queue addition (insert after I2 seed-ens, before std cross-product)
- **R2-1 IRM stability filter** on the EUR-bloc xpair-NY book — partition K environments, per-env sign-stability prune, refit invariant subset, evaluate decay slope via `fwd_holdout.py` on BOTH sides. KILL per fast-KILL above. ~half-day CPU. **Highest-value open problem** (directly targets the one certified weakness).
