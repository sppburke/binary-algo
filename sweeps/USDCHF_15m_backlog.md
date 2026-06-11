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

## Post-discovery experiment queue (after session landscape → carrier)
1. **A6 xpair xpbase** (cand #1, EUR-bloc pooling keystone) — ready, queued for heavy slot.
2. **A6b xpair dblortho** (cand #2, SNB-proxy resid).
3. **ECM band** (cand #3, NEW — build usdchf_15m_ecm.py).
4. **I2 seed-ens K=3** on carrier book (deliverable lever).
5. Lower-priority on-disk: risk-proxy lead-lag (#4, DOWN), DOWN-side PC1 retain (#5).
6. Standard improve cross-product (magdir/TB/xhor/OFI/kNN/meta/twospeed/spec/GRU/Optuna) — each forked individually (cross-pair PAIRS-list fix), vs BEST COMBO + frozen-forward.
