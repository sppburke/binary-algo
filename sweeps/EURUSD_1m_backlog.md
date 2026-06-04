> **SCOPE: EURUSD · 1m (60s)** (EXECUTABLE backlog — first-to-run queue + incumbents-to-beat). Ledger: sweeps/EURUSD_1m.md. Results: EURUSD_RESULTS.md § 60s. See REPO_MAP.md.

# EURUSD 60s — incumbents to beat
- **UP:** up-only filter on symmetric ensemble (`min1_updown.py`) — OOS .613, floor .520, regime-dependent, NOT 0.65, uncertified-for-deriv.
- **DOWN:** **dead** (.522/.522/.516). No certified DOWN edge. Mechanism = dip-buy works, rally-sell doesn't (EUR-up regime artifact; asymmetry vanishes 2024).
- Breakeven 0.541. **Honest prior: 60s direction is near-efficient (~0.50-0.51 AUC); the online-ARF keystone proves it. The forecastable quantity at 60s is MAGNITUDE, not sign.**

# FIRST-TO-RUN queue (both sides, mechanism-first; attack the confirmed cause)

Model-of-the-edge after D3a (2026-06-03): the DOWN side is unsolved because **cross-pair USD conditioning carries
no next-60s sign** — at 60s the USD-driver move is already priced within the minute and the most-USD-strong tail
even mean-reverts. So the next DOWN levers must NOT be another USD/cross-pair gate (subsumed). Attack via (1) loss
re-weighting that the 5m DOWN responded to, and (2) the magnitude→direction bridge (the one real 60s signal).

1. ~~[DOWN · on-disk] |return|-weighted (POW=0.5) retrain @60s~~ — **DONE / KILLED 2026-06-03** (`min1_magweight.py`,
   `min1_magweight_result.json`). DOWN binding-2025 cov0.05 **.507** (CI-lo .495) < 0.541 & < incumbent .522; UP worse
   than filter. **best_iter=8/4000** — the magweighted direction model has no learnable signal to weight toward. The
   5m razor-thin DOWN edge (cov0.05 p10 .5441) does NOT transfer to the more-efficient 60s. Loss-reweighting exhausted.
2. **[DOWN · on-disk · prior ~8%] magnitude→direction bridge @60s** — bet DOWN only on bars the magnitude model
   flags as large-move AND the direction model leans down, restricted to NY. Mechanism: |ret| is the one certified
   60s signal (magAUC .787); IF direction has any edge it lives on large moves (informational, not coverage —
   `_redteam_magdir60` showed the gap to .65 is informational). Falsifier: KILL unless large-move DOWN binding-2025
   CI-lo ≥ 0.541. Prior LOW (sign-invariance theorem predicts magnitude gate adds size not sign) — but it directly
   tests the one place sign could hide. Subsumes any further "avoid-losers" gate.
3. **[UP · improve · prior ~15%] |return|-weighted (POW=0.5) retrain @60s, UP-split** — same `min1_magweight.py`,
   UP head. The UP filter is the incumbent; magweight is the highest-EV improve lever not yet tried at 60s. Falsifier:
   beat UP floor .520 / OOS .613 on the worst held-out year with CI-lo clearing.
4. **[both · improve · prior ~10%] seed-ensemble (K=4) ⊕ GBM on the best 60s book** — variance-reduction on the UP
   filter and any DOWN survivor. (At 5m seed-ens did NOT rescue DOWN — same 2025 wall — so prior tempered.)

# Subsumed / dead at 60s (do NOT re-run without a NEW mechanism)
- USD/cross-pair conditioning DOWN (D3a, killed 2026-06-03) — and its 5m parent. Cross-pair sign-lead family null.
- Side-specialists (subset-training kills ranking). Online-ARF (efficiency keystone). HMM/Kalman/RMT/CCM/OFI/CKS
  (~24 channels, all null — EURUSD_RESULTS.md § 60s rows 1-17). Complexity/Hurst (sign-invariant).

# External-data-gated (NOT on-disk — acquisition prerequisite, per honest-frontier)
- **Option-implied risk-reversal sign** (the single remaining mechanism-grounded DOWN lever) — verified PAYWALLED.
- Intraday DE-US 2y rate differential (Dukascopy); true multi-level LOB depth. All enumerated in IDEAS_LOG H-TODO-*.

**Stop condition for the 1m DOWN side:** levers 1-2 run + shown sub-breakeven on the worst held-out year ⇒ DOWN
is honestly exhausted on-disk at 60s (DEAD), pending external data. UP side: levers 3-4 ⇒ improve-loop dry.
