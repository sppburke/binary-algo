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
2. ~~[DOWN · on-disk] magnitude→direction bridge @60s~~ — **DONE / KILLED 2026-06-03** (`min1_magdir.py`,
   `min1_magdir_result.json`). DOWN FLAT ~.50 across EVERY predicted-move-size gate (magq0.0 .500 → magq0.95 .498 in
   2025); UP same ~.50-.51. The magnitude model perfectly selects big moves (magAUC .787) but they carry ZERO
   direction — textbook sign-invariance AT THE OPERATING POINT. The 60s direction edge does NOT hide on large moves.
   **Subsumes any further confidence-only / "avoid-losers" DOWN gate.**
- ~~[both · discovered · prior LOW] BAR-IMAGE 2-D CNN (Sezer CNN-BI / GAF — last non-subsumed bar sub-lever)~~ —
   **DONE / KILLED 2026-06-05** (`barcnn_bars.py`/`barcnn_run.py`/`barcnn_cpcv.py`, `BARCNN_SPEC.md`). 3 image
   encodings (close-histogram / 3ch-OHLC wick+body / GAF-GASF+GADF), MNIST-class 2-D CNN, 60s wc_ret label. ALL null:
   VAL dirAUC ≈.50, test/oos AUC ≈.50, CPCV path_p10 .484–.499 with **0.0 paths clearing 0.541** at every coverage.
   A 4th model class (2-D conv over the rendered chart) confirms bar geometry carries magnitude not 60s sign — even
   the antisymmetric GADF sign-field is null. **Bar/candlestick 2-D-image family RUN + EXHAUSTED on-disk.** Kronos NOT
   built (RankIC/magnitude, no FX/60s/direction numbers; fine-tune deteriorates arXiv:2511.18578).
3. **[UP · improve · prior ~15%] |return|-weighted (POW=0.5) retrain @60s, UP-split** — same `min1_magweight.py`,
   UP head. The UP filter is the incumbent; magweight is the highest-EV improve lever not yet tried at 60s. Falsifier:
   beat UP floor .520 / OOS .613 on the worst held-out year with CI-lo clearing.
4. **[both · improve · prior ~10%] seed-ensemble (K=4) ⊕ GBM on the best 60s book** — variance-reduction on the UP
   filter and any DOWN survivor. (At 5m seed-ens did NOT rescue DOWN — same 2025 wall — so prior tempered.)

# DISCOVERY ROUND 1 (2026-06-03) — DRY (0 survivors / 12 vetted)
Adversarial fan-out (`discover-60s-down-levers` workflow, 12 Explore agents, each grounded in the repo kill
evidence) vetted every UNTESTED sign-aware microstructure DOWN/direction lever in CORPUS_LEVER_INVENTORY.md for
subsumption + sign-invariance at 60s. **ALL 12 subsumed/sign-invariant; 0 survivors.** The order-flow microstructure
family is uniformly a reparameterization of the signed-flow channel already dead at 60s (CKS-OFI VAL .4993,
cross-impact-OFI .5015, per-side-flow ~.50, online-ARF keystone .503-.508) and/or sign-invariant (magdir proof).
Vetted-subsumed (do NOT build without a NEW mechanism): OF-surprise/MRR residual · Lipton buy/sell drift asymmetry ·
SPDE depletion asymmetry · Bouchaud propagator past-sign · DAR(p) sign-AR · metaorder continuation P_m · Cont-de
Larrard queue-imbalance p_up · trade-time/event-clock labeling · Capponi-Cont idiosyncratic residual flow ·
long-OF-history GRU · microprice deviation anchor · predictability-conditioned impact gate. Recurring finding: the
60s DOWN wall is GENUINE EFFICIENCY, not a feature-discovery gap. → DOWN side honestly EXHAUSTED on-disk.

# DISCOVERY ROUND 2 (2026-06-03) — DRY (0 survivors / 8 vetted)
Second fan-out (`discover-60s-round2` workflow, 5 orthogonal angles → 8 candidates → adversarial vet), deliberately
DIFFERENT from round 1: UP-filter certification levers, cross-horizon 15m→60s transfer, non-microstructure DOWN
(fixing-window/calendar/momentum-term-structure), recent literature, novel on-disk combinations. **ALL 8 subsumed/
sign-invariant; 0 survivors.** Key findings: (1) NO UP-improvement lever attacks PATH VARIANCE — the UP CPCV near-miss
(p10 .524) is the STRUCTURAL 2024-regime wash, which seed-ens/calibration/ACI/Optuna cannot fix. (2) 15m→60s transfer
= subsumed by the killed cross-horizon stack (min1_stack, oos .53-.55). (3) WMR fixing-window DOWN = subsumed by
m30_fix (WMR reversal sign-FLIPPED OOS to .429) + sign-invariance. (4) intraday-momentum term-structure = subsumed by
N4 (min2_mim killed @2m). (5) calendar/option-expiry = sign-invariant (magnitude gates). **TWO consecutive DRY rounds
→ discover loop DRY.** The on-disk 60s direction space is saturated on both sides.

# Subsumed / dead at 60s (do NOT re-run without a NEW mechanism)
- USD/cross-pair conditioning DOWN (D3a, killed 2026-06-03) — and its 5m parent. Cross-pair sign-lead family null.
- Side-specialists (subset-training kills ranking). Online-ARF (efficiency keystone). HMM/Kalman/RMT/CCM/OFI/CKS
  (~24 channels, all null — EURUSD_RESULTS.md § 60s rows 1-17). Complexity/Hurst (sign-invariant).

# External-data-gated (NOT on-disk — acquisition prerequisite, per honest-frontier)
- **Option-implied risk-reversal sign** (the single remaining mechanism-grounded DOWN lever) — verified PAYWALLED.
- Intraday DE-US 2y rate differential (Dukascopy); true multi-level LOB depth. All enumerated in IDEAS_LOG H-TODO-*.

**Stop condition for the 1m DOWN side:** levers 1-2 run + shown sub-breakeven on the worst held-out year ⇒ DOWN
is honestly exhausted on-disk at 60s (DEAD), pending external data. UP side: levers 3-4 ⇒ improve-loop dry.
