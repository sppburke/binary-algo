# EURUSD · 5m — EXECUTABLE BACKLOG (key-specific)
SCOPE: EURUSD · 5m. Key-specific executable backlog — incumbents, scripts, numbers, discovery rounds.
Moved out of IDEAS_LOG.md on 2026-06-02 during the generic↔specific split (see REPO_MAP.md).
Generic methods/ideas: METHODS_CATALOG.md / SWEEP_MATRIX.md / IDEAS_LOG.md. Sweep ledger (status): sweeps/EURUSD_5m.md. Results of record: EURUSD_RESULTS.md.

---

## SESSION-2 (2026-05-31b) — 5-MIN re-push: CROSS-PAIR + ORDER-FLOW + META-LABELER (NEW orthogonal ideas)
Goal re-set to 5-min >65% OOS. New ideas tried (the genuinely-untried-at-5m set):
- **Cross-pair USD-common-factor / lead-lag** (m5_xpair.py): EURUSD = EUR-strength − USD-strength; build USD basket from the
  other 6 majors (sign-aligned: +ret for USD-base JPY/CHF/CAD, −ret for USD-quote GBP/AUD/NZD), per-pair lead-lag residual
  in EURUSD-equivalent terms, catch-up residual (EURUSD owes the basket move), EUR-idiosyncratic residual, cross-pair
  dispersion & agreement. **The relative-value RESIDUAL reversion is the FIRST orthogonal signal SIGN-STABLE across 2024 &
  2026** (spearman +0.015/+0.024) — momentum-continuation agreement is dead (sign-blind). Lifts honest book 0.566→0.586.
- **Order-flow as features + reliability** (features_of/: Kyle λ, OF persist/accel/uptick, normalized imbalance): 0 AUC lift
  (raw OF into a GBM is null), signed-OF flow-FOLLOWING loses (stably negative vs fwd ret) — but helps the book → 0.594.
- **Learned META-LABELER on ORTHOGONAL axes** (m5_meta.py): 2nd lgb predicts P(primary correct) from agreement/dispersion/OF/
  confidence (NOT the 239), abstain unless meta≥thr, threshold by WORST-VAL-HALF stability (kills the corr(VAL,OOS)=−0.54
  trap). Best honest 5-min book **combined ~0.61** (t24 0.64/t25 0.58/oos 0.62). The right way to "not select losers".
**Falsifier (P0 gate, m5_xp_analyze.py):** NO conditioning region (agreement/OF/disp/vol/session) lifts the 2025 window above
~0.557; ORACLE (hindsight) max-floor = 0.598-0.601. A meta-classifier ≤ its features' conditional accuracy → can't reach 0.65.
**5-MIN CONCLUSION:** honest frontier improved 0.566→**~0.61** (genuine, profitable vs 0.541), but **≥0.65 OOS-stable is NOT
reachable** — the 2025 regime is near-efficient for 5-min direction. Buildable-from-repo levers exhausted; remaining levers =
signed macro/news calendar (external) or a seconds/tick broker (real ≥0.65 edge). Deliverable: m5_xpair_production.py.

---
## Discovery round 1 for the EURUSD **5m** sweep — cross-disciplinary agent, 2026-06-01. Context: unlike 2m, 5m HAS a CERTIFIED cross-pair UP edge (m5xp side-split, binding 2025 0.577, CPCV 28/28 paths clear), so the ceiling is NOT fully closed — these target whether a SIGNED-causality channel captures MORE or a DOWN edge. Each vetted vs the killed list; signed mechanism surviving sign-invariance.
- **N10 Liang-Kleeman normalized information flow (SIGNED), leg→EURUSD @300s.** The IFR T₂→₁ is signed by construction (sign = direction the driver pushes the recipient's tendency, from cov(X₂, dX₁/dt)). The one explicitly-SIGNED causality measure left untried — CCM (unsigned coupling) and Schreiber/Massey directed-info (unsigned bits) were its cousins, both KILLED. Source: Liang PRE 92:022126 (2015) / arXiv:1501.03548; multivariate Entropy 23(6):679 (2021); finance PLOS One 2024 PMC11045085. On-disk (7 legs + OF_*). Prior ~9%. Fast-KILL: rolling sign(IFR(leg→EURUSD)) over W∈{60,120,300}s predicts next-300s sign at dir-AUC ≤0.51 worst-VAL-half for all legs/windows → dead.
- **N11 Bacry-Muzy 4-kernel Hawkes up/down cross-excitation imbalance @300s.** λ_up(t)−λ_down(t): difference of two signed conditional intensities (sign-covariant), with a MEAN-REVERTING impact kernel whose post-jump relaxation is a 300s-scale object. Distinct from killed N7 (scalar self-excitation ratio, no up↔down cross-kernel, decayed <120s). Source: Bacry-Muzy arXiv:1301.1135 / Quant.Finance 14(7) 2014. On-disk (EURUSD mid up/down-jump times + OF_uptick/sum). Prior ~8%. Fast-KILL: EW (uptick−downtick) imbalance over τ∈{30,60,120,300}s regresses next-300s sign at dir-AUC ≤0.51 worst-half all τ → cross-excitation buys nothing over N7 → dead.
- N12 Directed-HVG irreversibility SIGN-DECOMPOSED (KLD split on peaks vs troughs; sign-covariant, unlike the killed scalar I_W 0.503). Lacasa arXiv:1108.1691 / EPJB 85:217 (2012). On-disk. Prior ~6%. Completeness.
- N13 Time-reversal asymmetry of the SIGNED structure function E[r_t·|r_{t-τ}|²]−E[|r_t|²·r_{t-τ}] (odd functional → sign-carrying; continuation-vs-leverage regime gate). Lynch-Zumbach Quant.Finance 3:320 (2003); asym-MF-DFA Cao-Cao-Han Physica A 392 (2013). On-disk. Prior ~5%. Pre-kill if corr>0.7 with killed N8 RS-skew.
- N14 Liang MULTIVARIATE IFR dollar-source gate (7×7 IFR matrix → who is net info source; if USD is source, sign(dollar-basket Δ) propagates to EURUSD). Entropy 23(6):679 (2021). On-disk. Prior ~5%. Conditions the (unconditionally-killed N2/N3/5/6/9) cross-leg sign on the IFR source-state — new conditioning. Completeness.
Recommended order: N10, N11 only as live shots; N12-14 as completeness. Meta-caveat: online-ARF keystone ~0.51 every year says the SINGLE-PAIR 300s ceiling is a data limit — but the cross-pair channel is demonstrably live (m5xp UP), so N10/N14 (cross-leg signed flow) are the mechanistically-aligned bets.
RESULT (2026-06-01): m5_legsign.py family probe KILLED N10/N11/N14/N15/N16/N17 STANDALONE (best ~0.511; cross-leg signed-IFR/network-momentum, own signed-flow/propagator-residual/Hawkes-imbalance, own sign-autocorr all ~0.51). The certified m5xp UP is the nonlinear GBM COMBINATION of these — none predictive alone. N12/N13 (irreversibility/odd-moment) untested (low prior, I_W scalar already 0.503 at 2m).

## Discovery round 2 for the EURUSD **5m** sweep — DOWN-side agent, 2026-06-01. Targets the OPEN GAP (DOWN unsolved at every horizon). Mechanistic Q1 finding: DOWN is structurally near-efficient at 5m because down-moves are JUMP/INFORMED-dominated (downside energy → volatility/magnitude per sign-invariance, not a predictable drift); the UP/DOWN asymmetry is a regime-bound dip-buy DRIFT, not a law (2024 was symmetric UP≈DOWN). The only down-sign handles are FADES of over-shoots (which predict UP) + USD-driven DOWN regimes.
- **D1 Exogenous-jump over-shoot fade (post-macro down-spike reversal).** After a macro release down-spike that over-shoots, next-5m biased UP (exogenous jumps mean-revert). Marcaccioli-Bouchaud-Benzaquen arXiv:1805.05179 / DOI 10.1088/1742-5468/ac498c. Prior ~22% — BUT predicts UP (a fade), not DOWN; failure mode: FX prices surprise <60s so the over-shoot is gone by bar close. On-disk (macro_calendar + RV).
- **D2 Bad-semivariance exhaustion fade.** Downside-RV dominated + total-RV rolling over → down-bar reverses UP. Patton-Sheppard DOI 10.1162/rest_a_00503; BNS semivariance DOI 10.2139/ssrn.1262194. Prior ~18% — also a UP-fade, not DOWN. Risk: bad-RV predicts σ not sign (the theorem).
- **D3 USD-strength-conditioned DOWN (the one genuine DOWN-continuation idea).** EURUSD DOWN is the bet only when the DOLLAR is the driver (broad USD bid pulls EURUSD down coherently); rally-selling died in 2025 because it was an EUR-up regime with no USD tailwind. Condition DOWN on basket-USD-strength + coherence (residual not stretched) + NY. Cross-pair USD-residual is killed for SYMMETRIC direction; this is a narrower DOWN-only USD-gated re-slice. Prior ~15%. On-disk (m5_xpair basket). **TESTING (m5_downcond.py).**
- D4 Overbought-extreme contrarian DOWN (RSI/BB OB short-side, session-gated). Daniel-Moskowitz momentum-crash DOI 10.1016/j.jfineco.2015.12.002 (horizon-mismatched motivation). Prior ~10% — in the killed Sofien price-action family; completeness.
- D5 Microprice/imbalance down-pressure persistence (endogenous-accelerating sub-slice). Marcaccioli et al. Prior ~8% — signed-flow already killed; completeness.
Honest verdict: down side structurally efficient; tradeable "down prediction" is almost certainly "fade an over-extended exogenous down-spike back UP," not "sell a rally." Cheapest first move: D3 conditional up/down-rate split on the existing frozen m5xp DOWN trades (no new training).

---

# ============================================================================
# SYNTHESIZED EXECUTABLE BACKLOG — RAISE the 5m EURUSD edge (2026-06-01)
# ============================================================================
# Final-synthesizer pass. The 5m sweep (sweeps/EURUSD_5m.md) is COMPLETE: the input
# carriers are exhausted (~30 channels null/subsumed, online-ARF keystone ~0.51, 76-cand
# signed-lag family ~0.51, DL info-bound, sign-invariance theorem). So this backlog does
# NOT re-mine inputs — it RAISES the certified edge via {objective · ensembling ·
# calibration · gate · sizing · pooling} levers + the external-data acquisitions that are
# the only documented dominant-EV move.
#
# FULL LEVER SET: the complete 2050-lever extraction from all 370 papers is indexed in
#   /home/sean/git/academic-papers/_CORPUS_INDEX.md (per-paper, grep by lever_type/topic/`new_untried`)
#   and _extracted_levers.json (every field). This backlog is the RANKED/EXECUTABLE subset.
#   UPDATE 2026-06-01: A1 KILLED ACI under nested-refit CPCV (meta-gate < primary-confidence cover)
#   -> the adaptive-meta-gate family (FACI/SF-OGD etc.) is DOMINATED, deprioritize. B1 RESCUED DOWN
#   (POW=0.5 magweight, marginal — new DOWN leader). C3 (GBM seed-ensemble) running.
#
# INCUMBENTS TO BEAT (the bar; verified from result-JSONs this session):
#   • UP fixed gate     EURUSD.m5xp.v1 up-preds  = .605/.577/.615  (binding 2025 .577; CPCV frozen p10 .576; full-refit cov0.05 p10 .553, 96% folds)
#   • UP higher-conv    A8a m5_upfilter thr0.598 = .666/.616/.581  (binding 2025 .616; CPCV a8a p10 .608, 28/28)
#   • UP deployed       EURUSD.m5xp_aci.v1 (ACI) = binding 2025 .584 @n764 vs fixed .579 @n618, +36% trades  [CPCV cert IN FLIGHT: m5_aci_cpcv.py running]
#   • DOWN              m5xp down-preds          = .609/.533/.592  (binding 2025 .533 FAILS breakeven → the open gap)
#   • Breakeven 0.541 (R≈0.85). Binding test = worst-year (2025) moved-acc CI95-lo ≥ 0.541 AND CPCV path-clear@0.541 ≥ the incumbent's.
# DISCIPLINE (every row): deriv-faithful wc_ret ties-LOSE · nonoverlap_chrono 300s · per-year 2024/25/26 CI95 ·
#   select on WORST-VAL-HALF not VAL-max (corr(VAL,OOS)=−0.54) · moved-bars up-rate∈[.47,.53] tripwire ·
#   label-shuffle placebo must drop to ~0.50 · CERTIFY survivors on FULL-REFIT CPCV at the operating gate
#   (m5_cpcv_refit.py / cpcv_certify.py), metric = fraction of purged paths clearing 0.541, NOT AUC.

## TOP 30 EXPERIMENTS — ranked by (EV-to-beat-incumbent × cheapness), grouped by lever_type

### Lever A — CALIBRATION & GATE (cheapest, highest-ROI; our whole edge is a confidence-gated selective bet)
**A1. [FIRST] Finish the ACI-CPCV certification.** RUN: `m5_aci_cpcv.py` (already in flight, M5_STRIDE smoke first → full). Nested per-fold refit of the cross-pair primary + ONLINE ACI θ replayed inside each purged fold (no look-ahead). FALSIFIER: KILL/keep-as-fixed unless ACI path-clear@0.541 ≥ the fixed-gate refit (cov0.05 p10 .553, 96%) AND binding-fold p10 ≥ .553. INCUMBENT: m5xp_aci .584 (uncertified→certified). NOVELTY: certifies an already-registered improvement. cost≈running.
**A2. [FIRST] Temperature / Venn-Abers calibration of the meta probability, THEN re-pick the gate.** BUILD: new `m5_calib.py` — fit 1-param temperature (and Venn-Abers as alt) on VAL meta-scores, recompute the meta≥thr gate on calibrated p, re-run side-split + confidence curve. FALSIFIER: KILL unless calibrated gate lifts binding-2025 UP win OR coverage at equal win (≥.577 @ ≥n1379) with placebo ~0.50 and the up-rate tripwire clean. INCUMBENT: m5xp UP .577 / coverage. NOVELTY: NEW (gate was never calibrated; DL review lever #4). cost≈minutes.
**A3. ACI on the A8a higher-conviction primary-confidence gate (not just meta).** ADAPT: `m5_conformal.py` driving the A8a `m5_upfilter.py` primary thr instead of meta. FALSIFIER: KILL unless it holds 2025 ≥ .616 at ≥ the fixed-A8a coverage. INCUMBENT: A8a .616. NOVELTY: variant (ACI×different gate axis). cost≈minutes.
**A4. ACI target-w* sweep + dual-gate (meta AND primary-conf both ACI-controlled).** ADAPT: `m5_conformal.py`, w*∈{.55,.56,.57,.58}, two coupled θ's. FALSIFIER: KILL unless a w* beats single-ACI 2025 .584 on win×coverage with stable cross-year spread. INCUMBENT: ACI .584. NOVELTY: variant. cost≈minutes.

### Lever B — OBJECTIVE / |return|-WEIGHTING (the magnitude→direction bridge; our one strong edge feeds sign)
**B1. [FIRST] Certify the gentler-POW magnitude-weighted refit on CPCV.** ⚠ LIVE RESULT: `m5_magweight.py 0.5` already gives UP 2025 **.5779** (≈incumbent, no 2026 collapse) AND rebalances DOWN to .605/.559 in 2024/25 — POW=1.0 had collapsed 2026 UP. RUN: pipe the POW=0.5 retrain through `m5_cpcv_refit.py`. FALSIFIER: KILL unless POW=0.5 UP CPCV path-clear ≥ incumbent .553/.96 AND placebo ~0.50; KILL the DOWN-rebalance claim unless its 2025 CPCV p10 ≥ .541. INCUMBENT: UP .577 / DOWN-rescue. NOVELTY: variant of EXP-1 (POW grid is new — 1.0 was tried, 0.5 is the sweet spot). cost≈minutes.
**B2. POW micro-grid {0.25,0.5,0.75} × temperature-calibrated gate.** ADAPT: `m5_magweight.py` arg + A2 calibration stacked. FALSIFIER: KILL unless some POW beats B1 on binding-2025 win at equal coverage. INCUMBENT: B1. NOVELTY: variant. cost≈minutes.
**B3. GMADL operating-point loss (LGB custom obj: up-weight big-|ret|, down-weight ties-as-loss).** BUILD: `m5_gmadl.py` from `m5_magweight.py` skeleton, swap sample-weights for a Deriv-economics GMADL gradient (ties charged as losses). FALSIFIER: KILL unless cost-after-spread UP-gate ≥ A8a .665/.611/.581 AND CPCV path-clear > .536 (DL-review EXP-1 bar). INCUMBENT: A8a. NOVELTY: NEW (loss never matched to Deriv payout). cost≈30min.

### Lever C — ENSEMBLING / STACKING (DL as decorrelated member; floor/tail-stabilizer, not edge-finder)
**C1. [FIRST] Learned stack-weight instead of 50/50 MLP⊕GBM (earned by the corr=0.694 decorrelation).** ⚠ FACT: `m5_deep_ens_result.json` shows BLEND 2025 .5776 vs GBM .5765 (50/50 ≈ GBM). ADAPT: `m5_deep_ens.py` — fit the blend weight on WORST-VAL-HALF (not 50/50), temperature-calibrate the blend. FALSIFIER: KILL unless learned-weight BLEND 2025 ≥ .582 (a real >noise lift) AND CPCV path-clear > GBM-alone. INCUMBENT: m5xp .577. NOVELTY: variant (weight is new; blend exists). cost≈reuse (OOF cached) ~20min.
**C2. M=8 seed-ensemble (vs M=5) + average-prob, stacked as ONE OOF column into the GBM meta (not blended).** ADAPT: `m5_deep_ens.py` M=8, feed purged-OOF MLP-prob as a meta feature. FALSIFIER (DL-review EXP-2 bar): KILL unless OOF-corr<0.9 AND stacked UP-gate ≥ .605/.577/.615 AND CPCV path-clear>.536. INCUMBENT: m5xp UP. NOVELTY: variant. cost≈hours CPU.
**C3. Seed-ensemble the GBM primaries themselves (n_jobs multithread → no estimator seed → noisy tail; average K refits).** BUILD: `m5_gbm_seedens.py` — retrain the m5xp primary K=8× (different bagging seeds), average probs, re-gate. FALSIFIER: KILL unless averaged-primary tail (cov0.05 CPCV p10) > single-model .553 (the p10 .534 vs mean .542 gap IS tail variance per DL-review #1). INCUMBENT: refit p10 .553. NOVELTY: NEW (GBM-side seed-ensembling never done; registry notes the no-seed multithread). cost≈30min.

### Lever D — POOLED CROSS-PAIR (the one untested STRUCTURAL lever; raises floor not ceiling)
**D1. Pooled 7-pair weight-shared net: pairs as TRAINING ROWS + per-pair instance-norm + pair embedding.** BUILD: `m5_pooled.py` (Sirignano-Cont arXiv:1803.06917) — we cross-pair at feature level but never pooled pairs as rows. Train one net on all 7 majors' moved bars, predict EURUSD, stack OOF into meta. FALSIFIER (DL-review EXP-3): KILL unless it beats C2 stacked (contingent on C2 showing life). INCUMBENT: C2 / m5xp. NOVELTY: NEW structural. cost≈hours CPU (contingent — only if C1/C2 show life).
**D2. Pooled cross-pair GBM (cheap proxy of D1): same-feature GBM trained on 7-pair row-union, EURUSD-predicted.** BUILD: adapt `m5_xpair.py` build to stack 7 pairs' rows with a pair-id one-hot. FALSIFIER: KILL unless pooled-GBM UP-gate ≥ .577 binding with placebo ~.50. INCUMBENT: m5xp UP. NOVELTY: NEW (GBM pooling untried). cost≈30min. (Run BEFORE D1 — cheaper test of the same structural hypothesis.)

### Lever E — SIZING / DEPLOYMENT EV (raises realized $-edge without touching AUC; already partly built)
**E1. Confidence-proportional fractional-Kelly stake curve, sized on the .553 robust floor.** ADAPT: `m5_kelly.py` + `m5_equity.py` (exist) — bet size ∝ calibrated conf, capped ⅛-Kelly, sized on floor not optimistic. FALSIFIER: report-only (not an accuracy claim) — KILL the sizing lever unless floor-sized growth > flat-stake at equal max-DD. INCUMBENT: deployment spec. NOVELTY: variant (equity path exists; conf-proportional sizing is the new knob). cost≈minutes.
**E2. SLOW structural kill-switch (trailing few-hundred-trade win<0.55 over extended span) vs the naive one already shown to hurt.** ADAPT: `m5_equity.py` kill-switch block. FALSIFIER: KILL unless slow-switch improves regime-death-stress DD without costing >2% growth in the good regime (naive cost 25.95×→19.37×). INCUMBENT: no-switch. NOVELTY: variant. cost≈minutes.

### Lever F — DOWN-SIDE RESCUE (the open gap; mechanism-driven, low-but-nonzero prior)
**F1. [FIRST, cheap] |return|-weighted DOWN re-slice from B1.** The POW=0.5 magweight ALREADY lifts DOWN to 2025 .559 (`m5_magweight_result.json`) — split its down-preds, CPCV-certify. FALSIFIER: KILL unless DOWN 2025 CPCV p10 ≥ .541 (incumbent down p10 .5421 barely clears; must beat forward .538). INCUMBENT: DOWN .533/.538. NOVELTY: NEW angle on DOWN (magnitude-weighting rebalances toward two-sided). cost≈reuse B1 output.
**F2. Exogenous-jump over-shoot FADE as an UP signal in a DOWN regime (D1/D2 reframed: predict UP after a macro/bad-RV down-spike).** BUILD: `m5_overshoot_fade.py` — flag macro-release or bad-semivariance down-spikes (macro_calendar + RV on-disk), bet UP next-5m. FALSIFIER: KILL unless post-down-spike UP-rate ≥ .56 with 2025 CI-lo ≥ .541 (Marcaccioli arXiv:1805.05179). Risk: FX prices surprise <60s. INCUMBENT: DOWN-gap (this is a DOWN-regime UP bet, the honest "down prediction"). NOVELTY: NEW (over-shoot fade never built; D1/D2 were subsumed-on-theory, not run). cost≈30min.
**F3. DOWN under temperature-calibrated + ACI gate (apply A2/A1 to the down-side).** ADAPT: `m5_conformal.py` / A2 on down-preds. FALSIFIER: KILL unless calibrated/ACI DOWN holds 2025 CI-lo ≥ .541 at usable coverage (A8b got .558 but CI-lo .516). INCUMBENT: A8b .558. NOVELTY: variant. cost≈minutes.

### Lever G — REGIME/CONDITIONING re-slices of the certified channel (cheap, completeness-plus)
**G1. Volatility-regime split of the UP edge (does the dip-buy edge concentrate in low/high RV?).** ADAPT: `m5_confcurve.py` add an rv-bucket axis. FALSIFIER: KILL unless an RV bucket lifts binding-2025 UP ≥ .60 at usable coverage (oracle floor was .598). INCUMBENT: m5xp .577. NOVELTY: variant. cost≈minutes.
**G2. EUR-up vs USD-up macro-regime conditioning (the edge is dip-buy in EUR-up regimes; gate on basket-USD trend).** ADAPT: `m5_downcond.py` basket logic, applied to UP. FALSIFIER: KILL unless regime-gated UP beats ungated 2025 .577. INCUMBENT: m5xp .577. NOVELTY: variant. cost≈minutes.
**G3. Time-of-day / NY-subsession micro-gate (which NY hours carry the .577?).** ADAPT: `m5_confcurve.py` hour axis. FALSIFIER: KILL unless an hour-slice lifts 2025 at ≥ half the coverage. INCUMBENT: m5xp .577. NOVELTY: variant. cost≈minutes.
**G4. ACI on a two-asset confirm (only trade UP when the calibrated meta AND a momentum/RV co-signal agree).** ADAPT: `m5_conformal.py` + agreement filter. FALSIFIER: KILL unless agreement-gated 2025 > .584. INCUMBENT: ACI .584. NOVELTY: variant. cost≈minutes.

### Lever H — EXTERNAL-DATA-CONTINGENT (highest ceiling-EV but blocked on acquisition; see acquisition TODOs)
**H1. Intraday DE–US 2y rate-differential as a directional feature into m5xp.** Dominant-EV per DL review (the absent carrier). FALSIFIER: KILL unless rate-diff Δ adds ≥ .01 binding-2025 UP AUC with placebo ~.50. INCUMBENT: m5xp .577. NOVELTY: NEW input class. BLOCKED on H-TODO-1. cost≈acquisition+30min.
**H2. Daily implied-vol / 25Δ risk-reversal (FX options skew = signed positioning) as a daily-context gate.** FALSIFIER: KILL unless RR-sign conditions UP/DOWN asymmetry (2025 binding lift ≥ .01). INCUMBENT: m5xp / DOWN-gap. NOVELTY: NEW (risk-reversal is the textbook signed-positioning carrier — best DOWN-rescue shot). BLOCKED on H-TODO-2. cost≈acquisition+30min.
**H3. EURGBP ticks → cleaner EUR-leg isolation (algebraic USD-cancel with a 3rd cross).** FALSIFIER: KILL unless EURGBP-residual adds stable signed lift (triangular @2m was .499 — but on 1-min bars, not ticks). INCUMBENT: m5xp. NOVELTY: variant of triangular with finer data. BLOCKED on H-TODO-3. cost≈acquisition+30min.

### Lever I — LAST-RESORT / LOW-PRIOR (run only after A–G; documented for completeness, not padding)
**I1. Calibrated AdamW-MLP with M=8 seeds as a PURE tail-stabilizer of the existing gate (not a new signal).** Per DL-review #1 (the p10 .534 vs mean .542 gap is tail variance). FALSIFIER: KILL unless it raises CPCV cov0.05 p10 above .553. INCUMBENT: refit p10 .553. NOVELTY: variant. cost≈hours.
**I2. Venn-Abers multiprobability gate (interval-valued conformal) for the abstain decision.** FALSIFIER: KILL unless it beats scalar ACI on cross-regime spread. INCUMBENT: ACI .584. NOVELTY: variant. cost≈30min.
**I3. Cross-horizon soft-stack 15m→5m re-tuned with the calibrated 5m meta (m5stack had tripwire issues).** ADAPT: `m5_stack2.py` + A2 calibration. FALSIFIER: KILL unless it beats m5xp 2026 .606 without breaching the up-rate tripwire. INCUMBENT: m5xp 0.606 combined. NOVELTY: variant. cost≈30min.
**I4. Macro-surprise SIGN gate re-test under ACI (news null for AUC, but ACI may salvage event-window coverage).** ADAPT: `m5_news.py` + ACI. FALSIFIER: KILL unless news-window ACI UP > overall ACI .584. INCUMBENT: ACI .584. NOVELTY: variant (news was null, ACI-on-news untried). cost≈minutes.
**I5. Stacked super-learner over {GBM, MLP-ens, pooled, magweight} OOF with calibrated meta.** Only if ≥2 of B/C/D show life. FALSIFIER: KILL unless super-learner 2025 > best single member. INCUMBENT: best of B/C/D. NOVELTY: variant. cost≈30min (contingent).

## FIRST-TO-RUN QUEUE (genuinely NEW-or-live + high-priority + cheap — the immediate batch)
Ordered. All reuse existing scripts/outputs, all minutes-not-hours, each has a concrete kill:
1. **A1** — finish `m5_aci_cpcv.py` (IN FLIGHT): certify the already-deployed ACI .584. Nothing else competes until this lands.
2. **B1** — CPCV-certify the POW=0.5 magweight (LIVE: already shows UP .578 + DOWN-rebalance .559/.605 with no 2026 collapse — the single most promising un-certified result on disk).
3. **A2** — temperature/Venn-Abers calibrate the meta + re-pick gate (`m5_calib.py`; whole edge is a gated bet, never calibrated — cheapest structural fix).
4. **C1** — learned (worst-VAL-half) stack-weight on the cached MLP⊕GBM OOF (decorrelation .694 is earned; 50/50 was the only thing tried).
5. **F1** — split the POW=0.5 magweight DOWN-preds + CPCV (free piggyback on B1; the cheapest DOWN-rescue shot).
6. **C3** — GBM-side seed-ensemble (K=8 primary refits, average) to shrink the CPCV tail variance (p10 .534→? toward mean .542); NEW, never done.
7. **D2** — pooled-row cross-pair GBM (cheap proxy for the one untested STRUCTURAL lever before committing CPU to D1).
These seven are the queue; everything else is contingent on their outcomes or blocked on external data.

## EXTERNAL-DATA ACQUISITION TODOs (separate track — dominant ceiling-EV per the DL review; the binding constraint is INFORMATION not model)
- **H-TODO-1: Intraday DE–US 2y (Schatz/2Y-UST) yield differential.** Source: Dukascopy historical (per MEMORY note) or a bond-futures feed. Resample to 1-min, align to the EURUSD feature grid 2012-2026, add Δ(rate-diff) as a signed feature. The single most-cited absent carrier. → feeds H1.
- **H-TODO-2: Daily FX implied-vol + 25Δ risk-reversal (1w/1m tenors) for EURUSD.** Source: a vol-surface vendor (or scrape). Risk-reversal SIGN = market's directional skew = the textbook signed-positioning carrier → the most mechanistically-aligned DOWN-rescue input. → feeds H2.
- **H-TODO-3: EURGBP (and a 3rd USD cross) tick/1-s data** to algebraically isolate the EUR leg cleaner than the 6-pair basket. → feeds H3.
- **H-TODO-4 (lower prior): CFTC/positioning + cross-asset (DXY futures, Bund) intraday.** Completeness; ES/NQ already sign-flip-null.
NOTE: macro-surprise calendar is ALREADY on-disk (macro_calendar.parquet) and ALREADY tested null for direction (sign-invariant); it is NOT in this acquisition list except as the ACI-salvage I4.

## DOWN-SIDE RESCUE — explicit call-out (the open gap at every horizon)
The DOWN side is structurally hard (down-moves jump/informed-dominated → energy to MAGNITUDE not sign; sign-invariance theorem). Three ideas have a PLAUSIBLE mechanism and are ranked here:
1. **F1 |return|-weighted DOWN re-slice (cheapest, live):** the POW=0.5 magweight already pulls DOWN to .605/.559 in 2024/25 — the only on-disk result that touches the .533 wall. Must clear CPCV 2025 p10 ≥ .541. This is the best near-term DOWN shot.
2. **H2 risk-reversal sign (external, highest ceiling):** FX options skew is the one genuinely-SIGNED positioning carrier we don't have; if anything rescues DOWN it is this. Blocked on acquisition.
3. **F2 over-shoot fade reframed as UP-in-a-down-regime:** the honest "down prediction" is fading an over-extended exogenous down-spike back UP (Marcaccioli). Never actually built (subsumed on theory). Cheap to test.
Everything else DOWN-side (D3 USD-strength .526, A8c specialist .509 anti-transfer, A8b CI-lo .516, sidecurve) is already KILLED — do not re-run; F1/H2/F2 are the only live DOWN levers.

## HONEST META-NOTE
The 5m sweep already CERTIFIED a deployable UP edge (~.55–.57) and a deployed ACI improvement (.584). The realistic prize from this backlog is INCREMENTAL: a few points of binding-2025 win, a tighter CPCV tail, a rescued DOWN side, or better realized EV via calibration+sizing — NOT a jump to >0.65 (proven unreachable on these inputs). The ONE lever with a high ceiling is external data (H-track). Run the FIRST-TO-RUN seven, certify survivors on full-refit CPCV at the gate, freeze any winner as a new book (`m5_register_*.py` pattern), and update EURUSD_RESULTS.md + the leaderboard. Disproved-by-experiment beats untried — but here, prefer the cheap certifications of LIVE near-misses (B1/A1/C1) over re-mining dead inputs.


# ============================================================================
# SYNTHESIZED EXECUTABLE BACKLOG v2 — RAISE the 5m EURUSD edge (2026-06-01, FINAL SYNTHESIZER)
# ============================================================================
# This SUPERSEDES the v1 backlog above (which covered only the calibration/ensemble/sizing
# levers). v2 is the final pass over the academic-paper-grounded per-lever shortlists
# (feature/input · validation/certification · gate/selection · loss/labeling · ensembling ·
# regime/nonstationarity — ~150 candidate ideas). It is deduped against (a) the v1 rows
# A1–I5, (b) the killed list in sweeps/EURUSD_5m.md (~30 channels), and (c) the methods
# catalog. The v1 FIRST-TO-RUN seven (A1/B1/A2/C1/F1/C3/D2) remain the immediate near-miss
# certifications; v2 adds the NEW academic levers ranked by (EV-to-beat-incumbent × cheapness).
#
# INCUMBENTS TO BEAT (re-verified this session from result-JSONs):
#   • UP fixed gate   m5xp.v1 up-preds      = .605/.577/.615 (binding 2025 .577 @n1379; full-refit CPCV cov0.05 p10 .553/96%)
#   • UP deployed     m5xp_aci.v1 (ACI .57) = binding 2025 .5838 @n764 vs fixed .5793 @n618 (m5_conformal_result.json)
#   • UP higher-conv  A8a m5_upfilter thr.598= .665/.611/.581 (CPCV a8a p10 .608, 28/28)
#   • DOWN            m5xp down-preds        = .609/.533/.592 (binding 2025 .533 FAILS breakeven → open gap)
#   • magweight POW0.5 (LIVE, uncertified)   = UP .605/.578/.539, DOWN .605/.559/.549 (m5_magweight_result.json)
#   • Breakeven 0.541. Binding = worst-year (2025) moved-acc CI95-lo ≥ .541 AND full-refit-CPCV path-clear @ the gate.
# DISCIPLINE (every row, NON-NEGOTIABLE): deriv-faithful wc_ret ties-LOSE · nonoverlap_chrono 300s ·
#   per-year 2024/25/26 CI95 · select on WORST-VAL-HALF not VAL-max (corr(VAL,OOS)=−0.54) ·
#   moved-bars up-rate∈[.47,.53] tripwire · sign-invariance shuffle (within-|ret|-bin sign-shuffle: if the
#   edge survives it's MAGNITUDE not sign → KILL as a direction claim) · CERTIFY survivors on FULL-REFIT
#   CPCV at the operating gate (m5_cpcv_refit.py / cpcv_certify.py), metric = frac purged paths clearing 0.541.

## TOP 30 EXPERIMENTS — ranked by (EV-to-beat-incumbent × cheapness), grouped by lever_type
## (★ = genuinely NEW-untried; ⚡ = cheap, reuses on-disk outputs/scripts, minutes-not-hours)

### LEVER 1 — VALIDATION / CERTIFICATION (cheapest, decision-relevant; post-hoc on numbers already logged)
**V1. ★⚡ Multiple-testing haircut over the WHOLE sweep family (Holm + BHY-FDR + HLZ t≥3 + N̂-from-ρ̄).**
  EXPERIMENT: new `m5_mt_haircut.py` (mirror cpcv_certify.py producer) — scrape per-variant binding-year t-stats from
  `sweeps/EURUSD_{60s,2m,5m}.md` + IDEAS_LOG, convert win-rate-vs-.541 to t, apply Holm/BHY(c(M)=harmonic, ρ̄≈0.4),
  estimate N̂=ρ̄+(1−ρ̄)M, emit mt_adjusted_p per survivor + a column in SWEEP_MATRIX.md.
  FALSIFIER: the procedure is useless if it FAILS to flag the already-killed |return|-retrain-POW1.0 / Stoikov / 24 null-60s
  channels as non-significant; DOWNGRADE m5xp UP to UNCERTIFIED if its BHY-adjusted binding-2025 p>0.05 at realized M.
  INCUMBENT: m5xp UP .577 / ACI .584 (certification robustness). NOVELTY: new_untried. (Harvey-Liu-Zhu, Harvey-Liu DSR)
  ✅ **DONE 2026-06-02 (`m5_mt_haircut.py`, `m5_mt_haircut_result.json`): PROCEDURE VALID** (all kills — Stoikov 0.4984,
  2m-UP 0.5168, 60s-xofi 0.5015, POW1.0-retrain — flagged non-significant, BY-adj p=1.000, negative t). **FINDING (two
  lenses diverge): m5xp UP POOLED edge (wr .5927, n3052, t=5.81) SURVIVES deflation decisively** (Šidák-p@N̂ 7.3e-08,
  HLZ t≥3 ✓) — the edge is REAL and the refit-CPCV cert (p10 .553/96%) is untouched. **BUT the harsh single-binding-2025
  lens (wr .5765, n1379, t=2.67) DOWNGRADES**: BY-adj p **0.158**>0.05 (Holm 0.137), **fails HLZ t≥3** (2.67<3.0); robust
  to M (39 scored→program 70: fails harder). Passes the accurate E[max] test but fails Šidák-at-N̂. **HONEST VERDICT: the
  binding-year headline 0.577 is MULTIPLICITY-INFLATED — use the refit-CPCV floor .553 as the durable UP figure, not .577.**
  Bar-to-beat for new UP candidates: refit-CPCV p10 .553 AND ideally binding-year t≥3 (wr≳.581 @n1400) for clean MT-significance.
  DOWN side even thinner (magweight POW=0.5 DOWN forward-2025 .538 < breakeven → t<0 → MT-fragile; the POW×cov sub-search is
  textbook multiplicity). Incumbent NOT demoted (no competitor produced; pooled+CPCV stand) — record refined, not killed.
**V2. ★⚡ True CSCV/PBO + Deflated-Sharpe + N̂ + MinBTL on the real per-bar gated-PnL matrix (not the E[max] proxy).**
  EXPERIMENT: adapt `cpcv_certify.py`/`m5_cpcv.py` — build (T×M) per-bar gated deriv-return matrix across logged 5m
  variants, add ~40 lines CSCV (S=16 quarterly blocks, C(16,8)) + getExpMaxSR DSR with N̂=ρ̄+(1−ρ̄)M; emit PBO, OOS-degradation
  β, MinBTL, SD1/SD2 next to per-year CI95; gate book-freezing on PBO≤0.05 AND CPCV path-clear.
  FALSIFIER: if CSCV labels the certified m5xp/m5xp_aci PBO>0.05 OR DSR-deflated edge <0.541, DOWNGRADE to UNCERTIFIED; if
  the 5m sample is below MinBTL at realized N̂, the VAL-selected edge is noise. INCUMBENT: m5xp/ACI certification.
  NOVELTY: variant_of_tried (cpcv_certify exists; CSCV/PBO/DSR are the new exact computations). (Bailey-LdP PBO/DSR)
  ✅ **DONE 2026-06-02 (`m5_cscv_pbo.py`, `m5_cscv_pbo_result.json`):** T=5810 indep gated bars, 10 configs
  (UP/DOWN × cov{.30,.15,.10,.05,.02}), C(16,8)=12870 CSCV splits. **PBO 0.242 (>0.05 FAIL); DSR @M=10 .906 / @N̂=6
  .934 / @prog70 .745 (<0.95 FAIL); MinBTL @prog70=118 < deployed n153 (PASS).** Falsifier triggers — but the driver
  is operating-point SELECTION instability: **OOS-degradation slope −0.505** (= the corr(VAL,OOS)=−0.54 anti-transfer),
  NOT edge-absence: only **3.1%** of splits put the IS-best below breakeven; IS-best OOS win-rate mean .667 / **p10 .589**.
  Deployed UP_cov0.05 frozen-pools to .647 (overstated vs refit floor .553). **CONSISTENT WITH V1: edge real + robustly
  profitable; the cover/side/year selection is overfit-prone → size on the refit-CPCV floor .553, never the optimistic
  operating point.** Incumbent refined, not killed. (Caveat: combined UP+DOWN family + ultra-thin cov0.02 inflate PBO
  somewhat; the anti-transfer slope is the real signal.)
**V3. ★⚡ Diebold-Mariano HAC head-to-head as the formal challenger-promotion gate (replaces eyeball CI95).**
  EXPERIMENT: add `dm_test()` to the strategy-eval harness — per-bar directional-loss diff challenger−m5xp_aci per test year,
  Newey-West DM (lag≈horizon−1=overlap), Bonferroni over sweep count; promote only on significant DM AND CPCV path-clear.
  FALSIFIER: KILL any challenger whose DM vs m5xp_aci is not significant at 5% (Bonferroni) in binding 2025, even if point
  win-rate looks higher. INCUMBENT: m5xp_aci .584 (promotion gate). NOVELTY: new_untried. (Diebold-Mariano)
**V4. ★⚡ Impose-the-null double/block bootstrap Type-I&II certification (joint cross-year week-blocks).**
  EXPERIMENT: adapt `cpcv_certify.py` block_boot — demean each year's gated edge to exactly 0.50, block-bootstrap 5-day
  calendar weeks JOINTLY across years (B=10000, preserves cross-fold residual corr a per-fold CV ignores); optional known
  edge-injection p0 for Type-II power; report exact p alongside CPCV.
  FALSIFIER: KILL the UP edge if realized win-rate lies inside the central 95% of the impose-null distribution (p>0.05) OR
  Type-I at the operating gate >5% (a demeaned clone clears the gate ≥5% of the time). INCUMBENT: m5xp UP .577.
  NOVELTY: new_untried. (Harvey-Liu luck-vs-skill / false-and-missed-discoveries)
**V5. ★⚡ Empirical-Bayes / James-Stein shrinkage of every VAL edge BEFORE selection + LFDR re-rank.**
  EXPERIMENT: replay the sweep ledger — treat each row's per-year VAL AUC/wc_ret as r_i with bootstrap SE, fit σ_μ by MoM on
  cross-row dispersion in `sweeps/EURUSD_*.md`, re-rank survivors by SHRUNK-VAL and dependence-shifted LFDR lower bound; pick
  the operating book by shrunk-VAL NOT VAL-max; backtest the selection vs historical VAL-max picks.
  FALSIFIER: KILL if shrink/LFDR selection does NOT raise mean OOS wc_ret of the chosen book vs VAL-max picks (corr(shrunk,OOS)
  no less negative); LFDR is mis-specified if it fails to flag the 24 null-60s channels as high-LFDR. INCUMBENT: VAL-max rule
  (corr(VAL,OOS)=−0.54) / m5xp choice. NOVELTY: new_untried — DIRECTLY attacks the −0.54 pathology. (Chen-Zimmermann; Harvey-Sancetta-Zhao)
**V6. ⚡ Granular per-week eCDF lower-quantile dominance over 0.541 + Holm step-down across the 3 OOS years.**
  EXPERIMENT: add a granular-window quantile report to strategy-eval — per UP/DOWN candidate output per-week win-rate eCDF
  across test years, require 40th-percentile-week > 0.541 (rejects few-lucky-window wins); re-rank current survivors.
  FALSIFIER: FLAG the incumbent UP edge if its median-week win-rate <0.541 (regime-concentrated); a new method is real only if
  it lower-quantile-dominates 0.541 EVERY year. INCUMBENT: m5xp UP .577 (regime-concentration). NOVELTY: new_untried.

### LEVER 2 — GATE / SELECTION (cheap, our whole edge is a confidence-gated bet; ACI is the incumbent family)
**G1. ★⚡ Volatility-normalized conformity score (divide meta/primary margin by online RV before the ACI step).**
  EXPERIMENT: in `m5_conformal.aci_gate`, replace sm_t with sm_t/vol̂_t (EWMA RV from 1m bars; variant uses the magnitude-model
  output already on the m5_magweight path); rerun `m5_aci_cpcv.py` nested-refit; report per-year win-rate SPREAD + binding-2025
  win@n. FALSIFIER: KILL if vol-norm does not reduce the 2024/25/26 spread AND does not raise binding-2025 win at ≥ incumbent n
  (no gain over ACI .584@n764); KILL on up-rate∈[.47,.53] breach. INCUMBENT: m5xp_aci .584. NOVELTY: new_untried. (Gibbs-Candès; Bhatnagar)
**G2. ⚡ Multi-expert online-aggregated ACI (DtACI/FACI/AgACI): K experts over a γ-grid, exp-weight on pinball loss.**
  EXPERIMENT: wrap `m5_conformal.aci_gate`'s scalar θ in K=8 experts (γ∈{.001..0.128}) with DtACI weight update
  (w_i∝exp(−η·pinball)); drop into `m5_aci_cpcv.py`; certify nested-refit per year vs single-γ ACI. KILLS the single-γ choice
  that is dangerous given corr(VAL,OOS)=−0.54. FALSIFIER: KILL if no aggregated variant raises binding-2025 win above .584 at
  ≥ equal n AND raises the worst of {2024,25,26} above the incumbent's worst year under refit-CPCV; KILL if it merely matches
  the ex-post best single γ. INCUMBENT: m5xp_aci .584. NOVELTY: variant_of_tried. (Bhatnagar; Gibbs-Candès)
**G3. ⚡ Conformal PID gate (ACI = P-term; add Integral cumulative-error + Derivative scorecaster feed-forward).**
  EXPERIMENT: extend `m5_conformal.aci_gate` to θ_t = Kp·err + Ki·EWMA(err) + g(session,hour,vol_bucket), g fit on train+val
  only; sweep (Kp,Ki) worst-VAL-half; certify per-year + post-break recovery-lag via `m5_aci_cpcv.py`. Targets the 2026-collapse
  RECOVERY-LAG, not just reaction. FALSIFIER: KILL if PID does not beat ACI .584@n764 at ≥ equal n on worst-VAL-half under
  refit-CPCV (and not below the .579 fixed floor), OR if I/D terms help only in-sample. INCUMBENT: m5xp_aci .584.
  NOVELTY: variant_of_tried. (Angelopoulos conformal-PID)
**G4. ★ Difficulty-adaptive per-sample gate threshold via QRF/SPCI over [session,vol,OFI,xpair-residual].**
  EXPERIMENT: train an sklearn QRF mapping gate-features → meta-residual quantile (calibrated on VAL OOB residuals from the
  seed-ensemble), set per-bar threshold from it; slot into `m5_aci_cpcv.py`. Fires harder on "easy" bars, abstains on "hard".
  FALSIFIER: KILL if conditional-threshold gate does not exceed ACI .584 at n≥700 in binding-2025 under refit-CPCV, OR per-year
  CI95-lo dips below .541, OR QRF thresholds overfit (VAL-up/OOS-down). INCUMBENT: m5xp_aci .584. NOVELTY: new_untried. (AA-CRC/SPCI)
**G5. ★ Slow/fast directional-agreement gate (momentum-turning-point): SLOW companion GBM on 15m/30m-aggregated cross-pair
  features, trade only when sign(slow)==sign(fast); abstain on disagreement.** A NEW gate INPUT, not a conformal variant.
  EXPERIMENT: adapt `m5_xpair_production.py` (reuse MX_HOR machinery) to train a SLOW primary; gate = sign-agree ANDed onto the
  meta gate; deriv-faithful per-year + certify on `m5_cpcv_refit.py`. FALSIFIER: KILL if agreement-gated UP does not beat
  binding-2025 .577 at ≥80% of trade count, OR CPCV purged paths at the gate don't clear .541. INCUMBENT: m5xp UP .577.
  NOVELTY: new_untried. (Harvey-Goulding-Mazzoleni momentum turning points)
**G6. ★ Macro-release liquidity-regime TIME gate (hard-block ±60s of top-tier US release; over-weight the [+1m,+60m] window).**
  Distinct from the prior NULL "macro-news as a directional FEATURE" — this gates by TIME-since-release liquidity, not signal.
  EXPERIMENT: tag each 5m bar minutes-since-last-US-release from `macro_calendar.parquet` (loaded by fetch_calendar.py); add a
  hard release-minute block + post-release-window flag onto the `m5_aci_cpcv.py` UP stream; gate-on vs gate-off per year.
  FALSIFIER: KILL if the post-release-window subset UP win-rate is CI95-indistinguishable from all-bars in 2024 AND 2025, OR
  blocking the release-minute doesn't raise win without crushing n. INCUMBENT: m5xp_aci .584. NOVELTY: new_untried. (Chaboud "Rise of the Machines")

### LEVER 3 — LOSS / LABELING (the magnitude→direction bridge + the genuinely-new training targets)
**L1. ★ Differentiable deriv-payoff / Sharpe loss on a direct signed position (tanh head, ties-LOSE + breakeven baked in).**
  EXPERIMENT: adapt `m5_deep_ens.py`'s torch MLP — replace BCE with L=−mean(w_t·deriv_payoff_t), w_t=tanh(f(features)),
  payoff ties→loss + 0.541 offset; train 2012-21, threshold on worst-VAL-half, feed the trained margin into the meta+ACI gate.
  FALSIFIER: KILL if binding-2025 UP at the gate does not exceed .584 at ≥ comparable n, OR CPCV fails, OR the gain traces to
  magnitude (sign-shuffle). INCUMBENT: m5xp_aci .584. NOVELTY: variant_of_tried — the loss IS the Deriv economics, never done.
  (Lim-Zohren-Roberts deep-momentum Sharpe-loss; Harvey-Wang ML-meets-Markowitz)
**L2. ★ Triple-barrier path-dependent TRAINING labels (3-class +1/−1/0 on 1s mid barriers within 300s), EVALUATE on true
  300s mid-to-mid sign (ties-LOSE).** Different target GEOMETRY + built-in abstention class.
  EXPERIMENT: build TBL labels from 1s ticks (λ∈1-3bp, vertical=300s), train 3-class GBM on m5xp features
  (`m5_xpair_production.py`); UP only on class=+1; score ACTUAL 300s sign; tune λ worst-VAL-half. FALSIFIER: KILL if, scored on
  real settlement, TBL-gated UP fails to beat .584 binding-2025 at comparable n, OR DOWN class stays ≤.55, OR abstention
  collapses n. INCUMBENT: m5xp_aci .584 UP / DOWN open. NOVELTY: new_untried. (López de Prado AFML; Bujak-Michańków-Ślepaczuk)
**L3. ★⚡ Worst-window SoftMin / Entropic-VaR robust training objective (turn "select-on-worst-VAL-half" INTO the loss).**
  EXPERIMENT: wrap the `m5_deep_ens.py`/meta head loss as L=BCE_pool + λ·SoftMin_τ over per-quarter win-rate-minus-.541
  (τ∈{.05,.2,1}, λ∈{.1,.5}); for the GBM primary approximate via per-quarter sample-weighting up-weighting worst blocks; select
  worst-VAL-half, per-year. Directly attacks the 2026-collapse / −0.54 fragility. FALSIFIER: KILL if it does NOT raise the
  MINIMUM per-year UP win-rate across 2024/25/26 vs the incumbent's floor, OR pooled binding-2025 degrades below .584.
  INCUMBENT: m5xp_aci .584 + per-year floor. NOVELTY: new_untried. (DeePM regime-robust net-of-cost)
**L4. ★ Label-conditional asymmetric ACI (separate UP/DOWN error budgets, α_down ≪ α_up) — DOWN rescue without retraining.**
  EXPERIMENT: extend `m5_conformal.py` — split calibration by predicted side, run two ACI thresholds (α_up vs much stricter
  α_down); evaluate DOWN-only wc_ret per year with the up-rate tripwire; require DOWN n≥100. FALSIFIER: KILL if the
  label-conditional DOWN gate cannot produce ANY test year with DOWN-only win CI95-lo >0.541 at n≥100 (confirms DOWN
  unsolvable from these features). INCUMBENT: DOWN rescue (currently .533/dead 2025); must not degrade UP .584.
  NOVELTY: new_untried. (AA-CRC label-conditional risk control)
**L5. ★⚡ MADL as the SELECTION/gate-threshold metric ONLY (no retrain) — pick θ to max |return|-weighted directional PnL.**
  Distinct from the KILLED |return|-weighted RETRAIN (POW=1.0 collapsed 2026) because it NEVER rebalances training.
  EXPERIMENT: feed realized signed 5m returns of moved bars into a MADL scorer in the strategy-eval harness; tune the meta/ACI
  threshold worst-VAL-half by MADL=−(1/N)Σ sign(R·R̂)·|R|; eval wc_ret per year; full-refit CPCV at the MADL gate.
  FALSIFIER: KILL if the MADL-selected gate does not beat m5xp UP .577 binding-2025 at ≥ trade count, OR CPCV at the MADL gate
  fails. INCUMBENT: m5xp UP .577 / ACI .584. NOVELTY: new_untried. (Michańków-Sakowski-Ślepaczuk MADL; López-Herrera 8-FX)
**L6. ★ Extreme-value / focal asymmetric loss to RESCUE DOWN on the FULL set (no subsetting) — tail-up-weight large-|neg-ret|
  + asymmetric UP/DOWN cost matrix.** Distinct from the KILLED DOWN specialist (which subset-trained → anti-transferred).
  EXPERIMENT: retrain the cross-pair head with focal-γ / EVL tail-weighted BCE on large-negative-return samples + a
  DOWN-up-weighted cost matrix; adapt `m5_updown.py` for the DOWN-split eval; sweep γ; score DOWN-split per year.
  FALSIFIER: KILL if DOWN-split binding-2025 stays ≤~.52 across all γ (DOWN unsolvable), OR up-weighting the down-tail degrades
  certified UP below .584 (net-negative). INCUMBENT: DOWN rescue; must not degrade UP .584. NOVELTY: new_untried. (FinGAT; Lucchese cost-matrix)

### LEVER 4 — FEATURE / INPUT (NEW carriers that survive sign-invariance; on-disk only — externals are the H-track)
**F-PLS. ★ PLS supervised dimension-reduction of the 239 multi-TF features targeting SIGNED 5m return.** Genuinely different
  from the null PCA/RMT: PLS comp-1 is the maximally-DIRECTION-predictive linear combo (target-aware), attacking the
  sign-invariance wall head-on. EXPERIMENT: fit sklearn PLSRegression (n_comp 1-8, worst-VAL-half, TRAIN-only fit) on the
  239+cross-pair block, feed K PLS scores into the m5xp primary (`m5_xpair_production.py`); within-|ret|-bin sign-shuffle.
  FALSIFIER: KILL if PLS-comp-1 signed AUC ≤0.52 on VAL (as sign-invariant as PCA), OR UP 2025 fails to exceed .577 at n≥618,
  OR edge vanishes under sign-shuffle (then it's magnitude). INCUMBENT: m5xp UP .577. NOVELTY: new_untried. (Gu-Kelly-Xiu)
**F-INT. ★ Explicit micro×macro-state INTERACTION features (z = x_micro ⊗ c_macro): each top-20 m5xp feature × {RV-quantile,
  session one-hot, intraday-clock, trend-sign}.** GKX's core fact: nonlinear edge IS signal-by-state interactions; the DOWN side
  may be a 3-way interaction absent as a main effect. EXPERIMENT: add product columns to `m5_xpair_production.py`, retrain
  primary, refit meta, worst-VAL-half; inspect DOWN-split AUC per state bucket; CPCV at gate. FALSIFIER: KILL if neither UP
  (>.577) nor DOWN (>.52 in any state bucket) improves at the binding year, OR every interaction column has gain below the
  existing top-10. INCUMBENT: m5xp UP .577 / DOWN rescue. NOVELTY: variant_of_tried. (Gu-Kelly-Xiu; Chen-Pelger-Zhu)
**F-FFD. ★ Fractionally-differentiated channels at the min ADF-stationary d (~0.15-0.35) instead of d=1 returns — preserves
  ~0.99 of the long-memory standard returns throw away.** Untried at 5m; orthogonal to every prior return-feature.
  EXPERIMENT: compute FFD (fixed-width, sweep d∈0.05..0.6, per-window ADF-select TRAIN-only) of the 1m EURUSD mid AND the
  cross-pair USD-residual; add as new primary channels in `m5_xpair_production.py`; retrain, worst-VAL-half, CPCV.
  FALSIFIER: KILL if FFD does not lift 2025 UP above .584 AND clear all CPCV paths, OR up-rate∈[.47,.53] violated, OR
  sign-invariant (magAUC≫dirAUC). INCUMBENT: m5xp_aci .584. NOVELTY: new_untried. (López de Prado FFD; Bujak SAE-MLP)
**F-PATH. ★ Path/mark-to-market STATE features (signed cumret since NY-open, bars-since-MA-sign-flip, (close−VWAP)/ATR,
  time-in-state).** NOT the |return|-weighted retrain (reweighted samples) — this ADDS path/regime conditioning the static GBM
  lacks; the DRL paper's single largest gain was m2m state. EXPERIMENT: engineer 3-4 path-state features at each 5m decision
  point on 1m bars; add to `m5_xpair_production.py` primary, retrain, keep gate, refit meta; per-year + CPCV.
  FALSIFIER: KILL if path/m2m features move neither 2025 UP above .577 nor DOWN above .52; specifically KILL if up-rate∈[.47,.53]
  breached (feature leaks trivial drift). INCUMBENT: m5xp UP .577 / DOWN rescue. NOVELTY: new_untried. (Briola DRL active-HFT)
**F-OFI. ⚡ Integrated-OFI (PC1, L1-normalized) of the EURUSD OWN multi-window OF-proxy block + lags {1,2,3,5,10}, replacing
  the raw correlated OF_COLS.** Denoises the OWN-flow channel (distinct from the killed contemporaneous cross-impact terms).
  EXPERIMENT: stack the existing OF_COLS into a per-bar matrix, fit PCA on 2012-21 TRAIN only, project to L1-norm PC1, add
  ofi_I + lags to `m5_xpair_production.py` primary; retrain, worst-VAL-half, CPCV. FALSIFIER: KILL if integrated-OFI does not
  raise moved-bar dir-AUC above the raw-OF baseline by a CI-separated margin, OR adds nothing to 2025 UP at the gate, OR up-rate
  breached. INCUMBENT: m5xp UP .577. NOVELTY: variant_of_tried. (Cont multi-level OFI). NOTE: own-flow PC1 only — cross-impact
  matrix already null (.5015@60s, min1_xofi); do NOT re-run cross terms.
**F-LAG. ★⚡ VAR(p) per-feature AIC-selected heterogeneous lags (each cross-pair/OF input its OWN optimal lag) before it enters
  the primary, instead of one global lookback.** Cheap, CPU-only, untried as an explicit per-feature lag-select step.
  EXPERIMENT: fit VAR/AIC per feature on 2012-21 to get a per-feature lag, build the lagged matrix, retrain
  `m5_xpair_production.py`; freeze lags from TRAIN; per-year + CPCV. FALSIFIER: KILL if AIC-lagged inputs do not improve OOS
  dir-AUC (>+0.005) AND 2025 win, OR train-chosen lags don't transfer (re-estimate on VAL differs AND OOS drops).
  INCUMBENT: m5xp UP .577. NOVELTY: new_untried. (EURUSD info-fusion forecasting)

### LEVER 5 — ENSEMBLING / STACKING (DL as decorrelated member; floor/tail-stabilizer, NOT edge-finder)
**E1. ★⚡ Online dynamic performance-weighted ensemble (BOA / inverse-recent-loss softmax over frozen books {m5xp, m5xp_aci,
  seed-MLP, magweight}).** Cheapest possible — pure re-weighting on already-saved OOS prediction streams; attacks 2026 collapse.
  EXPERIMENT: tiny script reusing `m5_conformal.py`'s collect() — per-bar BCE per frozen book over rolling k∈{60,120,250},
  softmax-blend (τ∈{.1,.5,1,2}) AND a tuning-free BOA variant; select (k,τ) worst-VAL-half; deriv-faithful wc_ret + per-year.
  FALSIFIER: KILL if binding-2025 UP ≤ m5xp_aci .584 at ≥ n, OR weights collapse to constant (no regime dependence), OR CPCV at
  the gate fails; up-rate∈[.47,.53]. INCUMBENT: m5xp_aci .584. NOVELTY: new_untried — supersedes v1-C1 (which was a static
  worst-VAL-half WEIGHT; this is ONLINE). (Carta expert-aggregation; HAELT)
**E2. ⚡ BOA-on-conformalized-experts (conformalize each model with ACI FIRST, THEN online-aggregate via Bernstein Online
  Aggregation on pinball loss).** The EPF headline: conformalize-then-aggregate beats either layer alone; inference-only.
  EXPERIMENT: adapt `m5_conformal.py` — collect per-bar (ts,score,win) streams for K experts, wrap each in aci_gate(), run BOA
  over 2024/25/26 updating weights on realized 5m-sign pinball loss per day; decode the aggregated band's UP win-rate and n.
  FALSIFIER: KILL if BOA-on-conformalized ≤ best single conformalized expert (.584) at ≥ n in binding-2025, OR degenerates to
  follow-the-leader on one expert. INCUMBENT: m5xp_aci .584. NOVELTY: new_untried. (EPF adaptive-prob-forecasting; Carta)
**E3. ★ Pooled UNIVERSAL 7-pair model (pairs as TRAINING ROWS + pair-id embedding + per-pair γ thresholds), applied to
  EURUSD.** Multiplies samples ~7×; POOLS (adds data) not subsets (which killed ranking). The one untested STRUCTURAL lever.
  EXPERIMENT: adapt `m5_xpair_production.py`/`exp_allpairs.py` — stack moved-bar samples from all 7 majors (USD-residualized)
  with a pair-id embedding + per-pair balanced labels, train one GBM/MLP; eval ONLY EURUSD test rows at the m5xp gate; ablate
  the embedding to prove it's not just more EURUSD data; CPCV. FALSIFIER: KILL if pooled-universal EURUSD 2025 UP ≤ .577 at
  equal trades, OR anonymous pooling (no embedding) matches it, OR up-rate breached per pair. INCUMBENT: m5xp UP .577.
  NOVELTY: variant_of_tried — supersedes v1-D1/D2 with the embedding+per-pair-γ specifics. (Sirignano-Cont; GKX; Lucchese)
**E4. ⚡ Vincentization (quantile-averaging) with bias+scale correction (V_a^w) replacing the 50/50 probability blend.**
  Quantile aggregation is provably sharper than the linear pool for calibrated/overdispersed members; recovers ~60% of the
  ensemble gain the 50/50 blend (which TIED at .5776) leaves on the table. EXPERIMENT: take the saved `m5_deep_ens.py` member
  outputs, build per-member predictive dist (logit=location + fitted scale), Vincentize with (a,w0) fit on VAL by CRPS, decode
  sign at the median; gate via meta+ACI. FALSIFIER: KILL if Vincentized sign does not beat the .5776 50/50-blend / m5xp .577 on
  binding-2025 worst-VAL-half (again collapses to ~GBM). INCUMBENT: m5xp UP .577 + .5776 blend tie. NOVELTY: new_untried.
  (Aggregating-distribution-forecasts deep-ensembles)
**E5. ⚡ Lucky-Factors null-imposing incremental bootstrap as the ACCEPTANCE GATE for every ensemble member.** Residualize each
  candidate against the m5xp output, reattach only the mean-zero incremental component under the null, block-bootstrap, admit
  only if realized directional-loss reduction beats the null. Stops "lucky" members (standalone>.52 but zero incremental value).
  EXPERIMENT: wrapper around `m5_cpcv_refit.py` outputs — regress candidate=d0+d1·m5xp_score, form residual-null series,
  block-bootstrap (block≈300s), test augmented gated-edge improvement p vs m5xp; pass/fail before freezing any member.
  FALSIFIER: a candidate with standalone dir-AUC>0.52 but null-imposed p>0.05 is REJECTED regardless of standalone number; KILL
  the wrapper if NO candidate ever clears (group exhausted). INCUMBENT: m5xp primary (incremental). NOVELTY: variant_of_tried.
  (Harvey-Liu Lucky Factors)
**E6. ★ DOWN-side rescue via Sharpe-Indifference-Curve approval (admit DOWN as a DIVERSIFIER, not on standalone >.541).**
  A negative-edge DOWN strategy can be approved if its return-correlation to the live UP book is negative enough to raise the
  COMBINED book Sharpe (Eq.12). Genuinely new framing for the unsolved DOWN side. EXPERIMENT: reuse `m5_updown.py` side-split
  returns — treat UP-gated returns and a DOWN candidate (inverted cross-pair / rally-sell) as two strategies, compute realized
  correlation on overlapping 5m bars + each side's deriv Sharpe, plug into the indifference equation; approve only if ΔSR_B>0.
  FALSIFIER: REJECT (confirms DOWN dead even as a diversifier) if the DOWN candidate's correlation to UP is ABOVE the
  indifference threshold (positive/near-zero). INCUMBENT: combined UP+DOWN Sharpe vs UP-only. NOVELTY: new_untried.
  (Bailey-LdP-del Pozo strategy-approval)

### LEVER 6 — REGIME / NONSTATIONARITY (attacks corr(VAL,OOS)=−0.54 and the 2026 collapse directly)
**R1. ★⚡ Magnitude-as-dynamic-threshold gate (fire UP only when predicted |ret300| ∈ top trailing tercile).** The ONE place the
  certified-STRONG size signal (AUC .73-.79) can lift direction without violating sign-invariance — it SELECTS the regime where
  sign is inferable, not the sign. EXPERIMENT: load the magnitude model's predicted |ret300| per moved bar, AND-condition
  `mag_pred ≥ trailing-q66` onto the existing sess_ny & meta≥thr gate (`m5_conformal.py`/`m5_updown.py`); re-tune the magnitude
  pct + meta thr jointly worst-VAL-half; side-split per year + full-refit CPCV. FALSIFIER: KILL if mag-pct gating does not raise
  binding-2025 UP above .584 at ≥500 trades, OR the lift comes only from shrinking n (no EV/hr gain), OR CPCV p10 at the gate
  <0.541. INCUMBENT: m5xp_aci .584. NOVELTY: variant_of_tried — supersedes v1-G1 with the explicit magnitude-model coupling.
  (Gong HF-FX dynamic-α; the program's own magnitude edge)
  ❌ **KILLED 2026-06-02 (`m5_magdyn.py`, `m5_magdyn_result.json`) — MECHANISM-FIRST KILL.** Trained a 5m magnitude model
  (|ret300|≥Q75, 42 rv/bb/atr feats) and tested the premise on val+test gated UP bars: **win-rate is FLAT across
  mag-forecast quartiles (q0 .628/q1 .646/q2 .640/q3 .632; top−bottom lift +0.0031, non-monotonic).** The premise
  ("bigger forecast move → more inferable sign") is empirically FALSE — a **direct operational confirmation of the
  sign-invariance theorem at 5m** (and the book already gates on bb_width compression, so residual magnitude adds
  nothing). Stage-B dynamic gate (score=conf+λ·mag) confirmed: VAL-selected λ=0.2 looked better on worst-VAL-half
  (.680 vs .623) but **ANTI-TRANSFERRED** — binding-2025 .588 < fixed-conf baseline .603, pooled .596 < .675 (textbook
  corr(VAL,OOS)=−0.54). Does NOT beat the incumbent; magnitude×direction bridge is null for the GATE at 5m. (Mag stays
  a SIZE edge → MAGNITUDE_FINDINGS.md, not direction.)
**R2. ⚡ High-vol-regime AND-gate using GKYZ range-vol (we have 1m OHLC), abstain in calm/efficient regimes.** Multiple papers:
  ML directional predictability is ~7× concentrated in high-vol/high-uncertainty states; the nonlinear cross-pair edge should be
  largest there. EXPERIMENT: compute causal 5m GKYZ range-vol per moved bar in `m5_updown.py`, add as an AND-gate (like sess_ny)
  on the m5xp meta gate, re-select thr per vol-bucket worst-VAL-half; test the inverse (low-vol) as control; certify on
  `m5_cpcv_refit.py`. FALSIFIER: KILL if high-vol conditioning does not raise binding-2025 UP above .577 at non-trivial n, OR the
  high-vol subset edge fails to exceed the low-vol subset (no monotone vol→edge), OR CPCV p10<.541. INCUMBENT: m5xp UP .577.
  NOVELTY: variant_of_tried. (Avramov-Cheng-Metzker; Goulet; 8-FX)
**R3. ★ Nonparametric kNN-in-state-space direction matcher (predict 5m sign from the avg subsequent return of the K most-similar
  TRAIN-era bars).** Instance-based, parameter-free, naturally regime-adaptive — structurally different from the killed
  GBM/online-ARF families. EXPERIMENT: new script reusing the m5_xpair feature build + MX.nonoverlap_chrono — Z-score a curated
  state subset (vol, OFI, session, USD-residual, DXY/EURGBP proxies, trend) over a rolling 1-2yr window, mask last 3yr, kNN
  (K=15-20% nearest), signal=sign(mean subsequent 300s ret); gate on signal strength; side-split + CPCV. FALSIFIER: KILL if
  combined 5m dir-AUC ≤0.52 worst-VAL-half, OR binding-2025 UP does not exceed .584 at comparable n, OR CPCV paths fail p10≥.541.
  INCUMBENT: m5xp_aci .584. NOVELTY: new_untried. (Harvey-Mulliner-Xia-Fang-vanHemert Regimes)
**R4. ★⚡ Covariate-shift-weighted conformal calibration (weight VAL calibration points by an estimated train→test density-ratio
  w(x) from a classifier separating recent-test-like vs old-train-like rows).** Calibrates the gate to the CURRENT regime —
  attacks corr(VAL,OOS)=−0.54 at its root; needs only unlabeled test covariates. EXPERIMENT: in `m5_conformal.py`, train a
  logistic classifier on (RV, session, USD-residual dispersion) to distinguish 2024-26 from 2012-21 rows, use its odds as w(x),
  re-derive the gate weighting VAL points by w; compare year-to-year win-rate VARIANCE vs unweighted ACI. FALSIFIER: KILL if
  weighted calibration does not reduce year-to-year variance vs ACI AND does not lift the worst year; if w(x)≈constant (no
  detectable shift) there's nothing to exploit. INCUMBENT: m5xp_aci .584. NOVELTY: new_untried. (Angelopoulos conformal-risk; Tibshirani covariate-shift)
**R5. ★⚡ Recency-decay sample-weighting of the primary GBM (exp chronological w_t=exp(−(T−t)/τ), τ∈{2,4,8,∞}yr).** PURELY
  chronological recency — DISTINCT from the |return|-weighting that collapses 2026; the cheapest single-knob drift fix.
  EXPERIMENT: adapt `m5_magweight.py` (swap magweight() for exp time-decay), sweep τ, select worst-VAL-half (NOT VAL-max),
  side-split per year; KEY CHECK = does 2026 hold up better than the incumbent. FALSIFIER: KILL if no τ improves the WORST test
  year (esp. 2026) over the unweighted incumbent, OR if a τ that helps 2024/25 degrades 2026 (the |return|-retrain failure mode).
  INCUMBENT: m5xp UP .577 + 2026-collapse rescue. NOVELTY: new_untried. (Avramov-Cheng-Metzker; EPF windowing)
**R6. ★ VIX / risk-off DOWN-enabler (acquire free daily VIX, forward-fill to 5m, enable DOWN only in high-VIX risk-off
  windows).** Cheap external regime input aimed specifically at the unsolved DOWN side (vol spike → USD bid → EURUSD down-skew).
  EXPERIMENT: acquire daily VIX (Yahoo/Stooq), add VIX-state as a DOWN-side gate conditioner in `m5_downcond.py`/`m5_updown.py`;
  test whether DOWN win clears 0.541 in high-VIX bars across ALL THREE test years at n≥75/yr. FALSIFIER: KILL if VIX-state
  gating does not lift DOWN above 0.541 in high-VIX bars in all 3 years at n≥75, OR only cuts n without raising win, OR DOWN
  stays ~.52 (the D3-style outcome). INCUMBENT: DOWN rescue (~.52, dead 2025). NOVELTY: new_untried (lower-friction than the
  H-track externals — VIX is free). (info-fusion forecasting)

## FIRST-TO-RUN QUEUE (genuinely NEW-or-LIVE + high-priority + cheap — the immediate batch, ordered)
All reuse existing scripts/outputs, all minutes-to-≤1hr, each has a concrete kill. Run in this order:
1. **V1** ✅ **DONE 2026-06-02** — multiple-testing haircut over the whole sweep family (★⚡, pure post-hoc calc). VALID (kills
   non-sig). Pooled m5xp UP edge SURVIVES (t=5.81); binding-2025 .577 multiplicity-inflated (BY-adj p .158, fails HLZ t≥3).
   **Bar-to-beat moved: durable UP floor = refit-CPCV p10 .553, not .577.** Incumbent refined, not demoted. `m5_mt_haircut_result.json`.
2. **V2** — true CSCV/PBO + DSR + MinBTL on the real gated-PnL matrix (★⚡). Companion to V1; replaces the E[max] proxy already
   in cpcv_certify.py with the exact overfit probability. Together V1+V2 re-anchor every "beat .577/.584" claim.
3. **R1** — magnitude-as-dynamic-threshold gate (★⚡, the magnitude×direction bridge). Highest-EV cheap NEW gate: marries the
   program's ONE strong edge (mag AUC .73-.79) to the direction gate without violating sign-invariance. On-disk mag model.
4. **E1** — online performance-weighted / BOA ensemble over the 4 frozen books (★⚡, pure re-weighting on cached streams).
   Cheapest plausible 2026-collapse fix; supersedes the static 50/50 blend that tied.
5. **G1** — vol-normalized conformity score for the ACI gate (★⚡, one-line change in m5_conformal). Cheapest regime-stationarity
   fix to the gate; reuses the in-flight m5_aci_cpcv harness.
6. **L5** — MADL as the SELECTION metric only (★⚡, no retrain). Re-picks the gate by |return|-weighted directional PnL — the
   safe half of the magnitude-weighting idea (avoids the POW=1.0 retrain collapse).
7. **R5** — recency-decay sample-weighting τ-sweep (★⚡, reuse m5_magweight skeleton). The cheapest single-knob test of whether
   the 2026 collapse is a drift problem; pairs naturally with the v1 B1 magweight-POW certification.
NOTE: the v1 FIRST-TO-RUN seven (A1 ACI-CPCV, B1 magweight-POW0.5 CPCV, A2 calibrate-gate, C1 stack-weight, F1 DOWN re-slice,
C3 GBM seed-ens, D2 pooled-GBM) are the LIVE near-miss certifications and run IN PARALLEL with this v2 queue — they certify
what's already on disk; v2 adds the new academic levers. A1 (ACI-CPCV) is the gating prerequisite for V1/V2's "ACI .584" bar.

## DOWN-SIDE RESCUE — explicit call-out (the open gap; every standalone DOWN channel is killed/marginal)
Mechanism (re-stated): DOWN is structurally near-efficient at 5m because down-moves are JUMP/INFORMED-dominated → downside
energy goes to MAGNITUDE not a predictable drift (sign-invariance). KILLED already (do NOT re-run): D3 USD-strength-gated .526,
A8c specialist .509 anti-transfer, A8b CI-lo .516, sidecurve, DOWN-confcurve. The LIVE/NEW DOWN levers with a plausible
mechanism, ranked:
1. **L4 label-conditional asymmetric ACI** (★, cheapest NEW): separate strict α_down budget on the EXISTING primary — fires
   DOWN only on rare genuinely-high-confidence shorts. The cleanest test of "is there ANY high-confidence DOWN tail".
2. **L6 extreme-value / focal asymmetric loss on the FULL set** (★): tail-up-weight large-negative-return moves WITHOUT
   subsetting (fixes why the A8c specialist anti-transferred). Must not degrade UP .584.
3. **R6 VIX risk-off DOWN-enabler** (★, free external): enable DOWN only in high-VIX windows — the regime where USD-bid →
   EURUSD-down should cohere. Cheapest external (VIX is free vs the H-track rate-diff/RR acquisitions).
4. **E6 Sharpe-Indifference-Curve approval** (★): admit DOWN as a NEGATIVE-correlation DIVERSIFIER to the UP book even if it
   fails standalone .541 — the one framing under which a sub-breakeven DOWN side can still raise the combined book's Sharpe.
5. **H2 risk-reversal sign** (external, highest ceiling, BLOCKED): FX 25Δ risk-reversal is the one genuinely-SIGNED positioning
   carrier we don't have. If anything rescues DOWN standalone it is this. See H-TODO-2.
Plus the v1 DOWN levers (F1 magweight DOWN re-slice — LIVE at .559/2025; F2 over-shoot fade as UP-in-down-regime).

## EXTERNAL-DATA ACQUISITION TODOs (separate track — dominant ceiling-EV; the binding constraint is INFORMATION not model)
These are the only levers with a plausible path to a STEP-CHANGE (not incremental) since the on-disk input carriers are
exhausted (~30 channels null, info-bound ~0.52 raw AUC). Acquisition is the blocker, not modeling.
- **H-TODO-1: Intraday DE–US 2y (Schatz / 2Y-UST) yield differential.** Source: Dukascopy historical or a bond-futures feed.
  Resample to 1-min, align to the EURUSD feature grid 2012-2026, add Δ(rate-diff) + short slope as signed primary+gate features.
  The single most-cited absent carrier (the catalog's flagged EXTERNAL lever). → feeds F-PLS/F-INT/G/V feature-admission.
- **H-TODO-2: Daily FX implied-vol + 25Δ risk-reversal (1w/1m tenors) for EURUSD.** Source: a vol-surface vendor or scrape.
  Risk-reversal SIGN = market's directional skew = the textbook signed-positioning carrier → the most mechanistically-aligned
  DOWN-rescue input. → feeds H2 (DOWN) + a daily-context gate.
- **H-TODO-3: EURGBP (and a 3rd USD cross) tick/1-s data.** Algebraically isolate the EUR leg cleaner than the 6-pair basket
  (FinGAT EUR-homogeneity weighting). Triangular @2m on 1-min bars was .499 — finer tick data is the untested variant.
- **H-TODO-4: free daily VIX (Yahoo/Stooq)** — LOW-friction, feeds R6 DOWN-enabler. Acquire first; it's free.
- **H-TODO-5 (lower prior): CFTC positioning + DXY/Bund futures intraday.** Completeness; ES/NQ already sign-flip-null.
NOTE: every acquired candidate must pass the **Harvey-Liu / double-selection-LASSO feature-admission gate** (orthogonalize
vs the m5xp 239-feature set, DS-LASSO |t|≥3, weak-factor Wald reject, then incremental block-bootstrap) BEFORE it enters the
primary — this is the V-lever discipline applied to externals so we don't add saturated/lucky features.

## HONEST META-NOTE (v2)
The on-disk INPUT carriers are exhausted; v2's highest-ROI moves are therefore (a) the VALIDATION levers (V1/V2) that re-anchor
whether the certified edge survives multiplicity-deflation — this is decision-relevant BEFORE more search — and (b) the cheap
NEW GATE/REGIME levers (R1 magnitude-threshold, G1 vol-normalized score, E1 online-weighted ensemble, R5 recency-decay) that
attack the corr(VAL,OOS)=−0.54 / 2026-collapse fragility, which is where the realized-EV loss actually lives. The realistic
prize remains INCREMENTAL (a few points of binding-2025 win, a tighter CPCV tail, a marginally-rescued DOWN, better EV via
calibration) — NOT >0.65 (proven unreachable on these inputs). The ONLY step-change lever is external data (H-track), gated by
the feature-admission discipline. Run V1+V2 first (they may move the bar), then the cheap NEW gate/regime levers in parallel
with the v1 near-miss certifications; certify any survivor on full-refit CPCV at the operating gate, freeze as a new book
(`m5_register_*.py`), update EURUSD_RESULTS.md + the leaderboard. Disproved-by-experiment beats untried.
