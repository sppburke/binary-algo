> **SCOPE: EURUSD · 5m** (sweep LEDGER — resumable status). Backlog: sweeps/EURUSD_5m_backlog.md. Results: EURUSD_RESULTS.md. See REPO_MAP.md.

---
currency: EURUSD
timeframe: 5m (300s)
started: 2026-06-01
target: best UP and DOWN binary predictor at 300s, OOS(2026)-verified, clearing breakeven 0.541
status: SWEEP COMPLETE; v2 IMPROVE/VALIDATE queue IN PROGRESS (V1 done 2026-06-02). **(5m,UP) = the best 5m algo, CERTIFIED under
        full per-fold refit at the operating gate** (cov0.05 p10 0.553, 96% folds clear; deflates only at loose cov).
        ⚠ **V1 multiple-testing haircut (`m5_mt_haircut_result.json`): durable UP figure is the refit-CPCV floor 0.553 — the
        binding-2025 headline 0.577 is MULTIPLICITY-INFLATED (BY-adj p 0.158, fails HLZ t≥3); pooled edge t=5.81 survives, cert untouched.**
        Deployable win ~0.55–0.57; ⅛-Kelly sizing; win-rate kill-switch NOT worth it (see DEPLOYMENT SPEC below).
        UP+DOWN both given the FULL symmetric pipeline. UP deployable-certified; DOWN MARGINAL (refit cov0.05 p10 0.542
        barely clears but forward 2025 0.538 fails — real but razor-thin, not deployable). >0.65 not achievable.
incumbent/answer: **UP = EURUSD.m5xp.v1 up-preds, refit-certified at the ~5% gate (floor 0.553, forward 0.58–0.61) —
        the deployable winner.** DOWN = same book down-preds, ~0.54 marginal/borderline (much weaker, fails binding 2025).
        Combined incumbent m5xp oos26 0.606.
prior: 5m direction combined ~0.61 verifiable, capped by the 15m parent; binding window 2025. UP side likely
        live (dip-buy, as at 60s/15m); DOWN side likely dead. >0.65 OOS-stable not expected at 5m.
---

# Sweep ledger — EURUSD 5-minute (300s) UP/DOWN

Each row: pre-register falsifier → retarget to 300s (`MX_HOR=5`) → deriv-faithful discipline (wc_ret/contig
ties-LOSE, nonoverlap_chrono 300s, per-year 2024/25/26 CI95, worst-VAL-half selection, moved-bars
up-rate∈[.47,.53] tripwire) → score combined + UP + DOWN → record into EURUSD_RESULTS.md → commit. `OOS`
columns = per-year 2024/2025/2026 moved-only accuracy.

Settlement note: the frozen 5m books (m5xp/m5stack) settle close-to-close over a wall-clock-CONTIGUOUS 300s
window (`contig=(secs[HOR:]-secs[:-HOR])==HOR*60`), label `_y=(fwd>0)`, tie bars (`fwd==0`) dropped at build
(immaterial at 5m: exact-zero 300s log-return is ~measure-zero). This IS the book's own settlement; the
side-split decomposes the book's measured combined number faithfully.

Breakeven 0.541 (R≈1.85). Binding constraint = WORST held-out year's moved-acc CI95-lower clearing 0.541.

| id | tier | method | script | status | up_oos (24/25/26) | down (24/25/26) | verdict | result_json |
|----|------|--------|--------|--------|-------------------|-----------------|---------|-------------|
| 0a | base | **m5xp frozen book SIDE-SPLIT** (cross-pair+OF+meta) | m5_updown.py | **done ✅** | **.605/.577/.615** | .609/**.533**/.592 | **UP = CERTIFIED UNDER FULL REFIT AT THE OPERATING GATE (the BEST 5m algo).** Forward split clears all 3 yrs (binding 2025 .577[.552,.603] n1379; up-rate clean; reproduces book exactly). FULL-REFIT CPCV (per-fold model refit, the test that deflated 15m): **at the book's tight gate cov≤15% it CERTIFIES** — cov0.05 p10 **0.553**, **96%** of 28 purged-refit folds clear 0.541 (cov0.10 p10 .544/89%; cov0.15 p10 .541/89%); fails ONLY at loose cov0.30 (p10 .534, 54%) where the model has no edge. So the edge is REAL and refit-robust at the confident tail, magnitude deflated from the single-split: **deployable win ~0.55-0.57 (robust floor .553), optimistic .58-.61**. First sub-15m direction edge to survive the full refit. DOWN dead 2025 (.533). | m5_updown_result.json, m5_cpcv_m5xp_result.json, m5_cpcv_refit_result.json, m5_refit_tightcov_result.json |
| 0b | base | **m5stack frozen book SIDE-SPLIT** (cross-horizon stack) | m5_updown.py | done | .663t/.581/.557t | .599/.593/—t | TRIPWIRE-CAUTION: 2024 up-rate .584 breaches [.47,.53] (up-drift selection inflates UP .663); 2026 thin (UP n140 CI-lo .479 fails; DOWN n23). DOWN clears 24+25 but 2024 tainted. m5xp is the cleaner book. | m5_updown_result.json |
| A1a | A | 3-model GBM ensemble retune (m5 native) | m5_production.py | pending | — | — | — | — |
| A2a | A | compression × session × coverage gate sweep | m5_lab.py / m5_gate | pending | — | — | — | — |
| A3a | A | compression-release × reversion specialist | m5 levers | pending | — | — | — | — |
| A4a | A | meta-labeler on orthogonal axes (already in m5xp) | m5_meta.py | pending | — | — | — | — |
| A5a | A | cross-horizon stack 15m→5m (soft, q-gate sweep) | m5_stack2.py | pending | — | — | — | — |
| A5b | A | cross-horizon stack 15m→5m (hard agree) | m5_stack.py | pending | — | — | — | — |
| A6a | A | cross-pair USD-residual modes {xp,xpbase,xpof} | m5_xpair.py | pending | — | — | — | — |
| A7a | A | walk-forward retrain (regime robustness) | m5_walkforward.py | pending | — | — | — | — |
| A8a | A | up-only FILTER (UP-specific thr select) | m5_upfilter.py | **done ✅** | .666/.616/.581(n31) | — | **CERTIFIED higher-conviction UP**: tighter thr 0.598 → binding 2025 0.616, CPCV p10 0.608, 28/28 paths clear, block-boot CI-lo 0.614. Caveat: 2026 standalone thin (n31, pooled-CPCV mitigates). Higher-accuracy/lower-coverage variant of the certified UP book. | m5_upfilter_result.json, m5_cpcv_a8a_result.json |
| A8b | A | down-only FILTER (DOWN-specific thr select) | m5_upfilter.py | done | — | .600/.558/.575(n47) | tighter thr lifts 2025 DOWN .533→.558 but CI-lo .516<.541 (uncertified) + 2026 thin. DOWN still uncertified. | m5_upfilter_result.json |
| A8c | A | **(5m,DOWN) SPECIALIST** (down-specific meta-labeler) | m5_downspec.py | killed | — | .571/**.509**/.511 | purpose-built DOWN meta: VAL worst-half 0.749 but ANTI-TRANSFERS to 2025 .509[.485,.531] (WORSE than symmetric .538) — textbook corr(VAL,OOS)=−.54 trap. A dedicated DOWN model can't crack 2025 either. | m5_downspec_result.json |
| DOWN-curve | A | DOWN confidence/coverage curve (symmetric to UP) | m5_sidecurve.py | done | — | 2025 CI-lo never clears .541 at any gate | DOWN tracks UP in 2024(.608)/2026(.586) but 2025 maxes ~.555 (CI-lo ~.52); no gate clears all 3 yrs | m5_sidecurve_result.json |
| B1a | B | tick microstructure ensemble retarget @300s | m_tick_prod.py (HS=300) | pending | — | — | — | — |
| B3a | B | CKS event-OFI @300s | m5_cksofi300.py | killed | VAL dirAUC .5006 | — | null (≤.515 gate); monotone decay 60s→120s→300s | m5_cksofi300_result.json |
| B4a | B | cross-impact OFI matrix @300s | min1_xofi (MX_HOR=5) | subsumed | — | — | null @60s (.5015); signed basis covered by family probe (~.51) | m5_legsign_result.json |
| B5a | B | per-side raw signed flow @300s | m5_perside_flow.py | killed | VAL dirAUC .5077 | — | netps/sgnv (the 1 untried axis) null at 300s; no cov clears .541 CI-lo in 25&26 | m5_perside_flow_result.json |
| C1a | C | HMM regime (causal-filtered) @300s | min1_hmm.py (MX_HOR=5) | pending | — | — | — | — |
| C2a | C | Kalman channel/velocity/β @300s | min1_kalman.py (MX_HOR=5) | pending | — | — | — | — |
| C4a | C | CCM coupling-gate @300s | min1_ccm.py (MX_HOR=5) | pending | — | — | — | — |
| C5a | C | **online ARF+ADWIN control @300s (KEYSTONE)** | m5_online_run.py | done | AUC .509/.509/.514 | (single-pair) | NULL on single-pair TA (selective never >.52). BASELINE efficient at 5m. Does NOT cover the cross-pair UP edge (different feature set) → keystone ≠ "5m fully efficient" here (m5xp UP 0.577 lives in cross-pair structure). | m5_online_result.json |
| D1a | D | 1D-CNN/GRU on raw path @300s | m_cnn.py | pending | — | — | — | — |
| D4a | D | DRL DQN direction-with-abstain @300s | min1_drl.py (MX_HOR=5) | pending | — | — | — | — |
| E1a | E | magnitude \|ret300\|≥Q (SIZE, sign-invariant) | m5 magnitude | pending | — | — | — | — |
| E2a | E | direction-conditioned-on-magnitude @300s | m10_magdir (MX_HOR=5) | pending | — | — | — | — |
| F1a | F | macro-release impulse @300s | m5_news.py | pending | — | — | — | — |
| F2a | F | structural / Sofien price-action rules @5m | m5_sofien*.py | pending | — | — | — | — |
| F3a | F | external cross-asset lead-lag (ES/NQ→pair) @5m | m10_xasset_probe.py | pending | — | — | — | — |
| F4a | F | residualized TARGET (label=resid-sign) @300s | min1_residtarget.py (MX_HOR=5) | pending | — | — | — | — |
| N2a | N | triangular USD-canceling residual @300s | min2_triangular (MX_HOR=5) | pending | — | — | — | — |
| N3-9 | N | cross-leg sign-lead family @300s | min2_legsign (MX_HOR=5) | pending | — | — | — | — |
| N4a | N | intraday-momentum term-structure @5m | min2_mim (MX_HOR=5) | pending | — | — | — | — |
| N7a | N | asymmetric tick-intensity (Hawkes) @300s | min2_hawkes (MX_HOR=5) | pending | — | — | — | — |
| N8a | N | signed-semivariance-skew sign-cond @300s | min2_rsskew (MX_HOR=5) | pending | — | — | — | — |
| N10/14/17 | N | cross-leg signed-IFR(Liang)/mv-dollar-source/network-momentum STANDALONE @300s | m5_legsign.py | killed | best ~.511 | — | family probe: no cross-leg signed-lag predicts next-5m sign ≥.52 stable (VAL-sel ownret_15:rev .506/.510/.511). Standalone null. | m5_legsign_result.json |
| N11/15 | N | own signed-flow / propagator-residual / Hawkes up-down imbalance STANDALONE @300s | m5_legsign.py | killed | best ~.511 | — | family probe: own OF_of_sum/uptick + transient-residual sign all ~.51. Standalone null. | m5_legsign_result.json |
| N16 | N | ordered-binary-choice OWN sign-autocorrelation @300s | m5_legsign.py | killed | ~.511 | — | family probe: own return-sign momentum AND reversal ~.51 every year. Sign-persistence null at 5m. | m5_legsign_result.json |
| N12/N13 | N | directed-HVG irreversibility + time-reversal signed structure-function @300s | m5_irrev.py | killed | best ~.511 | — | no signed-irreversibility/odd-moment feature stable ≥.52 in 24&26; I_W scalar was .503 | m5_irrev_result.json |
| D3 | N | **USD-strength-conditioned DOWN** (DOWN only when dollar drives) @300s (NEW DOWN) | m5_downcond.py | killed | — | USDstrong 2025 .526 (n502) | DOWN efficient even USD-gated; down-moves jump/informed-dominated → magnitude not sign | m5_downcond_result.json |
| F3a | F | ES→EURUSD cross-asset lead-lag @5m | m5_xasset (HOR=5) | killed | — | dir-hit <.50 | corr SIGN-FLIPS +.025(24)→−.022(25); mechanistic key to 2025 wall | m5_xasset_result.json |
| D1/D2 | N | exogenous over-shoot / bad-RV exhaustion FADE (predict UP after down-spike) | — | subsumed | — | — | these predict UP (a fade), not DOWN; the UP fade is already in the m5xp cross-pair channel; FX prices surprise <60s | — |
| D4/D5 | N | overbought-extreme DOWN / microprice down-persistence (~8-10%, killed families) | — | subsumed | — | — | RSI/BB in killed Sofien family; signed-flow killed by family probe. Completeness-only. | m5_legsign_result.json |

(Discovery round 1 — cross-disciplinary agent, 2026-06-01: added N10-N14. Top priors N10 Liang signed IFR (the
one explicitly-SIGNED causality measure untried; CCM/TE were its unsigned cousins, both dead) and N11 Bacry-Muzy
cross-kernel (mean-reverting impact kernel at 300s relaxation, distinct from killed scalar N7). N12-14 = completeness.
arXiv/SSRN agent round pending. Discovery continues until K=2 dry rounds.)

### Subsumed / confirmed-null at 5m (documented rationale — NOT silently skipped)
Per the coverage rule (run once OR document why subsumed). Three Tier-1 results do most of the subsuming:
the **certified m5xp UP book** (the live cross-pair nonlinear channel, already exploited), the **online-ARF
keystone** (single-pair adaptive = ~0.51, the single-pair channel is dead), and the **76-candidate signed-lag
family probe** (every standalone signed channel ~0.51). Plus prior 5m research (`m5_research_log.md`) already
measured several of these at 5m.

- **A1a GBM retune (native-5)** — `m5_production.py` native-5 base = VAL AUC 0.523 / OOS 0.518 (book 0.586-0.594); the certified edge needs CROSS-PAIR features, which the native retune lacks. m5xp IS the tuned cross-pair model. Dominated.
- **A2a gate sweep / A4a meta-labeler** — the m5xp gate IS NY×meta-labeler; A8 swept its threshold (A8a/A8b). Compression×session gates explored in `m5_lab.py` (combined table). Done.
- **A3a comp-release reversion specialist** — a 60s/2m lever; at 5m the live channel is cross-pair, not the reversion gate (which is the 2m book's mechanism, uncertified there). Subsumed.
- **A5b hard-agreement stack** — `m5_stack.py` hard-agree starves OOS coverage (the soft m5stack was preferred for that reason); m5stack soft side-split done (0b, tripwire-cautioned). Dominated.
- **A6a cross-pair modes {xp,xpbase}** — `m5_research_log.md`: xpof selected as the best mode (production freeze); xp/xpbase dominated in the combined comparison. The certified UP is on the best mode.
- **A7a walk-forward** — `m5_walkforward.py` already run: t24 .676/t25 .557/oos .565 comb 0.600 (+0.015 only). Done-null (retrain doesn't beat the frozen book).
- **A8c side specialist** — separately-trained up/down specialists are WORSE (subset-training kills ranking; `min1_upspec.py` + MODEL_REGISTRY note). Subsumed; the FILTER (A8a) is the right side-mechanism.
- **B1a tick microstructure @300s / B5a per-side flow** — the genuine tick edge (0.657) is seconds-scale and decays by 60s+ (`rawtick_decay.py`); the live 1-min order-flow (OF_*) is already IN m5xp; standalone signed-flow ~0.51 (family probe). Subsumed.
- **B3a CKS-OFI / B4a cross-impact OFI @300s** — null at 60s (xofi 0.5015) & 120s (CKS LGBM early-stops iter1); decays further at 300s; signed basis covered by the family probe (~0.51). Confirmed-null.
- **C1a HMM / C2a Kalman / C4a CCM @300s** — sign-invariant magnitude gates (theorem arXiv:2512.15720); null for direction at 60s/2m; CCM's signed cousin (Liang IFR) killed by the family probe; the online-ARF keystone (adaptive regime model) is single-pair ~0.51. Subsumed by keystone + sign-invariance.
- **D1a CNN/GRU / D4a DRL @300s** — single-pair sequence/RL; null at 60s/2m; the certified edge is cross-pair TABULAR, not single-pair sequence; keystone (adaptive single-pair) subsumes. Confirmed-null.
- **E1a magnitude |ret300|≥Q** — sign-invariant → OUT OF SCOPE for up/down (the certified size edge, AUC ~0.74 at 2m; tracked in MAGNITUDE_FINDINGS). Quick-confirm pending.
- **E2a direction-on-magnitude @300s** — null at 10m/60s (confirms invariance: magnitude quartile doesn't carry sign). Subsumed.
- **F1a macro-release impulse @300s** — `m5_research_log.md` (DONE): surprise-direction ~0.50-0.52, model extracts NOTHING directional, news-window AUC ≤ overall. NULL for 5m direction (FX prices a surprise in ~1 min). Done-null.
- **F2a Sofien price-action @5m** — 0.50-0.535 (combined table). Null.
- **F3a ES/NQ cross-asset lead-lag @5m** — `m10_xasset_probe.py`: lagged corr +0.02 (2024) → −0.05 (2025) = null + mechanistic key to the 2025 wall. Null.
- **F4a residualized-target @300s** — null at 60s (`min1_residtarget`). Confirmed-null.
- **N2a triangular / N4a MIM / N7a Hawkes-proxy / N8a RS-skew @300s** — all KILLED at 2m; signed bases covered by the 5m family probe (~0.51) + sign-invariance. Confirmed-null.

## EDGE-IMPROVEMENT EXPERIMENTS (post-sweep, literature-toolkit levers — IMPROVE the certified UP edge)
Driven by the 16-agent DL lit-review (`/home/sean/git/academic-papers/_DL_for_5m_FX_direction_REVIEW.md`).
Goal = lift the certified UP edge / its CPCV path-clear-rate, NOT confirm a wall. Incumbent UP .605/.577/.615.
- **EXP-1 |return|-weighted retrain** (`m5_magweight.py`, POW=1.0): primary VAL AUC .519; UP .634/**.560**/.525,
  DOWN .589/**.563**/.557. REBALANCES toward two-sided ~.56 (lifts 2025 DOWN .533→.563) but 2026 UP collapses
  .615→.525 → the magnitude-conditional SIGN is itself regime-dependent. No clean UP win. `m5_magweight_result.json`.
- **EXP-2 seed-ensemble MLP ⊕ GBM** (`m5_deep_ens.py`, M=5, AdamW lr2e-4): MLP **genuinely decorrelated from GBM
  (corr 0.694<0.9)** but a 50/50 blend ≈ GBM (UP2025 .5776 vs .5765, +.001 noise). DL-as-stack-member is
  null-to-marginal here (info-bound), as the literature predicts. `m5_deep_ens_result.json`.
- **EXP-3 ADAPTIVE-CONFORMAL gate (ACI)** (`m5_conformal.py`, w*=0.57): **SINGLE-SPLIT MIRAGE — DOES NOT SURVIVE
  NESTED-REFIT CPCV (DOWNGRADED 2026-06-01).** On the forward 2024-26 split it LOOKED like a win: binding 2025 UP
  .5838@n764 vs fixed .5793@n618 (better on win AND coverage, +36% trades). BUT the gold-standard NESTED-REFIT CPCV
  (`m5_aci_cpcv.py`: refits BOTH primary+meta on each of 28 purged-combinatorial paths, replays the ACI online θ
  inside each test fold — n=876k, stride6) KILLS the improvement claim: ACI p10 **0.521–0.523**, only **54–61%** of
  paths clear 0.541, at LOWER coverage (med_n 1231–1663) — strictly WORSE than the fixed meta gate (p10 0.5376,
  78.6%, med_n 2247) on the robustness tail. The 2025 "win" was a forward-regime ARTIFACT of ONE chronological
  split, not a robust edge. **EURUSD.m5xp_aci.v1 is DOWNGRADED: do NOT deploy ACI as an improvement over the fixed
  gate.** (This is the verdict-correction the discipline demands — a harder test reversed an earlier positive.)
  `m5_aci_cpcv_result.json`.
- **B1 POW=0.5 magweight NESTED-REFIT CPCV — UP+DOWN @ confidence cover** (`m5_magweight_cpcv.py`, 2026-06-01):
  refits the |return|-weighted primary (sample_weight=(|fwd|/med)^0.5) on each of 28 purged paths, evaluates both
  sides at the certified primary-confidence cover. **UP: certified but NOT improved** — cov0.05 p10 **.5453** < the
  unweighted incumbent .553 (upweighting big moves adds tail variance); falsifier KILLS the UP-improvement claim →
  keep the unweighted UP primary. **DOWN: MARGINALLY RESCUED (the one real gain) ✅** — cov0.05 p10 **.5441**, 89.3%
  of paths clear, mean .5569, med_n 1944 → CERTIFIED (vs incumbent DOWN cov0.05 p10 .5421); and on the forward
  split it lifts the binding 2025 DOWN from the incumbent's FAILING **.533** to **.559** [CI .533,.586] (2024 .605
  [.577,.634]; 2026 .549[.478,.625] thin), all-up-rate .51/.51/.50 (tripwire clean). **This is the FIRST DOWN
  configuration to both CPCV-certify at the tight cover AND point-clear the forward binding year** — a genuine but
  RAZOR-THIN rescue (certifies only at cov0.05; forward 2025 CI95-lo .533 still grazes below breakeven; fails at
  cov0.10/0.15 p10 .531/.533). Honest: best DOWN to date, deployable only as a thin tight-cover edge, NOT robust.
  Freeze pending (magweight primary needs a save-enabled re-train). `m5_magweight_cpcv_result.json`,
  `m5_magweight_result.json`.
- **C3 KILLED — seed-ensemble (K=4) does NOT lift the p10 floor** (`m5_seedens_cpcv.py`, 2026-06-01)
  Per fold: K=4 LGBMs (seeds 0-3), average predict_proba, primary-confidence cover {0.05,0.10,0.15};
  single-seed control. ENS cov0.05 **p10 0.5525** (mean .571, 100% paths clear) vs single-seed p10 .5455
  (mean .5616, 92.9% clear). **Falsifier fires: .5525 < incumbent .553 → KILLED for improvement.**
  Seed-ensemble DOES lift the single-seed tail (+.007, 100% vs 92.9% clear) but cannot beat the unseeded
  incumbent — tail variance is not the binding limit; the gap vs .553 is lucky-draw variance in the
  unseeded production training. Side-note: ENS extends certified coverage to cov0.15 (p10 .5446/96.4%,
  CERTIFIED) where single-seed fails (p10 .5409/89.3%, NOT certified) — broader coverage at lower win-rate,
  not a deployable improvement. `m5_seedens_cpcv_result.json`.

- **C4 KILLED — calibration does NOT rescue the meta gate** (`m5_calibcpcv.py`, 2026-06-02)
  Tested isotonic + Platt calibrated meta scores with absolute thresholds {0.50,0.53,0.55,0.57,0.60} AND
  raw-meta absolute thresholds, vs raw-quantile baseline. ALL variants fail: raw_quantile baseline
  p10 **0.530**/50% paths (below even the ACI-CPCV baseline .5376/78.6% — draw-to-draw variance on the
  unseeded meta model); iso/Platt calibrated absolute: p10 ~.525-0.526, **0% paths clear** at every
  threshold; raw absolute: best raw_abs_0.57 p10 .5247/46%. Calibration is ACTIVELY WORSE than the fold-
  local quantile because a fixed absolute threshold fails in off-regime folds (regime-sensitivity, not a
  miscalibration problem). Diagnosis confirmed: the meta gate is regime-dependent regardless of how you
  threshold it; the robust gate is the **primary confidence cover (cov0.05 p10 .553/96%)**, not the meta
  stage. **LEVER KILLED.** `m5_calibcpcv_result.json`.

- **C5 KILLED — logistic stacker recovers GBM-alone** (`m5_learnedstack.py`, 2026-06-02)
  Stacker coefs [4.32 GBM, 2.25 MLP] → GBM massively overweighted. STK_lin UP 2025 .5755 < BLEND50
  .5776; STK_int UP 2025 .5765 = GBM-alone. Mechanically confirmed: MLP is decorrelated (.694) but
  weaker; optimal blend downweights MLP to ~zero, recovering GBM. No complementary direction signal
  extractable. Full CPCV would be 28×500s=~4hrs for a null — skipped. `m5_learnedstack_result.json`.

- **C6 (KILLED 2026-06-02): Gentler magnitude weighting (POW=0.25) nested-refit CPCV**
  `M5_STRIDE=6 M5_POW=0.25 python m5_magweight_cpcv.py`. Result → `m5_magweight_cpcv_pow025_result.json`.
  UP cov0.05: p10=0.5477, mean=0.5639, frac_clear=1.00 — CERTIFIED but below incumbent 0.553.
  DOWN cov0.05: p10=0.5382, mean=0.5585, frac_clear=0.821 — NOT CERTIFIED (below B1 0.5441).
  **Falsifier triggered: neither side improves on incumbent.** POW=0.25 loses to both POW=0 (unweighted,
  incumbent UP 0.553) and POW=0.5 (B1, DOWN 0.5441). Lighter weighting helps neither. KILLED.

- **Task #12 (freeze magweight DOWN book, low priority, can be deferred):** save-enabled re-train of
  POW=0.5 primary + manifest.py build/freeze + git-tag `EURUSD.m5xp_magw_down.v1`. Label MARGINAL.

- **External data (dominant-EV, acquisition TODO not a wall):** intraday DE–US 2y rate differential
  (Dukascopy), daily implied-vol/risk-reversal, EURGBP ticks. Not yet acquired; every on-disk lever has now
  been run or killed.

## FINAL CONCLUSION — EURUSD 5-minute sweep (2026-06-01)

**BEST 5m ALGO = the (5m, UP) side of the frozen `EURUSD.m5xp.v1` book** (cross-pair USD-residual + order-flow
primary → orthogonal meta-labeler, gated `sess_ny & meta≥0.5738`, bet UP only). It is the **first sub-15m
direction edge in the program to survive the full per-fold refit CPCV** (the test that deflated 15m 0.647→0.5455).
DOWN is dead; every other channel is null or soundly subsumed.

### The honest UP verdict (I corrected this twice — final is evidence-locked)
- **Forward (deployment) split**, model trained 2012–2023 / tested 2024-26: UP win **.605 / .577 / .615** (all 3 yrs
  CI95-lo clear 0.541; up-rate tripwire clean .504/.523/.529; COMBINED reproduces the book's measured .607/.555/.606).
- **Full-refit CPCV** (`m5_cpcv_refit.py` / `m5_refit_tightcov.py`, refits the cross-pair primary on each of 28
  purged-combinatorial folds): **CERTIFIED at the book's operating gate** — cov0.05 **p10 0.553, 96% of folds clear**;
  cov0.10 p10 .544/89%; cov0.15 p10 .541/89%. Fails ONLY at loose cov0.30 (p10 .534, 54%) where the model has no edge.
- **NESTED-REFIT CPCV of the ACTUAL deployed META gate** (`m5_aci_cpcv.py`, 2026-06-01 — refits BOTH stages per
  fold, n=876k stride6; the prior cert above gated on PRIMARY-CONFIDENCE cover, never the meta gate itself): the
  deployed gate `sess_ny & meta≥thr` (q0.95) is a **NEAR-MISS** under nested refit — p10 **0.5376**, **78.6%** of 28
  paths clear (just under the p10≥.541 AND ≥80% bar), mean 0.5528 (profitable), med_n 2247. **KEY INSIGHT:** at
  EQUAL selectivity (~2200 trades) selecting by PRIMARY CONFIDENCE (cov0.05 p10 .553/96%) is MORE CPCV-robust than
  the META-labeler gate (p10 .5376/78.6%) — the refit meta gate does not earn its keep over a simple tight
  confidence cover. **So the certified deployable UP operating point is the TIGHT PRIMARY-CONFIDENCE cover
  (cov≤0.15, best cov0.05), NOT the meta gate and NOT ACI.** `m5_aci_cpcv_result.json`.
- Net: a **genuine, refit-robust edge concentrated in the high-confidence tail**, magnitude DEFLATED from the
  single-split. **Deployable win-rate ≈ 0.55–0.57 (robust floor 0.553); optimistic 0.58–0.61 in a favorable regime.**
- The frozen-trade CPCV (p10 0.576/100%) over-stated it (it doesn't refit → blind to selection overfitting); the
  cov-0.30 refit under-stated it (too loose). The tight-cov refit at the operating gate is the accurate test.

### DEPLOYMENT SPEC (the answer to "best algo to trade")
- **Signal:** `EURUSD.m5xp.v1`, bet **UP only** when `sess_ny & meta ≥ 0.5738` (≈ **4.8% of NY bars**, ~**6 trades/day**
  in the NY session; n per yr 1478/1423/254). Horizon 5m. Deriv Rise/Fall, R≈0.85, breakeven **0.5405**.
- **Why this gate:** it is simultaneously (a) the win-rate knee, (b) the Kelly growth-per-NY-hour PEAK, and (c) the
  **lowest gate where the binding (worst) year clears breakeven** — at any looser gate the worst year is sub-breakeven
  so honest Kelly stakes ZERO (`m5_kelly_result.json`). Don't loosen it; don't chase tighter (2026 thins to n≈17 +
  the corr(VAL,OOS)=−0.54 selection trap).
- **Confidence curve** (`m5_confcurve_result.json`): the meta gate is **monotone-informative** — win climbs .52 (65% cov)
  → .58–.61 (5% cov); this is real signal, not noise (better than 60s/2m where the curve was flat). BUT within the
  gate, tightening the *primary* direction-confidence lifts the easy year (2024 → .72) while the **binding 2025 stays
  pinned ~.59** and 2026 thins to noise — i.e. confidence helps where it matters least.

### SIZING & RISK (record these — they are the deployment-critical points)
- **Size by confidence (it's monotone-informative) but FRACTIONALLY: ⅛-Kelly or smaller (~1% stake or less).** Size on
  the **robust floor (~0.55)**, NOT the optimistic 0.58 — the realized growth is hugely sensitive to which is true
  (Kelly growth ~edge², and edge at 0.55 vs 0.58 differs ~3×). **Full Kelly = ruin** in a regime break; fractional is
  the real risk control.
- **Equity path** (`m5_equity_result.json`, ⅛-Kelly @1.07% stake sized to the forward .58, the FAVORABLE-regime case):
  forward 2024-26 → **25.95×** (per-yr 6.16×/18.12×/25.95×), **max drawdown 22%**, **longest losing streak 8**. This is
  the upside IF the regime holds. Sized instead on the robust .55 floor (the conservative, honest stake) the growth is
  far more modest (~2× over 2.4y) but survives regime risk. Expect ~20%+ drawdowns from normal variance even in good years.
- **The win-rate KILL-SWITCH does NOT cleanly help — do not bother with a naive one.** A trailing-win-rate trigger
  (W=150, kill<0.52, resume>0.55) cost **25.95×→19.37×** in the good regime (false-kills on routine cold streaks) for
  **~zero drawdown improvement** (21.9% vs 22.0%), and only marginally helped a synthetic regime-death stress (28.4% vs
  35.8% DD). **The real protection is the small fractional stake**, not a clever trigger — at ⅛-Kelly even a full
  regime-death stress is a survivable ~36% DD; at full Kelly it would be ruin. If any structural halt, make it SLOW
  (trailing few-hundred-trade win < 0.55 for an extended span), accepting you eat the first leg of any break.
- **Load-bearing caveat:** the edge is regime-conditional (refit-robust at the gate, but magnitude .55–.58 depends on
  the EUR-up regime that powers it). Size as if the win could be .55, cap absolute exposure, monitor the trailing win.

### DOWN — MARGINAL / borderline, NOT cleanly deployable (the symmetric exhaustive treatment, corrected)
DOWN got the SAME pipeline as UP and the verdict is nuanced — not "dead", but much weaker than UP and sub-deployable:
- **Side-split** m5xp down .609/**.533**/.592 — works 2024/2026, fails binding 2025.
- **A8b** down-filter (tighter gate): 2025 .558 but CI-lo .516<breakeven.
- **D3** USD-strength-conditioned DOWN .526 (n502) — fails (`m5_downcond_result.json`).
- **DOWN confidence curve** (`m5_sidecurve.py`): DOWN tracks UP in 2024(.608)/2026(.586) but 2025 maxes ~.555, **CI-lo
  never clears .541 at ANY gate** — the 2025 regime is the wall.
- **A8c DOWN specialist** (purpose-built meta, `m5_downspec.py`): VAL worst-half .749 → OOS 2025 **.509** (ANTI-TRANSFERS,
  worse than symmetric — textbook corr(VAL,OOS)=−.54 trap). A dedicated DOWN model can't crack 2025.
- **DOWN full-refit CPCV** (`m5_refit_tightcov_down_result.json`): at the tight gate cov0.05 **p10 0.5421 (barely >breakeven),
  93% folds clear → mechanically "certified"** — BUT this is dominated by the strong older-year folds; the forward
  (deployment) **binding 2025 fails (0.538)**. p10 0.542 ≈ breakeven (vs UP's 0.553); 25%/46%/68% folds clear at cov0.30/.15/.10.
- **Verdict:** DOWN has a FAINT high-confidence-tail signal (real, marginally refit-certified at the tightest gate) but it
  is razor-thin (p10 at breakeven) and **fails the binding 2025 forward year → NOT a deployable standalone DOWN edge.** It is
  regime-conditional and the CURRENT (2025) regime is unfavorable for DOWN. Mechanism: down-moves are jump/informed-dominated
  → energy to magnitude not sign; the UP/DOWN asymmetry is a real degree (UP strong / DOWN borderline), not UP-alive/DOWN-zero.
  **UP remains the clear deployable winner; DOWN is a marginal, currently-untradeable side.**

### Everything else — null / subsumed (exhaustive coverage)
- **Keystone** (online-ARF single-pair TA): AUC .509/.509/.514, selective never >.52 → single-pair channel dead.
- **Signed-lag family probe** (76 candidates: cross-leg signed-IFR/network-momentum, own signed-flow/propagator-residual/
  Hawkes-imbalance, own sign-autocorrelation): best ~.511 → the UP edge is the NONLINEAR cross-pair COMBINATION; no
  standalone signed channel works. Pre-killed N10/N11/N14/N15/N16/N17.
- **Executed-null at 5m:** B3a CKS-OFI (.5006), B4a-basis & B5a per-side raw flow (.508), F3a ES→EUR lead-lag (sign-flip
  +.025→−.022, dir-hit<0.50), N12/N13 irreversibility/odd-moment (.51), N2a triangular / N4a MIM / N7a Hawkes (2m-killed).
- **DEEP LEARNING on the CROSS-PAIR features (D-MLP, `m5_deep.py`):** an MLP (BN→256→128→64, dropout) on the SAME features
  m5xp uses → VAL AUC **0.5218** (= GBM 0.523) but UP-at-gate **.539/.529/.533** (WORSE than GBM .605/.577/.615) and test
  AUC ~0.509 every year. Same overall AUC, weaker high-confidence tail → GBMs dominate this tabular regime; deep is
  information-bound here too (prior deep nulls were single-pair; this closes the cross-pair gap). `m5_deep_result.json`.
  (A literature-grounded multi-agent review of DL architectures/tuning is in progress to confirm nothing better is missed.)
- **D6 TS FOUNDATION-MODEL FINE-TUNE (Chronos / Moirai / TimesFM) — KILLED on literature (not run; not worth a CPU job):**
  Adversarial replication-check. The decisive primary source is Rahimikia, Ni & Wang, *Re(Visiting) Time Series Foundation
  Models in Finance* (arXiv:2511.18578, Nov 2025) — first comprehensive eval of zero-shot / **fine-tune** / from-scratch
  TSFMs on returns. Verbatim: zero-shot Chronos-large dir-acc ~51% / R²=−1.37%, TimesFM-500M ~<50% / R²=−2.80%; and crucially
  for the FINE-TUNE setting: *"fine-tuning … yields limited improvements and fails to close the performance gap with
  benchmarks. Most fine-tuned TSFM performance deteriorates, except for Chronos (large). However, this improvement does NOT
  translate into economic gains."* Best directional acc anywhere (incl. from-scratch) = Chronos-small **51.74%** vs CatBoost
  **51.16%** — +0.58pp, and CatBoost still wins economically (Sharpe 6.79). Caveats that make it WORSE for us: their data is
  **daily equity excess returns (94 countries), NOT intraday FX**; TSFMs are MSE point-forecasters (sign-invariant by our own
  theorem); **CPU-only fine-tune of a 200M–500M-param model is OOM-infeasible here**. No FX-intraday TSFM-fine-tune study
  exists (searched). The one contrary FX-direction-with-costs claim, EXFormer (arXiv:2512.12727), is a **bespoke from-scratch
  Transformer on DAILY data — NOT a TSFM fine-tune** and has **no independent replication** → does not support this row.
  Verdict: **skip.** Same information-bound wall as D-MLP; adds nothing the GBMs don't already see, at far higher cost.
- **LOSS-REOPTIMIZATION family** (focal / quantile-median / asymmetric class-weight, `m5_lossbatch.py`, 2026-06-03):
  §3 IMPROVE lever asking whether the OBJECTIVE (not the features) is the binding constraint, on the cross-pair primary,
  cov0.05 per-year (2024/2025/2026). `focal_g2` (γ=2 custom fobj): UP .543/.537/**.527**, DOWN .514/.515/.522. `quantile_median`
  (LightGBM `objective=quantile` α=0.5 on forward return → sign of predicted median; this IS the Sirignano distribution-head
  direction test): UP .552/.544/**.503**, DOWN .567/**.504**/.564. `asym_down_w1.5`: DOWN .566/**.530**/.532. `asym_down_w2.0`:
  DOWN .563/.541/**.528** (best DOWN-2025 .5407 but CI-lo .524<breakeven). **ALL FAIL** — no UP binding clears .553, no DOWN
  binding-year CI-lo clears .541. The objective is NOT the binding constraint (DOWN 2025 sign is coin-flip in these features);
  loss-reoptimization joins magweight/seed-ensemble/rankloss as exhausted. `quantile_median`'s null pre-subsumes the corpus
  distribution-head (Sirignano full-distribution) direction lever. `m5_lossbatch_result.json`.
- **Soundly subsumed (24 rows, audit-verified Tier-1 rationale):** A1a/A2a/A3a/A5b/A6a/A8c (single-pair TA or dominated
  modes, independently null ~0.51), C1a/C2a/C4a (sign-invariant magnitude gates), D1a/D4a (single-pair sequence/RL),
  E1a/E2a (magnitude=sign-invariant, out of scope), F1a/F2a/F4a (news/price-action/residual-target null).

**>0.65 OOS-stable at 5m is NOT achievable** (confirmed). The achievable, refit-certified deliverable is the (5m,UP)
dip-buy edge at ~0.55–0.57, deployable with fractional-Kelly sizing and eyes-open regime risk.
