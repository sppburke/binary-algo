---
currency: EURUSD
timeframe: 5m (300s)
started: 2026-06-01
target: best UP and DOWN binary predictor at 300s, OOS(2026)-verified, clearing breakeven 0.541
status: IN PROGRESS — row 0 done. **BREAKTHROUGH: (5m,UP) CERTIFIED** via m5xp side-split (binding 2025 0.577,
        all 3 yrs CI95-lo clear 0.541, CPCV 28/28 paths clear p10 0.576 — survives the test that killed 2m).
        First robustly-certified sub-15m direction key besides the seconds tick venue. DOWN dead (2025 0.533).
        Now sweeping Tier A→F + N to (a) try to BEAT m5xp UP, (b) find a DOWN edge, (c) confirm nulls.
incumbent/answer: **UP LEADER = EURUSD.m5xp.v1 up-preds (binding 2025 0.577, CERTIFIED).** DOWN = none certified
        (m5xp down 2025 0.533 dead; m5stack down tripwire-cautioned). Combined incumbent m5xp oos26 0.606.
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
| 0a | base | **m5xp frozen book SIDE-SPLIT** (cross-pair+OF+meta) | m5_updown.py | **done ✅** | **.605/.577/.615** | .609/**.533**/.592 | **CERTIFIED UP** (LEADER): all 3 yrs CI95-lo clear 0.541 (binding 2025 .577[.552,.603], n1379); up-rate clean .504/.523/.529; COMBINED reproduces book 0.607/0.555/0.606 exactly. **CPCV: p10 .576, 28/28 paths clear, block-boot CI-lo .575 → CERTIFIED** (2m failed same test). DOWN dead 2025 (.533<.541). | m5_updown_result.json, m5_cpcv_m5xp_result.json |
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
| A8c | A | side SPECIALIST (separately-trained up/down) | m5_upspec.py | pending | — | — | — | — |
| B1a | B | tick microstructure ensemble retarget @300s | m_tick_prod.py (HS=300) | pending | — | — | — | — |
| B3a | B | CKS event-OFI @300s | min2_cksofi (MX_HOR=5) | pending | — | — | — | — |
| B4a | B | cross-impact OFI matrix @300s | min1_xofi (MX_HOR=5) | pending | — | — | — | — |
| B5a | B | per-side raw signed flow @300s | _adj_perside_flow.py | pending | — | — | — | — |
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
| N12 | N | directed-HVG irreversibility sign-decomposed (peak/trough) @300s (~6%) | m5_dhvg.py | pending | — | — | — | — |
| N13 | N | time-reversal signed structure-function asymmetry @300s (~5%) | m5_trasf.py | pending | — | — | — | — |

(Discovery round 1 — cross-disciplinary agent, 2026-06-01: added N10-N14. Top priors N10 Liang signed IFR (the
one explicitly-SIGNED causality measure untried; CCM/TE were its unsigned cousins, both dead) and N11 Bacry-Muzy
cross-kernel (mean-reverting impact kernel at 300s relaxation, distinct from killed scalar N7). N12-14 = completeness.
arXiv/SSRN agent round pending. Discovery continues until K=2 dry rounds.)

## Notes / running conclusion
- (to be filled as rows complete)
