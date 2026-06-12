> **SCOPE: EURUSD · 2m** (sweep LEDGER — resumable status). Backlog: sweeps/EURUSD_2m_backlog.md. Results: results/EURUSD_RESULTS.md. See REPO_MAP.md.

---
currency: EURUSD
timeframe: 2m (120s)
started: 2026-06-01
target: best UP and DOWN binary predictor at 120s, OOS(2026)-verified, clearing breakeven 0.541
status: DONE 2026-06-04 — full (a)-(e) pipeline run symmetrically BOTH sides; both honestly EXHAUSTED on-disk.
        Keystone gap closed (cross-pair pooling RUN+KILLED refit-CPCV A6c); RFF + specialist-nested-CPCV also killed;
        discovery K=2 dry; Tier-G external queued. NO certified 2m direction edge (4 model classes ~.50-.52 AUC).
incumbent/answer: NO certified edge. Best point estimate EURUSD.min2.v1 UP 0.555 / 0.5445 ties-strict (uncertified);
        DOWN dead. Cross-pair refit-CPCV p10 UP .5096 / DOWN .5124 (0-7% paths clear) — efficiency-bound confirmed.
prior: 120s is in the efficiency zone (60s ~0.50-0.51, 5m ~0.52 AUC); cross-pair edge gradient none@60s→UP@5m→both@15m
       does NOT reach back to 2m. Confirmed by faithful refit-CPCV.
---

# Sweep ledger — EURUSD 2-minute (120s) UP/DOWN

Each row: pre-register falsifier → retarget to 120s → deriv-faithful discipline (wc_ret ties-LOSE,
nonoverlap_chrono, per-year CI95, worst-VAL-half selection, moved-bars up-rate∈[.47,.53]) → score
combined + UP + DOWN → record → commit. `OOS` columns = per-year 2024/2025/2026.

| id | tier | method | script | status | up_oos (24/25/26) | down (24/25/26) | verdict | result_json |
|----|------|--------|--------|--------|-------------------|-----------------|---------|-------------|
| 0 | base | **min2 frozen book SIDE-SPLIT (the answer)** | min2_updown.py | **done** | **.546/.565/.555** | .540/.512/.540 | **BEST.** UP clears breakeven all 3 yrs (floor .546); DOWN marginal (2025 fails). | min2_updown_result.json |
| A5a | A | cross-horizon stack 15m→2m (agree) | min2_stack.py | killed | .524/.544/.550 | .529/.492/.536 | agreement REDUCES every cell + sheds coverage; 15m adds no info to the 2m reversion bet | min2_stack_result.json |
| A8a | A | up-only filter refinement (conf-tighten) | min2_upfilter.py | killed | .62/**.534**/.697 | — | OOS26 .697 tempting but 2025 .534 FAILS breakeven — fragile/thin (corr(VAL,OOS) trap) | min2_upfilter_result.json |
| B3a | B | CKS event-OFI @120s | min2_cksofi_run.py | killed | VAL dirAUC 0.4995 | — | LGBM early-stops iter1; null as at 60s | min2_cksofi_result.json |
| C5a | C | **online ARF+ADWIN control @120s (KEYSTONE)** | min2_online_run.py | killed | AUC .501/.505/.507 | — | **adaptive forest ~0.50 AUC every year → 2m direction = GENUINE EFFICIENCY; no model beats the wall** | min2_online_result.json |
| F1a | F | macro-release impulse @120s | min2_news_run.py | killed | surprise-sign .353/.741t/.517 | — | regime-unstable (2024 worse than coin-flip); FX prices surprise <120s | min2_news_result.json |
| E1a | E | magnitude \|ret120\|≥Q (SIZE) | min2_mag_run.py | done(N/A) | AUC .775/.739/.705 | — | CONFIRMED size edge, but sign-invariant → OUT OF SCOPE for up/down (Touch/Range/Straddle) | min2_mag_result.json |
| N1 | N | session-conditioned UP | min2_session.py | killed | .604/.553/**.485** | — | VAL-best 'overlap' anti-transfers to OOS .485 | min2_session_result.json |
| N2 | N | **triangular USD-canceling residual** (top novel, 15%) | min2_triangular.py | killed | ~.50 | ~.50 | COMB .499/.497/.497 coin-flip; USD-immune residual reverts too slowly for 120s | min2_triangular_result.json |
| N3/5/6/9 | N | cross-leg sign-lead family (quantilogram/PCMCI/directed-info/cross-ordinal) | min2_legsign.py | killed | best leg .5025 | — | no USD-leg's sign predicts EURUSD next-2m sign → pre-kills the whole family | min2_legsign_result.json |
| N4 | N | intraday-momentum term-structure | min2_mim.py | killed | best .506 flat | — | no signed intraday-interval predictor stable across 2024&2026 | min2_mim_result.json |
| N7 | N | asymmetric tick-intensity (Hawkes-proxy) | min2_hawkes.py | killed | best .5057 | — | tick self-excitation decays before 120s | min2_hawkes_result.json |
| N8 | N | signed-semivariance-skew sign-conditioning | min2_rsskew.py | killed | up-rate ~.48-.52 | — | sign-invariance holds even under signed conditioner | min2_rsskew_result.json |
| D2 | N | Stoikov micro-vs-mid slope-divergence (disc.round-2 top, 8%) | min2_stoikov.py | killed | VAL dirAUC 0.4984 | — | signed fair-value lead decays before 120s (as tick-intensity/rawtick did) | min2_stoikov_result.json |
| A6c | A | **cross-pair POOLING primary + TIES-STRICT refit-CPCV** (the keystone lever that certified 5m UP & 15m both) | min2_xpair_probe.py / min2_xpair_cpcv.py | **killed** | refit p10 **.5096**, max .544, 1/15 paths clear | refit p10 **.5124**, 0/15 paths clear | **KILLED under the §4 standard.** Probe's frozen-VAL gate gave .61/.56 (2024/25) — an OVERFIT-GATE artifact that DISSOLVED when the gate is re-tuned per fold (15 purged paths, all well-powered n≥3.9k, up-rates balanced [.482,.524], moved-AUC .517). The slow USD-common-factor signal needs ≥5m; at 2m it's swamped. Properly RUN, not subsumed. | min2_xpair_cpcv_result.json |
| I-RFF | I | RFF virtue-of-complexity SDF on xpof features (4th model class) | min2_rff.py | **killed** | VAL AUC **.4997** (coin-flip, all gamma×ridge .4985-.4997) | — | **4th model class (random nonlinear basis) finds ZERO 2m sign signal** → channel genuinely empty, not a GBM-capacity artifact | min2_rff_result.json |
| C-spec | c | purpose-built UP/DOWN meta-labeler SPECIALIST (step c), FROZEN | min2_spec.py | done | UP frozen .575/.5525/.6009 (SURVIVES→CPCV) | DOWN .566/.526/.471 (KILL) | UP frozen-VAL meta-gate clears .545 all 3 yrs — but same shape as the probe's overfit-gate; escalated to nested-refit CPCV. DOWN killed outright. | min2_spec_result.json |
| C-spec-cpcv | c/e | UP specialist NESTED-REFIT CPCV (primary+meta refit/fold) | min2_spec_cpcv.py | **killed** | p10 **.4957**, 0/15 paths (mean .5024) | p10 **.4912**, 0/15 paths | **The UP frozen .575/.5525/.6009 was an OVERFIT-GATE artifact — collapses to ~.50 (below!) when the meta-gate is refit per fold.** Same dissolution as the probe→cross-pair-CPCV. Specialist step (c) RUN + KILLED both sides. | min2_spec_cpcv_result.json |
| NS1 | N | N-BEATS / N-HiTS path-forecast→sign (2026-06-08 neural+spectral sweep) | nbeats_nhits_dir.py | **killed** | best selacc **.5123** (2025) | — | valAUC ≤.508; path-forecaster carries move SIZE not SIGN (sign-invariance theorem) | nbeats_nhits_dir_2m_result.json |
| NS2 | N | DLinear / Autoformer / FEDformer-freq / TFT-quantile-fan decomposition→sign (2026-06-08 neural+spectral sweep) | decomp_dir.py | **killed** | best selacc **.5137** (dlinear) | — | decomp/freq forecasters below breakeven .541; size not sign | decomp_dir_2m_result.json |
| NS3 | N | causal DWT + SSA band-split→GBM/literal recombine→sign (2026-06-08 neural+spectral sweep) | spectral_dir.py | **killed** | best selacc **.524** (2026, CI-lo **.4904**) | — | flashiest cell's CI95-lower below breakeven → thin-coverage mirage; valAUC ≤.5155 | spectral_dir_2m_result.json |

### Subsumed by the keystone (not separately run — documented rationale)
The online-ARF keystone (C5a: a drift-adaptive forest finds ZERO 2m direction signal every year) **subsumes**
the remaining model-based rows: **A1a/A2a/A3a** (a retuned/gate-swept static ensemble cannot beat what a
continuously-adapting forest can't find), **C1a HMM / Kalman** (state-space regime models — the reversion-gate
baseline already embodies the regime channel; null at 60s), **B1/B5 tick-microstructure & per-side-flow** (the
min2 book IS the tick-microstructure ensemble; CKS-OFI already null). **E magnitude** is the certified edge but
sign-invariant → out of scope for up/down.
> **CORRECTION (2026-06-04):** the earlier "A6a cross-pair subsumed by cross-leg-sign-lead" note was WRONG —
> cross-leg sign-lead (`min2_legsign`, another pair's *sign* as a signal) is a DIFFERENT mechanism from cross-pair
> feature-row POOLING (one EURUSD-sign model trained on the pooled multi-pair feature rows, USD-factor removed).
> The pooling lever — which CERTIFIED 5m UP and BOTH 15m sides — was NEVER actually run at 2m. It has now been
> RUN (row A6c) under the full §4 refit-CPCV standard and KILLED. The subsumption is replaced by a real Tier-1 test.

## CONCLUSION — GOAL RESULT (corrected after adversarial verification + CPCV)
**There is NO robustly-certified 2-minute EURUSD direction edge.** The best point estimate is `EURUSD.min2.v1`
UP-side OOS 0.555 (moved-only) / **0.5445 ties-strict**, but a 3-agent adversarial audit + a purged-combinatorial
CPCV DOWNGRADED it from "robust" to **UNCERTIFIED**:
- **(EURUSD, 2m, UP) = ~0.55 point estimate, UNCERTIFIED ⚠** — no per-year CI95-lower clears 0.541
  (.486/.536/.508); pooled binomial p=0.053; CPCV path **p10 0.524**, block-boot CI-lo **0.521**, only **57% of
  28 paths clear 0.541**; per-year-strict 2024 **0.517** < breakeven; the UP side is `pred≡1` so acc≡up-rate-of-
  bet-up-bars (conditional **drift, not two-sided skill**); 2025 combined up-rate **0.534 breaches the mirage
  tripwire**. The structurally-identical 15m pipeline deflated 0.647→0.5455 (FAIL) under faithful CPCV.
- **(EURUSD, 2m, DOWN) = ~0.54, DEAD ❌** — sub-breakeven in ALL 3 years (.540/.512/.540).
- **Keystone:** online-ARF ~0.50 AUC every year → 2m direction is GENUINE EFFICIENCY (no model beats it).
(min2_cpcv_result.json; verify verdicts: leakage=clean, robustness=FAIL-major, rederive=clean.)

**13 distinct direction channels were swept** (reversion[best], cross-horizon stack, confidence filter,
session, CKS-OFI, news, intraday-momentum, triangular USD-canceling residual, cross-leg sign-lead family,
asymmetric tick-intensity, signed-semivariance-skew, online-adaptive) across Tiers A–F + **two discovery rounds**
(13 novel arXiv/cross-disciplinary ideas logged in IDEAS_LOG/SWEEP_MATRIX Tier-N; the highest-prior mechanisms
tested). **All killed except the baseline.** The online-ARF **keystone** proves the ~0.55 ceiling is genuine
market efficiency — no model, adaptive or static, finds a 2m direction edge. Even the novel mechanism designed
to bypass the 2025 USD-factor root cause (triangular cancellation) is a coin-flip. The only forecastable thing
at 2m is **magnitude** (AUC 0.74, size not sign). **2m direction is efficiency-bound; nothing beats min2 UP 0.555.**

## ★ 2026-06-04 RE-EXAMINATION — keystone gap closed, both sides honestly exhausted under the FULL (a)-(e) pipeline
Re-opened to run the goal's full symmetric pipeline. The prior closure had a real gap: the **cross-pair feature-row
POOLING** lever — the ONLY lever that CERTIFIED 5m UP and BOTH 15m sides — had never actually been run at 2m (it was
wrongly subsumed under the *different* cross-leg-sign-lead mechanism). Closed it + ran the remaining distinct shots:
- **A6c cross-pair pooling, TIES-STRICT refit-CPCV (15 purged paths, per-fold gate refit):** UP p10 **.5096** (1/15 clear),
  DOWN p10 **.5124** (0/15), COMBINED p10 .5116 (0/15), moved-AUC .517, up-rates balanced [.482,.524]. The probe's frozen-VAL
  gate gave .6126/.5603 — an OVERFIT-GATE artifact that dissolved under per-fold refit. **NOT CERTIFIED.** (`min2_xpair_cpcv_result.json`)
- **I-RFF virtue-of-complexity SDF (4th model class, P~T random nonlinear basis on the same xpof features):** best VAL AUC
  **.4997** (coin-flip). The 2m sign channel is empty across FOUR model classes (3-GBM ensemble, online-ARF, GBM, RFF) →
  capacity is not the constraint; the information isn't there. (`min2_rff_result.json`)
- **C-spec purpose-built UP/DOWN meta-labeler specialist (step c) + nested-refit CPCV:** DOWN frozen-killed; UP frozen-survived
  (.575/.5525/.6009) but **collapsed to p10 .4957 / .4912 (0/15 both sides) under nested refit** — overfit-gate, same as the probe.
- **mag×direction bridge:** already RUN at 120s (EXPERIMENT_LEDGER rows 41-42) — a magnitude gate on direction HURTS at 2m.
- **Discovery round 3** (corpus re-mine, 2050+550 levers): all on-disk direction candidates reduce to the cross-pair family
  (killed) or are subsumed (TMFG-graph, adversarial-gate, sign-restrictions, lambdarank). **K=2 dry.** External = Tier-G (gated).
- **Subsumed w/ Tier-1 cites:** base-book refit (its 239 features ⊂ the .51 xpof superset + tick-micro channels all null);
  |ret|-weight/GMADL (null @5m+15m, same family); seed-ens/calibration/ACI (AUC-preserving / nested-refit-killed @5m).

### FINAL GOAL RESULT (2026-06-04)
**NO certified 2m EURUSD direction edge exists — both sides honestly EXHAUSTED on-disk under the full (a)-(e) standard.**
- **(EURUSD, 2m, UP)** = best AVAILABLE `EURUSD.min2.v1` up-preds **0.555 point / 0.5445 ties-strict — UNCERTIFIED**. Every
  certification-grade test fails: frozen-book CPCV p10 .524; cross-pair refit p10 .5096; RFF .4997; specialist nested-refit .4957.
- **(EURUSD, 2m, DOWN)** = **DEAD** (~.52, sub-breakeven all years; cross-pair refit p10 .5124; specialist nested-refit .4912).
- **Keystone:** four model classes at ~.50-.52 AUC every year → 2m direction is GENUINE market efficiency. The cross-pair
  edge gradient (none@60s → UP@5m → both@15m) does NOT reach back to 2m: the slow USD-common-factor needs ≥5m to be exploitable.
- **Forecastable at 2m = magnitude only** (sign-invariant, AUC ~.68-.74; Touch/Range/Straddle, NOT Rise/Fall). Track in docs/MAGNITUDE_FINDINGS.md.
- **Redirect (not a wall):** external data — DE-US rate-diff / VIX-risk-reversal / GARCH-MIDAS / cross-asset leads (Tier-G, gated on user "go").
- **Venue:** 120s is BELOW deriv's 15m forex minimum → a research horizon, not directly deployable regardless.
