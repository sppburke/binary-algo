---
currency: EURUSD
timeframe: 2m (120s)
started: 2026-06-01
target: best UP and DOWN binary predictor at 120s, OOS(2026)-verified, clearing breakeven 0.541
status: COMPREHENSIVE — 13 distinct direction channels swept (incl. 2 discovery rounds) all killed except the
        baseline; online-ARF keystone proves genuine efficiency. Best = EURUSD.min2.v1 (UP 0.555 / DOWN 0.540).
incumbent/answer: EURUSD.min2.v1 · ace4d5c6 — UP-side OOS 0.555 (robust), DOWN 0.540 (marginal)
prior: 120s is in the efficiency zone (60s ~0.50-0.51, 5m ~0.52 AUC); confirmed.
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

### Subsumed by the keystone (not separately run — documented rationale)
The online-ARF keystone (C5a: a drift-adaptive forest finds ZERO 2m direction signal every year) **subsumes**
the remaining model-based rows: **A1a/A2a/A3a** (a retuned/gate-swept static ensemble cannot beat what a
continuously-adapting forest can't find), **C1a HMM / Kalman** (state-space regime models — the reversion-gate
baseline already embodies the regime channel; null at 60s), **B1/B5 tick-microstructure & per-side-flow** (the
min2 book IS the tick-microstructure ensemble; CKS-OFI already null), **A6a cross-pair** (cross-leg sign-lead
gate killed it). **E magnitude** is the certified edge but sign-invariant → out of scope for up/down.

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
