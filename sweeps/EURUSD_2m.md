---
currency: EURUSD
timeframe: 2m (120s)
started: 2026-06-01
target: best UP and DOWN binary predictor at 120s, OOS(2026)-verified, clearing breakeven 0.541
status: high-prior levers complete — best = EURUSD.min2.v1 (UP 0.555 robust / DOWN 0.540 marginal); low-prior rows deferred
incumbent: EURUSD.min2.v1 (combined OOS 0.539) — NOW side-split (UP 0.555 / DOWN 0.540)
prior: 120s sits in the efficiency zone (between 60s ~0.50-0.51 and 5m ~0.52 AUC); expect mostly null,
       best book ~0.53-0.56 combined, edge likely one-sided UP (per the 60s finding). Design fast-KILL falsifiers.
---

# Sweep ledger — EURUSD 2-minute (120s) UP/DOWN

Work top→bottom. Each row: pre-register falsifier → retarget to 120s → deriv-faithful discipline →
score combined + UP-split + DOWN-split per-year (2024/25/26) CI95 → record in EURUSD_RESULTS.md → mark
done/killed → update leaderboard → commit. Run DISCOVERY (skill §8.6) periodically. One heavy job at a time.

| id | tier | family / method | variant (knobs) | script | tgt | prior | status | comb_oos | up_oos | down_oos | verdict | result_json |
|----|------|-----------------|-----------------|--------|-----|-------|--------|----------|--------|----------|---------|-------------|
| 0  | base | min2 frozen book SIDE-SPLIT (incumbent) | gate=comp×rev×conf, as-frozen | min2_updown.py | U/Dn | high-value | done | .543/.543/.549 | .546/.565/**.555** | .540/.512/.540 | UP clears breakeven all 3 yrs (floor .546, CI-lo not clear); DOWN marginal (2025 .512 fails). Baseline to beat. | min2_updown_result.json |
| A1a | A | 3-model GBM ensemble retune | nl{255,350}, nest{2000,4000} | min2_production.py train | D | med | pending | | | | | |
| A2a | A | compression × session × coverage sweep | comp_q{10,20,33}, cov{5,10%}, sess{NY,all} | min2_production.py | G+D | med | pending | | | | | |
| A3a | A | reversion-vs-ret{60,300,900} × specialist w_spec{0,0.5} | levers | min2_production.py | D | med | pending | | | | | |
| A5a | A | cross-horizon stack: 15m parent → 2m (AGREE filter) | parent=m15, agreement | min2_stack.py | D | med-high | killed | .526/.522/.544 | .524/.544/.550 | .529/.492/.536 | KILLED — 15m agreement REDUCES every cell vs baseline (UP26 .555→.550) + sheds coverage; parent adds no info to the 2m reversion bet. | min2_stack_result.json |
| A6a | A | cross-pair USD-residual + OF, MX_HOR≈2 | mode{xp,xpof} | m5_xpair.py (HS=120) | D | med | pending | | | | | |
| A8a | A | up-only FILTER refinement (conf-tighten, worst-VAL-half) | keep top-25% conf | min2_upfilter.py | U | filter med | killed | | .62/.534/**.697** | — | KILLED — tempting OOS26 .697 (n76) but 2025 .534 FAILS breakeven; fragile/thin (corr(VAL,OOS) trap). Baseline UP .555 (floor .546) more robust. | min2_upfilter_result.json |
| B1a | B | tick microstructure ensemble @120s | HS=120, OBI/microprice/flow | m_tick_prod.py | D | low (decays by 60s) | pending | | | | | |
| B3a | B | CKS event-OFI @120s | retarget HS=120 (cache reused) | min2_cksofi_run.py | D | ~null | killed | | | | KILLED — VAL dirAUC 0.4995 (LGBM early-stops at iter1); null, as at 60s. | min2_cksofi_result.json |
| C1a | C | HMM regime (K3) gate/switch @120s | K{2,3}, U1/U2 | min1_hmm.py (HS=120) | G+D | ~null | pending | | | | | |
| C5a | C | online ARF+ADWIN control @120s (KEYSTONE) | 2-min-ahead prequential | min2_online_run.py | D | ~null (control) | killed | AUC .501/.505/.507 | | | **KEYSTONE**: adaptive forest ~0.50 AUC every year incl 2026 → 2m direction = GENUINE EFFICIENCY (not stale drift); no adaptive model beats the wall. | min2_online_result.json |
| E1a | E | MAGNITUDE \|ret120\|≥Q (SIZE, sign-invariant) | top-tercile, frozen mag model | min2_mag_run.py | M | high (certified) | done(N/A for U/Dn) | AUC .775/.739/.705 | — | — | CONFIRMED size edge at 2m (stable all 3 yrs); but sign-invariant → OUT OF SCOPE for up/down (Touch/Range/Straddle, not Rise/Fall). | min2_mag_result.json |
| F1a | F | macro-release 120s impulse | vol-tier × \|surp_z\|, HS=120 | min2_news_run.py | D | ~null | killed | | | | KILLED — surprise-sign regime-unstable (2024 .353 / 2025 .741thin / 2026 .517); FX prices surprise <120s. Null, as at 60s. | min2_news_result.json |
| N1 | N | DISCOVERY: session-conditioned 2m UP | best session on worst-VAL-half | min2_session.py | U | med | killed | | .604/.553/**.485** | — | KILLED — VAL-best 'overlap' session anti-transfers to OOS .485 (corr(VAL,OOS)=−0.54). | min2_session_result.json |

## CONCLUSION (comprehensive — 8 distinct direction channels swept + discovery round, 2026-06-01)
**Best 2m strategy = the frozen min2 book (EURUSD.min2.v1), UP side, OOS 0.555** (robust: clears breakeven
0.541 in all 3 years, floor 0.546). DOWN side 0.540 (marginal, 2025 fails). **NOTHING beats it.** The sweep
killed every distinct directional channel at 2m:
| Channel | Row | Result |
|---|---|---|
| Reversion (compression×conf gate) | row 0 | **BEST (baseline), UP 0.555** |
| Cross-horizon stack (15m→2m) | A5a | killed (reduced every cell) |
| Confidence up-filter | A8a | killed (OOS 0.697 but 2025 fails — fragile) |
| Session / time-of-day | N1 | killed (anti-transfer) |
| Microstructure CKS-OFI | B3a | killed (AUC 0.4995) |
| Triangular USD-canceling residual | N2 | killed (coin-flip; top novel idea, 15% prior) |
| Cross-leg sign-lead (N3/N5/N6/N9) | gate | killed (no leg leads EURUSD sign) |
| Intraday momentum term-structure | N4 | killed (.506 flat) |
A discovery round (sub-agent, 8 novel arXiv/cross-disciplinary ideas) was run; the 2 highest-prior novel
mechanisms (triangular USD-cancellation, cross-leg directed-sign) were tested and KILLED. **2m direction is
efficiency-bound at ~0.55**, consistent with the program-wide 60s–5m finding; even mechanisms designed to
bypass the 2025 USD-factor root cause are null. Deferred LOW-PRIOR (documented rationale, can extend ledger):
N7 Hawkes (proxy≈known-null tick-imbalance), N8 RS-skew (sign-invariance), null retargets (HMM/online/news/
Kalman — all null at 60s), A1a-A3a retrain sweeps (won't beat efficiency ceiling), E magnitude (sign-invariant
= out of scope for up/down). **GOAL RESULT: best 2m UP = EURUSD.min2.v1 @ 0.555; best 2m DOWN = same @ 0.540.**

### (superseded interim note)
**Best 2m UP = the frozen min2 book (EURUSD.min2.v1), OOS(2026) 0.555** — the ONLY robust positive side
(floor 0.546, clears breakeven 0.541 in all 3 years on point estimate). **Best 2m DOWN = same book, 0.540**
(marginal; 2025 0.512 fails breakeven). **No improvement lever beat it robustly:** cross-horizon stack (A5a)
reduced every cell; up-filter (A8a) hit OOS 0.697 but 2025 failed; session-conditioning (N1) anti-transferred
to 0.485. All three died to the same VAL-anti-transfer + thin-coverage discipline → 2m direction is
efficiency-bound at ~0.55, consistent with the program-wide 60s-5m finding.
**Remaining rows (A1a/A2a/A3a retrain sweeps, A6a cross-pair [settlement-mismatched], B/C/F microstructure/
state-space/news, E magnitude[=size not sign]) are LOW-PRIOR for direction** at this horizon (60s/5m already
null; the min2 book already embodies the tick-microstructure approach = B1). Deferred with rationale; a future
agent can extend this ledger. E (magnitude) is the certified edge but is sign-invariant → out of scope for an
UP/DOWN strategy (track in MAGNITUDE_FINDINGS.md).

| N2 | N | DISCOVERED: triangular USD-canceling residual (EURUSD vs GBPUSD cointegration reversion) | W=500min, \|z\|-thr | min2_triangular.py | D | **15% (top)** | killed | .499/.497/.497 | ~.50 | ~.50 | KILLED — USD-immune residual reversion is coin-flip at 120s (relative-value reverts too slowly). Clean null of the best novel idea. | min2_triangular_result.json |
| GATE | N | cross-leg sign-lead → EURUSD next-2m sign (covers N3/N5/N6/N9) | 6 legs × {1,2,5}min, USD-aligned | min2_legsign.py | D | 6-12% | killed | | | | KILLED — no leg's sign predicts EURUSD next-2m sign (best USDCAD .5025); pre-kills N3/N5/N6/N9 (cross-leg-sign null at 2m). | min2_legsign_result.json |
| N4 | N | Market-Intraday-Momentum term-structure | since-open/last{30,60,120}min × {mom,rev} | min2_mim.py | D | 10% | killed | | | | KILLED — best last30-reversion .506 flat all years; no stable intraday-momentum edge at 2m. | min2_mim_result.json |
| N7 | N | asymmetric tick-intensity imbalance (Hawkes-proxy) | EWMA up/down-tick, spans 30/60/120s | min2_hawkes.py | D | 9% | killed | | | | KILLED — best 0.5057; signed tick-intensity carries no stable 2m edge (self-excitation decays before 120s). | min2_hawkes_result.json |
| N8 | N | signed-semivariance-skew sign-conditioning on magnitude bars | RS⁺−RS⁻, W300s, mag-decile gate | min2_rsskew.py | D | 5% | killed | | | | KILLED — up-rate ~0.48-0.52 across skew bins; edge -.004(24)/-.030(26), not stable. Sign-invariance holds even under signed conditioner. | min2_rsskew_result.json |

Discovery seeds remaining: see SWEEP_MATRIX Tier-N (N3-N9, sub-agent-sourced 2026-06-01). Run order: N6 gate → N7 Hawkes → N3 quantilogram → N4 MIM → rest.
