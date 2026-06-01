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
| B3a | B | CKS event-OFI @120s | window{30,60,120}s | min1_cksofi.py (HS=120) | D | ~null | pending | | | | | |
| C1a | C | HMM regime (K3) gate/switch @120s | K{2,3}, U1/U2 | min1_hmm.py (HS=120) | G+D | ~null | pending | | | | | |
| C5a | C | online ARF+ADWIN control @120s | 10-tree | min1_online.py (HS=120) | D | ~null (control) | pending | | | | | |
| E1a | E | MAGNITUDE \|ret120\|≥Q | Q{0.75}, rv-windows | min2_v1.py | M | high (certified family) | pending | | | | | |
| F1a | F | macro-release 120s impulse | vol-tier × \|surp_z\| | min1_news60.py (HS=120) | D | ~null | pending | | | | | |
| N1 | N | DISCOVERY: session-conditioned 2m UP | best session on worst-VAL-half | min2_session.py | U | med | killed | | .604/.553/**.485** | — | KILLED — VAL-best 'overlap' session anti-transfers to OOS .485 (corr(VAL,OOS)=−0.54). | min2_session_result.json |

## CONCLUSION (interim — high-prior levers exhausted 2026-06-01)
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

Discovery seeds to vet/add: magnitude-conditioned 5m→2m stack · HMM-gated reversion · transfer-entropy
coupling gate · Hawkes up/down arrival imbalance · RL(IQN+CVaR) sizing on the 2m book.
