---
currency: EURUSD
timeframe: 2m (120s)
started: 2026-06-01
target: best UP and DOWN binary predictor at 120s, OOS(2026)-verified, clearing breakeven 0.541
status: running
incumbent: EURUSD.min2.v1 (combined OOS 0.539) — not yet side-split
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
| A8a | A | up/down FILTER vs SPECIALIST at 120s | side×{filter,specialist} | min2_updown.py / upspec | U/Dn | filter med | pending | | | | | |
| B1a | B | tick microstructure ensemble @120s | HS=120, OBI/microprice/flow | m_tick_prod.py | D | low (decays by 60s) | pending | | | | | |
| B3a | B | CKS event-OFI @120s | window{30,60,120}s | min1_cksofi.py (HS=120) | D | ~null | pending | | | | | |
| C1a | C | HMM regime (K3) gate/switch @120s | K{2,3}, U1/U2 | min1_hmm.py (HS=120) | G+D | ~null | pending | | | | | |
| C5a | C | online ARF+ADWIN control @120s | 10-tree | min1_online.py (HS=120) | D | ~null (control) | pending | | | | | |
| E1a | E | MAGNITUDE \|ret120\|≥Q | Q{0.75}, rv-windows | min2_v1.py | M | high (certified family) | pending | | | | | |
| F1a | F | macro-release 120s impulse | vol-tier × \|surp_z\| | min1_news60.py (HS=120) | D | ~null | pending | | | | | |
| N… | N | (discovered — append) | | | | | pending | | | | | |

Discovery seeds to vet/add: magnitude-conditioned 5m→2m stack · HMM-gated reversion · transfer-entropy
coupling gate · Hawkes up/down arrival imbalance · RL(IQN+CVaR) sizing on the 2m book.
