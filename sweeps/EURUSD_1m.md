> **SCOPE: EURUSD · 1m (60s)** (sweep LEDGER — resumable status). Backlog: sweeps/EURUSD_1m_backlog.md. Results: EURUSD_RESULTS.md (§ EURUSD × 60s). See REPO_MAP.md.

---
currency: EURUSD
timeframe: 1m (60s)
started: 2026-06-03 (formal ledger; the 1m channel was swept earlier — see min1_research_log.md + EXPERIMENT_LEDGER.md rows 27-55)
target: best UP and DOWN binary predictor at 60s, OOS(2026)-verified, clearing breakeven 0.541
status: MATURE — ~24 direction channels already null at 60s (online-ARF keystone proves genuine efficiency). Best
        = EURUSD.min1.v1 · d8b2c32c — UP-FILTER 0.520/0.584/0.613 (regime-dependent, does NOT clear 0.65); DOWN dead.
incumbent/answer: UP = up-only filter on symmetric ensemble (min1_updown.py) OOS .613, floor .520. DOWN = dead (.516).
prior: 60s EURUSD direction is near-efficient (~0.50-0.51 AUC across every model class); >0.65 OOS-stable not
       achievable on this data. The ONLY forecastable thing at 60s is MAGNITUDE (magAUC 0.787 vs dirAUC 0.510).
---

# Sweep ledger — EURUSD 1-minute (60s) UP/DOWN

Each row: pre-register falsifier → retarget to 60s (`MX_HOR=1` for bar/cross-pair models, `HS=60` for tick) →
deriv-faithful discipline (wc_ret ties-LOSE, nonoverlap_chrono, per-year CI95, worst-VAL-half selection, moved-bars
up-rate∈[.47,.53]) → score combined + UP + DOWN → record → commit. `oos` columns = per-year 2024/2025/2026.

**The bulk of the 1m sweep predates this ledger** — 24+ channels (base ensemble, cross-horizon stack, cross-pair
USD-residual, Hurst/VR, HMM, online-ARF keystone, macro impulse, cross-impact OFI, CKS-OFI, Kalman, kernel-SVM, RMT,
Neural-CDE, DRL DQN/IQN, ordinal-irreversibility, residualized-target, CCM) are logged as rows 1-17 in
EURUSD_RESULTS.md § 60s and in EXPERIMENT_LEDGER.md rows 27-55 / min1_research_log.md. This ledger tracks NEW rows
from 2026-06-03 onward (the both-sides-symmetric DOWN/UP push).

| id | tier | method | script | status | up_oos (24/25/26) | down (24/25/26) | verdict | result_json |
|----|------|--------|--------|--------|-------------------|-----------------|---------|-------------|
| base | base | **min1 frozen book SIDE-SPLIT (the answer)** | min1_updown.py | **done** | **.520/.584/.613** (filter) | .522/.522/.516 | UP-filter best (floor .520, regime-dependent); DOWN dead all yrs | EURUSD_RESULTS.md§60s |
| D3a | N | **USD-strength-conditioned DOWN** (retarget of 5m m5_downcond) | min1_downcond.py | **killed** | — (UP mirror .506/.504/.499) | USD-strong **.500/.494/.497** | DOWN efficient even USD-gated; USD-strong≈USD-weak (no separation); coverage curve FALLS to .475(25)/.464(26) at top-2% USD-strong tail (microstructure mean-reversion) → STRENGTHENS 5m D3 kill (5m had some lift .526; 60s has none) | min1_downcond_result.json |
| B1a | I | **\|ret\|-weighted magweight retrain** POW=0.5 (retarget of 5m m5_magweight, the only lever that ever certified 5m DOWN) | min1_magweight.py | **killed** | cov.15 .45/**.522**/.495 (worse than filter) | cov.05 .513/**.507**/.512 (CI-lo<.50) | tick substrate, deriv-faithful. **best_iter=8/4000** = no learnable direction signal to weight. DOWN binding-2025 .507 < breakeven & < incumbent .522; UP worse than filter. 5m razor-thin DOWN edge does NOT transfer to more-efficient 60s | min1_magweight_result.json |

### Prior-subsumed at 60s (documented, not re-run)
- **Side-specialists** (UP/DOWN trained on subset bars): killed — subset-training destroys the confidence ranking
  (`min1_upspec.py`: UP-spec .498/.537/.519 < filter; DOWN-spec .472/.521/.481 dead). EURUSD_RESULTS.md§60s + EXP_LEDGER row 33/55.
- **Online-ARF keystone** (`min1_online.py`): a drift-adaptive forest finds ZERO 60s direction signal every year
  (AUC .503-.508) → 60s direction is genuine efficiency; subsumes retuned/gate-swept static ensembles.
- **Magnitude** is the certified edge (magAUC 0.787) but sign-invariant → out of scope for up/down (→ MAGNITUDE_FINDINGS.md).

## Running conclusion (1m)
**UP** = `EURUSD.min1.v1` up-only filter, OOS .613 / floor .520 — a genuine but regime-dependent improvement over
symmetric; does NOT clear 0.65; UNCERTIFIED-for-deriv (60s < deriv's 15m forex floor). **DOWN** = dead; the
USD-driver conditioning (the one mechanism-grounded DOWN lever) is now KILLED at 60s as it was at 5m. Remaining
on-disk DOWN levers to attack: |return|-weighted (POW=0.5) retrain (the only lever that gave 5m DOWN a razor-thin
.5441) and the magnitude→direction bridge. External (risk-reversal sign) is verified-paywalled. See backlog.
