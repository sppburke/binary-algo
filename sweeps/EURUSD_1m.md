> **SCOPE: EURUSD · 1m (60s)** (sweep LEDGER — resumable status). Backlog: sweeps/EURUSD_1m_backlog.md. Results: EURUSD_RESULTS.md (§ EURUSD × 60s). See REPO_MAP.md.

---
currency: EURUSD
timeframe: 1m (60s)
started: 2026-06-03 (formal ledger; the 1m channel was swept earlier — see min1_research_log.md + EXPERIMENT_LEDGER.md rows 27-55)
target: best UP and DOWN binary predictor at 60s, OOS(2026)-verified, clearing breakeven 0.541
status: CLOSED (2026-06-03) — both sides certified-or-exhausted. ~24 prior channels + 3 new DOWN levers (D3a/magweight/
        mag-bridge) + 12-candidate discovery round (DRY) all null/subsumed; online-ARF keystone proves efficiency.
        Faithful CPCV: NEITHER side certified (UP p10 .524 regime-filter near-miss; DOWN p10 .474 dead). No certified
        60s direction edge; UP best-available = regime-dependent filter; DOWN exhausted on-disk (external-data redirect).
incumbent/answer: UP = up-only filter (min1_updown.py) moved .520/.584/.613, ties-strict .573 — best AVAILABLE,
                  UNCERTIFIED (CPCV p10 .524). DOWN = dead/exhausted (.516; CPCV p10 .474). 60s not deriv-deployable (15m floor).
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
| E-bridge | E→D | **magnitude→direction bridge** (frozen mag+dir; does direction hide on large moves?) | min1_magdir.py | **killed** | ~.50-.51 all magq | DOWN FLAT ~.50: magq0.0 .500/magq0.7 **.502**/magq0.95 **.498** (2025) | textbook sign-invariance AT THE OPERATING POINT: magnitude perfectly selects big moves (magAUC .787) but they carry ZERO direction. Conditioning DOWN on predicted-larger moves does NOT raise acc in any year. The 60s direction edge does NOT hide on large moves | min1_magdir_result.json |
| CPCV | cert | **faithful CPCV of UP-filter + DOWN** (frozen-book, ties-strict, 28 purged-comb paths + block-boot) | min1_cpcv.py | **done** | UP pooled-strict .573, CI-lo **.530**, p10 **.524**, 75% clear, 2024 .520 — **NOT certified** | DOWN pooled .519, p10 .474, 29% clear — **NOT certified** | formal (e)-step both sides. UP = regime-dependent filter (near-miss, same as 2m .5445 FAIL), NOT a robust edge; DOWN dead. No certified 60s direction edge | min1_cpcv_result.json |
| barimg | N (discovered) | **BAR-IMAGE 2-D CNN** (Sezer CNN-BI; the last non-subsumed bar/candlestick sub-lever) — 3 image encodings: hist/ohlc/gaf | barcnn_bars.py · barcnn_run.py · barcnn_cpcv.py | **killed** | combined VAL dirAUC hist .4992 / ohlc .5028 / gaf .5020; test/oos AUC ≈.50 (not side-split — symmetric, ≈.50 carries no UP edge) | same (≈.50, no DOWN edge) | new 4th model class (2-D conv). CPCV 28 purged paths: **path_p10 .484–.499, 0.0 paths clear 0.541 at every cov**. Bar geometry = magnitude not 60s sign (even GADF antisym sign-field null). Trips all 3 falsifier conditions. **Bar/candlestick 2-D-image family RUN + EXHAUSTED on-disk.** | barcnn_{hist,ohlc,gaf}_result.json + barcnn_cpcv_*_result.json |

### Prior-subsumed at 60s (documented, not re-run)
- **Side-specialists** (UP/DOWN trained on subset bars): killed — subset-training destroys the confidence ranking
  (`min1_upspec.py`: UP-spec .498/.537/.519 < filter; DOWN-spec .472/.521/.481 dead). EURUSD_RESULTS.md§60s + EXP_LEDGER row 33/55.
- **Online-ARF keystone** (`min1_online.py`): a drift-adaptive forest finds ZERO 60s direction signal every year
  (AUC .503-.508) → 60s direction is genuine efficiency; subsumes retuned/gate-swept static ensembles.
- **Magnitude** is the certified edge (magAUC 0.787) but sign-invariant → out of scope for up/down (→ MAGNITUDE_FINDINGS.md).

## GOAL RESULT (1m) — BOTH SIDES CLOSED, improve+discover loops DRY (2026-06-03)
**No certified 60s EURUSD direction edge exists** (faithful CPCV, `min1_cpcv.py`). Both improve and discover loops are
dry: the |ret|-magweight improve lever is killed (best_iter=8), no UP-improvement lever attacks the structural 2024
regime wall, and TWO consecutive adversarial discovery rounds came back DRY (round 1: 12 microstructure-sign levers
subsumed; round 2: 8 cross-horizon/calendar/UP-cert/recent-lit candidates subsumed). Detail:
- **UP** = `EURUSD.min1.v1` up-only filter — best AVAILABLE 60s UP point estimate (moved .520/.584/.613; ties-strict
  pooled .573), but **NOT CPCV-certified**: block-boot CI-lo **.530**, 28-path **p10 .524**, 75% of paths clear,
  2024-strict **.520**<breakeven. A regime-dependent dip-buy filter (works in the EUR-up 2025-26 regime, washes in
  2024), structurally identical to the 2m UP that FAILED the same test (.5445). Not a robust edge; not deriv-forex-
  deployable (60s < deriv's 15m floor) — research / synthetic-index / tick-venue only.
- **DOWN** = **dead + honestly EXHAUSTED on-disk.** Symmetric down-preds .522/.522/.516; CPCV pooled .519/p10 .474.
  This session killed the 3 remaining on-disk DOWN levers — USD-driver conditioning (D3a), |ret|-magweight (B1a,
  best_iter=8), magnitude→direction bridge (sign-invariance proof) — and a 12-candidate adversarial discovery round
  came back DRY (all subsumed/sign-invariant). The online-ARF keystone proves 60s direction is genuine efficiency.
  Only remaining DOWN redirect = EXTERNAL data (option-implied risk-reversal sign [verified-paywalled], intraday
  DE-US rate differential) — acquisition prerequisite, not a modeling task.

**Deriv deployability note:** 60s is NOT a deriv forex horizon (Rise/Fall forex min expiry = 15m). The deployable
EURUSD direction books are at 15m (`EURUSD.m15xp.v1`: UP .5673 / DOWN .5742, both refit-CPCV-certified). The 1m
work is a research-horizon characterization.

**ADDENDUM 2026-06-05 — bar/candlestick 2-D-image CNN lever RUN. DIRECTION → KILLED; MAGNITUDE → clears >65%
CPCV-certified.** `/goal` asked specifically for the Sezer CNN-BI + Kronos bar-pattern approach. Built 1m OHLCV from
ticks (`barcnn_bars.py`, moved up-rate ∈ [.497,.503] every yr) and faithfully reimplemented Sezer CNN-BI
(`barcnn_run.py`; method in `METHODS_CATALOG.md` §5.5) on the 60s wc_ret label, 3 image encodings (close-histogram /
3ch-OHLC / GAF-GASF+GADF). **DIRECTION** all null: VAL dirAUC ≈ .50, test/oos AUC ≈ .50, faithful CPCV
(`barcnn_cpcv.py`, 28 purged paths) **path_p10 .484–.499, 0.0 paths clear 0.541 at every coverage** (incl.
regime-gated, `barcnn_regime.py`). Even the antisymmetric GADF sign-field is null → the bar-image lever is magnitude,
not ≤60s sign; 60s direction near-efficiency now holds across a 4th model class. **MAGNITUDE** (`barcnn_mag.py
ohlcabs`, the sign-invariant Touch/Range outcome, NOT a 60s direction key): the absolute-scale bar image hits magAUC
.699/.714/.686 and selective large-call precision .68→.80 with **all 28 CPCV paths ≥0.65 at cov≤0.2 in 2024/2025/2026**
(`barcnn_mag_ohlcabs_result.json`, `MAGNITUDE_FINDINGS.md` §3) — bar patterns DO predict move-SIZE at >65%, just not
sign. Kronos NOT built (RankIC/magnitude gains, no FX/60s/direction numbers; fine-tune deteriorates arXiv:2511.18578).
