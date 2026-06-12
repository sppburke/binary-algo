> **SCOPE: EURUSD · 30m** (key-specific). Generic methods/ideas live in docs/METHODS_CATALOG.md / SWEEP_MATRIX.md / docs/IDEAS_LOG.md; cross-key theory in docs/THEORY.md. See REPO_MAP.md.

# 30-MINUTE EURUSD BINARY DIRECTION — RESEARCH LOG

**Goal:** a 30-minute EURUSD up/down model with **>75% prediction accuracy, OOS-verified** (2026 held out),
independent/tradeable (non-overlapping 30-min windows, chronological first-come — no look-ahead), reported with CI95.
30m is **deriv-tradeable** (above the 15m forex Rise/Fall floor). Settlement = mid-to-mid close-to-close,
ties lose, edge = payout deduction (breakeven ≈ 0.541 at ~15% deduction). NO spread.

## Methodology discipline (anti-bias — carried from the 2026-05 audit)
- Splits: **TRAIN 2012–2021 · VAL 2022–2023 · TEST 2024 & 2025 · OOS 2026**. Four independent eval windows.
- **corr(VAL_acc, OOS_acc) historically = −0.54** → VAL-max alone does NOT transfer. A claim only counts if it holds
  across ALL of {TEST24, TEST25, OOS26} with CI95 excluding breakeven. OOS is the FINAL confirmation, not a tuning knob.
- Every reported accuracy = **non-overlapping, chronological** independent trades (gap = 1800s), bootstrap CI95.
- I count how many configs I try (deflation awareness). I do not cherry-pick the max over a search.
- Labels: recomputed at HOR=30 1-min bars with strict wall-clock contiguity (no weekend/session gap). Ties dropped from
  training, counted as losses in backtest EV.

## Prior evidence I am building on (Tier-1, our own logs)
- **30m raw LightGBM (V8): OOS AUC 0.518** — noise floor. V25: AUC decays to ~0.50 by ≥89s horizon.
- **Best honest 15m: compress×NY selective ~0.60–0.64 OOS** (V23), stable across 4 windows. Never 75%.
- Dead ends (do NOT repeat): regime-specialist models (worse than gating+selectivity), meta/correctness models
  (OOS AUC 0.502 = random), economic-calendar features (0 AUC), intra-FX cross-pair (redundant, marginal),
  aggregating 3s tick signals to 15m (fewer than 1 confident call/window).
- Real >75% edge exists only at ≤5–8s microstructure scale — NOT tradeable as a binary expiry.
- Sofien corpus hypotheses to test: compression×session reversion, oscillator-extreme (RSI/Stoch/Z-score) reversion,
  autocorrelation/momentum-ratio regime, multi-TF trend confluence. (Most are mean-reversion in quiet regimes.)

## Data facts (verified this session)
- Valid 30m rows: VAL 738,843 (ȳ=0.5004) · TEST24 369,656 (0.5072) · TEST25 367,396 (0.5083) · OOS26 128,965 (0.4956).
- Base rate ≈ 0.50 everywhere — no directional tilt. 239 causal multi-TF features (1m/5m/15m/30m/1h/4h) available.

---

## ITERATIONS
(newest appended below; each: hypothesis → setup → result per window → verdict → next)

### Iteration 1 — Baseline ML selective + regime gates (single LGB, 239 feats)
**Setup:** LGB on all 239 feats, recompute label HOR=30. Cache probs; sweep gates×coverage; honest indep (non-overlap chrono 1800s) acc + CI95 across VAL/TEST24/TEST25/OOS26.
**AUC:** val 0.528 · test24 0.526 · test25 0.529 · **oos 0.516** (best_iter=37 — model barely fits; matches V8 0.518).
**Selective results (point est, OOS):**
- gate=none: oos 0.51→0.57 as cov 10%→1%. 
- gate=ny: oos ~0.578 @cov2% (n83). 
- comp×NY gates: best & most consistent lift (corroborates 15m compress×NY). comp1h_ny @cov2%: val 0.646 / test24 0.644 / test25 0.575 / **oos 0.639** (n83) — a real ~0.62–0.64 edge.
- At cov1% several gates SPIKE (comp30_ny oos 0.842 n19; comp15_ny oos 0.750 n12; test24 0.760) **but n=12–53 and test25 stays ~0.64** → small-n mirage, NOT consistent across all 4 windows.
**Verdict:** ML-confidence selection reproduces the honest ~0.60–0.64 frontier. NO config ≥0.75 across all 4 windows. Confidence-on-a-near-random-model is the wrong lever (AUC 0.52). 
**Next:** test a DIFFERENT mechanism — structural conditional base rates by pre-committed regime rules (oscillator-extreme reversion in quiet ranges; trend-confluence continuation). m30_regime.py across TRAIN/VAL/TEST24/TEST25/OOS.

### Iteration 2 — Cross-pair / USD-basket features (base 239 + 60 peer feats, single LGB) [exp_horizon.py 30]
**Result:** AUC val 0.528 / test 0.528 / **oos 0.518** — identical to base. VAL never reaches 65/70/75% at any coverage. @60%-target VAL thr → OOS 0.551 (overlapping n1389). **Verdict:** cross-pair adds ~0 at 30m (as it did at 15m, V16). Dead end.

### Iteration 3 — (pending) Ensemble lgb+xgb+cat + agreement-confidence + two-factor reversion overlay (m30_ens.py)

### Iteration 4 — FX fixing-window reversal (literature: Krohn-Mueller-Whelan JoF 2024) [m30_fix.py]
**Setup:** at fix minute bet REVERSAL of pre-fix 30/60m drift; settle +30m. Mapped time-of-day on TRAIN, verified pre-committed windows across all splits.
**Result:**
- TRAIN time-of-day map: **16:00 UTC (WMR London 4pm) is the single most special minute** — reversal_acc 0.566 (next best ~0.54). Confirms the literature W-shape.
- WMR_16utc reversal: train 0.566 / val 0.536 / test24 0.580 / test25 0.549 / **OOS26 0.429** — sign FLIPS in 2026 (continuation 0.571 OOS). n≈91 OOS.
- ECB/Tokyo/NYopen windows: all ~0.50–0.56, no consistent sign across splits.
**Verdict:** The fix reversal is real & literature-backed but peaks ~0.58 and is UNSTABLE OOS (post-2013 WMR reform changed it). Not ≥0.75; not even reliably profitable. Matches V21's "London-fix pocket decayed OOS."

### Web/scholarly synthesis (research agent, citations in transcript)
Always-on >75% directional accuracy at 30-min EURUSD is **NOT documented anywhere credible**. Peer-reviewed consensus: 50–53% unconditional (Petrova-Vilhelmsson-Nordén IJF 2026 "supports EMH"; Meese-Rogoff; Rossi JEL 2013). Order-flow edge significant at 1-min, **gone by 15/30-min** (FRB/EBS IFDP 830). Selective frontier ~0.60–0.65 @5–15% cov; 65–75% only at ~0.5–3% cov in mechanistic windows (fixings/macro-surprise) and mostly fails deflation (Bailey-López de Prado DSR/PBO; White Reality Check; Hansen SPA). Binary breakeven = 1/(1+payout) ≈ 0.55–0.59, not 0.75. Every credible ">75%" traces to equities@5s, daily horizons, or look-ahead/marketing.

**Converging evidence (5 internal + literature): the honest 30m EURUSD frontier is ~0.52 unconditional, ~0.62–0.64 selective. A robust OOS-verified >75% is not present in this data.**

### Iteration 3 (run) — Ensemble lgb+xgb+cat + agreement + two-factor (m30_ens.py)
**AUC:** ens val 0.528 / test24 0.526 / test25 0.528 / **oos 0.515** (3 models all ~0.51–0.53; ensemble adds nothing to AUC).
**Honest selective (cross-window):**
- Best STABLE pocket = ENS-conf **comp_1h @cov1%**: val 0.620 / test24 0.583 / test25 0.598 / **oos 0.621** (n124, CI[0.53,0.70]) → combined held-out ≈ **0.60**, all windows 0.58–0.62, OOS CI excludes 0.50. Profitable vs 0.541 breakeven.
- comp1h_ny @cov2%: test24 0.569 / test25 0.614 / oos 0.627 (~0.61, smaller n).
- Agreement / two-factor: ~0.57–0.61, no lift over single confidence. 
- Every ≥0.70 cell is single-window + small-n (val cov2% 0.71–0.73 anti-transfers; oos comp15_ny cov1% 0.769 **n13**; test24 ny cov1% 0.716). **No config ≥0.75 across all four windows.**
**Verdict:** ensemble tightens consistency but the ceiling is firmly ~0.60–0.64. 

## FINAL VERDICT (7 converging lines + literature + repo's own 16-report corpus)
1) raw AUC ~0.52; 2) structural regimes ~0.50–0.55 unstable; 3) ML selective ~0.62 non-transferring; 4) cross-pair +0; 5) FX fixing reversal ~0.58 sign-flips OOS; 6) ensemble+agreement ~0.60–0.64; 7) repo research E1–E5 backlog + SYNTHESIS independently concluded "always-on 75% at intraday FX is still open / not achieved; breakeven is ~57%, 0.63 is already economically live; open problem is coverage+stability not crossing 75%." Fresh scholarly review (Petrova-Vilhelmsson-Nordén IJF 2026 EMH; Meese-Rogoff; Rossi JEL 2013) concurs.
**A robust OOS-verified >75% 30-min EURUSD up/down does NOT exist in this data.** It is a seconds-scale phenomenon, untradeable as a binary expiry. The HONEST, tradeable, PROFITABLE deliverable = selective ensemble in compression×NY ≈ **0.60–0.64** (beats deriv's ~0.541 breakeven).
**Only untried levers that could ADD directional information (per repo E1/E2): EXTERNAL data not in this repo** — US-DE rate-diff + ES/Treasury futures ticks (E1), and an economic calendar with SIGNED macro-surprise (E2) — both flagged "small lift, concentrated in release windows," not a path to robust 0.75.

### FINAL DELIVERABLE — m30_production.py (compression-1h × NY selective ensemble)
**Trained & saved:** models/m30_EURUSD_{direction_lgb.txt,direction_xgb.json,direction_cat.cbm,strategy.json}.
**VAL-selected (pre-committed, frozen):** gate comp(1h)×NY, cov5%, conf_thr=0.0457 (VAL indep acc 0.644, n160).
**Held-out backtest (INDEPENDENT non-overlap 1800s, chronological, CI95):**
- TEST-2024: n281 acc 0.623 [0.566,0.680] EV@0.85 +0.152
- TEST-2025: n343 acc 0.589 [0.536,0.641] EV +0.090
- OOS-2026:  n183 acc 0.546 [0.475,0.617] EV +0.011  (weakest window; ~4mo data; corr(VAL,OOS)=-0.54 anti-transfer visible)
- **COMBINED held-out: n807 acc 0.591 [0.556,0.625]** — lower CI clears 0.541 breakeven; EV +0.064/+0.093/+0.123 at R=0.80/0.85/0.90.
**Leakage self-test (audit_leakage.py): ALL PASS**, incl. m30 exact-1800s wall-clock expiry, independence, split disjointness.
**Conclusion:** honest, deriv-faithful, methodology-clean 30m model. Real & profitable (~0.59 held-out, beats breakeven) — NOT 75%. The >75% target is not achievable at 30m on EURUSD with in-repo data (7 lines of evidence + literature). User accepted this honest model (2026-05-30).

### Iteration 5 — TICK MICROSTRUCTURE features for 30m (m30_tick.py) — the untried data source
**Setup:** min1's 62 causal microstructure feats (order-flow imbalance + persistence, microprice dev, rv30..rv3600, spread, tick intensity, stretch/rangepos/bbw≤3600s) on 1s bars; deriv-faithful label wc_ret HS=1800. LGB on 648k strided train rows; honest selective across val(2024H1)/test24/test25/oos.
**Result:** AUC val **0.501** / test24 0.503 / test25 0.514 / **oos 0.503 — PURE NOISE (0.50)**. lgb early-stopped at 28 trees. Best selective OOS ~0.54 (comp gate cov10%); every cov1% spike (val 0.69/0.65) is small-n, collapses OOS (0.49–0.56).
**Verdict:** microstructure STATE carries ~ZERO 30-min directional info — the LEAST informative source, confirming order-flow decays by 15-30m (FRB/EBS; repo report 02). 8th converging null.

## DEFINITIVE CONCLUSION (8 converging lines, all Tier-1 this session)
raw OHLCV AUC 0.52 · structural regimes ~0.52 · ML selective ~0.62 non-transferring · cross-pair +0 · FX fixing ~0.58 sign-flips OOS · ensemble+agreement ~0.60-0.64 · **tick microstructure 0.50 (noise)** · repo's own 16-report corpus + fresh scholarly review = same null. A robust OOS-verified >75% 30-min EURUSD up/down is **not achievable with any in-repo data/method**. Honest delivered model: m30_production.py ~0.59 combined held-out (profitable vs 0.541 breakeven). Only untried information-adding lever = EXTERNAL data (US-DE rates + ES/Treasury futures ticks; signed macro-surprise calendar) — not in repo, flagged small/release-window lift, not a guaranteed 75%.

### Iteration 6 — EXTERNAL cross-asset: CME ES (S&P futures) lead-lag, E1 lever (m30_es_feas.py)
**Data found on system:** LEAN `/home/sean/git/.../lean/data/future/cme/minute/{es,nq,mes,mnq}` — UTC-timestamped (market-hours db dataTimeZone=UTC), spans 2010→2026-05, **covers OOS**. (No Treasury/rate futures — CBOT has only grains — so E1's rate-diff half is untestable/unavailable anywhere accessible.)
**Alignment verified:** contemporaneous corr(es_r15, eur_r15) = +0.16/+0.22 (2024), +0.03/+0.18 NY (2025) — positive & NY-stronger, so UTC alignment is correct and ES↔EUR co-move *simultaneously*.
**30-min LEAD (the tradeable part) = NULL & sign-unstable:**
- corr(es_r15(t), eur_fwd30): +0.001 / −0.041 / +0.011 (2024/25/26); NY +0.020 / −0.107 / +0.021 — flips sign yearly.
- E1 catch-up corr(es15−eur15 gap, eur_fwd30): +0.002 / −0.096 / +0.043 — null, unstable.
- sign(es_r15)→dir acc 0.498/0.499/0.506; sign(gap15)→dir acc 0.504/0.503/0.512 — coin-flip.
**Verdict:** ES equity-risk co-moves with EURUSD contemporaneously (untradeable) but does NOT lead it at 30m — exactly Evans-Lyons / FRB-EBS / repo report 07. The one credible EXTERNAL lever is dead at 30m. 9th converging null, now using real external data through OOS.

## SEARCH EXHAUSTED (9 Tier-1 lines + literature + repo's 16-report corpus + external data through 2026)
No honest, robust, OOS-verified >75% 30-min EURUSD up/down exists. Deliverable = m30_production.py ~0.59 combined held-out (profitable vs 0.541 breakeven). Untried levers that remain require data that does NOT exist in any accessible form (minute Treasury-rate futures; proprietary bank/CLS customer order flow).

### Iteration 7 (final data check) — external multi-asset availability
Scanned LEAN data for assets covering OOS 2024-2026 at minute resolution: ONLY ES + NQ (US equity indices, ~same risk
factor; both proven null at 30m lead in iteration 6). Gold (GC) stops 2020, crude (CL)/Treasuries (ZN) ABSENT, gold/DAX
CFDs have days only. DXY-basket = intra-FX (cross-pair, already +0 at 30m, iteration 2). => NO untested external data
source exists that could add 30m directional signal. External data is EXHAUSTED.

## FINAL STATE (30m goal): >75% is NOT achievable on EURUSD with any data reachable on this system.
~15 experiments across 1s-30m, 4 data sources (OHLCV/tick/cross-pair/external-futures), 3 model types (GBM/CNN/GRU),
Sofien custom indicators + price-action setups, full research/ corpus + scholarly literature. Honest verified results:
30m 0.591 (m30_production) · 15m 0.647 (m15_production, BEST tradeable) · 3s 0.657 (mtick3, the only >65%). The ceiling
is the data's information content (~0.52 AUC at 30m), invariant to features/architecture. A robust >0.75 at 30m does not
exist; manufacturing one would require the small-n/multiple-testing bias this whole project exists to eliminate.
