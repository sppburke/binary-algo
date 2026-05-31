# EURUSD Research — FORWARD-LOOKING EXPERIMENT BACKLOG (2026-05-31, 8-agent proactive sweep)

Output of an 8-front literature/competition/alt-data/signal-processing/info-theory/microstructure/methodology sweep
(`scour-everything-fx-research`), each candidate **grep-verified genuinely-untried** against the codebase. Companion to the
backward-looking record `EXPERIMENT_LEDGER.md` / `METHODS_CATALOG.md`. Priors are the agents' honest estimates; discipline as in
`METHODS_CATALOG.md` (non-overlap, ties-lose, CI95, select-on-VAL-verify-each-window, corr(VAL,OOS)=−0.54).

## Honest assessment (verbatim spirit)
**Nothing credibly beats the ~0.55–0.60 60-second direction ceiling** — the online ARF+ADWIN control (AUC 0.503–0.508 every
window) proves the 2025 wall is genuine market efficiency, and sign-invariance (arXiv:2512.15720) explains why every flow/complexity
method was null for sign. **The real upside is elsewhere:** (1) **MAGNITUDE productization** (the AUC 0.73–0.79 |move| model is the
only robust edge and is currently fed only symmetric pe/rv features); (2) **METHODOLOGY** — CPCV + Deflated-Sharpe/PBO to certify
which numbers (15m 0.64, magnitude 0.79, 5m-stack 0.648@n45) survive trial-deflation (zero implementations exist; highest
information-per-line); (3) **DATA ACQUISITION** — daily implied vol (changes monetization) and intraday DE–US 2y rate differential
(mechanical driver whose 2025 failure mode differs from the inverted sentiment links).

## Run-now top experiments (all implementable on current data)

| # | Experiment | Prior @60s sign | Real value | Plan (short) |
|---|---|---|---|---|
| 1 | **CKS event-OFI** (Cont-Kukanov-Stoikov) — proper price-event order-flow imbalance from raw bid/ask changes, replacing tick-rule OFI everywhere | 12–18% | cleaner 2025-stable input to 15m + magnitude | `e_n` from `bid/ask` change-conditioned volumes (raw ticks, all 7 pairs); swap into min1/m15/magnitude; accept only on a real test25-floor lift w/ CI95 |
| 2 | **CPCV + Deflated-Sharpe / PBO** as the SELECTION criterion on the 15m + magnitude (+ 5m-stack) books | 3% (can't create signal) | **~70% helps the program** — certifies what's real vs mirage | install skfolio; CombinatorialPurgedCV(N=8,k=2,embargo=1H); require 10th-pct path > breakeven; deflate over ~70 trials |
| 3 | **Residualized TARGET** — label = sign(EURUSD_ret − β·USD-basket_ret), β causal-rolling; trade on residual∧raw agreement | 20–25% | attacks the 2025 constraint at its root; best at 15m | change the LABEL (not feature) in the xpair pipeline; eval residual-sign AND un-residualized sign per year |
| 4 | **Magnitude upgrade** — diurnal-deseasonalized RV + signed realized-semivariance (RS±, signed-jump) + macro-event-window features | 1–2% sign | **the project's real edge**; AUC 0.79→~0.80–0.82 + regime stability | add to `m30_magnitude.py` (+ a 60s variant); TRAIN-only seasonal profile; require per-window AUC up AND test25 not the floor |
| 5 | **Cross-impact OFI matrix** across 7 USD-pairs (Cont-Cucuringu-Zhang) | 8–12% | 15m + magnitude/abstain feature | 1s signed CKS-OFI per pair in EURUSD-equiv direction; [7×{0,1,2,5s lag}] block; LASSO/LGBM on fwd-15m/60s sign; keep only if test25 CI-clear |
| 6 | **Ordinal transition-asymmetry / time-irreversibility** (Neuman-Cohen-Tamir) — the one ordinal stat sign-invariance does NOT kill | 12–18% | **highest-prior genuinely-new DIRECTION-specific shot** | causal ordinal patterns d=3,4 over W∈{15,30,60,90}s on 1s mid; trailing transition-prob asymmetry as a directional feature |
| 7 | **Causalized Convergent Cross-Mapping (cCCM)** coupling-strength gate, majors→EURUSD | 8–12% | nonlinear 15m/magnitude gate | gate behind a 30-min mRMR MI screen first; kill if cross-asset MI ≈ label-shuffle null |
| 8 | **Sample-uniqueness weighting + sequential bootstrap** (de Prado) on overlapping 15m/magnitude labels | 2% | tighter CI / smaller VAL→OOS gap on 15m+magnitude | `sample_weight = 1/concurrency` from the fixed label horizon; retrain 15m + magnitude LGBM |

## Data worth acquiring (ranked, all free/cheap unless noted)
1. **Daily EURUSD short-dated implied vol + 25Δ risk-reversal + butterfly** (Investing.com scrape / EUVIX) — best external RV
   predictor; the ONLY input that converts the magnitude edge into a tradeable variance-risk-premium product. **Highest value (changes monetization).**
2. **Intraday DE–US 2y rate differential** (Schatz + 2y UST, or DE2Y/US2Y minute yields; Dukascopy, free 2021–2026) — the mechanical
   EURUSD driver; failure mode differs from the inverted sentiment links. Best payoff at 15m + macro-event-window 60s subset.
3. **EURGBP tick/1-min** (Dukascopy/HistData, free) — the missing leg for the true triangular no-arb residual (EURGBP vs EURUSD/GBPUSD).
4. **Intraday DXY / ICE Dollar Index futures** (free) — cleaner single-instrument USD factor for residualizing (vs the synthetic basket).
5. **Trade-tagged / depth-resolved FX LOB** (NOT known free) — the structural unlock for true multi-level OFI / trade-sign Hawkes /
   VPIN; our on-disk feed is indicative quote with lot-quantized sizes and no trade signs, which is why equity LOB headlines don't transfer.

## Execution order (proactive)
Direction-goal first (per standing goal), then the real-upside work: **#6 irreversibility → #1 CKS-OFI → #2 CPCV/PBO certification →
#3 residualized target → #4 magnitude upgrade.** Each judged per-window 2024/2025/2026 with CI95; kept only on a real test25-floor lift.
