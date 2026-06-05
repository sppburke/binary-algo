# USDJPY × 2m — Executable Backlog (FIRST-TO-RUN queue + reasoning)

SCOPE: USDJPY · 2m (120s). KEY-SPECIFIC. The reasoned, prioritized experiment queue. Ledger of record =
`sweeps/USDJPY_2m.md`; results of record = `USDJPY_RESULTS.md`. Update the MODEL OF THE EDGE in the ledger
after every result, then re-rank this queue.

## INCUMBENTS TO BEAT (start of sweep)
- No certified USDJPY 2m book yet. Cross-horizon refs: USDJPY 1m near-efficient (best UP uncertified .547/.534/.538);
  EURUSD 2m `EURUSD.min2.v1` UP 0.555 robust / DOWN 0.540 marginal (TICK-based — not directly portable; USDJPY has no
  tick cache, so USDJPY 2m is bar-based).
- Deriv breakeven 0.541. A side is CERTIFIED only via full per-fold-refit CPCV at the operating gate
  (refit p10 ≥ 0.541 AND ≥~80% folds clear).

## REASONING CHAIN (mechanism-first; each step attacks a named cause / builds on the closest result)
1. **A (base capacity).** First read: does the 2m horizon lift USDJPY UP toward/over 0.541? 1m UP plateaued ~.534 (data-starved
   tail, +1pp from more data still < BE). 2m doubles signal-to-microstructure-noise. Falsifier: VAL moved-AUC ≤ .515 OR no
   year COMBINED CI-lo ≥ .541. Build the strong config (stride6/leaves255) regardless — 1m showed the tail is data-hungry.
2. **D1 compression-release specialist (HIGH prior .30).** This is the *exact mechanism* that gives EURUSD its 2m edge
   (min2_production: "compression-release regime predicts the breakout direction"). The all-bars base GBM dilutes it.
   USDJPY's BoJ-pinned / range-then-break behaviour is, if anything, MORE compression-release-driven. Gate on low
   Bollinger-band-width (compression) + range-position, predict breakout sign. Mechanism carries DIRECTION (sign of the
   release), survives sign-invariance. Test UP-release and DOWN-release separately.
3. **C1 cross-pair POOLING (HIGH prior .40).** The 30m lesson: pooled base-feature training = ~94% of the cross-pair gain
   (NOT the OF factors). Pool 7-major base features (own-clock-aligned to USDJPY's 2m label) → more regime coverage,
   decorrelated noise. Mechanism: shared global reversion/trend structure; USDJPY's idiosyncratic (carry/BoJ) part stays
   in its own features. Attacks the data-starvation cause directly. Compare vs A and D1.
4. **B specialists + side-splits.** Score UP and DOWN separately at every row (free from covcurve). DOWN historically dead
   at short horizons (rally-sell), BUT EURUSD 2m DOWN reached .540 — so a purpose-built DOWN (rally-fade in compression)
   gets a real attempt before declaring DOWN dead.
5. **F Tier-I levers on the BEST edge found** (seed-ens, ACI gate, calibration, Optuna-worst-VAL-half) + their COMBINATIONS.
   Only after a base edge clears/approaches BE — levers lift a real edge, they don't manufacture one.
6. **E loss/DL/state-space (lower prior).** |ret|-weighted/GMADL coupling the objective to payoff; GRU decorrelated stack;
   forward-filter state-space (sign-invariance gate). Run once each (coverage rule) sized to prior.
7. **G CPCV cert** triggers the moment any row shows a worst held-out year CI-lo ≥ 0.541.

## FIRST-TO-RUN QUEUE (re-rank after each result)
1. [RUNNING] A1 `usdjpy_2m_base.py` (stride24/leaves127) — canonical baseline, fast read.
2. A2 `usdjpy_2m_base.py 6 255` — strong config (1m-best). Run regardless of A1 (tail is data-hungry).
3. D1 `usdjpy_2m_regime.py compress` — compression-release specialist (the EURUSD-2m mechanism). WRITE this script.
4. C1 `usdjpy_2m_xpair.py pool` — cross-pair pooled GBM (the 94%-lever). WRITE this script.
5. B3/B4 specialists; D2 session gate; then F Tier-I levers on the best; E lower-prior; G CPCV on any survivor.

## DISCOVERY (Tier-N) — run after Tier A–D drain; loop until 2 dry rounds
- Mine `/home/sean/git/academic-papers/_CORPUS_INDEX.md` + `_extracted_levers.json` for 2m-direction levers not yet tried
  at this key; fan out reader sub-agents. Vet each for a DIRECTION (sign) mechanism (sign-invariance) before adding.
- Candidate seeds to vet: Hawkes/self-exciting breakout timing → direction conditioning; triangular-arbitrage residual
  (USDJPY via EURUSD×EURJPY) sign; realized-skew / signed-jump sign-predictivity at 2m; intraday DE–US/US–JP 2y rate-diff
  (EXTERNAL data — gated). Append vetted rows to ledger Tier-N + IDEAS_LOG.md (generic) + here (per-key).

## EXTERNAL-DATA FRONTIER (gated on explicit user "go" — do not acquire without it)
- Intraday US–JP 2y rate differential (carry driver; BoJ vs Fed) via Dukascopy; daily JPY implied-vol / risk-reversal;
  EURJPY+EURUSD ticks for triangular USDJPY synthetic. These are the only inputs that could change a near-efficiency verdict.
