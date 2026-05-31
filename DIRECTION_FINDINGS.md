# EURUSD Binary DIRECTION — Exhaustive Investigation & Findings (2026-05-30/31)

Master summary of the multi-day effort to build a high-accuracy EURUSD up/down binary model across horizons
(1 second → 30 minutes), with the deriv-faithful, leakage-controlled, OOS-verified methodology established by the
2026-05 bias audit. **Every number here is held-out (TEST 2024 / 2025 + OOS 2026), independent (non-overlapping
windows, chronological/no-look-ahead), bootstrap-CI'd.** Companion logs: `m30_research_log.md`, `m5_research_log.md`,
`IDEAS_LOG.md` (orthogonal-math program), `research_log.md` (pre-audit + bias-audit history).

## TL;DR — the honest frontier

| Horizon | DIRECTION (sign) — best honest OOS | Tradeable? |
|---|---|---|
| 1–5 seconds | **~0.65–0.66** (tick microstructure ensemble) | only on a tick/seconds-expiry broker; latency-critical |
| 15 minutes | **~0.64** (compression × NY-session selective ensemble) | ✅ deriv (at the forex floor) |
| 30 minutes | **~0.59** (compression-1h × NY selective) | ✅ deriv |
| 5 minutes | **~0.56** | needs a ≤5m-expiry broker |

**A >75% (or even >65%) DIRECTIONAL edge does not exist at 5m/15m/30m on EURUSD.** Direction at ≥5 minutes is the
efficient-market part: ~0.50–0.52 AUC, ~0.515 unconditional (mild mean-reversion). The only place a >0.65 *directional*
edge is real is the **seconds scale**, and it decays to noise by ~60s.

## The genuine positive result: MAGNITUDE is forecastable, DIRECTION is not

The single most important finding (and the answer to "group the data to tell a different story"):

> **30-minute |return| (move SIZE) is strongly, OOS-verifiably forecastable — realized-vol → large-move AUC 0.73–0.78**
> (corr +0.38–0.44, stable across 2024/2025/2026). DIRECTION/sign stays ~0.515.

This is backed by a sign-invariance **theorem** (arXiv:2512.15720, Dec 2025): order-flow / permutation entropy is
invariant under sign permutation, so complexity/entropy measures detect the *presence/size* of informed moves
(magnitude), **not the sign**. That is *why* every complexity/regime/gate approach was null for direction — they gate
volatility, not direction. Magnitude is tradeable on **Touch/No-Touch, Range/Boundary, straddle** products (not up/down).

## Everything tried (exhaustive; all OOS cross-window honest)

**Models / data (direction):**
- OHLCV 239-feature ensemble (lgb+xgb+cat) at HOR 5/15/30 → AUC ~0.52; selective ~0.56–0.64.
- Tick microstructure (order-flow imbalance, microprice) HS 1–1800s → AUC ~0.50 @30m, **~0.65 @1–5s**.
- 1D-CNN & GRU on the raw tick path → AUC 0.525 (= GBM): **model type is not the bottleneck, the data is**.
- Cross-pair / USD-basket, external CME ES/NQ futures lead-lag (real data thru 2026) → +0 / zero 30m lead.
- Volume & dollar bars (López de Prado information bars) → null for direction (improve normality, not AUC).
- Sofien Kaabar's 45 custom indicators as features → 0 OOS AUC (rank high in importance, redundant OOS).

**Gates / rules (direction):**
- Larger-TF reversals (1h/4h/daily RSI/DeMarker/WillR/CCI/BB/z/Fisher extremes; confluence; chop-gated) → null.
- Sofien's 79 mined rules (`sofien_rules.json`): daily pivots, Internal Bar Strength, Connors RSI2, TD Setup-9,
  BB re-entry, divergence, confluence → null (best Connors RSI2 ~0.55 train/val → 0.47–0.51 OOS).
- "Avoid losers" complexity gates: permutation entropy, Hurst (variance-ratio + DFA), autocorrelation → null
  (flat accuracy across all bins → they gate magnitude, not sign).
- FX fixing-window reversals (WMR 16:00 / ECB / Tokyo, Krohn-Mueller-Whelan) → ~0.58 in-sample, sign-flips OOS.

**Outside-finance math (4 research workflows + builds):**
- Path signatures / Lévy area (price↔order-flow rotation) → null @30m, top-5 feature @5s (= the seconds edge).
- Hawkes processes, transfer entropy, convergent cross mapping, RQA/DFA → directional variants are sub-minute /
  hourly / need trader-resolved LOB we don't have; the rest are sign-invariant (magnitude).
- Permutation-entropy / weighted-PE / sample-entropy → magnitude gates (per the theorem), null for sign.

**Convergent external evidence:** Petrova-Vilhelmsson-Nordén (*Int. J. Forecasting* 2026, FX LOB, 1min–1h):
near-EMH predictability. Meese-Rogoff; Rossi (*JEL* 2013). The repo's own 16-report `research/` corpus reached the
same null independently.

## Lessons (methodology — carry forward)

1. **Direction ≠ magnitude.** At ≥5m, sign is ~EMH; volatility is highly forecastable. Don't conflate a high
   *magnitude* AUC with a *directional* edge. (The sign-invariance theorem makes this rigorous.)
2. **Model type / feature richness is not the bottleneck — information is.** GBM ≈ CNN ≈ GRU ≈ 0.525 AUC at the data
   ceiling; Sofien indicators and signatures add 0 OOS AUC; the limit is the data.
3. **The edge is at the seconds scale and decays fast** (5s ~0.65 → 60s ~0.52). Longer = more efficient.
4. **corr(VAL, OOS) ≈ −0.54**: VAL-max selection anti-transfers. Require a result to hold across ALL of
   {TEST24, TEST25, OOS26}; every single-window ≥0.75 was an n=12–50 / multiple-testing mirage.
5. **Don't fabricate from low-coverage pockets** — that is the exact bias the 2026-05 audit existed to kill.
6. **Information-driven bars** improve return normality, not directional AUC (matches the literature).

## Deliverable models (all in this repo; `models/` is gitignored)

- `m30_production.py` — 30m direction, compression-1h × NY selective ensemble, held-out **0.591** (CI[.556,.625]).
- `m15_production.py` — 15m direction, compression × NY, **0.647** combined — the best *tradeable directional* edge.
- `m_tick_prod.py` — 3s tick ensemble, **0.657** (the only >0.65 directional; needs a tick-expiry broker).
- `m5_production.py` — 5m direction, **0.566**.
- `m30_magnitude.py` — the **magnitude/volatility** model (large-move AUC ~0.75) — the genuine new edge; tradeable on
  volatility/touch products. **Recommended next productionization.**

## Research/experiment scripts (the lab — all reproducible)

`m30_lab.py` `m30_regime.py` `m30_ens.py` `m30_tick.py` `m30_fix.py` `m30_es_feas.py` `m30_gates.py`
`m30_complexity.py` `m30_magnitude.py` `m30_sig.py` `tickhz.py` · `m5_lab.py` `m5_patterns.py` `m5_tick.py`
`m5_sofien.py` `m_cnn.py` `m_tick_prod.py` `vbars.py` · `sofien_rules.json` (79 mined rules) · `ENVIRONMENT_libs.txt`.
