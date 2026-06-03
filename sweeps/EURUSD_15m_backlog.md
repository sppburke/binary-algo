# EURUSD · 15m — experiment backlog (queued, not yet run)

SCOPE: EURUSD 15m (key-specific). Generic levers/templates: `IDEAS_LOG.md`. Results of record: `EURUSD_RESULTS.md`.
The 15m book `EURUSD.m15.v1` is the **deriv-tradeable binary product** (deriv forex Rise/Fall min duration = 15m,
verified live 2026-06-03 via `contracts_for` — `m5_venue_feasibility_result.json`). Unlike 5m (research-grade, no binary
venue), a 15m UP/DOWN edge is **directly deployable as a deriv binary** (R≈0.85, breakeven 0.541).

## STATUS (as of 2026-06-03)
- 15m is **only COMBINED-tested** (`EURUSD_RESULTS.md`: 2026 slice 0.663; cross-era CPCV-faithful **0.579**, p10 0.557).
  **Never side-split.** PREREQUISITE for everything below: run the (15m,UP) and (15m,DOWN) **side-split** first
  (`m5_updown.py` pattern retargeted `MX_HOR=15`), then the full strategy-eval pipeline (a)–(e) per side.

## QUEUED EXPERIMENTS
### Q1 — Magnitude × direction gate on the 15m book  [from the 5m session, user-queued 2026-06-03]
- **Idea:** gate the 15m direction trades on a predicted move-size (magnitude) signal. Magnitude is the one strongly-
  forecastable quantity in this program (5m magnitude AUC ~0.74).
- **Mechanism / why 15m may differ from 5m:** at 5m this was a NULL — `m5_magspot_result.json` showed magnitude-gating
  *worsens* the edge (gating to large moves raised avg move 2.2→4.3 pip but dropped binding-year hit-rate, because large
  5m moves are jump/informed-driven and *less* sign-predictable; the 5m UP edge is a SMALL/diffusive-move dip-buy). At 15m
  the move composition is different (more trend, less microstructure noise), so the magnitude×direction interaction must be
  RE-TESTED, not assumed. ALSO: the 15m product is a BINARY (fixed payout) so move-size doesn't help payout directly — the
  ONLY way magnitude helps the 15m binary is if the directional HIT-RATE is higher in some magnitude regime (the opposite
  of the 5m finding) — that is the falsifiable question.
- **Prior:** GUARDED-LOW (~0.10) given the 5m null + sign-invariance theorem (arXiv:2512.15720), but the deployable-binary
  context + different-horizon regime make it worth a confirm-or-kill when the 15m book is the focus.
- **Test plan:** train a 15m move-size predictor on the 15m features (target |fwd_15m|≥median, leakage-safe feats≤t); bucket
  the 15m UP/DOWN gated trades by predicted-magnitude; per-year 2024/25/26 hit-rate (binary, ties-LOSE, breakeven 0.541) +
  CI95, worst-VAL-half selection, up-rate tripwire. Mirror `m5_magspot.py`.
- **Pre-registered falsifier:** KILL unless some magnitude bucket lifts the 15m binding-year hit-rate CI95-lo above the
  un-gated 15m side incumbent by >1 SE (i.e., the 15m hit-rate is genuinely HIGHER on a magnitude regime — the reverse of
  the 5m result). Expected null (sign-invariance), but RUN to confirm at 15m.
- **Script:** `m15_magdir.py` (retarget `m5_magspot.py`, MX_HOR=15). Data on-disk (15m features + frozen `EURUSD.m15.v1`).

## TIER-G — EXTERNAL-DATA FRONTIER (data-acquisition prerequisite; GATED ON USER "go") [discovery round 2, 2026-06-03]
Both 15m sides are CERTIFIED on-disk (UP refit-p10 .5475 / DOWN .5486); on-disk improve+discover is exhausted
(loss/labeling null, cross-pair under final CPCV). The corpus's remaining direction levers are EXTERNAL-blocked —
these are the only inputs that could materially widen the thin margins or strengthen the DOWN side. NOT a modeling
row until the data is acquired. Mechanism-ranked (several are DOWN-side specific — risk-off USD-bid drives EURUSD↓):
- **G1 — DE-US 2y/10y rate differential (intraday, Dukascopy/macro):** the carry/fundamental driver; a two-stream
  macro tower + agreement gate (corpus: EXFormer/fundamentals-tower). DOWN-relevant. Prior ~.15 (the documented #1).
- **G2 — Daily VIX / implied-vol / EUR risk-reversal (risk-off STATE gate):** carry crashes when vol spikes →
  risk-off USD-bid → EURUSD down. A DOWN-side gate. Risk-reversal verified PAYWALLED at 5m; VIX is free (CBOE).
- **G3 — Equity-vol spillover regime (GARCH-MIDAS on S&P realized vol):** low-freq vol component as a regime gate.
- **G4 — Cross-asset daily leads (S&P / oil / gold / 5y):** EXFormer DVS top-importance drivers; risk-on/off lead.
- **G5 — EURGBP ticks (triangular USD-canceling residual, N2):** on-disk-ish if EURGBP ticks acquired; isolates the
  EUR leg from USD. Prior ~.15.
**Falsifier (any G-row):** KILL unless the external signal lifts a side's binding-year refit-CPCV p10 above the
on-disk floor (.5475 UP / .5486 DOWN) by >1 SE under the full discipline. Acquire G1+G2 first (highest mechanism).
