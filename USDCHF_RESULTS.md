> **SCOPE: USDCHF** (key-specific — results of record + UP/DOWN leaderboard). Generic methods: METHODS_CATALOG.md. Sweep menu: SWEEP_MATRIX.md. See REPO_MAP.md. Bootstrapped 2026-06-10 (new currency; structure copied from USDCAD_RESULTS.md). Current scope: **15m only** (deriv-FX-deployable floor + program's best direction horizon).

# USDCHF — Results Ledger (unique key: currency × timeframe × side)

**Pair tag: `USDCHF`.** USDCHF is a **USD-BASE / CHF-QUOTE** pair (USDCHF up ⇒ USD strengthens vs CHF ⇒ "USD up / CHF down"). CHF is **THE premier safe-haven currency** (more than JPY). It is **doubly determined** and the central modelling question is which face dominates:
- **SAFE-HAVEN (USDJPY analog).** Risk-off ⇒ CHF bid ⇒ USDCHF **DOWN** (the sharp directional move = flight-to-safety / CHF strength). USD on the NUMERATOR like USDJPY. If this face dominates, USDCHF is **own-pair-specific + NY/risk-concentrated**, pooling DILUTES (the USDJPY/AUDUSD/USDCAD base-rate: 3/3 certified safe-haven/USD-base majors were own-pair-specific).
- **EUR-BLOC / SNB-MANAGED (EURUSD analog) — THE DIFFERENTIATING HYPOTHESIS.** CHF tracks EUR extremely tightly: the SNB ran a EURCHF≥1.20 floor 2011–2015 and still manages the franc (sight-deposit intervention + negative-rate era), so **USDCHF ≈ −EURUSD modulated by a slow EURCHF** — USDCHF is strongly (inversely) correlated to EURUSD. EURUSD and GBPUSD both **CERTIFIED via cross-pair EUR-BLOC POOLING** (own-pair was beaten). **IF the EUR-bloc face dominates CHF, cross-pair pooling could WIN here too — UNLIKE the safe-haven analogs.** Run cross-pair pooling EARLY, **sign-FLIPPED** (USDCHF moves OPPOSITE to EURUSD). This is the single most informative early experiment for the key.
- **SNB INTERVENTION REGIME (CHF-unique invent-lever).** Official one-directional flow (sight deposits / FX purchases) can impose persistent drift / regime structure no other major has. A SNB-regime gate is a bespoke discovery lever (off-disk SNB sight-deposit data = acquisition frontier). NB: 2015-01-15 SNB floor-removal crash is gap-excluded by the contiguity check.
- **SESSION.** CHF is a EUROPEAN (Zurich) currency → trades heavily in the LONDON/European window, so the **LDN-session prior is materially HIGHER here than for JPY/CAD/AUD** (hygiene: ldn≈33% of moved bars vs ny≈37%, asia≈38%). NY still strong (US risk events drive haven flow); Asia low.

**Side-asymmetry prediction (falsifiable).** The sharp, news-driven directional move in CHF is the risk-off flight = CHF strength = USDCHF **DOWN**. **Prediction: USDCHF DOWN (haven flight) is the more forecastable side.** BUT if the EUR-bloc face dominates, the side that pools best with EURUSD may lead instead — the side-split + pooling test decide; v0 is a hypothesis, not a conclusion.

**The unique result key is `(currency, timeframe, side)`, side ∈ {UP, DOWN}.** Where a key is unmeasured it is `UNTESTED`. COMBINED is context only, never an UP/DOWN number.

**Conventions.** Deriv-faithful settlement (bar-close approx at 15m, mid-to-mid, next-tick entry +1s, ties LOSE), breakeven **0.541** (deriv payout R≈1.85). Splits: bars train 2012-21 / val 2022-23 / test 2024 / test 2025 / oos 2026 (strict OOS). Selection on VAL worst-half (never VAL-acc-max; `corr(VAL,OOS)=−0.54`). Moved-bars-only; verify moved up-rate ∈ [0.47,0.53] (**USDCHF checked clean: .5034/.4996/.5041 for 2024/2025/2026, flat_frac <1.3%** — no fake-flat mirage). Per-year CI95 (bootstrap). de-overlap = `nonoverlap_chrono` gap=900s. CERTIFY a side ONLY via full per-fold-refit CPCV at the operating gate: p10 ≥ 0.541 AND ≥ ~80% of 15 purged paths clear 0.541; adversarially verify every positive (frozen-forward, trap#9). **Target: >65% accuracy** (achievable, if at all, only at tight coverage — neighbors hit ~.60–.65 at cov1–2%). Bar features on disk 2012–2026 (no tick data for USDCHF — bar-close proxy validated on USDJPY/AUDUSD −.0035). Last updated 2026-06-10 (**SWEEP OPEN** — bootstrapped; A1 base running).

---

## MASTER KEY TABLE — one row per (USDCHF, timeframe, side)

| Key (currency · timeframe · side) | **Best OOS % (2026)** | Model id · content_id | Description | Status |
|---|---|---|---|---|
| USDCHF · **15m** · UP | `UNTESTED` (A1 base running) | — | safe-haven vs EUR-bloc face TBD; pooling test pending | 🔄 IN PROGRESS |
| USDCHF · **15m** · DOWN | `UNTESTED` (A1 base running) | — | predicted sharper side (haven flight); pooling test pending | 🔄 IN PROGRESS |
| USDCHF · 1m/2m/5m/10m/30m · UP/DOWN | `UNTESTED` | — | Out of current scope (goal = 15m). Bar data present (no tick). Bootstrap when scoped. | UNTESTED |

---

# Per-timeframe detail

## USDCHF × 15m

### Combined-book / single-pair experiments
| # | Method (file) | Result (2024 / 2025 / 2026 moved-AUC; gate cov2% COMB wr) | Verdict |
|---|---|---|---|
| _pending_ | `usdchf_15m_base.py` | running | — |

### Coverage curve (step d) — `usdchf_15m_base.py` covcurve (frozen-2012-21 book; per-year, per-cov thr)
| cov | COMB wr 2024/25/26 | UP wr 2024/25/26 | DOWN wr 2024/25/26 |
|---|---|---|---|
| _pending_ | — | — | — |

### All-session refit-CPCV (per-fold-refit, 15 purged paths) — `usdchf_15m_cpcv_session.py all`
| cov | UP p10 (frac_clear) | DOWN p10 (frac) | COMB p10 (frac) | CERT |
|---|---|---|---|---|
| _pending_ | — | — | — | — |

### Session segmentation (A9) — `usdchf_15m_cpcv_session.py {ny,ldn,asia}` (symmetric refit-CPCV)
| session | UP p10 @cov (mean) | DOWN p10 @cov (mean) | COMB p10 | CERT | verdict |
|---|---|---|---|---|---|
| _pending_ | — | — | — | — | which session carries (NY vs LDN — CHF is European) |

### ★ Cross-pair EUR-bloc POOLING (the differentiating test) — `usdchf_15m_xpair.py` (sign-FLIPPED, USDCHF≈−EURUSD)
| variant | VAL AUC vs base | 2026 COMB/UP/DOWN | verdict |
|---|---|---|---|
| _pending_ | — | — | does CHF pool with EUR-bloc (WIN like EURUSD/GBPUSD) or own-pair (NULL like JPY/CAD)? |

### Improve cross-product (on the carrier book) — pending
| lever | file | result | verdict |
|---|---|---|---|
| _pending_ | — | — | — |

---

## FINAL CONCLUSION (USDCHF 15m direction) — _pending (sweep open)_

_To be written when both sides certified-or-honestly-exhausted and improve+discover loops are dry._

---

## UP/DOWN LEADERBOARD (current best per side, certified-or-best-available)
| Side | Best certified (refit-CPCV p10) | Best available (mean) | Book | Status |
|---|---|---|---|---|
| **15m UP** | `UNTESTED` | — | — | 🔄 sweep open (A1 base running) |
| **15m DOWN** | `UNTESTED` | — | — | 🔄 sweep open (A1 base running) |

_Provenance: every number traces to a `*_result.json` (Tier-1). Updated as rows complete._
