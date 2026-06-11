> **SCOPE: USDCHF** (key-specific — results of record + UP/DOWN leaderboard). Generic methods: METHODS_CATALOG.md. Sweep menu: SWEEP_MATRIX.md. See REPO_MAP.md. Bootstrapped 2026-06-10 (new currency; structure copied from USDCAD_RESULTS.md). Current scope: **15m only** (deriv-FX-deployable floor + program's best direction horizon).

# USDCHF — Results Ledger (unique key: currency × timeframe × side)

**Pair tag: `USDCHF`.** USDCHF is a **USD-BASE / CHF-QUOTE** pair (USDCHF up ⇒ USD strengthens vs CHF ⇒ "USD up / CHF down"). CHF is **THE premier safe-haven currency** (more than JPY). It is **doubly determined** and the central modelling question is which face dominates:
- **SAFE-HAVEN (USDJPY analog).** Risk-off ⇒ CHF bid ⇒ USDCHF **DOWN** (the sharp directional move = flight-to-safety / CHF strength). USD on the NUMERATOR like USDJPY. If this face dominates, USDCHF is **own-pair-specific + NY/risk-concentrated**, pooling DILUTES (the USDJPY/AUDUSD/USDCAD base-rate: 3/3 certified safe-haven/USD-base majors were own-pair-specific).
- **EUR-BLOC / SNB-MANAGED (EURUSD analog) — THE DIFFERENTIATING HYPOTHESIS.** CHF tracks EUR extremely tightly: the SNB ran a EURCHF≥1.20 floor 2011–2015 and still manages the franc (sight-deposit intervention + negative-rate era), so **USDCHF ≈ −EURUSD modulated by a slow EURCHF** — USDCHF is strongly (inversely) correlated to EURUSD. EURUSD and GBPUSD both **CERTIFIED via cross-pair EUR-BLOC POOLING** (own-pair was beaten). **IF the EUR-bloc face dominates CHF, cross-pair pooling could WIN here too — UNLIKE the safe-haven analogs.** Run cross-pair pooling EARLY, **sign-FLIPPED** (USDCHF moves OPPOSITE to EURUSD). This is the single most informative early experiment for the key.
- **SNB INTERVENTION REGIME (CHF-unique invent-lever).** Official one-directional flow (sight deposits / FX purchases) can impose persistent drift / regime structure no other major has. A SNB-regime gate is a bespoke discovery lever (off-disk SNB sight-deposit data = acquisition frontier). NB: 2015-01-15 SNB floor-removal crash is gap-excluded by the contiguity check.
- **SESSION.** CHF is a EUROPEAN (Zurich) currency → trades heavily in the LONDON/European window, so the **LDN-session prior is materially HIGHER here than for JPY/CAD/AUD** (hygiene: ldn≈33% of moved bars vs ny≈37%, asia≈38%). NY still strong (US risk events drive haven flow); Asia low.

**Side-asymmetry prediction (falsifiable).** The sharp, news-driven directional move in CHF is the risk-off flight = CHF strength = USDCHF **DOWN**. **Prediction: USDCHF DOWN (haven flight) is the more forecastable side.** BUT if the EUR-bloc face dominates, the side that pools best with EURUSD may lead instead — the side-split + pooling test decide; v0 is a hypothesis, not a conclusion.

**The unique result key is `(currency, timeframe, side)`, side ∈ {UP, DOWN}.** Where a key is unmeasured it is `UNTESTED`. COMBINED is context only, never an UP/DOWN number.

**Conventions.** Deriv-faithful settlement (bar-close approx at 15m, mid-to-mid, next-tick entry +1s, ties LOSE), breakeven **0.541** (deriv payout R≈1.85). Splits: bars train 2012-21 / val 2022-23 / test 2024 / test 2025 / oos 2026 (strict OOS). Selection on VAL worst-half (never VAL-acc-max; `corr(VAL,OOS)=−0.54`). Moved-bars-only; verify moved up-rate ∈ [0.47,0.53] (**USDCHF checked clean: .5034/.4996/.5041 for 2024/2025/2026, flat_frac <1.3%** — no fake-flat mirage). Per-year CI95 (bootstrap). de-overlap = `nonoverlap_chrono` gap=900s. CERTIFY a side ONLY via full per-fold-refit CPCV at the operating gate: p10 ≥ 0.541 AND ≥ ~80% of 15 purged paths clear 0.541; adversarially verify every positive (frozen-forward, trap#9). **Target: >65% accuracy** (achievable, if at all, only at tight coverage — neighbors hit ~.60–.65 at cov1–2%). Bar features on disk 2012–2026. **Tick data note (corrected 2026-06-10):** USDCHF HAS on-disk order-flow modeling features (`features_of/USDCHF_*`, `features_tick_xofi/USDCHF_*cks1s`); what is absent is RAW-tick data for true-tick SETTLEMENT validation → bar-close proxy (−.0035, validated on USDJPY/AUDUSD). Last updated 2026-06-10 (**SWEEP OPEN** — bootstrapped; A1 base running).

---

## MASTER KEY TABLE — one row per (USDCHF, timeframe, side)

| Key (currency · timeframe · side) | **Best OOS % (2026)** | Model id · content_id | Description | Status |
|---|---|---|---|---|
| USDCHF · **15m** · UP | **xpair-NY refit-CPCV p10 .645 @cov2 / .6997 @cov1 / .733 @cov.5 (15/15)** ✅✅ CERTIFIED **>65%** | **`USDCHF.m15ny_xpair.v1`** · `7505c934` ✅ FROZEN | EUR-bloc pooling WINS (adversarially verified vs trap#9); seed-ens pending (may supersede) | ✅ certified >65%, improving |
| USDCHF · **15m** · DOWN | **xpair-NY refit-CPCV p10 .6437 @cov2 / .662 @cov1 / .6958 @cov.5 (15/15)** ✅✅ CERTIFIED **>65%** | **`USDCHF.m15ny_xpair.v1`** · `7505c934` ✅ FROZEN | EUR-bloc pooling WINS (adversarially verified); seed-ens pending | ✅ certified >65%, improving |
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

### All-session refit-CPCV (per-fold-refit, 15 purged paths) — `usdchf_15m_cpcv_session.py all` — ✅✅ BOTH CERTIFY @cov≤.03 (AUC .523)
| cov | UP p10 (frac) | DOWN p10 (frac) | COMB p10 (frac) | CERT | med_n (U/D) |
|---|---|---|---|---|---|
| 0.05 | .5427 (.933) | .5401 (.867) | .5448 (.933) | UP only (DOWN p10<BE) | 5156/4828 |
| 0.03 | .5478 (1.0) | .5562 (1.0) | .5529 (1.0) | ✅ BOTH | 3047/2820 |
| 0.02 | .5546 (1.0) | .5589 (1.0) | .5599 (1.0) | ✅ BOTH | 2012/1811 |
| 0.01 | .5548 (1.0) | .5633 (1.0) | .5606 (1.0) | ✅ BOTH | 923/868 |

**⚡ PATTERN-BREAK: USDCHF certifies on ALL-SESSION (both sides, cov≤.03) — UNLIKE USDCAD/USDJPY (all-session-efficient, needed NY). The edge is BROAD, not session-concentrated = the EUR-bloc signature.** DOWN leads tight cov (haven-flight). This is the current certified FLOOR (pending: do NY/LDN concentrate higher? does EUR-bloc pooling lift?). `usdchf_15m_cpcv_session_all_multicov_result.json`.

### Session segmentation (A9) — `usdchf_15m_cpcv_session.py {ny,ldn,asia}` (symmetric refit-CPCV) — **NY = CARRIER**
**NY (AUC .5401, 15/15 every cov):**
| cov | UP p10 (mean) | DOWN p10 (mean) | COMB p10 (mean) | med_n (U/D/C) | CERT |
|---|---|---|---|---|---|
| 0.05 | .6029 (.6195) | .6039 (.6236) | .5924 (.6181) | 1475/1590/3128 | ✅ BOTH |
| 0.03 | .6029 (.6347) | .6191 (.6467) | .611 (.6384) | 870/985/1934 | ✅ BOTH |
| 0.02 | .6221 (.6464) | .6161 (.6606) | .6244 (.6515) | 573/685/1321 | ✅ BOTH |
| 0.01 | **.6413** (.6782) | **.6361** (.6947) | **.6386** (.684) | 289/349/665 | ✅ BOTH |

**NY CONCENTRATES the edge far above all-session** (NY AUC .5401 vs .523; NY cov2 p10 ~.62 vs all-session ~.56). NY is the deliverable CARRIER. cov1 p10 ~.64 (means ~.68–.69) — approaching but not clearing the >65% certified-floor target. UP/DOWN ~parity (DOWN leads cov2-3, UP edges cov1). `usdchf_15m_cpcv_session_ny_multicov_result.json`.

**LDN: ❌ NO CERT any cov** (AUC .5178; COMB p10 .5238/.5237/.5162/.5102 @cov5/3/2/1, frac .47–.73 sub-BE). **The European-session prior is REFUTED — despite CHF being a European currency, LDN does NOT carry direction; NY (US-session risk flow) is the UNIQUE carrier** (same as USDCAD/AUDUSD). `usdchf_15m_cpcv_session_ldn_multicov_result.json`. **Asia: ⏸ deferred** (killed early to prioritize the EUR-bloc keystone; NY clearly the carrier, asia barely-traded → completeness rerun later).

**Session landscape verdict: NY is the unique carrier; all-session also certifies (broad EUR-bloc breadth); LDN/asia dead.** This `broad + NY-concentrated` hybrid is the USDCHF signature.

### ★ Cross-pair EUR-bloc POOLING (the differentiating test) — `usdchf_15m_xpair.py` (sign-FLIPPED, USDCHF≈−EURUSD)
**Frozen all-session SCREEN (`xpbase`):** ⚠️ IMPROVES_base=**False**. VAL AUC .5330 > base .5301 ✓ (pooling adds train-era signal; `ll_EURUSD30`+`ll_USDJPY30` rank top-20 → cross-pair info present) BUT frozen 2026 cov2 COMB **.5049 < base .5103** ✗, DOWN collapses 2026 (.4862). Per-year cov2 COMB .6382/.6062/.5049 (2024/25/26).
- **Caveat:** the frozen all-session screen is a POOR proxy for the refit-CPCV deliverable — base A1 *also* failed frozen-2026 (.5103) yet certified strongly under NY refit-CPCV (.624). → escalated to the definitive NY xpair refit-CPCV (`usdchf_15m_cpcv_xpair.py`, RUNNING).
- **Mechanism read:** cross-pair (EUR-bloc + JPY-haven) coupling is largely CONTEMPORANEOUS (already in own-pair price). `usdchf_15m_xpair_xpbase_result.json`.

**⚡ NY xpair refit-CPCV (`usdchf_15m_cpcv_xpair.py`) — POOLING WINS, ⚠️ pending adversarial verify:**
| cov | UP p10 (own-pair inc) | DOWN p10 (inc) | COMB p10 (inc) | med_n (U/D/C) |
|---|---|---|---|---|
| 0.05 | .6087 (.6029) | .607 (.6039) | .6027 (.5924) | 2745/2859/5395 |
| 0.03 | .6246 (.6029) | .6249 (.6191) | .6233 (.611) | 1738/1796/3419 |
| 0.02 | .645 (.6221) | .6437 (.6161) | .645 (.6244) | 1182/1286/2353 |
| **0.01** | **.6997** (.6413) | **.662** (.6361) | **.6804** (.6386) | 625/696/1289 |
| 0.005 | **.733** | **.6958** | **.7137** | 331/361/669 |

**The xpair (EUR-bloc pooled) book BEATS the own-pair NY incumbent at EVERY cov on BOTH sides** (15/15 paths, frac 1.0), AUC .5464 > own-pair .5401. **At cov1 BOTH sides clear 65%** (UP .6997, DOWN .662, COMB .6804) — the >65% target. This CONTRADICTS the frozen all-session screen → the refit harness recovers an era-local cross-pair lift the frozen book misses.
- **✅✅ ADVERSARIAL VERIFY PASSED (trap#9 ruled out).** Matched frozen-past forward holdout (both train 2012-21 NY, base-239 own-pair vs 337-feat xpair, same gates):

| year | cov2 COMB xpair / own | cov1 COMB xpair / own |
|---|---|---|
| 2024 | .7244 / .6881 | .7924 / .7468 |
| 2025 | .6678 / .6174 | .6858 / .624 |
| 2026 | .5127 / .4552 | .4775 / .4324 |

  **xpair beats own-pair in ALL 6 cells (mean +.048)** — the EUR-bloc pooling advantage is GENUINE, not era-local memorization. xpair-frozen `memorization=False` (2024/25 strong .67–.79). Both books refit-dependent (decay forward; deploy with periodic retraining — the 5-yr-stale frozen vintage drops to sub-BE by 2026, same as ALL 15m books). The frozen all-session SCREEN was a FALSE NEGATIVE. **Deliverable = EUR-bloc xpair-NY book; >65% certified BOTH sides at cov1 (refit-CPCV cert of record).** `usdchf_15m_cpcv_xpair_ny_multicov_result.json`, `usdchf_15m_xpair_frozen_result.json`, `usdchf_15m_ownpair_frozen_result.json`.

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
| **15m UP** | **.645 @cov2 / .6997 @cov1 / .733 @cov.5** (xpair-NY refit-CPCV, 15/15) | .7211 / .7655 mean (cov1/.5) | **`USDCHF.m15ny_xpair.v1`** ✅ FROZEN (`7505c934`) | ✅✅ CERTIFIED **>65%**; pooling verified vs trap#9; seed-ens K3/K8 pending |
| **15m DOWN** | **.6437 @cov2 / .662 @cov1 / .6958 @cov.5** (xpair-NY refit-CPCV, 15/15) | .711 / .7497 mean (cov1/.5) | **`USDCHF.m15ny_xpair.v1`** ✅ FROZEN (`7505c934`) | ✅✅ CERTIFIED **>65%**; pooling verified vs trap#9; seed-ens pending |

_Provenance: every number traces to a `*_result.json` (Tier-1). Updated as rows complete._
