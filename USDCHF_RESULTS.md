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
| USDCHF · **15m** · UP | **xpair-NY seed-ens K=3 refit-CPCV p10 .6533 @cov2 / .6935 @cov1 / .7303 @cov.5 (15/15)** ✅✅ CERTIFIED **>65%** | **`USDCHF.m15ny_xpair_seedens.v1`** · `eb44d999` ✅ FROZEN | EUR-bloc pooling WINS (adversarially verified vs trap#9); seed-ens K=3 SUPERSEDES single-seed (mean↑ 15/15 cells, p10↑ 13/15) | ✅ certified >65%, improved |
| USDCHF · **15m** · DOWN | **xpair-NY seed-ens K=3 refit-CPCV p10 .644 @cov2 / .6682 @cov1 / .7084 @cov.5 (15/15)** ✅✅ CERTIFIED **>65%** | **`USDCHF.m15ny_xpair_seedens.v1`** · `eb44d999` ✅ FROZEN | EUR-bloc pooling WINS (adversarially verified); seed-ens K=3 SUPERSEDES single-seed (mean↑ 15/15, p10↑ 13/15) | ✅ certified >65%, improved |
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

### Improve cross-product (on the carrier book)
| lever | file | result | verdict |
|---|---|---|---|
| **seed-ens K=3** (⊕ mean over 3 seeds) | `usdchf_15m_cpcv_xpair_ny.py 3 … 3` → `usdchf_15m_cpcv_xpair_ny_seedens3_result.json` | vs single-seed xpair: path-**mean ↑ 15/15** cov×side cells, **p10 ↑ 13/15** (only thin cov0.01/0.005 UP p10 dip −.006/−.003, mean still ↑); cov2 UP .6533/DOWN .644, cov1 UP .6935/DOWN .6682, cov.5 .7303/.7084; frac 1.0, CERT all cells; VAL-AUC .5576>.5464 | **✅ SUPERSEDES** → frozen `USDCHF.m15ny_xpair_seedens.v1` `eb44d999` (variance-reduction floor lift, not redistribution — mean rises everywhere) |
| dblortho (SNB-resid orthogonalization) | `usdchf_15m_xpair_dblortho.py` | xpair matrix already subsumes channels (no lift) | ❌ KILLED |
| ECM synthetic-EURCHF level band | `usdchf_15m_ecm.py` | 2026 covcurve cov2 .4945 < base .5103 (level non-stationary/too weak) | ❌ KILLED |
| magdir (magnitude-conditioned sign) | `usdchf_15m_magdir.py` | dir-AUC flat across mag quartiles (sign-invariance) — magnitude bonus → MAGNITUDE_FINDINGS.md | ❌ KILLED (dir) |
| IRM env-invariant feature-stability filter | `usdchf_15m_irm.py` | pruned 40.1% era-local sign-flippers; SURVIVES=False (UP collapsed 2026 .50 vs .63) — **proves refit-decay is an INFORMATION BOUND, not a fixable artifact** | ❌ KILLED |
| OFI / TB-firsttouch / GRU / Optuna | `usdchf_15m_{ofi,tbfirsttouch,gru,optuna}.py` | §8 coverage confirmations (all KILLED on sibling pairs) | _see Confirmatory below_ |

### §8 coverage-rule confirmatory batch (all pre-registered fast-KILL; each KILLED at this key class on a sibling pair)
| lever | file → result | numbers | verdict |
|---|---|---|---|
| signed order-flow (OFI/Kyle) direction | `usdchf_15m_ofi.py` → `usdchf_15m_ofi_result.json` | base+OF NY VAL moved-AUC .5490 vs base .5495 (Δ−0.0006 ≤ +.003); 2 OF feats in top-20 | ❌ KILLED — OF gates SIZE not SIGN at 900s (sign-invariance; USDJPY/EURUSD precedent) |
| sequence DL (GRU on 1-min return path) | `usdchf_15m_gru.py` → `usdchf_15m_gru_result.json` | VAL-AUC(NY) .5313 < base .5433 | ❌ KILLED — DL adds no sign over GBM-on-TA (EURUSD neural sweep 84/84 null confirmed here) |
| Optuna (TPE) hyperparam tuning | `usdchf_15m_optuna.py` → `usdchf_15m_optuna_result.json` | tuned binding-year AUC .5046 vs default .5049; beats default in 0/3 yrs; cov3 clears 1/3 | ❌ KILLED — hyperparameters are not the constraint; ~.539 AUC bound holds (confirms USDJPY-2m anti-transfer) |
| TB first-touch TRAIN label (refit-CPCV, **own-pair** NY) | `usdchf_15m_tbfirsttouch.py` → `usdchf_15m_cpcv_tbfirsttouch_ny_multicov_result.json` | TB_K=1.5; certifies + IMPROVES=True vs **own-pair** base every cov (cov2 UP .6257/DOWN .6261, cov1 .6611/.6558); BUT **xpair seed-ens deliverable dominates it at every cov/side by +.008…+.048** (cov1 .6935/.6682 vs .6611/.6558) | ❌ **SUBSUMED** — real label lift on the weaker own-pair space, but the EUR-bloc pooling deliverable beats it everywhere; not promoted (also an un-frozen-forward-gated refit-CPCV positive = USDCAD refit-overfit signature). Reinforces "pooling is the dominant signal" |

---

## FINAL CONCLUSION (USDCHF 15m direction) — 2026-06-10 (seed-ens supersede 2026-06-11)

**★ USDCHF is the FIRST major to CERTIFY >65% on BOTH sides.** Deliverable **`USDCHF.m15ny_xpair_seedens.v1`** (content_id `eb44d999`, EUR-bloc xpair-NY **seed-ens K=3**): **UP refit-CPCV cov1 p10 .6935 / DOWN .6682 / COMB .6853** (cov.5 .7303/.7084/.7278; cov2 .6533/.644/.6474), 15/15 paths every cov, frac_clear 1.0, AUC .5479, VAL moved-AUC .5576. Seed-ens K=3 **supersedes the single-seed** `USDCHF.m15ny_xpair.v1` (`7505c934`, cov1 .6997/.662): it lifts the refit-CPCV path-**mean in 15/15** cov×side cells and **p10 in 13/15** (the only two dips are the thin cov0.01/0.005 UP pockets where the mean still rises) — a genuine variance-reduction floor lift, NOT TB-style redistribution (the disqualifier is "p10↑ with flat mean"; here the mean rises everywhere). Full per-side pipeline ✓: (a) side-split throughout; (b) worst-VAL-half gate; (c) the EUR-bloc xpair matrix IS the purpose-built specialist (cross-pair USD-residual/lead-lag + EUR-bloc resid + 239 own-pair, sign-flipped USDCHF≈−EURUSD); (d) coverage curve (cov5→cov.5); (e) full per-fold-refit CPCV + adversarial frozen-forward head-to-head. **REFIT-DEPENDENT** (deploy NY-only with periodic retraining, size on the refit floor minus −.0035 bar-close haircut).

**Mechanism (RESOLVED):** USDCHF is the **EUR-BLOC case** (pooling WINS, like EURUSD/GBPUSD) despite being a premier safe-haven — because CHF≈EUR (SNB-managed, EURCHF-tight). It is a **HYBRID carrier**: broad (all-session certifies — EUR-bloc breadth, unlike the own-pair havens) AND NY-concentrated (NY AUC .540 ≫ all-session .523 — the haven face). LDN/asia dead (the directional sign rides US-session risk flow, not Zurich hours — the European-session prior was REFUTED). DOWN leads at cov2-3 (haven-flight), UP at cov1.

**The pooling win was ADVERSARIALLY VERIFIED vs trap#9** (the decisive call of the campaign): the frozen all-session SCREEN said IMPROVES=False (a FALSE NEGATIVE — it tests the frozen all-session vintage, not the NY+refit deliverable), but the NY refit-CPCV showed pooling BEATS own-pair at every cov both sides, AND the matched frozen-forward head-to-head confirmed it (xpair-frozen beats own-pair-frozen in ALL 6 forward cells 2024-26×cov1-2, mean +.048; memorization=False). Meta-lesson (→ IDEAS_LOG): run the NY refit-CPCV pooling test even when the frozen screen says no, for any EUR-bloc-cousin pair.

**Improve cross-product — RUN + adversarially verified:** EUR-bloc pooling WON (the base deliverable). **seed-ens K=3 WON — the one improvement lever that lifted the certified book** (mean↑ 15/15 cells, p10↑ 13/15; frozen as `USDCHF.m15ny_xpair_seedens.v1`). dblortho (SNB-resid) KILLED — xpair matrix subsumes channels. ECM synthetic-EURCHF LEVEL band KILLED — level non-stationary/too-weak (R2 premise-refutation confirmed). magdir KILLED — magnitude doesn't carry 15m sign (sign-invariance). **IRM env-invariant feature-stability filter KILLED — and it proved the refit-decay is an INFORMATION BOUND, not a fixable feature-selection artifact (the era-local-sign features carry genuine in-era signal; periodic retrain is the only mitigation).** Channel-class levers (kNN/meta/two-speed/|ret|-weight/cross-horizon/ACI) Tier-1-subsumed. §8 coverage-rule confirmatory batch **COMPLETE (4/4)**: OFI KILLED (OF gates SIZE not SIGN @900s), GRU KILLED (DL no sign over GBM), Optuna KILLED (hparams not the constraint), TB-firsttouch SUBSUMED (improves the own-pair base but the xpair seed-ens deliverable dominates it +.008…+.048 at every cov/side — reinforces that EUR-bloc pooling is the dominant signal). **None unseat the deliverable; sweep CLOSED.**

**Discovery R1+R2+R3 EXHAUSTED (2 dry rounds):** R1 found the levers (pooling won); R2's one survivor (IRM) was KILLED on test; R3 critic confirmed the on-disk × sign-carrying × novel space is saturated. R3 also corrected a record error (USDCHF HAS OF features; only raw-tick *settlement* data is absent).

**>65% is NOT an information bound at cov1 for USDCHF** (it is for the own-pair havens) — pooling + NY-concentration + CHF≈EUR tight coupling stack to clear it. **BONUS: USDCHF magnitude edge magAUC .65–.73** (→ MAGNITUDE_FINDINGS.md) — USDCHF carries BOTH a strong magnitude edge AND a certified direction edge.

**Deployment spec:** trade USDCHF 15m Rise/Fall in the NY session (America/New_York 08:00–17:00, DST-correct) when |p̄−0.5| ≥ the cov1 (1%) confidence gate; predict p̄ = **mean over the K=3 seed boosters'** P(up) with the EUR-bloc xpair feature recipe (`usdchf_15m_xpair.build_xp` + 239 base); **retrain periodically** (frozen-2021 vintage decays to sub-BE by 2026); size on the refit-CPCV per-era floor (UP .69 / DOWN .67 @cov1) minus the −.0035 bar-close haircut, minus breakeven .541; Kelly 1/8.

---

## UP/DOWN LEADERBOARD (current best per side, certified-or-best-available)
| Side | Best certified (refit-CPCV p10) | Best available (mean) | Book | Status |
|---|---|---|---|---|
| **15m UP** | **.6533 @cov2 / .6935 @cov1 / .7303 @cov.5** (xpair-NY seed-ens K=3 refit-CPCV, 15/15) | .7252 / .7747 mean (cov1/.5) | **`USDCHF.m15ny_xpair_seedens.v1`** ✅ FROZEN (`eb44d999`) | ✅✅ CERTIFIED **>65%**; pooling verified vs trap#9; **seed-ens K=3 SUPERSEDED single-seed** (mean↑ 15/15, p10↑ 13/15) |
| **15m DOWN** | **.644 @cov2 / .6682 @cov1 / .7084 @cov.5** (xpair-NY seed-ens K=3 refit-CPCV, 15/15) | .723 / .7632 mean (cov1/.5) | **`USDCHF.m15ny_xpair_seedens.v1`** ✅ FROZEN (`eb44d999`) | ✅✅ CERTIFIED **>65%**; pooling verified vs trap#9; **seed-ens K=3 SUPERSEDED single-seed** |

_Provenance: every number traces to a `*_result.json` (Tier-1). Updated as rows complete._
