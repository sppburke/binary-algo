> **SCOPE: USDCHF** (key-specific — results of record + UP/DOWN leaderboard). Generic methods: docs/METHODS_CATALOG.md. Sweep menu: SWEEP_MATRIX.md. See REPO_MAP.md. Bootstrapped 2026-06-10 (new currency; structure copied from USDCAD_RESULTS.md). Current scope: **15m only** (deriv-FX-deployable floor + program's best direction horizon).

# USDCHF — Results Ledger (unique key: currency × timeframe × side)

**Pair tag: `USDCHF`.** USDCHF is a **USD-BASE / CHF-QUOTE** pair (USDCHF up ⇒ USD strengthens vs CHF ⇒ "USD up / CHF down"). CHF is **THE premier safe-haven currency** (more than JPY). It is **doubly determined** and the central modelling question is which face dominates:
- **SAFE-HAVEN (USDJPY analog).** Risk-off ⇒ CHF bid ⇒ USDCHF **DOWN** (the sharp directional move = flight-to-safety / CHF strength). USD on the NUMERATOR like USDJPY. If this face dominates, USDCHF is **own-pair-specific + NY/risk-concentrated**, pooling DILUTES (the USDJPY/AUDUSD/USDCAD base-rate: 3/3 certified safe-haven/USD-base majors were own-pair-specific).
- **EUR-BLOC / SNB-MANAGED (EURUSD analog) — THE DIFFERENTIATING HYPOTHESIS.** CHF tracks EUR extremely tightly: the SNB ran a EURCHF≥1.20 floor 2011–2015 and still manages the franc (sight-deposit intervention + negative-rate era), so **USDCHF ≈ −EURUSD modulated by a slow EURCHF** — USDCHF is strongly (inversely) correlated to EURUSD. EURUSD and GBPUSD both **CERTIFIED via cross-pair EUR-BLOC POOLING** (own-pair was beaten). **IF the EUR-bloc face dominates CHF, cross-pair pooling could WIN here too — UNLIKE the safe-haven analogs.** Run cross-pair pooling EARLY, **sign-FLIPPED** (USDCHF moves OPPOSITE to EURUSD). This is the single most informative early experiment for the key.
- **SNB INTERVENTION REGIME (CHF-unique invent-lever).** Official one-directional flow (sight deposits / FX purchases) can impose persistent drift / regime structure no other major has. A SNB-regime gate is a bespoke discovery lever (off-disk SNB sight-deposit data = acquisition frontier). NB: 2015-01-15 SNB floor-removal crash is gap-excluded by the contiguity check.
- **SESSION.** CHF is a EUROPEAN (Zurich) currency → trades heavily in the LONDON/European window, so the **LDN-session prior is materially HIGHER here than for JPY/CAD/AUD** (hygiene: ldn≈33% of moved bars vs ny≈37%, asia≈38%). NY still strong (US risk events drive haven flow); Asia low.

**Side-asymmetry prediction (falsifiable).** The sharp, news-driven directional move in CHF is the risk-off flight = CHF strength = USDCHF **DOWN**. **Prediction: USDCHF DOWN (haven flight) is the more forecastable side.** BUT if the EUR-bloc face dominates, the side that pools best with EURUSD may lead instead — the side-split + pooling test decide; v0 is a hypothesis, not a conclusion.

**The unique result key is `(currency, timeframe, side)`, side ∈ {UP, DOWN}.** Where a key is unmeasured it is `UNTESTED`. COMBINED is context only, never an UP/DOWN number.

**Conventions.** Deriv-faithful settlement (bar-close approx at 15m, mid-to-mid, next-tick entry +1s, ties LOSE), breakeven **0.541** (deriv payout R≈1.85). Splits: bars train 2012-21 / val 2022-23 / test 2024 / test 2025 / oos 2026 (strict OOS). Selection on VAL worst-half (never VAL-acc-max; `corr(VAL,OOS)=−0.54`). Moved-bars-only; verify moved up-rate ∈ [0.47,0.53] (**USDCHF checked clean: .5034/.4996/.5041 for 2024/2025/2026, flat_frac <1.3%** — no fake-flat mirage). Per-year CI95 (bootstrap). de-overlap = `nonoverlap_chrono` gap=900s. CERTIFY a side ONLY via full per-fold-refit CPCV at the operating gate: p10 ≥ 0.541 AND ≥ ~80% of 15 purged paths clear 0.541; adversarially verify every positive (frozen-forward, trap#9). **Target: >65% accuracy** (achievable, if at all, only at tight coverage — neighbors hit ~.60–.65 at cov1–2%). Bar features on disk 2012–2026. **Tick data note (corrected 2026-06-10):** USDCHF HAS on-disk order-flow modeling features (`features_of/USDCHF_*`, `features_tick_xofi/USDCHF_*cks1s`); what is absent is RAW-tick data for true-tick SETTLEMENT validation → bar-close proxy (−.0035, validated on USDJPY/AUDUSD). Last updated 2026-06-11 (**SWEEP CLOSED** — deliverable `USDCHF.m15ny_xpair_seedens.v1` `eb44d999` certified >65% BOTH sides; §8 confirmatory 4/4 done; on-disk exhausted).

---

## MASTER KEY TABLE — one row per (USDCHF, timeframe, side)

> **Deriv-feed program (2026-11-15):** the numbers in this table are Dukascopy-era evidence. On the Deriv feed (DEV 2025-08-11..2026-07-31) none of this pair's frozen 15m/30m books is SUPPORTED as deployed — see "USDCHF — DERIV-FEED PROGRAM" at the end of this file.

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
| A1 | `usdchf_15m_base.py` | moved-AUC .5288 / .5205 / .5129; gate cov2% COMB wr .6513 / .5675 / .5103; VAL moved-AUC .5301 | base floor (own-pair 239-feat); ✅ certifies under NY refit-CPCV → superseded by xpair-NY seed-ens deliverable |
| R9 | issue #9 date-only retrospective replay: A=current v1; B=clean capped legacy-date refit; C=clean capped date-refresh refit | Apr–May 2026 standardized whole-book correctness: A .5660 (n=53, yield .000316); B .4630 (n=54, −.000181); C .6250 (n=72, .000813). C UP .6765 (n=34); DOWN .5789 (n=38). | ⚪ **INCONCLUSIVE — `control_not_estimable`.** The positive point estimate is not promotable: simultaneous lower bounds are C−A yield −.000257, C−B yield +.000159, C-UP accuracy−.5 −.055559, C-DOWN accuracy−.5 −.223734. No survivor/candidate; incumbent unchanged. Pair: `results/json/m15_book_refresh_53547498c599f2877a2f6616ae725f8f27e11070e3b550e82eb26ec039fb23f6_USDCHF_replay_result.json`; joint/shadow: `results/json/m15_book_refresh_53547498c599f2877a2f6616ae725f8f27e11070e3b550e82eb26ec039fb23f6_joint_replay_result.json`, `results/json/m15_book_refresh_53547498c599f2877a2f6616ae725f8f27e11070e3b550e82eb26ec039fb23f6_shadow_spec_result.json`. |
| R18 | issue #18 latest-on-disk combined-policy refit package | **No efficacy measurement.** Exact result: `results/json/usdchf_m15_current_refit_v1_result.json`; seal `c955ab958c536b06f18830aea864324afaccdbd6b5c84e31559717cd900ea778`; terminal `16b1fd46951248bf3f5120fbdc164808aa6b9edbca9ed9dc18e2b22ea5ac661e`. | ⚪ **`INACTIVE_UNTESTED_PROSPECTIVE`.** Canonical/audit bytes match; package `results/artifacts/USDCHF.m15ny_xpair_seedens.r202605.v1/` is unpublished, absent from book/runtime registries, and changes no UP/DOWN key or incumbent. |

### Coverage curve (step d) — `usdchf_15m_base.py` covcurve (frozen-2012-21 book; per-year, per-cov thr)
| cov | COMB wr 2024/25/26 | UP wr 2024/25/26 | DOWN wr 2024/25/26 |
|---|---|---|---|
| 0.02 | .6277 / .573 / .504 | .601 / .5646 / .5269 | .6655 / .584 / .4793 |
| 0.01 | .6703 / .5858 / .5486 | .6569 / .565 / .5897 | .6914 / .6148 / .5092 |

### All-session refit-CPCV (per-fold-refit, 15 purged paths) — `usdchf_15m_cpcv_session.py all` — ✅✅ BOTH CERTIFY @cov≤.03 (AUC .523)
| cov | UP p10 (frac) | DOWN p10 (frac) | COMB p10 (frac) | CERT | med_n (U/D) |
|---|---|---|---|---|---|
| 0.05 | .5427 (.933) | .5401 (.867) | .5448 (.933) | UP only (DOWN p10<BE) | 5156/4828 |
| 0.03 | .5478 (1.0) | .5562 (1.0) | .5529 (1.0) | ✅ BOTH | 3047/2820 |
| 0.02 | .5546 (1.0) | .5589 (1.0) | .5599 (1.0) | ✅ BOTH | 2012/1811 |
| 0.01 | .5548 (1.0) | .5633 (1.0) | .5606 (1.0) | ✅ BOTH | 923/868 |

**⚡ PATTERN-BREAK: USDCHF certifies on ALL-SESSION (both sides, cov≤.03) — UNLIKE USDCAD/USDJPY (all-session-efficient, needed NY). The edge is BROAD, not session-concentrated = the EUR-bloc signature.** DOWN leads tight cov (haven-flight). This was the first certified FLOOR (since superseded: NY concentrates the edge AND EUR-bloc pooling lifts it — both confirmed below). `usdchf_15m_cpcv_session_all_multicov_result.json`.

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
- **Caveat:** the frozen all-session screen is a POOR proxy for the refit-CPCV deliverable — base A1 *also* failed frozen-2026 (.5103) yet certified strongly under NY refit-CPCV (.624). → escalated to the definitive NY xpair refit-CPCV (`usdchf_15m_cpcv_xpair.py`) — POOLING WINS (below).
- **Mechanism read:** cross-pair (EUR-bloc + JPY-haven) coupling is largely CONTEMPORANEOUS (already in own-pair price). `usdchf_15m_xpair_xpbase_result.json`.

**⚡ NY xpair refit-CPCV (`usdchf_15m_cpcv_xpair.py`) — POOLING WINS, ✅ adversarially verified (trap#9 ruled out, below):**
| cov | UP p10 (own-pair inc) | DOWN p10 (inc) | COMB p10 (inc) | med_n (U/D/C) |
|---|---|---|---|---|
| 0.05 | .6087 (.6029) | .607 (.6039) | .6027 (.5924) | 2745/2859/5395 |
| 0.03 | .6246 (.6029) | .6249 (.6191) | .6233 (.611) | 1738/1796/3419 |
| 0.02 | .645 (.6221) | .6437 (.6161) | .645 (.6244) | 1182/1286/2353 |
| **0.01** | **.6997** (.6413) | **.662** (.6361) | **.6804** (.6386) | 625/696/1289 |
| 0.005 | **.733** | **.6958** | **.7137** | 331/361/669 |

**The xpair (EUR-bloc pooled) book BEATS the own-pair NY incumbent at EVERY cov on BOTH sides** (15/15 paths, frac 1.0), AUC .5464 > own-pair .5401. **At cov1 BOTH sides clear 65%** (UP .6997, DOWN .662, COMB .6804) — the >65% target. This CONTRADICTS the frozen all-session screen → the refit harness recovers an era-local cross-pair lift the frozen book misses. _(These are the **single-seed** xpair numbers — the pooling test of record; the deployed deliverable is the **seed-ens K=3** book `USDCHF.m15ny_xpair_seedens.v1` which superseded this, cov1 .6935/.6682 — see Improve cross-product + FINAL CONCLUSION below.)_
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
| magdir (magnitude-conditioned sign) | `usdchf_15m_magdir.py` | dir-AUC flat across mag quartiles (sign-invariance) — magnitude bonus → docs/MAGNITUDE_FINDINGS.md | ❌ KILLED (dir) |
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

**The pooling win was ADVERSARIALLY VERIFIED vs trap#9** (the decisive call of the campaign): the frozen all-session SCREEN said IMPROVES=False (a FALSE NEGATIVE — it tests the frozen all-session vintage, not the NY+refit deliverable), but the NY refit-CPCV showed pooling BEATS own-pair at every cov both sides, AND the matched frozen-forward head-to-head confirmed it (xpair-frozen beats own-pair-frozen in ALL 6 forward cells 2024-26×cov1-2, mean +.048; memorization=False). Meta-lesson (→ docs/IDEAS_LOG.md): run the NY refit-CPCV pooling test even when the frozen screen says no, for any EUR-bloc-cousin pair.

**Improve cross-product — RUN + adversarially verified:** EUR-bloc pooling WON (the base deliverable). **seed-ens K=3 WON — the one improvement lever that lifted the certified book** (mean↑ 15/15 cells, p10↑ 13/15; frozen as `USDCHF.m15ny_xpair_seedens.v1`). dblortho (SNB-resid) KILLED — xpair matrix subsumes channels. ECM synthetic-EURCHF LEVEL band KILLED — level non-stationary/too-weak (R2 premise-refutation confirmed). magdir KILLED — magnitude doesn't carry 15m sign (sign-invariance). **IRM env-invariant feature-stability filter KILLED — and it proved the refit-decay is an INFORMATION BOUND, not a fixable feature-selection artifact (the era-local-sign features carry genuine in-era signal; periodic retrain is the only mitigation).** Channel-class levers (kNN/meta/two-speed/|ret|-weight/cross-horizon/ACI) Tier-1-subsumed. §8 coverage-rule confirmatory batch **COMPLETE (4/4)**: OFI KILLED (OF gates SIZE not SIGN @900s), GRU KILLED (DL no sign over GBM), Optuna KILLED (hparams not the constraint), TB-firsttouch SUBSUMED (improves the own-pair base but the xpair seed-ens deliverable dominates it +.008…+.048 at every cov/side — reinforces that EUR-bloc pooling is the dominant signal). **None unseat the deliverable; sweep CLOSED.**

**Discovery R1+R2+R3 EXHAUSTED (2 dry rounds):** R1 found the levers (pooling won); R2's one survivor (IRM) was KILLED on test; R3 critic confirmed the on-disk × sign-carrying × novel space is saturated. R3 also corrected a record error (USDCHF HAS OF features; only raw-tick *settlement* data is absent).

**>65% is NOT an information bound at cov1 for USDCHF** (it is for the own-pair havens) — pooling + NY-concentration + CHF≈EUR tight coupling stack to clear it. **BONUS: USDCHF magnitude edge magAUC .65–.73** (→ docs/MAGNITUDE_FINDINGS.md) — USDCHF carries BOTH a strong magnitude edge AND a certified direction edge.

**Deployment spec:** trade USDCHF 15m Rise/Fall in the NY session (America/New_York 08:00–17:00, DST-correct) when |p̄−0.5| ≥ the cov1 (1%) confidence gate; predict p̄ = **mean over the K=3 seed boosters'** P(up) with the EUR-bloc xpair feature recipe (`usdchf_15m_xpair.build_xp` + 239 base); **retrain periodically** (frozen-2021 vintage decays to sub-BE by 2026); size on the refit-CPCV per-era floor (UP .69 / DOWN .67 @cov1) minus the −.0035 bar-close haircut, minus breakeven .541; Kelly 1/8.

---

## UP/DOWN LEADERBOARD (current best per side, certified-or-best-available)
| Side | Best certified (refit-CPCV p10) | Best available (mean) | Book | Status |
|---|---|---|---|---|
| **15m UP** | **.6533 @cov2 / .6935 @cov1 / .7303 @cov.5** (xpair-NY seed-ens K=3 refit-CPCV, 15/15) | .7252 / .7747 mean (cov1/.5) | **`USDCHF.m15ny_xpair_seedens.v1`** ✅ FROZEN (`eb44d999`) | ✅✅ CERTIFIED **>65%**; pooling verified vs trap#9; **seed-ens K=3 SUPERSEDED single-seed** (mean↑ 15/15, p10↑ 13/15) |
| **15m DOWN** | **.644 @cov2 / .6682 @cov1 / .7084 @cov.5** (xpair-NY seed-ens K=3 refit-CPCV, 15/15) | .723 / .7632 mean (cov1/.5) | **`USDCHF.m15ny_xpair_seedens.v1`** ✅ FROZEN (`eb44d999`) | ✅✅ CERTIFIED **>65%**; pooling verified vs trap#9; **seed-ens K=3 SUPERSEDED single-seed** |

_Provenance: every number traces to a `*_result.json` (Tier-1). Updated as rows complete._

## USDCHF — DERIV-FEED PROGRAM (Deriv-only data; opened 2026-11-15)

> **SCOPE:** this key on the Deriv feed only — Deriv 1-second mid ticks (`/mnt/data/deriv`), no Dukascopy data. Labels settle on Deriv ticks (entry first tick >= decision + 1 s, exit last tick <= entry target + H, ties lose); DEV window 2025-08-11..2026-07-31 for training, selection and CPCV; HOLDOUT 2026-08-01..2026-11-14 sealed (opened once per finalized candidate). Protocol and tools: `docs/DERIV_SOTA_PROGRAM.md`. Tables below are rendered from the result JSONs by `scripts/render_ledger.py` (do not edit by hand).

<!-- DERIV-FEED-NOTES:BEGIN -->
- **Incumbent benchmark on TRADEABLE decisions, books as trained (R1bt, 13:00-16:35 + 18:00-22:00 ET):** UP best `USDCHF.m15ny_xpair_seedens.v1` 0.5854 [0.4390, 0.7317] n=41 (NOT SUPPORTED); DOWN best `USDCHF.m15ny_xpair.v1` 0.6905 [0.5476, 0.8333] n=42 (NOT SUPPORTED).
- **Books as deployed today (R1t, true clock, NY 08:00-16:35, tradeable):** UP best `USDCHF.m15ny_xpair_seedens.v1` 0.5965 [0.4737, 0.7193] n=57 (NOT SUPPORTED); DOWN best `USDCHF.m15ny_xpair.v1` 0.4906 [0.3962, 0.5849] n=106 (NOT SUPPORTED).
- **DEV-CPCV certified sides, strict certificate (>= 12 of 15 paths counted and clear, p10 >= .541; valid, tradeable runs):** none yet. DEV certification is necessary, not sufficient: the sealed holdout decides (docs/DERIV_SOTA_PROGRAM.md).
- DEV strict-certificate passes that failed their survivor check (not finalists; excluded from the list above): e20sg 15m UP (cov 0.03: CERTIFIED (15/15, 14 clear); forward 1/2); `deriv_a15null_USDCHF_15m_result.json` reads NOT SIGNIFICANT vs sign-null replay (or incomplete).
- Certified only under the legacy rule, not under the strict certificate (its reading in each item): c7comp3 15m UP (cov 0.03: no (10/15, 10 clear); forward 0/0); e3union 15m UP (cov 0.03: no (5/15, 4 clear); forward 0/0).
- **DEV leaderboard 15m UP** (closest to the goal rule at each run's operating coverage): `e20sg` 14/15 clear, p10 0.551, excess +0.098; `e23fd` 13/15 clear, p10 0.531, excess +0.092; `e24cx` 12/15 clear, p10 0.512, excess +0.092.
- **DEV leaderboard 15m DOWN** (closest to the goal rule at each run's operating coverage): `e67drfe` 6/15 clear, p10 0.457, excess +0.035; `e70play` 6/14 clear, p10 0.413, excess +0.038; `e20sg` 6/14 clear, p10 0.398, excess +0.035.
- **DEV leaderboard 30m UP** (closest to the goal rule at each run's operating coverage): `e5h30` 1/11 clear, p10 0.462, excess -0.003.
- **DEV leaderboard 30m DOWN** (closest to the goal rule at each run's operating coverage): `e5h30` 1/2 clear, p10 0.471, excess +0.038.
- **Deployment status (Deriv feed):** DO NOT DEPLOY - no candidate has passed both the DEV-CPCV goal rule (or the drift amendment) and a sealed-holdout look, and deployment needs both.
- **Payout-aware reading of `e20sg` 15m** (`deriv_a18pay_USDCHF_15m_result.json`, cov 0.03; the candidate's trades frozen, each priced with the payout reconstructed for its decision from the July 2026 quote archive - an estimate resting on the assumptions in docs/DERIV_SOTA_PROGRAM.md "Venue payouts", not a venue record; averages are over the counted paths): UP NOT CLEARED AT THE QUOTED PAYOUT, ADVERSELY PRICED: expected value per unit staked above 0 on 6 of 15 counted paths, p10 -0.0550, mean +0.0219; win rate 0.5966 at mean payout 1.7203 (mean breakeven faced 0.5826; always-UP payout 1.7898, expected value -0.1080); the lower-paid side was bought on 0.879 of the trades; 0 of 856 trades over the 15 paths not priced (clock cells with too few quotes in the archive); DOWN NOT CLEARED AT THE QUOTED PAYOUT, ADVERSELY PRICED: expected value per unit staked above 0 on 4 of 14 counted paths, p10 -0.3137, mean -0.0993; win rate 0.5206 at mean payout 1.7324 (mean breakeven faced 0.5781; always-DOWN payout 1.7892, expected value -0.1320); the lower-paid side was bought on 0.854 of the trades; 0 of 570 trades over the 15 paths not priced (clock cells with too few quotes in the archive).
- **Payout-aware reading of `e67drfe` 15m** (`deriv_a29pay_USDCHF_15m_result.json`, cov 0.03; the candidate's trades frozen, each priced with the payout reconstructed for its decision from the July 2026 quote archive - an estimate resting on the assumptions in docs/DERIV_SOTA_PROGRAM.md "Venue payouts", not a venue record; averages are over the counted paths): UP NOT CLEARED AT THE QUOTED PAYOUT, ADVERSELY PRICED: expected value per unit staked above 0 on 0 of 15 counted paths, p10 -0.2439, mean -0.1639; win rate 0.4843 at mean payout 1.7279 (mean breakeven faced 0.5793; always-UP payout 1.7898, expected value -0.1080); the lower-paid side was bought on 0.911 of the trades; 0 of 1836 trades over the 15 paths not priced (clock cells with too few quotes in the archive); DOWN NOT CLEARED AT THE QUOTED PAYOUT, ADVERSELY PRICED: expected value per unit staked above 0 on 2 of 15 counted paths, p10 -0.2169, mean -0.1032; win rate 0.5204 at mean payout 1.7242 (mean breakeven faced 0.5807; always-DOWN payout 1.7892, expected value -0.1316); the lower-paid side was bought on 0.891 of the trades; 0 of 1606 trades over the 15 paths not priced (clock cells with too few quotes in the archive).
- Superseded: `e1` and `r2` ran before the venue tradeability filter (they include untradeable blackout decisions); their tradeable re-runs are `e1t` and `r2t`.
- Robustness checks, not candidates (excluded from the lists and leaderboards above; tables below): `e20sgnc`, `e20sgs3`.
<!-- DERIV-FEED-NOTES:END -->

> **USDCHF 15m UP — drift placebo (2026-11-15, hand-written; numbers from the cited JSONs).** `e1t` (NY 08:00-16:35
> tradeable, coverage 0.03) is VALID (in-session moved up-rate 0.5090) and KILLED: UP p10 0.5292, 10 of 15 paths clear
> .541, moved AUC 0.500 over all rows. Its UP trades nevertheless beat always-UP on 14 of 15 paths (mean +0.069), and
> the drift placebo `deriv_a1placebo_USDCHF_15m_result.json` (E1t re-run with predictions dumped; rebuilt trades match
> the run exactly) finds +0.066 over always-UP baselines matched on New York hour × rv_60m tercile with 14 of 15 CPCV
> paths positive (pre-registered reading met: SKILL BEYOND DRIFT TIMING; forward folds +0.061, 3/3). The selection skill
> sits on a baseline near 0.50, so the side's absolute win rate does not reach the certification bar. DOWN: +0.008, 8 of
> 14 paths positive (no skill).

> **USDCHF 15m UP — Tier-I levers on the gate base (2026-11-15, hand-written; numbers from the cited JSONs).** Base
> `e33comp33` (LightGBM + compression gate 0.33, operating coverage 0.10): 11 of 15 paths clear, p10 0.5257, VALID.
> Pre-registered reading (lab PROGRESS.md, by 13:11 UTC): a lever IMPROVES iff paths clear >= the base's and p10 > the
> base's at the operating coverage on a VALID run. `c9compaci` (+ ACI gate) IMPROVES by that reading (11 of 15, p10
> 0.5361) but stays KILLED by the goal rule (11 < 12 paths, p10 < 0.541). It is not a holdout finalist: the placebo,
> delay and holdout tools rebuild trades with static thresholds and cannot reproduce ACI-gated trades without new code,
> and a DEV-killed configuration cannot reach certification. No improvement: `c7comp3b` (3 seeds; 11, 0.5144),
> `c8compabs` (|return| weights; 9, 0.5132), `c6nyscompb` (kernel under the gate; 5 of 14 counted, 0.4930),
> `c10compmeta` (meta gate keep 0.5, coverage doubled to 0.20 by design; 9, 0.4810). `c11compmag` (magnitude gate keep
> 0.5 on top of the compression gate) leaves no path with 25 trades (at most 18 on a path). `c6nyscomp` and `c7comp3`
> ran at coverage 0.03 by a queueing error and are recorded as coverage variants, not comparisons (`c7comp3`: 10 paths
> counted, all clear, p10 0.5697; fails the goal rule's 12 counted paths).
> Secondary-coverage observation: `e33comp33`'s own 0.05 reading passes the path test (13 counted, 13 clear, p10
> 0.5533; `deriv_e33comp33_USDCHF_15m_result.json`). It is not a certification or a finalist: the operating coverage
> 0.10 was fixed before the run; it is the only pass among 596 secondary-coverage readings on the six non-USDJPY pairs,
> so it carries that search's selection; no forward-walk fold reaches 25 trades at 0.05; and (estimate) its median 45
> trades per path (two ~8.5-week groups) implies about 40 trades on the ~15-week holdout, below n >= 50.

> **USDCHF 15m UP — `e20sg` passes the goal rule on DEV, then fails its survivor check (2026-11-16, hand-written;
> numbers from `deriv_e20sg_USDCHF_15m_result.json` and the other USDCHF 15m result JSONs).** E1t + the B7 path-signature block
> (`sg_`), coverage 0.03: 15 of 15 paths counted, 14 clear .541, p10 0.5513, VALID (moved up-rate 0.5089). It is the
> first VALID goal-rule pass outside USDJPY, and it does not pass at the neighbouring coverages: 0.02 gives 10 of 13 counted paths
> clear (p10 0.527), 0.05 gives 11 of 15 (p10 0.526). Forward walk at 0.03: 2 of 4 folds reach 25 trades, 1 clears;
> pooled, 63 of 104 trades won. Mean moved AUC 0.503. Selection: it ranks first among the 59 VALID USDCHF 15m UP runs
> with >= 12 counted paths, whose p10 averages 0.4876 (sd 0.0268); at +2.4 sd it sits about where the best of ~60
> correlated tries would land without a real effect, on a side where the E1t drift placebo above found a small
> selection skill. A survivor check was pre-registered (lab PROGRESS.md, 18:41 UTC) before any of its runs, and it
> failed at its last step. The registered no-clock ablation `e20sgnc` (15 counted, 13 clear, p10 0.5358), three seeds
> `e20sgs3` (15, 13, 0.5358) and the entry-delay stress `a14delay` (UP operationally robust: 14 of 15 clear at +4 s,
> p10 0.558; not a stale-feed artifact) passed. The sign-null replay `a15null` (`deriv_a15null_USDCHF_15m_result.json`:
> the same configuration re-run 16 times with labels shifted 3..48 weeks) did not read SIGNIFICANT. No null met the
> goal path test, but at coverage 0.03 eleven nulls left no UP path with 25 trades and five left one (p10 0.320-0.481),
> so the real p10 ranks 6 of 6 comparable values instead of the required >= 16 of 17. The replay could not build the
> null distribution its rule needs, so it neither confirms nor refutes the edge; the pre-registered reading counts that
> as a failure. `e20sg` is therefore recorded as a DEV goal-path pass that failed its survivor check: not a finalist,
> and the key stays DO NOT DEPLOY. A pass would have met a weak holdout in any case (estimate from the DEV forward
> walk's 13.4 trades a month): about 46 trades by 2026-11-14, below n >= 50; about 92 by 2027-02-26, where even the
> DEV win rate (0.597) would pass a single 0.025 look with probability about 0.17 (exact binomial, 60 of 92 needed).

<!-- DERIV-FEED-TABLES:BEGIN -->
**Frozen books on the Deriv feed — true clock, ny (America/New_York 08:00-17:00, DST-correct, sessions.session_mask), ALL decisions incl. the untradeable blackout** (`deriv_r1_frozen_books_dev_result.json`; strategy gate, tick-settled, ties lose; SUPPORTED iff n >= 50 and CI95-lo >= .541):

| Book | AUC (moved) | gate rate | UP n · WR [CI95] | DOWN n · WR [CI95] | COMB n · WR | verdict UP / DOWN |
|---|---|---|---|---|---|---|
| `USDCHF.m15ny_xpair.v1` | .5080 | .0028 | 58 · .5517 [.4310, .6724] | 106 · .4906 [.3962, .5849] | 164 · .5122 | NOT SUPPORTED / NOT SUPPORTED |
| `USDCHF.m15ny_xpair_seedens.v1` | .5088 | .0026 | 57 · .5965 [.4737, .7193] | 99 · .4848 [.3838, .5859] | 156 · .5256 | NOT SUPPORTED / NOT SUPPORTED |

**Frozen books on the Deriv feed — training clock (true UTC - 5 h), nylate (America/New_York 13:00-22:00, DST-correct), ALL decisions incl. the untradeable blackout** (`deriv_r1b_frozen_books_trainclock_nylate_dev_result.json`; strategy gate, tick-settled, ties lose; SUPPORTED iff n >= 50 and CI95-lo >= .541):

| Book | AUC (moved) | gate rate | UP n · WR [CI95] | DOWN n · WR [CI95] | COMB n · WR | verdict UP / DOWN |
|---|---|---|---|---|---|---|
| `USDCHF.m15ny_xpair.v1` | .5436 | .0098 | 218 · .7706 [.7156, .8257] | 191 · .8272 [.7749, .8796] | 359 · .7967 | SUPPORTED / SUPPORTED |
| `USDCHF.m15ny_xpair_seedens.v1` | .5437 | .0100 | 226 · .7832 [.7257, .8363] | 199 · .8342 [.7839, .8844] | 365 · .8082 | SUPPORTED / SUPPORTED |

**Frozen books on the Deriv feed — training clock (true UTC - 5 h), nylate (America/New_York 13:00-22:00, DST-correct), tradeable decisions only (Deriv purchase blackout excluded)** (`deriv_r1bt_frozen_books_trainclock_nylate_dev_tradeable_result.json`; strategy gate, tick-settled, ties lose; SUPPORTED iff n >= 50 and CI95-lo >= .541):

| Book | AUC (moved) | gate rate | UP n · WR [CI95] | DOWN n · WR [CI95] | COMB n · WR | verdict UP / DOWN |
|---|---|---|---|---|---|---|
| `USDCHF.m15ny_xpair.v1` | .5171 | .0017 | 38 · .5789 [.4211, .7368] | 42 · .6905 [.5476, .8333] | 79 · .6456 | NOT SUPPORTED / NOT SUPPORTED |
| `USDCHF.m15ny_xpair_seedens.v1` | .5173 | .0017 | 41 · .5854 [.4390, .7317] | 42 · .6667 [.5238, .8095] | 82 · .6341 | NOT SUPPORTED / NOT SUPPORTED |

**Frozen books on the Deriv feed — true clock, ny (America/New_York 08:00-17:00, DST-correct, sessions.session_mask), tradeable decisions only (Deriv purchase blackout excluded)** (`deriv_r1t_frozen_books_dev_tradeable_result.json`; strategy gate, tick-settled, ties lose; SUPPORTED iff n >= 50 and CI95-lo >= .541):

| Book | AUC (moved) | gate rate | UP n · WR [CI95] | DOWN n · WR [CI95] | COMB n · WR | verdict UP / DOWN |
|---|---|---|---|---|---|---|
| `USDCHF.m15ny_xpair.v1` | .5047 | .0029 | 58 · .5517 [.4310, .6724] | 106 · .4906 [.3962, .5849] | 164 · .5122 | NOT SUPPORTED / NOT SUPPORTED |
| `USDCHF.m15ny_xpair_seedens.v1` | .5054 | .0027 | 57 · .5965 [.4737, .7193] | 99 · .4848 [.3838, .5859] | 156 · .5256 | NOT SUPPORTED / NOT SUPPORTED |

**Per-fold-refit CPCV on the Deriv feed** (DEV 2025-08-11..2026-07-31, NY DST session, 15 paths; p10/frac = path win-rate 10th percentile / share of paths >= .541; fwd = forward-walk paths clearing .541 and their mean):

| Run | Model / features | H | cov | AUC mean (min) | UP p10 · frac · mean | UP fwd | DOWN p10 · frac · mean | DOWN fwd | always UP / DOWN | legacy verdict UP / DOWN | strict certificate UP / DOWN (paths counted) | drift amendment UP / DOWN (owner decision 1) | frac @.565 UP / DOWN | result JSON |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| c10compmeta | c10compmeta: all (66), nseed 1 | 15m | 0.2 | .5000 (.4900) | .4810 · .600 · .5557 | 0/2 · .5100 | .4459 · .273 · .5039 | 1/2 · .5338 | .4987 / .4855 | KILLED / KILLED | no (15/15, 9 clear) / no (11/15, 3 clear) | — / — | 0.40 / 0.09 | `deriv_c10compmeta_USDCHF_15m_result.json` |
| c11compmag | c11compmag: all (66), nseed 1 | 15m | 0.1 | .5000 (.4900) | nan · .000 · — | 0/0 · — | nan · .000 · — | 0/0 · — | .4987 / .4855 | KILLED / KILLED | no (0/15, 0 clear) / no (0/15, 0 clear) | — / — | — / — | `deriv_c11compmag_USDCHF_15m_result.json` |
| c6nyscomp | c6nyscomp: all (66), nseed 1 | 15m | 0.03 | .5059 (.4962) | .4361 · .500 · .5146 | 1/2 · .5530 | .4955 · .500 · .5443 | 0/0 · — | .4987 / .4855 | KILLED / KILLED | no (14/15, 7 clear) / no (4/15, 2 clear) | — / — | 0.14 / 0.25 | `deriv_c6nyscomp_USDCHF_15m_result.json` |
| c6nyscompb | c6nyscompb: all (66), nseed 1 | 15m | 0.1 | .5059 (.4962) | .4930 · .357 · .5251 | 2/4 · .5452 | .4354 · .200 · .4904 | 0/1 · .4688 | .4987 / .4855 | KILLED / KILLED | no (14/15, 5 clear) / no (10/15, 2 clear) | — / — | 0.07 / 0.20 | `deriv_c6nyscompb_USDCHF_15m_result.json` |
| c7comp3 | c7comp3: all (66), nseed 3 | 15m | 0.03 | .4999 (.4898) | .5697 · 1.0 · .6233 | 0/0 · — | nan · .000 · — | 0/0 · — | .4987 / .4855 | CERTIFIED / KILLED | no (10/15, 10 clear) / no (0/15, 0 clear) | — / — | 0.90 / — | `deriv_c7comp3_USDCHF_15m_result.json` |
| c7comp3b | c7comp3b: all (66), nseed 3 | 15m | 0.1 | .4999 (.4898) | .5144 · .733 · .5671 | 0/1 · .4706 | .4030 · .273 · .4792 | 0/0 · — | .4987 / .4855 | KILLED / KILLED | no (15/15, 11 clear) / no (11/15, 3 clear) | — / — | 0.53 / 0.18 | `deriv_c7comp3b_USDCHF_15m_result.json` |
| c8compabs | c8compabs: all (66), nseed 1 | 15m | 0.1 | .5012 (.4920) | .5132 · .600 · .5722 | 1/1 · .5833 | .3953 · .273 · .4950 | 0/1 · .5152 | .4987 / .4855 | KILLED / KILLED | no (15/15, 9 clear) / no (11/15, 3 clear) | — / — | 0.53 / 0.27 | `deriv_c8compabs_USDCHF_15m_result.json` |
| c9compaci | c9compaci: all (66), nseed 1 | 15m | 0.1 | .5000 (.4900) | .5361 · .733 · .5675 | 0/1 · .5000 | .4468 · .182 · .4953 | 0/1 · .5000 | .4987 / .4855 | KILLED / KILLED | no (15/15, 11 clear; never certified: an adaptive threshold: ACI moves it with settled test outcomes) / no (11/15, 2 clear; never certified: an adaptive threshold: ACI moves it with settled test outcomes) | — / — | 0.47 / 0.18 | `deriv_c9compaci_USDCHF_15m_result.json` |
| e10v02 | e10v02: all (82), nseed 1 | 15m | 0.03 | .5016 (.4921) | .4303 · .571 · .5404 | 0/0 · — | .3963 · .500 · .5022 | 0/0 · — | .4987 / .4855 | KILLED / KILLED | no (14/15, 8 clear) / no (6/15, 3 clear) | — / — | 0.36 / 0.33 | `deriv_e10v02_USDCHF_15m_result.json` |
| e13mag50 | e13mag50: all (66), nseed 1 | 15m | 0.03 | .5000 (.4900) | .4048 · .375 · .5048 | 0/0 · — | .3257 · .333 · .4516 | 0/0 · — | .4987 / .4855 | KILLED / KILLED | no (8/15, 3 clear) / no (3/15, 1 clear) | — / — | 0.12 / 0.33 | `deriv_e13mag50_USDCHF_15m_result.json` |
| e14nylate | e14nylate: all (66), nseed 1 | 15m | 0.03 | .5141 (.5030) | .5080 · .733 · .5704 | 1/2 · .4961 | .3864 · .500 · .4870 | 0/0 · — | .5017 / .4735 | KILLED / KILLED | no (15/15, 11 clear) / no (6/15, 3 clear) | — / — | 0.60 / 0.33 | `deriv_e14nylate_USDCHF_15m_result.json` |
| e15nypm | e15nypm: all (66), nseed 1 | 15m | 0.03 | .5121 (.4874) | .4839 · .667 · .5825 | 1/1 · .6154 | .3219 · .500 · .4837 | 0/0 · — | .5073 / .4701 | KILLED / KILLED | no (15/15, 10 clear) / no (4/15, 2 clear) | — / — | 0.60 / 0.50 | `deriv_e15nypm_USDCHF_15m_result.json` |
| e16nyeve | e16nyeve: all (66), nseed 1 | 15m | 0.03 | .5143 (.4938) | .4575 · .400 · .5266 | 1/1 · .5429 | .5180 · .000 · .5238 | 0/0 · — | .4962 / .4773 | KILLED / KILLED | no (5/15, 2 clear) / no (3/15, 0 clear) | — / — | 0.40 / 0.00 | `deriv_e16nyeve_USDCHF_15m_result.json` |
| e17nyam | e17nyam: all (66), nseed 1 | 15m | 0.03 | .4981 (.4864) | .4508 · .400 · .5051 | 0/0 · — | .4743 · .429 · .5330 | 0/0 · — | .4930 / .4962 | KILLED / KILLED | no (10/15, 4 clear) / no (7/15, 3 clear) | — / — | 0.20 / 0.43 | `deriv_e17nyam_USDCHF_15m_result.json` |
| e18lowcap | e18lowcap: all (66), nseed 1 | 15m | 0.03 | .5006 (.4929) | .4983 · .067 · .5176 | 2/4 · .5588 | .4251 · .067 · .4675 | 0/3 · .4817 | .4987 / .4855 | KILLED / KILLED | no (15/15, 1 clear) / no (15/15, 1 clear) | — / — | 0.07 / 0.07 | `deriv_e18lowcap_USDCHF_15m_result.json` |
| e19tf | e19tf: all+fam:tf (74), nseed 1 | 15m | 0.03 | .5024 (.4908) | .5030 · .800 · .5777 | 0/1 · .4231 | .3773 · .231 · .4752 | 0/1 · .5000 | .4987 / .4855 | KILLED / KILLED | no (15/15, 12 clear) / no (13/15, 3 clear) | — / — | 0.67 / 0.15 | `deriv_e19tf_USDCHF_15m_result.json` |
| e1 | e1: all (66), nseed 1 | 15m | 0.03 | .5049 (.4998) | .4600 · .600 · .5556 | 1/3 · .5480 | .4825 · .600 · .5612 | 1/1 · .6579 | .4975 / .4866 | KILLED / KILLED | no (15/15, 9 clear) / no (15/15, 9 clear) | — / — | 0.60 / 0.40 | `deriv_e1_USDCHF_15m_result.json` |
| e1t | e1t: all (66), nseed 1 | 15m | 0.03 | .5000 (.4900) | .5292 · .667 · .5681 | 1/3 · .5538 | .4045 · .214 · .4898 | 1/1 · .6071 | .4987 / .4855 | KILLED / KILLED | no (15/15, 10 clear) / no (14/15, 3 clear) | — / — | 0.40 / 0.07 | `deriv_e1t_USDCHF_15m_result.json` |
| e20sg | e20sg: all+fam:sg (71), nseed 1 | 15m | 0.03 | .5028 (.4916) | .5513 · .933 · .5966 | 1/2 · .5887 | .3979 · .429 · .5206 | 1/1 · .6333 | .4987 / .4855 | CERTIFIED / KILLED | CERTIFIED (15/15, 14 clear) / no (14/15, 6 clear) | — / — | 0.67 / 0.29 | `deriv_e20sg_USDCHF_15m_result.json` |
| e20sgnc | e20sgnc: noclock+fam:sg (68), nseed 1 | 15m | 0.03 | .5037 (.4961) | .5358 · .867 · .6032 | 1/3 · .5998 | .4127 · .467 · .5211 | 2/2 · .5662 | .4987 / .4855 | KILLED / KILLED | no (15/15, 13 clear) / no (15/15, 7 clear) | — / — | 0.73 / 0.27 | `deriv_e20sgnc_USDCHF_15m_result.json` |
| e20sgs3 | e20sgs3: all+fam:sg (71), nseed 3 | 15m | 0.03 | .5029 (.4935) | .5358 · .867 · .6071 | 1/1 · .7778 | .3754 · .417 · .5131 | 0/1 · .5385 | .4987 / .4855 | KILLED / KILLED | no (15/15, 13 clear) / no (12/15, 5 clear) | — / — | 0.73 / 0.42 | `deriv_e20sgs3_USDCHF_15m_result.json` |
| e21kf | e21kf: all+fam:kf (73), nseed 1 | 15m | 0.03 | .5023 (.4934) | .4953 · .800 · .5923 | 1/1 · .5926 | .4375 · .364 · .5322 | 0/0 · — | .4987 / .4855 | KILLED / KILLED | no (15/15, 12 clear) / no (11/15, 4 clear) | — / — | 0.60 / 0.36 | `deriv_e21kf_USDCHF_15m_result.json` |
| e22rm | e22rm: all+fam:rm (70), nseed 1 | 15m | 0.03 | .5041 (.4951) | .5082 · .800 · .5711 | 1/2 · .5385 | .3718 · .167 · .4724 | 0/0 · — | .4987 / .4855 | KILLED / KILLED | no (15/15, 12 clear) / no (12/15, 2 clear) | — / — | 0.53 / 0.17 | `deriv_e22rm_USDCHF_15m_result.json` |
| e23fd | e23fd: all+fam:fd (68), nseed 1 | 15m | 0.03 | .4992 (.4872) | .5308 · .867 · .5902 | 2/3 · .6148 | .3526 · .214 · .4786 | 0/1 · .5000 | .4987 / .4855 | KILLED / KILLED | no (15/15, 13 clear) / no (14/15, 3 clear) | — / — | 0.60 / 0.21 | `deriv_e23fd_USDCHF_15m_result.json` |
| e24cx | e24cx: all+fam:cx (74), nseed 1 | 15m | 0.03 | .5029 (.4924) | .5125 · .800 · .5908 | 0/0 · — | .3793 · .250 · .4689 | 0/0 · — | .4987 / .4855 | KILLED / KILLED | no (15/15, 12 clear) / no (12/15, 3 clear) | — / — | 0.60 / 0.25 | `deriv_e24cx_USDCHF_15m_result.json` |
| e25tk | e25tk: all+fam:tk (72), nseed 1 | 15m | 0.03 | .4999 (.4908) | .4632 · .600 · .5427 | 2/2 · .5879 | .3739 · .154 · .4712 | 1/1 · .5600 | .4987 / .4855 | KILLED / KILLED | no (15/15, 9 clear) / no (13/15, 2 clear) | — / — | 0.40 / 0.08 | `deriv_e25tk_USDCHF_15m_result.json` |
| e26pa | e26pa: all+fam:pa (78), nseed 1 | 15m | 0.03 | .5028 (.4916) | .4750 · .533 · .5589 | 0/0 · — | .3549 · .500 · .4995 | 0/0 · — | .4987 / .4855 | KILLED / KILLED | no (15/15, 8 clear) / no (8/15, 4 clear) | — / — | 0.47 / 0.25 | `deriv_e26pa_USDCHF_15m_result.json` |
| e27famall | e27famall: all+fam:tf,sg,kf,rm,fd,cx,tk,pa (118), nseed 1 | 15m | 0.03 | .5108 (.5014) | .4983 · .500 · .5964 | 0/0 · — | .4559 · .000 · .4832 | 0/0 · — | .4987 / .4855 | KILLED / KILLED | no (8/15, 4 clear) / no (4/15, 0 clear) | — / — | 0.50 / 0.00 | `deriv_e27famall_USDCHF_15m_result.json` |
| e28resid | e28resid: all (66), nseed 1 | 15m | 0.03 | .5067 (.4989) | .4998 · .467 · .5447 | 0/1 · .3556 | .4564 · .333 · .5240 | 1/1 · .5625 | .4987 / .4855 | KILLED / KILLED | no (15/15, 7 clear) / no (15/15, 5 clear) | — / — | 0.40 / 0.13 | `deriv_e28resid_USDCHF_15m_result.json` |
| e29logit | e29logit: all (66), nseed 1 | 15m | 0.03 | .5010 (.4909) | .4529 · .067 · .4813 | 0/4 · .4568 | .4459 · .333 · .5151 | 2/4 · .5015 | .4987 / .4855 | KILLED / KILLED | no (15/15, 1 clear) / no (15/15, 5 clear) | — / — | 0.00 / 0.13 | `deriv_e29logit_USDCHF_15m_result.json` |
| e2noclock | e2noclock: noclock (63), nseed 1 | 15m | 0.03 | .5007 (.4928) | .5167 · .733 · .5823 | 2/3 · .6289 | .3800 · .267 · .4904 | 2/2 · .5911 | .4987 / .4855 | KILLED / KILLED | no (15/15, 11 clear) / no (15/15, 4 clear) | — / — | 0.53 / 0.07 | `deriv_e2noclock_USDCHF_15m_result.json` |
| e30nystroem | e30nystroem: all (66), nseed 1 | 15m | 0.03 | .5059 (.4962) | .4834 · .071 · .5050 | 1/4 · .5291 | .4637 · .000 · .4893 | 0/2 · .5000 | .4987 / .4855 | KILLED / KILLED | no (14/15, 1 clear) / no (12/15, 0 clear) | — / — | 0.00 / 0.00 | `deriv_e30nystroem_USDCHF_15m_result.json` |
| e31ens3 | e31ens3: all (66), nseed 1 | 15m | 0.03 | .5017 (.4919) | .5272 · .714 · .5795 | 0/0 · — | .4037 · .333 · .5060 | 0/0 · — | .4987 / .4855 | KILLED / KILLED | no (14/15, 10 clear) / no (15/15, 5 clear) | — / — | 0.64 / 0.27 | `deriv_e31ens3_USDCHF_15m_result.json` |
| e32spec | e32spec: all (66), nseed 1 | 15m | 0.03 | .5048 (.4994) | .5061 · .533 · .5484 | 2/4 · .5219 | .4252 · .000 · .4698 | 0/3 · .4778 | .4987 / .4855 | KILLED / KILLED | no (15/15, 8 clear) / no (15/15, 0 clear) | — / — | 0.33 / 0.00 | `deriv_e32spec_USDCHF_15m_result.json` |
| e33comp33 | e33comp33: all (66), nseed 1 | 15m | 0.1 | .5000 (.4900) | .5257 · .733 · .5629 | 0/1 · .5135 | .4310 · .167 · .4987 | 0/1 · .5000 | .4987 / .4855 | KILLED / KILLED | no (15/15, 11 clear) / no (12/15, 2 clear) | — / — | 0.47 / 0.08 | `deriv_e33comp33_USDCHF_15m_result.json` |
| e34online | e34online: all (66), nseed 1 | 15m | 0.03 | .4962 (.4855) | .4691 · .000 · .4957 | 0/0 · — | .4354 · .000 · .4640 | 0/0 · — | .4987 / .4855 | KILLED / KILLED | no (15/15, 0 clear; never certified: a continuing policy: the online learner updates on settled outcomes during each test path) / no (15/15, 0 clear; never certified: a continuing policy: the online learner updates on settled outcomes during each test path) | — / — | 0.00 / 0.00 | `deriv_e34online_USDCHF_15m_result.json` |
| e35absret | e35absret: all (66), nseed 1 | 15m | 0.03 | .5012 (.4920) | .4743 · .417 · .5345 | 3/3 · .6284 | .4258 · .250 · .5103 | 0/1 · .5000 | .4987 / .4855 | KILLED / KILLED | no (12/15, 5 clear) / no (12/15, 3 clear) | — / — | 0.42 / 0.17 | `deriv_e35absret_USDCHF_15m_result.json` |
| e36aci | e36aci: all (66), nseed 1 | 15m | 0.03 | .5000 (.4900) | .5255 · .600 · .5591 | 1/3 · .5636 | .3639 · .231 · .4750 | 1/1 · .5862 | .4987 / .4855 | KILLED / KILLED | no (15/15, 9 clear; never certified: an adaptive threshold: ACI moves it with settled test outcomes) / no (13/15, 3 clear; never certified: an adaptive threshold: ACI moves it with settled test outcomes) | — / — | 0.40 / 0.08 | `deriv_e36aci_USDCHF_15m_result.json` |
| e37meta | e37meta: all (66), nseed 1 | 15m | 0.06 | .5000 (.4900) | .5109 · .667 · .5589 | 1/4 · .5427 | .3965 · .200 · .4842 | 0/1 · .5172 | .4987 / .4855 | KILLED / KILLED | no (15/15, 10 clear) / no (15/15, 3 clear) | — / — | 0.53 / 0.13 | `deriv_e37meta_USDCHF_15m_result.json` |
| e38release | e38release: all (66), nseed 1 | 15m | 0.03 | .4988 (.4943) | .4484 · .067 · .4858 | 0/4 · .4533 | .4365 · .000 · .4725 | 1/4 · .4991 | .4987 / .4855 | KILLED / KILLED | no (15/15, 1 clear) / no (15/15, 0 clear) | — / — | 0.00 / 0.00 | `deriv_e38release_USDCHF_15m_result.json` |
| e39cm | e39cm: all+fam:cm (75), nseed 1 | 15m | 0.03 | .5052 (.4906) | .4710 · .417 · .5306 | 0/0 · — | .4590 · .250 · .5193 | 0/0 · — | .4987 / .4855 | KILLED / KILLED | no (12/15, 5 clear) / no (8/15, 2 clear) | — / — | 0.17 / 0.25 | `deriv_e39cm_USDCHF_15m_result.json` |
| e3union | e3union: all (404), nseed 1 | 15m | 0.03 | .5180 (.5092) | .5414 · .800 · .6170 | 0/0 · — | nan · .000 · — | 0/0 · — | .4986 / .4856 | CERTIFIED / KILLED | no (5/15, 4 clear) / no (0/15, 0 clear) | — / — | 0.60 / — | `deriv_e3union_USDCHF_15m_result.json` |
| e40forest | e40forest: all (66), nseed 1 | 15m | 0.03 | .5114 (.5045) | .4942 · .200 · .5259 | 3/4 · .5434 | .3974 · .111 · .4669 | 1/2 · .4886 | .4987 / .4855 | KILLED / KILLED | no (15/15, 3 clear) / no (9/15, 1 clear) | — / — | 0.13 / 0.11 | `deriv_e40forest_USDCHF_15m_result.json` |
| e41pool | e41pool: all (66), nseed 1 | 15m | 0.03 | .5020 (.4910) | .4609 · .400 · .5303 | 1/4 · .5421 | .3761 · .200 · .4704 | 1/2 · .5479 | .4987 / .4855 | KILLED / KILLED | no (15/15, 6 clear) / no (15/15, 3 clear) | — / — | 0.27 / 0.07 | `deriv_e41pool_USDCHF_15m_result.json` |
| e4seed3 | e4seed3: all (66), nseed 3 | 15m | 0.03 | .4999 (.4898) | .4880 · .733 · .5654 | 1/1 · .5517 | .3802 · .154 · .4836 | 1/1 · .6333 | .4987 / .4855 | KILLED / KILLED | no (15/15, 11 clear) / no (13/15, 2 clear) | — / — | 0.47 / 0.08 | `deriv_e4seed3_USDCHF_15m_result.json` |
| e51fx | e51fx: all+fam:fx (82), nseed 1 | 15m | 0.03 | .5009 (.4876) | .4939 · .667 · .5592 | 1/2 · .5431 | .3805 · .214 · .4789 | 1/1 · .5600 | .4987 / .4855 | KILLED / KILLED | no (15/15, 10 clear) / no (14/15, 3 clear) | — / — | 0.33 / 0.14 | `deriv_e51fx_USDCHF_15m_result.json` |
| e52ck | e52ck: all+fam:ck (77), nseed 1 | 15m | 0.03 | .4994 (.4892) | .5154 · .533 · .5658 | 1/1 · .6000 | .4299 · .214 · .4987 | 0/1 · .5357 | .4987 / .4855 | KILLED / KILLED | no (15/15, 8 clear) / no (14/15, 3 clear) | — / — | 0.40 / 0.07 | `deriv_e52ck_USDCHF_15m_result.json` |
| e53ckcompact | e53ckcompact: list:compact_kernels.json (14), nseed 1 | 15m | 0.03 | .5018 (.4903) | .4714 · .200 · .5062 | 2/4 · .5317 | .4239 · .000 · .4773 | 0/4 · .4491 | .4987 / .4855 | KILLED / KILLED | no (15/15, 3 clear) / no (15/15, 0 clear) | — / — | 0.00 / 0.00 | `deriv_e53ckcompact_USDCHF_15m_result.json` |
| e54ckcompactlogit | e54ckcompactlogit: list:compact_kernels.json (14), nseed 1 | 15m | 0.03 | .5033 (.4951) | .4669 · .133 · .5002 | 2/4 · .5390 | .4405 · .364 · .5216 | 0/3 · .4986 | .4987 / .4855 | KILLED / KILLED | no (15/15, 2 clear) / no (11/15, 4 clear) | — / — | 0.00 / 0.27 | `deriv_e54ckcompactlogit_USDCHF_15m_result.json` |
| e55rl | e55rl: all+fam:rl (72), nseed 1 | 15m | 0.03 | .4975 (.4889) | .4820 · .643 · .5562 | 1/1 · .6400 | .3824 · .364 · .4948 | 0/0 · — | .4987 / .4855 | KILLED / KILLED | no (14/15, 9 clear) / no (11/15, 4 clear) | — / — | 0.64 / 0.36 | `deriv_e55rl_USDCHF_15m_result.json` |
| e56rs | e56rs: all+fam:rs (72), nseed 1 | 15m | 0.03 | .5017 (.4933) | .4609 · .733 · .5676 | 1/1 · .6452 | .3853 · .222 · .4918 | 0/0 · — | .4987 / .4855 | KILLED / KILLED | no (15/15, 11 clear) / no (9/15, 2 clear) | — / — | 0.60 / 0.00 | `deriv_e56rs_USDCHF_15m_result.json` |
| e57reflect | e57reflect: all (66), nseed 1 | 15m | 0.03 | .5041 (.4905) | .4327 · .571 · .5416 | 0/0 · — | .3910 · .357 · .4942 | 1/2 · .4915 | .4987 / .4855 | KILLED / KILLED | no (14/15, 8 clear) / no (14/15, 5 clear) | — / — | 0.43 / 0.36 | `deriv_e57reflect_USDCHF_15m_result.json` |
| e58nysreflect | e58nysreflect: all (66), nseed 1 | 15m | 0.03 | .5059 (.4982) | .4938 · .462 · .5270 | 3/4 · .5311 | .4753 · .214 · .5212 | 0/2 · .4754 | .4987 / .4855 | KILLED / KILLED | no (13/15, 6 clear) / no (14/15, 3 clear) | — / — | 0.15 / 0.14 | `deriv_e58nysreflect_USDCHF_15m_result.json` |
| e59brier | e59brier: all (66), nseed 1 | 15m | 0.03 | .4995 (.4901) | .4933 · .533 · .5596 | 1/3 · .5628 | .3950 · .333 · .5069 | 1/1 · .6923 | .4987 / .4855 | KILLED / KILLED | no (15/15, 8 clear) / no (15/15, 5 clear) | — / — | 0.47 / 0.27 | `deriv_e59brier_USDCHF_15m_result.json` |
| e5h30 | e5h30: all (66), nseed 1 | 30m | 0.03 | .5008 (.4859) | .4615 · .091 · .5084 | 0/0 · — | .4714 · .500 · .5108 | 0/0 · — | .5138 / .4761 | KILLED / KILLED | no (11/15, 1 clear) / no (2/15, 1 clear) | — / — | 0.09 / 0.00 | `deriv_e5h30_USDCHF_30m_result.json` |
| e60stack | e60stack: all (66), nseed 1 | 15m | 0.03 | .5005 (.4900) | .4900 · .667 · .5505 | 2/4 · .5458 | .4205 · .333 · .4952 | 0/1 · .4828 | .4987 / .4855 | KILLED / KILLED | no (15/15, 10 clear) / no (15/15, 5 clear) | — / — | 0.47 / 0.13 | `deriv_e60stack_USDCHF_15m_result.json` |
| e61ord4 | e61ord4: all (66), nseed 1 | 15m | 0.03 | .4990 (.4860) | .4792 · .667 · .5642 | 0/1 · .3704 | .3680 · .250 · .4754 | 0/0 · — | .4987 / .4855 | KILLED / KILLED | no (15/15, 10 clear) / no (12/15, 3 clear) | — / — | 0.53 / 0.17 | `deriv_e61ord4_USDCHF_15m_result.json` |
| e62har | e62har: all (66), nseed 1 | 15m | 0.03 | .5000 (.4900) | .3454 · .000 · .4362 | 0/0 · — | .3593 · .143 · .4647 | 0/0 · — | .4987 / .4855 | KILLED / KILLED | no (4/15, 0 clear) / no (7/15, 1 clear) | — / — | 0.00 / 0.14 | `deriv_e62har_USDCHF_15m_result.json` |
| e64boot | e64boot: all (66), nseed 1 | 15m | 0.03 | .4993 (.4937) | .4819 · .667 · .5741 | 1/3 · .5534 | .3527 · .267 · .4778 | 0/1 · .5000 | .4987 / .4855 | KILLED / KILLED | no (15/15, 10 clear) / no (15/15, 4 clear) | — / — | 0.60 / 0.07 | `deriv_e64boot_USDCHF_15m_result.json` |
| e65recency | e65recency: all (66), nseed 1 | 15m | 0.03 | .4992 (.4892) | .4321 · .250 · .4912 | 1/2 · .5386 | .4483 · .182 · .4889 | 0/0 · — | .4987 / .4855 | KILLED / KILLED | no (8/15, 2 clear) / no (11/15, 2 clear) | — / — | 0.12 / 0.09 | `deriv_e65recency_USDCHF_15m_result.json` |
| e66mh | e66mh: all (66), nseed 1 | 15m | 0.03 | .5005 (.4908) | .4953 · .500 · .5812 | 0/0 · — | .5284 · .500 · .5729 | 0/0 · — | .4987 / .4855 | KILLED / KILLED | no (6/15, 3 clear) / no (2/15, 1 clear) | — / — | 0.50 / 0.50 | `deriv_e66mh_USDCHF_15m_result.json` |
| e67drfe | e67drfe: all (66), nseed 1 | 15m | 0.03 | .5078 (.5042) | .4349 · .067 · .4843 | 2/4 · .5443 | .4571 · .400 · .5204 | 3/4 · .5598 | .4987 / .4855 | KILLED / KILLED | no (15/15, 1 clear) / no (15/15, 6 clear) | — / — | 0.00 / 0.20 | `deriv_e67drfe_USDCHF_15m_result.json` |
| e68agree | e68agree: all (66), nseed 1 | 15m | 0.03 | .4997 (.4902) | .4800 · .600 · .5658 | 2/2 · .6259 | .4096 · .143 · .4722 | 1/1 · .7143 | .4987 / .4855 | KILLED / KILLED | no (15/15, 9 clear) / no (14/15, 2 clear) | — / — | 0.60 / 0.07 | `deriv_e68agree_USDCHF_15m_result.json` |
| e6pool | e6pool: own (41), nseed 1, pooled +6 pairs | 15m | 0.03 | .5006 (.4872) | .4691 · .533 · .5389 | 3/4 · .5429 | .4351 · .133 · .4884 | 0/4 · .4989 | .4987 / .4855 | KILLED / KILLED | no (15/15, 8 clear) / no (15/15, 2 clear) | — / — | 0.40 / 0.00 | `deriv_e6pool_USDCHF_15m_result.json` |
| e70play | e70play: all+fam:hy (75), nseed 1 | 15m | 0.03 | .5006 (.4928) | .4883 · .467 · .5459 | 2/2 · .5575 | .4130 · .429 · .5226 | 0/1 · .5000 | .4987 / .4855 | KILLED / KILLED | no (15/15, 7 clear) / no (14/15, 6 clear) | — / — | 0.33 / 0.29 | `deriv_e70play_USDCHF_15m_result.json` |
| e71playreset | e71playreset: all+fam:hr (75), nseed 1 | 15m | 0.03 | .5009 (.4898) | .4930 · .533 · .5657 | 2/3 · .5897 | .4032 · .214 · .4814 | 1/1 · .6538 | .4987 / .4855 | KILLED / KILLED | no (15/15, 8 clear) / no (14/15, 3 clear) | — / — | 0.47 / 0.14 | `deriv_e71playreset_USDCHF_15m_result.json` |
| e72shock | e72shock: all+fam:d2 (67), nseed 1 | 15m | 0.03 | .5024 (.4896) | .4772 · .667 · .5665 | 0/0 · — | .3395 · .154 · .4750 | 0/0 · — | .4987 / .4855 | KILLED / KILLED | no (15/15, 10 clear) / no (13/15, 2 clear) | — / — | 0.47 / 0.15 | `deriv_e72shock_USDCHF_15m_result.json` |
| e73mono | e73mono: all (66), nseed 1 | 15m | 0.03 | .4992 (.4877) | .4887 · .467 · .5580 | 3/3 · .6293 | .3575 · .286 · .4624 | 1/1 · .5806 | .4987 / .4855 | KILLED / KILLED | no (15/15, 7 clear) / no (14/15, 4 clear) | — / — | 0.47 / 0.14 | `deriv_e73mono_USDCHF_15m_result.json` |
| e74logitsig | e74logitsig: all (66), nseed 1 | 15m | 0.03 | .5003 (.4881) | .4626 · .067 · .4900 | 0/4 · .4951 | .4400 · .267 · .5069 | 0/4 · .4709 | .4987 / .4855 | KILLED / KILLED | no (15/15, 1 clear) / no (15/15, 4 clear) | — / — | 0.00 / 0.00 | `deriv_e74logitsig_USDCHF_15m_result.json` |
| e75lgbsig | e75lgbsig: all (66), nseed 1 | 15m | 0.03 | .5078 (.5016) | .4871 · .267 · .5292 | 2/4 · .5439 | .4208 · .200 · .5000 | 0/4 · .5109 | .4987 / .4855 | KILLED / KILLED | no (15/15, 4 clear) / no (15/15, 3 clear) | — / — | 0.20 / 0.07 | `deriv_e75lgbsig_USDCHF_15m_result.json` |
| e76drfelgb | e76drfelgb: all (66), nseed 1 | 15m | 0.03 | .5085 (.5020) | .4789 · .467 · .5392 | 3/4 · .5820 | .4252 · .200 · .4919 | 0/4 · .4456 | .4987 / .4855 | KILLED / KILLED | no (15/15, 7 clear) / no (15/15, 3 clear) | — / — | 0.27 / 0.07 | `deriv_e76drfelgb_USDCHF_15m_result.json` |
| e79qorder | e79qorder: all+fam:qo (88), nseed 1 | 15m | 0.03 | .4980 (.4904) | .4717 · .400 · .5457 | 0/0 · — | .4225 · .083 · .4757 | 0/0 · — | .4987 / .4855 | KILLED / KILLED | no (15/15, 6 clear) / no (12/15, 1 clear) | — / — | 0.27 / 0.00 | `deriv_e79qorder_USDCHF_15m_result.json` |
| e7fitall | e7fitall: all (66), nseed 1 | 15m | 0.03 | .5010 (.4934) | .5146 · .733 · .5704 | 0/3 · .5051 | .4005 · .333 · .5062 | 1/1 · .5806 | .4987 / .4855 | KILLED / KILLED | no (15/15, 11 clear) / no (15/15, 5 clear) | — / — | 0.53 / 0.27 | `deriv_e7fitall_USDCHF_15m_result.json` |
| e80qperm | e80qperm: all+fam:qp (88), nseed 1 | 15m | 0.03 | .4989 (.4890) | .4335 · .667 · .5750 | 0/0 · — | .4248 · .308 · .5072 | 0/0 · — | .4987 / .4855 | KILLED / KILLED | no (15/15, 10 clear) / no (13/15, 4 clear) | — / — | 0.67 / 0.15 | `deriv_e80qperm_USDCHF_15m_result.json` |
| e81tail0 | e81tail0: all (66), nseed 1 | 15m | 0.03 | .5054 (.4951) | .4690 · .467 · .5353 | 1/2 · .5294 | .4195 · .000 · .4679 | 0/0 · — | .4987 / .4855 | KILLED / KILLED | no (15/15, 7 clear) / no (5/15, 0 clear) | — / — | 0.40 / 0.00 | `deriv_e81tail0_USDCHF_15m_result.json` |
| e82tailaux | e82tailaux: all (66), nseed 1 | 15m | 0.03 | .5040 (.4918) | .5089 · .667 · .6065 | 1/2 · .4877 | .4681 · .143 · .5271 | 0/0 · — | .4987 / .4855 | KILLED / KILLED | no (15/15, 10 clear) / no (7/15, 1 clear) | — / — | 0.53 / 0.14 | `deriv_e82tailaux_USDCHF_15m_result.json` |
| e84bootcomp | e84bootcomp: all (66), nseed 1 | 15m | 0.1 | .4993 (.4937) | .4962 · .400 · .5585 | 0/1 · .4634 | .4521 · .385 · .5187 | 0/1 · .4722 | .4987 / .4855 | KILLED / KILLED | no (15/15, 6 clear) / no (13/15, 5 clear) | — / — | 0.33 / 0.23 | `deriv_e84bootcomp_USDCHF_15m_result.json` |
| e85bootcomp0 | e85bootcomp0: all (66), nseed 1 | 15m | 0.1 | .4996 (.4914) | .5175 · .667 · .5786 | 0/1 · .4500 | .4595 · .091 · .4982 | 0/1 · .4688 | .4987 / .4855 | KILLED / KILLED | no (15/15, 10 clear) / no (11/15, 1 clear) | — / — | 0.47 / 0.09 | `deriv_e85bootcomp0_USDCHF_15m_result.json` |
| e8tb2 | e8tb2: all (66), nseed 1 | 15m | 0.03 | .5014 (.4927) | .4857 · .400 · .5313 | 1/2 · .5478 | .3710 · .143 · .4768 | 0/0 · — | .4987 / .4855 | KILLED / KILLED | no (15/15, 6 clear) / no (14/15, 2 clear) | — / — | 0.20 / 0.07 | `deriv_e8tb2_USDCHF_15m_result.json` |
| e9blockval | e9blockval: all (66), nseed 1 | 15m | 0.03 | .5025 (.4924) | .4844 · .267 · .5274 | 1/4 · .5208 | .4239 · .000 · .4667 | 0/4 · .5144 | .4987 / .4855 | KILLED / KILLED | no (15/15, 4 clear) / no (15/15, 0 clear) | — / — | 0.13 / 0.00 | `deriv_e9blockval_USDCHF_15m_result.json` |
| r2t | r2t: list:USDCHF.m15ny_xpair_seedens.v1.json (337), nseed 3 | 15m | 0.01 | .5197 (.5105) | nan · .000 · — | 0/0 · — | nan · .000 · — | 0/0 · — | .4986 / .4856 | KILLED / KILLED | no (0/15, 0 clear) / no (0/15, 0 clear) | — / — | — / — | `deriv_r2t_USDCHF_15m_result.json` |
| r3 | r3: list:USDCHF.m15ny_xpair_seedens.v1.json (337), nseed 3 | 15m | 0.01 | .5132 (.4994) | nan · .000 · — | 0/0 · — | nan · .000 · — | 0/0 · — | .5010 / .4742 | KILLED / KILLED | no (0/15, 0 clear) / no (0/15, 0 clear) | — / — | — / — | `deriv_r3_USDCHF_15m_result.json` |
| r4fixldn | r4fixldn: rule:fx_ldn_prefix_u60 (1), nseed 0 | 15m | 1.0 | .5270 (.4981) | .5349 · .667 · .5676 | 0/0 · — | .4775 · .333 · .5295 | 0/0 · — | .4987 / .4855 | KILLED / KILLED | no (15/15, 10 clear) / no (15/15, 5 clear) | — / — | 0.33 / 0.27 | `deriv_r4fixldn_USDCHF_15m_result.json` |
| r5fixecb | r5fixecb: rule:fx_ecb_prefix_u60 (1), nseed 0 | 15m | 1.0 | .5168 (.4953) | .4949 · .200 · .5202 | 0/0 · — | .4793 · .000 · .4957 | 0/0 · — | .4987 / .4855 | KILLED / KILLED | no (15/15, 3 clear) / no (15/15, 0 clear) | — / — | 0.00 / 0.00 | `deriv_r5fixecb_USDCHF_15m_result.json` |
| r6n-a003-p | r6n-a003-p: all (60), nseed 1 | 15m | 0.03 | .4941 (.4887) | .5005 · .000 · .5120 | 0/0 · — | .4820 · .333 · .5074 | 0/0 · — | .5008 / .4843 | PILOT (three balanced paths) / PILOT (three balanced paths) | no (3/3, 0 clear; incomplete: 3 of 15 paths evaluated; never certified: PILOT (three balanced paths)) / no (3/3, 1 clear; incomplete: 3 of 15 paths evaluated; never certified: PILOT (three balanced paths)) | — / — | 0.00 / 0.00 | `deriv_r6n-a003-p_USDCHF_15m_result.json` |
| r6n-a003fixed-p | r6n-a003fixed-p: all (60), nseed 1 | 15m | 0.03 | .5006 (.4964) | .5220 · .333 · .5744 | 0/0 · — | .4719 · .333 · .5279 | 0/0 · — | .5008 / .4843 | PILOT (three balanced paths) / PILOT (three balanced paths) | no (3/3, 1 clear; incomplete: 3 of 15 paths evaluated; never certified: PILOT (three balanced paths)) / no (3/3, 1 clear; incomplete: 3 of 15 paths evaluated; never certified: PILOT (three balanced paths)) | — / — | 0.33 / 0.33 | `deriv_r6n-a003fixed-p_USDCHF_15m_result.json` |
| r6n-a003linear-p | r6n-a003linear-p: all (60), nseed 1 | 15m | 0.03 | .5016 (.4975) | .5069 · .000 · .5172 | 0/0 · — | .4500 · .333 · .4858 | 0/0 · — | .5008 / .4843 | PILOT (three balanced paths) / PILOT (three balanced paths) | no (3/3, 0 clear; incomplete: 3 of 15 paths evaluated; never certified: PILOT (three balanced paths)) / no (3/3, 1 clear; incomplete: 3 of 15 paths evaluated; never certified: PILOT (three balanced paths)) | — / — | 0.00 / 0.00 | `deriv_r6n-a003linear-p_USDCHF_15m_result.json` |
| r6n-a003permuted-p | r6n-a003permuted-p: all (60), nseed 1 | 15m | 0.03 | .4942 (.4888) | .4631 · .000 · .4784 | 0/0 · — | .4627 · .333 · .4974 | 0/0 · — | .5008 / .4843 | PILOT (three balanced paths) / PILOT (three balanced paths) | no (3/3, 0 clear; incomplete: 3 of 15 paths evaluated; never certified: PILOT (three balanced paths)) / no (3/3, 1 clear; incomplete: 3 of 15 paths evaluated; never certified: PILOT (three balanced paths)) | — / — | 0.00 / 0.33 | `deriv_r6n-a003permuted-p_USDCHF_15m_result.json` |
| r6n-a005-p | r6n-a005-p: all (60), nseed 1 | 15m | 0.03 | .4898 (.4861) | .4533 · .000 · .4673 | 0/0 · — | .4357 · .000 · .4729 | 0/0 · — | .5008 / .4843 | PILOT (three balanced paths) / PILOT (three balanced paths) | no (3/3, 0 clear; incomplete: 3 of 15 paths evaluated; never certified: PILOT (three balanced paths)) / no (3/3, 0 clear; incomplete: 3 of 15 paths evaluated; never certified: PILOT (three balanced paths)) | — / — | 0.00 / 0.00 | `deriv_r6n-a005-p_USDCHF_15m_result.json` |
| r6n-a005linear-p | r6n-a005linear-p: all (60), nseed 1 | 15m | 0.03 | .5021 (.4985) | .4784 · .333 · .5061 | 0/0 · — | .4314 · .333 · .4832 | 0/0 · — | .5008 / .4843 | PILOT (three balanced paths) / PILOT (three balanced paths) | no (3/3, 1 clear; incomplete: 3 of 15 paths evaluated; never certified: PILOT (three balanced paths)) / no (3/3, 1 clear; incomplete: 3 of 15 paths evaluated; never certified: PILOT (three balanced paths)) | — / — | 0.00 / 0.00 | `deriv_r6n-a005linear-p_USDCHF_15m_result.json` |
| r6n-a016-f | r6n-a016-f: all (60), nseed 1 | 15m | 0.03 | .4981 (.4882) | .4687 · .533 · .5270 | 2/4 · .5530 | .4014 · .000 · .4674 | 0/4 · .4999 | .5007 / .4844 | KILLED / KILLED | no (15/15, 8 clear) / no (15/15, 0 clear) | — / — | 0.20 / 0.00 | `deriv_r6n-a016-f_USDCHF_15m_result.json` |
| r6n-a016-p | r6n-a016-p: all (60), nseed 1 | 15m | 0.03 | .4970 (.4955) | .4444 · .333 · .5084 | 0/0 · — | .4698 · .000 · .4787 | 0/0 · — | .5008 / .4843 | PILOT (three balanced paths) / PILOT (three balanced paths) | no (3/3, 1 clear; incomplete: 3 of 15 paths evaluated; never certified: PILOT (three balanced paths)) / no (3/3, 0 clear; incomplete: 3 of 15 paths evaluated; never certified: PILOT (three balanced paths)) | — / — | 0.33 / 0.00 | `deriv_r6n-a016-p_USDCHF_15m_result.json` |
| r6n-a069-p | r6n-a069-p: all (60), nseed 1 | 15m | 0.03 | .5018 (.5005) | .4869 · .000 · .4978 | 0/0 · — | .4405 · .000 · .4884 | 0/0 · — | .5008 / .4843 | PILOT (three balanced paths) / PILOT (three balanced paths) | no (3/3, 0 clear; incomplete: 3 of 15 paths evaluated; never certified: PILOT (three balanced paths)) / no (3/3, 0 clear; incomplete: 3 of 15 paths evaluated; never certified: PILOT (three balanced paths)) | — / — | 0.00 / 0.00 | `deriv_r6n-a069-p_USDCHF_15m_result.json` |
| r6n-a070-f | r6n-a070-f: all (60), nseed 1 | 15m | 0.03 | .5012 (.4870) | .4747 · .133 · .5063 | 1/4 · .5152 | .4698 · .214 · .5083 | 1/3 · .5374 | .5007 / .4844 | KILLED / KILLED | no (15/15, 2 clear) / no (14/15, 3 clear) | — / — | 0.07 / 0.07 | `deriv_r6n-a070-f_USDCHF_15m_result.json` |
| r6n-a070-p | r6n-a070-p: all (60), nseed 1 | 15m | 0.03 | .4949 (.4890) | .5017 · .333 · .5168 | 0/0 · — | .4555 · .333 · .5306 | 0/0 · — | .5008 / .4843 | PILOT (three balanced paths) / PILOT (three balanced paths) | no (3/3, 1 clear; incomplete: 3 of 15 paths evaluated; never certified: PILOT (three balanced paths)) / no (3/3, 1 clear; incomplete: 3 of 15 paths evaluated; never certified: PILOT (three balanced paths)) | — / — | 0.00 / 0.33 | `deriv_r6n-a070-p_USDCHF_15m_result.json` |
| r6n-a071-p | r6n-a071-p: all (60), nseed 1 | 15m | 0.03 | .4968 (.4890) | .4688 · .333 · .5029 | 0/0 · — | .4246 · .333 · .5147 | 0/0 · — | .5008 / .4843 | PILOT (three balanced paths) / PILOT (three balanced paths) | no (3/3, 1 clear; incomplete: 3 of 15 paths evaluated; never certified: PILOT (three balanced paths)) / no (3/3, 1 clear; incomplete: 3 of 15 paths evaluated; never certified: PILOT (three balanced paths)) | — / — | 0.00 / 0.33 | `deriv_r6n-a071-p_USDCHF_15m_result.json` |
| r6n-lgb2h-p | r6n-lgb2h-p: all (60), nseed 1 | 15m | 0.03 | .5004 (.4944) | .4950 · .000 · .5062 | 0/0 · — | .4714 · .333 · .4995 | 0/0 · — | .5008 / .4843 | PILOT (three balanced paths) / PILOT (three balanced paths) | no (3/3, 0 clear; incomplete: 3 of 15 paths evaluated; never certified: PILOT (three balanced paths)) / no (3/3, 1 clear; incomplete: 3 of 15 paths evaluated; never certified: PILOT (three balanced paths)) | — / — | 0.00 / 0.00 | `deriv_r6n-lgb2h-p_USDCHF_15m_result.json` |
| r6n-logit2h-p | r6n-logit2h-p: all (60), nseed 1 | 15m | 0.03 | .4883 (.4833) | .4577 · .000 · .4777 | 0/0 · — | .4629 · .333 · .5225 | 0/0 · — | .5008 / .4843 | PILOT (three balanced paths) / PILOT (three balanced paths) | no (3/3, 0 clear; incomplete: 3 of 15 paths evaluated; never certified: PILOT (three balanced paths)) / no (3/3, 1 clear; incomplete: 3 of 15 paths evaluated; never certified: PILOT (three balanced paths)) | — / — | 0.00 / 0.33 | `deriv_r6n-logit2h-p_USDCHF_15m_result.json` |
| r6n-p102-f | r6n-p102-f: all (60), nseed 1 | 15m | 0.03 | .4969 (.4906) | .4469 · .000 · .4774 | 3/4 · .5477 | .4605 · .267 · .5014 | 0/2 · .4788 | .5007 / .4844 | KILLED / KILLED | no (15/15, 0 clear) / no (15/15, 4 clear) | — / — | 0.00 / 0.13 | `deriv_r6n-p102-f_USDCHF_15m_result.json` |
| r6n-p10266-f | r6n-p10266-f: all (60), nseed 1 | 15m | 0.03 | .4963 (.4850) | .4114 · .000 · .4824 | 1/4 · .5076 | .4274 · .200 · .4892 | 0/1 · .4578 | .5007 / .4844 | KILLED / KILLED | no (15/15, 0 clear) / no (15/15, 3 clear) | — / — | 0.00 / 0.13 | `deriv_r6n-p10266-f_USDCHF_15m_result.json` |
<!-- DERIV-FEED-TABLES:END -->
