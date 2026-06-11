# NZDUSD 15m — Results of record

---

## SCOPE
**Pair:** NZDUSD · **Horizon:** 15m · **Label:** sign(close[t+15]−close[t]), ties LOSE
**Splits:** train 2012–21 | val 2022–23 | test24 2024 | test25 2025 | oos 2026
**Breakeven:** 0.541 (deriv payout R≈1.85) · **Settlement:** bar-close approx, nonoverlap_chrono gap=900s
**Cert rule:** per-side p10≥0.541 AND ≥80% of 15 CPCV paths clear 0.541
**Selection:** VAL worst-half stability (NEVER VAL-acc-max; corr(VAL,OOS)=−0.54)
**Key files:** `sweeps/NZDUSD_15m.md` (ledger) · `sweeps/NZDUSD_15m_backlog.md` (queue) · `books/` (frozen)
**Sweep status:** OPEN — both sides CERTIFIED+FROZEN; improve+discover loops running (Asia KILLED; Antipodean pooling pending)

---

## MASTER KEY TABLE

| key | best method | book | UP p10@cov | DOWN p10@cov | status |
|-----|-------------|------|------------|--------------|--------|
| NZDUSD.m15ny.UP | NY own-pair refit-CPCV seed-ens K=3 | NZDUSD.m15ny_seedens.v1 `f599708e` | .5749 @cov2 (14/15) | — | ✅ **CERTIFIED + FROZEN** |
| NZDUSD.m15ny.DOWN | NY own-pair refit-CPCV seed-ens K=3 | NZDUSD.m15ny_seedens.v1 `f599708e` | — | .5803 @cov2 (15/15) | ✅ **CERTIFIED + FROZEN** |

---

## Context

**NZD = Pacific commodity currency (Antipodean bloc, closest cousin to AUD)**

NZD is a small open commodity economy: dairy (GDT auctions drive fortnightly spikes), wool, agriculture, tourism. Macro drivers: RBNZ rate decisions (7–8/yr, ~02:00 UTC), China PMI/trade data, NZ CPI/jobs, risk-on/off commodity flows. NZD is the USD-major most correlated with AUD (r≈0.87–0.90 on 15m returns) — the "Antipodean twin."

**Session hypothesis:** NY (USD factor, risk-flow) is the dominant prior for all USD-major own-pair family pairs (AUDUSD/USDJPY/USDCAD all confirmed NY-dominant). Asia is a secondary candidate for NZD given China-NZ trade flows and RBNZ timing, but prior is low (Asia failed for AUD and most majors).

**Cross-pair question (THE open question):**
- **Own-pair hypothesis (prior ~0.70):** NZD is Antipodean/commodity — AUDUSD was own-pair specific, same mechanism likely applies. All-session KILLED (unlike USDCHF/EURUSD which survived) = EUR-bloc pooling DEMOTED. The all-session killed pattern matches USDJPY/USDCAD (own-pair family). Closest pooling candidate if xpair is tested: AUDUSD (Antipodean cousin, not EUR-bloc).
- **EUR-bloc hypothesis (prior ~0.15):** NZD is not EUR, GBP, or CHF — very low prior. DEMOTED by A1 all-session KILLED (EUR-bloc pairs like USDCHF survived all-session).
- **AUD-cousin pooling (prior ~0.15):** Pool NZDUSD+AUDUSD (Antipodean bloc). Possible incremental if own-pair NY certifies and runs dry.

**Side asymmetry:** DOWN (=NZD weakness, risk-off) has been MORE robust in all A1 readings (test24 DOWN .6327@cov1 vs UP .4767@cov1 — UP DIES at tight cov all-session). Pattern matches commodity/risk currencies where risk-off is sharper/more forecastable. Session restriction may rescue UP (NY risk-on/off flow).

**AUC ceiling:** ~0.532–0.535 expected (own-pair family: AUDUSD .536, USDJPY .539, USDCAD .532). >65% target requires pooling lift or external data.

---

## A1 — Baseline (all-session single-pair, 239 feats)

`nzdusd_15m_base.py` | val_auc=0.5219 | gate cov5% | **VERDICT: KILLED** (no year COMB CI-lo clears BE; 2026 COMB .5099)

| split | AUC | up-rate | COMB wr (CI95) | UP wr | DOWN wr |
|-------|-----|---------|----------------|-------|---------|
| test24 | .5224 | .5003 | .5448 [.521, .568] @cov5 | .5323 @cov5 | .5722 @cov5 |
| test25 | .5186 | .5016 | .5460 [.525, .567] @cov5 | .5423 @cov5 | .5611 @cov5 |
| oos   | .5158 | .4995 | .5099 [.475, .534] @cov5 | .4726 @cov5 | .5431 @cov5 |

**A1 covcurve** (COMB / UP / DOWN):

| cov | 2024 | 2025 | 2026 |
|-----|------|------|------|
| 0.10 | .539 / .539 / .540 | .534 / .525 / .544 | .506 / .500 / .511 |
| 0.05 | .550 / .532 / .572 | .552 / .542 / .561 | .510 / .473 / .543 |
| 0.03 | .540 / .505 / .584 | .561 / .548 / .574 | .499 / .463 / .530 |
| 0.02 | .556 / .521 / .602 | .562 / .538 / .586 | .531 / .524 / .536 |
| 0.01 | .542 / .477 / .633 | .586 / .571 / .601 | .533 / .525 / .542 |

**Key reads:**
- DOWN very strong at tight cov in 2024/25 (.633/.601 @cov1) but UP DIES (.477/.571 — noisy direction)
- 2026 OOS collapses for both sides (UP dies, DOWN fades to .530–.543) — frozen-forward decay, as expected
- Pattern: identical to USDJPY/USDCAD all-session killed → session restriction is the next lever
- DOWN signal is significantly more robust than UP all-session; UP may need session + refit

---

## A9 — Session CPCV (NY) — ✅ BOTH SIDES CERTIFIED

`nzdusd_15m_cpcv_session.py ny 2 0.03,0.02,0.01 1` | N_GROUPS=6 K_TEST=2 | AUC mean=.531 | **DONE 2026-06-11**

Result JSON: `nzdusd_15m_cpcv_session_ny_multicov_result.json`

### Certification summary

| cov | side | p10 | mean | min | frac_clear_BE | med_n/path | CERTIFIED |
|-----|------|-----|------|-----|---------------|------------|-----------|
| 3% | UP | .5623 | .5916 | .5423 | 1.00 (15/15) | 737 | ✅ |
| 3% | DOWN | .5664 | .5881 | .5541 | 1.00 (15/15) | 607 | ✅ |
| 3% | COMB | .5684 | .5891 | .5530 | 1.00 (15/15) | 1324 | ✅ |
| 2% | UP | .5740 | .5875 | .5415 | 1.00 (15/15) | 481 | ✅ |
| 2% | DOWN | .5716 | .5913 | .5543 | 1.00 (15/15) | 394 | ✅ |
| 2% | COMB | .5716 | .5887 | .5537 | 1.00 (15/15) | 836 | ✅ |
| 1% | UP | .5602 | .5923 | .5422 | 1.00 (15/15) | 220 | ✅ |
| 1% | DOWN | .5594 | .5959 | .5407 | 0.933 (14/15) | 181 | ✅ |
| 1% | COMB | .5745 | .5938 | .5726 | 1.00 (15/15) | 380 | ✅ |

**Best operating point (single-seed):** cov2% — UP p10=.574, DOWN p10=.572 (both peak here).
**AUC:** mean=.531, min=.526, max=.534 — confirms own-pair family ceiling (~.532–.535).
**Uprate tripwire:** PASS (all years ∈ [.47, .53]).

### All 15 paths @cov3%

| path | g | AUC | up-rate | UP | DOWN | COMB |
|------|---|-----|---------|-----|------|------|
| 1 | [0,1] | .5332 | .5024 | n570 .5965 | n528 .5663 | n1098 .5820 |
| 2 | [0,2] | .5333 | .5017 | n673 .6092 | n540 .6111 | n1207 .6098 |
| 3 | [0,3] | .5337 | .5038 | n737 .6133 | n790 .6076 | n1526 .6101 |
| 4 | [0,4] | .5327 | .4997 | n795 .6025 | n533 .5929 | n1324 .5982 |
| 5 | [0,5] | .5287 | .5025 | n694 .6138 | n711 .5541 | n1405 .5836 |
| 6 | [1,2] | .5323 | .5040 | n757 .5958 | n521 .5950 | n1276 .5956 |
| 7 | [1,3] | .5334 | .5061 | n741 .5965 | n607 .6030 | n1345 .5985 |
| 8 | [1,4] | .5329 | .5021 | n652 .5675 | n443 .5666 | n1093 .5672 |
| 9 | [1,5] | .5301 | .5048 | n782 .5588 | n666 .5841 | n1447 .5701 |
| 10 | [2,3] | .5305 | .5053 | n499 .5952 | n621 .5878 | n1119 .5907 |
| 11 | [2,4] | .5322 | .5013 | n715 .5832 | n446 .6345 | n1159 .6022 |
| 12 | [2,5] | .5285 | .5041 | n850 .5941 | n735 .5918 | n1581 .5927 |
| 13 | [3,4] | .5277 | .5034 | n651 .5991 | n613 .5856 | n1264 .5926 |
| 14 | [3,5] | .5302 | .5062 | n898 .6058 | n815 .5730 | n1711 .5903 |
| 15 | [4,5] | .5258 | .5021 | n780 .5423 | n549 .5683 | n1329 .5530 |

**Path-15 low:** UP .5423 is the worst single path — still above BE=.541. DOWN low = path 5 .5541. Both sides never fell below BE across 15 paths.

---

## I2 — Seed-ensemble K=3 (NY) — ✅ SUPERSEDES single-seed

`nzdusd_15m_cpcv_session.py ny 2 0.03,0.02,0.01 3` | nseed=3 (seeds 0,1,2) | AUC mean=.5321 | **DONE 2026-06-11**

Result JSON: `nzdusd_15m_cpcv_session_ny_seedens3_result.json`

**SUPERSEDE rule triggered: mean lifts BOTH sides ALL 3 cov levels → SUPERSEDES A9-ny single-seed.**

| cov | side | p10 | mean | min | frac_clear_BE | med_n/path | vs single-seed p10 | CERTIFIED |
|-----|------|-----|------|-----|---------------|------------|---------------------|-----------|
| 3% | UP | .5639 | .5931 | .5295 | 0.933 (14/15) | 667 | +.0016 | ✅ |
| 3% | DOWN | .5722 | .5974 | .5682 | 1.00 (15/15) | 520 | +.0058 | ✅ |
| 3% | COMB | .5821 | .5946 | .5661 | 1.00 (15/15) | 1158 | — | ✅ |
| 2% | UP | .5749 | .5959 | .5057 | 0.933 (14/15) | 421 | +.0009 | ✅ |
| 2% | DOWN | .5803 | .5973 | .5676 | 1.00 (15/15) | 338 | +.0087 | ✅ |
| 2% | COMB | .5817 | .5959 | .5661 | 1.00 (15/15) | 714 | — | ✅ |
| 1% | UP | .5591 | .5943 | .5389 | 0.933 (14/15) | 167 | −.0011 | ✅ |
| 1% | DOWN | .5526 | .6047 | .5407 | 0.933 (14/15) | 135 | −.0068 | ✅ |
| 1% | COMB | .5811 | .5980 | .5529 | 1.00 (15/15) | 282 | — | ✅ |

**Best OP (seed-ens K=3):** cov2% — UP p10=.5749 (14/15), DOWN p10=.5803 (15/15).
**AUC:** mean=.5321, min=.5269, max=.5353 (own-pair ceiling).
**Uprate tripwire:** PASS.

**Notes:**
- UP p10 regresses slightly at cov1% (−.0011) but MEAN lifts (+.002) — tighter cov has higher path variance; overall UP is stronger in seed-ens.
- Path-15 ([g4,5] = 2021–2026 era): UP=.5295@cov3% — this is the most recent era; indicates some frozen-2021 decay sensitivity for UP.
- Seed-ens mean lifts both sides at cov3%/cov2%/cov1% → SUPERSEDE is clean.
- **Incumbent book: `NZDUSD.m15ny_seedens.v1`** (pending freeze via `nzdusd_15m_freeze_ny.py`).

---

## A9 — Session CPCV (Asia) — ❌ KILLED (completeness check)

`nzdusd_15m_cpcv_session.py asia 2 0.03,0.02,0.01 1` | N_GROUPS=6 K_TEST=2 | AUC mean≈.519 | **KILLED 2026-06-11**

Result JSON: `nzdusd_15m_cpcv_session_asia_multicov_result.json`

### Kill summary

| cov | side | p10 | mean | frac_clear_BE | med_n | CERTIFIED |
|-----|------|-----|------|---------------|-------|-----------|
| 3% | UP | .5154 | .5431 | 0.60 (9/15) | 461 | ❌ |
| 3% | DOWN | .5199 | .5509 | 0.60 (9/15) | 507 | ❌ |
| 3% | COMB | .5230 | .5472 | 0.60 (9/15) | 955 | ❌ |
| 2% | UP | .5109 | .5390 | 0.40 (6/15) | 283 | ❌ |
| 2% | DOWN | .5245 | .5516 | 0.733 (11/15) | 334 | ❌ |
| 2% | COMB | .5206 | .5463 | 0.733 (11/15) | 589 | ❌ |
| 1% | UP | .5137 | .5491 | 0.60 (9/15) | 134 | ❌ |
| 1% | DOWN | .5143 | .5521 | 0.667 (10/15) | 170 | ❌ |
| 1% | COMB | .5272 | .5509 | 0.733 (11/15) | 273 | ❌ |

**Kill reasons:** p10 max=.5245 (all below BE=.541); frac_clear max=0.733 (below 0.80 threshold). Both sides fail all cov levels.

**Key reads:**
- AUC mean≈.519 vs NY mean=.531 — Asia carries substantially less USD-factor/risk-flow content.
- Paths 10–14 (recent era 2019–2026) are weakest: UP .491/.543/.533/.534 — systematic drift downward in recent eras.
- Own-pair family pattern confirmed for 4th time: Asia KILLED for AUDUSD, USDJPY, USDCAD, now NZDUSD. NY is the exclusive carrier.
- RBNZ 02:00 UTC + China proximity do not produce tradeable binary edge at 15m horizon.

---

## A6 — Antipodean Xpair CPCV (NY) — PARTIAL LIFT — ❌ NO SUPERSEDE

`nzdusd_15m_cpcv_xpair.py ny 2 0.03,0.02,0.01 1` | nseed=1 | feats=271 (239 base + 32 AUD) | AUC mean=.5345 | **DONE 2026-06-11** | 943s

Result JSON: `nzdusd_15m_cpcv_xpair_ny_result.json`

**Verdict: DOWN CERTIFIED (all 3 cov levels), UP NOT CERTIFIED (all 3 cov levels) — NO SUPERSEDE.**

| cov | side | p10 | mean | min | frac_clear_BE | med_n/path | CERTIFIED |
|-----|------|-----|------|-----|---------------|------------|-----------|
| 3% | UP | .5356 | .5654 | .5219 | .867 (13/15) | 510 | ❌ |
| 3% | DOWN | .5735 | .5969 | .5612 | 1.0 (15/15) | 435 | ✅ |
| 3% | COMB | .5535 | .5791 | .5482 | 1.0 (15/15) | 923 | ✅ |
| 2% | UP | .5371 | .5693 | .5354 | .733 (11/15) | 338 | ❌ |
| 2% | DOWN | .5772 | .6042 | .5734 | 1.0 (15/15) | 282 | ✅ |
| 2% | COMB | .5623 | .5841 | .5587 | 1.0 (15/15) | 603 | ✅ |
| 1% | UP | .5258 | .5761 | .4961 | .600 (9/15) | 161 | ❌ |
| 1% | DOWN | .5804 | .6069 | .5598 | 1.0 (15/15) | 142 | ✅ |
| 1% | COMB | .5557 | .5890 | .5449 | 1.0 (15/15) | 297 | ✅ |

**AUC:** mean=.5345, min=.5262, max=.5405 — strong lift over own-pair base .519 (+.015). Confirms AUDUSD cousin carries real common signal.
**Uprate tripwire:** PASS.

**No SUPERSEDE:** SUPERSEDE requires BOTH sides certified. UP p10 fails all cov levels (max .5371@cov2%); DOWN p10=.5772@cov2% also below incumbent .5803. Incumbent NZDUSD.m15ny_seedens.v1 retained.

**Directional asymmetry finding:**
- **DOWN**: 15/15 paths all covs (frac=1.0). Mean escalates .597 → .604 → .607 as coverage tightens. The "both-Pacific-down" (AUD+NZD correlated risk-off) is a clean, consistent signal. AUDUSD cousin features strongly reinforce DOWN.
- **UP**: 2 catastrophically bad paths drag p10 below .541 at all covs — g[0,4]@cov3%=.5219, g[2,5]@cov3%=.5285; g[4,5]@cov1%=.4961 (below 50%). NZD UP is idiosyncratic — AUDUSD features confuse rather than help when NZD strength decouples from AUD.
- **Mechanism**: "both Pacific currencies going down" (risk-off) = joint Pacific signal; "NZD UP" often means NZD-specific strength (RBNZ, dairy flows, NZ macro) not shared with AUD.

### All 15 paths @cov3%

| path | g | AUC | up-rate | UP | DOWN | COMB |
|------|---|-----|---------|-----|------|------|
| 1 | [0,1] | .5405 | .5006 | n355 .5859 | n396 .6111 | n750 .5987 |
| 2 | [0,2] | .5396 | .5015 | n578 .5709 | n350 .6400 | n927 .5976 |
| 3 | [0,3] | .5376 | .5041 | n529 .5917 | n566 .5830 | n1093 .5874 |
| 4 | [0,4] | .5377 | .5024 | n456 **.5219** | n452 .5796 | n906 .5508 |
| 5 | [0,5] | .5271 | .5024 | n414 .5531 | n515 .5612 | n929 .5576 |
| 6 | [1,2] | .5345 | .5006 | n555 .5532 | n330 .6303 | n885 .5819 |
| 7 | [1,3] | .5383 | .5032 | n360 .6000 | n453 .5872 | n812 .5924 |
| 8 | [1,4] | .5390 | .5015 | n348 .5862 | n371 .6361 | n719 .6120 |
| 9 | [1,5] | .5289 | .5015 | n510 .5667 | n428 .5724 | n938 .5704 |
| 10 | [2,3] | .5382 | .5040 | n587 .5520 | n499 .6032 | n1086 .5755 |
| 11 | [2,4] | .5371 | .5023 | n525 .5600 | n400 .5750 | n923 .5655 |
| 12 | [2,5] | .5273 | .5024 | n719 **.5285** | n382 .5864 | n1100 .5482 |
| 13 | [3,4] | .5357 | .5050 | n349 .5989 | n510 .5980 | n859 .5984 |
| 14 | [3,5] | .5301 | .5050 | n520 .5654 | n549 .5756 | n1069 .5706 |
| 15 | [4,5] | .5262 | .5033 | n465 .5462 | n435 .6138 | n900 .5789 |

**Failed UP paths at cov3%:** g[0,4]=.5219 and g[2,5]=.5285 (both below BE=.541).

---

## Books

| book | status | UP p10@cov | DOWN p10@cov | frozen-fwd COMB | frozen-fwd UP | frozen-fwd DOWN |
|------|--------|------------|--------------|-----------------|---------------|-----------------|
| NZDUSD.m15ny_seedens.v1 | ✅ FROZEN | .5749@cov2 | .5803@cov2 | .621→.602→.519 (24/25/26) | .565→.570→.497 | .718→.640→.542 |

### Freeze artifact details (NZDUSD.m15ny_seedens.v1)
`nzdusd_15m_freeze_ny.py nseed=3` | VAL(NY) AUC=0.5362 | cov2% thr=0.0835 | content_id=f599708e8a9c3e3f | 120s

| split | COMB wr (CI95) | n | UP wr (CI95) | n | DOWN wr (CI95) | n |
|-------|----------------|---|--------------|---|-----------------|---|
| test24 | .6219 [.575,.669] | 402 | .5652 [.506,.624] | 253 | .7181 [.644,.792] | 149 |
| test25 | .6019 [.560,.642] | 525 | .5699 [.514,.626] | 286 | .6402 [.582,.699] | 239 |
| oos (2026) | .5194 [.463,.576] | 283 | .4965 [.418,.575] | 141 | .5423 [.465,.620] | 142 |

**Key reads:**
- **DOWN frozen-forward STRONG:** .718 → .640 → .542 — DOWN retains >64% through 2025 but falls below BE in 2026 (frozen-2021 vintage decay). This is the expected pattern for refit-dependent books.
- **UP frozen-forward WEAK:** .565 → .570 → .497 — UP is borderline frozen. .497 in 2026 = below 50% (dead). Confirms UP edge is particularly regime-sensitive; retraining is mandatory for live UP deployment.
- **COMBINED frozen-forward:** .621 → .602 → .519 — above BE through 2025, below in 2026. Consistent with all own-pair family frozen books.
- **Training note:** early stopping hit at iterations 150/73/89 (out of 3000) — fast convergence on full train set; frozen-2021 signal structure not saturated.
- **REFIT-DEPENDENT:** size on the refit per-era floor (.5749/.5803), NOT these frozen numbers. Periodic retraining every 12-18 months is required for live deployment.
- **Seed-ens gap:** 3 models (seeds 0,1,2) in `models/m15ny_NZDUSD_s{0,1,2}_lgb.txt`. Deploy as probability mean.

---

## Supersessions / KILLS log

| experiment | result | date |
|-----------|--------|------|
| A1 all-session base | KILLED (val_auc=.5219; no CI-lo clears BE; 2026 .5099) | 2026-06-11 |
| A9-ny single-seed | ✅ CERTIFIED — UP p10=.574@cov2, DOWN p10=.572@cov2 — 15/15 paths both sides | 2026-06-11 |
| I2 seed-ens K=3 | ✅ SUPERSEDES single-seed — mean lifts both sides all 3 covs; UP p10=.5749, DOWN p10=.5803 @cov2; incumbent book NZDUSD.m15ny_seedens.v1 | 2026-06-11 |
| A9-asia | KILLED — UP p10=.5154 frac=0.60; DOWN p10=.5245 frac=0.733. Both sides fail all 3 covs. AUC mean≈.519 vs NY .531. Own-pair Asia pattern confirmed (4th major). | 2026-06-11 |
| A6 xpair NY (nseed=1) | PARTIAL LIFT — DOWN CERT all covs (p10=.5772@cov2%, frac=1.0); UP NOT CERT all covs (p10=.5371@cov2%, frac=.733). AUC mean=.5345 (+.015 over base). NO SUPERSEDE: UP fails + DOWN p10 < incumbent .5803. Directional asymmetry: DOWN=Pacific risk-off joint signal (strong, frac=1.0 all covs); UP=NZD-idiosyncratic (AUDUSD features confuse). | 2026-06-11 |
