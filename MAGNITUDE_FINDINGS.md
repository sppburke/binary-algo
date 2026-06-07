> **SCOPE: per-currency MAGNITUDE findings** (sign-invariant → no UP/DOWN key; label each result by PAIR). EURUSD + USDJPY (USDJPY·1m added 2026-06-04, `usdjpy_1m_magnitude_result.json`; USDJPY·2m added 2026-06-05, `usdjpy_2m_magnitude_result.json`). Generic theory: THEORY.md. See REPO_MAP.md.

# MAGNITUDE FINDINGS — the one deflation-proof edge in the binary-algo program

**Status:** documentation handoff. The user has decided NOT to build a trading product right now.
This file exists so a future session can pick up the **magnitude (move-SIZE) edge** cold.

**Evidence rule used here:** every number is tagged with the on-disk file it traces to and a tier.
- **VERIFIED (Tier-1):** read verbatim this session from a `*_result.json` or from code at the current commit.
- **LOG-RECORDED (Tier-3):** stated in a research-log / ledger `.md` file from a prior run, but the producing
  script prints to **stdout only** and writes **no result file** — so it was *not* re-confirmed by execution this session.
  Treat as plausible and self-consistent, **not** as proven-on-disk. **To promote: re-run the script and redirect stdout to a result file, then cite that file.**

The single fully-certified, execution-independent magnitude result is the **30-minute CPCV block in
`cpcv_certify_result.json`**. Everything else is LOG-RECORDED.

---

## 2026-06-06 — DST-correct session re-campaign + Kronos look-forward fix + full-suite audit

**Headline for THIS file: the sign-invariant MAGNITUDE edge is SESSION-ROBUST.** Re-running the whole magnitude
suite under DST-correct per-session masks (`sessions.py`: NY=8–17 `America/New_York`, LDN=8–16 `Europe/London`,
Asia=9–18 `Asia/Tokyo`; `session_mask` = local-tz hour, applied per-day across train+val+test+oos) does **not** break
magnitude. It certifies in **NY, LDN AND Asia** at **1m / 2m / 5m / 10m / 30m** — every session, every horizon tested.
Direction, by contrast, is null/killed in every non-NY session (and the edge that survives at all is decisively
NY-concentrated). The size edge is the one that travels across the clock.

**MAGNITUDE — certified in all three sessions, every horizon `[EURUSD]` (VERIFIED, Tier-1):**

| Horizon | Substrate / script | NY magAUC | LDN magAUC | Asia magAUC | Cert (cov≤10% frac_clear) | Result files |
|---|---|---|---|---|---|---|
| **1 m** | tick GBM (`session_1m.py`) | **0.675** | **0.728** | **0.717** | **1.0** all (p10 .60–.71) | `session_1m_mag_{ny,ldn,asia}_result.json` |
| **2 m** | tick GBM (`session_2m.py`) | certified | certified | certified | **1.0** all (p10 .58–.70) | `session_2m_mag_{ny,ldn,asia}_result.json` |
| **5 m** | base bar GBM (`session_bars.py`) | certified | certified | certified | **1.0** all (p10 .72–.80) | `session_5m_mag_{ny,ldn,asia}_result.json` |
| **10 m** | base bar GBM (`session_bars.py`) | certified | certified | certified | **1.0** all (p10 .72–.80) | `session_10m_mag_{ny,ldn,asia}_result.json` |
| **30 m** | base bar GBM (`session_bars.py`) | certified | certified | certified | **1.0** all (p10 .72–.80) | `session_30m_mag_{ny,ldn,asia}_result.json` |

Net: magAUC across sessions lands in **0.67–0.80**, `frac_clear` = **1.0** at cov ≤ 10% in **all** sessions and horizons.
Session-conditioning does not degrade the size edge — it is robust to which trading session the decision row falls in.
(15m base-GBM magnitude per-session is **in progress**.)

**DIRECTION (contrast, the sign-invariance signature persists per-session):** session-conditioned direction is
**null/KILLED in every session** for the tick + base-bar GBMs (1m pooled ~.504, p10 .500–.504, frac_clear 0.0; 2m p10
.496–.500; 5m/10m/30m base bar KILLED all, NY strongest e.g. 10m NY p10 .525). The only direction edge that survives is
the **cross-pair book**, and it is **NY-concentrated only** (`session_xpair.py`, STRICT session-only DST-correct):
NY certifies BOTH sides at every ≥10m horizon (10m NY UP .6053/DOWN .5896, 15m UP .5845/DOWN .5712, 30m UP .5681/DOWN
.5639, 15/15), while **LDN and Asia certify nothing** (~.49–.53). So within the same session masks where magnitude clears
1.0 everywhere, direction collapses outside NY — a clean operational restatement of sign-invariance.
(5m + 2m cross-pair sessions **in progress**.)

**KRONOS look-forward fix — DOES NOT affect any magnitude result (clean-up, recorded for completeness).** A user-caught
1-bar look-forward MISALIGNMENT in `kronos_dir.py` / `kronos_ft.py` (failure-mode **FM-F forecast-derivation**: it scored
`Pup = pred_close(i) > C[i-1]`, the move INTO the entry over window `[t[i-1],t[i]]`, while the deriv label `y[i]` is the
DISJOINT forward window `[t[i]+1, t[i]+61]`) produced a FALSE NULL — never a false positive. Fixed in `kronos_mtf.py`
(context ends AT bar i, predict H forward steps, `Pup = pred_close(+H) > C[i]`; forward label agrees with next-bar sign
**92.3%**, n=233,950). The corrected Kronos direction eval is **null at every horizon** (1/5/10/15/30m, zero-shot AND
fine-tuned, all sessions; pooled .50–.51, CPCV p10 .489–.500, all KILLED) — Kronos ingests only EURUSD's own OHLCV
candles, not the 7-pair cross-section that carries the NY direction edge. **This is a DIRECTION bug; no magnitude script
is in the Kronos family.** (Kronos FINE-mode chains + multi-TF ensembles **in progress**.)

**Full-suite correctness audit (19-agent workflow + lead Tier-1 proofs): NO magnitude cert invalidated.** The FM-F bug
is **isolated to the 2 Kronos scripts** — across all 308 scripts + a dedicated forecast-derivation sweep, no second
instance was found; GBM/CNN magnitude classifiers are trained DIRECTLY on the label and are structurally immune to FM-F.
The MAGNITUDE substrate was proven clean empirically: the BAR label is forward (0/2000 mismatch at H=5/10/30,
corr(y,future) ~.99 vs ~−.02, up-rate .498–.506; `harness` features + `contig_fwd`), the TICK label is forward
(corr(y,future) .486 vs corr(y,past) −.003), and BAR-feature causality holds (truncation `max|full−trunc| = 0.0` across
all 239 features). **Bounded flag touching this file:** `barcnn_mag.py:163` carries an FM-E selective-threshold chosen on
pooled test+oos for the MAGNITUDE target — it is **CPCV-deflated** (the `barcnn_mag_ohlcabs_result.json` cert in §3 uses
the 28-path purged distribution, not the selective point), so the 60s bar-image cert is **NOT** invalidated; the noted
fix is to set the selective threshold on VAL only. No certified magnitude (or direction) book is invalidated by the audit.

**New scripts this session (magnitude-relevant):** `sessions.py` (DST-correct `session_mask`/`SESSIONS`);
`session_1m.py`, `session_2m.py` (per-session tick GBM, dir+mag); `session_bars.py` (per-session bar GBM, any H, dir+mag);
`session_xpair.py` (cross-pair STRICT session-only, direction); plus the Kronos-family `kronos_ft.py`, `kronos_mtf.py`,
`kronos_bars.py`, `kronos_ensemble.py` and `barcnn_run.py` SESSION arg (direction-side, listed for provenance only).

**Hardware note:** box now has an NVIDIA RTX 5050 Laptop (8GB, Blackwell sm_120, CUDA13); venv torch swapped to
`2.12.0+cu130` so Kronos FT+inference runs on GPU. No bearing on the GBM magnitude results above.

---

## 1. TL;DR — the one certified edge

**Move SIZE (|return|) is forecastable out-of-sample and survives full Lopez-de-Prado deflation. Move SIGN (direction) is not.**

The headline, **certified** number (30-minute horizon, EURUSD, 2012–2026 pooled):

| Metric | Value | Source (Tier-1) |
|---|---|---|
| CPCV 28-path large-move AUC, **mean** | **0.7439** | `cpcv_certify_result.json` → `magnitude_30m.auc_mean` (0.7438972861624615) |
| CPCV 28-path AUC, p10 / min / max | 0.7177 / 0.7054 / 0.7819 | `magnitude_30m.auc_p10/auc_min/auc_max` |
| **Deflated expectation** (after E[max-of-N] + anti-selection penalty) | **0.7124** | `deflation.magnitude_auc.deflated_expectation` |
| Deflation bar | 0.55 | `deflation.magnitude_auc.bar` |
| p10 clears bar / deflated clears bar / P(typical path clears) | true / true / **1.0** | `deflation.magnitude_auc.*` |
| Top-vs-bottom-decile realized-|ret| **lift, mean** | **5.013×** | `magnitude_30m.lift_mean` (5.012685605457851) |
| Decile lift, p10 | 3.96× | `magnitude_30m.lift_p10` (3.955762839317322) |

**What it IS:** a forecast of *how big* the next 30-minute move will be (is |ret30| in the top quartile?).
The predictor that actually carries the signal is **recent realized volatility** (rv30, rv120). Every one of the
28 purged-combinatorial out-of-sample paths clears the bar; the headline 0.79 deflates to 0.712 and still clears with a wide margin.

**What it is NOT:** a direction (up/down) edge. At ≥5 minutes, sign is ~efficient-market. In the *same* CPCV run, the
celebrated 15-minute direction "selective book" headline 0.647 **deflates to a 28-path mean 0.5455 with p10 0.5306, below
the 0.541 break-even — it FAILS** (`deflation.direction_selective.p10_clears_bar = false`). Magnitude is the only edge in
the program that survives deflation, and it survives by a large margin.

**Tradeability caveat (from MEMORY, not on disk):** Deriv forex Rise/Fall has a **15-minute minimum expiry and is
directional**, so the magnitude edge is **not** tradeable as an up/down binary. Magnitude pays on Touch/No-Touch,
Range/Boundary, straddle/strangle, or variance-risk-premium structures (Section 6).

---

## 2. The sign-invariance THEORY — why magnitude is forecastable but direction is efficient

**Source:** arXiv:2512.15720 (Dec 2025), cited verbatim in the `m30_magnitude.py` docstring (lines 1–6) and in
`DIRECTION_FINDINGS.md`, `METHODS_CATALOG.md`, `EXPERIMENT_LEDGER.md`, `min1_research_log.md`.

**Statement (as recorded in `m30_magnitude.py` docstring, Tier-1 code):**
> "order-flow/permutation entropy is sign-invariant → gates |move| (volatility), not sign."

**Mechanism.** A large family of features — order-flow imbalance, permutation entropy, complexity / Hurst / autocorrelation,
HMM regime states, Kalman-filter measures — are **invariant under a sign permutation of the return series**. Flip the signs
of the returns and these statistics are (approximately) unchanged. Therefore they can detect the **presence and SIZE of an
informed / volatile move** but carry **no information about its DIRECTION**.

**Consequence.** Every complexity / regime / entropy / HMM / Hurst / Kalman gate is **null for direction yet positive for
magnitude** — they gate volatility, not sign. This makes |return| (a sign-invariant target) forecastable while sign stays ~EMH.

**Empirical confirmations on EURUSD (multiple independent ways — see Section 3 for the AUCs):**
1. `m10_magdir.py` — magnitude AUC 0.71–0.81 across held-out years, while direction accuracy is FLAT ~0.51–0.53 across
   *all* magnitude quartiles (the biggest-move bars are NOT more directionally predictable). [LOG-RECORDED: `m10_research_log.md` 186–191]
2. `_redteam_magdir60.py` — at 60s, magAUC 0.787 vs dirAUC 0.510 **on identical data**. [LOG-RECORDED: `min1_research_log.md` 84]
3. `min1_hmm.py` — Gaussian-HMM latent states have train P(up) ≈ 0.497–0.499 in all states (states are size/vol regimes, not direction). [LOG-RECORDED, per Gather B notes]
4. `m30_complexity.py` — flat direction accuracy ~0.515 across all PE/autocorr/Hurst bins. [LOG-RECORDED: ledger row 114]
5. Macro-news (`m5_news.py`, `min1_news60.py`) — FX prices a surprise within ~1 minute and the surprise is a magnitude/volatility
   event; sign-invariance holds even for fundamental news. [LOG-RECORDED]

**Important nuance — the theory's named mechanism is NOT what carries the EURUSD signal.** The theorem frames
*entropy/complexity* as the magnitude gate, and the CPCV feature set retains permutation-entropy (`-pe`) as a predictor.
But on EURUSD, **permutation entropy is NULL for magnitude** (corr ~0.01, AUC ~0.50) — the SPY-style entropy result does
**not** replicate on 1-minute FX returns. **Realized volatility (rv30, rv120) is the predictor that actually carries the
0.744 magnitude AUC.** [Source: `IDEAS_LOG.md` lines 82–86, LOG-RECORDED.] So the *edge* is real and certified; the
*mechanism on this asset* is volatility clustering / persistence, not entropy, despite the theorem framing.

**Magnitude→direction GATE bridge tested & null `[EURUSD·5m]` (2026-06-02, R1, `m5_magdyn_result.json`):** a 5m
magnitude model used as a DYNAMIC confidence-threshold on the certified m5xp direction gate added nothing — among
the book's confident UP bars, win-rate is FLAT across mag-forecast quartiles (.628/.646/.640/.632, lift +0.003), and
a VAL-tuned magnitude blend ANTI-transferred (binding-2025 .588 < fixed-conf .603). A direct operational confirmation
of sign-invariance: forecasting move SIZE does not make the SIGN more inferable. Magnitude stays a SIZE edge only.

---

## 3. EVIDENCE TABLE — magnitude result at every horizon

Target convention: **large-move = |forward return| ≥ a high quantile of TRAIN-window |ret|.** CPCV/m30/m10/redteam use
**top-quartile (Q75)**; production min1/min2 use **top-tercile (P67)**. Direction numbers are included only to show the
sign-invariance contrast — they are NOT the magnitude edge.

| Horizon | Magnitude AUC | Direction (contrast) | Target | Status | Source file |
|---|---|---|---|---|---|
| **1–5 s** | n/a (no pure-magnitude target below 60s) | direction ~0.65 (real, the seconds edge) | — | — | `METHODS_CATALOG.md` L50, `DIRECTION_FINDINGS.md` L13,148,163–165 — **this is a DIRECTION finding, listed only to mark no <60s magnitude target exists** |
| **60 s** | magAUC **0.7870** (vs dirAUC **0.5096** same VAL — cleanest sign-invariance proof) | dir ceiling 0.512→0.517 | \|ret60\| ≥ Q75-bucket | **VERIFIED (Tier-1)** | **`magnitude_verified.json` → `min1_60s_magnitude_auc`** (re-run + captured this session; `/tmp/mag_60s.log`: `VAL dirAUC=0.5096 magAUC=0.7870`). Script `_redteam_magdir60.py`. |
| **60 s [EURUSD] — BAR-IMAGE CNN** | magAUC **0.6985 / 0.7141 / 0.686** (VAL/test/oos); selective large-call **prediction-rate 0.68→0.80**, **all 28 CPCV purged paths ≥ 0.65 at cov ≤ 0.2 in every held-out year** (cov20% 2024 .717/2025 .722/2026 .674, p10 .678; cov5% .795/.807/.766, p10 .773) | same bars: dirAUC ≈ **.50** (image carries SIZE not SIGN) | \|ret60\| ≥ TRAIN-median (balanced) | **VERIFIED (Tier-1, CPCV-certified)** | **`barcnn_mag_ohlcabs_result.json`**. A 2-D CNN on a 30-bar ABSOLUTE-scale OHLC image (`barcnn_mag.py ohlcabs`) — recovers ~0.70 of the GBM's 0.787 via a pure bar-image 4th model class. Per-window min-max (`ohlc`) only reaches .64 (normalization strips the vol scale); preserving absolute scale crosses 65%. The SAME method's direction is null → single-method sign-invariance proof. Method: `METHODS_CATALOG.md` §5.5. |
| **120 s** | magAUC **0.682** | dir 0.510/0.509/0.512 | \|ret120\| ≥ p67 | **LOG-RECORDED** | `EXPERIMENT_LEDGER.md` row 38 (`min2_v1.py`). No result JSON. ⚠ a magnitude *gate on direction* HURTS at 120s (opposite of 60s) — ledger rows 41–42. |
| **5 m** | **NONE — no standalone 5m \|ret\| classifier was ever built** | dir ~0.519 | — | **UNVERIFIED / ABSENT** | searched `m5_research_log.md` (L225–227), `EXPERIMENT_LEDGER.md` — only direction + qualitative "news = magnitude event". The "0.73–0.79 everywhere" phrasing is **interpolation**, not a measured 5m number. |
| **10 m** | magAUC **0.813 (2024) / 0.741 (2025) / 0.706 (2026)** | dir FLAT 0.51–0.53 across all mag quartiles | \|fwd10\| ≥ Q75 (train 2012–2021) | **VERIFIED (Tier-1)** | **`magnitude_verified.json` → `m10_magnitude_auc`** (re-run + captured this session, `/tmp/mag_m10.log`). Script `m10_magdir.py` (700-tree LGBM, 239 features). |
| **30 m (CPCV — THE certified edge)** | **AUC mean 0.7439, p10 0.7177, min 0.7054, max 0.7819; deflated 0.7124; 5.013× decile lift** | dir 15m: 28-path mean 0.5198; selective book deflates 0.647→0.5455 (FAILS) | \|ret30\| ≥ train-Q75 | **VERIFIED (Tier-1)** | **`cpcv_certify_result.json` → `magnitude_30m` + `deflation.magnitude_auc`** (read verbatim this session). Producer: `cpcv_certify.py`. |
| **30 m (original chronological-split, pre-CPCV)** | rv30 large-move AUC **0.750/0.746/0.783/0.729/0.729** (train/val/t24/t25/oos); corr(rv30, future\|ret\|) +0.417/+0.441/+0.437/+0.398/+0.378 | sign ~0.515 (EMH) | \|ret30\| ≥ train-Q75 | **VERIFIED (Tier-1)** | **`magnitude_verified.json` → `m30_rv30_largemove_auc`** (re-run + captured this session, `/tmp/mag_m30.log`). Script `m30_magnitude.py`. PE corr ~0.01 / AUC ~0.50 (PE NULL on FX — rv is the real predictor). |
| **60 s production artifact** | **NO AUC stored on disk** (model exists, unscored) | — | \|ret60\| ≥ TRAIN-P67, thr=1.2034e-4 | **UNVERIFIED** | `models/min1_EURUSD_magnitude.joblib` exists (8.7 MB). `min1_EURUSD_strategy.json` stores only `mag_top_tercile_thr=1.2034e-4`; its `val_auc_inregime=0.5052` is the **DIRECTION** model — do NOT attribute it to magnitude. |
| **120 s production artifact** | **NO AUC stored on disk** | — | \|ret120\| ≥ TRAIN-P67, thr=1.7058e-4 | **UNVERIFIED** | `models/min2_EURUSD_magnitude.joblib` exists (8.0 MB). `min2_EURUSD_strategy.json`: `mag_top_tercile_thr=1.7058e-4`; `val_auc_inregime=0.5111` is DIRECTION. |
| **1 m [USDJPY·1m]** | magAUC OOS **Q67 0.716 / Q75 0.730 / Q90 0.790** (VAL 0.740/0.752/0.802; t24 0.764/0.778/0.825; t25 0.707/0.720/0.771; top-decile \|ret\| lift ~2.1–2.5×) | dirAUC ~0.515–0.52 (UP cov1% .545 SUB-breakeven; DOWN dead) | \|ret60\| ≥ train-Q{67,75,90} | **VERIFIED (Tier-1)** | **`usdjpy_1m_magnitude_result.json`** (this session). Script `usdjpy_1m_magnitude.py` (255-leaf LGBM, 239 bar feats, bar-based — no full USDJPY tick). USDJPY reproduces the EURUSD pattern: strong sign-invariant SIZE edge, no tradeable direction. |
| **2 m [USDJPY·2m]** | magAUC OOS **Q75 0.722 / Q90 0.785** (VAL 0.755/0.804; t24 0.780/0.827; t25 0.718/0.772; top-decile \|ret\| lift ~2.0–2.5×) | dirAUC ~0.524 (cross-pair-pooled CPCV win-rate mean .546 / **p10 .531 SUB-breakeven both sides**; ARF .508) | \|ret120\| ≥ train-Q{75,90} | **VERIFIED (Tier-1)** | **`usdjpy_2m_magnitude_result.json`** (2026-06-05). Script `usdjpy_2m_magnitude.py`. The sign-invariance signature holds at 2m: magAUC ~0.78 vs dirAUC ~0.52 on identical data. Direction is REAL-but-sub-BE here (unlike 1m near-efficient), but magnitude is the strong edge. |

**Net of the table:** magnitude AUC is comfortably **>0.65 at every horizon where it was measured** (60s 0.787, 10m 0.71–0.81,
30m 0.74 CPCV-certified / 0.73–0.78 single-split). Direction at the same horizons is ~0.51–0.52. **Verification status (this
session): the 30m CPCV result is deflation-certified (`cpcv_certify_result.json`); the 60s / 10m / 30m-single-split AUCs were
re-run-and-captured into `magnitude_verified.json` (Tier-1) — they MATCH the prior log-recorded values exactly. Still
LOG-RECORDED/unscored: 120s magAUC 0.682 (`min2_v1.py`, not re-captured) and the production `min1/min2_EURUSD_magnitude.joblib`
artifacts (carry no AUC on disk; their `val_auc_inregime` is the DIRECTION model — do not attribute to magnitude).** No standalone
5m magnitude classifier was ever built; "0.73–0.79 everywhere" is interpolation, not a measured 5m number.

---

## 4. The CPCV CERTIFICATION — why magnitude is the only deflation-proof edge

**Producer:** `cpcv_certify.py`. **Output (Tier-1, read this session):** `cpcv_certify_result.json`.

**Method (from `cpcv_certify.py` code + the `params` block in the JSON):**
- **CombinatorialPurgedCV:** `n_groups=8`, `k_test=2` → **C(8,2) = 28 purged-combinatorial OOS paths**.
- **Purge + embargo:** `embargo = 1 label-horizon` (line 153: `embargo_s = horizon_min*60`). Train rows whose
  `[t, t+horizon]` outcome window overlaps any test block are dropped, plus an embargo (lines 138–149).
- **Pooled** EURUSD 2012–2026, `subsample=100000`.
- **Deflation** (`deflated_metric()`, lines 270+): E[max-of-N] inflation with `n_trials=70`, plus an anti-selection
  penalty from `corr(VAL,OOS) = -0.54` (negative correlation → selecting the val-best path *hurts* OOS, so it is penalized).

**The 28-path magnitude AUC distribution (`magnitude_30m.all_aucs`, all 28 values present in the JSON):**
```
0.7414 0.7054 0.7101 0.7450 0.7298 0.7350 0.7197 0.7458 0.7485 0.7819
0.7678 0.7720 0.7582 0.7132 0.7508 0.7327 0.7390 0.7230 0.7482 0.7370
0.7391 0.7243 0.7706 0.7748 0.7580 0.7610 0.7456 0.7512
```
Mean 0.7439, std 0.02, min 0.7054, max 0.7819 — **every path is well above the 0.55 bar.**

**Deflation arithmetic (`deflation.magnitude_auc`):**
```
headline 0.79  →  path_mean 0.7439
              −  emax_inflation 0.0583   (E[max of 70 trials] penalty)
              −  neg_corr_penalty 0.0315 (corr(VAL,OOS)=-0.54 anti-selection)
              =  deflated_expectation 0.7124
bar 0.55 ; p10_clears_bar true ; deflated_exp_clears_bar true ; prob_typical_path_clears_bar 1.0
```

**Decile lift (`magnitude_30m`):** sort by predicted-magnitude; the **top decile realizes 5.013× the |ret| of the bottom
decile** (mean), p10 across paths still **3.96×**. This is the economically meaningful number for a volatility/straddle product.

**Why magnitude is the ONLY survivor — the same JSON, same machinery, on the direction books:**

| Book | Headline | 28-path mean | p10 | Bar | Clears? | JSON key |
|---|---|---|---|---|---|---|
| **Magnitude 30m** | 0.79 | **0.7439** | **0.7177** (deflated **0.7124**) | 0.55 | **YES, every path** | `deflation.magnitude_auc` |
| Direction 15m **(re-impl, single LGBM, q33, all-era)** | — | 0.5455 | 0.5306 | 0.541 | not a faithful test of `m15_production` | `deflation.direction_selective` |
| Direction 15m **raw AUC (re-impl)** | — | 0.5198 | 0.5153 (deflated 0.5146) | 0.50 | marginal | `deflation.direction_auc` |
| 5m cross-horizon **stack** | 0.648 | 0.5455 | 0.5306 | 0.541 | likely overfit (thin n45) | `deflation.stack5m_pbo` |

**⚠ CORRECTION (2026-05-31):** the MAGNITUDE row above is a FAITHFUL CPCV (it certified the actual rv30/rv120 predictors) — that
result is solid. The DIRECTION rows are **NOT** a faithful test of the real `m15_production` book: the CPCV used a **single LGBM**
(not the 3-model ensemble), a looser **q33** gate (not the VAL-tuned q{10/20/33}), and **all-era pooled** data (not the recent
held-out). The actual `m15_production backtest` reproduces **0.647** (2024 0.689 / 2025 0.582 / 2026 0.663, n677, CI[.612,.684],
`/tmp/m15_verify.log`) — real and live-faithful. So magnitude being the one *deflation-certified* edge stands, but do NOT read these
direction rows as deflating the 0.647 book; they only flag that the 15m edge is regime-dependent. A faithful CPCV of the actual
ensemble is the correct follow-up.

---

## 5. HOW IT'S BUILT — exact target, model, features, data, reproduce

### 5a. The certified 30m CPCV book (the one to trust)
- **Target:** `large-move = (|ret30| ≥ train-Q75)`. Q75 is computed on the **TRAIN window of each CPCV path only**
  (no leakage of the threshold). `ret30 = close[t+30]/close[t] − 1` over **30-minute** (30 one-minute-bar) horizon;
  rows require a *contiguous* 30-bar window (`secs[HOR:]−secs[:-HOR] == HOR*60`).
- **Features (3 predictors, `load_magnitude(hor=30)`, `cpcv_certify.py` line 120):**
  `np.column_stack([-pe[valid], rv30[valid], rv120[valid]])`
  - `-pe` = **negated** permutation entropy (d=4, tau=1, W=120 window); sign-flip because LOW PE → large move.
    **On EURUSD this column is effectively inert** (PE is NULL for FX magnitude — see Section 2 nuance).
  - `rv30` = 30-bar rolling std of log-returns. **This carries the signal.**
  - `rv120` = 120-bar rolling std of log-returns.
- **Model:** `mk_lgb(n_estimators=600)` (`cpcv_certify.py` lines 168–172, 247): LightGBM binary, `metric=auc`,
  `learning_rate=0.03`, `num_leaves=255`, `n_jobs=20`. One model fit per purged path (28 fits).
- **Eval:** ROC-AUC of predicted-large-move probability vs the binary target on each path's purged OOS block;
  plus top/bottom-decile realized-|ret| lift.

### 5b. The original chronological-split experiment (LOG-RECORDED)
- **Script:** `m30_magnitude.py`. **Windows** (code lines 13): train=2016–2021, val=2022–2023, test24=2024, test25=2025, oos=2026.
- Computes `corr(predictor, future|ret|)` and single-predictor large-move AUC for `pe`, `rv30`, `rv120` per window,
  plus mean |ret| by PE quintile. **The predictor IS the score (no LGBM in this script).**
- **Writes nothing to disk** (prints to stdout). To verify its 0.75/0.75/0.78/0.73/0.73 numbers, run with stdout captured.

### 5c. The per-horizon magnitude scripts (LOG-RECORDED)
- `m10_magdir.py` — 10m magnitude LGBM (700 trees, `num_leaves=255`, lr=0.03) on the **full 239-feature set**
  (`harness.feature_cols('EURUSD')`); also runs the magnitude-bucketed direction-null test.
- `min1_v10.py` — 60s magnitude LGBM on `|ret60| ≥ p67`; caches `models/probs_min1_mag.npz`.
- `min2_v1.py` — 120s magnitude LGBM on `|ret120| ≥ p67`; caches `models/probs_min2_v1.npz`.
- `_redteam_magdir60.py` — deriv-faithful 60s magnitude-vs-direction ceiling map; imports `min1_production`.

### 5d. The production magnitude artifacts (exist, UNSCORED)
- `min1_production.py` `train()` lines 188–192: `magthr = nanpercentile(|ret60|, 67)`; `M = mk_lgb(2500)`
  (`num_leaves=350`, `lr=0.02`, early_stop); `joblib.dump(M, 'min1_EURUSD_magnitude.joblib')`. The thr is written to
  `min1_EURUSD_strategy.json:mag_top_tercile_thr` (1.2034e-4). **Comment in code: "kept for info".** No AUC is scored or stored.
- `min2_production.py` — analogous, `mag_top_tercile_thr=1.7058e-4`.
- ⚠ **Do not attribute any magnitude AUC to these saved artifacts.** They are P67-tercile (not Q75-quartile) and unscored.

### 5e. Data + environment
- **Feature parquets:** `harness.H.FEAT_DIR/EURUSD_{YEAR}.parquet`, years 2012–2026 (`close` column + 239 feature cols).
  `harness.py` (module `H`) provides `FEAT_DIR`, `feature_cols(PAIR)` (239 names), `META_COLS`.
  **These are gitignored** (`DIRECTION_FINDINGS.md` L195: "models/ + *.parquet/*.npz are gitignored; regenerate from the scripts").
- **Venv:** `~/binary-algo-venv/bin/python`.

### 5f. Reproduce commands
```bash
# PRIMARY — the certified 30m magnitude result (regenerates cpcv_certify_result.json):
~/binary-algo-venv/bin/python /media/sean/CORSAIR/binary-algo/cpcv_certify.py

# SECONDARY single-split magnitude scripts — print to STDOUT ONLY, no result file written.
# To PROMOTE their LOG-RECORDED numbers to VERIFIED, redirect stdout to a result file and cite it:
~/binary-algo-venv/bin/python m30_magnitude.py   > m30_magnitude_result.txt   # 30m rv30/rv120/pe AUC by window
~/binary-algo-venv/bin/python m10_magdir.py      > m10_magdir_result.txt      # 10m |fwd10|>=Q75 magAUC 0.813/0.741/0.706
~/binary-algo-venv/bin/python _redteam_magdir60.py > redteam_magdir60_result.txt  # 60s magAUC 0.787 vs dirAUC 0.510
~/binary-algo-venv/bin/python min1_v10.py        > min1_v10_result.txt         # 60s |ret60|>=p67 val 0.680
~/binary-algo-venv/bin/python min2_v1.py         > min2_v1_result.txt          # 120s |ret120|>=p67 0.682

# PRODUCTION (already-trained, unscored magnitude artifacts):
~/binary-algo-venv/bin/python min1_production.py   # saves models/min1_EURUSD_magnitude.joblib (P67)
~/binary-algo-venv/bin/python min2_production.py   # saves models/min2_EURUSD_magnitude.joblib (P67)
```
**NOTE:** scripts were NOT executed this session (parquet data is gitignored / on-disk dependent). The 30m CPCV numbers
were read from `cpcv_certify_result.json` (Tier-1); all other AUCs are research-log records of prior runs (Tier-3).

---

## 6. PRODUCTIZATION (documentation only — NOT to build now)

**The magnitude edge is NOT a directional product.** It forecasts |move|, so it pays on structures whose payoff depends on
how far price travels (or doesn't), regardless of sign.

**Deriv constraint (from MEMORY + `DIRECTION_FINDINGS.md` L70, `EXPERIMENT_LEDGER.md` L7):**
Deriv **forex Rise/Fall minimum expiry = 15 minutes and is DIRECTIONAL.** Anything shorter is synthetic-index only for up/down.
A 30m magnitude horizon is *compatible* with the 15m floor in time, but Rise/Fall is the wrong payoff shape entirely.

**Products that pay on magnitude (verified from files — `DIRECTION_FINDINGS.md` L34, `METHODS_CATALOG.md` L87/127, `EXPERIMENT_LEDGER.md` L11/174, `m10_research_log.md` L75/141–142/191–192):**

| Product | Pays when | How the magnitude forecast is used |
|---|---|---|
| **Touch / No-Touch** | price touches (Touch) or avoids (No-Touch) a barrier before expiry | high predicted-|move| → buy Touch (or sell No-Touch); low → the reverse. This is the most direct fit. |
| **Range / Boundary (In/Out)** | price stays inside (In) or exits (Out) a band | high predicted-|move| → Out / "Goes Out"; low → In / "Stays In". |
| **Straddle / Strangle** (options-style) | realized move exceeds the combined premium in *either* direction | long straddle when predicted-|move| is in the top decile (5× lift), flat/short otherwise. |
| **Variance-risk-premium (VRP)** | realized variance ≠ implied variance | trade realized-vol forecast against the option-implied vol; the model's forecast is the realized leg. **Needs external data (below).** |

**News-time = a magnitude product** (`m5_research_log.md` L227, `DIRECTION_FINDINGS.md` L193): macro-event windows are
straddle/touch/volatility plays, not up/down — FX prices the surprise within ~1 minute and the surprise is a magnitude event.

**External data a VRP product needs — NOT on disk:**
- **Daily implied volatility / option-implied vol surface** is required for the VRP and straddle-pricing leg. **It is not on disk.**
  `min1_research_log.md` L98: "options risk-reversal = daily + no minute feed" — confirming **no intraday implied-vol feed exists**.
- Related external candidates the files list as not-on-disk (`METHODS_CATALOG.md` L129–131): options-implied risk-reversal/skew,
  intraday US-DE rate-differential futures, full depth-10 LOB volumes, CFTC COT positioning. (Listed there as direction candidates,
  but implied-vol/skew is the natural magnitude/VRP input.)

**Bottom line for a future builder:** Touch/No-Touch and Range/Boundary are buildable with on-disk data + the existing
magnitude model (re-trained per venue's barrier/expiry). Straddle and VRP additionally require a daily (ideally intraday)
implied-vol feed that does not currently exist in the repo.

---

## 6c. TESTED UPGRADE — Kronos per-path DISPERSION as a forward-vol feature → **KILLED (2026-06-06)**

The highest-EV lever from the Kronos web-research synthesis ("run first"): Kronos already samples K OHLCV paths per
window and discards the dispersion (`kronos.py:467` mean-collapses the `sample_count` paths before they reach us).
`kronos_disp.py` taps the per-path tensor BEFORE the collapse (copied `auto_regressive_inference` un-collapsed;
denormalized PER-SAMPLE in price space — the collapse happens in normalized space, which would corrupt the stats),
turns its dispersion into 6 FORWARD-looking vol features (terminal-return std / abs-mean / q90-q10 / IQR, predicted
path high-low range, within-path realized vol), and runs a paired ablation: baseline `[-pe, rv30, rv120]` vs
`baseline + dispersion`, on the SAME 28 CPCV paths, with cpcv_certify's exact target (`|ret_H| ≥ train-Q75`), LGBM
(`mk_lgb` 600), purge+embargo. Scope: a NONOVERLAPPING SUBSAMPLE of **15,041** decision bars (N=3000/yr, gap=30m,
2021-2026 1-min cache), K=24, pred_len=30 → dispersion derived for every sub-horizon from one generation. ~9h GPU.

**Pre-registered falsifier:** KILL horizon H unless base path-mean AUC ≥ 0.60 AND paired mean ΔAUC(+disp − base) >
+0.005 with bootstrap CI95 excluding 0 AND no path-p10 regression.

**Result — KILLED at every horizon** (`kronos_disp_disp_main_result.json`):

| H | base AUC (p10) | +disp AUC (p10) | paired ΔAUC mean [CI95] | verdict |
|---|---|---|---|---|
| 1m  | .7323 (.7138) | .7321 (.7128) | **−0.0003** [−.0019, +.0015] | KILLED (no effect) |
| 5m  | .7350 (.7170) | .7335 (.7133) | **−0.0015** [−.0026, −.0004] | KILLED (hurts) |
| 10m | .7268 (.7083) | .7253 (.7077) | **−0.0015** [−.0030, −.0000] | KILLED (hurts) |
| 15m | .7313 (.7103) | .7255 (.7039) | **−0.0058** [−.0076, −.0040] | KILLED (hurts) |
| 30m | .7187 (.7030) | .7117 (.6981) | **−0.0070** [−.0089, −.0052] | KILLED (hurts) |

The baseline reproduces the certified edge at subsample scale (sanity ✓). Adding Kronos dispersion adds nothing at
H=1 and significantly HURTS at H≥5 (CI strictly below 0). **Mechanism (not a power problem):** the GBM *does* split on
the dispersion features (gain: `range_mean` ~8–11k, `term_*` ~5–7k) but they correlate **0.46–0.83 with rv30** — they
are a noisier Monte-Carlo restatement of the realized volatility that `rv30`/`rv120` already measure directly and more
cleanly from the backward window. No magnitude information orthogonal to backward rv; at longer H the compounding
forecast noise anti-transfers (TRAIN-overfit → OOS-worse). Escalation (more K, full 2012-2026 span) would NOT change
this — the issue is collinearity, not estimation noise. **Generic lesson:** a single-pair generative path forecast does
not improve magnitude over cheap trailing realized-vol; the win, if any, must come from inputs rv can't see (macro
event windows, deseasonalized/semivariance RV, an external IV feed) — see §7. Files: `kronos_disp.py`,
`kronos_disp_disp_main.npz`, `kronos_disp_disp_main_result.json`.

---

## 6d. TESTED UPGRADE — Kronos `decode_s1` 512-d HIDDEN STATE as a frozen feature → **SMALL REAL MAGNITUDE LIFT (2026-06-06)**

Lever 2 of the Kronos synthesis. `decode_s1` returns the post-norm transformer hidden `[B, L, 512]` (kronos.py:305-308);
we take `x[:, -1, :]` — the learned representation AT the decision bar (it carries time-of-day via the additive time
embedding, kronos.py:298). `kronos_embed.py` extracts it with ONE forward pass per window (tokenize → decode_s1, no
autoregression, no K-sampling → ~190 win/s, 150× faster than Lever 1) at **18,077** nonoverlap bars (N=4000/yr,
2021-2026), and runs the SAME paired CPCV ablation as §6c for **both** targets per horizon.

**MAGNITUDE — first lever to ADD info beyond rv** (paired ΔAUC, base `[-pe,rv30,rv120]` vs base+512emb):

| H | base AUC (p10) | base+emb AUC (p10) | ΔAUC [CI95] | emb-only AUC | vs +0.005 bar |
|---|---|---|---|---|---|
| 1m  | .7327 (.7155) | .7394 (.7239) | **+0.0067** [+.0047,+.0088] | .6704 | **SURVIVES** |
| 5m  | .7269 (.7099) | .7291 (.7097) | +0.0022 [+.0001,+.0042] | .6545 | positive, sub-bar |
| 10m | .7314 (.7117) | .7390 (.7166) | **+0.0076** [+.0053,+.0099] | .6643 | **SURVIVES** |
| 15m | .7267 (.7107) | .7291 (.7104) | +0.0024 [+.0008,+.0042] | .6605 | positive, sub-bar |
| 30m | .7142 (.7023) | .7200 (.7093) | **+0.0058** [+.0039,+.0078] | .6519 | **SURVIVES** |

**Every horizon's ΔAUC CI95 excludes 0** (vs Lever 1 which was zero/negative). The 512-d embedding is the FIRST feature
to lift magnitude AUC over the certified backward-rv baseline — clears the pre-registered +0.005 bar at 1/10/30m,
positive-but-small at 5/15m. The lift is MODEST (rv .73 → .74); emb-only AUC .65–.67 is a real but weaker magnitude
predictor than rv alone. Causality is clean: the embedding is computed strictly on the context window `[i-L+1, i]`
(no forward bars), CPCV purge+embargo handles overlap, and the lift is an OOS gain across 28 paths (overfit would
show as ≤ base, not >).

**Mechanism caveat:** unlike rv30/rv120, the embedding encodes **time-of-day** (additive time-emb), so part of the lift
is plausibly intraday-vol SEASONALITY the embedding re-derives. Files: `kronos_embed.py`, `kronos_embed_embed_main.npz`,
`kronos_embed_embed_main_result.json`.

**RESOLUTION — the time-of-day decomposition (`kronos_embed_tod.py`, 2026-06-07).** 4-way paired CPCV ablation reusing
the same npz: base / base+tod / base+emb / base+tod+emb, where tod = `[hour, minute_of_day, dow, sin(t), cos(t)]` (UTC).
Key metric ΔAUC(emb BEYOND tod) = base+tod+emb − base+tod. **Horizon-dependent answer:**

| H | ΔAUC(+tod−base) | ΔAUC(emb BEYOND tod) [CI95] | resolution |
|---|---|---|---|
| 1m  | +0.0011 (ns)    | **+0.0057** [+.0038,+.0078] | emb adds REAL non-clock info |
| 5m  | −0.0000 (none)  | **+0.0032** [+.0005,+.0062] | emb adds REAL non-clock info |
| 10m | +0.0048         | **+0.0028** [+.0010,+.0049] | emb adds a bit beyond clock |
| 15m | +0.0025         | +0.0012 [−.0012,+.0036]     | lift IS time-of-day (CI incl 0) |
| 30m | +0.0087         | −0.0023 [−.0049,+.0006]     | **clock ALONE beats the embedding** |

Pooled-CPCV conclusions (SUPERSEDED — see the ⚠ forward reversal below): at 30m plain hour-of-day appeared to beat the
Kronos embedding (+.0087 vs +.0058); at 1–10m the embedding appeared to add genuine non-clock info (+.003–.006 beyond
clock). **ALL of these were measured on POOLED combinatorial CPCV ONLY and DO NOT HOLD FORWARD — see §6f.** Files:
`kronos_embed_tod.py`, `kronos_embed_tod_embed_main_result.json`.

**DIRECTION (emb-only, ties dropped) → NULL all horizons:** CPCV AUC .502/.506/.5085/.5086/.5096, path-p10 .49–.50,
all KILLED (bar .52). The single-pair learned representation carries no sign — consistent with every prior single-pair
direction null (Kronos zero-shot/FT, dispersion, bar-CNN, GBM, ARF, RFF, GRU/ESN). The sign edge stays cross-sectional;
recorded in DIRECTION_FINDINGS.md.

---

## 6e. TESTED UPGRADE — Chronos-2 forecast QUANTILE SPREAD as a forward-vol feature → small positive, **SUB-BAR (2026-06-07)**

Side-result of Lever 3 (`chronos2_xpair.py`, the 7-pair cross-sectional DIRECTION test — which KILLED on direction, see
DIRECTION_FINDINGS.md). The EURUSD forecast q90-q10 spread from Chronos-2 (conditioned on the 7-pair panel via
group-attention) was paired-ablated as a forward-vol feature vs the rv baseline on **44,999** bars (2012-2026): ΔAUC
+0.0026/+0.0039/+0.0025/+0.0018/−0.0012 at H=1/5/10/15/30m — positive and CI95-excludes-0 at H≤15m but **all below the
+0.005 bar** → KILLED by threshold (same small-positive-sub-bar pattern as Lever 2's embedding §6d). Base AUC .74–.75
(higher than §6c/d because this run spans the full 2012-2026). **Consistent verdict across all three Kronos/TSFM
magnitude levers (§6c dispersion KILLED, §6d embedding small-win, §6e Chronos-2 spread sub-bar): a forward-looking model
spread adds at most a sliver to backward rv, never enough to matter.** The real magnitude upgrade remains the cheap §7
items (semivariance, macro-event windows), not a TSFM. `chronos2_xpair_c2_main_result.json`.

---

## 6f. ⚠ TIME-OF-DAY / DESEASONALIZED-RV — pooled-CPCV WIN that FAILS FORWARD → **NOT DEPLOYABLE (2026-06-07)**

`deseason_mag.py` built the principled deseasonalized-RV upgrade on the FULL certified 2012-2026 frame (5.33M bars,
byte-faithful to `cpcv_certify.py`: same target |ret30|≥train-Q75, [-pe,rv30,rv120], 28-path CPCV, deflation). On
POOLED CPCV every clock arm beat base, the simplest most: **+tod** (raw `[hour,minute,dow,sin,cos]`) gave paired ΔAUC
**+0.0140** [CI +.0119,+.016], decile lift 5.02→5.77x, deflated 0.7128→0.7274, on all 28 paths. Base reproduced the
certified 0.744 exactly. Adversarial review (workflow wwtci9slp) confirmed it is **leakage-FREE** (tod is a pure
function of the decision-bar timestamp; placebo with shuffled timestamps gives ΔAUC +0.0004 vs real +0.0181) and the
intraday vol seasonality is **real** (cross-year hour-profile corr ~0.94; peak 7–10 UTC London/NY, trough 14–17 UTC).

**BUT IT DOES NOT TRANSFER FORWARD** (`deseason_fwd.py`, train≤2023 → test per-year, deployment-faithful):

| test year | base AUC (lift) | +tod AUC (lift) | ΔAUC (ΔLift) |
|---|---|---|---|
| 2024 | 0.7908 (5.14) | 0.8108 (6.35) | **+0.0200** (+1.20) |
| 2025 | 0.7388 (4.61) | 0.7299 (4.25) | **−0.0089** (−0.35) |
| 2026 | 0.7374 (3.80) | 0.6830 (2.95) | **−0.0544** (−0.84) |

The clock lift is **non-stationary and decays to strongly NEGATIVE** (2024 +.02 → 2025 −.009 → 2026 −.054): the model
overfits a historical intraday-vol shape that drifts (the 2025/26 profile flattened), so by 2026 the +tod model is
materially WORSE (.683 vs .737). **Verdict: KILLED for deployment.** The base rv-magnitude edge IS forward-robust
(.74–.79 every year); only the calendar addition fails. This RETRACTS the §6d "promote time-of-day" conclusion AND the
prior `kronos_embed_tod` pooled-CPCV result — both were pooled-CPCV-only and never forward-tested. (The embedding's
claimed beyond-clock 1–10m lift, §6d, was also pooled-CPCV-only → treat as UNPROVEN until forward-tested.)

> **⭐ GENERIC METHODOLOGY LESSON (the durable output — also added to METHODS_CATALOG leakage traps + the strategy-eval
> skill): pooled combinatorial CPCV does NOT catch NON-STATIONARY-FEATURE memorization.** 20/28 CPCV test folds are
> temporally FLANKED by train folds on both sides, so a model can memorize era-LOCAL structure (intraday-vol seasonality,
> calendar effects, slow regime features) from neighboring years and score high on held-out folds WITHOUT forward
> transfer. CPCV purge+embargo kills label-OVERLAP leakage; deflation penalizes multiple-testing on the LEVEL; NEITHER
> detects forward non-transfer. **Any CPCV-certified edge that leans on calendar/seasonal/slow-moving features MUST be
> confirmed with a frozen-past forward holdout (train≤Y, test Y+1,Y+2 per-year) before any deployment claim.** This is
> why the production books (m30_production etc.) use a frozen-past split — pooled CPCV alone is necessary, not sufficient.
> Files: `deseason_mag.py`, `deseason_mag_30m_result.json`, `deseason_fwd.py`, `deseason_fwd_30m_result.json`.

---

## 6g. TESTED UPGRADE — HAR / realized-measure ECONOMETRIC-VOL CANON → **REAL but SUB-BAR forward (2026-06-07)**

`mag_har.py` tested the entire econometric realized-vol canon the novel-methods research flagged as the slate's biggest
omission, as additive arms on the certified base `[-pe, rv30, rv120]`, gated by the **frozen-past FORWARD HOLDOUT**
(train≤2023 → per-year 2024/25/26, horizons 10/15/30m, |ret_H|≥train-Q75). Realized measures over rolling W∈{30,120,480}
on 1-min returns: multiscale **RV** (log sum-of-squares), **bipower BV / jump** (RV−BV continuous/jump split), **realized
semivariance RS⁺/RS⁻** + signed jump (Patton-Sheppard good/bad vol), **realized quarticity** RQ → HARQ attenuation `lRV·√RQ`.

**Falsifier = beat base by +0.005 AUC in ≥2 forward years AND ≥2 horizons. ALL ARMS FAIL:**

| arm (added to base) | mean fwd ΔAUC | deployable horizons | verdict |
|---|---|---|---|
| +har (multiscale RV) | +0.0010 | 2/3 | sub-bar |
| +jump (bipower split) | +0.0003 | 1/3 | null |
| +semivar (RS⁺/RS⁻/signed-jump) | −0.0000 | 1/3 | null |
| +harq (realized quarticity) | +0.0004 | 2/3 | null |
| **+all (stacked)** | **+0.0021** | **3/3** | **sub-bar** |

**Two findings.** (1) **Unlike §6f, the additions are forward-CONSISTENT, not non-stationary:** `+all` is POSITIVE in all
9 year×horizon cells (10m +.0021/+.0022/+.0033, 15m +.0016/+.0015/+.0022, 30m +.0007/+.0022/+.0029), deployable 3/3 — so
the HAR canon adds GENUINE forward signal, it is just economically negligible (~+0.002 AUC, well under the +0.005 bar).
The base `[-pe, rv30, rv120]` already extracts ~all forecastable magnitude information; the bipower jump-split, semivariance
asymmetry, and quarticity attenuation are near-redundant with 2-window rv (collinearity vs rv120: lRV120 0.89, RSp/RSm 0.70,
HARQ terms 0.58/−0.63; only signed-jump SJ120 is orthogonal at −0.04, and it carries nothing). **Verdict: KILLED as a
deployable upgrade; recorded as REAL-but-sub-bar (the USDJPY-2m pattern).** (2) **The run independently RE-VALIDATES the
certified magnitude edge on a clean deployment-faithful holdout:** base AUC 0.799/0.750/0.750 (2024/25/26) at 10m,
0.791/0.739/0.737 at 30m, decile lift 4–5.6×, **no decay** — stronger evidence than pooled CPCV that the magnitude edge
is forward-robust. Files: `mag_har.py`, `mag_har_result.json`.

---

## 7. UNTESTED UPGRADES — magnitude-model improvement backlog

These are NOT yet built or tested. They follow directly from the finding that **realized vol carries the signal and PE is inert on FX**:

1. **Deseasonalized realized variance.** ❌ **TESTED & FORWARD-KILLED 2026-06-07 — DO NOT DEPLOY (see §6f).** Raw time-of-day
   features lifted pooled-CPCV AUC (+0.014 at 30m) but the lift is NON-STATIONARY and decays to strongly negative on a
   frozen-past forward holdout (2024 +.02 → 2026 −.054); the FX intraday-vol seasonal shape drifts, so adding the clock
   overfits a stale pattern. The base rv30/rv120 magnitude edge is forward-robust on its own; do NOT add raw clock or a
   fixed deseasonalization. ONLY revisit with an ADAPTIVE/rolling seasonal profile (re-estimated on a trailing window) AND
   a per-year forward holdout as the gate — a fixed profile is forward-fragile. Lesson recorded in §6f.
2. **Signed realized-semivariance (RS+ / RS−).** ❌ **TESTED & SUB-BAR 2026-06-07 (see §6g).** RS⁺/RS⁻ + signed-jump +
   semivariance-asymmetry added to the certified base gave forward ΔAUC −0.0000 (mag) — null, near-redundant with rv120
   (corr 0.70). As part of `+all` it contributes to a forward-consistent but sub-bar +0.0021. Magnitude path is exhausted;
   the *direction* arm of signed-semivariance (Patton-Sheppard good/bad vol → DOWN side) is tested separately in the
   direction campaign (D7), not here.
3. **Macro-event-window feature.** Add a binary/decay feature for proximity to scheduled macro releases (the files already note
   news = magnitude event, `m5_news.py`). A "minutes-to-next-release" + "minutes-since-last-surprise" pair should lift predicted
   |move| precisely in the windows where straddle/touch payoffs are largest. Macro calendar source is available (MEMORY:
   Investing.com / TradingView XHR, actual+forecast, minute timestamps).
4. **Re-validate PE under the theorem properly.** PE was inert at d=4/W=120 on 1-min FX. Before discarding the entropy mechanism,
   test other embeddings (d=3,5,7; tau>1; bipower/jump-robust complexity) and intraday-bar vs tick inputs — the theorem is about
   tick order-flow, and the current PE is on 1-min bar returns.
5. **Promote the LOG-RECORDED horizons to certified.** Wrap m30/m10/60s/120s magnitude scripts to write `*_result.json`
   and run each through the same `cpcv_certify.py` deflation machinery, so every horizon has a deflated, on-disk number
   like the 30m one.

---

## 8. Cross-references (every file cited above, with what it backs)

**Tier-1 (verified this session):**
- `cpcv_certify_result.json` — `magnitude_30m` block (AUC mean 0.7439, p10 0.7177, min 0.7054, max 0.7819, lift_mean 5.013,
  lift_p10 3.96, all 28 `all_aucs`) and `deflation.magnitude_auc` block (headline 0.79, bar 0.55, deflated_expectation 0.7124,
  emax_inflation 0.0583, neg_corr_penalty 0.0315, p10_clears_bar/deflated_exp_clears_bar true, prob_typical_path_clears_bar 1.0).
  Also `direction_15m`, `deflation.direction_selective/direction_auc/stack5m_pbo` for the contrast.
- `cpcv_certify.py` — `load_magnitude` (features `[-pe, rv30, rv120]`, line 120), `mk_lgb` (lines 168–172),
  `run_magnitude` (`n_estimators=600`, line 247), `deflated_metric` (line 270), CPCV params (lines 129–164).
- `m30_magnitude.py` — docstring (arXiv:2512.15720, sign-invariance statement, lines 1–6), windows (line 13), feature/target code.
- `min1_production.py` — magnitude train block (lines 188–192: P67 thr, `mk_lgb(2500)`).
- `models/min1_EURUSD_strategy.json` — `mag_top_tercile_thr=1.2034e-4`, `val_auc_inregime=0.5052` (DIRECTION).
- `models/min2_EURUSD_strategy.json` — `mag_top_tercile_thr=1.7058e-4`, `val_auc_inregime=0.5111` (DIRECTION).
- `models/min1_EURUSD_magnitude.joblib` (8.7 MB), `models/min2_EURUSD_magnitude.joblib` (8.0 MB) — exist, unscored.
- `harness.py` — `FEAT_DIR`, `feature_cols('EURUSD')` (239 names), `META_COLS`.

**Tier-3 (LOG-RECORDED — script prints to stdout, no result file; re-run to promote):**
- `IDEAS_LOG.md` L82–86 — m30 per-window AUC 0.75/0.75/0.78/0.73/0.73, corr +0.42..+0.38, PE NULL on FX.
- `m10_research_log.md` L186–191 — 10m magAUC 0.813/0.741/0.706, direction flat 0.51–0.53.
- `min1_research_log.md` L84 (60s magAUC 0.787 vs dirAUC 0.510), L166–167 (30m magnitude survives, deflated 0.712, 5.0× lift),
  L98 (no minute implied-vol feed), L159–174 (direction deflation), L172–174 (CPCV is a faithful re-impl, not byte-identical m15_production).
- `m5_research_log.md` L225–227 — no 5m magnitude classifier; news = magnitude/volatility event.
- `EXPERIMENT_LEDGER.md` — row 20/127 (60s), row 38/41–42 (120s + gate-hurts), row 86 (10m), row 114 (complexity flat), row 115 (30m).
- `METHODS_CATALOG.md` L50 (path-sig = direction@5s), L87/127 (magnitude needs touch/range/straddle venue), L110–111 (theorem), L129–131 (external data not on disk).
- `DIRECTION_FINDINGS.md` L13/148/163–165 (seconds direction edge), L31 (theorem), L34 (magnitude venue), L70 (15m forex floor), L193 (news), L195 (gitignore).

**Theory:** arXiv:2512.15720 (Dec 2025) — sign-invariance theorem. (Tier-4 external paper; its *claim as applied here* is
quoted from the Tier-1 `m30_magnitude.py` docstring, and empirically validated by the EURUSD experiments above.)

---

### One-paragraph handoff for the next session
There is exactly **one** edge in this program that survives rigorous deflation: **30-minute move-SIZE.** Read it from
`cpcv_certify_result.json` (`magnitude_30m` + `deflation.magnitude_auc`): 28 purged-combinatorial paths, AUC mean **0.744**,
deflated **0.712** vs a 0.55 bar, every path clears, **5.0× top-vs-bottom-decile |ret| lift.** It is driven by **realized
volatility (rv30/rv120)**, not entropy (PE is inert on FX despite the theorem framing). Direction at ≥5m is ~EMH and the
15m direction book deflates and FAILS in the *same* file. To productize you need a **magnitude venue** (Touch/No-Touch,
Range/Boundary, straddle/strangle, or VRP) — NOT Deriv Rise/Fall, which is directional with a 15m floor — and for VRP you
need a **daily implied-vol feed that is not on disk.** Everything except the 30m CPCV number is **LOG-RECORDED only**:
re-run the scripts with stdout captured to a result file before quoting them as fact.
