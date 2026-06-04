# EURUSD · 30m — EXECUTABLE BACKLOG (key-specific)
SCOPE: EURUSD · 30m. Key-specific executable backlog — incumbents, scripts, numbers, discovery rounds.
Moved out of IDEAS_LOG.md on 2026-06-02 during the generic↔specific split (see REPO_MAP.md).
Generic methods/ideas: METHODS_CATALOG.md / SWEEP_MATRIX.md / IDEAS_LOG.md. Sweep ledger (status): sweeps/EURUSD_30m.md. Results of record: EURUSD_RESULTS.md.

---

## EXECUTION ORDER (live)
- [RUNNING] Workflow: Sofien gate mining (larger-TF reversal / flag gates).
- [RUNNING] Workflow: non-time bars / volume-price / VPIN / meta-labeling research.
- [TODO] Workflow: orthogonal-math academic research (A-F above), with reported numbers + implementability.
- [BUILDING] vbars.py: volume & dollar bars → 30m label, honest selective.
- [TODO] m30_complexity.py: permutation entropy + Hurst/DFA + RQA-DET + Lempel-Ziv gates on the 30m model; "avoid losers".
- [TODO] signatures as features; Hawkes intensity on event bars; transfer entropy volume→price.

## RESULTS (append as they come — be honest, cross-window-stable only)
(none yet this thread)

## RESULTS LOG (2026-05-31, honest, cross-window)
- **Larger-TF reversal/flag gates** (1h/4h/daily RSI/DeMarker/WillR/CCI/BB/z/Fisher extremes; confluence; chop-gated) — m30_gates.py: ALL null, none >=0.60 stable across held-out. Oscillator extremes at larger TFs do NOT predict 30m reversals.
- **Complexity/predictability gates** (Permutation Entropy d3 W30/60/120; lag-1 autocorr ac20; variance-ratio Hurst) — m30_complexity.py: NULL. Conditional accuracy is FLAT across all bins of all measures: momentum ~0.48, reversion ~0.515. Complexity does NOT isolate a predictable subset. Only signal = uniform weak ~0.515 mean-reversion (below 0.541 breakeven). The "low-entropy = tradeable" hypothesis FAILS for 30m EURUSD.
- IMPLICATION: 30m direction is ~0.515 mean-reverting, uniform, no regime/complexity gate isolates >0.58. Next untested: VOLUME/DOLLAR BARS (non-time grouping), VPIN, Hawkes intensity on event bars, path signatures, transfer entropy vol->price, Sofien's 79 mined rules (pivots/overextension-release/confluence).

## POSITIVE FINDING (validated OOS) — the real "different story": MAGNITUDE is forecastable, DIRECTION is not
m30_magnitude.py: target = 30m forward |return|.
- realized vol (rv30): corr with future|ret| = +0.42/+0.44/+0.44/+0.40/+0.38 (train/val/t24/t25/OOS). Large-move (|ret|>=Q75) classification **AUC 0.75/0.75/0.78/0.73/0.73 — STRONG, STABLE, OOS-verified**.
- permutation entropy: corr ~0.01, AUC 0.50, flat |ret| across quintiles — PE does NOT predict magnitude on EURUSD (the SPY-trades result doesn't replicate on 1m FX returns; realized vol is the real magnitude predictor).
=> CONFIRMED: 30m EURUSD MAGNITUDE/volatility is ~0.73-0.78 AUC forecastable; DIRECTION/sign is ~0.52 (EMH). The predictable structure is magnitude, exactly as the sign-invariance theorem predicts.
IMPLICATION for up/down binary: magnitude predictability lets you AVOID small-move/tie losers and size bets, but does NOT raise directional accuracy (sign within large moves is still ~0.515). Tradeable as a volatility/touch/straddle product, not up/down.

## THREAD SUMMARY (this session's orthogonal/creative push, all honest cross-window)
Tested & NULL for 30m DIRECTION: larger-TF reversal gates; complexity gates (PE/WPE/Hurst/ac20); Sofien 79 rules
(pivots/IBS/Connors-RSI2/TD-setup/BB-reentry/confluence); volume & dollar bars; (theory) transfer-entropy/Hawkes are
sub-minute/hourly/trader-resolved or sign-invariant. POSITIVE & VERIFIED: magnitude forecastable (rv, AUC ~0.75).
Untested-but-low-prior (sign-aware): path signatures (CNN-on-raw-path already null 0.525), 4D Bacry-Muzy Hawkes intensity
(needs LOB events, sub-minute). Direction at 30m remains ~0.52-0.515 mean-reverting; no method this thread broke it.

## PATH SIGNATURES (Lévy area, rough-path theory) — last sign-aware orthogonal lever (m30_sig.py)
Lévy area A(price, order-flow imbalance) = signed area = lead-lag ROTATION, computed causally over W={30,60,120,300}s.
- HS=1800 (30m GOAL): AUC val 0.515 / TEST 0.508 / OOS 0.509 — NULL. Model barely fits (1 tree). Signatures don't help 30m direction.
- HS=5s: signature Lévy areas rank TOP-5 importances; selective ~0.63-0.66 (= base tick edge). Order-flow→price path rotation IS directional AT SECONDS, decays to noise by 30m.
=> FINAL: every sign-aware orthogonal method (signatures, and by extension Hawkes/transfer-entropy which capture the same lead-lag) confirms: directional edge lives at the SECONDS scale; 30m direction is EMH (~0.515).

## DEFINITIVE THREAD CONCLUSION (orthogonal/creative push, 2026-05-31)
30m EURUSD DIRECTION is not predictable beyond ~0.515 by ANY method tried (now ~25 experiments + 4 research workflows +
a sign-invariance theorem). The predictable structure is MAGNITUDE (realized vol -> large-move AUC ~0.75 OOS). The only
directional edge is at the SECONDS scale (~0.65, confirmed by signatures). For an up/down 30m binary, >0.65/>0.75 is not
achievable on EURUSD; the honest tradeable edges are: 15m compression×NY ~0.64, 3s tick ~0.66, and the NEW magnitude/vol
model ~0.75 (volatility/touch products, not up/down).

---

## 2026-06-04 — NEW TWO-SIDED CERTIFICATION PUSH (/goal): the cross-pair lever was NEVER run at 30m
**The above "DEFINITIVE CONCLUSION" predates the cross-pair breakthrough.** It chased >75% (impossible) and its
"cross-pair" test (Iteration-2) bolted 60 peer FEATURES onto a single EURUSD LGB (+0). The DIFFERENT mechanism —
POOLED 7-major training + USD-common-factor residual + order-flow (xpof, m5_xpair) — certified BOTH sides at 10m
(.586/.568) and 15m (.5673/.5742). Goal now = certify best UP + best DOWN vs breakeven 0.541 (not 0.75). 30m is the
LONGEST deriv-tradeable horizon. Live: K1 m30_cpcv_side.py (base floor), K2 m30_xpair_cpcv.py (keystone cert).

### DISCOVERY ROUND 1 (workflow 2026-06-04; 4 agents over corpus+arXiv; 6 levers, vetted by sign-invariance)
Most cross-pair decomposition ideas (Giglio-Xiu 3-pass, PCA-residual, currency-strength, ECM) are SUBSUMED by xpof = K2.
Actionable NEW improve-levers (run ONLY after K2 certifies, build ON the frozen K2 book; each must beat K2 binding-side p10):
- **I-Aa — Antisymmetric cross-pair matrix component** [med/new/on-disk]. Fit lagged 7x7 cross-pair AR/prediction matrix
  A on TRAIN; split A = As (symmetric, own-pair-replicable) + Aa (antisymmetric = signed lead-lag rotation). Add Aa(X_t)
  as ONE extra primary feature on the K2 book; re-run side-split refit-CPCV. Sign-carrying via lead-lag rotation (Lévy-area
  family, but aggregated not seconds-only). Orthogonal-by-construction to K2's symmetric common factor. KILL if rank(Aa)<2
  after SVD, OR Aa OOS directional payoff <=0 on worst-VAL-half, OR fails to raise K2 binding-side p10 by >=0.005.
- **I-IPCA — Time-varying USD-loadings (instrumented betas)** [med/new/on-disk]. Replace K2's static USD-beta with IPCA
  (Kelly-Pruitt-Su): beta_{i,t}=z_{i,t}'Gamma linear in the 239 feats, single global Gamma by ALS on TRAIN; EUR 30m dir =
  beta'_{EUR,t} lambda_t. The ONLY lever that can FLIP the EUR-USD loading sign across regimes -> directly attacks the
  2025 USD-factor-inversion wall (documented cause of static-residual improve-lever KILLs at 10m). KILL if ALS <2 factors
  eigenvalue>1, OR OOS dir R2<0.5%, OR IPCA EUR-sign acc on 2025 worst-VAL-half does NOT beat the static K2 book.
- **I-GX — Giglio-Xiu weak-factor Wald/R2_g feature pre-screen** [med/variant/on-disk]. Sign-NEUTRAL hardening: keep only
  K2 xpof cols rejecting H0:h=0 (p<0.05) vs the 7-major PCA factors; refit K2 on survivors to shrink overfit surface
  (attacks corr(VAL,OOS)=-0.54). KILL if Spearman(R2_g rank, per-feat OOS dirAUC) < 0.1 on train+val (abort, no signal),
  OR pruning lowers K2 binding-side refit p10.
- **(gated, low) External DE-US 2y rate-diff + 25-delta risk-reversal** [low/new/OFF-disk]. Only non-redundant info channel
  left, but data not acquired + likely subsumed by realized-vol SIZE gate. Defer to external-data phase.
Generic versions of I-Aa / I-IPCA / I-GX -> IDEAS_LOG.md + SWEEP_MATRIX Tier-N during records phase.

### MECHANISM UPDATE (2026-06-04, Tier-1 feature-importance of frozen EURUSD.m30xp.v1)
The 30m cross-pair edge is carried by **POOLED TRAINING on the BASE 239 multi-TF features (94% gain)** — top: hour_sin/cos
seasonality, 1m/4h trend, 4h vol/autocorr. The cross-pair-SPECIFIC block carries almost nothing: lead-lag ll_ **2.1%**,
cross-pair factor (eu_r/usdbask/catchup/eurresid/disp/agree) **2.0%**, order-flow OF **1.8%**. => the lever is POOLING
(more data + cross-pair regularization of base feats, esp. rescuing DOWN), NOT the xpof factor features. SUBSUMES I-Aa
(linear combo of 2.1%-gain ll_ already in input) and I-IPCA (refines 2.0%-gain factor channel; its 2025-inversion
rationale also doesn't bind at 30m where 2025 is the STRONG fwd year). `m30_xpair_featimp_result.json`.

### IMPROVE-LEVER RESULTS (vs incumbent cross-pair p10 UP .5588 / DOWN .5525)
- **I-magw** (\|ret\|-weight POW=0.5, `m30_magweight_cpcv.py`): UP p10 .5528 / DOWN .5382 — **SUBSUMED/HURTS**. Magnitude-
  weighting pulls toward the sign-invariant SIZE signal, diluting directional sign (textbook sign-invariance).
- **I-Aa, I-IPCA**: **SUBSUMED (Tier-1 FI above)** — not run; the channels they refine carry ~2% gain.
- **I-seed** (seed-ensemble K=4, `m30_seedens_cpcv.py`): running (variance-reduction; tail-variance not expected binding).

### DISCOVERY ROUND 2 (workflow 2026-06-04; retargeted on the pooling finding) — effectively DRY (no surviving on-disk lever)
- **Month-end / quarter-end rebalancing-flow** [on-disk, low prior]: **KILLED** (`m30_monthend.py`/`_result.json`). No stable
  directional bias — ME up-rate 2024 .494/2025 .504/2026 .498 (sign FLIPS, never exits [.47,.53]); QE .500/.527/.492 same.
  Hour-of-day seasonality (already the book's #1/#3 feature) subsumes any calendar edge; up-rate tripwire constrains it.
- **Intraday DE-US 2y rate differential (CIP carrier)** [EXTERNAL, low prior, BLOCKED]: the one genuinely non-redundant
  sign-carrier (every on-disk channel is price/flow-derived). = standing backlog H-TODO-1, gated on data acquisition
  (Dukascopy intraday yields; FRED daily too coarse). The only frontier that could further move 30m direction — same
  conclusion as 60s/2m/5m. Falsifier if acquired: univariate dirAUC>.52 worst-VAL-half + lifts m30xp binding p10 >=.005.
=> R2 yields ZERO surviving on-disk levers (month-end KILLED; rate-diff external-gated). DRY round #1. R3 to confirm K=2.
