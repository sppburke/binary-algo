# THEORY — cross-key facts (GENERIC)

SCOPE: GENERIC (currency/timeframe-agnostic). The load-bearing theory that governs every key. Per-key
*measurements* that illustrate these facts are tagged `[PAIR·tf]` and their record-of-truth is the Tier-2
file (`<PAIR>_RESULTS.md`, `MAGNITUDE_FINDINGS.md`). See `REPO_MAP.md`.

---

## 1. Sign-invariance theorem (why most "direction" gauges are really MAGNITUDE)

`arXiv:2512.15720` (Dec 2025, SPY 38.5M trades): order-flow / permutation **ENTROPY is invariant under sign
permutation**, so it detects the *presence* of informed trading = **MAGNITUDE / volatility**, NOT **sign**.
Empirically there: at entropy<5th pct, `|5-min return|` ×2.89 (t=12.41) but directional accuracy ≈45% (chance).

**Generalization (the working law of this program):** entropy, order-flow imbalance, complexity (PE/RQA/LZ),
Hurst/DFA, Kalman/HMM **filter statistics**, signature **norms** — all gate move SIZE, not move SIGN.

**Operational test (apply to EVERY candidate before claiming "direction"):** the within-`|ret|`-bin **sign-shuffle
placebo**. Shuffle the sign of the label within magnitude bins; if the "edge" survives, it is a MAGNITUDE edge →
evaluate it as magnitude (`|ret|≥Q`, AUC) and record in `MAGNITUDE_FINDINGS.md`, NOT as direction. A genuine
*direction* mechanism must be **sign-AWARE** (e.g. signed Hawkes up/down cross-excitation asymmetry, signed
lead-lag / path signed-area, tail-conditional sign asymmetry) and must name which side's sign it carries.

**Cleanest single-method demonstration `[EURUSD·60s]` (2026-06-05):** the SAME 2-D bar-image CNN (Sezer CNN-BI /
GAF — a 4th model class beyond GBM/GRU/state-space) is **null on direction** (dirAUC ≈ .50, CPCV 0/28 paths clear
0.541, even the antisymmetric GADF sign-field) yet **clears >65% on magnitude** (large-vs-small move; magAUC
.699/.714/.686, selective precision .68→.80, all 28 CPCV paths ≥0.65 every held-out year). One representation, one
training pipeline, opposite verdicts by target — exactly the theorem. KEY detail: a bar image must keep its
**absolute volatility scale** to carry magnitude (per-window min-max normalization strips it → magAUC .64). See
`MAGNITUDE_FINDINGS.md` §3, `METHODS_CATALOG.md` §5.5, `barcnn_*.py`.

## 2. The direction ceiling (horizon-dependent; magnitude is the durable edge)

Working shape (currency-agnostic hypothesis; numbers below are `[EURUSD]`-measured evidence, see `EURUSD_RESULTS.md`):
- Sub-minute direction is **near-efficient** — AUC ≈0.50–0.51 across ~24 input channels (now incl. a 2-D bar-image
  CNN, the 4th model class — dirAUC ≈.50, CPCV 0/28 clear); `>0.65` OOS-stable is **not** achievable at ≤5m on
  clock-bar data. The best *available* 60s direction is the regime-gated UP dip-buy filter (uncertified: pooled .573,
  CPCV p10 .524, 2024 .520<breakeven); DOWN is dead. The only 60s edge that survives CPCV is **magnitude** (the
  Touch/Range straddle, ~.72–.80 selective precision). `[EURUSD·60s]`
- Short tick horizons (1–5s) carry a genuine `>0.65` directional edge but need a **tick venue** (deriv forex
  min duration is 15m — see §3). `[EURUSD·1-5s]`
- Intermediate horizons (≈10–15m) carry a modest **regime-dependent** direction edge (~0.58 cross-era / ~0.65
  recent). `[EURUSD·15m]` `[EURUSD·10m]`
- **CROSS-PAIR USD-COMMON-FACTOR is THE direction lever, and it is HORIZON-GATED — a gradient: none@60s → UP-only@5m
  → BOTH sides@10m, 15m & 30m; null again <5m.** Mechanism: the informed/jump component of a move (especially DOWN) is
  noise at short horizons but **averages out as the horizon lengthens**, so the slow USD-common-factor SIGN becomes
  forecastable at ≥10m. The carrier is the **CONCURRENT** cross-pair read (windows ending at t); strictly-LAGGED
  lead-lag is dominated. Per-fold-refit CPCV certified: 5m-UP (p10 .553), BOTH 10m sides (.586/.568), BOTH 15m sides
  (.567/.574), BOTH 30m sides (.5588/.5525 — the LONGEST deriv horizon, but REFIT-DEPENDENT: the frozen-2021 book's
  forward edge DECAYS by the 2026 OOS year, esp. UP → deploy with periodic retraining); KILLED at 2m (p10 ~.51).
  For a new (currency, ≥10m) key the cross-pair refit-CPCV side-split is the
  #1-prior lever. `[EURUSD·5m/10m/15m/2m]` (see `EURUSD_RESULTS.md`, `sweeps/EURUSD_{10,15}m.md`, `METHODS_CATALOG.md` A6)
- Once the cross-pair book certifies a ≥10m key, **loss/label/gate re-engineering does NOT beat the gated raw
  cross-pair sign** — magweight, GMADL/sign-coupled loss, residual-relabel, ACI gate, specialist, calibration,
  cross-horizon blend, lagged lead-lag, intraday-momentum all collapse on the binding-regime wall (a wrapper cannot
  create SIGN the regime erased). The redirect for a HIGHER number is **external data**, not another loss/gate variant.
  `[EURUSD·10m]` (11 levers dry, 2 rounds), `[EURUSD·30m]` (magweight HURTS — dilutes sign→magnitude; seed-ens/
  specialist/Aₐ/IPCA subsumed; 2 dry discovery rounds) — same outcome as 15m.
- **The cross-pair edge is carried by POOLED TRAINING on the BASE multi-TF features, NOT the cross-pair-specific
  features** (measured: frozen 30m cross-pair book feature-importance = base multi-TF **94%** gain vs lead-lag 2.1% /
  cross-pair-factor 2.0% / order-flow 1.8%). Pooling 6–7 USD-major rows gives more data + cross-sectional
  regularization of the SAME base features (esp. rescuing DOWN); the xpof factor/lead-lag block is near-zero gain.
  ⇒ factor-refinement levers (IPCA instrumented betas, antisymmetric lead-lag matrix Aₐ) are **subsumed** — they
  refine a ~2%-gain channel. CORRECTS the earlier speculation that factor/IPCA levers grow more relevant at longer H:
  what grows is the *pooling* benefit, not the factor features. `[EURUSD·30m]` (`m30_xpair_featimp_result.json`)
- **MAGNITUDE is the one CPCV-deflation-certified edge** at every horizon tested (large-move AUC ≈0.71–0.81).
  `[EURUSD·30m/60s]`
- Where a direction edge exists it is **regime-/horizon-specific** and may be **one-sided** (dip-buy UP at 60s/5m)
  OR **two-sided** (BOTH sides via cross-pair at 10m & 15m, DOWN as robust as UP) — never assume a side or copy one
  key's asymmetry to another; measure each side directly.

## 3. Deriv.com Rise/Fall settlement & breakeven (platform mechanics — generic)

- Settlement is **wall-clock mid-to-mid**: entry = NEXT tick after the order (+1s lag), exit = last tick ≤ expiry.
  No spread charged on settlement; **ties LOSE** (a flat outcome is a loss).
- **Breakeven win-rate ≈ 0.541** at payout R≈1.85 (R≈0.85 profit). Every certified edge must clear this on the
  binding (worst) held-out year's CI95-lower.
- **Forex minimum contract duration = 15m** on deriv; sub-15m edges need a different venue to be tradeable.
- Faithful evaluation MUST use `wc_ret()` (never `mid.shift(-N)` on gap-dropped bars — that shifts *bars* not
  *seconds*, the biggest historical inflation). See `METHODS_CATALOG.md` (validation family) for the full discipline.

## 4. corr(VAL, OOS) = −0.54 (selection anti-transfers)

Across the configs searched, validation accuracy is **negatively** correlated with OOS accuracy: VAL-maximal
pockets ANTI-transfer. Therefore SELECT thresholds/gates on the **worst-VAL-half**, never VAL-acc-max. This bites
purpose-built specialists hardest. (Generic; the −0.54 figure is `[EURUSD]`-measured.)

## 2026-06-06 — DST-correct session re-campaign + Kronos look-forward fix + full-suite audit

Three GENERIC results from a full DST-correct session re-campaign (`sessions.py`: NY 8–17 America/New_York,
LDN 8–16 Europe/London, Asia 9–18 Asia/Tokyo; `session_mask` = local-tz hour, applied per-day across train+val+
test+oos — GBM/xpair filter DECISION ROWS only on causal-continuous features; Kronos FT uses strict session-only
INPUT, filter-before-window + contiguity).

**(a) Corrected-Kronos confirmation of the sign-invariance theorem (§1) — single-pair price PATTERNS do not forecast
SIGN.** A 1-bar look-forward MISALIGNMENT in the original Kronos direction eval (`kronos_dir.py`/`kronos_ft.py`,
now legacy-gated behind `KRONOS_DIR_LEGACY=1`) scored `Pup = pred_close(i) > C[i-1]` — the move INTO the entry,
window `[t[i-1],t[i]]` — against a deriv label whose window is the FORWARD `[t[i]+1,t[i]+61]`: DISJOINT, off by one
bar. This is failure-mode **FM-F forecast-derivation** (deriving a binary signal from a generative price forecast
scored against a misaligned window); it can only yield a FALSE NULL, never a false POSITIVE. FIX = `kronos_mtf.py`:
context ends AT bar i (last close = entry ref `C[i]`), predict H FORWARD steps, `Pup = pred_close(+H) > C[i]`,
pred-side contiguity + nonoverlap GAP=HS+TOL; validated the forward label agrees with next-bar sign **92.3%**
(n=233,950). With the alignment corrected, **Kronos (a candlestick foundation model) is still NULL on direction at
EVERY horizon 1/5/10/15/30m, zero-shot AND fine-tuned, all sessions** (pooled .50–.51, CPCV p10 .489–.500, all
KILLED, up-rates in-band); fine-tune did NOT help direction. Crucially, **even at NY ≥10m where the cross-pair book
certifies .57–.61, Kronos reads ~.50** — because it ingests only EURUSD's OWN OHLCV candles, not the 7-pair USD
cross-section that carries the edge. This is the cleanest demonstration yet that **the certified direction edge is
CROSS-SECTIONAL (USD-common-factor, §2), not in-pattern**: single-pair candlestick patterns are sign-blind exactly
as the theorem predicts. `[EURUSD·1/5/10/15/30m]` (`kronos_dir_mtf_*_result.json`; FINE-mode "1m→Nm up-the-chain"
+ multi-TF ensembles **(in progress)**).

**(b) The direction edge is NY-CONCENTRATED; magnitude is session-INVARIANT (a new generic axis on §2).** Re-testing
the certified ≥10m cross-pair lever under STRICT session-only DST-correct masking (`session_xpair.py`, gate
`{2:1m_bb_width, 5/10:5m_bb_width, 15:15m_bb_width, 30:1h_bb_width}`): **NY certifies BOTH sides at every horizon;
LDN and Asia certify at NONE.** 10m NY UP .6053 / DOWN .5896 (15/15 each) vs LDN .523/.524, Asia .511/.516; 15m NY
.5845/.5712 vs LDN .527/.520, Asia .519/.496; 30m NY .5681/.5639 vs LDN .520/.514, Asia .487/.509. **NY BEATS the
legacy fixed-UTC gate** (10m legacy .586/.568, 30m legacy .559/.553) — the edge is decisively NY-concentrated, not
diluted across the UTC day. By contrast, **MAGNITUDE certifies in ALL sessions at every horizon** — 1m/2m tick GBM
(magAUC NY .675 / LDN .728 / Asia .717 at 1m; cov≤10% frac_clear 1.0, p10 .58–.71), and 5m/10m/30m base bar GBM all
session-certified (p10 .72–.80) — while **DIRECTION is KILLED in every session for the tick and base-bar GBMs**
(1m/2m pooled ~.504, p10 .496–.504, frac_clear 0.0; 5/10/30m base-bar dir KILLED, NY strongest e.g. 10m NY p10
.525). Generic lesson: **gate the direction book to the NY window; magnitude needs no session gate.**
`[EURUSD·1m/2m/5m/10m/30m]` (`session_1m_{dir,mag}_{ny,ldn,asia}_result.json`, `session_2m_*`,
`session_{5,10,30}m_{dir,mag}_{sess}_result.json`, `session_xpair_{10,15,30}m_{sess}_result.json`; 15m base GBM +
session_xpair 5m/2m **(in progress)**).

**(c) Full-suite FM-F correctness audit — the bug is ISOLATED, no certified book invalidated.** A 19-agent audit +
lead Tier-1 proofs swept all 308 scripts and a dedicated forecast-derivation sweep: **NO second instance of the
misalignment exists outside the Kronos family.** Every other forecast-derivation script (`usdjpy_{1m,2m}_statespace`,
`usdjpy_2m_xhorizon`, `m5_xhorizon`, `m5_lossbatch`, `f1_compound`) predicts the FORWARD quantity over the SAME
horizon as the label at the SAME bar = correctly aligned; GBM/CNN are classifiers trained DIRECTLY on the label and
scored against it ⇒ structurally immune to FM-F. Proven CLEAN empirically (Tier-1): TICK substrate (features causal
via truncation max|full−trunc|=0.0; label = forward deriv outcome, 0/4000 mismatch; corr(y,future)=.486 vs
corr(y,past)=−.003), BAR substrate (forward label 0/2000 mismatch at H=5/10/30; corr(y,future)~.99 vs ~−.02), XPAIR
substrate (0/2000 mismatch H=10), and BAR-FEATURE causality (truncation max|full−trunc|=0.0 across all 239
features). Bounded flags that do NOT invalidate any cert: `barcnn_mag.py:163` + `barcnn_regime.py:69,71` FM-E
selective-threshold on pooled test+oos (CPCV-deflated MAGNITUDE/regime target — fix = use VAL threshold);
`usdjpy_2m_cpcv2.py` FM-E max-p10 cell (best .5185 ≪ .541 → CERTIFIED=false regardless); `min2_mim.py:18` +
`min2_legsign.py:25` FM-A shift(−FWD) no contiguity guard (KILL screens, already KILLED); superseded pre-v3 cohort
(`min1_v*`/`min2_v*`/`tickmodel*`/`tick5s_final`/`tick_ensemble` — documented bar-shift label, no result.json,
replaced by `*_production` books). **Generic takeaway: FM-F is a forecast-derivation failure mode specific to
generative-forecast-→-binary pipelines; direct classifiers and same-bar/same-horizon forecast derivations are
immune, so a misaligned Kronos was a FALSE NULL, not a corruption of the certified GBM/CNN/cross-pair results.**

## 5. See also
- The 7 recurring leakage traps + the sign-invariance shuffle: `METHODS_CATALOG.md` (validation family) and the
  `strategy-eval` skill (`.claude/skills/strategy-eval/SKILL.md`).
- Multiple-testing deflation (Holm/BHY/HLZ/N̂, CSCV/PBO/DSR): `METHODS_CATALOG.md` validation family; per-key
  results in the key's backlog/results file.
