# NOVEL TS-METHODOLOGY RESEARCH — untried methods + input transforms (2026-06-07)

Source: 19-agent research workflow (17 search clusters → dedup/rank synthesis + completeness critic), 110 candidates →
~20 distinct mechanisms, all verified against repo ground truth. Raw slate: `novel_methods_candidates.json`. This doc is
the actionable distillation. Pointers in IDEAS_LOG.md / SWEEP_MATRIX.md.

## 0. TWO MANDATORY GATES (apply to EVERY item below — they post-date and reshape the whole slate)

1. **FROZEN-PAST FORWARD HOLDOUT, not pooled CPCV (leakage trap #9, MAGNITUDE_FINDINGS §6f).** The time-of-day magnitude
   "win" (+0.014 pooled CPCV) was RETRACTED 2026-06-07 — it decays 2024 +.020 → 2026 −.054 on a frozen-past holdout. So
   **the "0.758 rv+clock" baseline does NOT exist as a deployable target.** Rewrite every magnitude falsifier to: *beat
   the forward-robust BASE rv30/rv120 model (.74–.79 every year) on a per-year frozen-past holdout (`deseason_fwd.py`
   pattern, train≤Y → test Y+1,Y+2)* — never pooled CPCV alone. Anything leaning on calendar/seasonal/slow features is
   guilty until forward-proven.
2. **PHASE-RANDOMIZED / IAAFT SURROGATE-NULL (synthesis #5 — build this FIRST, reuse everywhere).** For any nonlinear /
   TDA / signature / multifractal feature, FFT-phase-randomize the source series (keep spectrum, destroy nonlinear
   structure), N=300 surrogates, recompute the OOS metric → null distribution. A candidate that doesn't exceed the 95th
   surrogate percentile is autocorrelation, not signal. `surrogate_null.py`: numpy FFT, ~50 lines, gates #1/#6/#8/#9/#18/#20.

Honest priors (repo-grounded): **MAGNITUDE** quick-wins are likeliest to certify (it's the one robust edge); **DIRECTION**
is ~efficient single-pair and the only edge is engineered cross-pair lead-lag — generic models (Kronos, Chronos-2) read
~.50, so cross-sectional shots are low-probability / high-payoff and must BEAT the certified `m*xp` residual book p10 by
CI95, not merely clear breakeven.

---

## 1. ⭐ INPUT TRANSFORMS — the highest-leverage layer (the user's emphasis; the critic's funded pick is here)

The slate is broad on *models*, narrow on *data representation* — the inverse of where repo evidence points. These reshape
the substrate and unlock many downstream methods at once. Do these before bolting new models onto stale clock-time bars.

| # | Transform | What / why | How on our data | Unlocks |
|---|---|---|---|---|
| T1 ⭐ | **Information-driven bars** (López de Prado: tick / dollar / volume / **imbalance** bars) | Clock-time FX returns are non-IID, heteroskedastic, near-unpredictable. Sampling a bar every N ticks / $X / fixed signed-imbalance restores near-IID returns AND **synchronizes the 7 pairs on INFORMATION, not wall-clock** — potentially sharpening cross-sectional alignment more than any model. The critic's single highest-leverage untried idea. | Rebar the 1s tick store (EURUSD has it; proxy others via 1m vol×range) into imbalance/dollar bars; re-emit the panel + 239-feat on the new clock. Pure preprocessing, reuses the whole GBM+CPCV harness, forward-robust by construction. | the entire cross-sectional + magnitude stack on a better clock |
| T2 ⭐ | **Fractional differentiation** (FFD, min ADF-stationary d*) | **GENUINELY UNTRIED** (grep `fracdiff`=0 hits — the memory note calling it "cataloged" was wrong). The certified cross-pair edge is built on integer-differenced (1-bar) returns, which DESTROY the long-memory level co-movement where a slow USD trend lives. FFD keeps memory while passing ADF. | `pip install fracdiff`. Per pair: pick smallest d* passing ADF on **train only** (freeze, no per-fold refit), causal fixed-width weights on log-close → FFD series; build USD factor/residual on FFD series + lead-lag 1–3. Feed `run_direction` at 15m/30m NY. | cross-sectional direction; a long-memory channel |
| T3 | **Vol-time subordination / return seasonal-adjustment** | The FORWARD-SAFE use of seasonality (vs the killed calendar dummies, §6f): divide *returns* by the trailing seasonal-vol profile so the model sees vol-standardized increments on a "vol clock." | Trailing per-minute-of-day RV profile (train-causal), `r̃ = r / seas_vol`; feed r̃ to direction & magnitude. Gate per-year forward holdout. | direction & magnitude, leakage-safe |
| T4 | **Cross-pair whitening** (ZCA / Cholesky of the 7-pair covariance) | USD-residual via factor regression *smears* lead-lag; whitening makes the residual channels statistically independent first, exposing lead-lag the regression hides. | Train-only covariance → ZCA/Cholesky transform the 7 standardized returns before the GBM / before residualization. | cross-sectional direction |
| T5 | **Analytic-signal phase / Hilbert envelope of the USD common factor** | A cyclicity coordinate (instantaneous phase/frequency of the factor) untouched by the slate; distinct from VMD per-mode IF. | Hilbert transform the (FFD or raw) USD factor → instantaneous phase + envelope as conditioning features / gate. | direction gate |
| #2 | **Build the full 7-pair return panel + frozen USD-factor/residual** (foundation infra) | `crosspair.py` merges only ~10 peer cols — the clean 7-pair panel + residual is NOT yet built. This is the substrate for ALL cross-sectional methods. | `build_panel.py` → `features/panel_<year>.parquet` (7 returns + factor + 7 residuals + lagged factor). **Validation = reproduce the certified m15xp p10 ~.567**; if it doesn't, the panel is leaky — fix before trusting anything downstream. | #3,#4,#8,#11,#16,T2,T4 |

---

## 2. MAGNITUDE — likeliest to certify (the robust edge). Each must beat forward-robust base-rv on a per-year holdout (§0).

**The critic's biggest omission = the entire GARCH/HAR econometric-vol canon** (the slate brought TDA/signature exotica
instead of the workhorses). These are forward-robust by construction (within-window decompositions, not calendar memorization):

- **M1 ⭐ HAR-RV-J (jump-decomposed HAR)** — split realized variance into continuous + jump (bipower variation) components;
  the jump part is orthogonal to rv30/rv120 and forward-robust. Also **realized-GARCH, EGARCH/GJR (leverage asymmetry),
  Markov-Switching Multifractal (MSM)**. *Falsifier:* beat base-rv forward holdout by +0.005 AUC, ≥2 horizons.
- **M2 ⭐ Realized SEMIVARIANCE (RS⁺ / RS⁻, Barndorff-Nielsen)** — upside vs downside realized variance from 1s/1m returns.
  Cheap, on-disk, queued in §7. **Bonus: the signed component RS⁺−RS⁻ has documented DIRECTIONAL content (Patton-Sheppard
  "Good/Bad Volatility")** — one of the few vol constructs matched to the dead DOWN side. *Falsifier:* mag +0.005 forward;
  direction must clear .541 NY.
- **M3 HARQ realized-quarticity** (synthesis #7) — `RQ = (N/3)Σr⁴` per bar; add `rv·√RQ` attenuation interaction. Trivial
  pandas. *Falsifier:* +0.005 forward AND not collinear with rv (|corr|<0.9, SHAP>rv120).
- **M4 Wasserstein-between-consecutive-persistence-diagrams** (synthesis #1, "Topological Tail Dependence") — Takens cloud
  per window → Rips diagram → `W(D_t,D_{t-1})` scalar; published lift concentrates in turbulent regimes. *Falsifier:*
  +0.005 forward, clears surrogate-null (T0 #2), SHAP>rv120.
- **M5 The "multiscale-vol descriptor" family — treat as ONE bet, cheapest member first** (critic): Euler-characteristic
  curves (#20, cheapest), signature even-order/QV norms (#6), Wavelet-Scattering-Spectra (#17), MFDFA Δα / wavelet-leader
  c2 (#18). rv30/rv120-over-windows is already a crude multiscale energy summary; the bar is "adds ORTHOGONAL info" — run
  ECC first, only escalate if it beats base-rv forward + surrogate-null.
- **M6 MOMENT frozen masked-reconstruction MSE** (synthesis #13) — atypicality/non-forecastability feature + abstain gate
  (drop high-MSE windows). *Falsifier:* +0.005 forward; abstain must raise per-trade wc_ret net of fewer trades.

---

## 3. CROSS-SECTIONAL DIRECTION — the real prize (low-prob, high-payoff). Must BEAT the certified m*xp book p10 by CI95.

All on the full 7-pair panel (#2 infra), NY-session, deriv-faithful, refit-CPCV + forward holdout. Generic models read ~.50;
these are *mechanism-matched* to the engineered lead-lag edge.

- **D1 ⭐ Lead-lag signature cross-terms** (synthesis #3, Hoff staircase → depth-2/3 log-signature) — the level-2 cross-terms
  between pair i's *lead* and pair j's *lag* directly encode signed lead-lag / quadratic-covariation. Cheapest cross-sectional
  shot, `iisignature` (CPU). *Falsifier:* beat residual book p10 CI95; AND a lead/lag-shuffle permutation MUST degrade it
  (else it's not using lead-lag → kill).
- **D2 ⭐ Untruncated SIGNATURE KERNEL on the cross-pair NY path** (synthesis #1-sig, Goursat-PDE, `sigkernel`+KeOps-GPU) —
  categorically new vs the killed single-pair Lévy-area (`m30_sig.py`); applied where the edge lives. Kernel-SVM (sign) +
  KRR (rank). *Falsifier:* beat residual book p10 CI95; ablate vs depth-1 RBF to prove the lift is signature-specific.
  Pair with T1 information bars for best shot.
- **D3 Randomized signatures** (synthesis #4, Cuchiero ~256-d frozen-random CDE state) — sidesteps the factorial blowup of
  true depth-3 sigs on d=8 channels (why prior work stalled at level-2). *Falsifier:* must beat BOTH the Lévy-area null
  (~.51) AND the ESN/GRU reservoir null (~.49–.52), else the signature framing adds nothing.
- **D4 ⭐ FASCL — Future-Aligned Soft Contrastive embedding** (synthesis #11, the principled big swing) — supervise the
  embedding geometry with FORWARD cross-sectional co-movement `S_ij = corr(fwd-return_i, fwd-return_j)` (purge t+H), fixing
  the exact sign-blindness that nulled Kronos/Chronos-2. Small dilated-conv encoder, 8GB GPU. *Falsifier:* (residual+FASCL)
  beats residual-only CI95; standalone must beat .52 (not the Chronos-2 null) and not be corr>0.9 with USD-residual.
- **D5 Causal-discovery lead-lag** (critic gap): PCMCI/PCMCI+, Granger-with-FDR, **structural-VAR with sign restrictions** —
  prune SPURIOUS contemporaneous correlation to isolate *which* pair causally leads under the USD factor. Arguably more
  on-target than any signature variant since the certified edge IS lead-lag. *Falsifier:* causal-pruned lead-lag features
  beat the associative residual book p10.
- **D6 HAVOK intermittent-forcing / Hankel-DMD mode-phases** (synthesis #9/#16) — a *signed* dynamic precursor (unlike the
  null CCM/perm-entropy): `v_r` forcing leads regime-changing moves; complex mode-phase angles = continuous lead-lag. An
  explicit global linear operator where Chronos-2 attention failed. *Falsifier:* signed-v_r beats .541 AND book gating lifts
  p10 ≥0.5pp, up-rate∈[.47,.53], surrogate-null.
- **D7 Realized signed-semivariance direction** (= M2's direction arm) + **D8 quantile-direction baseline** (critic): a plain
  **LightGBM-quantile / quantile-regression-forest** emitting `P(ret>0)` from the 239 features — the cheapest probabilistic-
  direction control, and the necessary fair-ablation partner before ANY TSFM-quantile claim (Sundial #15, Lag-Llama) is credible.

---

## 4. GATE / REGIME — condition the certified NY book (attacks the documented refit/regime-dependence)

- **G1 Persistent homology of the rolling 7-pair correlation cloud** (synthesis #8, Gidea-Katz landscape norm) — the canonical
  cross-sectional TDA regime detector; gate the m*xp trade on a benign-norm band. (7 points → shallow H1; enrich with crosses
  or fall back to H0.) *Falsifier:* lifts worst-VAL-half p10 ≥+0.005 at ≥70% trade count; clears surrogate-null.
- **G2 BOCPD / HMM-HSMM with duration** (critic gap) — Bayesian Online Change-Point gives a calibrated run-length posterior =
  a principled, cheap, well-understood gate that directly attacks the corr(VAL,OOS)=−0.54 non-stationarity trap. Far simpler
  than CROCKER/HAVOK tensors. (HMM forward FILTER only — leakage trap #1.)
- **G3 Online/windowed DMD reconstruction-residual + eigenvalue drift** (synthesis #10) — change-point/transition flag; trade
  only in DMD-stable windows. *Falsifier:* +0.5pp p10 on ≥2 horizons; residual must not just track rv.

---

## 5. REPRESENTATION / LABELING / VALIDATION (cross-cutting)

- TabPFN-v2 LOFO embeddings appended to the cross-pair GBM (synthesis #14, in-context support must respect CPCV purge).
- Sundial generative trajectory MC P(up) (synthesis #15) — only TSFM that samples sign-bearing paths; long-shot direction,
  cheap parallel magnitude (endpoint-std). Needs D8 quantile baseline as control.
- SSA last-point causal denoise before residual (synthesis #19); Mamba/S5 on returns (low prior — sequence/deep already null).
- **Validation infra to BUILD:** surrogate-null (§0 #2) and the per-year forward-holdout (§0 #1, `deseason_fwd.py` exists) —
  these are the highest-ROI items because they make every other experiment honest.

---

## 6. ❌ DO NOT RE-RUN (false novelty — already killed here)

- **Hurst / variance-ratio / roughness-index / MFDFA / persistence-scalar AS DIRECTION** — `m5_signinv_dir.py` already ran the
  sign-invariant complexity family standalone AND incremental into m5xp; the **sign-invariance theorem** says these gate SIZE
  not SIGN. Any "direction from a roughness scalar" candidate is a renamed killed experiment. Legitimate ONLY as magnitude.
- **Calendar/seasonal dummies, CoST seasonal head, TabPFN-TS calendar covariates, fixed deseasonalization** — §6f killed the
  forward edge; only an ADAPTIVE/rolling profile gated by a forward holdout may be revisited.
- **Multi-scale-vol exotica as four separate efforts** — collapse WSS/MODWST/SCINet/ECC/sig-QV/MFDFA into ONE "does a learned
  multiscale vol descriptor beat 2-window rv forward" bet; cheapest member (ECC) first.
- **Mamba-dt "activity clock"** — its own falsifier concedes it must beat "trade only in NY high-volume minutes" (sess flags).

---

## 7. RECOMMENDED EXECUTION ORDER

1. **Infra first** (unlocks + de-risks everything): surrogate-null gate (§0 #2) · per-year forward-holdout (§0 #1, exists) ·
   the full 7-pair panel + frozen residual (#2, validate by reproducing m15xp .567).
2. **Cheap magnitude workhorses** (likeliest to certify): HAR-RV-J / realized-GARCH (M1) · realized semivariance RS±/HARQ
   (M2/M3) — all forward-robust, on-disk, hours of CPU.
3. **The two highest-leverage INPUT transforms** (the user's theme): information-driven bars (T1) · fractional differencing
   (T2) — rebar/retransform, then re-run the certified books on the new substrate.
4. **Cross-sectional direction shots** (mechanism-matched, on the new panel/clock): lead-lag signature cross-terms (D1) →
   signature kernel (D2) → FASCL (D4) / causal lead-lag (D5) → HAVOK/Hankel-DMD (D6). Each must BEAT the m*xp book p10, clear
   surrogate-null, and survive the forward holdout.
5. **Gates** if a cross-sectional signal survives: TDA-corr-cloud (G1) / BOCPD (G2).
