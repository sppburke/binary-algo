# NOVEL TS-METHODOLOGY RESEARCH — untried methods + input transforms (2026-06-07)

Source: 19-agent research workflow (17 search clusters → dedup/rank synthesis + completeness critic), 110 candidates →
~20 distinct mechanisms, all verified against repo ground truth. Raw slate: `novel_methods_candidates.json`. This doc is
the actionable distillation. Pointers in docs/IDEAS_LOG.md / SWEEP_MATRIX.md.

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

**§0 OUTCOME (2026-06-07): BOTH GATES BUILT + SELF-CHECKED.** (1) `fwd_holdout.py` — reusable frozen-past forward-holdout
gate (magnitude AUC+lift / direction cov-selacc modes); self-check DEPLOYS a stationary signal (+0.10/yr) and REJECTS a
trap-#9 non-stationary deseason-+tod feature (decays to −0.015) — the exact failure mode it exists to catch. (2)
`surrogate_null.py` — phase-randomize/IAAFT null; self-check finds linear lag-1 autocorr NOT significant (real .6959 ≈
null_p95 .6961, spectrum preserved) while nonlinear |.| vol-clustering IS significant (real .0883 ≫ null_p95 .0199).
Both committed (051e525). Now MANDATORY on every item below.

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
| #2 ✅ | **Build the full 7-pair return panel + frozen USD-factor/residual** (foundation infra) | `crosspair.py` merges only ~10 peer cols — the clean 7-pair panel + residual is NOT yet built. This is the substrate for ALL cross-sectional methods. | `build_panel.py` → `features/panel_<year>.parquet` (7 returns + factor + 7 residuals + lagged factor). **Validation = reproduce the certified m15xp p10 ~.567**; if it doesn't, the panel is leaky — fix before trusting anything downstream. | #3,#4,#8,#11,#16,T2,T4 |
| | **#2 OUTCOME (2026-06-07): BUILT + FAITHFULNESS-CERTIFIED.** 5,345,437 bars 2012-2026, 7 eu-equiv returns + USD factor + 7 residuals + raw EURUSD close. The pooled-CPCV reproduction read p10 **.5355** (mean .5425), BELOW the .567 target — but the decisive gate is feature-faithfulness, not the absolute level under a weaker harness: `panel_faithcheck.py` proves all continuous channels (`eu_r/usdbask/catchup/eurresid/disp/ll_*`) are **BIT-IDENTICAL** to certified `build_xp` (max_abs 0.0); only `agree*` differs at 1 row/lookback (0.0003%, warmup off-by-one). The lower CPCV level is the DOCUMENTED weaker-reimplementation effect (single 600-tree pooled LGBM vs certified refit ~.54), NOT leakage (which would INFLATE). Up-rate .5074 inside the moved-bars tripwire. **Panel SAFE for all downstream.** Record: `panel_faithcheck_result.json`, `build_panel_validate_15m.json`. | |

**T2 OUTCOME (2026-06-07): TESTED → KILLED for direction.** `frac_diff.py` (hand-rolled FFD, fixed-width weights + ADF
d*-selection; self-check: random walk needs d*=0.1 to pass ADF p .019 while keeping 98.8% level-memory vs 0.9% for plain
returns — "stationarity with memory", genuinely untried). `frac_direction.py` folds FFD USD factor/residual/lead-lag into
the certified direction book (forward holdout, NY cov0.10 selacc). Per-pair d*: EURUSD/NZD/CHF=0.1, GBP/AUD/JPY/CAD=0.2.
**15m KILLED** (`frac_direction_15m_result.json`): base selacc .5900/.5520/.5254 (pooled .5642); +ffd .5792/.5301/.5403
(pooled .5537) → DECAYS, deployable=false; ffdonly pooled .5001 = coin flip. **30m ALSO KILLED, harder**
(`frac_direction_30m_result.json`): base .5842/.5483/.5007 (pooled .5558); +ffd .5546/.5135/.5064 (pooled .5308) → DECAYS,
deployable=false; ffdonly pooled .4958 below coin flip. FFD level-memory adds nothing to direction and hurts recent years.
(Note: base book itself decays .59→.55→.525 across forward years — consistent with documented refit-dependence.)

**T1 OUTCOME (2026-06-07): SUBSTRATE SCOUTED → DEFERRED behind T2.** `features_tick/{train,val,test,oos}_1s.parquet` holds
EURUSD 1s bars (cols mid/imb/micro/spread/nt/tsz) but is **EURUSD-ONLY and 2021+**, so cross-pair synchronization on an
information clock needs proxies for the other 6 pairs (NOVEL T1 anticipated this). NOT yet built — DEFERRED.

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

**M1/M2/M3 OUTCOME (2026-06-07): TESTED → REAL-but-SUB-BAR; magnitude on-disk EXHAUSTED.** `mag_har.py` +
`mag_har_result.json` (recorded MAGNITUDE_FINDINGS §6g). Arms added to certified base `[-pe,rv30,rv120]`, forward holdout,
horizons 10/15/30m, target |ret_H|≥train-Q75. Falsifier = +0.005 AUC in ≥2 forward years AND ≥2 horizons. **ALL ARMS FAIL:**
+har (multiscale RV) mean fwd ΔAUC +0.0010 (2/3 — sub-bar); +jump (bipower) +0.0003 (1/3 — null); +semivar (RS⁺/RS⁻/
signed-jump) −0.0000 (1/3 — null); +harq (realized quarticity) +0.0004 (2/3 — null); **+all (stacked) +0.0021, 3/3
deployable, NO decay — forward-CONSISTENT but economically negligible.** Base rv already extracts ~all magnitude
(collinearity vs rv120: lRV120 .89, RS .70, HARQ .58/−.63; only signed-jump SJ120 orthogonal −.04 and carries nothing).
**The one positive:** RE-VALIDATES the certified magnitude edge on a clean deployment-faithful holdout — base AUC
.799/.750/.750 (10m) .791/.739/.737 (30m), 4–5.6× decile lift, NO decay. VERDICT: KILLED as deployable upgrade.
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
  > **D1 OUTCOME (2026-06-07): TESTED → KILLED both horizons** (`xsec_direction.py`, `xsec_direction_sig_{15,30}m_result.json`;
  > iisignature won't compile (no Python.h) → depth-2 iterated integrals S^{ij}=∫∫dX^i dX^j + Lévy area A^{ij}=½(S^{ij}−S^{ji})
  > computed pure-numpy via cumsum over trailing 30-bar windows). Forward holdout, NY cov0.10 selacc. **15m:** +sig DECAYS
  > (Δ −.0043/−.0099/+.0032), sigonly pooled .5243 ≈ sig_shuf .5236 → FAILS the mechanism-specificity falsifier (the lead/lag
  > rotation surrogate does NOT degrade it → not genuine lead-lag content). **30m:** +sig DECAYS all 3 years
  > (−.0094/−.0043/−.0133), sigonly pooled .5283 (sub-base). Killed: adds nothing to the certified book, fails its own null.
- **D2 ⭐ Untruncated SIGNATURE KERNEL on the cross-pair NY path** (synthesis #1-sig, Goursat-PDE, `sigkernel`+KeOps-GPU) —
  categorically new vs the killed single-pair Lévy-area (`m30_sig.py`); applied where the edge lives. Kernel-SVM (sign) +
  KRR (rank). *Falsifier:* beat residual book p10 CI95; ablate vs depth-1 RBF to prove the lift is signature-specific.
  Pair with T1 information bars for best shot.
  > **D2 OUTCOME (2026-06-07): BLOCKED — not run.** `sigkernel`/KeOps need a GPU and a C-extension build that fails in this
  > env (no `Python.h` dev headers). Deferred to a GPU+headers environment; LOW prior given D1 (signatures, same family)
  > already null on its mechanism falsifier.
- **D3 Randomized signatures** (synthesis #4, Cuchiero ~256-d frozen-random CDE state) — sidesteps the factorial blowup of
  true depth-3 sigs on d=8 channels (why prior work stalled at level-2). *Falsifier:* must beat BOTH the Lévy-area null
  (~.51) AND the ESN/GRU reservoir null (~.49–.52), else the signature framing adds nothing.
- **D4 ⭐ FASCL — Future-Aligned Soft Contrastive embedding** (synthesis #11, the principled big swing) — supervise the
  embedding geometry with FORWARD cross-sectional co-movement `S_ij = corr(fwd-return_i, fwd-return_j)` (purge t+H), fixing
  the exact sign-blindness that nulled Kronos/Chronos-2. Small dilated-conv encoder, 8GB GPU. *Falsifier:* (residual+FASCL)
  beats residual-only CI95; standalone must beat .52 (not the Chronos-2 null) and not be corr>0.9 with USD-residual.
  > **D4 OUTCOME (2026-06-07): BLOCKED — not run.** The FASCL dilated-conv encoder needs an 8GB GPU not available in this
  > env. Deferred to a GPU environment.
- **D5 Causal-discovery lead-lag** (critic gap): PCMCI/PCMCI+, Granger-with-FDR, **structural-VAR with sign restrictions** —
  prune SPURIOUS contemporaneous correlation to isolate *which* pair causally leads under the USD factor. Arguably more
  on-target than any signature variant since the certified edge IS lead-lag. *Falsifier:* causal-pruned lead-lag features
  beat the associative residual book p10.
  > **D5 OUTCOME (2026-06-07): REASONED-SKIP — not run.** The certified base book ALREADY contains every peer's lagged
  > lead-lag feature (`ll_<pair><k>`, k∈{1,3,5,10,15,30}) which the GBM weights; D1 just proved signature lead-lag content
  > does NOT survive a rotation null. A Granger/PCMCI feature-SELECTION on top of an already-lead-lag-saturated GBM has
  > near-zero marginal prior. Re-open ONLY with EXTERNAL leaders (rate-diff), not more EURUSD-panel selection.
- **D6 HAVOK intermittent-forcing / Hankel-DMD mode-phases** (synthesis #9/#16) — a *signed* dynamic precursor (unlike the
  null CCM/perm-entropy): `v_r` forcing leads regime-changing moves; complex mode-phase angles = continuous lead-lag. An
  explicit global linear operator where Chronos-2 attention failed. *Falsifier:* signed-v_r beats .541 AND book gating lifts
  p10 ≥0.5pp, up-rate∈[.47,.53], surrogate-null.
  > **D6 OUTCOME (2026-06-07): TESTED → KILLED both horizons** (`xsec_direction.py`, `xsec_direction_havok_{15,30}m_result.json`;
  > frozen-basis HAVOK / Hankel-Koopman: delay-embed q=60 the USD-factor trend, SVD on TRAIN → freeze r=8 modes, causally
  > project to v1..v7 + intermittent forcing v_r + leading phase). Forward holdout, NY cov0.10 selacc. **15m:** +havok DECAYS
  > (Δ −.003/−.0024/+.0014), havokonly pooled .5148 — SUB-BREAKEVEN (far below base .5624), only marginally beats its
  > phase-randomized surrogate (.5148 vs .5063) and NEVER clears .541. **30m:** +havok DECAYS, havokonly pooled .5197
  > (sub-base). Killed: the signed Koopman forcing carries no deployable direction content.
- **D7 Realized signed-semivariance direction** (= M2's direction arm) + **D8 quantile-direction baseline** (critic): a plain
  **LightGBM-quantile / quantile-regression-forest** emitting `P(ret>0)` from the 239 features — the cheapest probabilistic-
  direction control, and the necessary fair-ablation partner before ANY TSFM-quantile claim (Sundial #15, Lag-Llama) is credible.
  > **D7 OUTCOME (2026-06-07): TESTED → REAL-but-SUB-BREAKEVEN, KILLED (non-deployable)** (`xsec_direction.py`; Patton-Sheppard
  > good/bad realized signed-semivariance). The ONLY direction shot that PASSES its mechanism null: semivaronly beats
  > semivar_shuf (sign-flip control) by **+.015 (15m) / +.030 (30m)** → the signed-vol asymmetry GENUINELY carries DIRECTIONAL
  > content (distinct from D1/D6, which fail their nulls). BUT weak and non-deployable: semivaronly pooled **.5258 (15m) /
  > .5382 (30m)** — below breakeven .541 (only 2024@30m .5516 clears it), DECAYS forward, and +semivar does NOT add to the
  > base book (Δ15m −.0154/−.0029/−.0107, Δ30m −.0081/+.0062/−.0139). The DIRECTION analog of magnitude's "real-but-sub-bar":
  > genuine content, sub-deployable, already subsumed by the cross-pair book. **NOTHING CERTIFIED; leaderboard UNCHANGED.**
  > **D8 quantile-direction baseline OUTCOME (2026-06-07): CONTROL-SKIP — not run.** D8 is a CONTROL to ablate a TSFM-quantile
  > claim (Sundial/Lag-Llama), which this campaign does not make. Skip.

> **§3 CONVERGENT VERDICT (2026-06-07):** FFD (T2) + signature lead-lag (D1) + HAVOK (D6) + signed-semivariance (D7) ALL
> killed forward — NO cross-sectional direction family beats/matches the certified book. The base cross-pair book itself
> DECAYS forward (15m .5895/.5518/.5234; 30m .5827/.5505/.5055 — documented refit-dependence). D7 confirms genuine-but-weak
> signed-vol direction content exists (clears its null) yet is sub-breakeven and subsumed. Corroborates the standing program
> conclusion: **direction beyond the engineered cross-pair book is EFFICIENT on existing data; the only frontier is EXTERNAL
> data (rate-diff / vol / EURGBP ticks).** Nothing certified; UP/DOWN leaderboard UNCHANGED.

---

## 4. GATE / REGIME — condition the certified NY book (attacks the documented refit/regime-dependence)

> **§4 OUTCOME (2026-06-07): G1/G2/G3 MOOT — none built.** A gate conditions a SURVIVING signal; no direction signal
> survived the forward holdout (§3 all killed), so there is nothing to gate. G1 also needs `gudhi` (unbuildable here, no
> Python.h); G2 (`ruptures`) and G3 are available. Re-open ONLY if a future (external-data) signal clears breakeven first.

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
   the full 7-pair panel + frozen residual (#2, validate by reproducing m15xp .567). **✅ DONE (2026-06-07):** both gates
   built+self-checked (`fwd_holdout.py`, `surrogate_null.py`); panel BUILT + faithfulness-certified (see §0 / §1 #2 outcomes).
2. **Cheap magnitude workhorses** (likeliest to certify): HAR-RV-J / realized-GARCH (M1) · realized semivariance RS±/HARQ
   (M2/M3) — all forward-robust, on-disk, hours of CPU. **✅ DONE (2026-06-07): all REAL-but-SUB-BAR / KILLED as upgrade;
   base magnitude re-validated, NO decay (see §2 M1/M2/M3 outcome). Magnitude on-disk EXHAUSTED.**
3. **The two highest-leverage INPUT transforms** (the user's theme): information-driven bars (T1) · fractional differencing
   (T2) — rebar/retransform, then re-run the certified books on the new substrate. **⏳ PARTIAL (2026-06-07): T2 FFD DONE →
   KILLED for direction at 15m & 30m (see §1 T2 outcome); T1 info-bars substrate scouted but NOT built — PENDING.**
4. **Cross-sectional direction shots** (mechanism-matched, on the new panel/clock): lead-lag signature cross-terms (D1) →
   signature kernel (D2) → FASCL (D4) / causal lead-lag (D5) → HAVOK/Hankel-DMD (D6). Each must BEAT the m*xp book p10, clear
   surrogate-null, and survive the forward holdout. **✅ DONE for the tractable families (2026-06-07): `xsec_direction.py` RAN
   D1 (depth-2 lead-lag signature + Lévy area, pure-numpy), D6 (frozen-basis HAVOK), and D7 (signed-semivariance), each with
   shuffle/surrogate controls — ALL KILLED forward (D1 & D6 fail their mechanism null; D7 REAL-but-sub-breakeven, passes its
   null yet non-deployable). See §3 D1/D6/D7 outcomes + the §3 convergent verdict. D2/D4 BLOCKED (GPU/headers), D5
   REASONED-SKIP (book already lead-lag-saturated), D8 control-skip — none built.**
5. **Gates** if a cross-sectional signal survives: TDA-corr-cloud (G1) / BOCPD (G2). **MOOT (2026-06-07): none built — no
   direction signal survived step 4 to gate (§4 outcome). Re-open only if a future external-data signal clears breakeven.**
