# EURUSD Binary-Direction Research — MASTER METHODOLOGY CATALOG

**Technique-centric, timeframe-agnostic.** Every methodology used in this program gets a self-contained entry: *what it is · how to use it · why to use it · process notes (leakage traps / discipline) · status*. Each entry cites its implementing file and points STATUS at where the verified result lives. For per-horizon **results** and the **up/down leaderboard**, see the companion `EURUSD_RESULTS.md`.

All scripts live in `/media/sean/CORSAIR/binary-algo/`. Last updated 2026-06-01 (xofi + CCM both KILLED).

---

## 0. Universal retargeting / discipline machinery (applies to every entry)

- **Horizon knobs.** Bar methods read horizon from env `MX_HOR=<minutes>` (`m5_xpair.py:22` `HOR=int(os.environ.get("MX_HOR","5"))`; GAP auto = HOR×60s; `MX_HOR=1` → 1-min label). Tick methods set wall-clock expiry `HS`/`HSEC` (seconds) + `TOL_S` entry tolerance. The 239 multi-TF features in `features/<PAIR>_<year>.parquet` (2012–2026, 7 majors) are horizon-independent; the 1s microstructure cache is in `features_tick*/`.
- **Splits.** Bars: train 2012–21 / val 2022–23 / test 2024 / test 2025 / oos 2026 (`harness.py` `SPLITS`). Tick: train 2021–23 / val 2024-H1 / test 2024.09–2025.11 / oos 2026 (tick data spans 2021–2026 only).
- **Settlement discipline (deriv-faithful).** `wc_ret()` = wall-clock mid-to-mid, next-tick entry (1s lag), exit = last tick ≤ expiry, **ties LOSE**, breakeven win-rate **0.541** (payout R≈1.85). `nonoverlap_chrono(ts,mask,gap)` = first-come de-overlap (no look-ahead). `boot()` = bootstrap CI95 over non-overlapping trades.
- **Selection discipline.** Select on VAL by **worst-VAL-half stability**, then require the edge on EACH of {2024,2025,2026} with CI95. Never VAL-acc-max — `corr(VAL_acc, OOS_acc) = −0.54` (`exp_15m_v8_validate.py`). Evaluate moved-bars-only (|ret|>0); verify moved up-rate ∈ [0.47,0.53] (fake-flat tripwire).

---

## Family 1 — Gradient-boosted / tabular direction models

### 1.1 OHLCV multi-TF GBM ensemble (lgb + xgb + cat)
- **What.** Three-family gradient-boosted ensemble over the 239 causal multi-timeframe OHLCV features, predicting `sign(close(t+H)−close(t))`. The workhorse direction model at all bar horizons.
- **How.** `m{5,30}_lab.py`, `m5_production.py`, `m15_production.py`, `m30_production.py`; baseline single-LGB `pipeline.py`+`exp_baseline.py`. Knobs: `MX_HOR`/`HOR`, coverage, regime gate (compression × NY). Feature/label store built by `pipeline.py`; ablation loader `dataset.py`. Frozen weights `models/m{5,10,15,30}_EURUSD_direction_{lgb.txt,xgb.json,cat.cbm}` (reusable as cross-horizon-stack parents). Retarget: set `MX_HOR`, re-freeze gate on VAL.
- **Why.** Targets the directional sign edge; richest-feature attempt to beat the ~0.50 wall.
- **Process notes.** The three families agree to the 3rd decimal → the constraint is the DATA, not the algorithm. Must use `nonoverlap_chrono`+`wc_ret` for the headline; VAL-acc-max anti-transfers.
- **STATUS.** 5m null ~0.52 AUC / book 0.583 (`m5_xpair_production.py`); 15m **survived 0.647** combined (`m15_production.py`, `models/m15_EURUSD_strategy.json`); 30m 0.591 (`m30_production.py`); 10m honest 0.602 (`m10_freeze_honest.py`).

### 1.2 Compression-release × reversion specialist book
- **What.** Two levers on the GBM ensemble: (a) trade only in a volatility-compression→release regime (`bbw1800≤q` AND `bbw300` expanding = squeeze breakout); (b) **reversion** filter — bet AGAINST the last move (`sign(p−0.5)==−sign(ret300)`); plus a compression-release LGBM specialist blended 50/50 with the all-bars ensemble.
- **How.** `min1_production.py` (the frozen 60s book), `m15_production.py` (compression×NY). Knobs: `bbw1800_q67`, `rel_p70`, `rel_tighten`, `w_spec`, `conf_thr` (in `models/min1_EURUSD_strategy.json`); `HS` for horizon. Live class `Min1Strategy`.
- **Why.** Confident bets concentrate in low-vol squeeze regimes and are counter-momentum (60s reversion): gate magnitude, then bet reversion direction.
- **Process notes.** Subset-trained specialists are WORSE at 60s/5m (kill the confidence ranking) but HELP at 120s/15m. Directional momentum-confirmation HURTS. The pre-audit headline (OOS 0.872) was the canonical inflation case (bar-shift + greedy de-overlap).
- **STATUS.** 60s corrected **null** 0.539/0.550 (`min1_production.py`); 15m the genuine edge feeding 0.647.

### 1.3 Magnitude-filtered / microprice / subset-trained label engineering
- **What.** Variants that change the training subset or label: microprice (Stoikov) label, drop noisiest 40% by |move|, or train direction ONLY on large-move bars.
- **How.** `min1_v2.py` (microprice+magnitude filter), `min1_v13.py` (large-move-only direction).
- **Why.** Hoped a cleaner label / noise-stripped training set exposes sign.
- **Process notes.** Both DESTROY the confidence ranking — recurring failure mode of subset-training at short horizons.
- **STATUS.** **null**.

### 1.4 TabNet attentive deep-tabular
- **What.** Attentive deep tabular net (sequential-attention feature selection) as a 4th model class on the 239 feats.
- **How.** `exp_tabnet.py` (CPU, subsampled).
- **Why.** Test whether attention captures nonlinearity GBM misses.
- **STATUS.** **null** — AUC 0.513–0.519, lands with GBM/GRU.

### 1.5 Cross-sectional rank / dollar-neutral falsifier
- **What.** Remove the common USD factor by ranking 7 majors cross-sectionally and trading a dollar-neutral long-short basket; tests whether idiosyncratic signal beats raw.
- **How.** `f2_rank.py` (15m); stat-arb residual `statarb.py`.
- **Process notes.** Gross Sharpe ~6 is inflated ~15× by label overlap + zero cost — charge cost and de-overlap.
- **STATUS.** **null** — idio AUC 0.5115 ≤ raw 0.5141.

---

## Family 2 — Regime gating & meta-labeling

### 2.1 Compression / session / coverage gate sweep
- **What.** Sweep regime conditions (vol-compression quantile × session × confidence coverage) and select the operating point that generalizes, scored against an ORACLE max-floor.
- **How.** `m5_lab.py`, `m10_gate_sweep.py` (200 configs), `m30_lab.py`, `exp_15m_v5_gates.py`/`exp_15m_v6_specialist.py`. Select by worst-VAL-year-half; knobs = compression q, session flag, coverage %.
- **Why.** Direction is ~0.52 on average but the average mixes predictable/coin-flip regimes; gate to the predictable subset.
- **Process notes.** VAL-acc-max picks anti-transfer pockets (the 10m 0.667 artifact); thin-coverage spikes (n<25–50) are multiple-testing mirages.
- **STATUS.** Capped ~0.60 honest everywhere; **helped/null** depending on horizon.

### 2.2 Meta-labeler on orthogonal axes (López de Prado)
- **What.** A 2nd model predicts P(primary direction correct) from axes ORTHOGONAL to the 239 (cross-pair agreement/dispersion, order-flow, parent confidence); abstain unless meta ≥ threshold (worst-VAL-half).
- **How.** `m5_meta.py` (best honest 5m ~0.61), `m15_meta.py`, `m10_stack.py`, `exp_15m_v7_meta.py`. Threshold frozen on VAL worst-half.
- **Why.** "Avoid losers" without the VAL-acc-max trap.
- **Process notes.** A meta-classifier ≤ its features' conditional accuracy → cannot exceed the ORACLE floor (~0.60). Meta-correctness AUC was random (0.502) at 15m.
- **STATUS.** **helped** at 5m (0.612); **null** at 15m parent (no lift) and 10m.

### 2.3 Hurst / variance-ratio persistence switch
- **What.** Causal VR(q,W)=Var_W(r_q)/(q·Var_W(r_1)); VR>1 ⇒ persistent (momentum engine), VR<1 ⇒ anti-persistent (reversion engine), VR≈1 ⇒ abstain.
- **How.** `min1_hurst.py`. Knobs: (q, W, m-band), conf, gate — selected on VAL worst-half; reuses `min1_production` blend.
- **Why.** Treat the 2025 wall as a price-memory inversion fixable by switching engine per persistence state.
- **STATUS.** **null** — worst-VAL-half floor 0.513, oracle 0.555.

### 2.4 Structural / price-action / Sofien rule gates
- **What.** Pre-committed RSI/BB-%b/range-position reversion, Connors-RSI2, TD-setup, NR7 breakout, 3-2-2 strat patterns, larger-TF oscillator-extreme reversals, FX fixing-window reversal (Krohn-Mueller-Whelan).
- **How.** `m5_patterns.py`, `m5_sofien.py`, `m5_sofien_confluence.py`, `m30_regime.py`, `m30_gates.py`, `m30_fix.py`; EDA `eda*.py`.
- **Process notes.** Corpus "hit ratios" are RR-exit-managed, not directional; the fixing-window edge is real in-sample but sign-flips OOS (post-2013 WMR reform).
- **STATUS.** all **null** for direction (~0.47–0.55).

---

## Family 3 — Cross-horizon & cross-pair information

### 3.1 Cross-horizon STACK (parent → child front-load)
- **What.** Front-load a longer-horizon parent ensemble's confident direction into the shorter outcome, gated by a learned meta-labeler P(parent correct on child horizon). **The single strongest method in the program.**
- **How.** `m5_stack2.py` (soft, strongest 5m), `m5_stack.py` (hard agreement), `min1_stack.py`, `m10_stack.py`, `m5_wf_stack.py` (walk-forward), `m5_xhorizon.py` (raw transfer probe). Knobs: child label horizon (`MX_HOR`), parent ensemble(s) from `models/m{5,10,15}_*`, meta threshold (worst-VAL-half).
- **Why.** Borrow the cleaner longer-horizon signal to beat a horizon's own noise floor.
- **Process notes.** Child asymptotes to the parent's native ceiling; front-loads weakly when child ≪ parent (60s off a 5m/15m parent ≈ 0.51). Hard agreement starves OOS coverage (n16).
- **STATUS.** **helped/best** at 5m (0.613 verifiable / 0.648 thin, `m5stack_EURUSD_strategy.json`); **null** at 60s (0.586) and 10m.

### 3.2 Cross-pair USD-residual / common-factor / lead-lag
- **What.** Decompose EURUSD into EUR-strength − USD-strength using a sign-aligned basket of the 6 other majors; the relative-value reversion residual, per-pair lead-lag residual, catch-up residual, dispersion/agreement.
- **How.** `m5_xpair.py` (env `MX_HOR`; modes `xp`/`xpbase`/`xpof`), probe `m5_xpair_probe.py`, merger `crosspair.py`, production `m5_xpair_production.py`. `augment(...,"xpof")` adds 239 base + 18 order-flow cols.
- **Why.** The one orthogonal family **sign-stable across 2024 & 2026** (momentum agreement is dead/sign-blind).
- **Process notes.** Strongest at 5m, fades by 10m, ~0.52 at 60s; does NOT lift the 15m parent.
- **STATUS.** **helped** 5m 0.586; **null/capped** 10m 0.556, 15m, 30m.

### 3.3 RMT (Marchenko-Pastur) cross-pair eigen-residual reversion
- **What.** Clean the 7-pair correlation matrix by keeping only eigenmodes above the MP upper edge λ+; project EURUSD onto the cleaned common factors; trade the idiosyncratic residual reversion — mechanistically immunized against the 2025 USD-factor inversion.
- **How.** `min1_rmt.py`. Knobs: `H` (1=60s, 15=15m), coverage by |standardized residual|; eigenvectors/MP-edge fit TRAIN-only; INNER JOIN no-ffill.
- **Why.** A different decomposition that discards the noise bulk that inverts in 2025.
- **Process notes.** Pre-registered falsifier: kill if cleaned-residual 2025 acc ≤ raw-xpair 0.534. Residual reversion is magnitude-gated (capped prior).
- **STATUS.** **null** — only one eigenvalue clears the MP edge (degenerate with the USD basket); 2025 selective 0.508–0.520, falsifier fires (`/tmp/min1_rmt.log`).

### 3.4 Residualized TARGET (label, not feature)
- **What.** Change the LABEL to `sign(eu_ret_H − β_t·basket_ret_H)` with causal rolling-OLS β; trade where residual-sign AND raw-sign agree.
- **How.** `min1_residtarget.py`. Knobs: `H` (15 then 1), `beta_win=500`, `beta_minp=100`, NY gate, agreement gate.
- **Why.** Attack the 2025 USD-inversion at the label root.
- **STATUS.** **KILLED** — resid-sign OOS<0.55, agreement-subset 2025 doesn't beat 0.534 (`min1_residtarget_result.json` `"KILLED": true`).

### 3.5 External cross-asset lead-lag (ES/NQ → EURUSD)
- **What.** Test whether S&P e-mini (ES) leads EURUSD's next-Nm direction via the risk-on channel (read-only external LEAN repo).
- **How.** `m10_xasset_probe.py`, `m30_es_feas.py`.
- **STATUS.** **null + mechanistic key** — contemp corr +0.16/0.22 but lagged ES→fwd corr FLIPS sign +0.02 (2024) → −0.05 (2025); *this is why* every method's test25 floor collapses.

### 3.6 Signal-compounding falsifier (3s → 15m)
- **What.** Hold/compound the 3s seconds-edge into a 15m bet, net of spread.
- **How.** `f1_compound.py`.
- **STATUS.** **null** — single-call decays 0.52@3s→0.50@900s; net of spread strongly negative.

---

## Family 4 — State-space & dynamical-systems

### 4.1 Hidden Markov regime model (Gaussian, causal-filtered)
- **What.** Gaussian HMM on causal emissions; assign held-out bars by the FORWARD-FILTERED posterior P(state_t|obs_1..t). Three uses: regime-gate / per-state engine-switch / soft-posteriors-as-meta-features.
- **How.** `min1_hmm.py`. Env: `HMM_K` (default 3), `HMM_EMIT` (csv emissions), `HMM_COV` (full|diag), `HMM_SEED`. Retarget: change label horizon, reuse `causal_states`. hmmlearn installed.
- **Why.** Does a latent regime switch break the wall.
- **Process notes (LEAKAGE TRAP).** NEVER use Viterbi/forward-backward — they peek at the whole series. Forward filter only.
- **STATUS.** **null** — states are pure volatility regimes (P(up)≈0.497–0.499), zero directional lift; best refinement (U2 engine-switch) ~0.60 thin-cov, CI spans breakeven (`models/min1_hmm_summary.json`).

### 4.2 Kalman filter (level/slope channel, velocity, time-varying β)
- **What.** Local-linear-trend Kalman → filtered fair-value (channel reversion), filtered slope (velocity/momentum), and time-varying cross-pair β residual.
- **How.** `min1_kalman.py`. Knobs: process/measurement noise ratio, signal choice, threshold (VAL worst-half). filterpy/pykalman installed.
- **Process notes (LEAKAGE TRAP).** Forward recursion only — the RTS smoother leaks the future.
- **STATUS.** **null** — channel 0.499, velocity 0.492, β-residual 0.504. Channel ≈ adaptive compression×reversion (magnitude, not sign).

### 4.3 Convergent Cross-Mapping (CCM) coupling-GATE
- **What.** Sugihara-2012 nonlinear-causality test: does driver X's shadow manifold cross-map EURUSD's, with skill that *converges* as library length L grows. Used as a GATE (when to trade) behind the frozen sign model — leakage-immune (no feature fed to the model).
- **How.** `min1_ccm.py` (full design in `CCM_DESIGN.md`). E=4, τ=1, E+1 neighbors, mandatory Theiler window ±(E·τ+tp), L∈{50…Lmax}; lagged-CCM forward tp*∈[1,60]s required (Ye 2015); Ebisuzaki + twin surrogates; θ_gate by VAL worst-half CI95-lower ≥0.515. Self-coupling sanity gate runs FIRST. `python min1_ccm.py selfcheck`.
- **Why.** High driver→EUR coupling ⇒ dynamics more driver-led/deterministic ⇒ frozen sign model more accurate there.
- **Process notes (FIREWALL).** Fixed 1s grid built ONLY for embedding (returns/OFI summed, empty second → 0.0, NEVER ffill); labels/trades/accuracy stay on the native clock via `wc_ret`. 6 pre-registered falsifiers. ~40 min, <1.5GB. **Implementation gotcha (fixed):** library neighbors near a window's end overflow `X[cand+tp]` for tp>0 — restrict the pool to `base+tp<len(X)` for BOTH targets and library; vectorize the per-target loop (the python loop is hours). Validate the core on a synthetic coupled-logistic (Y-xmap-X converges, reverse flat, surrogate→0) at the SAME tp you'll run.
- **STATUS.** **KILLED** — `min1_ccm_result.json` `FALSIFIER_KILLED:true` (4 conditions). Self-coupling sanity PASSED (EUR-own-OFI has the strongest convergence, slope 0.0066 — embedding is correct, null trustworthy), yet **even EURUSD's own order flow does not convergently cross-map its own 60s return** (slope 0.0066 ≤ 0.02, Δρ 0.036 < 0.05, surrogate-pass 0.44). All 7 drivers conv=False/surr=False; gated per-year 2024≈0.51 / 2025≈0.503 / 2026≈0.52. A dynamical-systems-lens confirmation of 60s efficiency, independent of the ML channels.

### 4.4 Singular-spectrum / fractional-diff / particle-filter / reservoir (backlog)
- **What.** SSA causal decomposition, fractional differentiation (stationary memory-preserving), particle-filter latent regime, Echo State Network.
- **How.** `IDEAS_LOG.md` E.19–22 / `EXPERIMENT_BACKLOG.md` W2-7; reservoir redirected to magnitude.
- **STATUS.** **not yet run** (backlog).

---

## Family 5 — Sequence / deep models

### 5.1 1D-CNN / GRU on raw tick or 1s path
- **What.** Conv/recurrent net over the last-W-seconds raw microstructure path (1s returns, OBI, microprice deviation, signed return) instead of hand-crafted summaries.
- **How.** `m_cnn.py` (`HS`,`W` args, seconds), `min1_v14_cnn.py` (60s), `exp_seq.py` (5m GRU). CPU, subsampled.
- **Why.** Is the ceiling the data or the model?
- **STATUS.** **null** — valAUC plateaus 0.49–0.525 ≈ GBM; confirms the ceiling is the DATA.

### 5.2 Neural-CDE on irregular tick path (Δt as control)
- **What.** Neural Controlled Differential Equation integrating `dz=f(z)dX` over the raw per-tick event path; the irregular inter-arrival Δt is encoded as ch0 (cumulative seconds) — the genuinely new channel every prior model discarded.
- **How.** `min1_ncde.py`. Critical ablation: ARM A irregular Δt vs ARM B constant-grid (same arch/seed). Knobs: 40-tick window, hidden=24, CPU. Falls back to GRU if torchcde missing (reports which).
- **Why.** Arrival-time-conditioned-on-sign is not sign-permutation-invariant → could survive the sign-invariance theorem.
- **STATUS.** **null** — irregular VAL AUC 0.498 ≤ grid 0.509; per-year 0.490–0.507, no CI clears (`min1_ncde_results.json`). Clean first-in-world negative.

### 5.3 Path-signature (Lévy area) features
- **What.** Level-2 signature antisymmetric part = Lévy area `A(X,Y)=0.5∮(X dY−Y dX)` = signed lead-lag rotation (does order flow lead price), causal over trailing windows.
- **How.** `m30_sig.py` (arg HS; run 1800 and 5). Channels: price-return, OFI imbalance, microprice deviation.
- **Why.** Genuinely sign-aware, not captured by hand-crafted feats or CNN.
- **STATUS.** **null at 30m** (AUC 0.51); **top-5 feature at HS=5s** (the seconds edge) — direction is a seconds phenomenon.

### 5.4 TS foundation models (Kronos zero-shot) — backlog
- **What.** Generative finance-native foundation model, zero-shot P(up)=frac(sampled close>open).
- **How.** `EXPERIMENT_BACKLOG.md` W2-5 (needs HF weights).
- **STATUS.** **not yet run** (backlog, prior 8–12%).

---

## Family 6 — Reinforcement learning

### 6.1 DQN direction-with-abstain + IQN/CVaR magnitude-gated abstain
- **What.** (A) DQN MLP Q-net with actions {LONG, SHORT, ABSTAIN}, deriv-settlement reward (correct +R, wrong −1, abstain 0, ties lose). (B) Implicit Quantile Network over the signed-return distribution: does quantile SKEW carry sign (theorem says no), and does a CVaR/spread abstain gate improve the magnitude book.
- **How.** `min1_drl.py`. Reuses `min1_production` settlement/splits/features verbatim. CPU torch, subsample ≤50k. Falsifier: traded-subset CI95-lower >0.65 each year at ≥5% cov (expected FAIL).
- **Why.** RL learns a POLICY (timing/sizing/abstain), not per-trade accuracy. Genuine value is only IQN+CVaR sizing on the magnitude book.
- **Process notes.** Bounded by the same information ceiling (the online-ARF control already proves adaptation recovers nothing; literature: RL converges to buy-and-hold).
- **STATUS.** **null** — DQN committed acc 0.482/0.484/0.483; IQN skew→dir 0.481/0.487/0.481; IQN+CVaR 0.489/0.494/0.496 (`_drl_run.log`). Below 0.50; quantile skew carries no sign.

### 6.2 Online concept-drift adaptive (river ARF + ADWIN)
- **What.** Adaptive Random Forest with ADWIN drift detector, trained PREQUENTIALLY (predict bar t, THEN learn x_t,y_t) in strict time order. The control proving whether the 2025 wall is stale-model drift vs genuine efficiency.
- **How.** `min1_online.py`. Warm-up 2022–23 (learn only); compact feature subset + stride for throughput. Capped 5-tree and uncapped 10-tree variants.
- **Why.** If an adaptive model still hits ~0.50, the wall is efficiency, not drift.
- **STATUS.** **null (KEYSTONE CONTROL)** — AUC 0.503–0.508 every window ⇒ **the 2025 wall is genuine efficiency** (`models/min1_online_summary.json`).

---

## Family 7 — Microstructure / order-flow

### 7.1 Tick microstructure ensemble (seconds–minutes)
- **What.** GBM ensemble over ~20–62 1s quote/microstructure features (OBI, microprice, spread, flow) at wall-clock expiry.
- **How.** `min1_production.py`/`min2_production.py` (`HS`/`GAP`), `m{5,30}_tick.py`, seconds frontier `tickhz.py` (authoritative), `tickmodel*.py`, `tick_ensemble.py`, production `m_tick_prod.py`. Cache builder `tick1s_cache.py`.
- **Why.** Microstructure carries the real directional edge at the 1–5s scale.
- **Process notes (LEAKAGE TRAP).** `tick1s_cache.py` drops empty seconds → `mid.shift(-HS)` shifts *bars* not *seconds* (median ~111s for a "60s" trade). Fix = `wc_ret()` wall-clock expiry. This + greedy de-overlap caused the program's biggest inflation.
- **STATUS.** **survived ~0.65 at 1–5s** (`tickhz.py`, `m_tick_prod.py` 3s 0.657/0.667 — untradeable on deriv's 15m floor); **null by 60s** (0.55) and ≥5m.

### 7.2 Raw order-book imbalance decay
- **What.** True bid/ask SIZE imbalance vs horizon — the documented short-horizon signal's decay curve.
- **How.** `rawtick_decay.py`, `rawtick_probe.py`.
- **STATUS.** **mechanistic** — next-tick 0.553 → 1min 0.501; the proof of the sub-minute wall.

### 7.3 CKS event order-flow imbalance (Cont-Kukanov-Stoikov)
- **What.** Proper price-event OFI: `e_n` conditions bid/ask volumes on the direction of the best-bid/ask price *move*, replacing tick-rule/static-size OFI. Aggregated to 1s, summed over 5/15/30/60s.
- **How.** `min1_cksofi.py` (`build` → `features_tick_cks/`, then `run`). Two arms: standalone 60s LGBM + swap-in to replace `imb` in the min1 reversion book. Firewall: missing second → 0.0, never ffill; moved-bars-only.
- **STATUS.** **KILLED** — standalone VAL dirAUC 0.4993, swap doesn't lift 2025 (`min1_cksofi_result.json` `FALSIFIER_KILLED: true`).

### 7.4 Cross-impact OFI matrix (Cont-Cucuringu-Zhang)
- **What.** Signed CKS event-OFI for all 7 majors expressed in EUR-equivalent USD-direction (keep XXXUSD sign, flip USDXXX), as a `[7-pair × {0,1,2,5s lag}]` block → one LGBM on EURUSD next-60s sign. Tests whether USD-wide informed flow appears first in another leg.
- **How.** `min1_xofi.py`. 24 cross-pair CKS 1s caches (`features_tick_xofi/`); EURUSD-own clock, missing-second → 0.0 OFI. Kill if VAL dirAUC ≤0.515 or no window CI-lower >0.515 or off-diagonal sign-flip 2024↔2026. Build is multi-hour (24 caches from raw ticks); idempotent (skips existing caches) so safe to resume after interruption.
- **STATUS.** **KILLED** — VAL dirAUC 0.5015; the LGBM put **78.9% of gain-importance off-diagonal** (on the other legs' flow) yet stayed coin-flip; no off-diagonal sign-flip (`min1_xofi_result.json` `FALSIFIER_KILLED: true`). Cleanest possible null: forced onto cross-pair flow, still nothing.

### 7.5 Per-side raw bid/ask signed order flow
- **What.** Per-1s net per-side volume + tick-rule signed volume + flow imbalance over 5/15/30/60s; next-60s sign on moved bars (clean EURUSD-only).
- **How.** `_adj_perside_flow.py`, `orderflow.py` (proxy 10s tick-rule), kernel input in `min1_kernel.py` (`flow`).
- **Process notes.** Caught a 7-pair ffill-flat-window false-positive AUC 0.728 — the recurring mirage tripwire.
- **STATUS.** **null** — moved-bar dirAUC 0.5086, CI-upper never >~0.52.

### 7.6 Kernel-SVM model class (Fletcher 2010)
- **What.** RBF kernel via Nyström feature map + SGD log-loss (scalable SVC surrogate) over Fletcher microstructure feats + signed per-side flow.
- **How.** `min1_kernel.py run` (H∈{15,30,60}s). One split at a time; cache flow; train ≤80k.
- **STATUS.** **null** — VAL AUC 0.502, train in-sample 0.529 (can't even memorize) ⇒ data, not model, is the ceiling (`models/min1_kernel_summary.json`).

### 7.7 Ordinal time-irreversibility (Neuman-Cohen-Tamir)
- **What.** Causal Bandt-Pompe ordinal-pattern arrow-of-time: (a) transition-asymmetry `I_W` (sign-invariant magnitude), (b) **signed** ascending-vs-descending transition imbalance (the sign-aware part the theorem does NOT forbid), (c) Pomeau 3rd-moment time-asymmetry. LGBM on 60s direction.
- **How.** `min1_irrev.py`. W∈{15,30,60,90} bars, d∈{3,4}; reports irrev-alone VAL AUC.
- **Why.** Highest-prior genuinely-new DIRECTION-specific shot (sign-invariance does not kill ordinal irreversibility).
- **STATUS.** **KILLED** — VAL AUC 0.503 alone / 0.5027 blend; per-year 0.479/0.488/0.489, neither clears breakeven (`min1_irrev_result.json` `clears_breakeven_all3: false`).

### 7.8 Macro-release directional impulse (news)
- **What.** Predict `sign(eurusd_signal)` from a macro surprise in the post-release window (no fitting); the 60s impulse timescale.
- **How.** `min1_news60.py`, `m5_news.py`/`m5_news_model.py`; calendar `macro_calendar.parquet` via `fetch_calendar.py`+`event_signs.py` (FXStreet). Selective by vol tier × |surprise z|.
- **STATUS.** **null/sign-unstable** — bigger surprise = MORE wrong OOS (HIGH-vol&|z|≥1: 0.167/0.364/0.000); FX prices the surprise in <60s. News = magnitude, not direction.

### 7.9 Up/down side asymmetry — filter vs specialist
- **What.** Split predictions by predicted side; exploit a one-sided edge as a trade FILTER on the symmetric model (a separately-trained side specialist is the failure control).
- **How.** `min1_updown.py` (diagnostic, inference-only on frozen child), `min1_upspec.py` (dedicated specialists), `_adj_perside_flow.py`.
- **STATUS.** **helped (filter) / null (specialist)** — 60s up/dip-buy carries (2025 .584/2026 .613), down dead; separately-trained specialist WORSE (subset-training kills ranking); regime-dependent. See `EURUSD_RESULTS.md` leaderboard.

### 7.10 Hawkes / transfer-entropy / VPIN (backlog)
- **What.** Self-exciting up-tick vs down-tick arrival intensity (Hawkes), directed info flow volume→price (transfer entropy), order-flow toxicity (VPIN).
- **How.** `IDEAS_LOG.md` B.9, D.16/18. Need trade-signed/depth-resolved LOB (on-disk feed is indicative quote, no trade signs).
- **STATUS.** **not run** — blocked on data.

---

## Family 8 — Magnitude / volatility (the one certified edge)

### 8.1 Magnitude (|return| ≥ Q) classifier
- **What.** A 2nd model predicts P(|ret_H| ≥ train-Q75) — the large-move probability. **The genuine forecastable edge** at every horizon (sign-invariance theorem arXiv:2512.15720: entropy/order-flow/complexity gate move SIZE, not sign).
- **How.** `m30_magnitude.py` (rv30+PE → |ret30|), `m10_magdir.py` (10m), `min1_v10.py`/`_redteam_magdir60.py` (60s), `min2_v1.py` (120s). Retarget: set the |ret_H| threshold + horizon; primary feature = realized vol (diurnal-deseasonalized RV / signed semivariance are the documented upgrades).
- **Why.** Volatility is forecastable; sign is ~EMH. Tradeable on Touch/Range/Straddle/VRP, NOT up/down.
- **Process notes.** Realized vol is the real predictor; permutation entropy is NULL for magnitude on FX (corr ~0.01).
- **STATUS.** **survived (BEST)** — 30m large-move AUC 0.73–0.78, 10m 0.706–0.813, 60s 0.787 vs dirAUC 0.510 (`magnitude_verified.json`; `MAGNITUDE_FINDINGS.md`). CPCV-certified (see 9.1).

### 8.2 Direction-conditioned-on-magnitude (sign-invariance test)
- **What.** Does direction become predictable on bars where the magnitude model predicts a big move.
- **How.** `m10_magdir.py`, `_redteam_magdir60.py`, `min1_v11.py`/`v13.py`.
- **STATUS.** **null (confirms theorem)** — direction flat ~0.51–0.53 across all magnitude quartiles.

### 8.3 Complexity / predictability gates (PE / Hurst / RQA / Lempel-Ziv)
- **What.** Permutation entropy (Bandt-Pompe d=3/4), lag-1 autocorr, variance-ratio Hurst, (backlog: RQA-DET, sample/multiscale entropy, Lyapunov, 0-1 chaos) as "bet only in low-complexity/predictable bins".
- **How.** `m30_complexity.py`; full catalog `IDEAS_LOG.md` A.1–8.
- **STATUS.** **null for direction** — conditional accuracy FLAT across all bins (~0.515); these gate magnitude, not sign.

### 8.4 Non-time information bars (volume / dollar / imbalance)
- **What.** Sample by information arrival (cumulative volume/dollar) instead of the clock; predict 30m time-forward direction (López de Prado AFML ch2).
- **How.** `vbars.py` (`build` / `model V` / `model D`).
- **STATUS.** **null for direction** — AUC 0.509–0.516; improves return normality, not directional AUC.

---

## Family 9 — Validation methodology

### 9.1 Combinatorial Purged CV + Deflated-Sharpe / PBO
- **What.** Replace the single chronological partition with C(N,k) purged+embargoed OOS paths → a DISTRIBUTION; a model whose 10th-percentile path clears the bar is real, one clearing only on the mean/lucky-split is a mirage. Plus PBO and deflation by N_trials given corr(VAL,OOS)=−0.54.
- **How.** `cpcv_certify.py` (generic, N=8/k=2/28 paths/70 trials/embargo=1 horizon), `min15_cpcv.py` (faithful CPCV of the actual 3-model m15 book, N=6/k=2/15 paths). Targets: 15m direction + 30m magnitude.
- **Why.** Certify which headline numbers survive trial-deflation.
- **Process notes.** `cpcv_certify.py` used a single LGBM + q33 + all-era pool (a *weaker* re-implementation) → its lower 15m number is NOT a faithful test; `min15_cpcv.py` replicates the real ensemble + VAL-tuned gate + recent-held-out and reproduces the edge. Memory: one parquet at a time, subsample ≤100k.
- **STATUS.** **survived (faithful)** — `min15_cpcv_result.json` mean 0.5787 / p10 0.5573 (15 paths); magnitude 30m AUC mean 0.744 / p10 0.718 / 5.0× decile lift survives (`cpcv_certify_result.json`). The generic 15m selacc (mean 0.545) is the deflated weaker re-implementation, not a refutation.

### 9.2 Honest deflation / pre-committed proof / multiple-testing audit
- **What.** Pick a single best pocket on VAL only, confirm on TEST, judge OOS once + 5000× bootstrap; measure corr(VAL_acc, OOS_acc); pre-commit the entire pipeline before judging.
- **How.** `exp_15m_v8_validate.py` (established corr=−0.54), `exp_15m_v13_proof.py` (pre-committed 0.642), `m30_lab.py`, `exp_15m_v10_eurchk.py` (pooled-dilution reconciliation).
- **STATUS.** **keystone** — corr(VAL,OOS)=−0.54 is the discipline anchor for the whole program.

### 9.3 Walk-forward retraining (1-yr gap)
- **What.** Annual expanding-window retrain to isolate regime-shift vs stale-model gap.
- **How.** `m5_walkforward.py`, `m10_walkforward.py`, `m15_walkforward.py`, `m5_wf_stack.py`.
- **STATUS.** **helped marginally / made 15m WORSE** — lifts the binding window only +0.01–0.02; 15m walk-forward 2024 .543/2025 .536/2026 .604 all BELOW frozen (`m15_walkforward_result.json`) ⇒ frozen m15 is optimal, 2025 not fixable by adaptation.

### 9.4 Leakage / bias-audit release blocker + adversarial verifiers
- **What.** 6 Tier-1 invariants (feature causality, true wall-clock expiry label, split disjointness, non-overlap independence, m15/m30 checks); exits non-zero on any FAIL. Plus greedy-vs-random de-overlap and threshold-sweep verifiers.
- **How.** `audit_leakage.py` (pre-deploy guard), `_verify_greedy_bias.py`, `_verify_coverage.py`, `_thr_sweep_audit.py`, `_verify_cache_preds.py`.
- **STATUS.** **tooling** — encodes the 2026-05 bias-audit; verifies min1 0.539/0.550, min2 0.528/0.539, m15 0.647.

### 9.5 Sample-uniqueness weighting / sequential bootstrap (backlog)
- **What.** `sample_weight = 1/concurrency` from the fixed label horizon to de-weight overlapping labels (de Prado).
- **How.** `EXPERIMENT_BACKLOG.md` #8.
- **STATUS.** **not yet run** (backlog; tighter CI, not new signal).

---

## Cross-cutting leakage traps (apply to every family)
1. **Sequence-model future-peeking** — HMM Viterbi/forward-backward, Kalman RTS smoother. Use the forward FILTER only.
2. **ffill-flat-window mirage** — intersecting pairs + ffill manufactures ~50% fake-flat bars → fake AUC 0.7+ collapsing to ~0.49 on moved bars. Caught repeatedly. Always: own-clock, missing→0.0 not ffill, eval moved-bars-only, check up-rate ∈[0.47,0.53].
3. **Bar-shift horizon mislabel** — `shift(-N)` on gap-dropped tick data shifts *bars* not *seconds*. Use `wc_ret`.
4. **Greedy-by-confidence de-overlap** — peeking to keep the most-confident bar; −3 to −9 acc points. Use `nonoverlap_chrono`.
5. **VAL-acc-max selection** — corr(VAL,OOS)=−0.54 anti-transfers. Select by worst-VAL-half.
6. **Thin-coverage mirages** — any n<25–50 pocket at 0.65+ is multiple-testing noise.
7. **Ties LOSE** — a ~0.50-AUC model's realized win-rate sits BELOW 0.50 once ties are charged.

---

## Program-level status summary
- **Survived/tradeable:** 15m compression×NY direction **0.647** (CPCV-faithful ~0.58 p10); 30m 0.591; seconds 1–5s tick ~0.65 (needs tick venue); **magnitude AUC 0.73–0.79 everywhere** (the one CPCV-deflation-certified edge).
- **Killed/null with recorded results:** CKS-OFI, cross-impact OFI, **CCM coupling-gate**, ordinal irreversibility, residualized target, Neural-CDE (all have result JSONs); HMM/Kalman/kernel/online/per-side-flow/news/RMT/DRL (ledger + `models/*_summary.json` + logs).
- **Coded / backlog:** foundation models (Kronos), SSA/fractional-diff/particle-filter/reservoir, Hawkes/TE/VPIN (LOB-blocked), sample-uniqueness weighting, magnitude upgrades (deseasonalized RV / signed semivariance). **The only inputs that could change the directional answer are external:** intraday DE–US 2y rate differential, daily implied-vol / risk-reversal, EURGBP ticks.

Files of record: `EURUSD_RESULTS.md` (per-horizon results + up/down leaderboard), `EXPERIMENT_LEDGER.md` (135-experiment master table), `MAGNITUDE_FINDINGS.md`, `DIRECTION_FINDINGS.md`, `CCM_DESIGN.md`, `EXPERIMENT_BACKLOG.md`/`IDEAS_LOG.md` (backlog), and the per-result JSONs cited per entry.
