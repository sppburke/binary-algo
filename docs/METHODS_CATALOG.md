> **SCOPE: GENERIC** (currency/timeframe-agnostic). Per-key numbers, if cited, are tagged [PAIR·tf] examples whose record-of-truth is the Tier-2 file. See REPO_MAP.md.

# EURUSD Binary-Direction Research — MASTER METHODOLOGY CATALOG

> **Full literature lever inventory:** `docs/CORPUS_LEVER_INVENTORY.md` documents all 550 testable levers mined from the academic corpus (methodology · 5m test-plan · sign-invariance · status · falsifier). This catalog holds the 9 method FAMILIES; the inventory holds every individual lever + its tested-status.


**Technique-centric, timeframe-agnostic.** Every methodology used in this program gets a self-contained entry: *what it is · how to use it · why to use it · process notes (leakage traps / discipline) · status*. Each entry cites its implementing file and points STATUS at where the verified result lives. For per-horizon **results** and the **up/down leaderboard**, see the companion `results/EURUSD_RESULTS.md`.

All scripts live in `/home/sean/git/binary-algo/`. Last major campaign update 2026-06-08 (**Neural-forecaster + spectral DIRECTION sweep** from mining external repos `directional-prediction`/`n-hits`: N-BEATS/N-HiTS path→sign, DLinear/Autoformer/FEDformer/TFT-quantile, causal DWT/SSA band-split — run at ALL 6 EURUSD tf single+cross-pair on GPU, deriv-faithful forward-holdout; **84 arms, 0 survivors, ALL KILLED** (valAUC ≤.519, no held-out CI95-lo ≥.541); flips `docs/CORPUS_LEVER_INVENTORY.md` 333/356/357/359 UNTESTED→killed; `nbeats_nhits_dir.py`/`decomp_dir.py`/`spectral_dir.py` + `neural_spectral_dir_sweep_result.json`; full record `results/EURUSD_RESULTS.md`/`docs/DIRECTION_FINDINGS.md` §2026-06-08). Prior 2026-06-07 (Phase 4 cross-sectional DIRECTION campaign: D1 lead-lag signature + D6 HAVOK both KILLED; D7 signed-semivariance REAL-but-sub-breakeven — entries 3.7/3.8/4.6/9.7).

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
- **STATUS.** 5m null ~0.52 AUC / book 0.583 (`m5_xpair_production.py`); 15m **survived 0.647** combined (`m15_production.py`, `models/m15_EURUSD_strategy.json`); 30m 0.591 (`m30_production.py`); 10m honest 0.602 combined (`m10_freeze_honest.py`), **base-book side-split refit-CPCV certifies both sides** (UP p10 .561 / DOWN .552, `m10_cpcv_side.py`) but is BEATEN by the cross-pair book (.586/.568) — see Family 3.3.

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

### 2.5 DST-correct session segmentation (NY / London / Asian)
- **What.** A generic conditioning/regime method: restrict decision rows to a single trading session, defined by **LOCAL exchange-tz wall-clock hour** so the boundary is correct across dayl-savings transitions (not a fixed-UTC window). Sessions: **NY** 08–17 `America/New_York`, **London** 08–16 `Europe/London`, **Asian** 09–18 `Asia/Tokyo`. Re-tests every freq×method per session to find where an edge concentrates.
- **How.** `sessions.py` (source of truth) — `session_mask` computed per-day from the local-tz hour and applied across train+val+test+oos; `SESSIONS` registry. Per-substrate drivers: `session_1m.py` / `session_2m.py` (tick GBM per session), `session_bars.py` (bar GBM, any `H`), `session_xpair.py` (the certified cross-pair book run STRICT session-only, any `H`, with per-H gate `{2:1m_bb_width, 5/10:5m_bb_width, 15:15m_bb_width, 30:1h_bb_width}`), and `barcnn_run.py` gained a `SESSION` arg (per-session bar-image CNN). Files of record `session_{1m,2m,5m,10m,30m,xpair}_*_{ny,ldn,asia}_result.json`.
- **Why.** Liquidity, participant mix, and which macro cross-section is live all vary by session; an edge averaged over 24h can be diluted by dead hours. DST-correct boundaries matter because a fixed-UTC window drifts ±1h across the year and smears the session.
- **Process notes (discipline).** The session filter restricts **DECISION ROWS** in every split, NOT the feature computation: GBM / cross-pair use causal-continuous features (rows-only filter, features still see the full stream). Kronos FT uses strict session-only INPUT (filter-before-window + contiguity guard) since its context must be a contiguous in-session candle stream. Apply the same selection discipline (worst-VAL-half, per-year CI95, CPCV p10) per session.
- **STATUS.** **DIRECTION edge is decisively NY-CONCENTRATED on the cross-pair book; null elsewhere.** Cross-pair (`session_xpair.py`, the certified ≥10m lever) certifies BOTH sides at every horizon **2m→30m in NY only** (2m NY .564/.560, 5m .596/.588, 10m NY UP .6053/DOWN .5896 15/15; 15m NY .5845/.5712; 30m NY .5681/.5639) and at NONE in London/Asia (.49–.53); NY even BEATS the legacy fixed-UTC gate (10m legacy .586/.568, 30m .559/.553). The 2m & 5m NY two-sided certs are NEW (legacy EURUSD 2m was dead; legacy 5m UP-only at .553). Base-bar GBM direction KILLED all sessions (NY strongest, e.g. 10m NY p10 .525; 15m NY .520/LDN .515/Asia .512); 1m/2m tick GBM direction null all sessions. **MAGNITUDE certified in ALL sessions** at every freq (1m magAUC NY .675/LDN .728/Asia .717, p10 .58–.71; 5/10/30m p10 .72–.80). See the dated 2026-06-06 section below. **Session re-campaign COMPLETE — all freq×method×session cells run.**

---

## Family 3 — Cross-horizon & cross-pair information

### 3.1 Cross-horizon STACK (parent → child front-load)
- **What.** Front-load a longer-horizon parent ensemble's confident direction into the shorter outcome, gated by a learned meta-labeler P(parent correct on child horizon). **The single strongest method in the program.**
- **How.** `m5_stack2.py` (soft, strongest 5m), `m5_stack.py` (hard agreement), `min1_stack.py`, `m10_stack.py`, `m5_wf_stack.py` (walk-forward), `m5_xhorizon.py` (raw transfer probe). Knobs: child label horizon (`MX_HOR`), parent ensemble(s) from `models/m{5,10,15}_*`, meta threshold (worst-VAL-half).
- **Why.** Borrow the cleaner longer-horizon signal to beat a horizon's own noise floor.
- **Process notes.** Child asymptotes to the parent's native ceiling; front-loads weakly when child ≪ parent (60s off a 5m/15m parent ≈ 0.51). Hard agreement starves OOS coverage (n16).
- **STATUS.** **helped/best** at 5m (0.613 verifiable / 0.648 thin, `m5stack_EURUSD_strategy.json`); **null** at 60s (0.586) and 10m.
- **VIABILITY CORR-GATE (cheap pre-check before building the full stack).** Compute `corr(p_parent, p_child)` on VAL at the operating session/horizon. corr>~.90 ⇒ collinear, the parent is the same own-pair GBM at a coarser horizon ⇒ no orthogonal sign ⇒ SUBSUMED ([USDJPY·15m] 30m parent corr .957). corr≪.90 is necessary but **NOT sufficient**: a decorrelated parent helps only if it ALSO carries information the child lacks (higher/complementary AUC); a decorrelated-but-equally-weak parent (≈equal AUC) just reduces variance like a seed-ensemble and is REDUNDANT with seed-ens — path-means stay flat, only the p10 order-statistic moves ([AUDUSD·15m] parent corr .72, AUC ≈ child, blend non-additive vs seed-ens). See `sweeps/AUDUSD_15m_backlog.md`.

### 3.2 Cross-pair USD-residual / common-factor / lead-lag
- **What.** Decompose EURUSD into EUR-strength − USD-strength using a sign-aligned basket of the 6 other majors; the relative-value reversion residual, per-pair lead-lag residual, catch-up residual, dispersion/agreement.
- **How.** `m5_xpair.py` (env `MX_HOR`; modes `xp`/`xpbase`/`xpof`), probe `m5_xpair_probe.py`, merger `crosspair.py`, production `m5_xpair_production.py`. `augment(...,"xpof")` adds 239 base + 18 order-flow cols.
- **Why.** The one orthogonal family **sign-stable across 2024 & 2026** (momentum agreement is dead/sign-blind).
- **Process notes.** **HORIZON-GATED**: the slow USD-common-factor needs ≥5m to be exploitable, so the certification gradient is **none@60s → UP@5m → BOTH@10m & 15m** and does NOT extend back below 5m. The informed/jump component of moves (esp. DOWN) averages out as the horizon lengthens, so the common-factor SIGN becomes forecastable at ≥10m. CONCURRENT cross-pair (windows ending at t) is the carrier; strictly-LAGGED lead-lag is dominated ([EURUSD·10m] `m10_leadlag` VAL AUC .517). ALWAYS verify a frozen-VAL gate positive under refit-CPCV — at 2m a frozen gate showed .61 that dissolved to ~.51 per-fold (overfit-gate).
- **STATUS.** **CERTIFIED via refit-CPCV** at 5m-UP (`EURUSD.m5xp.v1`, p10 .553), BOTH **10m** sides (`EURUSD.m10xp.v1`, UP p10 **.5863** / DOWN p10 **.5683**, 15/15 paths, `m10_xpair_cpcv.py` — improves base-book side floors .561/.552), and BOTH **15m** sides (`EURUSD.m15xp.v1`, UP .5673/DOWN .5742, 15/15 paths — program's best DOWN edge). At 10m, 11 improve/discover levers ALL failed to beat the gated raw cross-pair sign → edge is info-bound by the 2025 regime (same as 15m). **NULL below 5m:** 2m KILLED under ties-strict refit-CPCV (UP p10 .5096/DOWN .5124, `min2_xpair_cpcv.py`), 60s ~0.52. Combined-book 5m 0.586. **10m < deriv 15m forex min → research horizon; deployable sibling = m15xp.**

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

### 3.7 Depth-2 lead-lag SIGNATURE cross-terms / Lévy area (pure-numpy) — TESTED, KILLED both horizons (D1)
- **What.** The cross-pair / cross-sectional version of the path-signature idea (cf. the single-pair seconds Lévy area, Family 5.3): for EURUSD and each of the 6 peers, compute the level-2 iterated integrals `S^{ij}=∫∫ dX^i dX^j` and the antisymmetric Lévy area `A^{ij}=½(S^{ij}−S^{ji})` over a trailing window, which encode SIGNED lead-lag and quadratic covariation between the legs — a sign-aware cross-section feature feeding the certified direction book.
- **How.** `xsec_direction.py` arm **D1 sig** (unified cross-sectional harness on the certified 7-pair panel). Depth-2 lead-lag signature between EURUSD and each peer over a trailing 30-bar window; **iisignature can't compile (no `Python.h`) → computed in PURE NUMPY via cumsum** (dependency-free). **Forward holdout** (Family 9.6), direction NY cov0.10, arms {base, base+fam, famonly, fam_shuffle}. Falsifier: must beat book p10 CI95 AND a lead/lag-shuffle MUST degrade it (mechanism-specificity control).
- **Why.** Lead-lag rotation between currency legs is signed structure the concurrent cross-pair features (Family 3.2) and the GBM do not directly encode; a genuinely sign-aware probe of whether one leg's move leads EURUSD's next move.
- **Process notes (leakage / discipline).** Causal trailing windows only; the lead/lag-shuffle (rotation surrogate) is the pre-registered mechanism-specificity control (if the edge survives shuffling the lead-lag ordering, it is not actually lead-lag). The pure-numpy cumsum implementation is the reusable artifact (no `iisignature`/`sigkernel`/GPU needed). **DURABLE METHODOLOGICAL POINT:** this rotation/sign-flip shuffle control is exactly what separated the genuine D7 signed-semivariance signal (degrades under its sign-flip shuffle → real, Family 3.8) from this spurious D1 signature (does NOT degrade under the rotation shuffle → not actually lead-lag). The shuffle null is the load-bearing falsifier across the cross-sectional direction campaign.
- **STATUS.** **TESTED — KILLED at BOTH 15m and 30m.** `xsec_direction.py` which=sig, `xsec_direction_sig_{15,30}m_result.json`. The certified base book DECAYS forward (15m .5895/.5518/.5234; 30m .5827/.5505/.5055 — the documented refit-dependence) and the signature adds nothing. 15m: +sig DECAYS (Δ −.0043/−.0099/+.0032), sigonly pooled **.5243 ≈ sig_shuf .5236** → **NOT genuine lead-lag — the rotation surrogate does NOT degrade it, so it FAILS the mechanism-specificity falsifier.** 30m: +sig decays all 3 years (−.0094/−.0043/−.0133), sigonly pooled .5283 (sub-base). Lead-lag signature content is already saturated in the GBM's `ll_<pair>k` features and carries no orthogonal sign. `docs/CAMPAIGN_2026-06-07_FACTS.md` (Phase 4). (HAVOK / Hankel-Koopman is the sibling D6 arm in the same harness — see Family 4.6; signed-semivariance D7 — the ONLY arm to clear its shuffle null — is Family 3.8.)

### 3.8 Realized signed-SEMIVARIANCE direction (Patton-Sheppard good/bad vol) — TESTED, REAL-but-SUB-BREAKEVEN (D7)
- **What.** The directional realized-measure: decompose each pair's realized variance into up- and down-semivariance `RS⁺ = Σ r²·1{r>0}` and `RS⁻ = Σ r²·1{r<0}` (Patton-Sheppard "good vs bad volatility") over a trailing window and feed the SIGNED asymmetry `RS⁺−RS⁻` as a directional feature behind the certified cross-pair book. The cross-sectional version: per eu-equiv pair RS⁺/RS⁻/signed-asymmetry across the 7-pair USD panel, PLUS a USD-common-factor up-vs-down semivariance asymmetry. Distinct from the magnitude semivariance arm (Family 8.5, where RS± is a SIZE feature) — here the signed asymmetry is used as a SIGN feature, which the magnitude sign-invariance theorem does not forbid.
- **How.** `xsec_direction.py` which=semivar (unified cross-sectional harness on the certified 7-pair panel). Builds RS⁺/RS⁻/signed-asymmetry per eu-equiv pair + the USD-factor up/down asymmetry; **forward holdout** (Family 9.6), direction NY cov0.10, arms {base, base+semivar, semivaronly, **semivar_shuffle = SIGN-FLIP control**}. Falsifier / mechanism-specificity test: `semivaronly` must beat `semivar_shuf` (if randomly flipping the sign of the asymmetry does not degrade it, the directional content is spurious) AND it must clear breakeven .541. `xsec_direction_semivar_{15,30}m_result.json`.
- **Why.** Down-moves and up-moves of a currency carry genuinely different information (informed selling vs buying, leverage/risk-off asymmetry); the SIGNED semivariance asymmetry is sign-aware structure the concurrent cross-pair features and the GBM do not directly encode — a direct probe of whether good/bad-vol asymmetry forecasts the next directional move.
- **Process notes (leakage / discipline).** Causal trailing windows only; the **sign-flip shuffle is the pre-registered mechanism-specificity control** and is the distinguishing methodological point of the whole Phase-4 campaign: D7 is the ONLY cross-sectional direction arm whose signal SURVIVES its shuffle null, which is precisely what separates it from the spurious D1 signature (Family 3.7, whose rotation surrogate does NOT degrade it). A signal that clears its shuffle null is GENUINE directional content even when it is too weak to deploy — the methodologically honest "real-but-sub-bar" verdict, the direction analog of the magnitude HAR finding (Family 8.5).
- **STATUS.** **TESTED — REAL-but-SUB-BREAKEVEN (NOT null-on-mechanism like D1/D6); KILLED as deployable.** This is the ONLY direction shot in the campaign that PASSES its mechanism null: `semivaronly` beats `semivar_shuf` (sign-flip control) by **+.015 (15m) / +.030 (30m)** → the signed-vol asymmetry GENUINELY carries DIRECTIONAL content. BUT it is weak: `semivaronly` pooled **.5258 (15m) / .5382 (30m)** — below breakeven **.541** (only 2024@30m .5516 clears it), DECAYS forward, and `+semivar` does NOT add to the base book (decays vs base both horizons: Δ15m −.0154/−.0029/−.0107, Δ30m −.0081/+.0062/−.0139). The DIRECTION analog of magnitude's "real-but-sub-bar" (Family 8.5): genuine sign content exists, but it is sub-deployable and already subsumed by the cross-pair book. **Nothing certified; UP/DOWN leaderboard unchanged.** `xsec_direction_semivar_{15,30}m_result.json`, `docs/CAMPAIGN_2026-06-07_FACTS.md` (Phase 4). (Siblings in the same harness: D1 signature Family 3.7 — spurious; D6 HAVOK Family 4.6 — null-on-mechanism.)

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
- **How.** `min1_ccm.py` (full design in `docs/CCM_DESIGN.md`). E=4, τ=1, E+1 neighbors, mandatory Theiler window ±(E·τ+tp), L∈{50…Lmax}; lagged-CCM forward tp*∈[1,60]s required (Ye 2015); Ebisuzaki + twin surrogates; θ_gate by VAL worst-half CI95-lower ≥0.515. Self-coupling sanity gate runs FIRST. `python min1_ccm.py selfcheck`.
- **Why.** High driver→EUR coupling ⇒ dynamics more driver-led/deterministic ⇒ frozen sign model more accurate there.
- **Process notes (FIREWALL).** Fixed 1s grid built ONLY for embedding (returns/OFI summed, empty second → 0.0, NEVER ffill); labels/trades/accuracy stay on the native clock via `wc_ret`. 6 pre-registered falsifiers. ~40 min, <1.5GB. **Implementation gotcha (fixed):** library neighbors near a window's end overflow `X[cand+tp]` for tp>0 — restrict the pool to `base+tp<len(X)` for BOTH targets and library; vectorize the per-target loop (the python loop is hours). Validate the core on a synthetic coupled-logistic (Y-xmap-X converges, reverse flat, surrogate→0) at the SAME tp you'll run.
- **STATUS.** **KILLED** — `min1_ccm_result.json` `FALSIFIER_KILLED:true` (4 conditions). Self-coupling sanity PASSED (EUR-own-OFI has the strongest convergence, slope 0.0066 — embedding is correct, null trustworthy), yet **even EURUSD's own order flow does not convergently cross-map its own 60s return** (slope 0.0066 ≤ 0.02, Δρ 0.036 < 0.05, surrogate-pass 0.44). All 7 drivers conv=False/surr=False; gated per-year 2024≈0.51 / 2025≈0.503 / 2026≈0.52. A dynamical-systems-lens confirmation of 60s efficiency, independent of the ML channels.

### 4.4 Singular-spectrum / fractional-diff / particle-filter / reservoir (backlog)
- **What.** SSA causal decomposition, fractional differentiation (stationary memory-preserving), particle-filter latent regime, Echo State Network.
- **How.** `docs/IDEAS_LOG.md` E.19–22 / `docs/EXPERIMENT_BACKLOG.md` W2-7; reservoir redirected to magnitude.
- **STATUS.** **not yet run** (backlog). NOTE: fractional differentiation is now BUILT + TESTED — see 4.5 (this backlog row covers only SSA / particle-filter / reservoir).

### 4.5 Fractional differentiation / FFD ("stationarity with memory")
- **What.** Fixed-width fractional differencing (de Prado AFML ch5): apply the fractional-order `(1−B)^d` operator with fixed-width weights and choose the MINIMUM `d*` that passes an ADF stationarity test, so the transformed series is stationary while RETAINING most of the level/long-memory the integer-1 difference (plain returns) throws away. Tried as a feature transform feeding the certified cross-pair direction book (USD factor / residual / lead-lag, fractionally differenced instead of differenced to returns).
- **How.** `frac_diff.py` — hand-rolled FFD (fixed-width weights + ADF d*-selection). `frac_direction.py` (+ `frac_direction_15m_result.json` / `frac_direction_30m_result.json`) folds per-pair FFD channels into the direction book, **frozen-past forward holdout** (Family 9.6), NY cov0.10 selacc, arms {base, +ffd, ffdonly}. Per-pair d*: EURUSD/NZD/CHF = 0.1, GBP/AUD/JPY/CAD = 0.2 (windows ~500 bars), thresh 1e-4.
- **Why.** Plain returns destroy level memory; if a fractionally-differenced series keeps long memory AND is stationary it could expose a slow directional signal the return-space book misses.
- **Process notes (leakage / discipline).** d* and the ADF selection are TRAIN-only; FFD windows are causal fixed-width. MUST be judged on the forward holdout, not pooled CPCV — a memory-preserving transform of a globally-drifting factor is exactly a **leakage trap #9** candidate (could memorize era-local level under pooling). Self-check on a random walk: needs d*=0.1 to pass ADF (p .019) while retaining 98.8% level-memory vs 0.9% for plain returns — confirms "stationarity with memory" works as intended.
- **STATUS.** **TESTED — KILLED at BOTH 15m and 30m.** 15m: base selacc 2024 .5900 / 2025 .5520 / 2026 .5254 (pooled .5642); +ffd .5792/.5301/.5403 (pooled .5537) → DECAYS (Δ 2024 −0.0108, 2025 −0.0219, 2026 +0.0149), deployable=false; ffdonly pooled .5001 (coin flip). 30m harder: base 2024 .5842 / 2025 .5483 / 2026 .5007 (pooled .5558); +ffd .5546/.5135/.5064 (pooled .5308) → DECAYS (Δ 2024 −0.0296, 2025 −0.0348, 2026 +0.0057), deployable=false; ffdonly pooled .4958 (below coin flip). FFD level-memory adds nothing to direction and HURTS recent years. Note: the base cross-pair book itself decays over forward years (.59→.55→.525), consistent with the documented refit-dependence of the cross-pair edge. `docs/CAMPAIGN_2026-06-07_FACTS.md`. (T1 information-driven bars — the AFML companion to FFD — NOT yet built: tick substrate is EURUSD-only and 2021+, so cross-pair synchronization needs proxies for the other 6 pairs; DEFERRED behind this.)

### 4.6 Frozen-basis HAVOK / Hankel-Koopman direction — TESTED, KILLED both horizons (D6)
- **What.** HAVOK (Hankel Alternative View Of Koopman, Brunton 2017): delay-embed a scalar driver into a Hankel matrix, SVD it to get a linear Koopman-like coordinate system plus an intermittent FORCING term whose bursts mark regime transitions; use the signed forcing precursor as a directional feature behind the cross-pair book. Implemented as a frozen-basis variant: fit the SVD basis on TRAIN only, then causally project later bars onto it (no future leakage from re-fitting the basis).
- **How.** `xsec_direction.py` arm **D6 havok** (unified cross-sectional harness on the certified 7-pair panel). Delay-embed (q=60) the USD-factor trend, SVD on TRAIN → freeze r=8 modes, causally project to coords v1..v7 + intermittent forcing v_r (signed precursor) + leading phase; **forward holdout** (Family 9.6), direction NY cov0.10, arms {base, base+fam, famonly, fam_shuffle}. Falsifier: the signed feature must beat .541, book-gating must lift p10, AND it must clear the **phase-randomized surrogate-null** (Family 9.7, control built into the harness).
- **Why.** A dynamical-systems decomposition whose forcing term is explicitly a SIGNED transition precursor — the kind of structure the sign-invariance theorem does not forbid and the GBM/CNN families do not encode.
- **Process notes (leakage / discipline).** Freezing the SVD basis on TRAIN is the leakage firewall (re-fitting per fold would peek). Paired with a mechanism-specificity shuffle control and the phase-randomized surrogate-null so a positive is not just spectral re-encoding (trap-#9-adjacent).
- **STATUS.** **TESTED — KILLED at BOTH 15m and 30m.** `xsec_direction.py` which=havok, `xsec_direction_havok_{15,30}m_result.json`. 15m: +havok DECAYS (Δ −.003/−.0024/+.0014), `havokonly` pooled **.5148** (sub-breakeven, far below base .5624); the forcing only MARGINALLY beats its phase-randomized surrogate (.5148 vs .5063) and never clears .541. 30m: +havok decays, `havokonly` pooled .5197 (sub-base). The Koopman forcing precursor carries no deployable sign. `docs/CAMPAIGN_2026-06-07_FACTS.md` (Phase 4). (Siblings in the same harness: D1 signature Family 3.7 — also killed/spurious; D7 signed-semivariance Family 3.8 — the only arm to clear its shuffle null, REAL-but-sub-breakeven.)

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

### 5.4 Kronos candlestick foundation model (zero-shot + single-process GPU fine-tune) — DIRECTION null all horizons/sessions (BUILT 2026-06-06)
- **What.** Generative finance-native foundation model (arXiv:2508.02739): BSQ tokenizer → 2 tokens/bar → decoder-only AR transformer; open checkpoints mini 4.1M / small 24.7M / base 102.3M. Forecasts a forward OHLCV path from a window of context candles; the binary signal is DERIVED from the predicted path (`Pup = pred_close(+H) > entry_ref`) — see the corrected harness 5.5 and trap §8 FM-F.
- **How.** Zero-shot (NATIVE mode) needs only the HF weights + the corrected harness. Fine-tune = `kronos_ft.py`: a single-process GPU adaptation of the upstream GPU/DDP-only `finetune/train_predictor.py` — frozen base tokenizer, predictor FT on a session-only contiguous 1-min stream, AMP bf16, early-stop. Runs on the box's RTX 5050 (8GB, Blackwell sm_120, venv torch `2.12.0+cu130` matched to driver 580/CUDA13). Builder for its OHLCV + forward deriv-label input: `kronos_bars.py` (H-min / fine-grid OHLCV, generalizes `barcnn_bars.py`).
- **Why.** A 5th model class — a pretrained generative candlestick model — to test whether learned price-path priors carry 60s–30m SIGN that the GBM/CNN/sequence families miss.
- **Process notes (LEAKAGE TRAP — FM-F).** This is the canonical forecast-derivation-misalignment case (trap §8): the predicted window MUST equal the deriv label window. The original `kronos_dir.py`/`kronos_ft.py` eval was off by one bar → false NULL; now superseded and gated behind `KRONOS_DIR_LEGACY=1`. Use `kronos_mtf.py` (5.5).
- **STATUS.** **null at EVERY horizon (1/5/10/15/30m), zero-shot AND fine-tuned, all sessions** (alignment-corrected): pooled .50–.51, CPCV path_p10 .489–.500, all KILLED, up-rates in-band. Even at NY ≥10m where the cross-pair book CERTIFIES .57–.61, Kronos reads ~.50 — it ingests only EURUSD's OWN OHLCV candles, not the 7-pair USD cross-section that carries the directional edge (Family 3.2). Fine-tune did NOT help direction. Files `kronos_dir_mtf_*_result.json`. **FINE-mode "use a finer grid up the chain" (1m→2/5/10m, 5m→10m) all KILLED** (pooled .490/.509/.495/.503, CPCV p10 .481/.499/.475/.497, frac_clear 0.0) — finer sub-horizon context adds no sign. **Multi-TF vote-ensemble (`kronos_ensemble.py`) KILLED** (5m pooled .531/p10 .428, no year CI95-lo≥.541) or ABORT (10m only 12 common decision bars across 3 grids) — nonoverlap-chrono at different grids leaves too few shared decision instants, and combining single-pair views ≠ the cross-section. Both of the user's multi-TF ideas tested and null. **MAGNITUDE use — Kronos per-path DISPERSION as a forward-vol feature → KILLED 2026-06-06** (`kronos_disp.py`): taps the K sample paths that `kronos.py:467` discards (mean-collapse), builds 6 forward-dispersion features (terminal-ret std/absmean/q90-10/IQR, path-range, within-path rv), paired CPCV ablation vs `[-pe,rv30,rv120]` on 15,041 nonoverlap bars (K=24, pred_len=30, all horizons). ΔAUC = −.0003 (H=1, no effect) to −.0070 (H=30, hurts); CI95 excludes 0 below for H≥5. Dispersion is USED by the GBM but correlates .46–.83 with rv30 — a noisier Monte-Carlo restatement of backward realized vol, no orthogonal magnitude signal. `kronos_disp_disp_main_result.json`. See docs/MAGNITUDE_FINDINGS.md §6c. Lesson: single-pair generative path forecast doesn't beat cheap trailing rv for magnitude. **MAGNITUDE use #2 — `decode_s1` 512-d HIDDEN STATE as a frozen GBM feature → SMALL REAL LIFT 2026-06-06** (`kronos_embed.py`, Lever 2): one forward pass/window (no AR/sampling, ~190 win/s), paired CPCV ablation on 18,077 bars. FIRST feature to beat the rv baseline — paired ΔAUC +.0067/+.0022/+.0076/+.0024/+.0058 at H=1/5/10/15/30m, **every CI95 excludes 0** (clears +.005 bar at 1/10/30m; positive-sub-bar at 5/15m); rv .73→.74. Modest; emb carries time-of-day so likely captures intraday-vol SEASONALITY (cheaper via §7 deseasonalized-RV — next check: add hour-of-day to baseline). DIRECTION emb-only NULL (AUC .50–.51, KILLED). `kronos_embed_embed_main_result.json`, docs/MAGNITUDE_FINDINGS.md §6d. Confirms Kronos's value is magnitude/vol, not sign. **MULTIVARIATE TSFM — Chronos-2 GROUP-ATTENTION on the 7-pair USD panel → DIRECTION NULL 2026-06-07** (`chronos2_xpair.py`, Lever 3): the cross-sectional bet. amazon/chronos-2 (119.5M, the only mainstream TSFM whose group-attention mixes across variates) fed the L=512 close panel of all 7 USD pairs, EURUSD direction read 3 ways on 44,999 bars (forecast-sign, embed→GBM 1536-d, NY/LDN/Asia). **KILLED every horizon/method:** acc .505–.511, AUC .506–.514, CPCV p10 ~.50 (<breakeven .541, <.52 AUC bar). Faint NY-tilt + AUC>.50 (signal present, non-deployable). Forecast spread → magnitude sub-bar (§6e). **Generic group-attention on raw price LEVELS does NOT recover the cross-sectional sign — the deployable edge (.57–.61 NY, `session_xpair`) is in the ENGINEERED USD-residual/basket-catchup/lead-lag features, not foundation-model-recoverable** (matches arXiv:2511.18578). `chronos2_xpair_c2_main_result.json`. **All 3 Kronos/TSFM levers from the web-research synthesis now resolved: L1 dispersion KILLED, L2 embedding small-mag-win/dir-null, L3 Chronos-2 dir-null.**

### 5.5 Corrected multi-timeframe Kronos direction harness (`kronos_mtf.py`)
- **What.** The alignment-fixed eval harness that derives a binary direction call from the Kronos forecast and scores it deriv-faithfully. Native (zero-shot) + fine modes; per-session; emits ensemble-ready `.npz`.
- **How.** `kronos_mtf.py`. Context ends AT decision bar `i` (`slice(i-L+1,i+1)`, last close = entry ref `C[i]`); predict `pred_len = H/GRID` FORWARD steps; `Pup = pred_close(+H) > C[i]`. Modes: **NATIVE** = zero-shot base checkpoint; **FINE** = "predict-H-steps-from-1m up the chain" using the `kronos_ft.py` predictor. Multi-TF ensemble via `kronos_ensemble.py` (vote-combine across TFs). Pred-side contiguity guard `t[i+Hsteps]-t[i]==Hsteps·step`; nonoverlap `GAP = HS + TOL`.
- **Why.** Without this harness a generative forecast is scored against the wrong window (FM-F) → false NULL; this is the only deriv-faithful way to test a forecast-derived sign.
- **Process notes.** Replaces `kronos_dir.py` (now legacy-gated). The forward-label correctness check (agrees with next-bar sign 92.3%, n=233,950) is what licenses scoring `pred_close(+H)` against `y[i]`.
- **STATUS.** **tooling + null result** — see 5.4 for the numbers; the harness itself is correct (FM-F fixed).

### 5.5 Bar-image 2-D CNN (Sezer CNN-BI / GAF) — DIRECTION null, MAGNITUDE clears >65% (sign-invariance, by construction)
- **What.** Render a window of OHLC bars as a 2-D image and learn LOCAL pattern detectors with a 2-D conv — the bar/candlestick sub-lever NOT subsumed by the GBM's geometric feats or the 1-D GRU/CNN (5.1). Faithful to Sezer & Ozbayoglu CNN-BI (arXiv:1903.04610) + GAF (Wang-Oates arXiv:1506.00327); the look-ahead Sezer slope label is REPLACED by the deriv-faithful 60s `wc_ret` outcome.
- **Construction (full paper extractions in the `barcnn_*.py` script docstrings).** 30-bar window of 1-min OHLCV (built from 1s ticks, `barcnn_bars.py`); small MNIST-class CNN (Conv32 3×3 → Conv64 3×3 → MaxPool2 → Dropout.25 → Dense128 → Dropout.5 → 1-logit sigmoid, BCE, AdamW). **Image variants:** `hist` = Sezer bottom-anchored close-histogram 1×30×30; `ohlc` = 3ch wick+up-body+down-body 3×30×30; `gaf` = GASF(symmetric=magnitude)+GADF(antisymmetric=sign) 2×30×30; **`*abs` = ABSOLUTE-scale rendering** (center on window-mean close, fixed pips/pixel) which PRESERVES absolute volatility instead of per-window min-max normalizing it away — the key knob for magnitude. Refs/feasibility on Kronos: see 5.4.
- **How.** `barcnn_bars.py` → `barcnn_run.py <variant>` (direction) / `barcnn_mag.py <variant>` (magnitude) → `barcnn_cpcv.py` + `barcnn_regime.py`. Per-key numbers: DIRECTION → `docs/DIRECTION_FINDINGS.md` / `results/EURUSD_RESULTS.md` §60s row 22; MAGNITUDE → `docs/MAGNITUDE_FINDINGS.md`.
- **Why.** A 2-D conv over the rendered chart is a 4th model class; if any bar-pattern carried 60s SIGN the image-conv (incl. GADF) would catch it — it does not. Move SIZE, however, is exactly what a bar image encodes.
- **STATUS (sign-invariance, demonstrated by one method across both targets).** **DIRECTION = null/EXHAUSTED [EURUSD·60s]** — hist/ohlc/gaf VAL dirAUC ≈ .50, CPCV path_p10 .484–.499, 0.0 paths clear 0.541 (raw + regime-gated). **MAGNITUDE = clears >65% [EURUSD·60s]** — the `ohlcabs` absolute-scale image hits magAUC .699/.714/.686 (VAL/test/oos) and selective large-call precision .68→.80 with **all 28 CPCV paths ≥ 0.65 at cov ≤ 0.2, every held-out year** (`barcnn_mag_ohlcabs_result.json`). Per-window min-max (`ohlc`) only reaches magAUC .64 — preserving absolute scale is what crosses 65%. Bar images = the cleanest single-method sign-invariance proof in the program. Needs O/H/L → EURUSD-only on disk; external-data lever for close-only pairs (USDJPY).

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
- **STATUS.** **helped (filter) / null (specialist)** — 60s up/dip-buy carries (2025 .584/2026 .613), down dead; separately-trained specialist WORSE (subset-training kills ranking); regime-dependent. See `results/EURUSD_RESULTS.md` leaderboard.

### 7.10 Hawkes / transfer-entropy / VPIN (backlog)
- **What.** Self-exciting up-tick vs down-tick arrival intensity (Hawkes), directed info flow volume→price (transfer entropy), order-flow toxicity (VPIN).
- **How.** `docs/IDEAS_LOG.md` B.9, D.16/18. Need trade-signed/depth-resolved LOB (on-disk feed is indicative quote, no trade signs).
- **STATUS.** **not run** — blocked on data.

### 7.11 Paired venue-proposal skew — payout-aware, no-fill
- **What.** Pair contemporaneous CALL and PUT proposals, normalize each quoted ask by its payout to obtain comparable breakeven prices, and treat their imbalance as an indicative direction feature. A proposal is not a fill, so the method supports only `QUOTE_CONDITIONED_NO_FILL` claims until separately measured execution exists.
- **How.** Authenticate source identity, causal pairing, and preregistered minimum coverage before outcome access. The v1 operator, `gbpusd_m15_down_quote_skew_screen_v1.py` [GBPUSD·15m DOWN], implements only that outcome-blind source-feasibility gate; it does not compute skew efficacy or settlement.
- **Process notes.** Exact shared-cycle/source clocks and adequate retention are prerequisites for a prospective test. Source insufficiency is a capability result, not a directional null, and it authorizes neither new capture nor buying.
- **STATUS.** **SOURCE-FEASIBILITY TESTED; EFFICACY UNTESTED** [GBPUSD·15m DOWN]. Tier-2 records: `results/GBPUSD_RESULTS.md`, `sweeps/GBPUSD_15m.md`.

---

## Family 8 — Magnitude / volatility (the one certified edge)

### 8.1 Magnitude (|return| ≥ Q) classifier
- **What.** A 2nd model predicts P(|ret_H| ≥ train-Q75) — the large-move probability. **The genuine forecastable edge** at every horizon (sign-invariance theorem arXiv:2512.15720: entropy/order-flow/complexity gate move SIZE, not sign).
- **How.** `m30_magnitude.py` (rv30+PE → |ret30|), `m10_magdir.py` (10m), `min1_v10.py`/`_redteam_magdir60.py` (60s), `min2_v1.py` (120s). Retarget: set the |ret_H| threshold + horizon; primary feature = realized vol (diurnal-deseasonalized RV / signed semivariance are the documented upgrades).
- **Why.** Volatility is forecastable; sign is ~EMH. Tradeable on Touch/Range/Straddle/VRP, NOT up/down.
- **Process notes.** Realized vol is the real predictor; permutation entropy is NULL for magnitude on FX (corr ~0.01).
- **STATUS.** **survived (BEST)** — 30m large-move AUC 0.73–0.78, 10m 0.706–0.813, 60s 0.787 vs dirAUC 0.510 (`magnitude_verified.json`; `docs/MAGNITUDE_FINDINGS.md`). CPCV-certified (see 9.1).

### 8.2 Direction-conditioned-on-magnitude (sign-invariance test)
- **What.** Does direction become predictable on bars where the magnitude model predicts a big move.
- **How.** `m10_magdir.py`, `_redteam_magdir60.py`, `min1_v11.py`/`v13.py`.
- **STATUS.** **null (confirms theorem)** — direction flat ~0.51–0.53 across all magnitude quartiles.

### 8.3 Complexity / predictability gates (PE / Hurst / RQA / Lempel-Ziv)
- **What.** Permutation entropy (Bandt-Pompe d=3/4), lag-1 autocorr, variance-ratio Hurst, (backlog: RQA-DET, sample/multiscale entropy, Lyapunov, 0-1 chaos) as "bet only in low-complexity/predictable bins".
- **How.** `m30_complexity.py`; full catalog `docs/IDEAS_LOG.md` A.1–8.
- **STATUS.** **null for direction** — conditional accuracy FLAT across all bins (~0.515); these gate magnitude, not sign.

### 8.4 Non-time information bars (volume / dollar / imbalance)
- **What.** Sample by information arrival (cumulative volume/dollar) instead of the clock; predict 30m time-forward direction (López de Prado AFML ch2).
- **How.** `vbars.py` (`build` / `model V` / `model D`).
- **STATUS.** **null for direction** — AUC 0.509–0.516; improves return normality, not directional AUC.

### 8.5 HAR / realized-measure vol family (bipower jump split · realized semivariance RS± · realized quarticity / HARQ · signed jump)
- **What.** The canonical realized-volatility decomposition family layered on the certified magnitude model. **Multiscale HAR** (daily/weekly/monthly RV cascade, Corsi 2009); **bipower-variation JUMP split** (Barndorff-Nielsen-Shephard: separate the continuous diffusion from discontinuous jumps); **realized SEMIVARIANCE** RS⁺/RS⁻ + signed-jump (up- vs down-variation, the directional decomposition of RV); **realized QUARTICITY / HARQ** (Bollerslev-Patton-Quaedvlieg: scale the HAR persistence by the noise in the RV estimate itself). Each is an ARM added to the certified base magnitude features `[-pe, rv30, rv120]`.
- **How.** `mag_har.py` + `mag_har_result.json`. Arms {+har, +jump, +semivar, +harq, +all-stacked} on base `[-pe,rv30,rv120]`; **frozen-past forward holdout** (Family 9.6), horizons 10/15/30m, target `|ret_H| ≥ train-Q75`. Falsifier = +0.005 AUC in ≥2 forward years AND ≥2 horizons. Retarget via the standard `MX_HOR` / threshold knobs.
- **Why.** These are the textbook realized-measure upgrades to a pure-RV magnitude model; test whether jump/semivariance/quarticity carry magnitude signal ORTHOGONAL to the trailing RV the base already uses.
- **Process notes (leakage / discipline).** Judged on the forward holdout, NOT pooled CPCV, because seasonal/slow vol structure is a **leakage trap #9** risk. Collinearity audit vs rv120: lRV120 .89, RS .70, HARQ .58/−.63 — all heavily collinear; only signed-jump SJ120 is orthogonal (−.04) and it carries nothing. Read the forward-consistency of the arm across all year×horizon cells, not the pooled mean.
- **STATUS.** **TESTED — KILLED as a deployable upgrade (REAL-but-SUB-BAR); magnitude path EXHAUSTED on-disk.** Per-arm mean fwd ΔAUC: +har +0.0010 (deployable 2/3 — sub-bar); +jump +0.0003 (1/3 — null); +semivar −0.0000 (1/3 — null); +harq +0.0004 (2/3 — null); **+all stacked +0.0021, 3/3 deployable (no decay) — forward-CONSISTENT (positive in all 9 year×horizon cells) but economically negligible.** Two findings: (1) UNLIKE the §6f time-of-day trap, the additions are forward-CONSISTENT — genuine but tiny; base RV already extracts ~all magnitude. (2) **RE-VALIDATES the certified magnitude edge on a clean deployment-faithful holdout** — base AUC .799/.750/.750 (10m), .791/.739/.737 (30m), 4–5.6× decile lift, **NO decay**: the one positive of the campaign. `mag_har_result.json`, docs/MAGNITUDE_FINDINGS.md §6g, `docs/CAMPAIGN_2026-06-07_FACTS.md`.

---

## Family 9 — Validation methodology

### 9.1 Combinatorial Purged CV + Deflated-Sharpe / PBO
- **What.** Replace the single chronological partition with C(N,k) purged+embargoed OOS paths → a DISTRIBUTION; a model whose 10th-percentile path clears the bar is real, one clearing only on the mean/lucky-split is a mirage. Plus PBO and deflation by N_trials given corr(VAL,OOS)=−0.54.
- **How.** `cpcv_certify.py` (generic, N=8/k=2/28 paths/70 trials/embargo=1 horizon), `min15_cpcv.py` (faithful CPCV of the actual 3-model m15 book, N=6/k=2/15 paths). Targets: 15m direction + 30m magnitude.
- **Why.** Certify which headline numbers survive trial-deflation.
- **Process notes.** `cpcv_certify.py` used a single LGBM + q33 + all-era pool (a *weaker* re-implementation) → its lower 15m number is NOT a faithful test; `min15_cpcv.py` replicates the real ensemble + VAL-tuned gate + recent-held-out and reproduces the edge. Memory: one parquet at a time, subsample ≤100k.
- **STATUS.** **survived (faithful)** — `min15_cpcv_result.json` mean 0.5787 / p10 0.5573 (15 paths); magnitude 30m AUC mean 0.744 / p10 0.718 / 5.0× decile lift survives (`cpcv_certify_result.json`). The generic 15m selacc (mean 0.545) is the deflated weaker re-implementation, not a refutation.

### 9.1b A LEVER's refit-CPCV gain ≠ a deployable edge — frozen-forward-gate every improvement
- **What.** When a lever (alternative label, blend, loss-weight, stack) shows a refit-CPCV p10 *gain over the base book*, that gain can itself be **refit-overfit**: the per-fold-refitting model exploits the lever in each era's recent data, but a FROZEN book trained with the lever generalises **worse** than the frozen base. The refit-CPCV cert measures the per-era *floor*, not whether the lever's *advantage* deploys.
- **How.** After any refit-CPCV improvement, train the lever ONCE on the past (2012-21), freeze, and compare its per-year forward win-rate to the base book's forward in the binding years. Promote only if the lever's frozen-forward ≥ base in the binding years (not just refit-CPCV p10 > base).
- **STATUS (TESTED).** `[USDCAD·15m]` — **TWO levers each showed a clean refit-CPCV +.01–.02 UP p10 gain (TB first-touch label; 15m×30m cross-horizon blend, the latter even passing a matched-seed-ens redundancy test, corr .75) yet BOTH were refit-overfit**: frozen-forward UP/COMB fell *below* the base book (TB UP .5355<.5714 2026; xhblend COMB .5407<.5696 2025) → KILLED. `usdcad_15m_freeze_tb_result.json`, `usdcad_15m_freeze_xh_result.json`. Complements 9.1 (read p10 not mean) and the trap "ALWAYS verify a frozen-VAL gate under refit-CPCV" (Family 3.3) with its converse: **a refit-CPCV lever-gain must clear the FROZEN-FORWARD before promotion.** See `results/USDCAD_RESULTS.md` FINAL CONCLUSION.

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
- **How.** `docs/EXPERIMENT_BACKLOG.md` #8.
- **STATUS.** **not yet run** (backlog; tighter CI, not new signal).

### 9.6 Frozen-past FORWARD-HOLDOUT gate (the mandatory deployment-faithful falsifier)
- **What.** The deployment gate that catches what CPCV cannot: train on a FROZEN past (≤ year Y) and judge per-year on each later held-out year (Y+1, Y+2, …), so the model is scored ONLY on a future it never touched. This is the necessary complement to CPCV — pooled CPCV purge+embargo kills label-OVERLAP leakage but **does not detect forward NON-transfer**, the exact failure of a locally-stationary / globally-drifting calendar/seasonal/slow-regime feature (**leakage trap #9**, below). An edge that wins clean under pooled CPCV but DECAYS across forward years is a trap-#9 mirage, not a deployable edge.
- **How.** `fwd_holdout.py` — reusable gate with magnitude (AUC + decile-lift) and direction (cov-selacc) modes; train ≤2023 → per-year 2024 / 2025 / 2026. Used by `mag_har.py` (Family 8.5), `frac_direction.py` (Family 4.5), and the whole 2026-06-07 novel-methods campaign. Falsifier convention: require the arm to clear its bar in ≥2 forward years AND ≥2 horizons (no single-year luck).
- **Why.** Production books already use a frozen-past split for this reason; this gate makes the discipline a first-class, self-checking tool any new lever must pass before it can be called deployable. CPCV is necessary, NOT sufficient.
- **Process notes.** Self-checked at build time: it DEPLOYS a stationary injected signal (+0.10/yr) and REJECTS a deliberately trap-#9 non-stationary feature (decays to −0.015) — i.e. the deseason-+time-of-day failure mode it exists to catch. Read forward-year deltas, not the pooled mean: a positive pooled number with a negative forward slope = FAIL. Cross-reference **leakage trap #9** for the worked deseasonalized-RV example (clean pooled-CPCV ΔAUC +0.0140, all 28 paths positive, yet forward 2024 +.020 → 2025 −.009 → 2026 −.054).
- **STATUS.** **TESTED (BUILT + self-checked)** — gate committed `051e525` (NOVEL §0 #1); both injected controls behaved as designed (stationary deploys, non-stationary rejected). The campaign's positive finding is delivered THROUGH this gate: it RE-VALIDATED the certified magnitude edge on a clean deployment-faithful holdout (base AUC .799/.750/.750 @10m, .791/.739/.737 @30m, 4–5.6× decile lift, NO decay — see Family 8.5). `docs/CAMPAIGN_2026-06-07_FACTS.md`.

### 9.7 Phase-randomized / IAAFT SURROGATE-NULL gate
- **What.** A statistical null that asks whether a candidate signal is genuine NONLINEAR structure or merely a re-encoding of the series' linear spectrum. Generate surrogate series that PRESERVE the power spectrum (and, for IAAFT, the amplitude distribution) while destroying nonlinear phase structure; compute the discriminating statistic on real vs many surrogates; the signal is real only if the real statistic exceeds the surrogate null band (e.g. p95).
- **How.** `surrogate_null.py` — phase-randomize / IAAFT surrogate-null gate (NOVEL §0 #2). Intended as the pre-registered null for any nonlinear path/complexity lever (HAVOK forcing, signature, dynamical-systems direction methods — its phase-randomized control is built into the `xsec_direction.py` D6 falsifier, Family 4.6).
- **Why.** Many "nonlinear" features (entropy, signature, manifold coords) can score above a naive shuffle purely because they re-read the autocorrelation/spectrum the surrogate also has. This gate separates spectral re-encoding from genuine structure so a method is not credited for linear information already covered elsewhere.
- **Process notes.** Self-checked at build time: a LINEAR lag-1-autocorr statistic comes back NOT significant (real .6959 ≈ null_p95 .6961 — spectrum preserved, so the surrogate matches it), while a NONLINEAR |·| vol-clustering statistic comes back significant (real .0883 ≫ null_p95 .0199). I.e. it correctly fails to flag the linear quantity and correctly flags the nonlinear one.
- **STATUS.** **TESTED (BUILT + self-checked + APPLIED)** — gate committed `051e525`; both self-check arms behaved as designed. APPLIED in Phase 4: D6 HAVOK's Koopman forcing only marginally beats its phase-randomized surrogate (havokonly .5148 vs surrogate .5063) and never clears .541 → KILLED (Family 4.6). Companion mechanism-specificity shuffle controls in the same campaign distinguished spurious D1 signature (rotation surrogate does not degrade it, Family 3.7) from genuine D7 signed-semivariance (sign-flip shuffle DOES degrade it, +.015/.030, Family 3.8). `docs/CAMPAIGN_2026-06-07_FACTS.md`.

### 9.8 Brier-advantage vs a calibrated baseline (probabilistic-skill-vs-ranking audit)
- **What.** A calibration audit that complements AUC + selective-accuracy (both of which reward RANKING, not the quality of the probabilities). Score `badv = baseline_brier − model_brier`, where `model_brier = mean((p−y)²)` on CALIBRATED probs and the baseline is (a) flat-0.5 and (b) an **observability-safe trailing-persistence up-rate** (a past decision bar contributes only once its outcome is known, i.e. shifted by the horizon). `badv > 0` ⇒ the model's probabilities carry genuine skill ON TOP OF the baseline; `badv ≤ 0` ⇒ the AUC/win-rate edge is ranking-only with no probabilistic content over a naive forecast.
- **How.** `fwd_holdout.py` opt-in `brier=True` (params `persist_gap_s = horizon·60`, `calibrate=True` → train-only OOF-isotonic, `cal_k`). Reports per forward year, all-bars AND the selective bet-tail, CALIBRATED and RAW. Driver `brier_audit.py` runs it over the certified books. Source: imported from `evan-kolberg/prediction-market-backtesting` — the one transferable lever from the 2026-06-07 11-repo Polymarket scour (see memory `polymarket-repos-scour-outcome`).
- **Why.** Calibrate first (isotonic on train-only OOF), or you measure miscalibration not skill. The audit can only ever TIGHTEN a certification, never inflate one — it cannot create alpha, only verify whether ranking skill is also probabilistic skill. Watch the BASELINE quality: at intraday FX horizons the persistence baseline is itself worse-than-flat (Brier > 0.25, mild anti-persistence), so "beats persistence" is a weaker bar than "beats flat-0.5".
- **Process notes.** Self-checked at build time: a feature with genuine sign-skill scores `badv_persist` ≈ +0.026/yr (bet-tail +0.077), a pure-noise model ≈ −0.0002/yr. Read all-bars AND bet-tail: full-distribution Brier dilutes selective-tail skill with the no-skill bulk, so a tail-concentrated edge shows a razor-thin all-bars number.
- **STATUS.** **TESTED (BUILT + self-checked + APPLIED)** — Applied 2026-06-07 to all 4 certified cross-pair direction books (m5xp/m10xp/m15xp/m30xp): ALL PASS the falsifier (`badv_persist > 0` every forward year 2024/25/26, all-bars + bet-tail, calibrated + raw → none ranking-only), but vs the harsher flat-0.5 baseline the all-bars edge is razor-thin (`badv_flat ≤ +0.0013`, slightly NEGATIVE in 2026 at 15/30m) and the genuine skill is concentrated in the selective bet-tail (`badv_persist_sel +0.005…+0.017`). Net: certs survive, interpretation tightened to thin/tail-concentrated/refit-dependent; no downgrade. `brier_audit_result.json`, `brier_audit.py`. EXPERIMENT_LEDGER #161; DIRECTION_FINDINGS §2026-06-07.

---

## Cross-cutting leakage traps (apply to every family)
1. **Sequence-model future-peeking** — HMM Viterbi/forward-backward, Kalman RTS smoother. Use the forward FILTER only.
2. **ffill-flat-window mirage** — intersecting pairs + ffill manufactures ~50% fake-flat bars → fake AUC 0.7+ collapsing to ~0.49 on moved bars. Caught repeatedly. Always: own-clock, missing→0.0 not ffill, eval moved-bars-only, check up-rate ∈[0.47,0.53].
3. **Bar-shift horizon mislabel** — `shift(-N)` on gap-dropped tick data shifts *bars* not *seconds*. Use `wc_ret`.
4. **Greedy-by-confidence de-overlap** — peeking to keep the most-confident bar; −3 to −9 acc points. Use `nonoverlap_chrono`.
5. **VAL-acc-max selection** — corr(VAL,OOS)=−0.54 anti-transfers. Select by worst-VAL-half.
6. **Thin-coverage mirages** — any n<25–50 pocket at 0.65+ is multiple-testing noise.
7. **Ties LOSE** — a ~0.50-AUC model's realized win-rate sits BELOW 0.50 once ties are charged.
8. **FM-F forecast-derivation window misalignment** — when a binary up/down signal is DERIVED from a generative price/return FORECAST (a foundation model / state-space / x-horizon predictor that emits a price path, not a classifier scored directly on the label), the **predicted window must EXACTLY equal the label window** = entry reference price + the same forward horizon. The Kronos worked example: at decision bar `i` the eval fed context `slice(i-L,i)` = bars `[i-L..i-1]`, predicted bar `i`, and scored `Pup = pred_close(i) > C[i-1]` — i.e. the move INTO the entry over window `[t[i-1],t[i]]`. But the deriv label `y[i]` is the FORWARD window `[t[i]+1s, t[i]+61s]` (`barcnn_bars.labels_at`). The two windows are DISJOINT, off by one bar → the forecast was being judged against the wrong outcome. **A misaligned forecast yields a FALSE NULL, never a false POSITIVE** (you score a real prediction against an unrelated label = noise), so this trap suppresses edges rather than inflating them — the opposite failure direction from traps 2–4. **Fix** (`kronos_mtf.py`): context ends AT bar `i` (`slice(i-L+1,i+1)`, last close = entry ref `C[i]`), predict `pred_len = H/GRID` FORWARD steps, score `Pup = pred_close(+H) > C[i]`; enforce pred-side contiguity `t[i+Hsteps]-t[i] == Hsteps·step` and nonoverlap `GAP = HS + TOL`. Validation that the forward label is the right one: it agrees with the next-bar sign 92.3% (n=233,950). GBM/CNN classifiers are trained DIRECTLY on the label and scored against it → structurally immune. Full-suite audit (2026-06-06, below) found this trap ISOLATED to the 2 Kronos scripts; every other forecast-derivation script predicts the forward quantity over the SAME horizon as the label at the SAME bar = correctly aligned.
9. **Pooled-CPCV NON-STATIONARY-FEATURE memorization (2026-06-07)** — combinatorial purged CPCV (8 groups, k=2) builds 28 paths in which **20/28 test folds are temporally FLANKED by train folds on BOTH sides**. A feature that is LOCALLY stationary but GLOBALLY drifting — calendar/time-of-day, seasonality, slow regime state — lets the model memorize era-LOCAL structure from the neighboring train years and score high on the held-out fold **without any forward transfer**. CPCV purge+embargo only kills label-OVERLAP leakage; deflated-Sharpe only penalizes multiple-testing on the metric LEVEL; **NEITHER detects forward non-transfer**. Worked example: adding raw time-of-day `[hour,minute,dow,sin,cos]` to the certified 30m magnitude model gave a clean leakage-free pooled-CPCV ΔAUC **+0.0140** (all 28 paths positive, placebo-negative, real intraday-vol seasonality) — yet a frozen-past forward holdout (train≤2023) decayed **2024 +.020 → 2025 −.009 → 2026 −.054** (the seasonal shape flattened; the clock feature overfits a stale pattern). Unlike traps 2–4 this INFLATES (false positive), and unlike trap 8 it is invisible to CPCV+deflation. **Fix: any CPCV-certified edge leaning on calendar/seasonal/slow features MUST be confirmed by a per-year frozen-past forward holdout (train≤Y → test Y+1,Y+2) before deployment. Pooled CPCV is necessary, not sufficient.** Files `deseason_mag.py`/`deseason_fwd.py`, docs/MAGNITUDE_FINDINGS.md §6f. (This is why production books use a frozen-past split.)

---

## Family 9 — EDGE-IMPROVEMENT levers & MODEL COMBINATIONS (apply to every certified edge; the new incumbent)
> ⚠ NUMBERING NOTE: this is a SECOND "Family 9" (its 9.1–9.6 are edge-improvement levers). It is DISTINCT from the
> "Family 9 — Validation methodology" section above (whose 9.6 = forward-holdout gate, 9.7 = surrogate-null). Campaign
> entries that cite "Family 9.6/9.7" mean the VALIDATION ones and always name the method (forward-holdout / surrogate-null)
> alongside the label. (Collision predates the 2026-06-07 campaign; left as-is to avoid renumbering live cross-refs.)
**A certified book is the START. Run these ON it and evaluate COMBINATIONS — the incumbent at a new (currency,
timeframe) is the best COMBINATION in `books/INDEX.json`, not the old base GBM.** Info-bound caps raw AUC, so
score these on **binding-year win-rate, coverage, and CPCV path-clear-rate**. Saved lit-review + papers:
`/home/sean/git/academic-papers/_DL_for_5m_FX_direction_REVIEW.md`.
### 9.1 Adaptive-conformal (ACI) gate
- **What.** Replace the FIXED confidence/meta threshold with an ONLINE one: trade candidate `t` iff `meta_t ≥ θ_t`; update `θ_{t+1}=θ_t+γ(err_t−(1−w*))` on traded bars only (Gibbs-Candès 2021). Targets a selective win-rate `w*`, trading more in-regime / less off-regime. Causal (past-outcome feedback, no look-ahead) → deployable.
- **How.** `m5_conformal.py` (retarget the candidate filter + meta source). Sweep `w*`∈{.55,.56,.57}, `γ`≈.02.
- **Why / status.** **The improvement that WORKED @5m:** binding 2025 UP .584@n764 vs fixed .579@n618 (better win AND coverage), +36% trades → frozen as `EURUSD.m5xp_aci.v1`. The principled "kill-switch done right". CPCV-validation of the policy is the open follow-up.
### 9.2 Seed-ensemble (multiple LGB random-state seeds, avg probability)
- **What.** Train K copies of the same GBM with different `random_state`/`bagging_seed`/`feature_fraction_seed`; average the probabilities. Reduces seed-lottery variance at near-zero compute premium per seed.
- **Saturation-depth finding (2026-06-10, `[GBPUSD·15m]`).** K=3 SATURATES own-pair ~239-feat models: tested at 3/3 own-pair siblings (EURUSD 15m, USDJPY 15m, AUDUSD 15m — all K=3→K=8 lifts null). K=8 LIFTS the xpair 340-feat GBPUSD model on BOTH sides at cov1 AND cov2 (DOWN @cov2 +.0101, UP @cov2 +.0042, cov1 UP +.0082, DOWN +.0072 p10). Mechanism: a larger, richer feature matrix gives each random seed more independent variance to average away; saturation depth scales with feature-space size, not with K alone. **Rule of thumb:** if the feature matrix > ~300 diverse features, test K=8 before assuming K=3 saturates. For own-pair ~239-feat models, K=3 is sufficient and K>3 adds no p10.
- **Seed-ensemble net ⊕ GBM (decorrelated stack member).** M-seed AdamW MLP (lr 2e-4), average probabilities, blend/stack with the GBM. Only literature-endorsed DL use on tabular (Shwartz-Ziv; Grinsztajn). `m5_deep_ens.py`. @5m: MLP decorrelated (corr .694) but 50/50 blend ≈ GBM — try a LEARNED stack weight + purged OOF.
### 9.3 |return|-weighted / GMADL loss (magnitude→direction bridge)
- **What.** `sample_weight=|wc_ret|` (or GMADL objective): up-weight large-move bars (sign most predictable), leveraging the strong magnitude edge. `m5_magweight.py`. @5m: rebalances to two-sided ~.56, collapses 2026 UP (magnitude-conditional sign is regime-dependent) — GMADL operating-point selection still untried.
### 9.4 Calibration + selective threshold
- **What.** Temperature/Venn-Abers calibration so the confidence gate is honest; re-derive the gate post-calibration; verify the up-rate tripwire holds. Nearly free; the required wrapper around any confidence-gated edge.
### 9.5 Cross-pair POOLING (train-row pooling across majors) — TESTED: decorrelation, lifts MEAN not worst-regime p10
- **What.** Train all 7 USD-majors' base features as ROWS in one GBM, each labeled by its OWN forward sign; eval the target pair. (Distinct from cross-pair FEATURES §9.2 / `m5_xpair`, and from a weight-shared net.) Generic base-feature→direction map with Nx data + cross-pair regime diversity.
- **Status (TESTED).** `[USDJPY·2m]` (`usdjpy_2m_xpair.py pool`): pooling LIFTS the per-fold-refit-CPCV win-rate MEAN above the single-pair level (~.52→.546) — but this is **NOISE-DECORRELATION (smoother conditional-mean), NOT a cross-pair factor SIGN** (the cross-pair sign gradient is none@60s→UP@5m, does not reach 2m). The worst-regime **p10 SATURATES (~.531) and does NOT clear breakeven** — variance reduction (seed-ens, +data, multi-algo) plateaus there. **KEY: read the CPCV p10/frac-clear, never the mean.** Multi-algo (lgb+xgb+cat) decorrelation can HURT a thin signal (`[USDJPY·2m]` p10 .519 < seed-ens lgb .531). Use pooling to RAISE THE FLOOR on a thin edge; do not expect it to certify a sub-BE key. See `results/USDJPY_RESULTS.md` × 2m + `sweeps/USDJPY_2m.md`.
### 9.6 COMBINATIONS are first-class
- The model space is a cross-product: {base GBM · cross-pair · cross-horizon stack · seed-ensemble · pooled} × {fixed · ACI · calibrated gate} × {BCE · |ret|-weighted/GMADL} × {up-filter · down-filter · specialist}. Benchmark a new edge against the best combination; try novel combinations (cross-pair + GMADL + ACI + seed-ensemble); freeze each winner as its own `<PAIR>.<book>_<combo>.v1` book.

---

## 2026-06-06 — DST-correct session re-campaign + Kronos look-forward fix + full-suite audit

A session-spanning update touching three families. New scripts: `sessions.py` (DST-correct `session_mask`/`SESSIONS`); `session_1m.py`/`session_2m.py` (tick GBM per session); `session_bars.py` (bar GBM, any `H`); `session_xpair.py` (cross-pair STRICT session-only, any `H`, per-H gate); `kronos_ft.py` (single-process GPU Kronos predictor fine-tune); `kronos_mtf.py` (alignment-CORRECTED multi-TF Kronos direction harness, native + fine + ensemble); `kronos_bars.py` (H-min/fine-grid OHLCV + forward deriv-label builder); `kronos_ensemble.py` (multi-TF vote combine); `barcnn_run.py` gained a `SESSION` arg.

### (1) New leakage trap — FM-F forecast-derivation window misalignment
Added as cross-cutting trap **§8** (above). When a binary signal is DERIVED from a generative price/return forecast, the predicted window must EXACTLY equal the label window (entry ref + forward horizon). The Kronos eval scored `Pup` from the move INTO the entry (`[t[i-1],t[i]]`) against a label on the FORWARD window (`[t[i]+1s,t[i]+61s]`) — disjoint, off by one bar. **A misaligned forecast yields a FALSE NULL, never a false POSITIVE** (the opposite failure direction from the inflation traps 2–4). Worked example + fix in §8 and method 5.5.

### (2) Full-suite correctness audit — bug is ISOLATED, no certified book invalidated
A 19-agent workflow + lead Tier-1 proofs swept all 308 scripts plus a dedicated forecast-derivation sweep. **VERDICT: FM-F is isolated to the 2 Kronos scripts (`kronos_dir.py`, `kronos_ft.py` legacy eval), NOT systemic.** Every other forecast-derivation script (`usdjpy_{1m,2m}_statespace`, `usdjpy_2m_xhorizon`, `m5_xhorizon`, `m5_lossbatch`, `f1_compound`) predicts the FORWARD quantity over the SAME horizon as the label at the SAME bar = correctly aligned. GBM/CNN are classifiers trained directly on the label → structurally immune.
- **Proven clean (Tier-1 empirical):** TICK substrate (`min1`/`min2_production` feats — causal via truncation `max|full-trunc|=0.0`; label forward 0/4000 mismatch; `corr(y,future)=.486` vs `corr(y,past)=-.003`; up-rate .5006). BAR substrate (`harness` features + `contig_fwd` — forward 0/2000 mismatch at H=5/10/30, `corr(y,future)~.99` vs `~-.02`, up-rate .498–.506). XPAIR substrate (`m5_xpair.build_xp _y` — forward 0/2000 mismatch H=10). BAR-feature causality (`pipeline.py` 239 features — truncation `max|full-trunc|=0.0` across ALL features). **No certified direction/magnitude book is invalidated.**
- **Bounded flags (do NOT invalidate any cert):** `barcnn_mag.py:163` + `barcnn_regime.py:69,71` FM-E selective-threshold on pooled test+oos (magnitude/regime target, CPCV-deflated — fix = use VAL threshold); `usdjpy_2m_cpcv2.py` FM-E max-p10 cell (best .5185 ≪ .541 → certified=false anyway); `min2_mim.py:18` + `min2_legsign.py:25` FM-A `shift(-FWD)` no contiguity guard (KILL screens, already KILLED); superseded pre-v3 cohort (`min1_v*`/`min2_v*`/`tickmodel*`/`tick5s_final`/`tick_ensemble` — documented bar-shift label + greedy nonoverlap, no result.json, replaced by `*_production` books).

### (3) Hardware
Box has an NVIDIA RTX 5050 Laptop GPU (8GB, Blackwell sm_120, driver 580/CUDA13); venv torch swapped `2.12.0+cpu → 2.12.0+cu130` (cu130 matches the driver; sm_120 verified). Kronos fine-tune + inference now run on GPU. Training venv `~/binary-algo-venv` (uv).

### (4) DST-correct session re-campaign results (method 2.5)
Session definitions and discipline in 2.5. Headline (all Tier-1 result JSONs):
- **1m tick GBM** (`session_1m`): DIRECTION null all sessions (pooled ~.504, p10 .500–.504, frac_clear 0.0). MAGNITUDE certified all (cov≤10% frac 1.0; magAUC NY .675 / LDN .728 / Asia .717; p10 .60–.71). Files `session_1m_{dir,mag}_{ny,ldn,asia}_result.json`.
- **2m tick GBM** (`session_2m`): DIRECTION null all (p10 .496–.500); MAGNITUDE certified all (p10 .58–.70). `session_2m_*`.
- **5m/10m/15m/30m base-bar GBM** (`session_bars`): DIRECTION KILLED all (NY strongest, e.g. 10m NY p10 .525; 15m NY .520/LDN .515/Asia .512); MAGNITUDE certified all sessions (p10 .72–.80). `session_{5,10,15,30}m_{dir,mag}_{sess}`.
- **Cross-pair book STRICT session-only** (`session_xpair`, the certified ≥10m lever): **NY certifies BOTH sides at every horizon 2m→30m; London/Asia at NONE.** 2m NY .564/.560 (LDN .514/.506, Asia .499/.497); 5m NY .596/.588 (LDN .526/.516, Asia .510/.504); 10m NY UP **.6053** / DOWN **.5896** (15/15 each); LDN .523/.524; Asia .511/.516. 15m NY .5845/.5712; LDN .527/.520; Asia .519/.496. 30m NY .5681/.5639; LDN .520/.514; Asia .487/.509. The 2m & 5m NY two-sided certs are NEW (legacy 2m dead; legacy 5m UP-only). NY BEATS the legacy fixed-UTC gate (10m legacy .586/.568; 30m .559/.553). DIRECTION edge is decisively **NY-concentrated**. Files `session_xpair_{2,5,10,15,30}m_{sess}_result.json`.
- **Kronos direction CORRECTED** (`kronos_mtf`, alignment-fixed, methods 5.4/5.5): **NULL at EVERY horizon 1/5/10/15/30m, zero-shot AND fine-tuned, all sessions** (pooled .50–.51, CPCV p10 .489–.500, all KILLED, up-rates in-band). Even at NY ≥10m where cross-pair certifies .57–.61, Kronos reads ~.50 — it ingests only EURUSD's OWN OHLCV, not the 7-pair USD cross-section that carries the edge; fine-tune did not help direction. Files `kronos_dir_mtf_*_result.json`. **FINE "up the chain" (1m→2/5/10m, 5m→10m) all KILLED** (p10 .481/.499/.475/.497); **multi-TF ensemble KILLED/ABORT** (5m p10 .428; 10m n_common=12). Both user multi-TF ideas null.
- **Per-session bar-image CNN** (`barcnn_run.py SESSION`, Sezer CNN-BI, 4th model class): DIRECTION NULL all sessions (NY VAL .510/test .503/oos .505; LDN .509/.499/.503; Asia .501/.506/.500). `barcnn_hist_{sess}_result.json`. Confirms single-pair candle images carry no per-session sign.

**Net:** the only direction lever that survives the DST-correct session cut is the cross-pair book ≥10m **inside the NY session**, where it improves on the legacy fixed-UTC gate. Magnitude remains certified across every session. The Kronos family is a confirmed direction null (now correctly aligned); its value, if any, is magnitude/path, consistent with the sign-invariance theorem.

---

## Program-level status summary
- **Survived/tradeable:** **5m UP cross-pair direction ~0.55–0.57 — refit-CPCV-certified at the operating gate (first sub-15m direction edge to survive the full refit; `EURUSD.m5xp.v1`), IMPROVED by the adaptive-conformal gate (`EURUSD.m5xp_aci.v1`)**; 15m compression×NY direction **0.647** (CPCV-faithful ~0.58 p10); 30m 0.591; seconds 1–5s tick ~0.65 (needs tick venue); **magnitude AUC 0.73–0.79 everywhere** (the one CPCV-deflation-certified edge).
- **Killed/null with recorded results:** CKS-OFI, cross-impact OFI, **CCM coupling-gate**, ordinal irreversibility, residualized target, Neural-CDE (all have result JSONs); HMM/Kalman/kernel/online/per-side-flow/news/RMT/DRL (ledger + `models/*_summary.json` + logs).
- **Coded / backlog:** foundation models (Kronos), SSA/fractional-diff/particle-filter/reservoir, Hawkes/TE/VPIN (LOB-blocked), sample-uniqueness weighting, magnitude upgrades (deseasonalized RV / signed semivariance). **The only inputs that could change the directional answer are external:** intraday DE–US 2y rate differential, daily implied-vol / risk-reversal, EURGBP ticks.

Files of record: `results/EURUSD_RESULTS.md` (per-horizon results + up/down leaderboard), `docs/EXPERIMENT_LEDGER.md` (135-experiment master table), `docs/MAGNITUDE_FINDINGS.md`, `docs/DIRECTION_FINDINGS.md`, `docs/CCM_DESIGN.md`, `docs/EXPERIMENT_BACKLOG.md`/`docs/IDEAS_LOG.md` (backlog), and the per-result JSONs cited per entry.
