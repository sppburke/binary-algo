> **SCOPE: GENERIC** (currency/timeframe-agnostic). Per-key numbers, if cited, are tagged [PAIR·tf] examples whose record-of-truth is the Tier-2 file. See REPO_MAP.md.

# SWEEP MATRIX — the exhaustive permutation menu for a (currency, timeframe) strategy search

This is the **menu** an exhaustive strategy sweep works through, variant by variant, to find the best UP and
DOWN predictor for one `(currency, timeframe)`. It is the concrete enumeration behind `METHODS_CATALOG.md`
(which has the how/why per method) and `MODEL_REGISTRY.md` (frozen parents to reuse). Driven by the
`strategy-eval` skill in **sweep mode** (see that skill, §8).

**How to use:** at sweep start, instantiate this menu into a per-sweep ledger `sweeps/<PAIR>_<tf>.md` (one
row per method×variant, status `pending`). Work rows in ROI order (top → bottom). For EACH row: retarget to
the timeframe (`MX_HOR`/`HS`), run the `strategy-eval` discipline, evaluate **combined + UP-split + DOWN-split**,
write results into `<PAIR>_RESULTS.md`, mark the ledger row `done`, update the UP/DOWN leaderboard. **Never
skip a row; never stop until every row is `done`** (or killed by its falsifier). Each cell below is one or
more variants — expand the variant axes into individual ledger rows.

Legend: **Tgt** = what it predicts (D=direction, M=magnitude, G=gate, U/Dn=side). **Prior** = rough chance of
a tradeable edge (calibrate the falsifier accordingly; low-prior = design to KILL fast).

## Tier A — highest ROI (run first): GBM core, gates, cross-horizon, cross-pair
| # | Family / method | Script | Variant axes to enumerate | Tgt | Prior |
|---|---|---|---|---|---|
| A1 | 3-model GBM ensemble (lgb+xgb+cat) | `m{5,10,15,30}_production.py`, `min1_production.py` (`MX_HOR`/`HS`) | num_leaves {127,255,350}, n_est {600,2000,3000}, lr {0.02,0.03}, reg_lambda {8,10,20} | D | med |
| A2 | Compression × session × coverage gate sweep | `m10_gate_sweep.py`, `exp_15m_v5_gates.py`, `m{5,30}_lab.py` | compression q {10,20,33,50}, vol-window {300,900,1800,3600}, session {NY, London, overlap, all}, coverage {2,5,10%} | G+D | med |
| A3 | Compression-release × reversion specialist | `min1_production.py` levers | w_spec {0,0.5}, reversion-vs-ret{60,300,900} on/off, rel_tighten pct {70,80,90} | D | med |
| A4 | Meta-labeler on orthogonal axes | `m5_meta.py`, `m15_meta.py`, `m10_stack.py` | axis set {xpair, order-flow, parent-conf}, threshold sweep (worst-VAL-half) | G+D | med |
| A5 | Cross-horizon stack (parent → child front-load) | `m5_stack2.py` (soft), `m5_stack.py` (hard), `min1_stack.py` | parent ∈ frozen books {`<PAIR>.m15.v1`,`.m10.v1`,`.m5.v1`}, soft/hard, q-gate {0.95–0.99} | D | **high@5m** |
| A6 | Cross-pair USD-residual / lead-lag / dispersion | `m5_xpair.py` (`MX_HOR`, modes xp/xpbase/xpof) | mode {xp, xpbase, xpof}, basket {6-major}, residual {reversion, catch-up, lead-lag} | D | med@5m |
| A7 | Walk-forward retrain (regime robustness) | `m{5,10,15}_walkforward.py` | gap {1yr}, retrain-window {expanding, rolling} | D | low (often worse) |
| A8 | Up/down side asymmetry: FILTER vs SPECIALIST | `min1_updown.py` (filter), `min1_upspec.py` (specialist) | side {up, down} × {filter-on-symmetric, separately-trained} | U/Dn | filter med, spec ~null |

## Tier B — microstructure / order-flow (tick & sub-minute)
| # | Family / method | Script | Variant axes | Tgt | Prior |
|---|---|---|---|---|---|
| B1 | Tick microstructure ensemble | `m_tick_prod.py`, `tickhz.py`, `tick_ensemble.py` | HS {1,2,3,5,10,30,60}s, feature set {OBI, microprice, flow}, coverage {0.5,1,2%} | D | **high@1-5s** |
| B2 | Raw order-book imbalance / microprice decay | `rawtick_decay.py`, `rawtick_probe.py` | horizon-ticks {1,5,10,50}, bars {1,5,30} | D | high@ticks |
| B3 | CKS event-OFI (Cont-Kukanov-Stoikov) | `min1_cksofi.py` | window {5,15,30,60}s, standalone vs swap-into-book | D | ~null |
| B4 | Cross-impact OFI matrix | `min1_xofi.py` | pairs {7-major}, lags {0,1,2,5}s | D | ~null |
| B5 | Per-side raw signed flow | `_adj_perside_flow.py`, `orderflow.py` | window {5,15,30,60}s, net/tick-rule/imbalance | D | ~null |
| B6 | Kernel-SVM (Fletcher Nyström+SGD) | `min1_kernel.py` | H {15,30,60}s, kernel-dim, flow on/off | D | ~null |
| B7 | Path-signature / Lévy area | `m30_sig.py` (arg HS) | HS {5,60,300,1800}s, channels {ret, OFI, microprice} | D | top@5s only |

## Tier C — state-space & dynamical-systems
| # | Family / method | Script | Variant axes | Tgt | Prior |
|---|---|---|---|---|---|
| C1 | Hidden Markov regime (causal-filtered) | `min1_hmm.py` | K {2,3,4}, emissions, use {U1 gate, U2 engine-switch, U3 meta}, cov {full,diag} | G+D | ~null |
| C2 | Kalman (channel / velocity / β-residual) | `min1_kalman.py` | signal {channel, velocity, xpair-β}, noise ratio | D | ~null |
| C3 | RMT (Marchenko-Pastur) eigen-residual | `min1_rmt.py` | H {1,15}, coverage by |resid| | D | ~null |
| C4 | Convergent Cross-Mapping coupling-gate | `min1_ccm.py` (`CCM_DESIGN.md`) | drivers {own+6 cross-pair OFI}, E {3,4,5}, tp {1,5,15,30,60}s | G | ~null |
| C5 | Online concept-drift (river ARF+ADWIN) | `min1_online.py` | trees {5 capped, 10 uncapped}, ADWIN δ | D (control) | ~null (keystone) |
| C6 | SSA / fractional-diff / particle-filter / reservoir | backlog (`IDEAS_LOG.md` E.19-22) | — | D/M | low (untried) |

## Tier D — sequence / deep / RL
| # | Family / method | Script | Variant axes | Tgt | Prior |
|---|---|---|---|---|---|
| D1 | 1D-CNN / GRU on raw path | `m_cnn.py`, `min1_v14_cnn.py`, `exp_seq.py` | window-W {20,40,80}s, channels, arch {cnn,gru} | D | ~null |
| D2 | Neural-CDE on irregular Δt path | `min1_ncde.py` | irregular vs grid ablation, hidden {16,24,32} | D | ~null |
| D3 | TabNet attentive tabular | `exp_tabnet.py` | n_steps, width | D | ~null |
| D4 | DRL DQN direction-with-abstain | `min1_drl.py` | reward {deriv-settle}, abstain-cost, ε-schedule | D+policy | ~null |
| D5 | IQN / distributional + CVaR abstain | `min1_drl.py` | quantiles, CVaR-α, mag-gate | D+sizing | ~null (sizing only) |
| D6 | TS foundation model (Kronos zero-shot / Chronos·Moirai·TimesFM fine-tune) | KILLED-on-lit @5m (`sweeps/EURUSD_5m.md`; arXiv:2511.18578 — fine-tune deteriorates, no economic gain; daily-equity, MSE-sign-invariant, CPU-OOM) | sampling N | D | skip |

## Tier E — magnitude & complexity (sign-invariant → the CERTIFIED edge; track in MAGNITUDE_FINDINGS.md)
| # | Family / method | Script | Variant axes | Tgt | Prior |
|---|---|---|---|---|---|
| E1 | Magnitude |ret|≥Q classifier | `m30_magnitude.py`, `m10_magdir.py`, `_redteam_magdir60.py`, `min2_v1.py` | Q {0.67,0.75,0.90}, features {rv-windows, semivariance, deseasonalized-RV} | M | **high (certified)** |
| E2 | Direction-conditioned-on-magnitude | `m10_magdir.py`, `min1_v11/v13.py` | mag-quartile gate | D | ~null (confirms invariance) |
| E3 | Complexity gates (PE/Hurst/autocorr/RQA/LZ) | `m30_complexity.py` (`IDEAS_LOG.md` A.1-8) | stat {PE d3/d4, VR-Hurst, autocorr}, bins | G/M | ~null for D |
| E4 | Information bars (volume/dollar/imbalance) | `vbars.py` | bar-type {volume, dollar}, threshold | D/M | ~null for D |

## Tier F — exogenous / structural / news (mostly null for sub-15m direction)
| # | Family / method | Script | Variant axes | Tgt | Prior |
|---|---|---|---|---|---|
| F1 | Macro-release directional impulse | `min1_news60.py`, `m5_news*.py` | window post-release, vol-tier × |surprise-z| | D | ~null |
| F2 | Structural / price-action / Sofien rules | `m5_patterns.py`, `m5_sofien*.py`, `m30_{regime,gates,fix}.py` | rule set {RSI2, BB%b, NR7, TD, FX-fixing} | D | ~null |
| F3 | External cross-asset lead-lag (ES/NQ→pair) | `m10_xasset_probe.py`, `m30_es_feas.py` | lag, asset {ES,NQ} | D | ~null (mech. key) |
| F4 | Residualized TARGET (label = resid-sign) | `min1_residtarget.py` | β-window, agreement gate | D | ~null |

## NEW-INFORMATION frontier (only inputs that could move the sub-15m / 2025 direction answer)
Not on disk — require data acquisition first (a prerequisite, not a modeling row): **intraday DE–US 2y rate
differential** (Dukascopy), **daily implied-vol / risk-reversal**, **EURGBP ticks** (triangular residual).
Add these as Tier-G rows once acquired.

## Tier N — DISCOVERED / NOVEL (the sweep APPENDS here — keep this growing)
The menu above is a starting point, not a ceiling. To maximize the chance of finding an edge, the sweep
continuously **researches and invents** new candidates and adds them here (see `strategy-eval` skill §8.6).
Sources: arXiv (q-fin.TR/ST, stat.ML) / SSRN / journals; cross-disciplinary transfer (econophysics, info
theory / transfer-entropy / directed-information, point-processes/Hawkes, causal discovery, signal
processing); and **novel combinations** of existing methods. Each new row must pass the cheap vet (genuinely
new · plausible *direction* mechanism that survives sign-invariance · data available · prior + fast-KILL
falsifier) and be logged in `IDEAS_LOG.md` with its source citation.

> **Discovery-vetting outcome — sub-minute microstructure-sign family (generic, 2026-06-03):** two adversarial
> fan-out rounds (20 candidates: OF-surprise/MRR residual, propagator past-sign, DAR sign-AR, Cont-de Larrard queue
> p_up, metaorder continuation, SPDE depletion-asymmetry, microprice, event-clock, idiosyncratic-residual-flow,
> long-OF-history, Lipton drift, predictability-gate + cross-horizon/calendar/UP-cert/recent-lit angles) all returned
> **subsumed or sign-invariant** at ≤2m. Mechanism: they reparameterize the signed-flow channel that decays to noise by
> ~60s (raw OFI/CKS/per-side-flow null; online-ARF keystone ~0.50) or are functions of the return *distribution* not its
> *signed order*. **Do NOT re-add these as Tier-N rows without a genuinely new signed representation.** Per-key kill
> evidence: `sweeps/EURUSD_1m_backlog.md` (discovery rounds 1+2); see also `IDEAS_LOG.md`.

> **Validation-lever menu (generic): multiple-testing haircut** (Holm + BHY-FDR + HLZ t≥3 + N̂-from-ρ̄, via an
> `m{tf}_mt_haircut.py` mirroring `cpcv_certify.py`) — run on a key's whole sweep family to deflate the headline
> by the realized M before trusting a survivor; pre-register the falsifier "procedure must flag the known kills
> non-significant; DOWNGRADE the incumbent if its BY-adjusted binding-year p>0.05 at realized M." Companion levers:
> CSCV/PBO + Deflated-Sharpe + MinBTL on the gated-PnL matrix. **Per-key mt_adjusted_p results live in the key's
> backlog/results** (e.g. EURUSD·5m → `sweeps/EURUSD_5m_backlog.md` + `EURUSD_RESULTS.md`, `m5_mt_haircut_result.json`).

| # | Discovered method / combination | Source (cite) | Mechanism (why it could carry SIGN) | Data | Prior | Status |
|---|---|---|---|---|---|---|
| N2 | **Triangular USD-canceling residual** (EUR-vs-GBP relative-value reversion via EURUSD/GBPUSD) | arXiv:0812.0913 + xpair crumb | signed residual reversion with USD factor ALGEBRAICALLY removed (GBPUSD shares the USD leg) → immune to the 2025 USD-factor sign-inversion that killed RMT/xpair | on-disk (EURUSD+GBPUSD) | **~15% (top)** | testing |
| N3 | Cross-quantilogram tail-lead (Han-Linton-Oka-Whang) | arXiv:1402.1937 | tail-conditional SIGN asymmetry (q.1≠q.9) → not sign-invariant by construction | on-disk (7 legs) | ~12% | pending |
| N4 | Market-Intraday-Momentum term-structure (lagged interval→fwd sign) | Gao-Han-Li-Zhou JFE2018 | signed directional autocorrelation conditioned on clock-time interval (never isolated as the predictor) | on-disk (minute) | ~10% | pending |
| N5 | PCMCI+ causal-discovery sign-stable lead | Runge SciAdv2019 | conditions OUT the common USD factor before scoring a lagged link (separates direct lead from common-driver) | on-disk (7 legs) | ~8% | pending |
| N6 | Directed-information on SIGN sequences (Massey, not TE) | Quinn-Coleman-Kiyavash | binarized sign series → ignores magnitude by construction; clean go/no-go gate (pre-kills N3/N5/N9) | on-disk | ~6% | pending |
| N7 | Asymmetric up/down-tick Hawkes intensity imbalance | Bacry-Muzy arXiv:1301.1135 | cross-excitation asymmetry φuu−φdd (signed), from mid-tick events only (no trade signs) | on-disk (EURUSD mid) | ~9% | pending |
| N8 | Signed-semivariance-skew sign-conditioning on the magnitude model | Patton-Sheppard | RS⁺−RS⁻ skew is signed; condition direction on magnitude-flagged bars (§8.2 used raw quartiles, not RS-skew) | on-disk | ~5% | pending |
| N9 | Cross-pair signed ordinal transition imbalance | Bandt-Pompe / Neuman-Cohen-Tamir | cross-series ascending-vs-descending ordinal transition asymmetry (signed; §7.7 was within-series only) | on-disk (2 legs) | ~5% | pending |
| N10 | TAR-VECM error-correction speed-of-adjustment SIGN (cointegration as VELOCITY) | threshold-VECM lit | signed reversion-velocity of the cross-pair cointegrating residual (sign of EC term × adjustment speed) — NOT residual LEVEL (C3 RMT killed by USD-factor flip); predicts sign of mean-reversion | on-disk (7 legs) | ~15% | killed (m5_tarvecm: coin-flip .49-.51 all yrs) |
| N11 | DeltaLag monotonic pairwise SIGN-ranking loss (train objective) | DeltaLag arXiv | rank-bars-by-P(up) pairwise hinge loss instead of BCE — sign-aware training objective for the cross-pair GBM/MLP | on-disk | ~13% | subsumed by Tier-1 RUN nulls (D1-D6 DL info-bound + N10 + F3a lead-lag flip); audit |
| N12 | DeltaLag adaptive non-stationary lead-lag (per-bar TopK leader+lag, signed) | DeltaLag arXiv | learns WHICH major leads EURUSD + at WHAT lag per bar (sparse cross-attention), signed — vs FIXED lead-lag (F3a/N5 killed) | on-disk (7 legs) | ~12% | killed RUN (m5_rankloss: lambdarank does NOT beat BCE either side) |
| N13 | Queue replenishment/depletion RATE asymmetry (Bouchaud adaptive liquidity) | Bouchaud LOB | signed d(bidvol)/dt−d(askvol)/dt + one-sided depletion-event count per 5m (depth withdrawn ahead of move) — vs imbalance LEVEL (B5a killed); LOW prior (60s OFI nulls) | on-disk (tick quote-vol) | ~8% | subsumed-by-N14 RUN (whale-OIB coin-flip .50 both sides @5m) |
| N14 | Whale (top-1% tick-size) conditioned signed OIB conviction | ssrn-5331995 | net order imbalance from LARGE ticks only (informed conviction) vs pooled OFI noise (B5a killed); LOW prior at 5m | on-disk (tick tsz) | ~7% | killed RUN (m5_whale: .50/.50 both sides all yrs, CI-lo<BE) |
| N15 | Cont-deLarrard closed-form P(up\|queue) + neg tick-sign autocorr reversal | Cont-deLarrard | bounded nonlinear P(up) transform of queue imbalance + theory reversal curvature as GBM features; LOW prior (microstructure decays by 60s) | on-disk (tick quote-vol) | ~6% | subsumed-by-N14 RUN (same quote-OIB family, empirically null @5m) |
| N16 | **Redefined TRAIN label** (triple-barrier / trend-scan / jump-filter-diffusive / vol-deadband) | López de Prado AFML + Lucchese | RELABELS the sign target (not reweight/gate, which every prior kill did) using forward path/jump structure; eval on TRUE 300s sign | on-disk (features + 1m path proxy; 1s train starts 2021) | **~20% (top)** | RUNNING (m5_labels.py) |
| N17 | **Anti-contemporaneous lead-lag + transfer-entropy gate + RFF virtue-of-complexity** | Sirignano-Cont + Schreiber TE + Kelly-Malamud-Zhou VoC | strictly-LAGGED cross-pair (m5xp uses CONCURRENT) + directed-info gate + complexity ridge P≤1e5 — distinct sign-carrying channel/function-form | on-disk (7 legs + features_of) | ~13% | pending (m5_leadlag_te_rff.py, ~6h) |
| N18 | **Sign-COUPLED payoff objective** (GMADL/MADL loss + RRL diff-Sharpe tanh-position head) | GMADL lit + Moody-Saffell RRL | penalizes sign-WRONG large moves MORE (R·R̂ coupling) + realized-payoff-trained signed position — NOT BCE nor symmetric \|ret\|-weight (all killed) | on-disk (cross-pair) | ~10% | pending (m5_signedpayoff_torch.py) |
| N19 | **H=5 risk-residual RELABEL + DOWN-split** | Capponi-Cont common-factor residual | DOWN target = sign(eu_fwd − β_t·USD-basket_fwd) at the never-run H=5, with the DOWN-split min1_residtarget omits | on-disk (7-major basket) | ~8% | pending (m5_residlabel_down_h5.py) |
| … | _(loop-until-dry: stop discovering only after K rounds with no novel survivable idea)_ | | | | | |

> **Corpus subsumption audit (2026-06-03, `m5_corpus_audit_result.json`, 16-agent workflow):** clustered 257 of the 363
> UNTESTED corpus levers into 14 mechanism families → **8 subsumed** (each with a Tier-1 RUN cite), **40 magnitude-only**
> (sign-invariance theorem → `MAGNITUDE_FINDINGS.md`), **5 external-blocked** (need multi-level LOB depth / VIX / funded
> risk-reversal / DE-US rate-diff), and **4 genuinely-distinct direction family-killers** (N16–N19 above) surfaced after an
> adversarial-challenge pass REFUTED 3 subsumptions. **Coverage:** once N16–N19 run, the on-disk 5m DIRECTION mechanism
> space is exhaustively covered on BOTH sides; what remains genuinely open is ONLY external/funded data.

Candidate combination seeds (cheap novelty — start here): magnitude-conditioned cross-horizon stack ·
HMM-regime-gated CCM · RL (IQN+CVaR) sizing on the 15m book · online-adaptive meta-labeler · transfer-entropy
/ directed-information coupling gate · Hawkes up/down arrival-intensity imbalance · twin/seasonal-surrogate
significance on every microstructure feature · causal-discovery (PCMCI) lead-lag across the 7 majors.

## Tier I — EDGE-IMPROVEMENT levers (apply to EVERY edge found, and to COMBINATIONS — not just base models)
**The incumbent to beat at a new (currency, timeframe) is NOT the old base GBM — it is the BEST COMBINATION
already found (e.g. `<PAIR>.m5xp.v1` + the ACI gate). Run these levers ON the certified book(s), and evaluate
their COMBINATIONS, before declaring an edge final.** Each is a wrapper/objective/ensemble change, not a new
feature set; the info-bound caps raw AUC, so success is measured in **binding-year win-rate, coverage, and CPCV
path-clear-rate**, not global AUC. (EURUSD 5m results in parentheses — retarget via MX_HOR/HS.)
| # | Improvement lever | Script | What it does | Tgt | Prior / 5m result |
|---|---|---|---|---|---|
| I1 | **Adaptive-conformal (ACI) gate** | `m5_conformal.py` | online threshold targeting a win-rate w*; trades more in-regime, less off-regime (causal) | gate | **WIN @5m**: binding 2025 .584@n764 vs fixed .579@n618, +36% trades, regime-robust → book `<PAIR>.m5xp_aci.v1` |
| I2 | Seed-ensemble net ⊕ GBM (decorrelated stack member) | `m5_deep_ens.py` | M-seed AdamW MLP, prob-avg, blend/stack with GBM | D | 5m: decorrelated (corr .694) but 50/50 blend ≈ GBM (try LEARNED stack weight) |
| I3 | \|return\|-weighted / GMADL loss | `m5_magweight.py`, `min1_magweight.py` | up-weight large-move bars (sign more predictable) — magnitude→direction bridge | D | 5m: rebalances to two-sided ~.56, collapses 2026 UP (regime-dependent sign); GMADL operating-point untried. **[EURUSD·60s] NULL** (`min1_magweight.py`: best_iter=8/4000, no learnable signal; DOWN .507) + the explicit mag→dir bridge (`min1_magdir.py`) confirms sign-invariance at the operating point (magnitude finds big moves, zero direction) |
| I4 | Calibration (temperature/Venn-Abers) + selective threshold | (wrap any book) | calibrate probs so the confidence gate is honest; re-derive gate post-calibration | gate | nearly free; required wrapper for any confidence-gated edge |
| I5 | Cross-pair POOLING (weight-shared net across pairs) | (new) | train all majors as rows in one net + per-pair instance-norm + pair-id (Sirignano-Cont) | D | untested structural lever; run only if I2 shows life |
| I6 | AdamW + tuned LR + Optuna TPE/Hyperband | (wrap nets) | n_trials capped & LOGGED as multiplicity; select on worst-VAL-half | tuning | fixes the `Adam lr1e-3 single-seed` anti-patterns |

**COMBINATIONS are first-class rows.** The model space is a CROSS-PRODUCT: {base GBM · cross-pair · cross-horizon
stack · seed-ensemble · pooled} × {fixed gate · ACI gate · calibrated gate} × {BCE · \|ret\|-weighted/GMADL} ×
{up-filter · down-filter · specialist}. A future agent must benchmark a new edge against the BEST combination,
and try novel combinations (e.g. cross-pair book + GMADL loss + ACI gate + seed-ensemble), not just single
methods. Record each combination as its own ledger row with the incumbent it beat. Lit-review toolkit + saved
papers: `/home/sean/git/academic-papers/_DL_for_5m_FX_direction_REVIEW.md`.

---
### Sweep accounting rules
- One ledger row per **method × variant combination** you actually run. Expand the variant axes — e.g. A1
  with 3×3×2 knobs = up to 18 rows (prune obviously-dominated knobs, but log the prune).
- Each row records: combined per-year + CI, UP-split, DOWN-split, falsifier verdict, result-JSON path.
- **Don't relitigate known nulls blindly** at a new timeframe — but DO run them once per new (currency,
  timeframe), because nulls are horizon- and regime-specific (the whole point of the sweep). Use a cheap,
  fast-KILL falsifier for ~null-prior rows.
- Stop a row early the moment its pre-registered falsifier fires; record and move on.
- The sweep is **done** when every row is `done`/`killed`, **AND** the Tier-I improvement levers + their
  combinations have been run on the best edge(s) found, **AND** the discovery loop is dry. Then write the final
  best-UP and best-DOWN per the `(currency, timeframe, side)` keys and freeze the survivors — including the
  best COMBINATION (e.g. `<PAIR>.<book>_aci.v1`) — as books.
- **Incumbent = best COMBINATION, not base model.** When benchmarking a new method/timeframe, the number to beat
  is the best combination already in MODEL_REGISTRY.md / INDEX.json (e.g. cross-pair book + ACI gate), not the
  old base GBM. A future agent evaluating a neighboring timeframe (e.g. 10m after 5m) must retarget the Tier-I
  levers + the certified combinations to it and compare against ALL of them.
