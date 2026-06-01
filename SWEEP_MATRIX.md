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
| … | _(loop-until-dry: stop discovering only after K rounds with no novel survivable idea)_ | | | | | |

Candidate combination seeds (cheap novelty — start here): magnitude-conditioned cross-horizon stack ·
HMM-regime-gated CCM · RL (IQN+CVaR) sizing on the 15m book · online-adaptive meta-labeler · transfer-entropy
/ directed-information coupling gate · Hawkes up/down arrival-intensity imbalance · twin/seasonal-surrogate
significance on every microstructure feature · causal-discovery (PCMCI) lead-lag across the 7 majors.

---
### Sweep accounting rules
- One ledger row per **method × variant combination** you actually run. Expand the variant axes — e.g. A1
  with 3×3×2 knobs = up to 18 rows (prune obviously-dominated knobs, but log the prune).
- Each row records: combined per-year + CI, UP-split, DOWN-split, falsifier verdict, result-JSON path.
- **Don't relitigate known nulls blindly** at a new timeframe — but DO run them once per new (currency,
  timeframe), because nulls are horizon- and regime-specific (the whole point of the sweep). Use a cheap,
  fast-KILL falsifier for ~null-prior rows.
- Stop a row early the moment its pre-registered falsifier fires; record and move on.
- The sweep is **done** when every row is `done`/`killed`. Then write the final best-UP and best-DOWN per the
  `(currency, timeframe, side)` keys and freeze the survivors as `<PAIR>.<book>.v1` books.
