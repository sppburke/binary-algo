> **SCOPE: GENERIC** (currency/timeframe-agnostic). Per-key numbers, if cited, are tagged [PAIR·tf] examples whose record-of-truth is the Tier-2 file. See REPO_MAP.md.

# SWEEP MATRIX — the exhaustive permutation menu for a (currency, timeframe) strategy search

**NEW UNTRIED-METHODOLOGY BACKLOG (2026-06-07): `NOVEL_METHODS_RESEARCH.md`** — ranked runnable Tier-N candidates from the
110-candidate research slate (path-signatures/kernel, frac-diff, information-driven bars, HAR-RV-J/realized-GARCH/semivariance,
FASCL contrastive, HAVOK/Hankel-DMD, TDA gates, causal-PCMCI lead-lag, BOCPD). Pull rows from there into per-key sweeps; each
is gated by the frozen-past forward holdout + surrogate-null (leakage trap #9).

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
| A6 | Cross-pair USD-residual / lead-lag / dispersion | `m5_xpair.py` (`MX_HOR`, modes xp/xpbase/xpof) | mode {xp, xpbase, xpof}, basket {6-major}, residual {reversion, catch-up, lead-lag} | D | **HORIZON-GATED edge gradient: none@60s → UP@5m → BOTH@10m & 15m.** Cert both sides refit-CPCV @10m ([EURUSD·10m] UP p10 .586/DOWN p10 .568, `m10_xpair_cpcv.py`) + @15m; UP-only @5m; **null <5m** ([EURUSD·2m] p10 .51). Informed/jump component (esp. DOWN) averages out as horizon lengthens → cross-pair carries SIGN ≥10m. THE keystone direction lever ≥5m; CONCURRENT cross-pair (lagged dominated, [EURUSD·10m] `m10_leadlag` AUC .517). |
| A7 | Walk-forward retrain (regime robustness) | `m{5,10,15}_walkforward.py` | gap {1yr}, retrain-window {expanding, rolling} | D | low (often worse) |
| A8 | Up/down side asymmetry: FILTER vs SPECIALIST | `min1_updown.py` (filter), `min1_upspec.py` (specialist) | side {up, down} × {filter-on-symmetric, separately-trained} | U/Dn | filter med, spec ~null |
| A9 | **DST-correct SESSION segmentation** (conditioning AXIS on EVERY method) | `sessions.py` + `session_xpair.py` (xpair), `session_bars.py` (bar GBM, any H), `session_{1,2}m.py` (tick GBM) | session {NY 8-17 ET, LDN 8-16 London, Asia 9-18 Tokyo} × {filter decision-rows (GBM/xpair) \| strict session-only input (Kronos FT)} | D+M+G | **HIGH (NY) — direction edge is NY-CONCENTRATED.** [EURUSD] cross-pair certs BOTH sides EVERY H in NY, NONE in LDN/Asia (10m NY UP .6053/DOWN .5896 15/15, 15m .5845/.5712, 30m .5681/.5639; LDN/Asia .49-.53) and NY > legacy fixed-UTC gate. Base-feat dir KILLED all sessions; magnitude certified all sessions (magAUC .675-.80). `session_xpair_{2,5,10,15,30}m_{ny,ldn,asia}_result.json`, `session_{5,10,30}m_{dir,mag}_*`, `session_{1,2}m_*`. (COMPLETE: 2m NY .564/.560 + 5m NY .596/.588 both-sides certify — 2m+5m DOWN NEW; 15m base GBM null all sess .520/.515/.512) |

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
| C6 | SSA / fractional-diff / particle-filter / reservoir (ESN) | `usdjpy_2m_esn.py`, backlog (`IDEAS_LOG.md` E.19-22) | — | D/M | **[USDJPY·2m] reservoir KILLED** (`usdjpy_2m_esn.py`: 64-unit windowed ESN feats → pooled GBM; VAL .5228 not >base .524; held-out worse; no intra-window path-order the GBM misses — confirms GRU). Reservoir = strictly-weaker random GRU |

## Tier D — sequence / deep / RL
| # | Family / method | Script | Variant axes | Tgt | Prior |
|---|---|---|---|---|---|
| D1 | 1D-CNN / GRU on raw path | `m_cnn.py`, `min1_v14_cnn.py`, `exp_seq.py` | window-W {20,40,80}s, channels, arch {cnn,gru} | D | ~null |
| D2 | Neural-CDE on irregular Δt path | `min1_ncde.py` | irregular vs grid ablation, hidden {16,24,32} | D | ~null |
| D3 | TabNet attentive tabular | `exp_tabnet.py` | n_steps, width | D | ~null |
| D4 | DRL DQN direction-with-abstain | `min1_drl.py` | reward {deriv-settle}, abstain-cost, ε-schedule | D+policy | ~null |
| D5 | IQN / distributional + CVaR abstain | `min1_drl.py` | quantiles, CVaR-α, mag-gate | D+sizing | ~null (sizing only) |
| D6 | TS foundation model (Kronos zero-shot / Chronos·Moirai·TimesFM fine-tune) | KILLED-on-lit @5m (`sweeps/EURUSD_5m.md`; arXiv:2511.18578 — fine-tune deteriorates, no economic gain; daily-equity, MSE-sign-invariant, CPU-OOM) | sampling N | D | skip-on-lit; superseded by D7 RUN |
| D7 | **Kronos K-line foundation model — direction (alignment-corrected)**, zero-shot AND fine-tune, multi-TF | `kronos_mtf.py` (corrected FM-F: predict H/GRID FORWARD steps, Pup=pred_close(+H)>C[i]); `kronos_ft.py` (single-proc GPU predictor FT, frozen tokenizer, AMP bf16); `kronos_bars.py` (fwd deriv-label builder) | mode {native zero-shot, fine}, H {1,5,10,15,30}m, session {NY,LDN,Asia} | D | **RUN→NULL every H, zero-shot AND fine-tuned, all sessions** (pooled .50-.51, CPCV p10 .489-.500, up-rates in-band; ALL KILLED). Even at NY≥10m where xpair certs .57-.61, Kronos reads ~.50 — ingests only EURUSD OWN OHLCV, not the 7-pair USD cross-section; FT did NOT help. Corrects the legacy look-forward bug (`kronos_dir.py` gated `KRONOS_DIR_LEGACY=1`). `kronos_dir_mtf_*_result.json`. (COMPLETE: FINE "up the chain" 1m→2/5/10m & 5m→10m all KILLED p10 .481/.499/.475/.497; finer input grid never helps) |
| D8 | **Multi-timeframe Kronos vote-ensemble** (combine per-TF fine forecasts) | `kronos_ensemble.py` (npz vote-combine across `kronos_mtf.py` TFs) | TF set {1,5,10,15,30}m, vote {mean-prob, majority}, session | D | **RUN→KILLED/ABORT.** 5m ensemble (fine1+native) pooled .531 but CPCV p10 .428, no year CI95-lo≥.541 → KILLED; 10m ensemble only **12** common decision bars across 3 grids → ABORT. LIMITATION: nonoverlap-chrono at different grids leaves too few shared decision instants (5m n_common 267→147; 10m n=12). Combining single-pair views ≠ cross-section. `kronos_dir_mtf_ens_*_result.json` |
| D9 | **Per-session bar-image CNN** (Sezer CNN-BI under session filter) | `barcnn_run.py SESSION` (2-D OHLC image-conv, +SESSION arg) | session {NY,LDN,Asia} × variant {hist, ohlc, gaf} | D+M | **RUN→DIRECTION NULL all sessions** (NY VAL .510/test .503/oos .505; LDN .509/.499/.503; Asia .501/.506/.500; cov2-10% per-yr ~.50-.53, none robustly clear .541). 4th model class confirms single-pair candle images carry no sign per session. `barcnn_hist_{ny,ldn,asia}_result.json` |

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

> **Discovery-vetting outcome — BAR / CANDLESTICK PATTERN recognition family (generic, 2026-06-05):** corpus has the
> canonical papers (`OA_Sezer_AlgorithmicFinancialTrading_StockBarChartImageCNN` = CNN-BI 2-D OHLC bar-chart-image CNN;
> GAF/Markov-Transition-Field→2-D-CNN; Kronos K-line foundation-model tokenizer; SACLSTM). VET: (i) explicit candlestick
> FEATURES (body/wick/range, doji/engulfing/pin) = SUBSUMED — they are functions of O/H/L/C the GBM already accesses via
> `rangepos`/`atr_pct`/`gap_prev`/`bb_width`/returns (same re-representation kill as ESN/GRU); and classically gate move
> SIZE not SIGN (cf. `OA_Singha_HiddenOrderEntropy_MagnitudeNotDirection`; most named-pattern results die under
> deflated/purged CV — Bailey/Lopez-de-Prado corpus). (ii) The ONE non-subsumed sub-lever = **2-D CNN over rendered OHLC
> bar IMAGES** (Sezer CNN-BI) / Kronos — a conv learns LOCAL 2-D pattern detectors a per-bar GBM and a 1-D GRU cannot
> represent (GRU+ESN were tested, a 2-D image-conv was NOT). Prior LOW (corpus grades it `low`; FX-dir CNNs land ~.52-.55).
> **DATA GATE:** requires O/H/L — present only as `ohlc_cache/EURUSD_5m_*` + rebuildable from EURUSD ticks (`vbars.py` src);
> **NOT computable for USDJPY (feature store keeps only `close`; no USDJPY tick/OHLC on disk)** → for USDJPY it is an
> external-data lever. Runnable only on EURUSD. Per-key: `sweeps/USDJPY_2m_backlog.md` external frontier; `IDEAS_LOG.md`.
> **RUN [EURUSD·60s] 2026-06-05 → DIRECTION KILLED / MAGNITUDE clears >65% (`METHODS_CATALOG.md` §5.5, `EURUSD_RESULTS.md` row 22, `MAGNITUDE_FINDINGS.md` §3).** Built 1m OHLCV from ticks
> (`barcnn_bars.py`) + faithfully reimplemented the Sezer CNN-BI image-CNN (`barcnn_run.py`) on the 60s wc_ret label,
> 3 variants: **hist** (Sezer close-histogram), **ohlc** (3ch wick+up/down-body), **gaf** (GASF+GADF). ALL null: VAL
> dirAUC .499/.503/.502, test/oos AUC ≈.50, and **faithful CPCV (`barcnn_cpcv.py`, 28 purged paths) path_p10 .484–.499
> with frac_paths_clear_0.541 = 0.0 at every coverage** → trips all 3 falsifier conditions. The 2-D image-conv (the last
> non-subsumed bar sub-lever) confirms the bar geometry carries SIZE not 60s SIGN (even the GADF antisymmetric sign-field
> is null) — sign-invariance @60s. **Bar/candlestick 2-D-image family now RUN + EXHAUSTED on-disk for EURUSD ≤60s DIRECTION.**
> **BUT the SAME bar-image CNN pointed at the MAGNITUDE outcome CLEARS >65% CPCV-certified** (`barcnn_mag.py ohlcabs`,
> absolute-scale OHLC image: magAUC .699/.714/.686, selective large-call precision .68→.80, **all 28 purged paths ≥0.65
> at cov≤0.2 in 2024/2025/2026**; `barcnn_mag_ohlcabs_result.json`, `MAGNITUDE_FINDINGS.md` §3). The single cleanest
> sign-invariance demo: one method, null on sign, >65% on size — and the key knob is preserving ABSOLUTE vol scale
> (per-window min-max only reaches .64). Kronos NOT built: its gains are RankIC/magnitude (no FX/60s/direction numbers)
> and fine-tune deteriorates (arXiv:2511.18578) → magnitude probe, not a direction lever (deprioritized).

> **Discovery-vetting outcome — DST-correct SESSION segmentation + Kronos look-forward fix + full-suite audit
> (generic, 2026-06-06):** two structural additions to the menu. **(A) DST-correct session segmentation as a
> conditioning AXIS on every method** (`sessions.py`: NY=8-17 America/New_York, LDN=8-16 Europe/London,
> Asia=9-18 Asia/Tokyo; `session_mask` = local-tz hour applied per-day across train+val+test+oos). Discipline:
> GBM/xpair use causal-CONTINUOUS features and filter DECISION ROWS only; Kronos FT uses strict session-only
> INPUT (filter-before-window + contiguity). RESULT — the certified ≥10m cross-pair direction edge is decisively
> **NY-CONCENTRATED**: `session_xpair.py` (strict session-only, gate per-H) certifies BOTH sides at EVERY horizon
> in NY, NONE in LDN/Asia — 10m NY UP .6053/DOWN .5896 (15/15), 15m NY UP .5845/DOWN .5712, 30m NY UP .5681/DOWN
> .5639; LDN/Asia .49-.53 throughout. **NY BEATS the legacy fixed-UTC gate** (10m legacy .586/.568, 30m .559/.553)
> → re-running A6/the certified combinations under an NY session filter is a strict improvement, not just a slice.
> Base-feature GBM DIRECTION is KILLED in every session at every freq (`session_{1,2}m`, `session_bars`; NY
> strongest e.g. 10m NY p10 .525), but MAGNITUDE is certified in EVERY session at EVERY freq (magAUC .675-.80,
> p10 .58-.80) — sign-invariance holds per-session. **(B) Kronos look-forward bug fix (FM-F forecast-derivation).**
> The legacy `kronos_dir.py`/`kronos_ft.py` eval scored Pup from a forecast of the bar INTO the entry (window
> [t[i-1],t[i]]), DISJOINT/off-by-one from the forward deriv label [t[i]+1,t[i]+61] — a misaligned forecast yields
> a FALSE NULL, never a false positive. FIXED in `kronos_mtf.py` (context ends AT bar i, predict H/GRID FORWARD
> steps, Pup=pred_close(+H)>C[i], pred-side contiguity + nonoverlap GAP=HS+TOL; forward label agrees next-bar sign
> 92.3%, n=233,950); `kronos_dir.py` now gated behind `KRONOS_DIR_LEGACY=1`, legacy result superseded. A 19-agent
> full-suite correctness audit confirmed the bug is **ISOLATED to the Kronos family (2 scripts), NOT systemic** —
> no second FM-F instance across 308 scripts; every other forecast-derivation script (`usdjpy_{1m,2m}_statespace`,
> `usdjpy_2m_xhorizon`, `m5_xhorizon`, `m5_lossbatch`, `f1_compound`) predicts the FORWARD quantity over the SAME
> horizon at the SAME bar = correctly aligned; tick/bar/xpair substrates + 239 bar-features proven Tier-1 CAUSAL
> (truncation max|full-trunc|=0.0; label 0-mismatch independent recompute) → **no certified book invalidated**.
> **CORRECTED Kronos verdict: NULL at every horizon 1/5/10/15/30m, zero-shot AND fine-tuned, ALL sessions** (pooled
> .50-.51, CPCV p10 .489-.500, all KILLED, up-rates in-band) — even at NY≥10m where cross-pair certifies .57-.61,
> Kronos reads ~.50 because it ingests only EURUSD's OWN OHLCV candles, not the 7-pair USD cross-section that
> carries the edge; fine-tune did NOT help direction. The alignment-corrected null thus AGREES with the bar-image
> CNN null (`barcnn`) and the ESN/GRU subsumption: own-pair price geometry/forecasts carry SIZE not sub-30m SIGN.
> New scripts: `sessions.py`, `session_1m.py`, `session_2m.py`, `session_bars.py`, `session_xpair.py`,
> `kronos_ft.py` (single-process GPU adaptation of the DDP-only Kronos `finetune/train_predictor.py`),
> `kronos_mtf.py`, `kronos_bars.py`, `kronos_ensemble.py`, `barcnn_run.py`+SESSION arg. Files:
> `session_1m_{dir,mag}_{ny,ldn,asia}_result.json`, `session_2m_*`, `session_{5,10,30}m_{dir,mag}_{ny,ldn,asia}_result.json`,
> `session_xpair_{2,5,10,15,30}m_{ny,ldn,asia}_result.json`, `kronos_dir_mtf_*_result.json`. **CAMPAIGN COMPLETE** (all
> cells run): 15m base GBM null all sessions (NY p10 .520/LDN .515/Asia .512); `session_xpair` 2m & 5m NY both-sides
> certify (2m .564/.560, 5m .596/.588 — 2m & 5m DOWN are NEW certs); Kronos FINE "up the chain" (1m→2/5/10m, 5m→10m)
> all KILLED (p10 .481/.499/.475/.497); multi-TF vote-ensemble KILLED (5m p10 .428) / ABORT (10m n_common=12);
> per-session bar-image CNN null (NY .510/.503/.505, LDN .509/.499/.503, Asia .501/.506/.500).

> **Discovery-vetting outcome — USDJPY 1m (generic, 2026-06-04):** a fresh new-currency bootstrap (8-agent discovery
> R1 + 2-agent R2) confirms the **60s/1m near-efficiency keystone is currency-GENERIC, not EURUSD-specific** — USDJPY
> 1m direction is also near-efficient (dirAUC ~.514–.524; only signed structure = a thin ~.52–.54 dip-buy reversion
> below the 0.541 breakeven; magnitude STRONG magAUC ~.72–.79, sign-invariant). Two genuinely-new GENERIC levers were
> INVENTED + RUN here, now Tier-1 nulls at 1m (transferable): (i) **cross-pair ORDER-FLOW residual** (own signed OF −
> USD-up-basket OF from 6 majors' `features_of`; the untested delta vs returns-based xpair) — [USDJPY·1m] KILLED
> (`usdjpy_1m_ofresid_s8_result.json`, OOS DOWN .480); the cross-pair `none@60s` gradient holds in OF space too.
> (ii) **gotobi/Tokyo-fix calendar as GBM conditioning feature** — [USDJPY·1m] KILLED (`usdjpy_1m_gotobi_s6_result.json`,
> +.0004 AUC, flags rank #197/#245). Also confirmed at 1m: signed reversion/path-state feats SUBSUMED by base;
> regime-gating + model-confidence ANTI-transfers OOS (corr(VAL,OOS)=−.54); seed-ensemble can't close a ~1pp gap.
> Per-key evidence: `sweeps/USDJPY_1m{,_backlog}.md`, `USDJPY_RESULTS.md`.

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
| N4 | Market-Intraday-Momentum term-structure (lagged interval→fwd sign) | Gao-Han-Li-Zhou JFE2018 | signed directional autocorrelation conditioned on clock-time interval (never isolated as the predictor) | on-disk (minute) | ~10% | **[EURUSD·10m] RUN→KILLED** (`m10_intramom.py`: VAL AUC .526, 2025 UP .562/DOWN .506 < cross-pair cert; sign-invariant/dominated). See `sweeps/EURUSD_10m.md` |
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
| N17 | **Anti-contemporaneous lead-lag + transfer-entropy gate + RFF virtue-of-complexity** | Sirignano-Cont + Schreiber TE + Kelly-Malamud-Zhou VoC | strictly-LAGGED cross-pair (m5xp uses CONCURRENT) + directed-info gate + complexity ridge P≤1e5 — distinct sign-carrying channel/function-form | on-disk (7 legs + features_of) | ~13% | pending @5m; **[EURUSD·10m] RUN→KILLED bounded** (`m10_leadlag.py`: VAL AUC .517, 2025 UP .542/DOWN .519 < cert; lagged dominated by concurrent). See `sweeps/EURUSD_10m.md` |
| N18 | **Sign-COUPLED payoff objective** (GMADL/MADL loss + RRL diff-Sharpe tanh-position head) | GMADL lit + Moody-Saffell RRL | penalizes sign-WRONG large moves MORE (R·R̂ coupling) + realized-payoff-trained signed position — NOT BCE nor symmetric \|ret\|-weight (all killed) | on-disk (cross-pair) | ~10% | RUN@5m; **[EURUSD·10m] RUN→KILLED** (`m10_signedpayoff_gmadl.py`: all a∈{50,100}×b∈{1,2} give 2025 .44–.49 <0.5 — loss worse than gated BCE). See `sweeps/EURUSD_10m.md` |
| N19 | **H=5 risk-residual RELABEL + DOWN-split** | Capponi-Cont common-factor residual | DOWN target = sign(eu_fwd − β_t·USD-basket_fwd) at the never-run H=5, with the DOWN-split min1_residtarget omits | on-disk (7-major basket) | ~8% | RUN@5m; **[EURUSD·10m] RUN→KILLED @H=10** (`m10_residlabel_down_h10.py`: DOWN 2024 .577/2026 .577 but 2025 .5077 coin-flip < cert — residual sign collapses on 2025 wall). See `sweeps/EURUSD_10m.md` |
| N20 | **Antisymmetric cross-pair prediction-matrix component Aₐ** (signed lead-lag rotation feature) | discovery R1 (Giglio-Xiu adj.) | Aₐ=(A−Aᵀ)/2 of the lagged 7×7 cross-pair matrix = directional lead-lag own-pair can't self-replicate | on-disk (7 legs) | ~8% | **[EURUSD·30m] SUBSUMED (Tier-1 FI)** — lead-lag ll_ feats = 2.1% gain & already GBM inputs; Aₐ is a linear combo of them. `m30_xpair_featimp_result.json` |
| N21 | **IPCA time-varying USD-loadings** (instrumented betas, Kelly-Pruitt-Su) | discovery R1 | βᵢ,ₜ=zᵢ,ₜ′Γ linear in 239 feats, ALS on TRAIN; regime-adaptive EUR sign = β′λ — can flip loading sign across regimes | on-disk (7 legs + 239) | ~8% | **[EURUSD·30m] SUBSUMED (Tier-1 FI)** — refines the 2.0%-gain cross-pair-factor channel; edge is pooled BASE feats not factors; 2025-inversion rationale doesn't bind @30m (2025 = strong fwd yr). `m30_xpair_featimp_result.json` |
| N22 | **Month-end/quarter-end rebalancing-flow directional bias** (calendar gate) | Coval-Stafford JPE2007; discovery R2 | institutional period-end rebalancing forces signed FX flow independent of price feats | on-disk (timestamps) | ~4% | **[EURUSD·30m] KILLED** (`m30_monthend.py`: ME up-rate 2024 .494/2025 .504/2026 .498 sign-flips, never exits [.47,.53]; hour-seasonality already #1/#3 feat subsumes it) |

> **NOVEL-METHODS CAMPAIGN rows (2026-06-07): `NOVEL_METHODS_RESEARCH.md` §7 slate.** Phase-1 reusable infra committed 051e525
> (`build_panel.py`→7-pair USD return panel CONSTRUCTION-FAITHFUL vs certified `build_xp`, BIT-IDENTICAL channels; `fwd_holdout.py` =
> MANDATORY frozen-past forward-holdout gate, train≤2023→per-year 2024/25/26, NOT pooled CPCV which is leakage trap #9;
> `surrogate_null.py` = phase-randomize/IAAFT gate). Every row below is forward-holdout-gated. Ground truth: `CAMPAIGN_2026-06-07_FACTS.md`.
>
> **CONVERGENT VERDICT (Phase 4, 2026-06-07): every cross-sectional DIRECTION family KILLED on the forward holdout.** FFD (N24/T2)
> + signature lead-lag (N26/D1) + HAVOK (N27/D6) + signed-semivariance (N31/D7) all fail to beat/match the certified cross-pair book,
> and the BASE book itself DECAYS forward (15m .5895/.5518/.5234; 30m .5827/.5505/.5055 — the documented refit-dependence). The D7
> nuance matters: D7 is the ONLY shot that PASSES its mechanism null (signed-vol asymmetry genuinely carries directional content,
> semivaronly beats its sign-flip shuffle by +.015/+.030) yet is REAL-but-SUB-BREAKEVEN, decaying, non-additive — distinct from
> D1/D6 which FAIL their mechanism null. Corroborates the standing program conclusion: **direction beyond the engineered cross-pair
> book is EFFICIENT on existing data; the only frontier is EXTERNAL data** (rate-diff / implied-vol / EURGBP ticks). Nothing certified;
> UP/DOWN leaderboard UNCHANGED. Remaining slate scoped by EVIDENCE (resumable, deliberate-not-blind): N25/T1 needs-EXTERNAL-data;
> N28/D2 + N29/D4 + N39/M4 BLOCKED-no-GPU-or-headers; N30/D5 + N32/D8 REASONED-SKIP; N33–N35/G1-G3 MOOT-no-signal-to-gate;
> N36-N38/T3-T5 + N40-N41/M5-M6 low-prior-queued. Per-row evidence in the Status column.

| # | Discovered method / combination | Source (cite) | Mechanism (why it could carry SIGN/SIZE) | Data | Prior | Status |
|---|---|---|---|---|---|---|
| N23 | **HAR / realized-vol MAGNITUDE canon** (multiscale-RV, bipower-jump, signed-semivar, realized-quarticity, +all stacked) | Corsi HAR-RV; Barndorff-Nielsen-Shephard; Patton-Sheppard | richer realized-vol decompositions added to certified mag base `[-pe,rv30,rv120]`; target \|ret_H\|≥train-Q75, horizons 10/15/30m | on-disk (panel RV) | med (mag, certified family) | **RUN→KILLED (REAL-but-SUB-BAR, FORWARD-CONSISTENT).** `mag_har.py`/`mag_har_result.json` (MAGNITUDE_FINDINGS §6g, committed 7d45bf8). Falsifier = +0.005 AUC in ≥2 fwd yrs AND ≥2 horizons. +har mean fwd ΔAUC +0.0010 (2/3); +jump +0.0003 (1/3); +semivar −0.0000 (1/3); +harq +0.0004 (2/3); **+all (stacked) +0.0021, 3/3 deployable (NO decay) — forward-consistent but economically negligible** (base rv already extracts ~all magnitude; only signed-jump SJ120 orthogonal −.04 carries nothing). Bonus: RE-VALIDATES the certified magnitude edge on a clean fwd holdout — base AUC .799/.750/.750 (10m) .791/.739/.737 (30m), 4–5.6× decile lift, NO decay. Magnitude path EXHAUSTED on-disk. |
| N24 | **Fractional differentiation (FFD) → DIRECTION** (T2) | López de Prado AFML ch.5; hand-rolled fixed-width FFD + ADF d*-selection | "stationarity with memory" — preserve long-memory level info (98.8% vs 0.9% for plain returns) the differenced book discards, fold USD factor/residual/lead-lag FFD into certified direction book | on-disk (panel) | low-med | **RUN→KILLED both 15m & 30m.** `frac_diff.py` (self-check: RW needs d*=0.1, p .019); `frac_direction.py`/`frac_direction_{15m,30m}_result.json`. Per-pair d* {EUR/NZD/CHF=0.1, GBP/AUD/JPY/CAD=0.2}, thresh 1e-4, NY cov0.10 selacc. **15m:** base 2024 .5900/2025 .5520/2026 .5254 (pooled .5642); +ffd .5792/.5301/.5403 (pooled .5537) → DECAYS (Δ −.0108/−.0219/+.0149), deployable=false; ffdonly pooled .5001 = coin-flip. **30m (harder):** base 2024 .5842/2025 .5483/2026 .5007 (pooled .5558); +ffd .5546/.5135/.5064 (pooled .5308) → DECAYS (Δ −.0296/−.0348/+.0057), deployable=false; ffdonly pooled .4958 (sub-coin-flip). FFD level-memory adds nothing to direction, hurts recent years. (Note: base book itself DECAYS .59→.55→.525, consistent w/ documented refit-dependence of the cross-pair edge.) |
| N25 | **T1 information-driven bars** (volume/dollar/imbalance bars on 7-pair panel) | López de Prado AFML ch.2 | non-time-clock bars normalize information arrival → may expose signed structure time-bars smear | substrate scouted: `features_tick/*_1s.parquet` EURUSD-ONLY 2021+ → cross-pair sync needs proxies for other 6 pairs | low | **needs-EXTERNAL-data (NOT run).** Only EURUSD 1s on disk (2021+); the whole point (cross-pair synchronization) requires the other 6 pairs' 1s/tick data = external acquisition. Harness NOT built. Falsifier (unchanged): clears book p10 CI95 AND survives surrogate-null. |
| N26 | **D1 lead-lag SIGNATURE (depth-2)** | path-signatures lit; Lévy-area lead-lag | level-2 iterated integrals S^{ij}=∫∫dX^i dX^j + Lévy area A^{ij}=½(S^{ij}−S^{ji}) between EURUSD & each peer over trailing 30-bar window (signed lead-lag / quadratic covariation); pure-numpy cumsum (iisignature won't compile) | on-disk (panel) | ~null | **RUN→KILLED both 15m & 30m (fails its mechanism-specificity null).** `xsec_direction.py` (fam=sig); `xsec_direction_sig_{15,30}m_result.json`. Forward holdout, NY cov0.10 selacc; base book DECAYS forward (15m .5895/.5518/.5234; 30m .5827/.5505/.5055). **15m:** +sig DECAYS (Δ −.0043/−.0099/+.0032); sigonly pooled .5243 ≈ sig_shuf .5236 → rotation surrogate does NOT degrade it ⇒ NOT genuine lead-lag (fails the falsifier). **30m:** +sig DECAYS all 3 yrs (−.0094/−.0043/−.0133); sigonly pooled .5283 (sub-base). Signature content is not specific lead-lag. |
| N27 | **D6 HAVOK (Hankel-Koopman)** | Brunton et al. HAVOK | frozen-basis: delay-embed (q=60) USD-factor trend, SVD on TRAIN → freeze r=8 modes, causally project to v1..v7 + intermittent FORCING v_r (signed precursor) + leading phase | on-disk (panel) | ~null | **RUN→KILLED both 15m & 30m (sub-breakeven).** `xsec_direction.py` (fam=havok); `xsec_direction_havok_{15,30}m_result.json`. Forward holdout, NY cov0.10 selacc. **15m:** +havok DECAYS (Δ −.003/−.0024/+.0014); havokonly pooled .5148 — sub-breakeven, far below base .5624; the forcing only MARGINALLY beats its phase-randomized surrogate (.5148 vs .5063) and never clears .541. **30m:** +havok DECAYS; havokonly pooled .5197 (sub-base). Signed Koopman forcing carries no deployable sign. |
| N28 | **D2 untruncated signature KERNEL** | sig-kernel lit (Salvi et al.) | full (untruncated) path-signature kernel between EURUSD & peers — richer than depth-2 truncation (D1) | on-disk (panel) | ~null | **BLOCKED-no-GPU-or-headers (NOT built).** Needs sigkernel/KeOps GPU + a C-extension build that FAILS here (no `Python.h`). Deferred to a GPU+headers env; low prior given D1 (signatures, depth-2) is already null. Falsifier (unchanged): beats book p10 CI95 + shuffle-degrades. |
| N29 | **D4 FASCL future-aligned contrastive** | FASCL contrastive-learning lit | future-aligned self-supervised contrastive representation of cross-sectional path; learned embedding fed to direction head | on-disk (panel) | ~null | **BLOCKED-no-GPU-or-headers (NOT built).** FASCL encoder needs an 8GB GPU (unavailable here); deferred to a GPU env; low prior given D1 (signatures) already null. Falsifier (unchanged): beats book p10 CI95 + shuffle/surrogate-null. |
| N30 | **D5 causal lead-lag** (PCMCI / Granger-FDR / structural-VAR) | Runge PCMCI; Granger; structural-VAR | conditions OUT the common USD factor before scoring a lagged directional link (direct lead vs common-driver) | on-disk (panel) | ~null | **REASONED-SKIP (NOT built; statsmodels present).** The certified base book ALREADY contains every peer's lagged lead-lag feature (`ll_<pair>k`, k∈{1,3,5,10,15,30}) that the GBM weights, and D1 proved signature lead-lag content does NOT survive a rotation null → a Granger feature-SELECTION on top of an already-lead-lag-saturated GBM has near-zero marginal prior. Re-open only with EXTERNAL leaders (rate-diff), not more EURUSD-panel selection. Falsifier (if re-run): lagged signed link beats book p10 CI95 + shuffle-degrades. |
| N31 | **D7 signed-semivariance DIRECTION** | Patton-Sheppard semivariance | RS⁺−RS⁻ signed skew as a DIRECTION feature (not magnitude) into the cross-pair book | on-disk (panel) | ~null | **RUN→KILLED, but REAL-but-SUB-BREAKEVEN (the ONE shot that PASSES its mechanism null).** `xsec_direction.py` (fam=semivar); `xsec_direction_semivar_{15,30}m_result.json`. Forward holdout, NY cov0.10 selacc. semivaronly BEATS its sign-flip control semivar_shuf by +.015 (15m) / +.030 (30m) → the signed-vol asymmetry genuinely carries DIRECTIONAL content. **BUT sub-breakeven & non-additive:** semivaronly pooled .5258 (15m) / .5382 (30m), below breakeven .541 (only 2024@30m .5516 clears), DECAYS forward, and +semivar does NOT add to the base book (Δ15m −.0154/−.0029/−.0107; Δ30m −.0081/+.0062/−.0139). The DIRECTION analog of magnitude's "real-but-sub-bar" — genuine signed content, sub-deployable, already subsumed by the cross-pair book. Distinct from D1/D6 (which FAIL their mechanism null). |
| N32 | **D8 quantile-direction baseline** | quantile-regression baseline | conditional-quantile crossing as a direction baseline / reference for the path families | on-disk (panel) | ~null | **REASONED-SKIP (NOT built).** A CONTROL, not an edge candidate — only needed to ablate a TSFM-quantile claim, which this campaign does not make. Falsifier (if re-run): beats book p10 CI95. |
| N33 | **G1 persistent-homology corr-cloud GATE** | TDA / persistent homology | topological features of the rolling 7-pair correlation point-cloud as a regime gate on the certified book | on-disk (panel) | ~null (gate) | **MOOT-no-signal-to-gate (NOT built; also gudhi-unbuildable, no Python.h).** A gate conditions a SURVIVING signal; no direction signal survived the forward holdout (FFD/D1/D6/D7 all killed), so there is nothing to gate. Re-open only if a future (external-data) signal clears breakeven first. Falsifier (if re-run): gate lifts book p10 in ≥2 fwd yrs + survives surrogate-null. |
| N34 | **G2 BOCPD / HMM change-point GATE** | Adams-MacKay BOCPD; HMM | online change-point / regime detection (ruptures) gates trading to in-regime bars on the certified book | on-disk (panel) | ~null (gate) | **MOOT-no-signal-to-gate (NOT built; ruptures IS available).** A gate conditions a SURVIVING signal; no direction signal survived the forward holdout, so there is nothing to gate. Re-open only if a future (external-data) signal clears breakeven first. Falsifier (if re-run): gate lifts book p10 in ≥2 fwd yrs (must flag known kills non-significant). |
| N35 | **G3 windowed-DMD residual GATE** | Dynamic Mode Decomposition | windowed DMD spectral residual as a regime/anomaly gate on the certified book | on-disk (panel) | ~null (gate) | **MOOT-no-signal-to-gate (NOT built).** A gate conditions a SURVIVING signal; no direction signal survived the forward holdout, so there is nothing to gate. Re-open only if a future (external-data) signal clears breakeven first. Falsifier (if re-run): gate lifts book p10 in ≥2 fwd yrs + survives surrogate-null. |
| N36 | **T3 vol-time SUBORDINATION** (transform) | subordinated-process lit (Clark) | re-clock the path by realized-vol time before feeding the direction book (may expose signed structure clock-time smears) | on-disk (panel) | ~null | **low-prior-queued (NOT built; transform).** A MAGNITUDE transform, but the HAR canon (N23/§6g) showed base rv already extracts ~all magnitude → low marginal prior. Catalogued as queued; not run this campaign. Falsifier (if run): transformed input beats book p10 CI95 + survives surrogate-null. |
| N37 | **T4 cross-pair WHITENING** (transform) | whitening / decorrelation | whiten the 7-pair return covariance before scoring direction (remove common-factor masking of idiosyncratic sign) | on-disk (panel) | ~null | **low-prior-queued (NOT built; transform).** A DIRECTION transform, but direction beyond the engineered book is shown EFFICIENT (4 convergent nulls) → low marginal prior. Catalogued as queued; not run this campaign. Falsifier (if run): whitened input beats book p10 CI95 + shuffle-degrades. |
| N38 | **T5 Hilbert PHASE** (transform) | Hilbert transform / analytic-signal phase | instantaneous-phase / phase-coherence features as direction inputs | on-disk (panel) | ~null | **low-prior-queued (NOT built; transform).** A DIRECTION transform, but direction beyond the engineered book is shown EFFICIENT (4 convergent nulls) → low marginal prior. Catalogued as queued; not run this campaign. Falsifier (if run): phase feature beats book p10 CI95 + survives surrogate-null. |
| N39 | **M4 TDA-Wasserstein** (magnitude exotica) | TDA persistence + Wasserstein | Wasserstein distance between persistence diagrams of rolling windows as a magnitude feature | on-disk (panel) | ~null (mag) | **BLOCKED-no-GPU-or-headers (NOT built; gudhi-unbuildable, no Python.h).** Also low prior after the HAR canon (N23/§6g) showed the realized-vol family adds only a forward-consistent sliver over base rv. Falsifier (if built): +0.005 AUC in ≥2 fwd yrs AND ≥2 horizons + survives surrogate-null. |
| N40 | **M5 multiscale-ECC** (magnitude exotica) | Euler-characteristic curve / multiscale topology | multiscale Euler-characteristic-curve features for magnitude | on-disk (panel) | ~null (mag) | **low-prior-queued (NOT built; surrogate-null-gated, magnitude track).** Low prior after the HAR canon (N23/§6g) showed the realized-vol family adds only a forward-consistent sliver over base rv. Falsifier (if built): +0.005 AUC in ≥2 fwd yrs AND ≥2 horizons + survives surrogate-null. |
| N41 | **M6 MOMENT foundation model** (magnitude exotica) | MOMENT TS foundation model | zero-shot/fine-tuned MOMENT embedding as a magnitude feature | on-disk (panel) | ~null (mag) | **low-prior-queued (NOT built; surrogate-null-gated, magnitude track).** Low prior after the HAR canon (N23/§6g) showed the realized-vol family adds only a forward-consistent sliver over base rv. Falsifier (if built): +0.005 AUC in ≥2 fwd yrs AND ≥2 horizons + survives surrogate-null. |
| N42 | **FAM1 N-BEATS / N-HiTS path-forecast → sign** (neural decomposition forecaster, basis-expansion path→close→sign) | directional-prediction + n-hits external repos; Oreshkin N-BEATS, Challu N-HiTS | multi-rate basis-expansion path forecast; sign read from forecast(+H)>close — a neural analog of the Kronos/path forecasters | on-disk (panel) | 0.03 | **RUN→KILLED all 6 tf, single+cross-pair (deriv-faithful, GPU).** Retarget knob `MX_HOR=<min>`; `nbeats_nhits_dir.py`. Confirms the sign-invariance theorem (path forecast carries SIZE not SIGN; cf. prior N-HiTS@5m null). Falsifier: VAL dirAUC≤0.515 OR no held-out yr selacc CI95-lo≥0.541. [EURUSD·1-30m → `EURUSD_RESULTS.md` §2026-06-08; `neural_spectral_dir_sweep_result.json`, `nbeats_nhits_dir_<tf>m_result.json`]. |
| N43 | **FAM2 DLinear / Autoformer / FEDformer-freq / TFT-quantile-fan → sign** (decomposition/attention forecasters) | directional-prediction external repo; Zeng DLinear, Wu Autoformer, Zhou FEDformer, Lim TFT | trend/seasonal & frequency-domain decomposition + quantile fan; sign from forecast(+H)>close | on-disk (panel) | 0.03 | **RUN→KILLED all 6 tf, single+cross-pair (deriv-faithful, GPU).** Retarget knob `MX_HOR=<min>`; `decomp_dir.py`. Sign-invariance: decomposition forecasters carry move SIZE not SIGN. Falsifier: VAL dirAUC≤0.515 OR no held-out yr selacc CI95-lo≥0.541. [EURUSD·1-30m → `EURUSD_RESULTS.md` §2026-06-08; `neural_spectral_dir_sweep_result.json`, `decomp_dir_<tf>m_result.json`]. |
| N44 | **FAM3 causal DWT + SSA band-split → GBM/literal recombine → sign** (spectral band decomposition) | n-hits / spectral-decomposition lit; Mallat DWT, Vautard-Ghil SSA | causal wavelet + singular-spectrum band-split, per-band GBM or literal recombine to a forecast; sign from recombined path | on-disk (panel) | 0.03 | **RUN→KILLED all 6 tf, single+cross-pair (deriv-faithful, GPU).** Retarget knob `MX_HOR=<min>`; `spectral_dir.py`. Flashiest held-out cell is a thin-coverage mirage (CI95-lo fails). Sign-invariance: spectral bands carry SIZE not SIGN. Falsifier: VAL dirAUC≤0.515 OR no held-out yr selacc CI95-lo≥0.541. [EURUSD·1-30m → `EURUSD_RESULTS.md` §2026-06-08; `neural_spectral_dir_sweep_result.json`, `spectral_dir_<tf>m_result.json`]. |
| N45 | **EURGBP direct cross-rate as external GBM feature** (triangular price-discovery, independent signed series) | ScienceDirect 2023 triangular-arbitrage / price-discovery paper; Froot-Ramadorai (2005) cross-rate microstructure | EURGBP is priced by a separate dealer network; its signed return carries information about EUR and GBP supply/demand NOT captured by the algebraic USD-implied cross — direct cross-rate price discovery is a genuinely independent signal channel vs. N2 (which uses only on-disk EURUSD+GBPUSD algebraically) | **external** — EURGBP Dukascopy 15m bars (free download); also available via `processed/` if sibling pair was fetched | **~18% (top-external)** | **pending** (external data acquisition needed; distinct from N2 which is algebraic residual on-disk) |
| N46 | **UK macro surprise signed pre-release drift — intraday 30-min window** (BoE MPC + ONS CPI/GDP releases) | Kurov et al. JFQA 2019 "Do Financial Markets Respond to…" + Andersen-Bollerslev-Diebold-Vega AER 2003 | informed private positioning ~30 min before scheduled BoE MPC (12:00 LDN) and ONS data (07:00-08:30 LDN) generates signed FX drift IN the pre-release window; mechanism = asymmetric information, not post-release vol spike (F1 in matrix = POST-release impulse, ~null — this is the PRE-release channel) | **external** — BoE/ONS release calendar (free); on-disk GBPUSD 15m bars already exist | **~15%** | **pending** (needs release calendar; distinct from F1 which is post-release and ~null) |
| N47 | **London-open signed momentum — LDN 08:00–08:30 GBP-specific window** (first-half-hour directional autocorrelation) | Martins-Lopes arXiv:2411.16244 (Nov 2024) W-shaped intraday pattern + Elaut et al. 2018 intraday TSM; Gao-Han-Li-Zhou JFE2018 scoped to the LDN-open window only | GBP sees the sharpest vol spike at LDN open (Martins-Lopes 2024 Fig 3); signed return in 08:00–08:30 predicts 08:30–09:00 direction via the persistence-of-informed-order mechanism; DISTINCT from N4 (which ran market-wide intraday momentum at EURUSD 10m and was killed — N4 is clock-agnostic, this is scoped specifically to the LDN-open 30-min window for GBP) | on-disk (GBPUSD 15m) | **~12%** | **pending** (distinct from N4 which was clock-agnostic and killed; this is LDN-open scoped) |
| N48 | **WMR 4pm London fix pre-fix signed return feature** (systematic flow concentration) | ScienceDirect 2024 WMR-fix paper; Osler (2003) fix-flow concentration mechanism | institutional FX managers submit fix orders concentrated in the 16:00-16:05 LDN window; the signed direction of GBPUSD in 15:45–16:00 predicts 16:00–16:15 direction due to order-book imbalance from known fix flows; F2 in matrix = structural rule-based fix-signal ~null — this is ML feature from signed pre-fix return, not a calendar rule | on-disk (GBPUSD 15m timestamps) | **~8%** | **pending** (distinct from F2 which is calendar-rule ~null; this is a signed-return feature in a time-of-day window) |
| N49 | **Regime-conditioned EURGBP→GBPUSD lead-lag asymmetry** (Brexit/post-Brexit regime gate) | Basnarkov et al. arXiv:1906.10388; Granger-causality lit on GBP regime shifts | EURGBP→GBPUSD directional lead varies sign and magnitude across Brexit-era vs post-Brexit regimes; TRAIN years (pre-2021) contain the 2016–2020 Brexit period where EUR-GBP flow patterns differ structurally — a regime-gated version of EURGBP lead uses a TRAIN-only-defined regime classifier to avoid leakage | on-disk (EURUSD + GBPUSD bars; EURGBP if acquired for N45) | **~10%** | **pending** (depends on N45 EURGBP data; distinct from N17 anti-contemporaneous lead-lag which was killed without regime gating) |
| N50 | **Cross-sectional carry-momentum directional tilt** (high-yield vs safe-haven composite sign) | Lustig-Roussanov-Verdelhan JF 2011 carry factor + Elaut et al. 2018 intraday carry TSM | at each 15m bar, rank all majors by prior-period return; sign the GBP position as aligned with the cross-sectional momentum direction (carry/momentum composite) — a signed tilt feature not a standalone predictor; carries information about risk-on/risk-off regime that predicts GBP sign | on-disk (7-major panel — EURUSD, GBPUSD, AUDUSD, USDJPY, USDCAD, USDCHF, NZDUSD 15m bars) | **~10%** | **pending** (cross-sectional; distinct from N22 which was period-end calendar gate and killed) |
| N51 | **Pre-ONS data-release signed drift 06:45–07:00 GMT** (UK CPI/GDP/employment surprise direction) | Kurov et al. JFQA 2019 + Andersen-Bollerslev-Diebold-Vega AER 2003; Martins-Lopes arXiv:2411.16244 W-pattern peak at 07:00 LDN | same informed-positioning mechanism as N46 but scoped to 06:45–07:00 GMT window ahead of ONS data releases (typically 07:00 GMT); the 07:00 spike in Martins-Lopes 2024 Fig 3 is specifically the largest intraday peak for GBP pairs — actionable 15-min-before window | **external** — ONS release calendar (free, data.gov.uk); on-disk GBPUSD 15m bars | **~13%** | **pending** (sister candidate to N46 with different time window and release type; same external data acquisition) |
| N52 | **DeltaLag adaptive per-bar lead-lag as GBM input features** (pair-specific lag values from cross-attention) | arXiv:2511.00390 DeltaLag (Nov 2025) — Ha, Kim, Lee, Lee; cross-attention adaptive lag detection | DeltaLag learns WHICH cross-pair (EURUSD, AUDUSD, USDJPY etc.) leads GBPUSD and at WHAT lag on a per-bar basis via cross-attention; the inferred per-bar lag values and leader identity are themselves signed GBM input features — not used as a training LOSS/OBJECTIVE (N11 used DeltaLag as loss, N12 used fixed top-K leader lag, both killed); using lag VALUES as features is a distinct channel | on-disk (7-major 15m panel) | **~9%** | **pending** (distinct from N11 which used DeltaLag as a loss function and N12 which used fixed-lag features — both killed; this is per-bar adaptive lag values as features) |
| N53 | **GBPUSD LDN-session RS⁺−RS⁻ semivariance asymmetry — GBP flash-crash specific** (signed vol skew in LDN hours) | Patton-Sheppard RFS 2015 semivariance; GBP flash-crash (Oct 2016) literature | N31 found signed-semivariance GENUINE (passes mechanism null) but sub-breakeven and non-additive at EURUSD 15m/30m; GBPUSD flash-crash events (2016, 2019 mini) create extreme RS⁻ spikes specific to GBP liquidity structure — a GBP-specific RS⁺−RS⁻ feature computed on the LDN-session bars only, with the flash-crash windows in TRAIN, may show stronger directional skew than EURUSD | on-disk (GBPUSD 15m) | **~6%** | **pending** (distinct from N31 which ran EURUSD 15m/30m and was sub-breakeven; this is GBPUSD-specific LDN-session with flash-crash years in TRAIN) |
| N54 | **Signed OFI × announcement-day interaction feature** (order-flow informativeness elevated pre-release) | Takahashi arXiv:2508.06788 (Aug 2025) SVAR OFI-macro paper — price-impact coeff ANN_{t-1} = +0.129 (p<0.01) | Takahashi 2025 SVAR shows OFI price-impact rises +0.129 in the 15-min interval BEFORE macro announcements; interaction feature = (signed OFI at bar t) × (is_pre_announcement_window) as a GBM feature; OFI from on-disk tick data, announcement flag from external calendar; distinct from F1 (post-release impulse, ~null) and B5a (raw OFI, killed) — the INTERACTION of OFI magnitude with pre-announcement timing is the novel channel | on-disk tick data (OFI computable from bid/ask) + **external** announcement calendar | **~7%** | **pending** (needs external calendar; distinct from F1 post-release impulse ~null and B5a raw-OFI killed; the novel channel is the OFI×pre-announcement interaction) |
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
| I3 | \|return\|-weighted / GMADL loss | `m5_magweight.py`, `min1_magweight.py`, `usdjpy_2m_loss.py` | up-weight large-move bars (sign more predictable) — magnitude→direction bridge | D | 5m: rebalances to two-sided ~.56, collapses 2026 UP (regime-dependent sign); GMADL operating-point untried. **[EURUSD·60s] NULL** (`min1_magweight.py`); **[USDJPY·2m] KILLED on pooled** (`usdjpy_2m_loss.py`: |ret|^{.5,1} LOWER VAL-AUC .524→.517 + worse held-out) + mag→dir bridge `[USDJPY·2m]` high-mag bars NOT more sign-predictable (`usdjpy_2m_magdir.py`). Sign-invariance confirmed by experiment at 2m |
| I4 | Calibration (temperature/Venn-Abers) + selective threshold | (wrap any book) | calibrate probs so the confidence gate is honest; re-derive gate post-calibration | gate | nearly free; required wrapper for any confidence-gated edge |
| I5 | Cross-pair POOLING (train-row pooling across pairs) | `usdjpy_2m_xpair.py pool` | train all majors as ROWS in one GBM, each on its own sign; eval target (≠ cross-pair FEATURES; ≠ weight-shared net) | D | **TESTED [USDJPY·2m]:** lifts CPCV win-rate MEAN above single-pair (~.52→.546) = NOISE-DECORRELATION, not factor sign; worst-regime p10 SATURATES ~.531 < BE. **Read p10/frac-clear not mean.** Multi-algo (lgb+xgb+cat) decorrelation HURTS a thin signal (p10 .519<.531). See METHODS_CATALOG §9.5 |
| I6 | AdamW + tuned LR + Optuna TPE/Hyperband | (wrap nets), `usdjpy_2m_optuna.py` | n_trials capped & LOGGED as multiplicity; select on worst-VAL-half | tuning | fixes `Adam lr1e-3 single-seed` anti-patterns. **[USDJPY·2m] KILLED on pooled**: best worst-VAL-half .5496 (passes VAL!) but held-out WORSE than default — the corr(VAL,OOS)=−.54 ANTI-TRANSFER, by experiment. Tuning can't cross a signal bound |
| I7 | Recency-weighted training (exp decay by year) | `usdjpy_2m_recency.py`, `m30_recency.py` | up-weight recent years to match the deploy regime | D | **[USDJPY·2m] KILLED**: sharper recency monotonically LOWERS VAL-AUC + held-out (overfits the small recent window) |
| I8 | IRM / env-invariant feature-stability filter (prune era-local sign-flippers) | (wrap the certified book; per-era invariance score) | keep only features whose sign-contribution is stable across train eras, hoping to cure refit-decay without periodic retrain | D | **KILLED — SURVIVES=False**: pruning era-local sign-flippers does NOT recover a frozen-vintage edge → refit-decay is an INFORMATION BOUND, not a fixable feature-selection artifact (corroborates A7/I7: regime robustness is not buyable via selection; the book stays REFIT-DEPENDENT). [USDCHF·15m] example → `USDCHF_RESULTS.md` |

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
