> **SCOPE: EURUSD** (key-specific). Generic methods/ideas live in METHODS_CATALOG.md / SWEEP_MATRIX.md / IDEAS_LOG.md; cross-key theory in THEORY.md. See REPO_MAP.md.

# EURUSD Binary DIRECTION — Exhaustive Investigation & Findings (2026-05-30/31)

Master summary of the multi-day effort to build a high-accuracy EURUSD up/down binary model across horizons
(1 second → 30 minutes), with the deriv-faithful, leakage-controlled, OOS-verified methodology established by the
2026-05 bias audit. **Every number here is held-out (TEST 2024 / 2025 + OOS 2026), independent (non-overlapping
windows, chronological/no-look-ahead), bootstrap-CI'd.** Companion logs: `m30_research_log.md`, `m5_research_log.md`,
`IDEAS_LOG.md` (orthogonal-math program), `research_log.md` (pre-audit + bias-audit history).

## TL;DR — the honest frontier

| Horizon | DIRECTION (sign) — best honest OOS | Tradeable? |
|---|---|---|
| 1–5 seconds | **~0.65–0.66** (tick microstructure ensemble) | only on a tick/seconds-expiry broker; latency-critical |
| 1 minute (60 s) | **UP 0.520/0.584/0.613 but CPCV-UNCERTIFIED** (regime-dependent up-filter, path p10 .524, block-boot CI-lo .530); **DOWN dead/exhausted** (p10 .474). Both sides formally CLOSED 2026-06-03 — no certified 60s edge (`min1_cpcv.py`) | synthetic-index only (deriv forex floor = 15 m) |
| 2 minutes (120 s) | **UP 0.555 / 0.5445 ties-strict, CPCV-UNCERTIFIED**; **DOWN dead** (~0.52). Both sides formally CLOSED 2026-06-04 — keystone **cross-pair pooling RUN+KILLED** under ties-strict refit-CPCV (UP p10 .5096 / DOWN .5124, 0-7% paths), **RFF** .4997, **specialist** nested-refit .4957/.4912; 4 model classes ~.50-.52 AUC = genuine efficiency (`min2_xpair_cpcv.py`, `min2_rff.py`, `min2_spec_cpcv.py`) | synthetic-index only (deriv forex floor = 15 m) |
| 15 minutes | **0.647 recent (real) / ~0.58 robust** — the actual 3-model ensemble × compression×NY: chronological-split 2024-26 = **0.647** (2024 0.689/2025 0.582/2026 0.663, n677, CI[.612,.684], `m15_production.py backtest`); FAITHFUL purged-combinatorial CV of the SAME ensemble+gate (15 paths, all eras) = **mean 0.579, CI[.567,.591], 14/15 paths > breakeven** (`min15_cpcv_result.json`). So: real, regime-dependent, robustly profitable — 0.647 is the favorable-regime end, ~0.58 the durable cross-era number. NOT a mirage. (The earlier "0.545" was a weaker single-LGBM proxy — corrected.) | ✅ deriv — best tradeable up/down book |
| 10 minutes | **BOTH sides CERTIFIED (2026-06-04)** — cross-pair USD-residual+OF book `EURUSD.m10xp.v1`: per-side refit-CPCV **UP p10 .5863 / DOWN p10 .5683, 15/15 paths** (improves base-book side floors .561/.552; base combined .602). 11 improve/discover levers all KILLED → gated raw cross-pair sign, info-bound by 2025 regime; loops dry. Confirms gradient none@60s→UP@5m→BOTH@10m&15m (`m10_xpair_cpcv.py`, `sweeps/EURUSD_10m.md`) | synthetic-index only (deriv forex floor = 15 m; deployable sibling = `EURUSD.m15xp.v1`) |
| 30 minutes | **~0.59** (compression-1h × NY selective) | ✅ deriv |
| 5 minutes | **0.613 verifiable / 0.648 thin-cov** (cross-horizon stack: 15m edge × 5m cross-pair meta; 2026-05-31b) | needs a ≤5m-expiry broker |
| 10/15/30m **by session** (2026-06-06) | **NY-only** cross-pair: NY UP/DOWN 10m .6053/.5896, 15m .5845/.5712, 30m .5681/.5639 (15/15); **LDN & Asia certify NOTHING** (~.49-.53). DST-correct STRICT session-only; NY beats legacy fixed-UTC gate (`session_xpair_*_result.json`) | same venue as the parent horizon — edge is NY-concentrated |

**A >75% (or even >65%) DIRECTIONAL edge does not exist at 5m/15m/30m on EURUSD.** Direction at ≥5 minutes is the
efficient-market part: ~0.50–0.52 AUC, ~0.515 unconditional (mild mean-reversion). The only place a >0.65 *directional*
edge is real is the **seconds scale**, and it decays to noise by ~60s.

## The genuine positive result: MAGNITUDE is forecastable, DIRECTION is not

The single most important finding (and the answer to "group the data to tell a different story"):

> **30-minute |return| (move SIZE) is strongly, OOS-verifiably forecastable — realized-vol → large-move AUC 0.73–0.78**
> (corr +0.38–0.44, stable across 2024/2025/2026). DIRECTION/sign stays ~0.515.

This is backed by a sign-invariance **theorem** (arXiv:2512.15720, Dec 2025): order-flow / permutation entropy is
invariant under sign permutation, so complexity/entropy measures detect the *presence/size* of informed moves
(magnitude), **not the sign**. That is *why* every complexity/regime/gate approach was null for direction — they gate
volatility, not direction. Magnitude is tradeable on **Touch/No-Touch, Range/Boundary, straddle** products (not up/down).

## 2026-06-06 — DST-correct session re-campaign + Kronos look-forward fix + full-suite audit

Re-ran the direction question **segmented by trading session** (DST-correct: NY 8-17 America/New_York, LDN 8-16
Europe/London, Asia 9-18 Asia/Tokyo; `sessions.py` `session_mask` = local-tz hour, applied per-day across train+val+test+oos),
caught and fixed a 1-bar look-forward bug in the Kronos direction eval, and ran a full-suite forecast-derivation correctness
audit. **Net direction result: the certified ≥10m cross-pair edge is decisively NY-CONCENTRATED; LDN/Asia certify at NO
horizon; Kronos direction is null at every horizon after the alignment fix; 1m/2m direction stays null in every session.**
All numbers Tier-1 (result JSONs on disk).

### The Kronos look-forward bug (user-caught) + fix → corrected NULL at every horizon

`kronos_dir.py` / `kronos_ft.py` eval had a **1-bar look-forward MISALIGNMENT** (failure-mode "FM-F forecast-derivation"):
at decision bar `i` it fed context `slice(i-L,i)` = bars `[i-L..i-1]`, predicted bar `i`, and scored `Pup = pred_close(i) > C[i-1]`
— i.e. the move INTO the entry, window `[t[i-1],t[i]]`. But the deriv label `y[i]` (`barcnn_bars.labels_at`: entry `t[i]+1s`,
exit `t[i]+61s`) is the FORWARD window `[t[i]+1, t[i]+61]` — DISJOINT, off by one bar. A misaligned forecast yields a FALSE
NULL, never a false positive. **Fix = `kronos_mtf.py`:** context ends AT bar `i` (`slice(i-L+1,i+1)`, last close = entry ref
`C[i]`), predict `pred_len = H/GRID` FORWARD steps, `Pup = pred_close(+H) > C[i]`; pred-side contiguity
`t[i+Hsteps]-t[i] == Hsteps*step`; nonoverlap `GAP = HS+TOL`. Validated: forward label agrees with next-bar sign **92.3%**
(n=233,950). `kronos_dir.py` is now gated (`KRONOS_DIR_LEGACY=1` to override); the legacy result is superseded.

**Corrected Kronos direction (`kronos_mtf.py`, alignment-fixed) is NULL at EVERY horizon** 1/5/10/15/30m, zero-shot AND
fine-tuned, all sessions: pooled .50-.51, CPCV p10 .489-.500, all KILLED, up-rates in-band. Even at **NY ≥10m where cross-pair
certifies .57-.61, Kronos reads ~.50** — it ingests only EURUSD's OWN OHLCV candles, not the 7-pair USD cross-section that
carries the edge; fine-tuning did NOT help direction. Files `kronos_dir_mtf_*_result.json`. (FINE-mode "1m→Nm up the chain" +
multi-TF ensembles **in progress**.)

### Cross-pair book, STRICT session-only — the ≥10m direction edge is NY-only

`session_xpair.py` (the certified ≥10m lever, run STRICT session-only DST-correct; gate
`{2:1m_bb_width, 5/10:5m_bb_width, 15:15m_bb_width, 30:1h_bb_width}`). **NY certifies BOTH sides at every horizon; LDN and
Asia certify at NONE:**

| Horizon | NY UP / DOWN (p10) | LDN UP / DOWN | Asia UP / DOWN | vs legacy fixed-UTC gate |
|---|---|---|---|---|
| 10m | **.6053 / .5896** (15/15 each) | .523 / .524 | .511 / .516 | NY BEATS legacy .586/.568 |
| 15m | **.5845 / .5712** | .527 / .520 | .519 / .496 | — |
| 30m | **.5681 / .5639** | .520 / .514 | .487 / .509 | NY BEATS legacy .559/.553 |

NY at 10m/30m **beats the legacy fixed-UTC gate** (10m legacy .586/.568, 30m legacy .559/.553) — the direction edge is not
just present in NY, it is *stronger* when the session filter is DST-correct and strict. Files
`session_xpair_{10,15,30}m_{sess}_result.json`. (**5m + 2m in progress.**) Discipline: the session filter restricts DECISION
ROWS in train+val+test+oos (GBM/xpair use causal-continuous features, so this is rows-only).

### Per-session base GBM — direction null/killed every session; magnitude certified every session

- **1m tick GBM** (`session_1m.py`): DIRECTION **null all sessions** (pooled ~.504, p10 .500-.504, frac_clear 0.0). MAGNITUDE
  certified all (cov≤10% frac 1.0; magAUC NY .675 / LDN .728 / Asia .717; p10 .60-.71). Files
  `session_1m_{dir,mag}_{ny,ldn,asia}_result.json`.
- **2m tick GBM** (`session_2m.py`): DIRECTION **null all** (p10 .496-.500); MAGNITUDE certified all (p10 .58-.70). Files
  `session_2m_*`.
- **5m/10m/30m base bar GBM** (`session_bars.py`): DIRECTION **KILLED all sessions** (NY strongest, e.g. 10m NY p10 .525);
  MAGNITUDE certified all sessions (p10 .72-.80). Files `session_{5,10,30}m_{dir,mag}_{sess}_result.json`. (**15m base GBM in
  progress.**)

So: at the base-feature substrate, no session unlocks direction at any horizon — the certified ≥10m direction edge lives
ONLY in the cross-pair USD cross-section, and ONLY in the NY session. The session split reaffirms (not refutes) the
direction≠magnitude split per-session: magnitude certifies in every session at every horizon, direction does not.

### Full-suite forecast-derivation correctness audit — bug is ISOLATED, no cert invalidated

A 19-agent workflow + lead Tier-1 proofs swept all 308 scripts for the FM-F misalignment. **VERDICT: isolated to the Kronos
family (2 scripts); NOT systemic.** Every other forecast-derivation script (`usdjpy_{1m,2m}_statespace`,
`usdjpy_2m_xhorizon`, `m5_xhorizon`, `m5_lossbatch`, `f1_compound`) predicts the FORWARD quantity over the SAME horizon as the
label at the SAME bar = correctly aligned. GBM/CNN classifiers are trained DIRECTLY on the label and scored against it →
structurally immune to FM-F. **PROVEN CLEAN (Tier-1 empirical):** TICK substrate (features causal via truncation test
max|full−trunc|=0.0; label = forward deriv outcome, 0/4000 independent-recompute mismatch; corr(y,future)=.486 vs
corr(y,past)=−.003; up-rate .5006); BAR substrate (forward label 0/2000 mismatch at H=5/10/30, corr(y,future)~.99 vs ~−.02,
up-rate .498-.506); XPAIR substrate (`m5_xpair.build_xp _y` forward 0/2000 mismatch H=10); BAR-feature causality (239 features,
truncation max|full−trunc|=0.0 across ALL). **NO certified direction/magnitude book is invalidated.** Bounded flags (do NOT
invalidate any cert): `barcnn_mag.py:163` + `barcnn_regime.py:69,71` FM-E selective-threshold on pooled test+oos (magnitude/regime
target, CPCV-deflated; fix = use VAL threshold); `usdjpy_2m_cpcv2.py` FM-E max-p10 cell (best .5185 ≪ .541, certified=false
anyway); `min2_mim.py:18` + `min2_legsign.py:25` FM-A `shift(-FWD)` no contiguity guard (KILL screens, already KILLED);
superseded pre-v3 cohort (`min1_v*`/`min2_v*`/`tickmodel*`/`tick5s_final`/`tick_ensemble`, documented bar-shift label + greedy
nonoverlap, no result.json, replaced by `*_production` books).

### New scripts (this session)

`sessions.py` (DST-correct `session_mask`/`SESSIONS`); `session_1m.py`, `session_2m.py` (tick GBM per session),
`session_bars.py` (bar GBM per session, any H), `session_xpair.py` (cross-pair STRICT session-only, any H);
`kronos_ft.py` (single-process GPU adaptation of the GPU/DDP-only Kronos `finetune/train_predictor.py` — frozen base
tokenizer, predictor FT on session-only contiguous 1m stream, AMP bf16, early-stop); `kronos_mtf.py` (CORRECTED + multi-TF
Kronos direction eval, native + fine modes, per-session, ensemble-ready npz); `kronos_bars.py` (H-min/fine-grid OHLCV + forward
deriv-label builder, generalizes `barcnn_bars`); `kronos_ensemble.py` (multi-TF vote combine); `barcnn_run.py` gained a SESSION
arg (per-session bar-image CNN). Hardware note: box now has an NVIDIA RTX 5050 Laptop (8GB, Blackwell sm_120, driver 580/CUDA13);
venv torch swapped `2.12.0+cpu`→`2.12.0+cu130` → Kronos FT+inference now GPU.

## Session-4 (2026-05-31d) — 1-MIN re-push: lesson-transfer + Hidden Markov + online concept-drift

Goal re-set to **1-min >0.65 OOS**. Applied the methods discovered AFTER `min1_production.py` was frozen — none had ever
touched the 60s horizon (grep `m5/m15/m10_EURUSD` in `min1_*`/`min2_*` = 0 hits). Full journal: `min1_research_log.md`; two
multi-agent research workflows (prior-art/infra/sofien; and an HMM literature/feasibility pass). **>0.65 is NOT achievable —
six independent levers all pinned by the 2025 regime / market efficiency:**

| Method (file) | Honest selective | test25 | Note |
|---|---|---|---|
| cross-horizon STACK — 5m/15m parent direction front-loaded into 60s, meta-gated (`min1_stack.py`) | ~0.586–0.594 worst-half; **0.53–0.55 at verifiable cov (oos n≥100)** | 0.56→0.53 | standalone parent front-loads at only ~0.51 at 60s (vs 0.597 @5m) — 60s is too small a slice of the parent move |
| cross-pair USD-residual direct, MX_HOR=1 xpof (`m5_xpair.py`) | AUC **0.516**; combined 0.560 | 0.534 (CI[.518,.550]) | sign-stable but sub-0.56; the 5m lift doesn't survive to 60s |
| Hurst / variance-ratio persistence switch (`min1_hurst.py`) | worst-half FLOOR 0.513; **ORACLE max-floor 0.555** | 0.548 | persistence gates magnitude, not sign (sign-invariance) |
| **Hidden Markov regime** K=3, causal filtered posteriors (`min1_hmm.py`) | U1 gate 0.565 / **U2 engine-switch 0.600** (OOS CI[.519,.708]) / U3 meta 0.487 | 0.569 / 0.605 / 0.508 | states carry NO direction (train P(up)≈0.50 in all 3); best refinement = trade reversion only in vol-state 0 → ~0.60 floor, thin-cov, CI spans breakeven |
| **Online concept-drift** river ARF+ADWIN, prequential (`min1_online.py`) | AUC **0.503–0.505** every window; selective 0.49–0.51 | 0.503 | continuous adaptation recovers NO edge → the 2025 wall is GENUINE efficiency, not stale-model drift |
| **Macro-release 60s impulse** — sign(surprise) in the post-release window (`min1_news60.py`) | HIGH-vol 0.373/0.526/0.517; \|surp_z\|≥1 → 0.167/0.364/**0.000** | 0.526 | NULL/negative — FX prices the surprise in <60s and overshoots-reverts; larger surprises are MORE wrong OOS. Extends the 5m news null to the impulse timescale |

- **"Would HMMs help?" — No, empirically + literature.** A Gaussian HMM's latent states are volatility/size regimes (train
  P(up)=0.497–0.499 in all 3; momentum loses in every state → all map to the reversion engine), confirming sign-invariance
  (arXiv:2512.15720) at 60s. The academic record agrees: Markov-switching cannot beat a random walk OOS for FX direction
  (Empirical Economics 2019), the classic FX-HMM switches the variance not the sign (Dueker-Neely), jump-model/HMM benefit is
  risk reduction (arXiv:2402.05272), the one intraday momentum-HMM is equity futures with no cost-net sign edge (arXiv:2006.08307).
- **The online drift learner is the decisive control:** the only way the wall could be a fixable artifact is if it were stale-model
  drift; an adaptive forest that re-fits to recent bars sits at 0.503–0.505 AUC in 2024, 2025 AND 2026 — so 60s direction is
  genuinely efficient and no amount of regime cleverness recovers it.
- **Up/down asymmetry (user idea):** the book's edge lives entirely on the UP (dip-buy) side — up-predictions 2025 0.584 / 2026
  0.613 vs down-predictions ~0.52 (dead). An up-only FILTER on the symmetric model is the best honest directional book of the
  session, but a separately-TRAINED up-specialist is WORSE (0.50/0.54/0.52 — subset-training kills the ranking, à la v4/v6/v13), and
  the asymmetry vanishes in 2024 (regime-dependent); does not clear 0.65. **Scholarly deep search (3-agent + adjudicator):
  `beats_our_060 = False` — best published EUR/USD direction is 0.585 DAILY (Castillo 2024); honest 60s sign ceiling ~0.52–0.55 net
  of costs (Petrova-Vilhelmsson-Nordén IJF 2025); DeepLOB/OFI headlines are equities+depth-10+leakage+magnitude; >0.65 at 60s is
  unsupported anywhere.** Our ~0.60 is at/above the published frontier.
- **Best honest 1-min book ~0.55–0.60** (HMM vol-state-gated reversion, floor 0.60 thin-cov; profitable-on-point-estimate vs
  breakeven 0.541 but OOS CI not clear of it, NOT >0.65). Tooling installed for completeness: hmmlearn, statsmodels (Markov-switching),
  arch, ruptures, pomegranate, river, nolds, filterpy/pykalman, stumpy, tsfresh/tslearn/sktime/darts. **Venue:** deriv EUR/USD
  forex Rise/Fall floor = 15 min, so a 60s book is synthetic-index-only.

## Session-5 (2026-06-03) — 1-MIN FORMAL CLOSURE: both sides CPCV + 2 dry discovery rounds
Goal re-set to **best 1m UP & best 1m DOWN, OOS-verified**. Ran the remaining on-disk DOWN levers + a faithful CPCV of
both sides + two adversarial discovery rounds. Ledger `sweeps/EURUSD_1m{,_backlog}.md`; per-key rows EURUSD_RESULTS.md
§60s rows 18-21.

| Lever (file) | Result | Verdict |
|---|---|---|
| **USD-strength-conditioned DOWN** (`min1_downcond.py`, D3 retarget MX_HOR=1) | USD-strong 2025 DOWN .494 ≈ USD-weak; coverage curve FALLS to .475/.464 at top-2% USD-strong tail (mean-reverts) | ❌ KILLED — 60s DOWN efficient even USD-gated; strengthens the 5m D3 kill |
| **\|ret\|-weighted magweight retrain** (`min1_magweight.py`, POW=0.5, tick substrate) | **best_iter=8/4000** (no learnable signal); DOWN 2025 .507, UP worse than filter | ❌ KILLED — loss-reweighting doesn't unlock 60s DOWN; 5m razor-thin edge doesn't transfer |
| **magnitude→direction bridge** (`min1_magdir.py`, frozen mag+dir) | DOWN FLAT ~.50 across EVERY predicted-move-size gate incl. top-5% biggest moves (magAUC .787, zero direction) | ❌ KILLED — textbook sign-invariance at the operating point |
| **Faithful CPCV** (`min1_cpcv.py`, frozen-book, ties-strict, 28 purged paths + block-boot) | **UP** pooled-strict .573, CI-lo .530, p10 **.524**, 75% clear, 2024 .520; **DOWN** .519, p10 **.474** | ⚠/❌ NEITHER certified — UP regime-dependent filter (near-miss, like 2m UP .5445), DOWN dead |
| **Discovery rounds 1+2** (adversarial fan-out, 20 candidates) | round 1: 12 microstructure-sign levers; round 2: 8 cross-horizon/calendar/UP-cert/recent-lit — **ALL subsumed/sign-invariant, 0 survivors** | both rounds DRY → discover loop dry |

**VERDICT:** no certified 60s direction edge exists. **UP** = best-AVAILABLE up-only filter (.520/.584/.613 moved,
.573 ties-strict), regime-dependent dip-buy, **uncertified**; the 2024 wash is STRUCTURAL regime, so no variance lever
(seed-ens/ACI/calibration) can certify it. **DOWN** = dead + honestly EXHAUSTED on-disk; the online-ARF keystone proves
60s direction is genuine efficiency. Only DOWN redirect = EXTERNAL data (option-implied risk-reversal sign
[verified-paywalled], intraday DE-US rate differential) — acquisition prerequisite. 60s remains synthetic-index-only.

## Session-3 (2026-05-31c) — 10-MIN battery: native ensemble, gate sweep, cross-horizon stack, walk-forward

Goal set to **10-min >65% OOS**. Ran 6 independent Tier-1 methods (full journal: `m10_research_log.md`; a 4-agent
research workflow independently ranked the same combinations and pre-warned the verdict). **Honest 10-min direction
ceiling ≈ 0.60–0.61; >0.65 OOS-verified is NOT achievable** — every method is capped by the **test25 (2025) regime**:

| Method | Honest combined | test25 floor | Note |
|---|---|---|---|
| native-10, VAL-acc-max gate | **0.667** (n543) | 0.591 | artifact — buoyed by 2024+2026 (~0.70); 2025 CI dips to breakeven |
| native-10, HONEST gate sweep (200 cfg) | 0.576 (ORACLE 0.603) | ≤0.60 even with hindsight | corr(VAL,OOS)=−0.54 → can't select the good gate |
| cross-horizon stack dir15/dir10/agree + meta | ~0.59 | ≤0.56 | meta narrows coverage, can't create edge in 2025 |
| cross-pair USD-residual + order-flow | 0.599 (n2171) | 0.556 | the 5m sign-stable XP lift is weaker/absent at 10m |
| walk-forward (1yr-gap adaptive retrain) | 0.609 (n2381) | 0.573 | +0.017 on 2025 only — regime genuinely unpredictable |
| **HONEST DELIVERABLE** (native-10 × 5m_bb_width-NY cov10%) | **0.602** CI[.582,.621] (n2399) | **0.579** | all windows ≥0.579, EV **+0.113@0.85**, profitable; NOT >0.65 |

- **Native-10 raw AUC ≈ 0.525** (val, unconditional) — same noise floor as 5m/15m/30m. The compression×NY gate concentrates
  it into a ~0.60 selective book, exactly interpolating the horizon map (5min 0.583 < **10min ~0.60** < 15min 0.642).
- **The 0.667 "win" is the discipline trap made concrete:** picking the gate by VAL accuracy (corr(VAL,OOS)=−0.54) produced
  a combined number >0.65, but the binding 2025 window was only 0.591 and the honest sweep proves no gate gets all three
  windows above ~0.60 *even cheating with hindsight*. Combined-average >0.65 ≠ robust >0.65.
- **A better-aligned gate for 10m exists:** `5m_atr_pct`/`5m_bb_width` × London/overlap session gives a higher, more stable
  floor than the 15m-winner's `15m_bb_width × NY` — a genuine (small) refinement, but it does not break 0.65.
- **Walk-forward is the cleanest regime test and it closes the door:** adapting through 2024 to predict 2025 lifts the
  binding window only +0.017. The 2025 EURUSD 10-min regime is near-efficient for direction; gap-reduction isn't the lever.
  **(UPDATE 2026-06-04 — Session-1 above was COMBINED-only. The side-split re-push certifies BOTH 10m sides via the SELECTIVE compression×NY×confidence-gated cross-pair USD-common-factor sign: `EURUSD.m10xp.v1` refit-CPCV UP p10 .5863 / DOWN p10 .5683, 15/15 paths — see the top table + `m10_research_log.md` + `sweeps/EURUSD_10m.md`. Raw 10m direction does remain ~efficient (~.52 AUC); the certified edge is the thin gated cross-pair sign, info-bound by the 2025 regime, not a refutation of near-efficiency.)**
- **EXTERNAL cross-asset (ES S&P500 e-mini minute futures, the only untried directional lever) — null, and it reveals the
  mechanism:** the ES→EURUSD *lead-lag* corr is tiny (|corr|<0.055) AND **sign-flips from +0.02 in 2024 to −0.05 in 2025**;
  contemporaneous corr is real (+0.16..+0.22) but untradeable. The normal risk-on→USD-weakness link **decoheres/inverts in
  2025** — *that is why every method's test25 floor collapses*. External equity data can't fix a regime where the macro
  relationships themselves invert. (NQ ≈0.95-corr with ES → redundant; XAUUSD/DE30EUR CFDs have ~no coverage on disk.)
- Deliverables: `m10_production.py`, `m10_freeze_honest.py` (→ `models/m10_EURUSD_strategy_honest.json`), `m10_stack.py`,
  `m10_gate_sweep.py`, `m10_walkforward.py`, `m10_xstack_probe.py`, `m10_xasset_probe.py`. Models: `models/m10_EURUSD_*`.

## Session-2 (2026-05-31b) — fresh 5-min battery: cross-pair + order-flow + meta-labeler

Goal re-set to 5-min >65% OOS. Ran a NEW, genuinely-untried battery (+ a 5-agent research workflow that ran its own
falsifiers). Result: the honest 5-min frontier **improved 0.566 → ~0.61**, but **≥0.65 OOS-stable is still not reachable** —
confirmed a 4th independent way. Details in `m5_research_log.md` iter 8-11; lab: `m5_xpair.py` `m5_xpair_probe.py`
`m5_xp_analyze.py` `m5_meta.py` `m5_xpair_production.py`.
- **Cross-pair / USD-common-factor / lead-lag (NEW orthogonal family):** USD basket from the other 6 majors (sign-aligned),
  per-pair lead-lag residuals in EURUSD-equivalent terms, catch-up & EUR-idiosyncratic residual, dispersion/agreement.
  The relative-value RESIDUAL reversion is **sign-stable across 2024 & 2026** (+0.015/+0.024 spearman) — unlike OHLCV
  momentum which flips on 2025. As features it lifts the book to **0.586** (8 of top-20 feats are cross-pair). Cross-pair
  MOMENTUM-continuation agreement = dead; only the residual is live, and below breakeven standalone.
- **Order-flow (`features_of/`, 18 cols: Kyle λ, OF persistence/accel/uptick):** 0 AUC lift (raw OF into a GBM is null,
  independently verified), but as part of the model + meta-features lifts the book to **0.594**. Signed-OF flow-FOLLOWING
  loses (stably negative vs fwd return); only flow-as-reliability is usable.
- **P0 conditional gate:** NO conditioning (agreement / OF / dispersion / vol-compression / session) lifts the binding
  2025 window above ~0.557. **ORACLE (hindsight-cheating) max-floor = 0.601** — even cheating can't get all 3 windows to 0.65.
- **Learned meta-labeler on orthogonal axes (the "avoid losers" ask, done properly):** 2nd lgb predicts P(primary correct)
  from agreement/dispersion/OF/confidence (not the 239); abstain unless meta≥thr; threshold by **worst-VAL-half stability**
  (not VAL-acc-max). FROZEN production (`m5_xpair_production.py`, q0.95): **combined 0.583** (test24 0.607 / test25 0.555 /
  oos 0.606), EV +0.078@R0.85 — reaches **~0.61** at a more selective point. ORACLE floor 0.598 — test25 caps ~0.55-0.60. A
  meta-classifier cannot exceed its features' conditional accuracy (~0.557 on test25). **Not 0.65.**
- **Cross-horizon STACK (strongest method, `m5_xhorizon.py`/`m5_stack.py`/`m5_stack2.py`):** the repo's real 15-min edge
  (`m15_production`, 0.647) **front-loads** into the 5-min sub-move — the 15m ensemble's confident direction predicts the
  5-MIN outcome at **0.597** (better than the 5m-native model). Gating it with the 5m cross-pair meta-labeler and requiring
  AGREEMENT (two semi-independent edges) lifts the binding window over 0.60 for the first time. A soft learned stack
  (meta predicts P(dir15 correct on 5-min) from orthogonal axes) gives the honest frontier: **0.596 (oos n456) → 0.613
  (oos n163) → 0.648 (oos n45)**. Honest worst-VAL-half-stable pick = 0.648 combined, but test25 caps 0.597 and the >0.65
  region rests on oos n45. The 15m parent is itself only 0.647 on its native task, so the noisier 5-min sub-move approaches
  but cannot robustly exceed it. **Best honest verifiable 5-min book ≈ 0.61** (artifacts `models/m5stack_EURUSD_*`).
- **Macro-surprise EVENT-CONDITIONING — obtained the external data, tested, NULL for direction** (`fetch_calendar.py`,
  `event_signs.py`, `m5_news.py`, `m5_news_model.py`): a 6-agent search found FXStreet's free keyless API (actual+consensus,
  UTC-minute, 2012-2026). Fetched 8838 USD/EUR impactful events; mapped each to an EURUSD direction signal. Post-release 5-min
  direction: surprise-sign rule train **~0.50-0.52**; jump-continuation train ~0.49; jump-reversal held-out ~0.53 with train
  ~0.50 (noise). FX prices a macro surprise within ~1 MINUTE (efficient jump) — no exploitable 5-min directional drift; the
  surprise is a magnitude/volatility event (sign-invariance holds even for fundamental news). The "news calendar" lever I'd
  flagged as the real shot is now empirically CLOSED for up/down direction.
- **Why no robust >0.65:** the 2025 (test25) regime is near-efficient for 5-min EURUSD direction; the corr(VAL,OOS)=−0.54
  anti-transfer wall; OOS-2026 (~5 months) starves coverage at high concentration; and even external macro-surprise data is
  null for direction. The remaining edge is at the seconds/tick scale (mtick3 3s 0.657/0.667), not the 5-min up/down book.

## Everything tried (exhaustive; all OOS cross-window honest)

**Models / data (direction):**
- OHLCV 239-feature ensemble (lgb+xgb+cat) at HOR 5/15/30 → AUC ~0.52; selective ~0.56–0.64.
- Tick microstructure (order-flow imbalance, microprice) HS 1–1800s → AUC ~0.50 @30m, **~0.65 @1–5s**.
- 1D-CNN & GRU on the raw tick path → AUC 0.525 (= GBM): **model type is not the bottleneck, the data is**.
- **Bar-image 2-D CNN (Sezer CNN-BI / GAF, the 4th model class)** → 60s **dirAUC ≈ 0.50** across hist/ohlc/gaf images, CPCV path_p10 .484–.499, 0.0 paths clear breakeven (incl. regime-gated); even the GADF antisymmetric sign-field is null. The SAME image CNN on MAGNITUDE clears >65% (`MAGNITUDE_FINDINGS.md` §3) — cleanest single-method sign-invariance proof. Method: `METHODS_CATALOG.md` §5.5; scripts `barcnn_*.py`.
- Cross-pair / USD-basket, external CME ES/NQ futures lead-lag (real data thru 2026) → +0 / zero 30m lead.
- Volume & dollar bars (López de Prado information bars) → null for direction (improve normality, not AUC).
- Sofien Kaabar's 45 custom indicators as features → 0 OOS AUC (rank high in importance, redundant OOS).

**Gates / rules (direction):**
- Larger-TF reversals (1h/4h/daily RSI/DeMarker/WillR/CCI/BB/z/Fisher extremes; confluence; chop-gated) → null.
- Sofien's 79 mined rules (`sofien_rules.json`): daily pivots, Internal Bar Strength, Connors RSI2, TD Setup-9,
  BB re-entry, divergence, confluence → null (best Connors RSI2 ~0.55 train/val → 0.47–0.51 OOS).
- "Avoid losers" complexity gates: permutation entropy, Hurst (variance-ratio + DFA), autocorrelation → null
  (flat accuracy across all bins → they gate magnitude, not sign).
- FX fixing-window reversals (WMR 16:00 / ECB / Tokyo, Krohn-Mueller-Whelan) → ~0.58 in-sample, sign-flips OOS.

**Outside-finance math (4 research workflows + builds):**
- Path signatures / Lévy area (price↔order-flow rotation) → null @30m, top-5 feature @5s (= the seconds edge).
- Hawkes processes, transfer entropy, convergent cross mapping, RQA/DFA → directional variants are sub-minute /
  hourly / need trader-resolved LOB we don't have; the rest are sign-invariant (magnitude).
- Permutation-entropy / weighted-PE / sample-entropy → magnitude gates (per the theorem), null for sign.

**Convergent external evidence:** Petrova-Vilhelmsson-Nordén (*Int. J. Forecasting* 2026, FX LOB, 1min–1h):
near-EMH predictability. Meese-Rogoff; Rossi (*JEL* 2013). The repo's own 16-report `research/` corpus reached the
same null independently.

## Lessons (methodology — carry forward)

1. **Direction ≠ magnitude.** At ≥5m, sign is ~EMH; volatility is highly forecastable. Don't conflate a high
   *magnitude* AUC with a *directional* edge. (The sign-invariance theorem makes this rigorous.)
2. **Model type / feature richness is not the bottleneck — information is.** GBM ≈ CNN ≈ GRU ≈ 0.525 AUC at the data
   ceiling; Sofien indicators and signatures add 0 OOS AUC; the limit is the data.
3. **The edge is at the seconds scale and decays fast** (5s ~0.65 → 60s ~0.52). Longer = more efficient.
4. **corr(VAL, OOS) ≈ −0.54**: VAL-max selection anti-transfers. Require a result to hold across ALL of
   {TEST24, TEST25, OOS26}; every single-window ≥0.75 was an n=12–50 / multiple-testing mirage.
5. **Don't fabricate from low-coverage pockets** — that is the exact bias the 2026-05 audit existed to kill.
6. **Information-driven bars** improve return normality, not directional AUC (matches the literature).
7. **The 5-min ceiling = the 15-min ceiling (≈0.647).** The only way to beat the 5-min noise floor is to BORROW the cleaner
   longer-horizon signal (cross-horizon stack: the 15m model's confident direction predicts the 5-min move at ~0.60). The 15m
   parent is data-bound at 0.647 and **not liftable** — cross-pair features add 0 AUC at 15m, and a 15m meta-labeler gives 0.615
   < 0.647. So the 5-min stack asymptotes to ~0.65 and touches 0.648 only by starving OOS coverage (n→45).
8. **Cross-pair is the one orthogonal signal family that helped:** the USD-common-factor relative-value RESIDUAL reversion is
   sign-stable across 2024 & 2026 (momentum-continuation agreement is dead). It lifted the honest 5-min book 0.566→0.586.
9. **Walk-forward retraining (1-yr gap) helps the primary (+0.02) but does NOT break the wall:** the 2025 weakness is
   fundamental (near-efficiency), not a train-test-gap artifact. Reducing the gap lifted test25 only +0.011.
10. **Macro news is MAGNITUDE, not DIRECTION — even with the real data.** FX prices a surprise within ~1 minute (efficient
    jump); no exploitable 5-min directional drift (rule train ~0.50; model news-features rank bottom; news-window AUC ~0.51).
    The sign-invariance principle holds even for *fundamental* information. News-time is for straddle/touch/vol products.

## Deliverable models (all in this repo; `models/` + `*.parquet`/`*.npz` are gitignored — regenerate from the scripts)

- `m30_production.py` — 30m direction, compression-1h × NY selective ensemble, held-out **0.591** (CI[.556,.625]).
- `m15_production.py` — 15m direction, compression × NY, **0.647** combined — the best *tradeable directional* edge.
- `m_tick_prod.py` — 3s tick ensemble, **0.657** (the only >0.65 directional; needs a tick-expiry broker).
- `m5_production.py` — 5m direction, OHLCV ensemble, **0.566**.
- `m5_xpair_production.py` — 5m, cross-pair + order-flow primary → orthogonal meta-labeler, frozen **0.583** (oos 0.606), EV +0.078@R0.85.
- `m5_stack2.py` — 5m **CROSS-HORIZON STACK** (15m edge × 5m meta), **0.613 verifiable / 0.648 thin** — the best 5-min method (artifacts `models/m5stack_EURUSD_*`).
- `m30_magnitude.py` — the **magnitude/volatility** model (large-move AUC ~0.75) — the genuine new edge; tradeable on
  volatility/touch products. **Recommended productionization** (and where the macro-surprise data adds value).

## Research/experiment scripts (the lab — all reproducible)

30m: `m30_lab.py` `m30_regime.py` `m30_ens.py` `m30_tick.py` `m30_fix.py` `m30_es_feas.py` `m30_gates.py` `m30_complexity.py`
`m30_magnitude.py` `m30_sig.py` `tickhz.py`.
5m (session 1): `m5_lab.py` `m5_patterns.py` `m5_tick.py` `m5_sofien.py` `m_cnn.py` `m_tick_prod.py` `vbars.py`.
5m (session 2 — cross-pair/OF/meta/stack): `m5_xpair_probe.py` `m5_xpair.py` (MX_HOR param) `m5_xp_analyze.py` `m5_meta.py`
`m5_xpair_production.py` `m5_xhorizon.py` `m5_stack.py` `m5_stack2.py` `m15_meta.py` `m5_walkforward.py` `m5_wf_stack.py`
`m5_sofien_confluence.py`.
5m (session 2 — macro news): `fetch_calendar.py` (FXStreet API → `macro_calendar.parquet`) `event_signs.py` (event→EURUSD
direction map) `m5_news.py` (rule tests) `m5_news_model.py` (model-based conditioning).
10m: `m10_production.py` `m10_stack.py` `m10_gate_sweep.py` `m10_walkforward.py` `m10_xstack_probe.py` `m10_freeze_honest.py`
`m10_xasset_probe.py` `m10_magdir.py`.
1m (session 4 — lesson-transfer + new model classes): `min1_stack.py` (cross-horizon) `min1_hurst.py` (variance-ratio switch)
`min1_hmm.py` (Gaussian HMM, causal filtered) `min1_online.py` (river concept-drift) `min1_news60.py` (macro impulse)
`min1_updown.py`/`min1_upspec.py` (up/down asymmetry) `min1_kalman.py` (Kalman filter) `min1_kernel.py` (RBF kernel-SVM)
`min1_best.py` (lever-combination) · adversarial red-team: `_redteam_magdir60.py` `_redteam_trigger60.py` `_redteam_trigpop.py`
`_adj_perside_flow.py` · audit: `_verify_*.py` `_thr_sweep_audit.py`.
Refs: `sofien_rules.json` (79 mined rules) · `ENVIRONMENT_libs.txt`. Companion logs: `m5_research_log.md` (iter 1–19),
`m10_research_log.md`, `min1_research_log.md`, `IDEAS_LOG.md`.

**→ For applying any of these at a NEW horizon, start with [`METHODS_CATALOG.md`](METHODS_CATALOG.md)** — the timeframe-agnostic
catalog: how to retarget each method, the method×horizon results matrix (with the untested cells as a ready backlog), and the
leakage traps that recur at every horizon.
