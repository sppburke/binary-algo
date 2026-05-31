# 10-MINUTE EURUSD binary direction — research log

**Goal (set 2026-05-31):** a 10-minute EURUSD up/down model with **>65% accuracy, OOS-verified**. Be creative,
learn from prior lessons, log every iteration, use higher/lower timeframes & regimes, research deeply. OOS = held-out
2026 (plus stability across TEST2024/TEST2025). Settlement = deriv Rise/Fall mid-to-mid, breakeven ~0.541.

## Inherited Tier-1 lessons (from 1m/2m/5m/15m/30m sessions — do not relitigate)
- **Honest horizon map:** 1-5s ~0.65 · 8s 0.60 · 15s 0.58 · 30s 0.54 · 60s 0.52 · **5min 0.56 · 15min 0.64 · 30min 0.59**.
  The real directional edge is a SECONDS-scale phenomenon; it decays to noise by 60s, then the 15-min compression×NY
  regime is a separate, weaker local peak (~0.64). 10-min sits between 5 and 15.
- **15m winner** (m15_production.py): ensemble(lgb+xgb+cat) on 239 multi-TF features, gated to **15m_bb_width≤q AND
  sess_ny**, selective by confidence → held-out combined **~0.642**. This is the strongest tradeable book on liquid majors.
- **Cross-horizon stack** (m5_stack2.py): trade the 15m ensemble's DIRECTION on the shorter outcome + a META-LABELER
  predicting P(dir15 correct) from orthogonal axes (cross-pair agreement/dispersion, order-flow, conf) → 5-min **0.613
  verifiable / 0.648 thin-coverage**. The 5-min book is *capped by the 15-min parent* (data-bound 0.647).
- **Cross-pair residual reversion** is the one orthogonal signal family that is SIGN-STABLE across 2024 & 2026 (OHLCV
  momentum flips on 2025). Lifted the 5-min book 0.566→0.586.
- **News/macro-surprise is MAGNITUDE not DIRECTION** (FX prices a surprise within ~1 min). Null for 5-min direction.
- **Discipline:** select on VAL, then require the edge to hold across ALL of {TEST24, TEST25, OOS26} with CI95
  excluding breakeven. corr(VAL_acc,OOS_acc)=−0.54 → never VAL-max. **test25 is the persistent ceiling-breaker.**
- **Machine:** run trainings SEQUENTIALLY (concurrent heavy LightGBM OOM-killed the box before). venv
  `~/binary-algo-venv/bin/python`. 239 causal features in features/EURUSD_<year>.parquet; 7 majors; order-flow in
  features_of/; cross-pair engine m5_xpair.py (set env MX_HOR=10 → 10-min label + cross-pair features for free).

---

## Iteration 0 — dir15 → 10-min outcome transfer probe (m10_xstack_probe.py)
**Hypothesis:** 10-min is closer to 15 than 5-min was, so the already-trained 15m ensemble's direction should predict the
10-min sign BETTER than the 5-min sign (0.597 standalone → 0.648 stacked). Reuses models/m15_EURUSD_* (no retraining).

**Result (Tier-1, raw conf-selective, non-overlap 600s, CI95):**
- AUC(p15 vs 10-min label) ≈ **0.52** in every window (val .525 / t24 .519 / t25 .522 / oos .519). Weak.
- **GATE=NY**, selective by conf15: best combined **0.567** (thr .093, n3591), FLOOR 0.550. t24 0.601 / t25 0.550 / oos 0.555.
- **GATE=compression×NY**, selective: best combined **0.585** (thr .090, n1536), FLOOR **0.532**. t24 0.603 / t25 **0.532** / oos 0.653.

**Conclusion:** raw 15m→10min transfer is WEAKER than 15m→5min, and **test25 caps it at ~0.53** even as oos2026 looks
great (0.65). The raw transfer alone does NOT clear 0.65 with stability. BUT this is only the FLOOR of the stack method —
the 5-min lift came from the META-LABELER on orthogonal axes, not raw conf-selection. Next: (a) native-10 ensemble
(the 15m-native analog, the real baseline), (b) the full meta-labeler stack.

## Iteration 1 — native 10-min compression×NY ensemble (m10_production.py)
Direct analog of the 15m winner at HOR=10: ensemble(lgb+xgb+cat) on 239 feats, gate 15m_bb_width≤q × sess_ny,
selective by confidence; gate picked by **VAL-acc-max** (the production protocol). Raw VAL AUC=**0.5254** (lgb early-stopped
at 80 trees — weak raw signal, same as 5m/15m). SELECTED comp(q20)×NY @cov2% conf_thr=0.0837 VALacc=0.631.

**Backtest (Tier-1, independent non-overlap 600s, CI95):**
- 2024 n=305 acc=**0.705** CI95[0.652,0.754] · 2025 n=181 acc=**0.591** CI95[0.519,0.663] · 2026 n=57 acc=**0.702** CI95[0.579,0.807]
- **COMBINED n=543 acc=0.667 CI95[0.626,0.705]**, EV +0.233@0.85.

**Honesty flags (do NOT bank this as success):** (a) gate chosen by VAL-acc-max = the corr(VAL,OOS)=−0.54 anti-transfer
trap; (b) **test25 floor 0.591, its CI lower bound 0.519 dips below breakeven 0.541**; (c) cov2% thin (oos n57). The
0.667 leans on 2024+2026 (~0.70) while the binding window (2025) is 0.591. Must verify under HONEST selection → iter 2.

## Iteration 2 — HONEST gate sweep on native-10 (m10_gate_sweep.py)
Swept compression∈{5m,15m,30m bb_width, 5m/15m atr_pct} × session∈{ny,overlap,london,ny|overlap} × q∈{10,20,33,50} ×
cov∈{10,5,3%} = 200 configs; **select by worst-VAL-year-half stability (NOT VAL-acc-max)**, verify t24/t25/oos + CI95.

**Result (Tier-1):**
- **[HONEST worst-half-stable]** 15m_atr_pct×london q33 cov3% → FLOOR 0.549, COMB 0.576.
- **[ORACLE max-floor]** (hindsight-cheat, best held-out floor) 5m_atr_pct×overlap q33 cov3% → **FLOOR 0.599, COMB 0.603** (t24 .599/t25 .604/oos .613, n600).
- **[BEST VERIFIABLE oos n≥80]** same → COMB 0.603 CI[.565,.642], oos 0.613(n111).
- Top-10 by held-out floor: all FLOOR ~0.57–0.60, test25 ~0.57–0.60.

**Conclusion:** the native-10 **0.667 was a VAL-max/regime-luck artifact**. Under honest selection — and even under an
ORACLE that cheats by picking the best *held-out* floor — **no gate config gets all three windows above ~0.60**; test25
caps ~0.60. Native-10 honest tradeable operating point ≈ **0.59–0.60 combined** (profitable vs 0.541, NOT >0.65).
Interesting positive: **5m_atr_pct × overlap/london** (shorter compression + London/overlap session) gives a higher,
more *stable* floor than the 15m_bb_width×NY gate the 15m-winner used — a genuinely better-aligned 10-min gate.

## Research workflow (m10-research-scout) — COMPLETE (4 agents, 321k tokens)
Ranked 6 combinations; verdict converges with Tier-1 evidence. Honest 10m direction frontier **~0.58–0.62, test25-capped,
NOT robust >0.65**. Key warnings: test25 is the only window that matters; VAL-maxing anti-transfers; 15m parent ceiling
data-bound ~0.647 & not liftable so the cross-horizon stack inherits ≤~0.62–0.64 thin; thin-coverage inflation trap
(require oos n≥80); confirmed-null families (tick/OFI, signatures, news, complexity, full Sofien corpus, foundation
models) — do not re-attempt for direction. Only >0.65 OOS metric in this data = **magnitude** (|10m ret|≥Q75, AUC ~0.73–0.75),
a different product (touch/straddle), not Rise/Fall. Ranked next: (2) agreement stack, (3) dir15 meta stack — running now.

## Iteration 3 — cross-horizon stack variants (m10_stack.py dir15|dir10|agree)
dir15/dir10/agree primary, meta-labeler P(primary correct on 10m) on 39 orthogonal axes (cross-pair agree/disp, OF_*,
conf15/conf10, p15/p10, agreement, bbw, session), worst-VAL-half threshold; target = 10-min outcome.

**Result (Tier-1, non-overlap 600s, CI95):**
| primary | HONEST worst-half (FLOOR/COMB) | ORACLE max-floor | BEST VERIFIABLE oos≥80 (COMB, oos n) |
|---|---|---|---|
| dir15 | 0.534 / 0.590 | 0.556 | 0.584 (n516) |
| dir10 | 0.556 / 0.598 | **0.560** | 0.591 (n502) |
| agree | 0.546 / 0.587 | 0.546 | 0.587 (n350) |

**Mechanism (clear in the threshold curves):** as the meta threshold rises, **t24 climbs to 0.65–0.67 but test25 & oos
stay pinned ~0.55–0.57** and the FLOOR never exceeds ~0.56. The meta-labeler narrows coverage; it does NOT create edge in
the binding window. dir10 ≳ dir15 ≳ agree, but all three honest-combine ~0.59 and **every ORACLE max-floor ≤0.560**. The
5m precedent (agreement test25→0.601) does NOT reproduce at 10m. **Cross-horizon stack does not clear 0.65 at 10m.**

## Iteration 4 — cross-pair USD-residual + order-flow @ MX_HOR=10 (m5_xpair.py xpof)
The one sign-stable orthogonal family (5m: 0.566→0.586→0.594). 75 XP feats + 239 base + 18 OF (331 total), NY cov2% selective.
**Result (Tier-1):** VAL AUC **0.5262** (best_iter 201; +0.0008 over native — marginal). FROZEN NY cov2% VAL_indep_acc 0.640.
- test24 0.656(n829) · **test25 0.556(n1120)** · oos 0.604(n222) · **COMBINED 0.599 CI95[0.579,0.620]** (n2171).
- Top-20 feats include OF_kyle_15/5, OF_of_uptick_15, 4h_autocorr_10, hour_sin/cos — cross-pair residual cols do NOT
  dominate at 10m (unlike 5m where 8/20 were XP). The covcurve mirage is explicit: at conf 0.13, test24→0.735(n328) but
  **test25→0.559(n331)** — the high-confidence tip inflates t24/oos while the binding window stays ~0.55.

**Conclusion:** cross-pair+OF gives a diversified ~0.60 book (beats breakeven 0.541), but **test25 caps 0.556**; NOT >0.65.
The sign-stable cross-pair lift that helped at 5m is weaker/absent at 10m (the 30m finding that XP→+0 AUC at longer horizons).

## Iteration 5 — 10-min WALK-FORWARD (m10_walkforward.py)
Expanding-window annual retrain (1yr gap) directly targeting the 2025 regime wall. Cross-pair+base+OF, NY cov2%, non-overlap 600s.
**Result (Tier-1):** test24 0.668(n886) · **test25 0.573(n1347)** · oos 0.574(n148) · **COMBINED 0.609 CI95[0.589,0.629]**.
**Conclusion:** the 1yr-gap retrain lifts test25 0.556→**0.573 (+0.017)** — real but tiny (5m precedent was +0.011), and oos
DROPS 0.604→0.574 (the 2025 val-year used to set the threshold is itself a hard regime). Even adapting through 2024 to
predict 2025, the binding window only reaches 0.573. **The 2025 regime is genuinely hard to predict at 10m from any
available signal** — gap-reduction is not the missing lever. NOT >0.65.

## Iteration 6 — HONEST DELIVERABLE freeze (m10_freeze_honest.py)
Re-froze the native-10 ensemble on the healthiest honestly-selected gate (NOT the VAL-max cov2% tip): 5m_bb_width≤q20(VAL)
× sess_ny @cov10%. **Held-out (Tier-1):** test24 0.614(n1493) · test25 0.579(n741) · oos 0.594(n165) · **COMBINED 0.602
CI95[0.582,0.621] FLOOR 0.579**, EV **+0.113@0.85**. Saved models/m10_EURUSD_strategy_honest.json. This is the honest
tradeable 10-min book: ~0.60, all windows ≥0.579, healthy coverage, profitable vs 0.541 — but NOT a robust >0.65.

---

# VERDICT (2026-05-31): 10-min EURUSD DIRECTION caps at ~0.60–0.61 honest; >0.65 OOS-verified is NOT achievable on this data

Proven 6 independent Tier-1 ways (this session), every one capped by the **test25 (2025) regime** as the binding window:

| # | Method | Honest combined | test25 floor | Verdict |
|---|---|---|---|---|
| 0 | dir15→10m raw transfer probe | 0.585 (compxNY) | 0.532 | oos 0.653 is a thin-tip mirage |
| 1 | native-10, VAL-acc-max gate | **0.667** | 0.591 | artifact — buoyed by 2024+2026; 2025 CI touches breakeven |
| 2 | native-10, HONEST gate sweep (200 cfg) | 0.576 (ORACLE 0.603) | ≤0.60 even cheating | can't reliably select >0.60 floor |
| 3 | cross-horizon stack dir15/dir10/agree + meta | ~0.59 | ≤0.56 | meta narrows coverage, can't create edge |
| 4 | cross-pair USD-residual + order-flow | 0.599 | 0.556 | XP lift weaker/absent at 10m vs 5m |
| 5 | walk-forward (1yr-gap adaptive) | 0.609 | 0.573 | +0.017 only — regime genuinely unpredictable |
| — | **HONEST DELIVERABLE (native-10, healthy gate)** | **0.602 CI[.582,.621]** | **0.579** | ~0.60 tradeable book, EV +0.113@0.85, NOT >0.65 |

**Why:** raw 10-min directional AUC ≈ **0.525** (val), unconditional — the information content is ~noise. The compression×NY
gate concentrates the little edge that exists into a ~0.60 selective book, exactly interpolating the honest horizon map
(5min 0.583 < **10min ~0.60** < 15min 0.642). The recurring trap: every config that prints >0.65 on 2024 or 2026 collapses
to ~0.55–0.58 on 2025; corr(VAL_acc,OOS_acc)=−0.54 means you cannot SELECT the good config in advance. **No method —
native, gate-optimized, cross-horizon-stacked, cross-pair-diversified, or regime-adaptive — clears a robust >0.65.**

**The genuine >0.65 paths (unchanged from prior horizons):** (1) **MAGNITUDE** not direction — |10m return|≥Q75 is forecastable
at AUC ~0.73–0.75 (m30_magnitude recipe), tradeable on touch/straddle/volatility products, NOT Rise/Fall; (2) the
**seconds-scale** tick edge (mtick3 3s: 0.657/0.667), needs a seconds/tick-expiry broker, untradeable on a 10-min binary.
External cross-asset data (ES/NQ/DXY/rates) is the only untried directional lever but the 30m session found ES/NQ lead
corr≈0 and the research scout rates it "small lift, concentrated in release windows," not a path to robust 0.65.

**Deliverables:** m10_production.py (native-10 train/backtest), m10_freeze_honest.py (+ models/m10_EURUSD_strategy_honest.json),
m10_stack.py (cross-horizon stack), m10_gate_sweep.py, m10_walkforward.py, m10_xstack_probe.py. Models: models/m10_EURUSD_*.

## Iteration 7 — EXTERNAL cross-asset: ES (S&P500 e-mini) → EURUSD lead-lag (m10_xasset_probe.py)
The one untried *external* directional lever (the user said "leverage everything"). Minute-resolution ES futures exist on disk
(LEAN, /home/sean/git/reverse-engineered-trading/lean/data/future/cme/minute/es, UTC, front-month max-vol). (XAUUSD/DE30EUR
CFDs have ~no coverage — xauusd is only ~15 days of May-2014.) Risk-on channel hypothesis: ES↑ → USD weakness → EURUSD↑.

**Result (Tier-1, NY-gated):**
- CONTEMPORANEOUS corr(es_r1, eur_r1) = +0.195/+0.163/+0.216 for 2023/24/25 (validates UTC alignment + real risk link); 2026
  shows −0.002 at only 110k aligned mins → 2026 ES data sparse/unreliable, disregard.
- **LAGGED ES_ret_k → fwd-10m EURUSD corr: +0.009..+0.021 in 2023-2024, but −0.020..−0.054 in 2025 — the sign FLIPS**, and
  |corr|<0.055 everywhere (tiny). dir_hit 0.39–0.48 (inconsistent, mostly <0.5).

**Conclusion — null AND the mechanistic key to the whole wall:** ES gives no stable *lead* for 10-min EURUSD direction
(only untradeable contemporaneous co-movement), and critically the lead-lag **inverts between 2024(+) and 2025(−)**. The
2025 regime is one where the normal risk-on→USD-weakness relationship DECOHERES/inverts — **this is *why* every in-repo
method's test25 floor collapses to ~0.55-0.58.** External equity-futures data cannot fix it; it exhibits the same 2025
inversion. (NQ is ~0.95-correlated with ES → structurally redundant, same verdict.) Confirms the 30m ES-null at 10m.

## Iteration 8 — checked the rate-differential lever (the textbook EURUSD driver): DATA NOT ON DISK
EURUSD is fundamentally driven by the US−DE 2y rate differential — a DIFFERENT channel than the risk-on/ES link that
inverted in 2025, so potentially more regime-stable. Checked disk: EUREX minute = only `fesx` (Euro STOXX 50 equity — same
risk channel, redundant), CBOT minute = grains only (`zc/zs/zw`). **No Bund (fgbl), no Treasury futures (zn/zb/zf).**
Intraday rate-futures need a paid CME/EUREX feed; FRED yields are daily (useless at 10m). Lever real but **not feasible
without external data**. (Confirms the 30m verdict: US−DE rates = external data not in repo.)

**FINAL: 8 levers examined (7 tested Tier-1, all test25-capped; the 8th — intraday rate differential — is the one with
untested upside but its data does not exist on this machine). 10-min EURUSD direction honest ceiling ~0.60-0.61; robust
>0.65 is NOT achievable on any data available here. Deliverable = the honest ~0.602 native-10 book
(models/m10_EURUSD_strategy_honest.json). The ONLY untested-with-upside path requires acquiring intraday US/DE rate-futures
data; the genuine >0.65 in-data targets are MAGNITUDE (touch/straddle) and the seconds-scale tick edge (different broker).**

## Iteration 9 — direction CONDITIONED on predicted magnitude (m10_magdir.py): the last untested combination — NULL
Creative combine of the two pillars: magnitude IS forecastable (~0.73), direction isn't (~0.52). Test: does direction
become predictable on bars where a 10-min magnitude model predicts a BIG move? (sign-invariance theorem predicts null, but
it had never been RUN at 10m.) Trained a 10-min magnitude LGB (|fwd10|≥Q75) on 2012-21; bucketed held-out NY bars by
predicted magnitude; measured native-10 direction accuracy per bucket.
**Result (Tier-1):**
- **Magnitude model is STRONG: AUC 0.813(t24) / 0.741(t25) / 0.706(oos)** — forecastable & stable across all 3 held-out years.
- **Direction acc is FLAT ~0.51-0.53 across ALL magnitude quartiles** (Q4-HIGH = 0.516/0.515/0.518 — no better than Q1-low).
  HIGH-mag × dir-conf-top30% selective: combined **0.537, FLOOR 0.519** (WORSE than the compression gate 0.602).
**Conclusion:** clean empirical confirmation of sign-invariance — magnitude ⟂ sign; the bars with the biggest moves are
NOT more directionally predictable. The last creative in-data direction combination is null. **But the same run shows the
10-min MAGNITUDE model already clears the >0.65-equivalent bar (AUC 0.71-0.81)** — the ready-made genuine edge, tradeable on
touch/straddle/volatility products (NOT Rise/Fall up/down). 9 levers now; direction verdict is airtight.
