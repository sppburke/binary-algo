# 1-MINUTE EURUSD DIRECTION — Session-4 re-push (2026-05-31d)

**Goal (re-set):** "move back to the 1-minute prediction, give me a model with prediction accuracy > 0.65, OOS-verified.
Be creative, learn from previous lessons, leverage everything, don't give up." EURUSD, binary up/down, deriv-faithful.

This log covers the **lesson-informed re-attack** on the 60-second horizon, applying the methods discovered AFTER the original
`min1_production.py` book was frozen (cross-horizon stack, cross-pair USD-residual, meta-labeling, and — at the user's request —
Hidden Markov regime models), plus a thorough multi-agent research pass. Companion: `DIRECTION_FINDINGS.md`, `research_log.md`
(the original 1-min iterations v1–v15), `m5_research_log.md` / `m10_research_log.md` (the lessons being transferred).

## Discipline (unchanged, non-negotiable)
Select params on VAL; then REQUIRE the edge to hold on EACH of {2024, 2025, 2026} with bootstrap CI95 excluding breakeven
(~0.541). Non-overlapping 60s trades; deriv mid-to-mid, ties LOSE; entry = next tick (1s lag). corr(VAL_acc, OOS_acc) = −0.54,
so NEVER VAL-acc-max — report the worst-VAL-half-stable pick, the verifiable-coverage row, and the ORACLE ceiling separately.
Pre-commit ONE config per method. Tick book split: train 2021-23 / val 2024-H1 / test 2024.09–2025.11 / oos 2026.

## Hard prior carried in (Tier-1, our own logs)
- 60s DIRECTION is near-efficient: ~0.50–0.51 AUC across single LGBM (v1), 3-model ensemble (v3), large-move-only GBM (v13),
  and a 1D-CNN on the raw 1s path (v14, loss stuck at ln2, AUC 0.49). No model class beats it. MAGNITUDE is forecastable
  (|ret60| AUC ~0.68) — a different, non-up/down target.
- The honest `min1_production.py` book (reversion × compression-release, 5% coverage) = **TEST 0.539 / OOS 0.550** (CI includes 0.50).
- The binding constraint at every horizon is the **2025 regime** ("test25"): each method drives 2024 & 2026 to ~0.60–0.70 but
  2025 stays ~0.53–0.58, because the macro risk-on/USD relationship inverts in 2025 (ES→EURUSD lead-lag flips +0.02→−0.05).
- **VENUE:** deriv EUR/USD forex Rise/Fall min expiry = 15 min; a 60s EURUSD book is placeable only on synthetic indices.

## Research pass (two multi-agent workflows)
1. **Prior-art audit + infra + sofien (4 agents):** confirmed the 15 `min1_v*` iterations were exhaustive; the cross-horizon
   STACK, cross-pair residual, order-flow, and meta-labeler were GENUINELY untried at 1-min (grep `m5/m15/m10_EURUSD` in
   `min1_*`/`min2_*` = 0 hits). Surfaced new orthogonal regime levers from the sofien corpus: Hurst/R-S persistence, cross-pair
   lead-lag (CCF), Shannon-entropy abstain gate, dCor/Kendall coupling stability, HVR. Honest outlook: >0.65 unlikely; expect
   "cleared breakeven, did not clear 0.65".
2. **HMM research (4 agents, user-requested):** literature consensus (Empirical Economics 2019 — MS can't beat a random walk
   OOS even at BIC-optimal regime count; Dueker-Neely — FX HMM switches the VARIANCE not sign; jump-model arXiv:2402.05272 —
   benefit is risk reduction; Christensen 2020 intraday momentum-HMM — equity futures, lag-reduction, no cost-net sign edge).
   Verdict: an HMM gates VOLATILITY/magnitude, not direction; will not reach >0.65. No HMM had ever been run in this project.

## Iterations (this session) — all deriv-faithful, per-window 2024/2025/2026, CI95

| # | Method (file) | Honest selective result | test25 | >0.65? |
|---|---|---|---|---|
| 1 | **Cross-horizon stack** — 5m/15m/agree/softavg parent direction front-loaded into 60s, meta-gated (`min1_stack.py`) | worst-VAL-half COMBINED ~0.586–0.594; **verifiable-coverage (oos n≥100) 0.53–0.55** | 0.56 honest / 0.53 verifiable | ❌ |
| 2 | **Cross-pair USD-residual direct**, MX_HOR=1 xpof (cross-pair + 239 base + order-flow) (`m5_xpair.py`) | VAL AUC **0.516**; combined 0.560; covcurve test25 ≤0.56, oos ≤0.55 | 0.534 (CI[.518,.550]) | ❌ |
| 3 | **Hurst / variance-ratio persistence switch** — momentum-vs-reversion engine switched by VR(q,W); orthogonal to compression (`min1_hurst.py`) | HONEST worst-VAL-half: 2024 0.518 / 2025 0.548 / 2026 0.513, FLOOR 0.513; **ORACLE max-floor (hindsight) only 0.555** | 0.548 honest | ❌ |
| 4 | **Hidden Markov regime model** (K=3 Gaussian HMM, CAUSAL filtered forward posteriors, no look-ahead) — U1 gate / U2 engine-switch / U3 soft-posterior meta (`min1_hmm.py`) | states carry NO direction (train P(up)≈0.497–0.499 in all 3, momentum loses in every state); **U1** floor 0.565 (2024 .581/2025 .569/2026 .565, OOS CI[.492,.638]); **U2** floor 0.600 (2024 .600/2025 .605/2026 .613, OOS CI[.519,.708] spans breakeven, n≈106–162); **U3** anti-transferred to floor 0.487 (2025 0.508, gamma posteriors got top importance but replayed the corr(VAL,OOS)=−0.54 trap) | U1 0.569 / U2 0.605 / U3 0.508 | ❌ |
| 5 | **Online / concept-drift adaptive model** (river Adaptive Random Forest + ADWIN, PREQUENTIAL test-then-train, warm 2022-23) — tests whether the 2025 wall is stale-model drift vs genuine efficiency (`min1_online.py`) | **NOISE FLOOR everywhere**: AUC 2024 0.504–0.506 / 2025 0.503–0.508 / 2026 0.505–0.508; selective acc 0.49–0.51 at every confidence cut, all windows. **Robust across capacity** — both a depth-capped 5-tree forest AND a full uncapped 10-tree forest give the identical null. Continuous adaptation recovers NO direction edge → the 2025 wall is genuine efficiency, NOT stale-model drift | 0.503–0.508 | ❌ |
| 6 | **Macro-release 60s directional impulse** — pre-committed prediction = sign(eurusd_signal) from the surprise (actual−consensus mapped via event_signs), traded in the 60s window after USD/EUR releases; the 5m news null was retested at the IMPULSE timescale (`min1_news60.py`, `macro_calendar.parquet` 7650 signed events, 24.6M ticks) | **NULL / sign-unstable OOS**: HIGH-vol 2024 0.373 / 2025 0.526 / 2026 0.517; HIGH-vol & \|surp_z\|≥1 → 0.167 / 0.364 / **0.000** (the bigger the surprise, the MORE wrong the surprise-sign rule is OOS). EURUSD frequently moves AGAINST the surprise sign at 60s (fade / overshoot-revert / priced <60s). No window clears n≥25 & CI95-lower>0.65 | 0.526 | ❌ |

### Key mechanistic confirmations
- **HMM states are pure volatility/size regimes:** all 3 states have train P(up) = 0.497–0.499, and momentum loses in every state
  (mom_acc ≈ 0.49) → every state maps to the reversion engine. The HMM re-discovers, by likelihood, the same vol/compression
  regime our hand-coded switches found — confirming the sign-invariance theorem (arXiv:2512.15720) empirically at 60s.
- **The 2025 wall is GENUINE EFFICIENCY, not a stale-model / concept-drift artifact:** the online adaptive learner (river ARF +
  ADWIN, prequential, continuously re-fitting to recent bars) sits at AUC 0.503–0.505 in EVERY window, including 2025 — it cannot
  recover an edge by adapting, because there is no stable 60s direction signal to adapt to. This rules out the one hypothesis under
  which a smarter model could break the wall.
- **Six independent levers converge:** cross-horizon, cross-pair, persistence-switch, HMM, online-adaptive, and macro-release
  impulse all drive 2024 & 2026 to ~0.45–0.61 but the binding 2025 window stays ~0.50–0.61, with the floor's OOS CI95 touching
  breakeven; even the ORACLE (hindsight) gate floor is ~0.555–0.60, never 0.65.
- **Even information shocks don't give a 60s sign edge:** the macro-release impulse — the one moment when directional information
  demonstrably exists — is null/negative OOS (surprise-sign accuracy 0.17–0.53, worse for larger surprises), because FX prices the
  surprise in <60s and frequently overshoots-and-reverts. This closes the last "but surely *X* carries direction" objection.
- **The best-transferring config is the HMM vol-state-gated reversion book (U2): floor ~0.60 across all three years** — a modest
  refinement of `min1_production`'s 0.55, but at thin coverage (n≈106–162/window), OOS CI95 spanning breakeven, and NOT >0.65.

## Verdict (Session-4)
**A 1-minute EURUSD up/down model with >0.65 OOS-stable accuracy is NOT achievable on this data.** The 60s direction is the
efficient-market part (~0.50–0.51 AUC across every model class — LGBM, ensemble, large-move GBM, 1D-CNN, HMM, AND an
online drift-adaptive forest); the only real edges are MAGNITUDE (~0.68 AUC, a touch/straddle target, not up/down) and the
compression-release × reversion REGIME harvested selectively to ~0.55–0.60 at thin coverage. The 2025 wall is GENUINE efficiency
(the online learner can't beat it by adapting; even macro-release information shocks give no 60s sign edge — surprise-sign
accuracy 0.17–0.53 OOS, worse for larger surprises), not a fixable modeling artifact. Honest best deliverable: the HMM-vol-state-gated
reversion book at ~0.60 floor (documented; ~breakeven-significant, NOT >0.65, synthetic-index-only venue since deriv forex
Rise/Fall floor = 15 min). The existing frozen `min1_production.py` (0.539/0.550) remains the reference book.

## Adversarial red-team (3-agent workflow + adjudicator) — independent corroboration + 4 NEW experiments

After the 6 levers, a 3-agent red-team was explicitly tasked to BREAK the "exhausted" verdict (literature/web, codebase+data,
creative reframing) + an adjudicator. All four converged on **EXHAUSTED**, and — instead of only arguing — ran four genuinely-new
Tier-1 experiments (scripts: `_redteam_magdir60.py`, `_redteam_trigger60.py`, `_redteam_trigpop.py`, `_adj_perside_flow.py`):

- **PILLAR 1 — the informational ceiling (the definitive proof).** The 2025 60s directional conditional-accuracy ceiling is
  **0.512 at 10% coverage rising only to 0.517 at 0.5% coverage (n=11,283), CI95 upper ~0.53.** Going from 10%→0.5% coverage moves
  2025 by 0.5pt — so the ~13-point gap to 0.65 is **informational, not a coverage problem.** No meta / gate / selective subset over
  these features can reach 0.65 on the inversion year. (60s dirAUC 0.510 vs magAUC 0.787 — sign-invariance, same data.)
- **PILLAR 3 — the last untried on-disk axis is dead.** The raw per-side **bid-vol/ask-vol order flow** from `tick_data/raw/`
  (7 majors, 2012-2026, verified real & asymmetric — never used before; only top-of-book `imb` proxy existed) was built from scratch
  (net/tick-rule-signed/imbalance over 5/15/30/60s), clean EURUSD-only (no ffill artifact), tested on **moved bars** (|fwd|>0.5pip):
  VAL dirAUC **0.5086**; 2025 selective 0.498–0.506, CI95 upper never >~0.52, NEGATIVE at deep selectivity. Null.
- **Trigger-carry & subpopulation:** the 5s tick ensemble (which clears ~0.65 at 5s) carried to the 60s outcome = VAL AUC **0.5089**;
  2025 collapses to 0.48–0.53. The seconds-scale edge is bounce/reversion microstructure that does NOT propagate to 60s.
- **A caught FALSE POSITIVE (discipline working).** A 7-pair lagged-return LGBM superficially showed AUC 0.728 / selective CI95-lo>0.65
  *every year* — unmasked as a label artifact: the 7-pair timestamp-intersection+ffill manufactured ~50% fake flat windows
  (train up-rate 0.338 vs real EURUSD-1s up-rate 0.492; real flat only 1.4%). On moved-bars-only it collapses to 0.49–0.51. This is
  exactly the leakage a naive run would have reported as "GOAL ACHIEVED" — and the discipline killed it.
- **Literature:** no peer-reviewed/practitioner method clears >0.65 OOS sub-5-minute major-FX DIRECTION after costs. The one >0.65
  claim (Lee 2024, arXiv:2409.14157) is EQUITIES + full Nasdaq depth-10 volume imbalance (not on disk) + the authors' OWN documented
  leaky target (naive lagged-target predictor scores 64.8%); strip the leak → 0.50 ("supporting EMH"); the 67.5% is VOLATILITY.
  VPIN/order-flow toxicity = magnitude not sign; options risk-reversal = daily + no minute feed; news-NLP wins are daily horizon.

## Up-vs-down asymmetry (user idea) + scholarly-literature ceiling

**Up/down asymmetry (`min1_updown.py`, `min1_upspec.py`):** the symmetric book's edge lives ENTIRELY on the UP (dip-buy) side.
Splitting the frozen book's independent trades by predicted direction: UP-predictions 2024 0.520 / 2025 **0.584** / 2026 **0.613**;
DOWN-predictions 0.522 / 0.522 / 0.516 (dead noise every year). Mechanism: the reversion lever buys dips / sells rallies, and
dip-buying worked while rally-selling didn't over the EUR-up 2025-26 regime — but the asymmetry VANISHES in 2024 (up 0.520 vs down
0.522, a wash), so it's regime-dependent. Two refinements tested:
- **Up-only FILTER on the symmetric model: a genuine improvement** (2025 0.584 / 2026 0.613 vs symmetric 0.548 / 0.550), floor 0.520
  (the 2024 wash), CI95 does not clear 0.65 on any window. Best honest 1-min directional book of the session.
- **Dedicated up-SPECIALIST (separately trained only on dip-buy setups): WORSE — 0.498 / 0.537 / 0.519.** Subset-training destroys
  the confidence ranking; reproduces the v4/v6/v13 lesson exactly. The down-specialist control = 0.472 / 0.521 / 0.481 (dead).
  Setup base rates are ~coin-flip (dip-bounce 0.507, rally-down 0.514), so there is little for a specialist to sharpen.
Takeaway: treat the sides asymmetrically as a trade FILTER on the symmetric ensemble, never as separately-trained models; it lifts
the favorable-regime windows to ~0.58-0.61 but does not break 0.65 and is regime-dependent (2024 wash).

**Scholarly-literature deep search (3-agent workflow + adjudicator — answers "is ~0.60 the best you can do?"): YES.** Verdict
`beats_our_060 = False`, `LITERATURE_CONFIRMS_CEILING`. The best DEFENSIBLE published EUR/USD DIRECTION is **0.585 at the DAILY
horizon only** (Castillo 2024, arXiv:2409.04471 — not cost-netted, rides 2022 trend-luck); the honest **sub-5-minute / 60s sign
ceiling is ~0.52-0.55 net of costs** (Petrova-Vilhelmsson-Nordén, Int. J. Forecasting 2025 — FX LOB features show low predictability
"supporting EMH"; order-flow impact vanishes by 15-30 min). The famous high numbers don't apply: DeepLOB 83% F1 / OFI R² 87% are
EQUITIES + depth-10 volumes we lack + event-horizons + smoothed overlapping labels (leakage) + magnitude-not-sign; the one real
1-5s queue-imbalance SIGN signal moves WITHIN the spread (not executable after FX cost + binary tie-loss); fancy DL (TFT/xLSTM/
N-HiTS) gives no short-horizon sign edge (the daily 73% F1 was a wavelet-denoising lookahead leak); even the seconds-scale MKL study
(Fletcher 2010, EBS EUR/USD) shows 3-class accuracy peaking ~0.50 at 20s and decaying. Our ~0.60 selective book is at/above the
published frontier; our broad-coverage 0.55 matches the literature's 60s number exactly. **>0.65 at 60s is unsupported anywhere.**

## Answer to "would HMMs help?" and the final verdict

**Answer to "would HMMs help?" — No (empirically + literature-confirmed):** a Gaussian HMM's latent states are volatility/size
regimes (train P(up)≈0.50 in all states), not direction states; gating/switching/meta on them lands ~0.49–0.60, never >0.65,
matching the academic record (Markov-switching cannot beat a random walk OOS for FX direction; the FX-HMM literature switches the
variance, not the sign). HMM is a magnitude/risk tool, not a 60s-direction tool. See the HMM research workflow in the session log.
