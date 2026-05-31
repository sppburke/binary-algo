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
| 5 | **Online / concept-drift adaptive model** (river Adaptive Random Forest + ADWIN, PREQUENTIAL test-then-train, warm 2022-23) — tests whether the 2025 wall is stale-model drift vs genuine efficiency (`min1_online.py`) | **NOISE FLOOR everywhere**: AUC 2024 0.504 / 2025 0.503 / 2026 0.505; selective acc 0.49–0.51 at every confidence cut, all windows. Continuous adaptation recovers NO direction edge → the 2025 wall is genuine efficiency, NOT stale-model drift | 0.503–0.508 | ❌ |

### Key mechanistic confirmations
- **HMM states are pure volatility/size regimes:** all 3 states have train P(up) = 0.497–0.499, and momentum loses in every state
  (mom_acc ≈ 0.49) → every state maps to the reversion engine. The HMM re-discovers, by likelihood, the same vol/compression
  regime our hand-coded switches found — confirming the sign-invariance theorem (arXiv:2512.15720) empirically at 60s.
- **The 2025 wall is GENUINE EFFICIENCY, not a stale-model / concept-drift artifact:** the online adaptive learner (river ARF +
  ADWIN, prequential, continuously re-fitting to recent bars) sits at AUC 0.503–0.505 in EVERY window, including 2025 — it cannot
  recover an edge by adapting, because there is no stable 60s direction signal to adapt to. This rules out the one hypothesis under
  which a smarter model could break the wall.
- **Five independent model classes converge:** cross-horizon, cross-pair, persistence-switch, HMM, and online-adaptive all drive
  2024 & 2026 to ~0.55–0.61 selective but the binding 2025 window stays ~0.50–0.61, with the floor's OOS CI95 touching breakeven;
  even the ORACLE (hindsight) gate floor is ~0.555–0.60, never 0.65.
- **The best-transferring config is the HMM vol-state-gated reversion book (U2): floor ~0.60 across all three years** — a modest
  refinement of `min1_production`'s 0.55, but at thin coverage (n≈106–162/window), OOS CI95 spanning breakeven, and NOT >0.65.

## Verdict (Session-4)
**A 1-minute EURUSD up/down model with >0.65 OOS-stable accuracy is NOT achievable on this data.** The 60s direction is the
efficient-market part (~0.50–0.51 AUC across every model class — LGBM, ensemble, large-move GBM, 1D-CNN, HMM, AND an
online drift-adaptive forest); the only real edges are MAGNITUDE (~0.68 AUC, a touch/straddle target, not up/down) and the
compression-release × reversion REGIME harvested selectively to ~0.55–0.60 at thin coverage. The 2025 wall is GENUINE efficiency
(the online learner can't beat it by adapting), not a fixable modeling artifact. Honest best deliverable: the HMM-vol-state-gated
reversion book at ~0.60 floor (documented; ~breakeven-significant, NOT >0.65, synthetic-index-only venue since deriv forex
Rise/Fall floor = 15 min). The existing frozen `min1_production.py` (0.539/0.550) remains the reference book.

**Answer to "would HMMs help?" — No (empirically + literature-confirmed):** a Gaussian HMM's latent states are volatility/size
regimes (train P(up)≈0.50 in all states), not direction states; gating/switching/meta on them lands ~0.49–0.60, never >0.65,
matching the academic record (Markov-switching cannot beat a random walk OOS for FX direction; the FX-HMM literature switches the
variance, not the sign). HMM is a magnitude/risk tool, not a 60s-direction tool. See the HMM research workflow in the session log.
