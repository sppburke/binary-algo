> **SCOPE: EURUSD × 10m** (key-specific EXECUTABLE BACKLOG — ranked TOP-N experiments, first-to-run queue, incumbents-to-beat, discovery rounds). Ledger/status: `sweeps/EURUSD_10m.md`. Results: `EURUSD_RESULTS.md`. Generic ideas: `IDEAS_LOG.md`.

# EURUSD × 10m — EXECUTABLE BACKLOG

## Incumbents to beat (per key)
- **(10m, COMBINED)** context-only incumbent: `EURUSD.m10.v1` honest freeze — comb .602 [.582,.621], **floor (worst window) .579**, 2026 .594.
- **(10m, UP)**: UNTESTED → floor = base-book UP side-CPCV p10 (pending L0). A new UP edge must beat that p10 with CI95-lo clearing it AND ≥80% CPCV paths clear 0.541.
- **(10m, DOWN)**: UNTESTED → floor = base-book DOWN side-CPCV p10 (pending L0). Same bar.
- Cross-key reference floors (do NOT copy — context only): 5m UP cert p10 .553 / DOWN dead; 15m UP cert .5673 / DOWN cert .5742 (cross-pair `EURUSD.m15xp.v1`).

## FIRST-TO-RUN queue (gradient-informed; one heavy job at a time)
1. **L0 — base-book side refit-CPCV** (`m10_cpcv_side.py`) — RUNNING. Output: honest (10m,UP)/(10m,DOWN) floors. ← step a+e for the base book.
2. **A6c — KEYSTONE cross-pair side refit-CPCV @MX_HOR=10** (`m10_xpair_cpcv.py`) — the real candidate. Pre-registered falsifier: KILL a side iff its refit p10 < 0.541 OR <80% paths clear OR p10 ≤ base-book floor (no improvement). Expect UP to certify; DOWN is the crux.
3. **If A6c certifies a side → IMPROVE that side** (Tier-I cross-product): I1 ACI gate → I3 magweight/GMADL → I2 seed-ens stack → I4 calibration → I5 pooling → I6 Optuna. Each: beat the certified p10's binding-year and/or raise frac-paths-clear. Freeze winning combo as `EURUSD.m10xp{,_aci,...}.v1`.
4. **If A6c does NOT certify DOWN (likely, given 5m) → DOWN rescue path**: I3 |ret|-weighted POW∈{0.5,1.0} retrain (rescued 5m DOWN to a thin cov0.05 edge), then specialist (A8c), then coverage curve (Ad). Show the wall before declaring DOWN dead.
5. **Steps (b)(c)(d)** for any side not yet certified-or-dead: worst-VAL-half gate, purpose-built specialist/meta-labeler, confidence/coverage curve.
6. **Discovery round 0** (after 1–5): retarget Tier-N N16–N19 + signed-payoff GMADL to MX_HOR=10; corpus mine for any 10m-specific direction lever. Loop until 2 dry rounds.

## Pre-registered falsifiers (write into each result JSON BEFORE OOS)
- **A6c (cross-pair side cpcv):** per side, CERTIFIED iff p10≥0.541 AND frac_clear_BE≥0.80 AND p10 > base-book floor. KILL otherwise. (Harness writes verdict; book floor read dynamically.)
- **I1 ACI:** WIN iff binding-year (worst held-out) win-rate ≥ fixed-gate at ≥ same coverage, regime-robust. Else null.
- **I3 magweight:** WIN iff a POW makes the side BOTH CPCV-certify (p10≥.541, ≥80%) AND point-clear the binding year. Razor-thin caveat if certifies at only one coverage.
- **specialist (A8c):** WIN iff separately-trained side beats the filter-on-symmetric side under refit-CPCV. Prior: spec usually WORSE (subset-training kills ranking) — design to KILL fast.

## Notes / traps to honor
- 10m has NO native `10m_bb_width` feature → gate uses `5m_bb_width` (validated as the m10 deployable gate). 15m used `15m_bb_width`.
- Same-harness baseline rule: A6c (cross-pair) vs L0 (base) use the SAME CPCV harness design → honest comparison. frozen-book forward ≫ refit-CPCV ⇒ overstatement → trust refit.
- pooled n≈5.29M, moved up-rate 0.5018 ∈ [0.47,0.53] ✓ (no fake-flat mirage).
- 10m < deriv 15m forex minimum → research horizon; the deployable sibling edge is 15m (`EURUSD.m15xp.v1`).

## Discovery toolkit (mapped 2026-06-04 — all retarget via MX_HOR=10; 5m verdicts inform 10m priority)
| lever | script | 5m verdict | mechanism / sign-carrying? | 10m prior | when to run |
|---|---|---|---|---|---|
| **N19 residual-label DOWN-rescue** | `m5_residlabel_down_h5.py` (→ H=10) | distinct DOWN mech; ran 5m | RELABELS target = sign(eu_fwd − β·basket_fwd): "did EUR fall MORE than the dollar implies" — removes common-USD factor from the LABEL (not a reweight). Carries DOWN sign by construction. | **med-high (DOWN crux)** | iff A6c leaves DOWN uncertified |
| **N18 signed-payoff GMADL/RRL** | `m5_signedpayoff_torch.py` (→ H=10) | COMPLETE 5m (verdict to re-read) | GMADL L=−σ(a·R·R̂)|R|^b couples SIGNED score to SIGNED return (sign-wrong big move penalized more); RRL diff-Sharpe tanh head. Distinct from symmetric magweight. | med | both sides, esp DOWN |
| I3 magweight POW sweep | `m5_magweight.py POW` (→ MX_HOR=10) | rescued 5m DOWN thin (p10 .5441, fwd-2025 CI-lo .533 grazes) | |ret|^POW upweight large-move bars (magnitude→direction bridge) | med | DOWN rescue |
| N17 lead-lag TE+RFF | `m5_leadlag_te_rff.py` (→ H=10) | KILLED 5m (trending null) | anti-contemporaneous: strictly-lagged cross-pair + TE gate + RFF VoC | low (fast-KILL) | run-once coverage |
| I2 seed-ens MLP⊕GBM | `m5_deep_ens.py` (→ MX_HOR=10) | blend≈GBM 5m (corr .694, 2025 .5776 vs .5765) | decorrelated DL ensemble member | low | iff a side certifies & needs tail |
| I1 ACI gate | `m5_conformal.py` (→ MX_HOR=10) | WON 5m | online win-rate-targeting threshold | high (improve) | needs frozen meta-gated 10m book first (`m5_xpair_production.py`@MX_HOR=10) |

**Note:** I1 ACI / I2 deep_ens depend on `m5_xpair_production.py` (the META-gated book; meta_feats = agree*/disp*/comp60/OF_*). 15m's certified book used the bb_width×NY gate (not meta) — `m10_xpair_freeze.py` mirrors that. **Heed `m15_xpair_regate` lesson: cov5% can be 2026-thin/fragile → prefer cov10% if the thinnest forward year's per-side n<50.**

## Discovery log
- 2026-06-04: toolkit mapped (above). Discovery round opens after L0/A6c — priority set by which side A6c leaves open.
