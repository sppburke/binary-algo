# EURUSD · 30m — EXECUTABLE BACKLOG (key-specific)
SCOPE: EURUSD · 30m. Key-specific executable backlog — incumbents, scripts, numbers, discovery rounds.
Moved out of IDEAS_LOG.md on 2026-06-02 during the generic↔specific split (see REPO_MAP.md).
Generic methods/ideas: METHODS_CATALOG.md / SWEEP_MATRIX.md / IDEAS_LOG.md. Sweep ledger (status): sweeps/EURUSD_30m.md. Results of record: EURUSD_RESULTS.md.

---

## EXECUTION ORDER (live)
- [RUNNING] Workflow: Sofien gate mining (larger-TF reversal / flag gates).
- [RUNNING] Workflow: non-time bars / volume-price / VPIN / meta-labeling research.
- [TODO] Workflow: orthogonal-math academic research (A-F above), with reported numbers + implementability.
- [BUILDING] vbars.py: volume & dollar bars → 30m label, honest selective.
- [TODO] m30_complexity.py: permutation entropy + Hurst/DFA + RQA-DET + Lempel-Ziv gates on the 30m model; "avoid losers".
- [TODO] signatures as features; Hawkes intensity on event bars; transfer entropy volume→price.

## RESULTS (append as they come — be honest, cross-window-stable only)
(none yet this thread)

## RESULTS LOG (2026-05-31, honest, cross-window)
- **Larger-TF reversal/flag gates** (1h/4h/daily RSI/DeMarker/WillR/CCI/BB/z/Fisher extremes; confluence; chop-gated) — m30_gates.py: ALL null, none >=0.60 stable across held-out. Oscillator extremes at larger TFs do NOT predict 30m reversals.
- **Complexity/predictability gates** (Permutation Entropy d3 W30/60/120; lag-1 autocorr ac20; variance-ratio Hurst) — m30_complexity.py: NULL. Conditional accuracy is FLAT across all bins of all measures: momentum ~0.48, reversion ~0.515. Complexity does NOT isolate a predictable subset. Only signal = uniform weak ~0.515 mean-reversion (below 0.541 breakeven). The "low-entropy = tradeable" hypothesis FAILS for 30m EURUSD.
- IMPLICATION: 30m direction is ~0.515 mean-reverting, uniform, no regime/complexity gate isolates >0.58. Next untested: VOLUME/DOLLAR BARS (non-time grouping), VPIN, Hawkes intensity on event bars, path signatures, transfer entropy vol->price, Sofien's 79 mined rules (pivots/overextension-release/confluence).

## POSITIVE FINDING (validated OOS) — the real "different story": MAGNITUDE is forecastable, DIRECTION is not
m30_magnitude.py: target = 30m forward |return|.
- realized vol (rv30): corr with future|ret| = +0.42/+0.44/+0.44/+0.40/+0.38 (train/val/t24/t25/OOS). Large-move (|ret|>=Q75) classification **AUC 0.75/0.75/0.78/0.73/0.73 — STRONG, STABLE, OOS-verified**.
- permutation entropy: corr ~0.01, AUC 0.50, flat |ret| across quintiles — PE does NOT predict magnitude on EURUSD (the SPY-trades result doesn't replicate on 1m FX returns; realized vol is the real magnitude predictor).
=> CONFIRMED: 30m EURUSD MAGNITUDE/volatility is ~0.73-0.78 AUC forecastable; DIRECTION/sign is ~0.52 (EMH). The predictable structure is magnitude, exactly as the sign-invariance theorem predicts.
IMPLICATION for up/down binary: magnitude predictability lets you AVOID small-move/tie losers and size bets, but does NOT raise directional accuracy (sign within large moves is still ~0.515). Tradeable as a volatility/touch/straddle product, not up/down.

## THREAD SUMMARY (this session's orthogonal/creative push, all honest cross-window)
Tested & NULL for 30m DIRECTION: larger-TF reversal gates; complexity gates (PE/WPE/Hurst/ac20); Sofien 79 rules
(pivots/IBS/Connors-RSI2/TD-setup/BB-reentry/confluence); volume & dollar bars; (theory) transfer-entropy/Hawkes are
sub-minute/hourly/trader-resolved or sign-invariant. POSITIVE & VERIFIED: magnitude forecastable (rv, AUC ~0.75).
Untested-but-low-prior (sign-aware): path signatures (CNN-on-raw-path already null 0.525), 4D Bacry-Muzy Hawkes intensity
(needs LOB events, sub-minute). Direction at 30m remains ~0.52-0.515 mean-reverting; no method this thread broke it.

## PATH SIGNATURES (Lévy area, rough-path theory) — last sign-aware orthogonal lever (m30_sig.py)
Lévy area A(price, order-flow imbalance) = signed area = lead-lag ROTATION, computed causally over W={30,60,120,300}s.
- HS=1800 (30m GOAL): AUC val 0.515 / TEST 0.508 / OOS 0.509 — NULL. Model barely fits (1 tree). Signatures don't help 30m direction.
- HS=5s: signature Lévy areas rank TOP-5 importances; selective ~0.63-0.66 (= base tick edge). Order-flow→price path rotation IS directional AT SECONDS, decays to noise by 30m.
=> FINAL: every sign-aware orthogonal method (signatures, and by extension Hawkes/transfer-entropy which capture the same lead-lag) confirms: directional edge lives at the SECONDS scale; 30m direction is EMH (~0.515).

## DEFINITIVE THREAD CONCLUSION (orthogonal/creative push, 2026-05-31)
30m EURUSD DIRECTION is not predictable beyond ~0.515 by ANY method tried (now ~25 experiments + 4 research workflows +
a sign-invariance theorem). The predictable structure is MAGNITUDE (realized vol -> large-move AUC ~0.75 OOS). The only
directional edge is at the SECONDS scale (~0.65, confirmed by signatures). For an up/down 30m binary, >0.65/>0.75 is not
achievable on EURUSD; the honest tradeable edges are: 15m compression×NY ~0.64, 3s tick ~0.66, and the NEW magnitude/vol
model ~0.75 (volatility/touch products, not up/down).
