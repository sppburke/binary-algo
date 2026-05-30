# GOAL KICKOFF — Cheap Falsifiers First

**Read first:** `research/EXPERIMENT-BACKLOG.md`, `research/SYNTHESIS.md`, `research/GAPS-AND-CRITIQUE.md`,
`research_log.md` (V1–V17 ledger), `FINDINGS.md`. **Do not** re-run anything on the backlog's
"do-not-repeat" list.

**Why this scope.** Five literatures + our own 0.527 wall say always-on 75% at 15m on EURUSD does not
exist. The remaining probability mass is in three questions the feature-hunting never asked: *how to use
the fast edge we already have, what target to predict, and what instrument to predict.* Each falsifier
below is a **decision gate**, not a model to ship — ~1 day each, all on data in hand or free. Run them
**before** funding E1/E2/E3/E4. Any one can redirect the whole project.

## Standing context (already true)
- Data (read-only): `/media/sean/CORSAIR/tick_data/{processed,raw}` — 10s OHLCV + raw bid/ask ticks w/ sizes,
  7 USD pairs, 2012-01-02 → 2026-05-08. venv: `~/binary-algo-venv/bin/python`.
- Splits: TRAIN 2012–21 · VAL 2022–23 · TEST 2024–25 · **OOS 2026 locked**.
- Verified 3s edge exists: `models/probs_tickens_H3.npz` (TEST 0.756 / OOS 0.809 @0.05% cov).
- Best 15m so far: ~0.632 OOS @0.2% coverage (selective). AUC wall ~0.527.

## Validation contract (every experiment, non-negotiable)
1. Causal features only; purged + embargoed CV, embargo ≥ the label span at fold boundaries.
2. Pick threshold on **VAL only**, freeze, report TEST **and** 2026 OOS at that same threshold.
3. Deflate: Deflated Sharpe / PBO given ~17+ prior trials before believing any tail number.
4. Cost-realism: score net of half-spread from our own tick bid/ask. A 0.63@0.2% edge can be a spread artifact.

---

## F1 — Is the verified 3s edge compoundable to 15m?  *(highest EV — do first)*
**Question.** Can the real, fast-decaying 3s directional signal be optimally held into a 15m position whose
*sign over the window* inherits the edge, net of cost? If yes, **that is the project** and most feature ideas
are moot. If no, we have a rigorous reason 15m-from-microstructure is unreachable.
**Method.** Model the 3s signal (`probs_tickens_H3.npz`) as a noisy predictor with its measured decay + our
own per-trade tick cost; apply a Gârleanu–Pedersen "aim-in-front" aim-portfolio / Kalman-stack forward;
compute SNR and hit-rate of `sign(cost-aware position held 15m)`. Start as a back-of-envelope, then a sim.
**Gate.** Integrated fast edge clears break-even (~57% @0.75 payout) at 15m → pursue as the primary thesis.
Otherwise → record the negative result; microstructure→15m is closed.
*Source:* Gârleanu & Pedersen 2013 (DynTrad); arXiv:2502.04284 (alpha decay + cost).

## F2 — Cross-sectional rank target across the 7 pairs
**Question.** EURUSD is ~97% USD-factor. Does predicting the **rank / top-minus-bottom 15m spread** across the
7 pairs (dollar-neutral, strips the common factor) expose more predictable *idiosyncratic* signal than
per-pair sign? **Method.** Re-target the existing 239-feature stack to a learning-to-rank objective; bet the
top-minus-bottom spread; same OOS contract. **Gate.** Residual rank AUC materially > 0.527 → factor removal
is the lever. *Source:* arXiv:2105.10019.

## F3 — Touch-before-touch (±k) target vs end-sign
**Question.** Is "will price touch +k before −k within 15m" (the actual binary-option question) more
predictable than the sign of the 15m return? It rewards path/vol-asymmetry, not just drift. **Method.**
Relabel with a symmetric double-barrier (k in vol units from our tick spread); train the existing stack;
compare accuracy@coverage to end-sign. **Gate.** Touch target clears a higher accuracy at usable coverage →
switch the primary label.

## F4 — Exotic pair (less-efficient instrument, within FX)
**Question.** Does the *same* 239-feature stack that caps at ~0.52 on EURUSD reach materially higher on a
less-efficient liquid pair (USDMXN / USDZAR)? Cheapest "new instrument" test before any crypto pivot.
**Method.** Pull free Dukascopy history for the exotic; run `pipeline.py` + the baseline model; same OOS
contract. **Gate.** AUC materially > 0.52 → exotics are the FX path; if not → efficiency is the wall, consider
the crypto pivot (E16). *Source:* CME EM-FX; arXiv:0712.1624 (efficiency–predictability).

---

## Two correctness prerequisites to run alongside (cheap, high-leverage)
- **Re-derive the 0.632 under CPCV + Deflated Sharpe + adversarial validation** before building on it — with
  17+ trials it may not be distinguishable from 0.50. (Backlog E3 / critique item 8.)
- **Audit feed synchronization** (spot 15m bar boundaries vs true UTC vs any cross-asset feature; Dukascopy
  last-look/jitter) — misalignment both fakes backtest edge and destroys live edge. (Critique item 7.)

## Decision routing after the falsifiers
- F1 yes → pursue fast-edge aggregation as primary (new mini-backlog).
- F2 / F3 yes → re-target the pipeline (rank or touch) and re-run the directional overlays (E1/E2/E4) on it.
- F4 yes → port the stack to the exotic; F4 no + F1 no → crypto pivot (E16).
- All no → the honest deliverable is the conformal-hardened thin conditional book at break-even (E3), not 75%.
