# CCM Coupling-Gate Experiment — Design Document (`min1_ccm.py`)

Target: 60-second EURUSD binary direction (deriv.com Rise/Fall). Design pass 2026-06-01.
On-disk long-shot #2 (after cross-impact OFI / `min1_xofi.py`). Honest prior: **~5–10% chance of a tradeable gate; a documented null is the expected, valuable outcome** — 60s direction is near-efficient across ~24 channels.

## 1. Hypothesis
CCM (Sugihara et al., *Science* 2012) detects directional dynamical coupling `X → Y` by testing whether Y's time-delay **shadow manifold** can cross-map (recover) X, with skill that **converges** (rises) as library length L grows. Convergence is the signature separating causation from correlation.

- **Y (manifold / trading clock):** EURUSD log-return on its 1s clock (`Y1`) and EURUSD signed OFI `cks_e` (`Y2`).
- **X (7 drivers):** EURUSD own OFI (self-coupling sanity) + 6 cross-pair USD-direction-aligned signed OFI (GBP/AUD/NZD keep sign; JPY/CHF/CAD flip).
- **Tradeable refinement (lagged CCM, Ye et al. 2015):** require the cross-map skill peak at a **forward** `tp* ∈ [1,60]s` (driver's past predicts EUR's manifold) — `tp* ≤ 0` (EUR leads) is a non-tradeable discard.
- **Product = a GATE, not a feature.** High coupling ⇒ EURUSD 60s dynamics are more driver-led/deterministic ⇒ the **frozen production sign model** should be more accurate there. CCM decides *when to trade*; direction comes from `min1_production` `_blend`. This is leakage-immune (no new feature fed to the model).

## 2. Embedding & algorithm
- **E** per (X,Y) via simplex-projection argmax ρ + FNN<1% crosscheck; default **E=4** (high-noise micro-returns; E>6 over-fragments).
- **τ=1 step** (de-correlated returns; AMI first-minimum). Embed **returns/OFI** (stationary), never price levels.
- **Irregular-clock fix:** resample to a **fixed 1s grid** for the embedding ONLY (returns summed, OFI summed, **empty second → 0.0, NEVER ffill**). Labels/trades/accuracy stay on EURUSD's native clock via `wc_ret`. **The resampled grid never touches a label, trade, or accuracy number** — this is the firewall against the fake-flat mirage (intersection+ffill → fake AUC 0.7 collapsing to 0.49).
- **Cross-map:** E+1 nearest neighbors in M_Y (cKDTree), **Theiler window ±(E·τ+tp)** mandatory, exponential weights `w_i=exp(−d_i/d_1)` normalized; `X̂(t+tp)=Σ w_i X(t_i+tp)`; skill ρ = Pearson(X̂, X).
- **Convergence:** L∈{50,100,200,400,800,Lmax}, B=30 subsamples, slope of median ρ vs log L; require slope>0, Δρ≥0.05, Spearman≥0.6.

## 3. Coupling gate
- Non-overlapping **30-min windows** on EURUSD clock (drop windows straddling >5-min gaps).
- Per-window score `S_X(w)=ρ(Lmax,tp*)` if convergence holds else 0.
- **Threshold θ_X selected on VAL worst-half (NEVER acc-max):** smallest θ whose worst-VAL-half moved-bars CI95-lower ≥ 0.515.
- Trade gate-open windows with frozen model; `nonoverlap_chrono`+`wc_ret`; **moved-bars-only**, per-year 2024/25/26, `boot` CI95. Verify moved up-rate ∈ [0.47,0.53] (mirage tripwire).

## 4. Surrogate null
- **Ebisuzaki phase-randomized** (B=100): preserves power spectrum/autocorr/seasonality, destroys causal phase alignment. Observed ρ must exceed surrogate **95th pctile** (kills shared-session-cycle synchrony).
- **Twin-surrogate** confirmatory on Ebisuzaki-passers (guards nonlinearity).
- **Common-driver guard:** compute reverse ρ_EUR→X; if ≈ forward and both high, flag `bidirectional_warn` (not auto-kill).

## 5. Pre-registered falsifier (KILL if ANY)
1. **No convergence:** every driver slope ≤ +0.02 OR Δρ < 0.05 on VAL.
2. **No surrogate separation:** no driver's ρ exceeds Ebisuzaki 95th pctile.
3. **No forward lead:** every converging driver has tp* ≤ 0.
4. **Gate fails OOS-stably (decisive):** no gated subset clears moved-acc CI95-lower ≥ 0.515 in **each** of 2024, 2025, 2026.
5. **Mirage tripwire:** any gated moved up-rate ∉ [0.47,0.53] OR moved_frac < 0.40.
6. **Sanity floor:** EUR-own-OFI self-coupling is NOT the strongest convergence (embedding broken → fix before trusting cross-pairs).

Product-grade (vs effect-floor 0.515) requires CI95-lower ≥ 0.541 breakeven.

## 6. Compute budget
Hand-rolled CCM on `scipy.spatial.cKDTree`, `multiprocessing` over 20 cores. Surrogates only at Lmax/tp*/VAL/converging-drivers. Gate eval = 1 ρ per window per driver (~46k fits, ~4 min). Grid capped N=1800. **Total ~45–60 min, peak RAM <1.5GB.** Read only `mid`/`cks_e` columns per split. Optional pyEDM one-window cross-validation (|Δρ|<1e-3); optional numba if over budget.

## 7. Result file `min1_ccm_result.json`
Mirrors `min1_cksofi_result.json` convention + `drivers{}` (per-X: converges, conv_slope, rho_Lmin/max, tp_star, rho fwd/rev, surrogate p95, theta_gate, per-year gated moved-acc+CI), `sanity{self_coupling_strongest}`, `falsifier{FALSIFIER_KILLED, kill_reasons, thresholds, verdict}`, `compute{}`.

## 8. Implementation order
1. CCM core + self-coupling sanity gate (#6) FIRST — stop if EUR-own doesn't converge strongest.
2. Window iterator + 1s resampler (sum, empty=0).
3. E/τ selection on VAL, freeze.
4. Convergence + surrogates on VAL; θ by worst-VAL-half.
5. Gate eval VAL/TEST/OOS with frozen `min1_production` models; moved-bars/per-year.
6. Falsifier 1–6; write JSON.

Reused verbatim: `wc_ret`, `boot`, `nonoverlap_chrono`, `mk_lgb`, frozen `min1_EURUSD_*` artifacts.
