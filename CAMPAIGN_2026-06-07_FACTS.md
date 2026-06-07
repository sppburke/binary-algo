# NOVEL-METHODS CAMPAIGN — VERIFIED FACTS (2026-06-07)

Single source of truth for the "execute ALL of NOVEL_METHODS_RESEARCH §7" campaign. Every number here is from a committed
result JSON or captured run output (Tier 1). Documentation agents: use ONLY these facts; do not invent numbers. All gated
by the deployment-faithful FROZEN-PAST FORWARD HOLDOUT (train≤2023 → per-year 2024/25/26), NOT pooled CPCV (leakage trap #9).

## NEW REUSABLE INFRA (Phase 1) — committed 051e525

- **build_panel.py** → `features/panel_<year>.parquet` (gitignored data). The clean 7-pair USD return panel: 5,345,437 bars
  2012-2026; per-row cols `t`, `r_<PAIR>` (eu-equiv 1-min log-returns, 7), `fac` (USD common factor = mean of 7), `e_<PAIR>`
  (residual = r−fac), `c_eur` (raw EURUSD close). Substrate for ALL path-based cross-sectional methods.
- **panel_faithcheck.py** + `panel_faithcheck_result.json` — DECISIVE faithfulness A/B vs certified `build_xp` (year 2024):
  all continuous channels (`eu_r/usdbask/catchup/eurresid/disp/ll_*`) BIT-IDENTICAL (max_abs 0.0); only `agree*` differs at
  1 row/lookback (0.0003%, warmup off-by-one). Pooled-CPCV reproduction (`build_panel_validate_15m.json`) read selacc p10
  **.5355** / mean .5425 vs certified m15xp .567/.574 — this is the DOCUMENTED weaker-reimplementation effect (single
  600-tree pooled LGBM vs certified refit ~.54), NOT leakage (which would INFLATE). Up-rate .5074 inside [.47,.53] tripwire.
  VERDICT: panel CONSTRUCTION-FAITHFUL, safe for all downstream.
- **fwd_holdout.py** — reusable frozen-past forward-holdout gate (magnitude AUC+lift / direction cov-selacc modes). The
  MANDATORY deployment gate (NOVEL §0 #1). Self-checked: DEPLOYS a stationary signal (+0.10/yr), REJECTS a trap-#9
  non-stationary feature (decays to −0.015) — the exact deseason-+tod failure mode it exists to catch.
- **surrogate_null.py** — phase-randomize / IAAFT surrogate-null gate (NOVEL §0 #2). Self-checked: linear lag-1 autocorr →
  NOT significant (real .6959 ≈ null_p95 .6961, spectrum preserved); nonlinear |.| vol-clustering → significant
  (real .0883 ≫ null_p95 .0199). Separates spectral re-encoding from genuine nonlinear structure.

## PHASE 2 — MAGNITUDE HAR / realized-vol canon — committed 7d45bf8 (recorded MAGNITUDE_FINDINGS §6g)

`mag_har.py` + `mag_har_result.json`. Arms added to certified base `[-pe,rv30,rv120]`, forward holdout, horizons 10/15/30m,
target |ret_H|≥train-Q75. Falsifier = +0.005 AUC in ≥2 forward years AND ≥2 horizons.
**ALL ARMS FAIL (REAL-but-SUB-BAR):**
- +har (multiscale RV): mean fwd ΔAUC +0.0010, deployable 2/3 — sub-bar
- +jump (bipower split): +0.0003, 1/3 — null
- +semivar (RS⁺/RS⁻/signed-jump): −0.0000, 1/3 — null
- +harq (realized quarticity): +0.0004, 2/3 — null
- **+all (stacked): +0.0021, 3/3 deployable (no decay) — forward-CONSISTENT but economically negligible**

Two findings: (1) UNLIKE §6f time-of-day, additions are forward-CONSISTENT (+all positive in all 9 year×horizon cells) —
genuine but tiny; base rv already extracts ~all magnitude (collinearity vs rv120: lRV120 .89, RS .70, HARQ .58/−.63; only
signed-jump SJ120 orthogonal −.04 and carries nothing). (2) RE-VALIDATES the certified magnitude edge on a clean
deployment-faithful holdout: base AUC .799/.750/.750 (10m) .791/.739/.737 (30m), 4–5.6× decile lift, NO decay.
VERDICT: KILLED as deployable upgrade; magnitude path EXHAUSTED on-disk.

## PHASE 3 / T2 — FRACTIONAL DIFFERENTIATION

- **frac_diff.py** — hand-rolled FFD (fixed-width weights + ADF d*-selection). Self-check: random walk needs d*=0.1 to pass
  ADF (p .019) while retaining 98.8% level-memory vs 0.9% for plain returns ("stationarity with memory"). Genuinely untried.
- **frac_direction.py** + `frac_direction_15m_result.json` — FFD USD factor/residual/lead-lag into certified direction book,
  forward holdout, NY cov0.10 selacc. Per-pair d*: EURUSD/NZD/CHF=0.1, GBP/AUD/JPY/CAD=0.2 (windows ~500 bars), thresh 1e-4.
  **15m KILLED:** base selacc 2024 .5900 / 2025 .5520 / 2026 .5254 (pooled .5642); +ffd .5792/.5301/.5403 (pooled .5537) →
  DECAYS (Δ 2024 −0.0108, 2025 −0.0219, 2026 +0.0149), deployable=false; ffdonly .5188/.4985/.5479 (pooled .5001 = coin
  flip) → no standalone signal. FFD level-memory adds nothing to direction and hurts recent years. Note: base book itself
  DECAYS over forward years (.59→.55→.525), consistent with documented refit-dependence of the cross-pair edge.
  **30m ALSO KILLED (harder):** base selacc 2024 .5842 / 2025 .5483 / 2026 .5007 (pooled .5558); +ffd .5546/.5135/.5064
  (pooled .5308) → DECAYS (Δ 2024 −0.0296, 2025 −0.0348, 2026 +0.0057), deployable=false; ffdonly pooled .4958 (below
  coin flip). FFD direction DEAD at BOTH 15m and 30m. `frac_direction_30m_result.json`.
- **T1 information-driven bars** — NOT yet built. Substrate scouted: `features_tick/{train,val,test,oos}_1s.parquet` has
  EURUSD 1s bars (cols mid/imb/micro/spread/nt/tsz) but EURUSD-ONLY and 2021+ → cross-pair synchronization needs proxies
  for the other 6 pairs (NOVEL T1 anticipated this). DEFERRED behind T2.

## PHASE 4 — CROSS-SECTIONAL DIRECTION (harness WRITTEN, not yet run)

- **xsec_direction.py** — unified harness on the certified panel. Two dependency-free families with mechanism-specificity
  shuffle controls, each via forward holdout (direction, NY cov0.10), arms {base, base+fam, famonly, fam_shuffle}:
  - **D1 sig** — hand-rolled depth-2 LEAD-LAG SIGNATURE: level-2 iterated integrals S^{ij}=∫∫dX^i dX^j + Lévy area
    A^{ij}=½(S^{ij}−S^{ji}) between EURUSD and each peer over a trailing 30-bar window (encodes signed lead-lag /
    quadratic covariation). iisignature can't compile (no Python.h) → computed in pure numpy via cumsum. Falsifier: beats
    book p10 CI95 AND lead/lag-shuffle MUST degrade it.
  - **D6 havok** — frozen-basis HAVOK (Hankel-Koopman): delay-embed (q=60) the USD-factor trend, SVD on TRAIN → freeze r=8
    modes, causally project to coords v1..v7 + intermittent FORCING v_r (signed precursor) + leading phase. Falsifier:
    signed feature beats .541, book-gating lifts p10, surrogate-null (phase-randomized control built in).
- REMAINING SLATE (not built): D2 untruncated signature kernel (needs sigkernel/KeOps GPU), D4 FASCL future-aligned
  contrastive (8GB GPU), D5 causal lead-lag (PCMCI/Granger-FDR/structural-VAR), D7 signed-semivariance direction,
  D8 quantile-direction baseline.

## PHASE 5 — GATES (not built): G1 persistent-homology corr-cloud (needs gudhi), G2 BOCPD/HMM (ruptures available),
  G3 windowed-DMD residual.

## ENV NOTES
- venv ~/binary-algo-venv is a uv venv with NO pip → use `VIRTUAL_ENV=... uv pip install`. iisignature/sigkernel/gudhi
  unbuildable (no Python.h dev headers); statsmodels + ruptures present.
- Background launches: `run_in_background:true` with the python command in the FOREGROUND (no `nohup &`, which gets reaped).
  NEVER `pkill -f <script>.py` from a shell whose own command line contains that name (self-kill, exit 144).

## DONE-vs-FUTURE SUMMARY
- DONE+committed: Phase 1 infra (panel+faithcheck+2 gates), Phase 2 magnitude HAR (killed/sub-bar), Phase 3 T2 FFD-direction
  15m AND 30m (both KILLED). All forward-holdout-gated.
- FUTURE: fold FFD-30m; run Phase 4 (D1 sig, D6 havok — harness ready); build T1 info-bars; remaining direction slate
  (D2/D4/D5/D7/D8); Phase 5 gates (G1/G2/G3); transforms T3 vol-time-subordination / T4 whitening / T5 Hilbert; magnitude
  exotica M4 TDA-Wasserstein / M5 multiscale-ECC / M6 MOMENT (all surrogate-null-gated).
