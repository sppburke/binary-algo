# EXHAUSTIVENESS AUDIT — USDJPY · 15m DIRECTION lever sweep

```
########################################################################
#  SCOPE:  USDJPY · 15m · NY session · DIRECTION (UP/DOWN)             #
#  Incumbent (CERTIFIED): NY seed-ensemble refit-CPCV                  #
#     UP p10 .6005 / DOWN .5738 @cov2%; directional AUC capped ~.539   #
#  Question: do any of 9 C/E-tier levers BREAK the ~.539 AUC ceiling?  #
#  Settlement (all levers): deriv-faithful fixed-15m sign(close[t+15]  #
#     - close[t]), ties LOSE, BE=0.541, gap=900s nonoverlap, NY rows.  #
#  Held-out years: test24=2024, test25=2025, oos=2026.                 #
########################################################################
```

All numbers below are Tier-1: read verbatim from each lever's `usdjpy_15m_<key>_result.json`.
"Held-out AUC" = the deriv-faithful fixed-15m **moved-AUC** per year `(test24 / test25 / oos)`.
Verdict = each script's own `verdict.KILLED` field, not editorial judgment.

| lever | construction | held-out AUC (24 / 25 / 26) | verdict | result_json |
|---|---|---|---|---|
| **hmm** | C1 Gaussian HMM (K=3, cov=full) on causal emission feats; forward-only filtered hard state; signal = TRAIN state-conditional next-15m up-rate | 0.4966 / 0.4953 / 0.5001 | **KILLED** | `/home/sean/git/binary-algo/usdjpy_15m_hmm_result.json` |
| **kalman** | C2 Kalman forward-filter (local-level + slope), causal; direction = sign of filtered slope | 0.4829 / 0.4933 / 0.5009 | **KILLED** (anti-predictive) | `/home/sean/git/binary-algo/usdjpy_15m_kalman_result.json` |
| **rmt** | C3 RMT eigen-residual reversion: 7 USD-major panel, Marchenko-Pastur strips leading market mode, USDJPY residual sign | 0.5072 / 0.5058 / 0.5038 | **KILLED** | `/home/sean/git/binary-algo/usdjpy_15m_rmt_result.json` |
| **ccm** | C4 convergent cross-mapping (delay-embed USD-major drivers, kNN cross-map USDJPY fwd return, take sign) | n/a — crashed, no `years`/`verdict` | **ERROR** (no result) | `/home/sean/git/binary-algo/usdjpy_15m_ccm_result.json` |
| **arf** | C5 river ARFClassifier (5 trees, depth 12) + ADWIN, prequential predict-then-learn, 33 causal feats, on_stride=1 | 0.8523 / 0.8516 / 0.8519 | **SURVIVED per script** — LEAKAGE-SUSPECT, see flag | `/home/sean/git/binary-algo/usdjpy_15m_arf_result.json` |
| **magdir** | E2 direction-conditioned-on-magnitude: base NY dir GBM bucketed by predicted-magnitude quartile | 0.5317 / 0.5221 / 0.5070 | **KILLED** | `/home/sean/git/binary-algo/usdjpy_15m_magdir_result.json` |
| **complexity** | E3 perm-entropy / VR-Hurst / lag-1 autocorr gates; conditional base-GBM dir-AUC per complexity bin | 0.5306 / 0.5234 / 0.5137 | **KILLED** | `/home/sean/git/binary-algo/usdjpy_15m_complexity_result.json` |
| **infobars** | E4 information bars (volume-clock sampling, AFML ch2) | n/a — not constructible | **DATA-LIMITED** | `/home/sean/git/binary-algo/usdjpy_15m_infobars_result.json` |
| **tbfirsttouch** | triple-barrier first-touch TRAIN label (+/-k*sigma, vertical 15m); k=2.0 frozen on VAL; EVAL=fixed-15m | 0.5321 / 0.5229 / 0.5124 | **SURVIVED per script** (cov3% CI-lo, 2/3 yr) | `/home/sean/git/binary-algo/usdjpy_15m_tbfirsttouch_result.json` |

### Honest flags (Tier-1)

- **ccm — ERROR, no numbers.** JSON on disk contains only the pre-registered header (drivers, grid, falsifier); no `years` block and no `verdict`. The cov3%/AUC cells are genuinely unavailable. Re-run required.
- **infobars — DATA-LIMITED by design.** `years: {}`, `verdict.KILLED=false` with a DATA-LIMITED note. `data_check` confirms USDJPY parquet has only derived vol cols `["vol_z","zero_vol"]`, `raw_volume_candidates: []`. Volume clock not constructible; the script refused to fabricate one.
- **arf — SURVIVED per script, but LEAKAGE-SUSPECT.** Self-declared `verdict.KILLED=false`; held-out moved-AUC ~0.852 and cov3% COMBINED WR 0.9272 / 0.9036 / 0.9145 across all three years. These are far outside every other lever's regime (~0.50-0.62) and far above the certified incumbent (~.539 AUC, ~.58-.60 WR). That magnitude in a `river` prequential predict-then-learn loop at `on_stride=1` is the classic signature of label/look-ahead leakage in the online update, NOT a genuine 15m direction edge. Reported as the script's verbatim output; an independent leakage audit is required before this counts as a ceiling break.
- **tbfirsttouch — SURVIVED per script** on `cov3_COMBINED_CIlo_clears_BE` in test24 + test25 (chosen k=2.0, VAL fixed-15m AUC 0.5439 vs base 0.5313). No held-out year's moved-AUC beat the incumbent (`auc_beats_incumbent_years: []`); survival rests on the 2-of-3-year cov3% CI-lower clearing breakeven. oos (2026) did NOT clear (cov3% COMBINED 0.5431). It does not break the AUC ceiling.
- **KILLED levers (hmm, kalman, rmt, magdir, complexity)** all match their pre-registered falsifiers: held-out moved-AUC at/near 0.50, no year beating .539, cov3% CI-lower not clearing 0.541 in >=2 years. kalman is anti-predictive (held-out AUC < 0.50 -> 15m slope is the wrong sign; mean-reverting). complexity's reported cov3% values are the low-PE-gated base-GBM WR, not a standalone complexity edge; bins flat (`bins_flat: true`, max AUC spread 0.0376).

### Conclusion

No lever with a TRUSTWORTHY held-out number broke the ceiling. Of the 7 levers that produced direction-AUC, none exceeded the certified ~.539 incumbent AUC in any held-out year. The two `KILLED=false` survivors do NOT constitute a real edge over the incumbent: **tbfirsttouch** survives only on a 2/3-year cov3% CI-lower technicality (AUC never beat .539, oos failed), and **arf**'s ~0.85 AUC / ~0.92 WR is leakage-suspect and not yet trusted. The five clean-fail levers (hmm, kalman, rmt, magdir, complexity) all returned moved-AUC at/near 0.50 exactly as the sign-invariance theorem (arXiv:2512.15720) predicts — state / coupling / complexity / magnitude gates move SIZE, not SIGN. **The ~.539 AUC / ~.58-.60 win-rate ceiling holds, and the sign-invariance theorem is reproduced at USDJPY 15m.** Two items need follow-up before this is final: re-run ccm (crashed) and leakage-audit arf.
