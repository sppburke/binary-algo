SCOPE: USDCHF · 15m — EXECUTABLE backlog (TOP-N queue / incumbents-to-beat / discovery rounds). Ledger: `USDCHF_15m.md`. Results: `USDCHF_RESULTS.md`. Generic menu: `SWEEP_MATRIX.md`.

# USDCHF 15m direction — executable backlog

## FIRST-TO-RUN queue (highest ROI first)
1. **A1 base** — `usdchf_15m_base.py` ✅ DONE (SURVIVED; VAL AUC .5301; 2026 binding cov2 COMB .5103 sub-BE = refit-dependent).
2. **A9 session landscape** — `usdchf_15m_cpcv_session.py {all,ny,ldn,asia}` 🔄 RUNNING (chain). Find carrier (NY vs LDN — CHF is European). Honest refit floor.
3. **A6 EUR-bloc POOLING (keystone)** — `usdchf_15m_xpair.py xpbase` — THE differentiating test (CHF≈−EURUSD → pooling may WIN unlike havens). Ready to launch when heavy slot frees. Falsifier: IMPROVES iff VAL AUC>.5301 AND 2026 cov2 COMB>.5103.
4. **I2 seed-ens K=3** on the carrier book — deliverable improve lever.
5. Improve cross-product (magdir, TB, xhor, OFI, kNN, meta, two-speed, spec, GRU, Optuna) — each vs BEST COMBO incumbent + frozen-forward adversarial check.

## Incumbents to beat
- **A1 base (all-session frozen):** VAL AUC .5301; per-year AUC .5288/.5205/.5129; cov2 COMB .6513/.5675/.5103 (2024/25/26). 2026 binding sub-BE.
- Carrier-session refit-CPCV floor: pending (A9).

## Discovery rounds
- **R1 (corpus + web, CHF-specific):** 🔄 RUNNING (workflow usdchf-15m-discovery-r1). Angles: SNB intervention regime, EUR-bloc lead-lag / triangular arb, safe-haven risk-off direction, carry-unwind, corpus mine, fresh web. Synthesis → ranked on-disk candidates + acquisition-frontier list appended here on completion.

_Discovery synthesis pending — table will be appended below when R1 completes._
