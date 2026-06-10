> **SCOPE: GBPUSD · 15m** — per-key EXECUTABLE backlog (FIRST-TO-RUN queue + incumbents-to-beat + discovery rounds). Ledger/status: `sweeps/GBPUSD_15m.md`. Results of record: `GBPUSD_RESULTS.md`. Generic idea pool: `IDEAS_LOG.md` / `SWEEP_MATRIX.md`.

# GBPUSD 15m — executable backlog

## INCUMBENTS TO BEAT (cross-key references; the GBPUSD bar is set by its own best once certified)
- Sibling certified 15m floors (refit-CPCV p10 @cov2): EURUSD pooled UP .567/DOWN .574 · USDJPY NY own-pair UP .586/DOWN .572 · AUDUSD NY seed-ens K=3 UP .596/DOWN .596 (mean ~.62).
- GBPUSD incumbent: none yet. First certified book becomes the incumbent; after that, INCUMBENT = best COMBO in `books/INDEX.json`.

## FIRST-TO-RUN QUEUE (strict order; one heavy job at a time)
1. ~~A1 base all-session~~ ✅ DONE (SURVIVED via 2024; 2026 frozen-dead → refit arbiter).
2. ~~A9 refit-CPCV `all`~~ ✅ DONE — **BOTH SIDES CERTIFIED @cov2+cov3** (UP .5532/.5531, DOWN .5527/.5483). The bar all lift levers must now beat: cov2 UP p10 .5532 / DOWN .5527.
3. **A9 refit-CPCV `ny`** (RUNNING), then `ldn`, `asia` — does concentration LIFT the certified floor (AUD NY lifted +.03-.05) or does GBP's LDN info make restriction lose power?
4. **A6 xpair pooled/xpbase screen** (fork audusd_15m_xpair.py: USD-residual + lead-lag + **EURGBP RV** + risk factor) — EUR-bloc pooling discriminator. If VAL-AUC ≫ base → pooled refit-CPCV.
5. **N2 EURGBP triangular USD-canceling residual** (gbpusd_15m_trigresid.py, new) — resid-sign/velocity features; uniquely on-disk here (needs EURUSD+GBPUSD only). Falsifier: famonly ΔVAL-AUC ≤ +.002 → subsumed-by-base.
6. **I2 seed-ens K=3** on the best of {best session, pooled} → the deliverable candidate.
7. **Adv**: frozen-past forward (trap#9) + tick-settlement validation on the deliverable.
8. **I2b K=8 saturation**, A-tb TB-label (matched-cov falsifier), Ortho famonly (EURGBP-diff/risk/RS±), Meta-gate, Xhor 30m-parent corr-screen — fast-KILL falsifiers (low priors from 2-sibling nulls).
9. Tier B–F coverage rows: run-or-cite per family (most have generic Tier-1 kills at EURUSD; cite into ledger with [PAIR·tf] evidence or run the cheap version).

## PRE-REGISTERED FALSIFIER TEMPLATES (write into result JSON BEFORE OOS read)
- Session CPCV: side CERTIFIED iff p10 ≥ .541 AND frac_clear ≥ .80; KILL session if p10 < .541 every cov.
- xpair/pooling: KILL if VAL-AUC ≤ base+.002 AND no binding-year improvement @cov2 (the AUD/JPY dilution signature). ESCALATE to pooled CPCV only if screen passes.
- trigresid: KILL if famonly ΔVAL-AUC ≤ +.002 AND resid feats absent from top-20 FI (the orthochan precedent).
- improve levers: beat incumbent p10 at the operating cov on BOTH sides or matched-cov sign-consistent lift; mean-only lift = variance redistribution → KILL (USDJPY TB lesson).

## DISCOVERY ROUNDS (loop till K=2 dry)
- R1 (pending): corpus mine `/home/sean/git/academic-papers/` for GBP/cable-specific + EUR-bloc levers (use `_CORPUS_INDEX.md` + `_extracted_levers.json`); GBP angles: BoE MPC/minutes timing, UK data calendar (7am LDN releases!), gilt-FX spillover, EURGBP cross-dynamics, Brexit-era regime lessons (2016/2019 in TRAIN years).
- R2 (pending): repo-doc mine (EURUSD 5m/10m/15m/30m backlogs + USDJPY/AUDUSD exhausts) for levers never retargeted to GBPUSD.
- R3 (pending): fresh arXiv/SSRN sweep, mechanism-first, sign-carrying only.

## NOTES / LEARNINGS (newest first)
- 2026-06-09 A1: 2024 base is the strongest sibling-family year (.649 cov2, .720 cov1 UP) and SYMMETRIC; 2026 frozen dead both sides. best_iter=33 → fast saturation, capacity not the constraint. No side-asymmetry signal yet.
