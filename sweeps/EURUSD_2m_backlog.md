# EURUSD · 2m — EXECUTABLE BACKLOG (key-specific)
SCOPE: EURUSD · 2m. Key-specific executable backlog — incumbents, scripts, numbers, discovery rounds.
Moved out of IDEAS_LOG.md on 2026-06-02 during the generic↔specific split (see REPO_MAP.md).
Generic methods/ideas: METHODS_CATALOG.md / SWEEP_MATRIX.md / IDEAS_LOG.md. Sweep ledger (status): sweeps/EURUSD_2m.md. Results of record: EURUSD_RESULTS.md.

---

## Discovery round — novel 2m EURUSD DIRECTION ideas (2026-06-01, sub-agent research, for sweeps/EURUSD_2m)
Generated for the EURUSD 2m sweep (SWEEP_MATRIX Tier-N). Each vetted: genuinely new vs catalog, plausible
SIGN mechanism (survives sign-invariance arXiv:2512.15720), on-disk data, fast-KILL falsifier.
- **N2 Triangular USD-canceling residual** (EURUSD vs GBPUSD rolling cointegration; USD factor algebraically removed → immune to the 2025 USD-factor inversion). arXiv:0812.0913. Prior 15%. **TESTED → KILLED** (min2_triangular_result.json: COMB .499/.497/.497, coin-flip; relative-value reverts too slowly for 120s).
- N3 Cross-quantilogram tail-lead (Han-Linton-Oka-Whang arXiv:1402.1937) — tail-conditional sign asymmetry. Prior 12%. PENDING.
- N4 Market-Intraday-Momentum term-structure (Gao-Han-Li-Zhou JFE2018) — lagged clock-interval return → fwd sign. Prior 10%. PENDING.
- N5 PCMCI+ causal-discovery sign-stable lead (Runge SciAdv2019) — conditions out the common USD factor. Prior 8%. PENDING.
- N6 Directed-information on SIGN sequences (Massey; Quinn-Coleman-Kiyavash) — magnitude-blind go/no-go gate; pre-kills N3/N5/N9. Prior 6%. PENDING (run as gate).
- N7 Asymmetric up/down-tick Hawkes intensity imbalance (Bacry-Muzy arXiv:1301.1135) — cross-excitation asymmetry φuu−φdd, mid-tick only. Prior 9%. PENDING.
- N8 Signed-semivariance-skew sign-conditioning on the magnitude model (Patton-Sheppard) — RS⁺−RS⁻ skew is signed. Prior 5%. PENDING.
- N9 Cross-pair signed ordinal transition imbalance (Bandt-Pompe / Neuman-Cohen-Tamir) — cross-series, signed. Prior 5%. PENDING.

## Discovery round 2 — workflow fan-out (5 lenses), 2026-06-01. Novel 2m ideas beyond N2-N9 (priors honestly 4-9%):
- Stoikov microprice-velocity vs mid-velocity slope-divergence, gated reversion sign (within-EURUSD, USD-inversion-immune; signed fair-value lead, not a level). Prior 8%, on-disk. [TOP — test_spec in workflow output]
- Directed horizontal-visibility-graph peak-vs-trough irreversibility (signed by tail decomposition; sign-covariant, unlike the killed scalar I_W). Prior 7%, on-disk.
- PID synergy/redundancy across legs (higher-order info beyond pairwise directed-info, which the cross-leg gate killed). Prior 6%.
- CMI: condition next-2m sign on the surviving relative-value RESIDUAL STATE (not the slow residual itself). Prior 6%.
- Event-localized (not time-averaged) information bursts at CKS-OFI events. Prior 5%.
- WMR-fixing reversal at 2m, FLOW-conditioned (m30_fix found WMR sign-flipped to continuation in 2026 @30m; never tested @2m). Prior 5-6%. Coverage-starved (2026 = 5 month-turns).
- Turn-of-month signed drift book (only ever a 15m coverage gate, never standalone @2m). Prior 5%. Coverage-starved.
- ML: contrastive/SSL pretrain → linear sign-probe; conformal-abstain; focal/ordinal loss on the UP asymmetry. Prior 4-9% (program shows ceiling is DATA not model).
ASSESSMENT: all low-prior; the 13-channel sweep + online-ARF keystone + the UP-uncertified CPCV make these confirmatory. The Stoikov slope-divergence (within-EURUSD, USD-immune, signed) is the single most-worth-testing; tested as a completeness check.
