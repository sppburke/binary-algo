# EURUSD · 2m — EXECUTABLE BACKLOG (key-specific)
SCOPE: EURUSD · 2m. Key-specific executable backlog — incumbents, scripts, numbers, discovery rounds.
Moved out of docs/IDEAS_LOG.md on 2026-06-02 during the generic↔specific split (see REPO_MAP.md).
Generic methods/ideas: docs/METHODS_CATALOG.md / SWEEP_MATRIX.md / docs/IDEAS_LOG.md. Sweep ledger (status): sweeps/EURUSD_2m.md. Results of record: results/EURUSD_RESULTS.md.

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
NOTE (2026-06-04): rounds 1-2 "PENDING" items N3-N9 + Stoikov were all subsequently RUN and KILLED (see sweeps/EURUSD_2m.md rows N3/5/6/9 legsign, N4 mim, N7 hawkes, N8 rsskew, D2 stoikov).

## Discovery round 3 — corpus re-mine through the cross-pair-decorrelation lens (2026-06-04)
Re-opened after the 15m certification taught the cross-pair feature-row-POOLING mechanism (distinct from cross-leg
sign-lead). Filtered `_extracted_levers.json` (2050) + `docs/CORPUS_LEVER_INVENTORY.md` (550) programmatically →
946 high/med-prior, on-disk, sign-aware DIRECTION candidates. After dedup + sign-invariance/subsumption vetting, the
genuinely-NEW on-disk direction ideas ALL reduce to the cross-pair/cross-sectional family (matching the 15m round
conclusion). Disposition:
- **Cross-pair feature-row POOLING** (the keystone; certified 5m UP & 15m both) — **RUN + KILLED @2m** under ties-strict
  refit-CPCV (A6c: UP p10 .5096 / DOWN .5124, 0-7% paths clear). The probe's frozen-VAL .61 dissolved under per-fold refit.
- **RFF virtue-of-complexity SDF** (Didisheim-Ke-Kelly-Malamud; 4th model class on the SAME xpof features) — **RUN + KILLED**
  (min2_rff: best VAL AUC .4997, a coin-flip across gamma×ridge). Confirms the channel is empty, not a GBM-capacity artifact.
- TMFG / homological-CNN graph feature filtering (Briola) — SUBSUMED: a representation of a feature set whose 4-model-class
  AUC is .50-.52 cannot manufacture sign signal.
- Adversarial hard-regime miner gate (Chen-Pelger-Zhu) — SUBSUMED: the refit-CPCV already showed any per-fold-retuned gate
  on this base collapses (the overfit-gate the probe exhibited).
- Cross-pair learning-to-rank / lambdarank (Feng) — SUBSUMED: lambdarank already KILLED @5m (m5_rankloss); does not beat BCE.
- Campbell-Thompson sign restrictions — SUBSUMED: cannot restrict a .50-AUC non-signal into an edge.
**VERDICT: on-disk 2m direction discovery is DRY.** This is the 3rd consecutive effectively-dry round (rounds 1-2 also dry
after run) → K=2 dry achieved. The only NEW-information redirect is external data (Tier-G). Nulls are redirects, not walls.

## TIER-G — EXTERNAL-DATA FRONTIER for 2m (data-acquisition prerequisite; GATED ON USER "go")
The genuine binding constraint at 2m is DATA, not model (4 model classes + ~16 channels all flat). Same external drivers
as the 15m frontier; at 2m these act mainly as risk-off STATE gates (a 120s direction tilt during regime shifts), lower
prior than at 15m but the only untapped information:
- **G1 — DE-US 2y/10y rate differential (intraday, Dukascopy/macro):** carry/fundamental driver; two-stream gate.
- **G2 — VIX / implied-vol / EUR risk-reversal (risk-off STATE gate):** risk-off → USD-bid → EURUSD down (DOWN-side). VIX free (CBOE); risk-reversal paywalled.
- **G3 — Equity-vol spillover (GARCH-MIDAS on S&P realized vol):** low-freq vol regime gate.
- **G4 — Cross-asset daily leads (S&P / oil / gold / 5y):** risk-on/off lead.
**Falsifier (any G-row):** KILL unless the external signal lifts a side's binding-year refit-CPCV p10 above breakeven 0.541.
