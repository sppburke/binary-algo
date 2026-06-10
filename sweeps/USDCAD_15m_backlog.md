SCOPE: USDCAD · 15m — EXECUTABLE backlog (TOP-N queue, incumbents-to-beat, discovery rounds). Ledger/status: `USDCAD_15m.md`. Results: `USDCAD_RESULTS.md`. Generic ideas: `IDEAS_LOG.md`.

# USDCAD 15m direction — executable backlog

**Incumbents to beat (Tier-1):** none yet (new key, sweep opened 2026-06-10). First incumbent = A1 base VAL moved-AUC + cov2% COMB per-year (pending). Cross-key reference floors at 15m: USDJPY NY seed-ens p10 .60/.57, AUDUSD NY seed-ens p10 .596/.596, GBPUSD xpair-NY K=8 p10 .655/.640, EURUSD xpair p10 .567/.574. Any lever must beat the binding-year figure and/or raise the refit-CPCV path-clear rate. CERTIFY only via per-fold-refit CPCV (p10≥.541, ≥80% of 15 paths). Adversarially verify every positive vs a frozen-past forward holdout (trap#9).

## FIRST-TO-RUN queue (on-disk, ranked by prior) — discovery round 1 (2026-06-10)
| # | lever | side | mechanism (1-line) | data | fast-KILL | status |
|---|---|---|---|---|---|---|
| 1 | **A9 NY-session** concentration | BOTH | USDCAD = most-NA pair (BoC+Fed+US/CA+oil in LDN/NY); NY-concentration is the strongest prior of any pair | `usdcad_15m_cpcv_session.py` (forked) | KILL NY-prior if NY p10 < all-session AND no NY year CI-lo ≥ .541 @cov2-3% | pending |
| 2 | **A9-all all-session** refit-CPCV floor | BOTH | honest per-era floor; recovers frozen-forward decay | `usdcad_15m_cpcv_session.py all` | KILL if no side p10 ≥ .541 at any cov | pending |
| 3 | **Seed-ensemble (K=3) + calib** | BOTH | variance-reduce the cov2-3% tail; the ONLY robust own-pair improve-lift (USDJPY/AUDUSD/GBPUSD) | `usdcad_15m_cpcv_session.py` nseed>1 | KILL unless seed-ens p10 > single-seed on SAME folds & ≥80% paths | pending |
| 4 | **A6 cross-pair USD-residual pooling** (keystone) | BOTH | USDCAD owes the signed USD-basket move + residual reverts; does it pool (EURUSD/GBPUSD) or dilute (USDJPY/AUDUSD)? **Sign-align: USD in numerator flips residual vs AUD/NZD** | `usdcad_15m_xpair.py` (to fork) | KILL if VAL AUC ≤ base OR no year CI-lo ≥ .541 OR fails to beat base cov2% on 2026 | pending |
| 5 | **Oil/WTI-bloc residual (on-disk proxy)** s = usdcad_r vs commodity-bloc(aud/nzd, sign-flipped) | BOTH | petrocurrency: CAD lag-completes the commodity/risk move; panel-internal oil proxy w/o off-disk WTI | panel (build_panel.py / closes) | KILL if famonly AUC < .515 OR within .005 of base | pending |
| 6 | **Per-side coverage operating-point** (if UP/DOWN asymmetry confirmed) | BOTH | the predicted UP>DOWN asymmetry may want different cov per side | base probs | no-op if both peak same cov (AUDUSD case) | pending |

## IMPROVE levers (apply to the best certified edge; lower prior or known-pair-specific)
| # | lever | side | note / Tier-1 status | prior |
|---|---|---|---|---|
| 7 | Seed-ensemble K=8 (saturation check) | BOTH | K=3→K=8: GBPUSD lifted (xpair 340-feat richer), USDJPY/AUDUSD saturated at K=3. Own-pair 239-feat → expect saturation; confirm | med |
| 8 | Triple-barrier first-touch TRAIN-label (eval unchanged) | BOTH | USDJPY+AUDUSD: certified-at-level but adversarially NOT robust (p10 = single noisy order-stat). Fast confirm-and-kill | low-med |
| 9 | Meta-labeling 'avoid-losers' gate on orthogonal axes | BOTH | NULL at EURUSD (.502) / AUDUSD (.52). Fast pre-check (meta-AUC viability ≤ .53 KILL) | low |
| 10 | Cross-horizon STACK (30m parent → 15m child) | BOTH | AUDUSD: parent decorrelated .72 but equally weak → redundant w/ seed-ens (KILLED). USDJPY: collinear .957 (subsumed). Run corr-gate first | low-med |
| 11 | GMADL / \|ret\|-weighted operating-point SELECTION (not retrain) | BOTH | \|ret\|-RETRAIN KILLED EURUSD/USDJPY/AUDUSD (2026 UP-collapse). SELECTION-only variant untried | low |
| 12 | Asymmetric per-side conformal (AgACI) gate | BOTH | plain ACI KILLED EURUSD/USDJPY/AUDUSD; per-SIDE split is the new bit — run only if UP-live/DOWN-dead (or vice-versa) asymmetry confirmed | low |
| 13 | Realized semivariance asymmetry RS+/RS− | side | SUBSUMED at EURUSD/AUDUSD (famonly ΔAUC ≈ 0, sign-invariance: vol→magnitude); confirm once | low |

## TICK / EXTERNAL (deferred — run only if bar levers survive or stall)
| # | lever | note |
|---|---|---|
| X1 | **Intraday WTI/Brent oil + BoC-Fed 2y rate-differential + risk-on/off (VIX/ES)** | EXTERNAL, off-disk. **THE petrocurrency frontier** — oil is the dominant signed CAD driver; the only lever with real probability of beating the ~.536 AUC info-bound. Source if on-disk panel-internal commodity/risk proxies (rows 5) prove insufficient. |
| T1 | (no USDCAD tick data on disk) | tick-settlement validation N/A; bar-close proxy validated on USDJPY/AUDUSD (mean tick−bar −.0035, PRESERVED) → trust at 15m |

## Discovery rounds
- **R1 (2026-06-10):** queue above instantiated from the AUDUSD/USDJPY/GBPUSD ledgers + SWEEP_MATRIX, deduplicated by Tier-1 dispositions (don't re-run what's subsumed across all 3 prior pairs). Key USDCAD-SPECIFIC mechanisms to test (not in prior ledgers): oil/WTI-bloc residual (#5, petrocurrency), the UP-side-more-forecastable asymmetry prediction, most-NA NY-concentration. Corpus deep-mine pending (workflow) — focus: petrocurrency / oil lead-lag / BoC-Fed differential / commodity-FX direction papers in academic-papers/.
