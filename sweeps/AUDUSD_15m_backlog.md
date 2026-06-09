SCOPE: AUDUSD · 15m — EXECUTABLE backlog (TOP-N queue, incumbents-to-beat, discovery rounds). Ledger/status: `AUDUSD_15m.md`. Results: `AUDUSD_RESULTS.md`. Generic ideas: `IDEAS_LOG.md`.

# AUDUSD 15m direction — executable backlog

**Incumbents to beat (Tier-1):** A1 base all-session VAL moved-AUC **.5232**; cov2% COMB .604/.587/.548 (2024/25/26); binding-2026 is **DOWN-only** (DOWN .589@cov2, UP .493 collapsed). Any lever must beat the binding-2026 figure and/or raise the refit-CPCV path-clear rate. CERTIFY only via per-fold-refit CPCV (p10≥.541, ≥80% of 15 paths). Adversarially verify every positive vs a frozen-past forward holdout (trap#9) — the 2026 UP collapse already flags UP as regime-dependent.

## FIRST-TO-RUN queue (on-disk, high prior, ranked) — discovery round 1 (2026-06-09)
| # | lever | side | mechanism (1-line) | data | fast-KILL | status |
|---|---|---|---|---|---|---|
| 1 | **A6 cross-pair USD-residual pooling** (keystone) | BOTH | AUDUSD owes the signed USD-basket move (catchup) + residual reverts; EURUSD-certifying lever — does AUDUSD pool (EURUSD) or dilute (USDJPY)? | `audusd_15m_xpair.py` (coded) | KILL if VAL AUC ≤ .5232 OR no year CI-lo ≥ .541 OR fails to beat base cov2% on 2026 | CODED, staged |
| 2 | **A9 Asia-vs-NY session** conditioning | BOTH | AUD idiosyncratic info (RBA/China/AU data) arrives in ASIA — unlike EUR/JPY (NY). Symmetric test asia/ny/ldn/all | `audusd_15m_cpcv_session.py` (wired) | KILL Asia hypothesis if Asia AUC ≤ all-session .5232 AND no Asia year CI-lo ≥ .541 @cov2-3% | RUNNING (all) |
| 3 | **AUDNZD USD-canceling residual-difference** s=e_AUD−e_NZD = aud_r−nzd_r | BOTH | AUD vs NZD twin; USD factor cancels; mean-reverts around RBA-RBNZ anchor. Pooling never DIFFERENCES the legs | panel/closes (in `audusd_15m_xpair.py` as audnzd_r/dev) | KILL if famonly AUC < .515 OR within .005 of NZD-leg-shuffle OR sign-shuffle-within-\|s\|-bins preserves it | partly in A6; isolate marginal |
| 4 | **Carry-unwind DOWN-specialist, lagged risk-bloc gate** RISK=commod(AUD,NZD,CAD)−haven(CHF,JPY) | DOWN | carry unwinds asymmetrically (violent down in stress); gate DOWN bets on lagged RISK<0 & haven-rv high. Attacks binding-2026 DOWN-only | panel (risk feats in A6) | KILL unless gated DOWN beats ungated DOWN (.589@cov2 2026) by >.01 in ≥2 yrs; use SYMMETRIC model+GATE not subset-refit | pending |
| 5 | **Seed-ensemble (K=3→8) + temperature calib** | BOTH | variance-reduce the cov2-3% tail where the edge + 2026 UP collapse live; the ONLY robust USDJPY 15m improve-lift | `audusd_15m_cpcv_session.py` nseed>1 | KILL unless seed-ens p10 > single-seed on SAME folds & ≥80% paths clear | pending (on best edge) |
| 6 | China/commodity-residual consensus (commod-bloc mean + AUD-vs-bloc gap sign) | BOTH | common idiosyncratic comp of e_AUD/e_NZD/e_CAD = panel-internal China/commodity proxy; AUD lag-completes | panel | KILL if famonly AUC<.515 OR within .005 of CAD/NZD-rotation surrogate | pending (after A6) |
| 7 | Meta-labeling on orthogonal axes (xpair agree, AUDNZD sign, risk sign, session) | BOTH | gate symmetric primary on P(call correct); EURUSD-proven gate, untested AUDUSD | base probs + panel; m15_meta.py | KILL if meta-gated 2026 WR ≤ raw primary at ≥ n; corr(meta,primary)>.9 redundant | pending (improve) |

## IMPROVE levers (apply to the best certified edge; lower prior or known-pair-specific)
| # | lever | side | note / Tier-1 status | prior |
|---|---|---|---|---|
| 8 | Triple-barrier first-touch TRAIN-label (eval unchanged) | BOTH | USDJPY: certified-at-level but adversarially NOT a robust improvement (p10 = single noisy order-stat). Fast confirm-and-kill | med |
| 9 | GMADL / \|ret\|-weighted operating-point SELECTION (not retrain) | BOTH | \|ret\|-RETRAIN KILLED both pairs (2026-UP-collapse). SELECTION-only variant untried (METHODS_CATALOG 9.3) | low |
| 10 | Asymmetric per-side conformal (AgACI) gate | BOTH | plain ACI KILLED both pairs; per-SIDE split is the new bit — run only if DOWN-live/UP-dead asymmetry confirmed | low |
| 11 | Realized semivariance asymmetry RS+/RS− (signed feature) | DOWN | EURUSD D7 PASSED its sign-flip null but REAL-but-SUB-BE (.5258<.541). Must clear .541 & add to book | low |
| 12 | Cross-pair learning-to-rank (rank:pairwise over 7 majors) | BOTH | SUBSUMED both pairs (rank-loss → relative not own-sign; reduces to BCE). Must BEAT plain A6 or fully subsumed | low |
| 13 | Time-of-week / RBA-cycle calendar conditioning in Asia | unclear | USDJPY gotobi/Tokyo-fix N5 KILLED (+.0004); calendar = SIZE not SIGN elsewhere. Run once inside Asia seg | low |

## TICK / EXTERNAL (deferred — run only if bar levers survive or stall)
| # | lever | note |
|---|---|---|
| T1 | Tick-resolution AUDNZD RV + Asia signed-OFI (forward-holdout-gated, SIGNED-only) | AUDUSD+NZDUSD ticks on disk. STRONG caution: USDJPY 15m tick-micro KILLED forward (trap#9, regime-feature memorization). Use SIGNED imbalance ONLY, drop spread/intensity. Run after bar AUDNZD lever survives |
| X1 | Intraday AU-US rate-differential + risk-on/off (equity/VIX) + commodity (iron-ore/CRB) | EXTERNAL, off-disk. DL-review's dominant external direction carrier. Source only if on-disk panel-internal risk/commodity proxies (rows 4,6) prove insufficient |

## Discovery rounds
- **R1 (2026-06-09):** workflow `audusd-15m-discovery` — 12 corpus + 10 AUD-specific levers, 25-lever subsumption map. Above is the deduplicated ranked queue. Key NEW AUD-specific mechanisms (not in EURUSD/USDJPY ledgers): AUDNZD residual-difference (#3), carry-unwind risk-bloc DOWN gate (#4), commodity-residual consensus (#6), Asia-session hypothesis (#2). NOT yet dry — these are untested.
