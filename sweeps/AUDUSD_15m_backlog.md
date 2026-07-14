SCOPE: AUDUSD · 15m — EXECUTABLE backlog (TOP-N queue, incumbents-to-beat, discovery rounds). Ledger/status: `AUDUSD_15m.md`. Results: `results/AUDUSD_RESULTS.md`. Generic ideas: `docs/IDEAS_LOG.md`.

# AUDUSD 15m direction — executable backlog

**Incumbents to beat (Tier-1):** A1 base all-session VAL moved-AUC **.5232**; cov2% COMB .604/.587/.548 (2024/25/26); binding-2026 is **DOWN-only** (DOWN .589@cov2, UP .493 collapsed). Any lever must beat the binding-2026 figure and/or raise the refit-CPCV path-clear rate. CERTIFY only via per-fold-refit CPCV (p10≥.541, ≥80% of 15 paths). Adversarially verify every positive vs a frozen-past forward holdout (trap#9) — the 2026 UP collapse already flags UP as regime-dependent.

## FIRST-TO-RUN queue (on-disk, high prior, ranked) — discovery round 1 (2026-06-09)
| # | lever | side | mechanism (1-line) | data | fast-KILL | status |
|---|---|---|---|---|---|---|
| 1 | **A6 cross-pair USD-residual pooling** (keystone) | BOTH | AUDUSD owes the signed USD-basket move (catchup) + residual reverts; EURUSD-certifying lever — does AUDUSD pool (EURUSD) or dilute (USDJPY)? | `audusd_15m_xpair.py` (coded) | KILL if VAL AUC ≤ .5232 OR no year CI-lo ≥ .541 OR fails to beat base cov2% on 2026 | ❌ DONE — NULL (VAL .5244≈base; xpair feats not in top20; own-pair-specific = USDJPY case). `audusd_15m_xpair_xpbase_result.json` |
| 2 | **A9 Asia-vs-NY session** conditioning | BOTH | AUD idiosyncratic info (RBA/China/AU data) arrives in ASIA — unlike EUR/JPY (NY). Symmetric test asia/ny/ldn/all | `audusd_15m_cpcv_session.py` (wired) | KILL Asia hypothesis if Asia AUC ≤ all-session .5232 AND no Asia year CI-lo ≥ .541 @cov2-3% | ✅ DONE — NY carries both sides; Asia+LDN sub-BE (hypothesis REFUTED) |
| 3 | ~~AUDNZD USD-canceling residual-difference~~ s=aud_r−nzd_r | BOTH | AUD vs NZD twin; USD factor cancels | panel/closes | — | ❌ **SUBSUMED** — A6 (not in top20) + orthochan famonly base+audnzd ΔAUC +.0002 (`audusd_15m_orthochan_result.json`) |
| 4 | ~~Carry-unwind risk-bloc DOWN-gate~~ RISK=commod−haven | DOWN | carry unwind asymmetry | panel | — | ❌ **SUBSUMED** — orthochan famonly base+risk ΔAUC −.0009; risk feats not in A6 top20 |
| 5 | **Seed-ensemble (K=3) + calib** | BOTH | variance-reduce the cov2-3% tail; the ONLY robust USDJPY 15m improve-lift | `audusd_15m_cpcv_session.py` nseed>1 | KILL unless seed-ens p10 > single-seed on SAME folds & ≥80% paths | ✅ **DONE — LIFTS** (p10 +.005-.013 & mean up; FROZEN `AUDUSD.m15ny_seedens.v1`) |
| 6 | China/commodity-residual consensus (commod-bloc mean + AUD-vs-bloc gap sign) | BOTH | common idiosyncratic comp of e_AUD/e_NZD/e_CAD = panel-internal China/commodity proxy; AUD lag-completes | panel | KILL if famonly AUC<.515 OR within .005 of CAD/NZD-rotation surrogate | ❌ deferred/subsumed — orthochan risk/audnzd famonly NULL; commod-consensus same family, low prior |
| 7 | Meta-labeling on orthogonal axes (xpair agree, AUDNZD sign, risk sign, session) | BOTH | gate symmetric primary on P(call correct); EURUSD-proven gate, untested AUDUSD | base probs + panel; m15_meta.py | KILL if meta-gated 2026 WR ≤ raw primary at ≥ n; corr(meta,primary)>.9 redundant | ❌ DONE — meta-gate NULL (meta-correctness AUC .52, audusd_15m_metagate_result.json) |

## IMPROVE levers (apply to the best certified edge; lower prior or known-pair-specific)
| # | lever | side | note / Tier-1 status | prior |
|---|---|---|---|---|
| 8 | Triple-barrier first-touch TRAIN-label (eval unchanged) | BOTH | USDJPY: certified-at-level but adversarially NOT a robust improvement (p10 = single noisy order-stat). Fast confirm-and-kill | med |
| 9 | GMADL / \|ret\|-weighted operating-point SELECTION (not retrain) | BOTH | \|ret\|-RETRAIN KILLED both pairs (2026-UP-collapse). SELECTION-only variant untried (METHODS_CATALOG 9.3) | low |
| 10 | Asymmetric per-side conformal (AgACI) gate | BOTH | plain ACI KILLED both pairs; per-SIDE split is the new bit — run only if DOWN-live/UP-dead asymmetry confirmed | low |
| 11 | ~~Realized semivariance asymmetry RS+/RS−~~ | DOWN | ❌ **SUBSUMED** — orthochan famonly base+semi ΔAUC −.0012 (`audusd_15m_orthochan_result.json`); like EURUSD D7, no net-add | low |
| 12 | Cross-pair learning-to-rank (rank:pairwise over 7 majors) | BOTH | SUBSUMED both pairs (rank-loss → relative not own-sign; reduces to BCE). Must BEAT plain A6 or fully subsumed | low |
| 13 | Time-of-week / RBA-cycle calendar conditioning in Asia | unclear | USDJPY gotobi/Tokyo-fix N5 KILLED (+.0004); calendar = SIZE not SIGN elsewhere. Run once inside Asia seg | low |

## TICK / EXTERNAL (deferred — run only if bar levers survive or stall)
| # | lever | note |
|---|---|---|
| T1 | Tick-resolution AUDNZD RV + Asia signed-OFI (forward-holdout-gated, SIGNED-only) | AUDUSD+NZDUSD ticks on disk. STRONG caution: USDJPY 15m tick-micro KILLED forward (trap#9, regime-feature memorization). Use SIGNED imbalance ONLY, drop spread/intensity. Run after bar AUDNZD lever survives |
| X1 | Intraday AU-US rate-differential + risk-on/off (equity/VIX) + commodity (iron-ore/CRB) | EXTERNAL, off-disk. DL-review's dominant external direction carrier. Source only if on-disk panel-internal risk/commodity proxies (rows 4,6) prove insufficient |

## Discovery R2 (2026-06-09) — deep-mine of repo previous-model .md docs + research logs (workflow audusd-15m-docmine-r2)
Ranked levers to push accuracy higher on the certified NY seed-ens book (p10 .596@cov2 / mean .616). Status reflects this session's runs.
| # | lever | prior | status / disposition |
|---|---|---|---|
| R2-1 | Per-SIDE coverage operating-point (DOWN tight / UP looser) | high | ✅ ANALYZED — no-op for the K=3 ensemble: UP peaks cov2 (.596), DOWN peaks cov2 (.5962); shared-cov2 is already per-side-optimal (the lever only helped the single-seed where DOWN peaked cov3 .5972). Deploy nuance: DOWN@cov5 .587 gives more trades (med_n 1172 vs 441) — coverage/floor tradeoff, both certified. |
| R2-2 | True-tick-settlement VALIDATION (raw AUDUSD ticks, wc_ret) | high (hygiene) | ✅ DONE — PRESERVED (mean tick−bar −.0035; certified WR real on traded prices). `audusd_15m_ticksettle_result.json` |
| R2-3 | Seed-ensemble K=3→K=8 + temp calib | high | ✅ DONE — K=8 SATURATED (≈K=3, mixed ±.005-.016); K=3 retained. `audusd_15m_cpcv_session_ny_seedens8_result.json` |
| R2-4 | TB first-touch TRAIN-label denoise (eval unchanged) | med | ✅ DONE — NOT robust (UP regresses @cov2 vs seed-ens; means+.007 p10 mixed). KILLED as improvement. |
| R2-5 | Meta-label 'avoid-losers' gate on orthogonal axes | med | ✅ DONE — meta-correctness AUC .52 → NULL (KILLED, as EURUSD .502). |
| R2-6 | Carry-unwind DOWN-GATE (lagged RISK<0, not a feature) | med→low | low after orthochan (risk as feature NULL); the GATE construct distinct but low EV. Deferred. |
| R2-7 | Commodity-residual CONSENSUS (commod-bloc mean + AUD-vs-bloc gap sign) | low | famonly; distinct from risk-diff. Deferred (risk/audnzd already NULL). |
| R2-8 | Cross-horizon STACK (30m parent → 15m child) | low→med | ✅ DONE — corr .72–.81 NOT collinear (vs USDJPY .957) → pursued; but parent equally weak (AUC≈child) → blend/agree non-additive through seed-ens K=3 (KILLED). `audusd_15m_x{horizon,hstack}_ny_result.json` |
| R2-9 | Per-side AgACI (asymmetric conformal) | low | plain ACI killed both pairs; per-side split untried but nested-refit make-or-break. Low. |
| R2-10 | GMADL operating-point SELECTION (not retrain) | low | may reduce to the existing cov-gate (no-op). Deferred. |
| R2-11 | Lead-lag directed-TE + RFF (N17); DL-MLP stack; macro USD-surprise window | low | A6/CCM suggest cross-pair carries no 15m sign; DL = tail-stabilizer; macro-surprise NULL at EURUSD 5m. Deferred. |
| R2-12 | Tick signed-OFI / AUDNZD-RV (frozen-fwd-gated); Hawkes cross-excitation | low | DEFERRED — USDJPY 15m tickmicro KILLED forward (trap#9). Run only if a bar lever survives (none has). |
| **R2-ext** | **AU-US 2y rate-diff + VIX/ES + iron-ore/CRB (EXTERNAL, off-disk)** | med | **THE honest frontier** — DL-review's dominant signed carrier; the only lever with real probability of beating the ~.536 AUC info-bound. OFF-DISK (acquisition TODO). |
| — | A6 pooling; state/complexity/magnitude-bridge/symmetric-ACI/calibration | — | ❌ SUBSUMED (A6 killed; sign-invariance theorem + usdjpy_15m_exhaust_audit). DO NOT re-run. |

## Discovery rounds
- **R1 (2026-06-09):** workflow `audusd-15m-discovery` — 12 corpus + 10 AUD-specific levers, 25-lever subsumption map. Above is the deduplicated ranked queue. Key NEW AUD-specific mechanisms (not in EURUSD/USDJPY ledgers): AUDNZD residual-difference (#3), carry-unwind risk-bloc DOWN gate (#4), commodity-residual consensus (#6), Asia-session hypothesis (#2). NOT yet dry — these are untested.

## CLOSED measured handoff — issue #9 date refresh (2026-07-13)

- The preregistered Apr–May 2026 retrospective ended `INCONCLUSIVE` because the required C-DOWN endpoint was not estimable; pair source: `results/json/m15_book_refresh_53547498c599f2877a2f6616ae725f8f27e11070e3b550e82eb26ec039fb23f6_AUDUSD_replay_result.json`.
- Joint survivor set `S=[]`; no inactive candidate and no prospective shadow were opened. Do not rerun, retune, or reinterpret this spent one-look window. The incumbent and the external-data frontier above remain unchanged. Joint/shadow sources: `results/json/m15_book_refresh_53547498c599f2877a2f6616ae725f8f27e11070e3b550e82eb26ec039fb23f6_joint_replay_result.json`, `results/json/m15_book_refresh_53547498c599f2877a2f6616ae725f8f27e11070e3b550e82eb26ec039fb23f6_shadow_spec_result.json`.
