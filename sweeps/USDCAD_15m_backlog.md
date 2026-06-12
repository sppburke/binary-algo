SCOPE: USDCAD · 15m — EXECUTABLE backlog (TOP-N queue, incumbents-to-beat, discovery rounds). Ledger/status: `USDCAD_15m.md`. Results: `results/USDCAD_RESULTS.md`. Generic ideas: `docs/IDEAS_LOG.md`.

# USDCAD 15m direction — executable backlog

## ★ CAMPAIGN CLOSED 2026-06-10 — on-disk direction key EXHAUSTED (run + adversarially verified)
**Deliverable:** `USDCAD.m15ny_seedens.v1` (NY own-pair seed-ens K=3) — UP p10 .604@cov1/.5968@cov2, DOWN .5814@cov5/.5805@cov3, both 15/15. **Every distinct-mechanism on-disk lever RUN** (incl. the gap-closing campaign that replaced cite-subsumption with actual runs: TB-label, cross-horizon, specialist, |ret|-weight, DL-GRU, Optuna). **Two refit-CPCV positives (TB +.017 UP, cross-horizon +.01-.016) KILLED by frozen-forward = refit-overfit.** DL-GRU (.5203) + Optuna (anti-transfer) confirm the AUC ceiling ~.532 is fundamental. **Magnitude edge surfaced** (.70-.75 → docs/MAGNITUDE_FINDINGS.md). >65% = off-disk frontier (oil/rate-diff, user-excluded). Remaining queue items below are SUBSUMED by USDCAD's own Tier-1 results (xpair screen, magdir, |ret|-weight) or theorem (ACI) — not cross-pair argument. **Open beyond this key:** USDCAD magnitude certification; other timeframes; off-disk data.

**Incumbents (Tier-1):** USDCAD NY seed-ens K=3 p10 UP .5968/DOWN .5814 @op-cov (the certified deliverable). Cross-key 15m floors: USDJPY .60/.57, AUDUSD .596/.596, GBPUSD xpair-NY K=8 .655/.640, EURUSD xpair .567/.574. CERTIFY only via per-fold-refit CPCV (p10≥.541, ≥80% of 15 paths) AND survive a frozen-past forward holdout (trap#9) — the gate that killed TB + cross-horizon here.

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
- **R2 (2026-06-10) — workflow `usdcad-15m-discovery-r2` (coverage-audit + fresh web pass): DRY.** No new survivable on-disk direction lever. All R1 in-flight levers closed NULL (OFI +.0002, meta-gate .5298, kNN/two-speed/double-ortho/pooling killed). R2 candidates all already-run or Tier-1 cite-subsumed (cross-horizon, per-side specialist, compression-release=magnitude, ACI/calibration=sign-invariant wrappers). **Bonus:** on-disk oil ABSENT (NYMEX minute empty); ES/NQ on disk but ES→FX lead-lag already null (EURUSD m30_es_feas). **2nd consecutive dry round → discovery loop EXHAUSTED on-disk.** Off-disk frontier (WTI/CL, US-CA rate-diff, VIX/RR) re-confirmed as the only path to >65%.
- **R1 (2026-06-10) — workflow `usdcad-15m-discovery-r1` (corpus _extracted_levers.json + FX-direction papers + web arXiv/SSRN 2021-26 + 4-pair subsumption map; sign-invariance-filtered, deduped).** 16 levers vetted. **Honest synthesis:** every on-disk family caps directional AUC ~.539 and certified WR ~.60-.625 across all 5 pairs studied → the ONLY credible path past the ~.60 floor to >65% is OFF-DISK (oil/rate-diff/VIX), each trap-#9-gated by frozen-past forward holdout. On-disk USDCAD order-flow (features_of/, features_tick_xofi/) EXISTS but signed-OFI direction was run-and-killed at 15m on USDJPY (tickmicro fwd 2026 −.035) + EURUSD (sign lives at seconds, 900s too far) → low-prior confirm-once.

### R1 RUN queue (new on-disk/panel direction levers, ranked) — apply to certified NY base, refit-CPCV cov2
| # | lever | side | mechanism (sign) | data | prior | fast-KILL | status |
|---|---|---|---|---|---|---|---|
| R1-1 | **Two-speed momentum SIGN-agreement** (slow 1h/4h × fast 5m/15m sign-agree as a DIRECTION feature) | BOTH | agreed sign IS the predicted direction; disagreement=correction/rebound (no edge). Never run as a sign-agreement feature on any pair; NY window is where BoC/Fed/oil align the speeds | on-disk | **med** | KILL if both sides p10 ≤ NY base (UP .6044/DOWN .5791) AND agreement feat outside top-20 SHAP | pending |
| R1-2 | **Cross-pair USD-residual pooling** (panel e_USDCAD/fac, sign-aligned: USD in NUMERATOR → residual sign inverts vs AUD/NZD) | BOTH | EUR-bloc certifying keystone; does USDCAD pool (EURUSD/GBPUSD) or dilute (USDJPY/AUDUSD)? | panel | low | KILL if neither side p10 beats NY base AND VAL AUC ≤ .5319+.003 (expect dilute, run-don't-argue) | pending (=ledger A6) |
| R1-3 | **kNN regime-matcher** in Z-scored state space (instance-based, orthogonal to GBM partition) | BOTH | signal = sign(mean subsequent 15m ret of K nearest NY bars in top-20-feat space); free ensemble member if it decorrelates from GBM | on-disk | low | KILL if standalone p10 < BE both sides AND kNN-GBM blend ≤ seed-ens K=3 (AUDUSD xhstack standard: variance-only = redundant) | pending |
| R1-4 | **Double-orthogonalization residual** (purge USD fac AND commodity-bloc mean(e_AUD,e_NZD) → CAD-idiosyncratic = on-disk OIL PROXY) | DOWN | isolates oil-driven CAD flow w/o oil data; the predicted more-forecastable DOWN side | panel | low | KILL if DOWN p10 ≤ single-purge xpair (R1-2). Run only if R1-2 shows any DOWN signal | pending (contingent on R1-2) |
| R1-5 | **Signed-OFI / cks1s order-flow direction** (confirm-once; on-disk features_of/ + features_tick_xofi/) | BOTH | signed order-flow residual as direction feat | on-disk | low | KILL if p10 ≤ NY base both sides (expect KILL: USDJPY/EURUSD 15m precedent + sign-invariance, sign lives at seconds). Run only after R1-1..4 + before declaring on-disk dry | pending |

### R1 DEFER — OFF-DISK frontier (the honest path to >65%; data-acquisition TODO, each trap-#9 frozen-fwd-gated)
| # | lever | side | mechanism | acquire |
|---|---|---|---|---|
| X1 | **Intraday WTI/oil** (return + realized-skew; LLM oil sentiment) | DOWN | WTI↓⇒CAD weak⇒USDCAD↑; the USDCAD analog of AUDUSD's iron-ore frontier. **Highest-value next action** | WTI/Brent intraday series |
| X2 | **US-CA 2y/10y rate-differential** lead-lag | UP | widening Fed-BoC spread⇒USD-strength⇒USDCAD↑; most tractable rate-diff pair (both legs liquid NA) | US+CA 2y/10y intraday |
| X3 | **VIX / FX 25d risk-reversal** risk-off gate | UP | risk-off⇒USD-strength⇒USDCAD↑ (inverse of AUDUSD, USD-numerator) | VIX + FX RR series |

### R1 SUBSUMED (run-and-killed across ≥2 prior pairs — DO NOT re-run; Tier-1 rationale)
Order-flow/OFI direction (USDJPY+EURUSD 15m, sign-invariance) [kept as R1-5 confirm-once since on-disk]; Fourier/seasonal decomposition→sign (EURUSD FAM2, sign-invariance carries SIZE); pinball/quantile & ranking loss (gate-only, TFT-fan killed); time-of-day positional (trap-#9 forward-decay; base already has hour_sin/cos + NY restriction is the certified timing lever); cross-horizon 15m×30m stack (AUDUSD: decorrelated-but-equally-weak → redundant w/ seed-ens); meta-gate orthogonal-axes (EURUSD .502 / AUDUSD .529 < .53 bar); neural/attention/CNN/VAR-FNN direction (84-arm campaign 0 survivors, info-bound); RL/MtM state features (D4-D5 info-bound capped).

### Earlier R1 cross-pair-transfer queue (pre-corpus, superseded by the above)
Original FIRST-TO-RUN queue (NY session ✅done, all-session ✅done, seed-ens running) retained in the FIRST-TO-RUN table at top.
