SCOPE: CROSS-KEY CAMPAIGN LEDGER (resumable state) — DST-correct, per-session (NY/LDN/Asian) re-test of EVERY frequency × method. Generic methodology; per-cell numbers also land in the key-specific files (EURUSD_RESULTS.md / MAGNITUDE_FINDINGS.md / DIRECTION_FINDINGS.md / sweeps/). See REPO_MAP.md, sessions.py.

# Session re-campaign — DST-correct, session-segmented re-test of the whole suite

**Why (user directive 2026-06-05):** prior tests applied session-conditioning inconsistently — the deployed ≥5m
direction books gate on `sess_ny` but with a **fixed-UTC (12–21) DST-APPROXIMATE** window; the 1m/2m direction,
bar-image CNN, magnitude, and Kronos work used **all 24h, no session filter**. Re-test EVERY (frequency × method)
with **DST-correct exchange-local sessions** (`sessions.py`), restricting **train + val + test + OOS decision bars
to ONE session** (NY, LDN, Asian) and running each session **separately**. Features stay causal/continuous; only
the prediction/label rows are session-filtered.

**Sessions (DST-correct, `sessions.py`):** NY = 08–17 America/New_York · LDN = 08–16 Europe/London · Asia = 09–18
Asia/Tokyo. (Replaces the legacy fixed-UTC `pipeline.py` `sess_ny`.)

**Discipline (unchanged, per `.claude/skills/strategy-eval/SKILL.md`):** deriv-faithful wc_ret (ties LOSE,
breakeven 0.541), nonoverlap_chrono, per-year CI95 (2024/2025/2026), worst-VAL-half selection, moved-bars
up-rate∈[.47,.53], pre-registered falsifier, purged-combinatorial CPCV (path_p10 ≥ 0.541 to certify), moved-bars.

**Falsifier (per cell):** KILL if VAL dirAUC ≤ 0.515 OR no held-out year moved-acc CI95-lo ≥ 0.541 OR CPCV
path_p10 < 0.541. Magnitude cell: report magAUC + selective precision; "edge" if CPCV path_p10(precision) clears.

**Execution rule:** ONE heavy job at a time (OOM history — killed the box once). Serialize GBM/CNN/Kronos fits;
Kronos zero-shot is run ONCE per (freq) all-sessions and read out per-session via the breakdown (no training →
no need to session-filter inputs). Kronos FINE-TUNE is session-only and EXPENSIVE on CPU → selective.

## Substrates
- 1m(60s) / 2m(120s): tick store `features_tick/` (min1/min2 machinery, wc_ret on 1s ticks). OHLCV bars in `ohlc_cache/`.
- 5m/10m/15m/30m: 239-feature bar store `features/` via `harness.py` (MX_HOR), wc_ret-faithful per horizon.

## STATUS MATRIX  (✅ done · 🔄 running · ⏳ pending · ❌ killed/null · — n/a)
Legend cell = best result + verdict; full numbers in the result JSON named in the row.

### Frequency 1m (60s)  — PRIORITY 1 (do fully first)
| method | NY | LDN | Asia | result_json prefix |
|---|---|---|---|---|
| A. GBM direction (comb/UP/DOWN), session-only train+eval | ⏳ | ⏳ | ⏳ | `session_1m_dir_<sess>_result.json` |
| B. Magnitude \|ret60\|≥Q, session-only | ⏳ | ⏳ | ⏳ | `session_1m_mag_<sess>_result.json` |
| C. Kronos zero-shot direction (per-session via breakdown) | ❌ .493 | ❌ .503 | ❌ .506 | `kronos_dir_zeroshot_small_result.json` — pooled .501, CPCV p10 .487, null EVERY session |
| D. Kronos fine-tuned direction (session-only) | ⏳ | ⏳ | ⏳ | `kronos_dir_ft_<sess>_result.json` |
| E. Bar-image CNN dir + mag (session-only) | ⏳ | ⏳ | ⏳ | `barcnn_<sess>_*_result.json` |

### Frequency 2m (120s) — PRIORITY 2
| A. GBM direction | ⏳ | ⏳ | ⏳ | `session_2m_dir_<sess>_result.json` |
| B. Magnitude | ⏳ | ⏳ | ⏳ | `session_2m_mag_<sess>_result.json` |
| C. Kronos zero-shot | ⏳ | ⏳ | ⏳ | `kronos_dir_2m_zeroshot_result.json` |
| D. Kronos fine-tune | ⏳ | ⏳ | ⏳ | — |

### Frequency 5m — PRIORITY 3  (base single-LGBM, session-only; NOT the cross-pair certified book)
| A. GBM direction comb@5% | ❌ NY p10 .518 (2024 .570) | ❌ LDN p10 .500 | ❌ Asia p10 .508 | `session_5m_dir_<sess>_result.json` |
| B. Magnitude sel@10% | ✅ NY p10 **.793** (frac1.0) | ✅ LDN p10 **.798** | ✅ Asia p10 **.772** | `session_5m_mag_<sess>_result.json` |
| C. Kronos zero-shot | ⏳ | ⏳ | ⏳ | `kronos_dir_5m_zeroshot_result.json` |

### Frequency 10m — PRIORITY 4  (base single-LGBM)
| A. GBM direction comb@5% | ❌ NY p10 **.525** (2024 .568) | ❌ LDN p10 .513 (2026 .581) | ❌ Asia p10 .486 | `session_10m_dir_<sess>_result.json` |
| B. Magnitude sel@10% | ✅ NY p10 **.790** | ✅ LDN p10 **.797** | ✅ Asia p10 **.753** | `session_10m_mag_<sess>_result.json` |
| C. Kronos zero-shot | ⏳ | ⏳ | ⏳ | `kronos_dir_10m_zeroshot_result.json` |

### Frequency 30m — PRIORITY 5  (base single-LGBM)
| A. GBM direction comb@5% | ❌ NY p10 .504 (2024 .566/25 .558) | ⚠ LDN p10 **.538** (frac .82, near-miss) | ❌ Asia p10 .495 | `session_30m_dir_<sess>_result.json` |
| B. Magnitude sel@10% | ✅ NY p10 **.799** | ✅ LDN p10 **.764** | ✅ Asia p10 **.717** | `session_30m_mag_<sess>_result.json` |
| C. Kronos zero-shot | ⏳ | ⏳ | ⏳ | `kronos_dir_30m_zeroshot_result.json` |

## KEY FINDINGS SO FAR (session split, DST-correct)
1. **MAGNITUDE is session-ROBUST and certified >75% in ALL sessions × ALL frequencies (5/10/30m).** Selective
   large-call precision .72–.85, CPCV **frac-paths-clear = 1.0** in NY, LDN, AND Asia. The magnitude edge does NOT
   depend on session — it's everywhere. (Reinforces [[magnitude-edge]]; session-only training doesn't break it.)
2. **DIRECTION (base single-LGBM, session-only) does NOT certify in any session/frequency**, BUT signal concentrates
   exactly where the deployed NY-gated books live: **NY strongest at 5m/10m** (10m NY CPCV p10 .525, 2024 .568),
   **LDN strongest at 30m (p10 .538, 82% paths clear — near-miss)**, **Asia weakest throughout**. So the user's
   session intuition is VALIDATED directionally (NY/LDN carry the sign, Asia is dead) — but the BASE model still
   doesn't clear breakeven. CAVEAT: this is the base lgb, NOT the cross-pair book that certifies the deployed ≥5m
   direction; re-running the cross-pair book per-session (DST-correct) is the open higher-prior follow-up.
3. **Kronos zero-shot 1m direction = null in EVERY session** (NY .493 / LDN .503 / overlap .501 / Asia .506; pooled
   .501, CPCV p10 .487). Session-conditioning does not rescue Kronos for 1m direction.

## RUN LOG (append one line per completed cell — the resumable record)
- 2026-06-05 — campaign opened; `sessions.py` (DST-correct) built.
- 2026-06-05 23:39 — **5m/10m/30m GBM (base lgb) dir+mag × NY/LDN/Asia DONE** (18 cells, `session_{5,10,30}m_*`),
  ran CONCURRENTLY with Kronos (bar store mem-light). Direction all KILLED (NY/LDN strongest); magnitude all ✅ certified.
- 2026-06-05 23:41 — **Kronos zero-shot 1m DONE** (`kronos_dir_zeroshot_small_result.json`): null all sessions.
- 2026-06-05 23:41 — **session_1m (1m tick GBM dir+mag × sessions) RUNNING ALONE** (heavy feats() transient).
- TODO next: 1m tick results → record; then 2m tick (session_2m, TO WRITE); Kronos zero-shot at 2/5/10/30m;
  per-session CROSS-PAIR book re-run (the certified ≥5m direction lever) — the highest-prior remaining direction test.

## PRIOR-SESSION CONTEXT (what the legacy NY-gated books already say — to compare against)
- ≥5m deployed direction books are compression×NY(fixed-UTC): 5m UP .553 (cert), 10m UP .586/DOWN .568 (cert),
  15m .567/.574 (cert), 30m .5588/.5525 (cert) — all REFIT-CPCV-certified but with the DST-APPROXIMATE NY gate.
  The campaign re-tests these with DST-correct NY AND adds LDN/Asian splits (never done).
- 1m/2m direction = near-efficient all-session; 1m magnitude (GBM) magAUC .787; bar-image CNN: dir null / mag >65%.
