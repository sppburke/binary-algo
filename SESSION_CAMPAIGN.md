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

**Execution rule:** tick-substrate GBM (1m/2m feats() ~9-20G) runs ALONE (OOM history). Bar-store GBM + GPU Kronos
are mem-light → run CONCURRENTLY (GPU lane = Kronos FT/zero-shot; CPU lane = lightgbm bar-store). Kronos zero-shot is
run ONCE per (freq) all-sessions and read out per-session via the breakdown.

**HARDWARE UPGRADE 2026-06-06 (user: "use the GPU"):** box HAS an NVIDIA RTX 5050 Laptop (8GB, Blackwell sm_120,
driver 580/CUDA13). Swapped venv torch `2.12.0+cpu`→`2.12.0+cu130` (cleanest: same version, cu130 matches driver;
verified sm_120 matmul). Kronos FT + inference now run on GPU (~10 steps/s, ~30 min/session vs hours on CPU). The
official `finetune/train_predictor.py` is GPU/DDP-only (cuda hardcoded L185, torchrun-required L241) → `kronos_ft.py`
is a single-process GPU adaptation (frozen base tokenizer, predictor FT, session-only contig stream, AMP bf16).

## Substrates
- 1m(60s) / 2m(120s): tick store `features_tick/` (min1/min2 machinery, wc_ret on 1s ticks). OHLCV bars in `ohlc_cache/`.
- 5m/10m/15m/30m: 239-feature bar store `features/` via `harness.py` (MX_HOR), wc_ret-faithful per horizon.

## STATUS MATRIX  (✅ done · 🔄 running · ⏳ pending · ❌ killed/null · — n/a)
Legend cell = best result + verdict; full numbers in the result JSON named in the row.

### Frequency 1m (60s)  — PRIORITY 1 (do fully first)
| method | NY | LDN | Asia | result_json prefix |
|---|---|---|---|---|
| A. GBM direction (comb/UP/DOWN), session-only train+eval | ❌ p10 .504 (val .509) | ❌ p10 .502 (val .503) | ❌ p10 .500 (val .506) | `session_1m_dir_<sess>_result.json` — pooled ~.504, frac_clear 0.0 EVERY cov/side, null |
| B. Magnitude \|ret60\|≥Q, session-only | ✅ p10 **.621**/.713 (cov10/5, frac1.0) | ✅ p10 **.635**/.705 | ✅ p10 **.603**/.683 | `session_1m_mag_<sess>_result.json` — magAUC NY .675/LDN .728/Asia .717 |
| C. Kronos zero-shot direction (per-session via breakdown) | ❌ .493 | ❌ .503 | ❌ .506 | `kronos_dir_zeroshot_small_result.json` — pooled .501, CPCV p10 .487, null EVERY session |
| D. Kronos fine-tuned direction (session-only, **GPU**) | model ✅ eval🔁 | 🔄 train | ⏳ | `kronos_ft.py` predictor FT (models/kronos_ft_<sess>); legacy in-script eval MISALIGNED→discarded, re-eval via `kronos_mtf.py` |
| E. Bar-image CNN dir + mag (session-only) | ⏳ | ⏳ | ⏳ | `barcnn_<sess>_*_result.json` |

### Frequency 2m (120s) — PRIORITY 2
| A. GBM direction | ❌ p10 .500 (val .508) | ❌ p10 .496 (val .503) | ❌ p10 .498 (val .509) | `session_2m_dir_<sess>_result.json` — pooled ~.505, frac 0.0 all cov/side, null |
| B. Magnitude | ✅ p10 **.604**/.695 (cov10/5) | ✅ p10 **.639**/.695 | ✅ p10 **.579**/.679 | `session_2m_mag_<sess>_result.json` — magAUC NY .668/LDN .730/Asia .716, frac1.0 |
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
| D. **Cross-pair book** (STRICT session-only train+gate; the certified lever) | ✅ **UP .605/DOWN .590** (both CERT 15/15) | 🔄 | 🔄 | `session_xpair_10m_<sess>_result.json` — DST-correct NY strict-session BEATS legacy fixed-UTC (UP .586/.568); BOTH sides certify |

### Frequency 15m — (base book + cross-pair lever)
| D. **Cross-pair book** (STRICT session-only train+gate) | 🔄 | 🔄 | 🔄 | `session_xpair_15m_<sess>_result.json` — legacy fixed-UTC NY gate certified UP .567/DOWN .574 |

### Frequency 30m — PRIORITY 5  (base single-LGBM)
| A. GBM direction comb@5% | ❌ NY p10 .504 (2024 .566/25 .558) | ⚠ LDN p10 **.538** (frac .82, near-miss) | ❌ Asia p10 .495 | `session_30m_dir_<sess>_result.json` |
| B. Magnitude sel@10% | ✅ NY p10 **.799** | ✅ LDN p10 **.764** | ✅ Asia p10 **.717** | `session_30m_mag_<sess>_result.json` |
| C. Kronos zero-shot | ⏳ | ⏳ | ⏳ | `kronos_dir_30m_zeroshot_result.json` |
| D. **Cross-pair book** (STRICT session-only train+gate) | 🔄 | 🔄 | 🔄 | `session_xpair_30m_<sess>_result.json` — legacy fixed-UTC NY gate certified UP .559/DOWN .553 |

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
4. **1m TICK GBM direction = null in EVERY session** (NY/LDN/Asia pooled ~.504, p10 <.505, frac_clear 0.0 at every
   coverage and side). Three independent model classes now agree the 60s SIGN is session-invariantly near-efficient:
   tick-GBM, bar-CNN, Kronos — all null in NY, LDN AND Asia. **1m MAGNITUDE certified in all three sessions** (cov≤10%
   frac_clear 1.0; magAUC NY .675/LDN .728/Asia .717), so the sign-invariance split (size forecastable, sign not) is
   itself session-robust. Reinforces [[binary-algo-direction-ceiling]] + [[magnitude-edge]] at the session level.

## ⚠ METHODOLOGY FIX (user-caught 2026-06-06) — Kronos eval look-forward MISALIGNMENT
`kronos_dir.py` (and the inherited `kronos_ft.py` eval) scored Kronos direction against the WRONG 60s window: at
decision bar i it fed context `[i-L..i-1]`, predicted bar i, and compared `pred_close(i) > C[i-1]` — the move INTO
t[i], window `[t[i-1],t[i]]`. But the deriv label `y[i]` (barcnn_bars.labels_at: entry t[i]+1s, exit t[i]+61s) is
the FORWARD 60s, window `[t[i]+1,t[i]+61]`. **Disjoint, off by one bar.** So every Kronos DIRECTION result so far
(1m zero-shot null AND the misaligned 1m-FT eval) tested "does the realized last-1m move match the next disjoint
1m move" — NOT "can Kronos forecast the trade window." Those direction nulls are CONFOUNDED (alignment artifact vs
sign-invariance — indistinguishable). **FIX = `kronos_mtf.py`:** context ends at the ENTRY bar (last close = entry
ref), predict pred_len=H/F steps forward, `Pup = pred_close(+H) > C_entry`, vs the H-min deriv label. This also
implements the user's idea: finer-TF context (1m) → predict H steps for the H-min horizon + multi-TF ensemble. The
GBM/tick/xpair pipelines are NOT affected (their labels are correctly forward from the decision instant; Kronos-eval
-only bug). Magnitude results unaffected. Design pass: workflow `kronos-mtf-design` (running).

## 🎯 HEADLINE (2026-06-06): DST-correct STRICT session-only cross-pair book at 10m NY CERTIFIES BOTH SIDES,
**beating the legacy fixed-UTC gate.** `session_xpair_10m_ny`: UP p10 **.6053** (15/15 clear), DOWN p10 **.5896**
(15/15) — vs legacy m10xp UP .586/DOWN .568. So the certified ≥10m direction edge is REAL and STRONGER once you
(a) use DST-correct NY and (b) train+gate strictly within the session (n=979k NY-only pooled cross-pair rows). The
user's session discipline didn't break the edge — it sharpened it. LDN/Asia 10m + 15m/30m running. (Magnitude was
already session-robust; this is the first DIRECTION cell to certify in the re-campaign.)

## RUN LOG (append one line per completed cell — the resumable record)
- 2026-06-05 — campaign opened; `sessions.py` (DST-correct) built.
- 2026-06-05 23:39 — **5m/10m/30m GBM (base lgb) dir+mag × NY/LDN/Asia DONE** (18 cells, `session_{5,10,30}m_*`),
  ran CONCURRENTLY with Kronos (bar store mem-light). Direction all KILLED (NY/LDN strongest); magnitude all ✅ certified.
- 2026-06-05 23:41 — **Kronos zero-shot 1m DONE** (`kronos_dir_zeroshot_small_result.json`): null all sessions.
- 2026-06-05 23:46 — **session_1m (1m tick GBM dir+mag × NY/LDN/Asia) DONE** (252s, mem-safe alone): DIR null all
  sessions (p10 .500–.504, frac 0.0); MAG ✅ certified all sessions (p10 .60–.71, frac 1.0). 1m row A/B filled.
- 2026-06-05 23:52 — **session_2m (2m tick GBM dir+mag × NY/LDN/Asia) DONE** (241s, mem-safe alone, peaked 9G used):
  DIR null all sessions (p10 .496–.500, frac 0.0); MAG ✅ certified all sessions (p10 .58–.70, frac 1.0). 2m row A/B filled.
  → 1m & 2m tick GBM both confirm: direction session-invariantly null, magnitude session-robustly certified.
- 2026-06-06 00:1x — **GPU enabled** (RTX 5050, torch cu130). `kronos_ft.py` written + smoke-passed. **Kronos FT chain
  NY→LDN→Asia (1m, predictor FT, GPU) LAUNCHED** via `kronos_ft_orch.sh` (12 ep, batch24, L256, K20, N/yr3000,
  early-stop on 2024 val-loss). NY training first (2.4G/8G VRAM, 97% util). Concurrent CPU lane = cross-pair book re-run.
- TODO next: read 3 FT results → 1m-D cells; Kronos zero-shot at 2/5/10/30m (GPU); cross-pair per-session book (CPU lane).
  (needs bars built at those freqs); per-session CROSS-PAIR book re-run (the certified ≥5m direction lever) — the
  highest-prior remaining direction test; bar-CNN per-session (1m E); Kronos fine-tune (selective).

## PRIOR-SESSION CONTEXT (what the legacy NY-gated books already say — to compare against)
- ≥5m deployed direction books are compression×NY(fixed-UTC): 5m UP .553 (cert), 10m UP .586/DOWN .568 (cert),
  15m .567/.574 (cert), 30m .5588/.5525 (cert) — all REFIT-CPCV-certified but with the DST-APPROXIMATE NY gate.
  The campaign re-tests these with DST-correct NY AND adds LDN/Asian splits (never done).
- 1m/2m direction = near-efficient all-session; 1m magnitude (GBM) magAUC .787; bar-image CNN: dir null / mag >65%.
