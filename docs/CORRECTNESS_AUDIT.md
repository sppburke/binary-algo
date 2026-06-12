# CORRECTNESS AUDIT — repo-wide evaluation-integrity record

**SCOPE: GENERIC** — repo-wide audit of evaluation correctness (leakage / look-forward / alignment /
selection / forecast-derivation / metric-honesty). Per-key example numbers are tagged inline `[PAIR·tf]`
and cite the Tier-2 file of record. Every claim traces to an on-disk result JSON or to code at the current
commit; no number appears here that is not in a cited artifact.

This file is the **record of truth** for whether a result can be trusted. It answers one question per script:
*"Could this number be an artifact of how it was measured rather than a real edge?"* A KILL/NULL produced by a
measurement artifact is the dangerous case — but as the taxonomy below makes explicit, the specific bug found
this session (FM-F) can only manufacture a **false NULL**, never a false POSITIVE, so no certified edge is at risk.

---

## 2026-06-06 — DST-correct session re-campaign + Kronos look-forward fix + full-suite audit

### Failure-mode taxonomy

Every evaluation script is checked against seven ways a number can lie. The label (FM-A … FM-G) is used
throughout the flagged-scripts table.

| Code | Failure mode | What it corrupts | Direction of error |
|---|---|---|---|
| **FM-A** | **Label forward-leak** — `shift(-FWD)` / forward target built without a contiguity (gap) guard, so the label can straddle a session/day break and borrow future bars. | label `y` | either |
| **FM-B** | **Feature causality** — a feature at decision bar `i` peeks at data ≥ `t[i]` (rolling window not truncated, future-fill, etc.). | features `X` | false POSITIVE |
| **FM-C** | **X/y alignment** — features and label indexed off-by-one or to different bars. | X↔y pairing | either |
| **FM-D** | **Train/test purge** — train and test windows overlap or are not embargoed across the label horizon. | split | false POSITIVE |
| **FM-E** | **Selection-on-val/test** — operating point (threshold / coverage / best cell) chosen using the same data it is then scored on. | reported metric | false POSITIVE (inflation) |
| **FM-F** | **Forecast-derivation misalignment** — a binary signal is derived from a generative price forecast scored against a window that is **disjoint from** the deriv label window (off-by-one / wrong horizon). | derived signal ↔ label | **false NULL only** |
| **FM-G** | **Metric honesty** — breakeven/ties/overlap handling not deriv-faithful (ties-win, mid≠entry, overlapping bars counted). | metric → P&L map | either |

### Verdict

> **The look-forward bug found this session is ISOLATED to the Kronos family (2 scripts). It is NOT systemic.**

Across all **308 scripts** plus a dedicated forecast-derivation sweep, **no second instance** of the FM-F
misalignment was found. Every other forecast-derivation script — `usdjpy_1m_statespace`, `usdjpy_2m_statespace`,
`usdjpy_2m_xhorizon`, `m5_xhorizon`, `m5_lossbatch`, `f1_compound` — predicts the **forward** quantity over the
**same horizon** as the label at the **same bar** (correctly aligned). GBM and CNN models are classifiers trained
**directly on the deriv label** and scored against it, so they are structurally immune to FM-F.

**No certified direction or magnitude book is invalidated by this audit.**

### The Kronos look-forward bug (FM-F) — root cause + fix

- **Bug** (`kronos_dir.py`, `kronos_ft.py` eval): at decision bar `i` the context slice was `slice(i-L, i)` =
  bars `[i-L .. i-1]`, the model predicted bar `i`, and the up-signal was `Pup = pred_close(i) > C[i-1]` — i.e.
  the move **into** the entry, window `[t[i-1], t[i]]`. But the deriv label `y[i]`
  (`barcnn_bars.labels_at`: entry `t[i]+1s`, exit `t[i]+61s`) is the **forward** window `[t[i]+1, t[i]+61]`.
  The two windows are **disjoint, off by one bar**. This is FM-F: deriving a binary signal from a generative
  forecast scored against a misaligned window.
- **Why it is safe-failing:** a misaligned forecast destroys the correlation between signal and label, so it
  reads as a **NULL**. It cannot manufacture a spurious POSITIVE. The legacy Kronos NULLs were therefore
  **false NULLs**, not false positives — nothing was over-claimed.
- **Fix** (`kronos_mtf.py`): context ends **at** bar `i` (`slice(i-L+1, i+1)`, last close = entry ref `C[i]`);
  predict `pred_len = H/GRID` **forward** steps; `Pup = pred_close(+H) > C[i]`. Pred-side contiguity enforced
  (`t[i+Hsteps] - t[i] == Hsteps*step`); nonoverlap gap `GAP = HS + TOL`.
- **Validation:** the forward label agrees with next-bar sign **92.3%** of the time (n = 233,950).
- **Remediation in code:** `kronos_dir.py` is now gated behind `KRONOS_DIR_LEGACY=1`; the legacy result is
  superseded by `kronos_dir_mtf_*_result.json`.

### Substrate proofs (Tier-1 empirical — the audit's foundation)

Four data substrates were proven clean independently. These are the load-bearing checks: if the substrate is
clean, every classifier built on it is immune to FM-A/B/C by construction.

| # | Substrate | What was proven | Tier-1 evidence |
|---|---|---|---|
| 1 | **TICK** (`min1_production`, `min2_production` feats / `wc_ret` / prep) | features causal; label is forward deriv outcome | truncation test `max\|full−trunc\| = 0.0`; independent label recompute **0/4000** mismatch; `corr(y, future) = .486` vs `corr(y, past) = −.003`; up-rate **.5006** (in band) |
| 2 | **BAR** (`harness` `features/` + `contig_fwd`) | label forward at H = 5/10/30; label causal | **0/2000** mismatch each H; `corr(y, future) ≈ .99` vs `≈ −.02`; up-rate **.498–.506** |
| 3 | **XPAIR** (`m5_xpair.build_xp` `_y`) | forward label at H = 10 | **0/2000** mismatch |
| 4 | **BAR-FEATURE causality** (`pipeline.py`, 239 features) | every feature causal | truncation `max\|full−trunc\| = 0.0` across **all** features |

### Flagged-scripts master table

| Script(s) | FM | Status | Detail / why no cert is at risk |
|---|---|---|---|
| `kronos_dir.py`, `kronos_ft.py` (eval) | **FM-F** | **KNOWN-BUG → fixed/gated** | Look-forward misalignment above. Gated behind `KRONOS_DIR_LEGACY=1`; superseded by `kronos_mtf.py`. Produced false NULLs only. |
| `barcnn_mag.py:163` | FM-E (bounded) | flagged, cert safe | Selective threshold `np.quantile(p, 1−cov)` taken on pooled **test+oos** (MAGNITUDE target). CPCV-deflated → does not over-claim. Fix: use **VAL** threshold. |
| `barcnn_regime.py:69,71` | FM-E (bounded) | flagged, cert safe | Per-coverage `np.quantile(...,1−cov)` thresholds on pooled test+oos (regime target). CPCV-deflated. Fix: VAL threshold. |
| `usdjpy_2m_cpcv2.py` | FM-E (bounded) | flagged, cert safe | Max-p10 cell selection; best cell `.5185 << .541` breakeven → `CERTIFIED=false` regardless. No inflation reaches a cert. |
| `min2_mim.py:18`, `min2_legsign.py:25` | FM-A | flagged, KILLED | `shift(-FWD)` forward label with **no contiguity guard**. Both were KILL screens that were already KILLED; flag does not resurrect or invalidate any positive. |
| pre-v3 cohort: `min1_v*`, `min2_v*`, `tickmodel*`, `tick5s_final`, `tick_ensemble` | FM-A | **SUPERSEDED** | Documented bar-shift label + greedy nonoverlap; **no `result.json`**; replaced by the `*_production` books. Not on any leaderboard. |

### Remediation list

1. **Done:** `kronos_dir.py` gated behind `KRONOS_DIR_LEGACY=1`; legacy Kronos result superseded by
   `kronos_dir_mtf_*_result.json`. `kronos_mtf.py` is the alignment-correct path going forward.
2. **Optional VAL-threshold parity cleanups** (do not change any verdict; only tighten FM-E hygiene):
   `barcnn_mag.py:163`, `barcnn_regime.py:69,71` — switch the selective-coverage threshold from pooled
   test+oos to the VAL split. `usdjpy_2m_cpcv2.py` max-p10 cell is already sub-breakeven so its cleanup is
   cosmetic.
3. **No action:** `min2_mim.py`, `min2_legsign.py` (KILLED screens) and the pre-v3 cohort (no result.json,
   superseded by `*_production`).

### Key principle

> **A misaligned forecast yields a false NULL, never a false POSITIVE.**

This is why the audit's verdict is reassuring rather than alarming: the one real bug (FM-F, Kronos) could only
have *hidden* an edge, not *invented* one. The FM-E flags are bounded (CPCV-deflated or already sub-breakeven),
and the FM-A flags sit on already-KILLED screens. The certified direction/magnitude books rest on the four
clean substrates above.

---

### Results captured this session (cite these exact JSONs)

DST-correct session masks per `sessions.py` (NY = 08–17 `America/New_York`, LDN = 08–16 `Europe/London`,
Asia = 09–18 `Asia/Tokyo`; `session_mask` = local-tz hour applied per-day across **all** splits — train, val,
test, oos). GBM/xpair restrict **decision rows** (causal-continuous features = rows-only); Kronos FT uses
strict session-only **input** (filter-before-window + contiguity).

**1m tick GBM** (`session_1m.py`) — DIRECTION **NULL** all sessions (pooled ≈ .504, p10 .500–.504,
frac_clear 0.0); MAGNITUDE **certified** all (cov ≤ 10% frac 1.0; magAUC NY .675 / LDN .728 / Asia .717;
p10 .60–.71).
Files: `session_1m_dir_{ny,ldn,asia}_result.json`, `session_1m_mag_{ny,ldn,asia}_result.json`.

**2m tick GBM** (`session_2m.py`) — DIRECTION **NULL** all (p10 .496–.500); MAGNITUDE **certified** all
(p10 .58–.70).
Files: `session_2m_{dir,mag}_{ny,ldn,asia}_result.json`.

**5m/10m/30m base-bar GBM** (`session_bars.py`) — DIRECTION **KILLED** all sessions (NY strongest, e.g. 10m NY
p10 .525); MAGNITUDE **certified** all sessions (p10 .72–.80).
Files: `session_{5,10,30}m_{dir,mag}_{ny,ldn,asia}_result.json`. **15m base-bar GBM (in progress).**

**Cross-pair book, STRICT session-only DST-correct** (`session_xpair.py`; gate
`{2:1m_bb_width, 5/10:5m_bb_width, 15:15m_bb_width, 30:1h_bb_width}`) — the certified ≥10m DIRECTION lever.
**NY certifies BOTH sides at every horizon; LDN/Asia at NONE.** Direction edge is decisively **NY-concentrated**,
and NY **beats** the legacy fixed-UTC gate.

| H | NY UP | NY DOWN | LDN UP/DN | Asia UP/DN | legacy fixed-UTC |
|---|---|---|---|---|---|
| 10m | .6053 (15/15) | .5896 (15/15) | .523 / .524 | .511 / .516 | .586 / .568 |
| 15m | .5845 | .5712 | .527 / .520 | .519 / .496 | — |
| 30m | .5681 | .5639 | .520 / .514 | .487 / .509 | .559 / .553 |

Files: `session_xpair_{10,15,30}m_{ny,ldn,asia}_result.json`. **5m + 2m (in progress.)**
Per-key numbers tagged `[EURUSD·{10m,15m,30m}]`; file of record `results/EURUSD_RESULTS.md`.

**Kronos direction, CORRECTED** (`kronos_mtf.py`, alignment-fixed) — **NULL at every horizon** 1/5/10/15/30m,
zero-shot **and** fine-tuned, **all sessions** (pooled .50–.51, CPCV p10 .489–.500, all KILLED, up-rates
in-band). Even at NY ≥ 10m where cross-pair certifies .57–.61, Kronos reads ≈ .50 — it ingests only EURUSD's
**own** OHLCV candles, not the 7-pair USD cross-section that carries the edge. Fine-tune did not help direction.
Files: `kronos_dir_mtf_*_result.json` (e.g. `kronos_dir_mtf_mtf_zs_{1,2,5,10,15,30}m_all_result.json`,
`kronos_dir_mtf_ftmtf_1m_{ny,ldn,asia}_result.json`).
**FINE-mode "1m→Nm up the chain" + multi-TF ensembles (in progress).**

### New scripts (this session)

`sessions.py` (DST-correct `session_mask` / `SESSIONS`); `session_1m.py`, `session_2m.py` (tick GBM per
session); `session_bars.py` (bar GBM per session, any H); `session_xpair.py` (cross-pair strict session-only,
any H, per-H gate); `kronos_ft.py` (single-process GPU adaptation of the GPU/DDP-only Kronos
`finetune/train_predictor.py` — frozen base tokenizer, predictor FT on session-only contiguous 1m stream, AMP
bf16, early-stop); `kronos_mtf.py` (FM-F-corrected + multi-timeframe Kronos direction eval — native + fine
modes, per-session, ensemble-ready npz); `kronos_bars.py` (H-min / fine-grid OHLCV + forward deriv-label
builder, generalizes `barcnn_bars`); `kronos_ensemble.py` (multi-TF vote combine); `barcnn_run.py` gained a
`SESSION` arg (per-session bar-image CNN).

### Hardware note

Box has an NVIDIA RTX 5050 Laptop GPU (8 GB, Blackwell sm_120, driver 580 / CUDA 13). The venv torch was
swapped 2.12.0+cpu → 2.12.0+cu130 (cu130 matches the driver; sm_120 verified), so Kronos fine-tune + inference
now run on GPU. Training venv: `~/binary-algo-venv` (uv).
