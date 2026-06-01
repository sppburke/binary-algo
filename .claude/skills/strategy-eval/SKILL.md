---
name: strategy-eval
description: >
  Evaluate or reverse-engineer a binary up/down direction (or magnitude) strategy in THIS repo
  (/media/sean/CORSAIR/binary-algo) and record the result correctly. Use whenever the task is to test a
  model/methodology at a (currency, timeframe, side); retarget an existing method to a new horizon or
  currency; reference/load/freeze a model from the registry (books/); or write a result into a
  <PAIR>_RESULTS.md. Enforces the deriv-faithful evaluation discipline and the unique-key results schema so
  results are trustworthy and comparable. Triggers: "evaluate this strategy", "reverse engineer a model for
  <pair>/<timeframe>", "does <method> work at <horizon>", "add a result to the ledger", "freeze a book".
---

# Strategy evaluation & reverse-engineering (binary direction / magnitude)

This repo forecasts deriv.com Rise/Fall binaries: `sign(close(t+H) - close(t))` (direction) and `|ret_H|≥Q`
(magnitude). Years of work converged on hard constraints. **Follow this protocol exactly — most "edges" are
artifacts of broken discipline, and the program has already catalogued how.**

## 0. The unique key and where results go
Every result is keyed `(currency, timeframe, side)`, side ∈ {UP, DOWN} (COMBINED is context, not a side key).
Results for a currency live in `<PAIR>_RESULTS.md` (e.g. `EURUSD_RESULTS.md`). A new currency gets its own
file, copied from `EURUSD_RESULTS.md`'s exact structure. **Never pool across timeframes; never copy one
timeframe's or one side's number into another key.**

## 1. Before touching code — read, don't redo
1. `METHODS_CATALOG.md` — every methodology, how to retarget it (env `MX_HOR=<minutes>` for bar models;
   `HS`/`HSEC` seconds for tick models), its leakage traps, and its status. Most methods are already
   implemented and horizon-parameterized — check before writing anything new.
2. `<PAIR>_RESULTS.md` — is this exact key already measured? If so, don't redo; if you'll try to beat it,
   note the incumbent's binding-year stat (you must clear it on the WORST held-out year).
3. `MODEL_REGISTRY.md` + `books/INDEX.json` — the frozen books you can load as parents/baselines.

## 2. MANDATORY evaluation discipline (non-negotiable — a result that skips any of these is invalid)
- **Deriv-faithful settlement.** Reuse `wc_ret()` from `min1_production.py` (or the horizon's production
  script): wall-clock mid-to-mid, entry = NEXT tick after order (+1s lag), exit = last tick ≤ expiry,
  **ties LOSE**. Breakeven win-rate **0.541** (payout R≈1.85). Do NOT use `mid.shift(-N)` on gap-dropped
  bars — it shifts *bars* not *seconds* (the biggest historical inflation).
- **Independent trades.** `nonoverlap_chrono(ts, mask, gap)` (first-come, NO look-ahead). Never the greedy-by-
  confidence de-overlap for a headline (it peeks; costs −3..−9 acc pts).
- **Held-out per year.** Report 2024, 2025, AND 2026 separately with bootstrap CI95 (`boot()`); 2026 = strict
  OOS. A pooled/combined number alone is NOT sufficient — the binding constraint is almost always the 2025
  regime.
- **Selection.** Choose any threshold/gate on VAL by **worst-VAL-half** stability. NEVER VAL-acc-max:
  `corr(VAL_acc, OOS_acc) = −0.54` — VAL-maximal pockets anti-transfer.
- **Moved-bars only.** Evaluate on `|ret|>0` bars and **verify the moved up-rate ∈ [0.47, 0.53]**. An up-rate
  outside that band means a fake-flat mirage (see traps) — the result is invalid regardless of accuracy.
- **Pre-register a falsifier BEFORE looking at OOS.** State the exact KILL conditions (e.g. "KILL if VAL
  dirAUC ≤ 0.515, OR no held-out year's moved-acc CI95-lower clears <breakeven/target>"). Write it into the
  result JSON. A method that survives a pre-registered falsifier is real; one tuned until it passes is not.
- **Memory/compute.** One heavy job at a time (OOM history); subsample train ≤100–150k; build features
  month-by-month for tick data; the repo + data live on a flaky external USB SSD (`udisksctl mount -b
  /dev/sdb2` if it drops; builds should be idempotent/resumable).

## 3. The seven recurring leakage traps (each has produced a fake edge here — check all)
1. Sequence-model future-peek: HMM Viterbi/forward-backward, Kalman RTS smoother → use forward FILTER only.
2. ffill-flat-window mirage: intersecting pairs + ffill manufactures ~50% fake-flat bars → fake AUC ~0.7
   that collapses to ~0.49 on moved bars. Keep the target pair's OWN clock; missing → 0.0, never ffill;
   verify up-rate ∈ [0.47,0.53].
3. Bar-shift horizon mislabel (use `wc_ret`).
4. Greedy-by-confidence de-overlap (use `nonoverlap_chrono`).
5. VAL-acc-max selection (use worst-VAL-half).
6. Thin-coverage mirage: any n<25–50 pocket at a high number is multiple-testing noise.
7. Ties LOSE: a ~0.50-AUC model's realized win-rate sits BELOW 0.50 once ties are charged.

Also remember the **sign-invariance theorem** (arXiv:2512.15720): entropy / order-flow / complexity / HMM /
Hurst / Kalman statistics gate move SIZE (magnitude), not SIGN. If a "direction" idea is one of these, it is
almost certainly a magnitude edge — test it as magnitude (`|ret|≥Q`, AUC), not direction.

## 4. Reusable building blocks (import, don't reinvent)
- `harness.py` — `FEAT_DIR`, `feature_cols(pair)`, `SPLITS`, the 239 multi-TF bar features (2012–2026, 7 majors).
- `min1_production.py` — `wc_ret`, `boot`, `nonoverlap_chrono`, `mk_lgb`, the `Min1Strategy` load/predict pattern.
- Per-horizon production scripts (`m{5,10,15,30}_production.py`, `m_tick_prod.py`) — load/blend/gate templates.
- `manifest.py` — freeze a surviving model into the registry (see step 6).
- Interpreter: `~/binary-algo-venv/bin/python` (uv venv; lgb 4.6.0 / xgb 3.2.0 / cat 1.2.10 / pandas 2.3.3).

## 5. Workflow
1. **Scope** the key(s): currency, timeframe, side. Set `MX_HOR`/`HS` accordingly.
2. **Build label + features** with `wc_ret` (settlement) on the target pair's own clock; sanity-check moved
   up-rate ∈ [0.47,0.53].
3. **Pre-register the falsifier** (kill conditions) in a `<name>_result.json` stub.
4. **Fit on TRAIN, select on VAL worst-half**, evaluate per-year on held-out with `nonoverlap_chrono` +
   `boot` CI95, moved-bars only.
5. **Apply the falsifier.** Write the verdict (KILLED/SURVIVED + reasons + numbers) into the result JSON.
   If borderline/important, run the faithful CPCV (`min15_cpcv.py` pattern) before believing it.
6. **If it SURVIVES and is a deliverable book:** freeze it via `manifest.py` →
   `manifest.build(book_id="<PAIR>.<book>.v1", ...)` then `manifest.freeze(...)` → copies artifacts into
   `books/<id>/` + writes the manifest; `git tag book/<id>`; record env/data fingerprint automatically.
7. **Record the result** in `<PAIR>_RESULTS.md`:
   - add a row to that timeframe's combined-book experiments table (method, file, per-year stats + CI, verdict);
   - if side-split was run, update that timeframe's **Key results** AND the **MASTER KEY TABLE** (method +
     best OOS + frozen book id); unseat a key's leader only if the new method beats the incumbent's
     binding-year stat with CI95-lower clearing it under this discipline;
   - magnitude (|ret|) is sign-invariant → no up/down key; record it in `MAGNITUDE_FINDINGS.md`.
8. **Commit + push** with a message stating the verdict + the Tier-1 evidence (result JSON keys). Update
   `METHODS_CATALOG.md` if it's a new method or a method's status changed.

## 6. Evidence standard
Every number you write must trace to an on-disk result JSON or a research-log line (Tier-1) — never an
agent's prose summary. Flag thin-coverage (n<50) and any VAL-acc-max number as non-robust. If you cannot
cite it, you have not verified it.

## 7. Bootstrapping a NEW currency
1. Confirm the feature/tick data exists for the pair (`features/<PAIR>_<year>.parquet`,
   `features_tick_<PAIR>/`); if not, that's a data-acquisition prerequisite, not a modeling task.
2. Copy `EURUSD_RESULTS.md` → `<PAIR>_RESULTS.md`; reset all keys to UNTESTED; keep the structure and the
   maintenance protocol.
3. Retarget methods via `MX_HOR`/`HS` and the per-pair artifact convention (scripts are PAIR-parameterized).
4. Proceed through the workflow per key; freeze surviving books as `<PAIR>.<book>.v1`.

## Honest prior (do not waste compute relitigating)
60s EURUSD direction is near-efficient (~0.50–0.51 AUC across ~24 channels); >0.65 OOS-stable is not
achievable on this data at ≤5m. Real edges found: 15m direction ~0.58 robust / 0.647 recent, seconds (1–5s)
~0.65 (needs a tick venue), and **magnitude** AUC 0.71–0.81 (the one CPCV-deflation-certified edge). For a
new timeframe/currency, expect most direction levers to be null and design the falsifier to detect that fast.
