---
name: strategy-eval
description: >
  Evaluate or reverse-engineer a binary up/down direction (or magnitude) strategy in THIS repo
  (/media/sean/CORSAIR/binary-algo) and record the result correctly. Use whenever the task is to test a
  model/methodology at a (currency, timeframe, side); retarget an existing method to a new horizon or
  currency; reference/load/freeze a model from the registry (books/); or write a result into a
  <PAIR>_RESULTS.md. Enforces the deriv-faithful evaluation discipline and the unique-key results schema so
  results are trustworthy and comparable. Also drives the EXHAUSTIVE + GENERATIVE SWEEP (§8): work through every
  permutation in SWEEP_MATRIX.md variant by variant AND research/invent new methods to append, to find the best
  UP and DOWN predictor for a (currency, timeframe). Triggers: "evaluate this strategy", "reverse engineer a
  model for <pair>/<timeframe>", "does <method> work at <horizon>", "best <X>-minute strategy for <currency>",
  "sweep all the models/methods", "try every permutation", "find new model ideas", "add a result to the
  ledger", "freeze a book".
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

### 0a. GENERIC ↔ KEY-SPECIFIC file convention (READ `REPO_MAP.md` — enforce it)
Every doc is exactly one tier; keep them clean as keys multiply:
- **GENERIC** (currency/timeframe-agnostic): `METHODS_CATALOG.md` (methods), `SWEEP_MATRIX.md` (permutation
  menu), `IDEAS_LOG.md` (transferable idea→experiment backlog + falsifier templates), `THEORY.md` (sign-invariance,
  direction ceiling, settlement), `GOAL_PROMPT.md`, `research/**`. **Never bake a single key's incumbents/numbers
  into these** — if you must cite a result as an example, tag it inline `[PAIR·tf]` and point to the Tier-2 file.
- **KEY-SPECIFIC** (named/labeled by the key — ALL incumbents/numbers/results/backlogs live here):
  `<PAIR>_RESULTS.md` (results of record + leaderboard), `sweeps/<PAIR>_<tf>.md` (sweep LEDGER / status),
  **`sweeps/<PAIR>_<tf>_backlog.md` (the per-key EXECUTABLE backlog — TOP-N experiments, FIRST-TO-RUN queue,
  incumbents-to-beat, discovery rounds)**, `MAGNITUDE_FINDINGS.md` (per-currency), `MODEL_REGISTRY.md`+`books/`.
- Legacy key-specific files keep their names (`research_log.md`, `m{5,10,30}_research_log.md`, `EXPERIMENT_LEDGER.md`,
  `FINDINGS.md`, `DIRECTION_FINDINGS.md`, `CCM_DESIGN.md`) but each carries a `SCOPE:` banner declaring its key —
  don't rename (dense cross-refs into production scripts); the banner is the label.
- **When you run a sweep/idea on a key: record the idea generically once (IDEAS_LOG/SWEEP_MATRIX/METHODS_CATALOG),
  and the result ONLY in the key's Tier-2 files.** Every file opens with a 1-line `SCOPE:` banner — keep it.

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

## 8. EXHAUSTIVE SWEEP MODE (find the BEST strategy for a (currency, timeframe))
Use this when the task is "find the best <X>-minute up/down strategy for <CURRENCY>" — a systematic search
over EVERY method/ensemble/RL/DQN/etc permutation, not a single candidate. Driver = `SWEEP_MATRIX.md` (the
permutation menu). The goal is to converge on the best UP and the best DOWN predictor, with every result
recorded and the search resumable + exhaustive.

**Procedure:**
1. **Open a sweep ledger** `sweeps/<PAIR>_<tf>.md`. If it exists, resume from it (do NOT restart). If not,
   create it by instantiating `SWEEP_MATRIX.md` for this (currency, timeframe): expand each method's variant
   axes (`MX_HOR`/`HS` + the knob ranges) into individual rows, ordered Tier A → F (highest ROI first), each
   row `status: pending` with columns: id, family, method, variant, script, target, prior, status,
   combined_oos, up_oos, down_oos, verdict, result_json.
2. **Work rows top-to-bottom.** For each `pending` row:
   a. Pre-register its falsifier (cheap fast-KILL for low-prior rows).
   b. Retarget + run the §2 discipline; evaluate **combined + UP-split + DOWN-split** per-year with CI95.
   c. Write the row's result into `<PAIR>_RESULTS.md` (combined-book table + per-key) and the result JSON.
   d. Mark the ledger row `done` (or `killed`) with its numbers + JSON path.
   e. Update the **UP/DOWN leaderboard** in `<PAIR>_RESULTS.md` — unseat a side-leader only if it beats the
      incumbent's binding (worst held-out) year with CI95-lower clearing it under discipline.
   f. Commit (the ledger + results) so progress survives interruption / a drive drop.
3. **One heavy job at a time** (OOM history); use sub-agents for research (literature, new method variants)
   and for parallel evaluation of independent rows, but serialize the heavy fits.
4. **IMPROVE every edge you find (Tier-I levers + COMBINATIONS).** A certified book is the START, not the end.
   Run the SWEEP_MATRIX **Tier-I edge-improvement levers** — adaptive-conformal (ACI) gate, seed-ensemble ⊕ GBM
   stack, |return|-weighted/GMADL loss, calibration, cross-pair pooling, AdamW/Optuna tuning — ON the certified
   book, and evaluate their **COMBINATIONS** (the model space is a cross-product: {base · cross-pair · stack ·
   seed-ensemble · pooled} × {fixed · ACI · calibrated gate} × {BCE · GMADL loss} × {filter · specialist}). Each
   improvement gets a pre-registered falsifier (beat the incumbent's binding-year stat and/or lift the CPCV
   path-clear-rate). Freeze a winning combination as its own book (e.g. `<PAIR>.<book>_aci.v1`).
5. **Incumbent = the BEST COMBINATION already found, NOT the old base model.** When you benchmark a new method or
   retarget to a neighboring timeframe (e.g. 10m after 5m), the number to beat is the best combination in
   MODEL_REGISTRY.md / `books/INDEX.json` (e.g. cross-pair book + ACI gate), and you must retarget the Tier-I
   levers + the certified combinations to the new key and compare against **ALL** of them — never just the
   "previous" base books. The DL/improvement lit-review + saved papers live in
   `/home/sean/git/academic-papers/` (`_DL_for_5m_FX_direction_REVIEW.md`); grow that corpus when you research.
   **MINE IT EXHAUSTIVELY:** read every paper thoroughly (every word/equation/diagram/reference), extract every
   testable lever (logic/math/loss/arch/gating/labeling/validation/framing), and EXPERIMENT on all of them —
   better DISPROVED BY EXPERIMENT than never tried; never pre-dismiss on a hunch. Fan out reader sub-agents over
   the corpus → idea→experiment backlog in `IDEAS_LOG.md` → execute.
6. **Do not stop** until every row is `done`/`killed`, the Tier-I levers + combinations are exhausted on the
   best edge, and the discovery loop is dry. Then write the final best-UP and best-DOWN for the key, freeze the
   survivors (including the best combination) as books (§6), and report the leaderboard.
5. **Resumability:** the ledger IS the state. On any resume (new session, after a crash), re-open it and
   continue from the first `pending`/`running` row. Never repeat a `done` row.

**6. DISCOVERY — generate NEW methods/variants to expand the search (run continuously, don't just drain the menu).**
The fixed menu is a starting point; the goal is to maximize the chance of finding an edge, so keep ADDING
candidates:
   a. **When to discover:** at sweep start, after finishing each tier, and whenever `pending` rows run low —
      spawn research sub-agents (the program endorses this) to find genuinely new ideas.
   b. **Where to look:** (i) scholarly literature — arXiv (q-fin.TR/ST, stat.ML), SSRN, journals — for new FX
      microstructure / time-series / ML / RL / distributional methods; (ii) **cross-disciplinary** transfer —
      physics (econophysics, statistical mechanics, turbulence), info theory (transfer entropy, directed
      information), neuroscience/signal-processing (state-space, point processes, Hawkes), causal discovery;
      (iii) **novel COMBINATIONS** of existing methods (e.g. HMM-regime-gated CCM, magnitude-conditioned
      cross-horizon stack, RL sizing on the 15m book, online-adaptive meta-labeler) — combinations are often
      the cheapest novelty; (iv) re-read `IDEAS_LOG.md` + `EXPERIMENT_BACKLOG.md` for already-logged-but-untried
      ideas.
   c. **Vet before adding** (cheap filter, avoid junk rows): is it genuinely NEW (not already in
      `METHODS_CATALOG.md`/the ledger)? Is there a plausible MECHANISM by which it carries *direction* (sign),
      not just magnitude — i.e. does it survive the sign-invariance theorem, or is it really a magnitude/gate
      idea? What data does it need (on-disk vs acquisition)? Assign a prior + a fast-KILL falsifier.
   d. **Append** each vetted candidate as a new row in `SWEEP_MATRIX.md` (the "Tier N — discovered" section)
      AND the sweep ledger, and log it in `IDEAS_LOG.md` with its source citation + mechanism + prior. Then it
      gets run like any other row.
   e. **Loop-until-dry:** keep a discovery round going until it yields K consecutive rounds (e.g. 2) with no
      novel survivable idea; then the search space is saturated for now. The leaderboard always reflects the
      current best — discovery only ever ADDS chances to beat it.

**Coverage rule:** run every Tier-A–F method at least once at this (currency, timeframe) even if it was null
at another horizon — nulls are horizon/regime-specific and confirming them here is the point. But size the
falsifier to the prior so low-prior rows die fast. Log any variant you prune (don't silently skip).

## Honest prior (do not waste compute relitigating)
60s EURUSD direction is near-efficient (~0.50–0.51 AUC across ~24 channels); >0.65 OOS-stable is not
achievable on this data at ≤5m. Real edges found: 15m direction ~0.58 robust / 0.647 recent, seconds (1–5s)
~0.65 (needs a tick venue), and **magnitude** AUC 0.71–0.81 (the one CPCV-deflation-certified edge). For a
new timeframe/currency, expect most direction levers to be null and design the falsifier to detect that fast.
