# /goal prompt templates — exhaustive + generative strategy search

Fill `<X>` (timeframe, e.g. `3`) and `<CURRENCY>` (e.g. `EURUSD`). Paste after `/goal`.

## Full prompt (recommended)
```
/goal Find the best <X>-minute UP/DOWN binary strategy for <CURRENCY>, OOS-verified — this is your goal, do
not stop until it's achieved or every avenue is exhausted.

Follow the `strategy-eval` skill in EXHAUSTIVE SWEEP MODE. Work through SWEEP_MATRIX.md variant by variant
(Tier A→F), retargeting EVERY method/ensemble/RL/DQN/state-space/deep/microstructure/magnitude permutation to
<X> minutes via the catalog knobs (MX_HOR/HS). Reuse the frozen books in MODEL_REGISTRY.md / books/ as parents
and baselines.

ALSO continuously SEARCH FOR AND INVENT new methods, combinations, and variants to ADD to the menu — scour
scholarly literature (arXiv q-fin/stat.ML, SSRN), cross-disciplinary fields (econophysics, information theory,
point processes, causal discovery), and novel combinations of existing methods. Vet each (genuinely new,
plausible DIRECTION mechanism that survives sign-invariance, data available), append it to SWEEP_MATRIX.md
Tier-N + IDEAS_LOG.md, and test it. Use sub-agents for this research. The point is to maximize the chance of
finding an edge — keep expanding the search, don't just drain the fixed list.

For EVERY candidate: pre-register a falsifier; evaluate with the MANDATORY deriv-faithful discipline (wc_ret
ties-LOSE, nonoverlap_chrono, per-year 2024/2025/2026 with bootstrap CI95, selection on worst-VAL-half NOT
VAL-acc-max, moved-bars-only with up-rate∈[0.47,0.53] tripwire); score COMBINED + UP-split + DOWN-split;
record into <CURRENCY>_RESULTS.md (and MAGNITUDE_FINDINGS.md for magnitude); maintain the resumable ledger
sweeps/<CURRENCY>_<X>m.md and the UP/DOWN leaderboard; freeze every survivor as a <CURRENCY>.<book>.v1 book
via manifest.py and git-tag it.

Discipline: one heavy job at a time; commit the ledger + results frequently (the data lives on a flaky
external drive); be EVIDENCE-FIRST — every number traces to a result JSON, never prose; flag thin-coverage and
VAL-acc-max as non-robust; watch the 7 leakage traps in the skill. Always think before choosing the next
permutation. Don't give up.
```

## Short prompt
```
/goal Best <X>-minute UP/DOWN strategy for <CURRENCY>, OOS-verified. Use the strategy-eval skill in sweep
mode: run every SWEEP_MATRIX.md permutation at <X>m AND research/invent new ones to append, all under the
deriv-faithful discipline. Record combined+up+down into <CURRENCY>_RESULTS.md, keep the resumable
sweeps/<CURRENCY>_<X>m.md ledger + leaderboard, freeze survivors as books. Evidence-first; one heavy job at a
time; don't give up.
```

## Notes
- The skill (`.claude/skills/strategy-eval/SKILL.md`) is auto-discovered; the agent will invoke it. The prompt
  just sets the objective + scope.
- For a NEW currency with no data on disk, the first step is data acquisition (a prerequisite the skill flags),
  not modeling.
- To resume an interrupted sweep, the same prompt works — the ledger is the state; the agent picks up the
  first `pending` row.
