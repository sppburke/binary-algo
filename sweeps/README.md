# sweeps/ — per-(currency, timeframe) exhaustive-search ledgers

One ledger file per active search: `sweeps/<PAIR>_<tf>.md` (e.g. `sweeps/EURUSD_3m.md`, `sweeps/GBPUSD_15m.md`).
The ledger IS the state of an exhaustive strategy sweep — it makes the search **resumable** (survives session
end / drive drop) and **exhaustive** (never repeat or skip a permutation). Driven by the `strategy-eval` skill
(§8 sweep mode) off `SWEEP_MATRIX.md`. Final results land in `<PAIR>_RESULTS.md`; new ideas in `IDEAS_LOG.md`.

## Ledger format
Front-matter: `currency`, `timeframe`, `started`, `target` (e.g. ">0.65 OOS-stable UP or DOWN"), `status`.
Then one table, one row per **method × variant** instantiated from `SWEEP_MATRIX.md` (expand the variant axes):

| id | tier | family/method | variant (knobs) | script | target | prior | status | combined_oos | up_oos | down_oos | verdict | result_json |
|----|------|---------------|-----------------|--------|--------|-------|--------|--------------|--------|----------|---------|-------------|
| A1-a | A | GBM ensemble | nl255,nest3000,lr.02 | m15_production.py | D | med | pending | | | | | |
| ... | | | | | | | | | | | | |
| N1 | N | (discovered) transfer-entropy gate | E=4,tp5s | (new) | G | low | pending | | | | | |

`status` ∈ {pending, running, done, killed, pruned}. On `done`/`killed`, fill the OOS columns + verdict +
result_json path, and reflect it in `<PAIR>_RESULTS.md` + the UP/DOWN leaderboard. `pruned` = a dominated
variant skipped on purpose (log why in the row).

## Loop (per the skill)
Work rows top-to-bottom (Tier A→F→N). For each `pending`: pre-register falsifier → run the deriv-faithful
discipline → score combined + UP-split + DOWN-split per-year CI95 → write results → mark done/killed → update
leaderboard → commit. Periodically run DISCOVERY (skill §8.6): research + invent new candidates, vet them,
append as Tier-N rows here and in `SWEEP_MATRIX.md`. Done only when every row is done/killed AND discovery is
dry (K rounds, no novel survivable idea) — or the target is hit.

## Rules
- One heavy fit at a time (OOM history). Sub-agents for research + parallel eval of independent rows.
- Commit the ledger + results frequently (drive is a flaky external SSD).
- Every number Tier-1 (a result JSON), never prose. Flag thin-coverage (n<50) / VAL-acc-max as non-robust.
