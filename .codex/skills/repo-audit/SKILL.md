---
name: repo-audit
description: >-
  Audit already-shipped binary-algo state for internal consistency: agent
  instructions and skills against the tree, documentation against code,
  generated ledgers against result JSON, the book registry against tags and
  manifests, and stale or moved references. Use for a repository health check,
  "are the docs still true", "are the skills current", a stale-reference
  search, or cross-record reconciliation. Returns a read-only findings report
  and changes nothing.
---

# Repository audit

Follow `AGENTS.md` and `docs/_EVIDENCE-FIRST.md`. An audit reads; it never edits the checkout,
ledgers, books, data, logs, or checkpoints, and it keeps protected outcomes sealed. Corrections it
recommends are separate work for `dev-cycle` or `strategy-eval`.

## 1. Fix the question and the baseline

State the audit question, the paths or keys in scope, the baseline commit (`git rev-parse HEAD`),
and the history window. Say what is out of scope.

## 2. Gather the controlling authority for each claim

- Layout and reference claims: the relevant `README.md` and `REPO_MAP.md` sections.
- Behavior claims: the owning implementation, its call sites, and its focused tests.
- Measured claims: the key's result JSON and results ledger; the sweep ledger for experiment
  state; `books/INDEX.json`, `MODEL_REGISTRY.md`, and the manifest for freeze and mapping claims.
- Instruction claims: `AGENTS.md`, every `SKILL.md` under `.codex/skills/`, and
  `scripts/check_agent_contracts.py`.
- History: read the commits and patches in order. Open the current authority before using a
  commit message to explain a change.

## 3. Check for divergence

1. **Named surfaces.** For every path, script, symbol, command, unit, issue number, and number an
   instruction or document names, check that it exists where stated and says what is claimed. Mark
   each `ok`, `moved`, `missing`, `contradicted`, or `unverifiable`.
2. **Moves and renames.** Search the old and the new name after a known move.
3. **Record agreement.** Compare the relevant records on key, status, result identity, and book
   identity. A generic document must agree with the key-specific record it cites.
4. **Tags.** A `book/<id>` tag must resolve to a commit whose tree contains that book; compare it
   with the commit that completed the book, which `git-workflow` shows how to find.
5. **Coverage of real work.** Compare recent history with the instructions: name the kinds of work
   that happen but that no instruction routes or describes.
6. **Set claims.** Where a statement covers several keys, books, or files, check each member and
   record each result.
7. Tell an intentional exception from drift by its governing record: an archived result, a legacy
   file that carries a scope banner, and a superseded build are not defects.

Prefer existing read-only verification commands when a claim needs runtime evidence. Record what
cannot be resolved as `Checked / Showed / Unknown / Needed`.

## 4. Report

State the baseline, scope, and coverage achieved. Lead with discrepancies that can change a
decision or a result, then stale references with narrower effect. For each discrepancy give both
citations, the controlling source, the consequence, the earliest commit that introduced it when
verified, and the smallest correction the controlling source supports. List verified intentional
exceptions and unresolved gaps separately. `docs/CORRECTNESS_AUDIT.md` remains the record of
evaluation integrity; do not duplicate it.
