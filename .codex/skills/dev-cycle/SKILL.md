---
name: dev-cycle
description: >-
  Implement and ship one binary-algo code, infrastructure, tooling,
  agent-skill, or documentation change through analysis, current plan review,
  implementation, focused verification, adversarial review, final local gate,
  commit, and push to main. Use for "dev cycle", "implement", "ship", "next
  phase", "add the harness/module/script", "do the reorg", or work from a
  GitHub issue that must end on main. If the deliverable is measured strategy
  evidence—evaluation, sweep, retarget, freeze, or result recording—use
  strategy-eval instead.
---

# Development Cycle

Run one complete iteration:

`analyse -> plan/review -> implement -> test -> clean/archive -> review/fix -> final gate -> commit -> sync/re-gate -> push -> close out`

This repository ships directly to `main` with linear history, a single checkout, and local evidence
gates. Do not invent a pull-request, worktree, or CI workflow. Preserve the user's unrelated files
and changes.

Use the host runtime's equivalent primitives and native subagents when available. Do not require a
particular model, vendor, or nested agent process. Final summaries must disclose every shortcut,
skipped check, temporary workaround, and declined review finding; the expected default is `none`.

## GitHub issue authority

Treat `#<number>` as `sppburke/binary-algo#<number>`. Fetch title, body, comments, state, labels,
`updatedAt`, and URL before deciding scope. The consolidated body is the implementation authority;
comments supply history and review receipts, not contradictory live requirements.

After a successful push, comment with shipped behavior, commit SHA, checks, review outcomes, and
shortcuts/skips, then close the driving child issue only when `main` contains the commit and all
acceptance criteria hold. When the plan names a parent issue or machine projection, update and
verify both before declaring the child complete.

## Change classification and reading

Classify the intended diff; mixed work takes the union of gates:

- **code** — shared/runtime/experiment-support Python or shell behavior;
- **books/results** — measured result JSONs, Tier-2 ledgers, or frozen books;
- **docs** — prose or filing-only truth;
- **tooling** — operator scripts, environment/dependency files, or local automation;
- **agent-instructions/skills** — `AGENTS.md`, `CLAUDE.md`, `.codex/skills/`,
  `.claude/skills/`, or their validator.

Read `AGENTS.md` and `docs/_EVIDENCE-FIRST.md`, then route additional reading:

- layout/filing changes: relevant `README.md` and `REPO_MAP.md` sections;
- books/results: `README.md`, `REPO_MAP.md`, `MODEL_REGISTRY.md`, `books/INDEX.json`,
  `strategy-eval`, and applicable result/sweep records;
- code/infrastructure: owning implementation, call sites, focused tests, and task contracts;
- roadmap-linked work: authoritative parent/child issues and the named machine projection;
- agent/skill work: complete selected skills, their required references, interface metadata, and
  `scripts/check_agent_contracts.py`.

Do not load model registries or generic findings for unrelated docs/tooling/skill changes.

## 1. Analyse and bound authority

From the repository root:

```bash
git fetch origin main
git status --short
git branch --show-current
git rev-parse HEAD origin/main
git log --oneline -12
```

Require `main` and account for every existing worktree change. Never overwrite or stage unrelated
work. Read recent diffs touching the semantic owner. Prove the requested change is not already
shipped and identify the nearest reusable implementation and precedent.

Resolve an ambiguous "next" in this order: named driving issue, named parent roadmap, named machine
registry/backlog, then current git history. Never auto-promote a deferred trigger into authorized
work. If a roadmap says a new child is required, file that bounded child before implementation.

Define the intended behavior, authority, non-goals, acceptance and rejection conditions, expected
and protected files, and a **surface budget** covering public APIs, dependencies, persistent
schemas, and protected surfaces. A material budget expansion requires a plan delta and review.

## 2. Establish a current reviewed plan

Outline product/research behavior and the end-to-end implementation flow. Compare it with the
strongest simpler repository-native design and justify every new durable surface. Ask only when a
user-owned choice changes behavior, authority, protected access, persistence, contract, lifecycle,
accepted risk, or issue closure; otherwise use and state the repository-native default.

Before editing, establish `plan-review` coverage for the exact consolidated artifact and current
applicable baseline when work is driven by a material specification/roadmap or when architecture,
authority, contract, risk, or acceptance is not already clear. For a contained implementation with
no material specification, perform a direct evidence preflight against intent and repository
precedent; do not manufacture an issue, fingerprint, or independent review ceremony.

When `plan-review` coverage is required:

- match the artifact SHA-256 and reviewed repository HEAD from a valid receipt when available;
- refresh the issue/parent body, `origin/main`, owning code, and relevant authority;
- reuse coverage only when intervening changes do not affect a material premise, owner, touchpoint,
  precedent, contract, risk, or verification input;
- run a fresh review when proof is missing, the artifact materially changed, or relevant repository
  state changed;
- do not force a rereview for proven formatting-only or irrelevant changes.

Apply verified findings in one batch. Review localized corrections against their delta and affected
integration invariants; fully rereview material scope, architecture, behavior, authority, contract,
or acceptance changes. A caller-launched review never recursively launches another reviewer.

The user's explicit request to implement or ship authorizes proceeding once the plan is current and
no material user-owned choice remains unresolved.

## 3. Implement the bounded change

Edit only authorized paths in the single checkout. Follow the current semantic owner and neighboring
style. Search before adding any helper, module, script, schema, dependency, or documentation
authority. Keep strategy evaluation and measured-result decisions in `strategy-eval`.

Repository rules:

- Run Python from the repository root with `~/binary-algo-venv/bin/python scripts/<name>.py`.
- Keep shared modules flat under `scripts/` when existing bare imports require that layout.
- Run heavy fits one at a time; make long operations idempotent/resumable.
- Do not move root feature/data directories or `macro_calendar.parquet`.
- Use fail-closed sentinels for deferred behavior and state non-obvious call-order/clock
  preconditions.
- Keep issue-#9, active books/mappings, protected outcomes, and other protected surfaces unchanged
  when the plan names them as protected.

For linked roadmap work, update the tracked machine projection and parent-facing facts only as the
approved plan requires. Never delete a deferred capability or promote a fired trigger without a
separate decision.

## 4. Run preliminary focused verification

Write observable PASS/FAIL criteria before running checks. Build the gate from changed behavior and
blast radius:

- **code:** run existing focused tests for the changed owner, then compile/lint/smoke the affected
  path as applicable; exercise one real downstream use case for shared behavior;
- **books/results:** use the complete `strategy-eval` discipline and certification gate;
- **docs:** resolve changed paths/links and verify factual assertions against their authority;
- **tooling:** run syntax checks plus the exact operator command changed;
- **agent-instructions/skills:** run the canonical skill validator, validate frontmatter/interface
  metadata, exercise trigger/routing scenarios, and inspect resolved projection bytes.

For persistence, concurrency, protected access, security, identity/immutability, authority, or crash
recovery changes, require focused negative/adversarial tests for the named `DO NOT SHIP` conditions.
When the plan names expected/protected files, audit the exact changed-path allowlist and protected
hashes.

If claiming a failure is pre-existing, reproduce it on the untouched base revision or otherwise
provide direct evidence. A passing check proves only the failure modes it covers.

## 5. Clean and archive before review

Move intended result JSONs to `results/json/` and logs/checkpoints to their canonical ignored
locations. Remove only artifacts created by this cycle. Update required ledgers, registries,
documentation, parent projections, and issue-bound tracked state before review. Confirm the full
intended diff fits the approved surface budget and no unrelated path changed.

Do not defer tracked cleanup until after push.

## 6. Run adversarial implementation review

Review the complete intended diff after preliminary checks and cleanup. For contained docs/skills or
code, use one fresh native read-only reviewer when available; otherwise apply the same rubric
directly and disclose the fallback. An independent reviewer is mandatory for material protected
data, persistence, concurrency, security/authority, migration, or broad shared-infrastructure work.

Every review reports, in order:

1. **Reuse / precedent** — correct semantic owner and repository pattern;
2. **Simpler counterfactual** — strongest lower-complexity complete alternative;
3. **Complexity** — `under-engineered | minimum-complete | over-engineered`;
4. **Cleanliness** — ownership, duplication, seams, and unnecessary surface;
5. evidence-backed findings with claim, falsifier/reproduction, shipped impact, and minimum fix.

Verify every finding against the actual diff. Record `fixed | declined | deferred` with evidence and
reason. Fix all verified Blocking findings and rerun affected checks. After localized fixes, review
only the unreviewed delta plus affected integration invariants. Restart full review only when a fix
materially changes scope, architecture, authority, or a shared contract.

## 7. Run the final local evidence gate

The final applicable gate runs after the last edit and immediately before commit. It includes:

- every focused check whose input changed;
- end-to-end use-case verification for observable behavior;
- `git diff --check`;
- exact changed-path/surface-budget review;
- protected-file/hash audit when required;
- for agent/skill changes, `~/binary-algo-venv/bin/python scripts/check_agent_contracts.py`, skill
  frontmatter validation, Git mode/link-target audit, and platform-neutrality scan.

If any file changes after this gate, rerun the affected review and the full applicable final gate.
Never loosen an assertion merely to obtain a pass.

## 8. Commit, sync, and push main

Stage only intended paths and commit with an imperative summary of at most 72 characters. Include
the scope tag and why in the body; books/results commits also cite the result-JSON evidence and
verdict.

Fetch `origin/main` again. If it moved, use a linear-history-preserving rebase or equivalent
non-merge integration without force, inspect the combined diff, and rerun every gate whose inputs
changed. Rerun adversarial review when the sync materially changes scope, architecture, authority,
contract, or reviewed integration behavior. Complete any required conflict resolution, then push.
Never force-push `main`.

Confirm the remote contains the intended reviewed/verified commit and the worktree is clean. A
changed post-review HEAD invalidates affected evidence; do not substitute an older receipt.

After push, update/close the driving issue as described above. For named parent/child continuity,
verify the parent authority and machine projection state agree with the shipped commit before
closing the child.

## 9. Report completion

Return:

- pushed commit SHA and shipped behavior;
- scope classification;
- checks and exact outcomes;
- review findings grouped as fixed and declined/deferred with reasons;
- `Shortcuts / hacks taken: none` or the complete list;
- skipped checks and why;
- issue/parent/projection status when applicable;
- clean-worktree and artifact-archive confirmation.

## Failure handling

- Import failure: confirm the command ran from repository root through the pinned environment.
- Test/review failure: fix the bounded cause, rerun affected checks/review, then rerun the final gate.
- Non-fast-forward push: fetch, integrate, re-review/re-gate affected inputs, and retry without force.
- Unavailable GitHub authority: continue only disjoint read-only/local work; do not close, retarget,
  or claim completion until current remote state is verified.
- Unrecoverable material evidence gap or user-owned decision: preserve work, report
  `Checked / Showed / Unknown / Needed`, and stop before the affected mutation or push.
