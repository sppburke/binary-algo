---
name: dev-cycle
description: >-
  Implement and ship one binary-algo code, infrastructure, tooling,
  agent-skill, or documentation change through analysis, current plan review,
  implementation, focused verification, adversarial review, final local gate,
  commit, and push to main. Use for "dev cycle", "implement", "ship", "next
  phase", "add the harness/module/script", "do the reorg", or work from a
  GitHub issue, lab specification, or program record that must end on main. If
  the deliverable is measured strategy evidence—evaluation, sweep, retarget,
  freeze, or result recording—use strategy-eval instead. Operating the Deriv
  runtime or the research lab is not a dev-cycle; changing their code is.
---

# Development Cycle

Run one complete iteration:

`analyse -> plan/review -> implement -> test -> clean/archive -> review/fix -> final gate -> commit -> sync/re-gate -> push -> close out`

This repository ships directly to `main` with linear history, from the primary checkout, on local
evidence gates. Do not invent a pull-request, worktree, or CI workflow. Preserve the user's
unrelated files and changes.

The default endpoint is the pushed commit on `main`, however small the change. An explicit stop
boundary from the user ("do not push", "stop before commit", "read-only") pauses the cycle at that
point: preserve the work and its evidence, report exactly what is and is not done, and before
resuming refresh the base, the governing authority, the review, and every gate. Before any tracked
edit, complete the authority and other-writer checks in sections 1 and 2.

Use the host runtime's equivalent primitives and native subagents when available. Do not require a
particular model, vendor, or nested agent process. Final summaries must disclose every shortcut,
skipped check, temporary workaround, and declined review finding; the expected default is `none`.

## What this cycle owns

- Every hand-made change to a tracked file runs this cycle, including one-line fixes and wording
  sweeps made in the middle of a conversation. For a result-only delivery (a result JSON, its
  ledgers, a frozen book and its registry rows), `strategy-eval` owns the measurement, the verdict,
  the certification, and what the records say; enter this cycle before the first tracked write, for
  its authority and writer checks, review, final gate, push verification, and closeout.
- The lab's reviewed recorder commits batches of result files and their regenerated ledger blocks;
  the reviewed scheduled probe fold commits its one result file. Each runs under its own contract.
  Such a batch is not a development cycle. Changing either tool, the ledger renderer, the
  certificate reader, or what may be recorded is.
- Starting, stopping, or deploying runtime services belongs to `deriv-runtime-ops`. Running the
  research lab belongs to `deriv-lab-ops`. Git and GitHub commands live in `git-workflow`.

## Governing authority

The change is governed by a driving GitHub issue, by an owner-approved lab specification or
program record, or, for a contained change, by the user's request. Treat `#<number>` as
`sppburke/binary-algo#<number>` and read it with `git-workflow`: title, body, comments, state,
labels, `updatedAt`, and URL, before deciding scope. The consolidated body is the implementation
authority; comments supply history and review receipts, not contradictory live requirements. For a
local specification, record its path, the SHA-256 of the file, its review fingerprint, and the
owner approval that covers those bytes (a dated decision in the program log). For a backlog entry,
the user's filing request is the authority and the reviewed entry's hash governs the text inserted.

After a successful push, comment with shipped behavior, commit SHA, checks, review outcomes, and
shortcuts/skips, then close the driving child issue only when `main` contains the commit and all
acceptance criteria hold. Update and verify every parent record and machine projection the plan
names before declaring the child complete. When a program record governs instead of an issue,
record the same facts in the program log, through `deriv-lab-ops`.

## Change classification and reading

Classify the intended diff; mixed work takes the union of gates:

- **code** — shared/runtime/experiment-support Python or shell behavior;
- **books/results** — measured result JSONs, Tier-2 ledgers, or frozen books that travel with
  implementation work; a result-only delivery is `strategy-eval`;
- **docs** — prose or filing-only truth;
- **tooling** — operator scripts, environment/dependency files, or local automation;
- **agent-instructions/skills** — `AGENTS.md`, `CLAUDE.md`, `.codex/skills/`,
  `.claude/skills/`, or their validator.

Classify a configuration file, specification, manifest, or registry by the consumers it can
change, never as docs. A changed executable example is verified by running it or by its owning
gate.

Read `AGENTS.md` and `docs/_EVIDENCE-FIRST.md`, then route additional reading:

- layout/filing changes: relevant `README.md` and `REPO_MAP.md` sections;
- books/results: `README.md`, `REPO_MAP.md`, `MODEL_REGISTRY.md`, `books/INDEX.json`,
  `strategy-eval`, and applicable result/sweep records;
- code/infrastructure: owning implementation, call sites, focused tests, and task contracts;
- Deriv runtime code or `ops/` units: `deriv-runtime-ops` and the runbook it names;
- lab tools, the ledger renderer, or program records: `deriv-lab-ops` and
  `docs/DERIV_SOTA_PROGRAM.md`;
- roadmap-linked work: the authoritative issue or program record and the machine projection it
  names;
- agent/skill work: complete selected skills, their required references, interface metadata, and
  `scripts/check_agent_contracts.py`.

Do not load model registries or generic findings for unrelated docs/tooling/skill changes.

## 1. Analyse and bound authority

From the repository root, take the `git-workflow` inventory:

```bash
git --no-optional-locks status --short --branch --untracked-files=all
git fetch origin main
git worktree list
git rev-parse HEAD origin/main
git log --oneline -12
```

Require `main`. Record `HEAD` and every staged, unstaged, and untracked path before editing, and
identify what this cycle will own. Never overwrite or stage unrelated work. Before the first edit,
check for other writers of this checkout as `git-workflow` describes, and record each writer's
state. Stop a running result recorder, by `deriv-lab-ops`'s procedure, only immediately before the
first edit and after the plan and authority checks, because an uncommitted change or a stray file
makes it refuse and exit. A recorder frozen under an owner pause is left exactly as it is. Nothing
may be staged while the scheduled probe fold can fire. If a writer's state cannot be established,
stop before editing. Leave other worktrees and their
branches alone. Read recent diffs touching the
semantic owner. Prove the requested change is not already shipped and identify the nearest
reusable implementation and precedent.

Resolve an ambiguous "next" in this order: named driving issue, named parent roadmap or program
record, named machine registry/backlog, then current git history. Never auto-promote a deferred
trigger into authorized work. If a roadmap says a new child is required, have `feature-dev` draft,
review, and file that bounded child, then resume from its receipt.

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

- match the artifact SHA-256 and reviewed repository HEAD from a valid receipt when available: one
  that the review workflow recorded for this artifact, whose hash equals the artifact's current
  bytes, and that no later comment or decision withdraws;
- refresh each governing artifact, including a material parent and the projection it names,
  together with `origin/main`, owning code, and relevant authority;
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

Edit only authorized paths in the primary checkout. Follow the current semantic owner and
neighboring style. Search before adding any helper, module, script, schema, dependency, or
documentation authority. Keep strategy evaluation and measured-result decisions in `strategy-eval`.

Repository rules:

- Run Python from the repository root with `~/binary-algo-venv/bin/python scripts/<name>.py`.
- Keep shared modules flat under `scripts/` when existing bare imports require that layout.
- Run one heavy fit at a time unless an owner decision in the governing program record authorizes
  bounded concurrency; make long operations idempotent/resumable.
- Do not move root feature/data directories or `macro_calendar.parquet`.
- Use fail-closed sentinels for deferred behavior and state non-obvious call-order/clock
  preconditions.
- Preserve every protected surface the plan names: issue state, files, outcomes, books, runtime
  mappings, and anything else it lists.
- Start a bug fix with a reproducer that fails on the unfixed code.
- Fix the evidenced defect class within the authorized scope, not one example. When one change
  applies to several keys, sides, releases, or input variants, verify each member's own state.
- Add a guard, fallback, retry, or default only for a binding contract or a failure demonstrated on
  this revision. Record an unproven one as declined, with the check that would prove it.
- Pin an approved new dependency in its owning record (`ENVIRONMENT_libs.txt` for the research
  stack, `requirements-deriv-demo.txt` for the demo runtime) and verify the consumer that needs it.
- A change to the registry, a book, `live_features.py`, or `deriv_runtime_core.py` re-arms the
  runtime parity re-check; say so in the report. Shipping does not satisfy it.

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

For each required behavioral check record the command and directory, the revision and task state,
the input and its provenance, the expected output, and the observed result. Prove behavior through
the normal entry point and its real consumers on deterministic fixtures or recorded inputs; an exit
status alone is not proof. Capture each command's own exit status, never a pipeline's. Checks that
write `*_result.json` scratch output run where a stray file cannot disturb another writer.

For persistence, concurrency, protected access, security, identity/immutability, authority, or crash
recovery changes, require focused negative/adversarial tests for the named `DO NOT SHIP` conditions.
When the plan names expected/protected files, audit the exact changed-path allowlist and protected
hashes. Where they apply:

- a regression test must be seen to fail on the original behavior, for the expected reason, and
  pass on the fix;
- clock-dependent behavior is tested with a controllable clock or explicit timestamps across the
  affected boundary, including a timezone transition and a restart where relevant;
- a change to chunking or checkpoint continuation is compared against an uninterrupted run on the
  same small input.

If claiming a failure is pre-existing, reproduce it on the untouched base revision or otherwise
provide direct evidence.

## 5. Clean and archive before review

Move intended result JSONs to `results/json/` and logs/checkpoints to their canonical ignored
locations. Remove only artifacts created by this cycle. Stop, by its recorded identity, every
background job this cycle started, or record its authorized handoff and arrange to be told when
it ends. Update required ledgers,
registries, documentation, parent projections, and issue-bound tracked state before review.
Confirm the full intended diff fits the approved surface budget and no unrelated path changed.

Do not defer tracked cleanup until after push.

## 6. Run adversarial implementation review

Review the complete intended diff after preliminary checks and cleanup. For contained docs/skills or
code, use one fresh native read-only reviewer when available; otherwise apply the same rubric
directly and disclose the fallback. An independent reviewer is mandatory for material protected
data, persistence, concurrency, security/authority, migration, or broad shared-infrastructure work;
if one is mandatory and unavailable, stop before shipping.

Brief a reviewer with the exact diff, the intent, the acceptance criteria, the constraints, and
the underlying evidence. Label unproven premises as claims to attack and withhold your verdict.

Every review reports, in order:

1. **Reuse / precedent** — correct semantic owner and repository pattern;
2. **Simpler counterfactual** — strongest lower-complexity complete alternative;
3. **Complexity** — `under-engineered | minimum-complete | over-engineered`;
4. **Cleanliness** — ownership, duplication, seams, and unnecessary surface;
5. evidence-backed findings with claim, falsifier/reproduction, shipped impact, and minimum fix.

Verify every finding against the actual diff. Record `fixed | declined | deferred` with evidence and
reason; a finding is fixed only when its affected checks pass, and effort, size, or churn is never
a reason to decline. Fix all verified Blocking findings and rerun affected checks. After localized
fixes, review only the unreviewed delta plus affected integration invariants. Restart full review
only when a fix materially changes scope, architecture, authority, or a shared contract. Settle
reviewer disagreements against the diff and the evidence; if a material question stays open and the
reviewers are available, go back to them with the settled findings and only that question,
including the strongest case against the emerging answer.

## 7. Run the final local evidence gate

Stage the intended paths first (`git-workflow`), so that the gate, the staged audit, and the
commit all see the same bytes. The final applicable gate runs after the last edit and immediately
before commit. It includes:

- the staged-path and deletion audit from `git-workflow`;
- every focused check whose input changed;
- end-to-end use-case verification for observable behavior, or `N/A` with the reason;
- `git diff --check`;
- exact changed-path/surface-budget review;
- protected-file/hash audit when required;
- for agent/skill changes, `~/binary-algo-venv/bin/python scripts/check_agent_contracts.py`, skill
  frontmatter validation, Git mode/link-target audit, and platform-neutrality scan.

If any file changes after this gate, rerun the affected review and the full applicable final gate.
Never loosen an assertion merely to obtain a pass.

## 8. Commit, sync, and push main

Use `git-workflow` for the mechanics and the message form. Commit the staged bytes that passed the
gate.

Fetch `origin/main` again. If it moved, use a linear-history-preserving rebase or equivalent
non-merge integration without force. Resolve any conflict first, then inspect the combined diff and
rerun every gate whose inputs changed. Rerun adversarial review when the sync materially changes
scope, architecture, authority, contract, or reviewed integration behavior.

Immediately before each push attempt, read the governing authority again: the driving issue and
any material parent, or the local specification's hash and the latest applicable owner decision,
or, for a backlog entry, that the outgoing commit inserts the reviewed bytes at the approved
destination. A changed body, a withdrawn approval, or a new unresolved finding returns to the plan gate. Then
push. Never force-push `main`.

Confirm the remote contains the intended reviewed/verified commit by ancestry, as `git-workflow`
describes, including after a rejected or ambiguous push. Confirm the worktree holds nothing this
cycle created, and report any unrelated path that was there before or arrived meanwhile. A changed
post-review HEAD invalidates affected evidence; do not substitute an older receipt.

After the push is verified, or on any earlier exit, undo only a stop this cycle made: its stop file
and the recorder it stopped, by `deriv-lab-ops`'s procedure, and only when the checkout meets its
conditions and no pause has been ordered meanwhile; otherwise report the writer's state and why it
was not restored. Then
update/close the driving
issue as described above. For named parent/child continuity, verify the parent authority and
machine projection state agree with the shipped commit before closing the child.

## 9. Report completion

Lead with what now behaves differently and why, in plain language, then return:

- pushed commit SHA and shipped behavior;
- scope classification;
- checks and exact outcomes;
- review findings grouped as fixed and declined/deferred with reasons;
- `Shortcuts / hacks taken: none` or the complete list;
- skipped checks and why;
- issue/parent/projection status when applicable;
- for a change that reaches a running service or a required environment: the remaining operator
  action with its exact command, expected observation, and rollback point, whether the parity
  re-check is now pending, and that deployment is separately authorized;
- worktree and artifact-archive confirmation, and every writer that was stopped and restarted.

## Failure handling

- Import failure: confirm the command ran from repository root through the pinned environment.
- Test/review failure: fix the bounded cause, rerun affected checks/review, then rerun the final gate.
- Non-fast-forward push: fetch, integrate, re-review/re-gate affected inputs, and retry without force.
- Ambiguous push: test whether `origin/main` contains the candidate commit before any retry.
- Unavailable GitHub authority: continue only disjoint read-only/local work; do not close, retarget,
  or claim completion until current remote state is verified.
- Unrecoverable material evidence gap: preserve work, report `Checked / Showed / Unknown / Needed`,
  and stop before the affected mutation or push.
- User-owned decision: preserve work, ask it as a self-contained question with options and a
  marked recommendation, and stop before the affected mutation or push.
