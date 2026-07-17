---
name: feature-dev
description: >-
  Plan an evidence-backed binary-algo experiment, method, sweep, new-key
  bootstrap, infrastructure feature, or roadmap package and file the exact
  implementation-ready plan as a GitHub issue. Use for feature specs, "what
  should we try for pair/timeframe", "add this idea to the backlog", "spec out
  X", "design before building", or requests that should end in a durable
  handoff rather than measured results or shipped code. Do not implement, fit,
  evaluate, freeze, or continue into a delivery cycle.
---

# Feature Development Planning

Produce the smallest complete plan supported by the user's intent, current repository evidence,
and binary-algo precedent. End at an exact filed issue, or at an evidence-backed no-change
conclusion when implementation is unnecessary.

Use the host runtime's equivalent primitives for progress tracking, read-only subagents, repository
search, and GitHub access. Do not require a particular model, vendor, or number of agents.

## Governing rules

- Follow `docs/_EVIDENCE-FIRST.md`; define a falsifier before relying on each material premise.
- Extend an existing semantic owner unless evidence proves a net-new surface is necessary.
- Optimize for the minimum complete change. Reject speculative machinery and missing correctness
  controls equally.
- Keep `strategy-eval` authoritative for experiment execution and measured-result recording.
- Make the filed issue self-contained. Planning does not authorize implementation or protected-data
  access.

## 1. Normalize and classify

Establish the requested outcome, non-goals, authority, and intended handoff. If the request cites
`#<number>`, fetch the issue from `sppburke/binary-algo` with title, body, comments, state, labels,
and `updatedAt`; treat its consolidated body as the plan authority and comments as history or review
evidence.

Classify one primary plan kind:

- **experiment** — produces a measured strategy/method result and later runs through
  `strategy-eval`;
- **infrastructure** — changes reusable code, evidence controls, tooling, data contracts, runtime,
  or documentation and later runs through `dev-cycle`;
- **roadmap** — defines packages, dependencies, authority, and trigger-gated deferrals; child
  implementation later runs through the appropriate delivery skill.

Use `mixed` only when a real cross-kind contract cannot be separated. Do not force an infrastructure
or roadmap plan into experiment keys, incumbent metrics, or result-JSON KILL semantics.

Classify planning depth from evidence:

- **contained** — ownership, precedent, integration points, and failure shape are clear and local;
- **complex, risky, or uncertain** — protected data, identity/immutability, concurrency or crash
  recovery, security/authority, external contracts, persistent state, multiple packages, or unclear
  ownership is material.

One focused exploration pass is enough for contained work. Add independent lenses only for named
material uncertainties; agent count is never a goal.

## 2. Explore and falsify

Route each claim to the highest applicable authority:

1. repository instructions and selected skills for process;
2. current code, tests, data, registries, and issue bodies for repository state;
3. targeted read-only commands or safe authorized experiments for runtime behavior;
4. recent commits/issues for rationale and precedent;
5. exact-version primary upstream documentation when an external contract decides the design.

For each material finding record the source, observation, what it establishes, its falsifier, and
any remaining `Checked / Showed / Unknown / Needed` gap. Use safe experiments only when they remain
planning-only and inside existing authorization. Never mutate production or consume a protected look.

Load evidence by plan kind:

- **Experiment:** key `(pair, timeframe, side)`, horizon, mechanism/sign-invariance, current
  incumbent, prior result ledgers, on-disk data, settlement, leakage risks, evaluation discipline,
  and recording destinations.
- **Infrastructure:** current owner and consumers, exact behavior/invariants, existing tests,
  failure and recovery semantics, expected/protected files, public APIs, dependencies, and focused
  negative verification.
- **Roadmap:** program objective and authority, current parent issue and machine projection,
  dependency graph, package acceptance, deferrals, unlock triggers, owners, continuity rules, and
  program-level rejection conditions.

If using helpers, give them bounded independent questions and raw evidence. Read every load-bearing
source yourself and disposition each material helper finding as accepted, corrected, deduplicated,
or rejected.

Stop with a no-change conclusion if evidence shows the behavior already exists, the request is
obsolete, or implementation would add no justified value.

## 3. Resolve material choices

Ask only about a user-owned choice whose answer changes observable behavior, authority, protected
access, data/contract shape, persistence, lifecycle, rollout, accepted risk, or scope boundary.
Resolve technical facts from evidence. Adopt reversible, low-risk, repository-native defaults and
state them without ceremonial confirmation. Ask at most four prioritized questions in one batch.

Wait only when a material user-owned choice remains unresolved. Otherwise proceed with the
evidence-selected recommendation and allow correction.

## 4. Select the minimum-complete design

For contained work with one evidence-dominant solution, present that solution and one credible
rejected alternative. Name the concrete trade-off and failure mode; do not manufacture full option
sets. Present multiple designs only when genuinely different viable ownership, data, contract,
authority, or migration shapes remain.

Every design must identify:

- the existing semantic owner and reusable surfaces;
- every net-new durable surface and strict necessity;
- behavior, failure/recovery semantics, and non-goals;
- a **surface budget**: expected files, public APIs, dependencies, persistent schemas, and protected
  surfaces;
- acceptance and rejection criteria;
- focused verification and any rollout/rollback required by actual risk.

Apply the plan-kind falsifier:

- **Experiment:** pre-register exact KILL conditions before OOS, including settlement, selection,
  per-year evidence, incumbent comparison, leakage traps, and certification only when a deployable
  book is the deliverable.
- **Infrastructure:** pre-register exact `DO NOT SHIP` conditions and negative tests/audits for
  runtime-rejectable failures; never require a result JSON unless the feature actually measures a
  strategy.
- **Roadmap:** define package completion and program `DO NOT SHIP` conditions. Preserve every
  valuable deferral with `id`, `status`, `decision`, `why_valuable`, `unlock_trigger`, and
  `owner_issue`; a fired trigger requires a new decision and never silently enters the critical
  path.

Use one delivery phase unless an independently reviewable/deployable, migration, compatibility,
cutover, or blast-radius boundary requires more. Risk adds evidence and safeguards, not fake phases.

## 5. Compose, review, and file

Compose one consolidated issue artifact with an exact title and a body containing:

- authority and intended outcome;
- plan kind and scope/key where applicable;
- evidence-backed findings and unresolved evidence gaps;
- resolved decisions and defaults;
- chosen design, semantic owner, surface budget, and rejected alternative;
- plan-kind acceptance and rejection conditions;
- concrete implementation steps with owning paths/surfaces;
- focused verification with expected outcomes;
- expected and protected files;
- non-goals and preserved deferrals/follow-ups;
- parent/child and machine-projection continuity when named.

Map intent/decisions to acceptance criteria, steps to owning surfaces, and criteria to verification.
Check for duplicate open issues and relevant recent changes immediately before filing.

Self-review the exact candidate for unsupported premises, duplication, unnecessary surfaces,
missing failure behavior, unverifiable criteria, and surface-budget drift. Then give the exact
candidate, intent, repository target, and raw supporting evidence—without a desired verdict—to one
fresh read-only `plan-review` reviewer when available. A caller-launched independent review never
launches another reviewer. If unavailable, perform a separated direct pass and disclose the
fallback.

Verify and disposition every finding. File only an `approve` candidate or an
`approve with revisions` candidate after required revisions are resolved. A material change to
scope, architecture, behavior, contract, authority, or acceptance criteria requires a fresh full
review; a localized correction requires review of its delta and integration invariants.

Print the final title and body exactly as filed:

- New plan: create one issue in `sppburke/binary-algo` using the exact body file.
- Existing authoritative plan: replace the issue body with `gh issue edit --body-file`; use comments
  only for review receipts or change history, never as a contradictory live amendment stack.

Use an exact temporary body file outside the repository and remove it after filing. Calculate its
SHA-256 and record a compact issue comment after a successful file/update:

```text
plan-review-receipt/v1
artifact_sha256: <exact issue-body bytes hash>
repo_head: <reviewed git HEAD>
verdict: approve | approve-with-revisions
```

Report the issue URL, receipt, and any incomplete metadata action. Stop. Do not implement, evaluate,
fit, freeze, or begin a delivery cycle.
