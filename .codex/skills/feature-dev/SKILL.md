---
name: feature-dev
description: >-
  Plan an evidence-backed binary-algo experiment, method, sweep, new-key
  bootstrap, infrastructure feature, or roadmap package and hand the exact
  implementation-ready, reviewed plan to its authority: a GitHub issue it
  files, a lab specification or registration that deriv-lab-ops files, or a
  backlog entry that dev-cycle files. Use for feature specs, "what should we
  try for pair/timeframe", "add this idea to the backlog", "spec out X",
  "design before building", "apply the review to #<n>", or requests that
  should end in a durable handoff rather than measured results or shipped
  code. Do not implement, fit, evaluate, freeze, or deliver the planned work.
---

# Feature Development Planning

Produce the smallest complete plan supported by the user's intent, current repository evidence,
and binary-algo precedent. End at the first of these that applies:

1. the reviewed plan at its authoritative handoff (below);
2. an existing artifact that already owns the request, named with its evidence;
3. an evidence-backed no-change conclusion;
4. the exact candidate title and body plus what blocks publication: publishing is outside the
   request, it cannot be done, or the review still rejects or blocks the candidate. In the last
   case give the verdict and the evidence or decision still needed, and leave the existing
   authoritative artifact as it is.

Use the host runtime's equivalent primitives for progress tracking, read-only subagents, repository
search, and GitHub access. Do not require a particular model, vendor, or number of agents.

## Governing rules

- Follow `docs/_EVIDENCE-FIRST.md`; define a falsifier before relying on each material premise.
- Extend an existing semantic owner unless evidence proves a net-new surface is necessary.
- Optimize for the minimum complete change. Reject speculative machinery and missing correctness
  controls equally.
- Keep `strategy-eval` authoritative for experiment execution and measured-result recording.
- Make the plan self-contained. Planning does not authorize implementation, protected-data access,
  a holdout look, a host action, or new data acquisition.

## 1. Normalize, find the handoff, and classify

Establish the requested outcome, non-goals, authority, and intended handoff. If the request cites
`#<number>`, read the issue with `git-workflow` (title, body, comments, state, labels, `updatedAt`,
URL); treat its consolidated body as the plan authority and comments as history or review evidence.

Choose the authoritative handoff before drafting, from where this kind of work is governed today:

- **GitHub issue** — when the user asks for one, when a governing contract requires a reviewed
  issue, or for repository infrastructure with no other owner.
- **Lab specification or registration** — for Deriv-feed program work: a specification for a
  staged build, or the registration of a run. The plan is drafted and reviewed here as one file;
  `deriv-lab-ops` files it in the lab.
- **Backlog entry** — a generic idea (mechanism, sign-invariance note, falsifier template) goes in
  `docs/IDEAS_LOG.md`; a queued experiment for a named key goes in that key's
  `sweeps/<PAIR>_<tf>_backlog.md`. The entry is drafted and reviewed here; adding it to the tracked
  file is a `dev-cycle` change. Where both records are written, say which one controls execution
  and review each as its own candidate.

Name the destination, the authority that governs it, and the workflow that will consume it. Where
an issue and a file both govern, say what each controls and link their identities.

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
2. current code, tests, data, registries, issue bodies, and the program record for state;
3. targeted read-only commands or safe authorized experiments for runtime behavior;
4. recent commits/issues for rationale and precedent;
5. exact-version primary upstream documentation when an external contract decides the design.

For each material finding record the source, observation, what it establishes, its falsifier, and
any remaining `Checked / Showed / Unknown / Needed` gap. Use safe experiments only when they remain
planning-only and inside existing authorization. Never mutate production or consume a protected look.

Before drafting the design, apply the material-claim and named-surface checks that `plan-review`
defines to the load-bearing evidence, including its per-member check wherever the plan will treat
several keys, pairs, sides, arms, paths, or records alike.

Load evidence by plan kind:

- **Experiment:** key `(pair, timeframe, side)`, horizon, mechanism/sign-invariance, current
  incumbent, prior result ledgers, the governing evaluation protocol and its authority, the data
  substrate and each input's role, settlement, leakage risks, closures that apply, and recording
  destinations.
- **Infrastructure:** current owner and consumers, exact behavior/invariants, existing tests,
  failure and recovery semantics, expected/protected files, public APIs, dependencies, and focused
  negative verification. For the Deriv runtime, the consequences `deriv-runtime-ops` names.
- **Roadmap:** program objective and authority (an issue or an owner-approved program record), the
  current parent and any machine projection it names, dependency graph, package acceptance,
  deferrals, unlock triggers, owners, continuity rules, and program-level rejection conditions.

If using helpers, give them bounded independent questions and raw evidence. Read every load-bearing
source yourself and disposition each material helper finding as accepted, corrected, deduplicated,
or rejected.

Stop with a no-change conclusion if evidence shows the behavior already exists, the request is
obsolete, or implementation would add no justified value.

## 3. Resolve material choices

Ask only about a user-owned choice whose answer changes observable behavior, authority, protected
access, data/contract shape, persistence, lifecycle, rollout, accepted risk, or scope boundary.
Resolve technical facts from evidence. Adopt reversible, low-risk, repository-native defaults and
state them without ceremonial confirmation. Never default a choice that opens sealed data, adds a
data source, raises a compute cap, reopens a closed search, or touches a live account. Ask at most
four prioritized questions in one batch, each self-contained with its options and a marked
recommendation.

Wait only when a material user-owned choice remains unresolved. Otherwise proceed with the
evidence-selected recommendation and allow correction.

When an answer changes an earlier ruling, record the new answer and its evidence, name the exact
ruling it supersedes, and update every gate and check that depended on the old one.

## 4. Select the minimum-complete design

First test whether an existing command, contract, data source, or workflow already meets the
requested outcome. For contained work with one evidence-dominant solution, present that solution
and one credible rejected alternative. Name the concrete trade-off and failure mode; do not
manufacture full option sets. Present multiple designs only when genuinely different viable
ownership, data, contract, authority, or migration shapes remain. If a proposed fix addresses one
example and evidence shows a broader affected class, compare it with the smallest complete fix for
that class.

Every design must identify:

- the existing semantic owner and reusable surfaces;
- every net-new durable surface and strict necessity;
- behavior, failure/recovery semantics, and non-goals;
- for each added safeguard, the failure it prevents and the direct evidence or governing contract
  that requires it. A guard justified only by "a caller could" or "just in case" is recorded as a
  rejected alternative together with the check that would justify it;
- a **surface budget**: expected files, public APIs, dependencies, persistent schemas, and protected
  surfaces;
- every seal, pin, manifest, versioned contract, and frozen identity the change touches, with its
  current binding, its owner, what must stay immutable, and the authorized successor step;
- acceptance and rejection criteria;
- focused verification and any rollout/rollback required by actual risk.

Apply the plan-kind falsifier:

- **Experiment:** name the governing evaluation protocol and its authority. Record the dataset
  identity, clocks, and each input's role and split. Name every outcome already used to choose the
  candidate, and count those choices as the governing protocol requires. Before inspecting any
  out-of-sample outcome (legacy protocol) or before the arm's first run (Deriv-feed program), freeze
  the selection rule, the comparator, the ordered family where one applies, the exact KILL and terminal
  rules, and the authorized look schedule. Use the evidence windows that protocol requires:
  per-year evidence under the legacy protocol, the registered development and sealed windows under
  the Deriv-feed program. Include settlement, incumbent comparison, and leakage traps. Require
  book certification only when a deployable book is the deliverable, and the governing protocol's
  certificate wherever it selects or accepts a result.
- **Infrastructure:** pre-register exact `DO NOT SHIP` conditions and negative tests/audits for
  runtime-rejectable failures; never require a result JSON unless the feature actually measures a
  strategy.
- **Roadmap:** define package completion and program `DO NOT SHIP` conditions. Preserve every
  valuable deferral with `id`, `status`, `decision`, `why_valuable`, `unlock_trigger`, and
  `owner_issue`; a fired trigger requires a new decision and never silently enters the critical
  path.

Use one delivery phase unless an independently reviewable/deployable, migration, compatibility,
cutover, or blast-radius boundary requires more. Risk adds evidence and safeguards, not fake phases.
For each justified phase, name its direct and inherited prerequisites, its scope and owners, its
acceptance checks and retained evidence, its release decision, and which later work stays gated.

## 5. Compose, review, and hand off

Compose one consolidated plan artifact with an exact title and a body containing:

- authority, intended outcome, and the chosen handoff;
- plan kind and scope/key where applicable;
- evidence-backed findings and unresolved evidence gaps;
- resolved decisions and defaults, and any ruling they supersede;
- chosen design, semantic owner, surface budget, and rejected alternative;
- plan-kind acceptance and rejection conditions;
- ordered implementation stages with owning paths/surfaces, each with its own gate and stop rule;
- focused verification with expected outcomes. For each goal-bearing criterion give the exact
  command, its working directory, representative inputs, the assertion, and the evidence to keep.
  A behavior change is proved through its normal entry point and a real consumer on safe inputs;
  imports, unit results, and exit status alone prove nothing. Name the unchanged reference where
  parity matters and the deliberate break where a negative control is needed;
- expected and protected files;
- non-goals and preserved deferrals/follow-ups;
- parent/child and machine-projection continuity when named;
- when the user asks for a pickup brief, the read-first documents, allowed and protected surfaces,
  authorized environment and data, and the stop conditions for the next worker. The brief points
  to the plan; it does not copy it.

Map intent/decisions to acceptance criteria, steps to owning surfaces, and criteria to verification.
Define every term that can change implementation.
Check for duplicate open issues, an existing owner in the program record, and relevant recent
changes immediately before handing off.

Self-review the exact candidate for unsupported premises, duplication, unnecessary surfaces,
missing failure behavior, unverifiable criteria, and surface-budget drift. Then give the exact
candidate, intent, repository target, and raw supporting evidence—without a desired verdict—to one
fresh read-only `plan-review` reviewer when available, bound by path to this repository's
`.codex/skills/plan-review/SKILL.md`. A caller-launched independent review never launches another
reviewer. If unavailable, perform a separated direct pass and disclose the fallback.

Verify and disposition every finding, and take the reviewer's proposed revision as a candidate,
not as approved text. Hand off only an `approve` candidate or an `approve with revisions` candidate
after required revisions are resolved. A material change to scope, architecture, behavior,
contract, authority, or acceptance criteria requires a fresh full review; a localized correction
requires review of its delta and integration invariants. The fingerprint that is handed off is the
one the last review gave for the final bytes.

When revising an existing plan from findings the user approved, apply exactly that set. Report a
newly noticed concern separately and get approval before folding it in.

Print the final title and body exactly as filed, then publish by handoff kind. Publishing needs
the request to cover it: "review", "improve", or "make it more detailed" asks for proposed text
only.

- **New issue:** create one issue in `sppburke/binary-algo` using the exact body file.
- **Existing authoritative issue:** replace the body with `gh issue edit --body-file` after
  `git-workflow`'s re-read; reconcile any intervening change first. Use comments only for review
  receipts or change history, never as a contradictory live amendment stack.
- **Lab specification or registration:** return the reviewed file, its SHA-256, the verdict, and
  the destination. Filing it is `deriv-lab-ops`'s filing step.
- **Backlog entry:** return the reviewed entry, its SHA-256, and the destination file and section.
  Adding it to the tracked file is a `dev-cycle` change that takes this entry as its authority.

For an issue, `git-workflow` has the commands: the exact body file outside the repository (remove
it after filing), the hosted read-back whose SHA-256 must equal the approved body's before any
receipt, and the recovery after an ambiguous create. The body carries all executable scope; read
the hosted title and body back. Then record a compact issue comment:

```text
plan-review-receipt/v1
artifact_sha256: <SHA-256 of the exact issue-body bytes>
repo_head: <reviewed git HEAD>
verdict: approve | approve-with-revisions
```

Report the handoff location, the receipt or fingerprint, and any incomplete metadata action. If a
later step fails after the issue exists, return its URL and name what is incomplete. When the
request asked for a lab or backlog filing, start that filing in the same turn with the reviewed file,
and verify what it wrote; the plan itself is not started. Then stop. Do not implement, evaluate,
fit, freeze, or begin any other delivery cycle.
