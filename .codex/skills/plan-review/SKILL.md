---
name: plan-review
description: >-
  Review a proposed, not-yet-executed binary-algo experiment, method, sweep,
  new-key bootstrap, infrastructure specification, lab specification or
  registration, roadmap package, GitHub issue, or implementation handoff for
  evidence, internal consistency, repository fit, leakage, executability, and
  minimum-complete scope. Use for "review this plan", "is this ready", "is
  this leakage-free", "does this follow our discipline", global-optimum/MVP
  checks, or before strategy-eval or dev-cycle. Return an evidence-backed
  approve, approve-with-revisions, reject, or blocked verdict with exact plan
  deltas and one consolidated, more comprehensive proposed revision of the
  plan. Do not originate a plan, implement it, run an experiment, or review a
  completed diff.
---

# Plan Review

Review the exact plan artifact against the repository it would change. Remain read-only: the review
never edits the source artifact, the repository, a hosted issue, or any external state, and never
implements the plan, fits a model, consumes protected outcomes, freezes a book, or reviews
completed code. Its response always ends with one consolidated proposed revision of the plan.
Applying that revision is a separate, separately authorized step owned by `feature-dev`.

Use the host runtime's equivalent primitives and native read-only subagents when available. A
caller-launched independent review is contained and never launches another reviewer. For a direct
complex or materially uncertain review, use at most one fresh counter-reviewer: give it the exact
artifact, the intent, the target revision, the raw authorized evidence, and the premises to try to
disprove, and withhold your provisional verdict. Verify and disposition its findings yourself.
Disclose a direct fallback, with the reason, when independent review is unavailable or fails.

## Inputs and artifact authority

For `#<number>`, read the issue with `git-workflow`: title, body, comments, state, labels,
`updatedAt`, and URL. The consolidated issue body is the Artifact. Comments are context, change
history, or review receipts unless the user explicitly identifies a comment as the proposed
artifact. Do not treat contradictory append-only amendments as one coherent plan. Disposition
every substantive unresolved comment point that affects the Artifact, and name any receipt that
binds a different body.

The Artifact may equally be a file: a lab specification, a registration draft, or a backlog entry
prepared as its own file. Record its path and review exactly those bytes. A comment the user names
is hashed like an issue body, from its UTF-8 text. When the plan is only part of a larger file, or
exists only in the conversation, copy its exact bytes in memory or into a scratch file outside the
repository, review and hash that copy, and record where it came from.

If the exact plan artifact is unavailable, return `blocked`. Infer intent, constraints, and
touchpoints when repository evidence resolves them, and label each inferred acceptance criterion
and default as inferred. Ask only when ambiguity changes measured behavior, authority, protected
access, persistence, lifecycle, contract shape, accepted risk, or the deliverable. When such a
user-owned choice stays unresolved, name the exact decision, record it as a material evidence gap,
apply normal verdict precedence, and put the decision to the owner as a question with options and a
marked recommendation. Never invent the answer.

Classify the Artifact as `experiment`, `infrastructure`, `roadmap`, or a justified `mixed` plan.
Apply the universal audit plus the matching overlay below. An infrastructure or roadmap plan does
not require an experiment key, incumbent metric, or result-JSON KILL condition.

## Evidence and review standard

`docs/_EVIDENCE-FIRST.md` governs sources, falsifiers, conflicts, and evidence gaps. The Artifact is
a claim source, not proof. Before checking each material claim, record what would falsify it.

Apply these guidelines together:

1. accurate to the intent of the change;
2. internally consistent;
3. the minimum complete scope required for the intended deliverable;
4. re-uses as much as possible of the current code base;
5. leverages repo precedence when re-use is not possible;
6. is as close to a global optimum as feasible within that minimum-complete scope;
7. simplicity is important and essential;
8. code elegance is the highest form of beauty;
9. over-engineering and shortcuts are equally bad and undesirable.

Minimum viable means minimum complete, not smallest diff. Scope expansion requires evidence that a
smaller design fails intent, correctness, research validity, authority, recovery, or verification.

Safeguards face the same evidence bar as any other claim. A guard, validation branch, retry,
timeout, fallback, or fail-closed refusal the plan specifies must name the failure it prevents and
cite either direct evidence that the failure occurs on the current revision or the stated contract
that requires the safeguard. Ledger that justification like any claim: a contradicted
justification grades the safeguard over-engineered, and an unproven one is an evidence gap.

## Review method

### 1. Normalize

Extract and mark `explicit | inferred | missing | contradicted`:

- intent, authority, actor/key, observable outcome, and deliverable;
- acceptance and rejection/KILL/DO-NOT-SHIP conditions;
- non-goals, constraints, expected and protected surfaces;
- implementation ownership, dependencies, lifecycle/recovery, recording, and verification;
- parent/child or machine-projection continuity when named.

### 2. Build two complete inventories

Create a **material-claim ledger** for every assertion that can change the measured result,
research validity, identity/authority, protected access, behavior/contract, persistence, recovery,
deliverable, or verification. Resolve each row to exactly one state:

- `supported` — applicable authority supports it and its falsifier was checked;
- `contradicted` — applicable authority disproves it;
- `evidence-gap` — accessible applicable authority was exhausted without resolution.

A claim about a set that licenses uniform treatment is checked per member. When the plan treats
keys, pairs, sides, arms, paths, books, files, or consumers alike, verify each member against its
own authority and record each outcome and exception: one ledger row per member, or one batch row
whose falsifier reports every member. Preserve the authoritative order where order carries identity
or decides something.

Create a separate **named-surface/citation inventory** for every cited path, symbol, key/horizon,
number, result/book/issue ID, command, output, API/config field, expected file, protected file, and
machine-projection record. Promote consequential drift into the material ledger. A check of a
proposed file, command, or interface establishes only that its planned contract is precise and
plausible; report existing implementation and observed proof separately.

Complete both inventories before verdict. Coverage must balance: `N = S + C + G`, and every named
surface must be checked. Do not stop after the first verdict-determining defect.

### 3. Falsify and trace end to end

Check the highest applicable source and reconcile conflicts. Trace the proposed path from entry or
research question through semantic owner, state/IO, consumers, evidence, terminal decision, and
recording. Enumerate the consumers of every changed shared contract, evidence field, manifest,
configuration key, or helper. Run a non-destructive targeted check only when repository evidence is
insufficient and the check stays inside authorization. Missing evidence is an evidence gap, not
proof of a defect.

Checking a protected surface means checking its seal, family, receipt, and recovery contract from
unprotected evidence. It never means reading the protected outcome.

### 4. Search reuse and the simpler counterfactual

First establish which parts of the goal the tools and data already available can meet. Then
independently find the nearest reusable semantic owner and healthy precedent. Require strict
necessity for each new file, API, dependency, persistent schema, service, framework, or authority.
Compare every non-obvious choice with the strongest simpler repository-native design. When the
evidence shows a shared class of failures or an incomplete owner, also compare the smallest
complete design that covers every affected member, and state the gap each option leaves. Do not
demand alternatives ceremony when one established shape clearly fits.

Audit the surface budget. A plan must name expected files, public APIs, dependencies, persistent
schemas, and protected surfaces when applicable. Unbounded implementation rediscovery is a plan
defect; line-level edit instructions are not required.

### 5. Apply the plan-kind overlay

#### Experiment overlay

Identify the experiment's governing protocol before applying a checklist, and review the plan
against that protocol's own authority:

- **Deriv-feed program work** is governed by `docs/DERIV_SOTA_PROGRAM.md` and the newest applicable
  owner decision in the program log (see `deriv-lab-ops`). Verify against those sources, not
  against the legacy list: the data roles of every input, the development and sealed windows, the
  venue masks applied in training, validation, and test, the validation mode behind any
  label-based choice, registration of the arm before its first run with its falsifier and budget,
  the strict per-side certificate read through `scripts/certificate.py`, the payout reading as a
  separate deployment gate, the registered rule and authorization for any look, the compute lane
  and cap, and the recording path. A plan that copies a number the program can move, instead of
  citing where it lives, is a Should Fix. The legacy list applies to this work only where the
  governing record adopts it; a defect demonstrated independently is a finding either way.
- **Legacy feature-store work** is governed by `strategy-eval` and the list below.
- Work that touches the protected-evaluation or evidence contracts is also checked against
  `docs/PROTECTED_EVALUATION.md` and `docs/RESEARCH_EVIDENCE_CONTRACT.md`.

For legacy feature-store work, require and verify:

- exact `(pair, timeframe, side)` and `MX_HOR`/`HS`, evaluation target, and data substrate;
- a plausible sign-carrying mechanism or an explicit magnitude reframe;
- pre-registered KILL conditions written before OOS is inspected;
- Deriv-faithful `wc_ret`, ties LOSE, breakeven `0.541`, and target-pair clock integrity;
- per-year `2024 / 2025 / 2026` evidence with `2026` strict OOS, moved-bars only, moved up-rate
  sanity, `nonoverlap_chrono`, `boot` CI95, and selection on VAL worst-half rather than VAL-accuracy
  maximization;
- leakage controls for horizon alignment, ffill-flat mirage, future smoothing, greedy overlap,
  thin coverage, and sign-invariant features;
- comparison against the best current combination on its binding year;
- for a deployable book, 15-path per-fold-refit CPCV with `p10 >= 0.541`, at least `80%` of paths
  clearing breakeven, and the DST-correct NY gate;
- frozen-past forward evidence for calendar, seasonal, slow, or otherwise non-stationary features;
- one result JSON plus the correct Tier-2 ledger/sweep destinations, never a per-key number copied
  into a generic authority;
- one heavy fit at a time, with only the exception `AGENTS.md` allows, and resumable bounded
  compute.

Under either protocol: an experiment with no pre-registered falsifier is Blocking; a search closed
in `AGENTS.md` stays closed unless the plan cites the owner decision that reopened it for its
substrate, keys, and family; and a plan that would open sealed data needs the owner's
authorization for that named look. Keep detailed execution discipline in `strategy-eval`; do not
redesign it here.

#### Infrastructure overlay

Require and verify:

- objective, authority, invariant set, semantic owner, consumers, and why the proposed scope is
  minimum;
- exact behavior and failure/recovery semantics, including concurrency, crash, retry, idempotence,
  immutability, protected access, or secret handling only when touched;
- expected and protected files, compatibility boundaries, and prohibited authority expansion;
- explicit `DO NOT SHIP` conditions with automated negative tests for runtime-rejectable failures
  and recorded audits for protected surfaces or actions that tests must not perform;
- focused repo-root commands with asserted outcomes, plus rollout/rollback only when the change can
  affect live or persistent state;
- for a change to runtime behavior or a shared contract, a representative non-live proof through
  the normal entry point, the affected owners, and a real downstream consumer, with its command,
  inputs, and independently justified expected output. Imports, unit results, and exit status
  alone are deficient proof. Judge the proof's design now: running a well-specified proof is a
  delivery gate, not a gap in the review. Require an observed result only for a premise that claims
  the behavior already exists;
- for a change to the Deriv runtime, the consequences named in `deriv-runtime-ops`: buyer
  exclusivity, the parity re-check, and which host actions the plan would need separately
  authorized.

Do not require result JSONs, model metrics, or strategy looks for infrastructure acceptance.

#### Roadmap overlay

Require and verify:

- program objective, human authority (an issue or an owner-approved program record), the machine
  projection that authority names, current state, and minimal dependency graph;
- each package's value, prerequisites, acceptance, exclusions, and independently reviewable handoff;
- every deferral's `id`, `status`, `decision`, `why_valuable`, `unlock_trigger`, and `owner_issue`;
- durable continuity: IDs are not silently deleted, supersession is explicit, a fired trigger creates
  a decision point rather than automatic implementation, and dormant deferrals do not block MVP;
- child completion updates the authoritative parent and named machine projection consistently;
- program-level `DO NOT SHIP` conditions and a rule against speculative issue trees.

Do not turn optional future capability into an MVP dependency without cited falsifying evidence.

### 6. Check partial implementation and phases

Partial code or results may accompany an otherwise unexecuted plan only as evidence of drift.
Review plan quality first, then classify each step `done | partial | missing | extra |
changed-shape`. Material changed-shape drift contradicts the plan until the consolidated artifact
adopts the verified contract; extra work needs a scope justification. Leave code correctness to
implementation review and measured-number correctness to `strategy-eval`.

For multiple delivery phases, require a real deployable/reviewable, migration, compatibility,
cutover, or blast-radius boundary plus phase-local acceptance and verification. For each phase
that leaves visible or persistent state, require its compatibility contract, recovery point, and
proof that it can be reviewed and safely left in that state before the next phase starts. Do not
invent delivery phases for a coherent atomic change.

## Findings and verdict

Use these severities:

- **Blocking** — a supported defect that would invalidate correctness, research evidence,
  authority, protected access, contract, persistence/recovery, scope, or authoritative verification.
- **Should Fix** — a non-material revision required for clarity, explicit ownership, scope, evidence,
  or handoff quality.
- **Consider** — an optional supported trade-off that does not gate execution.

An **Evidence Gap** is a claim state, not a severity: unresolved proof, reported with the four
`Checked / Showed / Unknown / Needed` fields.

Every contradicted material claim must appear in a Blocking finding. For a missing required gate
or safeguard, cite the governing requirement, the passage that establishes the omission, and the
material consequence. It is Blocking when the requirement binds and the complete applicable
artifact, including any source it incorporates, shows the omission; where a source not yet read
could show the requirement is met, it is an evidence gap.

Every Blocking and Should Fix finding must cite evidence, tag applicable guidelines (`G1`–`G9`),
and provide exactly one structured delta:

```text
operation: add | replace | delete | move
target: <artifact section or claim>
anchor: <exact text the edit attaches to; for a move, both source and destination>
supersedes: <exact live claim or none>
replacement: <minimum required wording or behavior>
```

Check each proposed delta against reuse, precedent, leakage/authority, and minimum-complete scope.
Resolve every anchor uniquely in the exact artifact. Apply all required deltas together to an
in-memory copy, confirm that superseded directives are gone and interacting requirements agree,
and build the proposed revision from that result. Append-only amendments cannot earn approval
while superseded text remains live.

Apply verdict precedence exactly:

1. any Blocking finding or contradicted material claim -> `reject`;
2. otherwise any exhausted material evidence gap -> `blocked`;
3. otherwise any required non-material revision -> `approve with revisions`;
4. otherwise -> `approve`.

The verdict applies to the reviewed bytes only. A revised body earns approval through the
receiving workflow's coverage rules: delta review for a localized correction, a fresh full review
for a material change.

## Proposed revision

Every review, whatever its verdict, ends with one consolidated proposed revision of the plan,
written out in full and ready to paste:

- Start from the reviewed body with every required delta applied and every superseded instruction
  removed. Preserve the plan's authorized objective and non-goals. Mark every change that goes
  beyond a required delta as an added clarification so the plan owner can decline it.
- Define every term that can change implementation or a decision: thresholds, keys, windows,
  clocks, units, boundaries, comparators, states, configuration names, and staleness limits.
- Lay the work out as explicit phases. Name the delivery phases (one, unless a boundary section 6
  requires justifies more), and inside each, in order, its execution phases: prerequisites, each
  build or run step that a gate separates, and verification. For every phase give its scope, its
  working root and the files it reads and changes, its prerequisites, its own pre-registered
  falsifier or acceptance gate with the exact check and expected result, and the stop rule that
  decides whether the next phase may start. Map every KILL or DO-NOT-SHIP condition to the phase
  that tests it.
- Give each requirement one authoritative place and cross-reference it elsewhere. Keep every
  qualifier, command, input, and stop condition that affects execution or proof.
- Write unresolved material choices and missing proof as explicit stop conditions that name the
  decision or the evidence needed. Do not fill them in. Where a Blocking finding can only be
  resolved by a new scope or authority decision, keep the unaffected text and put the owner's
  question there, with options and a marked recommendation.

For a `reject` or `blocked` verdict the revision shows the plan as it must read to be reviewable
again; it is not approved text. When there is nothing to revise (no artifact, or an idea with no
proposed approach), return `blocked`, mark the fingerprint and coverage fields `not available`, and
state here the input that is needed; drafting a plan from an idea is `feature-dev`. Offer to apply
the revision through `feature-dev`; never apply it within the review.

## Output contract

```text
## Coverage
plan kind: experiment | infrastructure | roadmap | mixed
governing protocol: Deriv-feed program | legacy feature store | not-applicable
product / research logic: <planned outcome and governing rule>
material claims: N | supported: S | contradicted: C | evidence gaps: G
named surfaces/citations: X/Y
execution mode: contained-direct | caller-independent | counter-reviewer | direct-fallback

## Review Fingerprint
artifact: <issue URL, comment URL, or file path and its source>
artifact_sha256: <SHA-256 of the exact Artifact bytes>
repo_head: <git HEAD reviewed>
issue_updated_at: <value or not-applicable>

## Verdict
approve | approve with revisions | reject | blocked — <one sentence>

## Blocking       # omit when empty
- [G#] finding — evidence — structured delta

## Should Fix     # omit when empty
- [G#] finding — evidence — structured delta

## Consider       # omit when empty
- trade-off — evidence — optional delta

## Global Optimum
- Reuse / precedent: ...
- Simpler counterfactual: ...
- Complexity: under-engineered | minimum-complete | over-engineered
- Cleanliness: ...

## Evidence Gaps
- Checked: ...
  Showed: ...
  Unknown: ...
  Needed: ...

## Proposed revision
<the full consolidated plan body, in phases, with definitions and stop conditions>
```

Omit empty severity sections; write `None` for no evidence gaps. Name an issue number and URL when
reviewing one. State whether any material evidence came from uncommitted changes, and bind runtime
evidence to its environment and data identity where that matters. Do not post, edit, close, or
comment; publication is a separate action the user authorizes after the review.

Before emitting, confirm both inventories are complete and balanced, every set claim was checked
per member, every factual conclusion has applicable evidence or an exact gap, every delta passed
its own reuse and validity check and resolved its anchor, the proposed revision contains every
required delta and no superseded text, the verdict follows precedence, and the review remained
non-mutating.
