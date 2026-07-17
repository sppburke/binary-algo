---
name: plan-review
description: >-
  Review a proposed, not-yet-executed binary-algo experiment, method, sweep,
  new-key bootstrap, infrastructure specification, roadmap package, GitHub
  issue, or implementation handoff for evidence, internal consistency,
  repository fit, leakage, executability, and minimum-complete scope. Use for
  "review this plan", "is this ready", "is this leakage-free", "does this
  follow our discipline", global-optimum/MVP checks, or before strategy-eval or
  dev-cycle. Return an evidence-backed approve, approve-with-revisions, reject,
  or blocked verdict with exact plan deltas. Do not author a plan, implement it,
  run an experiment, or review a completed diff.
---

# Plan Review

Review the exact plan artifact against the repository it would change. Remain read-only: never
rewrite or mutate the plan, implement it, fit a model, consume protected outcomes, freeze a book,
or perform a completed-code review.

Use the host runtime's equivalent primitives and native read-only subagents when available. A
caller-launched independent review is contained and never launches another reviewer. For a direct
complex or materially uncertain review, use at most one fresh counter-reviewer; verify and
disposition its findings yourself. Disclose a direct fallback when independent review is
unavailable.

## Inputs and artifact authority

For `#<number>`, fetch `sppburke/binary-algo` issue title, body, comments, state, labels,
`updatedAt`, and URL. The consolidated issue body is the Artifact. Comments are context, change
history, or review receipts unless the user explicitly identifies a comment as the proposed
artifact. Do not treat contradictory append-only amendments as one coherent plan.

If no plan body exists, return `blocked`. Infer intent, constraints, and touchpoints when repository
evidence resolves them; ask only when ambiguity changes measured behavior, authority, protected
access, persistence, lifecycle, contract shape, accepted risk, or the deliverable.

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

Create a separate **named-surface/citation inventory** for every cited path, symbol, key/horizon,
number, result/book/issue ID, command, output, API/config field, expected file, protected file, and
machine-projection record. Promote consequential drift into the material ledger.

Complete both inventories before verdict. Coverage must balance: `N = S + C + G`, and every named
surface must be checked. Do not stop after the first verdict-determining defect.

### 3. Falsify and trace end to end

Check the highest applicable source and reconcile conflicts. Trace the proposed path from entry or
research question through semantic owner, state/IO, consumers, evidence, terminal decision, and
recording. Run a non-destructive targeted check only when repository evidence is insufficient and
the check stays inside authorization. Missing evidence is an evidence gap, not proof of a defect.

### 4. Search reuse and the simpler counterfactual

Independently find the nearest reusable semantic owner and healthy precedent. Require strict
necessity for each new file, API, dependency, persistent schema, service, framework, or authority.
Compare every non-obvious choice with the strongest simpler repository-native design. Do not demand
alternatives ceremony when one established shape clearly fits.

Audit the surface budget. A plan must name expected files, public APIs, dependencies, persistent
schemas, and protected surfaces when applicable. Unbounded implementation rediscovery is a plan
defect; line-level edit instructions are not required.

### 5. Apply the plan-kind overlay

#### Experiment overlay

Require and verify:

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
- one heavy fit at a time and resumable bounded compute.

An experiment with no pre-registered falsifier is Blocking. Keep detailed execution discipline in
`strategy-eval`; do not redesign it here.

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
  affect live or persistent state.

Do not require result JSONs, model metrics, or strategy looks for infrastructure acceptance.

#### Roadmap overlay

Require and verify:

- program objective, human authority, machine projection, current state, and minimal dependency
  graph;
- each package's value, prerequisites, acceptance, exclusions, and independently reviewable handoff;
- every deferral's `id`, `status`, `decision`, `why_valuable`, `unlock_trigger`, and `owner_issue`;
- durable continuity: IDs are not silently deleted, supersession is explicit, a fired trigger creates
  a decision point rather than automatic implementation, and dormant deferrals do not block MVP;
- child completion updates the authoritative parent and named machine projection consistently;
- program-level `DO NOT SHIP` conditions and a rule against speculative issue trees.

Do not turn optional future capability into an MVP dependency without cited falsifying evidence.

### 6. Check partial implementation and phases

If code or results already exist, review plan quality first, then classify each step
`done | partial | missing | extra | changed-shape`. Report plan/code contract drift; leave code
correctness to implementation review and measured-number correctness to `strategy-eval`.

For multiple phases, require a real deployable/reviewable, migration, compatibility, cutover, or
blast-radius boundary plus phase-local acceptance and verification. Do not invent phases for a
coherent atomic change.

## Findings and verdict

Use these severities:

- **Blocking** — a supported defect that would invalidate correctness, research evidence,
  authority, protected access, contract, persistence/recovery, scope, or authoritative verification.
- **Should Fix** — a non-material revision required for clarity, explicit ownership, scope, evidence,
  or handoff quality.
- **Consider** — an optional supported trade-off that does not gate execution.
- **Evidence Gap** — unresolved proof, not a defect severity.

Every Blocking and Should Fix finding must cite evidence, tag applicable guidelines (`G1`–`G9`),
and provide exactly one structured delta:

```text
operation: add | replace | delete | move
target: <artifact section or claim>
supersedes: <exact live claim or none>
replacement: <minimum required wording or behavior>
```

Check each proposed delta against reuse, precedent, leakage/authority, and minimum-complete scope.
Append-only amendments cannot earn approval while superseded text remains live.

Apply verdict precedence exactly:

1. any Blocking finding or contradicted material claim -> `reject`;
2. otherwise any exhausted material evidence gap -> `blocked`;
3. otherwise any required non-material revision -> `approve with revisions`;
4. otherwise -> `approve`.

## Output contract

```text
## Coverage
plan kind: experiment | infrastructure | roadmap | mixed
product / research logic: <planned outcome and governing rule>
material claims: N | supported: S | contradicted: C | evidence gaps: G
named surfaces/citations: X/Y
execution mode: contained-direct | caller-independent | counter-reviewer | direct-fallback

## Review Fingerprint
artifact_sha256: <exact Artifact bytes>
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
```

Omit empty severity sections; write `None` for no evidence gaps. Name an issue number/URL when
reviewing one. Do not post, edit, close, or comment unless the user explicitly authorizes that
mutation.

Before emitting, confirm both inventories are complete and balanced, every factual conclusion has
applicable evidence or an exact gap, every delta passed its own reuse and validity check, the verdict
follows precedence, and the review remained non-mutating.
