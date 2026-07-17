# Protected evaluation contract (ROS-3)

ROS-3 is the minimum seal-before-read control slice built on the canonical
evidence spine. It proves that one protected outcome cannot be inspected until
the hypothesis, falsifier, evaluator, and complete ordered arm family are
immutable, and that a finished decision can be reproduced without reading the
outcome again. It does not establish model quality, binary-contract economics,
or deployment authority.

## Evidence chain

Every stage is a `research-evidence-envelope/v1` object. The domain kind and
strict payload validator are owned by `scripts/protected_evaluation.py`.

| Stage | Domain kind | Binding |
|---|---|---|
| Sealed specification | `research.protected_evaluation.sealed_spec/v1` | Experiment ID, hypothesis, falsifier, declared protected `ArtifactRef`, exact evaluator path/function, and inactive terminal rule. |
| Multiplicity family | `research.protected_evaluation.family/v1` | Exact ordered registered arms; depends on the specification. |
| Access receipt | `research.protected_evaluation.access_receipt/v1` | Conservatively records `spent_may_have_been_read`; depends on the family. |
| Evaluation reference | `research.protected_evaluation.evaluation_reference/v1` | Exact normalized input and output from `shadow_empty_set_status`; depends on the receipt. |
| Terminal decision | `research.protected_evaluation.terminal_decision/v1` | Only `no_candidate` or `inactive_only`, always `activation=false`; depends on the evaluation. |

The protected `ArtifactRef` is deliberately payload data, not an envelope
artifact. Normal envelope publication and verification authenticate direct
artifacts, so placing it in `artifacts` would read outcomes before the receipt
or reread them during resume. The specification instead binds the unprotected
evaluator source as its direct artifact. After the receipt is durably created,
`ArtifactRef.read_verified()` authenticates and returns the protected bytes in
one hardened read. Before any content read, the adapter requires the protected
path to be a stable single-link regular file and rejects lexical or hard-link
aliases to its evaluator and evidence-store read surfaces.

## Access and recovery

`publish(..., require_new=True)` uses the evidence store's atomic no-replace
publication path. An existing object and a concurrent identical winner are
both errors. Therefore only the process that creates the receipt may proceed
to the protected read.

Before a new look, the adapter performs an identity-checked metadata scan of
the flat store and validates every ROS-3 object with the same experiment ID.
The scan never recursively reads unrelated artifacts; matching lineage and
domain bindings are checked directly. Its recovery table is closed:

| Durable state | Action |
|---|---|
| Terminal exists | Validate the full lineage, recompute from the evaluation reference, and return without an outcome read. |
| Evaluation exists, terminal absent | Validate and publish the deterministic terminal without an outcome read. |
| Receipt exists, evaluation absent | `BLOCKED`: the look is ambiguously spent; never reread. |
| Conflicting or malformed descendant | Fail closed as hidden state. |
| Only the exact specification and family exist | Attempt strict-new receipt publication, then read once if this process wins. |

The protected input is canonical JSON with exactly `schema`, `survivors`,
`shadow_eligible`, and `exclusions`. Arms must be unique registered members in
sealed family order. This bounded vertical accepts only an empty
`shadow_eligible` set. Every survivor must have exactly one non-empty exclusion
reason. Unknown fields, unknown arms, coercions, and approximations are errors;
the receipt remains spent. On initial evaluation and resume, the canonical
normalized-input bytes must exactly match the byte count and SHA-256 declared
by the sealed protected `ArtifactRef`.

## Verification

Run from the repository root in the pinned Python 3.12 environment:

```bash
~/binary-algo-venv/bin/python scripts/test_protected_evaluation.py
```

The gate uses temporary synthetic repositories only. It proves receipt-before-
read ordering, one protected read under concurrency, deterministic identities,
zero-read resume, spent ambiguity, strict domain rejection, completed issue-#9
parity, and fail-closed inactive loading. It performs no fit, replay, holdout,
result write, book publication, or activation.

## Deliberate exclusions

ROS-3 adds no generic evaluator, phase framework, workflow scheduler, database,
index, service, schema dependency, corpus/RAG path, economic reducer, strategy
run, candidate activation, or refactor of issue-#9 campaign code. Those remain
separate authorities and trigger-gated capabilities in parent issue #10 and
`docs/research_os_capabilities.json`.
