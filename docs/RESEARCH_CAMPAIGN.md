> **SCOPE: GENERIC** (campaign infrastructure only; no per-key empirical claims).

# Research campaign control plane (v1)

`scripts/research_campaign_v1.py` is the minimum reusable control plane between
the canonical evidence store and adapter-owned scientific workflows. It seals
the full ordered family before execution, runs one arm at a time, resumes exact
unprotected work after crashes, binds native result bytes without interpreting
their metrics, and publishes only an inactive terminal.

The runner does not evaluate a strategy by itself. It grants no protected-data,
certification, activation, deployment, venue, purchase, book, registry, sweep,
or result-ledger authority. Measured work remains owned by `strategy-eval`.

## Commands and Python API

Run from the repository root:

```bash
~/binary-algo-venv/bin/python scripts/research_campaign_v1.py run --spec <canonical-json-path>
~/binary-algo-venv/bin/python scripts/research_campaign_v1.py verify --campaign-id <id>
```

The public Python surface is:

```python
run_campaign(spec_path, *, repo_root=...) -> EvidenceEnvelope
verify_campaign(campaign_id, *, repo_root=...) -> EvidenceEnvelope
```

`run_campaign` holds the single repository-wide nonblocking process lock at
`logs/research_campaign/run_campaign.lock` for its complete mutating call.
`verify_campaign` is lock-free and read-only; it requires a completed family,
recomputes the pure family decision, and returns the verified terminal without
preflight, validation, or arm execution.

## Canonical input

The spec file must be finite canonical JSON with no trailing newline, schema
`research-campaign-spec-input/v1`, and exactly:

- `campaign_id`: lowercase 1–128 character slug;
- `key`: `pair`, `timeframe`, `target`, and `side`; direction uses `UP|DOWN`,
  magnitude uses `NA`, and `COMBINED` is not a result key;
- non-empty `question`, `hypothesis`, and preregistered `falsifier`;
- `data_use`: `synthetic` or `retrospective` only;
- non-empty ordered `ordered_arms` with unique `arm_id` and
  `results/json/**/*_result.json` destinations;
- for each arm, ordered unique `input_evidence_ids` plus a non-empty,
  order-preserving `data_use_basis_evidence_ids` subset;
- one exact direct `scripts/*.py` adapter path, the four fixed function names,
  and an adapter-owned `native_result_contract`;
- non-empty adapter-owned `stopping_rule`; and
- terminal decisions ordered exactly as `no_candidate`, `inactive_candidate`,
  `capability_deferred`, always with `activation=false`.

Unknown fields, coercions, noncanonical JSON, path escapes, symlinks, duplicate
arms/results/dependencies, absent evidence, `fresh_oos`, and protected access
fail before scientific workflow execution.

## Adapter contract

The exact adapter bytes are bound directly into the sealed spec. The runner
loads no registry or entry point and calls exactly:

```python
preflight_arm(*, campaign, arm, input_evidence) -> canonical JSON object
run_arm(*, campaign, arm, temporary_output_path) -> canonical receipt
validate_arm_result(*, campaign, arm, native_result_path) -> canonical receipt
evaluate_family(*, campaign, receipts, native_results) -> canonical decision
```

Campaign, arm, evidence, receipt, and native-reference arguments are recursive
read-only mappings/tuples. Output paths are `Path` objects. The adapter import
must be side-effect-free. Adapter code is trusted byte-bound code, not a
sandbox; the focused gate instruments the call order and audits repository
side effects.

Normal evidence publication authenticates the exact bytes of declared
synthetic or retrospective dependencies before preflight. This is an integrity
read only. `preflight_arm` must occur before adapter/domain semantic input
consumption or workflow execution and must not open scientific artifacts,
create output, use a network or credential, or mutate repository state. It
attests matching campaign/arm/data-use/basis identity,
`protected_access=false`, and admissibility.

`run_arm` may invoke only its known workflow and writes only its exact native
result to the campaign-owned temporary path. Both `completed` and
`not_computable` receipts require an actual native artifact.
`validate_arm_result` authenticates an already installed native result after a
durable attempt and never reruns the workflow. Both functions return
`research-campaign-adapter-receipt/v1` with exact sealed IDs, contract, input
ancestry, data-use attestation, final path, and `protected_access=false`.

`evaluate_family` receives the complete ordered receipts and native
`ArtifactRef` dictionaries. It returns
`research-campaign-family-decision/v1`, one ordered verdict per arm, an ordered
candidate subset, an allowed family decision, and reasons. It cannot change
the family, interpret authority, publish, freeze, activate, or update a ledger.

Once a campaign binds runner or adapter bytes at a path, that file is retained
unchanged. Behavioral changes use a successor versioned filename. This keeps
every older lineage and `verify-store` valid.

## Evidence chain

The domain owns four strict `research-evidence-envelope/v1` kinds:

| Kind | Binding |
|---|---|
| `research.campaign.spec/v1` | Exact normalized input; runner and adapter artifacts; ordered first-occurrence input ancestry. |
| `research.campaign.arm_attempt/v1` | Exact admitted preflight, arm index/ID and final path; depends on the spec and authorizes an unprotected retry. |
| `research.campaign.arm_result/v1` | Canonical adapter receipt and exact installed native artifact; depends on its attempt. |
| `research.campaign.terminal/v1` | Reproducible family decision, evaluator identity, candidates, data use, `activation=false`, and explicit claim limit; depends directly on all ordered results. |

The evidence store owns object identity, artifact verification, ancestry, and
immutable publication. This module owns strict campaign payload meaning and
rejects every hidden, duplicate, malformed, reordered, or conflicting campaign
descendant.

## Recovery

Arms always advance in sealed order under the global lock:

| Durable state | Behavior |
|---|---|
| No objects | Publish the exact spec before adapter import or calls. |
| Next arm has no attempt or final | Run preflight, publish the deterministic attempt, then run. |
| Final exists without its attempt | `BLOCKED` as foreign state. |
| Attempt exists and final is absent | Reverify inputs/code, remove only the exact campaign temp, and safely retry. |
| Attempt and final exist but result envelope does not | Call only `validate_arm_result`, bind the result, and never rerun. |
| Some results exist | Verify them and continue with the next ordered arm. |
| All results exist and terminal is absent | Call only the evaluator and publish the terminal. |
| Terminal exists | Verify lineage, recompute the evaluator, and execute zero arm functions. |
| Drift, malformed state, or conflicting descendants | `BLOCKED`; use an explicit successor campaign. |

Native output installation uses an atomic no-replace hard link from the
campaign-owned temporary file. Existing final bytes are never overwritten or
silently repaired.

## Verification

Run the focused and inherited gates from the repository root:

```bash
~/binary-algo-venv/bin/python scripts/test_research_campaign.py
~/binary-algo-venv/bin/python scripts/test_evidence_store.py
~/binary-algo-venv/bin/python scripts/test_protected_evaluation.py
~/binary-algo-venv/bin/python scripts/test_deriv_economics_ledger.py
~/binary-algo-venv/bin/python scripts/evidence_store.py verify-store
git diff --check
```

The focused gate uses temporary synthetic repositories and temporary adapters
only. It performs no model fit, strategy evaluation, real result/feature/replay
read, protected look, external action, or activation. It proves strict input
validation, spec-before-preflight ordering, serial concurrency, atomic output,
crash recovery and adoption, deterministic terminal replay, source/input drift
rejection, successor compatibility, protected-surface stability, and fresh-
clone evidence verification.

## Deliberate exclusions

V1 adds no production adapter, protected/fresh-OOS evaluator, common native
result schema, historical migration, scheduler, database/index, service, UI,
RAG layer, distributed executor, Rust component, or universal backtester. A
first real retrospective campaign requires its own reviewed experiment issue
and `strategy-eval` execution.
