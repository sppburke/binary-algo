> **SCOPE: GENERIC** (repository-wide evidence contract; no per-key empirical claims).

# Research evidence contract

ROS-1 supplies a small local integrity boundary for later research packages. It
answers three questions only:

1. Are these exactly the semantic bytes that received this identity?
2. Do the referenced repository files still have the bound bytes?
3. Do all directly referenced evidence objects still verify?

An envelope does not make its payload empirically true and grants no model,
lifecycle, deployment, purchase, or activation authority.

## Runtime objects

`scripts/evidence_store.py` defines exactly two data objects.

### `ArtifactRef`

An artifact reference has exactly:

- `path`: canonical POSIX repo-relative regular-file path;
- `bytes`: non-negative integer byte count;
- `sha256`: lowercase full SHA-256 of the file bytes.

Verification rejects absolute and alternate path spellings, `..`, symlinked
components, path escapes, missing files, and size or hash mismatches. File
timestamps are not identity.

### `EvidenceEnvelope`

An envelope has exactly:

- `schema`: `research-evidence-envelope/v1`;
- `object_id`: lowercase full SHA-256 identity;
- `kind`: semantic payload kind owned by the producing package;
- `payload`: finite canonical JSON;
- `artifacts`: ordered `ArtifactRef` list;
- `dependencies`: ordered list of prior evidence IDs.

List order is semantic. No sorting, deduplication, or inferred provenance is
performed. ROS-1 legacy objects use `research.legacy_sidecar/v1`; later
packages must define and validate their own payload kinds.

## Identity

`object_id` excludes the `object_id` field itself and binds every other
envelope field. The fixed domain is:

```text
binary-algo/research-evidence-envelope/object-id/v1
```

SHA-256 receives the UTF-8 domain followed by the canonical identity JSON,
with each component prefixed by its unsigned 64-bit big-endian byte length.
Canonical JSON is UTF-8, sorted-key, compact, finite, duplicate-key-free, and
has no trailing newline. The encoder is `manifest.canonical_json_bytes`; the
evidence module adds strict decoding and field validation.

## Storage and publication

Canonical objects live at:

```text
evidence/objects/<object-id>.json
```

The flat directory is authoritative. There is no index, cache, database, or
filesystem ordering authority. Publication requires the existing real store
directory, stages complete bytes in that directory, sets mode `0444`, fsyncs
the file, installs with atomic no-replace semantics, removes the temporary
name, and fsyncs the directory. An existing destination is accepted only as an
explicit retry after its complete bytes match and validate; it is never
overwritten or repaired.

The default publication API retains that idempotent retry behavior. A domain
that must grant a right only to the creating process may pass
`require_new=True`; the same atomic path then rejects both a pre-existing exact
object and a concurrent exact winner. ROS-3 uses this only for its conservative
semantic-access receipt.

Git does not preserve owner write bits, so checkout permissions are not
evidence identity. Runtime publication creates and authenticates mode `0444`;
verification of tracked objects relies on canonical bytes, derived identity,
and artifact hashes so a fresh clone remains portable. Editing a tracked
object invalidates its identity and is also visible to Git.

`ArtifactRef.read_verified()` performs the hardened regular-file read once,
checks the declared byte count and SHA-256, and returns those authenticated
bytes. Protected-evaluation ordering and the reason protected references stay
out of envelope `artifacts` are specified in
`docs/PROTECTED_EVALUATION.md`.

`ArtifactRef.isolated_identity()` reads metadata only and requires one stable
regular-file inode with exactly one hard link and the declared byte count. It
does not authenticate content; ROS-3 uses it only to reject pre-access aliases
before the later receipt-gated `read_verified()` call.

`inspect_store_metadata()` validates store entry shape, canonical envelope
bytes, and object identity without reading artifact targets or dependency
ancestry. It exists for pre-access discovery only; `verify-object` and
`verify-store` remain the full artifact-and-ancestry integrity gates.

## Verification

Run from the repository root with Python 3.12:

```bash
~/binary-algo-venv/bin/python scripts/evidence_store.py verify-store
~/binary-algo-venv/bin/python scripts/evidence_store.py verify-object <object-id>
~/binary-algo-venv/bin/python scripts/test_evidence_store.py
```

`verify-object` validates canonical bytes, identity, all bound artifacts, and
dependency ancestry. `verify-store` performs that operation for every public
object in deterministic ID order. Unexpected store entries fail closed.

## Legacy sidecars

The initial store contains one sidecar for each required fixture class:

- modern versioned campaign evidence;
- a heterogeneous pre-schema result;
- a frozen book and its stable files.

Each payload separates `authoritative`, `derived`, and `unmappable` facts.
Sidecars bind legacy bytes without rewriting them, assigning a missing schema,
or manufacturing relationships between historical and frozen paths. Mutable
registry projections are audited separately and are not artifact bindings.

## Capability continuity

`docs/research_os_capabilities.json` is the static machine projection of parent
issue #10. Every record has exactly `id`, `status`, `decision`, `why_valuable`,
`unlock_trigger`, and `owner_issue`. Literal states and the validated
`superseded_by:<existing-id>` pattern are checked, but ROS-1 implements no
transition engine. Deferred and triggered records are not automatic MVP work.

## Deliberate exclusions

ROS-1 adds no SQLite store, query service, graph or lineage engine, schema
framework, workflow scheduler, generated roadmap, model evaluator, corpus
ingestion, RAG layer, economics reducer, migration, or activation path.
