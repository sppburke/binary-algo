#!/usr/bin/env python3
"""Minimal immutable evidence objects for repository-local research records.

The store proves byte identity and direct dependency integrity only.  Domain
packages remain responsible for the meaning and authority of ``payload``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import secrets
import stat
import struct
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterator, Sequence

import manifest


REPO_ROOT = Path(__file__).resolve().parents[1]
ENVELOPE_SCHEMA = "research-evidence-envelope/v1"
LEGACY_SIDECAR_KIND = "research.legacy_sidecar/v1"
OBJECT_ID_DOMAIN = "binary-algo/research-evidence-envelope/object-id/v1"
CAPABILITY_REGISTRY_SCHEMA = "research-os-capabilities/v1"
STORE_PARTS = ("evidence", "objects")
HEX64 = re.compile(r"[0-9a-f]{64}")
CAPABILITY_FIELDS = frozenset(
    {"id", "status", "decision", "why_valuable", "unlock_trigger", "owner_issue"}
)
LITERAL_CAPABILITY_STATUSES = frozenset(
    {
        "active",
        "planned",
        "triggered_pending_decision",
        "deferred_until_trigger",
        "resolved_non_goal",
        "accepted",
        "blocked",
        "rejected_with_evidence",
    }
)


class EvidenceError(RuntimeError):
    """An evidence object, referenced artifact, or store failed verification."""


def _require_exact_fields(
    value: Any, expected: set[str] | frozenset[str], name: str
) -> None:
    if not isinstance(value, dict):
        raise EvidenceError(f"{name} must be a JSON object")
    actual = set(value)
    if actual != set(expected):
        missing = sorted(set(expected) - actual)
        unknown = sorted(actual - set(expected))
        raise EvidenceError(
            f"{name} fields differ: missing={missing} unknown={unknown}"
        )


def _validate_json_value(value: Any, *, location: str = "value") -> None:
    if value is None or type(value) in {bool, int, str}:
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise EvidenceError(f"{location} contains a non-finite number")
        return
    if type(value) is list:
        for index, item in enumerate(value):
            _validate_json_value(item, location=f"{location}[{index}]")
        return
    if type(value) is dict:
        for key, item in value.items():
            if not isinstance(key, str):
                raise EvidenceError(f"{location} contains a non-string object key")
            _validate_json_value(item, location=f"{location}.{key}")
        return
    raise EvidenceError(
        f"{location} is not an exact JSON value: {type(value).__name__}"
    )


def _canonical_json_bytes(value: Any) -> bytes:
    _validate_json_value(value)
    try:
        return manifest.canonical_json_bytes(value)
    except (TypeError, ValueError, manifest.CandidatePublicationError) as exc:
        raise EvidenceError(f"value is not canonical JSON: {exc}") from exc


def _duplicate_rejecting_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise EvidenceError(f"duplicate JSON key {key!r}")
        value[key] = item
    return value


def _reject_json_constant(value: str) -> None:
    raise EvidenceError(f"non-finite JSON constant {value!r}")


def _decode_canonical_json(raw: bytes) -> Any:
    try:
        text = raw.decode("utf-8")
        value = json.loads(
            text,
            object_pairs_hook=_duplicate_rejecting_object,
            parse_constant=_reject_json_constant,
        )
    except EvidenceError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvidenceError(f"invalid UTF-8 JSON: {exc}") from exc
    _validate_json_value(value)
    if _canonical_json_bytes(value) != raw:
        raise EvidenceError("JSON bytes are not canonical")
    return value


def _json_copy(value: Any) -> Any:
    return _decode_canonical_json(_canonical_json_bytes(value))


def _length_prefixed_digest(domain: str, components: Sequence[bytes]) -> str:
    if not isinstance(domain, str) or not domain:
        raise EvidenceError("identity domain must be a non-empty string")
    digest = hashlib.sha256()
    for item in [domain.encode("utf-8"), *components]:
        if not isinstance(item, bytes):
            raise EvidenceError("identity components must be bytes")
        digest.update(struct.pack(">Q", len(item)))
        digest.update(item)
    return digest.hexdigest()


def _is_hex64(value: Any) -> bool:
    return isinstance(value, str) and HEX64.fullmatch(value) is not None


def _canonical_relative_path(value: Any, *, name: str = "artifact path") -> str:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise EvidenceError(f"{name} must be a non-empty canonical POSIX path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or value != path.as_posix()
        or value in {".", ".."}
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise EvidenceError(f"{name} is not canonical repo-relative POSIX: {value!r}")
    return value


def _validated_repo(repo_root: str | os.PathLike[str]) -> Path:
    argument = Path(repo_root)
    if argument.is_symlink():
        raise EvidenceError(f"repository root must not be a symlink: {argument}")
    try:
        root = argument.resolve(strict=True)
    except OSError as exc:
        raise EvidenceError(f"repository root does not exist: {argument}") from exc
    if not root.is_dir():
        raise EvidenceError(f"repository root is not a directory: {root}")
    return root


def _same_inode(left: os.stat_result, right: os.stat_result) -> bool:
    return (left.st_dev, left.st_ino) == (right.st_dev, right.st_ino)


@contextmanager
def _open_directory_chain(
    root: Path, parts: Sequence[str], *, name: str
) -> Iterator[int]:
    """Hold no-follow directory descriptors through check and use."""

    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptors: list[int] = []
    bindings: list[tuple[int, str, int]] = []
    try:
        root_descriptor = os.open(root, flags)
        descriptors.append(root_descriptor)
        root_named = root.stat(follow_symlinks=False)
        if not _same_inode(os.fstat(root_descriptor), root_named):
            raise EvidenceError("repository root changed while being opened")
        for part in parts:
            parent = descriptors[-1]
            child = os.open(part, flags, dir_fd=parent)
            descriptors.append(child)
            named = os.stat(part, dir_fd=parent, follow_symlinks=False)
            if not _same_inode(os.fstat(child), named):
                raise EvidenceError(
                    f"{name} directory changed while being opened: {part}"
                )
            bindings.append((parent, part, child))
        yield descriptors[-1]
        if not _same_inode(os.fstat(root_descriptor), root.stat(follow_symlinks=False)):
            raise EvidenceError("repository root changed during operation")
        for parent, part, child in bindings:
            named = os.stat(part, dir_fd=parent, follow_symlinks=False)
            if not _same_inode(os.fstat(child), named):
                raise EvidenceError(
                    f"{name} directory changed during operation: {part}"
                )
    except EvidenceError:
        raise
    except OSError as exc:
        raise EvidenceError(
            f"cannot open {name} as a real directory chain: {exc}"
        ) from exc
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def _read_regular_at(
    parent_descriptor: int,
    filename: str,
    *,
    name: str,
    require_read_only: bool = False,
) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(filename, flags, dir_fd=parent_descriptor)
    except OSError as exc:
        raise EvidenceError(
            f"cannot open {name} as a non-symlink regular file: {filename}: {exc}"
        ) from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise EvidenceError(f"{name} is not a regular file: {filename}")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1 << 20)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    try:
        named = os.stat(filename, dir_fd=parent_descriptor, follow_symlinks=False)
    except OSError as exc:
        raise EvidenceError(f"{name} path disappeared: {filename}") from exc
    if (
        stat.S_ISLNK(named.st_mode)
        or not stat.S_ISREG(named.st_mode)
        or not _same_inode(before, after)
        or not _same_inode(after, named)
        or before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
        or before.st_ctime_ns != after.st_ctime_ns
        or after.st_size != named.st_size
        or after.st_mtime_ns != named.st_mtime_ns
        or after.st_ctime_ns != named.st_ctime_ns
    ):
        raise EvidenceError(f"{name} changed while being read: {filename}")
    if require_read_only and named.st_mode & 0o222:
        raise EvidenceError(f"{name} is writable: {filename}")
    return b"".join(chunks)


def _file_identity(root: Path, relative: str) -> tuple[int, str]:
    canonical = _canonical_relative_path(relative)
    parts = PurePosixPath(canonical).parts
    with _open_directory_chain(root, parts[:-1], name="artifact parent") as parent:
        raw = _read_regular_at(parent, parts[-1], name="artifact")
    return len(raw), hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class ArtifactRef:
    """Exact binding to one regular, non-symlink repository file."""

    path: str
    bytes: int
    sha256: str

    def __post_init__(self) -> None:
        _canonical_relative_path(self.path)
        if type(self.bytes) is not int or self.bytes < 0:
            raise EvidenceError("artifact bytes must be a non-negative integer")
        if not _is_hex64(self.sha256):
            raise EvidenceError("artifact sha256 must be lowercase 64-hex")

    @classmethod
    def from_dict(cls, value: Any) -> "ArtifactRef":
        _require_exact_fields(value, {"path", "bytes", "sha256"}, "ArtifactRef")
        return cls(path=value["path"], bytes=value["bytes"], sha256=value["sha256"])

    @classmethod
    def capture(
        cls,
        path: str,
        *,
        repo_root: str | os.PathLike[str] = REPO_ROOT,
    ) -> "ArtifactRef":
        root = _validated_repo(repo_root)
        size, digest = _file_identity(root, path)
        return cls(path=_canonical_relative_path(path), bytes=size, sha256=digest)

    def as_dict(self) -> dict[str, Any]:
        return {"path": self.path, "bytes": self.bytes, "sha256": self.sha256}

    def verify(self, *, repo_root: str | os.PathLike[str] = REPO_ROOT) -> None:
        root = _validated_repo(repo_root)
        size, digest = _file_identity(root, self.path)
        if size != self.bytes:
            raise EvidenceError(
                f"artifact size mismatch for {self.path}: expected={self.bytes} actual={size}"
            )
        if digest != self.sha256:
            raise EvidenceError(f"artifact hash mismatch for {self.path}")


def _identity_fields(
    *,
    schema: str,
    kind: str,
    payload: Any,
    artifacts: Sequence[ArtifactRef],
    dependencies: Sequence[str],
) -> dict[str, Any]:
    return {
        "schema": schema,
        "kind": kind,
        "payload": payload,
        "artifacts": [artifact.as_dict() for artifact in artifacts],
        "dependencies": list(dependencies),
    }


def _derive_object_id(
    *,
    schema: str,
    kind: str,
    payload: Any,
    artifacts: Sequence[ArtifactRef],
    dependencies: Sequence[str],
    domain: str = OBJECT_ID_DOMAIN,
) -> str:
    fields = _identity_fields(
        schema=schema,
        kind=kind,
        payload=payload,
        artifacts=artifacts,
        dependencies=dependencies,
    )
    return _length_prefixed_digest(domain, [_canonical_json_bytes(fields)])


@dataclass(frozen=True)
class EvidenceEnvelope:
    """Versioned evidence identity plus ordered byte and dependency bindings."""

    schema: str
    object_id: str
    kind: str
    payload: Any
    artifacts: tuple[ArtifactRef, ...]
    dependencies: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.schema != ENVELOPE_SCHEMA:
            raise EvidenceError(f"unsupported envelope schema {self.schema!r}")
        if not _is_hex64(self.object_id):
            raise EvidenceError("object_id must be lowercase 64-hex")
        if (
            not isinstance(self.kind, str)
            or not self.kind
            or self.kind.strip() != self.kind
            or any(ord(char) < 0x20 for char in self.kind)
        ):
            raise EvidenceError("kind must be a non-empty canonical string")
        _validate_json_value(self.payload, location="payload")
        if type(self.artifacts) is not tuple or any(
            not isinstance(item, ArtifactRef) for item in self.artifacts
        ):
            raise EvidenceError("artifacts must be an ordered ArtifactRef tuple")
        if type(self.dependencies) is not tuple or any(
            not _is_hex64(item) for item in self.dependencies
        ):
            raise EvidenceError("dependencies must be an ordered tuple of 64-hex IDs")
        self.verify_identity()

    @classmethod
    def create(
        cls,
        *,
        kind: str,
        payload: Any,
        artifacts: Sequence[ArtifactRef] = (),
        dependencies: Sequence[str] = (),
    ) -> "EvidenceEnvelope":
        payload_copy = _json_copy(payload)
        artifact_tuple = tuple(artifacts)
        dependency_tuple = tuple(dependencies)
        object_id = _derive_object_id(
            schema=ENVELOPE_SCHEMA,
            kind=kind,
            payload=payload_copy,
            artifacts=artifact_tuple,
            dependencies=dependency_tuple,
        )
        return cls(
            schema=ENVELOPE_SCHEMA,
            object_id=object_id,
            kind=kind,
            payload=payload_copy,
            artifacts=artifact_tuple,
            dependencies=dependency_tuple,
        )

    @classmethod
    def from_dict(cls, value: Any) -> "EvidenceEnvelope":
        _require_exact_fields(
            value,
            {"schema", "object_id", "kind", "payload", "artifacts", "dependencies"},
            "EvidenceEnvelope",
        )
        if type(value["artifacts"]) is not list:
            raise EvidenceError("envelope artifacts must be a JSON list")
        if type(value["dependencies"]) is not list:
            raise EvidenceError("envelope dependencies must be a JSON list")
        return cls(
            schema=value["schema"],
            object_id=value["object_id"],
            kind=value["kind"],
            payload=_json_copy(value["payload"]),
            artifacts=tuple(ArtifactRef.from_dict(item) for item in value["artifacts"]),
            dependencies=tuple(value["dependencies"]),
        )

    def identity_fields(self) -> dict[str, Any]:
        return _identity_fields(
            schema=self.schema,
            kind=self.kind,
            payload=_json_copy(self.payload),
            artifacts=self.artifacts,
            dependencies=self.dependencies,
        )

    def verify_identity(self) -> None:
        actual = _derive_object_id(
            schema=self.schema,
            kind=self.kind,
            payload=self.payload,
            artifacts=self.artifacts,
            dependencies=self.dependencies,
        )
        if actual != self.object_id:
            raise EvidenceError(
                f"envelope identity mismatch: expected={self.object_id} derived={actual}"
            )

    def as_dict(self) -> dict[str, Any]:
        self.verify_identity()
        value = self.identity_fields()
        return {
            "schema": value["schema"],
            "object_id": self.object_id,
            "kind": value["kind"],
            "payload": value["payload"],
            "artifacts": value["artifacts"],
            "dependencies": value["dependencies"],
        }

    def canonical_bytes(self) -> bytes:
        return _canonical_json_bytes(self.as_dict())


def derive_object_id(envelope: EvidenceEnvelope) -> str:
    """Recompute an envelope ID without trusting its stored ``object_id``."""

    if not isinstance(envelope, EvidenceEnvelope):
        raise EvidenceError("derive_object_id requires an EvidenceEnvelope")
    return _derive_object_id(
        schema=envelope.schema,
        kind=envelope.kind,
        payload=envelope.payload,
        artifacts=envelope.artifacts,
        dependencies=envelope.dependencies,
    )


def _object_path(store: Path, object_id: str) -> Path:
    if not _is_hex64(object_id):
        raise EvidenceError("object ID must be lowercase 64-hex")
    return store / f"{object_id}.json"


def _load_object(
    store: Path, store_descriptor: int, object_id: str
) -> tuple[EvidenceEnvelope, bytes]:
    path = _object_path(store, object_id)
    raw = _read_regular_at(store_descriptor, path.name, name="evidence object")
    value = _decode_canonical_json(raw)
    envelope = EvidenceEnvelope.from_dict(value)
    if envelope.object_id != object_id:
        raise EvidenceError(f"object filename does not match embedded ID: {path}")
    return envelope, raw


def _verify_object_recursive(
    object_id: str,
    *,
    root: Path,
    store: Path,
    store_descriptor: int,
    visiting: set[str],
    verified: dict[str, EvidenceEnvelope],
) -> EvidenceEnvelope:
    if object_id in verified:
        return verified[object_id]
    if object_id in visiting:
        raise EvidenceError(f"dependency cycle at {object_id}")
    visiting.add(object_id)
    try:
        envelope, _ = _load_object(store, store_descriptor, object_id)
        for artifact in envelope.artifacts:
            artifact.verify(repo_root=root)
        for dependency in envelope.dependencies:
            _verify_object_recursive(
                dependency,
                root=root,
                store=store,
                store_descriptor=store_descriptor,
                visiting=visiting,
                verified=verified,
            )
        verified[object_id] = envelope
        return envelope
    finally:
        visiting.remove(object_id)


def verify_object(
    object_id: str,
    *,
    repo_root: str | os.PathLike[str] = REPO_ROOT,
) -> EvidenceEnvelope:
    """Verify one canonical object, every artifact, and all direct ancestry."""

    root = _validated_repo(repo_root)
    store = root.joinpath(*STORE_PARTS)
    with _open_directory_chain(root, STORE_PARTS, name="evidence store") as descriptor:
        return _verify_object_recursive(
            object_id,
            root=root,
            store=store,
            store_descriptor=descriptor,
            visiting=set(),
            verified={},
        )


def verify_store(
    *,
    repo_root: str | os.PathLike[str] = REPO_ROOT,
) -> tuple[str, ...]:
    """Verify every public object in the flat store, independent of directory order."""

    root = _validated_repo(repo_root)
    store = root.joinpath(*STORE_PARTS)
    with _open_directory_chain(root, STORE_PARTS, name="evidence store") as descriptor:
        before = os.fstat(descriptor)
        try:
            names = os.listdir(descriptor)
        except OSError as exc:
            raise EvidenceError(f"cannot list evidence store: {exc}") from exc
        object_ids: list[str] = []
        for entry_name in names:
            match = re.fullmatch(r"([0-9a-f]{64})\.json", entry_name)
            try:
                entry = os.stat(entry_name, dir_fd=descriptor, follow_symlinks=False)
            except OSError as exc:
                raise EvidenceError(
                    f"cannot inspect evidence-store entry {entry_name}: {exc}"
                ) from exc
            if (
                match is None
                or stat.S_ISLNK(entry.st_mode)
                or not stat.S_ISREG(entry.st_mode)
            ):
                raise EvidenceError(f"unexpected evidence-store entry: {entry_name}")
            object_ids.append(match.group(1))
        object_ids.sort()
        verified: dict[str, EvidenceEnvelope] = {}
        for object_id in object_ids:
            _verify_object_recursive(
                object_id,
                root=root,
                store=store,
                store_descriptor=descriptor,
                visiting=set(),
                verified=verified,
            )
        after = os.fstat(descriptor)
        if (
            before.st_mtime_ns != after.st_mtime_ns
            or before.st_ctime_ns != after.st_ctime_ns
        ):
            raise EvidenceError("evidence store changed during verification")
        return tuple(object_ids)


def _fsync_directory_fd(descriptor: int) -> None:
    try:
        os.fsync(descriptor)
    except OSError as exc:
        raise EvidenceError(f"cannot fsync evidence store: {exc}") from exc


def _entry_exists_at(parent_descriptor: int, filename: str) -> bool:
    try:
        os.stat(filename, dir_fd=parent_descriptor, follow_symlinks=False)
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise EvidenceError(
            f"cannot inspect evidence-store entry {filename}: {exc}"
        ) from exc
    return True


def _create_temp_at(store_descriptor: int, object_id: str) -> tuple[int, str]:
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    for _ in range(100):
        name = f".{object_id}.tmp-{secrets.token_hex(16)}"
        try:
            descriptor = os.open(name, flags, 0o600, dir_fd=store_descriptor)
        except FileExistsError:
            continue
        except OSError as exc:
            raise EvidenceError(f"cannot create evidence staging file: {exc}") from exc
        return descriptor, name
    raise EvidenceError("cannot allocate a unique evidence staging file")


def _authenticate_publication(
    store_descriptor: int, object_id: str, expected: bytes
) -> None:
    filename = f"{object_id}.json"
    raw = _read_regular_at(
        store_descriptor,
        filename,
        name="published evidence object",
        require_read_only=True,
    )
    if raw != expected:
        raise EvidenceError(f"published evidence object differs: {filename}")
    value = _decode_canonical_json(raw)
    envelope = EvidenceEnvelope.from_dict(value)
    if object_id != envelope.object_id:
        raise EvidenceError(f"published evidence object name differs: {filename}")


def publish(
    envelope: EvidenceEnvelope,
    *,
    repo_root: str | os.PathLike[str] = REPO_ROOT,
) -> Path:
    """Atomically publish one read-only object, or authenticate an exact retry."""

    if not isinstance(envelope, EvidenceEnvelope):
        raise EvidenceError("publish requires an EvidenceEnvelope")
    envelope.verify_identity()
    root = _validated_repo(repo_root)
    for artifact in envelope.artifacts:
        artifact.verify(repo_root=root)
    expected = envelope.canonical_bytes()
    store = root.joinpath(*STORE_PARTS)
    destination = _object_path(store, envelope.object_id)
    with _open_directory_chain(root, STORE_PARTS, name="evidence store") as store_fd:
        verified: dict[str, EvidenceEnvelope] = {}
        for dependency in envelope.dependencies:
            _verify_object_recursive(
                dependency,
                root=root,
                store=store,
                store_descriptor=store_fd,
                visiting=set(),
                verified=verified,
            )
        filename = destination.name
        if _entry_exists_at(store_fd, filename):
            _authenticate_publication(store_fd, envelope.object_id, expected)
            _fsync_directory_fd(store_fd)
            _authenticate_publication(store_fd, envelope.object_id, expected)
            return destination

        descriptor = -1
        temporary_name: str | None = None
        directory_mutated = False
        directory_synced = False
        try:
            descriptor, temporary_name = _create_temp_at(store_fd, envelope.object_id)
            directory_mutated = True
            view = memoryview(expected)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise EvidenceError(
                        f"short write for evidence staging file {temporary_name}"
                    )
                view = view[written:]
            os.fchmod(descriptor, 0o444)
            os.fsync(descriptor)
            os.close(descriptor)
            descriptor = -1
            try:
                os.link(
                    temporary_name,
                    filename,
                    src_dir_fd=store_fd,
                    dst_dir_fd=store_fd,
                    follow_symlinks=False,
                )
            except FileExistsError:
                _authenticate_publication(store_fd, envelope.object_id, expected)
            os.unlink(temporary_name, dir_fd=store_fd)
            temporary_name = None
            _fsync_directory_fd(store_fd)
            directory_synced = True
            _authenticate_publication(store_fd, envelope.object_id, expected)
            return destination
        except OSError as exc:
            raise EvidenceError(
                f"cannot publish evidence object {destination}: {exc}"
            ) from exc
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            if temporary_name is not None:
                try:
                    os.unlink(temporary_name, dir_fd=store_fd)
                except FileNotFoundError:
                    pass
                except OSError as exc:
                    raise EvidenceError(
                        f"cannot remove evidence staging file {temporary_name}: {exc}"
                    ) from exc
            if directory_mutated and not directory_synced:
                _fsync_directory_fd(store_fd)


def validate_capability_registry(value: Any) -> tuple[str, ...]:
    """Validate the static roadmap projection without implementing transitions."""

    _require_exact_fields(value, {"schema", "capabilities"}, "capability registry")
    if value["schema"] != CAPABILITY_REGISTRY_SCHEMA:
        raise EvidenceError(
            f"unsupported capability registry schema {value['schema']!r}"
        )
    records = value["capabilities"]
    if type(records) is not list:
        raise EvidenceError("capabilities must be a JSON list")
    ids: list[str] = []
    for index, record in enumerate(records):
        _require_exact_fields(record, CAPABILITY_FIELDS, f"capability[{index}]")
        for field in CAPABILITY_FIELDS:
            if not isinstance(record[field], str) or not record[field]:
                raise EvidenceError(
                    f"capability[{index}].{field} must be a non-empty string"
                )
        ids.append(record["id"])
    if len(ids) != len(set(ids)):
        raise EvidenceError("capability registry contains duplicate IDs")
    known = set(ids)
    for record in records:
        status = record["status"]
        if status in LITERAL_CAPABILITY_STATUSES:
            continue
        if not status.startswith("superseded_by:"):
            raise EvidenceError(f"invalid capability status {status!r}")
        target = status.removeprefix("superseded_by:")
        if target not in known or target == record["id"]:
            raise EvidenceError(
                f"invalid supersession for {record['id']}: target={target!r}"
            )
    _canonical_json_bytes(value)
    return tuple(ids)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    commands = parser.add_subparsers(dest="command", required=True)
    verify_one = commands.add_parser(
        "verify-object", help="verify one object and ancestry"
    )
    verify_one.add_argument("object_id")
    commands.add_parser("verify-store", help="verify every object in the flat store")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "verify-object":
            envelope = verify_object(args.object_id, repo_root=args.repo_root)
            print(f"verified object {envelope.object_id}")
        else:
            object_ids = verify_store(repo_root=args.repo_root)
            print(f"verified store objects={len(object_ids)}")
    except EvidenceError as exc:
        print(f"evidence verification failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
