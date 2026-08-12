"""Integrity-checked artifact I/O and strict experiment resume support."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO, Iterator, Mapping

MANIFEST_SCHEMA_VERSION = 1
_CHUNK_SIZE = 1024 * 1024


class ArtifactError(RuntimeError):
    """Base class for artifact state errors."""


class ArtifactIntegrityError(ArtifactError):
    """An artifact does not match its recorded checksum or size."""


class ResumeMismatchError(ArtifactError):
    """Existing state is not an exact match for the requested run."""


def sha256_file(path: str | Path) -> str:
    """Return the lowercase SHA-256 digest of a file's exact bytes."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(_CHUNK_SIZE), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    """Encode JSON deterministically for fingerprints and manifests."""
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )


def sha256_json(value: Any) -> str:
    """Return the SHA-256 digest of the canonical JSON representation."""
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _fsync_directory(path: Path) -> None:
    """Best-effort directory sync so a rename survives a power loss."""
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@contextmanager
def atomic_output_path(destination: str | Path) -> Iterator[Path]:
    """Yield a same-directory temporary path and atomically publish it.

    The destination is replaced only after the context exits successfully.
    Callers must close files opened on the yielded path before returning.
    """
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        yield temporary
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def atomic_write_bytes(destination: str | Path, payload: bytes) -> None:
    """Atomically replace a file with exact bytes."""
    with atomic_output_path(destination) as temporary:
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())


def atomic_write_text(
    destination: str | Path, text: str, *, encoding: str = "utf-8"
) -> None:
    """Atomically replace a text file."""
    atomic_write_bytes(destination, text.encode(encoding))


def atomic_write_json(destination: str | Path, value: Any) -> None:
    """Atomically write canonical UTF-8 JSON with a trailing newline."""
    atomic_write_bytes(destination, canonical_json_bytes(value))


def copy_stream_and_hash(source: BinaryIO, destination: BinaryIO) -> tuple[str, int]:
    """Copy a binary stream while returning its SHA-256 and byte count."""
    digest = hashlib.sha256()
    size = 0
    while block := source.read(_CHUNK_SIZE):
        destination.write(block)
        digest.update(block)
        size += len(block)
    return digest.hexdigest(), size


def _logical_path(path: Path, base: Path) -> str:
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def file_record(path: str | Path, *, relative_to: str | Path) -> dict[str, Any]:
    """Build a portable checksum record for an existing regular file."""
    path = Path(path)
    if not path.is_file():
        raise ArtifactIntegrityError(f"artifact is missing or not a file: {path}")
    return {
        "path": _logical_path(path, Path(relative_to)),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _records(
    files: Mapping[str, str | Path], *, relative_to: str | Path
) -> dict[str, dict[str, Any]]:
    return {
        name: file_record(path, relative_to=relative_to)
        for name, path in sorted(files.items())
    }


def build_manifest(
    *,
    stage: str,
    run_fingerprint: str,
    inputs: Mapping[str, str | Path],
    outputs: Mapping[str, str | Path],
    parameters: Mapping[str, Any] | None = None,
    relative_to: str | Path,
) -> dict[str, Any]:
    """Build a manifest from files after every output is fully written."""
    if not stage:
        raise ValueError("stage must be non-empty")
    if not run_fingerprint:
        raise ValueError("run_fingerprint must be non-empty")
    if not outputs:
        raise ValueError("a manifest must commit at least one output")
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "stage": stage,
        "run_fingerprint": run_fingerprint,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "parameters": dict(parameters or {}),
        "inputs": _records(inputs, relative_to=relative_to),
        "outputs": _records(outputs, relative_to=relative_to),
    }


def write_manifest(
    manifest_path: str | Path,
    *,
    stage: str,
    run_fingerprint: str,
    inputs: Mapping[str, str | Path],
    outputs: Mapping[str, str | Path],
    parameters: Mapping[str, Any] | None = None,
    relative_to: str | Path | None = None,
) -> dict[str, Any]:
    """Atomically publish a manifest as the final artifact-set commit marker."""
    manifest_path = Path(manifest_path)
    base = Path(relative_to) if relative_to is not None else manifest_path.parent
    manifest = build_manifest(
        stage=stage,
        run_fingerprint=run_fingerprint,
        inputs=inputs,
        outputs=outputs,
        parameters=parameters,
        relative_to=base,
    )
    atomic_write_json(manifest_path, manifest)
    return manifest


def load_manifest(manifest_path: str | Path) -> dict[str, Any]:
    """Load and minimally validate a manifest mapping."""
    manifest_path = Path(manifest_path)
    try:
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ArtifactIntegrityError(f"invalid manifest: {manifest_path}") from error
    if not isinstance(value, dict):
        raise ArtifactIntegrityError(f"manifest must be a JSON object: {manifest_path}")
    required = {
        "schema_version",
        "stage",
        "run_fingerprint",
        "parameters",
        "inputs",
        "outputs",
    }
    if not required.issubset(value):
        missing = ", ".join(sorted(required - value.keys()))
        raise ArtifactIntegrityError(f"manifest is missing fields ({missing}): {manifest_path}")
    if value["schema_version"] != MANIFEST_SCHEMA_VERSION:
        raise ResumeMismatchError(
            f"unsupported manifest schema {value['schema_version']!r}: {manifest_path}"
        )
    return value


def _resolve_record_path(record_path: str, base: Path) -> Path:
    path = Path(record_path)
    return path if path.is_absolute() else base / path


def verify_file_record(record: Mapping[str, Any], *, relative_to: str | Path) -> Path:
    """Verify one manifest file record and return its resolved path."""
    try:
        path_value = record["path"]
        expected_size = record["bytes"]
        expected_sha256 = record["sha256"]
    except KeyError as error:
        raise ArtifactIntegrityError(f"malformed file record: missing {error.args[0]}") from error
    if not isinstance(path_value, str):
        raise ArtifactIntegrityError("malformed file record: path must be a string")
    path = _resolve_record_path(path_value, Path(relative_to))
    if not path.is_file():
        raise ArtifactIntegrityError(f"recorded artifact is missing: {path}")
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        raise ArtifactIntegrityError(
            f"artifact size mismatch for {path}: expected {expected_size}, got {actual_size}"
        )
    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise ArtifactIntegrityError(
            f"artifact SHA-256 mismatch for {path}: expected {expected_sha256}, "
            f"got {actual_sha256}"
        )
    return path


def verify_manifest_files(
    manifest: Mapping[str, Any], *, relative_to: str | Path
) -> None:
    """Verify every input and output recorded by a manifest."""
    for section in ("inputs", "outputs"):
        records = manifest.get(section)
        if not isinstance(records, dict):
            raise ArtifactIntegrityError(f"manifest field {section!r} must be an object")
        for name, record in records.items():
            if not isinstance(name, str) or not isinstance(record, dict):
                raise ArtifactIntegrityError(f"malformed {section} record {name!r}")
            verify_file_record(record, relative_to=relative_to)


def strict_resume(
    manifest_path: str | Path,
    *,
    stage: str,
    run_fingerprint: str,
    inputs: Mapping[str, str | Path],
    outputs: Mapping[str, str | Path],
    parameters: Mapping[str, Any] | None = None,
    relative_to: str | Path | None = None,
) -> bool:
    """Return whether an exact, intact artifact set can be resumed.

    A missing manifest is considered fresh only when none of the requested
    outputs exist. Any partial output, identity mismatch, changed input, or
    damaged output raises instead of silently mixing experimental states.
    """
    manifest_path = Path(manifest_path)
    base = Path(relative_to) if relative_to is not None else manifest_path.parent
    existing_outputs = [Path(path) for path in outputs.values() if Path(path).exists()]

    if not manifest_path.exists():
        if existing_outputs:
            joined = ", ".join(str(path) for path in existing_outputs)
            raise ResumeMismatchError(
                f"outputs exist without commit manifest {manifest_path}: {joined}"
            )
        return False

    manifest = load_manifest(manifest_path)
    expected_identity = {
        "stage": stage,
        "run_fingerprint": run_fingerprint,
        "parameters": dict(parameters or {}),
        "inputs": _records(inputs, relative_to=base),
    }
    for key, expected in expected_identity.items():
        if manifest.get(key) != expected:
            raise ResumeMismatchError(
                f"resume mismatch in {key!r} for {manifest_path}; "
                "use a new artifact directory"
            )

    expected_output_paths = {
        name: _logical_path(Path(path), base) for name, path in sorted(outputs.items())
    }
    recorded_outputs = manifest.get("outputs")
    if not isinstance(recorded_outputs, dict):
        raise ArtifactIntegrityError("manifest field 'outputs' must be an object")
    recorded_output_paths = {
        name: record.get("path") if isinstance(record, dict) else None
        for name, record in recorded_outputs.items()
    }
    if recorded_output_paths != expected_output_paths:
        raise ResumeMismatchError(
            f"resume output set differs from {manifest_path}; use a new artifact directory"
        )

    verify_manifest_files(manifest, relative_to=base)
    return True
