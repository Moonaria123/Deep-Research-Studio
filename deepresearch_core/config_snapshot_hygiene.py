"""Manifest and authorized redaction helpers for config snapshots."""

from __future__ import annotations

import copy
import hashlib
import hmac
import json
import os
import stat
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .config_snapshot import (
    MAX_CONFIG_BYTES,
    SNAPSHOT_SCHEMA,
    SnapshotError,
    atomic_write_json_document,
    build_secret_free_snapshot,
    read_json_object,
    validate_secret_free_snapshot,
)


MANIFEST_SCHEMA = "credential-snapshot-manifest/v1"
AUDIT_SCHEMA = "credential-snapshot-redaction-audit/v1"
CLEANUP_CONFIRMATION = "AUTHORIZE-151-CLEANUP"
MAX_MANIFEST_FILES = 4096
SNAPSHOT_PATTERNS = (
    "config.pre-*.json",
    "*-config.pre-*.json",
    "t51-live-*/docker/config.json",
)


class SnapshotManifestError(ValueError):
    """Raised when inventory or manifest CAS validation fails."""


class SnapshotRedactionError(SnapshotManifestError):
    """Raised after a redaction batch stops with a secret-free partial audit."""

    def __init__(self, message: str, audit: Mapping[str, Any]) -> None:
        super().__init__(message)
        self.audit = copy.deepcopy(dict(audit))


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise SnapshotManifestError("Snapshot manifest contains a non-JSON value") from error


def _document_hash(value: Mapping[str, Any], *, hash_field: str) -> str:
    body = copy.deepcopy(dict(value))
    body.pop(hash_field, None)
    return hashlib.sha256(_canonical_json_bytes(body)).hexdigest()


def _root_identity(root: Path) -> dict[str, Any]:
    try:
        if root.is_symlink():
            raise SnapshotManifestError("Snapshot root must not be a symlink")
        resolved = root.resolve(strict=True)
        info = resolved.stat()
    except FileNotFoundError as error:
        raise SnapshotManifestError("Snapshot root does not exist") from error
    if not stat.S_ISDIR(info.st_mode):
        raise SnapshotManifestError("Snapshot root must be a directory")
    return {
        "resolved": str(resolved),
        "device": int(info.st_dev),
        "inode": int(info.st_ino),
    }


def _relative_path(root: Path, candidate: Path) -> Path:
    try:
        resolved = candidate.resolve(strict=True)
        return resolved.relative_to(root.resolve(strict=True))
    except (FileNotFoundError, ValueError, OSError) as error:
        raise SnapshotManifestError("Snapshot candidate escapes the requested root") from error


def _matching_paths(root: Path) -> list[Path]:
    candidates: set[Path] = set()
    for pattern in SNAPSHOT_PATTERNS:
        candidates.update(root.rglob(pattern))
    return sorted(candidates, key=lambda path: path.as_posix().lower())


def _category(relative: Path) -> str:
    if relative.name.startswith("config.pre-"):
        return "config_pre"
    if relative.name == "config.json" and relative.parent.name == "docker":
        return "t51_docker_config_copy"
    return "scoped_config_pre"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _inspect_candidate(root: Path, path: Path) -> dict[str, Any]:
    try:
        info = path.lstat()
    except FileNotFoundError as error:
        raise SnapshotManifestError("Snapshot candidate disappeared during inventory") from error
    if stat.S_ISLNK(info.st_mode):
        raise SnapshotManifestError("Snapshot candidate must not be a symlink")
    if not stat.S_ISREG(info.st_mode):
        raise SnapshotManifestError("Snapshot candidate must be a regular file")
    if info.st_nlink > 1:
        raise SnapshotManifestError("Snapshot candidate must not be a hardlink")
    if info.st_size > MAX_CONFIG_BYTES:
        raise SnapshotManifestError("Snapshot candidate exceeds the 4 MiB limit")
    relative = _relative_path(root, path)
    try:
        read_json_object(path, label="Snapshot candidate")
    except SnapshotError as error:
        raise SnapshotManifestError(str(error)) from error
    return {
        "relative_path": relative.as_posix(),
        "category": _category(relative),
        "size": int(info.st_size),
        "pre_sha256": _file_sha256(path),
    }


def _collect_entries(root: Path) -> list[dict[str, Any]]:
    paths = _matching_paths(root)
    if len(paths) > MAX_MANIFEST_FILES:
        raise SnapshotManifestError("Snapshot inventory exceeds the 4096-file limit")
    return [_inspect_candidate(root, path) for path in paths]


def build_snapshot_manifest(root: Path) -> dict[str, Any]:
    """Build a secret-free point-in-time inventory for later CAS redaction."""

    root = Path(root)
    manifest: dict[str, Any] = {
        "schema": MANIFEST_SCHEMA,
        "generated_at": datetime.now(UTC).isoformat(),
        "root_identity": _root_identity(root),
        "patterns": list(SNAPSHOT_PATTERNS),
        "file_count": 0,
        "files": _collect_entries(root),
    }
    manifest["file_count"] = len(manifest["files"])
    manifest["manifest_sha256"] = _document_hash(manifest, hash_field="manifest_sha256")
    return manifest


def _validate_manifest_shape(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(manifest, Mapping) or manifest.get("schema") != MANIFEST_SCHEMA:
        raise SnapshotManifestError("Snapshot manifest schema is not supported")
    expected_hash = str(manifest.get("manifest_sha256") or "")
    actual_hash = _document_hash(manifest, hash_field="manifest_sha256")
    if len(expected_hash) != 64 or not hmac.compare_digest(expected_hash, actual_hash):
        raise SnapshotManifestError("Snapshot manifest hash does not match")
    files = manifest.get("files")
    if not isinstance(files, list) or not all(isinstance(item, dict) for item in files):
        raise SnapshotManifestError("Snapshot manifest file list is invalid")
    if len(files) > MAX_MANIFEST_FILES or manifest.get("file_count") != len(files):
        raise SnapshotManifestError("Snapshot manifest file count is invalid")
    names = [str(item.get("relative_path") or "") for item in files]
    if not all(names) or len(set(names)) != len(names):
        raise SnapshotManifestError("Snapshot manifest paths are invalid")
    return copy.deepcopy(files)


def verify_snapshot_manifest(root: Path, manifest: Mapping[str, Any]) -> None:
    """Require an unchanged root, file set, metadata, and pre-redaction hashes."""

    root = Path(root)
    expected_files = _validate_manifest_shape(manifest)
    if manifest.get("root_identity") != _root_identity(root):
        raise SnapshotManifestError("Snapshot manifest root identity changed")
    current_files = _collect_entries(root)
    expected_by_path = {str(item["relative_path"]): item for item in expected_files}
    current_by_path = {str(item["relative_path"]): item for item in current_files}
    if set(expected_by_path) != set(current_by_path):
        raise SnapshotManifestError("Snapshot manifest file set changed")
    for relative, expected in expected_by_path.items():
        current = current_by_path[relative]
        if current["pre_sha256"] != expected.get("pre_sha256"):
            raise SnapshotManifestError(f"Snapshot candidate hash changed: {relative}")
        if current["size"] != expected.get("size") or current["category"] != expected.get("category"):
            raise SnapshotManifestError(f"Snapshot candidate metadata changed: {relative}")


def _new_audit(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": AUDIT_SCHEMA,
        "generated_at": datetime.now(UTC).isoformat(),
        "manifest_sha256": manifest.get("manifest_sha256"),
        "root_identity": copy.deepcopy(manifest.get("root_identity")),
        "status": "in_progress",
        "processed_count": 0,
        "files": [],
    }


def _finalize_audit(audit: dict[str, Any], *, status_value: str) -> dict[str, Any]:
    audit["status"] = status_value
    audit["processed_count"] = len(audit["files"])
    audit["audit_sha256"] = _document_hash(audit, hash_field="audit_sha256")
    return audit


def redact_manifest_files(
    root: Path,
    manifest: Mapping[str, Any],
    *,
    confirm: str,
) -> dict[str, Any]:
    """Atomically redact exactly the files frozen by a verified manifest."""

    if confirm != CLEANUP_CONFIRMATION:
        raise PermissionError("Historical snapshot redaction requires exact operator authorization")
    root = Path(root)
    verify_snapshot_manifest(root, manifest)
    entries = _validate_manifest_shape(manifest)
    audit = _new_audit(manifest)
    initial_names = {str(item["relative_path"]) for item in entries}
    try:
        for item in entries:
            relative = Path(str(item["relative_path"]))
            path = root.resolve(strict=True) / relative
            inspected = _inspect_candidate(root, path)
            if inspected["pre_sha256"] != item.get("pre_sha256"):
                raise SnapshotManifestError(f"Snapshot candidate hash changed: {relative.as_posix()}")
            raw = read_json_object(path, label="Snapshot candidate")
            meta = raw.get("snapshot_meta")
            if isinstance(meta, Mapping):
                if meta.get("schema") != SNAPSHOT_SCHEMA:
                    raise SnapshotManifestError(
                        f"Snapshot candidate has an unsupported wrapper: {relative.as_posix()}"
                    )
                validate_secret_free_snapshot(raw)
                status_value = "already_secret_free"
                post_hash = inspected["pre_sha256"]
            else:
                sanitized = build_secret_free_snapshot(raw)
                validate_secret_free_snapshot(sanitized)
                atomic_write_json_document(path, sanitized, replace=True)
                status_value = "redacted"
                post_hash = _file_sha256(path)
            audit["files"].append(
                {
                    "relative_path": relative.as_posix(),
                    "category": item.get("category"),
                    "pre_sha256": item.get("pre_sha256"),
                    "post_sha256": post_hash,
                    "status": status_value,
                }
            )
        final_names = {
            _relative_path(root, path).as_posix()
            for path in _matching_paths(root)
        }
        if final_names != initial_names:
            raise SnapshotManifestError("Snapshot manifest file set changed during redaction")
    except Exception as error:
        partial = _finalize_audit(audit, status_value="partial")
        if isinstance(error, SnapshotManifestError):
            raise SnapshotRedactionError(str(error), partial) from error
        if isinstance(error, SnapshotError):
            raise SnapshotRedactionError(str(error), partial) from error
        raise SnapshotRedactionError("Snapshot redaction stopped after a safe partial batch", partial) from error
    return _finalize_audit(audit, status_value="completed")


def scan_exact_secret_values(
    paths: Iterable[Path],
    secret_values: Sequence[bytes],
) -> dict[str, Any]:
    """Count exact secret-byte hits without returning values or matching content."""

    unique = tuple(dict.fromkeys(value for value in secret_values if value))
    file_results: list[dict[str, Any]] = []
    total_hits = 0
    for raw_path in paths:
        path = Path(raw_path)
        try:
            info = path.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
            raise SnapshotManifestError("Forbidden-copy scan accepts regular files only")
        data = path.read_bytes()
        hit_count = sum(data.count(secret) for secret in unique)
        total_hits += hit_count
        file_results.append(
            {
                "path": str(path),
                "sha256": hashlib.sha256(data).hexdigest(),
                "hit_count": hit_count,
            }
        )
    return {
        "files_scanned": len(file_results),
        "hit_count": total_hits,
        "files": file_results,
    }
