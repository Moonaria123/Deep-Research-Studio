"""Credential-safe configuration snapshots (DR-ITER-151).

The helpers in this module operate on raw persisted JSON only. Effective
environment or secure-storage overlays must never be folded into a snapshot.
"""

from __future__ import annotations

import copy
import hashlib
import hmac
import json
import os
import stat
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping


SNAPSHOT_SCHEMA = "credential-safe-config/v1"
RESTORE_CANDIDATE_SCHEMA = "credential-safe-restore-candidate/v1"
MAX_CONFIG_BYTES = 4 * 1024 * 1024

_SECRET_FIELD_NAMES = frozenset(
    {
        "api_key",
        "llm_api_key",
        "brave_api_key",
        "firecrawl_api_key",
        "tavily_api_key",
        "exa_api_key",
        "access_token",
        "client_secret",
        "password",
        "authorization",
        "auth_header",
        "secret_ref",
    }
)


class SnapshotError(ValueError):
    """Raised when a config snapshot cannot be handled safely."""


@dataclass(frozen=True)
class SnapshotResult:
    """Secret-free metadata for one snapshot write."""

    source: Path
    target: Path
    schema: str
    redacted_field_count: int
    payload_sha256: str
    bytes_written: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": str(self.source),
            "target": str(self.target),
            "schema": self.schema,
            "redacted_field_count": self.redacted_field_count,
            "payload_sha256": self.payload_sha256,
            "bytes_written": self.bytes_written,
        }


def _is_secret_field(name: object) -> bool:
    return isinstance(name, str) and name.strip().lower() in _SECRET_FIELD_NAMES


def _sanitize(value: Any) -> tuple[Any, int]:
    if isinstance(value, Mapping):
        output: dict[str, Any] = {}
        removed = 0
        for raw_key, raw_value in value.items():
            key = str(raw_key)
            if _is_secret_field(key):
                removed += 1
                continue
            cleaned, nested_removed = _sanitize(raw_value)
            output[key] = cleaned
            removed += nested_removed
        return output, removed
    if isinstance(value, list):
        output_list: list[Any] = []
        removed = 0
        for item in value:
            cleaned, nested_removed = _sanitize(item)
            output_list.append(cleaned)
            removed += nested_removed
        return output_list, removed
    if isinstance(value, tuple):
        output_tuple: list[Any] = []
        removed = 0
        for item in value:
            cleaned, nested_removed = _sanitize(item)
            output_tuple.append(cleaned)
            removed += nested_removed
        return output_tuple, removed
    return copy.deepcopy(value), 0


def to_secret_free_config(raw_config: Mapping[str, Any]) -> dict[str, Any]:
    """Return a deep-copied config with credential material removed."""

    if not isinstance(raw_config, Mapping):
        raise SnapshotError("Config snapshot input must be a JSON object")
    sanitized, _ = _sanitize(raw_config)
    if not isinstance(sanitized, dict):  # pragma: no cover - Mapping guarantees this
        raise SnapshotError("Config snapshot input must be a JSON object")
    _canonical_json_bytes(sanitized)
    return sanitized


def _contains_secret_field(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(
            _is_secret_field(key) or _contains_secret_field(item)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_secret_field(item) for item in value)
    return False


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
        raise SnapshotError("Config snapshot contains a non-JSON value") from error


def _payload_sha256(config: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json_bytes(config)).hexdigest()


def build_secret_free_snapshot(
    raw_config: Mapping[str, Any],
    *,
    schema: str = SNAPSHOT_SCHEMA,
) -> dict[str, Any]:
    """Build a versioned snapshot wrapper without secret-derived metadata."""

    if not isinstance(raw_config, Mapping):
        raise SnapshotError("Config snapshot input must be a JSON object")
    sanitized, removed = _sanitize(raw_config)
    if not isinstance(sanitized, dict):  # pragma: no cover - Mapping guarantees this
        raise SnapshotError("Config snapshot input must be a JSON object")
    payload_hash = _payload_sha256(sanitized)
    return {
        "snapshot_meta": {
            "schema": schema,
            "created_at": datetime.now(UTC).isoformat(),
            "secrets_redacted": True,
            "redacted_field_count": removed,
            "payload_sha256": payload_hash,
        },
        "config": sanitized,
    }


def validate_secret_free_snapshot(
    snapshot: Mapping[str, Any],
    *,
    expected_schema: str = SNAPSHOT_SCHEMA,
) -> dict[str, Any]:
    """Validate a snapshot and return a deep copy of its config payload."""

    if not isinstance(snapshot, Mapping):
        raise SnapshotError("Config snapshot must be a JSON object")
    meta = snapshot.get("snapshot_meta")
    config = snapshot.get("config")
    if not isinstance(meta, Mapping) or not isinstance(config, Mapping):
        raise SnapshotError("Config snapshot is missing metadata or config payload")
    if meta.get("schema") != expected_schema:
        raise SnapshotError("Config snapshot schema is not supported")
    if meta.get("secrets_redacted") is not True:
        raise SnapshotError("Config snapshot is not marked secret-free")
    redacted_count = meta.get("redacted_field_count")
    if isinstance(redacted_count, bool) or not isinstance(redacted_count, int) or redacted_count < 0:
        raise SnapshotError("Config snapshot redacted field count is invalid")
    if _contains_secret_field(config):
        raise SnapshotError("Config snapshot contains a secret field")
    expected_hash = str(meta.get("payload_sha256") or "")
    actual_hash = _payload_sha256(config)
    if len(expected_hash) != 64 or not hmac.compare_digest(expected_hash, actual_hash):
        raise SnapshotError("Config snapshot payload hash does not match")
    return copy.deepcopy(dict(config))


def _deep_merge(current: Any, restored: Any, *, key: str = "") -> Any:
    if key == "credential_source" and current not in (None, ""):
        return copy.deepcopy(current)
    if key == "llm_providers" and isinstance(restored, list):
        return _merge_llm_providers(current, restored)
    if isinstance(current, Mapping) and isinstance(restored, Mapping):
        merged = copy.deepcopy(dict(current))
        for child_key, child_value in restored.items():
            merged[str(child_key)] = _deep_merge(
                current.get(child_key),
                child_value,
                key=str(child_key),
            )
        return merged
    return copy.deepcopy(restored)


def _merge_llm_providers(current: Any, restored: list[Any]) -> list[Any]:
    current_items = current if isinstance(current, list) else []
    current_by_id = {
        str(item.get("id")): item
        for item in current_items
        if isinstance(item, Mapping) and str(item.get("id") or "").strip()
    }
    restored_ids: set[str] = set()
    merged: list[Any] = []
    for item in restored:
        if not isinstance(item, Mapping):
            merged.append(copy.deepcopy(item))
            continue
        provider_id = str(item.get("id") or "").strip()
        if provider_id:
            restored_ids.add(provider_id)
        merged.append(
            _deep_merge(current_by_id.get(provider_id, {}), item, key="llm_provider")
        )
    for item in current_items:
        if not isinstance(item, Mapping):
            merged.append(copy.deepcopy(item))
            continue
        provider_id = str(item.get("id") or "").strip()
        if provider_id and provider_id in restored_ids:
            continue
        merged.append(copy.deepcopy(item))
    return merged


def _nonnegative_int(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return 0
    return max(parsed, 0)


def _reset_search_provider_probe_state(
    merged: dict[str, Any],
    current: Mapping[str, Any],
    restored: Mapping[str, Any],
) -> None:
    merged_state = merged.get("search_provider_state")
    if not isinstance(merged_state, dict):
        return
    current_state = current.get("search_provider_state")
    restored_state = restored.get("search_provider_state")
    current_state = current_state if isinstance(current_state, Mapping) else {}
    restored_state = restored_state if isinstance(restored_state, Mapping) else {}
    merged_state["revision"] = max(
        _nonnegative_int(current_state.get("revision")),
        _nonnegative_int(restored_state.get("revision")),
    )
    current_providers = current_state.get("providers")
    restored_providers = restored_state.get("providers")
    current_providers = current_providers if isinstance(current_providers, Mapping) else {}
    restored_providers = restored_providers if isinstance(restored_providers, Mapping) else {}
    merged_providers = merged_state.get("providers")
    if not isinstance(merged_providers, dict):
        return
    for provider_id, provider_state in merged_providers.items():
        if not isinstance(provider_state, dict):
            continue
        old = current_providers.get(provider_id)
        archived = restored_providers.get(provider_id)
        old = old if isinstance(old, Mapping) else {}
        archived = archived if isinstance(archived, Mapping) else {}
        revision = max(
            _nonnegative_int(old.get("credential_revision")),
            _nonnegative_int(archived.get("credential_revision")),
        )
        if revision:
            provider_state["credential_revision"] = revision
        else:
            provider_state.pop("credential_revision", None)
        for transient in (
            "last_probe",
            "probe_credential_revision",
            "probe_state",
            "checked_at",
            "stale_at",
            "retry_after",
        ):
            provider_state.pop(transient, None)


def merge_secret_free_config(
    current: Mapping[str, Any],
    snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    """Merge non-secret snapshot state while retaining current credentials."""

    if not isinstance(current, Mapping):
        raise SnapshotError("Current config must be a JSON object")
    restored = validate_secret_free_snapshot(snapshot)
    merged = _deep_merge(current, restored)
    if not isinstance(merged, dict):  # pragma: no cover - current/restored are mappings
        raise SnapshotError("Merged config must be a JSON object")
    _reset_search_provider_probe_state(merged, current, restored)
    _canonical_json_bytes(merged)
    return merged


def _validate_regular_json_file(path: Path, *, label: str) -> os.stat_result:
    try:
        info = path.lstat()
    except FileNotFoundError as error:
        raise SnapshotError(f"{label} does not exist") from error
    if stat.S_ISLNK(info.st_mode):
        raise SnapshotError(f"{label} must not be a symlink")
    if not stat.S_ISREG(info.st_mode):
        raise SnapshotError(f"{label} must be a regular file")
    if info.st_nlink > 1:
        raise SnapshotError(f"{label} must not be a hardlink")
    if info.st_size > MAX_CONFIG_BYTES:
        raise SnapshotError(f"{label} exceeds the 4 MiB limit")
    return info


def read_json_object(path: Path, *, label: str = "Config file") -> dict[str, Any]:
    """Read one bounded regular JSON object without exposing its content in errors."""

    _validate_regular_json_file(path, label=label)
    try:
        raw = path.read_bytes()
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SnapshotError(f"{label} must contain valid JSON") from error
    if not isinstance(payload, dict):
        raise SnapshotError(f"{label} must contain a JSON object")
    return payload


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def atomic_write_json_document(
    target: Path,
    payload: Mapping[str, Any],
    *,
    replace: bool,
) -> int:
    """Write JSON with mode 0600 and an atomic same-directory publication."""

    encoded = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8") + b"\n"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink():
        raise SnapshotError("Snapshot target must not be a symlink")
    if not replace and target.exists():
        raise SnapshotError("Snapshot target already exists")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=target.parent,
    )
    temporary = Path(temporary_name)
    published = False
    try:
        os.chmod(temporary, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        if replace:
            os.replace(temporary, target)
        else:
            try:
                os.link(temporary, target)
            except FileExistsError as error:
                raise SnapshotError("Snapshot target already exists") from error
            except OSError as error:
                raise SnapshotError("Snapshot target could not be published safely") from error
            temporary.unlink()
        published = True
        try:
            os.chmod(target, 0o600)
        except OSError:
            pass
        _fsync_directory(target.parent)
        return len(encoded)
    finally:
        if not published and temporary.exists():
            temporary.unlink()


def _assert_distinct_paths(source: Path, target: Path) -> None:
    try:
        same = source.resolve(strict=True) == target.resolve(strict=False)
    except OSError as error:
        raise SnapshotError("Snapshot paths could not be resolved safely") from error
    if same:
        raise SnapshotError("Snapshot source and target must not be the same file")


def write_secret_free_snapshot(source: Path, target: Path) -> SnapshotResult:
    """Create one new secret-free snapshot without overwriting the target."""

    source = Path(source)
    target = Path(target)
    _assert_distinct_paths(source, target)
    raw = read_json_object(source, label="Snapshot source")
    snapshot = build_secret_free_snapshot(raw)
    bytes_written = atomic_write_json_document(target, snapshot, replace=False)
    meta = snapshot["snapshot_meta"]
    return SnapshotResult(
        source=source,
        target=target,
        schema=SNAPSHOT_SCHEMA,
        redacted_field_count=int(meta["redacted_field_count"]),
        payload_sha256=str(meta["payload_sha256"]),
        bytes_written=bytes_written,
    )


def write_secret_free_restore_candidate(
    current_path: Path,
    snapshot_path: Path,
    target: Path,
) -> SnapshotResult:
    """Validate an in-memory merge and persist only its secret-free projection."""

    current_path = Path(current_path)
    snapshot_path = Path(snapshot_path)
    target = Path(target)
    _assert_distinct_paths(current_path, target)
    _assert_distinct_paths(snapshot_path, target)
    current = read_json_object(current_path, label="Current config")
    snapshot = read_json_object(snapshot_path, label="Secret-free snapshot")
    merged = merge_secret_free_config(current, snapshot)
    candidate = build_secret_free_snapshot(merged, schema=RESTORE_CANDIDATE_SCHEMA)
    candidate["snapshot_meta"]["source_snapshot_payload_sha256"] = snapshot[
        "snapshot_meta"
    ]["payload_sha256"]
    bytes_written = atomic_write_json_document(target, candidate, replace=False)
    meta = candidate["snapshot_meta"]
    return SnapshotResult(
        source=snapshot_path,
        target=target,
        schema=RESTORE_CANDIDATE_SCHEMA,
        redacted_field_count=int(meta["redacted_field_count"]),
        payload_sha256=str(meta["payload_sha256"]),
        bytes_written=bytes_written,
    )


def configured_secret_values(raw_config: Mapping[str, Any]) -> tuple[bytes, ...]:
    """Extract unique configured secret bytes for an in-memory exact scan."""

    values: list[bytes] = []

    def _visit(value: Any) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                if _is_secret_field(key) and isinstance(item, str) and item:
                    encoded = item.encode("utf-8")
                    if encoded not in values:
                        values.append(encoded)
                else:
                    _visit(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                _visit(item)

    _visit(raw_config)
    return tuple(values)
