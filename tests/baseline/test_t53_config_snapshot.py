"""DR-ITER-151 credential-safe config snapshot contracts."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from deepresearch_core.config_snapshot import (
    MAX_CONFIG_BYTES,
    SNAPSHOT_SCHEMA,
    RESTORE_CANDIDATE_SCHEMA,
    SnapshotError,
    build_secret_free_snapshot,
    configured_secret_values,
    merge_secret_free_config,
    to_secret_free_config,
    validate_secret_free_snapshot,
    write_secret_free_restore_candidate,
    write_secret_free_snapshot,
)
from deepresearch_core.config_snapshot_hygiene import (
    CLEANUP_CONFIRMATION,
    MANIFEST_SCHEMA,
    SnapshotManifestError,
    SnapshotRedactionError,
    build_snapshot_manifest,
    redact_manifest_files,
    scan_exact_secret_values,
    verify_snapshot_manifest,
)


ROOT = Path(__file__).resolve().parents[2]
SECRET_VALUES = (
    "t53-legacy-secret",
    "t53-brave-secret",
    "t53-registry-secret",
    "t53-nested-token",
    "t53-nested-password",
)


def _raw_config() -> dict:
    return {
        "llm_api_key": SECRET_VALUES[0],
        "provider": {
            "brave_api_key": SECRET_VALUES[1],
            "llm_model": "gpt-test",
            "credential_state": "accepted",
        },
        "llm_providers": [
            {
                "id": "poe",
                "label": "Poe",
                "api_key": SECRET_VALUES[2],
                "models": [{"id": "gpt-test"}],
            }
        ],
        "nested": {
            "access_token": SECRET_VALUES[3],
            "password": SECRET_VALUES[4],
            "authorization": "Bearer t53-authorization-secret",
            "auth_header": "t53-auth-header-secret",
            "keep": "safe",
        },
        "search_provider_state": {
            "revision": 4,
            "providers": {
                "brave": {
                    "enabled": True,
                    "credential_revision": 3,
                    "last_probe": {
                        "probe_state": "available",
                        "probe_credential_revision": 3,
                        "retry_after": 60,
                    },
                }
            },
        },
    }


def _contains_secret_key(value: object) -> bool:
    if isinstance(value, dict):
        for key, item in value.items():
            if key.lower() in {
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
            }:
                return True
            if _contains_secret_key(item):
                return True
    elif isinstance(value, list):
        return any(_contains_secret_key(item) for item in value)
    return False


def test_sanitizer_is_recursive_idempotent_and_does_not_mutate_input() -> None:
    raw = _raw_config()
    before = json.loads(json.dumps(raw))

    sanitized = to_secret_free_config(raw)

    assert raw == before
    assert to_secret_free_config(sanitized) == sanitized
    assert _contains_secret_key(sanitized) is False
    assert sanitized["provider"]["llm_model"] == "gpt-test"
    assert sanitized["provider"]["credential_state"] == "accepted"
    assert sanitized["search_provider_state"]["providers"]["brave"]["credential_revision"] == 3
    blob = json.dumps(sanitized)
    assert all(secret not in blob for secret in SECRET_VALUES)
    assert "***" not in blob


def test_sanitizer_removes_vault_references_even_outside_known_key_fields() -> None:
    raw = _raw_config()
    raw["vault_metadata"] = {
        "secret_ref": "vault-v1:00000000-0000-0000-0000-000000000000",
        "label": "keep",
    }

    sanitized = to_secret_free_config(raw)

    assert sanitized["vault_metadata"] == {"label": "keep"}


def test_snapshot_hash_is_bound_only_to_sanitized_payload() -> None:
    first = build_secret_free_snapshot(_raw_config())
    changed_secret = _raw_config()
    changed_secret["provider"]["brave_api_key"] = "different-secret"
    second = build_secret_free_snapshot(changed_secret)

    assert first["snapshot_meta"]["schema"] == SNAPSHOT_SCHEMA
    assert first["snapshot_meta"]["secrets_redacted"] is True
    assert first["snapshot_meta"]["payload_sha256"] == second["snapshot_meta"]["payload_sha256"]
    assert first["config"] == second["config"]
    validate_secret_free_snapshot(first)


def test_snapshot_validation_rejects_tampering_and_secret_fields() -> None:
    snapshot = build_secret_free_snapshot(_raw_config())
    snapshot["config"]["provider"]["llm_model"] = "tampered"
    with pytest.raises(SnapshotError, match="payload hash"):
        validate_secret_free_snapshot(snapshot)

    snapshot = build_secret_free_snapshot(_raw_config())
    snapshot["config"]["provider"]["api_key"] = "must-not-survive"
    with pytest.raises(SnapshotError, match="secret field"):
        validate_secret_free_snapshot(snapshot)


def test_merge_restore_preserves_credentials_and_monotonic_revisions() -> None:
    current = _raw_config()
    current["provider"]["llm_model"] = "current-model"
    current["search_provider_state"]["revision"] = 9
    current["search_provider_state"]["providers"]["brave"]["credential_revision"] = 8
    current["llm_providers"].append(
        {"id": "current-only", "api_key": "current-only-secret", "models": []}
    )
    archived = _raw_config()
    archived["provider"]["llm_model"] = "restored-model"
    archived["search_provider_state"]["revision"] = 2
    archived["search_provider_state"]["providers"]["brave"]["credential_revision"] = 1
    snapshot = build_secret_free_snapshot(archived)

    merged = merge_secret_free_config(current, snapshot)

    assert merged["llm_api_key"] == SECRET_VALUES[0]
    assert merged["provider"]["brave_api_key"] == SECRET_VALUES[1]
    assert merged["provider"]["llm_model"] == "restored-model"
    providers = {item["id"]: item for item in merged["llm_providers"]}
    assert providers["poe"]["api_key"] == SECRET_VALUES[2]
    assert providers["current-only"]["api_key"] == "current-only-secret"
    state = merged["search_provider_state"]
    assert state["revision"] == 9
    assert state["providers"]["brave"]["credential_revision"] == 8
    assert "last_probe" not in state["providers"]["brave"]


def test_restore_candidate_file_is_secret_free(tmp_path: Path) -> None:
    current = tmp_path / "current.json"
    snapshot_path = tmp_path / "snapshot.json"
    candidate = tmp_path / "candidate.json"
    current.write_text(json.dumps(_raw_config()), encoding="utf-8")
    write_secret_free_snapshot(current, snapshot_path)

    result = write_secret_free_restore_candidate(current, snapshot_path, candidate)

    payload = json.loads(candidate.read_text(encoding="utf-8"))
    assert payload["snapshot_meta"]["schema"] == RESTORE_CANDIDATE_SCHEMA
    assert _contains_secret_key(payload["config"]) is False
    assert all(secret not in candidate.read_text(encoding="utf-8") for secret in SECRET_VALUES)
    assert result.target == candidate


def test_snapshot_writer_refuses_overwrite_same_path_and_malformed_input(tmp_path: Path) -> None:
    source = tmp_path / "config.json"
    target = tmp_path / "snapshot.json"
    source.write_text(json.dumps(_raw_config()), encoding="utf-8")
    write_secret_free_snapshot(source, target)

    with pytest.raises(SnapshotError, match="already exists"):
        write_secret_free_snapshot(source, target)
    with pytest.raises(SnapshotError, match="same file"):
        write_secret_free_snapshot(source, source)

    malformed = tmp_path / "malformed.json"
    malformed.write_text("{not-json", encoding="utf-8")
    with pytest.raises(SnapshotError, match="valid JSON"):
        write_secret_free_snapshot(malformed, tmp_path / "unused.json")


def test_snapshot_writer_rejects_hardlinked_source(tmp_path: Path) -> None:
    source = tmp_path / "config.json"
    alias = tmp_path / "alias.json"
    source.write_text(json.dumps(_raw_config()), encoding="utf-8")
    try:
        os.link(source, alias)
    except OSError as error:  # pragma: no cover - filesystem capability
        pytest.skip(f"hardlinks unavailable: {error}")

    with pytest.raises(SnapshotError, match="hardlink"):
        write_secret_free_snapshot(source, tmp_path / "snapshot.json")


def test_snapshot_writer_rejects_oversized_source(tmp_path: Path) -> None:
    source = tmp_path / "large.json"
    source.write_bytes(b"{" + b" " * MAX_CONFIG_BYTES + b"}")

    with pytest.raises(SnapshotError, match="4 MiB"):
        write_secret_free_snapshot(source, tmp_path / "snapshot.json")


def _write_historical(path: Path, *, suffix: str = "") -> None:
    payload = _raw_config()
    payload["provider"]["llm_model"] = f"model{suffix}"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_manifest_discovers_historical_patterns_and_detects_drift(tmp_path: Path) -> None:
    _write_historical(tmp_path / "config.pre-first.json", suffix="-one")
    _write_historical(tmp_path / "nested" / "t51-config.pre-second.json", suffix="-two")
    _write_historical(tmp_path / "t51-live-r48" / "docker" / "config.json", suffix="-three")

    manifest = build_snapshot_manifest(tmp_path)

    assert manifest["schema"] == MANIFEST_SCHEMA
    assert manifest["file_count"] == 3
    assert {item["category"] for item in manifest["files"]} == {
        "config_pre",
        "scoped_config_pre",
        "t51_docker_config_copy",
    }
    verify_snapshot_manifest(tmp_path, manifest)

    _write_historical(tmp_path / "config.pre-third.json", suffix="-three")
    with pytest.raises(SnapshotManifestError, match="file set changed"):
        verify_snapshot_manifest(tmp_path, manifest)


def test_manifest_detects_content_hash_drift(tmp_path: Path) -> None:
    path = tmp_path / "config.pre-first.json"
    _write_historical(path)
    manifest = build_snapshot_manifest(tmp_path)
    path.write_text(json.dumps({"safe": True}), encoding="utf-8")

    with pytest.raises(SnapshotManifestError, match="hash changed"):
        verify_snapshot_manifest(tmp_path, manifest)


def test_manifest_rejects_malformed_and_hardlinked_candidates(tmp_path: Path) -> None:
    malformed = tmp_path / "config.pre-malformed.json"
    malformed.write_text("{not-json", encoding="utf-8")
    with pytest.raises(SnapshotManifestError, match="valid JSON"):
        build_snapshot_manifest(tmp_path)

    malformed.unlink()
    original = tmp_path / "config.pre-hardlink.json"
    alias = tmp_path / "nested" / "t51-config.pre-hardlink.json"
    _write_historical(original)
    alias.parent.mkdir(parents=True)
    try:
        os.link(original, alias)
    except OSError as error:  # pragma: no cover - filesystem capability
        pytest.skip(f"hardlinks unavailable: {error}")
    with pytest.raises(SnapshotManifestError, match="hardlink"):
        build_snapshot_manifest(tmp_path)


def test_redaction_requires_exact_token_and_is_idempotent_after_fresh_manifest(tmp_path: Path) -> None:
    path = tmp_path / "config.pre-first.json"
    _write_historical(path)
    manifest = build_snapshot_manifest(tmp_path)
    before = path.read_bytes()

    with pytest.raises(PermissionError, match="authorization"):
        redact_manifest_files(tmp_path, manifest, confirm="wrong-token")
    assert path.read_bytes() == before

    audit = redact_manifest_files(tmp_path, manifest, confirm=CLEANUP_CONFIRMATION)
    text = path.read_text(encoding="utf-8")
    assert json.loads(text)["snapshot_meta"]["schema"] == SNAPSHOT_SCHEMA
    assert all(secret not in text for secret in SECRET_VALUES)
    assert all(secret not in json.dumps(audit) for secret in SECRET_VALUES)

    fresh = build_snapshot_manifest(tmp_path)
    second = redact_manifest_files(tmp_path, fresh, confirm=CLEANUP_CONFIRMATION)
    assert second["files"][0]["status"] == "already_secret_free"


def test_redaction_stops_with_secret_free_partial_audit(tmp_path: Path, monkeypatch) -> None:
    from deepresearch_core import config_snapshot_hygiene as hygiene

    first = tmp_path / "config.pre-first.json"
    second = tmp_path / "config.pre-second.json"
    _write_historical(first, suffix="-one")
    _write_historical(second, suffix="-two")
    manifest = build_snapshot_manifest(tmp_path)
    original_write = hygiene.atomic_write_json_document
    writes = 0

    def _fail_second(path, payload, *, replace):  # noqa: ANN001, ANN202
        nonlocal writes
        writes += 1
        if writes == 2:
            raise OSError("synthetic disk failure")
        return original_write(path, payload, replace=replace)

    monkeypatch.setattr(hygiene, "atomic_write_json_document", _fail_second)
    with pytest.raises(SnapshotRedactionError) as captured:
        redact_manifest_files(tmp_path, manifest, confirm=CLEANUP_CONFIRMATION)

    assert captured.value.audit["status"] == "partial"
    assert captured.value.audit["processed_count"] == 1
    assert json.loads(first.read_text(encoding="utf-8"))["snapshot_meta"]["schema"] == SNAPSHOT_SCHEMA
    assert "snapshot_meta" not in json.loads(second.read_text(encoding="utf-8"))
    assert all(secret not in json.dumps(captured.value.audit) for secret in SECRET_VALUES)


def test_exact_scan_reports_counts_without_secret_values(tmp_path: Path) -> None:
    forbidden = tmp_path / "artifact.bin"
    forbidden.write_bytes(b"prefix-" + SECRET_VALUES[1].encode() + b"-suffix")
    secrets = configured_secret_values(_raw_config())

    result = scan_exact_secret_values([forbidden], secrets)

    assert result["hit_count"] == 1
    assert SECRET_VALUES[1] not in json.dumps(result)
    forbidden.write_bytes(b"safe")
    assert scan_exact_secret_values([forbidden], secrets)["hit_count"] == 0










def test_project_owned_operator_paths_do_not_raw_copy_active_config() -> None:
    command_pattern = re.compile(
        r"(?im)^\s*(?:copy-item|cp|copy|docker\s+(?:compose\s+)?cp)\b[^\n]*config\.json"
        r"|^\s*docker\s+compose\s+exec\b[^\n]*(?:cp|copy-item)\b[^\n]*config\.json"
        r"|^\s*get-content\b[^\n]*config\.json[^\n]*(?:set-content|out-file|>)"
    )
    roots = (
        ROOT / "scripts",
        ROOT / "apps" / "web" / "e2e",
        ROOT / "docker",
        ROOT / "codex",
        ROOT / "docs" / "operations",
    )
    offenders: list[str] = []
    for base in roots:
        for path in base.rglob("*"):
            if path.suffix.lower() not in {".py", ".ps1", ".sh", ".md"}:
                continue
            text = path.read_text(encoding="utf-8")
            if command_pattern.search(text):
                offenders.append(str(path.relative_to(ROOT)))
    assert offenders == []
