"""DR-ITER-079 memory PATCH contract tests."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from deepresearch_core.baseline import BaselineService, create_app
from deepresearch_core.baseline.errors import BaselineError, ErrorCode
from deepresearch_core.settings import ProviderSettings


pytest.importorskip("fastapi.testclient", reason="fastapi not installed")


def _service(tmp_path) -> BaselineService:
    return BaselineService(provider=ProviderSettings(), db_path=":memory:", snapshot_root=tmp_path)


def _memory(service: BaselineService, *, layer: str = "workspace", content: str = "Patch alpha") -> str:
    return service.add_memory({"layer": layer, "scope_key": "default", "content": content})["memory_id"]


def test_patch_memory_updates_content_salience_and_metadata(tmp_path):
    service = _service(tmp_path)
    memory_id = _memory(service, content="ObsidianAlphaToken")

    result = service.patch_memory(
        memory_id,
        {"content": "QuasarBetaToken", "salience": 2.5, "metadata": {"source": "manual_memory", "tag": "edited"}},
    )

    assert result["memory_id"] == memory_id
    assert result["library_index_warning"] is None
    assert result["memory"]["content"] == "QuasarBetaToken"
    assert result["memory"]["salience"] == 2.5
    assert result["memory"]["metadata"] == {"source": "manual_memory", "tag": "edited"}
    assert service.store.search_chunks_lexical("ObsidianAlphaToken", limit=5) == []
    assert service.store.search_chunks_lexical("QuasarBetaToken", limit=5) == []


def test_patch_memory_pin_clears_expiry_and_pinned_rejects_new_ttl(tmp_path):
    service = _service(tmp_path)
    memory_id = _memory(service, content="Durable pinned memory")

    with_expiry = service.patch_memory(memory_id, {"expires_at": 1893456000})
    assert with_expiry["memory"]["expires_at"] == 1893456000

    result = service.patch_memory(memory_id, {"pinned": True})

    assert result["memory"]["pinned"] is True
    assert result["memory"]["expires_at"] is None

    with pytest.raises(BaselineError) as exc:
        service.patch_memory(memory_id, {"expires_at": 1893456000})
    assert exc.value.code is ErrorCode.MEMORY_PINNED


def test_patch_memory_rejects_pin_and_ttl_in_same_request(tmp_path):
    service = _service(tmp_path)
    memory_id = _memory(service, content="Atomic pin contract")

    with pytest.raises(BaselineError) as exc:
        service.patch_memory(memory_id, {"pinned": True, "expires_at": 1893456000})

    assert exc.value.code is ErrorCode.MEMORY_PINNED


def test_patch_memory_can_unpin_and_set_ttl_atomically(tmp_path):
    service = _service(tmp_path)
    memory_id = _memory(service, content="Atomic unpin contract")
    service.patch_memory(memory_id, {"pinned": True})

    result = service.patch_memory(memory_id, {"pinned": False, "expires_at": 1893456000})

    assert result["memory"]["pinned"] is False
    assert result["memory"]["expires_at"] == 1893456000


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"content": "   "},
        {"salience": "high"},
        {"metadata": []},
        {"pinned": "yes"},
        {"expires_at": "soon"},
        {"expires_at": -1},
        {"layer": "user"},
        {"scope_key": "other"},
        {"kind": "workspace"},
    ],
)
def test_patch_memory_rejects_invalid_payloads(tmp_path, payload):
    service = _service(tmp_path)
    memory_id = _memory(service)

    with pytest.raises(BaselineError) as exc:
        service.patch_memory(memory_id, payload)

    assert exc.value.code is ErrorCode.INVALID_REQUEST


def test_patch_memory_rejects_unknown_archived_and_session_memories(tmp_path):
    service = _service(tmp_path)
    active_id = _memory(service)
    session_id = _memory(service, layer="session", content="Session layer")
    service.archive_memory(active_id)

    for memory_id in ("missing-memory", active_id, session_id):
        with pytest.raises(BaselineError) as exc:
            service.patch_memory(memory_id, {"content": "Should not update"})
        assert exc.value.code is ErrorCode.INVALID_REQUEST


def test_patch_memory_api_roundtrip(tmp_path):
    from fastapi.testclient import TestClient

    service = _service(tmp_path)
    client = TestClient(create_app(service))
    memory_id = client.post(
        "/api/v1/memories",
        json={"layer": "user", "scope_key": "default", "content": "API patch alpha"},
    ).json()["memory_id"]

    response = client.patch(f"/api/v1/memories/{memory_id}", json={"content": "API patch beta"})

    assert response.status_code == 200
    body = response.json()
    assert body["memory_id"] == memory_id
    assert body["library_index_warning"] is None
    assert body["memory"]["content"] == "API patch beta"


def test_patch_memory_keeps_warning_null_when_legacy_library_reindex_would_fail(tmp_path):
    service = _service(tmp_path)
    memory_id = _memory(service, content="Patch warning alpha")

    with patch.object(service._library_indexer, "index_memory", side_effect=RuntimeError("boom")):
        result = service.patch_memory(memory_id, {"content": "Patch warning beta"})

    assert result["memory"]["content"] == "Patch warning beta"
    assert result["library_index_warning"] is None
    assert service.store.search_chunks_lexical("Patch warning alpha", limit=5) == []
    assert service.store.search_chunks_lexical("Patch warning beta", limit=5) == []


def test_patch_memory_ignores_legacy_library_purge_failure_when_no_artifact_exists(tmp_path):
    service = _service(tmp_path)
    memory_id = _memory(service, content="RollbackAlphaToken")
    assert service.store.list_library_provenance(artifact_type="memory", artifact_id=memory_id) == []

    with patch.object(service.store, "_delete_library_artifact_locked", side_effect=RuntimeError("boom")):
        result = service.patch_memory(memory_id, {"content": "RollbackBetaToken"})

    row = service.store.get_memory(memory_id)
    assert row is not None
    assert result["library_index_warning"] is None
    assert row["content"] == "RollbackBetaToken"
    assert service.store.list_library_provenance(artifact_type="memory", artifact_id=memory_id) == []
    assert service.store.search_chunks_lexical("RollbackAlphaToken", limit=5) == []
    assert service.store.search_chunks_lexical("RollbackBetaToken", limit=5) == []
