"""DR-ITER-062 memory lifecycle and Library index cascade."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from deepresearch_core.baseline import BaselineService, create_app
from deepresearch_core.settings import ProviderSettings


pytest.importorskip("fastapi.testclient", reason="fastapi not installed")


def test_memory_archive_restore_delete_does_not_create_library_index(tmp_path):
    service = BaselineService(provider=ProviderSettings(), db_path=":memory:", snapshot_root=tmp_path)
    memory_id = service.add_memory(
        {"layer": "workspace", "scope_key": "default", "content": "Durable local memory alpha"}
    )["memory_id"]

    assert service.store.list_library_provenance(artifact_type="memory", artifact_id=memory_id) == []
    assert service.store.search_chunks_lexical("alpha", limit=5) == []

    service.archive_memory(memory_id)
    assert service.recall_memory({"workspace_key": "default", "user_key": "default"}) == []
    assert service.store.list_library_provenance(artifact_type="memory", artifact_id=memory_id) == []
    assert service.store.search_chunks_lexical("alpha", limit=5) == []

    service.restore_memory(memory_id)
    recalled = service.recall_memory({"workspace_key": "default", "user_key": "default"})
    assert recalled[0]["id"] == memory_id
    assert service.store.list_library_provenance(artifact_type="memory", artifact_id=memory_id) == []
    assert service.store.search_chunks_lexical("alpha", limit=5) == []

    service.delete_memory(memory_id)
    assert service.store.get_memory(memory_id) is None
    assert service.store.search_chunks_lexical("alpha", limit=5) == []


def test_memory_archive_ignores_legacy_library_purge_failure_when_no_artifact_exists(tmp_path):
    service = BaselineService(provider=ProviderSettings(), db_path=":memory:", snapshot_root=tmp_path)
    memory_id = service.add_memory(
        {"layer": "workspace", "scope_key": "default", "content": "Rollback archive alpha"}
    )["memory_id"]
    assert service.store.list_library_provenance(artifact_type="memory", artifact_id=memory_id) == []

    with patch.object(service.store, "_delete_library_artifact_locked", side_effect=RuntimeError("boom")):
        result = service.archive_memory(memory_id)

    assert result == {"memory_id": memory_id, "archived": True}
    assert service.store.get_memory(memory_id)["archived_at"] is not None


def test_memory_delete_ignores_legacy_library_purge_failure_when_no_artifact_exists(tmp_path):
    service = BaselineService(provider=ProviderSettings(), db_path=":memory:", snapshot_root=tmp_path)
    memory_id = service.add_memory(
        {"layer": "user", "scope_key": "default", "content": "Rollback delete beta"}
    )["memory_id"]
    assert service.store.list_library_provenance(artifact_type="memory", artifact_id=memory_id) == []

    with patch.object(service.store, "_delete_library_artifact_locked", side_effect=RuntimeError("boom")):
        result = service.delete_memory(memory_id)

    assert result == {"memory_id": memory_id, "deleted": True}
    assert service.store.get_memory(memory_id) is None


def test_archived_memory_api_lists_and_restores(tmp_path):
    from fastapi.testclient import TestClient

    service = BaselineService(provider=ProviderSettings(), db_path=":memory:", snapshot_root=tmp_path)
    client = TestClient(create_app(service))
    created = client.post(
        "/api/v1/memories",
        json={"layer": "user", "scope_key": "default", "content": "Remember audit cadence"},
    ).json()
    memory_id = created["memory_id"]

    archived = client.post(f"/api/v1/memories/{memory_id}/archive")
    assert archived.status_code == 200
    active = client.get("/api/v1/memories").json()["memories"]
    assert active == []
    archived_list = client.get("/api/v1/memories/archived").json()["memories"]
    assert archived_list[0]["id"] == memory_id
    assert archived_list[0]["archived_at"]

    restored = client.post(f"/api/v1/memories/{memory_id}/restore")
    assert restored.status_code == 200
    assert restored.json()["library_index_warning"] is None
    assert client.get("/api/v1/memories/archived").json()["memories"] == []


def test_restore_memory_keeps_warning_null_when_legacy_library_reindex_would_fail(tmp_path):
    from fastapi.testclient import TestClient

    service = BaselineService(provider=ProviderSettings(), db_path=":memory:", snapshot_root=tmp_path)
    client = TestClient(create_app(service))
    memory_id = client.post(
        "/api/v1/memories",
        json={"layer": "workspace", "scope_key": "default", "content": "Recall even when index fails"},
    ).json()["memory_id"]
    assert client.post(f"/api/v1/memories/{memory_id}/archive").status_code == 200

    with patch.object(service._library_indexer, "index_memory", side_effect=RuntimeError("index failed")):
        restored = client.post(f"/api/v1/memories/{memory_id}/restore")

    assert restored.status_code == 200
    assert restored.json()["library_index_warning"] is None
    recalled = client.get(
        "/api/v1/memories",
        params={"workspace_key": "default", "user_key": "default", "query": "index fails"},
    ).json()["memories"]
    assert {row["id"] for row in recalled} == {memory_id}


def test_pinned_memory_archive_is_rejected_with_conflict(tmp_path):
    from fastapi.testclient import TestClient

    service = BaselineService(provider=ProviderSettings(), db_path=":memory:", snapshot_root=tmp_path)
    client = TestClient(create_app(service))
    memory_id = client.post(
        "/api/v1/memories",
        json={"layer": "workspace", "scope_key": "default", "content": "Pinned archive guard"},
    ).json()["memory_id"]
    assert client.patch(f"/api/v1/memories/{memory_id}", json={"pinned": True}).status_code == 200

    response = client.post(f"/api/v1/memories/{memory_id}/archive")

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "MEMORY_PINNED"
    assert service.store.get_memory(memory_id)["archived_at"] is None
