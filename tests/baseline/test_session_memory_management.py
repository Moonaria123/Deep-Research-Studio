"""T43 Session-layer Memory management listing and lifecycle boundaries."""

from __future__ import annotations

import pytest

from deepresearch_core.baseline import BaselineService, create_app
from deepresearch_core.baseline.errors import BaselineError, ErrorCode
from deepresearch_core.settings import ProviderSettings


def _service() -> BaselineService:
    return BaselineService(provider=ProviderSettings(), db_path=":memory:")


def _session_memory(service: BaselineService, session_id: str, content: str) -> str:
    return service.add_memory(
        {"layer": "session", "scope_key": session_id, "content": content}
    )["memory_id"]


def test_session_memory_listing_groups_active_and_archived_without_recall_touch():
    service = _service()
    older = service.store.create_session(title="Older active", workspace="workspace-a")
    newer = service.store.create_session(title="Newer active", workspace="workspace-a")
    archived = service.store.create_session(title="Archived research", workspace="workspace-a")
    older_memory = _session_memory(service, older, "older memory")
    newer_memory = _session_memory(service, newer, "newer memory")
    archived_memory = _session_memory(service, archived, "archived session memory")
    service.store.add_message(newer, "assistant", "make newer most recent")
    service.archive_session(archived)
    before = service.store.get_memory(newer_memory)["last_used_at"]

    active = service.list_session_memories(
        {"workspace_key": "workspace-a", "limit": 10, "offset": 0}
    )

    assert [group["session_id"] for group in active["sessions"]] == [newer, older]
    assert active["total"] == 2
    assert active["truncated"] is False
    assert service.store.get_memory(newer_memory)["last_used_at"] == before

    all_groups = service.list_session_memories(
        {
            "workspace_key": "workspace-a",
            "include_archived_sessions": True,
            "limit": 10,
        }
    )
    assert [group["session_id"] for group in all_groups["sessions"]] == [newer, older, archived]
    assert {older_memory, newer_memory, archived_memory} == {
        memory["id"] for group in all_groups["sessions"] for memory in group["memories"]
    }


def test_session_memory_search_and_archived_memory_filtering():
    service = _service()
    alpha = service.store.create_session(title="Alpha topic")
    beta = service.store.create_session(title="Beta topic")
    alpha_id = _session_memory(service, alpha, "shared evidence")
    beta_id = _session_memory(service, beta, "quasar needle")
    service.archive_memory(alpha_id)

    by_title = service.list_session_memories({"query": "beta"})
    by_content = service.list_session_memories({"query": "quasar"})
    hidden_archived = service.list_session_memories({"session_id": alpha})
    shown_archived = service.list_session_memories(
        {"session_id": alpha, "include_archived_memories": True}
    )

    assert [group["session_id"] for group in by_title["sessions"]] == [beta]
    assert [group["session_id"] for group in by_content["sessions"]] == [beta]
    assert by_content["sessions"][0]["memories"][0]["id"] == beta_id
    assert hidden_archived["sessions"] == []
    assert shown_archived["sessions"][0]["memories"][0]["archived_at"] is not None


def test_session_memory_supports_only_archive_restore_delete_lifecycle():
    service = _service()
    session_id = service.store.create_session(title="Lifecycle")
    memory_id = _session_memory(service, session_id, "session lifecycle")

    assert service.archive_memory(memory_id) == {"memory_id": memory_id, "archived": True}
    assert service.restore_memory(memory_id)["archived"] is False
    with pytest.raises(BaselineError) as exc:
        service.patch_memory(memory_id, {"pinned": True})
    assert exc.value.code is ErrorCode.INVALID_REQUEST
    assert service.delete_memory(memory_id) == {"memory_id": memory_id, "deleted": True}


def test_session_memory_api_returns_bounded_grouped_shape():
    pytest.importorskip("fastapi.testclient", reason="fastapi not installed")
    from fastapi.testclient import TestClient

    service = _service()
    session_id = service.store.create_session(title="API session")
    memory_id = _session_memory(service, session_id, "API memory")
    client = TestClient(create_app(service))

    response = client.get(
        "/api/v1/memories/session",
        params={"workspace_key": "default", "limit": 20, "offset": 0},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 1
    assert body["limit"] == 20
    assert body["offset"] == 0
    assert body["sessions"][0]["session_title"] == "API session"
    assert body["sessions"][0]["memories"][0]["id"] == memory_id
