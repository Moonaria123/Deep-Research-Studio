"""T46 Memory TTL sweep and citable archived-memory contracts."""

from __future__ import annotations

import time

from deepresearch_core.baseline import BaselineService, create_app
from deepresearch_core.baseline.memory import MemoryEngine, MemoryLayer
from deepresearch_core.baseline.store import Store
from deepresearch_core.settings import ProviderSettings


def _add_memory(store: Store, content: str, *, namespace: str = "workspace") -> str:
    return store.add_memory(namespace=namespace, scope_key="default", content=content)


def test_expiry_sweep_is_bounded_stable_and_idempotent():
    store = Store(":memory:")
    now = time.time()
    first = _add_memory(store, "expired first")
    second = _add_memory(store, "expired second")
    future = _add_memory(store, "future memory")
    pinned = _add_memory(store, "pinned expired")
    store.update_memory(first, expires_at=now - 20, update_expires_at=True)
    store.update_memory(second, expires_at=now - 10, update_expires_at=True)
    store.update_memory(future, expires_at=now + 100, update_expires_at=True)
    store.update_memory(pinned, pinned=True, expires_at=now - 30, update_pinned=True, update_expires_at=True)

    batch_one = store.sweep_expired_memories(now=now, limit=1)
    batch_two = store.sweep_expired_memories(now=now, limit=10)
    final = store.sweep_expired_memories(now=now, limit=10)

    assert batch_one["memory_ids"] == [first]
    assert batch_one["remaining"] == 1
    assert batch_two["memory_ids"] == [second]
    assert final["archived"] == 0
    assert store.get_memory(future)["archived_at"] is None
    assert store.get_memory(pinned)["archived_at"] is None
    store.close()


def test_background_recall_sweeps_expired_but_evidence_recall_keeps_archived():
    store = Store(":memory:")
    engine = MemoryEngine(store)
    memory_id = engine.remember(
        MemoryLayer.WORKSPACE,
        "default",
        "T46 archive evidence marker",
    )
    store.update_memory(memory_id, expires_at=time.time() - 1, update_expires_at=True)

    assert engine.recall(workspace_key="default", user_key="", query="archive evidence marker") == []
    row = store.get_memory(memory_id)
    assert row is not None and row["archived_at"] is not None

    evidence = engine.recall_evidence(
        workspace_key="default",
        user_key="",
        query="archive evidence marker",
        limit=5,
    )

    assert [item.artifact_id for item in evidence] == [memory_id]
    assert evidence[0].origin == "memory"
    assert evidence[0].source_status == "archived"
    assert evidence[0].url == ""
    assert engine.recall_evidence(workspace_key="default", user_key="", query="unrelated zircon", limit=5) == []
    store.close()


def test_cleanup_api_sweeps_requested_scope(tmp_path):
    from fastapi.testclient import TestClient

    service = BaselineService(provider=ProviderSettings(), db_path=":memory:", snapshot_root=tmp_path)
    client = TestClient(create_app(service))
    memory_id = service.add_memory(
        {"layer": "workspace", "scope_key": "default", "content": "cleanup marker"}
    )["memory_id"]
    service.patch_memory(memory_id, {"expires_at": time.time() - 1})

    first = client.post(
        "/api/v1/memories/cleanup",
        json={"workspace_key": "default", "user_key": "default", "limit": 100},
    )
    second = client.post(
        "/api/v1/memories/cleanup",
        json={"workspace_key": "default", "user_key": "default", "limit": 100},
    )

    assert first.status_code == 200
    assert first.json()["archived"] == 1
    assert first.json()["memory_ids"] == [memory_id]
    assert second.status_code == 200
    assert second.json()["archived"] == 0


def test_cleanup_api_supports_one_explicit_namespace_scope(tmp_path):
    from fastapi.testclient import TestClient

    service = BaselineService(provider=ProviderSettings(), db_path=":memory:", snapshot_root=tmp_path)
    client = TestClient(create_app(service))
    workspace_id = service.add_memory(
        {"layer": "workspace", "scope_key": "default", "content": "workspace expired"}
    )["memory_id"]
    user_id = service.add_memory(
        {"layer": "user", "scope_key": "team-a", "content": "user expired"}
    )["memory_id"]
    service.patch_memory(workspace_id, {"expires_at": time.time() - 1})
    service.patch_memory(user_id, {"expires_at": time.time() - 1})

    response = client.post(
        "/api/v1/memories/cleanup",
        json={"namespace": "user", "scope_key": "team-a", "limit": 100},
    )

    assert response.status_code == 200
    assert response.json()["memory_ids"] == [user_id]
    assert service.store.get_memory(workspace_id)["archived_at"] is None
    assert service.store.get_memory(user_id)["archived_at"] is not None


def test_cleanup_api_rejects_ambiguous_targeting_fields(tmp_path):
    from fastapi.testclient import TestClient

    service = BaselineService(provider=ProviderSettings(), db_path=":memory:", snapshot_root=tmp_path)
    client = TestClient(create_app(service))

    response = client.post(
        "/api/v1/memories/cleanup",
        json={"namespace": "workspace", "scope_key": "default", "workspace_key": "other"},
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "INVALID_REQUEST"


def test_cleanup_remaining_counts_unscanned_scopes_after_global_limit(tmp_path):
    from fastapi.testclient import TestClient

    service = BaselineService(provider=ProviderSettings(), db_path=":memory:", snapshot_root=tmp_path)
    client = TestClient(create_app(service))
    for layer in ("workspace", "user"):
        memory_id = service.add_memory(
            {"layer": layer, "scope_key": "default", "content": f"{layer} expired"}
        )["memory_id"]
        service.patch_memory(memory_id, {"expires_at": time.time() - 1})

    response = client.post(
        "/api/v1/memories/cleanup",
        json={"workspace_key": "default", "user_key": "default", "limit": 1},
    )

    assert response.status_code == 200
    assert response.json()["archived"] == 1
    assert response.json()["remaining"] == 1
