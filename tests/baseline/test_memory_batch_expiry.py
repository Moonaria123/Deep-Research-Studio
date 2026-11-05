"""DR-ITER-129 memory batch expiry contract tests."""

from __future__ import annotations

import pytest

from deepresearch_core.baseline import BaselineService, create_app
from deepresearch_core.baseline.errors import BaselineError, ErrorCode
from deepresearch_core.settings import ProviderSettings

pytest.importorskip("fastapi.testclient", reason="fastapi not installed")


def _service(tmp_path) -> BaselineService:
    return BaselineService(provider=ProviderSettings(), db_path=":memory:", snapshot_root=tmp_path)


def _memory(service: BaselineService, *, layer: str = "workspace", content: str = "Batch memory") -> str:
    return service.add_memory({"layer": layer, "scope_key": "default", "content": content})["memory_id"]


def test_batch_memory_expiry_dry_run_and_apply_counts_protected_rows(tmp_path):
    service = _service(tmp_path)
    active_id = _memory(service, content="Active workspace memory")
    pinned_id = _memory(service, content="Pinned workspace memory")
    archived_id = _memory(service, content="Archived workspace memory")
    user_id = _memory(service, layer="user", content="User memory")
    service.patch_memory(pinned_id, {"pinned": True})
    service.archive_memory(archived_id)

    preview = service.batch_memory_expiry(
        {"namespace": "workspace", "workspace_key": "default", "expires_at": 1893455999, "dry_run": True}
    )

    assert preview == {
        "namespace": "workspace",
        "dry_run": True,
        "would_update": 1,
        "updated": 0,
        "skipped_pinned": 1,
        "skipped_archived": 1,
        "failed": 0,
    }
    assert service.store.get_memory(active_id)["expires_at"] is None

    result = service.batch_memory_expiry(
        {"namespace": "workspace", "workspace_key": "default", "expires_at": 1893455999, "dry_run": False}
    )

    assert result["would_update"] == 1
    assert result["updated"] == 1
    assert result["skipped_pinned"] == 1
    assert result["skipped_archived"] == 1
    assert result["failed"] == 0
    assert service.store.get_memory(active_id)["expires_at"] == 1893455999
    assert service.store.get_memory(pinned_id)["expires_at"] is None
    assert service.store.get_memory(archived_id)["expires_at"] is None
    assert service.store.get_memory(user_id)["expires_at"] is None


def test_batch_memory_expiry_clears_user_namespace_only(tmp_path):
    service = _service(tmp_path)
    workspace_id = _memory(service, content="Workspace memory")
    user_id = _memory(service, layer="user", content="User memory")
    service.patch_memory(workspace_id, {"expires_at": 1893455999})
    service.patch_memory(user_id, {"expires_at": 1893455999})

    result = service.batch_memory_expiry(
        {"namespace": "user", "user_key": "default", "expires_at": None, "dry_run": False}
    )

    assert result["updated"] == 1
    assert service.store.get_memory(workspace_id)["expires_at"] == 1893455999
    assert service.store.get_memory(user_id)["expires_at"] is None


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"namespace": "session", "expires_at": 1893455999},
        {"namespace": "workspace"},
        {"namespace": "workspace", "expires_at": "soon"},
        {"namespace": "workspace", "expires_at": -1},
        {"namespace": "workspace", "expires_at": 1893455999, "dry_run": "yes"},
        {"namespace": "workspace", "workspace_key": " ", "expires_at": 1893455999},
    ],
)
def test_batch_memory_expiry_rejects_invalid_payloads(tmp_path, payload):
    service = _service(tmp_path)

    with pytest.raises(BaselineError) as exc:
        service.batch_memory_expiry(payload)

    assert exc.value.code is ErrorCode.INVALID_REQUEST


def test_batch_memory_expiry_api_roundtrip(tmp_path):
    from fastapi.testclient import TestClient

    service = _service(tmp_path)
    client = TestClient(create_app(service))
    client.post("/api/v1/memories", json={"layer": "workspace", "scope_key": "default", "content": "Batch API"})

    preview = client.post(
        "/api/v1/memories/batch-expiry",
        json={"namespace": "workspace", "workspace_key": "default", "expires_at": 1893455999, "dry_run": True},
    )
    assert preview.status_code == 200
    assert preview.json()["would_update"] == 1
    assert preview.json()["updated"] == 0

    apply = client.post(
        "/api/v1/memories/batch-expiry",
        json={"namespace": "workspace", "workspace_key": "default", "expires_at": 1893455999, "dry_run": False},
    )
    assert apply.status_code == 200
    assert apply.json()["updated"] == 1
