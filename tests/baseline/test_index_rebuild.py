"""DR-ITER-073: synchronous index rebuild API."""

from __future__ import annotations

import pytest

from deepresearch_core.baseline import BaselineService, create_app
from deepresearch_core.baseline.errors import BaselineError, ErrorCode
from deepresearch_core.settings import ProviderSettings


def _service() -> BaselineService:
    return BaselineService(provider=ProviderSettings(), db_path=":memory:")


def _seed_unindexed_document(service: BaselineService, url: str, text: str) -> str:
    return service.store.upsert_document(url=url, title=url.rsplit("/", 1)[-1], text=text, content_type="web")


def test_rebuild_documents_scope_rechunks_existing_documents() -> None:
    service = _service()
    first = _seed_unindexed_document(service, "https://example.com/a", "Alpha evidence paragraph.")
    second = _seed_unindexed_document(service, "https://example.com/b", "Beta evidence paragraph.")

    result = service.rebuild_index({"scope": "documents", "document_ids": [first, second]})

    assert result["job_id"] is None
    assert result["status"] == "completed"
    assert result["documents"] == 2
    assert result["chunks"] >= 2
    assert result["errors"] == []
    chunks_by_doc = {row["id"]: row["chunks"] for row in service.store.list_documents()}
    assert chunks_by_doc[first] > 0
    assert chunks_by_doc[second] > 0


def test_rebuild_all_scope_uses_stored_documents_only() -> None:
    service = _service()
    _seed_unindexed_document(service, "https://example.com/a", "Alpha evidence paragraph.")
    _seed_unindexed_document(service, "https://example.com/b", "Beta evidence paragraph.")

    result = service.rebuild_index({"scope": "all"})

    assert result["status"] == "completed"
    assert result["documents"] == 2
    assert result["chunks"] >= 2


def test_rebuild_rejects_invalid_scope_unknown_ids_and_large_batches() -> None:
    service = _service()

    with pytest.raises(BaselineError) as invalid_scope:
        service.rebuild_index({"scope": "url", "url": "https://example.com"})
    assert invalid_scope.value.code is ErrorCode.INVALID_REQUEST

    with pytest.raises(BaselineError) as unknown_id:
        service.rebuild_index({"scope": "documents", "document_ids": ["doc_missing"]})
    assert unknown_id.value.code is ErrorCode.INVALID_REQUEST

    with pytest.raises(BaselineError) as too_many:
        service.rebuild_index({"scope": "documents", "document_ids": [f"doc_{i}" for i in range(51)]})
    assert too_many.value.code is ErrorCode.INVALID_REQUEST


def test_rebuild_endpoint_returns_completed_payload() -> None:
    pytest.importorskip("fastapi.testclient", reason="fastapi not installed")
    from fastapi.testclient import TestClient

    service = _service()
    doc_id = _seed_unindexed_document(service, "https://example.com/a", "Alpha evidence paragraph.")
    client = TestClient(create_app(service))

    response = client.post("/api/v1/index/rebuild", json={"scope": "documents", "document_ids": [doc_id]})

    assert response.status_code == 200
    assert response.json()["status"] == "completed"
