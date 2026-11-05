"""Library: persisted document listing + user notes (reused memories, namespace 'note')."""

from __future__ import annotations

import pytest

from deepresearch_core.baseline import BaselineService
from deepresearch_core.baseline.adapters import Document
from deepresearch_core.baseline.errors import BaselineError, ErrorCode
from deepresearch_core.baseline.indexing import Indexer
from deepresearch_core.baseline.store import Store
from deepresearch_core.settings import ProviderSettings


def _seed_document(store: Store) -> str:
    doc_id = store.upsert_document(url="https://e.com/a", title="Doc A", text="hello world", content_type="web")
    store.add_chunk(document_id=doc_id, chunk_index=0, text="hello world")
    store.add_chunk(document_id=doc_id, chunk_index=1, text="second chunk")
    return doc_id


def test_store_list_documents_reports_chunk_counts():
    store = Store(":memory:")
    doc_id = _seed_document(store)
    docs = store.list_documents()
    assert len(docs) == 1
    assert docs[0]["id"] == doc_id
    assert docs[0]["url"] == "https://e.com/a"
    assert docs[0]["title"] == "Doc A"
    assert docs[0]["chunks"] == 2


def test_service_list_documents():
    service = BaselineService(provider=ProviderSettings(), db_path=":memory:")
    _seed_document(service.store)
    result = service.list_documents()
    assert len(result["documents"]) == 1
    assert result["documents"][0]["chunks"] == 2


def test_notes_add_and_list_roundtrip():
    service = BaselineService(provider=ProviderSettings(), db_path=":memory:")
    created = service.add_note({"content": "Remember to cite primary sources"})
    assert created["note_id"]
    notes = service.list_notes()
    assert len(notes["notes"]) == 1
    assert notes["notes"][0]["content"] == "Remember to cite primary sources"


def test_add_note_requires_content():
    service = BaselineService(provider=ProviderSettings(), db_path=":memory:")
    with pytest.raises(BaselineError) as exc:
        service.add_note({"content": "   "})
    assert exc.value.code is ErrorCode.INVALID_REQUEST


def test_add_note_rejects_oversized_content():
    service = BaselineService(provider=ProviderSettings(), db_path=":memory:")
    with pytest.raises(BaselineError) as exc:
        service.add_note({"content": "x" * 9000})
    assert exc.value.code is ErrorCode.INVALID_REQUEST


def test_service_add_note_indexes_note_for_search():
    service = BaselineService(provider=ProviderSettings(), db_path=":memory:")

    created = service.add_note({"content": "Find this note via FTS", "session_id": "s1", "run_id": "r1"})

    assert created["note_id"]
    assert service.store.search_chunks_lexical("FTS", limit=5)
    provenance = service.store.list_library_provenance(artifact_type="note", artifact_id=created["note_id"])
    assert provenance[0]["session_id"] == "s1"


def test_service_index_url_uses_service_indexer_embedder():
    class Search:
        def fetch(self, url: str) -> Document:
            return Document(url=url, title="Doc", text="semantic document body", content_type="web")

    class Embedder:
        def embed(self, texts, *, model=None):  # noqa: ANN001, ARG002
            return [[1.0, float(index + 1)] for index, _ in enumerate(texts)]

    service = BaselineService(provider=ProviderSettings(), db_path=":memory:")
    service._search = Search()
    service._indexer = Indexer(service.store, embedder=Embedder())

    result = service.index_url({"url": "https://example.com/doc"})

    assert result["embedded"] is True


# --- HTTP --------------------------------------------------------------------

pytest.importorskip("fastapi.testclient", reason="fastapi not installed")


def _client():
    from fastapi.testclient import TestClient

    from deepresearch_core.baseline import create_app

    service = BaselineService(provider=ProviderSettings(), db_path=":memory:")
    _seed_document(service.store)
    return TestClient(create_app(service))


def test_get_documents_endpoint():
    client = _client()
    res = client.get("/api/v1/index/documents")
    assert res.status_code == 200
    body = res.json()
    assert len(body["documents"]) == 1
    assert body["documents"][0]["chunks"] == 2


def test_notes_endpoints():
    client = _client()
    assert client.get("/api/v1/notes").json()["notes"] == []
    res = client.post("/api/v1/notes", json={"content": "a note"})
    assert res.status_code == 200
    assert res.json()["note_id"]
    assert client.get("/api/v1/notes").json()["notes"][0]["content"] == "a note"


def test_library_notes_endpoint_indexes_and_returns_provenance():
    client = _client()
    res = client.post(
        "/api/v1/library/notes",
        json={"content": "library endpoint note", "session_id": "session_1", "run_id": "run_1"},
    )
    assert res.status_code == 200
    note_id = res.json()["note_id"]

    provenance = client.get(f"/api/v1/library/notes/{note_id}/provenance")
    assert provenance.status_code == 200
    body = provenance.json()
    assert body["provenance"][0]["session_id"] == "session_1"
    assert "embedding" not in str(body).lower()


def test_existing_notes_endpoint_still_works():
    client = _client()
    res = client.post("/api/v1/notes", json={"content": "compat note"})
    assert res.status_code == 200
    assert client.get("/api/v1/notes").json()["notes"][0]["content"] == "compat note"
