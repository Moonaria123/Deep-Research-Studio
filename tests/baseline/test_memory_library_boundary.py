"""T15 memory/Library boundary tests."""

from __future__ import annotations

from deepresearch_core.baseline import BaselineService, create_app
from deepresearch_core.baseline.indexing import Indexer
from deepresearch_core.baseline.library_index import LibraryIndexer
from deepresearch_core.baseline.store import Store
from deepresearch_core.settings import ProviderSettings


def test_add_memory_does_not_create_library_document(tmp_path):
    service = BaselineService(provider=ProviderSettings(), db_path=":memory:", snapshot_root=tmp_path)

    memory_id = service.add_memory(
        {"layer": "workspace", "scope_key": "default", "content": "Boundary memory alpha"}
    )["memory_id"]

    assert service.store.get_memory(memory_id)["content"] == "Boundary memory alpha"
    assert service.store.list_library_provenance(artifact_type="memory", artifact_id=memory_id) == []
    assert service.store.search_chunks_lexical("Boundary memory alpha", limit=5) == []


def test_library_documents_api_filters_local_memory_artifacts(tmp_path):
    from fastapi.testclient import TestClient

    service = BaselineService(provider=ProviderSettings(), db_path=":memory:", snapshot_root=tmp_path)
    client = TestClient(create_app(service))
    service.store.upsert_document(
        url="https://example.com/source",
        title="External source",
        text="public source",
        content_type="web",
    )
    LibraryIndexer(service.store, indexer=Indexer(service.store)).index_memory(
        memory_id="legacy_mem",
        namespace="workspace",
        scope_key="default",
        content="legacy memory",
    )

    documents = client.get("/api/v1/library/documents").json()["documents"]

    assert [doc["url"] for doc in documents] == ["https://example.com/source"]


def test_purge_memory_library_documents_dry_run_and_confirm():
    store = Store(":memory:")
    library = LibraryIndexer(store, indexer=Indexer(store))
    library.index_note(note_id="note_1", content="note detail")
    library.index_report(report_id="report_1", title="Report", body_md="report detail", mode="deep")
    library.index_memory(
        memory_id="mem_1",
        namespace="workspace",
        scope_key="default",
        content="legacy memory detail",
    )

    dry_run = store.purge_memory_library_documents(dry_run=True)

    assert dry_run["dry_run"] is True
    assert dry_run["documents"] == 1
    assert dry_run["chunks"] >= 1
    assert store.search_chunks_lexical("legacy memory", limit=5)

    purged = store.purge_memory_library_documents(dry_run=False)

    assert purged["dry_run"] is False
    assert purged["documents"] == 1
    assert store.search_chunks_lexical("legacy memory", limit=5) == []
    assert store.search_chunks_lexical("note detail", limit=5)
    assert store.search_chunks_lexical("report detail", limit=5)
    assert store.list_library_provenance(artifact_type="memory", artifact_id="mem_1") == []
