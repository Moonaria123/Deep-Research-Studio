"""DR-ITER-060 T8a: Library indexing + provenance contracts."""

from __future__ import annotations

import json

from deepresearch_core.baseline.indexing import Indexer
from deepresearch_core.baseline.library_index import LibraryIndexer
from deepresearch_core.baseline.store import Store


class _FakeEmbedder:
    def embed(self, texts, *, model=None):  # noqa: ANN001, ARG002
        return [[float(len(text)), 1.0] for text in texts]


def test_library_schema_is_created_for_new_store():
    store = Store(":memory:")
    tables = {
        row["name"]
        for row in store._conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }
    assert "library_provenance" in tables
    assert "reports" in tables
    assert "page_snapshots" in tables


def test_library_provenance_roundtrip_by_artifact_and_session():
    store = Store(":memory:")
    provenance_id = store.add_library_provenance(
        artifact_type="note",
        artifact_id="note_1",
        session_id="session_1",
        run_id="run_1",
        document_id="doc_1",
        url="local://library/note/note_1",
        metadata={"source": "manual"},
    )
    assert provenance_id

    by_artifact = store.list_library_provenance(artifact_type="note", artifact_id="note_1")
    assert len(by_artifact) == 1
    assert by_artifact[0]["session_id"] == "session_1"
    assert by_artifact[0]["metadata"]["source"] == "manual"

    by_session = store.list_library_artifacts_for_session("session_1")
    assert by_session[0]["artifact_id"] == "note_1"


def test_library_indexer_indexes_note_as_local_document_with_provenance():
    store = Store(":memory:")
    library = LibraryIndexer(store, indexer=Indexer(store, embedder=_FakeEmbedder()))

    result = library.index_note(
        note_id="note_1",
        content="Cite primary filings before summaries.",
        session_id="session_1",
        run_id="run_1",
        metadata={"source": "manual"},
    )

    assert result["artifact_type"] == "note"
    assert result["document_id"]
    assert result["embedded"] is True
    doc = store.get_document(result["document_id"])
    assert doc is not None
    assert doc["url"] == "local://library/note/note_1"
    assert doc["content_type"] == "note"
    assert json.loads(doc["metadata"])["note_id"] == "note_1"
    assert store.search_chunks_lexical("filings", limit=5)
    provenance = store.list_library_provenance(artifact_type="note", artifact_id="note_1")
    assert provenance[0]["run_id"] == "run_1"


def test_library_indexer_degrades_to_fts_without_embedder():
    store = Store(":memory:")
    library = LibraryIndexer(store, indexer=Indexer(store))

    result = library.index_note(note_id="note_2", content="Local-only library search")

    assert result["embedded"] is False
    assert store.search_chunks_lexical("library", limit=5)
