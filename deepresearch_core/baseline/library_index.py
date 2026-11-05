"""Library artifact indexing helpers.

Canonical artifacts live in their own tables or namespaces. This module creates
derived local documents/chunks so Library content can reuse the existing FTS and
optional vector indexing pipeline.
"""

from __future__ import annotations

from typing import Any

from .adapters import Document
from .indexing import Indexer
from .store import Store

JsonDict = dict[str, Any]


class LibraryIndexer:
    """Index Library artifacts as derived local documents with provenance."""

    def __init__(self, store: Store, *, indexer: Indexer) -> None:
        self._store = store
        self._indexer = indexer

    def index_note(
        self,
        *,
        note_id: str,
        content: str,
        session_id: str | None = None,
        run_id: str | None = None,
        metadata: JsonDict | None = None,
    ) -> JsonDict:
        """Index a user-authored Library note."""

        return self._index_artifact(
            artifact_type="note",
            artifact_id=note_id,
            url=f"local://library/note/{note_id}",
            title="Library note",
            text=content,
            content_type="note",
            session_id=session_id,
            run_id=run_id,
            metadata={"note_id": note_id, **(metadata or {})},
        )

    def index_memory(
        self,
        *,
        memory_id: str,
        namespace: str,
        scope_key: str,
        content: str,
        metadata: JsonDict | None = None,
    ) -> JsonDict:
        """Index a memory item as a local Library document."""

        return self._index_artifact(
            artifact_type="memory",
            artifact_id=memory_id,
            url=f"local://library/memory/{memory_id}",
            title=f"{namespace} memory",
            text=content,
            content_type="memory",
            metadata={
                "memory_id": memory_id,
                "namespace": namespace,
                "scope_key": scope_key,
                **(metadata or {}),
            },
        )

    def index_report(
        self,
        *,
        report_id: str,
        title: str,
        body_md: str,
        mode: str,
        session_id: str | None = None,
        run_id: str | None = None,
        metadata: JsonDict | None = None,
    ) -> JsonDict:
        """Index a report body as a local Library document."""

        return self._index_artifact(
            artifact_type="report",
            artifact_id=report_id,
            url=f"local://library/report/{report_id}",
            title=title or "Library report",
            text=body_md,
            content_type="report",
            session_id=session_id,
            run_id=run_id,
            metadata={"report_id": report_id, "mode": mode, **(metadata or {})},
        )

    def _index_artifact(
        self,
        *,
        artifact_type: str,
        artifact_id: str,
        url: str,
        title: str,
        text: str,
        content_type: str,
        session_id: str | None = None,
        run_id: str | None = None,
        metadata: JsonDict | None = None,
    ) -> JsonDict:
        safe_metadata = {"artifact_type": artifact_type, "artifact_id": artifact_id, **(metadata or {})}
        stats = self._indexer.index_document(
            Document(
                url=url,
                title=title,
                text=text,
                content_type=content_type,
                metadata=safe_metadata,
            )
        )
        document_id = str(stats["document_id"])
        self._store.add_library_provenance(
            artifact_type=artifact_type,
            artifact_id=artifact_id,
            session_id=session_id,
            run_id=run_id,
            document_id=document_id,
            url=url,
            metadata=safe_metadata,
        )
        return {
            "artifact_type": artifact_type,
            "artifact_id": artifact_id,
            "document_id": document_id,
            "chunks": stats.get("chunks", 0),
            "embedded": bool(stats.get("embedded")),
        }
