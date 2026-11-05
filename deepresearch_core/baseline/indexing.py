"""Indexing pipeline: ingest -> chunk -> embed -> store.

Chunking strategies are selected by content type (web/pdf/code). Embeddings are
optional: when no embedding backend is available, chunks are stored without vectors
and retrieval degrades to the lexical (FTS5) channel only.
"""

from __future__ import annotations

import hashlib
import inspect
import re
from dataclasses import dataclass
from typing import Sequence

from .adapters import Document, LLMAdapter, SearchCallResult, SearchToolAdapter
from .content_sanitizer import SanitizeResult, sanitize_ingest_text
from .search_options import SearchOptions
from .store import DocumentWriteRejected, Store


@dataclass(frozen=True)
class ChunkingConfig:
    """Word-budget chunking parameters."""

    max_words: int = 220
    overlap_words: int = 40
    max_code_lines: int = 60


@dataclass(frozen=True)
class PreparedDocument:
    """Sanitized/chunked document ready for one deterministic Store commit."""

    document: Document
    sanitized: SanitizeResult
    pieces: tuple[str, ...]
    embeddings: tuple[tuple[float, ...], ...] | None = None


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def chunk_web(text: str, config: ChunkingConfig) -> list[str]:
    """Paragraph-aware chunking for web/markdown content."""

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    buffer: list[str] = []
    count = 0
    for para in paragraphs:
        words = para.split()
        if count + len(words) > config.max_words and buffer:
            chunks.append(" ".join(buffer))
            tail = " ".join(buffer).split()[-config.overlap_words :]
            buffer = list(tail)
            count = len(tail)
        buffer.extend(words)
        count += len(words)
    if buffer:
        chunks.append(" ".join(buffer))
    return chunks or ([text.strip()] if text.strip() else [])


def chunk_code(text: str, config: ChunkingConfig) -> list[str]:
    """Line-block chunking for source code."""

    lines = text.splitlines()
    if not lines:
        return []
    step = max(1, config.max_code_lines)
    return ["\n".join(lines[i : i + step]) for i in range(0, len(lines), step)]


def chunk_text(text: str, *, content_type: str = "web", config: ChunkingConfig | None = None) -> list[str]:
    """Dispatch to the chunking strategy for a content type."""

    cfg = config or ChunkingConfig()
    if content_type == "code":
        return chunk_code(text, cfg)
    # web + pdf share paragraph-aware chunking (pdf text is paragraph-extracted upstream).
    return chunk_web(text, cfg)


class Indexer:
    """Ingests documents into the store with chunking and optional embeddings."""

    def __init__(self, store: Store, *, embedder: LLMAdapter | None = None, config: ChunkingConfig | None = None) -> None:
        self._store = store
        self._embedder = embedder
        self._config = config or ChunkingConfig()

    def index_document(
        self,
        document: Document,
        *,
        adoption_run_id: str | None = None,
        revive_deleted_identity: bool = False,
        expected_deletion_marker: str | None = None,
    ) -> dict:
        """Index a single document; returns ``{document_id, chunks, embedded}``.

        External page text is sanitized before chunking (DR-ITER-052): control-character
        obfuscation, executable markup, leaked secrets, and prompt-injection directives are
        stripped so the citable corpus stays trustworthy.
        """

        prepared = self.prepare_document(document)
        return self.commit_prepared_document(
            prepared,
            adoption_run_id=adoption_run_id,
            revive_deleted_identity=revive_deleted_identity,
            expected_deletion_marker=expected_deletion_marker,
        )

    def prepare_document(
        self,
        document: Document,
        *,
        include_embeddings: bool = True,
    ) -> PreparedDocument:
        """Sanitize and chunk without Store writes, optionally preparing vectors."""

        sanitized = sanitize_ingest_text(document.text or "")
        pieces = tuple(
            chunk_text(
                sanitized.text,
                content_type=document.content_type,
                config=self._config,
            )
        )
        raw_embeddings = self._embed(pieces) if include_embeddings else None
        embeddings = (
            tuple(tuple(float(value) for value in vector) for vector in raw_embeddings)
            if raw_embeddings is not None and len(raw_embeddings) == len(pieces)
            else None
        )
        return PreparedDocument(
            document=document,
            sanitized=sanitized,
            pieces=pieces,
            embeddings=embeddings,
        )

    def commit_prepared_document(
        self,
        prepared: PreparedDocument,
        *,
        adoption_run_id: str | None = None,
        revive_deleted_identity: bool = False,
        expected_deletion_marker: str | None = None,
    ) -> dict:
        """Commit one prepared document; this is the single-writer boundary."""

        document = prepared.document
        text = prepared.sanitized.text
        chunk_rows = [
            (
                piece,
                list(prepared.embeddings[index]) if prepared.embeddings else None,
            )
            for index, piece in enumerate(prepared.pieces)
        ]
        batch_upsert = getattr(self._store, "upsert_document_with_chunks", None)
        if callable(batch_upsert):
            try:
                doc_id, chunk_ids = batch_upsert(
                    url=document.url,
                    title=document.title,
                    text=text,
                    chunks=chunk_rows,
                    content_type=document.content_type,
                    doc_hash=_hash_text(text),
                    metadata=dict(document.metadata),
                    quarantine_library_candidate=bool(adoption_run_id),
                    revive_deleted_identity=revive_deleted_identity,
                    expected_deletion_marker=expected_deletion_marker,
                )
            except DocumentWriteRejected:
                return {
                    "document_id": "",
                    "chunk_ids": [],
                    "chunks": 0,
                    "embedded": False,
                    "skipped_deleted": True,
                    "sanitized": {
                        "injections_removed": prepared.sanitized.injections_removed,
                        "markup_removed": prepared.sanitized.markup_removed,
                        "truncated": prepared.sanitized.truncated,
                    },
                }
        else:
            doc_id = self._store.upsert_document(
                url=document.url,
                title=document.title,
                text=text,
                content_type=document.content_type,
                doc_hash=_hash_text(text),
                metadata=dict(document.metadata),
                quarantine_library_candidate=bool(adoption_run_id),
            )
            chunk_ids = [
                self._store.add_chunk(
                    document_id=doc_id,
                    chunk_index=index,
                    text=piece,
                    embedding=embedding,
                )
                for index, (piece, embedding) in enumerate(chunk_rows)
            ]
        return {
            "document_id": doc_id,
            "chunk_ids": chunk_ids,
            "chunks": len(prepared.pieces),
            "embedded": bool(prepared.embeddings),
            "sanitized": {
                "injections_removed": prepared.sanitized.injections_removed,
                "markup_removed": prepared.sanitized.markup_removed,
                "truncated": prepared.sanitized.truncated,
            },
        }

    def commit_local_import(
        self,
        prepared: PreparedDocument,
        *,
        content_sha256: str,
        filename: str,
        expected_deletion_marker: str | None = None,
    ) -> dict:
        """Commit an immutable lexical-only local import in one Store transaction."""

        document = prepared.document
        text = prepared.sanitized.text
        if prepared.embeddings is not None:
            raise ValueError("local imports must be prepared without embeddings")
        created, document_id, chunk_ids, row = self._store.insert_local_document_with_chunks_if_absent(
            url=document.url,
            title=document.title,
            text=text,
            chunks=[(piece, None) for piece in prepared.pieces],
            content_type=document.content_type,
            doc_hash=_hash_text(text),
            metadata={
                **dict(document.metadata),
                "content_sha256": content_sha256,
                "filename": filename,
            },
            expected_deletion_marker=expected_deletion_marker,
        )
        metadata = row.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}
        return {
            "status": "created" if created else "existing",
            "document_id": document_id,
            "title": str(row.get("title") or document.title),
            "filename": str(metadata.get("filename") or filename),
            "content_type": str(row.get("content_type") or document.content_type),
            "content_sha256": str(metadata.get("content_sha256") or content_sha256),
            "chunk_ids": chunk_ids,
            "chunks": len(chunk_ids),
            "embedded": False,
            "sanitized": {
                "injections_removed": prepared.sanitized.injections_removed,
                "markup_removed": prepared.sanitized.markup_removed,
                "truncated": prepared.sanitized.truncated,
            },
        }

    def replace_prepared_document_if_present(
        self,
        prepared: PreparedDocument,
        *,
        expected_document_id: str,
    ) -> dict | None:
        """Commit rebuilt chunks only if the original document identity still exists."""

        document = prepared.document
        text = prepared.sanitized.text
        chunk_ids = self._store.replace_document_chunks_if_present(
            expected_document_id=expected_document_id,
            url=document.url,
            text=text,
            chunks=[
                (
                    piece,
                    list(prepared.embeddings[index]) if prepared.embeddings else None,
                )
                for index, piece in enumerate(prepared.pieces)
            ],
            doc_hash=_hash_text(text),
        )
        if chunk_ids is None:
            return None
        return {
            "document_id": expected_document_id,
            "chunk_ids": chunk_ids,
            "chunks": len(chunk_ids),
            "embedded": bool(prepared.embeddings),
            "sanitized": {
                "injections_removed": prepared.sanitized.injections_removed,
                "markup_removed": prepared.sanitized.markup_removed,
                "truncated": prepared.sanitized.truncated,
            },
        }

    def index_search_results(
        self,
        search: SearchToolAdapter,
        query: str,
        *,
        count: int = 5,
        options: SearchOptions | None = None,
        adoption_run_id: str | None = None,
    ) -> list[dict]:
        """Search, fetch each hit, and index it; returns per-document index stats."""

        stats, _ = self.index_search_results_with_outcome(
            search,
            query,
            count=count,
            options=options,
            adoption_run_id=adoption_run_id,
        )
        return stats

    def index_search_results_with_outcome(
        self,
        search: SearchToolAdapter,
        query: str,
        *,
        count: int = 5,
        options: SearchOptions | None = None,
        recovery: bool = False,
        excluded_provenance_roots: Sequence[str] = (),
        relevance_query: str | None = None,
        adoption_run_id: str | None = None,
    ) -> tuple[list[dict], dict]:
        """Index one search call and return its immutable query-local status snapshot."""

        call_with_outcome = getattr(search, "search_with_outcome", None)
        if callable(call_with_outcome):
            kwargs = {"count": count, "options": options}
            parameters = inspect.signature(call_with_outcome).parameters
            if recovery and "recovery" in parameters:
                kwargs.update(
                    recovery=True,
                    excluded_provenance_roots=tuple(excluded_provenance_roots),
                    relevance_query=relevance_query,
                )
            call = call_with_outcome(query, **kwargs)
            if isinstance(call, SearchCallResult):
                hits = list(call.hits)
                search_status = dict(call.status)
            else:
                hits = list(getattr(call, "hits", ()) or ())
                search_status = dict(getattr(call, "status", {}) or {})
        else:
            search_fn = search.search
            kwargs = {"count": count, "options": options}
            parameters = inspect.signature(search_fn).parameters
            if recovery and "recovery" in parameters:
                kwargs.update(
                    recovery=True,
                    excluded_provenance_roots=tuple(excluded_provenance_roots),
                    relevance_query=relevance_query,
                )
            hits = list(search_fn(query, **kwargs))
            status_fn = getattr(search, "last_search_status", None)
            raw_status = status_fn() if callable(status_fn) else {}
            search_status = dict(raw_status) if isinstance(raw_status, dict) else {}

        stats: list[dict] = []
        for hit in hits:
            fallback_tier = str(hit.metadata.get("fallback_tier") or "")
            search_provider = str(hit.metadata.get("search_provider") or "")
            try:
                document = search.fetch(hit.url)
            except Exception:  # noqa: BLE001 - skip un-fetchable pages, keep going
                fallback_tier = fallback_tier or "snippet_only"
                document = Document(
                    url=hit.url,
                    title=hit.title,
                    text=hit.snippet,
                    content_type="web",
                    metadata={**hit.metadata, "fallback_tier": fallback_tier},
                )
                degraded_fetch = True
            else:
                degraded_fetch = False
            if not (document.text or "").strip():
                continue
            merged_metadata = {**hit.metadata, **document.metadata}
            if fallback_tier and "fallback_tier" not in merged_metadata:
                merged_metadata["fallback_tier"] = fallback_tier
            if merged_metadata != document.metadata:
                document = Document(
                    url=document.url,
                    title=document.title,
                    text=document.text,
                    content_type=document.content_type,
                    raw=document.raw,
                    metadata=merged_metadata,
                )
            result = self.index_document(document, adoption_run_id=adoption_run_id)
            fetch_provider = str(document.metadata.get("fetch_provider") or "")
            if search_provider:
                result["search_provider"] = search_provider
            if fetch_provider:
                result["fetch_provider"] = fetch_provider
            if fallback_tier:
                result["fallback_tier"] = fallback_tier
            if degraded_fetch:
                result["degraded_fetch"] = True
            stats.append(result)
        return stats, search_status

    def _embed(self, pieces: Sequence[str]) -> list[list[float]] | None:
        if not self._embedder or not pieces:
            return None
        try:
            return self._embedder.embed(list(pieces))
        except Exception:  # noqa: BLE001 - degrade to lexical-only on embed failure
            return None
