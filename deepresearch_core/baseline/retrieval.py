"""Retrieval Layer: dual-channel search (FTS5 + vector) fused with RRF.

Lexical (FTS5/LIKE) is always available. The vector channel is used only when an
embedding backend and stored embeddings exist; otherwise retrieval is lexical-only.
Results are fused with Reciprocal Rank Fusion, deduplicated by document, optionally
reranked, and returned as an evidence pack -- the only input the Answer Composer may cite.
"""

from __future__ import annotations

import json
import math
import os
import urllib.request
from dataclasses import dataclass, field
from typing import Callable, Sequence

from .adapters import LLMAdapter
from .deadline import current_provider_call_budget
from .errors import BaselineError, ErrorCode
from .store import ChunkRow, Store
from .vector_backend import BruteForceVectorBackend, VectorBackend
from ..observability import span

Reranker = Callable[[str, list["EvidenceItem"]], list["EvidenceItem"]]
_LOCAL_LIBRARY_CONTENT_TYPES = {"note", "report", "memory", "library_document"}


@dataclass(frozen=True)
class RetrievalConfig:
    """Retrieval tuning parameters."""

    rrf_k: int = 60
    lexical_top: int = 20
    vector_top: int = 20
    final_top: int = 8


@dataclass
class EvidenceItem:
    """A single retrieved, citable piece of evidence."""

    chunk_id: int
    document_id: str
    url: str
    title: str
    text: str
    score: float = 0.0
    channels: list[str] = field(default_factory=list)
    origin: str = "external"
    artifact_type: str = "web_document"
    artifact_id: str = ""
    grounding_eligible: bool = True
    provenance_refs: list[str] = field(default_factory=list)
    source_status: str = "active"
    published_at: float | None = None
    updated_at: float | None = None
    effective_at: float | None = None
    retrieved_at: float | None = None
    search_provider: str = ""
    fetch_provider: str = ""

    def to_dict(self) -> dict:
        """Return a JSON-serializable view."""

        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "url": self.url,
            "title": self.title,
            "text": self.text,
            "score": round(self.score, 6),
            "channels": list(self.channels),
            "origin": self.origin,
            "artifact_type": self.artifact_type,
            "artifact_id": self.artifact_id,
            "grounding_eligible": self.grounding_eligible,
            "provenance_refs": list(self.provenance_refs),
            "source_status": self.source_status,
            "published_at": self.published_at,
            "updated_at": self.updated_at,
            "effective_at": self.effective_at,
            "retrieved_at": self.retrieved_at,
            "search_provider": self.search_provider or None,
            "fetch_provider": self.fetch_provider or None,
        }


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Return cosine similarity of two equal-length vectors (0 when degenerate)."""

    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def embedding_reranker(embedder: LLMAdapter) -> Reranker:
    """Return a lightweight reranker that re-scores items by query/text cosine.

    Uses the existing embedding backend only (no cross-encoder dependency). On any
    embedding failure the input order is preserved, so it is always safe to attach.
    """

    def _rerank(query: str, items: list[EvidenceItem]) -> list[EvidenceItem]:
        if not items:
            return items
        try:
            vectors = embedder.embed([query] + [item.text for item in items])
        except BaselineError as error:
            if error.code in {ErrorCode.RUN_TIMEOUT, ErrorCode.RUN_CANCELLED}:
                raise
            return items
        except Exception:  # noqa: BLE001 - degrade to the incoming order
            return items
        if not vectors or len(vectors) != len(items) + 1:
            return items
        query_vec, item_vecs = vectors[0], vectors[1:]
        scored = sorted(
            zip(items, item_vecs),
            key=lambda pair: cosine_similarity(query_vec, pair[1]),
            reverse=True,
        )
        return [item for item, _ in scored]

    return _rerank


def http_reranker(
    url: str,
    *,
    model: str | None = None,
    api_key: str | None = None,
    timeout_s: float = 8.0,
    top_n: int | None = None,
    fallback: Reranker | None = None,
) -> Reranker:
    """Return a cross-encoder reranker backed by an HTTP rerank endpoint (DR-ITER-025).

    Speaks the de-facto rerank contract shared by Cohere / Jina / TEI:
    ``POST {model, query, documents:[str], top_n?}`` -> ``{results:[{index, relevance_score}]}``.
    On any transport, decoding, or shape error it degrades to ``fallback`` (typically the
    embedding reranker from DR-ITER-018) and finally to the incoming order, so attaching it
    never reduces availability.
    """

    def _rerank(query: str, items: list[EvidenceItem]) -> list[EvidenceItem]:
        if not items:
            return items
        documents = [item.text for item in items]
        payload: dict[str, object] = {"query": query, "documents": documents}
        if model:
            payload["model"] = model
        if top_n:
            payload["top_n"] = top_n
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        request = urllib.request.Request(
            url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        budget = current_provider_call_budget()
        bounded_timeout = timeout_s
        if budget is not None:
            budget.check("remote_rerank_connect")
            bounded_timeout = budget.call_timeout_s(timeout_s)
        try:
            with urllib.request.urlopen(request, timeout=bounded_timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
            if budget is not None:
                budget.check("remote_rerank_read")
            order = _parse_rerank_results(data, count=len(items))
        except BaselineError as error:
            if error.code in {ErrorCode.RUN_TIMEOUT, ErrorCode.RUN_CANCELLED}:
                raise
            return fallback(query, items) if fallback else items
        except Exception:  # noqa: BLE001 - any failure degrades, never raises into retrieval
            return fallback(query, items) if fallback else items
        if order is None:
            return fallback(query, items) if fallback else items
        return [items[i] for i in order]

    return _rerank


def _parse_rerank_results(data: object, *, count: int) -> list[int] | None:
    """Extract a full permutation of indices ``[0, count)`` from a rerank response.

    Returns ``None`` when the payload is malformed or does not cover every input item, so the
    caller can fall back instead of silently dropping or reordering evidence incorrectly.
    """

    if not isinstance(data, dict):
        return None
    results = data.get("results")
    if not isinstance(results, list) or len(results) != count:
        return None
    order: list[int] = []
    seen: set[int] = set()
    for entry in results:
        if not isinstance(entry, dict):
            return None
        index = entry.get("index")
        if not isinstance(index, int) or index < 0 or index >= count or index in seen:
            return None
        order.append(index)
        seen.add(index)
    return order if len(order) == count else None


def build_reranker(embedder: LLMAdapter | None) -> Reranker | None:
    """Pick a reranker from environment config (DR-ITER-025), defaulting to embedding rerank.

    ``DEEPRESEARCH_RERANK_MODE``:
      * ``embedding`` (default) -- DR-ITER-018 cosine reranker (requires an embedder);
      * ``http`` -- cross-encoder via ``DEEPRESEARCH_RERANK_URL``, degrading to embedding rerank;
      * ``off`` -- no reranking.
    """

    mode = (os.environ.get("DEEPRESEARCH_RERANK_MODE") or "embedding").strip().lower()
    if mode == "off":
        return None
    embedding_fallback = embedding_reranker(embedder) if embedder else None
    if mode == "http":
        url = (os.environ.get("DEEPRESEARCH_RERANK_URL") or "").strip()
        if not url:
            return embedding_fallback
        return http_reranker(
            url,
            model=(os.environ.get("DEEPRESEARCH_RERANK_MODEL") or "").strip() or None,
            api_key=(os.environ.get("DEEPRESEARCH_RERANK_API_KEY") or "").strip() or None,
            fallback=embedding_fallback,
        )
    return embedding_fallback


def reciprocal_rank_fusion(rankings: Sequence[Sequence[int]], *, k: int = 60) -> dict[int, float]:
    """Fuse multiple ranked id-lists into a combined score map (higher = better)."""

    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, item_id in enumerate(ranking):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank + 1)
    return scores


class Retriever:
    """Runs dual-channel retrieval + RRF fusion over the store."""

    def __init__(
        self,
        store: Store,
        *,
        embedder: LLMAdapter | None = None,
        reranker: Reranker | None = None,
        config: RetrievalConfig | None = None,
        vector_backend: VectorBackend | None = None,
    ) -> None:
        self._store = store
        self._embedder = embedder
        self._reranker = reranker
        self._config = config or RetrievalConfig()
        # ANN backend is a derived index over the store; default to the brute-force scan so
        # behavior is identical when no backend is injected (DR-ITER-020).
        self._vector_backend = vector_backend or BruteForceVectorBackend(store)

    def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
        excluded_provenance_roots: Sequence[str] = (),
        document_ids: Sequence[str] | None = None,
    ) -> list[EvidenceItem]:
        """Return a fused evidence pack, optionally scoped to selected documents."""

        cfg = self._config
        final_top = top_k or cfg.final_top
        selected_documents = (
            {str(value) for value in document_ids if str(value)}
            if document_ids is not None
            else None
        )
        if selected_documents is not None and not selected_documents:
            return []
        channel_multiplier = max(1, min(len(selected_documents or ()), 50))

        with span("retrieval.retrieve", final_top=final_top):
            lexical_ids = [
                cid
                for cid, _ in self._store.search_chunks_lexical(
                    query,
                    limit=cfg.lexical_top * channel_multiplier,
                )
            ]
            lexical_ids = self._filter_chunk_ids(lexical_ids, selected_documents)
            vector_ids = self._vector_channel(
                query,
                limit=cfg.vector_top * channel_multiplier,
                document_ids=selected_documents,
            )

            rankings = [lexical_ids]
            if vector_ids:
                rankings.append(vector_ids)
            fused = reciprocal_rank_fusion(rankings, k=cfg.rrf_k)

            channel_map: dict[int, list[str]] = {}
            for cid in lexical_ids:
                channel_map.setdefault(cid, []).append("lexical")
            for cid in vector_ids:
                channel_map.setdefault(cid, []).append("vector")

            items = self._hydrate(fused, channel_map)
            if selected_documents is not None:
                items = [item for item in items if item.document_id in selected_documents]
            excluded = {str(value) for value in excluded_provenance_roots if str(value)}
            if excluded:
                items = [
                    item
                    for item in items
                    if (str(item.provenance_refs[0]) if item.provenance_refs else item.url or item.document_id)
                    not in excluded
                ]
            items = _dedup_by_document(items)
            items.sort(key=lambda i: i.score, reverse=True)
            items = items[: max(final_top * 2, final_top)]
            if self._reranker and items:
                items = self._reranker(query, items)
            return items[:final_top]

    def _vector_channel(
        self,
        query: str,
        *,
        limit: int,
        document_ids: set[str] | None = None,
    ) -> list[int]:
        if not self._embedder:
            return []
        try:
            query_vec = self._embedder.embed([query])[0]
        except BaselineError as error:
            if error.code in {ErrorCode.RUN_TIMEOUT, ErrorCode.RUN_CANCELLED}:
                raise
            return []
        except Exception:  # noqa: BLE001 - degrade to lexical-only
            return []
        try:
            matches = self._vector_backend.search(query_vec, limit=limit)
        except Exception:  # noqa: BLE001 - any backend error degrades to lexical-only
            return []
        return self._filter_chunk_ids([cid for cid, _ in matches], document_ids)

    def _filter_chunk_ids(
        self,
        chunk_ids: Sequence[int],
        document_ids: set[str] | None,
    ) -> list[int]:
        if document_ids is None:
            return list(chunk_ids)
        selected: list[int] = []
        for chunk_id in chunk_ids:
            chunk = self._store.get_chunk(chunk_id)
            if chunk is not None and chunk.document_id in document_ids:
                selected.append(chunk_id)
        return selected

    def _hydrate(self, fused: dict[int, float], channel_map: dict[int, list[str]]) -> list[EvidenceItem]:
        items: list[EvidenceItem] = []
        for chunk_id, score in fused.items():
            chunk = self._store.get_chunk(chunk_id)
            if not chunk:
                continue
            doc = self._store.get_document(chunk.document_id)
            if _is_local_library_document(doc):
                continue
            metadata = _metadata((doc or {}).get("metadata"))
            url = str((doc or {}).get("url") or "")
            items.append(
                EvidenceItem(
                    chunk_id=chunk.id,
                    document_id=chunk.document_id,
                    url=url,
                    title=str((doc or {}).get("title") or ""),
                    text=chunk.text,
                    score=score,
                    channels=channel_map.get(chunk_id, []),
                    provenance_refs=[str(metadata.get("provenance_root") or url or chunk.document_id)],
                    published_at=_timestamp(metadata.get("published_at")),
                    updated_at=_timestamp(metadata.get("updated_at")),
                    effective_at=_timestamp(metadata.get("effective_at")),
                    retrieved_at=_timestamp(metadata.get("retrieved_at") or (doc or {}).get("created_at")),
                )
            )
        return items


def _is_local_library_document(doc: dict | None) -> bool:
    """Return True for Library artifacts owned by the dedicated local evidence lane."""

    if not doc:
        return False
    url = str(doc.get("url") or "")
    content_type = str(doc.get("content_type") or "")
    metadata = _metadata(doc.get("metadata"))
    return (
        url.startswith(("local://library/", "library://local/"))
        or content_type in _LOCAL_LIBRARY_CONTENT_TYPES
        or str(metadata.get("artifact_type") or "") in _LOCAL_LIBRARY_CONTENT_TYPES
        or str(metadata.get("source") or "") == "local_import"
    )


def _metadata(value: object) -> dict:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str) and value:
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _timestamp(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _dedup_by_document(items: list[EvidenceItem]) -> list[EvidenceItem]:
    """Keep the best-scoring chunk per document URL."""

    best: dict[str, EvidenceItem] = {}
    for item in items:
        key = item.url or f"doc:{item.document_id}:{item.chunk_id}"
        if key not in best or item.score > best[key].score:
            best[key] = item
    return list(best.values())
