"""Runtime retrieval over Library notes/reports for context and citable evidence."""

from __future__ import annotations

import json
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any

from .adapters import LLMAdapter
from .deadline import current_provider_call_budget, use_provider_call_budget
from .retrieval import EvidenceItem, reciprocal_rank_fusion
from .store import Store
from .vector_backend import BruteForceVectorBackend, VectorBackend

JsonDict = dict[str, Any]
_LIBRARY_TYPES = {"note", "report"}
_EVIDENCE_TYPES = {"note", "report", "library_document"}
_OPTIONAL_QUERY_EMBED_TIMEOUT_S = 1.0


@dataclass(frozen=True)
class LibraryRetrievalConfig:
    """Tuning for local Library context retrieval."""

    rrf_k: int = 60
    lexical_top: int = 20
    vector_top: int = 20
    final_top: int = 5
    max_context_chars: int = 3000
    max_item_chars: int = 700


@dataclass
class LibraryHit:
    """A local Library snippet with stable artifact identity."""

    chunk_id: int
    document_id: str
    artifact_type: str
    artifact_id: str
    title: str
    text: str
    score: float = 0.0
    channels: list[str] = field(default_factory=list)


class LibraryRetriever:
    """Retrieve local notes/reports for separate context and evidence paths."""

    def __init__(
        self,
        store: Store,
        *,
        embedder: LLMAdapter | None = None,
        config: LibraryRetrievalConfig | None = None,
        vector_backend: VectorBackend | None = None,
    ) -> None:
        self._store = store
        self._embedder = embedder
        self._config = config or LibraryRetrievalConfig()
        self._vector_backend = vector_backend or BruteForceVectorBackend(store)

    def retrieve(self, query: str, *, scope: str = "off", top_k: int | None = None) -> list[LibraryHit]:
        """Return local Library hits for ``scope``; failures degrade to empty."""

        allowed = _allowed_types(scope)
        return self._retrieve_types(query, allowed=allowed, top_k=top_k)

    def _retrieve_types(
        self,
        query: str,
        *,
        allowed: set[str],
        top_k: int | None = None,
    ) -> list[LibraryHit]:
        """Retrieve one explicit artifact allowlist without changing scope semantics."""

        if not allowed:
            return []
        cfg = self._config
        final_top = top_k or cfg.final_top
        lexical_ids = self._lexical_channel(query, limit=max(cfg.lexical_top, final_top * 4))
        active_budget = current_provider_call_budget()
        vector_ids = (
            self._vector_channel(query, limit=max(cfg.vector_top, final_top * 4))
            if self._store.has_document_content_types(allowed, require_embedding=True)
            and not (active_budget is not None and lexical_ids)
            else []
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
        hits = self._hydrate(fused, channel_map, allowed=allowed)
        hits = _dedup_by_artifact(hits)
        hits.sort(key=lambda hit: hit.score, reverse=True)
        return hits[:final_top]

    def retrieve_evidence(self, query: str, *, top_k: int | None = None) -> list[EvidenceItem]:
        """Return citable note/report evidence independently of background ``scope``."""

        hits = self._retrieve_types(query, allowed=set(_EVIDENCE_TYPES), top_k=top_k)
        retrieved_at = time.time()
        return [
            EvidenceItem(
                chunk_id=hit.chunk_id,
                document_id=hit.document_id,
                url="",
                title=hit.title,
                text=hit.text,
                score=hit.score,
                channels=list(hit.channels),
                origin="local",
                artifact_type={
                    "note": "library_note",
                    "report": "library_report",
                    "library_document": "library_document",
                }[hit.artifact_type],
                artifact_id=hit.artifact_id,
                grounding_eligible=True,
                provenance_refs=[
                    (
                        f"local:library_document:{hit.artifact_id}"
                        if hit.artifact_type == "library_document"
                        else f"local:{hit.artifact_type}:{hit.artifact_id}"
                    )
                ],
                source_status="active",
                retrieved_at=retrieved_at,
            )
            for hit in hits
        ]

    def _lexical_channel(self, query: str, *, limit: int) -> list[int]:
        try:
            return [cid for cid, _ in self._store.search_chunks_lexical(query, limit=limit)]
        except Exception:  # noqa: BLE001 - Library context is best-effort
            return []

    def _vector_channel(self, query: str, *, limit: int) -> list[int]:
        if not self._embedder:
            return []
        active_budget = current_provider_call_budget()
        budget_scope = (
            use_provider_call_budget(
                active_budget.with_timeout_cap(_OPTIONAL_QUERY_EMBED_TIMEOUT_S)
            )
            if active_budget is not None
            else nullcontext()
        )
        try:
            with budget_scope:
                query_vec = self._embedder.embed([query])[0]
                matches = self._vector_backend.search(query_vec, limit=limit)
        except Exception:  # noqa: BLE001 - degrade to lexical-only
            return []
        return [cid for cid, _ in matches]

    def _hydrate(
        self,
        fused: dict[int, float],
        channel_map: dict[int, list[str]],
        *,
        allowed: set[str],
    ) -> list[LibraryHit]:
        hits: list[LibraryHit] = []
        for chunk_id, score in fused.items():
            chunk = self._store.get_chunk(chunk_id)
            if not chunk:
                continue
            doc = self._store.get_document(chunk.document_id)
            artifact_type, artifact_id = _artifact_identity(doc)
            if artifact_type not in allowed:
                continue
            hits.append(
                LibraryHit(
                    chunk_id=chunk.id,
                    document_id=chunk.document_id,
                    artifact_type=artifact_type,
                    artifact_id=artifact_id,
                    title=str((doc or {}).get("title") or artifact_type.title()),
                    text=chunk.text,
                    score=score,
                    channels=channel_map.get(chunk_id, []),
                )
            )
        return hits


def render_library_context(hits: list[LibraryHit], *, max_chars: int = 3000, max_item_chars: int = 700) -> str:
    """Render Library hits as bounded, explicitly non-citable prompt context."""

    if not hits or max_chars <= 0:
        return ""
    header = "Library context (local notes/reports; for background only; do not cite as web sources):"
    lines = [header]
    total = len(header)
    for hit in hits:
        text = " ".join(hit.text.split())
        if len(text) > max_item_chars:
            text = text[:max_item_chars].rstrip() + "..."
        title = " ".join(hit.title.split()) or hit.artifact_type.title()
        line = f"- [{hit.artifact_type}] {title}: {text}"
        if total + 1 + len(line) > max_chars:
            remaining = max_chars - total - 1
            if remaining > 20:
                lines.append(line[:remaining].rstrip())
            break
        lines.append(line)
        total += 1 + len(line)
    return "\n".join(lines)[:max_chars]


def _allowed_types(scope: str) -> set[str]:
    normalized = (scope or "off").lower()
    if normalized == "notes":
        return {"note"}
    if normalized == "reports":
        return {"report"}
    if normalized == "all":
        return {"note", "report"}
    return set()


def _artifact_identity(doc: JsonDict | None) -> tuple[str, str]:
    if not doc:
        return ("", "")
    metadata = _metadata(doc.get("metadata"))
    content_type = str(doc.get("content_type") or "")
    url = str(doc.get("url") or "")
    artifact_type = str(metadata.get("artifact_type") or content_type)
    artifact_id = str(metadata.get("artifact_id") or metadata.get(f"{artifact_type}_id") or "")
    if not artifact_id and url.startswith("local://library/"):
        artifact_id = url.rsplit("/", 1)[-1]
    return artifact_type, artifact_id


def _metadata(value: object) -> JsonDict:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str) and value:
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _dedup_by_artifact(hits: list[LibraryHit]) -> list[LibraryHit]:
    best: dict[tuple[str, str], LibraryHit] = {}
    for hit in hits:
        key = (hit.artifact_type, hit.artifact_id or f"{hit.document_id}:{hit.chunk_id}")
        if key not in best or hit.score > best[key].score:
            best[key] = hit
    return list(best.values())
