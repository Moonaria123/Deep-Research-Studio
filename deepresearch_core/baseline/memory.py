"""Three-layer memory engine: session / workspace / user (baseline §记忆).

Memories carry a salience score that decays with age; recall ranks by effective
salience (salience * time-decay) so fresh, important items surface first.
"""

from __future__ import annotations

import math
import re
import time
from contextlib import nullcontext
from dataclasses import dataclass
from enum import Enum
from typing import Any

from .deadline import current_provider_call_budget, use_provider_call_budget
from .retrieval import EvidenceItem, cosine_similarity
from .store import Store


class MemoryLayer(str, Enum):
    """Memory namespaces, narrowest to broadest."""

    SESSION = "session"
    WORKSPACE = "workspace"
    USER = "user"


_DAY_SECONDS = 86400.0
_OPTIONAL_QUERY_EMBED_TIMEOUT_S = 1.0


@dataclass(frozen=True)
class RecallConfig:
    """Recall ranking parameters."""

    half_life_days: float = 14.0
    per_layer_limit: int = 10
    # Blend weight for semantic similarity vs. effective salience in query recall.
    similarity_weight: float = 0.6
    min_similarity: float = 0.35


class MemoryEngine:
    """Stores and recalls memories across the three layers with salience decay.

    When an ``embedder`` (any object exposing ``embed(list[str]) -> list[list[float]]``)
    is supplied, memories are embedded on write and ``recall(query=...)`` fuses cosine
    similarity with effective salience. Without an embedder (or on any embedding error),
    the engine degrades gracefully to pure salience ranking -- never raising.
    """

    def __init__(
        self, store: Store, *, config: RecallConfig | None = None, embedder: Any | None = None
    ) -> None:
        self._store = store
        self._config = config or RecallConfig()
        self._embedder = embedder

    def remember(
        self,
        layer: MemoryLayer,
        scope_key: str,
        content: str,
        *,
        salience: float = 1.0,
        metadata: dict | None = None,
        embed: bool = True,
    ) -> str:
        """Persist a memory item in a layer; returns its id.

        Embeds the content when requested and an embedder is configured so it
        becomes semantically recallable; embedding failures are swallowed (the
        memory is still stored, just without a vector).
        """

        return self._store.add_memory(
            namespace=layer.value,
            scope_key=scope_key,
            content=content,
            salience=max(0.0, salience),
            metadata=metadata,
            embedding=self._embed_one(content) if embed else None,
        )

    def _embed_one(self, text: str, *, timeout_cap_s: float | None = None) -> list[float] | None:
        """Embed a single string, returning ``None`` when unavailable/failed."""

        if not self._embedder or not text:
            return None
        active_budget = current_provider_call_budget()
        budget_scope = (
            use_provider_call_budget(active_budget.with_timeout_cap(timeout_cap_s))
            if active_budget is not None and timeout_cap_s is not None
            else nullcontext()
        )
        try:
            with budget_scope:
                vectors = self._embedder.embed([text])
        except Exception:  # noqa: BLE001 - embedding is best-effort (graceful degradation)
            return None
        if vectors and isinstance(vectors[0], list):
            return _valid_vector(vectors[0])
        return None

    def recall(
        self,
        *,
        session_key: str = "",
        workspace_key: str = "default",
        user_key: str = "default",
        limit: int = 12,
        query: str = "",
    ) -> list[dict]:
        """Recall across all three layers.

        With a ``query`` and an embedder, results are ranked by a blend of cosine
        similarity and effective (decayed) salience; otherwise by salience alone.
        Cross-conversation recall is provided naturally by the workspace/user
        layers, which span sessions.
        """

        now = time.time()
        self._sweep_scopes(
            session_key=session_key,
            workspace_key=workspace_key,
            user_key=user_key,
            now=now,
        )
        query_vec = (
            self._embed_one(query, timeout_cap_s=_OPTIONAL_QUERY_EMBED_TIMEOUT_S)
            if query
            else None
        )
        layer_keys = {
            MemoryLayer.SESSION: session_key,
            MemoryLayer.WORKSPACE: workspace_key,
            MemoryLayer.USER: user_key,
        }
        by_id: dict[str, dict] = {}
        for layer, key in layer_keys.items():
            if not key:
                continue
            self._collect_layer(layer, key, now=now, query_vec=query_vec, sink=by_id)
        candidates = list(by_id.values())
        if query_vec is not None:
            candidates = [
                item
                for item in candidates
                if item.get("_similarity_comparable")
                and item.get("similarity", 0.0) >= self._config.min_similarity
            ]
            weight = self._config.similarity_weight
            candidates.sort(
                key=lambda i: weight * i.get("similarity", 0.0)
                + (1.0 - weight) * i["effective_salience"],
                reverse=True,
            )
        else:
            candidates.sort(key=lambda i: i["effective_salience"], reverse=True)
        top = candidates[:limit]
        for item in top:
            item.pop("_similarity_comparable", None)
            self._store.touch_memory(item["id"])
        return top

    def recall_evidence(
        self,
        *,
        session_key: str = "",
        workspace_key: str = "default",
        user_key: str = "default",
        limit: int = 12,
        query: str,
    ) -> list[EvidenceItem]:
        """Return relevant active and archived memories as citable evidence."""

        cleaned_query = " ".join(str(query or "").split())
        if not cleaned_query:
            return []
        now = time.time()
        self._sweep_scopes(
            session_key=session_key,
            workspace_key=workspace_key,
            user_key=user_key,
            now=now,
        )
        query_terms = _search_terms(cleaned_query)
        scoped_rows: list[tuple[MemoryLayer, list[dict], dict[str, object]]] = []
        has_stored_vector = False
        for layer, scope_key in (
            (MemoryLayer.SESSION, session_key),
            (MemoryLayer.WORKSPACE, workspace_key),
            (MemoryLayer.USER, user_key),
        ):
            if not scope_key:
                continue
            rows = self._store.list_memories(
                namespace=layer.value,
                scope_key=scope_key,
                limit=max(limit * 5, self._config.per_layer_limit),
                include_archived=True,
            )
            if not rows:
                continue
            embeddings = {
                str(row.get("id")): row.get("embedding")
                for row in self._store.iter_memories_with_embeddings(
                    namespace=layer.value,
                    scope_key=scope_key,
                    limit=max(limit * 5, self._config.per_layer_limit),
                    include_archived=True,
                )
            }
            has_stored_vector = has_stored_vector or bool(embeddings)
            scoped_rows.append((layer, rows, embeddings))
        has_lexical_match = any(
            _lexical_similarity(query_terms, _search_terms(str(row.get("content") or ""))) > 0.0
            for _, rows, _ in scoped_rows
            for row in rows
        )
        active_budget = current_provider_call_budget()
        query_vec = (
            self._embed_one(
                cleaned_query,
                timeout_cap_s=_OPTIONAL_QUERY_EMBED_TIMEOUT_S,
            )
            if has_stored_vector and not (active_budget is not None and has_lexical_match)
            else None
        )
        candidates: list[tuple[float, dict, str]] = []
        for layer, rows, embeddings in scoped_rows:
            for row in rows:
                content = str(row.get("content") or "")
                lexical = _lexical_similarity(query_terms, _search_terms(content))
                vector = embeddings.get(str(row.get("id")))
                similarity = 0.0
                comparable = False
                candidate_vec = _valid_vector(vector) if isinstance(vector, list) else None
                if query_vec is not None and candidate_vec is not None and len(candidate_vec) == len(query_vec):
                    similarity = cosine_similarity(query_vec, candidate_vec)
                    comparable = similarity >= self._config.min_similarity
                if not comparable and lexical <= 0.0:
                    continue
                score = max(lexical, similarity if comparable else 0.0)
                candidates.append((score, row, layer.value))
        candidates.sort(key=lambda item: (item[0], float(item[1].get("salience") or 0.0)), reverse=True)
        evidence: list[EvidenceItem] = []
        seen: set[str] = set()
        for score, row, layer in candidates:
            memory_id = str(row.get("id") or "")
            if not memory_id or memory_id in seen:
                continue
            seen.add(memory_id)
            metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
            evidence.append(
                EvidenceItem(
                    chunk_id=0,
                    document_id="",
                    url="",
                    title=str(metadata.get("title") or f"Memory ({layer})"),
                    text=str(row.get("content") or ""),
                    score=score,
                    channels=["memory_semantic" if query_vec is not None else "memory_lexical"],
                    origin="memory",
                    artifact_type="memory",
                    artifact_id=memory_id,
                    grounding_eligible=True,
                    provenance_refs=[f"memory:memory:{memory_id}"],
                    source_status="archived" if row.get("archived_at") is not None else "active",
                    retrieved_at=now,
                )
            )
            if len(evidence) >= limit:
                break
        return evidence

    def _sweep_scopes(
        self,
        *,
        session_key: str,
        workspace_key: str,
        user_key: str,
        now: float,
    ) -> None:
        """Best-effort lazy expiry sweep; active reads still exclude expired rows on failure."""

        for namespace, scope_key in (
            (MemoryLayer.SESSION.value, session_key),
            (MemoryLayer.WORKSPACE.value, workspace_key),
            (MemoryLayer.USER.value, user_key),
        ):
            if not scope_key:
                continue
            try:
                self._store.sweep_expired_memories(
                    now=now,
                    limit=self._config.per_layer_limit,
                    namespace=namespace,
                    scope_key=scope_key,
                )
            except Exception:  # noqa: BLE001 - immediate read predicates remain fail-safe
                continue

    def _collect_layer(
        self, layer: MemoryLayer, key: str, *, now: float, query_vec: list[float] | None, sink: dict[str, dict]
    ) -> None:
        """Gather salience + (optional) semantic candidates for one layer into ``sink``."""

        for item in self._store.list_memories(
            namespace=layer.value, scope_key=key, limit=self._config.per_layer_limit
        ):
            self._register(dict(item), layer=layer, now=now, sink=sink)
        if query_vec is None:
            return
        # Pull embedded candidates too so semantically-relevant memories surface even
        # when they fall outside the salience-ordered window.
        for emb in self._store.iter_memories_with_embeddings(
            namespace=layer.value, scope_key=key, limit=self._config.per_layer_limit * 5
        ):
            vector = emb.pop("embedding", None)
            target = self._register(emb, layer=layer, now=now, sink=sink)
            candidate_vec = _valid_vector(vector) if isinstance(vector, list) else None
            if candidate_vec is not None and len(candidate_vec) == len(query_vec):
                target["similarity"] = cosine_similarity(query_vec, candidate_vec)
                target["_similarity_comparable"] = True

    def _register(self, item: dict, *, layer: MemoryLayer, now: float, sink: dict[str, dict]) -> dict:
        """Insert/return an item keyed by id, ensuring derived fields are set."""

        mid = str(item.get("id"))
        existing = sink.get(mid)
        if existing is not None:
            return existing
        item.pop("embedding", None)
        item["effective_salience"] = self._effective(item, now=now)
        item["layer"] = layer.value
        item.setdefault("similarity", 0.0)
        sink[mid] = item
        return item

    def count(self, scope_key: str, *, layer: MemoryLayer = MemoryLayer.WORKSPACE) -> int:
        """Return the number of stored memories in a layer/scope (DR-ITER-057).

        Used by the context-budget recommender to gauge how much long-term memory
        a workspace actually has. Best-effort: any store error yields ``0`` rather
        than failing the recommendation.
        """

        if not scope_key:
            return 0
        try:
            return self._store.count_memories(namespace=layer.value, scope_key=scope_key)
        except Exception:  # noqa: BLE001 - recommendation signal is best-effort
            return 0

    def forget(self, layer: MemoryLayer, scope_key: str) -> int:
        """Delete all memories in a layer/scope; returns count removed."""

        return self._store.delete_memories(namespace=layer.value, scope_key=scope_key)

    def _effective(self, item: dict, *, now: float) -> float:
        salience = float(item.get("salience", 1.0))
        created = float(item.get("created_at", now))
        age_days = max(0.0, (now - created) / _DAY_SECONDS)
        decay = math.pow(0.5, age_days / max(0.1, self._config.half_life_days))
        return salience * decay


def _valid_vector(vector: list[Any]) -> list[float] | None:
    """Coerce a finite, non-zero embedding or return ``None`` for salience fallback."""

    if not vector:
        return None
    try:
        values = [float(value) for value in vector]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in values):
        return None
    if not any(value != 0.0 for value in values):
        return None
    return values


_LATIN_SEARCH_TOKEN = re.compile(r"[a-z0-9]+")
_CJK_SEARCH_RUN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]+")


def _search_terms(text: str) -> set[str]:
    terms = {token for token in _LATIN_SEARCH_TOKEN.findall(text.lower()) if len(token) > 1}
    for run in _CJK_SEARCH_RUN.findall(text):
        terms.update(run[index : index + 2] for index in range(len(run) - 1))
    return terms


def _lexical_similarity(query: set[str], content: set[str]) -> float:
    if not query or not content:
        return 0.0
    return len(query & content) / len(query)
