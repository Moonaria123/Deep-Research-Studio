"""Pluggable vector-search backends for the retrieval vector channel (DR-ITER-020).

The SQLite ``chunks.embedding`` column remains the single source of truth. Every backend
here is a *derived* index built from the store, so switching backends never migrates or
double-writes data — at worst the index is rebuilt from SQLite.

Backends (resolved via ``DEEPRESEARCH_VECTOR_BACKEND``):

* ``bruteforce`` — pure-Python cosine scan. Always available; the safety fallback and the
  behavior shipped before DR-ITER-020.
* ``numpy`` — vectorized cosine over a cached matrix. Large-N speedup, exact results.
* ``chroma`` — approximate nearest neighbor via an in-memory Chroma collection (HNSW).
* ``auto`` (default) — pick the best available: chroma > numpy > bruteforce.

Resolution is **graceful**: an explicitly requested backend whose dependency is missing
falls back to the next best one (logged), rather than failing startup. This is deliberately
different from the fail-fast policy for auth/encryption — a missing ANN library must never
take retrieval offline.
"""

from __future__ import annotations

import logging
import json
import math
import os
from pathlib import Path
from typing import Protocol, Sequence, runtime_checkable

from .store import Store

logger = logging.getLogger(__name__)

VALID_BACKENDS = ("auto", "bruteforce", "numpy", "chroma")


@runtime_checkable
class VectorBackend(Protocol):
    """Returns the top chunk ids for a query embedding, best first."""

    name: str

    def search(self, query_embedding: Sequence[float], *, limit: int) -> list[tuple[int, float]]:
        """Return ``(chunk_id, similarity)`` pairs, highest similarity first."""
        ...

    def health(self) -> dict:
        """Return a small, secret-free status dict for ``/health``."""
        ...

    def invalidate(self) -> None:
        """Invalidate any derived state before the next query."""
        ...


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


class BruteForceVectorBackend:
    """Pure-Python cosine scan over the store. No cache, always correct."""

    name = "bruteforce"

    def __init__(self, store: Store) -> None:
        self._store = store

    def search(self, query_embedding: Sequence[float], *, limit: int) -> list[tuple[int, float]]:
        if not query_embedding:
            return []
        scored: list[tuple[int, float]] = []
        for chunk in self._store.iter_chunks_with_embeddings():
            if chunk.embedding:
                scored.append((chunk.id, _cosine(query_embedding, chunk.embedding)))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [(cid, score) for cid, score in scored[:limit] if score > 0.0]

    def health(self) -> dict:
        return {"vector_backend": self.name}

    def invalidate(self) -> None:
        """The brute-force backend has no derived state."""


class _StoreDerivedBackend:
    """Base for cached backends rebuilt from the store on a staleness signature change."""

    name = "derived"

    def __init__(self, store: Store) -> None:
        self._store = store
        self._signature: tuple[int, int] | None = None
        self._force_rebuild = False

    def _ensure_fresh(self) -> None:
        signature = self._store.embedding_index_signature()
        if signature == self._signature:
            return
        ids: list[int] = []
        vectors: list[list[float]] = []
        for chunk in self._store.iter_chunks_with_embeddings():
            if chunk.embedding:
                ids.append(chunk.id)
                vectors.append(list(chunk.embedding))
        self._rebuild(ids, vectors)
        self._signature = signature

    def _rebuild(self, ids: list[int], vectors: list[list[float]]) -> None:  # pragma: no cover
        raise NotImplementedError

    def health(self) -> dict:
        return {"vector_backend": self.name}

    def invalidate(self) -> None:
        """Force a Store-derived rebuild before the next vector query."""

        self._signature = None
        self._force_rebuild = True


class NumpyVectorBackend(_StoreDerivedBackend):
    """Vectorized exact cosine over a cached, L2-normalized matrix."""

    name = "numpy"

    def __init__(self, store: Store) -> None:
        super().__init__(store)
        import numpy as np  # local import: optional dependency

        self._np = np
        self._ids: list[int] = []
        self._matrix = None  # np.ndarray (n, d), rows L2-normalized

    def _rebuild(self, ids: list[int], vectors: list[list[float]]) -> None:
        np = self._np
        self._ids = ids
        if not vectors:
            self._matrix = None
            return
        matrix = np.asarray(vectors, dtype="float32")
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        self._matrix = matrix / norms

    def search(self, query_embedding: Sequence[float], *, limit: int) -> list[tuple[int, float]]:
        if not query_embedding:
            return []
        self._ensure_fresh()
        if self._matrix is None or not self._ids:
            return []
        np = self._np
        query = np.asarray(query_embedding, dtype="float32")
        qnorm = float(np.linalg.norm(query))
        if qnorm == 0.0:
            return []
        query = query / qnorm
        if query.shape[0] != self._matrix.shape[1]:
            return []
        sims = self._matrix @ query
        top = np.argsort(-sims)[: max(limit, 0)]
        return [(self._ids[i], float(sims[i])) for i in top if float(sims[i]) > 0.0]


class ChromaVectorBackend(_StoreDerivedBackend):
    """Approximate nearest neighbor over a Chroma (HNSW) collection."""

    name = "chroma"

    def __init__(self, store: Store, *, persist_path: str | Path | None = None) -> None:
        super().__init__(store)
        import chromadb  # local import: optional dependency
        from chromadb.config import Settings as ChromaSettings

        # Telemetry off: this is a local, embedded index — no phone-home.
        self._persist_path = Path(persist_path) if persist_path else None
        self._persist_mode = "disk" if self._persist_path is not None else "ephemeral"
        self._signature_path = self._persist_path / "chunks.signature.json" if self._persist_path else None
        if self._persist_path is not None:
            self._persist_path.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(
                path=str(self._persist_path),
                settings=ChromaSettings(anonymized_telemetry=False),
            )
        else:
            self._client = chromadb.EphemeralClient(ChromaSettings(anonymized_telemetry=False))
        self._collection = None

    def _ensure_fresh(self) -> None:
        signature = self._store.embedding_index_signature()
        if not self._force_rebuild and signature == self._signature:
            return
        if (
            not self._force_rebuild
            and self._persist_path is not None
            and self._load_persisted_signature() == signature
        ):
            try:
                self._collection = self._client.get_collection("chunks")
                if int(self._collection.count()) > 0 or signature == (0, 0):
                    self._signature = signature
                    return
            except Exception:  # noqa: BLE001 - missing/stale collection, rebuild below
                pass
        ids: list[int] = []
        vectors: list[list[float]] = []
        for chunk in self._store.iter_chunks_with_embeddings():
            if chunk.embedding:
                ids.append(chunk.id)
                vectors.append(list(chunk.embedding))
        self._rebuild(ids, vectors)
        self._signature = signature
        self._force_rebuild = False
        self._store_persisted_signature(signature)

    def _rebuild(self, ids: list[int], vectors: list[list[float]]) -> None:
        # Recreate the collection so removed/changed chunks never linger.
        try:
            self._client.delete_collection("chunks")
        except Exception:  # noqa: BLE001 - first build, nothing to delete
            pass
        self._collection = self._client.create_collection(
            name="chunks", metadata={"hnsw:space": "cosine"}
        )
        if ids:
            self._collection.add(ids=[str(i) for i in ids], embeddings=vectors)

    def _load_persisted_signature(self) -> tuple[int, int] | None:
        if self._signature_path is None or not self._signature_path.exists():
            return None
        try:
            data = json.loads(self._signature_path.read_text(encoding="utf-8"))
            signature = data.get("embedding_index_signature")
            if isinstance(signature, list) and len(signature) == 2:
                return int(signature[0]), int(signature[1])
        except Exception:  # noqa: BLE001 - malformed sidecar means rebuild from SQLite
            return None
        return None

    def _store_persisted_signature(self, signature: tuple[int, int]) -> None:
        if self._signature_path is None:
            return
        self._signature_path.write_text(
            json.dumps({"embedding_index_signature": list(signature)}, sort_keys=True),
            encoding="utf-8",
        )

    def search(self, query_embedding: Sequence[float], *, limit: int) -> list[tuple[int, float]]:
        if not query_embedding:
            return []
        self._ensure_fresh()
        if self._collection is None or limit <= 0:
            return []
        try:
            res = self._collection.query(query_embeddings=[list(query_embedding)], n_results=limit)
        except Exception:  # noqa: BLE001 - dimension mismatch / empty collection
            return []
        ids = (res.get("ids") or [[]])[0]
        distances = (res.get("distances") or [[]])[0]
        out: list[tuple[int, float]] = []
        for raw_id, distance in zip(ids, distances):
            # Chroma cosine "distance" is 1 - cosine_similarity.
            similarity = 1.0 - float(distance)
            if similarity > 0.0:
                out.append((int(raw_id), similarity))
        return out

    def health(self) -> dict:
        return {
            "vector_backend": self.name,
            "chroma_persist_mode": self._persist_mode,
            "chroma_persist_path": self._persist_path.name if self._persist_path else None,
            "chroma_persist_enabled": self._persist_mode == "disk",
        }


def _available(name: str) -> bool:
    import importlib.util

    module = {"numpy": "numpy", "chroma": "chromadb"}.get(name)
    return module is not None and importlib.util.find_spec(module) is not None


def resolve_backend_name(requested: str | None = None) -> str:
    """Resolve the configured backend to a concrete, available name (graceful fallback)."""

    name = (requested if requested is not None else os.environ.get("DEEPRESEARCH_VECTOR_BACKEND", "auto"))
    name = (name or "auto").strip().lower()
    if name not in VALID_BACKENDS:
        logger.warning("Unknown DEEPRESEARCH_VECTOR_BACKEND=%r; using auto", name)
        name = "auto"

    if name == "auto":
        for candidate in ("chroma", "numpy"):
            if _available(candidate):
                return candidate
        return "bruteforce"

    if name in ("numpy", "chroma") and not _available(name):
        logger.warning(
            "DEEPRESEARCH_VECTOR_BACKEND=%s requested but the dependency is missing; "
            "falling back to bruteforce.",
            name,
        )
        return "bruteforce"
    return name


def create_vector_backend(
    store: Store,
    *,
    requested: str | None = None,
    chroma_persist_enabled: bool = False,
    chroma_persist_path: str | Path | None = None,
    chroma_ephemeral_forced: bool = False,
) -> VectorBackend:
    """Build the configured vector backend, falling back gracefully when ANN libs are absent."""

    name = resolve_backend_name(requested)
    try:
        if name == "chroma":
            persist_path = (
                Path(chroma_persist_path)
                if chroma_persist_enabled and not chroma_ephemeral_forced and chroma_persist_path
                else None
            )
            return ChromaVectorBackend(store, persist_path=persist_path)
        if name == "numpy":
            return NumpyVectorBackend(store)
    except Exception as error:  # noqa: BLE001 - never let an index build take retrieval offline
        logger.warning("Vector backend %s failed to initialize (%s); using bruteforce", name, error)
    return BruteForceVectorBackend(store)
