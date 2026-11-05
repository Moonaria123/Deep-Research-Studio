"""Pluggable vector-search backends (DR-ITER-020)."""

from __future__ import annotations

import importlib.util

import pytest

from deepresearch_core.baseline.store import Store
from deepresearch_core.baseline.vector_backend import (
    BruteForceVectorBackend,
    ChromaVectorBackend,
    NumpyVectorBackend,
    create_vector_backend,
    resolve_backend_name,
)

_HAS_NUMPY = importlib.util.find_spec("numpy") is not None
_HAS_CHROMA = importlib.util.find_spec("chromadb") is not None


def _seed(store: Store) -> dict[str, int]:
    """Insert 3 orthogonal-ish 3-D embeddings; return {label: chunk_id}."""
    doc = store.upsert_document(url="u", title="t", text="body", content_type="web", doc_hash="h")
    return {
        "x": store.add_chunk(document_id=doc, chunk_index=0, text="x axis", embedding=[1.0, 0.0, 0.0]),
        "y": store.add_chunk(document_id=doc, chunk_index=1, text="y axis", embedding=[0.0, 1.0, 0.0]),
        "z": store.add_chunk(document_id=doc, chunk_index=2, text="z axis", embedding=[0.0, 0.0, 1.0]),
    }


def test_resolve_backend_name_explicit_and_unknown():
    assert resolve_backend_name("bruteforce") == "bruteforce"
    # Unknown value degrades to auto-resolution (never raises).
    assert resolve_backend_name("nonsense") in {"bruteforce", "numpy", "chroma"}


def test_auto_prefers_available_ann():
    name = resolve_backend_name("auto")
    if _HAS_CHROMA:
        assert name == "chroma"
    elif _HAS_NUMPY:
        assert name == "numpy"
    else:
        assert name == "bruteforce"


def test_bruteforce_search_orders_by_cosine():
    store = Store(":memory:")
    ids = _seed(store)
    backend = BruteForceVectorBackend(store)
    res = backend.search([0.9, 0.1, 0.0], limit=2)
    assert res[0][0] == ids["x"]  # closest to the x axis
    assert res[0][1] > res[1][1]
    store.close()


@pytest.mark.skipif(not _HAS_NUMPY, reason="numpy not installed")
def test_numpy_backend_matches_bruteforce_top_hit():
    store = Store(":memory:")
    ids = _seed(store)
    backend = NumpyVectorBackend(store)
    res = backend.search([0.0, 0.8, 0.2], limit=1)
    assert res and res[0][0] == ids["y"]
    store.close()


@pytest.mark.skipif(not _HAS_NUMPY, reason="numpy not installed")
def test_numpy_backend_rebuilds_on_new_chunk():
    store = Store(":memory:")
    _seed(store)
    backend = NumpyVectorBackend(store)
    backend.search([1.0, 0.0, 0.0], limit=1)  # build the cache
    doc = store.upsert_document(url="u2", title="t2", text="b", content_type="web", doc_hash="h2")
    new_id = store.add_chunk(document_id=doc, chunk_index=0, text="w", embedding=[1.0, 1.0, 1.0])
    res = backend.search([1.0, 1.0, 1.0], limit=1)
    assert res and res[0][0] == new_id  # cache invalidated by the staleness signature
    store.close()


@pytest.mark.skipif(not _HAS_CHROMA, reason="chromadb not installed")
def test_chroma_backend_matches_bruteforce_top_hit():
    store = Store(":memory:")
    ids = _seed(store)
    backend = ChromaVectorBackend(store)
    res = backend.search([0.1, 0.0, 0.9], limit=1)
    assert res and res[0][0] == ids["z"]
    store.close()


def test_create_vector_backend_returns_requested_or_fallback():
    store = Store(":memory:")
    assert isinstance(create_vector_backend(store, requested="bruteforce"), BruteForceVectorBackend)
    backend = create_vector_backend(store, requested="auto")
    assert backend.health()["vector_backend"] in {"bruteforce", "numpy", "chroma"}
    store.close()


@pytest.mark.skipif(not _HAS_CHROMA, reason="chromadb not installed")
def test_chroma_persistent_backend_reuses_disk_collection(tmp_path):
    db_path = tmp_path / "vectors.sqlite"
    persist_path = tmp_path / "chroma"
    store = Store(db_path)
    ids = _seed(store)
    backend = ChromaVectorBackend(store, persist_path=persist_path)
    assert backend.search([1.0, 0.0, 0.0], limit=1)[0][0] == ids["x"]
    assert (persist_path / "chunks.signature.json").exists()

    reloaded = ChromaVectorBackend(store, persist_path=persist_path)

    def _fail_rebuild(*_args, **_kwargs):  # noqa: ANN001
        raise AssertionError("persistent signature should avoid rebuild")

    reloaded._rebuild = _fail_rebuild  # type: ignore[method-assign]
    assert reloaded.search([1.0, 0.0, 0.0], limit=1)[0][0] == ids["x"]
    store.close()


@pytest.mark.skipif(not _HAS_CHROMA, reason="chromadb not installed")
def test_chroma_ephemeral_force_disables_persist_path(tmp_path):
    store = Store(":memory:")
    backend = create_vector_backend(
        store,
        requested="chroma",
        chroma_persist_enabled=True,
        chroma_persist_path=tmp_path / "chroma",
        chroma_ephemeral_forced=True,
    )

    health = backend.health()
    assert health["vector_backend"] == "chroma"
    assert health["chroma_persist_mode"] == "ephemeral"
    assert health["chroma_persist_enabled"] is False
    store.close()


def test_empty_store_returns_no_matches():
    store = Store(":memory:")
    for backend in (BruteForceVectorBackend(store), create_vector_backend(store)):
        assert backend.search([1.0, 0.0, 0.0], limit=5) == []
    store.close()
