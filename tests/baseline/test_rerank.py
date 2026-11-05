"""Lightweight embedding reranker: re-scores top-k by query/text cosine, no heavy deps."""

from __future__ import annotations

from deepresearch_core.baseline.retrieval import (
    EvidenceItem,
    build_reranker,
    embedding_reranker,
    http_reranker,
)

from tests.baseline.fakes import FakeLLM


def _item(chunk_id: int, text: str, score: float) -> EvidenceItem:
    return EvidenceItem(
        chunk_id=chunk_id,
        document_id=f"d{chunk_id}",
        url=f"https://e.com/{chunk_id}",
        title=f"Doc {chunk_id}",
        text=text,
        score=score,
    )


def test_reranker_promotes_query_relevant_item():
    rerank = embedding_reranker(FakeLLM())
    # Off-topic item has the higher initial score; on-topic item should be promoted.
    items = [
        _item(1, "banana smoothie recipe with yogurt", score=0.9),
        _item(2, "quantum computing qubits", score=0.1),
    ]
    out = rerank("quantum computing qubits", items)
    assert [i.chunk_id for i in out] == [2, 1]


def test_reranker_preserves_items_and_count():
    rerank = embedding_reranker(FakeLLM())
    items = [_item(1, "alpha", 0.5), _item(2, "beta", 0.4), _item(3, "gamma", 0.3)]
    out = rerank("alpha", items)
    assert {i.chunk_id for i in out} == {1, 2, 3}
    assert len(out) == 3


def test_reranker_degrades_gracefully_when_embedding_fails():
    class BrokenEmbedder(FakeLLM):
        def embed(self, texts, *, model=None):  # noqa: ARG002
            raise RuntimeError("no embeddings")

    rerank = embedding_reranker(BrokenEmbedder())
    items = [_item(1, "a", 0.5), _item(2, "b", 0.4)]
    out = rerank("a", items)
    # Original order preserved on failure.
    assert [i.chunk_id for i in out] == [1, 2]


def test_reranker_empty_input():
    rerank = embedding_reranker(FakeLLM())
    assert rerank("q", []) == []


# --- HTTP cross-encoder reranker (DR-ITER-025) ---------------------------------


def _patch_urlopen(monkeypatch, payload):
    import json as _json
    from contextlib import contextmanager

    class _Resp:
        def read(self):
            return _json.dumps(payload).encode("utf-8")

    @contextmanager
    def _fake_urlopen(request, timeout=None):  # noqa: ARG001
        yield _Resp()

    monkeypatch.setattr(
        "deepresearch_core.baseline.retrieval.urllib.request.urlopen", _fake_urlopen
    )


def test_http_reranker_applies_returned_order(monkeypatch):
    items = [_item(1, "a", 0.9), _item(2, "b", 0.5), _item(3, "c", 0.1)]
    # Endpoint says item index 2 (chunk 3) is most relevant, then 0, then 1.
    _patch_urlopen(
        monkeypatch,
        {"results": [{"index": 2, "relevance_score": 0.99},
                     {"index": 0, "relevance_score": 0.4},
                     {"index": 1, "relevance_score": 0.2}]},
    )
    rerank = http_reranker("https://rerank.example/v1/rerank")
    out = rerank("q", items)
    assert [i.chunk_id for i in out] == [3, 1, 2]


def test_http_reranker_falls_back_on_transport_error(monkeypatch):
    def _boom(request, timeout=None):  # noqa: ARG001
        raise OSError("connection refused")

    monkeypatch.setattr(
        "deepresearch_core.baseline.retrieval.urllib.request.urlopen", _boom
    )
    items = [_item(1, "a", 0.5), _item(2, "b", 0.4)]
    fallback_called = {"n": 0}

    def _fallback(query, evs):  # noqa: ARG001
        fallback_called["n"] += 1
        return list(reversed(evs))

    rerank = http_reranker("https://rerank.example", fallback=_fallback)
    out = rerank("q", items)
    assert fallback_called["n"] == 1
    assert [i.chunk_id for i in out] == [2, 1]


def test_http_reranker_falls_back_on_malformed_payload(monkeypatch):
    # results count mismatch -> treated as malformed -> preserve order (no fallback given).
    _patch_urlopen(monkeypatch, {"results": [{"index": 0, "relevance_score": 1.0}]})
    items = [_item(1, "a", 0.5), _item(2, "b", 0.4)]
    rerank = http_reranker("https://rerank.example")
    out = rerank("q", items)
    assert [i.chunk_id for i in out] == [1, 2]


def test_http_reranker_empty_input(monkeypatch):
    rerank = http_reranker("https://rerank.example")
    assert rerank("q", []) == []


def test_build_reranker_defaults_to_embedding(monkeypatch):
    monkeypatch.delenv("DEEPRESEARCH_RERANK_MODE", raising=False)
    rerank = build_reranker(FakeLLM())
    items = [_item(1, "banana", 0.9), _item(2, "quantum computing qubits", 0.1)]
    out = rerank("quantum computing qubits", items)
    assert [i.chunk_id for i in out] == [2, 1]


def test_build_reranker_off_returns_none(monkeypatch):
    monkeypatch.setenv("DEEPRESEARCH_RERANK_MODE", "off")
    assert build_reranker(FakeLLM()) is None


def test_build_reranker_http_without_url_falls_back_to_embedding(monkeypatch):
    monkeypatch.setenv("DEEPRESEARCH_RERANK_MODE", "http")
    monkeypatch.delenv("DEEPRESEARCH_RERANK_URL", raising=False)
    rerank = build_reranker(FakeLLM())
    assert rerank is not None
    # Behaves like the embedding reranker.
    items = [_item(1, "banana", 0.9), _item(2, "quantum", 0.1)]
    out = rerank("quantum", items)
    assert {i.chunk_id for i in out} == {1, 2}


def test_build_reranker_embedding_without_embedder_is_none(monkeypatch):
    monkeypatch.delenv("DEEPRESEARCH_RERANK_MODE", raising=False)
    assert build_reranker(None) is None
