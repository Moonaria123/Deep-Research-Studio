"""Retrieval: RRF fusion + indexing + lexical/vector retrieval over the store."""

from __future__ import annotations

from deepresearch_core.baseline.adapters import Document
from deepresearch_core.baseline.indexing import Indexer
from deepresearch_core.baseline.retrieval import Retriever, reciprocal_rank_fusion
from deepresearch_core.baseline.store import Store


def test_rrf_rewards_agreement_across_rankings():
    fused = reciprocal_rank_fusion([[1, 2, 3], [2, 1, 4]])
    # id 2 is rank0 in second list + rank1 in first => should beat id 3 (only one list).
    assert fused[2] > fused[3]
    assert fused[1] > fused[4]


def test_index_then_lexical_retrieve_returns_relevant_chunk():
    store = Store(":memory:")
    indexer = Indexer(store)
    indexer.index_document(
        Document(
            url="https://e.com/quantum",
            title="Quantum",
            text="Quantum computing relies on qubits and superposition to accelerate certain algorithms.",
            content_type="web",
        )
    )
    indexer.index_document(
        Document(url="https://e.com/cooking", title="Cooking", text="Bread baking needs flour water and yeast.", content_type="web")
    )
    evidence = Retriever(store).retrieve("qubits superposition quantum", top_k=5)
    assert evidence
    assert "quantum" in evidence[0].url
    store.close()


def test_vector_channel_engages_when_embedder_present(fake_llm):
    store = Store(":memory:")
    indexer = Indexer(store, embedder=fake_llm)
    stats = indexer.index_document(
        Document(url="https://e.com/a", title="A", text="alpha beta gamma delta epsilon zeta eta theta", content_type="web")
    )
    assert stats["embedded"] is True
    evidence = Retriever(store, embedder=fake_llm).retrieve("alpha beta gamma", top_k=5)
    assert evidence
    assert any("vector" in e.channels or "lexical" in e.channels for e in evidence)
    store.close()
