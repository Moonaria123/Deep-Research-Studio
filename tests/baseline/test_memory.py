"""Three-layer memory engine: recall ranking by salience and decay."""

from __future__ import annotations

import json
import time
from pathlib import Path

from deepresearch_core.baseline.deadline import (
    ProviderCallBudget,
    current_provider_call_budget,
    use_provider_call_budget,
)
from deepresearch_core.baseline.memory import MemoryEngine, MemoryLayer, RecallConfig
from deepresearch_core.baseline.store import Store


_MEMORY_GOLDEN = Path(__file__).with_name("fixtures") / "memory_recall_v1.jsonl"


def _memory_cases() -> list[dict]:
    return [json.loads(line) for line in _MEMORY_GOLDEN.read_text(encoding="utf-8").splitlines() if line]


class _FixtureEmbedder:
    """Returns explicit corpus vectors, avoiding model/provider nondeterminism."""

    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self._vectors = vectors

    def embed(self, texts):
        return [list(self._vectors[text]) for text in texts]


class _CountingEmbedder:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed(self, texts):  # noqa: ANN001, ANN201
        values = [str(text) for text in texts]
        self.calls.append(values)
        return [[1.0] for _ in values]


class _BudgetRecordingEmbedder:
    def __init__(self) -> None:
        self.calls: list[tuple[list[str], float | None]] = []

    def embed(self, texts):  # noqa: ANN001, ANN201
        values = [str(text) for text in texts]
        budget = current_provider_call_budget()
        self.calls.append((values, budget.remaining_s() if budget is not None else None))
        return [[1.0] for _ in values]


def test_remember_and_recall_across_layers():
    engine = MemoryEngine(Store(":memory:"))
    engine.remember(MemoryLayer.SESSION, "s1", "session note", salience=1.0)
    engine.remember(MemoryLayer.WORKSPACE, "default", "workspace fact", salience=1.0)
    engine.remember(MemoryLayer.USER, "default", "user preference", salience=1.0)

    recalled = engine.recall(session_key="s1", workspace_key="default", user_key="default", limit=12)
    contents = {r["content"] for r in recalled}
    assert {"session note", "workspace fact", "user preference"} <= contents
    assert all("effective_salience" in r and "layer" in r for r in recalled)


def test_higher_salience_ranks_first():
    engine = MemoryEngine(Store(":memory:"))
    engine.remember(MemoryLayer.WORKSPACE, "default", "low", salience=0.2)
    engine.remember(MemoryLayer.WORKSPACE, "default", "high", salience=5.0)
    recalled = engine.recall(workspace_key="default", limit=12)
    assert recalled[0]["content"] == "high"


def test_archived_memory_is_excluded_from_recall():
    store = Store(":memory:")
    engine = MemoryEngine(store)
    archived_id = engine.remember(MemoryLayer.WORKSPACE, "default", "archived", salience=9.0)
    engine.remember(MemoryLayer.WORKSPACE, "default", "active", salience=1.0)
    assert store.archive_memory(archived_id) is True

    recalled = engine.recall(workspace_key="default", limit=12)

    assert [row["content"] for row in recalled] == ["active"]


def test_recall_ignores_empty_scope_keys():
    engine = MemoryEngine(Store(":memory:"))
    engine.remember(MemoryLayer.SESSION, "s1", "only session", salience=1.0)
    # No session_key supplied -> session layer skipped, nothing returned.
    recalled = engine.recall(workspace_key="", user_key="")
    assert recalled == []


# --- semantic recall (embedder) ----------------------------------------------


class _AxisEmbedder:
    """Maps a keyword to a one-hot axis so cosine similarity is deterministic."""

    _AXES = {"apple": [1.0, 0.0, 0.0], "orange": [0.0, 1.0, 0.0], "banana": [0.0, 0.0, 1.0]}

    def embed(self, texts):
        out = []
        for text in texts:
            vec = [0.0, 0.0, 0.0]
            for word, axis in self._AXES.items():
                if word in text.lower():
                    vec = list(axis)
                    break
            out.append(vec)
        return out


def test_remember_embeds_content_when_embedder_present():
    store = Store(":memory:")
    engine = MemoryEngine(store, embedder=_AxisEmbedder())
    engine.remember(MemoryLayer.WORKSPACE, "default", "apple pie recipe")
    rows = store.iter_memories_with_embeddings(namespace="workspace", scope_key="default")
    assert len(rows) == 1 and rows[0]["embedding"] == [1.0, 0.0, 0.0]


def test_remember_can_skip_optional_embedding_and_still_persist():
    store = Store(":memory:")
    embedder = _CountingEmbedder()
    engine = MemoryEngine(store, embedder=embedder)

    engine.remember(MemoryLayer.SESSION, "s1", "deadline-safe memory", embed=False)

    assert embedder.calls == []
    assert [row["content"] for row in engine.recall(session_key="s1")] == [
        "deadline-safe memory"
    ]
    assert store.iter_memories_with_embeddings(namespace="session", scope_key="s1") == []
    store.close()


def test_query_recall_ranks_by_semantic_similarity():
    engine = MemoryEngine(Store(":memory:"), embedder=_AxisEmbedder())
    # Equal salience: only semantics should decide ordering.
    engine.remember(MemoryLayer.WORKSPACE, "default", "all about apple", salience=1.0)
    engine.remember(MemoryLayer.WORKSPACE, "default", "all about banana", salience=1.0)
    recalled = engine.recall(workspace_key="default", query="apple tips", limit=12)
    assert [row["content"] for row in recalled] == ["all about apple"]
    assert recalled[0]["similarity"] >= RecallConfig().min_similarity


def test_recall_without_query_uses_salience_even_with_embedder():
    engine = MemoryEngine(Store(":memory:"), embedder=_AxisEmbedder())
    engine.remember(MemoryLayer.WORKSPACE, "default", "apple low", salience=0.1)
    engine.remember(MemoryLayer.WORKSPACE, "default", "banana high", salience=9.0)
    recalled = engine.recall(workspace_key="default", limit=12)
    assert recalled[0]["content"] == "banana high"


def test_recall_degrades_gracefully_without_embedder():
    # No embedder + a query => behaves like salience recall, never raises.
    engine = MemoryEngine(Store(":memory:"))
    engine.remember(MemoryLayer.WORKSPACE, "default", "note a", salience=1.0)
    recalled = engine.recall(workspace_key="default", query="anything")
    assert recalled and recalled[0]["content"] == "note a"


def test_evidence_recall_skips_query_embedding_without_stored_memory_vectors():
    store = Store(":memory:")
    MemoryEngine(store).remember(
        MemoryLayer.WORKSPACE,
        "default",
        "Alpha evidence from a lexical-only memory.",
    )
    embedder = _CountingEmbedder()

    evidence = MemoryEngine(store, embedder=embedder).recall_evidence(
        workspace_key="default",
        user_key="",
        query="Alpha evidence",
        limit=5,
    )

    assert evidence
    assert evidence[0].channels == ["memory_lexical"]
    assert embedder.calls == []
    store.close()


def test_evidence_recall_uses_lexical_hits_without_blocking_on_optional_query_embedding():
    store = Store(":memory:")
    embedder = _CountingEmbedder()
    engine = MemoryEngine(store, embedder=embedder)
    engine.remember(
        MemoryLayer.WORKSPACE,
        "default",
        "NASA seasons evidence stored in memory.",
    )
    embedder.calls.clear()
    budget = ProviderCallBudget(
        request_deadline=time.monotonic() + 10.0,
        phase_deadline=time.monotonic() + 8.0,
    )

    with use_provider_call_budget(budget):
        evidence = engine.recall_evidence(
            workspace_key="default",
            user_key="",
            query="NASA seasons",
            limit=5,
        )

    assert evidence
    assert evidence[0].channels == ["memory_lexical"]
    assert embedder.calls == []
    store.close()


def test_embedding_failure_does_not_break_remember():
    class _BoomEmbedder:
        def embed(self, texts):
            raise RuntimeError("backend down")

    store = Store(":memory:")
    engine = MemoryEngine(store, embedder=_BoomEmbedder())
    mid = engine.remember(MemoryLayer.WORKSPACE, "default", "still stored")
    assert mid
    # Stored without a vector; recall still works.
    assert engine.recall(workspace_key="default")[0]["content"] == "still stored"


def test_memory_recall_v1_golden_corpus_rejects_irrelevant_memories():
    cases = _memory_cases()
    assert len(cases) == 11
    false_recall_cases = 0

    for case in cases:
        vectors = {case["query"]: case["query_vector"]}
        vectors.update({memory["content"]: memory["vector"] for memory in case["memories"]})
        store = Store(":memory:")
        engine = MemoryEngine(store, embedder=_FixtureEmbedder(vectors))
        for memory in case["memories"]:
            layer = MemoryLayer(memory.get("layer", "workspace"))
            scope_key = {MemoryLayer.SESSION: "s", MemoryLayer.WORKSPACE: "w", MemoryLayer.USER: "u"}[layer]
            memory_id = engine.remember(
                layer,
                scope_key,
                memory["content"],
                salience=memory["salience"],
            )
            if memory.get("archived"):
                assert store.archive_memory(memory_id)

        recalled = engine.recall(
            session_key="s",
            workspace_key="w",
            user_key="u",
            query=case["query"],
            limit=12,
        )
        contents = {row["content"] for row in recalled}
        expected = set(case["expected"])
        unexpected = set(case["unexpected"])
        if expected:
            assert contents & expected, case["id"]
        elif contents:
            false_recall_cases += 1
        assert contents.isdisjoint(unexpected), case["id"]
        store.close()

    assert false_recall_cases / len(cases) < 0.10


def test_invalid_query_embedding_preserves_salience_fallback():
    engine = MemoryEngine(Store(":memory:"), embedder=_AxisEmbedder())
    engine.remember(MemoryLayer.WORKSPACE, "default", "apple low", salience=0.1)
    engine.remember(MemoryLayer.WORKSPACE, "default", "banana high", salience=9.0)

    recalled = engine.recall(workspace_key="default", query="unknown topic", limit=12)

    assert [row["content"] for row in recalled] == ["banana high", "apple low"]


def test_query_recall_rejects_unembedded_candidate_when_query_embedding_is_valid():
    class _PartialEmbedder:
        def embed(self, texts):
            vectors = {
                "query": [1.0, 0.0],
                "embedded": [1.0, 0.0],
                "unembedded": [],
            }
            return [vectors[text] for text in texts]

    engine = MemoryEngine(Store(":memory:"), embedder=_PartialEmbedder())
    engine.remember(MemoryLayer.WORKSPACE, "default", "embedded", salience=1.0)
    engine.remember(MemoryLayer.WORKSPACE, "default", "unembedded", salience=2.0)

    recalled = engine.recall(workspace_key="default", query="query", limit=12)

    assert [row["content"] for row in recalled] == ["embedded"]


# --- pre-run injection into the orchestrator prompt (DR-ITER-054) -------------


def _memory_orchestrator(fake_llm, fake_search, memory):
    from deepresearch_core.baseline.indexing import Indexer
    from deepresearch_core.baseline.orchestrator import Orchestrator
    from deepresearch_core.baseline.retrieval import Retriever

    store = Store(":memory:")
    orch = Orchestrator(
        store=store,
        llm=fake_llm,
        retriever=Retriever(store),
        indexer=Indexer(store),
        search=fake_search,
        memory=memory,
        enforce_grounding_gate=True,
    )
    return orch, store


def test_orchestrator_injects_workspace_memory(fake_llm, fake_search):
    """policy=workspace recalls the workspace layer into a do-not-cite block."""

    from deepresearch_core.baseline.budget import Mode, RunRequest

    memory = MemoryEngine(Store(":memory:"))
    memory.remember(
        MemoryLayer.WORKSPACE, "default", "User prefers concise bullet-point answers.", salience=2.0
    )
    orch, store = _memory_orchestrator(fake_llm, fake_search, memory)
    sid = store.create_session(title="t")
    orch.run(
        RunRequest(query="quantum computing qubits", mode=Mode.FAST, session_id=sid, memory_policy="workspace")
    )
    prompt = fake_llm.calls[-1][-1]["content"]
    assert "Relevant memories (do not cite as sources)" in prompt
    assert "User prefers concise bullet-point answers." in prompt
    store.close()


def test_orchestrator_caps_optional_background_memory_embedding(fake_llm, fake_search):
    """Background continuity enrichment cannot consume the Fast writer reserve."""

    from deepresearch_core.baseline.budget import Mode, RunRequest

    embedder = _BudgetRecordingEmbedder()
    memory = MemoryEngine(Store(":memory:"), embedder=embedder)
    memory.remember(
        MemoryLayer.WORKSPACE,
        "default",
        "quantum computing qubits",
        salience=2.0,
    )
    embedder.calls.clear()
    orch, store = _memory_orchestrator(fake_llm, fake_search, memory)
    sid = store.create_session(title="t")

    orch.run(
        RunRequest(
            query="quantum computing qubits",
            mode=Mode.FAST,
            session_id=sid,
            memory_policy="workspace",
        )
    )

    query_calls = [remaining for texts, remaining in embedder.calls if texts == ["quantum computing qubits"]]
    assert query_calls
    assert query_calls[0] is not None
    assert 0.0 < query_calls[0] <= 1.0
    store.close()


def test_orchestrator_memory_policy_off_injects_nothing(fake_llm, fake_search):
    """The backend default (off) never injects memory, even when memories exist."""

    from deepresearch_core.baseline.budget import Mode, RunRequest

    memory = MemoryEngine(Store(":memory:"))
    memory.remember(MemoryLayer.WORKSPACE, "default", "User prefers concise answers.", salience=2.0)
    orch, store = _memory_orchestrator(fake_llm, fake_search, memory)
    sid = store.create_session(title="t")
    orch.run(RunRequest(query="quantum computing qubits", mode=Mode.FAST, session_id=sid))  # default off
    prompt = fake_llm.calls[-1][-1]["content"]
    assert "Relevant memories" not in prompt
    store.close()


def test_orchestrator_workspace_policy_excludes_user_layer(fake_llm, fake_search):
    """policy=workspace must not pull the user layer (only workspace_user does)."""

    from deepresearch_core.baseline.budget import Mode, RunRequest

    memory = MemoryEngine(Store(":memory:"))
    memory.remember(MemoryLayer.WORKSPACE, "default", "workspace-scope fact", salience=2.0)
    memory.remember(MemoryLayer.USER, "default", "user-scope secret", salience=9.0)
    orch, store = _memory_orchestrator(fake_llm, fake_search, memory)
    sid = store.create_session(title="t")
    orch.run(
        RunRequest(query="quantum computing qubits", mode=Mode.FAST, session_id=sid, memory_policy="workspace")
    )
    prompt = fake_llm.calls[-1][-1]["content"]
    assert "workspace-scope fact" in prompt
    assert "user-scope secret" not in prompt
    store.close()
