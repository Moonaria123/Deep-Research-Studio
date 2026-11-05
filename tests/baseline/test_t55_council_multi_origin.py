"""T55 Council parity with the shared multi-origin evidence plane."""

from __future__ import annotations

import json

import pytest

from deepresearch_core.baseline.adapters import Document, LLMResult, SearchHit
from deepresearch_core.baseline.budget import Mode, RunRequest, retrieval_budget_for
from deepresearch_core.baseline.errors import BaselineError, ErrorCode
from deepresearch_core.baseline.indexing import Indexer
from deepresearch_core.baseline.library_index import LibraryIndexer
from deepresearch_core.baseline.library_retrieval import LibraryRetriever
from deepresearch_core.baseline.memory import MemoryEngine, MemoryLayer
from deepresearch_core.baseline.orchestrator import Orchestrator
from deepresearch_core.baseline.retrieval import EvidenceItem, Retriever
from deepresearch_core.baseline.store import Store
from tests.baseline.fakes import FakeLLM, FakeSearch


def _orchestrator(
    llm: FakeLLM,
    search: FakeSearch,
    *,
    store: Store | None = None,
    memory: MemoryEngine | None = None,
    library: LibraryRetriever | None = None,
    members: list[FakeLLM] | None = None,
) -> tuple[Orchestrator, Store]:
    store = store or Store(":memory:")
    memory = memory or MemoryEngine(store, embedder=llm)
    return (
        Orchestrator(
            store=store,
            llm=llm,
            retriever=Retriever(store, embedder=llm),
            indexer=Indexer(store, embedder=llm),
            search=search,
            memory=memory,
            library_retriever=library,
            council_members=members or [FakeLLM("member-a"), FakeLLM("member-b")],
            enforce_grounding_gate=True,
        ),
        store,
    )


def test_council_retrieval_budget_enables_bounded_fetch_without_recovery() -> None:
    budget = retrieval_budget_for(Mode.COUNCIL)

    assert budget.results_per_query == 8
    assert budget.fetches_max == 4
    assert budget.fetch_concurrency == 3
    assert budget.recovery_fetches_max == 0
    assert budget.writer_reserve_s == 25.0
    assert budget.finalize_reserve_s == 3.0
    assert budget.cleanup_reserve_s == 2.0


def test_council_lane_failure_or_empty_does_not_skip_other_planned_lanes() -> None:
    order: list[str] = []

    class ExplodingLocal:
        def retrieve_evidence(self, *args, **kwargs):  # noqa: ANN002, ANN003
            order.append("local")
            raise RuntimeError("local unavailable")

    class RecordingMemory(MemoryEngine):
        def recall_evidence(self, **kwargs):  # noqa: ANN003
            order.append("memory")
            return []

    class RecordingSearch(FakeSearch):
        def search(self, query, *, count=5, options=None):  # noqa: ANN001
            order.append("external")
            return super().search(query, count=count, options=options)

    store = Store(":memory:")
    orch, store = _orchestrator(
        FakeLLM("judge"),
        RecordingSearch(
            {
                "https://example.test/q": (
                    "Quantum computing qubits use quantum state information."
                )
            }
        ),
        store=store,
        library=ExplodingLocal(),  # type: ignore[arg-type]
        memory=RecordingMemory(store),
    )

    outcome = orch.run(
        RunRequest(
            query="quantum computing qubits",
            mode=Mode.COUNCIL,
            memory_policy="off",
        )
    )

    assert order[:3] == ["local", "memory", "external"]
    assert outcome.status == "completed_degraded"
    assert [
        (item["lane"], item["status"])
        for item in outcome.to_dict()["retrieval_outcomes"]
    ] == [
        ("local", "failed"),
        ("memory", "empty"),
        ("external", "completed"),
    ]
    assert "retrieval_local_failed" in outcome.to_dict()["degraded_reasons"]
    store.close()


def test_council_uses_final_mixed_origin_snapshot_and_three_primary_outcomes() -> None:
    llm = FakeLLM("judge")
    store = Store(":memory:")
    indexer = Indexer(store, embedder=llm)
    LibraryIndexer(store, indexer=indexer).index_note(
        note_id="note-t55",
        content="ParallelFusionMarker local evidence supports the research question.",
    )
    library = LibraryRetriever(store, embedder=llm)
    memory = MemoryEngine(store, embedder=llm)
    memory_id = memory.remember(
        MemoryLayer.WORKSPACE,
        "default",
        "ParallelFusionMarker archived memory supports the research question.",
    )
    assert store.archive_memory(memory_id)
    orch, store = _orchestrator(
        llm,
        FakeSearch(
            {
                "https://example.test/t55": (
                    "ParallelFusionMarker external evidence supports the research question."
                )
            }
        ),
        store=store,
        memory=memory,
        library=library,
    )

    outcome = orch.run(
        RunRequest(
            query="ParallelFusionMarker research question",
            mode=Mode.COUNCIL,
            memory_policy="off",
            library_scope="off",
            overrides={"top_k": 3},
        )
    )
    payload = outcome.to_dict()

    assert [(item["lane_id"], item["lane"]) for item in payload["retrieval_outcomes"]] == [
        ("primary:local", "local"),
        ("primary:memory", "memory"),
        ("primary:external", "external"),
    ]
    assert {source["origin"] for source in payload["sources"]} == {
        "local",
        "memory",
        "external",
    }
    assert {citation.origin for citation in outcome.answer.citations} == {
        "local",
        "memory",
        "external",
    }
    assert any(
        citation.origin == "memory" and citation.source_status == "archived"
        for citation in outcome.answer.citations
    )
    assert payload["recovery_searches_max"] == 0
    assert payload["recovery_searches_used"] == 0
    store.close()


def test_council_retrieval_uses_research_focus_not_output_constraints() -> None:
    llm = FakeLLM("judge")
    orch, store = _orchestrator(
        llm,
        FakeSearch(
            {
                "https://www.sqlite.org/fts5.html": (
                    "SQLite FTS5 is a virtual table module for full-text search."
                )
            }
        ),
    )

    outcome = orch.run(
        RunRequest(
            query=(
                "What is SQLite FTS5? Use one authoritative source and answer in two concise sentences."
            ),
            mode=Mode.COUNCIL,
            memory_policy="off",
        )
    )
    payload = outcome.to_dict()
    focus = payload["research_intent"]["research_focus"]

    assert focus
    assert "authoritative source" not in focus.lower()
    assert "two concise sentences" not in focus.lower()
    assert {item["query"] for item in payload["retrieval_outcomes"]} == {focus}
    store.close()


def test_council_demand_fetches_an_unknown_snippet_within_its_normal_budget() -> None:
    class FetchRequiredSearch:
        name = "fetch-required"

        def __init__(self) -> None:
            self.fetches = 0

        def search(self, query: str, *, count: int = 5, options=None):  # noqa: ANN001, ARG002
            return [
                SearchHit(
                    title="SQLite FTS5",
                    url="https://www.sqlite.org/fts5.html",
                    snippet="",
                    source="fake",
                    rank=1,
                )
            ]

        def fetch(self, url: str) -> Document:
            self.fetches += 1
            return Document(
                url=url,
                title="SQLite FTS5",
                text="SQLite FTS5 is a virtual table module that provides full-text search.",
                content_type="web",
                metadata={"fetch_provider": "fake-fetch"},
            )

    search = FetchRequiredSearch()
    orch, store = _orchestrator(FakeLLM("judge"), search)  # type: ignore[arg-type]

    outcome = orch.run(
        RunRequest(
            query="What is SQLite FTS5?",
            mode=Mode.COUNCIL,
            memory_policy="off",
        )
    )
    payload = outcome.to_dict()

    assert search.fetches == 1
    assert payload["retrieval_budget"]["fetches_used"] == 1
    assert payload["retrieval_budget"]["fetches_used"] <= 4
    assert payload["retrieval_budget"]["recovery_fetches_used"] == 0
    assert payload["sources"][0]["fetch_provider"] == "fake-fetch"
    store.close()


def test_latest_council_adds_three_temporal_outcomes_and_discloses_gap() -> None:
    llm = FakeLLM("judge")
    orch, store = _orchestrator(
        llm,
        FakeSearch(
            {
                "https://example.test/orion": (
                    "Project Orion has a current launch status supported by this source."
                )
            }
        ),
    )

    outcome = orch.run(
        RunRequest(
            query="What is the latest Project Orion launch status today?",
            mode=Mode.COUNCIL,
            memory_policy="off",
        )
    )
    payload = outcome.to_dict()

    assert [item["lane_id"] for item in payload["retrieval_outcomes"]] == [
        "primary:local",
        "primary:memory",
        "primary:external",
        "temporal:local",
        "temporal:memory",
        "temporal:external",
    ]
    assert payload["temporal_coverage"]["intent"] == "latest"
    assert payload["temporal_coverage"]["status"] == "unknown"
    assert payload["temporal_coverage"]["note_required"] is True
    assert payload["status"] == "completed_degraded"
    store.close()


def test_contested_pack_conflict_reaches_members_and_public_conflicts() -> None:
    class ConflictAwareLLM(FakeLLM):
        def complete(self, messages, *, response_format=None, **kwargs):  # noqa: ANN001, ANN003
            if response_format is not None:
                self.classification_calls = getattr(self, "classification_calls", 0) + 1
                self.calls.append(list(messages))
                return LLMResult(
                    text=json.dumps(
                        {
                            "candidates": [
                                {
                                    "claim_id": "claim-1",
                                    "evidence_ids": ["c1", "c2"],
                                    "same_scope": True,
                                    "claim_scope": "Acme revenue in 2025",
                                    "conflict_type": "peer_unresolved",
                                    "assessments": [
                                        {
                                            "evidence_id": "c1",
                                            "authority_directness": 2,
                                            "independent_support": 1,
                                            "effective_at": None,
                                        },
                                        {
                                            "evidence_id": "c2",
                                            "authority_directness": 2,
                                            "independent_support": 1,
                                            "effective_at": None,
                                        },
                                    ],
                                    "reason": "independent sources remain tied",
                                }
                            ]
                        }
                    ),
                    model=self.model,
                )
            return super().complete(messages, response_format=response_format, **kwargs)

    class ConflictingLocalEvidence:
        def retrieve_evidence(self, *args, **kwargs):  # noqa: ANN002, ANN003
            return [
                EvidenceItem(
                    chunk_id=-1,
                    document_id="source-a",
                    url="",
                    title="Source A",
                    text="Acme revenue increased substantially in 2025.",
                    score=0.9,
                    origin="local",
                    artifact_type="library_note",
                    artifact_id="note-a",
                    provenance_refs=["root:a"],
                ),
                EvidenceItem(
                    chunk_id=-2,
                    document_id="source-b",
                    url="",
                    title="Source B",
                    text="Acme revenue decreased substantially in 2025.",
                    score=0.8,
                    origin="local",
                    artifact_type="library_note",
                    artifact_id="note-b",
                    provenance_refs=["root:b"],
                ),
            ]

    llm = ConflictAwareLLM("conflict-aware")
    members = [FakeLLM("member-a"), FakeLLM("member-b")]
    orch, store = _orchestrator(
        llm,
        FakeSearch({}),
        library=ConflictingLocalEvidence(),  # type: ignore[arg-type]
        members=members,
    )

    outcome = orch.run(
        RunRequest(
            query="Acme revenue direction in 2025",
            mode=Mode.COUNCIL,
            memory_policy="off",
        )
    )

    assert outcome.answer.confidence_label == "uncertain"
    assert len(outcome.answer.conflicts) == 1
    assert llm.classification_calls == 1
    assert set(outcome.answer.conflicts[0].evidence_ids) == {"c1", "c2"}
    assert all(
        not any(
            message["role"] == "system"
            and '"resolution":"contested"' in message["content"]
            for message in member.calls[0]
        )
        for member in members
    )
    persisted = store.get_run(outcome.run_id)["result"]
    assert persisted["verification"]["status"] in {"completed", "failed"}
    assert persisted["confidence_warning"]["acknowledged"] is False
    store.close()


def test_low_member_agreement_does_not_create_an_evidence_conflict() -> None:
    class FixedLLM(FakeLLM):
        def __init__(self, model: str, text: str) -> None:
            super().__init__(model)
            self.text = text

        def complete(self, messages, **kwargs):  # noqa: ANN001, ANN003
            self.calls.append(list(messages))
            return LLMResult(text=self.text, model=self.model)

    members = [
        FixedLLM("member-a", "Ocean tides follow lunar gravity. [c1]"),
        FixedLLM("member-b", "Coastal water levels respond to celestial attraction. [c1]"),
    ]
    orch, store = _orchestrator(
        FakeLLM("judge"),
        FakeSearch(
            {
                "https://example.test/tides": (
                    "Ocean tides and coastal water levels respond to lunar gravity and celestial attraction."
                )
            }
        ),
        members=members,
    )

    outcome = orch.run(
        RunRequest(
            query="What causes ocean tides and coastal water level changes?",
            mode=Mode.COUNCIL,
            memory_policy="off",
            council_strategy="parallel_vote",
        )
    )

    assert outcome.answer.confidence_label == "contested"
    assert outcome.answer.conflicts == []
    assert not any(
        event["type"] == "evidence_conflict_decision"
        for event in store.list_events(outcome.run_id)
    )
    store.close()


def test_council_delivers_without_eligible_evidence_then_marks_low_confidence() -> None:
    judge = FakeLLM("judge")
    members = [FakeLLM("member-a"), FakeLLM("member-b")]
    orch, store = _orchestrator(judge, FakeSearch({}), members=members)

    outcome = orch.run(
        RunRequest(
            query="No supporting evidence exists for this unique marker.",
            mode=Mode.COUNCIL,
            memory_policy="off",
        )
    )

    assert outcome.answer.text.strip()
    assert all(member.calls for member in members)
    assert judge.calls
    run = store.get_run(outcome.run_id)
    assert run["status"] in {"completed", "completed_degraded"}
    assert run["result"]["answer"]["confidence_label"] == "uncertain"
    assert run["result"]["confidence_warning"]["acknowledged"] is False
    store.close()
