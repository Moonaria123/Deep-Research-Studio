"""T46 independent local/memory/external retrieval plan integration."""

from __future__ import annotations

from collections import defaultdict
import json

import pytest

from deepresearch_core.baseline.adapters import LLMResult
from deepresearch_core.baseline.budget import Mode, RunRequest
from deepresearch_core.baseline.errors import BaselineError, ErrorCode
from deepresearch_core.baseline.indexing import Indexer
from deepresearch_core.baseline.library_index import LibraryIndexer
from deepresearch_core.baseline.library_retrieval import LibraryRetriever
from deepresearch_core.baseline.memory import MemoryEngine, MemoryLayer
from deepresearch_core.baseline.orchestrator import Orchestrator
from deepresearch_core.baseline.retrieval import EvidenceItem, Retriever
from deepresearch_core.baseline.store import Store
from tests.baseline.fakes import FakeSearch


def _orchestrator(fake_llm, search, *, store=None, library=None, memory=None):
    store = store or Store(":memory:")
    memory = memory or MemoryEngine(store)
    return Orchestrator(
        store=store,
        llm=fake_llm,
        retriever=Retriever(store, embedder=fake_llm),
        indexer=Indexer(store, embedder=fake_llm),
        search=search,
        memory=memory,
        library_retriever=library,
        enforce_grounding_gate=True,
    ), store


def test_lane_failure_or_empty_does_not_skip_other_planned_lanes(fake_llm):
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
        fake_llm,
        RecordingSearch({"https://example.test/q": "quantum computing qubits use quantum state information"}),
        store=store,
        library=ExplodingLocal(),
        memory=RecordingMemory(store),
    )

    outcome = orch.run(
        RunRequest(query="quantum computing qubits", mode=Mode.FAST, memory_policy="off")
    )

    assert order[:3] == ["local", "memory", "external"]
    assert outcome.status == "completed_degraded"
    outcomes = outcome.to_dict()["retrieval_outcomes"]
    assert [(item["lane"], item["status"]) for item in outcomes] == [
        ("local", "failed"),
        ("memory", "empty"),
        ("external", "completed"),
    ]
    assert "retrieval_local_failed" in outcome.to_dict()["degraded_reasons"]
    store.close()


def test_all_three_lanes_empty_delivers_report_with_advisory_warning(fake_llm):
    orch, store = _orchestrator(fake_llm, FakeSearch({}))

    outcome = orch.run(RunRequest(query="no matching evidence token", mode=Mode.FAST, memory_policy="off"))
    result = store.get_run(outcome.run_id)
    assert result["status"] in {"completed", "completed_degraded"}
    assert result["result"]["answer"]["text"]
    assert result["result"]["answer"]["confidence_label"] == "uncertain"
    assert "zero_citable_evidence" in result["result"]["confidence_warning"]["reasons"]
    assert result["result"]["verification"]["status"] == "completed"
    store.close()


def test_external_lane_builds_default_indexer_when_search_is_supplied(fake_llm):
    store = Store(":memory:")
    orch = Orchestrator(
        store=store,
        llm=fake_llm,
        retriever=Retriever(store, embedder=fake_llm),
        search=FakeSearch(
            {"https://example.test/q": "quantum computing qubits use quantum state information"}
        ),
        enforce_grounding_gate=True,
    )

    outcome = orch.run(
        RunRequest(query="quantum computing qubits", mode=Mode.FAST, memory_policy="off")
    )

    external = next(
        item for item in outcome.to_dict()["retrieval_outcomes"] if item["lane"] == "external"
    )
    assert external["status"] == "completed"
    assert external["hit_count"] > 0
    store.close()


def test_library_evidence_is_citable_even_when_library_background_scope_is_off(fake_llm):
    store = Store(":memory:")
    LibraryIndexer(store, indexer=Indexer(store, embedder=fake_llm)).index_note(
        note_id="note_1",
        content="AlphaMemo internal note confirms the launch target is July 2026.",
    )
    library = LibraryRetriever(store, embedder=fake_llm)
    orch, store = _orchestrator(fake_llm, FakeSearch({}), store=store, library=library)

    outcome = orch.run(
        RunRequest(
            query="AlphaMemo launch target July 2026",
            mode=Mode.FAST,
            library_scope="off",
            memory_policy="off",
        )
    )

    citation = outcome.answer.citations[0]
    assert citation.origin == "local"
    assert citation.artifact_type == "library_note"
    assert citation.artifact_id == "note_1"
    assert citation.url == ""
    assert citation.retrieved_at is not None
    store.close()


def test_archived_memory_remains_citable_when_memory_background_policy_is_off(fake_llm):
    store = Store(":memory:")
    memory = MemoryEngine(store, embedder=fake_llm)
    memory_id = memory.remember(
        MemoryLayer.WORKSPACE,
        "default",
        "Project Orion launch target is July 2026.",
    )
    assert store.archive_memory(memory_id)
    orch, store = _orchestrator(fake_llm, FakeSearch({}), store=store, memory=memory)

    outcome = orch.run(
        RunRequest(
            query="Project Orion launch target July 2026",
            mode=Mode.FAST,
            memory_policy="off",
        )
    )

    citation = outcome.answer.citations[0]
    assert citation.origin == "memory"
    assert citation.artifact_id == memory_id
    assert citation.source_status == "archived"
    store.close()


def test_saturated_memory_lane_cannot_evict_local_and_external_evidence(fake_llm):
    class LocalEvidence:
        def retrieve_evidence(self, *args, **kwargs):  # noqa: ANN002, ANN003
            return [
                EvidenceItem(
                    chunk_id=-1,
                    document_id="",
                    url="",
                    title="Local note",
                    text="ParallelFusionMarker local evidence supports the research question.",
                    score=0.01,
                    origin="local",
                    artifact_type="library_note",
                    artifact_id="note-1",
                    provenance_refs=["local:library_note:note-1"],
                )
            ]

    class SaturatedMemory:
        def recall_evidence(self, **kwargs):  # noqa: ANN003
            return [
                EvidenceItem(
                    chunk_id=-(index + 2),
                    document_id="",
                    url="",
                    title=f"Memory {index}",
                    text=f"ParallelFusionMarker memory evidence {index} supports the research question.",
                    score=1.0 - index * 0.01,
                    origin="memory",
                    artifact_type="memory",
                    artifact_id=f"memory-{index}",
                    provenance_refs=[f"memory:memory:memory-{index}"],
                )
                for index in range(5)
            ]

    store = Store(":memory:")
    orch, store = _orchestrator(
        fake_llm,
        FakeSearch(
            {
                "https://example.test/external": (
                    "ParallelFusionMarker external evidence supports the research question."
                )
            }
        ),
        store=store,
        library=LocalEvidence(),
        memory=SaturatedMemory(),
    )

    outcome = orch.run(
        RunRequest(
            query="ParallelFusionMarker research question",
            mode=Mode.FAST,
            memory_policy="off",
            overrides={"top_k": 3},
        )
    )

    assert {source["origin"] for source in outcome.to_dict()["sources"]} == {
        "local",
        "memory",
        "external",
    }
    store.close()


def test_retrieved_conflict_cannot_be_hidden_by_separate_single_citation_claims():
    class ConflictAwareLLM:
        name = "conflict-aware"

        def __init__(self):
            self.generation_messages: list[list[dict]] = []

        def complete(self, messages, *, response_format=None, **kwargs):  # noqa: ANN001, ANN003
            if response_format is not None:
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
                    model=self.name,
                )
            self.generation_messages.append(list(messages))
            return LLMResult(
                text=(
                    "Acme revenue increased substantially in 2025. [c1]\n\n"
                    "Acme revenue decreased substantially in 2025. [c2]"
                ),
                model=self.name,
            )

        def embed(self, texts, *, model=None):  # noqa: ANN001, ARG002
            return [[0.0] for _ in texts]

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

    llm = ConflictAwareLLM()
    orch, store = _orchestrator(
        llm,
        FakeSearch({}),
        library=ConflictingLocalEvidence(),
    )

    outcome = orch.run(
        RunRequest(query="Acme revenue direction in 2025", mode=Mode.FAST, memory_policy="off")
    )

    assert all(len(citation_ids) == 1 for citation_ids in outcome.answer.claim_map.values())
    assert outcome.answer.confidence_label == "uncertain"
    assert len(outcome.answer.conflicts) == 1
    assert set(outcome.answer.conflicts[0].evidence_ids) == {"c1", "c2"}
    saved = store.get_run(outcome.run_id)["result"]
    assert "source_conflict" in saved["confidence_warning"]["reasons"]
    assert saved["answer"]["text"] == outcome.answer.text
    assert saved["answer"]["conflicts"]
    assert all(
        "Unresolved evidence conflict" not in message["content"]
        for messages in llm.generation_messages
        for message in messages
    )
    store.close()


@pytest.mark.parametrize("mode", [Mode.FAST, Mode.MULTIPLE, Mode.DEEP])
def test_every_research_query_registers_all_three_lanes(fake_llm, mode):
    orch, store = _orchestrator(
        fake_llm,
        FakeSearch({"https://example.test/q": "quantum computing qubits use quantum state information"}),
    )

    outcome = orch.run(RunRequest(query="quantum computing qubits", mode=mode, memory_policy="off"))

    groups: dict[str, set[str]] = defaultdict(set)
    for item in outcome.to_dict()["retrieval_outcomes"]:
        group, lane = item["lane_id"].rsplit(":", 1)
        groups[group].add(lane)
    assert groups
    assert all(lanes == {"local", "memory", "external"} for lanes in groups.values())
    store.close()


def test_latest_query_adds_three_lane_temporal_check_and_degrades_without_fact_time(fake_llm):
    orch, store = _orchestrator(
        fake_llm,
        FakeSearch({"https://example.test/q": "Project Orion has a current launch status."}),
    )

    outcome = orch.run(
        RunRequest(query="What is the latest Project Orion launch status today?", mode=Mode.FAST)
    )
    payload = outcome.to_dict()

    assert payload["status"] == "completed_degraded"
    assert len(payload["retrieval_outcomes"]) == 6
    assert payload["temporal_coverage"]["intent"] == "latest"
    assert payload["temporal_coverage"]["status"] == "unknown"
    assert payload["temporal_coverage"]["note_required"] is True
    assert "temporal_coverage_unknown" in payload["degraded_reasons"]
    store.close()


def test_historical_as_of_contract_is_injected_into_generation_prompt(fake_llm):
    orch, store = _orchestrator(
        fake_llm,
        FakeSearch({"https://example.test/q": "Project Orion status evidence from 2024."}),
    )

    orch.run(RunRequest(query="Project Orion status as of 2024-06-30", mode=Mode.FAST))

    system_prompts = [
        message["content"]
        for call in fake_llm.calls
        for message in call
        if message.get("role") == "system"
    ]
    user_prompts = [
        message["content"]
        for call in fake_llm.calls
        for message in call
        if message.get("role") == "user"
    ]
    assert any("Historical as-of rule" in prompt for prompt in system_prompts)
    assert all("Historical as-of rule" not in prompt for prompt in user_prompts)
    assert any("retrieved_at is not a fact-time signal" in prompt for prompt in system_prompts)
    assert all("retrieved_at is not a fact-time signal" not in prompt for prompt in user_prompts)
    store.close()
