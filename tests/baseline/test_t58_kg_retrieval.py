from __future__ import annotations

import json

import pytest

from deepresearch_core.baseline.budget import Mode, RunRequest
from deepresearch_core.baseline.errors import BaselineError
from deepresearch_core.baseline.indexing import Indexer
from deepresearch_core.baseline.knowledge_graph.retrieval import (
    KGRetrievalDiagnostics,
    KnowledgeGraphEvidenceRetriever,
)
from deepresearch_core.baseline.knowledge_graph.store_ops import KnowledgeGraphStore
from deepresearch_core.baseline.memory import MemoryEngine
from deepresearch_core.baseline.orchestrator import Orchestrator
from deepresearch_core.baseline.retrieval import Retriever
from deepresearch_core.baseline.retrieval_settings import (
    coerce_retrieval_settings,
    default_retrieval_settings,
)
from deepresearch_core.baseline.store import Store
from deepresearch_core.baseline.service import BaselineService
from deepresearch_core.settings import ProviderSettings
from tests.baseline.fakes import FakeSearch


def _activate_graph(
    store: Store,
    *,
    workspace_id: str = "workspace-a",
    version_id: str = "kgv_test",
    signature: str = "evidence-signature-a",
    status: str = "active",
) -> str:
    store._conn.execute(  # noqa: SLF001 - fixture builds the existing KG schema directly
        "INSERT INTO kg_graph_versions("
        "id,workspace_id,status,is_active,base_snapshot_json,stats_json"
        ") VALUES (?,?,?,?,?,?)",
        (
            version_id,
            workspace_id,
            status,
            1,
            json.dumps({"canonical_evidence_signature": signature}),
            "{}",
        ),
    )
    store._conn.commit()  # noqa: SLF001
    return version_id


def _add_graph_node(
    store: Store,
    *,
    version_id: str,
    node_id: str,
    node_type: str,
    node_key: str,
    title: str,
    attrs: dict | None = None,
) -> None:
    store._conn.execute(  # noqa: SLF001
        "INSERT INTO kg_nodes("
        "id,version_id,node_type,node_key,title,summary,attrs_json,confidence,visible,created_at"
        ") VALUES (?,?,?,?,?,?,?,?,?,?)",
        (
            node_id,
            version_id,
            node_type,
            node_key,
            title,
            "",
            json.dumps(attrs or {}),
            1.0,
            1,
            1.0,
        ),
    )


def _add_concept_edge(
    store: Store,
    *,
    version_id: str,
    edge_id: str,
    concept_id: str,
    artifact_node_id: str,
    edge_type: str,
) -> None:
    store._conn.execute(  # noqa: SLF001
        "INSERT INTO kg_edges("
        "id,version_id,edge_type,src_node_id,dst_node_id,"
        "weight,confidence,visible,attrs_json,created_at"
        ") VALUES (?,?,?,?,?,?,?,?,?,?)",
        (
            edge_id,
            version_id,
            edge_type,
            concept_id,
            artifact_node_id,
            1.0,
            1.0,
            1,
            "{}",
            1.0,
        ),
    )


def test_retrieval_settings_default_off_and_preserve_unknown_keys() -> None:
    defaults = default_retrieval_settings()
    assert defaults.chroma_persist_enabled is True
    assert defaults.kg_aware_retrieval_enabled is False

    value = coerce_retrieval_settings(
        {
            "chroma_persist_enabled": False,
            "kg_aware_retrieval_enabled": True,
            "future_retrieval_policy": {"limit": 7},
        }
    )
    serialized = value.to_dict()

    assert serialized == {
        "future_retrieval_policy": {"limit": 7},
        "chroma_persist_enabled": False,
        "kg_aware_retrieval_enabled": True,
    }
    serialized["future_retrieval_policy"]["limit"] = 99
    assert value.to_dict()["future_retrieval_policy"]["limit"] == 7

    with pytest.raises(ValueError, match="kg_aware_retrieval_enabled"):
        coerce_retrieval_settings({"kg_aware_retrieval_enabled": "true"})


def test_run_request_round_trips_frozen_kg_retrieval_snapshot() -> None:
    request = RunRequest(
        query="SQLite FTS5",
        mode=Mode.DEEP,
        kg_aware_retrieval_enabled=True,
        kg_retrieval_snapshot={
            "version_id": "kgv_test",
            "canonical_evidence_signature": "sig-a",
            "wiki_snapshot_id": "kgw_test",
        },
    )

    restored = RunRequest.from_payload(request.to_payload())

    assert restored.kg_aware_retrieval_enabled is True
    assert restored.kg_retrieval_snapshot == request.kg_retrieval_snapshot
    with pytest.raises(BaselineError, match="kg_aware_retrieval_enabled"):
        RunRequest.from_payload(
            {
                "query": "SQLite FTS5",
                "kg_aware_retrieval_enabled": 1,
            }
        )


def test_checkpoint_diagnostics_fail_safe_on_malformed_legacy_fields() -> None:
    restored = KGRetrievalDiagnostics.from_checkpoint_dict(
        {
            "requested": "false",
            "effective": "true",
            "attempted": 1,
            "candidate_count": "not-a-number",
            "hydrated_count": True,
            "snapshot_version_id": 7,
            "wiki_snapshot_id": "  ",
            "reason_codes": "not-a-list",
            "discovered_evidence_ids": ["evidence-a", 7, "evidence-a"],
        },
        requested=True,
    )

    assert restored.requested is True
    assert restored.effective is False
    assert restored.attempted is False
    assert restored.candidate_count == 0
    assert restored.hydrated_count == 0
    assert restored.snapshot_version_id is None
    assert restored.wiki_snapshot_id is None
    assert restored.reason_codes == []
    assert restored.to_checkpoint_dict()["discovered_evidence_ids"] == [
        "evidence-a"
    ]


def test_fast_mode_never_queries_graph_even_when_requested() -> None:
    store = Store(":memory:")
    try:
        retriever = KnowledgeGraphEvidenceRetriever(store)
        result = retriever.retrieve(
            "SQLite FTS5",
            workspace_id="workspace-a",
            requested=True,
            mode="fast",
            snapshot_version_id="missing",
            snapshot_signature="missing",
        )

        assert result.evidence == ()
        assert result.diagnostics.to_dict() == {
            "requested": True,
            "effective": False,
            "attempted": False,
            "applied": False,
            "snapshot_version_id": None,
            "wiki_snapshot_id": None,
            "candidate_count": 0,
            "hydrated_count": 0,
            "adopted_evidence_ids": [],
            "reason_codes": ["mode_not_supported"],
        }
    finally:
        store.close()


def test_graph_snapshot_and_signature_must_match_frozen_run() -> None:
    store = Store(":memory:")
    try:
        version_id = _activate_graph(store)
        retriever = KnowledgeGraphEvidenceRetriever(store)

        result = retriever.retrieve(
            "SQLite FTS5",
            workspace_id="workspace-a",
            requested=True,
            mode="multiple",
            snapshot_version_id=version_id,
            snapshot_signature="different-signature",
        )

        assert result.evidence == ()
        assert result.diagnostics.effective is False
        assert result.diagnostics.attempted is False
        assert result.diagnostics.reason_codes == ["snapshot_signature_mismatch"]

        store._conn.execute(  # noqa: SLF001
            "UPDATE kg_graph_versions SET status='partial' WHERE id=?",
            (version_id,),
        )
        store._conn.commit()  # noqa: SLF001
        partial = retriever.retrieve(
            "SQLite FTS5",
            workspace_id="workspace-a",
            requested=True,
            mode="multiple",
            snapshot_version_id=version_id,
            snapshot_signature="evidence-signature-a",
        )
        assert partial.evidence == ()
        assert partial.diagnostics.reason_codes == ["graph_not_ready"]

        workspace_mismatch = retriever.retrieve(
            "SQLite FTS5",
            workspace_id="workspace-b",
            requested=True,
            mode="multiple",
            snapshot_version_id=version_id,
            snapshot_signature="evidence-signature-a",
        )
        assert workspace_mismatch.evidence == ()
        assert workspace_mismatch.diagnostics.reason_codes == ["workspace_mismatch"]
    finally:
        store.close()


def test_kg_hydration_preserves_canonical_provenance_families() -> None:
    store = Store(":memory:")
    try:
        version_id = _activate_graph(store)
        external_id, _ = store.upsert_document_with_chunks(
            url="https://sqlite.org/fts5.html",
            title="SQLite FTS5 Extension",
            text="SQLite FTS5 provides full text search.",
            chunks=[("SQLite FTS5 provides full text search.", None)],
            metadata={"provenance_root": "https://sqlite.org/fts5.html"},
        )
        created, local_id, _, _ = store.insert_local_document_with_chunks_if_absent(
            url="library://local/" + ("a" * 64),
            title="Local FTS5 Notes",
            text="A local document describes SQLite FTS5 tokenizers.",
            chunks=[("A local document describes SQLite FTS5 tokenizers.", None)],
            content_type="pdf",
            doc_hash="a" * 64,
            metadata={
                "artifact_type": "library_document",
                "artifact_id": "local-doc",
                "origin": "local",
            },
        )
        assert created is True
        report_id = store.add_report(
            title="FTS5 report",
            body_md="SQLite FTS5 is SQLite's full-text search extension.",
            mode="deep",
        )
        memory_id = store.add_memory(
            namespace="workspace",
            scope_key="workspace-a",
            content="SQLite FTS5 supports phrase queries.",
            metadata={"title": "Workspace FTS5 memory"},
        )

        _add_graph_node(
            store,
            version_id=version_id,
            node_id="concept_fts5",
            node_type="concept",
            node_key="concept:sqlite-fts5",
            title="SQLite FTS5",
        )
        artifacts = [
            (
                "source_external",
                "source",
                {
                    "document_id": external_id,
                    "artifact_type": "web_document",
                    "artifact_id": external_id,
                    "origin": "external",
                },
                "concept_supported_by_source",
            ),
            (
                "source_local",
                "source",
                {
                    "document_id": local_id,
                    "artifact_type": "library_document",
                    "artifact_id": local_id,
                    "origin": "local",
                },
                "concept_supported_by_source",
            ),
            (
                "report_local",
                "report",
                {
                    "artifact_type": "library_report",
                    "artifact_id": report_id,
                    "origin": "local",
                },
                "concept_summarized_by_report",
            ),
            (
                "memory_workspace",
                "memory",
                {
                    "artifact_type": "memory",
                    "artifact_id": memory_id,
                    "origin": "memory",
                },
                "concept_remembered_in_memory",
            ),
        ]
        for index, (node_id, node_type, attrs, edge_type) in enumerate(artifacts):
            _add_graph_node(
                store,
                version_id=version_id,
                node_id=node_id,
                node_type=node_type,
                node_key=f"{node_type}:{index}",
                title=f"Artifact {index}",
                attrs=attrs,
            )
            _add_concept_edge(
                store,
                version_id=version_id,
                edge_id=f"edge_{index}",
                concept_id="concept_fts5",
                artifact_node_id=node_id,
                edge_type=edge_type,
            )
        store._conn.commit()  # noqa: SLF001

        result = KnowledgeGraphEvidenceRetriever(store).retrieve(
            "What is SQLite FTS5?",
            workspace_id="workspace-a",
            requested=True,
            mode="deep",
            snapshot_version_id=version_id,
            snapshot_signature="evidence-signature-a",
        )

        assert result.diagnostics.effective is True
        assert result.diagnostics.attempted is True
        assert result.diagnostics.candidate_count == 4
        assert result.diagnostics.hydrated_count == 4
        assert {
            (item.origin, item.artifact_type)
            for item in result.evidence
        } == {
            ("external", "web_document"),
            ("local", "library_document"),
            ("local", "library_report"),
            ("memory", "memory"),
        }
        assert all(
            "concept_fts5" not in ref and not ref.startswith("edge_")
            for item in result.evidence
            for ref in item.provenance_refs
        )

        result.diagnostics.mark_adopted([result.evidence[0]])
        diagnostics = result.diagnostics.to_dict()
        assert diagnostics["applied"] is True
        assert len(diagnostics["adopted_evidence_ids"]) == 1
    finally:
        store.close()


def test_fresh_wiki_topic_can_expand_to_canonical_graph_artifact_at_query_time() -> None:
    store = Store(":memory:")
    try:
        version_id = _activate_graph(store)
        document_id, _ = store.upsert_document_with_chunks(
            url="https://sqlite.org/fts5.html",
            title="SQLite FTS5 Extension",
            text="SQLite FTS5 provides full text search.",
            chunks=[("SQLite FTS5 provides full text search.", None)],
            metadata={"provenance_root": "https://sqlite.org/fts5.html"},
        )
        _add_graph_node(
            store,
            version_id=version_id,
            node_id="source_fts5",
            node_type="source",
            node_key="source:sqlite-fts5",
            title="SQLite FTS5",
            attrs={
                "document_id": document_id,
                "artifact_type": "web_document",
                "artifact_id": document_id,
                "origin": "external",
            },
        )
        store._conn.execute(  # noqa: SLF001 - fixture builds existing Wiki schema
            "INSERT INTO kg_wiki_snapshots("
            "id,workspace_id,graph_version_id,status,is_active,stale,activated_at"
            ") VALUES (?,?,?,?,?,?,?)",
            ("kgwiki-fresh", "workspace-a", version_id, "succeeded", 1, 0, 2.0),
        )
        store._conn.execute(  # noqa: SLF001
            "INSERT INTO kg_wiki_topics("
            "id,snapshot_id,topic_key,title,attrs_json,created_at"
            ") VALUES (?,?,?,?,?,?)",
            (
                "topic-fts5",
                "kgwiki-fresh",
                "wiki:fts5",
                "SQLite FTS5",
                json.dumps(
                    {
                        "concept_key": "sqlite-fts5",
                        "evidence_refs": [
                            {"type": "source", "id": "source:sqlite-fts5"}
                        ],
                    }
                ),
                2.0,
            ),
        )
        store._conn.commit()  # noqa: SLF001

        result = KnowledgeGraphEvidenceRetriever(store).retrieve(
            "What is SQLite FTS5?",
            workspace_id="workspace-a",
            requested=True,
            mode="multiple",
            snapshot_version_id=version_id,
            snapshot_signature="evidence-signature-a",
        )

        assert result.diagnostics.wiki_snapshot_id == "kgwiki-fresh"
        assert result.diagnostics.candidate_count == 1
        assert result.diagnostics.hydrated_count == 1
        assert result.evidence[0].origin == "external"
        assert result.evidence[0].artifact_id == document_id
        assert result.evidence[0].channels == ["knowledge_graph"]
    finally:
        store.close()


def test_stale_or_graph_mismatched_wiki_snapshot_is_not_used_for_query_expansion() -> None:
    store = Store(":memory:")
    try:
        version_id = _activate_graph(store)
        store._conn.execute(  # noqa: SLF001 - fixture builds existing Wiki schema
            "INSERT INTO kg_wiki_snapshots("
            "id,workspace_id,graph_version_id,status,is_active,stale,activated_at"
            ") VALUES (?,?,?,?,?,?,?)",
            ("kgwiki-stale", "workspace-a", version_id, "succeeded", 1, 1, 2.0),
        )
        store._conn.execute(  # noqa: SLF001
            "INSERT INTO kg_wiki_topics("
            "id,snapshot_id,topic_key,title,attrs_json,created_at"
            ") VALUES (?,?,?,?,?,?)",
            (
                "topic-stale",
                "kgwiki-stale",
                "wiki:fts5",
                "SQLite FTS5",
                json.dumps(
                    {
                        "concept_key": "sqlite-fts5",
                        "evidence_refs": [
                            {"type": "source", "id": "source:missing"}
                        ],
                    }
                ),
                2.0,
            ),
        )
        store._conn.commit()  # noqa: SLF001

        result = KnowledgeGraphEvidenceRetriever(store).retrieve(
            "What is SQLite FTS5?",
            workspace_id="workspace-a",
            requested=True,
            mode="deep",
            snapshot_version_id=version_id,
            snapshot_signature="evidence-signature-a",
        )

        assert result.diagnostics.wiki_snapshot_id is None
        assert result.diagnostics.candidate_count == 0
        assert result.diagnostics.hydrated_count == 0
        assert result.diagnostics.reason_codes == ["no_concept_match"]
    finally:
        store.close()


def test_kg_query_caps_candidates_and_deduplicates_artifacts() -> None:
    store = Store(":memory:")
    try:
        version_id = _activate_graph(store)
        _add_graph_node(
            store,
            version_id=version_id,
            node_id="concept_budget",
            node_type="concept",
            node_key="concept:budget",
            title="Budget",
        )
        store._conn.commit()  # noqa: SLF001
        for index in range(14):
            document_id, _ = store.upsert_document_with_chunks(
                url=f"https://example.com/{index}",
                title=f"Budget source {index}",
                text=f"Budget evidence {index}",
                chunks=[(f"Budget evidence {index}", None)],
            )
            node_id = f"source_{index}"
            _add_graph_node(
                store,
                version_id=version_id,
                node_id=node_id,
                node_type="source",
                node_key=f"source:{index}",
                title=f"Source {index}",
                attrs={
                    "document_id": document_id,
                    "artifact_type": "web_document",
                    "artifact_id": document_id,
                },
            )
            _add_concept_edge(
                store,
                version_id=version_id,
                edge_id=f"edge_{index}",
                concept_id="concept_budget",
                artifact_node_id=node_id,
                edge_type="concept_supported_by_source",
            )
            store._conn.commit()  # noqa: SLF001

        result = KnowledgeGraphEvidenceRetriever(store).retrieve(
            "budget",
            workspace_id="workspace-a",
            requested=True,
            mode="council",
            snapshot_version_id=version_id,
            snapshot_signature="evidence-signature-a",
        )

        assert result.diagnostics.candidate_count == 12
        assert result.diagnostics.hydrated_count == 12
        assert len(result.evidence) == 12
    finally:
        store.close()


def test_diagnostics_merge_never_claims_adoption_before_final_snapshot() -> None:
    first = KGRetrievalDiagnostics(
        requested=True,
        effective=True,
        attempted=True,
        snapshot_version_id="kgv_test",
        candidate_count=2,
        hydrated_count=1,
        reason_codes=["no_candidates"],
    )
    second = KGRetrievalDiagnostics(
        requested=True,
        effective=True,
        attempted=True,
        snapshot_version_id="kgv_test",
        candidate_count=3,
        hydrated_count=2,
        reason_codes=["no_hydrated_candidates"],
    )

    first.merge(second)

    assert first.applied is False
    assert first.candidate_count == 5
    assert first.hydrated_count == 3
    assert first.reason_codes == ["no_candidates", "no_hydrated_candidates"]


def test_builder_persists_signature_and_bounded_view_preserves_non_concepts() -> None:
    service = BaselineService(provider=ProviderSettings(), db_path=":memory:")
    try:
        result = service.knowledge_graph_rebuild_now({"workspace_id": "default"})
        active = KnowledgeGraphStore(service.store).get_active_version("default")

        assert result["status"] == "succeeded"
        assert active is not None
        assert active["status"] == "active"
        assert len(
            active["base_snapshot_json"]["canonical_evidence_signature"]
        ) == 64

        ops = KnowledgeGraphStore(service.store)
        graph = ops.get_bounded_graph(workspace_id="default", limit=300)
        assert sum(node["type"] == "concept" for node in graph["nodes"]) <= 64
        assert len(graph["nodes"]) <= 300
    finally:
        service.store.close()


def test_service_freezes_fresh_and_stale_kg_state_for_new_runs() -> None:
    service = BaselineService(provider=ProviderSettings(), db_path=":memory:")
    service._llm_ready = True  # noqa: SLF001 - no provider call is made
    service._run_safe = lambda request, run_id: None  # type: ignore[method-assign]  # noqa: SLF001
    try:
        service.knowledge_graph_rebuild_now({"workspace_id": "default"})
        service.update_retrieval_settings(
            {"kg_aware_retrieval_enabled": True}
        )

        fresh = service.create_run(
            {"query": "fresh graph", "mode": "multiple"},
            background=False,
        )
        fresh_run = service.store.get_run(fresh["run_id"])
        assert fresh_run is not None
        assert fresh_run["metadata"]["kg_retrieval_snapshot"]["effective"] is True

        service.store.add_memory(
            namespace="workspace",
            scope_key="default",
            content="New canonical evidence after the active graph build.",
        )
        stale = service.create_run(
            {"query": "stale graph", "mode": "deep"},
            background=False,
        )
        stale_run = service.store.get_run(stale["run_id"])
        assert stale_run is not None
        assert stale_run["metadata"]["kg_retrieval_snapshot"]["effective"] is False
        assert stale_run["metadata"]["kg_retrieval_snapshot"]["reason_codes"] == [
            "snapshot_signature_mismatch"
        ]

        fast = service.create_run(
            {"query": "fast ignores graph", "mode": "fast"},
            background=False,
        )
        fast_run = service.store.get_run(fast["run_id"])
        assert fast_run is not None
        assert fast_run["metadata"]["kg_retrieval_snapshot"]["reason_codes"] == [
            "mode_not_supported"
        ]
    finally:
        service.store.close()


def test_orchestrator_keeps_three_lanes_and_applies_kg_only_after_final_adoption(
    fake_llm,
) -> None:
    store = Store(":memory:")
    try:
        version_id = _activate_graph(store)
        note_id = store.add_memory(
            namespace="note",
            scope_key="workspace-a",
            content=(
                "GraphBridgeMarker evidence explains deterministic knowledge graph "
                "candidate discovery."
            ),
            metadata={"title": "Graph bridge note"},
        )
        _add_graph_node(
            store,
            version_id=version_id,
            node_id="concept_graph_bridge",
            node_type="concept",
            node_key="concept:graphbridgemarker",
            title="GraphBridgeMarker",
        )
        _add_graph_node(
            store,
            version_id=version_id,
            node_id="note_graph_bridge",
            node_type="note",
            node_key=f"note:{note_id}",
            title="Graph bridge note",
            attrs={
                "artifact_type": "library_note",
                "artifact_id": note_id,
                "origin": "local",
            },
        )
        _add_concept_edge(
            store,
            version_id=version_id,
            edge_id="edge_graph_bridge_note",
            concept_id="concept_graph_bridge",
            artifact_node_id="note_graph_bridge",
            edge_type="concept_mentioned_in_note",
        )
        store._conn.commit()  # noqa: SLF001

        orchestrator = Orchestrator(
            store=store,
            llm=fake_llm,
            retriever=Retriever(store, embedder=fake_llm),
            indexer=Indexer(store, embedder=fake_llm),
            search=FakeSearch(
                {
                    "https://example.test/graph-bridge": (
                        "GraphBridgeMarker external evidence independently describes "
                        "knowledge graph candidate discovery."
                    )
                }
            ),
            memory=MemoryEngine(store, embedder=fake_llm),
            enforce_grounding_gate=True,
        )
        outcome = orchestrator.run(
            RunRequest(
                query="Explain GraphBridgeMarker candidate discovery",
                mode=Mode.MULTIPLE,
                workspace_key="workspace-a",
                memory_policy="off",
                kg_aware_retrieval_enabled=True,
                kg_retrieval_snapshot={
                    "requested": True,
                    "effective": True,
                    "workspace_id": "workspace-a",
                    "version_id": version_id,
                    "canonical_evidence_signature": "evidence-signature-a",
                    "reason_codes": [],
                },
            )
        )
        serialized = outcome.to_dict()
        assert {item["lane"] for item in serialized["retrieval_outcomes"]} == {
            "local",
            "memory",
            "external",
        }
        assert all(item["lane"] != "knowledge_graph" for item in serialized["retrieval_outcomes"])
        assert serialized["kg_retrieval"]["requested"] is True
        assert serialized["kg_retrieval"]["attempted"] is True
        assert serialized["kg_retrieval"]["applied"] is True
        assert serialized["kg_retrieval"]["adopted_evidence_ids"] == [
            f"local:library_note:{note_id}"
        ]
        assert any(
            citation["origin"] == "local"
            and citation["artifact_type"] == "library_note"
            and citation["artifact_id"] == note_id
            for citation in serialized["answer"]["citations"]
        )
    finally:
        store.close()
