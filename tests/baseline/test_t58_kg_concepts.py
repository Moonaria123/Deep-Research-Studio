from __future__ import annotations

from collections import Counter

from deepresearch_core.baseline.knowledge_graph.builder import KnowledgeGraphBuilder
from deepresearch_core.baseline.knowledge_graph.concept_extractor import (
    ConceptArtifact,
    extract_concepts,
)
from deepresearch_core.baseline.knowledge_graph.projector import KnowledgeGraphProjector
from deepresearch_core.baseline.knowledge_graph.wiki_models import WikiSnapshotDraft, WikiTopicDraft
from deepresearch_core.baseline.knowledge_graph.wiki_store import KnowledgeGraphWikiStore
from deepresearch_core.baseline.store import Store


def _citation(
    *,
    citation_id: str,
    origin: str,
    artifact_type: str,
    artifact_id: str,
    title: str,
    document_id: str = "",
    url: str = "",
    source_status: str = "active",
) -> dict:
    return {
        "id": citation_id,
        "document_id": document_id or f"{artifact_type}:{artifact_id}",
        "url": url,
        "title": title,
        "snippet": title,
        "score": 1.0,
        "origin": origin,
        "artifact_type": artifact_type,
        "artifact_id": artifact_id,
        "source_status": source_status,
        "provenance_refs": [f"{origin}:{artifact_type}:{artifact_id}"],
    }


def _seed_workspace(store: Store, *, workspace: str = "default") -> dict[str, str]:
    session_id = store.create_session(title="Concept session", workspace=workspace)
    run_id = store.create_run(session_id=session_id, mode="deep", query="Concepts")
    document_id = store.upsert_document(
        url=f"https://example.com/{workspace}/safety",
        title="Safety evidence",
        text="Evidence about safety taxonomy.",
    )
    note_id = store.add_memory(
        namespace="note",
        scope_key=workspace,
        content="# Operational Controls\nBounded note content.",
    )
    memory_id = store.add_memory(
        namespace="workspace",
        scope_key=workspace,
        content='{"topics":["Operational Controls","Risk Taxonomy"]}',
    )
    if workspace == "default":
        assert store.archive_memory(memory_id) is True
    report_id = store.add_report(
        title="Risk Taxonomy",
        body_md="# Risk Taxonomy\nBounded report.",
        mode="deep",
        session_id=session_id,
        run_id=run_id,
        metadata={"topics": ["Risk Taxonomy"]},
    )
    citations = [
        _citation(
            citation_id="source-citation",
            origin="external",
            artifact_type="web_document",
            artifact_id=document_id,
            document_id=document_id,
            title="Safety evidence",
            url=f"https://example.com/{workspace}/safety",
        ),
        _citation(
            citation_id="note-citation",
            origin="local",
            artifact_type="library_note",
            artifact_id=note_id,
            title="Operational Controls",
        ),
        _citation(
            citation_id="report-citation",
            origin="local",
            artifact_type="library_report",
            artifact_id=report_id,
            title="Risk Taxonomy",
        ),
        _citation(
            citation_id="memory-citation",
            origin="memory",
            artifact_type="memory",
            artifact_id=memory_id,
            title="Operational Controls",
            source_status="archived" if workspace == "default" else "active",
        ),
    ]
    store.update_run(
        run_id,
        status="completed",
        result={
            "run_id": run_id,
            "mode": "deep",
            "status": "completed",
            "answer": {
                "text": "Grounded answer.",
                "claim_map": {},
                "coverage": 1.0,
                "citations": citations,
            },
        },
    )
    return {
        "session_id": session_id,
        "run_id": run_id,
        "document_id": document_id,
        "note_id": note_id,
        "memory_id": memory_id,
        "report_id": report_id,
    }


def _activate_wiki(
    store: Store,
    *,
    graph_version_id: str,
    topics: tuple[WikiTopicDraft, ...],
) -> str:
    wiki = KnowledgeGraphWikiStore(store)
    snapshot_id = wiki.create_staging_snapshot(
        workspace_id="default",
        graph_version_id=graph_version_id,
    )
    draft = WikiSnapshotDraft(
        workspace_id="default",
        graph_version_id=graph_version_id,
        topics=topics,
    )
    stats = wiki.replace_snapshot_contents(snapshot_id, draft)
    wiki.activate_snapshot(snapshot_id, stats=stats)
    return snapshot_id


def _topic(
    *,
    key: str,
    title: str,
    concept_key: str,
    evidence_refs: list[dict],
    confidence: str = "high",
) -> WikiTopicDraft:
    return WikiTopicDraft(
        topic_key=key,
        title=title,
        summary=f"{title} is grounded in canonical evidence.",
        depth=1,
        attrs={
            "kind": "topic",
            "concept_key": concept_key,
            "confidence": confidence,
            "evidence_refs": evidence_refs,
        },
    )


def test_projector_projects_wiki_concepts_and_accumulates_all_provenance() -> None:
    store = Store(":memory:")
    ids = _seed_workspace(store)
    first_build = KnowledgeGraphBuilder(store).rebuild(workspace_id="default")
    source_key = f"source:{ids['document_id']}"
    wiki_snapshot_id = _activate_wiki(
        store,
        graph_version_id=first_build["version_id"],
        topics=(
            _topic(
                key="wiki:risk-1",
                title="Risk Taxonomy",
                concept_key="risk-taxonomy",
                evidence_refs=[
                    {"type": "source", "id": source_key, "excerpt": "Primary evidence."},
                    {"type": "report", "id": f"report:{ids['report_id']}", "excerpt": "Report evidence."},
                ],
            ),
            _topic(
                key="wiki:risk-2",
                title="Risk Taxonomy",
                concept_key="risk-taxonomy",
                evidence_refs=[
                    {"type": "source", "id": source_key, "excerpt": "Independent topic evidence."},
                ],
            ),
            _topic(
                key="wiki:message-only",
                title="Transcript Only",
                concept_key="transcript-only",
                evidence_refs=[{"type": "message", "id": "msg-only", "excerpt": "Not canonical."}],
            ),
        ),
    )

    graph = KnowledgeGraphProjector(store).project(workspace_id="default")
    node_by_key = {node.node_key: node for node in graph.nodes}
    edge_by_key = {
        (edge.edge_type, edge.src_key, edge.dst_key): edge
        for edge in graph.edges
    }
    concept_key = "concept:risk-taxonomy"

    assert concept_key in node_by_key
    assert "concept:transcript-only" not in node_by_key
    assert (
        "concept_supported_by_source",
        concept_key,
        source_key,
    ) in edge_by_key
    assert (
        "concept_summarized_by_report",
        concept_key,
        f"report:{ids['report_id']}",
    ) in edge_by_key
    assert node_by_key[concept_key].attrs["source_count"] >= 2

    node_provenance = [
        item
        for item in graph.provenance
        if item.owner_kind == "node"
        and item.owner_key == concept_key
        and item.source_kind == "wiki_topic"
    ]
    source_edge_owner = edge_by_key[
        ("concept_supported_by_source", concept_key, source_key)
    ].owner_key
    edge_provenance = [
        item
        for item in graph.provenance
        if item.owner_kind == "edge"
        and item.owner_key == source_edge_owner
        and item.source_kind == "wiki_topic"
    ]
    assert {item.source_id for item in node_provenance} == {
        "wiki:risk-1",
        "wiki:risk-2",
    }
    assert {item.source_id for item in edge_provenance} == {
        "wiki:risk-1",
        "wiki:risk-2",
    }
    assert graph.base_snapshot["schema"] == "canonical-evidence-signature/v1"
    assert graph.base_snapshot["wiki_seed_snapshot_id"] == wiki_snapshot_id
    assert graph.base_snapshot["wiki_seed_count"] == 2
    assert graph.base_snapshot["input_count"] >= 4
    assert len(graph.base_snapshot["canonical_evidence_signature"]) == 64
    repeated = KnowledgeGraphProjector(store).project(workspace_id="default")
    assert repeated.base_snapshot == graph.base_snapshot
    next_wiki_snapshot_id = _activate_wiki(
        store,
        graph_version_id=first_build["version_id"],
        topics=(
            _topic(
                key="wiki:operational",
                title="Operational Controls",
                concept_key="operational-controls",
                evidence_refs=[
                    {
                        "type": "note",
                        "id": f"note:{ids['note_id']}",
                        "excerpt": "Canonical note.",
                    }
                ],
            ),
        ),
    )
    after_wiki_refresh = KnowledgeGraphProjector(store).project(
        workspace_id="default"
    )
    assert (
        after_wiki_refresh.base_snapshot["canonical_evidence_signature"]
        == graph.base_snapshot["canonical_evidence_signature"]
    )
    assert (
        after_wiki_refresh.base_snapshot["wiki_seed_snapshot_id"]
        == next_wiki_snapshot_id
    )
    store.close()


def test_projector_keeps_archived_memory_but_excludes_user_workspace_and_tombstones() -> None:
    store = Store(":memory:")
    ids = _seed_workspace(store)
    other = _seed_workspace(store, workspace="other")
    user_memory_id = store.add_memory(
        namespace="user",
        scope_key="default",
        content='{"topics":["Private Profile Topic"]}',
    )
    deleted_document_id = store.upsert_document(
        url="https://example.com/deleted",
        title="Deleted Topic",
        text="Deleted evidence.",
    )
    store.add_library_provenance(
        artifact_type="_document_delete_tombstone",
        artifact_id=f"deleted:{deleted_document_id}",
        session_id=ids["session_id"],
        run_id=ids["run_id"],
        metadata={"deleted_document_id": deleted_document_id},
    )
    first_build = KnowledgeGraphBuilder(store).rebuild(workspace_id="default")
    _activate_wiki(
        store,
        graph_version_id=first_build["version_id"],
        topics=(
            _topic(
                key="wiki:memory",
                title="Operational Controls",
                concept_key="operational-controls",
                evidence_refs=[
                    {"type": "memory", "id": f"memory:{ids['memory_id']}", "excerpt": "Archived evidence."},
                ],
            ),
            _topic(
                key="wiki:user",
                title="Private Profile Topic",
                concept_key="private-profile-topic",
                evidence_refs=[
                    {"type": "memory", "id": f"memory:{user_memory_id}", "excerpt": "Unscoped user data."},
                ],
            ),
            _topic(
                key="wiki:other",
                title="Other Workspace",
                concept_key="other-workspace",
                evidence_refs=[
                    {"type": "report", "id": f"report:{other['report_id']}", "excerpt": "Other workspace."},
                ],
            ),
            _topic(
                key="wiki:deleted",
                title="Deleted Topic",
                concept_key="deleted-topic",
                evidence_refs=[
                    {"type": "source", "id": f"source:{deleted_document_id}", "excerpt": "Deleted."},
                ],
            ),
        ),
    )

    graph = KnowledgeGraphProjector(store).project(workspace_id="default")
    node_by_key = {node.node_key: node for node in graph.nodes}
    edges = {(edge.edge_type, edge.src_key, edge.dst_key) for edge in graph.edges}

    assert node_by_key[f"memory:{ids['memory_id']}"].attrs["source_status"] == "archived"
    assert (
        "concept_remembered_in_memory",
        "concept:operational-controls",
        f"memory:{ids['memory_id']}",
    ) in edges
    assert "concept:private-profile-topic" not in node_by_key
    assert "concept:other-workspace" not in node_by_key
    assert "concept:deleted-topic" not in node_by_key
    store.close()


def test_structured_and_title_heuristics_are_bounded_and_secret_safe() -> None:
    artifacts: list[ConceptArtifact] = []
    for artifact_index in range(40):
        topics = [f"Topic {artifact_index}-{topic_index}" for topic_index in range(10)]
        artifacts.append(
            ConceptArtifact(
                node_key=f"report:report-{artifact_index}",
                node_type="report",
                title=f"Report Topic {artifact_index}",
                metadata={"topics": topics},
                content="# Heading Topic\nSafe body.",
            )
        )
    artifacts.append(
        ConceptArtifact(
            node_key="memory:archived",
            node_type="memory",
            title="Memory archived",
            metadata={"source_status": "archived"},
            content='{"topics":["Safe Topic","sk-testsecret-should-not-leak"]}',
        )
    )

    result = extract_concepts(
        artifacts=artifacts,
        wiki_topics=[],
        wiki_snapshot_id="",
    )
    counts = Counter(edge.dst_key for edge in result.edges)
    serialized = " ".join(
        [
            *(node.title + node.summary + str(node.attrs) for node in result.nodes),
            *(str(edge.attrs) for edge in result.edges),
            *(str(item.evidence) for item in result.provenance),
        ]
    )

    assert len(result.nodes) <= 64
    assert len(result.edges) <= 256
    assert counts and max(counts.values()) <= 8
    assert all(edge.src_key.startswith("concept:") for edge in result.edges)
    assert "sk-testsecret" not in serialized
