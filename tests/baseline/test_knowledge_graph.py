"""T14 research knowledge graph core: schema, projection, builder, and API."""

from __future__ import annotations

import time

import pytest

from deepresearch_core.baseline import BaselineService, create_app
from deepresearch_core.baseline.errors import BaselineError, ErrorCode
from deepresearch_core.baseline.knowledge_graph.builder import KnowledgeGraphBuilder
from deepresearch_core.baseline.knowledge_graph.jobs import RQKnowledgeGraphJobBackend
from deepresearch_core.baseline.knowledge_graph.projector import KnowledgeGraphProjector
from deepresearch_core.baseline.knowledge_graph.service import KnowledgeGraphService
from deepresearch_core.baseline.knowledge_graph.store_ops import KnowledgeGraphStore
from deepresearch_core.baseline.knowledge_graph.wiki_generator import WikiGenerator
from deepresearch_core.baseline.store import Store
from deepresearch_core.settings import ProviderSettings


def _seed_graph_fixture(store: Store) -> dict[str, str]:
    sid = store.create_session(title="AI safety", workspace="default")
    rid = store.create_run(session_id=sid, mode="deep", query="AI safety")
    doc_id = store.upsert_document(
        url="https://example.com/safety",
        title="Safety source",
        text="Evidence body is not copied to KG summaries.",
    )
    citation_id = store.add_citation(
        run_id=rid,
        document_id=doc_id,
        url="https://example.com/safety",
        title="Safety source",
        snippet="Evidence snippet",
        score=0.9,
    )
    report_id = store.add_report(
        title="AI safety report",
        body_md="# Report\nFull body should stay out of KG summaries.",
        mode="deep",
        session_id=sid,
        run_id=rid,
    )
    note_id = store.add_memory(namespace="note", scope_key="default", content="private note body")
    session_memory_id = store.add_memory(namespace="session", scope_key=sid, content="session memory body")
    workspace_memory_id = store.add_memory(namespace="workspace", scope_key="default", content="workspace memory")
    orphan_note_id = store.add_memory(namespace="note", scope_key="default", content="no provenance")
    store.add_library_provenance(
        artifact_type="note",
        artifact_id=note_id,
        session_id=sid,
        run_id=rid,
        document_id=doc_id,
        url="https://example.com/safety",
    )
    store.add_library_provenance(
        artifact_type="memory",
        artifact_id=workspace_memory_id,
        document_id=doc_id,
        url="https://example.com/safety",
    )
    snapshot_id = store.add_page_snapshot(
        document_id=doc_id,
        url="https://example.com/safety",
        content_hash="hash",
        storage_path="snap.txt",
        run_id=rid,
    )
    return {
        "session_id": sid,
        "run_id": rid,
        "doc_id": doc_id,
        "citation_id": citation_id,
        "report_id": report_id,
        "note_id": note_id,
        "session_memory_id": session_memory_id,
        "workspace_memory_id": workspace_memory_id,
        "orphan_note_id": orphan_note_id,
        "snapshot_id": snapshot_id,
    }


def _edge_set(projector: KnowledgeGraphProjector) -> set[tuple[str, str, str]]:
    graph = projector.project(workspace_id="default")
    return {(edge.edge_type, edge.src_key, edge.dst_key) for edge in graph.edges}


def test_kg_schema_migrates_idempotently_and_active_version_is_unique() -> None:
    store = Store(":memory:")
    store._migrate()  # noqa: SLF001 - explicit idempotency check
    ops = KnowledgeGraphStore(store)
    first = ops.create_staging_version(workspace_id="default")
    second = ops.create_staging_version(workspace_id="default")

    ops.activate_version(first)
    ops.activate_version(second)

    active_rows = store._conn.execute(  # noqa: SLF001
        "SELECT id FROM kg_graph_versions WHERE workspace_id='default' AND is_active=1"
    ).fetchall()
    assert [row["id"] for row in active_rows] == [second]
    store.close()


def test_projector_replays_p0_p1_edges_without_guessing_memory_session_links() -> None:
    store = Store(":memory:")
    ids = _seed_graph_fixture(store)
    projector = KnowledgeGraphProjector(store)

    first_graph = projector.project(workspace_id="default")
    second_graph = projector.project(workspace_id="default")
    assert [node.node_key for node in first_graph.nodes] == [node.node_key for node in second_graph.nodes]
    assert _edge_set(projector) == _edge_set(projector)

    edges = {(edge.edge_type, edge.src_key, edge.dst_key) for edge in first_graph.edges}
    assert ("session_has_report", f"session:{ids['session_id']}", f"report:{ids['report_id']}") in edges
    assert ("session_used_source", f"session:{ids['session_id']}", f"source:{ids['doc_id']}") in edges
    assert ("report_cites_source", f"report:{ids['report_id']}", f"source:{ids['doc_id']}") in edges
    assert ("session_has_note", f"session:{ids['session_id']}", f"note:{ids['note_id']}") in edges
    assert ("note_cites_source", f"note:{ids['note_id']}", f"source:{ids['doc_id']}") in edges
    assert ("memory_derived_from_source", f"memory:{ids['workspace_memory_id']}", f"source:{ids['doc_id']}") in edges
    assert (
        "session_wrote_memory",
        f"session:{ids['session_id']}",
        f"memory:{ids['workspace_memory_id']}",
    ) not in edges
    assert all(
        edge.dst_key != f"note:{ids['orphan_note_id']}" or edge.edge_type != "note_cites_source"
        for edge in first_graph.edges
    )
    summaries = " ".join(node.summary for node in first_graph.nodes)
    assert "Evidence body is not copied to KG summaries." not in summaries
    assert "Full body should stay out of KG summaries." not in summaries
    memory_nodes = [node for node in first_graph.nodes if node.node_type == "memory"]
    note_nodes = [node for node in first_graph.nodes if node.node_type == "note"]
    assert memory_nodes and all(node.summary for node in memory_nodes)
    assert note_nodes and all(node.summary for node in note_nodes)
    store.close()


def test_projector_materializes_t46_typed_citation_nodes_and_cross_session_edges() -> None:
    store = Store(":memory:")
    first_session = store.create_session(title="First", workspace="default")
    second_session = store.create_session(title="Second", workspace="default")
    first_run = store.create_run(session_id=first_session, mode="deep", query="First")
    second_run = store.create_run(session_id=second_session, mode="deep", query="Second")
    document_id = store.upsert_document(
        url="https://example.com/shared",
        title="Shared public source",
        text="Shared evidence",
    )
    note_id = store.add_memory(namespace="note", scope_key="default", content="Shared local note")
    memory_id = store.add_memory(namespace="workspace", scope_key="default", content="Shared archived memory")
    assert store.archive_memory(memory_id) is True
    shared_report_id = store.add_report(
        title="Shared library report",
        body_md="# Shared",
        mode="deep",
        session_id=first_session,
        run_id=first_run,
    )
    second_report_id = store.add_report(
        title="Second output",
        body_md="# Second",
        mode="deep",
        session_id=second_session,
        run_id=second_run,
    )

    def result_for(run_id: str) -> dict:
        return {
            "run_id": run_id,
            "mode": "deep",
            "status": "completed",
            "answer": {
                "text": "Grounded answer.",
                "claim_map": {},
                "coverage": 1.0,
                "citations": [
                    {
                        "id": "c1",
                        "document_id": document_id,
                        "url": "https://example.com/shared",
                        "title": "Shared public source",
                        "snippet": "Shared evidence",
                        "score": 1.0,
                        "origin": "external",
                        "artifact_type": "web_document",
                        "artifact_id": document_id,
                        "provenance_refs": [f"external:web_document:{document_id}"],
                    },
                    {
                        "id": "c2",
                        "document_id": f"note:{note_id}",
                        "url": "",
                        "title": "Shared local note",
                        "snippet": "Shared local note",
                        "score": 1.0,
                        "origin": "local",
                        "artifact_type": "library_note",
                        "artifact_id": note_id,
                        "source_status": "active",
                    },
                    {
                        "id": "c3",
                        "document_id": f"report:{shared_report_id}",
                        "url": "",
                        "title": "Shared library report",
                        "snippet": "Shared report",
                        "score": 1.0,
                        "origin": "local",
                        "artifact_type": "library_report",
                        "artifact_id": shared_report_id,
                        "source_status": "active",
                    },
                    {
                        "id": "c4",
                        "document_id": f"memory:{memory_id}",
                        "url": "",
                        "title": "Shared archived memory",
                        "snippet": "Shared archived memory",
                        "score": 1.0,
                        "origin": "memory",
                        "artifact_type": "memory",
                        "artifact_id": memory_id,
                        "source_status": "archived",
                    },
                    {
                        "id": "c5",
                        "document_id": "memory:missing",
                        "url": "",
                        "title": "Missing memory",
                        "snippet": "must not resurrect",
                        "score": 1.0,
                        "origin": "memory",
                        "artifact_type": "memory",
                        "artifact_id": "missing",
                    },
                ],
            },
        }

    store.update_run(first_run, status="completed", result=result_for(first_run))
    store.update_run(second_run, status="completed", result=result_for(second_run))

    graph = KnowledgeGraphProjector(store).project(workspace_id="default")
    node_by_key = {node.node_key: node for node in graph.nodes}
    edges = {(edge.edge_type, edge.src_key, edge.dst_key) for edge in graph.edges}

    assert f"source:{document_id}" in node_by_key
    assert f"note:{note_id}" in node_by_key
    assert f"report:{shared_report_id}" in node_by_key
    assert f"memory:{memory_id}" in node_by_key
    assert "memory:missing" not in node_by_key
    assert node_by_key[f"memory:{memory_id}"].attrs["source_status"] == "archived"
    assert node_by_key[f"memory:{memory_id}"].attrs["jump_uri"] == f"/memory/{memory_id}"
    assert node_by_key[f"note:{note_id}"].attrs["evidence_key"] == f"local:library_note:{note_id}"

    for session_id in (first_session, second_session):
        assert ("session_used_source", f"session:{session_id}", f"source:{document_id}") in edges
        assert ("session_used_note", f"session:{session_id}", f"note:{note_id}") in edges
        assert ("session_used_report", f"session:{session_id}", f"report:{shared_report_id}") in edges
        assert ("session_used_memory", f"session:{session_id}", f"memory:{memory_id}") in edges
    assert ("report_cites_note", f"report:{second_report_id}", f"note:{note_id}") in edges
    assert ("report_cites_report", f"report:{second_report_id}", f"report:{shared_report_id}") in edges
    assert ("report_cites_memory", f"report:{second_report_id}", f"memory:{memory_id}") in edges

    build = KnowledgeGraphBuilder(store).rebuild(workspace_id="default")
    catalog = WikiGenerator(store=store, memory_llm=lambda: None)._session_evidence_catalog(build["version_id"])
    second_catalog = {(item["type"], item["id"]) for item in catalog[second_session]}
    assert ("source", f"source:{document_id}") in second_catalog
    assert ("note", f"note:{note_id}") in second_catalog
    assert ("report", f"report:{shared_report_id}") in second_catalog
    assert ("memory", f"memory:{memory_id}") in second_catalog
    store.close()


def test_projector_excludes_artifacts_outside_requested_workspace() -> None:
    store = Store(":memory:")
    ids = _seed_graph_fixture(store)
    other_sid = store.create_session(title="Other workspace", workspace="other")
    other_rid = store.create_run(session_id=other_sid, mode="deep", query="Other")
    other_doc_id = store.upsert_document(
        url="https://example.com/other",
        title="Other source",
        text="Other workspace evidence.",
    )
    store.add_citation(
        run_id=other_rid,
        document_id=other_doc_id,
        url="https://example.com/other",
        title="Other source",
        snippet="Other snippet",
        score=0.9,
    )
    other_report_id = store.add_report(
        title="Other report",
        body_md="# Other",
        mode="deep",
        session_id=other_sid,
        run_id=other_rid,
    )
    other_note_id = store.add_memory(namespace="note", scope_key="other", content="other note")
    other_memory_id = store.add_memory(namespace="workspace", scope_key="other", content="other memory")
    user_memory_id = store.add_memory(namespace="user", scope_key="default", content="user memory without provenance")
    store.add_library_provenance(
        artifact_type="note",
        artifact_id=other_note_id,
        session_id=other_sid,
        run_id=other_rid,
        document_id=other_doc_id,
        url="https://example.com/other",
    )

    graph = KnowledgeGraphProjector(store).project(workspace_id="default")
    keys = {node.node_key for node in graph.nodes}

    assert f"session:{ids['session_id']}" in keys
    assert f"source:{ids['doc_id']}" in keys
    assert f"session:{other_sid}" not in keys
    assert f"source:{other_doc_id}" not in keys
    assert f"report:{other_report_id}" not in keys
    assert f"note:{other_note_id}" not in keys
    assert f"memory:{other_memory_id}" not in keys
    assert f"memory:{user_memory_id}" not in keys
    store.close()


def test_projector_adds_safe_recent_assistant_summary_without_raw_secrets() -> None:
    store = Store(":memory:")
    sid = store.create_session(title="Assistant summary", workspace="default")
    store.add_message(sid, "user", "Summarize the graph.")
    secret_like = "sk-" + "testsecret123456"
    assistant_id = store.add_message(
        sid,
        "assistant",
        f"Final answer includes actionable findings and a token {secret_like} that must not leak.",
    )

    graph = KnowledgeGraphProjector(store).project(workspace_id="default")
    session_node = next(node for node in graph.nodes if node.node_key == f"session:{sid}")

    assert session_node.summary
    assert session_node.attrs["assistant_summary"] == session_node.summary
    assert session_node.attrs["assistant_message_id"] == assistant_id
    assert secret_like not in session_node.summary
    assert "[redacted]" in session_node.summary
    assert len(session_node.summary) <= 220
    store.close()


def test_projector_assistant_summary_is_absent_without_assistant_message() -> None:
    store = Store(":memory:")
    sid = store.create_session(title="No assistant yet", workspace="default")
    store.add_message(sid, "user", "Only a user message")

    graph = KnowledgeGraphProjector(store).project(workspace_id="default")
    session_node = next(node for node in graph.nodes if node.node_key == f"session:{sid}")

    assert session_node.summary == ""
    assert "assistant_summary" not in session_node.attrs
    store.close()


def test_projector_adds_bounded_memory_preview_without_raw_secrets() -> None:
    store = Store(":memory:")
    secret_like = "sk-" + "testsecret123456"
    memory_id = store.add_memory(
        namespace="session",
        scope_key=store.create_session(title="Memory preview", workspace="default"),
        content=f"Session distill summary with token {secret_like} that must not leak to the graph.",
    )

    graph = KnowledgeGraphProjector(store).project(workspace_id="default")
    memory_node = next(node for node in graph.nodes if node.node_key == f"memory:{memory_id}")

    assert memory_node.summary
    assert memory_node.attrs["content_excerpt"]
    assert memory_node.summary == memory_node.attrs["content_excerpt"] or len(memory_node.summary) <= len(
        memory_node.attrs["content_excerpt"]
    )
    assert secret_like not in memory_node.summary
    assert "[redacted]" in memory_node.summary
    assert len(memory_node.summary) <= 120
    assert len(memory_node.attrs["content_excerpt"]) <= 320
    store.close()


def test_projector_truncates_long_memory_content_in_preview() -> None:
    store = Store(":memory:")
    sid = store.create_session(title="Long memory", workspace="default")
    long_body = "linked workspace memory body must stay out of the map " + ("x" * 400)
    memory_id = store.add_memory(namespace="workspace", scope_key="default", content=long_body)

    graph = KnowledgeGraphProjector(store).project(workspace_id="default")
    memory_node = next(node for node in graph.nodes if node.node_key == f"memory:{memory_id}")

    assert memory_node.summary
    assert long_body not in memory_node.summary
    assert long_body not in memory_node.attrs["content_excerpt"]
    assert "linked workspace memory body" in memory_node.summary
    store.close()


def test_session_map_preserves_session_lanes_and_memory_reference_boundaries() -> None:
    store = Store(":memory:")
    ids = _seed_graph_fixture(store)
    linked_memory_id = store.add_memory(
        namespace="workspace",
        scope_key="default",
        content="linked workspace memory body must stay out of the map",
    )
    store.add_library_provenance(
        artifact_type="memory",
        artifact_id=linked_memory_id,
        session_id=ids["session_id"],
        run_id=ids["run_id"],
    )
    assistant_id = store.add_message(
        ids["session_id"],
        "assistant",
        "This is the safe latest assistant summary for the Session Wiki Map.",
    )
    KnowledgeGraphBuilder(store).rebuild(workspace_id="default")

    session_map = KnowledgeGraphService(store).get_session_map(
        workspace_id="default",
        session_limit=10,
        member_preview=10,
    )

    assert session_map["layout_mode"] == "session_tree"
    assert session_map["active_version"]["id"]
    assert len(session_map["sessions"]) == 1
    lane = session_map["sessions"][0]
    assert lane["session"]["key"] == f"session:{ids['session_id']}"
    assert lane["session"]["last_assistant_message_id"] == assistant_id
    assert lane["counts"]["reports"] == 1
    assert lane["counts"]["notes"] == 1
    assert lane["source_set"]["count"] == 1
    assert f"source:{ids['doc_id']}" in lane["source_set"]["node_keys"]
    assert f"memory:{ids['session_memory_id']}" in lane["memory_set"]["node_keys"]
    assert f"memory:{linked_memory_id}" in lane["memory_set"]["node_keys"]
    assert f"memory:{ids['workspace_memory_id']}" not in lane["memory_set"]["node_keys"]
    assert lane["memory_set"]["members"]

    linked_node = session_map["nodes_by_key"][f"memory:{linked_memory_id}"]
    assert linked_node["referencing_sessions"][0]["session_id"] == ids["session_id"]
    assert linked_node["summary"]
    assert linked_node["attrs"]["content_excerpt"]
    assert len(linked_node["summary"]) <= 120
    assert f"memory:{ids['workspace_memory_id']}" not in session_map["nodes_by_key"]
    store.close()


def test_session_map_orders_sessions_by_recent_activity_before_limit() -> None:
    store = Store(":memory:")
    old_sid = store.create_session(title="Old research", workspace="default")
    created_recent_sid = store.create_session(title="Created recent", workspace="default")
    updated_recent_sid = store.create_session(title="Updated recent", workspace="default")
    store._conn.executemany(  # noqa: SLF001 - pin session timestamps for deterministic ordering
        "UPDATE sessions SET created_at=?, updated_at=? WHERE id=?",
        [
            (100.0, 150.0, old_sid),
            (300.0, 300.0, created_recent_sid),
            (50.0, 500.0, updated_recent_sid),
        ],
    )
    store._conn.commit()  # noqa: SLF001
    KnowledgeGraphBuilder(store).rebuild(workspace_id="default")

    session_map = KnowledgeGraphService(store).get_session_map(
        workspace_id="default",
        session_limit=2,
        member_preview=0,
    )

    assert [lane["session"]["key"] for lane in session_map["sessions"]] == [
        f"session:{updated_recent_sid}",
        f"session:{created_recent_sid}",
    ]
    assert all(lane["session"]["key"] != f"session:{old_sid}" for lane in session_map["sessions"])
    store.close()


class _FailingProjector:
    def project(self, *, workspace_id: str = "default"):  # noqa: ANN201
        raise RuntimeError("projection boom")


class _FailingJobBackend:
    name = "rq"

    def enqueue(self, payload, job_id):  # noqa: ANN001, ANN201, ARG002
        secret_like = "sk-" + "testsecret"
        raise RuntimeError(f"Job ID must only contain letters, numbers, underscores and dashes: {secret_like}")


class _RecordingJobBackend:
    name = "rq"

    def __init__(self) -> None:
        self.calls: list[tuple[dict, str]] = []

    def enqueue(self, payload, job_id):  # noqa: ANN001, ANN201
        self.calls.append((dict(payload), str(job_id)))
        return str(job_id)


class _FakeQueue:
    def __init__(self) -> None:
        self.kwargs = {}

    def enqueue(self, *args, **kwargs):  # noqa: ANN002, ANN003, ANN201
        self.kwargs = dict(kwargs)


def test_builder_failure_preserves_previous_active_version() -> None:
    store = Store(":memory:")
    _seed_graph_fixture(store)
    ok = KnowledgeGraphBuilder(store).rebuild(workspace_id="default")
    active_before = KnowledgeGraphStore(store).get_active_version("default")
    assert active_before and active_before["id"] == ok["version_id"]

    with pytest.raises(RuntimeError):
        KnowledgeGraphBuilder(store, projector=_FailingProjector()).rebuild(workspace_id="default")

    active_after = KnowledgeGraphStore(store).get_active_version("default")
    failed_rows = store._conn.execute(  # noqa: SLF001
        "SELECT status FROM kg_graph_versions WHERE status='failed'"
    ).fetchall()
    assert active_after and active_after["id"] == active_before["id"]
    assert failed_rows
    store.close()


def test_rq_knowledge_graph_backend_uses_rq_safe_job_id() -> None:
    queue = _FakeQueue()
    backend = RQKnowledgeGraphJobBackend(queue)

    backend_job_id = backend.enqueue({"workspace_id": "default"}, "kgjob_abc123")

    assert backend_job_id == "kgjob_abc123"
    assert queue.kwargs["job_id"] == "kgjob_abc123"
    assert ":" not in queue.kwargs["job_id"]


def test_rebuild_enqueue_failure_marks_job_failed_and_allows_retry() -> None:
    store = Store(":memory:")
    service = KnowledgeGraphService(store)
    service._job_backend = _FailingJobBackend()  # noqa: SLF001 - precise backend failure contract

    with pytest.raises(BaselineError) as error:
        service.start_rebuild({"workspace_id": "default"})

    assert error.value.code == ErrorCode.INTERNAL_ERROR
    rows = store._conn.execute(  # noqa: SLF001
        "SELECT * FROM kg_build_jobs WHERE workspace_id='default' ORDER BY started_at"
    ).fetchall()
    assert len(rows) == 1
    assert rows[0]["status"] == "failed"
    assert rows[0]["error_code"] == "ENQUEUE_FAILED"
    assert "sk-" + "testsecret" not in rows[0]["error_message"]
    assert (
        rows[0]["error_message"]
        == "Knowledge graph rebuild could not be queued. Please retry."
    )

    retry_backend = _RecordingJobBackend()
    service._job_backend = retry_backend  # noqa: SLF001
    retry = service.start_rebuild({"workspace_id": "default"})

    assert retry["status"] == "queued"
    assert retry["backend_job_id"] == retry["id"]
    assert retry_backend.calls == [({"workspace_id": "default"}, retry["id"])]
    store.close()


def test_stale_orphan_queued_job_is_reconciled_before_rebuild() -> None:
    store = Store(":memory:")
    ops = KnowledgeGraphStore(store)
    orphan = ops.create_build_job(workspace_id="default", backend="rq")
    store._conn.execute(  # noqa: SLF001
        "UPDATE kg_build_jobs SET started_at=? WHERE id=?",
        (time.time() - 120, orphan["id"]),
    )
    store._conn.commit()  # noqa: SLF001
    service = KnowledgeGraphService(store)
    service._job_backend = _RecordingJobBackend()  # noqa: SLF001

    created = service.start_rebuild({"workspace_id": "default"})
    reconciled = ops.get_build_job(orphan["id"])

    assert reconciled
    assert reconciled["status"] == "failed"
    assert reconciled["error_code"] == "ENQUEUE_FAILED"
    assert created["status"] == "queued"
    store.close()


def test_knowledge_graph_api_rebuild_graph_node_and_running_conflict() -> None:
    pytest.importorskip("fastapi.testclient", reason="fastapi not installed")
    from fastapi.testclient import TestClient

    service = BaselineService(provider=ProviderSettings(), db_path=":memory:")
    ids = _seed_graph_fixture(service.store)
    client = TestClient(create_app(service))

    response = client.post("/api/v1/knowledge-graph/rebuild", json={"workspace_id": "default"})
    assert response.status_code == 200
    job_id = response.json()["id"]
    for _ in range(50):
        status = client.get("/api/v1/knowledge-graph/status").json()
        if status["last_build_job"]["id"] == job_id and status["last_build_job"]["status"] == "succeeded":
            break
        time.sleep(0.02)
    else:
        raise AssertionError("knowledge graph rebuild did not complete")

    graph = client.get("/api/v1/knowledge-graph/graph").json()
    assert graph["active_version"]["id"]
    assert len(graph["nodes"]) >= 4
    report_node = next(node for node in graph["nodes"] if node["key"] == f"report:{ids['report_id']}")
    assert "Full body" not in report_node["summary"]
    detail = client.get(f"/api/v1/knowledge-graph/nodes/{report_node['id']}").json()
    assert detail["node"]["id"] == report_node["id"]
    assert detail["node"]["provenance"]
    session_map = client.get("/api/v1/knowledge-graph/session-map?member_preview=10").json()
    assert session_map["layout_mode"] == "session_tree"
    assert session_map["sessions"][0]["session"]["key"] == f"session:{ids['session_id']}"
    assert session_map["sessions"][0]["source_set"]["members"]

    running = KnowledgeGraphStore(service.store).create_build_job(workspace_id="default", backend="thread")
    KnowledgeGraphStore(service.store).mark_build_job_running(running["id"], version_id=graph["active_version"]["id"])
    conflict = client.post("/api/v1/knowledge-graph/rebuild", json={"workspace_id": "default"})
    assert conflict.status_code == 409
    assert conflict.json()["error"]["code"] == "KNOWLEDGE_GRAPH_BUILD_RUNNING"
    service.store.close()


def test_knowledge_graph_settings_api_default_update_and_validation(tmp_path) -> None:
    pytest.importorskip("fastapi.testclient", reason="fastapi not installed")
    from fastapi.testclient import TestClient

    config_path = tmp_path / "config.json"
    service = BaselineService(provider=ProviderSettings(), db_path=":memory:", config_path=config_path)
    client = TestClient(create_app(service))

    assert client.get("/api/v1/settings/knowledge-graph").json() == {
        "post_run_rebuild_enabled": True,
        "wiki_auto_refresh_after_kg_rebuild": False,
    }
    saved = client.put(
        "/api/v1/settings/knowledge-graph",
        json={"post_run_rebuild_enabled": False, "wiki_auto_refresh_after_kg_rebuild": True},
    )
    assert saved.status_code == 200
    assert saved.json() == {"post_run_rebuild_enabled": False, "wiki_auto_refresh_after_kg_rebuild": True}

    reloaded = BaselineService(provider=ProviderSettings(), db_path=":memory:", config_path=config_path)
    assert reloaded.get_knowledge_graph_config() == {
        "post_run_rebuild_enabled": False,
        "wiki_auto_refresh_after_kg_rebuild": True,
    }

    invalid = client.put(
        "/api/v1/settings/knowledge-graph",
        json={"post_run_rebuild_enabled": "yes"},
    )
    assert invalid.status_code == 400
    assert invalid.json()["error"]["code"] == "INVALID_REQUEST"
    invalid_wiki = client.put(
        "/api/v1/settings/knowledge-graph",
        json={"post_run_rebuild_enabled": True, "wiki_auto_refresh_after_kg_rebuild": "yes"},
    )
    assert invalid_wiki.status_code == 400
    assert invalid_wiki.json()["error"]["code"] == "INVALID_REQUEST"
    service.store.close()
    reloaded.store.close()
