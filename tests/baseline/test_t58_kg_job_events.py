"""RED contracts for durable Knowledge Graph and Wiki Map job events."""

from __future__ import annotations

import json
import sqlite3
import time
import traceback
from typing import Any

import pytest
from fastapi.testclient import TestClient

from deepresearch_core.baseline import BaselineService, create_app
from deepresearch_core.baseline.knowledge_graph.job_events import (
    MAX_EVENTS_PER_JOB,
    append_job_event,
)
from deepresearch_core.baseline.knowledge_graph.service import KnowledgeGraphService
from deepresearch_core.baseline.knowledge_graph.store_ops import (
    KnowledgeGraphJobInactive,
    KnowledgeGraphStore,
)
from deepresearch_core.baseline.knowledge_graph.wiki_models import WikiSnapshotDraft
from deepresearch_core.baseline.knowledge_graph.wiki_store import (
    KnowledgeGraphWikiStore,
    WikiRefreshJobInactive,
)
from deepresearch_core.baseline.store import Store
from deepresearch_core.settings import ProviderSettings


TERMINAL_STATUSES = {"succeeded", "partial", "failed"}


def _ensure_event_table_fixture(store: Store) -> None:
    """Allow API RED cases to reach missing routes before the migration exists."""

    store._conn.execute(  # noqa: SLF001 - isolated in-memory compatibility fixture
        """
        CREATE TABLE IF NOT EXISTS kg_job_events (
            job_kind TEXT NOT NULL,
            job_id TEXT NOT NULL,
            seq INTEGER NOT NULL,
            event_type TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            created_at REAL NOT NULL,
            PRIMARY KEY(job_kind, job_id, seq)
        )
        """
    )
    store._conn.commit()  # noqa: SLF001


def _seed_events_if_missing(
    store: Store,
    *,
    job_kind: str,
    job_id: str,
    terminal_status: str,
) -> None:
    if _event_rows(store, job_kind, job_id):
        return
    now = time.time()
    rows = (
        (
            1,
            "job_progress",
            {
                "job_id": job_id,
                "job_kind": job_kind,
                "status": "queued",
                "stage": "queued",
            },
        ),
        (
            2,
            "job_progress",
            {
                "job_id": job_id,
                "job_kind": job_kind,
                "status": "running",
                "stage": "working",
            },
        ),
        (
            3,
            "job_terminal",
            {
                "job_id": job_id,
                "job_kind": job_kind,
                "status": terminal_status,
                "stage": "partial" if terminal_status == "partial" else "completed",
            },
        ),
    )
    store._conn.executemany(  # noqa: SLF001 - isolated in-memory SSE fixture
        "INSERT INTO kg_job_events(job_kind,job_id,seq,event_type,payload_json,created_at) "
        "VALUES (?,?,?,?,?,?)",
        [
            (job_kind, job_id, seq, event_type, json.dumps(payload), now + seq)
            for seq, event_type, payload in rows
        ],
    )
    store._conn.commit()  # noqa: SLF001


def _event_rows(store: Store, job_kind: str, job_id: str) -> list[dict[str, Any]]:
    rows = store._conn.execute(  # noqa: SLF001 - white-box persistence contract
        "SELECT seq,event_type,payload_json,created_at "
        "FROM kg_job_events WHERE job_kind=? AND job_id=? ORDER BY seq ASC",
        (job_kind, job_id),
    ).fetchall()
    return [
        {
            "seq": int(row["seq"]),
            "event_type": str(row["event_type"]),
            "payload": json.loads(str(row["payload_json"])),
            "created_at": float(row["created_at"]),
        }
        for row in rows
    ]


def _sse_frames(raw: str) -> list[dict[str, Any]]:
    frames: list[dict[str, Any]] = []
    for block in raw.replace("\r\n", "\n").split("\n\n"):
        if not block.strip() or block.lstrip().startswith(":"):
            continue
        frame: dict[str, Any] = {}
        data_lines: list[str] = []
        for line in block.splitlines():
            if line.startswith("id:"):
                frame["id"] = int(line.split(":", 1)[1].strip())
            elif line.startswith("event:"):
                frame["event"] = line.split(":", 1)[1].strip()
            elif line.startswith("data:"):
                data_lines.append(line.split(":", 1)[1].lstrip())
        if data_lines:
            frame["data"] = json.loads("\n".join(data_lines))
        frames.append(frame)
    return frames


def _finished_graph_job(store: Store, *, workspace_id: str = "default") -> tuple[KnowledgeGraphStore, str]:
    graph = KnowledgeGraphStore(store)
    job = graph.create_build_job(workspace_id=workspace_id, backend="thread")
    job_id = str(job["id"])
    version_id = graph.create_staging_version(
        workspace_id=workspace_id,
        base_snapshot={"workspace_id": workspace_id},
    )
    graph.mark_build_job_running(job_id, version_id=version_id, stage="projecting")
    graph.mark_build_job_succeeded(job_id, stats={"nodes": 1, "edges": 0})
    return graph, job_id


def _partial_wiki_job(
    store: Store,
    *,
    workspace_id: str = "default",
) -> tuple[KnowledgeGraphWikiStore, str]:
    graph = KnowledgeGraphStore(store)
    graph_version_id = graph.create_staging_version(
        workspace_id=workspace_id,
        base_snapshot={"workspace_id": workspace_id},
    )
    wiki = KnowledgeGraphWikiStore(store)
    job = wiki.create_refresh_job(
        workspace_id=workspace_id,
        graph_version_id=graph_version_id,
        backend="thread",
    )
    job_id = str(job["id"])
    snapshot_id = wiki.create_staging_snapshot(
        workspace_id=workspace_id,
        graph_version_id=graph_version_id,
    )
    wiki.mark_refresh_job_running(
        job_id,
        snapshot_id=snapshot_id,
        stage="collecting_sessions",
    )
    wiki.update_refresh_job_progress(
        job_id,
        stage="generating",
        attempted_sessions=2,
        succeeded_sessions=1,
        skipped_sessions=0,
        failed_sessions=1,
        progress=0.75,
        stats={"attempted_sessions": 2},
    )
    wiki.mark_refresh_job_succeeded(
        job_id,
        status="partial",
        stats={"attempted_sessions": 2, "succeeded_sessions": 1},
    )
    return wiki, job_id


def test_graph_terminal_is_immutable_and_rejects_late_writes() -> None:
    store = Store(":memory:")
    try:
        graph, job_id = _finished_graph_job(store, workspace_id="alpha")
        terminal = graph.get_build_job(job_id)
        events = _event_rows(store, "graph", job_id)

        assert terminal is not None
        assert terminal["status"] == "succeeded"
        assert [event["seq"] for event in events] == list(range(1, len(events) + 1))
        assert events[-1]["event_type"] == "job_terminal"
        assert events[-1]["payload"]["status"] == "succeeded"

        graph.mark_build_job_failed(
            job_id,
            error_code="LATE_FAILURE",
            error_message="late failure must be ignored",
        )
        graph.mark_build_job_running(
            job_id,
            version_id=str(terminal["version_id"]),
            stage="late_restart",
        )
        graph.mark_build_job_succeeded(job_id, stats={"late": True})

        assert graph.get_build_job(job_id) == terminal
        assert _event_rows(store, "graph", job_id) == events
    finally:
        store.close()


def test_wiki_partial_is_terminal_and_rejects_late_writes() -> None:
    store = Store(":memory:")
    try:
        wiki, job_id = _partial_wiki_job(store, workspace_id="alpha")
        terminal = wiki.get_refresh_job(job_id)
        events = _event_rows(store, "wiki", job_id)

        assert terminal is not None
        assert terminal["status"] == "partial"
        assert terminal["status"] in TERMINAL_STATUSES
        assert events[-1]["event_type"] == "job_terminal"
        assert events[-1]["payload"]["status"] == "partial"

        wiki.update_refresh_job_progress(
            job_id,
            stage="late_progress",
            attempted_sessions=99,
            succeeded_sessions=99,
            skipped_sessions=0,
            failed_sessions=0,
            progress=0.99,
            stats={"late": True},
        )
        wiki.mark_refresh_job_failed(
            job_id,
            error_code="LATE_FAILURE",
            error_message="late failure must be ignored",
        )
        wiki.mark_refresh_job_succeeded(job_id, status="succeeded", stats={"late": True})

        assert wiki.get_refresh_job(job_id) == terminal
        assert _event_rows(store, "wiki", job_id) == events
    finally:
        store.close()


def test_terminal_graph_worker_cannot_create_or_mutate_versions() -> None:
    store = Store(":memory:")
    try:
        graph = KnowledgeGraphStore(store)
        rejected = graph.create_build_job(workspace_id="alpha", backend="thread")
        rejected_id = str(rejected["id"])
        graph.mark_build_job_failed(rejected_id, error_code="STALE_WORKER")
        before_count = int(
            store._conn.execute("SELECT COUNT(*) FROM kg_graph_versions").fetchone()[0]  # noqa: SLF001
        )

        with pytest.raises(KnowledgeGraphJobInactive) as rejected_error:
            graph.begin_build_job_version(rejected_id, workspace_id="alpha")
        assert rejected_error.value.terminal is True
        assert int(
            store._conn.execute("SELECT COUNT(*) FROM kg_graph_versions").fetchone()[0]  # noqa: SLF001
        ) == before_count

        running = graph.create_build_job(workspace_id="alpha", backend="thread")
        running_id = str(running["id"])
        version_id = graph.begin_build_job_version(running_id, workspace_id="alpha")
        assert version_id
        graph.mark_build_job_failed(running_id, error_code="DEADLINE_EXCEEDED")
        terminal_events = _event_rows(store, "graph", running_id)

        with pytest.raises(KnowledgeGraphJobInactive):
            graph.replace_version_contents(
                version_id,
                nodes=[],
                edges=[],
                provenance=[],
                job_id=running_id,
            )
        with pytest.raises(KnowledgeGraphJobInactive):
            graph.activate_version(version_id, job_id=running_id)

        version = store._conn.execute(  # noqa: SLF001
            "SELECT status,is_active FROM kg_graph_versions WHERE id=?",
            (version_id,),
        ).fetchone()
        assert dict(version) == {"status": "staging", "is_active": 0}
        assert graph.get_active_version("alpha") is None
        assert _event_rows(store, "graph", running_id) == terminal_events
    finally:
        store.close()


def test_rq_worker_treats_terminalized_graph_job_as_noop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from deepresearch_core.baseline.knowledge_graph import worker

    class _Service:
        def knowledge_graph_rebuild_now(self, payload: dict, *, job_id: str) -> None:
            raise KnowledgeGraphJobInactive(job_id, reason="terminal", status="failed")

    monkeypatch.setattr(worker, "_get_service", lambda: _Service())

    worker.rebuild_job({"workspace_id": "default"}, "kgjob-terminal")


def test_rq_worker_propagates_nonterminal_claim_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from deepresearch_core.baseline.knowledge_graph import worker

    class _Service:
        def knowledge_graph_rebuild_now(self, payload: dict, *, job_id: str) -> None:
            raise KnowledgeGraphJobInactive(
                job_id,
                reason="workspace_mismatch",
                status="queued",
            )

    monkeypatch.setattr(worker, "_get_service", lambda: _Service())

    with pytest.raises(KnowledgeGraphJobInactive) as caught:
        worker.rebuild_job({"workspace_id": "wrong"}, "kgjob-active")
    assert caught.value.reason == "workspace_mismatch"
    assert caught.value.terminal is False


def test_rq_worker_ignores_duplicate_running_claim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from deepresearch_core.baseline.knowledge_graph import worker

    class _Service:
        def knowledge_graph_rebuild_now(self, payload: dict, *, job_id: str) -> None:
            raise KnowledgeGraphJobInactive(
                job_id,
                reason="already_running",
                status="running",
            )

    monkeypatch.setattr(worker, "_get_service", lambda: _Service())

    worker.rebuild_job({"workspace_id": "default"}, "kgjob-running")


def test_rq_worker_sanitizes_unexpected_exception_across_rq_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from deepresearch_core.baseline.knowledge_graph import worker

    sensitive = (
        "prompt=private user research content; "
        r"path=C:\Users\alice\private\report.md"
    )

    class _Service:
        def knowledge_graph_rebuild_now(self, payload: dict, *, job_id: str) -> None:
            raise RuntimeError(sensitive)

    monkeypatch.setattr(worker, "_get_service", lambda: _Service())

    with pytest.raises(worker.KnowledgeGraphWorkerError) as caught:
        worker.rebuild_job({"workspace_id": "default"}, "kgjob-failed")

    assert str(caught.value) == "Knowledge graph rebuild failed."
    assert caught.value.__suppress_context__ is True
    formatted = "".join(
        traceback.format_exception(
            type(caught.value),
            caught.value,
            caught.value.__traceback__,
        )
    )
    for forbidden in (
        "private user research content",
        r"C:\Users\alice\private\report.md",
    ):
        assert forbidden not in formatted


def test_duplicate_running_claim_preserves_authoritative_worker() -> None:
    store = Store(":memory:")
    try:
        graph = KnowledgeGraphStore(store)
        job = graph.create_build_job(workspace_id="alpha", backend="thread")
        job_id = str(job["id"])
        first_version = graph.begin_build_job_version(job_id, workspace_id="alpha")
        events_before = _event_rows(store, "graph", job_id)

        with pytest.raises(KnowledgeGraphJobInactive) as duplicate:
            graph.begin_build_job_version(job_id, workspace_id="alpha")

        assert duplicate.value.reason == "already_running"
        assert duplicate.value.benign is True
        current = graph.get_build_job(job_id)
        assert current is not None
        assert current["status"] == "running"
        assert current["version_id"] == first_version
        assert _event_rows(store, "graph", job_id) == events_before
        assert store._conn.execute(  # noqa: SLF001
            "SELECT COUNT(*) FROM kg_graph_versions"
        ).fetchone()[0] == 1
    finally:
        store.close()


def test_thread_worker_terminalizes_workspace_mismatch() -> None:
    store = Store(":memory:")
    try:
        graph = KnowledgeGraphStore(store)
        job = graph.create_build_job(workspace_id="alpha", backend="thread")
        job_id = str(job["id"])
        graph.set_build_job_backend_id(job_id, "thread-claimed")
        service = KnowledgeGraphService(store)

        service._run_rebuild_safe({"workspace_id": "beta"}, job_id)  # noqa: SLF001

        failed = graph.get_build_job(job_id)
        assert failed is not None
        assert failed["status"] == "failed"
        assert failed["error_code"] == "JOB_CLAIM_REJECTED"
        assert graph.get_running_build_job("alpha") is None
        assert store._conn.execute(  # noqa: SLF001
            "SELECT COUNT(*) FROM kg_graph_versions"
        ).fetchone()[0] == 0
        assert _event_rows(store, "graph", job_id)[-1]["event_type"] == "job_terminal"
    finally:
        store.close()


@pytest.mark.parametrize("initial_status", ["queued", "running"])
def test_thread_worker_terminalizes_unexpected_builder_failure(
    initial_status: str,
) -> None:
    store = Store(":memory:")
    try:
        graph = KnowledgeGraphStore(store)
        job = graph.create_build_job(workspace_id="alpha", backend="thread")
        job_id = str(job["id"])
        if initial_status == "running":
            graph.begin_build_job_version(job_id, workspace_id="alpha")
        service = KnowledgeGraphService(store)

        class _FailingBuilder:
            def rebuild(self, *, workspace_id: str, job_id: str) -> dict[str, Any]:
                raise RuntimeError(
                    "prompt=private user research content; "
                    r"path=C:\Users\alice\private\report.md"
                )

        service._builder = _FailingBuilder()  # type: ignore[assignment]  # noqa: SLF001
        service._run_rebuild_safe({"workspace_id": "alpha"}, job_id)  # noqa: SLF001

        failed = graph.get_build_job(job_id)
        assert failed is not None
        assert failed["status"] == "failed"
        assert failed["error_code"] == "INTERNAL_ERROR"
        public = service.get_rebuild_job(job_id, workspace_id="alpha")
        persisted = json.dumps(failed, sort_keys=True)
        exposed = json.dumps(public, sort_keys=True)
        for forbidden in (
            "private user research content",
            r"C:\Users\alice\private\report.md",
        ):
            assert forbidden not in persisted
            assert forbidden not in exposed
        assert _event_rows(store, "graph", job_id)[-1]["event_type"] == "job_terminal"
    finally:
        store.close()


def test_terminal_wiki_worker_cannot_create_or_mutate_snapshots() -> None:
    store = Store(":memory:")
    try:
        graph = KnowledgeGraphStore(store)
        graph_version_id = graph.create_staging_version(workspace_id="alpha")
        wiki = KnowledgeGraphWikiStore(store)
        rejected = wiki.create_refresh_job(
            workspace_id="alpha",
            graph_version_id=graph_version_id,
            backend="thread",
        )
        rejected_id = str(rejected["id"])
        wiki.mark_refresh_job_failed(rejected_id, error_code="STALE_WORKER")
        before_count = int(
            store._conn.execute("SELECT COUNT(*) FROM kg_wiki_snapshots").fetchone()[0]  # noqa: SLF001
        )

        assert (
            wiki.begin_refresh_job_snapshot(
                rejected_id,
                workspace_id="alpha",
                graph_version_id=graph_version_id,
            )
            is None
        )
        assert int(
            store._conn.execute("SELECT COUNT(*) FROM kg_wiki_snapshots").fetchone()[0]  # noqa: SLF001
        ) == before_count

        running = wiki.create_refresh_job(
            workspace_id="alpha",
            graph_version_id=graph_version_id,
            backend="thread",
        )
        running_id = str(running["id"])
        snapshot_id = wiki.begin_refresh_job_snapshot(
            running_id,
            workspace_id="alpha",
            graph_version_id=graph_version_id,
        )
        assert snapshot_id
        wiki.mark_refresh_job_failed(running_id, error_code="DEADLINE_EXCEEDED")
        terminal_events = _event_rows(store, "wiki", running_id)
        draft = WikiSnapshotDraft(
            workspace_id="alpha",
            graph_version_id=graph_version_id,
            topics=(),
            tree_edges=(),
            cross_links=(),
            stats={},
        )

        with pytest.raises(WikiRefreshJobInactive):
            wiki.replace_snapshot_contents(snapshot_id, draft, job_id=running_id)
        with pytest.raises(WikiRefreshJobInactive):
            wiki.activate_snapshot(snapshot_id, job_id=running_id)

        snapshot = store._conn.execute(  # noqa: SLF001
            "SELECT status,is_active FROM kg_wiki_snapshots WHERE id=?",
            (snapshot_id,),
        ).fetchone()
        assert dict(snapshot) == {"status": "staging", "is_active": 0}
        assert wiki.get_active_snapshot("alpha") is None
        assert _event_rows(store, "wiki", running_id) == terminal_events
    finally:
        store.close()


def test_progress_events_are_immutable_and_replay_same_stage_updates() -> None:
    store = Store(":memory:")
    try:
        job: dict[str, Any] = {
            "id": "kgjob_cursor",
            "status": "running",
            "stage": "projecting",
            "progress": 0.1,
            "updated_at": 1.0,
            "stats_json": {"nodes": 1},
        }
        first_seq = append_job_event(
            store._conn,  # noqa: SLF001
            job_kind="graph",
            job=job,
            terminal=False,
        )
        job.update(
            {
                "progress": 0.8,
                "updated_at": 2.0,
                "stats_json": {"nodes": 2},
            }
        )
        second_seq = append_job_event(
            store._conn,  # noqa: SLF001
            job_kind="graph",
            job=job,
            terminal=False,
        )
        job["updated_at"] = 3.0
        duplicate_seq = append_job_event(
            store._conn,  # noqa: SLF001
            job_kind="graph",
            job=job,
            terminal=False,
        )
        store._conn.commit()  # noqa: SLF001

        assert first_seq == 1
        assert second_seq == 2
        assert duplicate_seq is None
        events = _event_rows(store, "graph", "kgjob_cursor")
        assert events[0]["payload"]["progress"] == 0.1
        replay = KnowledgeGraphStore(store).list_build_job_events(
            "kgjob_cursor",
            after_seq=first_seq,
        )
        assert [event["seq"] for event in replay] == [second_seq]
        assert replay[0]["payload"]["progress"] == 0.8
        assert replay[0]["payload"]["stats"] == {"nodes": 2}
    finally:
        store.close()


def test_job_event_stats_use_numeric_allowlist_only() -> None:
    store = Store(":memory:")
    try:
        job: dict[str, Any] = {
            "id": "kgjob_safe_stats",
            "status": "running",
            "stage": "projecting",
            "progress": 0.5,
            "updated_at": 1.0,
            "stats_json": {
                "nodes": 3,
                "prompt_excerpt": "private user research content",
                "output_path": "C:/Users/alice/private/report.md",
                "unknown_numeric": 99,
                "wiki_seed_snapshot_id": "wikisnap_private",
            },
        }
        append_job_event(
            store._conn,  # noqa: SLF001
            job_kind="graph",
            job=job,
            terminal=False,
        )
        store._conn.commit()  # noqa: SLF001

        event = _event_rows(store, "graph", "kgjob_safe_stats")[0]
        assert event["payload"]["stats"] == {"nodes": 3}
        encoded = json.dumps(event["payload"], sort_keys=True)
        assert "private user research content" not in encoded
        assert "C:/Users/alice/private/report.md" not in encoded
        assert "wikisnap_private" not in encoded
    finally:
        store.close()


def test_job_event_cap_reserves_exactly_one_terminal_slot() -> None:
    store = Store(":memory:")
    try:
        job: dict[str, Any] = {
            "id": "kgjob_cap",
            "status": "running",
            "stage": "stage-0",
            "progress": 0.1,
            "started_at": time.time(),
        }
        for index in range(MAX_EVENTS_PER_JOB + 25):
            job["stage"] = f"stage-{index}"
            append_job_event(store._conn, job_kind="graph", job=job, terminal=False)  # noqa: SLF001
        job.update({"status": "succeeded", "stage": "completed", "progress": 1.0})
        append_job_event(store._conn, job_kind="graph", job=job, terminal=True)  # noqa: SLF001
        store._conn.commit()  # noqa: SLF001

        events = _event_rows(store, "graph", "kgjob_cap")
        assert len(events) == MAX_EVENTS_PER_JOB
        assert [event["seq"] for event in events] == list(range(1, MAX_EVENTS_PER_JOB + 1))
        assert events[-1]["event_type"] == "job_terminal"
        assert events[-1]["payload"]["status"] == "succeeded"
    finally:
        store.close()


def test_job_event_migration_is_idempotent_for_legacy_database(tmp_path) -> None:
    db_path = tmp_path / "legacy-pre-t58.db"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE legacy_marker(id INTEGER PRIMARY KEY, value TEXT NOT NULL)")
    conn.execute("INSERT INTO legacy_marker(value) VALUES ('preserved')")
    conn.commit()
    conn.close()

    for _ in range(2):
        reopened = Store(db_path)
        reopened.close()

    conn = sqlite3.connect(db_path)
    try:
        columns = {
            row[1]
            for row in conn.execute("PRAGMA table_info(kg_job_events)").fetchall()
        }
        marker = conn.execute("SELECT value FROM legacy_marker").fetchone()
        index_names = {
            row[1]
            for row in conn.execute("PRAGMA index_list(kg_job_events)").fetchall()
        }
        assert columns == {
            "job_kind",
            "job_id",
            "seq",
            "event_type",
            "payload_json",
            "created_at",
        }
        assert marker == ("preserved",)
        assert "idx_kg_job_events_cursor" in index_names
    finally:
        conn.close()


def test_job_events_persist_with_monotonic_sequence_and_replay_after_reopen(tmp_path) -> None:
    db_path = tmp_path / "t58-job-events.db"
    store = Store(db_path)
    _, job_id = _finished_graph_job(store)
    before = _event_rows(store, "graph", job_id)
    store.close()

    reopened = Store(db_path)
    try:
        after = _event_rows(reopened, "graph", job_id)
        assert after == before
        assert len(after) >= 3
        assert [event["seq"] for event in after] == list(range(1, len(after) + 1))
        assert after[0]["payload"]["status"] == "queued"
        assert after[-1]["payload"]["status"] == "succeeded"
    finally:
        reopened.close()


def test_graph_job_stream_replays_after_seq_and_query_cursor_wins_over_header() -> None:
    service = BaselineService(provider=ProviderSettings(), db_path=":memory:")
    store = service.store
    try:
        _, job_id = _finished_graph_job(store, workspace_id="alpha")
        _ensure_event_table_fixture(store)
        _seed_events_if_missing(
            store,
            job_kind="graph",
            job_id=job_id,
            terminal_status="succeeded",
        )
        persisted = _event_rows(store, "graph", job_id)
        assert len(persisted) >= 3

        with TestClient(create_app(service)) as client:
            response = client.get(
                f"/api/v1/knowledge-graph/jobs/{job_id}/stream",
                params={"workspace_id": "alpha", "after_seq": 1},
                headers={"Last-Event-ID": "0"},
            )

        assert response.status_code == 200
        frames = _sse_frames(response.text)
        assert [frame["id"] for frame in frames] == [
            event["seq"] for event in persisted if event["seq"] > 1
        ]
        assert all(frame["id"] > 1 for frame in frames)
        assert frames[-1]["event"] == "job_terminal"
        assert frames[-1]["data"]["status"] == "succeeded"
    finally:
        store.close()


def test_wiki_job_stream_replays_from_last_event_id() -> None:
    service = BaselineService(provider=ProviderSettings(), db_path=":memory:")
    store = service.store
    try:
        _, job_id = _partial_wiki_job(store, workspace_id="alpha")
        _ensure_event_table_fixture(store)
        _seed_events_if_missing(
            store,
            job_kind="wiki",
            job_id=job_id,
            terminal_status="partial",
        )
        persisted = _event_rows(store, "wiki", job_id)
        cursor = persisted[-2]["seq"]

        with TestClient(create_app(service)) as client:
            response = client.get(
                f"/api/v1/knowledge-graph/wiki-map/jobs/{job_id}/stream",
                params={"workspace_id": "alpha"},
                headers={"Last-Event-ID": str(cursor)},
            )

        assert response.status_code == 200
        frames = _sse_frames(response.text)
        assert [frame["id"] for frame in frames] == [
            event["seq"] for event in persisted if event["seq"] > cursor
        ]
        assert frames == [
            {
                "id": persisted[-1]["seq"],
                "event": "job_terminal",
                "data": persisted[-1]["payload"],
            }
        ]
    finally:
        store.close()


def test_legacy_terminal_without_events_is_synthesized_without_get_side_effect() -> None:
    service = BaselineService(provider=ProviderSettings(), db_path=":memory:")
    store = service.store
    try:
        _ensure_event_table_fixture(store)
        _, job_id = _finished_graph_job(store, workspace_id="alpha")
        store._conn.execute(  # noqa: SLF001 - construct a pre-T58 legacy row
            "DELETE FROM kg_job_events WHERE job_kind='graph' AND job_id=?",
            (job_id,),
        )
        store._conn.commit()  # noqa: SLF001

        with TestClient(create_app(service)) as client:
            response = client.get(
                f"/api/v1/knowledge-graph/jobs/{job_id}/stream",
                params={"workspace_id": "alpha", "after_seq": 0},
            )

        assert response.status_code == 200
        frames = _sse_frames(response.text)
        assert len(frames) == 1
        assert frames[0]["id"] == 1
        assert frames[0]["event"] == "job_terminal"
        assert frames[0]["data"]["job_id"] == job_id
        assert frames[0]["data"]["job_kind"] == "graph"
        assert frames[0]["data"]["status"] == "succeeded"
        assert frames[0]["data"]["stage"] == "completed"
        assert _event_rows(store, "graph", job_id) == []
    finally:
        store.close()


def test_graph_job_get_and_stream_enforce_workspace_isolation() -> None:
    service = BaselineService(provider=ProviderSettings(), db_path=":memory:")
    store = service.store
    try:
        _ensure_event_table_fixture(store)
        _, job_id = _finished_graph_job(store, workspace_id="alpha")
        with TestClient(create_app(service)) as client:
            allowed = client.get(
                f"/api/v1/knowledge-graph/jobs/{job_id}",
                params={"workspace_id": "alpha"},
            )
            denied_get = client.get(
                f"/api/v1/knowledge-graph/jobs/{job_id}",
                params={"workspace_id": "beta"},
            )
            denied_stream = client.get(
                f"/api/v1/knowledge-graph/jobs/{job_id}/stream",
                params={"workspace_id": "beta"},
            )

        assert allowed.status_code == 200
        assert allowed.json()["workspace_id"] == "alpha"
        assert denied_get.status_code == 400
        assert denied_get.json()["error"]["code"] == "INVALID_REQUEST"
        assert denied_stream.status_code == 400
        assert denied_stream.json()["error"]["code"] == "INVALID_REQUEST"
    finally:
        store.close()


def test_wiki_job_get_and_stream_enforce_workspace_isolation() -> None:
    service = BaselineService(provider=ProviderSettings(), db_path=":memory:")
    store = service.store
    try:
        _ensure_event_table_fixture(store)
        _, job_id = _partial_wiki_job(store, workspace_id="alpha")
        with TestClient(create_app(service)) as client:
            allowed = client.get(
                f"/api/v1/knowledge-graph/wiki-map/jobs/{job_id}",
                params={"workspace_id": "alpha"},
            )
            denied_get = client.get(
                f"/api/v1/knowledge-graph/wiki-map/jobs/{job_id}",
                params={"workspace_id": "beta"},
            )
            denied_stream = client.get(
                f"/api/v1/knowledge-graph/wiki-map/jobs/{job_id}/stream",
                params={"workspace_id": "beta"},
            )

        assert allowed.status_code == 200
        assert allowed.json()["workspace_id"] == "alpha"
        assert denied_get.status_code == 400
        assert denied_get.json()["error"]["code"] == "INVALID_REQUEST"
        assert denied_stream.status_code == 400
        assert denied_stream.json()["error"]["code"] == "INVALID_REQUEST"
    finally:
        store.close()


@pytest.mark.parametrize("cursor", ["-1", "not-a-number", str(2**63)])
def test_job_stream_rejects_invalid_cursor(cursor: str) -> None:
    service = BaselineService(provider=ProviderSettings(), db_path=":memory:")
    store = service.store
    try:
        _, job_id = _finished_graph_job(store, workspace_id="alpha")
        with TestClient(create_app(service)) as client:
            response = client.get(
                f"/api/v1/knowledge-graph/jobs/{job_id}/stream",
                params={"workspace_id": "alpha", "after_seq": cursor},
            )

        assert response.status_code == 400
        assert response.json()["error"]["code"] == "INVALID_REQUEST"
    finally:
        store.close()
