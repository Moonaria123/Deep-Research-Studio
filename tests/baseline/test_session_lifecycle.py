"""Session archive, restore, delete cascade, and content lock (410)."""

from __future__ import annotations

import pytest

from deepresearch_core.baseline import BaselineService
from deepresearch_core.baseline.errors import BaselineError, ErrorCode
from deepresearch_core.baseline.memory import MemoryEngine, MemoryLayer
from deepresearch_core.baseline.store import Store
from deepresearch_core.settings import ProviderSettings

pytest.importorskip("fastapi.testclient", reason="fastapi not installed")


def _service() -> BaselineService:
    return BaselineService(provider=ProviderSettings(), db_path=":memory:")


def test_archive_hides_from_active_and_lists_archived():
    service = _service()
    sid = service.store.create_session(title="archived-one")
    service.archive_session(sid)
    active_ids = [s["id"] for s in service.list_sessions()]
    archived = service.list_archived_sessions()
    assert sid not in active_ids
    assert any(s["id"] == sid for s in archived)
    assert archived[0]["archived_at"] is not None
    assert "latest_run_id" not in archived[0]
    assert archived[0]["latest_run_mode"] is None


def test_archived_sessions_include_latest_run_mode():
    service = _service()
    sid = service.store.create_session(title="archived-with-run")
    service.store.create_run(session_id=sid, mode="multiple", query="q", budget={})
    service.archive_session(sid)
    archived = service.list_archived_sessions()
    row = next(s for s in archived if s["id"] == sid)
    assert row["latest_run_mode"] == "multiple"


def test_restore_returns_to_active_list():
    service = _service()
    sid = service.store.create_session(title="restore-me")
    service.archive_session(sid)
    service.restore_session(sid)
    active_ids = [s["id"] for s in service.list_sessions()]
    assert sid in active_ids
    assert not any(s["id"] == sid for s in service.list_archived_sessions())


def test_delete_cascades_session_scoped_data():
    service = _service()
    store = service.store
    sid = store.create_session(title="delete-me")
    rid = store.create_run(session_id=sid, mode="fast", query="q", budget={})
    store.append_event(rid, "workflow_start", {"ok": True})
    store.add_citation(run_id=rid, document_id="doc1", url="u", title="t", snippet="s", score=0.5)
    store.add_message(sid, "user", "hello")
    MemoryEngine(store).remember(MemoryLayer.SESSION, sid, "session fact", salience=1.0)
    # Generic workspace memory must survive delete.
    MemoryEngine(store).remember(MemoryLayer.WORKSPACE, "default", "workspace fact", salience=1.0)
    store.upsert_workspace_session_memory(
        session_id=sid,
        workspace_key="default",
        content="session summary block",
        metadata={"session_title": "delete-me"},
    )

    service.delete_session(sid)

    assert store.get_session(sid) is None
    assert store.get_run(rid) is None
    assert store.list_messages(sid) == []
    assert store.list_events(rid) == []
    assert store.list_citations(rid) == []
    session_mems = store._conn.execute(
        "SELECT id FROM memories WHERE namespace=? AND scope_key=?",
        ("session", sid),
    ).fetchall()
    assert session_mems == []
    workspace_mems = store._conn.execute(
        "SELECT content, metadata FROM memories WHERE namespace=? AND scope_key=?",
        ("workspace", "default"),
    ).fetchall()
    assert len(workspace_mems) == 1
    assert workspace_mems[0]["content"] == "workspace fact"


def test_archived_session_blocks_content_access():
    service = _service()
    sid = service.store.create_session(title="locked")
    rid = service.store.create_run(session_id=sid, mode="fast", query="q", budget={})
    service.store.add_message(sid, "user", "secret")
    service.archive_session(sid)

    with pytest.raises(BaselineError) as run_err:
        service.get_run(rid)
    assert run_err.value.code == ErrorCode.SESSION_ARCHIVED

    with pytest.raises(BaselineError) as msg_err:
        service.list_messages(sid)
    assert msg_err.value.code == ErrorCode.SESSION_ARCHIVED

    with pytest.raises(BaselineError) as search_err:
        service.search_history(sid, {"query": "secret"})
    assert search_err.value.code == ErrorCode.SESSION_ARCHIVED


def test_http_archive_restore_delete_roundtrip():
    from fastapi.testclient import TestClient

    from deepresearch_core.baseline import create_app

    service = _service()
    sid = service.store.create_session(title="http")
    rid = service.store.create_run(session_id=sid, mode="fast", query="q", budget={})
    client = TestClient(create_app(service))

    res = client.post(f"/api/v1/sessions/{sid}/archive")
    assert res.status_code == 200
    assert res.json()["archived"] is True

    assert sid not in [s["id"] for s in client.get("/api/v1/sessions").json()["sessions"]]
    archived = client.get("/api/v1/sessions/archived").json()["sessions"]
    assert any(s["id"] == sid for s in archived)

    locked = client.get(f"/api/v1/runs/{rid}")
    assert locked.status_code == 410
    assert locked.json()["error"]["code"] == "SESSION_ARCHIVED"

    msgs = client.get(f"/api/v1/sessions/{sid}/messages")
    assert msgs.status_code == 410

    restore = client.post(f"/api/v1/sessions/{sid}/restore")
    assert restore.status_code == 200
    assert client.get(f"/api/v1/runs/{rid}").status_code == 200

    delete = client.delete(f"/api/v1/sessions/{sid}")
    assert delete.status_code == 200
    assert client.get(f"/api/v1/sessions/{sid}/messages").status_code == 404
