"""T43 durable Memory Dream job state, RQ dispatch, and cancellation contracts."""

from __future__ import annotations

from types import SimpleNamespace
import sqlite3

import pytest

from deepresearch_core.baseline import BaselineService
from deepresearch_core.baseline.errors import BaselineError, ErrorCode
from deepresearch_core.baseline.job_backend import RQJobBackend
from deepresearch_core.baseline.memory_distillation import MemoryDistillationCancelled, MemoryDistillationService
from deepresearch_core.baseline.store import Store
from deepresearch_core.settings import ProviderSettings
from tests.baseline.fakes import FakeLLM


class _FakeQueue:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple, dict]] = []

    def enqueue(self, *args, **kwargs):  # noqa: ANN002, ANN003, ANN201
        self.calls.append((args, kwargs))
        return SimpleNamespace(id=kwargs.get("job_id"))


class _ControlledDreamBackend:
    name = "rq"

    def __init__(self) -> None:
        self.enqueued: list[tuple[dict, str]] = []
        self.statuses: dict[str, str] = {}

    def enqueue(self, request, run_id):  # noqa: ANN001, ANN201, ARG002
        return None

    def enqueue_memory_dream(self, payload: dict, dream_run_id: str) -> str:
        backend_job_id = f"memory-dream-{dream_run_id}"
        self.enqueued.append((dict(payload), dream_run_id))
        self.statuses[backend_job_id] = "queued"
        return backend_job_id

    def memory_dream_job_status(self, backend_job_id: str) -> str | None:
        return self.statuses.get(backend_job_id)

    def cancel_memory_dream(self, backend_job_id: str) -> str:
        self.statuses[backend_job_id] = "cancelled"
        return "cancelled"

    def health(self) -> dict:
        return {"job_backend": self.name, "redis_reachable": True}


def _service() -> BaselineService:
    service = BaselineService(provider=ProviderSettings(), db_path=":memory:")
    service.configure_runtime_for_tests(llm=FakeLLM(), memory_llm=FakeLLM())
    return service


def test_store_reserves_only_one_active_dream_per_workspace(tmp_path):
    store = Store(tmp_path / "dream.sqlite3")

    first = store.create_memory_dream_run_if_idle(
        workspace_key="workspace-a",
        user_key="default",
        status="queued",
        backend="rq",
        metadata={"source": "memory_continuity_v1"},
    )
    second = store.create_memory_dream_run_if_idle(
        workspace_key="workspace-a",
        user_key="default",
        status="queued",
        backend="rq",
    )

    assert first["created"] is True
    assert first["run"]["status"] == "queued"
    assert first["run"]["backend"] == "rq"
    assert second == {"created": False, "run": first["run"]}

    assert store.update_memory_dream_run(
        first["run"]["id"],
        status="completed",
        expected_statuses={"queued"},
    )
    third = store.create_memory_dream_run_if_idle(
        workspace_key="workspace-a",
        user_key="default",
        status="queued",
        backend="rq",
    )
    assert third["created"] is True
    assert third["run"]["id"] != first["run"]["id"]
    store.close()


def test_memory_dream_schema_migrates_legacy_table_idempotently(tmp_path):
    path = tmp_path / "legacy.sqlite3"
    connection = sqlite3.connect(path)
    connection.execute(
        "CREATE TABLE memory_dream_runs ("
        "id TEXT PRIMARY KEY, workspace_key TEXT, user_key TEXT, status TEXT, "
        "started_at REAL, finished_at REAL, memories_created INTEGER DEFAULT 0, "
        "error TEXT, metadata TEXT DEFAULT '{}')"
    )
    connection.execute(
        "INSERT INTO memory_dream_runs(id,workspace_key,user_key,status,started_at,metadata) "
        "VALUES ('dream_legacy','default','default','running',1,'{}')"
    )
    connection.commit()
    connection.close()

    first = Store(path)
    first.close()
    second = Store(path)
    columns = {row["name"] for row in second._conn.execute("PRAGMA table_info(memory_dream_runs)").fetchall()}
    row = second.get_memory_dream_run("dream_legacy")

    assert {"backend", "backend_job_id", "updated_at", "cancel_requested_at"} <= columns
    assert row["backend"] == "thread"
    assert row["updated_at"] == 1
    second.close()


def test_late_worker_completion_cannot_overwrite_cancelled_run():
    store = Store(":memory:")
    reserved = store.create_memory_dream_run_if_idle(
        workspace_key="default",
        user_key="default",
        status="queued",
        backend="thread",
    )
    dream_id = reserved["run"]["id"]

    assert store.update_memory_dream_run(
        dream_id,
        status="cancelling",
        expected_statuses={"queued"},
        cancel_requested=True,
    )
    assert store.update_memory_dream_run(
        dream_id,
        status="cancelled",
        expected_statuses={"cancelling"},
    )
    assert not store.update_memory_dream_run(
        dream_id,
        status="completed",
        expected_statuses={"running"},
        memories_created=99,
    )
    row = store.get_memory_dream_run(dream_id)
    assert row["status"] == "cancelled"
    assert row["memories_created"] == 0
    store.close()


def test_cross_connection_worker_cannot_reopen_cancelled_dream(tmp_path):
    path = tmp_path / "cross-process.sqlite3"
    api_store = Store(path)
    worker_store = Store(path)
    reserved = api_store.create_memory_dream_run_if_idle(
        workspace_key="default",
        user_key="default",
        status="queued",
        backend="rq",
    )
    dream_id = reserved["run"]["id"]

    assert api_store.update_memory_dream_run(
        dream_id,
        status="cancelling",
        expected_statuses={"queued"},
        cancel_requested=True,
    )
    assert api_store.update_memory_dream_run(
        dream_id,
        status="cancelled",
        expected_statuses={"cancelling"},
    )
    assert not worker_store.update_memory_dream_run(
        dream_id,
        status="running",
        expected_statuses={"queued"},
    )
    assert worker_store.get_memory_dream_run(dream_id)["status"] == "cancelled"
    worker_store.close()
    api_store.close()


def test_reconcile_preserves_observable_started_job_even_when_old():
    service = _service()
    backend = _ControlledDreamBackend()
    service._job_backend = backend
    reserved = service.store.create_memory_dream_run_if_idle(
        workspace_key="default",
        user_key="default",
        status="running",
        backend="rq",
    )
    dream_id = reserved["run"]["id"]
    backend_job_id = f"memory-dream-{dream_id}"
    service.store.set_memory_dream_backend_job_id(dream_id, backend_job_id)
    backend.statuses[backend_job_id] = "started"
    service.store._conn.execute(
        "UPDATE memory_dream_runs SET updated_at=1 WHERE id=?", (dream_id,)
    )
    service.store._conn.commit()

    service._reconcile_memory_dream_runs(workspace_key="default")

    assert service.store.get_memory_dream_run(dream_id)["status"] == "running"


def test_reconcile_closes_old_missing_thread_job():
    service = _service()
    reserved = service.store.create_memory_dream_run_if_idle(
        workspace_key="default",
        user_key="default",
        status="running",
        backend="thread",
    )
    dream_id = reserved["run"]["id"]
    service.store.set_memory_dream_backend_job_id(dream_id, f"memory-dream-{dream_id}")
    service.store._conn.execute(
        "UPDATE memory_dream_runs SET updated_at=1 WHERE id=?", (dream_id,)
    )
    service.store._conn.commit()

    service._reconcile_memory_dream_runs(workspace_key="default")

    row = service.store.get_memory_dream_run(dream_id)
    assert row["status"] == "error"
    assert row["metadata"]["error_code"] == "MEMORY_DREAM_ORPHANED"


def test_rq_backend_enqueues_memory_dream_on_existing_queue(monkeypatch):
    monkeypatch.setenv("DEEPRESEARCH_JOB_TIMEOUT_S", "2400")
    queue = _FakeQueue()
    failure_callback = object()
    backend = RQJobBackend(
        queue,
        connection=None,
        url="redis://x",
        memory_dream_failure_callback=failure_callback,
    )
    payload = {
        "workspace_key": "default",
        "provider_ref": "provider:model",
        "thinking_budget": 0,
    }

    backend_job_id = backend.enqueue_memory_dream(payload, "dream_1")

    args, kwargs = queue.calls[0]
    assert args == (
        "deepresearch_core.baseline.worker.run_memory_dream_job",
        payload,
        "dream_1",
    )
    assert backend_job_id == "memory-dream-dream_1"
    assert kwargs["job_id"] == backend_job_id
    assert kwargs["job_timeout"] == 2400
    assert kwargs["on_failure"] is failure_callback


def test_service_enforces_single_flight_and_cancels_queued_dream():
    service = _service()
    backend = _ControlledDreamBackend()
    service._job_backend = backend

    started = service.start_memory_dream({"workspace_key": "default", "background": True})

    assert started["status"] == "queued"
    assert started["backend"] == "rq"
    assert started["backend_job_id"].startswith("memory-dream-")
    assert backend.enqueued[0][0]["provider_ref"] == ""
    assert "api_key" not in backend.enqueued[0][0]

    with pytest.raises(BaselineError) as exc:
        service.start_memory_dream({"workspace_key": "default", "background": True})
    assert exc.value.code is ErrorCode.MEMORY_DREAM_RUNNING
    assert exc.value.details["dream_run_id"] == started["dream_run_id"]

    cancelled = service.cancel_memory_dream(started["dream_run_id"])
    assert cancelled["status"] == "cancelled"
    assert cancelled["cancel_requested_at"] is not None


def test_cancel_missing_dream_uses_stable_not_found_code():
    service = _service()

    with pytest.raises(BaselineError) as exc:
        service.cancel_memory_dream("dream_missing")

    assert exc.value.code is ErrorCode.MEMORY_DREAM_NOT_FOUND


def test_memory_dream_worker_reload_and_execute(monkeypatch):
    from deepresearch_core.baseline import worker

    calls: list[tuple[str, object]] = []

    class _WorkerService:
        def reload_memory_runtime_for_worker(self, provider) -> None:  # noqa: ANN001
            calls.append(("reload", provider))

        def execute_memory_dream_job(self, payload: dict, dream_run_id: str) -> None:
            calls.append(("execute", (payload, dream_run_id)))

    provider = ProviderSettings(llm_model="worker-model")
    monkeypatch.setattr(worker, "_service", _WorkerService())
    monkeypatch.setattr(worker, "_load_worker_provider_settings", lambda: provider)

    worker.run_memory_dream_job({"workspace_key": "default"}, "dream_2")

    assert calls == [
        ("reload", provider),
        ("execute", ({"workspace_key": "default"}, "dream_2")),
    ]


def test_distillation_honors_cooperative_cancel_before_writes():
    store = Store(":memory:")
    session_id = store.create_session(title="Cancelled")
    store.add_message(session_id, "assistant", "Do not persist after cancellation")
    distiller = MemoryDistillationService(store=store, memory_llm=lambda: FakeLLM())

    with pytest.raises(MemoryDistillationCancelled):
        distiller.refresh(session_id=session_id, should_cancel=lambda: True)

    assert store.list_memories(namespace="workspace", scope_key="default", limit=10) == []
    assert store.list_memories(namespace="user", scope_key="default", limit=10) == []
    store.close()


def test_memory_dream_api_cancel_is_idempotent():
    pytest.importorskip("fastapi.testclient", reason="fastapi not installed")
    from fastapi.testclient import TestClient

    from deepresearch_core.baseline import create_app

    service = _service()
    service._job_backend = _ControlledDreamBackend()
    client = TestClient(create_app(service))

    started = client.post("/api/v1/memory/dream", json={"background": True}).json()
    first = client.post(f"/api/v1/memory/dream/{started['dream_run_id']}/cancel")
    second = client.post(f"/api/v1/memory/dream/{started['dream_run_id']}/cancel")

    assert first.status_code == 200
    assert first.json()["status"] == "cancelled"
    assert second.status_code == 200
    assert second.json()["status"] == "cancelled"
