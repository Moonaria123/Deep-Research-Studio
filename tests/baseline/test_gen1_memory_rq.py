"""Gen1 RQ post-turn Memory lifecycle regression coverage."""

from __future__ import annotations

import time
from types import SimpleNamespace

from deepresearch_core.baseline import BaselineService
from deepresearch_core.baseline import worker
from deepresearch_core.baseline.budget import Mode, RunRequest
from deepresearch_core.baseline.job_backend import DEFAULT_JOB_TIMEOUT_S, RQJobBackend
from deepresearch_core.settings import ProviderSettings
from tests.baseline.fakes import FakeLLM


class _FakeQueue:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple, dict]] = []

    def enqueue(self, *args, **kwargs) -> None:
        self.calls.append((args, kwargs))


def _request() -> RunRequest:
    return RunRequest.from_payload(
        {
            "query": "post-turn memory lifecycle",
            "mode": "fast",
            "deadline_at": time.time() + 60,
        }
    )


def test_rq_routes_post_turn_memory_to_independent_persistent_queue(monkeypatch) -> None:
    monkeypatch.delenv("DEEPRESEARCH_JOB_TIMEOUT_S", raising=False)
    research_queue = _FakeQueue()
    maintenance_queue = _FakeQueue()
    failure_callback = object()
    backend = RQJobBackend(
        research_queue,
        connection=None,
        url="redis://unused",
        maintenance_queue=maintenance_queue,
        post_turn_memory_failure_callback=failure_callback,
    )

    job_id = backend.enqueue_post_turn_memory(
        {
            "session_id": "session-rq",
            "run_id": "run-rq",
            "workspace_key": "default",
            "user_key": "default",
        }
    )

    assert research_queue.calls == []
    args, kwargs = maintenance_queue.calls[0]
    assert args == (
        "deepresearch_core.baseline.worker.run_post_turn_memory_job",
        {
            "session_id": "session-rq",
            "run_id": "run-rq",
            "workspace_key": "default",
            "user_key": "default",
        },
    )
    assert job_id == "post-turn-memory-run-rq"
    assert kwargs["job_id"] == job_id
    assert kwargs["job_timeout"] == DEFAULT_JOB_TIMEOUT_S
    assert kwargs["on_failure"] is failure_callback


def test_research_worker_reloads_memory_gate_but_does_not_wait_for_maintenance(monkeypatch) -> None:
    calls: list[str] = []

    class _WorkerService:
        def reconfigure(self, provider, **kwargs) -> None:  # noqa: ANN001, ANN003
            calls.append("reconfigure")

        def reload_post_run_memory_settings_for_worker(self) -> None:
            calls.append("reload_memory_settings")

        def execute_run(self, request, run_id: str) -> None:  # noqa: ANN001
            calls.append("execute")

    monkeypatch.setattr(worker, "_service", _WorkerService())
    monkeypatch.setattr(worker, "_last_provider_settings", None)
    monkeypatch.setattr(worker, "_last_provider_sources", None)
    monkeypatch.setattr(worker, "_last_llm_registry", None)
    monkeypatch.setattr(
        worker,
        "_load_worker_provider_settings",
        lambda: (ProviderSettings(), {}, None),
    )

    worker.run_job(_request().to_payload(), "run-rq")

    assert calls == ["reconfigure", "reload_memory_settings", "execute"]


def test_rq_service_persists_refresh_instead_of_starting_daemon_thread(monkeypatch) -> None:
    service = BaselineService(provider=ProviderSettings(), db_path=":memory:")
    accepted: list[dict] = []
    service._job_backend = SimpleNamespace(
        name="rq",
        enqueue_post_turn_memory=lambda payload: accepted.append(dict(payload)),
    )
    monkeypatch.setattr(
        service._memory_refresh_queue,
        "enqueue",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("daemon thread used")),
    )
    request = RunRequest(
        query="memory",
        mode=Mode.FAST,
        session_id="session-rq",
        workspace_key="workspace-rq",
        user_key="user-rq",
    )
    outcome = SimpleNamespace(
        run_id="run-rq",
        status="completed",
        answer=SimpleNamespace(text="Delivered report"),
    )

    service._enqueue_memory_refresh_after_run(request, outcome)

    assert accepted == [
        {
            "session_id": "session-rq",
            "run_id": "run-rq",
            "workspace_key": "workspace-rq",
            "user_key": "user-rq",
        }
    ]


def test_post_turn_maintenance_job_reloads_runtime_and_executes(monkeypatch) -> None:
    calls: list[object] = []
    provider = ProviderSettings()

    class _MaintenanceService:
        def reload_memory_runtime_for_worker(self, loaded_provider) -> None:  # noqa: ANN001
            calls.append(("reload", loaded_provider))

        def execute_post_turn_memory_job(self, payload):  # noqa: ANN001, ANN201
            calls.append(("execute", payload))
            return {"status": "completed"}

    monkeypatch.setattr(worker, "_service", _MaintenanceService())
    monkeypatch.setattr(
        worker,
        "_load_worker_provider_settings",
        lambda: (provider, {}, None),
    )
    payload = {
        "session_id": "session-rq",
        "run_id": "run-rq",
        "workspace_key": "default",
        "user_key": "default",
    }

    result = worker.run_post_turn_memory_job(payload)

    assert result == {"status": "completed"}
    assert calls == [("reload", provider), ("execute", payload)]


def test_maintenance_job_embeds_current_workspace_memory_without_rewriting_run(monkeypatch) -> None:
    service = BaselineService(provider=ProviderSettings(), db_path=":memory:")
    service.configure_runtime_for_tests(semantic_embedder=FakeLLM())
    session_id = service.store.create_session(title="Memory")
    run_id = service.store.create_run(session_id=session_id, mode="fast", query="Memory")
    service.store.update_run(run_id, status="completed", result={"answer": {"text": "report"}})
    memory_id = service.store.add_memory(
        namespace="workspace",
        scope_key="default",
        content="Tidal energy finding",
        metadata={"source": "post_turn_memory_v1", "run_id": run_id},
    )
    monkeypatch.setattr(
        service,
        "refresh_post_turn_memory",
        lambda **_kwargs: {"status": "completed", "memory_id": memory_id},
    )
    events_before = service.store.list_events(run_id, after_seq=0)

    result = service.execute_post_turn_memory_job(
        {
            "session_id": session_id,
            "run_id": run_id,
            "workspace_key": "default",
            "user_key": "default",
        }
    )
    vectors = service.store.iter_memories_with_embeddings(namespace="workspace", scope_key="default")
    vector_before_replay = vectors[0]["embedding"]
    replay = service.execute_post_turn_memory_job(
        {
            "session_id": session_id,
            "run_id": run_id,
            "workspace_key": "default",
            "user_key": "default",
        }
    )

    vectors = service.store.iter_memories_with_embeddings(namespace="workspace", scope_key="default")
    assert result["embedding_status"] == "completed"
    assert replay["embedding_status"] == "completed"
    assert [row["id"] for row in vectors] == [memory_id]
    assert vectors[0]["embedding"] == vector_before_replay
    assert service.store.get_run(run_id)["status"] == "completed"
    assert service.store.list_events(run_id, after_seq=0) == events_before


def test_rq_failure_callback_persists_failed_audit(monkeypatch) -> None:
    calls: list[tuple[dict, str]] = []

    class _Service:
        def fail_post_turn_memory_job(self, payload, *, error_code):  # noqa: ANN001
            calls.append((dict(payload), error_code))

    payload = {"run_id": "run-rq", "session_id": "session-rq"}
    job = SimpleNamespace(args=[payload])
    monkeypatch.setattr(worker, "_get_service", lambda: _Service())

    worker.mark_post_turn_memory_failed(job, None, TimeoutError)

    assert calls == [(payload, "TimeoutError")]


def test_rq_failure_callback_logs_audit_persistence_error(monkeypatch) -> None:
    logged: list[tuple[str, dict]] = []

    class _Service:
        def fail_post_turn_memory_job(self, payload, *, error_code):  # noqa: ANN001, ARG002
            raise OSError("database unavailable")

    class _Log:
        def warning(self, event, **fields):  # noqa: ANN001
            logged.append((event, fields))

        def exception(self, event, **fields):  # noqa: ANN001
            logged.append((event, fields))

    monkeypatch.setattr(worker, "_get_service", lambda: _Service())
    monkeypatch.setattr(worker, "get_logger", lambda _name: _Log())

    worker.mark_post_turn_memory_failed(
        SimpleNamespace(args=[{"run_id": "run-rq", "session_id": "session-rq"}]),
        None,
        TimeoutError,
    )

    assert logged == [
        (
            "mark_post_turn_memory_failed_error",
            {
                "run_id": "run-rq",
                "session_id": "session-rq",
                "reason": "TimeoutError",
                "error_type": "OSError",
            },
        )
    ]


def test_maintenance_failure_is_audited_without_rewriting_terminal_run(monkeypatch) -> None:
    service = BaselineService(provider=ProviderSettings(), db_path=":memory:")
    session_id = service.store.create_session(title="Memory")
    run_id = service.store.create_run(session_id=session_id, mode="fast", query="Memory")
    service.store.update_run(run_id, status="completed", result={"answer": {"text": "report"}})
    monkeypatch.setattr(
        service,
        "refresh_post_turn_memory",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("provider failed")),
    )
    events_before = service.store.list_events(run_id, after_seq=0)

    try:
        service.execute_post_turn_memory_job(
            {
                "session_id": session_id,
                "run_id": run_id,
                "workspace_key": "default",
                "user_key": "default",
            }
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError("maintenance failure did not propagate to RQ")

    audit = service.get_memory_write_audit({"workspace_key": "default", "user_key": "default"})
    assert audit["source"] == "post_turn_memory_v1"
    assert audit["run_id"] == run_id
    assert audit["session_id"] == session_id
    assert audit["attempted"] == 1
    assert audit["failed"] == 1
    assert service.store.get_run(run_id)["status"] == "completed"
    assert service.store.list_events(run_id, after_seq=0) == events_before


def test_session_deletion_removes_post_turn_audit() -> None:
    service = BaselineService(provider=ProviderSettings(), db_path=":memory:")
    session_id = service.store.create_session(title="Delete")
    run_id = service.store.create_run(session_id=session_id, mode="fast", query="Delete")
    service.store.upsert_post_turn_memory_run(
        run_id=run_id,
        session_id=session_id,
        workspace_key="default",
        user_key="default",
        status="failed",
        error_code="TEST",
    )

    assert service.store.delete_session(session_id) is True
    assert service.store.get_latest_post_turn_memory_run(workspace_key="default") is None


def test_failure_audit_preserves_provider_used_by_started_job() -> None:
    service = BaselineService(provider=ProviderSettings(), db_path=":memory:")
    session_id = service.store.create_session(title="Audit")
    run_id = service.store.create_run(session_id=session_id, mode="fast", query="Audit")
    service.store.update_run(run_id, status="completed", result={"answer": {"text": "report"}})
    service.store.upsert_post_turn_memory_run(
        run_id=run_id,
        session_id=session_id,
        workspace_key="default",
        user_key="default",
        status="running",
        provider_ref="actual:model",
    )
    events_before = service.store.list_events(run_id, after_seq=0)

    service.fail_post_turn_memory_job(
        {
            "run_id": run_id,
            "session_id": session_id,
            "workspace_key": "default",
            "user_key": "default",
        },
        error_code="HorseTerminated",
    )

    audit = service.store.get_post_turn_memory_run(run_id)
    assert audit["status"] == "failed"
    assert audit["provider_ref"] == "actual:model"
    assert service.store.get_run(run_id)["status"] == "completed"
    assert service.store.list_events(run_id, after_seq=0) == events_before


def test_late_failure_callback_cannot_rewrite_completed_audit() -> None:
    service = BaselineService(provider=ProviderSettings(), db_path=":memory:")
    session_id = service.store.create_session(title="Audit")
    run_id = service.store.create_run(session_id=session_id, mode="fast", query="Audit")
    service.store.update_run(run_id, status="completed", result={"answer": {"text": "report"}})
    service.store.upsert_post_turn_memory_run(
        run_id=run_id,
        session_id=session_id,
        workspace_key="default",
        user_key="default",
        status="completed",
        provider_ref="actual:model",
        embedding_status="completed",
    )
    events_before = service.store.list_events(run_id, after_seq=0)

    transitioned = service.fail_post_turn_memory_job(
        {"run_id": run_id, "session_id": session_id},
        error_code="HorseTerminated",
    )

    audit = service.store.get_post_turn_memory_run(run_id)
    assert transitioned is False
    assert audit["status"] == "completed"
    assert audit["provider_ref"] == "actual:model"
    assert audit["embedding_status"] == "completed"
    assert service.store.get_run(run_id)["status"] == "completed"
    assert service.store.list_events(run_id, after_seq=0) == events_before
