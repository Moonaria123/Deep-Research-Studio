"""Job backend selection, thread execution, rq fail-fast, and cancel semantics (DR-ITER-021)."""

from __future__ import annotations

import threading

import pytest

from deepresearch_core.baseline.budget import Mode, RunRequest
from deepresearch_core.baseline.errors import BaselineError, ErrorCode
from deepresearch_core.baseline.indexing import Indexer
from deepresearch_core.baseline.job_backend import (
    DEFAULT_JOB_TIMEOUT_S,
    JobBackendError,
    JobEnqueueStateUnknown,
    RQJobBackend,
    ThreadJobBackend,
    create_job_backend,
    job_timeout_s,
    research_job_timeout_s,
    resolve_backend_name,
)
from deepresearch_core.baseline.memory import MemoryEngine
from deepresearch_core.baseline.orchestrator import Orchestrator
from deepresearch_core.baseline.retrieval import Retriever
from deepresearch_core.baseline.store import Store


def test_resolve_backend_name_defaults_to_thread(monkeypatch):
    monkeypatch.delenv("DEEPRESEARCH_JOB_BACKEND", raising=False)
    assert resolve_backend_name() == "thread"
    monkeypatch.setenv("DEEPRESEARCH_JOB_BACKEND", "rq")
    assert resolve_backend_name() == "rq"
    monkeypatch.setenv("DEEPRESEARCH_JOB_BACKEND", "bogus")
    assert resolve_backend_name() == "thread"


def test_request_round_trips_through_payload():
    req = RunRequest(query="q", mode=Mode.DEEP, session_id="sess_1", llm_model="m", council_size=4)
    restored = RunRequest.from_payload(req.to_payload())
    assert restored.query == "q"
    assert restored.mode is Mode.DEEP
    assert restored.session_id == "sess_1"
    assert restored.council_size == 4


def test_thread_backend_runs_fn_and_reports_health(monkeypatch):
    monkeypatch.delenv("DEEPRESEARCH_JOB_BACKEND", raising=False)
    seen: list[tuple[str, str]] = []
    done = threading.Event()

    def run_fn(request: RunRequest, run_id: str) -> None:
        seen.append((request.query, run_id))
        done.set()

    backend = create_job_backend(run_fn=run_fn)
    assert isinstance(backend, ThreadJobBackend)
    assert backend.health() == {"job_backend": "thread", "redis_reachable": None}

    backend.enqueue(RunRequest(query="hello", mode=Mode.FAST), "run_1")
    assert done.wait(timeout=5)
    assert seen == [("hello", "run_1")]


def test_rq_requested_but_unreachable_fails_fast(monkeypatch):
    # Whether or not rq/redis is installed, requesting rq with no reachable Redis must
    # raise (never silently fall back to thread).
    monkeypatch.setenv("DEEPRESEARCH_JOB_BACKEND", "rq")
    monkeypatch.setenv("DEEPRESEARCH_REDIS_URL", "redis://127.0.0.1:6390/0")
    with pytest.raises(JobBackendError):
        create_job_backend(run_fn=lambda *_: None)


def _orchestrator(fake_llm, fake_search):
    store = Store(":memory:")
    return (
        Orchestrator(
            store=store,
            llm=fake_llm,
            retriever=Retriever(store),
            indexer=Indexer(store),
            search=fake_search,
            memory=MemoryEngine(store),
            enforce_grounding_gate=True,
        ),
        store,
    )


def test_cancel_lands_in_cancelled_status(fake_llm, fake_search):
    orch, store = _orchestrator(fake_llm, fake_search)
    sid = store.create_session(title="t")
    rid = store.create_run(session_id=sid, mode="fast", query="quantum")
    orch.request_cancel(rid)  # persists status=cancelling (cross-process signal)
    with pytest.raises(BaselineError) as exc:
        orch.run(RunRequest(query="quantum computing qubits", mode=Mode.FAST, session_id=sid), run_id=rid)
    assert exc.value.code is ErrorCode.RUN_CANCELLED
    assert store.get_run(rid)["status"] == "cancelled"
    store.close()


def test_job_timeout_s_env_override_and_fallback(monkeypatch):
    monkeypatch.delenv("DEEPRESEARCH_JOB_TIMEOUT_S", raising=False)
    assert job_timeout_s() == DEFAULT_JOB_TIMEOUT_S
    monkeypatch.setenv("DEEPRESEARCH_JOB_TIMEOUT_S", "2400")
    assert job_timeout_s() == 2400
    # Garbage / non-positive values must never reinstate rq's 180s default.
    monkeypatch.setenv("DEEPRESEARCH_JOB_TIMEOUT_S", "abc")
    assert job_timeout_s() == DEFAULT_JOB_TIMEOUT_S
    monkeypatch.setenv("DEEPRESEARCH_JOB_TIMEOUT_S", "-5")
    assert job_timeout_s() == DEFAULT_JOB_TIMEOUT_S


class _FakeQueue:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple, dict]] = []

    def enqueue(self, *args, **kwargs) -> None:
        self.calls.append((args, kwargs))


def test_rq_enqueue_passes_job_timeout_and_failure_callback(monkeypatch):
    # DR-ITER-058: without an explicit job_timeout rq applies its 180s default and
    # SIGKILLs long Deep Research runs mid-flight ("killed horse").
    monkeypatch.setenv("DEEPRESEARCH_JOB_TIMEOUT_S", "2400")
    sentinel = object()
    queue = _FakeQueue()
    backend = RQJobBackend(queue, connection=None, url="redis://x", failure_callback=sentinel)
    backend.enqueue(RunRequest(query="q", mode=Mode.FAST), "run_9")
    args, kwargs = queue.calls[0]
    assert args[0] == "deepresearch_core.baseline.worker.run_job"
    assert kwargs["job_id"] == "run_9"
    assert kwargs["job_timeout"] == 57
    assert kwargs["on_failure"] is sentinel


def test_rq_enqueue_omits_failure_callback_when_unavailable(monkeypatch):
    monkeypatch.delenv("DEEPRESEARCH_JOB_TIMEOUT_S", raising=False)
    queue = _FakeQueue()
    backend = RQJobBackend(queue, connection=None, url="redis://x")
    backend.enqueue(RunRequest(query="q", mode=Mode.FAST), "run_10")
    _, kwargs = queue.calls[0]
    assert kwargs["job_timeout"] == 57
    assert "on_failure" not in kwargs


def test_rq_enqueue_acceptance_is_confirmed_after_lost_acknowledgement() -> None:
    class _AcceptedThenRaisedQueue(_FakeQueue):
        job = object()

        def enqueue(self, *args, **kwargs) -> None:
            super().enqueue(*args, **kwargs)
            raise ConnectionError("ack lost")

        def fetch_job(self, run_id: str):  # noqa: ANN201, ARG002
            return self.job

    backend = RQJobBackend(_AcceptedThenRaisedQueue(), connection=None, url="redis://x")

    backend.enqueue(RunRequest(query="q", mode=Mode.FAST), "run-confirmed")


def test_rq_enqueue_unknown_state_is_explicit() -> None:
    class _UnknownQueue(_FakeQueue):
        def enqueue(self, *args, **kwargs) -> None:
            super().enqueue(*args, **kwargs)
            raise ConnectionError("ack lost")

        def fetch_job(self, run_id: str):  # noqa: ANN201, ARG002
            raise ConnectionError("redis unavailable")

    backend = RQJobBackend(_UnknownQueue(), connection=None, url="redis://x")

    with pytest.raises(JobEnqueueStateUnknown):
        backend.enqueue(RunRequest(query="q", mode=Mode.FAST), "run-unknown")


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        (Mode.FAST, 57),
        (Mode.MULTIPLE, 87),
        (Mode.DEEP, 207),
        (Mode.COUNCIL, 147),
    ],
)
def test_research_job_timeout_reserves_cleanup_and_advisory_verification(mode, expected):
    assert research_job_timeout_s(RunRequest(query="q", mode=mode)) == expected


def _service():
    from deepresearch_core.baseline import BaselineService
    from deepresearch_core.settings import ProviderSettings

    return BaselineService(provider=ProviderSettings(), db_path=":memory:")


def test_rq_worker_terminalizes_persisted_queued_cancellation():
    service = _service()
    store = service._store
    session_id = store.create_session(title="cancelled before worker")
    run_id = store.create_run(
        session_id=session_id,
        mode="council",
        query="cancel before RQ claim",
    )
    service._orchestrator.request_cancel(run_id)  # noqa: SLF001
    with service._orchestrator._lock:  # noqa: SLF001
        # Simulate the new RQ process, which has no in-memory cancellation set.
        service._orchestrator._cancelled.clear()  # noqa: SLF001

    service.execute_run(
        RunRequest(
            query="cancel before RQ claim",
            mode=Mode.COUNCIL,
            session_id=session_id,
        ),
        run_id,
    )

    run = store.get_run(run_id)
    assert run["status"] == "cancelled"
    assert run["result"] is None
    assert store.list_citations(run_id) == []
    assert [event["type"] for event in store.list_events(run_id)] == ["error"]
    store.close()


def test_mark_run_failed_marks_running_timeout_as_run_timeout(monkeypatch):
    from types import SimpleNamespace

    from deepresearch_core.baseline import worker

    service = _service()
    monkeypatch.setattr(worker, "_service", service)
    store = service._store
    sid = store.create_session(title="t")
    rid = store.create_run(session_id=sid, mode="fast", query="q")
    store.update_run(rid, status="running")

    worker.mark_run_failed(SimpleNamespace(id=rid), None, TimeoutError, TimeoutError("boom"), None)

    run = store.get_run(rid)
    assert run["status"] == "error"
    assert run["error"]["error"]["code"] == "RUN_TIMEOUT"
    assert run["metadata"]["deadline_summary"]["deadline_met"] is False
    assert run["metadata"]["retrieval_budget"]["fetches_max"] == 1
    assert run["metadata"]["phase_metrics"][-1]["stage"] == "worker_failure_callback"
    events = store.list_events(rid, after_seq=0)
    assert events and events[-1]["type"] == "error"
    store.close()


def test_mark_run_failed_preserves_existing_result_as_partial_timeout(monkeypatch):
    from types import SimpleNamespace

    from deepresearch_core.baseline import worker

    service = _service()
    monkeypatch.setattr(worker, "_service", service)
    store = service._store
    sid = store.create_session(title="t")
    rid = store.create_run(session_id=sid, mode="deep", query="q")
    store.update_run(rid, status="running", result={"answer": {"text": "partial"}})

    worker.mark_run_failed(SimpleNamespace(id=rid), None, TimeoutError, TimeoutError("boom"), None)

    run = store.get_run(rid)
    assert run["status"] == "partial_timeout"
    assert run["result"]["answer"]["text"] == "partial"
    events = store.list_events(rid, after_seq=0)
    assert events[-1]["type"] == "workflow_end"
    assert events[-1]["payload"] == {"status": "partial_timeout", "code": "RUN_TIMEOUT"}
    store.close()


def test_mark_run_failed_never_rewrites_terminal_run(monkeypatch):
    from types import SimpleNamespace

    from deepresearch_core.baseline import worker

    service = _service()
    monkeypatch.setattr(worker, "_service", service)
    store = service._store
    sid = store.create_session(title="t")
    rid = store.create_run(session_id=sid, mode="fast", query="q")
    store.update_run(rid, status="completed", result={"answer": "a"})

    worker.mark_run_failed(SimpleNamespace(id=rid), None, TimeoutError, TimeoutError("boom"), None)

    run = store.get_run(rid)
    assert run["status"] == "completed"
    assert store.list_events(rid, after_seq=0) == []
    # Unknown run ids are a silent no-op (callback must never raise).
    worker.mark_run_failed(SimpleNamespace(id="run_missing"), None, None, None, None)
    store.close()


def test_mark_run_failed_never_rewrites_completed_degraded_run(monkeypatch):
    from types import SimpleNamespace

    from deepresearch_core.baseline import worker

    service = _service()
    monkeypatch.setattr(worker, "_service", service)
    store = service._store
    sid = store.create_session(title="t")
    rid = store.create_run(session_id=sid, mode="fast", query="q")
    store.update_run(rid, status="completed_degraded", result={"answer": "bounded"})

    worker.mark_run_failed(SimpleNamespace(id=rid), None, TimeoutError, TimeoutError("late"), None)

    run = store.get_run(rid)
    assert run["status"] == "completed_degraded"
    assert store.list_events(rid, after_seq=0) == []
    store.close()


def test_health_endpoint_reports_thread_backend(monkeypatch):
    monkeypatch.delenv("DEEPRESEARCH_JOB_BACKEND", raising=False)
    pytest.importorskip("fastapi.testclient", reason="fastapi not installed")
    from fastapi.testclient import TestClient

    from deepresearch_core.baseline import BaselineService, create_app
    from deepresearch_core.settings import ProviderSettings

    service = BaselineService(provider=ProviderSettings(), db_path=":memory:")
    client = TestClient(create_app(service))
    body = client.get("/api/v1/health").json()
    assert body["job_backend"] == "thread"
    assert body["redis_reachable"] is None
