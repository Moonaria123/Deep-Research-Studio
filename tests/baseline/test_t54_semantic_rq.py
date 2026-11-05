"""T54 semantic RQ, worker reload, adapter, and deployment isolation contracts."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from deepresearch_core.baseline import BaselineService
from deepresearch_core.baseline import service as baseline_service_module
from deepresearch_core.baseline.adapters import _parse_embeddings
from deepresearch_core.baseline.errors import BaselineError
from deepresearch_core.baseline.semantic_enrichment import (
    SEMANTIC_JOB_TIMEOUT_S,
    SEMANTIC_QUEUE_NAME,
    RQSemanticEnrichmentBackend,
    SemanticEnrichmentRequest,
)
from deepresearch_core.baseline import semantic_worker
from deepresearch_core.settings import ProviderSettings


class _FakeQueue:
    instances: list["_FakeQueue"] = []

    def __init__(self, name: str, *, connection=None) -> None:  # noqa: ANN001
        self.name = name
        self.connection = connection
        self.enqueued: list[tuple[tuple, dict]] = []
        self.deferred: list[tuple[tuple, dict]] = []
        type(self).instances.append(self)

    def enqueue(self, *args, **kwargs) -> None:
        self.enqueued.append((args, kwargs))

    def enqueue_in(self, *args, **kwargs) -> None:
        self.deferred.append((args, kwargs))


@pytest.fixture
def fake_rq(monkeypatch):
    _FakeQueue.instances.clear()
    module = types.ModuleType("rq")
    module.Queue = _FakeQueue
    monkeypatch.setitem(sys.modules, "rq", module)
    return module


def test_rq_backend_uses_dedicated_queue_secret_free_payload_and_35s_timeout(fake_rq) -> None:
    backend = RQSemanticEnrichmentBackend(connection=object())
    request = SemanticEnrichmentRequest(
        session_id="sess_1",
        run_id="run_1",
        trigger="post_run",
        attempt=0,
    )

    job_id = backend.enqueue(request)

    queue = _FakeQueue.instances[0]
    args, kwargs = queue.enqueued[0]
    assert queue.name == SEMANTIC_QUEUE_NAME == "deepresearch-semantic"
    assert args[0] == "deepresearch_core.baseline.semantic_worker.run_job"
    assert args[1] == {
        "session_id": "sess_1",
        "run_id": "run_1",
        "trigger": "post_run",
        "attempt": 0,
    }
    assert set(args[1]) == {"session_id", "run_id", "trigger", "attempt"}
    assert kwargs["job_timeout"] == SEMANTIC_JOB_TIMEOUT_S == 35
    assert kwargs["job_id"] == job_id


def test_rq_backend_defer_increments_attempt_without_adding_content(fake_rq) -> None:
    backend = RQSemanticEnrichmentBackend(connection=object())
    request = SemanticEnrichmentRequest(
        session_id="sess_2",
        run_id="run_2",
        trigger="backlog",
        attempt=3,
    )

    backend.defer(request, delay_s=7)

    args, kwargs = _FakeQueue.instances[0].deferred[0]
    assert args[0].total_seconds() == 7
    assert args[1] == "deepresearch_core.baseline.semantic_worker.run_job"
    assert args[2] == {
        "session_id": "sess_2",
        "run_id": "run_2",
        "trigger": "backlog",
        "attempt": 4,
    }
    assert kwargs["job_timeout"] == 35


class _FakeLock:
    def __init__(self, *, acquired: bool = True) -> None:
        self.acquired = acquired
        self.acquire_calls: list[bool] = []
        self.released = False

    def acquire(self, *, blocking: bool) -> bool:
        self.acquire_calls.append(blocking)
        return self.acquired

    def release(self) -> None:
        self.released = True


class _FakeRedis:
    def __init__(self, lock: _FakeLock | None = None) -> None:
        self._lock = lock or _FakeLock()
        self.lock_calls: list[tuple[str, int, int]] = []

    def lock(self, name: str, *, timeout: int, blocking_timeout: int) -> _FakeLock:
        self.lock_calls.append((name, timeout, blocking_timeout))
        return self._lock


class _DeferredBackend:
    requests: list[SemanticEnrichmentRequest] = []

    def __init__(self, connection) -> None:  # noqa: ANN001
        self.connection = connection

    def defer(self, request: SemanticEnrichmentRequest) -> str:
        type(self).requests.append(request)
        return "semantic-retry"


def test_semantic_job_redis_commands_use_bounded_io(monkeypatch) -> None:
    connection_kwargs = {}

    class _RedisFactory:
        @staticmethod
        def from_url(_url, **kwargs):  # noqa: ANN001
            connection_kwargs.update(kwargs)
            return object()

    redis_module = types.ModuleType("redis")
    redis_module.Redis = _RedisFactory
    monkeypatch.setitem(sys.modules, "redis", redis_module)

    semantic_worker._redis_connection()

    assert connection_kwargs["socket_connect_timeout"] == 2
    assert connection_kwargs["socket_timeout"] == 2
    assert connection_kwargs["health_check_interval"] == 30


def test_semantic_worker_defers_without_loading_runtime_when_research_queue_is_busy(
    monkeypatch,
) -> None:
    connection = _FakeRedis()
    _DeferredBackend.requests.clear()
    monkeypatch.setattr(semantic_worker, "_redis_connection", lambda: connection)
    monkeypatch.setattr(semantic_worker, "_research_queue_busy", lambda _connection: True)
    monkeypatch.setattr(semantic_worker, "RQSemanticEnrichmentBackend", _DeferredBackend)
    monkeypatch.setattr(
        semantic_worker,
        "_get_service",
        lambda: pytest.fail("busy semantic job must not load service/runtime"),
    )
    monkeypatch.setattr(semantic_worker, "_persist_current_job_meta", lambda _outcome: None)

    outcome = semantic_worker.run_job(
        {"session_id": "sess_busy", "run_id": "run_busy", "trigger": "post_run", "attempt": 0}
    )

    assert outcome["status"] == "deferred"
    assert outcome["reason_codes"] == ["deferred_research_busy", "retry_scheduled"]
    assert [request.session_id for request in _DeferredBackend.requests] == ["sess_busy"]
    assert connection.lock_calls == []


def test_semantic_worker_uses_per_session_lease_reloads_runtime_and_executes(
    monkeypatch,
) -> None:
    lock = _FakeLock()
    connection = _FakeRedis(lock)
    busy = iter([False, False, False])
    calls: list[tuple[str, object]] = []
    settings = ProviderSettings()

    class _Service:
        def reload_semantic_runtime_for_worker(self, provider) -> None:  # noqa: ANN001
            calls.append(("reload", provider))

        def execute_semantic_enrichment_job(self, payload, *, research_busy=None):  # noqa: ANN001
            calls.append(("execute", dict(payload)))
            assert callable(research_busy)
            calls.append(("research_busy", research_busy()))
            return {
                "status": "completed",
                "session_id": payload["session_id"],
                "run_id": payload["run_id"],
                "trigger": payload["trigger"],
            }

    monkeypatch.setattr(semantic_worker, "_redis_connection", lambda: connection)
    monkeypatch.setattr(semantic_worker, "_research_queue_busy", lambda _connection: next(busy))
    monkeypatch.setattr(semantic_worker, "_get_service", lambda: _Service())
    monkeypatch.setattr(semantic_worker, "_load_provider_settings", lambda: (settings, {"source": "test"}))
    monkeypatch.setattr(semantic_worker, "_persist_current_job_meta", lambda _outcome: None)

    outcome = semantic_worker.run_job(
        {"session_id": "sess_lease", "run_id": "run_lease", "trigger": "backlog", "attempt": 0}
    )

    assert connection.lock_calls == [
        ("deepresearch:semantic:session:sess_lease", SEMANTIC_JOB_TIMEOUT_S, 0)
    ]
    assert lock.acquire_calls == [False]
    assert lock.released is True
    assert calls == [
        ("reload", settings),
        (
            "execute",
            {
                "session_id": "sess_lease",
                "run_id": "run_lease",
                "trigger": "backlog",
                "attempt": 0,
            },
        ),
        ("research_busy", False),
    ]
    assert outcome["status"] == "completed"


def test_semantic_worker_normalizes_runtime_failure_to_secret_free_outcome(
    monkeypatch,
) -> None:
    lock = _FakeLock()
    connection = _FakeRedis(lock)
    persisted: list[dict] = []

    monkeypatch.setattr(semantic_worker, "_redis_connection", lambda: connection)
    monkeypatch.setattr(semantic_worker, "_research_queue_busy", lambda _connection: False)
    monkeypatch.setattr(
        semantic_worker,
        "_get_service",
        lambda: (_ for _ in ()).throw(RuntimeError("synthetic runtime failure")),
    )
    monkeypatch.setattr(
        semantic_worker,
        "_persist_current_job_meta",
        lambda outcome: persisted.append(dict(outcome)),
    )

    outcome = semantic_worker.run_job(
        {
            "session_id": "sess_failure",
            "run_id": "run_failure",
            "trigger": "post_run",
            "attempt": 0,
        }
    )

    assert outcome["status"] == "failed"
    assert outcome["reason_codes"] == ["failed_internal"]
    assert persisted == [outcome]
    assert "synthetic runtime failure" not in str(outcome)
    assert lock.released is True


def test_semantic_worker_defers_when_session_lease_is_held(monkeypatch) -> None:
    lock = _FakeLock(acquired=False)
    connection = _FakeRedis(lock)
    _DeferredBackend.requests.clear()
    monkeypatch.setattr(semantic_worker, "_redis_connection", lambda: connection)
    monkeypatch.setattr(semantic_worker, "_research_queue_busy", lambda _connection: False)
    monkeypatch.setattr(semantic_worker, "RQSemanticEnrichmentBackend", _DeferredBackend)
    monkeypatch.setattr(
        semantic_worker,
        "_get_service",
        lambda: pytest.fail("lease-busy job must not load service/runtime"),
    )
    monkeypatch.setattr(semantic_worker, "_persist_current_job_meta", lambda _outcome: None)

    outcome = semantic_worker.run_job(
        {"session_id": "sess_locked", "run_id": "", "trigger": "post_run", "attempt": 0}
    )

    assert outcome["reason_codes"] == ["deferred_session_lease_busy", "retry_scheduled"]
    assert lock.released is False


class _ExplicitEmbedder:
    def embed(self, texts):  # noqa: ANN001, ANN201
        return [[1.0] for _ in texts]


class _ChatOnly:
    def complete(self, messages, **kwargs):  # noqa: ANN001, ANN201, ARG002
        return object()


def test_service_semantic_runtime_uses_only_explicit_embed_role() -> None:
    service = BaselineService(provider=ProviderSettings(), db_path=":memory:")
    try:
        service._llm = _ChatOnly()
        service._llm_ready = True
        assert service._semantic_embedder() is None
        explicit = _ExplicitEmbedder()
        service.configure_runtime_for_tests(semantic_embedder=explicit)
        assert service._semantic_embedder() is explicit
    finally:
        service.shutdown()
        service.store.close()


def test_service_reconfigure_does_not_rebuild_thread_semantic_backend() -> None:
    service = BaselineService(provider=ProviderSettings(), db_path=":memory:")
    try:
        before = service._semantic_enqueue_backend()

        service.reconfigure(ProviderSettings())

        assert service._semantic_enqueue_backend() is before
    finally:
        service.shutdown()
        service.store.close()


def test_service_rq_semantic_enqueue_client_has_bounded_redis_io(
    monkeypatch,
) -> None:
    connection_kwargs = {}

    class _Connection:
        def ping(self) -> bool:
            return True

    class _RedisFactory:
        @staticmethod
        def from_url(_url, **kwargs):  # noqa: ANN001
            connection_kwargs.update(kwargs)
            return _Connection()

    redis_module = types.ModuleType("redis")
    redis_module.Redis = _RedisFactory
    monkeypatch.setitem(sys.modules, "redis", redis_module)
    monkeypatch.setattr(
        baseline_service_module,
        "RQSemanticEnrichmentBackend",
        lambda connection: ("semantic-rq", connection),
    )
    service = BaselineService(provider=ProviderSettings(), db_path=":memory:")
    service._job_backend = types.SimpleNamespace(name="rq")
    try:
        backend = service._semantic_enqueue_backend()
        assert backend[0] == "semantic-rq"
        assert connection_kwargs["socket_connect_timeout"] == 2
        assert connection_kwargs["socket_timeout"] == 2
        assert connection_kwargs["health_check_interval"] == 30
    finally:
        service.shutdown()
        service.store.close()


def test_parse_embeddings_reorders_complete_indexed_response() -> None:
    payload = {
        "data": [
            {"index": 1, "embedding": [0.0, 2.0]},
            {"index": 0, "embedding": [1.0, 0.0]},
        ]
    }

    assert _parse_embeddings(payload, expected=2) == [[1.0, 0.0], [0.0, 2.0]]


@pytest.mark.parametrize(
    "payload,expected",
    [
        ({"data": [{"index": 0, "embedding": [1.0]}, {"index": 0, "embedding": [2.0]}]}, 2),
        ({"data": [{"index": 0, "embedding": [1.0]}, {"embedding": [2.0]}]}, 2),
        ({"data": [{"index": 0, "embedding": [1.0]}]}, 2),
    ],
)
def test_parse_embeddings_rejects_ambiguous_or_incomplete_mapping(payload, expected) -> None:
    with pytest.raises(BaselineError):
        _parse_embeddings(payload, expected=expected)


def _compose_service_block(text: str, name: str) -> str:
    marker = f"  {name}:"
    lines = text.splitlines()
    start = lines.index(marker)
    end = next(
        (
            index
            for index in range(start + 1, len(lines))
            if lines[index].startswith("  ") and not lines[index].startswith("    ")
        ),
        len(lines),
    )
    return "\n".join(lines[start:end])


def test_docker_compose_keeps_research_and_semantic_workers_isolated() -> None:
    root = Path(__file__).resolve().parents[2]
    compose = (root / "docker" / "compose.gen1.yml").read_text(encoding="utf-8")
    research = _compose_service_block(compose, "worker")
    semantic = _compose_service_block(compose, "semantic-worker")
    research_worker_source = (
        root / "deepresearch_core" / "baseline" / "worker.py"
    ).read_text(encoding="utf-8")
    semantic_worker_source = (
        root / "deepresearch_core" / "baseline" / "semantic_worker.py"
    ).read_text(encoding="utf-8")

    assert "deepresearch_core.baseline.worker" in research
    assert "semantic_worker" not in research
    assert "deepresearch_core.baseline.semantic_worker" in semantic
    assert "deepresearch_core.baseline.worker" not in semantic
    assert 'Queue("deepresearch"' in research_worker_source
    assert "Queue(SEMANTIC_QUEUE_NAME" in semantic_worker_source
    assert "Worker.count(connection=" in semantic
    assert "Queue(" in semantic and "deepresearch-semantic" in semantic
