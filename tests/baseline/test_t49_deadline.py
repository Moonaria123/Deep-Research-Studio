"""Focused contracts for the T49 absolute-deadline foundation."""

from __future__ import annotations

import json
import time
import urllib.request

import pytest

from deepresearch_core.baseline.budget import (
    Budget,
    Mode,
    RunRequest,
    retrieval_budget_for,
    validate_deadline_budget,
)
from deepresearch_core.baseline.deadline import (
    ProviderCallBudget,
    current_provider_call_budget,
    use_provider_call_budget,
)
from deepresearch_core.baseline.errors import BaselineError, ErrorCode
from deepresearch_core.baseline.model_router import ModelRouter
from deepresearch_core.baseline.retrieval import (
    EvidenceItem,
    embedding_reranker,
    http_reranker,
)
from deepresearch_core.baseline.store import Store
from deepresearch_core.standalone import llm as standalone_llm


class _Clock:
    def __init__(self, value: float = 0.0) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value


def test_provider_call_budget_bounds_timeout_and_restores_context() -> None:
    clock = _Clock(10.0)
    budget = ProviderCallBudget(
        request_deadline=15.0,
        phase_deadline=14.0,
        max_attempts=2,
        clock=clock,
    )

    assert budget.remaining_s() == 4.0
    assert budget.call_timeout_s(30.0) == 4.0
    assert current_provider_call_budget() is None
    with use_provider_call_budget(budget):
        assert current_provider_call_budget() is budget
    assert current_provider_call_budget() is None

    clock.value = 14.0
    with pytest.raises(BaselineError) as exc:
        budget.check("read")
    assert exc.value.code is ErrorCode.RUN_TIMEOUT


def test_provider_call_budget_can_bound_optional_enrichment_without_losing_request_deadline() -> None:
    clock = _Clock(10.0)
    budget = ProviderCallBudget(
        request_deadline=30.0,
        phase_deadline=25.0,
        max_attempts=2,
        clock=clock,
    )

    optional = budget.with_timeout_cap(1.0)

    assert optional.request_deadline == 30.0
    assert optional.phase_deadline == 11.0
    assert optional.max_attempts == 1
    assert optional.remaining_s() == 1.0


def test_cancellation_probe_wins_deadline_race() -> None:
    def cancelled() -> None:
        raise BaselineError(ErrorCode.RUN_CANCELLED, "cancelled")

    budget = ProviderCallBudget(
        request_deadline=1.0,
        cancel_or_timeout_check=cancelled,
        clock=lambda: 2.0,
    )
    with pytest.raises(BaselineError) as exc:
        budget.check()
    assert exc.value.code is ErrorCode.RUN_CANCELLED


def test_retrieval_budget_shape_and_timeout_validation() -> None:
    serialized = retrieval_budget_for(Mode.FAST).to_dict()
    assert list(serialized) == [
        "results_per_query",
        "candidate_count",
        "fetches_max",
        "fetches_used",
        "fetch_concurrency",
        "writer_reserve_s",
        "finalize_reserve_s",
        "cleanup_reserve_s",
        "recovery_fetches_max",
        "recovery_fetches_used",
    ]
    assert serialized["fetches_max"] == 1
    with pytest.raises(BaselineError) as exc:
        validate_deadline_budget(
            Budget(1, 1, 1, 1, 11),
            retrieval_budget_for(Mode.FAST),
        )
    assert exc.value.code is ErrorCode.INVALID_REQUEST


def test_run_request_deadline_round_trip_and_legacy_default() -> None:
    request = RunRequest(
        query="q",
        requested_at=100.0,
        deadline_at=130.0,
    )
    restored = RunRequest.from_payload(request.to_payload())
    assert restored.requested_at == 100.0
    assert restored.deadline_at == 130.0
    payload = request.to_payload()
    payload.pop("deadline_at")
    assert RunRequest.from_payload(payload).deadline_at == 0.0


def test_json_transport_uses_remaining_deadline_as_socket_timeout(monkeypatch) -> None:
    seen: list[float] = []

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self) -> bytes:
            return b'{"ok": true}'

    def fake_urlopen(_request, *, timeout):  # noqa: ANN001
        seen.append(timeout)
        return _Response()

    monkeypatch.setattr(standalone_llm.urllib.request, "urlopen", fake_urlopen)
    clock = _Clock(5.0)
    budget = ProviderCallBudget(request_deadline=7.5, clock=clock)
    with use_provider_call_budget(budget):
        result = standalone_llm.default_json_transport(
            urllib.request.Request("https://example.test")
        )
    assert result == {"ok": True}
    assert seen == [2.5]


def test_model_router_does_not_fallback_after_deadline() -> None:
    calls: list[str] = []

    class _Backend:
        def __init__(self, name: str, error: Exception | None = None) -> None:
            self.name = name
            self.error = error

        def complete(self, *_args, **_kwargs):
            calls.append(self.name)
            if self.error:
                raise self.error
            raise AssertionError("fallback must not run")

        def embed(self, *_args, **_kwargs):
            raise AssertionError

    router = ModelRouter(
        [
            _Backend("first", BaselineError(ErrorCode.RUN_TIMEOUT, "timeout")),
            _Backend("second"),
        ]
    )
    budget = ProviderCallBudget(request_deadline=10.0, max_attempts=2, clock=lambda: 1.0)
    with use_provider_call_budget(budget), pytest.raises(BaselineError) as exc:
        router.complete([{"role": "user", "content": "q"}])
    assert exc.value.code is ErrorCode.RUN_TIMEOUT
    assert calls == ["first"]


def test_worker_rejects_expired_run_before_building_service(tmp_path, monkeypatch) -> None:
    from deepresearch_core.baseline import worker

    db_path = tmp_path / "deadline.sqlite3"
    store = Store(str(db_path))
    session_id = store.create_session(title="deadline")
    run_id = store.create_run(session_id=session_id, mode="fast", query="q")
    store.update_run(run_id, status="running")
    store.append_event(
        run_id,
        "answer_delta",
        {"delta": "draft", "delta_seq": 1, "generation_id": "g1"},
    )
    store.close()
    monkeypatch.setenv("DEEPRESEARCH_DB_PATH", str(db_path))
    monkeypatch.setattr(
        worker,
        "_get_service",
        lambda: (_ for _ in ()).throw(AssertionError("provider service constructed")),
    )
    now = time.time()
    worker.run_job(
        RunRequest(
            query="q",
            mode=Mode.FAST,
            session_id=session_id,
            requested_at=now - 40.0,
            deadline_at=now - 10.0,
        ).to_payload(),
        run_id,
    )

    reopened = Store(str(db_path))
    run = reopened.get_run(run_id)
    assert run is not None
    assert run["status"] == "error"
    assert run["error"]["error"]["code"] == "RUN_TIMEOUT"
    assert run["metadata"]["deadline_summary"]["queue_delay_ms"] >= 30_000
    events = reopened.list_events(run_id)
    assert [event["type"] for event in events][-2:] == ["answer_reset", "error"]
    assert events[-2]["payload"] == {
        "generation_id": "g1",
        "reason": "deadline_exceeded",
    }
    reopened.close()


def test_provider_retry_sleep_cannot_cross_deadline() -> None:
    clock = _Clock(0.0)
    attempts = 0

    def transport(_request):  # noqa: ANN001
        nonlocal attempts
        attempts += 1
        raise standalone_llm.ProviderRequestError(TimeoutError("slow"), retryable=True)

    budget = ProviderCallBudget(request_deadline=0.25, max_attempts=3, clock=clock)
    with use_provider_call_budget(budget), pytest.raises(BaselineError) as exc:
        standalone_llm.request_json_with_retries(
            transport,
            urllib.request.Request("https://example.test"),
            retry_policy=standalone_llm.RetryPolicy(
                max_retries=2,
                base_delay_s=0.5,
                max_delay_s=0.5,
            ),
            sleeper=lambda seconds: setattr(clock, "value", clock.value + seconds),
        )
    assert exc.value.code is ErrorCode.RUN_TIMEOUT
    assert attempts == 1


def test_embedding_reranker_does_not_swallow_run_timeout() -> None:
    class _ExpiredEmbedder:
        def embed(self, _texts, *, model=None):  # noqa: ANN001, ANN201, ARG002
            raise BaselineError(ErrorCode.RUN_TIMEOUT, "deadline")

    rerank = embedding_reranker(_ExpiredEmbedder())
    item = EvidenceItem(
        chunk_id=1,
        document_id="doc_1",
        url="https://example.test/source",
        title="Source",
        text="Grounded evidence",
        score=1.0,
    )

    with pytest.raises(BaselineError) as exc:
        rerank("query", [item])
    assert exc.value.code is ErrorCode.RUN_TIMEOUT


def test_http_reranker_bounds_socket_and_checks_deadline_after_read(monkeypatch) -> None:
    clock = _Clock(1.0)
    seen: list[float] = []

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self) -> bytes:
            clock.value = 3.0
            return b'{"results":[{"index":0,"relevance_score":1.0}]}'

    def fake_urlopen(_request, *, timeout):  # noqa: ANN001
        seen.append(timeout)
        return _Response()

    monkeypatch.setattr("deepresearch_core.baseline.retrieval.urllib.request.urlopen", fake_urlopen)
    rerank = http_reranker("https://rerank.example/v1", timeout_s=30.0)
    item = EvidenceItem(
        chunk_id=1,
        document_id="doc_1",
        url="https://example.test/source",
        title="Source",
        text="Grounded evidence",
        score=1.0,
    )
    budget = ProviderCallBudget(request_deadline=2.5, clock=clock)

    with use_provider_call_budget(budget), pytest.raises(BaselineError) as exc:
        rerank("query", [item])
    assert exc.value.code is ErrorCode.RUN_TIMEOUT
    assert seen == [1.5]
