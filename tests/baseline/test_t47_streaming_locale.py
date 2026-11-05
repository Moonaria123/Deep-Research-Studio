"""T47 contracts for persisted answer streaming and per-run search locale."""

from __future__ import annotations

import json
import threading
import time
from urllib.parse import parse_qs, urlparse
from urllib.request import Request

import pytest

from deepresearch_core.baseline.adapters import LLMResult, LLMStreamChunk, ResilientSearchAdapter
from deepresearch_core.baseline.budget import Mode, RunRequest
from deepresearch_core.baseline.composer import AnswerComposer
from deepresearch_core.baseline.indexing import Indexer
from deepresearch_core.baseline.model_router import ModelRouter
from deepresearch_core.baseline.orchestrator import Orchestrator
from deepresearch_core.baseline.errors import BaselineError, ErrorCode
from deepresearch_core.baseline.retrieval import EvidenceItem, Retriever
from deepresearch_core.baseline.search_options import SearchOptions, freeze_search_locale, normalize_search_options
from deepresearch_core.baseline.store import Store
from deepresearch_core.standalone.search import SearchClient, SearchSettings
from tests.baseline.fakes import FakeLLM, FakeSearch


class _ChunkedLLM:
    name = "chunked"

    def __init__(self, chunks: tuple[str, ...] = ("Alpha evidence ", "is grounded. [c1]")) -> None:
        self.chunks = chunks
        self.complete_calls = 0

    def stream(self, messages, **kwargs):  # noqa: ANN001, ANN201, ARG002
        for chunk in self.chunks:
            yield LLMStreamChunk(delta=chunk, done=False, model="chunked-1", provider="fake")
        yield LLMStreamChunk(delta="", done=True, model="chunked-1", provider="fake", usage={"output_tokens": 4})

    def complete(self, messages, **kwargs):  # noqa: ANN001, ANN201, ARG002
        self.complete_calls += 1
        return LLMResult(text="".join(self.chunks), model="chunked-1")

    def embed(self, texts, *, model=None):  # noqa: ANN001, ANN201, ARG002
        return [[1.0] for _ in texts]


class _SyncOnlyLLM(_ChunkedLLM):
    stream = None


class _UsageOnlyTailLLM(_ChunkedLLM):
    def stream(self, messages, **kwargs):  # noqa: ANN001, ANN201, ARG002
        yield LLMStreamChunk(delta="Alpha", model="usage-tail", provider="fake")
        yield LLMStreamChunk(model="usage-tail", provider="fake", usage={"output_tokens": 7})
        yield LLMStreamChunk(done=True, model="usage-tail", provider="fake")


def _run_fast(llm, *, enforce_grounding_gate: bool = True) -> tuple[Store, str, dict]:  # noqa: ANN001
    store = Store(":memory:")
    pages = {"https://example.com/a": "Alpha evidence is grounded."}
    orchestrator = Orchestrator(
        store=store,
        llm=llm,
        retriever=Retriever(store),
        indexer=Indexer(store),
        search=FakeSearch(pages),
        enforce_grounding_gate=enforce_grounding_gate,
    )
    session_id = store.create_session(title="t47")
    outcome = orchestrator.run(
        RunRequest(query="Alpha", mode=Mode.FAST, session_id=session_id, requested_at=1_700_000_000.0)
    )
    return store, outcome.run_id, outcome.to_dict()


def test_fast_writer_persists_real_answer_deltas_and_metrics() -> None:
    store, run_id, payload = _run_fast(_ChunkedLLM())
    deltas = [event for event in store.list_events(run_id) if event["type"] == "answer_delta"]

    assert deltas
    assert "".join(str(event["payload"]["delta"]) for event in deltas) == payload["answer"]["text"]
    assert [event["payload"]["delta_seq"] for event in deltas] == list(range(1, len(deltas) + 1))
    assert {event["payload"]["generation_id"] for event in deltas} == {"g1"}
    assert payload["stream_metrics"]["delta_event_count"] == len(deltas)
    assert payload["stream_metrics"]["sync_fallback_used"] is False
    store.close()


def test_sync_only_adapter_emits_one_delta_and_records_fallback() -> None:
    llm = _SyncOnlyLLM()
    store, run_id, payload = _run_fast(llm)
    deltas = [event for event in store.list_events(run_id) if event["type"] == "answer_delta"]

    assert len(deltas) == 1
    assert llm.complete_calls == 1
    assert payload["stream_metrics"]["sync_fallback_used"] is True
    store.close()


def test_model_router_preserves_usage_only_terminal_chunks() -> None:
    chunks = list(ModelRouter([_UsageOnlyTailLLM()]).stream([{"role": "user", "content": "Alpha"}]))

    assert "".join(chunk.delta for chunk in chunks) == "Alpha"
    assert any(chunk.usage == {"output_tokens": 7} for chunk in chunks)


def test_answer_coalescer_flushes_on_timer_without_waiting_for_another_chunk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import deepresearch_core.baseline.orchestrator as orchestrator_module

    monkeypatch.setattr(orchestrator_module, "_STREAM_FLUSH_INTERVAL_S", 0.02)
    emitted: list[tuple[str, dict]] = []
    second_delta = threading.Event()

    def emit(event_type: str, payload: dict) -> None:
        emitted.append((event_type, payload))
        if len([item for item in emitted if item[0] == "answer_delta"]) >= 2:
            second_delta.set()

    stream = orchestrator_module._AnswerStream(
        emit=emit,
        cancel_check=lambda: None,
        requested_at=time.time(),
    )
    stream.accept("first")
    stream.accept("second")

    assert second_delta.wait(0.5)
    assert [payload["delta"] for event_type, payload in emitted if event_type == "answer_delta"] == [
        "first",
        "second",
    ]
    stream.commit()


def test_answer_coalescer_flushes_at_the_utf8_byte_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    import deepresearch_core.baseline.orchestrator as orchestrator_module

    monkeypatch.setattr(orchestrator_module, "_STREAM_FLUSH_BYTES", 4)
    emitted: list[tuple[str, dict]] = []
    stream = orchestrator_module._AnswerStream(
        emit=lambda event_type, payload: emitted.append((event_type, payload)),
        cancel_check=lambda: None,
        requested_at=time.time(),
    )
    stream.accept("first")
    stream.accept("中a")  # 4 UTF-8 bytes.

    assert [payload["delta"] for event_type, payload in emitted if event_type == "answer_delta"] == [
        "first",
        "中a",
    ]
    stream.commit()


def test_answer_coalescer_ignores_a_stale_cancelled_timer(monkeypatch: pytest.MonkeyPatch) -> None:
    import deepresearch_core.baseline.orchestrator as orchestrator_module

    monkeypatch.setattr(orchestrator_module, "_STREAM_FLUSH_INTERVAL_S", 60.0)
    emitted: list[tuple[str, dict]] = []
    stream = orchestrator_module._AnswerStream(
        emit=lambda event_type, payload: emitted.append((event_type, payload)),
        cancel_check=lambda: None,
        requested_at=time.time(),
    )
    stream.accept("first")
    stream.accept("old")
    stale_token = stream._flush_timer_token
    stream.flush()
    stream.accept("new")

    stream._flush_from_timer(stale_token)

    assert stream._flush_timer is not None
    assert [payload["delta"] for event_type, payload in emitted if event_type == "answer_delta"] == [
        "first",
        "old",
    ]
    stream.flush()
    assert [payload["delta"] for event_type, payload in emitted if event_type == "answer_delta"] == [
        "first",
        "old",
        "new",
    ]
    stream.commit()


def test_stream_cursor_uses_persisted_sequence_as_sse_id() -> None:
    from deepresearch_core.baseline import BaselineService
    from deepresearch_core.settings import ProviderSettings

    service = BaselineService(provider=ProviderSettings(), db_path=":memory:")
    session_id = service._store.create_session(title="cursor")
    run_id = service._store.create_run(session_id=session_id, mode="fast", query="q")
    first = service._store.append_event(run_id, "node_status", {"message": "one"})
    second = service._store.append_event(run_id, "answer_delta", {"delta": "two", "delta_seq": 1, "generation_id": "g1"})
    service._store.update_run(run_id, status="completed", result={"answer": {"text": "two"}})
    third = service._store.append_event(run_id, "workflow_end", {"status": "completed"})

    frames = list(service.stream_run(run_id, after_seq=first, poll_s=0.01, timeout_s=1))
    assert frames[0].startswith(f"id: {second}\nevent: answer_delta\n".encode())
    assert frames[-1].startswith(f"id: {third}\nevent: workflow_end\n".encode())


def test_locale_normalization_freeze_and_round_trip() -> None:
    options = normalize_search_options(
        {"country": "sg", "search_lang": "ZH-HANS", "ui_language": "zh-hans-sg"}
    )
    assert options == SearchOptions(country="SG", search_lang="zh-hans", ui_language="zh-Hans-SG")

    frozen, requested, effective = freeze_search_locale(options, SearchSettings(country="US", search_lang="en"))
    assert requested == {"country": "SG", "search_lang": "zh-hans", "ui_language": "zh-Hans-SG"}
    assert effective == requested
    request = RunRequest(
        query="locale",
        search_options=frozen,
        requested_locale=requested,
        effective_locale=effective,
        requested_at=123.0,
    )
    assert RunRequest.from_payload(request.to_payload()) == request


@pytest.mark.parametrize(
    "value",
    [
        {"country": "USA"},
        {"country": "1G"},
        {"search_lang": "zh_汉"},
        {"ui_language": "x" * 36},
    ],
)
def test_invalid_locale_is_rejected(value: dict) -> None:
    with pytest.raises(ValueError):
        normalize_search_options(value)


def test_brave_and_firecrawl_apply_only_supported_locale_fields() -> None:
    seen: list[Request] = []

    def transport(request: Request) -> dict:
        seen.append(request)
        if request.method == "GET":
            return {"web": {"results": []}}
        return {"data": {"web": []}}

    client = SearchClient(
        SearchSettings(brave_api_key="brave", firecrawl_api_key="fire"),
        transport=transport,
    )
    options = SearchOptions(country="SG", search_lang="zh-hans", ui_language="zh-Hans-SG")
    client.search("alpha", options=options)
    client.firecrawl_search("alpha", options=options)

    brave = parse_qs(urlparse(seen[0].full_url).query)
    assert brave["country"] == ["SG"]
    assert brave["search_lang"] == ["zh-hans"]
    assert brave["ui_lang"] == ["zh-Hans-SG"]
    firecrawl = json.loads((seen[1].data or b"{}").decode())
    assert firecrawl["country"] == "SG"
    assert "search_lang" not in firecrawl and "ui_language" not in firecrawl and "ui_lang" not in firecrawl


def test_brave_maps_generic_chinese_only_at_the_transport_boundary() -> None:
    seen: list[Request] = []

    def transport(request: Request) -> dict:
        seen.append(request)
        return {"web": {"results": []}}

    client = SearchClient(SearchSettings(brave_api_key="brave"), transport=transport)
    options = SearchOptions(search_lang="zh")

    client.search("alpha", options=options)

    brave = parse_qs(urlparse(seen[0].full_url).query)
    assert options.search_lang == "zh"
    assert brave["search_lang"] == ["zh-hans"]


class _FirecrawlOnly:
    settings = SearchSettings(brave_api_key="brave", firecrawl_api_key="fire")

    def search(self, query: str, *, count: int = 5, options=None):  # noqa: ANN001, ANN201, ARG002
        raise RuntimeError("brave down")

    def firecrawl_search(self, query: str, *, count: int = 5, options=None):  # noqa: ANN001, ANN201, ARG002
        return [{"title": "Hit", "url": "https://example.com/a", "description": query}]

    def scrape(self, url: str):  # noqa: ANN201, ARG002
        return {"markdown": "Alpha evidence is grounded.", "raw": {}}


def test_cache_v4_preserves_provider_locale_capability() -> None:
    store = Store(":memory:")
    adapter = ResilientSearchAdapter(_FirecrawlOnly(), store=store, retriever=Retriever(store))
    options = SearchOptions(country="SG", search_lang="zh", ui_language="zh-Hans-SG")

    live = adapter.search_with_outcome("alpha", options=options)
    assert live.status["locale"]["applied_fields"] == ["country"]
    assert live.status["locale"]["ignored_fields"] == ["search_lang", "ui_language"]
    assert adapter.search_cache_key("alpha", options=options) != adapter.search_cache_key(
        "alpha", options=SearchOptions(country="US", search_lang="zh", ui_language="zh-Hans-SG")
    )

    cached = store.cache_get(adapter.search_cache_key("alpha", options=options))
    assert cached["version"] == 4
    assert cached["producer_tier"] == "firecrawl_search"
    assert cached["locale"] == live.status["locale"]
    store.close()


class _BeforeDeltaFailure(_ChunkedLLM):
    def stream(self, messages, **kwargs):  # noqa: ANN001, ANN201, ARG002
        raise RuntimeError("stream unavailable")
        yield  # pragma: no cover


class _AfterDeltaFailure(_ChunkedLLM):
    def stream(self, messages, **kwargs):  # noqa: ANN001, ANN201, ARG002
        yield LLMStreamChunk(delta="Alpha evidence ", model="broken", provider="fake")
        raise RuntimeError("provider disconnected")


class _BeforeDeltaFailureWithEmptyCompletion(_BeforeDeltaFailure):
    def complete(self, messages, **kwargs):  # noqa: ANN001, ANN201, ARG002
        self.complete_calls += 1
        return LLMResult(text="", model="empty")


def test_stream_failure_before_first_delta_uses_sync_completion() -> None:
    llm = _BeforeDeltaFailure()
    store, run_id, payload = _run_fast(llm)
    assert llm.complete_calls == 1
    assert payload["stream_metrics"]["sync_fallback_used"] is True
    assert not [event for event in store.list_events(run_id) if event["type"] == "answer_reset"]
    store.close()


def test_stream_failure_with_empty_sync_fallback_does_not_retry_completion() -> None:
    llm = _BeforeDeltaFailureWithEmptyCompletion()

    with pytest.raises(BaselineError) as exc:
        AnswerComposer(llm).compose_stream("Alpha", [], on_delta=lambda _delta: None)

    assert exc.value.code is ErrorCode.UPSTREAM_ERROR
    assert llm.complete_calls == 1


def test_stream_failure_after_first_delta_resets_without_sync_splice() -> None:
    llm = _AfterDeltaFailure()
    store = Store(":memory:")
    orchestrator = Orchestrator(
        store=store,
        llm=llm,
        retriever=Retriever(store),
        indexer=Indexer(store),
        search=FakeSearch({"https://example.com/a": "Alpha evidence is grounded."}),
    )
    session_id = store.create_session(title="failure")
    with pytest.raises(BaselineError) as exc:
        orchestrator.run(RunRequest(query="Alpha", mode=Mode.FAST, session_id=session_id))

    assert exc.value.code is ErrorCode.UPSTREAM_ERROR
    assert llm.complete_calls == 0
    resets = [event for event in store.list_events(store.get_latest_run_id(session_id) or "") if event["type"] == "answer_reset"]
    assert resets[-1]["payload"]["reason"] == "provider_stream_failed"
    store.close()


def test_low_grounding_keeps_the_visible_draft_and_finishes_verification() -> None:
    store = Store(":memory:")
    orchestrator = Orchestrator(
        store=store,
        llm=_ChunkedLLM(("Unsupported assertion without overlap. [c1]",)),
        retriever=Retriever(store),
        indexer=Indexer(store),
        search=FakeSearch({"https://example.com/a": "Alpha evidence is grounded."}),
        enforce_grounding_gate=True,
    )
    session_id = store.create_session(title="grounding")
    outcome = orchestrator.run(RunRequest(query="Alpha", mode=Mode.FAST, session_id=session_id))

    assert outcome.answer.text == "Unsupported assertion without overlap. [c1]"
    run_id = store.get_latest_run_id(session_id) or ""
    events = store.list_events(run_id)
    assert not [event for event in events if event["type"] == "answer_reset"]
    run = store.get_run(run_id)
    assert run["status"] in {"completed", "completed_degraded"}
    assert run["result"]["verification"]["status"] in {"completed", "failed"}
    assert run["result"]["answer"]["confidence_label"] == "uncertain"
    assert run["result"]["confidence_warning"]["acknowledged"] is False
    metrics = store.get_run(run_id)["metadata"]["stream_metrics"]
    assert metrics["first_answer_delta_ms"] is not None
    assert metrics["delta_event_count"] == len([event for event in events if event["type"] == "answer_delta"])
    assert metrics["delta_bytes"] > 0
    store.close()


def test_stream_event_limit_resets_and_fails_with_stable_code(monkeypatch: pytest.MonkeyPatch) -> None:
    import deepresearch_core.baseline.orchestrator as orchestrator_module

    monkeypatch.setattr(orchestrator_module, "_STREAM_EVENT_MAX_COUNT", 2)
    chunks = tuple("Alpha evidence " * 24 for _ in range(3))
    store = Store(":memory:")
    orchestrator = Orchestrator(
        store=store,
        llm=_ChunkedLLM(chunks),
        retriever=Retriever(store),
        indexer=Indexer(store),
        search=FakeSearch({"https://example.com/a": "Alpha evidence is grounded."}),
    )
    session_id = store.create_session(title="limit")
    with pytest.raises(BaselineError) as exc:
        orchestrator.run(RunRequest(query="Alpha", mode=Mode.FAST, session_id=session_id))

    assert exc.value.code is ErrorCode.STREAM_LIMIT_EXCEEDED
    events = store.list_events(store.get_latest_run_id(session_id) or "")
    assert len([event for event in events if event["type"] == "answer_delta"]) == 2
    assert [event for event in events if event["type"] == "answer_reset"][-1]["payload"]["reason"] == "draft_invalidated"
    store.close()


def test_one_answer_event_never_exceeds_the_utf8_byte_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    import deepresearch_core.baseline.orchestrator as orchestrator_module

    monkeypatch.setattr(orchestrator_module, "_STREAM_EVENT_MAX_BYTES", 12)
    store, run_id, payload = _run_fast(
        _ChunkedLLM(("Alpha evidence 中文内容 is grounded. [c1]",)),
        enforce_grounding_gate=False,
    )
    deltas = [event["payload"]["delta"] for event in store.list_events(run_id) if event["type"] == "answer_delta"]
    assert all(len(str(delta).encode("utf-8")) <= 12 for delta in deltas)
    assert "".join(str(delta) for delta in deltas) == payload["answer"]["text"]
    store.close()


class _CancellingStream(_ChunkedLLM):
    def __init__(self) -> None:
        super().__init__()
        self.cancel = lambda: None
        self.closed = False

    def stream(self, messages, **kwargs):  # noqa: ANN001, ANN201, ARG002
        try:
            yield LLMStreamChunk(delta="Alpha evidence ", model="cancel", provider="fake")
            self.cancel()
            yield LLMStreamChunk(delta="is grounded. [c1]", model="cancel", provider="fake")
        finally:
            self.closed = True


def test_cancel_is_checked_at_chunk_boundary_and_closes_upstream_generator() -> None:
    llm = _CancellingStream()
    store = Store(":memory:")
    orchestrator = Orchestrator(
        store=store,
        llm=llm,
        retriever=Retriever(store),
        indexer=Indexer(store),
        search=FakeSearch({"https://example.com/a": "Alpha evidence is grounded."}),
    )
    session_id = store.create_session(title="cancel")
    run_id = store.create_run(session_id=session_id, mode="fast", query="Alpha")
    llm.cancel = lambda: orchestrator.request_cancel(run_id)

    with pytest.raises(BaselineError) as exc:
        orchestrator.run(RunRequest(query="Alpha", mode=Mode.FAST, session_id=session_id), run_id=run_id)

    assert exc.value.code is ErrorCode.RUN_CANCELLED
    assert llm.closed is True
    resets = [event for event in store.list_events(run_id) if event["type"] == "answer_reset"]
    assert resets[-1]["payload"]["reason"] == "cancelled"
    store.close()


def test_cancel_requested_before_worker_claim_survives_cross_process_handoff() -> None:
    store = Store(":memory:")
    common = {
        "store": store,
        "retriever": Retriever(store),
        "indexer": Indexer(store),
        "search": FakeSearch({"https://example.com/a": "Alpha evidence is grounded."}),
    }
    api_orchestrator = Orchestrator(llm=_ChunkedLLM(), **common)
    worker_orchestrator = Orchestrator(llm=_ChunkedLLM(), **common)
    session_id = store.create_session(title="cross-process-cancel")
    run_id = store.create_run(session_id=session_id, mode="fast", query="Alpha")

    api_orchestrator.request_cancel(run_id)
    with pytest.raises(BaselineError) as exc:
        worker_orchestrator.run(
            RunRequest(query="Alpha", mode=Mode.FAST, session_id=session_id),
            run_id=run_id,
        )

    assert exc.value.code is ErrorCode.RUN_CANCELLED
    assert store.get_run(run_id)["status"] == "cancelled"
    store.close()


def test_hermetic_first_answer_delta_slo_is_measured_from_requested_at() -> None:
    requested_at = time.time()
    store = Store(":memory:")
    orchestrator = Orchestrator(
        store=store,
        llm=_ChunkedLLM(),
        retriever=Retriever(store),
        indexer=Indexer(store),
        search=FakeSearch({"https://example.com/a": "Alpha evidence is grounded."}),
    )
    session_id = store.create_session(title="slo")
    outcome = orchestrator.run(
        RunRequest(query="Alpha", mode=Mode.FAST, session_id=session_id, requested_at=requested_at)
    )
    assert 0 <= float(outcome.to_dict()["stream_metrics"]["first_answer_delta_ms"]) < 2500
    store.close()


class _JudgeFailsAfterDelta(FakeLLM):
    def stream(self, messages, **kwargs):  # noqa: ANN001, ANN201, ARG002
        yield LLMStreamChunk(delta="Unfinished judge draft", model="judge", provider="fake")
        raise RuntimeError("judge disconnected")


def test_council_judge_failure_resets_then_streams_best_member_as_new_generation() -> None:
    judge = _JudgeFailsAfterDelta("judge")
    store = Store(":memory:")
    orchestrator = Orchestrator(
        store=store,
        llm=judge,
        retriever=Retriever(store),
        indexer=Indexer(store),
        search=FakeSearch({"https://example.com/a": "Alpha evidence is grounded."}),
        council_members=[FakeLLM("member-a"), FakeLLM("member-b")],
    )
    session_id = store.create_session(title="council")
    outcome = orchestrator.run(
        RunRequest(query="Alpha", mode=Mode.COUNCIL, session_id=session_id, council_strategy="judge_synthesis")
    )

    events = store.list_events(outcome.run_id)
    answer_events = [event for event in events if event["type"] in {"answer_delta", "answer_reset"}]
    assert [(event["type"], event["payload"]["generation_id"]) for event in answer_events] == [
        ("answer_delta", "g1"),
        ("answer_reset", "g1"),
        ("answer_delta", "g2"),
    ]
    assert answer_events[1]["payload"]["reason"] == "council_judge_fallback"
    assert answer_events[-1]["payload"]["delta"] == outcome.answer.text
    store.close()


def test_parallel_vote_emits_one_complete_delta() -> None:
    store = Store(":memory:")
    orchestrator = Orchestrator(
        store=store,
        llm=FakeLLM("judge"),
        retriever=Retriever(store),
        indexer=Indexer(store),
        search=FakeSearch({"https://example.com/a": "Alpha evidence is grounded."}),
        council_members=[FakeLLM("member-a"), FakeLLM("member-b")],
    )
    session_id = store.create_session(title="vote")
    outcome = orchestrator.run(
        RunRequest(query="Alpha", mode=Mode.COUNCIL, session_id=session_id, council_strategy="parallel_vote")
    )
    deltas = [event for event in store.list_events(outcome.run_id) if event["type"] == "answer_delta"]
    assert len(deltas) == 1
    assert deltas[0]["payload"]["delta"] == outcome.answer.text
    assert outcome.to_dict()["stream_metrics"]["sync_fallback_used"] is False
    store.close()


def test_service_freezes_requested_and_effective_locale_before_enqueue() -> None:
    from deepresearch_core.baseline import BaselineService
    from deepresearch_core.settings import ProviderSettings

    service = BaselineService(provider=ProviderSettings(), db_path=":memory:")
    service._llm_ready = True
    service._run_safe = lambda request, run_id: None  # type: ignore[method-assign]
    created = service.create_run(
        {
            "query": "locale",
            "mode": "fast",
            "search_options": {"country": "sg", "ui_language": "zh-hans-sg"},
        },
        background=False,
    )
    run = service._store.get_run(created["run_id"])
    assert run is not None
    assert run["metadata"]["requested_locale"] == {"country": "SG", "ui_language": "zh-Hans-SG"}
    assert run["metadata"]["effective_locale"] == {
        "country": "SG",
        "search_lang": "en",
        "ui_language": "zh-Hans-SG",
    }
    assert run["metadata"]["requested_at"] > 0
    assert run["metadata"]["deadline_at"] == pytest.approx(
        run["metadata"]["requested_at"] + 30.0
    )


def test_adopted_firecrawl_result_marks_partial_locale_application_degraded() -> None:
    store = Store(":memory:")
    retriever = Retriever(store)
    search = ResilientSearchAdapter(_FirecrawlOnly(), store=store, retriever=retriever)
    orchestrator = Orchestrator(
        store=store,
        llm=FakeLLM(),
        retriever=retriever,
        indexer=Indexer(store),
        search=search,
    )
    session_id = store.create_session(title="locale-degraded")
    outcome = orchestrator.run(
        RunRequest(
            query="Alpha",
            mode=Mode.FAST,
            session_id=session_id,
            search_options=SearchOptions(country="SG", search_lang="zh", ui_language="zh-Hans-SG"),
        )
    )
    external = next(item for item in outcome.to_dict()["retrieval_outcomes"] if item["lane"] == "external")
    assert outcome.status == "completed_degraded"
    assert "search_locale_partially_applied" in outcome.to_dict()["degraded_reasons"]
    assert external["locale"]["applied_fields"] == ["country"]
    assert external["locale"]["ignored_fields"] == ["search_lang", "ui_language"]
    store.close()


class _LocalEvidenceRetriever:
    def retrieve_evidence(self, query: str, *, top_k: int):  # noqa: ANN201, ARG002
        return [
            EvidenceItem(
                chunk_id=1,
                document_id="note-1",
                url=None,
                title="Local note",
                text="Alpha evidence is grounded.",
                score=1.0,
                origin="local",
                artifact_type="library_note",
                artifact_id="note-1",
            )
        ]


class _AllSearchProvidersFail(_FirecrawlOnly):
    def firecrawl_search(self, query: str, *, count: int = 5, options=None):  # noqa: ANN001, ANN201, ARG002
        raise RuntimeError("firecrawl down")


def test_unadopted_failed_provider_does_not_add_locale_degraded_reason() -> None:
    store = Store(":memory:")
    retriever = Retriever(store)
    search = ResilientSearchAdapter(_AllSearchProvidersFail(), store=store, retriever=retriever)
    orchestrator = Orchestrator(
        store=store,
        llm=FakeLLM(),
        retriever=retriever,
        indexer=Indexer(store),
        search=search,
        library_retriever=_LocalEvidenceRetriever(),
    )
    session_id = store.create_session(title="locale-unadopted")
    outcome = orchestrator.run(
        RunRequest(
            query="Alpha",
            mode=Mode.FAST,
            session_id=session_id,
            search_options=SearchOptions(search_lang="zh", ui_language="zh-Hans-SG"),
        )
    )
    assert "search_locale_partially_applied" not in outcome.to_dict()["degraded_reasons"]
    store.close()


def test_stream_api_query_cursor_takes_precedence_over_last_event_id() -> None:
    from fastapi.testclient import TestClient

    from deepresearch_core.baseline import BaselineService, create_app
    from deepresearch_core.settings import ProviderSettings

    service = BaselineService(provider=ProviderSettings(), db_path=":memory:")
    session_id = service._store.create_session(title="api-cursor")
    run_id = service._store.create_run(session_id=session_id, mode="fast", query="q")
    first = service._store.append_event(run_id, "node_status", {"message": "one"})
    second = service._store.append_event(
        run_id,
        "answer_delta",
        {"delta": "two", "delta_seq": 1, "generation_id": "g1"},
    )
    third = service._store.append_event(run_id, "workflow_end", {"status": "completed"})
    service._store.update_run(run_id, status="completed", result={"answer": {"text": "two"}})

    response = TestClient(create_app(service)).get(
        f"/api/v1/runs/{run_id}/stream?after_seq={first}",
        headers={"Last-Event-ID": str(second)},
    )
    assert response.status_code == 200
    assert f"id: {second}\nevent: answer_delta" in response.text
    assert f"id: {third}\nevent: workflow_end" in response.text


def test_invalid_locale_returns_invalid_request_before_provider_use() -> None:
    from fastapi.testclient import TestClient

    from deepresearch_core.baseline import BaselineService, create_app
    from deepresearch_core.settings import ProviderSettings

    service = BaselineService(provider=ProviderSettings(), db_path=":memory:")
    response = TestClient(create_app(service)).post(
        "/api/v1/runs",
        json={"query": "q", "mode": "fast", "search_options": {"country": "USA"}},
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "INVALID_REQUEST"


class _HybridStreamingLLM(FakeLLM):
    def stream(self, messages, **kwargs):  # noqa: ANN001, ANN201
        result = super().complete(messages, **kwargs)
        midpoint = max(1, len(result.text) // 2)
        yield LLMStreamChunk(delta=result.text[:midpoint], model=result.model, provider="fake")
        yield LLMStreamChunk(delta=result.text[midpoint:], model=result.model, provider="fake")
        yield LLMStreamChunk(done=True, model=result.model, provider="fake")


@pytest.mark.parametrize("mode", [Mode.FAST, Mode.MULTIPLE, Mode.DEEP])
def test_each_writer_mode_uses_the_streaming_adapter(mode: Mode) -> None:
    store = Store(":memory:")
    orchestrator = Orchestrator(
        store=store,
        llm=_HybridStreamingLLM(),
        retriever=Retriever(store),
        indexer=Indexer(store),
        search=FakeSearch({"https://example.com/a": "Alpha evidence is grounded."}),
    )
    session_id = store.create_session(title=mode.value)
    outcome = orchestrator.run(
        RunRequest(
            query="Alpha evidence",
            mode=mode,
            session_id=session_id,
            overrides={"max_searches": 1},
        )
    )
    deltas = [event for event in store.list_events(outcome.run_id) if event["type"] == "answer_delta"]
    assert deltas
    assert "".join(str(event["payload"]["delta"]) for event in deltas) == outcome.answer.text
    assert outcome.to_dict()["stream_metrics"]["sync_fallback_used"] is False
    store.close()


class _ToggleFirecrawl(_FirecrawlOnly):
    def __init__(self) -> None:
        self.fail = False

    def firecrawl_search(self, query: str, *, count: int = 5, options=None):  # noqa: ANN001, ANN201, ARG002
        if self.fail:
            raise RuntimeError("firecrawl down")
        return super().firecrawl_search(query, count=count, options=options)


def test_cache_hit_retains_producer_tier_and_current_locale_observability() -> None:
    store = Store(":memory:")
    client = _ToggleFirecrawl()
    adapter = ResilientSearchAdapter(client, store=store, retriever=Retriever(store))
    options = SearchOptions(country="SG", search_lang="zh", ui_language="zh-Hans-SG")
    adapter.search_with_outcome("alpha", options=options)
    client.fail = True

    cached = adapter.search_with_outcome("alpha", options=options)
    assert cached.status["fallback_tier"] == "cache"
    assert cached.status["producer_tier"] == "firecrawl_search"
    assert cached.status["locale"]["ignored_fields"] == ["search_lang", "ui_language"]
    assert cached.hits[0].metadata["search_provider"] == "firecrawl"
    store.close()


def test_out_of_process_failure_resets_any_persisted_draft_before_error() -> None:
    from deepresearch_core.baseline import BaselineService
    from deepresearch_core.settings import ProviderSettings

    service = BaselineService(provider=ProviderSettings(), db_path=":memory:")
    session_id = service._store.create_session(title="worker-failure")
    requested_at = time.time() - 0.05
    run_id = service._store.create_run(
        session_id=session_id,
        mode="fast",
        query="q",
        metadata={"requested_at": requested_at},
    )
    service._store.append_event(
        run_id,
        "answer_delta",
        {"delta": "draft", "delta_seq": 1, "generation_id": "g1"},
    )

    service.fail_run(run_id, "worker died")

    events = service._store.list_events(run_id)
    assert [event["type"] for event in events] == ["answer_delta", "answer_reset", "error"]
    assert events[1]["payload"] == {"generation_id": "g1", "reason": "draft_invalidated"}
    metrics = service._store.get_run(run_id)["metadata"]["stream_metrics"]
    assert metrics["first_answer_delta_ms"] >= 0
    assert metrics["delta_event_count"] == 1
    assert metrics["delta_bytes"] == len("draft".encode("utf-8"))


def test_out_of_process_failure_tolerates_malformed_delta_payload() -> None:
    from deepresearch_core.baseline import BaselineService
    from deepresearch_core.settings import ProviderSettings

    service = BaselineService(provider=ProviderSettings(), db_path=":memory:")
    session_id = service._store.create_session(title="worker-failure-malformed")
    run_id = service._store.create_run(session_id=session_id, mode="fast", query="q")
    service._store.append_event(run_id, "answer_delta", "malformed")  # type: ignore[arg-type]

    service.fail_run(run_id, "worker died")

    run = service._store.get_run(run_id)
    assert run["status"] == "error"
    assert run["metadata"]["stream_metrics"]["delta_event_count"] == 1
    assert run["metadata"]["stream_metrics"]["delta_bytes"] == 0
