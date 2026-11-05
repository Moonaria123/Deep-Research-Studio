"""First-generation Multiple mode query-prefetch regressions."""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

from deepresearch_core.baseline.adapters import ResilientSearchAdapter
from deepresearch_core.baseline.budget import Mode, RunRequest
from deepresearch_core.baseline.errors import BaselineError, ErrorCode
from deepresearch_core.baseline.indexing import Indexer
from deepresearch_core.baseline.orchestrator import Orchestrator
from deepresearch_core.baseline.retrieval import Retriever
from deepresearch_core.baseline.store import Store
from tests.baseline.fakes import FakeLLM


class _OwnerThreadStore(Store):
    def __init__(self) -> None:
        self.owner_thread_id = threading.get_ident()
        self.cache_write_threads: list[int] = []
        super().__init__(":memory:")

    def cache_get(self, key: str):  # noqa: ANN201
        assert threading.get_ident() == self.owner_thread_id
        return super().cache_get(key)

    def cache_set(self, key: str, value, *, ttl_s: float) -> None:  # noqa: ANN001
        current = threading.get_ident()
        assert current == self.owner_thread_id
        self.cache_write_threads.append(current)
        super().cache_set(key, value, ttl_s=ttl_s)


class _ParallelSearchClient:
    settings = SimpleNamespace(country="US", search_lang="en")

    def __init__(self, owner_thread_id: int) -> None:
        self.owner_thread_id = owner_thread_id
        self._lock = threading.Lock()
        self._first_wave = threading.Barrier(3)
        self.calls = 0
        self.active = 0
        self.max_active = 0

    def search(self, query: str, *, count: int = 5, options=None):  # noqa: ANN001, ANN201, ARG002
        assert threading.get_ident() != self.owner_thread_id
        with self._lock:
            self.calls += 1
            call_number = self.calls
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            if call_number <= 3:
                self._first_wave.wait(timeout=2.0)
            time.sleep(0.03)
            slug = query.replace(" ", "-").lower()
            return [
                {
                    "title": f"Evidence for {query}",
                    "url": f"https://example.test/{slug}",
                    "description": f"{query} has directly documented alpha evidence.",
                }
            ][:count]
        finally:
            with self._lock:
                self.active -= 1

    def firecrawl_search(self, *args, **kwargs):  # noqa: ANN002, ANN003, ANN201
        raise AssertionError("provider fallback should not run")

    def scrape(self, url: str):  # noqa: ANN201
        return {"markdown": f"Documented alpha evidence from {url}.", "raw": {}}


class _BlockingSearchClient(_ParallelSearchClient):
    def __init__(self) -> None:
        super().__init__(-1)
        self.wave_ready = threading.Event()
        self.release = threading.Event()

    def search(self, query: str, *, count: int = 5, options=None):  # noqa: ANN001, ANN201, ARG002
        with self._lock:
            self.calls += 1
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            if self.active == 3:
                self.wave_ready.set()
        try:
            assert self.release.wait(2.0)
            return [
                {
                    "title": f"Evidence for {query}",
                    "url": f"https://example.test/{self.calls}",
                    "description": f"{query} has directly documented alpha evidence.",
                }
            ][:count]
        finally:
            with self._lock:
                self.active -= 1


class _TrackingSearchAdapter(ResilientSearchAdapter):
    def __init__(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
        super().__init__(*args, **kwargs)
        self.persist_calls = 0

    def persist_search_cache(self, *args, **kwargs):  # noqa: ANN002, ANN003, ANN201
        self.persist_calls += 1
        return super().persist_search_cache(*args, **kwargs)


class _ManualClock:
    def __init__(self) -> None:
        self.now = 0.0
        self._lock = threading.Lock()

    def __call__(self) -> float:
        with self._lock:
            return self.now

    def advance(self, seconds: float) -> None:
        with self._lock:
            self.now += seconds


class _ExpireWriterReserveLane:
    def __init__(self, clock: _ManualClock) -> None:
        self.clock = clock
        self.calls = 0

    def retrieve_evidence(self, query: str, *, top_k: int):  # noqa: ANN201, ARG002
        self.calls += 1
        self.clock.advance(38.0)
        return []


def test_multiple_prefetches_network_queries_with_bounded_concurrency() -> None:
    store = _OwnerThreadStore()
    client = _ParallelSearchClient(store.owner_thread_id)
    search = ResilientSearchAdapter(client, store=store, retriever=Retriever(store))
    orchestrator = Orchestrator(
        store=store,
        llm=FakeLLM(),
        retriever=Retriever(store),
        indexer=Indexer(store),
        search=search,
        enforce_grounding_gate=False,
    )
    planned = ["alpha one", "alpha two", "alpha three", "alpha four"]
    orchestrator._subqueries = lambda *_args, **_kwargs: list(planned)  # type: ignore[method-assign]

    outcome = orchestrator.run(
        RunRequest(query="Compare alpha evidence", mode=Mode.MULTIPLE)
    )

    assert outcome.status in {"completed", "completed_degraded"}
    assert client.calls == 4
    assert client.max_active == 3
    assert [lane["query"] for lane in outcome.extra["lanes"]] == planned
    assert store.cache_write_threads
    assert set(store.cache_write_threads) == {store.owner_thread_id}
    store.close()


def test_multiple_cancel_stops_sliding_window_and_discards_late_network_results() -> None:
    store = Store(":memory:")
    client = _BlockingSearchClient()
    search = _TrackingSearchAdapter(client, store=store, retriever=Retriever(store))
    orchestrator = Orchestrator(
        store=store,
        llm=FakeLLM(),
        retriever=Retriever(store),
        indexer=Indexer(store),
        search=search,
        enforce_grounding_gate=False,
    )
    orchestrator._subqueries = (  # type: ignore[method-assign]
        lambda *_args, **_kwargs: ["alpha one", "alpha two", "alpha three", "alpha four"]
    )
    handles = []
    begin_external_search = orchestrator._begin_external_search

    def capture_handle(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        handle = begin_external_search(*args, **kwargs)
        if handle is not None:
            handles.append(handle)
        return handle

    orchestrator._begin_external_search = capture_handle  # type: ignore[method-assign]
    run_id = store.create_run(
        session_id=None,
        mode=Mode.MULTIPLE.value,
        query="Compare alpha evidence",
        budget=RunRequest(query="x", mode=Mode.MULTIPLE).resolved_budget().to_dict(),
    )
    errors: list[BaselineError] = []

    def execute() -> None:
        try:
            orchestrator.run(
                RunRequest(query="Compare alpha evidence", mode=Mode.MULTIPLE),
                run_id=run_id,
            )
        except BaselineError as error:
            errors.append(error)

    thread = threading.Thread(target=execute)
    thread.start()
    assert client.wave_ready.wait(2.0)
    orchestrator.request_cancel(run_id)
    client.release.set()
    thread.join(timeout=3.0)

    assert not thread.is_alive()
    assert [error.code for error in errors] == [ErrorCode.RUN_CANCELLED]
    assert client.calls == 3
    assert search.persist_calls == 0
    assert len(handles) == 3
    assert all(handle.executor is None for handle in handles)
    assert store.get_run(run_id)["status"] == "cancelled"
    store.close()


def test_multiple_writer_reserve_stops_fourth_query_and_discards_prefetched_results() -> None:
    clock = _ManualClock()
    store = Store(":memory:")
    client = _ParallelSearchClient(threading.get_ident())
    search = _TrackingSearchAdapter(client, store=store, retriever=Retriever(store))
    local_lane = _ExpireWriterReserveLane(clock)
    orchestrator = Orchestrator(
        store=store,
        llm=FakeLLM(),
        retriever=Retriever(store),
        indexer=Indexer(store),
        search=search,
        library_retriever=local_lane,
        enforce_grounding_gate=False,
        clock=clock,
    )
    orchestrator._subqueries = (  # type: ignore[method-assign]
        lambda *_args, **_kwargs: ["alpha one", "alpha two", "alpha three", "alpha four"]
    )
    contexts = []
    build_context = orchestrator._build_context

    def capture_context(request):  # noqa: ANN001, ANN202
        context = build_context(request)
        contexts.append(context)
        return context

    orchestrator._build_context = capture_context  # type: ignore[method-assign]

    try:
        orchestrator.run(
            RunRequest(query="Compare alpha evidence", mode=Mode.MULTIPLE)
        )
    except BaselineError as error:
        assert error.code is ErrorCode.RUN_TIMEOUT
    else:
        raise AssertionError("writer reserve exhaustion must time out the run")

    assert client.calls == 3
    assert local_lane.calls == 1
    assert contexts[0].ordinary_searches_used == 3
    assert search.persist_calls == 0
    store.close()
