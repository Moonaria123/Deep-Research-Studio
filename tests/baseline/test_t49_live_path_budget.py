"""T49 absolute-deadline, demand-ingest, and terminal diagnostics contracts."""

from __future__ import annotations

import time
import threading
from types import SimpleNamespace

import pytest

from deepresearch_core.baseline.adapters import (
    Document,
    LLMResult,
    LLMStreamChunk,
    ResilientSearchAdapter,
    SearchHit,
)
from deepresearch_core.baseline.budget import Budget, Mode, RunRequest
from deepresearch_core.baseline.deadline import current_provider_call_budget
from deepresearch_core.baseline.errors import BaselineError, ErrorCode
from deepresearch_core.baseline.indexing import Indexer
from deepresearch_core.baseline.memory import MemoryEngine, MemoryLayer
from deepresearch_core.baseline.orchestrator import Orchestrator
from deepresearch_core.baseline.retrieval import EvidenceItem, Retriever
from deepresearch_core.baseline.store import Store
from tests.baseline.fakes import FakeLLM, FakeSearch


class _ManualClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class _CountingSearch:
    name = "t49-search"

    def __init__(self, clock: _ManualClock | None = None, *, advance_on_fetch: float = 0.0) -> None:
        self.clock = clock
        self.advance_on_fetch = advance_on_fetch
        self.search_calls = 0
        self.fetch_calls: list[str] = []

    def search(self, query: str, *, count: int = 5, options=None):  # noqa: ANN001, ANN201, ARG002
        self.search_calls += 1
        return [
            SearchHit(
                title=f"Alpha evidence {index}",
                url=f"https://example.test/{index}",
                snippet="",
                source="example.test",
                rank=index,
            )
            for index in range(1, count + 1)
        ]

    def fetch(self, url: str) -> Document:
        self.fetch_calls.append(url)
        if self.clock is not None and self.advance_on_fetch:
            self.clock.now += self.advance_on_fetch
        return Document(
            url=url,
            title="Alpha evidence",
            text="Alpha evidence is grounded and directly supports the answer.",
            content_type="web",
        )


class _NamedSourceSearch:
    name = "named-source-search"

    def __init__(self) -> None:
        self.fetch_calls: list[str] = []

    def search(self, query: str, *, count: int = 5, options=None):  # noqa: ANN001, ANN201, ARG002
        return [
            SearchHit(
                title="Physical activity",
                url="https://en.wikipedia.org/wiki/Physical_activity",
                snippet=(
                    "WHO defines physical activity as any bodily movement produced by skeletal muscles."
                ),
                source="en.wikipedia.org",
                rank=0,
            ),
            SearchHit(
                title="Physical activity",
                url="https://www.who.int/news-room/fact-sheets/detail/physical-activity",
                snippet="Physical activity has significant health benefits for hearts and bodies.",
                source="who.int",
                rank=1,
            ),
        ][:count]

    def fetch(self, url: str) -> Document:
        self.fetch_calls.append(url)
        return Document(
            url=url,
            title="Physical activity",
            text=(
                "WHO defines physical activity as any bodily movement produced by skeletal muscles "
                "that requires energy expenditure."
            ),
            content_type="web",
        )


class _GuidanceTitleSearch:
    name = "guidance-title-search"

    def __init__(self) -> None:
        self.fetch_calls: list[str] = []

    def search(self, query: str, *, count: int = 5, options=None):  # noqa: ANN001, ANN201, ARG002
        return [
            SearchHit(
                title="FastStats: Sleep in Adults",
                url="https://www.cdc.gov/sleep/data-research/facts-stats/adults-sleep-facts-and-stats.html",
                snippet="",
                source="cdc.gov",
                rank=0,
            ),
            SearchHit(
                title="Module 2. How Much Sleep Do You Need?",
                url="https://www.cdc.gov/niosh/work-hour-training-for-nurses/longhours/mod2/08.html",
                snippet="",
                source="cdc.gov",
                rank=5,
            ),
        ][:count]

    def fetch(self, url: str) -> Document:
        self.fetch_calls.append(url)
        return Document(
            url=url,
            title="Module 2. How Much Sleep Do You Need?",
            text="CDC guidance says most adults should get at least seven hours of sleep each night.",
            content_type="web",
        )


class _GuidanceOverviewSearch:
    name = "guidance-overview-search"

    def __init__(self) -> None:
        self.fetch_calls: list[str] = []

    def search(self, query: str, *, count: int = 5, options=None):  # noqa: ANN001, ANN201, ARG002
        return [
            SearchHit(
                title="FastStats: Sleep in Adults",
                url="https://www.cdc.gov/sleep/data-research/facts-stats/adults-sleep-facts-and-stats.html",
                snippet="",
                source="cdc.gov",
                rank=0,
            ),
            SearchHit(
                title="About Sleep",
                url="https://www.cdc.gov/sleep/about/index.html",
                snippet="",
                source="cdc.gov",
                rank=1,
            ),
            SearchHit(
                title="Do You Get Enough Sleep?",
                url="https://stacks.cdc.gov/view/cdc/59858/cdc_59858_DS1.pdf",
                snippet="",
                source="cdc.gov",
                rank=5,
            ),
        ][:count]

    def fetch(self, url: str) -> Document:
        self.fetch_calls.append(url)
        return Document(
            url=url,
            title="About Sleep",
            text="Most adults need at least 7 hours of sleep per night.",
            content_type="web",
        )


class _PythonDocsSearch:
    name = "python-docs-search"

    def __init__(self) -> None:
        self.fetch_calls: list[str] = []

    def search(self, query: str, *, count: int = 5, options=None):  # noqa: ANN001, ANN201, ARG002
        return [
            SearchHit(
                title="pathlib — Object-oriented filesystem paths",
                url="https://docs.python.org/3/library/pathlib.html",
                snippet="",
                source="docs.python.org",
                rank=0,
            )
        ][:count]

    def fetch(self, url: str) -> Document:
        self.fetch_calls.append(url)
        return Document(
            url=url,
            title="pathlib — Object-oriented filesystem paths",
            text=(
                "pathlib — Object-oriented filesystem paths. This module offers classes "
                "representing filesystem paths with semantics appropriate for different "
                "operating systems."
            ),
            content_type="web",
        )


class _RedirectMismatchSearch(_CountingSearch):
    def fetch(self, url: str) -> Document:
        self.fetch_calls.append(url)
        return Document(
            url=url,
            title="Alpha evidence",
            text="Alpha evidence is grounded and directly supports the answer.",
            content_type="web",
            metadata={"canonical_url": "https://redirect.example/alpha"},
        )


class _BoilerplateFetchedSearch(_CountingSearch):
    def fetch(self, url: str) -> Document:
        self.fetch_calls.append(url)
        return Document(
            url=url,
            title="Alpha evidence",
            text=(
                ("Navigation menu. " * 130)
                + "\n\nAlpha evidence directly supports the current answer. "
                + "Alpha evidence is authoritative and grounded."
            ),
            content_type="web",
        )


class _BudgetAwareSequentialSearch(_CountingSearch):
    supports_search_usage_policy = True
    supports_focused_fetch = True

    def __init__(self) -> None:
        super().__init__()
        self.fetch_budget_seen = False
        self.focused_query_seen = ""

    def read_fetch_cache(self, url: str, **kwargs):  # noqa: ANN003, ANN201, ARG002
        return None

    def fetch_network(self, url: str, **kwargs) -> Document:  # noqa: ANN003
        budget = current_provider_call_budget()
        assert budget is not None
        budget.check("fast_sequential_fetch")
        self.fetch_budget_seen = True
        self.focused_query_seen = str(kwargs.get("relevance_query") or "")
        self.fetch_calls.append(url)
        return Document(
            url=url,
            title="Alpha evidence",
            text="Alpha evidence is grounded and directly supports the answer.",
            content_type="web",
        )

    def persist_fetch_cache(self, document: Document, **kwargs) -> None:  # noqa: ANN003, ARG002
        return None


class _CacheSpySearch(_BudgetAwareSequentialSearch):
    def __init__(self) -> None:
        super().__init__()
        self.cache_writes = 0

    def persist_fetch_cache(self, document: Document, **kwargs) -> None:  # noqa: ANN003, ARG002
        self.cache_writes += 1


class _TerminalizingIndexer(Indexer):
    def __init__(self, store: Store) -> None:
        super().__init__(store)
        self._test_store = store

    def prepare_document(self, document: Document):  # noqa: ANN201
        prepared = super().prepare_document(document)
        run_id = self._test_store.get_latest_run_id("") or ""
        assert self._test_store.transition_run(
            run_id,
            expected_statuses={"running"},
            status="error",
            error={"error": {"code": "RUN_TIMEOUT", "message": "terminalized during prepare"}},
        )
        return prepared


class _CountingLLM(FakeLLM):
    def __init__(self) -> None:
        super().__init__()
        self.writer_calls = 0

    def complete(self, messages, **kwargs):  # noqa: ANN001, ANN201
        self.writer_calls += 1
        return super().complete(messages, **kwargs)


class _CountingEmbedder:
    name = "counting-embedder"

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed(self, texts, *, model=None):  # noqa: ANN001, ANN201, ARG002
        values = [str(text) for text in texts]
        self.calls.append(values)
        return [[1.0] for _ in values]


class _DeadlineStreamLLM:
    name = "deadline-stream"

    def __init__(self, clock: _ManualClock) -> None:
        self.clock = clock

    def stream(self, messages, **kwargs):  # noqa: ANN001, ANN201, ARG002
        yield LLMStreamChunk(delta="Alpha evidence ", model="deadline", provider="fake")
        self.clock.now = 31.0
        yield LLMStreamChunk(delta="is grounded. [c1]", model="deadline", provider="fake")

    def complete(self, messages, **kwargs):  # noqa: ANN001, ANN201, ARG002
        return LLMResult(text="Alpha evidence is grounded. [c1]", model="deadline")

    def embed(self, texts, *, model=None):  # noqa: ANN001, ANN201, ARG002
        return [[1.0] for _ in texts]


def _orchestrator(*, llm, search, clock=None, resolver=None):  # noqa: ANN001, ANN201
    store = Store(":memory:")
    return (
        Orchestrator(
            store=store,
            llm=llm,
            llm_resolver=resolver,
            retriever=Retriever(store),
            indexer=Indexer(store),
            search=search,
            enforce_grounding_gate=False,
            clock=clock,
        ),
        store,
    )


def test_expired_absolute_deadline_fails_before_provider_resolution_or_search() -> None:
    search = _CountingSearch()
    resolved = 0

    def resolver(_request):  # noqa: ANN001, ANN202
        nonlocal resolved
        resolved += 1
        return FakeLLM()

    orchestrator, store = _orchestrator(llm=FakeLLM(), search=search, resolver=resolver)
    session_id = store.create_session(title="expired")
    run_id = store.create_run(session_id=session_id, mode="fast", query="Alpha")
    now = time.time()

    with pytest.raises(BaselineError) as exc:
        orchestrator.run(
            RunRequest(
                query="Alpha",
                mode=Mode.FAST,
                session_id=session_id,
                requested_at=now - 31,
                deadline_at=now - 1,
            ),
            run_id=run_id,
        )

    assert exc.value.code is ErrorCode.RUN_TIMEOUT
    assert resolved == 0
    assert search.search_calls == 0
    run = store.get_run(run_id)
    assert run["status"] == "error"
    assert run["metadata"]["deadline_summary"]["deadline_met"] is False
    store.close()


def test_redirect_domain_mismatch_never_reaches_index_and_is_reported_as_low_confidence() -> None:
    search = _RedirectMismatchSearch()
    llm = _CountingLLM()
    orchestrator, store = _orchestrator(llm=llm, search=search)

    outcome = orchestrator.run(RunRequest(query="What is Alpha evidence?", mode=Mode.FAST))

    assert outcome.answer.text.strip()
    assert llm.writer_calls == 1
    assert search.fetch_calls
    assert store.list_documents() == []
    run = store.get_run(outcome.run_id)
    assert run["result"]["answer"]["confidence_label"] == "uncertain"
    assert run["result"]["confidence_warning"]["acknowledged"] is False
    store.close()


def test_fast_demand_ingest_fetches_at_most_one_regular_candidate() -> None:
    search = _CountingSearch()
    orchestrator, store = _orchestrator(llm=FakeLLM(), search=search)
    session_id = store.create_session(title="demand")

    outcome = orchestrator.run(
        RunRequest(query="Alpha evidence", mode=Mode.FAST, session_id=session_id)
    )

    assert outcome.status == "completed"
    assert len(search.fetch_calls) == 1
    assert outcome.to_dict()["retrieval_budget"]["fetches_used"] == 1
    assert outcome.to_dict()["retrieval_budget"]["fetches_max"] == 1
    store.close()


def test_named_source_query_fetches_matching_publisher_instead_of_adopting_third_party_snippet() -> None:
    search = _NamedSourceSearch()
    orchestrator, store = _orchestrator(llm=FakeLLM(), search=search)

    outcome = orchestrator.run(
        RunRequest(
            query="According to WHO, what is the definition of physical activity?",
            mode=Mode.FAST,
        )
    )

    assert outcome.status == "completed"
    assert search.fetch_calls == [
        "https://www.who.int/news-room/fact-sheets/detail/physical-activity"
    ]
    assert [source["url"] for source in outcome.to_dict()["sources"]] == search.fetch_calls
    assert outcome.to_dict()["retrieval_budget"]["fetches_used"] == 1
    store.close()


def test_named_source_query_prefers_the_closest_title_phrase_within_one_fetch_budget() -> None:
    search = _GuidanceTitleSearch()
    orchestrator, store = _orchestrator(llm=FakeLLM(), search=search)

    outcome = orchestrator.run(
        RunRequest(
            query="According to CDC guidance, how much sleep should most adults get?",
            mode=Mode.FAST,
        )
    )

    assert outcome.status == "completed"
    assert search.fetch_calls == [
        "https://www.cdc.gov/niosh/work-hour-training-for-nurses/longhours/mod2/08.html"
    ]
    assert outcome.to_dict()["retrieval_budget"]["fetches_used"] == 1
    store.close()


def test_normative_named_source_prefers_guidance_over_ranked_statistics() -> None:
    search = _GuidanceOverviewSearch()
    orchestrator, store = _orchestrator(llm=FakeLLM(), search=search)

    outcome = orchestrator.run(
        RunRequest(
            query="According to CDC guidance, how much sleep should most adults get?",
            mode=Mode.FAST,
        )
    )

    assert outcome.status == "completed"
    assert search.fetch_calls == ["https://www.cdc.gov/sleep/about/index.html"]
    assert outcome.to_dict()["retrieval_budget"]["fetches_used"] == 1
    store.close()


def test_instruction_memory_cannot_suppress_named_source_body_fetch() -> None:
    store = Store(":memory:")
    memory = MemoryEngine(store)
    memory.remember(
        MemoryLayer.WORKSPACE,
        "default",
        "Provide a concise explanation using authoritative primary documentation.",
    )
    search = _PythonDocsSearch()
    orchestrator = Orchestrator(
        store=store,
        llm=FakeLLM(),
        retriever=Retriever(store),
        indexer=Indexer(store),
        search=search,
        memory=memory,
        enforce_grounding_gate=False,
    )

    outcome = orchestrator.run(
        RunRequest(
            query="According to the Python documentation, what problem does pathlib solve?",
            mode=Mode.FAST,
        )
    )

    assert outcome.status == "completed"
    assert search.fetch_calls == ["https://docs.python.org/3/library/pathlib.html"]
    assert outcome.to_dict()["retrieval_budget"]["fetches_used"] == 1
    store.close()


def test_fast_sequential_fetch_installs_provider_deadline_budget() -> None:
    search = _BudgetAwareSequentialSearch()
    orchestrator, store = _orchestrator(llm=FakeLLM(), search=search)

    outcome = orchestrator.run(RunRequest(query="Alpha evidence", mode=Mode.FAST))

    assert outcome.status == "completed"
    assert search.fetch_budget_seen is True
    assert search.focused_query_seen == "Alpha evidence"
    assert len(search.fetch_calls) == 1
    store.close()


def test_transient_fetched_evidence_does_not_block_writer_on_optional_embedding() -> None:
    store = Store(":memory:")
    embedder = _CountingEmbedder()
    orchestrator = Orchestrator(
        store=store,
        llm=FakeLLM(),
        retriever=Retriever(store),
        indexer=Indexer(store, embedder=embedder),
        search=_CountingSearch(),
        enforce_grounding_gate=False,
    )

    outcome = orchestrator.run(RunRequest(query="Alpha evidence", mode=Mode.FAST))

    assert outcome.status == "completed"
    assert outcome.answer.citations
    assert embedder.calls == []
    assert store.list_documents()
    store.close()


def test_fetched_citation_uses_relevant_excerpt_while_index_keeps_sanitized_body() -> None:
    search = _BoilerplateFetchedSearch()
    orchestrator, store = _orchestrator(llm=FakeLLM(), search=search)

    outcome = orchestrator.run(RunRequest(query="Alpha evidence", mode=Mode.FAST))

    citation = outcome.answer.citations[0]
    document = store.get_document(citation.document_id)
    assert citation.snippet.startswith("Alpha evidence")
    assert "Navigation menu" not in citation.snippet
    assert document is not None
    assert str(document["text"]).startswith("Navigation menu")
    store.close()


def test_cited_sentence_contract_avoids_fast_skeleton_and_grounding_repair() -> None:
    search = _BoilerplateFetchedSearch()
    llm = FakeLLM(
        scripted={
            "Alpha evidence": (
                "Alpha evidence directly supports the current answer. [c1] "
                "Alpha evidence is authoritative and grounded. [c1]"
            )
        }
    )
    orchestrator, store = _orchestrator(llm=llm, search=search)

    outcome = orchestrator.run(
        RunRequest(
            query="What is Alpha evidence? Answer in two cited sentences.",
            mode=Mode.FAST,
        )
    )

    assert outcome.status == "completed"
    assert not outcome.answer.text.startswith("##")
    assert len(llm.calls) == 1
    store.close()


def test_complementary_snippets_reach_writer_without_conflict_classifier_call() -> None:
    search = FakeSearch(
        {
            "https://example.test/definition": (
                "Physical activity is defined as any movement produced by skeletal muscles "
                "that requires energy expenditure."
            ),
            "https://example.test/benefits": (
                "Physical activity contributes to disease prevention and reduces symptoms "
                "of depression and anxiety."
            ),
        }
    )
    llm = FakeLLM(
        scripted={
            "What is physical activity?": (
                "Physical activity is movement produced by skeletal muscles. [c1] "
                "It requires energy expenditure. [c1]"
            )
        }
    )
    orchestrator, store = _orchestrator(llm=llm, search=search)

    outcome = orchestrator.run(
        RunRequest(query="What is physical activity?", mode=Mode.FAST)
    )

    assert outcome.status == "completed"
    assert len(llm.calls) == 1
    store.close()


def test_prepare_terminal_transition_blocks_late_fetch_cache_write() -> None:
    store = Store(":memory:")
    search = _CacheSpySearch()
    orchestrator = Orchestrator(
        store=store,
        llm=FakeLLM(),
        retriever=Retriever(store),
        indexer=_TerminalizingIndexer(store),
        search=search,
        enforce_grounding_gate=False,
    )

    with pytest.raises(BaselineError) as exc:
        orchestrator.run(RunRequest(query="Alpha evidence", mode=Mode.FAST))

    assert exc.value.code is ErrorCode.RUN_CANCELLED
    assert search.cache_writes == 0
    assert store.list_documents() == []
    store.close()


def test_writer_is_not_started_when_retrieval_consumes_writer_reserve() -> None:
    clock = _ManualClock()
    search = _CountingSearch(clock, advance_on_fetch=25.0)
    llm = _CountingLLM()
    orchestrator, store = _orchestrator(llm=llm, search=search, clock=clock)
    session_id = store.create_session(title="reserve")

    with pytest.raises(BaselineError) as exc:
        orchestrator.run(
            RunRequest(query="Alpha evidence", mode=Mode.FAST, session_id=session_id)
        )

    assert exc.value.code is ErrorCode.RUN_TIMEOUT
    assert llm.writer_calls == 0
    run_id = store.get_latest_run_id(session_id) or ""
    assert store.get_run(run_id)["metadata"]["deadline_summary"]["writer_started_at"] is None
    store.close()


def test_stream_deadline_resets_draft_with_deadline_reason_before_error() -> None:
    clock = _ManualClock()
    llm = _DeadlineStreamLLM(clock)
    search = _CountingSearch()
    orchestrator, store = _orchestrator(llm=llm, search=search, clock=clock)
    session_id = store.create_session(title="stream-deadline")

    with pytest.raises(BaselineError) as exc:
        orchestrator.run(
            RunRequest(query="Alpha evidence", mode=Mode.FAST, session_id=session_id)
        )

    assert exc.value.code is ErrorCode.RUN_TIMEOUT
    run_id = store.get_latest_run_id(session_id) or ""
    events = store.list_events(run_id)
    reset = next(event for event in events if event["type"] == "answer_reset")
    error = next(event for event in events if event["type"] == "error")
    assert reset["payload"]["reason"] == "deadline_exceeded"
    assert reset["seq"] < error["seq"]
    store.close()


def test_success_persists_deadline_phase_and_retrieval_diagnostics() -> None:
    orchestrator, store = _orchestrator(llm=FakeLLM(), search=_CountingSearch())
    session_id = store.create_session(title="diagnostics")

    outcome = orchestrator.run(
        RunRequest(query="Alpha evidence", mode=Mode.FAST, session_id=session_id)
    )
    payload = outcome.to_dict()
    run = store.get_run(outcome.run_id)

    assert payload["phase_metrics"]
    assert payload["deadline_summary"]["queue_delay_ms"] >= 0
    assert payload["deadline_summary"]["deadline_met"] is True
    assert payload["retrieval_budget"]["results_per_query"] == 6
    assert {item["stage"] for item in payload["phase_metrics"]} >= {
        "candidate_search",
        "evidence_lane",
        "final_writer",
    }
    assert run["metadata"]["deadline_summary"] == payload["deadline_summary"]
    assert run["metadata"]["retrieval_budget"] == payload["retrieval_budget"]
    store.close()


def test_terminal_transition_is_compare_and_set() -> None:
    store = Store(":memory:")
    session_id = store.create_session(title="cas")
    run_id = store.create_run(session_id=session_id, mode="fast", query="Alpha")
    assert store.claim_queued_run(run_id) is True

    assert store.transition_run(
        run_id,
        expected_statuses={"running"},
        status="error",
        error={"error": {"code": "RUN_TIMEOUT", "message": "timeout"}},
    ) is True
    assert store.transition_run(
        run_id,
        expected_statuses={"running"},
        status="completed",
        result={"answer": {"text": "late"}},
        citation_rows=[
            {
                "document_id": "doc_late",
                "url": "https://example.test/late",
                "title": "Late",
                "snippet": "Late citation",
                "score": 1.0,
            }
        ],
    ) is False
    assert store.get_run(run_id)["status"] == "error"
    assert store.list_citations(run_id) == []
    store.close()


class _OverlapClient:
    settings = SimpleNamespace(country="US", search_lang="en")

    def __init__(self, started: threading.Event, release: threading.Event) -> None:
        self.started = started
        self.release = release

    def search(self, query: str, *, count: int = 5, options=None):  # noqa: ANN001, ANN201, ARG002
        budget = current_provider_call_budget()
        assert budget is not None
        budget.check("overlap_search")
        self.started.set()
        assert self.release.wait(1.0), "local lane did not overlap pure external search"
        return [
            {
                "title": "Alpha direct evidence",
                "url": "https://example.test/alpha",
                "description": "Alpha evidence directly supports the researched answer.",
            }
        ]

    def firecrawl_search(self, *args, **kwargs):  # noqa: ANN002, ANN003, ANN201
        raise AssertionError("provider fallback should not run")

    def scrape(self, url: str):  # noqa: ANN201
        return {"markdown": "Alpha evidence directly supports the researched answer.", "raw": {}}


class _OverlapLocalLane:
    def __init__(self, started: threading.Event, release: threading.Event) -> None:
        self.started = started
        self.release = release

    def retrieve_evidence(self, query: str, *, top_k: int):  # noqa: ANN201, ARG002
        assert self.started.wait(1.0), "external network search was not started before local retrieval"
        self.release.set()
        return []


class _ThreadGuardStore(Store):
    def __init__(self) -> None:
        self.owner_thread_id = threading.get_ident()
        super().__init__(":memory:")

    def get_run(self, run_id: str):  # noqa: ANN201
        assert threading.get_ident() == self.owner_thread_id, (
            f"background network task accessed Store.get_run({run_id})"
        )
        return super().get_run(run_id)


class _ReadyLocalLane:
    def retrieve_evidence(self, query: str, *, top_k: int):  # noqa: ANN201, ARG002
        return [
            EvidenceItem(
                chunk_id=1,
                document_id="note_alpha",
                url="",
                title="Alpha first-hand note",
                text="Alpha evidence directly supports the researched answer.",
                score=1.0,
                origin="local",
                artifact_type="library_note",
                artifact_id="note_alpha",
            )
        ][:top_k]


def test_pure_external_search_overlaps_local_lane_without_store_writes() -> None:
    started = threading.Event()
    release = threading.Event()
    store = _ThreadGuardStore()
    search = ResilientSearchAdapter(
        _OverlapClient(started, release),
        store=store,
        retriever=Retriever(store),
    )
    orchestrator = Orchestrator(
        store=store,
        llm=FakeLLM(),
        retriever=Retriever(store),
        indexer=Indexer(store),
        search=search,
        library_retriever=_OverlapLocalLane(started, release),
        enforce_grounding_gate=False,
    )

    outcome = orchestrator.run(RunRequest(query="Alpha evidence", mode=Mode.FAST))

    assert outcome.status == "completed"
    assert started.is_set() and release.is_set()
    store.close()


def test_ready_local_evidence_skips_page_fetch_but_not_external_search_lane() -> None:
    search = _CountingSearch()
    store = Store(":memory:")
    orchestrator = Orchestrator(
        store=store,
        llm=FakeLLM(),
        retriever=Retriever(store),
        indexer=Indexer(store),
        search=search,
        library_retriever=_ReadyLocalLane(),
        enforce_grounding_gate=False,
    )

    outcome = orchestrator.run(RunRequest(query="Alpha evidence", mode=Mode.FAST))

    assert outcome.status == "completed"
    assert search.search_calls == 1
    assert search.fetch_calls == []
    assert {item["origin"] for item in outcome.extra["sources"]} == {"local"}
    store.close()


class _LateSearchClient:
    settings = SimpleNamespace(country="US", search_lang="en")

    def search(self, query: str, *, count: int = 5, options=None):  # noqa: ANN001, ANN201, ARG002
        time.sleep(0.2)
        return [
            {
                "title": "Late alpha evidence",
                "url": "https://example.test/late",
                "description": "Late alpha evidence should be discarded after terminal timeout.",
            }
        ]

    def firecrawl_search(self, *args, **kwargs):  # noqa: ANN002, ANN003, ANN201
        return []

    def scrape(self, url: str):  # noqa: ANN201
        return {"markdown": "Late alpha evidence should be discarded.", "raw": {}}


def test_late_network_return_cannot_write_after_timeout_terminal() -> None:
    store = Store(":memory:")
    search = ResilientSearchAdapter(
        _LateSearchClient(),
        store=store,
        retriever=Retriever(store),
    )
    orchestrator = Orchestrator(
        store=store,
        llm=FakeLLM(),
        retriever=Retriever(store),
        indexer=Indexer(store),
        search=search,
        enforce_grounding_gate=False,
    )
    budget = Budget(1, 256, 700, 1, 12.05, top_k=6)

    with pytest.raises(BaselineError) as exc:
        orchestrator.run(RunRequest(query="Alpha evidence", mode=Mode.FAST, budget=budget))

    assert exc.value.code is ErrorCode.RUN_TIMEOUT
    run_id = store.get_latest_run_id("") or ""
    before_events = store.list_events(run_id)
    time.sleep(0.3)
    assert store.list_events(run_id) == before_events
    assert store.list_citations(run_id) == []
    assert store.list_documents() == []
    store.close()


class _OutOfOrderFetchClient:
    settings = SimpleNamespace(country="US", search_lang="en")

    def __init__(self) -> None:
        self.completed: list[str] = []
        self._lock = threading.Lock()

    def search(self, query: str, *, count: int = 5, options=None):  # noqa: ANN001, ANN201, ARG002
        return [
            {"title": "Alpha first", "url": "https://example.test/1", "description": ""},
            {"title": "Alpha second", "url": "https://example.test/2", "description": ""},
        ][:count]

    def firecrawl_search(self, *args, **kwargs):  # noqa: ANN002, ANN003, ANN201
        raise AssertionError("provider fallback should not run")

    def scrape(self, url: str):  # noqa: ANN201
        suffix = url.rsplit("/", 1)[-1]
        if suffix == "1":
            time.sleep(0.05)
        with self._lock:
            self.completed.append(suffix)
        return {
            "markdown": f"Alpha evidence from ranked candidate {suffix} directly supports the answer.",
            "raw": {"metadata": {"canonicalUrl": url}},
        }


def test_out_of_order_fetch_completion_keeps_ranked_commit_and_source_order() -> None:
    store = Store(":memory:")
    client = _OutOfOrderFetchClient()
    search = ResilientSearchAdapter(client, store=store, retriever=Retriever(store))
    orchestrator = Orchestrator(
        store=store,
        llm=FakeLLM(),
        retriever=Retriever(store),
        indexer=Indexer(store),
        search=search,
        enforce_grounding_gate=False,
    )
    budget = Budget(1, 256, 800, 3, 60, top_k=6)

    outcome = orchestrator.run(
        RunRequest(query="Alpha evidence", mode=Mode.MULTIPLE, budget=budget)
    )

    assert client.completed == ["2", "1"]
    assert [source["url"] for source in outcome.to_dict()["sources"]] == [
        "https://example.test/1",
        "https://example.test/2",
    ]
    assert outcome.to_dict()["retrieval_budget"]["fetches_used"] == 2
    store.close()


class _CountingCacheClient:
    settings = SimpleNamespace(country="US", search_lang="en")

    def __init__(self) -> None:
        self.search_calls = 0

    def search(self, query: str, *, count: int = 5, options=None):  # noqa: ANN001, ANN201, ARG002
        self.search_calls += 1
        return [
            {
                "title": "Alpha authoritative evidence",
                "url": "https://example.test/cache-alpha",
                "description": "Alpha evidence directly supports the current researched answer.",
            }
        ]

    def firecrawl_search(self, *args, **kwargs):  # noqa: ANN002, ANN003, ANN201
        raise AssertionError("provider fallback should not run")

    def scrape(self, url: str):  # noqa: ANN201
        return {"markdown": "Alpha evidence directly supports the answer.", "raw": {}}


def test_stable_warm_run_uses_locale_v3_cache_but_latest_attempts_live_first() -> None:
    store = Store(":memory:")
    client = _CountingCacheClient()
    search = ResilientSearchAdapter(client, store=store, retriever=Retriever(store))
    orchestrator = Orchestrator(
        store=store,
        llm=FakeLLM(),
        retriever=Retriever(store),
        indexer=Indexer(store),
        search=search,
        enforce_grounding_gate=False,
    )

    orchestrator.run(RunRequest(query="Alpha evidence", mode=Mode.FAST))
    assert client.search_calls == 1
    orchestrator.run(RunRequest(query="Alpha evidence", mode=Mode.FAST))
    assert client.search_calls == 1

    original_read_cache = search.read_search_cache

    def fail_if_latest_cache_is_read(query: str, **kwargs):  # noqa: ANN003, ANN201
        if query == "latest Alpha evidence":
            raise AssertionError("latest query read cache before attempting live")
        return original_read_cache(query, **kwargs)

    search.read_search_cache = fail_if_latest_cache_is_read  # type: ignore[method-assign]
    orchestrator.run(RunRequest(query="latest Alpha evidence", mode=Mode.FAST))
    assert client.search_calls >= 2
    store.close()
