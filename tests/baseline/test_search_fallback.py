"""DR-ITER-071: resilient web search fallback and cache contracts."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from urllib.request import Request

from deepresearch_core.baseline.adapters import Document, ResilientSearchAdapter, SearchHit
from deepresearch_core.baseline.budget import Mode, RunRequest
from deepresearch_core.baseline.indexing import Indexer
from deepresearch_core.baseline.library_index import LibraryIndexer
from deepresearch_core.baseline.memory import MemoryEngine
from deepresearch_core.baseline.orchestrator import Orchestrator
from deepresearch_core.baseline.retrieval import Retriever
from deepresearch_core.baseline.store import Store
from deepresearch_core.standalone.llm import ProviderRequestError
from deepresearch_core.standalone.search import SearchClient, SearchSettings


def test_firecrawl_search_uses_v2_endpoint_and_normalizes_hits() -> None:
    seen: dict[str, object] = {}

    def transport(request: Request) -> dict:
        seen["url"] = request.full_url
        seen["headers"] = dict(request.header_items())
        seen["payload"] = json.loads((request.data or b"{}").decode("utf-8"))
        return {
            "success": True,
            "data": {
                "web": [
                    {
                        "title": "Fire result",
                        "url": "https://example.com/fire",
                        "description": "from firecrawl",
                        "publishedDate": "2026-07-01T00:00:00Z",
                        "metadata": {"updatedAt": "2026-07-02T00:00:00Z"},
                    }
                ]
            },
        }

    client = SearchClient(SearchSettings(firecrawl_api_key="fc-test"), transport=transport)

    hits = client.firecrawl_search("fallback query", count=3)

    assert seen["url"] == "https://api.firecrawl.dev/v2/search"
    assert seen["payload"] == {"query": "fallback query", "limit": 3, "sources": ["web"], "country": "US"}
    assert hits == [
        {
            "id": "F1",
            "title": "Fire result",
            "url": "https://example.com/fire",
            "description": "from firecrawl",
            "source": "example.com",
            "published_at": "2026-07-01T00:00:00Z",
            "updated_at": "2026-07-02T00:00:00Z",
        }
    ]


class SwitchingSearchClient:
    settings = SearchSettings(brave_api_key="brave", firecrawl_api_key="fire")

    def __init__(self) -> None:
        self.mode = "firecrawl"
        self.search_calls = 0
        self.firecrawl_calls = 0

    def search(self, query: str, *, count: int = 5, options=None):  # noqa: ANN201, ANN001
        self.search_calls += 1
        if self.mode == "brave":
            return [{"title": "Brave", "url": "https://example.com/brave", "description": "brave hit"}]
        raise ProviderRequestError("brave down", retryable=True)

    def firecrawl_search(self, query: str, *, count: int = 5, options=None):  # noqa: ANN201, ANN001
        self.firecrawl_calls += 1
        if self.mode == "firecrawl":
            return [{"title": "Fire", "url": "https://example.com/fire", "description": "fire hit"}]
        raise ProviderRequestError("firecrawl down", retryable=True)

    def scrape(self, url: str):  # noqa: ANN201
        return {"markdown": f"Fetched content for {url}", "raw": {}}


def test_brave_failure_falls_back_to_firecrawl_and_writes_cache() -> None:
    store = Store(":memory:")
    client = SwitchingSearchClient()
    adapter = ResilientSearchAdapter(client, store=store, retriever=Retriever(store))

    hits = adapter.search("alpha", count=2)

    assert hits[0].url == "https://example.com/fire"
    assert hits[0].metadata["fallback_tier"] == "firecrawl_search"
    assert client.search_calls == 1
    assert client.firecrawl_calls == 1
    assert store.cache_get(adapter.search_cache_key("alpha", count=2)) is not None
    assert adapter.last_search_status()["live_providers_down"] is True
    assert adapter.last_search_status()["failed_providers"] == ["brave"]
    store.close()


def test_normal_empty_provider_responses_are_not_reported_as_provider_down() -> None:
    class EmptySearchClient(SwitchingSearchClient):
        def search(self, query: str, *, count: int = 5, options=None):  # noqa: ANN201, ANN001
            self.search_calls += 1
            return []

        def firecrawl_search(self, query: str, *, count: int = 5, options=None):  # noqa: ANN201, ANN001
            self.firecrawl_calls += 1
            return []

    store = Store(":memory:")
    adapter = ResilientSearchAdapter(EmptySearchClient(), store=store, retriever=Retriever(store))

    result = adapter.search_with_outcome("nothing here", count=2)

    assert result.hits == ()
    assert result.status["live_providers_down"] is False
    assert result.status["empty_providers"] == ["brave", "firecrawl_search"]
    assert result.status["failed_providers"] == []
    store.close()


def test_exact_source_times_survive_normalization_and_search_cache() -> None:
    class TemporalSearchClient(SwitchingSearchClient):
        def search(self, query: str, *, count: int = 5, options=None):  # noqa: ANN201, ANN001
            self.search_calls += 1
            if self.mode == "fail":
                raise ProviderRequestError("brave down", retryable=True)
            return [
                {
                    "title": "Dated",
                    "url": "https://example.com/dated",
                    "description": "dated evidence",
                    "published_at": "2026-07-01T00:00:00Z",
                    "updatedAt": "2026-07-02T00:00:00Z",
                    "effectiveDate": "2026-06-30",
                    "age": "2 days ago",
                }
            ]

        def firecrawl_search(self, query: str, *, count: int = 5, options=None):  # noqa: ANN201, ANN001
            self.firecrawl_calls += 1
            if self.mode == "fail":
                raise ProviderRequestError("firecrawl down", retryable=True)
            return []

    store = Store(":memory:")
    client = TemporalSearchClient()
    adapter = ResilientSearchAdapter(client, store=store, retriever=Retriever(store))

    live = adapter.search("dated", count=1)[0]
    expected_published = datetime(2026, 7, 1, tzinfo=timezone.utc).timestamp()
    expected_updated = datetime(2026, 7, 2, tzinfo=timezone.utc).timestamp()
    expected_effective = datetime(2026, 6, 30, tzinfo=timezone.utc).timestamp()
    assert live.metadata["published_at"] == expected_published
    assert live.metadata["updated_at"] == expected_updated
    assert live.metadata["effective_at"] == expected_effective
    assert isinstance(live.metadata["retrieved_at"], float)
    assert live.metadata["source_age"] == "2 days ago"

    client.mode = "fail"
    cached = adapter.search("dated", count=1)[0]
    assert cached.metadata["published_at"] == expected_published
    assert cached.metadata["updated_at"] == expected_updated
    assert cached.metadata["effective_at"] == expected_effective
    assert cached.metadata["retrieved_at"] == live.metadata["retrieved_at"]
    store.close()


def test_index_stats_record_only_providers_that_actually_executed() -> None:
    store = Store(":memory:")
    client = SwitchingSearchClient()
    client.mode = "brave"
    adapter = ResilientSearchAdapter(client, store=store, retriever=Retriever(store))

    stats = Indexer(store).index_search_results(adapter, "alpha", count=1)

    assert stats
    assert stats[0]["search_provider"] == "brave"
    assert stats[0]["fetch_provider"] == "firecrawl"
    store.close()


def test_cache_is_used_when_web_providers_fail() -> None:
    store = Store(":memory:")
    client = SwitchingSearchClient()
    client.mode = "brave"
    adapter = ResilientSearchAdapter(client, store=store, retriever=Retriever(store))
    assert adapter.search("alpha", count=1)[0].metadata["fallback_tier"] == "brave"

    client.mode = "fail"
    hits = adapter.search("alpha", count=1)

    assert hits[0].url == "https://example.com/brave"
    assert hits[0].metadata["fallback_tier"] == "cache"
    assert adapter.last_search_status()["live_providers_down"] is True
    assert adapter.last_search_status()["fallback_tier"] == "cache"
    store.close()


def test_t48_recovery_skips_irrelevant_provider_and_uses_next_relevant_tier() -> None:
    class RelevanceSwitchingClient(SwitchingSearchClient):
        def search(self, query: str, *, count: int = 5, options=None):  # noqa: ANN201, ANN001
            self.search_calls += 1
            return [
                {
                    "title": "SQLCipher encryption",
                    "url": "https://example.com/sqlcipher",
                    "description": "SQLCipher adds transparent database encryption to SQLite.",
                }
            ]

        def firecrawl_search(self, query: str, *, count: int = 5, options=None):  # noqa: ANN201, ANN001
            self.firecrawl_calls += 1
            return [
                {
                    "title": "SQLite FTS5 extension",
                    "url": "https://www.sqlite.org/fts5.html",
                    "description": "SQLite FTS5 provides full-text search virtual tables and MATCH queries.",
                }
            ]

    store = Store(":memory:")
    client = RelevanceSwitchingClient()
    adapter = ResilientSearchAdapter(client, store=store, retriever=Retriever(store))

    result = adapter.search_with_outcome(
        "SQLite FTS5 authoritative source",
        count=2,
        recovery=True,
        relevance_query="What is SQLite FTS5? Use one authoritative source.",
    )

    assert [hit.url for hit in result.hits] == ["https://www.sqlite.org/fts5.html"]
    assert result.status["fallback_tier"] == "firecrawl_search"
    assert result.status["irrelevant_providers"] == ["brave"]
    assert result.status["providers_attempted"] == ["brave", "firecrawl_search"]
    store.close()


def test_t48_recovery_does_not_stop_on_a_relevant_but_non_authoritative_tier() -> None:
    class AuthoritySwitchingClient(SwitchingSearchClient):
        def search(self, query: str, *, count: int = 5, options=None):  # noqa: ANN201, ANN001
            self.search_calls += 1
            return [
                {
                    "title": "SQLite FTS5 overview",
                    "url": "https://sqlite.example.com/fts5",
                    "description": "SQLite FTS5 provides full-text search virtual tables.",
                }
            ]

        def firecrawl_search(self, query: str, *, count: int = 5, options=None):  # noqa: ANN201, ANN001
            self.firecrawl_calls += 1
            return [
                {
                    "title": "SQLite FTS5 Extension",
                    "url": "https://www.sqlite.org/fts5.html",
                    "description": "SQLite FTS5 provides full-text search virtual tables.",
                }
            ]

    store = Store(":memory:")
    client = AuthoritySwitchingClient()
    adapter = ResilientSearchAdapter(client, store=store, retriever=Retriever(store))

    result = adapter.search_with_outcome(
        "SQLite FTS5 authoritative official documentation",
        count=2,
        recovery=True,
        relevance_query="What is SQLite FTS5? Use one authoritative source.",
    )

    assert [hit.url for hit in result.hits] == ["https://www.sqlite.org/fts5.html"]
    assert result.status["fallback_tier"] == "firecrawl_search"
    assert result.status["providers_attempted"] == ["brave", "firecrawl_search"]
    store.close()


def test_t48_recovery_does_not_stop_on_an_unknown_only_tier() -> None:
    class UnknownSwitchingClient(SwitchingSearchClient):
        def search(self, query: str, *, count: int = 5, options=None):  # noqa: ANN201, ANN001
            self.search_calls += 1
            return [
                {
                    "title": "Automobile coverage",
                    "url": "https://example.com/coverage",
                    "description": "Automobile coverage protects drivers after collisions.",
                }
            ]

        def firecrawl_search(self, query: str, *, count: int = 5, options=None):  # noqa: ANN201, ANN001
            self.firecrawl_calls += 1
            return [
                {
                    "title": "SQLite FTS5 Extension",
                    "url": "https://www.sqlite.org/fts5.html",
                    "description": "SQLite FTS5 provides full-text search virtual tables.",
                }
            ]

    store = Store(":memory:")
    client = UnknownSwitchingClient()
    adapter = ResilientSearchAdapter(client, store=store, retriever=Retriever(store))

    result = adapter.search_with_outcome(
        "SQLite FTS5 authoritative official documentation",
        count=2,
        recovery=True,
        relevance_query="What is SQLite FTS5? Use one authoritative source.",
    )

    assert [hit.url for hit in result.hits] == ["https://www.sqlite.org/fts5.html"]
    assert result.status["providers_attempted"] == ["brave", "firecrawl_search"]
    store.close()


def test_t48_recovery_retains_the_first_unknown_tier_after_exhausting_the_chain() -> None:
    class AllUnknownClient(SwitchingSearchClient):
        def search(self, query: str, *, count: int = 5, options=None):  # noqa: ANN201, ANN001
            self.search_calls += 1
            return [
                {
                    "title": "Automobile coverage",
                    "url": "https://example.com/coverage",
                    "description": "Automobile coverage protects drivers after collisions.",
                }
            ]

        def firecrawl_search(self, query: str, *, count: int = 5, options=None):  # noqa: ANN201, ANN001
            self.firecrawl_calls += 1
            return [
                {
                    "title": "Vehicle policy",
                    "url": "https://example.org/policy",
                    "description": "A vehicle policy can include collision coverage.",
                }
            ]

    store = Store(":memory:")
    client = AllUnknownClient()
    adapter = ResilientSearchAdapter(client, store=store, retriever=Retriever(store))

    result = adapter.search_with_outcome(
        "SQLite FTS5 authoritative official documentation",
        count=2,
        recovery=True,
        relevance_query="What is SQLite FTS5? Use one authoritative source.",
    )

    assert [hit.url for hit in result.hits] == ["https://example.com/coverage"]
    assert result.status["providers_attempted"] == ["brave", "firecrawl_search"]
    store.close()


def test_t48_recovery_can_fetch_a_named_source_match_without_exhausting_the_chain() -> None:
    class NamedSourceClient(SwitchingSearchClient):
        def search(self, query: str, *, count: int = 5, options=None):  # noqa: ANN201, ANN001
            self.search_calls += 1
            return [
                {
                    "title": "RFC 9110 HTTP Semantics",
                    "url": "https://datatracker.ietf.org/doc/html/rfc9110",
                    "description": "A 404 status code means the target representation was not found.",
                }
            ]

    store = Store(":memory:")
    client = NamedSourceClient()
    adapter = ResilientSearchAdapter(client, store=store, retriever=Retriever(store))

    result = adapter.search_with_outcome(
        "HTTP status 404 IETF documentation",
        count=2,
        recovery=True,
        relevance_query="According to IETF documentation, what does HTTP status 404 mean?",
    )

    assert [hit.url for hit in result.hits] == [
        "https://datatracker.ietf.org/doc/html/rfc9110"
    ]
    assert result.status["providers_attempted"] == ["brave"]
    assert client.firecrawl_calls == 0
    store.close()


def test_t48_recovery_bypasses_an_old_search_cache_entry() -> None:
    store = Store(":memory:")
    client = SwitchingSearchClient()
    client.mode = "brave"
    adapter = ResilientSearchAdapter(client, store=store, retriever=Retriever(store))
    assert adapter.search("alpha", count=1)

    client.mode = "fail"
    result = adapter.search_with_outcome(
        "alpha",
        count=1,
        recovery=True,
        relevance_query="alpha",
    )

    assert result.hits == ()
    assert result.status["fallback_tier"] == "none"
    assert result.status["providers_attempted"] == ["brave", "firecrawl_search"]
    store.close()


def test_indexed_external_is_citable_last_resort_but_library_is_not() -> None:
    store = Store(":memory:")
    Indexer(store).index_document(
        Document(
            url="https://local.example/web",
            title="Local web",
            text="Alpha fallback evidence from an indexed web document.",
            content_type="web",
            metadata={
                "published_at": 1_700_000_000.0,
                "updated_at": 1_700_000_100.0,
                "effective_at": 1_699_999_900.0,
                "retrieved_at": 1_700_000_200.0,
            },
        )
    )
    LibraryIndexer(store, indexer=Indexer(store)).index_note(note_id="note_1", content="Alpha private library note")
    client = SwitchingSearchClient()
    client.mode = "fail"
    adapter = ResilientSearchAdapter(client, store=store, retriever=Retriever(store))

    hits = adapter.search("Alpha", count=5)

    assert hits
    assert hits[0].metadata["fallback_tier"] == "indexed_external"
    assert hits[0].metadata["published_at"] == 1_700_000_000.0
    assert hits[0].metadata["updated_at"] == 1_700_000_100.0
    assert hits[0].metadata["effective_at"] == 1_699_999_900.0
    assert hits[0].metadata["retrieved_at"] == 1_700_000_200.0
    assert hits[0].url.startswith("https://")
    assert all(not hit.url.startswith("local://library/") for hit in hits)
    assert adapter.last_search_status()["live_providers_down"] is True
    assert adapter.last_search_status()["fallback_tier"] == "indexed_external"
    store.close()


def test_library_only_fallback_does_not_create_web_hits() -> None:
    store = Store(":memory:")
    LibraryIndexer(store, indexer=Indexer(store)).index_note(note_id="note_1", content="Alpha private library note")
    client = SwitchingSearchClient()
    client.mode = "fail"
    adapter = ResilientSearchAdapter(client, store=store, retriever=Retriever(store))

    hits = adapter.search("Alpha", count=5)

    assert hits == []
    assert adapter.last_search_status()["live_providers_down"] is True
    assert adapter.last_search_status()["fallback_tier"] == "none"
    store.close()


class DegradedFakeSearch:
    name = "degraded-fake"

    def __init__(self) -> None:
        self._pages = {"https://example.com/a": "Alpha web evidence from fallback."}

    def search(self, query: str, *, count: int = 5, options=None):  # noqa: ANN201, ANN001
        return [
            SearchHit(
                title="Alpha",
                url="https://example.com/a",
                snippet="Alpha web evidence from fallback.",
                source="example.com",
                metadata={"fallback_tier": "indexed_external"},
            )
        ]

    def fetch(self, url: str) -> Document:
        return Document(url=url, title="Alpha", text=self._pages[url], content_type="web")

    def last_search_status(self) -> dict[str, object]:
        return {"live_providers_down": True, "fallback_tier": "indexed_external"}


def test_orchestrator_emits_search_degraded_event() -> None:
    from tests.baseline.fakes import FakeLLM

    store = Store(":memory:")
    events: list[dict] = []
    orch = Orchestrator(
        store=store,
        llm=FakeLLM(),
        retriever=Retriever(store),
        indexer=Indexer(store),
        search=DegradedFakeSearch(),
        memory=MemoryEngine(store),
        enforce_grounding_gate=False,
    )
    sid = store.create_session(title="t")

    orch.run(RunRequest(query="Alpha", mode=Mode.FAST, session_id=sid), sink=events.append)

    degraded = [event for event in events if event["type"] == "search_degraded"]
    assert degraded
    assert degraded[0]["payload"] == {"code": "SEARCH_PROVIDER_DOWN", "fallback_tier": "indexed_external"}
    store.close()
