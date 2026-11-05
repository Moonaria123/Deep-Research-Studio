"""DR-ITER-090: per-run advanced search options."""

from __future__ import annotations

import json
from urllib.request import Request

import pytest

from deepresearch_core.baseline.adapters import Document, ResilientSearchAdapter, SearchHit
from deepresearch_core.baseline.budget import Budget, Mode, RunRequest
from deepresearch_core.baseline.indexing import Indexer
from deepresearch_core.baseline.evidence_eligibility import parse_research_intent
from deepresearch_core.baseline.orchestrator import Orchestrator, _source_constrained_subqueries
from deepresearch_core.baseline.retrieval import Retriever
from deepresearch_core.baseline.search_options import SearchOptions, normalize_search_options
from deepresearch_core.baseline.store import Store
from deepresearch_core.standalone.search import SearchClient, SearchSettings
from tests.baseline.fakes import FakeLLM


def test_search_options_normalize_and_round_trip() -> None:
    options = normalize_search_options(
        {
            "include_domains": ["Example.com", "example.com", "docs.example.com"],
            "exclude_domains": ["spam.example"],
            "freshness": "PW",
            "extra_subqueries": ["alpha", "Alpha", "beta"],
        }
    )

    assert options == SearchOptions(
        include_domains=("example.com", "docs.example.com"),
        exclude_domains=("spam.example",),
        freshness="pw",
        extra_subqueries=("alpha", "beta"),
    )
    assert normalize_search_options(options.to_dict()) == options


def test_search_options_rejects_invalid_domains() -> None:
    with pytest.raises(ValueError):
        normalize_search_options({"include_domains": ["https://example.com/path"]})
    with pytest.raises(ValueError):
        normalize_search_options({"include_domains": ["example.com"], "exclude_domains": ["example.com"]})


def test_brave_search_rewrites_domains_and_freshness() -> None:
    seen: dict[str, object] = {}

    def transport(request: Request) -> dict:
        seen["url"] = request.full_url
        return {
            "web": {
                "results": [
                    {"title": "Result", "url": "https://example.com/a", "description": "body"},
                ]
            }
        }

    client = SearchClient(SearchSettings(brave_api_key="brave"), transport=transport)
    options = SearchOptions(
        include_domains=("example.com",),
        exclude_domains=("spam.example",),
        freshness="pm",
    )

    hits = client.search("quantum", count=3, options=options)

    assert hits[0]["url"] == "https://example.com/a"
    url = str(seen["url"])
    assert "freshness=pm" in url
    assert "site%3Aexample.com" in url
    assert "-site%3Aspam.example" in url


class _RecordingSearchClient:
    settings = SearchSettings(brave_api_key="brave", firecrawl_api_key="fire")

    def __init__(self) -> None:
        self.options_seen: list[SearchOptions | None] = []

    def search(self, query: str, *, count: int = 5, options: SearchOptions | None = None):  # noqa: ANN201
        self.options_seen.append(options)
        return [{"title": "Hit", "url": "https://example.com/a", "description": query}]

    def firecrawl_search(self, query: str, *, count: int = 5, options: SearchOptions | None = None):  # noqa: ANN201
        self.options_seen.append(options)
        return []

    def scrape(self, url: str):  # noqa: ANN201
        return {"markdown": "Grounded evidence for Alpha topic.", "raw": {}}


def test_resilient_cache_key_includes_options() -> None:
    store = Store(":memory:")
    client = _RecordingSearchClient()
    adapter = ResilientSearchAdapter(client, store=store, retriever=Retriever(store))
    options = SearchOptions(include_domains=("example.com",))

    assert adapter.search_cache_key("alpha", count=2) != adapter.search_cache_key("alpha", count=2, options=options)
    adapter.search("alpha", count=2, options=options)

    assert client.options_seen == [options]
    assert store.cache_get(adapter.search_cache_key("alpha", count=2, options=options)) is not None
    store.close()


class _RecordingSearch:
    name = "recording-search"

    def __init__(self) -> None:
        self.calls: list[tuple[str, SearchOptions | None]] = []

    def search(self, query: str, *, count: int = 5, options: SearchOptions | None = None):
        self.calls.append((query, options))
        return [SearchHit(title=query, url=f"https://example.com/{len(self.calls)}", snippet="Alpha evidence", rank=1)]

    def fetch(self, url: str) -> Document:
        return Document(url=url, title=url, text="Alpha evidence with grounded citation text.", content_type="web")


def test_multiple_mode_keeps_original_query_then_caps_expanded_subqueries() -> None:
    store = Store(":memory:")
    search = _RecordingSearch()
    options = SearchOptions(extra_subqueries=("extra one", "llm two", "extra two"))
    llm = FakeLLM(scripted={"Break the research question": "llm one\nllm two"})
    orch = Orchestrator(
        store=store,
        llm=llm,
        retriever=Retriever(store),
        indexer=Indexer(store),
        search=search,
        enforce_grounding_gate=False,
    )
    sid = store.create_session(title="t")
    request = RunRequest(
        query="Alpha",
        mode=Mode.MULTIPLE,
        session_id=sid,
        overrides={"max_searches": 3, "top_k": 3},
        search_options=options,
    )

    orch.run(request)

    assert [query for query, _ in search.calls] == ["Alpha", "llm one", "llm two"]
    assert all(seen_options is options for _, seen_options in search.calls)
    run = store.get_run(store.get_latest_run_id(sid) or "")
    assert run
    assert run["metadata"]["search_options"] == options.to_dict()
    assert json.dumps(run["result"], ensure_ascii=False)
    store.close()


def test_authoritative_comparison_prioritizes_each_subject_before_broad_queries() -> None:
    intent = parse_research_intent(
        "Compare Alpha Engine and Beta Search using authoritative sources. "
        "Answer in exactly 4 concise sentences with citations."
    )

    assert _source_constrained_subqueries(
        intent,
        [
            "Compare Alpha Engine and Beta Search.",
            "Alpha Engine versus Beta Search performance and use cases",
            "Alpha Engine indexing features compared with Beta Search",
        ],
        count=4,
    ) == [
        "Alpha Engine official documentation",
        "Beta Search official documentation",
        "Compare Alpha Engine and Beta Search.",
        "Alpha Engine versus Beta Search performance and use cases",
    ]


def test_unconstrained_or_noncomparison_queries_keep_existing_subquery_order() -> None:
    unconstrained = parse_research_intent("Compare Alpha Engine and Beta Search.")
    authoritative_definition = parse_research_intent(
        "Explain Alpha Engine using authoritative sources."
    )
    existing = ["Compare Alpha Engine and Beta Search.", "Alpha details"]

    assert _source_constrained_subqueries(unconstrained, existing, count=4) == existing
    assert _source_constrained_subqueries(authoritative_definition, existing, count=4) == existing
