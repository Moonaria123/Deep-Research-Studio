"""T49 demand-driven retrieval foundation contracts."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from deepresearch_core.baseline.adapters import Document, ResilientSearchAdapter
from deepresearch_core.baseline.evidence_eligibility import parse_research_intent
from deepresearch_core.baseline.external_candidates import (
    bind_fetched_content,
    build_external_candidate,
    canonicalize_external_url,
    fetched_body_may_contain_focused_evidence,
    select_fetched_evidence_excerpt,
    snippet_evidence,
)
from deepresearch_core.baseline.indexing import ChunkingConfig, Indexer
from deepresearch_core.baseline.retrieval import Retriever
from deepresearch_core.baseline.store import Store


class _NetworkClient:
    settings = SimpleNamespace(country="US", search_lang="en")

    def search(self, query: str, *, count: int = 5, options=None):  # noqa: ANN001, ANN201, ARG002
        return [
            {
                "title": "SQLite FTS5 Extension",
                "url": "https://www.sqlite.org/fts5.html",
                "description": "SQLite FTS5 provides full-text search virtual tables and MATCH queries.",
            }
        ][:count]

    def firecrawl_search(self, query: str, *, count: int = 5, options=None):  # noqa: ANN001, ANN201, ARG002
        raise AssertionError("Brave's usable result must stop provider fallback")

    def scrape(self, url: str):  # noqa: ANN201
        return {
            "markdown": "SQLite FTS5 provides full-text search virtual tables and MATCH queries.",
            "raw": {"metadata": {"canonicalUrl": url}},
        }


class _ForbiddenStore:
    def __getattr__(self, name: str):  # noqa: ANN204
        raise AssertionError(f"pure network path accessed Store.{name}")


def _intent():  # noqa: ANN202
    return parse_research_intent("What is SQLite FTS5? Use one authoritative source.")


def test_external_candidate_canonicalizes_and_only_materializes_direct_snippets() -> None:
    intent = _intent()
    candidate = build_external_candidate(
        plan_id="primary",
        query=intent.research_focus,
        title="SQLite FTS5 Extension",
        url="https://WWW.SQLite.org:443/fts5.html?utm_source=test#overview",
        snippet="SQLite FTS5 provides full-text search virtual tables and MATCH queries.",
        intent=intent,
        rank=0,
    )

    assert canonicalize_external_url(candidate.url) == "https://www.sqlite.org/fts5.html"
    assert candidate.canonical_url == "https://www.sqlite.org/fts5.html"
    assert candidate.provenance_root == candidate.canonical_url
    assert candidate.prefetch_decision == "candidate"
    assert snippet_evidence(candidate) is not None

    truncated = build_external_candidate(
        plan_id="primary",
        query=intent.research_focus,
        title="SQLite FTS5 Extension",
        url="https://www.sqlite.org/fts5.html",
        snippet="SQLite FTS5 provides full-text search...",
        intent=intent,
    )
    assert truncated.prefetch_decision == "unknown"
    assert "snippet_truncated" in truncated.reason_codes
    assert snippet_evidence(truncated) is None

    noncontiguous = build_external_candidate(
        plan_id="primary",
        query=intent.research_focus,
        title="SQLite FTS5 Extension",
        url="https://www.sqlite.org/fts5.html",
        snippet="SQLite FTS5 provides full-text search... PEP 428 describes object-oriented paths.",
        intent=intent,
    )
    assert noncontiguous.prefetch_decision == "unknown"
    assert "snippet_truncated" in noncontiguous.reason_codes
    assert snippet_evidence(noncontiguous) is None


def test_external_candidate_keeps_domain_mismatch_unknown_or_rejected() -> None:
    intent = _intent()
    candidate = build_external_candidate(
        plan_id="primary",
        query=intent.research_focus,
        title="SQLite FTS5 Extension",
        url="https://www.sqlite.org/fts5.html",
        canonical_url="https://redirect.example/fts5",
        snippet="SQLite FTS5 provides full-text search virtual tables.",
        intent=intent,
    )
    assert candidate.prefetch_decision != "candidate"
    assert "canonical_domain_mismatch" in candidate.reason_codes
    assert snippet_evidence(candidate) is None

    safe_search_candidate = build_external_candidate(
        plan_id="primary",
        query=intent.research_focus,
        title="SQLite FTS5 Extension",
        url="https://www.sqlite.org/fts5.html",
        snippet="SQLite FTS5 provides full-text search virtual tables.",
        intent=intent,
    )
    fetched = bind_fetched_content(
        safe_search_candidate,
        intent=intent,
        document_url="https://redirect.example/fts5",
        title="SQLite FTS5 Extension",
        text="SQLite FTS5 provides full-text search virtual tables.",
    )
    assert fetched.prefetch_decision != "candidate"
    assert "canonical_domain_mismatch" in fetched.reason_codes


def test_named_source_attribution_requires_matching_fetched_body() -> None:
    intent = parse_research_intent(
        "According to WHO, what is the definition of physical activity?"
    )
    direct = build_external_candidate(
        plan_id="primary",
        query=intent.research_focus,
        title="Physical activity",
        url="https://www.who.int/news-room/fact-sheets/detail/physical-activity",
        snippet="WHO defines physical activity as any bodily movement produced by skeletal muscles.",
        intent=intent,
    )
    third_party = build_external_candidate(
        plan_id="primary",
        query=intent.research_focus,
        title="Physical activity",
        url="https://en.wikipedia.org/wiki/Physical_activity",
        snippet="WHO defines physical activity as any bodily movement produced by skeletal muscles.",
        intent=intent,
        rank=1,
    )

    assert direct.prefetch_decision == "unknown"
    assert "source_attribution_identity_matched" in direct.reason_codes
    assert "source_attribution_requires_fetched_body" in direct.reason_codes
    assert snippet_evidence(direct) is None
    assert third_party.prefetch_decision == "rejected"
    assert "source_attribution_identity_mismatch" in third_party.reason_codes

    fetched = bind_fetched_content(
        direct,
        intent=intent,
        document_url=direct.url,
        title=direct.title,
        text="WHO defines physical activity as any bodily movement produced by skeletal muscles.",
    )
    assert fetched.prefetch_decision == "candidate"
    assert "source_attribution_identity_matched" in fetched.reason_codes


def test_named_source_identity_matching_covers_t51_provider_matrix_publishers() -> None:
    cases = (
        (
            "According to NASA, what causes Earth's seasons?",
            "https://science.nasa.gov/earth/facts/",
            "NASA Earth facts",
            "NASA explains that Earth's tilted axis causes the seasons.",
        ),
        (
            "According to the Python documentation, what problem does pathlib solve?",
            "https://docs.python.org/3/library/pathlib.html",
            "pathlib — Object-oriented filesystem paths",
            "Python pathlib solves filesystem path handling by offering object-oriented classes for operating systems.",
        ),
        (
            "According to the World Bank, what does GDP measure?",
            "https://www.worldbank.org/en/publication/world-development-indicators",
            "GDP metadata",
            "The World Bank says GDP measures the monetary value of goods and services produced in an economy.",
        ),
        (
            "According to CDC guidance, how much sleep should most adults get?",
            "https://www.cdc.gov/sleep/about/index.html",
            "About sleep",
            "CDC guidance says most adults should get at least seven hours of sleep each night.",
        ),
        (
            "According to IETF documentation, what does HTTP status 404 mean?",
            "https://datatracker.ietf.org/doc/html/rfc9110",
            "HTTP Semantics",
            "IETF HTTP status 404 means the origin server did not find a current representation for the target resource.",
        ),
        (
            "According to NIST, what is the purpose of the Cybersecurity Framework?",
            "https://www.nist.gov/cyberframework",
            "Cybersecurity Framework",
            "NIST says the Cybersecurity Framework helps organizations understand and reduce cybersecurity risk.",
        ),
    )

    for rank, (query, url, title, text) in enumerate(cases):
        intent = parse_research_intent(query)
        candidate = build_external_candidate(
            plan_id=f"matrix-{rank}",
            query=intent.research_focus,
            title=title,
            url=url,
            snippet=text,
            intent=intent,
        )
        assert candidate.prefetch_decision == "unknown", query
        assert "source_attribution_identity_matched" in candidate.reason_codes, query
        fetched = bind_fetched_content(
            candidate,
            intent=intent,
            document_url=url,
            title=title,
            text=text,
        )
        assert fetched.prefetch_decision == "candidate", query


def test_fetched_evidence_excerpt_starts_at_the_relevant_passage_not_page_boilerplate() -> None:
    intent = parse_research_intent(
        "According to NASA, what causes Earth's seasons? Answer in two cited sentences."
    )
    pieces = (
        "Navigation menu. " * 120,
        (
            "Cookie controls. Earth spins around the Sun, and our planet's slight tilt creates seasons. "
            "NASA missions also observe that the same tilt influences seasonal differences in space weather."
        ),
        "Unrelated mission directory and footer links.",
    )

    excerpt, piece_index = select_fetched_evidence_excerpt(intent, pieces)

    assert piece_index == 1
    assert excerpt.startswith("Earth spins around the Sun")
    assert "Navigation menu" not in excerpt[:240]


def test_fetched_excerpt_does_not_promote_a_topic_heading_as_purpose_evidence() -> None:
    intent = parse_research_intent(
        "According to NIST, what is the purpose of the Cybersecurity Framework? "
        "Answer in two cited sentences."
    )
    fetched_text = (
        "# Cybersecurity Framework\n"
        "Helping organizations to better understand and improve their management of cybersecurity risk.\n"
        "## CSF 2.0\n"
        "For industry, government, and organizations to reduce cybersecurity risks.\n"
        "## Contacts\n"
        "### For further information and questions about the Cybersecurity Framework"
    )

    excerpt, piece_index = select_fetched_evidence_excerpt(
        intent,
        [fetched_text],
        document_url="https://www.nist.gov/cyberframework",
        document_title="Cybersecurity Framework | NIST",
    )

    assert piece_index == 0
    assert excerpt != "# Cybersecurity Framework"
    assert "Helping organizations" in excerpt[:240]


def test_purpose_excerpt_rejects_a_flattened_footer_topic_match() -> None:
    intent = parse_research_intent(
        "According to NIST, what is the purpose of the Cybersecurity Framework? "
        "Answer in two cited sentences."
    )
    pieces = (
        (
            "# Cybersecurity Framework Helping organizations to better understand and "
            "improve their management of cybersecurity risk."
        ),
        (
            "Cybersecurity Framework National Institute of Standards and Technology logo "
            "HEADQUARTERS Webmaster Contact Us Other Offices"
        ),
    )

    excerpt, piece_index = select_fetched_evidence_excerpt(
        intent,
        pieces,
        document_url="https://www.nist.gov/cyberframework",
        document_title="Cybersecurity Framework | NIST",
    )

    assert piece_index == 0
    assert "Helping organizations" in excerpt[:240]
    assert "HEADQUARTERS" not in excerpt


def test_fetched_excerpt_prefers_answer_bearing_cause_over_negated_anchor_match() -> None:
    intent = parse_research_intent(
        "According to NASA, what causes Earth's seasons? Answer in two cited sentences."
    )
    pieces = (
        "Earth's distance from the Sun is not the cause of the seasons.",
        (
            "Earth's tilted axis creates seasons. As Earth orbits the Sun, the hemisphere tilted "
            "toward it receives more direct sunlight while the other receives less."
        ),
    )

    excerpt, piece_index = select_fetched_evidence_excerpt(intent, pieces)

    assert piece_index == 1
    assert excerpt.startswith("Earth's tilted axis creates seasons")


def test_fetched_excerpt_prefers_explicit_cause_over_unspecified_reason() -> None:
    intent = parse_research_intent(
        "According to NASA, what causes Earth's seasons? Answer in two cited sentences."
    )
    pieces = (
        "There is a different reason for Earth's seasons.",
        "Earth has seasons because its axis doesn't stand up straight.",
    )

    excerpt, piece_index = select_fetched_evidence_excerpt(intent, pieces)

    assert piece_index == 1
    assert excerpt == "Earth has seasons because its axis doesn't stand up straight."


def test_fetched_excerpt_prefers_definition_over_generic_topic_mentions() -> None:
    intent = parse_research_intent(
        "According to WHO, what is the definition of physical activity?"
    )
    pieces = (
        "Physical activity refers to all movement, including during leisure and transport. Regular activity benefits health.",
        "WHO reports global progress to increase physical activity by 15% by 2030 and monitors action-plan milestones.",
    )

    excerpt, piece_index = select_fetched_evidence_excerpt(intent, pieces)

    assert piece_index == 0
    assert excerpt.startswith("Physical activity refers to all movement")


def test_fetched_excerpt_accepts_indicates_as_a_numeric_definition_answer() -> None:
    intent = parse_research_intent(
        "What does HTTP status 429 mean? Use one authoritative source and answer in two concise sentences."
    )
    pieces = (
        "HTTP response status codes reference.",
        (
            "The HTTP 429 Too Many Requests client error response status code indicates the client "
            "has sent too many requests in a given amount of time. A Retry-After header may tell "
            "the client how long to wait before trying again."
        ),
    )

    excerpt, piece_index = select_fetched_evidence_excerpt(
        intent,
        pieces,
        document_url="https://developer.vendor.test/docs/reference/status/429",
        document_title="HTTP 429 reference documentation",
    )

    assert piece_index == 1
    assert excerpt.startswith("The HTTP 429 Too Many Requests")
    assert "Retry-After" in excerpt


def test_fetched_excerpt_resolves_a_dotted_definition_subject_from_its_class_name() -> None:
    intent = parse_research_intent(
        "What is Python pathlib.Path? Use one authoritative source and answer in two concise sentences."
    )
    pieces = (
        "The pathlib module offers classes representing filesystem paths.",
        (
            "Path is most likely what most applications need. It instantiates a concrete path "
            "for the platform on which the code is running."
        ),
    )

    excerpt, piece_index = select_fetched_evidence_excerpt(
        intent,
        pieces,
        document_url="https://docs.python.org/3/library/pathlib.html",
        document_title="pathlib — Object-oriented filesystem paths",
    )

    assert piece_index == 1
    assert excerpt.startswith("Path is most likely")
    assert "instantiates a concrete path" in excerpt


def test_fetched_excerpt_treats_plain_what_is_as_a_definition_request() -> None:
    intent = parse_research_intent(
        "According to WHO, what is physical activity? Answer in two cited sentences."
    )
    pieces = (
        "The global cost of physical inactivity is high. --- ## Overview",
        (
            "WHO defines physical activity as any bodily movement produced by skeletal muscles "
            "that requires energy expenditure. Physical activity refers to all movement, including "
            "during leisure time, for transport, or as part of work or domestic activities."
        ),
    )

    excerpt, piece_index = select_fetched_evidence_excerpt(intent, pieces)

    assert piece_index == 1
    assert excerpt.startswith("WHO defines physical activity as")
    assert "Physical activity refers to all movement" in excerpt
    assert "physical inactivity is high" not in excerpt


def test_fetched_excerpt_requires_numeric_definition_subject() -> None:
    intent = parse_research_intent(
        "According to IETF documentation, what does HTTP status 404 mean?"
    )
    pieces = (
        "Identifiers in HTTP are used as the means for identifying resources.",
        (
            "The 404 Not Found status code means the origin server did not find a current "
            "representation for the target resource."
        ),
    )

    excerpt, piece_index = select_fetched_evidence_excerpt(intent, pieces)

    assert piece_index == 1
    assert excerpt.startswith("The 404 Not Found status code means")


def test_fetched_excerpt_prefers_measurement_definition_over_generic_indicator_list() -> None:
    intent = parse_research_intent(
        "According to the World Bank, what does GDP measure? Answer in two cited sentences."
    )
    pieces = (
        "GDP does not accurately measure the relative size and material well-being of economies.",
        (
            "Economic indicators include measures of macroeconomic performance "
            "(gross domestic product [GDP], consumption, investment, and trade)."
        ),
        (
            "Gross Domestic Product (GDP), a widely used indicator, refers to the total "
            "gross value added by all resident producers in the economy."
        ),
    )

    excerpt, piece_index = select_fetched_evidence_excerpt(intent, pieces)

    assert piece_index == 2
    assert excerpt.startswith("Gross Domestic Product (GDP)")


def test_fetched_excerpt_prefers_numeric_guidance_over_survey_wording() -> None:
    intent = parse_research_intent(
        "According to CDC guidance, how much sleep should most adults get?"
    )
    pieces = (
        "BRFSS asks adults how many hours of sleep they get in a 24-hour period.",
        "Most adults need at least 7 hours of sleep per night.",
    )

    excerpt, piece_index = select_fetched_evidence_excerpt(intent, pieces)

    assert piece_index == 1
    assert excerpt == "Most adults need at least 7 hours of sleep per night."


def test_unknown_long_fetched_body_reaches_focused_assessment_without_becoming_evidence() -> None:
    intent = parse_research_intent(
        "According to CDC guidance, how much sleep should most adults get?"
    )
    candidate = build_external_candidate(
        plan_id="primary",
        query=intent.research_focus,
        title="About Sleep",
        url="https://www.cdc.gov/sleep/about/index.html",
        snippet="CDC sleep guidance for different age groups.",
        intent=intent,
    )
    full_body = (
        "About sleep and healthy routines.\n"
        "Age Group\n"
        "Age\n"
        "Sleep Recommended Daily\n"
        "Newborn\n"
        "0–3 months\n"
        "14–17 hours\n"
        "Infant\n"
        "4–12 months\n"
        "12–16 hours\n"
        "Adult\n"
        "18–60 years\n"
        "7 or more hours\n"
        "Footer"
    )
    fetched = bind_fetched_content(
        candidate,
        intent=intent,
        document_url=candidate.url,
        title=candidate.title,
        text=full_body,
    )

    assert fetched.prefetch_decision == "unknown"
    assert fetched_body_may_contain_focused_evidence(fetched)

    excerpt, _ = select_fetched_evidence_excerpt(
        intent,
        (full_body,),
        document_url=candidate.url,
        document_title=candidate.title,
    )
    focused = bind_fetched_content(
        candidate,
        intent=intent,
        document_url=candidate.url,
        title=candidate.title,
        text=excerpt,
    )

    assert focused.prefetch_decision == "candidate"


def test_fetched_excerpt_accepts_direct_capability_description_for_problem_query() -> None:
    intent = parse_research_intent(
        "According to the Python documentation, what problem does pathlib solve?"
    )
    pieces = (
        (
            "pathlib — Object-oriented filesystem paths. This module offers classes "
            "representing filesystem paths with semantics appropriate for different operating systems."
        ),
        "Pathlib release notes and documentation navigation.",
    )

    excerpt, piece_index = select_fetched_evidence_excerpt(intent, pieces)

    assert piece_index == 0
    assert "offers classes representing filesystem paths" in excerpt


def test_fetched_excerpt_uses_document_identity_for_pronoun_bound_capability() -> None:
    intent = parse_research_intent(
        "According to the Python documentation, what problem does pathlib solve?"
    )
    pieces = (
        (
            "This module offers classes representing filesystem paths with semantics "
            "appropriate for different operating systems. Path classes are divided between "
            "pure paths and concrete paths. The system's flavour determines path syntax; "
            "concrete paths provide I/O operations."
        ),
    )

    excerpt, piece_index = select_fetched_evidence_excerpt(
        intent,
        pieces,
        document_url="https://docs.python.org/3.11/library/pathlib.html",
        document_title="pathlib — Object-oriented filesystem paths",
    )

    assert piece_index == 0
    assert excerpt.startswith("This module offers classes")


def test_fetched_excerpt_prefers_scope_level_capability_over_specific_class_detail() -> None:
    intent = parse_research_intent(
        "According to the Python documentation, what problem does pathlib solve?"
    )
    pieces = (
        (
            "pathlib — Object-oriented filesystem paths. Added in version 3.4. "
            "This module offers classes representing filesystem paths with semantics "
            "appropriate for different operating systems."
        ),
        (
            "class pathlib.PureWindowsPath. A subclass of PurePath, this path flavour "
            "represents Windows filesystem paths, including UNC paths."
        ),
    )

    excerpt, piece_index = select_fetched_evidence_excerpt(intent, pieces)

    assert piece_index == 0
    assert "This module offers classes" in excerpt


def test_fetched_evidence_excerpt_does_not_promote_unknown_page_text() -> None:
    intent = parse_research_intent("What causes Earth's seasons?")

    excerpt, _ = select_fetched_evidence_excerpt(
        intent,
        ("Navigation, account controls, unrelated mission directory, and footer links.",),
    )

    assert excerpt == ""


def test_authoritative_fetched_excerpt_separates_content_relevance_from_source_identity() -> None:
    intent = parse_research_intent(
        "Explain SQLite FTS5 MATCH and ranking using authoritative SQLite documentation."
    )
    excerpt, piece_index = select_fetched_evidence_excerpt(
        intent,
        (
            "SQLite FTS5 queries use the MATCH operator with phrase and prefix syntax. "
            "SQLite FTS5 exposes a hidden rank column whose default ranking function is BM25.",
        ),
        document_url="https://www.sqlite.org/fts5.html",
        document_title="SQLite FTS5 Extension",
    )

    assert piece_index == 0
    assert "MATCH operator" in excerpt
    assert "rank column" in excerpt


def test_fetched_excerpt_skips_table_of_contents_anchor_for_assertive_text() -> None:
    intent = parse_research_intent(
        "Explain SQLite FTS5 using authoritative SQLite documentation."
    )
    excerpt, piece_index = select_fetched_evidence_excerpt(
        intent,
        (
            "[Overview of FTS5](#overview_of_fts5) [FTS5 Query Syntax](#fts5_strings)",
            "Otherwise, it returns an SQLite error code. In this case xDestroy is not invoked.",
            "SQLite FTS5 is a virtual table module that provides full-text search.",
        ),
        document_url="https://www.sqlite.org/fts5.html",
        document_title="SQLite FTS5 Extension",
    )

    assert piece_index == 2
    assert excerpt.startswith("SQLite FTS5 is a virtual table module")


def test_fetched_excerpt_prefers_overview_definition_over_narrow_subtype_example() -> None:
    intent = parse_research_intent(
        "Overview of SQLite FTS5 and its use cases using authoritative SQLite documentation."
    )
    excerpt, piece_index = select_fetched_evidence_excerpt(
        intent,
        (
            "If an FTS5 trigram tokenizer is created with case_sensitive enabled, "
            "it may only index GLOB queries, not LIKE.",
            "FTS5 is an SQLite virtual table module that provides full-text search "
            "functionality to database applications.",
        ),
        document_url="https://www.sqlite.org/fts5.html",
        document_title="SQLite FTS5 Extension",
    )

    assert piece_index == 1
    assert excerpt.startswith("FTS5 is an SQLite virtual table module")


def test_fetched_excerpt_prefers_match_syntax_details_over_compatibility_notice() -> None:
    intent = parse_research_intent(
        "SQLite FTS5 MATCH query syntax and search patterns using authoritative SQLite documentation."
    )
    excerpt, piece_index = select_fetched_evidence_excerpt(
        intent,
        (
            "A complicated FTS query string appears to the right of MATCH or the equals operator, "
            "or as the first argument to a table-valued function syntax.",
            "FTS5_TOKENIZE_QUERY means a MATCH query is being executed and the tokenizer handles "
            "a bareword or quoted string.",
            "match the query, prefix terms or phrases may appear within a NEAR query, or boolean "
            "combinations may be used.",
            "FTS5 queries are made up of phrases, and MATCH expressions may combine them "
            "with AND, OR, and NOT boolean operators plus column filters.",
        ),
        document_url="https://www.sqlite.org/fts5.html",
        document_title="SQLite FTS5 Extension",
    )

    assert piece_index == 3
    assert "AND, OR, and NOT" in excerpt


def test_fetched_excerpt_prefers_ranking_explanation_over_raw_sql_example() -> None:
    intent = parse_research_intent(
        "SQLite FTS5 ranking and relevance scoring using authoritative SQLite documentation."
    )
    excerpt, piece_index = select_fetched_evidence_excerpt(
        intent,
        (
            "SELECT * FROM ft WHERE ft MATCH ? AND rank MATCH 'bm25(10.0, 5.0)' ORDER BY rank.",
            "The minus-one term differs from most BM25 implementations. Without it, a better "
            "match would receive a numerically higher BM25 score.",
            "### The bm25 function\nThe built-in FTS5 bm25 function returns a real value indicating how well "
            "the current row matches the full-text query. Better matches receive smaller values.",
        ),
        document_url="https://www.sqlite.org/fts5.html",
        document_title="SQLite FTS5 Extension",
    )

    assert piece_index == 2
    assert "Better matches receive smaller values" in excerpt


def test_fetched_excerpt_trims_dangling_code_before_query_syntax_prose() -> None:
    intent = parse_research_intent(
        "SQLite FTS5 MATCH query syntax and search patterns using authoritative SQLite documentation."
    )
    excerpt, _ = select_fetched_evidence_excerpt(
        intent,
        (
            "... MATCH '(one OR two) NOT three' ``` Phrases and NEAR groups may also be "
            "connected by implicit AND operators.",
        ),
        document_url="https://www.sqlite.org/fts5.html",
        document_title="SQLite FTS5 Extension",
    )

    assert excerpt.startswith("Phrases and NEAR groups")
    assert "```" not in excerpt


def test_resilient_adapter_pure_network_methods_do_not_access_store_or_shared_status() -> None:
    forbidden = _ForbiddenStore()
    adapter = ResilientSearchAdapter(_NetworkClient(), store=forbidden, retriever=forbidden)

    result = adapter.search_network("SQLite FTS5", count=1)
    document = adapter.fetch_network(result.hits[0].url)

    assert result.status["fallback_tier"] == "brave"
    assert document.metadata["canonical_url"] == "https://www.sqlite.org/fts5.html"
    assert adapter.last_search_status() == {}


def test_resilient_adapter_cache_writes_are_explicit_for_pure_calls() -> None:
    store = Store(":memory:")
    adapter = ResilientSearchAdapter(_NetworkClient(), store=store, retriever=Retriever(store))

    result = adapter.search_network("SQLite FTS5", count=1)
    assert store.cache_get(adapter.search_cache_key("SQLite FTS5", count=1)) is None
    assert adapter.persist_search_cache(
        "SQLite FTS5",
        count=1,
        options=None,
        result=result,
    )
    cached = adapter.read_search_cache("SQLite FTS5", count=1)
    assert cached is not None
    assert cached.status["fallback_tier"] == "cache"

    document = adapter.fetch_network(result.hits[0].url)
    assert store.cache_get(adapter.fetch_cache_key(document.url)) is None
    adapter.persist_fetch_cache(document)
    cached_document = adapter.read_fetch_cache(document.url)
    assert cached_document is not None
    assert cached_document.url == document.url
    assert cached_document.text == document.text
    assert cached_document.metadata == document.metadata
    assert adapter.last_search_status() == {}
    store.close()


def test_prepare_document_has_no_store_write_before_single_writer_commit() -> None:
    store = Store(":memory:")
    indexer = Indexer(store)
    document = Document(
        url="https://www.sqlite.org/fts5.html",
        title="SQLite FTS5 Extension",
        text="SQLite FTS5 provides full-text search virtual tables and MATCH queries.",
    )

    prepared = indexer.prepare_document(document)

    assert store.list_documents() == []
    assert prepared.pieces
    committed = indexer.commit_prepared_document(prepared, adoption_run_id="run_t49")
    assert committed["chunks"] == len(prepared.pieces)
    assert len(committed["chunk_ids"]) == len(prepared.pieces)
    assert [item["id"] for item in store.list_documents()] == [committed["document_id"]]
    store.close()


def test_prepared_document_commit_replaces_document_and_chunks_atomically() -> None:
    store = Store(":memory:")
    indexer = Indexer(store, config=ChunkingConfig(max_words=2, overlap_words=0))
    document = Document(
        url="https://example.test/atomic",
        title="Atomic source",
        text="alpha beta gamma delta",
    )

    first = indexer.commit_prepared_document(indexer.prepare_document(document))
    replacement = indexer.commit_prepared_document(
        indexer.prepare_document(
            Document(
                url=document.url,
                title=document.title,
                text="replacement evidence",
            )
        )
    )

    assert replacement["document_id"] == first["document_id"]
    assert replacement["chunk_ids"] != first["chunk_ids"]
    replacement_hits = store.search_chunks_lexical("replacement")
    assert [store.get_chunk(chunk_id).text for chunk_id, _ in replacement_hits] == [
        "replacement evidence"
    ]
    assert store.search_chunks_lexical("alpha") == []
    store.close()


def test_atomic_document_chunk_replace_rolls_back_on_chunk_serialization_failure() -> None:
    store = Store(":memory:")
    indexer = Indexer(store)
    committed = indexer.index_document(
        Document(
            url="https://example.test/rollback",
            title="Original source",
            text="original searchable evidence",
        )
    )

    with pytest.raises(TypeError):
        store.upsert_document_with_chunks(
            url="https://example.test/rollback",
            title="Replacement source",
            text="replacement body",
            chunks=[("replacement chunk", None), ("invalid chunk", [object()])],  # type: ignore[list-item]
        )

    document = store.get_document(committed["document_id"])
    assert document is not None
    assert document["title"] == "Original source"
    assert document["text"] == "original searchable evidence"
    assert store.search_chunks_lexical("original")
    assert store.search_chunks_lexical("replacement") == []
    store.close()


def test_retriever_can_scope_results_to_current_document_ids() -> None:
    store = Store(":memory:")
    indexer = Indexer(store)
    first = indexer.index_document(
        Document(
            url="https://example.test/old",
            title="Old alpha",
            text="Alpha evidence from a historical indexed document.",
        )
    )["document_id"]
    current = indexer.index_document(
        Document(
            url="https://example.test/current",
            title="Current alpha",
            text="Alpha evidence from the current selected document.",
        )
    )["document_id"]

    evidence = Retriever(store).retrieve(
        "Alpha evidence",
        top_k=5,
        document_ids=[current],
    )

    assert evidence
    assert {item.document_id for item in evidence} == {current}
    assert first not in {item.document_id for item in evidence}
    assert Retriever(store).retrieve("Alpha evidence", document_ids=[]) == []
    store.close()
