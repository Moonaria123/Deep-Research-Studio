"""T48 evidence relevance, authority, and bounded recovery contracts."""

from __future__ import annotations

from dataclasses import replace

import pytest

from deepresearch_core.baseline.adapters import Document, LLMResult, LLMStreamChunk, SearchHit
from deepresearch_core.baseline.budget import Mode, RunRequest
from deepresearch_core.baseline.evidence_eligibility import (
    assess_evidence,
    focused_recovery_query,
    output_instruction,
    parse_research_intent,
)
from deepresearch_core.baseline.errors import BaselineError, ErrorCode
from deepresearch_core.baseline.external_candidates import select_fetched_evidence_excerpt
from deepresearch_core.baseline.indexing import Indexer
from deepresearch_core.baseline.orchestrator import (
    Orchestrator,
    _RunDeadline,
    _deep_section_focus,
    _filter_research_state_to_final_evidence,
    _grounding_repair_action,
)
from deepresearch_core.baseline.retrieval import EvidenceItem, Retriever
from deepresearch_core.baseline.store import Store
from tests.baseline.fakes import FakeLLM


def _evidence(
    *,
    title: str,
    text: str,
    url: str = "https://example.test/doc",
    origin: str = "external",
    artifact_type: str = "web_document",
    artifact_id: str = "",
) -> EvidenceItem:
    return EvidenceItem(
        chunk_id=-1,
        document_id=url or artifact_id,
        url=url,
        title=title,
        text=text,
        score=1.0,
        origin=origin,
        artifact_type=artifact_type,
        artifact_id=artifact_id,
        provenance_refs=[url or f"{origin}:{artifact_type}:{artifact_id}"],
    )


@pytest.mark.parametrize(
    ("query", "focus", "source_requirement", "citation_limit", "sentence_count"),
    [
        (
            "What is SQLite FTS5? Use one authoritative source and answer in two concise sentences.",
            "What is SQLite FTS5?",
            "authoritative",
            1,
            2,
        ),
        (
            "什么是 SQLite FTS5？请使用一个权威来源，并用两句简洁回答。",
            "什么是 SQLite FTS5？",
            "authoritative",
            1,
            2,
        ),
        (
            "According to NASA, what causes Earth's seasons? Answer in two cited sentences.",
            "According to NASA, what causes Earth's seasons?",
            "none",
            None,
            2,
        ),
        (
            "What is Python pathlib.Path? Use one authoritative source and answer in two concise sentences.",
            "What is Python pathlib.Path?",
            "authoritative",
            1,
            2,
        ),
        (
            "Compare SQLite FTS5 and PostgreSQL full-text search using authoritative sources. "
            "Answer in exactly 4 concise sentences with citations.",
            "Compare SQLite FTS5 and PostgreSQL full-text search.",
            "authoritative",
            None,
            4,
        ),
    ],
)
def test_research_intent_separates_focus_source_and_output_shape(
    query: str,
    focus: str,
    source_requirement: str,
    citation_limit: int | None,
    sentence_count: int,
) -> None:
    intent = parse_research_intent(query)

    assert intent.original_query == query
    assert intent.research_focus == focus
    assert intent.source_requirement == source_requirement
    assert intent.citation_limit == citation_limit
    assert intent.output_shape == "sentence_count"
    assert intent.sentence_count == sentence_count
    assert intent.parse_status == "parsed"


def test_authoritative_documentation_is_a_source_requirement_not_research_focus() -> None:
    intent = parse_research_intent(
        "Compare SQLite FTS5 indexing and query capabilities using authoritative documentation."
    )

    assert intent.research_focus == "Compare SQLite FTS5 indexing and query capabilities."
    assert intent.source_requirement == "authoritative"
    assert "source_requirement_authoritative" in intent.reason_codes


def test_scoped_authoritative_documentation_and_imperative_sentence_count_are_parsed() -> None:
    intent = parse_research_intent(
        "Write a concise research report explaining SQLite FTS5, its MATCH query syntax, "
        "and its ranking capabilities using authoritative SQLite documentation. "
        "Use exactly 6 concise sentences with citations."
    )

    assert intent.research_focus == (
        "Write a concise research report explaining SQLite FTS5, its MATCH query syntax, "
        "and its ranking capabilities."
    )
    assert intent.source_requirement == "authoritative"
    assert intent.output_shape == "sentence_count"
    assert intent.sentence_count == 6
    assert intent.parse_status == "parsed"


def test_deep_section_focus_preserves_parent_entity_for_authority() -> None:
    intent = parse_research_intent(
        "Write a concise research report explaining SQLite FTS5, its MATCH query syntax, "
        "and its ranking capabilities using authoritative SQLite documentation."
    )
    section_intent = replace(
        intent,
        research_focus=_deep_section_focus(
            intent.research_focus,
            "FTS5 MATCH Query Syntax and Operators",
        ),
    )

    outcome = assess_evidence(
        section_intent,
        [
            _evidence(
                title="SQLite FTS5 Extension",
                url="https://www.sqlite.org/fts5.html",
                text="SQLite FTS5 documents MATCH query syntax and ranking capabilities.",
            )
        ],
    ).outcomes[0]

    assert "SQLite FTS5" in section_intent.research_focus
    assert outcome.authority_status == "satisfied"
    assert outcome.decision == "eligible"


def test_http_status_recovery_query_requests_the_normative_rfc_surface() -> None:
    intent = parse_research_intent(
        "What does HTTP status 429 mean? Use one authoritative source and answer in two concise sentences."
    )

    assert focused_recovery_query(intent, "zero_citable_evidence") == (
        "What does HTTP status 429 mean? IETF RFC authoritative official documentation"
    )


def test_research_intent_is_conservative_for_quotes_negation_and_primary_key() -> None:
    quoted = parse_research_intent('Compare "Use one source" with SQLite FTS5, not SQLCipher.')
    primary_key = parse_research_intent("Explain a SQLite primary key in two examples.")
    negated_en = parse_research_intent("Explain SQLite FTS5. Do not use official sources.")
    negated_zh = parse_research_intent("解释 SQLite FTS5，请不要使用权威来源。")

    assert quoted.research_focus == quoted.original_query
    assert quoted.source_requirement == "none"
    assert primary_key.source_requirement == "none"
    assert primary_key.output_shape == "default"
    assert negated_en.source_requirement == "none"
    assert negated_en.citation_limit is None
    assert negated_zh.source_requirement == "none"
    assert negated_zh.citation_limit is None


@pytest.mark.parametrize(
    ("query", "focus"),
    [
        (
            "Explain photosynthesis using one authoritative source in two concise sentences.",
            "Explain photosynthesis.",
        ),
        ("用一个权威来源和两句简洁的话解释碳化硅。", "解释碳化硅。"),
    ],
)
def test_inline_output_constraints_are_removed_without_losing_the_focus(query: str, focus: str) -> None:
    intent = parse_research_intent(query)

    assert intent.research_focus == focus
    assert intent.source_requirement == "authoritative"
    assert intent.citation_limit == 1
    assert intent.output_shape == "sentence_count"
    assert intent.sentence_count == 2


def test_citation_limit_alone_does_not_disable_mode_report_structure() -> None:
    intent = parse_research_intent(
        "Write a Deep report about SQLite FTS5 using three authoritative sources."
    )

    instruction = output_instruction(intent)

    assert intent.output_shape == "default"
    assert intent.citation_limit == 3
    assert "Use no more than 3 public citation source(s)." in instruction
    assert "do not add headings or extra sections" not in instruction


def test_query_scoped_eligibility_rejects_irrelevant_but_accepts_official_and_synonym() -> None:
    intent = parse_research_intent(
        "What is SQLite FTS5? Use one authoritative source and answer in two concise sentences."
    )
    pack = [
        _evidence(
            title="SQLCipher",
            text="SQLCipher adds transparent database encryption to SQLite.",
            url="https://www.zetetic.net/sqlcipher/",
        ),
        _evidence(
            title="OpenAI API",
            text="The OpenAI API provides model inference endpoints.",
            url="https://platform.openai.com/docs/",
        ),
        _evidence(
            title="SQLite FTS5 Extension",
            text="SQLite FTS5 is the SQLite full-text search extension and supports MATCH queries.",
            url="https://www.sqlite.org/fts5.html",
        ),
        _evidence(
            title="SQLite full text search version 5",
            text="SQLite full text search version 5 creates searchable virtual tables.",
            url="https://sqlite.org/search-version-5",
        ),
    ]

    assessment = assess_evidence(intent, pack)
    decisions = {item.evidence_id: item for item in assessment.outcomes}

    assert "SQLCipher" not in [item.title for item in assessment.adopted]
    assert "OpenAI API" not in [item.title for item in assessment.adopted]
    assert decisions[assessment.outcomes[0].evidence_id].decision == "rejected"
    assert decisions[assessment.outcomes[1].evidence_id].decision == "unknown"
    assert all(item.authority_status == "satisfied" for item in assessment.outcomes[2:])


def test_explicit_definition_requires_answer_bearing_definition_signal() -> None:
    intent = parse_research_intent(
        "According to WHO, what is the definition of physical activity?"
    )
    generic = _evidence(
        title="Physical activity",
        text="Physical activity has significant benefits for hearts, bodies and minds.",
        url="https://www.who.int/health-topics/physical-activity",
    )
    direct = _evidence(
        title="Physical activity",
        text="Physical activity refers to all movement, including during leisure and transport.",
        url="https://www.who.int/health-topics/physical-activity",
    )

    generic_outcome = assess_evidence(intent, [generic]).outcomes[0]
    direct_outcome = assess_evidence(intent, [direct]).outcomes[0]

    assert generic_outcome.decision == "unknown"
    assert generic_outcome.relevance == "weak"
    assert "definition_signal_missing" in generic_outcome.reason_codes


def test_explicit_numeric_definition_requires_the_identifier_in_evidence_text() -> None:
    intent = parse_research_intent(
        "According to IETF documentation, what does HTTP status 404 mean?"
    )
    generic = EvidenceItem(
        chunk_id=1,
        document_id="rfc-generic",
        url="https://www.ietf.org/rfc/rfc9110.html",
        title="RFC 9110: HTTP Semantics",
        text=(
            "Identifiers in HTTP are used throughout the protocol as the means for "
            "identifying resources."
        ),
        score=1.0,
    )
    direct = EvidenceItem(
        chunk_id=2,
        document_id="rfc-404",
        url="https://www.ietf.org/rfc/rfc9110.html#status.404",
        title="RFC 9110: 404 Not Found",
        text="The 404 Not Found status code means the origin server did not find a current representation.",
        score=1.0,
    )

    assessment = assess_evidence(intent, [generic, direct])
    generic_outcome, direct_outcome = assessment.outcomes

    assert generic_outcome.decision != "eligible"
    assert "definition_subject_missing" in generic_outcome.reason_codes
    assert direct_outcome.decision == "eligible"
    assert direct_outcome.decision == "eligible"
    assert direct_outcome.relevance == "strong"


def test_measurement_question_accepts_subject_bound_definition_not_generic_mentions() -> None:
    intent = parse_research_intent(
        "According to the World Bank, what does GDP measure?"
    )
    generic = _evidence(
        title="WDI Economy",
        text="Economic indicators include measures of macroeconomic performance such as GDP and trade.",
        url="https://datatopics.worldbank.org/world-development-indicators/themes/economy.html",
    )
    direct = _evidence(
        title="WDI Economy",
        text=(
            "Gross Domestic Product (GDP) refers to the total gross value added by all "
            "resident producers in the economy."
        ),
        url="https://datatopics.worldbank.org/world-development-indicators/themes/economy.html",
    )
    negated = _evidence(
        title="WDI Economy",
        text="GDP does not accurately measure the relative size and material well-being of economies.",
        url="https://datatopics.worldbank.org/world-development-indicators/themes/economy.html",
    )

    generic_outcome = assess_evidence(intent, [generic]).outcomes[0]
    direct_outcome = assess_evidence(intent, [direct]).outcomes[0]
    negated_outcome = assess_evidence(intent, [negated]).outcomes[0]

    assert generic_outcome.decision == "unknown"
    assert "definition_signal_missing" in generic_outcome.reason_codes
    assert direct_outcome.decision == "eligible"
    assert direct_outcome.relevance == "strong"
    assert negated_outcome.decision == "unknown"
    assert "definition_signal_missing" in negated_outcome.reason_codes


def test_amount_guidance_requires_a_subject_bound_numeric_recommendation() -> None:
    intent = parse_research_intent(
        "According to CDC guidance, how much sleep should most adults get?"
    )
    survey = _evidence(
        title="FastStats: Sleep in Adults",
        text="BRFSS asks adults how many hours of sleep they get in a 24-hour period.",
        url="https://www.cdc.gov/sleep/data-research/facts-stats/adults-sleep-facts-and-stats.html",
    )
    recommendation = _evidence(
        title="About Sleep",
        text="Most adults need at least 7 hours of sleep per night.",
        url="https://www.cdc.gov/sleep/about/index.html",
    )
    wrong_age_group = _evidence(
        title="About Sleep",
        text="Sleep Recommended Daily: Newborn 0–3 months, 14–17 hours.",
        url="https://www.cdc.gov/sleep/about/index.html",
    )

    survey_outcome = assess_evidence(intent, [survey]).outcomes[0]
    recommendation_outcome = assess_evidence(intent, [recommendation]).outcomes[0]
    wrong_age_outcome = assess_evidence(intent, [wrong_age_group]).outcomes[0]

    assert survey_outcome.decision == "unknown"
    assert "recommendation_signal_missing" in survey_outcome.reason_codes
    assert recommendation_outcome.decision == "eligible"
    assert recommendation_outcome.relevance == "strong"
    assert "recommendation_signal_matched" in recommendation_outcome.reason_codes
    assert wrong_age_outcome.decision == "unknown"
    assert "recommendation_signal_missing" in wrong_age_outcome.reason_codes


def test_fetched_table_excerpt_centers_the_requested_recommendation_subject() -> None:
    intent = parse_research_intent(
        "According to CDC guidance, how much sleep should most adults get?"
    )
    table = (
        "Age Group Age Sleep Recommended Daily Newborn 0–3 months 14–17 hours "
        "Infant 4–12 months 12–16 hours including naps Toddler 1–2 years 11–14 hours "
        "Preschool 3–5 years 10–13 hours School age 6–12 years 9–12 hours "
        "Teen 13–17 years 8–10 hours[2](https://www.cdc.gov/sleep/ref) "
        "Adult 18–60 years 7 or more hours "
        "Adult 61–64 years 7–9 hours Adult 65 years and older 7–8 hours."
    )

    excerpt, piece_index = select_fetched_evidence_excerpt(
        intent,
        [table],
        document_url="https://www.cdc.gov/sleep/about/index.html",
        document_title="About Sleep",
    )
    citation_prefix = excerpt[:240]
    prefix_outcome = assess_evidence(
        intent,
        [
            _evidence(
                title="About Sleep",
                text=citation_prefix,
                url="https://www.cdc.gov/sleep/about/index.html",
            )
        ],
    ).outcomes[0]

    assert piece_index == 0
    assert citation_prefix.startswith("Adult 18–60 years 7 or more hours")
    assert prefix_outcome.decision == "eligible"


def test_cause_excerpt_rejects_question_only_heading_and_prefers_primary_cause() -> None:
    fetched_text = (
        "## What Causes Tides?\n"
        "Gravity is one major force that creates tides.\n"
        "Ocean tides result from the gravitational attraction of the sun and moon.\n"
        "Distance changes the strength of tidal forces.\n"
        "The two bodies have very different masses.\n"
        "Their distances from Earth are also different.\n"
        "Those values determine their relative effects.\n"
        "The sun’s tide-generating force is about half that of the moon, and the "
        "moon is the dominant force affecting the Earth’s tides."
    )
    url = "https://oceanservice.noaa.gov/education/tutorial_tides/tides02_cause.html"
    title = "What Causes Tides?"

    primary_excerpt, piece_index = select_fetched_evidence_excerpt(
        parse_research_intent(
            "According to NOAA, what primarily causes ocean tides? Answer in two cited sentences."
        ),
        [fetched_text],
        document_url=url,
        document_title=title,
    )
    general_excerpt, _ = select_fetched_evidence_excerpt(
        parse_research_intent("According to NOAA, what causes ocean tides?"),
        [fetched_text],
        document_url=url,
        document_title=title,
    )

    assert piece_index == 0
    assert "moon is the dominant force" in primary_excerpt
    assert general_excerpt != "## What Causes Tides?"
    assert "creates tides" in general_excerpt or "result from" in general_excerpt


def test_problem_solution_question_accepts_direct_capability_description() -> None:
    intent = parse_research_intent(
        "According to the Python documentation, what problem does pathlib solve?"
    )
    direct = _evidence(
        title="pathlib — Object-oriented filesystem paths",
        text=(
            "This module offers classes representing filesystem paths with semantics "
            "appropriate for different operating systems."
        ),
        url="https://docs.python.org/3/library/pathlib.html",
    )
    generic = _evidence(
        title="pathlib release notes",
        text="Pathlib is documented in the Python standard library.",
        url="https://docs.python.org/3/whatsnew/",
    )

    direct_outcome = assess_evidence(intent, [direct]).outcomes[0]
    generic_outcome = assess_evidence(intent, [generic]).outcomes[0]

    assert direct_outcome.decision == "eligible"
    assert "problem_solution_signal_matched" in direct_outcome.reason_codes
    assert generic_outcome.decision == "unknown"
    assert "problem_solution_signal_missing" in generic_outcome.reason_codes


def test_problem_solution_excerpt_does_not_rank_a_negated_system_limitation_as_overview() -> None:
    intent = parse_research_intent(
        "According to the Python documentation, what problem does pathlib solve? "
        "Answer in two cited sentences."
    )
    fetched_text = (
        "This module offers classes representing filesystem paths with semantics "
        "appropriate for different operating systems.\n"
        "Path classes are divided between pure paths, which provide purely "
        "computational operations without I/O, and concrete paths, which also "
        "provide I/O operations.\n"
        "Regardless of the system you’re running on, you can instantiate all of "
        "these classes, since they\n...\n’t provide any operation that does system calls."
    )

    excerpt, piece_index = select_fetched_evidence_excerpt(
        intent,
        [fetched_text],
        document_url="https://docs.python.org/3.11/library/pathlib.html",
        document_title="pathlib — Object-oriented filesystem paths",
    )

    assert piece_index == 0
    assert excerpt.startswith("This module offers classes representing filesystem paths")
    assert not excerpt.startswith("Regardless of the system")


def test_problem_solution_signal_requires_a_matched_subject_entity() -> None:
    intent = parse_research_intent(
        "According to the Python documentation, what problem does pathlib solve?"
    )
    preference_memory = _evidence(
        title="Research preference",
        text=(
            "Provide a concise explanation using authoritative primary documentation "
            "for technical questions."
        ),
    )

    outcome = assess_evidence(intent, [preference_memory]).outcomes[0]

    assert outcome.decision == "unknown"
    assert outcome.relevance == "weak"
    assert "problem_solution_signal_missing" in outcome.reason_codes


def test_true_synonyms_are_unknown_instead_of_false_rejected() -> None:
    cases = [
        ("myocardial infarction", "Heart attack", "A heart attack blocks blood flow to heart muscle."),
        ("car insurance", "Automobile coverage", "Automobile coverage protects drivers after collisions."),
    ]

    for query, title, text in cases:
        outcome = assess_evidence(parse_research_intent(query), [_evidence(title=title, text=text)]).outcomes[0]
        assert outcome.decision == "unknown"
        assert outcome.relevance == "unknown"


def test_relevant_internal_primary_evidence_is_authoritative_only_for_internal_scope() -> None:
    evidence = _evidence(
        title="Launch decision",
        text="Project Orion internal launch target is July 2026.",
        url="",
        origin="local",
        artifact_type="library_note",
        artifact_id="note-1",
    )
    internal = assess_evidence(
        parse_research_intent("What is our internal Project Orion launch target? Use one primary source."),
        [evidence],
    )
    public = assess_evidence(
        parse_research_intent("What is SQLite FTS5? Use one primary source."),
        [replace(evidence, text="SQLite FTS5 is a full text search extension.")],
    )

    assert internal.outcomes[0].decision == "eligible"
    assert internal.outcomes[0].authority_status == "satisfied"
    assert public.outcomes[0].decision != "eligible"


def test_internal_origin_alone_does_not_auto_promote_secondary_commentary() -> None:
    evidence = _evidence(
        title="Team commentary",
        text="Project Orion may launch in July 2026 according to an informal discussion.",
        url="",
        origin="local",
        artifact_type="library_note",
        artifact_id="note-secondary",
    )
    assessment = assess_evidence(
        parse_research_intent("What is our internal Project Orion launch target? Use one primary source."),
        [evidence],
    )

    assert assessment.outcomes[0].authority_status == "unknown"
    assert assessment.outcomes[0].decision == "unknown"
    assert assessment.trigger == "authority_unmet"


def test_internal_rumor_cannot_impersonate_a_primary_record_with_body_keywords() -> None:
    evidence = _evidence(
        title="Team rumor",
        text="An unconfirmed rumor says a policy status record sets Project Orion launch for July 2026.",
        url="",
        origin="memory",
        artifact_type="memory",
        artifact_id="rumor-1",
    )
    assessment = assess_evidence(
        parse_research_intent("What is our internal Project Orion launch target? Use one primary source."),
        [evidence],
    )

    assert assessment.outcomes[0].authority_status == "unknown"
    assert assessment.outcomes[0].decision == "unknown"
    assert assessment.trigger == "authority_unmet"


@pytest.mark.parametrize(
    ("query", "title", "url", "text"),
    [
        (
            "What is CSS Grid? Use one primary source.",
            "CSS Grid Layout Module Level 2",
            "https://www.w3.org/TR/css-grid-2/",
            "W3C Recommendation defining the CSS Grid layout model.",
        ),
        (
            "What is HTTP/3? Use one primary source.",
            "RFC 9114: HTTP/3",
            "https://www.rfc-editor.org/rfc/rfc9114.html",
            "RFC 9114 specifies HTTP/3.",
        ),
        (
            "What is TLS 1.3? Use one primary source.",
            "RFC 8446: The Transport Layer Security Protocol Version 1.3",
            "https://www.ietf.org/rfc/rfc8446.html",
            "RFC 8446 specifies TLS version 1.3.",
        ),
    ],
)
def test_direct_standards_documents_satisfy_primary_authority(
    query: str,
    title: str,
    url: str,
    text: str,
) -> None:
    outcome = assess_evidence(
        parse_research_intent(query),
        [_evidence(title=title, url=url, text=text)],
    ).outcomes[0]

    assert outcome.relevance == "strong"
    assert outcome.authority_status == "satisfied"
    assert outcome.decision == "eligible"


def test_publisher_owned_reference_surface_satisfies_authoritative_but_not_primary() -> None:
    item = _evidence(
        title="HTTP 429 reference documentation",
        url="https://developer.vendor.test/docs/reference/status/429",
        text="HTTP status 429 indicates the client sent too many requests in a given amount of time.",
    )

    authoritative = assess_evidence(
        parse_research_intent("What does HTTP status 429 mean? Use one authoritative source."),
        [item],
    ).outcomes[0]
    primary = assess_evidence(
        parse_research_intent("What does HTTP status 429 mean? Use one primary source."),
        [item],
    ).outcomes[0]

    assert authoritative.authority_status == "satisfied"
    assert authoritative.decision == "eligible"
    assert primary.authority_status == "unsatisfied"
    assert primary.decision == "rejected"


def test_comparison_accepts_direct_evidence_for_each_side_without_requiring_both_in_one_page() -> None:
    intent = parse_research_intent("Compare SQLite FTS5 and SQLCipher.")
    assessment = assess_evidence(
        intent,
        [
            _evidence(
                title="SQLite FTS5",
                text="SQLite FTS5 provides full-text search virtual tables.",
                url="https://sqlite.org/fts5.html",
            ),
            _evidence(
                title="SQLCipher",
                text="SQLCipher adds encryption to SQLite database files.",
                url="https://zetetic.net/sqlcipher/",
            ),
        ],
    )

    assert [outcome.decision for outcome in assessment.outcomes] == ["eligible", "eligible"]


def test_negated_entity_is_not_treated_as_a_required_focus_anchor() -> None:
    intent = parse_research_intent("Explain SQLite FTS5, not SQLCipher.")
    assessment = assess_evidence(
        intent,
        [
            _evidence(
                title="SQLite FTS5",
                text="SQLite FTS5 provides full-text search virtual tables.",
                url="https://sqlite.org/fts5.html",
            ),
            _evidence(
                title="SQLCipher",
                text="SQLCipher adds encryption to SQLite database files.",
                url="https://zetetic.net/sqlcipher/",
            ),
        ],
    )

    assert assessment.outcomes[0].decision == "eligible"
    assert assessment.outcomes[1].decision == "rejected"


def test_version_anchor_is_enforced_while_as_of_date_remains_temporal_context() -> None:
    intent = parse_research_intent("Python 3.12 logging changes as of 2026-07-01")
    assessment = assess_evidence(
        intent,
        [
            _evidence(title="Python 3.11 logging", text="Python 3.11 changed logging configuration."),
            _evidence(title="Python 3.12 logging", text="Python 3.12 changed logging configuration."),
        ],
    )

    assert assessment.outcomes[0].decision == "rejected"
    assert assessment.outcomes[1].decision == "eligible"
    assert "2026" not in assessment.outcomes[1].matched_anchors


def test_v_prefixed_semver_must_match_exactly() -> None:
    assessment = assess_evidence(
        parse_research_intent("OpenSSL v3.2 changes"),
        [
            _evidence(title="OpenSSL 3.1", text="OpenSSL version 3.1 changed provider behavior."),
            _evidence(title="OpenSSL v3.2", text="OpenSSL v3.2 changed provider behavior."),
        ],
    )

    assert assessment.outcomes[0].decision == "rejected"
    assert assessment.outcomes[1].decision == "eligible"
    assert "v3.2" in assessment.outcomes[1].matched_anchors


@pytest.mark.parametrize(
    "query",
    ["SQLite FTS5 as of July 2026", "截至2026年7月的 SQLite FTS5"],
)
def test_natural_language_as_of_dates_do_not_become_relevance_anchors(query: str) -> None:
    outcome = assess_evidence(
        parse_research_intent(query),
        [_evidence(title="SQLite FTS5", text="SQLite FTS5 provides full-text search virtual tables.")],
    ).outcomes[0]

    assert outcome.decision == "eligible"
    assert "2026" not in outcome.matched_anchors


def test_authority_requires_the_registrable_domain_not_an_impersonating_subdomain() -> None:
    intent = parse_research_intent("What is SQLite FTS5? Use one authoritative source.")
    assessment = assess_evidence(
        intent,
        [
            _evidence(
                title="SQLite FTS5",
                text="SQLite FTS5 provides full-text search virtual tables.",
                url="https://sqlite.example.com/fts5",
            ),
            _evidence(
                title="SQLite FTS5",
                text="SQLite FTS5 provides full-text search virtual tables.",
                url="https://www.sqlite.org/fts5.html",
            ),
        ],
    )

    assert assessment.outcomes[0].authority_status == "unsatisfied"
    assert assessment.outcomes[0].decision == "rejected"
    assert assessment.outcomes[1].authority_status == "satisfied"
    assert assessment.outcomes[1].decision == "eligible"


def test_community_page_on_the_entity_domain_is_not_authoritative_documentation() -> None:
    intent = parse_research_intent(
        "Compare SQLite FTS5 indexing and query capabilities using authoritative documentation."
    )
    assessment = assess_evidence(
        intent,
        [
            _evidence(
                title="SQLite User Forum: FTS5 query behavior",
                text="SQLite FTS5 MATCH and equality queries use different virtual-table paths.",
                url="https://sqlite.org/forum/forumpost/example",
            ),
            _evidence(
                title="SQLite FTS5 Extension",
                text="SQLite FTS5 documents indexing options and MATCH query syntax.",
                url="https://www.sqlite.org/fts5.html",
            ),
        ],
    )

    assert assessment.outcomes[0].authority_status == "unsatisfied"
    assert assessment.outcomes[0].decision == "rejected"
    assert assessment.outcomes[1].authority_status == "satisfied"
    assert assessment.outcomes[1].decision == "eligible"


@pytest.mark.parametrize("subdomain", ["community", "forum", "discuss"])
def test_community_subdomain_cannot_impersonate_authoritative_documentation(
    subdomain: str,
) -> None:
    intent = parse_research_intent(
        "Compare SQLite FTS5 indexing and query capabilities using authoritative documentation."
    )

    outcome = assess_evidence(
        intent,
        [
            _evidence(
                title="SQLite FTS5 query behavior",
                text="SQLite FTS5 documents indexing options and MATCH query syntax.",
                url=f"https://{subdomain}.sqlite.org/t/123",
            )
        ],
    ).outcomes[0]

    assert outcome.authority_status == "unsatisfied"
    assert outcome.decision == "rejected"


@pytest.mark.parametrize(
    ("title", "url"),
    [
        (
            "SQLite Extensions: Full-text search with FTS5",
            "https://blog.sqlite.ai/fts5-sqlite-text-search-extension",
        ),
        (
            "SQLite Full-Text Search: FTS5 Virtual Tables and MATCH",
            "https://coddy.tech/docs/sqlite/full-text-search",
        ),
    ],
)
def test_editorial_and_tutorial_surfaces_do_not_satisfy_sqlite_authority(
    title: str,
    url: str,
) -> None:
    outcome = assess_evidence(
        parse_research_intent(
            "Explain SQLite FTS5 MATCH and ranking using authoritative SQLite documentation."
        ),
        [
            _evidence(
                title=title,
                url=url,
                text="SQLite FTS5 supports MATCH queries and full-text search ranking.",
            )
        ],
    ).outcomes[0]

    assert outcome.authority_status == "unsatisfied"
    assert outcome.decision == "rejected"


def test_low_coverage_repair_retrieves_for_a_single_multiple_or_deep_source() -> None:
    assert _grounding_repair_action(
        reason="coverage_below_threshold",
        unknown_citations=0,
        citation_binding_valid=False,
        mode=Mode.FAST,
        evidence_count=1,
    ) == "rewrite_only"
    assert _grounding_repair_action(
        reason="coverage_below_threshold",
        unknown_citations=0,
        citation_binding_valid=False,
        mode=Mode.MULTIPLE,
        evidence_count=1,
    ) == "retrieve_and_rewrite"
    assert _grounding_repair_action(
        reason="coverage_below_threshold",
        unknown_citations=0,
        citation_binding_valid=False,
        mode=Mode.DEEP,
        evidence_count=1,
    ) == "retrieve_and_rewrite"
    assert _grounding_repair_action(
        reason="coverage_below_threshold",
        unknown_citations=0,
        citation_binding_valid=False,
        mode=Mode.MULTIPLE,
        evidence_count=2,
    ) == "rewrite_only"


def test_chinese_concept_and_comparison_sources_are_not_false_rejected() -> None:
    concept = assess_evidence(
        parse_research_intent("什么是碳化硅？用两句简洁回答。"),
        [_evidence(title="碳化硅简介", text="碳化硅是一种宽禁带半导体材料。")],
    )
    comparison = assess_evidence(
        parse_research_intent("比较碳化硅和氮化镓"),
        [
            _evidence(title="碳化硅", text="碳化硅适用于高压功率器件。"),
            _evidence(title="氮化镓", text="氮化镓适用于高频功率器件。"),
        ],
    )

    assert concept.outcomes[0].decision == "eligible"
    assert [outcome.decision for outcome in comparison.outcomes] == ["eligible", "eligible"]


class _TwoSentenceLLM:
    name = "t48-two-sentence"

    def __init__(
        self,
        *,
        first_low_coverage: bool = False,
        first_unknown_citation: bool = False,
    ) -> None:
        self.first_low_coverage = first_low_coverage
        self.first_unknown_citation = first_unknown_citation
        self.writer_calls = 0
        self.writer_messages: list[list[dict[str, str]]] = []

    def stream(self, messages, **kwargs):  # noqa: ANN001, ANN201, ARG002
        self.writer_calls += 1
        self.writer_messages.append(messages)
        if self.first_unknown_citation and self.writer_calls == 1:
            text = (
                "SQLite FTS5 is SQLite's full-text search extension [c99]. "
                "SQLite FTS5 creates searchable virtual tables [c99]."
            )
        elif self.first_low_coverage and self.writer_calls == 1:
            text = (
                "SQLite FTS5 is SQLite's full-text search extension [c1]. "
                "This unsupported sentence discusses unrelated weather forecasts."
            )
        else:
            text = (
                "SQLite FTS5 is SQLite's full-text search extension [c1]. "
                "SQLite FTS5 creates searchable virtual tables and supports MATCH queries [c1]."
            )
        yield LLMStreamChunk(delta=text, model="t48", provider="fake")
        yield LLMStreamChunk(done=True, model="t48", provider="fake")

    def complete(self, messages, **kwargs):  # noqa: ANN001, ANN201, ARG002
        user = next((item["content"] for item in reversed(messages) if item.get("role") == "user"), "")
        if "Create a concise research outline" in user:
            return LLMResult(text="SQLite FTS5", model="t48")
        return LLMResult(
            text=(
                "SQLite FTS5 is SQLite's full-text search extension [c1]. "
                "SQLite FTS5 creates searchable virtual tables and supports MATCH queries [c1]."
            ),
            model="t48",
        )

    def embed(self, texts, *, model=None):  # noqa: ANN001, ANN201, ARG002
        return [[1.0] for _ in texts]


class _RecoveringSearch:
    name = "recovering-search"

    def __init__(self) -> None:
        self.searches: list[str] = []
        self.pages = {
            "https://example.test/sqlcipher": "SQLCipher adds database encryption to SQLite.",
            "https://www.sqlite.org/fts5.html": (
                "SQLite FTS5 is SQLite's full-text search extension. "
                "SQLite FTS5 creates searchable virtual tables and supports MATCH queries."
            ),
        }

    def search(self, query: str, *, count: int = 5, options=None):  # noqa: ANN001
        self.searches.append(query)
        url = "https://example.test/sqlcipher" if len(self.searches) == 1 else "https://www.sqlite.org/fts5.html"
        return [SearchHit(title=url.rsplit("/", 1)[-1], url=url, snippet=self.pages[url], source="fake", rank=1)]

    def fetch(self, url: str) -> Document:
        return Document(url=url, title=url.rsplit("/", 1)[-1], text=self.pages[url], content_type="web")


class _AlwaysIrrelevantSearch:
    name = "always-irrelevant"

    def __init__(self) -> None:
        self.searches: list[str] = []

    def search(self, query: str, *, count: int = 5, options=None):  # noqa: ANN001
        self.searches.append(query)
        return [
            SearchHit(
                title="NOAA weather forecast",
                url="https://example.test/weather",
                snippet="NOAA publishes marine weather and rainfall forecasts.",
                source="fake",
                rank=1,
            )
        ]

    def fetch(self, url: str) -> Document:
        return Document(
            url=url,
            title="NOAA weather forecast",
            text="NOAA publishes marine weather and rainfall forecasts.",
            content_type="web",
        )


class _FocusedDeepSearch:
    name = "focused-deep-search"
    supports_focused_fetch = True

    def __init__(self) -> None:
        self.fetch_focuses: list[str] = []

    def search(self, query: str, *, count: int = 5, options=None):  # noqa: ANN001, ARG002
        return [
            SearchHit(
                title="SQLite FTS5 Extension",
                url="https://www.sqlite.org/fts5.html",
                snippet="",
                source="sqlite.org",
                rank=1,
            )
        ][:count]

    def fetch(self, url: str, **kwargs) -> Document:  # noqa: ANN003
        self.fetch_focuses.append(str(kwargs.get("relevance_query") or ""))
        return Document(
            url=url,
            title="SQLite FTS5 Extension",
            text=(
                "SQLite FTS5 is SQLite's full-text search extension and creates virtual tables. "
                "SQLite FTS5 queries use the MATCH operator with phrase, prefix, and boolean syntax. "
                "SQLite FTS5 exposes a hidden rank column whose default ranking function is BM25."
            ),
            content_type="web",
        )


class _FetchedDefinitionRecoverySearch:
    name = "fetched-definition-recovery"

    def __init__(self) -> None:
        self.searches: list[str] = []
        self.fetches: list[str] = []

    def search(self, query: str, *, count: int = 5, options=None):  # noqa: ANN001, ARG002
        self.searches.append(query)
        hits = [
            SearchHit(
                title="RFC 9110: HTTP Semantics",
                url="https://www.ietf.org/rfc/rfc9110.html",
                snippet="",
                source="ietf.org",
                rank=0,
            )
        ]
        if len(self.searches) > 1:
            hits.append(
                SearchHit(
                    title="404 Not Found",
                    url="https://example.test/http-404",
                    snippet="",
                    source="example.test",
                    rank=1,
                )
            )
        return hits[:count]

    def fetch(self, url: str) -> Document:
        self.fetches.append(url)
        text = (
            "HTTP Uniform Resource Identifiers are used as the means for identifying resources."
            if "rfc9110" in url
            else "The 404 Not Found status code means the origin server did not find a current representation."
        )
        return Document(url=url, title=url.rsplit("/", 1)[-1], text=text, content_type="web")


def _orchestrator(llm, search) -> tuple[Orchestrator, Store]:  # noqa: ANN001
    store = Store(":memory:")
    return (
        Orchestrator(
            store=store,
            llm=llm,
            retriever=Retriever(store),
            indexer=Indexer(store),
            search=search,
            enforce_grounding_gate=True,
        ),
        store,
    )


def test_prewriter_irrelevant_pack_is_advisory_and_keeps_generation_g1() -> None:
    llm = _TwoSentenceLLM()
    search = _RecoveringSearch()
    orchestrator, store = _orchestrator(llm, search)

    outcome = orchestrator.run(
        RunRequest(
            query="What is SQLite FTS5? Use one authoritative source and answer in two concise sentences.",
            mode=Mode.FAST,
        )
    )
    payload = outcome.to_dict()
    events = store.list_events(outcome.run_id)

    assert llm.writer_calls == 1
    assert len(search.searches) == 1
    assert len(payload["retrieval_outcomes"]) == 3
    assert payload["research_intent"]["research_focus"] == "What is SQLite FTS5?"
    assert payload["remediation"]["attempted"] is False
    assert payload["recovery_searches_max"] == 1
    assert payload["recovery_searches_used"] == 0
    assert payload["answer"]["citations"] == []
    assert len(payload["answer"]["text"].split(". ")) == 2
    assert payload["answer"]["coverage"] == 0.0
    assert payload["answer"]["confidence_label"] == "uncertain"
    assert payload["confidence_warning"]["acknowledged"] is False
    assert payload["sources"] == []
    assert store.list_external_documents() == []
    assert not [event for event in events if event["type"] == "answer_reset"]
    assert {event["payload"]["generation_id"] for event in events if event["type"] == "answer_delta"} == {"g1"}
    assert not any(
        event["type"] == "node_status" and event["payload"].get("node") == "source_recovery"
        for event in events
    )
    store.close()


def test_fetched_definition_rejection_stays_out_of_the_delivered_sources() -> None:
    search = _FetchedDefinitionRecoverySearch()
    orchestrator, store = _orchestrator(FakeLLM(), search)

    outcome = orchestrator.run(
        RunRequest(query="What does HTTP status 404 mean?", mode=Mode.FAST)
    )
    payload = outcome.to_dict()

    assert outcome.status == "completed"
    assert search.searches and len(search.searches) == 1
    assert search.fetches == ["https://www.ietf.org/rfc/rfc9110.html"]
    assert payload["remediation"]["attempted"] is False
    assert payload["sources"] == []
    assert payload["answer"]["confidence_label"] == "uncertain"
    assert store.list_external_documents() == []
    store.close()


def test_grounding_low_coverage_is_delivered_once_without_rewrite() -> None:
    llm = _TwoSentenceLLM(first_low_coverage=True)
    search = _RecoveringSearch()
    search.searches.append("prime-second-result")
    orchestrator, store = _orchestrator(llm, search)

    outcome = orchestrator.run(
        RunRequest(
            query="What is SQLite FTS5? Use one authoritative source and answer in two concise sentences.",
            mode=Mode.FAST,
        )
    )
    payload = outcome.to_dict()
    events = store.list_events(outcome.run_id)

    assert llm.writer_calls == 1
    assert len(search.searches) == 2
    assert len(payload["retrieval_outcomes"]) == 3
    assert payload["remediation"]["attempted"] is False
    assert payload["recovery_searches_used"] == 0
    assert not [event for event in events if event["type"] == "answer_reset"]
    generations = [
        event["payload"]["generation_id"]
        for event in events
        if event["type"] == "answer_delta"
    ]
    assert generations == ["g1"]
    assert payload["answer"]["coverage"] <= 0.5
    assert payload["answer"]["confidence_label"] == "uncertain"
    assert payload["confidence_warning"]["acknowledged"] is False
    store.close()


def test_unknown_citation_binding_is_delivered_once_as_low_confidence() -> None:
    llm = _TwoSentenceLLM(first_unknown_citation=True)
    search = _RecoveringSearch()
    search.searches.append("prime-second-result")
    orchestrator, store = _orchestrator(llm, search)

    outcome = orchestrator.run(
        RunRequest(
            query="What is SQLite FTS5? Use one authoritative source and answer in two concise sentences.",
            mode=Mode.FAST,
        )
    )
    payload = outcome.to_dict()

    assert llm.writer_calls == 1
    assert len(search.searches) == 2
    assert len(payload["retrieval_outcomes"]) == 3
    assert payload["remediation"]["attempted"] is False
    assert payload["recovery_searches_used"] == 0
    assert payload["answer"]["confidence_label"] == "uncertain"
    assert payload["confidence_warning"]["acknowledged"] is False
    store.close()


def test_retrieval_plan_checks_cancellation_between_lanes() -> None:
    llm = _TwoSentenceLLM()
    search = _RecoveringSearch()
    orchestrator, store = _orchestrator(llm, search)
    request = RunRequest(
        query="What is SQLite FTS5? Use one authoritative source.",
        mode=Mode.FAST,
        memory_policy="off",
    )
    run_id = store.create_run(
        session_id="",
        mode="fast",
        query=request.query,
        budget=request.resolved_budget().to_dict(),
    )

    class CancellingMemory:
        calls = 0

        def recall_evidence(self, **kwargs):  # noqa: ANN003, ANN201
            self.calls += 1
            if self.calls == 1:
                orchestrator.request_cancel(run_id)
            return []

    orchestrator._memory = CancellingMemory()  # noqa: SLF001
    ctx = orchestrator._build_context(request)  # noqa: SLF001

    with pytest.raises(BaselineError) as exc:
        orchestrator._retrieve_evidence_plan(  # noqa: SLF001
            request.query,
            request,
            request.resolved_budget(),
            run_id,
            None,
            ctx,
            plan_id="cancel-between-lanes",
            recovery=True,
        )

    assert exc.value.code is ErrorCode.RUN_CANCELLED
    assert store.get_run(run_id)["status"] == "cancelling"
    store.close()


def test_recovery_checks_deadline_before_starting_the_next_lane() -> None:
    llm = _TwoSentenceLLM()
    search = _RecoveringSearch()
    orchestrator, store = _orchestrator(llm, search)
    request = RunRequest(query="SQLite FTS5", mode=Mode.FAST, memory_policy="off")
    run_id = store.create_run(
        session_id="",
        mode="fast",
        query=request.query,
        budget=request.resolved_budget().to_dict(),
    )
    # T49 reserves writer/finalize/cleanup time before any recovery lane.  A
    # one-second internal deadline is therefore exhausted deterministically without
    # relying on an exact number of instrumentation clock reads.
    deadline = _RunDeadline(timeout_s=1, clock=lambda: 0.0)
    ctx = orchestrator._build_context(request)  # noqa: SLF001

    with pytest.raises(BaselineError) as exc:
        orchestrator._retrieve_evidence_plan(  # noqa: SLF001
            request.query,
            request,
            request.resolved_budget(),
            run_id,
            None,
            ctx,
            plan_id="deadline-recovery",
            recovery=True,
            deadline=deadline,
        )

    assert exc.value.code is ErrorCode.RUN_TIMEOUT
    assert search.searches == []
    store.close()


def test_low_grounding_exhaustion_delivers_without_clearing_draft() -> None:
    llm = _TwoSentenceLLM(first_low_coverage=True)
    search = _RecoveringSearch()
    search.searches.append("prime-second-result")
    orchestrator, store = _orchestrator(llm, search)

    original_stream = llm.stream

    def always_low(messages, **kwargs):  # noqa: ANN001, ANN202
        llm.writer_calls = 0
        yield from original_stream(messages, **kwargs)

    llm.stream = always_low  # type: ignore[method-assign]
    request = RunRequest(
        query="What is SQLite FTS5? Use one authoritative source and answer in two concise sentences.",
        mode=Mode.FAST,
    )
    run_id = store.create_run(
        session_id="",
        mode="fast",
        query=request.query,
        budget=request.resolved_budget().to_dict(),
    )
    outcome = orchestrator.run(request, run_id=run_id)

    assert outcome.answer.text.strip()
    assert outcome.answer.confidence_label == "uncertain"
    events = store.list_events(run_id)
    assert not [event for event in events if event["type"] == "answer_reset"]
    assert store.get_run(run_id)["status"] in {"completed", "completed_degraded"}
    store.close()


def test_deep_delivers_low_confidence_without_using_ineligible_evidence(monkeypatch) -> None:  # noqa: ANN001
    llm = _TwoSentenceLLM()
    search = _AlwaysIrrelevantSearch()
    orchestrator, store = _orchestrator(llm, search)
    request = RunRequest(
        query="Explain SQLite FTS5 in a report using authoritative sources.",
        mode=Mode.DEEP,
    )
    run_id = store.create_run(
        session_id="",
        mode="deep",
        query=request.query,
        budget=request.resolved_budget().to_dict(),
    )

    def forbidden_extract(*args, **kwargs):  # noqa: ANN001, ANN002, ANN003, ANN202
        raise AssertionError("ineligible evidence reached extract_notes")

    monkeypatch.setattr("deepresearch_core.baseline.orchestrator.extract_notes", forbidden_extract)

    outcome = orchestrator.run(request, run_id=run_id)

    assert outcome.answer.text.strip()
    assert outcome.answer.confidence_label == "uncertain"
    assert len(search.searches) == 1
    assert outcome.to_dict()["remediation"]["attempted"] is False
    assert outcome.to_dict()["recovery_searches_used"] == 0
    run = store.get_run(run_id)
    assert run is not None
    assert run["status"] in {"completed", "completed_degraded"}
    assert store.list_citations(run_id) == []
    assert store.list_external_documents() == []
    assert any(
        event["type"] == "budget_snapshot"
        and event["payload"].get("stage") == "evidence_ineligible"
        for event in store.list_events(run_id)
    )
    store.close()


def test_deep_fetch_uses_parent_entity_plus_each_section_focus(monkeypatch) -> None:  # noqa: ANN001
    llm = _TwoSentenceLLM()
    search = _FocusedDeepSearch()
    orchestrator, store = _orchestrator(llm, search)
    orchestrator._enforce = False
    monkeypatch.setattr(
        orchestrator,
        "_plan",
        lambda *args, **kwargs: [
            "Overview of SQLite FTS5",
            "FTS5 MATCH Query Syntax",
            "Ranking capabilities",
        ],
    )
    monkeypatch.setattr(orchestrator, "_detect_gaps", lambda *args, **kwargs: [])

    outcome = orchestrator.run(
        RunRequest(
            query=(
                "Explain SQLite FTS5, its MATCH query syntax, and its ranking capabilities "
                "using authoritative SQLite documentation."
            ),
            mode=Mode.DEEP,
        )
    )

    assert outcome.status == "completed"
    assert len(search.fetch_focuses) == 3
    assert all("SQLite FTS5" in focus for focus in search.fetch_focuses)
    assert "Overview of SQLite FTS5" in search.fetch_focuses[0]
    assert "FTS5 MATCH Query Syntax" in search.fetch_focuses[1]
    assert "ranking capabilities" in search.fetch_focuses[2].lower()
    assert outcome.to_dict()["remediation"]["attempted"] is False
    store.close()


def test_deep_planner_drift_stays_out_of_notes_and_is_advisory(monkeypatch) -> None:  # noqa: ANN001
    llm = _TwoSentenceLLM()
    search = _RecoveringSearch()
    orchestrator, store = _orchestrator(llm, search)
    seen: list[tuple[str, list[str]]] = []

    monkeypatch.setattr(orchestrator, "_plan", lambda *args, **kwargs: ["NOAA marine forecast"])
    monkeypatch.setattr(orchestrator, "_detect_gaps", lambda *args, **kwargs: [])

    def capture_notes(llm_adapter, title, evidence, **kwargs):  # noqa: ANN001, ANN202, ARG001
        seen.append((title, [item.url for item in evidence]))
        return "SQLite FTS5 grounded notes [c1]."

    monkeypatch.setattr("deepresearch_core.baseline.orchestrator.extract_notes", capture_notes)
    outcome = orchestrator.run(
        RunRequest(
            query="Explain SQLite FTS5 in a report using one authoritative source.",
            mode=Mode.DEEP,
        )
    )
    payload = outcome.to_dict()

    assert seen == []
    assert payload["remediation"]["attempted"] is False
    assert payload["recovery_searches_used"] == 0
    assert payload["answer"]["confidence_label"] == "uncertain"
    assert payload["research_state"]["evidence_matrix"] == [
        {
            "subtask": "NOAA marine forecast",
            "queries": ["SQLite FTS5 NOAA marine forecast"],
            "notes": "",
            "gaps": [],
            "source_count": 0,
            "status": "evidence_ineligible",
        }
    ]
    assert payload["sources"] == []
    store.close()


def test_deep_state_redacts_notes_from_sources_excluded_by_final_snapshot() -> None:
    retained = _evidence(
        title="SQLite FTS5",
        text="SQLite FTS5 provides full-text search.",
        url="https://www.sqlite.org/fts5.html",
    )
    excluded = _evidence(
        title="Secondary FTS5 overview",
        text="A secondary overview of SQLite FTS5.",
        url="https://example.test/secondary",
    )
    state = {
        "plan": ["SQLite FTS5", "Secondary overview"],
        "evidence_matrix": [
            {"subtask": "SQLite FTS5", "notes": "retained notes", "source_count": 1, "status": "researched"},
            {"subtask": "Secondary overview", "notes": "excluded notes", "source_count": 1, "status": "researched"},
        ],
    }
    sections = [
        {"_matrix_index": 0, "_evidence_keys": ["https://www.sqlite.org/fts5.html"]},
        {"_matrix_index": 1, "_evidence_keys": ["https://example.test/secondary"]},
    ]

    filtered = _filter_research_state_to_final_evidence(state, sections, [retained])

    assert filtered["evidence_matrix"][0]["notes"] == "retained notes"
    assert filtered["evidence_matrix"][1] == {
        "subtask": "Secondary overview",
        "notes": "",
        "source_count": 0,
        "status": "excluded_from_final",
    }
    assert state["evidence_matrix"][1]["notes"] == "excluded notes"


def test_existing_visible_library_document_is_not_hidden_by_a_later_candidate_run() -> None:
    store = Store(":memory:")
    indexer = Indexer(store)
    document = Document(
        url="https://example.test/existing",
        title="Existing source",
        text="Existing source text.",
        content_type="web",
    )

    indexer.index_document(document)
    indexer.index_document(document, adoption_run_id="run-later")

    assert [item["url"] for item in store.list_external_documents()] == [document.url]
    store.close()


def test_promoted_candidate_cannot_be_hidden_by_a_later_candidate_upsert() -> None:
    store = Store(":memory:")
    indexer = Indexer(store)
    document = Document(
        url="https://example.test/promoted",
        title="Promoted source",
        text="Promoted source text.",
        content_type="web",
    )

    first = indexer.index_document(document, adoption_run_id="run-first")
    assert store.list_external_documents() == []
    assert store.promote_document_to_library(first["document_id"]) is True
    indexer.index_document(document, adoption_run_id="run-later")

    assert [item["url"] for item in store.list_external_documents()] == [document.url]
    store.close()


def test_library_visibility_is_monotonic_across_store_connections(tmp_path) -> None:  # noqa: ANN001
    database = tmp_path / "library-visibility.sqlite3"
    first_store = Store(database)
    second_store = Store(database)
    first_indexer = Indexer(first_store)
    second_indexer = Indexer(second_store)
    document = Document(
        url="https://example.test/cross-connection",
        title="Cross-connection source",
        text="Cross-connection source text.",
        content_type="web",
    )

    candidate = first_indexer.index_document(document, adoption_run_id="run-first")
    assert second_store.promote_document_to_library(candidate["document_id"]) is True
    first_indexer.index_document(document, adoption_run_id="run-later")

    assert [item["url"] for item in second_store.list_external_documents()] == [document.url]
    first_store.close()
    second_store.close()
