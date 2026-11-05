"""First-class citations + GROUNDING_REQUIRED gate."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from deepresearch_core.baseline.citations import (
    GroundedAnswer,
    build_citations,
    claim_grounding_diagnostics,
    enforce_grounding,
    ground_answer,
)
from deepresearch_core.baseline.errors import BaselineError, ErrorCode
from deepresearch_core.baseline.retrieval import EvidenceItem


_CITATION_GOLDEN = Path(__file__).with_name("fixtures") / "citation_v1.jsonl"


def _citation_cases() -> list[dict]:
    return [json.loads(line) for line in _CITATION_GOLDEN.read_text(encoding="utf-8").splitlines() if line]


def _case_evidence(case: dict) -> list[EvidenceItem]:
    return [EvidenceItem(chunk_id=index, **row) for index, row in enumerate(case["evidence"], start=1)]


def _evidence() -> list[EvidenceItem]:
    return [
        EvidenceItem(
            chunk_id=1, document_id="d1", url="https://e.com/a", title="Photosynthesis",
            text="Photosynthesis converts sunlight carbon dioxide and water into glucose and oxygen.",
            score=0.9,
        ),
        EvidenceItem(
            chunk_id=2, document_id="d2", url="https://e.com/b", title="Chlorophyll",
            text="Chlorophyll pigment absorbs light energy mostly in the blue and red wavelengths.",
            score=0.7,
        ),
    ]


def test_build_citations_are_stable_ids():
    cites = build_citations(_evidence())
    assert [c.id for c in cites] == ["c1", "c2"]
    assert cites[0].document_id == "d1"


def test_ground_answer_maps_claims_to_citations():
    evidence = _evidence()
    answer_text = (
        "Photosynthesis converts sunlight carbon dioxide and water into glucose and oxygen. "
        "Chlorophyll pigment absorbs light energy in blue and red wavelengths."
    )
    grounded = ground_answer(answer_text, evidence)
    assert grounded.coverage == pytest.approx(1.0, abs=1e-6)
    assert grounded.claim_map  # at least one claim grounded


def test_enforce_grounding_blocks_ungrounded_answers():
    evidence = _evidence()
    grounded = ground_answer("The stock market rallied on unrelated macro news today.", evidence)
    assert grounded.coverage < 0.85
    with pytest.raises(BaselineError) as exc:
        enforce_grounding(grounded, min_coverage=0.85)
    assert exc.value.code is ErrorCode.GROUNDING_REQUIRED


def test_enforce_grounding_passes_well_grounded_answer():
    evidence = _evidence()
    grounded = ground_answer(
        "Photosynthesis converts sunlight carbon dioxide and water into glucose and oxygen.",
        evidence,
    )
    assert enforce_grounding(grounded, min_coverage=0.85) is grounded


def test_citation_v1_golden_corpus_contract():
    cases = _citation_cases()
    positives = [case for case in cases if case["grounded"]]
    negatives = [case for case in cases if not case["grounded"]]

    assert len(cases) == 5
    assert sum(case["expected_claims"] for case in positives) == 7
    assert len(negatives) == 1

    grounded_claims = 0
    unknown_citations = 0
    for case in positives:
        answer = ground_answer(case["answer"], _case_evidence(case))
        citation_ids = {citation.id for citation in answer.citations}
        mapped_ids = {citation_id for ids in answer.claim_map.values() for citation_id in ids}
        grounded_claims += len(answer.claim_map)
        unknown_citations += len(mapped_ids - citation_ids)
        assert len(answer.claim_map) == case["expected_claims"], case["id"]

    assert grounded_claims / 7 > 0.85
    assert unknown_citations == 0

    unsupported = negatives[0]
    answer = ground_answer(unsupported["answer"], _case_evidence(unsupported))
    with pytest.raises(BaselineError) as exc:
        enforce_grounding(answer, min_coverage=0.85)
    assert exc.value.code is ErrorCode.GROUNDING_REQUIRED


def test_no_substantive_claim_is_not_reported_as_fully_covered():
    answer = ground_answer("好。", _evidence())
    assert answer.claim_map == {}
    assert answer.coverage == 0.0


def test_cjk_hard_negative_is_not_grounded_by_one_common_bigram():
    evidence = [
        EvidenceItem(
            chunk_id=1,
            document_id="battery-range",
            url="https://fixtures.invalid/battery-range",
            title="Battery range",
            text="电池容量决定电动汽车的续航里程。",
            score=0.9,
        )
    ]

    answer = ground_answer("电池着火。", evidence)

    assert answer.claim_map == {}
    assert answer.coverage == 0.0
    with pytest.raises(BaselineError) as exc:
        enforce_grounding(answer, min_coverage=0.85)
    assert exc.value.code is ErrorCode.GROUNDING_REQUIRED


@pytest.mark.parametrize(
    ("claim", "evidence_text"),
    [
        ("电池容量导致车辆着火。", "电池容量决定电动汽车的续航里程。"),
        ("电池容量不会导致车辆着火。", "电池容量决定电动汽车的续航里程。"),
        ("该技术显著降低成本。", "该技术显著提高成本。"),
        ("电池容量会导致车辆着火。", "电池容量会影响车辆续航。"),
    ],
)
def test_cjk_shared_subject_does_not_ground_unrelated_or_contradictory_predicate(
    claim, evidence_text
):
    evidence = [
        EvidenceItem(
            chunk_id=1,
            document_id="battery-range",
            url="https://fixtures.invalid/battery-range",
            title="Battery range",
            text=evidence_text,
            score=0.9,
        )
    ]

    answer = ground_answer(claim, evidence)

    assert answer.claim_map == {}
    assert answer.coverage == 0.0
    with pytest.raises(BaselineError) as exc:
        enforce_grounding(answer, min_coverage=0.85)
    assert exc.value.code is ErrorCode.GROUNDING_REQUIRED


def test_cjk_reordered_supported_predicate_remains_grounded():
    evidence = [
        EvidenceItem(
            chunk_id=1,
            document_id="battery-range",
            url="https://fixtures.invalid/battery-range",
            title="Battery range",
            text="电池容量决定动力电池的续航里程。",
            score=0.9,
        )
    ]

    answer = ground_answer("动力电池续航里程由电池容量决定。", evidence)

    assert answer.coverage == 1.0
    assert enforce_grounding(answer) is answer


def test_latin_antonym_predicate_is_not_grounded_by_shared_context():
    evidence = [
        EvidenceItem(
            chunk_id=1,
            document_id="operating-cost",
            url="https://fixtures.invalid/operating-cost",
            title="Operating cost",
            text="Project operating costs increased substantially.",
            score=0.9,
        )
    ]

    answer = ground_answer("Project operating costs decreased substantially.", evidence)

    assert answer.coverage == 0.0
    with pytest.raises(BaselineError):
        enforce_grounding(answer)


def test_directional_binding_normalizes_regular_singular_and_plural_subjects():
    evidence = [
        EvidenceItem(
            chunk_id=1,
            document_id="nist-csf",
            url="https://www.nist.gov/cyberframework",
            title="Cybersecurity Framework | NIST",
            text="For industry, government, and organizations to reduce cybersecurity risks.",
            score=0.9,
        )
    ]

    answer = ground_answer(
        "The Cybersecurity Framework helps organizations reduce cybersecurity risk [c1].",
        evidence,
    )

    assert answer.coverage == 1.0
    assert enforce_grounding(answer) is answer


def test_citation_does_not_support_an_expanded_action_enumeration():
    evidence = [
        EvidenceItem(
            chunk_id=1,
            document_id="nist-csf-purpose",
            url="https://www.nist.gov/cyberframework",
            title="Cybersecurity Framework | NIST",
            text=(
                "The Cybersecurity Framework helps organizations better understand and "
                "improve their management of cybersecurity risk."
            ),
            score=0.9,
        )
    ]

    answer = ground_answer(
        "The Framework helps organizations understand, assess, prioritize, and "
        "communicate their cybersecurity efforts [c1].",
        evidence,
    )

    assert answer.coverage == 0.0
    with pytest.raises(BaselineError):
        enforce_grounding(answer)


def test_boolean_not_operator_is_not_treated_as_natural_language_negation():
    evidence = [
        EvidenceItem(
            chunk_id=1,
            document_id="fts5-query",
            url="https://sqlite.example/fts5",
            title="FTS5 query syntax",
            text=(
                "FTS5 queries can include boolean operators (AND, OR, NOT) and phrase queries. "
                "FTS5 index data can be compressed to save space."
            ),
            score=0.9,
        )
    ]

    answer = ground_answer(
        "FTS5 supports phrase queries for precise text matching [c1]. "
        "FTS5 index data can be compressed to save space [c1].",
        evidence,
    )

    assert answer.coverage == 1.0
    assert enforce_grounding(answer) is answer


def test_including_evidence_supports_an_includes_claim():
    evidence = [
        EvidenceItem(
            chunk_id=1,
            document_id="physical-activity",
            url="https://www.who.int/news-room/fact-sheets/detail/physical-activity",
            title="Physical activity",
            text=(
                "Physical activity refers to all movement including during leisure time, "
                "for transport to get to and from places, or as part of a person's work "
                "or domestic activities."
            ),
            score=0.9,
        )
    ]

    answer = ground_answer(
        "Physical activity includes movement during leisure time, transport, work, or domestic activities [c1].",
        evidence,
    )

    assert answer.coverage == 1.0
    assert enforce_grounding(answer) is answer


def test_source_reference_provided_is_not_a_capability_requirement():
    evidence = [
        EvidenceItem(
            chunk_id=1,
            document_id="world-bank-gdp",
            url="https://databank.worldbank.org/metadata/gdp",
            title="World Bank GDP glossary",
            text=(
                "PPP GDP is gross domestic product converted to international dollars "
                "using purchasing power parity rates."
            ),
            score=0.9,
        )
    ]

    answer = ground_answer(
        "According to the World Bank glossary entry provided, PPP GDP is gross domestic product "
        "converted to international dollars using purchasing power parity rates [c1].",
        evidence,
    )

    assert answer.coverage == 1.0
    assert enforce_grounding(answer) is answer


def test_modified_provided_example_is_not_a_capability_requirement():
    evidence = [
        EvidenceItem(
            chunk_id=1,
            document_id="fts5-query-guide",
            url="https://fixtures.invalid/fts5-query-guide",
            title="FTS5 query guide",
            text=(
                "FTS5's query language covers most of what people expect from a search box: "
                "AND is implicit between terms."
            ),
            score=0.9,
        )
    ]

    answer = ground_answer(
        "The provided FTS5 query-language example shows that AND is implicit between terms [c1].",
        evidence,
    )

    assert answer.coverage == 1.0
    assert enforce_grounding(answer) is answer


@pytest.mark.parametrize(
    "claim",
    [
        "The documentation provided shows PPP GDP is converted using parity rates [c1].",
        "The source provided confirms PPP GDP is converted using parity rates [c1].",
    ],
)
def test_postpositive_provided_source_reference_remains_supported(claim):
    evidence = [
        EvidenceItem(
            chunk_id=1,
            document_id="ppp-gdp",
            url="https://fixtures.invalid/ppp-gdp",
            title="PPP GDP definition",
            text="PPP GDP is converted using parity rates.",
            score=0.9,
        )
    ]

    answer = ground_answer(claim, evidence)

    assert answer.coverage == 1.0
    assert enforce_grounding(answer) is answer


def test_substantive_provided_capability_still_requires_support():
    evidence = [
        EvidenceItem(
            chunk_id=1,
            document_id="roadmap",
            url="https://example.test/roadmap",
            title="Acme roadmap",
            text="Acme plans data export for a future release.",
            score=0.9,
        )
    ]

    answer = ground_answer("Acme provided data export [c1].", evidence)

    assert answer.coverage == 0.0
    with pytest.raises(BaselineError):
        enforce_grounding(answer)


def test_mixed_provided_capability_is_not_hidden_by_source_attribution():
    evidence = [
        EvidenceItem(
            chunk_id=1,
            document_id="roadmap-example",
            url="https://fixtures.invalid/roadmap-example",
            title="Acme roadmap example",
            text="The roadmap example says audit logs are planned for a future release.",
            score=0.9,
        )
    ]

    answer = ground_answer(
        "Acme provided data export, and the provided roadmap example says audit logs are planned [c1].",
        evidence,
    )

    assert answer.coverage == 0.0
    with pytest.raises(BaselineError):
        enforce_grounding(answer)


@pytest.mark.parametrize("subject", ["examples", "sources"])
def test_source_like_subject_provided_capability_still_requires_support(subject):
    evidence = [
        EvidenceItem(
            chunk_id=1,
            document_id="import-guide",
            url="https://fixtures.invalid/import-guide",
            title="Import guide",
            text=f"The {subject} describe data import for customers.",
            score=0.9,
        )
    ]

    answer = ground_answer(
        f"The {subject} provided data export to customers [c1].",
        evidence,
    )

    assert answer.coverage == 0.0
    with pytest.raises(BaselineError):
        enforce_grounding(answer)


@pytest.mark.parametrize(
    ("claim", "evidence_text"),
    [
        (
            "Earth has seasons because its axis does not stand up straight [c1].",
            "Earth has seasons because its axis doesn't stand up straight.",
        ),
        (
            "Earth's axis doesn't stand up straight [c1].",
            "Earth's axis does not stand up straight.",
        ),
    ],
)
def test_latin_negation_contraction_and_expansion_have_the_same_polarity(
    claim,
    evidence_text,
):
    evidence = [
        EvidenceItem(
            chunk_id=1,
            document_id="seasons",
            url="https://space.example/seasons",
            title="What causes seasons",
            text=evidence_text,
            score=0.9,
        )
    ]

    answer = ground_answer(claim, evidence)

    assert answer.coverage == 1.0
    assert enforce_grounding(answer) is answer


def test_latin_negation_contraction_does_not_hide_a_real_polarity_conflict():
    evidence = [
        EvidenceItem(
            chunk_id=1,
            document_id="upright-axis",
            url="https://space.example/upright-axis",
            title="Upright axis",
            text="Earth's axis stands up straight.",
            score=0.9,
        )
    ]

    answer = ground_answer("Earth's axis doesn't stand up straight [c1].", evidence)

    assert answer.coverage == 0.0
    with pytest.raises(BaselineError):
        enforce_grounding(answer)


def test_latin_contracted_clause_can_support_a_positive_equivalent_restatement():
    evidence = [
        EvidenceItem(
            chunk_id=1,
            document_id="nasa-seasons",
            url="https://space.example/seasons",
            title="What causes seasons",
            text=(
                "Earth has seasons because its axis doesn't stand up straight. "
                "The tilted axis is the reason for the seasons."
            ),
            score=0.9,
        )
    ]

    answer = ground_answer(
        "Earth has seasons because its axis does not stand up straight [c1]. "
        "In other words, Earth's tilted axis causes the seasons [c1].",
        evidence,
    )

    assert answer.coverage == 1.0
    assert enforce_grounding(answer) is answer


@pytest.mark.parametrize(
    "answer_text",
    [
        (
            "The World Bank states that GDP measures the monetary value of goods and services "
            "produced in a given year [c1]. It is a measure of annual production expressed "
            "in monetary terms [c1]."
        ),
        (
            "According to the World Bank, GDP measures the monetary value of the goods and "
            "services produced in a given year [c1]. The World Bank adds that GDP does not "
            "provide a complete picture of a country's wealth or how sustainable that wealth "
            "will be over the long term [c1]."
        ),
        (
            "According to the World Bank, GDP measures the monetary value of the goods and "
            "services produced in a given year [c1]. It captures the value of a country's "
            "annual production of goods and services [c1]."
        ),
    ],
)
def test_latin_year_and_production_inflections_preserve_grounding_support(answer_text):
    evidence = [
        EvidenceItem(
            chunk_id=1,
            document_id="world-bank-gdp",
            url="https://www.worldbank.org/gdp",
            title="Beyond GDP",
            text=(
                "While GDP measures the monetary value of the goods and services produced "
                "in a given year, it doesn't provide a complete picture of a country's wealth, "
                "or how sustainable that wealth will be in the long term."
            ),
            score=0.9,
        )
    ]

    answer = ground_answer(answer_text, evidence)

    assert answer.coverage == 1.0
    assert enforce_grounding(answer, min_coverage=0.5) is answer


def test_latin_clause_local_negation_still_blocks_the_same_predicate():
    evidence = [
        EvidenceItem(
            chunk_id=1,
            document_id="world-bank-gdp-limit",
            url="https://www.worldbank.org/gdp-limit",
            title="GDP limitations",
            text="GDP does not provide a complete picture of a country's wealth.",
            score=0.9,
        )
    ]

    answer = ground_answer(
        "GDP provides a complete picture of a country's wealth [c1].",
        evidence,
    )

    assert answer.coverage == 0.0
    with pytest.raises(BaselineError):
        enforce_grounding(answer, min_coverage=0.5)


def test_capability_support_can_be_grounded_by_equivalent_include_wording():
    evidence = [
        EvidenceItem(
            chunk_id=1,
            document_id="fts5-capability",
            url="https://sqlite.example/fts5",
            title="FTS5 capabilities",
            text="FTS5 queries can include boolean operators and phrase queries.",
            score=0.9,
        )
    ]

    answer = ground_answer(
        "FTS5 supports boolean operators and phrase queries [c1].",
        evidence,
    )

    assert answer.coverage == 1.0


def test_capability_support_can_be_grounded_by_equivalent_ability_wording():
    evidence = [
        EvidenceItem(
            chunk_id=1,
            document_id="fts5-ranking-capability",
            url="https://sqlite.example/fts5-ranking",
            title="FTS5 ranking capabilities",
            text=(
                "FTS5 includes the ability to integrate ranking algorithms directly "
                "into search queries to sort results by relevance."
            ),
            score=0.9,
        )
    ]

    answer = ground_answer(
        "SQLite FTS5 can integrate ranking algorithms directly into search queries "
        "to sort results by relevance [c1].",
        evidence,
    )

    assert answer.coverage == 1.0


def test_absent_capability_is_not_grounded_by_ability_wording():
    evidence = [
        EvidenceItem(
            chunk_id=1,
            document_id="missing-ranking-capability",
            url="https://fixtures.invalid/missing-ranking-capability",
            title="Missing ranking capability",
            text="The product lacks the ability to integrate custom ranking algorithms.",
            score=0.9,
        )
    ]

    answer = ground_answer(
        "The product can integrate custom ranking algorithms [c1].",
        evidence,
    )

    assert answer.coverage == 0.0


@pytest.mark.parametrize(
    "evidence_text",
    [
        "End-to-end encryption is unavailable in the current release.",
        "End-to-end encryption is a future idea.",
        "Data export remains on the roadmap.",
        "Data export support is planned on the roadmap.",
        "Acme plans to support data export next year.",
    ],
)
def test_capability_support_is_not_grounded_by_unavailable_or_future_evidence(
    evidence_text: str,
) -> None:
    evidence = [
        EvidenceItem(
            chunk_id=1,
            document_id="capability-roadmap",
            url="https://fixtures.invalid/capability-roadmap",
            title="Capability roadmap",
            text=evidence_text,
            score=0.9,
        )
    ]
    claim = (
        "Acme supports data export [c1]."
        if "export" in evidence_text.lower()
        else "Acme supports end-to-end encryption [c1]."
    )

    answer = ground_answer(claim, evidence)

    assert answer.coverage == 0.0
    with pytest.raises(BaselineError):
        enforce_grounding(answer)


def test_cjk_current_capability_is_not_grounded_by_future_support() -> None:
    evidence = [
        EvidenceItem(
            chunk_id=1,
            document_id="future-capability",
            url="https://fixtures.invalid/future-capability",
            title="功能路线图",
            text="数据导出功能将在未来版本中支持。",
            score=0.9,
        )
    ]

    answer = ground_answer("当前版本支持数据导出 [c1]。", evidence)

    assert answer.coverage == 0.0
    with pytest.raises(BaselineError):
        enforce_grounding(answer)


def test_capability_paraphrase_keeps_full_subject_and_object_binding() -> None:
    evidence = [
        EvidenceItem(
            chunk_id=1,
            document_id="fts5-match",
            url="https://fixtures.invalid/fts5-match",
            title="FTS5 MATCH syntax",
            text=(
                "You can execute a full-text query against an FTS5 table using the MATCH "
                "operator in the WHERE clause of the SELECT statement."
            ),
            score=0.9,
        )
    ]

    answer = ground_answer(
        "FTS5 allows execution of full-text queries using the MATCH operator in the "
        "WHERE clause of a SELECT statement [c1].",
        evidence,
    )

    assert answer.coverage == 1.0
    assert enforce_grounding(answer) is answer


@pytest.mark.parametrize(
    ("claim", "evidence_text"),
    [
        (
            "The Acme index contained 500 records in March [c1].",
            "The Acme index contained 50 records in March.",
        ),
        (
            "The archive limit is 50 GB [c1].",
            "The archive limit is 50 MB.",
        ),
        (
            "The policy took effect on 2026-11-03 [c1].",
            "The policy took effect on 2026-11-30.",
        ),
        (
            "Contoso Servers support encrypted backups on disk [c1].",
            "Acme Servers support encrypted backups on disk.",
        ),
        (
            "SQLite FTS5 supports regular expression search [c1].",
            "SQLite FTS5 supports prefix search.",
        ),
        (
            "当前索引包含500条记录 [c1]。",
            "当前索引包含50条记录。",
        ),
        (
            "存档上限为50 GB [c1]。",
            "存档上限为50 MB。",
        ),
        (
            "该政策于2026-11-03生效 [c1]。",
            "该政策于2026-11-30生效。",
        ),
    ],
)
def test_changed_literal_facts_or_capability_objects_are_not_grounded(
    claim: str, evidence_text: str
) -> None:
    evidence = [
        EvidenceItem(
            chunk_id=1,
            document_id="fact-integrity",
            url="https://fixtures.invalid/fact-integrity",
            title="Fact integrity fixture",
            text=evidence_text,
            score=0.9,
        )
    ]

    answer = ground_answer(claim, evidence)

    assert answer.coverage == 0.0
    with pytest.raises(BaselineError):
        enforce_grounding(answer)


@pytest.mark.parametrize(
    ("claim", "evidence_text"),
    [
        (
            "The Acme index contained 50 records in March [c1].",
            "In March, the Acme index contained 50 records.",
        ),
        (
            "当前索引在3月包含50条记录 [c1]。",
            "3月时，当前索引包含50条记录。",
        ),
    ],
)
def test_reordered_wording_preserves_exact_fact_anchors(
    claim: str, evidence_text: str
) -> None:
    evidence = [
        EvidenceItem(
            chunk_id=1,
            document_id="fact-paraphrase",
            url="https://fixtures.invalid/fact-paraphrase",
            title="Fact paraphrase fixture",
            text=evidence_text,
            score=0.9,
        )
    ]

    answer = ground_answer(claim, evidence)

    assert answer.coverage == 1.0
    assert enforce_grounding(answer) is answer


def test_claim_grounding_diagnostics_are_advisory_and_recomputed() -> None:
    evidence = [
        EvidenceItem(
            chunk_id=1,
            document_id="advisory-check",
            url="https://fixtures.invalid/advisory-check",
            title="Advisory check",
            text="The Acme index contained 50 records in March.",
            score=0.9,
        )
    ]
    answer = ground_answer(
        "The Acme index contained 50 records in March [c1]. "
        "The Acme index contained 500 records in April [c1].",
        evidence,
    )

    diagnostics = claim_grounding_diagnostics(answer)

    assert [item.supported for item in diagnostics] == [True, False]
    assert diagnostics[0].citation_ids == ("c1",)
    assert diagnostics[1].citation_ids == ()
    assert answer.text.endswith("[c1].")


def test_markdown_section_headings_are_not_counted_as_factual_claims():
    evidence = [
        EvidenceItem(
            chunk_id=1,
            document_id="fts5-storage",
            url="https://sqlite.example/fts5",
            title="FTS5 storage",
            text="Contentless FTS5 tables do not support UPDATE or DELETE statements.",
            score=0.9,
        )
    ]

    answer = ground_answer(
        "## Index Storage Modes and Write Semantics\n\n"
        "Contentless FTS5 tables do not support UPDATE or DELETE statements [c1].",
        evidence,
    )

    assert answer.coverage == 1.0
    assert list(answer.claim_map) == [
        "Contentless FTS5 tables do not support UPDATE or DELETE statements [c1]."
    ]


def test_factual_markdown_heading_is_grounded_like_other_visible_claims() -> None:
    evidence = [
        EvidenceItem(
            chunk_id=1,
            document_id="company-history",
            url="https://fixtures.invalid/company-history",
            title="Company history",
            text="Acme was founded in 2020.",
            score=0.9,
        )
    ]

    answer = ground_answer(
        "## Acme is fully FDA-approved for pediatric treatment in 2026\n\n"
        "Acme was founded in 2020 [c1].",
        evidence,
    )

    assert answer.coverage < 1.0
    assert "## Acme is fully FDA-approved for pediatric treatment in 2026" not in answer.claim_map
    with pytest.raises(BaselineError):
        enforce_grounding(answer)


@pytest.mark.parametrize(
    "heading",
    [
        "## Acme FDA Approval Confirmed",
        "## Acme FDA Approval Granted",
        "## Acme Revenue Doubled",
    ],
)
def test_participial_factual_markdown_headings_cannot_bypass_grounding(
    heading: str,
) -> None:
    evidence = [
        EvidenceItem(
            chunk_id=1,
            document_id="company-history",
            url="https://fixtures.invalid/company-history",
            title="Company history",
            text="Acme was founded in 2020.",
            score=0.9,
        )
    ]

    answer = ground_answer(
        f"{heading}\n\nAcme was founded in 2020 [c1].",
        evidence,
    )

    assert answer.coverage < 1.0
    with pytest.raises(BaselineError):
        enforce_grounding(answer)


def test_supported_factual_heading_requires_and_accepts_an_inline_citation() -> None:
    evidence = [
        EvidenceItem(
            chunk_id=1,
            document_id="company-approval",
            url="https://fixtures.invalid/company-approval",
            title="Company approval and history",
            text="Acme FDA approval was confirmed in 2026. Acme was founded in 2020.",
            score=0.9,
        )
    ]

    answer = ground_answer(
        "## Acme FDA Approval Confirmed [c1]\n\nAcme was founded in 2020 [c1].",
        evidence,
    )

    assert answer.coverage == 1.0
    assert enforce_grounding(answer) is answer


@pytest.mark.parametrize(
    "heading",
    [
        "## 2026 Market Outlook",
        "## What is SQLite FTS5?",
    ],
)
def test_structural_or_question_markdown_headings_do_not_reduce_grounding(
    heading: str,
) -> None:
    evidence = [
        EvidenceItem(
            chunk_id=1,
            document_id="fts5",
            url="https://fixtures.invalid/fts5",
            title="SQLite FTS5",
            text="SQLite FTS5 provides full-text search virtual tables.",
            score=0.9,
        )
    ]

    answer = ground_answer(
        f"{heading}\n\nSQLite FTS5 provides full-text search virtual tables [c1].",
        evidence,
    )

    assert answer.coverage == 1.0
    assert list(answer.claim_map) == [
        "SQLite FTS5 provides full-text search virtual tables [c1]."
    ]


@pytest.mark.parametrize(
    "text",
    [
        "Revenue increased while operating costs decreased substantially.",
        "收入增加，成本减少。",
        "政策允许个人使用，但禁止商业转售。",
    ],
)
def test_compound_facts_with_both_directions_match_identical_evidence(text):
    evidence = [
        EvidenceItem(
            chunk_id=1,
            document_id="compound-fact",
            url="https://fixtures.invalid/compound-fact",
            title="Compound fact",
            text=text,
            score=0.9,
        )
    ]

    answer = ground_answer(text, evidence)

    assert answer.coverage == 1.0
    assert enforce_grounding(answer) is answer


@pytest.mark.parametrize(
    ("claim", "evidence_text"),
    [
        (
            "Revenue increased while operating costs decreased substantially.",
            "Revenue increased while operating costs remained stable.",
        ),
        ("收入增加，成本减少。", "收入增加，成本保持稳定。"),
        ("政策允许个人使用，但禁止商业转售。", "政策允许个人使用。"),
    ],
)
def test_compound_directional_claim_requires_evidence_for_each_direction(
    claim, evidence_text
):
    evidence = [
        EvidenceItem(
            chunk_id=1,
            document_id="partial-compound",
            url="https://fixtures.invalid/partial-compound",
            title="Partial compound",
            text=evidence_text,
            score=0.9,
        )
    ]

    answer = ground_answer(claim, evidence)

    assert answer.coverage == 0.0
    with pytest.raises(BaselineError):
        enforce_grounding(answer)


@pytest.mark.parametrize(
    ("claim", "evidence_text"),
    [
        (
            "Revenue increased while operating costs decreased substantially.",
            "Revenue decreased while operating costs increased substantially.",
        ),
        (
            "公司年度收入显著增加而运营成本显著减少。",
            "公司年度收入显著减少而运营成本显著增加。",
        ),
        (
            "Alpha revenue increased while Beta revenue decreased substantially.",
            "Beta revenue increased while Alpha revenue decreased substantially.",
        ),
        (
            "US revenue increased while UK revenue decreased substantially [c1].",
            "US revenue decreased while UK revenue increased substantially.",
        ),
        (
            "$F revenue increased while $T revenue decreased substantially [c1].",
            "$F revenue decreased while $T revenue increased substantially.",
        ),
        (
            "Ticker F revenue increased while Ticker T revenue decreased substantially [c1].",
            "Ticker F revenue decreased while Ticker T revenue increased substantially.",
        ),
        (
            "甲公司收入增加，乙公司收入减少。",
            "乙公司收入增加，甲公司收入减少。",
        ),
    ],
)
def test_direction_must_remain_bound_to_the_same_entity(claim, evidence_text):
    evidence = [
        EvidenceItem(
            chunk_id=1,
            document_id="swapped-directions",
            url="https://fixtures.invalid/swapped-directions",
            title="Swapped directions",
            text=evidence_text,
            score=0.9,
        )
    ]

    answer = ground_answer(claim, evidence)

    assert answer.coverage == 0.0
    with pytest.raises(BaselineError):
        enforce_grounding(answer)


def test_unsupported_inline_claim_cannot_hide_above_aggregate_coverage_threshold():
    labels = ["Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta"]
    supported_claims = [
        f"{label} project operating costs increased substantially [c{index}]."
        for index, label in enumerate(labels, start=1)
    ]
    unsupported_claim = "Eta project operating costs decreased substantially [c7]."
    evidence = [
        EvidenceItem(
            chunk_id=index,
            document_id=f"d{index}",
            url=f"https://fixtures.invalid/{index}",
            title=f"Source {index}",
            text=f"{label} project operating costs increased substantially.",
            score=0.9,
        )
        for index, label in enumerate(labels, start=1)
    ]
    evidence.append(
        EvidenceItem(
            chunk_id=7,
            document_id="d7",
            url="https://fixtures.invalid/7",
            title="Source 7",
            text="Eta project operating costs increased substantially.",
            score=0.9,
        )
    )

    answer = ground_answer(" ".join([*supported_claims, unsupported_claim]), evidence)

    assert answer.coverage == pytest.approx(6 / 7)
    assert unsupported_claim not in answer.claim_map
    with pytest.raises(BaselineError) as exc:
        enforce_grounding(answer)
    assert exc.value.code is ErrorCode.GROUNDING_REQUIRED


def test_unsupported_claim_without_inline_cannot_hide_above_coverage_threshold():
    labels = ["Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta"]
    supported_claims = [
        f"{label} project operating costs increased substantially [c{index}]."
        for index, label in enumerate(labels, start=1)
    ]
    unsupported_claim = "Eta project operating costs decreased substantially."
    evidence = [
        EvidenceItem(
            chunk_id=index,
            document_id=f"d{index}",
            url=f"https://fixtures.invalid/{index}",
            title=f"Source {index}",
            text=f"{label} project operating costs increased substantially.",
            score=0.9,
        )
        for index, label in enumerate(labels, start=1)
    ]
    evidence.append(
        EvidenceItem(
            chunk_id=7,
            document_id="d7",
            url="https://fixtures.invalid/7",
            title="Source 7",
            text="Eta project operating costs increased substantially.",
            score=0.9,
        )
    )

    answer = ground_answer(" ".join([*supported_claims, unsupported_claim]), evidence)

    assert answer.coverage == pytest.approx(6 / 7)
    assert unsupported_claim not in answer.claim_map
    with pytest.raises(BaselineError) as exc:
        enforce_grounding(answer)
    assert exc.value.code is ErrorCode.GROUNDING_REQUIRED


@pytest.mark.parametrize(
    ("claim", "evidence_text"),
    [
        ("未来汽车采用固态电池。", "今后汽车采用固态电池。"),
        ("无锡工厂采用机器人生产。", "该工厂采用机器人生产。"),
    ],
)
def test_cjk_lexical_names_are_not_misread_as_negation(claim, evidence_text):
    evidence = [
        EvidenceItem(
            chunk_id=1,
            document_id="cjk-lexical-name",
            url="https://fixtures.invalid/cjk-lexical-name",
            title="CJK lexical name",
            text=evidence_text,
            score=0.9,
        )
    ]

    answer = ground_answer(claim, evidence)

    assert answer.coverage == 1.0
    assert enforce_grounding(answer) is answer


def test_grounding_is_bound_to_the_inline_citation_id():
    evidence = [
        EvidenceItem(
            chunk_id=1,
            document_id="cost-increase",
            url="https://fixtures.invalid/cost-increase",
            title="Cost increase",
            text="Project operating costs increased substantially.",
            score=0.9,
        ),
        EvidenceItem(
            chunk_id=2,
            document_id="cost-decrease",
            url="https://fixtures.invalid/cost-decrease",
            title="Cost decrease",
            text="Project operating costs decreased substantially.",
            score=0.8,
        ),
    ]

    wrong = ground_answer(
        "Project operating costs decreased substantially [c1].", evidence
    )
    right = ground_answer(
        "Project operating costs decreased substantially [c2].", evidence
    )

    assert wrong.coverage == 0.0
    assert wrong.claim_map == {}
    with pytest.raises(BaselineError):
        enforce_grounding(wrong)
    assert right.coverage == 1.0
    assert list(right.claim_map.values()) == [["c2"]]
    assert enforce_grounding(right) is right


def test_every_inline_citation_on_a_claim_must_support_it():
    evidence = [
        EvidenceItem(
            chunk_id=1,
            document_id="cost-increase",
            url="https://fixtures.invalid/cost-increase",
            title="Cost increase",
            text="Project operating costs increased substantially.",
            score=0.9,
        ),
        EvidenceItem(
            chunk_id=2,
            document_id="cost-decrease",
            url="https://fixtures.invalid/cost-decrease",
            title="Cost decrease",
            text="Project operating costs decreased substantially.",
            score=0.8,
        ),
    ]

    answer = ground_answer(
        "Project operating costs decreased substantially [c1] [c2].", evidence
    )

    assert answer.coverage == 0.0
    assert answer.claim_map == {}
    with pytest.raises(BaselineError):
        enforce_grounding(answer)


def test_enforce_grounding_rejects_unknown_inline_citation_ids():
    evidence = _evidence()
    answer = ground_answer(
        "Photosynthesis converts sunlight carbon dioxide and water into glucose and oxygen [c1] [c99].",
        evidence,
    )

    assert answer.coverage == 0.0
    with pytest.raises(BaselineError) as exc:
        enforce_grounding(answer, min_coverage=0.85)
    assert exc.value.code is ErrorCode.GROUNDING_REQUIRED
    assert exc.value.details["unknown_citations"] == 1


def test_enforce_grounding_requires_strictly_more_than_eighty_five_percent():
    answer = GroundedAnswer(text="Grounded [c1].", coverage=0.85)

    with pytest.raises(BaselineError) as exc:
        enforce_grounding(answer, min_coverage=0.85)

    assert exc.value.code is ErrorCode.GROUNDING_REQUIRED


def test_sentence_boundaries_cover_cjk_semicolon_and_continuous_punctuation():
    evidence = [
        EvidenceItem(
            chunk_id=1,
            document_id="cjk-boundaries",
            url="https://fixtures.invalid/cjk-boundaries",
            title="CJK boundaries",
            text="电池容量支持长距离续航。快速充电可以缩短等待时间；热管理系统维持稳定性能！",
            score=0.9,
        )
    ]

    answer = ground_answer(
        "电池容量支持长距离续航。快速充电可以缩短等待时间；热管理系统维持稳定性能！？",
        evidence,
    )

    assert len(answer.claim_map) == 3
    assert answer.coverage == 1.0


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "FTS5 supports phrase queries.[c1] FTS5 supports prefix queries.[c1]",
            [
                "FTS5 supports phrase queries.[c1]",
                "FTS5 supports prefix queries.[c1]",
            ],
        ),
        (
            "FTS5 supports phrase queries. [c1] FTS5 supports prefix queries. [c1]",
            [
                "FTS5 supports phrase queries. [c1]",
                "FTS5 supports prefix queries. [c1]",
            ],
        ),
    ],
)
def test_citation_after_terminal_punctuation_binds_to_the_preceding_claim(
    text: str, expected: list[str]
):
    evidence = [
        EvidenceItem(
            chunk_id=1,
            document_id="fts5-query-boundary",
            url="https://sqlite.example/fts5",
            title="FTS5 queries",
            text="FTS5 supports phrase queries. FTS5 supports prefix queries.",
            score=0.9,
        )
    ]

    answer = ground_answer(text, evidence)

    assert answer.coverage == 1.0
    assert list(answer.claim_map) == expected


def test_sentence_boundaries_do_not_split_decimal_url_or_common_abbreviation():
    evidence = [
        EvidenceItem(
            chunk_id=1,
            document_id="protected-periods",
            url="https://example.com/research/v1.2",
            title="Protected periods",
            text=(
                "The measured value is 3.14 units according to Dr. Chen. "
                "Details remain available at https://example.com/research/v1.2 for verification."
            ),
            score=0.9,
        )
    ]

    answer = ground_answer(
        "The measured value is 3.14 units according to Dr. Chen. "
        "Details remain available at https://example.com/research/v1.2 for verification.",
        evidence,
    )

    assert len(answer.claim_map) == 2
    assert answer.coverage == 1.0


def test_sentence_boundaries_do_not_split_url_query_punctuation():
    evidence = [
        EvidenceItem(
            chunk_id=1,
            document_id="url-query",
            url="https://example.com/search?q=alpha;lang=en",
            title="URL query",
            text=(
                "Details are at https://example.com/search?q=alpha;lang=en for verification. "
                "The result remains independently stable."
            ),
            score=0.9,
        )
    ]

    answer = ground_answer(
        "Details are at https://example.com/search?q=alpha;lang=en for verification. "
        "The result remains independently stable.",
        evidence,
    )

    assert len(answer.claim_map) == 2
    assert answer.coverage == 1.0


def test_sentence_boundaries_do_not_split_terminal_punctuation_inside_inline_code():
    evidence = [
        EvidenceItem(
            chunk_id=1,
            document_id="postgres-ranking",
            url="https://www.postgresql.org/docs/current/textsearch-controls.html",
            title="PostgreSQL text search controls",
            text=(
                "PostgreSQL orders full-text search results with the ts_rank_cd ranking function. "
                "The query matches the textsearch value with the @@ operator."
            ),
            score=0.9,
        )
    ]

    answer = ground_answer(
        "PostgreSQL orders full-text search results with the `ts_rank_cd(...)` ranking function [c1]. "
        "The query matches the `textsearch` value with the `@@` operator [c1].",
        evidence,
    )

    assert len(answer.claim_map) == 2
    assert answer.coverage == 1.0
