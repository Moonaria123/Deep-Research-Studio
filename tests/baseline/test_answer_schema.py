"""Structured report schema: GroundedAnswer summary/key_takeaways + composer shaping."""

from __future__ import annotations

import pytest

from deepresearch_core.baseline.citations import (
    CONFIDENCE_CONTESTED,
    CONFIDENCE_ESTABLISHED,
    CONFIDENCE_UNCERTAIN,
    GroundedAnswer,
    derive_confidence_label,
    enforce_grounding,
    enrich_council_structure,
    enrich_structure,
    extract_report_structure,
    ground_answer,
    split_limitations,
)
from deepresearch_core.baseline.composer import AnswerComposer
from deepresearch_core.baseline.errors import BaselineError, ErrorCode
from deepresearch_core.baseline.retrieval import EvidenceItem

from tests.baseline.fakes import FakeLLM


def _evidence() -> list[EvidenceItem]:
    return [
        EvidenceItem(
            chunk_id=1, document_id="d1", url="https://e.com/a", title="Solar",
            text="Solar power generates electricity from sunlight using photovoltaic panels.",
            score=0.9,
        ),
    ]


def test_grounded_answer_to_dict_includes_structure():
    answer = GroundedAnswer(
        text="body",
        summary="exec",
        key_takeaways=["a", "b"],
        limitations="dated data",
        confidence_label="established",
    )
    payload = answer.to_dict()
    assert payload["summary"] == "exec"
    assert payload["key_takeaways"] == ["a", "b"]
    assert payload["text"] == "body"
    assert payload["limitations"] == "dated data"
    assert payload["confidence_label"] == "established"


def test_split_limitations_separates_section_from_body():
    text = (
        "## Executive Summary\n"
        "Solar is renewable. [c1]\n\n"
        "## Analysis\n"
        "Detail. [c1]\n\n"
        "## Limitations\n"
        "Data is from 2023 and may be dated. [c1]\n"
    )
    body, limitations = split_limitations(text)
    assert "Limitations" not in body
    assert "may be dated" not in body
    assert "may be dated" in limitations
    assert "## Analysis" in body


def test_split_limitations_recognizes_chinese_alias():
    text = "## 分析\n正文。[c1]\n\n## 局限性\n样本较小。[c1]\n"
    body, limitations = split_limitations(text)
    assert "局限性" not in body
    assert "样本较小" in limitations


def test_split_limitations_idempotent_when_absent():
    text = "## Analysis\nNo limitations here. [c1]\n"
    body, limitations = split_limitations(text)
    assert body == text
    assert limitations == ""


def test_split_limitations_stops_at_same_level_heading():
    text = (
        "## Limitations\n"
        "Caveat one. [c1]\n\n"
        "## Sources\n"
        "Source list.\n"
    )
    body, limitations = split_limitations(text)
    assert "## Sources" in body
    assert "Source list." in body
    assert "Caveat one." in limitations


def test_derive_confidence_label_thresholds():
    assert derive_confidence_label(0.9) == CONFIDENCE_ESTABLISHED
    assert derive_confidence_label(0.6) == CONFIDENCE_UNCERTAIN
    assert derive_confidence_label(0.3) == CONFIDENCE_UNCERTAIN
    # Council: high coverage but low cross-model agreement reads as contested.
    assert derive_confidence_label(0.9, agreement=0.3) == CONFIDENCE_CONTESTED
    assert derive_confidence_label(0.9, agreement=0.8) == CONFIDENCE_ESTABLISHED
    # Low coverage stays uncertain regardless of agreement.
    assert derive_confidence_label(0.4, agreement=0.9) == CONFIDENCE_UNCERTAIN


def test_enrich_structure_strips_limitations_and_sets_label():
    answer = GroundedAnswer(
        text=(
            "## Executive Summary\n"
            "Solar is renewable. [c1]\n\n"
            "## Key Takeaways\n"
            "- Solar is renewable [c1]\n\n"
            "## Limitations\n"
            "2023 data. [c1]\n"
        ),
        coverage=0.9,
    )
    enrich_structure(answer)
    assert "Limitations" not in answer.text
    assert "2023 data" in answer.limitations
    assert answer.summary
    assert answer.key_takeaways
    assert answer.confidence_label == CONFIDENCE_ESTABLISHED


def test_enrich_structure_keeps_limitations_in_grounding_binding_validation():
    evidence = _evidence()
    answer = ground_answer(
        "## Analysis\n"
        "Solar power generates electricity from sunlight using photovoltaic panels.[c1]\n\n"
        "## Limitations\n"
        "The cited source describes photovoltaic panels.[c1]",
        evidence,
    )

    enrich_structure(answer)

    assert "Limitations" not in answer.text
    assert "photovoltaic panels" in answer.limitations
    assert enforce_grounding(answer, min_coverage=0.5) is answer


def test_enrich_council_structure_splits_limitations_without_summary():
    answer = GroundedAnswer(
        text=(
            "## Agreement\n"
            "- The Moon's gravity is the dominant tidal driver. [c1]\n\n"
            "## Synthesis\n"
            "Lunar gravity dominates while solar gravity modulates ranges. [c1]\n\n"
            "## Limitations\n"
            "Some figures predate 2024. [c1]\n"
        ),
        coverage=0.9,
    )
    enrich_council_structure(answer)
    # Limitations are split out, but no Executive Summary / takeaways are fabricated
    # from the Agreement bullets.
    assert "Limitations" not in answer.text
    assert "predate 2024" in answer.limitations
    assert "## Agreement" in answer.text
    assert answer.summary == ""
    assert answer.key_takeaways == []
    assert answer.confidence_label == CONFIDENCE_ESTABLISHED


def test_enrich_council_structure_preserves_existing_label():
    answer = GroundedAnswer(text="## Synthesis\nA grounded view. [c1]", coverage=0.9, confidence_label=CONFIDENCE_CONTESTED)
    enrich_council_structure(answer)
    assert answer.confidence_label == CONFIDENCE_CONTESTED


def test_extract_report_structure_parses_heading_summary_and_takeaways():
    text = (
        "## Executive Summary\n"
        "Solar power is a renewable energy source. [c1]\n\n"
        "## Key Takeaways\n"
        "- Photovoltaic panels convert sunlight [c1]\n"
        "- Solar is renewable [c1]\n\n"
        "## Analysis\n"
        "Detailed analysis here. [c1]\n"
    )
    summary, takeaways = extract_report_structure(text)
    assert "renewable energy source" in summary
    assert takeaways == ["Photovoltaic panels convert sunlight [c1]", "Solar is renewable [c1]"]


def test_extract_report_structure_falls_back_to_first_paragraph():
    text = "Just a plain answer with no headings at all. [c1]\n\nSecond paragraph."
    summary, takeaways = extract_report_structure(text)
    assert summary.startswith("Just a plain answer")
    assert takeaways == []


def test_composer_structured_populates_summary_and_takeaways():
    report = (
        "## Executive Summary\n"
        "Solar power generates electricity from sunlight using photovoltaic panels. [c1]\n\n"
        "## Key Takeaways\n"
        "- Solar power generates electricity from sunlight using photovoltaic panels [c1]\n\n"
        "## Overview\n"
        "Solar power generates electricity from sunlight using photovoltaic panels. [c1]\n"
    )
    llm = FakeLLM(scripted={"renewable solar": report})
    composer = AnswerComposer(llm)
    answer = composer.compose("renewable solar", _evidence(), structured=True)
    assert "[c1]" in answer.text
    assert answer.summary
    assert answer.key_takeaways
    assert answer.coverage > 0.0
    payload = answer.to_dict()
    assert payload["summary"] == answer.summary
    assert payload["key_takeaways"] == answer.key_takeaways


def test_composer_structured_strips_limitations_into_field():
    report = (
        "## Executive Summary\n"
        "Solar power generates electricity from sunlight using photovoltaic panels. [c1]\n\n"
        "## Key Takeaways\n"
        "- Solar power generates electricity from sunlight using photovoltaic panels [c1]\n\n"
        "## Limitations\n"
        "Solar power generates electricity from sunlight using photovoltaic panels. [c1]\n"
    )
    llm = FakeLLM(scripted={"renewable solar": report})
    composer = AnswerComposer(llm)
    answer = composer.compose("renewable solar", _evidence(), structured=True)
    assert "## Limitations" not in answer.text
    assert "Limitations" not in answer.text
    assert answer.limitations
    assert answer.confidence_label


def test_composer_concise_does_not_enrich_structure():
    llm = FakeLLM()
    composer = AnswerComposer(llm)
    answer = composer.compose("solar power", _evidence(), structured=False)
    assert answer.summary == ""
    assert answer.key_takeaways == []


def test_composer_brief_populates_summary_and_splits_limitations():
    report = (
        "## Executive Summary\n"
        "Solar power generates electricity from sunlight using photovoltaic panels. [c1]\n\n"
        "## Overview\n"
        "Solar power generates electricity from sunlight using photovoltaic panels. [c1]\n\n"
        "## Limitations\n"
        "Solar power generates electricity from sunlight using photovoltaic panels. [c1]\n"
    )
    llm = FakeLLM(scripted={"renewable solar": report})
    composer = AnswerComposer(llm)
    answer = composer.compose("renewable solar", _evidence(), structured="brief")
    # Brief still enriches: summary is populated and limitations are split out of the body.
    assert answer.summary
    assert "## Limitations" not in answer.text
    assert answer.limitations
    assert answer.confidence_label


def test_composer_rejects_empty_llm_completion():
    llm = FakeLLM(scripted={"renewable solar": " \n\t"})
    composer = AnswerComposer(llm)

    with pytest.raises(BaselineError) as excinfo:
        composer.compose("renewable solar", _evidence(), structured="brief")

    assert excinfo.value.code is ErrorCode.UPSTREAM_ERROR


def test_composer_brief_prompt_uses_lightweight_skeleton():
    llm = FakeLLM()
    composer = AnswerComposer(llm)
    composer.compose("solar power", _evidence(), structured="brief")
    user_prompt = next(m["content"] for m in reversed(llm.calls[-1]) if m["role"] == "user")
    # The fast briefing skeleton omits the heavier Key Takeaways / Conclusion sections.
    assert "## Executive Summary" in user_prompt
    assert "## Evidence" in user_prompt
    assert "## Key Takeaways" not in user_prompt
    assert "## Conclusion" not in user_prompt
    assert "## <Themed section title>" not in user_prompt
    assert "do not add or rename headings" in user_prompt
    assert "up to 3 one-sentence bullets" in user_prompt
    assert "State each fact only once" in user_prompt
    assert "omit this section when the summary is sufficient" in user_prompt
    assert "Do not infer limitations from a title, URL, metadata" in user_prompt


def test_composer_system_prompt_matches_strict_claim_binding_contract():
    llm = FakeLLM()
    AnswerComposer(llm).compose("solar power", _evidence(), structured="brief")

    system_prompt = next(message["content"] for message in llm.calls[-1] if message["role"] == "system")

    assert "exactly one substantive claim in each sentence or bullet" in system_prompt
    assert "individually supports the entire claim" in system_prompt
    assert "Do not turn examples into broader" in system_prompt
    assert "Preserve the cited excerpt's exact predicate and modality" in system_prompt
    assert "A predicate in the question is not evidence" in system_prompt
    assert "Source titles, URLs, and provenance metadata identify sources but do not support claims" in system_prompt
    assert "If a heading itself makes a factual claim, cite the heading" in system_prompt


def test_composer_full_skeleton_mandates_conclusion():
    llm = FakeLLM()
    composer = AnswerComposer(llm)
    composer.compose("solar power", _evidence(), structured=True)
    user_prompt = next(m["content"] for m in reversed(llm.calls[-1]) if m["role"] == "user")
    assert "## Conclusion" in user_prompt
    assert "## Key Takeaways" in user_prompt


def test_composer_prompt_exposes_evidence_provenance_and_distinct_time_fields():
    llm = FakeLLM()
    evidence = [
        EvidenceItem(
            chunk_id=0,
            document_id="",
            url="",
            title="Archived project memory",
            text="Project Orion launch target is July 2026.",
            score=0.9,
            origin="memory",
            artifact_type="memory",
            artifact_id="mem_1",
            provenance_refs=["memory:memory:mem_1"],
            source_status="archived",
            published_at=1_719_792_000.0,
            effective_at=1_717_200_000.0,
            retrieved_at=1_720_000_000.0,
        )
    ]

    AnswerComposer(llm).compose("Orion launch", evidence)

    user_prompt = next(m["content"] for m in reversed(llm.calls[-1]) if m["role"] == "user")
    assert "origin=memory" in user_prompt
    assert "artifact=memory:mem_1" in user_prompt
    assert "status=archived" in user_prompt
    assert "published_at=" in user_prompt
    assert "effective_at=" in user_prompt
    assert "retrieved_at=" in user_prompt
