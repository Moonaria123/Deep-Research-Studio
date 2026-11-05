"""Wiki Map v2 schema and quality gate contracts."""

from __future__ import annotations

import pytest

from deepresearch_core.baseline.knowledge_graph.wiki_schema import (
    WikiSchemaError,
    safe_wiki_concept_key,
    session_wiki_tree_quality,
    validate_session_wiki_tree,
)


def _topic(**overrides):
    topic = {
        "title": "Risk taxonomy",
        "summary": "Groups model and operational risks into a reviewable structure.",
        "concept_key": "risk-taxonomy",
        "confidence": "high",
        "evidence_refs": [
            {
                "type": "message",
                "id": "msg_1",
                "excerpt": "The answer grouped model and operational risks.",
            }
        ],
        "source_message_ids": ["msg_1"],
        "children": [],
    }
    topic.update(overrides)
    return topic


def test_session_wiki_tree_v2_accepts_evidence_confidence_and_concept_key() -> None:
    tree = validate_session_wiki_tree({"topics": [_topic()]}, allowed_evidence_ids={"msg_1"})

    assert tree.topics[0].concept_key == "risk-taxonomy"
    assert tree.topics[0].confidence == "high"
    assert tree.topics[0].evidence_refs[0].id == "msg_1"


def test_safe_wiki_concept_key_is_public_and_stable() -> None:
    assert safe_wiki_concept_key("Risk taxonomy / 风险分类") == "risk-taxonomy-风险分类"
    assert safe_wiki_concept_key("") == "topic"


def test_session_wiki_tree_quality_is_recomputed_from_validated_topics() -> None:
    child = _topic(title="Operational controls", concept_key="operational-controls")
    tree = validate_session_wiki_tree(
        {"topics": [_topic(children=[child])]},
        allowed_evidence_ids={"msg_1"},
    )

    quality = session_wiki_tree_quality(tree)

    assert quality == {
        "topic_count": 2,
        "top_level_topic_count": 1,
        "max_depth": 2,
        "topics_with_evidence": 2,
        "evidence_coverage": 1.0,
        "question_like_title_count": 0,
        "invalid_summary_count": 0,
    }


@pytest.mark.parametrize(
    ("topic", "error_code"),
    [
        (_topic(evidence_refs=[]), "MISSING_EVIDENCE"),
        (_topic(evidence_refs=[{"type": "message", "id": "msg_unknown", "excerpt": "Unknown."}]), "INVALID_EVIDENCE_REF"),
        (_topic(title="https://example.com/raw"), "LOW_QUALITY_TOPIC_TITLE"),
        (_topic(title="memory:abc123"), "LOW_QUALITY_TOPIC_TITLE"),
        (_topic(title="What are the risks?"), "LOW_QUALITY_TOPIC_TITLE"),
        (_topic(summary=""), "LOW_QUALITY_TOPIC_SUMMARY"),
        (_topic(summary="One sentence. Two sentences. Three sentences."), "LOW_QUALITY_TOPIC_SUMMARY"),
        (_topic(summary="Token " + ("sk-" + "testsecret123456") + " must not leak."), "SECRET_LIKE_CONTENT"),
    ],
)
def test_session_wiki_tree_v2_rejects_low_quality_or_untrusted_topics(topic, error_code: str) -> None:
    with pytest.raises(WikiSchemaError) as exc:
        validate_session_wiki_tree({"topics": [topic]}, allowed_evidence_ids={"msg_1"})

    assert exc.value.error_code == error_code


def test_session_wiki_tree_rejects_more_than_seven_top_level_topics() -> None:
    topics = [_topic(title=f"Risk category {index}", concept_key=f"risk-{index}") for index in range(8)]

    with pytest.raises(WikiSchemaError) as exc:
        validate_session_wiki_tree({"topics": topics}, allowed_evidence_ids={"msg_1"})

    assert exc.value.error_code == "TOO_MANY_TOP_LEVEL_TOPICS"
