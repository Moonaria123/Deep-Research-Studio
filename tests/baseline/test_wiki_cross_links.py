"""Wiki Map v2 high-confidence cross-link contracts."""

from __future__ import annotations

from deepresearch_core.baseline.knowledge_graph.wiki_cross_links import build_conservative_cross_links
from deepresearch_core.baseline.knowledge_graph.wiki_models import WikiTopicDraft


def _topic(
    session_id: str,
    key: str,
    *,
    concept_key: str = "risk-taxonomy",
    confidence: str = "high",
    evidence_id: str = "source:doc1",
    evidence_type: str = "source",
) -> WikiTopicDraft:
    return WikiTopicDraft(
        topic_key=key,
        title="Risk taxonomy",
        depth=2,
        session_ids=(session_id,),
        attrs={
            "kind": "topic",
            "schema_version": "2",
            "concept_key": concept_key,
            "confidence": confidence,
            "evidence_refs": [{"type": evidence_type, "id": evidence_id, "excerpt": "Safety source"}],
        },
    )


def test_cross_links_require_shared_concept_high_confidence_and_evidence() -> None:
    links = build_conservative_cross_links(
        [
            _topic("s1", "wiki:topic:s1:risk:aaa"),
            _topic("s2", "wiki:topic:s2:risk:bbb"),
            _topic("s3", "wiki:topic:s3:risk:ccc", confidence="low"),
        ],
        session_evidence={"s1": {"source:doc1"}, "s2": {"source:doc1"}, "s3": {"source:doc1"}},
    )

    assert len(links) == 1
    assert links[0].src_key != links[0].dst_key
    assert links[0].evidence_type == "shared_kg_evidence"
    assert links[0].attrs["concept_key"] == "risk-taxonomy"
    assert "share 1 evidence" in links[0].attrs["reason"]


def test_cross_links_reject_same_concept_without_shared_source_or_memory_evidence() -> None:
    links = build_conservative_cross_links(
        [
            _topic("s1", "wiki:topic:s1:risk:aaa", evidence_id="source:doc1"),
            _topic("s2", "wiki:topic:s2:risk:bbb", evidence_id="source:doc2"),
        ],
        session_evidence={"s1": {"source:doc1"}, "s2": {"source:doc2"}},
    )

    assert links == []


def test_cross_links_accept_same_stable_note_report_or_memory_without_semantic_inference() -> None:
    for evidence_type, evidence_id in (
        ("note", "note:n1"),
        ("report", "report:r1"),
        ("memory", "memory:m1"),
    ):
        links = build_conservative_cross_links(
            [
                _topic("s1", f"wiki:topic:s1:{evidence_type}:aaa", evidence_id=evidence_id, evidence_type=evidence_type),
                _topic("s2", f"wiki:topic:s2:{evidence_type}:bbb", evidence_id=evidence_id, evidence_type=evidence_type),
            ],
            session_evidence={"s1": {evidence_id}, "s2": {evidence_id}},
        )

        assert len(links) == 1
        assert links[0].evidence_refs == ({"id": evidence_id, "node_key": evidence_id, "type": evidence_type},)
