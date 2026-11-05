"""T45 Wiki partial snapshot activation contracts."""

from __future__ import annotations

from deepresearch_core.baseline.knowledge_graph.wiki_activation import (
    annotate_session_roots,
    carry_forward_session,
    evaluate_activation,
)
from deepresearch_core.baseline.knowledge_graph.wiki_models import WikiTopicDraft


def _stored_session(session_id: str, evidence_id: str = "msg-1") -> list[dict]:
    root_id = f"root-{session_id}"
    return [
        {
            "id": root_id,
            "snapshot_id": "snapshot-old",
            "key": f"wiki:session:{session_id}",
            "title": f"Session {session_id}",
            "summary": "A stable session summary.",
            "depth": 1,
            "parent_id": "workspace-root",
            "session_ids": [session_id],
            "source_message_ids": [],
            "attrs": {"kind": "session_root", "session_id": session_id},
        },
        {
            "id": f"topic-{session_id}",
            "snapshot_id": "snapshot-old",
            "key": f"wiki:topic:{session_id}:risk",
            "title": "Risk taxonomy",
            "summary": "Groups the evidence into a stable taxonomy.",
            "depth": 2,
            "parent_id": root_id,
            "session_ids": [session_id],
            "source_message_ids": [evidence_id],
            "attrs": {
                "kind": "topic",
                "session_id": session_id,
                "evidence_refs": [{"type": "message", "id": evidence_id, "excerpt": "Bounded evidence."}],
            },
        },
    ]


def _draft_session(session_id: str, *, with_topic: bool = True, state: str = "fresh") -> list[WikiTopicDraft]:
    root = WikiTopicDraft(
        topic_key=f"wiki:session:{session_id}",
        title=f"Session {session_id}",
        summary="A stable session summary.",
        depth=1,
        parent_key="wiki:root",
        session_ids=(session_id,),
        attrs={"kind": "session_root", "session_id": session_id, "generation_state": state},
    )
    if not with_topic:
        return [root]
    return [
        root,
        WikiTopicDraft(
            topic_key=f"wiki:topic:{session_id}:risk",
            title="Risk taxonomy",
            summary="Groups the evidence into a stable taxonomy.",
            depth=2,
            parent_key=root.topic_key,
            session_ids=(session_id,),
            source_message_ids=(f"msg-{session_id}",),
            attrs={
                "kind": "topic",
                "session_id": session_id,
                "generation_state": state,
                "evidence_refs": [{"type": "message", "id": f"msg-{session_id}"}],
            },
        ),
    ]


def test_carry_forward_session_revalidates_evidence_and_marks_stale() -> None:
    carried = carry_forward_session(
        _stored_session("s1"),
        session_id="s1",
        allowed_evidence_ids={"msg-1"},
        source_snapshot_id="snapshot-old",
    )

    assert carried is not None
    assert [topic.attrs["kind"] for topic in carried.topics] == ["session_root", "topic"]
    assert all(topic.attrs["generation_state"] == "carried" for topic in carried.topics)
    assert all(topic.attrs["stale"] is True for topic in carried.topics)
    assert carried.topics[1].attrs["evidence_refs"] == [
        {"type": "message", "id": "msg-1", "excerpt": "Bounded evidence."}
    ]
    assert carried.edges[0].src_key == "wiki:root"
    assert carried.edges[1].src_key == "wiki:session:s1"


def test_carry_forward_session_rejects_tree_when_evidence_is_no_longer_allowed() -> None:
    carried = carry_forward_session(
        _stored_session("s1"),
        session_id="s1",
        allowed_evidence_ids={"different-message"},
        source_snapshot_id="snapshot-old",
    )

    assert carried is None


def test_carry_forward_session_rejects_legacy_tree_with_invalid_topic_summary() -> None:
    stored = _stored_session("s1")
    stored[1]["summary"] = ""

    carried = carry_forward_session(
        stored,
        session_id="s1",
        allowed_evidence_ids={"msg-1"},
        source_snapshot_id="snapshot-old",
    )

    assert carried is None


def test_activation_rejects_coverage_regression_and_orphan_roots() -> None:
    active = [*_stored_session("s1", "msg-s1"), *_stored_session("s2", "msg-s2")]
    coverage_regression = evaluate_activation(
        candidate_topics=_draft_session("s1"),
        active_topics=active,
        eligible_session_ids={"s1", "s2"},
        fresh_session_ids={"s1"},
        carried_session_ids=set(),
    )

    assert coverage_regression.activate is False
    assert coverage_regression.reason == "WIKI_COVERAGE_REGRESSION"
    assert coverage_regression.candidate_coverage == 0.5
    assert coverage_regression.active_coverage == 1.0

    orphan = evaluate_activation(
        candidate_topics=[*_draft_session("s1"), *_draft_session("s2", with_topic=False)],
        active_topics=active,
        eligible_session_ids={"s1", "s2"},
        fresh_session_ids={"s1"},
        carried_session_ids=set(),
    )
    assert orphan.activate is False
    assert orphan.reason == "WIKI_ORPHAN_ROOT_REJECTED"


def test_activation_requires_fresh_content_and_accepts_non_regressing_partial() -> None:
    active = [*_stored_session("s1", "msg-s1"), *_stored_session("s2", "msg-s2")]
    candidate = [*_draft_session("s1"), *_draft_session("s2", state="carried")]

    no_fresh = evaluate_activation(
        candidate_topics=candidate,
        active_topics=active,
        eligible_session_ids={"s1", "s2"},
        fresh_session_ids=set(),
        carried_session_ids={"s1", "s2"},
    )
    assert no_fresh.activate is False
    assert no_fresh.reason == "WIKI_NO_FRESH_TREE"

    accepted = evaluate_activation(
        candidate_topics=candidate,
        active_topics=active,
        eligible_session_ids={"s1", "s2"},
        fresh_session_ids={"s1"},
        carried_session_ids={"s2"},
    )
    assert accepted.activate is True
    assert accepted.reason == "activated"
    assert accepted.status == "partial"
    assert accepted.fresh_sessions == 1
    assert accepted.carried_sessions == 1
    assert accepted.uncovered_sessions == 0


def test_session_root_annotation_exposes_count_refresh_time_and_generation_state() -> None:
    topics = [*_draft_session("s1"), *_draft_session("s2", state="carried")]

    annotated = annotate_session_roots(topics, refreshed_at=1_750_000_000.0)
    roots = {topic.attrs["session_id"]: topic for topic in annotated if topic.attrs["kind"] == "session_root"}

    assert roots["s1"].attrs["topic_count"] == 1
    assert roots["s1"].attrs["last_refreshed_at"] == 1_750_000_000.0
    assert roots["s1"].attrs["generation_state"] == "fresh"
    assert roots["s1"].attrs["stale"] is False
    assert roots["s2"].attrs["topic_count"] == 1
    assert roots["s2"].attrs["generation_state"] == "carried"
    assert roots["s2"].attrs["stale"] is True
