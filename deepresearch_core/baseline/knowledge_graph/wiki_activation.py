"""Session-level activation rules for Wiki Map partial snapshots."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from .wiki_models import JsonDict, WikiTopicDraft, WikiTreeEdgeDraft
from .wiki_schema import WikiSchemaError, validate_wiki_summary, validate_wiki_topic_text


@dataclass(frozen=True)
class CarriedSession:
    """A previously active Session tree revalidated for a new snapshot."""

    topics: tuple[WikiTopicDraft, ...]
    edges: tuple[WikiTreeEdgeDraft, ...]


@dataclass(frozen=True)
class ActivationDecision:
    """Deterministic decision for promoting a staged Wiki snapshot."""

    activate: bool
    reason: str
    status: str
    eligible_sessions: int
    fresh_sessions: int
    carried_sessions: int
    uncovered_sessions: int
    candidate_coverage: float
    active_coverage: float
    orphan_sessions: tuple[str, ...] = ()

    def to_stats(self) -> JsonDict:
        """Return the bounded public stats stored with the refresh."""

        return {
            "eligible_sessions": self.eligible_sessions,
            "fresh_sessions": self.fresh_sessions,
            "carried_sessions": self.carried_sessions,
            "uncovered_sessions": self.uncovered_sessions,
            "candidate_coverage": self.candidate_coverage,
            "active_coverage": self.active_coverage,
            "activation_decision": "activated" if self.activate else "retained_previous",
            "activation_reason": self.reason,
            "orphan_sessions": len(self.orphan_sessions),
        }


def carry_forward_session(
    stored_topics: Iterable[JsonDict],
    *,
    session_id: str,
    allowed_evidence_ids: set[str],
    source_snapshot_id: str,
) -> CarriedSession | None:
    """Rebuild one old Session subtree after revalidating its evidence refs."""

    session_id = str(session_id or "").strip()
    if not session_id or not allowed_evidence_ids:
        return None

    rows = [topic for topic in stored_topics if _stored_session_id(topic) == session_id]
    rows.sort(key=lambda topic: (int(topic.get("depth") or 0), str(topic.get("key") or "")))
    if not rows:
        return None

    key_by_id = {str(topic.get("id") or ""): str(topic.get("key") or "") for topic in rows}
    retained_keys: set[str] = set()
    drafts: list[WikiTopicDraft] = []

    for row in rows:
        kind = _stored_kind(row)
        topic_key = str(row.get("key") or "")
        if not topic_key or kind not in {"session_root", "topic"}:
            continue

        if kind == "session_root":
            parent_key = "wiki:root"
            evidence_refs: list[JsonDict] = []
            try:
                title = str(row.get("title") or "")
                summary = validate_wiki_summary(str(row.get("summary") or ""))
            except WikiSchemaError:
                return None
        else:
            parent_key = key_by_id.get(str(row.get("parent_id") or ""), "")
            if not parent_key or parent_key not in retained_keys:
                continue
            evidence_refs = _allowed_evidence_refs(row, allowed_evidence_ids)
            if not evidence_refs:
                continue
            try:
                title, summary = validate_wiki_topic_text(
                    str(row.get("title") or ""),
                    str(row.get("summary") or ""),
                )
            except WikiSchemaError:
                continue

        attrs = dict(row.get("attrs") or {})
        attrs.update(
            {
                "generation_state": "carried",
                "source_snapshot_id": source_snapshot_id,
                "stale": True,
                "stale_reason": "session_generation_failed",
            }
        )
        if kind == "topic":
            attrs["evidence_refs"] = evidence_refs

        drafts.append(
            WikiTopicDraft(
                topic_key=topic_key,
                title=title,
                summary=summary,
                depth=int(row.get("depth") or 0),
                parent_key=parent_key,
                session_ids=tuple(str(value) for value in row.get("session_ids") or () if str(value)),
                source_message_ids=tuple(
                    str(value)
                    for value in row.get("source_message_ids") or ()
                    if str(value) in allowed_evidence_ids
                ),
                attrs=attrs,
            )
        )
        retained_keys.add(topic_key)

    if not any(topic.attrs.get("kind") == "topic" for topic in drafts):
        return None

    edges = tuple(
        WikiTreeEdgeDraft(src_key=topic.parent_key, dst_key=topic.topic_key)
        for topic in drafts
        if topic.parent_key
    )
    return CarriedSession(topics=tuple(drafts), edges=edges)


def mark_fresh(topics: Iterable[WikiTopicDraft]) -> tuple[WikiTopicDraft, ...]:
    """Return fresh copies without mutating generator-owned topic drafts."""

    fresh: list[WikiTopicDraft] = []
    for topic in topics:
        attrs = dict(topic.attrs or {})
        attrs.update({"generation_state": "fresh", "stale": False})
        attrs.pop("source_snapshot_id", None)
        attrs.pop("stale_reason", None)
        fresh.append(
            WikiTopicDraft(
                topic_key=topic.topic_key,
                title=topic.title,
                summary=topic.summary,
                depth=topic.depth,
                parent_key=topic.parent_key,
                session_ids=topic.session_ids,
                source_message_ids=topic.source_message_ids,
                attrs=attrs,
            )
        )
    return tuple(fresh)


def annotate_session_roots(
    topics: Iterable[WikiTopicDraft],
    *,
    refreshed_at: float,
) -> tuple[WikiTopicDraft, ...]:
    """Attach bounded display metadata to each Session root."""

    draft = tuple(topics)
    topic_counts: dict[str, int] = {}
    for topic in draft:
        if topic.attrs.get("kind") != "topic":
            continue
        session_id = _draft_session_id(topic)
        if session_id:
            topic_counts[session_id] = topic_counts.get(session_id, 0) + 1

    annotated: list[WikiTopicDraft] = []
    for topic in draft:
        attrs = dict(topic.attrs or {})
        if attrs.get("kind") == "session_root":
            session_id = _draft_session_id(topic)
            generation_state = str(attrs.get("generation_state") or "")
            attrs.update(
                {
                    "topic_count": topic_counts.get(session_id, 0),
                    "last_refreshed_at": float(refreshed_at),
                    "stale": generation_state == "carried",
                }
            )
        annotated.append(
            WikiTopicDraft(
                topic_key=topic.topic_key,
                title=topic.title,
                summary=topic.summary,
                depth=topic.depth,
                parent_key=topic.parent_key,
                session_ids=topic.session_ids,
                source_message_ids=topic.source_message_ids,
                attrs=attrs,
            )
        )
    return tuple(annotated)


def snapshot_quality_stats(topics: Iterable[WikiTopicDraft]) -> JsonDict:
    """Recompute aggregate quality metrics from the final staged draft."""

    draft = tuple(topics)
    kinds = {topic.topic_key: str(topic.attrs.get("kind") or "") for topic in draft}
    content_topics = [topic for topic in draft if topic.attrs.get("kind") == "topic"]
    topics_with_evidence = sum(
        1
        for topic in content_topics
        if isinstance(topic.attrs.get("evidence_refs"), list) and bool(topic.attrs.get("evidence_refs"))
    )
    top_level = sum(1 for topic in content_topics if kinds.get(topic.parent_key) == "session_root")
    max_depth = max((max(int(topic.depth) - 1, 1) for topic in content_topics), default=0)
    question_like = 0
    invalid_summary = 0
    for topic in content_topics:
        try:
            validate_wiki_topic_text(topic.title, topic.summary)
        except WikiSchemaError as error:
            if error.error_code == "LOW_QUALITY_TOPIC_TITLE":
                question_like += 1
            if error.error_code == "LOW_QUALITY_TOPIC_SUMMARY":
                invalid_summary += 1
    topic_count = len(content_topics)
    return {
        "topic_count": topic_count,
        "top_level_topic_count": top_level,
        "max_depth": max_depth,
        "topics_with_evidence": topics_with_evidence,
        "evidence_coverage": topics_with_evidence / max(topic_count, 1),
        "question_like_title_count": question_like,
        "invalid_summary_count": invalid_summary,
        "orphan_root_count": len(_orphan_draft_sessions(draft)),
    }


def evaluate_activation(
    *,
    candidate_topics: Iterable[WikiTopicDraft],
    active_topics: Iterable[JsonDict],
    eligible_session_ids: set[str],
    fresh_session_ids: set[str],
    carried_session_ids: set[str],
) -> ActivationDecision:
    """Compare staged and active Session coverage over the same eligible set."""

    eligible = {str(value) for value in eligible_session_ids if str(value)}
    candidate = tuple(candidate_topics)
    active = tuple(active_topics)
    candidate_covered = _draft_covered_sessions(candidate) & eligible
    active_covered = _stored_covered_sessions(active) & eligible
    fresh = {str(value) for value in fresh_session_ids if str(value)} & eligible
    carried = ({str(value) for value in carried_session_ids if str(value)} & eligible) - fresh
    orphan_sessions = tuple(sorted(_orphan_draft_sessions(candidate) & eligible))
    denominator = max(len(eligible), 1)

    if not fresh:
        activate = False
        reason = "WIKI_NO_FRESH_TREE"
    elif orphan_sessions:
        activate = False
        reason = "WIKI_ORPHAN_ROOT_REJECTED"
    elif len(candidate_covered) < len(active_covered):
        activate = False
        reason = "WIKI_COVERAGE_REGRESSION"
    else:
        activate = True
        reason = "activated"

    uncovered = eligible - candidate_covered
    status = "succeeded" if activate and fresh == eligible and not carried and not uncovered else "partial"
    return ActivationDecision(
        activate=activate,
        reason=reason,
        status=status,
        eligible_sessions=len(eligible),
        fresh_sessions=len(fresh),
        carried_sessions=len(carried),
        uncovered_sessions=len(uncovered),
        candidate_coverage=len(candidate_covered) / denominator,
        active_coverage=len(active_covered) / denominator,
        orphan_sessions=orphan_sessions,
    )


def _allowed_evidence_refs(topic: JsonDict, allowed_evidence_ids: set[str]) -> list[JsonDict]:
    refs = (topic.get("attrs") or {}).get("evidence_refs")
    if not isinstance(refs, list):
        return []
    return [
        dict(ref)
        for ref in refs
        if isinstance(ref, dict) and str(ref.get("id") or "") in allowed_evidence_ids
    ]


def _stored_kind(topic: JsonDict) -> str:
    return str((topic.get("attrs") or {}).get("kind") or "")


def _stored_session_id(topic: JsonDict) -> str:
    attrs = topic.get("attrs") or {}
    explicit = str(attrs.get("session_id") or "")
    if explicit:
        return explicit
    return next((str(value) for value in topic.get("session_ids") or () if str(value)), "")


def _draft_session_id(topic: WikiTopicDraft) -> str:
    explicit = str((topic.attrs or {}).get("session_id") or "")
    if explicit:
        return explicit
    return next((str(value) for value in topic.session_ids if str(value)), "")


def _draft_covered_sessions(topics: Iterable[WikiTopicDraft]) -> set[str]:
    return {
        session_id
        for topic in topics
        if topic.attrs.get("kind") == "topic"
        if (session_id := _draft_session_id(topic))
    }


def _stored_covered_sessions(topics: Iterable[JsonDict]) -> set[str]:
    return {
        session_id
        for topic in topics
        if _stored_kind(topic) == "topic"
        if (session_id := _stored_session_id(topic))
    }


def _orphan_draft_sessions(topics: Iterable[WikiTopicDraft]) -> set[str]:
    roots = {
        session_id
        for topic in topics
        if topic.attrs.get("kind") == "session_root"
        if (session_id := _draft_session_id(topic))
    }
    return roots - _draft_covered_sessions(topics)
