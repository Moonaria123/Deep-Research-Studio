"""Conservative evidence-backed Wiki Map cross-link construction."""

from __future__ import annotations

from collections import defaultdict

from .wiki_models import JsonDict, WikiCrossLinkDraft, WikiTopicDraft


def build_conservative_cross_links(
    topics: list[WikiTopicDraft],
    *,
    session_evidence: dict[str, set[str]],
    limit: int = 120,
) -> list[WikiCrossLinkDraft]:
    """Link v2 topics only when concept and deterministic evidence both agree."""

    topics_by_concept: dict[str, list[WikiTopicDraft]] = defaultdict(list)
    for topic in topics:
        session_id = _first_session(topic)
        if not session_id:
            continue
        concept_key = _concept_key(topic)
        if topic.attrs.get("kind") == "topic" and concept_key and _confidence(topic) == "high":
            topics_by_concept[concept_key].append(topic)

    links: list[WikiCrossLinkDraft] = []
    seen: set[tuple[str, str, str]] = set()
    for concept_key, concept_topics in sorted(topics_by_concept.items()):
        sorted_topics = sorted(concept_topics, key=lambda item: item.topic_key)
        for left_index, left_topic in enumerate(sorted_topics):
            left_session = _first_session(left_topic)
            for right_topic in sorted_topics[left_index + 1 :]:
                right_session = _first_session(right_topic)
                if not left_session or left_session == right_session:
                    continue
                shared = sorted(
                    (_topic_shared_evidence(left_topic, session_evidence) & _topic_shared_evidence(right_topic, session_evidence))
                )
                if not shared:
                    continue
                src_key, dst_key = sorted([left_topic.topic_key, right_topic.topic_key])
                edge_key = (src_key, dst_key, shared[0])
                if edge_key in seen:
                    continue
                seen.add(edge_key)
                links.append(
                    WikiCrossLinkDraft(
                        src_key=src_key,
                        dst_key=dst_key,
                        evidence_type="shared_kg_evidence",
                        evidence_refs=tuple(_evidence_ref(key) for key in shared[:5]),
                        attrs={
                            "source": "d7_conservative_v2",
                            "sessions": sorted([left_session, right_session]),
                            "concept_key": concept_key,
                            "confidence": "high",
                            "reason": _reason(_concept_label(left_topic, concept_key), len(shared)),
                        },
                    )
                )
                if len(links) >= limit:
                    return links
    return links


def _topic_shared_evidence(topic: WikiTopicDraft, session_evidence: dict[str, set[str]]) -> set[str]:
    refs = {
        str(ref.get("id") or ref.get("node_key") or "")
        for ref in _evidence_refs(topic)
        if str(ref.get("type") or _prefix(str(ref.get("id") or ref.get("node_key") or "")))
        in {"source", "note", "report", "memory"}
    }
    refs = {ref for ref in refs if _prefix(ref) in {"source", "note", "report", "memory"}}
    if refs:
        return refs
    session_id = _first_session(topic)
    return set(session_evidence.get(session_id) or set())


def _evidence_refs(topic: WikiTopicDraft) -> list[JsonDict]:
    refs = topic.attrs.get("evidence_refs")
    return refs if isinstance(refs, list) else []


def _concept_key(topic: WikiTopicDraft) -> str:
    return str(topic.attrs.get("concept_key") or "").strip().lower()


def _confidence(topic: WikiTopicDraft) -> str:
    return str(topic.attrs.get("confidence") or "").strip().lower()


def _concept_label(topic: WikiTopicDraft, concept_key: str) -> str:
    return str(topic.title or concept_key or "this concept")[:80]


def _reason(concept_label: str, shared_count: int) -> str:
    noun = "item" if shared_count == 1 else "items"
    return f"Both sessions discuss {concept_label} and share {shared_count} evidence {noun}."


def _prefix(key: str) -> str:
    text = str(key or "")
    if ":" not in text:
        return ""
    prefix, _ = text.split(":", 1)
    return prefix


def _first_session(topic: WikiTopicDraft) -> str:
    return str(topic.session_ids[0]) if topic.session_ids else str(topic.attrs.get("session_id") or "")


def _evidence_ref(key: str) -> JsonDict:
    text = str(key or "")
    prefix = _prefix(text)
    if not prefix:
        return {"id": text, "node_key": text, "type": "unknown"}
    return {"id": text, "node_key": text, "type": prefix}
