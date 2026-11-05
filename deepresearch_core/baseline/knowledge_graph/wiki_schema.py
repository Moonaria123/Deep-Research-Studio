"""Schema validation for Wiki Map model output."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

JsonDict = dict[str, Any]
CONFIDENCE_VALUES = {"high", "medium", "low"}
MAX_TOP_LEVEL_TOPICS = 7
MAX_TOPIC_TITLE = 90
MAX_TOPIC_SUMMARY = 280
_SECRET_RE = re.compile(
    r"((?<![A-Za-z0-9])sk-[A-Za-z0-9_-]{6,}|(?<![A-Za-z0-9])ghp_[A-Za-z0-9_]{6,}|AKIA[0-9A-Z]{8,}|"
    r"(?<![A-Za-z0-9])xox[abp]-[A-Za-z0-9-]{6,}|-----BEGIN [A-Z ]*PRIVATE KEY-----)",
    re.IGNORECASE,
)
_RAW_TITLE_RE = re.compile(r"^(https?://|[a-z]+:[A-Za-z0-9_-]{4,}|(?:memory|mem|note|source|session)[:\s_-])", re.IGNORECASE)
_QUESTION_TITLE_RE = re.compile(r"[?？]\s*$")
_SENTENCE_BOUNDARY_RE = re.compile(r"[.!?。！？；;]+")


class WikiSchemaError(ValueError):
    """Raised when structured Wiki Map output does not match the required schema."""

    def __init__(self, message: str, *, error_code: str = "SCHEMA_VALIDATION_FAILED") -> None:
        super().__init__(message)
        self.error_code = error_code


@dataclass(frozen=True)
class WikiEvidenceRef:
    """A bounded citation for one generated topic."""

    type: str
    id: str
    excerpt: str = ""

    def to_dict(self) -> JsonDict:
        return {"type": self.type, "id": self.id, "excerpt": self.excerpt}


@dataclass(frozen=True)
class WikiTopic:
    """A validated topic tree node."""

    title: str
    summary: str = ""
    concept_key: str = ""
    confidence: str = "medium"
    evidence_refs: tuple[WikiEvidenceRef, ...] = ()
    source_message_ids: tuple[str, ...] = ()
    children: tuple["WikiTopic", ...] = field(default_factory=tuple)

    def to_dict(self) -> JsonDict:
        return {
            "title": self.title,
            "summary": self.summary,
            "concept_key": self.concept_key,
            "confidence": self.confidence,
            "evidence_refs": [ref.to_dict() for ref in self.evidence_refs],
            "source_message_ids": list(self.source_message_ids),
            "children": [child.to_dict() for child in self.children],
        }


@dataclass(frozen=True)
class SessionWikiTree:
    """Validated per-session Wiki Map output."""

    topics: tuple[WikiTopic, ...]

    def to_dict(self) -> JsonDict:
        return {"topics": [topic.to_dict() for topic in self.topics]}


SESSION_WIKI_TREE_JSON_SCHEMA: JsonDict = {
    "type": "json_schema",
    "json_schema": {
        "name": "session_wiki_tree",
        "strict": True,
        "schema": {
            "$defs": {
                "topic": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["title", "summary", "concept_key", "confidence", "evidence_refs", "source_message_ids", "children"],
                    "properties": {
                        "title": {"type": "string", "minLength": 1},
                        "summary": {"type": "string"},
                        "concept_key": {"type": "string", "minLength": 1},
                        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                        "evidence_refs": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "required": ["type", "id", "excerpt"],
                                "properties": {
                                    "type": {"type": "string", "minLength": 1},
                                    "id": {"type": "string", "minLength": 1},
                                    "excerpt": {"type": "string"},
                                },
                            },
                        },
                        "source_message_ids": {"type": "array", "items": {"type": "string"}},
                        "children": {"type": "array", "items": {"$ref": "#/$defs/topic"}},
                    },
                }
            },
            "type": "object",
            "additionalProperties": False,
            "required": ["topics"],
            "properties": {"topics": {"type": "array", "items": {"$ref": "#/$defs/topic"}}},
        },
    },
}


def validate_session_wiki_tree(data: Any, *, allowed_evidence_ids: set[str] | None = None) -> SessionWikiTree:
    """Validate SessionWikiTree JSON and return a normalized object."""

    obj = _require_object(data)
    raw_topics = obj.get("topics")
    if not isinstance(raw_topics, list):
        raise WikiSchemaError("topics must be a list")
    if len(raw_topics) > MAX_TOP_LEVEL_TOPICS:
        raise WikiSchemaError(
            f"topics cannot contain more than {MAX_TOP_LEVEL_TOPICS} top-level items",
            error_code="TOO_MANY_TOP_LEVEL_TOPICS",
        )
    topics = tuple(_topic(raw, depth=0, allowed_evidence_ids=allowed_evidence_ids) for raw in raw_topics)
    if not topics:
        raise WikiSchemaError("topics cannot be empty", error_code="EMPTY_CONTENT")
    return SessionWikiTree(topics=topics)


def validate_wiki_topic_context(tree: SessionWikiTree, *, disallowed_titles: set[str]) -> None:
    """Reject topic titles that mechanically repeat the Session title or user query."""

    disallowed = {safe_wiki_concept_key(value) for value in disallowed_titles if _clean_text(value)}

    def walk(topics: tuple[WikiTopic, ...]) -> None:
        for topic in topics:
            if safe_wiki_concept_key(topic.title) in disallowed:
                raise WikiSchemaError(
                    "topic title must not repeat the Session title or user query",
                    error_code="LOW_QUALITY_TOPIC_TITLE",
                )
            walk(topic.children)

    walk(tree.topics)


def session_wiki_tree_quality(tree: SessionWikiTree) -> JsonDict:
    """Recompute public quality metrics from validated topics."""

    topic_count = 0
    topics_with_evidence = 0
    max_depth = 0
    question_like_title_count = 0
    invalid_summary_count = 0

    def walk(topics: tuple[WikiTopic, ...], depth: int) -> None:
        nonlocal topic_count, topics_with_evidence, max_depth, question_like_title_count, invalid_summary_count
        for topic in topics:
            topic_count += 1
            topics_with_evidence += int(bool(topic.evidence_refs))
            max_depth = max(max_depth, depth)
            question_like_title_count += int(bool(_QUESTION_TITLE_RE.search(topic.title)))
            invalid_summary_count += int(not _summary_is_valid(topic.summary))
            walk(topic.children, depth + 1)

    walk(tree.topics, 1)
    return {
        "topic_count": topic_count,
        "top_level_topic_count": len(tree.topics),
        "max_depth": max_depth,
        "topics_with_evidence": topics_with_evidence,
        "evidence_coverage": topics_with_evidence / max(topic_count, 1),
        "question_like_title_count": question_like_title_count,
        "invalid_summary_count": invalid_summary_count,
    }


def validate_wiki_summary(value: str) -> str:
    """Validate one bounded one-to-two-sentence Wiki summary."""

    summary = _clean_text(value)
    _reject_secret_like(summary, "summary")
    if not _summary_is_valid(summary):
        raise WikiSchemaError(
            "topic summary must contain one or two bounded sentences",
            error_code="LOW_QUALITY_TOPIC_SUMMARY",
        )
    return summary


def validate_wiki_topic_text(title: str, summary: str) -> tuple[str, str]:
    """Validate deterministic title and summary quality for fresh or carried topics."""

    cleaned_title = _clean_text(title)
    _reject_secret_like(cleaned_title, "title")
    if (
        not cleaned_title
        or len(cleaned_title) > MAX_TOPIC_TITLE
        or _RAW_TITLE_RE.search(cleaned_title)
        or _QUESTION_TITLE_RE.search(cleaned_title)
    ):
        raise WikiSchemaError(
            "topic title must be a bounded knowledge concept",
            error_code="LOW_QUALITY_TOPIC_TITLE",
        )
    return cleaned_title, validate_wiki_summary(summary)


def _topic(data: Any, *, depth: int, allowed_evidence_ids: set[str] | None) -> WikiTopic:
    if depth > 5:
        raise WikiSchemaError("topic tree is too deep")
    obj = _require_object(data)
    raw_children = obj.get("children", [])
    if raw_children is None:
        raw_children = []
    if not isinstance(raw_children, list):
        raise WikiSchemaError("children must be a list")
    title = _required_string(obj, "title")
    summary = _optional_string(obj, "summary")
    title, summary = validate_wiki_topic_text(title, summary)
    concept_key = safe_wiki_concept_key(_required_string(obj, "concept_key") if "concept_key" in obj else title)
    confidence = _optional_string(obj, "confidence") or "medium"
    if confidence not in CONFIDENCE_VALUES:
        raise WikiSchemaError("confidence must be high, medium, or low")
    evidence_refs = tuple(_evidence_ref(item, allowed_evidence_ids=allowed_evidence_ids) for item in _required_list(obj, "evidence_refs")[:8])
    if not evidence_refs:
        raise WikiSchemaError("topic requires at least one evidence ref", error_code="MISSING_EVIDENCE")
    return WikiTopic(
        title=title,
        summary=summary,
        concept_key=concept_key,
        confidence=confidence,
        evidence_refs=evidence_refs,
        source_message_ids=tuple(_safe_id(item) for item in _optional_string_list(obj, "source_message_ids")),
        children=tuple(_topic(child, depth=depth + 1, allowed_evidence_ids=allowed_evidence_ids) for child in raw_children[:12]),
    )


def _require_object(data: Any) -> JsonDict:
    if not isinstance(data, dict):
        raise WikiSchemaError("wiki output must be a JSON object", error_code="NON_OBJECT_JSON")
    return data


def _required_string(data: JsonDict, key: str) -> str:
    if key not in data:
        raise WikiSchemaError(f"missing required field: {key}")
    value = data.get(key)
    if not isinstance(value, str):
        raise WikiSchemaError(f"{key} must be a string")
    cleaned = _clean_text(value)
    if not cleaned:
        raise WikiSchemaError(f"{key} is empty", error_code="EMPTY_CONTENT")
    return cleaned


def _optional_string(data: JsonDict, key: str) -> str:
    value = data.get(key)
    if value is None:
        return ""
    if not isinstance(value, str):
        raise WikiSchemaError(f"{key} must be a string")
    return _clean_text(value)


def _optional_string_list(data: JsonDict, key: str) -> list[str]:
    value = data.get(key)
    if value is None:
        return []
    if not isinstance(value, list):
        raise WikiSchemaError(f"{key} must be a list")
    cleaned: list[str] = []
    for item in value[:20]:
        if isinstance(item, str):
            text = _safe_id(item)
            if text:
                cleaned.append(text)
    return cleaned


def _required_list(data: JsonDict, key: str) -> list[Any]:
    if key not in data:
        raise WikiSchemaError(f"missing required field: {key}")
    value = data.get(key)
    if not isinstance(value, list):
        raise WikiSchemaError(f"{key} must be a list")
    return value


def _evidence_ref(data: Any, *, allowed_evidence_ids: set[str] | None) -> WikiEvidenceRef:
    obj = _require_object(data)
    ref_type = _safe_id(_required_string(obj, "type"))[:24]
    ref_id = _safe_id(_required_string(obj, "id"))
    excerpt = _optional_string(obj, "excerpt")
    _reject_secret_like(excerpt, "evidence excerpt")
    if allowed_evidence_ids is not None and ref_id not in allowed_evidence_ids:
        raise WikiSchemaError("evidence ref is not in the allowed catalog", error_code="INVALID_EVIDENCE_REF")
    return WikiEvidenceRef(type=ref_type or "unknown", id=ref_id, excerpt=excerpt[:240])


def _safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9:_-]+", "", str(value or ""))[:80]


def safe_wiki_concept_key(value: str) -> str:
    """Return the stable public concept-key normalization used by Wiki Map."""

    text = re.sub(r"[^A-Za-z0-9\u4e00-\u9fff]+", "-", str(value or "").lower()).strip("-")
    return text[:80] or "topic"


def _reject_secret_like(value: str, label: str) -> None:
    if _SECRET_RE.search(str(value or "")):
        raise WikiSchemaError(f"{label} contains secret-like content", error_code="SECRET_LIKE_CONTENT")


def _clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _summary_is_valid(value: str) -> bool:
    text = _clean_text(value)
    if not text or len(text) > MAX_TOPIC_SUMMARY:
        return False
    sentences = [part.strip() for part in _SENTENCE_BOUNDARY_RE.split(text) if part.strip()]
    return 1 <= len(sentences) <= 2
