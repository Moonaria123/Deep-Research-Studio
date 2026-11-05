"""Schema validation for Memory distillation model output."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

JsonDict = dict[str, Any]


class MemorySchemaError(ValueError):
    """Raised when structured Memory output does not match the required schema."""

    def __init__(self, message: str, *, error_code: str = "SCHEMA_VALIDATION_FAILED") -> None:
        super().__init__(message)
        self.error_code = error_code


@dataclass(frozen=True)
class SessionMemorySummary:
    """Validated session-summary output."""

    summary: str

    def to_dict(self) -> JsonDict:
        return {"summary": self.summary}


@dataclass(frozen=True)
class UserProfileSection:
    """Validated user-profile section output."""

    id: str
    title: str
    body: str

    def to_dict(self) -> JsonDict:
        return {"id": self.id, "title": self.title, "body": self.body}


@dataclass(frozen=True)
class UserProfileSections:
    """Validated user-profile output."""

    sections: tuple[UserProfileSection, ...]

    def to_dict(self) -> JsonDict:
        return {"sections": [section.to_dict() for section in self.sections]}


@dataclass(frozen=True)
class MemoryFactCandidate:
    """Validated ADD-only fact candidate output."""

    content: str
    kind: str
    confidence: str

    def to_dict(self) -> JsonDict:
        return {"content": self.content, "kind": self.kind, "confidence": self.confidence}


@dataclass(frozen=True)
class MemoryFactCandidates:
    """Validated memory_fact_v1 candidate output."""

    memories: tuple[MemoryFactCandidate, ...]

    def to_dict(self) -> JsonDict:
        return {"memories": [memory.to_dict() for memory in self.memories]}


SESSION_SUMMARY_JSON_SCHEMA: JsonDict = {
    "type": "json_schema",
    "json_schema": {
        "name": "memory_session_summary",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["summary"],
            "properties": {"summary": {"type": "string", "minLength": 1}},
        },
    },
}

USER_PROFILE_JSON_SCHEMA: JsonDict = {
    "type": "json_schema",
    "json_schema": {
        "name": "memory_user_profile",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["sections"],
            "properties": {
                "sections": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["id", "title", "body"],
                        "properties": {
                            "id": {"type": "string", "minLength": 1},
                            "title": {"type": "string", "minLength": 1},
                            "body": {"type": "string", "minLength": 1},
                        },
                    },
                }
            },
        },
    },
}

MEMORY_FACT_JSON_SCHEMA: JsonDict = {
    "type": "json_schema",
    "json_schema": {
        "name": "memory_fact_v1",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["memories"],
            "properties": {
                "memories": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["content", "kind", "confidence"],
                        "properties": {
                            "content": {"type": "string", "minLength": 1},
                            "kind": {"type": "string", "enum": ["fact", "preference", "constraint"]},
                            "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
                        },
                    },
                }
            },
        },
    },
}


def validate_session_memory_summary(data: Any) -> SessionMemorySummary:
    """Validate session-summary JSON and return a normalized object."""

    obj = _require_object(data)
    summary = _required_string(obj, "summary")
    return SessionMemorySummary(summary=summary)


def validate_user_profile_sections(data: Any) -> UserProfileSections:
    """Validate user-profile JSON and return normalized sections."""

    obj = _require_object(data)
    raw_sections = obj.get("sections")
    if not isinstance(raw_sections, list):
        raise MemorySchemaError("sections must be a list")
    sections: list[UserProfileSection] = []
    for raw in raw_sections:
        item = _require_object(raw)
        sections.append(
            UserProfileSection(
                id=_section_id(_required_string(item, "id")),
                title=_required_string(item, "title"),
                body=_required_string(item, "body"),
            )
        )
    return UserProfileSections(tuple(sections))


def validate_memory_fact_candidates(data: Any) -> MemoryFactCandidates:
    """Validate memory_fact_v1 candidate JSON and return normalized candidates."""

    obj = _require_object(data)
    raw_memories = obj.get("memories")
    if not isinstance(raw_memories, list):
        raise MemorySchemaError("memories must be a list")
    memories: list[MemoryFactCandidate] = []
    for raw in raw_memories:
        item = _require_object(raw)
        kind = _required_enum(item, "kind", {"fact", "preference", "constraint"})
        confidence = _required_enum(item, "confidence", {"low", "medium", "high"})
        memories.append(
            MemoryFactCandidate(
                content=_required_string(item, "content"),
                kind=kind,
                confidence=confidence,
            )
        )
    return MemoryFactCandidates(tuple(memories))


def _require_object(data: Any) -> JsonDict:
    if not isinstance(data, dict):
        raise MemorySchemaError("memory output must be a JSON object", error_code="NON_OBJECT_JSON")
    return data


def _required_string(data: JsonDict, key: str) -> str:
    if key not in data:
        raise MemorySchemaError(f"missing required field: {key}")
    value = data.get(key)
    if not isinstance(value, str):
        raise MemorySchemaError(f"{key} must be a string")
    cleaned = re.sub(r"\s+", " ", value).strip()
    if not cleaned:
        raise MemorySchemaError(f"{key} is empty", error_code="EMPTY_CONTENT")
    return cleaned


def _required_enum(data: JsonDict, key: str, allowed: set[str]) -> str:
    value = _required_string(data, key).lower()
    if value not in allowed:
        raise MemorySchemaError(f"{key} must be one of: {', '.join(sorted(allowed))}")
    return value


def _section_id(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.strip().lower())
    return slug.strip("_") or "section"
