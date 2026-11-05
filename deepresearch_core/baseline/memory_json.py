"""Shared JSON parsing and diagnostics for Memory model outputs."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any

JsonDict = dict[str, Any]


class MemoryJsonError(ValueError):
    """Raised when a Memory model response cannot be parsed or validated safely."""

    def __init__(
        self,
        message: str,
        *,
        error_code: str = "JSON_PARSE_FAILED",
        diagnostics: JsonDict | None = None,
    ) -> None:
        super().__init__(message)
        self.error_code = error_code
        self.diagnostics = diagnostics or {}


def parse_json_object(text: str, *, validator: Callable[[JsonDict], Any] | None = None) -> JsonDict:
    """Extract one JSON object from LLM text and optionally validate it."""

    raw = str(text or "").strip()
    candidates = [raw]
    candidates.extend(match.group(1).strip() for match in re.finditer(r"```(?:json)?\s*(.*?)```", raw, re.IGNORECASE | re.DOTALL))
    balanced = first_balanced_json_object(raw)
    if balanced:
        candidates.append(balanced)

    seen: set[str] = set()
    last_error: json.JSONDecodeError | None = None
    for candidate in candidates:
        for item in (candidate, repair_json(candidate)):
            item = item.strip()
            if not item or item in seen:
                continue
            seen.add(item)
            try:
                data = json.loads(item)
            except json.JSONDecodeError as error:
                last_error = error
                continue
            if not isinstance(data, dict):
                raise MemoryJsonError(
                    "memory output must be a JSON object",
                    error_code="NON_OBJECT_JSON",
                    diagnostics=diagnostics_from_response(text, validation_error="NON_OBJECT_JSON"),
                )
            if validator is None:
                return data
            try:
                validated = validator(data)
            except Exception as error:  # noqa: BLE001 - normalize schema validators
                code = str(getattr(error, "error_code", "SCHEMA_VALIDATION_FAILED") or "SCHEMA_VALIDATION_FAILED")
                raise MemoryJsonError(
                    "memory output failed schema validation",
                    error_code=code,
                    diagnostics=diagnostics_from_response(text, validation_error=code),
                ) from error
            if hasattr(validated, "to_dict"):
                return validated.to_dict()
            if isinstance(validated, dict):
                return validated
            return data
    raise MemoryJsonError(
        "memory output is not valid JSON",
        error_code="JSON_PARSE_FAILED",
        diagnostics=diagnostics_from_response(text, validation_error="JSON_PARSE_FAILED"),
    ) from last_error


def build_repair_prompt(*, stage: str, error_code: str, schema: JsonDict | None) -> str:
    """Build a repair prompt that carries schema/error class but no raw previous output."""

    if stage == "user_profile":
        contract = '{"sections":[{"id":"preferences","title":"Research Preferences","body":"..."}]}'
    elif stage == "memory_fact":
        contract = '{"memories":[{"content":"...","kind":"fact","confidence":"medium"}]}'
    elif stage == "wiki_map":
        contract = (
            '{"topics":[{"title":"...","summary":"...","concept_key":"normalized-concept",'
            '"confidence":"high","evidence_refs":[{"type":"message","id":"msg_...","excerpt":"..."}],'
            '"source_message_ids":["msg_..."],"children":[]}]}'
        )
    else:
        contract = '{"summary":"..."}'
    schema_note = json.dumps(schema or {"type": "json_object"}, ensure_ascii=False)
    return (
        "Your previous response did not match the required JSON object contract. "
        f"Error class: {error_code}. Return only one JSON object matching this example: {contract}. "
        "Do not include markdown, comments, explanations, or any raw previous response. "
        f"Schema hint: {schema_note}"
    )


def attempt_diagnostics(text: str, finish_reason: str, validation_error: str, structured_strategy: str) -> JsonDict:
    """Return one safe attempt summary without raw model text."""

    return {
        "response_shape": response_shape(text),
        "finish_reason": finish_reason or "unknown",
        "validation_error": validation_error,
        "response_empty": not bool(str(text or "").strip()),
        "structured_strategy": structured_strategy or "unknown",
    }


def diagnostics_from_attempts(attempts: list[JsonDict]) -> JsonDict:
    """Collapse attempt diagnostics into the public Memory diagnostic shape."""

    last = attempts[-1] if attempts else {}
    return {
        "response_shape": last.get("response_shape") or "unknown",
        "finish_reason": last.get("finish_reason") or "unknown",
        "validation_error": last.get("validation_error") or "UNKNOWN",
        "response_empty": bool(last.get("response_empty")),
        "attempts": len(attempts),
        "structured_strategy": last.get("structured_strategy") or "unknown",
    }


def diagnostics_from_response(text: str, *, finish_reason: str = "unknown", validation_error: str = "") -> JsonDict:
    """Build safe diagnostics for one response."""

    return {
        "response_shape": response_shape(text),
        "finish_reason": finish_reason or "unknown",
        "validation_error": validation_error or None,
        "attempts": 1,
        "response_empty": not bool(str(text or "").strip()),
    }


def first_balanced_json_object(text: str) -> str:
    """Return the first balanced JSON object substring, or an empty string."""

    start = str(text or "").find("{")
    if start < 0:
        return ""
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return ""


def repair_json(text: str) -> str:
    """Apply tiny non-semantic JSON repairs before parsing."""

    cleaned = str(text or "").strip().lstrip("\ufeff")
    return re.sub(r",\s*([}\]])", r"\1", cleaned)


def response_shape(text: str) -> str:
    """Classify response shape without exposing response content."""

    raw = str(text or "").strip()
    if not raw:
        return "empty"
    if raw.startswith("{"):
        return "json_object"
    if raw.startswith("["):
        return "json_array"
    if raw.startswith("```"):
        return "fenced_json"
    if "{" in raw and "}" in raw:
        return "prose_wrapped_json"
    return "prose"
