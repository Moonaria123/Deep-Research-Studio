"""T19 shared Memory JSON parser contract tests."""

from __future__ import annotations

from deepresearch_core.baseline.memory_json import (
    MemoryJsonError,
    build_repair_prompt,
    parse_json_object,
    response_shape,
)
from deepresearch_core.baseline.memory_schema import validate_session_memory_summary


def test_parse_json_object_accepts_fenced_prose_and_trailing_comma():
    data = parse_json_object('Here:\n```json\n{"summary":"Stable JSON works.",}\n```\nDone.')

    assert data == {"summary": "Stable JSON works."}


def test_parse_json_object_rejects_root_array():
    try:
        parse_json_object('[{"summary":"wrong root"}]')
    except MemoryJsonError as error:
        assert error.error_code == "NON_OBJECT_JSON"
        assert error.diagnostics["response_shape"] == "json_array"
    else:  # pragma: no cover - defensive
        raise AssertionError("expected MemoryJsonError")


def test_parse_json_object_reports_schema_validation_without_raw_output():
    try:
        parse_json_object('{"wrong":"sk-secret should not leak"}', validator=validate_session_memory_summary)
    except MemoryJsonError as error:
        assert error.error_code == "SCHEMA_VALIDATION_FAILED"
        assert "sk-secret" not in str(error.diagnostics)
    else:  # pragma: no cover - defensive
        raise AssertionError("expected MemoryJsonError")


def test_build_repair_prompt_omits_original_output():
    prompt = build_repair_prompt(
        stage="session_distill",
        error_code="SCHEMA_VALIDATION_FAILED",
        schema={"type": "json_schema", "json_schema": {"name": "session"}},
    )

    assert "SCHEMA_VALIDATION_FAILED" in prompt
    assert "raw previous response" in prompt
    assert "sk-" not in prompt


def test_response_shape_classifies_plain_and_json_shapes():
    assert response_shape("") == "empty"
    assert response_shape('{"summary":"x"}') == "json_object"
    assert response_shape('[{"summary":"x"}]') == "json_array"
    assert response_shape("```json\n{}\n```") == "fenced_json"
    assert response_shape("answer {\"summary\":\"x\"}") == "prose_wrapped_json"
