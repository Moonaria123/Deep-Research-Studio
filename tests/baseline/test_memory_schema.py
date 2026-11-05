"""T18 Memory structured-output schema gate tests."""

from __future__ import annotations

import pytest

from deepresearch_core.baseline.memory_schema import (
    MemorySchemaError,
    validate_memory_fact_candidates,
    validate_session_memory_summary,
    validate_user_profile_sections,
)


def test_session_memory_summary_accepts_non_empty_object():
    result = validate_session_memory_summary({"summary": " Useful summary. "})

    assert result.to_dict() == {"summary": "Useful summary."}


def test_session_memory_summary_rejects_root_array():
    with pytest.raises(MemorySchemaError) as exc:
        validate_session_memory_summary([{"summary": "bad"}])

    assert exc.value.error_code == "NON_OBJECT_JSON"


def test_session_memory_summary_rejects_missing_required_field():
    with pytest.raises(MemorySchemaError):
        validate_session_memory_summary({"wrong": "shape"})


def test_session_memory_summary_rejects_empty_content():
    with pytest.raises(MemorySchemaError) as exc:
        validate_session_memory_summary({"summary": "   "})

    assert exc.value.error_code == "EMPTY_CONTENT"


def test_user_profile_sections_require_section_contract():
    result = validate_user_profile_sections(
        {"sections": [{"id": "Research Preferences", "title": "Prefs", "body": "Concise answers."}]}
    )

    assert result.to_dict() == {
        "sections": [{"id": "research_preferences", "title": "Prefs", "body": "Concise answers."}]
    }


def test_user_profile_sections_reject_missing_body():
    with pytest.raises(MemorySchemaError):
        validate_user_profile_sections({"sections": [{"id": "prefs", "title": "Prefs"}]})


def test_memory_fact_candidates_accept_add_only_contract():
    result = validate_memory_fact_candidates(
        {"memories": [{"content": "Prefers local-first memory.", "kind": "preference", "confidence": "high"}]}
    )

    assert result.to_dict() == {
        "memories": [{"content": "Prefers local-first memory.", "kind": "preference", "confidence": "high"}]
    }


def test_memory_fact_candidates_reject_invalid_enum_and_empty_content():
    with pytest.raises(MemorySchemaError):
        validate_memory_fact_candidates({"memories": [{"content": "", "kind": "fact", "confidence": "medium"}]})
    with pytest.raises(MemorySchemaError):
        validate_memory_fact_candidates(
            {"memories": [{"content": "x", "kind": "delete", "confidence": "medium"}]}
        )
