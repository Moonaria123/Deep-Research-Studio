"""T18 Memory format probe contract tests."""

from __future__ import annotations

from deepresearch_core.baseline.adapters import LLMResult
from deepresearch_core.baseline.errors import BaselineError, ErrorCode
from deepresearch_core.baseline.memory_probe import run_memory_format_probe
from tests.baseline.fakes import FakeLLM


class _ProbeLLM(FakeLLM):
    def __init__(self, *, unsupported_formats: set[str] | None = None, replies: list[str] | None = None) -> None:
        super().__init__()
        self.unsupported_formats = unsupported_formats or set()
        self.replies = replies or ['{"summary":"Synthetic memory probe passed."}']
        self.kwargs: list[dict] = []

    def complete(self, messages, **kwargs):  # type: ignore[override]
        self.calls.append(list(messages))
        self.kwargs.append(kwargs)
        response_format = kwargs.get("response_format")
        format_type = response_format.get("type") if isinstance(response_format, dict) else "none"
        if format_type in self.unsupported_formats:
            raise BaselineError(ErrorCode.UPSTREAM_ERROR, f"{format_type} response_format unsupported")
        index = min(len(self.calls) - 1, len(self.replies) - 1)
        return LLMResult(text=self.replies[index], finish_reason="stop", model=self.model)


def test_memory_probe_passes_with_strict_schema():
    llm = _ProbeLLM()

    result = run_memory_format_probe(llm, provider_ref="provider:model")

    assert result["status"] == "passed"
    assert result["provider_ref"] == "provider:model"
    assert result["capability"] == "json_schema_strict"
    assert result["structured_strategy"] == "json_schema_strict"
    assert llm.kwargs[0]["response_format"]["type"] == "json_schema"


def test_memory_probe_degrades_to_json_object_caution():
    llm = _ProbeLLM(unsupported_formats={"json_schema"})

    result = run_memory_format_probe(llm)

    assert result["status"] == "caution"
    assert result["capability"] == "json_object"
    assert llm.kwargs[1]["response_format"] == {"type": "json_object"}


def test_memory_probe_degrades_to_parser_fallback_caution():
    llm = _ProbeLLM(unsupported_formats={"json_schema", "json_object"})

    result = run_memory_format_probe(llm)

    assert result["status"] == "caution"
    assert result["structured_strategy"] == "parser_fallback"
    assert "response_format" not in llm.kwargs[-1]


def test_memory_probe_failed_result_is_secret_free():
    llm = _ProbeLLM(replies=["not json", "also not json", "sk-secret raw output should not surface"])

    result = run_memory_format_probe(llm)

    assert result["status"] == "failed"
    assert result["error_code"]
    assert "raw output" not in str(result)
    assert "sk-secret" not in str(result)
    assert "attempts" in result["diagnostics"]


def test_memory_probe_reports_missing_model():
    result = run_memory_format_probe(None)

    assert result["status"] == "failed"
    assert result["error_code"] == "MEMORY_MODEL_NOT_CONFIGURED"
