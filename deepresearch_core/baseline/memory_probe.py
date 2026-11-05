"""Synthetic structured-output probe for the Memory processing model."""

from __future__ import annotations

import time
from typing import Any

from .adapters import LLMAdapter, LLMResult
from .errors import BaselineError
from .memory_json import MemoryJsonError, diagnostics_from_response, parse_json_object
from .memory_schema import SESSION_SUMMARY_JSON_SCHEMA, validate_session_memory_summary

JsonDict = dict[str, Any]

_PROBE_PROMPT = (
    "Return only JSON for this DeepResearch memory format test. "
    "Use this exact contract: {\"summary\":\"...\"}. "
    "Write a short non-sensitive summary about a synthetic local research session."
)


def run_memory_format_probe(llm: LLMAdapter | None, *, provider_ref: str = "") -> JsonDict:
    """Probe whether a memory model can return safe structured JSON."""

    checked_at = time.time()
    if llm is None:
        return _failed(
            provider_ref=provider_ref,
            checked_at=checked_at,
            capability="none",
            strategy="none",
            error_code="MEMORY_MODEL_NOT_CONFIGURED",
        )

    strict = _try_probe(
        llm,
        response_format=SESSION_SUMMARY_JSON_SCHEMA,
        capability="json_schema_strict",
        strategy="json_schema_strict",
    )
    if strict["ok"]:
        return _passed(provider_ref=provider_ref, checked_at=checked_at, **strict)

    json_object = _try_probe(
        llm,
        response_format={"type": "json_object"},
        capability="json_object",
        strategy="json_object",
    )
    if json_object["ok"]:
        return _caution(provider_ref=provider_ref, checked_at=checked_at, **json_object)

    parser = _try_probe(
        llm,
        response_format=None,
        capability="plain_json_prompt",
        strategy="parser_fallback",
    )
    if parser["ok"]:
        return _caution(provider_ref=provider_ref, checked_at=checked_at, **parser)
    return _failed(
        provider_ref=provider_ref,
        checked_at=checked_at,
        **{
            **parser,
            "capability": "json_schema_strict",
            "structured_strategy": "json_schema_strict",
            "diagnostics": {
                "attempts": [
                    _attempt_summary(strict),
                    _attempt_summary(json_object),
                    _attempt_summary(parser),
                ]
            },
        },
    )


def _try_probe(
    llm: LLMAdapter,
    *,
    response_format: JsonDict | None,
    capability: str,
    strategy: str,
) -> JsonDict:
    try:
        kwargs: JsonDict = {
            "temperature": 0.0,
            "max_tokens": 120,
            "thinking_budget": 0,
        }
        if response_format is not None:
            kwargs["response_format"] = response_format
        result = llm.complete([{"role": "user", "content": _PROBE_PROMPT}], **kwargs)
    except BaselineError as error:
        return {
            "ok": False,
            "unsupported": _response_format_unsupported(error),
            "capability": capability,
            "structured_strategy": strategy,
            "error_code": "STRUCTURED_OUTPUT_UNSUPPORTED" if _response_format_unsupported(error) else "PROBE_REQUEST_FAILED",
            "diagnostics": {"validation_error": error.code.value if hasattr(error.code, "value") else str(error.code)},
        }
    except Exception:
        return {
            "ok": False,
            "unsupported": False,
            "capability": capability,
            "structured_strategy": strategy,
            "error_code": "PROBE_REQUEST_FAILED",
            "diagnostics": {"validation_error": "PROBE_REQUEST_FAILED"},
        }

    try:
        parse_json_object(result.text or "", validator=lambda item: validate_session_memory_summary(item).to_dict())
    except MemoryJsonError as error:
        return {
            "ok": False,
            "unsupported": False,
            "capability": capability,
            "structured_strategy": strategy,
            "error_code": error.error_code,
            "diagnostics": _diagnostics(result, validation_error=error.error_code),
        }
    return {
        "ok": True,
        "unsupported": False,
        "capability": capability,
        "structured_strategy": strategy,
        "error_code": "",
        "diagnostics": _diagnostics(result),
    }


def _passed(*, provider_ref: str, checked_at: float, **result: Any) -> JsonDict:
    return _payload("passed", provider_ref=provider_ref, checked_at=checked_at, **result)


def _caution(*, provider_ref: str, checked_at: float, **result: Any) -> JsonDict:
    return _payload("caution", provider_ref=provider_ref, checked_at=checked_at, **result)


def _failed(*, provider_ref: str, checked_at: float, **result: Any) -> JsonDict:
    return _payload("failed", provider_ref=provider_ref, checked_at=checked_at, **result)


def _payload(status: str, *, provider_ref: str, checked_at: float, **result: Any) -> JsonDict:
    payload: JsonDict = {
        "status": status,
        "provider_ref": provider_ref,
        "capability": str(result.get("capability") or ""),
        "structured_strategy": str(result.get("structured_strategy") or result.get("strategy") or ""),
        "checked_at": checked_at,
        "diagnostics": result.get("diagnostics") if isinstance(result.get("diagnostics"), dict) else {},
    }
    error_code = str(result.get("error_code") or "")
    if error_code:
        payload["error_code"] = error_code
    return payload


def _diagnostics(result: LLMResult, *, validation_error: str = "") -> JsonDict:
    return diagnostics_from_response(
        result.text or "",
        finish_reason=result.finish_reason or "unknown",
        validation_error=validation_error,
    )


def _attempt_summary(result: JsonDict) -> JsonDict:
    summary: JsonDict = {
        "ok": bool(result.get("ok")),
        "structured_strategy": str(result.get("structured_strategy") or ""),
    }
    error_code = str(result.get("error_code") or "")
    if error_code:
        summary["error_code"] = error_code
    diagnostics = result.get("diagnostics") if isinstance(result.get("diagnostics"), dict) else {}
    response_shape = str(diagnostics.get("response_shape") or "")
    if response_shape:
        summary["response_shape"] = response_shape
    return summary

def _response_format_unsupported(error: BaselineError) -> bool:
    text = str(error).lower()
    return "response_format" in text or "json_schema" in text or "json_object" in text or "structured output" in text
