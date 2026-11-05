import json
import re
from typing import Any

from deepresearch_core.conflict_resolution import CONFLICT_PROVIDER_VERIFICATION_MODE
from deepresearch_core.conflict_resolution import conflict_provider_verification_skipped
from deepresearch_core.conflict_resolution import normalize_conflict_resolution
from deepresearch_core.standalone.llm import parse_json_object, redact_text, user_facing_provider_error


JsonDict = dict[str, Any]
MAX_CONFLICT_SOURCES = 6
EVIDENCE_TEXT_LIMIT = 700
NOTE_LIMIT = 180
_LOCAL_PATH_RE = re.compile(
    r"(?i)file:///[^\r\n,;)]*|"
    r"(?<![A-Za-z0-9_])[A-Za-z]:[\\/][^\r\n,;)]*|"
    r"(?<![A-Za-z0-9_])(?:/home|/users|/var|/tmp|/mnt|/workspace|/root|/etc|/opt|/private)/[^\r\n,;)]*",
)


def verify_conflict_resolution_with_provider(
    llm_client: Any,
    conflict_resolution: Any,
    *,
    source_documents: list[JsonDict],
    query: Any = "",
    enabled: bool,
) -> JsonDict:
    """Use the configured BYOK LLM to adjudicate visible conflict sources when enabled."""

    if not enabled:
        return conflict_provider_verification_skipped()
    resolution = normalize_conflict_resolution(conflict_resolution)
    if not resolution:
        return _no_input(llm_client, "未找到可交给模型核验的冲突补证结论。")
    prompt_sources = _prompt_sources(resolution, source_documents)
    if not prompt_sources:
        return _no_input(llm_client, "未找到可交给模型核验的公开冲突来源摘录。")

    prompt_payload = {
        "task": CONFLICT_PROVIDER_VERIFICATION_MODE,
        "allowed_statuses": ["resolved", "still_conflicting", "unresolved"],
        "allowed_verdicts": ["positive", "negative", "conflict", "unknown"],
        "query": _safe_text(query, 220),
        "local_resolution": {
            "status": resolution.get("status", "unresolved"),
            "verdict": resolution.get("verdict", "unknown"),
            "summary": resolution.get("summary", ""),
        },
        "sources": prompt_sources,
    }
    messages = [
        {
            "role": "system",
            "content": (
                "You adjudicate visible source conflicts for a local deep research tool. "
                "Use only the provided excerpts. Return JSON only. Do not include chain-of-thought, secrets, or source full text."
            ),
        },
        {
            "role": "user",
            "content": (
                "Return JSON in this shape: "
                '{"status":"resolved|still_conflicting|unresolved",'
                '"verdict":"positive|negative|conflict|unknown",'
                '"confidence":0-100,"note":"short user-facing note"}.\n\n'
                f"Input JSON:\n{json.dumps(prompt_payload, ensure_ascii=False)}"
            ),
        },
    ]
    try:
        data = parse_json_object(llm_client.chat(messages, max_tokens=700, temperature=0.0), {})
    except Exception as error:
        return {
            **conflict_provider_verification_skipped(),
            "enabled": True,
            "status": "failed",
            "model": _safe_text(getattr(llm_client, "model", ""), 120),
            "note": _safe_text(f"模型冲突核验失败：{user_facing_provider_error(error)}", NOTE_LIMIT),
        }
    return {
        "enabled": True,
        "status": "completed",
        "mode": CONFLICT_PROVIDER_VERIFICATION_MODE,
        "model": _safe_text(getattr(llm_client, "model", ""), 120),
        "resolution_status": _choice(data.get("status"), {"resolved", "still_conflicting", "unresolved"}, "unresolved"),
        "verdict": _choice(data.get("verdict"), {"positive", "negative", "conflict", "unknown"}, "unknown"),
        "confidence": _score(data.get("confidence")),
        "note": _safe_text(data.get("note") or "模型已基于可见冲突来源摘录完成核验；结果仍需人工打开来源复核。", NOTE_LIMIT),
    }


def _no_input(llm_client: Any, note: str) -> JsonDict:
    """Return enabled no-input status without calling the provider."""

    return {
        **conflict_provider_verification_skipped(),
        "enabled": True,
        "status": "no_input",
        "model": _safe_text(getattr(llm_client, "model", ""), 120),
        "note": note,
    }


def _prompt_sources(resolution: JsonDict, source_documents: list[JsonDict]) -> list[JsonDict]:
    """Return bounded source excerpts for conflict adjudication."""

    prompt_items: list[JsonDict] = []
    for item in resolution.get("items") or []:
        if not isinstance(item, dict) or len(prompt_items) >= MAX_CONFLICT_SOURCES:
            continue
        source_index = _source_index(item.get("source_index") if "source_index" in item else item.get("sourceIndex"))
        source = source_documents[source_index - 1] if 1 <= source_index <= len(source_documents) else {}
        evidence = _safe_text(
            source.get("description") or source.get("snippet") or source.get("markdown") or source.get("text") or item.get("reason"),
            EVIDENCE_TEXT_LIMIT,
        )
        if not evidence:
            continue
        prompt_items.append(
            {
                "source_index": source_index,
                "title": _safe_text(source.get("title") or item.get("title") or source.get("url") or item.get("url"), 180),
                "url": _safe_text(source.get("url") or item.get("url"), 300),
                "local_polarity": _choice(item.get("polarity"), {"positive", "negative"}, "positive"),
                "evidence": evidence,
            }
        )
    return prompt_items


def _source_index(value: Any) -> int:
    """Return a positive source index."""

    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _choice(value: Any, choices: set[str], default: str) -> str:
    """Return a public enum value."""

    text = str(value or "").strip().lower()
    return text if text in choices else default


def _score(value: Any) -> int:
    """Return a bounded provider confidence score."""

    try:
        return max(0, min(100, int(round(float(value)))))
    except (TypeError, ValueError):
        return 0


def _safe_text(value: Any, limit: int) -> str:
    """Return compact public text without secrets or local paths."""

    text = _LOCAL_PATH_RE.sub("[local path]", redact_text(str(value or "")))
    text = " ".join(text.split()).strip()
    return text[: max(0, int(limit))]
