import re
from typing import Any

from deepresearch_core.report_sources import sanitize_report_sources
from deepresearch_core.standalone.llm import redact_text


CONFLICT_RESOLUTION_SCHEMA_VERSION = 1
CONFLICT_PROVIDER_VERIFICATION_MODE = "provider_conflict_adjudication"

_NEGATIVE_STATUS_PATTERNS = (
    r"\bnot\s+(?:yet\s+)?(?:released|available|launched|announced)\b",
    r"\bno\s+(?:official\s+)?(?:release|announcement|product)\b",
    r"\bdoes\s+not\s+exist\b",
    r"\bhas\s+not\s+been\s+(?:released|announced|launched)\b",
    r"未\s*(?:发布|推出|上线|开放|公布|宣布)",
    r"尚未\s*(?:发布|推出|上线|开放|公布|宣布)",
    r"没有\s*(?:发布|推出|上线|开放|公布|宣布)",
    r"不存在",
    r"无官方",
)

_POSITIVE_STATUS_PATTERNS = (
    r"\b(?:released|available|launched|announced)\b",
    r"\bofficial(?:ly)?\s+(?:released|available|launched|announced)\b",
    r"已经\s*(?:发布|推出|上线|开放|公布|宣布)",
    r"已\s*(?:发布|推出|上线|开放|公布|宣布)",
    r"正式\s*(?:发布|推出|上线|开放|公布|宣布)",
    r"可用",
)


def build_conflict_resolution(request: Any, sources: Any = None, *, query: Any = "") -> dict[str, Any]:
    """Return public conflict-resolution metadata for confirmed repair runs."""

    if not _is_conflict_repair_request(request):
        return {}
    safe_sources = sanitize_report_sources(sources, limit=12)
    polarity_items = _source_polarity_items(sources)
    positive_count = sum(1 for item in polarity_items if item["polarity"] == "positive")
    negative_count = sum(1 for item in polarity_items if item["polarity"] == "negative")
    if positive_count and negative_count:
        status = "still_conflicting"
        verdict = "conflict"
        label = "仍冲突"
        summary = "补证后来源仍存在相反表述，不能给出确定结论。"
    elif positive_count or negative_count:
        status = "resolved"
        verdict = "positive" if positive_count else "negative"
        label = "已确认"
        summary = "补证后来源形成单向判断，未在本轮可展示来源中发现相反表述。"
    else:
        status = "unresolved"
        verdict = "unknown"
        label = "无法确认"
        summary = "补证后仍未找到足以判断发布、可用性或官方状态的可展示来源。"
    return normalize_conflict_resolution(
        {
            "schema_version": CONFLICT_RESOLUTION_SCHEMA_VERSION,
            "status": status,
            "label": label,
            "verdict": verdict,
            "source_count": len(safe_sources),
            "positive_source_count": positive_count,
            "negative_source_count": negative_count,
            "summary": summary,
            "note": "该裁决只基于本轮可展示来源的明确表述；重要结论仍建议打开来源核对。",
            "items": polarity_items[:6],
            "provider_verification": conflict_provider_verification_skipped(),
        }
    )


def normalize_conflict_resolution(value: Any) -> dict[str, Any]:
    """Return a safe public conflict-resolution envelope."""

    if not isinstance(value, dict) or not value:
        return {}
    status = _choice(value.get("status"), {"resolved", "still_conflicting", "unresolved", "skipped"}, "unresolved")
    verdict = _choice(value.get("verdict"), {"positive", "negative", "conflict", "unknown"}, "unknown")
    label = _compact(value.get("label"), limit=48) or {
        "resolved": "已确认",
        "still_conflicting": "仍冲突",
        "unresolved": "无法确认",
        "skipped": "未执行",
    }[status]
    items = [_normalize_item(item) for item in value.get("items") or [] if isinstance(item, dict)]
    return {
        "schema_version": CONFLICT_RESOLUTION_SCHEMA_VERSION,
        "status": status,
        "label": label,
        "verdict": verdict,
        "source_count": _bounded_int(value.get("source_count") if "source_count" in value else value.get("sourceCount")),
        "positive_source_count": _bounded_int(value.get("positive_source_count") if "positive_source_count" in value else value.get("positiveSourceCount")),
        "negative_source_count": _bounded_int(value.get("negative_source_count") if "negative_source_count" in value else value.get("negativeSourceCount")),
        "summary": _compact(value.get("summary"), limit=160) or "补证结果需要人工复核。",
        "note": _compact(value.get("note"), limit=180) or "该裁决只基于本轮可展示来源；重要结论仍建议打开来源核对。",
        "items": items[:6],
        "provider_verification": normalize_conflict_provider_verification(
            value.get("provider_verification") if "provider_verification" in value else value.get("providerVerification")
        ),
    }


def conflict_provider_verification_skipped() -> dict[str, Any]:
    """Return the default-offline provider conflict verification object."""

    return {
        "enabled": False,
        "status": "skipped",
        "mode": CONFLICT_PROVIDER_VERIFICATION_MODE,
        "model": "",
        "resolution_status": "skipped",
        "verdict": "unknown",
        "confidence": 0,
        "note": "默认离线；设置 DEEPRESEARCH_PROVIDER_SEMANTIC_VERIFICATION=1 后才会使用本机 BYOK 模型核验冲突来源。",
    }


def normalize_conflict_provider_verification(value: Any) -> dict[str, Any]:
    """Return a safe public provider conflict-verification object."""

    if not isinstance(value, dict):
        return conflict_provider_verification_skipped()
    status = _choice(value.get("status"), {"skipped", "completed", "failed", "no_input"}, "skipped")
    return {
        "enabled": bool(value.get("enabled")),
        "status": status,
        "mode": _compact(value.get("mode"), limit=64) or CONFLICT_PROVIDER_VERIFICATION_MODE,
        "model": _compact(value.get("model"), limit=120),
        "resolution_status": _choice(
            value.get("resolution_status") if "resolution_status" in value else value.get("resolutionStatus"),
            {"resolved", "still_conflicting", "unresolved", "skipped"},
            "skipped" if status == "skipped" else "unresolved",
        ),
        "verdict": _choice(value.get("verdict"), {"positive", "negative", "conflict", "unknown"}, "unknown"),
        "confidence": _bounded_score(value.get("confidence")),
        "note": _compact(value.get("note"), limit=180) or "模型核验结果仍需人工打开来源复核。",
    }


def _is_conflict_repair_request(value: Any) -> bool:
    """Return true when the confirmed repair request targets source conflicts."""

    if not isinstance(value, dict) or not value:
        return False
    fields = " ".join(
        str(value.get(key) or "")
        for key in ("kind", "claim_id", "claimId", "action", "label", "prompt", "started_from", "startedFrom")
    )
    lowered = fields.lower()
    return "source_conflict" in lowered or "conflict" in lowered or "冲突" in fields or "核对冲突来源" in fields


def _source_polarity_items(sources: Any) -> list[dict[str, Any]]:
    """Return source status polarity items from explicit source text."""

    items: list[dict[str, Any]] = []
    for index, source in enumerate(sources if isinstance(sources, list) else [], start=1):
        if not isinstance(source, dict):
            continue
        text = _source_text(source)
        polarity = _status_polarity(text)
        if not polarity:
            continue
        items.append(
            {
                "source_index": index,
                "title": _compact(source.get("title") or source.get("url"), limit=96),
                "url": _compact(source.get("url"), limit=240),
                "polarity": polarity,
                "reason": "来源明确表示已发布/可用" if polarity == "positive" else "来源明确表示未发布/不可用",
            }
        )
    return items


def _status_polarity(text: str) -> str:
    """Return positive or negative for explicit release/availability text."""

    if not text:
        return ""
    has_negative = any(re.search(pattern, text, re.IGNORECASE) for pattern in _NEGATIVE_STATUS_PATTERNS)
    if has_negative:
        return "negative"
    has_positive = any(re.search(pattern, text, re.IGNORECASE) for pattern in _POSITIVE_STATUS_PATTERNS)
    return "positive" if has_positive else ""


def _source_text(source: dict[str, Any]) -> str:
    """Return compact searchable source text."""

    parts = [
        source.get("title"),
        source.get("description"),
        source.get("snippet"),
        source.get("markdown"),
        source.get("text"),
    ]
    return " ".join(str(part or "") for part in parts)


def _normalize_item(value: dict[str, Any]) -> dict[str, Any]:
    """Return one safe public resolution item."""

    return {
        "source_index": _bounded_int(value.get("source_index") if "source_index" in value else value.get("sourceIndex")),
        "title": _compact(value.get("title"), limit=96),
        "url": _compact(value.get("url"), limit=240),
        "polarity": _choice(value.get("polarity"), {"positive", "negative"}, "positive"),
        "reason": _compact(value.get("reason"), limit=120),
    }


def _choice(value: Any, choices: set[str], default: str) -> str:
    """Return a public enum value."""

    text = str(value or "").strip()
    return text if text in choices else default


def _bounded_int(value: Any) -> int:
    """Return a small non-negative integer."""

    try:
        return max(0, min(999, int(value)))
    except (TypeError, ValueError):
        return 0


def _bounded_score(value: Any) -> int:
    """Return a 0-100 integer score."""

    try:
        return max(0, min(100, int(round(float(value)))))
    except (TypeError, ValueError):
        return 0


def _compact(value: Any, *, limit: int) -> str:
    """Return compact redacted public text."""

    text = " ".join(redact_text(str(value or "")).split())
    return text[:limit]
