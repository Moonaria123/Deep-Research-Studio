import re
from collections.abc import Iterable
from typing import Any

from .report_sources import extract_report_sources
from .standalone.llm import redact_text


JsonDict = dict[str, Any]
_CONTEXT_SIGNAL_RE = re.compile(r"confidence\s*[\"'：:=\s]+\s*[\"']?(HIGH|MEDIUM|LOW)", re.IGNORECASE)
_BARE_SIGNAL_RE = re.compile(r"\b(HIGH|MEDIUM|LOW)\b|高置信|中置信|低置信")


def assess_report_confidence(markdown: Any, sources: Iterable[JsonDict] | None = None) -> JsonDict:
    """Return a safe confidence summary for a local Markdown report."""

    text = redact_text(str(markdown or ""))
    source_items = list(sources) if sources is not None else extract_report_sources(text)
    source_count = len(source_items)
    signals = _confidence_signals(text)
    level = _confidence_level(signals, source_count)
    return {
        "level": level,
        "label": _confidence_label(level),
        "source_count": source_count,
        "signals": signals,
        "note": _confidence_note(level, source_count, signals),
    }


def assess_quality_confidence(validation: Any, *, source_count: int) -> JsonDict:
    """Return confidence from structured quality-gate output."""

    signals = _quality_signals(validation)
    level = _confidence_level(signals, source_count)
    return {
        "level": level,
        "label": _confidence_label(level),
        "source_count": max(0, int(source_count or 0)),
        "signals": signals,
        "note": _confidence_note(level, max(0, int(source_count or 0)), signals),
    }


def normalize_report_confidence(value: Any, *, sources: Iterable[JsonDict] | None = None) -> JsonDict | None:
    """Return safe confidence metadata from a sidecar object."""

    if not isinstance(value, dict):
        return None
    source_count = len(list(sources)) if sources is not None else _safe_count(value.get("source_count"))
    raw_signals = value.get("signals") if isinstance(value.get("signals"), list) else []
    signals: list[str] = []
    for item in raw_signals:
        signal = _normalize_signal(str(item or ""))
        if signal and signal not in signals:
            signals.append(signal)
    level = str(value.get("level") or "").strip().lower()
    if level not in {"high", "medium", "low"}:
        level = _confidence_level(signals, source_count)
    label = _redact_local_paths(redact_text(str(value.get("label") or _confidence_label(level))))
    note = redact_text(str(value.get("note") or _confidence_note(level, source_count, signals)))
    return {
        "level": level,
        "label": label,
        "source_count": source_count,
        "signals": signals,
        "note": _redact_local_paths(note),
    }


def _confidence_signals(text: str) -> list[str]:
    """Extract explicit HIGH/MEDIUM/LOW confidence markers in first-seen order."""

    matches: list[tuple[int, str]] = []
    for match in _CONTEXT_SIGNAL_RE.finditer(text):
        matches.append((match.start(), match.group(1).upper()))
    for match in _BARE_SIGNAL_RE.finditer(text):
        matches.append((match.start(), _normalize_signal(match.group(1) or match.group(0))))
    signals: list[str] = []
    seen: set[str] = set()
    for _, signal in sorted(matches, key=lambda item: item[0]):
        if signal and signal not in seen:
            signals.append(signal)
            seen.add(signal)
    return signals


def _quality_signals(validation: Any) -> list[str]:
    """Extract confidence markers from quality-gate output."""

    if not isinstance(validation, dict):
        return []
    insights = validation.get("validated_insights")
    if not isinstance(insights, list):
        return []
    signals: list[str] = []
    for item in insights:
        if not isinstance(item, dict):
            continue
        signal = _normalize_signal(str(item.get("confidence") or ""))
        if signal and signal not in signals:
            signals.append(signal)
    return signals


def _normalize_signal(value: str) -> str:
    """Normalize one confidence marker to HIGH, MEDIUM, or LOW."""

    token = value.upper()
    if "高" in value or token == "HIGH":
        return "HIGH"
    if "中" in value or token == "MEDIUM":
        return "MEDIUM"
    if "低" in value or token == "LOW":
        return "LOW"
    return ""


def _confidence_level(signals: list[str], source_count: int) -> str:
    """Choose a conservative confidence level from markers and source count."""

    if "LOW" in signals:
        return "low"
    if "HIGH" in signals:
        return "high"
    if "MEDIUM" in signals or source_count >= 3:
        return "medium"
    return "low"


def _confidence_label(level: str) -> str:
    """Return the user-facing confidence label."""

    return {
        "high": "高置信",
        "medium": "中置信",
        "low": "谨慎参考",
    }.get(level, "谨慎参考")


def _confidence_note(level: str, source_count: int, signals: list[str]) -> str:
    """Return a concise note without copying sensitive report content."""

    marker_note = "包含显式置信标记" if signals else "未发现显式置信标记"
    if level == "high":
        return f"{marker_note}，并提供 {source_count} 个可核验来源；关键事实仍建议打开来源核对。"
    if level == "medium":
        return f"{marker_note}，提供 {source_count} 个可核验来源；适合作为分析起点，重要结论需继续核验。"
    return f"{marker_note}，提供 {source_count} 个可核验来源；请优先核对来源与不确定内容。"


def _safe_count(value: Any) -> int:
    """Return a non-negative integer count from untrusted metadata."""

    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _redact_local_paths(value: str) -> str:
    """Redact local path fragments in confidence notes."""

    text = str(value)
    text = re.sub(r"(?i)file:///[^\s,;)]+", "[local path]", text)
    text = re.sub(r"(?i)\b[A-Z]:\\[^\s,;)]+", "[local path]", text)
    text = re.sub(r"(?i)\b(?:/home|/users|/var|/tmp|/mnt)/[^\s,;)]+", "[local path]", text)
    return " ".join(text.split()).strip()
