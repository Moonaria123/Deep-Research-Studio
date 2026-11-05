from typing import Any, Iterable

from deepresearch_core.report_sources import sanitize_report_sources
from deepresearch_core.standalone.llm import redact_text


SOURCE_QUALITY_SCHEMA_VERSION = 1
SOURCE_QUALITY_CHECK_LIMIT = 6


def normalize_source_quality(
    value: Any = None,
    *,
    sources: Iterable[dict[str, Any]] | None = None,
    confidence: Any = None,
    provenance: Any = None,
    answer_evidence: Any = None,
) -> dict[str, Any]:
    """Return a compact source-quality summary for end users."""

    raw = value if isinstance(value, dict) else {}
    safe_sources = sanitize_report_sources(list(sources or []))
    confidence_data = confidence if isinstance(confidence, dict) else {}
    provenance_data = provenance if isinstance(provenance, dict) else {}
    answer_data = answer_evidence if isinstance(answer_evidence, dict) else {}

    source_count = len(safe_sources)
    source_relevance = provenance_data.get("source_relevance") if isinstance(provenance_data.get("source_relevance"), dict) else {}
    conflict_gate = provenance_data.get("conflict_gate") if isinstance(provenance_data.get("conflict_gate"), dict) else {}
    evidence_repair = provenance_data.get("evidence_repair") if isinstance(provenance_data.get("evidence_repair"), dict) else {}
    review_actions = provenance_data.get("review_actions") if isinstance(provenance_data.get("review_actions"), dict) else {}

    relevant_source_count = _safe_count(source_relevance.get("relevant_source_count"))
    if not relevant_source_count:
        relevant_source_count = sum(1 for item in _safe_items(source_relevance) if item.get("level") == "relevant")
    missing_claim_count = _safe_count(provenance_data.get("missing_claim_count"))
    conflict_count = _risk_count(conflict_gate) if str(conflict_gate.get("level") or "") == "conflict" else 0
    repair_count = max(_risk_count(evidence_repair), _risk_count(review_actions))
    source_relevance_level = str(source_relevance.get("level") or "")
    low_relevance_count = (
        0
        if source_relevance_level == "unknown"
        else sum(1 for item in _safe_items(source_relevance) if item.get("level") in {"weak", "unrelated"})
    )
    ranked_source_count = len(answer_data.get("ranked_sources") or []) if isinstance(answer_data.get("ranked_sources"), list) else 0
    confidence_level = _choice(confidence_data.get("level"), {"high", "medium", "low"}, raw.get("confidence_level") or "low")

    level, label, next_step, score, summary = _source_quality_status(
        source_count=source_count,
        relevant_source_count=relevant_source_count,
        missing_claim_count=missing_claim_count,
        conflict_count=conflict_count,
        repair_count=repair_count,
        low_relevance_count=low_relevance_count,
        confidence_level=confidence_level,
        answer_level=str(answer_data.get("level") or ""),
        source_relevance_level=source_relevance_level,
    )

    return {
        "schema_version": SOURCE_QUALITY_SCHEMA_VERSION,
        "level": _choice(raw.get("level"), {"ready", "usable", "review", "blocked", "missing"}, level),
        "label": _compact(raw.get("label"), limit=40) or label,
        "score": _bounded_score(raw.get("score") if raw else score) if raw.get("score") is not None else score,
        "source_count": source_count,
        "relevant_source_count": relevant_source_count,
        "ranked_source_count": ranked_source_count,
        "missing_claim_count": missing_claim_count,
        "conflict_count": conflict_count,
        "repair_count": repair_count,
        "low_relevance_count": low_relevance_count,
        "confidence_level": confidence_level,
        "next_step": _compact(raw.get("next_step"), limit=80) or next_step,
        "summary": _compact(raw.get("summary"), limit=220) or summary,
        "note": _compact(raw.get("note"), limit=220)
        or "该摘要合成来源数量、相关性、引用覆盖和冲突/复核信号，不代表事实真伪。",
        "checks": _source_quality_checks(
            source_count=source_count,
            relevant_source_count=relevant_source_count,
            missing_claim_count=missing_claim_count,
            conflict_count=conflict_count,
            repair_count=repair_count,
            low_relevance_count=low_relevance_count,
        ),
    }


def _source_quality_status(
    *,
    source_count: int,
    relevant_source_count: int,
    missing_claim_count: int,
    conflict_count: int,
    repair_count: int,
    low_relevance_count: int,
    confidence_level: str,
    answer_level: str,
    source_relevance_level: str,
) -> tuple[str, str, str, int, str]:
    """Return public quality status fields from source and provenance signals."""

    if conflict_count > 0 or answer_level == "blocked":
        return (
            "blocked",
            "先处理冲突",
            "resolve_conflicts_before_using",
            15,
            f"发现 {max(1, conflict_count)} 个冲突信号；请先打开来源核对，再决定是否使用回答。",
        )
    if source_count <= 0:
        return (
            "missing",
            "缺少来源",
            "add_sources_or_search",
            0,
            "没有可展示 HTTP(S) 来源；请先补充来源或重新搜索。",
        )

    score = 40 + min(source_count, 4) * 8 + min(relevant_source_count, 4) * 10
    if answer_level == "ready":
        score += 10
    if confidence_level == "high":
        score += 6
    elif confidence_level == "medium":
        score += 2
    else:
        score -= 8
    if source_relevance_level in {"missing", "unrelated"}:
        score -= 25
    elif source_relevance_level == "weak":
        score -= 10
    score -= missing_claim_count * 12
    score -= repair_count * 8
    score -= low_relevance_count * 4
    score = _bounded_score(score)

    if missing_claim_count or repair_count or low_relevance_count:
        return (
            "review",
            "建议先复核来源",
            "repair_evidence",
            score,
            f"有 {max(missing_claim_count, repair_count, low_relevance_count)} 个来源或引用信号建议先复核。",
        )
    if score >= 80 and source_count >= 2 and relevant_source_count >= 1:
        return (
            "ready",
            "来源质量较稳",
            "read_with_sources",
            score,
            f"{source_count} 个来源中有 {relevant_source_count} 个与问题直接相关；仍建议打开关键来源核对。",
        )
    return (
        "usable",
        "可带来源使用",
        "read_with_sources",
        score,
        f"当前有 {source_count} 个可展示来源；适合先读答案，再打开来源核对关键结论。",
    )


def _source_quality_checks(
    *,
    source_count: int,
    relevant_source_count: int,
    missing_claim_count: int,
    conflict_count: int,
    repair_count: int,
    low_relevance_count: int,
) -> list[dict[str, Any]]:
    """Return a short source-quality checklist."""

    checks = [
        _check("source_count", "来源数量", "ok" if source_count else "missing", source_count),
        _check("source_relevance", "来源相关性", "ok" if relevant_source_count else "review", relevant_source_count),
        _check("citation_coverage", "引用覆盖", "ok" if not missing_claim_count else "review", missing_claim_count),
        _check("conflict_gate", "冲突信号", "ok" if not conflict_count else "blocked", conflict_count),
        _check("repair_signals", "复核建议", "ok" if not repair_count else "review", repair_count),
        _check("low_relevance", "低相关来源", "ok" if not low_relevance_count else "review", low_relevance_count),
    ]
    return checks[:SOURCE_QUALITY_CHECK_LIMIT]


def _check(kind: str, label: str, status: str, count: int) -> dict[str, Any]:
    """Return one public quality check."""

    return {"kind": kind, "label": label, "status": status, "count": max(0, int(count or 0))}


def _safe_items(value: dict[str, Any]) -> list[dict[str, Any]]:
    """Return dict items from a public metadata block."""

    items = value.get("items")
    if not isinstance(items, list):
        return []
    return [item for item in items if isinstance(item, dict)]


def _risk_count(value: dict[str, Any]) -> int:
    """Return a public risk count from metadata items or claim ids."""

    items = _safe_items(value)
    if items:
        return len(items)
    claim_ids = value.get("claim_ids")
    if isinstance(claim_ids, list):
        return len(claim_ids)
    return 0


def _choice(value: Any, allowed: set[str], fallback: Any) -> str:
    """Return an allowed compact string."""

    text = _compact(value, limit=80)
    return text if text in allowed else _compact(fallback, limit=80)


def _safe_count(value: Any) -> int:
    """Return a non-negative integer from untrusted metadata."""

    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _bounded_score(value: Any) -> int:
    """Return a bounded 0-100 quality score."""

    try:
        score = int(round(float(value)))
    except (TypeError, ValueError):
        score = 0
    return max(0, min(100, score))


def _compact(value: Any, *, limit: int = 220) -> str:
    """Return redacted single-line public text."""

    text = redact_text(str(value or ""))
    text = " ".join(text.split()).strip()
    return text[: max(0, int(limit))]
