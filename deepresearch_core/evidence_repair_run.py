from typing import Any

from deepresearch_core.report_sources import sanitize_report_sources
from deepresearch_core.standalone.llm import redact_text


EVIDENCE_REPAIR_RUN_SCHEMA_VERSION = 1
EVIDENCE_REPAIR_PROMPT_LIMIT = 260


def build_evidence_repair_run(request: Any, sources: Any = None, *, query: Any = "") -> dict[str, Any]:
    """Return public metadata for a user-confirmed evidence repair run."""

    if not isinstance(request, dict) or not request:
        return {}
    safe_sources = sanitize_report_sources(sources, limit=12)
    status = "completed" if safe_sources else "no_sources"
    label = "已补充来源" if safe_sources else "未找到可展示来源"
    note = (
        f"已使用补证搜索策略，优先检索官方/一手/发布说明来源；本轮补充 {len(safe_sources)} 个可展示来源，重要结论仍建议打开来源核对。"
        if safe_sources
        else "已使用补证搜索策略，但本轮未找到可展示 HTTP(S) 来源；不要把这次回答直接当作已核验结论。"
    )
    return normalize_evidence_repair_run(
        {
            "schema_version": EVIDENCE_REPAIR_RUN_SCHEMA_VERSION,
            "status": status,
            "label": label,
            "source_count": len(safe_sources),
            "action": request.get("action") or "repair_evidence",
            "started_from": request.get("started_from") or request.get("startedFrom") or "answer_evidence_repair",
            "prompt": request.get("prompt") or query,
            "note": note,
        },
        sources=safe_sources,
        query=query,
    )


def normalize_evidence_repair_run(value: Any, sources: Any = None, *, query: Any = "") -> dict[str, Any]:
    """Return a safe public evidence-repair run envelope."""

    if not isinstance(value, dict) or not value:
        return {}
    safe_sources = sanitize_report_sources(sources, limit=12)
    raw_source_count = value.get("source_count") if "source_count" in value else value.get("sourceCount")
    source_count = _bounded_int(raw_source_count, default=len(safe_sources))
    status = _choice(value.get("status"), {"completed", "no_sources", "needs_review", "skipped"}, "completed" if source_count else "no_sources")
    label = _compact(value.get("label"), limit=48) or ("已补充来源" if status == "completed" else "补证结果待复核")
    prompt = _compact(value.get("prompt") or query, limit=EVIDENCE_REPAIR_PROMPT_LIMIT)
    action = _compact(value.get("action"), limit=80) or "repair_evidence"
    started_from = _compact(value.get("started_from") or value.get("startedFrom"), limit=64) or "answer_evidence_repair"
    note = _compact(value.get("note"), limit=180)
    if not note:
        note = (
            f"已使用补证搜索策略，优先检索官方/一手/发布说明来源；本轮补充 {source_count} 个可展示来源，重要结论仍建议打开来源核对。"
            if status == "completed"
            else "已使用补证搜索策略，但本轮补证未完成可直接引用的来源闭环，请继续补充来源或人工复核。"
        )
    return {
        "schema_version": EVIDENCE_REPAIR_RUN_SCHEMA_VERSION,
        "status": status,
        "label": label,
        "source_count": max(0, source_count),
        "action": action,
        "started_from": started_from,
        "prompt": prompt,
        "note": note,
    }


def _choice(value: Any, choices: set[str], default: str) -> str:
    """Return value when it is one of the public choices."""

    text = str(value or "").strip()
    return text if text in choices else default


def _bounded_int(value: Any, *, default: int = 0) -> int:
    """Return a non-negative bounded integer."""

    try:
        return max(0, min(999, int(value)))
    except (TypeError, ValueError):
        return max(0, min(999, int(default)))


def _compact(value: Any, *, limit: int) -> str:
    """Return compact redacted public text."""

    text = " ".join(redact_text(str(value or "")).split())
    return text[:limit]
