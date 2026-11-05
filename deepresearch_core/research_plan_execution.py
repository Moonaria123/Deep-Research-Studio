from __future__ import annotations

from typing import Any

from deepresearch_core.report_sources import sanitize_report_sources


RESEARCH_PLAN_EXECUTION_SCHEMA_VERSION = 1


def build_research_plan_execution(
    plan: Any,
    sources: Any,
    *,
    status: str = "completed",
) -> dict[str, Any]:
    """Return public metadata showing how a confirmed research plan executed."""

    items = _plan_items(plan)
    safe_sources = sanitize_report_sources(sources)
    raw_sources = sources if isinstance(sources, list) else []
    source_indexes_by_branch = _source_indexes_by_branch(raw_sources, safe_sources)
    execution_items = []
    for index, item in enumerate(items, start=1):
        source_indexes = source_indexes_by_branch.get(index, [])
        source_count = len(source_indexes)
        item_status = "sourced" if source_count else "no_sources"
        execution_items.append(
            {
                "index": index,
                "query": item["query"],
                "dimension": item["dimension"],
                "status": item_status,
                "label": f"找到 {source_count} 个来源" if source_count else "暂无来源",
                "source_count": source_count,
                "source_indexes": source_indexes,
                "note": "已回收可展示来源。" if source_count else "该搜索方向暂未回收可展示 HTTP(S) 来源。",
            }
        )
    source_count = len(safe_sources)
    query_count = len(execution_items)
    normalized_status = "completed" if status == "completed" and source_count else status
    if not source_count and normalized_status == "completed":
        normalized_status = "blocked"
    label = {
        "completed": "已按计划检索",
        "blocked": "计划执行受阻",
        "partial": "计划部分完成",
        "generated": "已生成计划执行结果",
    }.get(normalized_status, "计划执行情况")
    summary = (
        f"已执行 {query_count} 个搜索方向，找到 {source_count} 个可展示来源。"
        if source_count
        else f"已执行 {query_count} 个搜索方向，但暂未找到可展示来源。"
    )
    return normalize_research_plan_execution(
        {
            "schema_version": RESEARCH_PLAN_EXECUTION_SCHEMA_VERSION,
            "status": normalized_status,
            "label": label,
            "summary": summary,
            "query_count": query_count,
            "source_count": source_count,
            "edited_by_user": bool(_plan_value(plan, "edited_by_user")),
            "items": execution_items,
        }
    )


def normalize_research_plan_execution(value: Any, *, sources: Any = None) -> dict[str, Any]:
    """Return safe, stable research plan execution metadata."""

    if not isinstance(value, dict) or not value:
        source_count = len(sanitize_report_sources(sources or []))
        return {
            "schema_version": RESEARCH_PLAN_EXECUTION_SCHEMA_VERSION,
            "status": "not_started",
            "label": "尚未开始执行计划",
            "summary": "计划确认前不会执行搜索、抓取或报告生成。",
            "query_count": 0,
            "source_count": source_count,
            "edited_by_user": False,
            "items": [],
        }
    items = _execution_items(value.get("items"), limit=12)
    source_count = _non_negative_int(value.get("source_count"))
    if source_count == 0 and sources is not None:
        source_count = len(sanitize_report_sources(sources))
    query_count = _non_negative_int(value.get("query_count")) or len(items)
    status = _compact(value.get("status"), limit=40) or "completed"
    return {
        "schema_version": RESEARCH_PLAN_EXECUTION_SCHEMA_VERSION,
        "status": status,
        "label": _compact(value.get("label"), limit=48) or "计划执行情况",
        "summary": _compact(value.get("summary"), limit=180) or "已按确认计划执行搜索。",
        "query_count": query_count,
        "source_count": source_count,
        "edited_by_user": bool(value.get("edited_by_user")),
        "items": items,
    }


def _plan_items(plan: Any) -> list[dict[str, str]]:
    search_plan = _plan_value(plan, "search_plan")
    if not isinstance(search_plan, list):
        return []
    items = []
    for item in search_plan[:12]:
        if not isinstance(item, dict):
            continue
        query = _compact(item.get("query"), limit=220)
        if not query:
            continue
        items.append({"query": query, "dimension": _compact(item.get("dimension"), limit=80)})
    return items


def _execution_items(value: Any, *, limit: int) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    items = []
    for item in value:
        if not isinstance(item, dict):
            continue
        query = _compact(item.get("query"), limit=220)
        if not query:
            continue
        source_indexes = _source_indexes(item.get("source_indexes") or item.get("sourceIndexes"))
        source_count = _non_negative_int(item.get("source_count") or item.get("sourceCount")) or len(source_indexes)
        items.append(
            {
                "index": _non_negative_int(item.get("index")) or len(items) + 1,
                "query": query,
                "dimension": _compact(item.get("dimension"), limit=80),
                "status": _compact(item.get("status"), limit=32) or ("sourced" if source_count else "no_sources"),
                "label": _compact(item.get("label"), limit=64)
                or (f"找到 {source_count} 个来源" if source_count else "暂无来源"),
                "source_count": source_count,
                "source_indexes": source_indexes,
                "note": _compact(item.get("note"), limit=160),
            }
        )
        if len(items) >= limit:
            break
    return items


def _source_indexes_by_branch(raw_sources: list[Any], safe_sources: list[dict[str, Any]]) -> dict[int, list[int]]:
    url_to_index = {source.get("url"): index for index, source in enumerate(safe_sources, start=1)}
    result: dict[int, list[int]] = {}
    for raw in raw_sources:
        if not isinstance(raw, dict):
            continue
        branch = _non_negative_int(raw.get("branch"))
        safe_index = url_to_index.get(raw.get("url"))
        if not branch or not safe_index:
            continue
        result.setdefault(branch, [])
        if safe_index not in result[branch]:
            result[branch].append(safe_index)
    return result


def _source_indexes(value: Any) -> list[int]:
    if not isinstance(value, list):
        return []
    indexes = []
    for item in value:
        index = _non_negative_int(item)
        if index and index not in indexes:
            indexes.append(index)
    return indexes[:12]


def _plan_value(plan: Any, key: str) -> Any:
    return plan.get(key) if isinstance(plan, dict) else None


def _non_negative_int(value: Any) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _compact(value: Any, *, limit: int) -> str:
    text = " ".join(str(value or "").replace("\r", " ").replace("\n", " ").split())
    return text[:limit].strip()
