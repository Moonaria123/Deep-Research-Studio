from __future__ import annotations

from typing import Any

from deepresearch_core.report_sources import sanitize_report_sources


JsonDict = dict[str, Any]
SEARCH_RECOVERY_SCHEMA_VERSION = 1


def build_search_recovery(sources: Any) -> JsonDict:
    """Return public metadata about assisted and source-first recovery searches."""

    raw_sources = sources if isinstance(sources, list) else list(sources or [])
    safe_sources = sanitize_report_sources(raw_sources)
    diagnostics = _search_diagnostics(raw_sources)
    attempted_count = len(diagnostics)
    failed_count = sum(1 for item in diagnostics if item.get("status") != "ok")
    timed_out_count = sum(1 for item in diagnostics if item.get("error_type") == "timeout")
    all_failed = bool(attempted_count and failed_count >= attempted_count and not safe_sources)
    search_health = "all_failed" if all_failed else "partial_failure" if failed_count else "ok"
    assisted_labels = _unique_labels(_assist_label(source) for source in raw_sources)
    strategy_labels = _unique_labels(_strategy_label(source) for source in raw_sources)
    used_assisted = bool(assisted_labels)
    used_source_seeking = bool(strategy_labels)
    status = "used" if used_assisted or used_source_seeking else "not_used"
    if all_failed and timed_out_count:
        label = "检索超时"
        summary = "本轮搜索查询均未在时限内返回；没有用模型记忆补全事实结论。"
    elif all_failed:
        label = "检索失败"
        summary = "本轮搜索查询均未返回可用结果；没有用模型记忆补全事实结论。"
    elif failed_count:
        label = "部分检索失败"
        summary = "本轮部分搜索查询失败，但仍保留已返回的可展示来源。"
    elif used_source_seeking:
        label = "已启用备用检索"
        summary = "本轮自动追加了一手来源导向查询，以降低首轮搜索超时或召回过薄造成的失败。"
    elif used_assisted:
        label = "已启用辅助检索"
        summary = "本轮自动追加了实体或产品名规范查询，以降低误写或拆分写法造成的漏召回。"
    else:
        label = "未触发备用检索"
        summary = "本轮未从返回来源中识别到辅助检索或备用检索命中。"
    return normalize_search_recovery(
        {
            "schema_version": SEARCH_RECOVERY_SCHEMA_VERSION,
            "status": status,
            "label": label,
            "used_assisted_query": used_assisted,
            "used_source_seeking": used_source_seeking,
            "source_count": len(safe_sources),
            "search_health": search_health,
            "attempted_query_count": attempted_count,
            "failed_query_count": failed_count,
            "timed_out_query_count": timed_out_count,
            "all_queries_failed": all_failed,
            "assisted_labels": assisted_labels,
            "source_seeking_labels": strategy_labels,
            "summary": summary,
            "note": "该字段只说明检索策略是否被使用，不代表来源事实已经被最终证明。",
        }
    )


def normalize_search_recovery(value: Any, *, sources: Any = None) -> JsonDict:
    """Return safe, stable search recovery metadata."""

    if not isinstance(value, dict) or not value:
        source_count = len(sanitize_report_sources(sources or []))
        return {
            "schema_version": SEARCH_RECOVERY_SCHEMA_VERSION,
            "status": "not_used",
            "label": "未触发备用检索",
            "used_assisted_query": False,
            "used_source_seeking": False,
            "source_count": source_count,
            "search_health": "ok",
            "attempted_query_count": 0,
            "failed_query_count": 0,
            "timed_out_query_count": 0,
            "all_queries_failed": False,
            "assisted_labels": [],
            "source_seeking_labels": [],
            "summary": "本轮未从返回来源中识别到辅助检索或备用检索命中。",
            "note": "该字段只说明检索策略是否被使用，不代表来源事实已经被最终证明。",
        }
    return {
        "schema_version": SEARCH_RECOVERY_SCHEMA_VERSION,
        "status": _choice(value.get("status"), {"used", "not_used"}, "not_used"),
        "label": _compact(value.get("label"), limit=48) or "检索恢复状态",
        "used_assisted_query": bool(value.get("used_assisted_query")),
        "used_source_seeking": bool(value.get("used_source_seeking")),
        "source_count": _non_negative_int(value.get("source_count")),
        "search_health": _choice(value.get("search_health"), {"ok", "partial_failure", "all_failed"}, "ok"),
        "attempted_query_count": _non_negative_int(value.get("attempted_query_count")),
        "failed_query_count": _non_negative_int(value.get("failed_query_count")),
        "timed_out_query_count": _non_negative_int(value.get("timed_out_query_count")),
        "all_queries_failed": bool(value.get("all_queries_failed")),
        "assisted_labels": _label_list(value.get("assisted_labels")),
        "source_seeking_labels": _label_list(value.get("source_seeking_labels")),
        "summary": _compact(value.get("summary"), limit=180) or "检索恢复状态已记录。",
        "note": _compact(value.get("note"), limit=180) or "该字段只说明检索策略是否被使用，不代表来源事实已经被最终证明。",
    }


def _search_diagnostics(raw_sources: Any) -> list[JsonDict]:
    diagnostics: list[JsonDict] = []
    if not isinstance(raw_sources, list):
        return diagnostics
    for source in raw_sources:
        if not isinstance(source, dict):
            continue
        if source.get("_search_attempt"):
            diagnostics.append(source)
        nested = source.get("search_diagnostics")
        if isinstance(nested, list):
            diagnostics.extend(item for item in nested if isinstance(item, dict))
    return diagnostics


def _assist_label(source: Any) -> str:
    if not isinstance(source, dict):
        return ""
    assist = source.get("query_assist")
    if not isinstance(assist, dict):
        return ""
    return _compact(assist.get("label") or assist.get("kind") or "辅助检索", limit=48)


def _strategy_label(source: Any) -> str:
    if not isinstance(source, dict):
        return ""
    strategy = source.get("query_strategy")
    if not isinstance(strategy, dict):
        return ""
    return _compact(strategy.get("label") or strategy.get("kind") or "备用检索", limit=48)


def _unique_labels(values: Any) -> list[str]:
    labels: list[str] = []
    for value in values:
        label = _compact(value, limit=48)
        if label and label not in labels:
            labels.append(label)
        if len(labels) >= 6:
            break
    return labels


def _label_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return _unique_labels(value)


def _choice(value: Any, allowed: set[str], fallback: str) -> str:
    text = _compact(value, limit=32)
    return text if text in allowed else fallback


def _compact(value: Any, *, limit: int) -> str:
    text = " ".join(str(value or "").replace("\r", " ").replace("\n", " ").split()).strip()
    return text[:limit]


def _non_negative_int(value: Any) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0
