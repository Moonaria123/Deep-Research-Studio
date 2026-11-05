from __future__ import annotations

from typing import Any


PLAN_PREVIEW_SCHEMA_VERSION = 1


def build_research_plan_preview(
    plan: Any,
    *,
    route: Any = None,
    status: str = "pending_confirmation",
    requires_confirmation: bool = True,
) -> dict[str, Any]:
    """Return a public, secret-free research plan preview."""

    normalized = normalize_research_plan_preview(
        {
            "schema_version": PLAN_PREVIEW_SCHEMA_VERSION,
            "status": status,
            "requires_confirmation": requires_confirmation,
            "confirmation_label": "开始研究",
            "edit_label": "修改问题",
            "route_level": getattr(route, "level", "") if route is not None else "",
            "route_label": getattr(route, "label", "") if route is not None else "",
            "user_intent_summary": _plan_text(plan, "user_intent_summary"),
            "research_questions": _string_list(_plan_value(plan, "research_questions"), limit=6),
            "search_dimensions": _string_list(_plan_value(plan, "search_dimensions"), limit=6),
            "search_plan": _search_plan_items(_plan_value(plan, "search_plan"), limit=8),
            "source_policy": "确认后才会开始搜索和生成报告。",
            "safety_note": "计划确认前不会执行搜索、抓取网页或生成事实正文。",
        }
    )
    if not normalized.get("research_questions"):
        summary = normalized.get("user_intent_summary") or "本次研究问题"
        normalized["research_questions"] = [summary]
    return normalized


def normalize_research_plan_preview(value: Any) -> dict[str, Any]:
    """Normalize public research plan preview metadata."""

    if not isinstance(value, dict):
        value = {}
    has_preview = bool(value)
    status = _compact(value.get("status"), limit=40) or ("pending_confirmation" if has_preview else "not_required")
    preview = {
        "schema_version": PLAN_PREVIEW_SCHEMA_VERSION,
        "status": status,
        "requires_confirmation": bool(value.get("requires_confirmation")),
        "confirmation_label": _compact(value.get("confirmation_label"), limit=24) or "开始研究",
        "edit_label": _compact(value.get("edit_label"), limit=24) or "修改问题",
        "route_level": _compact(value.get("route_level"), limit=16),
        "route_label": _compact(value.get("route_label"), limit=32),
        "user_intent_summary": _compact(value.get("user_intent_summary"), limit=240),
        "research_questions": _string_list(value.get("research_questions"), limit=8),
        "search_dimensions": _string_list(value.get("search_dimensions"), limit=8),
        "search_plan": _search_plan_items(value.get("search_plan"), limit=12),
        "source_policy": _compact(value.get("source_policy"), limit=160) or "确认后才会开始搜索和生成报告。",
        "safety_note": _compact(value.get("safety_note"), limit=160) or "计划确认前不会执行搜索或章节生成。",
    }
    return preview


def confirmed_research_plan(value: Any) -> dict[str, Any]:
    """Return a normalized confirmed plan from input payload."""

    if not isinstance(value, dict):
        return {}
    plan = {
        "research_questions": _string_list(value.get("research_questions"), limit=8),
        "search_dimensions": _string_list(value.get("search_dimensions"), limit=8),
        "search_plan": _search_plan_items(value.get("search_plan"), limit=12),
        "user_intent_summary": _compact(value.get("user_intent_summary"), limit=240),
        "edited_by_user": bool(value.get("edited_by_user")),
        "edit_source": _compact(value.get("edit_source"), limit=40),
    }
    if not plan["search_plan"]:
        return {}
    if not plan["research_questions"] and plan["user_intent_summary"]:
        plan["research_questions"] = [plan["user_intent_summary"]]
    if not plan["search_dimensions"]:
        plan["search_dimensions"] = [item["dimension"] for item in plan["search_plan"] if item.get("dimension")]
    return plan


def _plan_value(plan: Any, key: str) -> Any:
    if isinstance(plan, dict):
        return plan.get(key)
    return None


def _plan_text(plan: Any, key: str) -> str:
    return _compact(_plan_value(plan, key), limit=240)


def _string_list(value: Any, *, limit: int) -> list[str]:
    if not isinstance(value, list):
        return []
    items: list[str] = []
    for item in value:
        text = _compact(item, limit=180)
        if not text:
            continue
        items.append(text)
        if len(items) >= limit:
            break
    return items


def _search_plan_items(value: Any, *, limit: int) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []
    items: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        query = _compact(item.get("query"), limit=220)
        dimension = _compact(item.get("dimension"), limit=80)
        if not query:
            continue
        items.append({"query": query, "dimension": dimension})
        if len(items) >= limit:
            break
    return items


def _compact(value: Any, *, limit: int) -> str:
    text = " ".join(str(value or "").replace("\r", " ").replace("\n", " ").split())
    return text[:limit].strip()
